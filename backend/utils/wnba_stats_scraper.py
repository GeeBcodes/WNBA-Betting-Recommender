import sys
import os
import logging
import pandas as pd
import numpy as np # Import numpy for np.nan
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError
from sqlalchemy import or_
from datetime import datetime, date
import polars as pl
from typing import Optional, List, Dict, Any

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from backend.db.session import SessionLocal
from backend.db import models as db_models
import sportsdataverse as sdv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper function to get or create Team ---
def get_or_create_team(db: Session, team_api_id_str: Optional[str], team_name_str: Optional[str]) -> Optional[db_models.Team]:
    if not team_api_id_str or pd.isna(team_api_id_str) or str(team_api_id_str).lower() == 'nan':
        logger.warning(f"Missing or invalid team_api_id: {team_api_id_str}. Cannot get/create team based on API ID.")
        if team_name_str and not pd.isna(team_name_str) and str(team_name_str).lower() != 'nan':
            logger.info(f"Attempting to find team by name: {team_name_str}")
            team = db.query(db_models.Team).filter(db_models.Team.team_name == team_name_str).first()
            if team:
                logger.info(f"Found team by name: {team_name_str} with ID {team.id}")
                return team
            else:
                logger.warning(f"Team with name '{team_name_str}' not found. Cannot create without API ID.")
                return None
        return None

    team_api_id = str(team_api_id_str)
    team = db.query(db_models.Team).filter(db_models.Team.api_team_id == team_api_id).first()
    if team:
        return team
    else:
        if not team_name_str or pd.isna(team_name_str) or str(team_name_str).lower() == 'nan':
            logger.warning(f"Team with api_team_id {team_api_id} not found, and no valid team_name provided to create it. Team name was: {team_name_str}")
            return None
        
        logger.info(f"Team with api_team_id {team_api_id} not found. Creating new team: {team_name_str}.")
        try:
            new_team = db_models.Team(api_team_id=team_api_id, team_name=str(team_name_str))
            db.add(new_team)
            db.commit()
            db.refresh(new_team)
            logger.info(f"Successfully created team {new_team.team_name} with api_team_id {new_team.api_team_id}")
            return new_team
        except IntegrityError:
            db.rollback()
            logger.error(f"IntegrityError creating team with api_team_id {team_api_id} and name {team_name_str}. Querying again.")
            # Query again in case of race condition
            return db.query(db_models.Team).filter(db_models.Team.api_team_id == team_api_id).first()
        except Exception as e:
            db.rollback()
            logger.error(f"Unexpected error creating team {team_name_str} (API ID: {team_api_id}): {e}")
            return None

# --- Transform Player Stats Data ---
def transform_player_stats_data(
    df_pl: pl.DataFrame, 
    game_datetime_obj: datetime,
    game_id_str: str,
    game_home_team_db: Optional[db_models.Team], 
    game_away_team_db: Optional[db_models.Team],
    game_season_str: str, 
    db: Session
) -> List[db_models.PlayerStat]:
    
    records = []
    
    # Define the exact stat column names as they should be in the Polars DataFrame 
    # (after renaming in load_wnba_stats) AND as they are in the PlayerStat model.
    STAT_COLUMNS_TO_STORE = [
        'minutes_played', 'points', 'rebounds', 'assists', 'steals', 'blocks', 
        'turnovers', 
        'fouls',  # This is the target model field, mapped from 'pf' in raw data
        'fgm', 'fga', 'fg_pct', 
        'three_pm', 'three_pa', 'three_p_pct', 
        'ftm', 'fta', 'ft_pct'
        # ENSURE 'personal_fouls' or other raw variations are NOT in this list.
    ]
    
    # These are columns expected in the Polars DF for player/team identification and game context
    REQUIRED_CONTEXT_COLS_FROM_DF_PL = [
        'player_api_id', 'player_name_from_boxscore', 
        'player_team_api_id_for_game', 'player_team_name_for_game',
        # 'home_away' # This column tells if player_team was home or away for THIS game
    ]
    
    # All columns that must be present in the Polars DataFrame rows
    ALL_REQUIRED_COLS_IN_ROW = REQUIRED_CONTEXT_COLS_FROM_DF_PL + STAT_COLUMNS_TO_STORE

    if not df_pl.is_empty():
        # Check if all required columns exist in the Polars DataFrame's schema at least once
        # This is a one-time check on the DataFrame schema, not per row.
        schema_cols = df_pl.columns
        missing_schema_cols = [col for col in ALL_REQUIRED_COLS_IN_ROW if col not in schema_cols]
        if 'home_away' not in schema_cols: # home_away is critical for determining is_home_team
            missing_schema_cols.append('home_away')
            
        if missing_schema_cols:
            logger.error(f"transform_player_stats_data: DataFrame is missing required columns: {missing_schema_cols}. Cannot process stats for game {game_id_str}.")
            return []

    for pl_row in df_pl.iter_rows(named=True): # pl_row is a dict
        player_api_id_str = str(pl_row.get('player_api_id'))
        player_name_str = str(pl_row.get('player_name_from_boxscore'))
        player_team_api_id_for_game_str = str(pl_row.get('player_team_api_id_for_game'))
        # player_team_name_for_game_str = str(pl_row.get('player_team_name_for_game')) # Not directly used below but good for context

        if not player_api_id_str or player_api_id_str.lower() == 'nan':
            logger.warning(f"Missing player_api_id for player '{player_name_str}' in game {game_id_str}. Skipping record.")
            continue

        # Determine if the player's team for this game was the home team
        # The 'home_away' column from sportsdataverse boxscore indicates 'HOME' or 'AWAY' for the player's team in that game.
        player_home_away_status = str(pl_row.get('home_away', '')).upper()
        is_home_team_for_player_stat: Optional[bool] = None
        if player_home_away_status == 'HOME':
            is_home_team_for_player_stat = True
        elif player_home_away_status == 'AWAY':
            is_home_team_for_player_stat = False
        else:
            logger.warning(f"Could not determine home/away status for player {player_name_str} (API ID: {player_api_id_str}) in game {game_id_str}. 'home_away' was '{player_home_away_status}'. 'is_home_team' will be None.")
            # Depending on model definition, we might need to skip or handle None for is_home_team
            # For now, let it be None and see if PlayerStat model handles it (it requires non-nullable)
            # The PlayerStat model has nullable=False, default=False for is_home_team.
            # So, if we can't determine, we MUST NOT proceed with None. Default to False or skip.
            # For now, let's default to False if unknown, as the DB column requires a value.
            is_home_team_for_player_stat = False # Defaulting if unknown

        # Get or create player's current team (for this game)
        # This team is just for context if needed, player.team_id is their primary team
        # player_current_game_team_db = get_or_create_team(db, player_team_api_id_for_game_str, player_team_name_for_game_str)
        # if not player_current_game_team_db:
        #     logger.warning(f"Could not get/create team (API ID: {player_team_api_id_for_game_str}, Name: {player_team_name_for_game_str}) for player {player_name_str} in game {game_id_str}. Skipping record.")
        #     continue

        # Get or create Player
        player = db.query(db_models.Player).filter(db_models.Player.api_player_id == player_api_id_str).first()
        if not player:
            logger.info(f"Player with api_player_id {player_api_id_str} ({player_name_str}) not found. Creating new player.")
            # For the primary team_id of the player, we should use the team they are currently associated with in the box score.
            # This assumes player_team_api_id_for_game is their current team.
            # This is a simplification; player's team might change. For now, link to their team in this game.
            primary_team_for_player = get_or_create_team(db, player_team_api_id_for_game_str, str(pl_row.get('player_team_name_for_game')))

            player = db_models.Player(
                api_player_id=player_api_id_str, 
                player_name=player_name_str,
                team_id=primary_team_for_player.id if primary_team_for_player else None
            )
            try:
                db.add(player)
                db.commit()
                db.refresh(player)
                logger.info(f"Successfully created player {player.id} with api_player_id {player_api_id_str}")
            except IntegrityError as e:
                db.rollback()
                logger.warning(f"IntegrityError creating player {player_api_id_str} ({player_name_str}): {e}. Querying again.")
                player = db.query(db_models.Player).filter(db_models.Player.api_player_id == player_api_id_str).first()
                if not player:
                    logger.error(f"Failed to create or find player {player_api_id_str} after IntegrityError. Skipping record for this player in this game.")
                    continue
            except Exception as e:
                db.rollback()
                logger.error(f"Error creating new player {player_api_id_str} ({player_name_str}): {e}")
                continue
        
        # Prepare the dictionary of stat values
        stat_values: Dict[str, Any] = {}
        for stat_key in STAT_COLUMNS_TO_STORE:
            raw_value = pl_row.get(stat_key)
            
            if pd.isna(raw_value) or (isinstance(raw_value, str) and raw_value.lower() == 'nan'):
                stat_values[stat_key] = None
            elif stat_key in ['fg_pct', 'three_p_pct', 'ft_pct']: # Percentages
                try:
                    stat_values[stat_key] = float(raw_value) if raw_value is not None else None
                except (ValueError, TypeError):
                    # logger.warning(f"Could not convert percentage stat '{stat_key}' ('{raw_value}') to float for player {player_name_str}. Setting to None.")
                    stat_values[stat_key] = None
            elif stat_key in ['minutes_played', 'points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'fouls', 
                              'fgm', 'fga', 'three_pm', 'three_pa', 'ftm', 'fta']: # Counts/integers/floats
                try:
                    # Try converting to float first, then int if it's a whole number, to handle cases like "10.0" or "10"
                    val = float(raw_value)
                    stat_values[stat_key] = int(val) if val.is_integer() else val
                except (ValueError, TypeError):
                    # logger.warning(f"Could not convert numeric stat '{stat_key}' ('{raw_value}') to number for player {player_name_str}. Setting to None.")
                    stat_values[stat_key] = None
            else:
                stat_values[stat_key] = raw_value # Should not happen if STAT_COLUMNS_TO_STORE is well-defined

        # Create Game record if it doesn't exist
        game = db.query(db_models.Game).filter(db_models.Game.external_id == game_id_str).first()
        if not game:
            if not game_home_team_db or not game_away_team_db:
                logger.warning(f"Home ({game_home_team_db.team_name if game_home_team_db else 'None'}) or Away ({game_away_team_db.team_name if game_away_team_db else 'None'}) team DB object missing for game {game_id_str}. Cannot create game record.")
                continue 

            logger.info(f"Game with external_id {game_id_str} not found. Creating new game.")
            game = db_models.Game(
                external_id=game_id_str,
                game_datetime=game_datetime_obj,
                home_team_id=game_home_team_db.id,
                away_team_id=game_away_team_db.id,
                season=game_season_str 
            )
            try:
                db.add(game)
                db.commit()
                db.refresh(game)
            except IntegrityError:
                db.rollback()
                logger.warning(f"Integrity error creating game {game_id_str}. Querying again.")
                game = db.query(db_models.Game).filter(db_models.Game.external_id == game_id_str).first()
                if not game:
                    logger.error(f"Failed to create or find game {game_id_str} after IntegrityError. Skipping stats for this player in this game.")
                    continue # Skip this player_stat if game cannot be established
            except Exception as e:
                db.rollback()
                logger.error(f"Error creating new game {game_id_str}: {e}")
                continue 
        
        # Check if PlayerStat record already exists
        existing_stat = db.query(db_models.PlayerStat).filter_by(player_id=player.id, game_id=game.id).first()
        if existing_stat:
            # logger.info(f"PlayerStat for player {player.id} game {game.id} already exists. Updating.")
            for key, value in stat_values.items():
                setattr(existing_stat, key, value)
            existing_stat.is_home_team = is_home_team_for_player_stat 
        else:
            # logger.info(f"Creating new PlayerStat for player {player.id} game {game.id}.")
            player_stat = db_models.PlayerStat(
                player_id=player.id,
                game_id=game.id,
                is_home_team=is_home_team_for_player_stat, # This is now correctly populated
                **stat_values # CRITICAL: stat_values should ONLY contain keys matching PlayerStat model fields
            )
            records.append(player_stat)

    return records


def load_wnba_stats(year: int, db: Session):
    logger.info(f"Fetching WNBA schedule for year: {year}.")
    raw_schedule_df_sdv = sdv.wnba.load_wnba_schedule(seasons=[year])
    
    if raw_schedule_df_sdv is None or (isinstance(raw_schedule_df_sdv, pl.DataFrame) and raw_schedule_df_sdv.is_empty()) or \
       (isinstance(raw_schedule_df_sdv, pd.DataFrame) and raw_schedule_df_sdv.empty):
        logger.warning(f"No schedule data found for {year} from sportsdataverse. Cannot proceed.")
        return 0

    schedule_data_pd = raw_schedule_df_sdv.to_pandas() if isinstance(raw_schedule_df_sdv, pl.DataFrame) else raw_schedule_df_sdv
    
    if schedule_data_pd is None or schedule_data_pd.empty:
        logger.warning(f"Schedule data became empty after potential conversion for {year}. Cannot proceed.")
        return 0

    logger.info(f"Raw schedule_data_pd columns from sportsdataverse: {schedule_data_pd.columns.tolist()}")
    
    schedule_rename_map_config = {
        'game_id_from_schedule': ['id', 'game_id'],
        'game_datetime_from_schedule': ['date', 'game_date', 'datetime'],
        'game_season_from_schedule': ['season', 'year'],
        'game_home_team_api_id_from_schedule': ['home_id', 'homeID', 'home_team_id'],
        'game_away_team_api_id_from_schedule': ['away_id', 'awayID', 'away_team_id'],
        'game_home_team_name_from_schedule': ['home_name', 'homeTeamName', 'home_display_name', 'home_short_display_name'],
        'game_away_team_name_from_schedule': ['away_name', 'awayTeamName', 'away_display_name', 'away_short_display_name']
    }
    
    actual_schedule_rename_map = {}
    all_schedule_cols_found = True
    for target_col, source_options in schedule_rename_map_config.items():
        found_source = None
        for source_col in source_options:
            if source_col in schedule_data_pd.columns:
                found_source = source_col
                break
        if found_source:
            actual_schedule_rename_map[found_source] = target_col
        else:
            logger.error(f"Critical schedule column for '{target_col}' not found (tried: {source_options}). Available: {schedule_data_pd.columns.tolist()}")
            all_schedule_cols_found = False
            
    if not all_schedule_cols_found:
        logger.error("One or more critical columns missing from schedule data. Aborting for this year.")
        return 0

    schedule_df_for_merge = schedule_data_pd.rename(columns=actual_schedule_rename_map)
    
    # Select only the columns we need after renaming
    final_schedule_cols_needed = list(schedule_rename_map_config.keys())
    schedule_df_for_merge = schedule_df_for_merge[final_schedule_cols_needed]

    try:
        schedule_df_for_merge['game_datetime_parsed'] = pd.to_datetime(schedule_df_for_merge['game_datetime_from_schedule'])
        schedule_df_for_merge['game_date_from_schedule'] = schedule_df_for_merge['game_datetime_parsed'].dt.date
    except Exception as e:
        logger.error(f"Error converting 'game_datetime_from_schedule' to datetime objects: {e}. Some games might be skipped.")
        schedule_df_for_merge['game_datetime_parsed'] = pd.NaT
        schedule_df_for_merge['game_date_from_schedule'] = pd.NaT

    schedule_df_for_merge['game_id_from_schedule'] = schedule_df_for_merge['game_id_from_schedule'].astype(str)
    schedule_df_for_merge['game_home_team_api_id_from_schedule'] = schedule_df_for_merge['game_home_team_api_id_from_schedule'].astype(str)
    schedule_df_for_merge['game_away_team_api_id_from_schedule'] = schedule_df_for_merge['game_away_team_api_id_from_schedule'].astype(str)
    schedule_df_for_merge['game_season_from_schedule'] = schedule_df_for_merge['game_season_from_schedule'].astype(str)


    cols_to_log_schedule = [
        'game_id_from_schedule', 'game_date_from_schedule', 'game_season_from_schedule',
        'game_home_team_api_id_from_schedule', 'game_away_team_api_id_from_schedule',
        'game_home_team_name_from_schedule', 'game_away_team_name_from_schedule'
    ]
    logger.info(f"Sample of schedule_df_for_merge (first 3 rows, relevant columns for merge):\\n{schedule_df_for_merge[cols_to_log_schedule].head(3).to_string(index=False)}")


    # --- Player Stats Processing ---
    logger.info(f"Fetching WNBA player boxscores for season: {year}.")
    player_stats_df_sdv_raw = sdv.wnba.load_wnba_player_boxscore(seasons=[year])
    
    if player_stats_df_sdv_raw is None or \
       (isinstance(player_stats_df_sdv_raw, pl.DataFrame) and player_stats_df_sdv_raw.is_empty()) or \
       (isinstance(player_stats_df_sdv_raw, pd.DataFrame) and player_stats_df_sdv_raw.empty):
        logger.warning(f"No player boxscores found for season {year}.")
        return 0
    
    player_stats_df_sdv = player_stats_df_sdv_raw.to_pandas() if isinstance(player_stats_df_sdv_raw, pl.DataFrame) else player_stats_df_sdv_raw

    if player_stats_df_sdv is None or player_stats_df_sdv.empty:
        logger.warning(f"Player boxscores became empty after potential conversion for {year}.")
        return 0

    player_stats_df_sdv['game_id'] = player_stats_df_sdv['game_id'].astype(str)
    
    player_rename_map = {
        # Context / ID columns (these seem to be consistent or already correct)
        'athlete_id': 'player_api_id',
        'athlete_display_name': 'player_name_from_boxscore',
        'team_id': 'player_team_api_id_for_game',
        'team_name': 'player_team_name_for_game', 
        'home_away': 'home_away', 

        # Statistical columns - UPDATED based on logs
        'minutes': 'minutes_played',
        'points': 'points',
        'rebounds': 'rebounds',
        'assists': 'assists',
        'steals': 'steals',
        'blocks': 'blocks',
        'turnovers': 'turnovers',
        'fouls': 'fouls',
        'field_goals_made': 'fgm',
        'field_goals_attempted': 'fga',
        'three_point_field_goals_made': 'three_pm',
        'three_point_field_goals_attempted': 'three_pa',
        'free_throws_made': 'ftm',
        'free_throws_attempted': 'fta'
        # 'plus_minus': 'plus_minus' # plus_minus is already correctly named if present
    }

    # Select and rename columns
    cols_to_rename = {k: v for k, v in player_rename_map.items() if k in player_stats_df_sdv.columns}
    missing_raw_cols = [k for k in player_rename_map.keys() if k not in player_stats_df_sdv.columns]
    if missing_raw_cols:
        logger.warning(f"Player stats raw data missing some expected columns (will be NaN/None if not renamed): {missing_raw_cols}. Available: {player_stats_df_sdv.columns.tolist()}")

    player_stats_df_sdv = player_stats_df_sdv.rename(columns=cols_to_rename)

    # Calculate percentage stats
    player_stats_df_sdv['fg_pct'] = (player_stats_df_sdv['fgm'] / player_stats_df_sdv['fga']).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    player_stats_df_sdv['three_p_pct'] = (player_stats_df_sdv['three_pm'] / player_stats_df_sdv['three_pa']).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    player_stats_df_sdv['ft_pct'] = (player_stats_df_sdv['ftm'] / player_stats_df_sdv['fta']).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Columns needed for merge and transformation
    player_cols_for_merge_and_transform = [
        'game_id', 'player_api_id', 'player_name_from_boxscore', 
        'player_team_api_id_for_game', 'player_team_name_for_game', 'home_away',
        'minutes_played', 'points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'fouls',
        'fgm', 'fga', 'fg_pct', 'three_pm', 'three_pa', 'three_p_pct', 'ftm', 'fta', 'ft_pct'
    ]
    player_cols_for_merge_and_transform = [c for c in player_cols_for_merge_and_transform if c in player_stats_df_sdv.columns] # Keep only available

    logger.info(f"Player stats columns after rename and percentage calculation (subset for logging): {player_cols_for_merge_and_transform[:15]}") # Log a subset
    if 'player_api_id' in player_cols_for_merge_and_transform : # ensure key columns exist
        logger.info(f"Sample of player_stats_df_sdv (first 3 rows, relevant columns for merge):\\n{player_stats_df_sdv[['game_id', 'player_api_id', 'player_name_from_boxscore', 'player_team_api_id_for_game', 'home_away'] + [c for c in ['fgm', 'fga', 'fg_pct'] if c in player_stats_df_sdv.columns]].head(3).to_string(index=False)}")
    else:
        logger.warning("Key columns like 'player_api_id' missing after rename, cannot log sample player stats.")

    # Merge player stats with schedule data
    if not schedule_df_for_merge['game_id_from_schedule'].is_unique:
        logger.warning("Duplicate game_id_from_schedule found in schedule data. Dropping duplicates before merge.")
        schedule_df_for_merge.drop_duplicates(subset=['game_id_from_schedule'], keep='first', inplace=True)

    merged_df = pd.merge(
        player_stats_df_sdv[player_cols_for_merge_and_transform],
        schedule_df_for_merge,
        left_on='game_id', # game_id from player_stats (already renamed)
        right_on='game_id_from_schedule', # game_id from schedule (renamed)
        how='left' # Keep all player stats, match with schedule if possible
    )

    if 'game_id_x' in merged_df.columns and 'game_id_y' in merged_df.columns:
        logger.info("Merge created 'game_id_x' and 'game_id_y'. Using 'game_id_x' as 'game_id'.")
        merged_df.rename(columns={'game_id_x': 'game_id'}, inplace=True)
        merged_df.drop(columns=['game_id_y'], inplace=True, errors='ignore')
    elif 'game_id_from_schedule' in merged_df.columns and 'game_id' not in merged_df.columns and 'game_id' in player_stats_df_sdv.columns:
         # If left 'game_id' was not suffixed, but right one (from schedule) might be present
         pass # 'game_id' from player_stats is already the primary game_id
    
    logger.info(f"Columns in merged_df after merge and game_id handling: {merged_df.columns.tolist()}")
    
    # Check for missing critical data after merge
    if merged_df['game_date_from_schedule'].isnull().any():
        logger.warning(f"Some player stats rows in merged_df could not be matched with schedule data (missing game_date_from_schedule). These rows might be problematic.")
        # merged_df.dropna(subset=['game_date_from_schedule'], inplace=True) # Option: drop rows that couldn't be matched

    # Log sample of merged_df
    merged_cols_to_log = ['game_id', 'player_api_id', 'game_id_from_schedule', 'game_datetime_from_schedule', 'game_home_team_api_id_from_schedule', 'game_away_team_api_id_from_schedule']
    merged_cols_to_log = [c for c in merged_cols_to_log if c in merged_df.columns]
    if merged_cols_to_log:
        logger.info(f"Sample of merged_df (first 3 rows) after merge and game_id handling:\\n{merged_df[merged_cols_to_log].head(3).to_string(index=False)}")
    else:
        logger.warning("Could not log sample of merged_df due to missing key columns.")

    all_stat_records = []
    games_processed_count = 0

    # Group by game for processing
    for (game_id_val), game_player_stats_pd in merged_df.groupby(['game_id']):
        if pd.isna(game_id_val):
            logger.warning("Found a group with NaN game_id. Skipping this group.")
            continue
        
        game_id_str = str(game_id_val)
        first_row_for_game = game_player_stats_pd.iloc[0]

        game_datetime_val = first_row_for_game.get('game_datetime_parsed')
        game_season_val = first_row_for_game.get('game_season_from_schedule')
        home_team_api_id_val = str(first_row_for_game.get('game_home_team_api_id_from_schedule', ''))
        away_team_api_id_val = str(first_row_for_game.get('game_away_team_api_id_from_schedule', ''))
        home_team_name_val = str(first_row_for_game.get('game_home_team_name_from_schedule', 'Unknown Home'))
        away_team_name_val = str(first_row_for_game.get('game_away_team_name_from_schedule', 'Unknown Away'))

        if pd.isna(game_datetime_val) or pd.isna(game_season_val):
            logger.warning(f"Missing game_datetime_parsed or game_season for game_id {game_id_str}. Cannot process stats for this game.")
            continue
        
        game_season_str = str(game_season_val)

        # Get/Create Team DB objects for the game
        game_home_team_db = get_or_create_team(db, home_team_api_id_val, home_team_name_val)
        game_away_team_db = get_or_create_team(db, away_team_api_id_val, away_team_name_val)

        if not game_home_team_db or not game_away_team_db:
            logger.warning(f"Could not ensure home/away team DB records for game {game_id_str} (Home: {home_team_name_val}, Away: {away_team_name_val}). Skipping stats for this game.")
            continue
            
        # Convert current game's player stats to Polars DataFrame for transform function
        try:
            player_df_pl = pl.from_pandas(game_player_stats_pd)
        except Exception as e:
            logger.error(f"Error converting pandas to polars for game_id {game_id_str}: {e}. Skipping game.")
            continue
            
        stat_records_for_game = transform_player_stats_data(
            df_pl=player_df_pl, 
            game_datetime_obj=game_datetime_val,
            game_id_str=game_id_str,
            game_home_team_db=game_home_team_db, 
            game_away_team_db=game_away_team_db,
            game_season_str=game_season_str, 
            db=db
        )
        
        if stat_records_for_game:
            all_stat_records.extend(stat_records_for_game)
            games_processed_count += 1

    if all_stat_records:
        try:
            db.add_all(all_stat_records)
            db.commit()
            logger.info(f"Successfully loaded {len(all_stat_records)} player stat entries for season {year} across {games_processed_count} games.")
            return len(all_stat_records)
        except IntegrityError as e:
            db.rollback()
            logger.error(f"IntegrityError during bulk insert of player stats: {e}")
            # Could attempt individual inserts here if needed, or log more details
            failed_inserts = 0
            for record in all_stat_records:
                try:
                    db.add(record)
                    db.commit()
                except IntegrityError:
                    db.rollback()
                    # logger.warning(f"Failed to insert individual record: Player {record.player_id}, Game {record.game_id}")
                    failed_inserts += 1
            success_inserts = len(all_stat_records) - failed_inserts
            logger.info(f"After individual retry: Successfully inserted {success_inserts} records, failed {failed_inserts}.")
            return success_inserts
            
        except Exception as e:
            db.rollback()
            logger.error(f"Unexpected error during bulk insert of player stats: {e}")
            return 0
    else:
        logger.info(f"No valid player stat records were generated to load for season {year}.")
        return 0


def main(seasons: Optional[List[int]] = None):
    if seasons is None:
        current_year = datetime.now().year
        seasons = [current_year] # Default to current year if no seasons are provided
        logger.info(f"No seasons provided, defaulting to current year: {current_year}")

    db: Session = SessionLocal()
    total_loaded_count = 0
    try:
        for year in seasons:
            logger.info(f"Processing WNBA stats for season: {year}")
            count = load_wnba_stats(year, db)
            total_loaded_count += count
        logger.info(f"Finished processing all seasons. Total player stat entries loaded: {total_loaded_count}")
    finally:
        db.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load WNBA player stats from sportsdataverse into the database.")
    parser.add_argument("--seasons", type=int, nargs="+", help="A list of seasons (years) to process.")
    args = parser.parse_args()
    
    main(seasons=args.seasons) 