import sys
import os
import logging
import pandas as pd
import numpy as np # Import numpy for np.nan
from sqlalchemy.ext.asyncio import AsyncSession 
from sqlalchemy import select, or_
from sqlalchemy.exc import IntegrityError
from datetime import datetime, date, timezone
import polars as pl
from typing import Optional, List, Dict, Any
import asyncio
import uuid 
import json

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from backend.db.session import AsyncSessionLocal as SessionLocal # This SessionLocal should now be an async_sessionmaker
from backend.db import models as db_models
import sportsdataverse as sdv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper for safe data type conversions ---
def safe_int(value, default=None) -> Optional[int]:
    if pd.notna(value):
        try:
            return int(float(value)) # Convert to float first to handle "10.0" then int
        except (ValueError, TypeError):
            return default
    return default

def safe_float(value, default=None) -> Optional[float]:
    if pd.notna(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    return default

def safe_bool(value, default=None) -> Optional[bool]:
    if pd.notna(value):
        if isinstance(value, str):
            if value.lower() == 'true': return True
            if value.lower() == 'false': return False
        try:
            return bool(value)
        except (ValueError, TypeError): 
            return default
    return default

def parse_made_attempted(stat_str, made_key, attempted_key, row_dict):
    """
    Parse a stat string like '7-10' into made and attempted integers.
    Assigns the values to row_dict[made_key] and row_dict[attempted_key].
    """
    if isinstance(stat_str, str) and '-' in stat_str:
        try:
            made, attempted = stat_str.split('-')
            row_dict[made_key] = int(made)
            row_dict[attempted_key] = int(attempted)
        except Exception:
            row_dict[made_key] = None
            row_dict[attempted_key] = None
    else:
        row_dict[made_key] = None
        row_dict[attempted_key] = None

# --- Helper function to get or create Team ---
async def get_or_create_team(db: AsyncSession, team_api_id_str: Optional[str], team_name_str: Optional[str]) -> Optional[db_models.Team]:
    parsed_team_api_id = str(team_api_id_str).strip() if team_api_id_str else None
    parsed_team_name = str(team_name_str).strip() if team_name_str else None

    if not parsed_team_api_id or parsed_team_api_id.lower() in ['nan', 'none', '']:
        logger.warning(f"get_or_create_team: Received invalid or missing team_api_id: '{team_api_id_str}'. Parsed as: '{parsed_team_api_id}'.")
        if parsed_team_name and parsed_team_name.lower() not in ['nan', 'none', '']:
            logger.info(f"get_or_create_team: Attempting to find team by name only: '{parsed_team_name}'")
            stmt = select(db_models.Team).filter(db_models.Team.team_name == parsed_team_name)
            result = await db.execute(stmt)
            team = result.scalars().first()
            if team:
                logger.info(f"get_or_create_team: Found team by name '{parsed_team_name}' with ID {team.id} and api_team_id {team.api_team_id}.")
                return team
            else:
                logger.error(f"get_or_create_team: Team with name '{parsed_team_name}' not found, and API ID was invalid. Cannot create team under these conditions.")
                return None
        else:
            logger.error(f"get_or_create_team: Both team_api_id ('{team_api_id_str}') and team_name ('{team_name_str}') are missing or invalid. Cannot get/create team.")
        return None

    stmt = select(db_models.Team).filter(db_models.Team.api_team_id == parsed_team_api_id)
    result = await db.execute(stmt)
    team = result.scalars().first()
    if team:
        return team
    else:
        if not parsed_team_name or parsed_team_name.lower() in ['nan', 'none', '']:
            logger.error(f"get_or_create_team: Team with api_team_id '{parsed_team_api_id}' not found, and no valid team_name ('{team_name_str}') provided to create it.")
            return None
        
        logger.info(f"get_or_create_team: Team with api_team_id '{parsed_team_api_id}' not found. Creating new team with name: '{parsed_team_name}'.")
        try:
            new_team = db_models.Team(api_team_id=parsed_team_api_id, team_name=parsed_team_name)
            db.add(new_team)
            await db.commit()
            await db.refresh(new_team)
            logger.info(f"Successfully created team {new_team.team_name} with api_team_id {new_team.api_team_id}")
            return new_team
        except IntegrityError:
            await db.rollback()
            logger.error(f"IntegrityError creating team with api_team_id {parsed_team_api_id} and name {parsed_team_name}. Querying again.")
            stmt_retry = select(db_models.Team).filter(db_models.Team.api_team_id == parsed_team_api_id)
            result_retry = await db.execute(stmt_retry)
            return result_retry.scalars().first()
        except Exception as e:
            await db.rollback()
            logger.error(f"Unexpected error creating team {parsed_team_name} (API ID: {parsed_team_api_id}): {e}")
            return None

# --- Transform Player Stats Data ---
async def transform_player_stats_data(
    db: AsyncSession, 
    df_pl: pl.DataFrame, 
    game_datetime_obj: datetime,
    game_id_str: str,
    game_db_id: uuid.UUID, 
    game_home_team_db: Optional[db_models.Team], 
    game_away_team_db: Optional[db_models.Team],
    game_season: int
) -> List[db_models.PlayerStat]:
    logger.info(f"Transforming {len(df_pl)} player stat entries for game {game_id_str} ({game_datetime_obj})")
    records = []

    if not game_datetime_obj: # Safeguard
        logger.error(f"CRITICAL: transform_player_stats_data called for game {game_id_str} but game_datetime_obj is None. Cannot set game_date. Skipping these player stats.")
        return records

    # Ensure game_db_id, game_home_team_db, game_away_team_db are valid
    if not all([game_db_id, game_home_team_db, game_away_team_db]):
        logger.error(f"CRITICAL: transform_player_stats_data called for game {game_id_str} but game_db_id, game_home_team_db, or game_away_team_db is None. Cannot set game_date. Skipping these player stats.")
        return records

    STAT_COLUMNS_TO_STORE = [
        'minutes_played', 'points', 'rebounds', 'offensive_rebounds', 'defensive_rebounds',
        'assists', 'steals', 'blocks', 
        'turnovers', 'fouls', 'field_goals_made', 'field_goals_attempted',
        'three_pointers_made', 'three_pointers_attempted',
        'free_throws_made', 'free_throws_attempted'
    ]
    REQUIRED_CONTEXT_COLS_FROM_DF_PL = [
        'player_api_id', 'player_name_from_boxscore', 
        'player_team_api_id_for_game', 'player_team_name_for_game',
    ]
    ALL_REQUIRED_COLS_IN_ROW = REQUIRED_CONTEXT_COLS_FROM_DF_PL + STAT_COLUMNS_TO_STORE

    if df_pl.is_empty():
        return []

    schema_cols = df_pl.columns
    missing_schema_cols = [col for col in ALL_REQUIRED_COLS_IN_ROW if col not in schema_cols]
    if 'home_away' not in schema_cols: 
        missing_schema_cols.append('home_away')
    if missing_schema_cols:
        logger.error(f"transform_player_stats_data: DataFrame is missing required columns: {missing_schema_cols}. Cannot process stats for game {game_id_str}.")
        return []

    for pl_row in df_pl.iter_rows(named=True): 
        player_api_id_str = str(pl_row.get('player_api_id'))
        player_name_str = str(pl_row.get('player_name_from_boxscore'))
        player_team_api_id_for_game_str = str(pl_row.get('player_team_api_id_for_game'))
        player_team_name_for_game_str = str(pl_row.get('player_team_name_for_game'))

        if not player_api_id_str or player_api_id_str.lower() == 'nan':
            logger.warning(f"Missing player_api_id for player '{player_name_str}' in game {game_id_str}. Skipping record.")
            continue

        player_home_away_status = str(pl_row.get('home_away', '')).upper()
        is_home_team_for_player_stat = False
        if player_home_away_status == 'HOME':
            is_home_team_for_player_stat = True
        elif player_home_away_status == 'AWAY':
            is_home_team_for_player_stat = False
        else:
            logger.warning(f"Could not determine home/away status for player {player_name_str} (API ID: {player_api_id_str}) in game {game_id_str}. 'home_away' was '{player_home_away_status}'. 'is_home_team' will be False.")

        player_stmt = select(db_models.Player).filter(db_models.Player.api_player_id == player_api_id_str)
        player_result = await db.execute(player_stmt)
        player = player_result.scalars().first()

        current_team_for_game = await get_or_create_team(db, player_team_api_id_for_game_str, player_team_name_for_game_str)

        if player:
            # Player exists, check if their team needs updating
            if current_team_for_game and player.team_id != current_team_for_game.id:
                old_team_id = player.team_id
                player.team_id = current_team_for_game.id
                db.add(player) # Stage the update
                logger.info(f"Player {player.player_name} (ID: {player.id}) changed teams. Old team ID: {old_team_id}, New team ID: {current_team_for_game.id}.")
        else:
            # Player does not exist, create new player
            logger.info(f"Player with api_player_id {player_api_id_str} ({player_name_str}) not found. Creating new player.")
            player = db_models.Player(
                api_player_id=player_api_id_str, 
                player_name=player_name_str,
                team_id=current_team_for_game.id if current_team_for_game else None
            )
            try:
                db.add(player)
                await db.commit()
                await db.refresh(player)
                logger.info(f"Successfully created player {player.id} with api_player_id {player_api_id_str}")
            except IntegrityError as e:
                await db.rollback()
                logger.warning(f"IntegrityError creating player {player_api_id_str} ({player_name_str}): {e}. Querying again.")
                player_stmt_retry = select(db_models.Player).filter(db_models.Player.api_player_id == player_api_id_str)
                player_result_retry = await db.execute(player_stmt_retry)
                player = player_result_retry.scalars().first()
                if not player:
                    logger.error(f"Failed to create or find player {player_api_id_str} after IntegrityError. Skipping record.")
                    continue
            except Exception as e:
                await db.rollback()
                logger.error(f"Error creating new player {player_api_id_str} ({player_name_str}): {e}")
                continue
        
        stat_values: Dict[str, Any] = {}
        for stat_key in STAT_COLUMNS_TO_STORE:
            raw_value = pl_row.get(stat_key)
            if pd.isna(raw_value) or (isinstance(raw_value, str) and raw_value.lower() == 'nan'):
                    stat_values[stat_key] = None
            elif stat_key in ['minutes_played', 'points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'fouls', 
                              'field_goals_made', 'field_goals_attempted', 
                              'three_pointers_made', 'three_pointers_attempted', 
                              'free_throws_made', 'free_throws_attempted',
                              'offensive_rebounds', 'defensive_rebounds']:
                stat_values[stat_key] = safe_int(raw_value)
            else:
                stat_values[stat_key] = raw_value
        
        # Calculate True Shooting Percentage (TS%)
        points = stat_values.get('points')
        fga = stat_values.get('field_goals_attempted')
        fta = stat_values.get('free_throws_attempted')

        if points is not None and fga is not None and fta is not None:
            denominator = 2 * (fga + 0.44 * fta)
            if denominator > 0:
                stat_values['true_shooting_percentage'] = points / denominator
            else:
                stat_values['true_shooting_percentage'] = None
        else:
            stat_values['true_shooting_percentage'] = None

        # Calculate Effective Field Goal Percentage (eFG%)
        fgm = stat_values.get('field_goals_made')
        three_pm = stat_values.get('three_pointers_made')
        if fgm is not None and three_pm is not None and fga is not None and fga > 0:
            stat_values['effective_field_goal_percentage'] = (fgm + 0.5 * three_pm) / fga
        else:
            stat_values['effective_field_goal_percentage'] = None

        # Calculate Summed Stats
        points = stat_values.get('points')
        rebounds = stat_values.get('rebounds')
        assists = stat_values.get('assists')
        blocks = stat_values.get('blocks')
        steals = stat_values.get('steals')

        if points is not None and rebounds is not None and assists is not None:
            stat_values['pra'] = points + rebounds + assists
        else:
            stat_values['pra'] = None

        if points is not None and rebounds is not None:
            stat_values['points_plus_rebounds'] = points + rebounds
        else:
            stat_values['points_plus_rebounds'] = None

        if points is not None and assists is not None:
            stat_values['points_plus_assists'] = points + assists
        else:
            stat_values['points_plus_assists'] = None

        if rebounds is not None and assists is not None:
            stat_values['rebounds_plus_assists'] = rebounds + assists
        else:
            stat_values['rebounds_plus_assists'] = None

        if blocks is not None and steals is not None:
            stat_values['blocks_plus_steals'] = blocks + steals
        else:
            stat_values['blocks_plus_steals'] = None

        existing_stat_stmt = select(db_models.PlayerStat).filter_by(
            game_id=game_db_id,
            player_id=player.id
        )
        existing_stat_result = await db.execute(existing_stat_stmt)
        existing_stat = existing_stat_result.scalars().first()

        player_stat_data = {
            "player_id": player.id,
            "game_id": game_db_id,
            "team_id": game_home_team_db.id if is_home_team_for_player_stat else game_away_team_db.id,
            "game_date": game_datetime_obj.date(),
            "is_home_team": is_home_team_for_player_stat,
            "season": game_season,
            "minutes_played": stat_values.get('minutes_played'),
            "points": stat_values.get('points'),
            "rebounds": stat_values.get('rebounds'),
            "assists": stat_values.get('assists'),
            "steals": stat_values.get('steals'),
            "blocks": stat_values.get('blocks'),
            "turnovers": stat_values.get('turnovers'),
            "fouls": stat_values.get('fouls'),
            "field_goals_made": stat_values.get('field_goals_made'),
            "field_goals_attempted": stat_values.get('field_goals_attempted'),
            "three_pointers_made": stat_values.get('three_pointers_made'),
            "three_pointers_attempted": stat_values.get('three_pointers_attempted'),
            "free_throws_made": stat_values.get('free_throws_made'),
            "free_throws_attempted": stat_values.get('free_throws_attempted'),
            "offensive_rebounds": stat_values.get('offensive_rebounds'),
            "defensive_rebounds": stat_values.get('defensive_rebounds'),
            "true_shooting_percentage": stat_values.get('true_shooting_percentage'),
            "effective_field_goal_percentage": stat_values.get('effective_field_goal_percentage'),
            "pra": stat_values.get('pra'),
            "points_plus_rebounds": stat_values.get('points_plus_rebounds'),
            "points_plus_assists": stat_values.get('points_plus_assists'),
            "rebounds_plus_assists": stat_values.get('rebounds_plus_assists'),
            "blocks_plus_steals": stat_values.get('blocks_plus_steals'),
        }

        if existing_stat:
            # Update existing stat record
            for key, value in player_stat_data.items():
                if key not in ["player_id", "game_id"]: # Don't overwrite PKs
                    setattr(existing_stat, key, value)
            logger.debug(f"Updating existing stat for player {player.id} game {game_db_id}")
        else:
            # Create a new PlayerStat object
            player_stat = db_models.PlayerStat(**player_stat_data)
            records.append(player_stat)
            logger.debug(f"Creating new stat for player {player.id} game {game_db_id}")

    return records


#--- New PBP Processing Logic (to be called from load_wnba_stats) ---
async def transform_and_store_pbp_data(db: AsyncSession, game_db_id: uuid.UUID, pbp_plays_df: pd.DataFrame):
    """
    Transforms and stores play-by-play data for a given game.
    """
    if pbp_plays_df.empty:
        logger.info(f"No PBP data to process for game_id: {game_db_id}")
        return

    logger.info(f"Processing {len(pbp_plays_df)} PBP events for game_id: {game_db_id}")

    # For now, we will store the raw JSON of the PBP data in a single record.
    # In the future, we could parse this into individual event records.
    
    # Check if PBP data already exists for this game
    existing_pbp_stmt = select(db_models.PlayByPlayEvent).filter_by(game_id=game_db_id)
    result = await db.execute(existing_pbp_stmt)
    existing_pbp = result.scalars().first()

    if existing_pbp:
        logger.info(f"PBP data already exists for game {game_db_id}. Overwriting.")
        pbp_record = existing_pbp
    else:
        logger.info(f"No existing PBP data for game {game_db_id}. Creating new record.")
        pbp_record = db_models.PlayByPlayEvent(game_id=game_db_id)
        db.add(pbp_record)

    # Convert DataFrame to a list of dicts (JSON serializable)
    try:
        # Convert all NaN to None for clean JSON
        pbp_plays_df_cleaned = pbp_plays_df.replace({np.nan: None})
        pbp_json_data = pbp_plays_df_cleaned.to_dict(orient='records')
        pbp_record.pbp_data = pbp_json_data
        pbp_record.updated_at = datetime.now(timezone.utc)
    except Exception as e:
        logger.error(f"Error converting PBP data to JSON for game {game_db_id}: {e}")
        # If there's an error, we probably don't want to store partial/bad data
        return
        
    try:
        await db.commit()
        logger.info(f"Successfully stored/updated PBP data for game {game_db_id}")
    except Exception as e:
        await db.rollback()
        logger.error(f"Error committing PBP data for game {game_db_id}: {e}")

# --- Main data loading function (heavily modified) ---
async def load_wnba_stats(year: int, db: AsyncSession, max_games_to_process: Optional[int] = None) -> int:
    """
    Fetches WNBA stats for a given year, processing games one by one to fetch box scores,
    player stats, and potentially PBP data. It now uses a more robust method to handle
    cases where box score data is not included in the main schedule endpoint.
    """
    logger.info(f"--- Starting WNBA stats scraper for year: {year} ---")
    processed_games_count = 0
    
    try:
        # Step 1: Fetch the schedule for the entire year
        schedule_df = sdv.wnba.load_wnba_schedule(seasons=[year], return_as_pandas=True)
        if schedule_df.empty:
            logger.warning(f"No schedule data found for year {year}. Exiting.")
            return 0
        logger.info(f"Successfully loaded schedule with {len(schedule_df)} games for {year}.")

    except Exception as e:
        logger.error(f"Failed to load schedule for year {year}. Error: {e}", exc_info=True)
        return 0

    games_to_process_df = schedule_df
    if max_games_to_process is not None:
        logger.info(f"Processing a maximum of {max_games_to_process} games.")
        # Taking a slice of the DataFrame to process
        games_to_process_df = schedule_df.head(max_games_to_process)

    # Step 2: Iterate through each game in the schedule
    for _, game_row in games_to_process_df.iterrows():
        game_id_str = str(game_row.get('game_id'))
        game_date_str = game_row.get('game_date')
        game_time_str = game_row.get('game_date_time')
        
        logger.info(f"Processing game_id: {game_id_str}")

        # Basic game info
        home_team_api_id = str(game_row.get('home_id'))
        away_team_api_id = str(game_row.get('away_id'))
        home_team_name = str(game_row.get('home_full_name'))
        away_team_name = str(game_row.get('away_full_name'))

        game_datetime_obj = None
        if pd.notna(game_time_str):
            try:
                # Assuming game_time_str is in a format like '2024-05-15T00:00:00Z'
                game_datetime_obj = datetime.fromisoformat(game_time_str.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                 logger.warning(f"Could not parse game_date_time: '{game_time_str}'. Falling back to game_date.")
                 game_datetime_obj = None

        if not game_datetime_obj and pd.notna(game_date_str):
            try:
                # Fallback if time is not available or parsing failed
                game_datetime_obj = datetime.strptime(str(game_date_str), '%Y-%m-%d')
                game_datetime_obj = game_datetime_obj.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                logger.error(f"Could not parse game_date: '{game_date_str}'. Cannot process game {game_id_str}.")
                continue
        
        if not game_datetime_obj:
            logger.error(f"No valid date/time found for game {game_id_str}. Skipping.")
            continue
            
        # Step 3: Get or create teams
        home_team_db = await get_or_create_team(db, home_team_api_id, home_team_name)
        away_team_db = await get_or_create_team(db, away_team_api_id, away_team_name)

        if not home_team_db or not away_team_db:
            logger.error(f"Could not find or create one of the teams for game {game_id_str}. Home: '{home_team_name}', Away: '{away_team_name}'. Skipping game.")
            continue
            
        # Step 4: Get or create the game record
        game_stmt = select(db_models.Game).filter(db_models.Game.the_odds_api_game_id == game_id_str)
        game_result = await db.execute(game_stmt)
        game = game_result.scalars().first()

        if not game:
            game = db_models.Game(
                the_odds_api_game_id=game_id_str,
                game_datetime=game_datetime_obj, # Use the full datetime object
                home_team_id=home_team_db.id,
                away_team_id=away_team_db.id,
                season=year
            )
            db.add(game)
            try:
                await db.commit()
                await db.refresh(game)
                logger.info(f"Created new game record for game_id: {game_id_str}")
            except IntegrityError:
                await db.rollback()
                logger.warning(f"IntegrityError creating game {game_id_str}. Querying again.")
                game_stmt_retry = select(db_models.Game).filter(db_models.Game.the_odds_api_game_id == game_id_str)
                game_result_retry = await db.execute(game_stmt_retry)
                game = game_result_retry.scalars().first()
                if not game:
                    logger.error(f"Failed to create or find game {game_id_str} after IntegrityError. Skipping.")
                    continue
            except Exception as e:
                await db.rollback()
                logger.error(f"An unexpected error occurred while creating game {game_id_str}: {e}")
                continue

        # Now `game` is guaranteed to be a valid DB object
        game_db_id = game.id
        
        # --- Player Stat Processing Logic ---
        player_box_df = pd.DataFrame() # Initialize as empty DataFrame
        
        # Step 5: Check for player box score data in the schedule data first
        # sdv.wnba.load_wnba_schedule often returns a 'player_box' column containing a DF
        if 'player_box' in game_row and isinstance(game_row['player_box'], pd.DataFrame) and not game_row['player_box'].empty:
            logger.info(f"Found 'player_box' data directly in schedule for game {game_id_str}.")
            player_box_df = game_row['player_box']
        else:
            # Step 6: If not in schedule, fallback to fetching box score endpoint
            logger.warning(f"'player_box' not in schedule data for {game_id_str}. Fetching from boxscore endpoint.")
            try:
                # This is the endpoint that fetches the detailed box score
                game_boxscore_data = sdv.wnba.espn_wnba_pbp(game_id=int(game_id_str))
                
                # The returned data is a dictionary. We need to find the player stats within it.
                # Based on observed structure, it's often in a 'boxscore' -> 'players' list of dicts
                if 'boxscore' in game_boxscore_data and 'players' in game_boxscore_data['boxscore']:
                    player_stats_list_of_dicts = game_boxscore_data['boxscore']['players']
                    if player_stats_list_of_dicts:
                        player_box_df = pd.DataFrame(player_stats_list_of_dicts)
                        logger.info(f"Successfully fetched and parsed boxscore for game {game_id_str}.")
                    else:
                        logger.warning(f"Boxscore fetched for {game_id_str}, but 'players' list was empty.")
                else:
                    logger.warning(f"Boxscore fetched for {game_id_str}, but 'boxscore' or 'players' key was not found.")
            except Exception as e:
                logger.error(f"Error fetching/parsing boxscore for game {game_id_str}: {e}")
                # Continue to next game if box score fails
                continue

        # Step 7: Standardize and Process the player box score DataFrame
        if not player_box_df.empty:
            # --- STANDARDIZATION LOGIC ---
            # This logic handles two different data structures returned by the API:
            # 1. A flat structure from the 'player_box' field in the schedule endpoint.
            # 2. A nested structure from the boxscore endpoint.
            standardized_rows = []
            
            # Case 1: Data is from the schedule endpoint (flat structure)
            if 'athlete_id' in player_box_df.columns:
                logger.info("Processing player data from schedule schema.")
                for _, player_row in player_box_df.iterrows():
                    standard_row = {}
                    standard_row['player_api_id'] = player_row.get('athlete_id')
                    standard_row['player_name_from_boxscore'] = player_row.get('athlete_display_name')
                    standard_row['player_team_api_id_for_game'] = player_row.get('team_id')
                    standard_row['player_team_name_for_game'] = player_row.get('team_short_display_name')
                    standard_row['home_away'] = player_row.get('home_away')
                    
                    standard_row['minutes_played'] = safe_int(player_row.get('min'))
                    standard_row['points'] = safe_int(player_row.get('pts'))
                    standard_row['rebounds'] = safe_int(player_row.get('reb'))
                    standard_row['assists'] = safe_int(player_row.get('ast'))
                    standard_row['steals'] = safe_int(player_row.get('stl'))
                    standard_row['blocks'] = safe_int(player_row.get('blk'))
                    standard_row['turnovers'] = safe_int(player_row.get('to'))
                    standard_row['fouls'] = safe_int(player_row.get('pf'))
                    standard_row['offensive_rebounds'] = safe_int(player_row.get('oreb'))
                    standard_row['defensive_rebounds'] = safe_int(player_row.get('dreb'))
                    
                    parse_made_attempted(player_row.get('fg'), 'field_goals_made', 'field_goals_attempted', standard_row)
                    parse_made_attempted(player_row.get('3pt'), 'three_pointers_made', 'three_pointers_attempted', standard_row)
                    parse_made_attempted(player_row.get('ft'), 'free_throws_made', 'free_throws_attempted', standard_row)
                    
                    if standard_row.get('player_api_id'):
                        standardized_rows.append(standard_row)

            # Case 2: Data is from the boxscore endpoint (nested structure)
            elif 'statistics' in player_box_df.columns:
                logger.info("Processing player data from boxscore schema.")
                for _, team_row in player_box_df.iterrows():
                    if not team_row.get('statistics'):
                        continue
                    
                    stats_data = team_row['statistics'][0]
                    team_info = team_row['team']
                    stat_keys = stats_data['keys']
                    
                    for player_data in stats_data['athletes']:
                        if not player_data.get('athlete') or not player_data.get('stats'):
                            continue

                        standard_row = {}
                        athlete_info = player_data['athlete']
                        player_stats_list = player_data['stats']
                        player_stats_dict = dict(zip(stat_keys, player_stats_list))
                        
                        standard_row['player_api_id'] = athlete_info.get('id')
                        standard_row['player_name_from_boxscore'] = athlete_info.get('displayName')
                        standard_row['player_team_api_id_for_game'] = team_info.get('id')
                        standard_row['player_team_name_for_game'] = team_info.get('shortDisplayName')
                        
                        if team_info.get('id') == home_team_api_id:
                            standard_row['home_away'] = 'HOME'
                        else:
                            standard_row['home_away'] = 'AWAY'

                        standard_row['minutes_played'] = safe_int(player_stats_dict.get('minutes'))
                        standard_row['points'] = safe_int(player_stats_dict.get('points'))
                        standard_row['rebounds'] = safe_int(player_stats_dict.get('rebounds'))
                        standard_row['assists'] = safe_int(player_stats_dict.get('assists'))
                        standard_row['steals'] = safe_int(player_stats_dict.get('steals'))
                        standard_row['blocks'] = safe_int(player_stats_dict.get('blocks'))
                        standard_row['turnovers'] = safe_int(player_stats_dict.get('turnovers'))
                        standard_row['fouls'] = safe_int(player_stats_dict.get('fouls'))
                        standard_row['offensive_rebounds'] = safe_int(player_stats_dict.get('offensiveRebounds'))
                        standard_row['defensive_rebounds'] = safe_int(player_stats_dict.get('defensiveRebounds'))

                        parse_made_attempted(player_stats_dict.get('fieldGoalsMade-fieldGoalsAttempted'), 'field_goals_made', 'field_goals_attempted', standard_row)
                        parse_made_attempted(player_stats_dict.get('threePointFieldGoalsMade-threePointFieldGoalsAttempted'), 'three_pointers_made', 'three_pointers_attempted', standard_row)
                        parse_made_attempted(player_stats_dict.get('freeThrowsMade-freeThrowsAttempted'), 'free_throws_made', 'free_throws_attempted', standard_row)
                        
                        if standard_row.get('player_api_id'):
                            standardized_rows.append(standard_row)
            
            if standardized_rows:
                player_stats_pl = pl.from_dicts(standardized_rows)
                
                player_stats_to_store = await transform_player_stats_data(
                    db=db,
                    df_pl=player_stats_pl,
                    game_datetime_obj=game_datetime_obj,
                    game_id_str=game_id_str,
                    game_db_id=game_db_id,
                    game_home_team_db=home_team_db,
                    game_away_team_db=away_team_db,
                    game_season=year
                )

                if player_stats_to_store:
                    try:
                        db.add_all(player_stats_to_store)
                        await db.commit()
                        logger.info(f"Successfully stored {len(player_stats_to_store)} player stat records for game {game_id_str}.")
                        processed_games_count += 1
                    except Exception as e:
                        await db.rollback()
                        logger.error(f"Error storing player stats for game {game_id_str}: {e}")
            else:
                logger.warning(f"Could not standardize any player data for game {game_id_str}. Skipping stat storage.")

        else:
            logger.warning(f"No player box score data could be found or fetched for game {game_id_str}.")

        # --- PBP Data Processing ---
        # The wnba_pbp function also returns PBP data if available. Let's check for it.
        # We re-fetch it here to ensure we have it, even if box score was in schedule.
        try:
            pbp_data = sdv.wnba.espn_wnba_pbp(game_id=int(game_id_str))
            if pbp_data and 'plays' in pbp_data and pbp_data['plays']:
                pbp_df = pd.DataFrame(pbp_data['plays'])
                # Call the async function to process and store PBP data
                await transform_and_store_pbp_data(db, game.id, pbp_df)
            else:
                logger.info(f"No PBP 'plays' data available for game {game_id_str}.")
        except Exception as e:
            logger.error(f"Failed to fetch or process PBP data for game {game_id_str}: {e}")
            
    logger.info(f"--- Finished scraper for year: {year}. Processed {processed_games_count} games with player stats. ---")
    return processed_games_count


async def main(seasons: Optional[List[int]] = None, max_games_per_season: Optional[int] = None):
    """
    Main function to run the WNBA stats scraper.
    """
    if seasons is None:
        seasons = [datetime.now().year] # Default to current year

    logger.info(f"Starting main scraper for seasons: {seasons}")
    
    # Use a single session for the entire run if possible
    async with SessionLocal() as db:
        for year in seasons:
            await load_wnba_stats(year, db, max_games_per_season)

    logger.info("Scraping process completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WNBA Stats Scraper")
    parser.add_argument("--years", type=int, nargs='*', help="List of years to scrape stats for.")
    parser.add_argument("--max-games", type=int, default=None, help="Maximum number of games to process per season.")
    
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    if loop.is_running():
        # This is the case for environments like Jupyter notebooks
        task = loop.create_task(main(seasons=args.years, max_games_per_season=args.max_games))
    else:
        # for standard script execution
        asyncio.run(main(seasons=args.years, max_games_per_season=args.max_games))