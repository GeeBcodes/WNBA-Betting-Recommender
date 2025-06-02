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

from backend.db.session import SessionLocal # This SessionLocal should now be an async_sessionmaker
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
    game_season_str: str
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

        if not player:
            logger.info(f"Player with api_player_id {player_api_id_str} ({player_name_str}) not found. Creating new player.")
            primary_team_for_player = await get_or_create_team(db, player_team_api_id_for_game_str, player_team_name_for_game_str)
            player = db_models.Player(
                api_player_id=player_api_id_str, 
                player_name=player_name_str,
                team_id=primary_team_for_player.id if primary_team_for_player else None
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
        fga_efg = stat_values.get('field_goals_attempted') # Use a different variable name to avoid conflict if needed

        if fgm is not None and three_pm is not None and fga_efg is not None:
            if fga_efg > 0:
                stat_values['effective_field_goal_percentage'] = (fgm + 0.5 * three_pm) / fga_efg
            else:
                stat_values['effective_field_goal_percentage'] = None
        else:
            stat_values['effective_field_goal_percentage'] = None

        # Calculate Turnover Percentage (TOV%)
        tov_player = stat_values.get('turnovers')
        fga_tov = stat_values.get('field_goals_attempted')
        fta_tov = stat_values.get('free_throws_attempted')

        if tov_player is not None and fga_tov is not None and fta_tov is not None:
            denominator_tov = fga_tov + 0.44 * fta_tov + tov_player
            if denominator_tov > 0:
                stat_values['turnover_percentage'] = 100 * tov_player / denominator_tov
            else:
                # If FGA, FTA, and TOV are all 0, TOV% is 0. If TOV > 0 but FGA/FTA are 0, it could be infinite, treat as high or None.
                # For simplicity, if denominator is 0, and TOV is also 0, TOV% is 0. Otherwise, it's undefined (None).
                stat_values['turnover_percentage'] = 0.0 if tov_player == 0 else None 
        else:
            stat_values['turnover_percentage'] = None

        existing_stat_stmt = select(db_models.PlayerStat).filter_by(player_id=player.id, game_id=game_db_id)
        existing_stat_res = await db.execute(existing_stat_stmt)
        existing_player_stat = existing_stat_res.scalars().first()

        if existing_player_stat:
            if existing_player_stat.game_date is None and game_datetime_obj:
                logger.info(f"PlayerStat for player {player_name_str} in game {game_id_str} exists but game_date is NULL. Updating game_date to {game_datetime_obj.date()}.")
                existing_player_stat.game_date = game_datetime_obj.date()
                # Potentially update other fields if they might also be stale, 
                # but for now, focusing on game_date as per the issue.
                # Example: existing_player_stat.season = int(game_season_str)
                # for stat_key_to_update, new_value in stat_values.items():
                #    setattr(existing_player_stat, stat_key_to_update, new_value)
                try:
                    await db.commit()
                    await db.refresh(existing_player_stat)
                except Exception as e_update:
                    await db.rollback()
                    logger.error(f"Error updating existing PlayerStat's game_date for player {player.id} game {game_db_id}: {e_update}")
            else:
                logger.info(f"PlayerStat for player {player_name_str} in game {game_id_str} already exists and game_date is populated ({existing_player_stat.game_date}). Skipping record creation.")
            continue # Continue to the next player row
        
        # If no existing record, create a new one
        record = db_models.PlayerStat(
            game_id=game_db_id,
            player_id=player.id,
            team_id=player.team_id if player else None, 
            game_date=game_datetime_obj.date() if game_datetime_obj else None, # Explicit check, though covered by safeguard above
            is_home_team=is_home_team_for_player_stat,
            season=int(game_season_str),
            **stat_values 
        )
        records.append(record)

    if records:
        try:
            db.add_all(records)
            await db.commit()
            logger.info(f"Successfully added {len(records)} player stat records for game {game_id_str}.")
        except Exception as e:
            await db.rollback()
            logger.error(f"Error bulk adding player stats for game {game_id_str}: {e}")

    return records

async def transform_and_store_pbp_data(db: AsyncSession, game_db_id: uuid.UUID, pbp_plays_df: pd.DataFrame):
    logger.info(f"transform_and_store_pbp_data for game DB ID: {game_db_id}. Processing {len(pbp_plays_df)} plays.")
    
    pbp_event_records = []
    for index, play_row in pbp_plays_df.iterrows():
        shot_made: Optional[bool] = None
        shot_type_detail: Optional[str] = None
        try:
            event_text_raw = play_row.get("text")
            type_text_raw = play_row.get("type_text")

            event_data = {
                "game_id": game_db_id,
                "external_event_id": str(play_row.get("id")) if pd.notna(play_row.get("id")) else None,
                "sequence_number": safe_int(play_row.get("sequence_number")),
                "event_text": str(event_text_raw) if pd.notna(event_text_raw) else None,
                "away_score_after_event": safe_int(play_row.get("away_score")),
                "home_score_after_event": safe_int(play_row.get("home_score")),
                "scoring_play": safe_bool(play_row.get("scoring_play")),
                "score_value": safe_int(play_row.get("score_value"), default=0),
                "wallclock_time": pd.to_datetime(play_row.get("wallclock")).isoformat() if pd.notna(play_row.get("wallclock")) else None,
                "shooting_play": safe_bool(play_row.get("shooting_play")),
                "event_type_id": safe_int(play_row.get("type_id")),
                "event_type_text": str(type_text_raw) if pd.notna(type_text_raw) else None,
                "period": safe_int(play_row.get("period_number")),
                "game_clock_display": str(play_row.get("clock_display_value")) if pd.notna(play_row.get("clock_display_value")) else None,
                "coordinate_x": safe_float(play_row.get("coordinate_x_raw")),
                "coordinate_y": safe_float(play_row.get("coordinate_y_raw")),
                "shot_made": shot_made,
                "shot_type_detail": shot_type_detail,
                "created_at": datetime.now(timezone.utc)
            }

            pbp_team_api_id = str(play_row.get("team_id")) if pd.notna(play_row.get("team_id")) else None
            if pbp_team_api_id:
                team_stmt = select(db_models.Team.id).filter(db_models.Team.api_team_id == pbp_team_api_id)
                team_res = await db.execute(team_stmt)
                team_db_id = team_res.scalars().first()
                if team_db_id:
                    event_data["team_id"] = team_db_id
                else:
                    logger.warning(f"PBP: Team with api_id {pbp_team_api_id} not found for game {game_db_id}, play {event_data.get('external_event_id')}.")
            
            for i in range(1, 4):
                player_api_id_col = f"athlete_id_{i}"
                player_key_in_event = f"player{i}_id"
                player_api_id = str(play_row.get(player_api_id_col)) if pd.notna(play_row.get(player_api_id_col)) else None
                if player_api_id:
                    player_stmt = select(db_models.Player.id).filter(db_models.Player.api_player_id == player_api_id)
                    player_res = await db.execute(player_stmt)
                    player_db_id = player_res.scalars().first()
                    if player_db_id:
                        event_data[player_key_in_event] = player_db_id
                    else:
                        logger.warning(f"PBP: Player with api_id {player_api_id} ({player_api_id_col}) not found for game {game_db_id}, play {event_data.get('external_event_id')}.")

            if event_data.get("shooting_play"):
                current_event_type_text = event_data.get("event_type_text", "").lower()
                current_event_text = event_data.get("event_text", "").lower()

                if "made" in current_event_type_text or "makes" in current_event_text:
                    event_data["shot_made"] = True
                elif "missed" in current_event_type_text or "misses" in current_event_text:
                    event_data["shot_made"] = False

                if "free throw" in current_event_type_text or "free throw" in current_event_text:
                    event_data["shot_type"] = "FT"
                elif "three point" in current_event_type_text or "3-pt" in current_event_text or "three point" in current_event_text:
                    event_data["shot_type"] = "3PT"
                elif ("field goal" in current_event_type_text or "shot" in current_event_type_text or \
                      "layup" in current_event_type_text or "dunk" in current_event_type_text or \
                      "jumper" in current_event_type_text) and not event_data.get("shot_type") == "FT":
                    event_data["shot_type"] = "2PT"
                elif ("two point" in current_event_text) and not event_data.get("shot_type") == "FT": 
                    event_data["shot_type"] = "2PT"
            
            pbp_event = db_models.PlayByPlayEvent(**event_data)
            pbp_event_records.append(pbp_event)

        except Exception as e:
            logger.error(f"Error processing PBP play row for game {game_db_id}: {play_row.get('id', 'UNKNOWN_ID')}. Error: {e}", exc_info=True)
            # continue # This continue was part of the error; loop will continue naturally

    if pbp_event_records:
        try:
            db.add_all(pbp_event_records)
            await db.commit()
            logger.info(f"Successfully stored {len(pbp_event_records)} PBP events for game {game_db_id}.")
        except Exception as e:
            await db.rollback()
            logger.error(f"Error bulk storing PBP events for game {game_db_id}: {e}", exc_info=True)

async def load_wnba_stats(year: int, db: AsyncSession, max_games_to_process: Optional[int] = None) -> int:
    logger.info(f"Starting WNBA stats loading for year: {year}")
    game_count = 0
    processed_stats_count = 0 
    actually_processed_game_count = 0

    try:
        schedule = sdv.wnba.load_wnba_schedule(seasons=year)
        schedule_pl = pl.from_pandas(schedule) if isinstance(schedule, pd.DataFrame) else schedule
        
        if schedule_pl.is_empty():
            logger.warning(f"No schedule data found for WNBA year {year}.")
            return 0
    
        logger.info(f"Found {len(schedule_pl)} games in schedule for {year}.")

        for game_info in schedule_pl.iter_rows(named=True):
            # Try block for processing a single game starts here
            try: 
                if max_games_to_process is not None and actually_processed_game_count >= max_games_to_process:
                    logger.info(f"Reached max_games_to_process limit ({max_games_to_process}). Stopping further game processing for year {year}.")
                    break

                g_id = str(game_info.get('game_id'))
                game_date_str = game_info.get('date')
                game_status = game_info.get('status_type_completed', False)
                game_datetime_obj: Optional[datetime] = None

                if game_date_str:
                    try:
                        game_datetime_obj = pd.to_datetime(game_date_str).to_pydatetime()
                        if game_datetime_obj.tzinfo is None: game_datetime_obj = game_datetime_obj.replace(tzinfo=timezone.utc)
                    except Exception as e: 
                        logger.error(f"Could not parse game_date_str '{game_date_str}' for game {g_id}. Error: {e}. Skipping game.")
                        continue # Skip this game if date parsing fails
                else:
                    logger.warning(f"Missing game date for game {g_id}. Skipping game.")
                    continue # Skip this game if date is missing
            
                if not game_status:
                    logger.info(f"Game {g_id} on {game_datetime_obj.strftime('%Y-%m-%d')} is not completed (status: {game_info.get('status_display_name')}). Skipping.")
                    continue # Skip this game if not completed
            
                logger.info(f"Processing game: {g_id} on {game_datetime_obj.strftime('%Y-%m-%d')}")
                actually_processed_game_count += 1

                home_team_api_id = str(game_info.get('home_id'))
                home_team_name = str(game_info.get('home_full_name'))
                away_team_api_id = str(game_info.get('away_id'))
                away_team_name = str(game_info.get('away_full_name'))

                home_team_db = await get_or_create_team(db, home_team_api_id, home_team_name)
                away_team_db = await get_or_create_team(db, away_team_api_id, away_team_name)

                if not home_team_db or not away_team_db:
                    logger.error(f"Could not get/create home or away team for game {g_id}. Home: {home_team_name}, Away: {away_team_name}. Skipping game.")
                    continue # Skip this game
                
                game_external_id = g_id
                game_season_str = str(game_info.get('season_year', year))
                
                scheduled_home_score = safe_int(game_info.get('home_score'))
                scheduled_away_score = safe_int(game_info.get('away_score'))

                game_stmt = select(db_models.Game).filter(db_models.Game.external_id == game_external_id)
                game_res = await db.execute(game_stmt)
                game_db = game_res.scalars().first()

                if not game_db:
                    logger.info(f"Game with external_id {game_external_id} not found. Creating new game.")
                    game_db = db_models.Game(
                        external_id=game_external_id,
                        game_datetime=game_datetime_obj,
                        home_team_id=home_team_db.id,
                        away_team_id=away_team_db.id,
                        season=int(game_season_str),
                        home_score=scheduled_home_score,
                        away_score=scheduled_away_score,
                    )
                    try:
                        db.add(game_db)
                        await db.commit()
                        await db.refresh(game_db)
                        logger.info(f"Successfully created game {game_db.id} (external: {game_external_id})")
                    except IntegrityError:
                        await db.rollback()
                        logger.warning(f"IntegrityError creating game {game_external_id}. Querying again.")
                        game_stmt_retry = select(db_models.Game).filter(db_models.Game.external_id == game_external_id)
                        game_res_retry = await db.execute(game_stmt_retry)
                        game_db = game_res_retry.scalars().first()
                        if not game_db:
                            logger.error(f"Failed to create or find game {game_external_id} after IntegrityError. Skipping for this game.")
                            continue # Skip this game
                    except Exception as e:
                        await db.rollback()
                        logger.error(f"Error creating new game {game_external_id}: {e}. Skipping for this game.")
                        continue # Skip this game
                
                pbp_raw_data = None
                player_box_data = None
                try:
                    logger.info(f"Fetching PBP and Boxscore data using espn_wnba_pbp for game {g_id} (DB ID: {game_db.id})...")
                    pbp_raw_data = sdv.wnba.espn_wnba_pbp(game_id=g_id)
                    logger.info(f"Fetching player box score data using load_wnba_player_boxscore for game {g_id}...")
                    player_box_data = sdv.wnba.load_wnba_player_boxscore(int(game_season_str))
                    if isinstance(player_box_data, pd.DataFrame):
                        if pd.api.types.is_numeric_dtype(player_box_data['game_id']):
                            filter_game_id = int(g_id)
                        else:
                            filter_game_id = str(g_id)
                        player_box_data = player_box_data[player_box_data['game_id'] == filter_game_id]
                    elif hasattr(player_box_data, 'filter'):
                        try:
                            filter_game_id = int(g_id)
                            player_box_data = player_box_data.filter(pl.col('game_id') == filter_game_id)
                        except Exception as e_filter: # Renamed 'e' to avoid conflict
                            logger.error(f"Error filtering Polars player_box_data for game_id {g_id}: {e_filter}")
                    logger.info(f"player_box_data type: {type(player_box_data)}")
                    logger.info(f"player_box_data content (truncated): {str(player_box_data)[:2000]}")
                except Exception as e_fetch: # Renamed 'e' to avoid conflict
                    logger.error(f"Error fetching PBP/Boxscore/player box data for game {g_id}: {e_fetch}", exc_info=True)
                    game_count += 1 # Increment game_count even on fetch error if we are to continue processing other games
                    # Consider if this 'continue' is desired or if partial data processing is possible
                    continue # Skip to next game if data fetch fails

                if pbp_raw_data and 'boxscore' in pbp_raw_data:
                    logger.info(f"pbp_raw_data['boxscore'] type: {type(pbp_raw_data['boxscore'])}")
                    logger.info(f"pbp_raw_data['boxscore'] content (truncated): {str(pbp_raw_data['boxscore'])[:2000]}")
                else:
                    logger.info(f"pbp_raw_data['boxscore'] is missing for game {g_id}.")

                player_stats_source = player_box_data
                all_player_stats_dfs_list = []
                if isinstance(player_stats_source, pd.DataFrame):
                    all_player_stats_dfs_list.append(player_stats_source)
                elif isinstance(player_stats_source, pl.DataFrame):
                    all_player_stats_dfs_list.append(player_stats_source.to_pandas())
                elif isinstance(player_stats_source, list):
                    for item in player_stats_source:
                        if isinstance(item, pd.DataFrame):
                            all_player_stats_dfs_list.append(item)
                        elif isinstance(item, pl.DataFrame):
                            all_player_stats_dfs_list.append(item.to_pandas())
                        elif isinstance(item, dict):
                            all_player_stats_dfs_list.append(pd.DataFrame([item]))
                        else:
                            logger.warning(f"Item in player_stats_source list is not a DataFrame or Dict. Type: {type(item)}. Game: {g_id}")
                            logger.warning(f"player_stats_source content: {str(item)[:1000]}")
                else:
                    logger.warning(f"player_stats_source is not a DataFrame or list for game {g_id}. Type: {type(player_stats_source)}")
                    logger.warning(f"player_stats_source content: {str(player_stats_source)[:1000]}")

                if all_player_stats_dfs_list:
                    player_box_sdv_combined = pd.concat(all_player_stats_dfs_list, ignore_index=True)
                    player_box_pl = pl.from_pandas(player_box_sdv_combined) if isinstance(player_box_sdv_combined, pd.DataFrame) else player_box_sdv_combined

                    if player_box_pl.is_empty():
                        logger.warning(f"No player box score data extracted after processing for game {g_id}.")
                    else:
                        logger.info(f"Successfully extracted {len(player_box_pl)} player stat entries for game {g_id}. Columns: {player_box_pl.columns}")
                        rename_map = {
                            'id': 'player_api_id', 
                            'athlete_id': 'player_api_id',
                            'athlete_id_from_struct': 'player_api_id',
                            'display_name': 'player_name_from_boxscore',
                            'athlete_display_name': 'player_name_from_boxscore',
                            'min': 'minutes_played_str', 'pts': 'points', 'reb': 'rebounds', 'ast': 'assists',
                            'stl': 'steals', 'blk': 'blocks', 'to': 'turnovers', 'pf': 'fouls',
                            'fg': 'fg_str', 'fgm':'field_goals_made', 'fga':'field_goals_attempted', 
                            '3pt': 'three_p_str', '3pm':'three_pointers_made', '3pa':'three_pointers_attempted', 
                            'ft': 'ft_str', 'ftm':'free_throws_made', 'fta':'free_throws_attempted', 
                            'oreb': 'offensive_rebounds', 'dreb': 'defensive_rebounds', 
                            'plus_minus': 'plus_minus',
                            'team_id': 'player_team_api_id_for_game',
                            'team_abbreviation': 'player_team_name_for_game', 
                            'homeAway': 'home_away' 
                        }
                        actual_renames = {k: v for k, v in rename_map.items() if k in player_box_pl.columns}
                        if not actual_renames and not player_box_pl.is_empty():
                            logger.warning(f"Player stats rename_map resulted in no columns being renamed for game {g_id}. Original columns: {player_box_pl.columns}")
                        player_box_pl_renamed = player_box_pl.rename(actual_renames)

                        processed_rows = []
                        for row_dict in player_box_pl_renamed.to_dicts():
                            minutes_val = row_dict.get('minutes_played_str')
                            if pd.notna(minutes_val):
                                try: row_dict['minutes_played'] = int(float(minutes_val))
                                except (ValueError, TypeError): row_dict['minutes_played'] = None
                            else:
                                row_dict['minutes_played'] = None
                            if not all(k in row_dict and pd.notna(row_dict[k]) for k in ['field_goals_made', 'field_goals_attempted']):
                                parse_made_attempted(row_dict.get('fg_str'), 'field_goals_made', 'field_goals_attempted', row_dict)
                            if not all(k in row_dict and pd.notna(row_dict[k]) for k in ['three_pointers_made', 'three_pointers_attempted']):
                                parse_made_attempted(row_dict.get('three_p_str'), 'three_pointers_made', 'three_pointers_attempted', row_dict)
                            if not all(k in row_dict and pd.notna(row_dict[k]) for k in ['free_throws_made', 'free_throws_attempted']):
                                parse_made_attempted(row_dict.get('ft_str'), 'free_throws_made', 'free_throws_attempted', row_dict)
                            critical_fields_present = True
                            for key_field in ['player_api_id', 'player_name_from_boxscore']:
                                if key_field not in row_dict or row_dict[key_field] is None or str(row_dict[key_field]).lower() == 'nan':
                                    logger.warning(f"Critical field '{key_field}' missing or invalid for a player stat record in game {g_id}. Record: {str(row_dict)[:100]}")
                                    critical_fields_present = False
                                    break
                            if not critical_fields_present: continue
                            if 'player_team_api_id_for_game' not in row_dict or pd.isna(row_dict['player_team_api_id_for_game']):
                                logger.warning(f"Missing 'player_team_api_id_for_game' for {row_dict.get('player_name_from_boxscore')}. Needs robust handling.")
                                row_dict['player_team_api_id_for_game'] = None
                            if 'home_away' not in row_dict or pd.isna(row_dict['home_away']):
                                logger.warning(f"Missing 'home_away' for {row_dict.get('player_name_from_boxscore')}. Needs robust handling.")
                                row_dict['home_away'] = "UNKNOWN"
                            processed_rows.append(row_dict)
                        if processed_rows:
                            player_box_pl_transformed = pl.DataFrame(processed_rows)
                            stats_records_added = await transform_player_stats_data(
                                db, player_box_pl_transformed, game_datetime_obj,
                                game_external_id, game_db.id, home_team_db, 
                                away_team_db, game_season_str
                            )
                            if stats_records_added: processed_stats_count += len(stats_records_added)
                            
                            home_team_total_minutes_for_game = 0.0
                            away_team_total_minutes_for_game = 0.0
                            stmt_player_stats_for_minutes = select(db_models.PlayerStat).where(db_models.PlayerStat.game_id == game_db.id)
                            res_player_stats_for_minutes = await db.execute(stmt_player_stats_for_minutes)
                            
                            for ps_record in res_player_stats_for_minutes.scalars().all():
                                if ps_record.is_home_team and ps_record.minutes_played is not None:
                                    home_team_total_minutes_for_game += ps_record.minutes_played
                                elif not ps_record.is_home_team and ps_record.minutes_played is not None:
                                    away_team_total_minutes_for_game += ps_record.minutes_played
                            
                            game_db.home_team_minutes_played = home_team_total_minutes_for_game
                            game_db.away_team_minutes_played = away_team_total_minutes_for_game
                            logger.info(f"Game {game_db.id}: Summed player minutes. Home: {home_team_total_minutes_for_game}, Away: {away_team_total_minutes_for_game}. To be committed.")
                else:
                    logger.warning(f"No player stats DataFrames could be concatenated from player_stats_source for game {g_id}.")

                if pbp_raw_data and 'boxscore' in pbp_raw_data and 'teams' in pbp_raw_data['boxscore']:
                    team_box_scores = pbp_raw_data['boxscore']['teams']
                    game_stats_to_update_from_box = {} 

                    for team_data in team_box_scores:
                        is_home_from_box = team_data.get('team', {}).get('id') == home_team_api_id
                        prefix = "home_team_" if is_home_from_box else "away_team_"

                        stats_list = team_data.get('statistics', [])
                        team_stats_dict = {stat['name']: stat['displayValue'] for stat in stats_list if 'name' in stat and 'displayValue' in stat}

                        game_stats_to_update_from_box[f'{prefix}field_goals_made'] = safe_int(team_stats_dict.get('fieldGoalsMade-fieldGoalsAttempted', '').split('-')[0])
                        game_stats_to_update_from_box[f'{prefix}field_goals_attempted'] = safe_int(team_stats_dict.get('fieldGoalsMade-fieldGoalsAttempted', '').split('-')[-1])
                        game_stats_to_update_from_box[f'{prefix}three_pointers_made'] = safe_int(team_stats_dict.get('threePointFieldGoalsMade-threePointFieldGoalsAttempted', '').split('-')[0])
                        game_stats_to_update_from_box[f'{prefix}three_pointers_attempted'] = safe_int(team_stats_dict.get('threePointFieldGoalsMade-threePointFieldGoalsAttempted', '').split('-')[-1])
                        game_stats_to_update_from_box[f'{prefix}free_throws_made'] = safe_int(team_stats_dict.get('freeThrowsMade-freeThrowsAttempted', '').split('-')[0])
                        game_stats_to_update_from_box[f'{prefix}free_throws_attempted'] = safe_int(team_stats_dict.get('freeThrowsMade-freeThrowsAttempted', '').split('-')[-1])
                        game_stats_to_update_from_box[f'{prefix}offensive_rebounds'] = safe_int(team_stats_dict.get('offensiveRebounds'))
                        game_stats_to_update_from_box[f'{prefix}defensive_rebounds'] = safe_int(team_stats_dict.get('defensiveRebounds'))
                        game_stats_to_update_from_box[f'{prefix}total_rebounds'] = safe_int(team_stats_dict.get('totalRebounds'))
                        game_stats_to_update_from_box[f'{prefix}assists'] = safe_int(team_stats_dict.get('assists'))
                        game_stats_to_update_from_box[f'{prefix}steals'] = safe_int(team_stats_dict.get('steals'))
                        game_stats_to_update_from_box[f'{prefix}blocks'] = safe_int(team_stats_dict.get('blocks'))
                        game_stats_to_update_from_box[f'{prefix}turnovers'] = safe_int(team_stats_dict.get('totalTurnovers'))
                        game_stats_to_update_from_box[f'{prefix}fouls'] = safe_int(team_stats_dict.get('fouls', team_stats_dict.get('totalFouls')))

                        current_score_in_box = safe_int(team_data.get('score'))
                        if is_home_from_box and current_score_in_box is not None:
                            game_db.home_score = current_score_in_box 
                        elif not is_home_from_box and current_score_in_box is not None:
                            game_db.away_score = current_score_in_box

                        team_fga = game_stats_to_update_from_box.get(f'{prefix}field_goals_attempted')
                        team_orb = game_stats_to_update_from_box.get(f'{prefix}offensive_rebounds')
                        team_tov = game_stats_to_update_from_box.get(f'{prefix}turnovers')
                        team_fta = game_stats_to_update_from_box.get(f'{prefix}free_throws_attempted')

                        logger.info(f"Game {g_id}, Team {'Home' if is_home_from_box else 'Away'} ({team_data.get('team', {}).get('abbreviation', 'N/A')} - API ID: {team_data.get('team', {}).get('id')}): Raw possession components: FGA={team_fga}, ORB={team_orb}, TOV={team_tov}, FTA={team_fta}")

                        team_possessions = None
                        primary_components_present = all(comp is not None for comp in [team_fga, team_orb, team_tov, team_fta])

                        if primary_components_present:
                            team_possessions = team_fga - team_orb + team_tov + (0.44 * team_fta)
                            logger.info(f"Game {g_id}, Team {'Home' if is_home_from_box else 'Away'}: Calculated possessions (primary): {team_possessions}")
                        else:
                            logger.warning(f"Game {g_id}, Team {'Home' if is_home_from_box else 'Away'}: Missing one or more primary components for possession calculation. Attempting fallback.")
                            if team_fga is not None: 
                                fb_fga = team_fga
                                fb_orb = team_orb or 0
                                fb_tov = team_tov or 0
                                fb_fta = team_fta or 0
                                team_possessions = fb_fga - fb_orb + fb_tov + (0.44 * fb_fta)
                                logger.info(f"Game {g_id}, Team {'Home' if is_home_from_box else 'Away'}: Calculated possessions (fallback): {team_possessions}. Original components: FGA={team_fga}, ORB={team_orb}, TOV={team_tov}, FTA={team_fta}")
                            else:
                                logger.error(f"Game {g_id}, Team {'Home' if is_home_from_box else 'Away'}: FGA is missing, cannot calculate possessions even with fallback. Setting to None.")
                                team_possessions = None
                        
                        game_stats_to_update_from_box[f'{prefix}possessions'] = team_possessions
                    
                    if game_stats_to_update_from_box:
                        for key, value in game_stats_to_update_from_box.items():
                            setattr(game_db, key, value) 
                        logger.info(f"Game {game_db.id}: Updated with team aggregate stats from boxscore data. To be committed.")
                else:
                    logger.info(f"pbp_raw_data['boxscore']['teams'] is missing for game {g_id}. Cannot extract team aggregates from boxscore.")

                try:
                    await db.commit() 
                    await db.refresh(game_db)
                    logger.info(f"Successfully finalized updates for game {game_db.id} (aggregates and summed minutes).")
                except Exception as e_final_game_commit:
                    await db.rollback()
                    logger.error(f"Error during final commit for game {game_db.id}: {e_final_game_commit}")

                home_team_pbp_fga = 0
                home_team_pbp_fta = 0
                home_team_pbp_orb = 0
                home_team_pbp_tov = 0
                away_team_pbp_fga = 0
                away_team_pbp_fta = 0
                away_team_pbp_orb = 0
                away_team_pbp_tov = 0
                pbp_aggregation_successful = False

                if pbp_raw_data and 'plays' in pbp_raw_data and isinstance(pbp_raw_data['plays'], list):
                    logger.info(f"Game {g_id}: Attempting to aggregate FGA, FTA, ORB, TOV from PBP data.")
                    for play in pbp_raw_data['plays']:
                        if not isinstance(play, dict):
                            logger.warning(f"Game {g_id}: Skipping non-dict item in PBP plays list.")
                            continue

                        play_team_id = str(play.get('team_id')) if pd.notna(play.get('team_id')) else None
                        type_text = str(play.get('type_text', '')).lower()
                        text = str(play.get('text', '')).lower() 
                        shooting_play_flag = play.get('shooting_play', False)

                        is_home_play = play_team_id == home_team_api_id
                        is_away_play = play_team_id == away_team_api_id

                        if shooting_play_flag and "free throw" not in type_text and "free throw" not in text:
                            if "shot" in type_text or "field goal" in type_text or "jumper" in type_text or "layup" in type_text or "dunk" in type_text or \
                               "shot" in text or "jumper" in text or "layup" in text or "dunk" in text: 
                                if is_home_play: home_team_pbp_fga += 1
                                elif is_away_play: away_team_pbp_fga += 1
                        
                        if "free throw" in type_text or "free throw" in text:
                            if "technical" not in type_text and "technical" not in text:
                                if is_home_play: home_team_pbp_fta += 1
                                elif is_away_play: away_team_pbp_fta += 1

                        if "rebound" in type_text or "rebound" in text:
                            if "offensive" in type_text or "offensive" in text:
                                if is_home_play: home_team_pbp_orb += 1
                                elif is_away_play: away_team_pbp_orb += 1
                        
                        if "turnover" in type_text or "turnover" in text:
                            if is_home_play: home_team_pbp_tov += 1
                            elif is_away_play: away_team_pbp_tov += 1
                    
                    logger.info(f"Game {g_id}: PBP Aggregation Results: Home (FGA:{home_team_pbp_fga}, FTA:{home_team_pbp_fta}, ORB:{home_team_pbp_orb}, TOV:{home_team_pbp_tov}), Away (FGA:{away_team_pbp_fga}, FTA:{away_team_pbp_fta}, ORB:{away_team_pbp_orb}, TOV:{away_team_pbp_tov})")
                    if home_team_pbp_fga > 0 or away_team_pbp_fga > 0: 
                        pbp_aggregation_successful = True
                else:
                    logger.warning(f"Game {g_id}: PBP aggregation did not yield FGA for either team. Will fall back to boxscore if available.")

                game_stats_to_update = {} 

                home_fga_source, home_orb_source, home_tov_source, home_fta_source = None, None, None, None
                away_fga_source, away_orb_source, away_tov_source, away_fta_source = None, None, None, None

                if pbp_aggregation_successful:
                    logger.info(f"Game {g_id}: Using PBP aggregated stats for possession calculation.")
                    home_fga_source, home_orb_source, home_tov_source, home_fta_source = home_team_pbp_fga, home_team_pbp_orb, home_team_pbp_tov, home_team_pbp_fta
                    away_fga_source, away_orb_source, away_tov_source, away_fta_source = away_team_pbp_fga, away_team_pbp_orb, away_team_pbp_tov, away_team_pbp_fta
                
                if not pbp_aggregation_successful or not all(v is not None for v in [home_fga_source, home_orb_source, home_tov_source, home_fta_source, away_fga_source, away_orb_source, away_tov_source, away_fta_source]):
                    logger.info(f"Game {g_id}: PBP aggregation incomplete or failed. Using boxscore data from pbp_raw_data['boxscore']['teams'] for team aggregates and possession components.")
                    if pbp_raw_data and 'boxscore' in pbp_raw_data and 'teams' in pbp_raw_data['boxscore']:
                        team_box_scores_data = pbp_raw_data['boxscore']['teams']
                        
                        for team_data_item in team_box_scores_data:
                            is_home_team_box = team_data_item.get('team', {}).get('id') == home_team_api_id
                            prefix_box = "home_team_" if is_home_team_box else "away_team_"
                            
                            stats_list_box = team_data_item.get('statistics', [])
                            team_stats_dict_box = {stat['name']: stat['displayValue'] for stat in stats_list_box if 'name' in stat and 'displayValue' in stat}

                            game_stats_to_update[f'{prefix_box}field_goals_made'] = safe_int(team_stats_dict_box.get('fieldGoalsMade-fieldGoalsAttempted', '').split('-')[0])
                            current_fga_box = safe_int(team_stats_dict_box.get('fieldGoalsMade-fieldGoalsAttempted', '').split('-')[-1])
                            game_stats_to_update[f'{prefix_box}field_goals_attempted'] = current_fga_box
                            game_stats_to_update[f'{prefix_box}three_pointers_made'] = safe_int(team_stats_dict_box.get('threePointFieldGoalsMade-threePointFieldGoalsAttempted', '').split('-')[0])
                            game_stats_to_update[f'{prefix_box}three_pointers_attempted'] = safe_int(team_stats_dict_box.get('threePointFieldGoalsMade-threePointFieldGoalsAttempted', '').split('-')[-1])
                            game_stats_to_update[f'{prefix_box}free_throws_made'] = safe_int(team_stats_dict_box.get('freeThrowsMade-freeThrowsAttempted', '').split('-')[0])
                            current_fta_box = safe_int(team_stats_dict_box.get('freeThrowsMade-freeThrowsAttempted', '').split('-')[-1])
                            game_stats_to_update[f'{prefix_box}free_throws_attempted'] = current_fta_box
                            current_orb_box = safe_int(team_stats_dict_box.get('offensiveRebounds'))
                            game_stats_to_update[f'{prefix_box}offensive_rebounds'] = current_orb_box
                            game_stats_to_update[f'{prefix_box}defensive_rebounds'] = safe_int(team_stats_dict_box.get('defensiveRebounds'))
                            game_stats_to_update[f'{prefix_box}total_rebounds'] = safe_int(team_stats_dict_box.get('totalRebounds'))
                            game_stats_to_update[f'{prefix_box}assists'] = safe_int(team_stats_dict_box.get('assists'))
                            game_stats_to_update[f'{prefix_box}steals'] = safe_int(team_stats_dict_box.get('steals'))
                            game_stats_to_update[f'{prefix_box}blocks'] = safe_int(team_stats_dict_box.get('blocks'))
                            current_tov_box = safe_int(team_stats_dict_box.get('totalTurnovers'))
                            game_stats_to_update[f'{prefix_box}turnovers'] = current_tov_box
                            game_stats_to_update[f'{prefix_box}fouls'] = safe_int(team_stats_dict_box.get('fouls', team_stats_dict_box.get('totalFouls')))
                            
                            current_score_box = safe_int(team_data_item.get('score'))
                            if is_home_team_box and current_score_box is not None: game_stats_to_update['home_score'] = current_score_box
                            elif not is_home_team_box and current_score_box is not None: game_stats_to_update['away_score'] = current_score_box

                            if not pbp_aggregation_successful:
                                if is_home_team_box:
                                    home_fga_source, home_orb_source, home_tov_source, home_fta_source = current_fga_box, current_orb_box, current_tov_box, current_fta_box
                                else:
                                    away_fga_source, away_orb_source, away_tov_source, away_fta_source = current_fga_box, current_orb_box, current_tov_box, current_fta_box
                        
                        if not pbp_aggregation_successful:
                            logger.info(f"Game {g_id}, Home Team (Boxscore for Poss): FGA={home_fga_source}, ORB={home_orb_source}, TOV={home_tov_source}, FTA={home_fta_source}")
                            logger.info(f"Game {g_id}, Away Team (Boxscore for Poss): FGA={away_fga_source}, ORB={away_orb_source}, TOV={away_tov_source}, FTA={away_fta_source}")
                    else: 
                        logger.error(f"Game {g_id}: PBP aggregation failed AND pbp_raw_data['boxscore']['teams'] is missing. Cannot calculate team possessions or reliably update other team aggregates.")
                
                home_possessions, away_possessions = None, None
                
                if all(comp is not None for comp in [home_fga_source, home_orb_source, home_tov_source, home_fta_source]):
                    home_possessions = home_fga_source - home_orb_source + home_tov_source + (0.44 * home_fta_source)
                    logger.info(f"Game {g_id}, Home Team: Calculated possessions (source: {'PBP' if pbp_aggregation_successful else 'Boxscore'}): {home_possessions}")
                else: 
                    logger.warning(f"Game {g_id}, Home Team: Missing components for possession calc even after PBP/Boxscore. FGA={home_fga_source}, ORB={home_orb_source}, TOV={home_tov_source}, FTA={home_fta_source}. Attempting final fallback.")
                    if home_fga_source is not None: 
                        fb_fga, fb_orb, fb_tov, fb_fta = home_fga_source, home_orb_source or 0, home_tov_source or 0, home_fta_source or 0
                        home_possessions = fb_fga - fb_orb + fb_tov + (0.44 * fb_fta)
                        logger.info(f"Game {g_id}, Home Team: Calculated possessions (final fallback): {home_possessions}. Using FGA={fb_fga}, ORB={fb_orb}, TOV={fb_tov}, FTA={fb_fta}")
                    else:
                        logger.error(f"Game {g_id}, Home Team: FGA is missing. Cannot calculate possessions. Setting to None.")
                game_stats_to_update['home_team_possessions'] = home_possessions

                if all(comp is not None for comp in [away_fga_source, away_orb_source, away_tov_source, away_fta_source]):
                    away_possessions = away_fga_source - away_orb_source + away_tov_source + (0.44 * away_fta_source)
                    logger.info(f"Game {g_id}, Away Team: Calculated possessions (source: {'PBP' if pbp_aggregation_successful else 'Boxscore'}): {away_possessions}")
                else:
                    logger.warning(f"Game {g_id}, Away Team: Missing components for possession calc. FGA={away_fga_source}, ORB={away_orb_source}, TOV={away_tov_source}, FTA={away_fta_source}. Attempting final fallback.")
                    if away_fga_source is not None:
                        fb_fga, fb_orb, fb_tov, fb_fta = away_fga_source, away_orb_source or 0, away_tov_source or 0, away_fta_source or 0
                        away_possessions = fb_fga - fb_orb + fb_tov + (0.44 * fb_fta)
                        logger.info(f"Game {g_id}, Away Team: Calculated possessions (final fallback): {away_possessions}. Using FGA={fb_fga}, ORB={fb_orb}, TOV={fb_tov}, FTA={fb_fta}")
                    else:
                        logger.error(f"Game {g_id}, Away Team: FGA is missing. Cannot calculate possessions. Setting to None.")
                game_stats_to_update['away_team_possessions'] = away_possessions

                if game_stats_to_update:
                    for key, value in game_stats_to_update.items():
                        if key in ['home_score', 'away_score'] and value is None:
                            if getattr(game_db, key) is not None: 
                                logger.info(f"Game {g_id}: Score for {key} is None in boxscore, retaining existing value: {getattr(game_db, key)}")
                                continue 
                        setattr(game_db, key, value)
                    logger.info(f"Game {game_db.id}: Updated with team aggregate/possession stats. Source for possession components: {'PBP' if pbp_aggregation_successful and home_possessions is not None and away_possessions is not None else 'Boxscore/Fallback'}. To be committed.")
                
                try:
                    await db.commit()
                    await db.refresh(game_db)
                    logger.info(f"Game {game_db.id}: Successfully committed game updates including possessions.")
                except Exception as e_game_update_commit:
                    await db.rollback()
                    logger.error(f"Game {game_db.id}: Error committing game updates with possessions: {e_game_update_commit}")

                if pbp_raw_data and 'plays' in pbp_raw_data:
                    pbp_plays_list = pbp_raw_data['plays'] 
                    
                    pbp_plays_df_final = None
                    if isinstance(pbp_plays_list, list):
                        if len(pbp_plays_list) > 0 and isinstance(pbp_plays_list[0], dict):
                            try:
                                pbp_plays_df_final = pd.DataFrame(pbp_plays_list)
                            except Exception as e_df_pbp:
                                logger.error(f"Error converting PBP list of dicts to DataFrame for game {g_id}: {e_df_pbp}")
                        elif len(pbp_plays_list) == 0:
                            logger.info(f"PBP plays list is empty for game {g_id}.")
                        else:
                            if len(pbp_plays_list) > 0 and isinstance(pbp_plays_list[0], pd.DataFrame):
                                 try:
                                     pbp_plays_df_final = pd.concat(pbp_plays_list, ignore_index=True)
                                 except Exception as e_concat_pbp:
                                     logger.error(f"Error concatenating list of PBP DataFrames for game {g_id}: {e_concat_pbp}")
                            else:
                                logger.error(f"PBP plays list for game {g_id} is not a list of dicts or DataFrames. First element type: {type(pbp_plays_list[0]) if pbp_plays_list else 'Empty List'}")
                    elif isinstance(pbp_plays_list, pd.DataFrame):
                         pbp_plays_df_final = pbp_plays_list
                    elif hasattr(pbp_plays_list, 'to_pandas'): 
                        pbp_plays_df_final = pbp_plays_list.to_pandas()
                    else:
                         logger.error(f"PBP data ('plays' key) for game {g_id} is not a list or recognized DataFrame. Type: {type(pbp_plays_list)}. Skipping PBP processing.")

                    if pbp_plays_df_final is not None and not pbp_plays_df_final.empty:
                        logger.info(f"Successfully converted/retrieved {len(pbp_plays_df_final)} PBP events for game {g_id}. Columns: {pbp_plays_df_final.columns.tolist()[:10]}...")
                        await transform_and_store_pbp_data(db, game_db.id, pbp_plays_df_final)
                    else:
                        logger.warning(f"Final PBP DataFrame ('plays' key) is empty or None for game {g_id}.")
                else:
                    logger.warning(f"PBP data ('plays' key) missing in pbp_raw_data for game {g_id}.")
                
                game_count += 1
                if game_count % 20 == 0: logger.info(f"Processed {game_count} games for year {year}...")
            
            # This except block is for errors within a single game's processing
            except Exception as e_game_processing: 
                logger.error(f"Major error processing a game (ID: {g_id if 'g_id' in locals() else 'Unknown'}) for year {year}: {e_game_processing}", exc_info=True)
                # Continue to the next game in the loop
                continue 
    
    except Exception as e_year_processing:
        logger.error(f"An error occurred during WNBA stats loading for year {year}: {e_year_processing}", exc_info=True)
    
    logger.info(f"Finished WNBA stats loading for year {year}. Attempted to process {actually_processed_game_count} games and added {processed_stats_count} player stat records.")
    return actually_processed_game_count

async def main(seasons: Optional[List[int]] = None, max_games_per_season: Optional[int] = None):
    if seasons is None:
        current_year = datetime.now().year
        seasons = [current_year, current_year - 1] 

    logger.info(f"Starting main WNBA scraper for seasons: {seasons}")
    async with SessionLocal() as db_session: 
        for year in seasons:
            logger.info(f"--- Processing season: {year} ---")
            await load_wnba_stats(year, db_session, max_games_to_process=max_games_per_season)
            logger.info(f"--- Finished season: {year} ---")
    logger.info("WNBA Scraper finished all specified seasons.")

if __name__ == "__main__":
    input_seasons = []
    max_games_arg = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--max_games="):
                try:
                    max_games_arg = int(arg.split("=")[1])
                except ValueError:
                    logger.error(f"Invalid value for --max_games: {arg.split('=')[1]}. Must be an integer.")
                    sys.exit(1)
            else:
                try:
                    input_seasons.append(int(arg))
                except ValueError:
                    logger.error(f"Invalid season year: {arg}. Must be an integer.")
                    sys.exit(1)
    
    if not input_seasons: 
        current_year = datetime.now().year
        input_seasons = [current_year, current_year -1]

    asyncio.run(main(seasons=input_seasons, max_games_per_season=max_games_arg))