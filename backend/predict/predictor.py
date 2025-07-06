import argparse
import asyncio
import logging
from datetime import datetime, timezone
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple
import uuid
import traceback
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import delete, select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

# Make sure to add project root to sys.path if running as a script
import sys
PROJECT_ROOT_FROM_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT_FROM_SCRIPT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_FROM_SCRIPT)


from backend.db.session import get_async_db_session as get_session
from backend.db import models as db_models
from backend.app.crud import predictions as crud_predictions, model_versions as crud_model_versions
from backend.schemas.prediction import PredictionCreate
from backend.features.feature_engineering_core import (
    generate_full_feature_set,
    get_team_defensive_rolling_averages,
    get_team_performance_rolling_averages,
)
from backend.app.core.config import (
    DEFAULT_PROP_MARKET_TO_STAT_MAP,
    ROLLING_WINDOWS
)
from backend.app.dependencies import get_db
from backend.app.core.config import DEFAULT_PROP_MARKET_TO_STAT_MAP
from backend.features.feature_engineering_core import generate_full_feature_set


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = PROJECT_ROOT


# --- Model Loading ---
async def load_model_artifacts(db: AsyncSession, model_name_prefix: str, model_type: str) -> Optional[Tuple[Any, Any, Any, Any]]:
    """Loads the latest model pipeline, ICP scores, and feature names from the database."""
    logger.info(f"Loading latest model artifacts for model type '{model_type}' with prefix: {model_name_prefix}")
    
    version_name_pattern = f"{model_name_prefix}_{model_type}"
    
    latest_version = await crud_model_versions.get_latest_model_version_by_prefix(db, prefix=version_name_pattern)
    
    if not latest_version:
        logger.warning(f"No model version found in DB matching pattern '{version_name_pattern}%'")
        return None, None, None, None

    logger.info(f"Found latest model version: {latest_version.version_name}")

    try:
        pipeline = joblib.load(latest_version.model_path)
        
        icp_scores_path = None
        if model_type == 'regression':
            icp_scores_path = latest_version.nonconformity_scores_path
        elif model_type == 'classification':
            icp_scores_path = latest_version.nonconformity_scores_clf_path
            
        icp_scores = joblib.load(icp_scores_path) if icp_scores_path and os.path.exists(icp_scores_path) else None

        # Load the clean feature names directly from the database record
        feature_names = latest_version.feature_names

        return pipeline, icp_scores, feature_names, latest_version.id
    except FileNotFoundError as e:
        logger.error(f"Error loading model artifacts for {latest_version.version_name}: {e}")
        return None, None, None, None
    except Exception as e:
        logger.error(f"A general error occurred loading artifacts for {latest_version.version_name}: {e}")
        return None, None, None, None


# --- ICP Prediction Logic ---
def get_regression_interval(
    point_prediction: float,
    nonconformity_scores: np.ndarray,
    confidence_level: float = 0.9,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Generates a conformal prediction interval for a regression model.
    """
    if point_prediction is None or nonconformity_scores is None:
        return None, None
        
    try:
        q_value = np.quantile(
            nonconformity_scores,
            min(1.0, confidence_level)
        )
        
        lower_bound = point_prediction - q_value
        upper_bound = point_prediction + q_value
        
        return max(0, lower_bound), upper_bound
    except Exception as e:
        logger.error(f"Error during regression interval calculation: {e}")
        logger.error(traceback.format_exc())
        return None, None


def get_classification_p_values(
    probas: np.ndarray,
    nonconformity_scores: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates calibrated p-values for Over/Under outcomes.
    """
    if probas is None or nonconformity_scores is None:
        return None, None
        
    try:
        # Assuming class 0 is Under, class 1 is Over
        p_under, p_over = probas[0], probas[1]

        alpha_under = 1 - p_under
        alpha_over = 1 - p_over
        
        p_value_under = (np.sum(nonconformity_scores >= alpha_under) + 1) / (len(nonconformity_scores) + 1)
        p_value_over = (np.sum(nonconformity_scores >= alpha_over) + 1) / (len(nonconformity_scores) + 1)
        
        return p_value_under, p_value_over
    except Exception as e:
        logger.error(f"Error during classification p-value calculation: {e}")
        logger.error(traceback.format_exc())
        return None, None


async def make_predictions_for_props(db: AsyncSession, predictions_for_csv: List[Dict[str, Any]]):
    logger.info("Fetching upcoming player props for games on or after today...")
    
    today = datetime.now(timezone.utc).date()
    
    stmt = (
        select(db_models.PlayerProp)
        .options(
                    joinedload(db_models.PlayerProp.game)
                    .joinedload(db_models.Game.home_team_ref),
                    joinedload(db_models.PlayerProp.game)
                    .joinedload(db_models.Game.away_team_ref),
                    joinedload(db_models.PlayerProp.player),
                    joinedload(db_models.PlayerProp.market)
        )
        .join(db_models.Game, db_models.PlayerProp.game_id == db_models.Game.id)
        .where(func.date(db_models.Game.game_datetime) >= today)
    )
    result = await db.execute(stmt)
    props_to_predict = result.scalars().unique().all()
    
    logger.info(f"Found {len(props_to_predict)} upcoming player props.")

    if not props_to_predict:
        return

    all_predictions_to_create = []

    # Cache historical stats per game date to avoid re-fetching
    historical_stats_cache = {}

    for prop in props_to_predict:
        try:
            game_date = prop.game.game_datetime.date()
            home_team_id = prop.game.home_team_id
            away_team_id = prop.game.away_team_id
            
            # --- NEW: Find the scraper-generated game_id ---
            game_day = prop.game.game_datetime.date()
            
            stmt = select(db_models.Game).where(
                and_(
                    func.date(db_models.Game.game_datetime) == game_day,
                    db_models.Game.home_team_id == home_team_id,
                    db_models.Game.away_team_id == away_team_id,
                    db_models.Game.the_odds_api_game_id.isnot(None)
                )
            )
            result = await db.execute(stmt)
            scraper_game = result.scalars().first()

            if not scraper_game:
                logger.warning(f"Could not find a scraper-generated game match for prop's game {prop.game_id} on {game_day}. Skipping prop.")
                continue

            scraper_game_id = scraper_game.id
            logger.info(f"Matched prop's game {prop.game_id} to scraper game {scraper_game_id}")
            # --- END NEW ---

            logger.info(f"Processing prop for game {prop.game.away_team_ref.team_name} @ {prop.game.home_team_ref.team_name} ({scraper_game_id})")

            if game_date not in historical_stats_cache:
                try:
                    logger.info(f"Fetching and caching historical stats for game date {game_date}...")
                    all_historical_stats = await fetch_all_historical_stats(db, game_date)
                    if all_historical_stats.empty:
                        logger.warning(f"No historical stats found before {game_date}. Team/opponent features may be NaN.")
                    historical_stats_cache[game_date] = all_historical_stats
                except Exception as e:
                    logger.error(f"Failed to load historical data for game date {game_date}: {e}")
                    historical_stats_cache[game_date] = pd.DataFrame() # Cache empty df to avoid retries
                    continue
            
            all_historical_stats = historical_stats_cache[game_date]
            
            prediction_to_create = None
            try:
                model_name_prefix = DEFAULT_PROP_MARKET_TO_STAT_MAP.get(prop.market.key)
                if not model_name_prefix:
                    logger.warning(f"No stat mapping for market key '{prop.market.key}'. Skipping prop.")
                    continue

                player_id = prop.player_id
                player_name = prop.player.player_name
                logger.info(f"--- Processing prop: {player_name} - {model_name_prefix} (Line: {prop.line}) ---")
                
                player_game_log_df = await fetch_player_game_log(db, player_id, game_date)
                if player_game_log_df.empty:
                    logger.warning(f"No game log found for {player_name} ({player_id}). Skipping prop.")
                    continue

                current_game_df = pd.DataFrame([{
                    'game_id': scraper_game_id, # Use the CORRECT game ID
                    'player_id': player_id,
                    'team_id': prop.player.team_id,
                    'opponent_team_id': home_team_id if prop.player.team_id == away_team_id else away_team_id,
                    'game_datetime': prop.game.game_datetime,
                    'is_home_game': 1 if prop.player.team_id == home_team_id else 0,
                }])
                
                base_df = pd.concat([player_game_log_df, current_game_df], ignore_index=True)

                logger.info(f"Generating features for {player_name} for stat '{model_name_prefix}'...")
                final_features_df = generate_full_feature_set(
                    base_df=base_df,
                    target_stat=model_name_prefix,
                    team_context_df=all_historical_stats,
                    rolling_windows=ROLLING_WINDOWS
                )
                
                if final_features_df.empty:
                    logger.warning(f"Feature generation for {player_name} resulted in an empty DataFrame. Skipping.")
                    continue

                current_features = final_features_df.iloc[[-1]]
                
                # --- Make Predictions using Regression Model ---
                (
                    reg_pipeline, 
                    reg_icp_scores, 
                    reg_feature_names, 
                    reg_model_version_id
                ) = await load_model_artifacts(db, model_name_prefix, "regression")

                predicted_value = None
                lower_bound, upper_bound = None, None

                if reg_pipeline and reg_feature_names:
                    current_features_aligned_reg = current_features.reindex(columns=reg_feature_names, fill_value=0)
                    
                    try:
                        predicted_value = reg_pipeline.predict(current_features_aligned_reg)[0]
                        logger.info(f"Regression Prediction: {predicted_value:.2f}")
                        
                        if reg_icp_scores is not None:
                            lower_bound, upper_bound = get_regression_interval(predicted_value, reg_icp_scores)
                            logger.info(f"Conformal Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")

                    except Exception as e:
                        logger.error(f"Error during regression prediction for {player_name}: {e}")
                        logger.error(traceback.format_exc())

                # --- Make Predictions using Classification Model ---
                (
                    clf_pipeline, 
                    clf_icp_scores, 
                    clf_feature_names, 
                    clf_model_version_id
                ) = await load_model_artifacts(db, model_name_prefix, "classification")
                
                p_value_over, p_value_under = None, None
                over_under_prediction = None
                probas = None
                
                if clf_pipeline and clf_feature_names:
                    current_features_clf = current_features.copy()
                    current_features_clf['prop_line'] = prop.line
                    
                    current_features_aligned_clf = current_features_clf.reindex(columns=clf_feature_names, fill_value=0)
                    
                    try:
                        probas = clf_pipeline.predict_proba(current_features_aligned_clf)[0]
                        over_under_prediction = "OVER" if probas[1] > 0.5 else "UNDER"
                        logger.info(f"Classification Prediction: {over_under_prediction} (Probs: {probas})")
                        
                        if clf_icp_scores is not None:
                            p_value_under, p_value_over = get_classification_p_values(probas, clf_icp_scores)
                            logger.info(f"P-Values -> Over: {p_value_over:.3f}, Under: {p_value_under:.3f}")

                    except Exception as e:
                        logger.error(f"Error during classification prediction for {player_name}: {e}")
                        logger.error(traceback.format_exc())
                
                if reg_model_version_id or clf_model_version_id:
                    prediction_to_create = PredictionCreate(
                        player_prop_id=prop.id,
                        regression_model_version_id=reg_model_version_id,
                        classification_model_version_id=clf_model_version_id,
                        predicted_value=predicted_value,
                        prediction_time=datetime.now(timezone.utc),
                        confidence_interval_lower=lower_bound,
                        confidence_interval_upper=upper_bound,
                        p_value_over=p_value_over,
                        p_value_under=p_value_under,
                        over_under_prediction=over_under_prediction
                    )
                    all_predictions_to_create.append(prediction_to_create)

                    csv_row = {
                        "Player": player_name,
                        "Game": f"{prop.game.away_team_ref.team_name} @ {prop.game.home_team_ref.team_name}",
                        "Market": prop.market.key,
                        "Line": prop.line,
                        "Over Prob.": f"{probas[1]:.3f}" if probas is not None else "N/A",
                        "Under Prob.": f"{probas[0]:.3f}" if probas is not None else "N/A",
                        "ICP Interval": f"[{lower_bound:.2f}, {upper_bound:.2f}]" if lower_bound is not None and upper_bound is not None else "N/A",
                        "Calibrated Over": f"{p_value_over:.3f}" if p_value_over is not None else "N/A",
                        "Calibrated Under": f"{p_value_under:.3f}" if p_value_under is not None else "N/A",
                        "ICP Set": "N/A"
                    }
                    predictions_for_csv.append(csv_row)

            except Exception as e:
                logger.error(f"Failed to process prop for {prop.player.player_name}: {e}")
                logger.error(traceback.format_exc())

        except Exception as e:
            logger.error(f"An unexpected error occurred processing prop {prop.id}: {e}")
            logger.error(traceback.format_exc())

    if all_predictions_to_create:
        logger.info(f"Saving {len(all_predictions_to_create)} new predictions to the database...")
        await crud_predictions.create_predictions_bulk(db, all_predictions_to_create)
        logger.info("Successfully saved predictions.")

async def save_predictions_to_csv(predictions_data: List[Dict[str, Any]], file_path: str):
    """Saves a list of prediction data to a CSV file."""
    if not predictions_data:
        logger.info("No prediction data to save to CSV.")
        return

    df = pd.DataFrame(predictions_data)
    
    columns = [
        "Player", "Game", "Market", "Line", "Over Prob.", "Under Prob.", 
        "ICP Interval", "Calibrated Over", "Calibrated Under", "ICP Set"
    ]
    df = df[columns]

    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved predictions to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions to CSV: {e}")


async def fetch_all_historical_stats(db: AsyncSession, until_date: datetime.date) -> pd.DataFrame:
    """
    Fetches all player stats for all games up to a certain date.
    This is used to generate team-level rolling averages.
    """
    logger.info(f"Fetching all historical stats for all teams before {until_date}...")
    
    stmt = (
        select(
            db_models.PlayerStat
        )
        .join(db_models.Game, db_models.PlayerStat.game_id == db_models.Game.id)
        .where(func.date(db_models.Game.game_datetime) < until_date)
        .options(joinedload(db_models.PlayerStat.game))
    )
    result = await db.execute(stmt)
    stats = result.scalars().all()

    if not stats:
        return pd.DataFrame()

    data = [
        {
            'game_id': stat.game_id,
            'player_id': stat.player_id,
            'team_id': stat.team_id,
            'home_team_id': stat.game.home_team_id,
            'away_team_id': stat.game.away_team_id,
            'game_datetime': stat.game.game_datetime,
            'points': stat.points,
            'rebounds': stat.rebounds,
            'assists': stat.assists,
            'steals': stat.steals,
            'blocks': stat.blocks,
            'turnovers': stat.turnovers,
            'field_goals_made': stat.field_goals_made,
            'field_goals_attempted': stat.field_goals_attempted,
            'three_pointers_made': stat.three_pointers_made,
            'three_pointers_attempted': stat.three_pointers_attempted,
            'free_throws_made': stat.free_throws_made,
            'free_throws_attempted': stat.free_throws_attempted,
        } for stat in stats
    ]
    df = pd.DataFrame(data)

    df['opponent_team_id'] = df.apply(
        lambda row: row['away_team_id'] if str(row['team_id']) == str(row['home_team_id']) else row['home_team_id'],
        axis=1
    )

    base_cols = ['points', 'rebounds', 'assists', 'steals', 'blocks']
    for col in base_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['pra'] = df['points'] + df['rebounds'] + df['assists']
    df['pr'] = df['points'] + df['rebounds']
    df['pa'] = df['points'] + df['assists']
    df['ra'] = df['rebounds'] + df['assists']
    df['blocks_plus_steals'] = df['blocks'] + df['steals']
    
    return df


async def fetch_player_game_log(db: AsyncSession, player_id: uuid.UUID, until_date: datetime.date) -> pd.DataFrame:
    """Fetches the game log for a specific player up to a certain date."""
    logger.info(f"Fetching game log for player {player_id} before {until_date}...")
    
    stmt = (
        select(db_models.PlayerStat)
        .join(db_models.Game, db_models.PlayerStat.game_id == db_models.Game.id)
        .where(db_models.PlayerStat.player_id == player_id)
        .where(func.date(db_models.Game.game_datetime) < until_date)
        .options(
            joinedload(db_models.PlayerStat.game).joinedload(db_models.Game.home_team_ref),
            joinedload(db_models.PlayerStat.game).joinedload(db_models.Game.away_team_ref)
        )
        .order_by(db_models.Game.game_datetime.desc())
    )
    result = await db.execute(stmt)
    game_logs = result.scalars().all()

    if not game_logs:
        return pd.DataFrame()
    
    data = [
        {
            'game_datetime': log.game.game_datetime,
            'team_id': log.team_id,
            'opponent_team_id': log.game.away_team_id if log.team_id == log.game.home_team_id else log.game.home_team_id,
            'is_home_game': int(log.team_id == log.game.home_team_id),
            'minutes_played': log.minutes_played,
            'points': log.points,
            'rebounds': log.rebounds,
            'assists': log.assists,
            'steals': log.steals,
            'blocks': log.blocks,
            'turnovers': log.turnovers,
            'field_goals_made': log.field_goals_made,
            'field_goals_attempted': log.field_goals_attempted,
            'three_pointers_made': log.three_pointers_made,
            'three_pointers_attempted': log.three_pointers_attempted,
            'free_throws_made': log.free_throws_made,
            'free_throws_attempted': log.free_throws_attempted,
            'offensive_rebounds': log.offensive_rebounds,
            'defensive_rebounds': log.defensive_rebounds,
            'fouls': log.fouls,
            'player_efficiency_rating': log.player_efficiency_rating,
            'usage_rate': log.usage_rate,
            'true_shooting_percentage': log.true_shooting_percentage,
            'effective_field_goal_percentage': log.effective_field_goal_percentage
        } for log in game_logs
    ]
    df = pd.DataFrame(data)

    base_cols = ['points', 'rebounds', 'assists', 'steals', 'blocks']
    for col in base_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['pra'] = df['points'] + df['rebounds'] + df['assists']
    df['pr'] = df['points'] + df['rebounds']
    df['pa'] = df['points'] + df['assists']
    df['ra'] = df['rebounds'] + df['assists']
    df['blocks_plus_steals'] = df['blocks'] + df['steals']
    
    return df


async def clear_old_predictions(db: AsyncSession):
    """Clears all existing entries from the 'predictions' table."""
    logger.info("Clearing old predictions from the database...")
    try:
        stmt = delete(db_models.Prediction)
        await db.execute(stmt)
        await db.commit()
        logger.info("Successfully cleared old predictions.")
    except Exception as e:
        logger.error(f"Error clearing old predictions: {e}")
        await db.rollback()

async def main():
    parser = argparse.ArgumentParser(description="Generate predictions for upcoming WNBA games.")
    parser.add_argument("--clear", action="store_true", help="Clear all existing predictions before generating new ones.")
    parser.add_argument("--output-csv", type=str, default="predictions.csv", help="Path to save the predictions CSV file.")
    args = parser.parse_args()

    predictions_for_csv = []

    async with get_session() as db:
        if args.clear:
            await clear_old_predictions(db)

        await make_predictions_for_props(db, predictions_for_csv)
        
        if predictions_for_csv:
            await save_predictions_to_csv(predictions_for_csv, args.output_csv)


if __name__ == "__main__":
    asyncio.run(main())