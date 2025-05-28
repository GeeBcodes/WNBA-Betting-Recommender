import sys
import os
import argparse
import logging
from pathlib import Path
import datetime
from typing import Optional, List, Dict, Any, Tuple
import re
import uuid

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import joblib
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, desc

from backend.db.session import SessionLocal
from backend.db import models as db_models
from backend.app.crud import predictions as crud_predictions
from backend.schemas import prediction as prediction_schema
# Placeholder for feature engineering logic, to be imported or defined
# from backend.models.train_model import feature_engineering, get_preprocessor # Or copy relevant parts

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_ARTIFACTS_DIR = Path(PROJECT_ROOT) / "backend" / "models" / "artifacts"

def get_db_session() -> Session:
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        logger.error(f"Error creating database session: {e}")
        if db:
            db.close()
        raise

def get_market_key_to_target_stat_map() -> Dict[str, str]:
    """Returns a mapping from market_key prefixes to target_stat names."""
    # This can be expanded as more player prop markets are handled
    return {
        "player_points": "points",
        "player_rebounds": "rebounds",
        "player_assists": "assists",
        "player_steals": "steals",
        "player_blocks": "blocks",
        "player_turnovers": "turnovers",
        "player_threes": "three_pointers_made", # Assuming model trained on three_pointers_made
        # Add mappings for other stats like PRA, P+R, P+A, R+A if models are trained for them
        # e.g., "player_pra": "pra" (if 'pra' is a column in PlayerStat or a target for a model)
    }

def map_market_key_to_target_stat(market_key: str) -> Optional[str]:
    """Maps a market key (e.g., player_points_over_under) to a target_stat (e.g., points)."""
    mapping = get_market_key_to_target_stat_map()
    for prefix, stat_name in mapping.items():
        if market_key.startswith(prefix):
            return stat_name
    logger.warning(f"No target_stat mapping found for market_key: {market_key}")
    return None

def get_upcoming_player_props(db: Session, game_date_cutoff: Optional[datetime.date] = None) -> List[db_models.PlayerProp]:
    """Fetches player props for upcoming games."""
    if game_date_cutoff is None:
        game_date_cutoff = datetime.date.today()

    logger.info(f"Fetching upcoming player props for games on or after {game_date_cutoff}...")
    
    query = (
        db.query(db_models.PlayerProp)
        .join(db_models.Game, db_models.PlayerProp.game_id == db_models.Game.id)
        .options(
            joinedload(db_models.PlayerProp.player),
            joinedload(db_models.PlayerProp.game),
            joinedload(db_models.PlayerProp.market),
            joinedload(db_models.PlayerProp.bookmaker)
        )
        .filter(db_models.Game.game_datetime >= datetime.datetime.combine(game_date_cutoff, datetime.time.min))
        .order_by(db_models.Game.game_datetime, db_models.PlayerProp.player_id)
    )
    
    player_props = query.all()
    logger.info(f"Found {len(player_props)} upcoming player props.")
    return player_props

def load_latest_model_for_target_stat(db: Session, target_stat: str) -> Optional[Tuple[Any, db_models.ModelVersion]]:
    logger.info(f"Attempting to load latest model for target_stat: {target_stat}")
    latest_model_version = (
        db.query(db_models.ModelVersion)
        .filter(db_models.ModelVersion.version_name.like(f"{target_stat}_model_v%"))
        .order_by(desc(db_models.ModelVersion.trained_at))
        .first()
    )
    if not latest_model_version:
        logger.warning(f"No model version found in DB for target_stat: {target_stat}")
        return None, None
    if not latest_model_version.model_path:
        logger.error(f"ModelVersion {latest_model_version.version_name} has no model_path defined.")
        return None, None
    model_full_path = Path(PROJECT_ROOT) / latest_model_version.model_path
    if not model_full_path.exists():
        logger.error(f"Model artifact not found at path: {model_full_path} for ModelVersion: {latest_model_version.version_name}")
        return None, None
    try:
        model_pipeline = joblib.load(model_full_path)
        logger.info(f"Successfully loaded model: {latest_model_version.version_name} from {model_full_path}")
        return model_pipeline, latest_model_version
    except Exception as e:
        logger.error(f"Error loading model artifact from {model_full_path}: {e}", exc_info=True)
        return None, None

# --- Start: Copied and adapted from backend.models.train_model ---
# (Original docstrings and some logging kept for context, may need minor adjustments)
def _copied_feature_engineering(df: pd.DataFrame, target_stat: str) -> pd.DataFrame:
    """
    Performs feature engineering. Creates lagged features, rolling averages, etc.
    (Adapted from train_model.py)
    """
    if df.empty:
        # In prediction context, this might mean no historical data, handle appropriately
        logger.warning("DataFrame is empty in _copied_feature_engineering.")
        return df
        
    logger.debug(f"Performing feature engineering for target stat: {target_stat} on df with shape {df.shape}")
    
    # Ensure game_datetime is datetime type for diff()
    df['game_datetime'] = pd.to_datetime(df['game_datetime'])
    df = df.sort_values(by=['player_id', 'game_datetime'])

    # Lagged features for the target stat
    for lag in [1, 2, 3]:
        df[f'{target_stat}_lag_{lag}'] = df.groupby('player_id', group_keys=False)[target_stat].shift(lag)
    
    # Rolling average for target_stat
    df[f'{target_stat}_roll_avg_3'] = df.groupby('player_id', group_keys=False)[target_stat].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    )
    
    # Rolling average for minutes_played
    if 'minutes_played' in df.columns:
        df[f'minutes_played_roll_avg_3'] = df.groupby('player_id', group_keys=False)['minutes_played'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
    else:
        # If minutes_played is not in historical data for some reason, fill roll_avg with NaN or 0
        df[f'minutes_played_roll_avg_3'] = np.nan 
        logger.warning("'minutes_played' column not found for rolling average calculation.")


    # Days since last game
    # fillna(7) is a default from training for the very first game of a player.
    # For prediction, if we have at least one historical game, this will be calculated.
    # If it's truly the player's first game ever (no history in hist_df), then it will be NaN
    # and the imputation later will handle it for the prediction row.
    df['days_since_last_game'] = df.groupby('player_id', group_keys=False)['game_datetime'].diff().dt.days
    
    logger.debug("Feature engineering complete in _copied_feature_engineering.")
    return df
# --- End: Copied and adapted from backend.models.train_model ---


def engineer_features_for_prediction(db: Session, player_id_uuid: uuid.UUID, game_id_uuid: uuid.UUID, game_datetime: datetime.datetime, target_stat: str) -> Optional[pd.DataFrame]:
    """
    Fetches historical data and engineers features for a single player for an upcoming game.
    Returns a DataFrame with a single row of features, ready for the model's preprocessor.
    """
    player_id_str = str(player_id_uuid)
    logger.info(f"Engineering features for player {player_id_str}, game {str(game_id_uuid)} ({game_datetime.strftime('%Y-%m-%d %H:%M')}), target: {target_stat}")
    
    lookback_limit = 10
    historical_stats_query = (
        db.query(
            db_models.PlayerStat.minutes_played,
            getattr(db_models.PlayerStat, target_stat).label(target_stat),
            db_models.Game.game_datetime,
            db_models.Player.id.label('player_id')
        )
        .join(db_models.Game, db_models.PlayerStat.game_id == db_models.Game.id)
        .join(db_models.Player, db_models.PlayerStat.player_id == db_models.Player.id)
        .filter(db_models.PlayerStat.player_id == player_id_uuid)
        .filter(db_models.Game.game_datetime < game_datetime)
        .order_by(db_models.Game.game_datetime.desc())
        .limit(lookback_limit)
    )
    hist_df = pd.read_sql_query(historical_stats_query.statement, db.bind)
    
    if not hist_df.empty:
        hist_df['player_id'] = hist_df['player_id'].astype(str)
        hist_df['game_datetime'] = pd.to_datetime(hist_df['game_datetime'])
        hist_df = hist_df.sort_values(by='game_datetime', ascending=True)
        logger.debug(f"Fetched {len(hist_df)} historical records for player {player_id_str}.")
    else:
        logger.info(f"No recent historical stats found for player {player_id_str} before {game_datetime}.")

    upcoming_game_db = db.query(db_models.Game).filter(db_models.Game.id == game_id_uuid).first()
    if not upcoming_game_db:
        logger.error(f"Could not find upcoming game with ID {str(game_id_uuid)} in DB.")
        return None

    current_game_data = {
        'player_id': player_id_str,
        'game_datetime': pd.to_datetime(game_datetime),
        target_stat: np.nan,
        'minutes_played': np.nan,
        'season': upcoming_game_db.season,
        'home_team': upcoming_game_db.home_team,
        'away_team': upcoming_game_db.away_team,
    }
    current_game_df_row = pd.DataFrame([current_game_data])

    if not hist_df.empty:
        for col in current_game_df_row.columns:
            if col not in hist_df.columns:
                hist_df[col] = np.nan
        combined_df = pd.concat([hist_df, current_game_df_row], ignore_index=True)
    else:
        combined_df = current_game_df_row
        
    combined_df['game_datetime'] = pd.to_datetime(combined_df['game_datetime'])
    combined_df = combined_df.sort_values(by=['player_id', 'game_datetime'])

    for lag in [1, 2, 3]:
        combined_df[f'{target_stat}_lag_{lag}'] = combined_df.groupby('player_id', group_keys=False)[target_stat].shift(lag)
    
    combined_df[f'{target_stat}_roll_avg_3'] = combined_df.groupby('player_id', group_keys=False)[target_stat].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    )
    
    if 'minutes_played' in combined_df.columns:
        combined_df[f'minutes_played_roll_avg_3'] = combined_df.groupby('player_id', group_keys=False)['minutes_played'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
    else:
        combined_df[f'minutes_played_roll_avg_3'] = np.nan

    combined_df['days_since_last_game'] = combined_df.groupby('player_id', group_keys=False)['game_datetime'].diff().dt.days

    features_for_prediction_row = combined_df.iloc[-1:].copy()

    if not hist_df.empty:
        last_historical_game_time = hist_df['game_datetime'].iloc[-1]
        days_diff = (pd.to_datetime(game_datetime) - last_historical_game_time).days
        features_for_prediction_row.loc[features_for_prediction_row.index[-1], 'days_since_last_game'] = days_diff
    else:
        features_for_prediction_row.loc[features_for_prediction_row.index[-1], 'days_since_last_game'] = 7

    if pd.isna(features_for_prediction_row.loc[features_for_prediction_row.index[-1], 'minutes_played']):
        avg_minutes = np.nan
        if not hist_df.empty and 'minutes_played' in hist_df and hist_df['minutes_played'].notna().any():
            avg_minutes = hist_df['minutes_played'].mean()
        imputed_minutes = avg_minutes if not pd.isna(avg_minutes) else 25
        features_for_prediction_row.loc[features_for_prediction_row.index[-1], 'minutes_played'] = imputed_minutes
        logger.info(f"Imputed 'minutes_played' with {imputed_minutes:.2f} for player {player_id_str} for prediction row.")

    numerical_feature_names = [
        'minutes_played', 'season',
        f'{target_stat}_lag_1', f'{target_stat}_lag_2', f'{target_stat}_lag_3',
        f'{target_stat}_roll_avg_3', 'minutes_played_roll_avg_3',
        'days_since_last_game'
    ]
    categorical_feature_names = ['home_team', 'away_team']
    all_feature_names_ordered = numerical_feature_names + categorical_feature_names

    for col_name in all_feature_names_ordered:
        if col_name not in features_for_prediction_row.columns:
            features_for_prediction_row[col_name] = np.nan
            logger.warning(f"Feature column '{col_name}' was missing; added as NaN.")
            
    final_features_df = features_for_prediction_row[all_feature_names_ordered]
    
    logger.info(f"Engineered features for player {player_id_str}, game {str(game_id_uuid)}. Shape: {final_features_df.shape}")
    logger.debug(f"Features for prediction:\n{final_features_df.to_string()}")
    return final_features_df

def calculate_probabilities(predicted_value: float, line: float, model_mse: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """
    Calculates over/under probabilities.
    Uses normal distribution assumption if model_mse is provided.
    """
    try:
        import scipy.stats
    except ImportError:
        logger.error("Scipy is not installed. Probabilities will be estimated using fallback.")
        # Fallback simple logic if scipy is not available
        if predicted_value > line:
            return 0.75, 0.25 
        elif predicted_value < line:
            return 0.25, 0.75
        else:
            return 0.5, 0.5

    if model_mse is not None and model_mse > 0:
        std_dev = np.sqrt(model_mse)
        if std_dev > 1e-6: # Check for practically non-zero std_dev to avoid division by zero or instability
            z = (line - predicted_value) / std_dev
            prob_over = 1 - scipy.stats.norm.cdf(z)
            prob_under = scipy.stats.norm.cdf(z) 
            return float(prob_over), float(prob_under)
        else:
            logger.warning(f"Standard deviation is too small ({std_dev:.2e}) from MSE ({model_mse:.2e}). Falling back to simple probability estimation.")
    elif model_mse is not None and model_mse <= 0:
        logger.warning(f"Model MSE ({model_mse}) is not positive. Falling back to simple probability estimation.")
    else: # model_mse is None
        logger.warning("Model MSE not provided. Falling back to simple probability estimation.")
    
    # Fallback simple logic if MSE is not suitable or not available
    if predicted_value > line:
        return 0.75, 0.25
    elif predicted_value < line:
        return 0.25, 0.75
    else:
        return 0.5, 0.5

def parse_line_from_outcomes(outcomes: List[Dict[str, Any]]) -> Optional[float]:
    """Parses the betting line (point) from the outcomes JSON."""
    if not outcomes or not isinstance(outcomes, list):
        return None
    # Assuming the line is consistent across Over/Under outcomes
    for outcome in outcomes:
        if isinstance(outcome, dict) and 'point' in outcome:
            return float(outcome['point'])
    return None


def make_predictions_for_props(db: Session, player_props: List[db_models.PlayerProp]):
    """
    Generates and stores predictions for a list of player props.
    """
    if not player_props:
        logger.info("No player props provided to make_predictions_for_props.")
        return

    # Group props by target_stat to load model only once per stat
    props_by_stat = {}
    for prop in player_props:
        target_stat = map_market_key_to_target_stat(prop.market.key)
        if target_stat:
            if target_stat not in props_by_stat:
                props_by_stat[target_stat] = []
            props_by_stat[target_stat].append(prop)

    for target_stat, props_for_stat in props_by_stat.items():
        logger.info(f"Processing {len(props_for_stat)} props for target_stat: {target_stat}")
        model_pipeline, model_version = load_latest_model_for_target_stat(db, target_stat)

        if not model_pipeline or not model_version:
            logger.warning(f"Skipping predictions for {target_stat} due to missing model.")
            continue
        
        model_mse = model_version.metrics.get('avg_mse') if model_version.metrics else None
        if model_mse is None:
            logger.warning(f"Avg MSE not found in metrics for model {model_version.version_name}. Probabilities will be estimated.")

        for prop_to_predict in props_for_stat:
            logger.info(f"Predicting for Prop ID: {prop_to_predict.id}, Player: {prop_to_predict.player.player_name}, Game: {prop_to_predict.game.game_datetime}, Market: {prop_to_predict.market.key}")

            # 1. Engineer features for this prop
            #    This requires player_id, game_id, game_datetime from the prop.
            features_df = engineer_features_for_prediction(
                db,
                prop_to_predict.player_id,
                prop_to_predict.game_id,
                prop_to_predict.game.game_datetime,
                target_stat
            )

            if features_df is None or features_df.empty:
                logger.warning(f"Could not engineer features for prop ID {prop_to_predict.id}. Skipping prediction.")
                continue
            
            # 2. Make prediction using the loaded model_pipeline
            try:
                predicted_stat_value = model_pipeline.predict(features_df)[0] # predict returns an array
                logger.info(f"Raw model prediction for {target_stat} for player {prop_to_predict.player.player_name}: {predicted_stat_value:.2f}")
            except Exception as e:
                logger.error(f"Error during model prediction for prop {prop_to_predict.id}: {e}", exc_info=True)
                continue

            # 3. Get the betting line from prop.outcomes
            betting_line = parse_line_from_outcomes(prop_to_predict.outcomes)
            if betting_line is None:
                logger.warning(f"Could not parse betting line from outcomes for prop {prop_to_predict.id}. Outcomes: {prop_to_predict.outcomes}. Skipping.")
                continue
            logger.info(f"Betting line for prop {prop_to_predict.id} is: {betting_line}")

            # 4. Calculate over/under probabilities
            prob_over, prob_under = calculate_probabilities(predicted_stat_value, betting_line, model_mse)
            logger.info(f"Calculated probabilities for prop {prop_to_predict.id} - Over: {prob_over:.2f}, Under: {prob_under:.2f}")

            # 5. Store the prediction in the database
            if prob_over is not None and prob_under is not None:
                prediction_data_dict = {
                    "player_prop_id": prop_to_predict.id,
                    "model_version_id": model_version.id,
                    "predicted_over_probability": prob_over,
                    "predicted_under_probability": prob_under,
                    "predicted_value": float(predicted_stat_value) 
                }
                logger.info(f"Prepared prediction_data_dict for DB: {prediction_data_dict}") # Log the dict
                prediction_create_data = prediction_schema.PredictionCreate(**prediction_data_dict)
                
                try:
                    existing_prediction = db.query(db_models.Prediction).filter(
                        db_models.Prediction.player_prop_id == prop_to_predict.id,
                        db_models.Prediction.model_version_id == model_version.id
                    ).first()

                    if existing_prediction:
                        logger.info(f"Prediction already exists for prop {prop_to_predict.id} and model {model_version.version_name}. Updating with predicted_value: {prediction_create_data.predicted_value}")
                        existing_prediction.predicted_over_probability = prediction_create_data.predicted_over_probability
                        existing_prediction.predicted_under_probability = prediction_create_data.predicted_under_probability
                        existing_prediction.predicted_value = prediction_create_data.predicted_value 
                        existing_prediction.prediction_datetime = datetime.datetime.utcnow()
                        db.commit()
                        db.refresh(existing_prediction)
                    else:
                        logger.info(f"No existing prediction found. Creating new prediction with data: {prediction_create_data.model_dump_json()}")
                        new_prediction = crud_predictions.create_prediction(db=db, prediction=prediction_create_data)
                        logger.info(f"Successfully created new prediction with ID: {new_prediction.id} and predicted_value: {new_prediction.predicted_value}")
                    logger.info(f"Successfully stored/updated prediction for prop ID {prop_to_predict.id}")
                except Exception as e:
                    logger.error(f"Failed to store prediction for prop {prop_to_predict.id}: {e}", exc_info=True)
                    db.rollback()
            else:
                logger.warning(f"Probabilities were None for prop {prop_to_predict.id}. Prediction not stored.")


def main(args):
    logger.info("Starting prediction generation process...")
    db: Optional[Session] = None
    try:
        db = get_db_session()
        
        date_filter = None
        if args.date:
            try:
                date_filter = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
                logger.info(f"Filtering for props on or after: {date_filter}")
            except ValueError:
                logger.error(f"Invalid date format: {args.date}. Please use YYYY-MM-DD.")
                return

        upcoming_props = get_upcoming_player_props(db, game_date_cutoff=date_filter)
        
        if not upcoming_props:
            logger.info("No upcoming player props found to process.")
            return
            
        make_predictions_for_props(db, upcoming_props)
            
    except Exception as e:
        logger.error(f"An error occurred in the main prediction pipeline: {e}", exc_info=True)
    finally:
        if db is not None:
            logger.info("Closing database session.")
            db.close()
    logger.info("Prediction generation process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions for WNBA player props.")
    parser.add_argument("--date", type=str, default=None,
                        help="Optional specific date (YYYY-MM-DD) to fetch props for (on or after this date). Defaults to today.")
    
    cli_args = parser.parse_args()
    
    # Need to import scipy for calculate_probabilities if using normal distribution
    # This is now handled within calculate_probabilities
    # try:
    #     import scipy.stats
    # except ImportError:
    #     logger.error("Scipy is not installed. Please install it ('pip install scipy') to use normal distribution for probability calculation. Falling back to simpler method.")
    #     # Modify calculate_probabilities to not rely on scipy or ensure it handles its absence
    #     # For now, the fallback is built-in, but a global flag could be better.
    #     pass 
        
    main(cli_args) 