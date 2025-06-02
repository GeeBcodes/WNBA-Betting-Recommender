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
from sqlalchemy import func, desc, String

from backend.db.session import SessionLocal
from backend.db import models as db_models
from backend.app.crud import predictions as crud_predictions
from backend.schemas import prediction as prediction_schema
# Placeholder for feature engineering logic, to be imported or defined
# from backend.models.train_model import feature_engineering, get_preprocessor # Or copy relevant parts

# Import the new shared feature engineering function
from backend.features.feature_engineering_core import generate_full_feature_set
# Temporary: Import data prep functions from train_model. Will be refactored.
from backend.models.train_model import get_team_defensive_rolling_averages, get_team_performance_rolling_averages 
# Aliasing to avoid potential conflicts if predictor.py also had a get_db_session
from backend.models.train_model import get_db_session as get_train_db_session_for_feature_helpers 

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
            joinedload(db_models.PlayerProp.game).joinedload(db_models.Game.home_team_ref),
            joinedload(db_models.PlayerProp.game).joinedload(db_models.Game.away_team_ref),
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

    def engineer_features_for_prediction(
        db: Session,
        player_id_uuid: uuid.UUID,
        game_id_uuid: uuid.UUID,
        game_datetime: datetime.datetime,
        target_stat: str,
        home_team_name_prop: str,
        away_team_name_prop: str,
        is_player_home_prop: bool
    ) -> Optional[pd.DataFrame]:
        player_id_str = str(player_id_uuid)
        logger.info(f"Starting feature engineering for player {player_id_str}, game {str(game_id_uuid)} ({game_datetime.strftime('%Y-%m-%d %H:%M')}), target: {target_stat}")

        # 1. Construct base_df (historical data + current game row)
        HomeTeamAliased = db_models.aliased(db_models.Team, name='hist_home_team_pred')
        AwayTeamAliased = db_models.aliased(db_models.Team, name='hist_away_team_pred')

    historical_stats_query = (
        db.query(
                db_models.PlayerStat.player_id.cast(String).label('player_id'),
                db_models.Game.game_datetime,
                getattr(db_models.PlayerStat, target_stat).label(target_stat),
            db_models.PlayerStat.minutes_played,
                db_models.PlayerStat.is_home_team,
                HomeTeamAliased.team_name.label('home_team_name'),
                AwayTeamAliased.team_name.label('away_team_name'),
                db_models.Game.season,
                db_models.PlayerStat.game_id.cast(String).label('game_id')
        )
        .join(db_models.Game, db_models.PlayerStat.game_id == db_models.Game.id)
            .join(HomeTeamAliased, db_models.Game.home_team_id == HomeTeamAliased.id)
            .join(AwayTeamAliased, db_models.Game.away_team_id == AwayTeamAliased.id)
        .filter(db_models.PlayerStat.player_id == player_id_uuid)
        .filter(db_models.Game.game_datetime < game_datetime)
            .order_by(db_models.Game.game_datetime.asc())
    )
    hist_df = pd.read_sql_query(historical_stats_query.statement, db.bind)
    
    if not hist_df.empty:
            hist_df['game_datetime'] = pd.to_datetime(hist_df['game_datetime'], utc=True)
    else:
            logger.info(f"No historical stats for player {player_id_str} before {game_datetime}.")

        upcoming_game_db_details = db.query(db_models.Game.season).filter(db_models.Game.id == game_id_uuid).first()
        current_game_season = upcoming_game_db_details.season if upcoming_game_db_details else datetime.datetime.now(datetime.timezone.utc).year

    current_game_data = {
        'player_id': player_id_str,
            'game_datetime': pd.to_datetime(game_datetime, utc=True),
        target_stat: np.nan,
        'minutes_played': np.nan,
            'is_home_team': is_player_home_prop,
            'home_team_name': home_team_name_prop,
            'away_team_name': away_team_name_prop,
            'season': current_game_season,
            'game_id': str(game_id_uuid)
    }
    current_game_df_row = pd.DataFrame([current_game_data])

    if not hist_df.empty:
            base_df_for_features = pd.concat([hist_df, current_game_df_row], ignore_index=True)
    else:
            base_df_for_features = current_game_df_row

        # Ensure all columns expected by generate_full_feature_set are present
        expected_base_cols = ['player_id', 'game_datetime', target_stat, 'minutes_played',
                              'is_home_team', 'home_team_name', 'away_team_name', 'season', 'game_id']
        for col in expected_base_cols:
            if col not in base_df_for_features.columns:
                base_df_for_features[col] = np.nan
        
        base_df_for_features = base_df_for_features.sort_values(by=['player_id', 'game_datetime'])

        # --- Logic to populate opponent_defense_df and team_performance_df ---
        logger.info(f"Fetching additional historical data for team rolling averages for game {str(game_id_uuid)} up to {game_datetime}")
        
        # Determine the season of the game to be predicted
        # current_game_season is already available from earlier in the function.
        
        # Fetch broader historical data for the current game's season up to the game's datetime
        # This data will be used for calculating team-level rolling averages.
        # We need columns: game_id, game_datetime, is_home_team, home_team_name, away_team_name, target_stat value, points value for all players.
        PredictionContextHomeTeam = db_models.aliased(db_models.Team, name='pred_ctx_home_team')
        PredictionContextAwayTeam = db_models.aliased(db_models.Team, name='pred_ctx_away_team')

        # Define the columns to query for the context data
        context_query_cols = [
            db_models.PlayerStat.game_id,
            db_models.PlayerStat.is_home_team,
            db_models.Game.game_datetime,
            PredictionContextHomeTeam.team_name.label('game_home_team_name'), # Renamed to avoid clash if base_df had these
            PredictionContextAwayTeam.team_name.label('game_away_team_name'), # Renamed to avoid clash
            getattr(db_models.PlayerStat, target_stat).label(target_stat),
            db_models.PlayerStat.points
        ]
        if target_stat == 'points': # Avoid duplicate 'points' column
            context_query_cols.pop(-2)

        historical_context_query = (
            db.query(*context_query_cols)
            .join(db_models.Game, db_models.PlayerStat.game_id == db_models.Game.id)
            .join(PredictionContextHomeTeam, db_models.Game.home_team_id == PredictionContextHomeTeam.id)
            .join(PredictionContextAwayTeam, db_models.Game.away_team_id == PredictionContextAwayTeam.id)
            .filter(db_models.Game.season == current_game_season)
            .filter(db_models.Game.game_datetime < game_datetime) # Only games *before* the one we're predicting for
            .filter(getattr(db_models.PlayerStat, target_stat).isnot(None))
            .filter(db_models.PlayerStat.points.isnot(None))
            .order_by(db_models.Game.game_datetime.asc())
        )
        
        historical_team_context_df = pd.read_sql_query(historical_context_query.statement, db.bind)

        if not historical_team_context_df.empty:
            historical_team_context_df['game_datetime'] = pd.to_datetime(historical_team_context_df['game_datetime'], utc=True)
            logger.info(f"Fetched {len(historical_team_context_df)} records for team context for season {current_game_season}.")
            
            # Prepare opponent defensive stats
            # The function get_team_defensive_rolling_averages expects a db session and seasons list.
            # For prediction, we are providing data up to a point in time.
            # We need to adapt its usage or re-implement its core logic for this context.
            # For now, let's assume we can call it by passing the *current* db session,
            # and the `historical_team_context_df` can be processed by a similar logic as inside that function.
            # This part requires careful adaptation of how get_team_defensive_rolling_averages is called or used.
            # The original function queries DB. Here we give it a DF.
            # Let's call the original function, it will re-query but filtered by season. This might be acceptable for now.
            
            opponent_defense_df_for_pred = get_team_defensive_rolling_averages(
                db=db, # db is still needed for the fallback case within the function
                target_stat=target_stat, 
                seasons=[current_game_season], # seasons is also for fallback or if function uses it internally
                all_player_stats_df=historical_team_context_df.copy() # Pass the DataFrame
            )
            # The result of get_team_defensive_rolling_averages needs to be shifted for prediction time.
            # The function itself applies a shift(1). We need to ensure game_ids from opponent_defense_df_for_pred are aligned or game_datetime.
            # And we only need the values *before* the current game_datetime.
            if opponent_defense_df_for_pred is not None and not opponent_defense_df_for_pred.empty:
                opponent_defense_df_for_pred['game_datetime'] = pd.to_datetime(opponent_defense_df_for_pred['game_datetime'], utc=True)
                # Filter to include data strictly before the current game's datetime
                opponent_defense_df_for_pred = opponent_defense_df_for_pred[opponent_defense_df_for_pred['game_datetime'] < game_datetime].copy()


            team_performance_df_for_pred = get_team_performance_rolling_averages(
                db=db, # Uses the existing db session
                target_stat=target_stat,
                all_player_stats_df_for_points=historical_team_context_df.copy(), # Pass the df we just fetched
                seasons=[current_game_season]
            )
            if team_performance_df_for_pred is not None and not team_performance_df_for_pred.empty:
                team_performance_df_for_pred['game_datetime'] = pd.to_datetime(team_performance_df_for_pred['game_datetime'], utc=True)
                team_performance_df_for_pred = team_performance_df_for_pred[team_performance_df_for_pred['game_datetime'] < game_datetime].copy()

        else:
            logger.warning(f"No historical team context data found for season {current_game_season} before {game_datetime}. Rolling averages for opponent/team will be NaN or missing.")
            opponent_defense_df_for_pred = pd.DataFrame() # Empty DataFrame
            team_performance_df_for_pred = pd.DataFrame()   # Empty DataFrame
        
        # --- THIS IS THE KEY CHANGE FOR THIS STEP ---
        # For now, we pass None for the team-based rolling average DataFrames.
        # The logic to populate these for the prediction context will be added next.
        # opponent_defense_df_for_pred = None # Now populated above
        # team_performance_df_for_pred = None # Now populated above
        # logger.warning("TEMPORARY: Passing None for opponent_defense_df and team_performance_df in predictor.")
        # --- END KEY CHANGE ---

        all_features_df = generate_full_feature_set(
            base_df=base_df_for_features,
            target_stat=target_stat,
            opponent_defense_df=opponent_defense_df_for_pred,
            team_performance_df=team_performance_df_for_pred
        )

        if all_features_df.empty:
            logger.warning(f"Feature generation returned empty DataFrame for player {player_id_str}, game {str(game_id_uuid)}.")
            return None

        prediction_features_row = all_features_df.iloc[-1:].copy()
        logger.info(f"Engineered features for player {player_id_str}, game {str(game_id_uuid)}. Shape: {prediction_features_row.shape}")
        return prediction_features_row

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
    if not player_props:
        logger.info("No player props provided to make_predictions_for_props.")
        return

    props_by_stat: Dict[str, List[db_models.PlayerProp]] = {}
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
        
        model_mse = None
        if model_version.metrics:
            model_mse = model_version.metrics.get('best_cv_mse', model_version.metrics.get('avg_mse')) 
        if model_mse is None:
            logger.warning(f"MSE not found in metrics for model {model_version.version_name}. Probabilities will be estimated.")

        for prop_to_predict in props_for_stat:
            if not (prop_to_predict.player and prop_to_predict.game and prop_to_predict.game.home_team_ref and prop_to_predict.game.away_team_ref):
                logger.warning(f"Prop ID {prop_to_predict.id} is missing player, game, or game team references. Skipping.")
                continue

            logger.info(f"Predicting for Prop ID: {prop_to_predict.id}, Player: {prop_to_predict.player.player_name}, Game: {prop_to_predict.game.game_datetime}, Market: {prop_to_predict.market.key}")

            game_obj = prop_to_predict.game
            player_obj = prop_to_predict.player
            is_player_home = False
            if player_obj.team_id and game_obj.home_team_id:
                if player_obj.team_id == game_obj.home_team_id:
                    is_player_home = True
            else:
                logger.warning(f"Could not reliably determine if player {player_obj.player_name} is home for prop {prop_to_predict.id}. Defaulting to False.")
            
            game_home_team_name = game_obj.home_team
            game_away_team_name = game_obj.away_team

            if not game_home_team_name or not game_away_team_name:
                logger.warning(f"Home or away team name missing for game ID {game_obj.id}. Skipping prop {prop_to_predict.id}")
                continue
            
            features_df = engineer_features_for_prediction(
                db,
                prop_to_predict.player_id,
                prop_to_predict.game_id,
                prop_to_predict.game.game_datetime,
                target_stat,
                home_team_name_prop=game_home_team_name,
                away_team_name_prop=game_away_team_name,
                is_player_home_prop=is_player_home
            )

            if features_df is None or features_df.empty:
                logger.warning(f"Could not engineer features for prop ID {prop_to_predict.id} (placeholder in engineer_features_for_prediction). Skipping prediction.")
                continue
            
            try:
                predicted_stat_value = model_pipeline.predict(features_df)[0] 
                logger.info(f"Raw model prediction for {target_stat} for player {player_obj.player_name}: {predicted_stat_value:.2f}")
            except Exception as e:
                logger.error(f"Error during model prediction for prop {prop_to_predict.id}: {e}", exc_info=True)
                logger.debug(f"Features DataFrame that caused error:\n{features_df.to_string()}")
                continue

            betting_line = parse_line_from_outcomes(prop_to_predict.outcomes)
            if betting_line is None:
                logger.warning(f"Could not parse betting line for prop {prop_to_predict.id}. Skipping.")
                continue

            prob_over, prob_under = calculate_probabilities(predicted_stat_value, betting_line, model_mse)
            logger.info(f"Calculated probabilities for prop {prop_to_predict.id} - Over: {prob_over:.2f if prob_over is not None else 'N/A'}, Under: {prob_under:.2f if prob_under is not None else 'N/A'}")

            if prob_over is not None and prob_under is not None:
                prediction_data_dict = {
                    "player_prop_id": prop_to_predict.id,
                    "model_version_id": model_version.id,
                    "predicted_over_probability": prob_over,
                    "predicted_under_probability": prob_under,
                    "predicted_value": float(predicted_stat_value) 
                }
                prediction_create_data = prediction_schema.PredictionCreate(**prediction_data_dict)
                try:
                    existing_prediction = db.query(db_models.Prediction).filter(
                        db_models.Prediction.player_prop_id == prop_to_predict.id,
                        db_models.Prediction.model_version_id == model_version.id
                    ).first()
                    if existing_prediction:
                        existing_prediction.predicted_over_probability = prediction_create_data.predicted_over_probability
                        existing_prediction.predicted_under_probability = prediction_create_data.predicted_under_probability
                        existing_prediction.predicted_value = prediction_create_data.predicted_value 
                        existing_prediction.prediction_datetime = datetime.datetime.now(datetime.timezone.utc)
                        db.commit()
                        db.refresh(existing_prediction)
                        logger.info(f"Successfully updated prediction for prop ID {prop_to_predict.id}")
                    else:
                        new_prediction = crud_predictions.create_prediction(db=db, prediction=prediction_create_data)
                        logger.info(f"Successfully created new prediction ID: {new_prediction.id} for prop ID {prop_to_predict.id}")
                except Exception as e:
                    logger.error(f"DB error storing prediction for prop {prop_to_predict.id}: {e}", exc_info=True)
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