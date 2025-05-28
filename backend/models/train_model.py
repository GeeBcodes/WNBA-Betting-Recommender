import sys
import os
import argparse # Added
import logging
from pathlib import Path # Path is used, ensure it's imported
import datetime # datetime is used, ensure it's imported
import uuid
from typing import Optional, List # Added List

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np # Added numpy
from sklearn.model_selection import TimeSeriesSplit # Changed from train_test_split
from sklearn.ensemble import RandomForestRegressor # Changed from Classifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Changed metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Added OneHotEncoder
from sklearn.compose import ColumnTransformer # Added
from sklearn.impute import SimpleImputer # Added
import joblib

# Database related imports
from sqlalchemy.orm import Session, joinedload # Added joinedload
from backend.db.session import SessionLocal
# Updated CRUD import for model versions
from backend.app.crud import model_versions as crud_mv
from backend.db import models as db_models # For querying DB models
from backend.schemas import model_version as mv_schema # For creating ModelVersion schema

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a named logger

# Define paths
MODEL_ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def get_db_session() -> Session:
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        logger.error(f"Error creating database session: {e}")
        if db:
            db.close()
        raise

def load_data(db: Session, target_stat: str, seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Loads player stats, game, and player data from the database for the specified target stat and seasons.
    """
    logger.info(f"Loading data for target stat: '{target_stat}', seasons: {seasons if seasons else 'all'}")

    query = (
        db.query(
            db_models.PlayerStat.id.label('player_stat_id'),
            db_models.PlayerStat.player_id,
            db_models.PlayerStat.game_id,
            db_models.PlayerStat.minutes_played,
            getattr(db_models.PlayerStat, target_stat).label(target_stat), # Dynamically get target stat
            db_models.Game.game_datetime,
            db_models.Game.home_team, # Changed from home_team_id
            db_models.Game.away_team, # Changed from away_team_id
            db_models.Game.season, # Assuming 'season' column exists in Game model
            db_models.Player.player_name # Example player feature
            # Add other relevant fields from PlayerStat, Game, Player
        )
        .join(db_models.Game, db_models.PlayerStat.game_id == db_models.Game.id)
        .join(db_models.Player, db_models.PlayerStat.player_id == db_models.Player.id)
    )

    if seasons:
        query = query.filter(db_models.Game.season.in_(seasons))

    # Add ordering for time-series consistency if needed before df conversion
    query = query.order_by(db_models.Game.game_datetime, db_models.PlayerStat.player_id)
    
    try:
        df = pd.read_sql_query(query.statement, db.bind)
        if df.empty:
            logger.warning(f"No data found for target_stat='{target_stat}' and seasons='{seasons}'.")
            return pd.DataFrame()
        logger.info(f"Successfully loaded {len(df)} records from the database.")
        # Basic data type conversions if necessary (e.g., game_datetime to datetime)
        df['game_datetime'] = pd.to_datetime(df['game_datetime'])
        return df
    except Exception as e:
        logger.error(f"Error loading data from database: {e}", exc_info=True)
        return pd.DataFrame()


def feature_engineering(df: pd.DataFrame, target_stat: str) -> pd.DataFrame:
    """
    Performs feature engineering. Creates lagged features, rolling averages, etc.
    """
    if df.empty:
        logger.warning("DataFrame is empty. Skipping feature engineering.")
        return df
        
    logger.info(f"Performing feature engineering for target stat: {target_stat}")
    
    df = df.sort_values(by=['player_id', 'game_datetime'])

    # Example: Lagged features for the target stat (e.g., performance in last 1, 2, 3 games)
    for lag in [1, 2, 3]:
        df[f'{target_stat}_lag_{lag}'] = df.groupby('player_id')[target_stat].shift(lag)
    
    # Example: Rolling average for target_stat (e.g., avg over last 3 games, excluding current)
    df[f'{target_stat}_roll_avg_3'] = df.groupby('player_id')[target_stat].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1) # shift(1) to avoid data leakage
    )
    df[f'minutes_played_roll_avg_3'] = df.groupby('player_id')['minutes_played'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    )

    # Example: Days since last game (rest days)
    df['days_since_last_game'] = df.groupby('player_id')['game_datetime'].diff().dt.days.fillna(7) # Assume 7 for first game or use a larger sensible default

    # Example: Home/Away status (requires team_id for the player's team on PlayerStat or Game)
    # This part needs the player's actual team_id for that game to compare with home_team_id
    # df['is_home'] = (df['player_team_id_column'] == df['home_team_id']).astype(int) 

    # Categorical features like opponent team ID might need encoding later
    # df['opponent_id'] can be derived from home_team_id/away_team_id and player's team_id

    # Drop rows with NaNs created by shift/rolling, or impute them in preprocessor
    # For simplicity here, we might rely on SimpleImputer in the pipeline
    
    logger.info("Feature engineering complete.")
    return df

def get_preprocessor(numerical_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for preprocessing.
    """
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Or 'constant' fill_value='missing'
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Or 'passthrough' if other columns are needed and handled
    )
    return preprocessor

def train_and_evaluate_model(df: pd.DataFrame, target_stat: str):
    """Trains a regression model and evaluates it using TimeSeriesSplit."""
    logger.info(f"Starting model training and evaluation for {target_stat}...")
    
    if df.empty or target_stat not in df.columns:
        logger.error(f"DataFrame is empty or target_stat '{target_stat}' column is missing. Skipping training.")
        return None, None

    # Ensure target is numeric and handle NaNs if any (should ideally be clean)
    df = df.dropna(subset=[target_stat]) 
    if df.empty:
        logger.error(f"DataFrame empty after dropping NaNs in target '{target_stat}'. Skipping training.")
        return None, None

    y = df[target_stat]
    # Define features: Exclude IDs, date/datetime, and the target itself
    # Explicitly list features to be used or columns to be dropped
    X = df.drop(columns=[target_stat, 'player_stat_id', 'player_id', 'game_id', 'game_datetime', 'player_name']) # Adjust as necessary

    if X.empty:
        logger.error("Feature set X is empty. Check feature engineering and column drops. Skipping training.")
        return None, None

    # Identify numerical and categorical features from X's columns
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist() # Or include=['object', 'category']

    logger.info(f"Numerical features for preprocessing: {numerical_features}")
    logger.info(f"Categorical features for preprocessing: {categorical_features}")

    if not numerical_features and not categorical_features:
        logger.error("No numerical or categorical features identified for the preprocessor. Skipping training.")
        return None, None
        
    preprocessor = get_preprocessor(numerical_features, categorical_features)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    
    # Time-series split requires data to be sorted by time
    # load_data already sorts by game_datetime, player_id. Ensure overall temporal order for split.
    # If multiple players per game_datetime, this split is reasonable.
    
    tscv = TimeSeriesSplit(n_splits=5)
    all_metrics = []

    logger.info("Starting cross-validation with TimeSeriesSplit...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        logger.info(f"Training on fold {fold+1}/5...")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if X_train.empty or y_train.empty:
            logger.warning(f"Fold {fold+1} has empty training data. Skipping this fold.")
            continue
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        all_metrics.append({'mse': mse, 'mae': mae, 'r2': r2})
        logger.info(f"  Fold {fold+1} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    if not all_metrics:
        logger.error("No cross-validation folds were successfully trained and evaluated.")
        return None, None

    # Aggregate metrics (e.g., average)
    avg_metrics = {
        'avg_mse': np.mean([m['mse'] for m in all_metrics]),
        'avg_mae': np.mean([m['mae'] for m in all_metrics]),
        'avg_r2': np.mean([m['r2'] for m in all_metrics])
    }
    logger.info(f"Average Cross-Validation Metrics for {target_stat}:")
    logger.info(f"  Avg MSE: {avg_metrics['avg_mse']:.4f}")
    logger.info(f"  Avg MAE: {avg_metrics['avg_mae']:.4f}")
    logger.info(f"  Avg R2: {avg_metrics['avg_r2']:.4f}")

    # Retrain on all data (or all but last fold for a final holdout)
    logger.info(f"Retraining model on all available data for {target_stat}...")
    pipeline.fit(X, y) # Retrain on the full dataset X, y
    logger.info("Final model retraining complete.")

    return pipeline, avg_metrics


def save_model_artifact_and_metadata(db: Session, model_pipeline: Pipeline, target_stat: str, metrics: dict):
    """Saves the trained pipeline and its metadata."""
    if model_pipeline is None:
        logger.warning("No model pipeline to save.")
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version_name = f"{target_stat}_model_v{timestamp}" # Incorporate target_stat
    artifact_filename = f"{model_version_name}.joblib"
    artifact_path = MODEL_ARTIFACTS_DIR / artifact_filename
    
    logger.info(f"Saving model artifact to {artifact_path}")
    joblib.dump(model_pipeline, artifact_path)
    logger.info("Model artifact saved successfully.")

    # Create ModelVersion entry in the database
    model_version_data = mv_schema.ModelVersionCreate(
        version_name=model_version_name,
        description=f"RandomForestRegressor for {target_stat}. Trained: {timestamp}. Avg Metrics: {metrics}",
        model_path=str(artifact_path.relative_to(PROJECT_ROOT)), # Store relative path from project root
        metrics=metrics, # Store the metrics dict
        # parameters: Optional - can store model hyperparameters or feature list used
    )
    try:
        # Use the imported crud_mv for creating model version
        db_model_version = crud_mv.create_model_version(db=db, model_version=model_version_data)
        logger.info(f"Successfully created ModelVersion record with ID: {db_model_version.id} and Name: {db_model_version.version_name}")
        return db_model_version
    except Exception as e:
        logger.error(f"Failed to create ModelVersion record in DB: {e}", exc_info=True)
        return None

def main(args):
    """Main function to orchestrate data loading, training, and saving."""
    logger.info(f"Starting the model training pipeline for target_stat: {args.target_stat}...")
    
    db: Optional[Session] = None
    try:
        db = get_db_session()
        
        seasons_list = [int(s.strip()) for s in args.seasons.split(',')] if args.seasons else None
        
        df_raw = load_data(db, target_stat=args.target_stat, seasons=seasons_list)
        
        if df_raw.empty:
            logger.error("Data loading returned empty DataFrame. Exiting pipeline.")
            return
            
        df_featured = feature_engineering(df_raw.copy(), target_stat=args.target_stat)

        if df_featured.empty:
            logger.error("DataFrame empty after feature engineering. Exiting pipeline.")
            return

        trained_pipeline, eval_metrics = train_and_evaluate_model(df_featured, args.target_stat)
        
        if trained_pipeline and eval_metrics:
            save_model_artifact_and_metadata(db, trained_pipeline, args.target_stat, eval_metrics)
        else:
            logger.warning("Model training or evaluation failed. Artifact not saved.")
            
    except Exception as e:
        logger.error(f"An error occurred in the main training pipeline: {e}", exc_info=True)
    finally:
        if db is not None:
            logger.info("Closing database session.")
            db.close()
        
    logger.info("Model training pipeline finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a WNBA player stat prediction model.")
    parser.add_argument("--target_stat", type=str, required=True, 
                        help="The player statistic to predict (e.g., 'points', 'rebounds', 'assists'). Must be a column in PlayerStat model.")
    parser.add_argument("--seasons", type=str, default=None,
                        help="Optional comma-separated list of seasons (years) to train on (e.g., '2022,2023'). Defaults to all available if not specified in load_data.")
    
    cli_args = parser.parse_args()
    main(cli_args) 