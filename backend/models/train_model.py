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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV # Changed from train_test_split, Added RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor # Changed from Classifier
import xgboost as xgb # Added for XGBoost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Changed metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Added OneHotEncoder
from sklearn.compose import ColumnTransformer # Added
from sklearn.impute import SimpleImputer # Added
import joblib

# Database related imports
from sqlalchemy.orm import Session, joinedload, aliased # Added joinedload and aliased
from backend.db.session import SessionLocal
# Updated CRUD import for model versions
from backend.app.crud import model_versions as crud_mv
from backend.db import models as db_models # For querying DB models
from backend.schemas import model_version as mv_schema # For creating ModelVersion schema

# Import the new shared feature engineering function
from backend.features.feature_engineering_core import generate_full_feature_set

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

    # Correctly create aliases for the Team table for distinct joins
    HomeTeamAliased = aliased(db_models.Team, name='home_team_table')
    AwayTeamAliased = aliased(db_models.Team, name='away_team_table')

    query = (
        db.query(
            db_models.PlayerStat.id.label('player_stat_id'),
            db_models.PlayerStat.player_id,
            db_models.PlayerStat.game_id,
            db_models.PlayerStat.minutes_played,
            getattr(db_models.PlayerStat, target_stat).label(target_stat), 
            db_models.PlayerStat.is_home_team,
            db_models.Game.game_datetime,
            # db_models.Game.home_team, # This was the issue
            # db_models.Game.away_team, # This was the issue
            HomeTeamAliased.team_name.label('home_team_name'), # Query name via join
            AwayTeamAliased.team_name.label('away_team_name'), # Query name via join
            db_models.Game.season, 
            db_models.Player.player_name
        )
        .join(db_models.Game, db_models.PlayerStat.game_id == db_models.Game.id)
        .join(db_models.Player, db_models.PlayerStat.player_id == db_models.Player.id)
        .join(HomeTeamAliased, db_models.Game.home_team_id == HomeTeamAliased.id) # Join for home team name
        .join(AwayTeamAliased, db_models.Game.away_team_id == AwayTeamAliased.id) # Join for away team name
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

def get_team_defensive_rolling_averages(
    db: Session, 
    target_stat: str, 
    seasons: Optional[List[int]] = None, 
    window_size: int = 10, 
    min_periods: int = 3,
    all_player_stats_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calculates for each team, their rolling average of a specific stat they have conceded to opponents 
    leading up to each game.
    Can use a pre-fetched DataFrame or query the database.
    Args:
        db: SQLAlchemy session, used if all_player_stats_df is None.
        target_stat: The target statistic.
        seasons: List of seasons, used if all_player_stats_df is None.
        window_size: Rolling window size.
        min_periods: Minimum periods for rolling calculation.
        all_player_stats_df: Optional pre-fetched DataFrame. Expected columns:
                             game_id, {target_stat} (as 'stat_value'), is_home_team, 
                             game_datetime, game_home_team_name, game_away_team_name.
    """
    logger.info(f"Calculating team defensive rolling averages for '{target_stat}' over {window_size} games (min {min_periods} periods).")

    if all_player_stats_df is not None and not all_player_stats_df.empty:
        logger.info("Using provided DataFrame for defensive rolling averages calculation.")
        # Ensure target_stat column is named 'stat_value' for consistent processing, or adapt logic.
        # The historical_team_context_df in predictor already has target_stat named correctly.
        # And it also has game_home_team_name, game_away_team_name.
        required_cols_from_df = ['game_id', target_stat, 'is_home_team', 'game_datetime', 'game_home_team_name', 'game_away_team_name']
        if not all(col in all_player_stats_df.columns for col in required_cols_from_df):
            missing = [col for col in required_cols_from_df if col not in all_player_stats_df.columns]
            logger.error(f"Provided DataFrame is missing required columns for defensive averages: {missing}. Cannot proceed with DataFrame.")
            return pd.DataFrame() # Or fall back to DB query if desired, but for now, error out if expected DF is bad.
        
        stats_df = all_player_stats_df.rename(columns={target_stat: 'stat_value'}).copy() # Use the target_stat column as 'stat_value'
    
    elif all_player_stats_df is not None and all_player_stats_df.empty:
        logger.warning("Provided DataFrame for defensive rolling averages is empty. Returning empty DataFrame.")
        return pd.DataFrame()
        
    else:
        logger.info(f"No DataFrame provided, querying database for defensive rolling averages. Seasons: {seasons if seasons else 'all'}")
        HomeTeamAliased = aliased(db_models.Team, name='home_team_conceded_calc')
        AwayTeamAliased = aliased(db_models.Team, name='away_team_conceded_calc')

        query = (
            db.query(
                db_models.PlayerStat.game_id,
                getattr(db_models.PlayerStat, target_stat).label('stat_value'),
                db_models.PlayerStat.is_home_team, 
                db_models.Game.game_datetime,
                HomeTeamAliased.team_name.label('game_home_team_name'),
                AwayTeamAliased.team_name.label('game_away_team_name')
            )
            .join(db_models.Game, db_models.PlayerStat.game_id == db_models.Game.id)
            .join(HomeTeamAliased, db_models.Game.home_team_id == HomeTeamAliased.id)
            .join(AwayTeamAliased, db_models.Game.away_team_id == AwayTeamAliased.id)
            .filter(getattr(db_models.PlayerStat, target_stat).isnot(None))
        )

        if seasons:
            query = query.filter(db_models.Game.season.in_(seasons))

        try:
            stats_df = pd.read_sql_query(query.statement, db.bind)
            if stats_df.empty:
                logger.warning(f"No player stats found from DB for defensive average calculation (target_stat='{target_stat}', seasons='{seasons}').")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading player stats from DB for defensive averages: {e}", exc_info=True)
            return pd.DataFrame()

    # Determine the defending team for each stat entry
    stats_df['defending_team_name'] = np.where(
        stats_df['is_home_team'], 
        stats_df['game_away_team_name'], 
        stats_df['game_home_team_name']
    )

    # Sum the target_stat conceded by the defending_team in each game
    conceded_per_game_df = stats_df.groupby(['game_id', 'defending_team_name', 'game_datetime'])['stat_value'].sum().reset_index()
    conceded_per_game_df.rename(columns={'stat_value': f'total_{target_stat}_conceded_in_game'}, inplace=True)
    
    # Sort by team and then by game date to prepare for rolling average calculation
    conceded_per_game_df = conceded_per_game_df.sort_values(by=['defending_team_name', 'game_datetime'])

    # Calculate rolling average of stat conceded, shifted to prevent data leakage from current game
    conceded_per_game_df[f'opponent_{target_stat}_conceded_roll_avg'] = (
        conceded_per_game_df.groupby('defending_team_name')[f'total_{target_stat}_conceded_in_game']
        .transform(lambda x: x.rolling(window=window_size, min_periods=min_periods).mean().shift(1))
    )

    # Select relevant columns to return
    # We need game_datetime to join correctly in feature_engineering if multiple games for a team on same day (unlikely for opponent stats, but good practice)
    result_df = conceded_per_game_df[['game_id', 'defending_team_name', 'game_datetime', f'opponent_{target_stat}_conceded_roll_avg']]
    
    logger.info(f"Successfully calculated team defensive rolling averages for {target_stat}. Resulting df shape: {result_df.shape}")
    return result_df

def get_team_performance_rolling_averages(db: Session, target_stat: str, all_player_stats_df_for_points: pd.DataFrame, seasons: Optional[List[int]] = None, window_size: int = 10, min_periods: int = 3) -> pd.DataFrame:
    """
    Calculates for each team, their rolling average of team performance (target_stat and total points) 
    leading up to each game.
    Uses a pre-fetched DataFrame of all player stats to avoid re-querying for total points calculation if target_stat is different.
    """
    logger.info(f"Calculating team performance rolling averages for '{target_stat}' and total points over {window_size} games (min {min_periods} periods). Seasons: {seasons if seasons else 'all'}")

    # Determine player's actual team for each stat entry
    # This reuses the logic from get_team_defensive_rolling_averages for identifying team names if we passed in the raw stats_df
    # For broader utility, this function should fetch its own base data or accept a generic player_stats_df.
    # For now, we will use the passed all_player_stats_df_for_points which should have game_home_team_name, game_away_team_name, is_home_team, points, and target_stat columns.

    # Ensure required columns are present (adjust if all_player_stats_df_for_points has different naming)
    required_cols = ['game_id', 'game_datetime', 'is_home_team', 'game_home_team_name', 'game_away_team_name', target_stat, 'points'] # 'points' for team total points
    if not all(col in all_player_stats_df_for_points.columns for col in required_cols):
        missing = [col for col in required_cols if col not in all_player_stats_df_for_points.columns]
        logger.error(f"Missing required columns in all_player_stats_df_for_points for team performance calculation: {missing}. Skipping.")
        return pd.DataFrame()

    # Make a copy to avoid modifying the original DataFrame passed in
    team_stats_df = all_player_stats_df_for_points.copy()

    team_stats_df['player_actual_team_name'] = np.where(
        team_stats_df['is_home_team'], 
        team_stats_df['game_home_team_name'], 
        team_stats_df['game_away_team_name']
    )

    # Sum the target_stat and total points for each team in each game
    # Need to ensure 'points' column exists from PlayerStat for total team points.
    # We assume target_stat column already has the specific stat value for each player.
    team_game_performance = team_stats_df.groupby(['game_id', 'player_actual_team_name', 'game_datetime']).agg(
        team_total_target_stat_in_game=(target_stat, 'sum'),
        team_total_points_in_game=('points', 'sum') # Assuming 'points' is the column name for player points
    ).reset_index()
    
    # Sort by team and then by game date for rolling average
    team_game_performance = team_game_performance.sort_values(by=['player_actual_team_name', 'game_datetime'])

    # Calculate rolling average of team's own target_stat performance
    team_game_performance[f'team_{target_stat}_roll_avg'] = (
        team_game_performance.groupby('player_actual_team_name')['team_total_target_stat_in_game']
        .transform(lambda x: x.rolling(window=window_size, min_periods=min_periods).mean().shift(1))
    )
    
    # Calculate rolling average of team's own total points performance
    team_game_performance[f'team_total_points_roll_avg'] = (
        team_game_performance.groupby('player_actual_team_name')['team_total_points_in_game']
        .transform(lambda x: x.rolling(window=window_size, min_periods=min_periods).mean().shift(1))
    )

    result_df = team_game_performance[[
        'game_id', 'player_actual_team_name', 'game_datetime', 
        f'team_{target_stat}_roll_avg', f'team_total_points_roll_avg'
    ]]
    
    logger.info(f"Successfully calculated team performance rolling averages. Resulting df shape: {result_df.shape}")
    return result_df

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
    
    # Define the model (XGBRegressor) - hyperparameters will be tuned
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', xgb_model)])
    
    # Define the hyperparameter distribution for RandomizedSearchCV
    # Note: parameters for the regressor step in the pipeline must be prefixed with 'regressor__'
    param_dist = {
        'regressor__n_estimators': [100, 200, 300, 500],
        'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 7, 9],
        'regressor__subsample': [0.7, 0.8, 0.9, 1.0], # Fraction of samples used for fitting the individual base learners
        'regressor__colsample_bytree': [0.7, 0.8, 0.9, 1.0], # Fraction of features used for fitting the individual base learners
        'regressor__gamma': [0, 0.1, 0.2, 0.3] # Minimum loss reduction required to make a further partition
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Setup RandomizedSearchCV
    # Using neg_mean_squared_error as scoring because RandomizedSearchCV tries to maximize the score
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,  # Number of parameter settings that are sampled. Increase for more thorough search.
        scoring='neg_mean_squared_error',
        cv=tscv,
        random_state=42,
        n_jobs=-1, # Use all available cores, be mindful if preprocessor also uses n_jobs
        verbose=1 # Logs the progress
    )

    logger.info("Starting RandomizedSearchCV for hyperparameter tuning...")
    random_search.fit(X, y)

    logger.info(f"Best hyperparameters found: {random_search.best_params_}")
    
    # The best_estimator_ is already refitted on the whole training data by RandomizedSearchCV
    best_pipeline = random_search.best_estimator_
    
    # To get evaluation metrics, we can predict on the test sets of the CV folds
    # Or, more simply for now, report the best score from the search
    # Note: random_search.best_score_ will be negative (e.g., -MSE).
    best_cv_mse = -random_search.best_score_ 
    logger.info(f"Best CV MSE from RandomizedSearchCV: {best_cv_mse:.4f}")

    # For a more complete set of metrics similar to before, we'd ideally re-evaluate the best_pipeline
    # on the TimeSeriesSplit folds. However, RandomizedSearchCV's `cv_results_` can provide this.
    # For simplicity here, we'll use the reported best_score_ as a proxy for avg_mse.
    # A proper evaluation would involve getting MAE and R2 with the best model on the splits.
    # Let's make a placeholder for other metrics. A full re-evaluation on splits would be more robust.
    
    # Re-calculating all metrics on the best model using the same CV splits (optional, for detailed reporting)
    # This is more complex as it requires iterating through tscv.split(X) again with the best_pipeline.
    # For now, we primarily rely on random_search.best_score_ for MSE.
    # We can enhance this later if needed to get MAE and R2 for the *best* model from the same CV splits.
    
    # For now, let's construct the metrics dictionary based on the best score and log the best params
    # This is a simplification. True average MAE/R2 for the *best* model would require re-running CV.
    avg_metrics = {
        'best_cv_mse': best_cv_mse,
        'avg_mae': None, # Placeholder - would need to re-evaluate best model on CV splits
        'avg_r2': None,  # Placeholder - would need to re-evaluate best model on CV splits
        'best_params': random_search.best_params_
    }
    logger.info(f"Average Cross-Validation Metrics for {target_stat} (using best XGBoost model):")
    logger.info(f"  Best CV MSE: {avg_metrics['best_cv_mse']:.4f} (from RandomizedSearch)")
    
    # The best_pipeline is already trained on the full data used in RandomizedSearch (X, y)
    # No explicit pipeline.fit(X,y) is needed here for best_pipeline from RandomizedSearchCV if refit=True (default)
    logger.info("Final model (best_pipeline from RandomizedSearchCV) is ready.")

    return best_pipeline, avg_metrics


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
        description=f"XGBRegressor for {target_stat} (tuned). Trained: {timestamp}. Avg Metrics: {metrics}",
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
        
        # Load raw data for all players and games (main dataset for features)
        df_raw = load_data(db, target_stat=args.target_stat, seasons=seasons_list)
        
        if df_raw.empty:
            logger.error("Data loading returned empty DataFrame. Exiting pipeline.")
            return
            
        # Prepare base data for team performance calculation (needs all player stats for aggregation)
        # We need a version of player stats that includes 'points' for team total points, and the target_stat.
        # This means load_data might need to be more flexible or we run it twice if target_stat is not points.
        # For now, assume df_raw (which is based on target_stat) also contains a 'points' column if different, 
        # or that get_team_performance_rolling_averages can handle if target_stat is 'points'.
        # A cleaner way would be for get_team_performance_rolling_averages to fetch its own broad player stats data.
        # Let's fetch a specific dataset for this that includes `points` and the current `target_stat` from PlayerStat model.
        
        # Query for all player stats needed for team performance aggregation
        # (game_id, game_datetime, is_home_team, home_team_name, away_team_name, target_stat value, points value)
        HomeTeamAliased = aliased(db_models.Team, name='home_team_perf_calc')
        AwayTeamAliased = aliased(db_models.Team, name='away_team_perf_calc')
        query_cols = [
            db_models.PlayerStat.game_id,
            db_models.PlayerStat.is_home_team,
            db_models.Game.game_datetime,
            HomeTeamAliased.team_name.label('game_home_team_name'),
            AwayTeamAliased.team_name.label('game_away_team_name'),
            getattr(db_models.PlayerStat, args.target_stat).label(args.target_stat), # Current target stat
            db_models.PlayerStat.points # Always include points for team total points
        ]
        # Ensure target_stat isn't added twice if it's 'points'
        if args.target_stat == 'points':
            query_cols.pop(-2) # Remove the getattr one, keep the explicit db_models.PlayerStat.points
            
        base_team_stats_query = (
            db.query(*query_cols)
            .join(db_models.Game, db_models.PlayerStat.game_id == db_models.Game.id)
            .join(HomeTeamAliased, db_models.Game.home_team_id == HomeTeamAliased.id)
            .join(AwayTeamAliased, db_models.Game.away_team_id == AwayTeamAliased.id)
            .filter(getattr(db_models.PlayerStat, args.target_stat).isnot(None))
            .filter(db_models.PlayerStat.points.isnot(None))
        )
        if seasons_list:
            base_team_stats_query = base_team_stats_query.filter(db_models.Game.season.in_(seasons_list))
        
        df_for_team_perf_calc = pd.read_sql_query(base_team_stats_query.statement, db.bind)
        
        # Get opponent defensive rolling averages
        df_opponent_defense_ravg = get_team_defensive_rolling_averages(db, target_stat=args.target_stat, seasons=seasons_list, all_player_stats_df=df_for_team_perf_calc.copy())
        
        # Get team's own performance rolling averages
        df_team_performance_ravg = get_team_performance_rolling_averages(db, target_stat=args.target_stat, all_player_stats_df_for_points=df_for_team_perf_calc.copy(), seasons=seasons_list)
            
        # df_featured = feature_engineering(df_raw.copy(), 
        #                                 target_stat=args.target_stat, 
        #                                 opponent_defense_df=df_opponent_defense_ravg,
        #                                 team_performance_df=df_team_performance_ravg)

        df_featured = generate_full_feature_set(base_df=df_raw.copy(),
                                                target_stat=args.target_stat,
                                                opponent_defense_df=df_opponent_defense_ravg,
                                                team_performance_df=df_team_performance_ravg)

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