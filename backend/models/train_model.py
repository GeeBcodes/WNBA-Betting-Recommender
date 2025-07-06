import asyncio
import sys
import os
import argparse
import logging
from pathlib import Path
import datetime
from typing import Optional, List, Any
import uuid
import traceback
import optuna
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import joblib
import pickle
import math
import re

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased,joinedload
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score
import xgboost as xgb

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.session import AsyncSessionLocal
from backend.db.models import PlayerStat, Game, Team, ModelVersion, Player
from backend.schemas.model_version import ModelVersionCreate
from backend.app.crud.model_versions import create_model_version
from backend.features.feature_engineering_core import generate_full_feature_set
from backend.app.core.config import DEFAULT_SEASONS, TARGET_STATS_TO_TRAIN, ROLLING_WINDOWS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Loading ---
async def load_data(
    db: AsyncSession,
    target_stat_type: str,
    model_type: str,
    seasons: List[int],
    game_id: Optional[str] = None,
) -> pd.DataFrame:
    """Loads player stats data from the database for the given seasons."""
    logger.info(f"Loading data for target_stat_type: '{target_stat_type}', model_type: '{model_type}', seasons: {seasons}")

    HomeTeam = aliased(Team, name="home_team")
    AwayTeam = aliased(Team, name="away_team")

    query = (
        select(
            PlayerStat.id, PlayerStat.player_id, PlayerStat.team_id, PlayerStat.game_id,
            PlayerStat.season, PlayerStat.minutes_played, PlayerStat.points, PlayerStat.rebounds,
            PlayerStat.assists, PlayerStat.steals, PlayerStat.blocks, PlayerStat.turnovers,
            PlayerStat.field_goals_made, PlayerStat.field_goals_attempted,
            PlayerStat.three_pointers_made, PlayerStat.three_pointers_attempted,
            PlayerStat.free_throws_made, PlayerStat.free_throws_attempted,
            PlayerStat.offensive_rebounds, PlayerStat.defensive_rebounds,
            PlayerStat.fouls, PlayerStat.player_efficiency_rating,
            PlayerStat.usage_rate, PlayerStat.true_shooting_percentage,
            PlayerStat.effective_field_goal_percentage,
            # Combined stats are calculated below, not loaded directly from DB
            Game.game_datetime, Game.home_team_id, Game.away_team_id,
            Game.home_score, Game.away_score,
            Team.team_name, Player.player_name,
            HomeTeam.team_name.label("home_team_name"),
            AwayTeam.team_name.label("away_team_name")
        )
        .join(Game, PlayerStat.game_id == Game.id)
        .join(Team, PlayerStat.team_id == Team.id)
        .join(Player, PlayerStat.player_id == Player.id)
        .join(HomeTeam, Game.home_team_id == HomeTeam.id)
        .join(AwayTeam, Game.away_team_id == AwayTeam.id)
        .filter(PlayerStat.season.in_(seasons))
        .filter(PlayerStat.minutes_played > 0)
    )

    if game_id:
        query = query.filter(PlayerStat.game_id == game_id)


    result = await db.execute(query)
    data = result.mappings().all()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # --- Calculate combined stats on the fly to ensure they exist with expected short names ---
    base_cols = ['points', 'rebounds', 'assists', 'steals', 'blocks']
    for col in base_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['pra'] = df['points'] + df['rebounds'] + df['assists']
    df['pr'] = df['points'] + df['rebounds']
    df['pa'] = df['points'] + df['assists']
    df['ra'] = df['rebounds'] + df['assists']
    df['blocks_plus_steals'] = df['blocks'] + df['steals']
    # ---

    df['opponent_team_id'] = df.apply(
        lambda row: row['away_team_id'] if str(row['team_id']) == str(row['home_team_id']) else row['home_team_id'],
        axis=1
    )
    return df

async def load_all_player_stats_for_seasons(db: AsyncSession, seasons: List[int]) -> pd.DataFrame:
    """Loads all player stats for all teams for the given seasons to build team-level features."""
    logger.info(f"Loading all player stats for seasons: {seasons} for feature generation.")

    query = (
        select(
            PlayerStat.team_id,
            PlayerStat.game_id,
            Game.game_datetime,
            Game.home_team_id,
            Game.away_team_id,
            PlayerStat.season,
            PlayerStat.points,
            PlayerStat.rebounds,
            PlayerStat.assists,
            PlayerStat.steals,
            PlayerStat.blocks,
            PlayerStat.turnovers,
            PlayerStat.field_goals_made,
            PlayerStat.field_goals_attempted,
            PlayerStat.three_pointers_made,
            PlayerStat.three_pointers_attempted,
            PlayerStat.free_throws_made,
            PlayerStat.free_throws_attempted
        )
        .join(Game, PlayerStat.game_id == Game.id)
        .filter(PlayerStat.season.in_(seasons))
    )
    result = await db.execute(query)
    data = result.mappings().all()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Determine opponent team ID for each player stat record
    df['opponent_team_id'] = df.apply(
        lambda row: row['away_team_id'] if str(row['team_id']) == str(row['home_team_id']) else row['home_team_id'],
        axis=1
    )
    return df

def generate_synthetic_prop_line(df: pd.DataFrame, target_stat: str) -> pd.DataFrame:
    """
    Generates a synthetic prop line for each game to enable classification training.
    This creates a binary target variable (Over/Under).
    """
    logger.info("Generating synthetic prop_line for classification model training...")

    valid_stats = df[target_stat].dropna()

    if valid_stats.empty:
        logger.warning(f"No valid data points found for '{target_stat}' to generate prop lines.")
        df['prop_line'] = np.nan
        df['target'] = np.nan
        return df

    quantiles = np.quantile(valid_stats, q=[0.1, 0.25, 0.5, 0.75, 0.9])
    quantiles = np.maximum(0, quantiles)

    if np.var(quantiles) < 1e-6:
        base_line = valid_stats.mean()
        prop_lines = np.random.choice([max(0.5, base_line - 0.5), max(0.5, base_line + 0.5)], size=len(df))
    else:
        noise = np.random.normal(0, 0.1, size=len(df))
        base_lines = np.random.choice(quantiles, size=len(df))
        prop_lines = np.round((base_lines + noise) * 2) / 2

    df['prop_line'] = np.maximum(0.5, prop_lines)
    df['target'] = (df[target_stat] > df['prop_line']).astype(int)
    df.dropna(subset=[target_stat], inplace=True)
    logger.info(f"Class balance - Over: {df['target'].sum()}, Under: {len(df) - df['target'].sum()}")
    logger.info(f"DataFrame size after generating synthetic prop_line and dropping NaNs: {len(df)}")
    return df

# --- NEW LEAKAGE_MAP AND HELPER FUNCTION ---
LEAKAGE_MAP = {
    'points': {
        'direct_components': {'field_goals_made', 'free_throws_made', 'three_pointers_made'},
        'advanced_metrics': {'player_efficiency_rating', 'true_shooting_percentage', 'effective_field_goal_percentage', 'usage_rate', 'points_per_40_min'}
    },
    'rebounds': {
        'direct_components': {'offensive_rebounds', 'defensive_rebounds'},
        'advanced_metrics': {'player_efficiency_rating', 'defensive_rebound_percentage', 'offensive_rebound_percentage', 'total_rebound_percentage', 'rebounds_per_40_min'}
    },
    'assists': {
        'direct_components': set(),
        'advanced_metrics': {'player_efficiency_rating', 'assist_percentage', 'assist_to_turnover_ratio', 'assists_per_40_min'}
    },
    'steals': {
        'direct_components': set(),
        'advanced_metrics': {'player_efficiency_rating', 'steals_per_40_min'}
    },
    'blocks': {
        'direct_components': set(),
        'advanced_metrics': {'player_efficiency_rating', 'block_percentage', 'blocks_per_40_min'}
    },
    'turnovers': {
        'direct_components': set(),
        'advanced_metrics': {'player_efficiency_rating', 'turnover_percentage', 'assist_to_turnover_ratio'}
    },
    'field_goals_made': {
        'direct_components': {'points'},
        'advanced_metrics': {'player_efficiency_rating', 'effective_field_goal_percentage', 'true_shooting_percentage', 'usage_rate'}
    },
    'three_pointers_made': {
        'direct_components': {'points', 'field_goals_made'},
        'advanced_metrics': {'player_efficiency_rating', 'effective_field_goal_percentage', 'true_shooting_percentage'}
    },
    'pra': {'direct_components': {'points', 'rebounds', 'assists'}},
    'pr': {'direct_components': {'points', 'rebounds'}},
    'pa': {'direct_components': {'points', 'assists'}},
    'ra': {'direct_components': {'rebounds', 'assists'}},
    'blocks_plus_steals': {'direct_components': {'blocks', 'steals'}},
}

STAT_COMPONENTS = {
    'pra': {'points', 'rebounds', 'assists'},
    'pr': {'points', 'rebounds'},
    'pa': {'points', 'assists'},
    'ra': {'rebounds', 'assists'},
    'blocks_plus_steals': {'blocks', 'steals'},
}

for base_stat, details in list(LEAKAGE_MAP.items()):
    if 'direct_components' in details:
        for combo_stat, combo_details in LEAKAGE_MAP.items():
            if 'direct_components' in combo_details and base_stat in combo_details['direct_components']:
                if 'part_of_combos' not in details:
                    details['part_of_combos'] = set()
                details['part_of_combos'].add(combo_stat)

def get_leaky_features_for_target(target_stat: str) -> set:
    """
    Identifies all features that could cause data leakage for a given target statistic.
    This includes:
    1. The direct component stats (e.g., 'points' for 'pr').
    2. Other combined stats that share components (e.g., 'pa' is leaky for 'pr').
    3. Advanced metrics calculated from the target or its components.
    """
    leaky_features = set()

    # Get the base components of the target_stat, if it's a combined stat.
    # If not a combined stat, its only component is itself (e.g., 'points').
    target_components = STAT_COMPONENTS.get(target_stat, {target_stat})

    # 1. Find all other combined stats that share any component with our target.
    # This finds siblings, like 'pa' when target is 'pr'.
    for combo_stat, components in STAT_COMPONENTS.items():
        if combo_stat != target_stat and not target_components.isdisjoint(components):
            leaky_features.add(combo_stat)

    # 2. Add all base components of the target stat itself.
    # For 'pra', this adds 'points', 'rebounds', and 'assists'.
    leaky_features.update(target_components)
    
    # 3. Recursively find components of components and their associated advanced metrics.
    stats_to_check = set(target_components)
    processed_stats = set()
    
    while stats_to_check:
        stat = stats_to_check.pop()
        if stat in processed_stats:
            continue
        processed_stats.add(stat)

        if stat in LEAKAGE_MAP:
            details = LEAKAGE_MAP[stat]
            # Add direct components and advanced metrics from the map.
            new_components = details.get('direct_components', set())
            leaky_features.update(new_components)
            leaky_features.update(details.get('advanced_metrics', set()))
            # Add the newly found components to the set to be checked recursively.
            stats_to_check.update(new_components)

    # The target stat itself should NEVER be in the list of features to remove.
    # This is the crucial fix for the "Target column not found" error.
    leaky_features.discard(target_stat)
    
    return leaky_features

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types and handle non-finite numbers for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64, float)): # Added standard float
        # Handle NaN, Infinity, -Infinity
        if not math.isfinite(obj):
            return None # Replace non-finite numbers with None (which becomes null in JSON)
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# --- Model Training ---
async def train_and_evaluate_model(
    df: pd.DataFrame,
    target_stat: str,
    model_type: str,
    seasons: List[int],
    db_session: AsyncSession,
    apply_box_cox: bool = False
) -> bool:
    """Trains a model, saves artifacts, and records metadata."""

    # --- Prevent Data Leakage ---
    # This is a critical step to prevent the model from "cheating" by using features
    # that are direct components or combinations of the target variable.

    # Get the list of stats that would leak information for the current target.
    # This set does NOT include the target_stat itself.
    leaky_base_stats = get_leaky_features_for_target(target_stat)
    leaky_columns = set()

    # Find columns containing leaky base stats. We use a regex with word boundaries (\b)
    # to prevent dropping 'pra' just because 'pr' is in the leaky set.
    if leaky_base_stats:
        pattern = r'\b(' + '|'.join(re.escape(s) for s in leaky_base_stats) + r')\b'
        leaky_columns.update({col for col in df.columns if re.search(pattern, col)})

    # Also, explicitly find and add derived features of the target stat itself,
    # like 'pra_rolling_10', being careful not to add the target 'pra' itself.
    for col in df.columns:
        if col.startswith(f"{target_stat}_"):
            leaky_columns.add(col)

    # For classification models, we must also remove the 'prop_line'
    if model_type == 'classification' and 'prop_line' in df.columns:
        leaky_columns.add('prop_line')
    
    # The target should never be dropped. As a final safeguard, remove it if it's present.
    leaky_columns.discard(target_stat)

    if leaky_columns:
        sorted_leaky_columns = sorted(list(leaky_columns))
        logger.info(f"Dropping columns to prevent data leakage: {sorted_leaky_columns}")
        df.drop(columns=sorted_leaky_columns, inplace=True, errors='ignore')


    logger.info(f"Starting training process for {target_stat} ({model_type})...")
    target_column = 'target' if model_type == 'classification' else target_stat

    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in DataFrame. Aborting.")
        return False
    
    # Drop rows where the target is NaN
    df.dropna(subset=[target_column], inplace=True)
    if df.empty:
        logger.error(f"DataFrame is empty after dropping NaNs in target column '{target_column}'. Aborting.")
        return False

    # Store original y_test for classification evaluation
    y_test_original = None
    if model_type == 'classification':
        y_test_original = df.loc[df.index.isin(df.sample(frac=0.15, random_state=42).index), target_column].copy()

    # --- Box-Cox Transformation (for regression) ---
    lmbda = None
    if model_type == 'regression' and apply_box_cox:
        logger.info("Applying Box-Cox transformation to the training target variable.")
        # Ensure target column is numeric and positive for Box-Cox
        y_train_series = df[target_column]
        if pd.to_numeric(y_train_series, errors='coerce').gt(0).all():
            y_train_transformed, lmbda = boxcox(y_train_series)
            df[target_column] = y_train_transformed
        else:
            logger.warning("Training target variable contains non-positive values. Skipping Box-Cox transformation.")

    y = df[target_column]
    X = df.drop(columns=[target_column], errors='ignore')
    X.columns = pd.Index([str(col) for col in X.columns])
    
    # --- Data Splitting (Train/Calibration/Test) ---
    X_train_calib, X_test, y_train_calib, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_calib, y_train, y_calib = train_test_split(X_train_calib, y_train_calib, test_size=0.2, random_state=42) # 0.2 * 0.85 = 0.17

    logger.info(f"Data split: {len(y_train)} train, {len(y_calib)} calibration, {len(y_test)} test samples.")

    # --- Feature Preprocessing ---
    # Explicitly drop ID and other non-feature columns before they enter the pipeline
    non_feature_cols = ['id', 'player_id', 'game_id', 'team_id', 'opponent_team_id', 'player_name', 'team_name', 'home_team_name', 'away_team_name', 'game_datetime', 'season']
    if model_type == 'classification':
        non_feature_cols.append(target_stat) # Drop the original stat column for classification
        
    X_train = X_train.drop(columns=[col for col in non_feature_cols if col in X_train.columns], errors='ignore')
    X_calib = X_calib.drop(columns=[col for col in non_feature_cols if col in X_calib.columns], errors='ignore')
    X_test = X_test.drop(columns=[col for col in non_feature_cols if col in X_test.columns], errors='ignore')

    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Ensure all feature names are strings before creating the ColumnTransformer
    numerical_features = [str(col) for col in numerical_features]
    categorical_features = [str(col) for col in categorical_features]
    
    final_feature_names = numerical_features + categorical_features

    logger.info(f"Using {len(numerical_features)} numerical features and {len(categorical_features)} categorical features.")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # --- Model Definition ---
    if model_type == 'regression':
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    else: # classification
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)

    best_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    logger.info("Fitting model pipeline on the training set...")
    best_pipeline.fit(X_train, y_train)

    # --- Get final feature names AFTER fitting the preprocessor ---
    try:
        # Access the preprocessor step from the final pipeline
        preprocessor_step = best_pipeline.named_steps['preprocessor']
        final_feature_names_from_pipeline = preprocessor_step.get_feature_names_out()
        logger.info(f"Successfully extracted {len(final_feature_names_from_pipeline)} feature names from pipeline. First 5: {final_feature_names_from_pipeline[:5]}")
    except Exception as e:
        logger.error(f"Could not get feature names from pipeline's preprocessor: {e}")
        # As a fallback, use the feature names list generated before fitting
        final_feature_names_from_pipeline = final_feature_names
        logger.warning("Using pre-fitting feature names as a fallback.")

    # --- Evaluation ---
    logger.info("Evaluating model performance on the test set...")
    y_pred = best_pipeline.predict(X_test)
    metrics = {}
    if model_type == 'regression':
        if lmbda is not None:
            y_test = inv_boxcox(y_test, lmbda)
            y_pred = inv_boxcox(y_pred, lmbda)
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)
        logger.info(f"Test Set Metrics for {target_stat} ({model_type}): {metrics}")
    else: # classification
        y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        # Ensure there's more than one class in y_test for roc_auc
        if len(np.unique(y_test)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        else:
            metrics['roc_auc'] = float('nan') # Not applicable
        logger.info(f"Test Set Metrics for {target_stat} ({model_type}): {metrics}")

    # --- ICP Score Calculation ---
    if model_type == 'regression':
        calib_pred = best_pipeline.predict(X_calib)
        if lmbda is not None:
            y_calib = inv_boxcox(y_calib, lmbda)
            calib_pred = inv_boxcox(calib_pred, lmbda)
        nonconformity_scores = np.abs(y_calib - calib_pred)
    else: # classification
        calib_pred_proba = best_pipeline.predict_proba(X_calib)
        # Get probabilities for the true class
        true_class_probs = calib_pred_proba[np.arange(len(y_calib)), y_calib.astype(int)]
        nonconformity_scores = 1 - true_class_probs

    # --- Save Artifacts and Metadata ---
    try:
        # Create a general name for the model type and a unique version name with a timestamp
        model_name_general = f"{target_stat}_{model_type}"
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        version_name = f"{model_name_general}_{timestamp}"

        # Define paths for saving artifacts
        model_dir = PROJECT_ROOT / "models" / model_name_general
        model_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline_path = model_dir / f"{version_name}_pipeline.joblib"
        scores_path = model_dir / f"{version_name}_icp_scores.joblib"

        # Save the trained pipeline and nonconformity scores
        joblib.dump(best_pipeline, pipeline_path)
        logger.info(f"Saved model pipeline to {pipeline_path}")
        
        joblib.dump(nonconformity_scores, scores_path)
        logger.info(f"Saved ICP nonconformity scores to {scores_path}")

        # Prepare metadata for DB, satisfying the schema
        model_version_data = ModelVersionCreate(
            model_name=model_name_general,
            version_name=version_name,
            model_type=model_type,
            target_stat=target_stat,
            seasons=seasons,
            description=f"XGBoost {model_type} model for {target_stat}",
            pipeline_path=str(pipeline_path.relative_to(PROJECT_ROOT)),
            model_path=str(pipeline_path.relative_to(PROJECT_ROOT)),
            metrics=convert_numpy_types(metrics),
            parameters=convert_numpy_types(best_pipeline.named_steps['model'].get_params()),
            feature_names= list(X.columns),
            nonconformity_scores_path=str(scores_path.relative_to(PROJECT_ROOT)) if model_type == 'regression' else None,
            nonconformity_scores_clf_path=str(scores_path.relative_to(PROJECT_ROOT)) if model_type == 'classification' else None,
            model_uuid=str(uuid.uuid4()),
            version=1,
            training_date=datetime.datetime.now()
        )

        created_model_version = await create_model_version(db_session, model_version_data)
        if created_model_version:
            logger.info(f"Successfully created model version entry with ID: {created_model_version.id}")
        else:
            logger.error("Failed to create model version entry in DB, but no exception was raised.")

    except Exception as e:
        logger.error(f"Failed to create model version in DB for {version_name}. Error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

    logger.info(f"Finished processing for {target_stat} ({model_type}).")
    return True


async def main(target_stat: str, model_type: str, seasons: List[int], transform: bool):
    """Main function to run the training process."""
    logger.info(f"Starting model training for stat: '{target_stat}', type: '{model_type}', seasons: {seasons}")
    
    async with AsyncSessionLocal() as db_session:
        try:
            df = await load_data(db_session, target_stat, model_type, seasons)

            if df is None or df.empty:
                logger.warning(f"Initial data load is too small for {target_stat} in seasons {seasons}. Skipping training.")
                return

            all_stats_df_for_features = await load_all_player_stats_for_seasons(db_session, seasons)

            if all_stats_df_for_features.empty:
                logger.error("Could not load player stats for feature generation. Aborting.")
                return

            df_featured = generate_full_feature_set(
                base_df=df,
                team_context_df=all_stats_df_for_features,
                target_stat=target_stat,
                rolling_windows=ROLLING_WINDOWS
            )

            if model_type == 'classification':
                df_featured = generate_synthetic_prop_line(df_featured, target_stat)
                if 'target' not in df_featured or df_featured['target'].isnull().all():
                     logger.error("Failed to generate a valid 'target' column for classification. Skipping.")
                     return

            await train_and_evaluate_model(
                df=df_featured,
                target_stat=target_stat,
                model_type=model_type,
                seasons=seasons,
                db_session=db_session,
                apply_box_cox=transform
            )

        except Exception as e:
            logger.error(f"An error occurred during the training process for {target_stat} ({model_type}): {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info(f"Finished processing for {target_stat} ({model_type}).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model for a specific player stat.")
    parser.add_argument("--target_stat", type=str, required=True, help="The target statistic to model (e.g., 'points').")
    parser.add_argument("--model_type", type=str, choices=['classification', 'regression'], required=True, help="The type of model to train.")
    parser.add_argument("--seasons", nargs='+', type=int, default=DEFAULT_SEASONS, help="List of seasons to use for training data.")
    parser.add_argument("--transform", action='store_true', help="Apply Box-Cox transformation to the target variable (for regression only).")
    args = parser.parse_args()

    asyncio.run(main(
        target_stat=args.target_stat,
        model_type=args.model_type,
        seasons=args.seasons,
        transform=args.transform
    ))