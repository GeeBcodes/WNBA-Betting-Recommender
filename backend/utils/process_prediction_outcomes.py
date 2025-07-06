import sys
import os
import argparse
import logging
import asyncio
from datetime import date, datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from backend.db.session import AsyncSessionLocal
from backend.db import models as db_models
from backend.app.schemas import performance as performance_schema
from backend.app.crud import performance as crud_performance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_target_stat_from_market_key(market_key: str) -> Optional[str]:
    """Maps a market key from the database to a target stat name."""
    # This mapping should ideally be centralized or loaded from a config.
    # For now, it mirrors the logic in the predictor.
    mapping = {
        "player_points_rebounds_assists": "pra",
        "player_points_rebounds": "points_rebounds",
        "player_points_assists": "points_assists",
        "player_rebounds_assists": "rebounds_assists",
        "player_blocks_steals": "blocks_steals",
        "player_points": "points",
        "player_rebounds": "rebounds",
        "player_assists": "assists",
        "player_threes": "three_pointers_made",
        "player_steals": "steals",
        "player_blocks": "blocks",
        "player_turnovers": "turnovers",
    }
    for key, stat in mapping.items():
        if market_key.startswith(key):
            return stat
    return None

async def fetch_processed_predictions(db: AsyncSession, analysis_date: date) -> pd.DataFrame:
    """
    Fetches predictions and joins them with actual outcomes from PlayerStat.
    It constructs a comprehensive DataFrame with all data needed for performance analysis.
    """
    logger.info(f"Fetching predictions for games on and after {analysis_date}...")
    
    stmt = (
        select(
            db_models.Prediction,
            db_models.PlayerProp,
            db_models.Game
        )
        .join(db_models.PlayerProp, db_models.Prediction.player_prop_id == db_models.PlayerProp.id)
        .join(db_models.Game, db_models.PlayerProp.game_id == db_models.Game.id)
        .where(db_models.Game.game_datetime >= analysis_date)
    )
    
    result = await db.execute(stmt)
    test_records = result.all()
    logger.info(f"DEBUG: Found {len(test_records)} prediction records before joining with PlayerStat.")
    if not test_records:
        logger.warning("No prediction records found for the given date, even before joining with stats.")
        return pd.DataFrame()

    stmt = (
        select(
            db_models.Prediction,
            db_models.PlayerProp,
            db_models.Game,
            db_models.Player,
            db_models.Market,
            db_models.ModelVersion,
            db_models.PlayerStat
        )
        .join(db_models.PlayerProp, db_models.Prediction.player_prop_id == db_models.PlayerProp.id)
        .join(db_models.Game, db_models.PlayerProp.game_id == db_models.Game.id)
        .join(db_models.Player, db_models.PlayerProp.player_id == db_models.Player.id)
        .join(db_models.Market, db_models.PlayerProp.market_id == db_models.Market.id)
        .join(db_models.ModelVersion, db_models.Prediction.model_version_id == db_models.ModelVersion.id)
        .join(
            db_models.PlayerStat,
            (db_models.PlayerProp.player_id == db_models.PlayerStat.player_id) &
            (db_models.PlayerProp.game_id == db_models.PlayerStat.game_id)
        )
        .where(db_models.Game.game_datetime >= analysis_date)
        .where(db_models.PlayerStat.minutes_played > 0) # Only include players who actually played
        .options(
            selectinload(db_models.Prediction.player_prop),
            selectinload(db_models.PlayerProp.game),
            selectinload(db_models.PlayerProp.player)
        )
    )

    result = await db.execute(stmt)
    
    records = result.all()
    if not records:
        logger.warning("No matching prediction records found in the database.")
        return pd.DataFrame()

    logger.info(f"Found {len(records)} prediction records to process.")

    data_for_df = []
    for pred, prop, game, player, market, model, p_stat in records:
        data_for_df.append({
            'prediction_id': pred.id,
            'player_prop_id': prop.id,
            'model_version': model.version_name,
            'model_type': model.model_type,
            'game_id': game.id,
            'game_datetime': game.game_datetime,
            'player_id': player.id,
            'player_name': player.player_name,
            'market_key': market.key,
            'prop_outcomes': prop.outcomes,
            'predicted_value': pred.predicted_value,
            'predicted_over_prob': pred.predicted_over_probability,
            'predicted_under_prob': pred.predicted_under_probability,
            'interval_lower': pred.predicted_value_interval_lower,
            'interval_upper': pred.predicted_value_interval_upper,
            'icp_confidence_regr': pred.conformal_confidence_level_regr,
            'prediction_set': pred.prediction_set,
            'over_p_value_calibrated': pred.over_p_value_calibrated,
            'under_p_value_calibrated': pred.under_p_value_calibrated,
            'icp_confidence_clf': pred.conformal_confidence_level_clf,
            # Actual stats from PlayerStat
            'actual_points': p_stat.points,
            'actual_rebounds': p_stat.rebounds,
            'actual_assists': p_stat.assists,
            'actual_blocks': p_stat.blocks,
            'actual_steals': p_stat.steals,
        })

    df = pd.DataFrame(data_for_df)
    
    # Calculate actuals for combo stats
    df['actual_pra'] = df['actual_points'] + df['actual_rebounds'] + df['actual_assists']
    df['actual_points_rebounds'] = df['actual_points'] + df['actual_rebounds']
    df['actual_points_assists'] = df['actual_points'] + df['actual_assists']
    df['actual_rebounds_assists'] = df['actual_rebounds'] + df['actual_assists']
    df['actual_blocks_steals'] = df['actual_blocks'] + df['actual_steals']
    
    # Map market key to target_stat name
    df['target_stat'] = df['market_key'].apply(get_target_stat_from_market_key)

    # Create the final 'actual_outcome' column
    def get_actual_outcome(row):
        stat_name = row['target_stat']
        if pd.isna(stat_name):
            return None
        # Handle simple stats and combo stats
        actual_col_name = f"actual_{stat_name}"
        if actual_col_name in row:
            return row[actual_col_name]
        return None
        
    df['actual_outcome'] = df.apply(get_actual_outcome, axis=1)

    # Extract the standard line from the outcomes JSON
    def get_standard_line(outcomes):
        if not isinstance(outcomes, list): return None
        for line_set in outcomes:
            if isinstance(line_set, dict) and line_set.get('line_id') == 'standard':
                return float(line_set.get('point', 0.0))
        # Fallback to first line if no 'standard' is found
        if outcomes and isinstance(outcomes[0], dict):
            return float(outcomes[0].get('point', 0.0))
        return None

    df['standard_line'] = df['prop_outcomes'].apply(get_standard_line)
    
    df.dropna(subset=['actual_outcome', 'standard_line'], inplace=True)
    
    # Create binary outcome for classification analysis
    df['binary_outcome'] = (df['actual_outcome'] > df['standard_line']).astype(int)

    logger.info(f"Successfully fetched and processed {len(df)} records.")
    return df


def calculate_performance_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates various performance metrics from the processed predictions DataFrame.
    It segments metrics by model type (classification vs. regression).
    """
    if df.empty:
        return {}

    metrics = {
        "overall": {"total_predictions": len(df)},
        "classification": {},
        "regression": {}
    }

    # --- Classification Model Performance ---
    clf_df = df[df['model_type'] == 'classification'].copy()
    if not clf_df.empty:
        logger.info(f"Calculating metrics for {len(clf_df)} classification predictions...")
        # Ensure necessary columns are not NaN
        clf_df.dropna(subset=['over_p_value_calibrated', 'binary_outcome'], inplace=True)
        
        if not clf_df.empty:
            # Brier Score for calibrated probabilities
            brier_score = np.mean((clf_df['over_p_value_calibrated'] - clf_df['binary_outcome'])**2)
            metrics['classification']['brier_score_calibrated'] = brier_score

            # Basic ROI calculation (assuming 1 unit bet, simple strategy)
            def get_standard_odds(outcomes):
                if not isinstance(outcomes, list): return None, None
                for line_set in outcomes:
                    if isinstance(line_set, dict) and line_set.get('line_id') == 'standard':
                        return line_set.get('over_odds'), line_set.get('under_odds')
                return None, None

            clf_df['over_odds'], clf_df['under_odds'] = zip(*clf_df['prop_outcomes'].apply(get_standard_odds))
            
            # Bet 'Over' if calibrated p-value > 0.5, else 'Under'
            clf_df['bet_on'] = np.where(clf_df['over_p_value_calibrated'] > 0.5, 'over', 'under')
            clf_df['win'] = np.where(
                (clf_df['bet_on'] == 'over') & (clf_df['binary_outcome'] == 1), 1,
                np.where((clf_df['bet_on'] == 'under') & (clf_df['binary_outcome'] == 0), 1, 0)
            )
            
            def calculate_profit(row):
                if row['win'] == 1:
                    odds = row['over_odds'] if row['bet_on'] == 'over' else row['under_odds']
                    if odds is None or odds < 100: return 0 # American odds assumed
                    return (odds / 100)
                return -1 # Lost 1 unit

            clf_df['profit'] = clf_df.apply(calculate_profit, axis=1)
            total_profit = clf_df['profit'].sum()
            total_wagered = len(clf_df)
            metrics['classification']['roi'] = (total_profit / total_wagered) if total_wagered > 0 else 0
            metrics['classification']['total_bets'] = total_wagered

    # --- Regression Model Performance (ICP) ---
    regr_df = df[df['model_type'] == 'regression'].copy()
    if not regr_df.empty:
        logger.info(f"Calculating metrics for {len(regr_df)} regression predictions...")
        regr_df.dropna(subset=['interval_lower', 'interval_upper', 'actual_outcome'], inplace=True)
        
        if not regr_df.empty:
            # ICP Coverage
            regr_df['is_covered'] = (regr_df['actual_outcome'] >= regr_df['interval_lower']) & \
                                    (regr_df['actual_outcome'] <= regr_df['interval_upper'])
            coverage = regr_df['is_covered'].mean()
            metrics['regression']['icp_actual_coverage'] = coverage
            
            # Average Interval Width
            regr_df['interval_width'] = regr_df['interval_upper'] - regr_df['interval_lower']
            avg_width = regr_df['interval_width'].mean()
            metrics['regression']['average_interval_width'] = avg_width
            
            # Confidence level (assuming it's mostly constant)
            metrics['regression']['icp_target_confidence'] = regr_df['icp_confidence_regr'].median()

    return metrics

def print_performance_report(metrics: Dict[str, Any]):
    """
    Prints a formatted report of the calculated performance metrics.
    """
    logger.info("--- Model Performance Report ---")
    if not metrics or not metrics.get("overall"):
        logger.warning("Metrics dictionary is empty or invalid. Nothing to report.")
        return

    logger.info(f"Overall Predictions Analyzed: {metrics['overall'].get('total_predictions', 'N/A')}")
    logger.info("-" * 30)

    # --- Classification Report ---
    if metrics.get("classification"):
        clf_metrics = metrics["classification"]
        logger.info("[Classification Model Performance]")
        if not clf_metrics:
            logger.info("  No data for classification models.")
    else:
            logger.info(f"  Brier Score (Calibrated): {clf_metrics.get('brier_score_calibrated', 'N/A'):.4f}")
            roi = clf_metrics.get('roi', 0)
            logger.info(f"  Return on Investment (ROI): {roi:.2%}")
            logger.info(f"  Total Bets Simulated: {clf_metrics.get('total_bets', 'N/A')}")
            logger.info("-" * 30)

    # --- Regression Report ---
    if metrics.get("regression"):
        reg_metrics = metrics["regression"]
        logger.info("[Regression (ICP) Model Performance]")
        if not reg_metrics:
            logger.info("  No data for regression models.")
        else:
            target_conf = reg_metrics.get('icp_target_confidence', 0)
            actual_cov = reg_metrics.get('icp_actual_coverage', 0)
            logger.info(f"  ICP Target Confidence: {target_conf:.1%}")
            logger.info(f"  ICP Actual Coverage:   {actual_cov:.1%}")
            logger.info(f"  Average Interval Width: {reg_metrics.get('average_interval_width', 'N/A'):.2f}")
        logger.info("-" * 30)

    logger.info("--- End of Report ---")


async def main(args):
    """
    Main function to orchestrate the performance analysis process.
    """
    logger.info("Starting prediction outcome processing...")

    async with AsyncSessionLocal() as db:
        # 1. Fetch predictions and their actual outcomes
        processed_df = await fetch_processed_predictions(db, args.date)
        
        if processed_df.empty:
            logger.warning("No processed predictions found for the given date. Exiting.")
        return

        # 2. Calculate and print overall performance metrics
        logger.info("\n" + "="*50)
        logger.info("Calculating Overall Performance Metrics")
        logger.info("="*50)
        overall_metrics = calculate_performance_metrics(processed_df)
        print_performance_report(overall_metrics)
        if args.save_to_db:
            report_data = performance_schema.PerformanceReportCreate(
                report_date=datetime.utcnow(),
                segment_type="overall",
                segment_value="overall",
                metrics=overall_metrics,
            )
            await crud_performance.create_performance_report(db=db, report=report_data)
            logger.info("Saved overall performance report to the database.")

        # 3. Calculate and print segmented performance if requested
        if args.segment_by:
            for segment_col in args.segment_by:
                if segment_col not in processed_df.columns:
                    logger.warning(f"Segment column '{segment_col}' not found in data. Skipping.")
                    continue
                
                logger.info("\n" + "="*50)
                logger.info(f"Calculating Segmented Performance by: {segment_col}")
                logger.info("="*50)

                unique_segments = processed_df[segment_col].unique()
                for segment_value in unique_segments:
                    logger.info(f"\n--- Segment: {segment_col} = {segment_value} ---")
                    segment_df = processed_df[processed_df[segment_col] == segment_value]
                    segment_metrics = calculate_performance_metrics(segment_df)
                    print_performance_report(segment_metrics)
                    if args.save_to_db:
                        report_data = performance_schema.PerformanceReportCreate(
                            report_date=datetime.utcnow(),
                            segment_type=segment_col,
                            segment_value=str(segment_value),
                            metrics=segment_metrics,
                        )
                        await crud_performance.create_performance_report(db=db, report=report_data)
                        logger.info(f"Saved report for segment '{segment_value}' to the database.")

    logger.info("Prediction outcome processing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and analyze prediction outcomes.")
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="The start date (YYYY-MM-DD) for fetching games to analyze."
    )
    parser.add_argument(
        "--segment-by",
        type=str,
        nargs='*',
        help="One or more column names to segment the performance analysis by (e.g., target_stat model_version player_name)."
    )
    parser.add_argument(
        '--save-to-db',
        action='store_true',
        help="If set, saves the generated performance reports to the database."
    )
    
    args = parser.parse_args()
    args.date = date.fromisoformat(args.date)

    asyncio.run(main(args)) 