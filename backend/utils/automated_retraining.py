import argparse
import logging
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
import json
import subprocess
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from backend.db.session import SyncSessionLocal as SessionLocal
from backend.db import models as db_models
from backend.utils.comprehensive_outcome_analysis import ComprehensiveOutcomeAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelDriftDetector:
    """Detects concept drift in model performance."""
    
    def __init__(self, db: Session):
        self.db = db
        self.analyzer = ComprehensiveOutcomeAnalyzer(db)
        
    def calculate_performance_window(self, 
                                   model_version_id: str,
                                   start_date: date,
                                   end_date: date) -> Dict[str, float]:
        """Calculate performance metrics for a specific time window."""
        report = self.analyzer.generate_performance_report(
            start_date=start_date,
            end_date=end_date,
            model_version_id=model_version_id
        )
        
        betting_metrics = report.get('betting_metrics', {})
        model_metrics = report.get('model_performance', {})
        
        return {
            'roi': betting_metrics.get('roi', 0),
            'win_rate': betting_metrics.get('win_rate', 0),
            'total_predictions': betting_metrics.get('total_predictions', 0),
            'brier_score': model_metrics.get('classification_metrics', {}).get('brier_score'),
            'log_loss': model_metrics.get('classification_metrics', {}).get('log_loss'),
            'mse': model_metrics.get('regression_metrics', {}).get('mse'),
            'mae': model_metrics.get('regression_metrics', {}).get('mae'),
            'ece': model_metrics.get('calibration_analysis', {}).get('expected_calibration_error'),
            'icp_coverage': model_metrics.get('icp_metrics', {}).get('regression_intervals', {}).get('coverage_rate')
        }
    
    def detect_drift(self, 
                    model_version_id: str,
                    window_days: int = 30,
                    comparison_windows: int = 3) -> Dict[str, Any]:
        """
        Detect drift by comparing recent performance to historical performance.
        
        Args:
            model_version_id: ID of the model version to check
            window_days: Number of days per performance window
            comparison_windows: Number of historical windows to compare against
        
        Returns:
            Dictionary with drift detection results
        """
        end_date = date.today()
        
        # Calculate metrics for recent window
        recent_start = end_date - timedelta(days=window_days)
        recent_metrics = self.calculate_performance_window(
            model_version_id, recent_start, end_date
        )
        
        # Calculate metrics for historical windows
        historical_metrics = []
        for i in range(1, comparison_windows + 1):
            hist_end = recent_start - timedelta(days=1)
            hist_start = hist_end - timedelta(days=window_days)
            
            hist_metrics = self.calculate_performance_window(
                model_version_id, hist_start, hist_end
            )
            
            if hist_metrics['total_predictions'] > 0:
                historical_metrics.append(hist_metrics)
        
        if not historical_metrics:
            return {
                'drift_detected': False,
                'reason': 'Insufficient historical data for comparison',
                'recent_metrics': recent_metrics
            }
        
        # Calculate drift indicators
        drift_indicators = []
        
        # ROI drift
        hist_roi_mean = np.mean([m['roi'] for m in historical_metrics])
        hist_roi_std = np.std([m['roi'] for m in historical_metrics])
        if hist_roi_std > 0:
            roi_z_score = abs(recent_metrics['roi'] - hist_roi_mean) / hist_roi_std
            if roi_z_score > 2:
                drift_indicators.append(f"ROI drift detected (Z-score: {roi_z_score:.2f})")
        
        # Win rate drift
        hist_wr_mean = np.mean([m['win_rate'] for m in historical_metrics])
        hist_wr_std = np.std([m['win_rate'] for m in historical_metrics])
        if hist_wr_std > 0:
            wr_z_score = abs(recent_metrics['win_rate'] - hist_wr_mean) / hist_wr_std
            if wr_z_score > 2:
                drift_indicators.append(f"Win rate drift detected (Z-score: {wr_z_score:.2f})")
        
        # Model-specific metrics drift
        if recent_metrics.get('brier_score') is not None:
            hist_brier = [m['brier_score'] for m in historical_metrics if m.get('brier_score') is not None]
            if hist_brier:
                brier_increase = recent_metrics['brier_score'] / np.mean(hist_brier)
                if brier_increase > 1.2:  # 20% increase in Brier score
                    drift_indicators.append(f"Brier score degradation ({brier_increase:.1%} increase)")
        
        if recent_metrics.get('mse') is not None:
            hist_mse = [m['mse'] for m in historical_metrics if m.get('mse') is not None]
            if hist_mse:
                mse_increase = recent_metrics['mse'] / np.mean(hist_mse)
                if mse_increase > 1.2:  # 20% increase in MSE
                    drift_indicators.append(f"MSE degradation ({mse_increase:.1%} increase)")
        
        # ICP coverage drift
        if recent_metrics.get('icp_coverage') is not None:
            expected_coverage = 0.9  # Assuming 90% confidence intervals
            coverage_deviation = abs(recent_metrics['icp_coverage'] - expected_coverage)
            if coverage_deviation > 0.05:  # More than 5% deviation
                drift_indicators.append(f"ICP coverage deviation ({recent_metrics['icp_coverage']:.1%} vs expected {expected_coverage:.1%})")
        
        return {
            'drift_detected': len(drift_indicators) > 0,
            'drift_indicators': drift_indicators,
            'recent_metrics': recent_metrics,
            'historical_metrics_summary': {
                'roi_mean': hist_roi_mean,
                'roi_std': hist_roi_std,
                'win_rate_mean': hist_wr_mean,
                'win_rate_std': hist_wr_std
            }
        }


class AutomatedRetrainer:
    """Handles automated model retraining."""
    
    def __init__(self, db: Session):
        self.db = db
        self.drift_detector = ModelDriftDetector(db)
        
    def check_retraining_needed(self, target_stat: str, model_type: str) -> Tuple[bool, List[str]]:
        """
        Check if retraining is needed for a specific target stat and model type.
        
        Returns:
            Tuple of (should_retrain, reasons)
        """
        reasons = []
        
        # Get the latest model version
        latest_model = self._get_latest_model_version(target_stat, model_type)
        
        if not latest_model:
            reasons.append("No existing model found")
            return True, reasons
        
        # Check model age
        model_age_days = (datetime.utcnow() - latest_model.trained_at).days
        if model_age_days > 30:  # Retrain if model is older than 30 days
            reasons.append(f"Model is {model_age_days} days old (threshold: 30 days)")
        
        # Check drift
        drift_result = self.drift_detector.detect_drift(str(latest_model.id))
        if drift_result['drift_detected']:
            reasons.append("Performance drift detected")
            reasons.extend(drift_result['drift_indicators'])
        
        # Check data availability
        new_data_count = self._count_new_data_since(latest_model.trained_at, target_stat)
        if new_data_count > 1000:  # Retrain if significant new data available
            reasons.append(f"{new_data_count} new data points available since last training")
        
        return len(reasons) > 0, reasons
    
    def _get_latest_model_version(self, target_stat: str, model_type: str) -> Optional[db_models.ModelVersion]:
        """Get the latest model version for a target stat and model type."""
        return (
            self.db.query(db_models.ModelVersion)
            .filter(
                db_models.ModelVersion.model_type == model_type,
                db_models.ModelVersion.version_name.like(f"{target_stat}_%")
            )
            .order_by(db_models.ModelVersion.trained_at.desc())
            .first()
        )
    
    def _count_new_data_since(self, since_date: datetime, target_stat: str) -> int:
        """Count new player stats records since a given date."""
        return (
            self.db.query(func.count(db_models.PlayerStat.id))
            .join(db_models.Game)
            .filter(
                db_models.Game.game_datetime > since_date,
                getattr(db_models.PlayerStat, target_stat).isnot(None)
            )
            .scalar()
        )
    
    def trigger_retraining(self, 
                         target_stat: str, 
                         model_type: str,
                         seasons: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Trigger model retraining for a specific target stat and model type.
        
        Args:
            target_stat: The target statistic (e.g., 'points', 'rebounds')
            model_type: 'regression' or 'classification'
            seasons: Optional list of seasons to train on
        
        Returns:
            Dictionary with retraining results
        """
        logger.info(f"Triggering retraining for {target_stat} ({model_type})")
        
        # Build command with positional arguments for target_stat and model_type
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "backend" / "models" / "train_model.py"),
            target_stat,
            model_type
        ]
        
        if seasons:
            cmd.extend(["--seasons", ",".join(map(str, seasons))])
        
        # Execute training
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(PROJECT_ROOT)
            )
            
            logger.info(f"Retraining completed successfully for {target_stat} ({model_type})")
            
            # Get the new model version
            new_model = self._get_latest_model_version(target_stat, model_type)
            
            return {
                'success': True,
                'target_stat': target_stat,
                'model_type': model_type,
                'new_model_id': str(new_model.id) if new_model else None,
                'new_model_name': new_model.version_name if new_model else None,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Retraining failed for {target_stat} ({model_type}): {e}")
            return {
                'success': False,
                'target_stat': target_stat,
                'model_type': model_type,
                'error': str(e),
                'stdout': e.stdout,
                'stderr': e.stderr
            }


class RetrainingScheduler:
    """Manages scheduled retraining of models."""
    
    def __init__(self, db: Session):
        self.db = db
        self.retrainer = AutomatedRetrainer(db)
        
    def get_model_configurations(self) -> List[Dict[str, Any]]:
        """Get all model configurations that should be checked for retraining."""
        # Define the standard configurations
        # This could be loaded from a config file or database table
        configurations = []
        
        # Standard stats for both regression and classification
        standard_stats = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'three_pointers_made']
        
        for stat in standard_stats:
            configurations.append({
                'target_stat': stat,
                'model_type': 'regression',
                'enable_icp': True
            })
            configurations.append({
                'target_stat': stat,
                'model_type': 'classification',
                'enable_icp': True
            })
        
        return configurations
    
    def run_scheduled_check(self, force_retrain: bool = False) -> List[Dict[str, Any]]:
        """
        Run scheduled check for all model configurations.
        
        Args:
            force_retrain: If True, retrain all models regardless of drift detection
        
        Returns:
            List of retraining results
        """
        configurations = self.get_model_configurations()
        results = []
        
        for config in configurations:
            logger.info(f"Checking {config['target_stat']} ({config['model_type']})")
            
            if force_retrain:
                should_retrain = True
                reasons = ["Forced retraining"]
            else:
                should_retrain, reasons = self.retrainer.check_retraining_needed(
                    config['target_stat'], 
                    config['model_type']
                )
            
            if should_retrain:
                logger.info(f"Retraining needed for {config['target_stat']} ({config['model_type']}): {reasons}")
                
                # Get current season for training
                current_season = datetime.now().year
                seasons = [current_season]  # Train on current season
                
                retrain_result = self.retrainer.trigger_retraining(
                    target_stat=config['target_stat'],
                    model_type=config['model_type'],
                    seasons=seasons
                )
                
                retrain_result['reasons'] = reasons
                results.append(retrain_result)
            else:
                logger.info(f"No retraining needed for {config['target_stat']} ({config['model_type']})")
                results.append({
                    'success': True,
                    'target_stat': config['target_stat'],
                    'model_type': config['model_type'],
                    'action': 'skipped',
                    'reasons': ['No retraining needed']
                })
        
        return results
    
    def save_retraining_log(self, results: List[Dict[str, Any]], output_path: Optional[Path] = None):
        """Save retraining results to a log file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"retraining_log_{timestamp}.json")
        
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_configurations': len(results),
            'retrained': sum(1 for r in results if r.get('new_model_id') is not None),
            'skipped': sum(1 for r in results if r.get('action') == 'skipped'),
            'failed': sum(1 for r in results if not r.get('success', True)),
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"Retraining log saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Automated model retraining with drift detection.")
    parser.add_argument("--check-drift", action="store_true", help="Check for model drift")
    parser.add_argument("--run-scheduled", action="store_true", help="Run scheduled retraining check")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining of all models")
    parser.add_argument("--target-stat", type=str, help="Specific target stat to check/retrain")
    parser.add_argument("--model-type", type=str, choices=['regression', 'classification'], 
                       help="Specific model type to check/retrain")
    parser.add_argument("--model-version-id", type=str, help="Specific model version ID for drift detection")
    parser.add_argument("--output", type=str, help="Output file path for logs")
    
    args = parser.parse_args()
    
    db = SessionLocal()
    try:
        if args.check_drift and args.model_version_id:
            # Check drift for specific model
            detector = ModelDriftDetector(db)
            drift_result = detector.detect_drift(args.model_version_id)
            
            print("\n=== DRIFT DETECTION RESULTS ===")
            print(f"Drift Detected: {drift_result['drift_detected']}")
            if drift_result['drift_detected']:
                print("\nDrift Indicators:")
                for indicator in drift_result['drift_indicators']:
                    print(f"  - {indicator}")
            
            print("\nRecent Performance Metrics:")
            for metric, value in drift_result['recent_metrics'].items():
                if value is not None:
                    print(f"  {metric}: {value:.3f}")
        
        elif args.run_scheduled or args.force_retrain:
            # Run scheduled retraining check
            scheduler = RetrainingScheduler(db)
            results = scheduler.run_scheduled_check(force_retrain=args.force_retrain)
            
            # Save log
            output_path = Path(args.output) if args.output else None
            scheduler.save_retraining_log(results, output_path)
            
            # Print summary
            print("\n=== RETRAINING SUMMARY ===")
            print(f"Total Configurations: {len(results)}")
            print(f"Retrained: {sum(1 for r in results if r.get('new_model_id') is not None)}")
            print(f"Skipped: {sum(1 for r in results if r.get('action') == 'skipped')}")
            print(f"Failed: {sum(1 for r in results if not r.get('success', True))}")
            
            # Print details for retrained models
            retrained = [r for r in results if r.get('new_model_id') is not None]
            if retrained:
                print("\nRetrained Models:")
                for r in retrained:
                    print(f"  - {r['target_stat']} ({r['model_type']}): {r['new_model_name']}")
        
        elif args.target_stat and args.model_type:
            # Check/retrain specific model
            retrainer = AutomatedRetrainer(db)
            should_retrain, reasons = retrainer.check_retraining_needed(args.target_stat, args.model_type)
            
            print(f"\n=== {args.target_stat.upper()} ({args.model_type}) ===")
            print(f"Retraining Needed: {should_retrain}")
            if reasons:
                print("Reasons:")
                for reason in reasons:
                    print(f"  - {reason}")
            
            if should_retrain and input("\nProceed with retraining? (y/n): ").lower() == 'y':
                result = retrainer.trigger_retraining(
                    args.target_stat, 
                    args.model_type,
                    seasons=[datetime.now().year]
                )
                
                if result['success']:
                    print(f"\nRetraining successful! New model: {result['new_model_name']}")
                else:
                    print(f"\nRetraining failed: {result.get('error', 'Unknown error')}")
        
        else:
            parser.print_help()
            
    finally:
        db.close()


if __name__ == "__main__":
    main() 