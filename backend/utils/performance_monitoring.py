import argparse
import logging
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
import json
from pathlib import Path

# Add project root to Python path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from backend.db.session import SyncSessionLocal as SessionLocal
from backend.db import models as db_models
from backend.utils.comprehensive_outcome_analysis import ComprehensiveOutcomeAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PerformanceMonitor:
    """Monitors and visualizes model performance over time."""
    
    def __init__(self, db: Session):
        self.db = db
        self.analyzer = ComprehensiveOutcomeAnalyzer(db)
        
    def generate_time_series_metrics(self, 
                                   model_version_id: Optional[str] = None,
                                   days_back: int = 90,
                                   window_size: int = 7) -> pd.DataFrame:
        """Generate time series of performance metrics."""
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        metrics_over_time = []
        
        current_date = start_date
        while current_date <= end_date - timedelta(days=window_size):
            window_end = current_date + timedelta(days=window_size)
            
            report = self.analyzer.generate_performance_report(
                start_date=current_date,
                end_date=window_end,
                model_version_id=model_version_id
            )
            
            betting_metrics = report.get('betting_metrics', {})
            model_metrics = report.get('model_performance', {})
            
            if betting_metrics.get('predictions_with_outcomes', 0) > 0:
                metrics_over_time.append({
                    'date': current_date,
                    'window_end': window_end,
                    'roi': betting_metrics.get('roi', 0),
                    'win_rate': betting_metrics.get('win_rate', 0),
                    'total_predictions': betting_metrics.get('total_predictions', 0),
                    'predictions_with_outcomes': betting_metrics.get('predictions_with_outcomes', 0),
                    'brier_score': model_metrics.get('classification_metrics', {}).get('brier_score'),
                    'log_loss': model_metrics.get('classification_metrics', {}).get('log_loss'),
                    'mse': model_metrics.get('regression_metrics', {}).get('mse'),
                    'mae': model_metrics.get('regression_metrics', {}).get('mae'),
                    'ece': model_metrics.get('calibration_analysis', {}).get('expected_calibration_error'),
                    'icp_coverage': model_metrics.get('icp_metrics', {}).get('regression_intervals', {}).get('coverage_rate')
                })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(metrics_over_time)
    
    def plot_performance_trends(self, 
                              metrics_df: pd.DataFrame,
                              output_dir: Path,
                              model_name: str = "Model"):
        """Create performance trend visualizations."""
        if metrics_df.empty:
            logger.warning("No data to plot")
            return
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. ROI and Win Rate Over Time
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # ROI plot
        ax1.plot(metrics_df['date'], metrics_df['roi'], marker='o', linewidth=2)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_ylabel('ROI (%)')
        ax1.set_title(f'{model_name} - ROI Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Add rolling average
        if len(metrics_df) > 7:
            rolling_roi = metrics_df['roi'].rolling(window=7, center=True).mean()
            ax1.plot(metrics_df['date'], rolling_roi, 'g--', label='7-day MA', alpha=0.7)
            ax1.legend()
        
        # Win rate plot
        ax2.plot(metrics_df['date'], metrics_df['win_rate'] * 100, marker='o', linewidth=2, color='orange')
        ax2.axhline(y=52.38, color='r', linestyle='--', alpha=0.5, label='Break-even (52.38%)')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_xlabel('Date')
        ax2.set_title(f'{model_name} - Win Rate Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'roi_winrate_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model Performance Metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Brier Score (Classification)
        if 'brier_score' in metrics_df.columns and metrics_df['brier_score'].notna().any():
            ax = axes[0, 0]
            mask = metrics_df['brier_score'].notna()
            ax.plot(metrics_df.loc[mask, 'date'], metrics_df.loc[mask, 'brier_score'], 
                   marker='o', linewidth=2, color='purple')
            ax.set_ylabel('Brier Score')
            ax.set_title('Brier Score Over Time (Lower is Better)')
            ax.grid(True, alpha=0.3)
        
        # MSE (Regression)
        if 'mse' in metrics_df.columns and metrics_df['mse'].notna().any():
            ax = axes[0, 1]
            mask = metrics_df['mse'].notna()
            ax.plot(metrics_df.loc[mask, 'date'], metrics_df.loc[mask, 'mse'], 
                   marker='o', linewidth=2, color='red')
            ax.set_ylabel('MSE')
            ax.set_title('Mean Squared Error Over Time (Lower is Better)')
            ax.grid(True, alpha=0.3)
        
        # ECE (Calibration)
        if 'ece' in metrics_df.columns and metrics_df['ece'].notna().any():
            ax = axes[1, 0]
            mask = metrics_df['ece'].notna()
            ax.plot(metrics_df.loc[mask, 'date'], metrics_df.loc[mask, 'ece'], 
                   marker='o', linewidth=2, color='green')
            ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Poor calibration threshold')
            ax.set_ylabel('ECE')
            ax.set_xlabel('Date')
            ax.set_title('Expected Calibration Error Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # ICP Coverage
        if 'icp_coverage' in metrics_df.columns and metrics_df['icp_coverage'].notna().any():
            ax = axes[1, 1]
            mask = metrics_df['icp_coverage'].notna()
            ax.plot(metrics_df.loc[mask, 'date'], metrics_df.loc[mask, 'icp_coverage'] * 100, 
                   marker='o', linewidth=2, color='brown')
            ax.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='Expected 90% coverage')
            ax.set_ylabel('Coverage (%)')
            ax.set_xlabel('Date')
            ax.set_title('ICP Interval Coverage Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_metrics_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Volume and Activity
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(metrics_df['date'], metrics_df['predictions_with_outcomes'], 
               alpha=0.6, label='Predictions with Outcomes')
        ax.bar(metrics_df['date'], 
               metrics_df['total_predictions'] - metrics_df['predictions_with_outcomes'],
               bottom=metrics_df['predictions_with_outcomes'],
               alpha=0.6, label='Pending Predictions')
        
        ax.set_ylabel('Number of Predictions')
        ax.set_xlabel('Date')
        ax.set_title(f'{model_name} - Prediction Volume Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_volume.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance trend plots saved to {output_dir}")
    
    def generate_model_comparison_report(self, 
                                       target_stat: str,
                                       days_back: int = 30) -> pd.DataFrame:
        """Compare performance of different model versions for the same target stat."""
        # Get all model versions for the target stat
        model_versions = (
            self.db.query(db_models.ModelVersion)
            .filter(db_models.ModelVersion.version_name.like(f"{target_stat}_%"))
            .order_by(db_models.ModelVersion.trained_at.desc())
            .limit(5)  # Compare last 5 versions
            .all()
        )
        
        comparison_data = []
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        for model in model_versions:
            report = self.analyzer.generate_performance_report(
                start_date=start_date,
                end_date=end_date,
                model_version_id=str(model.id)
            )
            
            betting_metrics = report.get('betting_metrics', {})
            model_metrics = report.get('model_performance', {})
            
            comparison_data.append({
                'model_id': str(model.id),
                'model_name': model.version_name,
                'model_type': model.model_type,
                'trained_at': model.trained_at,
                'total_predictions': betting_metrics.get('total_predictions', 0),
                'roi': betting_metrics.get('roi', 0),
                'win_rate': betting_metrics.get('win_rate', 0),
                'brier_score': model_metrics.get('classification_metrics', {}).get('brier_score'),
                'mse': model_metrics.get('regression_metrics', {}).get('mse'),
                'ece': model_metrics.get('calibration_analysis', {}).get('expected_calibration_error')
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, output_dir: Path):
        """Create model comparison visualizations."""
        if comparison_df.empty:
            logger.warning("No data for model comparison")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate by model type
        regression_models = comparison_df[comparison_df['model_type'] == 'regression']
        classification_models = comparison_df[comparison_df['model_type'] == 'classification']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ROI Comparison
        ax = axes[0, 0]
        if not comparison_df.empty:
            comparison_df.plot(x='model_name', y='roi', kind='bar', ax=ax, legend=False)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_ylabel('ROI (%)')
            ax.set_title('ROI by Model Version')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Win Rate Comparison
        ax = axes[0, 1]
        if not comparison_df.empty:
            comparison_df.plot(x='model_name', y='win_rate', kind='bar', ax=ax, legend=False, color='orange')
            ax.axhline(y=0.5238, color='r', linestyle='--', alpha=0.5, label='Break-even')
            ax.set_ylabel('Win Rate')
            ax.set_title('Win Rate by Model Version')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend()
        
        # Brier Score (Classification only)
        ax = axes[1, 0]
        if not classification_models.empty and 'brier_score' in classification_models.columns:
            mask = classification_models['brier_score'].notna()
            classification_models[mask].plot(x='model_name', y='brier_score', kind='bar', 
                                           ax=ax, legend=False, color='purple')
            ax.set_ylabel('Brier Score')
            ax.set_title('Brier Score by Classification Model (Lower is Better)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # MSE (Regression only)
        ax = axes[1, 1]
        if not regression_models.empty and 'mse' in regression_models.columns:
            mask = regression_models['mse'].notna()
            regression_models[mask].plot(x='model_name', y='mse', kind='bar', 
                                       ax=ax, legend=False, color='red')
            ax.set_ylabel('MSE')
            ax.set_title('MSE by Regression Model (Lower is Better)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plots saved to {output_dir}")
    
    def generate_alert_report(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """Generate alerts for performance issues."""
        alerts = []
        
        # Get all active model versions
        active_models = (
            self.db.query(db_models.ModelVersion)
            .order_by(db_models.ModelVersion.trained_at.desc())
            .all()
        )
        
        # Group by target stat to get latest of each
        latest_by_stat = {}
        for model in active_models:
            # Extract target stat from version name
            parts = model.version_name.split('_')
            if len(parts) >= 2:
                target_stat = parts[0]
                if target_stat not in latest_by_stat:
                    latest_by_stat[target_stat] = []
                latest_by_stat[target_stat].append(model)
        
        # Check each model's recent performance
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        for target_stat, models in latest_by_stat.items():
            # Get the most recent model for each type
            regression_model = next((m for m in models if m.model_type == 'regression'), None)
            classification_model = next((m for m in models if m.model_type == 'classification'), None)
            
            for model in [regression_model, classification_model]:
                if not model:
                    continue
                
                report = self.analyzer.generate_performance_report(
                    start_date=start_date,
                    end_date=end_date,
                    model_version_id=str(model.id)
                )
                
                betting_metrics = report.get('betting_metrics', {})
                model_metrics = report.get('model_performance', {})
                
                # Check for alerts
                if betting_metrics.get('roi', 0) < -10:
                    alerts.append({
                        'severity': 'HIGH',
                        'model': model.version_name,
                        'issue': f"ROI below -10% ({betting_metrics['roi']:.1f}%)",
                        'recommendation': 'Consider pausing predictions or retraining'
                    })
                
                if betting_metrics.get('win_rate', 1) < 0.45:
                    alerts.append({
                        'severity': 'MEDIUM',
                        'model': model.version_name,
                        'issue': f"Win rate below 45% ({betting_metrics['win_rate']:.1%})",
                        'recommendation': 'Review model performance and consider retraining'
                    })
                
                ece = model_metrics.get('calibration_analysis', {}).get('expected_calibration_error')
                if ece and ece > 0.15:
                    alerts.append({
                        'severity': 'MEDIUM',
                        'model': model.version_name,
                        'issue': f"Poor calibration (ECE: {ece:.3f})",
                        'recommendation': 'Model confidence is miscalibrated, consider recalibration'
                    })
                
                icp_coverage = model_metrics.get('icp_metrics', {}).get('regression_intervals', {}).get('coverage_rate')
                if icp_coverage and abs(icp_coverage - 0.9) > 0.1:
                    alerts.append({
                        'severity': 'LOW',
                        'model': model.version_name,
                        'issue': f"ICP coverage deviation ({icp_coverage:.1%} vs expected 90%)",
                        'recommendation': 'Review ICP calibration methodology'
                    })
        
        return alerts


def main():
    parser = argparse.ArgumentParser(description="Monitor and visualize model performance.")
    parser.add_argument("--days-back", type=int, default=90, help="Number of days to analyze")
    parser.add_argument("--window-size", type=int, default=7, help="Window size for rolling metrics")
    parser.add_argument("--model-version-id", type=str, help="Specific model version to monitor")
    parser.add_argument("--target-stat", type=str, help="Target stat for model comparison")
    parser.add_argument("--output-dir", type=str, default="monitoring_output", help="Output directory for plots")
    parser.add_argument("--generate-alerts", action="store_true", help="Generate performance alerts")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    db = SessionLocal()
    try:
        monitor = PerformanceMonitor(db)
        
        if args.generate_alerts:
            # Generate alerts
            alerts = monitor.generate_alert_report(days_back=7)
            
            if alerts:
                print("\n=== PERFORMANCE ALERTS ===")
                for alert in sorted(alerts, key=lambda x: ['HIGH', 'MEDIUM', 'LOW'].index(x['severity'])):
                    print(f"\n[{alert['severity']}] {alert['model']}")
                    print(f"  Issue: {alert['issue']}")
                    print(f"  Recommendation: {alert['recommendation']}")
            else:
                print("\n✓ No performance alerts")
            
            # Save alerts to file
            with open(output_dir / 'alerts.json', 'w') as f:
                json.dump(alerts, f, indent=2)
        
        elif args.target_stat:
            # Model comparison for specific target stat
            print(f"\nGenerating model comparison for {args.target_stat}...")
            comparison_df = monitor.generate_model_comparison_report(
                args.target_stat, 
                days_back=args.days_back
            )
            
            if not comparison_df.empty:
                # Save comparison data
                comparison_df.to_csv(output_dir / f'{args.target_stat}_model_comparison.csv', index=False)
                
                # Create plots
                monitor.plot_model_comparison(comparison_df, output_dir / args.target_stat)
                
                # Print summary
                print("\n=== MODEL COMPARISON SUMMARY ===")
                print(comparison_df[['model_name', 'model_type', 'roi', 'win_rate', 'total_predictions']].to_string())
            else:
                print(f"No models found for {args.target_stat}")
        
        else:
            # Time series monitoring
            model_name = "All Models"
            if args.model_version_id:
                model = db.query(db_models.ModelVersion).filter(
                    db_models.ModelVersion.id == args.model_version_id
                ).first()
                if model:
                    model_name = model.version_name
            
            print(f"\nGenerating performance metrics for {model_name}...")
            metrics_df = monitor.generate_time_series_metrics(
                model_version_id=args.model_version_id,
                days_back=args.days_back,
                window_size=args.window_size
            )
            
            if not metrics_df.empty:
                # Save metrics data
                metrics_df.to_csv(output_dir / 'performance_metrics.csv', index=False)
                
                # Create plots
                monitor.plot_performance_trends(metrics_df, output_dir, model_name)
                
                # Print summary statistics
                print("\n=== PERFORMANCE SUMMARY ===")
                print(f"Period: {metrics_df['date'].min()} to {metrics_df['date'].max()}")
                print(f"Average ROI: {metrics_df['roi'].mean():.2f}%")
                print(f"Average Win Rate: {metrics_df['win_rate'].mean():.2%}")
                print(f"Total Predictions: {metrics_df['total_predictions'].sum()}")
                
                # Recent trend
                if len(metrics_df) > 7:
                    recent = metrics_df.tail(7)
                    print(f"\nLast 7 Days:")
                    print(f"  ROI: {recent['roi'].mean():.2f}%")
                    print(f"  Win Rate: {recent['win_rate'].mean():.2%}")
            else:
                print("No performance data available for the specified period")
        
    finally:
        db.close()


if __name__ == "__main__":
    main() 