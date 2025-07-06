import argparse
import logging
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy import and_, func, case
import json

# Project specific imports
import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from backend.db.session import SyncSessionLocal as SessionLocal
from backend.db import models as db_models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveOutcomeAnalyzer:
    """Analyzes prediction outcomes with detailed betting and model performance metrics."""
    
    def __init__(self, db: Session):
        self.db = db
        
    def calculate_betting_metrics(self, predictions: List[db_models.Prediction]) -> Dict[str, Any]:
        """Calculate comprehensive betting performance metrics."""
        metrics = {
            'total_predictions': len(predictions),
            'predictions_with_outcomes': 0,
            'correct_predictions': 0,
            'pushes': 0,
            'total_units_wagered': 0,
            'total_units_won': 0,
            'total_units_lost': 0,
            'roi': 0.0,
            'win_rate': 0.0,
            'by_market': {},
            'by_confidence': {},
            'by_model_version': {},
            'by_time_period': {}
        }
        
        for pred in predictions:
            if pred.outcome is None:
                continue
                
            metrics['predictions_with_outcomes'] += 1
            
            # Assume 1 unit bet per prediction for now
            # In reality, this could be based on Kelly Criterion or confidence
            unit_bet = 1.0
            metrics['total_units_wagered'] += unit_bet
            
            # Get the odds from the player prop
            if pred.player_prop and pred.player_prop.outcomes:
                # Extract odds for the predicted side
                predicted_over = pred.predicted_over_probability > 0.5
                
                # Find the appropriate odds from outcomes
                odds_data = self._extract_odds_from_outcomes(pred.player_prop.outcomes, predicted_over)
                
                if odds_data:
                    # Calculate profit/loss
                    if pred.outcome == "PUSH":
                        metrics['pushes'] += 1
                        # Push returns the bet
                    elif (predicted_over and pred.outcome == "OVER") or (not predicted_over and pred.outcome == "UNDER"):
                        metrics['correct_predictions'] += 1
                        # Calculate winnings based on American odds
                        profit = self._calculate_profit(unit_bet, odds_data['odds'])
                        metrics['total_units_won'] += profit
                    else:
                        # Lost bet
                        metrics['total_units_lost'] += unit_bet
            
            # Track by market type
            if pred.player_prop and pred.player_prop.market:
                market_key = pred.player_prop.market.key
                if market_key not in metrics['by_market']:
                    metrics['by_market'][market_key] = {
                        'total': 0, 'correct': 0, 'roi': 0.0, 'units_won': 0, 'units_lost': 0
                    }
                metrics['by_market'][market_key]['total'] += 1
                if pred.outcome not in [None, "PUSH"]:
                    if (predicted_over and pred.outcome == "OVER") or (not predicted_over and pred.outcome == "UNDER"):
                        metrics['by_market'][market_key]['correct'] += 1
            
            # Track by confidence level (using predicted probability)
            confidence = max(pred.predicted_over_probability or 0, pred.predicted_under_probability or 0)
            confidence_bucket = f"{int(confidence * 10) * 10}-{int(confidence * 10) * 10 + 10}%"
            if confidence_bucket not in metrics['by_confidence']:
                metrics['by_confidence'][confidence_bucket] = {
                    'total': 0, 'correct': 0, 'roi': 0.0
                }
            metrics['by_confidence'][confidence_bucket]['total'] += 1
            if pred.outcome not in [None, "PUSH"] and (
                (predicted_over and pred.outcome == "OVER") or (not predicted_over and pred.outcome == "UNDER")
            ):
                metrics['by_confidence'][confidence_bucket]['correct'] += 1
            
            # Track by model version
            if pred.model_version:
                model_name = pred.model_version.version_name
                if model_name not in metrics['by_model_version']:
                    metrics['by_model_version'][model_name] = {
                        'total': 0, 'correct': 0, 'roi': 0.0
                    }
                metrics['by_model_version'][model_name]['total'] += 1
                if pred.outcome not in [None, "PUSH"] and (
                    (predicted_over and pred.outcome == "OVER") or (not predicted_over and pred.outcome == "UNDER")
                ):
                    metrics['by_model_version'][model_name]['correct'] += 1
        
        # Calculate overall metrics
        if metrics['predictions_with_outcomes'] > 0:
            metrics['win_rate'] = metrics['correct_predictions'] / (metrics['predictions_with_outcomes'] - metrics['pushes'])
            
        if metrics['total_units_wagered'] > 0:
            net_profit = metrics['total_units_won'] - metrics['total_units_lost']
            metrics['roi'] = (net_profit / metrics['total_units_wagered']) * 100
            
        # Calculate ROI for each segment
        for market_data in metrics['by_market'].values():
            if market_data['total'] > 0:
                market_data['win_rate'] = market_data['correct'] / market_data['total']
                
        for conf_data in metrics['by_confidence'].values():
            if conf_data['total'] > 0:
                conf_data['win_rate'] = conf_data['correct'] / conf_data['total']
                
        for model_data in metrics['by_model_version'].values():
            if model_data['total'] > 0:
                model_data['win_rate'] = model_data['correct'] / model_data['total']
        
        return metrics
    
    def calculate_model_performance_metrics(self, predictions: List[db_models.Prediction]) -> Dict[str, Any]:
        """Calculate detailed model performance metrics including calibration."""
        metrics = {
            'regression_metrics': {},
            'classification_metrics': {},
            'icp_metrics': {},
            'calibration_analysis': {}
        }
        
        # Separate predictions by model type
        regression_preds = []
        classification_preds = []
        
        for pred in predictions:
            if pred.outcome is None or pred.model_version is None:
                continue
                
            if 'regression' in pred.model_version.version_name.lower():
                regression_preds.append(pred)
            elif 'classification' in pred.model_version.version_name.lower():
                classification_preds.append(pred)
        
        # Regression metrics
        if regression_preds:
            actual_values = [p.actual_value for p in regression_preds if p.actual_value is not None]
            predicted_values = [p.predicted_value for p in regression_preds if p.predicted_value is not None]
            
            if len(actual_values) == len(predicted_values) and len(actual_values) > 0:
                metrics['regression_metrics'] = {
                    'mse': np.mean((np.array(actual_values) - np.array(predicted_values)) ** 2),
                    'mae': np.mean(np.abs(np.array(actual_values) - np.array(predicted_values))),
                    'rmse': np.sqrt(np.mean((np.array(actual_values) - np.array(predicted_values)) ** 2)),
                    'r2': self._calculate_r2(actual_values, predicted_values)
                }
        
        # Classification metrics and calibration
        if classification_preds:
            # Extract probabilities and outcomes
            probs_and_outcomes = []
            for pred in classification_preds:
                if pred.predicted_over_probability is not None and pred.outcome in ["OVER", "UNDER"]:
                    actual_over = 1 if pred.outcome == "OVER" else 0
                    probs_and_outcomes.append({
                        'predicted_prob': pred.predicted_over_probability,
                        'actual': actual_over
                    })
            
            if probs_and_outcomes:
                df = pd.DataFrame(probs_and_outcomes)
                
                # Brier score
                brier_score = np.mean((df['predicted_prob'] - df['actual']) ** 2)
                
                # Log loss
                epsilon = 1e-15
                df['predicted_prob'] = df['predicted_prob'].clip(epsilon, 1 - epsilon)
                log_loss = -np.mean(
                    df['actual'] * np.log(df['predicted_prob']) + 
                    (1 - df['actual']) * np.log(1 - df['predicted_prob'])
                )
                
                metrics['classification_metrics'] = {
                    'brier_score': brier_score,
                    'log_loss': log_loss,
                    'total_predictions': len(probs_and_outcomes)
                }
                
                # Calibration analysis
                calibration_bins = self._calculate_calibration_bins(df)
                metrics['calibration_analysis'] = calibration_bins
        
        # ICP metrics
        icp_coverage = self._analyze_icp_coverage(predictions)
        if icp_coverage:
            metrics['icp_metrics'] = icp_coverage
        
        return metrics
    
    def _calculate_calibration_bins(self, df: pd.DataFrame, n_bins: int = 10) -> Dict[str, Any]:
        """Calculate calibration bins for probability calibration analysis."""
        df['prob_bin'] = pd.cut(df['predicted_prob'], bins=n_bins, labels=False)
        
        calibration_data = []
        for bin_idx in range(n_bins):
            bin_data = df[df['prob_bin'] == bin_idx]
            if len(bin_data) > 0:
                calibration_data.append({
                    'bin_index': bin_idx,
                    'bin_range': f"{bin_idx * 10}-{(bin_idx + 1) * 10}%",
                    'mean_predicted_prob': bin_data['predicted_prob'].mean(),
                    'actual_frequency': bin_data['actual'].mean(),
                    'count': len(bin_data),
                    'calibration_error': abs(bin_data['predicted_prob'].mean() - bin_data['actual'].mean())
                })
        
        # Expected Calibration Error (ECE)
        total_count = len(df)
        ece = sum(
            (bin_info['count'] / total_count) * bin_info['calibration_error'] 
            for bin_info in calibration_data
        )
        
        return {
            'bins': calibration_data,
            'expected_calibration_error': ece,
            'max_calibration_error': max(bin_info['calibration_error'] for bin_info in calibration_data) if calibration_data else 0
        }
    
    def _analyze_icp_coverage(self, predictions: List[db_models.Prediction]) -> Dict[str, Any]:
        """Analyze ICP interval coverage and prediction set accuracy."""
        regression_coverage = {
            'total_with_intervals': 0,
            'covered': 0,
            'coverage_rate': 0.0,
            'by_confidence_level': {}
        }
        
        classification_sets = {
            'total_with_sets': 0,
            'correct_in_set': 0,
            'empty_sets': 0,
            'single_prediction_sets': 0,
            'both_prediction_sets': 0,
            'accuracy': 0.0
        }
        
        for pred in predictions:
            # Regression ICP analysis
            if (pred.predicted_value_interval_lower is not None and 
                pred.predicted_value_interval_upper is not None and
                pred.actual_value is not None):
                
                regression_coverage['total_with_intervals'] += 1
                
                if (pred.predicted_value_interval_lower <= pred.actual_value <= 
                    pred.predicted_value_interval_upper):
                    regression_coverage['covered'] += 1
                
                # Track by confidence level
                conf_level = pred.conformal_confidence_level_regr or 0.9
                conf_key = f"{int(conf_level * 100)}%"
                if conf_key not in regression_coverage['by_confidence_level']:
                    regression_coverage['by_confidence_level'][conf_key] = {
                        'total': 0, 'covered': 0, 'coverage_rate': 0.0
                    }
                regression_coverage['by_confidence_level'][conf_key]['total'] += 1
                if (pred.predicted_value_interval_lower <= pred.actual_value <= 
                    pred.predicted_value_interval_upper):
                    regression_coverage['by_confidence_level'][conf_key]['covered'] += 1
            
            # Classification ICP analysis
            if pred.prediction_set is not None and pred.outcome in ["OVER", "UNDER"]:
                classification_sets['total_with_sets'] += 1
                
                if len(pred.prediction_set) == 0:
                    classification_sets['empty_sets'] += 1
                elif len(pred.prediction_set) == 1:
                    classification_sets['single_prediction_sets'] += 1
                elif len(pred.prediction_set) == 2:
                    classification_sets['both_prediction_sets'] += 1
                
                if pred.outcome in pred.prediction_set:
                    classification_sets['correct_in_set'] += 1
        
        # Calculate rates
        if regression_coverage['total_with_intervals'] > 0:
            regression_coverage['coverage_rate'] = (
                regression_coverage['covered'] / regression_coverage['total_with_intervals']
            )
            
            for conf_data in regression_coverage['by_confidence_level'].values():
                if conf_data['total'] > 0:
                    conf_data['coverage_rate'] = conf_data['covered'] / conf_data['total']
        
        if classification_sets['total_with_sets'] > 0:
            classification_sets['accuracy'] = (
                classification_sets['correct_in_set'] / classification_sets['total_with_sets']
            )
        
        return {
            'regression_intervals': regression_coverage,
            'classification_sets': classification_sets
        }
    
    def _extract_odds_from_outcomes(self, outcomes: List[Dict], is_over: bool) -> Optional[Dict[str, Any]]:
        """Extract odds information from outcomes JSON."""
        if not outcomes or not isinstance(outcomes, list):
            return None
            
        for outcome in outcomes:
            if isinstance(outcome, dict) and 'options' in outcome:
                for option in outcome['options']:
                    if isinstance(option, dict) and 'name' in option and 'price' in option:
                        if (is_over and option['name'] == 'Over') or (not is_over and option['name'] == 'Under'):
                            return {
                                'odds': option['price'],
                                'line': outcome.get('point')
                            }
        return None
    
    def _calculate_profit(self, bet_amount: float, american_odds: int) -> float:
        """Calculate profit from American odds."""
        if american_odds > 0:
            return bet_amount * (american_odds / 100)
        else:
            return bet_amount * (100 / abs(american_odds))
    
    def _calculate_r2(self, actual: List[float], predicted: List[float]) -> float:
        """Calculate R-squared value."""
        actual_array = np.array(actual)
        predicted_array = np.array(predicted)
        
        ss_res = np.sum((actual_array - predicted_array) ** 2)
        ss_tot = np.sum((actual_array - np.mean(actual_array)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    def generate_performance_report(self, 
                                  start_date: Optional[date] = None,
                                  end_date: Optional[date] = None,
                                  model_version_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        # Build query
        query = self.db.query(db_models.Prediction).options(
            selectinload(db_models.Prediction.player_prop).selectinload(db_models.PlayerProp.game),
            selectinload(db_models.Prediction.player_prop).selectinload(db_models.PlayerProp.market),
            selectinload(db_models.Prediction.model_version)
        )
        
        # Apply filters
        if start_date:
            query = query.join(
                db_models.PlayerProp, 
                db_models.Prediction.player_prop_id == db_models.PlayerProp.id
            ).join(
                db_models.Game, 
                db_models.PlayerProp.game_id == db_models.Game.id
            ).filter(db_models.Game.game_datetime >= datetime.combine(start_date, datetime.min.time()))
        
        if end_date:
            if not start_date:  # Need to join if not already done
                query = query.join(
                    db_models.PlayerProp, 
                    db_models.Prediction.player_prop_id == db_models.PlayerProp.id
                ).join(
                    db_models.Game, 
                    db_models.PlayerProp.game_id == db_models.Game.id
                )
            query = query.filter(db_models.Game.game_datetime <= datetime.combine(end_date, datetime.max.time()))
        
        if model_version_id:
            query = query.filter(db_models.Prediction.model_version_id == model_version_id)
        
        predictions = query.all()
        
        # Generate comprehensive report
        report = {
            'summary': {
                'total_predictions': len(predictions),
                'date_range': {
                    'start': start_date.isoformat() if start_date else 'all',
                    'end': end_date.isoformat() if end_date else 'all'
                },
                'generated_at': datetime.utcnow().isoformat()
            },
            'betting_metrics': self.calculate_betting_metrics(predictions),
            'model_performance': self.calculate_model_performance_metrics(predictions),
            'recommendations': self._generate_recommendations(predictions)
        }
        
        return report
    
    def _generate_recommendations(self, predictions: List[db_models.Prediction]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Analyze betting metrics
        betting_metrics = self.calculate_betting_metrics(predictions)
        
        if betting_metrics['roi'] < -5:
            recommendations.append("Overall ROI is negative. Consider adjusting bet selection criteria or confidence thresholds.")
        
        # Check market-specific performance
        for market, data in betting_metrics['by_market'].items():
            if data['total'] > 20 and data.get('win_rate', 0) < 0.45:
                recommendations.append(f"Poor performance on {market} props (win rate: {data.get('win_rate', 0):.2%}). Consider removing or retraining.")
        
        # Check confidence calibration
        model_metrics = self.calculate_model_performance_metrics(predictions)
        if 'calibration_analysis' in model_metrics and model_metrics['calibration_analysis']:
            ece = model_metrics['calibration_analysis'].get('expected_calibration_error', 0)
            if ece > 0.1:
                recommendations.append(f"Model calibration is poor (ECE: {ece:.3f}). Consider recalibration or retraining.")
        
        # Check ICP coverage
        if 'icp_metrics' in model_metrics:
            reg_coverage = model_metrics['icp_metrics'].get('regression_intervals', {})
            if reg_coverage.get('coverage_rate', 0) < 0.85 and reg_coverage.get('total_with_intervals', 0) > 10:
                recommendations.append(f"ICP interval coverage ({reg_coverage['coverage_rate']:.1%}) is below expected. Consider adjusting calibration set or method.")
        
        return recommendations


def save_analysis_report(report: Dict[str, Any], output_path: Path):
    """Save analysis report to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Analysis report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive outcome analysis for WNBA predictions.")
    parser.add_argument("--start-date", type=str, help="Start date for analysis (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for analysis (YYYY-MM-DD)")
    parser.add_argument("--model-version", type=str, help="Specific model version ID to analyze")
    parser.add_argument("--output", type=str, help="Output file path for the report")
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid start date format: {args.start_date}")
            return
    
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid end date format: {args.end_date}")
            return
    
    # Run analysis
    db = SessionLocal()
    try:
        analyzer = ComprehensiveOutcomeAnalyzer(db)
        report = analyzer.generate_performance_report(
            start_date=start_date,
            end_date=end_date,
            model_version_id=args.model_version
        )
        
        # Save report
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"analysis_report_{timestamp}.json")
        
        save_analysis_report(report, output_path)
        
        # Print summary
        print("\n=== PERFORMANCE SUMMARY ===")
        print(f"Total Predictions: {report['summary']['total_predictions']}")
        print(f"Overall ROI: {report['betting_metrics']['roi']:.2f}%")
        print(f"Win Rate: {report['betting_metrics']['win_rate']:.2%}")
        
        if report['recommendations']:
            print("\n=== RECOMMENDATIONS ===")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
        
    finally:
        db.close()


if __name__ == "__main__":
    main() 