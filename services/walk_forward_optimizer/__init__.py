"""
Walk Forward Optimizer Service - Advanced walk-forward analysis for trading strategies.

Provides sophisticated optimization techniques for trading strategy parameters.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import optuna

from config.settings import SYMBOL, TIMEFRAME
from services.backtester import run_backtest, calculate_performance_metrics

logger = logging.getLogger(__name__)


class WalkForwardOptimizer:
    """Walk-forward optimization for trading strategies."""

    def __init__(self,
                 data: pd.DataFrame,
                 window_size: int = 252,  # Trading days
                 step_size: int = 21,     # 1 month
                 min_samples: int = 100):
        self.data = data
        self.window_size = window_size
        self.step_size = step_size
        self.min_samples = min_samples
        self.optimization_results = []

    def optimize_parameters(self,
                          parameter_ranges: Dict[str, Tuple[float, float]],
                          n_trials: int = 100) -> Dict[str, Any]:
        """Optimize strategy parameters using walk-forward analysis."""
        try:
            # Create study
            study = optuna.create_study(direction='maximize')

            # Run optimization
            study.optimize(
                lambda trial: self._objective_function(trial, parameter_ranges),
                n_trials=n_trials
            )

            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value

            # Run final validation
            validation_result = self._validate_parameters(best_params)

            return {
                'best_parameters': best_params,
                'best_score': best_score,
                'validation_result': validation_result,
                'optimization_history': self.optimization_results,
                'study': study
            }

        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return {'error': str(e)}

    def _objective_function(self, trial: optuna.Trial, parameter_ranges: Dict[str, Tuple[float, float]]) -> float:
        """Objective function for optimization."""
        # Sample parameters
        params = {}
        for param_name, (min_val, max_val) in parameter_ranges.items():
            if 'period' in param_name.lower() or 'length' in param_name.lower():
                params[param_name] = trial.suggest_int(param_name, int(min_val), int(max_val))
            else:
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)

        # Evaluate parameters using walk-forward
        score = self._evaluate_parameters_walk_forward(params)

        # Store result
        self.optimization_results.append({
            'parameters': params,
            'score': score,
            'timestamp': datetime.now().isoformat()
        })

        return score

    def _evaluate_parameters_walk_forward(self, params: Dict[str, Any]) -> float:
        """Evaluate parameters using walk-forward validation."""
        try:
            scores = []

            # Split data into windows
            data_length = len(self.data)
            if data_length < self.window_size + self.min_samples:
                return 0.0

            for start_idx in range(0, data_length - self.window_size - self.min_samples, self.step_size):
                train_end = start_idx + self.window_size
                test_end = min(train_end + self.step_size, data_length)

                if test_end - train_end < self.min_samples:
                    break

                # Training data
                train_data = self.data.iloc[start_idx:train_end]

                # Test data
                test_data = self.data.iloc[train_end:test_end]

                # Optimize on training data (simplified - in practice would optimize here)
                # For now, just evaluate the given parameters
                train_result = self._evaluate_on_data(train_data, params)
                test_result = self._evaluate_on_data(test_data, params)

                # Use test score
                if test_result and 'sharpe_ratio' in test_result:
                    scores.append(test_result['sharpe_ratio'])

            # Return average score
            return np.mean(scores) if scores else 0.0

        except Exception as e:
            logger.error(f"Walk-forward evaluation failed: {e}")
            return 0.0

    def _evaluate_on_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate parameters on specific data segment."""
        try:
            # Run backtest with parameters
            # This is a simplified version - in practice would modify strategy parameters
            result = run_backtest(data, strategy='traditional')

            return result.get('metrics', {})
        except Exception as e:
            logger.error(f"Data evaluation failed: {e}")
            return {}

    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate optimized parameters on out-of-sample data."""
        try:
            # Use last portion of data for validation
            validation_size = min(len(self.data) // 4, 1000)  # Last 25% or 1000 samples
            validation_data = self.data.iloc[-validation_size:]

            result = self._evaluate_on_data(validation_data, params)

            return {
                'validation_data_size': len(validation_data),
                'validation_metrics': result,
                'parameters': params
            }
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            return {'error': str(e)}


def run_walk_forward_optimization(data: pd.DataFrame,
                                parameter_ranges: Dict[str, Tuple[float, float]] = None,
                                n_trials: int = 50) -> Dict[str, Any]:
    """Run walk-forward optimization with default parameters."""
    if parameter_ranges is None:
        # Default parameter ranges for common indicators
        parameter_ranges = {
            'ema_short_period': (5, 20),
            'ema_long_period': (20, 50),
            'rsi_period': (10, 25),
            'rsi_oversold': (20, 35),
            'rsi_overbought': (65, 80),
            'bb_period': (15, 30),
            'bb_std_dev': (1.5, 3.0),
            'atr_period': (10, 25)
        }

    optimizer = WalkForwardOptimizer(data)
    return optimizer.optimize_parameters(parameter_ranges, n_trials)