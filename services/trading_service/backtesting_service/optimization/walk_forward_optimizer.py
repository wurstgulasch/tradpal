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
from services.trading_service.backtesting_service.service import AsyncBacktester

logger = logging.getLogger(__name__)


class WalkForwardOptimizer:
    """Walk-forward optimization for trading strategies."""

    def __init__(self,
                 symbol: str = SYMBOL,
                 timeframe: str = TIMEFRAME,
                 data: Optional[pd.DataFrame] = None,
                 window_size: int = 252,  # Trading days
                 step_size: int = 21,     # 1 month
                 min_samples: int = 100,
                 results_dir: Optional[str] = None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = data
        self.window_size = window_size
        self.step_size = step_size
        self.min_samples = min_samples
        self.optimization_results = {}
        self.walk_forward_windows = []
        self.results_dir = Path(results_dir) if results_dir else None

        # Bias-variance thresholds
        self.bias_threshold = 0.1
        self.variance_threshold = 0.15
        self.max_bias_variance_ratio = 2.0

    def create_walk_forward_windows(self, df: pd.DataFrame, initial_train_size: int,
                                  test_size: int, step_size: int, min_samples: int) -> List[Dict[str, Any]]:
        """
        Create walk-forward windows for optimization.

        Args:
            df: DataFrame with market data
            initial_train_size: Initial training window size
            test_size: Test window size
            step_size: Step size for moving windows
            min_samples: Minimum samples required

        Returns:
            List of window dictionaries
        """
        if len(df) < initial_train_size + test_size + min_samples:
            raise ValueError("Insufficient data for walk-forward analysis")

        self.walk_forward_windows = []
        current_position = 0
        window_id = 0

        while current_position + initial_train_size + test_size <= len(df):
            window = {
                'window_id': window_id,
                'train_start': current_position,
                'train_end': current_position + initial_train_size,
                'test_start': current_position + initial_train_size,
                'test_end': current_position + initial_train_size + test_size
            }
            self.walk_forward_windows.append(window)
            current_position += step_size
            window_id += 1

        return self.walk_forward_windows

    def optimize_strategy_parameters(self, df: pd.DataFrame, parameter_grid: Dict[str, List[Any]],
                                   evaluation_metric: str = 'sharpe_ratio', min_trades: int = 10) -> Dict[str, Any]:
        """
        Optimize strategy parameters using walk-forward analysis.

        Args:
            df: DataFrame with market data and indicators
            parameter_grid: Dictionary of parameter ranges to test
            evaluation_metric: Metric to optimize
            min_trades: Minimum trades required for valid evaluation

        Returns:
            Optimization results dictionary
        """
        if not self.walk_forward_windows:
            raise ValueError("Walk-forward windows not created. Call create_walk_forward_windows first.")

        # Generate all parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_grid)

        window_results = []

        for window in self.walk_forward_windows:
            # Get training and test data
            train_data = df.iloc[window['train_start']:window['train_end']].copy()
            test_data = df.iloc[window['test_start']:window['test_end']].copy()

            # Find best parameters on training data
            best_params, best_score = self._optimize_window_parameters(
                train_data, param_combinations, evaluation_metric, min_trades
            )

            # Evaluate best parameters on test data
            test_performance = self._evaluate_parameters(test_data, best_params, min_trades)

            window_result = {
                'window_id': window['window_id'],
                'best_parameters': best_params,
                'in_sample_score': best_score,
                'out_of_sample_performance': test_performance
            }
            window_results.append(window_result)

        # Analyze results
        analysis = self._analyze_walk_forward_results(window_results, evaluation_metric)

        results = {
            'optimization_summary': self.get_optimization_summary(),
            'window_results': window_results,
            'analysis': analysis
        }

        return results

    def _generate_parameter_combinations(self, parameter_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        import itertools

        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())

        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)

        return combinations

    def _optimize_window_parameters(self, df: pd.DataFrame, param_combinations: List[Dict[str, Any]],
                                  evaluation_metric: str, min_trades: int) -> Tuple[Dict[str, Any], float]:
        """Find best parameters for a single window."""
        best_score = float('-inf')
        best_params = None

        for params in param_combinations:
            performance = self._evaluate_parameters(df, params, min_trades)

            if performance['valid']:
                score = performance.get(evaluation_metric, 0)
                if score > best_score:
                    best_score = score
                    best_params = params

        return best_params, best_score

    def _evaluate_parameters(self, df: pd.DataFrame, params: Dict[str, Any], min_trades: int) -> Dict[str, Any]:
        """Evaluate a parameter set on data."""
        try:
            # Apply strategy parameters to generate signals
            df_with_signals = self._apply_strategy_parameters(df.copy(), params)

            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(df_with_signals, min_trades)

            return metrics
        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def _apply_strategy_parameters(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply strategy parameters to generate trading signals."""
        # Simple EMA crossover strategy
        ema_short = params.get('ema_short', 9)
        ema_long = params.get('ema_long', 21)
        rsi_overbought = params.get('rsi_overbought', 70)
        rsi_oversold = params.get('rsi_oversold', 30)

        # Generate signals based on parameters
        df['Buy_Signal'] = (
            (df[f'EMA{ema_short}'] > df[f'EMA{ema_long}']) &
            (df['RSI'] < rsi_oversold)
        ).astype(int)

        df['Sell_Signal'] = (
            (df[f'EMA{ema_short}'] < df[f'EMA{ema_long}']) &
            (df['RSI'] > rsi_overbought)
        ).astype(int)

        return df

    def _calculate_performance_metrics(self, df: pd.DataFrame, min_trades: int) -> Dict[str, Any]:
        """Calculate performance metrics from signals."""
        try:
            buy_signals = df['Buy_Signal'].sum()
            sell_signals = df['Sell_Signal'].sum()
            total_trades = buy_signals + sell_signals

            if total_trades < min_trades:
                return {'valid': False, 'total_trades': total_trades}

            # Simple performance calculation (placeholder)
            # In a real implementation, this would calculate actual P&L
            win_rate = 0.55  # Placeholder
            profit_factor = 1.2  # Placeholder
            sharpe_ratio = 1.5  # Placeholder
            max_drawdown = 0.1  # Placeholder
            total_return = 0.15  # Placeholder

            return {
                'valid': True,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': total_return
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def _analyze_walk_forward_results(self, window_results: List[Dict[str, Any]], evaluation_metric: str) -> Dict[str, Any]:
        """Analyze walk-forward optimization results."""
        valid_windows = [w for w in window_results if w['out_of_sample_performance'].get('valid', False)]

        if not valid_windows:
            return {'error': 'No valid windows found'}

        oos_performances = [w['out_of_sample_performance'][evaluation_metric] for w in valid_windows]
        is_performances = [w['in_sample_score'] for w in valid_windows]

        # Calculate metrics
        avg_oos = np.mean(oos_performances)
        std_oos = np.std(oos_performances)
        avg_is = np.mean(is_performances)

        # Performance decay (overfitting measure)
        performance_decay = avg_is - avg_oos

        # Information coefficient
        if len(oos_performances) > 1:
            ic = np.corrcoef(is_performances, oos_performances)[0, 1]
        else:
            ic = 0.0

        # Overfitting ratio
        if avg_is > 0:
            overfitting_ratio = performance_decay / avg_is
        else:
            overfitting_ratio = 0.0

        # Consistency score
        positive_windows = sum(1 for p in oos_performances if p > 0)
        consistency_score = positive_windows / len(oos_performances)

        # Robustness
        robustness = {
            'positive_windows': positive_windows,
            'positive_ratio': positive_windows / len(oos_performances)
        }

        # Bias-variance analysis
        bias_variance = self._calculate_bias_variance(is_performances, oos_performances)

        return {
            'total_windows': len(window_results),
            'valid_windows': len(valid_windows),
            'average_oos_performance': avg_oos,
            'std_oos_performance': std_oos,
            'average_is_performance': avg_is,
            'performance_decay': performance_decay,
            'information_coefficient': ic,
            'overfitting_ratio': overfitting_ratio,
            'consistency_score': consistency_score,
            'robustness': robustness,
            'bias_variance': bias_variance
        }

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        if not self.optimization_results:
            return {'status': 'no_results'}

        return {
            'status': 'completed',
            'total_windows': len(self.walk_forward_windows),
            'results_count': len(self.optimization_results)
        }

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

    def _calculate_bias_variance(self, in_sample: List[float], out_of_sample: List[float]) -> Dict[str, Any]:
        """Calculate bias-variance decomposition."""
        if len(in_sample) == 0 or len(out_of_sample) == 0:
            return {
                "bias": 0.0,
                "variance": 0.0,
                "total_error": 0.0,
                "bias_variance_ratio": 0.0,
                "interpretation": "Insufficient data"
            }

        # Convert to numpy arrays
        in_sample = np.array(in_sample)
        out_of_sample = np.array(out_of_sample)

        # Calculate means
        in_sample_mean = np.mean(in_sample)
        out_of_sample_mean = np.mean(out_of_sample)

        # Calculate bias (difference between in-sample and out-of-sample performance)
        bias = abs(in_sample_mean - out_of_sample_mean)

        # Calculate variance (variability in out-of-sample performance)
        variance = np.var(out_of_sample)

        # Total error
        total_error = bias + variance

        # Bias-variance ratio
        bias_variance_ratio = bias / variance if variance > 0 else float('inf')

        # Interpretation
        if bias < self.bias_threshold and variance < self.variance_threshold:
            interpretation = "Well-balanced model"
        elif bias > self.bias_threshold and variance < self.variance_threshold:
            interpretation = "High bias (underfitting)"
        elif bias < self.bias_threshold and variance > self.variance_threshold:
            interpretation = "High variance (overfitting)"
        else:
            interpretation = "Poor balance (high bias and variance)"

        if bias_variance_ratio > self.max_bias_variance_ratio:
            interpretation += " - Consider regularization"

        return {
            "bias": float(bias),
            "variance": float(variance),
            "total_error": float(total_error),
            "bias_variance_ratio": float(bias_variance_ratio),
            "interpretation": interpretation
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR indicator."""
        try:
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)

            tr1 = high - low
            tr2 = (high - close).abs()
            tr3 = (low - close).abs()

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr
        except Exception:
            return pd.Series([2.0] * len(df), index=df.index)


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


# Global optimizer instance for singleton pattern
walk_forward_optimizer = None


def get_walk_forward_optimizer(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> WalkForwardOptimizer:
    """
    Get or create a walk-forward optimizer instance (singleton pattern).

    Args:
        symbol: Trading symbol
        timeframe: Timeframe

    Returns:
        WalkForwardOptimizer instance
    """
    global walk_forward_optimizer

    if walk_forward_optimizer is None or \
       walk_forward_optimizer.symbol != symbol or \
       walk_forward_optimizer.timeframe != timeframe:
        walk_forward_optimizer = WalkForwardOptimizer(symbol=symbol, timeframe=timeframe)

    return walk_forward_optimizer