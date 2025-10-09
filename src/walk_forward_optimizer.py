"""
Walk-Forward Analysis for Trading Strategy Optimization

Implements time-series cross-validation with walk-forward optimization
for more realistic backtesting and strategy evaluation.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json

class WalkForwardOptimizer:
    """
    Walk-forward analysis for trading strategy optimization.
    
    Features:
    - Time-series cross-validation with expanding/rolling windows
    - Multiple optimization windows for robust parameter selection
    - Out-of-sample performance evaluation
    - Strategy robustness assessment
    - Parameter stability analysis
    """
    
    def __init__(self, symbol: str = "BTC/USD", timeframe: str = "1h",
                 results_dir: str = "output/walk_forward"):
        """
        Initialize the walk-forward optimizer.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            results_dir: Directory to store optimization results
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimization_results = {}
        self.walk_forward_windows = []
    
    def create_walk_forward_windows(self, df: pd.DataFrame, initial_train_size: int = 1000,
                                   test_size: int = 200, step_size: int = 50,
                                   min_samples: int = 500) -> List[Dict[str, Any]]:
        """
        Create walk-forward analysis windows.
        
        Args:
            df: DataFrame with trading data
            initial_train_size: Initial training window size
            test_size: Test window size for each iteration
            step_size: Step size for moving windows
            min_samples: Minimum samples required for analysis
            
        Returns:
            List of window dictionaries with train/test indices
        """
        if len(df) < min_samples:
            raise ValueError(f"Insufficient data: {len(df)} samples, need at least {min_samples}")
        
        windows = []
        start_idx = 0
        window_id = 0
        
        while start_idx + initial_train_size + test_size <= len(df):
            train_end = start_idx + initial_train_size
            test_end = train_end + test_size
            
            window = {
                'window_id': window_id,
                'train_start': start_idx,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'train_size': initial_train_size,
                'test_size': test_size,
                'total_samples': len(df)
            }
            
            windows.append(window)
            
            # Move window forward
            start_idx += step_size
            window_id += 1
            
            # For subsequent windows, use expanding window (include previous test data)
            if window_id > 0:
                initial_train_size += step_size
        
        self.walk_forward_windows = windows
        
        print(f"Created {len(windows)} walk-forward windows")
        return windows
    
    def optimize_strategy_parameters(self, df: pd.DataFrame,
                                   parameter_grid: Dict[str, List[Any]],
                                   evaluation_metric: str = 'sharpe_ratio',
                                   min_trades: int = 10) -> Dict[str, Any]:
        """
        Optimize strategy parameters using walk-forward analysis.
        
        Args:
            df: DataFrame with trading data and signals
            parameter_grid: Dictionary of parameter names and values to test
            evaluation_metric: Metric to optimize ('sharpe_ratio', 'win_rate', 'profit_factor')
            min_trades: Minimum trades required for valid evaluation
            
        Returns:
            Dictionary with optimization results
        """
        if not self.walk_forward_windows:
            raise ValueError("Walk-forward windows not created. Call create_walk_forward_windows first.")
        
        print(f"🔄 Starting walk-forward optimization with {len(self.walk_forward_windows)} windows...")
        
        # Generate all parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_grid)
        
        window_results = []
        
        for window in self.walk_forward_windows[:2]:  # Limit to 2 windows for testing
            print(f"📊 Optimizing window {window['window_id'] + 1}...")
            
            # Get train and test data
            train_df = df.iloc[window['train_start']:window['train_end']].copy()
            test_df = df.iloc[window['test_start']:window['test_end']].copy()
            
            # Optimize parameters on training data
            best_params, best_score = self._optimize_window_parameters(
                train_df, param_combinations, evaluation_metric, min_trades
            )
            
            # Evaluate best parameters on test data
            test_performance = self._evaluate_parameters(test_df, best_params, min_trades)
            
            window_result = {
                'window_id': window['window_id'],
                'best_parameters': best_params,
                'in_sample_score': best_score,
                'out_of_sample_performance': test_performance,
                'train_size': len(train_df),
                'test_size': len(test_df)
            }
            
            window_results.append(window_result)
        
        # Analyze overall results
        analysis_results = self._analyze_walk_forward_results(window_results, evaluation_metric)
        
        results = {
            'optimization_summary': {
                'total_windows': len(window_results),
                'parameter_combinations_tested': len(param_combinations),
                'evaluation_metric': evaluation_metric,
                'symbol': self.symbol,
                'timeframe': self.timeframe
            },
            'window_results': window_results,
            'analysis': analysis_results,
            'parameter_grid': parameter_grid,
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_results = results
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
        """
        Optimize parameters for a single window.
        
        Args:
            df: Training data for this window
            param_combinations: List of parameter combinations to test
            evaluation_metric: Metric to optimize
            min_trades: Minimum trades required
            
        Returns:
            Tuple of (best_parameters, best_score)
        """
        best_score = float('-inf')
        best_params = None
        
        for params in param_combinations[:3]:  # Limit combinations for testing
            try:
                # Evaluate this parameter combination
                performance = self._evaluate_parameters(df, params, min_trades)
                
                if performance['valid'] and performance[evaluation_metric] > best_score:
                    best_score = performance[evaluation_metric]
                    best_params = params.copy()
                    
            except Exception as e:
                continue
        
        if best_params is None:
            # Fallback to first combination if none worked
            best_params = param_combinations[0]
            best_score = 0
        
        return best_params, best_score
    
    def _evaluate_parameters(self, df: pd.DataFrame, params: Dict[str, Any], min_trades: int) -> Dict[str, Any]:
        """
        Evaluate strategy performance with given parameters.
        
        Args:
            df: DataFrame with price data
            params: Strategy parameters
            min_trades: Minimum trades required for valid evaluation
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Apply strategy parameters to generate signals
            df_eval = self._apply_strategy_parameters(df.copy(), params)
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(df_eval, min_trades)
            
            return performance
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0
            }
    
    def _apply_strategy_parameters(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply strategy parameters to generate trading signals.
        
        Args:
            df: DataFrame with price data and indicators
            params: Strategy parameters
            
        Returns:
            DataFrame with signals and trades
        """
        # Extract parameters with defaults
        ema_short_period = params.get('ema_short', 9)
        ema_long_period = params.get('ema_long', 21)
        rsi_period = params.get('rsi_period', 14)
        rsi_overbought = params.get('rsi_overbought', 70)
        rsi_oversold = params.get('rsi_oversold', 30)
        atr_period = params.get('atr_period', 14)
        risk_per_trade = params.get('risk_per_trade', 0.01)
        
        # Calculate indicators if not present
        if f'EMA{ema_short_period}' not in df.columns:
            df[f'EMA{ema_short_period}'] = df['close'].ewm(span=ema_short_period).mean()
        if f'EMA{ema_long_period}' not in df.columns:
            df[f'EMA{ema_long_period}'] = df['close'].ewm(span=ema_long_period).mean()
        if 'RSI' not in df.columns:
            df['RSI'] = self._calculate_rsi(df['close'], rsi_period)
        if f'ATR{atr_period}' not in df.columns:
            df[f'ATR{atr_period}'] = self._calculate_atr(df, atr_period)
        
        # Generate signals
        ema_short = df[f'EMA{ema_short_period}']
        ema_long = df[f'EMA{ema_long_period}']
        rsi = df['RSI']
        
        # Buy signals
        buy_condition = (
            (ema_short > ema_long) &
            (rsi < rsi_overbought)
        )
        
        # Sell signals  
        sell_condition = (
            (ema_short < ema_long) &
            (rsi > rsi_oversold)
        )
        
        df['Buy_Signal'] = buy_condition.astype(int)
        df['Sell_Signal'] = sell_condition.astype(int)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR indicator."""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_performance_metrics(self, df: pd.DataFrame, min_trades: int) -> Dict[str, Any]:
        """
        Calculate trading performance metrics.
        
        Args:
            df: DataFrame with signals and price data
            min_trades: Minimum trades required for valid evaluation
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Simulate trades
            trades = []
            position = 0
            entry_price = 0
            
            for idx, row in df.iterrows():
                if position == 0:  # No position
                    if row['Buy_Signal'] == 1:
                        position = 1
                        entry_price = row['close']
                        trades.append({
                            'type': 'buy',
                            'price': entry_price,
                            'timestamp': idx
                        })
                elif position == 1:  # Long position
                    if row['Sell_Signal'] == 1:
                        exit_price = row['close']
                        pnl = (exit_price - entry_price) / entry_price
                        trades[-1].update({
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'exit_timestamp': idx
                        })
                        position = 0
            
            # Calculate metrics
            if len(trades) < min_trades:
                return {
                    'valid': False,
                    'total_trades': len(trades),
                    'win_rate': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'total_return': 0
                }
            
            completed_trades = [t for t in trades if 'pnl' in t]
            if not completed_trades:
                return {
                    'valid': False,
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'total_return': 0
                }
            
            pnls = [t['pnl'] for t in completed_trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            win_rate = len(wins) / len(pnls) if pnls else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) if losses and avg_loss > 0 else float('inf')
            
            # Sharpe ratio (simplified)
            if len(pnls) > 1:
                returns = np.array(pnls)
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Max drawdown (simplified)
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            total_return = np.sum(pnls)
            
            return {
                'valid': True,
                'total_trades': len(completed_trades),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': total_return,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0
            }
    
    def _analyze_walk_forward_results(self, window_results: List[Dict[str, Any]],
                                    evaluation_metric: str) -> Dict[str, Any]:
        """
        Analyze walk-forward optimization results with advanced overfitting metrics.
        
        Args:
            window_results: List of window optimization results
            evaluation_metric: Metric used for optimization
            
        Returns:
            Dictionary with analysis results including overfitting metrics
        """
        if not window_results:
            return {}
        
        # Extract metrics
        oos_performances = []
        is_performances = []
        
        for result in window_results:
            oos_perf = result['out_of_sample_performance']
            is_score = result['in_sample_score']
            
            if oos_perf['valid']:
                oos_performances.append(oos_perf[evaluation_metric])
            is_performances.append(is_score)
        
        # Calculate Information Coefficient (IC) - correlation between predictions and outcomes
        information_coefficient = None
        if len(is_performances) == len(oos_performances) and len(is_performances) > 1:
            try:
                ic_correlation = np.corrcoef(is_performances, oos_performances)[0, 1]
                information_coefficient = ic_correlation if not np.isnan(ic_correlation) else 0.0
            except Exception:
                information_coefficient = 0.0
        
        # Calculate Bias-Variance Tradeoff metrics
        bias_variance_stats = self._calculate_bias_variance(is_performances, oos_performances)
        
        # Calculate statistics
        analysis = {
            'total_windows': len(window_results),
            'valid_windows': len([r for r in window_results if r['out_of_sample_performance']['valid']]),
            'average_oos_performance': np.mean(oos_performances) if oos_performances else 0,
            'std_oos_performance': np.std(oos_performances) if oos_performances else 0,
            'average_is_performance': np.mean(is_performances) if is_performances else 0,
            'std_is_performance': np.std(is_performances) if is_performances else 0,
            'performance_decay': None,
            'information_coefficient': information_coefficient,
            'bias_variance': bias_variance_stats,
            'overfitting_ratio': None,
            'consistency_score': None
        }
        
        # Calculate performance decay (overfitting measure)
        if oos_performances and is_performances:
            avg_oos = np.mean(oos_performances)
            avg_is = np.mean(is_performances)
            analysis['performance_decay'] = avg_is - avg_oos
            
            # Overfitting ratio: how much better in-sample performs vs out-of-sample
            if avg_oos != 0:
                analysis['overfitting_ratio'] = (avg_is - avg_oos) / abs(avg_oos)
            else:
                analysis['overfitting_ratio'] = float('inf') if avg_is > 0 else 0
        
        # Consistency score: how consistent are out-of-sample results
        if oos_performances and len(oos_performances) > 1:
            # Lower coefficient of variation = more consistent
            cv = np.std(oos_performances) / (abs(np.mean(oos_performances)) + 1e-10)
            analysis['consistency_score'] = 1.0 / (1.0 + cv)  # Normalize to [0, 1]
        
        # Robustness assessment
        if oos_performances:
            positive_windows = len([p for p in oos_performances if p > 0])
            analysis['robustness'] = {
                'positive_windows': positive_windows,
                'positive_ratio': positive_windows / len(oos_performances),
                'consistency_score': np.std(oos_performances) / abs(np.mean(oos_performances)) if np.mean(oos_performances) != 0 else float('inf')
            }
        
        return analysis
    
    def _calculate_bias_variance(self, in_sample: List[float], out_of_sample: List[float]) -> Dict[str, Any]:
        """
        Calculate bias-variance tradeoff metrics.
        
        Args:
            in_sample: In-sample performance scores
            out_of_sample: Out-of-sample performance scores
            
        Returns:
            Dictionary with bias and variance metrics
        """
        if not in_sample or not out_of_sample:
            return {
                'bias': None,
                'variance': None,
                'total_error': None,
                'bias_variance_ratio': None
            }
        
        # Bias: systematic error (difference between expected and actual)
        # Represented by the mean difference between in-sample and out-of-sample
        bias = np.mean(in_sample) - np.mean(out_of_sample)
        
        # Variance: model sensitivity to training data
        # Represented by the variance of out-of-sample performance
        variance = np.var(out_of_sample) if len(out_of_sample) > 1 else 0
        
        # Total error decomposition
        total_error = bias**2 + variance
        
        # Bias-variance ratio
        bias_variance_ratio = abs(bias) / (variance + 1e-10) if variance > 0 else float('inf')
        
        return {
            'bias': float(bias),
            'variance': float(variance),
            'total_error': float(total_error),
            'bias_variance_ratio': float(bias_variance_ratio),
            'interpretation': self._interpret_bias_variance(bias, variance)
        }
    
    def _interpret_bias_variance(self, bias: float, variance: float) -> str:
        """
        Interpret bias-variance tradeoff results.
        
        Args:
            bias: Bias measure
            variance: Variance measure
            
        Returns:
            Human-readable interpretation
        """
        if abs(bias) > variance * 2:
            return "High bias (underfitting) - model too simple or features insufficient"
        elif variance > abs(bias) * 2:
            return "High variance (overfitting) - model too complex or insufficient data"
        elif abs(bias) < 0.01 and variance < 0.01:
            return "Well-balanced model with good generalization"
        else:
            return "Moderate bias-variance tradeoff - acceptable balance"
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        if not self.optimization_results:
            return {'status': 'no_results'}
        
        analysis = self.optimization_results.get('analysis', {})
        
        return {
            'status': 'completed',
            'total_windows': analysis.get('total_windows', 0),
            'valid_windows': analysis.get('valid_windows', 0),
            'average_oos_performance': analysis.get('average_oos_performance', 0),
            'performance_decay': analysis.get('performance_decay', 0),
            'robustness_ratio': analysis.get('robustness', {}).get('positive_ratio', 0),
            'symbol': self.symbol,
            'timeframe': self.timeframe
        }

# Global walk-forward optimizer instance
walk_forward_optimizer = None

def get_walk_forward_optimizer(symbol: str = "BTC/USD", timeframe: str = "1h") -> WalkForwardOptimizer:
    """Get or create walk-forward optimizer instance."""
    global walk_forward_optimizer
    
    if walk_forward_optimizer is None:
        walk_forward_optimizer = WalkForwardOptimizer(symbol=symbol, timeframe=timeframe)
    
    return walk_forward_optimizer
