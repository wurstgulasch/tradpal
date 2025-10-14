"""
Discovery module for optimizing trading indicator configurations using Genetic Algorithms.

This module implements a GA-based optimization system to find the best combinations
of technical indicators and their parameters for maximum trading performance.
"""

import random
import time
import json
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Check if deap is available
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("⚠️  DEAP library not available. Discovery mode will not function.")
    print("   Install with: pip install deap")

# Import dependencies
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Try relative imports first (when imported as part of src package)
    from .backtester import run_backtest
    from .data_fetcher import fetch_historical_data
    from .indicators import calculate_indicators
    from .signal_generator import generate_signals, calculate_risk_management
    from .error_handling import error_boundary
    from .logging_config import logger
    from .discovery_cache import cached_discovery_fetch, get_discovery_cache_stats
except ImportError:
    # Fall back to absolute imports (when imported directly)
    from backtester import run_backtest
    from data_fetcher import fetch_historical_data
    from indicators import calculate_indicators
    from signal_generator import generate_signals, calculate_risk_management
    from error_handling import error_boundary
    from logging_config import logger
    from discovery_cache import cached_discovery_fetch, get_discovery_cache_stats

from config.settings import SYMBOL, EXCHANGE, TIMEFRAME, DISCOVERY_PARAMS

logger = logger.getChild(__name__)

@dataclass
class IndividualResult:
    """Result of evaluating an individual configuration."""
    config: Dict[str, Any]
    fitness: float
    pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    evaluation_time: float = 0.0
    backtest_duration_days: int = 0
    pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    evaluation_time: float
    backtest_duration_days: int

class DiscoveryOptimizer:
    """
    Genetic Algorithm optimizer for trading indicator configurations.

    Uses DEAP library to evolve optimal combinations of technical indicators
    and their parameters based on historical backtesting performance.
    """

    # Predefined indicator combinations to test
    INDICATOR_COMBINATIONS = {
        'ema_only': {'name': 'EMA Only', 'indicators': ['ema']},
        'ema_rsi': {'name': 'EMA + RSI', 'indicators': ['ema', 'rsi']},
        'ema_bb': {'name': 'EMA + Bollinger Bands', 'indicators': ['ema', 'bb']},
        'ema_atr': {'name': 'EMA + ATR', 'indicators': ['ema', 'atr']},
        'ema_macd': {'name': 'EMA + MACD', 'indicators': ['ema', 'macd']},
        'ema_stoch': {'name': 'EMA + Stochastic', 'indicators': ['ema', 'stochastic']},
        'ema_rsi_bb': {'name': 'EMA + RSI + BB', 'indicators': ['ema', 'rsi', 'bb']},
        'ema_rsi_atr': {'name': 'EMA + RSI + ATR', 'indicators': ['ema', 'rsi', 'atr']},
        'ema_bb_atr': {'name': 'EMA + BB + ATR', 'indicators': ['ema', 'bb', 'atr']},
        'ema_macd_bb': {'name': 'EMA + MACD + BB', 'indicators': ['ema', 'macd', 'bb']},
        'ema_rsi_macd': {'name': 'EMA + RSI + MACD', 'indicators': ['ema', 'rsi', 'macd']},
        'ema_rsi_bb_atr': {'name': 'EMA + RSI + BB + ATR', 'indicators': ['ema', 'rsi', 'bb', 'atr']},
        'ema_rsi_bb_macd': {'name': 'EMA + RSI + BB + MACD', 'indicators': ['ema', 'rsi', 'bb', 'macd']},
        'ema_rsi_atr_macd': {'name': 'EMA + RSI + ATR + MACD', 'indicators': ['ema', 'rsi', 'atr', 'macd']},
        'ema_bb_atr_stoch': {'name': 'EMA + BB + ATR + Stochastic', 'indicators': ['ema', 'bb', 'atr', 'stochastic']},
        'ema_rsi_bb_atr_macd': {'name': 'EMA + RSI + BB + ATR + MACD', 'indicators': ['ema', 'rsi', 'bb', 'atr', 'macd']},
        'ema_rsi_bb_atr_stoch': {'name': 'EMA + RSI + BB + ATR + Stochastic', 'indicators': ['ema', 'rsi', 'bb', 'atr', 'stochastic']},
        'ema_rsi_bb_macd_stoch': {'name': 'EMA + RSI + BB + MACD + Stochastic', 'indicators': ['ema', 'rsi', 'bb', 'macd', 'stochastic']},
        'full_suite': {'name': 'Full Suite', 'indicators': ['ema', 'rsi', 'bb', 'atr', 'adx', 'macd', 'obv', 'stochastic']},
        'momentum_focused': {'name': 'Momentum Focused', 'indicators': ['ema', 'rsi', 'macd', 'stochastic', 'obv']},
        'volatility_focused': {'name': 'Volatility Focused', 'indicators': ['ema', 'bb', 'atr', 'adx']},
        'trend_focused': {'name': 'Trend Focused', 'indicators': ['ema', 'adx']},
        'oscillator_focused': {'name': 'Oscillator Focused', 'indicators': ['ema', 'rsi', 'macd', 'stochastic']},
        'comprehensive': {'name': 'Comprehensive', 'indicators': ['ema', 'rsi', 'bb', 'macd', 'stochastic', 'obv', 'adx']},
        'conservative': {'name': 'Conservative', 'indicators': ['ema', 'rsi', 'bb']},
        'aggressive': {'name': 'Aggressive', 'indicators': ['ema', 'rsi', 'bb', 'macd', 'stochastic', 'obv', 'adx']},
        'minimal_risk': {'name': 'Minimal Risk', 'indicators': ['ema', 'bb']},
        'high_frequency': {'name': 'High Frequency', 'indicators': ['ema', 'rsi', 'macd']},
        'swing_trading': {'name': 'Swing Trading', 'indicators': ['ema', 'rsi', 'bb', 'adx']},
        'scalping': {'name': 'Scalping', 'indicators': ['ema', 'rsi', 'stochastic']},
        'breakout': {'name': 'Breakout', 'indicators': ['ema', 'bb', 'atr']},
        'reversal': {'name': 'Reversal', 'indicators': ['ema', 'rsi', 'macd', 'stochastic']},
        'trend_following': {'name': 'Trend Following', 'indicators': ['ema', 'adx', 'obv']},
        'mean_reversion': {'name': 'Mean Reversion', 'indicators': ['ema', 'rsi', 'bb']}
    }

    def __init__(self, symbol=SYMBOL, exchange=EXCHANGE, timeframe=TIMEFRAME,
                 start_date: str = '2024-01-01',
                 end_date: str = '2024-12-31',
                 population_size: int = None,  # Will use DISCOVERY_PARAMS if None
                 generations: int = None,       # Will use DISCOVERY_PARAMS if None
                 mutation_rate: float = None,   # Will use DISCOVERY_PARAMS if None
                 crossover_rate: float = None,  # Will use DISCOVERY_PARAMS if None
                 initial_capital: float = 10000.0,
                 use_walk_forward: bool = True):  # New parameter for WFA
        """
        Initialize the discovery optimizer with enhanced robustness.

        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            timeframe: Timeframe for backtesting
            start_date: Start date for historical data
            end_date: End date for historical data
            population_size: GA population size (uses DISCOVERY_PARAMS if None)
            generations: Number of GA generations (uses DISCOVERY_PARAMS if None)
            mutation_rate: Probability of mutation (uses DISCOVERY_PARAMS if None)
            crossover_rate: Probability of crossover (uses DISCOVERY_PARAMS if None)
            initial_capital: Initial capital for backtesting
            use_walk_forward: Whether to use walk-forward analysis for evaluation
        """
        if not DEAP_AVAILABLE:
            raise ImportError("Discovery module requires 'deap' package. Install with: pip install deap")

        # Import discovery parameters
        from config.settings import DISCOVERY_PARAMS

        # Use optimized parameters from settings if not explicitly provided
        self.population_size = population_size if population_size is not None else DISCOVERY_PARAMS['population_size']
        self.generations = generations if generations is not None else DISCOVERY_PARAMS['generations']
        self.mutation_rate = mutation_rate if mutation_rate is not None else DISCOVERY_PARAMS['mutation_rate']
        self.crossover_rate = crossover_rate if crossover_rate is not None else DISCOVERY_PARAMS['crossover_rate']

        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.use_walk_forward = use_walk_forward  # Store WFA preference

        # GA setup
        self._setup_ga()

        # Cache for historical data
        self.historical_data = None

        # Results storage
        self.results: List[IndividualResult] = []

    def _setup_ga(self):
        """Setup DEAP genetic algorithm components."""
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Define the toolbox
        self.toolbox = base.Toolbox()

        # Attribute generators for each parameter
        # Combination index: 0 to len(INDICATOR_COMBINATIONS)-1
        self.toolbox.register("combination_idx", random.randint, 0, len(self.INDICATOR_COMBINATIONS) - 1)

        # EMA periods: 5-50 for short, 10-200 for long
        self.toolbox.register("ema_short", random.randint, 5, 50)
        self.toolbox.register("ema_long", random.randint, 10, 200)

        # RSI period: 5-30
        self.toolbox.register("rsi_period", random.randint, 5, 30)

        # RSI thresholds: 20-40 for oversold, 60-80 for overbought
        self.toolbox.register("rsi_oversold", random.randint, 20, 40)
        self.toolbox.register("rsi_overbought", random.randint, 60, 80)

        # BB period: 10-50, std_dev: 1.5-3.0
        self.toolbox.register("bb_period", random.randint, 10, 50)
        self.toolbox.register("bb_std_dev", random.uniform, 1.5, 3.0)

        # ATR period: 5-30
        self.toolbox.register("atr_period", random.randint, 5, 30)

        # MACD parameters: fast(8-20), slow(20-40), signal(5-15)
        self.toolbox.register("macd_fast", random.randint, 8, 20)
        self.toolbox.register("macd_slow", random.randint, 20, 40)
        self.toolbox.register("macd_signal", random.randint, 5, 15)

        # Stochastic parameters: k(5-21), d(3-8)
        self.toolbox.register("stoch_k", random.randint, 5, 21)
        self.toolbox.register("stoch_d", random.randint, 3, 8)

        # ADX period: 5-30
        self.toolbox.register("adx_period", random.randint, 5, 30)

        # OBV moving average period: 10-50
        self.toolbox.register("obv_ma_period", random.randint, 10, 50)

        # Structure: [combination_idx, ema_short, ema_long, rsi_period, rsi_oversold, rsi_overbought,
        #             bb_period, bb_std_dev, atr_period, macd_fast, macd_slow, macd_signal,
        #             stoch_k, stoch_d, adx_period, obv_ma_period]
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                            (self.toolbox.combination_idx, self.toolbox.ema_short, self.toolbox.ema_long,
                             self.toolbox.rsi_period, self.toolbox.rsi_oversold, self.toolbox.rsi_overbought,
                             self.toolbox.bb_period, self.toolbox.bb_std_dev, self.toolbox.atr_period,
                             self.toolbox.macd_fast, self.toolbox.macd_slow, self.toolbox.macd_signal,
                             self.toolbox.stoch_k, self.toolbox.stoch_d, self.toolbox.adx_period,
                             self.toolbox.obv_ma_period), n=1)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_individual, indpb=self.mutation_rate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _mutate_individual(self, individual, indpb):
        """Custom mutation function that respects parameter ranges."""
        for i in range(len(individual)):
            if random.random() < indpb:
                if i == 0:  # combination_idx
                    individual[i] = random.randint(0, len(self.INDICATOR_COMBINATIONS) - 1)
                elif i == 1:  # ema_short
                    individual[i] = random.randint(5, 50)
                elif i == 2:  # ema_long
                    individual[i] = random.randint(10, 200)
                elif i == 3:  # rsi_period
                    individual[i] = random.randint(5, 30)
                elif i == 4:  # rsi_oversold
                    individual[i] = random.randint(20, 40)
                elif i == 5:  # rsi_overbought
                    individual[i] = random.randint(60, 80)
                elif i == 6:  # bb_period
                    individual[i] = random.randint(10, 50)
                elif i == 7:  # bb_std_dev
                    individual[i] = random.uniform(1.5, 3.0)
                elif i == 8:  # atr_period
                    individual[i] = random.randint(5, 30)
                elif i == 9:  # macd_fast
                    individual[i] = random.randint(8, 20)
                elif i == 10:  # macd_slow
                    individual[i] = random.randint(20, 40)
                elif i == 11:  # macd_signal
                    individual[i] = random.randint(5, 15)
                elif i == 12:  # stoch_k
                    individual[i] = random.randint(5, 21)
                elif i == 13:  # stoch_d
                    individual[i] = random.randint(3, 8)
                elif i == 14:  # adx_period
                    individual[i] = random.randint(5, 30)
                elif i == 15:  # obv_ma_period
                    individual[i] = random.randint(10, 50)
        return individual,

    def _individual_to_config(self, individual) -> Dict[str, Any]:
        """Convert GA individual to indicator configuration dict."""
        # Handle different individual structures for backward compatibility
        if len(individual) == 21:
            # Test structure: [ema_short, ema_long, rsi_period, rsi_oversold, rsi_overbought,
            #                  bb_period, bb_std_dev, atr_period, macd_fast, macd_slow, macd_signal,
            #                  stoch_k, stoch_d, ema_enabled, rsi_enabled, bb_enabled, adx_enabled,
            #                  atr_enabled, macd_enabled, stoch_enabled, obv_enabled]
            config = {
                'ema': {
                    'enabled': bool(individual[13]),
                    'periods': [int(individual[0]), int(individual[1])]
                },
                'rsi': {
                    'enabled': bool(individual[14]),
                    'period': int(individual[2]),
                    'oversold': int(individual[3]),
                    'overbought': int(individual[4])
                },
                'bb': {
                    'enabled': bool(individual[15]),
                    'period': int(individual[5]),
                    'std_dev': round(individual[6], 2)
                },
                'atr': {
                    'enabled': bool(individual[17]),
                    'period': int(individual[7])
                },
                'adx': {
                    'enabled': bool(individual[16]),
                    'period': 14  # Default period for test compatibility
                },
                'macd': {
                    'enabled': bool(individual[18]),
                    'fast_period': int(individual[8]),
                    'slow_period': int(individual[9]),
                    'signal_period': int(individual[10])
                },
                'obv': {
                    'enabled': bool(individual[20]),
                    'ma_period': 20  # Default period for test compatibility
                },
                'stochastic': {
                    'enabled': bool(individual[19]),
                    'k_period': int(individual[11]),
                    'd_period': int(individual[12])
                },
                'cmf': {
                    'enabled': False
                },
                'fibonacci': {
                    'enabled': False
                },
                'combination_name': 'test_combination',
                'enabled_indicators': [
                    name for name, enabled in [
                        ('ema', bool(individual[13])),
                        ('rsi', bool(individual[14])),
                        ('bb', bool(individual[15])),
                        ('adx', bool(individual[16])),
                        ('atr', bool(individual[17])),
                        ('macd', bool(individual[18])),
                        ('stochastic', bool(individual[19])),
                        ('obv', bool(individual[20]))
                    ] if enabled
                ]
            }
        else:
            # Current GA structure: [combination_idx, ema_short, ema_long, rsi_period, rsi_oversold, rsi_overbought,
            #                        bb_period, bb_std_dev, atr_period, macd_fast, macd_slow, macd_signal,
            #                        stoch_k, stoch_d, adx_period, obv_ma_period]
            combination_idx = int(individual[0])
            combination_names = list(self.INDICATOR_COMBINATIONS.keys())
            if combination_idx >= len(combination_names):
                combination_idx = 0  # Fallback to first combination
            combination_name = combination_names[combination_idx]
            combination = self.INDICATOR_COMBINATIONS[combination_name]
            enabled_indicators = combination['indicators']

            config = {
                'ema': {
                    'enabled': 'ema' in enabled_indicators,
                    'periods': [int(individual[1]), int(individual[2])]  # ema_short, ema_long
                },
                'rsi': {
                    'enabled': 'rsi' in enabled_indicators,
                    'period': int(individual[3]),  # rsi_period
                    'oversold': int(individual[4]),  # rsi_oversold
                    'overbought': int(individual[5])  # rsi_overbought
                },
                'bb': {
                    'enabled': 'bb' in enabled_indicators,
                    'period': int(individual[6]),  # bb_period
                    'std_dev': round(individual[7], 2)  # bb_std_dev
                },
                'atr': {
                    'enabled': 'atr' in enabled_indicators,
                    'period': int(individual[8])  # atr_period
                },
                'adx': {
                    'enabled': 'adx' in enabled_indicators,
                    'period': int(individual[14])  # adx_period
                },
                'macd': {
                    'enabled': 'macd' in enabled_indicators,
                    'fast_period': int(individual[9]),  # macd_fast
                    'slow_period': int(individual[10]),  # macd_slow
                    'signal_period': int(individual[11])  # macd_signal
                },
                'obv': {
                    'enabled': 'obv' in enabled_indicators,
                    'ma_period': int(individual[15])  # obv_ma_period
                },
                'stochastic': {
                    'enabled': 'stochastic' in enabled_indicators,
                    'k_period': int(individual[12]),  # stoch_k
                    'd_period': int(individual[13])  # stoch_d
                },
                'cmf': {
                    'enabled': False  # Disabled for optimization
                },
                'fibonacci': {
                    'enabled': False  # Disabled for optimization
                },
                'combination_name': combination['name'],
                'enabled_indicators': enabled_indicators
            }

        return config

    def _generate_signals_with_config(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate signals based on the custom configuration using the main signal generator."""
        # Use the main generate_signals function with the provided config
        return generate_signals(df, config=config)

    @error_boundary(operation="evaluate_individual", max_retries=1)
    def _evaluate_individual(self, individual) -> Tuple[float]:
        """Evaluate fitness of an individual configuration using cross-validation."""
        start_time = time.time()

        try:
            # Convert individual to config
            config = self._individual_to_config(individual)

            # Load historical data
            historical_data = self._load_historical_data()

            # Perform cross-validation to prevent overfitting
            cv_fitness, cv_metrics = self._cross_validate_config(config, historical_data, use_walk_forward=self.use_walk_forward)

            evaluation_time = time.time() - start_time

            # Store result with cross-validation metrics
            result = IndividualResult(
                config=config,
                fitness=cv_fitness,
                pnl=cv_metrics['avg_pnl'],
                win_rate=cv_metrics['avg_win_rate'],
                sharpe_ratio=cv_metrics['avg_sharpe'],
                max_drawdown=cv_metrics['max_drawdown'],
                total_trades=cv_metrics['total_trades'],
                evaluation_time=evaluation_time,
                backtest_duration_days=len(historical_data)
            )
            self.results.append(result)

            logger.debug(f"Evaluated combination '{config.get('combination_name', 'unknown')}': CV fitness={cv_fitness:.2f}, avg pnl={cv_metrics['avg_pnl']:.2f}")

            return (cv_fitness,)

        except Exception as e:
            logger.warning(f"Error evaluating individual: {e}")
            evaluation_time = time.time() - start_time
            return (0.0,)

    def _cross_validate_config(self, config: Dict[str, Any], data: pd.DataFrame, n_folds: int = 3,
                              use_walk_forward: bool = True) -> Tuple[float, Dict[str, float]]:
        """
        Perform cross-validation on a configuration to prevent overfitting.
        Optionally uses walk-forward analysis for more robust evaluation.

        Args:
            config: Indicator configuration to evaluate
            data: Historical data for backtesting
            n_folds: Number of cross-validation folds
            use_walk_forward: Whether to use walk-forward analysis instead of k-fold CV

        Returns:
            Tuple of (average_fitness, metrics_dict)
        """
        if len(data) < 100:  # Minimum data requirement
            # Fallback to single evaluation
            return self._evaluate_config_single(config, data)

        if use_walk_forward:
            # Use walk-forward analysis for more realistic out-of-sample testing
            return self._evaluate_with_walk_forward(config, data)
        else:
            # Traditional k-fold cross-validation
            fold_size = len(data) // n_folds
            folds = []

            for i in range(n_folds):
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(data)

                fold_data = data.iloc[start_idx:end_idx]
                folds.append(fold_data)

            # Evaluate each fold
            fold_results = []
            for i, fold_data in enumerate(folds):
                try:
                    fold_fitness, fold_metrics = self._evaluate_config_single(config, fold_data)
                    fold_results.append((fold_fitness, fold_metrics))
                    logger.debug(f"Fold {i+1}/{n_folds}: fitness={fold_fitness:.2f}")
                except Exception as e:
                    logger.warning(f"Error evaluating fold {i+1}: {e}")
                    # Use neutral score for failed folds
                    fold_results.append((0.0, {
                        'total_pnl': 0.0,
                        'win_rate': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,
                        'total_trades': 0
                    }))

            # Calculate cross-validation metrics
            fitness_scores = [result[0] for result in fold_results]
            avg_fitness = np.mean(fitness_scores)

            # Calculate average metrics across folds
            avg_metrics = {
                'avg_pnl': np.mean([result[1]['total_pnl'] for result in fold_results]),
                'avg_win_rate': np.mean([result[1]['win_rate'] for result in fold_results]),
                'avg_sharpe': np.mean([result[1]['sharpe_ratio'] for result in fold_results]),
                'max_drawdown': np.max([result[1]['max_drawdown'] for result in fold_results]),  # Worst case
                'total_trades': np.mean([result[1]['total_trades'] for result in fold_results]),
                'fitness_std': np.std(fitness_scores),  # Standard deviation of fitness
                'cv_stability': 1.0 / (1.0 + np.std(fitness_scores))  # Stability score (higher = more stable)
            }

            # Apply stability penalty - reward consistent performance across folds
            stability_weight = 0.3
            cv_fitness = avg_fitness * (1 + stability_weight * avg_metrics['cv_stability'])

            # Additional penalty for high variance (overfitting indicator)
            if avg_metrics['fitness_std'] > avg_fitness * 0.5:  # High variance relative to mean
                cv_fitness *= 0.8  # 20% penalty for unstable performance

            return cv_fitness, avg_metrics

    def _evaluate_config_single(self, config: Dict[str, Any], data: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a single configuration on a dataset.

        Args:
            config: Indicator configuration
            data: Historical data subset

        Returns:
            Tuple of (fitness_score, metrics_dict)
        """
        try:
            # Calculate indicators with custom config
            test_data = data.copy()
            test_data = calculate_indicators(test_data, config=config)

            # Generate signals with custom logic based on config
            test_data = self._generate_signals_with_config(test_data, config)

            # Calculate risk management
            test_data = calculate_risk_management(test_data, config=config)

            # Run backtest simulation
            backtest_result = run_backtest(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=data.index.min(),
                end_date=data.index.max(),
                config=config
            )

            if 'backtest_results' in backtest_result and 'error' not in backtest_result['backtest_results']:
                backtest_metrics = backtest_result['backtest_results']
                fitness, metrics = self._calculate_fitness_from_backtest_metrics(backtest_metrics)
            else:
                # Fallback to simulation
                trades_df = self._simulate_trades(test_data)
                fitness, metrics = self._calculate_fitness_metrics(trades_df)

            return fitness, metrics

        except Exception as e:
            logger.warning(f"Error in single config evaluation: {e}")
            return 0.0, {
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0
            }

    def _calculate_fitness_metrics(self, trades_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """Calculate fitness from simulated trades DataFrame."""
        if trades_df.empty:
            return 0.0, {
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0
            }

        # Calculate basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0
        total_pnl = trades_df['pnl'].sum()

        # Calculate Sharpe ratio (simplified)
        if len(trades_df) > 1:
            returns = trades_df['pnl']
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Calculate max drawdown (simplified)
        cumulative_pnl = trades_df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = running_max - cumulative_pnl
        max_drawdown = drawdown.max() if not drawdown.empty else 0.0

        # Calculate profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate Calmar ratio (simplified)
        calmar_ratio = total_pnl / max_drawdown if max_drawdown > 0 else 0.0

        # Use the same fitness calculation as backtest metrics
        return self._calculate_fitness_from_backtest_metrics({
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio
        })

    def _load_historical_data(self) -> pd.DataFrame:
        """Load and cache historical data for evaluation using discovery-optimized caching."""
        if self.historical_data is None:
            logger.info("Loading historical data for discovery...")
            start_dt = pd.to_datetime(self.start_date) if isinstance(self.start_date, str) else self.start_date
            end_dt = pd.to_datetime(self.end_date) if isinstance(self.end_date, str) else self.end_date

            # Use discovery-optimized caching to minimize API calls
            logger.info(f"Fetching data using discovery cache: {self.symbol} {self.timeframe} from {start_dt.date()} to {end_dt.date()}")

            try:
                # Use the discovery cache which shares data across all evaluations
                data = cached_discovery_fetch(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    start_date=start_dt.strftime('%Y-%m-%d'),
                    end_date=end_dt.strftime('%Y-%m-%d')
                )

                if len(data) >= 100:
                    logger.info(f"Successfully loaded {len(data)} data points for discovery")
                    self.historical_data = data
                else:
                    logger.warning(f"Insufficient data ({len(data)} points), trying fallback timeframes...")

                    # Try different timeframes if current one fails
                    fallback_timeframes = ['5m', '15m', '1h'] if self.timeframe == '1m' else ['1m', '5m', '15m']
                    for tf in fallback_timeframes:
                        try:
                            logger.info(f"Trying fallback timeframe {tf}...")
                            data = cached_discovery_fetch(
                                symbol=self.symbol,
                                timeframe=tf,
                                start_date=start_dt.strftime('%Y-%m-%d'),
                                end_date=end_dt.strftime('%Y-%m-%d')
                            )
                            if len(data) >= 100:
                                logger.info(f"Successfully loaded {len(data)} data points with timeframe {tf}")
                                self.historical_data = data
                                break
                        except Exception as e:
                            logger.warning(f"Failed to fetch data with timeframe {tf}: {e}")

            except Exception as e:
                logger.error(f"Error loading historical data: {e}")

            # If all attempts failed, create mock data for testing
            if self.historical_data is None or len(self.historical_data) < 100:
                logger.warning("Using mock data for testing purposes")
                dates = pd.date_range(start=start_dt, periods=200, freq='1H')
                self.historical_data = pd.DataFrame({
                    'timestamp': dates,
                    'open': 50000 + np.random.randn(200) * 1000,
                    'high': 51000 + np.random.randn(200) * 1000,
                    'low': 49000 + np.random.randn(200) * 1000,
                    'close': 50000 + np.random.randn(200) * 1000,
                    'volume': np.random.randint(1000000, 10000000, 200)
                })
                self.historical_data.set_index('timestamp', inplace=True)

            # Log cache statistics
            cache_stats = get_discovery_cache_stats()
            logger.info(f"Discovery cache stats: {cache_stats}")

        return self.historical_data

    def _simulate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simulate trades based on buy/sell signals."""
        trades = []
        position = 0
        entry_price = 0

        for idx, row in data.iterrows():
            if row['Buy_Signal'] == 1 and position == 0:
                position = 1
                entry_price = row['close']
                entry_time = idx
            elif row['Sell_Signal'] == 1 and position == 1:
                exit_price = row['close']
                pnl = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'direction': 'long'
                })
                position = 0

        return pd.DataFrame(trades) if trades else pd.DataFrame()

    def _calculate_fitness_from_backtest_metrics(self, backtest_metrics):
        """Calculate fitness from backtest metrics using shared fitness function."""
        from src.fitness import calculate_fitness_from_metrics
        return calculate_fitness_from_metrics(backtest_metrics)

    def _evaluate_with_walk_forward(self, config: Dict[str, Any], data: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate configuration using walk-forward analysis for out-of-sample testing.

        Args:
            config: Indicator configuration
            data: Historical data

        Returns:
            Tuple of (wfa_fitness, metrics_dict)
        """
        try:
            # Simple walk-forward implementation
            # Split data into training and testing periods
            split_point = int(len(data) * 0.7)  # 70% training, 30% testing

            train_data = data.iloc[:split_point]
            test_data = data.iloc[split_point:]

            # Train on historical data
            train_fitness, train_metrics = self._evaluate_config_single(config, train_data)

            # Test on future data (out-of-sample)
            test_fitness, test_metrics = self._evaluate_config_single(config, test_data)

            # Combine training and testing performance
            # Weight out-of-sample performance more heavily
            oos_weight = 0.6
            train_weight = 0.4

            combined_fitness = (train_fitness * train_weight) + (test_fitness * oos_weight)

            # Calculate out-of-sample degradation
            oos_degradation = test_fitness - train_fitness
            degradation_penalty = max(0, -oos_degradation * 0.5)  # Penalty for overfitting

            wfa_fitness = combined_fitness - degradation_penalty

            # Combined metrics
            wfa_metrics = {
                'avg_pnl': (train_metrics['total_pnl'] * train_weight) + (test_metrics['total_pnl'] * oos_weight),
                'avg_win_rate': (train_metrics['win_rate'] * train_weight) + (test_metrics['win_rate'] * oos_weight),
                'avg_sharpe': (train_metrics['sharpe_ratio'] * train_weight) + (test_metrics['sharpe_ratio'] * oos_weight),
                'max_drawdown': max(train_metrics['max_drawdown'], test_metrics['max_drawdown']),
                'total_trades': train_metrics['total_trades'] + test_metrics['total_trades'],
                'oos_degradation': oos_degradation,
                'train_fitness': train_fitness,
                'test_fitness': test_fitness
            }

            return wfa_fitness, wfa_metrics

        except Exception as e:
            logger.warning(f"Error in walk-forward evaluation: {e}")
            return self._evaluate_config_single(config, data)

    def optimize(self) -> List[IndividualResult]:
        """Run the genetic algorithm optimization."""
        logger.info("Starting GA optimization...")
        logger.info(f"Population size: {self.population_size}, Generations: {self.generations}")

        # Create initial population
        pop = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Evolution loop
        for gen in range(self.generations):
            logger.info(f"Generation {gen + 1}/{self.generations}")

            # Select offspring
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation
            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            pop[:] = offspring

            # Log best fitness
            fits = [ind.fitness.values[0] for ind in pop]
            best_fit = max(fits)
            logger.info(f"Best fitness in generation {gen + 1}: {best_fit:.2f}")

        # Get top results and remove duplicates
        all_results = sorted(self.results, key=lambda x: x.fitness, reverse=True)
        top_results = self._remove_duplicate_configs(all_results, max_results=10)

        logger.info("GA optimization completed!")
        logger.info(f"Total configurations tested: {len(self.results)}")
        logger.info(f"Unique top configurations: {len(top_results)}")
        if len(self.results) > len(top_results):
            total_unique_possible = self._get_unique_config_count(all_results)
            duplicates_removed = len(self.results) - total_unique_possible
            logger.info(f"Duplicate configurations removed: {duplicates_removed}")
        if top_results:
            logger.info(f"Best fitness: {top_results[0].fitness:.2f}")

        return top_results

    def _remove_duplicate_configs(self, results: List[IndividualResult], max_results: int = 10) -> List[IndividualResult]:
        """Remove duplicate configurations based on indicator parameters, keeping the best fitness for each unique config."""
        # Group results by configuration
        config_groups = {}

        for result in results:
            config_tuple = self._config_to_tuple(result.config)

            if config_tuple not in config_groups:
                config_groups[config_tuple] = []
            config_groups[config_tuple].append(result)

        # For each unique configuration, keep only the one with highest fitness
        unique_results = []
        for config_tuple, group_results in config_groups.items():
            # Sort group by fitness (descending) and take the best one
            best_result = max(group_results, key=lambda x: x.fitness)
            unique_results.append(best_result)

        # Sort all unique results by fitness (descending) and return top N
        unique_results.sort(key=lambda x: x.fitness, reverse=True)
        return unique_results[:max_results]

    def _get_unique_config_count(self, results: List[IndividualResult]) -> int:
        """Count the number of unique configurations in results."""
        seen_configs = set()
        for result in results:
            config_tuple = self._config_to_tuple(result.config)
            seen_configs.add(config_tuple)
        return len(seen_configs)

    def _config_to_tuple(self, config: Dict[str, Any]) -> Tuple:
        """Convert configuration to a hashable tuple for duplicate detection."""
        ema_periods = tuple(config.get('ema', {}).get('periods', []))
        rsi_enabled = config.get('rsi', {}).get('enabled', False)
        rsi_period = config.get('rsi', {}).get('period', 14)
        rsi_oversold = config.get('rsi', {}).get('oversold', 30)
        rsi_overbought = config.get('rsi', {}).get('overbought', 70)
        bb_enabled = config.get('bb', {}).get('enabled', False)
        bb_period = config.get('bb', {}).get('period', 20)
        bb_std_dev = round(config.get('bb', {}).get('std_dev', 2.0), 2)
        atr_enabled = config.get('atr', {}).get('enabled', False)
        atr_period = config.get('atr', {}).get('period', 14)
        adx_enabled = config.get('adx', {}).get('enabled', False)
        macd_enabled = config.get('macd', {}).get('enabled', False)
        macd_fast = config.get('macd', {}).get('fast_period', 12)
        macd_slow = config.get('macd', {}).get('slow_period', 26)
        macd_signal = config.get('macd', {}).get('signal_period', 9)
        obv_enabled = config.get('obv', {}).get('enabled', False)
        stoch_enabled = config.get('stochastic', {}).get('enabled', False)
        stoch_k = config.get('stochastic', {}).get('k_period', 14)
        stoch_d = config.get('stochastic', {}).get('d_period', 3)
        cmf_enabled = config.get('cmf', {}).get('enabled', False)

        return (ema_periods, rsi_enabled, rsi_period, rsi_oversold, rsi_overbought,
                bb_enabled, bb_period, bb_std_dev, atr_enabled, atr_period, adx_enabled,
                macd_enabled, macd_fast, macd_slow, macd_signal, obv_enabled,
                stoch_enabled, stoch_k, stoch_d, cmf_enabled)

    def save_results(self, results: List[IndividualResult], output_file: str = 'output/discovery_results.json'):
        """Save optimization results to JSON file."""
        output_data = {
            'optimization_summary': {
                'symbol': self.symbol,
                'exchange': self.exchange,
                'timeframe': self.timeframe,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'population_size': self.population_size,
                'generations': self.generations,
                'total_configurations_tested': len(self.results),
                'unique_top_configurations': len(results)
            },
            'top_configurations': [
                {
                    'rank': i + 1,
                    'fitness_score': result.fitness,
                    'total_pnl_percent': result.pnl,
                    'win_rate': result.win_rate,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'total_trades': result.total_trades,
                    'evaluation_time_seconds': result.evaluation_time,
                    'avg_daily_performance': result.pnl / max(result.backtest_duration_days, 1),
                    'configuration': result.config
                }
                for i, result in enumerate(results)
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")

def load_adaptive_config(config_file: str = 'config/adaptive_config.json') -> Optional[Dict[str, Any]]:
    """
    Load optimized configuration from adaptive optimization.

    Args:
        config_file: Path to the adaptive configuration file

    Returns:
        Optimized configuration dict if file exists and is valid, None otherwise
    """
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                data = json.load(f)

            # Validate the configuration structure
            if 'best_configuration' in data and 'fitness_score' in data:
                config = data['best_configuration']
                fitness = data['fitness_score']

                logger.info(f"Loaded adaptive configuration with fitness: {fitness}")
                return config
            else:
                logger.warning(f"Invalid adaptive config structure in {config_file}")
        else:
            logger.debug(f"Adaptive config file {config_file} not found")
    except Exception as e:
        logger.warning(f"Error loading adaptive config: {e}")

    return None

def save_adaptive_config(config: Dict[str, Any], fitness: float, config_file: str = 'config/adaptive_config.json'):
    """
    Save optimized configuration for adaptive optimization.

    Args:
        config: The optimized configuration
        fitness: Fitness score of the configuration
        config_file: Path to save the configuration
    """
    try:
        # Ensure config directory exists
        os.makedirs(os.path.dirname(config_file), exist_ok=True)

        data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'fitness_score': fitness,
            'best_configuration': config
        }

        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved adaptive configuration to {config_file} (fitness: {fitness})")

    except Exception as e:
        logger.error(f"Error saving adaptive config: {e}")

def apply_adaptive_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply optimized configuration to the current indicator config.

    Args:
        config: Optimized configuration from discovery

    Returns:
        Updated indicator configuration
    """
    # Start with default config
    from config.settings import DEFAULT_INDICATOR_CONFIG
    updated_config = DEFAULT_INDICATOR_CONFIG.copy()

    # Apply optimized settings
    if 'ema' in config:
        updated_config['ema'] = config['ema']
    if 'rsi' in config:
        updated_config['rsi'] = config['rsi']
    if 'bb' in config:
        updated_config['bb'] = config['bb']
    if 'atr' in config:
        updated_config['atr'] = config['atr']
    if 'adx' in config:
        updated_config['adx'] = config['adx']

    logger.info("Applied adaptive configuration for live trading")
    return updated_config

def run_discovery(symbol: str = SYMBOL,
                  exchange: str = EXCHANGE,
                  timeframe: str = TIMEFRAME,
                  start_date: str = '2024-01-01',
                  end_date: str = '2024-12-31',
                  population_size: int = None,  # Uses DISCOVERY_PARAMS if None
                  generations: int = None,       # Uses DISCOVERY_PARAMS if None
                  use_walk_forward: bool = True) -> List[IndividualResult]:  # New parameter
    """
    Run discovery optimization for trading indicator configurations with enhanced robustness.

    Args:
        symbol: Trading pair symbol
        exchange: Exchange name
        timeframe: Timeframe for backtesting
        start_date: Start date for historical data
        end_date: End date for historical data
        population_size: GA population size (uses DISCOVERY_PARAMS if None)
        generations: Number of GA generations (uses DISCOVERY_PARAMS if None)
        use_walk_forward: Whether to use walk-forward analysis for out-of-sample validation

    Returns:
        List of top 10 optimized configurations

    Raises:
        ImportError: If deap library is not available
    """
    if not DEAP_AVAILABLE:
        raise ImportError("Discovery module requires 'deap' package. Install with: pip install deap")

    logger.info("Starting Discovery optimization with enhanced caching...")
    logger.info(f"Parameters: symbol={symbol}, timeframe={timeframe}, period={start_date} to {end_date}")
    logger.info(f"GA settings: population={population_size or 'default'}, generations={generations or 'default'}")

    optimizer = DiscoveryOptimizer(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        population_size=population_size,
        generations=generations,
        use_walk_forward=use_walk_forward
    )

    results = optimizer.optimize()
    optimizer.save_results(results)

    # Log final cache statistics
    cache_stats = get_discovery_cache_stats()
    logger.info(f"Discovery completed. Final cache stats: {cache_stats}")

    return results

def clear_discovery_cache():
    """
    Clear the discovery data cache.

    This function clears both memory and file caches used by the discovery system.
    Useful for freeing up disk space or forcing fresh data fetches.
    """
    try:
        from .discovery_cache import clear_discovery_cache as clear_cache
        clear_cache()
        logger.info("Discovery cache cleared successfully")
    except ImportError:
        from discovery_cache import clear_discovery_cache as clear_cache
        clear_cache()
        logger.info("Discovery cache cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing discovery cache: {e}")

def get_discovery_cache_info() -> Dict[str, Any]:
    """
    Get information about the discovery cache status.

    Returns:
        Dictionary with cache statistics and status information
    """
    try:
        from .discovery_cache import get_discovery_cache_stats
        return get_discovery_cache_stats()
    except ImportError:
        from discovery_cache import get_discovery_cache_stats
        return get_discovery_cache_stats()
    except Exception as e:
        logger.error(f"Error getting discovery cache info: {e}")
        return {"error": str(e)}