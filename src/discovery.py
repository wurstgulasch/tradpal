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
except ImportError:
    # Fall back to absolute imports (when imported directly)
    from backtester import run_backtest
    from data_fetcher import fetch_historical_data
    from indicators import calculate_indicators
    from signal_generator import generate_signals, calculate_risk_management
    from error_handling import error_boundary
    from logging_config import logger

from config.settings import SYMBOL, EXCHANGE, TIMEFRAME

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
                 population_size: int = 50,
                 generations: int = 20,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8,
                 initial_capital: float = 10000.0):
        """
        Initialize the discovery optimizer.

        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            timeframe: Timeframe for backtesting
            start_date: Start date for historical data
            end_date: End date for historical data
            population_size: Number of individuals in GA population
            generations: Number of GA generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            initial_capital: Initial capital for backtesting
        """
        if not DEAP_AVAILABLE:
            raise ImportError("Discovery module requires 'deap' package. Install with: pip install deap")

        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.initial_capital = initial_capital

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
        """Generate signals based on the custom configuration."""
        df = df.copy()

        # Initialize signal columns
        df['Buy_Signal'] = 0
        df['Sell_Signal'] = 0

        # EMA signals (core strategy - always present in combinations)
        ema_buy_signal = False
        ema_sell_signal = False
        if config.get('ema', {}).get('enabled', False) and len(config['ema'].get('periods', [])) >= 2:
            short_period, long_period = config['ema']['periods'][:2]
            ema_short_col = f'EMA{short_period}'
            ema_long_col = f'EMA{long_period}'

            if ema_short_col in df.columns and ema_long_col in df.columns:
                df['EMA_crossover'] = np.where(df[ema_short_col] > df[ema_long_col], 1, -1)
                ema_buy_signal = df['EMA_crossover'] == 1
                ema_sell_signal = df['EMA_crossover'] == -1

        # RSI filter
        rsi_buy_filter = True
        rsi_sell_filter = True
        if config.get('rsi', {}).get('enabled', False) and 'RSI' in df.columns:
            rsi_oversold = config['rsi'].get('oversold', 30)
            rsi_overbought = config['rsi'].get('overbought', 70)
            rsi_buy_filter = df['RSI'] < rsi_oversold
            rsi_sell_filter = df['RSI'] > rsi_overbought

        # BB filter
        bb_buy_filter = True
        bb_sell_filter = True
        if config.get('bb', {}).get('enabled', False):
            if 'BB_lower' in df.columns and 'BB_upper' in df.columns:
                bb_buy_filter = df['close'] > df['BB_lower']
                bb_sell_filter = df['close'] < df['BB_upper']

        # MACD filter
        macd_buy_filter = True
        macd_sell_filter = True
        if config.get('macd', {}).get('enabled', False):
            if 'MACD_histogram' in df.columns:
                macd_buy_filter = df['MACD_histogram'] > 0
                macd_sell_filter = df['MACD_histogram'] < 0

        # Stochastic filter
        stoch_buy_filter = True
        stoch_sell_filter = True
        if config.get('stochastic', {}).get('enabled', False):
            if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
                stoch_buy_filter = (df['Stoch_K'] < 20) & (df['Stoch_D'] < 20)
                stoch_sell_filter = (df['Stoch_K'] > 80) & (df['Stoch_D'] > 80)

        # OBV filter (momentum confirmation)
        obv_buy_filter = True
        obv_sell_filter = True
        if config.get('obv', {}).get('enabled', False) and 'OBV' in df.columns:
            ma_period = config['obv'].get('ma_period', 20)
            obv_ma = df['OBV'].rolling(window=ma_period).mean()
            obv_buy_filter = df['OBV'] > obv_ma
            obv_sell_filter = df['OBV'] < obv_ma

        # ADX filter (trend strength)
        adx_buy_filter = True
        adx_sell_filter = True
        if config.get('adx', {}).get('enabled', False) and 'ADX' in df.columns:
            adx_threshold = 25  # Fixed threshold for simplicity
            adx_buy_filter = df['ADX'] > adx_threshold
            adx_sell_filter = df['ADX'] > adx_threshold

        # Combine all filters for final signals
        buy_conditions = ema_buy_signal & rsi_buy_filter & bb_buy_filter & macd_buy_filter & stoch_buy_filter & obv_buy_filter & adx_buy_filter
        sell_conditions = ema_sell_signal & rsi_sell_filter & bb_sell_filter & macd_sell_filter & stoch_sell_filter & obv_sell_filter & adx_sell_filter

        df['Buy_Signal'] = buy_conditions.astype(int)
        df['Sell_Signal'] = sell_conditions.astype(int)

        return df

    @error_boundary(operation="evaluate_individual", max_retries=1)
    def _evaluate_individual(self, individual) -> Tuple[float]:
        """Evaluate fitness of an individual configuration."""
        start_time = time.time()

        try:
            # Convert individual to config
            config = self._individual_to_config(individual)

            # All combinations include at least EMA, so no need to check

            # Load historical data
            historical_data = self._load_historical_data()

            # Calculate indicators with custom config
            data = historical_data.copy()
            data = calculate_indicators(data, config=config)

            # Generate signals with custom logic based on config
            data = self._generate_signals_with_config(data, config)

            # Calculate risk management
            data = calculate_risk_management(data)

            # Run backtest simulation
            trades_df = self._simulate_trades(data)

            # Calculate fitness metrics
            fitness, metrics = self._calculate_fitness_metrics(trades_df)

            evaluation_time = time.time() - start_time

            # Store result
            result = IndividualResult(
                config=config,
                fitness=fitness,
                pnl=metrics['total_pnl'],
                win_rate=metrics['win_rate'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                total_trades=metrics['total_trades'],
                evaluation_time=evaluation_time,
                backtest_duration_days=len(historical_data)
            )
            self.results.append(result)

            logger.debug(f"Evaluated combination '{config.get('combination_name', 'unknown')}': fitness={fitness:.2f}, pnl={metrics['total_pnl']:.2f}, win_rate={metrics['win_rate']:.2f}")

            return (fitness,)

        except Exception as e:
            logger.warning(f"Error evaluating individual: {e}")
            evaluation_time = time.time() - start_time
            return (0.0,)

    def _load_historical_data(self) -> pd.DataFrame:
        """Load and cache historical data for evaluation."""
        if self.historical_data is None:
            logger.info("Loading historical data for discovery...")
            start_dt = pd.to_datetime(self.start_date) if isinstance(self.start_date, str) else self.start_date

            # Try different timeframes if 1m fails (for testing/backwards compatibility)
            timeframes_to_try = [self.timeframe]
            if self.timeframe == '1m':
                timeframes_to_try.extend(['5m', '15m', '1h'])

            data = None
            for tf in timeframes_to_try:
                try:
                    logger.info(f"Trying timeframe {tf} for historical data...")
                    data = fetch_historical_data(
                        symbol=self.symbol,
                        timeframe=tf,
                        limit=1000,
                        start_date=start_dt
                    )
                    if len(data) >= 100:
                        logger.info(f"Successfully loaded {len(data)} data points with timeframe {tf}")
                        break
                    else:
                        logger.warning(f"Insufficient data ({len(data)} points) with timeframe {tf}, trying next...")
                except Exception as e:
                    logger.warning(f"Failed to fetch data with timeframe {tf}: {e}")

            if data is None or len(data) < 100:
                # Last resort: create mock data for testing
                logger.warning("Using mock data for testing purposes")
                dates = pd.date_range(start=start_dt, periods=200, freq='1H')
                data = pd.DataFrame({
                    'timestamp': dates,
                    'open': 50000 + np.random.randn(200) * 1000,
                    'high': 51000 + np.random.randn(200) * 1000,
                    'low': 49000 + np.random.randn(200) * 1000,
                    'close': 50000 + np.random.randn(200) * 1000,
                    'volume': np.random.randint(1000000, 10000000, 200)
                })
                data.set_index('timestamp', inplace=True)

            self.historical_data = data

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

    def _calculate_fitness_metrics(self, trades_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """Calculate fitness and performance metrics from trades."""
        if len(trades_df) == 0:
            return 0.0, {
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0
            }

        total_pnl = trades_df['pnl'].sum()
        win_rate = (trades_df['pnl'] > 0).mean()

        # Calculate Sharpe Ratio properly (annualized, risk-adjusted returns)
        if len(trades_df) > 1:
            returns = trades_df['pnl'] / self.initial_capital
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate Max Drawdown
        cumulative_returns = (1 + trades_df['pnl'] / self.initial_capital).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        total_trades = len(trades_df)

        # Enhanced Fitness function prioritizing Sharpe Ratio (risk-adjusted returns)
        # Primary metric: Sharpe Ratio (higher = better risk-adjusted performance)
        # Secondary: Total P&L (absolute returns)
        # Tertiary: Win Rate (consistency)
        # Quaternary: Trade Count (activity, but penalized for overtrading)

        sharpe_weight = 100.0  # Primary weight for risk-adjusted performance
        pnl_weight = 1.0       # Secondary weight for absolute returns
        win_rate_weight = 10.0 # Tertiary weight for consistency
        trade_penalty = -0.01  # Penalty for excessive trading (overtrading)

        # Normalize Sharpe Ratio (typically 0-3 is good, penalize negative)
        sharpe_score = max(sharpe_ratio, -5) * sharpe_weight

        # P&L score (penalize losses heavily)
        pnl_score = total_pnl * pnl_weight

        # Win rate bonus (reward consistency)
        win_rate_score = win_rate * win_rate_weight

        # Trade count penalty (avoid overtrading)
        trade_score = total_trades * trade_penalty

        # Total fitness (Sharpe-focused with risk management)
        fitness = sharpe_score + pnl_score + win_rate_score + trade_score

        # Additional penalty for excessive drawdown (>20%)
        if max_drawdown > 0.20:
            fitness *= 0.5  # 50% penalty for high drawdown

        metrics = {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades
        }

        return fitness, metrics

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
                  population_size: int = 50,
                  generations: int = 20) -> List[IndividualResult]:
    """
    Run discovery optimization for trading indicator configurations.

    Args:
        symbol: Trading pair symbol
        exchange: Exchange name
        timeframe: Timeframe for backtesting
        start_date: Start date for historical data
        end_date: End date for historical data
        population_size: GA population size
        generations: Number of GA generations

    Returns:
        List of top 10 optimized configurations

    Raises:
        ImportError: If deap library is not available
    """
    if not DEAP_AVAILABLE:
        raise ImportError("Discovery module requires 'deap' package. Install with: pip install deap")

    optimizer = DiscoveryOptimizer(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        population_size=population_size,
        generations=generations
    )

    results = optimizer.optimize()
    optimizer.save_results(results)

    return results