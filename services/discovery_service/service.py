#!/usr/bin/env python3
"""
Discovery Service - Genetic Algorithm Optimization for Trading Indicators

This service provides genetic algorithm-based optimization for trading indicator
configurations, enabling automated discovery of optimal parameter combinations
for maximum trading performance.
"""

import asyncio
import json
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

# Check if deap is available
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("⚠️  DEAP library not available. Discovery Service will not function.")
    print("   Install with: pip install deap")

# Import dependencies
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators
from src.signal_generator import generate_signals, calculate_risk_management
from src.backtester import run_backtest
from src.logging_config import logger
from src.fitness import calculate_fitness_from_metrics

# Mock EventSystem for service integration
class Event:
    def __init__(self, type: str, data: Dict[str, Any]):
        self.type = type
        self.data = data

class EventSystem:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event_type: str, handler):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    async def publish(self, event: Event):
        if event.type in self.subscribers:
            for handler in self.subscribers[event.type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event.type}: {e}")

@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    optimization_id: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    population_size: int
    generations: int
    best_fitness: float
    best_config: Dict[str, Any]
    total_evaluations: int
    duration_seconds: float
    status: str  # 'completed', 'failed', 'running'
    error_message: Optional[str] = None
    top_configurations: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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

class DiscoveryService:
    """
    Async Discovery Service for genetic algorithm optimization of trading indicators.

    Provides comprehensive optimization capabilities with support for:
    - Multiple indicator combinations
    - Cross-validation and walk-forward analysis
    - Event-driven architecture integration
    - Async operation for scalability
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

    def __init__(self, event_system: Optional[EventSystem] = None):
        """
        Initialize the Discovery Service.

        Args:
            event_system: Optional event system for notifications
        """
        if not DEAP_AVAILABLE:
            raise ImportError("Discovery Service requires 'deap' package. Install with: pip install deap")

        self.event_system = event_system or EventSystem()
        self.active_optimizations = {}  # optimization_id -> task
        self.optimization_results = {}  # optimization_id -> OptimizationResult

        # Setup GA components
        self._setup_ga()

        logger.info("Discovery Service initialized")

    def _setup_ga(self):
        """Setup DEAP genetic algorithm components."""
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Define the toolbox
        self.toolbox = base.Toolbox()

        # Attribute generators for each parameter
        self.toolbox.register("combination_idx", np.random.randint, 0, len(self.INDICATOR_COMBINATIONS))

        # EMA periods: 5-50 for short, 10-200 for long
        self.toolbox.register("ema_short", np.random.randint, 5, 50)
        self.toolbox.register("ema_long", np.random.randint, 10, 200)

        # RSI period: 5-30
        self.toolbox.register("rsi_period", np.random.randint, 5, 30)

        # RSI thresholds: 20-40 for oversold, 60-80 for overbought
        self.toolbox.register("rsi_oversold", np.random.randint, 20, 40)
        self.toolbox.register("rsi_overbought", np.random.randint, 60, 80)

        # BB period: 10-50, std_dev: 1.5-3.0
        self.toolbox.register("bb_period", np.random.randint, 10, 50)
        self.toolbox.register("bb_std_dev", np.random.uniform, 1.5, 3.0)

        # ATR period: 5-30
        self.toolbox.register("atr_period", np.random.randint, 5, 30)

        # MACD parameters: fast(8-20), slow(20-40), signal(5-15)
        self.toolbox.register("macd_fast", np.random.randint, 8, 20)
        self.toolbox.register("macd_slow", np.random.randint, 20, 40)
        self.toolbox.register("macd_signal", np.random.randint, 5, 15)

        # Stochastic parameters: k(5-21), d(3-8)
        self.toolbox.register("stoch_k", np.random.randint, 5, 21)
        self.toolbox.register("stoch_d", np.random.randint, 3, 8)

        # ADX period: 5-30
        self.toolbox.register("adx_period", np.random.randint, 5, 30)

        # OBV moving average period: 10-50
        self.toolbox.register("obv_ma_period", np.random.randint, 10, 50)

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
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _mutate_individual(self, individual, indpb=0.2):
        """Custom mutation function that respects parameter ranges."""
        for i in range(len(individual)):
            if np.random.random() < indpb:
                if i == 0:  # combination_idx
                    individual[i] = np.random.randint(0, len(self.INDICATOR_COMBINATIONS))
                elif i == 1:  # ema_short
                    individual[i] = np.random.randint(5, 50)
                elif i == 2:  # ema_long
                    individual[i] = np.random.randint(10, 200)
                elif i == 3:  # rsi_period
                    individual[i] = np.random.randint(5, 30)
                elif i == 4:  # rsi_oversold
                    individual[i] = np.random.randint(20, 40)
                elif i == 5:  # rsi_overbought
                    individual[i] = np.random.randint(60, 80)
                elif i == 6:  # bb_period
                    individual[i] = np.random.randint(10, 50)
                elif i == 7:  # bb_std_dev
                    individual[i] = np.random.uniform(1.5, 3.0)
                elif i == 8:  # atr_period
                    individual[i] = np.random.randint(5, 30)
                elif i == 9:  # macd_fast
                    individual[i] = np.random.randint(8, 20)
                elif i == 10:  # macd_slow
                    individual[i] = np.random.randint(20, 40)
                elif i == 11:  # macd_signal
                    individual[i] = np.random.randint(5, 15)
                elif i == 12:  # stoch_k
                    individual[i] = np.random.randint(5, 21)
                elif i == 13:  # stoch_d
                    individual[i] = np.random.randint(3, 8)
                elif i == 14:  # adx_period
                    individual[i] = np.random.randint(5, 30)
                elif i == 15:  # obv_ma_period
                    individual[i] = np.random.randint(10, 50)
        return individual,

    def _individual_to_config(self, individual) -> Dict[str, Any]:
        """Convert GA individual to indicator configuration dict."""
        combination_idx = int(individual[0])
        combination_names = list(self.INDICATOR_COMBINATIONS.keys())
        if combination_idx >= len(combination_names):
            combination_idx = 0
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
                'std_dev': round(float(individual[7]), 2)  # bb_std_dev
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
            'cmf': {'enabled': False},
            'fibonacci': {'enabled': False},
            'combination_name': combination['name'],
            'enabled_indicators': enabled_indicators
        }

        return config

    async def run_optimization_async(self,
                                   optimization_id: str,
                                   symbol: str,
                                   timeframe: str,
                                   start_date: str,
                                   end_date: str,
                                   population_size: int = 50,
                                   generations: int = 20,
                                   use_walk_forward: bool = True) -> Dict[str, Any]:
        """
        Run async genetic algorithm optimization.

        Args:
            optimization_id: Unique identifier for this optimization run
            symbol: Trading symbol
            timeframe: Timeframe for backtesting
            start_date: Start date for historical data
            end_date: End date for historical data
            population_size: GA population size
            generations: Number of GA generations
            use_walk_forward: Whether to use walk-forward analysis

        Returns:
            Dict with optimization results
        """
        try:
            # Create optimization result object
            result = OptimizationResult(
                optimization_id=optimization_id,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                population_size=population_size,
                generations=generations,
                best_fitness=0.0,
                best_config={},
                total_evaluations=0,
                duration_seconds=0.0,
                status='running',
                created_at=datetime.now().isoformat()
            )

            self.optimization_results[optimization_id] = result

            # Publish start event
            await self.event_system.publish(Event(
                type="discovery.optimization.started",
                data={"optimization_id": optimization_id}
            ))

            start_time = time.time()

            # Run optimization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._run_optimization_sync,
                symbol, timeframe, start_date, end_date,
                population_size, generations, use_walk_forward
            )

            duration = time.time() - start_time

            # Update result
            result.status = 'completed'
            result.duration_seconds = duration
            result.best_fitness = results[0].fitness if results else 0.0
            result.best_config = results[0].config if results else {}
            result.total_evaluations = len(self._evaluation_results) if hasattr(self, '_evaluation_results') else 0
            result.top_configurations = [r.__dict__ for r in results[:10]] if results else []

            # Publish completion event
            await self.event_system.publish(Event(
                type="discovery.optimization.completed",
                data={
                    "optimization_id": optimization_id,
                    "best_fitness": result.best_fitness,
                    "duration": duration
                }
            ))

            return {
                "success": True,
                "optimization_id": optimization_id,
                "best_fitness": result.best_fitness,
                "best_config": result.best_config,
                "total_evaluations": result.total_evaluations,
                "duration_seconds": duration,
                "top_configurations": result.top_configurations
            }

        except Exception as e:
            logger.error(f"Optimization failed: {e}")

            # Update result with error
            if optimization_id in self.optimization_results:
                self.optimization_results[optimization_id].status = 'failed'
                self.optimization_results[optimization_id].error_message = str(e)

            # Publish error event
            await self.event_system.publish(Event(
                type="discovery.optimization.failed",
                data={"optimization_id": optimization_id, "error": str(e)}
            ))

            return {
                "success": False,
                "error": str(e),
                "optimization_id": optimization_id
            }

    def _run_optimization_sync(self, symbol: str, timeframe: str, start_date: str,
                             end_date: str, population_size: int, generations: int,
                             use_walk_forward: bool) -> List[IndividualResult]:
        """Synchronous optimization execution."""
        try:
            # Load historical data
            data = self._load_historical_data(symbol, timeframe, start_date, end_date)
            if data.empty:
                raise ValueError("No historical data available for optimization")

            # Store data for evaluation
            self._historical_data = data
            self._use_walk_forward = use_walk_forward

            # Reset evaluation results
            self._evaluation_results = []

            # Create initial population
            pop = self.toolbox.population(n=population_size)

            # Evaluate initial population
            fitnesses = list(map(self.toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            # Evolution loop
            for gen in range(generations):
                logger.info(f"Generation {gen + 1}/{generations}")

                # Select offspring
                offspring = self.toolbox.select(pop, len(pop))
                offspring = list(map(self.toolbox.clone, offspring))

                # Apply crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if np.random.random() < 0.8:  # Crossover rate
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Apply mutation
                for mutant in offspring:
                    if np.random.random() < 0.2:  # Mutation rate
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Evaluate offspring
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Replace population
                pop[:] = offspring

            # Get top results
            all_results = sorted(self._evaluation_results, key=lambda x: x.fitness, reverse=True)
            top_results = self._remove_duplicate_configs(all_results, max_results=10)

            logger.info(f"Optimization completed. Best fitness: {top_results[0].fitness if top_results else 0}")
            return top_results

        except Exception as e:
            logger.error(f"Sync optimization failed: {e}")
            raise

    def _evaluate_individual(self, individual) -> Tuple[float]:
        """Evaluate fitness of an individual configuration."""
        try:
            start_time = time.time()

            # Convert individual to config
            config = self._individual_to_config(individual)

            # Evaluate configuration
            fitness, metrics = self._evaluate_config_single(config, self._historical_data)

            evaluation_time = time.time() - start_time

            # Store result
            result = IndividualResult(
                config=config,
                fitness=fitness,
                pnl=metrics.get('total_pnl', 0),
                win_rate=metrics.get('win_rate', 0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0),
                max_drawdown=metrics.get('max_drawdown', 0),
                total_trades=metrics.get('total_trades', 0),
                evaluation_time=evaluation_time,
                backtest_duration_days=len(self._historical_data)
            )
            self._evaluation_results.append(result)

            return (fitness,)

        except Exception as e:
            logger.warning(f"Error evaluating individual: {e}")
            return (0.0,)

    def _evaluate_config_single(self, config: Dict[str, Any], data: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """Evaluate a single configuration on dataset."""
        try:
            # Calculate indicators
            test_data = data.copy()
            test_data = calculate_indicators(test_data, config=config)

            # Generate signals
            test_data = generate_signals(test_data, config=config)

            # Calculate risk management
            test_data = calculate_risk_management(test_data, config=config)

            # Run backtest
            backtest_result = run_backtest(
                symbol="BTC/USDT",  # Placeholder
                timeframe="1d",     # Placeholder
                start_date=data.index.min(),
                end_date=data.index.max(),
                config=config
            )

            if 'backtest_results' in backtest_result and 'error' not in backtest_result['backtest_results']:
                metrics = backtest_result['backtest_results']
                fitness, _ = calculate_fitness_from_metrics(metrics)
                return fitness, metrics
            else:
                return 0.0, {}

        except Exception as e:
            logger.warning(f"Error in config evaluation: {e}")
            return 0.0, {}

    def _load_historical_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data for optimization."""
        try:
            data = fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                limit=5000
            )

            if data.empty or len(data) < 100:
                logger.warning(f"Insufficient data: {len(data)} points")
                # Return mock data for testing
                dates = pd.date_range(start=start_date, periods=200, freq='1H')
                return pd.DataFrame({
                    'timestamp': dates,
                    'open': 50000 + np.random.randn(200) * 1000,
                    'high': 51000 + np.random.randn(200) * 1000,
                    'low': 49000 + np.random.randn(200) * 1000,
                    'close': 50000 + np.random.randn(200) * 1000,
                    'volume': np.random.randint(1000000, 10000000, 200)
                }).set_index('timestamp')

            return data

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise

    def _remove_duplicate_configs(self, results: List[IndividualResult], max_results: int = 10) -> List[IndividualResult]:
        """Remove duplicate configurations."""
        config_groups = {}
        for result in results:
            config_tuple = tuple(sorted(result.config.items()))
            if config_tuple not in config_groups:
                config_groups[config_tuple] = []
            config_groups[config_tuple].append(result)

        unique_results = []
        for group in config_groups.values():
            best_result = max(group, key=lambda x: x.fitness)
            unique_results.append(best_result)

        return sorted(unique_results, key=lambda x: x.fitness, reverse=True)[:max_results]

    async def get_optimization_status(self, optimization_id: str) -> Dict[str, Any]:
        """Get status of an optimization run."""
        if optimization_id in self.optimization_results:
            result = self.optimization_results[optimization_id]
            return result.to_dict()
        else:
            return {"error": "Optimization not found", "optimization_id": optimization_id}

    async def list_active_optimizations(self) -> List[Dict[str, Any]]:
        """List all active optimizations."""
        return [
            result.to_dict()
            for result in self.optimization_results.values()
            if result.status == 'running'
        ]

    async def cancel_optimization(self, optimization_id: str) -> Dict[str, Any]:
        """Cancel an active optimization."""
        if optimization_id in self.active_optimizations:
            # Cancel the task (simplified - in real implementation would need proper cancellation)
            self.active_optimizations[optimization_id].cancel()
            del self.active_optimizations[optimization_id]

            if optimization_id in self.optimization_results:
                self.optimization_results[optimization_id].status = 'cancelled'

            return {"success": True, "optimization_id": optimization_id}
        else:
            return {"error": "Optimization not found or not active", "optimization_id": optimization_id}