"""
Backtesting Service - Microservice for historical trading strategy backtesting.

This service provides comprehensive backtesting capabilities including:
- Traditional indicator-based strategies
- ML-enhanced strategies (traditional ML, LSTM, Transformer)
- Multi-symbol and multi-timeframe backtesting
- Performance metrics calculation
- Walk-forward optimization
- Multi-model comparison

The service integrates with the event-driven architecture and provides
async endpoints for backtesting operations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from config.settings import (
    SYMBOL, TIMEFRAME, EXCHANGE, CAPITAL,
    OUTPUT_FILE, ML_ENABLED, ML_CONFIDENCE_THRESHOLD
)
from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators
from src.signal_generator import generate_signals
from src.risk_manager import RiskManager
# Event system - using mock implementation for now
EVENT_SYSTEM_AVAILABLE = False

class Event:
    def __init__(self, type: str, data: dict):
        self.type = type
        self.data = data

class EventSystem:
    def __init__(self):
        self.handlers = {}

    def subscribe(self, event_type: str, handler):
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    async def publish(self, event: Event):
        # Mock implementation - just call handlers synchronously
        if event.type in self.handlers:
            for handler in self.handlers[event.type]:
                try:
                    await handler(event)
                except Exception as e:
                    print(f"Event handler error: {e}")
from src.cache import Cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestingService:
    """
    Async backtesting service for historical trading strategy evaluation.

    Provides comprehensive backtesting capabilities with support for:
    - Multiple strategy types (traditional, ML-enhanced, LSTM, Transformer)
    - Performance metrics calculation
    - Risk analysis
    - Multi-symbol and multi-timeframe testing
    """

    def __init__(self, event_system: Optional[EventSystem] = None, cache: Optional[Cache] = None):
        """
        Initialize the backtesting service.

        Args:
            event_system: Optional event system for async communication
            cache: Optional cache system for data persistence
        """
        self.event_system = event_system or EventSystem()
        self.cache = cache or Cache()
        self.active_backtests = {}  # Track running backtests
        self.backtest_results = {}  # Store completed results

        # Register event handlers
        self._register_event_handlers()

        logger.info("BacktestingService initialized")

    def _register_event_handlers(self):
        """Register event handlers for service communication."""
        self.event_system.subscribe("backtest.request", self._handle_backtest_request)
        self.event_system.subscribe("backtest.multi_symbol.request", self._handle_multi_symbol_request)
        self.event_system.subscribe("backtest.multi_model.request", self._handle_multi_model_request)
        self.event_system.subscribe("backtest.walk_forward.request", self._handle_walk_forward_request)

    async def _handle_backtest_request(self, event: Event):
        """Handle single backtest requests."""
        try:
            data = event.data
            backtest_id = data.get('backtest_id', f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            logger.info(f"Starting backtest {backtest_id}")

            # Extract parameters
            symbol = data.get('symbol', SYMBOL)
            timeframe = data.get('timeframe', TIMEFRAME)
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            strategy = data.get('strategy', 'traditional')
            initial_capital = data.get('initial_capital', CAPITAL)
            config = data.get('config', {})

            # Run backtest
            result = await self.run_backtest_async(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy,
                initial_capital=initial_capital,
                config=config,
                backtest_id=backtest_id
            )

            # Publish completion event
            await self.event_system.publish(Event(
                type="backtest.completed",
                data={
                    "backtest_id": backtest_id,
                    "result": result
                }
            ))

        except Exception as e:
            logger.error(f"Backtest request failed: {e}")
            await self.event_system.publish(Event(
                type="backtest.failed",
                data={
                    "backtest_id": event.data.get('backtest_id'),
                    "error": str(e)
                }
            ))

    async def _handle_multi_symbol_request(self, event: Event):
        """Handle multi-symbol backtest requests."""
        try:
            data = event.data
            backtest_id = data.get('backtest_id', f"multi_symbol_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            symbols = data.get('symbols', [SYMBOL])
            timeframe = data.get('timeframe', TIMEFRAME)
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            initial_capital = data.get('initial_capital', CAPITAL)
            max_workers = data.get('max_workers')

            result = await self.run_multi_symbol_backtest_async(
                symbols=symbols,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                max_workers=max_workers,
                backtest_id=backtest_id
            )

            await self.event_system.publish(Event(
                type="backtest.multi_symbol.completed",
                data={
                    "backtest_id": backtest_id,
                    "result": result
                }
            ))

        except Exception as e:
            logger.error(f"Multi-symbol backtest failed: {e}")
            await self.event_system.publish(Event(
                type="backtest.multi_symbol.failed",
                data={
                    "backtest_id": event.data.get('backtest_id'),
                    "error": str(e)
                }
            ))

    async def _handle_multi_model_request(self, event: Event):
        """Handle multi-model backtest requests."""
        try:
            data = event.data
            backtest_id = data.get('backtest_id', f"multi_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            symbol = data.get('symbol', SYMBOL)
            timeframe = data.get('timeframe', TIMEFRAME)
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            initial_capital = data.get('initial_capital', CAPITAL)
            models_to_test = data.get('models_to_test')
            max_workers = data.get('max_workers')

            result = await self.run_multi_model_backtest_async(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                models_to_test=models_to_test,
                max_workers=max_workers,
                backtest_id=backtest_id
            )

            await self.event_system.publish(Event(
                type="backtest.multi_model.completed",
                data={
                    "backtest_id": backtest_id,
                    "result": result
                }
            ))

        except Exception as e:
            logger.error(f"Multi-model backtest failed: {e}")
            await self.event_system.publish(Event(
                type="backtest.multi_model.failed",
                data={
                    "backtest_id": event.data.get('backtest_id'),
                    "error": str(e)
                }
            ))

    async def _handle_walk_forward_request(self, event: Event):
        """Handle walk-forward optimization requests."""
        try:
            data = event.data
            backtest_id = data.get('backtest_id', f"walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            parameter_grid = data.get('parameter_grid', {})
            evaluation_metric = data.get('evaluation_metric', 'sharpe_ratio')
            symbol = data.get('symbol', SYMBOL)
            timeframe = data.get('timeframe', TIMEFRAME)

            result = await self.run_walk_forward_backtest_async(
                parameter_grid=parameter_grid,
                evaluation_metric=evaluation_metric,
                symbol=symbol,
                timeframe=timeframe,
                backtest_id=backtest_id
            )

            await self.event_system.publish(Event(
                type="backtest.walk_forward.completed",
                data={
                    "backtest_id": backtest_id,
                    "result": result
                }
            ))

        except Exception as e:
            logger.error(f"Walk-forward backtest failed: {e}")
            await self.event_system.publish(Event(
                type="backtest.walk_forward.failed",
                data={
                    "backtest_id": event.data.get('backtest_id'),
                    "error": str(e)
                }
            ))

    async def run_backtest_async(self, symbol: str = SYMBOL, timeframe: str = TIMEFRAME,
                                start_date: Optional[str] = None, end_date: Optional[str] = None,
                                strategy: str = 'traditional', initial_capital: float = CAPITAL,
                                config: Optional[Dict] = None, backtest_id: Optional[str] = None) -> Dict:
        """
        Run an asynchronous backtest with the specified parameters.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1d', '1h')
            start_date: Start date string
            end_date: End date string
            strategy: Strategy type ('traditional', 'ml_enhanced', 'lstm_enhanced', 'transformer_enhanced')
            initial_capital: Initial capital amount
            config: Optional configuration overrides
            backtest_id: Optional unique identifier for the backtest

        Returns:
            Dictionary with backtest results
        """
        if backtest_id is None:
            backtest_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.active_backtests[backtest_id] = {
            "status": "running",
            "start_time": datetime.now(),
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy
        }

        try:
            logger.info(f"Starting async backtest {backtest_id} for {symbol} {timeframe} with {strategy} strategy")

            # Create backtester instance
            backtester = AsyncBacktester(
                symbol=symbol,
                exchange=EXCHANGE,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                config=config
            )

            # Run backtest
            result = await backtester.run_backtest_async(strategy=strategy)

            # Store results
            self.backtest_results[backtest_id] = result
            self.active_backtests[backtest_id]["status"] = "completed"
            self.active_backtests[backtest_id]["end_time"] = datetime.now()

            # Save results to file
            await self._save_backtest_results_async(backtest_id, result, symbol, timeframe, start_date, end_date)

            logger.info(f"Completed backtest {backtest_id}: {result.get('trades_count', 0)} trades")

            return result

        except Exception as e:
            error_msg = f"Backtest {backtest_id} failed: {str(e)}"
            logger.error(error_msg)

            self.active_backtests[backtest_id]["status"] = "failed"
            self.active_backtests[backtest_id]["error"] = str(e)
            self.active_backtests[backtest_id]["end_time"] = datetime.now()

            return {"success": False, "error": error_msg}

    async def run_multi_symbol_backtest_async(self, symbols: List[str], timeframe: str = TIMEFRAME,
                                             start_date: Optional[str] = None, end_date: Optional[str] = None,
                                             initial_capital: float = CAPITAL, max_workers: Optional[int] = None,
                                             backtest_id: Optional[str] = None) -> Dict:
        """
        Run parallel backtests for multiple symbols asynchronously.

        Args:
            symbols: List of trading symbols
            timeframe: Timeframe for all backtests
            start_date: Start date for backtests
            end_date: End date for backtests
            initial_capital: Initial capital per backtest
            max_workers: Maximum parallel workers
            backtest_id: Optional unique identifier

        Returns:
            Dictionary with aggregated results
        """
        if backtest_id is None:
            backtest_id = f"multi_symbol_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting multi-symbol backtest {backtest_id} for {len(symbols)} symbols")

        # Use asyncio.gather for parallel execution
        tasks = []
        for symbol in symbols:
            task = self.run_backtest_async(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                backtest_id=f"{backtest_id}_{symbol.replace('/', '_')}"
            )
            tasks.append(task)

        # Execute in parallel with limited concurrency
        semaphore = asyncio.Semaphore(max_workers or min(len(symbols), 4))

        async def limited_task(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(*[limited_task(task) for task in tasks], return_exceptions=True)

        # Process results
        successful_results = {}
        failed_results = {}

        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, Exception):
                failed_results[symbol] = {"error": str(result)}
            elif isinstance(result, dict) and result.get("success", False):
                successful_results[symbol] = result
            else:
                failed_results[symbol] = result

        # Aggregate metrics
        aggregated_metrics = await self._aggregate_multi_symbol_metrics(successful_results)

        return {
            "backtest_id": backtest_id,
            "symbols_tested": symbols,
            "successful_backtests": list(successful_results.keys()),
            "failed_backtests": list(failed_results.keys()),
            "aggregated_metrics": aggregated_metrics,
            "individual_results": {**successful_results, **failed_results}
        }

    async def run_multi_model_backtest_async(self, symbol: str = SYMBOL, timeframe: str = TIMEFRAME,
                                            start_date: Optional[str] = None, end_date: Optional[str] = None,
                                            initial_capital: float = CAPITAL, models_to_test: Optional[List[str]] = None,
                                            max_workers: Optional[int] = None, backtest_id: Optional[str] = None) -> Dict:
        """
        Run parallel backtests comparing different ML models asynchronously.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for backtests
            start_date: Start date for backtests
            end_date: End date for backtests
            initial_capital: Initial capital per backtest
            models_to_test: List of model types to test
            max_workers: Maximum parallel workers
            backtest_id: Optional unique identifier

        Returns:
            Dictionary with model comparison results
        """
        if backtest_id is None:
            backtest_id = f"multi_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        available_models = ['traditional_ml', 'lstm', 'transformer', 'ensemble']
        if models_to_test is None:
            models_to_test = available_models

        models_to_test = [m for m in models_to_test if m in available_models]

        logger.info(f"Starting multi-model backtest {backtest_id} for {len(models_to_test)} models")

        # Run backtests for each model in parallel
        tasks = []
        for model_type in models_to_test:
            strategy = f"{model_type}_enhanced" if model_type != 'ensemble' else 'ml_enhanced'
            task = self.run_backtest_async(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy,
                initial_capital=initial_capital,
                backtest_id=f"{backtest_id}_{model_type}"
            )
            tasks.append(task)

        # Execute in parallel
        semaphore = asyncio.Semaphore(max_workers or min(len(models_to_test), 4))

        async def limited_task(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(*[limited_task(task) for task in tasks], return_exceptions=True)

        # Process results and create comparison
        model_results = {}
        for i, result in enumerate(results):
            model_type = models_to_test[i]
            if isinstance(result, Exception):
                model_results[model_type] = {"error": str(result)}
            else:
                model_results[model_type] = result

        comparison = await self._compare_model_results_async(model_results)

        return {
            "backtest_id": backtest_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "models_tested": models_to_test,
            "comparison": comparison,
            "individual_results": model_results
        }

    async def run_walk_forward_backtest_async(self, parameter_grid: Dict, evaluation_metric: str = 'sharpe_ratio',
                                             symbol: str = SYMBOL, timeframe: str = TIMEFRAME,
                                             backtest_id: Optional[str] = None) -> Dict:
        """
        Run walk-forward optimization asynchronously.

        Args:
            parameter_grid: Dictionary of parameters to optimize
            evaluation_metric: Metric to use for evaluation
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            backtest_id: Optional unique identifier

        Returns:
            Dictionary with optimization results
        """
        if backtest_id is None:
            backtest_id = f"walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting walk-forward optimization {backtest_id}")

        try:
            # Import walk-forward optimizer
            from src.walk_forward_optimizer import WalkForwardOptimizer

            # Create optimizer instance
            optimizer = WalkForwardOptimizer(symbol=symbol, timeframe=timeframe)

            # Run optimization (this might be CPU intensive, so we run it in a thread pool)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: optimizer.optimize_strategy_parameters(
                    parameter_grid=parameter_grid,
                    evaluation_metric=evaluation_metric
                )
            )

            if results.get('success'):
                # Run final backtest with best parameters
                best_params = results.get('best_parameters', {})
                final_backtest = await self.run_backtest_async(
                    symbol=symbol,
                    timeframe=timeframe,
                    backtest_id=f"{backtest_id}_final"
                )

                return {
                    "optimization_results": results,
                    "final_backtest": final_backtest
                }
            else:
                return {"error": results.get('error', 'Optimization failed')}

        except Exception as e:
            error_msg = f"Walk-forward optimization failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def _aggregate_multi_symbol_metrics(self, results: Dict) -> Dict:
        """Aggregate metrics from multiple symbol backtests."""
        if not results:
            return {}

        # Collect all metrics
        all_metrics = [result.get('metrics', {}) for result in results.values() if 'metrics' in result]

        if not all_metrics:
            return {}

        # Calculate averages
        aggregated = {}
        metrics_keys = ['total_pnl', 'win_rate', 'sharpe_ratio', 'max_drawdown', 'profit_factor', 'cagr']

        for key in metrics_keys:
            values = [m.get(key, 0) for m in all_metrics if key in m]
            if values:
                aggregated[f'avg_{key}'] = sum(values) / len(values)
                aggregated[f'min_{key}'] = min(values)
                aggregated[f'max_{key}'] = max(values)

        # Count total trades
        total_trades = sum(result.get('trades_count', 0) for result in results.values())
        aggregated['total_trades_all_symbols'] = total_trades

        return aggregated

    async def _compare_model_results_async(self, results: Dict) -> Dict:
        """Compare results from different models asynchronously."""
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if 'error' not in v and v.get('success')}

        if not successful_results:
            return {"error": "No successful model backtests"}

        # Create comparison DataFrame
        comparison_data = []
        for model_type, result in successful_results.items():
            metrics = result.get('metrics', {})
            row = {
                'Model': model_type,
                'Total P&L': metrics.get('total_pnl', 0),
                'Win Rate (%)': metrics.get('win_rate', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Profit Factor': metrics.get('profit_factor', float('inf')),
                'Max Drawdown (%)': metrics.get('max_drawdown', 0),
                'CAGR (%)': metrics.get('cagr', 0)
            }
            comparison_data.append(row)

        # Find best model
        best_model = max(successful_results.keys(),
                        key=lambda x: successful_results[x].get('metrics', {}).get('sharpe_ratio', 0))

        return {
            "comparison_table": comparison_data,
            "best_model": best_model,
            "best_metrics": successful_results[best_model].get('metrics', {}),
            "models_compared": list(successful_results.keys())
        }

    async def _save_backtest_results_async(self, backtest_id: str, result: Dict, symbol: str,
                                          timeframe: str, start_date: Optional[str], end_date: Optional[str]):
        """Save backtest results to file asynchronously."""
        try:
            results_data = {
                'backtest_info': {
                    'backtest_id': backtest_id,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'start_date': start_date,
                    'end_date': end_date,
                    'timestamp': datetime.now().isoformat()
                },
                'metrics': result.get('metrics', {}),
                'trades': result.get('trades', []),
                'trades_count': result.get('trades_count', 0)
            }

            filename = f"output/backtest_{backtest_id}_{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # Write to file (using executor for I/O)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: json.dump(results_data, open(filename, 'w'), indent=4, default=str)
            )

            logger.info(f"Backtest results saved to {filename}")

        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")

    async def get_backtest_status(self, backtest_id: str) -> Dict:
        """Get the status of a specific backtest."""
        if backtest_id in self.active_backtests:
            return self.active_backtests[backtest_id]
        elif backtest_id in self.backtest_results:
            return {
                "status": "completed",
                "result": self.backtest_results[backtest_id]
            }
        else:
            return {"status": "not_found"}

    async def list_active_backtests(self) -> List[Dict]:
        """List all currently active backtests."""
        return list(self.active_backtests.values())

    async def cleanup_completed_backtests(self, max_age_hours: int = 24):
        """Clean up old completed backtests from memory."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        to_remove = []
        for backtest_id, info in self.active_backtests.items():
            if info.get("status") in ["completed", "failed"] and info.get("end_time", datetime.min) < cutoff_time:
                to_remove.append(backtest_id)

        for backtest_id in to_remove:
            del self.active_backtests[backtest_id]
            if backtest_id in self.backtest_results:
                del self.backtest_results[backtest_id]

        logger.info(f"Cleaned up {len(to_remove)} old backtests")


class AsyncBacktester:
    """
    Async version of the backtester for running backtests asynchronously.
    """

    def __init__(self, symbol: str, exchange: str, timeframe: str,
                 start_date: Optional[str] = None, end_date: Optional[str] = None,
                 initial_capital: float = CAPITAL, config: Optional[Dict] = None):
        """
        Initialize async backtester.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe string
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            config: Configuration overrides
        """
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.config = config or {}
        self.commission = 0.001  # 0.1% commission

        # Initialize data and results
        self.data = None
        self.trades = []
        self.portfolio_values = []

    async def run_backtest_async(self, strategy: str = 'traditional') -> Dict:
        """
        Run the backtest asynchronously.

        Args:
            strategy: Strategy type to use

        Returns:
            Dictionary with backtest results
        """
        try:
            # Fetch data asynchronously
            self.data = await self._fetch_data_async()

            if self.data.empty:
                return {"success": False, "error": "No data available for backtest period"}

            # Prepare signals based on strategy
            if strategy == 'traditional':
                self.data = await self._prepare_traditional_signals_async(self.data)
            elif strategy == 'ml_enhanced':
                self.data = await self._prepare_ml_enhanced_signals_async(self.data)
            elif strategy == 'lstm_enhanced':
                self.data = await self._prepare_lstm_enhanced_signals_async(self.data)
            elif strategy == 'transformer_enhanced':
                self.data = await self._prepare_transformer_enhanced_signals_async(self.data)
            else:
                return {"success": False, "error": f"Unknown strategy: {strategy}"}

            # Simulate trades
            self.trades = await self._simulate_trades_async(self.data)

            # Calculate performance metrics
            metrics = await self._calculate_metrics_async()

            # Add strategy info
            metrics['strategy'] = strategy
            metrics['total_return_pct'] = metrics.get('return_pct', 0)

            return {
                "success": True,
                "metrics": metrics,
                "trades": self.trades,
                "trades_count": len(self.trades)
            }

        except Exception as e:
            error_msg = f"Backtest failed: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    async def _fetch_data_async(self) -> pd.DataFrame:
        """Fetch historical data asynchronously."""
        # Calculate limit based on timeframe and date range
        days = (self.end_date - self.start_date).days if self.start_date and self.end_date else 365
        if self.timeframe == '1m':
            limit = min(days * 24 * 60, 50000)
        elif self.timeframe == '5m':
            limit = min(days * 24 * 12, 50000)
        elif self.timeframe == '15m':
            limit = min(days * 24 * 4, 50000)
        elif self.timeframe == '1h':
            limit = min(days * 24, 50000)
        elif self.timeframe == '4h':
            limit = min(days * 6, 50000)
        elif self.timeframe == '1d':
            limit = min(days, 50000)
        else:
            limit = 10000

        limit = int(limit)

        # Run data fetching in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: fetch_historical_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=limit,
                start_date=self.start_date
            )
        )

        return data

    async def _prepare_traditional_signals_async(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with traditional indicators and signals asynchronously."""
        # Run calculations in thread pool
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: calculate_indicators(data, config=self.config)
        )
        data = await loop.run_in_executor(
            None,
            lambda: generate_signals(data, config=self.config)
        )
        data = await loop.run_in_executor(
            None,
            lambda: self._calculate_risk_management(data, self.config)
        )
        return data

    async def _prepare_ml_enhanced_signals_async(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with ML-enhanced signals asynchronously."""
        # First prepare traditional signals
        data = await self._prepare_traditional_signals_async(data)

        # Add ML enhancement if available
        if ML_ENABLED:
            try:
                from src.ml_predictor import get_ml_predictor, is_ml_available
                if is_ml_available():
                    predictor = get_ml_predictor(symbol=self.symbol, timeframe=self.timeframe)
                    if predictor and predictor.is_trained:
                        logger.info("🤖 Applying ML signal enhancement...")
                        # Run ML prediction in thread pool
                        loop = asyncio.get_event_loop()
                        data = await loop.run_in_executor(
                            None,
                            lambda: self._apply_ml_enhancement(data, predictor, 'ml_enhanced')
                        )
                    else:
                        logger.info("⚠️  ML predictor not available or not trained, using traditional signals")
            except Exception as e:
                logger.warning(f"ML enhancement failed: {e}, using traditional signals")

        return data

    async def _prepare_lstm_enhanced_signals_async(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with LSTM-enhanced signals asynchronously."""
        # First prepare traditional signals
        data = await self._prepare_traditional_signals_async(data)

        # Add LSTM enhancement if available
        if ML_ENABLED:
            try:
                from src.ml_predictor import get_lstm_predictor, is_lstm_available
                if is_lstm_available():
                    predictor = get_lstm_predictor(symbol=self.symbol, timeframe=self.timeframe)
                    if predictor and predictor.is_trained:
                        logger.info("🧠 Applying LSTM signal enhancement...")
                        loop = asyncio.get_event_loop()
                        data = await loop.run_in_executor(
                            None,
                            lambda: self._apply_ml_enhancement(data, predictor, 'lstm_enhanced')
                        )
                    else:
                        logger.info("⚠️  LSTM predictor not available or not trained, using traditional signals")
            except Exception as e:
                logger.warning(f"LSTM enhancement failed: {e}, using traditional signals")

        return data

    async def _prepare_transformer_enhanced_signals_async(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with Transformer-enhanced signals asynchronously."""
        # First prepare traditional signals
        data = await self._prepare_traditional_signals_async(data)

        # Add Transformer enhancement if available
        if ML_ENABLED:
            try:
                from src.ml_predictor import get_transformer_predictor, is_transformer_available
                if is_transformer_available():
                    predictor = get_transformer_predictor(symbol=self.symbol, timeframe=self.timeframe)
                    if predictor and predictor.is_trained:
                        logger.info("🔄 Applying Transformer signal enhancement...")
                        loop = asyncio.get_event_loop()
                        data = await loop.run_in_executor(
                            None,
                            lambda: self._apply_ml_enhancement(data, predictor, 'transformer_enhanced')
                        )
                    else:
                        logger.info("⚠️  Transformer predictor not available or not trained, using traditional signals")
            except Exception as e:
                logger.warning(f"Transformer enhancement failed: {e}, using traditional signals")

        return data

    def _apply_ml_enhancement(self, data: pd.DataFrame, predictor, strategy_name: str) -> pd.DataFrame:
        """Apply ML enhancement to signals (synchronous helper)."""
        # This is the same logic as in the original backtester
        # Add ML signal columns
        data = data.copy()
        data['ML_Signal'] = 'HOLD'
        data['ML_Confidence'] = 0.0
        data['ML_Reason'] = ''
        data['Enhanced_Signal'] = 'HOLD'
        data['Signal_Source'] = 'TRADITIONAL'

        # Get traditional signals
        buy_signals = data.get('Buy_Signal', 0).values
        sell_signals = data.get('Sell_Signal', 0).values

        # Process predictions
        for idx in range(len(data)):
            try:
                row_df = pd.DataFrame([data.iloc[idx]])
                prediction = predictor.predict_signal(row_df, threshold=ML_CONFIDENCE_THRESHOLD)

                if prediction['confidence'] >= ML_CONFIDENCE_THRESHOLD:
                    data.iloc[idx, data.columns.get_loc('ML_Signal')] = prediction['signal']
                    data.iloc[idx, data.columns.get_loc('ML_Confidence')] = prediction['confidence']
                    data.iloc[idx, data.columns.get_loc('ML_Reason')] = prediction.get('reason', '')
                    data.iloc[idx, data.columns.get_loc('Enhanced_Signal')] = prediction['signal']
                    data.iloc[idx, data.columns.get_loc('Signal_Source')] = strategy_name.upper()
                else:
                    traditional_signal = 'BUY' if buy_signals[idx] == 1 else 'SELL' if sell_signals[idx] == 1 else 'HOLD'
                    data.iloc[idx, data.columns.get_loc('Enhanced_Signal')] = traditional_signal
                    data.iloc[idx, data.columns.get_loc('Signal_Source')] = 'TRADITIONAL'

            except Exception as e:
                logger.warning(f"Prediction failed for row {idx}: {e}")
                continue

        # Override original signals
        enhanced_mask = data['Signal_Source'] != 'TRADITIONAL'
        data.loc[enhanced_mask, 'Buy_Signal'] = np.where(
            data.loc[enhanced_mask, 'Enhanced_Signal'] == 'BUY', 1, 0
        )
        data.loc[enhanced_mask, 'Sell_Signal'] = np.where(
            data.loc[enhanced_mask, 'Enhanced_Signal'] == 'SELL', 1, 0
        )

        return data

    async def _simulate_trades_async(self, data: pd.DataFrame) -> List[Dict]:
        """Simulate trades asynchronously."""
        # Run trade simulation in thread pool
        loop = asyncio.get_event_loop()
        trades = await loop.run_in_executor(
            None,
            lambda: self._simulate_trades_sync(data)
        )
        return trades

    def _simulate_trades_sync(self, data: pd.DataFrame) -> List[Dict]:
        """Synchronous trade simulation (same logic as original)."""
        if data.empty:
            return []

        # Drop rows with NaN ATR values
        if 'ATR' in data.columns:
            data = data.dropna(subset=['ATR'])

        required_cols = ['close', 'Buy_Signal', 'Sell_Signal']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data missing required columns: {required_cols}")

        # Create position signals
        buy_signals = (data['Buy_Signal'] == 1)
        sell_signals = (data['Sell_Signal'] == 1)

        position_changes = pd.Series(0, index=data.index)
        position_changes[buy_signals] = 1
        position_changes[sell_signals] = 0

        positions = position_changes.cumsum()

        trades = []
        current_position = 0
        entry_idx = None
        entry_price = None

        position_array = positions.values
        timestamps = data.index
        close_prices = data['close'].values

        slippage_pct = 0.001
        position_sizes = data.get('Position_Size_Absolute', np.full(len(data), 1000))
        stop_loss_buy = data.get('Stop_Loss_Buy', data['close'] * 0.98).values
        stop_loss_sell = data.get('Stop_Loss_Sell', data['close'] * 1.02).values
        take_profit_buy = data.get('Take_Profit_Buy', data['close'] * 1.02).values
        take_profit_sell = data.get('Take_Profit_Sell', data['close'] * 0.98).values

        for idx in range(len(data)):
            new_position = position_array[idx]

            # Position entry
            if current_position == 0 and new_position != 0:
                entry_idx = idx
                if new_position == 1:
                    entry_price = close_prices[idx] * (1 + slippage_pct)
                else:
                    entry_price = close_prices[idx] * (1 - slippage_pct)

                current_position = new_position
                position_size = position_sizes[idx]

                entry_commission = entry_price * position_size * self.commission

                trade = {
                    'type': 'buy' if current_position == 1 else 'sell',
                    'entry_time': timestamps[idx],
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'stop_loss': stop_loss_buy[idx] if current_position == 1 else stop_loss_sell[idx],
                    'take_profit': take_profit_buy[idx] if current_position == 1 else take_profit_sell[idx],
                    'exit_time': None,
                    'exit_price': None,
                    'pnl': 0,
                    'entry_commission': entry_commission,
                    'exit_commission': 0,
                    'status': 'open'
                }
                trades.append(trade)

            # Position exit
            elif current_position != 0 and new_position != current_position:
                if entry_idx is not None:
                    if current_position == 1:
                        exit_price = close_prices[idx] * (1 - slippage_pct)
                    else:
                        exit_price = close_prices[idx] * (1 + slippage_pct)

                    exit_time = timestamps[idx]
                    exit_commission = exit_price * trades[-1]['position_size'] * self.commission

                    if current_position == 1:
                        gross_pnl = (exit_price - entry_price) * trades[-1]['position_size'] / entry_price
                    else:
                        gross_pnl = (entry_price - exit_price) * trades[-1]['position_size'] / entry_price

                    net_pnl = gross_pnl - trades[-1]['entry_commission'] - exit_commission

                    trades[-1].update({
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'exit_commission': exit_commission,
                        'status': 'closed'
                    })

                    self.current_capital += net_pnl

                current_position = 0
                entry_idx = None

        # Close remaining positions
        if current_position != 0 and trades and trades[-1]['status'] == 'open':
            final_price = data.iloc[-1]['close']
            final_time = data.index[-1]

            if current_position == 1:
                exit_price = final_price * (1 - slippage_pct)
            else:
                exit_price = final_price * (1 + slippage_pct)

            exit_commission = exit_price * trades[-1]['position_size'] * self.commission

            if current_position == 1:
                gross_pnl = (exit_price - entry_price) * trades[-1]['position_size'] / entry_price
            else:
                gross_pnl = (entry_price - exit_price) * trades[-1]['position_size'] / entry_price

            net_pnl = gross_pnl - trades[-1]['entry_commission'] - exit_commission

            trades[-1].update({
                'exit_time': final_time,
                'exit_price': exit_price,
                'pnl': net_pnl,
                'exit_commission': exit_commission,
                'status': 'closed'
            })

            self.current_capital += net_pnl

        return trades

    async def _calculate_metrics_async(self) -> Dict:
        """Calculate performance metrics asynchronously."""
        # Run metrics calculation in thread pool
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(
            None,
            lambda: self._calculate_metrics_sync()
        )
        return metrics

    def _calculate_metrics_sync(self) -> Dict:
        """Synchronous metrics calculation (same logic as original)."""
        if not self.trades:
            return {"error": "No trades executed during backtest"}

        closed_trades = [t for t in self.trades if t['status'] == 'closed']

        if not closed_trades:
            return {"error": "No closed trades to analyze"}

        trades_df = pd.DataFrame(closed_trades)

        total_trades = len(trades_df)
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] <= 0).sum()

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0

        total_commissions = trades_df['entry_commission'].sum() + trades_df['exit_commission'].sum()

        returns = trades_df['pnl'] / self.initial_capital
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        capital_series = self.initial_capital + trades_df['pnl'].cumsum()
        peak = capital_series.expanding().max()
        drawdown = (peak - capital_series) / peak * 100
        max_drawdown = drawdown.max()

        days = (self.end_date - self.start_date).days if self.start_date and self.end_date else 365
        if days > 0:
            cagr = ((self.current_capital / self.initial_capital) ** (365 / days) - 1) * 100
        else:
            cagr = 0

        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        return {
            'total_trades': total_trades,
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'total_commissions': round(total_commissions, 2),
            'net_pnl_after_costs': round(total_pnl - total_commissions, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'cagr': round(cagr, 2),
            'final_capital': round(self.current_capital, 2),
            'return_pct': round((self.current_capital / self.initial_capital - 1) * 100, 2)
        }

    def _calculate_risk_management(self, data: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """Calculate risk management parameters for backtesting."""
        # Simple risk management calculation
        # Add position size calculations based on ATR and risk parameters
        if 'ATR' in data.columns:
            # Risk per trade (1% of capital)
            risk_per_trade = 0.01

            # Position size = (Capital * Risk per trade) / (ATR * multiplier)
            atr_multiplier = 1.5
            data['Position_Size_Absolute'] = 10000 * risk_per_trade / (data['ATR'] * atr_multiplier)

            # Stop loss levels
            data['Stop_Loss_Buy'] = data['close'] - (data['ATR'] * 1.5)
            data['Stop_Loss_Sell'] = data['close'] + (data['ATR'] * 1.5)

            # Take profit levels
            data['Take_Profit_Buy'] = data['close'] + (data['ATR'] * 2.0)
            data['Take_Profit_Sell'] = data['close'] - (data['ATR'] * 2.0)

        return data