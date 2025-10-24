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
from config.service_settings import ENABLE_DATA_SERVICE
from services.data_service.client import DataService
# Use central service client instead of direct imports
from services.trading_service.central_service_client import get_central_client

logger = logging.getLogger(__name__)


class BacktestingService:
    """
    Comprehensive backtesting service with ML-enhanced strategies.

    Features:
    - Multi-strategy backtesting (traditional + ML)
    - Walk-forward optimization
    - Performance analytics
    - Risk management integration
    - Event-driven communication
    """

    def __init__(self):
        self.client = None
        self.active_backtests = {}
        self.backtest_results = {}

    async def initialize(self):
        """Initialize service and establish connections"""
        self.client = await get_central_client()
        logger.info("BacktestingService initialized with central client")

    async def get_historical_data(self, symbol: str, start_date: str,
                                end_date: str, timeframe: str = "1h") -> pd.DataFrame:
        """Get historical data using central service client"""
        try:
            response = await self.client.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            return pd.DataFrame(response.get("data", []))
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            raise

    async def calculate_indicators(self, data: pd.DataFrame,
                                 indicators: List[str]) -> pd.DataFrame:
        """Calculate technical indicators using core service"""
        try:
            data_dict = data.to_dict('records')
            response = await self.client.calculate_indicators(
                data={"ohlcv": data_dict},
                indicators=indicators
            )
            # Merge indicators back into dataframe
            indicators_df = pd.DataFrame(response.get("indicators", []))
            return pd.concat([data, indicators_df], axis=1)
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            raise

    async def generate_signals(self, data: pd.DataFrame,
                             strategy_config: Dict[str, Any]) -> pd.DataFrame:
        """Generate trading signals using ML service"""
        try:
            signals = []
            for _, row in data.iterrows():
                market_data = {
                    "close": row.get("close", 0),
                    "high": row.get("high", 0),
                    "low": row.get("low", 0),
                    "volume": row.get("volume", 0),
                    "indicators": {k: v for k, v in row.items() if k not in ["timestamp", "open", "high", "low", "close", "volume"]}
                }

                response = await self.client.get_trading_signal(
                    market_data=market_data,
                    symbol=strategy_config.get("symbol", "BTC/USDT")
                )

                signals.append({
                    "timestamp": row.get("timestamp"),
                    "signal": response.get("action", "HOLD"),
                    "confidence": response.get("confidence", 0.0)
                })

            signals_df = pd.DataFrame(signals)
            return pd.merge(data, signals_df, on="timestamp", how="left")
        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            raise

    async def run_backtest(self, strategy_config: Dict[str, Any],
                         start_date: str, end_date: str,
                         backtest_id: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive backtesting simulation"""
        if not backtest_id:
            backtest_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting backtest {backtest_id}")

        try:
            # Get historical data
            symbol = strategy_config.get("symbol", SYMBOL)
            timeframe = strategy_config.get("timeframe", TIMEFRAME)

            data = await self.get_historical_data(symbol, start_date, end_date, timeframe)

            # Calculate indicators
            indicators = strategy_config.get("indicators", ["SMA_20", "SMA_50", "RSI", "MACD"])
            data = await self.calculate_indicators(data, indicators)

            # Generate signals
            data = await self.generate_signals(data, strategy_config)

            # Run simulation
            results = await self._simulate_trading(data, strategy_config)

            # Calculate performance metrics
            performance = self._calculate_performance_metrics(results, strategy_config)

            # Store results
            self.backtest_results[backtest_id] = {
                "config": strategy_config,
                "results": results,
                "performance": performance,
                "data_points": len(data),
                "start_date": start_date,
                "end_date": end_date,
                "timestamp": datetime.now().isoformat()
            }

            # Publish event
            await self.client.publish_event("backtest_completed", {
                "backtest_id": backtest_id,
                "performance": performance
            })

            logger.info(f"Backtest {backtest_id} completed successfully")
            return self.backtest_results[backtest_id]

        except Exception as e:
            logger.error(f"Backtest {backtest_id} failed: {e}")
            await self.client.publish_event("backtest_failed", {
                "backtest_id": backtest_id,
                "error": str(e)
            })
            raise

    async def _simulate_trading(self, data: pd.DataFrame,
                               strategy_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate trading based on signals"""
        trades = []
        position = None
        capital = strategy_config.get("capital", CAPITAL)

        for _, row in data.iterrows():
            signal = row.get("signal", "HOLD")
            confidence = row.get("confidence", 0.0)
            price = row.get("close", 0)
            timestamp = row.get("timestamp")

            if signal == "BUY" and position is None and confidence > ML_CONFIDENCE_THRESHOLD:
                # Open long position
                position_size = capital * strategy_config.get("position_size", 0.1)
                quantity = position_size / price

                position = {
                    "type": "long",
                    "entry_price": price,
                    "quantity": quantity,
                    "entry_time": timestamp,
                    "confidence": confidence
                }

            elif signal == "SELL" and position is not None:
                # Close position
                exit_price = price
                pnl = (exit_price - position["entry_price"]) * position["quantity"]

                trade = {
                    "entry_time": position["entry_time"],
                    "exit_time": timestamp,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "quantity": position["quantity"],
                    "pnl": pnl,
                    "type": position["type"],
                    "confidence": position["confidence"]
                }

                trades.append(trade)
                position = None

        return trades

    def _calculate_performance_metrics(self, trades: List[Dict[str, Any]],
                                    strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {"total_trades": 0, "total_pnl": 0, "win_rate": 0}

        df_trades = pd.DataFrame(trades)
        total_trades = len(trades)
        winning_trades = len(df_trades[df_trades["pnl"] > 0])
        losing_trades = len(df_trades[df_trades["pnl"] < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = df_trades["pnl"].sum()
        avg_win = df_trades[df_trades["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades["pnl"] < 0]["pnl"].mean() if losing_trades > 0 else 0

        # Sharpe ratio (simplified)
        returns = df_trades["pnl"].pct_change().fillna(0)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf'),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }

    async def get_backtest_results(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """Get backtest results by ID"""
        return self.backtest_results.get(backtest_id)

    async def list_backtests(self) -> List[str]:
        """List all completed backtest IDs"""
        return list(self.backtest_results.keys())

    async def optimize_strategy(self, strategy_config: Dict[str, Any],
                              optimization_config: Dict[str, Any],
                              start_date: str, end_date: str) -> Dict[str, Any]:
        """Run strategy optimization using discovery service"""
        try:
            opt_config = {
                "strategy_config": strategy_config,
                "optimization_config": optimization_config,
                "backtest_period": {
                    "start_date": start_date,
                    "end_date": end_date
                }
            }

            response = await self.client.optimize_parameters(opt_config)
            return response
        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")
            raise

    async def compare_strategies(self, strategy_configs: List[Dict[str, Any]],
                               start_date: str, end_date: str) -> Dict[str, Any]:
        """Compare multiple strategies"""
        results = {}

        for i, config in enumerate(strategy_configs):
            strategy_name = config.get("name", f"Strategy_{i+1}")
            try:
                result = await self.run_backtest(config, start_date, end_date)
                results[strategy_name] = result["performance"]
            except Exception as e:
                logger.error(f"Failed to run strategy {strategy_name}: {e}")
                results[strategy_name] = {"error": str(e)}

        return results

    async def cleanup(self):
        """Cleanup service resources"""
        if self.client:
            await self.client.close()
        logger.info("BacktestingService cleanup completed")

def fetch_historical_data(symbol, timeframe, limit=1000, start_date=None):
    """Fallback data fetching - returns empty DataFrame."""
    import pandas as pd
    return pd.DataFrame()

class RiskManager:
    """Fallback risk manager."""
    def __init__(self, config=None):
        self.config = config or {}
    
    def calculate_position_size(self, capital, risk_per_trade=0.01):
        return capital * risk_per_trade
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
# Local cache implementation
class SimpleCache:
    """Simple in-memory cache for the backtesting service."""
    
    def __init__(self):
        self.data = {}
    
    def get(self, key: str, default=None):
        return self.data.get(key, default)
    
    def set(self, key: str, value, ttl: int = None):
        self.data[key] = value
    
    def delete(self, key: str):
        self.data.pop(key, None)
    
    def clear(self):
        self.data.clear()

# from src.cache import Cache

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

    def __init__(self, event_system: Optional[EventSystem] = None, cache: Optional[SimpleCache] = None):
        """
        Initialize the backtesting service.

        Args:
            event_system: Optional event system for async communication
            cache: Optional cache system for data persistence
        """
        self.event_system = event_system or EventSystem()
        self.cache = cache or SimpleCache()
        self.active_backtests = {}  # Track running backtests
        self.backtest_results = {}  # Store completed results

        # Initialize data service client if available
        self.data_client = None
        if ENABLE_DATA_SERVICE:
            self.data_client = DataService()

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
                                config: Optional[Dict] = None, backtest_id: Optional[str] = None,
                                data_source: str = 'kaggle') -> Dict:
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
                config=config,
                data_source=data_source,
                data_client=self.data_client
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
            try:
                from src.walk_forward_optimizer import WalkForwardOptimizer
                optimizer_available = True
            except ImportError:
                optimizer_available = False
                return {"error": "Walk-forward optimizer not available"}

            if not optimizer_available:
                return {"error": "Walk-forward optimizer not available"}

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
                 initial_capital: float = CAPITAL, config: Optional[Dict] = None,
                 data_source: str = 'kaggle', data_client: Optional['DataService'] = None):
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
            data_source: Data source to use ('kaggle', 'ccxt', 'yahoo')
            data_client: DataService instance
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
        self.data_source = data_source
        self.data_client = data_client

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

            # Fetch additional data sources (Funding Rate, Liquidation)
            await self._fetch_additional_data_async()

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
        try:
            # Try to use data service if available and working
            if self.data_client:
                try:
                    await self.data_client.authenticate()
                    response = await self.data_client.fetch_historical_data(
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        data_source=self.data_source
                    )

                    if response.get("success"):
                        # Convert response data to DataFrame
                        data_dict = response.get("data", {})
                        if isinstance(data_dict, dict) and "ohlcv" in data_dict:
                            # Convert from service format to DataFrame
                            ohlcv_data = data_dict["ohlcv"]
                            df = pd.DataFrame.from_dict(ohlcv_data, orient='index')
                            df.index = pd.to_datetime(df.index)
                            df = df.sort_index()
                            return df
                        else:
                            logger.warning("Unexpected data format from data service")
                except Exception as e:
                    logger.warning(f"Data service failed: {e}, falling back to legacy method")

            # Fallback to legacy method
            return await self._fetch_data_legacy_async()

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    async def _fetch_data_legacy_async(self) -> pd.DataFrame:
        """Legacy data fetching method for fallback."""
        try:
            # Try Yahoo Finance first for reliable data
            from services.data_service.data_service.data_sources.yahoo_finance import YahooFinanceDataSource
            yahoo_source = YahooFinanceDataSource()

            # Convert symbol format for Yahoo Finance
            yahoo_symbol = self.symbol.replace('/', '-')
            if 'BTC' in yahoo_symbol and 'USDT' in yahoo_symbol:
                yahoo_symbol = 'BTC-USD'  # Yahoo Finance uses BTC-USD for Bitcoin

            df = yahoo_source.fetch_historical_data(
                symbol=yahoo_symbol,
                timeframe=self.timeframe,
                start_date=self.start_date,
                end_date=self.end_date
            )

            if not df.empty:
                logger.info(f"Successfully fetched {len(df)} records from Yahoo Finance")
                # Ensure timezone naive for consistency
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df

        except Exception as e:
            logger.warning(f"Yahoo Finance failed: {e}, falling back to legacy method")

        # Original legacy method as final fallback
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

    async def _fetch_additional_data_async(self) -> None:
        """
        Fetch additional data sources (Funding Rate, Liquidation) for enhanced signals.
        """
        try:
            logger.info("Fetching additional data sources...")

            # Import data sources directly
            from services.data_service.data_service.data_sources.funding_rate import FundingRateDataSource
            from services.data_service.data_service.data_sources.liquidation import LiquidationDataSource

            # Initialize data sources
            funding_source = FundingRateDataSource()
            liquidation_source = LiquidationDataSource()

            # Fetch funding rate data (run in thread pool since it's synchronous)
            funding_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: funding_source.fetch_historical_data(
                    symbol=self.symbol.replace('/', ''),  # Remove slash for Binance format
                    timeframe='8h',  # Funding rates are 8-hour intervals
                    start_date=self.start_date,
                    end_date=self.end_date
                )
            )

            # Ensure consistent timezone handling
            if not funding_data.empty:
                if funding_data.index.tz is not None:
                    funding_data.index = funding_data.index.tz_localize(None)

            if not funding_data.empty:
                # Merge funding rate data
                self.data = self.data.merge(
                    funding_data[['funding_rate', 'market_regime']],
                    left_index=True,
                    right_index=True,
                    how='left'
                )
                logger.info(f"Added funding rate data: {len(funding_data)} records")
            else:
                logger.warning("No funding rate data available")

            # Fetch liquidation data (aggregated over timeframes)
            liquidation_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: liquidation_source.fetch_historical_data(
                    symbol=self.symbol.replace('/', ''),  # Remove slash for exchange format
                    timeframe=self.timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
            )

            # Ensure consistent timezone handling
            if not liquidation_data.empty:
                # Make timezone naive if needed
                if liquidation_data.index.tz is not None:
                    liquidation_data.index = liquidation_data.index.tz_localize(None)

            if not liquidation_data.empty:
                # Merge liquidation data
                self.data = self.data.merge(
                    liquidation_data[['liquidation_signal', 'total_value_liquidated']],
                    left_index=True,
                    right_index=True,
                    how='left'
                )
                logger.info(f"Added liquidation data: {len(liquidation_data)} records")
            else:
                # Try alternative volatility data as fallback
                logger.info("Liquidation data not available, trying volatility indicators as alternative...")
                try:
                    from services.data_service.data_service.data_sources.volatility import VolatilityDataSource
                    volatility_source = VolatilityDataSource()
                    volatility_data = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: volatility_source.fetch_historical_data(
                            symbol=self.symbol.replace('/', ''),
                            timeframe=self.timeframe,
                            start_date=self.start_date,
                            end_date=self.end_date
                        )
                    )

                    # Ensure consistent timezone handling
                    if not volatility_data.empty:
                        if volatility_data.index.tz is not None:
                            volatility_data.index = volatility_data.index.tz_localize(None)

                    if not volatility_data.empty:
                        # Use volatility data as liquidation proxy
                        self.data = self.data.merge(
                            volatility_data[['volatility_signal', 'liquidation_proxy']].rename(
                                columns={'volatility_signal': 'liquidation_signal', 'liquidation_proxy': 'total_value_liquidated'}
                            ),
                            left_index=True,
                            right_index=True,
                            how='left'
                        )
                        logger.info(f"Added volatility-based liquidation proxy data: {len(volatility_data)} records")
                    else:
                        logger.warning("No alternative volatility data available")
                except Exception as e:
                    logger.warning(f"Failed to fetch alternative volatility data: {e}")

            # Fill missing values
            self.data['funding_rate'] = self.data['funding_rate'].fillna(0)
            self.data['market_regime'] = self.data['market_regime'].fillna('neutral')
            self.data['liquidation_signal'] = self.data['liquidation_signal'].fillna(0)
            self.data['total_value_liquidated'] = self.data['total_value_liquidated'].fillna(0)

        except Exception as e:
            logger.error(f"Error fetching additional data: {e}")
            # Continue without additional data if there's an error

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
        # Integrate additional data sources into signals
        data = await loop.run_in_executor(
            None,
            lambda: self._integrate_additional_signals(data)
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
            except (ImportError, Exception) as e:
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
            except (ImportError, Exception) as e:
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
            except (ImportError, Exception) as e:
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
        """Synchronous trade simulation with corrected signal interpretation."""
        if data.empty:
            return []

        # Drop rows with NaN ATR values
        if 'ATR' in data.columns:
            data = data.dropna(subset=['ATR'])

        required_cols = ['close', 'Buy_Signal', 'Sell_Signal']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data missing required columns: {required_cols}")

        # Correct signal interpretation:
        # Buy_Signal = 1 means go long (position = 1)
        # Sell_Signal = 1 means go short (position = -1)
        # No signal means maintain current position

        trades = []
        current_position = 0  # 0 = neutral, 1 = long, -1 = short
        entry_idx = None
        entry_price = None

        position_array = np.zeros(len(data))
        timestamps = data.index
        close_prices = data['close'].values

        slippage_pct = 0.001
        position_sizes = data.get('Position_Size_Absolute', np.full(len(data), 1000))

        for idx in range(len(data)):
            buy_signal = data['Buy_Signal'].iloc[idx] == 1
            sell_signal = data['Sell_Signal'].iloc[idx] == 1

            # Determine desired position
            if buy_signal and not sell_signal:
                desired_position = 1  # Go long
            elif sell_signal and not buy_signal:
                desired_position = -1  # Go short
            else:
                desired_position = current_position  # Maintain position

            position_array[idx] = desired_position

            # Position entry or change
            if current_position == 0 and desired_position != 0:
                # Opening new position
                entry_idx = idx
                if desired_position == 1:
                    entry_price = close_prices[idx] * (1 + slippage_pct)
                else:
                    entry_price = close_prices[idx] * (1 - slippage_pct)

                current_position = desired_position
                position_size = position_sizes[idx]

                entry_commission = entry_price * position_size * self.commission

                trade = {
                    'type': 'buy' if current_position == 1 else 'sell',
                    'entry_time': timestamps[idx],
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'stop_loss': None,  # Will be set if available
                    'take_profit': None,
                    'exit_time': None,
                    'exit_price': None,
                    'pnl': 0,
                    'entry_commission': entry_commission,
                    'exit_commission': 0,
                    'status': 'open'
                }
                trades.append(trade)

            # Position exit or reversal
            elif current_position != 0 and desired_position != current_position:
                # Closing or reversing position
                if entry_idx is not None:
                    if current_position == 1:
                        exit_price = close_prices[idx] * (1 - slippage_pct)
                    else:
                        exit_price = close_prices[idx] * (1 + slippage_pct)

                    exit_time = timestamps[idx]
                    exit_commission = exit_price * trades[-1]['position_size'] * self.commission

                    if current_position == 1:
                        gross_pnl = (exit_price - entry_price) * trades[-1]['position_size']
                    else:
                        gross_pnl = (entry_price - exit_price) * trades[-1]['position_size']

                    net_pnl = gross_pnl - trades[-1]['entry_commission'] - exit_commission

                    trades[-1].update({
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'exit_commission': exit_commission,
                        'status': 'closed'
                    })

                    self.current_capital += net_pnl

                # Handle position change
                if desired_position != 0:
                    # Reversal - close old position and open new one
                    entry_idx = idx
                    if desired_position == 1:
                        entry_price = close_prices[idx] * (1 + slippage_pct)
                    else:
                        entry_price = close_prices[idx] * (1 - slippage_pct)

                    current_position = desired_position
                    position_size = position_sizes[idx]

                    entry_commission = entry_price * position_size * self.commission

                    trade = {
                        'type': 'buy' if current_position == 1 else 'sell',
                        'entry_time': timestamps[idx],
                        'entry_price': entry_price,
                        'position_size': position_size,
                        'stop_loss': None,
                        'take_profit': None,
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': 0,
                        'entry_commission': entry_commission,
                        'exit_commission': 0,
                        'status': 'open'
                    }
                    trades.append(trade)
                else:
                    current_position = 0
                    entry_idx = None

        # Close remaining positions at the end
        if current_position != 0 and trades and trades[-1]['status'] == 'open':
            final_price = data.iloc[-1]['close']
            final_time = data.index[-1]

            if current_position == 1:
                exit_price = final_price * (1 - slippage_pct)
            else:
                exit_price = final_price * (1 + slippage_pct)

            exit_commission = exit_price * trades[-1]['position_size'] * self.commission

            if current_position == 1:
                gross_pnl = (exit_price - entry_price) * trades[-1]['position_size']
            else:
                gross_pnl = (entry_price - exit_price) * trades[-1]['position_size']

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
        """Calculate adaptive risk management parameters based on market regime."""
        # For crypto trading, use fixed position size (1 unit of the asset)
        data['Position_Size_Absolute'] = 1.0  # 1 BTC or 1 unit of the asset

        # Adaptive risk management based on market regime
        if 'ATR' in data.columns:
            # Detect market regime for each row
            regime_params = self._get_market_regime_risk_params(data)

            # Apply regime-specific risk parameters
            data['Stop_Loss_Buy'] = data['close'] - (data['ATR'] * regime_params['stop_loss_atr_buy'])
            data['Stop_Loss_Sell'] = data['close'] + (data['ATR'] * regime_params['stop_loss_atr_sell'])
            data['Take_Profit_Buy'] = data['close'] + (data['ATR'] * regime_params['take_profit_atr_buy'])
            data['Take_Profit_Sell'] = data['close'] - (data['ATR'] * regime_params['take_profit_atr_sell'])

            # Store regime information for analysis
            data['Market_Regime'] = regime_params['regime']
            data['Risk_Level'] = regime_params['risk_level']

        return data

    def _get_market_regime_risk_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Determine market regime and return appropriate risk management parameters.

        Returns regime-specific parameters for:
        - Bull Markets: Higher risk tolerance, wider take profits
        - Bear Markets: Conservative risk, tighter stops
        - Sideways Markets: Moderate risk, balanced parameters
        """
        if not all(col in data.columns for col in ['SMA_20', 'SMA_50', 'RSI', 'close']):
            # Fallback to conservative parameters if indicators not available
            return {
                'regime': 'unknown',
                'risk_level': 'conservative',
                'stop_loss_atr_buy': 1.5,
                'stop_loss_atr_sell': 1.5,
                'take_profit_atr_buy': 2.0,
                'take_profit_atr_sell': 2.0
            }

        # Calculate trend indicators
        sma_20 = data['SMA_20']
        sma_50 = data['SMA_50']
        rsi = data['RSI']
        close = data['close']

        # Trend strength (SMA difference)
        trend_strength = abs(sma_20 - sma_50) / sma_50

        # Price momentum (recent price change)
        price_change_pct = close.pct_change(5)  # 5-day change

        # Determine market regime
        bull_market = (
            (sma_20 > sma_50) &  # Uptrend
            (rsi > 50) &  # Bullish momentum
            (price_change_pct > 0.02)  # Strong upward momentum
        )

        bear_market = (
            (sma_20 < sma_50) &  # Downtrend
            (rsi < 50) &  # Bearish momentum
            (price_change_pct < -0.02)  # Strong downward momentum
        )

        # Sideways when neither bull nor bear conditions are strongly met
        sideways_market = ~(bull_market | bear_market)

        # Apply regime-specific risk parameters
        if bull_market.any() and bull_market.iloc[-1]:  # Current regime is bull
            return {
                'regime': 'bull',
                'risk_level': 'aggressive',
                'stop_loss_atr_buy': 2.0,    # Wider stops (more tolerance for volatility)
                'stop_loss_atr_sell': 2.0,
                'take_profit_atr_buy': 3.5,  # Much wider take profits (let winners run)
                'take_profit_atr_sell': 3.5
            }
        elif bear_market.any() and bear_market.iloc[-1]:  # Current regime is bear
            return {
                'regime': 'bear',
                'risk_level': 'conservative',
                'stop_loss_atr_buy': 1.0,    # Tighter stops (quick exits)
                'stop_loss_atr_sell': 1.0,
                'take_profit_atr_buy': 1.5,  # Moderate take profits
                'take_profit_atr_sell': 1.5
            }
        else:  # Sideways market
            return {
                'regime': 'sideways',
                'risk_level': 'moderate',
                'stop_loss_atr_buy': 1.5,    # Balanced parameters
                'stop_loss_atr_sell': 1.5,
                'take_profit_atr_buy': 2.5,  # Moderate take profits
                'take_profit_atr_sell': 2.5
            }

    def _integrate_additional_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate funding rate and liquidation signals into traditional signals.

        This enhances the trading strategy by incorporating:
        - Funding Rate: Market sentiment and regime confirmation
        - Liquidation Data: Volatility and momentum signals
        """
        # Ensure we have the additional data columns
        if 'funding_rate' not in data.columns:
            data['funding_rate'] = 0
        if 'market_regime' not in data.columns:
            data['market_regime'] = 'neutral'
        if 'liquidation_signal' not in data.columns:
            data['liquidation_signal'] = 0
        if 'liquidation_volume' not in data.columns:
            data['liquidation_volume'] = 0

        # Initialize enhanced signal columns
        data['Enhanced_Buy_Signal'] = data['Buy_Signal'].copy()
        data['Enhanced_Sell_Signal'] = data['Sell_Signal'].copy()
        data['Signal_Strength'] = 0.0
        data['Signal_Reason'] = ''

        # Process each row to integrate additional signals
        for idx in range(len(data)):
            buy_signal = data['Buy_Signal'].iloc[idx] == 1
            sell_signal = data['Sell_Signal'].iloc[idx] == 1
            funding_rate = data['funding_rate'].iloc[idx]
            market_regime = data['market_regime'].iloc[idx]
            liquidation_signal = data['liquidation_signal'].iloc[idx]
            liquidation_volume = data['liquidation_volume'].iloc[idx]

            # Calculate signal strength based on multiple factors
            strength = 0.0
            reasons = []

            # 1. Traditional signal strength (base level)
            if buy_signal or sell_signal:
                strength += 0.4
                reasons.append('traditional')

            # 2. Funding rate confirmation
            if funding_rate != 0:
                if buy_signal and funding_rate < -0.01:  # Negative funding rate in bull signal
                    strength += 0.3
                    reasons.append('funding_rate_bullish')
                elif sell_signal and funding_rate > 0.01:  # Positive funding rate in bear signal
                    strength += 0.3
                    reasons.append('funding_rate_bearish')

            # 3. Market regime confirmation
            if market_regime != 'neutral':
                if buy_signal and market_regime == 'bull':
                    strength += 0.2
                    reasons.append('regime_bullish')
                elif sell_signal and market_regime == 'bear':
                    strength += 0.2
                    reasons.append('regime_bearish')

            # 4. Liquidation signal integration
            if liquidation_signal != 0:
                if buy_signal and liquidation_signal < 0:  # Long liquidations during buy signals
                    strength += 0.2
                    reasons.append('liquidation_bullish')
                elif sell_signal and liquidation_signal > 0:  # Short liquidations during sell signals
                    strength += 0.2
                    reasons.append('liquidation_bearish')

            # 5. Liquidation volume as momentum indicator
            if liquidation_volume > 1000000:  # Significant liquidation volume ($1M+)
                if buy_signal:
                    strength += 0.1
                    reasons.append('high_volume_bullish')
                elif sell_signal:
                    strength += 0.1
                    reasons.append('high_volume_bearish')

            # Apply enhanced signals based on strength threshold
            if strength >= 0.6:  # Strong combined signal
                data.iloc[idx, data.columns.get_loc('Enhanced_Buy_Signal')] = 1 if buy_signal else 0
                data.iloc[idx, data.columns.get_loc('Enhanced_Sell_Signal')] = 1 if sell_signal else 0
            elif strength >= 0.3:  # Moderate signal - keep original
                pass  # Keep original signals
            else:  # Weak signal - reduce or eliminate
                if strength < 0.2:
                    data.iloc[idx, data.columns.get_loc('Enhanced_Buy_Signal')] = 0
                    data.iloc[idx, data.columns.get_loc('Enhanced_Sell_Signal')] = 0

            # Store signal metadata
            data.iloc[idx, data.columns.get_loc('Signal_Strength')] = strength
            data.iloc[idx, data.columns.get_loc('Signal_Reason')] = ', '.join(reasons)

        # Override original signals with enhanced versions
        data['Buy_Signal'] = data['Enhanced_Buy_Signal']
        data['Sell_Signal'] = data['Enhanced_Sell_Signal']

        logger.info(f"Enhanced {data['Buy_Signal'].sum()} buy signals and {data['Sell_Signal'].sum()} sell signals with additional data")

        return data