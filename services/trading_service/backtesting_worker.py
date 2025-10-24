#!/usr/bin/env python3
"""
Backtesting Worker Process - Dedicated process for resource-intensive backtesting operations.

This worker provides:
- Isolated execution environment for CPU-intensive backtesting
- Event-driven communication with main trading system
- Dedicated resource allocation (CPU cores, memory)
- Asynchronous processing of backtesting requests

The worker runs as a separate process to avoid interfering with
real-time trading operations while maintaining code consolidation.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional
import multiprocessing
import os
import psutil

from config.settings import LOG_FILE
from services.trading_service.backtesting_service.service import BacktestingService
from services.infrastructure_service.event_system_service import (
    EventSystem, Event, EventType, get_event_system
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE.replace('.log', '_backtesting_worker.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class BacktestingWorker:
    """
    Dedicated worker process for backtesting operations.

    Provides isolated execution environment with:
    - Dedicated CPU cores for intensive computations
    - Separate memory space
    - Event-driven communication
    - Graceful shutdown handling
    """

    def __init__(self, worker_id: str = "backtesting_worker_1"):
        self.worker_id = worker_id
        self.backtesting_service: Optional[BacktestingService] = None
        self.event_system: Optional[EventSystem] = None
        self.running = False
        self.active_backtests: Dict[str, asyncio.Task] = {}

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    async def initialize(self) -> None:
        """Initialize the backtesting worker."""
        try:
            logger.info(f"Initializing Backtesting Worker {self.worker_id}")

            # Initialize backtesting service
            self.backtesting_service = BacktestingService()
            await self.backtesting_service.initialize()

            # Initialize event system
            self.event_system = await get_event_system()

            # Register event handlers
            await self._register_event_handlers()

        except Exception as e:
            logger.error(f"Failed to initialize Backtesting Worker: {e}")
            raise

    async def _register_event_handlers(self) -> None:
        """Register event handlers for backtesting requests."""
        # Register handlers for different types of backtesting requests
        self.event_system.register_handler(EventType.BACKTEST_REQUEST, self._handle_backtest_request)
        self.event_system.register_handler(EventType.MULTI_SYMBOL_BACKTEST_REQUEST, self._handle_multi_symbol_request)
        self.event_system.register_handler(EventType.MULTI_MODEL_BACKTEST_REQUEST, self._handle_multi_model_request)
        self.event_system.register_handler(EventType.WALK_FORWARD_BACKTEST_REQUEST, self._handle_walk_forward_request)
        self.event_system.register_handler(EventType.STRATEGY_OPTIMIZATION_REQUEST, self._handle_optimization_request)

        logger.info("Event handlers registered")

    async def _configure_resource_isolation(self) -> None:
        """Configure resource isolation for optimal backtesting performance."""
        try:
            current_process = psutil.Process()

            # Set lower priority to avoid interfering with live trading
            current_process.nice(10)  # Lower priority (higher nice value)

            # Configure CPU affinity if multiple cores available
            cpu_count = psutil.cpu_count()
            if cpu_count > 2:
                # Reserve cores for live trading, use remaining for backtesting
                reserved_cores = min(2, cpu_count // 4)  # Reserve at least 2 cores or 25%
                available_cores = list(range(reserved_cores, cpu_count))
                if available_cores:
                    current_process.cpu_affinity(available_cores)
                    logger.info(f"CPU affinity set to cores: {available_cores}")

            # Configure memory limits if available
            memory_info = psutil.virtual_memory()
            total_memory_gb = memory_info.total / (1024**3)

            # Reserve 2GB for system/live trading, use rest for backtesting
            if total_memory_gb > 4:
                reserved_memory_gb = 2
                available_memory_gb = total_memory_gb - reserved_memory_gb

                # Set memory limit (soft limit)
                try:
                    import resource
                    memory_limit_bytes = int(available_memory_gb * 1024**3)
                    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
                    logger.info(f"Memory limit set to {available_memory_gb:.1f}GB")
                except ImportError:
                    logger.warning("resource module not available, skipping memory limits")

            logger.info("Resource isolation configured for backtesting worker")

        except Exception as e:
            logger.warning(f"Failed to configure resource isolation: {e}")
            # Continue without resource isolation if configuration fails

    async def _handle_backtest_request(self, event: Event) -> None:
        """Handle single backtest request."""
        try:
            event_data = event.data
            backtest_id = event_data.get('backtest_id', f"bt_{asyncio.get_event_loop().time()}")

            logger.info(f"Processing backtest request {backtest_id}")

            # Create task for async processing
            task = asyncio.create_task(self._process_backtest(backtest_id, event_data))
            self.active_backtests[backtest_id] = task

            # Wait for completion
            result = await task

            # Publish completion event
            await self.event_system.publish_event(Event(
                event_type=EventType.BACKTEST_COMPLETED,
                source=self.worker_id,
                data={
                    "backtest_id": backtest_id,
                    "result": result,
                    "worker_id": self.worker_id
                }
            ))

            # Cleanup
            del self.active_backtests[backtest_id]

        except Exception as e:
            logger.error(f"Failed to handle backtest request: {e}")
            await self.event_system.publish_event(Event(
                event_type=EventType.BACKTEST_FAILED,
                source=self.worker_id,
                data={
                    "backtest_id": event_data.get('backtest_id'),
                    "error": str(e),
                    "worker_id": self.worker_id
                }
            ))

    async def _process_backtest(self, backtest_id: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single backtest asynchronously."""
        try:
            # Extract parameters
            strategy_config = event_data.get('strategy_config', {})
            start_date = event_data.get('start_date')
            end_date = event_data.get('end_date')

            # Run backtest
            result = await self.backtesting_service.run_backtest(
                strategy_config=strategy_config,
                start_date=start_date,
                end_date=end_date,
                backtest_id=backtest_id
            )

            logger.info(f"Backtest {backtest_id} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Backtest {backtest_id} failed: {e}")
            raise

    async def _handle_multi_symbol_request(self, event: Event) -> None:
        """Handle multi-symbol backtest request."""
        try:
            event_data = event.data
            backtest_id = event_data.get('backtest_id', f"multi_symbol_{asyncio.get_event_loop().time()}")

            logger.info(f"Processing multi-symbol backtest request {backtest_id}")

            # Extract parameters
            symbols = event_data.get('symbols', [])
            timeframe = event_data.get('timeframe', '1h')
            start_date = event_data.get('start_date')
            end_date = event_data.get('end_date')
            initial_capital = event_data.get('initial_capital', 10000)
            max_workers = event_data.get('max_workers', min(4, len(symbols)))

            # Run multi-symbol backtest
            result = await self.backtesting_service.run_multi_symbol_backtest_async(
                symbols=symbols,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                max_workers=max_workers,
                backtest_id=backtest_id
            )

            # Publish completion event
            await self.event_system.publish_event(Event(
                event_type=EventType.MULTI_SYMBOL_BACKTEST_COMPLETED,
                source=self.worker_id,
                data={
                    "backtest_id": backtest_id,
                    "result": result,
                    "worker_id": self.worker_id
                }
            ))

        except Exception as e:
            logger.error(f"Multi-symbol backtest failed: {e}")
            await self.event_system.publish_event(Event(
                event_type=EventType.MULTI_SYMBOL_BACKTEST_FAILED,
                source=self.worker_id,
                data={
                    "backtest_id": event_data.get('backtest_id'),
                    "error": str(e),
                    "worker_id": self.worker_id
                }
            ))

    async def _handle_multi_model_request(self, event: Event) -> None:
        """Handle multi-model backtest request."""
        try:
            event_data = event.data
            backtest_id = event_data.get('backtest_id', f"multi_model_{asyncio.get_event_loop().time()}")

            logger.info(f"Processing multi-model backtest request {backtest_id}")

            # Extract parameters
            symbol = event_data.get('symbol', 'BTC/USDT')
            timeframe = event_data.get('timeframe', '1h')
            start_date = event_data.get('start_date')
            end_date = event_data.get('end_date')
            initial_capital = event_data.get('initial_capital', 10000)
            models_to_test = event_data.get('models_to_test')
            max_workers = event_data.get('max_workers', 4)

            # Run multi-model backtest
            result = await self.backtesting_service.run_multi_model_backtest_async(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                models_to_test=models_to_test,
                max_workers=max_workers,
                backtest_id=backtest_id
            )

            # Publish completion event
            await self.event_system.publish_event(Event(
                event_type=EventType.MULTI_MODEL_BACKTEST_COMPLETED,
                source=self.worker_id,
                data={
                    "backtest_id": backtest_id,
                    "result": result,
                    "worker_id": self.worker_id
                }
            ))

        except Exception as e:
            logger.error(f"Multi-model backtest failed: {e}")
            await self.event_system.publish_event(Event(
                event_type=EventType.MULTI_MODEL_BACKTEST_FAILED,
                source=self.worker_id,
                data={
                    "backtest_id": event_data.get('backtest_id'),
                    "error": str(e),
                    "worker_id": self.worker_id
                }
            ))

    async def _handle_walk_forward_request(self, event: Event) -> None:
        """Handle walk-forward optimization request."""
        try:
            event_data = event.data
            backtest_id = event_data.get('backtest_id', f"walk_forward_{asyncio.get_event_loop().time()}")

            logger.info(f"Processing walk-forward optimization request {backtest_id}")

            # Extract parameters
            parameter_grid = event_data.get('parameter_grid', {})
            evaluation_metric = event_data.get('evaluation_metric', 'sharpe_ratio')
            symbol = event_data.get('symbol', 'BTC/USDT')
            timeframe = event_data.get('timeframe', '1h')

            # Run walk-forward optimization
            result = await self.backtesting_service.run_walk_forward_backtest_async(
                parameter_grid=parameter_grid,
                evaluation_metric=evaluation_metric,
                symbol=symbol,
                timeframe=timeframe,
                backtest_id=backtest_id
            )

            # Publish completion event
            await self.event_system.publish_event(Event(
                event_type=EventType.WALK_FORWARD_BACKTEST_COMPLETED,
                source=self.worker_id,
                data={
                    "backtest_id": backtest_id,
                    "result": result,
                    "worker_id": self.worker_id
                }
            ))

        except Exception as e:
            logger.error(f"Walk-forward optimization failed: {e}")
            await self.event_system.publish_event(Event(
                event_type=EventType.WALK_FORWARD_BACKTEST_FAILED,
                source=self.worker_id,
                data={
                    "backtest_id": event_data.get('backtest_id'),
                    "error": str(e),
                    "worker_id": self.worker_id
                }
            ))

    async def _handle_optimization_request(self, event: Event) -> None:
        """Handle strategy optimization request."""
        try:
            event_data = event.data
            backtest_id = event_data.get('backtest_id', f"optimization_{asyncio.get_event_loop().time()}")

            logger.info(f"Processing optimization request {backtest_id}")

            # Extract parameters
            strategy_config = event_data.get('strategy_config', {})
            optimization_config = event_data.get('optimization_config', {})
            start_date = event_data.get('start_date')
            end_date = event_data.get('end_date')

            # Run optimization
            result = await self.backtesting_service.optimize_strategy(
                strategy_config=strategy_config,
                optimization_config=optimization_config,
                start_date=start_date,
                end_date=end_date
            )

            # Publish completion event
            await self.event_system.publish_event(Event(
                event_type=EventType.STRATEGY_OPTIMIZATION_COMPLETED,
                source=self.worker_id,
                data={
                    "backtest_id": backtest_id,
                    "result": result,
                    "worker_id": self.worker_id
                }
            ))

        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")
            await self.event_system.publish_event(Event(
                event_type=EventType.STRATEGY_OPTIMIZATION_FAILED,
                source=self.worker_id,
                data={
                    "backtest_id": event_data.get('backtest_id'),
                    "error": str(e),
                    "worker_id": self.worker_id
                }
            ))

    async def run(self) -> None:
        """Run the backtesting worker main loop."""
        self.running = True
        logger.info(f"Starting Backtesting Worker {self.worker_id}")

        # Start consuming events
        consume_task = asyncio.create_task(self.event_system.start_consuming())

        try:
            while self.running:
                # Publish heartbeat
                await self.event_system.publish_event(Event(
                    event_type=EventType.BACKTESTING_WORKER_HEARTBEAT,
                    source=self.worker_id,
                    data={
                        "worker_id": self.worker_id,
                        "active_backtests": len(self.active_backtests),
                        "status": "running"
                    }
                ))

                # Small delay to prevent busy waiting
                await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"Error in worker main loop: {e}")
        finally:
            # Cancel consume task
            consume_task.cancel()
            try:
                await consume_task
            except asyncio.CancelledError:
                pass
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown the backtesting worker gracefully."""
        logger.info(f"Shutting down Backtesting Worker {self.worker_id}")

        self.running = False

        # Cancel active backtests
        for backtest_id, task in self.active_backtests.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled active backtest {backtest_id}")

        # Close services
        if self.backtesting_service:
            await self.backtesting_service.cleanup()

        if self.event_system:
            await self.event_system.close()

        logger.info(f"Backtesting Worker {self.worker_id} shutdown complete")


async def main():
    """Main entry point for the backtesting worker process."""
    try:
        # Get worker ID from environment or command line
        worker_id = os.getenv('BACKTESTING_WORKER_ID', f"worker_{os.getpid()}")

        # Create and initialize worker
        worker = BacktestingWorker(worker_id=worker_id)
        await worker.initialize()

        # Run worker
        await worker.run()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        sys.exit(1)


def start_backtesting_worker():
    """Function to start backtesting worker in a separate process."""
    asyncio.run(main())


if __name__ == "__main__":
    # Configure process for optimal backtesting performance
    multiprocessing.set_start_method('spawn', force=True)

    # Run the worker
    asyncio.run(main())