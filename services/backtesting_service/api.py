"""
Backtesting Service API - REST endpoints for backtesting operations.

Provides HTTP endpoints for:
- Single backtest execution
- Multi-symbol backtesting
- Multi-model comparison
- Walk-forward optimization
- Backtest status monitoring
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
import uvicorn

from .backtesting.service import BacktestingService
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
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    print(f"Error in event handler: {e}")

# Local cache implementation for service
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


# Pydantic models for request/response validation
class BacktestRequest(BaseModel):
    """Request model for single backtest."""
    symbol: str = Field(default="BTC/USDT", description="Trading symbol")
    timeframe: str = Field(default="1d", description="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    strategy: str = Field(default="traditional", description="Strategy type")
    initial_capital: float = Field(default=10000.0, description="Initial capital")
    config: Optional[Dict] = Field(None, description="Configuration overrides")
    data_source: str = Field(default="kaggle", description="Data source (kaggle, ccxt, yahoo)")


class MultiSymbolRequest(BaseModel):
    """Request model for multi-symbol backtest."""
    symbols: List[str] = Field(..., description="List of trading symbols")
    timeframe: str = Field(default="1d", description="Timeframe for all backtests")
    start_date: Optional[str] = Field(None, description="Start date")
    end_date: Optional[str] = Field(None, description="End date")
    initial_capital: float = Field(default=10000.0, description="Initial capital per backtest")
    max_workers: Optional[int] = Field(None, description="Maximum parallel workers")


class MultiModelRequest(BaseModel):
    """Request model for multi-model backtest."""
    symbol: str = Field(default="BTC/USDT", description="Trading symbol")
    timeframe: str = Field(default="1d", description="Timeframe")
    start_date: Optional[str] = Field(None, description="Start date")
    end_date: Optional[str] = Field(None, description="End date")
    initial_capital: float = Field(default=10000.0, description="Initial capital")
    models_to_test: Optional[List[str]] = Field(None, description="Models to test")
    max_workers: Optional[int] = Field(None, description="Maximum parallel workers")


class WalkForwardRequest(BaseModel):
    """Request model for walk-forward optimization."""
    parameter_grid: Dict = Field(..., description="Parameters to optimize")
    evaluation_metric: str = Field(default="sharpe_ratio", description="Evaluation metric")
    symbol: str = Field(default="BTC/USDT", description="Trading symbol")
    timeframe: str = Field(default="1d", description="Timeframe")


class BacktestResponse(BaseModel):
    """Response model for backtest results."""
    backtest_id: str
    status: str
    result: Optional[Dict] = None
    error: Optional[str] = None


class BacktestingAPI:
    """
    FastAPI-based REST API for the Backtesting Service.

    Provides endpoints for all backtesting operations with proper
    request validation and response formatting.
    """

    def __init__(self, service: BacktestingService):
        """
        Initialize the API.

        Args:
            service: BacktestingService instance
        """
        self.service = service
        self.app = FastAPI(
            title="TradPal Backtesting Service API",
            description="REST API for historical trading strategy backtesting",
            version="1.0.0"
        )

        # Background tasks for async operations
        self.background_tasks = BackgroundTasks()

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "service": "TradPal Backtesting Service",
                "version": "1.0.0",
                "description": "Historical trading strategy backtesting API",
                "endpoints": {
                    "POST /backtest": "Run single backtest",
                    "POST /backtest/multi-symbol": "Run multi-symbol backtest",
                    "POST /backtest/multi-model": "Run multi-model comparison",
                    "POST /backtest/walk-forward": "Run walk-forward optimization",
                    "GET /backtest/{backtest_id}": "Get backtest status/result",
                    "GET /backtest/active": "List active backtests",
                    "DELETE /backtest/completed": "Cleanup completed backtests"
                }
            }

        @self.app.post("/backtest", response_model=BacktestResponse)
        async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
            """Run a single backtest asynchronously."""
            try:
                backtest_id = f"api_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                # Add to background tasks
                background_tasks.add_task(
                    self._run_backtest_background,
                    backtest_id,
                    request.symbol,
                    request.timeframe,
                    request.start_date,
                    request.end_date,
                    request.strategy,
                    request.initial_capital,
                    request.config,
                    request.data_source
                )

                return BacktestResponse(
                    backtest_id=backtest_id,
                    status="running",
                    result=None
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to start backtest: {str(e)}")

        @self.app.post("/backtest/multi-symbol", response_model=BacktestResponse)
        async def run_multi_symbol_backtest(request: MultiSymbolRequest, background_tasks: BackgroundTasks):
            """Run multi-symbol backtest asynchronously."""
            try:
                backtest_id = f"api_multi_symbol_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                background_tasks.add_task(
                    self._run_multi_symbol_background,
                    backtest_id,
                    request.symbols,
                    request.timeframe,
                    request.start_date,
                    request.end_date,
                    request.initial_capital,
                    request.max_workers
                )

                return BacktestResponse(
                    backtest_id=backtest_id,
                    status="running",
                    result=None
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to start multi-symbol backtest: {str(e)}")

        @self.app.post("/backtest/multi-model", response_model=BacktestResponse)
        async def run_multi_model_backtest(request: MultiModelRequest, background_tasks: BackgroundTasks):
            """Run multi-model backtest asynchronously."""
            try:
                backtest_id = f"api_multi_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                background_tasks.add_task(
                    self._run_multi_model_background,
                    backtest_id,
                    request.symbol,
                    request.timeframe,
                    request.start_date,
                    request.end_date,
                    request.initial_capital,
                    request.models_to_test,
                    request.max_workers
                )

                return BacktestResponse(
                    backtest_id=backtest_id,
                    status="running",
                    result=None
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to start multi-model backtest: {str(e)}")

        @self.app.post("/backtest/walk-forward", response_model=BacktestResponse)
        async def run_walk_forward_optimization(request: WalkForwardRequest, background_tasks: BackgroundTasks):
            """Run walk-forward optimization asynchronously."""
            try:
                backtest_id = f"api_walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                background_tasks.add_task(
                    self._run_walk_forward_background,
                    backtest_id,
                    request.parameter_grid,
                    request.evaluation_metric,
                    request.symbol,
                    request.timeframe
                )

                return BacktestResponse(
                    backtest_id=backtest_id,
                    status="running",
                    result=None
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to start walk-forward optimization: {str(e)}")

        @self.app.get("/backtest/{backtest_id}", response_model=BacktestResponse)
        async def get_backtest_status(backtest_id: str):
            """Get the status and result of a specific backtest."""
            try:
                status = await self.service.get_backtest_status(backtest_id)

                if status.get("status") == "not_found":
                    raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

                return BacktestResponse(
                    backtest_id=backtest_id,
                    status=status.get("status", "unknown"),
                    result=status.get("result"),
                    error=status.get("error")
                )

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get backtest status: {str(e)}")

        @self.app.get("/backtest/active")
        async def list_active_backtests():
            """List all currently active backtests."""
            try:
                active_backtests = await self.service.list_active_backtests()
                return {
                    "active_backtests": active_backtests,
                    "count": len(active_backtests)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to list active backtests: {str(e)}")

        @self.app.delete("/backtest/completed")
        async def cleanup_completed_backtests(max_age_hours: int = Query(24, description="Maximum age in hours")):
            """Clean up old completed backtests from memory."""
            try:
                await self.service.cleanup_completed_backtests(max_age_hours)
                return {"message": f"Cleaned up backtests older than {max_age_hours} hours"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to cleanup backtests: {str(e)}")

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "backtesting-service",
                "timestamp": datetime.now().isoformat()
            }

    async def _run_backtest_background(self, backtest_id: str, symbol: str, timeframe: str,
                                     start_date: Optional[str], end_date: Optional[str],
                                     strategy: str, initial_capital: float, config: Optional[Dict],
                                     data_source: str = 'kaggle'):
        """Background task for running single backtest."""
        try:
            result = await self.service.run_backtest_async(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy,
                initial_capital=initial_capital,
                config=config,
                backtest_id=backtest_id,
                data_source=data_source
            )
            # Result is automatically stored in service
        except Exception as e:
            # Error is handled by service
            pass

    async def _run_multi_symbol_background(self, backtest_id: str, symbols: List[str], timeframe: str,
                                         start_date: Optional[str], end_date: Optional[str],
                                         initial_capital: float, max_workers: Optional[int]):
        """Background task for running multi-symbol backtest."""
        try:
            result = await self.service.run_multi_symbol_backtest_async(
                symbols=symbols,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                max_workers=max_workers,
                backtest_id=backtest_id
            )
        except Exception as e:
            pass

    async def _run_multi_model_background(self, backtest_id: str, symbol: str, timeframe: str,
                                        start_date: Optional[str], end_date: Optional[str],
                                        initial_capital: float, models_to_test: Optional[List[str]],
                                        max_workers: Optional[int]):
        """Background task for running multi-model backtest."""
        try:
            result = await self.service.run_multi_model_backtest_async(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                models_to_test=models_to_test,
                max_workers=max_workers,
                backtest_id=backtest_id
            )
        except Exception as e:
            pass

    async def _run_walk_forward_background(self, backtest_id: str, parameter_grid: Dict,
                                         evaluation_metric: str, symbol: str, timeframe: str):
        """Background task for running walk-forward optimization."""
        try:
            result = await self.service.run_walk_forward_backtest_async(
                parameter_grid=parameter_grid,
                evaluation_metric=evaluation_metric,
                symbol=symbol,
                timeframe=timeframe,
                backtest_id=backtest_id
            )
        except Exception as e:
            pass

    def run_server(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the FastAPI server."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )


# Standalone server runner
async def run_backtesting_service(host: str = "0.0.0.0", port: int = 8001):
    """
    Run the backtesting service with API endpoints.

    Args:
        host: Host to bind to
        port: Port to bind to
    """
    # Initialize service components
    event_system = EventSystem()
    cache = SimpleCache()
    service = BacktestingService(event_system=event_system, cache=cache)

    # Create API
    api = BacktestingAPI(service)

    # Start background cleanup task
    async def cleanup_task():
        while True:
            await asyncio.sleep(3600)  # Run every hour
            await service.cleanup_completed_backtests()

    cleanup_coro = asyncio.create_task(cleanup_task())

    # Run server
    try:
        api.run_server(host=host, port=port)
    finally:
        cleanup_coro.cancel()


if __name__ == "__main__":
    # Run service directly
    asyncio.run(run_backtesting_service())