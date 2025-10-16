#!/usr/bin/env python3
"""
Core Service API - FastAPI REST endpoints for core trading logic.

Provides RESTful API endpoints for:
- Signal generation and validation
- Indicator calculations
- Trading strategy execution
- Market analysis and insights
"""

import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .service import CoreService, EventSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
core_service: Optional[CoreService] = None
event_system = EventSystem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global core_service

    # Startup
    logger.info("Starting Core Service API")
    core_service = CoreService(event_system=event_system)

    yield

    # Shutdown
    logger.info("Shutting down Core Service API")


# Create FastAPI app
app = FastAPI(
    title="Core Service API",
    description="Core trading logic and signal generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignalRequest(BaseModel):
    """Request model for signal generation."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    data: List[Dict[str, Any]] = Field(..., description="OHLCV data")
    strategy_config: Optional[Dict[str, Any]] = Field(None, description="Strategy configuration")


class IndicatorRequest(BaseModel):
    """Request model for indicator calculation."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    data: List[Dict[str, Any]] = Field(..., description="OHLCV data")
    indicators: List[str] = Field(..., description="Indicators to calculate")


class StrategyExecutionRequest(BaseModel):
    """Request model for strategy execution."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    signal: Dict[str, Any] = Field(..., description="Trading signal")
    capital: float = Field(..., description="Available capital")
    risk_config: Dict[str, Any] = Field(..., description="Risk management config")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Core Service",
        "version": "1.0.0",
        "description": "Core trading logic and signal generation"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if not core_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    health_data = await core_service.health_check()
    return health_data


@app.post("/signals/generate", response_model=Dict[str, Any])
async def generate_signals(request: SignalRequest):
    """
    Generate trading signals based on market data and strategy.

    Analyzes market data using technical indicators and generates
    buy/sell/hold signals with confidence scores.
    """
    if not core_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert data to DataFrame
        import pandas as pd
        df = pd.DataFrame(request.data)

        # Generate signals
        signals = await core_service.generate_signals(
            symbol=request.symbol,
            timeframe=request.timeframe,
            data=df,
            strategy_config=request.strategy_config
        )

        logger.info(f"Signals generated for {request.symbol}: {len(signals)} signals")

        return {
            "success": True,
            "signals": signals,
            "symbol": request.symbol,
            "timeframe": request.timeframe
        }

    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/indicators/calculate", response_model=Dict[str, Any])
async def calculate_indicators(request: IndicatorRequest):
    """
    Calculate technical indicators for market data.

    Computes various technical indicators like EMA, RSI, Bollinger Bands, etc.
    """
    if not core_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert data to DataFrame
        import pandas as pd
        df = pd.DataFrame(request.data)

        # Calculate indicators
        indicators = await core_service.calculate_indicators(
            symbol=request.symbol,
            timeframe=request.timeframe,
            data=df,
            indicators=request.indicators
        )

        logger.info(f"Indicators calculated for {request.symbol}: {list(indicators.keys())}")

        return {
            "success": True,
            "indicators": indicators,
            "symbol": request.symbol,
            "timeframe": request.timeframe
        }

    except Exception as e:
        logger.error(f"Indicator calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/strategy/execute", response_model=Dict[str, Any])
async def execute_strategy(request: StrategyExecutionRequest):
    """
    Execute trading strategy based on signal and risk parameters.

    Validates signal, applies risk management, and generates trade execution details.
    """
    if not core_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Execute strategy
        execution = await core_service.execute_strategy(
            symbol=request.symbol,
            timeframe=request.timeframe,
            signal=request.signal,
            capital=request.capital,
            risk_config=request.risk_config
        )

        logger.info(f"Strategy executed for {request.symbol}: {execution.get('action', 'N/A')}")

        return {
            "success": True,
            "execution": execution,
            "symbol": request.symbol,
            "timeframe": request.timeframe
        }

    except Exception as e:
        logger.error(f"Strategy execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/market/{symbol}")
async def get_market_analysis(symbol: str, timeframe: str = "1h"):
    """Get comprehensive market analysis for a symbol."""
    if not core_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        analysis = await core_service.get_market_analysis(symbol, timeframe)
        return analysis
    except Exception as e:
        logger.error(f"Market analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/strategies")
async def list_strategies():
    """List available trading strategies."""
    if not core_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "strategies": core_service.get_available_strategies(),
        "count": len(core_service.get_available_strategies())
    }


@app.get("/indicators")
async def list_indicators():
    """List available technical indicators."""
    if not core_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "indicators": core_service.get_available_indicators(),
        "count": len(core_service.get_available_indicators())
    }


@app.get("/performance/{symbol}")
async def get_performance_metrics(symbol: str):
    """Get performance metrics for a symbol."""
    if not core_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        metrics = await core_service.get_performance_metrics(symbol)
        return metrics
    except Exception as e:
        logger.error(f"Performance metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Event handlers for logging
async def handle_signal_events(event_data: Dict[str, Any]):
    """Handle signal generation events."""
    logger.info(f"Signal event: {event_data}")

async def handle_strategy_events(event_data: Dict[str, Any]):
    """Handle strategy execution events."""
    logger.info(f"Strategy event: {event_data}")

# Subscribe to events
event_system.subscribe("core.signal_generated", handle_signal_events)
event_system.subscribe("core.strategy_executed", handle_strategy_events)


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )