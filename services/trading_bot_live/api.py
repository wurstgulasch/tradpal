#!/usr/bin/env python3
"""
Trading Bot Live Service API - FastAPI REST endpoints for live trading execution.

Provides RESTful API endpoints for:
- Live trading execution and monitoring
- Order management and position tracking
- Risk management and position sizing
- Trading strategy execution
- Real-time performance monitoring
"""

import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .service import TradingBotLiveService, EventSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
trading_bot_service: Optional[TradingBotLiveService] = None
event_system = EventSystem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global trading_bot_service

    # Startup
    logger.info("Starting Trading Bot Live Service API")
    trading_bot_service = TradingBotLiveService(event_system=event_system)

    yield

    # Shutdown
    logger.info("Shutting down Trading Bot Live Service API")


# Create FastAPI app
app = FastAPI(
    title="Trading Bot Live Service API",
    description="Live trading execution and monitoring",
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


class StartTradingRequest(BaseModel):
    """Request model for starting live trading."""
    symbol: str = Field(..., description="Trading symbol")
    strategy: str = Field(..., description="Trading strategy")
    timeframe: str = Field("1m", description="Trading timeframe")
    capital: float = Field(..., description="Initial capital")
    risk_per_trade: float = Field(0.01, description="Risk per trade (0.01 = 1%)")
    max_positions: int = Field(1, description="Maximum open positions")
    enable_paper_trading: bool = Field(True, description="Enable paper trading mode")


class StopTradingRequest(BaseModel):
    """Request model for stopping trading."""
    symbol: str = Field(..., description="Trading symbol")
    close_positions: bool = Field(True, description="Close all open positions")


class OrderRequest(BaseModel):
    """Request model for manual order placement."""
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Order side (buy/sell)")
    quantity: float = Field(..., description="Order quantity")
    order_type: str = Field("market", description="Order type")
    price: Optional[float] = Field(None, description="Limit price (for limit orders)")


class RiskUpdateRequest(BaseModel):
    """Request model for updating risk parameters."""
    symbol: str = Field(..., description="Trading symbol")
    risk_per_trade: Optional[float] = Field(None, description="New risk per trade")
    max_positions: Optional[int] = Field(None, description="New max positions")
    max_drawdown: Optional[float] = Field(None, description="New max drawdown")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Trading Bot Live Service",
        "version": "1.0.0",
        "description": "Live trading execution and monitoring"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    health_data = await trading_bot_service.health_check()
    return health_data


@app.post("/start", response_model=Dict[str, Any])
async def start_trading(request: StartTradingRequest):
    """
    Start live trading for a symbol.

    Initializes trading bot with specified parameters and begins monitoring for signals.
    """
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await trading_bot_service.start_trading(
            symbol=request.symbol,
            strategy=request.strategy,
            timeframe=request.timeframe,
            capital=request.capital,
            risk_per_trade=request.risk_per_trade,
            max_positions=request.max_positions,
            enable_paper_trading=request.enable_paper_trading
        )

        logger.info(f"Trading started for {request.symbol} with strategy {request.strategy}")

        return {
            "success": True,
            "message": "Trading started successfully",
            "trading_id": result.get("trading_id"),
            "symbol": request.symbol,
            "strategy": request.strategy
        }

    except Exception as e:
        logger.error(f"Start trading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop", response_model=Dict[str, Any])
async def stop_trading(request: StopTradingRequest):
    """
    Stop live trading for a symbol.

    Stops signal monitoring and optionally closes all positions.
    """
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await trading_bot_service.stop_trading(
            symbol=request.symbol,
            close_positions=request.close_positions
        )

        logger.info(f"Trading stopped for {request.symbol}")

        return {
            "success": True,
            "message": "Trading stopped successfully",
            "symbol": request.symbol,
            "positions_closed": result.get("positions_closed", 0)
        }

    except Exception as e:
        logger.error(f"Stop trading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/order", response_model=Dict[str, Any])
async def place_order(request: OrderRequest):
    """
    Place a manual order.

    Executes a buy or sell order for the specified symbol.
    """
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await trading_bot_service.place_manual_order(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            order_type=request.order_type,
            price=request.price
        )

        logger.info(f"Manual order placed: {request.side} {request.quantity} {request.symbol}")

        return {
            "success": True,
            "message": "Order placed successfully",
            "order_id": result.get("order_id"),
            "symbol": request.symbol,
            "side": request.side,
            "quantity": request.quantity
        }

    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/risk/update", response_model=Dict[str, Any])
async def update_risk_parameters(request: RiskUpdateRequest):
    """
    Update risk management parameters.

    Modifies risk settings for active trading sessions.
    """
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await trading_bot_service.update_risk_parameters(
            symbol=request.symbol,
            risk_per_trade=request.risk_per_trade,
            max_positions=request.max_positions,
            max_drawdown=request.max_drawdown
        )

        logger.info(f"Risk parameters updated for {request.symbol}")

        return {
            "success": True,
            "message": "Risk parameters updated successfully",
            "symbol": request.symbol
        }

    except Exception as e:
        logger.error(f"Risk parameter update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_trading_status():
    """Get status of all trading sessions."""
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        status = await trading_bot_service.get_trading_status()
        return status
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{symbol}")
async def get_symbol_status(symbol: str):
    """Get trading status for a specific symbol."""
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        status = await trading_bot_service.get_symbol_status(symbol)
        return status
    except Exception as e:
        logger.error(f"Symbol status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/positions")
async def get_positions():
    """Get all open positions."""
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        positions = await trading_bot_service.get_positions()
        return positions
    except Exception as e:
        logger.error(f"Positions retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/positions/{symbol}")
async def get_symbol_positions(symbol: str):
    """Get open positions for a specific symbol."""
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        positions = await trading_bot_service.get_symbol_positions(symbol)
        return {"positions": positions, "count": len(positions)}
    except Exception as e:
        logger.error(f"Symbol positions retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance")
async def get_performance():
    """Get overall trading performance."""
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        performance = await trading_bot_service.get_performance_metrics()
        return performance
    except Exception as e:
        logger.error(f"Performance retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/{symbol}")
async def get_symbol_performance(symbol: str):
    """Get performance metrics for a specific symbol."""
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        performance = await trading_bot_service.get_symbol_performance(symbol)
        return performance
    except Exception as e:
        logger.error(f"Symbol performance retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/orders")
async def get_order_history(limit: int = 100):
    """Get recent order history."""
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        orders = await trading_bot_service.get_order_history(limit=limit)
        return {"orders": orders, "count": len(orders)}
    except Exception as e:
        logger.error(f"Order history retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/orders/{symbol}")
async def get_symbol_orders(symbol: str, limit: int = 50):
    """Get order history for a specific symbol."""
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        orders = await trading_bot_service.get_symbol_orders(symbol, limit=limit)
        return {"orders": orders, "count": len(orders)}
    except Exception as e:
        logger.error(f"Symbol order history retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emergency/stop")
async def emergency_stop():
    """Emergency stop all trading activities."""
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await trading_bot_service.emergency_stop()

        logger.warning("Emergency stop activated")

        return {
            "success": True,
            "message": "Emergency stop activated",
            "positions_closed": result.get("positions_closed", 0),
            "trading_sessions_stopped": result.get("sessions_stopped", 0)
        }

    except Exception as e:
        logger.error(f"Emergency stop failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio")
async def get_portfolio():
    """Get portfolio information."""
    if not trading_bot_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        portfolio = await trading_bot_service.get_portfolio()
        return portfolio
    except Exception as e:
        logger.error(f"Portfolio retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Event handlers for logging
async def handle_trade_events(event_data: Dict[str, Any]):
    """Handle trade execution events."""
    logger.info(f"Trade event: {event_data}")

async def handle_signal_events(event_data: Dict[str, Any]):
    """Handle trading signal events."""
    logger.info(f"Signal event: {event_data}")

async def handle_risk_events(event_data: Dict[str, Any]):
    """Handle risk management events."""
    logger.info(f"Risk event: {event_data}")

# Subscribe to events
event_system.subscribe("trading.order_executed", handle_trade_events)
event_system.subscribe("trading.signal_generated", handle_signal_events)
event_system.subscribe("trading.risk_triggered", handle_risk_events)


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        log_level="info"
    )