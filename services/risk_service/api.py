#!/usr/bin/env python3
"""
Risk Service API - REST API for the Risk Service.

Provides endpoints for:
- Position sizing calculations
- Risk assessment and metrics
- Portfolio risk management
- Risk parameter configuration
"""

import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .service import RiskService, RiskRequest, RiskResponse, EventSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
risk_service: Optional[RiskService] = None
event_system = EventSystem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global risk_service

    # Startup
    logger.info("Starting Risk Service API")
    risk_service = RiskService(event_system=event_system)

    yield

    # Shutdown
    logger.info("Shutting down Risk Service API")


# Create FastAPI app
app = FastAPI(
    title="Risk Service API",
    description="Comprehensive risk management for trading operations",
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


class PositionSizingRequest(BaseModel):
    """Request model for position sizing."""
    symbol: str = Field(..., description="Trading symbol")
    capital: float = Field(..., description="Available capital")
    entry_price: float = Field(..., description="Entry price")
    position_type: str = Field(..., description="Position type (long/short)")
    atr_value: Optional[float] = Field(None, description="ATR value for SL/TP calc")
    volatility: Optional[float] = Field(None, description="Current volatility")
    risk_percentage: float = Field(0.01, description="Risk percentage per trade")


class RiskAssessmentRequest(BaseModel):
    """Request model for risk assessment."""
    returns_data: list = Field(..., description="List of return values")
    time_horizon: str = Field("daily", description="Risk assessment horizon")


class RiskParametersUpdate(BaseModel):
    """Request model for updating risk parameters."""
    max_risk_per_trade: Optional[float] = Field(None, description="Max risk per trade")
    max_portfolio_risk: Optional[float] = Field(None, description="Max portfolio risk")
    max_leverage: Optional[float] = Field(None, description="Maximum leverage")
    stop_loss_atr_multiplier: Optional[float] = Field(None, description="SL ATR multiplier")
    take_profit_atr_multiplier: Optional[float] = Field(None, description="TP ATR multiplier")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Risk Service",
        "version": "1.0.0",
        "description": "Comprehensive risk management for trading operations"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if not risk_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    health_data = await risk_service.health_check()
    return health_data


@app.post("/position/sizing", response_model=RiskResponse)
async def calculate_position_sizing(request: PositionSizingRequest):
    """
    Calculate optimal position sizing with risk management.

    Calculates position size, stop-loss, take-profit, and leverage
    based on risk parameters and market conditions.
    """
    if not risk_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert to service request
        service_request = RiskRequest(**request.dict())

        # Calculate position sizing
        sizing = await risk_service.calculate_position_sizing(service_request)

        logger.info(f"Position sizing calculated for {request.symbol}: "
                   f"Size={sizing.position_size:.4f}, Risk={sizing.risk_amount:.2f}")

        return RiskResponse(
            success=True,
            position_sizing=sizing.to_dict()
        )

    except Exception as e:
        logger.error(f"Position sizing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/assess", response_model=RiskResponse)
async def assess_portfolio_risk(request: RiskAssessmentRequest):
    """
    Assess portfolio risk with comprehensive metrics.

    Calculates Sharpe ratio, VaR, maximum drawdown, and other risk metrics.
    """
    if not risk_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        import pandas as pd
        import numpy as np

        # Convert returns data to pandas Series
        returns = pd.Series(request.returns_data)

        # Assess portfolio risk
        assessment = await risk_service.assess_portfolio_risk(
            returns, request.time_horizon
        )

        logger.info(f"Portfolio risk assessed: {assessment.risk_level.value} "
                   f"(score: {assessment.risk_score:.1f})")

        return RiskResponse(
            success=True,
            risk_assessment=assessment.to_dict()
        )

    except Exception as e:
        logger.error(f"Portfolio assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio/exposure")
async def get_portfolio_exposure():
    """Get current portfolio risk exposure."""
    if not risk_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        exposure = await risk_service.get_portfolio_exposure()
        return exposure
    except Exception as e:
        logger.error(f"Portfolio exposure retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/parameters")
async def update_risk_parameters(request: RiskParametersUpdate):
    """Update risk management parameters."""
    if not risk_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert to dict, filtering out None values
        params = {k: v for k, v in request.dict().items() if v is not None}

        await risk_service.update_risk_parameters(params)

        return {
            "message": "Risk parameters updated successfully",
            "updated_parameters": params
        }

    except Exception as e:
        logger.error(f"Parameter update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/parameters")
async def get_risk_parameters():
    """Get current risk management parameters."""
    if not risk_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return risk_service.default_params.to_dict()


@app.get("/metrics/{symbol}")
async def get_position_metrics(symbol: str):
    """Get risk metrics for a specific position."""
    if not risk_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if symbol not in risk_service.portfolio_positions:
        raise HTTPException(status_code=404, detail=f"Position not found: {symbol}")

    position = risk_service.portfolio_positions[symbol]

    # Calculate additional metrics
    metrics = {
        "position_size": position.position_size,
        "position_value": position.position_value,
        "risk_amount": position.risk_amount,
        "leverage": position.leverage,
        "risk_percentage": position.risk_percentage,
        "reward_risk_ratio": position.reward_risk_ratio,
        "stop_loss_distance": abs(position.stop_loss_price - position.position_value / position.position_size),
        "take_profit_distance": abs(position.take_profit_price - position.position_value / position.position_size),
        "calculated_at": position.calculated_at.isoformat()
    }

    return metrics


@app.delete("/portfolio/{symbol}")
async def close_position(symbol: str):
    """Close/remove a position from portfolio tracking."""
    if not risk_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if symbol not in risk_service.portfolio_positions:
        raise HTTPException(status_code=404, detail=f"Position not found: {symbol}")

    # Remove position
    del risk_service.portfolio_positions[symbol]

    logger.info(f"Position closed: {symbol}")

    return {
        "message": f"Position {symbol} closed successfully"
    }


@app.get("/history")
async def get_risk_history(limit: int = 10):
    """Get risk assessment history."""
    if not risk_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    history = risk_service.risk_history[-limit:] if risk_service.risk_history else []

    return {
        "history": history,
        "total_entries": len(risk_service.risk_history)
    }


# Event handlers for logging
async def handle_risk_events(event_data: Dict[str, Any]):
    """Handle risk service events."""
    event_type = asyncio.current_task().get_name() if asyncio.current_task() else "unknown"

    if "risk.position_sized" in str(event_type):
        logger.info(f"Position sized: {event_data}")
    elif "risk.portfolio_assessed" in str(event_type):
        logger.info(f"Portfolio assessed: {event_data}")
    elif "risk.parameters_updated" in str(event_type):
        logger.info(f"Risk parameters updated: {event_data}")

# Subscribe to events
event_system.subscribe("risk.position_sized", handle_risk_events)
event_system.subscribe("risk.portfolio_assessed", handle_risk_events)
event_system.subscribe("risk.parameters_updated", handle_risk_events)


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )