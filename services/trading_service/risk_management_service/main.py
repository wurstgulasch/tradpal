"""
TradPal Risk Management Service API
FastAPI application for risk management and position sizing
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import asyncio

from service import RiskManagementService

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TradPal Risk Management Service",
    description="Specialized service for risk management and position sizing",
    version="1.0.0"
)

# Global service instance
risk_service = RiskManagementService()


# Pydantic models for API requests/responses
class RiskConfig(BaseModel):
    max_risk_per_trade: float = Field(default=0.02, description="Maximum risk per trade (decimal)")
    max_portfolio_risk: float = Field(default=0.06, description="Maximum portfolio risk (decimal)")
    max_drawdown: float = Field(default=0.10, description="Maximum drawdown (decimal)")
    risk_free_rate: float = Field(default=0.02, description="Risk-free rate for Sharpe ratio")

class PositionSizeRequest(BaseModel):
    capital: float = Field(..., description="Available capital")
    entry_price: float = Field(..., description="Entry price")
    stop_loss: float = Field(..., description="Stop loss price")
    risk_config: RiskConfig = Field(default_factory=RiskConfig, description="Risk configuration")

class PositionSizeResponse(BaseModel):
    position_size: float
    risk_amount: float
    risk_percentage: float
    sharpe_ratio: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class PortfolioRiskRequest(BaseModel):
    positions: List[Dict[str, Any]] = Field(..., description="Current positions")
    capital: float = Field(..., description="Total capital")
    risk_config: RiskConfig = Field(default_factory=RiskConfig, description="Risk configuration")

class PortfolioRiskResponse(BaseModel):
    total_risk: float
    risk_percentage: float
    max_risk_exceeded: bool
    recommendations: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Starting Risk Management Service...")
    await risk_service.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Risk Management Service...")
    await risk_service.close()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "risk_management_service"}


@app.post("/position-size", response_model=PositionSizeResponse)
async def calculate_position_size(request: PositionSizeRequest):
    """Calculate optimal position size based on risk parameters"""
    try:
        result = await risk_service.calculate_position_size(
            request.capital,
            request.entry_price,
            request.stop_loss,
            request.risk_config.dict()
        )

        return PositionSizeResponse(
            position_size=result["position_size"],
            risk_amount=result["risk_amount"],
            risk_percentage=result["risk_percentage"],
            sharpe_ratio=result.get("sharpe_ratio")
        )

    except Exception as e:
        logger.error(f"Position size calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio-risk", response_model=PortfolioRiskResponse)
async def assess_portfolio_risk(request: PortfolioRiskRequest):
    """Assess overall portfolio risk"""
    try:
        result = await risk_service.assess_portfolio_risk(
            request.positions,
            request.capital,
            request.risk_config.dict()
        )

        return PortfolioRiskResponse(
            total_risk=result["total_risk"],
            risk_percentage=result["risk_percentage"],
            max_risk_exceeded=result["max_risk_exceeded"],
            recommendations=result.get("recommendations", [])
        )

    except Exception as e:
        logger.error(f"Portfolio risk assessment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kelly-criterion")
async def calculate_kelly_criterion(win_rate: float, win_loss_ratio: float):
    """Calculate Kelly Criterion for position sizing"""
    try:
        kelly = risk_service.calculate_kelly_criterion(win_rate, win_loss_ratio)
        return {"kelly_percentage": kelly}
    except Exception as e:
        logger.error(f"Kelly criterion calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/var/{confidence}")
async def calculate_var(confidence: float, returns: List[float]):
    """Calculate Value at Risk"""
    try:
        var = risk_service.calculate_var(confidence, returns)
        return {"var": var, "confidence": confidence}
    except Exception as e:
        logger.error(f"VaR calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate-trade")
async def validate_trade(trade: Dict[str, Any], risk_config: RiskConfig):
    """Validate if a trade meets risk management criteria"""
    try:
        validation = await risk_service.validate_trade(trade, risk_config.dict())
        return validation
    except Exception as e:
        logger.error(f"Trade validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8015,  # Different port for risk management service
        reload=True
    )