"""
TradPal Market Regime Service API
FastAPI application for market regime detection and analysis
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import asyncio

from service import MarketRegimeService

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TradPal Market Regime Service",
    description="Specialized service for market regime detection and analysis",
    version="1.0.0"
)

# Global service instance
regime_service = MarketRegimeService()


# Pydantic models for API requests/responses
class RegimeAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    data: List[Dict[str, Any]] = Field(..., description="OHLCV data")
    lookback_periods: int = Field(default=20, description="Lookback periods for analysis")

class RegimeAnalysisResponse(BaseModel):
    regime: str
    confidence: float
    indicators: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class MultiTimeframeRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    timeframes: List[str] = Field(default_factory=lambda: ["1h", "4h", "1d"], description="Timeframes to analyze")
    data: Dict[str, List[Dict[str, Any]]] = Field(..., description="OHLCV data per timeframe")

class MultiTimeframeResponse(BaseModel):
    consensus_regime: str
    timeframe_regimes: Dict[str, str] = Field(default_factory=dict)
    strength_score: float
    alignment_score: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Starting Market Regime Service...")
    await regime_service.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Market Regime Service...")
    await regime_service.close()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "market_regime_service"}


@app.post("/analyze", response_model=RegimeAnalysisResponse)
async def analyze_regime(request: RegimeAnalysisRequest):
    """Analyze market regime for given data"""
    try:
        result = await regime_service.analyze_market_regime(
            request.symbol,
            request.data,
            request.lookback_periods
        )

        return RegimeAnalysisResponse(
            regime=result["regime"],
            confidence=result["confidence"],
            indicators=result.get("indicators", {})
        )

    except Exception as e:
        logger.error(f"Regime analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-multi", response_model=MultiTimeframeResponse)
async def analyze_multi_timeframe(request: MultiTimeframeRequest):
    """Analyze market regime across multiple timeframes"""
    try:
        result = await regime_service.analyze_multi_timeframe_regime(
            request.symbol,
            request.timeframes,
            request.data
        )

        return MultiTimeframeResponse(
            consensus_regime=result["consensus_regime"],
            timeframe_regimes=result["timeframe_regimes"],
            strength_score=result["strength_score"],
            alignment_score=result["alignment_score"]
        )

    except Exception as e:
        logger.error(f"Multi-timeframe analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/regimes")
async def list_available_regimes():
    """List all available market regime types"""
    try:
        regimes = regime_service.get_available_regimes()
        return {"regimes": regimes}
    except Exception as e:
        logger.error(f"Failed to list regimes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics/{symbol}")
async def get_regime_statistics(symbol: str):
    """Get regime statistics for a symbol"""
    try:
        stats = await regime_service.get_regime_statistics(symbol)
        return stats
    except Exception as e:
        logger.error(f"Failed to get regime statistics: {str(e)}")
        raise HTTPException(status_code=404, detail=f"No statistics found for {symbol}")


@app.post("/predict/{symbol}")
async def predict_regime(symbol: str, data: List[Dict[str, Any]]):
    """Predict market regime for future data"""
    try:
        prediction = await regime_service.predict_regime(symbol, data)
        return prediction
    except Exception as e:
        logger.error(f"Regime prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8014,  # Different port for market regime service
        reload=True
    )