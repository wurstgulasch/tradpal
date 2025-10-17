"""
Market Regime Detection Service - REST API
Provides market regime analysis using clustering algorithms and generates trading signals.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .regime_analyzer import RegimeAnalyzer, MarketRegime, TradingSignal, RiskLevel
from .clustering_engine import ClusteringEngine

# Event system imports with fallback
try:
    from services.event_system import (
        publish_market_regime_change,
        publish_regime_signal,
        publish_regime_transition,
        EVENT_SYSTEM_AVAILABLE
    )
except ImportError:
    EVENT_SYSTEM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Event system not available, running without event publishing")

    # Fallback functions
    async def publish_market_regime_change(*args, **kwargs):
        logger.debug("Event system not available - skipping market regime change event")

    async def publish_regime_signal(*args, **kwargs):
        logger.debug("Event system not available - skipping regime signal event")

    async def publish_regime_transition(*args, **kwargs):
        logger.debug("Event system not available - skipping regime transition event")

logger = logging.getLogger(__name__)

# Global service instance
regime_analyzer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global regime_analyzer

    # Startup
    logger.info("Starting Market Regime Detection Service")
    regime_analyzer = RegimeAnalyzer()

    yield

    # Shutdown
    logger.info("Shutting down Market Regime Detection Service")


# FastAPI app
app = FastAPI(
    title="Market Regime Detection Service",
    description="AI-powered market regime detection using clustering algorithms",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class MarketDataRequest(BaseModel):
    """Request model for market data analysis."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTC/USDT')")
    data: Dict[str, List[Any]] = Field(..., description="OHLCV and technical indicator data")
    include_signals: bool = Field(True, description="Whether to include trading signals")
    background_analysis: bool = Field(False, description="Run analysis in background")


class RegimeAnalysisResponse(BaseModel):
    """Response model for regime analysis."""
    timestamp: datetime
    symbol: str
    current_regime: str
    regime_confidence: float
    volatility_regime: str
    trend_strength: float
    feature_importance: Dict[str, float]


class TradingSignalResponse(BaseModel):
    """Response model for trading signals."""
    timestamp: datetime
    symbol: str
    regime: str
    signal: str
    confidence: float
    risk_level: str
    position_size_multiplier: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    reasoning: str


class RegimeStatisticsResponse(BaseModel):
    """Response model for regime statistics."""
    total_observations: int
    regime_frequencies: Dict[str, int]
    most_common_regime: Optional[str]
    avg_duration_days: Dict[str, float]
    observation_period_days: int


class RegimePredictionResponse(BaseModel):
    """Response model for regime predictions."""
    current_regime: str
    days_in_current_regime: int
    expected_duration_days: int
    transition_probability: float
    predicted_next_regimes: Dict[str, float]
    horizon_days: int


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "market_regime_detection",
        "timestamp": datetime.now(),
        "event_system": EVENT_SYSTEM_AVAILABLE
    }


@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_market_regime(request: MarketDataRequest, background_tasks: BackgroundTasks):
    """
    Analyze market regime for given symbol and data.

    Performs clustering analysis and generates trading signals based on detected regime.
    """
    try:
        if regime_analyzer is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        # Convert data to DataFrame
        df = pd.DataFrame(request.data)

        # Validate required columns
        required_cols = ['close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )

        # Perform analysis
        if request.background_analysis:
            # Run in background
            background_tasks.add_task(
                perform_background_analysis,
                df, request.symbol, request.include_signals
            )
            return {
                "status": "analysis_started",
                "symbol": request.symbol,
                "message": "Analysis running in background"
            }
        else:
            # Run synchronously
            return await perform_analysis(df, request.symbol, request.include_signals)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def perform_analysis(data: pd.DataFrame, symbol: str, include_signals: bool) -> Dict[str, Any]:
    """Perform market regime analysis."""
    try:
        # Analyze regime
        regime_analysis, signal = regime_analyzer.analyze_market_regime(data, symbol)

        # Check for regime transition
        transition = regime_analyzer.detect_regime_transition(
            symbol, regime_analysis.current_regime, regime_analysis.timestamp
        )

        # Publish events if available
        if EVENT_SYSTEM_AVAILABLE:
            try:
                # Publish regime change
                await publish_market_regime_change({
                    "symbol": symbol,
                    "regime": regime_analysis.current_regime.value,
                    "confidence": regime_analysis.regime_confidence,
                    "volatility_regime": regime_analysis.volatility_regime,
                    "trend_strength": regime_analysis.trend_strength,
                    "timestamp": regime_analysis.timestamp.isoformat()
                })

                # Publish trading signal
                if include_signals:
                    await publish_regime_signal({
                        "symbol": symbol,
                        "regime": signal.regime.value,
                        "signal": signal.signal.value,
                        "confidence": signal.confidence,
                        "risk_level": signal.risk_level.value,
                        "position_size_multiplier": signal.position_size_multiplier,
                        "timestamp": signal.timestamp.isoformat()
                    })

                # Publish regime transition if detected
                if transition:
                    await publish_regime_transition({
                        "symbol": symbol,
                        "from_regime": transition.from_regime.value,
                        "to_regime": transition.to_regime.value,
                        "transition_confidence": transition.transition_confidence,
                        "duration_previous_days": transition.duration_in_previous_regime.days,
                        "expected_duration_days": transition.expected_duration_in_new_regime.days,
                        "trading_implications": transition.trading_implications,
                        "timestamp": transition.timestamp.isoformat()
                    })

                logger.info(f"Published regime events for {symbol}")

            except Exception as e:
                logger.error(f"Failed to publish events for {symbol}: {e}")

        # Prepare response
        response = {
            "analysis": RegimeAnalysisResponse(
                timestamp=regime_analysis.timestamp,
                symbol=regime_analysis.symbol,
                current_regime=regime_analysis.current_regime.value,
                regime_confidence=regime_analysis.regime_confidence,
                volatility_regime=regime_analysis.volatility_regime,
                trend_strength=regime_analysis.trend_strength,
                feature_importance=regime_analysis.feature_importance
            )
        }

        if include_signals:
            response["signal"] = TradingSignalResponse(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                regime=signal.regime.value,
                signal=signal.signal.value,
                confidence=signal.confidence,
                risk_level=signal.risk_level.value,
                position_size_multiplier=signal.position_size_multiplier,
                stop_loss_multiplier=signal.stop_loss_multiplier,
                take_profit_multiplier=signal.take_profit_multiplier,
                reasoning=signal.reasoning
            )

        if transition:
            response["transition"] = {
                "from_regime": transition.from_regime.value,
                "to_regime": transition.to_regime.value,
                "transition_confidence": transition.transition_confidence,
                "duration_in_previous_regime_days": transition.duration_in_previous_regime.days,
                "expected_duration_in_new_regime_days": transition.expected_duration_in_new_regime.days,
                "trading_implications": transition.trading_implications
            }

        return response

    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {e}")
        raise


async def perform_background_analysis(data: pd.DataFrame, symbol: str, include_signals: bool):
    """Perform analysis in background and log results."""
    try:
        result = await perform_analysis(data, symbol, include_signals)
        logger.info(f"Background analysis completed for {symbol}: {result['analysis'].current_regime}")
    except Exception as e:
        logger.error(f"Background analysis failed for {symbol}: {e}")


@app.get("/statistics/{symbol}", response_model=RegimeStatisticsResponse)
async def get_regime_statistics(symbol: str):
    """Get regime statistics for a symbol."""
    try:
        if regime_analyzer is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        stats = regime_analyzer.get_regime_statistics(symbol)

        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])

        return RegimeStatisticsResponse(**stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get statistics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/{symbol}", response_model=RegimePredictionResponse)
async def predict_regime(symbol: str, horizon_days: int = 30):
    """Predict future regime for a symbol."""
    try:
        if regime_analyzer is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        prediction = regime_analyzer.get_regime_prediction(symbol, horizon_days)

        if "error" in prediction:
            raise HTTPException(status_code=404, detail=prediction["error"])

        return RegimePredictionResponse(**prediction)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prediction for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/regimes")
async def list_available_regimes():
    """List all available market regimes."""
    return {
        "regimes": [regime.value for regime in MarketRegime],
        "signals": [signal.value for signal in TradingSignal],
        "risk_levels": [level.value for level in RiskLevel]
    }


@app.post("/batch_analyze")
async def batch_analyze_regimes(requests: List[MarketDataRequest], background_tasks: BackgroundTasks):
    """
    Analyze multiple symbols in batch.

    Useful for portfolio-wide regime analysis.
    """
    try:
        if regime_analyzer is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        results = []

        for req in requests:
            try:
                df = pd.DataFrame(req.data)

                if req.background_analysis:
                    background_tasks.add_task(
                        perform_background_analysis,
                        df, req.symbol, req.include_signals
                    )
                    results.append({
                        "symbol": req.symbol,
                        "status": "analysis_started"
                    })
                else:
                    result = await perform_analysis(df, req.symbol, req.include_signals)
                    results.append(result)

            except Exception as e:
                results.append({
                    "symbol": req.symbol,
                    "error": str(e)
                })

        return {
            "batch_results": results,
            "total_requested": len(requests),
            "completed": len([r for r in results if "error" not in r])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Start server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )