"""
TradPal Trading Service API
FastAPI application for unified trading operations
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio

from orchestrator import TradingServiceOrchestrator

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TradPal Trading Service",
    description="Unified API for automated trading, risk management, and AI-powered trading operations",
    version="1.0.0"
)

# Global orchestrator instance
orchestrator = TradingServiceOrchestrator()


# Pydantic models for API requests/responses
class TradingConfig(BaseModel):
    capital: float = Field(default=10000.0, description="Initial capital")
    risk_per_trade: float = Field(default=0.02, description="Risk per trade (decimal)")
    max_positions: int = Field(default=5, description="Maximum open positions")
    paper_trading: bool = Field(default=True, description="Enable paper trading mode")
    strategy: str = Field(default="smart_ai", description="Trading strategy")
    rl_enabled: bool = Field(default=True, description="Enable reinforcement learning")
    regime_detection: bool = Field(default=True, description="Enable market regime detection")

class StartTradingRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    config: TradingConfig = Field(default_factory=TradingConfig, description="Trading configuration")

class StartTradingResponse(BaseModel):
    success: bool
    session: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class MarketData(BaseModel):
    current_price: float = Field(..., description="Current market price")
    prices: List[float] = Field(default_factory=list, description="Historical price data")
    volumes: List[float] = Field(default_factory=list, description="Historical volume data")

class SmartTradeRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    market_data: MarketData = Field(..., description="Current market data")

class SmartTradeResponse(BaseModel):
    success: bool
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class TradingStatusResponse(BaseModel):
    success: bool
    status: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class PerformanceReportResponse(BaseModel):
    success: bool
    report: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup"""
    try:
        await orchestrator.initialize()
        logger.info("Trading Service API started successfully")
    except Exception as e:
        logger.error(f"Failed to start Trading Service API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the orchestrator on shutdown"""
    try:
        await orchestrator.shutdown()
        logger.info("Trading Service API shut down successfully")
    except Exception as e:
        logger.error(f"Error during Trading Service API shutdown: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "trading_service",
        "timestamp": datetime.now().isoformat(),
        "components": orchestrator.get_service_status()
    }

@app.post("/trading/start", response_model=StartTradingResponse)
async def start_automated_trading(request: StartTradingRequest):
    """Start automated trading for a symbol"""
    try:
        result = await orchestrator.start_automated_trading(request.symbol, request.config.dict())
        return StartTradingResponse(success=True, session=result)
    except Exception as e:
        logger.error(f"Failed to start trading for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trading/smart-trade", response_model=SmartTradeResponse)
async def execute_smart_trade(request: SmartTradeRequest):
    """Execute a smart trade using AI and risk management"""
    try:
        result = await orchestrator.execute_smart_trade(request.symbol, request.market_data.dict())
        return SmartTradeResponse(success=True, result=result)
    except Exception as e:
        logger.error(f"Failed to execute smart trade for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading/status", response_model=TradingStatusResponse)
async def get_trading_status():
    """Get comprehensive trading status"""
    try:
        status = await orchestrator.get_trading_status()
        return TradingStatusResponse(success=True, status=status)
    except Exception as e:
        logger.error(f"Failed to get trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trading/stop")
async def stop_all_trading():
    """Emergency stop all trading activities"""
    try:
        result = await orchestrator.stop_all_trading()
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Failed to stop trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading/performance", response_model=PerformanceReportResponse)
async def get_performance_report(symbol: Optional[str] = None):
    """Get performance report"""
    try:
        report = await orchestrator.get_performance_report(symbol)
        return PerformanceReportResponse(success=True, report=report)
    except Exception as e:
        logger.error(f"Failed to get performance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_service_status():
    """Get detailed service status"""
    return orchestrator.get_service_status()

@app.get("/config/default")
async def get_default_config():
    """Get default trading configuration"""
    return orchestrator.get_default_trading_config()


# Main entry point for running the service
def main():
    """Run the Trading Service API"""
    import argparse

    parser = argparse.ArgumentParser(description="TradPal Trading Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    logger.info(f"Starting Trading Service API on {args.host}:{args.port}")

    uvicorn.run(
        "services.trading_service.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()