#!/usr/bin/env python3
"""
Web UI Service API - FastAPI REST endpoints for web interface.

Provides RESTful API endpoints for:
- Dashboard data and analytics
- Strategy configuration and management
- Live trading monitoring
- Backtesting visualization
- User authentication and authorization
"""

import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, status
from pydantic import BaseModel, Field

from .service import WebUIService, EventSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
web_ui_service: Optional[WebUIService] = None
event_system = EventSystem()

# Security
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global web_ui_service

    # Startup
    logger.info("Starting Web UI Service API")
    web_ui_service = WebUIService(event_system=event_system)

    yield

    # Shutdown
    logger.info("Shutting down Web UI Service API")


# Create FastAPI app
app = FastAPI(
    title="Web UI Service API",
    description="Web interface for trading platform",
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


class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class StrategyConfig(BaseModel):
    """Strategy configuration model."""
    name: str = Field(..., description="Strategy name")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")


class BacktestRequest(BaseModel):
    """Backtest request model."""
    strategy_name: str = Field(..., description="Strategy name")
    symbol: str = Field(..., description="Trading symbol")
    start_date: str = Field(..., description="Start date")
    end_date: str = Field(..., description="End date")
    initial_capital: float = Field(10000.0, description="Initial capital")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    # Simple token validation (in production, use proper JWT)
    if credentials.credentials != "tradpal_token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return {"username": "admin"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Web UI Service",
        "version": "1.0.0",
        "description": "Web interface for trading platform"
    }


@app.post("/auth/login")
async def login(request: LoginRequest):
    """User login endpoint."""
    if request.username == "admin" and request.password == "admin123":
        return {
            "success": True,
            "token": "tradpal_token",
            "user": {"username": "admin", "role": "admin"}
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")


@app.get("/health")
async def health():
    """Health check endpoint."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    health_data = await web_ui_service.health_check()
    return health_data


@app.get("/dashboard")
async def get_dashboard_data(current_user: dict = Depends(get_current_user)):
    """Get dashboard data."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        dashboard_data = await web_ui_service.get_dashboard_data()
        return dashboard_data
    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/live")
async def get_live_dashboard_data(current_user: dict = Depends(get_current_user)):
    """Get live dashboard data with real-time updates."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        live_data = await web_ui_service.get_live_dashboard_data()
        return live_data
    except Exception as e:
        logger.error(f"Live dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/strategies")
async def get_strategies(current_user: dict = Depends(get_current_user)):
    """Get available trading strategies."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        strategies = await web_ui_service.get_strategies()
        return {"strategies": strategies, "count": len(strategies)}
    except Exception as e:
        logger.error(f"Strategies retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/strategies")
async def create_strategy(
    config: StrategyConfig,
    current_user: dict = Depends(get_current_user)
):
    """Create a new trading strategy."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await web_ui_service.create_strategy(
            name=config.name,
            symbol=config.symbol,
            timeframe=config.timeframe,
            parameters=config.parameters
        )

        logger.info(f"Strategy created: {config.name}")

        return {
            "success": True,
            "message": "Strategy created successfully",
            "strategy_id": result.get("strategy_id"),
            "name": config.name
        }

    except Exception as e:
        logger.error(f"Strategy creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/strategies/{strategy_id}")
async def get_strategy(
    strategy_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get strategy details."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        strategy = await web_ui_service.get_strategy(strategy_id)
        return strategy
    except Exception as e:
        logger.error(f"Strategy retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/strategies/{strategy_id}")
async def update_strategy(
    strategy_id: str,
    config: StrategyConfig,
    current_user: dict = Depends(get_current_user)
):
    """Update an existing strategy."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await web_ui_service.update_strategy(
            strategy_id=strategy_id,
            name=config.name,
            symbol=config.symbol,
            timeframe=config.timeframe,
            parameters=config.parameters
        )

        logger.info(f"Strategy updated: {strategy_id}")

        return {
            "success": True,
            "message": "Strategy updated successfully",
            "strategy_id": strategy_id
        }

    except Exception as e:
        logger.error(f"Strategy update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/strategies/{strategy_id}")
async def delete_strategy(
    strategy_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a strategy."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        success = await web_ui_service.delete_strategy(strategy_id)

        if success:
            return {
                "success": True,
                "message": f"Strategy {strategy_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Strategy not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Strategy deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest")
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Run a backtest for a strategy."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Start backtest in background
        background_tasks.add_task(
            run_backtest_background,
            request.strategy_name,
            request.symbol,
            request.start_date,
            request.end_date,
            request.initial_capital
        )

        return {
            "success": True,
            "message": "Backtest started in background",
            "strategy": request.strategy_name,
            "symbol": request.symbol
        }

    except Exception as e:
        logger.error(f"Backtest request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_backtest_background(
    strategy_name: str,
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float
):
    """Background task to run backtest."""
    try:
        await web_ui_service.run_backtest(
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )

        logger.info(f"Backtest completed for {strategy_name} on {symbol}")

    except Exception as e:
        logger.error(f"Background backtest failed: {e}")


@app.get("/backtests")
async def get_backtest_results(
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get recent backtest results."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        results = await web_ui_service.get_backtest_results(limit=limit)
        return {"backtests": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Backtest results retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtests/{backtest_id}")
async def get_backtest_details(
    backtest_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed backtest results."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        details = await web_ui_service.get_backtest_details(backtest_id)
        return details
    except Exception as e:
        logger.error(f"Backtest details retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trading/status")
async def get_trading_status(current_user: dict = Depends(get_current_user)):
    """Get live trading status."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        status = await web_ui_service.get_trading_status()
        return status
    except Exception as e:
        logger.error(f"Trading status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trading/performance")
async def get_trading_performance(current_user: dict = Depends(get_current_user)):
    """Get live trading performance."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        performance = await web_ui_service.get_trading_performance()
        return performance
    except Exception as e:
        logger.error(f"Trading performance retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/charts/{symbol}")
async def get_chart_data(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """Get chart data for visualization."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        chart_data = await web_ui_service.get_chart_data(symbol, timeframe, limit)
        return chart_data
    except Exception as e:
        logger.error(f"Chart data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/portfolio")
async def get_portfolio_analytics(current_user: dict = Depends(get_current_user)):
    """Get portfolio analytics."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        analytics = await web_ui_service.get_portfolio_analytics()
        return analytics
    except Exception as e:
        logger.error(f"Portfolio analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/risk")
async def get_risk_analytics(current_user: dict = Depends(get_current_user)):
    """Get risk analytics."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        analytics = await web_ui_service.get_risk_analytics()
        return analytics
    except Exception as e:
        logger.error(f"Risk analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/settings")
async def get_settings(current_user: dict = Depends(get_current_user)):
    """Get application settings."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        settings = await web_ui_service.get_settings()
        return settings
    except Exception as e:
        logger.error(f"Settings retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/settings")
async def update_settings(
    settings: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Update application settings."""
    if not web_ui_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await web_ui_service.update_settings(settings)

        logger.info("Settings updated")

        return {
            "success": True,
            "message": "Settings updated successfully"
        }

    except Exception as e:
        logger.error(f"Settings update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Event handlers for logging
async def handle_ui_events(event_data: Dict[str, Any]):
    """Handle UI-related events."""
    logger.info(f"UI event: {event_data}")

async def handle_trading_events(event_data: Dict[str, Any]):
    """Handle trading events for UI updates."""
    logger.info(f"Trading event for UI: {event_data}")

# Subscribe to events
event_system.subscribe("ui.*", handle_ui_events)
event_system.subscribe("trading.*", handle_trading_events)


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info"
    )