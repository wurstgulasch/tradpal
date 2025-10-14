#!/usr/bin/env python3
"""
Data Service API - REST API for the Data Service.

Provides endpoints for:
- Data fetching with caching and validation
- Data quality monitoring
- Cache management
- Health checks
"""

import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .service import DataService, DataRequest, DataResponse, EventSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
data_service: Optional[DataService] = None
event_system = EventSystem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global data_service

    # Startup
    logger.info("Starting Data Service API")
    data_service = DataService(event_system=event_system)

    yield

    # Shutdown
    logger.info("Shutting down Data Service API")


# Create FastAPI app
app = FastAPI(
    title="Data Service API",
    description="Centralized time-series data management for TradPal",
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


class FetchDataRequest(BaseModel):
    """Request model for data fetching."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe (e.g., '1d', '1h', '15m')")
    start_date: str = Field(..., description="Start date in ISO format")
    end_date: str = Field(..., description="End date in ISO format")
    source: Optional[str] = Field("ccxt", description="Preferred data source")
    provider: Optional[str] = Field("binance", description="Preferred data provider")
    use_cache: bool = Field(True, description="Whether to use cached data")
    validate_quality: bool = Field(True, description="Whether to validate data quality")


class DataInfoResponse(BaseModel):
    """Response model for data info."""
    symbol: str
    timeframe: str
    available_sources: list
    cache_enabled: bool
    quality_thresholds: Dict[str, float]


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Data Service",
        "version": "1.0.0",
        "description": "Centralized time-series data management"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    health_data = await data_service.health_check()
    status_code = 200 if health_data["status"] == "healthy" else 503

    return health_data


@app.post("/data/fetch", response_model=DataResponse)
async def fetch_data(request: FetchDataRequest, background_tasks: BackgroundTasks):
    """
    Fetch time-series data.

    Fetches OHLCV data from various sources with automatic fallbacks,
    caching, and quality validation.
    """
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert to service request
        service_request = DataRequest(**request.dict())

        # Fetch data
        response = await data_service.fetch_data(service_request)

        # Log the request
        logger.info(f"Data fetch request: {request.symbol} {request.timeframe} "
                   f"({request.start_date} to {request.end_date}) - "
                   f"Success: {response.success}, Cache hit: {response.cache_hit}")

        if not response.success:
            raise HTTPException(status_code=400, detail=response.error)

        return response

    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/info", response_model=DataInfoResponse)
async def get_data_info(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe")
):
    """Get information about available data for a symbol/timeframe."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        info = await data_service.get_data_info(symbol, timeframe)
        return DataInfoResponse(**info)
    except Exception as e:
        logger.error(f"Data info request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
async def clear_cache(pattern: str = Query("*", description="Cache key pattern to clear")):
    """Clear cache entries matching the pattern."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        deleted_count = await data_service.clear_cache(pattern)
        return {
            "message": f"Cleared {deleted_count} cache entries",
            "pattern": pattern
        }
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    if not data_service or not data_service.redis_client:
        return {"cache_enabled": False, "stats": {}}

    try:
        # Get Redis info
        info = data_service.redis_client.info()
        keys = data_service.redis_client.keys("data:*")

        return {
            "cache_enabled": True,
            "total_keys": len(keys) if keys else 0,
            "redis_info": {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "uptime_in_days": info.get("uptime_in_days", 0)
            }
        }
    except Exception as e:
        logger.error(f"Cache stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/validate")
async def validate_data_quality(request: FetchDataRequest):
    """
    Validate data quality without storing results.

    Useful for checking data quality before using it in analysis.
    """
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert to service request
        service_request = DataRequest(**request.dict())

        # Fetch data
        response = await data_service.fetch_data(service_request)

        if not response.success:
            return {
                "valid": False,
                "error": response.error,
                "quality_score": 0.0,
                "quality_level": "invalid"
            }

        # Extract quality info from metadata
        metadata = response.metadata
        if metadata:
            return {
                "valid": True,
                "quality_score": metadata.get("quality_score", 0.0),
                "quality_level": metadata.get("quality_level", "unknown"),
                "record_count": metadata.get("record_count", 0),
                "fallback_used": metadata.get("fallback_used", False)
            }

        return {
            "valid": False,
            "error": "No metadata available",
            "quality_score": 0.0,
            "quality_level": "unknown"
        }

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Event handlers for logging
async def handle_data_events(event_data: Dict[str, Any]):
    """Handle data service events."""
    event_type = asyncio.current_task().get_name() if asyncio.current_task() else "unknown"

    if "data.fetch_success" in str(event_type):
        logger.info(f"Data fetch success: {event_data}")
    elif "data.fetch_failed" in str(event_type):
        logger.warning(f"Data fetch failed: {event_data}")
    elif "data.cache_hit" in str(event_type):
        logger.info(f"Cache hit: {event_data}")

# Subscribe to events
event_system.subscribe("data.fetch_success", handle_data_events)
event_system.subscribe("data.fetch_failed", handle_data_events)
event_system.subscribe("data.cache_hit", handle_data_events)


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )