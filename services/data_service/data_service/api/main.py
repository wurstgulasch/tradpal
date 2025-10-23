"""
TradPal Data Service API
REST API for data fetching and management
"""

import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Import unified data service
from ..service import DataService, DataRequest, DataResponse, DataInfoResponse

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="TradPal Data Service API",
    description="REST API for market data fetching and management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data service instance
data_service = DataService()

@app.on_event("startup")
async def startup_event():
    """Initialize data service on startup"""
    await data_service.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown data service on shutdown"""
    await data_service.shutdown()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "TradPal Data Service API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health = await data_service.health_check()
        return health
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "service": "data_service",
            "error": str(e)
        }

@app.get("/info")
async def get_service_info():
    """Get service information"""
    try:
        info = await data_service.get_service_info()
        return info
    except Exception as e:
        logger.error(f"Service info retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get service info: {str(e)}")

@app.get("/data/{symbol}")
async def get_market_data(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe (e.g., 1m, 1h, 1d)"),
    limit: int = Query(100, description="Number of data points", ge=1, le=1000),
    source: str = Query("ccxt", description="Data source")
) -> Dict[str, Any]:
    """
    Fetch market data for a symbol

    Returns OHLCV data as JSON
    """
    try:
        # Ensure data service is initialized
        if not data_service.is_initialized:
            await data_service.initialize()

        data = await data_service.fetch_data(symbol, timeframe, limit, source)

        # Convert to dict for JSON response
        response_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": source,
            "data": data.reset_index().to_dict('records')
        }

        return response_data

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")

@app.get("/data/{symbol}/info")
async def get_data_info(symbol: str) -> Dict[str, Any]:
    """Get information about available data for a symbol"""
    try:
        # For now, return basic info
        info = DataInfoResponse(symbol=symbol)
        return {
            "symbol": info.symbol,
            "available": info.available,
            "quality_score": info.quality_score
        }

    except Exception as e:
        logger.error(f"Error getting data info for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data info: {str(e)}")

@app.get("/sources")
async def get_available_sources():
    """Get list of available data sources"""
    try:
        sources = data_service.get_available_sources()
        return {"sources": sources}

    except Exception as e:
        logger.error(f"Error getting available sources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sources: {str(e)}")

@app.post("/data/validate")
async def validate_data(data: Dict[str, Any]):
    """Validate data quality"""
    try:
        # Convert dict to DataFrame
        df = pd.DataFrame(data["data"]) if "data" in data else pd.DataFrame()

        quality_metrics = await data_service.validate_data_quality(df)
        return quality_metrics

    except Exception as e:
        logger.error(f"Error validating data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate data: {str(e)}")

# Alternative Data Endpoints

@app.get("/sentiment/{symbol}")
async def analyze_sentiment(
    symbol: str,
    hours: int = Query(24, description="Hours of historical data to analyze", ge=1, le=168)
):
    """Analyze sentiment for a symbol"""
    try:
        result = await data_service.analyze_sentiment(symbol, hours)
        if not result["success"]:
            raise HTTPException(status_code=503, detail=result["error"])
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@app.get("/fear-greed")
async def get_fear_greed_index():
    """Get current Fear & Greed Index"""
    try:
        result = await data_service.get_fear_greed_index()
        if not result["success"]:
            raise HTTPException(status_code=503, detail=result["error"])
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching Fear & Greed Index: {e}")
        raise HTTPException(status_code=500, detail=f"Fear & Greed Index fetch failed: {str(e)}")

@app.get("/onchain/{symbol}")
async def get_onchain_metrics(
    symbol: str,
    metrics: Optional[str] = Query(None, description="Comma-separated list of specific metrics")
):
    """Get on-chain metrics for a symbol"""
    try:
        metrics_list = [m.strip() for m in metrics.split(",")] if metrics else None
        result = await data_service.get_onchain_metrics(symbol, metrics_list)
        if not result["success"]:
            raise HTTPException(status_code=503, detail=result["error"])
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching on-chain metrics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"On-chain metrics fetch failed: {str(e)}")

@app.get("/economic")
async def get_economic_indicators(
    indicators: Optional[str] = Query(None, description="Comma-separated list of specific indicators")
):
    """Get economic indicators"""
    try:
        indicators_list = [i.strip() for i in indicators.split(",")] if indicators else None
        result = await data_service.get_economic_indicators(indicators_list)
        if not result["success"]:
            raise HTTPException(status_code=503, detail=result["error"])
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching economic indicators: {e}")
        raise HTTPException(status_code=500, detail=f"Economic indicators fetch failed: {str(e)}")

@app.get("/alternative-data/status")
async def get_alternative_data_status():
    """Get status of alternative data components"""
    try:
        status = await data_service.get_alternative_data_status()
        return status

    except Exception as e:
        logger.error(f"Error getting alternative data status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alternative data status: {str(e)}")

@app.get("/alternative-data/{symbol}")
async def collect_alternative_data(symbol: str):
    """Collect complete alternative data packet for a symbol"""
    try:
        result = await data_service.collect_alternative_data(symbol)
        if not result["success"]:
            raise HTTPException(status_code=503, detail=result["error"])
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error collecting alternative data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Alternative data collection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
