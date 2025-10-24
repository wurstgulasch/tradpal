#!/usr/bin/env python3
"""
Alternative Data Service - Main Service Implementation

Provides REST API for alternative data collection and processing.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .sentiment_analyzer import SentimentAnalyzer
from .onchain_collector import OnChainDataCollector
from .economic_collector import EconomicDataCollector
from .data_processor import AlternativeDataProcessor
from .__init__ import AlternativeDataPacket, ProcessedFeatures

# Event system imports
try:
    from services.event_system import (
        publish_alternative_data,
        publish_sentiment_data,
        publish_onchain_data,
        publish_economic_data,
        publish_feature_vector
    )
    EVENT_SYSTEM_AVAILABLE = True
except ImportError:
    EVENT_SYSTEM_AVAILABLE = False
    logger.warning("Event system not available, events will not be published")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
sentiment_analyzer = None
onchain_collector = None
economic_collector = None
data_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global sentiment_analyzer, onchain_collector, economic_collector, data_processor

    # Initialize services
    logger.info("Initializing Alternative Data Service...")
    sentiment_analyzer = SentimentAnalyzer()
    onchain_collector = OnChainDataCollector()
    economic_collector = EconomicDataCollector()
    data_processor = AlternativeDataProcessor()

    # Start background tasks
    asyncio.create_task(initialize_data_collection())

    yield

    # Cleanup
    logger.info("Shutting down Alternative Data Service...")


async def initialize_data_collection():
    """Initialize background data collection."""
    try:
        # Warm up data sources
        await sentiment_analyzer.initialize()
        await onchain_collector.initialize()
        await economic_collector.initialize()

        logger.info("Alternative Data Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize data collection: {e}")


# FastAPI app
app = FastAPI(
    title="TradPal Alternative Data Service",
    description="Advanced alternative data collection and processing for AI trading",
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


# Pydantic models for API
class SentimentRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., BTC/USDT)")
    hours: int = Field(24, description="Hours of historical data to analyze")


class OnChainRequest(BaseModel):
    symbol: str = Field(..., description="Crypto symbol (e.g., BTC)")
    metrics: Optional[List[str]] = Field(None, description="Specific metrics to fetch")


class EconomicRequest(BaseModel):
    indicators: Optional[List[str]] = Field(None, description="Specific indicators to fetch")


class DataProcessingRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    include_sentiment: bool = Field(True, description="Include sentiment features")
    include_onchain: bool = Field(True, description="Include on-chain features")
    include_economic: bool = Field(True, description="Include economic features")


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "alternative_data_service",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/sentiment/analyze")
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment for a symbol."""
    try:
        result = await sentiment_analyzer.analyze_symbol_sentiment(
            symbol=request.symbol,
            hours=request.hours
        )
        return result
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


@app.get("/sentiment/fear-greed")
async def get_fear_greed_index():
    """Get current Fear & Greed Index."""
    try:
        result = await sentiment_analyzer.get_fear_greed_index()
        return result
    except Exception as e:
        logger.error(f"Fear & Greed Index fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fear & Greed Index fetch failed: {str(e)}")


@app.post("/onchain/metrics")
async def get_onchain_metrics(request: OnChainRequest):
    """Get on-chain metrics for a symbol."""
    try:
        result = await onchain_collector.get_metrics(
            symbol=request.symbol,
            metrics=request.metrics
        )
        return result
    except Exception as e:
        logger.error(f"On-chain metrics fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"On-chain metrics fetch failed: {str(e)}")


@app.post("/economic/indicators")
async def get_economic_indicators(request: EconomicRequest):
    """Get economic indicators."""
    try:
        result = await economic_collector.get_indicators(
            indicators=request.indicators
        )
        return result
    except Exception as e:
        logger.error(f"Economic indicators fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Economic indicators fetch failed: {str(e)}")


@app.post("/data/collect")
async def collect_alternative_data(symbol: str, background_tasks: BackgroundTasks):
    """Collect complete alternative data packet for a symbol."""
    try:
        # Start background collection
        background_tasks.add_task(collect_data_background, symbol)

        return {
            "status": "collection_started",
            "symbol": symbol,
            "message": "Data collection started in background"
        }
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data collection failed: {str(e)}")


@app.get("/data/{symbol}")
async def get_alternative_data(symbol: str):
    """Get latest alternative data for a symbol."""
    try:
        # Collect data synchronously for immediate response
        sentiment_data = await sentiment_analyzer.analyze_symbol_sentiment(symbol)
        onchain_data = await onchain_collector.get_metrics(symbol)
        economic_data = await economic_collector.get_indicators()
        fear_greed = await sentiment_analyzer.get_fear_greed_index()

        packet = AlternativeDataPacket(
            symbol=symbol,
            sentiment_data=sentiment_data if isinstance(sentiment_data, list) else [sentiment_data],
            onchain_data=onchain_data if isinstance(onchain_data, list) else [onchain_data],
            economic_data=economic_data if isinstance(economic_data, list) else [economic_data],
            fear_greed_index=fear_greed.get('value') if fear_greed else None
        )

        return packet
    except Exception as e:
        logger.error(f"Data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data retrieval failed: {str(e)}")


@app.post("/data/process")
async def process_alternative_data(request: DataProcessingRequest):
    """Process alternative data into ML features."""
    try:
        # Get raw data
        raw_data = await get_alternative_data(request.symbol)

        # Process into features
        features = await data_processor.process_to_features(
            data_packet=raw_data,
            include_sentiment=request.include_sentiment,
            include_onchain=request.include_onchain,
            include_economic=request.include_economic
        )

        return features
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data processing failed: {str(e)}")


@app.get("/metrics")
async def get_service_metrics():
    """Get service performance metrics."""
    try:
        return {
            "sentiment_analyzer": await sentiment_analyzer.get_metrics(),
            "onchain_collector": await onchain_collector.get_metrics(),
            "economic_collector": await economic_collector.get_metrics(),
            "data_processor": await data_processor.get_metrics()
        }
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


async def collect_data_background(symbol: str):
    """Background data collection task."""
    try:
        logger.info(f"Starting background data collection for {symbol}")

        # Collect all data types
        sentiment_task = sentiment_analyzer.analyze_symbol_sentiment(symbol)
        onchain_task = onchain_collector.get_metrics(symbol)
        economic_task = economic_collector.get_indicators()
        fear_greed_task = sentiment_analyzer.get_fear_greed_index()

        results = await asyncio.gather(
            sentiment_task, onchain_task, economic_task, fear_greed_task,
            return_exceptions=True
        )

        # Process results and publish to event system
        sentiment_data, onchain_data, economic_data, fear_greed = results

        # Publish individual data events
        if EVENT_SYSTEM_AVAILABLE:
            try:
                # Publish sentiment data
                if sentiment_data and not isinstance(sentiment_data, Exception):
                    await publish_sentiment_data({
                        "symbol": symbol,
                        "sentiment_data": [s.__dict__ if hasattr(s, '__dict__') else s for s in sentiment_data] if isinstance(sentiment_data, list) else [sentiment_data.__dict__ if hasattr(sentiment_data, '__dict__') else sentiment_data],
                        "timestamp": datetime.utcnow().isoformat()
                    })

                # Publish on-chain data
                if onchain_data and not isinstance(onchain_data, Exception):
                    await publish_onchain_data({
                        "symbol": symbol,
                        "onchain_data": [o.__dict__ if hasattr(o, '__dict__') else o for o in onchain_data] if isinstance(onchain_data, list) else [onchain_data.__dict__ if hasattr(onchain_data, '__dict__') else onchain_data],
                        "timestamp": datetime.utcnow().isoformat()
                    })

                # Publish economic data
                if economic_data and not isinstance(economic_data, Exception):
                    await publish_economic_data({
                        "economic_data": [e.__dict__ if hasattr(e, '__dict__') else e for e in economic_data] if isinstance(economic_data, list) else [economic_data.__dict__ if hasattr(economic_data, '__dict__') else economic_data],
                        "timestamp": datetime.utcnow().isoformat()
                    })

                # Create and publish complete alternative data packet
                if all(not isinstance(r, Exception) for r in results):
                    packet = AlternativeDataPacket(
                        symbol=symbol,
                        sentiment_data=sentiment_data if isinstance(sentiment_data, list) else [sentiment_data],
                        onchain_data=onchain_data if isinstance(onchain_data, list) else [onchain_data],
                        economic_data=economic_data if isinstance(economic_data, list) else [economic_data],
                        fear_greed_index=fear_greed.get('value') if fear_greed else None
                    )

                    # Process into features
                    features = await data_processor.process_to_features(packet)

                    # Publish alternative data packet
                    await publish_alternative_data({
                        "symbol": symbol,
                        "packet": packet.__dict__,
                        "features": features.__dict__,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    # Publish feature vector
                    await publish_feature_vector({
                        "symbol": symbol,
                        "features": features.__dict__,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                logger.info(f"Published alternative data events for {symbol}")

            except Exception as e:
                logger.error(f"Failed to publish events for {symbol}: {e}")
        else:
            logger.debug(f"Event system not available, skipping event publishing for {symbol}")

        logger.info(f"Background data collection completed for {symbol}")

    except Exception as e:
        logger.error(f"Background data collection failed for {symbol}: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8005,  # Alternative Data Service port
        reload=True
    )