#!/usr/bin/env python3
"""
TradPal Event Service

Central event broker service providing:
- Event publishing and subscription
- Event persistence and replay
- Event monitoring and metrics
- REST API for event management
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from . import (
    EventSystem, Event, EventType, EventPublisher, EventSubscriber, EventStore,
    publish_market_data, publish_trading_signal, publish_portfolio_update
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TradPal Event Service",
    description="Central event broker for TradPal microservices",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global event system
event_system: Optional[EventSystem] = None

# Prometheus metrics
EVENTS_PUBLISHED = Counter(
    'event_service_events_published_total',
    'Total number of events published',
    ['event_type']
)

EVENTS_CONSUMED = Counter(
    'event_service_events_consumed_total',
    'Total number of events consumed',
    ['event_type']
)

ACTIVE_SUBSCRIBERS = Gauge(
    'event_service_active_subscribers',
    'Number of active event subscribers'
)

STREAM_LENGTH = Gauge(
    'event_service_stream_length',
    'Current length of the event stream'
)


@app.on_event("startup")
async def startup_event():
    """Initialize event system on startup"""
    global event_system
    event_system = EventSystem()
    await event_system.initialize()

    # Start background tasks
    asyncio.create_task(monitor_stream())
    asyncio.create_task(update_metrics())

    logger.info("Event Service started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global event_system
    if event_system:
        await event_system.close()


async def monitor_stream():
    """Monitor stream length for metrics"""
    while True:
        try:
            if event_system and event_system.redis:
                # Get stream info
                info = await event_system.redis.xinfo_stream("tradpal_events")
                STREAM_LENGTH.set(info["length"])
        except Exception as e:
            logger.warning(f"Error monitoring stream: {e}")

        await asyncio.sleep(30)  # Update every 30 seconds


async def update_metrics():
    """Update Prometheus metrics"""
    while True:
        try:
            # This would be updated based on actual consumption metrics
            # For now, just keep the gauge updated
            ACTIVE_SUBSCRIBERS.set(1)  # Placeholder
        except Exception as e:
            logger.warning(f"Error updating metrics: {e}")

        await asyncio.sleep(60)  # Update every minute


@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "event_service"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


@app.post("/events")
async def publish_event(event_data: Dict[str, Any]):
    """Publish an event"""
    try:
        event_type = EventType(event_data["event_type"])
        event = Event(
            event_type=event_type,
            source=event_data.get("source", "api"),
            data=event_data.get("data", {}),
            metadata=event_data.get("metadata", {})
        )

        message_id = await event_system.publish_event(event)

        # Update metrics
        EVENTS_PUBLISHED.labels(event_type=event_type.value).inc()

        return {
            "message_id": message_id,
            "event_id": event.event_id,
            "status": "published"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {e}")
    except Exception as e:
        logger.error(f"Error publishing event: {e}")
        raise HTTPException(status_code=500, detail="Failed to publish event")


@app.get("/events")
async def get_events(
    event_type: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None)
):
    """Retrieve events with optional filtering"""
    try:
        et = EventType(event_type) if event_type else None
        events = await event_system.store.get_events(et, start_time, end_time, limit)

        return {
            "events": [event.to_dict() for event in events],
            "count": len(events)
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {e}")
    except Exception as e:
        logger.error(f"Error retrieving events: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve events")


@app.post("/events/replay")
async def replay_events(
    event_type: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None)
):
    """Replay historical events"""
    try:
        et = EventType(event_type) if event_type else None

        # For replay, we'll just log the events (in production, you'd send to handlers)
        replayed_count = 0

        async def replay_handler(event: Event):
            nonlocal replayed_count
            logger.info(f"Replaying event: {event.event_type.value} from {event.timestamp}")
            replayed_count += 1

        await event_system.replay_events(et, start_time, end_time)

        return {
            "status": "replay_complete",
            "events_replayed": replayed_count
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {e}")
    except Exception as e:
        logger.error(f"Error replaying events: {e}")
        raise HTTPException(status_code=500, detail="Failed to replay events")


@app.get("/events/types")
async def get_event_types():
    """Get available event types"""
    return {
        "event_types": [et.value for et in EventType]
    }


@app.get("/events/stats")
async def get_event_stats():
    """Get event stream statistics"""
    try:
        if event_system and event_system.redis:
            info = await event_system.redis.xinfo_stream("tradpal_events")
            groups = await event_system.redis.xinfo_groups("tradpal_events")

            return {
                "stream_length": info["length"],
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
                "consumer_groups": len(groups),
                "groups": [
                    {
                        "name": group["name"],
                        "consumers": group["consumers"],
                        "pending": group["pending"],
                        "last-delivered-id": group["last-delivered-id"]
                    }
                    for group in groups
                ]
            }
        else:
            return {"error": "Event system not initialized"}

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


# Convenience endpoints for common events
@app.post("/events/market-data")
async def publish_market_data_event(data: Dict[str, Any]):
    """Publish market data update event"""
    try:
        message_id = await publish_market_data(
            data["symbol"],
            data["market_data"],
            data.get("source", "api")
        )
        return {"message_id": message_id, "status": "published"}
    except Exception as e:
        logger.error(f"Error publishing market data: {e}")
        raise HTTPException(status_code=500, detail="Failed to publish market data")


@app.post("/events/trading-signal")
async def publish_trading_signal_event(data: Dict[str, Any]):
    """Publish trading signal event"""
    try:
        message_id = await publish_trading_signal(
            data["signal"],
            data.get("source", "api")
        )
        return {"message_id": message_id, "status": "published"}
    except Exception as e:
        logger.error(f"Error publishing trading signal: {e}")
        raise HTTPException(status_code=500, detail="Failed to publish trading signal")


@app.post("/events/portfolio-update")
async def publish_portfolio_update_event(data: Dict[str, Any]):
    """Publish portfolio update event"""
    try:
        message_id = await publish_portfolio_update(
            data["updates"],
            data.get("source", "api")
        )
        return {"message_id": message_id, "status": "published"}
    except Exception as e:
        logger.error(f"Error publishing portfolio update: {e}")
        raise HTTPException(status_code=500, detail="Failed to publish portfolio update")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8011)