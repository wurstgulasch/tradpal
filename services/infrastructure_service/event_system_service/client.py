#!/usr/bin/env python3
"""
TradPal Event Client

Client library for interacting with the Event Service from other microservices.
Provides easy-to-use methods for publishing and subscribing to events.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Awaitable
import aiohttp

from . import Event, EventType

logger = logging.getLogger(__name__)


class EventClient:
    """Client for interacting with the Event Service"""

    def __init__(self, event_service_url: str = "http://event_service:8011"):
        self.event_service_url = event_service_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _ensure_session(self):
        """Ensure we have an active session"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def publish_event(self, event: Event) -> str:
        """Publish an event to the event service"""
        await self._ensure_session()

        url = f"{self.event_service_url}/events"
        event_data = {
            "event_type": event.event_type.value,
            "source": event.source,
            "data": event.data,
            "metadata": event.metadata
        }

        async with self.session.post(url, json=event_data) as response:
            if response.status == 200:
                result = await response.json()
                return result["message_id"]
            else:
                error = await response.text()
                raise Exception(f"Failed to publish event: {error}")

    async def publish_market_data(self, symbol: str, data: Dict[str, Any], source: str = "client") -> str:
        """Publish market data update"""
        await self._ensure_session()

        url = f"{self.event_service_url}/events/market-data"
        payload = {
            "symbol": symbol,
            "market_data": data,
            "source": source
        }

        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result["message_id"]
            else:
                error = await response.text()
                raise Exception(f"Failed to publish market data: {error}")

    async def publish_trading_signal(self, signal: Dict[str, Any], source: str = "client") -> str:
        """Publish trading signal"""
        await self._ensure_session()

        url = f"{self.event_service_url}/events/trading-signal"
        payload = {
            "signal": signal,
            "source": source
        }

        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result["message_id"]
            else:
                error = await response.text()
                raise Exception(f"Failed to publish trading signal: {error}")

    async def publish_portfolio_update(self, updates: Dict[str, Any], source: str = "client") -> str:
        """Publish portfolio update"""
        await self._ensure_session()

        url = f"{self.event_service_url}/events/portfolio-update"
        payload = {
            "updates": updates,
            "source": source
        }

        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result["message_id"]
            else:
                error = await response.text()
                raise Exception(f"Failed to publish portfolio update: {error}")

    async def get_events(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """Retrieve recent events"""
        await self._ensure_session()

        url = f"{self.event_service_url}/events"
        params = {}
        if event_type:
            params["event_type"] = event_type
        if limit:
            params["limit"] = str(limit)

        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                result = await response.json()
                return result["events"]
            else:
                error = await response.text()
                raise Exception(f"Failed to get events: {error}")

    async def get_event_stats(self) -> Dict[str, Any]:
        """Get event stream statistics"""
        await self._ensure_session()

        url = f"{self.event_service_url}/events/stats"

        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.text()
                raise Exception(f"Failed to get stats: {error}")

    async def health_check(self) -> Dict[str, Any]:
        """Check event service health"""
        await self._ensure_session()

        url = f"{self.event_service_url}/health"

        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.text()
                raise Exception(f"Health check failed: {error}")


class EventSubscriberClient:
    """Client for subscribing to events via polling"""

    def __init__(self, event_client: EventClient, poll_interval: float = 5.0):
        self.event_client = event_client
        self.poll_interval = poll_interval
        self.handlers: Dict[EventType, list] = {}
        self.last_event_id: Optional[str] = None
        self.running = False

    def register_handler(self, event_type: EventType, handler: Callable[[Event], Awaitable[None]]):
        """Register an event handler"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")

    async def start_subscribing(self):
        """Start subscribing to events"""
        self.running = True
        logger.info("Started event subscription")

        while self.running:
            try:
                await self._poll_events()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in event subscription: {e}")
                await asyncio.sleep(self.poll_interval)

    async def stop_subscribing(self):
        """Stop subscribing to events"""
        self.running = False
        logger.info("Stopped event subscription")

    async def _poll_events(self):
        """Poll for new events"""
        try:
            # Get recent events (this is a simple implementation)
            # In production, you'd want a more sophisticated approach
            events_data = await self.event_client.get_events(limit=50)

            for event_data in events_data:
                event = Event.from_dict(event_data)

                # Skip if we've already processed this event
                if self.last_event_id and event.event_id <= self.last_event_id:
                    continue

                # Handle event
                await self._handle_event(event)

            # Update last processed event ID
            if events_data:
                self.last_event_id = events_data[-1]["event_id"]

        except Exception as e:
            logger.error(f"Error polling events: {e}")

    async def _handle_event(self, event: Event):
        """Handle a single event"""
        if event.event_type in self.handlers:
            for handler in self.handlers[event.event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event.event_type.value}: {e}")


# Convenience functions
async def create_event_client(service_url: str = "http://event_service:8011") -> EventClient:
    """Create an event client"""
    client = EventClient(service_url)
    await client._ensure_session()
    return client


async def publish_event_async(
    event_type: EventType,
    data: Dict[str, Any],
    source: str = "client",
    service_url: str = "http://event_service:8011"
) -> str:
    """Convenience function to publish an event"""
    async with EventClient(service_url) as client:
        event = Event(event_type=event_type, source=source, data=data)
        return await client.publish_event(event)