#!/usr/bin/env python3
"""
TradPal Event System

Event-Driven Architecture using Redis Streams for service communication.
Provides publish-subscribe pattern with event persistence and replay capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import redis.asyncio as redis
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for the trading system"""
    MARKET_DATA_UPDATE = "market_data_update"
    TRADING_SIGNAL = "trading_signal"
    ORDER_EXECUTED = "order_executed"
    PORTFOLIO_UPDATE = "portfolio_update"
    RISK_ALERT = "risk_alert"
    SYSTEM_HEALTH = "system_health"
    ML_MODEL_UPDATE = "ml_model_update"
    BACKTEST_COMPLETE = "backtest_complete"

    # Advanced Features Events
    ALTERNATIVE_DATA_UPDATE = "alternative_data_update"
    SENTIMENT_DATA_UPDATE = "sentiment_data_update"
    ONCHAIN_DATA_UPDATE = "onchain_data_update"
    ECONOMIC_DATA_UPDATE = "economic_data_update"
    MARKET_REGIME_CHANGE = "market_regime_change"
    RL_MODEL_UPDATE = "rl_model_update"
    RL_ACTION_TAKEN = "rl_action_taken"
    FEATURE_VECTOR_UPDATE = "feature_vector_update"

    # Backtesting Events
    BACKTEST_REQUEST = "backtest_request"
    BACKTEST_COMPLETED = "backtest_completed"
    BACKTEST_FAILED = "backtest_failed"
    MULTI_SYMBOL_BACKTEST_REQUEST = "multi_symbol_backtest_request"
    MULTI_SYMBOL_BACKTEST_COMPLETED = "multi_symbol_backtest_completed"
    MULTI_SYMBOL_BACKTEST_FAILED = "multi_symbol_backtest_failed"
    MULTI_MODEL_BACKTEST_REQUEST = "multi_model_backtest_request"
    MULTI_MODEL_BACKTEST_COMPLETED = "multi_model_backtest_completed"
    MULTI_MODEL_BACKTEST_FAILED = "multi_model_backtest_failed"
    WALK_FORWARD_BACKTEST_REQUEST = "walk_forward_backtest_request"
    WALK_FORWARD_BACKTEST_COMPLETED = "walk_forward_backtest_completed"
    WALK_FORWARD_BACKTEST_FAILED = "walk_forward_backtest_failed"
    STRATEGY_OPTIMIZATION_REQUEST = "strategy_optimization_request"
    STRATEGY_OPTIMIZATION_COMPLETED = "strategy_optimization_completed"
    STRATEGY_OPTIMIZATION_FAILED = "strategy_optimization_failed"
    BACKTESTING_WORKER_HEARTBEAT = "backtesting_worker_heartbeat"


@dataclass
class Event:
    """Represents an event in the system"""
    event_type: EventType
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "event_type": self.event_type.value,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        return cls(
            event_type=EventType(data["event_type"]),
            event_id=data["event_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            data=data["data"],
            metadata=data["metadata"]
        )


class EventPublisher:
    """Publishes events to Redis streams"""

    def __init__(self, redis_client: redis.Redis, stream_name: str = "tradpal_events"):
        self.redis = redis_client
        self.stream_name = stream_name

    async def publish(self, event: Event) -> str:
        """Publish an event to the stream"""
        event_dict = event.to_dict()

        # Add event to stream
        message_id = await self.redis.xadd(
            self.stream_name,
            {"event": json.dumps(event_dict)}
        )

        logger.info(f"Published event {event.event_type.value} with ID {message_id}")
        return message_id

    async def publish_batch(self, events: List[Event]) -> List[str]:
        """Publish multiple events in batch"""
        message_ids = []
        for event in events:
            message_id = await self.publish(event)
            message_ids.append(message_id)
        return message_ids


class EventSubscriber:
    """Subscribes to events from Redis streams"""

    def __init__(self, redis_client: redis.Redis, stream_name: str = "tradpal_events"):
        self.redis = redis_client
        self.stream_name = stream_name
        self.consumer_group = "tradpal_consumers"
        self.consumer_name = f"consumer_{uuid.uuid4().hex[:8]}"
        self.handlers: Dict[EventType, List[Callable[[Event], Awaitable[None]]]] = {}

    async def initialize(self):
        """Initialize consumer group"""
        try:
            await self.redis.xgroup_create(
                self.stream_name,
                self.consumer_group,
                "$",
                mkstream=True
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def register_handler(self, event_type: EventType, handler: Callable[[Event], Awaitable[None]]):
        """Register an event handler"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")

    async def process_events(self):
        """Process events from the stream"""
        while True:
            try:
                # Read events from consumer group
                messages = await self.redis.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.stream_name: ">"},
                    count=10,
                    block=1000
                )

                for stream_name, message_list in messages:
                    for message_id, message_data in message_list:
                        try:
                            # Parse event
                            event_data = json.loads(message_data["event"])
                            event = Event.from_dict(event_data)

                            # Process event
                            await self._handle_event(event)

                            # Acknowledge message
                            await self.redis.xack(
                                self.stream_name,
                                self.consumer_group,
                                message_id
                            )

                        except Exception as e:
                            logger.error(f"Error processing message {message_id}: {e}")
                            # Don't acknowledge failed messages, they'll be retried

            except Exception as e:
                logger.error(f"Error reading from stream: {e}")
                await asyncio.sleep(1)

    async def _handle_event(self, event: Event):
        """Handle a single event"""
        if event.event_type in self.handlers:
            for handler in self.handlers[event.event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event.event_type.value}: {e}")


class EventStore:
    """Provides event persistence and replay capabilities"""

    def __init__(self, redis_client: redis.Redis, stream_name: str = "tradpal_events"):
        self.redis = redis_client
        self.stream_name = stream_name

    async def get_events(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Event]:
        """Retrieve events with optional filtering"""
        # Build Redis stream query
        start_id = "0" if not start_time else self._datetime_to_stream_id(start_time)
        end_id = "+" if not end_time else self._datetime_to_stream_id(end_time)

        # Read events from stream
        messages = await self.redis.xrange(self.stream_name, start_id, end_id, count=limit)

        events = []
        for message_id, message_data in messages:
            try:
                event_data = json.loads(message_data["event"])
                event = Event.from_dict(event_data)

                # Filter by event type if specified
                if event_type is None or event.event_type == event_type:
                    events.append(event)
            except Exception as e:
                logger.warning(f"Error parsing event from message {message_id}: {e}")

        return events

    async def replay_events(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        handler: Callable[[Event], Awaitable[None]] = None
    ):
        """Replay events through a handler"""
        events = await self.get_events(event_type, start_time, end_time)

        for event in events:
            if handler:
                await handler(event)
            else:
                logger.info(f"Replayed event: {event.event_type.value} at {event.timestamp}")

    def _datetime_to_stream_id(self, dt: datetime) -> str:
        """Convert datetime to Redis stream ID format"""
        # Redis stream IDs are in format: <millisecondsTime>-<sequenceNumber>
        # For simplicity, we'll use milliseconds since epoch
        return str(int(dt.timestamp() * 1000))


class EventSystem:
    """Main event system coordinator"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.publisher: Optional[EventPublisher] = None
        self.subscriber: Optional[EventSubscriber] = None
        self.store: Optional[EventStore] = None

    async def initialize(self):
        """Initialize the event system"""
        self.redis = redis.from_url(self.redis_url)
        self.publisher = EventPublisher(self.redis)
        self.subscriber = EventSubscriber(self.redis)
        self.store = EventStore(self.redis)

        await self.subscriber.initialize()
        logger.info("Event system initialized")

    async def close(self):
        """Close the event system"""
        if self.redis:
            await self.redis.close()

    async def publish_event(self, event: Event) -> str:
        """Publish a single event"""
        return await self.publisher.publish(event)

    def register_handler(self, event_type: EventType, handler: Callable[[Event], Awaitable[None]]):
        """Register an event handler"""
        self.subscriber.register_handler(event_type, handler)

    async def start_consuming(self):
        """Start consuming events"""
        await self.subscriber.process_events()

    async def replay_events(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        """Replay historical events"""
        await self.store.replay_events(event_type, start_time, end_time)


# Global event system instance
event_system: Optional[EventSystem] = None


async def get_event_system() -> EventSystem:
    """Get or create the global event system instance"""
    global event_system
    if event_system is None:
        event_system = EventSystem()
        await event_system.initialize()
    return event_system


# Convenience functions for common event publishing
async def publish_market_data(symbol: str, data: Dict[str, Any], source: str = "data_service"):
    """Publish market data update event"""
    event_system = await get_event_system()
    event = Event(
        event_type=EventType.MARKET_DATA_UPDATE,
        source=source,
        data={"symbol": symbol, **data}
    )
    return await event_system.publish_event(event)


async def publish_trading_signal(signal: Dict[str, Any], source: str = "trading_bot"):
    """Publish trading signal event"""
    event_system = await get_event_system()
    event = Event(
        event_type=EventType.TRADING_SIGNAL,
        source=source,
        data=signal
    )
    return await event_system.publish_event(event)


async def publish_portfolio_update(updates: Dict[str, Any], source: str = "risk_service"):
    """Publish portfolio update event"""
    event_system = await get_event_system()
    event = Event(
        event_type=EventType.PORTFOLIO_UPDATE,
        source=source,
        data=updates
    )
    return await event_system.publish_event(event)


async def publish_alternative_data(data_packet: Dict[str, Any], source: str = "alternative_data_service"):
    """Publish alternative data update event"""
    event_system = await get_event_system()
    event = Event(
        event_type=EventType.ALTERNATIVE_DATA_UPDATE,
        source=source,
        data=data_packet
    )
    return await event_system.publish_event(event)


async def publish_sentiment_data(sentiment_data: Dict[str, Any], source: str = "alternative_data_service"):
    """Publish sentiment data update event"""
    event_system = await get_event_system()
    event = Event(
        event_type=EventType.SENTIMENT_DATA_UPDATE,
        source=source,
        data=sentiment_data
    )
    return await event_system.publish_event(event)


async def publish_onchain_data(onchain_data: Dict[str, Any], source: str = "alternative_data_service"):
    """Publish on-chain data update event"""
    event_system = await get_event_system()
    event = Event(
        event_type=EventType.ONCHAIN_DATA_UPDATE,
        source=source,
        data=onchain_data
    )
    return await event_system.publish_event(event)


async def publish_economic_data(economic_data: Dict[str, Any], source: str = "alternative_data_service"):
    """Publish economic data update event"""
    event_system = await get_event_system()
    event = Event(
        event_type=EventType.ECONOMIC_DATA_UPDATE,
        source=source,
        data=economic_data
    )
    return await event_system.publish_event(event)


async def publish_market_regime_change(regime_data: Dict[str, Any], source: str = "market_regime_service"):
    """Publish market regime change event"""
    event_system = await get_event_system()
    event = Event(
        event_type=EventType.MARKET_REGIME_CHANGE,
        source=source,
        data=regime_data
    )
    return await event_system.publish_event(event)


async def publish_regime_signal(signal_data: Dict[str, Any], source: str = "market_regime_service"):
    """Publish regime-based trading signal event"""
    event_system = await get_event_system()
    event = Event(
        event_type=EventType.TRADING_SIGNAL,
        source=source,
        data={
            **signal_data,
            "signal_type": "regime_based",
            "regime_signal": True
        }
    )
    return await event_system.publish_event(event)


async def publish_regime_transition(transition_data: Dict[str, Any], source: str = "market_regime_service"):
    """Publish regime transition event"""
    event_system = await get_event_system()
    event = Event(
        event_type=EventType.MARKET_REGIME_CHANGE,
        source=source,
        data={
            **transition_data,
            "event_type": "regime_transition"
        }
    )
    return await event_system.publish_event(event)


async def publish_rl_model_update(model_data: Dict[str, Any], source: str = "rl_service"):
    """Publish RL model update event"""
    event_system = await get_event_system()
    event = Event(
        event_type=EventType.RL_MODEL_UPDATE,
        source=source,
        data=model_data
    )
    return await event_system.publish_event(event)


async def publish_rl_action(action_data: Dict[str, Any], source: str = "rl_service"):
    """Publish RL action taken event"""
    event_system = await get_event_system()
    event = Event(
        event_type=EventType.RL_ACTION_TAKEN,
        source=source,
        data=action_data
    )
    return await event_system.publish_event(event)


async def publish_feature_vector(features: Dict[str, Any], source: str = "feature_processor"):
    """Publish feature vector update event"""
    event_system = await get_event_system()
    event = Event(
        event_type=EventType.FEATURE_VECTOR_UPDATE,
        source=source,
        data=features
    )
    return await event_system.publish_event(event)