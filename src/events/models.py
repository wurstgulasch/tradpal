"""
Event-Driven Architecture for TradPal Trading System

This module defines the core events and event streaming infrastructure
for the trading system. Events represent significant trading occurrences
and enable decoupled, scalable processing of trading signals and decisions.
"""

import json
import asyncio
import redis.asyncio as redis
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


class EventType(Enum):
    """Trading event types"""
    MARKET_DATA_RECEIVED = "market_data_received"
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_CONFIRMED = "signal_confirmed"
    SIGNAL_REJECTED = "signal_rejected"
    TRADE_EXECUTED = "trade_executed"
    TRADE_CANCELLED = "trade_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RISK_ASSESSMENT_COMPLETED = "risk_assessment_completed"
    PORTFOLIO_UPDATED = "portfolio_updated"
    SYSTEM_STATUS_CHANGED = "system_status_changed"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class TradingEvent:
    """Base event class for all trading events"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    symbol: str
    timeframe: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingEvent':
        """Create event from dictionary"""
        data_copy = data.copy()
        data_copy['event_type'] = EventType(data_copy['event_type'])
        data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])
        return cls(**data_copy)

    def to_json(self) -> str:
        """Serialize event to JSON"""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'TradingEvent':
        """Deserialize event from JSON"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class MarketDataEvent(TradingEvent):
    """Event for market data updates"""
    def __init__(self, symbol: str, timeframe: str, ohlcv_data: Dict[str, Any],
                 indicators: Optional[Dict[str, Any]] = None):
        super().__init__(
            event_id=f"market_{symbol}_{timeframe}_{datetime.now().timestamp()}",
            event_type=EventType.MARKET_DATA_RECEIVED,
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            data={
                'ohlcv': ohlcv_data,
                'indicators': indicators or {}
            }
        )


@dataclass
class SignalEvent(TradingEvent):
    """Event for trading signal generation"""
    def __init__(self, symbol: str, timeframe: str, signal_type: str,
                 confidence: float, indicators: Dict[str, Any],
                 reasoning: str, source: str = "traditional"):
        super().__init__(
            event_id=f"signal_{symbol}_{timeframe}_{datetime.now().timestamp()}",
            event_type=EventType.SIGNAL_GENERATED,
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            data={
                'signal_type': signal_type,
                'confidence': confidence,
                'indicators': indicators,
                'reasoning': reasoning,
                'source': source
            }
        )


@dataclass
class TradeEvent(TradingEvent):
    """Event for trade execution"""
    def __init__(self, symbol: str, timeframe: str, side: str, size: float,
                 price: float, order_id: str, reason: str):
        super().__init__(
            event_id=f"trade_{symbol}_{order_id}_{datetime.now().timestamp()}",
            event_type=EventType.TRADE_EXECUTED,
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            data={
                'side': side,
                'size': size,
                'price': price,
                'order_id': order_id,
                'reason': reason
            }
        )


@dataclass
class RiskEvent(TradingEvent):
    """Event for risk assessment"""
    def __init__(self, symbol: str, timeframe: str, assessment_result: str,
                 position_size: float, risk_amount: float, atr_value: float):
        super().__init__(
            event_id=f"risk_{symbol}_{timeframe}_{datetime.now().timestamp()}",
            event_type=EventType.RISK_ASSESSMENT_COMPLETED,
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            data={
                'assessment_result': assessment_result,
                'position_size': position_size,
                'risk_amount': risk_amount,
                'atr_value': atr_value
            }
        )


class EventBus:
    """Central event bus for publishing and subscribing to trading events"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.subscribers: Dict[EventType, List[callable]] = {}

    async def connect(self):
        """Connect to Redis"""
        self.redis = redis.from_url(self.redis_url)

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()

    async def publish_event(self, event: TradingEvent):
        """Publish event to Redis stream"""
        if not self.redis:
            await self.connect()

        try:
            # Publish to Redis stream
            stream_data = event.to_dict()
            await self.redis.xadd(f"trading_events:{event.symbol}", stream_data)

            # Also publish to event type specific stream
            await self.redis.xadd(f"events:{event.event_type.value}", stream_data)

            # Notify local subscribers
            if event.event_type in self.subscribers:
                for callback in self.subscribers[event.event_type]:
                    try:
                        await callback(event)
                    except Exception as e:
                        print(f"Error in event subscriber: {e}")

        except Exception as e:
            print(f"Failed to publish event: {e}")

    def subscribe(self, event_type: EventType, callback: callable):
        """Subscribe to event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    async def consume_events(self, symbol: str, last_id: str = '0'):
        """Consume events from Redis stream"""
        if not self.redis:
            await self.connect()

        try:
            # Read from symbol-specific stream
            streams = await self.redis.xread({f"trading_events:{symbol}": last_id}, block=1000)

            events = []
            for stream_name, messages in streams:
                for message_id, message_data in messages:
                    try:
                        event = TradingEvent.from_dict(message_data)
                        events.append((message_id, event))
                    except Exception as e:
                        print(f"Failed to parse event: {e}")

            return events

        except Exception as e:
            print(f"Failed to consume events: {e}")
            return []


class EventStore:
    """Event sourcing store for audit trails and replay capabilities"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None

    async def connect(self):
        """Connect to Redis"""
        self.redis = redis.from_url(self.redis_url)

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()

    async def store_event(self, event: TradingEvent):
        """Store event for audit trail"""
        if not self.redis:
            await self.connect()

        try:
            # Store in time-series key for the symbol
            event_key = f"event_store:{event.symbol}:{event.timestamp.date().isoformat()}"
            await self.redis.rpush(event_key, event.to_json())

            # Also store by event type
            type_key = f"event_store:type:{event.event_type.value}:{event.timestamp.date().isoformat()}"
            await self.redis.rpush(type_key, event.to_json())

            # Set expiration (keep events for 90 days)
            await self.redis.expire(event_key, 90 * 24 * 60 * 60)
            await self.redis.expire(type_key, 90 * 24 * 60 * 60)

        except Exception as e:
            print(f"Failed to store event: {e}")

    async def get_events(self, symbol: str, date: str, event_type: Optional[EventType] = None) -> List[TradingEvent]:
        """Retrieve events for audit/analysis"""
        if not self.redis:
            await self.connect()

        try:
            if event_type:
                key = f"event_store:type:{event_type.value}:{date}"
            else:
                key = f"event_store:{symbol}:{date}"

            event_jsons = await self.redis.lrange(key, 0, -1)
            events = []

            for event_json in event_jsons:
                try:
                    event = TradingEvent.from_json(event_json)
                    events.append(event)
                except Exception as e:
                    print(f"Failed to parse stored event: {e}")

            return events

        except Exception as e:
            print(f"Failed to retrieve events: {e}")
            return []


# Global instances
event_bus = EventBus()
event_store = EventStore()


async def initialize_event_system():
    """Initialize the event system"""
    await event_bus.connect()
    await event_store.connect()
    print("🎯 Event-driven architecture initialized")


async def shutdown_event_system():
    """Shutdown the event system"""
    await event_bus.disconnect()
    await event_store.disconnect()
    print("🎯 Event-driven architecture shutdown")</content>
<parameter name="filePath">/Users/danielsadowski/VSCodeProjects/tradpal_indicator/src/events/__init__.py