"""
Event-Driven Architecture for TradPal Trading System

This module defines the core events and event streaming infrastructure
for the trading system. Events represent significant trading occurrences
and enable decoupled, scalable processing of trading signals and decisions.
"""

import json
import asyncio
import redis.asyncio as redis
import pandas as pd
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
    print("🎯 Event-driven architecture shutdown")


# Integration functions for existing signal_generator.py
def create_market_data_event(symbol: str, timeframe: str, df: pd.DataFrame) -> Optional[MarketDataEvent]:
    """Create market data event from DataFrame"""
    if df.empty:
        return None

    # Get latest row
    latest = df.iloc[-1]

    # Extract OHLCV data
    ohlcv_data = {
        'open': latest.get('open', 0),
        'high': latest.get('high', 0),
        'low': latest.get('low', 0),
        'close': latest.get('close', 0),
        'volume': latest.get('volume', 0),
        'timestamp': latest.name.isoformat() if hasattr(latest, 'name') else datetime.now().isoformat()
    }

    # Extract indicators
    indicators = {}
    indicator_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    for col in indicator_columns:
        if pd.notna(latest.get(col)):
            indicators[col] = float(latest[col])

    # Create event
    return MarketDataEvent(symbol, timeframe, ohlcv_data, indicators)


def create_signal_event(symbol: str, timeframe: str, signal_type: str,
                       confidence: float, indicators: Dict[str, Any],
                       reasoning: str, source: str = "traditional") -> SignalEvent:
    """Create signal generation event"""
    return SignalEvent(symbol, timeframe, signal_type, confidence, indicators, reasoning, source)


def create_risk_event(symbol: str, timeframe: str, assessment_result: str,
                     position_size: float, risk_amount: float, atr_value: float) -> RiskEvent:
    """Create risk assessment event"""
    return RiskEvent(symbol, timeframe, assessment_result, position_size, risk_amount, atr_value)


def create_trade_event(symbol: str, timeframe: str, side: str, size: float,
                      price: float, order_id: str, reason: str) -> TradeEvent:
    """Create trade execution event"""
    return TradeEvent(symbol, timeframe, side, size, price, order_id, reason)


# Event publishing should be done in async context by the caller


class EventHandler:
    """Base class for event handlers"""

    async def handle_event(self, event: TradingEvent):
        """Handle a trading event"""
        raise NotImplementedError


class MarketDataHandler(EventHandler):
    """Handles market data events"""

    def __init__(self):
        self.latest_data: Dict[str, Dict[str, Any]] = {}

    async def handle_event(self, event: TradingEvent):
        if event.event_type != EventType.MARKET_DATA_RECEIVED:
            return

        # Store latest market data
        symbol_key = f"{event.symbol}_{event.timeframe}"
        self.latest_data[symbol_key] = {
            'data': event.data,
            'timestamp': event.timestamp
        }

        # Log market data receipt (placeholder - integrate with audit_logger later)
        print(f"Market data received for {event.symbol} {event.timeframe}")


class SignalHandler(EventHandler):
    """Handles signal generation events"""

    def __init__(self):
        self.signal_history: Dict[str, list] = {}

    async def handle_event(self, event: TradingEvent):
        if event.event_type != EventType.SIGNAL_GENERATED:
            return

        # Store signal in history
        symbol_key = f"{event.symbol}_{event.timeframe}"
        if symbol_key not in self.signal_history:
            self.signal_history[symbol_key] = []

        self.signal_history[symbol_key].append({
            'timestamp': event.timestamp,
            'signal_type': event.data['signal_type'],
            'confidence': event.data['confidence'],
            'source': event.data.get('source', 'traditional')
        })

        # Keep only last 100 signals per symbol
        if len(self.signal_history[symbol_key]) > 100:
            self.signal_history[symbol_key] = self.signal_history[symbol_key][-100:]

        # Log signal generation (placeholder)
        print(f"Signal generated: {event.data['signal_type']} for {event.symbol} {event.timeframe}")


class RiskHandler(EventHandler):
    """Handles risk assessment events"""

    def __init__(self):
        self.risk_assessments: Dict[str, list] = {}

    async def handle_event(self, event: TradingEvent):
        if event.event_type != EventType.RISK_ASSESSMENT_COMPLETED:
            return

        # Store risk assessment
        symbol_key = f"{event.symbol}_{event.timeframe}"
        if symbol_key not in self.risk_assessments:
            self.risk_assessments[symbol_key] = []

        self.risk_assessments[symbol_key].append({
            'timestamp': event.timestamp,
            'result': event.data['assessment_result'],
            'position_size': event.data['position_size'],
            'risk_amount': event.data['risk_amount']
        })

        # Keep only last 50 assessments per symbol
        if len(self.risk_assessments[symbol_key]) > 50:
            self.risk_assessments[symbol_key] = self.risk_assessments[symbol_key][-50:]

        # Alert on risk issues
        if event.data['assessment_result'] in ['REJECTED', 'MODIFIED']:
            print(f"Risk assessment issue: {event.data['assessment_result']} for {event.symbol}")


class TradeHandler(EventHandler):
    """Handles trade execution events"""

    def __init__(self):
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: Dict[str, list] = {}

    async def handle_event(self, event: TradingEvent):
        if event.event_type not in [EventType.TRADE_EXECUTED, EventType.POSITION_OPENED, EventType.POSITION_CLOSED]:
            return

        symbol = event.symbol

        if event.event_type == EventType.TRADE_EXECUTED:
            # Record trade
            if symbol not in self.trade_history:
                self.trade_history[symbol] = []

            self.trade_history[symbol].append({
                'timestamp': event.timestamp,
                'side': event.data['side'],
                'size': event.data['size'],
                'price': event.data['price'],
                'order_id': event.data['order_id']
            })

            # Update position
            if event.data['side'] == 'BUY':
                if symbol not in self.active_positions:
                    self.active_positions[symbol] = {
                        'size': 0,
                        'avg_price': 0,
                        'trades': []
                    }
                # Update position (simplified)
                self.active_positions[symbol]['size'] += event.data['size']
                self.active_positions[symbol]['trades'].append(event.data)

            elif event.data['side'] == 'SELL':
                if symbol in self.active_positions:
                    self.active_positions[symbol]['size'] -= event.data['size']
                    self.active_positions[symbol]['trades'].append(event.data)

                    # Close position if size <= 0
                    if self.active_positions[symbol]['size'] <= 0:
                        # Create position closed event
                        closed_event = TradingEvent(
                            event_id=f"position_closed_{symbol}_{datetime.now().timestamp()}",
                            event_type=EventType.POSITION_CLOSED,
                            timestamp=datetime.now(),
                            symbol=symbol,
                            timeframe=event.timeframe,
                            data={'final_size': self.active_positions[symbol]['size']}
                        )
                        await event_bus.publish_event(closed_event)
                        del self.active_positions[symbol]

        elif event.event_type == EventType.POSITION_OPENED:
            # Handle position opened
            self.active_positions[symbol] = event.data

        elif event.event_type == EventType.POSITION_CLOSED:
            # Handle position closed
            if symbol in self.active_positions:
                del self.active_positions[symbol]


class EventProcessor:
    """Main event processor that coordinates all handlers"""

    def __init__(self):
        self.handlers = {
            EventType.MARKET_DATA_RECEIVED: MarketDataHandler(),
            EventType.SIGNAL_GENERATED: SignalHandler(),
            EventType.RISK_ASSESSMENT_COMPLETED: RiskHandler(),
            EventType.TRADE_EXECUTED: TradeHandler(),
            EventType.POSITION_OPENED: TradeHandler(),
            EventType.POSITION_CLOSED: TradeHandler()
        }

    async def process_event(self, event: TradingEvent):
        """Process a single event"""
        # Store event for audit trail
        await event_store.store_event(event)

        # Handle event with appropriate handler
        if event.event_type in self.handlers:
            handler = self.handlers[event.event_type]
            await handler.handle_event(event)

    async def process_event_stream(self, symbol: str):
        """Process event stream for a symbol"""
        last_id = '0'

        while True:
            try:
                events = await event_bus.consume_events(symbol, last_id)

                for message_id, event in events:
                    await self.process_event(event)
                    last_id = message_id

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"Error processing event stream: {e}")
                await asyncio.sleep(1)


# Global event processor
event_processor = EventProcessor()


async def start_event_processing(symbol: str):
    """Start event processing for a symbol"""
    print(f"🎯 Starting event processing for {symbol}")
    await event_processor.process_event_stream(symbol)


# Publishing functions for easy integration
async def publish_market_data_event(symbol: str, timeframe: str, df: pd.DataFrame):
    """Publish market data event"""
    event = create_market_data_event(symbol, timeframe, df)
    if event:
        await event_bus.publish_event(event)


async def publish_signal_event(symbol: str, timeframe: str, signal_type: str,
                              confidence: float, indicators: Dict[str, Any],
                              reasoning: str, source: str = "traditional"):
    """Publish signal generation event"""
    event = create_signal_event(symbol, timeframe, signal_type, confidence, indicators, reasoning, source)
    await event_bus.publish_event(event)


async def publish_risk_event(symbol: str, timeframe: str, assessment_result: str,
                            position_size: float, risk_amount: float, atr_value: float):
    """Publish risk assessment event"""
    event = create_risk_event(symbol, timeframe, assessment_result, position_size, risk_amount, atr_value)
    await event_bus.publish_event(event)


async def publish_trade_event(symbol: str, timeframe: str, side: str, size: float,
                             price: float, order_id: str, reason: str):
    """Publish trade execution event"""
    event = create_trade_event(symbol, timeframe, side, size, price, order_id, reason)
    await event_bus.publish_event(event)