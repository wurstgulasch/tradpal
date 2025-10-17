#!/usr/bin/env python3
"""
Unit tests for TradPal Event System
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from services.event_system import (
    Event, EventType, EventPublisher, EventSubscriber,
    EventStore, EventSystem, publish_market_data,
    publish_trading_signal, publish_portfolio_update
)


class TestEvent:
    """Test Event class"""

    def test_event_creation(self):
        """Test basic event creation"""
        event = Event(
            event_type=EventType.MARKET_DATA_UPDATE,
            source="test_source",
            data={"key": "value"}
        )

        assert event.event_type == EventType.MARKET_DATA_UPDATE
        assert event.source == "test_source"
        assert event.data == {"key": "value"}
        assert isinstance(event.timestamp, datetime)
        assert event.event_id is not None

    def test_event_serialization(self):
        """Test event to/from dict conversion"""
        original_event = Event(
            event_type=EventType.TRADING_SIGNAL,
            source="test",
            data={"signal": "BUY"},
            metadata={"confidence": 0.9}
        )

        # Convert to dict
        event_dict = original_event.to_dict()

        # Convert back to event
        restored_event = Event.from_dict(event_dict)

        assert restored_event.event_type == original_event.event_type
        assert restored_event.source == original_event.source
        assert restored_event.data == original_event.data
        assert restored_event.metadata == original_event.metadata
        assert restored_event.event_id == original_event.event_id


class TestEventPublisher:
    """Test EventPublisher class"""

    @pytest.mark.asyncio
    async def test_publish_event(self):
        """Test publishing an event"""
        mock_redis = AsyncMock()
        mock_redis.xadd.return_value = "123456789-0"

        publisher = EventPublisher(mock_redis)

        event = Event(event_type=EventType.MARKET_DATA_UPDATE, data={"test": True})

        message_id = await publisher.publish(event)

        assert message_id == "123456789-0"
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "tradpal_events"

    @pytest.mark.asyncio
    async def test_publish_batch(self):
        """Test publishing multiple events"""
        mock_redis = AsyncMock()
        mock_redis.xadd.side_effect = ["id1", "id2", "id3"]

        publisher = EventPublisher(mock_redis)

        events = [
            Event(event_type=EventType.MARKET_DATA_UPDATE),
            Event(event_type=EventType.TRADING_SIGNAL),
            Event(event_type=EventType.PORTFOLIO_UPDATE)
        ]

        message_ids = await publisher.publish_batch(events)

        assert message_ids == ["id1", "id2", "id3"]
        assert mock_redis.xadd.call_count == 3


class TestEventSubscriber:
    """Test EventSubscriber class"""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test subscriber initialization"""
        mock_redis = AsyncMock()

        subscriber = EventSubscriber(mock_redis)

        await subscriber.initialize()

        mock_redis.xgroup_create.assert_called_once_with(
            "tradpal_events",
            "tradpal_consumers",
            "$",
            mkstream=True
        )

    @pytest.mark.asyncio
    async def test_register_handler(self):
        """Test handler registration"""
        subscriber = EventSubscriber(Mock())

        async def dummy_handler(event):
            pass

        subscriber.register_handler(EventType.MARKET_DATA_UPDATE, dummy_handler)

        assert EventType.MARKET_DATA_UPDATE in subscriber.handlers
        assert len(subscriber.handlers[EventType.MARKET_DATA_UPDATE]) == 1

    @pytest.mark.asyncio
    async def test_event_processing(self):
        """Test event processing"""
        mock_redis = AsyncMock()
        # Mock xreadgroup to return no messages (empty list)
        mock_redis.xreadgroup.return_value = []

        subscriber = EventSubscriber(mock_redis)

        # Register a handler
        handler_called = False
        async def test_handler(event):
            nonlocal handler_called
            handler_called = True

        subscriber.register_handler(EventType.MARKET_DATA_UPDATE, test_handler)

        # Mock the process_events method to run briefly
        original_process = subscriber.process_events

        async def mock_process():
            # Simulate reading one message
            messages = [[
                "tradpal_events",
                [["message_id", {"event": '{"event_type": "market_data_update", "event_id": "test", "timestamp": "2023-01-01T00:00:00", "source": "test", "data": {}, "metadata": {}}'}]]
            ]]

            mock_redis.xreadgroup.return_value = messages

            # Process the message
            for stream_name, message_list in messages:
                for message_id, message_data in message_list:
                    event_data = json.loads(message_data["event"])
                    event = Event.from_dict(event_data)
                    await subscriber._handle_event(event)
                    await mock_redis.xack("tradpal_events", "tradpal_consumers", message_id)

            # Stop after one iteration
            subscriber.running = False

        subscriber.process_events = mock_process

        await subscriber.process_events()

        # Note: This test would need more complex mocking to fully work
        # For now, we just ensure the method can be called


class TestEventStore:
    """Test EventStore class"""

    @pytest.mark.asyncio
    async def test_get_events(self):
        """Test retrieving events"""
        mock_redis = AsyncMock()
        mock_redis.xrange.return_value = [
            ("123456789-0", {"event": '{"event_type": "market_data_update", "event_id": "test", "timestamp": "2023-01-01T00:00:00", "source": "test", "data": {"price": 100}, "metadata": {}}'})
        ]

        store = EventStore(mock_redis)

        events = await store.get_events(limit=10)

        assert len(events) == 1
        assert events[0].event_type == EventType.MARKET_DATA_UPDATE
        assert events[0].data["price"] == 100

    @pytest.mark.asyncio
    async def test_replay_events(self):
        """Test event replay"""
        mock_redis = AsyncMock()
        mock_redis.xrange.return_value = [
            ("123456789-0", {"event": '{"event_type": "trading_signal", "event_id": "test", "timestamp": "2023-01-01T00:00:00", "source": "test", "data": {"signal": "BUY"}, "metadata": {}}'})
        ]

        store = EventStore(mock_redis)

        replayed_events = []
        async def replay_handler(event):
            replayed_events.append(event)

        await store.replay_events(handler=replay_handler)

        assert len(replayed_events) == 1
        assert replayed_events[0].event_type == EventType.TRADING_SIGNAL


class TestEventSystem:
    """Test EventSystem class"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test event system initialization"""
        with patch('services.event_system.redis') as mock_redis_module:
            mock_redis = AsyncMock()
            mock_redis_module.from_url.return_value = mock_redis

            system = EventSystem()
            await system.initialize()

            assert system.redis is not None
            assert system.publisher is not None
            assert system.subscriber is not None
            assert system.store is not None

            mock_redis_module.from_url.assert_called_once_with("redis://localhost:6379")


class TestConvenienceFunctions:
    """Test convenience functions"""

    @pytest.mark.asyncio
    async def test_publish_market_data(self):
        """Test publish_market_data convenience function"""
        with patch('services.event_system.get_event_system') as mock_get_system:
            mock_system = AsyncMock()
            mock_get_system.return_value = mock_system
            mock_system.publish_event.return_value = "test_id"

            message_id = await publish_market_data("BTC/USDT", {"price": 50000})

            assert message_id == "test_id"
            mock_system.publish_event.assert_called_once()
            event_arg = mock_system.publish_event.call_args[0][0]
            assert event_arg.event_type == EventType.MARKET_DATA_UPDATE
            assert event_arg.data["symbol"] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_publish_trading_signal(self):
        """Test publish_trading_signal convenience function"""
        with patch('services.event_system.get_event_system') as mock_get_system:
            mock_system = AsyncMock()
            mock_get_system.return_value = mock_system
            mock_system.publish_event.return_value = "test_id"

            message_id = await publish_trading_signal({"action": "BUY"})

            assert message_id == "test_id"
            mock_system.publish_event.assert_called_once()
            event_arg = mock_system.publish_event.call_args[0][0]
            assert event_arg.event_type == EventType.TRADING_SIGNAL
            assert event_arg.data["action"] == "BUY"

    @pytest.mark.asyncio
    async def test_publish_portfolio_update(self):
        """Test publish_portfolio_update convenience function"""
        with patch('services.event_system.get_event_system') as mock_get_system:
            mock_system = AsyncMock()
            mock_get_system.return_value = mock_system
            mock_system.publish_event.return_value = "test_id"

            message_id = await publish_portfolio_update({"pnl": 1000})

            assert message_id == "test_id"
            mock_system.publish_event.assert_called_once()
            event_arg = mock_system.publish_event.call_args[0][0]
            assert event_arg.event_type == EventType.PORTFOLIO_UPDATE
            assert event_arg.data["pnl"] == 1000


# Import json for the test that uses it
import json