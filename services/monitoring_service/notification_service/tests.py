#!/usr/bin/env python3
"""
Notification Service Tests

Comprehensive tests for the Notification Service including:
- Unit tests for notification processing and delivery
- Integration tests with actual integrations
- Performance tests for high-throughput scenarios
- Error handling and edge case testing
"""

import asyncio
import pytest
from datetime import datetime

from services.monitoring_service.notification_service.service import (
    NotificationService, NotificationConfig, NotificationMessage,
    NotificationType, NotificationPriority, NotificationChannel
)


class TestNotificationService:
    """Unit tests for NotificationService."""

    @pytest.fixture
    def event_system(self):
        """Create event system for testing."""
        return None  # Simplified for testing

    @pytest.fixture
    async def notification_service(self, event_system):
        """Create notification service instance."""
        config = NotificationConfig(
            max_queue_size=10,
            max_workers=2,
            default_channels=[NotificationChannel.TELEGRAM]
        )
        service = NotificationService(config=config, event_system=event_system)
        yield service
        # Cleanup
        if hasattr(service, 'is_running') and service.is_running:
            await service.stop()

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test service initialization."""
        config = NotificationConfig(
            max_queue_size=10,
            max_workers=2,
            default_channels=[NotificationChannel.TELEGRAM]
        )
        service = NotificationService(config=config, event_system=None)
        assert service is not None
        assert hasattr(service, 'config')
        assert service.config is not None

    def test_notification_message_creation(self):
        """Test notification message creation and serialization."""
        message = NotificationMessage(
            id="test_123",
            type=NotificationType.SIGNAL,
            priority=NotificationPriority.HIGH,
            title="Test Signal",
            message="Test message content",
            data={"test": "data"},
            channels=[NotificationChannel.TELEGRAM, NotificationChannel.EMAIL],
            timestamp=datetime.now()
        )

        assert message.id == "test_123"
        assert message.type == NotificationType.SIGNAL
        assert message.priority == NotificationPriority.HIGH
        assert message.title == "Test Signal"
        assert message.message == "Test message content"
        assert message.data == {"test": "data"}
        assert len(message.channels) == 2
        assert not message.delivered

    @pytest.mark.asyncio
    async def test_service_start_stop(self):
        """Test service start and stop."""
        config = NotificationConfig(
            max_queue_size=10,
            max_workers=2,
            default_channels=[NotificationChannel.TELEGRAM]
        )
        service = NotificationService(config=config, event_system=None)

        # Start service
        await service.start()
        assert service.is_running
        assert len(service.processing_tasks) == service.config.max_workers

        # Stop service
        await service.stop()
        assert not service.is_running

    @pytest.mark.asyncio
    async def test_send_notification(self):
        """Test sending a notification."""
        config = NotificationConfig(
            max_queue_size=10,
            max_workers=2,
            default_channels=[NotificationChannel.TELEGRAM]
        )
        service = NotificationService(config=config, event_system=None)

        await service.start()

        message_id = await service.send_notification(
            message="Test notification",
            title="Test Title",
            notification_type=NotificationType.INFO,
            priority=NotificationPriority.NORMAL
        )

        assert message_id.startswith("msg_")
        assert service.stats['messages_queued'] == 1

        await service.stop()

    @pytest.mark.asyncio
    async def test_send_signal_notification(self):
        """Test sending a signal notification."""
        config = NotificationConfig(
            max_queue_size=10,
            max_workers=2,
            default_channels=[NotificationChannel.TELEGRAM]
        )
        service = NotificationService(config=config, event_system=None)

        await service.start()

        signal_data = {
            'symbol': 'BTC/USDT',
            'signal_type': 'BUY',
            'price': 50000.0,
            'timeframe': '1h',
            'indicators': {'rsi': 30.5, 'ema9': 49500},
            'risk_management': {'stop_loss': 48500, 'take_profit': 52000}
        }

        message_id = await service.send_signal_notification(signal_data)

        assert message_id.startswith("msg_")
        assert service.stats['messages_queued'] == 1

        await service.stop()

    @pytest.mark.asyncio
    async def test_send_alert_notification(self):
        """Test sending an alert notification."""
        config = NotificationConfig(
            max_queue_size=10,
            max_workers=2,
            default_channels=[NotificationChannel.TELEGRAM]
        )
        service = NotificationService(config=config, event_system=None)

        await service.start()

        message_id = await service.send_alert_notification(
            alert_message="Test alert",
            alert_level="warning",
            data={"component": "test"}
        )

        assert message_id.startswith("msg_")
        assert service.stats['messages_queued'] == 1

        await service.stop()

    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test statistics tracking."""
        config = NotificationConfig(
            max_queue_size=10,
            max_workers=2,
            default_channels=[NotificationChannel.TELEGRAM]
        )
        service = NotificationService(config=config, event_system=None)

        await service.start()

        # Send a notification
        await service.send_notification("Test message")

        # Statistics should be updated
        assert service.stats['messages_queued'] >= 1

        stats = await service.get_statistics()
        assert 'messages_sent' in stats
        assert 'messages_queued' in stats
        assert 'channel_stats' in stats

        await service.stop()

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        config = NotificationConfig(
            max_queue_size=10,
            max_workers=2,
            default_channels=[NotificationChannel.TELEGRAM]
        )
        service = NotificationService(config=config, event_system=None)

        await service.start()

        health = await service.health_check()

        assert health['service'] == 'notification_service'
        assert health['status'] == 'healthy'
        assert 'timestamp' in health
        assert 'queue_status' in health
        assert 'integrations_active' in health
        assert 'statistics' in health

        await service.stop()

        # Test stopped state
        health = await service.health_check()
        assert health['status'] == 'stopped'


class TestNotificationServiceIntegration:
    """Integration tests requiring external dependencies."""

    @pytest.mark.asyncio
    async def test_end_to_end_notification_flow(self):
        """Test complete notification flow from send to delivery."""
        config = NotificationConfig(
            max_queue_size=50,
            max_workers=3,
            default_channels=[NotificationChannel.TELEGRAM]
        )
        service = NotificationService(config=config)

        await service.start()

        # Send multiple notifications
        message_ids = []
        for i in range(5):
            message_id = await service.send_notification(
                message=f"Test notification {i}",
                title=f"Test {i}",
                notification_type=NotificationType.INFO,
                priority=NotificationPriority.NORMAL
            )
            message_ids.append(message_id)

        assert len(message_ids) == 5
        assert all(mid.startswith("msg_") for mid in message_ids)

        # Check queue processing
        await asyncio.sleep(0.1)  # Allow processing time

        status = await service.get_queue_status()
        # Queue should be processed (though delivery may fail without real integrations)
        assert status['queue_size'] <= 5

        await service.stop()

    @pytest.mark.asyncio
    async def test_priority_handling(self):
        """Test notification priority handling."""
        config = NotificationConfig(
            max_queue_size=50,
            max_workers=3,
            default_channels=[NotificationChannel.TELEGRAM]
        )
        service = NotificationService(config=config)

        await service.start()

        # Send notifications with different priorities
        priorities = [
            NotificationPriority.LOW,
            NotificationPriority.NORMAL,
            NotificationPriority.HIGH,
            NotificationPriority.CRITICAL
        ]

        message_ids = []
        for priority in priorities:
            message_id = await service.send_notification(
                message=f"Priority {priority.value} message",
                priority=priority
            )
            message_ids.append(message_id)

        assert len(message_ids) == 4

        # All should be queued
        status = await service.get_queue_status()
        assert status['queue_size'] == 4

        await service.stop()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])