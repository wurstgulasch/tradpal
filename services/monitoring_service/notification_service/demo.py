#!/usr/bin/env python3
"""
Notification Service Demo

Demonstrates the Notification Service capabilities including:
- Multi-channel notification delivery (Telegram, Discord, Email)
- Priority-based queuing and processing
- Signal notifications with formatted content
- Alert notifications for system events
- Real-time statistics and monitoring
- Async processing with worker pools
- Rate limiting and retry logic
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

from services.monitoring_service.notification_service.service import (
    NotificationService, NotificationConfig, NotificationMessage,
    NotificationType, NotificationPriority, NotificationChannel
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NotificationServiceDemo:
    """Demo class for Notification Service functionality."""

    def __init__(self):
        """Initialize demo with notification service."""
        self.config = NotificationConfig(
            max_queue_size=50,
            max_workers=3,
            default_channels=[NotificationChannel.TELEGRAM],
            rate_limits={
                'telegram': 10,  # messages per minute
                'discord': 10,
                'email': 5
            }
        )
        self.service = NotificationService(config=self.config)

    async def setup(self):
        """Setup the notification service."""
        logger.info("Setting up Notification Service...")
        await self.service.start()
        logger.info("Notification Service started successfully")

    async def teardown(self):
        """Teardown the notification service."""
        logger.info("Stopping Notification Service...")
        await self.service.stop()
        logger.info("Notification Service stopped")

    async def demo_basic_notifications(self):
        """Demonstrate basic notification sending."""
        logger.info("\n=== Basic Notification Demo ===")

        # Send different types of notifications
        notifications = [
            {
                'message': "System startup completed successfully",
                'title': "System Status",
                'type': NotificationType.INFO,
                'priority': NotificationPriority.NORMAL,
                'channels': [NotificationChannel.TELEGRAM]
            },
            {
                'message': "High memory usage detected: 85%",
                'title': "System Alert",
                'type': NotificationType.ALERT,
                'priority': NotificationPriority.HIGH,
                'channels': [NotificationChannel.TELEGRAM, NotificationChannel.EMAIL]
            },
            {
                'message': "Daily backup completed",
                'title': "Backup Status",
                'type': NotificationType.INFO,
                'priority': NotificationPriority.LOW,
                'channels': [NotificationChannel.DISCORD]
            }
        ]

        message_ids = []
        for notif in notifications:
            message_id = await self.service.send_notification(
                message=notif['message'],
                title=notif['title'],
                notification_type=notif['type'],
                priority=notif['priority'],
                channels=notif['channels']
            )
            message_ids.append(message_id)
            logger.info(f"Sent notification: {message_id}")

        # Wait for processing
        await asyncio.sleep(2)

        # Check status
        status = await self.service.get_queue_status()
        logger.info(f"Queue status: {status}")

        return message_ids

    async def demo_signal_notifications(self):
        """Demonstrate trading signal notifications."""
        logger.info("\n=== Trading Signal Notification Demo ===")

        # Sample trading signals
        signals = [
            {
                'symbol': 'BTC/USDT',
                'signal_type': 'BUY',
                'price': 45000.0,
                'timeframe': '1h',
                'indicators': {
                    'rsi': 28.5,
                    'ema9': 44500.0,
                    'ema21': 45500.0,
                    'bb_lower': 44000.0
                },
                'risk_management': {
                    'stop_loss': 43000.0,
                    'take_profit': 47000.0,
                    'position_size_percent': 1.0
                }
            },
            {
                'symbol': 'ETH/USDT',
                'signal_type': 'SELL',
                'price': 2800.0,
                'timeframe': '4h',
                'indicators': {
                    'rsi': 72.3,
                    'ema9': 2850.0,
                    'ema21': 2750.0,
                    'bb_upper': 2900.0
                },
                'risk_management': {
                    'stop_loss': 2900.0,
                    'take_profit': 2600.0,
                    'position_size_percent': 0.8
                }
            }
        ]

        signal_ids = []
        for signal in signals:
            message_id = await self.service.send_signal_notification(signal)
            signal_ids.append(message_id)
            logger.info(f"Sent signal notification: {message_id}")

        # Wait for processing
        await asyncio.sleep(2)

        return signal_ids

    async def demo_alert_notifications(self):
        """Demonstrate alert notifications."""
        logger.info("\n=== Alert Notification Demo ===")

        alerts = [
            {
                'alert_message': "API rate limit exceeded for Binance",
                'alert_level': "warning",
                'data': {
                    'exchange': 'binance',
                    'endpoint': 'klines',
                    'retry_after': 60
                }
            },
            {
                'alert_message': "Critical: Database connection lost",
                'alert_level': "critical",
                'data': {
                    'component': 'database',
                    'error': 'Connection timeout',
                    'impact': 'high'
                }
            },
            {
                'alert_message': "ML model accuracy dropped below threshold",
                'alert_level': "error",
                'data': {
                    'model': 'signal_predictor',
                    'accuracy': 0.65,
                    'threshold': 0.75
                }
            }
        ]

        alert_ids = []
        for alert in alerts:
            message_id = await self.service.send_alert_notification(
                alert_message=alert['alert_message'],
                alert_level=alert['alert_level'],
                data=alert['data']
            )
            alert_ids.append(message_id)
            logger.info(f"Sent alert notification: {message_id}")

        # Wait for processing
        await asyncio.sleep(2)

        return alert_ids

    async def demo_bulk_notifications(self):
        """Demonstrate bulk notification sending."""
        logger.info("\n=== Bulk Notification Demo ===")

        # Send multiple notifications concurrently
        num_notifications = 20
        logger.info(f"Sending {num_notifications} notifications concurrently...")

        start_time = asyncio.get_event_loop().time()

        tasks = []
        for i in range(num_notifications):
            task = self.service.send_notification(
                message=f"Bulk notification #{i+1}",
                title=f"Bulk Test {i+1}",
                notification_type=NotificationType.INFO,
                priority=NotificationPriority.NORMAL if i % 2 == 0 else NotificationPriority.LOW
            )
            tasks.append(task)

        message_ids = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()

        total_time = end_time - start_time
        logger.info(f"Sent {len(message_ids)} notifications in {total_time:.2f} seconds")
        logger.info(".2f")

        # Wait for processing
        await asyncio.sleep(3)

        return message_ids

    async def demo_monitoring_and_stats(self):
        """Demonstrate monitoring and statistics."""
        logger.info("\n=== Monitoring and Statistics Demo ===")

        # Get current statistics
        stats = await self.service.get_statistics()
        logger.info("Current Statistics:")
        logger.info(json.dumps(stats, indent=2, default=str))

        # Get queue status
        queue_status = await self.service.get_queue_status()
        logger.info("Queue Status:")
        logger.info(json.dumps(queue_status, indent=2))

        # Get health check
        health = await self.service.health_check()
        logger.info("Health Check:")
        logger.info(json.dumps(health, indent=2, default=str))

    async def demo_priority_handling(self):
        """Demonstrate priority-based notification handling."""
        logger.info("\n=== Priority Handling Demo ===")

        # Send notifications with different priorities
        priorities = [
            NotificationPriority.LOW,
            NotificationPriority.NORMAL,
            NotificationPriority.HIGH,
            NotificationPriority.CRITICAL
        ]

        priority_ids = []
        for priority in priorities:
            message_id = await self.service.send_notification(
                message=f"Priority {priority.value} notification",
                title=f"Priority Test - {priority.value}",
                priority=priority,
                channels=[NotificationChannel.TELEGRAM, NotificationChannel.DISCORD]
            )
            priority_ids.append(message_id)
            logger.info(f"Sent {priority.value} priority notification: {message_id}")

        # Wait for processing
        await asyncio.sleep(3)

        # Check processing order (higher priority should be processed first)
        logger.info("Priority notifications sent, check delivery order in logs")

        return priority_ids

    async def demo_rate_limiting(self):
        """Demonstrate rate limiting functionality."""
        logger.info("\n=== Rate Limiting Demo ===")

        # Send notifications rapidly to test rate limiting
        logger.info("Sending rapid notifications to test rate limiting...")

        tasks = []
        for i in range(15):  # More than rate limit
            task = self.service.send_notification(
                message=f"Rate limit test #{i+1}",
                channels=[NotificationChannel.TELEGRAM]
            )
            tasks.append(task)

        message_ids = await asyncio.gather(*tasks)
        logger.info(f"Sent {len(message_ids)} notifications for rate limiting test")

        # Wait and check rate limiting effects
        await asyncio.sleep(2)

        # Check statistics for rate limiting info
        stats = await self.service.get_statistics()
        telegram_stats = stats.get('channel_stats', {}).get('telegram', {})
        logger.info(f"Telegram channel stats: {telegram_stats}")

        return message_ids

    async def run_full_demo(self):
        """Run complete notification service demo."""
        logger.info("🚀 Starting Notification Service Demo")
        logger.info("=" * 50)

        try:
            # Setup
            await self.setup()

            # Run all demo scenarios
            await self.demo_basic_notifications()
            await self.demo_signal_notifications()
            await self.demo_alert_notifications()
            await self.demo_priority_handling()
            await self.demo_bulk_notifications()
            await self.demo_rate_limiting()

            # Final monitoring
            await self.demo_monitoring_and_stats()

            logger.info("\n✅ Notification Service Demo completed successfully!")

        except Exception as e:
            logger.error(f"❌ Demo failed with error: {e}")
            raise
        finally:
            # Cleanup
            await self.teardown()


async def main():
    """Main demo function."""
    demo = NotificationServiceDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())