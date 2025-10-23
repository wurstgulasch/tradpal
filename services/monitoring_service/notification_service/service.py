#!/usr/bin/env python3
"""
Notification Service

A comprehensive notification microservice for TradPal that handles:
- Telegram notifications
- Discord webhooks
- Email alerts
- SMS notifications (future)
- Webhook integrations

This service provides unified notification management with:
- Async processing for high throughput
- Event-driven architecture
- Comprehensive error handling and retries
- Template-based message formatting
- Priority-based notification queuing
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import os

from integrations.base import (
    IntegrationManager, BaseIntegration, IntegrationConfig,
    SignalData, integration_manager
)
from integrations.telegram.bot import TelegramIntegration, TelegramConfig
from integrations.discord.discord import DiscordIntegration, DiscordConfig
from integrations.email_integration.email import EmailIntegration, EmailConfig


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationType(Enum):
    """Types of notifications"""
    SIGNAL = "signal"
    ALERT = "alert"
    STATUS = "status"
    ERROR = "error"
    INFO = "info"


class NotificationChannel(Enum):
    """Available notification channels"""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"


@dataclass
class NotificationMessage:
    """Standardized notification message structure"""
    id: str
    type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    data: Dict[str, Any]
    channels: List[NotificationChannel]
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    delivered: bool = False
    delivery_time: Optional[datetime] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['type'] = self.type.value
        result['priority'] = self.priority.value
        result['channels'] = [c.value for c in self.channels]
        result['timestamp'] = self.timestamp.isoformat()
        if self.delivery_time:
            result['delivery_time'] = self.delivery_time.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NotificationMessage':
        """Create from dictionary"""
        data_copy = data.copy()
        data_copy['type'] = NotificationType(data['type'])
        data_copy['priority'] = NotificationPriority(data['priority'])
        data_copy['channels'] = [NotificationChannel(c) for c in data['channels']]
        data_copy['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'delivery_time' in data and data['delivery_time']:
            data_copy['delivery_time'] = datetime.fromisoformat(data['delivery_time'])
        return cls(**data_copy)


@dataclass
class NotificationConfig:
    """Configuration for Notification Service"""
    enabled: bool = True
    max_queue_size: int = 1000
    max_workers: int = 5
    retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    backoff_factor: float = 2.0
    default_channels: List[NotificationChannel] = None
    rate_limits: Dict[str, int] = None  # messages per minute per channel

    def __post_init__(self):
        if self.default_channels is None:
            self.default_channels = [NotificationChannel.TELEGRAM, NotificationChannel.EMAIL]
        if self.rate_limits is None:
            self.rate_limits = {
                'telegram': 30,  # 30 messages per minute
                'discord': 60,
                'email': 10,
                'sms': 5
            }


class EventSystem:
    """Simple event system for notifications"""

    def __init__(self):
        self.handlers: Dict[str, List[callable]] = {}

    def subscribe(self, event_type: str, handler: callable):
        """Subscribe to an event"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish an event"""
        if event_type in self.handlers:
            tasks = []
            for handler in self.handlers[event_type]:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(data))
                else:
                    tasks.append(asyncio.to_thread(handler, data))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


class NotificationService:
    """Main notification service class"""

    def __init__(self, config: NotificationConfig = None, event_system: EventSystem = None):
        self.config = config or NotificationConfig()
        self.event_system = event_system or EventSystem()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Queue and processing
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.processing_tasks: List[asyncio.Task] = []
        self.is_running = False

        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_failed': 0,
            'messages_queued': 0,
            'messages_retried': 0,
            'channel_stats': {}
        }

        # Rate limiting
        self.last_send_times: Dict[str, datetime] = {}
        self.rate_limit_counters: Dict[str, int] = {}

        # Initialize integrations
        self._setup_integrations()

    def _setup_integrations(self):
        """Setup notification integrations"""
        # Telegram
        if os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'):
            telegram_config = TelegramConfig.from_env()
            telegram_integration = TelegramIntegration(telegram_config)
            integration_manager.register_integration("telegram", telegram_integration)

        # Discord
        if os.getenv('DISCORD_WEBHOOK_URL'):
            discord_config = DiscordConfig.from_env()
            discord_integration = DiscordIntegration(discord_config)
            integration_manager.register_integration("discord", discord_integration)

        # Email
        if os.getenv('EMAIL_USERNAME') and os.getenv('EMAIL_PASSWORD'):
            email_config = EmailConfig.from_env()
            email_integration = EmailIntegration(email_config)
            integration_manager.register_integration("email", email_integration)

    async def start(self):
        """Start the notification service"""
        if self.is_running:
            return

        self.logger.info("Starting Notification Service...")

        # Initialize integrations
        init_results = integration_manager.initialize_all()
        successful = sum(1 for success in init_results.values() if success)
        self.logger.info(f"Initialized {successful}/{len(init_results)} integrations")

        # Start processing tasks
        self.is_running = True
        for i in range(self.config.max_workers):
            task = asyncio.create_task(self._process_messages())
            self.processing_tasks.append(task)

        # Subscribe to events
        self.event_system.subscribe("notification.send", self._handle_send_event)
        self.event_system.subscribe("notification.batch_send", self._handle_batch_send_event)

        self.logger.info(f"Notification Service started with {len(self.processing_tasks)} workers")

    async def stop(self):
        """Stop the notification service"""
        if not self.is_running:
            return

        self.logger.info("Stopping Notification Service...")

        # Stop processing
        self.is_running = False

        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)

        # Shutdown integrations
        integration_manager.shutdown_all()

        self.logger.info("Notification Service stopped")

    async def send_notification(self,
                              message: str,
                              title: str = "",
                              notification_type: NotificationType = NotificationType.INFO,
                              priority: NotificationPriority = NotificationPriority.NORMAL,
                              channels: List[NotificationChannel] = None,
                              data: Dict[str, Any] = None) -> str:
        """Send a notification"""
        message_id = f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(message) % 10000}"

        notification = NotificationMessage(
            id=message_id,
            type=notification_type,
            priority=priority,
            title=title or f"{notification_type.value.upper()} Notification",
            message=message,
            data=data or {},
            channels=channels or self.config.default_channels.copy(),
            timestamp=datetime.now()
        )

        # Check queue size
        if self.message_queue.qsize() >= self.config.max_queue_size:
            self.logger.warning("Message queue full, dropping notification")
            return message_id

        # Add to queue
        await self.message_queue.put(notification)
        self.stats['messages_queued'] += 1

        self.logger.debug(f"Queued notification: {message_id}")
        return message_id

    async def send_signal_notification(self, signal_data: Dict[str, Any]) -> str:
        """Send a trading signal notification"""
        signal = SignalData.from_trading_signal(signal_data)
        signal_type = signal.signal_type

        if signal_type == "BUY":
            emoji = "🟢"
            priority = NotificationPriority.HIGH
        elif signal_type == "SELL":
            emoji = "🔴"
            priority = NotificationPriority.HIGH
        else:
            emoji = "⚪"
            priority = NotificationPriority.NORMAL

        title = f"{emoji} {signal_type} Signal - {signal.symbol}"
        message = self._format_signal_message(signal)

        return await self.send_notification(
            message=message,
            title=title,
            notification_type=NotificationType.SIGNAL,
            priority=priority,
            data={"signal": signal.to_dict()}
        )

    async def send_alert_notification(self,
                                    alert_message: str,
                                    alert_level: str = "info",
                                    data: Dict[str, Any] = None) -> str:
        """Send an alert notification"""
        priority_map = {
            "info": NotificationPriority.NORMAL,
            "warning": NotificationPriority.HIGH,
            "error": NotificationPriority.CRITICAL,
            "critical": NotificationPriority.CRITICAL
        }

        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨"
        }

        priority = priority_map.get(alert_level.lower(), NotificationPriority.NORMAL)
        emoji = emoji_map.get(alert_level.lower(), "ℹ️")

        title = f"{emoji} Alert - {alert_level.upper()}"
        message = f"**{alert_level.upper()}**: {alert_message}"

        return await self.send_notification(
            message=message,
            title=title,
            notification_type=NotificationType.ALERT,
            priority=priority,
            data=data or {}
        )

    async def send_status_notification(self, status_message: str, data: Dict[str, Any] = None) -> str:
        """Send a status notification"""
        return await self.send_notification(
            message=status_message,
            title="📊 Status Update",
            notification_type=NotificationType.STATUS,
            priority=NotificationPriority.NORMAL,
            data=data or {}
        )

    def _format_signal_message(self, signal: SignalData) -> str:
        """Format a signal notification message"""
        lines = [
            f"💰 **Symbol:** {signal.symbol}",
            f"💰 **Price:** {signal.price:.5f}",
            f"📊 **Timeframe:** {signal.timeframe}",
        ]

        # Add indicators
        if signal.indicators:
            indicator_lines = []
            for key, value in signal.indicators.items():
                if isinstance(value, float):
                    indicator_lines.append(f"**{key.upper()}:** {value:.4f}")
                else:
                    indicator_lines.append(f"**{key.upper()}:** {value}")
            if indicator_lines:
                lines.append("📈 **Indicators:**")
                lines.extend([f"  • {line}" for line in indicator_lines])

        # Add risk management
        if signal.risk_management:
            risk_lines = []
            for key, value in signal.risk_management.items():
                if isinstance(value, float):
                    risk_lines.append(f"**{key.replace('_', ' ').title()}:** {value:.2f}")
                else:
                    risk_lines.append(f"**{key.replace('_', ' ').title()}:** {value}")
            if risk_lines:
                lines.append("⚠️ **Risk Management:**")
                lines.extend([f"  • {line}" for line in risk_lines])

        # Add timestamp
        lines.append(f"🕐 **Time:** {signal.timestamp.strftime('%H:%M:%S %d.%m.%Y')}")

        return "\n".join(lines)

    async def _process_messages(self):
        """Process messages from the queue"""
        while self.is_running:
            try:
                # Get message from queue
                notification = await self.message_queue.get()

                # Process the message
                await self._deliver_notification(notification)

                # Mark task as done
                self.message_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")

    async def _deliver_notification(self, notification: NotificationMessage):
        """Deliver a notification to all specified channels"""
        success_count = 0
        errors = []

        for channel in notification.channels:
            try:
                # Check rate limits
                if not self._check_rate_limit(channel.value):
                    self.logger.warning(f"Rate limit exceeded for {channel.value}")
                    continue

                # Deliver to channel
                success = await self._deliver_to_channel(notification, channel)
                if success:
                    success_count += 1
                    self._update_channel_stats(channel.value, True)
                else:
                    self._update_channel_stats(channel.value, False)
                    errors.append(f"{channel.value}: delivery failed")

            except Exception as e:
                error_msg = f"{channel.value}: {str(e)}"
                errors.append(error_msg)
                self._update_channel_stats(channel.value, False)
                self.logger.error(f"Error delivering to {channel.value}: {e}")

        # Update notification status
        if success_count > 0:
            notification.delivered = True
            notification.delivery_time = datetime.now()
            self.stats['messages_sent'] += 1
        else:
            notification.delivered = False
            notification.error_message = "; ".join(errors)
            self.stats['messages_failed'] += 1

            # Retry logic
            if notification.retry_count < notification.max_retries:
                notification.retry_count += 1
                self.stats['messages_retried'] += 1

                # Exponential backoff
                delay = min(
                    self.config.retry_delay * (self.config.backoff_factor ** notification.retry_count),
                    self.config.max_retry_delay
                )

                self.logger.info(f"Retrying notification {notification.id} in {delay:.1f}s")
                await asyncio.sleep(delay)

                # Re-queue
                await self.message_queue.put(notification)

    async def _deliver_to_channel(self, notification: NotificationMessage, channel: NotificationChannel) -> bool:
        """Deliver notification to a specific channel"""
        # Convert notification to signal data format for integrations
        signal_data = {
            'signal_type': notification.type.value.upper(),
            'symbol': notification.data.get('symbol', 'SYSTEM'),
            'price': notification.data.get('price', 0),
            'timeframe': notification.data.get('timeframe', 'N/A'),
            'indicators': notification.data.get('indicators', {}),
            'risk_management': notification.data.get('risk_management', {}),
            'timestamp': notification.timestamp,
            'title': notification.title,
            'message': notification.message,
            'priority': notification.priority.value
        }

        # Send via integration manager
        results = integration_manager.send_signal_to_all(signal_data)

        # Check if the specific channel was successful
        channel_name = channel.value
        return results.get(channel_name, False)

    def _check_rate_limit(self, channel: str) -> bool:
        """Check if rate limit allows sending"""
        now = datetime.now()
        limit = self.config.rate_limits.get(channel, 60)  # default 60 per minute

        # Reset counter if minute has passed
        if channel not in self.last_send_times or (now - self.last_send_times[channel]).seconds >= 60:
            self.rate_limit_counters[channel] = 0
            self.last_send_times[channel] = now

        # Check limit
        if self.rate_limit_counters[channel] >= limit:
            return False

        self.rate_limit_counters[channel] += 1
        return True

    def _update_channel_stats(self, channel: str, success: bool):
        """Update channel statistics"""
        if channel not in self.stats['channel_stats']:
            self.stats['channel_stats'][channel] = {'sent': 0, 'failed': 0}

        if success:
            self.stats['channel_stats'][channel]['sent'] += 1
        else:
            self.stats['channel_stats'][channel]['failed'] += 1

    async def _handle_send_event(self, data: Dict[str, Any]):
        """Handle notification.send event"""
        await self.send_notification(**data)

    async def _handle_batch_send_event(self, data: Dict[str, Any]):
        """Handle notification.batch_send event"""
        notifications = data.get('notifications', [])
        tasks = []
        for notification_data in notifications:
            task = self.send_notification(**notification_data)
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status"""
        return {
            'queue_size': self.message_queue.qsize(),
            'max_queue_size': self.config.max_queue_size,
            'active_workers': len([t for t in self.processing_tasks if not t.done()]),
            'total_workers': len(self.processing_tasks)
        }

    async def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self.stats,
            'uptime': str(datetime.now() - datetime.now()),  # Would need start time
            'integrations': integration_manager.get_status_overview()
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        queue_status = await self.get_queue_status()

        return {
            'service': 'notification_service',
            'status': 'healthy' if self.is_running else 'stopped',
            'timestamp': datetime.now().isoformat(),
            'queue_status': queue_status,
            'integrations_active': len(integration_manager.integrations),
            'statistics': self.stats
        }