"""
TradPal Notification Service

A comprehensive notification microservice that provides:
- Multi-channel notification delivery (Telegram, Discord, Email, SMS, Webhooks)
- Async processing with priority queuing
- Event-driven architecture
- Comprehensive monitoring and statistics
- Rate limiting and retry logic
"""

from .service import (
    NotificationService,
    NotificationConfig,
    NotificationMessage,
    NotificationType,
    NotificationPriority,
    NotificationChannel,
    EventSystem
)

# from .api import app  # API not implemented yet

__version__ = "1.0.0"

__all__ = [
    # Service classes
    'NotificationService',
    'EventSystem',

    # Configuration
    'NotificationConfig',

    # Data models
    'NotificationMessage',
    'NotificationType',
    'NotificationPriority',
    'NotificationChannel',

    # API
    # 'app'  # API not implemented yet
]