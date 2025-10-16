# Notification Service

A comprehensive, async notification service for TradPal that provides multi-channel notification delivery with priority queuing, rate limiting, and comprehensive monitoring.

## Features

- **Multi-Channel Delivery**: Support for Telegram, Discord, Email, and SMS notifications
- **Priority-Based Queuing**: Critical, High, Normal, and Low priority levels
- **Async Processing**: Non-blocking notification delivery with worker pools
- **Rate Limiting**: Per-channel rate limiting to prevent API abuse
- **Retry Logic**: Exponential backoff for failed deliveries
- **Event-Driven Architecture**: Integration with EventSystem for real-time notifications
- **Comprehensive Monitoring**: Statistics, health checks, and queue monitoring
- **Signal Notifications**: Specialized formatting for trading signals
- **Alert Notifications**: System alert notifications with severity levels

## Architecture

```
Notification Service
├── service.py          # Core NotificationService class
├── api.py             # FastAPI REST API
├── demo.py            # Demonstration script
├── tests.py           # Comprehensive test suite
└── __init__.py        # Package initialization
```

## Quick Start

### Basic Usage

```python
from services.notification_service import NotificationService, NotificationConfig

# Create configuration
config = NotificationConfig(
    max_queue_size=100,
    max_workers=3,
    default_channels=['telegram']
)

# Create and start service
service = NotificationService(config=config)
await service.start()

# Send a notification
message_id = await service.send_notification(
    message="Hello World!",
    title="Test Notification",
    priority="normal"
)

# Stop service
await service.stop()
```

### Signal Notifications

```python
# Send trading signal notification
signal_data = {
    'symbol': 'BTC/USDT',
    'signal_type': 'BUY',
    'price': 45000.0,
    'timeframe': '1h',
    'indicators': {'rsi': 28.5, 'ema9': 44500.0},
    'risk_management': {'stop_loss': 43000.0, 'take_profit': 47000.0}
}

message_id = await service.send_signal_notification(signal_data)
```

### Alert Notifications

```python
# Send system alert
message_id = await service.send_alert_notification(
    alert_message="Database connection lost",
    alert_level="critical",
    data={'component': 'database', 'error': 'timeout'}
)
```

## Configuration

### NotificationConfig

```python
from services.notification_service import NotificationConfig

config = NotificationConfig(
    max_queue_size=100,           # Maximum queue size
    max_workers=3,                # Number of async workers
    default_channels=['telegram'], # Default notification channels
    rate_limits={                 # Per-channel rate limits (messages/minute)
        'telegram': 10,
        'discord': 10,
        'email': 5,
        'sms': 2
    },
    retry_attempts=3,             # Maximum retry attempts
    retry_delay=1.0,              # Initial retry delay (seconds)
    max_retry_delay=60.0          # Maximum retry delay (seconds)
)
```

## API Endpoints

The service provides a REST API for external integration:

### Send Notification
```http
POST /notifications/send
Content-Type: application/json

{
    "message": "Notification message",
    "title": "Notification Title",
    "type": "info",
    "priority": "normal",
    "channels": ["telegram", "email"]
}
```

### Send Signal Notification
```http
POST /notifications/signal
Content-Type: application/json

{
    "symbol": "BTC/USDT",
    "signal_type": "BUY",
    "price": 45000.0,
    "timeframe": "1h",
    "indicators": {...},
    "risk_management": {...}
}
```

### Get Queue Status
```http
GET /queue/status
```

### Get Statistics
```http
GET /statistics
```

### Health Check
```http
GET /health
```

## Notification Types

- **INFO**: General information notifications
- **SIGNAL**: Trading signal notifications
- **ALERT**: System alerts and warnings
- **ERROR**: Error notifications

## Priority Levels

- **CRITICAL**: Immediate attention required
- **HIGH**: Important notifications
- **NORMAL**: Standard notifications
- **LOW**: Low-priority notifications

## Supported Channels

- **TELEGRAM**: Telegram bot notifications
- **DISCORD**: Discord webhook notifications
- **EMAIL**: SMTP email notifications
- **SMS**: SMS notifications (via integrations)

## Integration Examples

### With Event System

```python
from services.notification_service import EventSystem

# Create event system
event_system = EventSystem()

# Create notification service with event system
service = NotificationService(config=config, event_system=event_system)

# Subscribe to events
await event_system.subscribe("trading.signal", service.handle_signal_event)
await event_system.subscribe("system.alert", service.handle_alert_event)
```

### With FastAPI Application

```python
from fastapi import FastAPI
from services.notification_service.api import create_notification_app

# Create main app
app = FastAPI()

# Include notification service
notification_app = create_notification_app()
app.mount("/notifications", notification_app)
```

## Monitoring and Statistics

The service provides comprehensive monitoring:

```python
# Get statistics
stats = await service.get_statistics()
print(f"Messages sent: {stats['messages_sent']}")
print(f"Messages queued: {stats['messages_queued']}")
print(f"Channel stats: {stats['channel_stats']}")

# Get queue status
queue_status = await service.get_queue_status()
print(f"Queue size: {queue_status['queue_size']}")
print(f"Processing: {queue_status['processing']}")

# Health check
health = await service.health_check()
print(f"Status: {health['status']}")
```

## Rate Limiting

Rate limiting prevents API abuse:

```python
# Configure rate limits
config = NotificationConfig(
    rate_limits={
        'telegram': 10,  # 10 messages per minute
        'email': 5       # 5 emails per minute
    }
)

# Rate limiting is automatically enforced
# Exceeded messages are queued for later delivery
```

## Error Handling and Retry Logic

The service includes robust error handling:

- **Automatic Retries**: Failed deliveries are retried with exponential backoff
- **Circuit Breaker**: Prevents cascading failures
- **Graceful Degradation**: Continues operation even if some channels fail
- **Detailed Logging**: Comprehensive error logging for debugging

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest services/notification_service/tests.py -v

# Run specific test categories
python -m pytest services/notification_service/tests.py::TestNotificationService -v
python -m pytest services/notification_service/tests.py::TestNotificationServiceIntegration -v

# Run performance tests
python -m pytest services/notification_service/tests.py::TestNotificationServicePerformance -v -m performance
```

## Demo

Run the demonstration script:

```bash
python services/notification_service/demo.py
```

The demo showcases:
- Basic notification sending
- Signal notifications
- Alert notifications
- Priority handling
- Bulk notifications
- Rate limiting
- Monitoring and statistics

## Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY services/notification_service/ ./services/notification_service/
COPY integrations/ ./integrations/

EXPOSE 8001
CMD ["uvicorn", "services.notification_service.api:app", "--host", "0.0.0.0", "--port", "8001"]
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: notification-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: notification-service
  template:
    metadata:
      labels:
        app: notification-service
    spec:
      containers:
      - name: notification-service
        image: tradpal/notification-service:latest
        ports:
        - containerPort: 8001
        env:
        - name: NOTIFICATION_MAX_WORKERS
          value: "3"
        - name: NOTIFICATION_QUEUE_SIZE
          value: "100"
```

## Configuration via Environment Variables

```bash
# Service configuration
NOTIFICATION_MAX_WORKERS=3
NOTIFICATION_QUEUE_SIZE=100
NOTIFICATION_DEFAULT_CHANNELS=telegram,email

# Rate limits
NOTIFICATION_RATE_LIMIT_TELEGRAM=10
NOTIFICATION_RATE_LIMIT_DISCORD=10
NOTIFICATION_RATE_LIMIT_EMAIL=5

# Retry configuration
NOTIFICATION_RETRY_ATTEMPTS=3
NOTIFICATION_RETRY_DELAY=1.0
NOTIFICATION_MAX_RETRY_DELAY=60.0
```

## Best Practices

1. **Configure Rate Limits**: Set appropriate rate limits for each channel to avoid API restrictions
2. **Monitor Queue Size**: Keep queue size reasonable to prevent memory issues
3. **Use Priorities Wisely**: Reserve CRITICAL priority for truly urgent notifications
4. **Handle Errors Gracefully**: Implement proper error handling in your application
5. **Test Integrations**: Verify all notification channels work before production deployment
6. **Monitor Statistics**: Regularly check service statistics for performance insights

## Troubleshooting

### Common Issues

1. **Notifications Not Delivered**
   - Check channel configuration and credentials
   - Verify rate limits haven't been exceeded
   - Check service logs for error messages

2. **Queue Growing**
   - Increase worker count or reduce load
   - Check for failing deliveries causing retries
   - Monitor rate limiting effects

3. **High Latency**
   - Optimize worker pool size
   - Check network connectivity to external services
   - Monitor queue processing times

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure backward compatibility

## License

MIT License - see LICENSE file for details.