# TradPal Event-Driven Architecture

## Overview

The Event-Driven Architecture enables real-time, asynchronous communication between TradPal microservices using Redis Streams. This architecture provides loose coupling, scalability, and resilience for the distributed trading system.

## Architecture

```
[Services] → [Event Publisher] → [Redis Streams] → [Event Subscribers] → [Services]
                    ↑                                              ↓
              [Event Store] ← [Event Replay] ← [Monitoring] ← [Event Clients]
```

## Key Components

### 1. Event System (`services/event_system/__init__.py`)

Core event system providing publish-subscribe functionality:

- **EventPublisher**: Publishes events to Redis streams
- **EventSubscriber**: Subscribes to events with consumer groups
- **EventStore**: Provides event persistence and replay
- **EventSystem**: Main coordinator for event operations

### 2. Event Service (`services/event_system/main.py`)

REST API service for event management:

- Event publishing via HTTP endpoints
- Event retrieval and filtering
- Event statistics and monitoring
- Prometheus metrics integration

### 3. Event Client (`services/event_system/client.py`)

Client library for service integration:

- Easy-to-use methods for publishing events
- Subscription client for consuming events
- Convenience functions for common event types

## Event Types

```python
class EventType(Enum):
    MARKET_DATA_UPDATE = "market_data_update"
    TRADING_SIGNAL = "trading_signal"
    ORDER_EXECUTED = "order_executed"
    PORTFOLIO_UPDATE = "portfolio_update"
    RISK_ALERT = "risk_alert"
    SYSTEM_HEALTH = "system_health"
    ML_MODEL_UPDATE = "ml_model_update"
    BACKTEST_COMPLETE = "backtest_complete"
```

## Usage Examples

### Publishing Events

```python
from services.event_system.client import EventClient

async with EventClient() as client:
    # Publish market data
    await client.publish_market_data("BTC/USDT", {
        "price": 45000.0,
        "volume": 1234.56
    })

    # Publish trading signal
    await client.publish_trading_signal({
        "symbol": "BTC/USDT",
        "action": "BUY",
        "confidence": 0.85
    })
```

### Subscribing to Events

```python
from services.event_system.client import EventSubscriberClient

subscriber = EventSubscriberClient(event_client)

async def handle_signal(event):
    print(f"Received signal: {event.data}")

subscriber.register_handler(EventType.TRADING_SIGNAL, handle_signal)
await subscriber.start_subscribing()
```

### Service Integration

Services can integrate events into their workflows:

```python
# In a service client
async def generate_signals(self, symbol, data):
    signals = await self._call_service("generate_signals", data)

    # Publish event for each signal
    for signal in signals:
        await self.event_client.publish_trading_signal({
            "symbol": symbol,
            "signal": signal
        })

    return signals
```

## Docker Integration

The event system is integrated into the monitoring stack:

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  event-service:
    build: ./services/event_system
    ports:
      - "8011:8011"
    depends_on:
      - redis
```

## API Endpoints

### Event Service API

- `POST /events` - Publish custom event
- `GET /events` - Retrieve events with filtering
- `POST /events/replay` - Replay historical events
- `GET /events/stats` - Stream statistics
- `GET /metrics` - Prometheus metrics

### Convenience Endpoints

- `POST /events/market-data` - Publish market data
- `POST /events/trading-signal` - Publish trading signal
- `POST /events/portfolio-update` - Publish portfolio update

## Event Persistence

Events are persisted in Redis Streams with:

- **Unlimited retention**: Events remain available for replay
- **Consumer groups**: Multiple subscribers can consume events
- **Message IDs**: Unique identifiers for each event
- **Timestamps**: Precise event timing

## Monitoring & Observability

### Prometheus Metrics

- `event_service_events_published_total` - Events published by type
- `event_service_events_consumed_total` - Events consumed by type
- `event_service_active_subscribers` - Active subscribers
- `event_service_stream_length` - Current stream length

### Health Checks

- Service health endpoint: `GET /health`
- Circuit breaker integration
- Redis connection monitoring

## Benefits

### 1. Loose Coupling
Services communicate through events rather than direct API calls, reducing dependencies.

### 2. Scalability
Event-driven architecture scales horizontally - add more subscribers as needed.

### 3. Resilience
Services continue operating even if other services are unavailable.

### 4. Real-time Communication
Immediate event propagation enables real-time trading decisions.

### 5. Audit Trail
Complete event history for debugging and compliance.

### 6. Event Replay
Reprocess historical events for backtesting or system recovery.

## Configuration

### Environment Variables

```bash
# Redis configuration
REDIS_URL=redis://localhost:6379

# Event service configuration
EVENT_SERVICE_URL=http://event_service:8011
```

### Service Registration

Services register with the event system during initialization:

```python
# In service startup
await event_system.initialize()
```

## Testing

Run the event system tests:

```bash
pytest tests/unit/services/event_system/
```

Run the demo:

```bash
python examples/event_driven_trading_demo.py
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check Redis service is running
   - Verify connection string

2. **Events Not Received**
   - Check consumer group status
   - Verify event handler registration

3. **High Memory Usage**
   - Monitor stream length
   - Implement event cleanup policies

### Debugging

Enable debug logging:

```python
import logging
logging.getLogger('services.event_system').setLevel(logging.DEBUG)
```

Check event stream:

```python
stats = await event_client.get_event_stats()
print(f"Stream length: {stats['stream_length']}")
```

## Future Enhancements

1. **Event Schema Validation** - JSON Schema validation for events
2. **Event Routing** - Advanced routing rules and filters
3. **Event Aggregation** - Combine related events
4. **Event Archiving** - Move old events to persistent storage
5. **Event Analytics** - Real-time event stream analytics

## Integration Examples

### Core Service Integration

```python
# services/core/client.py
async def generate_signals(self, symbol, data):
    signals = await self._generate_signals_internal(symbol, data)

    # Publish events
    for signal in signals:
        await self.event_client.publish_trading_signal({
            "symbol": symbol,
            "signal": signal
        })

    return signals
```

### Trading Bot Integration

```python
# services/trading_bot_live/client.py
async def execute_trade(self, signal):
    result = await self._execute_trade_internal(signal)

    # Publish execution event
    await self.event_client.publish_event({
        "event_type": "order_executed",
        "data": {
            "symbol": signal["symbol"],
            "execution": result
        }
    })

    return result
```

This event-driven architecture provides the foundation for scalable, resilient microservices communication in the TradPal trading system.