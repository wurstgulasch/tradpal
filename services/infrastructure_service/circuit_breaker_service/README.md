# TradPal Circuit Breaker Service

## Overview

The Circuit Breaker Service provides resilience patterns for TradPal microservices architecture. It implements circuit breaker patterns to prevent cascading failures and enable automatic recovery from service outages.

## Features

- **Circuit Breaker Pattern**: Automatic failure detection and recovery
- **HTTP Circuit Breakers**: Specialized for HTTP service communication
- **Prometheus Metrics**: Comprehensive monitoring and alerting
- **REST API**: Management and monitoring endpoints
- **Async Support**: Full asyncio compatibility
- **Configurable Thresholds**: Customizable failure thresholds and recovery timeouts

## Architecture

### Circuit Breaker States

- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Failure threshold exceeded, requests blocked
- **HALF_OPEN**: Testing recovery, limited requests allowed

### Components

- `CircuitBreaker`: Core circuit breaker implementation
- `HttpCircuitBreaker`: HTTP-specific circuit breaker with retry logic
- `CircuitBreakerRegistry`: Centralized management of all circuit breakers
- `CircuitBreakerServiceClient`: Async client for service communication

## Configuration

### Environment Variables

```bash
# Circuit Breaker Service
CIRCUIT_BREAKER_SERVICE_URL=http://localhost:8012
CIRCUIT_BREAKER_SERVICE_TIMEOUT=30.0

# Default Circuit Breaker Settings
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60.0
CIRCUIT_BREAKER_SUCCESS_THRESHOLD=3
```

### Circuit Breaker Configuration

```python
from services.infrastructure_service.circuit_breaker_service import CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=60.0,    # Wait 60 seconds before testing
    success_threshold=3,      # Close after 3 successes in half-open
    timeout=30.0,            # Request timeout
    name="my_service"        # Circuit breaker name
)
```

## Usage

### Basic Circuit Breaker

```python
from services.infrastructure_service.circuit_breaker_service import get_circuit_breaker

# Get or create circuit breaker
breaker = get_circuit_breaker("api_gateway")

# Use in async context
try:
    result = await breaker.call(async_function, arg1, arg2)
except Exception as e:
    print(f"Request failed: {e}")
```

### HTTP Circuit Breaker

```python
from services.infrastructure_service.circuit_breaker_service import get_http_circuit_breaker

# Get HTTP circuit breaker
http_breaker = get_http_circuit_breaker("data_service")

# Make HTTP request
async with aiohttp.ClientSession() as session:
    try:
        result = await http_breaker.get(session, "http://data-service:8001/data")
        data = await result.json()
    except Exception as e:
        print(f"HTTP request failed: {e}")
```

### Service Client Integration

```python
from services.infrastructure_service.circuit_breaker_service.client import circuit_breaker_client

# Check circuit breaker status
status = await circuit_breaker_client.get_circuit_breaker_info("trading_service")

# Reset circuit breaker
await circuit_breaker_client.reset_circuit_breaker("trading_service")

# Monitor all circuit breakers
dashboard = await circuit_breaker_client.get_dashboard_data()
```

## API Endpoints

### Health Check
```
GET /health
```

### Metrics
```
GET /metrics  # Prometheus metrics
```

### Circuit Breaker Management
```
GET    /circuit-breakers              # List all breakers
GET    /circuit-breakers/{name}       # Get breaker info
GET    /circuit-breakers/{name}/state # Get breaker state
POST   /circuit-breakers/{name}/reset # Reset breaker
POST   /circuit-breakers/reset-all    # Reset all breakers
PUT    /circuit-breakers/{name}/config # Update config
```

### Monitoring
```
GET /dashboard  # Dashboard data
GET /alerts     # Active alerts
```

## Monitoring

### Prometheus Metrics

The service exposes the following metrics:

- `circuit_breaker_requests_total{name, state}`: Total requests
- `circuit_breaker_failures_total{name}`: Total failures
- `circuit_breaker_successes_total{name}`: Total successes
- `circuit_breaker_state{name, state}`: Current state (0=closed, 1=open, 2=half_open)

### Grafana Dashboard

Import the provided dashboard configuration for circuit breaker monitoring.

## Integration Examples

### Service Client with Circuit Breaker

```python
from services.infrastructure_service.circuit_breaker_service import get_http_circuit_breaker
from services.infrastructure_service.circuit_breaker_service.client import circuit_breaker_client

class TradingServiceClient:
    def __init__(self):
        self.http_breaker = get_http_circuit_breaker("trading_service")

    async def get_portfolio(self, user_id: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            try:
                result = await self.http_breaker.get(
                    session,
                    f"http://trading-service:8002/portfolio/{user_id}"
                )
                return await result.json()
            except Exception as e:
                # Check if circuit breaker is open
                is_open = await circuit_breaker_client.is_circuit_breaker_open("trading_service")
                if is_open:
                    raise HTTPException(503, "Trading service temporarily unavailable")
                raise
```

### Event-Driven Circuit Breaker Reset

```python
from services.event_system_service.client import event_system_client
from services.infrastructure_service.circuit_breaker_service.client import circuit_breaker_client

async def handle_service_recovery(service_name: str):
    """Handle service recovery events"""
    await circuit_breaker_client.reset_circuit_breaker(service_name)
    logger.info(f"Reset circuit breaker for {service_name} due to service recovery")

# Subscribe to service recovery events
await event_system_client.subscribe("service.recovery", handle_service_recovery)
```

## Testing

### Unit Tests

```bash
# Run circuit breaker tests
pytest tests/services/infrastructure_service/circuit_breaker_service/
```

### Integration Tests

```python
import pytest
from services.infrastructure_service.circuit_breaker_service import get_circuit_breaker

@pytest.mark.asyncio
async def test_circuit_breaker_failure_handling():
    breaker = get_circuit_breaker("test_service")

    # Simulate failures
    for _ in range(6):  # Exceed threshold
        try:
            await breaker.call(failing_function)
        except:
            pass

    # Should be open
    assert breaker.get_state() == CircuitBreakerState.OPEN

    # Reset and test recovery
    await breaker._transition_to_half_open()
    await breaker.call(successful_function)  # Should succeed
    assert breaker.get_state() == CircuitBreakerState.CLOSED
```

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8012

CMD ["python", "services/infrastructure_service/circuit_breaker_service/main.py"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  circuit_breaker_service:
    build: .
    ports:
      - "8012:8012"
    environment:
      - CIRCUIT_BREAKER_SERVICE_URL=http://localhost:8012
    depends_on:
      - redis
```

## Troubleshooting

### Common Issues

1. **Circuit Breaker Stuck Open**
   - Check service health: `GET /health`
   - Reset manually: `POST /circuit-breakers/{name}/reset`
   - Check failure threshold configuration

2. **High False Positives**
   - Increase `failure_threshold`
   - Adjust `recovery_timeout`
   - Check network connectivity

3. **Slow Recovery**
   - Decrease `recovery_timeout`
   - Increase `success_threshold`
   - Monitor service recovery events

### Logs

```bash
# View service logs
docker logs circuit_breaker_service

# Check circuit breaker state changes
grep "Circuit breaker" logs/tradpal.log
```

### Metrics Analysis

```bash
# Query Prometheus for failure rate
rate(circuit_breaker_failures_total[5m]) / rate(circuit_breaker_requests_total[5m])
```

## Dependencies

- `aiohttp`: Async HTTP client
- `prometheus-client`: Metrics collection
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `redis`: Event system integration

## Contributing

1. Follow the established service pattern
2. Add comprehensive tests
3. Update documentation
4. Ensure async-first design
5. Add Prometheus metrics for new features

## License

See project LICENSE file.