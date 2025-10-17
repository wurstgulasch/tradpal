# TradPal API Gateway

## Overview

The API Gateway serves as the central entry point for all TradPal microservices, providing unified access to distributed services with advanced routing, security, and monitoring capabilities.

## Architecture

```
[Client Requests] → [API Gateway :8000]
                        ↓
               ┌─────────────────┐
               │ Service Registry │
               │ Load Balancer   │
               │ Auth Middleware │
               │ Rate Limiter    │
               └─────────────────┘
                        ↓
        ┌───────┬───────┬───────┬───────┐
        │ Core  │ Data  │ Trading│ ...  │
        │ Service│ Service│ Bot   │      │
        └───────┴───────┴───────┴───────┘
```

## Key Features

### Service Discovery & Routing
- **Dynamic Routing**: Automatic service discovery and path-based routing
- **Load Balancing**: Round-robin distribution across service instances
- **Service Registry**: Centralized service configuration management
- **Health Monitoring**: Automatic service health checking and failover

### Security & Authentication
- **JWT Authentication**: Token-based authentication for API access
- **Rate Limiting**: Per-service and per-client rate limiting
- **Request Validation**: Input validation and sanitization
- **Audit Logging**: Comprehensive request/response logging

### Monitoring & Observability
- **Prometheus Metrics**: Real-time metrics collection
- **Request Tracing**: Distributed tracing for request flows
- **Health Endpoints**: Service health status monitoring
- **Performance Metrics**: Latency, throughput, and error tracking

## Service Configuration

### Service Registration

Services are registered with the API Gateway through configuration:

```python
from services.api_gateway.main import ServiceRegistry, ServiceConfig, ServiceInstance

registry = ServiceRegistry()

# Register core service
core_service = ServiceConfig(
    name="core_service",
    path_prefix="/api/core",
    instances=[
        ServiceInstance(url="http://core_service:8002"),
        ServiceInstance(url="http://core_service:8003"),  # Load balancing
    ],
    load_balancing="round_robin",
    rate_limit=1000,  # requests per minute
    auth_required=True,
    timeout=30.0
)

registry.register_service(core_service)
```

### Service Endpoints

All services are accessible through the gateway:

```
/api/core/*     → core_service
/api/data/*     → data_service
/api/trading/*  → trading_bot_live
/api/backtest/* → backtesting_service
/api/discovery/* → discovery_service
/api/risk/*     → risk_service
/api/notifications/* → notification_service
/               → web_ui (public access)
```

## API Endpoints

### Gateway Management

- `GET /health` - Gateway health status
- `GET /services` - List registered services
- `POST /auth/login` - JWT token generation (demo)

### Metrics & Monitoring

- `GET /metrics` - Prometheus metrics endpoint
- `GET /health` - Detailed health status

## Authentication

### JWT Token Flow

1. **Login Request**:
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'
```

2. **Response**:
```json
{
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": "admin",
  "roles": ["admin", "user"]
}
```

3. **Authenticated Request**:
```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/core/signals/generate
```

## Rate Limiting

### Configuration

Rate limits are configured per service:

```python
ServiceConfig(
    name="core_service",
    rate_limit=1000,  # requests per minute per client
    # ...
)
```

### Client Identification

Clients are identified by IP address (configurable for production):

```python
def get_client_id(request: Request) -> str:
    return request.client.host
```

### Rate Limit Headers

The gateway returns standard rate limit headers:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1638360000
```

## Load Balancing

### Round-Robin Algorithm

```python
def get_service_instance(self, service_name: str) -> Optional[ServiceInstance]:
    service = self.services[service_name]
    healthy_instances = [
        instance for instance in service.instances
        if instance.status == ServiceStatus.HEALTHY
    ]

    if not healthy_instances:
        return None

    # Round-robin selection
    current_index = self._current_index[service_name]
    instance = healthy_instances[current_index % len(healthy_instances)]
    self._current_index[service_name] = (current_index + 1) % len(healthy_instances)

    return instance
```

### Health Checks

Services are monitored continuously:

```python
async def health_check_services(self):
    for service_name, service in self.services.items():
        for instance in service.instances:
            try:
                async with session.get(f"{instance.url}/health") as response:
                    if response.status == 200:
                        self.update_service_health(service_name, instance.url, ServiceStatus.HEALTHY)
                    else:
                        self.update_service_health(service_name, instance.url, ServiceStatus.UNHEALTHY)
            except Exception:
                self.update_service_health(service_name, instance.url, ServiceStatus.UNHEALTHY)
```

## Monitoring & Metrics

### Prometheus Integration

The gateway exposes comprehensive metrics:

```python
# Request metrics
REQUEST_COUNT = Counter(
    'api_gateway_requests_total',
    'Total number of requests processed',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_gateway_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

# Service health
SERVICE_HEALTH_STATUS = Gauge(
    'api_gateway_service_health_status',
    'Health status of backend services',
    ['service_name']
)

# Rate limiting
RATE_LIMIT_EXCEEDED = Counter(
    'api_gateway_rate_limit_exceeded_total',
    'Total number of rate limit violations',
    ['client_id']
)
```

### Grafana Dashboards

Pre-configured dashboards show:

- Request throughput and latency
- Service health status
- Rate limiting violations
- Error rates by endpoint
- Load balancing distribution

## Configuration

### Environment Variables

```bash
# Gateway Configuration
API_GATEWAY_HOST=0.0.0.0
API_GATEWAY_PORT=8000
PROMETHEUS_METRICS_PORT=8001

# Security
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Services
CORE_SERVICE_URL=http://core_service:8002
DATA_SERVICE_URL=http://data_service:8001
# ... other service URLs
```

### Service Discovery

For dynamic service discovery, integrate with:

- **Consul**: Service registry and health checking
- **Kubernetes**: Service discovery via DNS
- **etcd**: Distributed key-value store

## Error Handling

### HTTP Status Codes

- `200` - Success
- `401` - Authentication required
- `403` - Forbidden (invalid token)
- `404` - Service not found
- `429` - Rate limit exceeded
- `500` - Internal server error
- `502` - Bad gateway (service unavailable)
- `503` - Service unavailable
- `504` - Gateway timeout

### Error Response Format

```json
{
  "detail": "Rate limit exceeded",
  "type": "rate_limit_error",
  "timestamp": "2023-12-01T10:00:00Z"
}
```

## Development & Testing

### Local Development

Start the gateway locally:

```bash
cd services/api_gateway
python main.py
```

### Docker Development

```bash
docker-compose -f infrastructure/monitoring/docker-compose.yml up api-gateway
```

### Testing

Run gateway tests:

```bash
pytest tests/unit/services/api_gateway/
```

### Load Testing

Use tools like Artillery or k6:

```javascript
// artillery.yml
config:
  target: 'http://localhost:8000'
  phases:
    - duration: 60
      arrivalRate: 10

scenarios:
  - name: 'API Gateway Load Test'
    requests:
      - get:
          url: '/api/core/health'
```

## Troubleshooting

### Common Issues

1. **Service Unavailable (503)**
   - Check service health: `GET /health`
   - Verify service URLs in configuration
   - Check network connectivity

2. **Authentication Failed (401)**
   - Verify JWT token validity
   - Check token expiration
   - Validate token signature

3. **Rate Limit Exceeded (429)**
   - Check rate limit configuration
   - Monitor client request patterns
   - Adjust rate limits if needed

4. **High Latency**
   - Monitor downstream service performance
   - Check load balancing distribution
   - Review circuit breaker status

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('services.api_gateway').setLevel(logging.DEBUG)
```

### Health Checks

Monitor gateway health:

```bash
# Gateway health
curl http://localhost:8000/health

# Service health status
curl http://localhost:8000/health | jq '.services'
```

## Security Considerations

### Production Deployment

1. **TLS Termination**: Use reverse proxy (nginx/traefik) for TLS
2. **API Keys**: Implement proper API key management
3. **OAuth2**: Replace demo JWT with OAuth2 flows
4. **Rate Limiting**: Configure appropriate limits per client
5. **CORS**: Configure CORS policies for web clients

### Security Headers

The gateway adds security headers:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000
```

## Performance Optimization

### Caching

Implement response caching for static data:

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

FastAPICache.init(RedisBackend(redis_client), prefix="api-gateway:")
```

### Connection Pooling

Configure HTTP client connection pooling:

```python
connector = aiohttp.TCPConnector(
    limit=100,  # Max connections
    limit_per_host=10,  # Per host
    ttl_dns_cache=300  # DNS cache TTL
)
```

### Async Processing

All operations are async for high performance:

- Async request handling
- Async service communication
- Async health checks
- Async metrics collection

## Future Enhancements

### Advanced Features

1. **Service Mesh Integration**: Istio integration for advanced routing
2. **API Versioning**: Version-aware routing and backward compatibility
3. **Request Transformation**: Data transformation and protocol conversion
4. **Circuit Breaker**: Advanced circuit breaker patterns
5. **WebSocket Support**: Real-time bidirectional communication

### Monitoring Enhancements

1. **Distributed Tracing**: Jaeger/Zipkin integration
2. **Log Aggregation**: ELK stack integration
3. **Custom Metrics**: Business-specific metrics
4. **Alerting**: Advanced alerting rules

### Security Enhancements

1. **OAuth2 Integration**: Full OAuth2/OpenID Connect support
2. **API Gateway Firewall**: Web Application Firewall (WAF)
3. **Rate Limit Management**: Dynamic rate limiting
4. **Audit Logging**: Comprehensive security audit trails

This API Gateway provides a robust, scalable foundation for microservices communication in the TradPal trading system.