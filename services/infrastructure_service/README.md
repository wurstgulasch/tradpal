# TradPal Infrastructure Service

## Overview

The **Infrastructure Service** is the foundational layer of TradPal's microservices architecture, providing essential platform services that enable secure, resilient, and observable communication between all microservices.

This consolidated service follows the **moderate isolation pattern** - combining related infrastructure components while maintaining runtime separation for resource management and fault isolation.

## Architecture

### Service Components

The infrastructure service consolidates 5 critical platform services:

#### 1. API Gateway Service (`api_gateway_service/`)
**Purpose**: Central entry point for all microservice communication
**Port**: 8000
**Features**:
- Service discovery and load balancing
- Authentication and authorization (JWT)
- Rate limiting and request throttling
- Request/response transformation
- Centralized monitoring and logging

#### 2. Event System Service (`event_system_service/`)
**Purpose**: Event-driven communication backbone using Redis Streams
**Port**: 8011
**Features**:
- Publish-subscribe pattern with guaranteed delivery
- Event persistence and historical replay
- Consumer groups for load balancing
- REST API for event management
- Prometheus metrics and monitoring

#### 3. Security Service (`security_service/`)
**Purpose**: Zero-trust security with mTLS and JWT authentication
**Port**: 8009
**Features**:
- mTLS certificate management and CA
- JWT token generation and validation
- Secrets management (Vault integration)
- Service credential issuance
- Security monitoring and audit

#### 4. Circuit Breaker Service (`circuit_breaker_service/`)
**Purpose**: Fault tolerance and resilience patterns
**Port**: 8012
**Features**:
- Circuit breaker state management
- Failure detection and recovery
- Service health monitoring
- Metrics collection and alerting
- Configuration management

#### 5. Falco Security Service (`falco_security_service/`)
**Purpose**: Runtime security monitoring and threat detection
**Specialized Tool**: Uses Falco for container runtime security
**Features**:
- Container activity monitoring
- Filesystem and network security
- Trading-specific security rules
- Real-time alerting and notifications

### Moderate Isolation Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    Infrastructure Service                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ API Gateway │ │ Event       │ │ Security    │           │
│  │  (Port 8000)│ │ System      │ │ Service     │           │
│  │             │ │ (Port 8011) │ │ (Port 8009) │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│                                                             │
│  ┌─────────────┐ ┌─────────────┐                           │
│  │ Circuit     │ │ Falco       │                           │
│  │ Breaker     │ │ Security    │                           │
│  │ (Port 8012) │ │ (Monitoring)│                           │
│  └─────────────┘ └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

**Benefits**:
- **Code Consolidation**: Shared infrastructure components
- **Runtime Separation**: Independent processes for fault isolation
- **Resource Management**: CPU affinity and memory limits per service
- **Unified Deployment**: Single service with multiple endpoints

## Core Features

### 🔐 Zero-Trust Security
- **mTLS Authentication**: Mutual TLS for all service-to-service communication
- **JWT Tokens**: Bearer token authentication through API Gateway
- **Certificate Authority**: Automated certificate issuance and renewal
- **Secrets Management**: Secure storage with HashiCorp Vault integration

### 🌐 API Gateway
- **Load Balancing**: Round-robin, least-connections, and weighted algorithms
- **Rate Limiting**: Configurable request limits per service and client
- **Request Routing**: Path-based routing to backend services
- **Authentication**: JWT validation and user authorization

### 📡 Event-Driven Communication
- **Redis Streams**: Persistent event storage with consumer groups
- **Event Types**: 25+ predefined event types for trading operations
- **Replay Capability**: Historical event replay for debugging and backtesting
- **Monitoring**: Event throughput and delivery metrics

### 🛡️ Resilience Patterns
- **Circuit Breakers**: Automatic failure detection and recovery
- **Health Checks**: Continuous service health monitoring
- **Retry Logic**: Exponential backoff for transient failures
- **Fallback Mechanisms**: Graceful degradation during outages

### 👁️ Runtime Security Monitoring
- **Container Security**: Falco-based runtime threat detection
- **Trading Rules**: Specialized rules for financial data protection
- **Alert Integration**: Real-time alerts to notification services
- **Audit Logging**: Comprehensive security event logging

## Service Endpoints

### API Gateway (Port 8000)
```bash
GET  /health                    # Health check
GET  /services                  # List registered services
POST /auth/login               # Authentication endpoint
GET  /metrics                  # Prometheus metrics
# All other requests routed to backend services
```

### Event System (Port 8011)
```bash
GET  /health                   # Health check
GET  /metrics                  # Prometheus metrics
POST /events                   # Publish custom events
GET  /events                   # Retrieve events with filtering
POST /events/replay            # Replay historical events
POST /events/market-data       # Publish market data
POST /events/trading-signal    # Publish trading signals
GET  /events/stats             # Stream statistics
```

### Security Service (Port 8009)
```bash
GET  /health                   # Health check
GET  /metrics                  # Prometheus metrics
POST /credentials/{service}    # Issue mTLS credentials
GET  /credentials/{service}    # Get service credentials
POST /tokens/{service}         # Generate JWT tokens
POST /tokens/validate          # Validate JWT tokens
PUT  /secrets/{path}           # Store secrets
GET  /secrets/{path}           # Retrieve secrets
GET  /ca/certificate           # Get CA certificate
```

### Circuit Breaker (Port 8012)
```bash
GET  /health                   # Health check
GET  /metrics                  # Prometheus metrics
GET  /circuit-breakers         # List all breakers
GET  /circuit-breakers/{name}  # Get breaker info
POST /circuit-breakers/{name}/reset  # Reset breaker
GET  /dashboard                # Dashboard data
GET  /alerts                   # Active alerts
```

## Configuration

### Environment Variables

```bash
# API Gateway
API_GATEWAY_PORT=8000
JWT_SECRET_KEY=your-secret-key
RATE_LIMIT_DEFAULT=1000

# Event System
EVENT_SERVICE_PORT=8011
REDIS_URL=redis://localhost:6379
EVENT_STREAM_NAME=tradpal_events

# Security Service
SECURITY_SERVICE_PORT=8009
ENABLE_MTLS=true
ENABLE_JWT=true
ENABLE_VAULT=false

# Circuit Breaker
CIRCUIT_BREAKER_PORT=8012
FAILURE_THRESHOLD=5
RECOVERY_TIMEOUT=60

# Shared
LOG_LEVEL=INFO
METRICS_ENABLED=true
```

### Service Registration

Services are automatically registered in the API Gateway:

```python
# Example service registration
core_service = ServiceConfig(
    name="core_service",
    path_prefix="/api/core",
    instances=[ServiceInstance(url="http://core_service:8002")],
    rate_limit=1000,
    auth_required=True
)
service_registry.register_service(core_service)
```

## Security Implementation

### mTLS Setup
1. **Certificate Authority**: Auto-generated on first startup
2. **Service Certificates**: Issued per service with unique identities
3. **Mutual Authentication**: Both client and server verify certificates
4. **Certificate Rotation**: Automated renewal before expiration

### JWT Authentication
1. **Token Generation**: Services request tokens from Security Service
2. **Token Validation**: API Gateway validates all incoming requests
3. **Permission Scoping**: Role-based access control per service
4. **Token Expiration**: Automatic cleanup of expired tokens

### Secrets Management
1. **Vault Integration**: Optional HashiCorp Vault for production
2. **Local Storage**: File-based storage for development
3. **Access Control**: Service-specific secret isolation
4. **Audit Logging**: All secret access is logged

## Monitoring & Observability

### Metrics Collection
- **Prometheus Integration**: All services expose `/metrics` endpoints
- **Custom Metrics**: Service-specific performance indicators
- **Health Checks**: Automated health monitoring every 30 seconds
- **Alert Rules**: Pre-configured alerting for critical conditions

### Logging
- **Structured Logging**: JSON format with correlation IDs
- **Log Levels**: Configurable verbosity (DEBUG, INFO, WARNING, ERROR)
- **Central Aggregation**: All logs collected for analysis
- **Security Events**: Dedicated security event logging

### Tracing
- **Request Tracing**: End-to-end request tracking
- **Service Dependencies**: Call graph visualization
- **Performance Monitoring**: Latency and throughput metrics
- **Error Correlation**: Error tracing across services

## Usage Examples

### Service Authentication

```python
from services.infrastructure_service.security_service.client import SecurityClient

# Initialize security client
security = SecurityClient()

# Issue mTLS credentials
creds = await security.issue_credentials("trading_service")

# Generate JWT token
token = await security.generate_token("trading_service", ["read", "trade"])

# Use in API calls
headers = {"Authorization": f"Bearer {token}"}
```

### Event Publishing

```python
from services.infrastructure_service.event_system_service.client import EventClient

# Initialize event client
events = EventClient()

# Publish trading signal
await events.publish_trading_signal({
    "symbol": "BTC/USDT",
    "action": "BUY",
    "confidence": 0.85
})

# Subscribe to events
subscriber = EventSubscriberClient(events)
await subscriber.register_handler(EventType.MARKET_DATA_UPDATE, handle_market_data)
await subscriber.start_subscribing()
```

### Circuit Breaker Monitoring

```python
from services.infrastructure_service.circuit_breaker_service.client import CircuitBreakerClient

# Initialize circuit breaker client
cb_client = CircuitBreakerClient()

# Check breaker status
status = await cb_client.get_breaker_status("data_service")

# Reset breaker if needed
if status["state"] == "open":
    await cb_client.reset_breaker("data_service")
```

## Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  infrastructure:
    build: ./services/infrastructure_service
    ports:
      - "8000:8000"   # API Gateway
      - "8009:8009"   # Security Service
      - "8011:8011"   # Event System
      - "8012:8012"   # Circuit Breaker
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: infrastructure-service
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: api-gateway
        image: tradpal/infrastructure:latest
        ports:
        - containerPort: 8000
        command: ["python", "-m", "api_gateway_service.main"]
      - name: event-system
        image: tradpal/infrastructure:latest
        ports:
        - containerPort: 8011
        command: ["python", "-m", "event_system_service.main"]
      # ... additional containers for other services
```

## Development

### Running Individual Services

```bash
# API Gateway
python -m services.infrastructure_service.api_gateway_service.main

# Event System
python -m services.infrastructure_service.event_system_service.main

# Security Service
python -m services.infrastructure_service.security_service.main

# Circuit Breaker
python -m services.infrastructure_service.circuit_breaker_service.main
```

### Testing

```bash
# Run all infrastructure service tests
pytest services/infrastructure_service/ -v

# Test individual services
pytest services/infrastructure_service/api_gateway_service/tests.py
pytest services/infrastructure_service/event_system_service/tests.py
pytest services/infrastructure_service/security_service/tests.py
```

## Troubleshooting

### Common Issues

#### API Gateway Connection Refused
```
Error: Connection refused on port 8000
```
**Solution**: Ensure API Gateway service is running and accessible

#### mTLS Certificate Errors
```
Error: Certificate verification failed
```
**Solution**: Check certificate validity and CA trust store

#### Event Stream Full
```
Error: Stream length exceeded maximum
```
**Solution**: Configure Redis stream trimming or increase max length

#### Circuit Breaker Stuck Open
```
Breaker state: OPEN (not recovering)
```
**Solution**: Manually reset breaker or check service health

### Debug Commands

```bash
# Check service health
curl http://localhost:8000/health
curl http://localhost:8009/health
curl http://localhost:8011/health
curl http://localhost:8012/health

# View event stream
redis-cli XINFO STREAM tradpal_events

# Check circuit breaker status
curl http://localhost:8012/circuit-breakers

# View security service logs
docker logs infrastructure_security_1
```

## Future Enhancements

### Advanced Security
- **SPIFFE/SPIRE Integration**: Automated identity management
- **OAuth 2.0**: Enhanced authentication flows
- **API Security**: OWASP compliance and vulnerability scanning

### Performance Optimization
- **Service Mesh**: Istio integration for advanced routing
- **Caching Layer**: Redis caching for frequently accessed data
- **Load Balancing**: Advanced algorithms and traffic shaping

### Observability
- **Distributed Tracing**: Jaeger/OpenTelemetry integration
- **Log Aggregation**: ELK stack for centralized logging
- **Custom Dashboards**: Grafana dashboards for infrastructure monitoring

---

**Service Status**: ✅ **Fully Consolidated**
**Ports**: 8000 (Gateway), 8009 (Security), 8011 (Events), 8012 (Circuit Breaker)
**Dependencies**: Redis, FastAPI, Cryptography, Prometheus
**Security**: Zero-Trust with mTLS + JWT</content>
<parameter name="filePath">/Users/danielsadowski/VSCodeProjects/tradpal/tradpal/services/infrastructure_service/README.md