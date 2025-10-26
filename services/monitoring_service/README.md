# TradPal Monitoring Service

A comprehensive, consolidated monitoring service for TradPal providing enterprise-grade observability, ML operations, and automated alerting across the entire trading platform.

## Overview

The Monitoring Service is a consolidated microservice that orchestrates 5 specialized monitoring components to provide complete observability and operational intelligence for the TradPal trading platform. It combines ML operations, model monitoring, parameter optimization, alert forwarding, and multi-channel notifications into a unified monitoring ecosystem.

## Architecture

```
Monitoring Service Ecosystem
├── MLOps Service              # ML experiment tracking & model deployment
├── Discovery Service          # Genetic algorithm optimization
├── Model Monitoring Service   # ML model performance & drift detection
├── Alert Forwarder Service    # Security alert processing
└── Notification Service       # Multi-channel notifications
```

## Core Components

### 🤖 MLOps Service
**Purpose**: Complete ML lifecycle management and experiment tracking
- **MLflow Integration**: Experiment tracking and model registry
- **BentoML Deployment**: Model serving and API management
- **Drift Detection**: Automated model drift monitoring with Alibi Detect
- **REST API**: FastAPI-based endpoints for all operations
- **Port**: 8001

### 🔍 Discovery Service
**Purpose**: Automated parameter optimization using genetic algorithms
- **Genetic Algorithm Optimization**: Uses DEAP library for evolutionary computation
- **Multi-Objective Fitness**: Optimizes Sharpe ratio, profit factor, and drawdown
- **Cross-Validation**: Prevents overfitting with walk-forward analysis
- **25+ Indicator Combinations**: Comprehensive trading indicator optimization
- **Port**: 8003

### 📊 Model Monitoring Service
**Purpose**: Real-time ML model performance monitoring and drift detection
- **Drift Detection**: Population Stability Index (PSI) for continuous features
- **Performance Tracking**: MSE, RMSE, MAE, Accuracy, F1-Score, Directional Accuracy
- **Alert Management**: Multi-level alert escalation (Info → Warning → Error → Critical)
- **Automated Alert Routing**: Telegram, Email, SMS notifications
- **Port**: 8004

### 🚨 Alert Forwarder Service
**Purpose**: Automated security alert processing and forwarding
- **Falco Integration**: Real-time monitoring of Falco security logs
- **Intelligent Parsing**: Automatic detection of JSON and plain-text Falco logs
- **Priority Filtering**: Configurable minimum priority for alert forwarding
- **Alert Batching**: Groups multiple alerts for efficient notifications
- **Rate Limiting**: Protection against alert spam

### 📱 Notification Service
**Purpose**: Multi-channel notification delivery system
- **Multi-Channel Support**: Telegram, Discord, Email, SMS
- **Priority Queuing**: Critical, High, Normal, Low priority levels
- **Async Processing**: Non-blocking delivery with worker pools
- **Rate Limiting**: Per-channel rate limiting to prevent API abuse
- **Retry Logic**: Exponential backoff for failed deliveries
- **Port**: 8002

## Quick Start

### Local Development Setup

1. **Install Dependencies**
   ```bash
   # Install all monitoring service dependencies
   cd services/monitoring_service
   pip install -r requirements-all.txt  # Consolidated requirements file
   ```

2. **Configure Environment**
   ```bash
   # Copy and edit environment configuration
   cp .env.example .env
   # Edit .env with your API keys, database connections, etc.
   ```

3. **Start Individual Services**
   ```bash
   # Start MLOps Service
   python services/monitoring_service/mlops_service/service.py

   # Start Discovery Service
   python services/monitoring_service/discovery_service/service.py

   # Start Model Monitoring Service
   python services/monitoring_service/model_monitoring_service/main.py

   # Start Notification Service
   python services/monitoring_service/notification_service/service.py

   # Start Alert Forwarder Service
   python services/monitoring_service/alert_forwarder_service/forwarder.py
   ```

4. **Run Comprehensive Demo**
   ```bash
   python services/monitoring_service/demo.py
   ```

### Docker Compose Deployment

```yaml
version: '3.8'
services:
  mlops-service:
    build: ./services/monitoring_service/mlops_service
    ports:
      - "8001:8001"
    environment:
      - MLFLOW_TRACKING_URI=http://localhost:5000

  discovery-service:
    build: ./services/monitoring_service/discovery_service
    ports:
      - "8003:8003"

  model-monitoring-service:
    build: ./services/monitoring_service/model_monitoring_service
    ports:
      - "8004:8004"

  notification-service:
    build: ./services/monitoring_service/notification_service
    ports:
      - "8002:8002"
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}

  alert-forwarder-service:
    build: ./services/monitoring_service/alert_forwarder_service
    environment:
      - FALCO_LOG_PATH=/var/log/falco/falco.log
```

### Kubernetes Deployment

```bash
# Deploy all monitoring services
kubectl apply -f services/monitoring_service/k8s/

# Check deployment status
kubectl get pods -l app=monitoring-service

# View logs
kubectl logs -l app=monitoring-service
```

## API Endpoints

### MLOps Service (Port 8001)
- `GET /health` - Service health check
- `GET /stats` - Service statistics
- `POST /experiments/log` - Log ML experiment
- `GET /experiments/history` - Get experiment history
- `POST /models/deploy` - Deploy model with BentoML
- `GET /models/{model_name}/versions` - Get model versions
- `POST /drift-detectors/create` - Create drift detector
- `GET /drift-detectors` - List active drift detectors

### Discovery Service (Port 8003)
- `GET /health` - Service health check
- `POST /optimize/start` - Start parameter optimization
- `GET /optimize/status/{optimization_id}` - Get optimization status
- `GET /optimize/active` - List active optimizations
- `POST /optimize/cancel/{optimization_id}` - Cancel optimization

### Model Monitoring Service (Port 8004)
- `GET /health` - Service health check
- `POST /register` - Register model for monitoring
- `GET /models` - List registered models
- `DELETE /models/{model_id}` - Remove model from monitoring
- `POST /monitor` - Monitor model prediction
- `GET /status/{model_id}` - Get model status
- `GET /alerts/{model_id}` - Get model alerts

### Notification Service (Port 8002)
- `GET /health` - Service health check
- `POST /notifications/send` - Send notification
- `POST /notifications/signal` - Send trading signal notification
- `GET /queue/status` - Get queue status
- `GET /statistics` - Get service statistics

### Alert Forwarder Service
- Runs as background service, no direct API endpoints
- Integrates with Falco logs and notification service

## Configuration

### Environment Variables

```bash
# Database Configuration
MONITORING_DB_HOST=localhost
MONITORING_DB_PORT=5432
MONITORING_DB_NAME=tradpal_monitoring
MONITORING_DB_USER=tradpal
MONITORING_DB_PASSWORD=your_password

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_ARTIFACT_STORE=s3://your-bucket/mlflow

# Notification Service
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
DISCORD_WEBHOOK_URL=your_discord_webhook
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Alert Forwarder
FALCO_LOG_PATH=/var/log/falco/falco.log
ALERT_MIN_PRIORITY=WARNING
ALERT_BATCHING=true
ALERT_BATCH_INTERVAL=300
ALERT_RATE_LIMIT=10

# Model Monitoring
MODEL_DRIFT_THRESHOLD=0.1
MODEL_PERFORMANCE_DEGRADATION_THRESHOLD=0.2
ALERT_COOLDOWN_MINUTES=30

# Discovery Service
OPTIMIZATION_POPULATION_SIZE=50
OPTIMIZATION_GENERATIONS=20
OPTIMIZATION_CROSSOVER_RATE=0.8
OPTIMIZATION_MUTATION_RATE=0.1
```

### Service-Specific Configuration

Each service has its own configuration file:

- `mlops_service/config.py` - MLflow and BentoML settings
- `discovery_service/config.py` - Genetic algorithm parameters
- `model_monitoring_service/config.py` - Drift detection thresholds
- `notification_service/config.py` - Channel configurations
- `alert_forwarder_service/config.py` - Falco integration settings

## Monitoring & Observability

### Health Checks
All services provide comprehensive health endpoints:

```bash
# Check all service health
curl http://localhost:8001/health  # MLOps
curl http://localhost:8002/health  # Notifications
curl http://localhost:8003/health  # Discovery
curl http://localhost:8004/health  # Model Monitoring
```

### Metrics Collection
Services expose Prometheus metrics:

```bash
# MLOps metrics
curl http://localhost:8001/metrics

# Notification metrics
curl http://localhost:8002/metrics

# Model monitoring metrics
curl http://localhost:8004/metrics
```

### Grafana Dashboards
Pre-configured dashboards for:
- ML Experiment Tracking
- Model Performance Monitoring
- Alert Management
- Notification Delivery Statistics
- Optimization Progress

## ML Operations Workflow

### 1. Experiment Tracking
```python
from services.monitoring_service.mlops_service.client import MLOpsClient

client = MLOpsClient("http://localhost:8001")

# Log experiment
await client.log_experiment({
    "experiment_name": "btc_trading_model_v1",
    "parameters": {"learning_rate": 0.001, "epochs": 100},
    "metrics": {"accuracy": 0.85, "loss": 0.23},
    "model_path": "/path/to/model.pkl"
})
```

### 2. Model Deployment
```python
# Deploy model
deployment_id = await client.deploy_model({
    "model_name": "btc_trading_model",
    "model_version": "v1.0",
    "model_path": "/path/to/model.pkl"
})
```

### 3. Model Monitoring Setup
```python
from services.monitoring_service.model_monitoring_service.client import ModelMonitoringClient

monitoring_client = ModelMonitoringClient("http://localhost:8004")

# Register model for monitoring
await monitoring_client.register_model({
    "model_id": "btc_trading_model_v1",
    "baseline_features": {...},
    "baseline_metrics": {...},
    "drift_threshold": 0.1
})
```

### 4. Parameter Optimization
```python
from services.monitoring_service.discovery_service.client import DiscoveryClient

discovery_client = DiscoveryClient("http://localhost:8003")

# Start optimization
optimization_id = await discovery_client.start_optimization({
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "population_size": 50,
    "generations": 20
})
```

## Alert Management

### Automated Alert Routing
```python
from services.monitoring_service.notification_service.client import NotificationServiceClient

notification_client = NotificationServiceClient("http://localhost:8002")

# Send critical alert
await notification_client.send_alert({
    "message": "Model drift detected on BTC trading model",
    "level": "critical",
    "channels": ["telegram", "email"],
    "data": {
        "model_id": "btc_trading_model_v1",
        "drift_score": 0.15,
        "threshold": 0.1
    }
})
```

### Alert Forwarder Integration
The alert forwarder automatically processes Falco security alerts:

```python
# Configuration in .env
FALCO_LOG_PATH=/var/log/falco/falco.log
ALERT_MIN_PRIORITY=WARNING
TELEGRAM_ENABLED=true
DISCORD_ENABLED=true
```

## Testing

### Unit Tests
```bash
# Test individual services
pytest services/monitoring_service/mlops_service/tests.py
pytest services/monitoring_service/discovery_service/tests.py
pytest services/monitoring_service/model_monitoring_service/tests/
pytest services/monitoring_service/notification_service/tests.py
pytest services/monitoring_service/alert_forwarder_service/tests.py
```

### Integration Tests
```bash
# Test service interactions
pytest tests/integration/monitoring_services/
```

### Performance Tests
```bash
# Load testing
pytest tests/performance/monitoring_services/ -m performance
```

## Demo Scripts

### Comprehensive Monitoring Demo
```bash
python services/monitoring_service/demo.py
```

This demo showcases:
- ML experiment logging and model deployment
- Model monitoring setup and drift detection
- Parameter optimization with genetic algorithms
- Alert generation and notification delivery
- Cross-service integration and data flow

### Individual Service Demos
```bash
# MLOps demo
python services/monitoring_service/mlops_service/demo.py

# Discovery demo
python services/monitoring_service/discovery_service/demo.py

# Model monitoring demo
python services/monitoring_service/model_monitoring_service/test_monitoring.py

# Notification demo
python services/monitoring_service/notification_service/demo.py
```

## Docker Development

### Build All Services
```bash
# Build all monitoring service images
make build-all

# Or build individually
make build-mlops
make build-discovery
make build-model-monitoring
make build-notification
make build-alert-forwarder
```

### Development Environment
```bash
# Start development stack
make dev-up

# View logs
make logs

# Run tests in containers
make test
```

## Kubernetes Production Deployment

### Complete Monitoring Stack
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: monitoring-stack
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: mlops-service
        image: tradpal/mlops-service:latest
      - name: discovery-service
        image: tradpal/discovery-service:latest
      - name: model-monitoring-service
        image: tradpal/model-monitoring-service:latest
      - name: notification-service
        image: tradpal/notification-service:latest
      - name: alert-forwarder-service
        image: tradpal/alert-forwarder-service:latest
```

### Service Mesh Configuration
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: monitoring-gateway
spec:
  gateways:
  - monitoring-gateway
  http:
  - match:
    - uri:
        prefix: "/mlops"
    route:
    - destination:
        host: mlops-service
  - match:
    - uri:
        prefix: "/discovery"
    route:
    - destination:
        host: discovery-service
```

## Troubleshooting

### Common Issues

1. **Service Connectivity**
   - Check service ports are not conflicting
   - Verify network policies in Kubernetes
   - Check service discovery configuration

2. **MLflow Connection Issues**
   - Verify MLFLOW_TRACKING_URI is correct
   - Check artifact store permissions
   - Ensure MLflow server is running

3. **Alert Delivery Failures**
   - Verify notification service API keys
   - Check rate limits on external services
   - Review alert forwarder log parsing

4. **Model Drift False Positives**
   - Adjust drift detection thresholds
   - Review baseline data quality
   - Check feature engineering consistency

### Debug Mode
Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
export MONITORING_DEBUG=true
```

### Health Monitoring
```bash
# Check all services
curl -s http://localhost:8001/health && echo " - MLOps: OK"
curl -s http://localhost:8002/health && echo " - Notifications: OK"
curl -s http://localhost:8003/health && echo " - Discovery: OK"
curl -s http://localhost:8004/health && echo " - Model Monitoring: OK"
```

## Performance Optimization

### Caching Strategies
- Redis caching for frequently accessed metrics
- In-memory caching for real-time monitoring data
- Distributed caching for multi-instance deployments

### Resource Management
- Horizontal pod autoscaling based on CPU/memory usage
- Resource limits and requests for predictable performance
- Connection pooling for database and external API calls

### Monitoring Overhead
- Configurable sampling rates for high-frequency metrics
- Asynchronous processing for non-critical monitoring tasks
- Batch processing for bulk metric collection

## Security Considerations

### Authentication & Authorization
- JWT-based authentication for API access
- Role-based access control (RBAC)
- API key management for external integrations

### Data Protection
- Encryption at rest for sensitive monitoring data
- Secure communication between services (mTLS)
- Audit logging for all monitoring operations

### Alert Security
- Alert payload validation and sanitization
- Rate limiting to prevent alert spam attacks
- Secure webhook endpoints for external notifications

## Contributing

1. **Code Standards**: Follow existing patterns and async-first design
2. **Testing**: Add comprehensive tests for new features
3. **Documentation**: Update READMEs and API documentation
4. **Security**: Implement security best practices for new features

## Future Enhancements

- **Advanced Analytics**: Predictive monitoring and anomaly detection
- **Custom Dashboards**: User-configurable monitoring dashboards
- **Alert Correlation**: Intelligent alert grouping and root cause analysis
- **Multi-Cloud Support**: Cross-cloud monitoring and alerting
- **Edge Computing**: Distributed monitoring for edge deployments

## License

MIT License - see LICENSE file for details.</content>
<parameter name="filePath">/Users/danielsadowski/VSCodeProjects/tradpal/tradpal/services/monitoring_service/README.md