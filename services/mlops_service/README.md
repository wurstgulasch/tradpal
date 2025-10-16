# MLOps Service

A comprehensive MLOps service for TradPal providing experiment tracking, model deployment, and drift detection capabilities.

## Features

- **MLflow Integration**: Experiment tracking and model registry
- **BentoML Deployment**: Model serving and API management
- **Drift Detection**: Automated model drift monitoring with Alibi Detect
- **REST API**: FastAPI-based endpoints for all operations
- **Async Processing**: Non-blocking operations with background tasks
- **Health Monitoring**: Comprehensive service health checks
- **Docker & Kubernetes**: Containerized deployment with orchestration

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MLflow UI     │    │   BentoML API   │    │ Drift Detection │
│   (Port 5000)   │    │   (Port 3000)   │    │   Monitoring     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   MLOps API     │
                    │   (Port 8001)   │
                    │                 │
                    │ • Experiment    │
                    │   Tracking      │
                    │ • Model Deploy  │
                    │ • Health Checks │
                    │ • Statistics    │
                    └─────────────────┘
```

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   make install
   ```

2. **Setup environment:**
   ```bash
   make setup-env
   # Edit .env file with your configuration
   ```

3. **Run the demo:**
   ```bash
   make demo
   ```

4. **Start the service:**
   ```bash
   make run
   ```

### Docker Deployment

1. **Build image:**
   ```bash
   make build
   ```

2. **Run with Docker:**
   ```bash
   make run-docker
   ```

### Kubernetes Deployment

1. **Deploy to Kubernetes:**
   ```bash
   make deploy
   ```

## API Endpoints

### Health & Monitoring
- `GET /health` - Service health check
- `GET /stats` - Service statistics

### Experiment Tracking
- `POST /experiments/log` - Log ML experiment
- `GET /experiments/history` - Get experiment history

### Model Deployment
- `POST /models/deploy` - Deploy model with BentoML
- `GET /models/{model_name}/versions` - Get model versions

### Drift Detection
- `POST /drift-detectors/create` - Create drift detector
- `GET /drift-detectors` - List active drift detectors

## Configuration

### Environment Variables

```bash
# MLflow Configuration
MLOPS_MLFLOW_TRACKING_URI=http://localhost:5000
MLOPS_MLFLOW_EXPERIMENT_NAME=tradpal_trading_models

# BentoML Configuration
MLOPS_BENTOML_API_PORT=3000
MLOPS_BENTOML_ENABLE_CORS=true

# Drift Detection
MLOPS_DRIFT_ENABLE_DETECTION=true
MLOPS_DRIFT_THRESHOLD=0.05
MLOPS_DRIFT_CHECK_INTERVAL_MINUTES=60

# Service
MLOPS_MAX_WORKERS=4
MLOPS_ENABLE_NOTIFICATIONS=true
```

### Python Configuration

```python
from services.mlops_service import MLOpsConfig, MLOpsService

config = MLOpsConfig(
    mlflow=MLflowConfig(
        tracking_uri="http://localhost:5000",
        experiment_name="tradpal_trading_models"
    ),
    drift_detection=DriftDetectionConfig(
        enable_drift_detection=True,
        drift_threshold=0.05
    )
)

service = MLOpsService(config)
await service.start()
```

## Usage Examples

### Log an Experiment

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Log experiment
result = await mlops_service.log_experiment(
    model_name="trading_model_v1",
    metrics={"accuracy": 0.85, "precision": 0.82},
    parameters={"n_estimators": 100, "max_depth": 10},
    model=model,
    framework="sklearn"
)
```

### Deploy a Model

```python
deployment_path = await mlops_service.deploy_model(
    model_name="trading_model_v1",
    model=trained_model,
    version="1.0.0"
)
```

### Setup Drift Detection

```python
import numpy as np

# Reference data from training
reference_data = np.array(X_train)

await mlops_service.create_drift_detector(
    model_name="trading_model_v1",
    reference_data=reference_data
)
```

## Drift Detection

The service uses Alibi Detect with MMD (Maximum Mean Discrepancy) for drift detection:

- **Reference Data**: Baseline data from model training
- **Threshold**: Configurable drift sensitivity (default: 0.05)
- **Monitoring**: Continuous background monitoring
- **Alerts**: Integration with Notification Service for drift alerts

## Integration with TradPal

### Notification Service Integration

The MLOps service integrates with the Notification Service for alerts:

```python
from services.notification_service import NotificationService

notification_service = NotificationService(config)
mlops_service = MLOpsService(mlops_config, notification_service)
```

### Service Communication

All TradPal services can log experiments and deploy models:

```python
# From any service
await mlops_service.log_experiment(
    model_name=f"{service_name}_model",
    metrics=performance_metrics,
    parameters=model_params,
    model=trained_model
)
```

## Monitoring & Observability

### Health Checks

```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "service": "mlops_service",
  "status": "healthy",
  "components": {
    "mlflow": true,
    "bentoml": true,
    "drift_detection": true
  },
  "models_deployed": 3,
  "experiments_logged": 25
}
```

### Metrics

- Experiment count and success rate
- Model deployment status
- Drift detection alerts
- Service performance metrics

## Development

### Running Tests

```bash
make test
make test-coverage
```

### Code Quality

```bash
make quality    # Run linting, type checking, and tests
make format     # Format code with black
```

### Demo Script

```bash
make demo
```

Shows complete workflow:
1. Experiment logging with MLflow
2. Model deployment with BentoML
3. Drift detector setup
4. Service monitoring

## Troubleshooting

### Common Issues

1. **MLflow Connection Failed**
   - Check MLflow server is running
   - Verify `MLOPS_MLFLOW_TRACKING_URI`

2. **Model Deployment Failed**
   - Ensure model is serializable
   - Check BentoML version compatibility

3. **Drift Detection Errors**
   - Verify reference data format
   - Check numpy/pytorch versions

### Logs

```bash
make logs          # Show service logs
tail -f mlops.log  # Follow logs in real-time
```

## Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Use type hints and docstrings

## License

MIT License - see LICENSE file for details.