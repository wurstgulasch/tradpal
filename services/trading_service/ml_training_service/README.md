# ML Training Service

Specialized microservice for machine learning model training and ensemble methods in the TradPal trading system.

## Overview

The ML Training Service is responsible for training machine learning models for trading predictions, including:

- Ensemble model training
- Feature engineering
- Model validation and evaluation
- Model persistence and management

## Architecture

- **Port**: 8012
- **Framework**: FastAPI
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Optimization**: Optuna for hyperparameter tuning

## API Endpoints

### Training
- `POST /train` - Start model training
- `GET /models` - List available models
- `GET /model/{model_id}` - Get model information
- `DELETE /model/{model_id}` - Delete a model

### Health
- `GET /health` - Service health check

## Usage

```python
from services.trading_service.ml_training_service.client import get_ml_training_client

async def train_model():
    async with get_ml_training_client() as client:
        result = await client.train_model("BTC/USDT", {
            "model_type": "ensemble",
            "target_horizon": 24
        })
        print(f"Training started: {result}")
```

## Dependencies

See `requirements.txt` for service-specific dependencies. This service has minimal dependencies to maintain microservice independence.

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run service
python main.py

# Run tests
pytest tests/
```

## Integration

This service integrates with:
- Data Service (for training data)
- Trading AI Service (for model deployment)
- Monitoring Service (for training metrics)