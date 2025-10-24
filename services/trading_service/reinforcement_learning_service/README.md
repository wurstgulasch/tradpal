# Reinforcement Learning Service

Specialized microservice for reinforcement learning-based trading strategies in the TradPal trading system.

## Overview

The Reinforcement Learning Service is responsible for training and deploying RL agents for trading decisions, including:

- PPO, DQN, and A2C algorithms
- Custom trading environments
- Action prediction and confidence scoring
- Agent training and evaluation

## Architecture

- **Port**: 8013
- **Framework**: FastAPI
- **RL Libraries**: Stable Baselines3, Gymnasium
- **Deep Learning**: PyTorch

## API Endpoints

### Training
- `POST /train` - Start agent training
- `GET /status/{agent_id}` - Get training status

### Inference
- `POST /action` - Get trading action from agent

### Management
- `GET /agents` - List available agents
- `GET /agent/{agent_id}` - Get agent information
- `DELETE /agent/{agent_id}` - Delete an agent

### Health
- `GET /health` - Service health check

## Usage

```python
from services.trading_service.reinforcement_learning_service.client import get_reinforcement_learning_client

async def get_trading_action():
    async with get_reinforcement_learning_client() as client:
        action, confidence = await client.get_action(
            "BTC/USDT",
            [0.1, 0.2, 0.3, 0.4],  # market state
            "ppo_agent_2024"
        )
        print(f"Action: {action}, Confidence: {confidence}")
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
- Data Service (for market state data)
- Trading Execution Service (for action deployment)
- Monitoring Service (for training metrics)