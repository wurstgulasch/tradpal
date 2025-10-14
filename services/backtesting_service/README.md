# Backtesting Service

A microservice for comprehensive historical trading strategy backtesting in the TradPal Trading System.

## Overview

The Backtesting Service provides async, event-driven backtesting capabilities for trading strategies. It supports:

- **Single Strategy Backtesting**: Traditional, ML-enhanced, LSTM, and Transformer strategies
- **Multi-Symbol Backtesting**: Parallel backtesting across multiple trading pairs
- **Multi-Model Comparison**: Compare performance of different ML models
- **Walk-Forward Optimization**: Parameter optimization with walk-forward analysis
- **Real-time Monitoring**: Track backtest progress and results
- **Event Integration**: Seamless integration with the event-driven architecture

## Architecture

### Components

- **`service.py`**: Core `BacktestingService` and `AsyncBacktester` classes
- **`api.py`**: FastAPI REST endpoints for HTTP access
- **`tests.py`**: Comprehensive unit test suite
- **`test_integration.py`**: Integration tests for end-to-end validation

### Key Features

#### Async Processing
- Non-blocking backtest execution using asyncio
- Thread pool execution for CPU-intensive operations
- Concurrent processing of multiple backtests

#### Event-Driven Design
- Integration with Redis event streams
- Publish/subscribe pattern for service communication
- Real-time status updates and notifications

#### Comprehensive Metrics
- Standard performance metrics (Sharpe ratio, win rate, drawdown)
- Transaction cost analysis
- Risk-adjusted return calculations
- CAGR and profit factor analysis

#### Strategy Support
- **Traditional**: EMA, RSI, Bollinger Bands, ATR, ADX
- **ML-Enhanced**: PyTorch models with confidence scoring
- **LSTM**: Time series prediction models
- **Transformer**: Advanced sequence modeling

## API Endpoints

### Core Endpoints

#### `POST /backtest`
Run a single backtest.

**Request:**
```json
{
  "symbol": "BTC/USDT",
  "timeframe": "1d",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "strategy": "traditional",
  "initial_capital": 10000.0,
  "config": {}
}
```

**Response:**
```json
{
  "backtest_id": "backtest_20241231_143052",
  "status": "running"
}
```

#### `POST /backtest/multi-symbol`
Run parallel backtests for multiple symbols.

**Request:**
```json
{
  "symbols": ["BTC/USDT", "ETH/USDT", "ADA/USDT"],
  "timeframe": "1d",
  "start_date": "2024-01-01",
  "end_date": "2024-06-01",
  "initial_capital": 5000.0,
  "max_workers": 3
}
```

#### `POST /backtest/multi-model`
Compare different ML models.

**Request:**
```json
{
  "symbol": "BTC/USDT",
  "timeframe": "1d",
  "models_to_test": ["traditional_ml", "lstm", "transformer"],
  "max_workers": 3
}
```

#### `POST /backtest/walk-forward`
Run parameter optimization.

**Request:**
```json
{
  "parameter_grid": {
    "ema_short": [5, 9, 12],
    "ema_long": [21, 26, 50]
  },
  "evaluation_metric": "sharpe_ratio"
}
```

### Monitoring Endpoints

#### `GET /backtest/{backtest_id}`
Get status and results of a specific backtest.

#### `GET /backtest/active`
List all currently running backtests.

#### `DELETE /backtest/completed`
Clean up old completed backtests.

#### `GET /health`
Service health check.

## Usage Examples

### Python Client

```python
import requests

# Single backtest
response = requests.post("http://localhost:8001/backtest", json={
    "symbol": "BTC/USDT",
    "timeframe": "1d",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "strategy": "ml_enhanced",
    "initial_capital": 10000.0
})

backtest_id = response.json()["backtest_id"]

# Check status
status = requests.get(f"http://localhost:8001/backtest/{backtest_id}")
print(f"Status: {status.json()['status']}")
```

### Event-Driven Usage

```python
from src.event_system import EventSystem

event_system = EventSystem()

# Subscribe to backtest completion
event_system.subscribe("backtest.completed", handle_completion)

# Publish backtest request
event_system.publish(Event(
    type="backtest.request",
    data={
        "backtest_id": "my_backtest",
        "symbol": "BTC/USDT",
        "strategy": "traditional"
    }
))
```

## Configuration

### Environment Variables

- `REDIS_URL`: Redis connection URL (default: redis://localhost:6379)
- `BACKTEST_TIMEOUT`: Default backtest timeout in seconds (default: 300)
- `MAX_WORKERS`: Maximum parallel workers (default: 4)

### Strategy Configuration

Backtests can be customized with indicator parameters:

```json
{
  "ema_short": 9,
  "ema_long": 21,
  "rsi_period": 14,
  "rsi_oversold": 30,
  "rsi_overbought": 70,
  "bb_period": 20,
  "bb_std": 2.0,
  "atr_period": 14
}
```

## Performance Metrics

The service calculates comprehensive performance metrics:

- **Return Metrics**: Total P&L, CAGR, Return %
- **Risk Metrics**: Sharpe Ratio, Maximum Drawdown, Volatility
- **Trade Metrics**: Win Rate, Profit Factor, Average Win/Loss
- **Cost Analysis**: Total Commissions, Net P&L after costs

## Deployment

### Docker

```bash
# Build image
docker build -t tradpal/backtesting-service ./services/backtesting_service

# Run container
docker run -p 8001:8001 -v $(pwd)/output:/app/output tradpal/backtesting-service
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f services/backtesting_service/k8s-deployment.yaml
```

### Local Development

```bash
# Install dependencies
pip install -r services/backtesting_service/requirements.txt

# Run service
python -m services.backtesting_service.api
```

## Testing

### Unit Tests

```bash
# Run unit tests
pytest services/backtesting_service/tests.py -v
```

### Integration Tests

```bash
# Run integration tests (requires running service)
python services/backtesting_service/test_integration.py

# Quick smoke test
python services/backtesting_service/test_integration.py --smoke-only
```

### Test Coverage

The test suite covers:
- Service initialization and event handling
- Single and multi-symbol backtesting
- ML model comparison
- Walk-forward optimization
- Error handling and edge cases
- API endpoint validation

## Monitoring

### Health Checks

The service provides health check endpoints for monitoring:

- `/health`: Basic service health
- `/backtest/active`: Active backtest count
- Service logs: Comprehensive logging with structured output

### Metrics

Key metrics to monitor:
- Active backtest count
- Backtest completion rate
- Error rate
- Average backtest duration
- Memory and CPU usage

## Error Handling

The service implements comprehensive error handling:

- **Timeout Management**: Configurable timeouts for long-running backtests
- **Resource Limits**: Memory and CPU limits to prevent resource exhaustion
- **Graceful Degradation**: Continues operation when individual backtests fail
- **Detailed Logging**: Structured logging for debugging and monitoring

## Integration

### Event System Integration

The service integrates with the TradPal event system for:

- **Backtest Requests**: Receive backtest execution requests
- **Status Updates**: Publish real-time progress updates
- **Result Distribution**: Broadcast completed results
- **Error Notifications**: Alert on backtest failures

### Data Sources

Supports multiple data sources:
- **CCXT**: Cryptocurrency exchanges
- **Local Cache**: Redis/file-based caching
- **Custom Sources**: Extensible data provider interface

## Development

### Adding New Strategies

1. Implement strategy logic in `AsyncBacktester`
2. Add strategy method (e.g., `_prepare_custom_strategy_async`)
3. Update API validation
4. Add unit tests

### Extending Metrics

1. Add metric calculation in `_calculate_metrics_sync`
2. Update response models
3. Add tests for new metrics

### Performance Optimization

- Use vectorized operations for calculations
- Implement caching for repeated data access
- Optimize memory usage for large datasets
- Consider GPU acceleration for ML models

## Troubleshooting

### Common Issues

1. **Timeout Errors**: Increase timeout values or optimize backtest parameters
2. **Memory Issues**: Reduce data size or implement streaming processing
3. **ML Model Errors**: Check model training status and dependencies
4. **Event System Issues**: Verify Redis connectivity and event routing

### Debugging

- Enable debug logging: Set `LOG_LEVEL=DEBUG`
- Check service logs for detailed error information
- Use integration tests to isolate issues
- Monitor resource usage during backtest execution

## License

This service is part of the TradPal Trading System and follows the same MIT license.