# TradPal Trading Service

## Overview

The Trading Service is a unified microservice that consolidates all live-trading functionality from the previous 25+ microservices architecture into a single, cohesive service. It provides AI-powered automated trading with risk management, market regime detection, and comprehensive monitoring.

## Features

### Core Trading Engine
- **Automated Trading**: AI-driven trade execution with reinforcement learning
- **Risk Management**: Position sizing, stop-loss, and portfolio risk controls
- **Market Regime Detection**: Adaptive strategies based on market conditions
- **Order Execution**: Reliable order submission and management
- **Performance Tracking**: Real-time P&L, Sharpe ratio, and risk metrics

### AI & Machine Learning
- **Reinforcement Learning**: Trained agents for optimal trading decisions
- **Market Analysis**: Technical indicators and sentiment analysis
- **Adaptive Strategies**: Dynamic strategy adjustment based on market conditions
- **Model Performance**: Continuous model evaluation and retraining

### Risk Management
- **Position Sizing**: Kelly criterion and risk-based position calculations
- **Stop Loss/Take Profit**: Automated risk controls
- **Portfolio Limits**: Maximum drawdown and risk exposure controls
- **Circuit Breakers**: Emergency trading halts during extreme conditions

### Monitoring & Observability
- **Real-time Metrics**: CPU, memory, and trading performance monitoring
- **Alert System**: Configurable alerts for risk and performance thresholds
- **Health Checks**: Service health and dependency monitoring
- **Performance Reports**: Comprehensive trading analytics

## Architecture

### Service Components

```
trading_service/
├── orchestrator.py          # Main coordinator
├── trading/                 # Core trading logic
├── execution/              # Order execution
├── risk_management/        # Risk controls
├── reinforcement_learning/ # AI trading
├── market_regime/          # Market analysis
├── monitoring/             # System monitoring
├── main.py                 # FastAPI application
├── tests.py               # Comprehensive tests
└── requirements.txt       # Dependencies
```

### API Endpoints

#### Trading Operations
- `POST /trading/start` - Start automated trading session
- `POST /trading/smart-trade` - Execute AI-powered trade
- `GET /trading/status` - Get comprehensive trading status
- `POST /trading/stop` - Emergency stop all trading
- `GET /trading/performance` - Get performance report

#### Health & Monitoring
- `GET /health` - Service health check
- `GET /status` - Detailed service status
- `GET /config/default` - Default configuration

## Configuration

### Trading Parameters
```python
{
    "capital": 10000.0,           # Initial capital
    "risk_per_trade": 0.02,       # Risk per trade (2%)
    "max_positions": 5,           # Maximum open positions
    "paper_trading": true,        # Enable paper trading
    "strategy": "smart_ai",       # Trading strategy
    "rl_enabled": true,           # Enable RL trading
    "regime_detection": true      # Enable market regime detection
}
```

### Environment Variables
```bash
TRADING_SERVICE_HOST=0.0.0.0
TRADING_SERVICE_PORT=8002
LOG_LEVEL=INFO
REDIS_URL=redis://localhost:6379
```

## Usage

### Starting the Service

```bash
# Install dependencies
pip install -r services/trading_service/requirements.txt

# Run the service
python services/trading_service/main.py

# Or with custom host/port
python services/trading_service/main.py --host 0.0.0.0 --port 8002
```

### API Usage Examples

#### Start Automated Trading
```python
import requests

response = requests.post("http://localhost:8002/trading/start", json={
    "symbol": "BTC/USDT",
    "config": {
        "capital": 10000.0,
        "risk_per_trade": 0.02,
        "paper_trading": True
    }
})
```

#### Execute Smart Trade
```python
response = requests.post("http://localhost:8002/trading/smart-trade", json={
    "symbol": "BTC/USDT",
    "market_data": {
        "current_price": 50000.0,
        "prices": [49000, 49500, 50000, 50500, 51000],
        "volumes": [100, 120, 150, 130, 140]
    }
})
```

#### Get Trading Status
```python
response = requests.get("http://localhost:8002/trading/status")
status = response.json()
print(f"Active sessions: {status['active_sessions']}")
print(f"Total P&L: {status['total_pnl']}")
```

## Testing

### Running Tests
```bash
# Run all tests
pytest services/trading_service/tests.py

# Run with coverage
pytest --cov=services.trading_service tests.py

# Run specific test class
pytest services/trading_service/tests.py::TestTradingServiceOrchestrator
```

### Test Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Async Testing**: Proper async/await testing with pytest-asyncio
- **Mocking**: External dependencies properly mocked

## Dependencies

### Core Dependencies
- **FastAPI**: Web framework for REST API
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation

### Optional Dependencies
- **Redis**: Event system integration
- **psutil**: System monitoring
- **pytest**: Testing framework

## Performance Characteristics

### Throughput
- **Trades/second**: Up to 100 concurrent trades
- **API Latency**: <50ms for most operations
- **Memory Usage**: ~200MB base + 50MB per active session

### Scalability
- **Horizontal Scaling**: Multiple service instances
- **Load Balancing**: API Gateway integration
- **Resource Limits**: Configurable memory and CPU limits

## Security

### Authentication
- **JWT Tokens**: API authentication via API Gateway
- **Service Authentication**: mTLS for service-to-service communication
- **Zero-Trust**: All requests authenticated and authorized

### Data Protection
- **Encryption**: All sensitive data encrypted at rest
- **Audit Logging**: Complete activity trails
- **Access Controls**: Role-based access to trading operations

## Monitoring & Alerting

### Metrics Collected
- **Trading Metrics**: P&L, win rate, Sharpe ratio
- **Risk Metrics**: Drawdown, VaR, position sizes
- **System Metrics**: CPU, memory, latency
- **Error Rates**: Failed trades, API errors

### Alert Types
- **Risk Alerts**: Portfolio risk threshold exceeded
- **Performance Alerts**: Significant P&L changes
- **System Alerts**: Service health issues
- **Trading Alerts**: Order execution failures

## Development

### Code Structure
- **Modular Design**: Clear separation of concerns
- **Async First**: All I/O operations use asyncio
- **Type Hints**: Full type annotation coverage
- **Error Handling**: Comprehensive exception handling

### Testing Strategy
- **TDD Approach**: Tests written before implementation
- **Mocking**: External dependencies properly mocked
- **Integration Tests**: End-to-end workflow validation
- **Performance Tests**: Load and stress testing

## Deployment

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8002
CMD ["python", "services/trading_service/main.py"]
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trading-service
  template:
    metadata:
      labels:
        app: trading-service
    spec:
      containers:
      - name: trading-service
        image: tradpal/trading-service:latest
        ports:
        - containerPort: 8002
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
```

## Troubleshooting

### Common Issues

#### Service Won't Start
- Check dependencies: `pip install -r requirements.txt`
- Verify environment variables
- Check port availability (default: 8002)

#### High Memory Usage
- Monitor active trading sessions
- Check for memory leaks in RL models
- Adjust position limits

#### Trading Performance Issues
- Check market data feed latency
- Verify RL model performance
- Monitor API rate limits

### Logs
```bash
# View service logs
tail -f logs/trading_service.log

# Debug mode
LOG_LEVEL=DEBUG python services/trading_service/main.py
```

## Future Enhancements

### Planned Features
- **Advanced RL Models**: Multi-agent reinforcement learning
- **Alternative Data**: Social sentiment, on-chain metrics
- **Portfolio Optimization**: Multi-asset portfolio management
- **Real-time Analytics**: Live performance dashboards

### Performance Improvements
- **GPU Acceleration**: CUDA support for ML models
- **Memory Optimization**: Streaming data processing
- **Caching**: Redis caching for market data

---

*This service is part of the TradPal microservices architecture. For more information, see the main project documentation.*