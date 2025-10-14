# Risk Service

## Overview

The Risk Service is a comprehensive risk management microservice for the TradPal trading system. It provides advanced position sizing, portfolio risk assessment, and real-time risk monitoring capabilities to ensure safe and profitable trading operations.

## Features

### Position Sizing
- **Kelly Criterion**: Optimal position sizing based on win rate and risk-reward ratio
- **ATR-based Stop Loss/Take Profit**: Dynamic risk management using Average True Range
- **Volatility-adjusted Leverage**: Automatic leverage adjustment based on market volatility
- **Risk Percentage Control**: Configurable risk per trade limits

### Portfolio Risk Assessment
- **Comprehensive Risk Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown, Value at Risk (VaR), Conditional VaR (CVaR)
- **Risk Level Classification**: Automatic risk assessment with actionable recommendations
- **Portfolio Exposure Tracking**: Real-time monitoring of total portfolio risk and position correlations

### Risk Monitoring
- **Real-time Risk Updates**: Continuous portfolio risk monitoring with alerts
- **Risk Parameter Adjustment**: Dynamic risk parameter updates based on market conditions
- **Event-driven Architecture**: Integration with event system for risk alerts and notifications

## Architecture

```
services/risk_service/
├── service.py          # Core RiskService implementation
├── api.py             # FastAPI REST endpoints
├── demo.py            # Comprehensive demonstration script
├── tests.py           # Unit and integration tests
└── __init__.py        # Package initialization
```

## API Endpoints

### Position Sizing
```http
POST /position/sizing
```
Calculate optimal position size with risk management parameters.

**Request Body:**
```json
{
  "symbol": "BTC/USDT",
  "capital": 10000,
  "entry_price": 50000,
  "position_type": "long",
  "atr_value": 1000,
  "volatility": 0.03,
  "risk_percentage": 0.01
}
```

**Response:**
```json
{
  "symbol": "BTC/USDT",
  "position_size": 0.2,
  "position_value": 10000,
  "risk_amount": 100,
  "stop_loss_price": 49000,
  "take_profit_price": 52000,
  "leverage": 1.0,
  "risk_percentage": 0.01,
  "reward_risk_ratio": 2.0,
  "calculated_at": "2024-01-01T12:00:00Z",
  "parameters": {...}
}
```

### Portfolio Risk Assessment
```http
POST /portfolio/assess
```
Assess overall portfolio risk with comprehensive metrics.

**Request Body:**
```json
{
  "returns": [0.01, -0.005, 0.008, ...],
  "frequency": "daily"
}
```

**Response:**
```json
{
  "symbol": "PORTFOLIO",
  "risk_level": "LOW",
  "risk_score": 0.3,
  "metrics": {
    "sharpe_ratio": 1.8,
    "sortino_ratio": 2.1,
    "max_drawdown": -0.08,
    "value_at_risk": -0.025,
    "conditional_var": -0.035,
    "volatility": 0.15
  },
  "recommendations": [
    "Consider increasing position sizes",
    "Monitor volatility closely"
  ],
  "assessed_at": "2024-01-01T12:00:00Z"
}
```

### Portfolio Exposure
```http
GET /portfolio/exposure
```
Get current portfolio exposure and risk breakdown.

**Response:**
```json
{
  "total_positions": 3,
  "total_exposure": 25000,
  "total_risk": 250,
  "positions": [
    {
      "symbol": "BTC/USDT",
      "value": 15000,
      "risk": 150
    }
  ]
}
```

### Risk Parameters
```http
PUT /risk/parameters
```
Update risk management parameters.

**Request Body:**
```json
{
  "max_risk_per_trade": 0.02,
  "max_portfolio_risk": 0.06,
  "max_leverage": 2.0
}
```

### Health Check
```http
GET /health
```
Service health and status information.

## Usage Examples

### Basic Position Sizing

```python
from services.risk_service.service import RiskService, RiskRequest

# Initialize service
risk_service = RiskService()

# Create position sizing request
request = RiskRequest(
    symbol="BTC/USDT",
    capital=10000,
    entry_price=50000,
    position_type="long",
    atr_value=1000,
    volatility=0.03,
    risk_percentage=0.01
)

# Calculate position size
sizing = await risk_service.calculate_position_sizing(request)

print(f"Position Size: ${sizing.position_value:.0f}")
print(f"Stop Loss: ${sizing.stop_loss_price:.0f}")
print(f"Take Profit: ${sizing.take_profit_price:.0f}")
```

### Portfolio Risk Assessment

```python
import pandas as pd

# Generate portfolio returns
returns = pd.Series([0.01, -0.005, 0.008, -0.012, 0.015])

# Assess portfolio risk
assessment = await risk_service.assess_portfolio_risk(returns)

print(f"Risk Level: {assessment.risk_level}")
print(f"Sharpe Ratio: {assessment.metrics['sharpe_ratio']:.2f}")
print("Recommendations:")
for rec in assessment.recommendations:
    print(f"- {rec}")
```

### Risk Parameter Adjustment

```python
# Update risk parameters for volatile market
await risk_service.update_risk_parameters({
    "max_risk_per_trade": 0.005,  # Reduce risk in volatile conditions
    "max_portfolio_risk": 0.03,
    "max_leverage": 1.5
})
```

## Risk Management Strategies

### Conservative Strategy
- Max risk per trade: 0.5%
- Max portfolio risk: 2%
- Leverage: 1.0-1.5x
- Suitable for: High volatility, uncertain markets

### Balanced Strategy
- Max risk per trade: 1%
- Max portfolio risk: 4%
- Leverage: 1.5-2.0x
- Suitable for: Normal market conditions

### Aggressive Strategy
- Max risk per trade: 2%
- Max portfolio risk: 8%
- Leverage: 2.0-3.0x
- Suitable for: Bull markets, low volatility

## Configuration

### Default Risk Parameters

```python
default_params = RiskParameters(
    max_risk_per_trade=0.01,      # 1% max risk per trade
    max_portfolio_risk=0.05,      # 5% max portfolio risk
    max_leverage=2.0,             # 2x maximum leverage
    min_leverage=1.0,             # 1x minimum leverage
    kelly_fraction=1.0,           # Full Kelly criterion
    atr_multiplier_sl=1.5,        # 1.5x ATR for stop loss
    atr_multiplier_tp=3.0,        # 3x ATR for take profit
    volatility_lookback=20,       # 20-period volatility
    risk_free_rate=0.02           # 2% risk-free rate
)
```

### Environment Variables

```bash
# Risk Service Configuration
RISK_MAX_RISK_PER_TRADE=0.01
RISK_MAX_PORTFOLIO_RISK=0.05
RISK_MAX_LEVERAGE=2.0
RISK_MIN_LEVERAGE=1.0

# Service Configuration
RISK_SERVICE_HOST=0.0.0.0
RISK_SERVICE_PORT=8001
```

## Testing

### Running Tests

```bash
# Unit tests
python -m pytest services/risk_service/tests.py::TestRiskService -v

# Integration tests
python -m pytest services/risk_service/tests.py::TestRiskServiceIntegration -v

# Performance tests
python -m pytest services/risk_service/tests.py::TestRiskServicePerformance -v --benchmark-only
```

### Demo Script

```bash
# Run comprehensive demonstration
python services/risk_service/demo.py
```

## Docker Deployment

### Build Image

```bash
docker build -t tradpal/risk-service:latest .
```

### Run Container

```bash
docker run -p 8001:8001 \
  -e RISK_MAX_RISK_PER_TRADE=0.01 \
  tradpal/risk-service:latest
```

### Docker Compose

```yaml
version: '3.8'
services:
  risk-service:
    image: tradpal/risk-service:latest
    ports:
      - "8001:8001"
    environment:
      - RISK_MAX_RISK_PER_TRADE=0.01
      - RISK_MAX_PORTFOLIO_RISK=0.05
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: risk-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: risk-service
  template:
    metadata:
      labels:
        app: risk-service
    spec:
      containers:
      - name: risk-service
        image: tradpal/risk-service:latest
        ports:
        - containerPort: 8001
        env:
        - name: RISK_MAX_RISK_PER_TRADE
          value: "0.01"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Monitoring

### Health Checks

The service provides comprehensive health monitoring:

```json
{
  "service": "risk_service",
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "portfolio_positions": 3,
  "risk_history": 150,
  "parameters": {...}
}
```

### Metrics

Key metrics exposed for monitoring:

- `risk_service_positions_total`: Total active positions
- `risk_service_portfolio_risk`: Current portfolio risk level
- `risk_service_calculation_duration`: Position sizing calculation time
- `risk_service_assessment_duration`: Risk assessment calculation time

### Alerts

Automatic alerts for:

- Portfolio risk exceeding thresholds
- Individual position risk violations
- Service health degradation
- Parameter validation failures

## Integration

### Event System Integration

The Risk Service integrates with the event system for:

- Position sizing events (`risk.position_sized`)
- Risk assessment events (`risk.portfolio_assessed`)
- Risk alerts (`risk.alert_triggered`)
- Parameter updates (`risk.parameters_updated`)

### Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **fastapi**: REST API framework
- **uvicorn**: ASGI server
- **pydantic**: Data validation

## Best Practices

### Risk Management
1. **Never risk more than 1-2% per trade** in normal conditions
2. **Reduce position sizes in high volatility** periods
3. **Use stop losses on all positions** without exception
4. **Monitor portfolio correlation** to avoid concentrated risk
5. **Regularly reassess risk parameters** based on market conditions

### Performance Optimization
1. **Cache risk calculations** for frequently accessed data
2. **Use async operations** for non-blocking risk assessments
3. **Batch position sizing** calculations when possible
4. **Monitor calculation performance** and optimize bottlenecks

### Security
1. **Validate all input parameters** thoroughly
2. **Implement rate limiting** on API endpoints
3. **Use secure communication** (HTTPS) in production
4. **Log all risk decisions** for audit trails
5. **Implement access controls** for parameter updates

## Troubleshooting

### Common Issues

**High Risk Scores:**
- Check volatility calculations
- Review position concentration
- Assess correlation between assets

**Position Sizing Errors:**
- Verify ATR calculations
- Check capital availability
- Validate risk percentage limits

**API Timeouts:**
- Check calculation complexity
- Monitor system resources
- Consider caching strategies

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Profiling

```python
import cProfile
cProfile.run('asyncio.run(demo.run_full_demo())')
```

## Contributing

1. **Add comprehensive tests** for new features
2. **Update documentation** for API changes
3. **Follow risk management best practices**
4. **Ensure backward compatibility** for existing integrations
5. **Add performance benchmarks** for critical functions

## License

This service is part of the TradPal trading system and follows the same MIT license terms.