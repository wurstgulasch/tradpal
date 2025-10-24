# Risk Management Service

Specialized microservice for risk management and position sizing in the TradPal trading system.

## Overview

The Risk Management Service is responsible for all risk-related calculations and validations, including:

- Position sizing calculations
- Portfolio risk assessment
- Kelly Criterion optimization
- Value at Risk (VaR) calculations
- Trade validation against risk criteria

## Architecture

- **Port**: 8015
- **Framework**: FastAPI
- **Risk Models**: Kelly Criterion, VaR, Sharpe Ratio
- **Data Processing**: Pandas, NumPy, SciPy

## API Endpoints

### Position Sizing
- `POST /position-size` - Calculate optimal position size
- `GET /kelly-criterion` - Calculate Kelly Criterion

### Risk Assessment
- `POST /portfolio-risk` - Assess portfolio risk
- `GET /var/{confidence}` - Calculate Value at Risk
- `POST /validate-trade` - Validate trade against risk criteria

### Health
- `GET /health` - Service health check

## Usage

```python
from services.trading_service.risk_management_service.client import get_risk_management_client

async def calculate_position():
    async with get_risk_management_client() as client:
        result = await client.calculate_position_size(
            capital=10000,
            entry_price=50000,
            stop_loss=49000,
            risk_config={"max_risk_per_trade": 0.02}
        )
        print(f"Position size: {result['position_size']}")
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
- Trading Execution Service (for position sizing)
- Portfolio Service (for risk assessment)
- Monitoring Service (for risk metrics)