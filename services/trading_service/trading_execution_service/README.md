# Trading Execution Service

Specialized microservice for trade execution and order management in the TradPal trading system.

## Overview

The Trading Execution Service is responsible for executing trades and managing orders, including:

- Order placement and cancellation
- Portfolio management
- Position tracking and P&L calculation
- Account balance monitoring
- Integration with trading exchanges via CCXT

## Architecture

- **Port**: 8016
- **Framework**: FastAPI
- **Exchange Integration**: CCXT library
- **Order Types**: Market, Limit, Stop orders

## API Endpoints

### Orders
- `POST /order` - Place new order
- `DELETE /order/{order_id}` - Cancel order
- `GET /order/{order_id}` - Get order status
- `GET /orders` - Get open orders

### Portfolio
- `GET /portfolio` - Get portfolio information
- `GET /balance` - Get account balance
- `POST /close-position` - Close position

### Health
- `GET /health` - Service health check

## Usage

```python
from services.trading_service.trading_execution_service.client import get_trading_execution_client

async def place_market_order():
    async with get_trading_execution_client() as client:
        result = await client.place_order(
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            quantity=0.001
        )
        print(f"Order placed: {result['order_id']}")
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
- Broker APIs (via CCXT)
- Risk Management Service (for position sizing)
- Monitoring Service (for execution metrics)
- Portfolio Service (for position tracking)