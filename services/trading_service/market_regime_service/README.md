# Market Regime Service

Specialized microservice for market regime detection and analysis in the TradPal trading system.

## Overview

The Market Regime Service is responsible for detecting and analyzing different market conditions, including:

- Bull, Bear, and Sideways market detection
- Multi-timeframe regime analysis
- Regime strength and alignment scoring
- Statistical regime analysis

## Architecture

- **Port**: 8014
- **Framework**: FastAPI
- **Analysis**: Technical indicators, statistical methods
- **Data Processing**: Pandas, NumPy, SciPy

## API Endpoints

### Analysis
- `POST /analyze` - Analyze single timeframe regime
- `POST /analyze-multi` - Analyze multi-timeframe regime
- `POST /predict/{symbol}` - Predict future regime

### Information
- `GET /regimes` - List available regime types
- `GET /statistics/{symbol}` - Get regime statistics

### Health
- `GET /health` - Service health check

## Usage

```python
from services.trading_service.market_regime_service.client import get_market_regime_client

async def analyze_market():
    async with get_market_regime_client() as client:
        result = await client.analyze_market_regime(
            "BTC/USDT",
            ohlcv_data,
            lookback_periods=20
        )
        print(f"Regime: {result['regime']}, Confidence: {result['confidence']}")
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
- Data Service (for market data)
- Trading AI Service (for regime-based strategies)
- Monitoring Service (for regime metrics)