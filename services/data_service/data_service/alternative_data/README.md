# Alternative Data Service

Advanced alternative data collection and processing service for TradPal's AI trading system. This service provides sentiment analysis, on-chain metrics, economic indicators, and ML feature engineering to enhance trading performance.

## Overview

The Alternative Data Service collects and processes multiple data sources to create comprehensive ML features:

- **Sentiment Analysis**: Twitter, Reddit, News, Fear & Greed Index
- **On-Chain Metrics**: Blockchain data (BTC/ETH), NVT ratios, whale movements
- **Economic Indicators**: Fed rates, CPI, unemployment, market regime signals
- **Feature Engineering**: Normalized features, composite indicators, risk scores

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Sentiment      │    │  On-Chain        │    │  Economic       │
│  Analyzer       │    │  Collector       │    │  Collector      │
│                 │    │                  │    │                 │
│ • Twitter API   │    │ • Glassnode API  │    │ • FRED API      │
│ • Reddit API    │    │ • Blockchain.com │    │ • BLS API       │
│ • News API      │    │ • Custom metrics │    │ • Alpha Vantage │
│ • Fear & Greed  │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────────────┐
                    │  Data Processor    │
                    │                    │
                    │ • Feature Eng.     │
                    │ • Normalization    │
                    │ • Composite feats  │
                    │ • Risk indicators  │
                    └────────────────────┘
                             │
                    ┌────────────────────┐
                    │   REST API         │
                    │                    │
                    │ • /sentiment       │
                    │ • /onchain         │
                    │ • /economic        │
                    │ • /data/process    │
                    └────────────────────┘
```

## Quick Start

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set environment variables:**
```bash
export TWITTER_BEARER_TOKEN="your_token"
export GLASSNODE_API_KEY="your_key"
export FRED_API_KEY="your_key"
# ... other API keys
```

3. **Run the service:**
```bash
cd services/alternative_data_service
python main.py
```

4. **Test the service:**
```bash
python test_service.py
```

### Docker Deployment

```bash
# Build and run with docker-compose
docker-compose --profile advanced-features up alternative-data-service

# Or run specific profile
docker-compose --profile advanced-features up -d
```

## API Endpoints

### Health Check
```http
GET /health
```

### Sentiment Analysis
```http
POST /sentiment/analyze
Content-Type: application/json

{
  "symbol": "BTC/USDT",
  "hours": 24
}
```

```http
GET /sentiment/fear-greed
```

### On-Chain Metrics
```http
POST /onchain/metrics
Content-Type: application/json

{
  "symbol": "BTC",
  "metrics": ["nvt_ratio", "active_addresses"]
}
```

### Economic Indicators
```http
POST /economic/indicators
Content-Type: application/json

{
  "indicators": ["fed_funds_rate", "cpi"]
}
```

### Data Processing
```http
GET /data/{symbol}
POST /data/process
Content-Type: application/json

{
  "symbol": "BTC/USDT",
  "include_sentiment": true,
  "include_onchain": true,
  "include_economic": true
}
```

### Background Collection
```http
POST /data/collect?symbol=BTC/USDT
```

### Metrics
```http
GET /metrics
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TWITTER_BEARER_TOKEN` | Twitter API Bearer Token | No |
| `REDDIT_CLIENT_ID` | Reddit API Client ID | No |
| `REDDIT_CLIENT_SECRET` | Reddit API Client Secret | No |
| `NEWS_API_KEY` | News API Key | No |
| `GLASSNODE_API_KEY` | Glassnode API Key | No |
| `FRED_API_KEY` | Federal Reserve Economic Data API Key | No |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API Key | No |
| `BLS_API_KEY` | Bureau of Labor Statistics API Key | No |
| `REDIS_URL` | Redis connection URL | No |
| `PROMETHEUS_ENABLED` | Enable Prometheus metrics | No |
| `PROMETHEUS_PORT` | Prometheus metrics port | No |

### Fallback Mechanisms

The service includes comprehensive fallback mechanisms:

- **Missing Dependencies**: Graceful degradation when optional libraries aren't installed
- **API Failures**: Mock data generation for development/testing
- **Rate Limits**: Exponential backoff and request throttling
- **Network Issues**: Circuit breaker pattern with automatic recovery

## Data Sources

### Sentiment Data
- **Twitter**: Real-time sentiment analysis using RoBERTa models
- **Reddit**: Cryptocurrency subreddit sentiment
- **News**: Financial news sentiment from major sources
- **Fear & Greed Index**: CNN Business Fear & Greed Index

### On-Chain Data
- **Glassnode**: Professional blockchain analytics
- **Blockchain.com**: Public blockchain data
- **Custom Metrics**: NVT ratio, active addresses, transaction volume

### Economic Data
- **FRED**: Federal Reserve Economic Data
- **BLS**: Bureau of Labor Statistics
- **Alpha Vantage**: Economic indicators and market data

## Feature Engineering

### Sentiment Features
- `twitter_sentiment`: Twitter sentiment score (-1 to 1)
- `reddit_sentiment`: Reddit sentiment score
- `news_sentiment`: News sentiment score
- `avg_sentiment`: Cross-platform average
- `sentiment_momentum`: Sentiment change vs historical average

### On-Chain Features
- `nvt_ratio_current`: Network Value to Transactions ratio
- `active_addresses`: Daily active addresses
- `transaction_volume`: Daily transaction volume
- `nvt_overvalued`: Binary indicator (>150)
- `nvt_undervalued`: Binary indicator (<50)

### Economic Features
- `fed_funds_rate_current`: Current federal funds rate
- `cpi_current`: Consumer Price Index
- `unemployment_rate_current`: Unemployment rate
- `rate_environment_low/high/normal`: Rate environment classification

### Composite Features
- `fear_greed_index`: Fear & Greed Index (0-100)
- `sentiment_onchain_correlation`: Correlation between sentiment and on-chain activity
- `economic_sentiment_impact`: Economic conditions impact on sentiment
- `combined_risk_score`: Risk score combining NVT and sentiment

## Testing

Run the comprehensive test suite:

```bash
cd services/alternative_data_service
python test_service.py
```

Tests cover:
- Component initialization
- Data collection from all sources
- Feature processing pipeline
- Error handling and fallbacks
- Full integration testing

## Integration

### Event System Integration

The service integrates with TradPal's Redis Streams event system:

```python
# Publish alternative data events
await redis.xadd("alternative_data", {
    "symbol": "BTC/USDT",
    "features": json.dumps(processed_features),
    "timestamp": datetime.utcnow().isoformat()
})
```

### Service Dependencies

- **Redis**: Event streaming and caching
- **Prometheus**: Metrics collection (optional)
- **External APIs**: Various data providers (with fallbacks)

## Performance

### Optimization Features
- **Async I/O**: Full asyncio implementation for concurrent requests
- **Connection Pooling**: HTTP session reuse for API calls
- **Memory Management**: Rolling window buffers for historical data
- **Caching**: Redis-based caching for frequently accessed data

### Benchmarks
- **Data Collection**: <5 seconds for complete alternative data packet
- **Feature Processing**: <1 second for ML feature generation
- **API Response Time**: <2 seconds for individual endpoints
- **Memory Usage**: <500MB with full historical data

## Monitoring

### Metrics
- Request/response times
- API call success rates
- Data collection latency
- Feature processing performance
- Error rates by component

### Health Checks
- Service availability
- API connectivity
- Data freshness
- Memory usage
- Error rates

## Development

### Project Structure
```
services/alternative_data_service/
├── __init__.py          # Data models and enums
├── main.py              # FastAPI service
├── sentiment_analyzer.py # Sentiment analysis
├── onchain_collector.py  # On-chain data collection
├── economic_collector.py # Economic data collection
├── data_processor.py    # Feature engineering
├── test_service.py      # Test suite
└── README.md           # This file
```

### Adding New Data Sources

1. Create a new collector class in a separate file
2. Implement the required interface methods
3. Add API endpoints in `main.py`
4. Update the data processor for new features
5. Add tests and documentation

### Extending Feature Engineering

1. Add new feature calculation methods in `DataProcessor`
2. Update the `process_to_features` method
3. Add feature normalization if needed
4. Update tests and documentation

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Service works with mock data for development
2. **Rate Limits**: Automatic backoff and retry mechanisms
3. **Network Issues**: Circuit breaker prevents cascade failures
4. **Memory Issues**: Rolling windows limit historical data retention

### Logs
Check service logs for detailed error information:
```bash
docker-compose logs alternative-data-service
```

### Debugging
Use the test script for component-level debugging:
```bash
python test_service.py --verbose
```

## Future Enhancements

- **Real-time Streaming**: WebSocket connections for live data
- **Advanced ML**: Deep learning sentiment models
- **Alternative Data**: Satellite imagery, web scraping
- **Predictive Features**: Time series forecasting integration
- **Multi-asset Support**: Extended beyond crypto to traditional assets

## Contributing

1. Follow the existing code patterns and async architecture
2. Add comprehensive tests for new features
3. Update documentation and API specs
4. Ensure backward compatibility
5. Test with the full integration suite

## License

See project root LICENSE file.