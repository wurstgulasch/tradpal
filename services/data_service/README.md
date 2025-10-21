# TradPal Data Service

The Data Service is the centralized data management hub for the TradPal trading system, providing unified access to market data, alternative data, and market regime analysis.

## Overview

The Data Service consolidates the following previously separate services:
- `data_service/` - Core market data fetching and caching
- `alternative_data_service/` - Sentiment analysis, on-chain data, economic indicators
- `market_regime_detection_service/` - Market regime classification and clustering

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TradPal Data Service                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │  Data Sources   │  │ Alternative     │  │ Market      │  │
│  │                 │  │ Data Service    │  │ Regime      │  │
│  │ - CCXT          │  │                 │  │ Detection   │  │
│  │ - Yahoo Finance │  │ - Sentiment     │  │             │  │
│  │ - Kaggle        │  │ - On-chain      │  │ - Clustering │  │
│  │ - Caching       │  │ - Economic      │  │ - ML Class. │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Data Quality & Caching                  │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Data Sources Management
- **Multi-Source Data Fetching**: CCXT exchanges, Yahoo Finance, Kaggle datasets
- **Intelligent Caching**: Redis-based caching with TTL and invalidation
- **Fallback Systems**: Automatic fallback between data sources
- **Data Validation**: OHLC validation, gap detection, outlier removal
- **Quality Scoring**: Data quality metrics and reliability scoring

### Alternative Data Collection
- **Sentiment Analysis**: Twitter sentiment, news analysis, social media monitoring
- **On-Chain Data**: Blockchain metrics, whale transactions, network health
- **Economic Indicators**: Interest rates, GDP, employment data, inflation
- **Real-time Processing**: Streaming data collection and processing
- **Feature Engineering**: Automated feature extraction and normalization

### Market Regime Detection
- **Clustering Analysis**: Unsupervised learning for market regime identification
- **Classification Models**: ML models for regime prediction and transition detection
- **Multi-timeframe Analysis**: Regime detection across different timeframes
- **Confidence Scoring**: Probability scores for regime classifications
- **Historical Analysis**: Backtesting regime detection on historical data

## API Endpoints

### Market Data
- `GET /api/data/fetch/{symbol}` - Fetch market data for a symbol
- `POST /api/data/bulk` - Bulk data fetching for multiple symbols
- `GET /api/data/sources` - List available data sources
- `GET /api/data/quality/{symbol}` - Get data quality metrics

### Alternative Data
- `GET /api/data/alternative/sentiment` - Get sentiment analysis data
- `GET /api/data/alternative/onchain` - Get blockchain/on-chain metrics
- `GET /api/data/alternative/economic` - Get economic indicators
- `POST /api/data/alternative/process` - Process raw alternative data

### Market Regime
- `GET /api/data/regime/{symbol}` - Get current market regime classification
- `GET /api/data/regime/history/{symbol}` - Get historical regime transitions
- `POST /api/data/regime/train` - Train regime detection models
- `GET /api/data/regime/confidence/{symbol}` - Get regime confidence scores

### Cache Management
- `GET /api/cache/stats` - Get cache statistics and hit rates
- `POST /api/cache/invalidate` - Invalidate cache entries
- `POST /api/cache/preload` - Preload frequently used data

## Configuration

### Environment Variables

```bash
# Service Configuration
DATA_SERVICE_HOST=0.0.0.0
DATA_SERVICE_PORT=8001

# Data Sources
ENABLE_CCXT=true
ENABLE_YAHOO_FINANCE=true
ENABLE_KAGGLE_DATASETS=true

# Alternative Data
ENABLE_SENTIMENT_ANALYSIS=true
ENABLE_ONCHAIN_DATA=true
ENABLE_ECONOMIC_DATA=true

# Market Regime
REGIME_DETECTION_ALGORITHM=clustering
REGIME_LOOKBACK_PERIODS=100
REGIME_CONFIDENCE_THRESHOLD=0.7

# Caching
REDIS_URL=redis://localhost:6379
CACHE_TTL_HOURS=24
CACHE_MAX_SIZE_GB=10

# Quality Control
DATA_QUALITY_THRESHOLD=0.8
VALIDATION_STRICT_MODE=true
```

### Data Source Configuration

```json
{
  "sources": {
    "ccxt": {
      "enabled": true,
      "exchanges": ["binance", "coinbase", "kraken"],
      "rate_limit": 100
    },
    "yahoo": {
      "enabled": true,
      "timeout": 30
    },
    "kaggle": {
      "enabled": true,
      "datasets": ["bitcoin-historical-data", "crypto-market-data"]
    }
  }
}
```

## Usage Examples

### Starting the Service

```bash
# Development mode
python -m services.data_service.main

# Production mode
uvicorn services.data_service.main:app --host 0.0.0.0 --port 8001
```

### Fetching Market Data

```python
import httpx

# Fetch BTC/USDT data
response = await httpx.get("http://localhost:8001/api/data/fetch/BTC/USDT?timeframe=1h&limit=100")
data = response.json()

# Bulk data fetching
symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
response = await httpx.post("http://localhost:8001/api/data/bulk", json={
    "symbols": symbols,
    "timeframe": "1h",
    "limit": 50
})
```

### Alternative Data Collection

```python
# Get sentiment data
response = await httpx.get("http://localhost:8001/api/data/alternative/sentiment?symbol=BTC")
sentiment = response.json()

# Get on-chain metrics
response = await httpx.get("http://localhost:8001/api/data/alternative/onchain?symbol=BTC")
onchain = response.json()

# Get economic indicators
response = await httpx.get("http://localhost:8001/api/data/alternative/economic")
economic = response.json()
```

### Market Regime Analysis

```python
# Get current market regime
response = await httpx.get("http://localhost:8001/api/data/regime/BTC/USDT")
regime = response.json()

# Get regime confidence
response = await httpx.get("http://localhost:8001/api/data/regime/confidence/BTC/USDT")
confidence = response.json()
```

## Data Quality Assurance

### Quality Metrics
- **Completeness**: Percentage of valid data points
- **Accuracy**: Deviation from known good sources
- **Timeliness**: Age of data and update frequency
- **Consistency**: Cross-source validation scores

### Validation Rules
- **OHLC Validation**: Open ≤ High, Close ≤ High, etc.
- **Gap Detection**: Missing data point identification
- **Outlier Removal**: Statistical outlier detection and filtering
- **Cross-Source Validation**: Consistency checks across data sources

### Quality Scoring
```python
# Quality score calculation
quality_score = (
    completeness_weight * completeness_score +
    accuracy_weight * accuracy_score +
    timeliness_weight * timeliness_score +
    consistency_weight * consistency_score
)
```

## Caching Strategy

### Cache Layers
- **L1 Cache**: In-memory cache for frequently accessed data
- **L2 Cache**: Redis cache for shared data across service instances
- **L3 Cache**: HDF5 files for historical data archive

### Cache Invalidation
- **Time-based**: TTL-based expiration
- **Event-based**: Invalidation on data updates
- **Manual**: Administrative cache clearing
- **Adaptive**: Dynamic TTL based on data volatility

### Performance Optimization
- **Preloading**: Frequently used data preloaded at startup
- **Compression**: Data compression for storage efficiency
- **Indexing**: Fast lookup indexes for time-series data

## Alternative Data Processing

### Sentiment Analysis
- **Source Aggregation**: Multiple social media and news sources
- **Language Processing**: Multi-language sentiment detection
- **Weighted Scoring**: Source credibility and recency weighting
- **Real-time Updates**: Streaming sentiment analysis

### On-Chain Analytics
- **Transaction Monitoring**: Large transaction tracking
- **Network Health**: Hash rate, difficulty, block time analysis
- **Whale Tracking**: Large holder movement analysis
- **Derivatives Data**: Futures open interest and funding rates

### Economic Data Integration
- **Central Bank Data**: Interest rate decisions and announcements
- **Macroeconomic Indicators**: GDP, inflation, employment data
- **Geopolitical Events**: News and event impact analysis
- **Currency Strength**: Cross-currency analysis

## Market Regime Detection

### Algorithms
- **Clustering**: K-means and hierarchical clustering for regime identification
- **Hidden Markov Models**: Sequential regime transition modeling
- **Machine Learning**: Supervised classification for regime prediction
- **Statistical Methods**: Volatility-based regime detection

### Features Used
- **Volatility Metrics**: Realized volatility, implied volatility
- **Trend Indicators**: Moving averages, trend strength
- **Momentum**: RSI, MACD, stochastic oscillators
- **Volume Analysis**: Volume patterns and anomalies

### Regime Types
- **Bull Market**: Strong upward trends with low volatility
- **Bear Market**: Strong downward trends with high volatility
- **Sideways**: Low volatility, no clear trend
- **High Volatility**: Extreme price movements regardless of direction

## Deployment

### Docker

```dockerfile
FROM python:3.10-slim

COPY services/data_service/ /app/
COPY requirements.txt /app/

RUN pip install -r requirements.txt

EXPOSE 8001
CMD ["python", "-m", "services.data_service.main"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: data-service
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: data-service
        image: tradpal/data-service:latest
        ports:
        - containerPort: 8001
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Monitoring

### Health Checks

The service provides comprehensive health checks:

- **Overall Health**: `/health`
- **Component Health**: Individual component status
- **Data Sources**: Connectivity and response times
- **Cache Health**: Hit rates and memory usage

### Metrics

Prometheus metrics are exposed at `/metrics`:

- Data fetch success/failure rates
- Cache hit/miss ratios
- Processing latency per data source
- Alternative data collection rates
- Regime detection accuracy

### Logging

Structured logging with correlation IDs:

```json
{
  "timestamp": "2025-01-21T10:30:00Z",
  "level": "INFO",
  "service": "data_service",
  "component": "data_sources",
  "request_id": "req-12345",
  "symbol": "BTC/USDT",
  "message": "Fetched 100 candles from CCXT"
}
```

## Development

### Project Structure

```
services/data_service/
├── main.py              # Main service orchestrator
├── api/
│   └── main.py          # FastAPI application (migrated)
├── data_sources/
│   └── service.py       # Core data fetching (migrated)
├── alternative_data/
│   ├── main.py          # Alternative data service (migrated)
│   ├── sentiment_analyzer.py
│   ├── onchain_collector.py
│   └── economic_collector.py
├── market_regime/
│   └── client.py        # Market regime detection (migrated)
├── tests/               # Test files
├── requirements.txt     # Dependencies
└── README.md           # This file
```

### Testing

```bash
# Unit tests
pytest tests/unit/services/data_service/

# Integration tests
pytest tests/integration/test_data_service.py

# Data quality tests
pytest tests/data_quality/test_data_validation.py
```

### Adding New Data Sources

1. **Implement Data Fetcher**: Create new fetcher class in `data_sources/`
2. **Add to Service**: Register in `DataService.__init__()`
3. **Configure**: Add environment variables for the new source
4. **Test**: Add comprehensive tests for the new source

## Migration Notes

This service consolidates functionality from:

- `services/data_service/` - Core data management
- `services/alternative_data_service/` - Alternative data collection
- `services/market_regime_detection_service/` - Market regime analysis

### Breaking Changes

- API endpoints changed from individual services to `/api/data/*` paths
- Direct service communication should go through data service
- Some configuration parameters renamed for consistency

### Backward Compatibility

- Legacy API endpoints maintained during transition
- Gradual migration with feature flags
- Comprehensive testing ensures no data loss

## Performance

### Benchmarks

- **Data Fetching**: <500ms average response time for cached data
- **Alternative Data**: <2s processing time for sentiment analysis
- **Regime Detection**: <100ms classification time
- **Cache Hit Rate**: >90% for frequently accessed data

### Scaling

- Horizontal scaling through Kubernetes
- Redis clustering for distributed caching
- Data source rate limiting and backoff
- Async processing for heavy computations

## Security

### Data Validation

- Input sanitization for all data sources
- Rate limiting to prevent abuse
- API key management for external services
- Data encryption at rest and in transit

### Access Control

- Service-level authentication through core service
- Data access logging and audit trails
- Sensitive data masking in logs
- Compliance with data protection regulations

## Troubleshooting

### Common Issues

1. **Data Source Unavailable**: Check network connectivity and API keys
2. **Cache Misses**: Verify Redis connectivity and cache configuration
3. **High Latency**: Check data source rate limits and implement backoff
4. **Memory Issues**: Monitor cache size and implement cleanup policies

### Debugging

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python -m services.data_service.main
```

Check data quality:

```bash
curl http://localhost:8001/api/data/quality/BTC/USDT
```

Monitor cache performance:

```bash
curl http://localhost:8001/api/cache/stats
```

## Contributing

1. Follow the established patterns for data sources
2. Add comprehensive tests for new features
3. Update documentation and API specs
4. Ensure data quality validation
5. Get code review approval

## License

MIT License - see LICENSE file for details.