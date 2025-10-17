# TradPal - AI Trading System

TradPal is a fully autonomous AI trading system based on a complete microservices architecture. The goal is consistent outperformance of Buy&Hold and traditional indicators through advanced ML models, ensemble methods, and risk management.

## � October 2025 Highlights
- Reinforcement Learning service integration suite restored with an async-safe `httpx` test client and event-publishing mocks.
- Market Regime Detection integration tests rewritten to match the current FastAPI contract and run end-to-end in CI.
- Core EMA implementation now guarantees the first `period-1` slots are `NaN` regardless of TA-Lib availability, keeping analytics consistent across environments.

## �🏗️ Project Structure

```
tradpal_indicator/
├── services/                    # Microservices Architecture
│   ├── core/                    # Core calculations & Memory optimization
│   ├── data_service/            # Data Management (CCXT, Kaggle, Yahoo Finance, Caching, HDF5)
│   │   └── data_sources/        # Modular data sources (Kaggle Bitcoin Datasets, Exchanges)
│   │       ├── liquidation.py   # Liquidation data with fallback chain
│   │       ├── volatility.py    # Volatility indicators as liquidation proxy
│   │       ├── sentiment.py     # Sentiment analysis data source
│   │       ├── onchain.py       # On-chain metrics data source
│   │       └── factory.py       # Data source factory with 8+ sources
│   ├── trading_bot_live/        # Live-Trading-Engine with AI models
│   ├── backtesting_service/     # Historical simulation
│   ├── discovery_service/       # ML parameter optimization
│   ├── risk_service/            # Risk management & position sizing
│   ├── notification_service/    # Alerts (Telegram, Discord, Email)
│   ├── mlops_service/           # ML experiment tracking
│   ├── security_service/        # Zero-trust authentication
│   ├── event_system/            # Event-Driven Architecture (Redis Streams)
│   └── web_ui/                  # Streamlit/Plotly dashboard
├── config/                      # Central configuration
│   ├── settings.py              # Main configuration
│   ├── .env                     # Environment variables
│   ├── .env.example             # Example configuration
│   ├── .env.light               # Light profile (without AI/ML)
│   └── .env.heavy               # Heavy profile (full features)
├── data/                        # Data directories
│   ├── cache/                   # Cache files
│   ├── logs/                    # Log files
│   └── output/                  # Output files (backtests, reports)
├── infrastructure/              # Infrastructure & deployment
│   ├── deployment/              # AWS, Kubernetes configurations
│   └── monitoring/              # Prometheus, Grafana setups
├── tests/                       # Test suite (organized by best practices)
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── services/                # Service-specific tests
├── scripts/                     # Utility scripts for training/demos
├── examples/                    # Jupyter notebooks and demos
├── integrations/                # External integrations
├── docs/                        # Documentation
├── main.py                      # Hybrid orchestrator with service clients
└── pyproject.toml               # Python project configuration
```

## 🚀 Quick Start for Developers

### New Docker-based Development Environment (Recommended)
```bash
# One-time setup
make setup

# Start services
make dev-up

# Open Web UI (optional)
make dev-ui
```

**Available Services:**
- 📊 Data Service: http://localhost:8001
- 📈 Backtesting Service: http://localhost:8002
- 🎨 Web UI: http://localhost:8501
- 🚀 API Gateway: http://localhost:8000
- 📡 Event Service: http://localhost:8011
- 📊 Monitoring: http://localhost:9090 (Prometheus), http://localhost:3000 (Grafana)

**Useful Commands:**
```bash
make test          # Run tests
make backtest      # Run backtest
make quality-check # Check code quality
make help          # Show all commands
```

📖 **[Detailed Guide](DEVELOPMENT.md)**

### Traditional Setup

### Prerequisites
- Python 3.10+
- Conda/Miniconda
- Git

### Installation

1. **Clone repository:**
   ```bash
   git clone https://github.com/wurstgulasch/tradpal.git
   cd tradpal_indicator
   ```

2. **Set up environment:**
   ```bash
   conda env create -f environment.yml
   conda activate tradpal-env
   ```

3. **Configuration:**
   ```bash
   cp config/.env.example config/.env
   # Edit .env file
   ```

4. **Run tests:**
   ```bash
   pytest tests/
   ```

### Usage

```bash
# Live trading with light profile (without AI)
python main.py --profile light --mode live

# Backtest with all features
python main.py --profile heavy --mode backtest --start-date 2024-01-01

# Performance benchmark
python scripts/performance_benchmark.py
```

## 🧪 Test Organization

The test suite follows best practices for microservices:

- **Unit Tests** (`tests/unit/`): Isolated component tests
- **Integration Tests** (`tests/integration/`): Service interactions
- **Service Tests** (`tests/services/`): Service-specific tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Service-specific tests
pytest tests/services/core/

# Integration focus areas
pytest tests/integration/test_market_regime_service.py
pytest tests/integration/test_reinforcement_learning_service.py
```

## 📊 Performance & Benchmarks

Current benchmarks show significant improvements:

- **Memory Optimization**: 10.25x faster than traditional methods, constant memory usage (~85 MB)
- **GPU Acceleration**: 3-12x faster training and inference depending on model type
- **Parallel Processing**: 4-8x faster on multi-core systems
- **Intelligent Caching**: 10-100x faster for repeated operations
- **Vectorization**: 5-15x faster indicator calculations
- **Test Coverage**: 100% for implemented features (490 tests passing)
- **Data Sources**: Modular architecture with Kaggle Bitcoin Datasets, Yahoo Finance, CCXT for improved backtesting

Detailed benchmarks: [docs/PERFORMANCE_ENHANCEMENTS.md](docs/PERFORMANCE_ENHANCEMENTS.md)

## 🔌 Data Sources Features

TradPal offers a modular data sources architecture for optimal backtesting results:

### Available Data Sources
- **Kaggle Bitcoin Datasets**: High-quality historical Bitcoin data with minute resolution
- **Yahoo Finance**: Stocks, ETFs and cryptocurrencies
- **CCXT Integration**: 100+ crypto exchanges for live data
- **Alternative Data Sources**: Sentiment analysis, on-chain metrics, volatility indicators

### Advanced Fallback System
When primary liquidation data is unavailable (API authentication issues), the system automatically falls back to alternative data sources:

```python
from services.data_service.data_sources.factory import DataSourceFactory

# Automatic fallback chain: Liquidation → Volatility → Sentiment → On-Chain → Open Interest
liquidation_source = DataSourceFactory.create_data_source('liquidation')
data = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=100)

# Data will contain either real liquidation data or proxy data from alternative sources
# with 'data_source' field indicating which fallback was used
```

### Alternative Data Sources

#### Sentiment Analysis (`sentiment.py`)
- **Fear & Greed Index**: Real-time market sentiment from Alternative.me
- **Social Sentiment**: Simulated social media sentiment analysis
- **News Sentiment**: Market news sentiment indicators
- **Market Sentiment**: Overall market psychology proxy

#### On-Chain Metrics (`onchain.py`)
- **Active Addresses**: Daily active blockchain addresses
- **Transaction Volume**: On-chain transaction volumes
- **Hash Rate**: Network mining difficulty proxy
- **Exchange Flows**: Net capital flows to/from exchanges

#### Volatility Indicators (`volatility.py`)
- **Open Interest**: Futures market positioning
- **24h Volume**: Trading volume analysis
- **Funding Rates**: Perpetual futures funding rates
- **Order Book**: Market depth and bid-ask spreads
- **Recent Trades**: Trade flow analysis and momentum

### Data Quality & Validation
- Automatic OHLC validation
- NaN value handling
- Consistent data formats
- Chunked processing for large datasets

## 🏛️ Architecture Principles

- **Microservices-First**: Every new functionality as a separate service
- **Event-Driven**: Redis Streams for real-time service communication
- **API Gateway**: Centralized service routing, authentication, and load balancing
- **Zero-Trust-Security**: mTLS, OAuth/JWT, secrets management
- **Observability**: Prometheus/Grafana monitoring, distributed tracing, metrics, logs
- **Resilience**: Circuit breaker, retry patterns, health checks, chaos engineering

## 🔧 Development

### Adding New Features
1. Create service in `services/`
2. Add tests in `tests/services/`
3. Update documentation
4. Extend CI/CD pipeline

### Code Quality
- **Type Safety**: Complete type hints
- **Testing**: >90% coverage, integration tests
- **Linting**: flake8, mypy
- **Formatting**: black, isort

## 📈 Roadmap 2025

✅ **Completed:**
- Event-Driven Architecture with Redis Streams
- API Gateway with service discovery and load balancing
- Centralized monitoring with Prometheus/Grafana
- Circuit breaker and health check resilience patterns

🔄 **In Progress:**
1. **AI Outperformance**: ML models that consistently outperform benchmarks
2. **Service Optimization**: Performance, scalability, reliability
3. **Advanced Features**: Reinforcement learning, market regime detection, alternative data
4. **Data Sources Expansion**: Additional datasets and real-time feeds for improved backtesting
5. **Enterprise Readiness**: Security, monitoring, deployment automation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Write tests
4. Commit changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to branch (`git push origin feature/AmazingFeature`)
6. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/wurstgulasch/tradpal/issues)
- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)

---

**TradPal v3.0.1** - *Last updated: October 2025*
