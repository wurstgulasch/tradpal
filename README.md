# TradPal - AI Trading System

TradPal is a fully autonomous AI trading system based on a complete microservices architecture. The goal is consistent outperformance of Buy&Hold and traditional indicators through advanced ML models, ensemble methods, and risk management.

## 🎯 October 2025 Highlights
- **98 Test Files**: Comprehensive test coverage with organized test structure (unit/integration/services/e2e)
- **Service Consolidation**: Partial consolidation of 25+ services into unified trading, data, and backtesting services
- **Advanced ML Integration**: ML-enhanced signal generation with ensemble methods and risk management
- **Modular Data Sources**: Kaggle Bitcoin Datasets, Yahoo Finance, CCXT integration for optimal backtesting
- **Centralized Test Suite**: Organized test structure with conftest.py, fixtures, and comprehensive coverage

## 🏗️ Project Structure

```
tradpal/
├── services/                    # Microservices Architecture (25+ services, partial consolidation)
│   ├── core/                    # Core calculations & Memory optimization
│   ├── data_service/            # Data Management (CCXT, Kaggle, Yahoo Finance, caching, HDF5)
│   │   └── data_sources/        # Modular data sources (Kaggle Bitcoin Datasets, Exchanges)
│   │       ├── liquidation.py   # Liquidation data with fallback chain
│   │       ├── volatility.py    # Volatility indicators as liquidation proxy
│   │       ├── sentiment.py     # Sentiment analysis data source
│   │       ├── onchain.py       # On-chain metrics data source
│   │       └── factory.py       # Data source factory with 8+ sources
│   ├── trading_service/         # Consolidated AI-powered trading service
│   │   ├── orchestrator.py      # Main trading orchestrator
│   │   ├── execution/           # Order execution
│   │   ├── risk_management/     # Risk management
│   │   ├── reinforcement_learning/ # RL agents
│   │   ├── market_regime/       # Market regime detection
│   │   └── monitoring/          # Trading monitoring
│   ├── backtesting_service/     # Historical simulation and ML training
│   ├── discovery_service/       # ML parameter optimization
│   ├── risk_service/            # Risk management and position sizing
│   ├── notification_service/    # Alerts (Telegram, Discord, Email)
│   ├── mlops_service/           # ML experiment tracking and model management
│   ├── security_service/        # Zero-trust authentication
│   ├── event_system/            # Event-Driven Architecture (Redis Streams)
│   ├── api_gateway/             # Centralized service routing and authentication
│   └── [20+ additional services]/ # Individual microservices (pending consolidation)
├── config/                      # Central configuration
│   ├── settings.py              # Main configuration (imports from modules)
│   ├── core_settings.py         # Core trading and risk management
│   ├── ml_settings.py           # Machine learning and AI configurations
│   ├── service_settings.py      # Microservices and data mesh settings
│   ├── security_settings.py     # Security and authentication settings
│   ├── performance_settings.py  # Performance optimization settings
│   ├── .env                     # Environment variables
│   ├── .env.example             # Example configuration
│   ├── .env.light               # Light profile (without AI/ML)
│   └── .env.heavy               # Heavy profile (full features)
├── tests/                       # Centralized test suite (98 test files)
│   ├── conftest.py              # Central test configuration and fixtures
│   ├── unit/                    # Unit tests (25+ files)
│   ├── integration/             # Integration tests (13+ files)
│   ├── services/                # Service-specific tests
│   ├── e2e/                     # End-to-end tests
│   ├── config/                  # Configuration tests
│   └── integrations/            # Integration setup tests
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
- **Test Coverage**: 100% for implemented features (537 tests passing)
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

## 📦 Dependency Management

TradPal uses a sophisticated dependency management system designed for microservice independence:

### Central Dependency Catalog
- **Single Source of Truth**: `dependency_catalog.txt` defines approved package versions
- **Version Consistency**: All services use exact versions from the catalog
- **Automated Validation**: Scripts ensure compliance across all services

### Service-Specific Requirements
Each service in `services/` has its own `requirements.txt` with:
- Only packages the service actually needs
- Exact pinned versions from the catalog
- Independent deployment capability

### Management Tools
```bash
# Validate all service dependencies
python scripts/manage_dependencies.py validate

# List all approved dependencies
python scripts/manage_dependencies.py list

# Update package version across all services
python scripts/manage_dependencies.py update <package> <version>
```

### Benefits
- **True Microservice Independence**: Services can be deployed without shared dependencies
- **Version Drift Prevention**: Automated validation prevents conflicts
- **Simplified Updates**: Single command updates versions across all services
- **Clean Architecture**: No dependency conflicts between services

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

**TradPal v2.5.1** - *Last updated: October 21, 2025*
