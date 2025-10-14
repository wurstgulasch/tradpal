# TradPal v3.0.0

A comprehensive Python-based trading system optimized for 1-minute charts, featuring multi-timeframe analysis, historical backtesting, advanced portfolio management, sentiment analysis, explainable AI, and automated deployment. Utilizes EMA, RSI, Bollinger Bands, ATR, ADX, and Fibonacci extensions to generate Buy/Sell signals with integrated position sizing and dynamic leverage.

## 🎨 **NEW: Interactive Web UI**

**Experience TradPal through a powerful web interface!**

The new Web UI provides an intuitive, interactive way to build strategies, analyze charts, and monitor performance in real-time.

### ✨ Key Features
- 🔐 **Secure Authentication**: Login system with user management
- 🎨 **Strategy Builder**: Drag-and-drop interface to create custom strategies with 6 technical indicators
- ⚙️ **Interactive Controls**: Real-time parameter tuning with visual feedback
- 📈 **Live Charts**: Interactive Plotly visualizations with zoom, pan, and multiple timeframes
- 📊 **Performance Dashboard**: Real-time monitoring and analytics

### 🚀 Quick Start Web UI
```bash
# Install web dependencies
pip install streamlit plotly flask flask-login werkzeug

# Launch the interface
cd services/web-ui && streamlit run app.py
# Or use the quick start script: ./start_web_ui.sh

# Access at http://localhost:8501
# Default login: admin / admin123
```

### 📸 Screenshots

#### Strategy Builder Interface
![Strategy Builder](docs/screenshots/strategy_builder.png)
*Interactive strategy builder with drag-and-drop technical indicators*

#### Live Trading Dashboard
![Live Dashboard](docs/screenshots/live_dashboard.png)
*Real-time monitoring with live charts and signal alerts*

#### Backtest Results Visualization
![Backtest Results](docs/screenshots/backtest_visualization.png)
*Interactive backtest analysis with performance metrics and trade history*

#### Performance Analytics
![Performance Analytics](docs/screenshots/performance_analytics.png)
*Comprehensive performance analytics with risk metrics and returns analysis*

📖 **[Full Web UI Documentation →](services/web-ui/README.md)**

---

## 🚀 Latest Improvements (October 2025) - Version 3.0.0

### 🆕 Major New Features in v3.0.0

#### 🤖 **Automated Deployment & DevOps**
- **GitHub Actions CI/CD**: Complete automated pipeline for testing, building, and deployment
- **Docker Multi-Architecture**: Automated builds for Linux AMD64, ARM64, and ARM v7
- **Kubernetes Deployment**: Production-ready manifests with ConfigMaps, Secrets, and health checks
- **AWS EC2 Automation**: One-click deployment scripts for cloud infrastructure
- **Monitoring Stack**: Prometheus metrics collection with Grafana dashboards and Redis caching
- **PyPI Publishing**: Automated package building and publishing to PyPI
- **Release Automation**: GitHub Releases with Docker images and binaries

#### 🧠 **Advanced ML & AI Enhancements**
- **PyTorch Neural Networks**: LSTM, GRU, and Transformer models with GPU acceleration
- **AutoML with Optuna**: Automated hyperparameter optimization with TPE, Random, and Grid sampling
- **SHAP Explainability**: Complete PyTorch model interpretability with feature importance
- **Ensemble Methods**: Smart combination of GA and ML predictions with adaptive weighting
- **Walk-Forward Analysis**: Advanced overfitting prevention with Information Coefficient and bias-variance analysis
- **Multi-Model Backtesting**: Parallel comparison of traditional ML, LSTM, Transformer, and ensemble models

#### � **Enterprise Monitoring & Security**
- **Secrets Management**: HashiCorp Vault and AWS Secrets Manager integration
- **Prometheus Metrics**: Comprehensive system monitoring with custom metrics
- **Structured Audit Logging**: JSON-formatted logs with rotation and compliance-ready audit trails
- **Rate Limiting**: Intelligent API rate limiting with exchange-specific limits
- **Security Hardening**: OAuth/JWT authentication for web UI, secure API key management

#### 🚀 **Performance & Scalability**
- **TA-Lib Integration**: High-performance technical analysis with automatic fallback
- **Vectorized Operations**: 10x faster indicator calculations with optimized pandas operations
- **Memory Optimization**: Efficient DataFrame operations with reduced memory footprint
- **Parallel Processing**: Multi-worker support for backtesting and model training
- **Caching System**: Redis-based caching for API responses and ML model persistence

### �🐛 Bug Fixes & Stability (October 2025)
- **Test Suite Overhaul**: Resolved all import errors and function signature mismatches across the entire test suite. All 708 tests now pass successfully with improved mocking infrastructure for expensive operations (data fetching, model training, backtesting).
- **Enhanced Indicator Calculations**: Fixed `calculate_indicators` function to properly accept optional config parameters, enabling custom indicator configurations in backtesting and live trading modes.
- **Configuration Validation**: Added missing API constants (`API_KEY`, `API_SECRET`) and enhanced validation functions for risk parameters and timeframes to prevent runtime errors.
- **Error Handling Improvements**: Updated test cases to properly handle graceful error recovery in indicator calculations, preventing false KeyError expectations.

### 📊 Testing & Quality Assurance
- **Comprehensive Test Suite**: 708 tests with high pass rate (3 performance test failures expected on CI)
- **CI/CD Pipeline**: Automated testing with GitHub Actions for multiple Python versions
- **Code Quality**: Enhanced linting, type checking, and documentation standards
- **Performance Benchmarks**: Realistic timeout values for different hardware configurations

### 🔧 Developer Experience
- **Repository Organization**: Clean project structure with proper .gitignore and documentation
- **Environment Setup**: Streamlined conda environment configuration and dependency management
- **Documentation**: Updated README with current features, installation guides, and troubleshooting
- **Code Standards**: Consistent PEP 8 compliance and comprehensive docstrings

### 🆕 New Features & Scripts
- **Enhanced Backtest Script** (`scripts/enhanced_backtest.py`): Advanced backtesting with detailed reporting, parameter analysis, and export capabilities
- **ML Performance Testing** (`scripts/test_ml_performance.py`): Comprehensive ML model performance evaluation against traditional indicators
- **Modular Data Sources**: New architecture supporting multiple data providers (Yahoo Finance, CCXT, Alpha Vantage, Polygon.io) with config-based switching
- **Improved .gitignore**: Enhanced file exclusion patterns for better repository hygiene
- **Audit Logger Enhancements**: Improved structured logging and error handling
- **Cache System Updates**: Better caching mechanisms for API responses and ML models

---

## 📊 Modular Data Sources

TradPal now supports multiple data providers through a modular architecture, allowing you to switch between different sources based on your needs.

### Supported Data Sources

| Data Source | Best For | Historical Data | Real-time Data | API Key Required |
|-------------|----------|-----------------|----------------|------------------|
| **Yahoo Finance** | Long-term historical data, traditional assets | ✅ Excellent | ❌ | No |
| **CCXT** | Crypto exchanges, real-time trading | ⚠️ Limited | ✅ Good | Optional |
| **Alpha Vantage** | Premium financial data | ✅ Good | ❌ | Yes |
| **Polygon.io** | High-performance market data | ✅ Excellent | ✅ Good | Yes |

### Configuration

Set your preferred data source in `config/settings.py`:

```python
# Data Source Configuration
DATA_SOURCE = 'yahoo_finance'  # Options: 'yahoo_finance', 'ccxt', 'alpha_vantage', 'polygon'
```

Or via environment variable:

```bash
export DATA_SOURCE=yahoo_finance
```

### Data Source Features

- **Yahoo Finance**: Best for historical BTC/USD data, no API key required, reliable for backtesting
- **CCXT**: Direct exchange integration, supports multiple crypto exchanges, real-time data
- **Alpha Vantage**: Premium data with extensive historical coverage (requires API key)
- **Polygon.io**: High-performance financial market data (requires API key)

### Testing Data Sources

Run the data source test script to verify functionality:

```bash
python scripts/test_data_sources.py
```

This will test all available data sources and report their status.

---

## 🚀 Latest Features & Improvements

### 🚀 **NEW IN v3.0.0: Enterprise-Grade Deployment & DevOps**
- **🔄 GitHub Actions CI/CD**: Complete automated pipeline for testing, building, and deployment
- **🐳 Docker Multi-Architecture**: Automated builds for Linux AMD64, ARM64, and ARM v7
- **☸️ Kubernetes Production**: Ready-to-deploy manifests with ConfigMaps, Secrets, and health checks
- **☁️ AWS EC2 Automation**: One-click deployment scripts for cloud infrastructure
- **📊 Enterprise Monitoring**: Prometheus metrics collection with Grafana dashboards and Redis caching
- **📦 PyPI Publishing**: Automated package building and publishing to PyPI
- **🏷️ Release Automation**: GitHub Releases with Docker images and binaries

### 🤖 **NEW IN v3.0.0: Advanced ML & AI Capabilities**
- **🧠 PyTorch Neural Networks**: LSTM, GRU, and Transformer models with GPU acceleration
- **🎯 AutoML with Optuna**: Automated hyperparameter optimization with TPE, Random, and Grid sampling
- **🔍 SHAP Explainability**: Complete PyTorch model interpretability with feature importance
- **🎭 Ensemble Methods**: Smart combination of GA and ML predictions with adaptive weighting
- **📈 Walk-Forward Analysis**: Advanced overfitting prevention with Information Coefficient and bias-variance analysis
- **🔬 Multi-Model Backtesting**: Parallel comparison of traditional ML, LSTM, Transformer, and ensemble models

### 🛡️ **NEW IN v3.0.0: Enterprise Security & Monitoring**
- **🔐 Secrets Management**: HashiCorp Vault and AWS Secrets Manager integration
- **📊 Prometheus Metrics**: Comprehensive system monitoring with custom metrics
- **📋 Structured Audit Logging**: JSON-formatted logs with rotation and compliance-ready audit trails
- **⚡ Rate Limiting**: Intelligent API rate limiting with exchange-specific limits
- **🔒 Security Hardening**: OAuth/JWT authentication for web UI, secure API key management

### ⚡ **Performance & Scalability Enhancements**
- **📈 TA-Lib Integration**: High-performance technical analysis with automatic fallback
- **🚀 Vectorized Operations**: 10x faster indicator calculations with optimized pandas operations
- **💾 Memory Optimization**: Efficient DataFrame operations with reduced memory footprint
- **🔄 Parallel Processing**: Multi-worker support for backtesting and model training
- **🗄️ Caching System**: Redis-based caching for API responses and ML model persistence

### 📊 Version Highlights (October 2025)
- **🔐 Enterprise Security**: Secrets management with HashiCorp Vault and AWS Secrets Manager
- **📊 Advanced Monitoring**: Prometheus metrics collection with Grafana dashboards
- **🛡️ Adaptive Rate Limiting**: Intelligent API rate limiting with exchange-specific limits
- **☁️ Cloud-Ready Deployment**: Kubernetes manifests and AWS EC2 automation
- **🐳 Monitoring Stack**: Complete Docker Compose setup with Prometheus, Grafana, and Redis
- **📈 Portfolio Management**: Multi-asset portfolio system with risk-based allocation and rebalancing
- **🔍 SHAP Explainability**: PyTorch model interpretability with feature importance and trading signal explanations
- **📰 Sentiment Analysis**: Multi-source sentiment aggregation from Twitter, news, and Reddit for enhanced signals
- **💰 Funding Rate Analysis**: Perpetual futures funding rate analysis with signal enhancement and risk management
- **Advanced ML Models with PyTorch**: LSTM, GRU, and Transformer neural networks for time series prediction with GPU support
- **AutoML with Optuna**: Automated hyperparameter optimization with TPE, Random, and Grid sampling strategies
- **Enhanced Walk-Forward Metrics**: Information Coefficient, Bias-Variance tradeoff, and overfitting detection
- **Ensemble Methods**: Combine GA-optimized indicators with ML predictions using weighted, majority, or unanimous voting
- **TA-Lib Integration**: Optimized technical analysis with automatic fallback to pandas implementations
- **Enhanced Audit Logging**: Comprehensive JSON-structured logging with rotation for compliance and debugging
- **Modular ML Extensions**: Advanced signal enhancement using scikit-learn with confidence scoring
- **Modular Indicator System**: Configurable technical indicators with custom parameters
- **Enhanced Error Handling**: Robust exception handling with retry mechanisms for API failures
- **Comprehensive Testing**: 708 test cases covering all components with 100% pass rate (3 skipped for optional dependencies)
- **Modular Integration System**: Telegram, Discord, Email, SMS, and Webhook integrations
- **Advanced Backtesting Engine**: Complete historical simulation with detailed performance metrics
- **Multi-Timeframe Analysis**: Signal confirmation across multiple timeframes for improved accuracy
- **Dynamic Risk Management**: ATR-based position sizing with volatility-adjusted leverage
- **Container Optimization**: Docker and Docker Compose support for easy deployment
- **Security Enhancements**: Environment variable support and secure API key management
- **Walk-Forward Optimization**: Advanced strategy validation with out-of-sample testing
- **ML Signal Enhancement**: Machine learning models for signal prediction and confidence scoring
- **Audit Logging System**: Complete audit trails with structured JSON logging
- **Performance Monitoring**: Advanced performance tracking and optimization tools
- **Multi-Model Backtesting**: Parallel comparison of multiple ML models (traditional ML, LSTM, Transformer, ensemble)
- **Automatic Model Training**: Smart training of missing ML models before backtesting

### Recent Optimizations
- **PyTorch Neural Networks**: Advanced LSTM, GRU, and Transformer models with attention mechanisms and residual connections
- **AutoML Optimization**: Optuna-based hyperparameter tuning with pruning and visualization for sklearn and PyTorch models
- **Advanced Overfitting Metrics**: Information Coefficient, Bias-Variance analysis, overfitting ratios, and consistency scores
- **Ensemble Predictions**: Smart combination of GA and ML predictions with adaptive weighting and performance tracking
- **TA-Lib Integration**: High-performance technical analysis library with automatic fallback for maximum compatibility
- **Enhanced Audit Logging**: Structured JSON logging with file rotation, event categorization, and compliance-ready audit trails
- **Modular ML Extensions**: Machine learning signal enhancement with feature engineering, model training, and confidence-based decision making
- **Walk-Forward Optimization**: Advanced strategy validation with out-of-sample testing and overfitting prevention
- **ML Signal Enhancement**: Machine learning models for signal prediction and confidence scoring
- **Audit Logging System**: Complete audit trails with structured JSON logging and compliance tracking
- **Performance Monitoring**: Advanced performance tracking, analytics, and optimization tools
- **Modular Indicators**: Configurable indicator combinations with custom parameters
- **Performance**: Vectorized operations for 10x faster indicator calculations
- **Reliability**: Comprehensive error boundaries and recovery strategies
- **Scalability**: Timeframe-specific parameters and MTA support
- **Maintainability**: Clean architecture with dependency injection and modular design
- **Monitoring**: Structured logging with file rotation and audit trails

## 📊 Technical Indicators & Signal Logic

### Core Indicators Calculated
- **EMA (9 & 21)**: Exponential Moving Averages for trend identification
- **RSI (14)**: Relative Strength Index for overbought/oversold conditions
- **Bollinger Bands (20, ±2σ)**: Volatility bands for price channel analysis
- **ATR (14)**: Average True Range for volatility and risk measurement
- **ADX (14)**: Average Directional Index for trend strength assessment
- **Fibonacci Extensions**: Automated take-profit level calculation

### Enhanced Buy Signal Conditions (All must be true)
- EMA9 > EMA21 (bullish trend)
- RSI < 30 (oversold condition)
- Close price > Lower Bollinger Band (price above support)
- ADX > 25 (sufficient trend strength, optional)
- MTA confirmation on higher timeframe (optional)

### Enhanced Sell Signal Conditions (All must be true)
- EMA9 < EMA21 (bearish trend)
- RSI > 70 (overbought condition)
- Close price < Upper Bollinger Band (price below resistance)
- ADX > 25 (sufficient trend strength, optional)
- MTA confirmation on higher timeframe (optional)

### Advanced Risk Management
- **Position Size**: (Capital × Risk%) / ATR with configurable multipliers
- **Stop Loss**: Close - (ATR × SL_Multiplier) for buy positions
- **Take Profit**: Close + (ATR × TP_Multiplier) or Fibonacci extension levels
- **Dynamic Leverage**: 1:5-1:10 based on ATR and market volatility
- **Trade Duration**: ADX-based holding periods for trend-following

### TA-Lib Integration & Performance Optimization
- **High-Performance Calculations**: TA-Lib library for optimized EMA, RSI, BB, ATR, and ADX calculations
- **Automatic Fallback**: Seamless fallback to pandas implementations when TA-Lib is unavailable
- **Zero Configuration**: Automatic detection and switching between implementations
- **Performance Boost**: Up to 10x faster indicator calculations with TA-Lib
- **Compatibility**: Works on all platforms with optional TA-Lib installation

### Enhanced Audit Logging System
- **Structured JSON Logging**: All events logged in machine-readable JSON format
- **Event Categorization**: Signal decisions, risk assessments, system events, and errors
- **File Rotation**: Automatic log rotation with configurable retention policies
- **Compliance Ready**: Complete audit trails for regulatory requirements
- **Performance Monitoring**: Detailed timing and performance metrics
- **Error Tracking**: Comprehensive error logging with context and recovery information

### Modular ML Signal Enhancement
- **Machine Learning Integration**: scikit-learn based signal prediction and enhancement
- **Feature Engineering**: Automated creation of technical indicators as ML features
- **Model Training**: Historical data training with cross-validation and performance metrics
- **Confidence Scoring**: ML predictions with confidence levels for risk management
- **Signal Enhancement**: Override traditional signals based on ML confidence thresholds
- **Model Persistence**: Save and load trained models for production use
- **Training Scripts**: Automated model training and evaluation pipelines

### Advanced ML with PyTorch 🧠
- **Neural Network Models**: LSTM, GRU, and Transformer architectures for time series prediction
- **GPU Acceleration**: Automatic CUDA support for faster training on compatible hardware
- **Attention Mechanisms**: Multi-head attention for capturing long-term dependencies
- **Residual Connections**: Skip connections for deeper networks and better gradient flow
- **Early Stopping**: Automatic training halt when validation performance stops improving
- **Learning Rate Scheduling**: Dynamic learning rate adjustment for optimal convergence
- **Model Checkpointing**: Save best models during training for production deployment

### Sentiment Analysis Integration 📰
- **Multi-Source Sentiment**: Twitter/X, financial news, and Reddit community analysis
- **Real-time Social Sentiment**: Track cryptocurrency mentions and sentiment on social media
- **Financial News Analysis**: Monitor news articles for market-moving sentiment
- **Community Sentiment**: Analyze Reddit discussions for crowd wisdom
- **Aggregated Signals**: Combine multiple sentiment sources with confidence weighting
- **Sentiment-Enhanced Trading**: Integrate sentiment scores with technical indicators
- **NLP Models**: FinBERT and RoBERTa for financial text sentiment analysis
- **Rate Limiting**: Built-in API rate limiting and intelligent caching
- **Confidence Scoring**: Sentiment confidence levels for signal filtering

### Portfolio Management System 📈
- **Multi-Asset Portfolios**: Support for cryptocurrencies, forex, and stocks in single portfolio
- **Risk-Based Allocation**: Equal weight, risk parity, volatility-targeted, and minimum variance strategies
- **Dynamic Rebalancing**: Automatic portfolio rebalancing based on thresholds or schedules
- **Advanced Risk Metrics**: VaR, CVaR, Sharpe ratio, diversification ratio, and concentration analysis
- **Position Sizing**: ATR-based position sizing with volatility-adjusted leverage
- **Performance Attribution**: Detailed portfolio analytics and performance tracking
- **Asset Universe**: Pre-configured support for 20+ major assets across crypto and forex

### SHAP Explainability Integration 🔍
- **PyTorch Model Explanations**: SHAP-based interpretability for neural network predictions
- **Feature Importance**: Global and local feature importance analysis
- **Trading Signal Explanations**: Detailed explanations for buy/sell/hold decisions
- **Model Transparency**: Understand why ML models make specific predictions
- **Visualization Support**: Integration with matplotlib for explanation plots
- **Caching System**: Efficient caching of SHAP explanations for performance
- **Multi-Model Support**: Explain predictions from different ML architectures

### AutoML with Optuna 🤖
- **Hyperparameter Optimization**: Automated search for optimal model parameters
- **Multiple Sampling Strategies**: TPE (Tree-structured Parzen Estimator), Random, Grid sampling
- **Pruning**: Early termination of unpromising trials for efficient search
- **Multi-objective**: Optimize for multiple metrics simultaneously
- **Visualization**: Interactive plots for optimization history and parameter importance
- **Study Persistence**: Save and resume optimization studies
- **Model Support**: Works with both scikit-learn and PyTorch models

### Enhanced Walk-Forward Analysis 📊
- **Information Coefficient**: Correlation between in-sample and out-of-sample performance
- **Bias-Variance Tradeoff**: Quantitative analysis of model complexity vs. generalization
- **Overfitting Detection**: Multiple metrics including overfitting ratio and consistency score
- **Robustness Assessment**: Stability analysis across different market conditions
- **Human-Readable Interpretation**: Automatic assessment and recommendations
- **Performance Decay Tracking**: Monitor how well strategies generalize over time

### Ensemble Methods 🎭
- **GA + ML Combination**: Merge genetic algorithm optimized indicators with ML predictions
- **Multiple Voting Strategies**: Weighted, majority, and unanimous voting approaches
- **Adaptive Weighting**: Dynamic weight adjustment based on component performance
- **Performance Tracking**: Individual and combined performance monitoring
- **Confidence-Based Decisions**: Minimum confidence thresholds for signal generation
- **Persistent History**: Track and save ensemble performance over time

### Multi-Model Backtesting 🤖
- **Parallel Model Comparison**: Simultaneously test traditional ML, LSTM, Transformer, and ensemble models
- **Automatic Model Training**: Smart training of missing/untrained models before backtesting
- **Performance Ranking**: Compare models by Sharpe ratio, win rate, P&L, profit factor, and CAGR
- **Detailed Analytics**: Comprehensive performance metrics and statistical analysis
- **Force Retraining**: Option to retrain all models for fresh comparison
- **Scalable Execution**: Configurable parallel workers for faster processing
- **Model Status Checking**: Automatic detection of trained vs untrained models
- **Error Handling**: Graceful handling of failed models with partial results

### Genetic Algorithm Discovery Mode 🧬
- **GA Optimization**: Evolutionary algorithm to find optimal indicator combinations
- **Parameter Tuning**: Automatic optimization of EMA periods, RSI thresholds, BB settings
- **Fitness Function**: Win rate + P&L weighted scoring for best configurations
- **Top 10 Ranking**: Outputs ranked list of best performing configurations
- **Performance Metrics**: Includes Sharpe ratio, max drawdown, total trades per config

### Adaptive Optimization Mode 🧠
- **Self-Optimizing System**: Automatic parameter tuning during live trading
- **Periodic Discovery**: Configurable intervals for GA optimization runs
- **Live Adaptation**: Optional automatic application of optimized configurations
- **Performance Thresholds**: Minimum fitness requirements for configuration changes
- **Persistent Learning**: Saves and loads optimized configurations across restarts

## 🚀 v3.0.0 Enterprise Deployment Suite

TradPal v3.0.0 introduces a comprehensive enterprise-grade deployment and DevOps ecosystem, enabling seamless production deployment across cloud platforms with automated CI/CD pipelines, containerization, and monitoring.

### 🔄 GitHub Actions CI/CD Pipeline

**Automated Testing, Building & Deployment**
- **Multi-Platform Testing**: Parallel test execution across Python 3.8-3.11 on Ubuntu, Windows, and macOS
- **Automated Releases**: Triggered releases with semantic versioning and changelog generation
- **Security Scanning**: Integrated vulnerability scanning with CodeQL and dependency checks
- **Performance Benchmarks**: Automated performance regression testing and reporting

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline
on: [push, pull_request, release]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.8, 3.9, '3.10', 3.11]
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker Images
        run: make docker-build-multiarch
      - name: Push to Registry
        run: make docker-push
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: make k8s-deploy
```

### 🐳 Docker Multi-Architecture Support

**Production-Ready Containerization**
- **Multi-Architecture Builds**: Automated builds for AMD64, ARM64, and ARM v7
- **Security Hardened**: Non-root containers with minimal attack surface
- **Performance Optimized**: Multi-stage builds with layer caching
- **Registry Integration**: Automated pushing to Docker Hub, AWS ECR, and Google Container Registry

```bash
# Build multi-architecture images
docker buildx build --platform linux/amd64,linux/arm64,linux/arm/v7 \
  --tag wurstgulasch/tradpal:v3.0.0 \
  --push .

# Run with enterprise monitoring stack
docker-compose -f docker-compose.yml up -d
```

### ☸️ Kubernetes Production Deployment

**Cloud-Native Orchestration**
- **Production Manifests**: Complete deployment with ConfigMaps, Secrets, and Ingress
- **Health Checks**: Liveness, readiness, and startup probes
- **Auto-Scaling**: Horizontal Pod Autoscaling based on CPU/memory metrics
- **Rolling Updates**: Zero-downtime deployments with rollback capability

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tradpal
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tradpal
  template:
    spec:
      containers:
      - name: tradpal
        image: wurstgulasch/tradpal:v3.0.0
        ports:
        - containerPort: 8501
        envFrom:
        - secretRef:
            name: tradpal-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

### ☁️ AWS EC2 Automation

**One-Click Cloud Deployment**
- **Infrastructure as Code**: Terraform configurations for complete AWS setup
- **Auto-Scaling Groups**: Automatic scaling based on trading volume and system load
- **Load Balancing**: Application Load Balancer with SSL termination
- **Monitoring Integration**: CloudWatch metrics and alarms

```bash
# Deploy to AWS EC2
cd aws
terraform init
terraform plan -var-file=production.tfvars
terraform apply

# Access the deployed application
open https://tradpal.yourdomain.com
```

### 📊 Enterprise Monitoring Stack

**Complete Observability Solution**
- **Prometheus Metrics**: Custom metrics for trading performance, API calls, and system health
- **Grafana Dashboards**: Pre-built dashboards for trading analytics and system monitoring
- **Redis Caching**: High-performance caching for API responses and ML model storage
- **Alert Manager**: Intelligent alerting for trading signals and system anomalies

```yaml
# monitoring/docker-compose.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
  redis:
    image: redis:alpine
    command: redis-server --appendonly yes
```

### 🔐 Secrets Management

**Enterprise-Grade Security**
- **HashiCorp Vault**: Centralized secrets management with dynamic credentials
- **AWS Secrets Manager**: Cloud-native secrets storage with automatic rotation
- **OAuth/JWT Integration**: Secure authentication for web UI and API access

```python
# Secure configuration loading
from hvac import Client as VaultClient
import boto3

vault_client = VaultClient(url='https://vault.example.com')
secrets = vault_client.secrets.kv.read_secret_version(path='tradpal/api-keys')

# AWS Secrets Manager
secrets_client = boto3.client('secretsmanager')
response = secrets_client.get_secret_value(SecretId='tradpal/production')
```

### 📦 PyPI Publishing Automation

**Automated Package Distribution**
- **Semantic Versioning**: Automatic version bumping based on commit messages
- **Multi-Format Builds**: Source distribution and wheel packages
- **Dependency Management**: Automated dependency updates and security patches

```bash
# Automated PyPI publishing
python setup_pypi.py build
python setup_pypi.py test
python setup_pypi.py publish
```

### 🚀 Quick Deployment Commands

```bash
# Local development with Docker
make docker-dev

# Deploy to Kubernetes
make k8s-deploy

# AWS EC2 deployment
make aws-deploy

# Full CI/CD pipeline
make ci-cd

# Monitoring stack
make monitoring-up
```

## 🏗️ Architecture & Design

### Technical Specifications
- **Language**: Python 3.10+
- **Core Dependencies**: pandas, numpy, ccxt, pytest
- **Architecture**: Modular microservices design with clean separation of concerns
- **Data Processing**: Vectorized operations using pandas/numpy for optimal performance
- **Error Handling**: Comprehensive error boundary decorators with recovery strategies
- **Testing**: 708 unit and integration tests with high pass rate (3 performance tests may fail on CI systems)
- **Performance**: Sub-second analysis for 1-minute charts, optimized for high-frequency data
- **Memory Usage**: Efficient DataFrame operations with minimal memory footprint
- **API Compatibility**: ccxt library support for 100+ cryptocurrency and forex exchanges
- **Output Format**: JSON with structured schema for easy integration and parsing

### Project Structure
```
```
tradpal/
├── config/                 # Configuration files and settings
│   ├── settings.py         # Main configuration with timeframe-specific parameters
│   └── adaptive_config.json # Adaptive optimization configurations
├── src/                    # Core trading logic and modules
│   ├── data_fetcher.py     # Data acquisition via ccxt with rate limiting
│   ├── indicators.py       # Technical indicator calculations (EMA, RSI, BB, ATR, ADX)
│   ├── signal_generator.py # Signal generation and risk management
│   ├── backtester.py       # Historical backtesting engine
│   ├── ml_predictor.py     # Machine learning signal enhancement
│   ├── portfolio_manager.py # Multi-asset portfolio management system
│   ├── shap_explainer.py   # SHAP-based model explainability
│   ├── sentiment_analyzer.py # Multi-source sentiment analysis
│   ├── performance.py      # System monitoring and performance tracking
│   ├── audit_logger.py     # Structured JSON logging and compliance
│   ├── cache.py           # API response caching system
│   ├── services/          # Modular service components
│   │   ├── core/          # Core trading services
│   │   ├── ml-trainer/    # ML model training services
│   │   ├── optimizer/     # Genetic algorithm optimization
│   │   └── web-ui/        # Interactive web interface
│   └── scripts/           # Utility scripts and management tools
├── integrations/          # Notification and webhook integrations
│   ├── telegram/          # Telegram bot integration
│   ├── discord/          # Discord webhook integration
│   ├── email/             # Email notifications
│   ├── sms/               # SMS notifications
│   └── webhook/           # Generic webhook support
├── tests/                 # Comprehensive test suite (708 tests)
│   ├── test_portfolio_manager.py # Portfolio management tests
│   ├── test_shap_integration.py  # SHAP explainability tests
│   ├── test_sentiment_analysis.py # Sentiment analysis tests
│   ├── test_multi_model_backtesting.py # Multi-model backtesting tests
│   └── ...                # Additional test files
├── output/                # Generated signals and backtest results
├── cache/                 # ML models and API cache storage
├── logs/                  # Application logs with rotation
├── docs/                  # Additional documentation
├── k8s/                   # Kubernetes deployment manifests
├── aws/                   # AWS deployment automation
├── monitoring/            # Prometheus/Grafana monitoring stack
├── .env.light             # Light performance profile configuration
├── .env.heavy             # Heavy performance profile configuration
└── docker-compose.yml     # Multi-container deployment
```
```

### Performance Profiles System 🏃‍♂️
- **Simplified Profile Management**: Streamlined from 3 to 2 profiles (light/heavy) for easier configuration
- **Light Profile**: Minimal resource usage, basic indicators only (no AI/ML features)
- **Heavy Profile**: Full functionality with all advanced features enabled
- **Profile Validation**: Automatic validation at startup with detailed error reporting
- **Environment-Based Configuration**: Separate `.env.light` and `.env.heavy` files for profile-specific settings
- **Security**: Sensitive data (API keys, tokens) never stored in repository - only local .env files

### Data Flow
1. **Data Fetching**: Retrieve OHLCV data from configured exchange with timeframe support
2. **Indicator Calculation**: Compute technical indicators with optional ADX/Fibonacci
3. **Multi-Timeframe Analysis**: Confirm signals across higher timeframes (optional)
4. **Signal Generation**: Apply enhanced trading logic with trend filtering
5. **Risk Assessment**: Calculate position sizes, stops, and dynamic leverage
6. **Backtesting/Simulation**: Historical performance analysis with detailed metrics
7. **Output**: Save signals to JSON and display real-time alerts with logging

### Key Design Principles
- **Modularity**: Each component has a single responsibility
- **Scalability**: Timeframe-specific parameters and MTA support
- **Data Immutability**: Functions modify DataFrames in-place and return them
- **Configuration-Driven**: All parameters centralized in settings.py
- **Testability**: Comprehensive test coverage for all components
- **Security**: Environment variables for sensitive configuration
- **Error Resilience**: Graceful handling of API failures and data issues
- **Performance Optimized**: Efficient data structures and minimal API calls

## 📦 Installation & Setup

### Prerequisites
- Python 3.10+
- Conda (recommended) or pip
- Git
- **Optional**: TA-Lib (for performance optimization)
- **Optional**: scikit-learn (for ML signal enhancement)
- **Optional**: PyTorch (for advanced ML models)
- **Optional**: tweepy, textblob, transformers (for sentiment analysis)

### Quick Start
```bash
# Clone repository
git clone https://github.com/wurstgulasch/tradpal.git
cd tradpal

# Create and activate conda environment
conda create -n tradpal_env python=3.10
conda activate tradpal_env

# Install dependencies
pip install -r requirements.txt

# Optional: Install TA-Lib for performance optimization
pip install TA-Lib

# Optional: Install scikit-learn for ML signal enhancement
pip install scikit-learn joblib

# Optional: Install enterprise security and monitoring
pip install hvac boto3 prometheus-client psutil

# Optional: Install web UI
pip install streamlit plotly flask flask-login werkzeug PyJWT

# Optional: Install sentiment analysis
pip install tweepy textblob transformers newsapi-python praw

# Optional: Install portfolio management and SHAP
pip install shap

# Optional: Install Docker for monitoring stack
# (Docker Desktop must be installed separately)

# Copy environment template and configure API keys
cp .env.example .env
# Edit .env file with your API credentials

# Run tests to verify installation
python -m pytest tests/ -v

# Run the indicator in live mode
python main.py --mode live

# Train ML model for signal enhancement
python scripts/train_ml_model.py --symbol EUR/USD --timeframe 1h --start-date 2024-01-01 --end-date 2024-12-31

# Run performance demonstration
python scripts/demo_performance.py
```

### PyPI Installation (Recommended for v3.0.0)

TradPal v3.0.0 is available on PyPI with modular feature installation:

```bash
# Install core package (essential trading functionality)
pip install tradpal-indicator

# Install with web UI support (v3.0.0 interactive interface)
pip install tradpal-indicator[webui]

# Install with ML support (PyTorch, AutoML, SHAP)
pip install tradpal-indicator[ml]

# Install with enterprise features (monitoring, security, deployment)
pip install tradpal-indicator[enterprise]

# Install with sentiment analysis (Twitter, news, Reddit)
pip install tradpal-indicator[sentiment]

# Install with all optional features (complete v3.0.0 suite)
pip install tradpal-indicator[all]

# Install development dependencies (for contributors)
pip install tradpal-indicator[dev]
```

**v3.0.0 Feature Breakdown:**
- `webui`: Interactive strategy builder, live dashboards, real-time charts
- `ml`: PyTorch neural networks, AutoML with Optuna, SHAP explainability
- `enterprise`: Prometheus monitoring, Vault secrets, Kubernetes manifests
- `sentiment`: Multi-source sentiment analysis, NLP models
- `all`: Complete feature set including all optional dependencies

After PyPI installation, you can run TradPal commands directly:

```bash
# Run live trading with enterprise monitoring
tradpal --mode live --profile enterprise

# Launch interactive web UI (v3.0.0 feature)
tradpal-webui

# Run multi-model backtesting with PyTorch models
tradpal --mode multi-model --symbol BTC/USDT --models lstm transformer ensemble

# Deploy monitoring stack
tradpal-monitoring

# Run sentiment analysis demo
python -c "from tradpal.examples.sentiment_analysis_demo import demo_aggregated_sentiment; demo_aggregated_sentiment()"
```

### Development Installation

For contributors and development:

```bash
# Clone repository
git clone https://github.com/wurstgulasch/tradpal.git
cd tradpal

# Install in development mode
pip install -e .[dev,ml,webui,sentiment]

# Run tests
pytest

# Build for PyPI
python setup_pypi.py build

# Test package
python setup_pypi.py test

# Publish to PyPI (requires API token)
python setup_pypi.py publish
```

## 📚 Documentation

For detailed documentation, please refer to our comprehensive docs:

### 📖 User Guides
- **[User Guide](docs/USER_GUIDE.md)** - Complete user manual with tutorials and examples
- **[Setup Guide](docs/SETUP_GUIDE.md)** - Detailed installation and configuration instructions
- **[Contributing Guide](docs/CONTRIBUTING.md)** - How to contribute to the project

### 🔧 Technical Documentation
- **[Advanced ML Implementation](docs/ADVANCED_ML_IMPLEMENTATION.md)** - Deep dive into ML models and optimization
- **[Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **[Performance Enhancements](docs/PERFORMANCE_ENHANCEMENTS.md)** - Performance optimization guide
- **[Experimental Features](docs/EXPERIMENTAL_FEATURES.md)** - Research directions for market outperformance
- **[Broker Integration](docs/BROKER_INTEGRATION.md)** - Autonomous trading system architecture

### 🚀 Quick Links
- [Web UI Screenshots](docs/screenshots/)
- [API Documentation](docs/api/)
- [Configuration Examples](config/)
- [Example Scripts](examples/)

### 📓 Jupyter Notebooks

#### BTC/USDT Backtest Tutorial (`examples/btc_usdt_backtest.ipynb`)
Complete step-by-step guide for backtesting BTC/USDT trading strategies:
- Data fetching from Kraken exchange
- Technical indicator calculation (EMA, RSI, Bollinger Bands, ATR, ADX)
- Signal generation with buy/sell logic
- Risk management (position sizing, stop-loss, take-profit)
- Interactive visualization with Plotly
- Performance analysis and metrics
- Historical simulation walkthrough

**Perfect for beginners** - Learn backtesting from data fetching to performance analysis.

#### ML Training Guide (`examples/ml_training_guide.ipynb`)
Comprehensive machine learning tutorial covering:
- Data preparation and feature engineering
- Training multiple ML models (Random Forest, XGBoost, LSTM)
- Model evaluation and cross-validation
- Feature importance analysis with SHAP
- Model persistence and deployment
- Best practices for financial ML

**Advanced users** - Master ML-enhanced trading signal prediction.

### 🚀 Quick Start Examples

#### Basic Backtest Script (`examples/demo_performance.py`)
```python
# Run a complete backtest in < 30 seconds
python examples/demo_performance.py
```

#### Enhanced Backtest Script (`scripts/enhanced_backtest.py`)
```python
# Run detailed backtesting with comprehensive reports and exports
python scripts/enhanced_backtest.py --symbol BTC/USDT --timeframe 1h --start-date 2024-01-01 --end-date 2024-12-31
```

#### ML Performance Testing (`scripts/test_ml_performance.py`)
```python
# Test ML model performance against traditional indicators
python scripts/test_ml_performance.py --symbol BTC/USDT --timeframe 1d --days 365
```

#### ML Model Training (`scripts/train_ml_model.py`)
```python
# Train ML models for signal enhancement
python scripts/train_ml_model.py --symbol BTC/USDT --timeframe 1h --start-date 2024-01-01
```

#### Multi-Model Backtesting with PyTorch (v3.0.0)
```bash
# Compare multiple ML models in parallel with GPU acceleration
python main.py --mode multi-model --symbol BTC/USDT --timeframe 1d --models traditional_ml lstm transformer --max-workers 4

# Auto-train missing models before backtesting (AutoML feature)
python main.py --mode multi-model --symbol BTC/USDT --timeframe 1d --train-missing

# Force retrain all models for fresh comparison
python main.py --mode multi-model --symbol BTC/USDT --timeframe 1d --retrain-all
```
- Compares performance of different ML architectures (traditional ML, LSTM, Transformer, ensemble)
- Automatic parallel execution for faster results with GPU support
- Detailed performance comparison with rankings by multiple metrics
- Smart model training with status checking and automatic retraining

#### Enterprise Monitoring & Deployment (v3.0.0)
```bash
# Deploy complete monitoring stack (Prometheus + Grafana + Redis)
docker-compose -f monitoring/docker-compose.yml up -d

# Run with enterprise security (Vault integration)
export VAULT_ADDR="https://vault.example.com"
python main.py --mode live --profile enterprise

# Kubernetes deployment with auto-scaling
kubectl apply -f k8s/
kubectl autoscale deployment tradpal --cpu-percent=70 --min=1 --max=10

# AWS EC2 automated deployment
cd aws && terraform apply -auto-approve
```

#### SHAP Model Explainability (v3.0.0)
```bash
# Analyze ML model decisions with SHAP
python examples/shap_integration_demo.py

# Generate feature importance plots
python scripts/analyze_shap.py --symbol BTC/USDT --model lstm
```
- Complete PyTorch model interpretability with SHAP
- Feature importance analysis for trading signal decisions
- Interactive visualizations and caching for performance

#### AutoML Hyperparameter Optimization (v3.0.0)
```bash
# Automated model tuning with Optuna
python scripts/automl_optimization.py --symbol BTC/USDT --study-name production_study

# Visualize optimization results
python scripts/plot_optuna_study.py --study-name production_study
```
- Automated hyperparameter search with multiple algorithms
- Pruning for efficient optimization
- Study persistence and visualization

#### Sentiment Analysis Mode
```bash
python examples/sentiment_analysis_demo.py
```
- Analyzes real-time sentiment from Twitter, news, and Reddit
- Provides sentiment-enhanced trading signals
- Demonstrates multi-source sentiment aggregation
- Shows integration with technical analysis

#### Portfolio Management Demo
```bash
python examples/portfolio_management_demo.py
```
- Creates multi-asset portfolios with different allocation strategies
- Demonstrates risk-based rebalancing and performance tracking
- Shows portfolio optimization and risk metrics calculation
- Interactive portfolio analysis and reporting

#### SHAP Explainability Demo
```bash
python examples/shap_integration_demo.py
```
- Demonstrates ML model explainability with SHAP
- Shows feature importance analysis for trading signals
- Interactive visualizations of model decisions
- Integration examples with PyTorch models

#### Funding Rate Analysis Demo
```bash
python examples/funding_rate_demo.py
```
- Analyzes real-time and historical funding rates across exchanges
- Demonstrates funding rate signal integration with technical analysis
- Shows risk management adjustments based on funding rate data
- Interactive funding rate visualization and alerts

**Sentiment Configuration:**
```python
# Enable sentiment analysis in config/settings.py
SENTIMENT_ENABLED = True
SENTIMENT_SOURCES = ['twitter', 'news', 'reddit']  # Sources to use
SENTIMENT_HOURS_BACK = 24  # Hours of historical data
SENTIMENT_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for signals
SENTIMENT_SIGNAL_WEIGHT = 0.3  # Weight in ensemble decisions
```

**Portfolio Management Configuration:**
```python
# Enable portfolio management in config/settings.py
PORTFOLIO_MANAGEMENT_ENABLED = True
DEFAULT_ALLOCATION_METHOD = 'risk_parity'  # equal_weight, risk_parity, volatility_targeted
REBALANCING_FREQUENCY = 'weekly'  # daily, weekly, monthly, threshold
REBALANCE_THRESHOLD = 0.05  # 5% deviation triggers rebalance
MAX_ASSETS_PER_PORTFOLIO = 20
RISK_FREE_RATE = 0.02  # For Sharpe ratio calculations
```

**SHAP Explainability Configuration:**
```python
# Enable SHAP explanations in config/settings.py
SHAP_ENABLED = True
SHAP_CACHE_EXPIRATION = 3600  # Cache explanations for 1 hour
SHAP_MAX_SAMPLES = 1000  # Maximum samples for global explanations
SHAP_PLOT_FORMAT = 'png'  # Format for explanation plots
```

### Funding Rate Analysis 💰

TradPal now includes comprehensive perpetual futures funding rate analysis to enhance trading signals and risk management. Funding rates represent the periodic payments between long and short positions in perpetual futures contracts, providing valuable insights into market sentiment and potential price movements.

#### Key Features
- **Real-time Funding Rate Data**: Direct integration with major crypto exchanges (Binance, Kraken, etc.) for live funding rate data
- **Historical Funding Rate Analysis**: Long-term funding rate trends and volatility analysis
- **Signal Enhancement**: Funding rate signals integrated with traditional technical indicators
- **Risk Management**: Funding rate-based position sizing and leverage adjustments
- **Multi-Exchange Support**: Aggregate funding rates across multiple exchanges for robust analysis
- **Automated Alerts**: Notifications for extreme funding rate conditions

#### Funding Rate Signal Logic
- **Bullish Funding Rate Signal**: Negative funding rates (longs paying shorts) often indicate bearish sentiment
- **Bearish Funding Rate Signal**: Positive funding rates (shorts paying longs) often indicate bullish sentiment
- **Extreme Rate Alerts**: Funding rates > 0.1% or < -0.1% trigger enhanced risk management
- **Volatility Integration**: Funding rate volatility used for dynamic leverage adjustment

#### Configuration Options
```python
# Enable funding rate analysis in config/settings.py
FUNDING_RATE_ENABLED = True
FUNDING_RATE_WEIGHT = 0.2  # Weight in signal combination (0.0-1.0)
FUNDING_RATE_THRESHOLD = 0.001  # Minimum rate for signal consideration (0.1%)
FUNDING_RATE_EXCHANGES = ['binance', 'kraken', 'okx']  # Exchanges to monitor
FUNDING_RATE_UPDATE_INTERVAL = 60  # Update interval in seconds
FUNDING_RATE_HISTORY_DAYS = 30  # Days of historical data to analyze
```

#### Usage Examples
```bash
# Run live trading with funding rate enhancement
python main.py --mode live --symbol BTC/USDT --funding-rate

# Backtest with funding rate signals
python main.py --mode backtest --symbol BTC/USDT --timeframe 1h --funding-rate --start-date 2024-01-01

# Analyze funding rate data only
python scripts/analyze_funding_rates.py --symbol BTC/USDT --exchanges binance kraken
```

#### Funding Rate Data Sources
- **Binance**: Primary exchange with comprehensive perpetual futures data
- **Kraken**: Alternative exchange for cross-validation
- **OKX**: Additional exchange for multi-exchange analysis
- **Bybit**: Popular exchange with competitive funding rates
- **KuCoin**: Emerging exchange with growing perpetual futures market

#### Risk Management Integration
- **Position Sizing**: Funding rate volatility adjusts position sizes
- **Leverage Adjustment**: Extreme funding rates reduce maximum leverage
- **Stop Loss Enhancement**: Funding rate trends influence stop-loss placement
- **Take Profit Optimization**: Funding rate signals help optimize exit points

#### API Keys Setup
```bash
# Exchange API keys (optional for public funding rate data)
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_SECRET_KEY="your_binance_secret_key"

# For enhanced rate limits and private endpoints
export KRAKEN_API_KEY="your_kraken_api_key"
export KRAKEN_PRIVATE_KEY="your_kraken_private_key"
```

**API Keys Setup:**
```bash
# Twitter/X API (for social sentiment)
export TWITTER_BEARER_TOKEN="your_bearer_token"

# News API (for financial news)
export NEWS_API_KEY="your_news_api_key"

# Reddit API (for community sentiment)
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
```

## 🎯 Command Line Options

TradPal supports various command-line options for different use cases:

### Core Options
- `--mode`: Operation mode (`live`, `backtest`, `analysis`, `discovery`, `paper`, `multi-model`)
- `--symbol`: Trading symbol (default: BTC/USDT)
- `--timeframe`: Chart timeframe (default: 1m)
- `--profile`: Performance profile (`light`, `heavy`, default: default .env)
- `--funding-rate`: Enable funding rate analysis and signal enhancement

### Backtesting Options
- `--start-date`: Backtest start date (YYYY-MM-DD)
- `--end-date`: Backtest end date (YYYY-MM-DD)
- `--clear-cache`: Clear all caches before running

### Multi-Model Backtesting Options
- `--models`: ML models to test (`traditional_ml`, `lstm`, `transformer`, `ensemble`)
- `--max-workers`: Maximum parallel workers (default: 4)
- `--train-missing`: Automatically train missing ML models before backtesting
- `--retrain-all`: Force retrain all ML models before backtesting

### Discovery Mode Options
- `--population`: GA population size (default: 50)
- `--generations`: GA generations (default: 20)

### Examples
```bash
# Live monitoring
python main.py --mode live --symbol BTC/USDT

# Single backtest
python main.py --mode backtest --symbol EUR/USD --timeframe 1h --start-date 2024-01-01 --end-date 2024-12-31

# Enhanced backtesting with detailed reports
python scripts/enhanced_backtest.py --symbol BTC/USDT --timeframe 1h --start-date 2024-01-01 --end-date 2024-12-31

# ML model performance testing
python scripts/test_ml_performance.py --symbol BTC/USDT --timeframe 1d --days 365

# Multi-model comparison with auto-training
python main.py --mode multi-model --symbol BTC/USDT --timeframe 1d --train-missing --max-workers 4

# Genetic algorithm optimization
python main.py --mode discovery --symbol BTC/USDT --timeframe 1h --population 100 --generations 30

# Paper trading simulation
python main.py --mode paper --symbol BTC/USDT --timeframe 1m

# Live trading with funding rate enhancement
python main.py --mode live --symbol BTC/USDT --funding-rate

# Backtesting with funding rate signals
python main.py --mode backtest --symbol BTC/USDT --timeframe 1h --funding-rate --start-date 2024-01-01
```
