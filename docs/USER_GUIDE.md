# TradPal User Guide - v3.0.1

This comprehensive user guide covers all aspects of using TradPal's microservices-based AI trading system, from basic backtesting to advanced live trading with ML models.

## 🚀 Quick Start

### Docker Compose Setup (Recommended)

```bash
# Clone and start all services
git clone https://github.com/wurstgulasch/tradpal.git
cd tradpal_indicator
docker-compose up -d

# Access web UI
open http://localhost:8501
```

### First Backtest

```bash
# Run basic backtest
python main.py --mode backtest --start-date 2024-01-01

# View results in web UI or output folder
ls output/
```

## 📋 Table of Contents

- [Core Concepts](#core-concepts)
- [Configuration](#configuration)
- [Backtesting](#backtesting)
- [Live Trading](#live-trading)
- [ML/AI Features](#mlai-features)
- [Web Interface](#web-interface)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## 🏗️ Core Concepts

### Microservices Architecture

TradPal v3.0.1 uses a microservices architecture where each component runs as an independent service:

- **Core Service** (`services/core/`): Technical indicators, signal generation
- **Data Service** (`services/data_service/`): Multi-source data fetching (Kaggle, Yahoo, CCXT)
- **Trading Bot** (`services/trading_bot_live/`): Live execution with risk management
- **Backtesting Service** (`services/backtesting_service/`): Historical simulation
- **ML Ops Service** (`services/mlops_service/`): AI model training and serving
- **Risk Service** (`services/risk_service/`): Position sizing, drawdown control
- **Notification Service** (`services/notification_service/`): Alerts via Telegram/Discord

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Trading Configuration
TRADING_SYMBOL=BTC/USDT
TRADING_TIMEFRAME=1h
INITIAL_CAPITAL=10000
MAX_POSITION_SIZE=0.1

# API Keys (for live trading)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# ML Configuration
ML_MODELS_ENABLED=true
GPU_ACCELERATION_ENABLED=true

# Performance Settings
MEMORY_OPTIMIZATION_ENABLED=true
PARALLEL_PROCESSING_ENABLED=true

# Notifications
TELEGRAM_BOT_TOKEN=your_telegram_token
DISCORD_WEBHOOK_URL=your_discord_webhook
```

### Service Configuration

Modify `config/settings.py` for advanced settings:

```python
# Performance tuning
MEMORY_OPTIMIZATION_ENABLED = True
CHUNK_SIZE = 1000000  # Data processing chunk size
MAX_WORKERS = 0  # 0 = auto-detect CPU cores

# Risk management
MAX_DRAWDOWN = 0.1  # 10% maximum drawdown
POSITION_SIZE_LIMIT = 0.05  # 5% of capital per trade

# ML settings
LSTM_HIDDEN_SIZE = 128
TRANSFORMER_HEADS = 8
ENSEMBLE_MODELS = ['lstm', 'transformer', 'xgboost']
```

## 📊 Backtesting

### Basic Backtest

```bash
# Simple backtest with default settings
python main.py --mode backtest --start-date 2024-01-01 --end-date 2024-12-31

# Custom symbol and timeframe
python main.py --mode backtest --symbol ETH/USDT --timeframe 4h \
               --start-date 2024-06-01 --capital 50000
```

### Advanced Backtesting

```bash
```bash
# Multi-symbol backtest with ML
python main.py --mode backtest \
               --symbols BTC/USDT,ETH/USDT,SOL/USDT \
               --ml-enabled --gpu \
               --start-date 2024-01-01
```

# Walk-forward analysis
python scripts/walk_forward_analysis.py --symbol BTC/USDT \
                                       --train-window 180 \
                                       --test-window 30 \
                                       --step-size 30
```

### Backtest Results

Results are saved in `output/` folder:
- `backtest_results.json`: Detailed metrics
- `performance_summary.json`: Key performance indicators
- `trades_log.csv`: Individual trades
- `charts/`: Performance charts

Key metrics include:
- **Total Return**: Overall P&L percentage
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-valley decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

## 🤖 Live Trading

### Paper Trading (Recommended First)

```bash
# Start paper trading
python main.py --mode live --paper-trading true \
               --symbol BTC/USDT --capital 10000

# With ML signals
python main.py --mode live --paper-trading true
```

### Live Trading Setup

⚠️ **Caution**: Live trading involves real money. Start with paper trading!

```bash
# Live trading (requires API keys)
python main.py --mode live --symbol BTC/USDT \
               --api-key $BINANCE_API_KEY \
               --api-secret $BINANCE_API_SECRET
```

### Risk Management

```python
# Risk settings in config/settings.py
RISK_PARAMETERS = {
    'max_position_size': 0.02,  # 2% of capital
    'max_drawdown': 0.05,       # 5% max drawdown
    'daily_loss_limit': 0.02,   # 2% daily loss limit
    'max_open_positions': 3,     # Maximum concurrent positions
    'stop_loss_pct': 0.01,       # 1% stop loss
    'take_profit_pct': 0.03      # 3% take profit
}
```

## 🧠 ML/AI Features

### Available Models

- **LSTM**: Time series prediction for price movements
- **Transformer**: Attention-based sequence modeling
- **Ensemble**: Combined predictions from multiple models
- **XGBoost**: Gradient boosting for feature importance

### Training ML Models

```bash
# Train LSTM model
python scripts/train_ml_model.py --model lstm --symbol BTC/USDT \
                                --epochs 100 --batch-size 64

# Train ensemble model
python scripts/train_ml_model.py --model ensemble \
                                --symbols BTC/USDT,ETH/USDT \
                                --gpu
```

### ML Integration in Trading

```bash
# Backtest with ML signals
python main.py --mode backtest --ml-enabled \
               --model lstm --confidence-threshold 0.7

# Live trading with ML
python main.py --mode live --paper-trading true \
               --ml-enabled --model ensemble
```

### Model Performance Monitoring

```bash
# View model metrics
python scripts/debug_ml_signals.py --model lstm

# Analyze feature importance
python scripts/analyze_feature_importance.py
```

## 🌐 Web Interface

### Accessing the Web UI

```bash
# Start web interface
python main.py --mode webui

# Or with Docker
docker-compose up web_ui

# Access at http://localhost:8501
```

### Web UI Features

- **Dashboard**: Real-time P&L, positions, performance metrics
- **Backtesting**: Interactive backtest configuration and results
- **Live Trading**: Monitor active positions and signals
- **ML Models**: Model performance, predictions, feature analysis
- **Charts**: Interactive price charts with indicators
- **Logs**: Real-time system logs and alerts

### API Access

```python
# REST API access
import requests

# Get account balance
response = requests.get('http://localhost:8000/balance')

# Submit order
order = {
    'symbol': 'BTC/USDT',
    'side': 'buy',
    'quantity': 0.001,
    'type': 'market'
}
response = requests.post('http://localhost:8000/order', json=order)
```

## 📈 Monitoring and Analytics

### Grafana Dashboards

Access monitoring at `http://localhost:3000` (admin/admin):

- **Service Health**: Uptime, response times, error rates
- **Trading Performance**: P&L charts, win/loss ratios
- **ML Metrics**: Model accuracy, prediction confidence
- **System Resources**: CPU, memory, disk usage

### Prometheus Metrics

```bash
# Query metrics
curl http://localhost:9090/api/v1/query?query=tradpal_trading_pnl_total

# Service health
curl http://localhost:8000/metrics
```

### Logging

```bash
# View application logs
tail -f logs/tradpal.log

# Docker service logs
docker-compose logs -f trading_service

# Structured logging with levels
export LOG_LEVEL=DEBUG
```

## 🔧 Advanced Usage

### Custom Strategies

Create custom strategies in `services/core/strategies/`:

```python
# services/core/strategies/custom_strategy.py
from services.core.base_strategy import BaseStrategy

class CustomStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Your custom logic here
        sma_short = data['close'].rolling(20).mean()
        sma_long = data['close'].rolling(50).mean()

        signals = pd.Series(0, index=data.index)
        signals[sma_short > sma_long] = 1   # Buy
        signals[sma_short < sma_long] = -1  # Sell

        return signals
```

### Parameter Optimization

```bash
# Genetic algorithm optimization
python main.py --mode discovery --generations 100 \
               --population 50 --symbol BTC/USDT

# Grid search
python scripts/optimize_discovery_params.py --method grid \
                                           --param-range 0.1,0.5
```

### Multi-Asset Trading

```bash
# Portfolio backtest
python scripts/portfolio_management_demo.py \
    --symbols BTC/USDT,ETH/USDT,ADA/USDT \
    --allocation 0.5,0.3,0.2

# Correlation analysis
python scripts/analyze_portfolio_correlation.py
```

## 🐛 Troubleshooting

### Common Issues

#### Service Connection Issues
```bash
# Check service health
curl http://localhost:8001/health  # Core service

# Restart services
docker-compose restart

# Check Docker networks
docker network ls
```

#### Memory Issues
```bash
# Enable memory optimization
export MEMORY_OPTIMIZATION_ENABLED=true

# Use memory profiling
python -m memory_profiler main.py

# Monitor memory usage
docker stats
```

#### ML/GPU Issues
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU usage
export PYTORCH_DEVICE=cpu
export GPU_ACCELERATION_ENABLED=false
```

#### API Rate Limits
```bash
# Check rate limit status
curl http://localhost:8002/rate-limit

# Adjust API settings
export API_REQUESTS_PER_MINUTE=500
```

### Performance Issues

```bash
# Run performance benchmark
python scripts/performance_benchmark.py

# Profile execution
python -m cProfile main.py --mode backtest

# Memory profiling
python -m memory_profiler main.py
```

### Getting Help

1. **Check Logs**: `docker-compose logs -f`
2. **Run Diagnostics**: `python scripts/crash_analysis.py`
3. **GitHub Issues**: Search or create issues
4. **Community**: Join Discord for support

## 📚 Examples and Tutorials

### Example Scripts

```bash
# Run all examples
python examples/demo_performance_features.py
python examples/advanced_ml_examples.py
python examples/portfolio_management_demo.py

# Jupyter notebooks
jupyter notebook examples/
```

### Tutorial: Complete Workflow

1. **Setup**: `docker-compose up -d`
2. **Backtest**: `python main.py --mode backtest`
3. **Optimize**: `python main.py --mode discovery --generations 50`
4. **Paper Trade**: `python main.py --mode live --paper-trading true`
5. **Go Live**: Configure API keys and run live trading

### Best Practices

- Always start with paper trading
- Use stop-loss and position size limits
- Monitor drawdown closely
- Regular model retraining
- Keep detailed trading logs

## 🔐 Security

### API Key Management
- Use read-only keys for data fetching
- Enable 2FA on exchange accounts
- Rotate keys regularly
- Store keys securely (not in code)

### System Security
- Run in isolated Docker containers
- Use mTLS for service communication
- Regular security updates
- Monitor for unauthorized access

## 📞 Support

### Resources
- **Documentation**: `/docs/` folder
- **Examples**: `/examples/` folder
- **Scripts**: `/scripts/` folder
- **Tests**: `/tests/` folder

### Community
- **GitHub**: Issues and discussions
- **Discord**: Real-time support
- **Documentation**: Wiki and guides

### Professional Support
- Enterprise deployment assistance
- Custom model development
- Performance optimization consulting

---

**Version**: v3.0.1
**Last Updated**: October 17, 2025
**Test Coverage**: 100% (490 tests passing)
**Architecture**: Microservices with Docker Compose