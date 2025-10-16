# Setup and Installation

This guide walks you through the installation and configuration of TradPal Indicator.

## 🔧 System Requirements

### Minimum Requirements
- **Python**: 3.10+
- **RAM**: 4GB
- **Storage**: 2GB free
- **OS**: Linux, macOS, Windows

### Recommended Requirements
- **Python**: 3.10+
- **RAM**: 8GB+
- **Storage**: 10GB+ SSD
- **OS**: Linux (Ubuntu 20.04+)

## 📦 Installation

### Option 1: Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/wurstgulasch/tradpal.git
cd tradpal_indicator

# Create conda environment
conda env create -f environment.yml
conda activate tradpal-env

# Install dependencies
pip install -r requirements.txt
```

### Option 2: pip

```bash
# Clone repository
git clone https://github.com/wurstgulasch/tradpal.git
cd tradpal_indicator

# Create virtual environment
python -m venv tradpal_env
source tradpal_env/bin/activate  # Linux/macOS
# or tradpal_env\Scripts\activate  # Windows

# Install dependencies
pip install -e .[dev,ml,webui]
```

### Option 3: Docker

```bash
# Clone repository
git clone https://github.com/wurstgulasch/tradpal.git
cd tradpal_indicator

# Build Docker image
docker build -t tradpal-indicator .

# Start container
docker run -p 8501:8501 tradpal-indicator
```

## ⚙️ Configuration

### Basic Configuration

1. **Configure API Keys**
```bash
# Create .env file
cp config/.env.example config/.env

# Add API keys
echo "BINANCE_API_KEY=your_api_key" >> config/.env
echo "BINANCE_API_SECRET=your_api_secret" >> config/.env
```

2. **Adjust Settings**
```python
# Edit config/settings.py
SYMBOL = 'BTC/USDT'  # Trading pair
TIMEFRAME = '1h'     # Timeframe
CAPITAL = 10000      # Starting capital
```

### Performance Profiles

TradPal supports different performance profiles:

#### Light Profile (Resource-friendly)
```bash
python main.py --profile light
```
- No ML/AI
- Minimal indicators
- Faster execution

#### Heavy Profile (All Features)
```bash
python main.py --profile heavy
```
- All ML models
- Maximum indicators
- Comprehensive analysis

## 🚀 First Execution

### Run Backtest
```bash
# Simple backtest
python main.py --mode backtest --symbol BTC/USDT --timeframe 1d

# With Web UI
python main.py --mode webui
```

### Discovery Mode (Genetic Algorithms)
```bash
# Optimize parameters
python main.py --mode discovery --symbol BTC/USDT --generations 50
```

### Live Trading
```bash
# Paper trading (recommended for testing)
python main.py --mode paper --symbol BTC/USDT

# Live trading (caution!)
python main.py --mode live --symbol BTC/USDT
```

## 🔍 Installation Verification

### Run Tests
```bash
# All tests
pytest tests/ -v

# Specific tests
pytest tests/unit/test_indicators.py -v
pytest tests/unit/test_backtester.py -v
```

### Run Example Script
```python
# Run examples/demo_performance_features.py
python examples/demo_performance_features.py
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```
ModuleNotFoundError: No module named 'talib'
```
**Solution:**
```bash
# Install TA-Lib
conda install -c conda-forge ta-lib
# or
pip install TA-Lib
```

#### 2. Memory Errors
```
MemoryError: Unable to allocate array
```
**Solution:**
- Use more RAM
- Enable light profile
- Reduce data range

#### 3. API Errors
```
APIError: Invalid API key
```
**Solution:**
- Check API keys in `.env`
- Verify IP whitelist at exchange
- Respect rate limits

#### 4. Docker Issues
```
docker: Error response from daemon: pull access denied
```
**Solution:**
- Build Docker image locally
- Check registry credentials

### Check Logs
```bash
# View logs
tail -f logs/tradpal.log

# Enable debug mode
export PYTHONPATH=/app
python main.py --debug
```

## 🔄 Updates

### Automatic Updates
```bash
# Update repository
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
```

### Manual Updates
```bash
# Download new version
wget https://github.com/wurstgulasch/tradpal/releases/latest/download/tradpal-indicator.tar.gz
tar -xzf tradpal-indicator.tar.gz
cd tradpal-indicator

# Repeat installation
pip install -e .
```

## 🌐 Network Configuration

### Proxy Settings
```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

### Firewall
Ensure the following ports are open:
- **8501**: Streamlit Web UI
- **8000**: Prometheus Metrics (optional)

## 📊 Monitoring

### System Monitoring
```bash
# Monitor resource usage
top -p $(pgrep -f tradpal)

# Memory usage
free -h
```

### Application Monitoring
```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health
```

## 🆘 Help

For issues:
1. Read **documentation**
2. Search **GitHub Issues**
3. Create **new Issue**
4. Contact **community**

### Collect Support Information
```bash
# System info
python -c "import sys; print(f'Python: {sys.version}')"

# Dependencies
pip list | grep -E "(pandas|numpy|talib|pytorch)"

# Logs
tail -50 logs/tradpal.log
```