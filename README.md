# TradPal Indicator

A comprehensive Python-based trading indicator system optimized for 1-minute charts, featuring multi-timeframe analysis, historical backtesting, and advanced risk management. Utilizes EMA, RSI, Bollinger Bands, ATR, ADX, and Fibonacci extensions to generate Buy/Sell signals with integrated position sizing and dynamic leverage.

## 🎨 **NEW: Interactive Web UI**

**Experience TradPal Indicator through a powerful web interface!**

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

📖 **[Full Web UI Documentation →](services/web-ui/README.md)**

---

## 🚀 Latest Features & Improvements

### Version Highlights (October 2025)
- **Advanced ML Models with PyTorch**: LSTM, GRU, and Transformer neural networks for time series prediction with GPU support
- **AutoML with Optuna**: Automated hyperparameter optimization with TPE, Random, and Grid sampling strategies
- **Enhanced Walk-Forward Metrics**: Information Coefficient, Bias-Variance tradeoff, and overfitting detection
- **Ensemble Methods**: Combine GA-optimized indicators with ML predictions using weighted, majority, or unanimous voting
- **TA-Lib Integration**: Optimized technical analysis with automatic fallback to pandas implementations
- **Enhanced Audit Logging**: Comprehensive JSON-structured logging with rotation for compliance and debugging
- **Modular ML Extensions**: Advanced signal enhancement using scikit-learn with confidence scoring
- **Modular Indicator System**: Configurable technical indicators with custom parameters
- **Enhanced Error Handling**: Robust exception handling with retry mechanisms for API failures
- **Comprehensive Testing**: 458+ test cases covering all components with 100% pass rate
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

## 🏗️ Architecture & Design

### Technical Specifications
- **Language**: Python 3.10+
- **Core Dependencies**: pandas, numpy, ccxt, pytest
- **Architecture**: Modular microservices design with clean separation of concerns
- **Data Processing**: Vectorized operations using pandas/numpy for optimal performance
- **Error Handling**: Comprehensive error boundary decorators with recovery strategies
- **Testing**: 458+ unit and integration tests with 100% pass rate
- **Performance**: Sub-second analysis for 1-minute charts, optimized for high-frequency data
- **Memory Usage**: Efficient DataFrame operations with minimal memory footprint
- **API Compatibility**: ccxt library support for 100+ cryptocurrency and forex exchanges
- **Output Format**: JSON with structured schema for easy integration and parsing

### Project Structure
```
tradpal_indicator/
├── config/
│   └── settings.py          # Configuration (parameters, exchanges, output)
├── src/
│   ├── __init__.py          # Package initialization
│   ├── data_fetcher.py      # Data fetching with ccxt library and error handling
│   ├── indicators.py        # Modular technical indicator calculations
│   ├── signal_generator.py  # Signal generation and risk management
│   ├── output.py            # JSON output formatting and saving
│   ├── backtester.py        # Historical backtesting engine
│   ├── discovery.py         # Genetic algorithm optimization system
│   ├── walk_forward_optimizer.py  # Walk-forward analysis and optimization
│   ├── ml_predictor.py      # Machine learning signal enhancement
│   ├── audit_logger.py      # Audit logging and compliance tracking
│   ├── performance.py       # Performance monitoring and analytics
│   ├── cache.py             # API call caching system
│   ├── config_validation.py # Configuration validation utilities
│   ├── input_validation.py  # Input validation utilities
│   ├── logging_config.py    # Logging configuration
│   └── error_handling.py    # Error recovery and logging system
├── services/
│   ├── core/                # Core trading services
│   ├── ml-trainer/          # ML model training services
│   ├── optimizer/           # Optimization services
│   └── web-ui/              # Web interface services
├── integrations/
│   ├── __init__.py
│   ├── telegram/
│   ├── discord/
│   ├── email/
│   ├── sms/
│   └── webhook/
├── scripts/
│   ├── __init__.py
│   ├── manage_integrations.py
│   ├── test_integrations.py
│   ├── run_integrated.py
│   ├── train_ml_model.py
│   └── demo_performance.py
├── output/                  # Generated JSON signal files
├── logs/                    # Application logs
├── tests/                   # Comprehensive test suite
├── main.py                  # Main orchestration script
├── requirements.txt         # Python dependencies
├── pytest.ini              # Test configuration
├── Dockerfile               # Container build configuration
├── docker-compose.yml       # Multi-container orchestration
├── .env.example             # Environment variables template
├── .env.test                # Test environment configuration
├── .github/copilot-instructions.md  # AI assistant guidelines
└── README.md
```

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

### Quick Start
```bash
# Clone repository
git clone https://github.com/wurstgulasch/tradpal_indicator.git
cd tradpal_indicator

# Create and activate conda environment
conda create -n tradpal_env python=3.10
conda activate tradpal_env

# Install dependencies
pip install -r requirements.txt

# Optional: Install TA-Lib for performance optimization
pip install TA-Lib

# Optional: Install scikit-learn for ML signal enhancement
pip install scikit-learn joblib

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

### Environment Configuration
Create a `.env` file in the project root with your API credentials:

```bash
# Exchange API Keys (optional, for authenticated endpoints)
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_API_SECRET=your_kraken_api_secret

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/tradpal_indicator.log

# Advanced Settings
ENABLE_MTA=true
ADX_THRESHOLD=25
FIBONACCI_LEVELS=161.8,261.8
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run with Docker directly
docker build -t tradpal-indicator .
docker run --env-file .env -v $(pwd)/output:/app/output tradpal-indicator
```

## ⚙️ Configuration

Edit `config/settings.py` to customize behavior:

```python
# Trading pair and exchange
SYMBOL = 'EUR/USD'          # Trading pair (ccxt format)
EXCHANGE = 'kraken'         # Exchange name (must be supported by ccxt)
TIMEFRAME = '1m'            # Chart timeframe

# Timeframe-specific parameters for scalability
TIMEFRAME_PARAMS = {
    '1m': {
        'ema_short': 9, 'ema_long': 21, 'rsi_period': 14, 'bb_period': 20,
        'atr_period': 14, 'adx_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70
    },
    '5m': {
        'ema_short': 12, 'ema_long': 26, 'rsi_period': 14, 'bb_period': 20,
        'atr_period': 14, 'adx_period': 14, 'rsi_oversold': 35, 'rsi_overbought': 65
    },
    '1h': {
        'ema_short': 50, 'ema_long': 200, 'rsi_period': 14, 'bb_period': 20,
        'atr_period': 14, 'adx_period': 14, 'rsi_oversold': 40, 'rsi_overbought': 60
    }
}

# Multi-Timeframe Analysis (MTA)
ENABLE_MTA = True           # Enable signal confirmation on higher timeframes
MTA_TIMEFRAMES = ['5m', '15m']  # Higher timeframes for confirmation

# Advanced Indicators
ENABLE_ADX = True           # Enable ADX for trend strength filtering
ADX_THRESHOLD = 25          # Minimum ADX for valid signals
ENABLE_FIBONACCI = True     # Enable Fibonacci extensions for take-profit
FIBONACCI_LEVELS = [161.8, 261.8]  # Fibonacci extension levels

# Modular Indicator Configuration
DEFAULT_INDICATOR_CONFIG = {
    'ema': {'enabled': True, 'periods': [9, 21]},
    'rsi': {'enabled': True, 'period': 14},
    'bb': {'enabled': True, 'period': 20, 'std_dev': 2},
    'atr': {'enabled': True, 'period': 14},
    'adx': {'enabled': False, 'period': 14},
    'fibonacci': {'enabled': False}
}

# Risk management
CAPITAL = 10000             # Total trading capital
RISK_PER_TRADE = 0.01       # Risk per trade (1% of capital)
SL_MULTIPLIER = 1.5         # Stop-loss multiplier (ATR × SL_MULTIPLIER)
TP_MULTIPLIER = 3.0         # Take-profit multiplier (ATR × TP_Multiplier)
LEVERAGE_BASE = 10          # Base leverage
LEVERAGE_MIN = 5            # Minimum leverage
LEVERAGE_MAX = 10           # Maximum leverage

# Backtesting parameters
BACKTEST_START_DATE = '2024-01-01'  # Default backtest start date
BACKTEST_END_DATE = '2024-12-31'    # Default backtest end date

# Data and output
LOOKBACK_DAYS = 7           # Historical data for analysis
OUTPUT_FORMAT = 'json'      # Output format
OUTPUT_FILE = 'output/signals.json'  # Output file path
BACKTEST_OUTPUT_FILE = 'output/signals_backtest.json'  # Backtest output file

# Machine Learning settings
ML_ENABLED = True  # Enable/disable ML signal enhancement
ML_MODEL_DIR = 'cache/ml_models'  # Directory to store trained ML models
ML_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for ML signal override
ML_TRAINING_HORIZON = 5  # Prediction horizon for training labels (periods ahead)
ML_RETRAINING_INTERVAL_HOURS = 24  # How often to retrain models (hours)
ML_MIN_TRAINING_SAMPLES = 1000  # Minimum samples required for training
ML_TEST_SIZE = 0.2  # Fraction of data for testing
ML_CV_FOLDS = 5  # Number of cross-validation folds
ML_FEATURE_ENGINEERING = True  # Enable advanced feature engineering

# Advanced ML Configuration (PyTorch)
ML_USE_PYTORCH = False  # Enable PyTorch models (LSTM, GRU, Transformer)
ML_PYTORCH_MODEL_TYPE = 'lstm'  # Options: 'lstm', 'gru', 'transformer'
ML_PYTORCH_HIDDEN_SIZE = 128  # Hidden layer size for PyTorch models
ML_PYTORCH_NUM_LAYERS = 2  # Number of layers for PyTorch models
ML_PYTORCH_DROPOUT = 0.2  # Dropout rate for regularization
ML_PYTORCH_LEARNING_RATE = 0.001  # Learning rate for training
ML_PYTORCH_BATCH_SIZE = 32  # Batch size for training
ML_PYTORCH_EPOCHS = 100  # Maximum training epochs
ML_PYTORCH_EARLY_STOPPING_PATIENCE = 10  # Early stopping patience

# AutoML Configuration (Optuna)
ML_USE_AUTOML = False  # Enable automated hyperparameter optimization
ML_AUTOML_N_TRIALS = 100  # Number of Optuna trials for hyperparameter search
ML_AUTOML_TIMEOUT = 3600  # Maximum time for AutoML optimization (seconds)
ML_AUTOML_STUDY_NAME = 'tradpal_automl'  # Name for Optuna study
ML_AUTOML_STORAGE = None  # Database URL for Optuna storage (None = in-memory)
ML_AUTOML_SAMPLER = 'tpe'  # Sampler type: 'tpe', 'random', 'grid'
ML_AUTOML_PRUNER = 'median'  # Pruner type: 'median', 'hyperband', 'none'

# Ensemble Methods Configuration
ML_USE_ENSEMBLE = False  # Enable ensemble predictions (GA + ML)
ML_ENSEMBLE_WEIGHTS = {'ml': 0.6, 'ga': 0.4}  # Weights for ensemble combination
ML_ENSEMBLE_VOTING = 'weighted'  # Voting strategy: 'weighted', 'majority', 'unanimous'
ML_ENSEMBLE_MIN_CONFIDENCE = 0.7  # Minimum confidence for ensemble signal
```

## 🎯 Usage

### Command Line Interface

The system supports multiple operational modes:

```bash
# Live monitoring mode (default)
python main.py --mode live

# Historical backtesting mode
python main.py --mode backtest --symbol EUR/USD --timeframe 1h --start-date 2024-01-01 --end-date 2024-01-15

# Genetic Algorithm Discovery mode
python main.py --mode discovery --symbol EUR/USD --timeframe 1h --population 50 --generations 20

# Single analysis mode
python main.py --mode analysis

# Run test suite
python -m pytest tests/ -v
```

### Operational Modes

#### Live Monitoring Mode
```bash
python main.py --mode live
```
- Runs continuous market monitoring
- Monitors every 30 seconds for 1-minute charts
- Only displays output when signals are generated
- Automatically saves signals to JSON with timestamps
- Includes MTA confirmation if enabled
- **New**: Enhanced audit logging for all signal decisions and system events
- **New**: ML signal enhancement if enabled and model is trained
- Press `Ctrl+C` to stop gracefully

#### Backtesting Mode
```bash
python main.py --mode backtest --symbol EUR/USD --timeframe 1h --start-date 2024-01-01 --end-date 2024-12-31
```
- Performs historical backtesting on specified date range
- Calculates comprehensive performance metrics
- Outputs detailed results including win rate, P&L, drawdown, Sharpe ratio
- Saves backtest results to separate JSON file
- Supports all timeframes and symbols

#### Genetic Algorithm Discovery Mode
```bash
python main.py --mode discovery --symbol EUR/USD --timeframe 1h --population 50 --generations 20
```
- Uses genetic algorithms to optimize technical indicator combinations
- Evolves optimal EMA periods, RSI thresholds, Bollinger Band settings
- Tests hundreds of configurations automatically
- Outputs top 10 performing configurations with detailed metrics
- Saves results to `output/discovery_results.json`
- Useful for systematic strategy development and parameter optimization

**Discovery Parameters:**
- `--population`: Number of configurations tested per generation (default: 50)
- `--generations`: Number of evolution cycles (default: 20)
- `--symbol`: Trading pair to optimize for
- `--timeframe`: Chart timeframe for backtesting
- `--start-date`/`--end-date`: Historical data period for optimization

**Example Output:**
```
🧬 Starting Discovery Mode - Genetic Algorithm Optimization
Optimizing indicators for EUR/USD on 1h timeframe
Population: 50, Generations: 20

🏆 Discovery Results - Top 10 Configurations:
#1 - Fitness: 85.23
   P&L: 2.45%, Win Rate: 85.0%
   Sharpe: 1.67, Trades: 12
   Indicators: EMA[12, 45], RSI(21), BB(25), ATR(14)
```

#### Adaptive Optimization Mode (Self-Learning System)
```bash
# Enable adaptive optimization in config/settings.py
ADAPTIVE_OPTIMIZATION_ENABLED = True
ADAPTIVE_OPTIMIZATION_INTERVAL_HOURS = 24  # Run optimization every 24 hours
ADAPTIVE_AUTO_APPLY_BEST = True  # Automatically apply optimized configurations

# Then run live mode normally
python main.py --mode live
```
- Runs discovery optimization automatically during live trading at configured intervals
- Uses recent market data (configurable lookback period) for optimization
- Optionally applies best configurations automatically if they meet performance thresholds
- Saves optimized configurations to persist across system restarts
- Provides detailed logging of optimization runs and configuration changes

**Adaptive Configuration Parameters:**
- `ADAPTIVE_OPTIMIZATION_ENABLED`: Master switch for adaptive optimization
- `ADAPTIVE_OPTIMIZATION_INTERVAL_HOURS`: Hours between optimization runs
- `ADAPTIVE_OPTIMIZATION_POPULATION`: GA population size (smaller for live mode)
- `ADAPTIVE_OPTIMIZATION_GENERATIONS`: GA generations (fewer for faster results)
- `ADAPTIVE_OPTIMIZATION_LOOKBACK_DAYS`: Historical data period for optimization
- `ADAPTIVE_AUTO_APPLY_BEST`: Whether to automatically apply optimized configs
- `ADAPTIVE_MIN_PERFORMANCE_THRESHOLD`: Minimum fitness score for auto-application
- `ADAPTIVE_CONFIG_FILE`: File path for storing optimized configurations

**Example Live Output with Adaptive Optimization:**
```
Starting TradPal Indicator - Continuous Monitoring Mode...
🔄 Starting adaptive optimization...
🧬 Running adaptive optimization for EUR/USD...
   Period: 2024-09-08 to 2024-10-08
   Population: 30, Generations: 10
✅ Adaptive optimization completed!
   Best Fitness: 75.23
   Win Rate: 75.0%
   Total P&L: 1.45%
🔄 Applied new optimized configuration (fitness: 75.23)
🟢 BUY SIGNAL at 14:30:15...
```
#### ML Model Training Mode
```bash
python scripts/train_ml_model.py --symbol EUR/USD --timeframe 1h --start-date 2024-01-01 --end-date 2024-12-31
```
- Trains machine learning models for signal enhancement using historical data
- Uses technical indicators as features to predict future price movements
- Performs cross-validation and evaluates model performance
- Saves trained models to cache/ml_models/ for use in live trading
- Supports multiple algorithms (Random Forest, Gradient Boosting, Neural Networks)
- Outputs detailed performance metrics and feature importance

**ML Training Parameters:**
- `--symbol`: Trading pair to train on
- `--timeframe`: Chart timeframe for training data
- `--start-date`/`--end-date`: Historical data period
- `--algorithm`: ML algorithm to use (rf, gb, nn)
- `--test-size`: Fraction of data for testing (default: 0.2)
- `--cv-folds`: Number of cross-validation folds (default: 5)

#### Walk-Forward Optimization Mode
```bash
python main.py --mode walk-forward --symbol EUR/USD --timeframe 1h --start-date 2024-01-01 --end-date 2024-12-31 --window-size 30 --step-size 7
```
- **Out-of-Sample Testing**: Validates strategies on unseen data to prevent overfitting
- **Rolling Window Analysis**: Uses expanding windows of historical data for robust validation
- **Performance Stability**: Measures strategy consistency across different market conditions
- **Overfitting Prevention**: Identifies strategies that perform well only on in-sample data

**Walk-Forward Parameters:**
- `--window-size`: Initial training window size (days)
- `--step-size`: How many days to advance the window each step
- `--min-trades`: Minimum trades required for valid analysis
- `--stability-threshold`: Minimum performance stability required

**Example Output:**
```
🧪 Walk-Forward Analysis - EUR/USD (1h)
Window Size: 30 days, Step Size: 7 days

Window 1 (2024-01-01 to 2024-01-30):
  In-Sample: Win Rate 68.5%, Sharpe 1.45
  Out-of-Sample: Win Rate 65.2%, Sharpe 1.32
  Stability Score: 0.89

Window 2 (2024-01-08 to 2024-02-06):
  In-Sample: Win Rate 71.2%, Sharpe 1.52
  Out-of-Sample: Win Rate 69.8%, Sharpe 1.48
  Stability Score: 0.95

Overall Stability: 0.92 (Excellent)
✅ Strategy shows strong out-of-sample performance
```

### Programmatic Usage

#### Basic Analysis Pipeline
```python
from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators
from src.signal_generator import generate_signals, calculate_risk_management
from src.output import save_signals_to_json

# Fetch historical data
data = fetch_historical_data()

# Process data through pipeline
data = calculate_indicators(data)
data = generate_signals(data)
data = calculate_risk_management(data)

# Save results
save_signals_to_json(data)
```

#### Backtesting Example
```python
from src.backtester import run_backtest

# Run backtest with custom parameters
results = run_backtest(
    symbol='EUR/USD',
    timeframe='1h',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

print(f"Win Rate: {results['win_rate']}%")
print(f"Total P&L: ${results['total_pnl']:.2f}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

#### Custom Indicator Configuration
```python
from src.indicators import calculate_indicators

# Use default configuration
data = calculate_indicators(data)

# Use custom configuration
custom_config = {
    'ema': {'enabled': True, 'periods': [5, 10]},
    'rsi': {'enabled': True, 'period': 21},
    'bb': {'enabled': True, 'period': 15, 'std_dev': 1.5},
    'atr': {'enabled': True, 'period': 21},
    'adx': {'enabled': True, 'period': 14}
}
data = calculate_indicators(data, config=custom_config)
```

#### Real-time Monitoring Script
```python
import time
import json
from src.data_fetcher import fetch_data
from src.indicators import calculate_indicators
from src.signal_generator import generate_signals

def monitor_realtime():
    """Monitor market in real-time and alert on signals."""
    last_signal_time = None

    while True:
        try:
            # Fetch latest data
            data = fetch_data(limit=100)

            # Process indicators and signals
            data = calculate_indicators(data)
            data = generate_signals(data)

            # Check for new signals
            latest_row = data.iloc[-1]
            current_time = latest_row.name

            if (latest_row['Buy_Signal'] == 1 or latest_row['Sell_Signal'] == 1):
                if last_signal_time != current_time:
                    signal_type = "BUY" if latest_row['Buy_Signal'] == 1 else "SELL"
                    price = latest_row['close']

                    print(f"🚨 {signal_type} Signal at {current_time}: {price}")

                    # Send to integrations (if configured)
                    from integrations.integration_manager import IntegrationManager
                    manager = IntegrationManager()
                    signal_data = {
                        'type': signal_type,
                        'price': price,
                        'time': current_time.isoformat(),
                        'symbol': 'EUR/USD'
                    }
                    manager.send_signal_to_all(signal_data)

                    last_signal_time = current_time

            time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print("Monitoring stopped by user")
            break
        except Exception as e:
            print(f"Error in monitoring: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_realtime()
```

#### Performance Analysis Script
```python
import pandas as pd
from src.backtester import run_backtest, calculate_performance_metrics
import matplotlib.pyplot as plt

def analyze_strategy_performance():
    """Analyze strategy performance across multiple timeframes."""

    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    results = {}

    for timeframe in timeframes:
        print(f"Backtesting {timeframe} timeframe...")

        result = run_backtest(
            symbol='EUR/USD',
            timeframe=timeframe,
            start_date='2024-01-01',
            end_date='2024-12-31'
        )

        results[timeframe] = result['backtest_results']

    # Create performance comparison
    performance_df = pd.DataFrame(results).T

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Win Rate
    performance_df['win_rate'].plot(kind='bar', ax=axes[0,0], title='Win Rate by Timeframe')
    axes[0,0].set_ylabel('Win Rate (%)')

    # Total P&L
    performance_df['total_pnl'].plot(kind='bar', ax=axes[0,1], title='Total P&L by Timeframe', color='green')
    axes[0,1].set_ylabel('P&L ($)')

    # Sharpe Ratio
    performance_df['sharpe_ratio'].plot(kind='bar', ax=axes[1,0], title='Sharpe Ratio by Timeframe', color='blue')
    axes[1,0].set_ylabel('Sharpe Ratio')

    # Max Drawdown
    performance_df['max_drawdown'].plot(kind='bar', ax=axes[1,1], title='Max Drawdown by Timeframe', color='red')
    axes[1,1].set_ylabel('Max Drawdown (%)')

    plt.tight_layout()
    plt.savefig('output/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\n=== Performance Summary ===")
    print(performance_df.round(2))

    # Find best performing timeframe
    best_timeframe = performance_df['sharpe_ratio'].idxmax()
    print(f"\nBest performing timeframe (Sharpe Ratio): {best_timeframe}")
    print(f"Sharpe Ratio: {performance_df.loc[best_timeframe, 'sharpe_ratio']:.2f}")

if __name__ == "__main__":
    analyze_strategy_performance()
```

## 🔌 Integration & API

### JSON Output Format
Signals are saved to `output/signals.json` with the following enhanced structure:

```json
[
  {
    "timestamp": "2025-10-07T10:30:00.000Z",
    "open": 1.16846,
    "high": 1.1687,
    "low": 1.16846,
    "close": 1.1686,
    "volume": 8988.1630231,
    "EMA9": 1.1684444672614611,
    "EMA21": 1.1685168880440795,
    "RSI": 74.15730337079353,
    "BB_upper": 1.1686394881889806,
    "BB_middle": 1.168366,
    "BB_lower": 1.1680925118110195,
    "ATR": 0.00013500000000002795,
    "ADX": 28.45,
    "DI_plus": 25.12,
    "DI_minus": 18.67,
    "EMA_crossover": -1,
    "Buy_Signal": 0,
    "Sell_Signal": 1,
    "Position_Size_Absolute": 740740.7407405874,
    "Position_Size_Percent": 1.0,
    "Stop_Loss_Buy": 1.168465,
    "Take_Profit_Buy": 1.16887,
    "Fibonacci_TP_161": 1.16925,
    "Fibonacci_TP_262": 1.17015,
    "Leverage": 5,
    "MTA_Confirmed": true,
    "timeframe": "1m"
  }
]
```

### Backtesting Output Format
Backtest results are saved to `output/signals_backtest.json` with additional performance metrics:

```json
{
  "backtest_results": {
    "total_trades": 45,
    "winning_trades": 28,
    "losing_trades": 17,
    "win_rate": 62.22,
    "total_pnl": 1250.75,
    "gross_profit": 2850.50,
    "gross_loss": -1599.75,
    "profit_factor": 1.78,
    "max_drawdown": 8.45,
    "max_drawdown_percentage": 8.45,
    "sharpe_ratio": 1.23,
    "cagr": 24.56,
    "final_capital": 11250.75,
    "total_return_percentage": 12.51
  },
  "trades": [...]
}
```

### Integration Examples

## 🔌 Integration System

The project includes a modular integration system for sending trading signals to various platforms:

### Available Integrations
- **Telegram Bot**: Automatic signal notifications
- **Discord**: Webhook integration for Discord servers
- **E-Mail**: Signal notifications via email
- **SMS**: SMS notifications for critical signals
- **Webhooks**: Generic HTTP webhook support
- **Web UI**: Web-based dashboard for monitoring and configuration

### Setup Integration

```bash
# Setup Telegram integration
python scripts/manage_integrations.py --setup

# Test integrations
python scripts/test_integrations.py

# Start integrated system (Indicator + Integrations)
python scripts/run_integrated.py

# Run performance demonstration
python scripts/demo_performance.py
```

### Management Commands

```bash
# List all integrations
python scripts/manage_integrations.py --list

# Check status
python scripts/manage_integrations.py --status

# Test individual integration
python scripts/manage_integrations.py --test telegram
```

### Test Environment Safety

The integration system automatically detects test environments and prevents sending real messages:

- **Automatic Detection**: Test environments are detected via `TEST_ENVIRONMENT=true` or pytest execution
- **Safe Integrations**: All integrations send no real messages during testing
- **Logging**: Test environment skips are logged for transparency
- **Mocking**: Comprehensive mock strategies for all external dependencies

```bash
# Run tests - no real messages are sent
pytest tests/integrations/ -v

# Normal execution - real integrations work normally
python main.py --mode live
```

For more details, see `integrations/README.md`.

## 🔧 Services Architecture

The project includes modular services for advanced functionality:

### Core Services (`services/core/`)
- **Trading Engine**: Core trading logic and signal processing
- **Data Pipeline**: Optimized data fetching and processing pipeline
- **Configuration Management**: Dynamic configuration loading and validation

### ML Trainer Services (`services/ml-trainer/`)
- **Model Training**: Automated ML model training pipelines
- **Feature Engineering**: Advanced feature creation and selection
- **Model Validation**: Cross-validation and performance evaluation
- **Model Persistence**: Save and load trained models

### Optimizer Services (`services/optimizer/`)
- **Walk-Forward Optimization**: Out-of-sample strategy validation
- **Parameter Optimization**: Automated parameter tuning
- **Strategy Validation**: Robustness testing across market conditions

### Web UI Services (`services/web-ui/`)
Comprehensive interactive web interface built with Streamlit, Plotly, and Flask-Login.

#### 🎯 Features
- **🔐 Authentication System**: Secure login with user management and role-based access
- **🎨 Strategy Builder**: Drag-and-drop interface for creating custom trading strategies
  - 6 technical indicators (EMA, RSI, Bollinger Bands, ATR, ADX, MACD)
  - Real-time parameter adjustment with interactive sliders
  - Preset strategies (Trend Following, Mean Reversion, Scalping)
  - Save/load custom strategies
  - Integrated backtesting
- **⚙️ Interactive Controls**: Real-time parameter tuning
  - Timeframe-specific parameter sets (1m, 5m, 1h, 1d, etc.)
  - Visual parameter validation and feedback
  - Quick preset configurations (Scalping, Trend, Conservative)
  - Export/import configurations as JSON
- **📈 Live Charts with Plotly**: Interactive visualizations
  - Candlestick, Line, and OHLC chart types
  - Real-time indicator overlay (EMA, RSI, BB, Volume)
  - Buy/Sell signal markers
  - Multi-panel synchronized charts
  - Zoom, pan, and hover details
  - Auto-refresh capability
- **📊 Monitoring Dashboard**: Real-time performance tracking
  - Key metrics (Win Rate, Sharpe Ratio, Drawdown)
  - System health monitoring
  - Alert management
  - Performance charts

#### 🚀 Quick Start
```bash
# Install web UI dependencies
pip install streamlit plotly flask flask-login werkzeug

# Start web UI service
cd services/web-ui && streamlit run app.py

# Access at http://localhost:8501
# Default credentials: admin / admin123

# Run ML training service - sklearn models
cd services/ml-trainer && python train_service.py --mode sklearn --model-type random_forest

# Run ML training service - PyTorch models
cd services/ml-trainer && python train_service.py --mode pytorch --model-type lstm

# Run ML training with AutoML
cd services/ml-trainer && python train_service.py --mode automl --model-type gradient_boosting

# Train ensemble model (combines sklearn + PyTorch)
cd services/ml-trainer && python train_service.py --mode ensemble

# Execute walk-forward optimization with enhanced metrics
cd services/optimizer && python optimize_service.py --metric sharpe_ratio

# Example with custom parameters
cd services/ml-trainer && python train_service.py \
  --symbol BTC/USDT \
  --timeframe 1h \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --mode pytorch \
  --model-type transformer
```

#### Webhook Integration
```python
import requests
import json
import time

WEBHOOK_URL = 'https://your-trading-platform.com/webhook'

def send_webhook_alerts():
    last_signal_count = 0

    while True:
        try:
            with open('output/signals.json', 'r') as f:
                signals = json.load(f)

            if len(signals) > last_signal_count:
                new_signals = signals[last_signal_count:]
                for signal in new_signals:
                    if signal['Buy_Signal'] == 1 or signal['Sell_Signal'] == 1:
                        requests.post(WEBHOOK_URL, json=signal)

                last_signal_count = len(signals)

            time.sleep(30)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)
```

## 🐳 Docker Deployment

### Development
```bash
# Build development image
docker build -t tradpal-indicator:dev .

# Run with volume mounting for live output
docker run -v $(pwd)/output:/app/output tradpal-indicator:dev
```

### Production with Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  tradpal-indicator:
    build: .
    volumes:
      - ./output:/app/output
      - ./config:/app/config
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped

  web-ui:
    build: ./services/web-ui
    ports:
      - "8080:8080"
    volumes:
      - ./output:/app/output
    environment:
      - PYTHONPATH=/app
    depends_on:
      - tradpal-indicator
    restart: unless-stopped
```

```bash
# Deploy to production
docker-compose up -d

# View logs
docker-compose logs -f tradpal-indicator

# Access web UI
open http://localhost:8080

# Stop service
docker-compose down
```

## 🧪 Testing

Das Projekt verwendet pytest für umfassende Tests. Alle Tests sind in `tests/` organisiert und decken Unit-Tests, Integrationstests, Edge-Cases und Performance-Tests ab.

### Test-Struktur
```
tests/
├── src/                    # Unit-Tests für Kernmodule
├── config/                 # Konfigurationstests
├── integrations/           # Integrationstests
├── scripts/                # Skript-Tests
├── test_error_handling.py  # Error-Handling Tests
├── test_edge_cases.py      # Edge-Case Tests
└── test_performance.py     # Performance-Tests
```

### Tests ausführen

#### Mit pytest (empfohlen)
```bash
# Alle Tests ausführen
pytest

# Mit ausführlicher Ausgabe
pytest -v

# Mit Coverage-Report
pytest --cov=src --cov-report=html

# Nur schnelle Tests (ohne langsame Tests)
pytest -m "not slow"

# Spezifische Test-Datei
pytest tests/test_edge_cases.py

# Spezifischer Test
pytest tests/test_edge_cases.py::TestClass::test_method -v
```

#### Mit Makefile
```bash
# Alle Tests
make test

# Mit Coverage
make test-coverage

# Nur schnelle Tests
make test-fast

# Edge-Cases Tests
make test-edge-cases
```

#### Alternatives Test-Skript
```bash
# Einfacher Wrapper (Legacy)
python test.py

# Ausführliche Test-Suite (Legacy)
python run_tests_legacy.py -v
```

### Test-Konfiguration

Die pytest-Konfiguration ist in `pytest.ini` definiert und beinhaltet automatische Test-Environment-Erkennung:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --strict-config
    --disable-warnings
    --tb=short
    -ra
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    performance: marks tests as performance tests
env =
    TEST_ENVIRONMENT=true  # Automatische Test-Environment-Aktivierung
```

### Test Environment Safety

The system automatically detects test environments and prevents sending real messages during test execution:

- **Automatic Detection**: Test environments are detected via `TEST_ENVIRONMENT=true` or pytest execution
- **Safe Integrations**: All integrations (Telegram, Discord, Webhook, SMS, Email) send no real messages in tests
- **Logging**: Test environment skips are logged for transparency
- **Mocking**: Comprehensive mock strategies for all external dependencies

```bash
# Run tests - no real messages are sent
pytest tests/integrations/ -v

# Normal execution - real integrations work normally
python main.py --mode live
```

### Test-Entwicklung

#### Neuen Test hinzufügen
1. Test-Datei in entsprechendem Verzeichnis erstellen
2. pytest-Konventionen befolgen:
   - Dateien: `test_*.py`
   - Klassen: `Test*`
   - Methoden: `test_*`

```python
import pytest
from src.indicators import ema

class TestEMA:
    def test_ema_basic_calculation(self):
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(data, 3)
        assert not result.isna().all()
        assert len(result) == len(data)
```

#### Test mit Markern
```python
import pytest

@pytest.mark.slow
def test_slow_operation():
    # Langsamere Tests mit @pytest.mark.slow markieren
    pass

@pytest.mark.integration
def test_full_pipeline():
    # Integrationstests markieren
    pass
```

### CI/CD Integration

Für kontinuierliche Integration können Tests automatisch ausgeführt werden:

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## 🛠️ Troubleshooting

### Common Issues

#### Environment Variable Errors
```
ModuleNotFoundError: No module named 'dotenv'
```
**Solution**: Install python-dotenv:
```bash
pip install python-dotenv
```

#### API Key Configuration
```
ccxt.base.errors.AuthenticationError: kraken requires "apiKey" and "secret"
```
**Solution**: Ensure `.env` file exists with correct API credentials:
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

#### No Data Available
```
Error: No data loaded
```
**Solution**: Check exchange API status, internet connection, and verify SYMBOL/EXCHANGE settings in `config/settings.py`.

#### Backtest Date Parsing Errors
```
'str' object has no attribute 'date'
```
**Solution**: Use YYYY-MM-DD format for backtest dates:
```bash
python main.py --mode backtest --start-date 2024-01-01 --end-date 2024-12-31
```

#### Test Environment Issues
```
Integration sends real messages during tests
```
**Solution**: The system automatically detects test environments. If issues occur:
```bash
# Manually set test environment
export TEST_ENVIRONMENT=true
pytest tests/integrations/

# Or use pytest directly (detects automatically)
pytest tests/integrations/
```

#### Integration Test Failures
```
FAILED tests/integrations/test_integrations.py::TestIntegrationManager::test_send_signal_to_all
```
**Solution**: Ensure all integrations are properly mocked:
```bash
# Run only integration tests
pytest tests/integrations/ -v

# With detailed output
pytest tests/integrations/ -v -s
```

#### Permission Errors
```
Permission denied: output/signals.json
```
**Solution**: Ensure write permissions on output directory:
```bash
chmod 755 output/
mkdir -p logs/
chmod 755 logs/
```

#### Exchange API Limits
```
ccxt.base.errors.RateLimitExceeded: kraken GET https://api.kraken.com/0/public/OHLC
```
**Solution**: Reduce monitoring frequency, switch exchanges, or implement API key authentication for higher limits.

### Performance Optimization
- For high-frequency monitoring, consider using websockets instead of REST API
- Implement caching for indicator calculations
- Use multiprocessing for parallel signal processing
- Enable MTA only when necessary (increases API calls)

### Logging and Debugging
The system includes comprehensive logging. Check `logs/tradpal_indicator.log` for detailed information:

```python
# Adjust log level in .env file
LOG_LEVEL=DEBUG  # Options: DEBUG, INFO, WARNING, ERROR

# View recent logs
tail -f logs/tradpal_indicator.log
```

### Backtesting Tips
- Start with shorter date ranges for faster testing
- Use multiple timeframes to validate strategy robustness
- Monitor drawdown metrics closely
- Consider transaction costs in real trading scenarios

### Advanced Configuration
For production deployment, consider:
- Setting up log rotation policies
- Implementing health checks and monitoring
- Using environment-specific configuration files
- Setting up automated backups of output data

## ⚠️ Risk Disclaimer

**This is an educational project for learning purposes only.**

- **Not Financial Advice**: This software is for educational and research purposes. Do not use for actual trading without thorough backtesting and professional consultation.
- **No Guarantees**: Past performance does not predict future results. Trading involves substantial risk of loss.
- **Backtesting Required**: Always backtest strategies on historical data before live trading.
- **Risk Management**: Never risk more than you can afford to lose. The default 1% risk per trade is conservative.
- **Professional Consultation**: Consult with a qualified financial advisor before making trading decisions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the code documentation in each module

---

**Last Updated**: October 9, 2025
