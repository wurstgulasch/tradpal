# Configuration for TradPal

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Environment variables (for security)

def load_environment():
    """Load appropriate environment file based on context"""
    # Check if we're in a test environment
    if os.getenv('TEST_ENVIRONMENT') == 'true' or 'pytest' in os.sys.argv[0]:
        env_file = '.env.test'
    else:
        env_file = '.env'

    # Load the environment file if it exists
    if os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        # Fallback to default .env
        load_dotenv()

# Load environment on import
load_environment()

# Constants for hardcoded values
DEFAULT_DATA_LIMIT = 200  # Default limit for data fetching
HISTORICAL_DATA_LIMIT = 1000  # Default limit for historical data
MTA_DATA_LIMIT = 50  # Limit for multi-timeframe analysis data
VOLATILITY_WINDOW = 14  # Window for volatility calculations
TREND_LOOKBACK = 10  # Lookback period for trend analysis
MAX_RETRIES_LIVE = 3  # Max retries for live data fetching
MAX_RETRIES_HISTORICAL = 2  # Max retries for historical data fetching
CACHE_TTL_LIVE = 30  # Cache TTL for live data (seconds)
CACHE_TTL_HISTORICAL = 300  # Cache TTL for historical data (seconds)
KRAKEN_MAX_PER_REQUEST = 720  # Kraken's max candles per request for 1m timeframe
DEFAULT_HISTORICAL_DAYS = 365  # Default historical data period in days
JSON_INDENT = 4  # JSON output indentation
SYMBOL = 'BTC/USDT'  # For ccxt
EXCHANGE = 'kraken'  # Example exchange

# Timeframe
TIMEFRAME = '1m'

# Timeframe-specific parameter tables
TIMEFRAME_PARAMS = {
    '1m': {
        'ema_short': 9,
        'ema_long': 21,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'bb_period': 20,
        'bb_std_dev': 2,
        'atr_period': 14,
        'atr_sl_multiplier': 1.0,
        'atr_tp_multiplier': 2.0,
        'leverage_max': 10,
        'adx_period': 14,
        'adx_threshold': 25
    },
    '5m': {
        'ema_short': 12,
        'ema_long': 26,
        'rsi_period': 14,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'bb_period': 20,
        'bb_std_dev': 2,
        'atr_period': 14,
        'atr_sl_multiplier': 1.2,
        'atr_tp_multiplier': 2.5,
        'leverage_max': 8,
        'adx_period': 14,
        'adx_threshold': 25
    },
    '15m': {
        'ema_short': 15,
        'ema_long': 30,
        'rsi_period': 14,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'bb_period': 20,
        'bb_std_dev': 2,
        'atr_period': 14,
        'atr_sl_multiplier': 1.5,
        'atr_tp_multiplier': 3.0,
        'leverage_max': 6,
        'adx_period': 14,
        'adx_threshold': 25
    },
    '1h': {
        'ema_short': 20,
        'ema_long': 50,
        'rsi_period': 14,
        'rsi_oversold': 40,
        'rsi_overbought': 60,
        'bb_period': 20,
        'bb_std_dev': 2,
        'atr_period': 14,
        'atr_sl_multiplier': 1.5,
        'atr_tp_multiplier': 3.0,
        'leverage_max': 5,
        'adx_period': 14,
        'adx_threshold': 25
    },
    '1d': {
        'ema_short': 50,
        'ema_long': 200,
        'rsi_period': 14,
        'rsi_oversold': 40,
        'rsi_overbought': 60,
        'bb_period': 20,
        'bb_std_dev': 2,
        'atr_period': 14,
        'atr_sl_multiplier': 2.0,
        'atr_tp_multiplier': 4.0,
        'leverage_max': 3,
        'adx_period': 14,
        'adx_threshold': 25
    }
}

# Get parameters for current timeframe
def get_timeframe_params(timeframe=None):
    """Get parameters for specified timeframe, fallback to current TIMEFRAME"""
    if timeframe is None:
        timeframe = TIMEFRAME
    elif timeframe not in TIMEFRAME_PARAMS:
        return None
    return TIMEFRAME_PARAMS.get(timeframe, None)

# Extract current parameters for backward compatibility
current_params = get_timeframe_params()
EMA_SHORT = current_params['ema_short']
EMA_LONG = current_params['ema_long']
RSI_PERIOD = current_params['rsi_period']
BB_PERIOD = current_params['bb_period']
BB_STD_DEV = current_params['bb_std_dev']
ATR_PERIOD = current_params['atr_period']

# Risk management
CAPITAL = 10000  # Realistic starting capital for trading
RISK_PER_TRADE = 0.8  # 80% risk per trade (conservative risk management)

# Risk management parameters
SL_MULTIPLIER = current_params['atr_sl_multiplier']
TP_MULTIPLIER = current_params['atr_tp_multiplier']
MAX_LEVERAGE = current_params['leverage_max']
LEVERAGE_BASE = current_params['leverage_max']  # Base leverage same as max for simplicity
LEVERAGE_MIN = 1  # Minimum leverage is 1:1
LEVERAGE_MAX = current_params['leverage_max']

# RSI thresholds
RSI_OVERSOLD = current_params['rsi_oversold']
RSI_OVERBOUGHT = current_params['rsi_overbought']

# Multi-timeframe analysis
MTA_ENABLED = True
MTA_HIGHER_TIMEFRAME = '5m'  # Higher timeframe for trend confirmation
MTA_TIMEFRAMES = ['5m', '15m']  # Available higher timeframes

# Optional indicators
ADX_ENABLED = False
ADX_THRESHOLD = current_params['adx_threshold']  # ADX threshold for trend strength
FIBONACCI_ENABLED = False
VOLATILITY_FILTER_ENABLED = False

# Signal generation mode
STRICT_SIGNALS_ENABLED = True  # If False, use only EMA crossover for signals (for backtesting)

# Configuration Mode
CONFIG_MODE = 'conservative'  # 'conservative' or 'discovery'

# Conservative Configuration (fixed parameters)
CONSERVATIVE_CONFIG = {
    'ema': {'enabled': True, 'periods': [9, 21]},
    'rsi': {'enabled': True, 'period': 14, 'oversold': 30, 'overbought': 70},
    'bb': {'enabled': True, 'period': 20, 'std_dev': 2.0},
    'atr': {'enabled': True, 'period': 14},
    'adx': {'enabled': False, 'period': 14},
    'fibonacci': {'enabled': False},
    'macd': {'enabled': True, 'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
    'obv': {'enabled': True},
    'stochastic': {'enabled': True, 'k_period': 14, 'd_period': 3},
    'cmf': {'enabled': False, 'period': 21}
}

# Discovery Configuration (adaptive parameters)
DISCOVERY_CONFIG = {
    'ema': {'enabled': True, 'periods': [9, 21]},  # Initial values, will be optimized
    'rsi': {'enabled': True, 'period': 14, 'oversold': 30, 'overbought': 70},
    'bb': {'enabled': True, 'period': 20, 'std_dev': 2.0},
    'atr': {'enabled': True, 'period': 14},
    'adx': {'enabled': False, 'period': 14},
    'fibonacci': {'enabled': False},
    'macd': {'enabled': True, 'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
    'obv': {'enabled': True},
    'stochastic': {'enabled': True, 'k_period': 14, 'd_period': 3},
    'cmf': {'enabled': False, 'period': 21}
}

# Discovery Mode Parameters
DISCOVERY_POPULATION_SIZE = 100  # Population size for GA optimization
DISCOVERY_GENERATIONS = 20  # Number of generations
DISCOVERY_MUTATION_RATE = 0.2  # Mutation probability
DISCOVERY_CROSSOVER_RATE = 0.8  # Crossover probability
DISCOVERY_LOOKBACK_DAYS = 30  # Historical data period for optimization

# Adaptive optimization settings (for discovery mode)
ADAPTIVE_OPTIMIZATION_ENABLED = os.getenv('ADAPTIVE_OPTIMIZATION_ENABLED', 'true').lower() == 'true'  # Enable/disable periodic discovery optimization
ADAPTIVE_OPTIMIZATION_INTERVAL_HOURS = int(os.getenv('ADAPTIVE_OPTIMIZATION_INTERVAL_HOURS', '24'))  # How often to run discovery (in hours)
ADAPTIVE_AUTO_APPLY_BEST = os.getenv('ADAPTIVE_AUTO_APPLY_BEST', 'true').lower() == 'true'  # Automatically apply best configuration found
ADAPTIVE_MIN_PERFORMANCE_THRESHOLD = float(os.getenv('ADAPTIVE_MIN_PERFORMANCE_THRESHOLD', '0.5'))  # Minimum fitness score to consider applying
ADAPTIVE_CONFIG_FILE = os.getenv('ADAPTIVE_CONFIG_FILE', 'config/adaptive_config.json')  # File to store optimized config

# Adaptive optimization settings (for live mode)
ADAPTIVE_OPTIMIZATION_ENABLED_LIVE = os.getenv('ADAPTIVE_OPTIMIZATION_ENABLED', 'true').lower() == 'true'  # Enable/disable periodic discovery optimization
ADAPTIVE_OPTIMIZATION_INTERVAL_HOURS_LIVE = int(os.getenv('ADAPTIVE_OPTIMIZATION_INTERVAL_HOURS', '1'))  # How often to run discovery (in hours)
ADAPTIVE_OPTIMIZATION_POPULATION = int(os.getenv('ADAPTIVE_OPTIMIZATION_POPULATION', '1000'))  # Smaller population for live optimization
ADAPTIVE_OPTIMIZATION_GENERATIONS = int(os.getenv('ADAPTIVE_OPTIMIZATION_GENERATIONS', '50'))  # Fewer generations for faster results
ADAPTIVE_OPTIMIZATION_LOOKBACK_DAYS = int(os.getenv('ADAPTIVE_OPTIMIZATION_LOOKBACK_DAYS', '30'))  # Historical data period for optimization
ADAPTIVE_AUTO_APPLY_BEST_LIVE = os.getenv('ADAPTIVE_AUTO_APPLY_BEST', 'true').lower() == 'true'  # Automatically apply best configuration found
ADAPTIVE_MIN_PERFORMANCE_THRESHOLD_LIVE = float(os.getenv('ADAPTIVE_MIN_PERFORMANCE_THRESHOLD', '1'))  # Minimum fitness score to consider applying
ADAPTIVE_CONFIG_FILE_LIVE = os.getenv('ADAPTIVE_CONFIG_FILE', 'config/adaptive_config.json')  # File to store optimized config

# Machine Learning settings
ML_ENABLED = os.getenv('ML_ENABLED', 'true').lower() == 'true'  # Enable/disable ML signal enhancement
ML_MODEL_DIR = 'cache/ml_models'  # Directory to store trained ML models
ML_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for ML signal override (lowered from 0.6)
ML_TRAINING_HORIZON = 5  # Prediction horizon for training labels (periods ahead)
ML_RETRAINING_INTERVAL_HOURS = 24  # How often to retrain models (hours)
ML_MIN_TRAINING_SAMPLES = 1000  # Minimum samples required for training
ML_TEST_SIZE = 0.2  # Fraction of data for testing
ML_CV_FOLDS = 5  # Number of cross-validation folds
ML_FEATURE_ENGINEERING = True  # Enable advanced feature engineering

# Preferred ML Model Configuration
ML_PREFERRED_MODEL = 'random_forest'  # Preferred model: 'gradient_boosting', 'xgboost', 'random_forest', 'svm', 'logistic_regression' - CHANGED TO FASTER MODEL
ML_MODEL_SELECTION_CRITERIA = 'f1'  # Selection criteria: 'f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy'

# Gradient Boosting Specific Configuration
ML_GRADIENT_BOOSTING_N_ESTIMATORS = 50  # Number of boosting stages - REDUCED FOR FASTER TESTS
ML_GRADIENT_BOOSTING_LEARNING_RATE = 0.1  # Learning rate
ML_GRADIENT_BOOSTING_MAX_DEPTH = 6  # Maximum depth of individual trees
ML_GRADIENT_BOOSTING_MIN_SAMPLES_SPLIT = 20  # Minimum samples required to split
ML_GRADIENT_BOOSTING_MIN_SAMPLES_LEAF = 10  # Minimum samples required at leaf node
ML_GRADIENT_BOOSTING_SUBSAMPLE = 0.8  # Fraction of samples used for fitting
ML_GRADIENT_BOOSTING_MAX_FEATURES = 'sqrt'  # Number of features to consider for best split

# XGBoost Specific Configuration
ML_XGBOOST_N_ESTIMATORS = 50  # Number of boosting rounds - REDUCED FOR FASTER TESTS
ML_XGBOOST_LEARNING_RATE = 0.1  # Learning rate
ML_XGBOOST_MAX_DEPTH = 6  # Maximum depth of trees
ML_XGBOOST_MIN_CHILD_WEIGHT = 1  # Minimum sum of instance weight needed in a child
ML_XGBOOST_SUBSAMPLE = 0.8  # Subsample ratio of training instances
ML_XGBOOST_COLSAMPLE_BYTREE = 0.8  # Subsample ratio of columns when constructing each tree
ML_XGBOOST_GAMMA = 0  # Minimum loss reduction required to make a further partition

# Random Forest Specific Configuration
ML_RF_N_ESTIMATORS = 50  # Number of trees in the forest - REDUCED FOR FASTER TESTS
ML_RF_MAX_DEPTH = None  # Maximum depth of trees
ML_RF_MIN_SAMPLES_SPLIT = 2  # Minimum samples required to split
ML_RF_MIN_SAMPLES_LEAF = 1  # Minimum samples required at leaf node
ML_RF_MAX_FEATURES = 'sqrt'  # Number of features to consider for best split
ML_RF_BOOTSTRAP = True  # Whether bootstrap samples are used

# SVM Specific Configuration
ML_SVM_C = 1.0  # Regularization parameter
ML_SVM_KERNEL = 'rbf'  # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
ML_SVM_GAMMA = 'scale'  # Kernel coefficient
ML_SVM_CLASS_WEIGHT = 'balanced'  # Class weight balancing

# Logistic Regression Specific Configuration
ML_LR_C = 1.0  # Inverse of regularization strength
ML_LR_PENALTY = 'l2'  # Regularization type: 'l1', 'l2', 'elasticnet', 'none'
ML_LR_SOLVER = 'lbfgs'  # Solver algorithm
ML_LR_MAX_ITER = 1000  # Maximum number of iterations

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
ML_USE_AUTOML = False  # Enable automated hyperparameter optimization - DISABLED FOR FASTER TESTS
ML_AUTOML_N_TRIALS = 5  # Number of Optuna trials for hyperparameter search - REDUCED FOR TESTS
ML_AUTOML_TIMEOUT = 300  # Maximum time for AutoML optimization (seconds) - REDUCED FOR TESTS
ML_AUTOML_STUDY_NAME = 'tradpal_gradient_boosting_optimization'  # Name for Optuna study
ML_AUTOML_STORAGE = None  # Database URL for Optuna storage (None = in-memory)
ML_AUTOML_SAMPLER = 'tpe'  # Sampler type: 'tpe', 'random', 'grid'
ML_AUTOML_PRUNER = 'median'  # Pruner type: 'median', 'hyperband', 'none'

# Ensemble Methods Configuration
ML_USE_ENSEMBLE = False  # Enable ensemble predictions (GA + ML)
ML_ENSEMBLE_WEIGHTS = {'ml': 0.6, 'ga': 0.4}  # Weights for ensemble combination
ML_ENSEMBLE_VOTING = 'weighted'  # Voting strategy: 'weighted', 'majority', 'unanimous'
ML_ENSEMBLE_MIN_CONFIDENCE = 0.7  # Minimum confidence for ensemble signal

# Kelly Criterion Configuration
KELLY_ENABLED = os.getenv('KELLY_ENABLED', 'false').lower() == 'true'  # Enable Kelly Criterion position sizing
KELLY_FRACTION = float(os.getenv('KELLY_FRACTION', '0.5'))  # Fractional Kelly (0.5 = half Kelly)
KELLY_LOOKBACK_TRADES = int(os.getenv('KELLY_LOOKBACK_TRADES', '100'))  # Lookback period for win rate calculation
KELLY_MIN_TRADES = int(os.getenv('KELLY_MIN_TRADES', '20'))  # Minimum trades required for Kelly calculation

# Sentiment Analysis Configuration
SENTIMENT_ENABLED = os.getenv('SENTIMENT_ENABLED', 'false').lower() == 'true'  # Enable sentiment analysis
SENTIMENT_SOURCES = os.getenv('SENTIMENT_SOURCES', 'twitter,news,reddit').split(',')  # Data sources to use
SENTIMENT_UPDATE_INTERVAL = int(os.getenv('SENTIMENT_UPDATE_INTERVAL', '300'))  # Update interval in seconds
SENTIMENT_CACHE_TTL = int(os.getenv('SENTIMENT_CACHE_TTL', '1800'))  # Cache TTL for sentiment data (seconds)
SENTIMENT_WEIGHT = float(os.getenv('SENTIMENT_WEIGHT', '0.2'))  # Weight of sentiment in signal generation (0-1)
SENTIMENT_THRESHOLD = float(os.getenv('SENTIMENT_THRESHOLD', '0.1'))  # Minimum sentiment score for signal influence

# Twitter Sentiment Configuration
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', '')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', '')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')
TWITTER_SEARCH_TERMS = os.getenv('TWITTER_SEARCH_TERMS', 'BTC,Bitcoin,crypto').split(',')  # Search terms for sentiment
TWITTER_MAX_TWEETS = int(os.getenv('TWITTER_MAX_TWEETS', '100'))  # Max tweets to analyze per update
TWITTER_LANGUAGE = os.getenv('TWITTER_LANGUAGE', 'en')  # Language filter for tweets

# News Sentiment Configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
NEWS_SOURCES = os.getenv('NEWS_SOURCES', 'coindesk,cointelegraph,bitcoinmagazine').split(',')  # News sources
NEWS_SEARCH_TERMS = os.getenv('NEWS_SEARCH_TERMS', 'Bitcoin,BTC,cryptocurrency').split(',')  # Search terms
NEWS_MAX_ARTICLES = int(os.getenv('NEWS_MAX_ARTICLES', '50'))  # Max articles to analyze per update
NEWS_LANGUAGE = os.getenv('NEWS_LANGUAGE', 'en')  # Language filter for news

# Reddit Sentiment Configuration
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'TradPal/1.0')
REDDIT_SUBREDDITS = os.getenv('REDDIT_SUBREDDITS', 'bitcoin,crypto,cryptocurrency').split(',')  # Subreddits to monitor
REDDIT_MAX_POSTS = int(os.getenv('REDDIT_MAX_POSTS', '50'))  # Max posts to analyze per update
REDDIT_TIME_FILTER = os.getenv('REDDIT_TIME_FILTER', 'day')  # Time filter: hour, day, week, month, year, all

# Sentiment Analysis Model Configuration
SENTIMENT_MODEL_TYPE = os.getenv('SENTIMENT_MODEL_TYPE', 'vader')  # Options: 'vader', 'textblob', 'transformers'
SENTIMENT_MODEL_CACHE_DIR = 'cache/sentiment_models'  # Directory for sentiment models
SENTIMENT_PREPROCESSING_ENABLED = os.getenv('SENTIMENT_PREPROCESSING_ENABLED', 'true').lower() == 'true'  # Enable text preprocessing
SENTIMENT_REMOVE_STOPWORDS = os.getenv('SENTIMENT_REMOVE_STOPWORDS', 'true').lower() == 'true'  # Remove stopwords
SENTIMENT_LEMMATIZE = os.getenv('SENTIMENT_LEMMATIZE', 'false').lower() == 'true'  # Apply lemmatization

# Paper Trading Configuration
PAPER_TRADING_ENABLED = os.getenv('PAPER_TRADING_ENABLED', 'false').lower() == 'true'  # Enable paper trading mode
PAPER_TRADING_INITIAL_BALANCE = float(os.getenv('PAPER_TRADING_INITIAL_BALANCE', '10000'))  # Starting balance in USD
PAPER_TRADING_FEE_RATE = float(os.getenv('PAPER_TRADING_FEE_RATE', '0.001'))  # Trading fee rate (0.1%)
PAPER_TRADING_SLIPPAGE = float(os.getenv('PAPER_TRADING_SLIPPAGE', '0.0005'))  # Price slippage (0.05%)
PAPER_TRADING_MAX_POSITION_SIZE = float(os.getenv('PAPER_TRADING_MAX_POSITION_SIZE', '0.1'))  # Max position size as % of balance
PAPER_TRADING_DATA_SOURCE = os.getenv('PAPER_TRADING_DATA_SOURCE', 'live')  # 'live' or 'historical'
PAPER_TRADING_SAVE_TRADES = os.getenv('PAPER_TRADING_SAVE_TRADES', 'true').lower() == 'true'  # Save trades to file
PAPER_TRADING_TRADE_LOG_FILE = os.getenv('PAPER_TRADING_TRADE_LOG_FILE', 'output/paper_trades.json')  # Trade log file
PAPER_TRADING_PERFORMANCE_LOG_FILE = os.getenv('PAPER_TRADING_PERFORMANCE_LOG_FILE', 'output/paper_performance.json')  # Performance log file

# Paper Trading Risk Management
PAPER_TRADING_STOP_LOSS_ENABLED = os.getenv('PAPER_TRADING_STOP_LOSS_ENABLED', 'true').lower() == 'true'
PAPER_TRADING_TAKE_PROFIT_ENABLED = os.getenv('PAPER_TRADING_TAKE_PROFIT_ENABLED', 'true').lower() == 'true'
PAPER_TRADING_MAX_DRAWDOWN = float(os.getenv('PAPER_TRADING_MAX_DRAWDOWN', '0.2'))  # Max drawdown before stopping (20%)
PAPER_TRADING_MAX_TRADES_PER_DAY = int(os.getenv('PAPER_TRADING_MAX_TRADES_PER_DAY', '10'))  # Max trades per day

# Secrets Management Configuration
SECRETS_BACKEND = os.getenv('SECRETS_BACKEND', 'env')  # Options: 'env', 'vault', 'aws-secretsmanager'
VAULT_ADDR = os.getenv('VAULT_ADDR', 'http://localhost:8200')  # Vault server address
VAULT_TOKEN = os.getenv('VAULT_TOKEN', '')  # Vault authentication token
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')  # AWS region for Secrets Manager

# Monitoring Configuration
PROMETHEUS_ENABLED = os.getenv('PROMETHEUS_ENABLED', 'false').lower() == 'true'
PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', '8000'))
MONITORING_STACK_ENABLED = os.getenv('MONITORING_STACK_ENABLED', 'false').lower() == 'true'
DEPLOYMENT_ENV = os.getenv('DEPLOYMENT_ENV', 'local')  # Options: 'local', 'aws', 'kubernetes'

# Rate Limiting Configuration
RATE_LIMIT_ENABLED = True  # Enable adaptive rate limiting
ADAPTIVE_RATE_LIMITING_ENABLED = os.getenv('ADAPTIVE_RATE_LIMITING_ENABLED', 'true').lower() == 'true'
RATE_LIMIT_MAX_RETRIES = 5  # Maximum retries for rate-limited requests
RATE_LIMIT_BASE_BACKOFF = 2.0  # Base backoff multiplier for retries
RATE_LIMIT_MAX_BACKOFF = 300  # Maximum backoff time in seconds

def get_current_indicator_config() -> Dict[str, Any]:
    """
    Get the current indicator configuration based on the selected mode.

    Returns:
        Dictionary containing the current indicator configuration
    """
    if CONFIG_MODE == 'conservative':
        return CONSERVATIVE_CONFIG.copy()
    elif CONFIG_MODE == 'discovery':
        # Try to load optimized config first
        try:
            if os.path.exists(ADAPTIVE_CONFIG_FILE):
                with open(ADAPTIVE_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                if 'best_configuration' in data:
                    config = data['best_configuration']
                    print(f"Loaded optimized discovery configuration (fitness: {data.get('fitness_score', 'N/A')})")
                    return config
        except Exception as e:
            print(f"Warning: Could not load optimized config: {e}")

        # Fallback to default discovery config
        return DISCOVERY_CONFIG.copy()
    else:
        return CONSERVATIVE_CONFIG.copy()

# Set DEFAULT_INDICATOR_CONFIG based on mode
DEFAULT_INDICATOR_CONFIG = get_current_indicator_config()

# Data and output
LOOKBACK_DAYS = 365  # Increased for longer backtest periods
OUTPUT_FORMAT = 'json'
OUTPUT_FILE = 'output/signals.json'

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/tradpal.log'

API_KEY = os.getenv('TRADPAL_API_KEY', '')
API_SECRET = os.getenv('TRADPAL_API_SECRET', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

def validate_timeframe(timeframe):
    """Validate timeframe string format and logical constraints."""
    if not isinstance(timeframe, str) or not timeframe:
        return False

    # Check format: number + unit (s, m, h, d, w, M)
    import re
    pattern = r'^(\d+)([smhdwM])$'
    match = re.match(pattern, timeframe)
    if not match:
        return False

    # Extract number and check it's positive
    number = int(match.group(1))
    if number <= 0:
        return False

    # Valid units
    unit = match.group(2)
    valid_units = ['s', 'm', 'h', 'd', 'w', 'M']
    if unit not in valid_units:
        return False

    return True

def validate_risk_params(params):
    """Validate risk management parameters."""
    required_keys = ['capital', 'risk_per_trade', 'sl_multiplier']

    # Check required keys
    if not all(key in params for key in required_keys):
        return False

    # Check value ranges
    if params['capital'] <= 0:
        return False
    if not 0 < params['risk_per_trade'] <= 1:  # Risk should be 0-100%
        return False
    if params['sl_multiplier'] <= 0:
        return False

    return True

# Performance Optimization Settings
PERFORMANCE_ENABLED = os.getenv('PERFORMANCE_ENABLED', 'true').lower() == 'true'  # Enable/disable performance optimizations
PARALLEL_PROCESSING_ENABLED = os.getenv('PARALLEL_PROCESSING_ENABLED', 'true').lower() == 'true'  # Enable parallel processing for indicators
VECTORIZATION_ENABLED = True  # Enable vectorized calculations
MEMORY_OPTIMIZATION_ENABLED = True  # Enable memory optimization for DataFrames
PERFORMANCE_MONITORING_ENABLED = os.getenv('PERFORMANCE_MONITORING_ENABLED', 'true').lower() == 'true'  # Enable performance monitoring
MAX_WORKERS = None  # Maximum worker threads (None = auto-detect CPU cores)
CHUNK_SIZE = 1000  # Chunk size for parallel processing
PERFORMANCE_LOG_LEVEL = 'INFO'  # Performance logging level

# WebSocket Streaming Settings
WEBSOCKET_ENABLED = True  # Enable/disable WebSocket server
WEBSOCKET_HOST = '0.0.0.0'  # WebSocket server host
WEBSOCKET_PORT = 8765  # WebSocket server port
WEBSOCKET_MAX_CONNECTIONS = 100  # Maximum concurrent WebSocket connections
WEBSOCKET_PING_INTERVAL = 30  # Ping interval in seconds
WEBSOCKET_TIMEOUT = 60  # Connection timeout in seconds
WEBSOCKET_BUFFER_SIZE = 1024  # Message buffer size
WEBSOCKET_COMPRESSION = True  # Enable message compression

# Real-time Streaming Settings
REALTIME_ENABLED = True  # Enable real-time data streaming
REALTIME_UPDATE_INTERVAL = 10  # Update interval in seconds
REALTIME_MAX_SUBSCRIPTIONS = 50  # Maximum active subscriptions per client
REALTIME_DATA_RETENTION = 3600  # Data retention time in seconds (1 hour)
REALTIME_BROADCAST_SIGNALS = True  # Broadcast trading signals
REALTIME_BROADCAST_MARKET_DATA = True  # Broadcast market data updates

# WebSocket Data Fetching Settings (New)
WEBSOCKET_DATA_ENABLED = os.getenv('WEBSOCKET_DATA_ENABLED', 'false').lower() == 'true'  # Enable WebSocket data fetching
WEBSOCKET_RECONNECT_ATTEMPTS = 5  # Number of reconnection attempts
WEBSOCKET_RECONNECT_DELAY = 5  # Delay between reconnection attempts (seconds)
WEBSOCKET_PING_TIMEOUT = 30  # Ping timeout for WebSocket connections

# Redis Cache Settings (New)
REDIS_ENABLED = os.getenv('REDIS_ENABLED', 'false').lower() == 'true'  # Enable Redis caching
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')  # Redis server host
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))  # Redis server port
REDIS_DB = int(os.getenv('REDIS_DB', '0'))  # Redis database number
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)  # Redis password (if required)
REDIS_TTL_INDICATORS = 300  # Cache TTL for indicators (seconds)
REDIS_TTL_API = 60  # Cache TTL for API calls (seconds)
REDIS_MAX_CONNECTIONS = 10  # Maximum Redis connection pool size

# Parallel Processing Settings (New)
PARALLEL_BACKTESTING_ENABLED = os.getenv('PARALLEL_BACKTESTING_ENABLED', 'true').lower() == 'true'  # Enable parallel backtesting
MAX_BACKTEST_WORKERS = int(os.getenv('MAX_BACKTEST_WORKERS', '0'))  # Max workers for parallel backtesting (0 = auto)
BACKTEST_BATCH_SIZE = int(os.getenv('BACKTEST_BATCH_SIZE', '10'))  # Batch size for parallel processing

