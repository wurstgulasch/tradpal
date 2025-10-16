# Configuration for TradPal

import os
import json
import secrets
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
EXCHANGE = 'binance'  # Changed from kraken to binance for funding rate support

# Redis Configuration
REDIS_ENABLED = os.getenv('REDIS_ENABLED', 'false').lower() == 'true'
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_DB = int(os.getenv('REDIS_DB', '0'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
REDIS_TTL_INDICATORS = int(os.getenv('REDIS_TTL_INDICATORS', '3600'))  # 1 hour for indicators
REDIS_TTL_API = int(os.getenv('REDIS_TTL_API', '300'))  # 5 minutes for API responses
REDIS_MAX_CONNECTIONS = int(os.getenv('REDIS_MAX_CONNECTIONS', '20'))

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
RISK_PER_TRADE = 0.01  # 1% risk per trade (conservative risk management)
INITIAL_CAPITAL = CAPITAL  # Initial capital for paper trading and backtesting

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
CONFIG_MODE = 'optimized'  # 'conservative', 'discovery', or 'optimized'

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

# Optimized Configuration (from Discovery Mode - best performing parameters)
OPTIMIZED_CONFIG = {
    'ema': {'enabled': True, 'periods': [7, 107]},
    'rsi': {'enabled': True, 'period': 10, 'oversold': 34, 'overbought': 74},
    'bb': {'enabled': True, 'period': 15, 'std_dev': 2.87},
    'atr': {'enabled': True, 'period': 8},
    'adx': {'enabled': False, 'period': 21},
    'fibonacci': {'enabled': False},
    'macd': {'enabled': False, 'fast_period': 16, 'slow_period': 23, 'signal_period': 5},
    'obv': {'enabled': False, 'ma_period': 36},
    'stochastic': {'enabled': False, 'k_period': 16, 'd_period': 5},
    'cmf': {'enabled': False}
}

# Get current indicator configuration based on mode
def get_current_indicator_config():
    """Get the current indicator configuration based on CONFIG_MODE"""
    if CONFIG_MODE == 'conservative':
        return CONSERVATIVE_CONFIG
    elif CONFIG_MODE == 'discovery':
        return DISCOVERY_CONFIG
    elif CONFIG_MODE == 'optimized':
        return OPTIMIZED_CONFIG
    else:
        # Default fallback
        return DEFAULT_INDICATOR_CONFIG

# Discovery Mode Parameters
DISCOVERY_POPULATION_SIZE = 100  # Population size for GA optimization
DISCOVERY_GENERATIONS = 20  # Number of generations
DISCOVERY_MUTATION_RATE = 0.2  # Mutation probability
DISCOVERY_CROSSOVER_RATE = 0.8  # Crossover probability
DISCOVERY_LOOKBACK_DAYS = 30  # Historical data period for optimization

# LOOKBACK_DAYS for backward compatibility
LOOKBACK_DAYS = DISCOVERY_LOOKBACK_DAYS

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
ML_MODELS_DIR = ML_MODEL_DIR  # Alias for backward compatibility
ML_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for ML signal override
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

# Advanced ML Features Configuration
ML_ADVANCED_FEATURES_ENABLED = os.getenv('ML_ADVANCED_FEATURES_ENABLED', 'true').lower() == 'true'  # Enable advanced ML features
ML_ENSEMBLE_MODELS = os.getenv('ML_ENSEMBLE_MODELS', 'torch_ensemble,random_forest,gradient_boosting,lstm,transformer').split(',')  # Models to include in ensemble
ML_MARKET_REGIME_DETECTION = os.getenv('ML_MARKET_REGIME_DETECTION', 'true').lower() == 'true'  # Enable market regime detection
ML_REINFORCEMENT_LEARNING = os.getenv('ML_REINFORCEMENT_LEARNING', 'false').lower() == 'true'  # Enable reinforcement learning (future feature)
ML_GPU_OPTIMIZATION = os.getenv('ML_GPU_OPTIMIZATION', 'true').lower() == 'true'  # Enable GPU optimization when available

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

# Zero Trust Security Configuration
ENABLE_MTLS = os.getenv('ENABLE_MTLS', 'true').lower() == 'true'  # Enable mutual TLS authentication
MTLS_CERT_PATH = os.getenv('MTLS_CERT_PATH', 'cache/security/certs/client.crt')  # Client certificate path
MTLS_KEY_PATH = os.getenv('MTLS_KEY_PATH', 'cache/security/certs/client.key')  # Client private key path
CA_CERT_PATH = os.getenv('CA_CERT_PATH', 'cache/security/ca/ca_cert.pem')  # CA certificate path
SECURITY_SERVICE_URL = os.getenv('SECURITY_SERVICE_URL', 'http://localhost:8007')  # Security service URL
SECURITY_SERVICE_PORT = int(os.getenv('SECURITY_SERVICE_PORT', '8007'))  # Security service port
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))  # JWT secret key
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')  # JWT algorithm
JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))  # JWT token expiration

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

# Fitness Function Configuration (Shared between Discovery and Backtester)
# These weights determine how different metrics contribute to the overall fitness score
FITNESS_WEIGHTS = {
    'sharpe_ratio': float(os.getenv('FITNESS_SHARPE_WEIGHT', '30.0')),      # Risk-adjusted returns (30%)
    'calmar_ratio': float(os.getenv('FITNESS_CALMAR_WEIGHT', '25.0')),     # Return-to-drawdown ratio (25%)
    'total_pnl': float(os.getenv('FITNESS_PNL_WEIGHT', '30.0')),           # Total profit/loss - OUTPERFORMANCE (30%)
    'profit_factor': float(os.getenv('FITNESS_PROFIT_FACTOR_WEIGHT', '10.0')), # Gross profit / gross loss (10%)
    'win_rate': float(os.getenv('FITNESS_WIN_RATE_WEIGHT', '5.0'))         # Win rate percentage (5%)
}

# Fitness calculation bounds and normalization
FITNESS_BOUNDS = {
    'sharpe_ratio': {'min': -5, 'max': 8},      # Sharpe ratio bounds for normalization
    'calmar_ratio': {'min': -10, 'max': 15},    # Calmar ratio bounds
    'total_pnl': {'min': -100, 'max': 200},     # Total P&L percentage bounds
    'profit_factor': {'min': 0, 'max': 10},     # Profit factor bounds
    'win_rate': {'min': 0, 'max': 100}          # Win rate percentage bounds
}

# Risk penalty multipliers
FITNESS_RISK_PENALTIES = {
    'max_drawdown_15': 0.9,    # 10% penalty for drawdown > 15%
    'max_drawdown_20': 0.7,    # 30% penalty for drawdown > 20%
    'max_drawdown_30': 0.5,    # 50% penalty for drawdown > 30%
    'insufficient_trades': 0.7,  # 30% penalty for < 10 trades
    'overtrading': 0.8,        # 20% penalty for > 500 trades
    'negative_pnl_high_risk': 0.3,  # 70% penalty for negative P&L + high drawdown
    'positive_pnl_bonus': 1.05   # 5% bonus for positive P&L
}

# Default Indicator Configuration (fallback for adaptive optimization)
DEFAULT_INDICATOR_CONFIG = {
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

# Discovery Mode Configuration (Genetic Algorithm Parameters)
DISCOVERY_PARAMS = {
    'population_size': 120,      # Optimal: 120 (balance between diversity and speed)
    'generations': 30,           # Optimal: 30 (enough evolution without overfitting)
    'mutation_rate': 0.18,       # Optimal: 0.18 (good exploration vs stability)
    'crossover_rate': 0.87,      # Optimal: 0.87 (standard in GA literature)
    'tournament_size': 3,        # Selection pressure
    'elitism_count': 5,          # Preserve best individuals
    'max_evaluations': 2000,     # Safety limit for evaluations
    'early_stopping_patience': 10,  # Stop if no improvement for N generations
    'diversity_threshold': 0.1,   # Minimum population diversity
    'fitness_convergence_threshold': 0.001  # Stop if fitness change < 0.1%
}

# Output configuration
OUTPUT_FILE = 'output/signals.json'  # Default output file for signals
OUTPUT_FORMAT = 'json'  # Output format: 'json', 'csv', 'both'

# Logging configuration
LOG_FILE = 'logs/tradpal.log'  # Main log file
LOG_LEVEL = 'INFO'  # Logging level: DEBUG, INFO, WARNING, ERROR
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB max log file size
LOG_BACKUP_COUNT = 5  # Number of backup log files to keep

# Data Source Configuration
DATA_SOURCE = os.getenv('DATA_SOURCE', 'yahoo_finance')  # Default data source: 'yahoo_finance', 'ccxt', 'funding_rate'

# WebSocket Configuration
WEBSOCKET_DATA_ENABLED = os.getenv('WEBSOCKET_DATA_ENABLED', 'false').lower() == 'true'

# Data source configurations
DATA_SOURCE_CONFIG = {
    'yahoo_finance': {
        'adjust_prices': True,
        'auto_adjust': True,
        'prepost': False,
        'description': 'Yahoo Finance - Best for historical data and traditional assets'
    },
    'ccxt': {
        'exchange': os.getenv('CCXT_EXCHANGE', 'binance'),
        'api_key': os.getenv('CCXT_API_KEY', ''),
        'api_secret': os.getenv('CCXT_API_SECRET', ''),
        'description': 'CCXT - Best for crypto exchanges with real-time data'
    },
    'funding_rate': {
        'exchange': os.getenv('FUNDING_RATE_EXCHANGE', 'binance'),
        'api_key': os.getenv('FUNDING_RATE_API_KEY', ''),
        'api_secret': os.getenv('FUNDING_RATE_API_SECRET', ''),
        'description': 'Funding Rate - Specialized for perpetual futures funding rate analysis'
    }
}

# Broker Configuration for Live Trading
BROKER_ENABLED = os.getenv('BROKER_ENABLED', 'false').lower() == 'true'  # Enable/disable live trading
BROKER_EXCHANGE = os.getenv('BROKER_EXCHANGE', 'binance')  # Exchange for live trading
BROKER_API_KEY = os.getenv('BROKER_API_KEY', '')  # API key for exchange
BROKER_API_SECRET = os.getenv('BROKER_API_SECRET', '')  # API secret for exchange
BROKER_TESTNET = os.getenv('BROKER_TESTNET', 'true').lower() == 'true'  # Use testnet/sandbox
BROKER_MAX_POSITION_SIZE_PERCENT = float(os.getenv('BROKER_MAX_POSITION_SIZE_PERCENT', '1.0'))  # Max position size as % of capital
BROKER_MIN_ORDER_SIZE = float(os.getenv('BROKER_MIN_ORDER_SIZE', '10.0'))  # Minimum order size in USD
BROKER_RETRY_ATTEMPTS = int(os.getenv('BROKER_RETRY_ATTEMPTS', '3'))  # Retry attempts for failed orders
BROKER_RETRY_DELAY = float(os.getenv('BROKER_RETRY_DELAY', '1.0'))  # Delay between retries
BROKER_TIMEOUT = int(os.getenv('BROKER_TIMEOUT', '30'))  # Request timeout in seconds

# Live Trading Risk Management
LIVE_TRADING_ENABLED = BROKER_ENABLED  # Alias for backward compatibility
LIVE_TRADING_MAX_DRAWDOWN = float(os.getenv('LIVE_TRADING_MAX_DRAWDOWN', '0.1'))  # Max drawdown before emergency stop (10%)
LIVE_TRADING_MAX_TRADES_PER_DAY = int(os.getenv('LIVE_TRADING_MAX_TRADES_PER_DAY', '5'))  # Conservative limit
LIVE_TRADING_MIN_SIGNAL_CONFIDENCE = float(os.getenv('LIVE_TRADING_MIN_SIGNAL_CONFIDENCE', '0.7'))  # Minimum signal confidence
LIVE_TRADING_AUTO_EXECUTE = os.getenv('LIVE_TRADING_AUTO_EXECUTE', 'false').lower() == 'true'  # Auto-execute orders
LIVE_TRADING_CONFIRMATION_REQUIRED = os.getenv('LIVE_TRADING_CONFIRMATION_REQUIRED', 'true').lower() == 'true'  # Require manual confirmation

# Live Trading Monitoring
LIVE_TRADING_MONITOR_ENABLED = os.getenv('LIVE_TRADING_MONITOR_ENABLED', 'true').lower() == 'true'
LIVE_TRADING_POSITION_UPDATE_INTERVAL = int(os.getenv('LIVE_TRADING_POSITION_UPDATE_INTERVAL', '60'))  # Update interval in seconds
LIVE_TRADING_PNL_LOG_FILE = os.getenv('LIVE_TRADING_PNL_LOG_FILE', 'output/live_pnl.json')  # P&L log file
LIVE_TRADING_TRADE_LOG_FILE = os.getenv('LIVE_TRADING_TRADE_LOG_FILE', 'output/live_trades.json')  # Trade log file

# Additional API configuration constants for testing
API_KEY = os.getenv('API_KEY', '')  # Generic API key
API_SECRET = os.getenv('API_SECRET', '')  # Generic API secret
WEBSOCKET_RECONNECT_ATTEMPTS = int(os.getenv('WEBSOCKET_RECONNECT_ATTEMPTS', '5'))
PERFORMANCE_ENABLED = os.getenv('PERFORMANCE_ENABLED', 'true').lower() == 'true'
MAX_BACKTEST_WORKERS = int(os.getenv('MAX_BACKTEST_WORKERS', '4'))
WEBSOCKET_RECONNECT_DELAY = float(os.getenv('WEBSOCKET_RECONNECT_DELAY', '1.0'))
PARALLEL_PROCESSING_ENABLED = os.getenv('PARALLEL_PROCESSING_ENABLED', 'true').lower() == 'true'
BACKTEST_BATCH_SIZE = int(os.getenv('BACKTEST_BATCH_SIZE', '1000'))
WEBSOCKET_PING_TIMEOUT = int(os.getenv('WEBSOCKET_PING_TIMEOUT', '30'))
VECTORIZATION_ENABLED = os.getenv('VECTORIZATION_ENABLED', 'true').lower() == 'true'
MEMORY_OPTIMIZATION_ENABLED = os.getenv('MEMORY_OPTIMIZATION_ENABLED', 'true').lower() == 'true'
PERFORMANCE_MONITORING_ENABLED = os.getenv('PERFORMANCE_MONITORING_ENABLED', 'true').lower() == 'true'
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
PERFORMANCE_LOG_LEVEL = os.getenv('PERFORMANCE_LOG_LEVEL', 'INFO')
FUNDING_RATE_ENABLED = os.getenv('FUNDING_RATE_ENABLED', 'false').lower() == 'true'
FUNDING_RATE_THRESHOLD = float(os.getenv('FUNDING_RATE_THRESHOLD', '0.01'))
FUNDING_RATE_WEIGHT = float(os.getenv('FUNDING_RATE_WEIGHT', '0.2'))

# Additional constants for services
DEFAULT_SYMBOL = SYMBOL  # Default trading symbol
DEFAULT_TIMEFRAME = TIMEFRAME  # Default timeframe
MAX_POSITIONS = 5  # Maximum open positions per symbol
MAX_DRAWDOWN = 0.1  # Maximum drawdown before stopping (10%)
ORDER_TIMEOUT = 30  # Order timeout in seconds
POSITION_UPDATE_INTERVAL = 60  # Position update interval in seconds
OUTPUT_DIR = 'output'  # Output directory for results

# Validation functions
def validate_timeframe(timeframe: str) -> bool:
    """
    Validate if a timeframe string is supported.

    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')

    Returns:
        bool: True if timeframe is valid, False otherwise
    """
    valid_timeframes = ['1s', '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    return timeframe in valid_timeframes

def validate_risk_params(risk_config) -> bool:
    """
    Validate risk management parameters.

    Args:
        risk_config: Dictionary containing risk parameters

    Returns:
        bool: True if parameters are valid, False otherwise
    """
    if not isinstance(risk_config, dict):
        return False

    # Extract parameters with defaults
    capital = risk_config.get('capital', 0)
    risk_per_trade = risk_config.get('risk_per_trade', 0)
    sl_multiplier = risk_config.get('sl_multiplier', 1)

    # Validate parameters
    return (capital > 0 and                    # Capital must be positive
            0 < risk_per_trade <= 0.1 and     # Risk per trade: 0 < x <= 10%
            sl_multiplier > 0)                # Stop loss multiplier must be positive


# Additional configuration constants for testing and modules
PARALLEL_BACKTESTING_ENABLED = os.getenv('PARALLEL_BACKTESTING_ENABLED', 'true').lower() == 'true'
WEBSOCKET_RECONNECT_ATTEMPTS = int(os.getenv('WEBSOCKET_RECONNECT_ATTEMPTS', '5'))
PERFORMANCE_ENABLED = os.getenv('PERFORMANCE_ENABLED', 'true').lower() == 'true'
MAX_BACKTEST_WORKERS = int(os.getenv('MAX_BACKTEST_WORKERS', '4'))
WEBSOCKET_RECONNECT_DELAY = float(os.getenv('WEBSOCKET_RECONNECT_DELAY', '1.0'))
PARALLEL_PROCESSING_ENABLED = os.getenv('PARALLEL_PROCESSING_ENABLED', 'true').lower() == 'true'
BACKTEST_BATCH_SIZE = int(os.getenv('BACKTEST_BATCH_SIZE', '1000'))
WEBSOCKET_PING_TIMEOUT = int(os.getenv('WEBSOCKET_PING_TIMEOUT', '30'))
VECTORIZATION_ENABLED = os.getenv('VECTORIZATION_ENABLED', 'true').lower() == 'true'
MEMORY_OPTIMIZATION_ENABLED = os.getenv('MEMORY_OPTIMIZATION_ENABLED', 'true').lower() == 'true'
PERFORMANCE_MONITORING_ENABLED = os.getenv('PERFORMANCE_MONITORING_ENABLED', 'true').lower() == 'true'
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
PERFORMANCE_LOG_LEVEL = os.getenv('PERFORMANCE_LOG_LEVEL', 'INFO')
FUNDING_RATE_ENABLED = os.getenv('FUNDING_RATE_ENABLED', 'false').lower() == 'true'
FUNDING_RATE_THRESHOLD = float(os.getenv('FUNDING_RATE_THRESHOLD', '0.01'))
FUNDING_RATE_WEIGHT = float(os.getenv('FUNDING_RATE_WEIGHT', '0.2'))

# Additional API configuration constants for testing
API_KEY = os.getenv('API_KEY', '')  # Generic API key
API_SECRET = os.getenv('API_SECRET', '')  # Generic API secret

# ML Training Configuration
ML_FEATURES = [
    'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
    'bb_upper', 'bb_lower', 'bb_width', 'atr',
    'volume_ratio', 'returns', 'momentum'
]  # Features used for ML training
ML_TARGET_HORIZON = ML_TRAINING_HORIZON  # Prediction horizon for target labels
ML_TRAINING_WINDOW = 1000  # Minimum training data window size
ML_VALIDATION_SPLIT = ML_TEST_SIZE  # Validation split ratio
ML_RANDOM_STATE = 42  # Random state for reproducibility

# Microservices Configuration
DATA_SERVICE_URL = os.getenv('DATA_SERVICE_URL', 'http://localhost:8001')
CORE_SERVICE_URL = os.getenv('CORE_SERVICE_URL', 'http://localhost:8002')
ML_TRAINER_URL = os.getenv('ML_TRAINER_URL', 'http://localhost:8003')
BACKTESTING_SERVICE_URL = os.getenv('BACKTESTING_SERVICE_URL', 'http://localhost:8004')
TRADING_BOT_LIVE_URL = os.getenv('TRADING_BOT_LIVE_URL', 'http://localhost:8005')
RISK_SERVICE_URL = os.getenv('RISK_SERVICE_URL', 'http://localhost:8006')
NOTIFICATION_SERVICE_URL = os.getenv('NOTIFICATION_SERVICE_URL', 'http://localhost:8007')
WEB_UI_URL = os.getenv('WEB_UI_URL', 'http://localhost:8501')
DISCOVERY_SERVICE_URL = os.getenv('DISCOVERY_SERVICE_URL', 'http://localhost:8008')
MLOPS_SERVICE_URL = os.getenv('MLOPS_SERVICE_URL', 'http://localhost:8009')
OPTIMIZER_URL = os.getenv('OPTIMIZER_URL', 'http://localhost:8010')

# Service Feature Flags
ENABLE_DATA_SERVICE = os.getenv('ENABLE_DATA_SERVICE', 'true').lower() == 'true'
ENABLE_CORE_SERVICE = os.getenv('ENABLE_CORE_SERVICE', 'true').lower() == 'true'
ENABLE_ML_TRAINER = os.getenv('ENABLE_ML_TRAINER', 'true').lower() == 'true'
ENABLE_BACKTESTING = os.getenv('ENABLE_BACKTESTING', 'true').lower() == 'true'
ENABLE_LIVE_TRADING = os.getenv('ENABLE_LIVE_TRADING', 'false').lower() == 'true'
ENABLE_RISK_SERVICE = os.getenv('ENABLE_RISK_SERVICE', 'true').lower() == 'true'
ENABLE_NOTIFICATIONS = os.getenv('ENABLE_NOTIFICATIONS', 'true').lower() == 'true'
ENABLE_WEB_UI = os.getenv('ENABLE_WEB_UI', 'true').lower() == 'true'
ENABLE_DISCOVERY = os.getenv('ENABLE_DISCOVERY', 'true').lower() == 'true'
ENABLE_MLOPS = os.getenv('ENABLE_MLOPS', 'true').lower() == 'true'
ENABLE_OPTIMIZER = os.getenv('ENABLE_OPTIMIZER', 'true').lower() == 'true'

# Service Feature Flags (Legacy Aliases)
ENABLE_ML = ENABLE_MLOPS
ENABLE_BACKTEST = ENABLE_BACKTESTING

# Service URLs (specific service URLs)
ML_TRAINER_SERVICE_URL = MLOPS_SERVICE_URL  # ML Trainer uses MLOps service URL
TRADING_BOT_LIVE_SERVICE_URL = TRADING_BOT_LIVE_URL  # Trading Bot Live service URL
WEB_UI_SERVICE_URL = WEB_UI_URL  # Web UI service URL
REQUEST_TIMEOUT = 30  # HTTP request timeout in seconds

# UI Configuration
UI_REFRESH_INTERVAL = 30  # UI refresh interval in seconds

# Additional constant for backtest results limit
MAX_BACKTEST_RESULTS = 100  # Maximum backtest results to display

# Data Mesh Configuration
DATA_MESH_ENABLED = os.getenv('DATA_MESH_ENABLED', 'true').lower() == 'true'  # Enable Data Mesh architecture

# TimeSeries Database Configuration (InfluxDB)
INFLUXDB_ENABLED = os.getenv('INFLUXDB_ENABLED', 'true').lower() == 'true'
INFLUXDB_URL = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN', '')
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG', 'tradpal')
INFLUXDB_BUCKET = os.getenv('INFLUXDB_BUCKET', 'market_data')
INFLUXDB_MEASUREMENT_PREFIX = os.getenv('INFLUXDB_MEASUREMENT_PREFIX', 'ohlcv_')

# Data Lake Configuration (MinIO/S3)
DATA_LAKE_ENABLED = os.getenv('DATA_LAKE_ENABLED', 'true').lower() == 'true'
DATA_LAKE_TYPE = os.getenv('DATA_LAKE_TYPE', 'minio')  # 'minio' or 's3'
DATA_LAKE_ENDPOINT = os.getenv('DATA_LAKE_ENDPOINT', 'http://localhost:9000')
DATA_LAKE_ACCESS_KEY = os.getenv('DATA_LAKE_ACCESS_KEY', 'minioadmin')
DATA_LAKE_SECRET_KEY = os.getenv('DATA_LAKE_SECRET_KEY', 'minioadmin')
DATA_LAKE_BUCKET = os.getenv('DATA_LAKE_BUCKET', 'tradpal-data-lake')
DATA_LAKE_REGION = os.getenv('DATA_LAKE_REGION', 'us-east-1')
DATA_LAKE_ARCHIVE_PREFIX = os.getenv('DATA_LAKE_ARCHIVE_PREFIX', 'archive/market_data/')

# Feature Store Configuration (Redis)
FEATURE_STORE_ENABLED = os.getenv('FEATURE_STORE_ENABLED', 'true').lower() == 'true'
FEATURE_STORE_KEY_PREFIX = os.getenv('FEATURE_STORE_KEY_PREFIX', 'features:')
FEATURE_STORE_METADATA_PREFIX = os.getenv('FEATURE_STORE_METADATA_PREFIX', 'feature_metadata:')
FEATURE_STORE_VERSION_PREFIX = os.getenv('FEATURE_STORE_VERSION_PREFIX', 'feature_versions:')

# Data Product Registry Configuration
DATA_PRODUCT_REGISTRY_ENABLED = os.getenv('DATA_PRODUCT_REGISTRY_ENABLED', 'true').lower() == 'true'
DATA_PRODUCT_KEY_PREFIX = os.getenv('DATA_PRODUCT_KEY_PREFIX', 'data_products:')
DATA_PRODUCT_SCHEMA_PREFIX = os.getenv('DATA_PRODUCT_SCHEMA_PREFIX', 'data_schemas:')

# Data Mesh Governance Configuration
DATA_MESH_GOVERNANCE_ENABLED = os.getenv('DATA_MESH_GOVERNANCE_ENABLED', 'true').lower() == 'true'
DATA_MESH_AUDIT_LOG_ENABLED = os.getenv('DATA_MESH_AUDIT_LOG_ENABLED', 'true').lower() == 'true'
DATA_MESH_AUDIT_KEY_PREFIX = os.getenv('DATA_MESH_AUDIT_KEY_PREFIX', 'audit:')
DATA_MESH_ACCESS_CONTROL_ENABLED = os.getenv('DATA_MESH_ACCESS_CONTROL_ENABLED', 'true').lower() == 'true'

# Data Quality Configuration for Data Mesh
DATA_MESH_QUALITY_ENABLED = os.getenv('DATA_MESH_QUALITY_ENABLED', 'true').lower() == 'true'
DATA_MESH_QUALITY_CHECK_INTERVAL = int(os.getenv('DATA_MESH_QUALITY_CHECK_INTERVAL', '3600'))  # 1 hour
DATA_MESH_QUALITY_ALERT_THRESHOLD = float(os.getenv('DATA_MESH_QUALITY_ALERT_THRESHOLD', '0.8'))

# Data Mesh Performance Configuration
DATA_MESH_BATCH_SIZE = int(os.getenv('DATA_MESH_BATCH_SIZE', '1000'))
DATA_MESH_PARALLEL_WORKERS = int(os.getenv('DATA_MESH_PARALLEL_WORKERS', '4'))
DATA_MESH_CACHE_TTL = int(os.getenv('DATA_MESH_CACHE_TTL', '3600'))  # 1 hour

# Data Mesh Domains
DATA_MESH_DOMAINS = {
    'market_data': {
        'description': 'Real-time and historical market data (OHLCV, volume, etc.)',
        'retention_days': 365 * 2,  # 2 years
        'data_quality_required': True,
        'owners': ['data_team', 'trading_team']
    },
    'trading_signals': {
        'description': 'Generated trading signals and indicators',
        'retention_days': 365,  # 1 year
        'data_quality_required': True,
        'owners': ['trading_team', 'ml_team']
    },
    'ml_features': {
        'description': 'Machine learning features and training data',
        'retention_days': 365,  # 1 year
        'data_quality_required': True,
        'owners': ['ml_team', 'data_team']
    },
    'performance_metrics': {
        'description': 'Trading performance and backtest results',
        'retention_days': 365 * 3,  # 3 years
        'data_quality_required': False,
        'owners': ['trading_team', 'analytics_team']
    },
    'risk_data': {
        'description': 'Risk management data and calculations',
        'retention_days': 365 * 2,  # 2 years
        'data_quality_required': True,
        'owners': ['risk_team', 'trading_team']
    }
}

# Data Mesh Data Products
DATA_MESH_PRODUCTS = {
    'btc_usdt_ohlcv': {
        'domain': 'market_data',
        'description': 'BTC/USDT OHLCV data across all timeframes',
        'schema': {
            'timestamp': 'datetime',
            'open': 'float',
            'high': 'float',
            'low': 'float',
            'close': 'float',
            'volume': 'float',
            'symbol': 'string',
            'timeframe': 'string'
        },
        'owners': ['data_team'],
        'tags': ['crypto', 'btc', 'usdt', 'ohlcv']
    },
    'trading_signals_combined': {
        'domain': 'trading_signals',
        'description': 'Combined trading signals from all strategies',
        'schema': {
            'timestamp': 'datetime',
            'symbol': 'string',
            'signal': 'string',  # 'buy', 'sell', 'hold'
            'confidence': 'float',
            'strategy': 'string',
            'indicators': 'json',
            'metadata': 'json'
        },
        'owners': ['trading_team'],
        'tags': ['signals', 'trading', 'indicators']
    },
    'ml_features_technical': {
        'domain': 'ml_features',
        'description': 'Technical analysis features for ML models',
        'schema': {
            'timestamp': 'datetime',
            'symbol': 'string',
            'sma_20': 'float',
            'sma_50': 'float',
            'rsi': 'float',
            'macd': 'float',
            'macd_signal': 'float',
            'bb_upper': 'float',
            'bb_lower': 'float',
            'atr': 'float',
            'returns': 'float',
            'momentum': 'float'
        },
        'owners': ['ml_team'],
        'tags': ['ml', 'features', 'technical']
    },
    'backtest_performance': {
        'domain': 'performance_metrics',
        'description': 'Backtesting performance results',
        'schema': {
            'backtest_id': 'string',
            'symbol': 'string',
            'timeframe': 'string',
            'start_date': 'datetime',
            'end_date': 'datetime',
            'total_trades': 'int',
            'win_rate': 'float',
            'total_pnl': 'float',
            'sharpe_ratio': 'float',
            'max_drawdown': 'float',
            'profit_factor': 'float'
        },
        'owners': ['trading_team'],
        'tags': ['backtest', 'performance', 'metrics']
    }
}

# Data Governance Configuration
DATA_GOVERNANCE_ENABLED = os.getenv('DATA_GOVERNANCE_ENABLED', 'true').lower() == 'true'  # Enable Data Governance
DATA_GOVERNANCE_AUDIT_ENABLED = os.getenv('DATA_GOVERNANCE_AUDIT_ENABLED', 'true').lower() == 'true'  # Enable audit logging
DATA_GOVERNANCE_ACCESS_CONTROL_ENABLED = os.getenv('DATA_GOVERNANCE_ACCESS_CONTROL_ENABLED', 'true').lower() == 'true'  # Enable access control
DATA_GOVERNANCE_QUALITY_MONITORING_ENABLED = os.getenv('DATA_GOVERNANCE_QUALITY_MONITORING_ENABLED', 'true').lower() == 'true'  # Enable quality monitoring

# Audit Logging Configuration
AUDIT_LOG_RETENTION_DAYS = int(os.getenv('AUDIT_LOG_RETENTION_DAYS', '90'))  # Audit log retention period
AUDIT_LOG_MAX_EVENTS = int(os.getenv('AUDIT_LOG_MAX_EVENTS', '10000'))  # Max audit events in memory
AUDIT_LOG_COMPLIANCE_MODE = os.getenv('AUDIT_LOG_COMPLIANCE_MODE', 'false').lower() == 'true'  # Strict compliance mode

# Access Control Configuration
ACCESS_CONTROL_DEFAULT_POLICY = os.getenv('ACCESS_CONTROL_DEFAULT_POLICY', 'deny')  # 'allow' or 'deny'
ACCESS_CONTROL_CACHE_TTL = int(os.getenv('ACCESS_CONTROL_CACHE_TTL', '300'))  # Access decision cache TTL
ACCESS_CONTROL_MAX_POLICIES = int(os.getenv('ACCESS_CONTROL_MAX_POLICIES', '1000'))  # Max policies per resource type

# Data Quality Monitoring Configuration
QUALITY_MONITOR_CHECK_INTERVAL = int(os.getenv('QUALITY_MONITOR_CHECK_INTERVAL', '3600'))  # Quality check interval (seconds)
QUALITY_MONITOR_ALERT_THRESHOLD = float(os.getenv('QUALITY_MONITOR_ALERT_THRESHOLD', '0.8'))  # Quality alert threshold
QUALITY_MONITOR_MAX_ISSUES = int(os.getenv('QUALITY_MONITOR_MAX_ISSUES', '100'))  # Max issues to track per resource
QUALITY_MONITOR_AUTO_CORRECTION = os.getenv('QUALITY_MONITOR_AUTO_CORRECTION', 'false').lower() == 'true'  # Auto-correct quality issues

# Governance Roles Configuration
GOVERNANCE_ROLES = {
    'data_admin': {
        'description': 'Full access to all data resources and governance functions',
        'permissions': ['*'],
        'max_access_level': 'admin'
    },
    'data_steward': {
        'description': 'Manage data products and quality within assigned domains',
        'permissions': ['data_product:*', 'domain:read', 'quality:*'],
        'max_access_level': 'write'
    },
    'data_consumer': {
        'description': 'Read access to approved data products',
        'permissions': ['data_product:read', 'domain:read'],
        'max_access_level': 'read'
    },
    'ml_engineer': {
        'description': 'Access to ML features and model data',
        'permissions': ['feature_set:*', 'ml_features:*', 'data_product:read'],
        'max_access_level': 'write'
    },
    'trading_service': {
        'description': 'Access for automated trading systems',
        'permissions': ['market_data:read', 'trading_signals:*', 'risk_data:read'],
        'max_access_level': 'write'
    },
    'analyst': {
        'description': 'Read access for data analysis and reporting',
        'permissions': ['data_product:read', 'performance_metrics:read', 'domain:read'],
        'max_access_level': 'read'
    },
    'auditor': {
        'description': 'Access to audit logs and compliance reports',
        'permissions': ['audit_logs:read', 'compliance_reports:read'],
        'max_access_level': 'read'
    }
}

# Governance Policies Configuration
GOVERNANCE_POLICIES = {
    'market_data_policy': {
        'name': 'Market Data Access Policy',
        'description': 'Controls access to real-time market data',
        'resource_type': 'domain',
        'resource_name': 'market_data',
        'rules': {
            'allowed_roles': ['data_admin', 'trading_service', 'analyst'],
            'max_access_level': 'read',
            'time_restrictions': {'weekdays_only': True},
            'rate_limits': {'requests_per_hour': 1000}
        }
    },
    'trading_signals_policy': {
        'name': 'Trading Signals Policy',
        'description': 'Controls access to trading signals and strategies',
        'resource_type': 'domain',
        'resource_name': 'trading_signals',
        'rules': {
            'allowed_roles': ['data_admin', 'trading_service'],
            'max_access_level': 'write',
            'approval_required': True,
            'audit_required': True
        }
    },
    'ml_features_policy': {
        'name': 'ML Features Policy',
        'description': 'Controls access to ML training data and features',
        'resource_type': 'domain',
        'resource_name': 'ml_features',
        'rules': {
            'allowed_roles': ['data_admin', 'ml_engineer', 'data_steward'],
            'max_access_level': 'write',
            'data_classification': 'sensitive',
            'encryption_required': True
        }
    },
    'audit_logs_policy': {
        'name': 'Audit Logs Access Policy',
        'description': 'Controls access to audit logs and compliance data',
        'resource_type': 'system',
        'resource_name': 'audit_logs',
        'rules': {
            'allowed_roles': ['data_admin', 'auditor'],
            'max_access_level': 'read',
            'retention_period_days': AUDIT_LOG_RETENTION_DAYS,
            'immutable': True
        }
    }
}

# Quality Rules Configuration
DATA_QUALITY_RULES = {
    'completeness_check': {
        'name': 'Data Completeness Check',
        'description': 'Ensures critical data fields are not missing',
        'rule_type': 'completeness',
        'parameters': {
            'critical_fields': ['timestamp', 'close', 'volume'],
            'max_missing_ratio': 0.05
        },
        'severity': 'high',
        'enabled': True
    },
    'accuracy_check': {
        'name': 'Data Accuracy Check',
        'description': 'Validates data ranges and logical consistency',
        'rule_type': 'accuracy',
        'parameters': {
            'price_range': {'min': 0.000001, 'max': 1000000},
            'volume_range': {'min': 0, 'max': 1000000000},
            'ohlc_consistency': True
        },
        'severity': 'critical',
        'enabled': True
    },
    'timeliness_check': {
        'name': 'Data Timeliness Check',
        'description': 'Ensures data freshness and timeliness',
        'rule_type': 'timeliness',
        'parameters': {
            'max_age_hours': {'1m': 1, '1h': 24, '1d': 168},  # Max age by timeframe
            'staleness_threshold_minutes': 5
        },
        'severity': 'medium',
        'enabled': True
    },
    'consistency_check': {
        'name': 'Data Consistency Check',
        'description': 'Checks for logical consistency across data points',
        'rule_type': 'consistency',
        'parameters': {
            'duplicate_check': True,
            'sequence_check': True,
            'gap_detection': {'max_gap_minutes': 60}
        },
        'severity': 'medium',
        'enabled': True
    }
}

# Compliance Configuration
COMPLIANCE_ENABLED = os.getenv('COMPLIANCE_ENABLED', 'true').lower() == 'true'  # Enable compliance features
COMPLIANCE_REPORT_INTERVAL = int(os.getenv('COMPLIANCE_REPORT_INTERVAL', '86400'))  # Daily compliance reports
COMPLIANCE_RETENTION_YEARS = int(os.getenv('COMPLIANCE_RETENTION_YEARS', '7'))  # 7-year retention for compliance
COMPLIANCE_AUTO_REPORTING = os.getenv('COMPLIANCE_AUTO_REPORTING', 'true').lower() == 'true'  # Auto-generate reports

# Governance Monitoring Configuration
GOVERNANCE_MONITORING_ENABLED = os.getenv('GOVERNANCE_MONITORING_ENABLED', 'true').lower() == 'true'
GOVERNANCE_ALERT_EMAILS = os.getenv('GOVERNANCE_ALERT_EMAILS', '').split(',') if os.getenv('GOVERNANCE_ALERT_EMAILS') else []
GOVERNANCE_SLACK_WEBHOOK = os.getenv('GOVERNANCE_SLACK_WEBHOOK', '')  # Slack webhook for alerts
GOVERNANCE_METRICS_PREFIX = os.getenv('GOVERNANCE_METRICS_PREFIX', 'tradpal_governance_')  # Prometheus metrics prefix

def get_settings():
    """
    Get all configuration settings as a dictionary.

    Returns:
        dict: Dictionary containing all configuration settings
    """
    return {
        # Core settings
        'symbol': SYMBOL,
        'exchange': EXCHANGE,
        'timeframe': TIMEFRAME,
        'capital': CAPITAL,
        'risk_per_trade': RISK_PER_TRADE,

        # Timeframe parameters
        'timeframe_params': TIMEFRAME_PARAMS,
        'current_params': current_params,

        # Risk management
        'sl_multiplier': SL_MULTIPLIER,
        'tp_multiplier': TP_MULTIPLIER,
        'max_leverage': MAX_LEVERAGE,
        'leverage_base': LEVERAGE_BASE,
        'leverage_min': LEVERAGE_MIN,
        'leverage_max': LEVERAGE_MAX,

        # RSI settings
        'rsi_oversold': RSI_OVERSOLD,
        'rsi_overbought': RSI_OVERBOUGHT,
        'rsi_period': RSI_PERIOD,

        # Indicator periods
        'ema_short': EMA_SHORT,
        'ema_long': EMA_LONG,
        'bb_period': BB_PERIOD,
        'bb_std_dev': BB_STD_DEV,
        'atr_period': ATR_PERIOD,

        # Optional indicators
        'adx_enabled': ADX_ENABLED,
        'adx_threshold': ADX_THRESHOLD,
        'fibonacci_enabled': FIBONACCI_ENABLED,
        'volatility_filter_enabled': VOLATILITY_FILTER_ENABLED,

        # Signal generation
        'strict_signals_enabled': STRICT_SIGNALS_ENABLED,
        'config_mode': CONFIG_MODE,

        # Indicator configurations
        'conservative_config': CONSERVATIVE_CONFIG,
        'discovery_config': DISCOVERY_CONFIG,
        'optimized_config': OPTIMIZED_CONFIG,

        # Discovery mode
        'discovery_population_size': DISCOVERY_POPULATION_SIZE,
        'discovery_generations': DISCOVERY_GENERATIONS,
        'discovery_mutation_rate': DISCOVERY_MUTATION_RATE,
        'discovery_crossover_rate': DISCOVERY_CROSSOVER_RATE,
        'discovery_lookback_days': DISCOVERY_LOOKBACK_DAYS,

        # Adaptive optimization
        'adaptive_optimization_enabled': ADAPTIVE_OPTIMIZATION_ENABLED,
        'adaptive_optimization_interval_hours': ADAPTIVE_OPTIMIZATION_INTERVAL_HOURS,
        'adaptive_auto_apply_best': ADAPTIVE_AUTO_APPLY_BEST,
        'adaptive_min_performance_threshold': ADAPTIVE_MIN_PERFORMANCE_THRESHOLD,
        'adaptive_config_file': ADAPTIVE_CONFIG_FILE,

        # ML settings
        'ml_enabled': ML_ENABLED,
        'ml_model_dir': ML_MODEL_DIR,
        'ml_confidence_threshold': ML_CONFIDENCE_THRESHOLD,
        'ml_training_horizon': ML_TRAINING_HORIZON,
        'ml_retraining_interval_hours': ML_RETRAINING_INTERVAL_HOURS,
        'ml_min_training_samples': ML_MIN_TRAINING_SAMPLES,
        'ml_test_size': ML_TEST_SIZE,
        'ml_cv_folds': ML_CV_FOLDS,
        'ml_feature_engineering': ML_FEATURE_ENGINEERING,

        # ML model preferences
        'ml_preferred_model': ML_PREFERRED_MODEL,
        'ml_model_selection_criteria': ML_MODEL_SELECTION_CRITERIA,

        # ML model configurations
        'ml_gradient_boosting_n_estimators': ML_GRADIENT_BOOSTING_N_ESTIMATORS,
        'ml_gradient_boosting_learning_rate': ML_GRADIENT_BOOSTING_LEARNING_RATE,
        'ml_gradient_boosting_max_depth': ML_GRADIENT_BOOSTING_MAX_DEPTH,
        'ml_rf_n_estimators': ML_RF_N_ESTIMATORS,
        'ml_rf_max_depth': ML_RF_MAX_DEPTH,

        # Advanced ML
        'ml_use_pytorch': ML_USE_PYTORCH,
        'ml_pytorch_model_type': ML_PYTORCH_MODEL_TYPE,
        'ml_use_automl': ML_USE_AUTOML,
        'ml_use_ensemble': ML_USE_ENSEMBLE,
        'ml_ensemble_weights': ML_ENSEMBLE_WEIGHTS,

        # Kelly Criterion
        'kelly_enabled': KELLY_ENABLED,
        'kelly_fraction': KELLY_FRACTION,
        'kelly_lookback_trades': KELLY_LOOKBACK_TRADES,
        'kelly_min_trades': KELLY_MIN_TRADES,

        # Sentiment Analysis
        'sentiment_enabled': SENTIMENT_ENABLED,
        'sentiment_sources': SENTIMENT_SOURCES,
        'sentiment_weight': SENTIMENT_WEIGHT,

        # Paper Trading
        'paper_trading_enabled': PAPER_TRADING_ENABLED,
        'paper_trading_initial_balance': PAPER_TRADING_INITIAL_BALANCE,
        'paper_trading_fee_rate': PAPER_TRADING_FEE_RATE,
        'paper_trading_max_position_size': PAPER_TRADING_MAX_POSITION_SIZE,

        # Redis
        'redis_enabled': REDIS_ENABLED,
        'redis_host': REDIS_HOST,
        'redis_port': REDIS_PORT,
        'redis_db': REDIS_DB,

        # Data sources
        'data_source': DATA_SOURCE,
        'data_source_config': DATA_SOURCE_CONFIG,

        # Broker/Live Trading
        'broker_enabled': BROKER_ENABLED,
        'broker_exchange': BROKER_EXCHANGE,
        'broker_testnet': BROKER_TESTNET,
        'live_trading_enabled': LIVE_TRADING_ENABLED,
        'live_trading_max_drawdown': LIVE_TRADING_MAX_DRAWDOWN,

        # Service URLs
        'data_service_url': DATA_SERVICE_URL,
        'core_service_url': CORE_SERVICE_URL,
        'ml_trainer_url': ML_TRAINER_URL,
        'backtesting_service_url': BACKTESTING_SERVICE_URL,
        'trading_bot_live_url': TRADING_BOT_LIVE_URL,
        'risk_service_url': RISK_SERVICE_URL,
        'notification_service_url': NOTIFICATION_SERVICE_URL,
        'web_ui_url': WEB_UI_URL,

        # Service flags
        'enable_data_service': ENABLE_DATA_SERVICE,
        'enable_core_service': ENABLE_CORE_SERVICE,
        'enable_ml_trainer': ENABLE_ML_TRAINER,
        'enable_backtesting': ENABLE_BACKTESTING,
        'enable_live_trading': ENABLE_LIVE_TRADING,
        'enable_risk_service': ENABLE_RISK_SERVICE,
        'enable_notifications': ENABLE_NOTIFICATIONS,
        'enable_web_ui': ENABLE_WEB_UI,

        # Output and logging
        'output_file': OUTPUT_FILE,
        'output_format': OUTPUT_FORMAT,
        'log_file': LOG_FILE,
        'log_level': LOG_LEVEL,

        # Performance settings
        'parallel_processing_enabled': PARALLEL_PROCESSING_ENABLED,
        'vectorization_enabled': VECTORIZATION_ENABLED,
        'memory_optimization_enabled': MEMORY_OPTIMIZATION_ENABLED,
        'max_workers': MAX_WORKERS,

        # Data limits
        'default_data_limit': DEFAULT_DATA_LIMIT,
        'historical_data_limit': HISTORICAL_DATA_LIMIT,
        'max_backtest_results': MAX_BACKTEST_RESULTS,

        # Fitness function weights
        'fitness_weights': FITNESS_WEIGHTS,
        'fitness_bounds': FITNESS_BOUNDS,

        # Discovery parameters
        'discovery_params': DISCOVERY_PARAMS,

        # Data mesh
        'data_mesh_enabled': DATA_MESH_ENABLED,
        'data_mesh_domains': DATA_MESH_DOMAINS,
        'data_mesh_products': DATA_MESH_PRODUCTS,

        # Data governance
        'data_governance_enabled': DATA_GOVERNANCE_ENABLED,
        'governance_roles': GOVERNANCE_ROLES,
        'governance_policies': GOVERNANCE_POLICIES,
        'data_quality_rules': DATA_QUALITY_RULES,

        # Security
        'enable_mtls': ENABLE_MTLS,
        'jwt_secret_key': JWT_SECRET_KEY,
        'jwt_algorithm': JWT_ALGORITHM,

        # Monitoring
        'prometheus_enabled': PROMETHEUS_ENABLED,
        'monitoring_stack_enabled': MONITORING_STACK_ENABLED,

        # Rate limiting
        'rate_limit_enabled': RATE_LIMIT_ENABLED,
        'adaptive_rate_limiting_enabled': ADAPTIVE_RATE_LIMITING_ENABLED,

        # WebSocket
        'websocket_data_enabled': WEBSOCKET_DATA_ENABLED,

        # Funding rate
        'funding_rate_enabled': FUNDING_RATE_ENABLED,
        'funding_rate_threshold': FUNDING_RATE_THRESHOLD,
        'funding_rate_weight': FUNDING_RATE_WEIGHT,

        # Multi-timeframe analysis
        'mta_enabled': MTA_ENABLED,
        'mta_higher_timeframe': MTA_HIGHER_TIMEFRAME,
        'mta_timeframes': MTA_TIMEFRAMES,
        'mta_data_limit': MTA_DATA_LIMIT,

        # Cache settings
        'cache_ttl_live': CACHE_TTL_LIVE,
        'cache_ttl_historical': CACHE_TTL_HISTORICAL,

        # Retry settings
        'max_retries_live': MAX_RETRIES_LIVE,
        'max_retries_historical': MAX_RETRIES_HISTORICAL,

        # UI settings
        'ui_refresh_interval': UI_REFRESH_INTERVAL,

        # Request timeout
        'request_timeout': REQUEST_TIMEOUT,

        # Output directory
        'output_dir': OUTPUT_DIR,

        # Max positions
        'max_positions': MAX_POSITIONS,

        # Max drawdown
        'max_drawdown': MAX_DRAWDOWN,

        # Order timeout
        'order_timeout': ORDER_TIMEOUT,

        # Position update interval
        'position_update_interval': POSITION_UPDATE_INTERVAL,

        # Default historical days
        'default_historical_days': DEFAULT_HISTORICAL_DAYS,

        # Kraken max per request
        'kraken_max_per_request': KRAKEN_MAX_PER_REQUEST,

        # Volatility window
        'volatility_window': VOLATILITY_WINDOW,

        # Trend lookback
        'trend_lookback': TREND_LOOKBACK,

        # JSON indent
        'json_indent': JSON_INDENT,

        # Initial capital
        'initial_capital': INITIAL_CAPITAL,

        # Lookback days
        'lookback_days': LOOKBACK_DAYS,

        # ML features
        'ml_features': ML_FEATURES,
        'ml_target_horizon': ML_TARGET_HORIZON,
        'ml_training_window': ML_TRAINING_WINDOW,
        'ml_validation_split': ML_VALIDATION_SPLIT,
        'ml_random_state': ML_RANDOM_STATE,

        # ML training config
        'ml_models_dir': ML_MODELS_DIR,

        # Adaptive optimization (live)
        'adaptive_optimization_enabled_live': ADAPTIVE_OPTIMIZATION_ENABLED_LIVE,
        'adaptive_optimization_interval_hours_live': ADAPTIVE_OPTIMIZATION_INTERVAL_HOURS_LIVE,
        'adaptive_optimization_population': ADAPTIVE_OPTIMIZATION_POPULATION,
        'adaptive_optimization_generations': ADAPTIVE_OPTIMIZATION_GENERATIONS,
        'adaptive_optimization_lookback_days': ADAPTIVE_OPTIMIZATION_LOOKBACK_DAYS,
        'adaptive_auto_apply_best_live': ADAPTIVE_AUTO_APPLY_BEST_LIVE,
        'adaptive_min_performance_threshold_live': ADAPTIVE_MIN_PERFORMANCE_THRESHOLD_LIVE,
        'adaptive_config_file_live': ADAPTIVE_CONFIG_FILE_LIVE,

        # Live trading monitoring
        'live_trading_monitor_enabled': LIVE_TRADING_MONITOR_ENABLED,
        'live_trading_position_update_interval': LIVE_TRADING_POSITION_UPDATE_INTERVAL,
        'live_trading_pnl_log_file': LIVE_TRADING_PNL_LOG_FILE,
        'live_trading_trade_log_file': LIVE_TRADING_TRADE_LOG_FILE,

        # Paper trading risk management
        'paper_trading_stop_loss_enabled': PAPER_TRADING_STOP_LOSS_ENABLED,
        'paper_trading_take_profit_enabled': PAPER_TRADING_TAKE_PROFIT_ENABLED,
        'paper_trading_max_drawdown': PAPER_TRADING_MAX_DRAWDOWN,
        'paper_trading_max_trades_per_day': PAPER_TRADING_MAX_TRADES_PER_DAY,

        # Live trading risk management
        'live_trading_max_trades_per_day': LIVE_TRADING_MAX_TRADES_PER_DAY,
        'live_trading_min_signal_confidence': LIVE_TRADING_MIN_SIGNAL_CONFIDENCE,
        'live_trading_auto_execute': LIVE_TRADING_AUTO_EXECUTE,
        'live_trading_confirmation_required': LIVE_TRADING_CONFIRMATION_REQUIRED,

        # Broker settings
        'broker_max_position_size_percent': BROKER_MAX_POSITION_SIZE_PERCENT,
        'broker_min_order_size': BROKER_MIN_ORDER_SIZE,
        'broker_retry_attempts': BROKER_RETRY_ATTEMPTS,
        'broker_retry_delay': BROKER_RETRY_DELAY,
        'broker_timeout': BROKER_TIMEOUT,

        # API keys (masked for security)
        'api_key': '***masked***' if API_KEY else '',
        'api_secret': '***masked***' if API_SECRET else '',
        'broker_api_key': '***masked***' if BROKER_API_KEY else '',
        'broker_api_secret': '***masked***' if BROKER_API_SECRET else '',

        # Twitter API (masked)
        'twitter_api_key': '***masked***' if TWITTER_API_KEY else '',
        'twitter_api_secret': '***masked***' if TWITTER_API_SECRET else '',
        'twitter_access_token': '***masked***' if TWITTER_ACCESS_TOKEN else '',
        'twitter_access_token_secret': '***masked***' if TWITTER_ACCESS_TOKEN_SECRET else '',
        'twitter_bearer_token': '***masked***' if TWITTER_BEARER_TOKEN else '',

        # News API
        'news_api_key': '***masked***' if NEWS_API_KEY else '',

        # Reddit API
        'reddit_client_id': '***masked***' if REDDIT_CLIENT_ID else '',
        'reddit_client_secret': '***masked***' if REDDIT_CLIENT_SECRET else '',

        # Vault/Secrets
        'vault_addr': VAULT_ADDR,
        'vault_token': '***masked***' if VAULT_TOKEN else '',
        'aws_region': AWS_REGION,
        'aws_secretsmanager_access_key': '***masked***' if os.getenv('AWS_ACCESS_KEY_ID') else '',
        'aws_secretsmanager_secret_key': '***masked***' if os.getenv('AWS_SECRET_ACCESS_KEY') else '',

        # MTLS
        'mtls_cert_path': MTLS_CERT_PATH,
        'mtls_key_path': MTLS_KEY_PATH,
        'ca_cert_path': CA_CERT_PATH,

        # Security service
        'security_service_url': SECURITY_SERVICE_URL,
        'security_service_port': SECURITY_SERVICE_PORT,

        # JWT
        'jwt_expiration_hours': JWT_EXPIRATION_HOURS,

        # Prometheus
        'prometheus_port': PROMETHEUS_PORT,

        # Deployment
        'deployment_env': DEPLOYMENT_ENV,

        # Rate limiting
        'rate_limit_max_retries': RATE_LIMIT_MAX_RETRIES,
        'rate_limit_base_backoff': RATE_LIMIT_BASE_BACKOFF,
        'rate_limit_max_backoff': RATE_LIMIT_MAX_BACKOFF,

        # WebSocket
        'websocket_reconnect_attempts': WEBSOCKET_RECONNECT_ATTEMPTS,
        'websocket_reconnect_delay': WEBSOCKET_RECONNECT_DELAY,
        'websocket_ping_timeout': WEBSOCKET_PING_TIMEOUT,

        # Performance
        'performance_enabled': PERFORMANCE_ENABLED,
        'performance_monitoring_enabled': PERFORMANCE_MONITORING_ENABLED,
        'performance_log_level': PERFORMANCE_LOG_LEVEL,

        # Backtesting
        'parallel_backtesting_enabled': PARALLEL_BACKTESTING_ENABLED,
        'max_backtest_workers': MAX_BACKTEST_WORKERS,
        'backtest_batch_size': BACKTEST_BATCH_SIZE,

        # Chunk processing
        'chunk_size': CHUNK_SIZE,

        # InfluxDB
        'influxdb_enabled': INFLUXDB_ENABLED,
        'influxdb_url': INFLUXDB_URL,
        'influxdb_token': '***masked***' if INFLUXDB_TOKEN else '',
        'influxdb_org': INFLUXDB_ORG,
        'influxdb_bucket': INFLUXDB_BUCKET,
        'influxdb_measurement_prefix': INFLUXDB_MEASUREMENT_PREFIX,

        # Data Lake
        'data_lake_enabled': DATA_LAKE_ENABLED,
        'data_lake_type': DATA_LAKE_TYPE,
        'data_lake_endpoint': DATA_LAKE_ENDPOINT,
        'data_lake_access_key': '***masked***' if DATA_LAKE_ACCESS_KEY else '',
        'data_lake_secret_key': '***masked***' if DATA_LAKE_SECRET_KEY else '',
        'data_lake_bucket': DATA_LAKE_BUCKET,
        'data_lake_region': DATA_LAKE_REGION,
        'data_lake_archive_prefix': DATA_LAKE_ARCHIVE_PREFIX,

        # Feature Store
        'feature_store_enabled': FEATURE_STORE_ENABLED,
        'feature_store_key_prefix': FEATURE_STORE_KEY_PREFIX,
        'feature_store_metadata_prefix': FEATURE_STORE_METADATA_PREFIX,
        'feature_store_version_prefix': FEATURE_STORE_VERSION_PREFIX,

        # Data Product Registry
        'data_product_registry_enabled': DATA_PRODUCT_REGISTRY_ENABLED,
        'data_product_key_prefix': DATA_PRODUCT_KEY_PREFIX,
        'data_product_schema_prefix': DATA_PRODUCT_SCHEMA_PREFIX,

        # Data Mesh Governance
        'data_mesh_governance_enabled': DATA_MESH_GOVERNANCE_ENABLED,
        'data_mesh_audit_log_enabled': DATA_MESH_AUDIT_LOG_ENABLED,
        'data_mesh_audit_key_prefix': DATA_MESH_AUDIT_KEY_PREFIX,
        'data_mesh_access_control_enabled': DATA_MESH_ACCESS_CONTROL_ENABLED,

        # Data Quality
        'data_mesh_quality_enabled': DATA_MESH_QUALITY_ENABLED,
        'data_mesh_quality_check_interval': DATA_MESH_QUALITY_CHECK_INTERVAL,
        'data_mesh_quality_alert_threshold': DATA_MESH_QUALITY_ALERT_THRESHOLD,

        # Data Mesh Performance
        'data_mesh_batch_size': DATA_MESH_BATCH_SIZE,
        'data_mesh_parallel_workers': DATA_MESH_PARALLEL_WORKERS,
        'data_mesh_cache_ttl': DATA_MESH_CACHE_TTL,

        # Data Governance
        'data_governance_audit_enabled': DATA_GOVERNANCE_AUDIT_ENABLED,
        'data_governance_access_control_enabled': DATA_GOVERNANCE_ACCESS_CONTROL_ENABLED,
        'data_governance_quality_monitoring_enabled': DATA_GOVERNANCE_QUALITY_MONITORING_ENABLED,

        # Audit
        'audit_log_retention_days': AUDIT_LOG_RETENTION_DAYS,
        'audit_log_max_events': AUDIT_LOG_MAX_EVENTS,
        'audit_log_compliance_mode': AUDIT_LOG_COMPLIANCE_MODE,

        # Access Control
        'access_control_default_policy': ACCESS_CONTROL_DEFAULT_POLICY,
        'access_control_cache_ttl': ACCESS_CONTROL_CACHE_TTL,
        'access_control_max_policies': ACCESS_CONTROL_MAX_POLICIES,

        # Quality Monitoring
        'quality_monitor_check_interval': QUALITY_MONITOR_CHECK_INTERVAL,
        'quality_monitor_alert_threshold': QUALITY_MONITOR_ALERT_THRESHOLD,
        'quality_monitor_max_issues': QUALITY_MONITOR_MAX_ISSUES,
        'quality_monitor_auto_correction': QUALITY_MONITOR_AUTO_CORRECTION,

        # Compliance
        'compliance_enabled': COMPLIANCE_ENABLED,
        'compliance_report_interval': COMPLIANCE_REPORT_INTERVAL,
        'compliance_retention_years': COMPLIANCE_RETENTION_YEARS,
        'compliance_auto_reporting': COMPLIANCE_AUTO_REPORTING,

        # Governance Monitoring
        'governance_monitoring_enabled': GOVERNANCE_MONITORING_ENABLED,
        'governance_alert_emails': GOVERNANCE_ALERT_EMAILS,
        'governance_slack_webhook': '***masked***' if GOVERNANCE_SLACK_WEBHOOK else '',
        'governance_metrics_prefix': GOVERNANCE_METRICS_PREFIX,

        # Log settings
        'log_max_bytes': LOG_MAX_BYTES,
        'log_backup_count': LOG_BACKUP_COUNT,

        # Secrets backend
        'secrets_backend': SECRETS_BACKEND,

        # Redis password (masked)
        'redis_password': '***masked***' if REDIS_PASSWORD else '',

        # Current indicator config
        'current_indicator_config': get_current_indicator_config(),

        # Validation functions (not serializable, so excluded)
        # 'validate_timeframe': validate_timeframe,
        # 'validate_risk_params': validate_risk_params,
    }

