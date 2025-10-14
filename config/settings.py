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

