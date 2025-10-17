# Machine Learning configuration settings
# ML model parameters, training configurations, and AI features

import os
from typing import Dict, Any

# Machine Learning settings
ML_ENABLED = os.getenv('ML_ENABLED', 'true').lower() == 'true'  # Enable/disable ML signal enhancement
ML_MODEL_DIR = os.getenv('ML_MODEL_DIR', 'cache/ml_models')  # Directory to store trained ML models
ML_MODELS_DIR = ML_MODEL_DIR  # Alias for backward compatibility
ML_CONFIDENCE_THRESHOLD = float(os.getenv('ML_CONFIDENCE_THRESHOLD', '0.5'))  # Minimum confidence for ML signal override
ML_TRAINING_HORIZON = int(os.getenv('ML_TRAINING_HORIZON', '5'))  # Prediction horizon for training labels (periods ahead)
ML_RETRAINING_INTERVAL_HOURS = int(os.getenv('ML_RETRAINING_INTERVAL_HOURS', '24'))  # How often to retrain models (hours)
ML_MIN_TRAINING_SAMPLES = int(os.getenv('ML_MIN_TRAINING_SAMPLES', '1000'))  # Minimum samples required for training
ML_TEST_SIZE = float(os.getenv('ML_TEST_SIZE', '0.2'))  # Fraction of data for testing
ML_CV_FOLDS = int(os.getenv('ML_CV_FOLDS', '5'))  # Number of cross-validation folds
ML_FEATURE_ENGINEERING = os.getenv('ML_FEATURE_ENGINEERING', 'true').lower() == 'true'  # Enable advanced feature engineering

# Preferred ML Model Configuration
ML_PREFERRED_MODEL = os.getenv('ML_PREFERRED_MODEL', 'random_forest')  # Preferred model
ML_MODEL_SELECTION_CRITERIA = os.getenv('ML_MODEL_SELECTION_CRITERIA', 'f1')  # Selection criteria

# Gradient Boosting Specific Configuration
ML_GRADIENT_BOOSTING_N_ESTIMATORS = int(os.getenv('ML_GRADIENT_BOOSTING_N_ESTIMATORS', '50'))
ML_GRADIENT_BOOSTING_LEARNING_RATE = float(os.getenv('ML_GRADIENT_BOOSTING_LEARNING_RATE', '0.1'))
ML_GRADIENT_BOOSTING_MAX_DEPTH = int(os.getenv('ML_GRADIENT_BOOSTING_MAX_DEPTH', '6'))
ML_GRADIENT_BOOSTING_MIN_SAMPLES_SPLIT = int(os.getenv('ML_GRADIENT_BOOSTING_MIN_SAMPLES_SPLIT', '20'))
ML_GRADIENT_BOOSTING_MIN_SAMPLES_LEAF = int(os.getenv('ML_GRADIENT_BOOSTING_MIN_SAMPLES_LEAF', '10'))
ML_GRADIENT_BOOSTING_SUBSAMPLE = float(os.getenv('ML_GRADIENT_BOOSTING_SUBSAMPLE', '0.8'))
ML_GRADIENT_BOOSTING_MAX_FEATURES = os.getenv('ML_GRADIENT_BOOSTING_MAX_FEATURES', 'sqrt')

# XGBoost Specific Configuration
ML_XGBOOST_N_ESTIMATORS = int(os.getenv('ML_XGBOOST_N_ESTIMATORS', '50'))
ML_XGBOOST_LEARNING_RATE = float(os.getenv('ML_XGBOOST_LEARNING_RATE', '0.1'))
ML_XGBOOST_MAX_DEPTH = int(os.getenv('ML_XGBOOST_MAX_DEPTH', '6'))
ML_XGBOOST_MIN_CHILD_WEIGHT = int(os.getenv('ML_XGBOOST_MIN_CHILD_WEIGHT', '1'))
ML_XGBOOST_SUBSAMPLE = float(os.getenv('ML_XGBOOST_SUBSAMPLE', '0.8'))
ML_XGBOOST_COLSAMPLE_BYTREE = float(os.getenv('ML_XGBOOST_COLSAMPLE_BYTREE', '0.8'))
ML_XGBOOST_GAMMA = float(os.getenv('ML_XGBOOST_GAMMA', '0'))

# Random Forest Specific Configuration
ML_RF_N_ESTIMATORS = int(os.getenv('ML_RF_N_ESTIMATORS', '50'))
ML_RF_MAX_DEPTH = os.getenv('ML_RF_MAX_DEPTH', 'None')  # Can be None or int
ML_RF_MIN_SAMPLES_SPLIT = int(os.getenv('ML_RF_MIN_SAMPLES_SPLIT', '2'))
ML_RF_MIN_SAMPLES_LEAF = int(os.getenv('ML_RF_MIN_SAMPLES_LEAF', '1'))
ML_RF_MAX_FEATURES = os.getenv('ML_RF_MAX_FEATURES', 'sqrt')
ML_RF_BOOTSTRAP = os.getenv('ML_RF_BOOTSTRAP', 'true').lower() == 'true'

# SVM Specific Configuration
ML_SVM_C = float(os.getenv('ML_SVM_C', '1.0'))
ML_SVM_KERNEL = os.getenv('ML_SVM_KERNEL', 'rbf')
ML_SVM_GAMMA = os.getenv('ML_SVM_GAMMA', 'scale')
ML_SVM_CLASS_WEIGHT = os.getenv('ML_SVM_CLASS_WEIGHT', 'balanced')

# Logistic Regression Specific Configuration
ML_LR_C = float(os.getenv('ML_LR_C', '1.0'))
ML_LR_PENALTY = os.getenv('ML_LR_PENALTY', 'l2')
ML_LR_SOLVER = os.getenv('ML_LR_SOLVER', 'lbfgs')
ML_LR_MAX_ITER = int(os.getenv('ML_LR_MAX_ITER', '1000'))

# Advanced ML Configuration (PyTorch)
ML_USE_PYTORCH = os.getenv('ML_USE_PYTORCH', 'false').lower() == 'true'  # Enable PyTorch models
ML_PYTORCH_MODEL_TYPE = os.getenv('ML_PYTORCH_MODEL_TYPE', 'lstm')  # Options: 'lstm', 'gru', 'transformer'
ML_PYTORCH_HIDDEN_SIZE = int(os.getenv('ML_PYTORCH_HIDDEN_SIZE', '128'))  # Hidden layer size
ML_PYTORCH_NUM_LAYERS = int(os.getenv('ML_PYTORCH_NUM_LAYERS', '2'))  # Number of layers
ML_PYTORCH_DROPOUT = float(os.getenv('ML_PYTORCH_DROPOUT', '0.2'))  # Dropout rate
ML_PYTORCH_LEARNING_RATE = float(os.getenv('ML_PYTORCH_LEARNING_RATE', '0.001'))  # Learning rate
ML_PYTORCH_BATCH_SIZE = int(os.getenv('ML_PYTORCH_BATCH_SIZE', '32'))  # Batch size
ML_PYTORCH_EPOCHS = int(os.getenv('ML_PYTORCH_EPOCHS', '100'))  # Maximum training epochs
ML_PYTORCH_EARLY_STOPPING_PATIENCE = int(os.getenv('ML_PYTORCH_EARLY_STOPPING_PATIENCE', '10'))  # Early stopping patience

# AutoML Configuration (Optuna)
ML_USE_AUTOML = os.getenv('ML_USE_AUTOML', 'false').lower() == 'true'  # Enable automated hyperparameter optimization
ML_AUTOML_N_TRIALS = int(os.getenv('ML_AUTOML_N_TRIALS', '5'))  # Number of Optuna trials
ML_AUTOML_TIMEOUT = int(os.getenv('ML_AUTOML_TIMEOUT', '300'))  # Maximum time for AutoML optimization (seconds)
ML_AUTOML_STUDY_NAME = os.getenv('ML_AUTOML_STUDY_NAME', 'tradpal_gradient_boosting_optimization')  # Name for Optuna study
ML_AUTOML_STORAGE = os.getenv('ML_AUTOML_STORAGE', None)  # Database URL for Optuna storage
ML_AUTOML_SAMPLER = os.getenv('ML_AUTOML_SAMPLER', 'tpe')  # Sampler type
ML_AUTOML_PRUNER = os.getenv('ML_AUTOML_PRUNER', 'median')  # Pruner type

# Ensemble Methods Configuration
ML_USE_ENSEMBLE = os.getenv('ML_USE_ENSEMBLE', 'false').lower() == 'true'  # Enable ensemble predictions
ML_ENSEMBLE_WEIGHTS = {'ml': 0.6, 'ga': 0.4}  # Weights for ensemble combination
ML_ENSEMBLE_VOTING = os.getenv('ML_ENSEMBLE_VOTING', 'weighted')  # Voting strategy
ML_ENSEMBLE_MIN_CONFIDENCE = float(os.getenv('ML_ENSEMBLE_MIN_CONFIDENCE', '0.7'))  # Minimum confidence for ensemble signal

# Advanced ML Features Configuration
ML_ADVANCED_FEATURES_ENABLED = os.getenv('ML_ADVANCED_FEATURES_ENABLED', 'true').lower() == 'true'  # Enable advanced ML features
ML_ENSEMBLE_MODELS = os.getenv('ML_ENSEMBLE_MODELS', 'torch_ensemble,random_forest,gradient_boosting,lstm,transformer').split(',')  # Models to include in ensemble
ML_MARKET_REGIME_DETECTION = os.getenv('ML_MARKET_REGIME_DETECTION', 'true').lower() == 'true'  # Enable market regime detection
ML_REINFORCEMENT_LEARNING = os.getenv('ML_REINFORCEMENT_LEARNING', 'false').lower() == 'true'  # Enable reinforcement learning
ML_GPU_OPTIMIZATION = os.getenv('ML_GPU_OPTIMIZATION', 'true').lower() == 'true'  # Enable GPU optimization when available

# ML Training Configuration
ML_FEATURES = [
    'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
    'bb_upper', 'bb_lower', 'bb_width', 'atr',
    'volume_ratio', 'returns', 'momentum'
]  # Features used for ML training

ML_TARGET_HORIZON = ML_TRAINING_HORIZON  # Prediction horizon for target labels
ML_TRAINING_WINDOW = int(os.getenv('ML_TRAINING_WINDOW', '1000'))  # Minimum training data window size
ML_VALIDATION_SPLIT = ML_TEST_SIZE  # Validation split ratio
ML_RANDOM_STATE = int(os.getenv('ML_RANDOM_STATE', '42'))  # Random state for reproducibility

# Kelly Criterion Configuration
KELLY_ENABLED = os.getenv('KELLY_ENABLED', 'false').lower() == 'true'  # Enable Kelly Criterion position sizing
KELLY_FRACTION = float(os.getenv('KELLY_FRACTION', '0.5'))  # Fractional Kelly (0.5 = half Kelly)
KELLY_LOOKBACK_TRADES = int(os.getenv('KELLY_LOOKBACK_TRADES', '100'))  # Lookback period for win rate calculation
KELLY_MIN_TRADES = int(os.getenv('KELLY_MIN_TRADES', '20'))  # Minimum trades required for Kelly calculation

# Sentiment Analysis Configuration
SENTIMENT_ENABLED = os.getenv('SENTIMENT_ENABLED', 'false').lower() == 'true'  # Enable sentiment analysis
SENTIMENT_SOURCES = os.getenv('SENTIMENT_SOURCES', 'twitter,news,reddit').split(',')  # Data sources to use
SENTIMENT_UPDATE_INTERVAL = int(os.getenv('SENTIMENT_UPDATE_INTERVAL', '300'))  # Update interval in seconds
SENTIMENT_CACHE_TTL = int(os.getenv('SENTIMENT_CACHE_TTL', '1800'))  # Cache TTL for sentiment data
SENTIMENT_WEIGHT = float(os.getenv('SENTIMENT_WEIGHT', '0.2'))  # Weight of sentiment in signal generation
SENTIMENT_THRESHOLD = float(os.getenv('SENTIMENT_THRESHOLD', '0.1'))  # Minimum sentiment score for signal influence

# Twitter Sentiment Configuration
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', '')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', '')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')
TWITTER_SEARCH_TERMS = os.getenv('TWITTER_SEARCH_TERMS', 'BTC,Bitcoin,crypto').split(',')  # Search terms
TWITTER_MAX_TWEETS = int(os.getenv('TWITTER_MAX_TWEETS', '100'))  # Max tweets to analyze per update
TWITTER_LANGUAGE = os.getenv('TWITTER_LANGUAGE', 'en')  # Language filter

# News Sentiment Configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
NEWS_SOURCES = os.getenv('NEWS_SOURCES', 'coindesk,cointelegraph,bitcoinmagazine').split(',')  # News sources
NEWS_SEARCH_TERMS = os.getenv('NEWS_SEARCH_TERMS', 'Bitcoin,BTC,cryptocurrency').split(',')  # Search terms
NEWS_MAX_ARTICLES = int(os.getenv('NEWS_MAX_ARTICLES', '50'))  # Max articles to analyze per update
NEWS_LANGUAGE = os.getenv('NEWS_LANGUAGE', 'en')  # Language filter

# Reddit Sentiment Configuration
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'TradPal/1.0')
REDDIT_SUBREDDITS = os.getenv('REDDIT_SUBREDDITS', 'bitcoin,crypto,cryptocurrency').split(',')  # Subreddits
REDDIT_MAX_POSTS = int(os.getenv('REDDIT_MAX_POSTS', '50'))  # Max posts to analyze per update
REDDIT_TIME_FILTER = os.getenv('REDDIT_TIME_FILTER', 'day')  # Time filter

# Sentiment Analysis Model Configuration
SENTIMENT_MODEL_TYPE = os.getenv('SENTIMENT_MODEL_TYPE', 'vader')  # Options: 'vader', 'textblob', 'transformers'
SENTIMENT_MODEL_CACHE_DIR = os.getenv('SENTIMENT_MODEL_CACHE_DIR', 'cache/sentiment_models')  # Model cache directory
SENTIMENT_PREPROCESSING_ENABLED = os.getenv('SENTIMENT_PREPROCESSING_ENABLED', 'true').lower() == 'true'  # Enable preprocessing
SENTIMENT_REMOVE_STOPWORDS = os.getenv('SENTIMENT_REMOVE_STOPWORDS', 'true').lower() == 'true'  # Remove stopwords
SENTIMENT_LEMMATIZE = os.getenv('SENTIMENT_LEMMATIZE', 'false').lower() == 'true'  # Apply lemmatization