# Configuration for TradPal - Main Configuration File
# This file provides lazy loading of modular configurations

import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

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

class LazyConfig:
    """
    Lazy configuration loader that only loads modules when accessed.
    This reduces memory footprint and improves startup time.
    """

    def __init__(self):
        self._loaded_modules: Dict[str, Dict[str, Any]] = {}
        self._module_loaders = {
            'core': self._load_core_settings,
            'ml': self._load_ml_settings,
            'service': self._load_service_settings,
            'security': self._load_security_settings,
            'performance': self._load_performance_settings,
        }

    def _load_core_settings(self) -> Dict[str, Any]:
        """Lazy load core trading and risk management settings"""
        from .core_settings import (
            DEFAULT_DATA_LIMIT, SYMBOL, EXCHANGE, OUTPUT_FILE, LOG_FILE,
            LOOKBACK_DAYS, API_KEY, API_SECRET, TIMEFRAME, EMA_SHORT, EMA_LONG,
            RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT, BB_PERIOD, BB_STD_DEV, ATR_PERIOD, SL_MULTIPLIER,
            TP_MULTIPLIER, LEVERAGE_BASE, LEVERAGE_MIN, LEVERAGE_MAX,
            ADX_THRESHOLD, OUTPUT_FORMAT, TIMEFRAME_PARAMS
        )
        from .ml_settings import ML_ENABLED
        return {
            'DEFAULT_DATA_LIMIT': DEFAULT_DATA_LIMIT,
            'SYMBOL': SYMBOL,
            'EXCHANGE': EXCHANGE,
            'OUTPUT_FILE': OUTPUT_FILE,
            'LOG_FILE': LOG_FILE,
            'LOOKBACK_DAYS': LOOKBACK_DAYS,
            'API_KEY': API_KEY,
            'API_SECRET': API_SECRET,
            'TIMEFRAME': TIMEFRAME,
            'EMA_SHORT': EMA_SHORT,
            'EMA_LONG': EMA_LONG,
            'RSI_PERIOD': RSI_PERIOD,
            'RSI_OVERSOLD': RSI_OVERSOLD,
            'RSI_OVERBOUGHT': RSI_OVERBOUGHT,
            'BB_PERIOD': BB_PERIOD,
            'BB_STD_DEV': BB_STD_DEV,
            'ATR_PERIOD': ATR_PERIOD,
            'SL_MULTIPLIER': SL_MULTIPLIER,
            'TP_MULTIPLIER': TP_MULTIPLIER,
            'LEVERAGE_BASE': LEVERAGE_BASE,
            'LEVERAGE_MIN': LEVERAGE_MIN,
            'LEVERAGE_MAX': LEVERAGE_MAX,
            'ADX_THRESHOLD': ADX_THRESHOLD,
            'OUTPUT_FORMAT': OUTPUT_FORMAT,
            'TIMEFRAME_PARAMS': TIMEFRAME_PARAMS,
            'ML_ENABLED': ML_ENABLED,
        }

    def _load_ml_settings(self) -> Dict[str, Any]:
        """Lazy load machine learning and AI configurations"""
        from .ml_settings import (
            ML_ENABLED, ML_MODEL_DIR, ML_MODELS_DIR, ML_MODEL_TYPE, ML_TRAINING_ENABLED, ML_MODEL_PATH,
            FEATURE_ENGINEERING_ENABLED, ENSEMBLE_METHOD, CONFIDENCE_THRESHOLD,
            RISK_ADJUSTMENT_ENABLED, MARKET_REGIME_DETECTION_ENABLED,
            ADAPTIVE_LEARNING_RATE, MODEL_UPDATE_FREQUENCY,
            ML_FEATURES, ML_TARGET_HORIZON, ML_TRAINING_WINDOW, ML_VALIDATION_SPLIT, ML_RANDOM_STATE
        )
        return {
            'ML_ENABLED': ML_ENABLED,
            'ML_MODEL_DIR': ML_MODEL_DIR,
            'ML_MODELS_DIR': ML_MODELS_DIR,
            'ML_MODEL_TYPE': ML_MODEL_TYPE,
            'ML_TRAINING_ENABLED': ML_TRAINING_ENABLED,
            'ML_MODEL_PATH': ML_MODEL_PATH,
            'FEATURE_ENGINEERING_ENABLED': FEATURE_ENGINEERING_ENABLED,
            'ENSEMBLE_METHOD': ENSEMBLE_METHOD,
            'CONFIDENCE_THRESHOLD': CONFIDENCE_THRESHOLD,
            'RISK_ADJUSTMENT_ENABLED': RISK_ADJUSTMENT_ENABLED,
            'MARKET_REGIME_DETECTION_ENABLED': MARKET_REGIME_DETECTION_ENABLED,
            'ADAPTIVE_LEARNING_RATE': ADAPTIVE_LEARNING_RATE,
            'MODEL_UPDATE_FREQUENCY': MODEL_UPDATE_FREQUENCY,
            'ML_FEATURES': ML_FEATURES,
            'ML_TARGET_HORIZON': ML_TARGET_HORIZON,
            'ML_TRAINING_WINDOW': ML_TRAINING_WINDOW,
            'ML_VALIDATION_SPLIT': ML_VALIDATION_SPLIT,
            'ML_RANDOM_STATE': ML_RANDOM_STATE,
        }

    def _load_service_settings(self) -> Dict[str, Any]:
        """Lazy load microservices and data mesh settings"""
        from .service_settings import (
            ENABLE_MTLS, MTLS_CERT_PATH, MTLS_KEY_PATH, CA_CERT_PATH,
            SECURITY_SERVICE_URL, MTA_ENABLED, MTA_TIMEFRAMES,
            MAX_BACKTEST_RESULTS, MONITORING_STACK_ENABLED,
            PERFORMANCE_MONITORING_ENABLED, ADAPTIVE_OPTIMIZATION_ENABLED_LIVE
        )
        return {
            'ENABLE_MTLS': ENABLE_MTLS,
            'MTLS_CERT_PATH': MTLS_CERT_PATH,
            'MTLS_KEY_PATH': MTLS_KEY_PATH,
            'CA_CERT_PATH': CA_CERT_PATH,
            'SECURITY_SERVICE_URL': SECURITY_SERVICE_URL,
            'MTA_ENABLED': MTA_ENABLED,
            'MTA_TIMEFRAMES': MTA_TIMEFRAMES,
            'MAX_BACKTEST_RESULTS': MAX_BACKTEST_RESULTS,
            'MONITORING_STACK_ENABLED': MONITORING_STACK_ENABLED,
            'PERFORMANCE_MONITORING_ENABLED': PERFORMANCE_MONITORING_ENABLED,
            'ADAPTIVE_OPTIMIZATION_ENABLED_LIVE': ADAPTIVE_OPTIMIZATION_ENABLED_LIVE,
        }

    def _load_security_settings(self) -> Dict[str, Any]:
        """Lazy load security and authentication settings"""
        from .security_settings import (
            JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRATION_HOURS,
            ENCRYPTION_KEY, VAULT_ENABLED, VAULT_URL, VAULT_TOKEN,
            API_RATE_LIMIT, API_RATE_LIMIT_WINDOW
        )
        return {
            'JWT_SECRET_KEY': JWT_SECRET_KEY,
            'JWT_ALGORITHM': JWT_ALGORITHM,
            'JWT_EXPIRATION_HOURS': JWT_EXPIRATION_HOURS,
            'ENCRYPTION_KEY': ENCRYPTION_KEY,
            'VAULT_ENABLED': VAULT_ENABLED,
            'VAULT_URL': VAULT_URL,
            'VAULT_TOKEN': VAULT_TOKEN,
            'API_RATE_LIMIT': API_RATE_LIMIT,
            'API_RATE_LIMIT_WINDOW': API_RATE_LIMIT_WINDOW,
        }

    def _load_performance_settings(self) -> Dict[str, Any]:
        """Lazy load performance optimization settings"""
        from .performance_settings import (
            GPU_ACCELERATION_ENABLED, MEMORY_MAPPED_DATA_ENABLED,
            CHUNK_SIZE, CACHE_ENABLED, CACHE_TTL, ASYNC_PROCESSING_ENABLED,
            VECTORIZATION_ENABLED, OPTIMIZATION_LEVEL
        )
        return {
            'GPU_ACCELERATION_ENABLED': GPU_ACCELERATION_ENABLED,
            'MEMORY_MAPPED_DATA_ENABLED': MEMORY_MAPPED_DATA_ENABLED,
            'CHUNK_SIZE': CHUNK_SIZE,
            'CACHE_ENABLED': CACHE_ENABLED,
            'CACHE_TTL': CACHE_TTL,
            'ASYNC_PROCESSING_ENABLED': ASYNC_PROCESSING_ENABLED,
            'VECTORIZATION_ENABLED': VECTORIZATION_ENABLED,
            'OPTIMIZATION_LEVEL': OPTIMIZATION_LEVEL,
        }

    def get(self, module: str, key: str, default: Any = None) -> Any:
        """Get a configuration value from a specific module"""
        if module not in self._loaded_modules:
            if module in self._module_loaders:
                self._loaded_modules[module] = self._module_loaders[module]()
            else:
                raise ValueError(f"Unknown configuration module: {module}")

        return self._loaded_modules[module].get(key, default)

    def get_module(self, module: str) -> Dict[str, Any]:
        """Get all settings from a specific module"""
        if module not in self._loaded_modules:
            if module in self._module_loaders:
                self._loaded_modules[module] = self._module_loaders[module]()
            else:
                raise ValueError(f"Unknown configuration module: {module}")

        return self._loaded_modules[module].copy()

    def get_all_loaded(self) -> Dict[str, Dict[str, Any]]:
        """Get all currently loaded configuration modules"""
        return self._loaded_modules.copy()

    def clear_cache(self):
        """Clear the loaded modules cache"""
        self._loaded_modules.clear()

# Global lazy configuration instance
config = LazyConfig()

# Legacy constants for backward compatibility (lazy loaded)
def get_legacy_constant(name: str, default: Any = None) -> Any:
    """Get legacy constant with lazy loading"""
    # Map legacy names to new module structure
    legacy_mapping = {
        'DEFAULT_DATA_LIMIT': ('core', 'DEFAULT_DATA_LIMIT'),
        'SYMBOL': ('core', 'SYMBOL'),
        'EXCHANGE': ('core', 'EXCHANGE'),
        'OUTPUT_FILE': ('core', 'OUTPUT_FILE'),
        'LOG_FILE': ('core', 'LOG_FILE'),
        'LOOKBACK_DAYS': ('core', 'LOOKBACK_DAYS'),
        'API_KEY': ('core', 'API_KEY'),
        'API_SECRET': ('core', 'API_SECRET'),
        'TIMEFRAME': ('core', 'TIMEFRAME'),
        'EMA_SHORT': ('core', 'EMA_SHORT'),
        'EMA_LONG': ('core', 'EMA_LONG'),
        'RSI_OVERSOLD': ('core', 'RSI_OVERSOLD'),
        'RSI_OVERBOUGHT': ('core', 'RSI_OVERBOUGHT'),
        'BB_STD_DEV': ('core', 'BB_STD_DEV'),
        'ATR_PERIOD': ('core', 'ATR_PERIOD'),
        'SL_MULTIPLIER': ('core', 'SL_MULTIPLIER'),
        'TP_MULTIPLIER': ('core', 'TP_MULTIPLIER'),
        'LEVERAGE_BASE': ('core', 'LEVERAGE_BASE'),
        'LEVERAGE_MIN': ('core', 'LEVERAGE_MIN'),
        'LEVERAGE_MAX': ('core', 'LEVERAGE_MAX'),
        'ADX_THRESHOLD': ('core', 'ADX_THRESHOLD'),
        'OUTPUT_FORMAT': ('core', 'OUTPUT_FORMAT'),
        'CAPITAL': ('core', 'CAPITAL'),
        'INITIAL_CAPITAL': ('core', 'CAPITAL'),
        'RISK_PER_TRADE': ('core', 'RISK_PER_TRADE'),
        'TIMEFRAME_PARAMS': ('core', 'TIMEFRAME_PARAMS'),
        'ENABLE_MTLS': ('service', 'ENABLE_MTLS'),
        'MTLS_CERT_PATH': ('service', 'MTLS_CERT_PATH'),
        'MTLS_KEY_PATH': ('service', 'MTLS_KEY_PATH'),
        'CA_CERT_PATH': ('service', 'CA_CERT_PATH'),
        'SECURITY_SERVICE_URL': ('service', 'SECURITY_SERVICE_URL'),
        'MTA_ENABLED': ('service', 'MTA_ENABLED'),
        'MTA_TIMEFRAMES': ('service', 'MTA_TIMEFRAMES'),
        'MAX_BACKTEST_RESULTS': ('service', 'MAX_BACKTEST_RESULTS'),
        'MONITORING_STACK_ENABLED': ('service', 'MONITORING_STACK_ENABLED'),
        'PERFORMANCE_MONITORING_ENABLED': ('service', 'PERFORMANCE_MONITORING_ENABLED'),
        'ADAPTIVE_OPTIMIZATION_ENABLED_LIVE': ('service', 'ADAPTIVE_OPTIMIZATION_ENABLED_LIVE'),
        'ML_MODEL_TYPE': ('ml', 'ML_MODEL_TYPE'),
        'ML_TRAINING_ENABLED': ('ml', 'ML_TRAINING_ENABLED'),
        'ML_MODELS_DIR': ('ml', 'ML_MODELS_DIR'),
        'GPU_ACCELERATION_ENABLED': ('performance', 'GPU_ACCELERATION_ENABLED'),
        'CACHE_ENABLED': ('performance', 'CACHE_ENABLED'),
        'ENABLE_MTLS': ('service', 'ENABLE_MTLS'),
        'ML_ENABLED': ('ml', 'ML_ENABLED'),
        'ML_CONFIDENCE_THRESHOLD': ('ml', 'CONFIDENCE_THRESHOLD'),
        'ML_FEATURES': ('ml', 'ML_FEATURES'),
        'PERFORMANCE_ENABLED': ('performance', 'PERFORMANCE_MONITORING_ENABLED'),
        'PARALLEL_PROCESSING_ENABLED': ('performance', 'PARALLEL_PROCESSING_ENABLED'),
        'ML_TARGET_HORIZON': ('ml', 'ML_TARGET_HORIZON'),
        'VECTORIZATION_ENABLED': ('performance', 'VECTORIZATION_ENABLED'),
        'ML_TRAINING_WINDOW': ('ml', 'ML_TRAINING_WINDOW'),
        'MEMORY_OPTIMIZATION_ENABLED': ('performance', 'MEMORY_OPTIMIZATION_ENABLED'),
        'ML_VALIDATION_SPLIT': ('ml', 'ML_VALIDATION_SPLIT'),
        'MAX_WORKERS': ('performance', 'MAX_WORKERS'),
        'ML_RANDOM_STATE': ('ml', 'ML_RANDOM_STATE'),
        'CHUNK_SIZE': ('performance', 'CHUNK_SIZE'),
        'PERFORMANCE_LOG_LEVEL': ('performance', 'PERFORMANCE_LOG_LEVEL'),
        'REDIS_ENABLED': ('performance', 'REDIS_ENABLED'),
        'REDIS_HOST': ('performance', 'REDIS_HOST'),
        'REDIS_PORT': ('performance', 'REDIS_PORT'),
        'REDIS_DB': ('performance', 'REDIS_DB'),
        'REDIS_PASSWORD': ('performance', 'REDIS_PASSWORD'),
        'REDIS_TTL_INDICATORS': ('service', 'REDIS_TTL_INDICATORS'),
        'REDIS_TTL_API': ('service', 'REDIS_TTL_API'),
        'REDIS_MAX_CONNECTIONS': ('service', 'REDIS_MAX_CONNECTIONS'),
        'ADVANCED_SIGNAL_GENERATION_ENABLED': ('service', 'ADVANCED_SIGNAL_GENERATION_ENABLED'),
        'ADVANCED_SIGNAL_GENERATION_MODE': ('service', 'ADVANCED_SIGNAL_GENERATION_MODE'),
    }

    if name in legacy_mapping:
        module, key = legacy_mapping[name]
        return config.get(module, key, default)
    else:
        return default

# Backward compatibility - expose commonly used constants
DEFAULT_DATA_LIMIT = get_legacy_constant('DEFAULT_DATA_LIMIT', 200)
SYMBOL = get_legacy_constant('SYMBOL', 'BTC/USDT')
EXCHANGE = get_legacy_constant('EXCHANGE', 'binance')
OUTPUT_FILE = get_legacy_constant('OUTPUT_FILE', 'output/signals.json')
LOG_FILE = get_legacy_constant('LOG_FILE', 'logs/tradpal.log')
LOOKBACK_DAYS = get_legacy_constant('LOOKBACK_DAYS', 30)
API_KEY = get_legacy_constant('API_KEY', '')
API_SECRET = get_legacy_constant('API_SECRET', '')
TIMEFRAME = get_legacy_constant('TIMEFRAME', '1m')
EMA_SHORT = get_legacy_constant('EMA_SHORT', 9)
EMA_LONG = get_legacy_constant('EMA_LONG', 21)
RSI_PERIOD = get_legacy_constant('RSI_PERIOD', 14)
RSI_OVERSOLD = get_legacy_constant('RSI_OVERSOLD', 30)
RSI_OVERBOUGHT = get_legacy_constant('RSI_OVERBOUGHT', 70)
BB_PERIOD = get_legacy_constant('BB_PERIOD', 20)
BB_STD_DEV = get_legacy_constant('BB_STD_DEV', 2.0)
ATR_PERIOD = get_legacy_constant('ATR_PERIOD', 14)
SL_MULTIPLIER = get_legacy_constant('SL_MULTIPLIER', 1.5)
TP_MULTIPLIER = get_legacy_constant('TP_MULTIPLIER', 3.0)
LEVERAGE_BASE = get_legacy_constant('LEVERAGE_BASE', 5)
LEVERAGE_MIN = get_legacy_constant('LEVERAGE_MIN', 1)
LEVERAGE_MAX = get_legacy_constant('LEVERAGE_MAX', 10)
ADX_THRESHOLD = get_legacy_constant('ADX_THRESHOLD', 25)
OUTPUT_FORMAT = get_legacy_constant('OUTPUT_FORMAT', 'json')
CAPITAL = get_legacy_constant('CAPITAL', 10000)
INITIAL_CAPITAL = get_legacy_constant('INITIAL_CAPITAL', 10000)
RISK_PER_TRADE = get_legacy_constant('RISK_PER_TRADE', 0.01)
TIMEFRAME_PARAMS = get_legacy_constant('TIMEFRAME_PARAMS', {})
ENABLE_MTLS = get_legacy_constant('ENABLE_MTLS', True)
MTLS_CERT_PATH = get_legacy_constant('MTLS_CERT_PATH', 'cache/security/certs/client.crt')
MTLS_KEY_PATH = get_legacy_constant('MTLS_KEY_PATH', 'cache/security/certs/client.key')
CA_CERT_PATH = get_legacy_constant('CA_CERT_PATH', 'cache/security/ca/ca_cert.pem')
SECURITY_SERVICE_URL = get_legacy_constant('SECURITY_SERVICE_URL', 'http://localhost:8012')
MTA_ENABLED = get_legacy_constant('MTA_ENABLED', True)
MTA_TIMEFRAMES = get_legacy_constant('MTA_TIMEFRAMES', ['5m', '15m'])
MAX_BACKTEST_RESULTS = get_legacy_constant('MAX_BACKTEST_RESULTS', 100)
ADAPTIVE_OPTIMIZATION_ENABLED_LIVE = get_legacy_constant('ADAPTIVE_OPTIMIZATION_ENABLED_LIVE', False)
MONITORING_STACK_ENABLED = get_legacy_constant('MONITORING_STACK_ENABLED', True)
PERFORMANCE_MONITORING_ENABLED = get_legacy_constant('PERFORMANCE_MONITORING_ENABLED', True)
ML_MODEL_TYPE = get_legacy_constant('ML_MODEL_TYPE', 'xgboost')
ML_TRAINING_ENABLED = get_legacy_constant('ML_TRAINING_ENABLED', True)
GPU_ACCELERATION_ENABLED = get_legacy_constant('GPU_ACCELERATION_ENABLED', True)
CACHE_ENABLED = get_legacy_constant('CACHE_ENABLED', True)
ML_MODELS_DIR = get_legacy_constant('ML_MODELS_DIR', 'cache/ml_models')
ML_ENABLED = get_legacy_constant('ML_ENABLED', True)
ML_CONFIDENCE_THRESHOLD = get_legacy_constant('ML_CONFIDENCE_THRESHOLD', 0.7)
ML_FEATURES = get_legacy_constant('ML_FEATURES', [])
PERFORMANCE_ENABLED = get_legacy_constant('PERFORMANCE_ENABLED', True)
PARALLEL_PROCESSING_ENABLED = get_legacy_constant('PARALLEL_PROCESSING_ENABLED', True)
ML_TARGET_HORIZON = get_legacy_constant('ML_TARGET_HORIZON', 5)
VECTORIZATION_ENABLED = get_legacy_constant('VECTORIZATION_ENABLED', True)
ML_TRAINING_WINDOW = get_legacy_constant('ML_TRAINING_WINDOW', 1000)
MEMORY_OPTIMIZATION_ENABLED = get_legacy_constant('MEMORY_OPTIMIZATION_ENABLED', True)
ML_VALIDATION_SPLIT = get_legacy_constant('ML_VALIDATION_SPLIT', 0.2)
MAX_WORKERS = get_legacy_constant('MAX_WORKERS', 4)
ML_RANDOM_STATE = get_legacy_constant('ML_RANDOM_STATE', 42)
CHUNK_SIZE = get_legacy_constant('CHUNK_SIZE', 1000)
PERFORMANCE_LOG_LEVEL = get_legacy_constant('PERFORMANCE_LOG_LEVEL', 'INFO')
REDIS_ENABLED = get_legacy_constant('REDIS_ENABLED', True)
REDIS_HOST = get_legacy_constant('REDIS_HOST', 'localhost')
REDIS_PORT = get_legacy_constant('REDIS_PORT', 6379)
REDIS_DB = get_legacy_constant('REDIS_DB', 1)
REDIS_PASSWORD = get_legacy_constant('REDIS_PASSWORD', '')
REDIS_TTL_INDICATORS = get_legacy_constant('REDIS_TTL_INDICATORS', 3600)
REDIS_TTL_API = get_legacy_constant('REDIS_TTL_API', 300)
REDIS_MAX_CONNECTIONS = get_legacy_constant('REDIS_MAX_CONNECTIONS', 20)
ADVANCED_SIGNAL_GENERATION_ENABLED = get_legacy_constant('ADVANCED_SIGNAL_GENERATION_ENABLED', True)
ADVANCED_SIGNAL_GENERATION_MODE = get_legacy_constant('ADVANCED_SIGNAL_GENERATION_MODE', 'hybrid')
ALTERNATIVE_DATA_UPDATE_INTERVAL = get_legacy_constant('ALTERNATIVE_DATA_UPDATE_INTERVAL', 300)
SENTIMENT_DATA_SOURCES = get_legacy_constant('SENTIMENT_DATA_SOURCES', ['twitter', 'reddit', 'news'])
ONCHAIN_DATA_SOURCES = get_legacy_constant('ONCHAIN_DATA_SOURCES', ['glassnode', 'blockchain_com'])
ECONOMIC_DATA_SOURCES = get_legacy_constant('ECONOMIC_DATA_SOURCES', ['fred', 'bureau_labor', 'alpha_vantage'])

# ML and service enablement flags
ENABLE_ML = get_legacy_constant('ENABLE_ML', True)
ENABLE_ML_TRAINER = get_legacy_constant('ENABLE_ML_TRAINER', True)
ENABLE_MLOPS = get_legacy_constant('ENABLE_MLOPS', True)
ENABLE_DISCOVERY = get_legacy_constant('ENABLE_DISCOVERY', True)
ENABLE_BACKTEST = get_legacy_constant('ENABLE_BACKTEST', True)
ENABLE_LIVE_TRADING = get_legacy_constant('ENABLE_LIVE_TRADING', True)
ENABLE_NOTIFICATIONS = get_legacy_constant('ENABLE_NOTIFICATIONS', True)

def get_timeframe_params(timeframe=None):
    """Get parameters for specified timeframe, fallback to current TIMEFRAME"""
    timeframe_params = config.get('core', 'TIMEFRAME_PARAMS', {})
    if timeframe is None:
        timeframe = config.get('core', 'TIMEFRAME', '1m')
    return timeframe_params.get(timeframe, None)

# Import validation functions from core_settings
from .core_settings import validate_timeframe, validate_risk_params
