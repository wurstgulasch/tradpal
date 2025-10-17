# Configuration for TradPal - Main Configuration File
# This file imports settings from modular configuration files

import os
from dotenv import load_dotenv

# Import modular configurations
from .core_settings import *
from .ml_settings import *
from .service_settings import *
from .security_settings import *
from .performance_settings import *

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

# Legacy constants for backward compatibility
DEFAULT_DATA_LIMIT = 200
SYMBOL = 'BTC/USDT'
EXCHANGE = 'binance'
OUTPUT_FILE = 'output/signals.json'
LOG_FILE = 'logs/tradpal.log'

# Additional constants needed by various modules
LOOKBACK_DAYS = 30
API_KEY = os.getenv('API_KEY', '')
API_SECRET = os.getenv('API_SECRET', '')
ENABLE_MTLS = os.getenv('ENABLE_MTLS', 'true').lower() == 'true'
MAX_BACKTEST_RESULTS = 100
TIMEFRAME = '1m'
EMA_SHORT = 9
EMA_LONG = 21
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD_DEV = 2.0
ATR_PERIOD = 14
SL_MULTIPLIER = 1.5
TP_MULTIPLIER = 3.0
LEVERAGE_BASE = 5
LEVERAGE_MIN = 1
LEVERAGE_MAX = 10
MTA_ENABLED = True
MTA_TIMEFRAMES = ['5m', '15m']
ADX_THRESHOLD = 25
OUTPUT_FORMAT = 'json'
MTLS_CERT_PATH = os.getenv('MTLS_CERT_PATH', 'cache/security/certs/client.crt')
MTLS_KEY_PATH = os.getenv('MTLS_KEY_PATH', 'cache/security/certs/client.key')
CA_CERT_PATH = os.getenv('CA_CERT_PATH', 'cache/security/ca/ca_cert.pem')
SECURITY_SERVICE_URL = os.getenv('SECURITY_SERVICE_URL', 'http://localhost:8012')

# Additional legacy constants for backward compatibility
ADAPTIVE_OPTIMIZATION_ENABLED_LIVE = os.getenv('ADAPTIVE_OPTIMIZATION_ENABLED_LIVE', 'false').lower() == 'true'
MONITORING_STACK_ENABLED = os.getenv('MONITORING_STACK_ENABLED', 'true').lower() == 'true'

# Timeframe parameters (simplified version)
TIMEFRAME_PARAMS = {
    '1m': {
        'ema_short': 9, 'ema_long': 21, 'rsi_period': 14, 'bb_period': 20,
        'atr_period': 14, 'adx_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70
    },
    '5m': {
        'ema_short': 12, 'ema_long': 26, 'rsi_period': 14, 'bb_period': 20,
        'atr_period': 14, 'adx_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70
    },
    '1h': {
        'ema_short': 20, 'ema_long': 50, 'rsi_period': 14, 'bb_period': 20,
        'atr_period': 14, 'adx_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70
    }
}

def get_timeframe_params(timeframe=None):
    """Get parameters for specified timeframe, fallback to current TIMEFRAME"""
    if timeframe is None:
        timeframe = TIMEFRAME
    return TIMEFRAME_PARAMS.get(timeframe, None)
