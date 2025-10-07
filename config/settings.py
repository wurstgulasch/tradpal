# Configuration for TradPal Indicator

# Trading pair and exchange
SYMBOL = 'EUR/USD'  # For ccxt
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
CAPITAL = 10000
RISK_PER_TRADE = 0.01

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

# Data and output
LOOKBACK_DAYS = 7
OUTPUT_FORMAT = 'json'
OUTPUT_FILE = 'output/signals.json'

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/tradpal_indicator.log'

# Environment variables (for security)
import os
from dotenv import load_dotenv

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
