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

def get_settings():
    """Get all configuration settings as a dictionary."""
    return {
        'symbol': SYMBOL,
        'exchange': EXCHANGE,
        'capital': CAPITAL,
        'risk_per_trade': RISK_PER_TRADE,
        'output_file': OUTPUT_FILE,
        'log_file': LOG_FILE,
        'ml_enabled': ML_ENABLED,
        'data_service_url': DATA_SERVICE_URL,
    }
