# Core trading configuration settings
# Basic trading parameters, risk management, and timeframe configurations

import os
from typing import Dict, Any

# Additional constants needed by various modules
LOOKBACK_DAYS = 30
API_KEY = os.getenv('API_KEY', '')
API_SECRET = os.getenv('API_SECRET', '')
MAX_BACKTEST_RESULTS = 100
DEFAULT_DATA_LIMIT = 200  # Default limit for data queries
HISTORICAL_DATA_LIMIT = 1000  # Default limit for historical data
MTA_DATA_LIMIT = 50  # Limit for multi-timeframe analysis data
VOLATILITY_WINDOW = 14  # Window for volatility calculations
TREND_LOOKBACK = 10  # Lookback period for trend analysis
MAX_RETRIES_LIVE = 3  # Max retries for live data fetching
MAX_RETRIES_HISTORICAL = 2  # Max retries for historical data fetching
CACHE_TTL_LIVE = 30  # Cache TTL for live data (seconds)
CACHE_TTL_HISTORICAL = 300  # Cache TTL for historical data (seconds)
KRAKEN_MAX_PER_REQUEST = 720  # Kraken's max candles per request for 1m timeframe
JSON_INDENT = 4  # JSON output indentation
ML_RANDOM_STATE = int(os.getenv('ML_RANDOM_STATE', '42'))  # Random state for ML reproducibility

# Basic Trading Configuration
SYMBOL = os.getenv('TRADING_SYMBOL', 'BTC/USDT')  # Trading pair
EXCHANGE = os.getenv('TRADING_EXCHANGE', 'binance')  # Exchange for trading
TIMEFRAME = os.getenv('TRADING_TIMEFRAME', '1h')  # Default timeframe

# Risk Management
CAPITAL = float(os.getenv('INITIAL_CAPITAL', '10000'))  # Starting capital
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.01'))  # Risk per trade (1%)
INITIAL_CAPITAL = CAPITAL  # Alias for backward compatibility

# Multi-timeframe Analysis
MTA_ENABLED = os.getenv('MTA_ENABLED', 'true').lower() == 'true'
MTA_HIGHER_TIMEFRAME = os.getenv('MTA_HIGHER_TIMEFRAME', '5m')  # Higher timeframe for trend confirmation
MTA_TIMEFRAMES = ['5m', '15m']  # Available higher timeframes

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

# Optional indicators
ADX_ENABLED = os.getenv('ADX_ENABLED', 'false').lower() == 'true'
ADX_THRESHOLD = current_params['adx_threshold']  # ADX threshold for trend strength
FIBONACCI_ENABLED = os.getenv('FIBONACCI_ENABLED', 'false').lower() == 'true'
VOLATILITY_FILTER_ENABLED = os.getenv('VOLATILITY_FILTER_ENABLED', 'false').lower() == 'true'

# Signal generation mode
STRICT_SIGNALS_ENABLED = os.getenv('STRICT_SIGNALS_ENABLED', 'true').lower() == 'true'  # If False, use only EMA crossover for signals

# Configuration Mode
CONFIG_MODE = os.getenv('CONFIG_MODE', 'optimized')  # 'conservative', 'discovery', or 'optimized'

# Indicator Configurations
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
        return CONSERVATIVE_CONFIG

# Discovery Mode Parameters
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

# Fitness Function Configuration
FITNESS_WEIGHTS = {
    'sharpe_ratio': float(os.getenv('FITNESS_SHARPE_WEIGHT', '30.0')),      # Risk-adjusted returns (30%)
    'calmar_ratio': float(os.getenv('FITNESS_CALMAR_WEIGHT', '25.0')),     # Return-to-drawdown ratio (25%)
    'total_pnl': float(os.getenv('FITNESS_PNL_WEIGHT', '30.0')),           # Total profit/loss - OUTPERFORMANCE (30%)
    'profit_factor': float(os.getenv('FITNESS_PROFIT_FACTOR_WEIGHT', '10.0')), # Gross profit / gross loss (10%)
    'win_rate': float(os.getenv('FITNESS_WIN_RATE_WEIGHT', '5.0'))         # Win rate percentage (5%)
}

FITNESS_BOUNDS = {
    'sharpe_ratio': {'min': -5, 'max': 8},      # Sharpe ratio bounds for normalization
    'calmar_ratio': {'min': -10, 'max': 15},    # Calmar ratio bounds
    'total_pnl': {'min': -100, 'max': 200},     # Total P&L percentage bounds
    'profit_factor': {'min': 0, 'max': 10},     # Profit factor bounds
    'win_rate': {'min': 0, 'max': 100}          # Win rate percentage bounds
}

FITNESS_RISK_PENALTIES = {
    'max_drawdown_15': 0.9,    # 10% penalty for drawdown > 15%
    'max_drawdown_20': 0.7,    # 30% penalty for drawdown > 20%
    'max_drawdown_30': 0.5,    # 50% penalty for drawdown > 30%
    'insufficient_trades': 0.7,  # 30% penalty for < 10 trades
    'overtrading': 0.8,        # 20% penalty for > 500 trades
    'negative_pnl_high_risk': 0.3,  # 70% penalty for negative P&L + high drawdown
    'positive_pnl_bonus': 1.05   # 5% bonus for positive P&L
}

# Output configuration
OUTPUT_FILE = os.getenv('OUTPUT_FILE', 'output/signals.json')  # Default output file for signals
OUTPUT_FORMAT = os.getenv('OUTPUT_FORMAT', 'json')  # Output format: 'json', 'csv', 'both'
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')  # Output directory for results

# Logging configuration
LOG_FILE = os.getenv('LOG_FILE', 'logs/tradpal.log')  # Main log file
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')  # Logging level: DEBUG, INFO, WARNING, ERROR
LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', '10485760'))  # 10MB max log file size
LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))  # Number of backup log files to keep

# Data Source Configuration
DATA_SOURCE = os.getenv('DATA_SOURCE', 'yahoo_finance')  # Default data source

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