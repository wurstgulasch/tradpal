import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from config.settings import LOG_LEVEL, LOG_FILE


class StructuredJSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        # Create base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                             'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                             'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                             'thread', 'threadName', 'processName', 'process', 'message']:
                    log_entry[key] = value

        return json.dumps(log_entry, default=str)


class TradingLogger:
    """Enhanced logger with trading-specific methods."""

    def __init__(self):
        self.logger = None
        self.setup_logging()

    def setup_logging(self):
        """Setup comprehensive logging system."""
        # Create logs directory if it doesn't exist
        log_dir = Path(LOG_FILE).parent
        log_dir.mkdir(exist_ok=True)

        # Configure logger
        self.logger = logging.getLogger('tradpal_indicator')
        self.logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatters
        file_formatter = StructuredJSONFormatter()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # File handler with rotation (JSON format)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        # Console handler (human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)

        # Error handler (separate file for errors)
        error_handler = logging.handlers.RotatingFileHandler(
            str(Path(LOG_FILE).parent / 'errors.log'), maxBytes=5*1024*1024, backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(error_handler)

    def log_trading_signal(self, signal_data: Dict[str, Any]):
        """Log trading signal with structured data."""
        self.logger.info("Trading signal generated", extra={
            'signal_type': signal_data.get('signal_type'),
            'symbol': signal_data.get('symbol', 'EUR/USD'),
            'price': signal_data.get('price'),
            'indicators': {
                'rsi': signal_data.get('rsi'),
                'ema9': signal_data.get('ema9'),
                'ema21': signal_data.get('ema21'),
                'bb_upper': signal_data.get('bb_upper'),
                'bb_lower': signal_data.get('bb_lower'),
                'atr': signal_data.get('atr')
            },
            'risk_management': {
                'position_size': signal_data.get('position_size'),
                'stop_loss': signal_data.get('stop_loss'),
                'take_profit': signal_data.get('take_profit'),
                'leverage': signal_data.get('leverage')
            }
        })

    def log_system_status(self, status: str, **kwargs):
        """Log system status updates."""
        self.logger.info(f"System status: {status}", extra=kwargs)

    def log_error(self, error_msg: str, **kwargs):
        """Log errors with context."""
        self.logger.error(f"Error: {error_msg}", extra=kwargs)

    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics."""
        self.logger.info("Performance metrics calculated", extra={
            'metrics': metrics,
            'total_trades': metrics.get('total_trades'),
            'win_rate': metrics.get('win_rate'),
            'total_pnl': metrics.get('total_pnl'),
            'max_drawdown': metrics.get('max_drawdown')
        })

    def log_api_call(self, endpoint: str, success: bool, duration: float, **kwargs):
        """Log API call details."""
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, f"API call to {endpoint}", extra={
            'api_call': {
                'endpoint': endpoint,
                'success': success,
                'duration': duration,
                **kwargs
            }
        })

    def log_cache_operation(self, operation: str, key: str, hit: bool = None, **kwargs):
        """Log cache operations."""
        extra_data = {
            'cache_operation': operation,
            'cache_key': key,
            **kwargs
        }
        if hit is not None:
            extra_data['cache_hit'] = hit

        self.logger.debug(f"Cache {operation}: {key}", extra=extra_data)


# Global logger instance
trading_logger = TradingLogger()
logger = trading_logger.logger

def log_signal(signal_type, price, rsi, ema9, ema21, position_size_pct, stop_loss, take_profit, leverage):
    """
    Log trading signal details for audit trail.
    """
    signal_data = {
        'signal_type': signal_type,
        'price': price,
        'rsi': rsi,
        'ema9': ema9,
        'ema21': ema21,
        'position_size': position_size_pct,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'leverage': leverage
    }
    trading_logger.log_trading_signal(signal_data)

def log_error(error_msg, exc_info=None):
    """
    Log error with optional exception info.
    """
    trading_logger.log_error(error_msg, exc_info=exc_info)

def log_trade_execution(trade_details):
    """
    Log trade execution details.
    """
    logger.info(f"TRADE EXECUTED: {trade_details}")

def log_system_status(status_msg):
    """
    Log system status updates.
    """
    trading_logger.log_system_status(status_msg)

def log_api_call(endpoint, params=None, response_time=None):
    """
    Log API calls for monitoring.
    """
    success = response_time is not None  # Assume success if response time provided
    trading_logger.log_api_call(endpoint, success, response_time or 0, params=params)

def log_performance_metrics(metrics):
    """
    Log performance metrics.
    """
    trading_logger.log_performance_metrics(metrics)