import logging
import logging.handlers
from pathlib import Path
from config.settings import LOG_LEVEL, LOG_FILE

def setup_logging():
    """
    Setup comprehensive logging system for audit trails and debugging.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(LOG_FILE).parent
    log_dir.mkdir(exist_ok=True)

    # Configure logger
    logger = logging.getLogger('tradpal_indicator')
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Global logger instance
logger = setup_logging()

def log_signal(signal_type, price, rsi, ema9, ema21, position_size_pct, stop_loss, take_profit, leverage):
    """
    Log trading signal details for audit trail.
    """
    logger.info(f"SIGNAL: {signal_type} | Price: {price:.5f} | RSI: {rsi:.2f} | "
                f"EMA9: {ema9:.5f} | EMA21: {ema21:.5f} | Position: {position_size_pct:.2f}% | "
                f"SL: {stop_loss:.5f} | TP: {take_profit:.5f} | Leverage: {leverage}x")

def log_error(error_msg, exc_info=None):
    """
    Log error with optional exception info.
    """
    logger.error(error_msg, exc_info=exc_info)

def log_trade_execution(trade_details):
    """
    Log trade execution details.
    """
    logger.info(f"TRADE EXECUTED: {trade_details}")

def log_system_status(status_msg):
    """
    Log system status updates.
    """
    logger.info(f"SYSTEM: {status_msg}")

def log_api_call(endpoint, params=None, response_time=None):
    """
    Log API calls for monitoring.
    """
    params_str = f" | Params: {params}" if params else ""
    time_str = f" | Time: {response_time:.2f}s" if response_time else ""
    logger.debug(f"API CALL: {endpoint}{params_str}{time_str}")

def log_performance_metrics(metrics):
    """
    Log performance metrics.
    """
    logger.info(f"PERFORMANCE: {metrics}")