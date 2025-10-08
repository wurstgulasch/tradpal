"""
Test logging configuration functionality for the trading indicator system.
Tests structured JSON formatting, trading logger methods, and logging setup.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from src.logging_config import (
    StructuredJSONFormatter, TradingLogger, trading_logger, logger,
    log_signal, log_error, log_trade_execution, log_system_status,
    log_api_call, log_performance_metrics
)


class TestStructuredJSONFormatter:
    """Test JSON formatter functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.formatter = StructuredJSONFormatter()

    def test_format_basic_log_record(self):
        """Test formatting of basic log record."""
        # Create a mock log record
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Test message',
            args=(),
            exc_info=None
        )
        record.created = 1640995200.0  # 2022-01-01 00:00:00 UTC

        result = self.formatter.format(record)
        parsed = json.loads(result)

        assert parsed['timestamp'] == '2022-01-01T00:00:00'
        assert parsed['level'] == 'INFO'
        assert parsed['logger'] == 'test_logger'
        assert parsed['message'] == 'Test message'
        assert parsed['module'] == 'test'
        assert parsed['function'] == '<unknown>'
        assert parsed['line'] == 10

    def test_format_log_record_with_exception(self):
        """Test formatting of log record with exception info."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name='test_logger',
            level=logging.ERROR,
            pathname='test.py',
            lineno=15,
            msg='Error occurred',
            args=(),
            exc_info=exc_info
        )

        result = self.formatter.format(record)
        parsed = json.loads(result)

        assert parsed['level'] == 'ERROR'
        assert parsed['message'] == 'Error occurred'
        assert 'exception' in parsed
        assert 'ValueError' in parsed['exception']

    def test_format_log_record_with_extra_fields(self):
        """Test formatting of log record with extra fields."""
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='test.py',
            lineno=20,
            msg='Message with extra data',
            args=(),
            exc_info=None
        )

        # Add extra fields
        record.custom_field = 'custom_value'
        record.numeric_field = 42

        result = self.formatter.format(record)
        parsed = json.loads(result)

        assert parsed['custom_field'] == 'custom_value'
        assert parsed['numeric_field'] == 42


class TestTradingLogger:
    """Test TradingLogger class functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test.log')
        self.error_file = os.path.join(self.temp_dir, 'errors.log')

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.logging_config.LOG_FILE', new_callable=lambda: property(lambda self: '/tmp/test.log'))
    @patch('src.logging_config.LOG_LEVEL', 'INFO')
    def test_trading_logger_initialization(self, mock_log_file, mock_log_level):
        """Test TradingLogger initialization."""
        with patch('pathlib.Path.mkdir'), \
             patch('logging.handlers.RotatingFileHandler'), \
             patch('logging.StreamHandler'):

            logger_instance = TradingLogger()

            assert logger_instance.logger is not None
            assert logger_instance.logger.name == 'tradpal_indicator'

    def test_log_trading_signal(self):
        """Test logging of trading signals."""
        with patch.object(trading_logger.logger, 'info') as mock_info:
            signal_data = {
                'signal_type': 'BUY',
                'symbol': 'EUR/USD',
                'price': 1.0500,
                'rsi': 25.5,
                'ema9': 1.0480,
                'ema21': 1.0490,
                'bb_upper': 1.0520,
                'bb_lower': 1.0480,
                'atr': 0.0020,
                'position_size': 1000,
                'stop_loss': 1.0450,
                'take_profit': 1.0600,
                'leverage': 10
            }

            trading_logger.log_trading_signal(signal_data)

            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert call_args[0][0] == "Trading signal generated"
            assert 'signal_type' in call_args[1]['extra']
            assert call_args[1]['extra']['signal_type'] == 'BUY'

    def test_log_system_status(self):
        """Test logging of system status."""
        with patch.object(trading_logger.logger, 'info') as mock_info:
            trading_logger.log_system_status("System initialized", version="1.0.0")

            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert "System status: System initialized" in call_args[0][0]
            assert call_args[1]['extra']['version'] == "1.0.0"

    def test_log_error_with_exception(self):
        """Test logging of errors with exception info."""
        with patch.object(trading_logger.logger, 'error') as mock_error:
            try:
                raise ValueError("Test error")
            except ValueError:
                trading_logger.log_error("An error occurred", exc_info=sys.exc_info())

            mock_error.assert_called_once()
            call_args = mock_error.call_args
            assert "Error: An error occurred" in call_args[0][0]
            assert call_args[1]['exc_info'] is not None

    def test_log_error_without_exception(self):
        """Test logging of errors without exception info."""
        with patch.object(trading_logger.logger, 'error') as mock_error:
            trading_logger.log_error("Simple error message")

            mock_error.assert_called_once()
            call_args = mock_error.call_args
            assert call_args[0][0] == "Error: Simple error message"
            assert call_args[1]['exc_info'] is None

    def test_log_performance_metrics(self):
        """Test logging of performance metrics."""
        with patch.object(trading_logger.logger, 'info') as mock_info:
            metrics = {
                'total_trades': 50,
                'win_rate': 0.65,
                'total_pnl': 1250.50,
                'max_drawdown': 150.25,
                'sharpe_ratio': 1.8
            }

            trading_logger.log_performance_metrics(metrics)

            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert call_args[0][0] == "Performance metrics calculated"
            assert call_args[1]['extra']['metrics'] == metrics

    def test_log_api_call_success(self):
        """Test logging of successful API calls."""
        with patch.object(trading_logger.logger, 'log') as mock_log:
            trading_logger.log_api_call("test_endpoint", True, 0.5, param1="value1")

            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == logging.INFO
            assert "API call to test_endpoint" in call_args[0][1]
            assert call_args[1]['extra']['api_call']['success'] is True
            assert call_args[1]['extra']['api_call']['duration'] == 0.5

    def test_log_api_call_failure(self):
        """Test logging of failed API calls."""
        with patch.object(trading_logger.logger, 'log') as mock_log:
            trading_logger.log_api_call("failing_endpoint", False, 2.0, error="timeout")

            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == logging.WARNING
            assert call_args[1]['extra']['api_call']['success'] is False

    def test_log_cache_operation(self):
        """Test logging of cache operations."""
        with patch.object(trading_logger.logger, 'debug') as mock_debug:
            trading_logger.log_cache_operation("get", "test_key", hit=True, size=1024)

            mock_debug.assert_called_once()
            call_args = mock_debug.call_args
            assert "Cache get: test_key" in call_args[0][0]
            assert call_args[1]['extra']['cache_hit'] is True
            assert call_args[1]['extra']['cache_key'] == "test_key"


class TestGlobalLoggingFunctions:
    """Test global logging convenience functions."""

    def test_log_signal_function(self):
        """Test log_signal convenience function."""
        with patch.object(trading_logger, 'log_trading_signal') as mock_log:
            log_signal(
                signal_type="BUY",
                price=1.0500,
                rsi=25.5,
                ema9=1.0480,
                ema21=1.0490,
                position_size_pct=1.0,
                stop_loss=1.0450,
                take_profit=1.0600,
                leverage=10
            )

            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert call_args['signal_type'] == "BUY"
            assert call_args['price'] == 1.0500

    def test_log_error_function(self):
        """Test log_error convenience function."""
        with patch.object(trading_logger, 'log_error') as mock_log:
            log_error("Test error message")

            mock_log.assert_called_once_with("Test error message", exc_info=None)

    def test_log_trade_execution_function(self):
        """Test log_trade_execution convenience function."""
        with patch.object(logger, 'info') as mock_info:
            trade_details = {"symbol": "EUR/USD", "side": "BUY", "quantity": 1000}
            log_trade_execution(trade_details)

            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "TRADE EXECUTED:" in call_args

    def test_log_system_status_function(self):
        """Test log_system_status convenience function."""
        with patch.object(trading_logger, 'log_system_status') as mock_log:
            log_system_status("System ready")

            mock_log.assert_called_once_with("System ready")

    def test_log_api_call_function(self):
        """Test log_api_call convenience function."""
        with patch.object(trading_logger, 'log_api_call') as mock_log:
            log_api_call("test_endpoint", params={"key": "value"}, response_time=1.5)

            mock_log.assert_called_once_with("test_endpoint", True, 1.5, params={"key": "value"})

    def test_log_performance_metrics_function(self):
        """Test log_performance_metrics convenience function."""
        with patch.object(trading_logger, 'log_performance_metrics') as mock_log:
            metrics = {"win_rate": 0.75, "total_pnl": 500}
            log_performance_metrics(metrics)

            mock_log.assert_called_once_with(metrics)


class TestLoggerSetup:
    """Test logger setup and configuration."""

    def test_logger_instance_creation(self):
        """Test that logger instance is created."""
        assert trading_logger is not None
        assert isinstance(trading_logger, TradingLogger)
        assert logger is not None
        assert logger.name == 'tradpal_indicator'

    @patch('pathlib.Path.mkdir')
    @patch('logging.handlers.RotatingFileHandler')
    @patch('logging.StreamHandler')
    def test_setup_logging_creates_handlers(self, mock_stream_handler, mock_file_handler, mock_mkdir):
        """Test that setup_logging creates appropriate handlers."""
        logger_instance = TradingLogger()

        # Verify that handlers were created
        assert mock_file_handler.called
        assert mock_stream_handler.called

        # Verify that handlers were added to logger
        logger_instance.logger.addHandler.assert_called()

    def test_logger_level_configuration(self):
        """Test logger level configuration."""
        # This should work without errors
        assert logger.level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]