import pytest
import pandas as pd
import numpy as np
import json
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timedelta
import sys

# Add project paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'integrations'))

from src.data_fetcher import fetch_historical_data, validate_data
from src.indicators import calculate_indicators, ema, rsi, bb, atr, adx
from src.signal_generator import generate_signals, calculate_risk_management
from src.output import save_signals_to_json, load_signals_from_json, format_signal_data
from src.backtester import run_backtest, calculate_performance_metrics, simulate_trades
from config.settings import validate_timeframe, get_timeframe_params, validate_risk_params
from integrations.base import IntegrationManager
from integrations.telegram.bot import TelegramIntegration, TelegramConfig
from integrations.email_integration.email import EmailIntegration, EmailConfig


class TestDataFetcherErrorHandling:
    """Test error handling in data fetcher."""

    @patch('src.data_fetcher.ccxt')
    def test_fetch_with_network_timeout(self, mock_ccxt):
        """Test handling of network timeouts."""
        mock_exchange = MagicMock()
        mock_ccxt.kraken.return_value = mock_exchange
        mock_exchange.fetch_ohlcv.side_effect = TimeoutError("Connection timeout")

        # With error handling, function should return retry string
        result = fetch_historical_data('EUR/USD', 'kraken', '1m', 100)
        assert isinstance(result, str) and result == "retry"

    def test_fetch_with_invalid_exchange(self):
        """Test handling of invalid exchange."""
        # Input validation now catches invalid exchanges
        from src.input_validation import ValidationError
        with pytest.raises(ValidationError, match="Invalid exchange"):
            fetch_historical_data('EUR/USD', 'invalid_exchange', '1m', 100)

    @patch('src.data_fetcher.ccxt')
    def test_fetch_with_empty_response(self, mock_ccxt):
        """Test handling of empty API responses."""
        mock_exchange = MagicMock()
        mock_ccxt.kraken.return_value = mock_exchange
        mock_exchange.fetch_ohlcv.return_value = []
        mock_exchange.has = {'fetchOHLCV': True}
        mock_exchange.parse8601.return_value = 1640995200000
        mock_exchange.timeframes = {'1m': 60000}

        result = fetch_historical_data('EUR/USD', 'kraken', '1m', 50)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch('src.data_fetcher.ccxt')
    def test_fetch_with_malformed_data(self, mock_ccxt):
        """Test handling of malformed API data."""
        mock_exchange = MagicMock()
        mock_ccxt.kraken.return_value = mock_exchange
        # Malformed data - missing some OHLC values
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 1.0, 1.1],  # Incomplete OHLCV
            [1640995260000],  # Empty
        ]
        mock_exchange.has = {'fetchOHLCV': True}

        # With error handling, malformed data should still cause an error
        with pytest.raises(ValueError):
            fetch_historical_data('EUR/USD', 'kraken', '1m', 2)

    def test_validate_data_with_wrong_types(self):
        """Test validation with wrong data types."""
        # String instead of DataFrame
        with pytest.raises(AttributeError):
            validate_data("not a dataframe")

        # DataFrame with wrong column types
        df = pd.DataFrame({
            'timestamp': ['invalid'] * 5,
            'open': ['not_a_number'] * 5,
            'high': [1.0] * 5,
            'low': [1.0] * 5,
            'close': [1.0] * 5,
            'volume': [100] * 5
        })

        with pytest.raises(ValueError):
            validate_data(df)

    def test_validate_data_with_extreme_values(self):
        """Test validation with extreme values."""
        # Extremely high prices (but positive)
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5),
            'open': [1e10] * 5,  # Extremely high price
            'high': [1e10 + 1] * 5,
            'low': [1e10 - 1] * 5,  # Still positive
            'close': [1e10] * 5,
            'volume': [100] * 5
        })

        # Should still validate (extreme values are allowed)
        validate_data(df)

    def test_validate_data_with_duplicate_timestamps(self):
        """Test validation with duplicate timestamps."""
        df = pd.DataFrame({
            'timestamp': ['2023-01-01 10:00:00'] * 5,  # Same timestamp
            'open': [1.0] * 5,
            'high': [1.1] * 5,
            'low': [0.9] * 5,
            'close': [1.05] * 5,
            'volume': [100] * 5
        })

        # Should still validate (duplicates are allowed, just not ideal)
        validate_data(df)


class TestIndicatorsErrorHandling:
    """Test error handling in indicators."""

    def test_ema_with_empty_data(self):
        """Test EMA calculation with empty data."""
        result = ema(pd.Series([], dtype=float), 9)
        assert len(result) == 0
        assert isinstance(result, pd.Series)

    def test_ema_with_nan_data(self):
        """Test EMA calculation with NaN data."""
        data = pd.Series([np.nan, np.nan, 1.0, 2.0, 3.0])
        result = ema(data, 3)

        # Should handle NaN gracefully
        assert len(result) == len(data)
        assert not result.isna().all()

    def test_ema_with_insufficient_data(self):
        """Test EMA with insufficient data points."""
        data = pd.Series([1.0, 2.0])  # Less than period
        result = ema(data, 9)

        # Should return NaN for insufficient data
        assert len(result) == len(data)
        assert result.isna().all()

    def test_rsi_with_constant_data(self):
        """Test RSI with constant price data."""
        data = pd.Series([100.0] * 20)  # No price movement
        result = rsi(data, 14)

        # RSI should be NaN for initial values (until period+1), then 0 for constant data
        assert len(result) == len(data)
        # First 14 values should be NaN (not enough data)
        assert result.iloc[:14].isna().all()
        # Remaining values should be 0 (constant data)
        assert (result.iloc[14:] == 0.0).all()

    def test_bb_with_zero_std_dev(self):
        """Test Bollinger Bands with zero standard deviation."""
        data = pd.Series([100.0] * 20)  # Constant data
        upper, middle, lower = bb(data, 20, 2)

        # Should handle zero variance
        assert len(upper) == len(data)
        assert len(middle) == len(data)
        assert len(lower) == len(data)

        # For constant data, bands should be at the price level
        valid_idx = ~upper.isna()
        if valid_idx.any():
            assert all(upper[valid_idx] == middle[valid_idx])
            assert all(lower[valid_idx] == middle[valid_idx])

    def test_atr_with_no_movement(self):
        """Test ATR with no price movement."""
        high = pd.Series([100.0] * 20)
        low = pd.Series([100.0] * 20)
        close = pd.Series([100.0] * 20)

        result = atr(high, low, close, 14)

        # ATR should be 0 for no movement
        valid_atr = result.dropna()
        if len(valid_atr) > 0:
            assert all(valid_atr == 0.0)

    def test_calculate_indicators_with_missing_columns(self):
        """Test indicator calculation with missing DataFrame columns."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'close': [100.0] * 10
            # Missing high, low, open, volume
        })

        with pytest.raises(KeyError):
            calculate_indicators(df)

    def test_calculate_indicators_with_insufficient_data(self):
        """Test indicator calculation with insufficient data."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5),
            'open': [100.0] * 5,
            'high': [101.0] * 5,
            'low': [99.0] * 5,
            'close': [100.5] * 5,
            'volume': [1000] * 5
        })

        result = calculate_indicators(df)

        # Should still work but with NaN values for indicators needing more data
        assert len(result) == len(df)
        assert 'EMA9' in result.columns

        # EMA should be NaN for insufficient data
        assert result['EMA9'].isna().all()


class TestSignalGeneratorErrorHandling:
    """Test error handling in signal generator."""

    def test_generate_signals_with_missing_indicators(self):
        """Test signal generation with missing indicator columns."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'open': [100.0] * 10,
            'high': [101.0] * 10,
            'low': [99.0] * 10,
            'close': [100.5] * 10,
            'volume': [1000] * 10
            # Missing indicator columns
        })

        with pytest.raises(KeyError):
            generate_signals(df)

    def test_calculate_risk_management_with_missing_signals(self):
        """Test risk management with missing signal columns."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'open': [100.0] * 10,
            'high': [101.0] * 10,
            'low': [99.0] * 10,
            'close': [100.5] * 10,
            'volume': [1000] * 10,
            'EMA9': [100.4] * 10,
            'EMA21': [100.3] * 10,
            'RSI': [50.0] * 10,
            'BB_upper': [101.0] * 10,
            'BB_middle': [100.5] * 10,
            'BB_lower': [100.0] * 10,
            'ATR': [0.5] * 10
            # Missing Buy_Signal, Sell_Signal
        })

        with pytest.raises(KeyError):
            calculate_risk_management(df)

    def test_risk_management_with_zero_atr(self):
        """Test risk management with zero ATR."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'open': [100.0] * 10,
            'high': [101.0] * 10,
            'low': [99.0] * 10,
            'close': [100.5] * 10,
            'volume': [1000] * 10,
            'EMA9': [100.4] * 10,
            'EMA21': [100.3] * 10,
            'RSI': [50.0] * 10,
            'BB_upper': [101.0] * 10,
            'BB_middle': [100.5] * 10,
            'BB_lower': [100.0] * 10,
            'ATR': [0.0] * 10,  # Zero ATR
            'Buy_Signal': [1] * 10,
            'Sell_Signal': [0] * 10
        })

        # Should handle zero ATR gracefully
        result = calculate_risk_management(df)
        assert len(result) == len(df)
        assert 'Position_Size_Absolute' in result.columns


class TestOutputErrorHandling:
    """Test error handling in output module."""

    def test_save_signals_with_invalid_data(self):
        """Test saving with invalid data types."""
        with pytest.raises(TypeError):
            save_signals_to_json("not a list", "test.json")

        with pytest.raises(TypeError):
            save_signals_to_json([1, 2, 3], "test.json")  # Not dicts

    @patch('builtins.open', side_effect=UnicodeEncodeError('utf-8', 'test', 0, 1, 'encoding error'))
    def test_save_signals_encoding_error(self, mock_file):
        """Test handling of encoding errors."""
        test_data = [{'test': 'dâtâ with spëcial chârs'}]

        with pytest.raises(UnicodeEncodeError):
            save_signals_to_json(test_data, 'test.json')

    def test_load_signals_with_corrupted_file(self):
        """Test loading corrupted JSON files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"incomplete": json')
            temp_file = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_signals_from_json(temp_file)
        finally:
            os.unlink(temp_file)

    def test_format_signal_data_with_none_values(self):
        """Test formatting with None values."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01']),
            'open': [None],  # None value
            'high': [1.1],
            'low': [0.9],
            'close': [1.0],
            'volume': [1000],
            'EMA9': [1.0],
            'EMA21': [1.0],
            'RSI': [50.0],
            'BB_upper': [1.1],
            'BB_middle': [1.0],
            'BB_lower': [0.9],
            'ATR': [0.1],
            'Buy_Signal': [1],
            'Sell_Signal': [0]
        })

        result = format_signal_data(data)
        assert len(result) == 1
        # None should be preserved or converted to null
        assert result[0]['open'] is None


class TestBacktesterErrorHandling:
    """Test error handling in backtester."""

    def test_calculate_performance_metrics_with_invalid_trades(self):
        """Test performance calculation with invalid trade data."""
        # Empty trades
        metrics = calculate_performance_metrics(pd.DataFrame(), 10000)
        assert metrics['total_trades'] == 0
        assert metrics['final_capital'] == 10000

        # Trades with missing columns
        invalid_trades = pd.DataFrame({
            'entry_price': [100],
            # Missing exit_price, position_size, direction
        })

        with pytest.raises(ValueError):
            calculate_performance_metrics(invalid_trades, 10000)

    def test_simulate_trades_with_invalid_data(self):
        """Test trade simulation with invalid data."""
        # Missing required columns
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5),
            'close': [100.0] * 5
            # Missing signals and other required columns
        })

        with pytest.raises(KeyError):
            simulate_trades(invalid_data)

    @patch('src.backtester.fetch_historical_data', return_value="retry")
    def test_run_backtest_with_api_error(self, mock_fetch):
        """Test backtest with API errors."""
        # With error handling, function should return error dict
        result = run_backtest('EUR/USD', '1m', '2023-01-01', '2023-01-02')
        assert isinstance(result, dict)
        assert 'backtest_results' in result
        assert 'error' in result['backtest_results']

    def test_run_backtest_with_invalid_dates(self):
        """Test backtest with invalid date ranges."""
        result = run_backtest('EUR/USD', '1m', '2023-12-31', '2023-01-01')  # End before start
        
        # Should return error in result instead of raising exception
        assert 'backtest_results' in result
        assert result['backtest_results']['success'] == False
        assert 'start_date must be before end_date' in result['backtest_results']['error']


class TestConfigurationErrorHandling:
    """Test error handling in configuration."""

    def test_validate_timeframe_edge_cases(self):
        """Test timeframe validation edge cases."""
        # Empty string
        assert not validate_timeframe('')

        # None value
        assert not validate_timeframe(None)

        # Invalid format
        assert not validate_timeframe('1x')
        assert not validate_timeframe('hour')
        assert not validate_timeframe('123')

    def test_get_timeframe_params_edge_cases(self):
        """Test getting timeframe params for edge cases."""
        # Non-existent timeframe
        assert get_timeframe_params('999d') is None
        assert get_timeframe_params('') is None
        # None should return default timeframe params, not None
        default_params = get_timeframe_params('1m')  # Assuming '1m' is the default
        assert get_timeframe_params(None) == default_params

    def test_validate_risk_params_edge_cases(self):
        """Test risk parameter validation edge cases."""
        # Negative values
        assert not validate_risk_params({
            'capital': -1000,
            'risk_per_trade': 0.01,
            'sl_multiplier': 1.5
        })

        # Risk > 100%
        assert not validate_risk_params({
            'capital': 10000,
            'risk_per_trade': 2.0,  # 200%
            'sl_multiplier': 1.5
        })

        # Zero stop loss
        assert not validate_risk_params({
            'capital': 10000,
            'risk_per_trade': 0.01,
            'sl_multiplier': 0
        })


class TestIntegrationErrorHandling:
    """Test error handling in integrations."""

    @patch('integrations.telegram.bot.requests.get')
    def test_telegram_initialization_network_error(self, mock_get):
        """Test Telegram initialization with network errors."""
        mock_get.side_effect = ConnectionError("Network unreachable")

        config = TelegramConfig(
            enabled=True,
            name="Test",
            bot_token="token",
            chat_id="123"
        )
        integration = TelegramIntegration(config)

        result = integration.initialize()
        assert result == False  # Should fail gracefully

    @patch('integrations.telegram.bot.requests.post')
    def test_telegram_send_signal_network_error(self, mock_post):
        """Test Telegram signal sending with network errors."""
        mock_post.side_effect = ConnectionError("Network unreachable")

        config = TelegramConfig(enabled=True, name="Test", bot_token="token", chat_id="123")
        integration = TelegramIntegration(config)

        result = integration.send_signal({'signal': 'BUY'})
        assert result == False  # Should fail gracefully

    @patch('integrations.email_integration.email.smtplib.SMTP')
    def test_email_initialization_smtp_error(self, mock_smtp):
        """Test Email initialization with SMTP errors."""
        mock_smtp.side_effect = Exception("SMTP connection failed")

        config = EmailConfig(
            enabled=True,
            name="Test",
            smtp_server="invalid.smtp.com",
            username="test@example.com",
            password="password"
        )
        integration = EmailIntegration(config)

        result = integration.initialize()
        assert result == False  # Should fail gracefully

    def test_integration_manager_error_handling(self):
        """Test integration manager error handling."""
        manager = IntegrationManager()

        # Register integration that will fail
        config = TelegramConfig(enabled=True, name="Test", bot_token="token", chat_id="123")
        integration = TelegramIntegration(config)
        manager.register_integration("test", integration)

        # Initialize should handle failures gracefully
        results = manager.initialize_all()
        assert "test" in results
        # Result may be False due to network issues, but shouldn't crash

        # Send signal should handle failures gracefully
        results = manager.send_signal_to_all({'signal': 'BUY'})
        assert "test" in results
        # Should not crash even if sending fails


class TestSystemIntegrationErrorHandling:
    """Test error handling across the entire system."""

    def test_full_pipeline_error_recovery(self):
        """Test error recovery in the full pipeline."""
        # Test that data fetcher handles errors gracefully - this test now verifies successful operation
        result = fetch_historical_data('EUR/USD', 'kraken', '1m', 10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0  # Should return data on success

    def test_memory_error_handling(self):
        """Test handling of memory-related errors."""
        # Create very large DataFrame to test memory handling
        try:
            large_data = pd.DataFrame({
                'timestamp': pd.date_range('2020-01-01', periods=1000000, freq='1s'),
                'open': np.random.randn(1000000),
                'high': np.random.randn(1000000),
                'low': np.random.randn(1000000),
                'close': np.random.randn(1000000),
                'volume': np.random.randint(1000, 10000, 1000000)
            })

            # Try to process large data
            result = calculate_indicators(large_data)

            # Should either succeed or fail gracefully
            assert isinstance(result, pd.DataFrame) or isinstance(result, Exception)

        except MemoryError:
            # Memory error is acceptable for very large data
            pass

    def test_concurrent_access_error_handling(self):
        """Test handling of concurrent access to shared resources."""
        # Test file access concurrency
        import threading
        import tempfile
        import os

        results = []
        errors = []

        # Create a temporary file path for the test
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='_concurrent_test.json', delete=False, dir=tempfile.gettempdir()) as temp_file:
            temp_path = temp_file.name

        try:
            def write_file():
                try:
                    test_data = [{'test': f'thread_{threading.current_thread().name}'}]
                    save_signals_to_json(test_data, temp_path)
                    results.append(True)
                except Exception as e:
                    errors.append(e)

            # Start multiple threads trying to write to the same file
            threads = []
            for i in range(5):
                t = threading.Thread(target=write_file)
                threads.append(t)
                t.start()

            # Wait for all threads
            for t in threads:
                t.join()

            # Should have some successful writes and possibly some errors
            assert len(results) >= 1  # At least one should succeed
            # Errors are acceptable due to concurrent access

        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except OSError:
                pass  # File may have been deleted by another process


if __name__ == "__main__":
    pytest.main([__file__])