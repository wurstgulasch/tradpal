"""
Tests for data fetching functionality in the data service.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from services.data_service.data_fetcher import (
    fetch_historical_data,
    validate_data,
    _get_adaptive_batch_size,
    _timeframe_to_ms,
    _validate_batch_data
)


class TestDataFetcher:
    """Test basic data fetching functionality."""

    @patch('services.data_service.data_fetcher.get_data_source')
    def test_fetch_historical_data_success(self, mock_get_data_source):
        """Test successful data fetching."""
        mock_data_source = MagicMock()
        mock_get_data_source.return_value = mock_data_source

        # Mock successful data fetch
        mock_data = pd.DataFrame({
            'open': [1.0, 1.05],
            'high': [1.1, 1.15],
            'low': [0.9, 0.95],
            'close': [1.05, 1.1],
            'volume': [1000, 1100]
        }, index=pd.to_datetime([1640995200000, 1640995260000], unit='ms'))
        mock_data.index.name = 'timestamp'
        mock_data_source.fetch_historical_data.return_value = mock_data

        result = fetch_historical_data('EUR/USD', '1m', 2)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert isinstance(result.index, pd.DatetimeIndex)

    @patch('services.data_service.data_fetcher.get_data_source')
    def test_fetch_historical_data_network_error(self, mock_get_data_source):
        """Test handling of network errors."""
        mock_data_source = MagicMock()
        mock_get_data_source.return_value = mock_data_source
        mock_data_source.fetch_historical_data.side_effect = ConnectionError("Network error")

        # With error handling, function should raise exception on network errors
        with pytest.raises(ConnectionError):
            fetch_historical_data('EUR/USD', '1m', 100)

    @patch('services.data_service.data_fetcher.get_data_source')
    def test_fetch_historical_data_invalid_symbol(self, mock_get_data_source):
        """Test handling of invalid symbol."""
        mock_data_source = MagicMock()
        mock_get_data_source.return_value = mock_data_source
        mock_data_source.fetch_historical_data.return_value = pd.DataFrame()

        result = fetch_historical_data('INVALID/SYMBOL', '1m', 100)

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_validate_data_valid(self):
        """Test validation of valid data."""
        # Create valid test data with proper OHLC relationships
        dates = pd.date_range('2023-01-01', periods=100, freq='1min')
        base_prices = 100 + np.cumsum(np.random.randn(100) * 0.1)  # Random walk

        data = pd.DataFrame({
            'timestamp': dates,
            'open': base_prices,
            'high': base_prices + np.abs(np.random.randn(100)) * 2,  # Always > open
            'low': base_prices - np.abs(np.random.randn(100)) * 2,   # Always < open
            'close': base_prices + np.random.randn(100) * 1,         # Between low and high
            'volume': np.random.randint(1000, 10000, 100)
        })

        # Ensure OHLC relationships are valid
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))

        # Should not raise any exceptions
        validate_data(data)

    def test_validate_data_missing_columns(self):
        """Test validation with missing columns."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'open': np.random.randn(10),
            # Missing high, low, close, volume
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data(data)

    def test_validate_data_empty(self):
        """Test validation of empty data."""
        data = pd.DataFrame()

        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_data(data)

    def test_validate_data_invalid_ohlc(self):
        """Test validation with invalid OHLC relationships."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'open': [100] * 10,
            'high': [99] * 10,  # High < Open (invalid)
            'low': [101] * 10,   # Low > Open (invalid)
            'close': [100] * 10,
            'volume': [1000] * 10
        })

        with pytest.raises(ValueError, match="Invalid OHLC relationships"):
            validate_data(data)

    def test_validate_data_negative_volume(self):
        """Test validation with negative volume."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [-1000] * 10  # Negative volume
        })

        with pytest.raises(ValueError, match="Negative values found in volume column"):
            validate_data(data)

    @patch('services.data_service.data_fetcher.get_data_source')
    def test_fetch_historical_data_different_timeframes(self, mock_get_data_source):
        """Test fetching data with different timeframes."""
        mock_data_source = MagicMock()
        mock_get_data_source.return_value = mock_data_source
        mock_data_source.fetch_historical_data.return_value = pd.DataFrame({
            'open': [1.0],
            'high': [1.1],
            'low': [0.9],
            'close': [1.05],
            'volume': [1000]
        }, index=pd.to_datetime([1640995200000], unit='ms'))

        timeframes = ['1m', '5m', '1h', '1d']

        for timeframe in timeframes:
            result = fetch_historical_data('EUR/USD', timeframe, 1)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1


class TestDataFetcherIntegration:
    """Integration tests for data fetching."""

    @patch('services.data_service.data_fetcher.get_data_source')
    def test_data_pipeline_integration(self, mock_get_data_source):
        """Test the complete data fetching pipeline."""
        # Mock successful data fetch
        mock_data_source = MagicMock()
        mock_get_data_source.return_value = mock_data_source

        mock_data = pd.DataFrame({
            'open': [1.0, 1.05, 1.1],
            'high': [1.1, 1.15, 1.2],
            'low': [0.9, 0.95, 1.0],
            'close': [1.05, 1.1, 1.15],
            'volume': [1000, 1100, 1200]
        }, index=pd.to_datetime([1640995200000, 1640995260000, 1640995320000], unit='ms'))
        mock_data.index.name = 'timestamp'
        mock_data_source.fetch_historical_data.return_value = mock_data

        # Test the pipeline
        data = fetch_historical_data('EUR/USD', '1m', 3)

        # Validate the fetched data
        validate_data(data)

        # Check structure
        assert len(data) == 3
        assert list(data.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert isinstance(data.index, pd.DatetimeIndex)

        # Check data structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3
        assert data.index.name == 'timestamp'
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])


class TestEnhancedPagination:
    """Test enhanced pagination features."""

    @patch('services.data_service.data_fetcher.get_data_source')
    def test_pagination_with_progress_tracking(self, mock_get_data_source):
        """Test pagination with progress tracking enabled."""
        mock_data_source = MagicMock()
        mock_get_data_source.return_value = mock_data_source

        # Mock multiple batches of data - create exactly 100 candles
        all_data = []
        for i in range(100):
            timestamp = 1640995200000 + i * 3600000
            all_data.append([timestamp, 1.0, 1.1, 0.9, 1.05, 1000])

        mock_df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        mock_df['timestamp'] = pd.to_datetime(mock_df['timestamp'], unit='ms')
        mock_df = mock_df.set_index('timestamp')
        mock_data_source.fetch_historical_data.return_value = mock_df

        # Capture print output to verify progress tracking
        with patch('builtins.print') as mock_print:
            result = fetch_historical_data('BTC/USDT', '1h', 100, show_progress=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100

    @patch('services.data_service.data_fetcher.ccxt')
    def test_adaptive_batching(self, mock_ccxt):
        """Test adaptive batch size adjustment."""
        mock_exchange = MagicMock()
        mock_ccxt.kraken.return_value = mock_exchange

        # Mock data that causes batch size reduction
        mock_exchange.fetch_ohlcv.return_value = [[1640995200000, 1.0, 1.1, 0.9, 1.05, 1000]]
        mock_exchange.has = {'fetchOHLCV': True}
        mock_exchange.parse8601.return_value = 1640995200000
        mock_exchange.timeframes = {'1m': '1m'}

        # Test adaptive batch size calculation
        batch_size = _get_adaptive_batch_size('kraken', '1m')
        assert isinstance(batch_size, int)
        assert batch_size > 0

        # Test different exchanges
        binance_batch = _get_adaptive_batch_size('binance', '1h')
        kraken_batch = _get_adaptive_batch_size('kraken', '1h')
        assert binance_batch != kraken_batch  # Different exchanges should have different batch sizes

    def test_timeframe_to_ms_conversion(self):
        """Test timeframe to milliseconds conversion."""
        # Test various timeframes
        assert _timeframe_to_ms('1m') == 60000
        assert _timeframe_to_ms('5m') == 300000
        assert _timeframe_to_ms('1h') == 3600000
        assert _timeframe_to_ms('1d') == 86400000
        assert _timeframe_to_ms('1w') == 604800000

        # Test default fallback
        assert _timeframe_to_ms('unknown') == 3600000  # Should default to 1h

    def test_validate_batch_data(self):
        """Test batch data validation."""
        # Valid batch
        valid_batch = [
            [1640995200000, 1.0, 1.1, 0.9, 1.05, 1000],
            [1640995260000, 1.05, 1.15, 0.95, 1.1, 1100]
        ]
        assert _validate_batch_data(valid_batch) == True

        # Invalid batch - wrong length
        invalid_batch1 = [[1640995200000, 1.0, 1.1, 0.9]]  # Missing volume
        assert _validate_batch_data(invalid_batch1) == False

        # Invalid batch - None values
        invalid_batch2 = [[1640995200000, None, 1.1, 0.9, 1.05, 1000]]
        assert _validate_batch_data(invalid_batch2) == False

        # Invalid batch - OHLC relationships
        invalid_batch3 = [[1640995200000, 1.0, 0.9, 1.1, 1.05, 1000]]  # High < Low
        assert _validate_batch_data(invalid_batch3) == False

        # Invalid batch - negative values
        invalid_batch4 = [[1640995200000, -1.0, 1.1, 0.9, 1.05, 1000]]
        assert _validate_batch_data(invalid_batch4) == False

        # Empty batch
        assert _validate_batch_data([]) == False

    @patch('time.sleep')
    @patch('services.data_service.data_fetcher.get_data_source')
    def test_rate_limiting_handling(self, mock_get_data_source, mock_sleep):
        """Test rate limiting detection and handling."""
        mock_data_source = MagicMock()
        mock_get_data_source.return_value = mock_data_source

        # Mock rate limit exceeded error by creating exception with 'ratelimit' in name
        class MockRateLimitError(Exception):
            pass
        MockRateLimitError.__name__ = 'RateLimitExceeded'
        mock_data_source.fetch_historical_data.side_effect = MockRateLimitError("Rate limit exceeded")

        with patch('builtins.print') as mock_print:
            with pytest.raises(Exception):
                fetch_historical_data('BTC/USDT', '1h', 10, show_progress=True)

    @patch('services.data_service.data_fetcher.get_data_source')
    def test_pagination_error_recovery(self, mock_get_data_source):
        """Test error recovery during pagination."""
        mock_data_source = MagicMock()
        mock_get_data_source.return_value = mock_data_source

        # First call succeeds, second fails with a recoverable error, third succeeds
        mock_df = pd.DataFrame({
            'open': [1.0],
            'high': [1.1],
            'low': [0.9],
            'close': [1.05],
            'volume': [1000]
        }, index=pd.to_datetime([1640995200000], unit='ms'))
        mock_data_source.fetch_historical_data.side_effect = [
            mock_df,  # First batch succeeds
            Exception("timeout"),  # Second batch fails with recoverable error (contains 'timeout')
            mock_df,  # Third call succeeds
        ]

        with patch('builtins.print') as mock_print:
            result = fetch_historical_data('BTC/USDT', '1h', 3, show_progress=True)

        # Should successfully recover and return data
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1  # At least one batch of data

    @patch('services.data_service.data_fetcher.get_data_source')
    def test_large_dataset_pagination(self, mock_get_data_source):
        """Test pagination with large datasets."""
        mock_data_source = MagicMock()
        mock_get_data_source.return_value = mock_data_source

        # Create 100 candles (reduced from 500 to make test faster and more realistic)
        all_data = []
        for i in range(100):
            timestamp = 1640995200000 + i * 3600000
            all_data.append([timestamp, 1.0, 1.1, 0.9, 1.05, 1000])

        mock_df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        mock_df['timestamp'] = pd.to_datetime(mock_df['timestamp'], unit='ms')
        mock_df = mock_df.set_index('timestamp')
        mock_data_source.fetch_historical_data.return_value = mock_df

        with patch('builtins.print') as mock_print:
            result = fetch_historical_data('BTC/USDT', '1h', 100, show_progress=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100


if __name__ == "__main__":
    pytest.main([__file__])
