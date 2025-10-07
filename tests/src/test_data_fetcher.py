import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_fetcher import fetch_historical_data, validate_data


class TestDataFetcher:
    """Test data fetching functionality."""

    @patch('src.data_fetcher.ccxt')
    def test_fetch_historical_data_success(self, mock_ccxt):
        """Test successful data fetching."""
        # Mock exchange
        mock_exchange = MagicMock()
        mock_ccxt.kraken.return_value = mock_exchange

        # Mock OHLCV data
        mock_data = [
            [1640995200000, 1.0, 1.1, 0.9, 1.05, 1000],  # timestamp, open, high, low, close, volume
            [1640995260000, 1.05, 1.15, 0.95, 1.1, 1100],
            [1640995320000, 1.1, 1.2, 1.0, 1.15, 1200]
        ]
        mock_exchange.fetch_ohlcv.return_value = mock_data
        mock_exchange.has = {'fetchOHLCV': True}

        # Test the function
        result = fetch_historical_data('EUR/USD', 'kraken', '1m', 3)

        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result['close'].iloc[0] == 1.05
        assert result['close'].iloc[-1] == 1.15

        # Verify the call was made correctly
        mock_exchange.fetch_ohlcv.assert_called_once()
        args, kwargs = mock_exchange.fetch_ohlcv.call_args
        assert args[0] == 'EUR/USD'
        assert args[1] == '1m'

    @patch('src.data_fetcher.ccxt')
    def test_fetch_historical_data_exchange_error(self, mock_ccxt):
        """Test handling of exchange errors."""
        mock_exchange = MagicMock()
        mock_ccxt.kraken.return_value = mock_exchange
        mock_exchange.fetch_ohlcv.side_effect = Exception("Exchange API error")

        with pytest.raises(Exception, match="Exchange API error"):
            fetch_historical_data('EUR/USD', 'kraken', '1m', 100)

    @patch('src.data_fetcher.ccxt')
    def test_fetch_historical_data_invalid_symbol(self, mock_ccxt):
        """Test handling of invalid symbol."""
        mock_exchange = MagicMock()
        mock_ccxt.kraken.return_value = mock_exchange
        mock_exchange.fetch_ohlcv.return_value = []

        result = fetch_historical_data('INVALID/SYMBOL', 'kraken', '1m', 100)

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

    @patch('src.data_fetcher.ccxt')
    def test_fetch_historical_data_different_timeframes(self, mock_ccxt):
        """Test fetching data with different timeframes."""
        mock_exchange = MagicMock()
        mock_ccxt.kraken.return_value = mock_exchange
        mock_exchange.fetch_ohlcv.return_value = [[1640995200000, 1.0, 1.1, 0.9, 1.05, 1000]]
        mock_exchange.has = {'fetchOHLCV': True}

        timeframes = ['1m', '5m', '1h', '1d']

        for timeframe in timeframes:
            result = fetch_historical_data('EUR/USD', 'kraken', timeframe, 1)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1

    @patch('src.data_fetcher.ccxt')
    def test_fetch_historical_data_rate_limiting(self, mock_ccxt):
        """Test handling of rate limiting."""
        mock_exchange = MagicMock()
        mock_ccxt.kraken.return_value = mock_exchange

        # Simulate rate limiting error
        from ccxt import RateLimitExceeded
        mock_exchange.fetch_ohlcv.side_effect = RateLimitExceeded("Rate limit exceeded")

        with pytest.raises(RateLimitExceeded):
            fetch_historical_data('EUR/USD', 'kraken', '1m', 100)

    @patch('src.data_fetcher.ccxt')
    def test_fetch_historical_data_network_error(self, mock_ccxt):
        """Test handling of network errors."""
        mock_exchange = MagicMock()
        mock_ccxt.kraken.return_value = mock_exchange
        mock_exchange.fetch_ohlcv.side_effect = ConnectionError("Network error")

        with pytest.raises(ConnectionError):
            fetch_historical_data('EUR/USD', 'kraken', '1m', 100)


class TestDataFetcherIntegration:
    """Integration tests for data fetching."""

    @patch('src.data_fetcher.ccxt')
    def test_data_pipeline_integration(self, mock_ccxt):
        """Test the complete data fetching pipeline."""
        # Mock successful data fetch
        mock_exchange = MagicMock()
        mock_ccxt.kraken.return_value = mock_exchange

        mock_data = [
            [1640995200000, 1.0, 1.1, 0.9, 1.05, 1000],
            [1640995260000, 1.05, 1.15, 0.95, 1.1, 1100],
            [1640995320000, 1.1, 1.2, 1.0, 1.15, 1200]
        ]
        mock_exchange.fetch_ohlcv.return_value = mock_data
        mock_exchange.has = {'fetchOHLCV': True}

        # Test the pipeline
        data = fetch_historical_data('EUR/USD', 'kraken', '1m', 3)

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


if __name__ == "__main__":
    pytest.main([__file__])