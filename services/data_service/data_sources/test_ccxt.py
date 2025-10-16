"""
Unit Tests for CCXT Data Source

This module contains comprehensive unit tests for the CCXTDataSource class.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call
import logging

from .ccxt_source import CCXTDataSource


class TestCCXTDataSource:
    """Test cases for CCXTDataSource."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {'exchange': 'binance', 'timeout': 30000, 'enableRateLimit': True}

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_initialization_success(self, mock_ccxt_module):
        """Test successful initialization."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}

        source = CCXTDataSource(self.config)

        assert source.name == "CCXT"
        assert source.config['exchange'] == 'binance'
        assert source.config['timeout'] == 30000
        assert source.config['enableRateLimit'] is True

        # Verify exchange was initialized
        mock_exchange_class.assert_called_once_with({
            'timeout': 30000,
            'enableRateLimit': True
        })
        mock_exchange_instance.load_markets.assert_called_once()

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_initialization_default_config(self, mock_ccxt_module):
        """Test initialization with default config."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}

        source = CCXTDataSource()

        assert source.config['exchange'] == 'binance'  # Default
        assert source.config['timeout'] == 30000
        assert source.config['enableRateLimit'] is True

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', False)
    def test_initialization_ccxt_not_available(self):
        """Test initialization fails when CCXT not available."""
        with pytest.raises(ImportError, match="ccxt library is required"):
            CCXTDataSource()

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_exchange_initialization_failure(self, mock_ccxt_module):
        """Test handling of exchange initialization failure."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.side_effect = Exception("Exchange error")

        with pytest.raises(Exception, match="Exchange error"):
            CCXTDataSource(self.config)

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_timeframe_to_minutes(self, mock_ccxt_module):
        """Test timeframe to minutes conversion."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {}

        source = CCXTDataSource()

        assert source._timeframe_to_minutes('1m') == 1
        assert source._timeframe_to_minutes('5m') == 5
        assert source._timeframe_to_minutes('1h') == 60
        assert source._timeframe_to_minutes('1d') == 1440
        assert source._timeframe_to_minutes('1w') == 10080
        assert source._timeframe_to_minutes('1M') == 43200

        # Test default
        assert source._timeframe_to_minutes('unknown') == 60  # 1h default

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_ohlcv_to_dataframe(self, mock_ccxt_module):
        """Test conversion of OHLCV list to DataFrame."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {}

        source = CCXTDataSource()

        # Test with valid data
        ohlcv_data = [
            [1640995200000, 100.0, 105.0, 95.0, 103.0, 1000.0],  # 2022-01-01 00:00:00
            [1641081600000, 103.0, 108.0, 98.0, 106.0, 1100.0],  # 2022-01-02 00:00:00
        ]

        result = source._ohlcv_to_dataframe(ohlcv_data)

        assert not result.empty
        assert len(result) == 2
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index[0] == pd.Timestamp('2022-01-01')
        assert result.iloc[0]['open'] == 100.0
        assert result.iloc[0]['high'] == 105.0
        assert result.iloc[0]['low'] == 95.0
        assert result.iloc[0]['close'] == 103.0
        assert result.iloc[0]['volume'] == 1000.0

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_ohlcv_to_dataframe_empty(self, mock_ccxt_module):
        """Test conversion with empty OHLCV data."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {}

        source = CCXTDataSource()

        result = source._ohlcv_to_dataframe([])
        assert result.empty
        assert isinstance(result, pd.DataFrame)

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_fetch_historical_data_success(self, mock_ccxt_module, caplog):
        """Test successful historical data fetching."""
        caplog.set_level(logging.INFO)

        # Mock exchange setup
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}

        # Mock OHLCV data
        ohlcv_data = [
            [1640995200000, 100.0, 105.0, 95.0, 103.0, 1000.0],
            [1641081600000, 103.0, 108.0, 98.0, 106.0, 1100.0],
        ]
        mock_exchange_instance.fetch_ohlcv.return_value = ohlcv_data

        source = CCXTDataSource(self.config)
        result = source.fetch_historical_data(
            symbol='BTC/USDT',
            timeframe='1d',
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2022, 1, 2)
        )

        # Verify result
        assert not result.empty
        assert len(result) == 2
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])

        # Verify API call
        mock_exchange_instance.fetch_ohlcv.assert_called_once_with(
            symbol='BTC/USDT',
            timeframe='1d',
            since=1640995200000,  # 2022-01-01 timestamp
            limit=1000
        )

        # Check logging
        assert "Fetching BTC/USDT 1d data" in caplog.text
        assert "Successfully fetched 2 candles" in caplog.text

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_fetch_historical_data_with_limit(self, mock_ccxt_module):
        """Test fetching with custom limit."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}

        ohlcv_data = [[1640995200000, 100.0, 105.0, 95.0, 103.0, 1000.0]]
        mock_exchange_instance.fetch_ohlcv.return_value = ohlcv_data

        source = CCXTDataSource()
        result = source.fetch_historical_data(
            symbol='BTC/USDT',
            timeframe='1d',
            limit=500
        )

        mock_exchange_instance.fetch_ohlcv.assert_called_once_with(
            symbol='BTC/USDT',
            timeframe='1d',
            since=mock_exchange_instance.fetch_ohlcv.call_args[1]['since'],
            limit=500
        )

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_fetch_historical_data_empty_response(self, mock_ccxt_module, caplog):
        """Test handling of empty data response."""
        caplog.set_level(logging.WARNING)

        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}
        mock_exchange_instance.fetch_ohlcv.return_value = []  # Empty response

        source = CCXTDataSource()
        result = source.fetch_historical_data('BTC/USDT', '1d')

        assert result.empty
        assert "No data received for BTC/USDT" in caplog.text

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_fetch_historical_data_api_error(self, mock_ccxt_module, caplog):
        """Test handling of API errors."""
        caplog.set_level(logging.ERROR)

        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}
        mock_exchange_instance.fetch_ohlcv.side_effect = Exception("API Error")

        source = CCXTDataSource()

        with pytest.raises(Exception, match="API Error"):
            source.fetch_historical_data('BTC/USDT', '1d')

        assert "Error fetching data from CCXT: API Error" in caplog.text

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_fetch_recent_data(self, mock_ccxt_module):
        """Test fetching recent data."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}

        ohlcv_data = [[1640995200000, 100.0, 105.0, 95.0, 103.0, 1000.0]]
        mock_exchange_instance.fetch_ohlcv.return_value = ohlcv_data

        source = CCXTDataSource()
        result = source.fetch_recent_data('BTC/USDT', '1d', limit=100)

        assert not result.empty
        assert len(result) == 1

        # Should call fetch_historical_data with limit
        mock_exchange_instance.fetch_ohlcv.assert_called_once()

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_fetch_historical_data_default_dates(self, mock_ccxt_module):
        """Test default date handling."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}

        ohlcv_data = [[1640995200000, 100.0, 105.0, 95.0, 103.0, 1000.0]]
        mock_exchange_instance.fetch_ohlcv.return_value = ohlcv_data

        source = CCXTDataSource()

        with patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.datetime') as mock_datetime:
            fixed_now = datetime(2023, 12, 31, 12, 0, 0)
            mock_datetime.now.return_value = fixed_now
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs) if args else fixed_now

            result = source.fetch_historical_data('BTC/USDT', '1d')

            # Should calculate start_date based on timeframe and limit
            call_args = mock_exchange_instance.fetch_ohlcv.call_args
            since_timestamp = call_args[1]['since']
            expected_start = fixed_now - timedelta(minutes=60 * 1000)  # 1000 candles * 1h * 60min
            expected_since = int(expected_start.timestamp() * 1000)
            assert abs(since_timestamp - expected_since) < 1000  # Allow small difference

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_fetch_historical_data_end_date_filter(self, mock_ccxt_module):
        """Test filtering by end_date."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}

        # Mock data extending beyond end_date
        ohlcv_data = [
            [1640995200000, 100.0, 105.0, 95.0, 103.0, 1000.0],  # 2022-01-01
            [1641081600000, 103.0, 108.0, 98.0, 106.0, 1100.0],  # 2022-01-02
            [1641168000000, 106.0, 111.0, 101.0, 109.0, 1200.0],  # 2022-01-03
        ]
        mock_exchange_instance.fetch_ohlcv.return_value = ohlcv_data

        source = CCXTDataSource()
        result = source.fetch_historical_data(
            symbol='BTC/USDT',
            timeframe='1d',
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2022, 1, 2)  # Should filter out 2022-01-03
        )

        assert len(result) == 2  # Should only include up to end_date

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_data_validation_called(self, mock_ccxt_module):
        """Test that data validation is called."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}

        ohlcv_data = [[1640995200000, 100.0, 105.0, 95.0, 103.0, 1000.0]]
        mock_exchange_instance.fetch_ohlcv.return_value = ohlcv_data

        source = CCXTDataSource()

        with patch.object(source, 'validate_data') as mock_validate:
            source.fetch_historical_data('BTC/USDT', '1d')
            mock_validate.assert_called_once()

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_get_available_symbols(self, mock_ccxt_module):
        """Test getting available symbols."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}, 'ETH/USDT': {}}

        source = CCXTDataSource()
        symbols = source.get_available_symbols()

        assert 'BTC/USDT' in symbols
        assert 'ETH/USDT' in symbols
        assert len(symbols) == 2

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_get_available_symbols_error(self, mock_ccxt_module):
        """Test error handling in get_available_symbols."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}
        mock_exchange_instance.markets = None  # Simulate error

        source = CCXTDataSource()
        symbols = source.get_available_symbols()

        assert symbols == []

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_get_exchange_info(self, mock_ccxt_module):
        """Test getting exchange information."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}

        # Mock exchange attributes
        mock_exchange_instance.name = 'Binance'
        mock_exchange_instance.id = 'binance'
        mock_exchange_instance.countries = ['US', 'CN']
        mock_exchange_instance.urls = {'www': 'https://binance.com'}
        mock_exchange_instance.has = {'fetchOHLCV': True}
        mock_exchange_instance.timeframes = {'1m': '1m', '1d': '1d'}
        mock_exchange_instance.markets = {'BTC/USDT': {}, 'ETH/USDT': {}}

        source = CCXTDataSource()
        info = source.get_exchange_info()

        assert info['name'] == 'Binance'
        assert info['id'] == 'binance'
        assert info['countries'] == ['US', 'CN']
        assert info['markets_count'] == 2

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_get_exchange_info_no_exchange(self, mock_ccxt_module):
        """Test get_exchange_info when exchange is not initialized."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.side_effect = Exception("Init failed")

        source = CCXTDataSource()
        # Force exchange to None by triggering init failure
        source.exchange = None

        info = source.get_exchange_info()
        assert info == {}

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_get_info(self, mock_ccxt_module):
        """Test get_info method."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}

        source = CCXTDataSource(self.config)
        info = source.get_info()

        assert info['name'] == 'CCXT'
        assert 'description' in info
        assert 'supported_timeframes' in info
        assert 'config' in info
        assert info['config']['exchange'] == 'binance'

    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.CCXT_AVAILABLE', True)
    @patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt')
    def test_is_available(self, mock_ccxt_module):
        """Test is_available method."""
        mock_exchange_class = MagicMock()
        mock_ccxt_module.binance = mock_exchange_class
        mock_exchange_instance = MagicMock()
        mock_exchange_class.return_value = mock_exchange_instance
        mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}

        source = CCXTDataSource()
        assert source.is_available() is True