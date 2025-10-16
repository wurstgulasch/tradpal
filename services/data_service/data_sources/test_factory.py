"""
Integration Tests for Data Source Factory

This module contains integration tests for the data source factory
and all implemented data sources.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from .factory import DataSourceFactory, create_data_source


class TestDataSourceFactory:
    """Test cases for DataSourceFactory."""

    def test_create_yahoo_finance_source(self):
        """Test creating Yahoo Finance data source."""
        with patch('tradpal_indicator.services.data_service.data_sources.factory.YAHOO_AVAILABLE', True):
            source = DataSourceFactory.create_data_source('yahoo_finance')
            assert source.name == "Yahoo Finance"
            assert hasattr(source, 'fetch_historical_data')

    def test_create_ccxt_source(self):
        """Test creating CCXT data source."""
        with patch('tradpal_indicator.services.data_service.data_sources.factory.CCXT_AVAILABLE', True):
            source = DataSourceFactory.create_data_source('ccxt')
            assert source.name == "CCXT"
            assert hasattr(source, 'fetch_historical_data')

    def test_create_kaggle_source(self):
        """Test creating Kaggle data source."""
        with patch('tradpal_indicator.services.data_service.data_sources.factory.KAGGLE_AVAILABLE', True):
            source = DataSourceFactory.create_data_source('kaggle')
            assert source.name == "Kaggle"
            assert hasattr(source, 'fetch_historical_data')

    def test_create_unknown_source(self):
        """Test creating unknown data source raises error."""
        with pytest.raises(ValueError, match="Unknown data source 'unknown'"):
            DataSourceFactory.create_data_source('unknown')

    def test_auto_select_source_priority(self):
        """Test auto-selection prioritizes sources correctly."""
        # Mock availability: kaggle > yahoo > ccxt
        with patch('tradpal_indicator.services.data_service.data_sources.factory.KAGGLE_AVAILABLE', True), \
             patch('tradpal_indicator.services.data_service.data_sources.factory.YAHOO_AVAILABLE', True), \
             patch('tradpal_indicator.services.data_service.data_sources.factory.CCXT_AVAILABLE', True):

            source = DataSourceFactory.create_data_source()
            assert source.name == "Kaggle"  # Should select highest priority

    def test_auto_select_fallback(self):
        """Test auto-selection falls back when higher priority sources unavailable."""
        # Only Yahoo available
        with patch('tradpal_indicator.services.data_service.data_sources.factory.KAGGLE_AVAILABLE', False), \
             patch('tradpal_indicator.services.data_service.data_sources.factory.YAHOO_AVAILABLE', True), \
             patch('tradpal_indicator.services.data_service.data_sources.factory.CCXT_AVAILABLE', False):

            source = DataSourceFactory.create_data_source()
            assert source.name == "Yahoo Finance"

    def test_auto_select_no_sources(self):
        """Test auto-selection fails when no sources available."""
        with patch('tradpal_indicator.services.data_service.data_sources.factory.KAGGLE_AVAILABLE', False), \
             patch('tradpal_indicator.services.data_service.data_sources.factory.YAHOO_AVAILABLE', False), \
             patch('tradpal_indicator.services.data_service.data_sources.factory.CCXT_AVAILABLE', False):

            with pytest.raises(ValueError, match="No data sources are available"):
                DataSourceFactory.create_data_source()

    def test_get_available_sources(self):
        """Test getting availability status of all sources."""
        with patch('tradpal_indicator.services.data_service.data_sources.factory.KAGGLE_AVAILABLE', True), \
             patch('tradpal_indicator.services.data_service.data_sources.factory.YAHOO_AVAILABLE', False), \
             patch('tradpal_indicator.services.data_service.data_sources.factory.CCXT_AVAILABLE', True):

            available = DataSourceFactory.get_available_sources()

            assert available['kaggle'] is True
            assert available['yahoo_finance'] is False
            assert available['ccxt'] is True

    def test_list_sources(self):
        """Test listing available source names."""
        with patch('tradpal_indicator.services.data_service.data_sources.factory.KAGGLE_AVAILABLE', True), \
             patch('tradpal_indicator.services.data_service.data_sources.factory.YAHOO_AVAILABLE', False), \
             patch('tradpal_indicator.services.data_service.data_sources.factory.CCXT_AVAILABLE', True):

            sources = DataSourceFactory.list_sources()

            assert 'kaggle' in sources
            assert 'yahoo_finance' not in sources
            assert 'ccxt' in sources


class TestDataSourceIntegration:
    """Integration tests for data source functionality."""

    @patch('tradpal_indicator.services.data_service.data_sources.factory.YAHOO_AVAILABLE', True)
    def test_yahoo_finance_integration(self, mock_yahoo):
        """Test Yahoo Finance data source integration."""
        with patch('tradpal_indicator.services.data_service.data_sources.yahoo_finance.yf.Ticker') as mock_ticker:
            # Mock ticker response
            mock_ticker_instance = MagicMock()
            mock_ticker.return_value = mock_ticker_instance

            # Create mock data
            dates = pd.date_range('2023-01-01', periods=5, freq='1d')
            mock_df = pd.DataFrame({
                'Open': [100, 101, 102, 103, 104],
                'High': [105, 106, 107, 108, 109],
                'Low': [95, 96, 97, 98, 99],
                'Close': [103, 104, 105, 106, 107],
                'Volume': [1000, 1100, 1200, 1300, 1400]
            }, index=dates)

            mock_ticker_instance.history.return_value = mock_df

            source = create_data_source('yahoo_finance')
            result = source.fetch_historical_data(
                symbol='BTC-USD',
                timeframe='1d',
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 5)
            )

            assert not result.empty
            assert len(result) == 5
            assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    @patch('tradpal_indicator.services.data_service.data_sources.factory.CCXT_AVAILABLE', True)
    def test_ccxt_integration(self, mock_ccxt):
        """Test CCXT data source integration."""
        with patch('tradpal_indicator.services.data_service.data_sources.ccxt_source.ccxt') as mock_ccxt_module:
            # Mock exchange
            mock_exchange_class = MagicMock()
            mock_ccxt_module.binance = mock_exchange_class

            mock_exchange_instance = MagicMock()
            mock_exchange_class.return_value = mock_exchange_instance
            mock_exchange_instance.load_markets.return_value = {'BTC/USDT': {}}

            # Mock OHLCV data
            ohlcv_data = [
                [1640995200000, 100.0, 105.0, 95.0, 103.0, 1000.0],  # 2022-01-01
                [1641081600000, 103.0, 108.0, 98.0, 106.0, 1100.0],  # 2022-01-02
            ]
            mock_exchange_instance.fetch_ohlcv.return_value = ohlcv_data

            source = create_data_source('ccxt', {'exchange': 'binance'})
            result = source.fetch_historical_data(
                symbol='BTC/USDT',
                timeframe='1d',
                start_date=datetime(2022, 1, 1),
                end_date=datetime(2022, 1, 2)
            )

            assert not result.empty
            assert len(result) == 2
            assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    @patch('tradpal_indicator.services.data_service.data_sources.factory.KAGGLE_AVAILABLE', True)
    def test_kaggle_integration(self, mock_kaggle):
        """Test Kaggle data source integration."""
        with patch('tradpal_indicator.services.data_service.data_sources.kaggle.Path') as mock_path, \
             patch('builtins.open', create=True) as mock_open, \
             patch('pandas.read_csv') as mock_read_csv:

            # Mock file system
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.stat.return_value.st_size = 1000

            # Mock CSV data
            mock_df = pd.DataFrame({
                'Timestamp': pd.date_range('2023-01-01', periods=3, freq='1min'),
                'Open': [100.0, 101.0, 102.0],
                'High': [105.0, 106.0, 107.0],
                'Low': [95.0, 96.0, 97.0],
                'Close': [103.0, 104.0, 105.0],
                'Volume_(BTC)': [10.0, 11.0, 12.0],
                'Volume_(Currency)': [1000.0, 1100.0, 1200.0],
                'Weighted_Price': [101.0, 102.0, 103.0]
            })
            mock_read_csv.return_value = mock_df

            source = create_data_source('kaggle')
            result = source.fetch_historical_data(
                symbol='BTC/USDT',
                timeframe='1m',
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 1, 0, 2)
            )

            assert not result.empty
            assert len(result) == 3
            assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_convenience_function(self):
        """Test the convenience create_data_source function."""
        with patch('tradpal_indicator.services.data_service.data_sources.factory.DataSourceFactory.create_data_source') as mock_create:
            mock_create.return_value = MagicMock()

            result = create_data_source('test_source')

            mock_create.assert_called_once_with('test_source', None)
            assert result is not None


class TestDataSourceCompatibility:
    """Test compatibility between different data sources."""

    def test_all_sources_have_common_interface(self):
        """Test that all data sources implement the common interface."""
        with patch('tradpal_indicator.services.data_service.data_sources.factory.YAHOO_AVAILABLE', True), \
             patch('tradpal_indicator.services.data_service.data_sources.factory.CCXT_AVAILABLE', True), \
             patch('tradpal_indicator.services.data_service.data_sources.factory.KAGGLE_AVAILABLE', True):

            sources = ['yahoo_finance', 'ccxt', 'kaggle']

            for source_name in sources:
                source = create_data_source(source_name)

                # Check required methods exist
                assert hasattr(source, 'fetch_historical_data')
                assert hasattr(source, 'fetch_recent_data')
                assert hasattr(source, 'validate_data')
                assert hasattr(source, 'get_info')
                assert hasattr(source, 'is_available')

                # Check name attribute
                assert hasattr(source, 'name')
                assert isinstance(source.name, str)

    def test_sources_return_consistent_data_format(self):
        """Test that all sources return data in the same format."""
        # This would require mocking all sources to return similar data
        # and checking that the DataFrame structure is consistent
        pass