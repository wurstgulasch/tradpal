"""
Unit Tests for Yahoo Finance Data Source
"""

import pytest
from unittest.mock import patch

from .yahoo_finance import YahooFinanceDataSource


class TestYahooFinanceDataSource:
    """Test cases for YahooFinanceDataSource."""

    @patch('services.data_service.data_sources.yahoo_finance.YFINANCE_AVAILABLE', True)
    def test_initialization_success(self):
        """Test successful initialization."""
        source = YahooFinanceDataSource()
        assert source.name == "Yahoo Finance"

    @patch('services.data_service.data_sources.yahoo_finance.YFINANCE_AVAILABLE', False)
    def test_initialization_yfinance_not_available(self):
        """Test initialization fails when yfinance not available."""
        with pytest.raises(ImportError, match="yfinance library is required"):
            YahooFinanceDataSource()

    @patch('services.data_service.data_sources.yahoo_finance.YFINANCE_AVAILABLE', True)
    def test_convert_symbol_format(self):
        """Test symbol format conversion."""
        source = YahooFinanceDataSource()
        assert source._convert_symbol_format('BTC/USDT') == 'BTC-USD'
        assert source._convert_symbol_format('AAPL') == 'AAPL'

    @patch('services.data_service.data_sources.yahoo_finance.YFINANCE_AVAILABLE', True)
    def test_map_timeframe(self):
        """Test timeframe mapping."""
        source = YahooFinanceDataSource()
        assert source._map_timeframe('1d') == '1d'
        assert source._map_timeframe('1h') == '60m'
        assert source._map_timeframe('unknown') == '1d'

    @patch('services.data_service.data_sources.yahoo_finance.YFINANCE_AVAILABLE', True)
    def test_is_available(self):
        """Test is_available method."""
        source = YahooFinanceDataSource()
        assert source.is_available() is True
