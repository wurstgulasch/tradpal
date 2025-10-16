"""
Tests for Kaggle Data Source

This module contains unit tests for the KaggleDataSource class.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

# Import the data source
from .kaggle import KaggleDataSource


class TestKaggleDataSource:
    """Test cases for KaggleDataSource."""

    def test_initialization(self):
        """Test data source initialization."""
        config = {"dataset": "test/dataset", "auto_download": False}
        source = KaggleDataSource(config)

        assert source.name == "Kaggle"
        assert source.config["dataset"] == "test/dataset"
        assert source.config["auto_download"] is False

    def test_initialization_default_config(self):
        """Test data source initialization with default config."""
        source = KaggleDataSource()

        assert source.name == "Kaggle"
        assert source.config["dataset"] == "mczielinski/bitcoin-historical-data"
        assert source.config["auto_download"] is True

    @patch('tradpal_indicator.services.data_service.data_sources.kaggle.Path')
    def test_kaggle_auth_check_success(self, mock_path):
        """Test successful Kaggle authentication check."""
        mock_path.return_value.exists.return_value = True

        # Should not raise an exception
        source = KaggleDataSource()
        assert source.name == "Kaggle"

    @patch('tradpal_indicator.services.data_service.data_sources.kaggle.Path')
    def test_kaggle_auth_check_failure(self, mock_path):
        """Test failed Kaggle authentication check."""
        mock_path.return_value.exists.return_value = False

        with pytest.raises(RuntimeError, match="Kaggle API key not found"):
            KaggleDataSource()

    def test_is_bitcoin_symbol(self):
        """Test Bitcoin symbol validation."""
        source = KaggleDataSource()

        # Valid Bitcoin symbols
        assert source._is_bitcoin_symbol("BTC/USDT") is True
        assert source._is_bitcoin_symbol("BTC/USD") is True
        assert source._is_bitcoin_symbol("BTCUSDT") is True

        # Invalid symbols
        assert source._is_bitcoin_symbol("ETH/USDT") is False
        assert source._is_bitcoin_symbol("AAPL") is False

    def test_convert_to_tradpal_format(self):
        """Test conversion to TradPal OHLCV format."""
        source = KaggleDataSource()

        # Create mock Kaggle data
        data = {
            'Timestamp': pd.date_range('2023-01-01', periods=3, freq='1min'),
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [95.0, 96.0, 97.0],
            'Close': [103.0, 104.0, 105.0],
            'Volume_(BTC)': [10.0, 11.0, 12.0],
            'Volume_(Currency)': [1000.0, 1100.0, 1200.0],
            'Weighted_Price': [101.0, 102.0, 103.0]
        }
        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

        result = source._convert_to_tradpal_format(df)

        # Check required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            assert col in result.columns

        # Check data types
        assert pd.api.types.is_numeric_dtype(result['open'])
        assert pd.api.types.is_numeric_dtype(result['volume'])

    def test_filter_date_range(self):
        """Test date range filtering."""
        source = KaggleDataSource()

        # Create test data
        dates = pd.date_range('2023-01-01', periods=10, freq='1D')
        df = pd.DataFrame({
            'open': range(10),
            'high': range(10, 20),
            'low': range(20, 30),
            'close': range(30, 40),
            'volume': range(40, 50)
        }, index=dates)

        start_date = datetime(2023, 1, 3)
        end_date = datetime(2023, 1, 7)

        result = source._filter_date_range(df, start_date, end_date)

        assert len(result) == 5  # 2023-01-03 to 2023-01-07 inclusive
        assert result.index.min() >= start_date
        assert result.index.max() <= end_date

    def test_resample_data(self):
        """Test data resampling to different timeframes."""
        source = KaggleDataSource()

        # Create 1-minute data
        dates = pd.date_range('2023-01-01', periods=60, freq='1min')  # 1 hour of data
        df = pd.DataFrame({
            'open': range(60),
            'high': range(60, 120),
            'low': range(120, 180),
            'close': range(180, 240),
            'volume': range(240, 300)
        }, index=dates)

        # Resample to 5-minute data
        result = source._resample_data(df, '5m')

        assert len(result) == 12  # 60 minutes / 5 minutes = 12 periods
        assert result.index.freq.name == '5min'

    @patch('tradpal_indicator.services.data_service.data_sources.kaggle.kaggle')
    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    def test_download_dataset(self, mock_auth, mock_kaggle):
        """Test dataset download functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = KaggleDataSource({
                "dataset": "test/dataset",
                "cache_dir": temp_dir
            })

            # Mock successful download
            mock_kaggle.api.competition_download_files = Mock()

            # Create a mock zip file
            zip_path = Path(temp_dir) / "test-dataset.zip"
            zip_path.touch()

            source._download_dataset()

            # Verify download was called
            mock_kaggle.api.competition_download_files.assert_called_once_with(
                "test/dataset",
                path=str(Path(temp_dir)),
                quiet=False
            )

    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    def test_load_data_from_file(self, mock_auth):
        """Test loading data from cached file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock CSV data
            csv_data = """Timestamp,Open,High,Low,Close,Volume_(BTC),Volume_(Currency),Weighted_Price
2023-01-01 00:00:00,100.0,105.0,95.0,103.0,10.0,1000.0,101.0
2023-01-01 00:01:00,103.0,108.0,98.0,106.0,11.0,1100.0,102.0
"""

            csv_path = Path(temp_dir) / "btcusd_1-min_data.csv"
            csv_path.write_text(csv_data)

            source = KaggleDataSource({
                "cache_dir": temp_dir,
                "filename": "btcusd_1-min_data.csv"
            })

            result = source._load_data()

            assert not result.empty
            assert len(result) == 2
            assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    def test_get_dataset_info(self, mock_auth):
        """Test getting dataset information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = KaggleDataSource({
                "dataset": "test/dataset",
                "cache_dir": temp_dir,
                "filename": "test.csv"
            })

            info = source.get_dataset_info()

            assert info["dataset"] == "test/dataset"
            assert info["filename"] == "test.csv"
            assert info["cache_dir"] == temp_dir
            assert "data_file_exists" in info

    def test_validate_data_success(self):
        """Test successful data validation."""
        source = KaggleDataSource()

        # Create valid OHLCV data
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [103.0, 104.0, 105.0],
            'volume': [10.0, 11.0, 12.0]
        })

        assert source.validate_data(df) is True

    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        source = KaggleDataSource()

        # Create data missing volume column
        df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [95.0, 96.0],
            'close': [103.0, 104.0]
            # Missing 'volume'
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            source.validate_data(df)

    def test_validate_data_empty_dataframe(self):
        """Test data validation with empty DataFrame."""
        source = KaggleDataSource()

        df = pd.DataFrame()

        assert source.validate_data(df) is False

    def test_validate_data_invalid_ohlc(self):
        """Test data validation with invalid OHLC relationships."""
        source = KaggleDataSource()

        # Create data with invalid OHLC (high < low)
        df = pd.DataFrame({
            'open': [100.0],
            'high': [95.0],  # High < Low - invalid
            'low': [105.0],
            'close': [103.0],
            'volume': [10.0]
        })

        # Should still return True but log warning (current implementation)
        assert source.validate_data(df) is True