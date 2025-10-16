"""
Tests for Kaggle Data Source

This module contains unit tests for the KaggleDataSource class.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import os
import zipfile
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

    def test_kaggle_not_available(self):
        """Test behavior when kaggle library is not available."""
        with patch('.kaggle.KAGGLE_AVAILABLE', False):
            with pytest.raises(ImportError, match="kaggle library is required"):
                KaggleDataSource()

    def test_is_bitcoin_symbol(self):
        """Test Bitcoin symbol validation."""
        source = KaggleDataSource()

        # Valid Bitcoin symbols
        assert source._is_bitcoin_symbol("BTC/USDT") is True
        assert source._is_bitcoin_symbol("BTC/USD") is True
        assert source._is_bitcoin_symbol("BTCUSDT") is True
        assert source._is_bitcoin_symbol("BTC/USDC") is True
        assert source._is_bitcoin_symbol("BTCUSDC") is True

        # Invalid symbols
        assert source._is_bitcoin_symbol("ETH/USDT") is False
        assert source._is_bitcoin_symbol("AAPL") is False
        assert source._is_bitcoin_symbol("btc/usdt") is False  # Case sensitive
        assert source._is_bitcoin_symbol("") is False
        assert source._is_bitcoin_symbol("BTC") is False

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

    def test_filter_date_range_no_start(self):
        """Test date range filtering with no start date."""
        source = KaggleDataSource()

        dates = pd.date_range('2023-01-01', periods=5, freq='1D')
        df = pd.DataFrame({'open': range(5)}, index=dates)

        end_date = datetime(2023, 1, 3)
        result = source._filter_date_range(df, None, end_date)

        assert len(result) == 3
        assert result.index.max() <= end_date

    def test_filter_date_range_no_end(self):
        """Test date range filtering with no end date."""
        source = KaggleDataSource()

        dates = pd.date_range('2023-01-01', periods=5, freq='1D')
        df = pd.DataFrame({'open': range(5)}, index=dates)

        start_date = datetime(2023, 1, 3)
        result = source._filter_date_range(df, start_date, None)

        assert len(result) == 3
        assert result.index.min() >= start_date

    def test_filter_date_range_no_filters(self):
        """Test date range filtering with no filters."""
        source = KaggleDataSource()

        dates = pd.date_range('2023-01-01', periods=5, freq='1D')
        df = pd.DataFrame({'open': range(5)}, index=dates)

        result = source._filter_date_range(df, None, None)

        assert len(result) == 5
        pd.testing.assert_frame_equal(result, df)

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

    def test_resample_data_different_timeframes(self):
        """Test data resampling for various timeframes."""
        source = KaggleDataSource()

        # Create test data
        dates = pd.date_range('2023-01-01', periods=1440, freq='1min')  # 1 day of data
        df = pd.DataFrame({
            'open': range(1440),
            'high': range(1440, 2880),
            'low': range(2880, 4320),
            'close': range(4320, 5760),
            'volume': range(5760, 7200)
        }, index=dates)

        # Test different timeframes
        test_cases = [
            ('1m', 1440),  # No resampling
            ('5m', 288),   # 1440 / 5 = 288
            ('15m', 96),   # 1440 / 15 = 96
            ('1h', 24),    # 1440 / 60 = 24
            ('1d', 1),     # 1 day
        ]

        for timeframe, expected_len in test_cases:
            result = source._resample_data(df, timeframe)
            assert len(result) == expected_len, f"Failed for timeframe {timeframe}"

    def test_resample_data_unknown_timeframe(self):
        """Test data resampling with unknown timeframe."""
        source = KaggleDataSource()

        dates = pd.date_range('2023-01-01', periods=60, freq='1min')
        df = pd.DataFrame({'open': range(60)}, index=dates)

        # Unknown timeframe should default to 1H
        result = source._resample_data(df, 'unknown')
        assert len(result) == 1  # 60 minutes -> 1 hour

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

    @patch('tradpal_indicator.services.data_service.data_sources.kaggle.kaggle')
    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    def test_download_dataset_with_extraction(self, mock_auth, mock_kaggle):
        """Test dataset download with zip extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = KaggleDataSource({
                "dataset": "test/dataset",
                "cache_dir": temp_dir
            })

            mock_kaggle.api.competition_download_files = Mock()

            # Create a real zip file with test content
            zip_path = Path(temp_dir) / "test-dataset.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr('test.csv', 'col1,col2\n1,2\n3,4')

            source._download_dataset()

            # Check that zip was extracted and removed
            assert not zip_path.exists()
            assert (Path(temp_dir) / 'test.csv').exists()

    @patch('tradpal_indicator.services.data_service.data_sources.kaggle.kaggle')
    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    def test_download_dataset_failure(self, mock_auth, mock_kaggle):
        """Test dataset download failure."""
        source = KaggleDataSource({"dataset": "test/dataset"})

        mock_kaggle.api.competition_download_files.side_effect = Exception("Download failed")

        with pytest.raises(Exception, match="Download failed"):
            source._download_dataset()

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
    def test_load_data_missing_columns(self, mock_auth):
        """Test loading data with missing required columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create CSV with missing volume column
            csv_data = """Timestamp,Open,High,Low,Close
2023-01-01 00:00:00,100.0,105.0,95.0,103.0
"""

            csv_path = Path(temp_dir) / "btcusd_1-min_data.csv"
            csv_path.write_text(csv_data)

            source = KaggleDataSource({
                "cache_dir": temp_dir,
                "filename": "btcusd_1-min_data.csv"
            })

            with pytest.raises(ValueError, match="Missing required columns"):
                source._load_data()

    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    def test_load_data_with_nan_values(self, mock_auth):
        """Test loading data with NaN values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create CSV with NaN values
            csv_data = """Timestamp,Open,High,Low,Close,Volume_(BTC),Volume_(Currency),Weighted_Price
2023-01-01 00:00:00,100.0,105.0,,103.0,10.0,1000.0,101.0
2023-01-01 00:01:00,103.0,108.0,98.0,106.0,11.0,1100.0,102.0
"""

            csv_path = Path(temp_dir) / "btcusd_1-min_data.csv"
            csv_path.write_text(csv_data)

            source = KaggleDataSource({
                "cache_dir": temp_dir,
                "filename": "btcusd_1-min_data.csv"
            })

            result = source._load_data()

            # Should have only 1 row (NaN row removed)
            assert len(result) == 1
            assert result.iloc[0]['low'] == 98.0

    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    def test_load_data_file_not_found(self, mock_auth):
        """Test loading data when file doesn't exist."""
        source = KaggleDataSource({
            "cache_dir": "/nonexistent",
            "filename": "missing.csv"
        })

        with pytest.raises(FileNotFoundError):
            source._load_data()

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

    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    def test_get_dataset_info_with_existing_file(self, mock_auth):
        """Test getting dataset information with existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = Path(temp_dir) / "test.csv"
            test_file.write_text("test,data\n1,2")

            source = KaggleDataSource({
                "dataset": "test/dataset",
                "cache_dir": temp_dir,
                "filename": "test.csv"
            })

            info = source.get_dataset_info()

            assert info["data_file_exists"] is True
            assert info["data_file_size"] > 0

    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    @patch.object(KaggleDataSource, '_load_data')
    @patch.object(KaggleDataSource, '_resample_data')
    @patch.object(KaggleDataSource, '_filter_date_range')
    def test_fetch_historical_data_success(self, mock_filter, mock_resample, mock_load, mock_auth):
        """Test successful historical data fetching."""
        # Setup mocks
        mock_load.return_value = pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [103.0],
            'volume': [10.0]
        }, index=pd.date_range('2023-01-01', periods=1))

        mock_resample.return_value = mock_load.return_value
        mock_filter.return_value = mock_load.return_value

        source = KaggleDataSource()
        result = source.fetch_historical_data(
            symbol='BTC/USDT',
            timeframe='1h',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 2)
        )

        assert not result.empty
        assert len(result) == 1

        # Verify method calls
        mock_load.assert_called_once()
        mock_resample.assert_called_once()
        mock_filter.assert_called_once()

    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    def test_fetch_historical_data_invalid_symbol(self, mock_auth):
        """Test fetching with invalid symbol."""
        source = KaggleDataSource()

        with pytest.raises(ValueError, match="only supports Bitcoin symbols"):
            source.fetch_historical_data(symbol='ETH/USDT', timeframe='1h')

    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    @patch.object(KaggleDataSource, '_download_dataset')
    def test_fetch_historical_data_auto_download(self, mock_download, mock_auth):
        """Test automatic dataset download."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = KaggleDataSource({
                "cache_dir": temp_dir,
                "auto_download": True
            })

            # Simulate file not existing
            assert not source.data_file.exists()

            with patch.object(source, '_load_data') as mock_load:
                mock_load.return_value = pd.DataFrame({
                    'open': [100.0],
                    'high': [105.0],
                    'low': [95.0],
                    'close': [103.0],
                    'volume': [10.0]
                }, index=pd.date_range('2023-01-01', periods=1))

                with patch.object(source, '_resample_data', return_value=mock_load.return_value):
                    with patch.object(source, '_filter_date_range', return_value=mock_load.return_value):
                        source.fetch_historical_data('BTC/USDT', '1h')

                        mock_download.assert_called_once()

    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    def test_fetch_historical_data_no_auto_download(self, mock_auth):
        """Test behavior when auto download is disabled and file missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = KaggleDataSource({
                "cache_dir": temp_dir,
                "auto_download": False
            })

            with pytest.raises(FileNotFoundError):
                source.fetch_historical_data('BTC/USDT', '1h')

    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    @patch.object(KaggleDataSource, 'fetch_historical_data')
    def test_fetch_recent_data(self, mock_fetch, mock_auth):
        """Test fetching recent data."""
        source = KaggleDataSource()
        expected_df = pd.DataFrame({'test': [1, 2, 3]})
        mock_fetch.return_value = expected_df

        result = source.fetch_recent_data('BTC/USDT', '1h', limit=50)

        mock_fetch.assert_called_once_with(
            symbol='BTC/USDT',
            timeframe='1h',
            limit=50
        )
        assert result is expected_df

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

    def test_get_info(self):
        """Test getting data source information."""
        source = KaggleDataSource({
            "dataset": "test/dataset",
            "filename": "test.csv"
        })

        info = source.get_info()

        assert info['name'] == 'Kaggle'
        assert info['type'] == 'KaggleDataSource'
        assert 'dataset' in info['config']
        assert 'filename' in info['config']

    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    def test_is_available_true(self, mock_auth):
        """Test availability check when data can be fetched."""
        source = KaggleDataSource()

        with patch.object(source, 'fetch_recent_data') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({'open': [100.0]})

            assert source.is_available() is True
            mock_fetch.assert_called_once_with('BTC/USDT', '1d', limit=1)

    @patch.object(KaggleDataSource, '_check_kaggle_auth')
    def test_is_available_false(self, mock_auth):
        """Test availability check when data fetch fails."""
        source = KaggleDataSource()

        with patch.object(source, 'fetch_recent_data') as mock_fetch:
            mock_fetch.side_effect = Exception("Fetch failed")

            assert source.is_available() is False