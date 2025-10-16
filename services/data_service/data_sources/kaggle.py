"""
Kaggle Data Source for TradPal Indicator System

This module provides data fetching from Kaggle datasets, specifically
optimized for Bitcoin historical data for backtesting purposes.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import os
import zipfile
from pathlib import Path

try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    kaggle = None

from .base import BaseDataSource

logger = logging.getLogger(__name__)

class KaggleDataSource(BaseDataSource):
    """
    Data source for fetching historical Bitcoin data from Kaggle datasets.

    This source is optimized for backtesting with high-quality historical data.
    """

    # Default Kaggle dataset for Bitcoin data
    DEFAULT_DATASET = "mczielinski/bitcoin-historical-data"
    DEFAULT_FILE = "btcusd_1-min_data.csv"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Kaggle data source.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__("Kaggle", config)

        if not KAGGLE_AVAILABLE:
            raise ImportError("kaggle library is required for Kaggle data source. Install with: pip install kaggle")

        # Default configuration
        self.config.setdefault('dataset', self.DEFAULT_DATASET)
        self.config.setdefault('filename', self.DEFAULT_FILE)
        self.config.setdefault('cache_dir', './cache/kaggle_data')
        self.config.setdefault('auto_download', True)
        self.config.setdefault('force_refresh', False)

        # Ensure cache directory exists
        self.cache_dir = Path(self.config['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Local data file path
        self.data_file = self.cache_dir / self.config['filename']

        # Check if Kaggle API key is configured
        self._check_kaggle_auth()

    def _check_kaggle_auth(self):
        """Check if Kaggle API authentication is configured."""
        kaggle_dir = Path.home() / '.kaggle'
        api_key_file = kaggle_dir / 'kaggle.json'

        if not api_key_file.exists():
            raise RuntimeError(
                "Kaggle API key not found. Please:\n"
                "1. Go to https://www.kaggle.com/account\n"
                "2. Create API token (kaggle.json)\n"
                "3. Place kaggle.json in ~/.kaggle/ directory\n"
                "4. Ensure proper permissions: chmod 600 ~/.kaggle/kaggle.json"
            )

    def fetch_historical_data(self, symbol: str, timeframe: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical Bitcoin data from Kaggle dataset.

        Args:
            symbol: Trading symbol (must be 'BTC/USDT' or similar Bitcoin pair)
            timeframe: Timeframe string ('1m', '1h', '1d')
            start_date: Start date for historical data
            end_date: End date for historical data
            limit: Maximum number of candles

        Returns:
            DataFrame with OHLCV data
        """
        # Validate symbol (only Bitcoin supported for now)
        if not self._is_bitcoin_symbol(symbol):
            raise ValueError(f"Kaggle data source only supports Bitcoin symbols, got: {symbol}")

        try:
            # Ensure data is downloaded
            if not self.data_file.exists() or self.config['force_refresh']:
                if self.config['auto_download']:
                    self._download_dataset()
                else:
                    raise FileNotFoundError(f"Data file not found: {self.data_file}")

            # Load data from cache
            df = self._load_data()

            if df.empty:
                self.logger.warning("No data loaded from Kaggle dataset")
                return pd.DataFrame()

            # Resample to requested timeframe
            df = self._resample_data(df, timeframe)

            # Filter by date range
            df = self._filter_date_range(df, start_date, end_date)

            # Apply limit
            if limit and len(df) > limit:
                df = df.tail(limit)

            # Validate data
            self.validate_data(df)

            self.logger.info(f"Successfully loaded {len(df)} candles for {symbol} {timeframe}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data from Kaggle: {e}")
            raise

    def fetch_recent_data(self, symbol: str, timeframe: str,
                         limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent Bitcoin data from Kaggle dataset.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            limit: Number of recent candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        # For recent data, get the most recent candles
        return self.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )

    def _download_dataset(self):
        """Download the Kaggle dataset."""
        try:
            self.logger.info(f"Downloading Kaggle dataset: {self.config['dataset']}")

            # Download and unzip the dataset
            kaggle.api.competition_download_files(
                self.config['dataset'],
                path=str(self.cache_dir),
                quiet=False
            )

            # Find the downloaded zip file
            zip_files = list(self.cache_dir.glob("*.zip"))
            if zip_files:
                zip_path = zip_files[0]
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.cache_dir)
                zip_path.unlink()  # Remove zip file after extraction

            self.logger.info("Kaggle dataset downloaded and extracted successfully")

        except Exception as e:
            self.logger.error(f"Failed to download Kaggle dataset: {e}")
            raise

    def _load_data(self) -> pd.DataFrame:
        """Load Bitcoin data from the cached CSV file."""
        try:
            self.logger.info(f"Loading data from {self.data_file}")

            # Load CSV with proper parsing
            df = pd.read_csv(
                self.data_file,
                parse_dates=['Timestamp'],
                index_col='Timestamp'
            )

            # Rename columns to match expected format
            column_map = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume_(BTC)': 'volume',
                'Volume_(Currency)': 'volume_currency',
                'Weighted_Price': 'weighted_price'
            }

            df = df.rename(columns=column_map)

            # Ensure we have the required OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Missing required columns in Kaggle data: {missing_cols}")

            # Convert to numeric and handle NaN values
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows with NaN values in critical columns
            df = df.dropna(subset=['open', 'high', 'low', 'close'])

            # Sort by timestamp
            df = df.sort_index()

            return df

        except Exception as e:
            self.logger.error(f"Failed to load data from {self.data_file}: {e}")
            raise

    def _resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to the requested timeframe.

        Args:
            df: Input DataFrame (1-minute data)
            timeframe: Target timeframe

        Returns:
            Resampled DataFrame
        """
        # Map timeframe to pandas frequency
        freq_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W',
            '1M': '1M'
        }

        freq = freq_map.get(timeframe, '1H')  # Default to 1 hour

        # Resample OHLCV data
        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Remove NaN values created by resampling
        resampled = resampled.dropna()

        return resampled

    def _filter_date_range(self, df: pd.DataFrame,
                          start_date: Optional[datetime],
                          end_date: Optional[datetime]) -> pd.DataFrame:
        """
        Filter DataFrame by date range.

        Args:
            df: Input DataFrame
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Filtered DataFrame
        """
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return df

    def _is_bitcoin_symbol(self, symbol: str) -> bool:
        """
        Check if the symbol is a Bitcoin-related symbol.

        Args:
            symbol: Trading symbol

        Returns:
            True if Bitcoin symbol, False otherwise
        """
        bitcoin_symbols = ['BTC/USDT', 'BTC/USD', 'BTCUSDT', 'BTCUSD', 'BTC/USDC', 'BTCUSDC']
        return symbol.upper() in bitcoin_symbols

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the current Kaggle dataset.

        Returns:
            Dictionary with dataset information
        """
        return {
            'dataset': self.config['dataset'],
            'filename': self.config['filename'],
            'cache_dir': str(self.cache_dir),
            'data_file_exists': self.data_file.exists(),
            'data_file_size': self.data_file.stat().st_size if self.data_file.exists() else 0,
            'auto_download': self.config['auto_download'],
            'force_refresh': self.config['force_refresh']
        }