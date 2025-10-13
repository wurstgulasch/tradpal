"""
Yahoo Finance Data Source Implementation

Provides historical market data using yfinance library.
Best for long-term historical data and traditional assets.
"""

import yfinance as yf
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

from . import DataSource

logger = logging.getLogger(__name__)

class YahooFinanceDataSource(DataSource):
    """Yahoo Finance data source implementation."""

    # Mapping from our timeframe format to yfinance intervals
    TIMEFRAME_MAPPING = {
        '1m': '1m',
        '2m': '2m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '1d': '1d',
        '5d': '5d',
        '1wk': '1wk',
        '1mo': '1mo',
        '3mo': '3mo'
    }

    # Supported symbols (crypto pairs mapped to Yahoo tickers)
    SYMBOL_MAPPING = {
        'BTC/USDT': 'BTC-USD',
        'ETH/USDT': 'ETH-USD',
        'BNB/USDT': 'BNB-USD',
        'ADA/USDT': 'ADA-USD',
        'SOL/USDT': 'SOL-USD',
        'DOT/USDT': 'DOT-USD',
        'AVAX/USDT': 'AVAX-USD',
        'LTC/USDT': 'LTC-USD',
        'LINK/USDT': 'LINK-USD',
        'UNI/USDT': 'UNI-USD',
        # Add more mappings as needed
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adjust_prices = config.get('adjust_prices', True)
        self.auto_adjust = config.get('auto_adjust', True)
        self.prepost = config.get('prepost', False)

    def fetch_historical_data(self, symbol: str, timeframe: str, start_date: datetime,
                            end_date: Optional[datetime] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe string
            start_date: Start date
            end_date: End date (defaults to now)
            limit: Maximum records (ignored by yfinance)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Map symbol to Yahoo ticker
            yahoo_symbol = self.SYMBOL_MAPPING.get(symbol, symbol.replace('/', '-'))
            self.logger.info(f"Fetching {yahoo_symbol} data from {start_date} to {end_date or 'now'}")

            # Map timeframe
            interval = self.TIMEFRAME_MAPPING.get(timeframe)
            if not interval:
                raise ValueError(f"Unsupported timeframe: {timeframe}")

            # Set end_date if not provided
            if end_date is None:
                end_date = datetime.now()

            # Add buffer to ensure we get all data
            end_date = end_date + timedelta(days=1)

            # Fetch data
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=self.auto_adjust,
                prepost=self.prepost
            )

            if df.empty:
                self.logger.warning(f"No data found for {yahoo_symbol}")
                return pd.DataFrame()

            # Rename columns to match our format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Ensure we have all required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0.0

            # Reset index to have timestamp as column
            df = df.reset_index()
            if 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'timestamp'})
            elif 'Date' in df.columns:
                df = df.rename(columns={'Date': 'timestamp'})

            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')

            # Sort by timestamp
            df = df.sort_index()

            # Remove any duplicate indices
            df = df[~df.index.duplicated(keep='first')]

            self.logger.info(f"Fetched {len(df)} records for {yahoo_symbol}")
            return df[required_cols]

        except Exception as e:
            self.logger.error(f"Error fetching data from Yahoo Finance: {e}")
            raise

    def get_supported_timeframes(self) -> List[str]:
        """Return supported timeframes."""
        return list(self.TIMEFRAME_MAPPING.keys())

    def get_supported_symbols(self) -> List[str]:
        """Return supported symbols."""
        return list(self.SYMBOL_MAPPING.keys())