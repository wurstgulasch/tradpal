"""
CCXT Data Source Implementation

Provides real-time and historical market data using CCXT library.
Best for crypto exchanges with API access.
"""

import ccxt
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

from . import DataSource

logger = logging.getLogger(__name__)

class CCXTDataSource(DataSource):
    """CCXT data source implementation."""

    # Mapping from our timeframe format to CCXT timeframes
    TIMEFRAME_MAPPING = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
        '1w': '1w',
        '1M': '1M'
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.exchange_name = config.get('exchange', 'binance')
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')

        # Initialize exchange
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                }
            })
            self.logger.info(f"Initialized {self.exchange_name} exchange")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.exchange_name}: {e}")
            raise

    def fetch_historical_data(self, symbol: str, timeframe: str, start_date: datetime,
                            end_date: Optional[datetime] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical data from CCXT exchange.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe string
            start_date: Start date
            end_date: End date (defaults to now)
            limit: Maximum records per request

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Map timeframe
            ccxt_timeframe = self.TIMEFRAME_MAPPING.get(timeframe)
            if not ccxt_timeframe:
                raise ValueError(f"Unsupported timeframe: {timeframe}")

            # Set defaults
            if end_date is None:
                end_date = datetime.now()
            if limit is None:
                limit = 1000

            self.logger.info(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}")

            # Convert dates to timestamps
            since = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)

            all_data = []
            current_since = since

            while current_since < end_timestamp:
                try:
                    # Fetch batch of data
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=ccxt_timeframe,
                        since=current_since,
                        limit=limit
                    )

                    if not ohlcv:
                        break

                    all_data.extend(ohlcv)

                    # Update since for next batch
                    last_timestamp = ohlcv[-1][0]
                    current_since = last_timestamp + (self.exchange.timeframes[ccxt_timeframe] * 1000)

                    # Safety check to prevent infinite loops
                    if len(ohlcv) < limit:
                        break

                    # Rate limiting
                    time.sleep(self.exchange.rateLimit / 1000)

                except Exception as e:
                    self.logger.warning(f"Error fetching batch: {e}")
                    break

            if not all_data:
                self.logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            # Sort and remove duplicates
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]

            self.logger.info(f"Fetched {len(df)} records for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data from CCXT: {e}")
            raise

    def get_supported_timeframes(self) -> List[str]:
        """Return supported timeframes."""
        return list(self.TIMEFRAME_MAPPING.keys())

    def get_supported_symbols(self) -> List[str]:
        """Return supported symbols from exchange."""
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            self.logger.error(f"Error loading markets: {e}")
            return []