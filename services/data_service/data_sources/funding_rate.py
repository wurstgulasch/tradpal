"""
Funding Rate Data Source for TradPal Indicator System

This module provides data fetching for funding rate data from cryptocurrency exchanges.
Funding rates are crucial for understanding market sentiment and regime detection.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import time
import requests

from .base import BaseDataSource

logger = logging.getLogger(__name__)

class FundingRateDataSource(BaseDataSource):
    """
    Data source for fetching funding rate data from cryptocurrency exchanges.

    Funding rates indicate market sentiment:
    - Positive funding rates: Long positions are paying shorts (bullish bias)
    - Negative funding rates: Short positions are paying longs (bearish bias)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Funding Rate data source.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__("Funding Rate", config)

        # Default configuration
        self.config.setdefault('exchange', 'binance')  # Default exchange
        self.config.setdefault('api_base_url', 'https://fapi.binance.com')  # Binance Futures API
        self.config.setdefault('timeout', 30)  # Request timeout
        self.config.setdefault('max_retries', 3)  # Max retry attempts
        self.config.setdefault('retry_delay', 1)  # Delay between retries

        # Validate exchange support
        supported_exchanges = ['binance']
        if self.config['exchange'] not in supported_exchanges:
            raise ValueError(f"Exchange '{self.config['exchange']}' not supported. Supported: {supported_exchanges}")

    def fetch_historical_data(self, symbol: str, timeframe: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical funding rate data.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe string (ignored for funding rates, always 8h intervals)
            start_date: Start date for historical data
            end_date: End date for historical data
            limit: Maximum number of funding rate entries

        Returns:
            DataFrame with funding rate data indexed by timestamp
        """
        if self.config['exchange'] == 'binance':
            return self._fetch_binance_funding_rate(symbol, start_date, end_date, limit)
        else:
            raise NotImplementedError(f"Exchange {self.config['exchange']} not implemented")

    def fetch_recent_data(self, symbol: str, timeframe: str,
                         limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent funding rate data.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe string (ignored for funding rates)
            limit: Number of recent funding rate entries

        Returns:
            DataFrame with funding rate data
        """
        # For recent data, fetch from the last 'limit' funding rate periods
        end_date = datetime.now()
        # Each funding rate is every 8 hours, so calculate start date
        start_date = end_date - timedelta(hours=8 * limit)

        return self.fetch_historical_data(symbol, timeframe, start_date, end_date, limit)

    def _fetch_binance_funding_rate(self, symbol: str,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None,
                                   limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch funding rate data from Binance Futures API.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start_date: Start date
            end_date: End date
            limit: Maximum number of entries

        Returns:
            DataFrame with funding rate data
        """
        try:
            # Binance endpoint for historical funding rates
            endpoint = f"{self.config['api_base_url']}/fapi/v1/fundingRate"

            params = {
                'symbol': symbol.upper(),
                'limit': min(limit or 100, 1000)  # Binance max is 1000
            }

            if start_date:
                params['startTime'] = int(start_date.timestamp() * 1000)
            if end_date:
                params['endTime'] = int(end_date.timestamp() * 1000)

            self.logger.info(f"Fetching funding rate data for {symbol} from Binance")

            # Make request with retries
            for attempt in range(self.config['max_retries']):
                try:
                    response = requests.get(endpoint, params=params,
                                          timeout=self.config['timeout'])
                    response.raise_for_status()
                    data = response.json()
                    break
                except Exception as e:
                    if attempt == self.config['max_retries'] - 1:
                        raise e
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(self.config['retry_delay'])

            if not data:
                self.logger.warning(f"No funding rate data received for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Rename columns to standard format
            column_mapping = {
                'fundingTime': 'timestamp',
                'fundingRate': 'funding_rate',
                'symbol': 'symbol'
            }

            df = df.rename(columns=column_mapping)

            # Convert timestamp to datetime (ensure tz-naive)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            # Convert funding rate to float
            df['funding_rate'] = df['funding_rate'].astype(float)

            # Set timestamp as index
            df = df.set_index('timestamp')

            # Sort by timestamp
            df = df.sort_index()

            # Add additional columns for analysis
            df['funding_rate_pct'] = df['funding_rate'] * 100  # Convert to percentage
            df['funding_rate_bps'] = df['funding_rate'] * 10000  # Convert to basis points

            # Add market regime indicator based on funding rate
            df['market_regime'] = df['funding_rate'].apply(self._classify_market_regime)

            self.logger.info(f"Successfully fetched {len(df)} funding rate records for {symbol}")

            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch funding rate data for {symbol}: {e}")
            return pd.DataFrame()

    def _classify_market_regime(self, funding_rate: float) -> str:
        """
        Classify market regime based on funding rate.

        Args:
            funding_rate: Funding rate value

        Returns:
            Market regime classification
        """
        if funding_rate > 0.01:  # > 1%
            return 'extremely_bullish'
        elif funding_rate > 0.005:  # > 0.5%
            return 'bullish'
        elif funding_rate > 0.001:  # > 0.1%
            return 'mildly_bullish'
        elif funding_rate > -0.001:  # -0.1% to 0.1%
            return 'neutral'
        elif funding_rate > -0.005:  # -0.5% to -0.1%
            return 'mildly_bearish'
        elif funding_rate > -0.01:  # -1% to -0.5%
            return 'bearish'
        else:  # < -1%
            return 'extremely_bearish'

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate funding rate data.

        Args:
            df: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        if df.empty:
            return False

        required_columns = ['funding_rate']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for NaN values
        if df['funding_rate'].isna().any():
            raise ValueError("NaN values found in funding_rate column")

        # Check funding rate ranges (should be reasonable)
        if (df['funding_rate'].abs() > 1.0).any():  # More than 100% seems unreasonable
            self.logger.warning("Unusually high funding rates detected")

        return True

    def get_funding_rate_stats(self, symbol: str,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get funding rate statistics for analysis.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with funding rate statistics
        """
        df = self.fetch_historical_data(symbol, '8h', start_date, end_date)

        if df.empty:
            return {}

        stats = {
            'mean_funding_rate': df['funding_rate'].mean(),
            'median_funding_rate': df['funding_rate'].median(),
            'std_funding_rate': df['funding_rate'].std(),
            'min_funding_rate': df['funding_rate'].min(),
            'max_funding_rate': df['funding_rate'].max(),
            'current_funding_rate': df['funding_rate'].iloc[-1] if not df.empty else None,
            'positive_rate_percentage': (df['funding_rate'] > 0).mean() * 100,
            'negative_rate_percentage': (df['funding_rate'] < 0).mean() * 100,
            'neutral_rate_percentage': (df['funding_rate'] == 0).mean() * 100,
            'regime_distribution': df['market_regime'].value_counts().to_dict(),
            'data_points': len(df)
        }

        return stats