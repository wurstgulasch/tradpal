"""
Funding Rate Data Source Implementation

Provides funding rate data for perpetual futures contracts using CCXT library.
Funding rates are essential for perpetual futures trading strategies.
"""

import ccxt
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

from . import DataSource

logger = logging.getLogger(__name__)

class FundingRateDataSource(DataSource):
    """Funding rate data source for perpetual futures."""

    # Funding rates are typically updated every 8 hours (480 minutes)
    # but we'll support various timeframes for historical analysis
    TIMEFRAME_MAPPING = {
        '1h': '1h',
        '4h': '4h',
        '8h': '8h',  # Most common funding interval
        '1d': '1d',
        '1w': '1w'
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
            self.logger.info(f"Initialized {self.exchange_name} exchange for funding rates")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.exchange_name}: {e}")
            raise

    def fetch_historical_data(self, symbol: str, timeframe: str, start_date: datetime,
                            end_date: Optional[datetime] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical funding rate data.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT:USDT' for perpetual futures)
            timeframe: Timeframe string (funding rates are typically 8h intervals)
            start_date: Start date
            end_date: End date (defaults to now)
            limit: Maximum records per request

        Returns:
            DataFrame with funding rate data
        """
        try:
            # Set defaults
            if end_date is None:
                end_date = datetime.now()
            if limit is None:
                limit = 1000

            self.logger.info(f"Fetching funding rates for {symbol} from {start_date} to {end_date}")

            # Check if exchange supports funding rates
            if not hasattr(self.exchange, 'fetchFundingRateHistory'):
                raise ValueError(f"Exchange {self.exchange_name} does not support funding rate history")

            # Convert dates to timestamps
            since = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)

            all_data = []
            current_since = since

            while current_since < end_timestamp:
                try:
                    # Fetch funding rate history
                    # Note: CCXT's fetchFundingRateHistory may have different parameters per exchange
                    funding_data = self.exchange.fetchFundingRateHistory(
                        symbol=symbol,
                        since=current_since,
                        limit=limit
                    )

                    if not funding_data:
                        break

                    all_data.extend(funding_data)

                    # Update since for next batch (funding rates are typically every 8 hours)
                    if funding_data:
                        last_timestamp = funding_data[-1]['timestamp']
                        current_since = last_timestamp + (8 * 60 * 60 * 1000)  # 8 hours in ms

                    # Safety check
                    if len(funding_data) < limit:
                        break

                    # Rate limiting
                    time.sleep(self.exchange.rateLimit / 1000)

                except Exception as e:
                    self.logger.warning(f"Error fetching funding rate batch: {e}")
                    break

            if not all_data:
                self.logger.warning(f"No funding rate data found for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(all_data)

            # Standardize column names
            column_mapping = {
                'timestamp': 'timestamp',
                'fundingRate': 'funding_rate',
                'funding_rate': 'funding_rate',
                'rate': 'funding_rate'
            }

            df = df.rename(columns=column_mapping)

            # Ensure we have the required columns
            if 'timestamp' not in df.columns:
                self.logger.error("No timestamp column found in funding rate data")
                return pd.DataFrame()

            if 'funding_rate' not in df.columns:
                self.logger.error("No funding_rate column found in funding rate data")
                return pd.DataFrame()

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            # Sort and remove duplicates
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]

            # Convert funding rate to percentage if needed
            if 'funding_rate' in df.columns:
                # Some exchanges return rates as decimals, some as percentages
                # Normalize to decimal form (e.g., 0.01 for 1%)
                df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')

                # If rates are very large (>1), they might be in basis points or similar
                # Convert to decimal if they appear to be percentages
                if df['funding_rate'].abs().max() > 1:
                    df['funding_rate'] = df['funding_rate'] / 100.0

            # Add derived columns
            df['funding_cost_annualized'] = df['funding_rate'] * 3 * 365  # Assuming 3 funding payments per day
            df['funding_direction'] = df['funding_rate'].apply(lambda x: 'long' if x > 0 else 'short')

            self.logger.info(f"Fetched {len(df)} funding rate records for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching funding rate data: {e}")
            raise

    def fetch_current_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch current funding rate for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dict with current funding rate info or None if not available
        """
        try:
            if hasattr(self.exchange, 'fetchFundingRate'):
                funding_info = self.exchange.fetchFundingRate(symbol)
                return {
                    'symbol': symbol,
                    'funding_rate': funding_info.get('fundingRate', 0),
                    'next_funding_time': funding_info.get('nextFundingTime'),
                    'timestamp': datetime.now()
                }
            else:
                self.logger.warning(f"Exchange {self.exchange_name} does not support fetchFundingRate")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching current funding rate for {symbol}: {e}")
            return None

    def get_supported_timeframes(self) -> List[str]:
        """Return supported timeframes for funding rate analysis."""
        return list(self.TIMEFRAME_MAPPING.keys())

    def get_supported_symbols(self) -> List[str]:
        """Return supported perpetual futures symbols."""
        try:
            markets = self.exchange.load_markets()
            # Filter for perpetual futures (typically contain ':USDT' or similar)
            perpetual_symbols = [
                symbol for symbol in markets.keys()
                if ':USDT' in symbol or ':BUSD' in symbol or 'PERP' in symbol.upper()
            ]
            return perpetual_symbols
        except Exception as e:
            self.logger.error(f"Error loading perpetual markets: {e}")
            return []

    def get_funding_rate_stats(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Get funding rate statistics for analysis.

        Args:
            symbol: Trading pair
            days: Number of days to analyze

        Returns:
            Dict with funding rate statistics
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            df = self.fetch_historical_data(symbol, '8h', start_date, end_date)

            if df.empty:
                return {}

            stats = {
                'symbol': symbol,
                'period_days': days,
                'mean_funding_rate': df['funding_rate'].mean(),
                'median_funding_rate': df['funding_rate'].median(),
                'std_funding_rate': df['funding_rate'].std(),
                'min_funding_rate': df['funding_rate'].min(),
                'max_funding_rate': df['funding_rate'].max(),
                'positive_rate_percentage': (df['funding_rate'] > 0).mean() * 100,
                'negative_rate_percentage': (df['funding_rate'] < 0).mean() * 100,
                'annualized_cost_mean': df['funding_cost_annualized'].mean(),
                'last_funding_rate': df['funding_rate'].iloc[-1] if not df.empty else None,
                'last_update': df.index[-1] if not df.empty else None
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating funding rate stats for {symbol}: {e}")
            return {}