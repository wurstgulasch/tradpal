"""
Liquidation Data Source for TradPal Indicator System

This module provides data fetching for liquidation data from cryptocurrency exchanges.
Liquidation data indicates market volatility and potential reversal points.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import time
import requests

from .base import BaseDataSource

logger = logging.getLogger(__name__)

class LiquidationDataSource(BaseDataSource):
    """
    Data source for fetching liquidation data from cryptocurrency exchanges.

    Liquidation data shows forced position closures:
    - Long liquidations: Bearish signal (traders forced to sell)
    - Short liquidations: Bullish signal (traders forced to buy)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Liquidation data source.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__("Liquidation Data", config)

        # Default configuration
        self.config.setdefault('exchange', 'binance')  # Default exchange
        if self.config['exchange'] == 'binance':
            self.config.setdefault('api_base_url', 'https://fapi.binance.com')  # Binance Futures API
        elif self.config['exchange'] == 'bybit':
            self.config.setdefault('api_base_url', 'https://api.bybit.com')  # Bybit API
        elif self.config['exchange'] == 'okx':
            self.config.setdefault('api_base_url', 'https://www.okx.com')  # OKX API
        elif self.config['exchange'] == 'alternative':
            # Alternative data sources that don't require auth
            self.config.setdefault('api_base_url', 'https://api.alternative.me')  # Alternative.me API
        self.config.setdefault('timeout', 30)  # Request timeout
        self.config.setdefault('max_retries', 3)  # Max retry attempts
        self.config.setdefault('retry_delay', 1)  # Delay between retries

        # Validate exchange support
        supported_exchanges = ['binance', 'bybit', 'okx', 'alternative']
        if self.config['exchange'] not in supported_exchanges:
            raise ValueError(f"Exchange '{self.config['exchange']}' not supported. Supported: {supported_exchanges}")

    def fetch_historical_data(self, symbol: str, timeframe: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical liquidation data.

        Note: Binance doesn't provide historical liquidation data via API.
        This method aggregates recent liquidation data over time periods.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe string (used for aggregation period)
            start_date: Start date for historical data
            end_date: End date for historical data
            limit: Maximum number of data points

        Returns:
            DataFrame with aggregated liquidation data
        """
        # Since Binance doesn't provide historical liquidation data,
        # we'll aggregate recent data over time periods
        if self.config['exchange'] == 'binance':
            return self._aggregate_binance_liquidations(symbol, timeframe, start_date, end_date, limit)
        elif self.config['exchange'] == 'bybit':
            return self._aggregate_bybit_liquidations(symbol, timeframe, start_date, end_date, limit)
        elif self.config['exchange'] == 'okx':
            return self._aggregate_okx_liquidations(symbol, timeframe, start_date, end_date, limit)
        elif self.config['exchange'] == 'alternative':
            return self._fetch_alternative_volatility_data(symbol, timeframe, start_date, end_date, limit)
        else:
            raise NotImplementedError(f"Exchange {self.config['exchange']} not implemented")

    def fetch_recent_data(self, symbol: str, timeframe: str,
                         limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent liquidation data from configured exchange.

        If primary exchange fails (e.g., due to auth issues), falls back to
        alternative data sources that can serve as liquidation proxies.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe string (ignored, always returns individual liquidations)
            limit: Number of recent liquidations

        Returns:
            DataFrame with recent liquidation data or proxy data
        """
        try:
            # Try primary exchange first
            if self.config['exchange'] == 'binance':
                df = self._fetch_binance_recent_liquidations(symbol, limit)
                if not df.empty:
                    return df
            elif self.config['exchange'] == 'bybit':
                df = self._fetch_bybit_recent_liquidations(symbol, limit)
                if not df.empty:
                    return df
            elif self.config['exchange'] == 'okx':
                df = self._fetch_okx_recent_liquidations(symbol, limit)
                if not df.empty:
                    return df
            elif self.config['exchange'] == 'alternative':
                return self._fetch_alternative_recent_data(symbol, limit)

            # If primary exchange failed, try alternative data sources
            self.logger.warning(f"Primary exchange {self.config['exchange']} failed, trying alternative data sources")
            return self._fetch_alternative_recent_data(symbol, limit)

        except Exception as e:
            self.logger.error(f"Failed to fetch recent liquidation data for {symbol}: {e}")
            # Final fallback to alternative data sources
            try:
                self.logger.info("Trying alternative data sources as final fallback")
                return self._fetch_alternative_recent_data(symbol, limit)
            except Exception as fallback_e:
                self.logger.error(f"All data sources failed: {fallback_e}")
                return pd.DataFrame()

    def _fetch_binance_recent_liquidations(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent liquidation data from Binance Futures API.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            limit: Number of recent liquidations

        Returns:
            DataFrame with liquidation data
        """
        try:
            # Binance endpoint for force orders (liquidations)
            endpoint = f"{self.config['api_base_url']}/fapi/v1/forceOrders"

            params = {
                'symbol': symbol.upper(),
                'limit': min(limit, 1000)  # Binance max is 1000
            }

            self.logger.info(f"Fetching recent liquidation data for {symbol} from Binance")

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
                self.logger.warning(f"No liquidation data received for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Rename columns to standard format
            column_mapping = {
                'updateTime': 'timestamp',
                'symbol': 'symbol',
                'price': 'price',
                'origQty': 'quantity',
                'averagePrice': 'average_price',
                'side': 'side'  # 'BUY' or 'SELL'
            }

            df = df.rename(columns=column_mapping)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Convert numeric columns
            df['price'] = df['price'].astype(float)
            df['quantity'] = df['quantity'].astype(float)
            df['average_price'] = df['average_price'].astype(float)

            # Set timestamp as index
            df = df.set_index('timestamp')

            # Sort by timestamp (most recent first)
            df = df.sort_index(ascending=False)

            # Add liquidation type based on side
            df['liquidation_type'] = df['side'].map({
                'BUY': 'long_liquidation',   # Long positions liquidated (bearish signal)
                'SELL': 'short_liquidation'  # Short positions liquidated (bullish signal)
            })

            # Calculate liquidation value
            df['liquidation_value'] = df['price'] * df['quantity']

            self.logger.info(f"Successfully fetched {len(df)} liquidation records for {symbol}")

            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch liquidation data for {symbol}: {e}")
            return pd.DataFrame()

    def _aggregate_binance_liquidations(self, symbol: str, timeframe: str,
                                       start_date: Optional[datetime] = None,
                                       end_date: Optional[datetime] = None,
                                       limit: Optional[int] = None) -> pd.DataFrame:
        """
        Aggregate liquidation data over time periods.

        Since Binance doesn't provide historical liquidation data,
        this method fetches recent data and aggregates it.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for aggregation
            start_date: Start date
            end_date: End date
            limit: Maximum number of aggregated periods

        Returns:
            DataFrame with aggregated liquidation data
        """
        # Fetch recent liquidations (max available)
        recent_df = self._fetch_binance_recent_liquidations(symbol, limit=1000)

        if recent_df.empty:
            return pd.DataFrame()

        # Resample/aggregate by timeframe
        # Convert timeframe to pandas frequency
        freq_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }

        freq = freq_map.get(timeframe, '1H')  # Default to 1 hour

        # Group by time periods and aggregate
        aggregated = recent_df.groupby(pd.Grouper(freq=freq)).agg({
            'quantity': 'sum',
            'liquidation_value': 'sum',
            'price': 'mean',  # Average price
            'liquidation_type': lambda x: x.value_counts().to_dict()  # Count by type
        }).rename(columns={
            'quantity': 'total_quantity_liquidated',
            'liquidation_value': 'total_value_liquidated',
            'price': 'avg_price',
            'liquidation_type': 'liquidation_breakdown'
        })

        # Add derived metrics
        aggregated['long_liquidations'] = aggregated['liquidation_breakdown'].apply(
            lambda x: x.get('long_liquidation', 0))
        aggregated['short_liquidations'] = aggregated['liquidation_breakdown'].apply(
            lambda x: x.get('short_liquidation', 0))
        aggregated['total_liquidations'] = aggregated['long_liquidations'] + aggregated['short_liquidations']

        # Calculate liquidation ratio (long/short ratio)
        aggregated['liquidation_ratio'] = aggregated.apply(
            lambda row: row['long_liquidations'] / max(row['short_liquidations'], 1), axis=1)

        # Add market signal based on liquidation activity
        aggregated['liquidation_signal'] = aggregated.apply(self._classify_liquidation_signal, axis=1)

        # Filter by date range if specified
        if start_date or end_date:
            if start_date:
                aggregated = aggregated[aggregated.index >= start_date]
            if end_date:
                aggregated = aggregated[aggregated.index <= end_date]

        # Apply limit
        if limit:
            aggregated = aggregated.tail(limit)

        return aggregated

    def _classify_liquidation_signal(self, row) -> str:
        """
        Classify market signal based on liquidation activity.

        Args:
            row: DataFrame row with liquidation data

        Returns:
            Signal classification
        """
        long_liq = row['long_liquidations']
        short_liq = row['short_liquidations']
        total_liq = row['total_liquidations']

        if total_liq == 0:
            return 'neutral'

        long_ratio = long_liq / total_liq
        short_ratio = short_liq / total_liq

        # High long liquidations = bearish signal
        if long_ratio > 0.7:
            return 'strong_bearish'
        elif long_ratio > 0.6:
            return 'bearish'
        elif short_ratio > 0.7:
            return 'strong_bullish'
        elif short_ratio > 0.6:
            return 'bullish'
        else:
            return 'neutral'

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate liquidation data.

        Args:
            df: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        if df.empty:
            return False

        # For recent data (individual liquidations)
        if 'liquidation_type' in df.columns:
            required_columns = ['price', 'quantity', 'liquidation_type']
        else:
            # For aggregated data
            required_columns = ['total_quantity_liquidated', 'total_value_liquidated']

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for negative values
        if 'price' in df.columns and (df['price'] <= 0).any():
            raise ValueError("Invalid price values found (must be positive)")

        if 'quantity' in df.columns and (df['quantity'] <= 0).any():
            raise ValueError("Invalid quantity values found (must be positive)")

        return True

    def get_liquidation_stats(self, symbol: str,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get liquidation statistics for analysis.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with liquidation statistics
        """
        df = self.fetch_recent_data(symbol, '1h', limit=1000)

        if df.empty:
            return {}

        stats = {
            'total_liquidations': len(df),
            'total_value_liquidated': df['liquidation_value'].sum(),
            'avg_liquidation_value': df['liquidation_value'].mean(),
            'max_liquidation_value': df['liquidation_value'].max(),
            'long_liquidations': (df['side'] == 'BUY').sum(),
            'short_liquidations': (df['side'] == 'SELL').sum(),
            'long_liquidation_value': df[df['side'] == 'BUY']['liquidation_value'].sum(),
            'short_liquidation_value': df[df['side'] == 'SELL']['liquidation_value'].sum(),
            'avg_price': df['price'].mean(),
            'price_volatility': df['price'].std(),
            'most_recent_liquidation': df.index.max().isoformat() if not df.empty else None
        }

        return stats

    def _fetch_bybit_recent_liquidations(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent liquidation data from Bybit API.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            limit: Number of recent liquidations

        Returns:
            DataFrame with liquidation data
        """
        try:
            # Bybit endpoint for liquidation data (public API)
            endpoint = "https://api.bybit.com/v5/market/recent-big-deal"

            params = {
                'category': 'linear',  # Futures
                'symbol': symbol.upper(),
                'limit': min(limit, 100)  # Bybit max is 100
            }

            self.logger.info(f"Fetching recent liquidation data for {symbol} from Bybit")

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

            if not data or data.get('retCode') != 0:
                self.logger.warning(f"No liquidation data received for {symbol} from Bybit")
                return pd.DataFrame()

            liquidation_data = data.get('result', {}).get('list', [])

            if not liquidation_data:
                self.logger.warning(f"No liquidation data received for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(liquidation_data)

            # Rename columns to standard format
            column_mapping = {
                'time': 'timestamp',
                'symbol': 'symbol',
                'price': 'price',
                'qty': 'quantity',
                'side': 'side'  # 'Buy' or 'Sell' in Bybit
            }

            df = df.rename(columns=column_mapping)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            # Convert numeric columns
            df['price'] = df['price'].astype(float)
            df['quantity'] = df['quantity'].astype(float)

            # Set timestamp as index
            df = df.set_index('timestamp')

            # Sort by timestamp (most recent first)
            df = df.sort_index(ascending=False)

            # Standardize side values (Bybit uses 'Buy'/'Sell', we want 'BUY'/'SELL')
            df['side'] = df['side'].str.upper()

            # Add liquidation type based on side
            df['liquidation_type'] = df['side'].map({
                'BUY': 'long_liquidation',   # Long positions liquidated (bearish signal)
                'SELL': 'short_liquidation'  # Short positions liquidated (bullish signal)
            })

            # Calculate liquidation value
            df['liquidation_value'] = df['price'] * df['quantity']

            self.logger.info(f"Successfully fetched {len(df)} liquidation records for {symbol} from Bybit")

            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch liquidation data for {symbol} from Bybit: {e}")
            return pd.DataFrame()

    def _aggregate_bybit_liquidations(self, symbol: str, timeframe: str,
                                     start_date: Optional[datetime] = None,
                                     end_date: Optional[datetime] = None,
                                     limit: Optional[int] = None) -> pd.DataFrame:
        """
        Aggregate Bybit liquidation data over time periods.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for aggregation
            start_date: Start date
            end_date: End date
            limit: Maximum number of aggregated periods

        Returns:
            DataFrame with aggregated liquidation data
        """
        # Fetch recent liquidations (max available)
        recent_df = self._fetch_bybit_recent_liquidations(symbol, limit=100)

        if recent_df.empty:
            return pd.DataFrame()

        # Resample/aggregate by timeframe
        # Convert timeframe to pandas frequency
        freq_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }

        freq = freq_map.get(timeframe, '1H')  # Default to 1 hour

        # Group by time periods and aggregate
        aggregated = recent_df.groupby(pd.Grouper(freq=freq)).agg({
            'quantity': 'sum',
            'liquidation_value': 'sum',
            'price': 'mean',  # Average price
            'liquidation_type': lambda x: x.value_counts().to_dict()  # Count by type
        }).rename(columns={
            'quantity': 'total_quantity_liquidated',
            'liquidation_value': 'total_value_liquidated',
            'price': 'avg_price',
            'liquidation_type': 'liquidation_breakdown'
        })

        # Add derived metrics
        aggregated['long_liquidations'] = aggregated['liquidation_breakdown'].apply(
            lambda x: x.get('long_liquidation', 0))
        aggregated['short_liquidations'] = aggregated['liquidation_breakdown'].apply(
            lambda x: x.get('short_liquidation', 0))
        aggregated['total_liquidations'] = aggregated['long_liquidations'] + aggregated['short_liquidations']

        # Calculate liquidation ratio (long/short ratio)
        aggregated['liquidation_ratio'] = aggregated.apply(
            lambda row: row['long_liquidations'] / max(row['short_liquidations'], 1), axis=1)

        # Add market signal based on liquidation activity
        aggregated['liquidation_signal'] = aggregated.apply(self._classify_liquidation_signal, axis=1)

        # Filter by date range if specified
        if start_date or end_date:
            if start_date:
                aggregated = aggregated[aggregated.index >= start_date]
            if end_date:
                aggregated = aggregated[aggregated.index <= end_date]

        # Apply limit
        if limit:
            aggregated = aggregated.tail(limit)

        return aggregated
    def _fetch_okx_recent_liquidations(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent liquidation data from OKX API.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT')
            limit: Number of recent liquidations

        Returns:
            DataFrame with liquidation data
        """
        try:
            # OKX endpoint for liquidation data (public API)
            endpoint = "https://www.okx.com/api/v5/public/liquidation-orders"

            # Convert symbol format (BTCUSDT -> BTC-USDT for OKX)
            okx_symbol = symbol.replace('USDT', '-USDT')

            params = {
                'instId': okx_symbol,
                'limit': min(limit, 100)  # OKX max is 100
            }

            self.logger.info(f"Fetching recent liquidation data for {symbol} from OKX")

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

            if not data or data.get('code') != '0':
                self.logger.warning(f"No liquidation data received for {symbol} from OKX")
                return pd.DataFrame()

            liquidation_data = data.get('data', [])

            if not liquidation_data:
                self.logger.warning(f"No liquidation data received for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(liquidation_data)

            # Rename columns to standard format
            column_mapping = {
                'ts': 'timestamp',
                'instId': 'symbol',
                'px': 'price',
                'sz': 'quantity',
                'side': 'side',  # 'long' or 'short' in OKX
                'posSide': 'position_side'
            }

            df = df.rename(columns=column_mapping)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Convert numeric columns
            df['price'] = df['price'].astype(float)
            df['quantity'] = df['quantity'].astype(float)

            # Set timestamp as index
            df = df.set_index('timestamp')

            # Sort by timestamp (most recent first)
            df = df.sort_index(ascending=False)

            # Standardize side values (OKX uses 'long'/'short', we want 'BUY'/'SELL')
            df['side'] = df['side'].map({
                'long': 'BUY',   # Long positions liquidated
                'short': 'SELL'  # Short positions liquidated
            })

            # Add liquidation type based on side
            df['liquidation_type'] = df['side'].map({
                'BUY': 'long_liquidation',   # Long positions liquidated (bearish signal)
                'SELL': 'short_liquidation'  # Short positions liquidated (bullish signal)
            })

            # Calculate liquidation value
            df['liquidation_value'] = df['price'] * df['quantity']

            self.logger.info(f"Successfully fetched {len(df)} liquidation records for {symbol} from OKX")

            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch liquidation data for {symbol} from OKX: {e}")
            return pd.DataFrame()

    def _aggregate_okx_liquidations(self, symbol: str, timeframe: str,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None,
                                   limit: Optional[int] = None) -> pd.DataFrame:
        """
        Aggregate OKX liquidation data over time periods.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for aggregation
            start_date: Start date
            end_date: End date
            limit: Maximum number of aggregated periods

        Returns:
            DataFrame with aggregated liquidation data
        """
        # Fetch recent liquidations (max available)
        recent_df = self._fetch_okx_recent_liquidations(symbol, limit=100)

        if recent_df.empty:
            return pd.DataFrame()

        # Resample/aggregate by timeframe
        # Convert timeframe to pandas frequency
        freq_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }

        freq = freq_map.get(timeframe, '1H')  # Default to 1 hour

        # Group by time periods and aggregate
        aggregated = recent_df.groupby(pd.Grouper(freq=freq)).agg({
            'quantity': 'sum',
            'liquidation_value': 'sum',
            'price': 'mean',  # Average price
            'liquidation_type': lambda x: x.value_counts().to_dict()  # Count by type
        }).rename(columns={
            'quantity': 'total_quantity_liquidated',
            'liquidation_value': 'total_value_liquidated',
            'price': 'avg_price',
            'liquidation_type': 'liquidation_breakdown'
        })

        # Add derived metrics
        aggregated['long_liquidations'] = aggregated['liquidation_breakdown'].apply(
            lambda x: x.get('long_liquidation', 0))
        aggregated['short_liquidations'] = aggregated['liquidation_breakdown'].apply(
            lambda x: x.get('short_liquidation', 0))
        aggregated['total_liquidations'] = aggregated['long_liquidations'] + aggregated['short_liquidations']

        # Calculate liquidation ratio (long/short ratio)
        aggregated['liquidation_ratio'] = aggregated.apply(
            lambda row: row['long_liquidations'] / max(row['short_liquidations'], 1), axis=1)

        # Add market signal based on liquidation activity
        aggregated['liquidation_signal'] = aggregated.apply(self._classify_liquidation_signal, axis=1)

        # Filter by date range if specified
        if start_date or end_date:
            if start_date:
                aggregated = aggregated[aggregated.index >= start_date]
            if end_date:
                aggregated = aggregated[aggregated.index <= end_date]

        # Apply limit
        if limit:
            aggregated = aggregated.tail(limit)

        return aggregated

    def _fetch_alternative_recent_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch alternative data sources as proxy for liquidation data.

        Tries multiple fallback data sources in order of preference:
        1. Volatility indicators (best proxy for liquidation activity)
        2. Sentiment analysis (market psychology proxy)
        3. On-chain metrics (network activity proxy)
        4. Open Interest (basic volatility proxy)

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            limit: Number of recent data points

        Returns:
            DataFrame with alternative data as liquidation proxy
        """
        from .factory import DataSourceFactory

        fallback_sources = [
            ('volatility', 'volatility_indicators'),
            ('sentiment', 'sentiment_analysis'),
            ('onchain', 'onchain_metrics')
        ]

        for source_name, source_type in fallback_sources:
            try:
                self.logger.info(f"Trying fallback data source: {source_name}")

                # Create the alternative data source
                alt_ds = DataSourceFactory.create_data_source(source_name)

                # Fetch recent data
                alt_df = alt_ds.fetch_recent_data(symbol, '1h', limit=limit)

                if not alt_df.empty:
                    self.logger.info(f"Successfully fetched {source_name} data as liquidation proxy")

                    # Convert alternative data to liquidation format
                    alt_df = self._convert_alternative_to_liquidation_format(alt_df, source_type)

                    return alt_df

            except Exception as e:
                self.logger.warning(f"Failed to fetch {source_name} data: {e}")
                continue

        # Final fallback: Open Interest data
        try:
            self.logger.info("Trying final fallback: Open Interest data")
            return self._fetch_open_interest_fallback(symbol, limit)
        except Exception as e:
            self.logger.error(f"All fallback data sources failed: {e}")
            return pd.DataFrame()

    def _convert_alternative_to_liquidation_format(self, df: pd.DataFrame, source_type: str) -> pd.DataFrame:
        """
        Convert alternative data source format to liquidation data format.

        Args:
            df: DataFrame from alternative data source
            source_type: Type of alternative data source

        Returns:
            DataFrame in liquidation format
        """
        df = df.copy()

        if source_type == 'volatility_indicators':
            # Convert volatility signals to liquidation format
            df['liquidation_signal'] = df.get('volatility_signal', 0)
            df['liquidation_volume'] = df.get('volatility_proxy', 0) * 1000  # Scale up
            df['total_value_liquidated'] = df['liquidation_volume']
            df['data_source'] = 'volatility_proxy'

        elif source_type == 'sentiment_analysis':
            # Convert sentiment signals to liquidation format
            sentiment_signal = df.get('sentiment_signal', 0)
            sentiment_strength = df.get('sentiment_strength', 0)

            # Extreme negative sentiment can indicate potential liquidations
            df['liquidation_signal'] = -sentiment_signal  # Invert: negative sentiment = bullish for liquidations
            df['liquidation_volume'] = abs(sentiment_signal) * sentiment_strength * 10000
            df['total_value_liquidated'] = df['liquidation_volume']
            df['data_source'] = 'sentiment_proxy'

        elif source_type == 'onchain_metrics':
            # Convert on-chain signals to liquidation format
            onchain_signal = df.get('onchain_signal', 0)
            onchain_strength = df.get('onchain_strength', 0)

            # Network stress can indicate liquidation activity
            df['liquidation_signal'] = onchain_signal
            df['liquidation_volume'] = abs(onchain_signal) * onchain_strength * 5000
            df['total_value_liquidated'] = df['liquidation_volume']
            df['data_source'] = 'onchain_proxy'

        else:
            # Default conversion
            df['liquidation_signal'] = 0
            df['liquidation_volume'] = 0
            df['total_value_liquidated'] = 0
            df['data_source'] = 'unknown_proxy'

        # Ensure required columns exist
        required_columns = ['liquidation_signal', 'liquidation_volume', 'total_value_liquidated', 'data_source']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0 if col != 'data_source' else 'unknown'

        return df

    def _fetch_open_interest_fallback(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch Open Interest data as final fallback for liquidation proxy.

        Args:
            symbol: Trading symbol
            limit: Number of data points

        Returns:
            DataFrame with Open Interest data as liquidation proxy
        """
        try:
            # Use Binance Open Interest data as final fallback
            endpoint = f"{self.config['api_base_url']}/fapi/v1/openInterest"

            params = {
                'symbol': symbol.upper()
            }

            self.logger.info(f"Fetching Open Interest data for {symbol} from Binance")

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
                self.logger.warning(f"No Open Interest data received for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame([data])  # Single data point

            # Rename columns to standard format
            column_mapping = {
                'symbol': 'symbol',
                'openInterest': 'open_interest',
                'time': 'timestamp'
            }

            df = df.rename(columns=column_mapping)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Convert numeric columns
            df['open_interest'] = df['open_interest'].astype(float)

            # Set timestamp as index
            df = df.set_index('timestamp')

            # Sort by timestamp (most recent first)
            df = df.sort_index(ascending=False)

            # Convert to liquidation format
            df['liquidation_signal'] = 0  # Neutral signal
            df['liquidation_volume'] = df['open_interest'] * 0.001  # Small proxy volume
            df['total_value_liquidated'] = df['liquidation_volume']
            df['data_source'] = 'open_interest_fallback'

            self.logger.info(f"Successfully fetched Open Interest data as liquidation proxy for {symbol}")

            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch Open Interest fallback data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_alternative_volatility_data(self, symbol: str, timeframe: str,
                                         start_date: Optional[datetime] = None,
                                         end_date: Optional[datetime] = None,
                                         limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch alternative volatility data over time periods.

        Since we can't get historical open interest easily, we'll create
        synthetic volatility signals based on market conditions.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for aggregation
            start_date: Start date
            end_date: End date
            limit: Maximum number of aggregated periods

        Returns:
            DataFrame with alternative volatility data
        """
        # For now, return empty DataFrame as we need historical data
        # This would need to be implemented with a data provider that offers historical OI
        self.logger.warning("Alternative historical volatility data not available - requires premium data source")
        return pd.DataFrame()

    def fetch_current_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch current liquidation data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')

        Returns:
            DataFrame with current liquidation data
        """
        try:
            # Try to fetch real liquidation data first
            current_data = self._fetch_binance_current_liquidations(symbol)

            if current_data:
                # Convert to DataFrame
                df = pd.DataFrame([current_data])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                logger.info(f"Fetched current liquidation data for {symbol}")
                return df
            else:
                # Fallback to alternative volatility data
                logger.info(f"No liquidation data available for {symbol}, trying alternative volatility...")
                alt_df = self._fetch_alternative_recent_data(symbol)

                if not alt_df.empty:
                    # Rename columns to match liquidation format
                    alt_df = alt_df.rename(columns={
                        'volatility_signal': 'liquidation_signal',
                        'liquidation_proxy': 'total_value_liquidated'
                    })

                    logger.info(f"Fetched alternative volatility data as liquidation proxy for {symbol}")
                    return alt_df
                else:
                    logger.warning(f"No alternative data available for {symbol}")
                    return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch current liquidation data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_binance_current_liquidations(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current liquidation data from Binance.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with current liquidation data or None if not available
        """
        try:
            # Get recent liquidations and aggregate for current snapshot
            df = self._fetch_binance_recent_liquidations(symbol, limit=50)

            if df.empty:
                return None

            # Aggregate recent liquidations
            total_long_liq = df[df['side'] == 'LONG']['quantity'].sum()
            total_short_liq = df[df['side'] == 'SHORT']['quantity'].sum()
            total_value = df['quantity'].sum()

            # Create signal based on liquidation imbalance
            if total_long_liq > total_short_liq * 1.5:
                signal = 1  # More long liquidations = bearish
            elif total_short_liq > total_long_liq * 1.5:
                signal = -1  # More short liquidations = bullish
            else:
                signal = 0

            return {
                'timestamp': int(datetime.now().timestamp() * 1000),
                'liquidation_signal': signal,
                'total_value_liquidated': total_value,
                'long_liquidations': total_long_liq,
                'short_liquidations': total_short_liq
            }

        except Exception as e:
            self.logger.error(f"Failed to fetch current liquidations: {e}")
            return None
