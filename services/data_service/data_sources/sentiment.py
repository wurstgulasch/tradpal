"""
Sentiment Data Source for TradPal Indicator System

This module provides sentiment analysis data that can serve as
additional market signals when other data sources are unavailable.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import math
import requests

from .base import BaseDataSource

logger = logging.getLogger(__name__)

class SentimentDataSource(BaseDataSource):
    """
    Data source for sentiment analysis that can serve as market signal proxy.

    Provides alternative market sentiment signals when other data sources fail:
    - Fear & Greed Index
    - Social media sentiment (simulated)
    - News sentiment (simulated)
    - Market sentiment indicators
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Sentiment data source.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__("Sentiment Analysis", config)

        # Default configuration
        self.config.setdefault('fear_greed_api', 'https://api.alternative.me/fng/')
        self.config.setdefault('timeout', 30)  # Request timeout
        self.config.setdefault('max_retries', 3)  # Max retry attempts
        self.config.setdefault('retry_delay', 1)  # Delay between retries

    def fetch_historical_data(self, symbol: str, timeframe: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical sentiment data.

        Args:
            symbol: Trading symbol (used for context, not directly relevant for sentiment)
            timeframe: Timeframe string (used for aggregation period)
            start_date: Start date for historical data
            end_date: End date for historical data
            limit: Maximum number of data points

        Returns:
            DataFrame with sentiment indicators
        """
        try:
            # Fetch current sentiment and create synthetic historical data
            current_sentiment = self._fetch_current_sentiment()

            if not current_sentiment:
                logger.warning("No sentiment data available")
                return pd.DataFrame()

            # Create synthetic historical sentiment data
            sentiment_data = self._generate_historical_sentiment(
                current_sentiment, start_date, end_date, limit
            )

            return sentiment_data

        except Exception as e:
            logger.error(f"Failed to fetch historical sentiment data: {e}")
            return pd.DataFrame()

    def fetch_recent_data(self, symbol: str, timeframe: str,
                         limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent sentiment data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (ignored)
            limit: Number of recent data points

        Returns:
            DataFrame with current sentiment data
        """
        try:
            current_sentiment = self._fetch_current_sentiment()

            if current_sentiment:
                # Convert to DataFrame
                df = pd.DataFrame([current_sentiment])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                if df['timestamp'].dt.tz is not None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                df = df.set_index('timestamp')

                # Calculate sentiment signals
                df = self._calculate_sentiment_signals(df)

                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch recent sentiment data: {e}")
            return pd.DataFrame()

    def _fetch_current_sentiment(self) -> Optional[Dict]:
        """
        Fetch current market sentiment data.

        Returns:
            Dictionary with current sentiment data
        """
        try:
            # Fetch Fear & Greed Index
            fear_greed_data = self._fetch_fear_greed_index()

            if fear_greed_data:
                # Create comprehensive sentiment data
                sentiment_data = {
                    'timestamp': fear_greed_data['timestamp'],
                    'fear_greed_value': fear_greed_data['value'],
                    'fear_greed_classification': fear_greed_data['value_classification'],
                    # Simulated additional sentiment indicators
                    'social_sentiment': self._simulate_social_sentiment(fear_greed_data['value']),
                    'news_sentiment': self._simulate_news_sentiment(fear_greed_data['value']),
                    'market_sentiment': self._calculate_market_sentiment(fear_greed_data['value'])
                }

                return sentiment_data
            else:
                # Fallback to simulated sentiment if API fails
                return self._generate_fallback_sentiment()

        except Exception as e:
            logger.warning(f"Failed to fetch current sentiment: {e}")
            return self._generate_fallback_sentiment()

    def _fetch_fear_greed_index(self) -> Optional[Dict]:
        """
        Fetch Fear & Greed Index from Alternative.me API.

        Returns:
            Dictionary with Fear & Greed Index data
        """
        try:
            params = {'limit': 1}
            response = requests.get(self.config['fear_greed_api'], params=params,
                                  timeout=self.config['timeout'])
            response.raise_for_status()
            data = response.json()

            if data.get('data') and len(data['data']) > 0:
                latest = data['data'][0]
                return {
                    'timestamp': int(latest['timestamp']),
                    'value': int(latest['value']),
                    'value_classification': latest['value_classification']
                }

        except Exception as e:
            logger.warning(f"Failed to fetch Fear & Greed Index: {e}")
            return None

    def _simulate_social_sentiment(self, fear_greed_value: int) -> float:
        """
        Simulate social media sentiment based on Fear & Greed Index.

        Args:
            fear_greed_value: Fear & Greed Index value (0-100)

        Returns:
            Social sentiment score (-1 to 1)
        """
        # Social sentiment tends to be more extreme than Fear & Greed
        if fear_greed_value < 25:  # Extreme Fear
            return -0.8 + (fear_greed_value / 25) * 0.3  # -0.8 to -0.5
        elif fear_greed_value < 45:  # Fear
            return -0.5 + ((fear_greed_value - 25) / 20) * 0.7  # -0.5 to 0.2
        elif fear_greed_value < 55:  # Neutral
            return 0.2 + ((fear_greed_value - 45) / 10) * 0.6  # 0.2 to 0.8
        elif fear_greed_value < 75:  # Greed
            return 0.8 - ((fear_greed_value - 55) / 20) * 0.3  # 0.8 to 0.5
        else:  # Extreme Greed
            return 0.5 - ((fear_greed_value - 75) / 25) * 0.3  # 0.5 to 0.2

    def _simulate_news_sentiment(self, fear_greed_value: int) -> float:
        """
        Simulate news sentiment based on Fear & Greed Index.

        Args:
            fear_greed_value: Fear & Greed Index value (0-100)

        Returns:
            News sentiment score (-1 to 1)
        """
        # News sentiment is more measured but follows market sentiment
        if fear_greed_value < 30:  # Fear
            return -0.6
        elif fear_greed_value < 70:  # Neutral/Greed
            return (fear_greed_value - 30) / 40 * 1.2 - 0.6  # -0.6 to 0.6
        else:  # Extreme Greed
            return 0.6

    def _calculate_market_sentiment(self, fear_greed_value: int) -> float:
        """
        Calculate overall market sentiment.

        Args:
            fear_greed_value: Fear & Greed Index value (0-100)

        Returns:
            Market sentiment score (-1 to 1)
        """
        # Convert Fear & Greed (0-100) to sentiment (-1 to 1)
        # 0 = Extreme Fear = -1 (bearish)
        # 100 = Extreme Greed = 1 (bullish)
        return (fear_greed_value - 50) / 50

    def _generate_fallback_sentiment(self) -> Dict:
        """
        Generate fallback sentiment data when API is unavailable.

        Returns:
            Dictionary with simulated sentiment data
        """
        # Create neutral sentiment as fallback
        return {
            'timestamp': int(datetime.now().timestamp()),
            'fear_greed_value': 50,  # Neutral
            'fear_greed_classification': 'Neutral',
            'social_sentiment': 0.0,
            'news_sentiment': 0.0,
            'market_sentiment': 0.0
        }

    def _calculate_sentiment_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment-based signals.

        Args:
            df: DataFrame with sentiment data

        Returns:
            DataFrame with calculated signals
        """
        # Initialize signal columns
        df['sentiment_signal'] = 0
        df['sentiment_strength'] = 0.0
        df['market_sentiment_proxy'] = 0.0

        for idx in range(len(df)):
            row = df.iloc[idx]
            signal = 0
            strength = 0.0
            proxy = 0.0

            # Fear & Greed Index signals
            fg_value = row.get('fear_greed_value', 50)
            if fg_value < 25:  # Extreme Fear
                signal -= 2
                strength += 0.4
                proxy -= 100
            elif fg_value < 45:  # Fear
                signal -= 1
                strength += 0.3
                proxy -= 50
            elif fg_value > 75:  # Extreme Greed
                signal += 2
                strength += 0.4
                proxy += 100
            elif fg_value > 55:  # Greed
                signal += 1
                strength += 0.3
                proxy += 50

            # Social sentiment signals
            social_sentiment = row.get('social_sentiment', 0)
            if abs(social_sentiment) > 0.5:
                signal += 1 if social_sentiment > 0.5 else -1
                strength += 0.2
                proxy += social_sentiment * 50

            # News sentiment signals
            news_sentiment = row.get('news_sentiment', 0)
            if abs(news_sentiment) > 0.4:
                signal += 1 if news_sentiment > 0.4 else -1
                strength += 0.3
                proxy += news_sentiment * 75

            # Market sentiment signals
            market_sentiment = row.get('market_sentiment', 0)
            if abs(market_sentiment) > 0.3:
                signal += 1 if market_sentiment > 0.3 else -1
                strength += 0.3
                proxy += market_sentiment * 80

            # Normalize signal
            signal = max(-3, min(3, signal))
            strength = min(1.0, strength)

            df.iloc[idx, df.columns.get_loc('sentiment_signal')] = signal
            df.iloc[idx, df.columns.get_loc('sentiment_strength')] = strength
            df.iloc[idx, df.columns.get_loc('market_sentiment_proxy')] = proxy

        return df

    def _generate_historical_sentiment(self, current_sentiment: Dict,
                                     start_date: Optional[datetime] = None,
                                     end_date: Optional[datetime] = None,
                                     limit: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic historical sentiment data.

        Args:
            current_sentiment: Current sentiment data
            start_date: Start date
            end_date: End date
            limit: Maximum number of data points

        Returns:
            DataFrame with historical sentiment data
        """
        synthetic_data = []

        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        if start_date is None:
            start_date = end_date - timedelta(days=30)
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)

        # Generate data points every 4 hours
        current_time = end_date
        step_hours = 4

        base_fg = current_sentiment['fear_greed_value']

        while current_time >= start_date:
            # Add some realistic variation to sentiment
            time_factor = (pd.Timestamp(current_time).timestamp() % 86400) / 86400  # Daily cycle
            variation = 0.8 + 0.4 * (0.5 + 0.5 * math.sin(time_factor * 2 * math.pi))  # 0.8-1.2 range

            fg_value = max(0, min(100, int(base_fg * variation)))

            synthetic_row = {
                'timestamp': current_time,
                'fear_greed_value': fg_value,
                'fear_greed_classification': self._classify_fear_greed(fg_value),
                'social_sentiment': self._simulate_social_sentiment(fg_value),
                'news_sentiment': self._simulate_news_sentiment(fg_value),
                'market_sentiment': self._calculate_market_sentiment(fg_value)
            }
            synthetic_data.append(synthetic_row)

            current_time -= timedelta(hours=step_hours)

        # Convert to DataFrame
        df = pd.DataFrame(synthetic_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.sort_index()

        # Calculate signals
        df = self._calculate_sentiment_signals(df)

        # Apply limit
        if limit:
            df = df.tail(limit)

        return df

    def _classify_fear_greed(self, value: int) -> str:
        """
        Classify Fear & Greed Index value.

        Args:
            value: Fear & Greed Index value

        Returns:
            Classification string
        """
        if value <= 25:
            return "Extreme Fear"
        elif value <= 45:
            return "Fear"
        elif value <= 55:
            return "Neutral"
        elif value <= 75:
            return "Greed"
        else:
            return "Extreme Greed"

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate sentiment data.

        Args:
            df: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        if df.empty:
            return False

        required_columns = ['sentiment_signal', 'sentiment_strength']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return True

    def get_sentiment_stats(self, symbol: str) -> Dict[str, Any]:
        """
        Get current sentiment statistics.

        Args:
            symbol: Trading symbol (for context)

        Returns:
            Dictionary with sentiment statistics
        """
        df = self.fetch_recent_data(symbol, '1h', limit=1)

        if df.empty:
            return {}

        latest = df.iloc[0]

        stats = {
            'current_sentiment_signal': latest.get('sentiment_signal', 0),
            'current_sentiment_strength': latest.get('sentiment_strength', 0),
            'fear_greed_value': latest.get('fear_greed_value', 50),
            'fear_greed_classification': latest.get('fear_greed_classification', 'Neutral'),
            'social_sentiment': latest.get('social_sentiment', 0),
            'news_sentiment': latest.get('news_sentiment', 0),
            'market_sentiment': latest.get('market_sentiment', 0),
            'market_sentiment_proxy': latest.get('market_sentiment_proxy', 0),
            'timestamp': latest.name.isoformat() if hasattr(latest, 'name') else None
        }

        return stats