"""
Unit tests for SentimentDataSource.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from services.data_service.data_sources.sentiment import SentimentDataSource


class TestSentimentDataSource:
    """Test cases for SentimentDataSource."""

    @pytest.fixture
    def sentiment_source(self):
        """Create SentimentDataSource instance for testing."""
        return SentimentDataSource()

    def test_initialization(self, sentiment_source):
        """Test SentimentDataSource initialization."""
        assert sentiment_source.name == "Sentiment Analysis"
        assert sentiment_source.config['fear_greed_api'] == 'https://api.alternative.me/fng/'
        assert sentiment_source.config['timeout'] == 30

    def test_fetch_recent_data_success(self, sentiment_source):
        """Test successful fetch of recent sentiment data."""
        with patch.object(sentiment_source, '_fetch_current_sentiment') as mock_fetch:
            mock_fetch.return_value = {
                'timestamp': 1638360000,  # 2021-12-01
                'fear_greed_value': 25,
                'fear_greed_classification': 'Extreme Fear',
                'social_sentiment': -0.8,
                'news_sentiment': -0.6,
                'market_sentiment': -0.75
            }

            result = sentiment_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

            assert not result.empty
            assert len(result) == 1
            assert 'sentiment_signal' in result.columns
            assert 'sentiment_strength' in result.columns
            assert 'market_sentiment_proxy' in result.columns
            assert result.iloc[0]['sentiment_signal'] == -3  # Extreme fear signal

    def test_fetch_recent_data_failure(self, sentiment_source):
        """Test fetch of recent sentiment data when API fails."""
        with patch.object(sentiment_source, '_fetch_current_sentiment') as mock_fetch:
            mock_fetch.return_value = None

            result = sentiment_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

            assert result.empty

    def test_fetch_historical_data(self, sentiment_source):
        """Test fetch of historical sentiment data."""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        result = sentiment_source.fetch_historical_data(
            'BTC/USDT', '1h', start_date, end_date, limit=24
        )

        assert not result.empty
        assert len(result) <= 24
        assert 'sentiment_signal' in result.columns
        assert 'sentiment_strength' in result.columns
        assert result.index.name == 'timestamp'

    @patch('services.data_service.data_sources.sentiment.requests.get')
    def test_fetch_fear_greed_index_success(self, mock_get, sentiment_source):
        """Test successful Fear & Greed Index API call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'data': [{
                'timestamp': '1638360000',
                'value': '25',
                'value_classification': 'Extreme Fear'
            }]
        }
        mock_get.return_value = mock_response

        result = sentiment_source._fetch_fear_greed_index()

        assert result is not None
        assert result['timestamp'] == 1638360000
        assert result['value'] == 25
        assert result['value_classification'] == 'Extreme Fear'

    @patch('services.data_service.data_sources.sentiment.requests.get')
    def test_fetch_fear_greed_index_failure(self, mock_get, sentiment_source):
        """Test Fear & Greed Index API call failure."""
        mock_get.side_effect = Exception("API Error")

        result = sentiment_source._fetch_fear_greed_index()

        assert result is None

    def test_simulate_social_sentiment(self, sentiment_source):
        """Test social sentiment simulation."""
        # Extreme Fear
        result = sentiment_source._simulate_social_sentiment(15)
        assert -0.8 <= result <= -0.5

        # Greed
        result = sentiment_source._simulate_social_sentiment(85)
        assert 0.2 <= result <= 0.5

        # Neutral
        result = sentiment_source._simulate_social_sentiment(50)
        assert 0.2 <= result <= 0.8

    def test_simulate_news_sentiment(self, sentiment_source):
        """Test news sentiment simulation."""
        # Fear
        result = sentiment_source._simulate_news_sentiment(20)
        assert result < -0.4

        # Greed
        result = sentiment_source._simulate_news_sentiment(80)
        assert result > 0.4

    def test_calculate_market_sentiment(self, sentiment_source):
        """Test market sentiment calculation."""
        # Extreme Fear (0) -> -1
        result = sentiment_source._calculate_market_sentiment(0)
        assert result == -1.0

        # Extreme Greed (100) -> 1
        result = sentiment_source._calculate_market_sentiment(100)
        assert result == 1.0

        # Neutral (50) -> 0
        result = sentiment_source._calculate_market_sentiment(50)
        assert result == 0.0

    def test_calculate_sentiment_signals(self, sentiment_source):
        """Test sentiment signal calculation."""
        data = pd.DataFrame({
            'fear_greed_value': [25, 75, 50],
            'social_sentiment': [-0.8, 0.5, 0.0],
            'news_sentiment': [-0.6, 0.6, 0.0],
            'market_sentiment': [-0.75, 0.75, 0.0]
        })

        result = sentiment_source._calculate_sentiment_signals(data)

        assert 'sentiment_signal' in result.columns
        assert 'sentiment_strength' in result.columns
        assert 'market_sentiment_proxy' in result.columns

        # Check extreme fear signal
        assert result.iloc[0]['sentiment_signal'] == -3
        assert result.iloc[0]['sentiment_strength'] > 0.8

        # Check extreme greed signal
        assert result.iloc[1]['sentiment_signal'] == 3
        assert result.iloc[1]['sentiment_strength'] > 0.8

    def test_validate_data_valid(self, sentiment_source):
        """Test data validation with valid data."""
        data = pd.DataFrame({
            'sentiment_signal': [1, -1, 0],
            'sentiment_strength': [0.8, 0.6, 0.3],
            'market_sentiment_proxy': [50, -25, 0]
        })

        result = sentiment_source.validate_data(data)
        assert result is True

    def test_validate_data_missing_columns(self, sentiment_source):
        """Test data validation with missing columns."""
        data = pd.DataFrame({
            'some_other_column': [1, 2, 3]
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            sentiment_source.validate_data(data)

    def test_get_sentiment_stats(self, sentiment_source):
        """Test sentiment statistics retrieval."""
        with patch.object(sentiment_source, '_fetch_current_sentiment') as mock_fetch:
            mock_fetch.return_value = {
                'timestamp': 1638360000,
                'fear_greed_value': 25,
                'fear_greed_classification': 'Extreme Fear',
                'social_sentiment': -0.8,
                'news_sentiment': -0.6,
                'market_sentiment': -0.75
            }

            stats = sentiment_source.get_sentiment_stats('BTC/USDT')

            assert 'current_sentiment_signal' in stats
            assert 'current_sentiment_strength' in stats
            assert 'fear_greed_value' in stats
            assert 'fear_greed_classification' in stats
            assert stats['fear_greed_value'] == 25
            assert stats['fear_greed_classification'] == 'Extreme Fear'

    def test_generate_fallback_sentiment(self, sentiment_source):
        """Test fallback sentiment generation."""
        result = sentiment_source._generate_fallback_sentiment()

        assert result['fear_greed_value'] == 50
        assert result['fear_greed_classification'] == 'Neutral'
        assert result['social_sentiment'] == 0.0
        assert result['news_sentiment'] == 0.0
        assert result['market_sentiment'] == 0.0