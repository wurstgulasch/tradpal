"""
Tests for Sentiment Analysis Module

Tests the sentiment analysis functionality including:
- SentimentAnalyzer class
- SentimentData dataclass
- Convenience functions
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import sentiment analyzer components
try:
    from sentiment_analyzer import SentimentAnalyzer, SentimentData, get_current_sentiment, get_sentiment_score
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    pytest.skip("Sentiment analysis module not available", allow_module_level=True)


class TestSentimentData:
    """Test suite for SentimentData dataclass."""

    def setup_method(self):
        """Skip test if sentiment analysis is not available."""
        if not SENTIMENT_AVAILABLE:
            pytest.skip("Sentiment analysis module not available")

    def test_sentiment_data_creation(self):
        """Test SentimentData object creation."""
        data = SentimentData(
            timestamp=datetime.now(),
            source="test",
            sentiment_score=0.7,
            confidence=0.85,
            text_sample="Test text",
            metadata={"test": "data"}
        )

        assert data.sentiment_score == 0.7
        assert data.confidence == 0.85
        assert data.source == "test"
        assert data.text_sample == "Test text"
        assert data.metadata == {"test": "data"}
        assert isinstance(data.timestamp, datetime)

    def test_sentiment_data_with_extreme_values(self):
        """Test SentimentData with extreme values."""
        # Test positive sentiment
        positive_data = SentimentData(
            timestamp=datetime.now(),
            source="twitter",
            sentiment_score=1.0,
            confidence=1.0,
            text_sample="Amazing bullish signal!",
            metadata={"likes": 1000}
        )
        assert positive_data.sentiment_score == 1.0
        assert positive_data.confidence == 1.0

        # Test negative sentiment
        negative_data = SentimentData(
            timestamp=datetime.now(),
            source="news",
            sentiment_score=-1.0,
            confidence=0.9,
            text_sample="Market crash incoming",
            metadata={"source": "reuters"}
        )
        assert negative_data.sentiment_score == -1.0
        assert negative_data.confidence == 0.9


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()

    @patch('sentiment_analyzer.TwitterSentimentClient')
    @patch('sentiment_analyzer.NewsSentimentClient')
    @patch('sentiment_analyzer.RedditSentimentClient')
    def test_sentiment_analyzer_initialization(self, mock_reddit, mock_news, mock_twitter):
        """Test SentimentAnalyzer initialization."""
        analyzer = SentimentAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'get_aggregated_sentiment')
        assert hasattr(analyzer, 'get_weighted_sentiment_score')

    @patch('sentiment_analyzer.TwitterSentimentClient')
    @patch('sentiment_analyzer.NewsSentimentClient')
    @patch('sentiment_analyzer.RedditSentimentClient')
    def test_get_aggregated_sentiment(self, mock_reddit, mock_news, mock_twitter):
        """Test getting aggregated sentiment."""
        # Mock the clients
        mock_twitter.return_value.get_sentiment.return_value = {
            'sentiment_score': 0.8,
            'confidence': 0.9,
            'source': 'twitter'
        }
        mock_news.return_value.get_sentiment.return_value = {
            'sentiment_score': 0.6,
            'confidence': 0.8,
            'source': 'news'
        }
        mock_reddit.return_value.get_sentiment.return_value = {
            'sentiment_score': 0.4,
            'confidence': 0.7,
            'source': 'reddit'
        }

        result = self.analyzer.get_aggregated_sentiment()
        assert isinstance(result, dict)
        assert 'score' in result
        assert 'confidence' in result
        assert 'enabled' in result

    def test_get_weighted_sentiment_score(self):
        """Test getting weighted sentiment score."""
        score, confidence = self.analyzer.get_weighted_sentiment_score()
        assert isinstance(score, (float, int))
        assert isinstance(confidence, (float, int))
        assert -1 <= score <= 1
        assert 0 <= confidence <= 1


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def setup_method(self):
        """Skip test if sentiment analysis is not available."""
        if not SENTIMENT_AVAILABLE:
            pytest.skip("Sentiment analysis module not available")

    @pytest.mark.asyncio
    async def test_get_current_sentiment(self):
        """Test get_current_sentiment function."""
        result = await get_current_sentiment()
        assert isinstance(result, dict)
        assert 'enabled' in result
        assert 'score' in result
        assert 'confidence' in result
        assert isinstance(result['enabled'], bool)
        assert isinstance(result['score'], (int, float))
        assert isinstance(result['confidence'], (int, float))

    def test_get_sentiment_score(self):
        """Test get_sentiment_score function."""
        score, confidence = get_sentiment_score()
        assert isinstance(score, (int, float))
        assert isinstance(confidence, (int, float))
        assert -1 <= score <= 1
        assert 0 <= confidence <= 1


class TestSentimentIntegration:
    """Integration tests for sentiment analysis."""

    def setup_method(self):
        """Skip test if sentiment analysis is not available."""
        if not SENTIMENT_AVAILABLE:
            pytest.skip("Sentiment analysis module not available")

    def test_sentiment_data_integration(self):
        """Test SentimentData in integration scenario."""
        # Create multiple sentiment data points
        data_points = [
            SentimentData(
                timestamp=datetime.now(),
                source="twitter",
                sentiment_score=0.8,
                confidence=0.9,
                text_sample="BTC to the moon!",
                metadata={"likes": 150}
            ),
            SentimentData(
                timestamp=datetime.now(),
                source="news",
                sentiment_score=-0.3,
                confidence=0.7,
                text_sample="Bitcoin volatility increases",
                metadata={"source": "bloomberg"}
            )
        ]

        # Test aggregation logic
        total_score = sum(data.sentiment_score * data.confidence for data in data_points)
        total_confidence = sum(data.confidence for data in data_points)
        weighted_average = total_score / total_confidence if total_confidence > 0 else 0

        assert len(data_points) == 2
        assert -1 <= weighted_average <= 1