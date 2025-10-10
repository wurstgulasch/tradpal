"""
Tests for Sentiment Analysis Module

Tests the sentiment analysis functionality including:
- SentimentAnalyzer class
- SentimentResult dataclass
- Convenience functions
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentResult,
    get_btc_sentiment,
    get_sentiment_signal
)


class TestSentimentResult:
    """Test suite for SentimentResult dataclass."""

    def test_sentiment_result_creation(self):
        """Test SentimentResult object creation."""
        result = SentimentResult(
            timestamp=datetime.now(),
            symbol="BTC",
            sentiment_score=0.7,
            confidence=0.85,
            source="test",
            text_sample="Test text",
            volume=100,
            metadata={"test": "data"}
        )

        assert result.symbol == "BTC"
        assert result.sentiment_score == 0.7
        assert result.confidence == 0.85
        assert result.source == "test"
        assert result.text_sample == "Test text"
        assert result.volume == 100
        assert result.metadata == {"test": "data"}
        assert isinstance(result.timestamp, datetime)


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache_manager = Mock()
        self.analyzer = SentimentAnalyzer(cache_manager=self.cache_manager)

    def test_analyzer_initialization(self):
        """Test SentimentAnalyzer initialization."""
        assert self.analyzer.cache_manager == self.cache_manager
        assert hasattr(self.analyzer, 'api_keys')
        assert isinstance(self.analyzer.api_keys, dict)

    @patch('src.sentiment_analyzer.TextBlob')
    def test_basic_sentiment_analysis(self, mock_textblob):
        """Test basic sentiment analysis fallback."""
        # Mock TextBlob
        mock_blob = Mock()
        mock_blob.sentiment.polarity = 0.5
        mock_textblob.return_value = mock_blob

        # Test with bullish text (should be positive)
        score = self.analyzer._basic_sentiment_analysis("This is bullish!")
        assert score == 1.0  # (1 positive - 0 negative) / 1 total = 1.0

        # Test with bearish text (should be negative)
        score = self.analyzer._basic_sentiment_analysis("This is bearish!")
        assert score == -1.0  # (0 positive - 1 negative) / 1 total = -1.0

        # Test with neutral text
        score = self.analyzer._basic_sentiment_analysis("This is neutral news")
        assert score == 0.0  # (0 positive - 0 negative) / 0 total = 0.0

    def test_sentiment_signal_generation(self):
        """Test trading signal generation from sentiment."""
        # Create a mock sentiment result
        result = SentimentResult(
            timestamp=datetime.now(),
            symbol="BTC",
            sentiment_score=0.8,
            confidence=0.9,
            source="test",
            text_sample="Bullish news",
            volume=100,
            metadata={}
        )

        signal = self.analyzer.get_sentiment_signal(result, threshold=0.5)
        assert signal == 'BUY'

        # Test bearish signal
        result.sentiment_score = -0.8
        signal = self.analyzer.get_sentiment_signal(result, threshold=0.5)
        assert signal == 'SELL'

        # Test neutral signal
        result.sentiment_score = 0.1
        signal = self.analyzer.get_sentiment_signal(result, threshold=0.5)
        assert signal is None

        # Test low confidence
        result.confidence = 0.2
        result.sentiment_score = 0.8
        signal = self.analyzer.get_sentiment_signal(result, threshold=0.5)
        assert signal is None


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    @patch('src.sentiment_analyzer.SentimentAnalyzer')
    def test_get_btc_sentiment(self, mock_analyzer_class):
        """Test get_btc_sentiment convenience function."""
        mock_analyzer = Mock()
        mock_result = SentimentResult(
            timestamp=datetime.now(),
            symbol="BTC",
            sentiment_score=0.6,
            confidence=0.8,
            source="aggregated",
            text_sample="BTC sentiment",
            volume=50,
            metadata={}
        )
        mock_analyzer.get_aggregated_sentiment.return_value = mock_result
        mock_analyzer_class.return_value = mock_analyzer

        result = get_btc_sentiment(hours_back=12)

        assert result == mock_result
        mock_analyzer.get_aggregated_sentiment.assert_called_once_with('BTC', 12)

    @patch('src.sentiment_analyzer.SentimentAnalyzer')
    def test_get_sentiment_signal(self, mock_analyzer_class):
        """Test get_sentiment_signal convenience function."""
        mock_analyzer = Mock()
        mock_analyzer.get_sentiment_signal.return_value = 'BUY'
        mock_analyzer_class.return_value = mock_analyzer

        signal = get_sentiment_signal('ETH', hours_back=24, threshold=0.3)

        assert signal == 'BUY'
        mock_analyzer.get_sentiment_signal.assert_called_once()


class TestSentimentIntegration:
    """Integration tests for sentiment analysis."""

    def test_sentiment_analyzer_interface(self):
        """Test that SentimentAnalyzer has expected interface."""
        analyzer = SentimentAnalyzer()

        # Test that analyzer has expected methods
        assert hasattr(analyzer, 'get_aggregated_sentiment')
        assert hasattr(analyzer, 'get_sentiment_signal')
        assert hasattr(analyzer, '_basic_sentiment_analysis')

    def test_sentiment_result_interface(self):
        """Test SentimentResult dataclass interface."""
        result = SentimentResult(
            timestamp=datetime.now(),
            symbol="TEST",
            sentiment_score=0.0,
            confidence=0.0,
            source="test",
            text_sample="",
            volume=0,
            metadata={}
        )

        # Test that all expected attributes exist
        assert hasattr(result, 'timestamp')
        assert hasattr(result, 'symbol')
        assert hasattr(result, 'sentiment_score')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'source')
        assert hasattr(result, 'text_sample')
        assert hasattr(result, 'volume')
        assert hasattr(result, 'metadata')