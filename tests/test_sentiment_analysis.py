"""
Tests for Sentiment Analysis Module

Tests the sentiment analysis functionality including:
- Multi-source sentiment aggregation
- API integrations (Twitter, Reddit, News)
- Sentiment scoring and confidence
- Caching and error handling
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from src.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentSource,
    SentimentScore,
    SentimentAggregator,
    TwitterSentimentSource,
    RedditSentimentSource,
    NewsSentimentSource
)


class TestSentimentScore:
    """Test suite for SentimentScore class."""

    def test_sentiment_score_creation(self):
        """Test SentimentScore object creation."""
        score = SentimentScore(
            score=0.7,
            confidence=0.85,
            source="twitter",
            timestamp=datetime.now(),
            text="Great news about BTC!",
            metadata={"mentions": 100}
        )

        assert score.score == 0.7
        assert score.confidence == 0.85
        assert score.source == "twitter"
        assert isinstance(score.timestamp, datetime)
        assert score.text == "Great news about BTC!"
        assert score.metadata == {"mentions": 100}

    def test_sentiment_score_validation(self):
        """Test SentimentScore validation."""
        # Valid score
        score = SentimentScore(score=0.5, confidence=0.8, source="test")
        assert score.score == 0.5

        # Score out of range should raise error
        with pytest.raises(ValueError):
            SentimentScore(score=1.5, confidence=0.8, source="test")

        with pytest.raises(ValueError):
            SentimentScore(score=-0.5, confidence=0.8, source="test")

        # Confidence out of range should raise error
        with pytest.raises(ValueError):
            SentimentScore(score=0.5, confidence=1.2, source="test")


class TestSentimentSource:
    """Test suite for SentimentSource base class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.source = SentimentSource(name="test_source", api_key="test_key")

    def test_source_initialization(self):
        """Test SentimentSource initialization."""
        assert self.source.name == "test_source"
        assert self.source.api_key == "test_key"
        assert not self.source.is_available

    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            self.source.fetch_sentiment("BTC")

        with pytest.raises(NotImplementedError):
            self.source.is_available = True


class TestTwitterSentimentSource:
    """Test suite for TwitterSentimentSource."""

    def setup_method(self):
        """Set up test fixtures."""
        self.source = TwitterSentimentSource(api_key="test_key")

    @patch('src.sentiment_analyzer.tweepy')
    def test_twitter_initialization(self, mock_tweepy):
        """Test Twitter source initialization."""
        mock_api = Mock()
        mock_tweepy.API.return_value = mock_api

        self.source.initialize()
        assert self.source.is_available
        mock_tweepy.API.assert_called_once()

    @patch('src.sentiment_analyzer.tweepy')
    def test_fetch_twitter_sentiment(self, mock_tweepy):
        """Test fetching sentiment from Twitter."""
        # Mock Twitter API
        mock_api = Mock()
        mock_tweepy.API.return_value = mock_api

        # Mock tweets
        mock_tweets = [
            Mock(full_text="BTC is mooning! 🚀🚀🚀", created_at=datetime.now(), retweet_count=50),
            Mock(full_text="BTC crash incoming", created_at=datetime.now(), retweet_count=20),
            Mock(full_text="Neutral BTC news", created_at=datetime.now(), retweet_count=10)
        ]
        mock_api.search_tweets.return_value = mock_tweets

        self.source.initialize()
        scores = self.source.fetch_sentiment("BTC", limit=10)

        assert len(scores) == 3
        assert all(isinstance(score, SentimentScore) for score in scores)
        assert all(score.source == "twitter" for score in scores)

    @patch('src.sentiment_analyzer.tweepy')
    def test_twitter_error_handling(self, mock_tweepy):
        """Test Twitter API error handling."""
        mock_tweepy.API.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            self.source.initialize()

        # Should return empty list on fetch error
        scores = self.source.fetch_sentiment("BTC")
        assert scores == []


class TestRedditSentimentSource:
    """Test suite for RedditSentimentSource."""

    def setup_method(self):
        """Set up test fixtures."""
        self.source = RedditSentimentSource(client_id="test_id", client_secret="test_secret")

    @patch('src.sentiment_analyzer.praw')
    def test_reddit_initialization(self, mock_praw):
        """Test Reddit source initialization."""
        mock_reddit = Mock()
        mock_praw.Reddit.return_value = mock_reddit

        self.source.initialize()
        assert self.source.is_available
        mock_praw.Reddit.assert_called_once()

    @patch('src.sentiment_analyzer.praw')
    def test_fetch_reddit_sentiment(self, mock_praw):
        """Test fetching sentiment from Reddit."""
        # Mock Reddit API
        mock_reddit = Mock()
        mock_praw.Reddit.return_value = mock_reddit

        # Mock subreddit and posts
        mock_subreddit = Mock()
        mock_reddit.subreddit.return_value = mock_subreddit

        mock_posts = [
            Mock(title="BTC to the moon!", selftext="", score=150, created_utc=datetime.now().timestamp()),
            Mock(title="Selling BTC", selftext="Bearish outlook", score=75, created_utc=datetime.now().timestamp()),
            Mock(title="BTC analysis", selftext="Technical review", score=50, created_utc=datetime.now().timestamp())
        ]
        mock_subreddit.hot.return_value = mock_posts

        self.source.initialize()
        scores = self.source.fetch_sentiment("BTC", limit=10)

        assert len(scores) == 3
        assert all(isinstance(score, SentimentScore) for score in scores)
        assert all(score.source == "reddit" for score in scores)


class TestNewsSentimentSource:
    """Test suite for NewsSentimentSource."""

    def setup_method(self):
        """Set up test fixtures."""
        self.source = NewsSentimentSource(api_key="test_key")

    @patch('src.sentiment_analyzer.requests')
    def test_news_initialization(self, mock_requests):
        """Test News source initialization."""
        self.source.initialize()
        assert self.source.is_available

    @patch('src.sentiment_analyzer.requests')
    def test_fetch_news_sentiment(self, mock_requests):
        """Test fetching sentiment from news sources."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "articles": [
                {
                    "title": "Bitcoin surges to new highs",
                    "description": "BTC reaches all-time high",
                    "publishedAt": datetime.now().isoformat(),
                    "source": {"name": "Crypto News"}
                },
                {
                    "title": "Bitcoin faces regulatory challenges",
                    "description": "New regulations may impact BTC",
                    "publishedAt": datetime.now().isoformat(),
                    "source": {"name": "Financial Times"}
                }
            ]
        }
        mock_requests.get.return_value = mock_response

        self.source.initialize()
        scores = self.source.fetch_sentiment("BTC", limit=10)

        assert len(scores) == 2
        assert all(isinstance(score, SentimentScore) for score in scores)
        assert all(score.source == "news" for score in scores)


class TestSentimentAggregator:
    """Test suite for SentimentAggregator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache_manager = Mock()
        self.aggregator = SentimentAggregator(cache_manager=self.cache_manager)

    def test_aggregator_initialization(self):
        """Test SentimentAggregator initialization."""
        assert self.aggregator.sources == []
        assert self.aggregator.weights == {}

    def test_add_source(self):
        """Test adding sentiment sources."""
        mock_source = Mock()
        mock_source.name = "test_source"

        self.aggregator.add_source(mock_source, weight=0.5)

        assert mock_source in self.aggregator.sources
        assert self.aggregator.weights["test_source"] == 0.5

    def test_aggregate_sentiment(self):
        """Test sentiment aggregation."""
        # Create mock sources with scores
        mock_source1 = Mock()
        mock_source1.name = "source1"
        mock_source1.fetch_sentiment.return_value = [
            SentimentScore(score=0.8, confidence=0.9, source="source1"),
            SentimentScore(score=0.6, confidence=0.8, source="source1")
        ]

        mock_source2 = Mock()
        mock_source2.name = "source2"
        mock_source2.fetch_sentiment.return_value = [
            SentimentScore(score=0.7, confidence=0.85, source="source2")
        ]

        self.aggregator.add_source(mock_source1, weight=0.6)
        self.aggregator.add_source(mock_source2, weight=0.4)

        result = self.aggregator.aggregate_sentiment("BTC")

        assert 'overall_score' in result
        assert 'confidence' in result
        assert 'source_scores' in result
        assert 'timestamp' in result
        assert 'sample_count' in result

    def test_weighted_aggregation(self):
        """Test weighted sentiment aggregation."""
        # Create scores with different weights
        scores = [
            SentimentScore(score=1.0, confidence=1.0, source="source1"),  # Weight 0.7
            SentimentScore(score=0.0, confidence=1.0, source="source2")   # Weight 0.3
        ]

        result = self.aggregator._weighted_aggregation(scores, {"source1": 0.7, "source2": 0.3})

        # Expected: (1.0 * 0.7) + (0.0 * 0.3) = 0.7
        assert abs(result['overall_score'] - 0.7) < 0.001

    def test_empty_sources(self):
        """Test aggregation with no sources."""
        result = self.aggregator.aggregate_sentiment("BTC")

        assert result['overall_score'] == 0.0
        assert result['confidence'] == 0.0
        assert result['sample_count'] == 0


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer main class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache_manager = Mock()
        self.analyzer = SentimentAnalyzer(cache_manager=self.cache_manager)

    def test_analyzer_initialization(self):
        """Test SentimentAnalyzer initialization."""
        assert isinstance(self.analyzer.aggregator, SentimentAggregator)

    @patch('src.sentiment_analyzer.TwitterSentimentSource')
    @patch('src.sentiment_analyzer.RedditSentimentSource')
    @patch('src.sentiment_analyzer.NewsSentimentSource')
    def test_setup_sources(self, mock_news, mock_reddit, mock_twitter):
        """Test setting up sentiment sources."""
        config = {
            'twitter': {'api_key': 'tw_key'},
            'reddit': {'client_id': 'rd_id', 'client_secret': 'rd_secret'},
            'news': {'api_key': 'nw_key'}
        }

        self.analyzer.setup_sources(config)

        mock_twitter.assert_called_once_with(api_key='tw_key')
        mock_reddit.assert_called_once_with(client_id='rd_id', client_secret='rd_secret')
        mock_news.assert_called_once_with(api_key='nw_key')

    def test_analyze_asset_sentiment(self):
        """Test asset sentiment analysis."""
        # Mock aggregator
        mock_result = {
            'overall_score': 0.65,
            'confidence': 0.8,
            'source_scores': {},
            'timestamp': datetime.now(),
            'sample_count': 10
        }
        self.analyzer.aggregator.aggregate_sentiment.return_value = mock_result

        result = self.analyzer.analyze_asset_sentiment("BTC")

        assert result == mock_result
        self.analyzer.aggregator.aggregate_sentiment.assert_called_once_with("BTC")

    def test_get_sentiment_trend(self):
        """Test sentiment trend calculation."""
        # Mock historical data
        historical_scores = [
            {'overall_score': 0.5, 'timestamp': datetime.now() - timedelta(hours=2)},
            {'overall_score': 0.6, 'timestamp': datetime.now() - timedelta(hours=1)},
            {'overall_score': 0.7, 'timestamp': datetime.now()}
        ]

        self.cache_manager.get.return_value = historical_scores

        trend = self.analyzer.get_sentiment_trend("BTC", hours=3)

        assert 'trend' in trend
        assert 'change_rate' in trend
        assert 'volatility' in trend
        assert 'data_points' in trend

    def test_sentiment_signal_generation(self):
        """Test trading signal generation from sentiment."""
        # Bullish sentiment
        bullish_result = {
            'overall_score': 0.8,
            'confidence': 0.9,
            'sample_count': 50
        }

        signal = self.analyzer.generate_sentiment_signal(bullish_result, "BTC")

        assert 'signal' in signal
        assert 'strength' in signal
        assert 'confidence' in signal
        assert 'reasoning' in signal

        # Bearish sentiment
        bearish_result = {
            'overall_score': 0.2,
            'confidence': 0.85,
            'sample_count': 50
        }

        signal = self.analyzer.generate_sentiment_signal(bearish_result, "BTC")
        assert signal['signal'] in ['SELL', 'HOLD']

    def test_cache_functionality(self):
        """Test sentiment caching."""
        # Test caching
        test_data = {"score": 0.7}
        success = self.analyzer.cache_sentiment_result("BTC", test_data)
        assert success
        self.cache_manager.set.assert_called_once()

        # Test retrieval
        cached = self.analyzer.get_cached_sentiment("BTC")
        assert cached == test_data
        self.cache_manager.get.assert_called_once()


class TestSentimentIntegration:
    """Integration tests for sentiment analysis."""

    def test_complete_sentiment_workflow(self):
        """Test complete sentiment analysis workflow."""
        analyzer = SentimentAnalyzer()

        # Test that analyzer has expected interface
        assert hasattr(analyzer, 'setup_sources')
        assert hasattr(analyzer, 'analyze_asset_sentiment')
        assert hasattr(analyzer, 'get_sentiment_trend')
        assert hasattr(analyzer, 'generate_sentiment_signal')

        # Test aggregator interface
        aggregator = analyzer.aggregator
        assert hasattr(aggregator, 'add_source')
        assert hasattr(aggregator, 'aggregate_sentiment')

    def test_error_handling_integration(self):
        """Test error handling in integrated workflow."""
        analyzer = SentimentAnalyzer()

        # Should handle missing sources gracefully
        result = analyzer.analyze_asset_sentiment("BTC")
        assert result['overall_score'] == 0.0
        assert result['sample_count'] == 0

        # Should handle cache failures gracefully
        analyzer.cache_manager.set.side_effect = Exception("Cache error")
        success = analyzer.cache_sentiment_result("BTC", {"test": "data"})
        assert not success