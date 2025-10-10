"""
Sentiment Analysis Module for TradPal

This module provides sentiment analysis capabilities for cryptocurrency news and social media,
specifically optimized for BTC/USDT trading signals.

Features:
- Twitter/X API integration for real-time sentiment
- News API integration for financial news
- Pre-trained NLP models for sentiment classification
- Rate limiting and caching for API efficiency
- Integration with trading signals for sentiment-enhanced decisions

Author: TradPal Team
Version: 2.5.0
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Optional dependencies
try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    tweepy = None

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    TextBlob = None

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    NewsApiClient = None

from src.cache import CacheManager
from src.error_handling import APIError, RateLimitError, handle_api_errors
from src.audit_logger import audit_log

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""
    timestamp: datetime
    symbol: str
    sentiment_score: float  # -1 (bearish) to +1 (bullish)
    confidence: float  # 0 to 1
    source: str  # 'twitter', 'news', 'reddit', etc.
    text_sample: str
    volume: int  # Number of mentions/posts
    metadata: Dict[str, Any]

class SentimentAnalyzer:
    """
    Main sentiment analysis class for cryptocurrency markets.

    Provides real-time sentiment analysis from multiple sources:
    - Twitter/X API for social sentiment
    - News APIs for financial news
    - Reddit API for community sentiment
    - Pre-trained NLP models for text analysis
    """

    def __init__(self,
                 cache_manager: Optional[CacheManager] = None,
                 api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the sentiment analyzer.

        Args:
            cache_manager: Cache manager for API responses
            api_keys: Dictionary of API keys for different services
        """
        self.cache_manager = cache_manager or CacheManager()
        self.api_keys = api_keys or self._load_api_keys()

        # Initialize API clients
        self.twitter_client = None
        self.news_client = None
        self.reddit_client = None

        # Initialize NLP models
        self.sentiment_pipeline = None
        self._initialize_clients()
        self._initialize_nlp_models()

        # Rate limiting
        self.rate_limits = {
            'twitter': {'calls': 0, 'reset_time': time.time(), 'limit': 300},  # 300 calls per 15min
            'news': {'calls': 0, 'reset_time': time.time(), 'limit': 100},     # 100 calls per day
            'reddit': {'calls': 0, 'reset_time': time.time(), 'limit': 60},    # 60 calls per minute
        }

        logger.info("SentimentAnalyzer initialized successfully")

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables."""
        return {
            'twitter_bearer_token': os.getenv('TWITTER_BEARER_TOKEN', ''),
            'twitter_api_key': os.getenv('TWITTER_API_KEY', ''),
            'twitter_api_secret': os.getenv('TWITTER_API_SECRET', ''),
            'twitter_access_token': os.getenv('TWITTER_ACCESS_TOKEN', ''),
            'twitter_access_secret': os.getenv('TWITTER_ACCESS_SECRET', ''),
            'news_api_key': os.getenv('NEWS_API_KEY', ''),
            'reddit_client_id': os.getenv('REDDIT_CLIENT_ID', ''),
            'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET', ''),
        }

    def _initialize_clients(self):
        """Initialize API clients for different sentiment sources."""
        # Twitter/X API v2
        if TWEEPY_AVAILABLE and self.api_keys.get('twitter_bearer_token'):
            try:
                self.twitter_client = tweepy.Client(
                    bearer_token=self.api_keys['twitter_bearer_token'],
                    consumer_key=self.api_keys.get('twitter_api_key'),
                    consumer_secret=self.api_keys.get('twitter_api_secret'),
                    access_token=self.api_keys.get('twitter_access_token'),
                    access_token_secret=self.api_keys.get('twitter_access_secret'),
                    wait_on_rate_limit=True
                )
                logger.info("Twitter API client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Twitter client: {e}")

        # News API
        if NEWSAPI_AVAILABLE and self.api_keys.get('news_api_key'):
            try:
                self.news_client = NewsApiClient(api_key=self.api_keys['news_api_key'])
                logger.info("News API client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize News API client: {e}")

        # Reddit API (PRAW)
        if self.api_keys.get('reddit_client_id'):
            try:
                import praw
                self.reddit_client = praw.Reddit(
                    client_id=self.api_keys['reddit_client_id'],
                    client_secret=self.api_keys['reddit_client_secret'],
                    user_agent='TradPal v2.5.0'
                )
                logger.info("Reddit API client initialized")
            except ImportError:
                logger.warning("PRAW not installed, Reddit sentiment unavailable")
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit client: {e}")

    def _initialize_nlp_models(self):
        """Initialize pre-trained NLP models for sentiment analysis."""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use FinBERT for financial sentiment analysis
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    return_all_scores=True
                )
                logger.info("FinBERT sentiment model loaded")
            except Exception as e:
                logger.warning(f"Failed to load FinBERT model: {e}")
                # Fallback to basic model
                try:
                    self.sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        return_all_scores=True
                    )
                    logger.info("Twitter RoBERTa sentiment model loaded")
                except Exception as e:
                    logger.warning(f"Failed to load fallback sentiment model: {e}")

    @handle_api_errors
    def _check_rate_limit(self, source: str) -> bool:
        """Check and update rate limiting for API calls."""
        current_time = time.time()
        limit_info = self.rate_limits[source]

        # Reset counter if time window passed
        if source == 'twitter' and current_time - limit_info['reset_time'] > 900:  # 15 minutes
            limit_info['calls'] = 0
            limit_info['reset_time'] = current_time
        elif source == 'news' and current_time - limit_info['reset_time'] > 86400:  # 24 hours
            limit_info['calls'] = 0
            limit_info['reset_time'] = current_time
        elif source == 'reddit' and current_time - limit_info['reset_time'] > 60:  # 1 minute
            limit_info['calls'] = 0
            limit_info['reset_time'] = current_time

        if limit_info['calls'] >= limit_info['limit']:
            raise RateLimitError(f"Rate limit exceeded for {source} API")

        limit_info['calls'] += 1
        return True

    @handle_api_errors
    def get_twitter_sentiment(self, symbol: str = "BTC", hours_back: int = 24) -> SentimentResult:
        """
        Get sentiment from Twitter/X for a cryptocurrency symbol.

        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            hours_back: Hours of historical data to analyze

        Returns:
            SentimentResult with aggregated Twitter sentiment
        """
        if not self.twitter_client:
            raise APIError("Twitter client not initialized")

        self._check_rate_limit('twitter')

        # Search query for cryptocurrency mentions
        query = f"#{symbol} OR ${symbol} OR {symbol}USD -is:retweet lang:en"
        start_time = datetime.utcnow() - timedelta(hours=hours_back)

        try:
            # Search recent tweets
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                start_time=start_time,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics', 'text']
            )

            if not tweets.data:
                return SentimentResult(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    sentiment_score=0.0,
                    confidence=0.0,
                    source='twitter',
                    text_sample='No recent tweets found',
                    volume=0,
                    metadata={'query': query, 'hours_back': hours_back}
                )

            # Analyze sentiment of tweets
            sentiments = []
            texts = []

            for tweet in tweets.data:
                text = tweet.text
                texts.append(text)

                # Use NLP model for sentiment
                if self.sentiment_pipeline:
                    try:
                        results = self.sentiment_pipeline(text)
                        # Convert to -1 to +1 scale
                        sentiment_score = self._convert_sentiment_scores(results)
                        sentiments.append(sentiment_score)
                    except Exception as e:
                        logger.warning(f"Sentiment analysis failed for tweet: {e}")
                        continue
                elif TEXTBLOB_AVAILABLE:
                    # Fallback to TextBlob
                    blob = TextBlob(text)
                    sentiments.append(blob.sentiment.polarity)
                else:
                    # Basic keyword analysis
                    sentiments.append(self._basic_sentiment_analysis(text))

            if not sentiments:
                sentiment_score = 0.0
                confidence = 0.0
            else:
                sentiment_score = np.mean(sentiments)
                confidence = min(1.0, len(sentiments) / 50)  # Higher confidence with more data

            # Sample text for audit
            text_sample = texts[0] if texts else "No text available"

            result = SentimentResult(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                sentiment_score=float(sentiment_score),
                confidence=float(confidence),
                source='twitter',
                text_sample=text_sample[:200],  # Truncate for storage
                volume=len(tweets.data),
                metadata={
                    'query': query,
                    'hours_back': hours_back,
                    'tweets_analyzed': len(sentiments),
                    'avg_sentiment': sentiment_score,
                    'sentiment_std': np.std(sentiments) if sentiments else 0.0
                }
            )

            # Cache result
            cache_key = f"sentiment_twitter_{symbol}_{hours_back}h"
            self.cache_manager.set(cache_key, result, ttl=3600)  # Cache for 1 hour

            audit_log('sentiment_analysis', 'twitter', {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'volume': len(tweets.data)
            })

            return result

        except Exception as e:
            logger.error(f"Twitter sentiment analysis failed: {e}")
            raise APIError(f"Twitter API error: {e}")

    @handle_api_errors
    def get_news_sentiment(self, symbol: str = "bitcoin", days_back: int = 7) -> SentimentResult:
        """
        Get sentiment from financial news for a cryptocurrency.

        Args:
            symbol: Cryptocurrency name (e.g., 'bitcoin', 'ethereum')
            days_back: Days of historical news to analyze

        Returns:
            SentimentResult with aggregated news sentiment
        """
        if not self.news_client:
            raise APIError("News API client not initialized")

        self._check_rate_limit('news')

        try:
            # Search for news articles
            from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            articles = self.news_client.get_everything(
                q=f'"{symbol}" cryptocurrency OR "{symbol}" crypto',
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page_size=50
            )

            if not articles['articles']:
                return SentimentResult(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    sentiment_score=0.0,
                    confidence=0.0,
                    source='news',
                    text_sample='No recent news found',
                    volume=0,
                    metadata={'query': f'"{symbol}" cryptocurrency', 'days_back': days_back}
                )

            # Analyze sentiment of article titles and descriptions
            sentiments = []
            texts = []

            for article in articles['articles']:
                # Combine title and description
                text = f"{article['title']} {article.get('description', '')}"
                texts.append(article['title'])

                # Analyze sentiment
                if self.sentiment_pipeline:
                    try:
                        results = self.sentiment_pipeline(text)
                        sentiment_score = self._convert_sentiment_scores(results)
                        sentiments.append(sentiment_score)
                    except Exception as e:
                        logger.warning(f"News sentiment analysis failed: {e}")
                        continue
                elif TEXTBLOB_AVAILABLE:
                    blob = TextBlob(text)
                    sentiments.append(blob.sentiment.polarity)
                else:
                    sentiments.append(self._basic_sentiment_analysis(text))

            if not sentiments:
                sentiment_score = 0.0
                confidence = 0.0
            else:
                sentiment_score = np.mean(sentiments)
                confidence = min(1.0, len(sentiments) / 20)  # Higher confidence with more articles

            text_sample = texts[0] if texts else "No headlines available"

            result = SentimentResult(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                sentiment_score=float(sentiment_score),
                confidence=float(confidence),
                source='news',
                text_sample=text_sample[:200],
                volume=len(articles['articles']),
                metadata={
                    'query': f'"{symbol}" cryptocurrency',
                    'days_back': days_back,
                    'articles_analyzed': len(sentiments),
                    'avg_sentiment': sentiment_score,
                    'sentiment_std': np.std(sentiments) if sentiments else 0.0
                }
            )

            # Cache result
            cache_key = f"sentiment_news_{symbol}_{days_back}d"
            self.cache_manager.set(cache_key, result, ttl=7200)  # Cache for 2 hours

            audit_log('sentiment_analysis', 'news', {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'volume': len(articles['articles'])
            })

            return result

        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            raise APIError(f"News API error: {e}")

    def get_reddit_sentiment(self, symbol: str = "Bitcoin", hours_back: int = 24) -> SentimentResult:
        """
        Get sentiment from Reddit for a cryptocurrency.

        Args:
            symbol: Cryptocurrency name (e.g., 'Bitcoin', 'Ethereum')
            hours_back: Hours of historical data to analyze

        Returns:
            SentimentResult with aggregated Reddit sentiment
        """
        if not self.reddit_client:
            raise APIError("Reddit client not initialized")

        self._check_rate_limit('reddit')

        try:
            # Search relevant subreddits
            subreddits = ['cryptocurrency', 'bitcoin', 'CryptoMarkets', 'CryptoCurrencies']
            posts = []

            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    # Get recent posts mentioning the symbol
                    for post in subreddit.search(f'"{symbol}"', time_filter='day', limit=25):
                        # Check if post is within time window
                        post_time = datetime.utcfromtimestamp(post.created_utc)
                        if datetime.utcnow() - post_time < timedelta(hours=hours_back):
                            posts.append({
                                'title': post.title,
                                'text': post.selftext,
                                'score': post.score,
                                'created_utc': post.created_utc
                            })
                except Exception as e:
                    logger.warning(f"Failed to search r/{subreddit_name}: {e}")
                    continue

            if not posts:
                return SentimentResult(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    sentiment_score=0.0,
                    confidence=0.0,
                    source='reddit',
                    text_sample='No recent posts found',
                    volume=0,
                    metadata={'subreddits': subreddits, 'hours_back': hours_back}
                )

            # Analyze sentiment of posts
            sentiments = []
            texts = []

            for post in posts:
                text = f"{post['title']} {post['text']}"
                texts.append(post['title'])

                # Weight by post score (upvotes)
                weight = max(1, post['score'])

                if self.sentiment_pipeline:
                    try:
                        results = self.sentiment_pipeline(text)
                        sentiment_score = self._convert_sentiment_scores(results)
                        # Apply weight
                        sentiments.extend([sentiment_score] * weight)
                    except Exception as e:
                        logger.warning(f"Reddit sentiment analysis failed: {e}")
                        continue
                elif TEXTBLOB_AVAILABLE:
                    blob = TextBlob(text)
                    sentiments.extend([blob.sentiment.polarity] * weight)
                else:
                    sentiment = self._basic_sentiment_analysis(text)
                    sentiments.extend([sentiment] * weight)

            if not sentiments:
                sentiment_score = 0.0
                confidence = 0.0
            else:
                sentiment_score = np.mean(sentiments)
                confidence = min(1.0, len(posts) / 10)  # Higher confidence with more posts

            text_sample = texts[0] if texts else "No posts available"

            result = SentimentResult(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                sentiment_score=float(sentiment_score),
                confidence=float(confidence),
                source='reddit',
                text_sample=text_sample[:200],
                volume=len(posts),
                metadata={
                    'subreddits': subreddits,
                    'hours_back': hours_back,
                    'posts_analyzed': len(posts),
                    'avg_sentiment': sentiment_score,
                    'sentiment_std': np.std(sentiments) if sentiments else 0.0,
                    'total_weighted_samples': len(sentiments)
                }
            )

            # Cache result
            cache_key = f"sentiment_reddit_{symbol}_{hours_back}h"
            self.cache_manager.set(cache_key, result, ttl=3600)  # Cache for 1 hour

            audit_log('sentiment_analysis', 'reddit', {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'volume': len(posts)
            })

            return result

        except Exception as e:
            logger.error(f"Reddit sentiment analysis failed: {e}")
            raise APIError(f"Reddit API error: {e}")

    def get_aggregated_sentiment(self, symbol: str = "BTC",
                                hours_back: int = 24) -> SentimentResult:
        """
        Get aggregated sentiment from all available sources.

        Args:
            symbol: Cryptocurrency symbol/name
            hours_back: Hours of historical data to analyze

        Returns:
            SentimentResult with weighted aggregate sentiment
        """
        results = []
        weights = {'twitter': 0.4, 'news': 0.4, 'reddit': 0.2}  # Weight by reliability

        # Get sentiment from each source
        try:
            twitter_result = self.get_twitter_sentiment(symbol, hours_back)
            if twitter_result.confidence > 0:
                results.append((twitter_result, weights['twitter']))
        except Exception as e:
            logger.warning(f"Twitter sentiment failed: {e}")

        try:
            # Convert symbol for news API (BTC -> bitcoin)
            news_symbol = {'BTC': 'bitcoin', 'ETH': 'ethereum'}.get(symbol, symbol.lower())
            news_result = self.get_news_sentiment(news_symbol, max(1, hours_back // 24))
            if news_result.confidence > 0:
                results.append((news_result, weights['news']))
        except Exception as e:
            logger.warning(f"News sentiment failed: {e}")

        try:
            # Convert symbol for Reddit (BTC -> Bitcoin)
            reddit_symbol = {'BTC': 'Bitcoin', 'ETH': 'Ethereum'}.get(symbol, symbol)
            reddit_result = self.get_reddit_sentiment(reddit_symbol, hours_back)
            if reddit_result.confidence > 0:
                results.append((reddit_result, weights['reddit']))
        except Exception as e:
            logger.warning(f"Reddit sentiment failed: {e}")

        if not results:
            return SentimentResult(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                source='aggregated',
                text_sample='No sentiment data available',
                volume=0,
                metadata={'sources_attempted': list(weights.keys()), 'hours_back': hours_back}
            )

        # Calculate weighted average
        total_weight = sum(weight for _, weight in results)
        weighted_sentiment = sum(result.sentiment_score * weight for result, weight in results) / total_weight
        avg_confidence = np.mean([result.confidence for result, _ in results])

        # Combine metadata
        all_metadata = {
            'sources_used': [result.source for result, _ in results],
            'individual_results': [
                {
                    'source': result.source,
                    'sentiment': result.sentiment_score,
                    'confidence': result.confidence,
                    'volume': result.volume
                } for result, _ in results
            ],
            'weights': weights,
            'hours_back': hours_back
        }

        # Sample text from highest confidence source
        best_result = max(results, key=lambda x: x[0].confidence)[0]
        text_sample = f"{best_result.source.upper()}: {best_result.text_sample}"

        result = SentimentResult(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            sentiment_score=float(weighted_sentiment),
            confidence=float(avg_confidence),
            source='aggregated',
            text_sample=text_sample[:200],
            volume=sum(result.volume for result, _ in results),
            metadata=all_metadata
        )

        audit_log('sentiment_analysis', 'aggregated', {
            'symbol': symbol,
            'sentiment_score': weighted_sentiment,
            'confidence': avg_confidence,
            'sources_used': len(results),
            'total_volume': result.volume
        })

        return result

    def _convert_sentiment_scores(self, pipeline_results: List[Dict]) -> float:
        """Convert transformer pipeline results to -1 to +1 scale."""
        if not pipeline_results:
            return 0.0

        # Handle different model outputs
        if isinstance(pipeline_results[0], dict) and 'label' in pipeline_results[0]:
            # FinBERT style: [{'label': 'positive', 'score': 0.8}, ...]
            score_map = {'negative': -1, 'neutral': 0, 'positive': 1}
            weighted_score = 0
            total_weight = 0

            for result in pipeline_results:
                label = result['label'].lower()
                score = result['score']
                sentiment_value = score_map.get(label, 0)
                weighted_score += sentiment_value * score
                total_weight += score

            return weighted_score / total_weight if total_weight > 0 else 0.0
        else:
            # Other models might have different formats
            return 0.0

    def _basic_sentiment_analysis(self, text: str) -> float:
        """Basic keyword-based sentiment analysis as fallback."""
        text_lower = text.lower()

        positive_words = ['bullish', 'moon', 'pump', 'up', 'buy', 'long', 'green', 'profit', 'gain']
        negative_words = ['bearish', 'dump', 'down', 'sell', 'short', 'red', 'loss', 'crash', 'drop']

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total_words = positive_count + negative_count
        if total_words == 0:
            return 0.0

        return (positive_count - negative_count) / total_words

    def get_sentiment_signal(self, sentiment_result: SentimentResult,
                           threshold: float = 0.1) -> Optional[str]:
        """
        Convert sentiment score to trading signal.

        Args:
            sentiment_result: Sentiment analysis result
            threshold: Minimum sentiment score for signal generation

        Returns:
            'BUY', 'SELL', or None
        """
        if sentiment_result.confidence < 0.3:  # Minimum confidence threshold
            return None

        if sentiment_result.sentiment_score > threshold:
            return 'BUY'
        elif sentiment_result.sentiment_score < -threshold:
            return 'SELL'

        return None

# Convenience functions for easy integration
def get_btc_sentiment(hours_back: int = 24) -> SentimentResult:
    """Get current BTC sentiment from all sources."""
    analyzer = SentimentAnalyzer()
    return analyzer.get_aggregated_sentiment('BTC', hours_back)

def get_sentiment_signal(symbol: str = 'BTC', hours_back: int = 24,
                        threshold: float = 0.1) -> Optional[str]:
    """Get sentiment-based trading signal."""
    analyzer = SentimentAnalyzer()
    result = analyzer.get_aggregated_sentiment(symbol, hours_back)
    return analyzer.get_sentiment_signal(result, threshold)

if __name__ == "__main__":
    # Example usage
    print("🔍 Testing Sentiment Analysis...")

    try:
        # Test aggregated sentiment for BTC
        result = get_btc_sentiment(hours_back=6)
        print(f"📊 BTC Sentiment: {result.sentiment_score:.3f} (confidence: {result.confidence:.3f})")
        print(f"📝 Sample: {result.text_sample}")
        print(f"📊 Volume: {result.volume} mentions")
        print(f"🔗 Sources: {result.metadata.get('sources_used', [])}")

        # Test signal generation
        signal = get_sentiment_signal('BTC', hours_back=6)
        if signal:
            print(f"🚨 Sentiment Signal: {signal}")
        else:
            print("🤔 No clear sentiment signal")

    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        print("💡 Make sure to set API keys in environment variables or .env file")