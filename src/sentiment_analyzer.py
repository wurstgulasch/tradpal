"""
Sentiment Analysis Module for TradPal Trading System.

This module provides sentiment analysis capabilities from multiple sources:
- Twitter API for real-time social sentiment
- News APIs for financial news sentiment
- Reddit API for community sentiment

The sentiment scores are integrated into the trading signal generation
to enhance decision making with market psychology.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
import aiohttp
import time
from dataclasses import dataclass
import re
import json

from config.settings import (
    SENTIMENT_ENABLED, SENTIMENT_SOURCES, SENTIMENT_UPDATE_INTERVAL,
    SENTIMENT_CACHE_TTL, SENTIMENT_WEIGHT, SENTIMENT_THRESHOLD,
    TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN,
    TWITTER_ACCESS_TOKEN_SECRET, TWITTER_BEARER_TOKEN, TWITTER_SEARCH_TERMS,
    TWITTER_MAX_TWEETS, TWITTER_LANGUAGE,
    NEWS_API_KEY, NEWS_SOURCES, NEWS_SEARCH_TERMS, NEWS_MAX_ARTICLES,
    NEWS_LANGUAGE,
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    REDDIT_SUBREDDITS, REDDIT_MAX_POSTS, REDDIT_TIME_FILTER,
    SENTIMENT_MODEL_TYPE, SENTIMENT_MODEL_CACHE_DIR,
    SENTIMENT_PREPROCESSING_ENABLED, SENTIMENT_REMOVE_STOPWORDS,
    SENTIMENT_LEMMATIZE
)

logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    """Container for sentiment analysis results."""
    timestamp: datetime
    source: str
    sentiment_score: float  # -1 (negative) to +1 (positive)
    confidence: float  # 0 to 1
    text_sample: str
    metadata: Dict[str, Any]

class SentimentAnalyzer:
    """
    Main sentiment analysis class that aggregates sentiment from multiple sources.
    """

    def __init__(self):
        if not SENTIMENT_ENABLED:
            logger.info("Sentiment analysis is disabled in configuration")
            return

        self.last_update = None
        self.sentiment_cache = {}
        self.model = None

        # Initialize sentiment analysis model
        self._initialize_model()

        # Initialize API clients
        self.twitter_client = None
        self.news_client = None
        self.reddit_client = None

        if 'twitter' in SENTIMENT_SOURCES:
            self.twitter_client = TwitterSentimentClient()
        if 'news' in SENTIMENT_SOURCES:
            self.news_client = NewsSentimentClient()
        if 'reddit' in SENTIMENT_SOURCES:
            self.reddit_client = RedditSentimentClient()

    def _initialize_model(self):
        """Initialize the sentiment analysis model."""
        if not SENTIMENT_ENABLED:
            return

        try:
            if SENTIMENT_MODEL_TYPE == 'vader':
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.model = SentimentIntensityAnalyzer()
                logger.info("Initialized VADER sentiment analyzer")

            elif SENTIMENT_MODEL_TYPE == 'textblob':
                from textblob import TextBlob
                self.model = TextBlob
                logger.info("Initialized TextBlob sentiment analyzer")

            elif SENTIMENT_MODEL_TYPE == 'transformers':
                from transformers import pipeline
                self.model = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    cache_dir=SENTIMENT_MODEL_CACHE_DIR
                )
                logger.info("Initialized Transformers sentiment analyzer")

            else:
                logger.warning(f"Unknown sentiment model type: {SENTIMENT_MODEL_TYPE}, falling back to VADER")
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.model = SentimentIntensityAnalyzer()

        except ImportError as e:
            logger.error(f"Failed to initialize sentiment model: {e}")
            logger.info("Falling back to basic sentiment analysis")
            self.model = None

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

    def should_update(self) -> bool:
        """Check if sentiment data should be updated."""
        if not SENTIMENT_ENABLED:
            return False

        if self.last_update is None:
            return True

        time_since_update = (datetime.now() - self.last_update).total_seconds()
        return time_since_update >= SENTIMENT_UPDATE_INTERVAL

    async def update_sentiment(self) -> Dict[str, float]:
        """
        Update sentiment data from all configured sources.

        Returns:
            Dictionary with aggregated sentiment scores by source
        """
        if not SENTIMENT_ENABLED:
            return {}

        try:
            logger.info("Updating sentiment data...")

            tasks = []
            if self.twitter_client:
                tasks.append(self.twitter_client.get_sentiment())
            if self.news_client:
                tasks.append(self.news_client.get_sentiment())
            if self.reddit_client:
                tasks.append(self.reddit_client.get_sentiment())

            if not tasks:
                logger.warning("No sentiment sources configured")
                return {}

            # Run all sentiment collection tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            aggregated_sentiment = {}
            all_sentiment_data = []

            for i, result in enumerate(results):
                source_name = SENTIMENT_SOURCES[i] if i < len(SENTIMENT_SOURCES) else f"source_{i}"

                if isinstance(result, Exception):
                    logger.error(f"Error getting sentiment from {source_name}: {result}")
                    continue

                if result:
                    aggregated_sentiment[source_name] = result.get('score', 0.0)
                    if 'data' in result:
                        all_sentiment_data.extend(result['data'])

            # Cache the results
            self.sentiment_cache = {
                'timestamp': datetime.now(),
                'aggregated': aggregated_sentiment,
                'raw_data': all_sentiment_data
            }

            self.last_update = datetime.now()
            logger.info(f"Updated sentiment data: {aggregated_sentiment}")

            return aggregated_sentiment

        except Exception as e:
            logger.error(f"Error updating sentiment: {e}")
            return {}

    def get_aggregated_sentiment(self) -> Dict[str, Any]:
        """
        Get the current aggregated sentiment data.

        Returns:
            Dictionary with sentiment scores and metadata
        """
        if not SENTIMENT_ENABLED:
            return {'enabled': False, 'score': 0.0, 'confidence': 0.0}

        # Check if cache is still valid
        if self.sentiment_cache and self.last_update:
            cache_age = (datetime.now() - self.last_update).total_seconds()
            if cache_age < SENTIMENT_CACHE_TTL:
                return self.sentiment_cache

        # Return cached data if available, even if expired
        if self.sentiment_cache:
            return self.sentiment_cache

        # No data available
        return {
            'timestamp': datetime.now(),
            'aggregated': {},
            'raw_data': [],
            'score': 0.0,
            'confidence': 0.0
        }

    def get_weighted_sentiment_score(self) -> Tuple[float, float]:
        """
        Get weighted sentiment score for signal generation.

        Returns:
            Tuple of (weighted_score, confidence)
        """
        if not SENTIMENT_ENABLED:
            return 0.0, 0.0

        sentiment_data = self.get_aggregated_sentiment()
        aggregated_scores = sentiment_data.get('aggregated', {})

        if not aggregated_scores:
            return 0.0, 0.0

        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0

        for source, score in aggregated_scores.items():
            # Different weights for different sources
            if source == 'twitter':
                weight = 0.4
            elif source == 'news':
                weight = 0.4
            elif source == 'reddit':
                weight = 0.2
            else:
                weight = 0.1

            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0, 0.0

        final_score = weighted_sum / total_weight

        # Calculate confidence based on number of sources and score consistency
        num_sources = len(aggregated_scores)
        confidence = min(0.9, num_sources * 0.3)  # Base confidence on number of sources

        # Reduce confidence if scores are inconsistent
        scores = list(aggregated_scores.values())
        if len(scores) > 1:
            score_std = np.std(scores)
            confidence *= max(0.5, 1.0 - score_std)  # Reduce confidence with higher variance

        return final_score, confidence

    def analyze_text_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of a single text using the configured model.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (sentiment_score, confidence)
        """
        if not self.model or not text:
            return 0.0, 0.0

        try:
            # Preprocess text if enabled
            if SENTIMENT_PREPROCESSING_ENABLED:
                text = self._preprocess_text(text)

            if SENTIMENT_MODEL_TYPE == 'vader':
                scores = self.model.polarity_scores(text)
                sentiment = scores['compound']  # -1 to 1
                confidence = (abs(sentiment) + 1) / 2  # Convert to 0-1 confidence

            elif SENTIMENT_MODEL_TYPE == 'textblob':
                blob = self.model(text)
                sentiment = blob.sentiment.polarity  # -1 to 1
                confidence = blob.sentiment.subjectivity  # 0-1

            elif SENTIMENT_MODEL_TYPE == 'transformers':
                result = self.model(text)[0]
                label = result['label']
                confidence = result['score']

                # Convert label to score
                if label == 'LABEL_2':  # Positive
                    sentiment = confidence
                elif label == 'LABEL_0':  # Negative
                    sentiment = -confidence
                else:  # Neutral
                    sentiment = 0.0

            else:
                return 0.0, 0.0

            return sentiment, confidence

        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return 0.0, 0.0

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis."""
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove mentions and hashtags symbols
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        if SENTIMENT_REMOVE_STOPWORDS:
            try:
                import nltk
                from nltk.corpus import stopwords
                nltk.download('stopwords', quiet=True)
                stop_words = set(stopwords.words('english'))
                words = text.split()
                text = ' '.join([word for word in words if word not in stop_words])
            except ImportError:
                logger.warning("NLTK not available for stopword removal")

        if SENTIMENT_LEMMATIZE:
            try:
                import nltk
                from nltk.stem import WordNetLemmatizer
                nltk.download('wordnet', quiet=True)
                lemmatizer = WordNetLemmatizer()
                words = text.split()
                text = ' '.join([lemmatizer.lemmatize(word) for word in words])
            except ImportError:
                logger.warning("NLTK not available for lemmatization")

        return text


class TwitterSentimentClient:
    """Client for Twitter sentiment analysis."""

    def __init__(self):
        self.bearer_token = TWITTER_BEARER_TOKEN
        self.api_key = TWITTER_API_KEY
        self.api_secret = TWITTER_API_SECRET
        self.access_token = TWITTER_ACCESS_TOKEN
        self.access_token_secret = TWITTER_ACCESS_TOKEN_SECRET

    async def get_sentiment(self) -> Optional[Dict[str, Any]]:
        """Get sentiment from Twitter API."""
        if not self.bearer_token:
            logger.warning("Twitter API credentials not configured")
            return None

        try:
            # Use Twitter API v2 for recent tweets
            search_query = ' OR '.join(TWITTER_SEARCH_TERMS)
            url = f"https://api.twitter.com/2/tweets/search/recent?query={search_query}&max_results={TWITTER_MAX_TWEETS}&tweet.fields=text,created_at,lang"

            headers = {"Authorization": f"Bearer {self.bearer_token}"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        tweets = data.get('data', [])

                        sentiment_scores = []
                        sentiment_data = []

                        analyzer = SentimentAnalyzer()

                        for tweet in tweets:
                            if tweet.get('lang') == TWITTER_LANGUAGE:
                                text = tweet['text']
                                score, confidence = analyzer.analyze_text_sentiment(text)

                                sentiment_scores.append(score)
                                sentiment_data.append(SentimentData(
                                    timestamp=datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                                    source='twitter',
                                    sentiment_score=score,
                                    confidence=confidence,
                                    text_sample=text[:100],
                                    metadata={'tweet_id': tweet.get('id'), 'lang': tweet.get('lang')}
                                ))

                        if sentiment_scores:
                            avg_score = np.mean(sentiment_scores)
                            return {
                                'score': avg_score,
                                'confidence': np.mean([d.confidence for d in sentiment_data]),
                                'data': sentiment_data
                            }

            return None

        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment: {e}")
            return None


class NewsSentimentClient:
    """Client for news sentiment analysis."""

    def __init__(self):
        self.api_key = NEWS_API_KEY

    async def get_sentiment(self) -> Optional[Dict[str, Any]]:
        """Get sentiment from news APIs."""
        if not self.api_key:
            logger.warning("News API key not configured")
            return None

        try:
            # Use NewsAPI for recent articles
            query = ' OR '.join(NEWS_SEARCH_TERMS)
            url = f"https://newsapi.org/v2/everything?q={query}&language={NEWS_LANGUAGE}&pageSize={NEWS_MAX_ARTICLES}&apiKey={self.api_key}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])

                        sentiment_scores = []
                        sentiment_data = []

                        analyzer = SentimentAnalyzer()

                        for article in articles:
                            title = article.get('title', '')
                            description = article.get('description', '')
                            text = f"{title} {description}".strip()

                            if text:
                                score, confidence = analyzer.analyze_text_sentiment(text)

                                sentiment_scores.append(score)
                                sentiment_data.append(SentimentData(
                                    timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                                    source='news',
                                    sentiment_score=score,
                                    confidence=confidence,
                                    text_sample=text[:100],
                                    metadata={
                                        'source': article.get('source', {}).get('name'),
                                        'url': article.get('url'),
                                        'title': title
                                    }
                                ))

                        if sentiment_scores:
                            avg_score = np.mean(sentiment_scores)
                            return {
                                'score': avg_score,
                                'confidence': np.mean([d.confidence for d in sentiment_data]),
                                'data': sentiment_data
                            }

            return None

        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return None


class RedditSentimentClient:
    """Client for Reddit sentiment analysis."""

    def __init__(self):
        self.client_id = REDDIT_CLIENT_ID
        self.client_secret = REDDIT_CLIENT_SECRET
        self.user_agent = REDDIT_USER_AGENT

    async def get_sentiment(self) -> Optional[Dict[str, Any]]:
        """Get sentiment from Reddit API."""
        if not self.client_id or not self.client_secret:
            logger.warning("Reddit API credentials not configured")
            return None

        try:
            # Reddit API authentication
            auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
            data = {'grant_type': 'client_credentials'}

            async with aiohttp.ClientSession() as session:
                # Get access token
                async with session.post('https://www.reddit.com/api/v1/access_token',
                                      auth=auth, data=data) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        access_token = token_data.get('access_token')

                        if access_token:
                            headers = {'Authorization': f'bearer {access_token}',
                                     'User-Agent': self.user_agent}

                            sentiment_scores = []
                            sentiment_data = []

                            analyzer = SentimentAnalyzer()

                            # Get posts from each subreddit
                            for subreddit in REDDIT_SUBREDDITS:
                                url = f"https://oauth.reddit.com/r/{subreddit}/hot?t={REDDIT_TIME_FILTER}&limit={REDDIT_MAX_POSTS}"

                                async with session.get(url, headers=headers) as response:
                                    if response.status == 200:
                                        data = await response.json()
                                        posts = data.get('data', {}).get('children', [])

                                        for post in posts:
                                            post_data = post.get('data', {})
                                            title = post_data.get('title', '')
                                            selftext = post_data.get('selftext', '')
                                            text = f"{title} {selftext}".strip()

                                            if text:
                                                score, confidence = analyzer.analyze_text_sentiment(text)

                                                sentiment_scores.append(score)
                                                sentiment_data.append(SentimentData(
                                                    timestamp=datetime.fromtimestamp(post_data.get('created_utc', 0)),
                                                    source='reddit',
                                                    sentiment_score=score,
                                                    confidence=confidence,
                                                    text_sample=text[:100],
                                                    metadata={
                                                        'subreddit': subreddit,
                                                        'post_id': post_data.get('id'),
                                                        'score': post_data.get('score'),
                                                        'num_comments': post_data.get('num_comments')
                                                    }
                                                ))

                            if sentiment_scores:
                                avg_score = np.mean(sentiment_scores)
                                return {
                                    'score': avg_score,
                                    'confidence': np.mean([d.confidence for d in sentiment_data]),
                                    'data': sentiment_data
                                }

            return None

        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment: {e}")
            return None


# Global sentiment analyzer instance
sentiment_analyzer = SentimentAnalyzer()

async def get_current_sentiment() -> Dict[str, Any]:
    """
    Get current sentiment data from all sources.

    Returns:
        Dictionary with sentiment information
    """
    if not SENTIMENT_ENABLED:
        return {'enabled': False, 'score': 0.0, 'confidence': 0.0}

    if sentiment_analyzer.should_update():
        await sentiment_analyzer.update_sentiment()

    return sentiment_analyzer.get_aggregated_sentiment()

def get_sentiment_score() -> Tuple[float, float]:
    """
    Get weighted sentiment score for trading signals.

    Returns:
        Tuple of (sentiment_score, confidence)
    """
    if not SENTIMENT_ENABLED:
        return 0.0, 0.0

    return sentiment_analyzer.get_weighted_sentiment_score()

def analyze_text_sentiment(text: str) -> Tuple[float, float]:
    """
    Analyze sentiment of a text string.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (sentiment_score, confidence)
    """
    return sentiment_analyzer.analyze_text_sentiment(text)