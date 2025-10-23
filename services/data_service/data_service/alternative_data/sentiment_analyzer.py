#!/usr/bin/env python3
"""
Sentiment Analyzer - Social Media and News Sentiment Analysis

Provides comprehensive sentiment analysis from multiple sources:
- Twitter API for real-time social sentiment
- Reddit API for community sentiment
- News APIs for financial news sentiment
- Fear & Greed Index integration
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

import aiohttp
import pandas as pd
import numpy as np

# Optional imports with fallbacks
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .__init__ import SentimentData, SentimentSource

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Sentiment analysis score."""
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    magnitude: float  # 0.0 to 1.0
    source: str
    timestamp: datetime


class SentimentAnalyzer:
    """
    Advanced sentiment analyzer with multiple data sources and ML models.
    """

    def __init__(self):
        self.session = None
        self.sentiment_model = None
        self.fear_greed_cache = {}
        self.twitter_bearer_token = None
        self.reddit_client_id = None
        self.reddit_client_secret = None
        self.news_api_key = None

        # Initialize API keys from environment
        self._load_api_keys()

    def _load_api_keys(self):
        """Load API keys from environment variables."""
        import os
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.news_api_key = os.getenv('NEWS_API_KEY')

    async def initialize(self):
        """Initialize the sentiment analyzer."""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )

            # Initialize sentiment model (using transformers for better accuracy)
            try:
                if TRANSFORMERS_AVAILABLE:
                    self.sentiment_model = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        return_all_scores=True
                    )
                    logger.info("Advanced sentiment model loaded")
                else:
                    self.sentiment_model = None
                    logger.info("Using basic sentiment analysis (transformers not available)")
            except Exception as e:
                logger.warning(f"Could not load advanced model, using basic analysis: {e}")
                self.sentiment_model = None

            logger.info("Sentiment Analyzer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Sentiment Analyzer: {e}")
            raise

    async def close(self):
        """Close resources."""
        if self.session:
            await self.session.close()

    async def analyze_symbol_sentiment(self, symbol: str, hours: int = 24) -> List[SentimentData]:
        """
        Analyze sentiment for a trading symbol from multiple sources.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            hours: Hours of historical data to analyze

        Returns:
            List of sentiment data from different sources
        """
        try:
            # Extract base symbol for social media search
            base_symbol = symbol.split('/')[0]  # BTC from BTC/USDT

            # Collect sentiment from all sources in parallel
            tasks = [
                self._analyze_twitter_sentiment(base_symbol, hours),
                self._analyze_reddit_sentiment(base_symbol, hours),
                self._analyze_news_sentiment(base_symbol, hours)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            sentiment_data = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Sentiment source {i} failed: {result}")
                    continue

                if result:
                    sentiment_data.extend(result)

            return sentiment_data

        except Exception as e:
            logger.error(f"Symbol sentiment analysis failed: {e}")
            return []

    async def _analyze_twitter_sentiment(self, symbol: str, hours: int) -> List[SentimentData]:
        """Analyze Twitter sentiment for a symbol."""
        if not self.twitter_bearer_token:
            logger.warning("Twitter API key not configured")
            return []

        try:
            # Twitter API v2 recent search
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {"Authorization": f"Bearer {self.twitter_bearer_token}"}

            # Search query for the symbol
            query = f"({symbol} OR ${symbol}) -is:retweet lang:en"
            params = {
                "query": query,
                "max_results": 100,
                "tweet.fields": "created_at,public_metrics,text",
                "start_time": (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
            }

            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Twitter API error: {response.status}")
                    return []

                data = await response.json()

            tweets = data.get('data', [])
            if not tweets:
                return []

            # Analyze sentiment of tweets
            sentiments = []
            for tweet in tweets:
                text = tweet['text']
                score = await self._analyze_text_sentiment(text)

                sentiment_data = SentimentData(
                    symbol=symbol,
                    source=SentimentSource.TWITTER,
                    sentiment_score=score['score'],
                    confidence=score['confidence'],
                    volume=1,
                    timestamp=datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                    raw_data={
                        'tweet_id': tweet['id'],
                        'text': text[:100],  # Truncate for storage
                        'likes': tweet['public_metrics']['like_count'],
                        'retweets': tweet['public_metrics']['retweet_count']
                    }
                )
                sentiments.append(sentiment_data)

            # Aggregate sentiment
            if sentiments:
                avg_score = np.mean([s.sentiment_score for s in sentiments])
                total_volume = len(sentiments)
                avg_confidence = np.mean([s.confidence for s in sentiments])

                return [SentimentData(
                    symbol=symbol,
                    source=SentimentSource.TWITTER,
                    sentiment_score=avg_score,
                    confidence=avg_confidence,
                    volume=total_volume,
                    timestamp=datetime.utcnow()
                )]
            else:
                return []

        except Exception as e:
            logger.error(f"Twitter sentiment analysis failed: {e}")
            return []

    async def _analyze_reddit_sentiment(self, symbol: str, hours: int) -> List[SentimentData]:
        """Analyze Reddit sentiment for a symbol."""
        if not all([self.reddit_client_id, self.reddit_client_secret]):
            logger.warning("Reddit API credentials not configured")
            return []

        try:
            # Reddit API authentication
            auth = aiohttp.BasicAuth(self.reddit_client_id, self.reddit_client_secret)
            data = {"grant_type": "client_credentials"}

            async with self.session.post("https://www.reddit.com/api/v1/access_token",
                                       auth=auth, data=data) as response:
                if response.status != 200:
                    logger.warning(f"Reddit auth failed: {response.status}")
                    return []

                token_data = await response.json()
                access_token = token_data.get('access_token')

            if not access_token:
                return []

            # Search Reddit posts
            headers = {"Authorization": f"bearer {access_token}"}
            url = "https://oauth.reddit.com/search"
            params = {
                "q": f"{symbol} OR ${symbol}",
                "sort": "new",
                "limit": 50,
                "t": "day",  # Last 24 hours
                "type": "link"
            }

            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Reddit API error: {response.status}")
                    return []

                data = await response.json()

            posts = data.get('data', {}).get('children', [])
            if not posts:
                return []

            # Analyze sentiment
            sentiments = []
            for post in posts:
                post_data = post['data']
                title = post_data['title']
                selftext = post_data.get('selftext', '')

                # Combine title and text
                full_text = f"{title} {selftext}".strip()
                if full_text:
                    score = await self._analyze_text_sentiment(full_text)

                    sentiment_data = SentimentData(
                        symbol=symbol,
                        source=SentimentSource.REDDIT,
                        sentiment_score=score['score'],
                        confidence=score['confidence'],
                        volume=1,
                        timestamp=datetime.fromtimestamp(post_data['created_utc']),
                        raw_data={
                            'post_id': post_data['id'],
                            'subreddit': post_data['subreddit'],
                            'title': title[:100],
                            'score': post_data['score'],
                            'num_comments': post_data['num_comments']
                        }
                    )
                    sentiments.append(sentiment_data)

            # Aggregate sentiment
            if sentiments:
                avg_score = np.mean([s.sentiment_score for s in sentiments])
                total_volume = len(sentiments)
                avg_confidence = np.mean([s.confidence for s in sentiments])

                return [SentimentData(
                    symbol=symbol,
                    source=SentimentSource.REDDIT,
                    sentiment_score=avg_score,
                    confidence=avg_confidence,
                    volume=total_volume,
                    timestamp=datetime.utcnow()
                )]
            else:
                return []

        except Exception as e:
            logger.error(f"Reddit sentiment analysis failed: {e}")
            return []

    async def _analyze_news_sentiment(self, symbol: str, hours: int) -> List[SentimentData]:
        """Analyze news sentiment for a symbol."""
        if not self.news_api_key:
            logger.warning("News API key not configured")
            return []

        try:
            # NewsAPI integration
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f"{symbol} OR ${symbol}",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 50,
                "apiKey": self.news_api_key,
                "from": (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
            }

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"News API error: {response.status}")
                    return []

                data = await response.json()

            articles = data.get('articles', [])
            if not articles:
                return []

            # Analyze sentiment
            sentiments = []
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')

                # Combine title and description
                full_text = f"{title} {description}".strip()
                if full_text:
                    score = await self._analyze_text_sentiment(full_text)

                    sentiment_data = SentimentData(
                        symbol=symbol,
                        source=SentimentSource.NEWS,
                        sentiment_score=score['score'],
                        confidence=score['confidence'],
                        volume=1,
                        timestamp=datetime.fromisoformat(
                            article['publishedAt'].replace('Z', '+00:00')
                        ),
                        raw_data={
                            'title': title,
                            'source': article.get('source', {}).get('name'),
                            'url': article.get('url'),
                            'description': description[:200] if description else None
                        }
                    )
                    sentiments.append(sentiment_data)

            # Aggregate sentiment
            if sentiments:
                avg_score = np.mean([s.sentiment_score for s in sentiments])
                total_volume = len(sentiments)
                avg_confidence = np.mean([s.confidence for s in sentiments])

                return [SentimentData(
                    symbol=symbol,
                    source=SentimentSource.NEWS,
                    sentiment_score=avg_score,
                    confidence=avg_confidence,
                    volume=total_volume,
                    timestamp=datetime.utcnow()
                )]
            else:
                return []

        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return []

    async def get_fear_greed_index(self) -> Dict[str, Any]:
        """Get current Fear & Greed Index."""
        try:
            # Check cache first (Fear & Greed Index updates every few hours)
            cache_key = datetime.utcnow().strftime("%Y-%m-%d-%H")
            if cache_key in self.fear_greed_cache:
                return self.fear_greed_cache[cache_key]

            # Fetch from API
            url = "https://api.alternative.me/fng/"
            params = {"limit": 1}

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Fear & Greed API error: {response.status}")
                    return {"value": 50, "classification": "Neutral"}  # Default

                data = await response.json()

            if data.get('data'):
                fng_data = data['data'][0]
                result = {
                    "value": int(fng_data['value']),
                    "classification": fng_data['value_classification'],
                    "timestamp": datetime.fromtimestamp(int(fng_data['timestamp'])),
                    "source": "alternative.me"
                }

                # Cache result
                self.fear_greed_cache[cache_key] = result
                return result
            else:
                return {"value": 50, "classification": "Neutral"}

        except Exception as e:
            logger.error(f"Fear & Greed Index fetch failed: {e}")
            return {"value": 50, "classification": "Neutral"}

    async def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using available models."""
        try:
            # Clean text
            text = self._clean_text(text)

            if not text:
                return {"score": 0.0, "confidence": 0.0}

            # Use advanced model if available
            if self.sentiment_model:
                try:
                    results = self.sentiment_model(text)
                    # Convert to -1 to 1 scale
                    score = self._convert_roberta_sentiment(results)
                    confidence = max([r['score'] for r in results])
                    return {"score": score, "confidence": confidence}
                except Exception as e:
                    logger.warning(f"Advanced model failed, falling back to TextBlob: {e}")

            # Fallback to TextBlob or basic analysis
            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 to 1
                subjectivity = blob.sentiment.subjectivity  # 0 to 1

                return {
                    "score": polarity,
                    "confidence": min(subjectivity + 0.5, 1.0)  # Boost confidence
                }
            else:
                # Basic sentiment analysis using keyword matching
                return self._basic_sentiment_analysis(text)

        except Exception as e:
            logger.error(f"Text sentiment analysis failed: {e}")
            return {"score": 0.0, "confidence": 0.0}

    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis."""
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags symbols but keep text
        text = re.sub(r'#', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()

    def _convert_roberta_sentiment(self, results: List[Dict]) -> float:
        """Convert RoBERTa sentiment results to -1 to 1 scale."""
        # RoBERTa labels: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
        scores = {r['label']: r['score'] for r in results}

        negative = scores.get('LABEL_0', 0.0)
        neutral = scores.get('LABEL_1', 0.0)
        positive = scores.get('LABEL_2', 0.0)

        # Convert to -1 to 1 scale
        if positive > negative:
            return positive
        elif negative > positive:
            return -negative
        else:
            return 0.0

    def _basic_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Basic sentiment analysis using keyword matching."""
        text_lower = text.lower()

        # Positive and negative word lists
        positive_words = [
            'bull', 'bullish', 'buy', 'long', 'up', 'rise', 'gain', 'profit',
            'moon', 'pump', 'green', 'breakout', 'support', 'accumulation',
            'hodl', 'diamond', 'strong', 'good', 'great', 'excellent'
        ]

        negative_words = [
            'bear', 'bearish', 'sell', 'short', 'down', 'fall', 'loss', 'crash',
            'dump', 'red', 'breakdown', 'resistance', 'distribution',
            'weak', 'bad', 'terrible', 'horrible', 'fear', 'panic'
        ]

        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total_words = len(text.split())
        if total_words == 0:
            return {"score": 0.0, "confidence": 0.0}

        # Calculate sentiment score
        if positive_count > negative_count:
            score = min(positive_count / total_words, 1.0)
        elif negative_count > positive_count:
            score = -min(negative_count / total_words, 1.0)
        else:
            score = 0.0

        # Calculate confidence based on word matches
        total_matches = positive_count + negative_count
        confidence = min(total_matches / max(total_words * 0.1, 1), 1.0)

        return {"score": score, "confidence": confidence}

    async def get_metrics(self) -> Dict[str, Any]:
        """Get sentiment analyzer metrics."""
        return {
            "model_loaded": self.sentiment_model is not None,
            "twitter_configured": self.twitter_bearer_token is not None,
            "reddit_configured": all([self.reddit_client_id, self.reddit_client_secret]),
            "news_configured": self.news_api_key is not None,
            "cache_size": len(self.fear_greed_cache)
        }