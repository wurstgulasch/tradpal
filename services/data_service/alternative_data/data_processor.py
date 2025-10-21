#!/usr/bin/env python3
"""
Alternative Data Processor - Feature Engineering Pipeline

Processes raw alternative data into ML-ready features for trading models.
Combines sentiment, on-chain, and economic data into comprehensive feature sets.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .__init__ import ProcessedFeatures, AlternativeDataPacket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlternativeDataProcessor:
    """
    Processes alternative data into ML features with normalization and feature engineering.
    """

    def __init__(self):
        self.scalers = {}
        self.feature_history = {}
        self.lookback_periods = [1, 7, 30]  # Days for rolling features

    async def process_to_features(
        self,
        data_packet: AlternativeDataPacket,
        include_sentiment: bool = True,
        include_onchain: bool = True,
        include_economic: bool = True
    ) -> ProcessedFeatures:
        """
        Process alternative data packet into ML features.

        Args:
            data_packet: Raw alternative data
            include_sentiment: Whether to include sentiment features
            include_onchain: Whether to include on-chain features
            include_economic: Whether to include economic features

        Returns:
            Processed features ready for ML models
        """
        try:
            features = {}

            # Process sentiment features
            if include_sentiment:
                sentiment_features = await self._process_sentiment_features(data_packet.sentiment_data)
                features.update(sentiment_features)

            # Process on-chain features
            if include_onchain:
                onchain_features = await self._process_onchain_features(data_packet.onchain_data)
                features.update(onchain_features)

            # Process economic features
            if include_economic:
                economic_features = await self._process_economic_features(data_packet.economic_data)
                features.update(economic_features)

            # Create composite features
            composite_features = await self._create_composite_features(
                features, data_packet.fear_greed_index
            )

            # Normalize features
            normalized_features = await self._normalize_features(features)
            normalized_composite = await self._normalize_features(composite_features)

            return ProcessedFeatures(
                symbol=data_packet.symbol,
                sentiment_features=normalized_features if include_sentiment else {},
                onchain_features={},  # Would be populated if include_onchain
                economic_features={},  # Would be populated if include_economic
                composite_features=normalized_composite,
                timestamp=data_packet.timestamp
            )

        except Exception as e:
            logger.error(f"Feature processing failed: {e}")
            # Return empty features on error
            return ProcessedFeatures(
                symbol=data_packet.symbol,
                sentiment_features={},
                onchain_features={},
                economic_features={},
                composite_features={},
                timestamp=data_packet.timestamp
            )

    async def _process_sentiment_features(self, sentiment_data: List) -> Dict[str, float]:
        """Process sentiment data into features."""
        try:
            if not sentiment_data:
                return {}

            features = {}

            # Aggregate sentiment by source
            twitter_sentiment = None
            reddit_sentiment = None
            news_sentiment = None

            for data in sentiment_data:
                if data.source.value == 'twitter':
                    twitter_sentiment = data
                elif data.source.value == 'reddit':
                    reddit_sentiment = data
                elif data.source.value == 'news':
                    news_sentiment = data

            # Twitter features
            if twitter_sentiment:
                features['twitter_sentiment'] = twitter_sentiment.sentiment_score
                features['twitter_confidence'] = twitter_sentiment.confidence
                features['twitter_volume'] = twitter_sentiment.volume

            # Reddit features
            if reddit_sentiment:
                features['reddit_sentiment'] = reddit_sentiment.sentiment_score
                features['reddit_confidence'] = reddit_sentiment.confidence
                features['reddit_volume'] = reddit_sentiment.volume

            # News features
            if news_sentiment:
                features['news_sentiment'] = news_sentiment.sentiment_score
                features['news_confidence'] = news_sentiment.confidence
                features['news_volume'] = news_sentiment.volume

            # Cross-source features
            sentiments = [s.sentiment_score for s in sentiment_data if s.sentiment_score is not None]
            if sentiments:
                features['avg_sentiment'] = np.mean(sentiments)
                features['sentiment_std'] = np.std(sentiments)
                features['sentiment_range'] = max(sentiments) - min(sentiments)

                # Sentiment momentum (compare to historical average)
                symbol_key = f"sentiment_{sentiment_data[0].symbol if sentiment_data else 'unknown'}"
                if symbol_key in self.feature_history:
                    historical_avg = np.mean([
                        h.get('avg_sentiment', 0)
                        for h in self.feature_history[symbol_key][-10:]  # Last 10 periods
                    ])
                    features['sentiment_momentum'] = features['avg_sentiment'] - historical_avg
                else:
                    features['sentiment_momentum'] = 0.0

            # Store for historical comparison
            symbol_key = f"sentiment_{sentiment_data[0].symbol if sentiment_data else 'unknown'}"
            if symbol_key not in self.feature_history:
                self.feature_history[symbol_key] = []
            self.feature_history[symbol_key].append(features.copy())

            # Keep only last 100 entries
            if len(self.feature_history[symbol_key]) > 100:
                self.feature_history[symbol_key] = self.feature_history[symbol_key][-100:]

            return features

        except Exception as e:
            logger.error(f"Sentiment feature processing failed: {e}")
            return {}

    async def _process_onchain_features(self, onchain_data: List) -> Dict[str, float]:
        """Process on-chain data into features."""
        try:
            if not onchain_data:
                return {}

            features = {}

            # Group by metric
            metrics_data = {}
            for data in onchain_data:
                metric_key = data.metric.value
                if metric_key not in metrics_data:
                    metrics_data[metric_key] = []
                metrics_data[metric_key].append(data)

            # Process each metric
            for metric, data_list in metrics_data.items():
                if not data_list:
                    continue

                values = [d.value for d in data_list]
                latest_value = values[-1] if values else 0

                # Basic features
                features[f'{metric}_current'] = latest_value
                features[f'{metric}_mean'] = np.mean(values) if values else 0
                features[f'{metric}_std'] = np.std(values) if len(values) > 1 else 0

                # Trend features (if we have historical data)
                if len(values) > 1:
                    features[f'{metric}_trend'] = (values[-1] - values[0]) / len(values)
                    features[f'{metric}_volatility'] = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0

                # Metric-specific features
                if metric == 'nvt_ratio':
                    # NVT ratio interpretation
                    features['nvt_overvalued'] = 1 if latest_value > 150 else 0
                    features['nvt_undervalued'] = 1 if latest_value < 50 else 0
                elif metric == 'active_addresses':
                    # Active addresses momentum
                    if len(values) > 7:
                        weekly_avg = np.mean(values[-7:])
                        prev_week_avg = np.mean(values[-14:-7]) if len(values) > 14 else weekly_avg
                        features['active_addresses_momentum'] = (weekly_avg - prev_week_avg) / prev_week_avg if prev_week_avg != 0 else 0

            return features

        except Exception as e:
            logger.error(f"On-chain feature processing failed: {e}")
            return {}

    async def _process_economic_features(self, economic_data: List) -> Dict[str, float]:
        """Process economic data into features."""
        try:
            if not economic_data:
                return {}

            features = {}

            # Group by indicator
            indicators_data = {}
            for data in economic_data:
                indicator_key = data.indicator.value
                if indicator_key not in indicators_data:
                    indicators_data[indicator_key] = []
                indicators_data[indicator_key].append(data)

            # Process each indicator
            for indicator, data_list in indicators_data.items():
                if not data_list:
                    continue

                values = [d.value for d in data_list]
                latest_value = values[-1] if values else 0

                features[f'{indicator}_current'] = latest_value

                # Indicator-specific interpretations
                if indicator == 'fed_funds_rate':
                    # Rate environment classification
                    features['rate_environment_low'] = 1 if latest_value < 2.0 else 0
                    features['rate_environment_normal'] = 1 if 2.0 <= latest_value <= 4.0 else 0
                    features['rate_environment_high'] = 1 if latest_value > 4.0 else 0
                elif indicator == 'cpi':
                    # Inflation environment
                    features['inflation_moderate'] = 1 if 290 <= latest_value <= 310 else 0
                    features['inflation_high'] = 1 if latest_value > 310 else 0
                    features['inflation_low'] = 1 if latest_value < 290 else 0
                elif indicator == 'unemployment_rate':
                    # Employment environment
                    features['employment_strong'] = 1 if latest_value < 4.0 else 0
                    features['employment_weak'] = 1 if latest_value > 6.0 else 0

            return features

        except Exception as e:
            logger.error(f"Economic feature processing failed: {e}")
            return {}

    async def _create_composite_features(self, features: Dict[str, float], fear_greed_index: Optional[float]) -> Dict[str, float]:
        """Create composite features combining multiple data sources."""
        try:
            composite = {}

            # Fear & Greed Index integration
            if fear_greed_index is not None:
                composite['fear_greed_index'] = fear_greed_index
                # Classification
                composite['fear_extreme'] = 1 if fear_greed_index < 25 else 0
                composite['greed_extreme'] = 1 if fear_greed_index > 75 else 0
                composite['neutral_sentiment'] = 1 if 45 <= fear_greed_index <= 55 else 0

            # Sentiment-On-Chain correlation features
            sentiment_keys = [k for k in features.keys() if 'sentiment' in k and 'avg' in k]
            onchain_keys = [k for k in features.keys() if any(metric in k for metric in ['nvt_ratio', 'active_addresses', 'transaction_volume'])]

            if sentiment_keys and onchain_keys:
                sentiment_vals = [features[k] for k in sentiment_keys if k in features]
                onchain_vals = [features[k] for k in onchain_keys if k in features]

                if sentiment_vals and onchain_vals:
                    # Correlation between sentiment and on-chain activity
                    try:
                        correlation = np.corrcoef(sentiment_vals, onchain_vals)[0, 1]
                        composite['sentiment_onchain_correlation'] = correlation
                    except:
                        composite['sentiment_onchain_correlation'] = 0.0

            # Economic-Sentiment interaction
            economic_keys = [k for k in features.keys() if any(indicator in k for indicator in ['fed_funds', 'cpi', 'unemployment'])]
            if sentiment_keys and economic_keys:
                sentiment_avg = np.mean([features[k] for k in sentiment_keys if k in features])
                economic_avg = np.mean([features[k] for k in economic_keys if k in features])

                # Economic policy impact on sentiment
                composite['economic_sentiment_impact'] = sentiment_avg * economic_avg

            # Market regime indicators
            if features:
                # Volatility proxy from feature variance
                feature_values = list(features.values())
                if len(feature_values) > 1:
                    composite['market_volatility_proxy'] = np.std(feature_values) / np.mean(np.abs(feature_values)) if np.mean(np.abs(feature_values)) != 0 else 0

                # Trend strength proxy
                positive_features = sum(1 for v in feature_values if v > 0)
                composite['trend_strength_proxy'] = positive_features / len(feature_values)

            # Risk indicators
            if 'nvt_ratio_current' in features and 'fear_greed_index' in composite:
                nvt = features['nvt_ratio_current']
                fear_greed = composite['fear_greed_index']

                # Risk score combining NVT and sentiment
                composite['combined_risk_score'] = (nvt / 100) * (100 - fear_greed) / 100

            return composite

        except Exception as e:
            logger.error(f"Composite feature creation failed: {e}")
            return {}

    async def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features using rolling statistics."""
        try:
            if not features:
                return {}

            normalized = {}

            for feature_name, value in features.items():
                # Get or create scaler for this feature
                if feature_name not in self.scalers:
                    self.scalers[feature_name] = StandardScaler()

                # For normalization, we need historical values
                # For now, use simple min-max scaling as approximation
                if feature_name in self.feature_history:
                    historical_values = [
                        h.get(feature_name, value)
                        for h in self.feature_history[feature_name][-30:]  # Last 30 periods
                    ]
                    if len(historical_values) > 1:
                        min_val = np.min(historical_values)
                        max_val = np.max(historical_values)
                        if max_val != min_val:
                            normalized[feature_name] = (value - min_val) / (max_val - min_val)
                        else:
                            normalized[feature_name] = 0.5  # Default for constant features
                    else:
                        normalized[feature_name] = value
                else:
                    # No history, return raw value
                    normalized[feature_name] = value

            return normalized

        except Exception as e:
            logger.error(f"Feature normalization failed: {e}")
            return features  # Return unnormalized on error

    async def get_feature_importance(self, symbol: str) -> Dict[str, float]:
        """Calculate feature importance based on historical variance."""
        try:
            symbol_key = f"features_{symbol}"
            if symbol_key not in self.feature_history:
                return {}

            # Analyze feature variance over time
            feature_variances = {}
            all_features = set()

            # Collect all feature names
            for historical in self.feature_history[symbol_key]:
                all_features.update(historical.keys())

            # Calculate variance for each feature
            for feature in all_features:
                values = [
                    h.get(feature, 0)
                    for h in self.feature_history[symbol_key]
                    if feature in h
                ]
                if len(values) > 1:
                    feature_variances[feature] = np.var(values)
                else:
                    feature_variances[feature] = 0.0

            # Normalize to get relative importance
            if feature_variances:
                max_variance = max(feature_variances.values())
                if max_variance > 0:
                    importance = {
                        k: v / max_variance
                        for k, v in feature_variances.items()
                    }
                    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

            return feature_variances

        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return {}

    async def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        return {
            "scalers_count": len(self.scalers),
            "historical_data_points": sum(len(v) for v in self.feature_history.values()),
            "lookback_periods": self.lookback_periods
        }