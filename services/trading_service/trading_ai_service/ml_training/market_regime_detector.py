#!/usr/bin/env python3
"""
Advanced Market Regime Detection Service.

This service provides sophisticated market regime detection using:
- Multi-dimensional regime classification
- Unsupervised learning for regime identification
- Temporal regime transition analysis
- Regime-specific feature engineering
- Confidence scoring for regime predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Advanced market regime detection using machine learning."""

    def __init__(self, n_regimes: int = 6, lookback_window: int = 252):
        """
        Initialize market regime detector.

        Args:
            n_regimes: Number of market regimes to detect
            lookback_window: Historical window for regime analysis (default: 1 year)
        """
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window

        # Regime detection models
        self.kmeans_model = None
        self.gmm_model = None
        self.scaler = StandardScaler()

        # Regime characteristics
        self.regime_characteristics = {}
        self.regime_transition_matrix = None

        # Regime labels and descriptions
        self.regime_labels = {
            0: 'bull_trend',
            1: 'bear_trend',
            2: 'sideways',
            3: 'high_volatility',
            4: 'low_volatility',
            5: 'breakout'
        }

    def detect_market_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes using advanced ML techniques.

        Args:
            data: OHLCV data with technical indicators

        Returns:
            DataFrame with regime classifications and confidence scores
        """
        if data.empty or len(data) < self.lookback_window:
            logger.warning("Insufficient data for regime detection")
            return data

        logger.info(f"Detecting market regimes using {self.n_regimes} regime classes...")

        # Prepare regime features
        regime_features = self._extract_regime_features(data)

        # Train regime detection models if not already trained
        if self.kmeans_model is None:
            self._train_regime_models(regime_features)

        # Classify regimes
        data = data.copy()
        data = self._classify_regimes(data, regime_features)

        # Add regime transition analysis
        data = self._add_regime_transitions(data)

        # Add regime-specific features
        data = self._add_regime_specific_features(data)

        logger.info("Market regime detection completed")
        return data

    def _extract_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive features for regime detection."""
        features = pd.DataFrame(index=data.index)

        # Trend features
        features['trend_strength'] = data.get('adx', 25)
        features['trend_direction'] = data.get('trend_direction_num', 0)

        # Volatility features
        features['volatility_20'] = data.get('volatility_20', data['close'].pct_change().rolling(20).std())
        features['volatility_50'] = data.get('volatility_50', data['close'].pct_change().rolling(50).std())
        features['volatility_ratio'] = features['volatility_20'] / features['volatility_50']

        # Momentum features
        features['rsi'] = data.get('rsi', 50)
        features['momentum_20'] = data.get('momentum_20d', 0)
        features['macd_signal'] = data.get('macd_signal', 0)

        # Volume features (if available)
        if 'volume' in data.columns:
            features['volume_trend'] = data['volume'].rolling(20).mean().pct_change(5)
            features['volume_volatility'] = data['volume'].rolling(20).std() / data['volume'].rolling(20).mean()

        # Price action features
        features['returns'] = data['close'].pct_change().fillna(0)
        features['returns_volatility'] = features['returns'].rolling(20).std()
        features['sharpe_ratio'] = features['returns'].rolling(20).mean() / features['returns_volatility']

        # Bollinger Band features
        if 'bb_width' in data.columns:
            features['bb_width'] = data['bb_width']
            features['bb_position'] = (data['close'] - data.get('bb_lower', data['close'])) / \
                                    (data.get('bb_upper', data['close'] + 1) - data.get('bb_lower', data['close']))

        # Statistical features
        features['skewness'] = features['returns'].rolling(20).skew()
        features['kurtosis'] = features['returns'].rolling(20).kurt()

        # Fill NaN values
        features = features.ffill().bfill().fillna(0)

        return features

    def _train_regime_models(self, features: pd.DataFrame):
        """Train unsupervised models for regime detection."""
        # Prepare training data
        X = self.scaler.fit_transform(features.values)

        # Remove NaN and infinite values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Train K-means clustering
        self.kmeans_model = KMeans(
            n_clusters=self.n_regimes,
            random_state=42,
            n_init=10
        )
        kmeans_labels = self.kmeans_model.fit_predict(X)

        # Train Gaussian Mixture Model
        self.gmm_model = GaussianMixture(
            n_components=self.n_regimes,
            random_state=42,
            covariance_type='full'
        )
        gmm_labels = self.gmm_model.fit_predict(X)

        # Analyze regime characteristics
        self._analyze_regime_characteristics(features, kmeans_labels)

        # Calculate transition probabilities
        self._calculate_regime_transitions(kmeans_labels)

    def _analyze_regime_characteristics(self, features: pd.DataFrame, labels: np.ndarray):
        """Analyze characteristics of each detected regime."""
        for regime_id in range(self.n_regimes):
            regime_mask = labels == regime_id
            regime_data = features[regime_mask]

            if len(regime_data) > 0:
                self.regime_characteristics[regime_id] = {
                    'avg_volatility': regime_data['volatility_20'].mean(),
                    'avg_trend_strength': regime_data['trend_strength'].mean(),
                    'avg_momentum': regime_data['momentum_20'].mean(),
                    'avg_returns': regime_data['returns'].mean(),
                    'volatility_percentile': stats.percentileofscore(features['volatility_20'], regime_data['volatility_20'].mean()),
                    'trend_percentile': stats.percentileofscore(features['trend_strength'], regime_data['trend_strength'].mean()),
                    'sample_size': len(regime_data)
                }

    def _calculate_regime_transitions(self, labels: np.ndarray):
        """Calculate regime transition probability matrix."""
        n_samples = len(labels)
        self.regime_transition_matrix = np.zeros((self.n_regimes, self.n_regimes))

        for i in range(1, n_samples):
            from_regime = labels[i-1]
            to_regime = labels[i]
            self.regime_transition_matrix[from_regime, to_regime] += 1

        # Convert to probabilities
        row_sums = self.regime_transition_matrix.sum(axis=1)
        self.regime_transition_matrix = self.regime_transition_matrix / row_sums[:, np.newaxis]
        self.regime_transition_matrix = np.nan_to_num(self.regime_transition_matrix, 0)

    def _classify_regimes(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Classify market regimes for each data point."""
        # Scale features
        X = self.scaler.transform(features.values)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Get predictions from both models
        kmeans_labels = self.kmeans_model.predict(X)
        gmm_labels = self.gmm_model.predict(X)
        gmm_probs = self.gmm_model.predict_proba(X)

        # Ensemble classification with confidence scoring
        data = data.copy()
        data['regime_kmeans'] = kmeans_labels
        data['regime_gmm'] = gmm_labels
        data['regime_confidence'] = np.max(gmm_probs, axis=1)

        # Final regime classification (weighted ensemble)
        data['market_regime'] = kmeans_labels  # Primary classification
        data['regime_label'] = data['market_regime'].map(self.regime_labels)

        # Add regime characteristics
        for col in ['avg_volatility', 'avg_trend_strength', 'avg_momentum', 'avg_returns']:
            data[f'regime_{col}'] = data['market_regime'].map(
                lambda x: self.regime_characteristics.get(x, {}).get(col, 0)
            )

        return data

    def _add_regime_transitions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add regime transition analysis."""
        data = data.copy()

        # Calculate regime changes
        data['regime_changed'] = data['market_regime'].diff() != 0
        data['regime_change_points'] = data['regime_changed'].astype(int)

        # Calculate time in current regime
        regime_groups = (data['market_regime'] != data['market_regime'].shift()).cumsum()
        data['regime_duration'] = data.groupby(regime_groups).cumcount() + 1

        # Transition probabilities
        data['transition_probability'] = 0.0
        for i in range(1, len(data)):
            from_regime = data.iloc[i-1]['market_regime']
            to_regime = data.iloc[i]['market_regime']
            if from_regime != to_regime:
                prob = self.regime_transition_matrix[int(from_regime), int(to_regime)]
                data.iloc[i, data.columns.get_loc('transition_probability')] = prob

        return data

    def _add_regime_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add regime-specific feature engineering."""
        data = data.copy()

        # Regime-specific normalization
        for regime_id in range(self.n_regimes):
            regime_mask = data['market_regime'] == regime_id

            if regime_mask.sum() > 10:  # Minimum samples for normalization
                # Normalize returns within regime
                regime_returns = data.loc[regime_mask, 'returns']
                data.loc[regime_mask, 'regime_normalized_returns'] = (
                    regime_returns - regime_returns.mean()
                ) / regime_returns.std()

                # Regime-specific volatility adjustment
                regime_vol = data.loc[regime_mask, 'volatility_20'].mean()
                data.loc[regime_mask, 'regime_volatility_adjusted'] = (
                    data.loc[regime_mask, 'returns'] / regime_vol
                )

        # Fill NaN values from regime-specific features
        data = data.fillna(0)

        return data

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regime statistics."""
        stats = {
            'n_regimes': self.n_regimes,
            'regime_labels': self.regime_labels,
            'regime_characteristics': self.regime_characteristics,
            'transition_matrix': self.regime_transition_matrix.tolist() if self.regime_transition_matrix is not None else None
        }

        return stats

    def predict_regime_probability(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities for new data."""
        if self.gmm_model is None:
            raise ValueError("Model not trained yet")

        X = self.scaler.transform(features.values)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        return self.gmm_model.predict_proba(X)