"""
Market Regime Detection Service - Clustering Engine
Identifies market regimes using unsupervised clustering algorithms.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ClusteringAlgorithm(Enum):
    """Available clustering algorithms."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    AGGLOMERATIVE = "agglomerative"


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class ClusteringResult:
    """Result of clustering analysis."""
    algorithm: ClusteringAlgorithm
    n_clusters: int
    labels: np.ndarray
    centroids: Optional[np.ndarray]
    silhouette_score: Optional[float]
    calinski_harabasz_score: Optional[float]
    regime_labels: List[MarketRegime]
    confidence_scores: List[float]
    features_used: List[str]


@dataclass
class MarketRegimeAnalysis:
    """Complete market regime analysis result."""
    timestamp: datetime
    symbol: str
    current_regime: MarketRegime
    regime_confidence: float
    regime_history: List[Tuple[datetime, MarketRegime, float]]
    clustering_results: List[ClusteringResult]
    feature_importance: Dict[str, float]
    volatility_regime: str
    trend_strength: float


class ClusteringEngine:
    """Engine for market regime detection using clustering algorithms."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for clustering engine."""
        return {
            'algorithms': [ClusteringAlgorithm.KMEANS, ClusteringAlgorithm.DBSCAN],
            'kmeans': {
                'n_clusters_range': [3, 4, 5, 6],
                'max_iter': 300,
                'n_init': 10,
                'random_state': 42
            },
            'dbscan': {
                'eps_range': [0.3, 0.5, 0.7, 1.0],
                'min_samples_range': [5, 10, 15]
            },
            'agglomerative': {
                'n_clusters_range': [3, 4, 5],
                'linkage': 'ward'
            },
            'feature_weights': {
                'returns': 0.3,
                'volatility': 0.25,
                'volume': 0.2,
                'momentum': 0.15,
                'rsi': 0.1
            }
        }

    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for clustering analysis.

        Args:
            data: DataFrame with OHLCV and technical indicators

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        features = []

        # Price returns
        if 'close' in data.columns:
            returns = data['close'].pct_change().fillna(0)
            features.append(('returns', returns))

        # Volatility (rolling standard deviation)
        if 'close' in data.columns:
            volatility = data['close'].pct_change().rolling(20).std().fillna(0)
            features.append(('volatility', volatility))

        # Volume (normalized)
        if 'volume' in data.columns:
            volume_norm = (data['volume'] - data['volume'].mean()) / data['volume'].std()
            features.append(('volume', volume_norm.fillna(0)))

        # Momentum (rate of change)
        if 'close' in data.columns:
            momentum = data['close'].pct_change(10).fillna(0)
            features.append(('momentum', momentum))

        # RSI if available
        if 'rsi' in data.columns:
            features.append(('rsi', data['rsi']))

        # MACD if available
        if 'macd' in data.columns:
            features.append(('macd', data['macd']))

        # Bollinger Bands position
        if all(col in data.columns for col in ['close', 'bb_upper', 'bb_lower']):
            bb_position = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            bb_position = bb_position.fillna(0.5)  # Neutral position for NaN
            features.append(('bb_position', bb_position))

        # Create feature matrix
        feature_names = [name for name, _ in features]
        feature_matrix = np.column_stack([values for _, values in features])

        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        feature_matrix = self.scaler.fit_transform(feature_matrix)

        return feature_matrix, feature_names

    def apply_kmeans(self, X: np.ndarray, n_clusters: int) -> ClusteringResult:
        """Apply K-Means clustering."""
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=self.config['kmeans']['max_iter'],
            n_init=self.config['kmeans']['n_init'],
            random_state=self.config['kmeans']['random_state']
        )

        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        # Calculate quality metrics
        silhouette = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else None
        calinski = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else None

        # Convert cluster labels to market regimes
        regime_labels, confidence_scores = self._clusters_to_regimes(
            labels, centroids, X, ClusteringAlgorithm.KMEANS
        )

        return ClusteringResult(
            algorithm=ClusteringAlgorithm.KMEANS,
            n_clusters=n_clusters,
            labels=labels,
            centroids=centroids,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski,
            regime_labels=regime_labels,
            confidence_scores=confidence_scores,
            features_used=self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else []
        )

    def apply_dbscan(self, X: np.ndarray, eps: float, min_samples: int) -> ClusteringResult:
        """Apply DBSCAN clustering."""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        # DBSCAN doesn't have centroids
        centroids = None

        # Calculate quality metrics (only for valid clusters)
        valid_labels = labels[labels != -1]  # Exclude noise points
        if len(np.unique(valid_labels)) > 1:
            silhouette = silhouette_score(X[labels != -1], valid_labels)
            calinski = calinski_harabasz_score(X[labels != -1], valid_labels)
        else:
            silhouette = None
            calinski = None

        # Convert cluster labels to market regimes
        regime_labels, confidence_scores = self._clusters_to_regimes(
            labels, centroids, X, ClusteringAlgorithm.DBSCAN
        )

        return ClusteringResult(
            algorithm=ClusteringAlgorithm.DBSCAN,
            n_clusters=len(np.unique(labels[labels != -1])),  # Exclude noise
            labels=labels,
            centroids=centroids,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski,
            regime_labels=regime_labels,
            confidence_scores=confidence_scores,
            features_used=self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else []
        )

    def apply_agglomerative(self, X: np.ndarray, n_clusters: int) -> ClusteringResult:
        """Apply Agglomerative clustering."""
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.config['agglomerative']['linkage']
        )

        labels = agg.fit_predict(X)
        centroids = None  # Agglomerative doesn't provide centroids

        # Calculate quality metrics
        silhouette = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else None
        calinski = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else None

        # Convert cluster labels to market regimes
        regime_labels, confidence_scores = self._clusters_to_regimes(
            labels, centroids, X, ClusteringAlgorithm.AGGLOMERATIVE
        )

        return ClusteringResult(
            algorithm=ClusteringAlgorithm.AGGLOMERATIVE,
            n_clusters=n_clusters,
            labels=labels,
            centroids=centroids,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski,
            regime_labels=regime_labels,
            confidence_scores=confidence_scores,
            features_used=self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else []
        )

    def _clusters_to_regimes(self, labels: np.ndarray, centroids: Optional[np.ndarray],
                           X: np.ndarray, algorithm: ClusteringAlgorithm) -> Tuple[List[MarketRegime], List[float]]:
        """
        Convert cluster labels to market regime classifications.

        This is a simplified mapping - in practice, this would be more sophisticated
        based on domain knowledge and historical analysis.
        """
        unique_labels = np.unique(labels)
        regime_labels = []
        confidence_scores = []

        for label in labels:
            if label == -1:  # DBSCAN noise points
                regime_labels.append(MarketRegime.SIDEWAYS)
                confidence_scores.append(0.5)
            else:
                # Simplified regime mapping based on cluster characteristics
                # In practice, this would use trained models or expert rules
                if algorithm == ClusteringAlgorithm.KMEANS and centroids is not None:
                    # Map based on centroid characteristics
                    centroid = centroids[label]
                    regime, confidence = self._centroid_to_regime(centroid)
                else:
                    # Default mapping for other algorithms
                    regime, confidence = self._default_cluster_mapping(label, len(unique_labels))

                regime_labels.append(regime)
                confidence_scores.append(confidence)

        return regime_labels, confidence_scores

    def _centroid_to_regime(self, centroid: np.ndarray) -> Tuple[MarketRegime, float]:
        """Map centroid characteristics to market regime."""
        # Simplified mapping - in practice would be more sophisticated
        # Assume features are ordered: [returns, volatility, volume, momentum, ...]

        returns_idx = 0  # returns feature index
        volatility_idx = 1  # volatility feature index

        returns_val = centroid[returns_idx] if len(centroid) > returns_idx else 0
        volatility_val = centroid[volatility_idx] if len(centroid) > volatility_idx else 0

        # High positive returns + low volatility = Bull market
        if returns_val > 0.5 and volatility_val < -0.5:
            return MarketRegime.BULL_MARKET, 0.8
        # High negative returns + high volatility = Bear market
        elif returns_val < -0.5 and volatility_val > 0.5:
            return MarketRegime.BEAR_MARKET, 0.8
        # High volatility regardless of returns = High volatility regime
        elif volatility_val > 0.5:
            return MarketRegime.HIGH_VOLATILITY, 0.7
        # Low volatility = Low volatility regime
        elif volatility_val < -0.5:
            return MarketRegime.LOW_VOLATILITY, 0.7
        # Default to sideways
        else:
            return MarketRegime.SIDEWAYS, 0.6

    def _default_cluster_mapping(self, label: int, n_clusters: int) -> Tuple[MarketRegime, float]:
        """Default cluster to regime mapping."""
        regimes = [
            MarketRegime.BULL_MARKET,
            MarketRegime.BEAR_MARKET,
            MarketRegime.SIDEWAYS,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.LOW_VOLATILITY
        ]

        regime = regimes[label % len(regimes)]
        confidence = 0.6  # Default confidence

        return regime, confidence

    def detect_regime(self, data: pd.DataFrame, symbol: str) -> MarketRegimeAnalysis:
        """
        Perform complete market regime detection analysis.

        Args:
            data: OHLCV data with technical indicators
            symbol: Trading symbol

        Returns:
            Complete market regime analysis
        """
        self.logger.info(f"Starting market regime detection for {symbol}")

        # Prepare features
        X, feature_names = self.prepare_features(data)

        if len(X) == 0:
            raise ValueError("No valid features could be extracted from data")

        # Apply clustering algorithms
        clustering_results = []

        # K-Means with different cluster numbers
        for n_clusters in self.config['kmeans']['n_clusters_range']:
            try:
                result = self.apply_kmeans(X, n_clusters)
                clustering_results.append(result)
            except Exception as e:
                self.logger.warning(f"K-Means with {n_clusters} clusters failed: {e}")

        # DBSCAN with different parameters
        for eps in self.config['dbscan']['eps_range']:
            for min_samples in self.config['dbscan']['min_samples_range']:
                try:
                    result = self.apply_dbscan(X, eps, min_samples)
                    clustering_results.append(result)
                except Exception as e:
                    self.logger.warning(f"DBSCAN with eps={eps}, min_samples={min_samples} failed: {e}")

        # Select best clustering result
        best_result = self._select_best_clustering_result(clustering_results)

        # Determine current regime
        current_regime = best_result.regime_labels[-1]  # Most recent data point
        regime_confidence = best_result.confidence_scores[-1]

        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(data)

        # Determine volatility regime
        volatility_regime = self._determine_volatility_regime(data)

        # Create regime history (simplified - would use historical data)
        regime_history = [(datetime.now(), current_regime, regime_confidence)]

        # Calculate feature importance (simplified)
        feature_importance = {name: 1.0 / len(feature_names) for name in feature_names}

        analysis = MarketRegimeAnalysis(
            timestamp=datetime.now(),
            symbol=symbol,
            current_regime=current_regime,
            regime_confidence=regime_confidence,
            regime_history=regime_history,
            clustering_results=clustering_results,
            feature_importance=feature_importance,
            volatility_regime=volatility_regime,
            trend_strength=trend_strength
        )

        self.logger.info(f"Market regime detection completed for {symbol}: {current_regime.value}")
        return analysis

    def _select_best_clustering_result(self, results: List[ClusteringResult]) -> ClusteringResult:
        """Select the best clustering result based on quality metrics."""
        if not results:
            raise ValueError("No clustering results available")

        # Score each result
        scored_results = []
        for result in results:
            score = 0

            # Prefer higher silhouette scores
            if result.silhouette_score is not None:
                score += result.silhouette_score * 0.5

            # Prefer higher Calinski-Harabasz scores
            if result.calinski_harabasz_score is not None:
                score += min(result.calinski_harabasz_score / 1000, 1.0) * 0.3

            # Prefer reasonable number of clusters (not too many, not too few)
            if 2 <= result.n_clusters <= 6:
                score += 0.2

            scored_results.append((result, score))

        # Return result with highest score
        best_result, _ = max(scored_results, key=lambda x: x[1])
        return best_result

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength indicator."""
        if 'close' not in data.columns or len(data) < 20:
            return 0.0

        # Simple trend strength using linear regression slope
        prices = data['close'].values[-20:]  # Last 20 periods
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]

        # Normalize slope by average price
        avg_price = np.mean(prices)
        normalized_slope = slope / avg_price if avg_price != 0 else 0

        # Convert to 0-1 scale
        trend_strength = 1 / (1 + np.exp(-normalized_slope * 10))  # Sigmoid

        return float(trend_strength)

    def _determine_volatility_regime(self, data: pd.DataFrame) -> str:
        """Determine volatility regime."""
        if 'close' not in data.columns or len(data) < 20:
            return "unknown"

        # Calculate recent volatility
        returns = data['close'].pct_change().dropna()
        recent_volatility = returns.tail(20).std()

        # Calculate historical volatility (longer period)
        historical_volatility = returns.tail(100).std() if len(returns) >= 100 else returns.std()

        if recent_volatility > historical_volatility * 1.5:
            return "high_volatility"
        elif recent_volatility < historical_volatility * 0.7:
            return "low_volatility"
        else:
            return "normal_volatility"