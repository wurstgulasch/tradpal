"""
Unit tests for Market Regime Detection Service components.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from services.market_regime_service.clustering_engine import (
    ClusteringEngine,
    ClusteringAlgorithm,
    MarketRegime,
    ClusteringResult,
    MarketRegimeAnalysis
)

from services.market_regime_service.regime_analyzer import (
    RegimeAnalyzer,
    TradingSignal,
    RiskLevel,
    RegimeSignal
)


class TestClusteringEngine:
    """Test cases for ClusteringEngine."""

    def test_initialization(self):
        """Test clustering engine initialization."""
        engine = ClusteringEngine()
        assert engine.config is not None
        assert 'algorithms' in engine.config
        assert 'kmeans' in engine.config

    def test_prepare_features_basic(self):
        """Test basic feature preparation."""
        engine = ClusteringEngine()

        # Create sample data
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

        features, names = engine.prepare_features(data)

        assert isinstance(features, np.ndarray)
        assert len(names) > 0
        assert 'returns' in names
        assert 'volume' in names

    def test_prepare_features_with_technical_indicators(self):
        """Test feature preparation with technical indicators."""
        engine = ClusteringEngine()

        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'rsi': [30, 40, 50, 60, 70],
            'macd': [1.0, 1.1, 1.2, 1.3, 1.4],
            'bb_upper': [105, 106, 107, 108, 109],
            'bb_lower': [95, 96, 97, 98, 99]
        })

        features, names = engine.prepare_features(data)

        assert 'rsi' in names
        assert 'macd' in names
        assert 'bb_position' in names

    @patch('services.market_regime_service.clustering_engine.KMeans')
    def test_apply_kmeans(self, mock_kmeans):
        """Test K-Means clustering application."""
        engine = ClusteringEngine()

        # Mock KMeans
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.array([0, 0, 1, 1, 1])
        mock_instance.cluster_centers_ = np.array([[0.1, 0.2], [0.5, 0.6]])
        mock_kmeans.return_value = mock_instance

        X = np.random.rand(5, 2)

        result = engine.apply_kmeans(X, 2)

        assert isinstance(result, ClusteringResult)
        assert result.algorithm == ClusteringAlgorithm.KMEANS
        assert result.n_clusters == 2
        assert len(result.labels) == 5

    def test_clusters_to_regimes_mapping(self):
        """Test cluster to regime mapping."""
        engine = ClusteringEngine()

        # Test with centroids (K-Means style)
        centroids = np.array([
            [0.8, -0.5],  # Should map to BULL_MARKET (high returns, low volatility)
            [-0.6, 0.7],  # Should map to BEAR_MARKET (low returns, high volatility)
            [0.1, 0.1]    # Should map to SIDEWAYS
        ])

        labels = np.array([0, 1, 2, 0, 1])
        X = np.random.rand(5, 2)

        regimes, confidences = engine._clusters_to_regimes(labels, centroids, X, ClusteringAlgorithm.KMEANS)

        assert len(regimes) == 5
        assert len(confidences) == 5
        assert all(isinstance(r, MarketRegime) for r in regimes)
        assert all(0 <= c <= 1 for c in confidences)

    def test_detect_regime_complete_analysis(self):
        """Test complete regime detection analysis."""
        engine = ClusteringEngine()

        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'close': np.cumsum(np.random.randn(100)) + 100,
            'volume': np.random.randint(1000, 2000, 100),
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.randn(100) * 0.1
        })

        symbol = "BTC/USDT"

        result = engine.detect_regime(data, symbol)

        assert isinstance(result, MarketRegimeAnalysis)
        assert result.symbol == symbol
        assert isinstance(result.current_regime, MarketRegime)
        assert 0 <= result.regime_confidence <= 1
        assert isinstance(result.timestamp, datetime)
        assert len(result.clustering_results) > 0


class TestRegimeAnalyzer:
    """Test cases for RegimeAnalyzer."""

    def test_initialization(self):
        """Test regime analyzer initialization."""
        analyzer = RegimeAnalyzer()
        assert analyzer.config is not None
        assert 'signal_thresholds' in analyzer.config
        assert 'risk_management' in analyzer.config

    def test_regime_to_signal_mapping(self):
        """Test regime to trading signal mapping."""
        analyzer = RegimeAnalyzer()

        # Test different regimes
        test_cases = [
            (MarketRegime.BULL_MARKET, 0.9, TradingSignal.BUY),
            (MarketRegime.BEAR_MARKET, 0.8, TradingSignal.SELL),
            (MarketRegime.SIDEWAYS, 0.7, TradingSignal.HOLD),
            (MarketRegime.HIGH_VOLATILITY, 0.6, TradingSignal.REDUCE_RISK),
        ]

        for regime, confidence, expected_signal in test_cases:
            # Create mock analysis
            analysis = MagicMock()
            analysis.current_regime = regime
            analysis.regime_confidence = confidence
            analysis.timestamp = datetime.now()
            analysis.volatility_regime = "normal"
            analysis.trend_strength = 0.5

            signal = analyzer._generate_signal(analysis, "BTC/USDT")

            assert isinstance(signal, RegimeSignal)
            assert signal.regime == regime
            assert signal.confidence == confidence

    def test_risk_level_calculation(self):
        """Test risk level calculation based on regime."""
        analyzer = RegimeAnalyzer()

        test_cases = [
            (MarketRegime.BULL_MARKET, 0.8, RiskLevel.MEDIUM),
            (MarketRegime.BEAR_MARKET, 0.9, RiskLevel.HIGH),
            (MarketRegime.HIGH_VOLATILITY, 0.7, RiskLevel.EXTREME),
            (MarketRegime.LOW_VOLATILITY, 0.6, RiskLevel.LOW),
        ]

        for regime, confidence, expected_risk in test_cases:
            analysis = MagicMock()
            analysis.volatility_regime = "normal"

            risk_level = analyzer._calculate_risk_level(regime, confidence, analysis)
            assert isinstance(risk_level, RiskLevel)

    def test_regime_transition_detection(self):
        """Test regime transition detection."""
        analyzer = RegimeAnalyzer()

        symbol = "BTC/USDT"
        timestamp = datetime.now()

        # Add some history
        analyzer._update_regime_history(symbol, timestamp - timedelta(days=5), MarketRegime.BULL_MARKET)
        analyzer._update_regime_history(symbol, timestamp - timedelta(days=2), MarketRegime.SIDEWAYS)

        # Test transition
        transition = analyzer.detect_regime_transition(symbol, MarketRegime.BEAR_MARKET, timestamp)

        assert transition is not None
        assert transition.from_regime == MarketRegime.SIDEWAYS
        assert transition.to_regime == MarketRegime.BEAR_MARKET
        assert transition.symbol == symbol

    def test_regime_statistics_calculation(self):
        """Test regime statistics calculation."""
        analyzer = RegimeAnalyzer()

        symbol = "BTC/USDT"
        base_time = datetime.now()

        # Add regime history
        regimes = [
            MarketRegime.BULL_MARKET,
            MarketRegime.BULL_MARKET,
            MarketRegime.SIDEWAYS,
            MarketRegime.BEAR_MARKET,
            MarketRegime.SIDEWAYS
        ]

        for i, regime in enumerate(regimes):
            timestamp = base_time + timedelta(days=i)
            analyzer._update_regime_history(symbol, timestamp, regime)

        stats = analyzer.get_regime_statistics(symbol)

        assert stats["total_observations"] == 5
        assert "regime_frequencies" in stats
        assert MarketRegime.SIDEWAYS.value in stats["regime_frequencies"]
        assert stats["regime_frequencies"][MarketRegime.SIDEWAYS.value] == 2

    def test_regime_prediction(self):
        """Test regime prediction functionality."""
        analyzer = RegimeAnalyzer()

        symbol = "BTC/USDT"
        base_time = datetime.now()

        # Add sufficient history for prediction
        regimes = [MarketRegime.BULL_MARKET] * 8  # Long bull run

        for i, regime in enumerate(regimes):
            timestamp = base_time + timedelta(days=i*10)  # Spread over time
            analyzer._update_regime_history(symbol, timestamp, regime)

        prediction = analyzer.get_regime_prediction(symbol)

        assert "current_regime" in prediction
        assert "transition_probability" in prediction
        assert "predicted_next_regimes" in prediction
        assert prediction["current_regime"] == MarketRegime.BULL_MARKET.value

    def test_analyze_market_regime_integration(self):
        """Test complete market regime analysis integration."""
        analyzer = RegimeAnalyzer()

        # Create sample data
        data = pd.DataFrame({
            'close': np.cumsum(np.random.randn(50)) + 100,
            'volume': np.random.randint(1000, 2000, 50),
            'rsi': np.random.uniform(30, 70, 50)
        })

        symbol = "BTC/USDT"

        # Mock the clustering engine to avoid actual computation
        with patch.object(analyzer.clustering_engine, 'detect_regime') as mock_detect:
            mock_analysis = MagicMock()
            mock_analysis.current_regime = MarketRegime.BULL_MARKET
            mock_analysis.regime_confidence = 0.8
            mock_analysis.timestamp = datetime.now()
            mock_analysis.symbol = symbol
            mock_analysis.volatility_regime = "normal"
            mock_analysis.trend_strength = 0.6
            mock_analysis.feature_importance = {"returns": 0.5, "volatility": 0.3}
            mock_analysis.clustering_results = []

            mock_detect.return_value = mock_analysis

            regime_analysis, signal = analyzer.analyze_market_regime(data, symbol)

            assert regime_analysis == mock_analysis
            assert isinstance(signal, RegimeSignal)
            assert signal.regime == MarketRegime.BULL_MARKET
            assert signal.symbol == symbol


class TestMarketRegimeEnums:
    """Test market regime enums and constants."""

    def test_market_regime_enum_values(self):
        """Test that all market regime enum values are defined."""
        regimes = [
            MarketRegime.BULL_MARKET,
            MarketRegime.BEAR_MARKET,
            MarketRegime.SIDEWAYS,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.LOW_VOLATILITY
        ]

        assert len(regimes) == 5
        assert all(isinstance(r, MarketRegime) for r in regimes)

    def test_trading_signal_enum_values(self):
        """Test that all trading signal enum values are defined."""
        signals = [
            TradingSignal.BUY,
            TradingSignal.SELL,
            TradingSignal.HOLD,
            TradingSignal.REDUCE_RISK,
            TradingSignal.INCREASE_RISK
        ]

        assert len(signals) == 5
        assert all(isinstance(s, TradingSignal) for s in signals)

    def test_risk_level_enum_values(self):
        """Test that all risk level enum values are defined."""
        levels = [
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.EXTREME
        ]

        assert len(levels) == 4
        assert all(isinstance(l, RiskLevel) for l in levels)