#!/usr/bin/env python3
"""
Tests for Market Regime Analysis Module

This module contains comprehensive unit tests for the market regime detection,
multi-timeframe analysis, and adaptive strategy components.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from services.monitoring_service.mlops_service.market_regime_analysis import (
    MarketRegimeDetector, MultiTimeframeAnalyzer, AdaptiveStrategyManager,
    MarketRegime, RegimeConfig, TimeframeConfig,
    detect_market_regime, analyze_multi_timeframe, get_adaptive_strategy_config
)

class TestMarketRegimeDetector(unittest.TestCase):
    """Test cases for MarketRegimeDetector."""

    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        np.random.seed(42)

        # Create synthetic OHLCV data
        base_price = 100
        prices = [base_price]
        for i in range(1, 100):
            noise = np.random.normal(0, 0.01)
            new_price = prices[-1] * (1 + noise)
            prices.append(new_price)

        self.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': np.random.uniform(100000, 1000000, 100)
        }, index=dates)

    def test_regime_detection_basic(self):
        """Test basic regime detection functionality."""
        detector = MarketRegimeDetector()
        regimes = detector.detect_regime(self.test_data)

        # Check that regimes are detected
        self.assertIsInstance(regimes, pd.Series)
        self.assertEqual(len(regimes), len(self.test_data))
        self.assertTrue(all(isinstance(r, MarketRegime) for r in regimes))

    def test_regime_detection_with_config(self):
        """Test regime detection with custom configuration."""
        config = RegimeConfig(
            lookback_periods=[10, 20],
            volatility_threshold=2.0
        )
        detector = MarketRegimeDetector(config)
        regimes = detector.detect_regime(self.test_data)

        self.assertIsInstance(regimes, pd.Series)
        self.assertEqual(len(regimes), len(self.test_data))

    def test_trend_slope_calculation(self):
        """Test trend slope calculation."""
        detector = MarketRegimeDetector()
        slopes = detector._calculate_trend_slope(self.test_data['close'].values, 20)

        self.assertIsInstance(slopes, pd.Series)
        self.assertEqual(len(slopes), len(self.test_data))

    def test_bb_position_calculation(self):
        """Test Bollinger Band position calculation."""
        detector = MarketRegimeDetector()
        bb_pos = detector._calculate_bb_position(self.test_data['close'].values, 20)

        self.assertIsInstance(bb_pos, pd.Series)
        self.assertEqual(len(bb_pos), len(self.test_data))
        # BB position should be between -1 and 1
        self.assertTrue(all(bb_pos.between(-1, 1)))

    def test_regime_classification(self):
        """Test regime classification logic."""
        detector = MarketRegimeDetector()

        # Create mock indicators
        indicators = {
            'trend_slope_50': pd.Series([0.01] * 10),  # Uptrend
            'adx_50': pd.Series([30] * 10),  # Strong trend
            'atr_50': pd.Series([0.01] * 10),  # Normal volatility
            'rsi_50': pd.Series([50] * 10),  # Neutral
            'bb_position_50': pd.Series([0] * 10),  # Middle
            'volume_sma_ratio': pd.Series([1] * 10),  # Normal volume
            'price_acceleration': pd.Series([0] * 10),  # No acceleration
        }

        regime = detector._classify_regime(indicators, 0)
        self.assertIsInstance(regime, MarketRegime)

class TestMultiTimeframeAnalyzer(unittest.TestCase):
    """Test cases for MultiTimeframeAnalyzer."""

    def setUp(self):
        """Set up multi-timeframe test data."""
        dates_1h = pd.date_range('2023-01-01', periods=50, freq='H')
        dates_4h = pd.date_range('2023-01-01', periods=12, freq='4H')

        np.random.seed(42)

        # Create 1h data
        prices_1h = 100 + np.random.normal(0, 1, 50).cumsum()
        data_1h = pd.DataFrame({
            'open': prices_1h,
            'high': prices_1h * 1.01,
            'low': prices_1h * 0.99,
            'close': prices_1h,
            'volume': np.random.uniform(100000, 1000000, 50)
        }, index=dates_1h)

        # Create 4h data
        prices_4h = 100 + np.random.normal(0, 2, 12).cumsum()
        data_4h = pd.DataFrame({
            'open': prices_4h,
            'high': prices_4h * 1.02,
            'low': prices_4h * 0.98,
            'close': prices_4h,
            'volume': np.random.uniform(200000, 2000000, 12)
        }, index=dates_4h)

        self.test_data_dict = {
            '1h': data_1h,
            '4h': data_4h
        }

    def test_multi_timeframe_analysis(self):
        """Test multi-timeframe analysis."""
        analyzer = MultiTimeframeAnalyzer()
        results = analyzer.analyze_multi_timeframe(self.test_data_dict)

        # Check result structure
        self.assertIn('regime_results', results)
        self.assertIn('alignment_score', results)
        self.assertIn('mtf_features', results)
        self.assertIn('timeframe_strength', results)
        self.assertIn('consensus_regime', results)

        # Check that all timeframes have regime results
        for tf in self.test_data_dict.keys():
            self.assertIn(tf, results['regime_results'])

        # Check alignment score is between 0 and 1
        self.assertGreaterEqual(results['alignment_score'], 0)
        self.assertLessEqual(results['alignment_score'], 1)

        # Check consensus regime
        self.assertIsInstance(results['consensus_regime'], MarketRegime)

    def test_regime_alignment_calculation(self):
        """Test regime alignment calculation."""
        analyzer = MultiTimeframeAnalyzer()

        # Create mock regime results
        regime_results = {
            '1h': pd.Series([MarketRegime.TREND_UP, MarketRegime.TREND_UP]),
            '4h': pd.Series([MarketRegime.TREND_UP, MarketRegime.SIDEWAYS])
        }

        alignment = analyzer._calculate_regime_alignment(regime_results)
        self.assertGreaterEqual(alignment, 0)
        self.assertLessEqual(alignment, 1)

    def test_timeframe_strength_calculation(self):
        """Test timeframe strength calculation."""
        analyzer = MultiTimeframeAnalyzer()
        strength = analyzer._calculate_timeframe_strength(self.test_data_dict)

        # Check that all timeframes have strength scores
        for tf in self.test_data_dict.keys():
            self.assertIn(tf, strength)
            self.assertGreaterEqual(strength[tf], 0)
            self.assertLessEqual(strength[tf], 1)

    def test_consensus_regime(self):
        """Test consensus regime determination."""
        analyzer = MultiTimeframeAnalyzer()

        # Test with single regime
        regime_results = {
            '1h': pd.Series([MarketRegime.TREND_UP]),
            '4h': pd.Series([MarketRegime.TREND_UP])
        }
        consensus = analyzer._get_consensus_regime(regime_results)
        self.assertEqual(consensus, MarketRegime.TREND_UP)

        # Test with mixed regimes
        regime_results = {
            '1h': pd.Series([MarketRegime.TREND_UP]),
            '4h': pd.Series([MarketRegime.SIDEWAYS])
        }
        consensus = analyzer._get_consensus_regime(regime_results)
        self.assertIn(consensus, [MarketRegime.TREND_UP, MarketRegime.SIDEWAYS])

class TestAdaptiveStrategyManager(unittest.TestCase):
    """Test cases for AdaptiveStrategyManager."""

    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2023-01-01', periods=50, freq='H')
        prices = 100 + np.random.normal(0, 1, 50).cumsum()

        self.test_data_dict = {
            '1h': pd.DataFrame({
                'open': prices,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': np.random.uniform(100000, 1000000, 50)
            }, index=dates)
        }

    def test_adaptive_config_generation(self):
        """Test adaptive configuration generation."""
        manager = AdaptiveStrategyManager()
        config = manager.get_adaptive_config(self.test_data_dict)

        # Check required fields
        required_fields = ['model_type', 'position_size', 'stop_loss', 'take_profit',
                          'features', 'current_regime', 'confidence_score']
        for field in required_fields:
            self.assertIn(field, config)

        # Check value ranges
        self.assertGreaterEqual(config['position_size'], 0)
        self.assertLessEqual(config['position_size'], 1)
        self.assertGreaterEqual(config['confidence_score'], 0)
        self.assertLessEqual(config['confidence_score'], 1)

    def test_strategy_configs_loading(self):
        """Test strategy configuration loading."""
        manager = AdaptiveStrategyManager()
        configs = manager.strategy_configs

        # Check that all regimes have configurations
        for regime in MarketRegime:
            self.assertIn(regime, configs)

            config = configs[regime]
            required_keys = ['model_type', 'position_size', 'stop_loss', 'take_profit', 'features']
            for key in required_keys:
                self.assertIn(key, config)

class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2023-01-01', periods=50, freq='H')
        prices = 100 + np.random.normal(0, 1, 50).cumsum()

        self.test_data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(100000, 1000000, 50)
        }, index=dates)

        self.test_data_dict = {'1h': self.test_data}

    def test_detect_market_regime_function(self):
        """Test detect_market_regime convenience function."""
        regimes = detect_market_regime(self.test_data)

        self.assertIsInstance(regimes, pd.Series)
        self.assertEqual(len(regimes), len(self.test_data))

    def test_analyze_multi_timeframe_function(self):
        """Test analyze_multi_timeframe convenience function."""
        results = analyze_multi_timeframe(self.test_data_dict)

        self.assertIn('consensus_regime', results)
        self.assertIn('alignment_score', results)

    def test_get_adaptive_strategy_config_function(self):
        """Test get_adaptive_strategy_config convenience function."""
        config = get_adaptive_strategy_config(self.test_data_dict)

        self.assertIn('model_type', config)
        self.assertIn('position_size', config)

class TestMarketRegimeEnum(unittest.TestCase):
    """Test MarketRegime enum."""

    def test_regime_enum_values(self):
        """Test that all expected regime values exist."""
        expected_regimes = [
            'trend_up', 'trend_down', 'mean_reversion', 'high_volatility',
            'low_volatility', 'sideways', 'breakout', 'consolidation'
        ]

        for regime in MarketRegime:
            self.assertIn(regime.value, expected_regimes)

    def test_regime_enum_uniqueness(self):
        """Test that all regime values are unique."""
        values = [regime.value for regime in MarketRegime]
        self.assertEqual(len(values), len(set(values)))

if __name__ == '__main__':
    unittest.main()