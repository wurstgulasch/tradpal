#!/usr/bin/env python3
"""
Tests for ML Outperformance Models - Ensemble methods and benchmark outperformance.

Tests the advanced ML training capabilities including:
- Ensemble model training
- Benchmark outperformance analysis
- Enhanced feature engineering
- Multiple ML algorithm integration
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, patch
from datetime import datetime, timedelta

from services.trading_service.trading_ai_service.ml_training.ml_trainer import (
    MLTrainerService,
    EnsembleTrainer,
    EventSystem
)


class TestEnsembleTrainer:
    """Test the EnsembleTrainer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ensemble_trainer = EnsembleTrainer()

    def test_create_ensemble_models(self):
        """Test creation of ensemble models."""
        models = self.ensemble_trainer.create_ensemble_models()

        expected_models = ['rf_optimized', 'gb_optimized', 'xgb_optimized', 'lgb_optimized', 'svm_optimized']
        assert all(model_name in models for model_name in expected_models)
        assert len(models) == 5

    def test_create_voting_ensemble(self):
        """Test creation of voting ensemble."""
        models = self.ensemble_trainer.create_ensemble_models()
        voting_ensemble = self.ensemble_trainer.create_voting_ensemble(models)

        assert voting_ensemble is not None
        assert hasattr(voting_ensemble, 'fit')
        assert hasattr(voting_ensemble, 'predict')

    @patch('services.trading_service.trading_ai_service.ml_training.ml_trainer.logger')
    def test_train_ensemble(self, mock_logger):
        """Test ensemble training with sample data."""
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 12

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        # Train ensemble
        result = self.ensemble_trainer.train_ensemble(X, y)

        # Verify results
        assert 'model' in result
        assert 'model_name' in result
        assert 'performance' in result
        assert 'all_models' in result
        assert 'model_performance' in result

        # Check performance metrics
        performance = result['performance']
        assert 'accuracy' in performance
        assert 'f1_score' in performance
        assert 'roc_auc' in performance

        # Verify model was trained
        assert result['model'] is not None


class TestMLOutperformanceTraining:
    """Test ML outperformance training capabilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.event_system = EventSystem()
        self.ml_trainer = MLTrainerService(self.event_system)

    @pytest.mark.asyncio
    async def test_train_benchmark_outperforming_model(self):
        """Test training benchmark-outperforming model."""
        # Mock the data fetching
        with patch.object(self.ml_trainer, '_fetch_enhanced_training_data',
                         new_callable=AsyncMock) as mock_fetch:

            # Create sample data
            dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
            np.random.seed(42)

            sample_data = pd.DataFrame({
                'timestamp': dates,
                'open': 50000 + np.random.normal(0, 1000, 1000),
                'high': 50000 + np.random.normal(0, 1000, 1000) + 100,
                'low': 50000 + np.random.normal(0, 1000, 1000) - 100,
                'close': 50000 + np.random.normal(0, 1000, 1000),
                'volume': np.random.normal(100, 20, 1000)
            })

            mock_fetch.return_value = sample_data

            # Train model
            result = await self.ml_trainer.train_benchmark_outperforming_model(
                symbol='BTCUSDT',
                timeframe='1h',
                start_date='2023-01-01',
                end_date='2023-02-01'
            )

            # Verify results
            assert result['success'] is True
            assert 'model_name' in result
            assert 'performance' in result
            assert 'outperformance_analysis' in result
            assert 'selected_model' in result

            # Check outperformance analysis
            analysis = result['outperformance_analysis']
            assert 'metric_outperformance' in analysis
            assert 'overall_outperformance_pct' in analysis
            assert analysis['overall_outperformance_pct'] >= 0

    def test_enhanced_feature_preparation(self):
        """Test enhanced feature preparation."""
        # Create sample data with required columns
        dates = pd.date_range('2023-01-01', periods=200, freq='1H')  # More data for indicators
        np.random.seed(42)

        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': 50000 + np.random.normal(0, 1000, 200),
            'high': 50000 + np.random.normal(0, 1000, 200) + 100,
            'low': 50000 + np.random.normal(0, 1000, 200) - 100,
            'close': 50000 + np.random.normal(0, 1000, 200),
            'volume': np.random.normal(100, 20, 200)
        })

        # Add target column
        sample_data['future_return'] = sample_data['close'].shift(-5) / sample_data['close'] - 1
        sample_data['target'] = (sample_data['future_return'] > 0).astype(int)

        # First add all required features by processing through the enhanced data pipeline
        df_with_features = self.ml_trainer._add_advanced_technical_indicators(sample_data.copy())
        df_with_features = self.ml_trainer._add_market_regime_indicators(df_with_features)
        df_with_features = self.ml_trainer._add_momentum_volatility_indicators(df_with_features)

        # Then prepare enhanced features
        X, y, feature_names = self.ml_trainer._prepare_enhanced_features(df_with_features)        # Verify results
        assert X.shape[0] > 0
        assert len(feature_names) > 12  # Should have more than basic features
        assert len(y) == X.shape[0]

        # Check that enhanced features are included
        enhanced_features = ['stoch_k', 'cci', 'adx', 'volatility_regime']
        for feature in enhanced_features:
            assert feature in feature_names

    def test_outperformance_analysis(self):
        """Test outperformance analysis calculation."""
        # Mock ensemble result
        ensemble_result = {
            'performance': {
                'accuracy': 0.65,
                'precision': 0.62,
                'recall': 0.60,
                'f1_score': 0.61,
                'roc_auc': 0.66
            },
            'model_name': 'voting_ensemble'
        }

        # Benchmark metrics
        benchmark_metrics = {
            'accuracy': 0.55,
            'precision': 0.52,
            'recall': 0.50,
            'f1_score': 0.51,
            'roc_auc': 0.53
        }

        # Sample data for analysis
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        # Analyze outperformance
        analysis = self.ml_trainer._analyze_outperformance(
            ensemble_result, benchmark_metrics, X, y
        )

        # Verify analysis
        assert 'metric_outperformance' in analysis
        assert 'overall_outperformance_pct' in analysis
        assert analysis['overall_outperformance_pct'] > 0  # Should show outperformance

        # Check individual metrics
        for metric in ['accuracy', 'f1_score', 'roc_auc']:
            assert metric in analysis['metric_outperformance']
            metric_data = analysis['metric_outperformance'][metric]
            assert 'improvement_pct' in metric_data
            assert metric_data['improvement_pct'] > 0  # Should show improvement

    @pytest.mark.asyncio
    async def test_enhanced_training_data_fetching(self):
        """Test enhanced training data fetching."""
        with patch.object(self.ml_trainer, '_fetch_training_data',
                         new_callable=AsyncMock) as mock_fetch:

            # Create sample data
            dates = pd.date_range('2023-01-01', periods=100, freq='1H')
            sample_data = pd.DataFrame({
                'timestamp': dates,
                'open': 50000 + np.random.normal(0, 1000, 100),
                'high': 50000 + np.random.normal(0, 1000, 100) + 100,
                'low': 50000 + np.random.normal(0, 1000, 100) - 100,
                'close': 50000 + np.random.normal(0, 1000, 100),
                'volume': np.random.normal(100, 20, 100)
            })

            mock_fetch.return_value = sample_data

            # Fetch enhanced data
            enhanced_data = await self.ml_trainer._fetch_enhanced_training_data(
                'BTCUSDT', '1h', '2023-01-01', '2023-02-01'
            )

            # Verify enhanced indicators were added
            enhanced_indicators = ['stoch_k', 'cci', 'adx', 'volatility_regime']
            for indicator in enhanced_indicators:
                assert indicator in enhanced_data.columns

    def test_advanced_technical_indicators(self):
        """Test calculation of advanced technical indicators."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        np.random.seed(42)

        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': 50000 + np.random.normal(0, 1000, 100),
            'high': 50000 + np.random.normal(0, 1000, 100) + 100,
            'low': 50000 + np.random.normal(0, 1000, 100) - 100,
            'close': 50000 + np.random.normal(0, 1000, 100),
            'volume': np.random.normal(100, 20, 100)
        })

        # Add advanced indicators
        enhanced_data = self.ml_trainer._add_advanced_technical_indicators(sample_data)

        # Verify indicators were added
        expected_indicators = ['stoch_k', 'stoch_d', 'williams_r', 'cci', 'cmf', 'obv']
        for indicator in expected_indicators:
            assert indicator in enhanced_data.columns
            assert not enhanced_data[indicator].isna().all()  # Should have some valid values

    def test_market_regime_indicators(self):
        """Test market regime indicator calculations."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        np.random.seed(42)

        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': 50000 + np.random.normal(0, 1000, 100),
            'high': 50000 + np.random.normal(0, 1000, 100) + 100,
            'low': 50000 + np.random.normal(0, 1000, 100) - 100,
            'close': 50000 + np.random.normal(0, 1000, 100),
            'volume': np.random.normal(100, 20, 100)
        })

        # Add market regime indicators
        regime_data = self.ml_trainer._add_market_regime_indicators(sample_data)

        # Verify indicators were added
        expected_indicators = ['adx', 'volatility_regime', 'trend_direction']
        for indicator in expected_indicators:
            assert indicator in regime_data.columns


class TestBenchmarkOutperformanceIntegration:
    """Integration tests for benchmark outperformance."""

    def setup_method(self):
        """Set up test fixtures."""
        self.event_system = EventSystem()
        self.ml_trainer = MLTrainerService(self.event_system)
        self.ensemble_trainer = EnsembleTrainer()

    def test_ensemble_model_selection(self):
        """Test that ensemble selects the best performing model."""
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        n_features = 12

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        # Train ensemble
        result = self.ensemble_trainer.train_ensemble(X, y)

        # Verify a model was selected
        assert result['model_name'] in ['rf_optimized', 'gb_optimized', 'xgb_optimized',
                                       'lgb_optimized', 'svm_optimized', 'voting_ensemble']

        # Verify performance is reasonable
        performance = result['performance']
        assert performance['accuracy'] > 0.5  # Should be better than random
        assert performance['f1_score'] > 0.4

    def test_outperformance_metrics_calculation(self):
        """Test calculation of outperformance metrics."""
        # Create mock performance data
        model_performance = {
            'accuracy': 0.70,
            'f1_score': 0.68,
            'roc_auc': 0.72
        }

        benchmark_metrics = {
            'accuracy': 0.55,
            'f1_score': 0.51,
            'roc_auc': 0.53
        }

        ensemble_result = {'performance': model_performance, 'model_name': 'test_ensemble'}
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        # Calculate outperformance
        analysis = self.ml_trainer._analyze_outperformance(
            ensemble_result, benchmark_metrics, X, y
        )

        # Verify calculations
        assert analysis['overall_outperformance_pct'] == 100.0  # All metrics outperform

        # Check individual improvements
        for metric in ['accuracy', 'f1_score', 'roc_auc']:
            metric_data = analysis['metric_outperformance'][metric]
            assert metric_data['improvement'] > 0
            assert metric_data['improvement_pct'] > 0
            assert metric_data['outperforms'] is True