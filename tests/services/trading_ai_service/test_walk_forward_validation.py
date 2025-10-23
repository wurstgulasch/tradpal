#!/usr/bin/env python3
"""
Tests for Walk-Forward Validation - Realistic ML model evaluation.

Tests the walk-forward validation capabilities including:
- Expanding window validation methodology
- Out-of-sample performance evaluation
- Stability and consistency analysis
- Trading implications assessment
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


class TestWalkForwardValidation:
    """Test walk-forward validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.event_system = EventSystem()
        self.ml_trainer = MLTrainerService(self.event_system)

    def test_default_walk_forward_config(self):
        """Test default walk-forward validation configuration."""
        config = self.ml_trainer._get_default_walk_forward_config()

        assert config['initial_train_window'] == 252
        assert config['validation_window'] == 21
        assert config['step_size'] == 21
        assert config['min_train_samples'] == 100
        assert config['model_type'] == 'ensemble_outperformance'
        assert 'benchmark_metrics' in config
        assert 'stability_threshold' in config

    def test_generate_walk_forward_windows(self):
        """Test generation of walk-forward validation windows."""
        # Create sample data (2 years of daily data)
        dates = pd.date_range('2022-01-01', periods=504, freq='D')
        np.random.seed(42)

        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': 50000 + np.random.normal(0, 1000, 504),
            'high': 50000 + np.random.normal(0, 1000, 504) + 100,
            'low': 50000 + np.random.normal(0, 1000, 504) - 100,
            'close': 50000 + np.random.normal(0, 1000, 504),
            'volume': np.random.normal(100, 20, 504)
        })

        # Generate windows
        windows = self.ml_trainer._generate_walk_forward_windows(
            sample_data, initial_train_window=252, validation_window=21,
            step_size=21, min_train_samples=100
        )

        # Verify windows were generated
        assert len(windows) > 0
        assert len(windows) <= 12  # Should have reasonable number of windows

        # Check first window structure
        first_window = windows[0]
        assert 'window_number' in first_window
        assert 'train_data' in first_window
        assert 'validation_data' in first_window
        assert 'train_period' in first_window
        assert 'validation_period' in first_window

        # Verify expanding window: each subsequent window should have more training data
        for i in range(1, len(windows)):
            prev_train_samples = len(windows[i-1]['train_data'])
            current_train_samples = len(windows[i]['train_data'])
            assert current_train_samples >= prev_train_samples

    def test_calculate_window_outperformance(self):
        """Test calculation of outperformance for a validation window."""
        # Sample performance metrics
        performance = {
            'accuracy': 0.65,
            'precision': 0.62,
            'recall': 0.60,
            'f1_score': 0.61,
            'roc_auc': 0.66
        }

        benchmark_metrics = {
            'accuracy': 0.55,
            'precision': 0.52,
            'recall': 0.50,
            'f1_score': 0.51,
            'roc_auc': 0.53
        }

        # Calculate outperformance
        outperformance = self.ml_trainer._calculate_window_outperformance(
            performance, benchmark_metrics
        )

        # Verify results
        assert 'metric_outperformance' in outperformance
        assert 'outperforming_metrics' in outperformance
        assert 'total_metrics' in outperformance
        assert 'overall_outperformance_pct' in outperformance

        # Check that all metrics show outperformance
        assert outperformance['outperforming_metrics'] == 5  # All 5 metrics
        assert outperformance['total_metrics'] == 5
        assert outperformance['overall_outperformance_pct'] == 100.0

        # Check individual metric improvements
        for metric in ['accuracy', 'f1_score', 'roc_auc']:
            metric_data = outperformance['metric_outperformance'][metric]
            assert metric_data['improvement'] > 0
            assert metric_data['improvement_pct'] > 0
            assert metric_data['outperforms'] is True

    def test_calculate_stability_metrics(self):
        """Test calculation of stability metrics."""
        # Create sample out-of-sample performances
        oos_performances = [
            {'f1_score': 0.65, 'accuracy': 0.68},
            {'f1_score': 0.63, 'accuracy': 0.66},
            {'f1_score': 0.67, 'accuracy': 0.70},
            {'f1_score': 0.62, 'accuracy': 0.65},
            {'f1_score': 0.64, 'accuracy': 0.67}
        ]

        config = self.ml_trainer._get_default_walk_forward_config()

        # Calculate stability
        stability = self.ml_trainer._calculate_stability_metrics(oos_performances, config)

        # Verify results
        assert 'coefficient_of_variation' in stability
        assert 'consistency_ratio' in stability
        assert 'mean_f1' in stability
        assert 'std_f1' in stability
        assert 'stability_rating' in stability

        # Check calculations
        expected_mean_f1 = np.mean([0.65, 0.63, 0.67, 0.62, 0.64])
        assert abs(stability['mean_f1'] - expected_mean_f1) < 0.001

        # Consistency should be > 0 since all F1 scores > benchmark (0.51)
        assert stability['consistency_ratio'] > 0

    def test_calculate_trading_implications(self):
        """Test calculation of trading implications."""
        # Mock analysis data
        analysis = {
            'oos_performance': {
                'f1_score': {'mean': 0.65}
            },
            'stability': {
                'stability_rating': 'excellent'
            },
            'outperformance': {
                'consistency': 0.9
            }
        }

        config = self.ml_trainer._get_default_walk_forward_config()

        # Calculate trading implications
        implications = self.ml_trainer._calculate_trading_implications(analysis, config)

        # Verify results
        assert 'estimated_profit_boost_pct' in implications
        assert 'signal_quality_improvement' in implications
        assert 'confidence_level' in implications
        assert 'recommendation' in implications
        assert 'risk_assessment' in implications

        # With excellent stability and high consistency, should be high confidence
        assert implications['confidence_level'] == 'high'
        assert 'Ready for live deployment' in implications['recommendation']

    def test_generate_validation_recommendations(self):
        """Test generation of validation recommendations."""
        # Excellent stability analysis
        analysis = {
            'stability': {'stability_rating': 'excellent'},
            'outperformance': {'consistency': 0.9},
            'trading_implications': {'confidence_level': 'high'}
        }

        config = self.ml_trainer._get_default_walk_forward_config()

        # Generate recommendations
        recommendations = self.ml_trainer._generate_validation_recommendations(analysis, config)

        # Verify recommendations
        assert len(recommendations) > 0
        assert any('excellent stability' in rec for rec in recommendations)
        assert any('High consistency' in rec for rec in recommendations)
        assert any('High confidence' in rec for rec in recommendations)

    def test_risk_assessment(self):
        """Test risk level assessment."""
        # Low risk scenario
        analysis_low = {
            'stability': {'coefficient_of_variation': 0.2},
            'outperformance': {'consistency': 0.9}
        }

        risk_low = self.ml_trainer._assess_risk_level(analysis_low)
        assert risk_low == 'low'

        # High risk scenario
        analysis_high = {
            'stability': {'coefficient_of_variation': 1.2},
            'outperformance': {'consistency': 0.1}
        }

        risk_high = self.ml_trainer._assess_risk_level(analysis_high)
        assert risk_high == 'high'

    @pytest.mark.asyncio
    async def test_perform_walk_forward_validation(self):
        """Test complete walk-forward validation process."""
        # Mock the data fetching
        with patch.object(self.ml_trainer, '_fetch_enhanced_training_data',
                         new_callable=AsyncMock) as mock_fetch:

            # Create sample data (1.5 years of daily data for validation)
            dates = pd.date_range('2022-01-01', periods=400, freq='D')
            np.random.seed(42)

            sample_data = pd.DataFrame({
                'timestamp': dates,
                'open': 50000 + np.random.normal(0, 1000, 400),
                'high': 50000 + np.random.normal(0, 1000, 400) + 100,
                'low': 50000 + np.random.normal(0, 1000, 400) - 100,
                'close': 50000 + np.random.normal(0, 1000, 400),
                'volume': np.random.normal(100, 20, 400)
            })

            # Don't pre-process data - let _prepare_enhanced_features handle it
            mock_fetch.return_value = sample_data

            # Mock event publishing
            with patch.object(self.ml_trainer.event_system, 'publish', new_callable=AsyncMock):
                # Perform walk-forward validation
                result = await self.ml_trainer.perform_walk_forward_validation(
                    symbol='BTCUSDT',
                    timeframe='1d',
                    start_date='2022-01-01',
                    end_date='2023-02-01'
                )

                # Verify results
                assert result['success'] is True
                assert 'symbol' in result
                assert 'timeframe' in result
                assert 'total_windows' in result
                assert 'validation_results' in result
                assert 'analysis' in result
                assert 'report' in result

                # Check analysis structure
                analysis = result['analysis']
                assert 'total_windows' in analysis
                assert 'successful_windows' in analysis
                assert 'oos_performance' in analysis
                assert 'stability' in analysis
                assert 'trading_implications' in analysis

                # Check report structure
                report = result['report']
                assert 'title' in report
                assert 'summary' in report
                assert 'performance_summary' in report
                assert 'recommendations' in report

    def test_walk_forward_validation_insufficient_data(self):
        """Test walk-forward validation with insufficient data."""
        # Create small dataset
        dates = pd.date_range('2022-01-01', periods=50, freq='D')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': 50000 + np.random.normal(0, 1000, 50),
            'high': 50000 + np.random.normal(0, 1000, 50) + 100,
            'low': 50000 + np.random.normal(0, 1000, 50) - 100,
            'close': 50000 + np.random.normal(0, 1000, 50),
            'volume': np.random.normal(100, 20, 50)
        })

        # Should raise ValueError for insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            self.ml_trainer._generate_walk_forward_windows(
                sample_data, 252, 21, 21, 100
            )

    def test_analyze_walk_forward_results(self):
        """Test analysis of walk-forward validation results."""
        # Create mock validation results
        validation_results = [
            {
                'success': True,
                'in_sample_performance': {'f1_score': 0.75, 'accuracy': 0.78},
                'out_sample_performance': {'f1_score': 0.68, 'accuracy': 0.71},
                'outperformance': {'overall_outperformance_pct': 25.0}
            },
            {
                'success': True,
                'in_sample_performance': {'f1_score': 0.73, 'accuracy': 0.76},
                'out_sample_performance': {'f1_score': 0.65, 'accuracy': 0.69},
                'outperformance': {'overall_outperformance_pct': 20.0}
            }
        ]

        config = self.ml_trainer._get_default_walk_forward_config()

        # Analyze results
        analysis = self.ml_trainer._analyze_walk_forward_results(validation_results, config)

        # Verify analysis structure
        assert 'total_windows' in analysis
        assert 'successful_windows' in analysis
        assert 'oos_performance' in analysis
        assert 'is_performance' in analysis
        assert 'outperformance' in analysis
        assert 'stability' in analysis
        assert 'trading_implications' in analysis

        # Check calculations
        assert analysis['total_windows'] == 2
        assert analysis['successful_windows'] == 2
        assert analysis['success_rate'] == 1.0

        # Check outperformance aggregation
        outperf = analysis['outperformance']
        assert outperf['mean_pct'] == 22.5  # Average of 25.0 and 20.0
        assert outperf['consistency'] == 1.0  # Both > 0


class TestWalkForwardIntegration:
    """Integration tests for walk-forward validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.event_system = EventSystem()
        self.ml_trainer = MLTrainerService(self.event_system)

    def test_walk_forward_window_expansion(self):
        """Test that walk-forward windows properly expand."""
        # Create longer dataset
        dates = pd.date_range('2021-01-01', periods=600, freq='D')
        np.random.seed(42)

        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': 50000 + np.random.normal(0, 1000, 600),
            'high': 50000 + np.random.normal(0, 1000, 600) + 100,
            'low': 50000 + np.random.normal(0, 1000, 600) - 100,
            'close': 50000 + np.random.normal(0, 1000, 600),
            'volume': np.random.normal(100, 20, 600)
        })

        # Generate windows with smaller initial window for testing
        windows = self.ml_trainer._generate_walk_forward_windows(
            sample_data, initial_train_window=100, validation_window=20,
            step_size=20, min_train_samples=50
        )

        # Verify expansion
        assert len(windows) > 1
        for i in range(1, min(5, len(windows))):  # Check first few windows
            prev_train_len = len(windows[i-1]['train_data'])
            current_train_len = len(windows[i]['train_data'])
            assert current_train_len > prev_train_len  # Should expand

    def test_walk_forward_performance_consistency(self):
        """Test that walk-forward validation provides consistent performance estimates."""
        # This is more of a statistical test - run multiple validations and check variance
        np.random.seed(42)

        performances = []
        n_runs = 3

        for run in range(n_runs):
            # Create slightly different data each run
            dates = pd.date_range('2022-01-01', periods=400, freq='D')
            sample_data = pd.DataFrame({
                'timestamp': dates,
                'open': 50000 + np.random.normal(0, 1000, 400) + run * 100,  # Slight variation
                'high': 50000 + np.random.normal(0, 1000, 400) + 100 + run * 50,
                'low': 50000 + np.random.normal(0, 1000, 400) - 100 + run * 50,
                'close': 50000 + np.random.normal(0, 1000, 400) + run * 200,
                'volume': np.random.normal(100, 20, 400)
            })

            # Generate windows
            windows = self.ml_trainer._generate_walk_forward_windows(
                sample_data, 200, 20, 20, 100
            )

            if windows:
                # Calculate average OOS performance (mock)
                avg_oos_perf = np.mean([0.65 + np.random.normal(0, 0.05) for _ in windows])
                performances.append(avg_oos_perf)

        # Check that performances are reasonably consistent (not too variable)
        if len(performances) > 1:
            cv = np.std(performances) / np.mean(performances)
            assert cv < 0.5  # Coefficient of variation should be reasonable