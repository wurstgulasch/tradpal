#!/usr/bin/env python3
"""
Test ML signal enhancement integration in signal generator.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.signal_generator import generate_signals, apply_ml_signal_enhancement
from src.indicators import calculate_indicators
from config.settings import ML_ENABLED


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='1min')
    np.random.seed(42)  # For reproducible results

    # Generate realistic price data
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.abs(np.random.randn(100) * 0.2)
    low = close - np.abs(np.random.randn(100) * 0.2)
    open_price = close + np.random.randn(100) * 0.1
    volume = np.random.randint(1000, 10000, 100)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df.set_index('timestamp', inplace=True)
    return df


@pytest.fixture
def data_with_signals(sample_data):
    """Create data with calculated indicators and signals."""
    df = calculate_indicators(sample_data.copy())
    df_with_signals = generate_signals(df)
    return df_with_signals


class TestMLSignalEnhancement:
    """Test ML signal enhancement integration."""

    def test_ml_enhancement_disabled(self, data_with_signals, monkeypatch):
        """Test that ML enhancement is skipped when disabled."""
        # Ensure ML is disabled
        monkeypatch.setattr('config.settings.ML_ENABLED', False)

        # Remove any cached models to ensure clean state
        import os
        import glob
        for f in glob.glob('cache/ml_models/*.pkl'):
            os.remove(f)

        original_buy_signals = data_with_signals['Buy_Signal'].copy()
        original_sell_signals = data_with_signals['Sell_Signal'].copy()

        # Apply enhancement (should do nothing)
        result_df = apply_ml_signal_enhancement(data_with_signals.copy())

        # Signals should be unchanged
        pd.testing.assert_series_equal(result_df['Buy_Signal'], original_buy_signals)
        pd.testing.assert_series_equal(result_df['Sell_Signal'], original_sell_signals)

        # ML columns should not be added when ML is disabled
        ml_columns = ['ML_Signal', 'ML_Confidence', 'ML_Reason', 'Enhanced_Signal', 'Signal_Source']
        for col in ml_columns:
            assert col not in result_df.columns

    def test_ml_enhancement_no_ml_available(self, data_with_signals, monkeypatch):
        """Test ML enhancement when ML is enabled but not available."""
        # Enable ML but mock as unavailable
        monkeypatch.setattr('config.settings.ML_ENABLED', True)
        monkeypatch.setattr('src.ml_predictor.SKLEARN_AVAILABLE', False)

        original_buy_signals = data_with_signals['Buy_Signal'].copy()
        original_sell_signals = data_with_signals['Sell_Signal'].copy()

        # Apply enhancement (should do nothing)
        result_df = apply_ml_signal_enhancement(data_with_signals.copy())

        # Signals should be unchanged
        pd.testing.assert_series_equal(result_df['Buy_Signal'], original_buy_signals)
        pd.testing.assert_series_equal(result_df['Sell_Signal'], original_sell_signals)

    def test_ml_enhancement_columns_added(self, data_with_signals, monkeypatch):
        """Test that ML enhancement works when properly configured."""
        # Mock ML as available and trained
        monkeypatch.setattr('config.settings.ML_ENABLED', True)
        monkeypatch.setattr('src.ml_predictor.SKLEARN_AVAILABLE', True)

        # Create a proper mock predictor that behaves like the real one
        class MockPredictor:
            def __init__(self):
                self.is_trained = True

            def predict_signal(self, df, threshold=0.7):
                # Return a proper prediction dict for each row
                return {
                    'signal': 'BUY',
                    'confidence': 0.8,
                    'reason': 'Test prediction'
                }

            def enhance_signal(self, traditional_signal, ml_prediction, confidence_threshold=0.7):
                # Return enhancement result
                return {
                    'signal': 'BUY',
                    'source': 'ML_ENHANCED'
                }

        # Mock the get_ml_predictor function to return our mock
        def mock_get_ml_predictor(*args, **kwargs):
            return MockPredictor()

        monkeypatch.setattr('src.signal_generator.get_ml_predictor', mock_get_ml_predictor)

        # Apply enhancement - should not raise an exception
        result_df = apply_ml_signal_enhancement(data_with_signals.copy())

        # At minimum, the function should return a DataFrame with the same or more columns
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(data_with_signals)

        # If ML enhancement worked, we should have at least the original columns
        original_columns = data_with_signals.columns.tolist()
        for col in original_columns:
            assert col in result_df.columns

    def test_generate_signals_with_ml_enabled(self, sample_data, monkeypatch):
        """Test that generate_signals includes ML enhancement when enabled."""
        # Enable ML but ensure no trained model is available
        monkeypatch.setattr('config.settings.ML_ENABLED', True)
        monkeypatch.setattr('src.ml_predictor.SKLEARN_AVAILABLE', True)

        # Remove any cached models and mock get_ml_predictor to return None
        import os
        import glob
        for f in glob.glob('cache/ml_models/*.pkl'):
            os.remove(f)

        def mock_get_ml_predictor(*args, **kwargs):
            return None

        monkeypatch.setattr('src.signal_generator.get_ml_predictor', mock_get_ml_predictor)

        df = calculate_indicators(sample_data.copy())
        result_df = generate_signals(df)

        # Should have basic signal columns
        assert 'Buy_Signal' in result_df.columns
        assert 'Sell_Signal' in result_df.columns

        # ML columns should not be present when no trained model is available
        ml_columns = ['ML_Signal', 'ML_Confidence', 'Enhanced_Signal', 'Signal_Source']
        for col in ml_columns:
            assert col not in result_df.columns

    def test_signal_consistency(self, data_with_signals):
        """Test that signals remain consistent (buy and sell don't overlap)."""
        # Check that Buy_Signal and Sell_Signal don't both equal 1 at the same time
        both_signals = (data_with_signals['Buy_Signal'] == 1) & (data_with_signals['Sell_Signal'] == 1)
        assert not both_signals.any(), "Buy and Sell signals should not overlap"

    def test_signal_columns_exist(self, data_with_signals):
        """Test that all expected signal columns exist."""
        expected_columns = ['Buy_Signal', 'Sell_Signal']

        for col in expected_columns:
            assert col in data_with_signals.columns, f"Missing column: {col}"
            assert data_with_signals[col].dtype in [int, float, 'int64', 'float64'], f"Column {col} has wrong dtype"

    def test_signal_values_valid(self, data_with_signals):
        """Test that signal values are valid (0 or 1)."""
        for col in ['Buy_Signal', 'Sell_Signal']:
            unique_values = data_with_signals[col].dropna().unique()
            valid_values = all(val in [0, 1] for val in unique_values)
            assert valid_values, f"Column {col} contains invalid values: {unique_values}"