#!/usr/bin/env python3
"""
Test TA-Lib integration and fallback functionality.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Check TA-Lib availability
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

from src.indicators import ema, rsi, bb, atr, adx


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


class TestTALibIntegration:
    """Test TA-Lib integration and fallback functionality."""

    def test_ta_lib_availability(self):
        """Test TA-Lib availability detection."""
        # This should not raise an exception
        assert isinstance(TALIB_AVAILABLE, bool)

    def test_ema_with_ta_lib(self, sample_data):
        """Test EMA calculation with TA-Lib if available."""
        ema_values = ema(sample_data['close'], period=9)
        assert len(ema_values) == len(sample_data)
        assert not ema_values.isna().all()

        # Check that values are reasonable
        assert ema_values.iloc[-1] > 0
        assert not np.isnan(ema_values.iloc[-1])

    def test_ema_fallback(self, sample_data, monkeypatch):
        """Test EMA fallback when TA-Lib is not available."""
        # Mock TA-Lib as unavailable
        monkeypatch.setattr('src.indicators.TALIB_AVAILABLE', False)

        # Reload the module to use fallback
        import importlib
        import src.indicators
        importlib.reload(src.indicators)

        ema_values = src.indicators.ema(sample_data['close'], period=9)
        assert len(ema_values) == len(sample_data)
        assert not ema_values.isna().all()

    def test_rsi_with_ta_lib(self, sample_data):
        """Test RSI calculation with TA-Lib if available."""
        rsi_values = rsi(sample_data['close'], period=14)
        assert len(rsi_values) == len(sample_data)

        # RSI should be between 0 and 100
        valid_rsi = rsi_values.dropna()
        if len(valid_rsi) > 0:
            assert all(0 <= x <= 100 for x in valid_rsi)

    def test_rsi_fallback(self, sample_data, monkeypatch):
        """Test RSI fallback when TA-Lib is not available."""
        # Mock TA-Lib as unavailable
        monkeypatch.setattr('src.indicators.TALIB_AVAILABLE', False)

        # Reload the module
        import importlib
        import src.indicators
        importlib.reload(src.indicators)

        rsi_values = src.indicators.rsi(sample_data['close'], period=14)
        assert len(rsi_values) == len(sample_data)

        # RSI should still be between 0 and 100
        valid_rsi = rsi_values.dropna()
        if len(valid_rsi) > 0:
            assert all(0 <= x <= 100 for x in valid_rsi)

    def test_bb_with_ta_lib(self, sample_data):
        """Test Bollinger Bands calculation with TA-Lib if available."""
        bb_upper, bb_middle, bb_lower = bb(sample_data['close'], period=20, std_dev=2)

        assert len(bb_upper) == len(sample_data)
        assert len(bb_middle) == len(sample_data)
        assert len(bb_lower) == len(sample_data)

        # Upper should be above middle, middle above lower
        valid_idx = ~(bb_upper.isna() | bb_middle.isna() | bb_lower.isna())
        if valid_idx.any():
            assert all(bb_upper[valid_idx] >= bb_middle[valid_idx])
            assert all(bb_middle[valid_idx] >= bb_lower[valid_idx])

    def test_bb_fallback(self, sample_data, monkeypatch):
        """Test Bollinger Bands fallback when TA-Lib is not available."""
        # Mock TA-Lib as unavailable
        monkeypatch.setattr('src.indicators.TALIB_AVAILABLE', False)

        # Reload the module
        import importlib
        import src.indicators
        importlib.reload(src.indicators)

        bb_upper, bb_middle, bb_lower = src.indicators.bb(sample_data['close'], period=20, std_dev=2)

        assert len(bb_upper) == len(sample_data)
        assert len(bb_middle) == len(sample_data)
        assert len(bb_lower) == len(sample_data)

    def test_atr_with_ta_lib(self, sample_data):
        """Test ATR calculation with TA-Lib if available."""
        atr_values = atr(sample_data['high'], sample_data['low'], sample_data['close'], period=14)

        assert len(atr_values) == len(sample_data)
        # ATR should be positive
        valid_atr = atr_values.dropna()
        if len(valid_atr) > 0:
            assert all(x >= 0 for x in valid_atr)

    def test_atr_fallback(self, sample_data, monkeypatch):
        """Test ATR fallback when TA-Lib is not available."""
        # Mock TA-Lib as unavailable
        monkeypatch.setattr('src.indicators.TALIB_AVAILABLE', False)

        # Reload the module
        import importlib
        import src.indicators
        importlib.reload(src.indicators)

        atr_values = src.indicators.atr(sample_data['high'], sample_data['low'], sample_data['close'], period=14)

        assert len(atr_values) == len(sample_data)
        # ATR should still be positive
        valid_atr = atr_values.dropna()
        if len(valid_atr) > 0:
            assert all(x >= 0 for x in valid_atr)

    def test_adx_with_ta_lib(self, sample_data):
        """Test ADX calculation with TA-Lib if available."""
        adx_values, di_plus, di_minus = adx(sample_data['high'], sample_data['low'], sample_data['close'], period=14)

        assert len(adx_values) == len(sample_data)
        assert len(di_plus) == len(sample_data)
        assert len(di_minus) == len(sample_data)
        # ADX should be between 0 and 100
        valid_adx = adx_values.dropna()
        if len(valid_adx) > 0:
            assert all(0 <= x <= 100 for x in valid_adx)

    def test_adx_fallback(self, sample_data, monkeypatch):
        """Test ADX fallback when TA-Lib is not available."""
        # Mock TA-Lib as unavailable
        monkeypatch.setattr('src.indicators.TALIB_AVAILABLE', False)

        # Reload the module
        import importlib
        import src.indicators
        importlib.reload(src.indicators)

        adx_values, di_plus, di_minus = src.indicators.adx(sample_data['high'], sample_data['low'], sample_data['close'], period=14)

        assert len(adx_values) == len(sample_data)
        assert len(di_plus) == len(sample_data)
        assert len(di_minus) == len(sample_data)
        # ADX should still be between 0 and 100
        valid_adx = adx_values.dropna()
        if len(valid_adx) > 0:
            assert all(0 <= x <= 100 for x in valid_adx)

    def test_calculate_indicators_with_fallback(self, sample_data, monkeypatch):
        """Test that calculate_indicators works with TA-Lib fallback."""
        # Mock TA-Lib as unavailable
        monkeypatch.setattr('src.indicators.TALIB_AVAILABLE', False)

        # Reload the module
        import importlib
        import src.indicators
        importlib.reload(src.indicators)

        # Config with ADX enabled
        config = {
            'ema': {'enabled': True, 'periods': [9, 21]},
            'rsi': {'enabled': True, 'period': 14},
            'bb': {'enabled': True, 'period': 20, 'std_dev': 2},
            'atr': {'enabled': True, 'period': 14},
            'adx': {'enabled': True, 'period': 14}
        }

        result_df = src.indicators.calculate_indicators(sample_data.copy(), config=config)

        # Should have all expected columns
        expected_columns = ['EMA9', 'EMA21', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'ADX']
        for col in expected_columns:
            assert col in result_df.columns, f"Missing column: {col}"
            assert not result_df[col].isna().all(), f"Column {col} is all NaN"