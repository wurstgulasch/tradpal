import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.indicators import ema, rsi, bb, atr, adx, fibonacci_extensions, calculate_indicators, calculate_indicators_with_config
from src.signal_generator import generate_signals, calculate_risk_management
from config.settings import DEFAULT_INDICATOR_CONFIG

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
def data_with_indicators(sample_data):
    """Create data with calculated indicators."""
    return calculate_indicators(sample_data.copy())

class TestIndicators:
    """Test technical indicator calculations."""

    def test_ema_calculation(self, sample_data):
        """Test EMA calculation."""
        ema_values = ema(sample_data['close'], period=9)
        assert len(ema_values) == len(sample_data)
        assert not ema_values.isna().all()
        # EMA should be close to close price for longer periods
        assert abs(ema_values.iloc[-1] - sample_data['close'].iloc[-1]) < sample_data['close'].iloc[-1] * 0.1

    def test_rsi_calculation(self, sample_data):
        """Test RSI calculation."""
        rsi_values = rsi(sample_data['close'], period=14)
        assert len(rsi_values) == len(sample_data)
        assert not rsi_values.isna().all()
        # RSI should be between 0 and 100
        valid_rsi = rsi_values.dropna()
        assert all((valid_rsi >= 0) & (valid_rsi <= 100))

    def test_bb_calculation(self, sample_data):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = bb(sample_data['close'], period=20, std_dev=2)
        assert len(upper) == len(sample_data)
        assert len(middle) == len(sample_data)
        assert len(lower) == len(sample_data)
        # Check that we have some valid (non-NaN) values
        assert not upper.isna().all()
        assert not middle.isna().all()
        assert not lower.isna().all()
        # For valid data points, upper should be above middle, middle above lower
        valid_idx = upper.notna() & middle.notna() & lower.notna()
        if valid_idx.any():
            assert all(upper[valid_idx] >= middle[valid_idx])
            assert all(middle[valid_idx] >= lower[valid_idx])

    def test_atr_calculation(self, sample_data):
        """Test ATR calculation."""
        atr_values = atr(sample_data['high'], sample_data['low'], sample_data['close'], period=14)
        assert len(atr_values) == len(sample_data)
        assert not atr_values.isna().all()
        # ATR should be positive
        valid_atr = atr_values.dropna()
        assert all(valid_atr > 0)

    def test_adx_calculation(self, sample_data):
        """Test ADX calculation."""
        adx_values, di_plus, di_minus = adx(sample_data['high'], sample_data['low'], sample_data['close'], period=14)
        assert len(adx_values) == len(sample_data)
        assert len(di_plus) == len(sample_data)
        assert len(di_minus) == len(sample_data)
        # ADX should be between 0 and 100
        valid_adx = adx_values.dropna()
        assert all((valid_adx >= 0) & (valid_adx <= 100))

    def test_calculate_indicators(self, sample_data):
        """Test complete indicator calculation pipeline."""
        result = calculate_indicators(sample_data.copy())
        expected_columns = ['EMA9', 'EMA21', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']
        for col in expected_columns:
            assert col in result.columns
        assert len(result) == len(sample_data)

    def test_calculate_indicators_custom_config(self, sample_data):
        """Test indicator calculation with custom configuration."""
        custom_config = {
            'ema': {'enabled': True, 'periods': [5, 10]},
            'rsi': {'enabled': True, 'period': 21},
            'bb': {'enabled': True, 'period': 10, 'std_dev': 1.5},
            'atr': {'enabled': True, 'period': 21}
        }
        result = calculate_indicators_with_config(sample_data.copy(), config=custom_config)
        expected_columns = ['EMA5', 'EMA10', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']
        for col in expected_columns:
            assert col in result.columns
        assert len(result) == len(sample_data)
        # Check that custom periods are used (shorter EMA should be more responsive)
        assert 'EMA5' in result.columns
        assert 'EMA10' in result.columns

    def test_default_indicator_config(self):
        """Test that DEFAULT_INDICATOR_CONFIG is properly defined."""
        assert isinstance(DEFAULT_INDICATOR_CONFIG, dict)
        required_keys = ['ema', 'rsi', 'bb', 'atr']
        for key in required_keys:
            assert key in DEFAULT_INDICATOR_CONFIG
            assert isinstance(DEFAULT_INDICATOR_CONFIG[key], dict)
            assert 'enabled' in DEFAULT_INDICATOR_CONFIG[key]

class TestSignalGenerator:
    """Test signal generation logic."""

    def test_generate_signals(self, data_with_indicators):
        """Test signal generation."""
        result = generate_signals(data_with_indicators.copy())
        assert 'Buy_Signal' in result.columns
        assert 'Sell_Signal' in result.columns
        # Signals should be 0 or 1
        assert all(result['Buy_Signal'].isin([0, 1]))
        assert all(result['Sell_Signal'].isin([0, 1]))

    def test_calculate_risk_management(self, data_with_indicators):
        """Test risk management calculation."""
        signals_data = generate_signals(data_with_indicators.copy())
        result = calculate_risk_management(signals_data.copy())

        expected_columns = ['Position_Size_Absolute', 'Position_Size_Percent',
                          'Stop_Loss_Buy', 'Take_Profit_Buy', 'Leverage']
        for col in expected_columns:
            assert col in result.columns

        # Position size percent should be reasonable
        valid_pct = result['Position_Size_Percent'].dropna()
        assert all((valid_pct > 0) & (valid_pct < 100))  # Should be less than 100%

class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_full_pipeline(self, sample_data):
        """Test the complete analysis pipeline."""
        # Calculate indicators
        data = calculate_indicators(sample_data.copy())

        # Generate signals
        data = generate_signals(data)

        # Calculate risk management
        data = calculate_risk_management(data)

        # Check that all expected columns exist
        required_columns = [
            'EMA9', 'EMA21', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR',
            'Buy_Signal', 'Sell_Signal', 'Position_Size_Absolute', 'Position_Size_Percent',
            'Stop_Loss_Buy', 'Take_Profit_Buy', 'Leverage'
        ]

        for col in required_columns:
            assert col in data.columns, f"Missing column: {col}"

        # Check data integrity
        assert not data.empty
        # Data length may be reduced due to indicator calculations (dropna)
        assert len(data) <= len(sample_data)
        assert len(data) >= len(sample_data) * 0.8  # At least 80% of data should remain

if __name__ == "__main__":
    pytest.main([__file__])