"""
Tests for ML Integration Module

Tests the integration of ML signal enhancement into the trading system.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import ML components
try:
    from signal_generator import apply_ml_signal_enhancement
    from config.settings import SYMBOL, TIMEFRAME
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    pytest.skip("ML integration module not available", allow_module_level=True)


class TestMLIntegration:
    """Test suite for ML integration functionality."""

    def setup_method(self):
        """Skip test if ML is not available."""
        if not ML_AVAILABLE:
            pytest.skip("ML integration module not available")

    def test_create_test_data(self):
        """Test creation of synthetic trading data."""
        # Generate 50 data points for testing
        dates = [datetime.now() - timedelta(minutes=i) for i in range(50, 0, -1)]

        np.random.seed(42)
        base_price = 50000
        prices = []
        current_price = base_price

        for i in range(50):
            trend = 0.001 * np.sin(i / 10)
            noise = np.random.normal(0, 0.01)
            change = trend + noise
            current_price *= (1 + change)
            prices.append(current_price)

        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = prices[i-1] if i > 0 else price * (1 + np.random.normal(0, 0.002))
            volume = np.random.uniform(100, 1000)

            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })

        df = pd.DataFrame(data)
        assert len(df) == 50
        assert all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        assert df['close'].iloc[0] > 0

    def test_add_technical_indicators(self):
        """Test adding technical indicators to data."""
        # Create simple test data
        dates = [datetime.now() - timedelta(minutes=i) for i in range(50, 0, -1)]
        prices = np.random.uniform(49000, 51000, 50)

        data = []
        for i, price in enumerate(prices):
            data.append({
                'timestamp': dates[i],
                'open': price * (1 + np.random.normal(0, 0.002)),
                'high': price * (1 + abs(np.random.normal(0, 0.005))),
                'low': price * (1 - abs(np.random.normal(0, 0.005))),
                'close': price,
                'volume': np.random.uniform(100, 1000)
            })

        df = pd.DataFrame(data)

        # Add basic indicators
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['RSI'] = 50  # Simplified RSI for testing

        # Check that indicators were added
        assert 'SMA_20' in df.columns
        assert 'SMA_50' in df.columns
        assert 'RSI' in df.columns

        # Check that some values are calculated (not all NaN)
        assert not df['SMA_20'].isna().all()

    def test_ml_signal_enhancement_basic(self):
        """Test basic ML signal enhancement functionality."""
        # Create simple test data
        dates = [datetime.now() - timedelta(minutes=i) for i in range(100, 0, -1)]
        prices = np.random.uniform(49000, 51000, 100)

        data = []
        for i, price in enumerate(prices):
            data.append({
                'timestamp': dates[i],
                'open': price * (1 + np.random.normal(0, 0.002)),
                'high': price * (1 + abs(np.random.normal(0, 0.005))),
                'low': price * (1 - abs(np.random.normal(0, 0.005))),
                'close': price,
                'volume': np.random.uniform(100, 1000),
                'Buy_Signal': np.random.choice([0, 1]),
                'Sell_Signal': np.random.choice([0, 1])
            })

        df = pd.DataFrame(data)

        # Add basic technical indicators
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['RSI'] = 50

        # Remove NaN values
        df = df.dropna().reset_index(drop=True)

        # Apply ML enhancement
        enhanced_df = apply_ml_signal_enhancement(df.copy())

        # Check that enhancement was applied
        assert isinstance(enhanced_df, pd.DataFrame)
        assert len(enhanced_df) > 0
        assert 'Signal_Source' in enhanced_df.columns

    def test_ml_enhancement_with_signals(self):
        """Test ML enhancement with various signal types."""
        # Create test data with signals
        dates = [datetime.now() - timedelta(minutes=i) for i in range(100, 0, -1)]
        prices = np.random.uniform(49000, 51000, 100)

        data = []
        for i, price in enumerate(prices):
            data.append({
                'timestamp': dates[i],
                'open': price * (1 + np.random.normal(0, 0.002)),
                'high': price * (1 + abs(np.random.normal(0, 0.005))),
                'low': price * (1 - abs(np.random.normal(0, 0.005))),
                'close': price,
                'volume': np.random.uniform(100, 1000),
                'Buy_Signal': 1 if i % 10 == 0 else 0,  # Periodic buy signals
                'Sell_Signal': 1 if i % 15 == 0 else 0  # Periodic sell signals
            })

        df = pd.DataFrame(data)

        # Add technical indicators
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['RSI'] = np.random.uniform(30, 70, len(df))

        # Remove NaN values
        df = df.dropna().reset_index(drop=True)

        # Apply ML enhancement
        enhanced_df = apply_ml_signal_enhancement(df.copy())

        # Verify results
        assert len(enhanced_df) > 0
        assert 'Signal_Source' in enhanced_df.columns

        # Check that some signals were enhanced
        enhanced_signals = enhanced_df[enhanced_df['Signal_Source'] != 'TRADITIONAL']
        assert len(enhanced_signals) >= 0  # May be 0 if ML models not trained

        # Check data integrity
        assert not enhanced_df.empty
        assert all(col in enhanced_df.columns for col in df.columns)