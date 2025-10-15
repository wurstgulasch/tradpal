"""
Tests for Kelly Criterion Position Sizing

Tests the Kelly Criterion implementation for optimal position sizing.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import Kelly components
try:
    from signal_generator import calculate_kelly_position_size
    from config.settings import KELLY_ENABLED, KELLY_FRACTION, KELLY_LOOKBACK_TRADES, KELLY_MIN_TRADES
    KELLY_AVAILABLE = True
except ImportError:
    KELLY_AVAILABLE = False
    pytest.skip("Kelly Criterion module not available", allow_module_level=True)


class TestKellyCriterion:
    """Test suite for Kelly Criterion position sizing."""

    def setup_method(self):
        """Skip test if Kelly is not available."""
        if not KELLY_AVAILABLE:
            pytest.skip("Kelly Criterion module not available")

    def test_calculate_kelly_position_size_disabled(self):
        """Test Kelly position sizing when disabled."""
        # Create test data
        dates = [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
        data = []
        for i, date in enumerate(dates):
            data.append({
                'timestamp': date,
                'close': 50000 + i * 100,
                'Buy_Signal': 1 if i % 3 == 0 else 0,
                'Sell_Signal': 1 if i % 4 == 0 else 0,
                'Trade_Result': 0.05 if i % 2 == 0 else -0.03,  # Mock trade results
                'Position_Size': 0.1
            })

        df = pd.DataFrame(data)

        # Mock Kelly as disabled
        with patch('config.settings.KELLY_ENABLED', False):
            sizes = calculate_kelly_position_size(df, capital=10000, risk_per_trade=0.02)

            # Should return traditional position sizing
            assert len(sizes) == len(df)
            assert all(size == 0.02 for size in sizes)  # Traditional fixed size

    def test_calculate_kelly_position_size_insufficient_data(self):
        """Test Kelly with insufficient historical data."""
        # Create test data with few trades
        dates = [datetime.now() - timedelta(days=i) for i in range(5, 0, -1)]
        data = []
        for i, date in enumerate(dates):
            data.append({
                'timestamp': date,
                'close': 50000 + i * 100,
                'Buy_Signal': 1,
                'Sell_Signal': 0,
                'Trade_Result': 0.05,
                'Position_Size': 0.1
            })

        df = pd.DataFrame(data)

        # Mock Kelly as enabled but with high min trades requirement
        with patch('config.settings.KELLY_ENABLED', True), \
             patch('config.settings.KELLY_MIN_TRADES', 10):
            sizes = calculate_kelly_position_size(df, capital=10000, risk_per_trade=0.02)

            # Should fall back to traditional sizing due to insufficient data
            assert len(sizes) == len(df)
            assert all(size == 0.02 for size in sizes)

    def test_calculate_kelly_position_size_with_data(self):
        """Test Kelly calculation with sufficient historical data."""
        # Create test data with good trading history
        dates = [datetime.now() - timedelta(days=i) for i in range(50, 0, -1)]
        data = []
        for i, date in enumerate(dates):
            # Create realistic trade results
            win_rate = 0.6  # 60% win rate
            avg_win = 0.08  # 8% average win
            avg_loss = -0.04  # 4% average loss

            is_win = np.random.random() < win_rate
            trade_result = avg_win if is_win else avg_loss

            data.append({
                'timestamp': date,
                'close': 50000 + i * 50,
                'Buy_Signal': 1 if i % 5 == 0 else 0,
                'Sell_Signal': 1 if i % 7 == 0 else 0,
                'Trade_Result': trade_result,
                'Position_Size': 0.1
            })

        df = pd.DataFrame(data)

        with patch('config.settings.KELLY_ENABLED', True), \
             patch('config.settings.KELLY_MIN_TRADES', 20), \
             patch('config.settings.KELLY_FRACTION', 0.5):
            sizes = calculate_kelly_position_size(df, capital=10000, risk_per_trade=0.02)

            # Should calculate Kelly-based sizes
            assert len(sizes) == len(df)
            assert all(isinstance(size, (int, float)) for size in sizes)
            assert all(0.01 <= size <= 0.25 for size in sizes)  # Within bounds

    def test_kelly_bounds_checking(self):
        """Test that Kelly sizes are properly bounded."""
        # Create data that would result in extreme Kelly values
        dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
        data = []

        # Create unrealistic winning scenario (100% win rate, high returns)
        for i, date in enumerate(dates):
            data.append({
                'timestamp': date,
                'close': 50000 + i * 10,
                'Buy_Signal': 1,
                'Sell_Signal': 0,
                'Trade_Result': 0.50,  # 50% return per trade (unrealistic)
                'Position_Size': 0.1
            })

        df = pd.DataFrame(data)

        with patch('config.settings.KELLY_ENABLED', True), \
             patch('config.settings.KELLY_MIN_TRADES', 10):
            sizes = calculate_kelly_position_size(df, capital=10000, risk_per_trade=0.02)

            # Should be clamped to maximum (25%)
            assert len(sizes) == len(df)
            assert all(size <= 0.25 for size in sizes)
            assert all(size >= 0.01 for size in sizes)  # Minimum bound

    def test_kelly_with_mixed_results(self):
        """Test Kelly calculation with realistic mixed trading results."""
        np.random.seed(42)  # For reproducible results

        # Create 100 trades with realistic performance
        dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
        data = []

        for i, date in enumerate(dates):
            # Realistic trading scenario
            win_rate = 0.55  # 55% win rate
            avg_win_ratio = 1.5  # Winners are 1.5x larger than losers

            is_win = np.random.random() < win_rate
            if is_win:
                trade_result = np.random.uniform(0.02, 0.15)  # 2-15% wins
            else:
                trade_result = -np.random.uniform(0.01, 0.08)  # 1-8% losses

            data.append({
                'timestamp': date,
                'close': 50000 + i * 25,
                'Buy_Signal': 1 if i % 4 == 0 else 0,
                'Sell_Signal': 1 if i % 6 == 0 else 0,
                'Trade_Result': trade_result,
                'Position_Size': 0.1
            })

        df = pd.DataFrame(data)

        with patch('config.settings.KELLY_ENABLED', True), \
             patch('config.settings.KELLY_MIN_TRADES', 20), \
             patch('config.settings.KELLY_FRACTION', 0.5):
            sizes = calculate_kelly_position_size(df, capital=10000, risk_per_trade=0.02)

            # Verify results
            assert len(sizes) == len(df)
            assert all(isinstance(size, (int, float)) for size in sizes)

            # Kelly should suggest reasonable position sizes
            avg_size = np.mean(sizes)
            assert 0.05 <= avg_size <= 0.20  # Reasonable range for this scenario

            # Kelly criterion typically uses the same size for all positions based on historical performance
            # The variation comes from different market conditions (RSI, ATR, ML confidence)
            # Since this test data doesn't include RSI/ATR, all sizes should be the same
            unique_sizes = set(round(size, 4) for size in sizes)  # Round to handle floating point precision
            assert len(unique_sizes) == 1  # All sizes should be the same for Kelly criterion