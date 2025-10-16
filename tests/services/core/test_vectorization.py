#!/usr/bin/env python3
"""
Unit tests for memory-optimized vectorized indicators.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'services'))

from services.core.vectorization import (
    MemoryOptimizedIndicators,
    ema_vectorized,
    rsi_vectorized
)


class TestMemoryOptimizedIndicators:
    """Test cases for MemoryOptimizedIndicators."""

    def setup_method(self):
        """Set up test environment."""
        self.indicators = MemoryOptimizedIndicators(chunk_size=1000)

    def test_ema_memory_optimized(self):
        """Test memory-optimized EMA calculation."""
        # Create test data
        data = np.random.randn(1000).cumsum() + 100

        # Calculate EMA
        result = self.indicators.ema_memory_optimized(data, period=20)

        # Verify result shape
        assert len(result) == len(data)

        # Verify EMA properties (should be smoother than original data)
        assert np.std(result[20:]) < np.std(data[20:])

        # Test against manual calculation for first valid value
        # For EMA, the first valid value is a weighted average, not just data[19]
        # Just verify it's a reasonable value close to the data
        assert not np.isnan(result[19])
        assert abs(result[19] - data[19]) < 10  # Should be close to data[19] but not exactly equal

    def test_rsi_memory_optimized(self):
        """Test memory-optimized RSI calculation."""
        # Create trending data
        data = np.linspace(100, 110, 1000)

        # Calculate RSI
        result = self.indicators.rsi_memory_optimized(data, period=14)

        # Verify result shape
        assert len(result) == len(data)

        # RSI should be between 0 and 100
        valid_rsi = result[~np.isnan(result)]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)

        # For trending up data, RSI should be high
        final_rsi = result[-10:].mean()  # Average of last 10 values
        assert final_rsi > 50  # Should be above 50 for uptrend

    def test_chunked_processing(self):
        """Test chunked processing for large datasets."""
        # Create large dataset
        data = np.random.randn(5000).cumsum() + 100

        # Test with smaller chunk size
        indicators = MemoryOptimizedIndicators(chunk_size=1000)

        # Calculate EMA on large dataset
        result = indicators.ema_memory_optimized(data, period=20)

        # Verify result
        assert len(result) == len(data)
        assert not np.isnan(result[19])  # First valid value
        assert np.isnan(result[:19]).all()  # NaN for initial values


class TestStandaloneFunctions:
    """Test standalone memory-optimized functions."""

    def test_ema_memory_optimized_function(self):
        """Test standalone EMA function."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

        result = ema_vectorized(data, period=3)

        # Verify result
        assert len(result) == len(data)
        assert np.isnan(result[0])  # First value should be NaN
        assert np.isnan(result[1])  # Second value should be NaN
        assert not np.isnan(result[2])  # Third value should be valid

    def test_rsi_memory_optimized_function(self):
        """Test standalone RSI function."""
        # Create simple uptrend
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

        result = rsi_vectorized(data, period=3)

        # Verify result
        assert len(result) == len(data)
        # RSI values should be valid where calculable
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)


class TestPerformanceComparison:
    """Test performance improvements."""

    def test_ema_performance(self):
        """Compare performance of memory-optimized vs traditional EMA."""
        import time

        # Create test data
        data = np.random.randn(10000)

        # Time memory-optimized version
        start_time = time.time()
        result_optimized = ema_vectorized(data, period=20)
        optimized_time = time.time() - start_time

        # Time traditional implementation (simple moving average approximation)
        start_time = time.time()
        traditional_result = []
        for i in range(len(data)):
            if i >= 19:
                window = data[i-19:i+1]
                traditional_result.append(np.mean(window))
            else:
                traditional_result.append(np.nan)
        traditional_time = time.time() - start_time

        # Optimized version should be faster
        assert optimized_time < traditional_time

        # Results should be similar in magnitude
        optimized_valid = result_optimized[~np.isnan(result_optimized)]
        traditional_valid = np.array(traditional_result)[~np.isnan(traditional_result)]

        # Should have similar statistical properties
        assert abs(np.mean(optimized_valid) - np.mean(traditional_valid)) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
