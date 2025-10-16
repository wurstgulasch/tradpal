#!/usr/bin/env python3
"""
Vectorization Performance Demo
Demonstrates performance improvements of vectorized indicator calculations.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from pathlib import Path

# Add src to path FIRST
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add project root for indicators.py

# Now import modules
from vectorization import (
    VectorizedIndicators,
    PerformanceBenchmark,
    ema_vectorized,
    rsi_vectorized,
    rsi_numba,
    bb_vectorized,
    atr_vectorized,
    atr_numba,
    macd_vectorized,
    stochastic_vectorized
)
from indicators import ema, rsi, bb, atr, macd, stochastic

def create_large_dataset(n_rows: int = 100000) -> pd.DataFrame:
    """Create a large synthetic OHLCV dataset for benchmarking."""
    print(f"Creating large dataset with {n_rows} rows...")

    # Generate timestamps
    start_date = pd.Timestamp('2020-01-01')
    timestamps = pd.date_range(start_date, periods=n_rows, freq='1min')

    # Generate synthetic OHLCV data
    np.random.seed(42)
    base_price = 50000

    # Simulate price movements with realistic volatility
    price_changes = np.random.normal(0, 0.001, n_rows).cumsum()
    close_prices = base_price * (1 + price_changes)

    # Generate OHLC from close prices with some noise
    high_noise = np.random.uniform(0, 0.005, n_rows)
    low_noise = np.random.uniform(0, 0.005, n_rows)

    opens = np.roll(close_prices, 1)
    opens[0] = base_price

    highs = np.maximum(opens, close_prices) * (1 + high_noise)
    lows = np.minimum(opens, close_prices) * (1 - low_noise)

    # Generate volume
    volumes = np.random.lognormal(10, 1, n_rows)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': volumes
    })

    df.set_index('timestamp', inplace=True)
    return df

def benchmark_ema(df: pd.DataFrame):
    """Benchmark EMA calculations."""
    print("\n🚀 EMA Benchmark")
    print("=" * 30)

    series = df['close']

    # Compare implementations
    comparison = PerformanceBenchmark.compare_implementations(
        ema, ema_vectorized, series, 20, runs=5
    )

    print(".4f")
    print(".4f")
    print(".1f")

    return comparison

def benchmark_rsi(df: pd.DataFrame):
    """Benchmark RSI calculations."""
    print("\n🚀 RSI Benchmark")
    print("=" * 30)

    series = df['close']

    # Compare pandas vs vectorized vs numba implementations
    print("Comparing RSI implementations...")

    pandas_stats = PerformanceBenchmark.benchmark_indicator(rsi, series, 14, runs=3)
    vectorized_stats = PerformanceBenchmark.benchmark_indicator(rsi_vectorized, series, 14, runs=3)
    numba_stats = PerformanceBenchmark.benchmark_indicator(rsi_numba, series, 14, runs=3)

    print(".4f")
    print(".4f")
    print(".4f")

    # Calculate speedups
    pandas_vs_vectorized = pandas_stats['mean'] / vectorized_stats['mean']
    pandas_vs_numba = pandas_stats['mean'] / numba_stats['mean']
    vectorized_vs_numba = vectorized_stats['mean'] / numba_stats['mean']

    print(".1f")
    print(".1f")
    print(".1f")

    return {
        'pandas': pandas_stats,
        'vectorized': vectorized_stats,
        'numba': numba_stats,
        'speedups': {
            'pandas_vs_vectorized': pandas_vs_vectorized,
            'pandas_vs_numba': pandas_vs_numba,
            'vectorized_vs_numba': vectorized_vs_numba
        }
    }

def benchmark_atr(df: pd.DataFrame):
    """Benchmark ATR calculations."""
    print("\n🚀 ATR Benchmark")
    print("=" * 30)

    # Compare implementations
    comparison = PerformanceBenchmark.compare_implementations(
        atr, atr_vectorized, df['high'], df['low'], df['close'], 14, runs=3
    )

    print(".4f")
    print(".4f")
    print(".1f")

    # Also test numba version
    numba_stats = PerformanceBenchmark.benchmark_indicator(
        atr_numba, df['high'], df['low'], df['close'], 14, runs=3
    )

    numba_speedup = comparison['original']['mean'] / numba_stats['mean']
    print(".4f")
    print(".1f")

    return comparison

def benchmark_bollinger_bands(df: pd.DataFrame):
    """Benchmark Bollinger Bands calculations."""
    print("\n🚀 Bollinger Bands Benchmark")
    print("=" * 30)

    series = df['close']

    # Compare implementations
    comparison = PerformanceBenchmark.compare_implementations(
        bb, bb_vectorized, series, 20, 2.0, runs=3
    )

    print(".4f")
    print(".4f")
    print(".1f")

    return comparison

def benchmark_macd(df: pd.DataFrame):
    """Benchmark MACD calculations."""
    print("\n🚀 MACD Benchmark")
    print("=" * 30)

    series = df['close']

    # Compare implementations
    comparison = PerformanceBenchmark.compare_implementations(
        macd, macd_vectorized, series, 12, 26, 9, runs=3
    )

    print(".4f")
    print(".4f")
    print(".1f")

    return comparison

def benchmark_stochastic(df: pd.DataFrame):
    """Benchmark Stochastic Oscillator calculations."""
    print("\n🚀 Stochastic Oscillator Benchmark")
    print("=" * 30)

    # Compare implementations
    comparison = PerformanceBenchmark.compare_implementations(
        stochastic, stochastic_vectorized, df['high'], df['low'], df['close'], 14, 3, runs=3
    )

    print(".4f")
    print(".4f")
    print(".1f")

    return comparison

def benchmark_rolling_statistics(df: pd.DataFrame):
    """Benchmark rolling statistics calculations."""
    print("\n🚀 Rolling Statistics Benchmark")
    print("=" * 30)

    series = df['close']
    window = 20

    # Pandas rolling
    def pandas_rolling_stats(data, window):
        s = pd.Series(data)
        return s.rolling(window).mean(), s.rolling(window).std(), s.rolling(window).min(), s.rolling(window).max()

    # Numba rolling
    def numba_rolling_stats(data, window):
        return VectorizedIndicators.rolling_statistics_numba(data, window)

    # Compare implementations
    comparison = PerformanceBenchmark.compare_implementations(
        pandas_rolling_stats, numba_rolling_stats, series.values, window, runs=3
    )

    print(".4f")
    print(".4f")
    print(".1f")

    return comparison

def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite."""
    print("🚀 Vectorization Performance Benchmark Suite")
    print("=" * 50)

    # Create test dataset
    df = create_large_dataset(50000)
    print(f"Dataset: {len(df)} rows")
    print(".2f")

    results = {}

    # Run all benchmarks
    results['ema'] = benchmark_ema(df)
    results['rsi'] = benchmark_rsi(df)
    results['atr'] = benchmark_atr(df)
    results['bb'] = benchmark_bollinger_bands(df)
    results['macd'] = benchmark_macd(df)
    results['stochastic'] = benchmark_stochastic(df)
    results['rolling'] = benchmark_rolling_statistics(df)

    # Summary
    print("\n🎯 PERFORMANCE SUMMARY")
    print("=" * 30)

    total_speedup = 0
    count = 0

    for indicator, result in results.items():
        if 'speedup' in result:
            speedup = result['speedup']
            print("15")
            total_speedup += speedup
            count += 1
        elif 'speedups' in result:
            # RSI has multiple speedups
            for comparison, speedup in result['speedups'].items():
                print("25")

    if count > 0:
        avg_speedup = total_speedup / count
        print(".1f")
        print(".1f")

    print("\n✅ Benchmark suite completed!")
    return results

def demo_accuracy_verification():
    """Verify that vectorized implementations produce accurate results."""
    print("\n🔍 Accuracy Verification")
    print("=" * 30)

    # Create smaller dataset for accuracy testing
    df = create_large_dataset(1000)

    # Test EMA
    ema_orig = ema(df['close'], 20)
    ema_vec = ema_vectorized(df['close'], 20)

    ema_diff = np.abs(ema_orig.values - ema_vec.values)
    ema_max_diff = np.nanmax(ema_diff)
    print(".2e")

    # Test RSI
    rsi_orig = rsi(df['close'], 14)
    rsi_vec = rsi_vectorized(df['close'], 14)
    rsi_numba_result = rsi_numba(df['close'], 14)

    rsi_diff_vec = np.abs(rsi_orig.values - rsi_vec.values)
    rsi_diff_numba = np.abs(rsi_orig.values - rsi_numba_result.values)

    print(".2e")
    print(".2e")

    # Test ATR
    atr_orig = atr(df['high'], df['low'], df['close'], 14)
    atr_vec = atr_vectorized(df['high'], df['low'], df['close'], 14)

    atr_diff = np.abs(atr_orig.values - atr_vec.values)
    atr_max_diff = np.nanmax(atr_diff)
    print(".2e")

    print("✅ Accuracy verification completed!")

if __name__ == "__main__":
    try:
        # Run accuracy verification first
        demo_accuracy_verification()

        # Run comprehensive benchmarks
        results = run_comprehensive_benchmark()

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()