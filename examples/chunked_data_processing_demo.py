#!/usr/bin/env python3
"""
Chunked Data Processing Demo
Demonstrates efficient processing of large datasets in chunks.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from memory_mapped_data import ChunkedDataProcessor, TimeSeriesOptimizer

def create_large_dataset(n_rows: int = 100000) -> pd.DataFrame:
    """Create a large synthetic OHLCV dataset."""
    print(f"Creating large dataset with {n_rows} rows...")

    # Generate timestamps
    start_date = pd.Timestamp('2020-01-01')
    timestamps = pd.date_range(start_date, periods=n_rows, freq='1min')

    # Generate synthetic OHLCV data
    np.random.seed(42)
    base_price = 50000

    # Simulate price movements
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

def demo_chunked_processing():
    """Demonstrate chunked data processing."""
    print("🚀 Chunked Data Processing Demo")
    print("=" * 40)

    # Create large dataset
    df = create_large_dataset(50000)
    print(f"Created DataFrame: {df.shape}")
    print(".2f")

    # Initialize processors
    chunk_processor = ChunkedDataProcessor(chunk_size=5000)
    ts_optimizer = TimeSeriesOptimizer()

    # Optimize the data
    print("\nOptimizing time-series data...")
    optimized_df = ts_optimizer.optimize_ohlcv_data(df)
    print(f"Optimized DataFrame: {optimized_df.shape}")
    print(".2f")

    # Process in chunks - calculate moving averages
    def calculate_indicators(chunk):
        """Calculate technical indicators for a chunk."""
        result = chunk.copy()

        # Simple moving averages
        result['sma_20'] = result['close'].rolling(20).mean()
        result['sma_50'] = result['close'].rolling(50).mean()

        # RSI calculation
        delta = result['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        result['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        sma_20 = result['close'].rolling(20).mean()
        std_20 = result['close'].rolling(20).std()
        result['bb_upper'] = sma_20 + (std_20 * 2)
        result['bb_lower'] = sma_20 - (std_20 * 2)

        return result

    print("\nProcessing data in chunks...")
    start_time = time.time()

    results = chunk_processor.process_in_chunks(optimized_df, calculate_indicators)

    processing_time = time.time() - start_time
    print(".2f")

    # Combine results
    processed_df = pd.concat(results, axis=0)
    print(f"Processed DataFrame: {processed_df.shape}")

    # Save in chunks
    output_file = Path("output/chunked_data_demo.h5")
    output_file.parent.mkdir(exist_ok=True)

    print(f"\nSaving data in chunks to {output_file}...")
    chunk_processor.save_in_chunks(processed_df, str(output_file))

    # Load data back
    print("Loading data from chunks...")
    loaded_df = chunk_processor.load_in_chunks(str(output_file))
    print(f"Loaded DataFrame: {loaded_df.shape}")

    # Verify data integrity (compare with tolerance for floating point)
    print("\nVerifying data integrity...")
    try:
        # Drop rows with NaN values for comparison
        original_clean = processed_df.dropna()
        loaded_clean = loaded_df.dropna()

        # Compare a sample of rows
        sample_size = min(1000, len(original_clean))
        original_sample = original_clean.tail(sample_size)
        loaded_sample = loaded_clean.tail(sample_size)

        # Use pandas testing for approximate equality
        pd.testing.assert_frame_equal(original_sample, loaded_sample, atol=1e-10, rtol=1e-10)
        print("✅ Data integrity verified!")
    except AssertionError as e:
        print("❌ Data integrity check failed!")
        print(f"Details: {str(e)[:200]}...")

    # Rolling windows demo
    print("\nCreating rolling window calculations...")
    window_sizes = [10, 20, 50]
    rolling_results = ts_optimizer.create_rolling_windows(
        optimized_df, window_sizes, columns=['close', 'volume']
    )

    for window_name, window_df in rolling_results.items():
        print(f"  {window_name}: {window_df.shape}")

    print("\n✅ Chunked processing demo completed!")

def demo_memory_efficiency():
    """Demonstrate memory efficiency with large datasets."""
    print("\n🚀 Memory Efficiency Demo")
    print("=" * 30)

    # Test with different dataset sizes
    sizes = [10000, 50000, 100000]

    for size in sizes:
        print(f"\nTesting with {size} rows...")

        # Create dataset
        df = create_large_dataset(size)

        # Measure memory usage
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(".2f")

        # Initialize chunk processor with smaller chunks for larger datasets
        chunk_size = min(5000, size // 10)
        processor = ChunkedDataProcessor(chunk_size=chunk_size)

        # Process in chunks
        start_time = time.time()
        results = processor.process_in_chunks(df, lambda x: x['close'].rolling(20).mean())
        processing_time = time.time() - start_time

        print(".2f")
        print(f"  Number of chunks: {len(results)}")

if __name__ == "__main__":
    try:
        demo_chunked_processing()
        demo_memory_efficiency()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()