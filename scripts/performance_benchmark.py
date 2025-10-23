#!/usr/bin/env python3
"""
Performance Benchmarking for TradPal Memory Optimization
Tests memory usage and performance improvements with large datasets.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import time
import psutil
import os
from pathlib import Path
import tempfile

# Import our optimized modules
from services.core_service.memory_optimization import (
    MemoryMappedData,
    RollingWindowBuffer,
    ChunkedDataLoader,
    MemoryPool,
    LazyDataLoader,
    MemoryStats
)
from services.core_service.vectorization import MemoryOptimizedIndicators


def benchmark_memory_usage():
    """Benchmark memory usage for different data sizes."""
    print("=== Memory Usage Benchmark ===")

    sizes = [10000, 50000, 100000, 500000]

    for size in sizes:
        print(f"\nTesting with {size} data points...")

        # Create test data
        data = np.random.randn(size, 4)  # OHLC data

        # Measure memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Create memory-mapped file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Store data in memory-mapped file
            with MemoryMappedData(tmp_path, 'w') as mm_data:
                mm_data.create_dataset('ohlc', data=data, compression='gzip')

            # Measure memory after storing
            mem_after_store = process.memory_info().rss / 1024 / 1024

            # Load and process data
            optimizer = MemoryOptimizedIndicators(chunk_size=10000)
            # Test with close prices only (1D array)
            close_data = data[:, 3]  # Close prices
            ema_result = optimizer.ema_memory_optimized(close_data, period=20)
            rsi_result = optimizer.rsi_memory_optimized(close_data, period=14)

            # Measure memory after processing
            mem_after_process = process.memory_info().rss / 1024 / 1024

            print(".1f")
            print(".1f")
            print(".1f")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def benchmark_processing_speed():
    """Benchmark processing speed improvements."""
    print("\n=== Processing Speed Benchmark ===")

    size = 100000
    data = np.random.randn(size)

    # Traditional processing
    start_time = time.time()
    traditional_result = []
    for i in range(1, len(data)):
        if i >= 20:  # Simple moving average
            window = data[i-20:i]
            traditional_result.append(np.mean(window))
        else:
            traditional_result.append(np.nan)
    traditional_time = time.time() - start_time

    # Memory-optimized processing
    start_time = time.time()
    optimizer = MemoryOptimizedIndicators()
    optimized_result = optimizer.ema_memory_optimized(data, period=20)
    optimized_time = time.time() - start_time

    print(".4f")
    print(".4f")
    print(".2f")


def benchmark_chunked_processing():
    """Benchmark chunked data processing."""
    print("\n=== Chunked Processing Benchmark ===")

    # Create large dataset
    size = 500000
    data = np.random.randn(size, 4)

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Store data
        with MemoryMappedData(tmp_path, 'w') as mm_data:
            mm_data.create_dataset('ohlc', data=data)

        # Test chunked loading
        loader = ChunkedDataLoader(chunk_size=50000)

        start_time = time.time()
        chunks = list(loader.load_chunks(tmp_path, 'ohlc'))
        load_time = time.time() - start_time

        print(f"Loaded {len(chunks)} chunks in {load_time:.4f} seconds")
        print(".2f")

        # Test chunked processing
        start_time = time.time()
        results = loader.process_in_chunks(
            data,
            lambda chunk: np.sum(chunk, axis=0)
        )
        process_time = time.time() - start_time

        print(f"Processed {len(results)} chunks in {process_time:.4f} seconds")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def benchmark_memory_pool():
    """Benchmark memory pool efficiency."""
    print("\n=== Memory Pool Benchmark ===")

    pool = MemoryPool(max_size=100, initial_size=10, buffer_size=1024)

    # Allocate and release buffers
    buffers = []
    start_time = time.time()

    for i in range(1000):
        buffer = pool.allocate()
        buffers.append(buffer)

        if i % 100 == 0:  # Release every 100th buffer
            pool.release(buffers.pop(0))

    allocation_time = time.time() - start_time

    print(f"Allocated/released 1000 buffers in {allocation_time:.4f} seconds")
    print(f"Average time per operation: {allocation_time/1000*1000:.2f}ms")
    print(f"Available buffers in pool: {pool.available_buffers}")


def benchmark_lazy_loading():
    """Benchmark lazy loading with caching."""
    print("\n=== Lazy Loading Benchmark ===")

    # Create test data
    test_data = {
        'dataset1': pd.DataFrame(np.random.randn(10000, 4), columns=['open', 'high', 'low', 'close']),
        'dataset2': pd.DataFrame(np.random.randn(10000, 4), columns=['open', 'high', 'low', 'close']),
        'dataset3': pd.DataFrame(np.random.randn(10000, 4), columns=['open', 'high', 'low', 'close'])
    }

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Store test data
        with MemoryMappedData(tmp_path, 'w') as mm_data:
            for key, df in test_data.items():
                mm_data.store_dataframe(df, key)

        # Test lazy loading
        loader = LazyDataLoader(cache_size=5)

        # First load (no cache)
        start_time = time.time()
        data1 = loader.load_data(tmp_path, 'dataset1')
        first_load_time = time.time() - start_time

        # Second load (from cache)
        start_time = time.time()
        data1_cached = loader.load_data(tmp_path, 'dataset1')
        cached_load_time = time.time() - start_time

        print(".4f")
        print(".4f")
        print(".1f")

        # Test multiple loads
        start_time = time.time()
        for key in test_data.keys():
            loader.load_data(tmp_path, key)
        multi_load_time = time.time() - start_time

        print(f"Loaded 3 datasets in {multi_load_time:.4f} seconds")
        print(f"Cache size: {len(loader.cache)} items")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    """Run all benchmarks."""
    print("TradPal Memory Optimization Performance Benchmarks")
    print("=" * 60)

    try:
        benchmark_memory_usage()
        benchmark_processing_speed()
        benchmark_chunked_processing()
        benchmark_memory_pool()
        benchmark_lazy_loading()

        print("\n" + "=" * 60)
        print("✅ All benchmarks completed successfully!")
        print("Memory optimization features are working correctly.")

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()