#!/usr/bin/env python3
"""
Demo script for Memory-Mapped Data Manager functionality.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory_mapped_data import MemoryMappedDataManager

def create_sample_data(n_rows=5000):
    """Create sample OHLCV dataset."""
    timestamps = pd.date_range(start='2023-01-01', periods=n_rows, freq='1min')
    np.random.seed(42)

    data = {
        'open': 50000 + np.random.normal(0, 100, n_rows),
        'high': 50100 + np.random.normal(0, 50, n_rows),
        'low': 49900 + np.random.normal(0, 50, n_rows),
        'close': 50000 + np.random.normal(0, 100, n_rows),
        'volume': np.random.randint(100, 1000, n_rows)
    }

    df = pd.DataFrame(data, index=timestamps)
    return df

def demo_basic_functionality():
    """Demonstrate basic memory-mapped DataFrame functionality."""
    print("🚀 Memory-Mapped Data Demo")
    print("=" * 40)

    # Create sample data
    df = create_sample_data()
    print(f"Created DataFrame: {df.shape}")
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Memory usage: {memory_usage:.1f}MB")

    # Initialize manager
    manager = MemoryMappedDataManager(max_memory_mb=100)

    try:
        # Create memory-mapped DataFrame
        print("\nCreating memory-mapped DataFrame...")
        name = manager.create_memory_mapped_dataframe(df, "test_data")
        print(f"✅ Created: {name}")

        # Load it back
        print("Loading memory-mapped DataFrame...")
        loaded_df = manager.load_mapped_dataframe(name)
        print(f"✅ Loaded: {loaded_df.shape}")

        # Check memory usage
        memory_info = manager.get_memory_usage()
        print(f"Memory usage: {memory_info['total_mb']:.1f}MB")

        # List mapped files
        mapped = manager.list_mapped_dataframes()
        print(f"Mapped DataFrames: {mapped}")

    finally:
        manager.cleanup()

    print("✅ Demo completed!")

if __name__ == "__main__":
    demo_basic_functionality()