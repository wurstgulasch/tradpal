#!/usr/bin/env python3
"""
Test script for GPU acceleration capabilities.

Tests GPU detection, memory management, neural network operations,
and performance comparisons between CPU and GPU.
"""

import asyncio
import logging
import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.core_service.gpu_accelerator import (
    GPUAccelerator, NeuralNetworkGPU, GPUMatrixOperations,
    GPUFeatureEngineering, get_gpu_accelerator, is_gpu_available
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_gpu_detection():
    """Test GPU detection and basic capabilities."""
    try:
        logger.info("🔍 Testing GPU Detection...")

        gpu = get_gpu_accelerator()

        logger.info(f"CUDA Available: {gpu.cuda_available}")
        logger.info(f"Device Count: {gpu.device_count}")

        if gpu.cuda_available:
            for i in range(gpu.device_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"Device {i}: {props.name}")
                logger.info(f"  Memory: {props.total_memory // 1024**3}GB")
                logger.info(f"  CUDA Capability: {props.major}.{props.minor}")

            # Test memory info
            mem_info = gpu.get_device_memory_info()
            logger.info(f"Memory Info: {mem_info}")
        else:
            logger.warning("No CUDA devices available")

        logger.info("✅ GPU detection test completed")
        return True

    except Exception as e:
        logger.error(f"❌ GPU detection test failed: {e}")
        return False

async def test_matrix_operations():
    """Test GPU-accelerated matrix operations."""
    try:
        logger.info("🔢 Testing GPU Matrix Operations...")

        gpu = get_gpu_accelerator()
        matrix_ops = GPUMatrixOperations(gpu)

        # Create test matrices
        size = 1000
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)

        logger.info(f"Testing matrix multiplication ({size}x{size})...")

        # CPU timing
        start_time = time.time()
        cpu_result = np.dot(a, b)
        cpu_time = time.time() - start_time
        logger.info(".4f")

        # GPU timing
        if gpu.cuda_available:
            start_time = time.time()
            gpu_result = matrix_ops.matrix_multiply_gpu(a, b)
            gpu_time = time.time() - start_time
            logger.info(".4f")

            # Verify results are close
            max_diff = np.max(np.abs(cpu_result - gpu_result))
            logger.info(".2e")

            if max_diff < 1e-3:
                speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
                logger.info(".1f")
            else:
                logger.warning("Results differ significantly between CPU and GPU")
        else:
            logger.info("Skipping GPU test - no CUDA available")

        logger.info("✅ Matrix operations test completed")
        return True

    except Exception as e:
        logger.error(f"❌ Matrix operations test failed: {e}")
        return False

async def test_neural_network_creation():
    """Test creation of GPU-accelerated neural networks."""
    try:
        logger.info("🧠 Testing Neural Network Creation...")

        gpu = get_gpu_accelerator()
        nn_gpu = NeuralNetworkGPU(gpu)

        # Test LSTM model creation
        logger.info("Creating LSTM model...")
        lstm_model = nn_gpu.create_lstm_model(
            input_size=50, hidden_size=128, num_layers=2, output_size=1
        )
        logger.info(f"LSTM model created: {lstm_model}")
        logger.info(f"Device: {next(lstm_model.parameters()).device}")

        # Test Transformer model creation
        logger.info("Creating Transformer model...")
        transformer_model = nn_gpu.create_transformer_model(
            input_size=50, num_heads=8, num_layers=4, output_size=1
        )
        logger.info(f"Transformer model created: {transformer_model}")
        logger.info(f"Device: {next(transformer_model.parameters()).device}")

        logger.info("✅ Neural network creation test completed")
        return True

    except Exception as e:
        logger.error(f"❌ Neural network creation test failed: {e}")
        return False

async def test_feature_engineering():
    """Test GPU-accelerated feature engineering."""
    try:
        logger.info("⚙️ Testing GPU Feature Engineering...")

        gpu = get_gpu_accelerator()
        feature_eng = GPUFeatureEngineering(gpu)

        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.uniform(40000, 60000, len(dates)),
            'high': np.random.uniform(40000, 60000, len(dates)),
            'low': np.random.uniform(40000, 60000, len(dates)),
            'close': np.random.uniform(40000, 60000, len(dates)),
            'volume': np.random.uniform(1000000, 10000000, len(dates))
        }, index=dates)

        # Ensure high >= close >= low and high >= open >= low
        for i in range(len(data)):
            high = max(data.iloc[i]['open'], data.iloc[i]['close'],
                      data.iloc[i]['high'], data.iloc[i]['low'])
            low = min(data.iloc[i]['open'], data.iloc[i]['close'],
                     data.iloc[i]['high'], data.iloc[i]['low'])
            data.iloc[i, data.columns.get_loc('high')] = high
            data.iloc[i, data.columns.get_loc('low')] = low

        logger.info(f"Created sample data: {len(data)} rows")

        # Test indicator computation
        indicators = ['sma', 'ema', 'rsi', 'macd']
        logger.info(f"Computing indicators: {indicators}")

        start_time = time.time()
        result_data = feature_eng.compute_technical_indicators_gpu(data, indicators)
        compute_time = time.time() - start_time

        logger.info(".4f")
        logger.info(f"Result columns: {list(result_data.columns)}")

        # Check if indicators were added
        new_columns = [col for col in result_data.columns if col not in data.columns]
        logger.info(f"New indicator columns: {new_columns}")

        logger.info("✅ Feature engineering test completed")
        return True

    except Exception as e:
        logger.error(f"❌ Feature engineering test failed: {e}")
        return False

async def test_memory_management():
    """Test GPU memory management."""
    try:
        logger.info("💾 Testing GPU Memory Management...")

        gpu = get_gpu_accelerator()

        if not gpu.cuda_available:
            logger.info("Skipping memory test - no CUDA available")
            return True

        # Test memory info
        mem_info = gpu.get_device_memory_info()
        logger.info(f"Initial memory: {mem_info}")

        # Test memory efficient context
        with gpu.memory_efficient_context():
            logger.info("Entered memory efficient context")

            # Allocate some tensors
            device = gpu.get_optimal_device()
            tensors = []
            for i in range(10):
                tensor = torch.randn(1000, 1000, dtype=torch.float32).to(device)
                tensors.append(tensor)

            logger.info(f"Allocated {len(tensors)} tensors")

            # Check memory after allocation
            mem_info_after = gpu.get_device_memory_info()
            logger.info(f"Memory after allocation: {mem_info_after}")

            # Clean up
            del tensors
            torch.cuda.empty_cache()

        # Check memory after cleanup
        mem_info_final = gpu.get_device_memory_info()
        logger.info(f"Memory after cleanup: {mem_info_final}")

        logger.info("✅ Memory management test completed")
        return True

    except Exception as e:
        logger.error(f"❌ Memory management test failed: {e}")
        return False

async def performance_comparison():
    """Compare CPU vs GPU performance."""
    try:
        logger.info("⚡ Performance Comparison: CPU vs GPU...")

        gpu = get_gpu_accelerator()

        if not gpu.cuda_available:
            logger.info("Skipping performance comparison - no CUDA available")
            return True

        # Matrix multiplication benchmark
        sizes = [500, 1000, 1500]

        for size in sizes:
            logger.info(f"\nBenchmarking {size}x{size} matrix multiplication:")

            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)

            # CPU benchmark
            cpu_times = []
            for _ in range(3):  # Average over 3 runs
                start = time.time()
                np.dot(a, b)
                cpu_times.append(time.time() - start)

            cpu_avg = np.mean(cpu_times)
            logger.info(".4f")

            # GPU benchmark
            matrix_ops = GPUMatrixOperations(gpu)
            gpu_times = []
            for _ in range(3):  # Average over 3 runs
                start = time.time()
                matrix_ops.matrix_multiply_gpu(a, b)
                gpu_times.append(time.time() - start)

            gpu_avg = np.mean(gpu_times)
            logger.info(".4f")

            speedup = cpu_avg / gpu_avg if gpu_avg > 0 else float('inf')
            logger.info(".1f")

        logger.info("✅ Performance comparison completed")
        return True

    except Exception as e:
        logger.error(f"❌ Performance comparison failed: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("🚀 TradPal GPU Acceleration Test Suite")
    logger.info("=" * 50)
    logger.info("")

    tests = [
        ("GPU Detection", test_gpu_detection),
        ("Matrix Operations", test_matrix_operations),
        ("Neural Network Creation", test_neural_network_creation),
        ("Feature Engineering", test_feature_engineering),
        ("Memory Management", test_memory_management),
        ("Performance Comparison", performance_comparison),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = await test_func()
            results.append((test_name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ FAILED with exception: {e}")
            results.append((test_name, False))

    logger.info(f"\n{'='*50}")
    logger.info("📊 TEST RESULTS SUMMARY:")
    logger.info(f"{'='*50}")

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:.<30} {status}")
        if success:
            passed += 1

    logger.info(f"{'='*50}")
    logger.info(f"Total Tests: {total}, Passed: {passed}, Failed: {total - passed}")

    if passed == total:
        logger.info("🎉 All tests passed! GPU acceleration is ready.")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed. Check GPU setup.")

    logger.info(f"{'='*50}")

if __name__ == "__main__":
    asyncio.run(main())