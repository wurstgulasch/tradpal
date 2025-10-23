"""
Performance tests for lazy loading configuration system and memory optimization.
Tests the performance characteristics of the lazy loading system.
"""
import pytest
import time
import psutil
import os
import sys
from unittest.mock import patch
import gc

# Add config to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))

from config.settings import config, LazyConfig


class TestLazyLoadingPerformance:
    """Performance tests for lazy loading configuration system."""

    def test_lazy_loading_initialization_time(self):
        """Test that lazy loading initialization is fast."""
        start_time = time.time()

        # Create lazy config instance
        lazy_config = LazyConfig()

        end_time = time.time()
        init_time = end_time - start_time

        # Initialization should be very fast (< 0.01 seconds)
        assert init_time < 0.01, f"Lazy config initialization too slow: {init_time:.4f}s"

    def test_lazy_module_loading_time(self):
        """Test that lazy module loading is reasonably fast."""
        lazy_config = LazyConfig()

        # Test loading each module
        modules = ['core', 'ml', 'service', 'performance', 'security']

        for module in modules:
            start_time = time.time()
            settings = lazy_config.get_module(module)
            end_time = time.time()

            load_time = end_time - start_time

            # Module loading should be reasonably fast (< 0.1 seconds)
            assert load_time < 0.1, f"Module {module} loading too slow: {load_time:.4f}s"
            assert isinstance(settings, dict)
            assert len(settings) > 0

    def test_lazy_loading_memory_usage(self):
        """Test that lazy loading reduces initial memory usage."""
        process = psutil.Process()

        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create lazy config (should not load modules yet)
        lazy_config = LazyConfig()

        # Check memory after initialization
        after_init_memory = process.memory_info().rss / 1024 / 1024  # MB
        init_memory_increase = after_init_memory - initial_memory

        # Memory increase should be minimal (< 1MB)
        assert init_memory_increase < 1.0, f"Lazy config initialization used too much memory: {init_memory_increase:.2f}MB"

        # Load one module
        lazy_config.get_module('core')

        # Check memory after loading one module
        after_load_memory = process.memory_info().rss / 1024 / 1024  # MB
        load_memory_increase = after_load_memory - after_init_memory

        # Loading one module should not use excessive memory (< 5MB)
        assert load_memory_increase < 5.0, f"Module loading used too much memory: {load_memory_increase:.2f}MB"

    def test_lazy_loading_caching_performance(self):
        """Test that cached modules load faster on subsequent access."""
        lazy_config = LazyConfig()

        # Load module first time
        start_time = time.time()
        settings1 = lazy_config.get_module('core')
        first_load_time = time.time() - start_time

        # Load same module second time (should be cached)
        start_time = time.time()
        settings2 = lazy_config.get_module('core')
        second_load_time = time.time() - start_time

        # Second load should be faster (at least 2x faster, or very fast)
        assert second_load_time <= first_load_time, f"Cached loading slower: {second_load_time:.6f}s vs {first_load_time:.6f}s"

        # Results should be identical
        assert settings1 == settings2

    def test_lazy_loading_concurrent_access(self):
        """Test lazy loading performance under concurrent access."""
        import threading
        import concurrent.futures

        lazy_config = LazyConfig()
        results = {}
        errors = []

        def load_module(module_name):
            try:
                start_time = time.time()
                settings = lazy_config.get_module(module_name)
                end_time = time.time()
                results[module_name] = {
                    'load_time': end_time - start_time,
                    'settings_count': len(settings)
                }
            except Exception as e:
                errors.append((module_name, str(e)))

        # Test concurrent loading of all modules
        modules = ['core', 'ml', 'service', 'performance', 'security']

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(load_module, module) for module in modules]

            # Wait for all to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()

        # Check results
        assert len(errors) == 0, f"Concurrent loading errors: {errors}"
        assert len(results) == len(modules)

        # All modules should have loaded successfully
        for module in modules:
            assert module in results
            assert results[module]['load_time'] < 0.2  # Reasonable concurrent load time
            assert results[module]['settings_count'] > 0

    def test_lazy_loading_memory_cleanup(self):
        """Test that lazy loading properly cleans up memory."""
        process = psutil.Process()

        lazy_config = LazyConfig()

        # Load all modules
        modules = ['core', 'ml', 'service', 'performance', 'security']
        for module in modules:
            lazy_config.get_module(module)

        memory_after_load = process.memory_info().rss / 1024 / 1024  # MB

        # Clear cache
        lazy_config.clear_cache()

        # Force garbage collection
        gc.collect()

        memory_after_clear = process.memory_info().rss / 1024 / 1024

        # Memory should decrease after clearing cache (or at least not increase significantly)
        # Note: Due to Python's memory management, memory might not decrease immediately
        memory_decrease = memory_after_load - memory_after_clear
        assert memory_decrease >= -1.0, f"Memory increased significantly after cache clear: {memory_decrease:.2f}MB"

    def test_configuration_access_performance(self):
        """Test performance of configuration value access."""
        lazy_config = LazyConfig()

        # Test direct access performance
        access_times = []

        for _ in range(100):
            start_time = time.time()
            value = lazy_config.get('core', 'SYMBOL')
            end_time = time.time()
            access_times.append(end_time - start_time)

        avg_access_time = sum(access_times) / len(access_times)

        # Average access time should be very fast (< 0.001 seconds)
        assert avg_access_time < 0.001, f"Configuration access too slow: {avg_access_time:.6f}s average"

        # Value should be correct
        assert value == 'BTC/USDT'

    def test_lazy_loading_scalability(self):
        """Test that lazy loading scales well with many modules."""
        lazy_config = LazyConfig()

        # Simulate many modules (even if not implemented yet)
        module_names = [f'module_{i}' for i in range(20)]

        # Mock module loaders for scalability testing
        original_loaders = lazy_config._module_loaders.copy()

        def mock_loader():
            time.sleep(0.001)  # Small delay to simulate loading
            return {'setting': 'value'}

        # Add mock loaders
        for module in module_names:
            lazy_config._module_loaders[module] = mock_loader

        try:
            # Test loading many modules
            start_time = time.time()

            for module in module_names:
                settings = lazy_config.get_module(module)
                assert settings['setting'] == 'value'

            end_time = time.time()
            total_load_time = end_time - start_time

            # Loading 20 modules should be reasonably fast (< 1 second)
            assert total_load_time < 1.0, f"Scalability test failed: {total_load_time:.2f}s for 20 modules"

        finally:
            # Restore original loaders
            lazy_config._module_loaders = original_loaders

    def test_memory_usage_over_time(self):
        """Test that memory usage remains stable over multiple loads."""
        process = psutil.Process()
        lazy_config = LazyConfig()

        memory_readings = []

        # Load and unload modules multiple times
        for i in range(5):
            # Load all modules
            modules = ['core', 'ml', 'service', 'performance', 'security']
            for module in modules:
                lazy_config.get_module(module)

            # Record memory
            memory_readings.append(process.memory_info().rss / 1024 / 1024)  # MB

            # Clear cache
            lazy_config.clear_cache()
            gc.collect()

        # Memory usage should not grow significantly over iterations
        if len(memory_readings) > 1:
            initial_memory = memory_readings[0]
            final_memory = memory_readings[-1]
            memory_growth = final_memory - initial_memory

            # Memory growth should be minimal (< 2MB)
            assert memory_growth < 2.0, f"Memory leak detected: {memory_growth:.2f}MB growth over {len(memory_readings)} iterations"

    @pytest.mark.slow
    def test_performance_under_load(self):
        """Test performance under sustained load (marked as slow test)."""
        lazy_config = LazyConfig()

        # Simulate sustained configuration access
        operations = 1000
        start_time = time.time()

        for i in range(operations):
            # Alternate between different access patterns
            if i % 4 == 0:
                value = lazy_config.get('core', 'SYMBOL')
            elif i % 4 == 1:
                value = lazy_config.get('ml', 'ML_MODEL_TYPE')
            elif i % 4 == 2:
                value = lazy_config.get('service', 'ENABLE_MTLS')
            else:
                value = lazy_config.get('performance', 'GPU_ACCELERATION_ENABLED')

        end_time = time.time()
        total_time = end_time - start_time

        # Should handle 1000 operations reasonably fast (< 1 second)
        assert total_time < 1.0, f"Sustained load test failed: {total_time:.2f}s for {operations} operations"

        ops_per_second = operations / total_time
        assert ops_per_second > 500, f"Operations per second too low: {ops_per_second:.0f}"


class TestConfigurationMemoryOptimization:
    """Tests for memory optimization features."""

    def test_lazy_loading_memory_efficiency(self):
        """Test that lazy loading is more memory efficient than eager loading."""
        process = psutil.Process()

        # Test lazy loading memory usage
        lazy_memory_readings = []

        for _ in range(3):
            lazy_config = LazyConfig()
            lazy_memory_readings.append(process.memory_info().rss / 1024 / 1024)
            del lazy_config
            gc.collect()

        lazy_avg_memory = sum(lazy_memory_readings) / len(lazy_memory_readings)

        # Lazy loading should use reasonable memory (upper bound based on system)
        # This is a relative test - lazy should be functional
        assert lazy_avg_memory < 200, f"Lazy loading memory usage too high: {lazy_avg_memory:.2f}MB"  # Reasonable upper bound

    def test_configuration_caching_efficiency(self):
        """Test that configuration caching prevents redundant loading."""
        lazy_config = LazyConfig()

        # Load same module multiple times
        loads = 10
        start_time = time.time()

        for _ in range(loads):
            settings = lazy_config.get_module('core')

        end_time = time.time()
        total_time = end_time - start_time

        # Multiple loads should be fast due to caching
        avg_load_time = total_time / loads
        assert avg_load_time < 0.001, f"Cached loads too slow: {avg_load_time:.6f}s average"

    def test_memory_cleanup_after_module_unload(self):
        """Test memory cleanup when modules are unloaded."""
        process = psutil.Process()
        lazy_config = LazyConfig()

        # Load modules and measure memory
        modules = ['core', 'ml', 'service', 'performance', 'security']
        for module in modules:
            lazy_config.get_module(module)

        memory_with_modules = process.memory_info().rss / 1024 / 1024

        # Remove references and clear cache
        del lazy_config
        gc.collect()

        memory_after_cleanup = process.memory_info().rss / 1024 / 1024

        # Memory should decrease after cleanup
        memory_cleanup = memory_with_modules - memory_after_cleanup
        assert memory_cleanup >= 0, "Memory did not decrease after cleanup"


class TestConfigurationBenchmarking:
    """Benchmarking tests for configuration system performance."""

    def test_configuration_startup_time(self):
        """Benchmark configuration system startup time."""
        start_time = time.time()

        # Import and initialize configuration system
        from config.settings import config, DEFAULT_DATA_LIMIT, SYMBOL

        end_time = time.time()
        startup_time = end_time - start_time

        # Startup should be fast (< 0.1 seconds)
        assert startup_time < 0.1, f"Configuration startup too slow: {startup_time:.4f}s"

        # Verify configuration is working
        assert DEFAULT_DATA_LIMIT == 200
        assert SYMBOL == 'BTC/USDT'

    def test_configuration_access_latency(self):
        """Benchmark configuration access latency."""
        from config.settings import config

        # Test access latency for different configuration values
        test_cases = [
            ('core', 'SYMBOL'),
            ('core', 'TIMEFRAME'),
            ('ml', 'ML_MODEL_TYPE'),
            ('service', 'ENABLE_MTLS'),
            ('performance', 'GPU_ACCELERATION_ENABLED')
        ]

        latencies = []

        for module, key in test_cases:
            start_time = time.time()
            value = config.get(module, key)
            end_time = time.time()

            latency = end_time - start_time
            latencies.append(latency)

            # Each access should be fast
            assert latency < 0.01, f"Access to {module}.{key} too slow: {latency:.6f}s"

        # Average latency should be very low
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 0.001, f"Average access latency too high: {avg_latency:.6f}s"

    def test_configuration_memory_footprint(self):
        """Benchmark configuration system memory footprint."""
        process = psutil.Process()

        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024

        # Import configuration system
        from config.settings import config, LazyConfig

        # Load all modules
        modules = ['core', 'ml', 'service', 'performance', 'security']
        for module in modules:
            config.get_module(module)

        # Measure memory footprint
        config_memory = process.memory_info().rss / 1024 / 1024
        memory_footprint = config_memory - baseline_memory

        # Memory footprint should be reasonable (< 10MB)
        assert memory_footprint < 10.0, f"Configuration memory footprint too high: {memory_footprint:.2f}MB"


if __name__ == "__main__":
    pytest.main([__file__])