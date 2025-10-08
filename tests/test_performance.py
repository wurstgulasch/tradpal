import pytest
import pandas as pd
import numpy as np
import time
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
import threading
import concurrent.futures

# Add project paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'integrations'))

from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators, ema, rsi, bb, atr
from src.signal_generator import generate_signals, calculate_risk_management
from src.output import save_signals_to_json, load_signals_from_json
from src.backtester import run_backtest, calculate_performance_metrics
from integrations.base import IntegrationManager
from integrations.telegram.bot import TelegramIntegration, TelegramConfig


class TestPerformanceBenchmarks:
    """Test performance benchmarks for critical functions."""

    def test_ema_performance_small_dataset(self):
        """Test EMA performance with small dataset."""
        data = pd.Series(np.random.randn(100))
        start_time = time.time()
        result = ema(data, 9)
        end_time = time.time()

        # Should complete quickly (< 0.1 seconds)
        assert end_time - start_time < 0.1
        assert len(result) == len(data)

    def test_ema_performance_large_dataset(self):
        """Test EMA performance with large dataset."""
        data = pd.Series(np.random.randn(10000))
        start_time = time.time()
        result = ema(data, 21)
        end_time = time.time()

        # Should complete reasonably fast (< 1 second)
        assert end_time - start_time < 1.0
        assert len(result) == len(data)

    def test_rsi_performance(self):
        """Test RSI performance."""
        data = pd.Series(np.random.randn(1000) + 100)
        start_time = time.time()
        result = rsi(data, 14)
        end_time = time.time()

        # Should complete quickly
        assert end_time - start_time < 0.5
        assert len(result) == len(data)

    def test_bb_performance(self):
        """Test Bollinger Bands performance."""
        data = pd.Series(np.random.randn(1000) + 100)
        start_time = time.time()
        upper, middle, lower = bb(data, 20, 2)
        end_time = time.time()

        # Should complete quickly
        assert end_time - start_time < 0.5
        assert len(upper) == len(data)
        assert len(middle) == len(data)
        assert len(lower) == len(data)

    def test_atr_performance(self):
        """Test ATR performance."""
        high = pd.Series(np.random.randn(1000) + 101)
        low = pd.Series(np.random.randn(1000) + 99)
        close = pd.Series(np.random.randn(1000) + 100)

        start_time = time.time()
        result = atr(high, low, close, 14)
        end_time = time.time()

        # Should complete quickly
        assert end_time - start_time < 0.5
        assert len(result) == len(high)


class TestIndicatorCalculationPerformance:
    """Test performance of indicator calculation pipeline."""

    def test_calculate_indicators_small_dataset(self):
        """Test indicator calculation with small dataset."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'close': np.random.randn(100) + 100.5,
            'volume': np.random.randint(1000, 10000, 100)
        })

        start_time = time.time()
        result = calculate_indicators(df)
        end_time = time.time()

        # Should complete quickly (< 0.5 seconds)
        assert end_time - start_time < 0.5
        assert len(result) == len(df)

    def test_calculate_indicators_medium_dataset(self):
        """Test indicator calculation with medium dataset."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1min'),
            'open': np.random.randn(1000) + 100,
            'high': np.random.randn(1000) + 101,
            'low': np.random.randn(1000) + 99,
            'close': np.random.randn(1000) + 100.5,
            'volume': np.random.randint(1000, 10000, 1000)
        })

        start_time = time.time()
        result = calculate_indicators(df)
        end_time = time.time()

        # Should complete reasonably fast (< 2 seconds)
        assert end_time - start_time < 2.0
        assert len(result) == len(df)

    def test_calculate_indicators_large_dataset(self):
        """Test indicator calculation with large dataset."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10000, freq='1min'),
            'open': np.random.randn(10000) + 100,
            'high': np.random.randn(10000) + 101,
            'low': np.random.randn(10000) + 99,
            'close': np.random.randn(10000) + 100.5,
            'volume': np.random.randint(1000, 10000, 10000)
        })

        start_time = time.time()
        result = calculate_indicators(df)
        end_time = time.time()

        # Should complete within reasonable time (< 10 seconds)
        assert end_time - start_time < 10.0
        assert len(result) == len(df)


class TestSignalGenerationPerformance:
    """Test performance of signal generation."""

    def test_generate_signals_performance(self):
        """Test signal generation performance."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1min'),
            'open': np.random.randn(1000) + 100,
            'high': np.random.randn(1000) + 101,
            'low': np.random.randn(1000) + 99,
            'close': np.random.randn(1000) + 100.5,
            'volume': np.random.randint(1000, 10000, 1000),
            'EMA9': np.random.randn(1000) + 100.5,
            'EMA21': np.random.randn(1000) + 100.3,
            'RSI': np.random.uniform(20, 80, 1000),
            'BB_upper': np.random.randn(1000) + 101.5,
            'BB_middle': np.random.randn(1000) + 100.5,
            'BB_lower': np.random.randn(1000) + 99.5,
            'ATR': np.random.uniform(0.1, 2.0, 1000)
        })

        start_time = time.time()
        result = generate_signals(df)
        end_time = time.time()

        # Should complete reasonably fast (< 4 seconds for signal generation)
        assert end_time - start_time < 4.0
        assert len(result) == len(df)

    def test_risk_management_performance(self):
        """Test risk management calculation performance."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1min'),
            'open': np.random.randn(1000) + 100,
            'high': np.random.randn(1000) + 101,
            'low': np.random.randn(1000) + 99,
            'close': np.random.randn(1000) + 100.5,
            'volume': np.random.randint(1000, 10000, 1000),
            'EMA9': np.random.randn(1000) + 100.5,
            'EMA21': np.random.randn(1000) + 100.3,
            'RSI': np.random.uniform(20, 80, 1000),
            'BB_upper': np.random.randn(1000) + 101.5,
            'BB_middle': np.random.randn(1000) + 100.5,
            'BB_lower': np.random.randn(1000) + 99.5,
            'ATR': np.random.uniform(0.1, 2.0, 1000),
            'Buy_Signal': np.random.choice([0, 1], 1000),
            'Sell_Signal': np.random.choice([0, 1], 1000)
        })

        start_time = time.time()
        result = calculate_risk_management(df)
        end_time = time.time()

        # Should complete quickly (< 0.5 seconds)
        assert end_time - start_time < 0.5
        assert len(result) == len(df)


class TestOutputPerformance:
    """Test performance of output operations."""

    def test_save_signals_performance_small(self):
        """Test saving small signal dataset."""
        signals = [{'signal': 'BUY', 'price': 100.0, 'timestamp': '2023-01-01'} for _ in range(10)]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            start_time = time.time()
            save_signals_to_json(signals, temp_file)
            end_time = time.time()

            # Should complete very quickly (< 0.1 seconds)
            assert end_time - start_time < 0.1

            # Verify file was written
            assert os.path.exists(temp_file)

        finally:
            os.unlink(temp_file)

    def test_save_signals_performance_large(self):
        """Test saving large signal dataset."""
        signals = [{'signal': 'BUY', 'price': 100.0 + i*0.1, 'timestamp': f'2023-01-{i+1:02d}'} for i in range(1000)]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            start_time = time.time()
            save_signals_to_json(signals, temp_file)
            end_time = time.time()

            # Should complete reasonably fast (< 1 second)
            assert end_time - start_time < 1.0

            # Verify file was written
            assert os.path.exists(temp_file)

        finally:
            os.unlink(temp_file)

    def test_load_signals_performance(self):
        """Test loading signals performance."""
        # Create test file with many signals
        signals = [{'signal': 'BUY', 'price': 100.0 + i*0.1, 'timestamp': f'2023-01-{i+1:02d}'} for i in range(1000)]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            import json
            json.dump(signals, f)

        try:
            start_time = time.time()
            result = load_signals_from_json(temp_file)
            end_time = time.time()

            # Should complete quickly (< 0.5 seconds)
            assert end_time - start_time < 0.5
            assert len(result) == 1000

        finally:
            os.unlink(temp_file)


class TestBacktesterPerformance:
    """Test performance of backtesting operations."""

    @patch('src.backtester.fetch_historical_data')
    def test_run_backtest_performance(self, mock_fetch):
        """Test backtest performance."""
        # Mock data
        mock_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1min'),
            'open': np.random.randn(1000) + 100,
            'high': np.random.randn(1000) + 101,
            'low': np.random.randn(1000) + 99,
            'close': np.random.randn(1000) + 100.5,
            'volume': np.random.randint(1000, 10000, 1000)
        })
        mock_fetch.return_value = mock_data

        start_time = time.time()
        result = run_backtest('EUR/USD', '1m', '2023-01-01', '2023-01-02')
        end_time = time.time()

        # Should complete within reasonable time (< 5 seconds)
        assert end_time - start_time < 5.0
        # Result may contain 'backtest_results' with metrics or error
        assert 'backtest_results' in result

    def test_calculate_performance_metrics_performance(self):
        """Test performance metrics calculation performance."""
        # Create many trades
        n_trades = 1000
        trades = pd.DataFrame({
            'entry_price': np.random.uniform(95, 105, n_trades),
            'exit_price': np.random.uniform(95, 105, n_trades),
            'position_size': np.random.uniform(100, 1000, n_trades),
            'direction': np.random.choice(['long', 'short'], n_trades),
            'entry_time': pd.date_range('2023-01-01', periods=n_trades, freq='1h'),
            'exit_time': pd.date_range('2023-01-01 01:00:00', periods=n_trades, freq='1h')
        })

        start_time = time.time()
        metrics = calculate_performance_metrics(trades, 10000)
        end_time = time.time()

        # Should complete quickly (< 0.5 seconds)
        assert end_time - start_time < 0.5
        assert 'total_trades' in metrics
        assert metrics['total_trades'] == n_trades


class TestConcurrencyPerformance:
    """Test performance under concurrent load."""

    def test_concurrent_indicator_calculation(self):
        """Test concurrent indicator calculations."""
        def calculate_single(df):
            return calculate_indicators(df)

        # Create multiple datasets
        datasets = []
        for i in range(5):
            df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=500, freq='1min'),
                'open': np.random.randn(500) + 100,
                'high': np.random.randn(500) + 101,
                'low': np.random.randn(500) + 99,
                'close': np.random.randn(500) + 100.5,
                'volume': np.random.randint(1000, 10000, 500)
            })
            datasets.append(df)

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(calculate_single, datasets))
        end_time = time.time()

        # Should complete within reasonable time (< 5 seconds)
        assert end_time - start_time < 5.0
        assert len(results) == 5
        for result in results:
            assert len(result) == 500

    def test_concurrent_file_operations(self):
        """Test concurrent file operations."""
        def save_and_load_signals(signals, filename):
            save_signals_to_json(signals, filename)
            return load_signals_from_json(filename)

        # Create multiple signal sets
        signal_sets = []
        filenames = []
        for i in range(3):
            signals = [{'signal': 'BUY', 'price': 100.0 + j, 'id': f'{i}_{j}'} for j in range(100)]
            signal_sets.append(signals)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                filenames.append(f.name)

        try:
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(save_and_load_signals, signal_sets, filenames))
            end_time = time.time()

            # Should complete quickly (< 1 second)
            assert end_time - start_time < 1.0
            assert len(results) == 3
            for result in results:
                assert len(result) == 100

        finally:
            for filename in filenames:
                try:
                    os.unlink(filename)
                except:
                    pass


class TestMemoryUsage:
    """Test memory usage under different loads."""

    def test_memory_usage_small_dataset(self):
        """Test memory usage with small dataset."""
        # This is a basic test - in real scenarios, you'd use memory profiling tools
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'close': np.random.randn(100) + 100.5,
            'volume': np.random.randint(1000, 10000, 100)
        })

        # Process and check that it doesn't crash
        result = calculate_indicators(df)
        result = generate_signals(result)
        result = calculate_risk_management(result)

        assert len(result) == 100

    def test_memory_usage_medium_dataset(self):
        """Test memory usage with medium dataset."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5000, freq='1min'),
            'open': np.random.randn(5000) + 100,
            'high': np.random.randn(5000) + 101,
            'low': np.random.randn(5000) + 99,
            'close': np.random.randn(5000) + 100.5,
            'volume': np.random.randint(1000, 10000, 5000)
        })

        # Process and check that it completes
        result = calculate_indicators(df)
        result = generate_signals(result)
        result = calculate_risk_management(result)

        assert len(result) == 5000


class TestIntegrationPerformance:
    """Test performance of integration components."""

    def test_integration_manager_bulk_operations(self):
        """Test integration manager with bulk operations."""
        manager = IntegrationManager()

        # Register multiple integrations
        for i in range(5):
            config = TelegramConfig(
                enabled=True,
                name=f"Test{i}",
                bot_token=f"token{i}",
                chat_id=f"chat{i}"
            )
            integration = TelegramIntegration(config)
            manager.register_integration(f"test{i}", integration)

        signal = {'signal': 'BUY', 'price': 100.0, 'timestamp': '2023-01-01'}

        start_time = time.time()
        results = manager.send_signal_to_all(signal)
        end_time = time.time()

        # Should complete reasonably fast even with multiple integrations (< 2 seconds)
        assert end_time - start_time < 2.0
        assert len(results) == 5


class TestSystemLoadTests:
    """Test system under various load conditions."""

    def test_full_pipeline_load_test(self):
        """Test full pipeline under load."""
        # Create large dataset
        n_rows = 5000
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n_rows, freq='1min'),
            'open': np.random.randn(n_rows) + 100,
            'high': np.random.randn(n_rows) + 101,
            'low': np.random.randn(n_rows) + 99,
            'close': np.random.randn(n_rows) + 100.5,
            'volume': np.random.randint(1000, 10000, n_rows)
        })

        start_time = time.time()

        # Run full pipeline
        df_indicators = calculate_indicators(df)
        df_signals = generate_signals(df_indicators)
        df_risk = calculate_risk_management(df_signals)

        # Save results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            signals = [{'test': f'signal_{i}'} for i in range(min(100, len(df_risk)))]
            save_signals_to_json(signals, temp_file)

            end_time = time.time()

            # Should complete within reasonable time (< 15 seconds)
            assert end_time - start_time < 15.0
            assert len(df_risk) == n_rows

        finally:
            try:
                os.unlink(temp_file)
            except:
                pass

    def test_backtest_load_test(self):
        """Test backtesting under load."""
        @patch('src.backtester.fetch_historical_data')
        def run_backtest_load(mock_fetch):
            # Mock large dataset
            mock_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=5000, freq='1min'),
                'open': np.random.randn(5000) + 100,
                'high': np.random.randn(5000) + 101,
                'low': np.random.randn(5000) + 99,
                'close': np.random.randn(5000) + 100.5,
                'volume': np.random.randint(1000, 10000, 5000)
            })
            mock_fetch.return_value = mock_data

            start_time = time.time()
            result = run_backtest('EUR/USD', '1m', '2023-01-01', '2023-01-02')
            end_time = time.time()

            # Should complete within reasonable time (< 10 seconds)
            assert end_time - start_time < 10.0
            assert 'backtest_results' in result

        run_backtest_load()


if __name__ == "__main__":
    pytest.main([__file__])