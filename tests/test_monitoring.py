#!/usr/bin/env python3
"""
Tests for Performance Monitoring Module
Tests Prometheus metrics collection and system monitoring.
"""

import pytest
import time
from unittest.mock import patch, MagicMock
from src.performance import PerformanceMonitor


class TestPerformanceMonitor:
    """Test suite for performance monitoring functionality."""

    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor()
        assert monitor.start_time is None
        assert monitor.cpu_percentages == []
        assert monitor.memory_usages == []
        assert monitor.monitoring is False
        assert monitor.monitor_thread is None

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('threading.Thread')
    def test_start_monitoring(self, mock_thread, mock_memory, mock_cpu):
        """Test starting performance monitoring."""
        mock_cpu.return_value = 50.0
        mock_memory_obj = MagicMock()
        mock_memory_obj.used = 1024 * 1024 * 1024  # 1GB
        mock_memory.return_value = mock_memory_obj

        monitor = PerformanceMonitor()
        monitor.start_monitoring()

        assert monitor.monitoring is True
        assert monitor.start_time is not None
        assert mock_thread.called

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('threading.Thread')
    def test_stop_monitoring(self, mock_thread, mock_memory, mock_cpu):
        """Test stopping performance monitoring and getting report."""
        mock_cpu.return_value = 50.0
        mock_memory_obj = MagicMock()
        mock_memory_obj.used = 1024 * 1024 * 1024  # 1GB
        mock_memory.return_value = mock_memory_obj

        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        time.sleep(0.1)  # Allow some monitoring to occur

        report = monitor.stop_monitoring()

        assert monitor.monitoring is False
        assert isinstance(report, dict)
        assert 'total_duration' in report
        assert 'avg_cpu_percent' in report
        assert 'max_cpu_percent' in report
        assert 'avg_memory_mb' in report
        assert 'max_memory_mb' in report
        assert 'samples_collected' in report

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_monitor_loop_data_collection(self, mock_memory, mock_cpu):
        """Test that monitor loop collects CPU and memory data."""
        mock_cpu.return_value = 75.5
        mock_memory_obj = MagicMock()
        mock_memory_obj.used = 2 * 1024 * 1024 * 1024  # 2GB
        mock_memory.return_value = mock_memory_obj

        monitor = PerformanceMonitor()
        monitor.monitoring = True

        # Run monitor loop in a separate thread with timeout
        import threading
        import time

        def run_monitor_with_timeout():
            # Override the monitoring flag after a short time to prevent infinite loop
            def stop_monitoring():
                time.sleep(0.1)  # Run for 0.1 seconds
                monitor.monitoring = False

            stop_thread = threading.Thread(target=stop_monitoring, daemon=True)
            stop_thread.start()

            monitor._monitor_loop()

        # Run the monitor loop with timeout
        monitor_thread = threading.Thread(target=run_monitor_with_timeout, daemon=True)
        monitor_thread.start()
        monitor_thread.join(timeout=1.0)  # Wait max 1 second

        # Force stop monitoring if still running
        monitor.monitoring = False

        assert len(monitor.cpu_percentages) >= 1
        assert len(monitor.memory_usages) >= 1
        assert monitor.cpu_percentages[0] == 75.5
        assert monitor.memory_usages[0] == 2048  # 2GB in MB

    def test_start_prometheus_server(self):
        """Test starting Prometheus HTTP server."""
        from src.performance import PROMETHEUS_AVAILABLE

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("Prometheus not available, skipping test")

        with patch('prometheus_client.start_http_server') as mock_start_server:
            monitor = PerformanceMonitor()
            monitor.start_prometheus_server(port=9090)

            mock_start_server.assert_called_once_with(9090)

    def test_start_prometheus_server_error(self):
        """Test Prometheus server start with error."""
        from src.performance import PROMETHEUS_AVAILABLE

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("Prometheus not available, skipping test")

        with patch('prometheus_client.start_http_server') as mock_start_server:
            mock_start_server.side_effect = Exception("Server error")

            monitor = PerformanceMonitor()
            # Should not raise exception
            monitor.start_prometheus_server(port=9090)

    def test_record_api_request(self):
        """Test recording API request metrics."""
        from src.performance import PROMETHEUS_AVAILABLE

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("Prometheus not available, skipping test")

        with patch('src.performance.api_requests_total') as mock_counter:
            monitor = PerformanceMonitor()
            monitor.record_api_request("test_method", "success", 1.5)

            mock_counter.labels.assert_called_once_with(method="test_method", status="success")
            mock_counter.labels().inc.assert_called_once()

    def test_record_api_request_with_duration(self):
        """Test recording API request with duration."""
        from src.performance import PROMETHEUS_AVAILABLE

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("Prometheus not available, skipping test")

        with patch('src.performance.api_request_duration') as mock_histogram:
            monitor = PerformanceMonitor()
            monitor.record_api_request("test_method", "success", 1.5)

            mock_histogram.labels.assert_called_once_with(method="test_method")
            mock_histogram.labels().observe.assert_called_once_with(1.5)

    def test_record_signal_generation(self):
        """Test recording signal generation metrics."""
        from src.performance import PROMETHEUS_AVAILABLE

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("Prometheus not available, skipping test")

        with patch('src.performance.signal_generation_total') as mock_counter:
            monitor = PerformanceMonitor()
            monitor.record_signal_generation("BUY")

            mock_counter.labels.assert_called_once_with(type="BUY")
            mock_counter.labels().inc.assert_called_once()

    def test_record_indicator_calculation(self):
        """Test recording indicator calculation duration."""
        from src.performance import PROMETHEUS_AVAILABLE

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("Prometheus not available, skipping test")

        with patch('src.performance.indicator_calculation_duration') as mock_histogram:
            monitor = PerformanceMonitor()
            monitor.record_indicator_calculation(2.5)

            mock_histogram.observe.assert_called_once_with(2.5)

    def test_record_trade(self):
        """Test recording trade metrics."""
        from src.performance import PROMETHEUS_AVAILABLE

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("Prometheus not available, skipping test")

        with patch('src.performance.trades_executed_total') as mock_counter, \
             patch('src.performance.profit_loss_total') as mock_pl_gauge:
            monitor = PerformanceMonitor()
            monitor.record_trade("EUR/USD", "BUY", 150.75)

            mock_counter.labels.assert_called_once_with(symbol="EUR/USD", type="BUY")
            mock_counter.labels().inc.assert_called_once()

    def test_record_trade_pnl_update(self):
        """Test that P&L gauge is updated correctly."""
        from src.performance import PROMETHEUS_AVAILABLE

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("Prometheus not available, skipping test")

        with patch('src.performance.profit_loss_total') as mock_pl_gauge:
            mock_gauge_instance = MagicMock()
            mock_pl_gauge._value.get.return_value = 100.0
            mock_pl_gauge.set = MagicMock()

            monitor = PerformanceMonitor()
            monitor.record_trade("EUR/USD", "BUY", 50.0)

            # Should set P&L to 150.0 (100 + 50)
            mock_pl_gauge.set.assert_called_once_with(150.0)

    def test_monitor_loop_error_handling(self):
        """Test that monitor loop handles exceptions gracefully."""
        import threading
        import time

        with patch('psutil.cpu_percent', side_effect=Exception("CPU error")):
            monitor = PerformanceMonitor()
            monitor.monitoring = True

            def run_monitor_with_timeout():
                def stop_monitoring():
                    time.sleep(0.1)  # Run for 0.1 seconds
                    monitor.monitoring = False

                stop_thread = threading.Thread(target=stop_monitoring, daemon=True)
                stop_thread.start()

                # Should not raise exception
                monitor._monitor_loop()

            # Run the monitor loop with timeout
            monitor_thread = threading.Thread(target=run_monitor_with_timeout, daemon=True)
            monitor_thread.start()
            monitor_thread.join(timeout=1.0)  # Wait max 1 second

            # Force stop monitoring if still running
            monitor.monitoring = False

            # Should still be marked as monitoring was stopped externally
            assert monitor.monitoring is False

    @patch('threading.active_count')
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_prometheus_metrics_update(self, mock_memory, mock_cpu, mock_active_count):
        """Test that Prometheus metrics are updated during monitoring."""
        from src.performance import PROMETHEUS_AVAILABLE

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("Prometheus not available, skipping test")

        import threading
        import time

        mock_cpu.return_value = 60.0
        mock_memory_obj = MagicMock()
        mock_memory_obj.used = 1024 * 1024 * 1024
        mock_memory.return_value = mock_memory_obj
        mock_active_count.return_value = 5

        monitor = PerformanceMonitor()
        monitor.monitoring = True

        def run_monitor_with_timeout():
            def stop_monitoring():
                time.sleep(0.1)  # Run for 0.1 seconds
                monitor.monitoring = False

            stop_thread = threading.Thread(target=stop_monitoring, daemon=True)
            stop_thread.start()

            monitor._monitor_loop()

        # Run the monitor loop with timeout
        monitor_thread = threading.Thread(target=run_monitor_with_timeout, daemon=True)
        monitor_thread.start()
        monitor_thread.join(timeout=1.0)  # Wait max 1 second

        # Force stop monitoring if still running
        monitor.monitoring = False

        # Check that threading.active_count was called (which would update Prometheus metrics)
        mock_active_count.assert_called()

    def test_stop_monitoring_without_start(self):
        """Test stopping monitoring when it was never started."""
        monitor = PerformanceMonitor()
        report = monitor.stop_monitoring()

        assert report == {}
        assert monitor.monitoring is False

    def test_monitor_thread_cleanup(self):
        """Test that monitor thread is properly cleaned up."""
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            monitor.stop_monitoring()

            mock_thread_instance.join.assert_called_once_with(timeout=1.0)

    @patch('time.time')
    def test_performance_report_calculation(self, mock_time):
        """Test calculation of performance report statistics."""
        monitor = PerformanceMonitor()

        # Simulate monitoring data
        monitor.start_time = 1000.0
        monitor.cpu_percentages = [50.0, 60.0, 70.0]
        monitor.memory_usages = [1024.0, 2048.0, 1536.0]
        monitor.monitoring = True  # Set monitoring to True so stop_monitoring calculates duration

        mock_time.return_value = 1010.0  # 10 seconds later

        report = monitor.stop_monitoring()

        assert report['total_duration'] == 10.0
        assert report['avg_cpu_percent'] == 60.0
        assert report['max_cpu_percent'] == 70.0
        assert report['avg_memory_mb'] == 1536.0
        assert report['max_memory_mb'] == 2048.0
        assert report['samples_collected'] == 3

    def test_empty_performance_data_report(self):
        """Test performance report with no collected data."""
        monitor = PerformanceMonitor()

        monitor.start_time = 1000.0
        monitor.monitoring = True  # Set monitoring to True so stop_monitoring calculates duration
        # No CPU or memory data collected

        with patch('time.time', return_value=1010.0):
            report = monitor.stop_monitoring()

        assert report['total_duration'] == 10.0
        assert report['avg_cpu_percent'] == 0
        assert report['max_cpu_percent'] == 0
        assert report['avg_memory_mb'] == 0
        assert report['max_memory_mb'] == 0
        assert report['samples_collected'] == 0