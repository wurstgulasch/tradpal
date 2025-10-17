#!/usr/bin/env python3
"""
Prometheus Metrics Exporter for TradPal Services

Exports metrics for Circuit Breakers, Health Checks, and Service Performance
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
import time

logger = logging.getLogger(__name__)


class TradPalMetricsCollector:
    """
    Custom Prometheus collector for TradPal service metrics
    """

    def __init__(self):
        self.registry = CollectorRegistry()

        # Circuit Breaker Metrics
        self.cb_total_requests = Counter(
            'tradpal_circuit_breaker_requests_total',
            'Total requests through circuit breaker',
            ['service_name', 'circuit_state'],
            registry=self.registry
        )

        self.cb_failed_requests = Counter(
            'tradpal_circuit_breaker_requests_failed_total',
            'Total failed requests through circuit breaker',
            ['service_name'],
            registry=self.registry
        )

        self.cb_state = Gauge(
            'tradpal_circuit_breaker_state',
            'Current circuit breaker state (0=closed, 1=open, 2=half_open)',
            ['service_name'],
            registry=self.registry
        )

        self.cb_consecutive_failures = Gauge(
            'tradpal_circuit_breaker_consecutive_failures',
            'Consecutive failures for circuit breaker',
            ['service_name'],
            registry=self.registry
        )

        # Health Check Metrics
        self.hc_total_checks = Counter(
            'tradpal_health_checks_total',
            'Total health checks performed',
            ['service_name', 'check_type'],
            registry=self.registry
        )

        self.hc_check_duration = Histogram(
            'tradpal_health_check_duration_seconds',
            'Health check duration in seconds',
            ['service_name', 'check_type'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
            registry=self.registry
        )

        self.hc_status = Gauge(
            'tradpal_health_status',
            'Current health status (0=healthy, 1=degraded, 2=unhealthy)',
            ['service_name', 'check_type'],
            registry=self.registry
        )

        # Service Performance Metrics
        self.service_requests_total = Counter(
            'tradpal_service_requests_total',
            'Total service requests',
            ['service_name', 'method', 'status'],
            registry=self.registry
        )

        self.service_request_duration = Histogram(
            'tradpal_service_request_duration_seconds',
            'Service request duration in seconds',
            ['service_name', 'method'],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
            registry=self.registry
        )

        self.service_active_connections = Gauge(
            'tradpal_service_active_connections',
            'Number of active connections',
            ['service_name'],
            registry=self.registry
        )

        # System Metrics
        self.system_cpu_usage = Gauge(
            'tradpal_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )

        self.system_memory_usage = Gauge(
            'tradpal_system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )

        self.system_disk_usage = Gauge(
            'tradpal_system_disk_usage_percent',
            'System disk usage percentage',
            ['mount_point'],
            registry=self.registry
        )

    def update_circuit_breaker_metrics(self, service_name: str, metrics: Dict[str, Any]):
        """Update circuit breaker metrics"""
        try:
            # State mapping
            state_value = {'closed': 0, 'open': 1, 'half_open': 2}.get(metrics['state'], 0)
            self.cb_state.labels(service_name=service_name).set(state_value)

            # Request counters
            self.cb_total_requests.labels(
                service_name=service_name,
                circuit_state=metrics['state']
            ).inc(metrics['metrics']['total_requests'])

            self.cb_failed_requests.labels(service_name=service_name).inc(
                metrics['metrics']['failed_requests']
            )

            # Consecutive failures
            self.cb_consecutive_failures.labels(service_name=service_name).set(
                metrics['metrics']['consecutive_failures']
            )

        except Exception as e:
            logger.error(f"Error updating circuit breaker metrics for {service_name}: {e}")

    def update_health_check_metrics(self, service_name: str, result: Dict[str, Any]):
        """Update health check metrics"""
        try:
            check_type = result.get('name', 'unknown')

            # Status mapping
            status_value = {'healthy': 0, 'degraded': 1, 'unhealthy': 2}.get(result['status'], 2)
            self.hc_status.labels(service_name=service_name, check_type=check_type).set(status_value)

            # Check counter
            self.hc_total_checks.labels(service_name=service_name, check_type=check_type).inc()

            # Duration histogram
            duration_seconds = result.get('duration_ms', 0) / 1000.0
            self.hc_check_duration.labels(
                service_name=service_name,
                check_type=check_type
            ).observe(duration_seconds)

        except Exception as e:
            logger.error(f"Error updating health check metrics for {service_name}: {e}")

    def update_service_metrics(self, service_name: str, method: str, status: str, duration: float):
        """Update service request metrics"""
        try:
            self.service_requests_total.labels(
                service_name=service_name,
                method=method,
                status=status
            ).inc()

            self.service_request_duration.labels(
                service_name=service_name,
                method=method
            ).observe(duration)

        except Exception as e:
            logger.error(f"Error updating service metrics for {service_name}: {e}")

    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            disk_usage = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = usage.percent
                except:
                    pass  # Skip inaccessible partitions

            self.system_cpu_usage.set(cpu_percent)
            self.system_memory_usage.set(memory_percent)

            for mount_point, usage_percent in disk_usage.items():
                self.system_disk_usage.labels(mount_point=mount_point).set(usage_percent)

        except ImportError:
            logger.warning("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def update_from_circuit_breaker_registry(self, registry):
        """Update metrics from circuit breaker registry"""
        try:
            all_metrics = registry.get_all_metrics()
            for name, metrics in all_metrics.items():
                self.update_circuit_breaker_metrics(name, metrics)
        except Exception as e:
            logger.error(f"Error updating from circuit breaker registry: {e}")

    async def update_from_health_check_registry(self, registry):
        """Update metrics from health check registry"""
        try:
            results = await registry.run_all_checks()
            for name, result in results.items():
                self.update_health_check_metrics(name, result.__dict__)
        except Exception as e:
            logger.error(f"Error updating from health check registry: {e}")

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')


# Global metrics collector instance
metrics_collector = TradPalMetricsCollector()


class MetricsExporter:
    """
    HTTP server for exposing Prometheus metrics
    """

    def __init__(self, port: int = 8000):
        self.port = port
        self.app = None
        self._setup_app()

    def _setup_app(self):
        """Setup FastAPI app for metrics endpoint"""
        try:
            from fastapi import FastAPI
            from fastapi.responses import PlainTextResponse

            self.app = FastAPI(title="TradPal Metrics Exporter")

            @self.app.get("/metrics")
            async def metrics():
                """Prometheus metrics endpoint"""
                return PlainTextResponse(metrics_collector.get_metrics())

            @self.app.get("/health")
            async def health():
                """Health check endpoint"""
                return {"status": "healthy"}

        except ImportError:
            logger.warning("FastAPI not available, metrics exporter disabled")

    async def start(self):
        """Start the metrics exporter server"""
        if not self.app:
            logger.warning("Metrics exporter not available (FastAPI not installed)")
            return

        try:
            import uvicorn
            config = uvicorn.Config(
                self.app,
                host="0.0.0.0",
                port=self.port,
                log_level="warning"
            )
            server = uvicorn.Server(config)
            await server.serve()
        except ImportError:
            logger.warning("uvicorn not available, cannot start metrics server")


# Convenience functions for easy integration
def record_service_request(service_name: str, method: str, status: str, duration: float):
    """Record a service request for monitoring"""
    metrics_collector.update_service_metrics(service_name, method, status, duration)


def get_metrics_collector() -> TradPalMetricsCollector:
    """Get the global metrics collector instance"""
    return metrics_collector


async def start_metrics_exporter(port: int = 8000):
    """Start the metrics exporter server"""
    exporter = MetricsExporter(port)
    await exporter.start()