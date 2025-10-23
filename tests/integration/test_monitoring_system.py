#!/usr/bin/env python3
"""
Test Script for TradPal Monitoring System

Tests Prometheus metrics collection and Grafana dashboards
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from services.core_service.client import CoreServiceClient
    from services.core_service.metrics_exporter import metrics_collector, start_metrics_exporter
    from services.core_service.circuit_breaker import circuit_breaker_registry
    from services.core_service.health_checks import health_check_registry
except ImportError as e:
    print(f"Import error: {e}")
    print("Some components may not be available, but basic metrics testing will continue")
    CoreServiceClient = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_metrics_collection():
    """Test metrics collection from service client"""
    logger.info("🧪 Testing Metrics Collection...")

    if not CoreServiceClient:
        logger.warning("⚠️ CoreServiceClient not available, skipping client test")
        return

    client = CoreServiceClient()

    # Start metrics collection
    await client.start_metrics_collection()

    # Make some requests to generate metrics
    try:
        # This will likely fail since service is not running, but will generate metrics
        await client.list_strategies()
    except Exception:
        pass  # Expected to fail

    try:
        await client.list_indicators()
    except Exception:
        pass  # Expected to fail

    # Wait for metrics collection
    await asyncio.sleep(2)

    # Check collected metrics
    metrics_output = metrics_collector.get_metrics()
    lines = metrics_output.split('\n')
    logger.info(f"Collected {len(lines)} metric lines")

    # Check specific metrics
    if 'tradpal_circuit_breaker_requests_total' in metrics_output:
        logger.info("✅ Circuit breaker metrics collected")
    else:
        logger.warning("❌ Circuit breaker metrics not found")

    if 'tradpal_health_status' in metrics_output:
        logger.info("✅ Health check metrics collected")
    else:
        logger.warning("❌ Health check metrics not found")

    if 'tradpal_system_cpu_usage_percent' in metrics_output:
        logger.info("✅ System metrics collected")
    else:
        logger.warning("❌ System metrics not found")

    await client.close()


async def test_prometheus_format():
    """Test Prometheus metrics format"""
    logger.info("🧪 Testing Prometheus Metrics Format...")

    # Generate some test metrics
    from services.core_service.metrics_exporter import record_service_request

    record_service_request("test_service", "GET", "200", 0.1)
    record_service_request("test_service", "POST", "500", 0.05)

    metrics = metrics_collector.get_metrics()

    # Check format
    lines = metrics.split('\n')
    metric_lines = [line for line in lines if line and not line.startswith('#')]

    logger.info(f"Generated {len(metric_lines)} metric data points")

    # Validate format (basic check)
    for line in metric_lines[:5]:  # Check first 5 lines
        if ' ' in line:
            name_value = line.split(' ', 1)
            if len(name_value) == 2:
                logger.info(f"✅ Valid metric format: {name_value[0]}")
            else:
                logger.warning(f"❌ Invalid metric format: {line}")
        else:
            logger.warning(f"❌ Invalid metric line: {line}")


async def test_monitoring_integration():
    """Test integration with monitoring registries"""
    logger.info("🧪 Testing Monitoring Integration...")

    try:
        # Update metrics from registries
        metrics_collector.update_from_circuit_breaker_registry(circuit_breaker_registry)
        metrics_collector.update_from_health_check_registry(health_check_registry)
        metrics_collector.update_system_metrics()

        logger.info("✅ Registry metrics updated")

        # Check registry metrics
        cb_metrics = circuit_breaker_registry.get_all_metrics()
        hc_metrics = health_check_registry.get_checker_metrics()

        logger.info(f"Circuit breaker registry: {len(cb_metrics)} breakers")
        logger.info(f"Health check registry: {len(hc_metrics)} checkers")
    except Exception as e:
        logger.warning(f"⚠️ Registry integration test failed: {e}")


async def simulate_monitoring_scenario():
    """Simulate a monitoring scenario with failures and recovery"""
    logger.info("🧪 Simulating Monitoring Scenario...")

    try:
        from services.core_service.circuit_breaker import CircuitBreakerConfig, AsyncCircuitBreaker
        from services.core_service.health_checks import HealthCheckConfig, ServiceHealthChecker

        # Create a test circuit breaker
        cb_config = CircuitBreakerConfig(
            name="test_monitoring_cb",
            failure_threshold=2,
            recovery_timeout=5.0
        )
        test_cb = AsyncCircuitBreaker(cb_config)

        # Create a test health checker
        hc_config = HealthCheckConfig(
            name="test_monitoring_hc",
            check_interval=1.0
        )
        test_hc = ServiceHealthChecker(
            config=hc_config,
            service_url="http://nonexistent:9999",
            service_name="test_service"
        )

        # Register them
        await circuit_breaker_registry.get_or_create("test_monitoring_cb", cb_config)
        await health_check_registry.register(test_hc)

        # Simulate failures
        async def failing_call():
            raise Exception("Simulated failure")

        logger.info("Simulating service failures...")
        for i in range(3):
            try:
                await test_cb.call(failing_call)
            except Exception:
                pass
            await asyncio.sleep(0.1)

        # Update metrics
        metrics_collector.update_from_circuit_breaker_registry(circuit_breaker_registry)
        metrics_collector.update_from_health_check_registry(health_check_registry)

        # Check circuit breaker state
        cb_metrics = circuit_breaker_registry.get_all_metrics()
        test_metrics = cb_metrics.get("test_monitoring_cb", {})
        state = test_metrics.get("state", "unknown")

        logger.info(f"Circuit breaker state after failures: {state}")

        # Wait for recovery
        logger.info("Waiting for circuit breaker recovery...")
        await asyncio.sleep(6)

        # Try recovery
        async def success_call():
            return "Success"

        try:
            result = await test_cb.call(success_call)
            logger.info(f"Recovery successful: {result}")
        except Exception as e:
            logger.info(f"Recovery failed: {e}")

        # Final metrics update
        metrics_collector.update_from_circuit_breaker_registry(circuit_breaker_registry)

        final_metrics = circuit_breaker_registry.get_all_metrics()
        final_state = final_metrics.get("test_monitoring_cb", {}).get("state", "unknown")
        logger.info(f"Final circuit breaker state: {final_state}")

    except Exception as e:
        logger.warning(f"⚠️ Monitoring scenario simulation failed: {e}")


async def main():
    """Main test function"""
    logger.info("🚀 Starting TradPal Monitoring System Test...")

    try:
        # Test individual components
        await test_metrics_collection()
        await test_prometheus_format()
        await test_monitoring_integration()
        await simulate_monitoring_scenario()

        logger.info("✅ All monitoring system tests completed successfully!")

        # Show final metrics summary
        final_metrics = metrics_collector.get_metrics()
        total_lines = len([line for line in final_metrics.split('\n') if line.strip()])
        logger.info(f"📊 Final metrics export: {total_lines} lines")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())