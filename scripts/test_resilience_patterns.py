#!/usr/bin/env python3
"""
Test Script for Resilience Patterns

Tests Circuit Breaker and Health Check implementations
"""

import asyncio
import logging
import time
from services.core.circuit_breaker import AsyncCircuitBreaker, CircuitBreakerConfig, with_circuit_breaker
from services.core.health_checks import (
    HealthCheckRegistry, SystemHealthChecker, ServiceHealthChecker,
    HealthCheckConfig, HealthCheckRunner
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("\n=== Testing Circuit Breaker ===")

    config = CircuitBreakerConfig(
        name="test_service",
        failure_threshold=3,
        recovery_timeout=5.0,
        success_threshold=2,
        timeout=2.0
    )

    breaker = AsyncCircuitBreaker(config)

    # Mock failing function
    async def failing_function():
        raise Exception("Service temporarily unavailable")

    # Mock succeeding function
    async def succeeding_function():
        return "Success!"

    print("1. Testing failure handling...")
    for i in range(5):
        try:
            result = await breaker.call(failing_function)
            print(f"   Attempt {i+1}: Success - {result}")
        except Exception as e:
            print(f"   Attempt {i+1}: Failed - {e}")

    print(f"   Circuit state: {breaker.state.value}")
    print(f"   Consecutive failures: {breaker.metrics.consecutive_failures}")

    print("\n2. Testing recovery...")
    # Wait for recovery timeout
    await asyncio.sleep(6)

    # Test recovery attempts
    for i in range(3):
        try:
            result = await breaker.call(succeeding_function)
            print(f"   Recovery attempt {i+1}: Success - {result}")
        except Exception as e:
            print(f"   Recovery attempt {i+1}: Failed - {e}")

    print(f"   Final circuit state: {breaker.state.value}")
    print(f"   Consecutive successes: {breaker.metrics.consecutive_successes}")

    # Show metrics
    metrics = breaker.get_metrics()
    print(f"\n3. Circuit Breaker Metrics:")
    print(f"   Total requests: {metrics['metrics']['total_requests']}")
    print(f"   Success rate: {metrics['metrics']['success_rate']:.2%}")


async def test_health_checks():
    """Test health check functionality"""
    print("\n=== Testing Health Checks ===")

    registry = HealthCheckRegistry()

    # Add system health checker
    system_checker = SystemHealthChecker(
        config=HealthCheckConfig(
            name="system_monitor",
            check_interval=1.0,  # Check every second for demo
            failure_threshold=2
        )
    )
    await registry.register(system_checker)

    # Add service health checker (using a dummy URL)
    service_checker = ServiceHealthChecker(
        config=HealthCheckConfig(
            name="dummy_service",
            check_interval=1.0,
            timeout=1.0
        ),
        service_url="http://localhost:9999",  # Non-existent service
        service_name="dummy"
    )
    await registry.register(service_checker)

    print("1. Running initial health checks...")
    results = await registry.run_all_checks()

    for name, result in results.items():
        print(f"   {name}: {result.status.value} - {result.message}")
        print(f"      Duration: {result.duration_ms:.1f}ms")

    print("\n2. Running health checks again...")
    await asyncio.sleep(2)  # Wait for next check interval
    results = await registry.run_all_checks()

    for name, result in results.items():
        print(f"   {name}: {result.status.value} - {result.message}")

    print("\n3. Overall system health...")
    overall = await registry.get_overall_health()
    print(f"   Status: {overall['status']}")
    print(f"   Total checks: {overall['checks']['total']}")
    print(f"   Healthy: {overall['checks']['healthy']}")
    print(f"   Degraded: {overall['checks']['degraded']}")
    print(f"   Unhealthy: {overall['checks']['unhealthy']}")


async def test_convenience_function():
    """Test the convenience circuit breaker function"""
    print("\n=== Testing Convenience Function ===")

    config = CircuitBreakerConfig(
        name="convenience_test",
        failure_threshold=2,
        recovery_timeout=3.0
    )

    # Mock function that fails twice then succeeds
    call_count = 0
    async def mock_service():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise Exception(f"Call {call_count} failed")
        return f"Call {call_count} succeeded"

    print("Testing with_circuit_breaker convenience function...")

    for i in range(5):
        try:
            result = await with_circuit_breaker(config, mock_service)
            print(f"   Attempt {i+1}: {result}")
        except Exception as e:
            print(f"   Attempt {i+1}: Failed - {e}")

        await asyncio.sleep(0.1)  # Small delay between calls


async def test_background_health_runner():
    """Test background health check runner"""
    print("\n=== Testing Background Health Runner ===")

    registry = HealthCheckRegistry()

    # Add a simple system checker
    system_checker = SystemHealthChecker(
        config=HealthCheckConfig(
            name="background_system",
            check_interval=2.0  # Check every 2 seconds
        )
    )
    await registry.register(system_checker)

    # Start background runner
    runner = HealthCheckRunner(registry)
    await runner.start()

    print("Background health runner started. Monitoring for 10 seconds...")

    # Monitor health status for a short period
    for i in range(5):
        await asyncio.sleep(2)
        overall = await registry.get_overall_health()
        print(f"   Check {i+1}: Status = {overall['status']}, "
              f"Healthy = {overall['checks']['healthy']}")

    # Stop the runner
    await runner.stop()
    print("Background health runner stopped.")


async def main():
    """Run all resilience pattern tests"""
    print("🛡️  TradPal Resilience Patterns Test Suite")
    print("=" * 50)

    try:
        await test_circuit_breaker()
        await test_health_checks()
        await test_convenience_function()
        await test_background_health_runner()

        print("\n" + "=" * 50)
        print("✅ All resilience pattern tests completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• Circuit Breaker: Prevents cascading failures")
        print("• Health Checks: Monitors service and system health")
        print("• Automatic Recovery: Circuits close when services recover")
        print("• Background Monitoring: Continuous health assessment")
        print("• Metrics Collection: Detailed performance tracking")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())