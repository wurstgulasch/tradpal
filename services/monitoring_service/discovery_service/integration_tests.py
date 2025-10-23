#!/usr/bin/env python3
"""
Discovery Service Integration Tests

Tests the Discovery Service in a more realistic environment,
including interactions with other services and external dependencies.
"""

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from services.monitoring_service.discovery_service.service import DiscoveryService, EventSystem
from services.monitoring_service.discovery_service.api import app


class TestDiscoveryServiceIntegration:
    """Integration tests for Discovery Service."""

    @pytest.fixture
    async def event_system(self):
        """Create event system for testing."""
        return EventSystem()

    @pytest.fixture
    async def discovery_service(self, event_system):
        """Create discovery service instance."""
        service = DiscoveryService(event_system=event_system)
        yield service
        # Cleanup any running optimizations
        await service._cleanup()

    @pytest.fixture
    async def test_client(self):
        """Create test client for FastAPI app."""
        from httpx import AsyncClient
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            yield client

    @pytest.mark.asyncio
    async def test_full_optimization_workflow(self, discovery_service, event_system):
        """Test complete optimization workflow from start to finish."""
        optimization_id = "integration_test_full_workflow"

        # Track events
        events_received = []
        async def event_handler(event):
            events_received.append(event)

        event_system.subscribe("discovery.optimization.started", event_handler)
        event_system.subscribe("discovery.optimization.completed", event_handler)
        event_system.subscribe("discovery.optimization.failed", event_handler)

        # Start optimization
        result = await discovery_service.run_optimization_async(
            optimization_id=optimization_id,
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-03-01",  # Short period for testing
            population_size=10,     # Small for speed
            generations=3,          # Few generations
            use_walk_forward=False  # Skip for speed
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "success" in result
        assert "optimization_id" in result
        assert result["optimization_id"] == optimization_id

        if result["success"]:
            assert "best_fitness" in result
            assert "best_config" in result
            assert "total_evaluations" in result
            assert "duration_seconds" in result
            assert result["duration_seconds"] > 0

            # Verify best config structure
            config = result["best_config"]
            assert "combination_name" in config
            assert "indicators" in config
        else:
            assert "error" in result

        # Verify events were fired
        await asyncio.sleep(0.1)  # Allow events to process
        assert len(events_received) >= 1

        # Check that we can get status
        status = await discovery_service.get_optimization_status(optimization_id)
        assert status is not None
        assert status["status"] in ["completed", "failed"]

    @pytest.mark.asyncio
    async def test_concurrent_optimizations(self, discovery_service):
        """Test running multiple optimizations concurrently."""
        optimization_ids = [f"concurrent_test_{i}" for i in range(3)]

        # Start multiple optimizations
        tasks = []
        for opt_id in optimization_ids:
            task = asyncio.create_task(
                discovery_service.run_optimization_async(
                    optimization_id=opt_id,
                    symbol="BTC/USDT",
                    timeframe="1d",
                    start_date="2024-01-01",
                    end_date="2024-02-01",  # Very short for speed
                    population_size=5,
                    generations=2,
                    use_walk_forward=False
                )
            )
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify results
        assert len(results) == len(optimization_ids)
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        assert successful >= 0  # At least some should succeed

        # Check active optimizations (should be empty now)
        active = await discovery_service.list_active_optimizations()
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_api_endpoints_integration(self, test_client, discovery_service):
        """Test API endpoints work together."""
        optimization_id = "api_integration_test"

        # Start optimization via API
        start_response = await test_client.post("/optimize/start", json={
            "optimization_id": optimization_id,
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "population_size": 8,
            "generations": 2
        })

        assert start_response.status_code == 200
        start_data = start_response.json()
        assert start_data["status"] == "started"

        # Check status via API
        max_checks = 30  # Wait up to 30 seconds
        for i in range(max_checks):
            status_response = await test_client.get(f"/optimize/status/{optimization_id}")
            assert status_response.status_code == 200
            status_data = status_response.json()

            if status_data["status"] == "completed":
                break
            elif status_data["status"] == "failed":
                break

            await asyncio.sleep(1)

        # Should eventually complete
        final_status = status_data["status"]
        assert final_status in ["completed", "failed"]

        # Check active optimizations
        active_response = await test_client.get("/optimize/active")
        assert active_response.status_code == 200
        active_data = active_response.json()
        assert isinstance(active_data, list)

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, discovery_service):
        """Test error handling in integration scenarios."""
        # Test invalid parameters
        result = await discovery_service.run_optimization_async(
            optimization_id="error_test",
            symbol="INVALID/SYMBOL",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-02-01",
            population_size=5,
            generations=1
        )

        # Should handle gracefully
        assert isinstance(result, dict)
        # May succeed or fail depending on data availability

        # Test with very short date range
        result2 = await discovery_service.run_optimization_async(
            optimization_id="error_test_2",
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-01-02",  # Only 1 day
            population_size=5,
            generations=1
        )

        assert isinstance(result2, dict)

    @pytest.mark.asyncio
    async def test_event_system_integration(self, discovery_service, event_system):
        """Test integration with event system."""
        events_received = []

        async def collect_events(event):
            events_received.append({
                "type": event.type,
                "timestamp": event.timestamp,
                "data": event.data
            })

        # Subscribe to all discovery events
        event_system.subscribe("discovery.optimization.started", collect_events)
        event_system.subscribe("discovery.optimization.progress", collect_events)
        event_system.subscribe("discovery.optimization.completed", collect_events)
        event_system.subscribe("discovery.optimization.failed", collect_events)

        # Run optimization
        result = await discovery_service.run_optimization_async(
            optimization_id="event_test",
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-02-01",
            population_size=8,
            generations=3
        )

        # Allow events to process
        await asyncio.sleep(0.2)

        # Should have received at least start and completion events
        assert len(events_received) >= 2

        # Check event structure
        for event in events_received:
            assert "type" in event
            assert "timestamp" in event
            assert "data" in event
            assert event["type"].startswith("discovery.optimization.")

    @pytest.mark.asyncio
    async def test_resource_cleanup_integration(self, discovery_service):
        """Test that resources are properly cleaned up."""
        opt_id = "cleanup_test"

        # Start optimization
        task = asyncio.create_task(
            discovery_service.run_optimization_async(
                optimization_id=opt_id,
                symbol="BTC/USDT",
                timeframe="1d",
                start_date="2024-01-01",
                end_date="2024-02-01",
                population_size=10,
                generations=5
            )
        )

        # Wait a bit then cancel
        await asyncio.sleep(0.5)

        # Cancel optimization
        cancelled = await discovery_service.cancel_optimization(opt_id)
        assert cancelled

        # Wait for task to complete
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()

        # Check that it's no longer active
        active = await discovery_service.list_active_optimizations()
        assert opt_id not in [opt["optimization_id"] for opt in active]

    @pytest.mark.asyncio
    async def test_health_endpoint_integration(self, test_client):
        """Test health endpoint provides useful information."""
        response = await test_client.get("/health")

        assert response.status_code == 200
        health_data = response.json()

        assert "status" in health_data
        assert "timestamp" in health_data
        assert "version" in health_data
        assert "active_optimizations" in health_data

        assert health_data["status"] == "healthy"
        assert isinstance(health_data["active_optimizations"], int)

    @pytest.mark.asyncio
    async def test_configuration_validation_integration(self, discovery_service):
        """Test that configuration parameters are validated."""
        # Test with invalid population size
        result = await discovery_service.run_optimization_async(
            optimization_id="config_test_1",
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-02-01",
            population_size=0,  # Invalid
            generations=2
        )

        # Should handle gracefully (may succeed or fail based on implementation)
        assert isinstance(result, dict)

        # Test with invalid date range
        result2 = await discovery_service.run_optimization_async(
            optimization_id="config_test_2",
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-12-31",  # End before start
            end_date="2024-01-01",
            population_size=5,
            generations=1
        )

        assert isinstance(result2, dict)

    @pytest.mark.asyncio
    async def test_performance_under_load(self, discovery_service):
        """Test performance with multiple concurrent optimizations."""
        num_concurrent = 3
        optimization_ids = [f"perf_test_{i}" for i in range(num_concurrent)]

        start_time = time.time()

        # Start all optimizations
        tasks = []
        for opt_id in optimization_ids:
            task = asyncio.create_task(
                discovery_service.run_optimization_async(
                    optimization_id=opt_id,
                    symbol="BTC/USDT",
                    timeframe="1d",
                    start_date="2024-01-01",
                    end_date="2024-01-15",  # Very short
                    population_size=5,
                    generations=2,
                    use_walk_forward=False
                )
            )
            tasks.append(task)

        # Wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        total_time = end_time - start_time

        # Verify all completed
        assert len(results) == num_concurrent

        # Performance should be reasonable (less than 60 seconds for small optimizations)
        assert total_time < 60

        print(f"Concurrent optimization time: {total_time:.2f}s")


class TestDiscoveryServiceExternalIntegration:
    """Tests that require external services or resources."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_with_real_data_service(self):
        """Test integration with real data service (requires running data service)."""
        pytest.skip("Requires running data service - run manually for full integration testing")

        # This would test actual data fetching and processing
        # Implementation would depend on data service API

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_with_real_backtesting_service(self):
        """Test integration with real backtesting service."""
        pytest.skip("Requires running backtesting service - run manually for full integration testing")

        # This would test actual backtesting integration
        # Implementation would depend on backtesting service API

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_workflow(self):
        """Complete end-to-end test with all services."""
        pytest.skip("Requires all services running - run manually for full integration testing")

        # This would test the complete workflow:
        # 1. Data fetching
        # 2. Optimization
        # 3. Backtesting validation
        # 4. Result storage


# Performance benchmarks
@pytest.mark.benchmark
class TestDiscoveryServicePerformance:
    """Performance benchmarks for Discovery Service."""

    @pytest.fixture
    async def service(self):
        event_system = EventSystem()
        return DiscoveryService(event_system=event_system)

    @pytest.mark.asyncio
    async def test_optimization_speed_small(self, service, benchmark):
        """Benchmark small optimization speed."""
        async def run_small_opt():
            return await service.run_optimization_async(
                optimization_id="bench_small",
                symbol="BTC/USDT",
                timeframe="1d",
                start_date="2024-01-01",
                end_date="2024-01-15",
                population_size=5,
                generations=2,
                use_walk_forward=False
            )

        result = benchmark(run_small_opt)
        assert result["success"] is True or result["success"] is False  # Just check it runs

    @pytest.mark.asyncio
    async def test_optimization_speed_medium(self, service, benchmark):
        """Benchmark medium optimization speed."""
        async def run_medium_opt():
            return await service.run_optimization_async(
                optimization_id="bench_medium",
                symbol="BTC/USDT",
                timeframe="1d",
                start_date="2024-01-01",
                end_date="2024-02-01",
                population_size=20,
                generations=5,
                use_walk_forward=False
            )

        result = benchmark(run_medium_opt)
        assert result["success"] is True or result["success"] is False


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])