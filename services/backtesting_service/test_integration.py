#!/usr/bin/env python3
"""
Integration tests for Backtesting Service.

This script runs comprehensive integration tests to verify that the
Backtesting Service works correctly in a real environment.
"""

import asyncio
import json
import time
import requests
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.backtesting_service import BacktestingService
# Event system - using mock implementation for now
EVENT_SYSTEM_AVAILABLE = False

class Event:
    def __init__(self, type: str, data: dict):
        self.type = type
        self.data = data

class EventSystem:
    def __init__(self):
        self.handlers = {}

    def subscribe(self, event_type: str, handler):
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    async def publish(self, event: Event):
        # Mock implementation - just call handlers synchronously
        if event.type in self.handlers:
            for handler in self.handlers[event.type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    print(f"Error in event handler: {e}")

from src.cache import Cache


class BacktestingServiceIntegrationTest:
    """Integration test suite for Backtesting Service."""

    def __init__(self, service_url: str = "http://localhost:8001"):
        """
        Initialize integration tests.

        Args:
            service_url: URL of the running backtesting service
        """
        self.service_url = service_url
        self.test_results = []

    def log_test_result(self, test_name: str, success: bool, message: str = "", duration: float = 0.0):
        """Log a test result."""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)

        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message} ({duration:.2f}s)")

    async def run_all_tests(self):
        """Run all integration tests."""
        print("🚀 Starting Backtesting Service Integration Tests")
        print("=" * 60)

        # Test 1: Health check
        await self.test_health_check()

        # Test 2: Single backtest
        await self.test_single_backtest()

        # Test 3: Multi-symbol backtest
        await self.test_multi_symbol_backtest()

        # Test 4: Multi-model backtest
        await self.test_multi_model_backtest()

        # Test 5: Walk-forward optimization
        await self.test_walk_forward_optimization()

        # Test 6: Backtest status monitoring
        await self.test_backtest_status_monitoring()

        # Test 7: Active backtests listing
        await self.test_active_backtests_listing()

        # Test 8: Cleanup functionality
        await self.test_cleanup_functionality()

        # Summary
        self.print_summary()

    async def test_health_check(self):
        """Test service health check."""
        start_time = time.time()

        try:
            response = requests.get(f"{self.service_url}/health", timeout=10)
            response.raise_for_status()

            data = response.json()
            success = data.get("status") == "healthy" and "backtesting-service" in data.get("service", "")

            message = f"Health check returned: {data}" if success else f"Unexpected health response: {data}"

        except Exception as e:
            success = False
            message = f"Health check failed: {str(e)}"

        duration = time.time() - start_time
        self.log_test_result("Health Check", success, message, duration)

    async def test_single_backtest(self):
        """Test single backtest execution."""
        start_time = time.time()

        try:
            payload = {
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-03-01",
                "strategy": "traditional",
                "initial_capital": 10000.0
            }

            response = requests.post(f"{self.service_url}/backtest", json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            backtest_id = data.get("backtest_id")

            if not backtest_id:
                raise ValueError("No backtest_id returned")

            # Wait for completion (poll status)
            await self._wait_for_backtest_completion(backtest_id, timeout=120)

            # Get final result
            result_response = requests.get(f"{self.service_url}/backtest/{backtest_id}", timeout=10)
            result_response.raise_for_status()
            result_data = result_response.json()

            success = result_data.get("status") == "completed" and result_data.get("result", {}).get("success")

            message = f"Backtest completed with {result_data.get('result', {}).get('trades_count', 0)} trades"

        except Exception as e:
            success = False
            message = f"Single backtest failed: {str(e)}"

        duration = time.time() - start_time
        self.log_test_result("Single Backtest", success, message, duration)

    async def test_multi_symbol_backtest(self):
        """Test multi-symbol backtest execution."""
        start_time = time.time()

        try:
            payload = {
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "initial_capital": 5000.0,
                "max_workers": 2
            }

            response = requests.post(f"{self.service_url}/backtest/multi-symbol", json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            backtest_id = data.get("backtest_id")

            if not backtest_id:
                raise ValueError("No backtest_id returned")

            # Wait for completion
            await self._wait_for_backtest_completion(backtest_id, timeout=180)

            # Get final result
            result_response = requests.get(f"{self.service_url}/backtest/{backtest_id}", timeout=10)
            result_response.raise_for_status()
            result_data = result_response.json()

            success = result_data.get("status") == "completed"
            if success:
                result = result_data.get("result", {})
                successful_backtests = len(result.get("successful_backtests", []))
                message = f"Multi-symbol backtest completed: {successful_backtests} successful backtests"
            else:
                message = "Multi-symbol backtest failed"

        except Exception as e:
            success = False
            message = f"Multi-symbol backtest failed: {str(e)}"

        duration = time.time() - start_time
        self.log_test_result("Multi-Symbol Backtest", success, message, duration)

    async def test_multi_model_backtest(self):
        """Test multi-model backtest execution."""
        start_time = time.time()

        try:
            payload = {
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "initial_capital": 10000.0,
                "models_to_test": ["traditional_ml", "lstm"],
                "max_workers": 2
            }

            response = requests.post(f"{self.service_url}/backtest/multi-model", json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            backtest_id = data.get("backtest_id")

            if not backtest_id:
                raise ValueError("No backtest_id returned")

            # Wait for completion (may take longer due to ML models)
            await self._wait_for_backtest_completion(backtest_id, timeout=300)

            # Get final result
            result_response = requests.get(f"{self.service_url}/backtest/{backtest_id}", timeout=10)
            result_response.raise_for_status()
            result_data = result_response.json()

            success = result_data.get("status") == "completed"
            if success:
                result = result_data.get("result", {})
                best_model = result.get("best_model")
                message = f"Multi-model backtest completed. Best model: {best_model}"
            else:
                message = "Multi-model backtest failed"

        except Exception as e:
            success = False
            message = f"Multi-model backtest failed: {str(e)}"

        duration = time.time() - start_time
        self.log_test_result("Multi-Model Backtest", success, message, duration)

    async def test_walk_forward_optimization(self):
        """Test walk-forward optimization."""
        start_time = time.time()

        try:
            payload = {
                "parameter_grid": {
                    "ema_short": [5, 9, 12],
                    "ema_long": [21, 26, 50]
                },
                "evaluation_metric": "sharpe_ratio",
                "symbol": "BTC/USDT",
                "timeframe": "1d"
            }

            response = requests.post(f"{self.service_url}/backtest/walk-forward", json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            backtest_id = data.get("backtest_id")

            if not backtest_id:
                raise ValueError("No backtest_id returned")

            # Wait for completion (optimization can take time)
            await self._wait_for_backtest_completion(backtest_id, timeout=240)

            # Get final result
            result_response = requests.get(f"{self.service_url}/backtest/{backtest_id}", timeout=10)
            result_response.raise_for_status()
            result_data = result_response.json()

            success = result_data.get("status") == "completed"
            if success:
                result = result_data.get("result", {})
                optimization = result.get("optimization_results", {})
                best_score = optimization.get("best_score")
                message = f"Walk-forward optimization completed. Best score: {best_score}"
            else:
                message = "Walk-forward optimization failed"

        except Exception as e:
            success = False
            message = f"Walk-forward optimization failed: {str(e)}"

        duration = time.time() - start_time
        self.log_test_result("Walk-Forward Optimization", success, message, duration)

    async def test_backtest_status_monitoring(self):
        """Test backtest status monitoring."""
        start_time = time.time()

        try:
            # Start a quick backtest
            payload = {
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2024-01-15",
                "strategy": "traditional",
                "initial_capital": 10000.0
            }

            response = requests.post(f"{self.service_url}/backtest", json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            backtest_id = data.get("backtest_id")

            # Check status immediately (should be running)
            status_response = requests.get(f"{self.service_url}/backtest/{backtest_id}", timeout=10)
            status_response.raise_for_status()
            status_data = status_response.json()

            initial_status = status_data.get("status")

            # Wait for completion
            await self._wait_for_backtest_completion(backtest_id, timeout=60)

            # Check final status
            final_status_response = requests.get(f"{self.service_url}/backtest/{backtest_id}", timeout=10)
            final_status_response.raise_for_status()
            final_status_data = final_status_response.json()

            final_status = final_status_data.get("status")

            success = initial_status == "running" and final_status == "completed"
            message = f"Status monitoring: {initial_status} -> {final_status}"

        except Exception as e:
            success = False
            message = f"Status monitoring test failed: {str(e)}"

        duration = time.time() - start_time
        self.log_test_result("Backtest Status Monitoring", success, message, duration)

    async def test_active_backtests_listing(self):
        """Test listing active backtests."""
        start_time = time.time()

        try:
            # Start a couple of backtests
            backtest_ids = []
            for i in range(2):
                payload = {
                    "symbol": "BTC/USDT",
                    "timeframe": "1d",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-10",
                    "strategy": "traditional",
                    "initial_capital": 10000.0
                }

                response = requests.post(f"{self.service_url}/backtest", json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                backtest_ids.append(data.get("backtest_id"))

            # Check active backtests
            active_response = requests.get(f"{self.service_url}/backtest/active", timeout=10)
            active_response.raise_for_status()
            active_data = active_response.json()

            active_count = active_data.get("count", 0)

            # Wait for completion
            for backtest_id in backtest_ids:
                await self._wait_for_backtest_completion(backtest_id, timeout=60)

            # Check active backtests again
            final_active_response = requests.get(f"{self.service_url}/backtest/active", timeout=10)
            final_active_response.raise_for_status()
            final_active_data = final_active_response.json()

            final_active_count = final_active_data.get("count", 0)

            success = active_count >= 2 and final_active_count < active_count
            message = f"Active backtests: {active_count} -> {final_active_count}"

        except Exception as e:
            success = False
            message = f"Active backtests listing failed: {str(e)}"

        duration = time.time() - start_time
        self.log_test_result("Active Backtests Listing", success, message, duration)

    async def test_cleanup_functionality(self):
        """Test cleanup functionality."""
        start_time = time.time()

        try:
            # Trigger cleanup
            cleanup_response = requests.delete(f"{self.service_url}/backtest/completed?max_age_hours=0", timeout=30)
            cleanup_response.raise_for_status()

            data = cleanup_response.json()
            message = data.get("message", "Cleanup completed")

            success = "Cleaned up" in message or "completed" in message.lower()
            message = f"Cleanup: {message}"

        except Exception as e:
            success = False
            message = f"Cleanup test failed: {str(e)}"

        duration = time.time() - start_time
        self.log_test_result("Cleanup Functionality", success, message, duration)

    async def _wait_for_backtest_completion(self, backtest_id: str, timeout: int = 60):
        """Wait for a backtest to complete."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.service_url}/backtest/{backtest_id}", timeout=10)
                response.raise_for_status()
                data = response.json()

                if data.get("status") == "completed":
                    return
                elif data.get("status") == "failed":
                    raise Exception(f"Backtest {backtest_id} failed: {data.get('error')}")

                await asyncio.sleep(2)  # Wait 2 seconds before checking again

            except requests.RequestException:
                await asyncio.sleep(2)  # Retry on network errors

        raise TimeoutError(f"Backtest {backtest_id} did not complete within {timeout} seconds")

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 INTEGRATION TEST SUMMARY")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(".1f")

        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['message']}")

        print("\n✅ PASSED TESTS:")
        for result in self.test_results:
            if result["success"]:
                print(f"  - {result['test']}: {result['message']}")

        # Save results to file
        results_file = f"output/backtesting_service_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_run": {
                    "timestamp": datetime.now().isoformat(),
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": failed_tests,
                    "success_rate": passed_tests / total_tests if total_tests > 0 else 0
                },
                "results": self.test_results
            }, f, indent=2)

        print(f"\n📄 Detailed results saved to: {results_file}")


async def run_smoke_test(service_url: str = "http://localhost:8001"):
    """Run a quick smoke test."""
    print("🚀 Running Backtesting Service Smoke Test")

    try:
        # Health check
        response = requests.get(f"{service_url}/health", timeout=10)
        response.raise_for_status()

        data = response.json()
        if data.get("status") == "healthy":
            print("✅ Smoke test passed: Service is healthy")
            return True
        else:
            print("❌ Smoke test failed: Service not healthy")
            return False

    except Exception as e:
        print(f"❌ Smoke test failed: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtesting Service Integration Tests")
    parser.add_argument("--url", default="http://localhost:8001", help="Service URL")
    parser.add_argument("--smoke-only", action="store_true", help="Run only smoke test")

    args = parser.parse_args()

    if args.smoke_only:
        success = asyncio.run(run_smoke_test(args.url))
        sys.exit(0 if success else 1)
    else:
        test_suite = BacktestingServiceIntegrationTest(args.url)
        asyncio.run(test_suite.run_all_tests())

        # Exit with appropriate code
        failed_tests = sum(1 for result in test_suite.test_results if not result["success"])
        sys.exit(0 if failed_tests == 0 else 1)