#!/usr/bin/env python3
"""
Test script for service clients

Tests all available service clients to ensure they work correctly.
"""

import asyncio
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import service clients
try:
    from services.data_service import DataServiceClient
    DATA_CLIENT_AVAILABLE = True
except ImportError:
    DATA_CLIENT_AVAILABLE = False

try:
    from services.backtesting_service.client import BacktestingServiceClient
    BACKTEST_CLIENT_AVAILABLE = True
except ImportError:
    BACKTEST_CLIENT_AVAILABLE = False

try:
    from services.notification_service.client import NotificationServiceClient
    NOTIFICATION_CLIENT_AVAILABLE = True
except ImportError:
    NOTIFICATION_CLIENT_AVAILABLE = False

try:
    from services.risk_service.client import RiskServiceClient
    RISK_CLIENT_AVAILABLE = True
except ImportError:
    RISK_CLIENT_AVAILABLE = False

try:
    from services.discovery_service.client import DiscoveryServiceClient
    DISCOVERY_CLIENT_AVAILABLE = True
except ImportError:
    DISCOVERY_CLIENT_AVAILABLE = False

try:
    from services.mlops_service.client import MLOpsServiceClient
    MLOPS_CLIENT_AVAILABLE = True
except ImportError:
    MLOPS_CLIENT_AVAILABLE = False

try:
    from services.security_service.client import SecurityServiceClient
    SECURITY_CLIENT_AVAILABLE = True
except ImportError:
    SECURITY_CLIENT_AVAILABLE = False
    SecurityServiceClient = None  # Define as None for type checking


async def test_service_client(name: str, client_class, test_func) -> Dict[str, Any]:
    """Test a service client"""
    logger.info(f"🧪 Testing {name} client...")

    try:
        client = client_class()
        result = await test_func(client)
        await client.close()

        logger.info(f"✅ {name} client test passed")
        return {"service": name, "status": "passed", "result": result}

    except Exception as e:
        logger.error(f"❌ {name} client test failed: {e}")
        return {"service": name, "status": "failed", "error": str(e)}


async def test_data_client(client: DataServiceClient) -> Dict[str, Any]:
    """Test data service client"""
    # Just test health check for now
    healthy = await client.health_check()
    return {"healthy": healthy}


async def test_backtest_client(client: BacktestingServiceClient) -> Dict[str, Any]:
    """Test backtesting service client"""
    # Just test health check for now
    healthy = await client.health_check()
    return {"healthy": healthy}


async def test_notification_client(client: NotificationServiceClient) -> Dict[str, Any]:
    """Test notification service client"""
    # Just test health check for now
    healthy = await client.health_check()
    return {"healthy": healthy}


async def test_risk_client(client: RiskServiceClient) -> Dict[str, Any]:
    """Test risk service client"""
    # Just test health check for now
    healthy = await client.health_check()
    return {"healthy": healthy}


async def test_discovery_client(client: DiscoveryServiceClient) -> Dict[str, Any]:
    """Test discovery service client"""
    # Just test health check for now
    healthy = await client.health_check()
    return {"healthy": healthy}


async def test_mlops_client(client: MLOpsServiceClient) -> Dict[str, Any]:
    """Test MLOps service client"""
    # Just test health check for now
    healthy = await client.health_check()
    return {"healthy": healthy}


async def test_security_client(client) -> Dict[str, Any]:
    """Test security service client"""
    # Test health check and authentication
    healthy = await client.health_check()
    if healthy:
        # Try to authenticate
        auth_success = await client.authenticate("test_client")
        return {"healthy": healthy, "authenticated": auth_success}
    return {"healthy": healthy}


async def main():
    """Run all service client tests"""
    logger.info("🚀 Starting service client tests...")

    test_results = []

    # Test each available client
    if DATA_CLIENT_AVAILABLE:
        result = await test_service_client("Data", DataServiceClient, test_data_client)
        test_results.append(result)

    if BACKTEST_CLIENT_AVAILABLE:
        result = await test_service_client("Backtesting", BacktestingServiceClient, test_backtest_client)
        test_results.append(result)

    if NOTIFICATION_CLIENT_AVAILABLE:
        result = await test_service_client("Notification", NotificationServiceClient, test_notification_client)
        test_results.append(result)

    if RISK_CLIENT_AVAILABLE:
        result = await test_service_client("Risk", RiskServiceClient, test_risk_client)
        test_results.append(result)

    if DISCOVERY_CLIENT_AVAILABLE:
        result = await test_service_client("Discovery", DiscoveryServiceClient, test_discovery_client)
        test_results.append(result)

    if MLOPS_CLIENT_AVAILABLE:
        result = await test_service_client("MLOps", MLOpsServiceClient, test_mlops_client)
        test_results.append(result)

    if SECURITY_CLIENT_AVAILABLE:
        result = await test_service_client("Security", SecurityServiceClient, test_security_client)
        test_results.append(result)

    # Summary
    passed = sum(1 for r in test_results if r["status"] == "passed")
    total = len(test_results)

    logger.info(f"📊 Test Summary: {passed}/{total} clients passed")

    # Print detailed results
    print("\n=== Service Client Test Results ===")
    for result in test_results:
        status = "✅" if result["status"] == "passed" else "❌"
        print(f"{status} {result['service']}: {result['status']}")
        if "error" in result:
            print(f"   Error: {result['error']}")
        if "result" in result:
            print(f"   Details: {result['result']}")

    print(f"\nOverall: {passed}/{total} services available and working")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)