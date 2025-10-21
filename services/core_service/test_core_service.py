#!/usr/bin/env python3
"""
Test script for TradPal Core Service

This script tests the basic functionality of the core service:
- Service initialization
- Health checks
- Component integration
- Basic API endpoints
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.core_service import CoreService


async def test_core_service():
    """Test the core service functionality."""
    print("🧪 Testing TradPal Core Service...")

    # Initialize the service
    print("📦 Initializing core service...")
    service = CoreService()

    try:
        # Test initialization
        print("🔧 Testing service initialization...")
        await service.startup()
        print("✅ Service initialized successfully")

        # Test health check
        print("🏥 Testing health check...")
        health_status = await service.health_check()
        print(f"✅ Health status: {health_status}")

        # Test component health
        print("🔍 Testing component health...")
        components = ['api_gateway', 'event_system', 'security', 'calculations']
        for component in components:
            if hasattr(service, f"{component}_health"):
                health = await getattr(service, f"{component}_health")()
                print(f"✅ {component}: {health}")

        # Test metrics
        print("📊 Testing metrics collection...")
        metrics = await service.get_metrics()
        print(f"✅ Metrics collected: {len(metrics)} metrics")

        print("🎉 All core service tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        await service.shutdown()
        print("✅ Service shut down successfully")

    return True


async def test_api_gateway():
    """Test the API gateway component."""
    print("\n🌐 Testing API Gateway...")

    from services.core_service.api.gateway import APIGateway

    gateway = APIGateway()
    try:
        await gateway.initialize()
        print("✅ API Gateway initialized")

        # Test service registration
        await gateway.register_service("test_service", "http://localhost:8001")
        print("✅ Service registration successful")

        # Test service discovery
        services = await gateway.get_registered_services()
        print(f"✅ Registered services: {list(services.keys())}")

    except Exception as e:
        print(f"❌ API Gateway test failed: {e}")
        return False

    finally:
        await gateway.shutdown()

    return True


async def test_event_system():
    """Test the event system component."""
    print("\n📡 Testing Event System...")

    from services.core_service.events.system import EventSystemService

    event_system = EventSystemService()
    try:
        await event_system.initialize()
        print("✅ Event System initialized")

        # Test basic functionality
        health = await event_system.health_check()
        print(f"✅ Event System health: {health}")

    except Exception as e:
        print(f"❌ Event System test failed: {e}")
        return False

    finally:
        await event_system.shutdown()

    return True


async def test_security_service():
    """Test the security service component."""
    print("\n🔒 Testing Security Service...")

    from services.core_service.security.service_wrapper import SecurityService

    security = SecurityService()
    try:
        await security.initialize()
        print("✅ Security Service initialized")

        health = await security.health_check()
        print(f"✅ Security Service health: {health}")

    except Exception as e:
        print(f"❌ Security Service test failed: {e}")
        return False

    finally:
        await security.shutdown()

    return True


async def test_calculation_service():
    """Test the calculation service component."""
    print("\n🧮 Testing Calculation Service...")

    from services.core_service.calculations.service import CalculationService

    calculations = CalculationService()
    try:
        await calculations.initialize()
        print("✅ Calculation Service initialized")

        health = await calculations.health_check()
        print(f"✅ Calculation Service health: {health}")

    except Exception as e:
        print(f"❌ Calculation Service test failed: {e}")
        return False

    finally:
        await calculations.shutdown()

    return True


async def main():
    """Run all tests."""
    print("🚀 Starting TradPal Core Service Tests\n")

    tests = [
        ("Core Service", test_core_service),
        ("API Gateway", test_api_gateway),
        ("Event System", test_event_system),
        ("Security Service", test_security_service),
        ("Calculation Service", test_calculation_service),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Core service is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)