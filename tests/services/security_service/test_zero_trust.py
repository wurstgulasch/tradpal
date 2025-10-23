#!/usr/bin/env python3
"""
Zero Trust Security Test Script

Tests the Zero Trust Security implementation including:
- mTLS authentication
- JWT token management
- Service authentication
- Security service functionality
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config.settings import (
    ENABLE_MTLS, SECURITY_SERVICE_URL,
    MTLS_CERT_PATH, MTLS_KEY_PATH, CA_CERT_PATH
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#!/usr/bin/env python3
"""
Zero Trust Security Test Script

Tests the Zero Trust Security implementation including:
- mTLS authentication
- JWT token management
- Service authentication
- Security service functionality
"""

import asyncio
import logging
import sys
import os
import pytest
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config.settings import (
    ENABLE_MTLS, SECURITY_SERVICE_URL,
    MTLS_CERT_PATH, MTLS_KEY_PATH, CA_CERT_PATH
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.skip(reason="Integration test requiring full service setup - skipped for unit test suite")
async def test_security_service():
    """Test the security service functionality"""
    logger.info("🧪 Testing Security Service...")

    try:
        from services.infrastructure_service.security_service.service import SecurityService, SecurityConfig

        # Create security config
        config = SecurityConfig(
            enable_mtls=ENABLE_MTLS,
            enable_jwt=True,
            enable_vault=False
        )

        # Initialize security service
        security_service = SecurityService(config)
        await security_service.start()

        logger.info("✅ Security service initialized")

        # Test JWT token generation
        token = await security_service.generate_jwt_token("test_service", ["read", "write"])
        logger.info(f"✅ JWT token generated: {token.token[:20]}...")

        # Test token validation
        validated = await security_service.validate_jwt_token(token.token)
        if validated:
            logger.info("✅ JWT token validation successful")
        else:
            logger.error("❌ JWT token validation failed")

        # Test mTLS credentials (if enabled)
        if config.enable_mtls:
            try:
                credentials = await security_service.issue_service_credentials("test_service")
                logger.info(f"✅ mTLS credentials issued for service: {credentials.service_name}")
            except Exception as e:
                logger.warning(f"⚠️  mTLS credentials test failed (expected if CA not initialized): {e}")

        # Test secrets management
        test_secret = {"api_key": "test_key", "api_secret": "test_secret"}
        success = await security_service.store_secret("test/path", test_secret)
        if success:
            logger.info("✅ Secret stored successfully")
        else:
            logger.error("❌ Secret storage failed")

        # Test secret retrieval
        retrieved = await security_service.retrieve_secret("test/path")
        if retrieved:
            logger.info("✅ Secret retrieved successfully")
        else:
            logger.error("❌ Secret retrieval failed")

        # Get health status
        health = await security_service.health_check()
        logger.info(f"✅ Security service health: {health['status']}")

        await security_service.stop()
        logger.info("✅ Security service test completed")

        return True

    except Exception as e:
        logger.error(f"❌ Security service test failed: {e}")
        return False


@pytest.mark.skip(reason="Integration test requiring full service setup - skipped for unit test suite")
async def test_service_clients():
    """Test service clients with Zero Trust authentication"""
    logger.info("🧪 Testing Service Clients with Zero Trust...")

    try:
        # Test Core Service Client
        from services.core_service.client import CoreServiceClient
        core_client = CoreServiceClient()

        # Test authentication
        auth_success = await core_client.authenticate()
        if auth_success:
            logger.info("✅ Core service client authenticated")
        else:
            logger.warning("⚠️  Core service client authentication failed (service may not be running)")

        # Test Data Service Client
        from services.data_service.data_service.service import DataService
        data_client = DataService()

        auth_success = await data_client.authenticate()
        if auth_success:
            logger.info("✅ Data service client authenticated")
        else:
            logger.warning("⚠️  Data service client authentication failed (service may not be running)")

        # Test Backtesting Service Client
        from services.trading_service.backtesting_service.client import BacktestingServiceClient
        backtesting_client = BacktestingServiceClient()

        await backtesting_client.initialize()
        auth_success = await backtesting_client.authenticate()
        if auth_success:
            logger.info("✅ Backtesting service client authenticated")
        else:
            logger.warning("⚠️  Backtesting service client authentication failed (service may not be running)")

        # Test Discovery Service Client
        from services.monitoring_service.discovery_service.client import DiscoveryServiceClient
        discovery_client = DiscoveryServiceClient()

        auth_success = await discovery_client.authenticate()
        if auth_success:
            logger.info("✅ Discovery service client authenticated")
        else:
            logger.warning("⚠️  Discovery service client authentication failed (service may not be running)")

        logger.info("✅ Service clients test completed")
        return True

    except Exception as e:
        logger.error(f"❌ Service clients test failed: {e}")
        return False


@pytest.mark.skip(reason="Integration test requiring full service setup - skipped for unit test suite")
async def test_main_orchestrator():
    """Test main orchestrator with Zero Trust services"""
    logger.info("🧪 Testing Main Orchestrator with Zero Trust...")

    try:
        from main import TradPalOrchestrator

        orchestrator = TradPalOrchestrator()
        success = await orchestrator.initialize_services()

        if success:
            logger.info("✅ Main orchestrator initialized with Zero Trust services")
            logger.info(f"Available services: {list(orchestrator.services.keys())}")

            # Check for security features
            for service_name, service_client in orchestrator.services.items():
                if hasattr(service_client, 'mtls_enabled'):
                    status = "enabled" if service_client.mtls_enabled else "disabled"
                    logger.info(f"🔒 {service_name}: mTLS {status}")

                if hasattr(service_client, 'jwt_token') and service_client.jwt_token:
                    logger.info(f"🔑 {service_name}: JWT authenticated")

        await orchestrator.shutdown()
        logger.info("✅ Main orchestrator test completed")

        return success

    except Exception as e:
        logger.error(f"❌ Main orchestrator test failed: {e}")
        return False

async def main():
    """Run all Zero Trust security tests"""
    logger.info("🚀 Starting Zero Trust Security Tests...")

    results = []

    # Test 1: Security Service
    logger.info("\n" + "="*50)
    logger.info("TEST 1: Security Service Functionality")
    logger.info("="*50)
    result1 = await test_security_service()
    results.append(("Security Service", result1))

    # Test 2: Service Clients
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Service Clients with Zero Trust")
    logger.info("="*50)
    result2 = await test_service_clients()
    results.append(("Service Clients", result2))

    # Test 3: Main Orchestrator
    logger.info("\n" + "="*50)
    logger.info("TEST 3: Main Orchestrator Integration")
    logger.info("="*50)
    result3 = await test_main_orchestrator()
    results.append(("Main Orchestrator", result3))

    # Summary
    logger.info("\n" + "="*50)
    logger.info("ZERO TRUST SECURITY TEST SUMMARY")
    logger.info("="*50)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All Zero Trust Security tests passed!")
        return 0
    else:
        logger.warning("⚠️  Some tests failed - check service availability and configuration")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)