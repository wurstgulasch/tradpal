#!/usr/bin/env python3
"""
Test script for TradPal Data Service
Tests the unified data service functionality
"""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.data_service.main import DataServiceOrchestrator
from services.data_service.data_sources.service import DataService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_data_service():
    """Test the data service functionality"""
    logger.info("🧪 Testing Data Service...")

    try:
        # Test orchestrator initialization
        orchestrator = DataServiceOrchestrator()
        await orchestrator.initialize()
        logger.info("✓ Orchestrator initialized successfully")

        # Test health check
        health = await orchestrator.health_check()
        logger.info(f"✓ Health check: {health['status']}")

        # Test service info
        info = await orchestrator.get_service_info()
        logger.info(f"✓ Service info: {info['name']} v{info['version']}")

        # Test data service directly
        data_service = orchestrator.data_service

        # Test available sources
        sources = data_service.get_available_sources()
        logger.info(f"✓ Available sources: {sources}")

        # Test data fetching
        test_symbol = "BTC/USDT"
        data = await data_service.fetch_data(test_symbol, "1h", 10)
        logger.info(f"✓ Fetched {len(data)} data points for {test_symbol}")
        logger.info(f"  Sample data: {data.head(2).to_dict('records')}")

        # Test data quality validation
        quality = await data_service.validate_data_quality(data)
        logger.info(f"✓ Data quality: {quality}")

        # Test shutdown
        await orchestrator.shutdown()
        logger.info("✓ Orchestrator shut down successfully")

        logger.info("🎉 All Data Service tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Data Service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_api_endpoints():
    """Test API endpoints"""
    logger.info("🧪 Testing API endpoints...")

    try:
        from services.data_service.api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        logger.info("✓ Root endpoint works")

        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
        logger.info("✓ Health endpoint works")

        # Test sources endpoint
        response = client.get("/sources")
        assert response.status_code == 200
        sources_data = response.json()
        assert "sources" in sources_data
        logger.info("✓ Sources endpoint works")

        # Test data endpoint
        response = client.get("/data/BTC-USDT?limit=5")
        assert response.status_code == 200
        data_response = response.json()
        assert "symbol" in data_response
        assert "data" in data_response
        logger.info("✓ Data endpoint works")

        # Test data info endpoint
        response = client.get("/data/BTC-USDT/info")
        assert response.status_code == 200
        info_response = response.json()
        assert "symbol" in info_response
        logger.info("✓ Data info endpoint works")

        logger.info("🎉 All API endpoint tests passed!")
        return True

    except ImportError:
        logger.warning("⚠️ FastAPI test client not available, skipping API tests")
        return True
    except Exception as e:
        logger.error(f"❌ API endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    logger.info("🚀 Starting Data Service Tests...")

    success = True

    # Test data service
    if not await test_data_service():
        success = False

    # Test API endpoints
    if not await test_api_endpoints():
        success = False

    if success:
        logger.info("🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())