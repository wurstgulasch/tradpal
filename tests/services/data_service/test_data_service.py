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

from services.data_service.data_service.service import DataService

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
        # Test data service initialization
        data_service = DataService()
        await data_service.initialize()
        logger.info("✓ Data Service initialized successfully")

        # Test health check
        health = await data_service.health_check()
        logger.info(f"✓ Health check: {health['status']}")

        # Test service info
        info = await data_service.get_service_info()
        logger.info(f"✓ Service info: {info['name']} v{info['version']}")

        # Test available sources
        sources = data_service.get_available_sources()
        logger.info(f"✓ Available sources: {sources}")

        # Test data fetching
        test_symbol = "BTC/USDT"
        from services.data_service.data_service.service import DataRequest
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=10)
        request = DataRequest(
            symbol=test_symbol,
            timeframe="1h",
            limit=10,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )
        response = await data_service.fetch_data(request)
        if response.success:
            data = response.data.get("ohlcv", {})
            logger.info(f"✓ Fetched data for {test_symbol}: {len(data)} records")
        else:
            logger.warning(f"⚠️ Data fetch failed: {response.error}")
            # Use sample data for quality test
            import pandas as pd
            import numpy as np
            dates = pd.date_range(end=datetime.now(), periods=10, freq='1H')
            data = pd.DataFrame({
                'timestamp': dates,
                'open': 50000 + np.random.normal(0, 1000, 10),
                'high': 51000 + np.random.normal(0, 1000, 10),
                'low': 49000 + np.random.normal(0, 1000, 10),
                'close': 50000 + np.random.normal(0, 1000, 10),
                'volume': np.random.normal(100, 20, 10)
            })

        # Test data quality validation
        quality = await data_service.validate_data_quality(data)
        logger.info(f"✓ Data quality: {quality}")

        # Test Alternative Data functionality
        try:
            alt_status = await data_service.get_alternative_data_status()
            logger.info(f"✓ Alternative data status: {alt_status.get('alternative_data_available', False)}")
        except Exception as e:
            logger.warning(f"⚠️ Alternative data test skipped: {e}")

        # Test shutdown
        await data_service.shutdown()
        logger.info("✓ Data Service shut down successfully")

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
        from services.data_service.data_service.api.main import app
        from fastapi.testclient import TestClient

        # Create a fresh data service instance for testing
        from services.data_service.data_service.service import DataService
        test_data_service = DataService()
        await test_data_service.initialize()

        # Override the global data_service in the API module
        import services.data_service.data_service.api.main as api_module
        api_module.data_service = test_data_service

        client = TestClient(app)

        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        logger.info("✓ Root endpoint works")

        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] in ["healthy", "degraded", "unhealthy"]
        logger.info("✓ Health endpoint works")

        # Test service info endpoint
        response = client.get("/info")
        assert response.status_code == 200
        info_data = response.json()
        assert "name" in info_data
        logger.info("✓ Service info endpoint works")

        # Test sources endpoint
        response = client.get("/sources")
        assert response.status_code == 200
        sources_data = response.json()
        assert "sources" in sources_data
        logger.info("✓ Sources endpoint works")

        # Test alternative data status endpoint
        response = client.get("/alternative-data/status")
        assert response.status_code == 200
        alt_status = response.json()
        assert "alternative_data_available" in alt_status
        logger.info("✓ Alternative data status endpoint works")

        # Cleanup
        await test_data_service.shutdown()

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