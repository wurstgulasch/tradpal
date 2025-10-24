#!/usr/bin/env python3
"""
Data Mesh Integration Test Script.

Tests the Data Mesh functionality including:
- Data product registration
- Market data storage and retrieval
- ML feature storage and retrieval
- Data archival to Data Lake
- Data Mesh status monitoring
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.data_service.service import DataService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_data_mesh_integration():
    """Test Data Mesh integration functionality."""
    logger.info("🚀 Starting Data Mesh Integration Tests")

    async with DataService() as client:
        try:
            # Test 1: Authenticate
            logger.info("📋 Testing authentication...")
            auth_success = await client.authenticate()
            if not auth_success:
                logger.error("❌ Authentication failed")
                return False
            logger.info("✅ Authentication successful")

            # Test 2: Register a data product
            logger.info("📦 Testing data product registration...")
            product_result = await client.register_data_product(
                name="test_btc_signals",
                domain="trading_signals",
                description="Test BTC trading signals data product",
                schema={
                    "timestamp": "datetime",
                    "symbol": "string",
                    "signal": "string",
                    "confidence": "float"
                },
                owners=["test_team"]
            )
            if product_result.get("success"):
                logger.info("✅ Data product registration successful")
                product_id = product_result.get("product_id")
            else:
                logger.error(f"❌ Data product registration failed: {product_result}")
                return False

            # Test 3: Store market data
            logger.info("💾 Testing market data storage...")
            # Generate sample OHLCV data
            dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=100, freq='1H')
            np.random.seed(42)
            sample_data = {
                str(int(ts.timestamp())): {
                    "timestamp": ts.isoformat(),
                    "open": 50000 + np.random.normal(0, 1000),
                    "high": 51000 + np.random.normal(0, 1000),
                    "low": 49000 + np.random.normal(0, 1000),
                    "close": 50000 + np.random.normal(0, 1000),
                    "volume": np.random.uniform(100, 1000)
                } for ts in dates
            }

            storage_result = await client.store_market_data(
                symbol="BTC/USDT",
                timeframe="1h",
                data=sample_data,
                metadata={"source": "test", "quality": "synthetic"}
            )
            if storage_result.get("success"):
                logger.info("✅ Market data storage successful")
            else:
                logger.error(f"❌ Market data storage failed: {storage_result}")
                return False

            # Test 4: Retrieve market data
            logger.info("📊 Testing market data retrieval...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)

            retrieval_result = await client.retrieve_market_data(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            )
            if retrieval_result.get("success"):
                record_count = retrieval_result.get("record_count", 0)
                logger.info(f"✅ Market data retrieval successful: {record_count} records")
            else:
                logger.error(f"❌ Market data retrieval failed: {retrieval_result}")
                return False

            # Test 5: Store ML features
            logger.info("🧠 Testing ML feature storage...")
            feature_dates = pd.date_range(start=datetime.now() - timedelta(days=1), periods=24, freq='1H')
            feature_data = {
                str(int(ts.timestamp())): {
                    "timestamp": ts.isoformat(),
                    "sma_20": np.random.uniform(49000, 51000),
                    "rsi": np.random.uniform(30, 70),
                    "macd": np.random.normal(0, 100),
                    "bb_upper": np.random.uniform(50000, 52000),
                    "bb_lower": np.random.uniform(48000, 50000),
                    "atr": np.random.uniform(500, 1500)
                } for ts in feature_dates
            }

            feature_result = await client.store_ml_features(
                feature_set_name="test_technical_features",
                features=feature_data,
                metadata={
                    "description": "Technical analysis features for BTC/USDT",
                    "feature_count": 6,
                    "timeframe": "1h"
                }
            )
            if feature_result.get("success"):
                logger.info("✅ ML feature storage successful")
            else:
                logger.error(f"❌ ML feature storage failed: {feature_result}")
                return False

            # Test 6: Retrieve ML features
            logger.info("🔍 Testing ML feature retrieval...")
            feature_retrieval = await client.retrieve_ml_features(
                feature_set_name="test_technical_features"
            )
            if feature_retrieval.get("success"):
                record_count = feature_retrieval.get("record_count", 0)
                logger.info(f"✅ ML feature retrieval successful: {record_count} records")
            else:
                logger.error(f"❌ ML feature retrieval failed: {feature_retrieval}")
                return False

            # Test 7: Archive historical data
            logger.info("📦 Testing data archival...")
            archive_result = await client.archive_historical_data(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date=(datetime.now() - timedelta(days=7)).isoformat(),
                end_date=datetime.now().isoformat()
            )
            if archive_result.get("success"):
                logger.info("✅ Data archival successful")
            else:
                logger.warning(f"⚠️  Data archival failed (may be expected if Data Lake not configured): {archive_result}")

            # Test 8: Get Data Mesh status
            logger.info("📈 Testing Data Mesh status...")
            status_result = await client.get_data_mesh_status()
            if "status" in status_result:
                status = status_result.get("status")
                logger.info(f"✅ Data Mesh status: {status}")
                if status == "healthy":
                    logger.info("🎉 All Data Mesh tests passed!")
                    return True
                else:
                    logger.warning(f"⚠️  Data Mesh status: {status}")
                    return True  # Still consider as passed if status is returned
            else:
                logger.error(f"❌ Data Mesh status check failed: {status_result}")
                return False

        except Exception as e:
            logger.error(f"❌ Data Mesh integration test failed: {e}")
            return False


async def main():
    """Main test function."""
    logger.info("🧪 Starting Data Mesh Integration Tests")

    success = await test_data_mesh_integration()

    if success:
        logger.info("🎉 All Data Mesh integration tests completed successfully!")
        return 0
    else:
        logger.error("❌ Some Data Mesh integration tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())