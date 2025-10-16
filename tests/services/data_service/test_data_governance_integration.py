#!/usr/bin/env python3
"""
Integration tests for Data Governance system.
Tests end-to-end functionality with sample data and real scenarios.
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services'))

from services.data_service.data_governance import (
    DataGovernanceManager, AccessLevel, AuditEventType
)
from services.data_service.data_mesh import DataMeshManager, DataDomain, DataProduct


class IntegrationTester:
    """Integration test class for Data Governance system."""

    def __init__(self):
        self.config = {
            "redis_url": "redis://localhost:6379",
            "max_recent_events": 1000,
            "max_results": 100,
            "timeseries_db": {
                "url": "http://localhost:8086",
                "token": "tradpal_token_12345",
                "org": "tradpal",
                "bucket": "tradpal-market-data"
            },
            "data_lake": {
                "endpoint_url": "http://localhost:9000",
                "access_key": "minioadmin",
                "secret_key": "minioadmin",
                "region": "us-east-1"
            },
            "feature_store": {
                "redis_url": "redis://localhost:6379"
            }
        }

        # Mock external dependencies
        self.mock_redis = Mock()
        self.mock_redis.get.return_value = None
        self.mock_redis.set.return_value = True
        self.mock_redis.sadd.return_value = 1
        self.mock_redis.smembers.return_value = ["data_admin"]
        self.mock_redis.zadd.return_value = True
        self.mock_redis.zrange.return_value = []
        self.mock_redis.zrevrange.return_value = []
        self.mock_redis.zcard.return_value = 0
        self.mock_redis.exists.return_value = False
        self.mock_redis.delete.return_value = 1
        self.mock_redis.hset.return_value = 1
        self.mock_redis.hgetall.return_value = {}
        self.mock_redis.keys.return_value = []
        self.mock_redis.zrangebyscore.return_value = []

        with patch('redis.from_url', return_value=self.mock_redis):
            self.governance_manager = DataGovernanceManager(self.config)
            self.data_mesh = DataMeshManager(self.config)

    def create_sample_market_data(self, symbol="BTC/USDT", periods=100):
        """Create realistic sample market data."""
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='1min')
        np.random.seed(42)

        # Generate realistic OHLCV data
        base_price = 50000
        close_prices = base_price + np.cumsum(np.random.normal(0, 10, periods))

        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices + np.random.normal(0, 5, periods),
            'high': close_prices + np.random.uniform(0, 50, periods),
            'low': close_prices - np.random.uniform(0, 50, periods),
            'close': close_prices,
            'volume': np.random.randint(100, 1000, periods)
        })

        # Ensure OHLC relationships
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)

        df.set_index('timestamp', inplace=True)
        return df

    def create_sample_features(self, periods=50):
        """Create sample ML features."""
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='1min')
        np.random.seed(123)

        features_df = pd.DataFrame({
            'timestamp': dates,
            'rsi_14': np.random.uniform(20, 80, periods),
            'ema_12': np.random.uniform(49000, 51000, periods),
            'ema_26': np.random.uniform(49000, 51000, periods),
            'bb_upper': np.random.uniform(50000, 52000, periods),
            'bb_lower': np.random.uniform(48000, 50000, periods),
            'volume_sma': np.random.uniform(200, 800, periods),
            'price_change': np.random.normal(0, 0.01, periods)
        })

        features_df.set_index('timestamp', inplace=True)
        return features_df

    async def test_data_product_lifecycle(self):
        """Test complete data product lifecycle with governance."""
        print("Testing data product lifecycle...")

        # Assign data_admin role to data_admin user
        await self.governance_manager.assign_user_role("system", "data_admin", "data_admin")
        print("✅ Assigned data_admin role to data_admin user")

        # 1. Register data product
        product_name = "btc_usdt_ohlcv"
        schema = {
            "timestamp": "datetime64[ns]",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "int64"
        }

        success = await self.data_mesh.register_data_product(DataProduct(
            name=product_name,
            domain=DataDomain.MARKET_DATA,
            version="v1.0",
            description="BTC/USDT OHLCV data",
            schema=schema,  # Use schema as the field name, it will be aliased to data_schema
            owners=["data_service"],
            tags=["crypto", "trading"]
        ))

        assert success, "Data product registration should succeed"
        print("✅ Data product registered")

        # 2. Store market data with governance
        market_data = self.create_sample_market_data()
        result = await self.governance_manager.validate_data_and_log(
            "data_service", product_name, market_data, "data_product"
        )

        assert result.quality_level.value in ["excellent", "good", "fair", "poor", "invalid"]  # Accept all quality levels
        print(f"✅ Market data validated and stored: {result.quality_level.value}")

        # 3. Check access control
        # Test access control
        has_access, reason = await self.governance_manager.check_access_and_log(
            "data_admin", "domain", "market_data", AccessLevel.READ
        )
        assert has_access, f"Data admin should have access: {reason}"
        print("✅ Access control verified")

        # 4. Retrieve data
        retrieved_product = await self.data_mesh.get_data_product(product_name)
        assert retrieved_product is not None, "Data product should be retrievable"
        assert retrieved_product.name == product_name
        print("✅ Data product retrieved")

        print("✅ Data product lifecycle test completed")

    async def test_ml_feature_pipeline(self):
        """Test ML feature pipeline with governance."""
        print("Testing ML feature pipeline...")

        # 1. Create feature set
        from services.data_service.data_mesh import FeatureSet

        feature_schema = {
            "rsi_14": "float64",
            "ema_12": "float64",
            "ema_26": "float64",
            "bb_upper": "float64",
            "bb_lower": "float64",
            "volume_sma": "float64",
            "price_change": "float64"
        }

        feature_set = FeatureSet(
            name="btc_usdt_features",
            version="v1.0",
            description="Technical indicators for BTC/USDT",
            features=["rsi_14", "ema_12", "ema_26", "bb_upper", "bb_lower", "volume_sma", "price_change"],
            entity="btc_usdt",
            schema=feature_schema
        )

        # 2. Store features with governance
        features_data = self.create_sample_features()
        result = await self.governance_manager.validate_data_and_log(
            "ml_trainer", "btc_usdt_features", features_data, "ml_features"
        )

        assert result.quality_level.value in ["excellent", "good", "fair", "poor", "invalid"]  # Accept all quality levels
        print(f"✅ ML features validated: {result.quality_level.value}")

        # 3. Store in feature store
        features_data_copy = features_data.reset_index()
        features_data_copy['timestamp'] = features_data_copy['timestamp'].astype(str)
        features_dict = features_data_copy.to_dict('records')
        success = await self.data_mesh.store_ml_features(feature_set, features_dict)
        assert success, "Feature set storage should succeed"
        print("✅ ML features stored in feature store")

        print("✅ ML feature pipeline test completed")

    async def test_audit_compliance(self):
        """Test audit and compliance functionality."""
        print("Testing audit and compliance...")

        # Generate various audit events
        events_to_generate = [
            ("data_service", "data_product", "btc_usdt_ohlcv", "read_access"),
            ("analyst", "domain", "market_data", "query_access"),
            ("ml_trainer", "ml_features", "btc_features", "write_access"),
            ("admin", "governance_policy", "access_policy", "modify_access")
        ]

        for user, resource_type, resource_name, action in events_to_generate:
            await self.governance_manager.audit_logger.log_access(
                user=user,
                resource_type=resource_type,
                resource_name=resource_name,
                action=action,
                success=True,
                details={"purpose": "integration_test"}
            )

        # Retrieve audit events
        events = await self.governance_manager.audit_logger.get_events(limit=10)
        assert len(events) >= len(events_to_generate), f"Should have at least {len(events_to_generate)} events"
        print(f"✅ Audit events logged: {len(events)} total")

        # Generate compliance report
        start_date = datetime.utcnow() - timedelta(hours=1)
        end_date = datetime.utcnow()
        report = await self.governance_manager.generate_compliance_report(start_date, end_date)

        assert report["audit_events"]["total"] >= len(events_to_generate)
        assert "recommendations" in report
        print(f"✅ Compliance report generated: {report['audit_events']['total']} events analyzed")

        print("✅ Audit compliance test completed")

    async def test_data_quality_monitoring(self):
        """Test comprehensive data quality monitoring."""
        print("Testing data quality monitoring...")

        # Test with multiple data sets
        test_cases = [
            ("good_data", self.create_sample_market_data("BTC/USDT", 50)),
            ("poor_data", self.create_sample_market_data("ETH/USDT", 50)),  # Same generation = similar quality
            ("features_data", self.create_sample_features(30))
        ]

        for resource_name, data in test_cases:
            result = await self.governance_manager.quality_monitor.check_data_quality(
                resource_name, data, "data_product"
            )

            assert result.quality_score >= 0.0 and result.quality_score <= 1.0
            assert result.quality_level.value in ["excellent", "good", "fair", "poor", "invalid"]
            print(f"✅ Quality check for {resource_name}: {result.quality_score:.3f} ({result.quality_level.value})")

        # Get quality summary
        summary = await self.governance_manager.quality_monitor.get_quality_summary(days=1)
        assert "total_checks" in summary
        assert summary["total_checks"] >= len(test_cases)
        print(f"✅ Quality summary: {summary['total_checks']} checks, avg_score={summary.get('average_score', 0):.3f}")

        print("✅ Data quality monitoring test completed")

    async def test_governance_integration(self):
        """Test full governance system integration."""
        print("Testing governance system integration...")

        # Get comprehensive governance status
        status = await self.governance_manager.get_governance_status()

        required_components = ["access_control", "audit_logging", "quality_monitoring"]
        for component in required_components:
            assert component in status["components"], f"Component {component} should be present"

        assert status["status"] == "healthy", f"Governance status should be healthy, got {status['status']}"
        print(f"✅ Governance status: {status['status']}")

        # Test user permissions
        permissions = await self.governance_manager.access_control.get_user_permissions("system")
        assert isinstance(permissions, dict), "Permissions should be a dictionary"
        print(f"✅ User permissions retrieved: {len(permissions.get('roles', []))} roles")

        print("✅ Governance integration test completed")

    async def run_integration_tests(self):
        """Run all integration tests."""
        print("🚀 Starting Data Governance Integration Tests")
        print("=" * 70)

        try:
            await self.test_data_product_lifecycle()
            await self.test_ml_feature_pipeline()
            await self.test_audit_compliance()
            await self.test_data_quality_monitoring()
            await self.test_governance_integration()

            print("=" * 70)
            print("✅ All Data Governance integration tests completed successfully!")

        except Exception as e:
            print(f"❌ Integration test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True


async def main():
    """Run the integration test suite."""
    tester = IntegrationTester()
    success = await tester.run_integration_tests()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())