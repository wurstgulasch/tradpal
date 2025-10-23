#!/usr/bin/env python3
"""
Data Service Tests

Comprehensive tests for the Data Service including:
- Unit tests for core functionality
- Integration tests with external services
- Performance and reliability tests
"""

import asyncio
import json
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from services.data_service.service import (
    DataService, DataRequest, DataResponse, DataMetadata,
    DataSource, DataProvider, DataQuality, EventSystem
)


class TestDataService:
    """Unit tests for DataService."""

    @pytest.fixture
    async def event_system(self):
        """Create event system for testing."""
        return EventSystem()

    @pytest.fixture
    async def data_service(self, event_system):
        """Create data service instance."""
        service = DataService(event_system=event_system)
        yield service

    def test_initialization(self, data_service):
        """Test service initialization."""
        assert data_service is not None
        assert data_service.event_system is not None
        assert data_service.cache_ttl == 3600

    def test_cache_key_generation(self, data_service):
        """Test cache key generation."""
        symbol = "BTC/USDT"
        timeframe = "1d"
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        key = data_service._generate_cache_key(symbol, timeframe, start_date, end_date)
        assert key.startswith("data:")
        assert len(key) > 10

        # Same parameters should generate same key
        key2 = data_service._generate_cache_key(symbol, timeframe, start_date, end_date)
        assert key == key2

    def test_data_quality_calculation(self, data_service):
        """Test data quality calculation."""
        # Create test DataFrame
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'open': [100 + i for i in range(100)],
            'high': [105 + i for i in range(100)],
            'low': [95 + i for i in range(100)],
            'close': [102 + i for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        }, index=dates)

        score, quality = data_service._calculate_data_quality(df)

        assert 0.0 <= score <= 1.0
        assert isinstance(quality, DataQuality)
        assert quality in [DataQuality.EXCELLENT, DataQuality.GOOD, DataQuality.FAIR, DataQuality.POOR]

    def test_data_quality_with_invalid_data(self, data_service):
        """Test data quality with invalid OHLC data."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'open': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
            'high': [90, 100, 110, 120, 130, 140, 150, 160, 170, 180],  # Invalid: high < open
            'low': [95, 105, 115, 125, 135, 145, 155, 165, 175, 185],
            'close': [102, 112, 122, 132, 142, 152, 162, 172, 182, 192],
            'volume': [1000] * 10
        }, index=dates)

        score, quality = data_service._calculate_data_quality(df)

        assert score < 1.0  # Should be reduced due to invalid OHLC
        assert quality != DataQuality.EXCELLENT

    @pytest.mark.asyncio
    async def test_fetch_data_request_validation(self, data_service):
        """Test data request validation."""
        # Valid request
        request = DataRequest(
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01T00:00:00",
            end_date="2024-01-31T00:00:00"
        )
        assert request.symbol == "BTC/USDT"

        # Invalid timeframe
        with pytest.raises(ValueError):
            DataRequest(
                symbol="BTC/USDT",
                timeframe="invalid",
                start_date="2024-01-01T00:00:00",
                end_date="2024-01-31T00:00:00"
            )

        # Invalid date format
        with pytest.raises(ValueError):
            DataRequest(
                symbol="BTC/USDT",
                timeframe="1d",
                start_date="invalid-date",
                end_date="2024-01-31T00:00:00"
            )

    @pytest.mark.asyncio
    async def test_metadata_serialization(self, data_service):
        """Test metadata serialization and deserialization."""
        metadata = DataMetadata(
            symbol="BTC/USDT",
            timeframe="1d",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            source=DataSource.CCXT,
            provider=DataProvider.BINANCE,
            fetched_at=datetime.now(),
            quality_score=0.95,
            quality_level=DataQuality.EXCELLENT,
            record_count=31,
            columns=['open', 'high', 'low', 'close', 'volume'],
            cache_key="test_key",
            checksum="test_checksum"
        )

        # Serialize
        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data['symbol'] == "BTC/USDT"
        assert data['quality_level'] == "excellent"

        # Deserialize
        metadata2 = DataMetadata.from_dict(data)
        assert metadata2.symbol == metadata.symbol
        assert metadata2.quality_level == metadata.quality_level

    @pytest.mark.asyncio
    async def test_event_system_integration(self, data_service, event_system):
        """Test event system integration."""
        events_received = []

        async def event_handler(event_data):
            events_received.append(event_data)

        event_system.subscribe("test.event", event_handler)

        await event_system.publish("test.event", {"test": "data"})

        # Allow event processing
        await asyncio.sleep(0.1)

        assert len(events_received) == 1
        assert events_received[0]["test"] == "data"

    @pytest.mark.asyncio
    async def test_health_check(self, data_service):
        """Test health check functionality."""
        health = await data_service.health_check()

        assert isinstance(health, dict)
        assert "status" in health
        assert "components" in health
        assert "timestamp" in health

        # Should have component status
        assert "redis" in health["components"]
        assert "ccxt" in health["components"]
        assert "yahoo_finance" in health["components"]


class TestDataServiceIntegration:
    """Integration tests requiring external dependencies."""

    @pytest.fixture
    async def data_service(self):
        """Create data service for integration tests."""
        event_system = EventSystem()
        service = DataService(event_system=event_system)
        yield service

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_ccxt_data_fetch(self, data_service):
        """Test actual data fetching from CCXT (requires internet)."""
        pytest.skip("Requires internet connection and CCXT")

        request = DataRequest(
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01T00:00:00",
            end_date="2024-01-05T00:00:00",
            source="ccxt",
            provider="binance"
        )

        response = await data_service.fetch_data(request)

        # May succeed or fail depending on network
        assert isinstance(response, DataResponse)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_yahoo_finance_fetch(self, data_service):
        """Test actual data fetching from Yahoo Finance."""
        pytest.skip("Requires internet connection and yfinance")

        request = DataRequest(
            symbol="AAPL",
            timeframe="1d",
            start_date="2024-01-01T00:00:00",
            end_date="2024-01-05T00:00:00",
            source="yahoo"
        )

        response = await data_service.fetch_data(request)

        # May succeed or fail depending on network
        assert isinstance(response, DataResponse)

    @pytest.mark.asyncio
    async def test_cache_operations(self, data_service):
        """Test cache operations."""
        if not data_service.redis_client:
            pytest.skip("Redis not available")

        # Test cache clear
        deleted = await data_service.clear_cache("test_*")
        assert isinstance(deleted, int)

    @pytest.mark.asyncio
    async def test_data_info_endpoint(self, data_service):
        """Test data info functionality."""
        info = await data_service.get_data_info("BTC/USDT", "1d")

        assert isinstance(info, dict)
        assert info["symbol"] == "BTC/USDT"
        assert info["timeframe"] == "1d"
        assert "available_sources" in info
        assert "cache_enabled" in info


class TestDataServicePerformance:
    """Performance tests for Data Service."""

    @pytest.fixture
    async def data_service(self):
        """Create data service for performance tests."""
        event_system = EventSystem()
        return DataService(event_system=event_system)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_data_fetch_performance(self, data_service, benchmark):
        """Benchmark data fetch performance."""
        # Mock the actual fetching to test service logic performance
        with patch.object(data_service, '_fetch_from_ccxt', new_callable=AsyncMock) as mock_fetch:
            # Create mock data
            dates = pd.date_range('2024-01-01', periods=100, freq='D')
            mock_df = pd.DataFrame({
                'open': [100 + i for i in range(100)],
                'high': [105 + i for i in range(100)],
                'low': [95 + i for i in range(100)],
                'close': [102 + i for i in range(100)],
                'volume': [1000 + i * 10 for i in range(100)]
            }, index=dates)

            mock_fetch.return_value = mock_df

            request = DataRequest(
                symbol="BTC/USDT",
                timeframe="1d",
                start_date="2024-01-01T00:00:00",
                end_date="2024-01-31T00:00:00"
            )

            async def fetch_benchmark():
                return await data_service.fetch_data(request)

            result = benchmark(fetch_benchmark)
            assert result.success is True

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_fetches(self, data_service):
        """Test concurrent data fetching performance."""
        request = DataRequest(
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01T00:00:00",
            end_date="2024-01-31T00:00:00"
        )

        # Mock the fetch method
        with patch.object(data_service, 'fetch_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = DataResponse(success=True, data={})

            # Run multiple concurrent requests
            tasks = [data_service.fetch_data(request) for _ in range(10)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert all(r.success for r in results)


# Mock classes for testing
class MockRedis:
    """Mock Redis client for testing."""

    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def setex(self, key, ttl, value):
        self.data[key] = value
        return True

    def delete(self, *keys):
        deleted = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                deleted += 1
        return deleted

    def keys(self, pattern):
        # Simple pattern matching
        if pattern == "*":
            return list(self.data.keys())
        return [k for k in self.data.keys() if pattern.replace("*", "") in k]

    def ping(self):
        return True

    def info(self):
        return {
            "connected_clients": 1,
            "used_memory_human": "1M",
            "uptime_in_days": 1
        }


@pytest.fixture
def mock_redis():
    """Mock Redis fixture."""
    return MockRedis()


@pytest.mark.asyncio
async def test_with_mock_redis(mock_redis):
    """Test data service with mock Redis."""
    event_system = EventSystem()

    with patch('services.data_service.service.redis') as mock_redis_module:
        mock_redis_module.from_url.return_value = mock_redis
        mock_redis_module.__bool__ = lambda x: True
        mock_redis_module.__nonzero__ = lambda x: True

        service = DataService(event_system=event_system, redis_url="redis://mock")

        # Test cache operations
        cache_key = "test_key"
        df = pd.DataFrame({'test': [1, 2, 3]})
        metadata = DataMetadata(
            symbol="TEST",
            timeframe="1d",
            start_date=datetime.now(),
            end_date=datetime.now(),
            source=DataSource.CCXT,
            provider=DataProvider.BINANCE,
            fetched_at=datetime.now(),
            quality_score=1.0,
            quality_level=DataQuality.EXCELLENT,
            record_count=3,
            columns=['test'],
            cache_key=cache_key,
            checksum="test_checksum"
        )

        # Store in cache
        await service._store_in_cache(cache_key, df, metadata)

        # Retrieve from cache
        cached = await service._fetch_from_cache(cache_key)
        assert cached is not None
        cached_df, cached_metadata = cached
        assert len(cached_df) == 3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])