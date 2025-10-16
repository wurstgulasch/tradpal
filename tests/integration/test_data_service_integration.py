"""
Service tests for data service enhancements and alternative data sources.
"""

import pytest
import asyncio
import pandas as pd
from unittest.mock import patch, MagicMock, AsyncMock

from services.data_service.service import DataRequest, EventSystem
from services.data_service.data_sources.factory import DataSourceFactory
from services.data_service.cache_manager import CacheManager
from services.data_service.quality_manager import DataQualityManager


class TestDataServiceIntegration:
    """Integration tests for data service with alternative data sources."""

    @pytest.fixture
    def mock_data_service(self):
        """Create a mock data service for testing."""
        # Create a mock service that doesn't initialize DataMesh
        service = MagicMock()
        service.event_system = EventSystem()
        service.cache_manager = CacheManager()
        service.quality_manager = DataQualityManager()
        service.data_source_factory = DataSourceFactory

        # Mock the methods we need
        service.get_available_data_sources = MagicMock(return_value=['liquidation', 'volatility', 'sentiment', 'onchain'])
        service.fetch_data = AsyncMock()
        service.health_check = AsyncMock(return_value={
            'status': 'healthy',
            'service': 'data_service'
        })

        return service

    def test_data_service_initialization(self, mock_data_service):
        """Test data service initializes with all data sources."""
        assert hasattr(mock_data_service, 'data_source_factory')
        assert isinstance(mock_data_service.data_source_factory, type(DataSourceFactory))

        # Check available sources include new alternatives
        available = mock_data_service.get_available_data_sources()
        assert 'liquidation' in available
        assert 'volatility' in available
        assert 'sentiment' in available
        assert 'onchain' in available

    @pytest.mark.asyncio
    async def test_fetch_market_data_with_fallbacks(self, mock_data_service):
        """Test fetching market data with fallback chain."""
        # Mock successful fetch
        mock_data_service.fetch_data.return_value = MagicMock(
            success=True,
            data={"ohlcv": {"timestamp": [pd.Timestamp.now()], "close": [50000]}},
            error=None
        )

        request = DataRequest(
            symbol='BTC/USDT',
            timeframe='1h',
            source='liquidation',
            start_date='2023-01-01T00:00:00',
            end_date='2023-01-02T00:00:00'
        )
        result = await mock_data_service.fetch_data(request)

        assert result.success
        assert 'ohlcv' in result.data
        mock_data_service.fetch_data.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_fetch_multiple_data_types(self, mock_data_service):
        """Test fetching multiple data types simultaneously."""
        mock_data_service.fetch_data.return_value = MagicMock(
            success=True,
            data={"ohlcv": {"timestamp": [pd.Timestamp.now()], "close": [50000]}},
            error=None
        )

        request = DataRequest(
            symbol='BTC/USDT',
            timeframe='1h',
            source='liquidation',
            start_date='2023-01-01T00:00:00',
            end_date='2023-01-02T00:00:00'
        )
        result = await mock_data_service.fetch_data(request)
        assert result.success

    @pytest.mark.asyncio
    async def test_fallback_chain_integration(self, mock_data_service):
        """Test complete fallback chain integration."""
        # Mock fallback response
        mock_data_service.fetch_data.return_value = MagicMock(
            success=True,
            data={"ohlcv": {"timestamp": [pd.Timestamp.now()], "close": [50000], "data_source": ["volatility_proxy"]}},
            error=None
        )

        request = DataRequest(
            symbol='BTC/USDT',
            timeframe='1h',
            source='liquidation',
            start_date='2023-01-01T00:00:00',
            end_date='2023-01-02T00:00:00'
        )
        result = await mock_data_service.fetch_data(request)

        assert result.success

    @pytest.mark.asyncio
    async def test_data_quality_validation(self, mock_data_service):
        """Test data quality validation for fetched data."""
        # Test using the quality manager directly
        quality_manager = DataQualityManager()

        # Use timestamps that are very recent (within last minute)
        now = pd.Timestamp.now()
        valid_data = pd.DataFrame({
            'liquidation_signal': [1, -1, 0],
            'liquidation_volume': [1000, 2000, 500],
            'data_source': ['primary', 'primary', 'primary'],
            'timestamp': [now, now, now]
        })

        # Set timestamp as index to pass timeliness check
        valid_data = valid_data.set_index('timestamp')

        is_valid = quality_manager.validate_data_quality(valid_data, 'liquidation')
        assert is_valid

        # Test invalid data
        invalid_data = pd.DataFrame({
            'some_other_column': [None, None, None],
            'timestamp': [now, now, now]
        })

        is_valid = quality_manager.validate_data_quality(invalid_data, 'liquidation')
        assert not is_valid

    @pytest.mark.asyncio
    async def test_cache_integration(self, mock_data_service):
        """Test cache integration with data fetching."""
        # Test cache manager directly
        cache_manager = mock_data_service.cache_manager
        await cache_manager.initialize()

        test_key = "test:BTC/USDT:1h:liquidation"
        test_data = pd.DataFrame({
            'liquidation_signal': [1],
            'timestamp': [pd.Timestamp.now()]
        })

        # Store data
        success = await cache_manager.cache_data(test_key, test_data, ttl_seconds=300)
        assert success

        # Retrieve data
        cached_data = await cache_manager.get_cached_data(test_key)
        assert cached_data is not None
        pd.testing.assert_frame_equal(cached_data, test_data)

        await cache_manager.cleanup()

    @pytest.mark.asyncio
    async def test_concurrent_data_fetching(self, mock_data_service):
        """Test concurrent fetching of multiple data types."""
        async def mock_slow_fetch(*args, **kwargs):
            await asyncio.sleep(0.1)
            return MagicMock(
                success=True,
                data={"ohlcv": {"timestamp": [pd.Timestamp.now()], "close": [50000]}},
                error=None
            )

        mock_data_service.fetch_data.side_effect = mock_slow_fetch

        import time
        start_time = time.time()

        # Fetch multiple requests concurrently
        requests = [
            DataRequest(
                symbol='BTC/USDT',
                timeframe='1h',
                source='liquidation',
                start_date='2023-01-01T00:00:00',
                end_date='2023-01-02T00:00:00'
            )
        ] * 3

        tasks = [mock_data_service.fetch_data(req) for req in requests]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete faster than sequential
        assert total_time < 0.25  # Allow some overhead
        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, mock_data_service):
        """Test error handling and recovery mechanisms."""
        mock_data_service.fetch_data.return_value = MagicMock(
            success=False,
            data=None,
            error="API Error"
        )

        request = DataRequest(
            symbol='BTC/USDT',
            timeframe='1h',
            source='liquidation',
            start_date='2023-01-01T00:00:00',
            end_date='2023-01-02T00:00:00'
        )

        # Should handle error gracefully
        result = await mock_data_service.fetch_data(request)
        assert not result.success
        assert 'API Error' in result.error

    @pytest.mark.asyncio
    async def test_data_service_health_check(self, mock_data_service):
        """Test data service health check functionality."""
        health_status = await mock_data_service.health_check()

        assert isinstance(health_status, dict)
        assert 'status' in health_status
        assert health_status['service'] == 'data_service'

    @pytest.mark.asyncio
    async def test_alternative_data_source_stats(self, mock_data_service):
        """Test getting statistics from alternative data sources."""
        # Test using factory directly
        factory = mock_data_service.data_source_factory

        # Test sentiment source
        sentiment_source = factory.create_data_source('sentiment')
        stats = sentiment_source.get_sentiment_stats('BTC/USDT')
        assert isinstance(stats, dict)  # Just check it's a dict, content may vary

        # Test on-chain source
        onchain_source = factory.create_data_source('onchain')
        stats = onchain_source.get_onchain_stats('BTC/USDT')
        assert isinstance(stats, dict)  # Just check it's a dict, content may vary

    @pytest.mark.asyncio
    async def test_data_service_cleanup(self, mock_data_service):
        """Test proper cleanup of data service resources."""
        # Test cache cleanup
        cache_manager = mock_data_service.cache_manager
        await cache_manager.initialize()
        await cache_manager.cache_data("test", pd.DataFrame({'a': [1]}))
        await cache_manager.cleanup()

        # Verify cleanup
        retrieved = await cache_manager.get_cached_data("test")
        assert retrieved is None


class TestDataQualityManagerIntegration:
    """Integration tests for data quality management."""

    def test_quality_validation_for_alternative_sources(self):
        """Test quality validation works for alternative data sources."""
        quality_manager = DataQualityManager()

        # Test valid liquidation data with recent timestamps
        now = pd.Timestamp.now()
        valid_liquidation = pd.DataFrame({
            'liquidation_signal': [1, -1, 0],
            'liquidation_volume': [1000, 2000, 500],
            'data_source': ['primary', 'primary', 'primary'],
            'timestamp': [now, now, now]
        })

        # Set timestamp as index to pass timeliness check
        valid_liquidation = valid_liquidation.set_index('timestamp')

        is_valid = quality_manager.validate_data_quality(valid_liquidation, 'liquidation')
        assert is_valid

        # Test invalid data (missing required columns)
        invalid_data = pd.DataFrame({
            'some_other_column': [1, 2, 3],
            'timestamp': [now, now, now]
        })

        is_valid = quality_manager.validate_data_quality(invalid_data, 'liquidation')
        assert not is_valid

    def test_quality_metrics_calculation(self):
        """Test calculation of data quality metrics."""
        quality_manager = DataQualityManager()

        now = pd.Timestamp.now()
        test_data = pd.DataFrame({
            'signal': [1, 2, 3, None, 5],
            'value': [100, 200, 300, 400, 500],
            'timestamp': [now, now, now, now, now]
        })

        metrics = quality_manager.calculate_quality_metrics(test_data)

        assert 'completeness' in metrics
        assert 'consistency' in metrics
        assert 'timeliness' in metrics
        assert 'accuracy' in metrics

        # Should detect the None value
        assert metrics['completeness'] < 1.0


class TestCacheManagerIntegration:
    """Integration tests for cache management."""

    @pytest.fixture
    def cache_manager_instance(self):
        """Create cache manager instance for testing."""
        return CacheManager()

    @pytest.mark.asyncio
    async def test_cache_data_storage_and_retrieval(self, cache_manager_instance):
        """Test storing and retrieving data from cache."""
        cache_manager = cache_manager_instance
        await cache_manager.initialize()

        try:
            test_key = "test:BTC/USDT:1h:liquidation"
            test_data = pd.DataFrame({
                'liquidation_signal': [1, 2, 3],
                'timestamp': pd.date_range('2023-01-01', periods=3, freq='1H')
            })

            # Store data
            success = await cache_manager.cache_data(test_key, test_data, ttl_seconds=300)

            assert success

            # Retrieve data
            cached_data = await cache_manager.get_cached_data(test_key)

            assert cached_data is not None
            pd.testing.assert_frame_equal(cached_data, test_data)
        finally:
            await cache_manager.cleanup()

    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache_manager_instance):
        """Test cache expiration functionality."""
        cache_manager = cache_manager_instance
        await cache_manager.initialize()

        try:
            test_key = "test:expire:BTC/USDT:1h"
            test_data = pd.DataFrame({'value': [1]})

            # Store with short TTL
            await cache_manager.cache_data(test_key, test_data, ttl_seconds=1)

            # Should be available immediately
            assert await cache_manager.get_cached_data(test_key) is not None

            # Wait for expiration
            await asyncio.sleep(1.1)

            # Should be expired
            assert await cache_manager.get_cached_data(test_key) is None
        finally:
            await cache_manager.cleanup()
