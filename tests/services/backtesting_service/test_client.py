"""
Unit tests for the Backtesting Service Client
Tests the async client functionality.
"""

import pytest
import aiohttp
from unittest.mock import Mock, patch, AsyncMock

from services.trading_service.backtesting_service.client import BacktestingServiceClient


class TestBacktestingServiceClient:
    """Test cases for BacktestingServiceClient."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing."""
        return BacktestingServiceClient(base_url="http://test-server:8030")

    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing."""
        return {
            'timestamp': ['2023-01-01T00:00:00', '2023-01-01T01:00:00'],
            'open': [50000, 50100],
            'high': [50200, 50300],
            'low': [49900, 50000],
            'close': [50100, 50000],
            'volume': [100, 150]
        }

    @pytest.fixture
    def sample_strategy_config(self):
        """Sample strategy configuration."""
        return {
            'name': 'SMA Crossover',
            'type': 'technical',
            'parameters': {'fast_period': 10, 'slow_period': 20}
        }

    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.base_url == "http://test-server:8030"
        assert client.timeout.total == 300.0  # 5 minutes
        assert client._session is None

    @pytest.mark.asyncio
    async def test_authenticate_placeholder(self, client):
        """Test authentication placeholder."""
        result = await client.authenticate()
        assert result is True

    @pytest.mark.asyncio
    async def test_run_backtest_success(self, client, sample_market_data, sample_strategy_config):
        """Test successful backtest execution."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Mock successful response
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "status": "completed",
                "backtest_id": "test-123",
                "data": {"performance": {}, "trades": [], "metrics": {}}
            }
            mock_session.post.return_value.__aenter__.return_value = mock_response

            result = await client.run_backtest(
                sample_strategy_config,
                sample_market_data,
                {"initial_capital": 10000}
            )

            # Verify the call
            mock_session.post.assert_called_once_with(
                "http://test-server:8030/backtest",
                json={
                    "strategy": sample_strategy_config,
                    "data": sample_market_data,
                    "config": {"initial_capital": 10000}
                }
            )

            assert "performance" in result
            assert "trades" in result

    @pytest.mark.asyncio
    async def test_run_backtest_failure(self, client):
        """Test backtest execution failure."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Mock failed response
            mock_response = AsyncMock()
            mock_response.raise_for_status.side_effect = aiohttp.ClientError("Server error")
            mock_session.post.return_value.__aenter__.return_value = mock_response

            with pytest.raises(aiohttp.ClientError):
                await client.run_backtest({}, {}, {})

    @pytest.mark.asyncio
    async def test_optimize_strategy(self, client, sample_market_data):
        """Test strategy optimization."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "best_parameters": {"fast_period": 10, "slow_period": 20},
                "best_score": 0.85,
                "optimization_history": []
            }
            mock_session.post.return_value.__aenter__.return_value = mock_response

            result = await client.optimize_strategy(
                "SMA Crossover",
                {"fast_period": [5, 15], "slow_period": [15, 25]},
                sample_market_data
            )

            assert "best_parameters" in result
            assert "best_score" in result

    @pytest.mark.asyncio
    async def test_get_backtest_status(self, client):
        """Test getting backtest status."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "status": "success",
                "data": {"status": "running", "progress": 0.5}
            }
            mock_session.get.return_value.__aenter__.return_value = mock_response

            result = await client.get_backtest_status("test-123")

            assert result["status"] == "running"
            assert result["progress"] == 0.5

    @pytest.mark.asyncio
    async def test_get_backtest_results(self, client):
        """Test getting backtest results."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "status": "success",
                "data": {"performance": {}, "trades": []}
            }
            mock_session.get.return_value.__aenter__.return_value = mock_response

            result = await client.get_backtest_results("test-123")

            assert "performance" in result
            assert "trades" in result

    @pytest.mark.asyncio
    async def test_list_active_backtests(self, client):
        """Test listing active backtests."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "status": "success",
                "data": {
                    "active_backtests": [{"id": "test-1", "status": "running"}],
                    "count": 1
                }
            }
            mock_session.get.return_value.__aenter__.return_value = mock_response

            result = await client.list_active_backtests()

            assert len(result) == 1
            assert result[0]["id"] == "test-1"

    @pytest.mark.asyncio
    async def test_list_completed_backtests(self, client):
        """Test listing completed backtests."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "status": "success",
                "data": {
                    "backtest_ids": ["test-1", "test-2"],
                    "count": 2
                }
            }
            mock_session.get.return_value.__aenter__.return_value = mock_response

            result = await client.list_completed_backtests()

            assert len(result) == 2
            assert "test-1" in result

    @pytest.mark.asyncio
    async def test_cancel_backtest(self, client):
        """Test cancelling a backtest."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "status": "success",
                "message": "Backtest cancelled"
            }
            mock_session.post.return_value.__aenter__.return_value = mock_response

            result = await client.cancel_backtest("test-123")

            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_get_service_health(self, client):
        """Test getting service health."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"status": "healthy"}
            mock_session.get.return_value.__aenter__.return_value = mock_response

            result = await client.get_service_health()

            assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_cleanup_old_results(self, client):
        """Test cleanup of old results."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "status": "success",
                "data": {"removed_count": 5}
            }
            mock_session.post.return_value.__aenter__.return_value = mock_response

            result = await client.cleanup_old_results(days=30)

            assert result["data"]["removed_count"] == 5

    @pytest.mark.asyncio
    async def test_batch_backtests(self, client, sample_market_data, sample_strategy_config):
        """Test batch backtest execution."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Mock successful responses for each backtest
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"performance": {}, "trades": []}
            mock_session.post.return_value.__aenter__.return_value = mock_response

            configs = [
                {"strategy": sample_strategy_config, "data": sample_market_data, "config": {}},
                {"strategy": sample_strategy_config, "data": sample_market_data, "config": {}}
            ]

            results = await client.batch_backtests(configs)

            assert len(results) == 2
            # Verify post was called twice
            assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_compare_strategies(self, client, sample_market_data):
        """Test strategy comparison."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "strategies": ["Strategy A", "Strategy B"],
                "comparison_metrics": {},
                "rankings": []
            }
            mock_session.post.return_value.__aenter__.return_value = mock_response

            strategies = [
                {"name": "Strategy A", "type": "technical", "parameters": {}},
                {"name": "Strategy B", "type": "technical", "parameters": {}}
            ]

            result = await client.compare_strategies(strategies, sample_market_data)

            assert "strategies" in result
            assert "comparison_metrics" in result

    @pytest.mark.asyncio
    async def test_get_performance_summary(self, client):
        """Test getting performance summary."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "status": "success",
                "data": {"summary_metrics": {}}
            }
            mock_session.post.return_value.__aenter__.return_value = mock_response

            result = await client.get_performance_summary(["test-1", "test-2"])

            assert "summary_metrics" in result

    @pytest.mark.asyncio
    async def test_client_context_manager(self, client):
        """Test client as async context manager."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            async with client:
                # Within context, session should be managed
                pass

            # After context, close should be called
            mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_method(self, client):
        """Test explicit close method."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            # Manually set session
            client._session = mock_session

            await client.close()

            mock_session.close.assert_called_once()
            assert client._session is None

    def test_client_with_custom_timeout(self):
        """Test client with custom timeout."""
        client = BacktestingServiceClient(timeout=600.0)
        assert client.timeout.total == 600.0

    @pytest.mark.asyncio
    async def test_batch_backtests_with_errors(self, client):
        """Test batch backtests with some failures."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Mock responses: first succeeds, second fails
            success_response = AsyncMock()
            success_response.raise_for_status.return_value = None
            success_response.json.return_value = {"performance": {}, "trades": []}

            fail_response = AsyncMock()
            fail_response.raise_for_status.side_effect = aiohttp.ClientError("Failed")

            # Configure post to return different responses
            mock_session.post.side_effect = [
                success_response.__aenter__(),
                fail_response.__aenter__()
            ]

            configs = [
                {"strategy": {}, "data": {}, "config": {}},
                {"strategy": {}, "data": {}, "config": {}}
            ]

            results = await client.batch_backtests(configs)

            assert len(results) == 2
            assert "performance" in results[0]  # First succeeded
            assert "error" in results[1]  # Second failed

    @pytest.mark.asyncio
    async def test_session_reuse(self, client):
        """Test that HTTP sessions are reused."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            # First call should create session
            with patch.object(client, '_get_session') as mock_get_session:
                mock_get_session.return_value.__aenter__.return_value = mock_session
                mock_get_session.return_value.__aexit__.return_value = None

                await client.get_service_health()
                await client.get_service_health()

                # _get_session should be called twice, but session creation once
                assert mock_get_session.call_count == 2