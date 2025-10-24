"""
Integration tests for the Backtesting Service API
Tests the client and service integration.
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

from services.trading_service.backtesting_service.main import app
from services.trading_service.backtesting_service.client import BacktestingServiceClient


class TestBacktestingServiceIntegration:
    """Integration tests for Backtesting Service."""

    @pytest.fixture
    def test_client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing."""
        return {
            'timestamp': ['2023-01-01T00:00:00', '2023-01-01T01:00:00', '2023-01-01T02:00:00'],
            'open': [50000, 50100, 49900],
            'high': [50200, 50300, 50100],
            'low': [49900, 50000, 49800],
            'close': [50100, 50000, 50050],
            'volume': [100, 150, 120]
        }

    @pytest.fixture
    def sample_strategy_config(self):
        """Sample strategy configuration."""
        return {
            'name': 'SMA Crossover',
            'type': 'technical',
            'parameters': {
                'fast_period': 2,
                'slow_period': 3
            }
        }

    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "backtesting_service"
        assert "timestamp" in data

    def test_run_backtest_endpoint(self, test_client, sample_market_data, sample_strategy_config):
        """Test backtest endpoint."""
        payload = {
            "strategy": sample_strategy_config,
            "data": sample_market_data,
            "config": {
                "initial_capital": 10000,
                "run_async": False
            }
        }

        response = test_client.post("/backtest", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "data" in data

        result = data["data"]
        assert "performance" in result
        assert "trades" in result
        assert "metrics" in result

    def test_run_backtest_async_endpoint(self, test_client, sample_market_data, sample_strategy_config):
        """Test async backtest endpoint."""
        payload = {
            "strategy": sample_strategy_config,
            "data": sample_market_data,
            "config": {
                "initial_capital": 10000,
                "run_async": True
            }
        }

        response = test_client.post("/backtest", json=payload)

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "accepted"
        assert "backtest_id" in data

    def test_get_backtest_status_endpoint(self, test_client):
        """Test backtest status endpoint."""
        # First create an async backtest
        payload = {
            "strategy": {
                "name": "Test Strategy",
                "type": "technical",
                "parameters": {"fast_period": 2, "slow_period": 3}
            },
            "data": {
                "timestamp": ["2023-01-01T00:00:00"],
                "open": [50000], "high": [50200], "low": [49900], "close": [50100], "volume": [100]
            },
            "config": {"run_async": True}
        }

        create_response = test_client.post("/backtest", json=payload)
        backtest_id = create_response.json()["backtest_id"]

        # Check status
        status_response = test_client.get(f"/backtest/{backtest_id}/status")
        assert status_response.status_code == 200

        status_data = status_response.json()
        assert status_data["status"] == "success"
        assert "data" in status_data

    def test_compare_strategies_endpoint(self, test_client, sample_market_data):
        """Test strategy comparison endpoint."""
        payload = {
            "strategies": [
                {
                    "name": "Strategy A",
                    "type": "technical",
                    "parameters": {"fast_period": 2, "slow_period": 3}
                },
                {
                    "name": "Strategy B",
                    "type": "technical",
                    "parameters": {"fast_period": 1, "slow_period": 2}
                }
            ],
            "data": sample_market_data,
            "config": {"initial_capital": 10000}
        }

        response = test_client.post("/compare", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data

    def test_list_active_backtests_endpoint(self, test_client):
        """Test list active backtests endpoint."""
        response = test_client.get("/backtests/active")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "active_backtests" in data["data"]
        assert "count" in data["data"]

    def test_list_completed_backtests_endpoint(self, test_client):
        """Test list completed backtests endpoint."""
        response = test_client.get("/backtests/completed")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "backtest_ids" in data["data"]
        assert "count" in data["data"]

    def test_cancel_backtest_endpoint(self, test_client):
        """Test cancel backtest endpoint."""
        # Create a backtest first
        payload = {
            "strategy": {
                "name": "Test Strategy",
                "type": "technical",
                "parameters": {"fast_period": 2, "slow_period": 3}
            },
            "data": {
                "timestamp": ["2023-01-01T00:00:00"],
                "open": [50000], "high": [50200], "low": [49900], "close": [50100], "volume": [100]
            },
            "config": {"run_async": True}
        }

        create_response = test_client.post("/backtest", json=payload)
        backtest_id = create_response.json()["backtest_id"]

        # Cancel it
        cancel_response = test_client.post(f"/backtest/{backtest_id}/cancel")
        assert cancel_response.status_code == 200

        cancel_data = cancel_response.json()
        assert cancel_data["status"] == "success"

    def test_cleanup_endpoint(self, test_client):
        """Test cleanup endpoint."""
        response = test_client.post("/maintenance/cleanup", json={"days": 30})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data

    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint."""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data

    def test_invalid_backtest_request(self, test_client):
        """Test invalid backtest request handling."""
        # Missing strategy
        payload = {
            "data": {"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
        }

        response = test_client.post("/backtest", json=payload)
        assert response.status_code == 422  # Validation error

    def test_backtest_not_found(self, test_client):
        """Test handling of non-existent backtest."""
        response = test_client.get("/backtest/non-existent-id/status")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_client_integration(self, sample_market_data, sample_strategy_config):
        """Test client integration with service."""
        # Mock the HTTP client to avoid actual network calls
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Mock successful response
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "status": "completed",
                "backtest_id": "test-id",
                "data": {"performance": {}, "trades": [], "metrics": {}}
            }
            mock_session.post.return_value.__aenter__.return_value = mock_response

            client = BacktestingServiceClient()

            result = await client.run_backtest(
                sample_strategy_config,
                sample_market_data,
                {"initial_capital": 10000}
            )

            # Verify the call was made correctly
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            assert call_args[0][0] == "http://localhost:8030/backtest"

            # Verify payload structure
            payload = call_args[1]["json"]
            assert "strategy" in payload
            assert "data" in payload
            assert "config" in payload

    @pytest.mark.asyncio
    async def test_client_error_handling(self):
        """Test client error handling."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Mock error response
            mock_response = AsyncMock()
            mock_response.raise_for_status.side_effect = aiohttp.ClientError("Connection failed")
            mock_session.post.return_value.__aenter__.return_value = mock_response

            client = BacktestingServiceClient()

            with pytest.raises(aiohttp.ClientError):
                await client.run_backtest({}, {}, {})

    @pytest.mark.asyncio
    async def test_client_batch_operations(self, sample_market_data, sample_strategy_config):
        """Test client batch operations."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Mock successful responses
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"performance": {}, "trades": []}
            mock_session.post.return_value.__aenter__.return_value = mock_response

            client = BacktestingServiceClient()

            configs = [
                {"strategy": sample_strategy_config, "data": sample_market_data, "config": {}},
                {"strategy": sample_strategy_config, "data": sample_market_data, "config": {}}
            ]

            results = await client.batch_backtests(configs)

            # Should have 2 results
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_client_health_check(self):
        """Test client health check."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Mock health response
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"status": "healthy"}
            mock_session.get.return_value.__aenter__.return_value = mock_response

            client = BacktestingServiceClient()

            health = await client.get_service_health()

            assert health["status"] == "healthy"

    def test_cors_headers(self, test_client):
        """Test CORS headers are set correctly."""
        response = test_client.options("/health")

        # CORS should be configured
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers

    def test_invalid_json_handling(self, test_client):
        """Test handling of invalid JSON."""
        response = test_client.post(
            "/backtest",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        # Should return 422 for invalid JSON
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test client context manager."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            async with BacktestingServiceClient() as client:
                # Client should be usable within context
                assert client._session is None  # Not created yet

                # Trigger session creation
                with patch.object(client, '_get_session') as mock_get_session:
                    mock_get_session.return_value.__aenter__.return_value = mock_session
                    mock_get_session.return_value.__aexit__.return_value = None

                    # This would normally create a session
                    # await client.get_service_health()

            # Session should be closed after context exit
            mock_session.close.assert_called_once()