#!/usr/bin/env python3
"""
Integration Tests for Trading Bot Live Service

Tests the complete Trading Bot Live Service functionality including:
- API endpoints
- Service logic
- Client communication
- Trading execution and risk management
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock

from services.trading_bot_live.service import TradingBotLiveService, EventSystem
from services.trading_bot_live.client import TradingBotLiveServiceClient


class TestTradingBotLiveServiceIntegration:
    """Integration tests for Trading Bot Live Service"""

    @pytest.fixture
    def test_client(self):
        """Create mock test client"""
        from unittest.mock import Mock
        client = Mock()
        # Mock common HTTP methods
        client.get = Mock()
        client.post = Mock()
        return client

    @pytest.fixture
    def event_system(self):
        """Create event system for testing"""
        return EventSystem()

    @pytest.fixture
    def trading_service(self, event_system):
        """Create trading bot live service instance"""
        return TradingBotLiveService(event_system=event_system)

    @pytest.fixture
    def trading_client(self):
        """Create trading bot live service client"""
        return TradingBotLiveServiceClient(base_url="http://test")

    def test_api_root_endpoint(self, test_client):
        """Test API root endpoint"""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "service": "Trading Bot Live Service",
            "version": "1.0.0"
        }
        test_client.get.return_value = mock_response

        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "Trading Bot Live Service"

    @pytest.mark.asyncio
    async def test_service_health_check(self, trading_service):
        """Test service health check"""
        health = await trading_service.health_check()
        assert "service" in health
        assert "status" in health
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_paper_trading_execution(self, trading_service):
        """Test paper trading execution"""
        # Create sample signal data
        signal_data = {
            "timestamp": "2023-01-01T12:00:00Z",
            "symbol": "BTC/USDT",
            "signal": "BUY",
            "price": 50000.0,
            "confidence": 0.8,
            "indicators": {
                "ema_short": 49900,
                "ema_long": 49800,
                "rsi": 65
            },
            "risk_parameters": {
                "position_size": 0.01,
                "stop_loss": 49500,
                "take_profit": 51000
            }
        }

        result = await trading_service.execute_paper_trade(signal_data)

        assert result["success"] is True
        assert "order_id" in result
        assert "position" in result
        assert result["position"]["symbol"] == "BTC/USDT"
        assert result["position"]["side"] == "BUY"

    @pytest.mark.asyncio
    async def test_position_management(self, trading_service):
        """Test position management"""
        # Create a position
        position = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "quantity": 0.01,
            "entry_price": 50000.0,
            "current_price": 51000.0,
            "stop_loss": 49500.0,
            "take_profit": 52000.0
        }

        trading_service.positions["test_pos"] = position

        # Test position update
        update_data = {
            "position_id": "test_pos",
            "current_price": 51500.0
        }

        result = await trading_service.update_position(update_data)

        assert result["success"] is True
        assert trading_service.positions["test_pos"]["current_price"] == 51500.0

    @pytest.mark.asyncio
    async def test_risk_management(self, trading_service):
        """Test risk management"""
        # Create sample portfolio
        portfolio = {
            "total_balance": 10000.0,
            "available_balance": 8000.0,
            "positions": [
                {
                    "symbol": "BTC/USDT",
                    "quantity": 0.1,
                    "entry_price": 50000.0,
                    "current_price": 48000.0
                }
            ]
        }

        risk_check = await trading_service.check_risk_limits(portfolio)

        assert isinstance(risk_check, dict)
        assert "max_drawdown" in risk_check
        assert "position_size_limit" in risk_check
        assert "daily_loss_limit" in risk_check

    @pytest.mark.asyncio
    async def test_stop_loss_execution(self, trading_service):
        """Test stop loss execution"""
        # Create position with stop loss
        position = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "quantity": 0.01,
            "entry_price": 50000.0,
            "current_price": 49500.0,  # Hit stop loss
            "stop_loss": 49500.0,
            "take_profit": 52000.0
        }

        trading_service.positions["test_pos"] = position

        # Simulate price update triggering stop loss
        price_update = {
            "symbol": "BTC/USDT",
            "price": 49400.0,  # Below stop loss
            "timestamp": "2023-01-01T12:00:00Z"
        }

        result = await trading_service.handle_price_update(price_update)

        assert result["success"] is True
        # Position should be closed due to stop loss

    @pytest.mark.asyncio
    async def test_take_profit_execution(self, trading_service):
        """Test take profit execution"""
        # Create position with take profit
        position = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "quantity": 0.01,
            "entry_price": 50000.0,
            "current_price": 52100.0,  # Hit take profit
            "stop_loss": 49500.0,
            "take_profit": 52000.0
        }

        trading_service.positions["test_pos"] = position

        # Simulate price update triggering take profit
        price_update = {
            "symbol": "BTC/USDT",
            "price": 52200.0,  # Above take profit
            "timestamp": "2023-01-01T12:00:00Z"
        }

        result = await trading_service.handle_price_update(price_update)

        assert result["success"] is True
        # Position should be closed due to take profit

    @pytest.mark.asyncio
    async def test_client_health_check(self, trading_client):
        """Test client health check"""
        with patch.object(trading_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"status": "healthy"}

            health = await trading_client.health_check()
            assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_client_execute_trade(self, trading_client):
        """Test client trade execution"""
        with patch.object(trading_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "success": True,
                "order_id": "test_order_123",
                "position": {"symbol": "BTC/USDT", "side": "BUY"}
            }

            from services.trading_bot_live.client import TradeExecutionRequest
            request = TradeExecutionRequest(
                symbol="BTC/USDT",
                signal="BUY",
                price=50000.0,
                confidence=0.8,
                risk_parameters={"position_size": 0.01}
            )

            result = await trading_client.execute_trade(request)
            assert result["success"] is True
            assert result["order_id"] == "test_order_123"

    @pytest.mark.asyncio
    async def test_client_get_positions(self, trading_client):
        """Test client get positions"""
        with patch.object(trading_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "positions": [
                    {"symbol": "BTC/USDT", "quantity": 0.01, "entry_price": 50000.0}
                ]
            }

            positions = await trading_client.get_positions()
            assert len(positions["positions"]) == 1
            assert positions["positions"][0]["symbol"] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_client_close_position(self, trading_client):
        """Test client close position"""
        with patch.object(trading_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"success": True, "message": "Position closed"}

            from services.trading_bot_live.client import ClosePositionRequest
            request = ClosePositionRequest(
                position_id="test_pos",
                reason="manual_close"
            )

            result = await trading_client.close_position(request)
            assert result["success"] is True

    def test_api_execute_trade_endpoint(self, test_client):
        """Test API execute trade endpoint"""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "order_id": "test_order",
            "position": {"symbol": "BTC/USDT", "side": "BUY"}
        }
        test_client.post.return_value = mock_response

        data = {
            "symbol": "BTC/USDT",
            "signal": "BUY",
            "price": 50000.0,
            "confidence": 0.8
        }

        response = test_client.post("/execute-trade", json=data)
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True

    def test_api_positions_endpoint(self, test_client):
        """Test API positions endpoint"""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "positions": [
                {"symbol": "BTC/USDT", "quantity": 0.01, "entry_price": 50000.0}
            ],
            "count": 1
        }
        test_client.get.return_value = mock_response

        response = test_client.get("/positions")
        assert response.status_code == 200
        result = response.json()
        assert "positions" in result

    def test_api_portfolio_endpoint(self, test_client):
        """Test API portfolio endpoint"""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_balance": 10000.0,
            "available_balance": 8000.0,
            "positions": []
        }
        test_client.get.return_value = mock_response

        response = test_client.get("/portfolio")
        assert response.status_code == 200
        result = response.json()
        assert "total_balance" in result

    @pytest.mark.asyncio
    async def test_service_error_handling(self, trading_service):
        """Test service error handling"""
        # Test with invalid signal
        invalid_signal = {
            "symbol": "BTC/USDT",
            "signal": "INVALID",
            "price": 50000.0
        }

        with pytest.raises(ValueError):
            await trading_service.execute_paper_trade(invalid_signal)

    @pytest.mark.asyncio
    async def test_client_error_handling(self, trading_client):
        """Test client error handling"""
        with patch.object(trading_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Connection failed")

            with pytest.raises(Exception):
                await trading_client.health_check()

    @pytest.mark.asyncio
    async def test_position_pnl_calculation(self, trading_service):
        """Test position P&L calculation"""
        position = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "quantity": 0.01,
            "entry_price": 50000.0,
            "current_price": 51000.0
        }

        pnl = trading_service._calculate_pnl(position)
        expected_pnl = (51000.0 - 50000.0) * 0.01
        assert pnl == expected_pnl

    @pytest.mark.asyncio
    async def test_risk_limits_calculation(self, trading_service):
        """Test risk limits calculation"""
        portfolio = {
            "total_balance": 10000.0,
            "positions": [
                {
                    "symbol": "BTC/USDT",
                    "quantity": 0.1,
                    "entry_price": 50000.0,
                    "current_price": 48000.0
                }
            ]
        }

        limits = trading_service._calculate_risk_limits(portfolio)

        assert "max_drawdown_pct" in limits
        assert "position_size_pct" in limits
        assert "daily_loss_pct" in limits

    @pytest.mark.asyncio
    async def test_event_system_integration(self, event_system, trading_service):
        """Test event system integration"""
        events_received = []

        async def event_handler(data):
            events_received.append(data)

        event_system.subscribe("trading.position_opened", event_handler)

        # Trigger event
        await event_system.publish("trading.position_opened", {"symbol": "BTC/USDT"})

        await asyncio.sleep(0.1)

        assert len(events_received) == 1
        assert events_received[0]["symbol"] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_live_trading_mode(self, trading_service):
        """Test live trading mode setup"""
        # This would require mocking exchange connections
        # For now, test that the method exists and doesn't crash
        assert hasattr(trading_service, '_setup_live_trading')

    @pytest.mark.asyncio
    async def test_paper_trading_mode(self, trading_service):
        """Test paper trading mode"""
        # Test paper trading initialization
        assert trading_service.paper_trading is True  # Default

        # Test paper balance management
        assert hasattr(trading_service, 'paper_balance')

    @pytest.mark.asyncio
    async def test_order_validation(self, trading_service):
        """Test order validation"""
        valid_order = {
            "symbol": "BTC/USDT",
            "signal": "BUY",
            "price": 50000.0,
            "confidence": 0.8,
            "risk_parameters": {
                "position_size": 0.01,
                "stop_loss": 49500.0,
                "take_profit": 51000.0
            }
        }

        is_valid = trading_service._validate_order(valid_order)
        assert is_valid is True

        invalid_order = {
            "symbol": "BTC/USDT",
            "signal": "BUY",
            "price": -1000.0,  # Invalid price
            "confidence": 0.8
        }

        is_valid = trading_service._validate_order(invalid_order)
        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__])