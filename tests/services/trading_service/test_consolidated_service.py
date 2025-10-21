"""
Tests for consolidated TradingService with integrated execution and risk management
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from services.trading_service.trading.service import TradingService


class TestConsolidatedTradingService:
    """Test cases for consolidated TradingService"""

    @pytest.fixture
    def trading_service(self):
        """Create a TradingService instance for testing"""
        return TradingService()

    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self, trading_service):
        """Test service initialization and shutdown"""
        assert not trading_service.is_initialized

        await trading_service.initialize()
        assert trading_service.is_initialized

        await trading_service.shutdown()
        assert not trading_service.is_initialized

    @pytest.mark.asyncio
    async def test_start_trading_session(self, trading_service):
        """Test starting a trading session"""
        await trading_service.initialize()

        config = {
            "capital": 10000.0,
            "risk_per_trade": 0.02,
            "paper_trading": True
        }

        result = await trading_service.start_trading_session("BTC/USDT", config)

        assert "session_id" in result
        assert result["symbol"] == "BTC/USDT"
        assert result["capital"] == 10000.0
        assert result["paper_trading"] is True

        await trading_service.shutdown()

    @pytest.mark.asyncio
    async def test_calculate_position_size(self, trading_service):
        """Test position size calculation (from RiskManagementService)"""
        await trading_service.initialize()

        result = await trading_service.calculate_position_size(
            capital=10000.0,
            risk_per_trade=0.02,
            stop_loss_pct=0.02,
            current_price=50000.0
        )

        expected_quantity = (10000.0 * 0.02) / (50000.0 * 0.02)  # 0.2
        assert result["quantity"] == expected_quantity

        await trading_service.shutdown()

    def test_check_risk_limits(self, trading_service):
        """Test risk limit checking (from RiskManagementService)"""
        trading_service.is_initialized = True  # Mock initialization

        result = trading_service.check_risk_limits(
            portfolio_value=10000.0,
            daily_loss=150.0,
            max_daily_loss=0.05
        )

        assert result["within_limits"] is True
        assert result["current_loss"] == 150.0
        assert result["max_allowed_loss"] == 500.0

    def test_calculate_stop_loss(self, trading_service):
        """Test stop loss calculation (from RiskManagementService)"""
        trading_service.is_initialized = True  # Mock initialization

        stop_loss = trading_service.calculate_stop_loss(
            entry_price=50000.0,
            stop_loss_pct=0.02
        )

        assert stop_loss == 49000.0  # 2% below entry

    @pytest.mark.asyncio
    async def test_calculate_take_profit(self, trading_service):
        """Test take profit calculation (from RiskManagementService)"""
        await trading_service.initialize()

        take_profit = await trading_service.calculate_take_profit(
            entry_price=50000.0,
            stop_loss_price=49000.0,
            reward_risk_ratio=2.0
        )

        risk_amount = 50000.0 - 49000.0  # 1000
        expected_take_profit = 50000.0 + (risk_amount * 2.0)  # 52000
        assert take_profit == expected_take_profit

        await trading_service.shutdown()

    @pytest.mark.asyncio
    async def test_get_portfolio_risk_metrics(self, trading_service):
        """Test portfolio risk metrics calculation (from RiskManagementService)"""
        await trading_service.initialize()

        # Sample returns: +1%, -0.5%, +2%, -1%
        returns = [0.01, -0.005, 0.02, -0.01]

        result = await trading_service.get_portfolio_risk_metrics(returns)

        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "volatility" in result
        assert "total_return" in result

        # Total return should be approximately (1+0.01)*(1-0.005)*(1+0.02)*(1-0.01) - 1
        expected_total_return = (1.01 * 0.995 * 1.02 * 0.99) - 1
        assert abs(result["total_return"] - expected_total_return) < 0.0001

        await trading_service.shutdown()

    @pytest.mark.asyncio
    async def test_submit_order(self, trading_service):
        """Test order submission (from ExecutionService)"""
        await trading_service.initialize()

        # Start a session first
        config = {"capital": 10000.0, "paper_trading": True}
        await trading_service.start_trading_session("BTC/USDT", config)

        order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": 0.1,
            "price": 50000.0
        }

        result = trading_service.submit_order(order)

        assert result["success"] is True
        assert "order_id" in result

        # Get the order details using the order_id
        order_details = trading_service.get_order_status(result["order_id"])

        assert order_details["symbol"] == "BTC/USDT"
        assert order_details["side"] == "buy"
        assert order_details["quantity"] == 0.1
        assert order_details["price"] == 50000.0
        assert order_details["status"] == "filled"

        await trading_service.shutdown()

    @pytest.mark.asyncio
    async def test_get_execution_stats(self, trading_service):
        """Test execution statistics (from ExecutionService)"""
        await trading_service.initialize()

        stats = await trading_service.get_execution_stats()

        assert "total_orders" in stats
        assert "filled_orders" in stats
        assert "pending_orders" in stats
        assert "avg_execution_time" in stats

        await trading_service.shutdown()

    def test_get_default_risk_config(self, trading_service):
        """Test default risk configuration"""
        trading_service.is_initialized = True  # Mock initialization

        config = trading_service.get_default_risk_config()

        assert "max_risk_per_trade" in config
        assert "max_portfolio_risk" in config
        assert "max_drawdown" in config
        assert config["max_risk_per_trade"] == 0.02
        assert config["max_portfolio_risk"] == 0.05
        assert config["max_drawdown"] == 0.1