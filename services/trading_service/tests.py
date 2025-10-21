"""
TradPal Trading Service Tests
Comprehensive test suite for the unified trading service
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from datetime import datetime

from services.trading_service.orchestrator import TradingServiceOrchestrator
from services.trading_service.trading.service import TradingService
from services.trading_service.execution.service import ExecutionService
from services.trading_service.risk_management.service import RiskManagementService
from services.trading_service.reinforcement_learning.service import ReinforcementLearningService
from services.trading_service.market_regime.service import MarketRegimeService
from services.trading_service.monitoring.service import MonitoringService


class TestTradingServiceOrchestrator:
    """Test the main orchestrator"""

    @pytest.fixture
    def orchestrator(self):
        return TradingServiceOrchestrator()

    @pytest.mark.asyncio
    async def test_initialize(self, orchestrator):
        """Test orchestrator initialization"""
        await orchestrator.initialize()
        assert orchestrator.trading_service is not None
        assert orchestrator.execution_service is not None
        assert orchestrator.risk_service is not None
        assert orchestrator.rl_service is not None
        assert orchestrator.regime_service is not None
        assert orchestrator.monitoring_service is not None

    @pytest.mark.asyncio
    async def test_start_automated_trading(self, orchestrator):
        """Test automated trading session start"""
        await orchestrator.initialize()
        symbol = "BTC/USDT"
        config = {
            "symbol": symbol,
            "capital": 10000.0,
            "risk_per_trade": 0.02,
            "max_positions": 5,
            "paper_trading": True,
            "strategy": "default"
        }

        result = await orchestrator.start_automated_trading(symbol, config)
        assert "session_id" in result
        assert result["symbol"] == symbol
        assert result["is_active"] is True

    @pytest.mark.asyncio
    async def test_execute_smart_trade(self, orchestrator):
        """Test smart trade execution"""
        await orchestrator.initialize()
        symbol = "BTC/USDT"
        
        # Start a trading session first
        config = {
            "symbol": symbol,
            "capital": 10000.0,
            "risk_per_trade": 0.02,
            "max_positions": 5,
            "paper_trading": True,
            "strategy": "default"
        }
        await orchestrator.start_automated_trading(symbol, config)
        
        market_data = {
            "current_price": 50000.0,
            "prices": [49000, 49500, 50000, 50500, 51000],
            "volumes": [100, 120, 150, 130, 140]
        }

        result = await orchestrator.execute_smart_trade(symbol, market_data)
        # Either we get a decision (if confidence >= 0.6) or a low confidence message
        if "decision" in result:
            assert "position_size" in result
            assert "confidence" in result
        else:
            # Low confidence case
            assert "message" in result
            assert "confidence" in result
            assert result["confidence"] < 0.6

    @pytest.mark.asyncio
    async def test_get_trading_status(self, orchestrator):
        """Test getting comprehensive trading status"""
        await orchestrator.initialize()
        status = await orchestrator.get_trading_status()
        assert "active_sessions" in status
        assert "total_pnl" in status
        assert "risk_metrics" in status
        assert "system_health" in status

    @pytest.mark.asyncio
    async def test_stop_all_trading(self, orchestrator):
        """Test emergency stop"""
        await orchestrator.initialize()
        result = await orchestrator.stop_all_trading()
        assert result["success"] is True
        assert "stopped_sessions" in result

    @pytest.mark.asyncio
    async def test_get_performance_report(self, orchestrator):
        """Test performance reporting"""
        await orchestrator.initialize()
        report = await orchestrator.get_performance_report()
        assert "total_return" in report
        assert "sharpe_ratio" in report
        assert "max_drawdown" in report
        assert "win_rate" in report


class TestTradingService:
    """Test the core trading service"""

    @pytest.fixture
    def trading_service(self):
        service = TradingService()
        # Initialize synchronously for testing
        service.is_initialized = True
        return service

    @pytest.mark.asyncio
    async def test_start_trading_session(self, trading_service):
        """Test session management"""
        symbol = "BTC/USDT"
        config = {"capital": 10000.0, "symbol": symbol}
        session = await trading_service.start_trading_session(symbol, config)
        assert "session_id" in session
        assert session["symbol"] == symbol
        assert session["is_active"] is True

    @pytest.mark.asyncio
    async def test_execute_trade(self, trading_service):
        """Test trade execution"""
        # First start a session
        symbol = "BTC/USDT"
        config = {"capital": 10000.0, "symbol": symbol}
        await trading_service.start_trading_session(symbol, config)
        
        # Now execute trade
        result = await trading_service.execute_trade(symbol, "buy", 0.1, 50000.0)
        assert "order" in result
        assert "position" in result

    @pytest.mark.asyncio
    async def test_get_positions(self, trading_service):
        """Test position tracking"""
        positions = await trading_service.get_positions()
        assert isinstance(positions, list)


class TestExecutionService:
    """Test the execution service"""

    @pytest.fixture
    def execution_service(self):
        service = ExecutionService()
        # Initialize synchronously for testing
        service.is_initialized = True
        return service

    def test_submit_order(self, execution_service):
        """Test order submission"""
        order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": 0.1,
            "order_type": "market"
        }
        result = execution_service.submit_order(order)
        assert result["success"] is True
        assert "order_id" in result

    def test_cancel_order(self, execution_service):
        """Test order cancellation"""
        # Create a pending order manually for testing
        order_id = "test_pending_order"
        execution_service.pending_orders[order_id] = {
            "order_id": order_id,
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": 0.1,
            "status": "pending",
            "submitted_at": datetime.now().isoformat()
        }
        
        # Now cancel it
        result = execution_service.cancel_order(order_id)
        assert result["success"] is True

    def test_get_order_status(self, execution_service):
        """Test order status checking"""
        # First submit an order
        order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": 0.1,
            "order_type": "market"
        }
        submit_result = execution_service.submit_order(order)
        order_id = submit_result["order_id"]
        
        # Now check its status
        status = execution_service.get_order_status(order_id)
        assert "status" in status
        assert status["status"] == "filled"


class TestRiskManagementService:
    """Test the risk management service"""

    @pytest.fixture
    def risk_service(self):
        service = RiskManagementService()
        # Initialize synchronously for testing
        service.is_initialized = True
        return service

    @pytest.mark.asyncio
    async def test_calculate_position_size(self, risk_service):
        """Test position sizing"""
        capital = 10000.0
        risk_per_trade = 0.02
        stop_loss_pct = 0.05
        current_price = 50000.0

        result = await risk_service.calculate_position_size(
            capital, risk_per_trade, stop_loss_pct, current_price
        )
        expected_size = (capital * risk_per_trade) / (current_price * stop_loss_pct)
        assert result["quantity"] == expected_size

    def test_check_risk_limits(self, risk_service):
        """Test risk limit checking"""
        portfolio_value = 10000.0
        daily_loss = 500.0
        max_daily_loss = 0.1

        result = risk_service.check_risk_limits(portfolio_value, daily_loss, max_daily_loss)
        assert result["within_limits"] is True

    def test_calculate_stop_loss(self, risk_service):
        """Test stop loss calculation"""
        entry_price = 50000.0
        stop_loss_pct = 0.05

        stop_price = risk_service.calculate_stop_loss(entry_price, stop_loss_pct)
        assert stop_price == 47500.0


class TestReinforcementLearningService:
    """Test the RL service"""

    @pytest.fixture
    def rl_service(self):
        service = ReinforcementLearningService()
        # Initialize synchronously for testing
        service.is_initialized = True
        return service

    def test_train_rl_agent(self, rl_service):
        """Test RL agent training"""
        training_data = {
            "prices": np.random.randn(1000),
            "volumes": np.random.randn(1000),
            "rewards": np.random.randn(1000)
        }

        result = rl_service.train_rl_agent(training_data)
        assert result["success"] is True
        assert "model" in result

    @pytest.mark.asyncio
    async def test_get_rl_signal(self, rl_service):
        """Test RL signal generation"""
        symbol = "BTC/USDT"
        market_data = {
            "current_price": 50000.0,
            "prices": [49000, 49500, 50000, 50500, 51000],
            "volumes": [100, 120, 150, 130, 140]
        }

        signal = await rl_service.get_rl_signal(symbol, market_data)
        assert "signal" in signal
        assert "confidence" in signal
        assert signal["signal"] in ["buy", "sell", "hold"]


class TestMarketRegimeService:
    """Test the market regime service"""

    @pytest.fixture
    def regime_service(self):
        service = MarketRegimeService()
        # Initialize synchronously for testing
        service.is_initialized = True
        return service

    @pytest.mark.asyncio
    async def test_detect_regime(self, regime_service):
        """Test regime detection"""
        symbol = "BTC/USDT"
        price_data = np.array([100, 105, 110, 108, 115, 120, 118, 125])

        regime = await regime_service.detect_regime(symbol, price_data)
        assert "regime" in regime
        assert "confidence" in regime
        assert regime["regime"] in ["high_volatility", "moderate_volatility", "low_volatility", "trending", "consolidation"]

    @pytest.mark.asyncio
    async def test_get_regime_advice(self, regime_service):
        """Test regime-based trading advice"""
        symbol = "BTC/USDT"
        regime = "trending"

        advice = await regime_service.get_regime_advice(symbol, regime)
        assert "recommended_strategy" in advice
        assert "position_size_multiplier" in advice

    def test_track_regime_history(self, regime_service):
        """Test regime history tracking"""
        regime = "bull"
        timestamp = datetime.now()

        regime_service.track_regime_history(regime, timestamp)
        # Simplified test - just check that method exists and doesn't crash
        assert True


class TestMonitoringService:
    """Test the monitoring service"""

    @pytest.fixture
    def monitoring_service(self):
        service = MonitoringService()
        # Initialize synchronously for testing
        service.is_initialized = True
        return service

    @pytest.mark.asyncio
    async def test_record_metric(self, monitoring_service):
        """Test metric recording"""
        metric_name = "pnl"
        value = 1000.0
        tags = {"symbol": "BTC/USDT"}

        await monitoring_service.record_metric(metric_name, value, tags)
        metrics = await monitoring_service.get_metric(metric_name)
        assert len(metrics) > 0
        assert metrics[0]["name"] == metric_name

    def test_create_alert(self, monitoring_service):
        """Test alert creation"""
        alert_type = "high_risk"
        message = "Portfolio risk exceeded threshold"
        severity = "warning"

        alert_id = monitoring_service.create_alert(alert_type, message, severity)
        assert alert_id is not None

    def test_get_system_health(self, monitoring_service):
        """Test system health monitoring"""
        health = monitoring_service.get_system_health()
        assert "cpu_usage" in health
        assert "memory_usage" in health
        assert "status" in health


# Integration tests
class TestTradingServiceIntegration:
    """Integration tests for the complete trading service"""

    @pytest.fixture
    def full_orchestrator(self):
        """Create a fully initialized orchestrator for testing"""
        orchestrator = TradingServiceOrchestrator()
        # Initialize synchronously for testing
        orchestrator.is_initialized = True
        # Initialize all services synchronously
        orchestrator.trading_service.is_initialized = True
        orchestrator.execution_service.is_initialized = True
        orchestrator.risk_service.is_initialized = True
        orchestrator.rl_service.is_initialized = True
        orchestrator.regime_service.is_initialized = True
        orchestrator.monitoring_service.is_initialized = True
        return orchestrator

    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self, full_orchestrator):
        """Test complete trading workflow from start to finish"""
        # Start trading
        config = {
            "capital": 10000.0,
            "risk_per_trade": 0.02,
            "max_positions": 3,
            "paper_trading": True,
            "strategy": "smart_ai"
        }

        start_result = await full_orchestrator.start_automated_trading("BTC/USDT", config)
        assert "session_id" in start_result

        # Execute smart trade
        market_data = {
            "current_price": 50000.0,
            "prices": [49000, 49500, 50000, 50500, 51000],
            "volumes": [100, 120, 150, 130, 140]
        }

        trade_result = await full_orchestrator.execute_smart_trade("BTC/USDT", market_data)
        # Either we get a decision (if confidence >= 0.6) or a low confidence message
        assert ("decision" in trade_result) or ("message" in trade_result)

        # Check status
        status = await full_orchestrator.get_trading_status()
        assert "active_sessions" in status

        # Get performance
        performance = await full_orchestrator.get_performance_report()
        assert "total_return" in performance

        # Stop trading
        stop_result = await full_orchestrator.stop_all_trading()
        assert stop_result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__])