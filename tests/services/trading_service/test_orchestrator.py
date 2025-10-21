"""
Tests for consolidated TradingServiceOrchestrator
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from services.trading_service.orchestrator import TradingServiceOrchestrator


class TestConsolidatedTradingServiceOrchestrator:
    """Test cases for consolidated TradingServiceOrchestrator"""

    @pytest.fixture
    def orchestrator(self):
        """Create a TradingServiceOrchestrator instance for testing"""
        return TradingServiceOrchestrator()

    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self, orchestrator):
        """Test orchestrator initialization and shutdown"""
        assert not orchestrator.is_initialized

        await orchestrator.initialize()
        assert orchestrator.is_initialized

        await orchestrator.shutdown()
        assert not orchestrator.is_initialized

    @pytest.mark.asyncio
    async def test_start_automated_trading(self, orchestrator):
        """Test starting automated trading"""
        await orchestrator.initialize()

        config = {
            "capital": 10000.0,
            "risk_per_trade": 0.02,
            "paper_trading": True
        }

        result = await orchestrator.start_automated_trading("BTC/USDT", config)

        assert "session_id" in result
        assert result["symbol"] == "BTC/USDT"
        assert result["capital"] == 10000.0
        assert result["paper_trading"] is True

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_execute_smart_trade(self, orchestrator):
        """Test smart trade execution with consolidated services"""
        await orchestrator.initialize()

        # Start a session first
        config = {
            "capital": 10000.0,
            "risk_per_trade": 0.02,
            "paper_trading": True
        }
        await orchestrator.start_automated_trading("BTC/USDT", config)

        # Mock market data
        market_data = {
            "current_price": 50000.0,
            "prices": [49000.0, 49500.0, 50000.0, 50500.0, 51000.0]
        }

        # Mock the RL service to return a signal
        orchestrator.rl_service.get_rl_signal = AsyncMock(return_value={
            "signal": "buy",
            "confidence": 0.8
        })

        # Mock regime service
        orchestrator.regime_service.detect_regime = AsyncMock(return_value={
            "regime": "bull",
            "confidence": 0.9
        })
        orchestrator.regime_service.get_regime_advice = AsyncMock(return_value={
            "position_size_multiplier": 1.0
        })

        result = await orchestrator.execute_smart_trade("BTC/USDT", market_data)

        # Should execute trade since confidence > 0.6
        assert "decision" in result
        assert result["decision"] == "buy"
        assert "position_size" in result
        assert "confidence" in result
        assert result["confidence"] == 0.8

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_get_trading_status(self, orchestrator):
        """Test getting comprehensive trading status"""
        await orchestrator.initialize()

        status = await orchestrator.get_trading_status()

        assert "active_sessions" in status
        assert "total_pnl" in status
        assert "risk_metrics" in status
        assert "system_health" in status
        assert "execution_stats" in status
        assert "active_alerts" in status
        assert "timestamp" in status

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_get_performance_report(self, orchestrator):
        """Test performance report generation"""
        await orchestrator.initialize()

        report = await orchestrator.get_performance_report()

        assert "total_return" in report
        assert "sharpe_ratio" in report
        assert "max_drawdown" in report
        assert "win_rate" in report
        assert "total_trades" in report
        assert "active_sessions" in report
        assert "system_metrics" in report
        assert "timestamp" in report

        await orchestrator.shutdown()

    def test_get_service_status(self, orchestrator):
        """Test service status reporting"""
        orchestrator.is_initialized = True  # Mock initialization
        orchestrator.trading_service.is_initialized = True
        orchestrator.rl_service.is_initialized = True
        orchestrator.regime_service.is_initialized = True
        orchestrator.monitoring_service.is_initialized = True

        status = orchestrator.get_service_status()

        assert status["orchestrator_initialized"] is True
        assert status["trading_service"] is True
        assert status["rl_service"] is True
        assert status["regime_service"] is True
        assert status["monitoring_service"] is True
        assert "timestamp" in status

    def test_get_default_trading_config(self, orchestrator):
        """Test default trading configuration"""
        config = orchestrator.get_default_trading_config()

        assert config["capital"] == 10000.0
        assert config["risk_per_trade"] == 0.02
        assert config["max_positions"] == 5
        assert config["paper_trading"] is True
        assert config["strategy"] == "smart_ai"
        assert config["rl_enabled"] is True
        assert config["regime_detection"] is True
        assert config["monitoring"] is True