#!/usr/bin/env python3
"""
Risk Service Tests

Comprehensive tests for the Risk Service including:
- Unit tests for position sizing and risk calculations
- Integration tests with portfolio management
- Performance tests for risk metrics
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from services.risk_service.service import (
    RiskService, RiskRequest, RiskParameters, PositionSizing,
    RiskAssessment, RiskLevel, PositionType, RiskMetric, EventSystem
)


class TestRiskService:
    """Unit tests for RiskService."""

    @pytest.fixture
    def event_system(self):
        """Create event system for testing."""
        return EventSystem()

    @pytest.fixture
    def risk_service(self, event_system):
        """Create risk service instance."""
        return RiskService(event_system=event_system)

    def test_initialization(self, risk_service):
        """Test service initialization."""
        assert risk_service is not None
        assert risk_service.event_system is not None
        assert isinstance(risk_service.default_params, RiskParameters)

    def test_kelly_criterion_calculation(self, risk_service):
        """Test Kelly Criterion position sizing."""
        # Test case: 60% win rate, 2:1 win/loss ratio
        win_rate = 0.6
        win_loss_ratio = 2.0
        risk_percentage = 0.02
        capital = 10000

        position_size = risk_service._calculate_position_size_kelly(
            win_rate, win_loss_ratio, risk_percentage, capital
        )

        # Kelly percentage = 0.6 - (1-0.6)/2 = 0.6 - 0.2 = 0.4
        expected_kelly = 0.4
        expected_size = capital * min(expected_kelly, risk_percentage)

        assert abs(position_size - expected_size) < 0.01

    def test_atr_stop_loss_calculation(self, risk_service):
        """Test ATR-based stop loss calculation."""
        entry_price = 50000
        atr_value = 1000
        multiplier = 1.5

        # Long position
        sl_long = risk_service._calculate_atr_stop_loss(
            entry_price, atr_value, PositionType.LONG, multiplier
        )
        assert sl_long == entry_price - (atr_value * multiplier)

        # Short position
        sl_short = risk_service._calculate_atr_stop_loss(
            entry_price, atr_value, PositionType.SHORT, multiplier
        )
        assert sl_short == entry_price + (atr_value * multiplier)

    def test_atr_take_profit_calculation(self, risk_service):
        """Test ATR-based take profit calculation."""
        entry_price = 50000
        atr_value = 1000
        multiplier = 3.0

        # Long position
        tp_long = risk_service._calculate_atr_take_profit(
            entry_price, atr_value, PositionType.LONG, multiplier
        )
        assert tp_long == entry_price + (atr_value * multiplier)

        # Short position
        tp_short = risk_service._calculate_atr_take_profit(
            entry_price, atr_value, PositionType.SHORT, multiplier
        )
        assert tp_short == entry_price - (atr_value * multiplier)

    def test_volatility_adjusted_leverage(self, risk_service):
        """Test volatility-adjusted leverage calculation."""
        base_leverage = 2.0
        volatility = 0.03  # 3% volatility
        mean_volatility = 0.02  # 2% mean

        leverage = risk_service._calculate_volatility_adjusted_leverage(
            volatility, base_leverage, mean_volatility
        )

        # Should reduce leverage due to higher volatility
        expected_factor = mean_volatility / volatility  # 0.02/0.03 = 0.667
        expected_leverage = base_leverage * expected_factor

        assert abs(leverage - expected_leverage) < 0.01
        assert leverage <= risk_service.default_params.max_leverage
        assert leverage >= risk_service.default_params.min_leverage

    def test_risk_metrics_calculation(self, risk_service):
        """Test comprehensive risk metrics calculation."""
        # Create sample returns data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)

        metrics = risk_service._calculate_risk_metrics(returns)

        # Check that all expected metrics are present
        expected_metrics = [metric.value for metric in RiskMetric]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

        # Sharpe ratio should be reasonable
        assert -5 <= metrics[RiskMetric.SHARPE_RATIO.value] <= 5

        # Max drawdown should be negative or zero
        assert metrics[RiskMetric.MAX_DRAWDOWN.value] <= 0

        # VaR should be negative (loss)
        assert metrics[RiskMetric.VALUE_AT_RISK.value] <= 0

    def test_risk_level_assessment(self, risk_service):
        """Test risk level assessment."""
        # Low risk scenario
        low_risk_metrics = {
            RiskMetric.SHARPE_RATIO.value: 2.0,
            RiskMetric.MAX_DRAWDOWN.value: -0.05,
            RiskMetric.VOLATILITY.value: 0.15,
            RiskMetric.VALUE_AT_RISK.value: -0.02
        }

        level, score, recommendations = risk_service._assess_risk_level(low_risk_metrics, 0.01)

        assert level in [RiskLevel.VERY_LOW, RiskLevel.LOW]
        assert score >= 0
        assert isinstance(recommendations, list)

        # High risk scenario
        high_risk_metrics = {
            RiskMetric.SHARPE_RATIO.value: -0.5,
            RiskMetric.MAX_DRAWDOWN.value: -0.25,
            RiskMetric.VOLATILITY.value: 0.60,
            RiskMetric.VALUE_AT_RISK.value: -0.10
        }

        level, score, recommendations = risk_service._assess_risk_level(high_risk_metrics, 0.05)

        assert level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]
        assert score > 0
        assert len(recommendations) > 0

    @pytest.mark.asyncio
    async def test_position_sizing_calculation(self, risk_service):
        """Test complete position sizing calculation."""
        request = RiskRequest(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
            position_type="long",
            atr_value=1000,
            volatility=0.03,
            risk_percentage=0.01
        )

        sizing = await risk_service.calculate_position_sizing(request)

        assert isinstance(sizing, PositionSizing)
        assert sizing.symbol == "BTC/USDT"
        assert sizing.position_size > 0
        assert sizing.risk_amount == request.capital * request.risk_percentage
        assert sizing.leverage >= 1.0
        assert sizing.stop_loss_price < request.entry_price  # Long position
        assert sizing.take_profit_price > request.entry_price  # Long position
        assert sizing.reward_risk_ratio > 1.0

        # Check that position was stored
        assert "BTC/USDT" in risk_service.portfolio_positions

    @pytest.mark.asyncio
    async def test_portfolio_risk_assessment(self, risk_service):
        """Test portfolio risk assessment."""
        # Create sample portfolio returns
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)

        assessment = await risk_service.assess_portfolio_risk(returns, "daily")

        assert isinstance(assessment, RiskAssessment)
        assert assessment.symbol == "PORTFOLIO"
        assert assessment.risk_level in RiskLevel
        assert assessment.risk_score >= 0
        assert isinstance(assessment.metrics, dict)
        assert isinstance(assessment.recommendations, list)

        # Check that assessment was stored in history
        assert len(risk_service.risk_history) > 0

    @pytest.mark.asyncio
    async def test_portfolio_exposure_calculation(self, risk_service):
        """Test portfolio exposure calculation."""
        # Add some test positions
        pos1 = PositionSizing(
            symbol="BTC/USDT",
            position_size=0.1,
            position_value=5000,
            risk_amount=50,
            stop_loss_price=49000,
            take_profit_price=51000,
            leverage=1.0,
            risk_percentage=0.01,
            reward_risk_ratio=2.0,
            calculated_at=datetime.now(),
            parameters={}
        )

        pos2 = PositionSizing(
            symbol="ETH/USDT",
            position_size=1.0,
            position_value=3000,
            risk_amount=30,
            stop_loss_price=2900,
            take_profit_price=3100,
            leverage=1.0,
            risk_percentage=0.01,
            reward_risk_ratio=2.0,
            calculated_at=datetime.now(),
            parameters={}
        )

        risk_service.portfolio_positions["BTC/USDT"] = pos1
        risk_service.portfolio_positions["ETH/USDT"] = pos2

        exposure = await risk_service.get_portfolio_exposure()

        assert exposure["total_positions"] == 2
        assert exposure["total_exposure"] == 8000  # 5000 + 3000
        assert exposure["total_risk"] == 80  # 50 + 30
        assert "positions" in exposure
        assert len(exposure["positions"]) == 2

    @pytest.mark.asyncio
    async def test_risk_parameters_update(self, risk_service):
        """Test risk parameters update."""
        original_max_risk = risk_service.default_params.max_risk_per_trade

        new_params = {"max_risk_per_trade": 0.05}
        await risk_service.update_risk_parameters(new_params)

        assert risk_service.default_params.max_risk_per_trade == 0.05

    @pytest.mark.asyncio
    async def test_health_check(self, risk_service):
        """Test health check functionality."""
        health = await risk_service.health_check()

        assert isinstance(health, dict)
        assert health["service"] == "risk_service"
        assert health["status"] == "healthy"
        assert "timestamp" in health
        assert "portfolio_positions" in health
        assert "parameters" in health


class TestRiskServiceIntegration:
    """Integration tests requiring external dependencies."""

    @pytest.fixture
    def risk_service(self):
        """Create risk service for integration tests."""
        event_system = EventSystem()
        return RiskService(event_system=event_system)

    @pytest.mark.asyncio
    async def test_end_to_end_position_sizing_workflow(self, risk_service):
        """Test complete position sizing workflow."""
        # Simulate a trading scenario
        capital = 10000
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]

        for symbol in symbols:
            # Mock market data
            entry_price = 50000 if symbol == "BTC/USDT" else 3000 if symbol == "ETH/USDT" else 0.5
            atr_value = entry_price * 0.02  # 2% ATR
            volatility = 0.03  # 3% volatility

            request = RiskRequest(
                symbol=symbol,
                capital=capital,
                entry_price=entry_price,
                position_type="long",
                atr_value=atr_value,
                volatility=volatility,
                risk_percentage=0.01
            )

            sizing = await risk_service.calculate_position_sizing(request)

            # Validate sizing
            assert sizing.position_size > 0
            assert sizing.risk_amount <= capital * request.risk_percentage
            assert sizing.leverage >= 1.0
            assert sizing.stop_loss_price < entry_price
            assert sizing.take_profit_price > entry_price

        # Check portfolio exposure
        exposure = await risk_service.get_portfolio_exposure()
        assert exposure["total_positions"] == len(symbols)
        assert exposure["total_risk"] <= capital * 0.03  # Max 3% total risk

    @pytest.mark.asyncio
    async def test_portfolio_risk_monitoring_workflow(self, risk_service):
        """Test portfolio risk monitoring workflow."""
        # Create a portfolio with multiple positions
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]

        for symbol in symbols:
            entry_price = 50000 if symbol == "BTC/USDT" else 3000 if symbol == "ETH/USDT" else 0.5
            request = RiskRequest(
                symbol=symbol,
                capital=10000,
                entry_price=entry_price,
                position_type="long",
                atr_value=entry_price * 0.02,
                risk_percentage=0.005  # 0.5% per position
            )

            await risk_service.calculate_position_sizing(request)

        # Generate synthetic portfolio returns
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        # Simulate realistic returns with some volatility
        returns = pd.Series(
            np.random.normal(0.001, 0.025, 252) +  # Base return
            np.sin(np.arange(252) * 0.1) * 0.005,    # Seasonal component
            index=dates
        )

        # Assess portfolio risk
        assessment = await risk_service.assess_portfolio_risk(returns)

        assert assessment.risk_level in RiskLevel
        assert len(assessment.recommendations) >= 0

        # Check risk history
        history = risk_service.risk_history
        assert len(history) > 0
        assert history[-1]["assessment"]["risk_level"] == assessment.risk_level.value


class TestRiskServicePerformance:
    """Performance tests for Risk Service."""

    @pytest.fixture
    def risk_service(self):
        """Create risk service for performance tests."""
        event_system = EventSystem()
        return RiskService(event_system=event_system)

    @pytest.mark.asyncio
    async def test_position_sizing_performance(self, risk_service):
        """Test position sizing calculation performance."""
        import time

        request = RiskRequest(
            symbol="BTC/USDT",
            capital=10000,
            entry_price=50000,
            position_type="long",
            atr_value=1000,
            volatility=0.03,
            risk_percentage=0.01
        )

        # Measure performance
        start_time = time.time()
        result = await risk_service.calculate_position_sizing(request)
        end_time = time.time()

        duration = end_time - start_time

        assert result is not None
        assert result.position_size > 0
        assert duration < 1.0  # Should complete within 1 second

    @pytest.mark.asyncio
    async def test_portfolio_assessment_performance(self, risk_service):
        """Test portfolio risk assessment performance."""
        import time

        # Create large returns dataset
        dates = pd.date_range('2024-01-01', periods=1000, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000), index=dates)

        # Measure performance
        start_time = time.time()
        result = await risk_service.assess_portfolio_risk(returns)
        end_time = time.time()

        duration = end_time - start_time

        assert result is not None
        assert result.risk_level in RiskLevel
        assert duration < 5.0  # Should complete within 5 seconds

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_position_sizing(self, risk_service):
        """Test concurrent position sizing calculations."""
        requests = []
        for i in range(10):
            request = RiskRequest(
                symbol=f"ASSET_{i}",
                capital=10000,
                entry_price=100 + i * 10,
                position_type="long",
                atr_value=2.0,
                volatility=0.03,
                risk_percentage=0.01
            )
            requests.append(request)

        # Execute concurrently
        start_time = asyncio.get_event_loop().time()
        tasks = [risk_service.calculate_position_sizing(req) for req in requests]
        results = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()

        total_time = end_time - start_time

        assert len(results) == 10
        assert all(isinstance(r, PositionSizing) for r in results)
        assert total_time < 5.0  # Should complete within 5 seconds

        print(f"Concurrent sizing time: {total_time:.2f}s")


# Mock classes for testing
class MockEventSystem:
    """Mock event system for testing."""

    def __init__(self):
        self.events = []

    async def publish(self, event_type: str, data: dict):
        self.events.append((event_type, data))


@pytest.fixture
def mock_event_system():
    """Mock event system fixture."""
    return MockEventSystem()


@pytest.mark.asyncio
async def test_with_mock_event_system(mock_event_system):
    """Test risk service with mock event system."""
    service = RiskService(event_system=mock_event_system)

    request = RiskRequest(
        symbol="BTC/USDT",
        capital=10000,
        entry_price=50000,
        position_type="long",
        atr_value=1000,
        risk_percentage=0.01
    )

    await service.calculate_position_sizing(request)

    # Check that events were published
    assert len(mock_event_system.events) > 0
    assert any("risk.position_sized" in event[0] for event in mock_event_system.events)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])