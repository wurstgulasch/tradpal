"""
Tests for Portfolio Management Module

Tests the advanced portfolio management functionality including:
- Multi-asset portfolio creation and management
- Risk-based allocation methods
- Rebalancing logic
- Performance metrics calculation
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.portfolio_manager import (
    PortfolioManager,
    AllocationMethod,
    RebalancingFrequency,
    PortfolioConstraints
)


class TestPortfolioManager:
    """Test suite for PortfolioManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache_manager = Mock()
        self.manager = PortfolioManager(cache_manager=self.cache_manager)

    def test_initialization(self):
        """Test PortfolioManager initialization."""
        assert self.manager.portfolios == {}
        assert len(self.manager.asset_universe) > 0
        assert 'BTC/USDT' in self.manager.asset_universe

    def test_create_portfolio(self):
        """Test portfolio creation."""
        assets = ['BTC/USDT', 'ETH/USDT', 'EUR/USD']
        portfolio = self.manager.create_portfolio(
            name="test_portfolio",
            initial_capital=10000,
            assets=assets,
            allocation_method=AllocationMethod.EQUAL_WEIGHT
        )

        assert portfolio.name == "test_portfolio"
        assert portfolio.initial_capital == 10000
        assert len(portfolio.positions) == 3
        assert "test_portfolio" in self.manager.portfolios

    def test_equal_weight_allocation(self):
        """Test equal weight allocation method."""
        assets = ['BTC/USDT', 'ETH/USDT', 'EUR/USD']
        weights = self.manager._calculate_target_weights(assets, AllocationMethod.EQUAL_WEIGHT)

        assert len(weights) == 3
        assert all(weight == pytest.approx(1/3, abs=0.01) for weight in weights.values())

    def test_risk_parity_allocation(self):
        """Test risk parity allocation method."""
        assets = ['BTC/USDT', 'ETH/USDT', 'EUR/USD']

        # Mock volatility data
        with patch.object(self.manager, '_get_asset_volatilities') as mock_vol:
            mock_vol.return_value = {'BTC/USDT': 0.02, 'ETH/USDT': 0.03, 'EUR/USD': 0.01}
            weights = self.manager._calculate_target_weights(assets, AllocationMethod.RISK_PARITY)

            assert len(weights) == 3
            assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_portfolio_metrics_calculation(self):
        """Test portfolio metrics calculation."""
        # Create portfolio
        assets = ['BTC/USDT', 'ETH/USDT']
        portfolio = self.manager.create_portfolio(
            "test_portfolio",
            10000,
            assets,
            AllocationMethod.EQUAL_WEIGHT
        )

        # Calculate metrics
        metrics = self.manager.calculate_portfolio_metrics("test_portfolio")

        # Verify metrics object has expected attributes
        assert hasattr(metrics, 'total_value')
        assert hasattr(metrics, 'cumulative_return')
        assert hasattr(metrics, 'volatility')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'var_95')
        assert hasattr(metrics, 'diversification_ratio')
        assert hasattr(metrics, 'concentration_ratio')

        # Verify reasonable values
        assert metrics.total_value > 0
        assert isinstance(metrics.cumulative_return, (int, float))
        assert metrics.volatility >= 0
        assert isinstance(metrics.sharpe_ratio, (int, float))
        assert isinstance(metrics.var_95, (int, float))
        assert metrics.diversification_ratio > 0
        assert metrics.concentration_ratio >= 0

    def test_rebalancing_check(self):
        """Test rebalancing logic."""
        assets = ['BTC/USDT', 'ETH/USDT']
        portfolio = self.manager.create_portfolio(
            name="rebalance_test",
            initial_capital=10000,
            assets=assets
        )

        needs_rebalance, deviations = self.manager.check_rebalancing_needed("rebalance_test")

        assert isinstance(needs_rebalance, bool)
        assert isinstance(deviations, dict)
        assert len(deviations) == len(assets)

    def test_asset_addition(self):
        """Test adding assets to portfolio."""
        assets = ['BTC/USDT', 'ETH/USDT']
        self.manager.create_portfolio(
            name="add_asset_test",
            initial_capital=10000,
            assets=assets
        )

        success = self.manager.add_asset_to_portfolio("add_asset_test", "EUR/USD", 0.1)

        assert success
        portfolio = self.manager.portfolios["add_asset_test"]
        assert "EUR/USD" in portfolio.positions
        assert len(portfolio.positions) == 3

    def test_asset_removal(self):
        """Test removing assets from portfolio."""
        assets = ['BTC/USDT', 'ETH/USDT', 'EUR/USD']
        self.manager.create_portfolio(
            name="remove_asset_test",
            initial_capital=10000,
            assets=assets
        )

        success = self.manager.remove_asset_from_portfolio("remove_asset_test", "EUR/USD")

        assert success
        portfolio = self.manager.portfolios["remove_asset_test"]
        assert "EUR/USD" not in portfolio.positions
        assert len(portfolio.positions) == 2

    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        assets = ['BTC/USDT', 'ETH/USDT']
        self.manager.create_portfolio(
            name="summary_test",
            initial_capital=10000,
            assets=assets
        )

        summary = self.manager.get_portfolio_summary("summary_test")

        required_keys = ['portfolio_name', 'positions', 'metrics', 'constraints']
        for key in required_keys:
            assert key in summary

        assert len(summary['positions']) == 2
        assert 'total_value' in summary['metrics']

    def test_invalid_portfolio_operations(self):
        """Test error handling for invalid operations."""
        # Test accessing non-existent portfolio
        with pytest.raises(ValueError):
            self.manager.update_portfolio_prices("non_existent")

        with pytest.raises(ValueError):
            self.manager.get_portfolio_summary("non_existent")

    def test_constraints_validation(self):
        """Test portfolio constraints validation."""
        constraints = PortfolioConstraints(
            max_weight_per_asset=0.3,
            max_assets=5,
            max_volatility=0.2
        )

        assets = ['BTC/USDT', 'ETH/USDT']
        portfolio = self.manager.create_portfolio(
            name="constraints_test",
            initial_capital=10000,
            assets=assets,
            constraints=constraints
        )

        assert portfolio.constraints.max_weight_per_asset == 0.3
        assert portfolio.constraints.max_assets == 5


class TestAllocationMethods:
    """Test different allocation methods."""

    def setup_method(self):
        self.manager = PortfolioManager()

    def test_volatility_targeted_allocation(self):
        """Test volatility-targeted allocation."""
        assets = ['BTC/USDT', 'ETH/USDT', 'EUR/USD']

        with patch.object(self.manager, '_get_asset_volatilities') as mock_vol:
            mock_vol.return_value = {'BTC/USDT': 0.02, 'ETH/USDT': 0.03, 'EUR/USD': 0.01}
            weights = self.manager._calculate_target_weights(assets, AllocationMethod.VOLATILITY_TARGETED)

            assert len(weights) == 3
            assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_minimum_variance_fallback(self):
        """Test fallback to equal weight when volatility data unavailable."""
        assets = ['BTC/USDT', 'ETH/USDT']

        with patch.object(self.manager, '_get_asset_volatilities') as mock_vol:
            mock_vol.return_value = {}  # No volatility data
            weights = self.manager._calculate_target_weights(assets, AllocationMethod.RISK_PARITY)

            assert len(weights) == 2
            assert all(weight == pytest.approx(0.5, abs=0.01) for weight in weights.values())


class TestPortfolioPosition:
    """Test individual portfolio position functionality."""

    def test_position_calculations(self):
        """Test position-level calculations."""
        from src.portfolio_manager import PortfolioPosition

        position = PortfolioPosition(
            symbol="BTC/USDT",
            quantity=10,
            entry_price=50000,
            current_price=55000,
            market_value=550000,
            weight=0.5,
            unrealized_pnl=50000,
            volatility=0.02,
            correlation=0.0,
            last_updated=None
        )

        assert position.market_value == 550000
        assert position.unrealized_pnl == 50000
        assert position.weight == 0.5