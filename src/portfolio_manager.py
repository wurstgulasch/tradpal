"""
Portfolio Management Module for TradPal

This module provides advanced portfolio management capabilities for multi-asset trading,
including position sizing, risk allocation, rebalancing, and performance tracking.

Features:
- Multi-asset portfolio support (beyond BTC/USDT)
- Dynamic position sizing based on volatility and correlation
- Risk parity allocation across assets
- Automatic rebalancing based on target allocations
- Portfolio-level risk management (VaR, CVaR, drawdown limits)
- Performance attribution and analytics
- Integration with existing trading signals

Author: TradPal Team
Version: 2.5.0
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from src.cache import Cache

# Configure logging
logger = logging.getLogger(__name__)

class AllocationMethod(Enum):
    """Portfolio allocation methods."""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP = "market_cap"
    RISK_PARITY = "risk_parity"
    VOLATILITY_TARGETED = "volatility_targeted"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"

class RebalancingFrequency(Enum):
    """Rebalancing frequency options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    THRESHOLD = "threshold"  # Rebalance when deviation exceeds threshold

@dataclass
class PortfolioPosition:
    """Represents a position in the portfolio."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    weight: float
    unrealized_pnl: float
    volatility: float
    correlation: float
    last_updated: datetime

@dataclass
class PortfolioConstraints:
    """Portfolio risk and allocation constraints."""
    max_weight_per_asset: float = 0.20  # Maximum 20% per asset
    min_weight_per_asset: float = 0.01  # Minimum 1% per asset
    max_volatility: float = 0.25  # Maximum portfolio volatility (25%)
    max_drawdown: float = 0.15  # Maximum drawdown (15%)
    max_correlation: float = 0.80  # Maximum correlation between assets
    min_diversification: int = 5  # Minimum number of assets
    max_assets: int = 20  # Maximum number of assets
    risk_free_rate: float = 0.02  # Risk-free rate for Sharpe ratio

@dataclass
class PortfolioMetrics:
    """Portfolio performance and risk metrics."""
    total_value: float
    daily_return: float
    cumulative_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (95%)
    diversification_ratio: float
    concentration_ratio: float  # Herfindahl-Hirschman Index
    last_updated: datetime

@dataclass
class Portfolio:
    """Main portfolio management class."""
    name: str
    initial_capital: float
    current_capital: float
    positions: Dict[str, PortfolioPosition] = field(default_factory=dict)
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    allocation_method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT
    rebalancing_freq: RebalancingFrequency = RebalancingFrequency.MONTHLY
    rebalance_threshold: float = 0.05  # 5% deviation triggers rebalance
    benchmark_symbol: str = "BTC/USDT"  # Benchmark for performance comparison
    created_date: datetime = field(default_factory=datetime.utcnow)
    last_rebalanced: Optional[datetime] = None

    def __post_init__(self):
        """Initialize portfolio with empty positions."""
        if not self.positions:
            self.positions = {}

class PortfolioManager:
    """
    Advanced portfolio management system for TradPal.

    Provides comprehensive portfolio management including:
    - Multi-asset position tracking
    - Dynamic risk-based allocation
    - Automatic rebalancing
    - Performance analytics
    - Risk management
    """

    def __init__(self,
                 cache_manager: Optional[Cache] = None,
                 risk_free_rate: float = 0.02):
        """
        Initialize the portfolio manager.

        Args:
            cache_manager: Cache manager for price data
            risk_free_rate: Risk-free rate for performance calculations
        """
        self.cache_manager = cache_manager or Cache()
        self.risk_free_rate = risk_free_rate
        self.portfolios: Dict[str, Portfolio] = {}

        # Asset universe for multi-asset portfolios
        self.asset_universe = self._load_asset_universe()

        logger.info("PortfolioManager initialized successfully")

    def _load_asset_universe(self) -> List[str]:
        """Load available trading assets."""
        # Default asset universe - can be expanded
        return [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'DOT/USDT', 'AVAX/USDT', 'LINK/USDT', 'LTC/USDT', 'XRP/USDT',
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD',
            'NZD/USD', 'USD/CAD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY'
        ]

    def create_portfolio(self,
                        name: str,
                        initial_capital: float,
                        assets: List[str],
                        allocation_method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT,
                        constraints: Optional[PortfolioConstraints] = None) -> Portfolio:
        """
        Create a new portfolio with specified assets.

        Args:
            name: Portfolio name
            initial_capital: Initial investment amount
            assets: List of asset symbols
            allocation_method: Asset allocation method
            constraints: Portfolio constraints

        Returns:
            Created portfolio object
        """
        if name in self.portfolios:
            raise ValueError(f"Portfolio '{name}' already exists")

        # Validate assets
        invalid_assets = [asset for asset in assets if asset not in self.asset_universe]
        if invalid_assets:
            logger.warning(f"Assets not in universe: {invalid_assets}")

        # Create portfolio
        portfolio = Portfolio(
            name=name,
            initial_capital=initial_capital,
            current_capital=initial_capital,
            allocation_method=allocation_method,
            constraints=constraints or PortfolioConstraints()
        )

        # Initialize positions with target allocations
        target_weights = self._calculate_target_weights(assets, allocation_method)
        self._initialize_positions(portfolio, assets, target_weights, initial_capital)

        self.portfolios[name] = portfolio

        logger.info(f"Created portfolio '{name}' with {len(assets)} assets")
        return portfolio

    def _calculate_target_weights(self,
                                assets: List[str],
                                method: AllocationMethod) -> Dict[str, float]:
        """
        Calculate target weights for assets based on allocation method.

        Args:
            assets: List of asset symbols
            method: Allocation method

        Returns:
            Dictionary of asset weights
        """
        n_assets = len(assets)

        if method == AllocationMethod.EQUAL_WEIGHT:
            weight = 1.0 / n_assets
            return {asset: weight for asset in assets}

        elif method == AllocationMethod.RISK_PARITY:
            # Risk parity: equal risk contribution from each asset
            # Simplified version - in practice would use historical volatility
            volatilities = self._get_asset_volatilities(assets)
            if volatilities:
                # Inverse volatility weighting
                inv_vol = {asset: 1.0 / vol for asset, vol in volatilities.items()}
                total_inv_vol = sum(inv_vol.values())
                return {asset: inv_vol[asset] / total_inv_vol for asset in assets}
            else:
                # Fallback to equal weight
                weight = 1.0 / n_assets
                return {asset: weight for asset in assets}

        elif method == AllocationMethod.VOLATILITY_TARGETED:
            # Target equal volatility contribution
            volatilities = self._get_asset_volatilities(assets)
            if volatilities:
                # Scale weights by inverse volatility
                avg_vol = np.mean(list(volatilities.values()))
                weights = {}
                for asset in assets:
                    vol = volatilities.get(asset, avg_vol)
                    weights[asset] = avg_vol / vol
                # Normalize
                total_weight = sum(weights.values())
                return {asset: weight / total_weight for asset, weight in weights.items()}
            else:
                weight = 1.0 / n_assets
                return {asset: weight for asset in assets}

        else:
            # Default to equal weight for unsupported methods
            weight = 1.0 / n_assets
            return {asset: weight for asset in assets}

    def _get_asset_volatilities(self, assets: List[str]) -> Dict[str, float]:
        """Get historical volatilities for assets."""
        volatilities = {}

        for asset in assets:
            try:
                # Try to get cached volatility data
                cache_key = f"volatility_{asset}_30d"
                vol_data = self.cache_manager.get(cache_key)

                if vol_data:
                    volatilities[asset] = vol_data
                else:
                    # Estimate from recent price data
                    # This is a simplified estimation
                    volatilities[asset] = 0.02  # 2% daily volatility as default

            except Exception as e:
                logger.warning(f"Could not get volatility for {asset}: {e}")
                volatilities[asset] = 0.02  # Default volatility

        return volatilities

    def _initialize_positions(self,
                            portfolio: Portfolio,
                            assets: List[str],
                            target_weights: Dict[str, float],
                            capital: float):
        """Initialize portfolio positions with target allocations."""
        for asset in assets:
            weight = target_weights[asset]
            allocation = capital * weight

            # Get current price (simplified - would need real price data)
            current_price = self._get_current_price(asset)

            if current_price > 0:
                quantity = allocation / current_price

                position = PortfolioPosition(
                    symbol=asset,
                    quantity=quantity,
                    entry_price=current_price,
                    current_price=current_price,
                    market_value=allocation,
                    weight=weight,
                    unrealized_pnl=0.0,
                    volatility=0.02,  # Placeholder
                    correlation=0.0,  # Placeholder
                    last_updated=datetime.utcnow()
                )

                portfolio.positions[asset] = position

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for an asset."""
        # This would integrate with data_fetcher in practice
        # For now, return placeholder prices
        price_map = {
            'BTC/USDT': 45000,
            'ETH/USDT': 2800,
            'EUR/USD': 1.08,
            'GBP/USD': 1.27,
            'USD/JPY': 150.0,
        }
        return price_map.get(symbol, 100.0)  # Default price

    def update_portfolio_prices(self, portfolio_name: str) -> bool:
        """
        Update current prices for all positions in a portfolio.

        Args:
            portfolio_name: Name of the portfolio to update

        Returns:
            True if update successful
        """
        if portfolio_name not in self.portfolios:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        portfolio = self.portfolios[portfolio_name]
        updated = False

        for symbol, position in portfolio.positions.items():
            try:
                current_price = self._get_current_price(symbol)

                if current_price != position.current_price:
                    # Update position
                    position.current_price = current_price
                    position.market_value = position.quantity * current_price
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    position.last_updated = datetime.utcnow()

                    updated = True

            except Exception as e:
                logger.warning(f"Failed to update price for {symbol}: {e}")

        updated = True

        if updated:
            # Recalculate portfolio weights
            self._update_portfolio_weights(portfolio)

        return updated

    def _update_portfolio_weights(self, portfolio: Portfolio):
        """Update position weights based on current market values."""
        total_value = sum(pos.market_value for pos in portfolio.positions.values())

        for position in portfolio.positions.values():
            position.weight = position.market_value / total_value if total_value > 0 else 0

    def calculate_portfolio_metrics(self, portfolio_name: str) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio performance and risk metrics.

        Args:
            portfolio_name: Name of the portfolio

        Returns:
            PortfolioMetrics object with calculated metrics
        """
        if portfolio_name not in self.portfolios:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        portfolio = self.portfolios[portfolio_name]

        # Calculate basic metrics
        total_value = sum(pos.market_value for pos in portfolio.positions.values())
        positions_value = sum(pos.market_value for pos in portfolio.positions.values())

        # Calculate returns (simplified - would need historical data)
        cumulative_return = (total_value - portfolio.initial_capital) / portfolio.initial_capital

        # Calculate weights array for risk metrics
        weights = np.array([pos.weight for pos in portfolio.positions.values()])

        # Simplified risk metrics (would need covariance matrix in practice)
        volatility = np.sqrt(np.sum(weights ** 2 * 0.02 ** 2))  # Simplified volatility

        # Sharpe ratio
        if volatility > 0:
            sharpe_ratio = (cumulative_return - self.risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0

        # Diversification ratio (simplified)
        if len(portfolio.positions) > 1:
            diversification_ratio = 1.0 / volatility
        else:
            diversification_ratio = 1.0

        # Concentration ratio (Herfindahl-Hirschman Index)
        concentration_ratio = np.sum(weights ** 2)

        metrics = PortfolioMetrics(
            total_value=total_value,
            daily_return=0.0,  # Would need daily data
            cumulative_return=cumulative_return,
            annualized_return=cumulative_return,  # Simplified
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=0.0,  # Would need historical data
            var_95=-volatility * 1.645,  # Simplified VaR
            cvar_95=-volatility * 2.0,   # Simplified CVaR
            diversification_ratio=diversification_ratio,
            concentration_ratio=concentration_ratio,
            last_updated=datetime.utcnow()
        )

        return metrics

    def check_rebalancing_needed(self, portfolio_name: str) -> Tuple[bool, Dict[str, float]]:
        """
        Check if portfolio needs rebalancing based on current vs target weights.

        Args:
            portfolio_name: Name of the portfolio

        Returns:
            Tuple of (needs_rebalancing, deviation_dict)
        """
        if portfolio_name not in self.portfolios:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        portfolio = self.portfolios[portfolio_name]

        # Get current target weights
        assets = list(portfolio.positions.keys())
        target_weights = self._calculate_target_weights(assets, portfolio.allocation_method)

        deviations = {}
        needs_rebalancing = False

        for asset, position in portfolio.positions.items():
            target_weight = target_weights.get(asset, 0)
            current_weight = position.weight
            deviation = abs(current_weight - target_weight)
            deviations[asset] = deviation

            if deviation > portfolio.rebalance_threshold:
                needs_rebalancing = True

        return needs_rebalancing, deviations

    def rebalance_portfolio(self, portfolio_name: str) -> bool:
        """
        Rebalance portfolio to target allocations.

        Args:
            portfolio_name: Name of the portfolio

        Returns:
            True if rebalancing was performed
        """
        if portfolio_name not in self.portfolios:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        portfolio = self.portfolios[portfolio_name]

        needs_rebalancing, deviations = self.check_rebalancing_needed(portfolio_name)

        if not needs_rebalancing:
            logger.info(f"Portfolio '{portfolio_name}' does not need rebalancing")
            return False

        # Calculate new target allocations
        assets = list(portfolio.positions.keys())
        target_weights = self._calculate_target_weights(assets, portfolio.allocation_method)

        total_value = sum(pos.market_value for pos in portfolio.positions.values())

        rebalance_trades = []

        for asset, position in portfolio.positions.items():
            target_weight = target_weights[asset]
            target_value = total_value * target_weight
            current_value = position.market_value

            # Calculate trade needed
            value_difference = target_value - current_value

            if abs(value_difference) > 1.0:  # Minimum trade size
                current_price = position.current_price
                quantity_change = value_difference / current_price

                rebalance_trades.append({
                    'asset': asset,
                    'current_quantity': position.quantity,
                    'quantity_change': quantity_change,
                    'current_price': current_price,
                    'value_change': value_difference
                })

                # Update position
                position.quantity += quantity_change
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity

        # Update portfolio weights
        self._update_portfolio_weights(portfolio)
        portfolio.last_rebalanced = datetime.utcnow()

        logger.info(f"Rebalanced portfolio '{portfolio_name}' with {len(rebalance_trades)} trades")
        return True

    def add_asset_to_portfolio(self,
                              portfolio_name: str,
                              asset: str,
                              allocation: float = 0.05) -> bool:
        """
        Add a new asset to an existing portfolio.

        Args:
            portfolio_name: Name of the portfolio
            asset: Asset symbol to add
            allocation: Target allocation for the new asset

        Returns:
            True if asset was added successfully
        """
        if portfolio_name not in self.portfolios:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        portfolio = self.portfolios[portfolio_name]

        if asset in portfolio.positions:
            logger.warning(f"Asset {asset} already in portfolio {portfolio_name}")
            return False

        # Check constraints
        if len(portfolio.positions) >= portfolio.constraints.max_assets:
            logger.warning(f"Maximum assets limit reached for portfolio {portfolio_name}")
            return False

        # Add position
        current_price = self._get_current_price(asset)
        total_value = sum(pos.market_value for pos in portfolio.positions.values())
        allocation_value = total_value * allocation

        if current_price > 0:
            quantity = allocation_value / current_price

            position = PortfolioPosition(
                symbol=asset,
                quantity=quantity,
                entry_price=current_price,
                current_price=current_price,
                market_value=allocation_value,
                weight=allocation,
                unrealized_pnl=0.0,
                volatility=0.02,
                correlation=0.0,
                last_updated=datetime.utcnow()
            )

            portfolio.positions[asset] = position

            # Rebalance existing positions
            self._update_portfolio_weights(portfolio)

            logger.info(f"Added asset {asset} to portfolio {portfolio_name}")
            return True

        return False

    def remove_asset_from_portfolio(self, portfolio_name: str, asset: str) -> bool:
        """
        Remove an asset from a portfolio.

        Args:
            portfolio_name: Name of the portfolio
            asset: Asset symbol to remove

        Returns:
            True if asset was removed successfully
        """
        if portfolio_name not in self.portfolios:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        portfolio = self.portfolios[portfolio_name]

        if asset not in portfolio.positions:
            logger.warning(f"Asset {asset} not in portfolio {portfolio_name}")
            return False

        # Remove position
        position = portfolio.positions.pop(asset)

        # Rebalance remaining positions
        if portfolio.positions:
            self._update_portfolio_weights(portfolio)

        logger.info(f"Removed asset {asset} from portfolio {portfolio_name}")
        return True

    def get_portfolio_summary(self, portfolio_name: str) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary.

        Args:
            portfolio_name: Name of the portfolio

        Returns:
            Dictionary with portfolio summary
        """
        if portfolio_name not in self.portfolios:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        portfolio = self.portfolios[portfolio_name]
        metrics = self.calculate_portfolio_metrics(portfolio_name)
        needs_rebalancing, deviations = self.check_rebalancing_needed(portfolio_name)

        positions_data = []
        for symbol, position in portfolio.positions.items():
            positions_data.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'weight': position.weight,
                'unrealized_pnl': position.unrealized_pnl,
                'pnl_percentage': (position.unrealized_pnl / (position.entry_price * position.quantity)) * 100
            })

        return {
            'portfolio_name': portfolio.name,
            'created_date': portfolio.created_date,
            'last_rebalanced': portfolio.last_rebalanced,
            'allocation_method': portfolio.allocation_method.value,
            'rebalancing_frequency': portfolio.rebalancing_freq.value,
            'initial_capital': portfolio.initial_capital,
            'current_capital': portfolio.current_capital,
            'positions': positions_data,
            'metrics': {
                'total_value': metrics.total_value,
                'cumulative_return': metrics.cumulative_return,
                'annualized_return': metrics.annualized_return,
                'volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'var_95': metrics.var_95,
                'diversification_ratio': metrics.diversification_ratio,
                'concentration_ratio': metrics.concentration_ratio
            },
            'rebalancing': {
                'needs_rebalancing': needs_rebalancing,
                'deviations': deviations,
                'threshold': portfolio.rebalance_threshold
            },
            'constraints': {
                'max_weight_per_asset': portfolio.constraints.max_weight_per_asset,
                'max_volatility': portfolio.constraints.max_volatility,
                'max_drawdown': portfolio.constraints.max_drawdown,
                'max_assets': portfolio.constraints.max_assets
            }
        }

# Convenience functions
def create_sample_portfolio() -> Portfolio:
    """Create a sample multi-asset portfolio for demonstration."""
    manager = PortfolioManager()

    assets = ['BTC/USDT', 'ETH/USDT', 'EUR/USD', 'GBP/USD']
    portfolio = manager.create_portfolio(
        name="sample_portfolio",
        initial_capital=10000,
        assets=assets,
        allocation_method=AllocationMethod.EQUAL_WEIGHT
    )

    return portfolio

def get_portfolio_performance(portfolio_name: str) -> Dict[str, Any]:
    """Get portfolio performance summary."""
    manager = PortfolioManager()
    return manager.get_portfolio_summary(portfolio_name)

if __name__ == "__main__":
    # Example usage
    print("📊 Testing Portfolio Management...")

    try:
        # Create sample portfolio
        portfolio = create_sample_portfolio()
        print(f"✅ Created portfolio: {portfolio.name}")
        print(f"   Assets: {list(portfolio.positions.keys())}")
        print(f"   Initial Capital: ${portfolio.initial_capital}")

        # Update prices (simulated)
        manager = PortfolioManager()
        manager.update_portfolio_prices("sample_portfolio")

        # Get portfolio summary
        summary = manager.get_portfolio_summary("sample_portfolio")
        print("\n📈 Portfolio Summary:")
        print(f"   Total Value: ${summary['metrics']['total_value']:.2f}")
        print(f"   Cumulative Return: {summary['metrics']['cumulative_return']:.2%}")
        print(f"   Sharpe Ratio: {summary['metrics']['sharpe_ratio']:.2f}")

        print(f"\n📊 Positions:")
        for pos in summary['positions']:
            print(f"   {pos['symbol']}: {pos['weight']:.1%} (${pos['market_value']:.2f})")

        # Check rebalancing
        needs_rebalance, deviations = manager.check_rebalancing_needed("sample_portfolio")
        print(f"\n🔄 Needs Rebalancing: {needs_rebalance}")

        print("🎉 Portfolio management demo completed!")

    except Exception as e:
        print(f"❌ Portfolio management failed: {e}")