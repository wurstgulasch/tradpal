"""
Multi-Asset Portfolio Management for Trading System

This module provides comprehensive portfolio management capabilities including:
- Multi-asset diversification and risk allocation
- Dynamic position sizing based on volatility and correlation
- Portfolio optimization using Modern Portfolio Theory
- Risk parity and minimum variance strategies
- Performance tracking and rebalancing
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Config imports
from config.settings import (
    SYMBOL, CAPITAL, RISK_PER_TRADE, INITIAL_CAPITAL,
    MAX_LEVERAGE, LEVERAGE_BASE
)

# Import risk management
# from src.risk_management import RiskManager

# Import performance tracking
# from src.performance import PerformanceTracker

# Import audit logging
from src.audit_logger import audit_logger


@dataclass
class AssetPosition:
    """Represents a position in a specific asset."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    allocation_pct: float
    volatility: float
    correlation: float
    last_updated: datetime


@dataclass
class PortfolioMetrics:
    """Portfolio performance and risk metrics."""
    total_value: float
    total_pnl: float
    daily_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    diversification_ratio: float
    risk_parity_score: float
    last_updated: datetime


class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimization strategies."""

    @abstractmethod
    def optimize_weights(self, assets: List[str], returns: pd.DataFrame,
                        constraints: Dict[str, Any]) -> Dict[str, float]:
        """
        Optimize portfolio weights.

        Args:
            assets: List of asset symbols
            returns: Historical returns DataFrame
            constraints: Optimization constraints

        Returns:
            Dictionary of asset weights
        """
        pass


class ModernPortfolioTheoryOptimizer(PortfolioOptimizer):
    """MPT-based portfolio optimization."""

    def optimize_weights(self, assets: List[str], returns: pd.DataFrame,
                        constraints: Dict[str, Any]) -> Dict[str, float]:
        """
        Optimize using Modern Portfolio Theory (mean-variance optimization).

        Args:
            assets: List of asset symbols
            returns: Historical returns DataFrame
            constraints: Optimization constraints

        Returns:
            Optimal asset weights
        """
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean()
            cov_matrix = returns.cov()

            # Target return constraint
            target_return = constraints.get('target_return', expected_returns.mean())

            # Risk-free rate
            risk_free_rate = constraints.get('risk_free_rate', 0.02)

            # Number of assets
            n_assets = len(assets)

            # Constraints for optimization
            bounds = constraints.get('bounds', [(0, 1) for _ in range(n_assets)])
            weight_constraint = constraints.get('weight_constraint', {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

            from scipy.optimize import minimize

            # Objective function: minimize portfolio variance
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))

            # Constraint: target return
            def portfolio_return_constraint(weights):
                return np.dot(weights, expected_returns) - target_return

            constraints_opt = [
                weight_constraint,
                {'type': 'eq', 'fun': portfolio_return_constraint}
            ]

            # Initial weights (equal weight)
            initial_weights = np.array([1/n_assets] * n_assets)

            # Optimize
            result = minimize(
                portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_opt
            )

            if result.success:
                weights = dict(zip(assets, result.x))
                return weights
            else:
                # Fallback to equal weights
                return {asset: 1/n_assets for asset in assets}

        except Exception as e:
            print(f"MPT optimization failed: {e}")
            # Fallback to equal weights
            return {asset: 1/n_assets for asset in assets}


class RiskParityOptimizer(PortfolioOptimizer):
    """Risk parity portfolio optimization."""

    def optimize_weights(self, assets: List[str], returns: pd.DataFrame,
                        constraints: Dict[str, Any]) -> Dict[str, float]:
        """
        Optimize using risk parity approach.

        Args:
            assets: List of asset symbols
            returns: Historical returns DataFrame
            constraints: Optimization constraints

        Returns:
            Risk-parity asset weights
        """
        try:
            # Calculate covariance matrix
            cov_matrix = returns.cov()

            # Risk parity: equal risk contribution from each asset
            n_assets = len(assets)

            # Initial equal weights
            weights = np.array([1/n_assets] * n_assets)

            # Iterative optimization for risk parity
            max_iter = 100
            tolerance = 1e-6

            for _ in range(max_iter):
                # Calculate portfolio volatility contribution of each asset
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                asset_vol_contrib = weights * np.dot(cov_matrix, weights) / portfolio_vol

                # Risk parity target
                target_risk = portfolio_vol / n_assets

                # Update weights
                new_weights = asset_vol_contrib / target_risk
                new_weights = new_weights / np.sum(new_weights)  # Normalize

                # Check convergence
                if np.max(np.abs(new_weights - weights)) < tolerance:
                    break

                weights = new_weights

            weights_dict = dict(zip(assets, weights))
            return weights_dict

        except Exception as e:
            print(f"Risk parity optimization failed: {e}")
            # Fallback to equal weights
            return {asset: 1/len(assets) for asset in assets}


class MinimumVarianceOptimizer(PortfolioOptimizer):
    """Minimum variance portfolio optimization."""

    def optimize_weights(self, assets: List[str], returns: pd.DataFrame,
                        constraints: Dict[str, Any]) -> Dict[str, float]:
        """
        Optimize for minimum portfolio variance.

        Args:
            assets: List of asset symbols
            returns: Historical returns DataFrame
            constraints: Optimization constraints

        Returns:
            Minimum variance asset weights
        """
        try:
            cov_matrix = returns.cov()
            n_assets = len(assets)

            from scipy.optimize import minimize

            # Objective: minimize portfolio variance
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))

            # Constraints: weights sum to 1, no short selling
            constraints_opt = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]

            bounds = [(0, 1) for _ in range(n_assets)]  # No short selling
            initial_weights = np.array([1/n_assets] * n_assets)

            result = minimize(
                portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_opt
            )

            if result.success:
                weights = dict(zip(assets, result.x))
                return weights
            else:
                return {asset: 1/n_assets for asset in assets}

        except Exception as e:
            print(f"Minimum variance optimization failed: {e}")
            return {asset: 1/len(assets) for asset in assets}


class MultiAssetPortfolioManager:
    """
    Multi-Asset Portfolio Manager with advanced risk management and optimization.

    Features:
    - Dynamic asset allocation and rebalancing
    - Multiple optimization strategies (MPT, Risk Parity, Min Variance)
    - Real-time risk monitoring and position sizing
    - Correlation-based diversification
    - Performance attribution and reporting
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL,
                 max_assets: int = 10, rebalance_threshold: float = 0.05,
                 results_dir: str = "output/portfolio"):
        """
        Initialize portfolio manager.

        Args:
            initial_capital: Starting portfolio capital
            max_assets: Maximum number of assets in portfolio
            rebalance_threshold: Threshold for rebalancing (5% deviation)
            results_dir: Directory to store portfolio results
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_assets = max_assets
        self.rebalance_threshold = rebalance_threshold
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Portfolio components
        self.positions: Dict[str, AssetPosition] = {}
        self.target_weights: Dict[str, float] = {}
        self.optimizers: Dict[str, PortfolioOptimizer] = {
            'mpt': ModernPortfolioTheoryOptimizer(),
            'risk_parity': RiskParityOptimizer(),
            'min_variance': MinimumVarianceOptimizer()
        }

        # Risk management
        # self.risk_manager = RiskManager()

        # Performance tracking
        # self.performance_tracker = PerformanceTracker()

        # Historical data
        self.asset_returns: pd.DataFrame = pd.DataFrame()
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()

        # Portfolio metrics
        self.metrics = PortfolioMetrics(
            total_value=initial_capital,
            total_pnl=0.0,
            daily_pnl=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            diversification_ratio=1.0,
            risk_parity_score=0.0,
            last_updated=datetime.now()
        )

        # Load existing portfolio if available
        self._load_portfolio()

    def add_asset(self, symbol: str, initial_allocation: float = 0.0,
                  volatility: float = 0.0, correlation: float = 0.0) -> bool:
        """
        Add an asset to the portfolio.

        Args:
            symbol: Asset symbol
            initial_allocation: Initial allocation percentage (0-1)
            volatility: Asset volatility
            correlation: Average correlation with portfolio

        Returns:
            True if asset was added successfully
        """
        if len(self.positions) >= self.max_assets:
            print(f"❌ Cannot add {symbol}: Maximum assets ({self.max_assets}) reached")
            return False

        if symbol in self.positions:
            print(f"⚠️  {symbol} already in portfolio")
            return False

        # Create position
        position = AssetPosition(
            symbol=symbol,
            quantity=0.0,
            entry_price=0.0,
            current_price=0.0,
            unrealized_pnl=0.0,
            allocation_pct=initial_allocation,
            volatility=volatility,
            correlation=correlation,
            last_updated=datetime.now()
        )

        self.positions[symbol] = position

        # Update target weights
        if initial_allocation > 0:
            self.target_weights[symbol] = initial_allocation
            self._rebalance_weights()

        print(f"✅ Added {symbol} to portfolio")
        return True

    def remove_asset(self, symbol: str) -> bool:
        """
        Remove an asset from the portfolio.

        Args:
            symbol: Asset symbol to remove

        Returns:
            True if asset was removed successfully
        """
        if symbol not in self.positions:
            print(f"⚠️  {symbol} not in portfolio")
            return False

        # Close position and realize P&L
        position = self.positions[symbol]
        if position.quantity != 0:
            realized_pnl = position.unrealized_pnl
            self.current_capital += realized_pnl

            # Log position closure
            audit_logger.log_system_event(
                event_type="PORTFOLIO_ASSET_REMOVED",
                message=f"Removed {symbol} from portfolio",
                details={
                    'symbol': symbol,
                    'realized_pnl': realized_pnl,
                    'final_allocation': position.allocation_pct
                }
            )

        # Remove from portfolio
        del self.positions[symbol]
        if symbol in self.target_weights:
            del self.target_weights[symbol]

        # Rebalance remaining assets
        self._rebalance_weights()

        print(f"✅ Removed {symbol} from portfolio")
        return True

    def update_prices(self, price_updates: Dict[str, float]) -> None:
        """
        Update current prices for portfolio assets.

        Args:
            price_updates: Dictionary of symbol -> price updates
        """
        updated_positions = []

        for symbol, price in price_updates.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = price
                position.unrealized_pnl = (price - position.entry_price) * position.quantity
                position.last_updated = datetime.now()
                updated_positions.append(symbol)

        if updated_positions:
            self._update_portfolio_metrics()

            # Check for rebalancing
            if self._should_rebalance():
                self.rebalance_portfolio()

    def calculate_position_size(self, symbol: str, signal_strength: float = 1.0,
                               volatility_adjustment: bool = True) -> float:
        """
        Calculate optimal position size for an asset.

        Args:
            symbol: Asset symbol
            signal_strength: Signal strength multiplier (0-1)
            volatility_adjustment: Whether to adjust for volatility

        Returns:
            Position size in base currency
        """
        if symbol not in self.positions:
            return 0.0

        position = self.positions[symbol]

        # Base position size from risk management
        # base_risk = self.risk_manager.calculate_position_size(
        #     capital=self.current_capital,
        #     risk_per_trade=RISK_PER_TRADE,
        #     volatility=position.volatility if volatility_adjustment else 0.1
        # )

        # Simple position sizing based on volatility
        risk_per_trade = 0.01  # 1% risk per trade
        base_risk = self.current_capital * risk_per_trade
        if volatility_adjustment and position.volatility > 0:
            base_risk = base_risk / position.volatility  # Adjust for volatility

        # Apply signal strength
        position_size = base_risk * signal_strength

        # Apply leverage constraints
        max_position_value = self.current_capital * MAX_LEVERAGE * position.allocation_pct
        position_size = min(position_size, max_position_value)

        # Ensure minimum order size
        position_size = max(position_size, 10.0)  # Minimum $10 order

        return position_size

    def execute_trade(self, symbol: str, signal: str, price: float,
                     quantity: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a trade for the given asset.

        Args:
            symbol: Asset symbol
            signal: Trade signal ('BUY', 'SELL', 'CLOSE')
            price: Execution price
            quantity: Trade quantity (calculated if None)

        Returns:
            Trade execution results
        """
        if symbol not in self.positions:
            return {'success': False, 'error': f'Asset {symbol} not in portfolio'}

        position = self.positions[symbol]

        # Calculate quantity if not provided
        if quantity is None:
            if signal in ['BUY', 'SELL']:
                position_value = self.calculate_position_size(symbol)
                quantity = position_value / price
            else:
                quantity = position.quantity  # Close entire position

        # Apply signal direction
        if signal == 'SELL':
            quantity = -quantity
        elif signal == 'CLOSE':
            quantity = -position.quantity  # Close position

        # Update position
        old_quantity = position.quantity
        position.quantity += quantity
        position.last_updated = datetime.now()

        # Update entry price (weighted average)
        if position.quantity != 0:
            if old_quantity == 0:
                position.entry_price = price
            else:
                total_cost = (old_quantity * position.entry_price) + (quantity * price)
                position.entry_price = total_cost / position.quantity

        # Calculate realized P&L for closed portion
        realized_pnl = 0.0
        if signal == 'CLOSE' or (old_quantity > 0 and position.quantity < 0) or (old_quantity < 0 and position.quantity > 0):
            # Position direction changed or closed
            closed_quantity = abs(quantity) if abs(quantity) <= abs(old_quantity) else abs(old_quantity)
            realized_pnl = closed_quantity * (price - position.entry_price) * (1 if old_quantity > 0 else -1)

        # Update capital
        trade_value = abs(quantity) * price
        if signal == 'BUY':
            self.current_capital -= trade_value
        elif signal == 'SELL':
            self.current_capital += trade_value
        elif signal == 'CLOSE':
            self.current_capital += realized_pnl

        # Update metrics
        self._update_portfolio_metrics()

        # Log trade
        audit_logger.log_system_event(
            event_type="PORTFOLIO_TRADE_EXECUTED",
            message=f"Executed {signal} trade for {symbol}",
            details={
                'symbol': symbol,
                'signal': signal,
                'price': price,
                'quantity': quantity,
                'trade_value': trade_value,
                'realized_pnl': realized_pnl
            }
        )

        return {
            'success': True,
            'symbol': symbol,
            'signal': signal,
            'price': price,
            'quantity': quantity,
            'trade_value': trade_value,
            'realized_pnl': realized_pnl,
            'new_position': position.quantity
        }

    def optimize_portfolio(self, optimization_method: str = 'risk_parity',
                          constraints: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Optimize portfolio weights using specified method.

        Args:
            optimization_method: Optimization method ('mpt', 'risk_parity', 'min_variance')
            constraints: Optimization constraints

        Returns:
            Optimized asset weights
        """
        if not self.asset_returns.empty and len(self.positions) > 1:
            assets = list(self.positions.keys())

            if optimization_method not in self.optimizers:
                optimization_method = 'risk_parity'

            optimizer = self.optimizers[optimization_method]

            if constraints is None:
                constraints = {
                    'target_return': 0.02,  # 2% target return
                    'risk_free_rate': 0.01,  # 1% risk-free rate
                    'bounds': [(0, 0.3) for _ in assets]  # Max 30% per asset
                }

            try:
                optimized_weights = optimizer.optimize_weights(
                    assets, self.asset_returns[assets], constraints
                )

                self.target_weights = optimized_weights
                print(f"✅ Portfolio optimized using {optimization_method}")

                return optimized_weights

            except Exception as e:
                print(f"❌ Portfolio optimization failed: {e}")
                return self._equal_weights()

        else:
            # Fallback to equal weights
            return self._equal_weights()

    def rebalance_portfolio(self) -> Dict[str, Any]:
        """
        Rebalance portfolio to target weights.

        Returns:
            Rebalancing results
        """
        if not self.target_weights:
            return {'success': False, 'error': 'No target weights set'}

        total_value = self.metrics.total_value
        rebalance_trades = []

        for symbol, target_weight in self.target_weights.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                current_value = position.quantity * position.current_price
                target_value = total_value * target_weight
                value_difference = target_value - current_value

                # Only rebalance if deviation is significant
                if abs(value_difference / total_value) > self.rebalance_threshold:
                    # Calculate required quantity change
                    if position.current_price > 0:
                        quantity_change = value_difference / position.current_price

                        # Execute rebalancing trade
                        signal = 'BUY' if quantity_change > 0 else 'SELL'
                        trade_result = self.execute_trade(
                            symbol=symbol,
                            signal=signal,
                            price=position.current_price,
                            quantity=abs(quantity_change)
                        )

                        if trade_result['success']:
                            rebalance_trades.append(trade_result)

        # Log rebalancing
        audit_logger.log_system_event(
            event_type="PORTFOLIO_REBALANCED",
            message=f"Portfolio rebalanced with {len(rebalance_trades)} trades",
            details={
                'total_value': total_value,
                'trades_executed': len(rebalance_trades),
                'target_weights': self.target_weights.copy()
            }
        )

        return {
            'success': True,
            'trades_executed': len(rebalance_trades),
            'rebalance_trades': rebalance_trades,
            'total_value': total_value
        }

    def update_asset_data(self, asset_returns: pd.DataFrame) -> None:
        """
        Update historical asset returns for optimization.

        Args:
            asset_returns: DataFrame with asset returns
        """
        self.asset_returns = asset_returns

        # Update correlation matrix
        if len(asset_returns.columns) > 1:
            self.correlation_matrix = asset_returns.corr()

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get current portfolio metrics."""
        self._update_portfolio_metrics()
        return self.metrics

    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.

        Returns:
            Dictionary of risk metrics
        """
        if self.asset_returns.empty:
            return {}

        # Portfolio returns
        portfolio_returns = self._calculate_portfolio_returns()

        # Basic metrics
        metrics = {
            'volatility': portfolio_returns.std() * np.sqrt(252),  # Annualized
            'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'var_95': np.percentile(portfolio_returns, 5),  # 95% VaR
            'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean(),
        }

        # Diversification ratio
        if not self.correlation_matrix.empty:
            avg_correlation = self.correlation_matrix.mean().mean()
            metrics['diversification_ratio'] = 1 / (1 + avg_correlation)

        # Risk parity score
        if len(self.positions) > 1:
            metrics['risk_parity_score'] = self._calculate_risk_parity_score()

        return metrics

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio report.

        Returns:
            Portfolio report dictionary
        """
        metrics = self.get_portfolio_metrics()
        risk_metrics = self.get_risk_metrics()

        # Position details
        positions_data = []
        for symbol, position in self.positions.items():
            positions_data.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'allocation_pct': position.allocation_pct,
                'volatility': position.volatility,
                'correlation': position.correlation
            })

        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_metrics': {
                'total_value': metrics.total_value,
                'total_pnl': metrics.total_pnl,
                'daily_pnl': metrics.daily_pnl,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'volatility': metrics.volatility,
                'diversification_ratio': metrics.diversification_ratio,
                'risk_parity_score': metrics.risk_parity_score
            },
            'risk_metrics': risk_metrics,
            'positions': positions_data,
            'target_weights': self.target_weights.copy(),
            'asset_count': len(self.positions),
            'correlation_matrix': self.correlation_matrix.to_dict() if not self.correlation_matrix.empty else {}
        }

        return report

    def _should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced."""
        if not self.target_weights:
            return False

        total_value = self.metrics.total_value

        for symbol, target_weight in self.target_weights.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                current_value = position.quantity * position.current_price
                current_weight = current_value / total_value if total_value > 0 else 0

                if abs(current_weight - target_weight) > self.rebalance_threshold:
                    return True

        return False

    def _rebalance_weights(self) -> None:
        """Rebalance weights after position changes."""
        if not self.positions:
            return

        # Equal weights for now (can be improved with optimization)
        equal_weight = 1.0 / len(self.positions)
        self.target_weights = {symbol: equal_weight for symbol in self.positions.keys()}

    def _equal_weights(self) -> Dict[str, float]:
        """Generate equal weights for all assets."""
        if not self.positions:
            return {}

        equal_weight = 1.0 / len(self.positions)
        return {symbol: equal_weight for symbol in self.positions.keys()}

    def _update_portfolio_metrics(self) -> None:
        """Update portfolio performance metrics."""
        total_value = self.current_capital
        total_pnl = 0.0

        # Calculate total value and P&L from positions
        for position in self.positions.values():
            if position.quantity != 0 and position.current_price > 0:
                position_value = position.quantity * position.current_price
                total_value += position_value
                total_pnl += position.unrealized_pnl

        # Calculate daily P&L (simplified)
        daily_pnl = total_pnl * 0.01  # Placeholder

        # Calculate risk metrics
        risk_metrics = self.get_risk_metrics()

        # Update metrics
        self.metrics.total_value = total_value
        self.metrics.total_pnl = total_pnl
        self.metrics.daily_pnl = daily_pnl
        self.metrics.sharpe_ratio = risk_metrics.get('sharpe_ratio', 0.0)
        self.metrics.max_drawdown = risk_metrics.get('max_drawdown', 0.0)
        self.metrics.volatility = risk_metrics.get('volatility', 0.0)
        self.metrics.diversification_ratio = risk_metrics.get('diversification_ratio', 1.0)
        self.metrics.risk_parity_score = risk_metrics.get('risk_parity_score', 0.0)
        self.metrics.last_updated = datetime.now()

    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate portfolio returns time series."""
        if self.asset_returns.empty or not self.target_weights:
            return pd.Series()

        # Weight the asset returns
        portfolio_returns = pd.Series(0.0, index=self.asset_returns.index)

        for symbol, weight in self.target_weights.items():
            if symbol in self.asset_returns.columns:
                portfolio_returns += self.asset_returns[symbol] * weight

        return portfolio_returns

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        if returns.empty:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_risk_parity_score(self) -> float:
        """Calculate risk parity score (0-1, higher is better)."""
        if len(self.positions) < 2:
            return 1.0

        # Calculate risk contribution of each asset
        volatilities = np.array([pos.volatility for pos in self.positions.values()])
        allocations = np.array([pos.allocation_pct for pos in self.positions.values()])

        if np.sum(volatilities) == 0:
            return 1.0

        # Risk contributions
        risk_contributions = volatilities * allocations

        # Risk parity score: 1 - coefficient of variation of risk contributions
        cv = np.std(risk_contributions) / np.mean(risk_contributions)
        risk_parity_score = max(0, 1 - cv)

        return risk_parity_score

    def _save_portfolio(self) -> None:
        """Save portfolio state to disk."""
        try:
            portfolio_data = {
                'current_capital': self.current_capital,
                'positions': {
                    symbol: {
                        'quantity': pos.quantity,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'allocation_pct': pos.allocation_pct,
                        'volatility': pos.volatility,
                        'correlation': pos.correlation,
                        'last_updated': pos.last_updated.isoformat()
                    }
                    for symbol, pos in self.positions.items()
                },
                'target_weights': self.target_weights,
                'metrics': {
                    'total_value': self.metrics.total_value,
                    'total_pnl': self.metrics.total_pnl,
                    'sharpe_ratio': self.metrics.sharpe_ratio,
                    'max_drawdown': self.metrics.max_drawdown,
                    'last_updated': self.metrics.last_updated.isoformat()
                },
                'last_saved': datetime.now().isoformat()
            }

            portfolio_file = self.results_dir / "portfolio_state.json"
            with open(portfolio_file, 'w') as f:
                json.dump(portfolio_data, f, indent=2, default=str)

        except Exception as e:
            print(f"❌ Failed to save portfolio: {e}")

    def _load_portfolio(self) -> None:
        """Load portfolio state from disk."""
        try:
            portfolio_file = self.results_dir / "portfolio_state.json"
            if not portfolio_file.exists():
                return

            with open(portfolio_file, 'r') as f:
                portfolio_data = json.load(f)

            # Restore capital
            self.current_capital = portfolio_data.get('current_capital', self.initial_capital)

            # Restore positions
            positions_data = portfolio_data.get('positions', {})
            for symbol, pos_data in positions_data.items():
                position = AssetPosition(
                    symbol=symbol,
                    quantity=pos_data['quantity'],
                    entry_price=pos_data['entry_price'],
                    current_price=pos_data['current_price'],
                    unrealized_pnl=0.0,  # Will be recalculated
                    allocation_pct=pos_data['allocation_pct'],
                    volatility=pos_data['volatility'],
                    correlation=pos_data['correlation'],
                    last_updated=datetime.fromisoformat(pos_data['last_updated'])
                )
                self.positions[symbol] = position

            # Restore target weights
            self.target_weights = portfolio_data.get('target_weights', {})

            print(f"📂 Loaded portfolio with {len(self.positions)} positions")

        except Exception as e:
            print(f"❌ Failed to load portfolio: {e}")


# Global portfolio manager instance
portfolio_manager = None


def get_portfolio_manager(initial_capital: float = INITIAL_CAPITAL) -> MultiAssetPortfolioManager:
    """Get or create portfolio manager instance."""
    global portfolio_manager

    if portfolio_manager is None:
        portfolio_manager = MultiAssetPortfolioManager(initial_capital=initial_capital)

    return portfolio_manager


def create_sample_portfolio() -> MultiAssetPortfolioManager:
    """Create a sample portfolio for demonstration."""
    manager = get_portfolio_manager()

    # Add sample assets
    sample_assets = [
        ('BTC/USDT', 0.4, 0.05, 0.0),  # 40% allocation, 5% volatility, low correlation
        ('ETH/USDT', 0.3, 0.07, 0.6),  # 30% allocation, 7% volatility, moderate correlation
        ('ADA/USDT', 0.2, 0.08, 0.7),  # 20% allocation, 8% volatility, high correlation
        ('DOT/USDT', 0.1, 0.06, 0.5),  # 10% allocation, 6% volatility, moderate correlation
    ]

    for symbol, allocation, volatility, correlation in sample_assets:
        manager.add_asset(symbol, allocation, volatility, correlation)

    return manager