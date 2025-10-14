"""
Risk Management Module for TradPal

This module provides comprehensive risk management capabilities for trading,
including position sizing, drawdown control, and Kelly criterion calculations.

Features:
- Kelly Criterion position sizing
- Drawdown limit monitoring
- Risk-adjusted position sizing
- Portfolio-level risk controls
- Dynamic risk management

Author: TradPal Team
Version: 2.5.0
"""

import os
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    """Risk management parameters."""
    max_drawdown: float = 0.10  # Maximum portfolio drawdown (10%)
    max_daily_loss: float = 0.05  # Maximum daily loss (5%)
    max_position_size: float = 0.10  # Maximum position size (10% of portfolio)
    max_leverage: float = 5.0  # Maximum leverage
    risk_per_trade: float = 0.01  # Risk per trade (1%)
    kelly_fraction: float = 0.5  # Fraction of Kelly size to use (50%)
    var_confidence: float = 0.95  # VaR confidence level
    cvar_confidence: float = 0.95  # CVaR confidence level


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    portfolio_value: float = 0.0
    peak_portfolio_value: float = 0.0
    total_risk: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class RiskManager:
    """
    Advanced risk management system for TradPal.

    Provides comprehensive risk control including:
    - Kelly criterion position sizing
    - Drawdown limit monitoring
    - Value at Risk (VaR) calculations
    - Dynamic risk adjustments
    - Portfolio-level risk controls
    """

    def __init__(self,
                 max_drawdown: float = 0.10,
                 max_daily_loss: float = 0.05,
                 kelly_enabled: bool = True,
                 kelly_fraction: float = 0.5,
                 initial_portfolio_value: float = 10000.0):
        """
        Initialize the risk manager.

        Args:
            max_drawdown: Maximum allowed portfolio drawdown
            max_daily_loss: Maximum allowed daily loss
            kelly_enabled: Whether to use Kelly criterion
            kelly_fraction: Fraction of Kelly size to use
            initial_portfolio_value: Initial portfolio value
        """
        self.parameters = RiskParameters(
            max_drawdown=max_drawdown,
            max_daily_loss=max_daily_loss,
            kelly_fraction=kelly_fraction
        )

        self.kelly_enabled = kelly_enabled
        self.metrics = RiskMetrics(portfolio_value=initial_portfolio_value)
        self.metrics.peak_portfolio_value = initial_portfolio_value

        # Risk tracking
        self.daily_trades = []
        self.portfolio_history = []
        self.position_sizes = {}

        logger.info("RiskManager initialized successfully")

    def calculate_kelly_position_size(self,
                                    win_rate: float,
                                    avg_win: float,
                                    avg_loss: float) -> float:
        """
        Calculate Kelly criterion position size.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average win size (as decimal)
            avg_loss: Average loss size (as decimal, positive)

        Returns:
            Kelly position size as fraction of portfolio
        """
        try:
            if win_rate <= 0 or win_rate >= 1:
                logger.warning(f"Invalid win rate: {win_rate}")
                return self.parameters.risk_per_trade  # Fallback to fixed risk

            if avg_loss <= 0:
                logger.warning(f"Invalid average loss: {avg_loss}")
                return self.parameters.risk_per_trade

            # Kelly formula: K = (bp - q) / b
            # Where: b = odds (avg_win/avg_loss), p = win_rate, q = loss_rate
            b = avg_win / avg_loss
            kelly_size = (win_rate * b - (1 - win_rate)) / b

            # Apply fraction of Kelly
            kelly_size *= self.parameters.kelly_fraction

            # Ensure reasonable bounds
            kelly_size = max(0.001, min(kelly_size, self.parameters.max_position_size))

            logger.debug(f"Kelly size calculated: {kelly_size:.4f} "
                        f"(win_rate={win_rate:.2f}, avg_win={avg_win:.4f}, avg_loss={avg_loss:.4f})")

            return kelly_size

        except Exception as e:
            logger.error(f"Error calculating Kelly position size: {str(e)}")
            return self.parameters.risk_per_trade  # Fallback

    def check_drawdown_limits(self) -> bool:
        """
        Check if current drawdown exceeds limits.

        Returns:
            True if trading should continue, False if limits exceeded
        """
        try:
            current_drawdown = self._calculate_current_drawdown()

            if current_drawdown > self.parameters.max_drawdown:
                logger.warning(f"Drawdown limit exceeded: {current_drawdown:.4f} > {self.parameters.max_drawdown:.4f}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking drawdown limits: {str(e)}")
            return False

    def check_daily_loss_limits(self) -> bool:
        """
        Check if daily loss exceeds limits.

        Returns:
            True if trading should continue, False if limits exceeded
        """
        try:
            daily_loss = abs(min(0, self.metrics.daily_pnl))

            if daily_loss > self.parameters.max_daily_loss * self.metrics.portfolio_value:
                logger.warning(f"Daily loss limit exceeded: ${daily_loss:.2f} > "
                             f"${self.parameters.max_daily_loss * self.metrics.portfolio_value:.2f}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking daily loss limits: {str(e)}")
            return False

    def calculate_position_size(self,
                              portfolio_value: float,
                              risk_per_trade: Optional[float] = None,
                              volatility: float = 0.02,
                              current_price: float = 0.0) -> float:
        """
        Calculate position size based on risk parameters.

        Args:
            portfolio_value: Current portfolio value
            risk_per_trade: Risk per trade (overrides default)
            volatility: Asset volatility
            current_price: Current asset price

        Returns:
            Position size as dollar amount
        """
        try:
            risk_amount = risk_per_trade or self.parameters.risk_per_trade
            position_size = portfolio_value * risk_amount

            # Adjust for volatility (higher volatility = smaller position)
            if volatility > 0.01:  # Only adjust if volatility is meaningful
                vol_adjustment = min(1.0, 0.02 / volatility)  # Target 2% volatility
                position_size *= vol_adjustment

            # Apply leverage limits
            max_position = portfolio_value * self.parameters.max_position_size
            position_size = min(position_size, max_position)

            # Convert to quantity if price provided
            if current_price > 0:
                quantity = position_size / current_price
                # Apply leverage cap
                max_quantity = (portfolio_value * self.parameters.max_leverage) / current_price
                quantity = min(quantity, max_quantity)
                return quantity

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return portfolio_value * self.parameters.risk_per_trade

    def update_portfolio_value(self, new_value: float):
        """
        Update portfolio value and recalculate risk metrics.

        Args:
            new_value: New portfolio value
        """
        try:
            old_value = self.metrics.portfolio_value
            self.metrics.portfolio_value = new_value

            # Update peak value
            if new_value > self.metrics.peak_portfolio_value:
                self.metrics.peak_portfolio_value = new_value

            # Calculate drawdown
            self.metrics.current_drawdown = self._calculate_current_drawdown()

            # Store in history
            self.portfolio_history.append({
                'timestamp': datetime.utcnow(),
                'value': new_value,
                'drawdown': self.metrics.current_drawdown
            })

            # Keep only last 1000 entries
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]

            logger.debug(f"Portfolio updated: ${new_value:.2f} "
                        f"(drawdown: {self.metrics.current_drawdown:.4f})")

        except Exception as e:
            logger.error(f"Error updating portfolio value: {str(e)}")

    def update_daily_pnl(self, pnl: float):
        """
        Update daily P&L.

        Args:
            pnl: Profit/Loss amount
        """
        self.metrics.daily_pnl += pnl

    def reset_daily_pnl(self):
        """Reset daily P&L at start of new trading day."""
        self.metrics.daily_pnl = 0.0
        self.daily_trades = []

    def record_trade(self, trade: Dict[str, Any]):
        """
        Record a trade for risk tracking.

        Args:
            trade: Trade information dictionary
        """
        try:
            self.daily_trades.append(trade)

            # Update position sizes
            asset = trade.get('asset', 'unknown')
            if asset not in self.position_sizes:
                self.position_sizes[asset] = 0

            quantity = trade.get('quantity', 0)
            side = trade.get('side', 'BUY')

            if side.upper() == 'BUY':
                self.position_sizes[asset] += quantity
            elif side.upper() == 'SELL':
                self.position_sizes[asset] -= quantity

        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")

    def calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: List of historical returns
            confidence: Confidence level (0-1)

        Returns:
            VaR value
        """
        try:
            if not returns or len(returns) < 10:
                return 0.0

            returns_array = np.array(returns)
            var = np.percentile(returns_array, (1 - confidence) * 100)

            self.metrics.var_95 = abs(var)  # Store as positive value
            return abs(var)

        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0

    def calculate_cvar(self, returns: List[float], confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).

        Args:
            returns: List of historical returns
            confidence: Confidence level (0-1)

        Returns:
            CVaR value
        """
        try:
            if not returns or len(returns) < 10:
                return 0.0

            returns_array = np.array(returns)
            var = np.percentile(returns_array, (1 - confidence) * 100)

            # CVaR is the average of returns below VaR
            tail_returns = returns_array[returns_array <= var]
            cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var

            self.metrics.cvar_95 = abs(cvar)  # Store as positive value
            return abs(cvar)

        except Exception as e:
            logger.error(f"Error calculating CVaR: {str(e)}")
            return 0.0

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.

        Returns:
            Dictionary of risk metrics
        """
        return {
            'current_drawdown': self.metrics.current_drawdown,
            'daily_pnl': self.metrics.daily_pnl,
            'portfolio_value': self.metrics.portfolio_value,
            'peak_portfolio_value': self.metrics.peak_portfolio_value,
            'total_risk': self.metrics.total_risk,
            'var_95': self.metrics.var_95,
            'cvar_95': self.metrics.cvar_95,
            'max_drawdown_limit': self.parameters.max_drawdown,
            'max_daily_loss_limit': self.parameters.max_daily_loss,
            'kelly_enabled': self.kelly_enabled,
            'kelly_fraction': self.parameters.kelly_fraction,
            'last_updated': self.metrics.last_updated
        }

    def should_stop_trading(self) -> Tuple[bool, str]:
        """
        Check if trading should be stopped due to risk limits.

        Returns:
            Tuple of (should_stop, reason)
        """
        try:
            # Check drawdown
            if not self.check_drawdown_limits():
                return True, f"Drawdown limit exceeded ({self.metrics.current_drawdown:.1%})"

            # Check daily loss
            if not self.check_daily_loss_limits():
                daily_loss_pct = abs(self.metrics.daily_pnl) / self.metrics.portfolio_value
                return True, f"Daily loss limit exceeded ({daily_loss_pct:.1%})"

            return False, "Risk limits OK"

        except Exception as e:
            logger.error(f"Error checking trading stop conditions: {str(e)}")
            return True, f"Risk check error: {str(e)}"

    def _calculate_current_drawdown(self) -> float:
        """Calculate current portfolio drawdown."""
        try:
            if self.metrics.peak_portfolio_value <= 0:
                return 0.0

            drawdown = (self.metrics.peak_portfolio_value - self.metrics.portfolio_value) / self.metrics.peak_portfolio_value
            return max(0.0, drawdown)

        except Exception as e:
            logger.error(f"Error calculating drawdown: {str(e)}")
            return 0.0

    def get_position_sizes(self) -> Dict[str, float]:
        """
        Get current position sizes.

        Returns:
            Dictionary of asset position sizes
        """
        return self.position_sizes.copy()

    def get_daily_trades(self) -> List[Dict[str, Any]]:
        """
        Get today's trades.

        Returns:
            List of daily trades
        """
        return self.daily_trades.copy()


# Convenience functions
def create_risk_manager(max_drawdown: float = 0.10,
                       max_daily_loss: float = 0.05,
                       kelly_enabled: bool = True) -> RiskManager:
    """
    Create a risk manager with default settings.

    Args:
        max_drawdown: Maximum drawdown limit
        max_daily_loss: Maximum daily loss limit
        kelly_enabled: Enable Kelly criterion

    Returns:
        Configured RiskManager instance
    """
    return RiskManager(
        max_drawdown=max_drawdown,
        max_daily_loss=max_daily_loss,
        kelly_enabled=kelly_enabled
    )


def calculate_kelly_size(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.5) -> float:
    """
    Convenience function to calculate Kelly position size.

    Args:
        win_rate: Win rate (0-1)
        avg_win: Average win (decimal)
        avg_loss: Average loss (decimal, positive)
        fraction: Fraction of Kelly to use

    Returns:
        Position size as fraction of portfolio
    """
    manager = RiskManager(kelly_fraction=fraction)
    return manager.calculate_kelly_position_size(win_rate, avg_win, avg_loss)


if __name__ == "__main__":
    # Example usage
    print("🛡️ Testing Risk Management...")

    try:
        # Create risk manager
        risk_manager = create_risk_manager()

        # Test Kelly calculation
        kelly_size = risk_manager.calculate_kelly_position_size(
            win_rate=0.6, avg_win=0.05, avg_loss=0.03
        )
        print(f"✅ Kelly position size: {kelly_size:.4f}")

        # Test drawdown check
        risk_manager.update_portfolio_value(9500)  # 5% loss
        can_trade = risk_manager.check_drawdown_limits()
        print(f"✅ Drawdown check (5% loss): {can_trade}")

        # Test position sizing
        position_size = risk_manager.calculate_position_size(10000, current_price=50000)
        print(f"✅ Position size: {position_size:.6f} BTC")

        # Get risk metrics
        metrics = risk_manager.get_risk_metrics()
        print(f"📊 Risk metrics: Drawdown={metrics['current_drawdown']:.1%}")

        print("🎉 Risk management demo completed!")

    except Exception as e:
        print(f"❌ Risk management failed: {e}")