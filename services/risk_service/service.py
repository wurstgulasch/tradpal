#!/usr/bin/env python3
"""
Risk Service - Comprehensive risk management for trading operations.

This service provides:
- Position sizing based on risk percentage
- Stop-loss and take-profit calculations
- Leverage management based on volatility
- Risk metrics (VaR, Sharpe ratio, drawdown)
- Portfolio risk assessment
- Risk-adjusted position sizing
"""

import asyncio
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PositionType(Enum):
    """Position types."""
    LONG = "long"
    SHORT = "short"


class RiskMetric(Enum):
    """Available risk metrics."""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VALUE_AT_RISK = "value_at_risk"
    EXPECTED_SHORTFALL = "expected_shortfall"
    VOLATILITY = "volatility"
    BETA = "beta"


@dataclass
class RiskParameters:
    """Risk management parameters."""
    max_risk_per_trade: float = 0.01  # 1% max risk per trade
    max_portfolio_risk: float = 0.05   # 5% max portfolio risk
    max_leverage: float = 5.0          # Maximum leverage
    min_leverage: float = 1.0          # Minimum leverage
    stop_loss_atr_multiplier: float = 1.5  # ATR multiplier for SL
    take_profit_atr_multiplier: float = 3.0  # ATR multiplier for TP
    risk_free_rate: float = 0.02       # Risk-free rate for Sharpe
    confidence_level: float = 0.95     # Confidence level for VaR
    max_drawdown_limit: float = 0.20   # 20% max drawdown
    volatility_lookback: int = 20      # Periods for volatility calc

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PositionSizing:
    """Position sizing calculation result."""
    symbol: str
    position_size: float
    position_value: float
    risk_amount: float
    stop_loss_price: float
    take_profit_price: float
    leverage: float
    risk_percentage: float
    reward_risk_ratio: float
    calculated_at: datetime
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['calculated_at'] = self.calculated_at.isoformat()
        return data


@dataclass
class RiskAssessment:
    """Risk assessment for a portfolio or position."""
    symbol: str
    risk_level: RiskLevel
    risk_score: float
    metrics: Dict[str, float]
    recommendations: List[str]
    assessed_at: datetime
    time_horizon: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['risk_level'] = self.risk_level.value
        data['assessed_at'] = self.assessed_at.isoformat()
        return data


class RiskRequest(BaseModel):
    """Request model for risk calculations."""
    symbol: str = Field(..., description="Trading symbol")
    capital: float = Field(..., description="Available capital")
    entry_price: float = Field(..., description="Entry price")
    position_type: str = Field(..., description="Position type (long/short)")
    atr_value: Optional[float] = Field(None, description="ATR value for SL/TP calc")
    volatility: Optional[float] = Field(None, description="Current volatility")
    risk_percentage: float = Field(0.01, description="Risk percentage per trade")

    @validator('position_type')
    def validate_position_type(cls, v):
        """Validate position type."""
        if v not in ['long', 'short']:
            raise ValueError("Position type must be 'long' or 'short'")
        return v

    @validator('capital', 'entry_price', 'risk_percentage')
    def validate_positive(cls, v):
        """Validate positive values."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class RiskResponse(BaseModel):
    """Response model for risk calculations."""
    success: bool
    position_sizing: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class EventSystem:
    """Simple event system for service communication."""

    def __init__(self):
        self._handlers: Dict[str, List[callable]] = {}

    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish an event."""
        if event_type in self._handlers:
            tasks = []
            for handler in self._handlers[event_type]:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(data))
                else:
                    tasks.append(asyncio.to_thread(handler, data))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def subscribe(self, event_type: str, handler: callable):
        """Subscribe to an event."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)


class RiskService:
    """
    Comprehensive risk management service for trading operations.

    Features:
    - Kelly Criterion and risk-adjusted position sizing
    - ATR-based stop-loss and take-profit calculations
    - Volatility-adjusted leverage management
    - Portfolio risk metrics and diversification
    - Real-time risk monitoring and alerts
    """

    def __init__(self, event_system: Optional[EventSystem] = None):
        self.event_system = event_system or EventSystem()
        self.default_params = RiskParameters()
        self.portfolio_positions: Dict[str, PositionSizing] = {}
        self.risk_history: List[Dict[str, Any]] = []

        logger.info("Risk Service initialized")

    def _calculate_position_size_kelly(self, win_rate: float, win_loss_ratio: float,
                                     risk_percentage: float, capital: float) -> float:
        """
        Calculate position size using Kelly Criterion.

        Kelly % = W - (1-W)/R
        Where W = win rate, R = win/loss ratio
        """
        if win_loss_ratio <= 0:
            return capital * risk_percentage  # Fallback to fixed percentage

        kelly_percentage = win_rate - (1 - win_rate) / win_loss_ratio
        kelly_percentage = max(0, min(kelly_percentage, risk_percentage))  # Bound Kelly

        return capital * kelly_percentage

    def _calculate_atr_stop_loss(self, entry_price: float, atr_value: float,
                               position_type: PositionType,
                               multiplier: float = 1.5) -> float:
        """Calculate ATR-based stop loss."""
        atr_distance = atr_value * multiplier

        if position_type == PositionType.LONG:
            return entry_price - atr_distance
        else:
            return entry_price + atr_distance

    def _calculate_atr_take_profit(self, entry_price: float, atr_value: float,
                                 position_type: PositionType,
                                 multiplier: float = 3.0) -> float:
        """Calculate ATR-based take profit."""
        atr_distance = atr_value * multiplier

        if position_type == PositionType.LONG:
            return entry_price + atr_distance
        else:
            return entry_price - atr_distance

    def _calculate_volatility_adjusted_leverage(self, volatility: float,
                                               base_leverage: float = 1.0,
                                               volatility_mean: float = 0.02) -> float:
        """
        Calculate leverage adjusted for volatility.

        Higher volatility = lower leverage for risk control
        """
        if volatility <= 0:
            return base_leverage

        # Volatility adjustment factor (inverse relationship)
        adjustment_factor = volatility_mean / volatility
        adjustment_factor = max(0.1, min(3.0, adjustment_factor))  # Bound factor

        leverage = base_leverage * adjustment_factor
        leverage = max(self.default_params.min_leverage,
                      min(self.default_params.max_leverage, leverage))

        return leverage

    def _calculate_risk_metrics(self, returns: pd.Series,
                              risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        if returns.empty:
            return {metric.value: 0.0 for metric in RiskMetric}

        metrics = {}

        # Sharpe Ratio
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if excess_returns.std() > 0:
            metrics[RiskMetric.SHARPE_RATIO.value] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        else:
            metrics[RiskMetric.SHARPE_RATIO.value] = 0.0

        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            metrics[RiskMetric.SORTINO_RATIO.value] = (returns.mean() - risk_free_rate / 252) / downside_returns.std() * np.sqrt(252)
        else:
            metrics[RiskMetric.SORTINO_RATIO.value] = 0.0

        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics[RiskMetric.MAX_DRAWDOWN.value] = drawdown.min()

        # Value at Risk (95% confidence)
        metrics[RiskMetric.VALUE_AT_RISK.value] = np.percentile(returns, 5)

        # Expected Shortfall (CVaR)
        var_95 = metrics[RiskMetric.VALUE_AT_RISK.value]
        metrics[RiskMetric.EXPECTED_SHORTFALL.value] = returns[returns <= var_95].mean()

        # Volatility (annualized)
        metrics[RiskMetric.VOLATILITY.value] = returns.std() * np.sqrt(252)

        # Beta (simplified - would need market returns for full calculation)
        metrics[RiskMetric.BETA.value] = 1.0  # Placeholder

        return metrics

    def _assess_risk_level(self, metrics: Dict[str, float],
                          position_size_pct: float) -> Tuple[RiskLevel, float, List[str]]:
        """Assess overall risk level and generate recommendations."""
        score = 0.0
        recommendations = []

        # Sharpe Ratio assessment
        sharpe = metrics.get(RiskMetric.SHARPE_RATIO.value, 0)
        if sharpe < 0.5:
            score += 3
            recommendations.append("Low Sharpe ratio indicates poor risk-adjusted returns")
        elif sharpe < 1.0:
            score += 1
            recommendations.append("Moderate Sharpe ratio - consider improving return consistency")

        # Maximum Drawdown assessment
        max_dd = abs(metrics.get(RiskMetric.MAX_DRAWDOWN.value, 0))
        if max_dd > 0.20:
            score += 3
            recommendations.append(f"High maximum drawdown ({max_dd:.1%}) - consider reducing position sizes")
        elif max_dd > 0.10:
            score += 1
            recommendations.append(f"Moderate drawdown ({max_dd:.1%}) - monitor closely")

        # Volatility assessment
        vol = metrics.get(RiskMetric.VOLATILITY.value, 0)
        if vol > 0.50:  # 50% annualized volatility
            score += 2
            recommendations.append(f"High volatility ({vol:.1%}) - consider reducing leverage")
        elif vol > 0.30:
            score += 1
            recommendations.append(f"Moderate volatility ({vol:.1%}) - monitor market conditions")

        # Position size assessment
        if position_size_pct > 0.05:  # >5% of capital
            score += 2
            recommendations.append(f"Large position size ({position_size_pct:.1%}) increases portfolio risk")
        elif position_size_pct > 0.02:
            score += 1
            recommendations.append(f"Moderate position size ({position_size_pct:.1%}) - acceptable risk")

        # VaR assessment
        var = abs(metrics.get(RiskMetric.VALUE_AT_RISK.value, 0))
        if var > 0.05:  # >5% daily VaR
            score += 2
            recommendations.append(f"High Value at Risk ({var:.1%}) - consider hedging")

        # Determine risk level
        if score >= 8:
            risk_level = RiskLevel.VERY_HIGH
        elif score >= 5:
            risk_level = RiskLevel.HIGH
        elif score >= 3:
            risk_level = RiskLevel.MODERATE
        elif score >= 1:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.VERY_LOW

        return risk_level, score, recommendations

    async def calculate_position_sizing(self, request: RiskRequest,
                                      historical_returns: Optional[pd.Series] = None) -> PositionSizing:
        """
        Calculate optimal position sizing with risk management.

        Args:
            request: Risk calculation parameters
            historical_returns: Historical returns for risk metrics (optional)

        Returns:
            PositionSizing with complete risk-adjusted position details
        """
        try:
            position_type = PositionType(request.position_type)

            # Calculate risk amount
            risk_amount = request.capital * request.risk_percentage

            # Calculate position size based on stop loss distance
            if request.atr_value and request.atr_value > 0:
                # ATR-based stop loss
                sl_price = self._calculate_atr_stop_loss(
                    request.entry_price, request.atr_value, position_type,
                    self.default_params.stop_loss_atr_multiplier
                )

                # Calculate position size based on risk and stop loss distance
                sl_distance = abs(request.entry_price - sl_price)
                if sl_distance > 0:
                    position_size = risk_amount / sl_distance
                else:
                    position_size = request.capital * request.risk_percentage  # Fallback

                # Calculate take profit
                tp_price = self._calculate_atr_take_profit(
                    request.entry_price, request.atr_value, position_type,
                    self.default_params.take_profit_atr_multiplier
                )
            else:
                # Fallback without ATR
                position_size = request.capital * request.risk_percentage
                sl_distance = request.entry_price * 0.02  # 2% default stop loss
                sl_price = request.entry_price - sl_distance if position_type == PositionType.LONG else request.entry_price + sl_distance
                tp_price = request.entry_price + sl_distance * 2 if position_type == PositionType.LONG else request.entry_price - sl_distance * 2

            # Calculate position value
            position_value = position_size * request.entry_price

            # Calculate leverage based on volatility
            leverage = 1.0
            if request.volatility:
                leverage = self._calculate_volatility_adjusted_leverage(request.volatility)

            # Apply leverage limits
            position_value *= leverage
            position_size *= leverage

            # Calculate reward/risk ratio
            tp_distance = abs(tp_price - request.entry_price)
            sl_distance = abs(sl_price - request.entry_price)
            reward_risk_ratio = tp_distance / sl_distance if sl_distance > 0 else 1.0

            # Create position sizing result
            sizing = PositionSizing(
                symbol=request.symbol,
                position_size=position_size,
                position_value=position_value,
                risk_amount=risk_amount,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                leverage=leverage,
                risk_percentage=request.risk_percentage,
                reward_risk_ratio=reward_risk_ratio,
                calculated_at=datetime.now(),
                parameters=self.default_params.to_dict()
            )

            # Store in portfolio
            self.portfolio_positions[request.symbol] = sizing

            # Publish event
            await self.event_system.publish("risk.position_sized", {
                "symbol": request.symbol,
                "position_size": position_size,
                "risk_amount": risk_amount,
                "leverage": leverage
            })

            logger.info(f"Calculated position sizing for {request.symbol}: "
                       f"Size={position_size:.4f}, Risk={risk_amount:.2f}, "
                       f"Leverage={leverage:.1f}")

            return sizing

        except Exception as e:
            logger.error(f"Position sizing calculation failed: {e}")
            await self.event_system.publish("risk.calculation_error", {
                "symbol": request.symbol,
                "error": str(e)
            })
            raise

    async def assess_portfolio_risk(self, portfolio_returns: pd.Series,
                                  time_horizon: str = "daily") -> RiskAssessment:
        """
        Assess overall portfolio risk.

        Args:
            portfolio_returns: Time series of portfolio returns
            time_horizon: Risk assessment horizon

        Returns:
            RiskAssessment with comprehensive risk analysis
        """
        try:
            # Calculate risk metrics
            metrics = self._calculate_risk_metrics(
                portfolio_returns,
                self.default_params.risk_free_rate
            )

            # Calculate total portfolio risk exposure
            total_exposure = sum(pos.position_value for pos in self.portfolio_positions.values())
            total_risk = sum(pos.risk_amount for pos in self.portfolio_positions.values())

            # Assess risk level
            position_size_pct = total_risk / 100000 if total_exposure > 0 else 0  # Assume 100k capital
            risk_level, risk_score, recommendations = self._assess_risk_level(metrics, position_size_pct)

            # Add portfolio-specific recommendations
            if len(self.portfolio_positions) > 10:
                recommendations.append("High number of positions - consider diversification limits")

            if total_risk > self.default_params.max_portfolio_risk * 100000:  # Assume 100k capital
                recommendations.append("Portfolio risk exceeds maximum threshold")

            assessment = RiskAssessment(
                symbol="PORTFOLIO",
                risk_level=risk_level,
                risk_score=risk_score,
                metrics=metrics,
                recommendations=recommendations,
                assessed_at=datetime.now(),
                time_horizon=time_horizon
            )

            # Store assessment
            self.risk_history.append({
                "timestamp": datetime.now(),
                "assessment": assessment.to_dict(),
                "portfolio_size": len(self.portfolio_positions),
                "total_exposure": total_exposure
            })

            # Publish event
            await self.event_system.publish("risk.portfolio_assessed", {
                "risk_level": risk_level.value,
                "risk_score": risk_score,
                "metrics": metrics
            })

            logger.info(f"Portfolio risk assessment: {risk_level.value} "
                       f"(score: {risk_score:.1f})")

            return assessment

        except Exception as e:
            logger.error(f"Portfolio risk assessment failed: {e}")
            await self.event_system.publish("risk.assessment_error", {
                "error": str(e)
            })
            raise

    async def get_portfolio_exposure(self) -> Dict[str, Any]:
        """Get current portfolio risk exposure."""
        total_exposure = sum(pos.position_value for pos in self.portfolio_positions.values())
        total_risk = sum(pos.risk_amount for pos in self.portfolio_positions.values())
        total_leverage = sum(pos.leverage for pos in self.portfolio_positions.values())

        return {
            "total_positions": len(self.portfolio_positions),
            "total_exposure": total_exposure,
            "total_risk": total_risk,
            "total_leverage": total_leverage,
            "risk_percentage": total_risk / total_exposure if total_exposure > 0 else 0,
            "positions": {symbol: pos.to_dict() for symbol, pos in self.portfolio_positions.items()}
        }

    async def update_risk_parameters(self, parameters: Dict[str, Any]):
        """Update risk management parameters."""
        for key, value in parameters.items():
            if hasattr(self.default_params, key):
                setattr(self.default_params, key, value)

        logger.info(f"Updated risk parameters: {parameters}")

        await self.event_system.publish("risk.parameters_updated", {
            "parameters": parameters
        })

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "service": "risk_service",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "portfolio_positions": len(self.portfolio_positions),
            "risk_history_entries": len(self.risk_history),
            "parameters": self.default_params.to_dict()
        }