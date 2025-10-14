"""
Risk Service - Comprehensive risk management for trading operations.

This package provides:
- Position sizing based on risk percentage
- Stop-loss and take-profit calculations
- Leverage management based on volatility
- Risk metrics (VaR, Sharpe ratio, drawdown)
- Portfolio risk assessment
"""

from .service import (
    RiskService, RiskRequest, RiskResponse, RiskParameters,
    PositionSizing, RiskAssessment, RiskLevel, PositionType,
    RiskMetric, EventSystem
)

__version__ = "1.0.0"
__all__ = [
    "RiskService",
    "RiskRequest",
    "RiskResponse",
    "RiskParameters",
    "PositionSizing",
    "RiskAssessment",
    "RiskLevel",
    "PositionType",
    "RiskMetric",
    "EventSystem"
]