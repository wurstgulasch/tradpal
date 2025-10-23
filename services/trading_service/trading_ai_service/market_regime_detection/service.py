"""
Market Regime Detection Service - Integrated into Trading Service
Provides market regime classification and analysis for trading decisions.
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class RegimeFeatures:
    """Features used for regime classification."""
    trend_strength: float
    volatility: float
    volume_trend: float
    momentum: float
    support_resistance: float


class MarketRegimeDetectionService:
    """Market regime detection and classification service."""

    def __init__(self, event_system=None):
        self.event_system = event_system
        self.is_initialized = False

    async def initialize(self):
        """Initialize the market regime detection service"""
        logger.info("Initializing Market Regime Detection Service...")
        self.is_initialized = True
        logger.info("Market Regime Detection Service initialized")

    async def shutdown(self):
        """Shutdown the market regime detection service"""
        logger.info("Market Regime Detection Service shut down")
        self.is_initialized = False

    async def detect_market_regime(self, data: pd.DataFrame, lookback_periods: int = 100) -> MarketRegime:
        """
        Detect current market regime from price data.

        Args:
            data: OHLCV DataFrame
            lookback_periods: Number of periods to analyze

        Returns:
            Detected market regime
        """
        if not self.is_initialized:
            raise RuntimeError("Market regime detection service not initialized")

        if len(data) < lookback_periods:
            return MarketRegime.SIDEWAYS

        # Use recent data
        recent_data = data.tail(lookback_periods).copy()

        # Calculate trend strength (slope of linear regression on closing prices)
        prices = recent_data['close'].values
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        trend_strength = slope / np.mean(prices)  # Normalize

        # Calculate volatility (standard deviation of returns)
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized

        # Determine regime based on trend and volatility
        if abs(trend_strength) > 0.02:  # Strong trend
            if trend_strength > 0:
                return MarketRegime.BULL_MARKET
            else:
                return MarketRegime.BEAR_MARKET
        elif volatility > 0.5:  # High volatility
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.2:  # Low volatility
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.SIDEWAYS

    async def get_regime_features(self, data: pd.DataFrame) -> RegimeFeatures:
        """
        Extract features for regime classification.

        Args:
            data: OHLCV DataFrame

        Returns:
            Regime features
        """
        if not self.is_initialized:
            raise RuntimeError("Market regime detection service not initialized")

        if len(data) < 50:
            return RegimeFeatures(0, 0, 0, 0, 0)

        # Trend strength (20-period linear regression slope)
        prices = data['close'].tail(20).values
        x = np.arange(len(prices))
        trend_strength = np.polyfit(x, prices, 1)[0] / np.mean(prices)

        # Volatility (20-period standard deviation of returns)
        returns = data['close'].pct_change().tail(20).dropna()
        volatility = returns.std()

        # Volume trend (correlation between price and volume)
        price_returns = data['close'].pct_change().tail(20).dropna()
        volume_changes = data['volume'].pct_change().tail(20).dropna()
        if len(price_returns) == len(volume_changes):
            volume_trend = np.corrcoef(price_returns, volume_changes)[0, 1]
        else:
            volume_trend = 0

        # Momentum (RSI-like indicator)
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        avg_gain = gains.rolling(14).mean().iloc[-1] if len(gains) >= 14 else gains.mean()
        avg_loss = losses.rolling(14).mean().iloc[-1] if len(losses) >= 14 else losses.mean()
        if avg_loss != 0:
            momentum = 100 - (100 / (1 + avg_gain / avg_loss))
        else:
            momentum = 100

        # Support/resistance (distance from recent highs/lows)
        recent_high = data['high'].tail(20).max()
        recent_low = data['low'].tail(20).min()
        current_price = data['close'].iloc[-1]
        support_resistance = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5

        return RegimeFeatures(
            trend_strength=float(trend_strength),
            volatility=float(volatility),
            volume_trend=float(volume_trend),
            momentum=float(momentum),
            support_resistance=float(support_resistance)
        )

    async def get_volatility_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Classify volatility regime.

        Args:
            data: OHLCV DataFrame

        Returns:
            Volatility regime information
        """
        if not self.is_initialized:
            raise RuntimeError("Market regime detection service not initialized")

        returns = data['close'].pct_change().dropna()
        if len(returns) < 20:
            return {"regime": "normal", "volatility": 0.0}

        volatility = returns.std() * np.sqrt(252)

        if volatility > 0.8:
            regime = "high"
        elif volatility < 0.3:
            regime = "low"
        else:
            regime = "normal"

        return {
            "regime": regime,
            "volatility": float(volatility),
            "threshold_high": 0.8,
            "threshold_low": 0.3
        }

    async def get_trend_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Classify trend regime.

        Args:
            data: OHLCV DataFrame

        Returns:
            Trend regime information
        """
        if not self.is_initialized:
            raise RuntimeError("Market regime detection service not initialized")

        if len(data) < 20:
            return {"regime": "sideways", "strength": 0.0}

        # Calculate trend strength
        prices = data['close'].tail(20).values
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        trend_strength = abs(slope / np.mean(prices))

        if trend_strength > 0.02:
            regime = "trending"
        elif trend_strength > 0.01:
            regime = "weak_trend"
        else:
            regime = "sideways"

        return {
            "regime": regime,
            "strength": float(trend_strength),
            "direction": "up" if slope > 0 else "down",
            "slope": float(slope)
        }

    async def get_regime_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate regime statistics.

        Args:
            data: OHLCV DataFrame

        Returns:
            Regime statistics
        """
        if not self.is_initialized:
            raise RuntimeError("Market regime detection service not initialized")

        if len(data) < 100:
            return {"error": "Insufficient data"}

        # Calculate rolling regime classifications
        window_size = 50
        regimes = []

        for i in range(window_size, len(data), 10):  # Sample every 10 periods
            window_data = data.iloc[i-window_size:i]
            regime = await self.detect_market_regime(window_data, window_size)
            regimes.append(regime.value)

        # Calculate regime distribution
        regime_counts = {}
        for regime in MarketRegime:
            regime_counts[regime.value] = regimes.count(regime.value)

        total_regimes = len(regimes)
        regime_percentages = {
            regime: count / total_regimes if total_regimes > 0 else 0
            for regime, count in regime_counts.items()
        }

        # Most common regime
        most_common = max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else "unknown"

        return {
            "total_periods": total_regimes,
            "regime_distribution": regime_counts,
            "regime_percentages": regime_percentages,
            "most_common_regime": most_common,
            "regime_stability": max(regime_percentages.values()) if regime_percentages else 0
        }