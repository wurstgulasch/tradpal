"""
TradPal Trading Service - Market Regime Detection
Simplified implementation for unified service consolidation
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketRegimeService:
    """Simplified market regime detection service"""

    def __init__(self, event_system=None):
        self.event_system = event_system
        self.current_regimes: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False

    async def initialize(self):
        """Initialize the market regime service"""
        logger.info("Initializing Market Regime Service...")
        self.is_initialized = True
        logger.info("Market Regime Service initialized")

    async def shutdown(self):
        """Shutdown the market regime service"""
        logger.info("Market Regime Service shut down")
        self.is_initialized = False

    async def detect_regime(self, symbol: str, price_data: List[float]) -> Dict[str, Any]:
        """Detect current market regime"""
        if not self.is_initialized:
            raise RuntimeError("Market regime service not initialized")

        # Convert to list if numpy array
        if hasattr(price_data, 'tolist'):
            price_data = price_data.tolist()

        if not price_data:
            return {"regime": "unknown", "confidence": 0.0}

        # Calculate basic metrics
        returns = np.diff(price_data) / np.array(price_data[:-1])
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility

        # Simple regime classification based on volatility and trend
        if volatility > 0.8:
            regime = "high_volatility"
        elif volatility < 0.2:
            regime = "low_volatility"
        else:
            # Trend-based classification
            trend = np.mean(returns[-20:])  # Last 20 periods
            if trend > 0.001:
                regime = "bull_market"
            elif trend < -0.001:
                regime = "bear_market"
            else:
                regime = "sideways"

        # Calculate confidence based on consistency
        recent_regime = self._calculate_recent_regime_consistency(price_data)
        confidence = min(recent_regime, 0.9)  # Cap at 90%

        result = {
            "regime": regime,
            "confidence": confidence,
            "volatility": volatility,
            "trend": trend if 'trend' in locals() else 0.0,
            "timestamp": datetime.now().isoformat()
        }

        # Store current regime
        self.current_regimes[symbol] = result

        return result

    def _calculate_recent_regime_consistency(self, price_data: List[float]) -> float:
        """Calculate how consistent the recent regime has been"""
        if len(price_data) < 20:
            return 0.5

        # Calculate rolling volatility over last 20 periods
        returns = np.diff(price_data[-21:]) / np.array(price_data[-21:-1])
        rolling_volatility = np.std(returns)

        # Higher consistency = lower volatility variation
        consistency = 1.0 / (1.0 + rolling_volatility * 10)
        return consistency

    async def get_regime_history(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical regime detections for a symbol"""
        if not self.is_initialized:
            raise RuntimeError("Market regime service not initialized")

        # Placeholder - in real implementation, this would query a database
        # For now, return current regime repeated
        current_regime = self.current_regimes.get(symbol, {"regime": "unknown", "confidence": 0.0})
        history = [current_regime.copy() for _ in range(min(limit, 10))]

        # Add timestamps
        base_time = datetime.now()
        for i, regime in enumerate(history):
            regime["timestamp"] = (base_time.replace(hour=i, minute=0, second=0, microsecond=0)).isoformat()

        return history

    async def analyze_regime_transition(self, symbol: str, old_regime: str, new_regime: str) -> Dict[str, Any]:
        """Analyze the significance of a regime transition"""
        if not self.is_initialized:
            raise RuntimeError("Market regime service not initialized")

        # Define transition significance matrix
        significance_matrix = {
            ("bull_market", "bear_market"): "high",
            ("bear_market", "bull_market"): "high",
            ("sideways", "bull_market"): "medium",
            ("sideways", "bear_market"): "medium",
            ("bull_market", "sideways"): "low",
            ("bear_market", "sideways"): "low",
            ("high_volatility", "low_volatility"): "medium",
            ("low_volatility", "high_volatility"): "medium",
        }

        transition_key = (old_regime, new_regime)
        significance = significance_matrix.get(transition_key, "unknown")

        return {
            "symbol": symbol,
            "old_regime": old_regime,
            "new_regime": new_regime,
            "significance": significance,
            "requires_attention": significance in ["high", "medium"],
            "timestamp": datetime.now().isoformat()
        }

    async def get_regime_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get statistical summary of regime behavior"""
        if not self.is_initialized:
            raise RuntimeError("Market regime service not initialized")

        # Placeholder statistics
        return {
            "symbol": symbol,
            "total_observations": 100,
            "regime_distribution": {
                "bull_market": 0.3,
                "bear_market": 0.2,
                "sideways": 0.4,
                "high_volatility": 0.05,
                "low_volatility": 0.05
            },
            "average_regime_duration": {
                "bull_market": 15.2,
                "bear_market": 12.8,
                "sideways": 8.5,
                "high_volatility": 3.2,
                "low_volatility": 25.1
            },
            "regime_transitions": 45,
            "last_updated": datetime.now().isoformat()
        }