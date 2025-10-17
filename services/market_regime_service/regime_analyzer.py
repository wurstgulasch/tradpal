"""
Market Regime Analyzer - Interprets clustering results and generates trading signals
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from .clustering_engine import MarketRegime, MarketRegimeAnalysis, ClusteringEngine

logger = logging.getLogger(__name__)


class TradingSignal(Enum):
    """Trading signals based on market regime."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE_RISK = "reduce_risk"
    INCREASE_RISK = "increase_risk"


class RiskLevel(Enum):
    """Risk management levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RegimeSignal:
    """Trading signal based on market regime analysis."""
    timestamp: datetime
    symbol: str
    regime: MarketRegime
    signal: TradingSignal
    confidence: float
    risk_level: RiskLevel
    position_size_multiplier: float  # 0.0 to 2.0
    stop_loss_multiplier: float  # 1.0 to 3.0
    take_profit_multiplier: float  # 1.0 to 2.0
    reasoning: str


@dataclass
class RegimeTransition:
    """Market regime transition event."""
    timestamp: datetime
    symbol: str
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_confidence: float
    duration_in_previous_regime: timedelta
    expected_duration_in_new_regime: timedelta
    trading_implications: str


class RegimeAnalyzer:
    """Analyzes market regime clustering results and generates trading signals."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.clustering_engine = ClusteringEngine(self.config.get('clustering', {}))
        self.logger = logging.getLogger(__name__)

        # Regime transition history for context
        self.regime_history: Dict[str, List[Tuple[datetime, MarketRegime]]] = {}

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for regime analyzer."""
        return {
            'clustering': {
                'algorithms': ['kmeans', 'dbscan'],
                'kmeans': {'n_clusters_range': [3, 4, 5]},
                'dbscan': {'eps_range': [0.3, 0.5], 'min_samples_range': [5, 10]}
            },
            'signal_thresholds': {
                'min_confidence': 0.6,
                'high_confidence': 0.8,
                'extreme_confidence': 0.9
            },
            'risk_management': {
                'bull_market': {'position_size': 1.2, 'stop_loss': 1.5, 'take_profit': 2.0},
                'bear_market': {'position_size': 0.3, 'stop_loss': 2.0, 'take_profit': 1.2},
                'sideways': {'position_size': 0.8, 'stop_loss': 1.2, 'take_profit': 1.5},
                'high_volatility': {'position_size': 0.5, 'stop_loss': 2.5, 'take_profit': 1.8},
                'low_volatility': {'position_size': 1.0, 'stop_loss': 1.0, 'take_profit': 1.3}
            },
            'regime_expectations': {
                'bull_market': {'avg_duration_days': 150, 'volatility': 'medium'},
                'bear_market': {'avg_duration_days': 80, 'volatility': 'high'},
                'sideways': {'avg_duration_days': 45, 'volatility': 'low'},
                'high_volatility': {'avg_duration_days': 25, 'volatility': 'extreme'},
                'low_volatility': {'avg_duration_days': 60, 'volatility': 'low'}
            }
        }

    def analyze_market_regime(self, data: pd.DataFrame, symbol: str) -> Tuple[MarketRegimeAnalysis, RegimeSignal]:
        """
        Perform complete market regime analysis and generate trading signal.

        Args:
            data: OHLCV data with technical indicators
            symbol: Trading symbol

        Returns:
            Tuple of (regime_analysis, trading_signal)
        """
        self.logger.info(f"Analyzing market regime for {symbol}")

        # Perform clustering analysis
        regime_analysis = self.clustering_engine.detect_regime(data, symbol)

        # Generate trading signal based on regime
        signal = self._generate_signal(regime_analysis, symbol)

        # Update regime history
        self._update_regime_history(symbol, regime_analysis.timestamp, regime_analysis.current_regime)

        self.logger.info(f"Regime analysis completed for {symbol}: {regime_analysis.current_regime.value} -> {signal.signal.value}")
        return regime_analysis, signal

    def _generate_signal(self, analysis: MarketRegimeAnalysis, symbol: str) -> RegimeSignal:
        """Generate trading signal based on regime analysis."""
        regime = analysis.current_regime
        confidence = analysis.regime_confidence

        # Get regime-specific parameters
        regime_config = self.config['risk_management'].get(regime.value, self.config['risk_management']['sideways'])

        # Determine signal based on regime
        signal, reasoning = self._regime_to_signal(regime, confidence, analysis)

        # Determine risk level
        risk_level = self._calculate_risk_level(regime, confidence, analysis)

        # Adjust parameters based on confidence
        confidence_multiplier = min(confidence / 0.7, 1.5)  # Scale up to 1.5x for high confidence

        signal_obj = RegimeSignal(
            timestamp=analysis.timestamp,
            symbol=symbol,
            regime=regime,
            signal=signal,
            confidence=confidence,
            risk_level=risk_level,
            position_size_multiplier=regime_config['position_size'] * confidence_multiplier,
            stop_loss_multiplier=regime_config['stop_loss'],
            take_profit_multiplier=regime_config['take_profit'],
            reasoning=reasoning
        )

        return signal_obj

    def _regime_to_signal(self, regime: MarketRegime, confidence: float,
                          analysis: MarketRegimeAnalysis) -> Tuple[TradingSignal, str]:
        """Convert market regime to trading signal with reasoning."""

        thresholds = self.config['signal_thresholds']

        if confidence < thresholds['min_confidence']:
            return TradingSignal.HOLD, f"Low confidence ({confidence:.2f}) in {regime.value} regime - maintain current positions"

        if regime == MarketRegime.BULL_MARKET:
            if confidence >= thresholds['high_confidence']:
                return TradingSignal.BUY, f"Strong bull market detected with high confidence ({confidence:.2f}) - increase long positions"
            else:
                return TradingSignal.INCREASE_RISK, f"Bull market detected with moderate confidence ({confidence:.2f}) - gradually increase exposure"

        elif regime == MarketRegime.BEAR_MARKET:
            if confidence >= thresholds['high_confidence']:
                return TradingSignal.SELL, f"Strong bear market detected with high confidence ({confidence:.2f}) - reduce long positions"
            else:
                return TradingSignal.REDUCE_RISK, f"Bear market detected with moderate confidence ({confidence:.2f}) - reduce risk exposure"

        elif regime == MarketRegime.SIDEWAYS:
            return TradingSignal.HOLD, f"Sideways market detected - maintain positions but reduce leverage"

        elif regime == MarketRegime.HIGH_VOLATILITY:
            return TradingSignal.REDUCE_RISK, f"High volatility regime detected - implement strict risk management"

        elif regime == MarketRegime.LOW_VOLATILITY:
            if analysis.trend_strength > 0.6:
                return TradingSignal.BUY, f"Low volatility with strong trend - favorable for directional trades"
            else:
                return TradingSignal.HOLD, f"Low volatility, weak trend - wait for better setup"

        # Default fallback
        return TradingSignal.HOLD, f"Unable to determine clear signal for {regime.value} regime"

    def _calculate_risk_level(self, regime: MarketRegime, confidence: float,
                             analysis: MarketRegimeAnalysis) -> RiskLevel:
        """Calculate appropriate risk level based on regime and conditions."""

        # Base risk from regime
        regime_risks = {
            MarketRegime.BULL_MARKET: RiskLevel.MEDIUM,
            MarketRegime.BEAR_MARKET: RiskLevel.HIGH,
            MarketRegime.SIDEWAYS: RiskLevel.MEDIUM,
            MarketRegime.HIGH_VOLATILITY: RiskLevel.EXTREME,
            MarketRegime.LOW_VOLATILITY: RiskLevel.LOW
        }

        base_risk = regime_risks.get(regime, RiskLevel.MEDIUM)

        # Adjust based on confidence and volatility
        if confidence < 0.6:
            # Low confidence increases risk
            if base_risk == RiskLevel.LOW:
                return RiskLevel.MEDIUM
            elif base_risk == RiskLevel.MEDIUM:
                return RiskLevel.HIGH
            else:
                return RiskLevel.EXTREME

        if analysis.volatility_regime == "high_volatility":
            # High volatility increases risk
            if base_risk in [RiskLevel.LOW, RiskLevel.MEDIUM]:
                return RiskLevel.HIGH
            else:
                return RiskLevel.EXTREME

        return base_risk

    def detect_regime_transition(self, symbol: str, new_regime: MarketRegime,
                               timestamp: datetime) -> Optional[RegimeTransition]:
        """
        Detect if a regime transition has occurred.

        Args:
            symbol: Trading symbol
            new_regime: Newly detected regime
            timestamp: Detection timestamp

        Returns:
            RegimeTransition if transition detected, None otherwise
        """
        if symbol not in self.regime_history:
            return None

        history = self.regime_history[symbol]
        if not history:
            return None

        # Get previous regime
        prev_timestamp, prev_regime = history[-1]

        if prev_regime != new_regime:
            # Transition detected
            duration_prev = timestamp - prev_timestamp

            # Estimate expected duration for new regime
            expectations = self.config['regime_expectations'].get(new_regime.value, {})
            expected_duration = timedelta(days=expectations.get('avg_duration_days', 30))

            # Generate trading implications
            implications = self._get_transition_implications(prev_regime, new_regime)

            transition = RegimeTransition(
                timestamp=timestamp,
                symbol=symbol,
                from_regime=prev_regime,
                to_regime=new_regime,
                transition_confidence=0.8,  # Could be calculated more sophisticatedly
                duration_in_previous_regime=duration_prev,
                expected_duration_in_new_regime=expected_duration,
                trading_implications=implications
            )

            return transition

        return None

    def _get_transition_implications(self, from_regime: MarketRegime, to_regime: MarketRegime) -> str:
        """Generate trading implications for regime transitions."""

        transitions = {
            (MarketRegime.BULL_MARKET, MarketRegime.BEAR_MARKET): "Bull to bear transition - immediate risk reduction required",
            (MarketRegime.BEAR_MARKET, MarketRegime.BULL_MARKET): "Bear to bull transition - opportunity for long positions",
            (MarketRegime.SIDEWAYS, MarketRegime.BULL_MARKET): "Breaking out of sideways - monitor for trend continuation",
            (MarketRegime.SIDEWAYS, MarketRegime.BEAR_MARKET): "Breaking down from sideways - reduce exposure",
            (MarketRegime.HIGH_VOLATILITY, MarketRegime.LOW_VOLATILITY): "Volatility contraction - favorable for directional trades",
            (MarketRegime.LOW_VOLATILITY, MarketRegime.HIGH_VOLATILITY): "Volatility expansion - implement strict risk controls",
        }

        return transitions.get((from_regime, to_regime), f"Transition from {from_regime.value} to {to_regime.value}")

    def _update_regime_history(self, symbol: str, timestamp: datetime, regime: MarketRegime):
        """Update the regime history for a symbol."""
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []

        # Keep only recent history (last 100 entries)
        self.regime_history[symbol].append((timestamp, regime))
        if len(self.regime_history[symbol]) > 100:
            self.regime_history[symbol] = self.regime_history[symbol][-100:]

    def get_regime_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get statistical summary of regime history for a symbol."""
        if symbol not in self.regime_history:
            return {"error": "No regime history available for symbol"}

        history = self.regime_history[symbol]
        if not history:
            return {"error": "Empty regime history"}

        # Calculate regime frequencies
        regime_counts = {}
        total_duration = timedelta(0)

        for i, (timestamp, regime) in enumerate(history):
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

            if i > 0:
                duration = timestamp - history[i-1][0]
                total_duration += duration

        # Calculate average duration per regime
        avg_duration_per_regime = {}
        for regime in regime_counts:
            if total_duration.total_seconds() > 0:
                avg_duration_per_regime[regime.value] = (
                    total_duration.total_seconds() / regime_counts[regime]
                ) / 86400  # Convert to days

        return {
            "total_observations": len(history),
            "regime_frequencies": {r.value: count for r, count in regime_counts.items()},
            "most_common_regime": max(regime_counts.items(), key=lambda x: x[1])[0].value if regime_counts else None,
            "avg_duration_days": avg_duration_per_regime,
            "observation_period_days": total_duration.days if history else 0
        }

    def get_regime_prediction(self, symbol: str, horizon_days: int = 30) -> Dict[str, Any]:
        """
        Predict future regime based on historical patterns.

        This is a simplified prediction - in practice would use ML models.
        """
        if symbol not in self.regime_history:
            return {"error": "No regime history available for prediction"}

        history = self.regime_history[symbol]
        if len(history) < 5:
            return {"error": "Insufficient history for prediction"}

        # Simple prediction based on recent trend and regime expectations
        recent_regimes = [regime for _, regime in history[-5:]]
        current_regime = recent_regimes[-1]

        # Get expected duration for current regime
        expectations = self.config['regime_expectations'].get(current_regime.value, {})
        expected_days = expectations.get('avg_duration_days', 30)

        # Calculate how long we've been in current regime
        current_start = None
        for i in range(len(history) - 1, -1, -1):
            if history[i][1] != current_regime:
                current_start = history[i+1][0] if i+1 < len(history) else history[0][0]
                break

        if current_start is None:
            current_start = history[0][0]

        days_in_regime = (datetime.now() - current_start).days

        # Predict transition probability
        if days_in_regime > expected_days:
            transition_probability = min((days_in_regime - expected_days) / expected_days, 0.8)
        else:
            transition_probability = 0.2

        # Predict most likely next regime based on historical transitions
        next_regime_probs = self._calculate_transition_probabilities(history)

        return {
            "current_regime": current_regime.value,
            "days_in_current_regime": days_in_regime,
            "expected_duration_days": expected_days,
            "transition_probability": transition_probability,
            "predicted_next_regimes": next_regime_probs,
            "horizon_days": horizon_days
        }

    def _calculate_transition_probabilities(self, history: List[Tuple[datetime, MarketRegime]]) -> Dict[str, float]:
        """Calculate transition probabilities from historical data."""
        if len(history) < 2:
            return {}

        transitions = {}
        total_transitions = 0

        for i in range(1, len(history)):
            prev_regime = history[i-1][1]
            curr_regime = history[i][1]

            if prev_regime != curr_regime:
                key = (prev_regime, curr_regime)
                transitions[key] = transitions.get(key, 0) + 1
                total_transitions += 1

        # Convert to probabilities
        probabilities = {}
        for (from_regime, to_regime), count in transitions.items():
            probabilities[f"{from_regime.value}_to_{to_regime.value}"] = count / total_transitions

        return probabilities