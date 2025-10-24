"""
Market Regime Detection Module for Data Service

Advanced market regime detection and analysis integrated into the data service.
Provides comprehensive regime classification, multi-timeframe analysis, and statistical modeling.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
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
    UNKNOWN = "unknown"


@dataclass
class RegimeAnalysis:
    """Comprehensive regime analysis result."""
    regime: MarketRegime
    confidence: float
    strength: float
    volatility: float
    trend_strength: float
    momentum: float
    support_resistance_alignment: float
    volume_profile: str
    timeframe_alignment: float
    statistical_significance: float
    indicators: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]


class MarketRegimeDetector:
    """
    Advanced market regime detector with statistical and technical analysis.

    Features:
    - Multi-timeframe regime analysis
    - Statistical regime classification
    - Technical indicator integration
    - Volume profile analysis
    - Confidence scoring
    """

    def __init__(self):
        self.regime_history: Dict[str, List[RegimeAnalysis]] = {}
        self.regime_thresholds = {
            'volatility_high': 0.8,
            'volatility_low': 0.2,
            'trend_strong': 0.001,
            'momentum_threshold': 0.5,
            'alignment_threshold': 0.7
        }

    async def analyze_regime(self, symbol: str, df: pd.DataFrame,
                           lookback_periods: int = 100) -> RegimeAnalysis:
        """
        Perform comprehensive market regime analysis.

        Args:
            symbol: Trading symbol
            df: OHLCV DataFrame
            lookback_periods: Number of periods to analyze

        Returns:
            Comprehensive regime analysis
        """
        if df.empty or len(df) < lookback_periods:
            return self._create_unknown_regime(symbol)

        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns for {symbol}")
            return self._create_unknown_regime(symbol)

        # Use only the lookback period
        analysis_df = df.tail(lookback_periods).copy()

        # Calculate all regime indicators
        indicators = await self._calculate_regime_indicators(analysis_df)

        # Determine primary regime
        regime, confidence = self._classify_regime(indicators)

        # Calculate additional metrics
        strength = self._calculate_regime_strength(indicators)
        statistical_sig = self._calculate_statistical_significance(analysis_df)

        # Create comprehensive analysis
        analysis = RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            strength=strength,
            volatility=indicators['volatility'],
            trend_strength=indicators['trend_strength'],
            momentum=indicators['momentum'],
            support_resistance_alignment=indicators['sr_alignment'],
            volume_profile=indicators['volume_profile'],
            timeframe_alignment=1.0,  # Single timeframe
            statistical_significance=statistical_sig,
            indicators=indicators,
            timestamp=datetime.now(),
            metadata={
                'symbol': symbol,
                'lookback_periods': lookback_periods,
                'data_points': len(analysis_df),
                'analysis_method': 'comprehensive_technical'
            }
        )

        # Store in history
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        self.regime_history[symbol].append(analysis)

        # Keep only last 100 analyses
        if len(self.regime_history[symbol]) > 100:
            self.regime_history[symbol] = self.regime_history[symbol][-100:]

        return analysis

    async def analyze_multi_timeframe_regime(self, symbol: str,
                                           timeframes: List[str],
                                           data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze regime across multiple timeframes for consensus.

        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to analyze
            data_dict: Dictionary of DataFrames per timeframe

        Returns:
            Multi-timeframe regime analysis
        """
        timeframe_analyses = {}

        # Analyze each timeframe
        for tf in timeframes:
            if tf in data_dict:
                analysis = await self.analyze_regime(symbol, data_dict[tf])
                timeframe_analyses[tf] = analysis

        if not timeframe_analyses:
            return {
                'consensus_regime': MarketRegime.UNKNOWN.value,
                'timeframe_regimes': {},
                'strength_score': 0.0,
                'alignment_score': 0.0,
                'confidence': 0.0
            }

        # Calculate consensus
        consensus_result = self._calculate_consensus_regime(timeframe_analyses)

        return {
            'consensus_regime': consensus_result['regime'].value,
            'timeframe_regimes': {tf: analysis.regime.value for tf, analysis in timeframe_analyses.items()},
            'strength_score': consensus_result['strength'],
            'alignment_score': consensus_result['alignment'],
            'confidence': consensus_result['confidence'],
            'timeframe_analyses': {tf: self._analysis_to_dict(analysis) for tf, analysis in timeframe_analyses.items()}
        }

    async def _calculate_regime_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive regime indicators."""
        indicators = {}

        # Basic price metrics
        close_prices = df['close'].values
        indicators['current_price'] = close_prices[-1]
        indicators['price_change_pct'] = (close_prices[-1] - close_prices[0]) / close_prices[0]

        # Volatility calculation (annualized)
        returns = np.diff(close_prices) / close_prices[:-1]
        indicators['volatility'] = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0

        # Trend strength using linear regression
        if len(close_prices) > 10:
            x = np.arange(len(close_prices))
            slope, _ = np.polyfit(x, close_prices, 1)
            indicators['trend_strength'] = slope / np.mean(close_prices)
        else:
            indicators['trend_strength'] = 0.0

        # Momentum indicators
        if len(close_prices) > 14:
            # RSI calculation
            rsi = self._calculate_rsi(close_prices)
            indicators['rsi'] = rsi[-1] if len(rsi) > 0 else 50.0

            # MACD
            macd, signal = self._calculate_macd(close_prices)
            indicators['macd'] = macd[-1] if len(macd) > 0 else 0.0
            indicators['macd_signal'] = signal[-1] if len(signal) > 0 else 0.0
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']

            # Momentum as rate of change
            indicators['momentum'] = (close_prices[-1] - close_prices[-14]) / close_prices[-14]
        else:
            indicators['rsi'] = 50.0
            indicators['macd'] = 0.0
            indicators['macd_signal'] = 0.0
            indicators['macd_histogram'] = 0.0
            indicators['momentum'] = 0.0

        # Volume analysis
        if 'volume' in df.columns:
            volume = df['volume'].values
            indicators['volume_trend'] = np.polyfit(np.arange(len(volume)), volume, 1)[0]
            indicators['volume_profile'] = self._analyze_volume_profile(volume, close_prices)
        else:
            indicators['volume_trend'] = 0.0
            indicators['volume_profile'] = 'unknown'

        # Support/Resistance alignment
        indicators['sr_alignment'] = self._calculate_support_resistance_alignment(df)

        # Moving averages alignment
        indicators['ma_alignment'] = self._calculate_ma_alignment(close_prices)

        return indicators

    def _classify_regime(self, indicators: Dict[str, Any]) -> Tuple[MarketRegime, float]:
        """Classify market regime based on indicators."""
        volatility = indicators['volatility']
        trend_strength = indicators['trend_strength']
        momentum = indicators['momentum']
        rsi = indicators.get('rsi', 50.0)

        # High volatility regime
        if volatility > self.regime_thresholds['volatility_high']:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(volatility / 1.0, 0.9)  # Scale confidence

        # Low volatility regime
        elif volatility < self.regime_thresholds['volatility_low']:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = min((0.5 - volatility) / 0.5, 0.9)

        # Trend-based classification
        else:
            # Strong uptrend
            if trend_strength > self.regime_thresholds['trend_strong'] and momentum > 0.001 and rsi > 55:
                regime = MarketRegime.BULL_MARKET
                confidence = min(abs(trend_strength) * 1000, 0.85)

            # Strong downtrend
            elif trend_strength < -self.regime_thresholds['trend_strong'] and momentum < -0.001 and rsi < 45:
                regime = MarketRegime.BEAR_MARKET
                confidence = min(abs(trend_strength) * 1000, 0.85)

            # Sideways/consolidation
            else:
                regime = MarketRegime.SIDEWAYS
                # Confidence based on how flat the trend is
                flatness = 1.0 - min(abs(trend_strength) * 1000, 1.0)
                confidence = min(flatness * 0.8, 0.7)

        return regime, confidence

    def _calculate_regime_strength(self, indicators: Dict[str, Any]) -> float:
        """Calculate overall regime strength score."""
        strength_factors = []

        # Volatility contribution (lower volatility = stronger regime)
        vol_strength = 1.0 - min(indicators['volatility'] / 1.0, 1.0)
        strength_factors.append(vol_strength)

        # Trend strength contribution
        trend_strength = min(abs(indicators['trend_strength']) * 1000, 1.0)
        strength_factors.append(trend_strength)

        # Momentum alignment
        momentum_alignment = abs(indicators['momentum'])
        strength_factors.append(min(momentum_alignment * 10, 1.0))

        # MA alignment
        ma_alignment = indicators.get('ma_alignment', 0.5)
        strength_factors.append(ma_alignment)

        # Average strength
        return np.mean(strength_factors)

    def _calculate_statistical_significance(self, df: pd.DataFrame) -> float:
        """Calculate statistical significance of the regime."""
        if len(df) < 30:
            return 0.5

        # Test for stationarity (ADF test approximation)
        close_prices = df['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]

        # Simple statistical tests
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # T-statistic for mean return significance
        if std_return > 0:
            t_stat = abs(mean_return) / (std_return / np.sqrt(len(returns)))
            significance = min(t_stat / 3.0, 1.0)  # Scale to 0-1
        else:
            significance = 0.5

        return significance

    def _calculate_consensus_regime(self, timeframe_analyses: Dict[str, RegimeAnalysis]) -> Dict[str, Any]:
        """Calculate consensus regime across timeframes."""
        if not timeframe_analyses:
            return {
                'regime': MarketRegime.UNKNOWN,
                'strength': 0.0,
                'alignment': 0.0,
                'confidence': 0.0
            }

        # Count regime votes
        regime_votes = {}
        total_strength = 0.0
        total_confidence = 0.0

        for analysis in timeframe_analyses.values():
            regime = analysis.regime
            if regime not in regime_votes:
                regime_votes[regime] = 0
            regime_votes[regime] += 1
            total_strength += analysis.strength
            total_confidence += analysis.confidence

        # Find consensus regime
        consensus_regime = max(regime_votes.keys(), key=lambda x: regime_votes[x])

        # Calculate alignment (how many timeframes agree)
        alignment = regime_votes[consensus_regime] / len(timeframe_analyses)

        # Average metrics
        avg_strength = total_strength / len(timeframe_analyses)
        avg_confidence = total_confidence / len(timeframe_analyses)

        return {
            'regime': consensus_regime,
            'strength': avg_strength,
            'alignment': alignment,
            'confidence': avg_confidence
        }

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return np.array([50.0])

        gains = np.maximum(np.diff(prices), 0)
        losses = np.maximum(-np.diff(prices), 0)

        avg_gain = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(losses, np.ones(period)/period, mode='valid')

        rs = avg_gain / np.maximum(avg_loss, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD indicator."""
        if len(prices) < slow:
            return np.array([0.0]), np.array([0.0])

        # Calculate EMAs
        fast_ema = self._calculate_ema(prices, fast)
        slow_ema = self._calculate_ema(prices, slow)

        # MACD line
        macd = fast_ema - slow_ema

        # Signal line
        signal_line = self._calculate_ema(macd, signal)

        return macd, signal_line

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return np.array([np.mean(data)] * len(data))

        ema = np.zeros_like(data)
        ema[period-1] = np.mean(data[:period])

        multiplier = 2 / (period + 1)
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]

        return ema

    def _analyze_volume_profile(self, volume: np.ndarray, prices: np.ndarray) -> str:
        """Analyze volume profile characteristics."""
        if len(volume) < 10:
            return 'insufficient_data'

        # Volume trend
        volume_trend = np.polyfit(np.arange(len(volume)), volume, 1)[0]

        # Price-volume correlation
        price_changes = np.diff(prices) / prices[:-1]
        volume_changes = np.diff(volume) / np.maximum(volume[:-1], 1)

        if len(price_changes) > 5 and len(volume_changes) > 5:
            correlation = np.corrcoef(price_changes[-10:], volume_changes[-10:])[0, 1]
        else:
            correlation = 0.0

        # Classify volume profile
        if volume_trend > 0.1 and correlation > 0.3:
            return 'accumulating'
        elif volume_trend < -0.1 and correlation < -0.3:
            return 'distributing'
        elif abs(volume_trend) < 0.05:
            return 'neutral'
        else:
            return 'mixed'

    def _calculate_support_resistance_alignment(self, df: pd.DataFrame) -> float:
        """Calculate alignment with support/resistance levels."""
        if len(df) < 20:
            return 0.5

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Simple support/resistance calculation
        recent_high = np.max(high[-20:])
        recent_low = np.min(low[-20:])
        current_price = close[-1]

        # Distance from levels
        range_size = recent_high - recent_low
        if range_size > 0:
            distance_from_support = (current_price - recent_low) / range_size
            distance_from_resistance = (recent_high - current_price) / range_size

            # Alignment score (closer to levels = higher alignment)
            alignment = 1.0 - min(distance_from_support, distance_from_resistance, 0.5) * 2
        else:
            alignment = 0.5

        return alignment

    def _calculate_ma_alignment(self, prices: np.ndarray) -> float:
        """Calculate moving average alignment."""
        if len(prices) < 50:
            return 0.5

        # Calculate multiple MAs
        ma20 = self._calculate_sma(prices, 20)
        ma50 = self._calculate_sma(prices, 50)

        if len(ma20) > 0 and len(ma50) > 0:
            # Check if MAs are aligned (close to each other)
            ma_diff = abs(ma20[-1] - ma50[-1])
            avg_price = np.mean(prices[-50:])
            alignment = 1.0 - min(ma_diff / avg_price, 1.0)
        else:
            alignment = 0.5

        return alignment

    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        if len(data) < period:
            return np.array([np.mean(data)] * len(data))

        weights = np.ones(period) / period
        return np.convolve(data, weights, mode='valid')

    def _create_unknown_regime(self, symbol: str) -> RegimeAnalysis:
        """Create unknown regime analysis for insufficient data."""
        return RegimeAnalysis(
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            strength=0.0,
            volatility=0.0,
            trend_strength=0.0,
            momentum=0.0,
            support_resistance_alignment=0.0,
            volume_profile='unknown',
            timeframe_alignment=0.0,
            statistical_significance=0.0,
            indicators={},
            timestamp=datetime.now(),
            metadata={'symbol': symbol, 'error': 'insufficient_data'}
        )

    def _analysis_to_dict(self, analysis: RegimeAnalysis) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            'regime': analysis.regime.value,
            'confidence': analysis.confidence,
            'strength': analysis.strength,
            'volatility': analysis.volatility,
            'trend_strength': analysis.trend_strength,
            'momentum': analysis.momentum,
            'support_resistance_alignment': analysis.support_resistance_alignment,
            'volume_profile': analysis.volume_profile,
            'timeframe_alignment': analysis.timeframe_alignment,
            'statistical_significance': analysis.statistical_significance,
            'timestamp': analysis.timestamp.isoformat(),
            'metadata': analysis.metadata
        }

    async def get_regime_history(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get historical regime analyses for a symbol."""
        if symbol not in self.regime_history:
            return []

        analyses = self.regime_history[symbol][-limit:]
        return [self._analysis_to_dict(analysis) for analysis in analyses]

    async def get_regime_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get statistical summary of regime behavior."""
        if symbol not in self.regime_history:
            return {
                'symbol': symbol,
                'total_analyses': 0,
                'regime_distribution': {},
                'average_confidence': 0.0,
                'average_strength': 0.0
            }

        analyses = self.regime_history[symbol]
        if not analyses:
            return {
                'symbol': symbol,
                'total_analyses': 0,
                'regime_distribution': {},
                'average_confidence': 0.0,
                'average_strength': 0.0
            }

        # Calculate statistics
        regime_counts = {}
        total_confidence = 0.0
        total_strength = 0.0

        for analysis in analyses:
            regime = analysis.regime
            if regime not in regime_counts:
                regime_counts[regime] = 0
            regime_counts[regime] += 1
            total_confidence += analysis.confidence
            total_strength += analysis.strength

        regime_distribution = {
            regime.value: count / len(analyses)
            for regime, count in regime_counts.items()
        }

        return {
            'symbol': symbol,
            'total_analyses': len(analyses),
            'regime_distribution': regime_distribution,
            'average_confidence': total_confidence / len(analyses),
            'average_strength': total_strength / len(analyses),
            'most_common_regime': max(regime_counts.keys(), key=lambda x: regime_counts[x]).value if regime_counts else None,
            'last_updated': analyses[-1].timestamp.isoformat() if analyses else None
        }
