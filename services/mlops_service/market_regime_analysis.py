#!/usr/bin/env python3
"""
Market Regime Detection and Multi-Timeframe Analysis for TradPal

This module provides advanced market analysis capabilities including:
- Market regime detection (trend, mean-reversion, volatile, etc.)
- Multi-timeframe feature engineering
- Regime-adaptive trading strategies
- Cross-timeframe signal validation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import talib
from scipy import stats
from scipy.signal import find_peaks

from services.core.gpu_accelerator import get_gpu_accelerator

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications."""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    MEAN_REVERSION = "mean_reversion"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    SIDEWAYS = "sideways"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"

@dataclass
class RegimeConfig:
    """Configuration for market regime detection."""
    lookback_periods: List[int] = None
    volatility_threshold: float = 1.5
    trend_strength_threshold: float = 0.02
    mean_reversion_threshold: float = 2.0
    consolidation_threshold: float = 0.01
    min_regime_duration: int = 20
    regime_stability_window: int = 10

    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [20, 50, 100, 200]

@dataclass
class TimeframeConfig:
    """Configuration for multi-timeframe analysis."""
    timeframes: List[str] = None
    base_timeframe: str = "1h"
    higher_timeframes: List[str] = None
    lower_timeframes: List[str] = None

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["5m", "15m", "1h", "4h", "1d"]
        if self.higher_timeframes is None:
            self.higher_timeframes = ["4h", "1d"]
        if self.lower_timeframes is None:
            self.lower_timeframes = ["5m", "15m"]

class MarketRegimeDetector:
    """Advanced market regime detection using multiple indicators."""

    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
        self.gpu_accelerator = get_gpu_accelerator()

    def detect_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect market regime for each point in time series.

        Args:
            data: OHLCV DataFrame

        Returns:
            Series of MarketRegime values
        """
        logger.info("Detecting market regimes...")

        # Calculate regime indicators
        indicators = self._calculate_regime_indicators(data)

        # Combine indicators to determine regime
        regimes = []
        for i in range(len(data)):
            regime = self._classify_regime(indicators, i)
            regimes.append(regime)

        regime_series = pd.Series(regimes, index=data.index, name='regime')
        logger.info(f"Detected {len(regime_series.unique())} different regimes")

        return regime_series

    def _calculate_regime_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate various indicators for regime detection."""
        indicators = {}

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(data))

        # Trend indicators
        for period in self.config.lookback_periods:
            if len(data) > period:
                # Linear trend slope
                indicators[f'trend_slope_{period}'] = self._calculate_trend_slope(close, period)

                # ADX (Average Directional Index)
                indicators[f'adx_{period}'] = self._calculate_adx(high, low, close, period)

                # Volatility (ATR)
                indicators[f'atr_{period}'] = self._calculate_atr(high, low, close, period)

                # RSI for mean reversion signals
                indicators[f'rsi_{period}'] = talib.RSI(close, timeperiod=min(period, len(close)-1))

                # Bollinger Band position
                indicators[f'bb_position_{period}'] = self._calculate_bb_position(close, period)

        # Volume indicators
        indicators['volume_sma_ratio'] = self._calculate_volume_ratio(volume, 20)
        indicators['volume_volatility'] = self._calculate_volume_volatility(volume, 20)

        # Price action indicators
        indicators['price_acceleration'] = self._calculate_price_acceleration(close)
        indicators['support_resistance'] = self._calculate_support_resistance(close)

        return indicators

    def _classify_regime(self, indicators: Dict[str, pd.Series], index: int) -> MarketRegime:
        """Classify market regime based on indicators."""
        # Get current values (handle edge cases)
        try:
            trend_slope = indicators.get('trend_slope_50', pd.Series([0]))[index]
            adx = indicators.get('adx_50', pd.Series([20]))[index]
            atr = indicators.get('atr_50', pd.Series([0.01]))[index]
            rsi = indicators.get('rsi_50', pd.Series([50]))[index]
            bb_pos = indicators.get('bb_position_50', pd.Series([0]))[index]
            vol_ratio = indicators.get('volume_sma_ratio', pd.Series([1]))[index]
            price_acc = indicators.get('price_acceleration', pd.Series([0]))[index]
        except (IndexError, KeyError):
            return MarketRegime.SIDEWAYS

        # High volatility regime
        if atr > self.config.volatility_threshold * np.mean(list(indicators.get('atr_50', [0.01]))):
            return MarketRegime.HIGH_VOLATILITY

        # Strong trend regimes
        if abs(trend_slope) > self.config.trend_strength_threshold and adx > 25:
            if trend_slope > 0:
                return MarketRegime.TREND_UP
            else:
                return MarketRegime.TREND_DOWN

        # Mean reversion regime
        if abs(bb_pos) > 0.8 and rsi > 70 or rsi < 30:
            return MarketRegime.MEAN_REVERSION

        # Breakout regime
        if abs(price_acc) > 0.05 and vol_ratio > 1.5:
            return MarketRegime.BREAKOUT

        # Consolidation/sideways regime
        if abs(trend_slope) < self.config.consolidation_threshold and adx < 20:
            return MarketRegime.CONSOLIDATION

        # Low volatility regime
        if atr < 0.5 * np.mean(list(indicators.get('atr_50', [0.01]))):
            return MarketRegime.LOW_VOLATILITY

        # Default to sideways
        return MarketRegime.SIDEWAYS

    def _calculate_trend_slope(self, prices: np.ndarray, period: int) -> pd.Series:
        """Calculate linear trend slope over rolling windows."""
        slopes = []
        for i in range(len(prices)):
            if i >= period - 1:
                x = np.arange(period)
                y = prices[i-period+1:i+1]
                if len(y) == len(x):
                    slope, _ = np.polyfit(x, y, 1)
                    slopes.append(slope)
                else:
                    slopes.append(0)
            else:
                slopes.append(0)
        return pd.Series(slopes, name=f'trend_slope_{period}')

    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> pd.Series:
        """Calculate Average Directional Index."""
        try:
            adx = talib.ADX(high, low, close, timeperiod=period)
            return pd.Series(adx, name=f'adx_{period}')
        except:
            return pd.Series([20] * len(high), name=f'adx_{period}')

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> pd.Series:
        """Calculate Average True Range."""
        try:
            atr = talib.ATR(high, low, close, timeperiod=period)
            return pd.Series(atr, name=f'atr_{period}')
        except:
            return pd.Series([0.01] * len(high), name=f'atr_{period}')

    def _calculate_bb_position(self, prices: np.ndarray, period: int) -> pd.Series:
        """Calculate Bollinger Band position."""
        try:
            upper, middle, lower = talib.BBANDS(prices, timeperiod=period, nbdevup=2, nbdevdn=2, matype=0)

            # Handle potential NaN or inf values
            bb_pos = (prices - lower) / (upper - lower)

            # Replace inf and -inf with 0
            bb_pos = np.where(np.isinf(bb_pos), 0, bb_pos)

            # Replace NaN with 0
            bb_pos = np.where(np.isnan(bb_pos), 0, bb_pos)

            # Clip to [-1, 1] range
            bb_pos = np.clip(bb_pos, -1, 1)

            return pd.Series(bb_pos, name=f'bb_position_{period}')
        except:
            return pd.Series([0] * len(prices), name=f'bb_position_{period}')

    def _calculate_volume_ratio(self, volume: np.ndarray, period: int) -> pd.Series:
        """Calculate volume ratio vs moving average."""
        if len(volume) < period:
            return pd.Series([1] * len(volume))

        volume_sma = pd.Series(volume).rolling(period).mean()
        ratio = volume / volume_sma
        return ratio.fillna(1)

    def _calculate_volume_volatility(self, volume: np.ndarray, period: int) -> pd.Series:
        """Calculate volume volatility."""
        if len(volume) < period:
            return pd.Series([0] * len(volume))

        volume_std = pd.Series(volume).rolling(period).std()
        volume_mean = pd.Series(volume).rolling(period).mean()
        vol_volatility = volume_std / volume_mean
        return vol_volatility.fillna(0)

    def _calculate_price_acceleration(self, prices: np.ndarray) -> pd.Series:
        """Calculate price acceleration (second derivative)."""
        returns = np.diff(prices) / prices[:-1]
        acceleration = np.diff(returns)
        acceleration = np.concatenate([[0, 0], acceleration])  # Pad to match length
        return pd.Series(acceleration, name='price_acceleration')

    def _calculate_support_resistance(self, prices: np.ndarray) -> pd.Series:
        """Calculate distance to nearest support/resistance levels."""
        # Simple implementation - find local peaks and valleys
        peaks, _ = find_peaks(prices, distance=20)
        valleys, _ = find_peaks(-prices, distance=20)

        sr_levels = np.concatenate([prices[peaks], prices[valleys]])
        sr_levels = np.sort(sr_levels)

        distances = []
        for price in prices:
            nearest_sr = sr_levels[np.argmin(np.abs(sr_levels - price))]
            distance = abs(price - nearest_sr) / price
            distances.append(distance)

        return pd.Series(distances, name='support_resistance')

class MultiTimeframeAnalyzer:
    """Multi-timeframe analysis for enhanced trading signals."""

    def __init__(self, config: TimeframeConfig = None):
        self.config = config or TimeframeConfig()
        self.regime_detector = MarketRegimeDetector()

    def analyze_multi_timeframe(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform multi-timeframe analysis.

        Args:
            data_dict: Dictionary of DataFrames for different timeframes

        Returns:
            Analysis results with cross-timeframe signals
        """
        logger.info("Performing multi-timeframe analysis...")

        # Detect regimes for each timeframe
        regime_results = {}
        for tf, data in data_dict.items():
            regime_results[tf] = self.regime_detector.detect_regime(data)

        # Calculate cross-timeframe alignment
        alignment_score = self._calculate_regime_alignment(regime_results)

        # Generate multi-timeframe features
        mtf_features = self._create_multi_timeframe_features(data_dict)

        # Calculate timeframe strength
        tf_strength = self._calculate_timeframe_strength(data_dict)

        results = {
            'regime_results': regime_results,
            'alignment_score': alignment_score,
            'mtf_features': mtf_features,
            'timeframe_strength': tf_strength,
            'consensus_regime': self._get_consensus_regime(regime_results)
        }

        logger.info(f"Multi-timeframe analysis complete. Consensus regime: {results['consensus_regime']}")
        return results

    def _calculate_regime_alignment(self, regime_results: Dict[str, pd.Series]) -> float:
        """Calculate alignment score between different timeframes."""
        if len(regime_results) < 2:
            return 1.0

        # Get the most recent regime for each timeframe
        current_regimes = {}
        for tf, regimes in regime_results.items():
            if len(regimes) > 0:
                current_regimes[tf] = regimes.iloc[-1]

        # Calculate agreement ratio
        if not current_regimes:
            return 0.0

        base_regime = list(current_regimes.values())[0]
        agreements = sum(1 for regime in current_regimes.values() if regime == base_regime)
        alignment = agreements / len(current_regimes)

        return alignment

    def _create_multi_timeframe_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create features that incorporate multiple timeframes."""
        # Use base timeframe as primary
        base_data = data_dict.get(self.config.base_timeframe)
        if base_data is None:
            logger.warning(f"Base timeframe {self.config.base_timeframe} not found")
            return pd.DataFrame()

        features = pd.DataFrame(index=base_data.index)

        # Add higher timeframe features
        for htf in self.config.higher_timeframes:
            if htf in data_dict:
                htf_data = data_dict[htf]
                # Resample to base timeframe (simplified - in practice would need proper resampling)
                htf_features = self._extract_timeframe_features(htf_data, f"htf_{htf}")
                features = features.join(htf_features, how='left')

        # Add lower timeframe features
        for ltf in self.config.lower_timeframes:
            if ltf in data_dict:
                ltf_data = data_dict[ltf]
                # Aggregate to base timeframe
                ltf_features = self._extract_timeframe_features(ltf_data, f"ltf_{ltf}")
                features = features.join(ltf_features, how='left')

        return features.ffill().fillna(0)

    def _extract_timeframe_features(self, data: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Extract key features from a timeframe."""
        features = pd.DataFrame(index=data.index)

        if len(data) < 20:
            return features

        close = data['close'].values

        # Trend features
        features[f'{prefix}_sma_20'] = talib.SMA(close, timeperiod=20)
        features[f'{prefix}_sma_50'] = talib.SMA(close, timeperiod=50)
        features[f'{prefix}_trend'] = features[f'{prefix}_sma_20'] - features[f'{prefix}_sma_50']

        # Volatility features
        features[f'{prefix}_atr'] = talib.ATR(data['high'].values, data['low'].values, close, timeperiod=14)

        # Momentum features
        features[f'{prefix}_rsi'] = talib.RSI(close, timeperiod=14)
        features[f'{prefix}_macd'], features[f'{prefix}_macd_signal'], _ = talib.MACD(close)

        return features

    def _calculate_timeframe_strength(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate strength/confidence score for each timeframe."""
        strength_scores = {}

        for tf, data in data_dict.items():
            if len(data) < 50:
                strength_scores[tf] = 0.5
                continue

            # Calculate trend strength
            close = data['close'].values
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)

            trend_strength = abs(sma_20[-1] - sma_50[-1]) / sma_50[-1] if sma_50[-1] != 0 else 0

            # Calculate volatility consistency
            returns = np.diff(close) / close[:-1]
            vol_consistency = 1 / (1 + np.std(returns[-50:]))  # Lower volatility = higher consistency

            # Combine scores
            strength_scores[tf] = (trend_strength + vol_consistency) / 2

        return strength_scores

    def _get_consensus_regime(self, regime_results: Dict[str, pd.Series]) -> MarketRegime:
        """Get consensus regime across timeframes."""
        if not regime_results:
            return MarketRegime.SIDEWAYS

        # Count regime occurrences
        regime_counts = {}
        for regimes in regime_results.values():
            if len(regimes) > 0:
                current_regime = regimes.iloc[-1]
                regime_counts[current_regime] = regime_counts.get(current_regime, 0) + 1

        # Return most common regime
        if regime_counts:
            return max(regime_counts, key=regime_counts.get)

        return MarketRegime.SIDEWAYS

class AdaptiveStrategyManager:
    """Adaptive strategy management based on market regimes."""

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.strategy_configs = self._load_strategy_configs()

    def _load_strategy_configs(self) -> Dict[MarketRegime, Dict[str, Any]]:
        """Load optimal strategy configurations for each regime."""
        return {
            MarketRegime.TREND_UP: {
                'model_type': 'lstm',
                'position_size': 1.0,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'features': ['trend', 'momentum', 'volume']
            },
            MarketRegime.TREND_DOWN: {
                'model_type': 'lstm',
                'position_size': 1.0,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'features': ['trend', 'momentum', 'volume']
            },
            MarketRegime.MEAN_REVERSION: {
                'model_type': 'ensemble',
                'position_size': 0.5,
                'stop_loss': 0.02,
                'take_profit': 0.05,
                'features': ['rsi', 'bb_position', 'mean_reversion']
            },
            MarketRegime.HIGH_VOLATILITY: {
                'model_type': 'transformer',
                'position_size': 0.3,
                'stop_loss': 0.08,
                'take_profit': 0.10,
                'features': ['volatility', 'breakout', 'momentum']
            },
            MarketRegime.LOW_VOLATILITY: {
                'model_type': 'ensemble',
                'position_size': 0.7,
                'stop_loss': 0.03,
                'take_profit': 0.08,
                'features': ['mean_reversion', 'support_resistance']
            },
            MarketRegime.SIDEWAYS: {
                'model_type': 'ensemble',
                'position_size': 0.4,
                'stop_loss': 0.025,
                'take_profit': 0.04,
                'features': ['rsi', 'bb_position', 'volume']
            },
            MarketRegime.BREAKOUT: {
                'model_type': 'transformer',
                'position_size': 0.8,
                'stop_loss': 0.06,
                'take_profit': 0.12,
                'features': ['breakout', 'volume', 'momentum']
            },
            MarketRegime.CONSOLIDATION: {
                'model_type': 'ensemble',
                'position_size': 0.3,
                'stop_loss': 0.02,
                'take_profit': 0.03,
                'features': ['support_resistance', 'volume']
            }
        }

    def get_adaptive_config(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Get adaptive strategy configuration based on current market conditions.

        Args:
            data_dict: Multi-timeframe data

        Returns:
            Adaptive configuration dictionary
        """
        # Analyze multi-timeframe data
        analysis = self.mtf_analyzer.analyze_multi_timeframe(data_dict)

        # Get consensus regime
        current_regime = analysis['consensus_regime']

        # Get base configuration for regime
        base_config = self.strategy_configs.get(current_regime, self.strategy_configs[MarketRegime.SIDEWAYS])

        # Adjust based on alignment and strength
        alignment = analysis['alignment_score']
        tf_strength = analysis['timeframe_strength']

        # Modify position size based on confidence
        confidence_multiplier = alignment * np.mean(list(tf_strength.values()))
        adjusted_position_size = base_config['position_size'] * confidence_multiplier

        # Create adaptive config
        adaptive_config = base_config.copy()
        adaptive_config['position_size'] = min(adjusted_position_size, 1.0)  # Cap at 100%
        adaptive_config['current_regime'] = current_regime
        adaptive_config['confidence_score'] = confidence_multiplier
        adaptive_config['regime_analysis'] = analysis

        logger.info(f"Adaptive config for {current_regime.value}: position_size={adaptive_config['position_size']:.2f}, confidence={adaptive_config['confidence_score']:.2f}")

        return adaptive_config

# Global instances
regime_detector = MarketRegimeDetector()
mtf_analyzer = MultiTimeframeAnalyzer()
adaptive_strategy = AdaptiveStrategyManager()

def detect_market_regime(data: pd.DataFrame) -> pd.Series:
    """Convenience function for market regime detection."""
    return regime_detector.detect_regime(data)

def analyze_multi_timeframe(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Convenience function for multi-timeframe analysis."""
    return mtf_analyzer.analyze_multi_timeframe(data_dict)

def get_adaptive_strategy_config(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Convenience function for adaptive strategy configuration."""
    return adaptive_strategy.get_adaptive_config(data_dict)
