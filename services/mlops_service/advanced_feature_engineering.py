#!/usr/bin/env python3
"""
Advanced Feature Engineering for TradPal ML Models

This module provides advanced feature engineering techniques for trading predictions,
including technical indicators, statistical features, and market microstructure features.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import talib
from dataclasses import dataclass
from abc import ABC, abstractmethod

from services.core.gpu_accelerator import get_gpu_accelerator, is_gpu_available

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    include_technical_indicators: bool = True
    include_statistical_features: bool = True
    include_microstructure_features: bool = True
    include_volatility_features: bool = True
    include_momentum_features: bool = True
    include_volume_features: bool = True
    include_price_features: bool = True
    sequence_length: int = 60
    lookback_periods: List[int] = None
    scaling_method: str = 'robust'  # 'standard', 'robust', 'minmax'
    feature_selection_k: Optional[int] = None
    use_pca: bool = False
    pca_components: Optional[int] = None

    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 30, 50, 100]

class BaseFeatureEngineer(ABC):
    """Abstract base class for feature engineering."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scaler = None
        self.pca = None
        self.selected_features = None

    @abstractmethod
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw data."""
        pass

    def fit_scaler(self, data: pd.DataFrame) -> None:
        """Fit the scaler on training data."""
        if self.config.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.config.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.config.scaling_method}")

        self.scaler.fit(data)

    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale features using fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler must be fitted before scaling")
        scaled_data = self.scaler.transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

    def fit_pca(self, data: pd.DataFrame) -> None:
        """Fit PCA on training data."""
        if self.config.use_pca and self.config.pca_components:
            self.pca = PCA(n_components=self.config.pca_components)
            self.pca.fit(data)

    def apply_pca(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA transformation."""
        if self.pca is None:
            raise ValueError("PCA must be fitted before applying")
        pca_data = self.pca.transform(data)
        columns = [f'pca_{i}' for i in range(pca_data.shape[1])]
        return pd.DataFrame(pca_data, columns=columns, index=data.index)

    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = None) -> pd.DataFrame:
        """Select best features using statistical tests."""
        if k is None:
            k = self.config.feature_selection_k or min(50, X.shape[1])

        # Choose appropriate scoring function
        if y.dtype in ['int64', 'int32', 'category']:
            score_func = f_classif
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask]

        self.selected_features = selected_features.tolist()

        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

class TechnicalIndicatorFeatures(BaseFeatureEngineer):
    """Technical indicator feature engineering."""

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features."""
        features = pd.DataFrame(index=data.index)

        if not self.config.include_technical_indicators:
            return features

        # Price-based indicators
        if self.config.include_price_features:
            features = self._create_price_features(data, features)

        # Momentum indicators
        if self.config.include_momentum_features:
            features = self._create_momentum_features(data, features)

        # Volatility indicators
        if self.config.include_volatility_features:
            features = self._create_volatility_features(data, features)

        # Volume indicators
        if self.config.include_volume_features:
            features = self._create_volume_features(data, features)

        return features

    def _create_price_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features."""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        open_price = data['open'].values

        # Moving averages
        for period in self.config.lookback_periods:
            features[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
            features[f'ema_{period}'] = talib.EMA(close, timeperiod=period)

        # MACD
        macd, macdsignal, macdhist = talib.MACD(close)
        features['macd'] = macd
        features['macd_signal'] = macdsignal
        features['macd_hist'] = macdhist

        # RSI
        features['rsi_14'] = talib.RSI(close, timeperiod=14)

        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(high, low, close)
        features['stoch_k'] = slowk
        features['stoch_d'] = slowd

        # Williams %R
        features['willr'] = talib.WILLR(high, low, close)

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close)
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = (upper - lower) / middle

        # Price rate of change
        for period in [1, 5, 10, 20]:
            features[f'roc_{period}'] = talib.ROC(close, timeperiod=period)

        return features

    def _create_momentum_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-based features."""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        # Momentum
        for period in [5, 10, 14, 20]:
            features[f'momentum_{period}'] = talib.MOM(close, timeperiod=period)

        # Commodity Channel Index
        features['cci'] = talib.CCI(high, low, close, timeperiod=14)

        # Money Flow Index
        if 'volume' in data.columns:
            volume = data['volume'].values
            features['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)

        # Ultimate Oscillator
        features['ultosc'] = talib.ULTOSC(high, low, close)

        # Average Directional Movement Index
        features['adx'] = talib.ADX(high, low, close, timeperiod=14)

        return features

    def _create_volatility_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features."""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        # Average True Range
        features['atr'] = talib.ATR(high, low, close, timeperiod=14)

        # Normalized Average True Range
        features['natr'] = talib.NATR(high, low, close, timeperiod=14)

        # True Range
        features['trange'] = talib.TRANGE(high, low, close)

        # Standard deviation of returns
        for period in self.config.lookback_periods:
            returns = np.log(close[1:]) - np.log(close[:-1])
            # Pad with NaN to match original length
            volatility = pd.Series(returns).rolling(period).std()
            padded_volatility = pd.Series([np.nan] + volatility.tolist(), index=data.index)
            features[f'volatility_{period}'] = padded_volatility

        return features

    def _create_volume_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        if 'volume' not in data.columns:
            return features

        volume = data['volume'].values
        close = data['close'].values

        # Volume moving averages
        for period in self.config.lookback_periods:
            features[f'volume_sma_{period}'] = talib.SMA(volume, timeperiod=period)

        # On Balance Volume
        features['obv'] = talib.OBV(close, volume)

        # Chaikin A/D Line
        high = data['high'].values
        low = data['low'].values
        features['ad'] = talib.AD(high, low, close, volume)

        # Volume Rate of Change
        for period in [5, 10, 20]:
            features[f'volume_roc_{period}'] = talib.ROC(volume, timeperiod=period)

        return features

class StatisticalFeatures(BaseFeatureEngineer):
    """Statistical feature engineering."""

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        features = pd.DataFrame(index=data.index)

        if not self.config.include_statistical_features:
            return features

        # Rolling statistics for price
        price_cols = ['close', 'high', 'low', 'open']
        for col in price_cols:
            if col in data.columns:
                features = self._create_rolling_stats(data[col], features, col)

        # Rolling statistics for volume
        if 'volume' in data.columns:
            features = self._create_rolling_stats(data['volume'], features, 'volume')

        # Price distribution features
        features = self._create_distribution_features(data, features)

        # Correlation features
        features = self._create_correlation_features(data, features)

        return features

    def _create_rolling_stats(self, series: pd.Series, features: pd.DataFrame,
                            prefix: str) -> pd.DataFrame:
        """Create rolling statistical features."""
        for period in self.config.lookback_periods:
            # Basic statistics
            features[f'{prefix}_mean_{period}'] = series.rolling(period).mean()
            features[f'{prefix}_std_{period}'] = series.rolling(period).std()
            features[f'{prefix}_skew_{period}'] = series.rolling(period).skew()
            features[f'{prefix}_kurt_{period}'] = series.rolling(period).kurt()

            # Quantiles
            features[f'{prefix}_q25_{period}'] = series.rolling(period).quantile(0.25)
            features[f'{prefix}_q75_{period}'] = series.rolling(period).quantile(0.75)
            features[f'{prefix}_iqr_{period}'] = (series.rolling(period).quantile(0.75) -
                                                series.rolling(period).quantile(0.25))

            # Min/Max
            features[f'{prefix}_min_{period}'] = series.rolling(period).min()
            features[f'{prefix}_max_{period}'] = series.rolling(period).max()
            features[f'{prefix}_range_{period}'] = (series.rolling(period).max() -
                                                   series.rolling(period).min())

        return features

    def _create_distribution_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Create distribution-based features."""
        if 'close' not in data.columns:
            return features

        close = data['close']

        # Price levels relative to recent range
        for period in [20, 50, 100]:
            rolling_min = close.rolling(period).min()
            rolling_max = close.rolling(period).max()
            features[f'price_level_{period}'] = (close - rolling_min) / (rolling_max - rolling_min)

        # Distance from moving averages
        for period in [20, 50, 100]:
            ma = close.rolling(period).mean()
            features[f'dist_from_ma_{period}'] = (close - ma) / ma

        return features

    def _create_correlation_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Create correlation-based features."""
        if len(data.columns) < 2:
            return features

        # Rolling correlations between price and volume
        if 'close' in data.columns and 'volume' in data.columns:
            for period in [20, 50, 100]:
                corr = data['close'].rolling(period).corr(data['volume'])
                features[f'price_volume_corr_{period}'] = corr

        # Autocorrelation of returns
        if 'close' in data.columns:
            returns = data['close'].pct_change()
            for lag in [1, 5, 10, 20]:
                autocorr = returns.rolling(100).corr(returns.shift(lag))
                features[f'returns_autocorr_lag_{lag}'] = autocorr

        return features

class MicrostructureFeatures(BaseFeatureEngineer):
    """Market microstructure feature engineering."""

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create microstructure features."""
        features = pd.DataFrame(index=data.index)

        if not self.config.include_microstructure_features:
            return features

        # Order flow features
        features = self._create_order_flow_features(data, features)

        # Liquidity features
        features = self._create_liquidity_features(data, features)

        # Trade intensity features
        features = self._create_trade_intensity_features(data, features)

        return features

    def _create_order_flow_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Create order flow features."""
        if not all(col in data.columns for col in ['high', 'low', 'close', 'open']):
            return features

        # Bid-ask spread proxy (high-low range)
        features['spread_proxy'] = (data['high'] - data['low']) / data['close']

        # Realized volatility
        returns = data['close'].pct_change()
        features['realized_vol'] = returns.rolling(20).std() * np.sqrt(252)

        # Jump detection
        features = self._detect_jumps(data, features)

        return features

    def _create_liquidity_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Create liquidity features."""
        if 'volume' not in data.columns:
            return features

        # Volume-based liquidity measures
        features['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

        # Turnover ratio (if market cap available)
        # This would need additional data

        # Amihud illiquidity measure
        returns = abs(data['close'].pct_change())
        dollar_volume = data['close'] * data['volume']
        features['amihud_illiquidity'] = returns / dollar_volume

        return features

    def _create_trade_intensity_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Create trade intensity features."""
        if 'volume' not in data.columns:
            return features

        # Volume spikes
        volume_ma = data['volume'].rolling(20).mean()
        volume_std = data['volume'].rolling(20).std()
        features['volume_spike'] = (data['volume'] - volume_ma) / volume_std

        # Price impact
        returns = data['close'].pct_change()
        features['price_impact'] = returns / data['volume']

        return features

    def _detect_jumps(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Detect price jumps."""
        returns = data['close'].pct_change()

        # Simple jump detection based on threshold
        threshold = returns.std() * 3  # 3 standard deviations
        features['price_jump'] = (abs(returns) > threshold).astype(int)

        # Jump magnitude
        features['jump_magnitude'] = returns.where(abs(returns) > threshold, 0)

        return features

class AdvancedFeatureEngineer:
    """Main feature engineering orchestrator."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.engineers = [
            TechnicalIndicatorFeatures(config),
            StatisticalFeatures(config),
            MicrostructureFeatures(config)
        ]

    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all features from raw data."""
        all_features = pd.DataFrame(index=data.index)

        for engineer in self.engineers:
            features = engineer.create_features(data)
            all_features = pd.concat([all_features, features], axis=1)

        # Remove NaN values
        all_features = all_features.dropna()

        logger.info(f"Created {all_features.shape[1]} features from {data.shape[0]} data points")

        return all_features

    def preprocess_features(self, features: pd.DataFrame, y: Optional[pd.Series] = None,
                          fit: bool = True) -> pd.DataFrame:
        """Preprocess features with scaling, PCA, and selection."""
        processed_features = features.copy()

        # Feature selection
        if y is not None and self.config.feature_selection_k:
            processed_features = self._select_features(processed_features, y)

        # Scaling
        if fit:
            self._fit_scaler(processed_features)
        processed_features = self._scale_features(processed_features)

        # PCA
        if self.config.use_pca and self.config.pca_components:
            if fit:
                self._fit_pca(processed_features)
            processed_features = self._apply_pca(processed_features)

        return processed_features

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select best features."""
        selector = SelectKBest(score_func=f_regression, k=self.config.feature_selection_k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    def _fit_scaler(self, data: pd.DataFrame) -> None:
        """Fit scaler on data."""
        if data.empty or data.shape[0] == 0:
            logger.warning("Cannot fit scaler on empty data")
            self.scaler = None
            return

        if self.config.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.config.scaling_method == 'robust':
            self.scaler = RobustScaler()
        self.scaler.fit(data)

    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale features."""
        if self.scaler is None:
            logger.warning("Scaler not fitted, returning unscaled data")
            return data
        scaled_data = self.scaler.transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

    def _fit_pca(self, data: pd.DataFrame) -> None:
        """Fit PCA."""
        self.pca = PCA(n_components=self.config.pca_components)
        self.pca.fit(data)

    def _apply_pca(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA."""
        if self.pca is None:
            raise ValueError("PCA must be fitted before applying")
        pca_data = self.pca.transform(data)
        columns = [f'pca_{i}' for i in range(pca_data.shape[1])]
        return pd.DataFrame(pca_data, columns=columns, index=data.index)

# Global feature engineer instance
def create_feature_engineer(config: FeatureConfig) -> AdvancedFeatureEngineer:
    """Create a feature engineer instance."""
    return AdvancedFeatureEngineer(config)

def create_default_feature_config() -> FeatureConfig:
    """Create default feature configuration."""
    return FeatureConfig(
        include_technical_indicators=True,
        include_statistical_features=True,
        include_microstructure_features=True,
        include_volatility_features=True,
        include_momentum_features=True,
        include_volume_features=True,
        include_price_features=True,
        sequence_length=60,
        lookback_periods=[5, 10, 20, 30, 50, 100],
        scaling_method='robust',
        feature_selection_k=50,
        use_pca=False,
        pca_components=None
    )