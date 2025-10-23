"""
Advanced Signal Generator - ML-based signal generation with ensemble methods
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
import pickle
from pathlib import Path

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available, ML features disabled")

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not available, deep learning features disabled")

# Configure GPU/CPU
if TENSORFLOW_AVAILABLE:
    try:
        # Check for CUDA GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ CUDA GPU available, using GPU acceleration")
        else:
            print("❌ CUDA not available, falling back to CPU")
    except Exception as e:
        print(f"⚠️  GPU configuration failed: {e}, using CPU")

from config.settings import *
from services.core_service.service import CoreService

logger = logging.getLogger(__name__)

class MLEnsembleModel:
    """ML Ensemble Model for signal prediction"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        features = pd.DataFrame(index=data.index)

        # Technical indicators
        features['sma_20'] = data['close'].rolling(20).mean()
        features['sma_50'] = data['close'].rolling(50).mean()
        features['ema_12'] = data['close'].ewm(span=12).mean()
        features['ema_26'] = data['close'].ewm(span=26).mean()

        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()

        # Bollinger Bands
        sma_20 = features['sma_20']
        std_20 = data['close'].rolling(20).std()
        features['bb_upper'] = sma_20 + (std_20 * 2)
        features['bb_lower'] = sma_20 - (std_20 * 2)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20

        # ATR
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14).mean()

        # Volume indicators
        features['volume_sma'] = data['volume'].rolling(20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_sma']

        # Price momentum
        features['returns'] = data['close'].pct_change()
        features['momentum'] = data['close'] / data['close'].shift(10) - 1

        # Volatility
        features['volatility'] = data['close'].pct_change().rolling(20).std()

        # Fill NaN values
        features = features.bfill().fillna(0)

        return features

    def create_labels(self, data: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """Create target labels for prediction"""
        future_returns = data['close'].shift(-horizon) / data['close'] - 1

        # Create numeric labels directly, handling NaN values
        conditions = [
            future_returns < -0.02,
            (future_returns >= -0.02) & (future_returns < -0.005),
            (future_returns >= -0.005) & (future_returns <= 0.005),
            (future_returns > 0.005) & (future_returns <= 0.02),
            future_returns > 0.02
        ]
        choices = [0, 1, 2, 3, 4]  # strong_sell, sell, hold, buy, strong_buy

        labels = pd.Series(np.select(conditions, choices, default=2), index=data.index, dtype=int)
        return labels

    def train(self, data: pd.DataFrame) -> bool:
        """Train the ensemble model"""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, skipping ML training")
            return False

        try:
            # Prepare features and labels
            features = self.prepare_features(data)
            labels = self.create_labels(data)

            # Remove NaN values
            valid_idx = features.notna().all(axis=1) & labels.notna()
            features = features[valid_idx]
            labels = labels[valid_idx]

            if len(features) < 100:
                logger.warning("Insufficient data for training")
                return False

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train models
            self.models['rf'] = RandomForestClassifier(n_estimators=100, random_state=42)
            self.models['gb'] = GradientBoostingClassifier(n_estimators=100, random_state=42)

            if 'xgb' in globals():
                self.models['xgb'] = xgb.XGBClassifier(n_estimators=100, random_state=42)

            for name, model in self.models.items():
                model.fit(X_train_scaled, y_train)
                train_score = accuracy_score(y_train, model.predict(X_train_scaled))
                test_score = accuracy_score(y_test, model.predict(X_test_scaled))
                logger.info(f"{name.upper()} - Train: {train_score:.3f}, Test: {test_score:.3f}")

            self.feature_names = features.columns.tolist()
            self.is_trained = True

            return True

        except Exception as e:
            logger.error(f"ML training failed: {e}")
            return False

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions with ensemble model"""
        if not self.is_trained or not self.models:
            return pd.DataFrame()

        try:
            features = self.prepare_features(data)
            features_scaled = self.scaler.transform(features)

            predictions = {}
            probabilities = {}

            for name, model in self.models.items():
                pred = model.predict(features_scaled)
                prob = model.predict_proba(features_scaled)

                predictions[name] = pred
                probabilities[name] = prob

            # Ensemble voting
            ensemble_pred = []
            ensemble_conf = []

            # Label mapping for conversion back to strings
            label_map = {0: 'strong_sell', 1: 'sell', 2: 'hold', 3: 'buy', 4: 'strong_buy'}

            for i in range(len(features)):
                votes = {}
                conf_scores = []

                for name in self.models.keys():
                    pred = predictions[name][i]
                    prob = probabilities[name][i]
                    max_prob = max(prob)

                    votes[pred] = votes.get(pred, 0) + 1
                    conf_scores.append(max_prob)

                # Majority vote
                final_pred_numeric = max(votes, key=votes.get)
                final_pred = label_map.get(final_pred_numeric, 'hold')
                avg_conf = np.mean(conf_scores)

                ensemble_pred.append(final_pred)
                ensemble_conf.append(avg_conf)

            result = pd.DataFrame({
                'prediction': ensemble_pred,
                'confidence': ensemble_conf
            }, index=features.index)

            return result

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return pd.DataFrame()

class MarketRegimeDetector:
    """Detect market regimes using unsupervised learning"""

    def __init__(self):
        self.regime_model = None
        self.scaler = StandardScaler()

    def detect_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime"""
        try:
            # Calculate regime indicators
            returns = data['close'].pct_change().dropna()

            if len(returns) < 20:
                return 'unknown'

            # Volatility (20-period rolling std)
            volatility = returns.rolling(20).std().iloc[-1]

            # Trend strength (ADX-like)
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift()).abs()
            low_close = (data['low'] - data['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]

            # Trend direction
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1]
            trend = (sma_20 - sma_50) / sma_50

            # Volume analysis
            volume_ratio = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]

            # Classify regime
            if volatility > 0.03:  # High volatility
                if abs(trend) < 0.01:
                    return 'high_volatility_sideways'
                elif trend > 0.02:
                    return 'high_volatility_uptrend'
                else:
                    return 'high_volatility_downtrend'
            elif abs(trend) > 0.015:  # Strong trend
                return 'trending' if trend > 0 else 'trending_down'
            elif volume_ratio > 1.2:  # High volume
                return 'accumulation' if trend > 0 else 'distribution'
            else:
                return 'consolidation'

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return 'unknown'

class AdvancedFeatureEngineer:
    """Advanced feature engineering for signal generation"""

    def __init__(self):
        self.feature_cache = {}

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced features"""
        features = pd.DataFrame(index=data.index)

        # Basic price features
        features['close'] = data['close']
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))

        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'ema_{period}'] = data['close'].ewm(span=period).mean()

        # Momentum indicators
        features['rsi'] = self.calculate_rsi(data['close'])
        features['stoch_k'], features['stoch_d'] = self.calculate_stoch(data)
        features['williams_r'] = self.calculate_williams_r(data)

        # Trend indicators
        features['adx'] = self.calculate_adx(data)
        features['macd'], features['macd_signal'], features['macd_hist'] = self.calculate_macd(data['close'])

        # Volatility indicators
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = self.calculate_bollinger_bands(data['close'])
        features['atr'] = self.calculate_atr(data)

        # Volume indicators
        features['volume_sma'] = data['volume'].rolling(20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_sma']
        features['obv'] = self.calculate_obv(data)

        # Statistical features
        features['skewness'] = data['close'].rolling(20).skew()
        features['kurtosis'] = data['close'].rolling(20).kurt()
        features['zscore'] = (data['close'] - features['sma_20']) / data['close'].rolling(20).std()

        # Time-based features
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        features['month'] = data.index.month

        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_lag_{lag}'] = features['volume_ratio'].shift(lag)

        # Fill NaN values
        features = features.bfill().fillna(0)

        return features

    def calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_stoch(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        """Calculate Stochastic Oscillator"""
        lowest_low = data['low'].rolling(k_period).min()
        highest_high = data['high'].rolling(k_period).max()
        k = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(d_period).mean()
        return k, d

    def calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = data['high'].rolling(period).max()
        lowest_low = data['low'].rolling(period).min()
        return -100 * (highest_high - data['close']) / (highest_high - lowest_low)

    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX"""
        high = data['high']
        low = data['low']
        close = data['close']

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()

        dm_plus = (high - high.shift()).where(high - high.shift() > low.shift() - low, 0)
        dm_minus = (low.shift() - low).where(low.shift() - low > high - high.shift(), 0)

        di_plus = 100 * (dm_plus.rolling(period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(period).mean() / atr)

        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(period).mean()

        return adx

    def calculate_macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def calculate_bollinger_bands(self, close: pd.Series, period: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands"""
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high = data['high']
        low = data['low']
        close = data['close']

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        return tr.rolling(period).mean()

    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        close = data['close']
        volume = data['volume']

        # Pandas implementation (fallback)
        obv = pd.Series(0.0, index=close.index, dtype=float)
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

class AdaptiveStrategySelector:
    """Adaptive strategy selection based on market conditions"""

    def __init__(self):
        self.performance_history = {}
        self.market_regimes = {}

    def select_strategy(self, market_regime: str, available_strategies: List[str]) -> str:
        """Select best strategy for current market regime"""
        if market_regime not in self.performance_history:
            return available_strategies[0] if available_strategies else 'default'

        # Get performance scores for this regime
        regime_performance = self.performance_history[market_regime]

        # Select strategy with best performance
        best_strategy = max(regime_performance.items(), key=lambda x: x[1])
        return best_strategy[0]

    def update_performance(self, market_regime: str, strategy: str, performance_score: float):
        """Update strategy performance for market regime"""
        if market_regime not in self.performance_history:
            self.performance_history[market_regime] = {}

        self.performance_history[market_regime][strategy] = performance_score

class MultiTimeframeSignalIntegrator:
    """Integrate signals across multiple timeframes"""

    def __init__(self):
        self.timeframe_weights = {
            '1m': 0.1,
            '5m': 0.2,
            '15m': 0.3,
            '1h': 0.4
        }

    def integrate_signals(self, signals_by_timeframe: Dict[str, List[Dict]]) -> List[Dict]:
        """Integrate signals from multiple timeframes"""
        if not signals_by_timeframe:
            return []

        # Find common timestamps
        all_timestamps = set()
        for timeframe_signals in signals_by_timeframe.values():
            for signal in timeframe_signals:
                if 'timestamp' in signal:
                    all_timestamps.add(signal['timestamp'])

        integrated_signals = []

        for timestamp in sorted(all_timestamps):
            signal_votes = {'buy': 0, 'sell': 0, 'hold': 0}
            total_weight = 0
            confidence_sum = 0

            for timeframe, weight in self.timeframe_weights.items():
                if timeframe in signals_by_timeframe:
                    # Find signal for this timestamp and timeframe
                    matching_signals = [
                        s for s in signals_by_timeframe[timeframe]
                        if s.get('timestamp') == timestamp
                    ]

                    if matching_signals:
                        signal = matching_signals[0]
                        action = signal.get('action', 'hold')
                        confidence = signal.get('confidence', 0.5)

                        signal_votes[action] += weight
                        total_weight += weight
                        confidence_sum += confidence * weight

            if total_weight > 0:
                # Determine final action by weighted majority
                final_action = max(signal_votes.items(), key=lambda x: x[1])[0]
                avg_confidence = confidence_sum / total_weight

                integrated_signals.append({
                    'timestamp': timestamp,
                    'action': final_action,
                    'confidence': avg_confidence,
                    'integrated': True,
                    'timeframes_used': len([tf for tf in self.timeframe_weights.keys() if tf in signals_by_timeframe])
                })

        return integrated_signals

class AdvancedSignalGenerator:
    """Advanced ML-based signal generator with ensemble methods"""

    def __init__(self, core_service: Optional[CoreService] = None):
        self.core_service = core_service or CoreService()
        self.ml_model = MLEnsembleModel()
        self.regime_detector = MarketRegimeDetector()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.strategy_selector = AdaptiveStrategySelector()
        self.timeframe_integrator = MultiTimeframeSignalIntegrator()

        # Model persistence
        self.model_dir = Path("cache/ml_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("AdvancedSignalGenerator initialized")

    async def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate advanced signals using ML and traditional methods"""
        try:
            signals = []

            if len(data) < 50:
                logger.warning("Insufficient data for signal generation")
                return signals

            # Detect market regime
            market_regime = self.regime_detector.detect_regime(data)

            # Engineer features
            features = self.feature_engineer.engineer_features(data)

            # Get ML predictions if model is trained
            ml_predictions = pd.DataFrame()
            if self.ml_model.is_trained:
                ml_predictions = self.ml_model.predict(data)

            # Generate signals for each data point
            for i in range(len(data)):
                signal = await self._generate_single_signal(
                    data.iloc[:i+1], features.iloc[i:i+1], ml_predictions.iloc[i:i+1] if not ml_predictions.empty else pd.DataFrame(),
                    market_regime
                )
                if signal:
                    signals.append(signal)

            logger.info(f"Generated {len(signals)} advanced signals")
            return signals

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return []

    async def _generate_single_signal(self, historical_data: pd.DataFrame,
                                    features: pd.DataFrame, ml_pred: pd.DataFrame,
                                    market_regime: str) -> Optional[Dict[str, Any]]:
        """Generate signal for single data point"""
        try:
            if len(historical_data) < 20:
                return None

            current_data = historical_data.iloc[-1]
            current_features = features.iloc[-1] if not features.empty else {}

            # Get traditional signal from core service
            traditional_signals = await self.core_service.generate_signals("BTC/USDT", "1h", historical_data)
            traditional_signal = traditional_signals[-1] if traditional_signals else 'hold'

            # ML prediction
            ml_signal = 'hold'
            ml_confidence = 0.5

            if not ml_pred.empty and len(ml_pred) > 0:
                ml_signal = ml_pred.iloc[-1].get('prediction', 'hold')
                ml_confidence = ml_pred.iloc[-1].get('confidence', 0.5)

            # Adaptive strategy selection
            available_strategies = ['conservative', 'balanced', 'aggressive']
            selected_strategy = self.strategy_selector.select_strategy(market_regime, available_strategies)

            # Combine signals based on strategy
            final_signal, confidence = self._combine_signals(
                traditional_signal, ml_signal, ml_confidence, selected_strategy, market_regime
            )

            return {
                'timestamp': current_data.name if hasattr(current_data, 'name') else datetime.now(),
                'action': final_signal,
                'confidence': confidence,
                'market_regime': market_regime,
                'strategy': selected_strategy,
                'ml_support': ml_confidence > 0.6,
                'traditional_signal': traditional_signal,
                'ml_signal': ml_signal,
                'features': current_features.to_dict() if hasattr(current_features, 'to_dict') else {}
            }

        except Exception as e:
            logger.error(f"Single signal generation failed: {e}")
            return None

    def _combine_signals(self, traditional: str, ml: str, ml_confidence: float,
                        strategy: str, regime: str) -> Tuple[str, float]:
        """Combine traditional and ML signals based on strategy and regime"""

        # Strategy weights
        strategy_weights = {
            'conservative': {'traditional': 0.7, 'ml': 0.3},
            'balanced': {'traditional': 0.5, 'ml': 0.5},
            'aggressive': {'traditional': 0.3, 'ml': 0.7}
        }

        weights = strategy_weights.get(strategy, strategy_weights['balanced'])

        # Adjust weights based on ML confidence
        if ml_confidence > 0.8:
            weights['ml'] *= 1.2
            weights['traditional'] *= 0.8
        elif ml_confidence < 0.4:
            weights['ml'] *= 0.5
            weights['traditional'] *= 1.5

        # Normalize weights
        total_weight = weights['traditional'] + weights['ml']
        weights['traditional'] /= total_weight
        weights['ml'] /= total_weight

        # Signal mapping
        signal_map = {'buy': 1, 'hold': 0, 'sell': -1}

        traditional_score = signal_map.get(traditional, 0)
        ml_score = signal_map.get(ml, 0)

        # Weighted combination
        combined_score = (traditional_score * weights['traditional'] +
                         ml_score * weights['ml'])

        # Convert back to signal
        if combined_score > 0.3:
            final_signal = 'buy'
        elif combined_score < -0.3:
            final_signal = 'sell'
        else:
            final_signal = 'hold'

        # Calculate confidence
        agreement = 1 if traditional == ml else 0
        confidence = (ml_confidence * weights['ml'] +
                     agreement * weights['traditional'])

        return final_signal, min(confidence, 1.0)

    async def train_ml_model(self, data: pd.DataFrame, symbol: str = "BTC/USDT") -> bool:
        """Train the ML model with historical data"""
        try:
            logger.info(f"Training ML model for {symbol} with {len(data)} data points")

            success = self.ml_model.train(data)

            if success:
                # Save model
                model_path = self.model_dir / f"ml_model_{symbol.replace('/', '_')}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(self.ml_model, f)

                logger.info(f"ML model saved to {model_path}")

            return success

        except Exception as e:
            logger.error(f"ML training failed: {e}")
            return False

    def load_ml_model(self, symbol: str = "BTC/USDT") -> bool:
        """Load trained ML model"""
        try:
            model_path = self.model_dir / f"ml_model_{symbol.replace('/', '_')}.pkl"

            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.ml_model = pickle.load(f)

                logger.info(f"ML model loaded from {model_path}")
                return True
            else:
                logger.warning(f"No saved model found for {symbol}")
                return False

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False

    async def generate_advanced_signals(self, symbol: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate advanced signals with full ML pipeline"""
        return await self.generate_signals(data)