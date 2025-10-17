# Erweiterte Signalgenerierung: ML-Ensemble und Advanced Analytics

## Übersicht der Verbesserungen

Die aktuelle Signalgenerierung basiert auf einfachen technischen Indikatoren und regelbasierten Strategien. Wir implementieren fortschrittliche Methoden für intelligentere Signale:

### 1. Machine Learning Ensemble System

**Ziel**: Kombinieren verschiedener ML-Modelle für robustere Vorhersagen

```python
class MLEnsembleSignalGenerator:
    """ML-basierte Signalgenerierung mit Ensemble-Methoden"""

    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100),
            'xgboost': XGBClassifier(),
            'neural_net': self._build_neural_network(),
            'svm': SVC(probability=True)
        }
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ensemble_voter = EnsembleVoter()

    def generate_ml_signals(self, data: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """Generiert ML-basierte Signale"""

        # Feature Engineering
        features = self.feature_engineer.create_features(data)

        # Ensemble Prediction
        predictions = {}
        probabilities = {}

        for model_name, model in self.models.items():
            pred = model.predict(features)
            prob = model.predict_proba(features)
            predictions[model_name] = pred
            probabilities[model_name] = prob

        # Ensemble Voting mit Confidence Weighting
        ensemble_signals = self.ensemble_voter.vote(
            predictions, probabilities, features
        )

        return ensemble_signals
```

### 2. Market Regime Detection

**Ziel**: Automatische Anpassung an Marktbedingungen

```python
class MarketRegimeDetector:
    """Erkennt verschiedene Marktregime für adaptive Strategien"""

    def __init__(self):
        self.regime_models = {
            'trend': TrendRegimeModel(),
            'mean_reversion': MeanReversionRegimeModel(),
            'high_volatility': VolatilityRegimeModel(),
            'low_volatility': LowVolatilityRegimeModel(),
            'breakout': BreakoutRegimeModel()
        }

    def detect_regime(self, data: pd.DataFrame) -> str:
        """Klassifiziert aktuelles Marktregime"""

        # Multi-Faktor Analyse
        volatility = self._calculate_volatility_regime(data)
        trend_strength = self._calculate_trend_regime(data)
        volume_profile = self._calculate_volume_regime(data)

        # Regime-Klassifikation
        regime_scores = {}
        for regime_name, model in self.regime_models.items():
            score = model.score_regime(data, volatility, trend_strength, volume_profile)
            regime_scores[regime_name] = score

        # Bestes Regime auswählen
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]

        return best_regime, confidence

    def _calculate_volatility_regime(self, data: pd.DataFrame) -> float:
        """Berechnet Volatilitäts-Regime-Score"""
        returns = data['close'].pct_change()
        current_vol = returns.rolling(20).std().iloc[-1]
        avg_vol = returns.rolling(100).std().mean()

        return current_vol / avg_vol if avg_vol > 0 else 1.0
```

### 3. Advanced Feature Engineering

**Ziel**: Umfassende Feature-Extraktion für bessere ML-Modelle

```python
class AdvancedFeatureEngineer:
    """Erweiterte Feature-Engineering Pipeline"""

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Erstellt umfassende Feature-Matrix"""

        features = pd.DataFrame(index=data.index)

        # Basis technische Indikatoren
        features = self._add_technical_indicators(features, data)

        # Preis-Pattern Features
        features = self._add_price_patterns(features, data)

        # Volumen-Features
        features = self._add_volume_features(features, data)

        # Zeit-basierte Features
        features = self._add_temporal_features(features, data)

        # Statistische Features
        features = self._add_statistical_features(features, data)

        # Intermarket Features (falls verfügbar)
        features = self._add_intermarket_features(features, data)

        return features.fillna(method='ffill').fillna(0)

    def _add_technical_indicators(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Fügt technische Indikatoren hinzu"""
        # Erweiterte Indikatoren-Kombinationen
        features['ema_5'] = ta.EMA(data['close'], 5)
        features['ema_10'] = ta.EMA(data['close'], 10)
        features['ema_20'] = ta.EMA(data['close'], 20)
        features['ema_50'] = ta.EMA(data['close'], 50)

        # RSI Varianten
        features['rsi_6'] = ta.RSI(data['close'], 6)
        features['rsi_14'] = ta.RSI(data['close'], 14)
        features['rsi_21'] = ta.RSI(data['close'], 21)

        # MACD Signale
        macd, macdsignal, macdhist = ta.MACD(data['close'])
        features['macd'] = macd
        features['macd_signal'] = macdsignal
        features['macd_hist'] = macdhist

        # Bollinger Bands Features
        upper, middle, lower = ta.BBANDS(data['close'])
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = (upper - lower) / middle
        features['bb_position'] = (data['close'] - lower) / (upper - lower)

        # Momentum Indikatoren
        features['roc_5'] = ta.ROC(data['close'], 5)
        features['roc_10'] = ta.ROC(data['close'], 10)
        features['mom_10'] = ta.MOM(data['close'], 10)

        # Volatilität
        features['atr_14'] = ta.ATR(data['high'], data['low'], data['close'], 14)

        # Trend Strength
        features['adx_14'] = ta.ADX(data['high'], data['low'], data['close'], 14)

        return features

    def _add_price_patterns(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Fügt Preis-Pattern-Erkennung hinzu"""

        # Candlestick Patterns
        features['doji'] = ta.CDLDOJI(data['open'], data['high'], data['low'], data['close'])
        features['hammer'] = ta.CDLHAMMER(data['open'], data['high'], data['low'], data['close'])
        features['shooting_star'] = ta.CDLSHOOTINGSTAR(data['open'], data['high'], data['low'], data['close'])

        # Support/Resistance Levels
        features['near_support'] = self._calculate_near_support(data)
        features['near_resistance'] = self._calculate_near_resistance(data)

        # Price Gaps
        features['gap_up'] = (data['open'] > data['close'].shift(1)) * 1
        features['gap_down'] = (data['open'] < data['close'].shift(1)) * 1

        return features

    def _add_volume_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Fügt Volumen-basierte Features hinzu"""

        # Volume Indikatoren
        features['obv'] = ta.OBV(data['close'], data['volume'])
        features['volume_sma_20'] = ta.SMA(data['volume'], 20)
        features['volume_ratio'] = data['volume'] / ta.SMA(data['volume'], 20)

        # Volume Price Trend
        features['vpt'] = ta.AD(data['high'], data['low'], data['close'], data['volume'])

        # Chaikin Money Flow
        features['cmf'] = ta.ADOSC(data['high'], data['low'], data['close'], data['volume'])

        return features

    def _add_temporal_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Fügt zeitbasierte Features hinzu"""

        # Zeit des Tages (für Intraday)
        if hasattr(data.index, 'hour'):
            features['hour'] = data.index.hour
            features['minute'] = data.index.minute
            features['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)

        # Wochentag
        features['weekday'] = data.index.weekday
        features['is_monday'] = (data.index.weekday == 0) * 1
        features['is_friday'] = (data.index.weekday == 4) * 1

        # Monat
        features['month'] = data.index.month

        return features

    def _add_statistical_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Fügt statistische Features hinzu"""

        # Returns über verschiedene Perioden
        features['return_1d'] = data['close'].pct_change(1)
        features['return_5d'] = data['close'].pct_change(5)
        features['return_20d'] = data['close'].pct_change(20)

        # Volatilität
        features['volatility_5d'] = data['close'].pct_change().rolling(5).std()
        features['volatility_20d'] = data['close'].pct_change().rolling(20).std()

        # Skewness und Kurtosis
        features['skew_20d'] = data['close'].pct_change().rolling(20).skew()
        features['kurtosis_20d'] = data['close'].pct_change().rolling(20).kurt()

        # Quantile Features
        features['close_quantile_20'] = data['close'].rolling(20).quantile(0.2)
        features['close_quantile_80'] = data['close'].rolling(20).quantile(0.8)

        return features
```

### 4. Adaptive Strategy Selection

**Ziel**: Automatische Strategie-Auswahl basierend auf Marktbedingungen

```python
class AdaptiveStrategySelector:
    """Wählt optimale Strategien basierend auf Marktregime"""

    def __init__(self):
        self.strategies = {
            'trend_following': TrendFollowingStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'breakout': BreakoutStrategy(),
            'scalping': ScalpingStrategy(),
            'swing': SwingStrategy()
        }

        self.performance_tracker = StrategyPerformanceTracker()
        self.market_regime_detector = MarketRegimeDetector()

    def select_optimal_strategy(self, data: pd.DataFrame, symbol: str) -> str:
        """Wählt beste Strategie für aktuelle Marktbedingungen"""

        # Aktuelles Marktregime erkennen
        regime, confidence = self.market_regime_detector.detect_regime(data)

        # Historische Performance für dieses Regime abrufen
        strategy_scores = {}
        for strategy_name, strategy in self.strategies.items():
            performance = self.performance_tracker.get_regime_performance(
                strategy_name, regime, symbol
            )
            strategy_scores[strategy_name] = performance

        # Beste Strategie auswählen
        best_strategy = max(strategy_scores, key=strategy_scores.get)

        # Adaptivität: Wenn Confidence niedrig, konservative Strategie wählen
        if confidence < 0.6:
            best_strategy = 'mean_reversion'  # Sicherere Strategie

        return best_strategy

    def update_strategy_performance(self, strategy_name: str, regime: str,
                                  symbol: str, pnl: float):
        """Aktualisiert Strategie-Performance für zukünftige Entscheidungen"""
        self.performance_tracker.update_performance(strategy_name, regime, symbol, pnl)
```

### 5. Multi-Timeframe Signal Integration

**Ziel**: Kombination von Signalen über verschiedene Timeframes

```python
class MultiTimeframeSignalIntegrator:
    """Integriert Signale über mehrere Timeframes"""

    def __init__(self):
        self.timeframes = ['5m', '15m', '1h', '4h', '1d']
        self.timeframe_weights = {
            '5m': 0.1,   # Kurzfristig, weniger Gewicht
            '15m': 0.2,
            '1h': 0.3,  # Mittelfristig, höheres Gewicht
            '4h': 0.25,
            '1d': 0.15  # Langfristig, moderates Gewicht
        }

    def integrate_signals(self, signals_by_timeframe: Dict[str, List[Dict]]) -> List[Dict]:
        """Integriert Signale über alle Timeframes"""

        integrated_signals = []

        # Finde alle eindeutigen Zeitpunkte
        all_timestamps = set()
        for timeframe_signals in signals_by_timeframe.values():
            for signal in timeframe_signals:
                all_timestamps.add(signal['timestamp'])

        for timestamp in sorted(all_timestamps):
            # Sammle Signale für diesen Zeitpunkt über alle Timeframes
            signals_at_time = []
            weights_at_time = []

            for timeframe, signals in signals_by_timeframe.items():
                # Finde Signal für diesen Zeitpunkt in diesem Timeframe
                signal_at_time = self._find_signal_at_timestamp(signals, timestamp)
                if signal_at_time:
                    signals_at_time.append(signal_at_time)
                    weights_at_time.append(self.timeframe_weights[timeframe])

            if signals_at_time:
                # Gewichtete Integration
                integrated_signal = self._weighted_signal_integration(
                    signals_at_time, weights_at_time
                )
                integrated_signals.append(integrated_signal)

        return integrated_signals

    def _weighted_signal_integration(self, signals: List[Dict], weights: List[float]) -> Dict:
        """Gewichtete Signal-Integration"""

        # Normalisiere Gewichte
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Gewichtete Confidence
        weighted_confidence = sum(s['confidence'] * w for s, w in zip(signals, normalized_weights))

        # Signal-Konsensus (mehrheitliche Entscheidung)
        actions = [s['action'] for s in signals]
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        consensus_action = max(action_counts, key=action_counts.get)

        # Nur wenn Konsensus stark genug
        if action_counts[consensus_action] / len(signals) >= 0.6:  # 60% Mehrheit
            return {
                'timestamp': signals[0]['timestamp'],
                'action': consensus_action,
                'confidence': weighted_confidence,
                'reason': f'Multi-timeframe consensus ({len(signals)} timeframes)',
                'integrated_signals': len(signals)
            }

        return None
```

### 6. Sentiment Analysis Integration

**Ziel**: Einbeziehung von Markt-Sentiment in Signalgenerierung

```python
class SentimentSignalIntegrator:
    """Integriert Sentiment-Analyse in Signalgenerierung"""

    def __init__(self):
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.sentiment_cache = {}

    async def enhance_signals_with_sentiment(self, signals: List[Dict],
                                           symbol: str) -> List[Dict]:
        """Erweitert Signale um Sentiment-Informationen"""

        enhanced_signals = []

        for signal in signals:
            # Hole aktuelles Sentiment
            sentiment_data = await self._get_current_sentiment(symbol)

            # Adjust Confidence basierend auf Sentiment
            adjusted_confidence = self._adjust_confidence_with_sentiment(
                signal['confidence'], sentiment_data
            )

            # Sentiment-basiertes Signal-Filtering
            if self._should_filter_signal(signal, sentiment_data):
                continue

            # Erweitere Signal um Sentiment-Info
            enhanced_signal = signal.copy()
            enhanced_signal['confidence'] = adjusted_confidence
            enhanced_signal['sentiment'] = sentiment_data
            enhanced_signal['reason'] += f" | Sentiment: {sentiment_data['overall']}"

            enhanced_signals.append(enhanced_signal)

        return enhanced_signals

    async def _get_current_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Ruft aktuelles Markt-Sentiment ab"""

        # Cache-Check
        cache_key = f"sentiment_{symbol}"
        if cache_key in self.sentiment_cache:
            cached_data, timestamp = self.sentiment_cache[cache_key]
            if datetime.now() - timestamp < timedelta(minutes=15):
                return cached_data

        # Sammle Sentiment aus verschiedenen Quellen
        sentiment_data = await self.sentiment_analyzer.analyze_sentiment(symbol)

        # Cache für 15 Minuten
        self.sentiment_cache[cache_key] = (sentiment_data, datetime.now())

        return sentiment_data

    def _adjust_confidence_with_sentiment(self, base_confidence: float,
                                        sentiment_data: Dict) -> float:
        """Passt Confidence basierend auf Sentiment an"""

        sentiment_score = sentiment_data.get('overall_score', 0.5)  # 0-1 Scale

        # Sentiment Verstärkung
        if sentiment_score > 0.7:  # Stark positives Sentiment
            adjustment = 0.1  # +10% Confidence
        elif sentiment_score < 0.3:  # Stark negatives Sentiment
            adjustment = -0.1  # -10% Confidence
        else:
            adjustment = 0.0  # Neutral

        adjusted = base_confidence + adjustment
        return max(0.0, min(1.0, adjusted))  # Clamp to [0,1]
```

### 7. Risk-Adjusted Signal Scoring

**Ziel**: Signale mit Risiko-Metriken gewichten

```python
class RiskAdjustedSignalScorer:
    """Bewertet Signale basierend auf Risiko-Metriken"""

    def __init__(self):
        self.risk_models = {
            'var_model': ValueAtRiskModel(),
            'sharpe_model': SharpeRatioModel(),
            'sortino_model': SortinoRatioModel(),
            'calmar_model': CalmarRatioModel()
        }

    def score_signal_with_risk(self, signal: Dict, historical_data: pd.DataFrame,
                              portfolio_value: float) -> Dict[str, Any]:
        """Bewertet Signal mit Risiko-Adjustierung"""

        # Berechne Risiko-Metriken für das Signal
        risk_metrics = {}

        for metric_name, model in self.risk_models.items():
            risk_metrics[metric_name] = model.calculate_metric(
                signal, historical_data, portfolio_value
            )

        # Kombinierte Risiko-Score
        risk_score = self._calculate_combined_risk_score(risk_metrics)

        # Risk-adjusted Confidence
        risk_adjusted_confidence = signal['confidence'] * (1 - risk_score)

        return {
            'original_confidence': signal['confidence'],
            'risk_adjusted_confidence': risk_adjusted_confidence,
            'risk_score': risk_score,
            'risk_metrics': risk_metrics,
            'risk_level': self._classify_risk_level(risk_score)
        }

    def _calculate_combined_risk_score(self, risk_metrics: Dict[str, float]) -> float:
        """Berechnet kombinierten Risiko-Score"""

        # Gewichtete Kombination verschiedener Risiko-Metriken
        weights = {
            'var_model': 0.3,      # Value at Risk
            'sharpe_model': 0.25,  # Sharpe Ratio
            'sortino_model': 0.25, # Sortino Ratio
            'calmar_model': 0.2    # Calmar Ratio
        }

        combined_score = 0.0
        for metric, weight in weights.items():
            # Normalisiere zu 0-1 Scale (höher = riskanter)
            normalized_metric = self._normalize_risk_metric(risk_metrics[metric], metric)
            combined_score += normalized_metric * weight

        return combined_score

    def _normalize_risk_metric(self, value: float, metric_type: str) -> float:
        """Normalisiert Risiko-Metrik zu 0-1 Scale"""

        if metric_type == 'var_model':
            # VaR: höher = riskanter, normalisiere zu 0-1
            return min(1.0, max(0.0, value / 0.1))  # 10% VaR als Maximum
        elif metric_type in ['sharpe_model', 'sortino_model']:
            # Sharpe/Sortino: höher = besser, invertiere für Risiko-Score
            return max(0.0, 1.0 - (value / 3.0))  # 3.0 als sehr guter Wert
        elif metric_type == 'calmar_model':
            # Calmar: höher = besser, invertiere für Risiko-Score
            return max(0.0, 1.0 - (value / 5.0))  # 5.0 als sehr guter Wert

        return 0.5  # Default neutral
```

## Implementierungsplan

### Phase 1: Grundlegende ML-Integration (2 Wochen)
1. **Random Forest Modell** für Signal-Klassifikation
2. **Feature Engineering Pipeline** mit erweiterten Indikatoren
3. **Backtesting Framework** für ML-Modelle

### Phase 2: Ensemble-System (2 Wochen)
1. **Multi-Modell Ensemble** (RF, XGBoost, Neural Net)
2. **Ensemble Voting System** mit Confidence Weighting
3. **Model Performance Tracking**

### Phase 3: Adaptive Systeme (3 Wochen)
1. **Market Regime Detection**
2. **Adaptive Strategy Selection**
3. **Dynamic Parameter Adjustment**

### Phase 4: Advanced Features (3 Wochen)
1. **Multi-Timeframe Integration**
2. **Sentiment Analysis Integration**
3. **Risk-Adjusted Scoring**

### Phase 5: Production Optimization (2 Wochen)
1. **Performance Optimization**
2. **Model Retraining Pipeline**
3. **Monitoring und Alerting**

## Erwartete Verbesserungen

### Quantitative Verbesserungen:
- **Win Rate**: +15-25% durch ML-Ensemble
- **Sharpe Ratio**: +0.5-1.0 durch Risiko-Adjustierung
- **Maximum Drawdown**: -20% durch Market Regime Adaptation
- **Signal Accuracy**: +30% durch Multi-Timeframe Integration

### Qualitative Verbesserungen:
- **Adaptivität**: Automatische Anpassung an Marktbedingungen
- **Robustheit**: Weniger False Signals durch Ensemble-Methoden
- **Risikomanagement**: Bessere Risiko-Adjustierung
- **Transparenz**: Erklärbare ML-Entscheidungen

## Monitoring und Evaluation

```python
class AdvancedSignalMonitor:
    """Überwacht Performance der erweiterten Signalgenerierung"""

    def __init__(self):
        self.metrics = {
            'ml_model_accuracy': [],
            'ensemble_performance': [],
            'regime_detection_accuracy': [],
            'risk_adjustment_effectiveness': [],
            'sentiment_impact': []
        }

    def track_signal_performance(self, signal: Dict, outcome: Dict):
        """Verfolgt Signal-Performance über Zeit"""

        # ML Model Accuracy
        if 'ml_prediction' in signal:
            accuracy = 1.0 if signal['ml_prediction'] == outcome['actual'] else 0.0
            self.metrics['ml_model_accuracy'].append(accuracy)

        # Ensemble Performance
        if 'ensemble_vote' in signal:
            ensemble_correct = 1.0 if signal['ensemble_vote'] == outcome['actual'] else 0.0
            self.metrics['ensemble_performance'].append(ensemble_correct)

        # Risk Adjustment Effectiveness
        risk_improvement = outcome.get('risk_adjusted_pnl', 0) - outcome.get('unadjusted_pnl', 0)
        self.metrics['risk_adjustment_effectiveness'].append(risk_improvement)

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generiert detaillierten Performance-Report"""

        report = {}
        for metric_name, values in self.metrics.items():
            if values:
                report[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
            else:
                report[metric_name] = {'count': 0}

        return report
```

Diese Erweiterungen würden die Signalgenerierung von einem einfachen regelbasierten System zu einem intelligenten, adaptiven ML-System transformieren, das besser auf Marktbedingungen reagiert und robustere Handelsentscheidungen trifft.</content>
<parameter name="filePath">/Users/danielsadowski/VSCodeProjects/tradpal_indicator/docs/ADVANCED_SIGNAL_GENERATION.md