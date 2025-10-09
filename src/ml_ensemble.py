"""
Ensemble Methods for Combining GA and ML Predictions

Provides ensemble prediction strategies that combine:
- Genetic Algorithm optimized indicators
- Machine Learning models
- Multiple prediction strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import logging

from config.settings import (
    SYMBOL, TIMEFRAME, ML_USE_ENSEMBLE, ML_ENSEMBLE_WEIGHTS,
    ML_ENSEMBLE_VOTING, ML_ENSEMBLE_MIN_CONFIDENCE
)


class EnsemblePredictor:
    """
    Ensemble predictor combining GA-optimized indicators with ML predictions.
    
    Features:
    - Multiple voting strategies (weighted, majority, unanimous)
    - Confidence-based weighting
    - Adaptive ensemble weights
    - Performance tracking per component
    """
    
    def __init__(self, symbol: str = SYMBOL, timeframe: str = TIMEFRAME,
                 results_dir: str = "output/ensemble"):
        """
        Initialize ensemble predictor.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            results_dir: Directory to store ensemble results
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Component weights
        self.weights = ML_ENSEMBLE_WEIGHTS.copy()
        
        # Performance tracking
        self.performance_history = {
            'ml': [],
            'ga': [],
            'ensemble': []
        }
        
        # Load previous performance if available
        self._load_performance_history()
    
    def predict(self, ml_prediction: Dict[str, Any], ga_prediction: Dict[str, Any],
               traditional_signal: str = 'NEUTRAL') -> Dict[str, Any]:
        """
        Generate ensemble prediction from ML and GA components.
        
        Args:
            ml_prediction: Prediction from ML model
            ga_prediction: Prediction from GA-optimized indicators
            traditional_signal: Traditional technical analysis signal
            
        Returns:
            Ensemble prediction with combined signal and confidence
        """
        # Extract signals and confidences
        ml_signal = ml_prediction.get('signal', 'NEUTRAL')
        ml_confidence = ml_prediction.get('confidence', 0.5)
        
        ga_signal = ga_prediction.get('signal', traditional_signal)
        ga_confidence = ga_prediction.get('confidence', 0.7)  # GA typically has higher confidence
        
        # Apply voting strategy
        if ML_ENSEMBLE_VOTING == 'weighted':
            ensemble_result = self._weighted_voting(
                ml_signal, ml_confidence,
                ga_signal, ga_confidence
            )
        elif ML_ENSEMBLE_VOTING == 'majority':
            ensemble_result = self._majority_voting(
                ml_signal, ml_confidence,
                ga_signal, ga_confidence,
                traditional_signal
            )
        elif ML_ENSEMBLE_VOTING == 'unanimous':
            ensemble_result = self._unanimous_voting(
                ml_signal, ml_confidence,
                ga_signal, ga_confidence
            )
        else:
            ensemble_result = self._weighted_voting(
                ml_signal, ml_confidence,
                ga_signal, ga_confidence
            )
        
        # Add metadata
        ensemble_result.update({
            'ml_signal': ml_signal,
            'ml_confidence': ml_confidence,
            'ga_signal': ga_signal,
            'ga_confidence': ga_confidence,
            'traditional_signal': traditional_signal,
            'voting_strategy': ML_ENSEMBLE_VOTING,
            'weights': self.weights.copy()
        })
        
        return ensemble_result
    
    def _weighted_voting(self, ml_signal: str, ml_confidence: float,
                        ga_signal: str, ga_confidence: float) -> Dict[str, Any]:
        """
        Weighted voting strategy.
        
        Combines signals based on confidence-weighted voting.
        """
        # Convert signals to numeric scores
        signal_scores = {'BUY': 1.0, 'NEUTRAL': 0.0, 'SELL': -1.0}
        
        ml_score = signal_scores.get(ml_signal, 0.0) * ml_confidence
        ga_score = signal_scores.get(ga_signal, 0.0) * ga_confidence
        
        # Apply weights
        ml_weight = self.weights.get('ml', 0.6)
        ga_weight = self.weights.get('ga', 0.4)
        
        # Normalize weights
        total_weight = ml_weight + ga_weight
        ml_weight /= total_weight
        ga_weight /= total_weight
        
        # Calculate weighted score
        ensemble_score = (ml_score * ml_weight) + (ga_score * ga_weight)
        
        # Calculate ensemble confidence
        ensemble_confidence = abs(ensemble_score)
        
        # Determine final signal
        if ensemble_score > ML_ENSEMBLE_MIN_CONFIDENCE:
            final_signal = 'BUY'
        elif ensemble_score < -ML_ENSEMBLE_MIN_CONFIDENCE:
            final_signal = 'SELL'
        else:
            final_signal = 'NEUTRAL'
        
        return {
            'signal': final_signal,
            'confidence': ensemble_confidence,
            'ensemble_score': ensemble_score,
            'method': 'weighted_voting'
        }
    
    def _majority_voting(self, ml_signal: str, ml_confidence: float,
                        ga_signal: str, ga_confidence: float,
                        traditional_signal: str) -> Dict[str, Any]:
        """
        Majority voting strategy.
        
        Signal must be agreed upon by at least 2 out of 3 sources.
        """
        signals = [ml_signal, ga_signal, traditional_signal]
        confidences = [ml_confidence, ga_confidence, 0.5]
        
        # Count signal occurrences
        signal_counts = {}
        signal_confidences = {}
        
        for signal, confidence in zip(signals, confidences):
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
            if signal not in signal_confidences:
                signal_confidences[signal] = []
            signal_confidences[signal].append(confidence)
        
        # Find majority signal
        max_count = max(signal_counts.values())
        
        if max_count >= 2:
            # Get signal(s) with max count
            majority_signals = [s for s, c in signal_counts.items() if c == max_count]
            
            if len(majority_signals) == 1:
                final_signal = majority_signals[0]
                # Average confidence of agreeing sources
                ensemble_confidence = np.mean(signal_confidences[final_signal])
            else:
                # Tie - use confidence-weighted selection
                best_signal = max(majority_signals, 
                                key=lambda s: np.mean(signal_confidences[s]))
                final_signal = best_signal
                ensemble_confidence = np.mean(signal_confidences[best_signal])
        else:
            # No majority - return neutral
            final_signal = 'NEUTRAL'
            ensemble_confidence = 0.3
        
        return {
            'signal': final_signal,
            'confidence': ensemble_confidence,
            'signal_counts': signal_counts,
            'method': 'majority_voting'
        }
    
    def _unanimous_voting(self, ml_signal: str, ml_confidence: float,
                         ga_signal: str, ga_confidence: float) -> Dict[str, Any]:
        """
        Unanimous voting strategy.
        
        Both ML and GA must agree, otherwise returns NEUTRAL.
        """
        if ml_signal == ga_signal and ml_signal != 'NEUTRAL':
            # Both agree on non-neutral signal
            final_signal = ml_signal
            # Use minimum confidence for conservative estimation
            ensemble_confidence = min(ml_confidence, ga_confidence)
        else:
            # No agreement or one is neutral
            final_signal = 'NEUTRAL'
            ensemble_confidence = abs(ml_confidence - ga_confidence) / 2
        
        return {
            'signal': final_signal,
            'confidence': ensemble_confidence,
            'agreement': ml_signal == ga_signal,
            'method': 'unanimous_voting'
        }
    
    def update_performance(self, actual_outcome: float, ml_prediction: Dict[str, Any],
                          ga_prediction: Dict[str, Any], ensemble_prediction: Dict[str, Any]):
        """
        Update performance tracking for adaptive weighting.
        
        Args:
            actual_outcome: Actual price movement (1.0 for up, -1.0 for down, 0.0 for neutral)
            ml_prediction: ML prediction that was made
            ga_prediction: GA prediction that was made
            ensemble_prediction: Ensemble prediction that was made
        """
        # Calculate accuracy for each component
        ml_signal = ml_prediction.get('signal', 'NEUTRAL')
        ga_signal = ga_prediction.get('signal', 'NEUTRAL')
        ensemble_signal = ensemble_prediction.get('signal', 'NEUTRAL')
        
        signal_to_value = {'BUY': 1.0, 'SELL': -1.0, 'NEUTRAL': 0.0}
        
        ml_value = signal_to_value.get(ml_signal, 0.0)
        ga_value = signal_to_value.get(ga_signal, 0.0)
        ensemble_value = signal_to_value.get(ensemble_signal, 0.0)
        
        # Calculate correctness (1.0 if correct direction, 0.0 if wrong, 0.5 if neutral)
        def calculate_correctness(predicted, actual):
            if predicted == 0.0:
                return 0.5  # Neutral
            elif (predicted > 0 and actual > 0) or (predicted < 0 and actual < 0):
                return 1.0  # Correct
            else:
                return 0.0  # Incorrect
        
        ml_correct = calculate_correctness(ml_value, actual_outcome)
        ga_correct = calculate_correctness(ga_value, actual_outcome)
        ensemble_correct = calculate_correctness(ensemble_value, actual_outcome)
        
        # Update history
        self.performance_history['ml'].append(ml_correct)
        self.performance_history['ga'].append(ga_correct)
        self.performance_history['ensemble'].append(ensemble_correct)
        
        # Keep only recent history (last 100 predictions)
        max_history = 100
        for key in self.performance_history:
            if len(self.performance_history[key]) > max_history:
                self.performance_history[key] = self.performance_history[key][-max_history:]
        
        # Update weights adaptively based on recent performance
        self._update_adaptive_weights()
        
        # Save performance history
        self._save_performance_history()
    
    def _update_adaptive_weights(self):
        """Update ensemble weights based on recent performance."""
        min_samples = 10
        
        ml_history = self.performance_history['ml']
        ga_history = self.performance_history['ga']
        
        if len(ml_history) < min_samples or len(ga_history) < min_samples:
            return  # Not enough data yet
        
        # Calculate recent accuracy
        ml_accuracy = np.mean(ml_history[-20:])
        ga_accuracy = np.mean(ga_history[-20:])
        
        # Avoid division by zero
        total_accuracy = ml_accuracy + ga_accuracy
        if total_accuracy == 0:
            return
        
        # Update weights proportionally to accuracy
        # Use exponential moving average for smooth transitions
        alpha = 0.1  # Smoothing factor
        
        new_ml_weight = ml_accuracy / total_accuracy
        new_ga_weight = ga_accuracy / total_accuracy
        
        self.weights['ml'] = (1 - alpha) * self.weights['ml'] + alpha * new_ml_weight
        self.weights['ga'] = (1 - alpha) * self.weights['ga'] + alpha * new_ga_weight
        
        # Normalize
        total_weight = self.weights['ml'] + self.weights['ga']
        self.weights['ml'] /= total_weight
        self.weights['ga'] /= total_weight
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for each component and ensemble."""
        stats = {}
        
        for component in ['ml', 'ga', 'ensemble']:
            history = self.performance_history[component]
            if len(history) > 0:
                stats[component] = {
                    'accuracy': np.mean(history),
                    'recent_accuracy': np.mean(history[-20:]) if len(history) >= 20 else np.mean(history),
                    'total_predictions': len(history),
                    'std_dev': np.std(history)
                }
            else:
                stats[component] = {
                    'accuracy': 0.0,
                    'recent_accuracy': 0.0,
                    'total_predictions': 0,
                    'std_dev': 0.0
                }
        
        stats['current_weights'] = self.weights.copy()
        
        return stats
    
    def _save_performance_history(self):
        """Save performance history to disk."""
        # Ensure directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Replace / in symbol for file path
        safe_symbol = self.symbol.replace('/', '_')
        history_path = self.results_dir / f"{safe_symbol}_{self.timeframe}_performance_history.json"
        
        data = {
            'performance_history': self.performance_history,
            'weights': self.weights,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(history_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_performance_history(self):
        """Load performance history from disk."""
        # Replace / in symbol for file path
        safe_symbol = self.symbol.replace('/', '_')
        history_path = self.results_dir / f"{safe_symbol}_{self.timeframe}_performance_history.json"
        
        if not history_path.exists():
            return
        
        try:
            with open(history_path, 'r') as f:
                data = json.load(f)
            
            self.performance_history = data.get('performance_history', self.performance_history)
            self.weights = data.get('weights', self.weights)
            
            print(f"📂 Loaded ensemble performance history from {history_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to load performance history: {e}")


# Global ensemble predictor
ensemble_predictor = None


def get_ensemble_predictor(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> EnsemblePredictor:
    """Get or create ensemble predictor instance."""
    global ensemble_predictor
    
    if ensemble_predictor is None:
        ensemble_predictor = EnsemblePredictor(symbol=symbol, timeframe=timeframe)
    
    return ensemble_predictor


def combine_predictions(ml_prediction: Dict[str, Any], ga_prediction: Dict[str, Any],
                       traditional_signal: str = 'NEUTRAL', symbol: str = SYMBOL,
                       timeframe: str = TIMEFRAME) -> Dict[str, Any]:
    """
    Convenience function to combine predictions.
    
    Args:
        ml_prediction: ML model prediction
        ga_prediction: GA-optimized indicator prediction
        traditional_signal: Traditional technical analysis signal
        symbol: Trading symbol
        timeframe: Timeframe
        
    Returns:
        Combined ensemble prediction
    """
    if not ML_USE_ENSEMBLE:
        # If ensemble not enabled, return ML prediction or GA prediction based on availability
        if ml_prediction.get('confidence', 0) > ga_prediction.get('confidence', 0):
            return ml_prediction
        else:
            return ga_prediction
    
    predictor = get_ensemble_predictor(symbol, timeframe)
    return predictor.predict(ml_prediction, ga_prediction, traditional_signal)


def update_ensemble_performance(actual_outcome: float, ml_prediction: Dict[str, Any],
                               ga_prediction: Dict[str, Any], ensemble_prediction: Dict[str, Any],
                               symbol: str = SYMBOL, timeframe: str = TIMEFRAME):
    """
    Update ensemble performance tracking.
    
    Args:
        actual_outcome: Actual price movement
        ml_prediction: ML prediction
        ga_prediction: GA prediction
        ensemble_prediction: Ensemble prediction
        symbol: Trading symbol
        timeframe: Timeframe
    """
    if not ML_USE_ENSEMBLE:
        return
    
    predictor = get_ensemble_predictor(symbol, timeframe)
    predictor.update_performance(actual_outcome, ml_prediction, ga_prediction, ensemble_prediction)
