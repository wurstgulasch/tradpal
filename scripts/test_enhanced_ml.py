#!/usr/bin/env python3
"""
Test script for enhanced ML models with XGBoost, LightGBM, and RFE + SHAP feature selection.
"""

import sys
import os

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Add src and config to path BEFORE any imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import config.settings directly
import config.settings

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.ml_predictor import MLSignalPredictor, is_ml_available
from src.data_fetcher import fetch_historical_data, fetch_data
from src.indicators import calculate_indicators

def test_enhanced_ml_models():
    """Test XGBoost, LightGBM, and RFE + SHAP feature selection."""
    print("Testing enhanced ML models with XGBoost, LightGBM, and RFE + SHAP...")

    if not is_ml_available():
        print("❌ ML features not available")
        return False

    try:
        # Fetch sample data
        print("📊 Fetching sample data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data

        # Fetch sample data using simpler fetch_data function
        df = fetch_data(limit=1000)  # Increased limit for better training data

        if len(df) < 200:  # Increased minimum requirement
            print("❌ Insufficient data for testing")
            return False

        # Calculate indicators
        print("📈 Calculating technical indicators...")
        df = calculate_indicators(df)

        # Initialize ML predictor with faster settings for testing
        print("🤖 Initializing ML predictor...")
        predictor = MLSignalPredictor(symbol='BTC/USDT', timeframe='1d')

        # Train models with reduced trials for faster testing
        print("🎯 Training enhanced ML models (fast mode)...")
        # Override the default n_trials for faster testing
        import config.settings as settings
        original_n_trials = settings.ML_AUTOML_N_TRIALS
        settings.ML_AUTOML_N_TRIALS = 10  # Reduce trials for testing

        try:
            training_result = predictor.train_models(df, test_size=0.2, prediction_horizon=3)
        finally:
            # Restore original setting
            settings.ML_AUTOML_N_TRIALS = original_n_trials

        if not training_result.get('success', False):
            print(f"❌ Training failed: {training_result.get('error', 'Unknown error')}")
            return False

        print("✅ Training completed successfully!")
        print(f"   Samples: {training_result['samples']}")
        print(f"   Features: {training_result['features']}")

        # Show model performance
        performance = training_result['performance']
        print("\n📊 Model Performance:")
        for model_name, metrics in performance.items():
            if isinstance(metrics, dict) and 'f1_score' in metrics:
                print(".3f"
                      ".3f"
                      ".3f"
                      ".3f")

        # Test prediction
        print("\n🔮 Testing prediction...")
        prediction = predictor.predict_signal(df)
        print(f"   Signal: {prediction['signal']}")
        print(".2f")
        print(f"   Reason: {prediction['reason']}")

        # Show model info
        model_info = predictor.get_model_info()
        print("\n📋 Model Info:")
        print(f"   Status: {model_info['status']}")
        print(f"   Features: {model_info['feature_count']}")
        if model_info['features']:
            print(f"   Top features: {', '.join(model_info['features'][:5])}")

        # Test SHAP explanation
        if prediction.get('explanation'):
            explanation = prediction['explanation']
            if explanation.get('shap_available'):
                print("\n🔍 SHAP Explanation:")
                top_features = explanation.get('top_features', [])
                if top_features:
                    print("   Top influential features:")
                    for i, (feature, importance) in enumerate(top_features[:5]):
                        print(f"     {i+1}. {feature}: {importance:.4f}")

        print("\n✅ Enhanced ML models test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_ml_models()
    sys.exit(0 if success else 1)