#!/usr/bin/env python3
"""
Test script for SHAP integration in ML training service.
Tests model interpretability and SHAP explanations.
"""

import asyncio
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up basic environment
os.environ.setdefault('LOG_LEVEL', 'INFO')

# Import only the SHAP interpreter directly to avoid service dependencies
from services.trading_service.trading_ai_service.ml_training.shap_interpreter import SHAPInterpreter


async def test_shap_integration():
    """Test SHAP integration with ML trainer."""
    print("Testing SHAP Integration...")

    # For now, just test the SHAP interpreter directly
    # Full integration test would require setting up the entire service environment
    print("Skipping full integration test - requires service setup")
    print("Testing SHAP interpreter directly instead...")

    await test_shap_interpreter_directly()


async def test_shap_interpreter_directly():
    """Test SHAP interpreter directly."""
    print("\nTesting SHAP Interpreter Directly...")

    try:
        interpreter = SHAPInterpreter()

        # Create sample data
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        feature_names = [f'feature_{i}' for i in range(n_features)]

        # Set feature names (simulate loading a model)
        interpreter.feature_names = feature_names

        # Set background data for KernelExplainer
        interpreter.set_background_data(X[:10])  # Small background dataset

        # Initialize a simple model for testing
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Manually set the model and initialize explainer
        interpreter.model = model
        interpreter._initialize_explainer()

        # Test global importance
        print("Testing global importance...")
        global_imp = interpreter.explain_global_importance(X)

        if 'error' not in global_imp:
            print("Global importance successful!")
            if 'top_features' in global_imp:
                print(f"Top features: {global_imp['top_features'][:3]}")
        else:
            print(f"Global importance failed: {global_imp['error']}")

        # Test local explanation
        print("Testing local explanation...")
        sample_idx = 0
        local_exp = interpreter.explain_prediction(X[sample_idx:sample_idx+1], 0)

        if 'error' not in local_exp:
            print("Local explanation successful!")
            print(f"Explanation keys: {list(local_exp.keys())}")
            if 'feature_importance' in local_exp:
                feature_imp = local_exp['feature_importance']
                top_contribs = sorted(feature_imp.items(), key=lambda x: abs(x[1]['shap_value']), reverse=True)[:3]
                print("Top feature contributions:")
                for feature, data in top_contribs:
                    print(f"  {feature}: {data['shap_value']:.4f}")
        else:
            print(f"Local explanation failed: {local_exp['error']}")

        # Test trading decision explanation
        print("Testing trading decision explanation...")
        sample_features = {f'feature_{i}': X[sample_idx, i] for i in range(n_features)}
        trading_exp = interpreter.explain_trading_decision(sample_features, 0.0)

        if 'error' not in trading_exp:
            print("Trading decision explanation successful!")
            if 'trading_interpretation' in trading_exp:
                interp = trading_exp['trading_interpretation']
                print(f"Signal strength: {interp.get('signal_strength', 'N/A')}")
                print(f"Confidence level: {interp.get('confidence_level', 'N/A')}")
        else:
            print(f"Trading decision explanation failed: {trading_exp['error']}")

        print("Direct SHAP Interpreter Test Completed!")

    except Exception as e:
        print(f"Direct test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_shap_integration())


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_shap_integration())
    asyncio.run(test_shap_interpreter_directly())