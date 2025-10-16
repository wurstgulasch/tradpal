#!/usr/bin/env python3
"""
Test script for ML prediction functionality.
"""

import sys
import os

# Add services to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services'))

def test_import():
    """Test if we can import the module."""
    try:
        from services.ml_predictor import get_ml_predictor, is_ml_available, MLPredictor
        print("✅ Import successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    try:
        from services.ml_predictor import get_ml_predictor, MLPredictor
        
        predictor = get_ml_predictor()
        assert predictor is None or hasattr(predictor, 'is_trained')
        
        predictor2 = MLPredictor()
        assert hasattr(predictor2, 'is_trained')
        assert predictor2.is_trained is False
        
        print("✅ Basic functionality test passed")
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing ML Predictor Service...")
    
    if test_import() and test_basic_functionality():
        print("🎉 All tests passed!")
    else:
        print("💥 Some tests failed!")
        sys.exit(1)
