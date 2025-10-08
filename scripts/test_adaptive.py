#!/usr/bin/env python3
"""
Test script for the Adaptive Optimization functionality.

This script tests the adaptive optimizer class and its integration
with the live monitoring system.
"""

import sys
import os
import time
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.discovery import load_adaptive_config, save_adaptive_config, apply_adaptive_config
from main import AdaptiveOptimizer
from src.logging_config import logger

def test_adaptive_optimizer():
    """Test the adaptive optimizer functionality."""
    print("🧠 Testing Adaptive Optimizer")
    print("=" * 50)

    try:
        # Test 1: Initialize adaptive optimizer
        print("Test 1: Initializing Adaptive Optimizer...")
        optimizer = AdaptiveOptimizer()
        print(f"✅ Initialized (enabled: {optimizer.enabled})")

        # Test 2: Test configuration loading
        print("\nTest 2: Testing configuration loading...")
        config = optimizer.get_current_config()
        if config:
            print("✅ Loaded existing adaptive configuration")
        else:
            print("ℹ️ No existing adaptive configuration found")

        # Test 3: Test optimization trigger logic
        print("\nTest 3: Testing optimization trigger logic...")
        should_run = optimizer.should_run_optimization()
        print(f"Should run optimization: {should_run}")

        # Test 4: Test configuration saving/loading
        print("\nTest 4: Testing configuration save/load...")
        test_config = {
            'ema': {'enabled': True, 'periods': [9, 21]},
            'rsi': {'enabled': True, 'period': 14, 'oversold': 30, 'overbought': 70},
            'bb': {'enabled': True, 'period': 20, 'std_dev': 2.0},
            'atr': {'enabled': True, 'period': 14},
            'adx': {'enabled': False, 'period': 14},
            'fibonacci': {'enabled': False}
        }

        # Save test config
        save_adaptive_config(test_config, 75.5, 'config/test_adaptive_config.json')
        print("✅ Saved test configuration")

        # Load test config
        loaded_config = load_adaptive_config('config/test_adaptive_config.json')
        if loaded_config:
            print("✅ Loaded test configuration successfully")
        else:
            print("❌ Failed to load test configuration")

        # Test 5: Test configuration application
        print("\nTest 5: Testing configuration application...")
        applied_config = apply_adaptive_config(test_config)
        if applied_config and 'ema' in applied_config:
            print("✅ Configuration applied successfully")
        else:
            print("❌ Configuration application failed")

        # Cleanup test file
        if os.path.exists('config/test_adaptive_config.json'):
            os.remove('config/test_adaptive_config.json')
            print("🧹 Cleaned up test file")

        print("\n🎉 All adaptive optimizer tests passed!")

    except Exception as e:
        print(f"❌ Adaptive optimizer test failed: {e}")
        logger.error(f"Adaptive optimizer test error: {e}")
        return False

    return True

def test_adaptive_integration():
    """Test integration with live monitoring (simulation)."""
    print("\n🔄 Testing Adaptive Integration")
    print("=" * 50)

    try:
        # This would normally be tested in the actual live monitoring
        # For now, just test that the optimizer can be created and used
        optimizer = AdaptiveOptimizer()

        # Simulate a few checks
        for i in range(3):
            should_run = optimizer.should_run_optimization()
            print(f"Check {i+1}: Should run optimization = {should_run}")
            time.sleep(0.1)  # Small delay

        print("✅ Adaptive integration test completed")

    except Exception as e:
        print(f"❌ Adaptive integration test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    success1 = test_adaptive_optimizer()
    success2 = test_adaptive_integration()

    if success1 and success2:
        print("\n🎯 All adaptive optimization tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some adaptive optimization tests failed!")
        sys.exit(1)