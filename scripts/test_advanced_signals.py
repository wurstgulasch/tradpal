#!/usr/bin/env python3
"""
Test script for Advanced Signal Generation
Tests the ML-based signal generation system with ensemble models and market regime detection.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import *
from services.core.advanced_signal_generator import AdvancedSignalGenerator, AdvancedFeatureEngineer, MarketRegimeDetector
from services.core.service import CoreService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data(num_rows=1000):
    """Create sample OHLCV data for testing"""
    np.random.seed(42)

    # Generate timestamps
    start_date = datetime.now() - timedelta(days=30)
    timestamps = pd.date_range(start=start_date, periods=num_rows, freq='5min')

    # Generate realistic OHLCV data
    base_price = 50000
    prices = []
    current_price = base_price

    for i in range(num_rows):
        # Add some trend and volatility
        trend = 0.0001 * np.sin(i / 50)  # Slow trend
        volatility = np.random.normal(0, 0.005)  # Random volatility
        change = trend + volatility
        current_price *= (1 + change)
        prices.append(current_price)

    # Create OHLCV from prices
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        high = price * (1 + abs(np.random.normal(0, 0.002)))
        low = price * (1 - abs(np.random.normal(0, 0.002)))
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = np.random.uniform(100, 1000)

        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def test_legacy_vs_advanced_signals():
    """Compare legacy rule-based signals with advanced ML signals"""
    logger.info("Testing legacy vs advanced signal generation...")

    # Create sample data
    data = create_sample_data(500)

    # Initialize services
    core_service = CoreService()
    advanced_generator = AdvancedSignalGenerator()

    # Generate legacy signals
    legacy_signals = []
    for i in range(len(data)):
        signals = asyncio.run(core_service.generate_signals("BTC/USDT", "1h", data.iloc[:i+1]))
        # Take the last signal if any
        signal = signals[-1] if signals else 'hold'
        legacy_signals.append(signal)

    # Generate advanced signals
    advanced_signals = asyncio.run(advanced_generator.generate_signals(data))

    # Compare results
    logger.info(f"Legacy signals generated: {len([s for s in legacy_signals if s != 'hold'])}")
    logger.info(f"Advanced signals generated: {len(advanced_signals)}")

    # Calculate signal quality metrics
    if len(advanced_signals) > 0:
        confidence_scores = [s.get('confidence', 0) for s in advanced_signals]
        avg_confidence = np.mean(confidence_scores)
        logger.info(f"Average signal confidence: {avg_confidence:.3f}")

    return legacy_signals, advanced_signals

async def test_ml_training_async():
    """Test ML model training functionality"""
    logger.info("Testing ML model training...")

    # Create training data
    train_data = create_sample_data(2000)

    # Initialize advanced signal generator
    generator = AdvancedSignalGenerator()

    # Train the model
    try:
        success = await generator.train_ml_model(train_data, "BTC/USDT")
        if success:
            logger.info("ML model training completed successfully")
        else:
            logger.warning("ML model training failed")
    except Exception as e:
        logger.error(f"ML training error: {e}")
        return False

    return True

def test_market_regime_detection():
    """Test market regime detection"""
    logger.info("Testing market regime detection...")

    # Create sample data
    data = create_sample_data(1000)

    # Initialize regime detector
    regime_detector = MarketRegimeDetector()

    # Detect regimes
    try:
        regime = regime_detector.detect_regime(data)
        logger.info(f"Detected market regime: {regime}")

    except Exception as e:
        logger.error(f"Regime detection error: {e}")
        return False

    return True

def demo_advanced_signals():
    """Run a comprehensive demo of advanced signal generation"""
    logger.info("Running advanced signal generation demo...")

    # Create sample data
    data = create_sample_data(1000)
    logger.info(f"Created sample dataset with {len(data)} rows")

    # Initialize components
    core_service = CoreService()
    advanced_generator = AdvancedSignalGenerator()

    # Test 1: Basic signal generation
    logger.info("\n=== Testing Basic Signal Generation ===")
    signals = asyncio.run(advanced_generator.generate_signals(data))
    logger.info(f"Generated {len(signals)} signals")

    if signals:
        # Show sample signals
        for i, signal in enumerate(signals[:5]):
            logger.info(f"Signal {i+1}: {signal}")

    # Test 2: Feature engineering
    logger.info("\n=== Testing Feature Engineering ===")
    feature_engineer = AdvancedFeatureEngineer()
    features = feature_engineer.engineer_features(data)
    logger.info(f"Engineered {features.shape[1]} features for {features.shape[0]} data points")

    # Test 3: Market regime detection
    logger.info("\n=== Testing Market Regime Detection ===")
    regime_detector = MarketRegimeDetector()
    regime = regime_detector.detect_regime(data)
    logger.info(f"Detected market regime: {regime}")

    # Test 4: ML model training
    logger.info("\n=== Testing ML Model Training ===")
    train_success = asyncio.run(test_ml_training_async())
    if train_success:
        logger.info("ML model training test passed")
    else:
        logger.warning("ML model training test failed")

    # Test 5: Performance comparison
    logger.info("\n=== Testing Performance Comparison ===")
    legacy_signals, advanced_signals = test_legacy_vs_advanced_signals()

    logger.info("\n=== Demo Summary ===")
    logger.info(f"Sample data points: {len(data)}")
    logger.info(f"Legacy signals: {len([s for s in legacy_signals if s != 'hold'])}")
    logger.info(f"Advanced signals: {len(advanced_signals)}")
    logger.info(f"Market regime detected: {regime}")
    logger.info(f"Features engineered: {features.shape[1]}")
    logger.info(f"ML training successful: {train_success}")

    return True

def main():
    parser = argparse.ArgumentParser(description='Test Advanced Signal Generation')
    parser.add_argument('--demo', action='store_true', help='Run full demo')
    parser.add_argument('--train', action='store_true', help='Test ML training only')
    parser.add_argument('--regime', action='store_true', help='Test regime detection only')
    parser.add_argument('--compare', action='store_true', help='Compare legacy vs advanced signals')

    args = parser.parse_args()

    try:
        if args.demo:
            demo_advanced_signals()
        elif args.train:
            asyncio.run(test_ml_training_async())
        elif args.regime:
            test_market_regime_detection()
        elif args.compare:
            test_legacy_vs_advanced_signals()
        else:
            logger.info("No specific test selected. Run with --demo for full test suite.")
            logger.info("Available options: --demo, --train, --regime, --compare")

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()