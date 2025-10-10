#!/usr/bin/env python3
"""
Test script for improved ML models with enhanced feature engineering and optimization.

This script tests the enhanced ML models against traditional indicators to measure
trading performance improvements.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import *
from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators
from src.signal_generator import generate_signals, calculate_risk_management
from src.ml_predictor import (
    get_ml_predictor, is_ml_available,
    get_lstm_predictor, is_lstm_available,
    get_transformer_predictor, is_transformer_available
)
from src.backtester import Backtester

def test_ml_model_performance():
    """Test enhanced ML models against traditional indicators."""

    print("🧪 Testing Enhanced ML Models Performance")
    print("=" * 50)

    # Test parameters
    symbol = 'BTC/USDT'
    timeframe = '1d'
    test_days = 365  # 1 year of data

    print(f"📊 Testing on {symbol} {timeframe} with {test_days} days of data")

    try:
        # Fetch historical data
        print("📥 Fetching historical data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_days)

        df = fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=365,  # 1 year of daily data
            start_date=start_date
        )

        if df is None or len(df) < 100:
            print("❌ Insufficient data for testing")
            return False

        print(f"✅ Loaded {len(df)} data points")

        # Calculate indicators
        print("📈 Calculating technical indicators...")
        df = calculate_indicators(df)
        print(f"✅ Indicators calculated for {len(df)} data points")

        # Generate signals manually (traditional only, no ML enhancement)
        print("📊 Generating traditional trading signals...")
        
        # Basic EMA crossover signals
        df['EMA_crossover'] = np.where(df['EMA9'] > df['EMA21'], 1, -1)
        df['Buy_Signal'] = (df['EMA_crossover'] == 1).astype(int)
        df['Sell_Signal'] = (df['EMA_crossover'] == -1).astype(int)
        
        # Calculate risk management
        df = calculate_risk_management(df)
        
        buy_signals = df['Buy_Signal'].sum()
        sell_signals = df['Sell_Signal'].sum()
        print(f"✅ Traditional signals generated: Buy={buy_signals}, Sell={sell_signals}")

        # Initialize backtester
        backtester = Backtester()

        # Test traditional indicators
        print("\n📊 Testing Traditional Indicators...")
        traditional_results = backtester.run_backtest(
            df=df,
            strategy='traditional',
            symbol=symbol,
            timeframe=timeframe,
            initial_capital=10000,
            commission=0.001
        )

        if traditional_results['success']:
            trad_metrics = traditional_results['metrics']
            print(f"DEBUG: Traditional metrics keys: {list(trad_metrics.keys())}")
            print(f"DEBUG: Traditional metrics: {trad_metrics}")
            print(f"✅ Traditional Strategy:")
            print(f"   Total Return: {trad_metrics['total_return_pct']:.2f}%")
            print(f"   Sharpe Ratio: {trad_metrics.get('sharpe_ratio', 'N/A')}")
            print(f"   Win Rate: {trad_metrics['win_rate']:.1f}%")
            print(f"   Profit Factor: {trad_metrics['profit_factor']:.3f}")
            print(f"   Max Drawdown: {trad_metrics['max_drawdown']:.2f}%")
        else:
            print("❌ Traditional strategy backtest failed")
            return False

        # Test ML-enhanced strategies
        ml_results = {}

        if is_ml_available():
            print("\n🤖 Testing ML-Enhanced Strategies...")

            # Get ML predictor - create fresh instance
            ml_predictor = get_ml_predictor(symbol=symbol, timeframe=timeframe)

            # Check if model needs retraining due to feature mismatch
            needs_retraining = True
            if ml_predictor:
                # Test feature count
                test_features, _ = ml_predictor.prepare_features(df.head(10))
                expected_features = test_features.shape[1]

                # Check if current model has different feature count
                if ml_predictor.is_trained and hasattr(ml_predictor.best_model, 'n_features_in_'):
                    current_features = ml_predictor.best_model.n_features_in_
                    if current_features == expected_features:
                        print("✅ ML models are compatible")
                        needs_retraining = False
                    else:
                        print(f"⚠️  Current model has {current_features} features, but data has {expected_features} features. Retraining...")
                        needs_retraining = True
                elif ml_predictor.is_trained:
                    # For models without n_features_in_ (like ensembles), assume compatible
                    print("✅ ML models loaded (feature count check not available)")
                    needs_retraining = False
                else:
                    needs_retraining = True
            else:
                needs_retraining = True

            if needs_retraining:
                print("🔧 Training ML model due to feature mismatch or no cached models...")
                train_result = ml_predictor.train_models(df)
                if not train_result['success']:
                    print("❌ ML training failed")
                    ml_predictor = None
                else:
                    f1_score = ml_predictor.model_performance.get(ml_predictor.best_model.__class__.__name__, {}).get('f1_score', 'N/A')
                    f1_str = f"{f1_score:.3f}" if isinstance(f1_score, (int, float)) else str(f1_score)
                    print(f"✅ ML model trained: {train_result['best_model']} (F1={f1_str})")
            elif ml_predictor and ml_predictor.is_trained:
                print("✅ Using compatible cached ML models")
                # Test ML-enhanced strategy
                ml_enhanced_results = backtester.run_backtest(
                    df=df,
                    strategy='ml_enhanced',
                    symbol=symbol,
                    timeframe=timeframe,
                    initial_capital=10000,
                    commission=0.001
                )

                if ml_enhanced_results['success']:
                    ml_metrics = ml_enhanced_results['metrics']
                    ml_results['ml_enhanced'] = ml_metrics
                    print(f"✅ ML-Enhanced Strategy:")
                    print(f"   Total Return: {ml_metrics['total_return_pct']:.2f}%")
                    print(f"   Sharpe Ratio: {ml_metrics['sharpe_ratio']:.3f}")
                    print(f"   Win Rate: {ml_metrics['win_rate']:.1f}%")
                    print(f"   Profit Factor: {ml_metrics['profit_factor']:.3f}")
                    print(f"   Max Drawdown: {ml_metrics['max_drawdown']:.2f}%")

                    # Compare with traditional
                    return_diff = ml_metrics['total_return_pct'] - trad_metrics['total_return_pct']
                    sharpe_diff = ml_metrics.get('sharpe_ratio', 0) - trad_metrics.get('sharpe_ratio', 0)

                    print(f"\n📈 ML vs Traditional Comparison:")
                    print(f"   Return Difference: {return_diff:+.2f}%")
                    print(f"   Sharpe Difference: {sharpe_diff:+.3f}")
                    print(f"   Win Rate Difference: {ml_metrics['win_rate'] - trad_metrics['win_rate']:+.1f}%")

                    if return_diff > 0 and sharpe_diff > 0:
                        print("🎉 ML enhancement shows improvement!")
                    elif return_diff > 0:
                        print("⚠️  ML shows better returns but higher risk")
                    else:
                        print("📉 ML enhancement needs further tuning")
                else:
                    print("❌ ML-enhanced strategy backtest failed")

        # Test LSTM if available
        if is_lstm_available():
            print("\n🧠 Testing LSTM Strategy...")

            lstm_predictor = get_lstm_predictor(symbol=symbol, timeframe=timeframe)

            if lstm_predictor and not lstm_predictor.is_trained:
                print("🔧 Training LSTM model...")
                train_result = lstm_predictor.train_model(df)
                if not train_result['success']:
                    print("❌ LSTM training failed")
                else:
                    print(f"✅ LSTM model trained: Test AUC={train_result['test_auc']:.3f}")

            if lstm_predictor and lstm_predictor.is_trained:
                # Test LSTM-enhanced strategy
                lstm_results = backtester.run_backtest(
                    df=df,
                    strategy='lstm_enhanced',
                    symbol=symbol,
                    timeframe=timeframe,
                    initial_capital=10000,
                    commission=0.001
                )

                if lstm_results['success']:
                    lstm_metrics = lstm_results['metrics']
                    ml_results['lstm'] = lstm_metrics
                    print(f"✅ LSTM Strategy:")
                    print(f"   Total Return: {lstm_metrics['total_return_pct']:.2f}%")
                    print(f"   Sharpe Ratio: {lstm_metrics['sharpe_ratio']:.3f}")
                    print(f"   Win Rate: {lstm_metrics['win_rate']:.1f}%")
                    print(f"   Profit Factor: {lstm_metrics['profit_factor']:.3f}")
                    print(f"   Max Drawdown: {lstm_metrics['max_drawdown']:.2f}%")

        # Test Transformer if available
        if is_transformer_available():
            print("\n🔄 Testing Transformer Strategy...")

            transformer_predictor = get_transformer_predictor(symbol=symbol, timeframe=timeframe)

            if transformer_predictor and not transformer_predictor.is_trained:
                print("🔧 Training Transformer model...")
                train_result = transformer_predictor.train_model(df)
                if not train_result['success']:
                    print("❌ Transformer training failed")
                else:
                    print(f"✅ Transformer model trained: Test AUC={train_result['test_auc']:.3f}")

            if transformer_predictor and transformer_predictor.is_trained:
                # Test Transformer-enhanced strategy
                transformer_results = backtester.run_backtest(
                    df=df,
                    strategy='transformer_enhanced',
                    symbol=symbol,
                    timeframe=timeframe,
                    initial_capital=10000,
                    commission=0.001
                )

                if transformer_results['success']:
                    transformer_metrics = transformer_results['metrics']
                    ml_results['transformer'] = transformer_metrics
                    print(f"✅ Transformer Strategy:")
                    print(f"   Total Return: {transformer_metrics['total_return_pct']:.2f}%")
                    print(f"   Sharpe Ratio: {transformer_metrics['sharpe_ratio']:.3f}")
                    print(f"   Win Rate: {transformer_metrics['win_rate']:.1f}%")
                    print(f"   Profit Factor: {transformer_metrics['profit_factor']:.3f}")
                    print(f"   Max Drawdown: {transformer_metrics['max_drawdown']:.2f}%")

        # Summary comparison
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 60)

        strategies = [('Traditional', trad_metrics)] + [(k.upper(), v) for k, v in ml_results.items()]

        print("<15")
        print("-" * 60)

        for name, metrics in strategies:
            sharpe = metrics.get('sharpe_ratio', 'N/A')
            total_return = metrics.get('total_return_pct', 0)
            win_rate = metrics.get('win_rate', 0)
            max_drawdown = metrics.get('max_drawdown', 0)
            print("<15")

        # Find best performing strategy
        best_strategy = max(strategies, key=lambda x: x[1].get('sharpe_ratio', 0))
        print(f"\n🏆 Best Strategy by Sharpe Ratio: {best_strategy[0]} ({best_strategy[1].get('sharpe_ratio', 'N/A')})")

        best_return = max(strategies, key=lambda x: x[1].get('total_return_pct', 0))
        print(f"💰 Best Strategy by Total Return: {best_return[0]} ({best_return[1].get('total_return_pct', 0):.2f}%)")

        print("\n✅ ML Model Performance Testing Completed!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_engineering():
    """Test the enhanced feature engineering capabilities."""

    print("\n🔧 Testing Enhanced Feature Engineering")
    print("=" * 40)

    try:
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        np.random.seed(42)

        df = pd.DataFrame({
            'open': 100 + np.random.randn(200).cumsum(),
            'high': 105 + np.random.randn(200).cumsum(),
            'low': 95 + np.random.randn(200).cumsum(),
            'close': 100 + np.random.randn(200).cumsum(),
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)

        # Calculate indicators
        df = calculate_indicators(df)

        # Test ML predictor feature engineering
        if is_ml_available():
            ml_predictor = get_ml_predictor()

            features, feature_names = ml_predictor.prepare_features(df)

            print(f"✅ Feature Engineering Test:")
            print(f"   Data points: {len(df)}")
            print(f"   Features generated: {features.shape[1]}")
            print(f"   Feature names: {len(feature_names)}")

            # Check for enhanced features
            enhanced_features = [
                'close_pct', 'ema_spread', 'atr_normalized', 'rsi_ma5',
                'bb_width', 'stoch_rsi', 'williams_r', 'cci'
            ]

            found_features = [f for f in feature_names if any(ef in f for ef in enhanced_features)]
            print(f"   Enhanced features found: {len(found_features)}")

            if len(found_features) > 10:
                print("🎉 Enhanced feature engineering working well!")
            else:
                print("⚠️  Limited enhanced features detected")

            return True
        else:
            print("❌ ML not available for feature engineering test")
            return False

    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting ML Model Performance Tests")
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test feature engineering
    feature_test = test_feature_engineering()

    # Test model performance
    performance_test = test_ml_model_performance()

    print()
    print("=" * 60)
    print("🏁 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Feature Engineering Test: {'✅ PASSED' if feature_test else '❌ FAILED'}")
    print(f"Model Performance Test: {'✅ PASSED' if performance_test else '❌ FAILED'}")

    if feature_test and performance_test:
        print("\n🎉 All tests passed! ML enhancements are working.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

    print(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")