#!/usr/bin/env python3
"""
Test script for Sentiment Analysis integration.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def test_sentiment_analysis():
    """Test sentiment analysis integration."""
    print("Testing Sentiment Analysis integration...")

    try:
        # Test sentiment analyzer import
        print("📊 Testing sentiment analyzer import...")
        from src.sentiment_analyzer import get_sentiment_score, SENTIMENT_ENABLED
        print(f"   Sentiment analysis enabled: {SENTIMENT_ENABLED}")

        if SENTIMENT_ENABLED:
            # Test sentiment score retrieval
            print("🧠 Testing sentiment score retrieval...")
            score, confidence = get_sentiment_score()
            print(".3f")
        else:
            print("⚠️  Sentiment analysis is disabled in configuration")
            print("💡 To enable sentiment analysis, set SENTIMENT_ENABLED=true in your environment or .env file")
            print("   Required API keys: TWITTER_BEARER_TOKEN, NEWS_API_KEY, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET")
            return True

        # Test signal generator with sentiment
        print("📊 Testing signal generation with sentiment...")
        from src.signal_generator import generate_signals, calculate_indicators

        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Generate realistic BTC/USDT price data
        base_price = 30000
        prices = []
        current_price = base_price

        for i in range(100):
            trend = 0.001 * np.sin(i / 20)
            noise = np.random.normal(0, 0.02)
            change = trend + noise
            current_price *= (1 + change)
            prices.append(current_price)

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': [np.random.uniform(1000000, 5000000) for _ in range(100)]
        })
        df.set_index('timestamp', inplace=True)

        # Calculate indicators
        df = calculate_indicators(df)

        # Generate signals (this will include sentiment enhancement)
        df_signals = generate_signals(df)

        # Check for sentiment columns
        sentiment_columns = ['Sentiment_Score', 'Sentiment_Confidence', 'Sentiment_Enhanced']
        found_columns = [col for col in sentiment_columns if col in df_signals.columns]

        print(f"   Found sentiment columns: {found_columns}")

        if found_columns:
            # Show sample sentiment-enhanced signals
            sentiment_signals = df_signals[df_signals['Sentiment_Enhanced'] == True]
            print(f"   Sentiment-enhanced signals: {len(sentiment_signals)}")

            if len(sentiment_signals) > 0:
                sample = sentiment_signals.head(1)
                print("   Sample sentiment-enhanced signal:")
                print(f"     Signal: {sample['Enhanced_Signal'].iloc[0]}")
                print(".3f")
                print(".3f")

        print("\n✅ Sentiment analysis integration test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sentiment_analysis()
    sys.exit(0 if success else 1)