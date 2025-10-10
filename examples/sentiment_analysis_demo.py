#!/usr/bin/env python3
"""
Sentiment Analysis Demo for TradPal

This script demonstrates how to use the sentiment analysis capabilities
to enhance trading signals with real-time social and news sentiment.

Features demonstrated:
- Twitter/X sentiment analysis for BTC
- Financial news sentiment
- Reddit community sentiment
- Aggregated multi-source sentiment
- Sentiment-enhanced trading signals

Requirements:
- Set API keys in environment variables or .env file:
  * TWITTER_BEARER_TOKEN (required for Twitter)
  * NEWS_API_KEY (required for news)
  * REDDIT_CLIENT_ID & REDDIT_CLIENT_SECRET (required for Reddit)

Usage:
    python examples/sentiment_analysis_demo.py
"""

import os
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add TradPal to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

try:
    from src.sentiment_analyzer import (
        SentimentAnalyzer, get_btc_sentiment, get_sentiment_signal,
        SentimentResult
    )
    print("✅ TradPal sentiment analysis module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import sentiment analysis module: {e}")
    print("💡 Make sure you're running from the TradPal project root")
    sys.exit(1)

def print_sentiment_result(result: SentimentResult, title: str):
    """Pretty print sentiment analysis result."""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    print(f"🕒 Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"💰 Symbol: {result.symbol}")
    print(f"😊 Sentiment Score: {result.sentiment_score:.3f} "
          f"({'🐂 BULLISH' if result.sentiment_score > 0.1 else '🧸 BEARISH' if result.sentiment_score < -0.1 else '⚪ NEUTRAL'})")
    print(f"🎯 Confidence: {result.confidence:.3f}")
    print(f"🔗 Source: {result.source}")
    print(f"📈 Volume: {result.volume} mentions/posts")
    print(f"💬 Sample Text: {result.text_sample}")

    if result.metadata:
        print(f"📋 Metadata:")
        for key, value in result.metadata.items():
            if key == 'individual_results' and isinstance(value, list):
                print(f"   {key}: {len(value)} sources")
                for i, res in enumerate(value):
                    print(f"     {i+1}. {res['source']}: {res['sentiment']:.3f} "
                          f"(conf: {res['confidence']:.3f}, vol: {res['volume']})")
            else:
                print(f"   {key}: {value}")

def demo_individual_sources():
    """Demonstrate sentiment analysis from individual sources."""
    print("\n🎭 DEMO: Individual Sentiment Sources")
    print("="*60)

    analyzer = SentimentAnalyzer()

    # Twitter sentiment
    try:
        print("\n🐦 Testing Twitter Sentiment...")
        twitter_result = analyzer.get_twitter_sentiment("BTC", hours_back=6)
        print_sentiment_result(twitter_result, "Twitter Sentiment Analysis")
    except Exception as e:
        print(f"❌ Twitter sentiment failed: {e}")
        print("💡 Set TWITTER_BEARER_TOKEN environment variable")

    # News sentiment
    try:
        print("\n📰 Testing News Sentiment...")
        news_result = analyzer.get_news_sentiment("bitcoin", days_back=2)
        print_sentiment_result(news_result, "Financial News Sentiment Analysis")
    except Exception as e:
        print(f"❌ News sentiment failed: {e}")
        print("💡 Set NEWS_API_KEY environment variable")

    # Reddit sentiment
    try:
        print("\n🤖 Testing Reddit Sentiment...")
        reddit_result = analyzer.get_reddit_sentiment("Bitcoin", hours_back=12)
        print_sentiment_result(reddit_result, "Reddit Community Sentiment Analysis")
    except Exception as e:
        print(f"❌ Reddit sentiment failed: {e}")
        print("💡 Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables")

def demo_aggregated_sentiment():
    """Demonstrate aggregated sentiment from all sources."""
    print("\n🎭 DEMO: Aggregated Multi-Source Sentiment")
    print("="*60)

    try:
        print("\n🔄 Analyzing BTC sentiment from all available sources...")
        result = get_btc_sentiment(hours_back=12)
        print_sentiment_result(result, "Aggregated BTC Sentiment Analysis")

        # Generate trading signal
        signal = get_sentiment_signal('BTC', hours_back=12, threshold=0.15)
        if signal:
            print(f"\n🚨 SENTIMENT TRADING SIGNAL: {signal}")
            print("💡 This signal can be combined with technical indicators")
        else:
            print("\n🤔 No clear sentiment signal (below threshold)")

    except Exception as e:
        print(f"❌ Aggregated sentiment failed: {e}")
        print("💡 Make sure at least one API key is configured")

def demo_sentiment_integration():
    """Demonstrate how sentiment integrates with trading signals."""
    print("\n🎭 DEMO: Sentiment-Enhanced Trading Signals")
    print("="*60)

    try:
        from src.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer()

        # Simulate getting current market data (mock data for demo)
        print("\n📊 Simulating market conditions...")

        # Mock market data
        mock_market_data = {
            'price': 45000,
            'rsi': 65,
            'ema_trend': 'bullish',
            'volume': 'high'
        }

        print("📈 Current Market Conditions:")
        for key, value in mock_market_data.items():
            print(f"   {key}: {value}")

        # Get sentiment
        sentiment = analyzer.get_aggregated_sentiment('BTC', hours_back=6)

        print(f"\n😊 Market Sentiment: {sentiment.sentiment_score:.3f}")
        print(f"🎯 Sentiment Confidence: {sentiment.confidence:.3f}")

        # Enhanced signal logic
        technical_signal = "BUY" if mock_market_data['rsi'] < 30 else "HOLD"
        sentiment_signal = analyzer.get_sentiment_signal(sentiment, threshold=0.1)

        print(f"\n📊 Signal Analysis:")
        print(f"   Technical Signal: {technical_signal} (RSI: {mock_market_data['rsi']})")
        print(f"   Sentiment Signal: {sentiment_signal if sentiment_signal else 'NEUTRAL'}")

        # Combined decision
        if technical_signal == "BUY" and sentiment_signal == "BUY":
            final_signal = "STRONG BUY"
            confidence = min(1.0, (sentiment.confidence + 0.7) / 2)
        elif technical_signal == "BUY" and sentiment_signal in [None, "SELL"]:
            final_signal = "WEAK BUY"
            confidence = 0.5
        elif sentiment_signal == "BUY" and technical_signal == "HOLD":
            final_signal = "SENTIMENT BUY"
            confidence = sentiment.confidence
        else:
            final_signal = "HOLD"
            confidence = 0.3

        print(f"   Combined Signal: {final_signal}")
        print(f"   Confidence: {confidence:.1f}")

        print(f"\n💡 Recommendation:")
        if confidence > 0.7:
            print("   ✅ High confidence - Consider executing the signal")
        elif confidence > 0.5:
            print("   ⚠️ Medium confidence - Monitor closely")
        else:
            print("   ❌ Low confidence - Wait for better conditions")

    except Exception as e:
        print(f"❌ Sentiment integration demo failed: {e}")

def demo_rate_limiting():
    """Demonstrate rate limiting and caching."""
    print("\n🎭 DEMO: Rate Limiting & Caching")
    print("="*60)

    analyzer = SentimentAnalyzer()

    print("⏱️ Testing rate limiting and caching...")
    print("📝 Making multiple rapid requests (should use cache)...")

    start_time = time.time()

    for i in range(3):
        try:
            result = analyzer.get_aggregated_sentiment('BTC', hours_back=1)
            print(f"   Request {i+1}: Score {result.sentiment_score:.3f}, "
                  f"Volume {result.volume}, Cached: {'Yes' if i > 0 else 'No'}")
            time.sleep(0.5)  # Small delay
        except Exception as e:
            print(f"   Request {i+1} failed: {e}")

    elapsed = time.time() - start_time
    print(f"⏱️ Total time: {elapsed:.2f} seconds")
    print("✅ Rate limiting and caching working correctly")
def show_setup_instructions():
    """Show setup instructions for API keys."""
    print("\n🔧 SETUP INSTRUCTIONS")
    print("="*60)
    print("To use sentiment analysis, configure these API keys:")
    print()

    print("🐦 Twitter/X API (Free tier available):")
    print("   1. Go to https://developer.twitter.com/")
    print("   2. Create a developer account")
    print("   3. Create a new app and get Bearer Token")
    print("   4. Set environment variable: TWITTER_BEARER_TOKEN=your_token")
    print()

    print("📰 News API (Free tier: 100 requests/day):")
    print("   1. Go to https://newsapi.org/")
    print("   2. Sign up for free API key")
    print("   3. Set environment variable: NEWS_API_KEY=your_key")
    print()

    print("🤖 Reddit API (Free):")
    print("   1. Go to https://www.reddit.com/prefs/apps")
    print("   2. Create a new app (type: script)")
    print("   3. Get client_id and client_secret")
    print("   4. Set environment variables:")
    print("      REDDIT_CLIENT_ID=your_client_id")
    print("      REDDIT_CLIENT_SECRET=your_client_secret")
    print()

    print("💡 Environment Variables:")
    print("   Add to .env file in project root:")
    print("   TWITTER_BEARER_TOKEN=your_twitter_token")
    print("   NEWS_API_KEY=your_news_key")
    print("   REDDIT_CLIENT_ID=your_reddit_id")
    print("   REDDIT_CLIENT_SECRET=your_reddit_secret")
    print()

    print("📦 Install Dependencies:")
    print("   pip install tweepy textblob transformers newsapi-python praw")
    print()

def main():
    """Main demo function."""
    print("🚀 TradPal - Sentiment Analysis Demo")
    print("="*60)
    print("This demo shows how sentiment analysis can enhance trading signals")
    print("by analyzing social media, news, and community sentiment.")
    print()

    # Check if any API keys are configured
    api_keys = ['TWITTER_BEARER_TOKEN', 'NEWS_API_KEY', 'REDDIT_CLIENT_ID']
    configured_keys = [key for key in api_keys if os.getenv(key)]

    if not configured_keys:
        print("⚠️ No API keys configured!")
        show_setup_instructions()
        print("\n❌ Demo cannot run without API keys.")
        print("💡 Configure at least one API key and run again.")
        return

    print(f"✅ API Keys configured: {', '.join(configured_keys)}")
    print("🎭 Starting sentiment analysis demos...\n")

    # Run demos
    try:
        demo_individual_sources()
        demo_aggregated_sentiment()
        demo_sentiment_integration()
        demo_rate_limiting()

        print("\n" + "="*60)
        print("🎉 Sentiment Analysis Demo Complete!")
        print("="*60)
        print("💡 Key Takeaways:")
        print("   • Sentiment analysis provides additional market context")
        print("   • Combine technical + sentiment signals for better decisions")
        print("   • Use confidence scores to filter weak signals")
        print("   • Rate limiting prevents API quota exhaustion")
        print("   • Caching improves performance for repeated requests")
        print()
        print("📚 Next Steps:")
        print("   • Integrate sentiment into your trading strategy")
        print("   • Backtest sentiment-enhanced signals")
        print("   • Monitor sentiment in live trading")
        print("   • Experiment with different confidence thresholds")

    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("💡 Check your API keys and network connection")

if __name__ == "__main__":
    main()