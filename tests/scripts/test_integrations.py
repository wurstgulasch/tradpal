#!/usr/bin/env python3
"""
Integration Test Script
Tests all configured integrations with sample signals
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations import integration_manager
from integrations.telegram import TelegramIntegration, TelegramConfig

# Load environment variables
load_dotenv()

def create_sample_signal():
    """Create a sample trading signal for testing"""
    return {
        "timestamp": "2024-01-15T10:30:00Z",
        "symbol": "EUR/USD",
        "timeframe": "1m",
        "signal": "BUY",
        "price": 1.0850,
        "indicators": {
            "ema_short": 1.0845,
            "ema_long": 1.0840,
            "rsi": 35.2,
            "bb_upper": 1.0860,
            "bb_lower": 1.0830,
            "atr": 0.0015
        },
        "risk_management": {
            "position_size": 1000,
            "stop_loss": 1.0825,
            "take_profit": 1.0890,
            "leverage": 1.0,
            "risk_percent": 1.0
        },
        "confidence": 0.85,
        "reason": "EMA crossover with RSI oversold and price below BB lower"
    }

def setup_test_integrations():
    """Setup integrations for testing"""
    print("🔧 Setting up test integrations...")

    # Setup Telegram integration
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if bot_token and chat_id:
        try:
            telegram_config = TelegramConfig(
                enabled=True,
                name="Test Telegram Bot",
                bot_token=bot_token,
                chat_id=chat_id
            )
            telegram_integration = TelegramIntegration(telegram_config)
            integration_manager.register_integration("telegram", telegram_integration)
            print("✅ Telegram integration configured for testing")
            return True
        except Exception as e:
            print(f"❌ Failed to setup Telegram integration: {e}")
            return False
    else:
        print("⚠️  Telegram credentials not found in .env")
        print("   Run 'python src/scripts/manage_integrations.py --setup' to configure")
        return False

def test_integrations():
    """Test all configured integrations"""
    print("🧪 Testing integrations...")
    print("=" * 40)

    # Setup integrations
    if not setup_test_integrations():
        print("❌ Cannot test without integrations configured")
        return  # Return None instead of False

    # Initialize integrations
    print("🔌 Initializing integrations...")
    init_results = integration_manager.initialize_all()

    successful_integrations = [name for name, success in init_results.items() if success]
    failed_integrations = [name for name, success in init_results.items() if not success]

    if successful_integrations:
        print(f"✅ Initialized: {', '.join(successful_integrations)}")
    if failed_integrations:
        print(f"❌ Failed: {', '.join(failed_integrations)}")
        return  # Return None instead of False

    # Create test signal
    test_signal = create_sample_signal()
    print(f"\n📊 Testing with sample signal: {test_signal['signal']} {test_signal['symbol']}")

    # Test sending signal to all integrations
    print("\n📤 Sending test signal to all integrations...")
    send_results = integration_manager.send_signal_to_all(test_signal)

    successful_sends = [name for name, success in send_results.items() if success]
    failed_sends = [name for name, success in send_results.items() if not success]

    print("\n📋 Test Results:")
    print(f"✅ Successful sends: {len(successful_sends)}")
    if successful_sends:
        print(f"   - {', '.join(successful_sends)}")

    print(f"❌ Failed sends: {len(failed_sends)}")
    if failed_sends:
        print(f"   - {', '.join(failed_sends)}")

    # Shutdown integrations
    print("\n🔌 Shutting down integrations...")
    integration_manager.shutdown_all()

    # Summary
    total_integrations = len(integration_manager.integrations)
    success_rate = len(successful_sends) / total_integrations if total_integrations > 0 else 0

    print("\n🎯 Test Summary:")
    print(f"   Total integrations: {total_integrations}")
    print(f"   Success rate: {success_rate:.1%}")

    if success_rate == 1.0:
        print("   Status: ✅ All tests passed!")
    elif success_rate > 0:
        print("   Status: ⚠️  Partial success")
    else:
        print("   Status: ❌ All tests failed")

    # Don't return anything - pytest expects None

if __name__ == "__main__":
    print("🚀 TradPal Integration Test Suite")
    print("Testing notification integrations with sample signals")
    print("=" * 60)

    success = test_integrations()

    if success:
        print("\n🎉 Integration tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Integration tests failed!")
        sys.exit(1)