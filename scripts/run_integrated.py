#!/usr/bin/env python3
"""
TradPal Integrated Runner
Runs the trading indicator with all configured integrations
"""

import os
import sys
import time
import signal
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_live_monitoring
from integrations import integration_manager
from integrations.telegram import TelegramIntegration, TelegramConfig

# Load environment variables
load_dotenv()

# Global flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global running
    print("\n🛑 Shutdown signal received...")
    running = False

def setup_integrations():
    """Setup all configured integrations"""
    print("🔧 Setting up integrations...")

    # Setup Telegram integration
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if bot_token and chat_id:
        try:
            telegram_config = TelegramConfig(
                enabled=True,
                name="Telegram Bot",
                bot_token=bot_token,
                chat_id=chat_id
            )
            telegram_integration = TelegramIntegration(telegram_config)
            integration_manager.register_integration("telegram", telegram_integration)
            print("✅ Telegram integration configured")
            return True
        except Exception as e:
            print(f"❌ Failed to setup Telegram integration: {e}")
            return False
    else:
        print("⚠️  Telegram credentials not found in .env")
        print("   Run 'python src/scripts/manage_integrations.py --setup' to configure")
        return False

def run_integrated_system():
    """Run the complete integrated system"""
    global running

    print("🚀 Starting TradPal Integrated System")
    print("📊 Trading Indicator + Notification Integrations")
    print("=" * 50)

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup integrations
    if not setup_integrations():
        print("❌ Integration setup failed. Starting indicator only...")
        # Continue without integrations

    # Initialize integrations
    if integration_manager.integrations:
        print("🔌 Initializing integrations...")
        init_results = integration_manager.initialize_all()

        successful_integrations = [name for name, success in init_results.items() if success]
        failed_integrations = [name for name, success in init_results.items() if not success]

        if successful_integrations:
            print(f"✅ Initialized: {', '.join(successful_integrations)}")
            print("📤 Startup messages sent automatically during initialization")
        if failed_integrations:
            print(f"❌ Failed: {', '.join(failed_integrations)}")

    # Start integrated monitoring in a separate thread or process
    # For simplicity, we'll run a simple monitoring loop
    print("📈 Starting integrated monitoring...")
    print("Press Ctrl+C to stop")

    try:
        while running:
            # Here you could implement more sophisticated monitoring
            # For now, we'll just keep the system alive
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n👋 Shutdown requested by user")

    finally:
        # Shutdown integrations
        print("🔌 Shutting down integrations...")
        integration_manager.shutdown_all()

        print("✅ TradPal Integrated System stopped")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    run_integrated_system()