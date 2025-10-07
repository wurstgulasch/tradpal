#!/usr/bin/env python3
"""
TradPal Integration Manager
Central script for managing all integrations
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations import integration_manager
from integrations.telegram import TelegramIntegration, TelegramConfig
from integrations.email_integration import EmailIntegration, EmailConfig

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def setup_integrations():
    """Setup all configured integrations"""
    print("🔧 Setting up TradPal Integrations...")

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
        except Exception as e:
            print(f"❌ Failed to setup Telegram integration: {e}")
    else:
        print("⚠️  Telegram credentials not found. Run 'python integrations/telegram/config.py' to setup")

    # Setup Email integration
    email_username = os.getenv('EMAIL_USERNAME')
    email_password = os.getenv('EMAIL_PASSWORD')
    email_recipients = os.getenv('EMAIL_RECIPIENTS')

    if email_username and email_password and email_recipients:
        try:
            email_config = EmailConfig.from_env()
            email_integration = EmailIntegration(email_config)
            integration_manager.register_integration("email", email_integration)
            print("✅ Email integration configured")
        except Exception as e:
            print(f"❌ Failed to setup Email integration: {e}")
    else:
        print("⚠️  Email credentials not found. Run 'python integrations/email/config.py' to setup")

    # Future: Setup other integrations here

def test_integrations():
    """Test all configured integrations"""
    print("🧪 Testing integrations...")

    # Initialize integrations
    init_results = integration_manager.initialize_all()
    print("\n📊 Initialization Results:")
    for name, success in init_results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")

    # Test connections
    connection_results = integration_manager.test_all_connections()
    print("\n🔌 Connection Test Results:")
    for name, success in connection_results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")

    # Overall status
    all_success = all(init_results.values()) and all(connection_results.values())
    if all_success:
        print("\n🎉 All integrations are working!")
    else:
        print("\n⚠️  Some integrations have issues. Check the logs above.")

    return all_success

def show_status():
    """Show status of all integrations"""
    status = integration_manager.get_status_overview()

    print("📊 Integration Status Overview")
    print("=" * 40)
    print(f"Total Integrations: {status['total_integrations']}")
    print(f"Enabled Integrations: {status['enabled_integrations']}")

    print("\n🔧 Integration Details:")
    for name, info in status['integrations'].items():
        enabled = "✅" if info['enabled'] else "❌"
        initialized = "✅" if info['initialized'] else "❌"
        print(f"  {name}: {enabled} Enabled, {initialized} Initialized")

def interactive_setup():
    """Interactive setup wizard"""
    print("🎯 TradPal Integration Setup Wizard")
    print("=" * 40)

    # Telegram setup
    setup_telegram = input("Setup Telegram bot integration? (y/n): ").lower().strip()
    if setup_telegram == 'y':
        print("\n🤖 Setting up Telegram integration...")
        from integrations.telegram.config import setup_telegram_integration
        if setup_telegram_integration():
            print("✅ Telegram setup completed!")
        else:
            print("❌ Telegram setup failed")

    # Email setup
    setup_email = input("Setup Email integration? (y/n): ").lower().strip()
    if setup_email == 'y':
        print("\n📧 Setting up Email integration...")
        from integrations.email_integration.config import setup_email_integration
        try:
            email_config = setup_email_integration()
            email_integration = EmailIntegration(email_config)
            integration_manager.register_integration("email", email_integration)
            print("✅ Email setup completed!")
        except Exception as e:
            print(f"❌ Email setup failed: {e}")

    # Future integrations can be added here
    print("\n🎉 Setup wizard completed!")
    print("Run 'python scripts/manage_integrations.py --test' to verify everything works")

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='TradPal Integration Manager')
    parser.add_argument('--setup', action='store_true', help='Run interactive setup wizard')
    parser.add_argument('--test', action='store_true', help='Test all integrations')
    parser.add_argument('--status', action='store_true', help='Show integration status')
    parser.add_argument('--init', action='store_true', help='Initialize all integrations')

    args = parser.parse_args()

    if args.setup:
        interactive_setup()
    elif args.test:
        setup_integrations()
        test_integrations()
    elif args.status:
        setup_integrations()
        show_status()
    elif args.init:
        setup_integrations()
        init_results = integration_manager.initialize_all()
        print("🔧 Initialization completed")
        for name, success in init_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {name}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()