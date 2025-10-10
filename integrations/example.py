#!/usr/bin/env python3
"""
TradPal - Integration Example
Demonstrates how to use all available integrations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'integrations'))

from integrations.base import integration_manager
from integrations.telegram.bot import TelegramIntegration, TelegramConfig
from integrations.email_integration.email import EmailIntegration, EmailConfig
from integrations.discord import DiscordIntegration, DiscordConfig
from integrations.webhook.webhook import WebhookIntegration, WebhookConfig
from integrations.sms.sms import SMSIntegration, SMSConfig


def setup_example_integrations():
    """Setup example integrations for demonstration"""

    print("🚀 Setting up TradPal Integrations Example")
    print("=" * 50)

    # Example Telegram integration
    telegram_config = TelegramConfig(
        enabled=True,
        name="Example Telegram",
        bot_token="YOUR_TELEGRAM_BOT_TOKEN",
        chat_id="YOUR_CHAT_ID"
    )
    telegram = TelegramIntegration(telegram_config)
    integration_manager.register_integration("telegram", telegram)

    # Example Email integration
    email_config = EmailConfig(
        enabled=True,
        name="Example Email",
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username="your-email@gmail.com",
        password="your-app-password",
        recipients=["recipient@example.com"],
        use_tls=True
    )
    email = EmailIntegration(email_config)
    integration_manager.register_integration("email", email)

    # Example Discord integration
    discord_config = DiscordConfig(
        enabled=True,
        name="Example Discord",
        webhook_url="https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN",
        username="TradPal Bot",
        embed_color=0x00ff00
    )
    discord = DiscordIntegration(discord_config)
    integration_manager.register_integration("discord", discord)

    # Example Webhook integration
    webhook_config = WebhookConfig(
        enabled=True,
        name="Example Webhook",
        webhook_urls=["https://httpbin.org/post", "https://webhook.site/YOUR_WEBHOOK_ID"],
        method="POST",
        auth_token="your-bearer-token",
        auth_type="Bearer"
    )
    webhook = WebhookIntegration(webhook_config)
    integration_manager.register_integration("webhook", webhook)

    # Example SMS integration (requires twilio package)
    sms_config = SMSConfig(
        enabled=True,
        name="Example SMS",
        account_sid="YOUR_TWILIO_ACCOUNT_SID",
        auth_token="YOUR_TWILIO_AUTH_TOKEN",
        from_number="+1234567890",
        to_numbers=["+0987654321"]
    )
    sms = SMSIntegration(sms_config)
    integration_manager.register_integration("sms", sms)

    print("✅ All integrations registered!")


def test_all_integrations():
    """Test all integrations with a sample signal"""

    print("\n🧪 Testing All Integrations")
    print("=" * 30)

    # Initialize all integrations
    init_results = integration_manager.initialize_all()
    print("📋 Initialization Results:")
    for name, success in init_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {name}")

    # Create a sample trading signal
    sample_signal = {
        "timestamp": "2024-01-15T10:30:00Z",
        "symbol": "EUR/USD",
        "timeframe": "1m",
        "signal_type": "BUY",
        "price": 1.05234,
        "indicators": {
            "ema9": 1.05210,
            "ema21": 1.05195,
            "rsi": 65.5,
            "bb_upper": 1.05350,
            "bb_middle": 1.05225,
            "bb_lower": 1.05100,
            "atr": 0.00050
        },
        "risk_management": {
            "position_size_percent": 2.0,
            "position_size_absolute": 2000.0,
            "stop_loss_buy": 1.05150,
            "take_profit_buy": 1.05400,
            "leverage": 5.0
        },
        "confidence": 0.85,
        "reason": "Strong bullish momentum with RSI above 60 and price above EMA21"
    }

    print("\n📤 Sending sample BUY signal to all integrations...")
    # Send signal to all integrations
    send_results = integration_manager.send_signal_to_all(sample_signal)
    print("📋 Send Results:")
    for name, success in send_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {name}")

    # Test connections
    print("\n🔍 Testing connections...")
    test_results = integration_manager.test_all_connections()
    print("📋 Connection Test Results:")
    for name, success in test_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {name}")

    # Show integration status
    print("\n📊 Integration Status Overview:")
    status = integration_manager.get_status_overview()
    print(f"   Total: {status['total_integrations']}")
    print(f"   Enabled: {status['enabled_integrations']}")
    print("   Details:")
    for name, info in status['integrations'].items():
        enabled = "✅" if info['enabled'] else "❌"
        initialized = "✅" if info['initialized'] else "❌"
        print(f"      {name}: {enabled} enabled, {initialized} initialized")

    # Shutdown
    print("\n🛑 Shutting down integrations...")
    integration_manager.shutdown_all()
    print("✅ All integrations shut down.")


def show_configuration_help():
    """Show configuration help for all integrations"""

    print("\n📚 Integration Configuration Help")
    print("=" * 40)

    configs = {
        "Telegram": {
            "env_vars": ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"],
            "setup": "Create a bot with @BotFather on Telegram",
            "docs": "https://core.telegram.org/bots"
        },
        "Email": {
            "env_vars": ["EMAIL_USERNAME", "EMAIL_PASSWORD", "EMAIL_RECIPIENTS"],
            "setup": "Use Gmail/App Passwords or SMTP provider",
            "docs": "Configure SMTP settings in integrations/email/config.py"
        },
        "Discord": {
            "env_vars": ["DISCORD_WEBHOOK_URL"],
            "setup": "Create webhook in Discord server settings",
            "docs": "https://discord.com/developers/docs/resources/webhook"
        },
        "Webhook": {
            "env_vars": ["WEBHOOK_URLS", "WEBHOOK_AUTH_TOKEN"],
            "setup": "Configure HTTP endpoints to receive POST requests",
            "docs": "Supports custom headers and authentication"
        },
        "SMS (Twilio)": {
            "env_vars": ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER", "SMS_TO_NUMBERS"],
            "setup": "Sign up at twilio.com and get phone numbers",
            "docs": "https://www.twilio.com/docs/sms",
            "note": "Requires: pip install twilio"
        }
    }

    for name, info in configs.items():
        print(f"\n🔧 {name}")
        print(f"   Environment Variables: {', '.join(info['env_vars'])}")
        print(f"   Setup: {info['setup']}")
        if 'docs' in info:
            print(f"   Docs: {info['docs']}")
        if 'note' in info:
            print(f"   Note: {info['note']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_configuration_help()
    else:
        setup_example_integrations()
        test_all_integrations()

        print("\n💡 Tip: Run with --help to see configuration instructions")
        print("💡 Tip: Configure your .env file with actual credentials for real testing")