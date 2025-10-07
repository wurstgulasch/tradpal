"""
Configuration utilities for Telegram integration.
"""

import os
import requests
from typing import Optional

def get_telegram_chat_id(bot_token: str) -> Optional[str]:
    """Get the chat ID by checking recent messages to the bot"""
    if not bot_token:
        print("❌ No bot token provided")
        return None

    base_url = f"https://api.telegram.org/bot{bot_token}"

    try:
        # Get updates (last messages)
        response = requests.get(f"{base_url}/getUpdates", timeout=10)
        response.raise_for_status()

        data = response.json()

        if data['ok'] and data['result']:
            print("📨 Recent messages to your bot:")
            for update in data['result'][-5:]:  # Show last 5 messages
                if 'message' in update:
                    chat_id = update['message']['chat']['id']
                    username = update['message']['chat'].get('username', 'Unknown')
                    text = update['message']['text']
                    print(f"  👤 User: @{username}")
                    print(f"  🆔 Chat ID: {chat_id}")
                    print(f"  💬 Message: {text}")
                    print("  " + "="*50)

            # Return the most recent chat ID
            if data['result']:
                latest_chat_id = data['result'][-1]['message']['chat']['id']
                print(f"\n🎯 Your Chat ID is: {latest_chat_id}")
                return str(latest_chat_id)

        else:
            print("❌ No messages found. Send a message to your bot first!")
            print("1. Open Telegram and find your bot")
            print("2. Send any message to the bot")
            print("3. Run this script again")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your TELEGRAM_BOT_TOKEN is correct")
        return None

def setup_telegram_integration():
    """Interactive setup for Telegram integration"""
    print("🤖 TradPal Telegram Bot Setup")
    print("=" * 40)

    # Check if already configured
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if bot_token and chat_id:
        print("✅ Telegram bot is already configured!")
        print(f"Bot Token: {bot_token[:20]}...")
        print(f"Chat ID: {chat_id}")
        return True

    # Get bot token
    if not bot_token:
        print("\n1. Create a bot with @BotFather on Telegram")
        print("   Send /newbot and follow the instructions")
        bot_token = input("Enter your bot token: ").strip()
        if not bot_token:
            print("❌ No bot token provided")
            return False

    # Test bot token
    print("\n🔍 Testing bot token...")
    try:
        response = requests.get(f"https://api.telegram.org/bot{bot_token}/getMe", timeout=10)
        if response.status_code == 200 and response.json().get('ok'):
            bot_info = response.json()['result']
            print(f"✅ Connected to bot: @{bot_info['username']}")
        else:
            print("❌ Invalid bot token")
            return False
    except Exception as e:
        print(f"❌ Error testing bot token: {e}")
        return False

    # Get chat ID
    if not chat_id:
        print("\n2. Send a message to your bot")
        print("   Open Telegram, find your bot, and send any message")
        input("Press Enter when you've sent a message to your bot...")

        chat_id = get_telegram_chat_id(bot_token)
        if not chat_id:
            print("❌ Failed to get chat ID")
            return False

    # Save to .env
    env_file = '.env'
    env_content = f"""
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN={bot_token}
TELEGRAM_CHAT_ID={chat_id}
"""

    try:
        with open(env_file, 'a') as f:
            f.write(env_content)
        print(f"\n✅ Configuration saved to {env_file}")
        print("🔄 Please restart your application to load the new configuration")
        return True
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

if __name__ == "__main__":
    setup_telegram_integration()