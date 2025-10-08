#!/usr/bin/env python3
"""
Demo script showing the new configuration startup messages for integrations.
This script simulates what happens when integrations start up.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def demo_startup_messages():
    """Demonstrate the new configuration startup messages"""

    print("🚀 TradPal Integration Startup Messages Demo")
    print("=" * 50)

    from config.settings import (
        SYMBOL, EXCHANGE, TIMEFRAME, DEFAULT_INDICATOR_CONFIG,
        RISK_PER_TRADE, SL_MULTIPLIER, TP_MULTIPLIER, LEVERAGE_BASE,
        MTA_ENABLED, ADX_ENABLED, ADX_THRESHOLD, FIBONACCI_ENABLED
    )

    # Format indicator configuration
    indicators = []
    if DEFAULT_INDICATOR_CONFIG.get('ema', {}).get('enabled'):
        periods = DEFAULT_INDICATOR_CONFIG['ema'].get('periods', [9, 21])
        indicators.append(f"EMA{periods}")
    if DEFAULT_INDICATOR_CONFIG.get('rsi', {}).get('enabled'):
        period = DEFAULT_INDICATOR_CONFIG['rsi'].get('period', 14)
        indicators.append(f"RSI({period})")
    if DEFAULT_INDICATOR_CONFIG.get('bb', {}).get('enabled'):
        period = DEFAULT_INDICATOR_CONFIG['bb'].get('period', 20)
        indicators.append(f"BB({period})")
    if DEFAULT_INDICATOR_CONFIG.get('atr', {}).get('enabled'):
        period = DEFAULT_INDICATOR_CONFIG['atr'].get('period', 14)
        indicators.append(f"ATR({period})")
    if DEFAULT_INDICATOR_CONFIG.get('adx', {}).get('enabled'):
        indicators.append("ADX")

    indicators_str = ', '.join(indicators) if indicators else 'None'

    print("📊 Current Configuration:")
    print(f"   • Symbol: {SYMBOL}")
    print(f"   • Exchange: {EXCHANGE}")
    print(f"   • Timeframe: {TIMEFRAME}")
    print(f"   • Indicators: {indicators_str}")
    print(f"   • Risk per Trade: {RISK_PER_TRADE*100:.1f}%")
    print(f"   • Stop Loss Multiplier: {SL_MULTIPLIER}x ATR")
    print(f"   • Take Profit Multiplier: {TP_MULTIPLIER}x ATR")
    print(f"   • Base Leverage: {LEVERAGE_BASE}x")
    print(f"   • MTA Enabled: {MTA_ENABLED}")
    print(f"   • ADX Enabled: {ADX_ENABLED}")
    print(f"   • Fibonacci TP: {FIBONACCI_ENABLED}")
    print()

    print("📱 Integration Startup Messages:")
    print()

    # Telegram example
    print("🤖 TELEGRAM:")
    telegram_msg = f"""🤖 *TradPal Telegram Bot Started*

✅ Bot is now monitoring for trading signals

📊 *Current Configuration:*
• *Symbol:* {SYMBOL}
• *Exchange:* {EXCHANGE}
• *Timeframe:* {TIMEFRAME}
• *Indicators:* {indicators_str}
• *Risk per Trade:* {RISK_PER_TRADE*100:.1f}%
• *Stop Loss Multiplier:* {SL_MULTIPLIER}x ATR
• *Take Profit Multiplier:* {TP_MULTIPLIER}x ATR
• *Base Leverage:* {LEVERAGE_BASE}x
• *MTA Enabled:* {'Yes' if MTA_ENABLED else 'No'}
• *ADX Enabled:* {'Yes' if ADX_ENABLED else 'No'}
• *Fibonacci TP:* {'Yes' if FIBONACCI_ENABLED else 'No'}

🔔 You will receive notifications when:
• New BUY signals are generated
• New SELL signals are generated

⏱️ Check interval: 30 seconds
📱 Bot Status: Active"""
    print(telegram_msg)
    print()

    # Discord example (shortened)
    print("🎮 DISCORD:")
    discord_msg = f"""🤖 TradPal Discord Bot Started
✅ Bot is now monitoring for trading signals

📊 Current Configuration:
• Symbol: {SYMBOL}
• Exchange: {EXCHANGE}
• Timeframe: {TIMEFRAME}
• Indicators: {indicators_str}

⚙️ Risk Settings:
• Risk per Trade: {RISK_PER_TRADE*100:.1f}%
• SL Multiplier: {SL_MULTIPLIER}x ATR
• TP Multiplier: {TP_MULTIPLIER}x ATR
• Base Leverage: {LEVERAGE_BASE}x

🔧 Advanced Features:
• MTA: {'Enabled' if MTA_ENABLED else 'Disabled'}
• ADX: {'Enabled' if ADX_ENABLED else 'Disabled'}
• Fibonacci TP: {'Enabled' if FIBONACCI_ENABLED else 'Disabled'}

Bot Status: Active"""
    print(discord_msg)
    print()

    # SMS example (very short)
    print("📱 SMS:")
    sms_msg = f"🤖 TradPal SMS Bot Started | {SYMBOL} {TIMEFRAME} | Indicators: {indicators_str.replace(', ', '+')} | Status: Active"
    print(sms_msg)
    print()

    print("✅ All integrations now send configuration details on startup!")
    print("💡 This helps you verify the bot is running with the correct settings.")

if __name__ == "__main__":
    demo_startup_messages()