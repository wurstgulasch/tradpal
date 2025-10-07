"""
Telegram integration for TradPal trading signals.
"""

import os
import time
import json
import requests
from typing import Dict, Any, Optional
from datetime import datetime

from ..base import BaseIntegration, IntegrationConfig, SignalData

class TelegramConfig(IntegrationConfig):
    """Configuration for Telegram integration"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bot_token = kwargs.get('bot_token', os.getenv('TELEGRAM_BOT_TOKEN'))
        self.chat_id = kwargs.get('chat_id', os.getenv('TELEGRAM_CHAT_ID'))
        self.check_interval = kwargs.get('check_interval', 30)
        self.signal_file = kwargs.get('signal_file', 'output/signals.json')

        # Validate required config
        if not self.bot_token:
            raise ValueError("Telegram bot_token is required")
        if not self.chat_id:
            raise ValueError("Telegram chat_id is required")

class TelegramIntegration(BaseIntegration):
    """Telegram bot integration for sending trading signals"""

    def __init__(self, config: TelegramConfig):
        super().__init__(config)
        self.config: TelegramConfig = config
        self.base_url = f"https://api.telegram.org/bot{self.config.bot_token}"
        self.last_signal_count = 0

    def initialize(self) -> bool:
        """Initialize Telegram bot connection"""
        try:
            # Test bot token
            response = requests.get(f"{self.base_url}/getMe", timeout=self.config.timeout)
            response.raise_for_status()
            bot_info = response.json()

            if bot_info.get('ok'):
                self.logger.info(f"Connected to Telegram bot: @{bot_info['result']['username']}")
                self._initialized = True
                return True
            else:
                self.logger.error("Invalid bot token")
                return False

        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram bot: {e}")
            return False

    def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=10)
            return response.status_code == 200 and response.json().get('ok', False)
        except Exception:
            return False

    def send_message(self, text: str, parse_mode: str = 'Markdown') -> bool:
        """Send a message via Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.config.chat_id,
                'text': text,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            response = requests.post(url, data=data, timeout=self.config.timeout)
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            return False

    def format_signal_message(self, signal: SignalData) -> str:
        """Format a trading signal for Telegram"""
        try:
            signal_type = "🟢 BUY" if signal.signal_type == "BUY" else "🔴 SELL"
            price = signal.price
            rsi = signal.indicators.get('rsi', 0)
            ema9 = signal.indicators.get('ema9', 0)
            ema21 = signal.indicators.get('ema21', 0)
            position_size = signal.risk_management.get('position_size_percent', 0)
            leverage = signal.risk_management.get('leverage', 1)

            # Get appropriate stop loss and take profit
            if signal.signal_type == "BUY":
                stop_loss = signal.risk_management.get('stop_loss_buy', 0)
                take_profit = signal.risk_management.get('take_profit_buy', 0)
            else:
                stop_loss = signal.risk_management.get('stop_loss_sell', 0)
                take_profit = signal.risk_management.get('take_profit_sell', 0)

            message = f"""
🚨 *{signal_type} SIGNAL* 🚨

💰 *Symbol:* {signal.symbol}
💰 *Price:* {price:.5f}
📊 *RSI:* {rsi:.2f}
📈 *EMA9:* {ema9:.5f}
📉 *EMA21:* {ema21:.5f}
💼 *Position:* {position_size:.2f}% of portfolio
🛑 *Stop Loss:* {stop_loss:.5f}
🎯 *Take Profit:* {take_profit:.5f}
⚡ *Leverage:* {leverage}x
⏰ *Timeframe:* {signal.timeframe}

🕐 {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}
            """.strip()

            return message

        except Exception as e:
            self.logger.error(f"Error formatting signal: {e}")
            return f"🚨 New {signal.signal_type} Signal detected!"

    def send_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Send a trading signal via Telegram"""
        try:
            signal = SignalData(**signal_data)
            message = self.format_signal_message(signal)
            return self.send_message(message)
        except Exception as e:
            self.logger.error(f"Error sending signal: {e}")
            return False

    def send_startup_message(self) -> bool:
        """Send a startup message"""
        if self.is_test_environment():
            self.logger.info(f"TEST ENVIRONMENT: Skipping startup message for {self.__class__.__name__}")
            return True

        startup_msg = f"""
🤖 *TradPal Telegram Bot Started*

✅ Bot is now monitoring for trading signals
📊 Will send notifications for BUY/SELL signals
⏱️ Check interval: {self.config.check_interval} seconds

🔔 You will receive notifications when:
• New BUY signals are generated
• New SELL signals are generated

📱 Bot Status: Active
        """.strip()

        return self.send_message(startup_msg)

    def send_shutdown_message(self) -> bool:
        """Send a shutdown message"""
        if self.is_test_environment():
            self.logger.info(f"TEST ENVIRONMENT: Skipping shutdown message for {self.__class__.__name__}")
            return True

        shutdown_msg = """
🛑 *TradPal Telegram Bot Stopped*

Bot has been manually stopped.
You will no longer receive signal notifications.
        """.strip()

        return self.send_message(shutdown_msg)

    def check_for_new_signals(self) -> list:
        """Check for new signals in the signals file"""
        try:
            if not os.path.exists(self.config.signal_file):
                return []

            with open(self.config.signal_file, 'r') as f:
                signals = json.load(f)

            # Check if we have new signals
            if len(signals) > self.last_signal_count:
                new_signals = signals[self.last_signal_count:]
                self.last_signal_count = len(signals)
                return new_signals

            return []

        except Exception as e:
            self.logger.error(f"Error checking signals: {e}")
            return []

    def run_monitoring_loop(self):
        """Run the signal monitoring loop"""
        self.logger.info(f"🔄 Monitoring signals every {self.config.check_interval} seconds...")
        print("Press Ctrl+C to stop the bot")

        try:
            while True:
                # Check for new signals
                new_signals = self.check_for_new_signals()

                # Send messages for new signals
                for signal in new_signals:
                    if signal.get('Buy_Signal') == 1 or signal.get('Sell_Signal') == 1:
                        success = self.send_signal(SignalData.from_trading_signal(signal).to_dict())
                        if success:
                            self.logger.info(f"✅ Signal sent: {'BUY' if signal.get('Buy_Signal') == 1 else 'SELL'}")
                        else:
                            self.logger.warning("❌ Failed to send signal")

                # Wait before next check
                time.sleep(self.config.check_interval)

        except KeyboardInterrupt:
            self.logger.info("👋 Monitoring stopped by user")