"""
Discord Integration for TradPal Indicator
Sends trading signals via Discord webhooks
"""

import os
import json
import requests
from typing import Dict, Any, Optional
from integrations.base import BaseIntegration, IntegrationConfig


class DiscordConfig(IntegrationConfig):
    """Configuration for Discord integration"""

    def __init__(self,
                 enabled: bool = True,
                 name: str = "Discord Notifications",
                 webhook_url: str = "",
                 username: str = "TradPal Indicator",
                 avatar_url: str = ""):
        super().__init__(enabled=enabled, name=name)
        self.webhook_url = webhook_url
        self.username = username
        self.avatar_url = avatar_url

    @classmethod
    def from_env(cls) -> 'DiscordConfig':
        """Create config from environment variables"""
        return cls(
            enabled=bool(os.getenv('DISCORD_WEBHOOK_URL')),
            name="Discord Notifications",
            webhook_url=os.getenv('DISCORD_WEBHOOK_URL', ''),
            username=os.getenv('DISCORD_USERNAME', 'TradPal Indicator'),
            avatar_url=os.getenv('DISCORD_AVATAR_URL', '')
        )


class DiscordIntegration(BaseIntegration):
    """Discord integration for sending trading signals via webhooks"""

    def __init__(self, config: DiscordConfig):
        super().__init__(config)
        self.config: DiscordConfig = config

    def initialize(self) -> bool:
        """Initialize Discord integration"""
        try:
            if not self.config.webhook_url:
                self.logger.error("Discord webhook URL is required")
                return False

            if not self.config.webhook_url.startswith('https://discord.com/api/webhooks/'):
                self.logger.error("Invalid Discord webhook URL format")
                return False

            self.logger.info("Discord integration initialized")
            self._initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Discord integration: {e}")
            return False

    def send_signal(self, signal_data: dict) -> bool:
        """Send trading signal via Discord webhook"""
        if not self.config.webhook_url:
            self.logger.warning("Discord webhook URL not configured")
            return False

        try:
            embed = self._create_embed(signal_data)

            payload = {
                "username": self.config.username,
                "embeds": [embed]
            }

            if self.config.avatar_url:
                payload["avatar_url"] = self.config.avatar_url

            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=self.config.timeout
            )

            if response.status_code == 204:
                self.logger.info("Discord webhook sent successfully")
                return True
            else:
                self.logger.error(f"Discord webhook failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"Error sending Discord webhook: {e}")
            return False

    def test_connection(self) -> bool:
        """Test Discord webhook connection"""
        try:
            if not self.config.webhook_url:
                return False

            # Send test message
            test_embed = {
                "title": "TradPal Indicator Test",
                "description": "This is a test message from TradPal Indicator",
                "color": 0x00ff00,
                "footer": {
                    "text": "Test completed successfully"
                }
            }

            payload = {
                "username": self.config.username,
                "embeds": [test_embed]
            }

            if self.config.avatar_url:
                payload["avatar_url"] = self.config.avatar_url

            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10
            )

            return response.status_code == 204

        except Exception as e:
            self.logger.error(f"Discord connection test failed: {e}")
            return False

    def _create_embed(self, signal_data: dict) -> Dict[str, Any]:
        """Create Discord embed from signal data"""
        signal_type = signal_data.get('signal_type', 'UNKNOWN')
        symbol = signal_data.get('symbol', 'UNKNOWN')
        price = signal_data.get('price', 0)

        # Set color based on signal type
        if signal_type.upper() == 'BUY':
            color = 0x00ff00  # Green
            emoji = '🟢'
        elif signal_type.upper() == 'SELL':
            color = 0xff0000  # Red
            emoji = '🔴'
        else:
            color = 0x808080  # Gray
            emoji = '⚪'

        embed = {
            "title": f"{emoji} {signal_type.upper()} Signal",
            "description": f"**Symbol:** {symbol}\n**Price:** {price:.4f}",
            "color": color,
            "fields": []
        }

        # Add risk management info
        risk = signal_data.get('risk_management', {})
        if risk:
            sl = risk.get('stop_loss_buy') or risk.get('stop_loss_sell')
            tp = risk.get('take_profit_buy') or risk.get('take_profit_sell')

            if sl and tp:
                embed["fields"].append({
                    "name": "Risk Management",
                    "value": f"**SL:** {sl:.4f}\n**TP:** {tp:.4f}",
                    "inline": True
                })

        # Add timeframe
        timeframe = signal_data.get('timeframe', '')
        if timeframe:
            embed["fields"].append({
                "name": "Timeframe",
                "value": timeframe,
                "inline": True
            })

        # Add timestamp
        timestamp = signal_data.get('timestamp')
        if timestamp:
            if hasattr(timestamp, 'isoformat'):
                embed["timestamp"] = timestamp.isoformat()
            else:
                embed["timestamp"] = timestamp

        # Add footer
        embed["footer"] = {
            "text": "TradPal Indicator"
        }

        return embed