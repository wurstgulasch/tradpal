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
                 username: str = "TradPal Bot",
                 avatar_url: str = "",
                 embed_color: int = 0x00ff00):
        super().__init__(enabled=enabled, name=name)
        self.webhook_url = webhook_url
        self.username = username
        self.avatar_url = avatar_url
        self.embed_color = embed_color

    @classmethod
    def from_env(cls) -> 'DiscordConfig':
        """Create config from environment variables"""
        return cls(
            enabled=bool(os.getenv('DISCORD_WEBHOOK_URL')),
            name="Discord Notifications",
            webhook_url=os.getenv('DISCORD_WEBHOOK_URL', ''),
            username=os.getenv('DISCORD_USERNAME', 'TradPal Bot'),
            avatar_url=os.getenv('DISCORD_AVATAR_URL', ''),
            embed_color=int(os.getenv('DISCORD_EMBED_COLOR', '5763719'))  # Green color
        )


class DiscordIntegration(BaseIntegration):
    """Discord integration for sending trading signals via webhooks"""

    def __init__(self, config: DiscordConfig):
        super().__init__(config)
        self.config: DiscordConfig = config

    def initialize(self) -> bool:
        """Initialize Discord webhook connection"""
        try:
            if not self.config.webhook_url:
                self.logger.error("Discord webhook URL is required")
                return False

            # Test webhook URL format
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
                self.logger.info("Discord message sent successfully")
                return True
            else:
                self.logger.error(f"Discord webhook error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to send Discord message: {e}")
            return False

    def test_connection(self) -> bool:
        """Test Discord webhook connection"""
        try:
            # Send a test message
            test_embed = {
                "title": "🧪 Test Message",
                "description": "This is a test message from TradPal Indicator",
                "color": 0x3498db,  # Blue color
                "footer": {
                    "text": "TradPal Indicator Test"
                },
                "timestamp": "2024-01-01T00:00:00.000Z"
            }

            payload = {
                "username": self.config.username,
                "embeds": [test_embed]
            }

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
        """Create Discord embed for signal"""
        signal_type = signal_data.get('signal_type', 'UNKNOWN')
        symbol = signal_data.get('symbol', 'UNKNOWN')
        timeframe = signal_data.get('timeframe', 'UNKNOWN')
        price = signal_data.get('price', 0)

        # Color coding for signals
        if signal_type.upper() == 'BUY':
            color = 0x28a745  # Green
            emoji = '🟢'
        elif signal_type.upper() == 'SELL':
            color = 0xdc3545  # Red
            emoji = '🔴'
        else:
            color = 0x6c757d  # Gray
            emoji = '⚪'

        # Create title
        title = f"{emoji} {signal_type.upper()} SIGNAL - {symbol}"

        # Create description
        description = f"**Price:** {price:.4f}\n**Timeframe:** {timeframe}"

        # Create fields for indicators
        fields = []
        indicators = signal_data.get('indicators', {})

        if indicators:
            indicator_text = ""
            for key, value in list(indicators.items())[:5]:  # Limit to 5 indicators
                if isinstance(value, float):
                    indicator_text += f"**{key.upper()}:** {value:.4f}\n"
                else:
                    indicator_text += f"**{key.upper()}:** {value}\n"

            if indicator_text:
                fields.append({
                    "name": "📊 Indicators",
                    "value": indicator_text[:1024],  # Discord field value limit
                    "inline": True
                })

        # Create fields for risk management
        risk = signal_data.get('risk_management', {})

        if risk:
            risk_text = ""
            for key, value in list(risk.items())[:5]:  # Limit to 5 risk params
                if isinstance(value, float):
                    risk_text += f"**{key.replace('_', ' ').title()}:** {value:.2f}\n"
                else:
                    risk_text += f"**{key.replace('_', ' ').title()}:** {value}\n"

            if risk_text:
                fields.append({
                    "name": "⚠️ Risk Management",
                    "value": risk_text[:1024],  # Discord field value limit
                    "inline": True
                })

        # Create embed
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {
                "text": "TradPal Indicator"
            },
            "timestamp": signal_data.get('timestamp', '2024-01-01T00:00:00.000Z')
        }

        return embed