import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import json

# Add project paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrations.discord import DiscordIntegration, DiscordConfig


class TestDiscordConfig:
    """Test Discord configuration functionality."""

    def test_discord_config_creation(self):
        """Test DiscordConfig creation."""
        config = DiscordConfig(
            enabled=True,
            name="Test Discord",
            webhook_url="https://discord.com/api/webhooks/123456789/abcdef",
            username="TestBot",
            avatar_url="https://example.com/avatar.png",
            embed_color=0x00ff00
        )

        assert config.enabled == True
        assert config.name == "Test Discord"
        assert config.webhook_url == "https://discord.com/api/webhooks/123456789/abcdef"
        assert config.username == "TestBot"
        assert config.avatar_url == "https://example.com/avatar.png"
        assert config.embed_color == 0x00ff00

    @patch.dict(os.environ, {
        'DISCORD_WEBHOOK_URL': 'https://discord.com/api/webhooks/env_id/env_token',
        'DISCORD_USERNAME': 'EnvBot',
        'DISCORD_AVATAR_URL': 'https://example.com/env_avatar.png',
        'DISCORD_EMBED_COLOR': '16711680'
    })
    def test_discord_config_from_env(self):
        """Test DiscordConfig from environment variables."""
        config = DiscordConfig.from_env()

        assert config.enabled == True
        assert config.webhook_url == 'https://discord.com/api/webhooks/env_id/env_token'
        assert config.username == 'EnvBot'
        assert config.avatar_url == 'https://example.com/env_avatar.png'
        assert config.embed_color == 16711680

    @patch.dict(os.environ, {}, clear=True)
    def test_discord_config_from_env_disabled(self):
        """Test DiscordConfig from environment variables when disabled."""
        config = DiscordConfig.from_env()

        assert config.enabled == False
        assert config.webhook_url == ''


class TestDiscordIntegration:
    """Test Discord integration functionality."""

    def test_discord_initialization_success(self):
        """Test successful Discord initialization."""
        config = DiscordConfig(
            enabled=True,
            name="Test Discord",
            webhook_url="https://discord.com/api/webhooks/123456789/abcdef",
            username="TestBot"
        )

        integration = DiscordIntegration(config)
        result = integration.initialize()

        assert result == True
        assert integration._initialized == True

    def test_discord_initialization_failure_invalid_url(self):
        """Test Discord initialization failure with invalid URL."""
        config = DiscordConfig(
            enabled=True,
            name="Test Discord",
            webhook_url="https://invalid-webhook.com/test",
            username="TestBot"
        )

        integration = DiscordIntegration(config)
        result = integration.initialize()

        assert result == False
        assert integration._initialized == False

    def test_discord_initialization_failure_empty_url(self):
        """Test Discord initialization failure with empty URL."""
        config = DiscordConfig(
            enabled=True,
            name="Test Discord",
            webhook_url="",
            username="TestBot"
        )

        integration = DiscordIntegration(config)
        result = integration.initialize()

        assert result == False

    @patch('integrations.discord.requests.post')
    def test_discord_send_signal_success(self, mock_post):
        """Test successful signal sending via Discord."""
        # Mock successful webhook request
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        config = DiscordConfig(
            enabled=True,
            name="Test Discord",
            webhook_url="https://discord.com/api/webhooks/123456789/abcdef",
            username="TestBot"
        )

        integration = DiscordIntegration(config)
        integration._initialized = True

        # Create test signal data
        signal_data = {
            "signal_type": "BUY",
            "symbol": "EUR/USD",
            "price": 1.0850,
            "timeframe": "1m",
            "indicators": {
                "ema9": 1.0845,
                "ema21": 1.0830,
                "rsi": 65.5
            },
            "risk_management": {
                "stop_loss_buy": 1.0820,
                "take_profit_buy": 1.0900
            }
        }

        result = integration.send_signal(signal_data)

        assert result == True
        mock_post.assert_called_once()

        # Check the payload structure
        call_args = mock_post.call_args
        payload = call_args[1]['json']

        assert payload['username'] == 'TestBot'
        assert 'embeds' in payload
        assert len(payload['embeds']) == 1

        embed = payload['embeds'][0]
        assert 'BUY SIGNAL' in embed['title']
        assert 'EUR/USD' in embed['title']
        assert '1.0850' in embed['description']
        assert embed['color'] == 0x28a745  # Green for BUY

    @patch('integrations.discord.requests.post')
    def test_discord_send_signal_sell_signal(self, mock_post):
        """Test sending SELL signal via Discord."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        config = DiscordConfig(
            enabled=True,
            name="Test Discord",
            webhook_url="https://discord.com/api/webhooks/123456789/abcdef"
        )

        integration = DiscordIntegration(config)
        integration._initialized = True

        signal_data = {
            "signal_type": "SELL",
            "symbol": "GBP/USD",
            "price": 1.2750
        }

        result = integration.send_signal(signal_data)
        assert result == True

        # Check embed color for SELL (red)
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        embed = payload['embeds'][0]
        assert embed['color'] == 0xdc3545  # Red for SELL

    @patch('integrations.discord.requests.post')
    def test_discord_send_signal_failure(self, mock_post):
        """Test signal sending failure via Discord."""
        # Mock failed webhook request
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        config = DiscordConfig(
            enabled=True,
            name="Test Discord",
            webhook_url="https://discord.com/api/webhooks/123456789/abcdef"
        )

        integration = DiscordIntegration(config)
        integration._initialized = True

        signal_data = {"signal_type": "BUY", "price": 100.0}
        result = integration.send_signal(signal_data)

        assert result == False

    @patch('integrations.discord.requests.post')
    def test_discord_send_signal_network_error(self, mock_post):
        """Test signal sending with network error via Discord."""
        mock_post.side_effect = Exception("Network connection failed")

        config = DiscordConfig(
            enabled=True,
            name="Test Discord",
            webhook_url="https://discord.com/api/webhooks/123456789/abcdef"
        )

        integration = DiscordIntegration(config)
        integration._initialized = True

        signal_data = {"signal_type": "BUY", "price": 100.0}
        result = integration.send_signal(signal_data)

        assert result == False

    def test_discord_send_signal_not_initialized(self):
        """Test sending signal when Discord integration is not initialized."""
        config = DiscordConfig(
            enabled=True,
            name="Test Discord",
            webhook_url="https://discord.com/api/webhooks/123456789/abcdef"
        )

        integration = DiscordIntegration(config)
        # Don't initialize

        result = integration.send_signal({"signal_type": "BUY"})
        assert result == False

    @patch('integrations.discord.requests.post')
    def test_discord_test_connection_success(self, mock_post):
        """Test successful connection test via Discord."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        config = DiscordConfig(
            enabled=True,
            name="Test Discord",
            webhook_url="https://discord.com/api/webhooks/123456789/abcdef"
        )

        integration = DiscordIntegration(config)
        integration._initialized = True

        result = integration.test_connection()
        assert result == True
        mock_post.assert_called_once()

    @patch('integrations.discord.requests.post')
    def test_discord_test_connection_failure(self, mock_post):
        """Test connection test failure via Discord."""
        mock_post.side_effect = Exception("Connection failed")

        config = DiscordConfig(
            enabled=True,
            name="Test Discord",
            webhook_url="https://discord.com/api/webhooks/123456789/abcdef"
        )

        integration = DiscordIntegration(config)
        integration._initialized = True

        result = integration.test_connection()
        assert result == False

    def test_discord_test_connection_not_initialized(self):
        """Test connection test when not initialized."""
        config = DiscordConfig(
            enabled=True,
            name="Test Discord",
            webhook_url="https://discord.com/api/webhooks/123456789/abcdef"
        )

        integration = DiscordIntegration(config)

        result = integration.test_connection()
        assert result == False

    def test_discord_embed_creation_buy_signal(self):
        """Test Discord embed creation for BUY signal."""
        config = DiscordConfig(
            enabled=True,
            name="Test Discord",
            webhook_url="https://discord.com/api/webhooks/123456789/abcdef"
        )

        integration = DiscordIntegration(config)

        signal_data = {
            "signal_type": "BUY",
            "symbol": "EUR/USD",
            "price": 1.0850,
            "timeframe": "1m",
            "indicators": {
                "ema9": 1.0845,
                "ema21": 1.0830,
                "rsi": 65.5,
                "bb_upper": 1.0870,
                "bb_middle": 1.0850,
                "bb_lower": 1.0830,
                "atr": 0.0010
            },
            "risk_management": {
                "position_size_percent": 2.0,
                "stop_loss_buy": 1.0820,
                "take_profit_buy": 1.0900,
                "leverage": 5.0
            },
            "timestamp": "2024-01-15T10:30:00Z"
        }

        embed = integration._create_embed(signal_data)

        assert embed['title'] == "🟢 BUY SIGNAL - EUR/USD"
        assert "1.0850" in embed['description']
        assert "1m" in embed['description']
        assert embed['color'] == 0x28a745  # Green
        assert len(embed['fields']) == 2  # Indicators and Risk Management
        assert embed['footer']['text'] == "TradPal Indicator"
        assert embed['timestamp'] == "2024-01-15T10:30:00Z"

    def test_discord_embed_creation_sell_signal(self):
        """Test Discord embed creation for SELL signal."""
        config = DiscordConfig(enabled=True, name="Test Discord", webhook_url="https://discord.com/api/webhooks/123456789/abcdef")
        integration = DiscordIntegration(config)

        signal_data = {
            "signal_type": "SELL",
            "symbol": "GBP/USD",
            "price": 1.2750
        }

        embed = integration._create_embed(signal_data)

        assert embed['title'] == "🔴 SELL SIGNAL - GBP/USD"
        assert embed['color'] == 0xdc3545  # Red

    def test_discord_embed_creation_neutral_signal(self):
        """Test Discord embed creation for neutral/unknown signal."""
        config = DiscordConfig(enabled=True, name="Test Discord", webhook_url="https://discord.com/api/webhooks/123456789/abcdef")
        integration = DiscordIntegration(config)

        signal_data = {
            "signal_type": "HOLD",
            "symbol": "USD/JPY",
            "price": 145.50
        }

        embed = integration._create_embed(signal_data)

        assert embed['title'] == "⚪ HOLD SIGNAL - USD/JPY"
        assert embed['color'] == 0x6c757d  # Gray

    def test_discord_embed_creation_with_avatar(self):
        """Test Discord embed creation with avatar URL."""
        config = DiscordConfig(
            enabled=True,
            name="Test Discord",
            webhook_url="https://discord.com/api/webhooks/123456789/abcdef",
            avatar_url="https://example.com/avatar.png"
        )

        integration = DiscordIntegration(config)
        signal_data = {"signal_type": "BUY", "symbol": "EUR/USD", "price": 1.0850}

        # Test that avatar_url is included in payload when set
        with patch('integrations.discord.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 204
            mock_post.return_value = mock_response

            integration._initialized = True
            result = integration.send_signal(signal_data)

            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert 'avatar_url' in payload
            assert payload['avatar_url'] == "https://example.com/avatar.png"

    def test_discord_embed_creation_minimal_data(self):
        """Test Discord embed creation with minimal signal data."""
        config = DiscordConfig(enabled=True, name="Test Discord", webhook_url="https://discord.com/api/webhooks/123456789/abcdef")
        integration = DiscordIntegration(config)

        signal_data = {
            "signal_type": "BUY",
            "symbol": "EUR/USD",
            "price": 1.0850
        }

        embed = integration._create_embed(signal_data)

        assert embed['title'] == "🟢 BUY SIGNAL - EUR/USD"
        assert "1.0850" in embed['description']
        assert embed['color'] == 0x28a745
        assert embed['fields'] == []  # No indicators or risk management data


if __name__ == "__main__":
    pytest.main([__file__])