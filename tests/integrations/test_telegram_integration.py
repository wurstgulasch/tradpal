import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add project paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrations.telegram.bot import TelegramIntegration, TelegramConfig


class TestTelegramIntegration:
    """Test Telegram integration functionality."""

    @patch('integrations.telegram.bot.requests.get')
    @patch('integrations.telegram.bot.requests.post')
    def test_telegram_initialization_success(self, mock_post, mock_get):
        """Test successful Telegram initialization."""
        # Mock successful bot info request
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "result": {"username": "TestBot"}}
        mock_get.return_value = mock_response

        config = TelegramConfig(
            enabled=True,
            name="Test Telegram",
            bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
            chat_id="123456789"
        )

        integration = TelegramIntegration(config)
        result = integration.initialize()

        assert result == True
        # Check that initialization was successful
        assert integration._initialized == True
        mock_get.assert_called_once()

    @patch('integrations.telegram.bot.requests.get')
    def test_telegram_initialization_failure(self, mock_get):
        """Test Telegram initialization failure."""
        # Mock failed bot info request
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "description": "Invalid token"}
        mock_get.return_value = mock_response

        config = TelegramConfig(
            enabled=True,
            name="Test Telegram",
            bot_token="invalid_token",
            chat_id="123456789"
        )

        integration = TelegramIntegration(config)
        result = integration.initialize()

        assert result == False

    @patch('integrations.telegram.bot.requests.post')
    def test_telegram_send_signal_success(self, mock_post):
        """Test successful signal sending via Telegram."""
        # Mock successful send message request
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        mock_post.return_value = mock_response

        config = TelegramConfig(
            enabled=True,
            name="Test Telegram",
            bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
            chat_id="123456789"
        )

        integration = TelegramIntegration(config)
        # Manually set initialized state
        integration._initialized = True

        # Create SignalData directly to ensure correct format
        from integrations.base import SignalData
        signal_data = SignalData(
            signal_type='BUY',
            symbol='EUR/USD',
            price=1.0850,
            timeframe='1m'
        )

        result = integration.send_signal(signal_data.to_dict())

        assert result == True
        mock_post.assert_called_once()
        # Check that the message contains the signal information
        call_args = mock_post.call_args
        message_text = call_args[1]['data']['text']
        assert "BUY" in message_text
        assert "EUR/USD" in message_text
        assert "1.0850" in message_text

    @patch('integrations.telegram.bot.requests.post')
    def test_telegram_send_signal_failure(self, mock_post):
        """Test signal sending failure via Telegram."""
        # Mock failed send message request
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Chat not found")
        mock_post.return_value = mock_response

        config = TelegramConfig(
            enabled=True,
            name="Test Telegram",
            bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
            chat_id="invalid_chat_id"
        )

        integration = TelegramIntegration(config)
        integration._initialized = True

        test_signal = {"signal": "BUY", "price": 100.0}
        result = integration.send_signal(test_signal)

        assert result == False

    def test_telegram_send_signal_not_initialized(self):
        """Test sending signal when not initialized."""
        config = TelegramConfig(
            enabled=True,
            name="Test Telegram",
            bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
            chat_id="123456789"
        )

        integration = TelegramIntegration(config)
        # Don't initialize

        result = integration.send_signal({"signal": "BUY"})
        assert result == False

    @patch('integrations.telegram.bot.requests.get')
    def test_telegram_test_connection_success(self, mock_get):
        """Test successful connection test via Telegram."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_get.return_value = mock_response

        config = TelegramConfig(
            enabled=True,
            name="Test Telegram",
            bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
            chat_id="123456789"
        )

        integration = TelegramIntegration(config)
        integration._initialized = True

        result = integration.test_connection()
        assert result == True
        mock_get.assert_called_once()

    @patch('integrations.telegram.bot.requests.post')
    def test_telegram_test_connection_failure(self, mock_post):
        """Test connection test failure via Telegram."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Connection failed")
        mock_post.return_value = mock_response

        config = TelegramConfig(
            enabled=True,
            name="Test Telegram",
            bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
            chat_id="123456789"
        )

        integration = TelegramIntegration(config)
        integration._initialized = True

        result = integration.test_connection()
        assert result == False


