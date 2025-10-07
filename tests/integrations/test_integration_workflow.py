import pytest
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock

# Add project paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from integrations.base import IntegrationManager, BaseIntegration, SignalData
from integrations.telegram.bot import TelegramIntegration, TelegramConfig
from integrations.email_integration.email import EmailIntegration, EmailConfig
from integrations.webhook.webhook import WebhookIntegration
from integrations.discord.discord import DiscordIntegration
from integrations.sms.sms import SMSIntegration


class TestIntegrationWorkflow:
    """Test complete integration workflows."""

    @patch('integrations.telegram.bot.requests.get')
    @patch('integrations.telegram.bot.requests.post')
    def test_complete_telegram_workflow(self, mock_post, mock_get):
        """Test complete Telegram integration workflow."""
        # Mock successful initialization
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"ok": True, "result": {"username": "TestBot"}}
        mock_get.return_value = mock_get_response

        # Mock successful message sending
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {"ok": True}
        mock_post.return_value = mock_post_response

        # Setup integration
        config = TelegramConfig(
            enabled=True,
            name="Test Telegram",
            bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
            chat_id="123456789"
        )

        integration = TelegramIntegration(config)

        # Test workflow
        init_result = integration.initialize()
        assert init_result == True

        signal_result = integration.send_signal({"signal": "BUY", "price": 100.0})
        assert signal_result == True

    @patch('integrations.email_integration.email.smtplib.SMTP')
    def test_complete_email_workflow(self, mock_smtp):
        """Test complete Email integration workflow."""
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server

        # Setup integration
        config = EmailConfig(
            enabled=True,
            name="Test Email",
            smtp_server="smtp.gmail.com",
            username="test@example.com",
            password="password123",
            recipients=["test@example.com"]
        )

        integration = EmailIntegration(config)

        # Test workflow
        init_result = integration.initialize()
        assert init_result == True

        # Create SignalData directly
        from integrations.base import SignalData
        signal_data = SignalData(signal_type='BUY', price=100.0)

        signal_result = integration.send_signal(signal_data.to_dict())
        assert signal_result == True

    def test_integration_manager_workflow(self):
        """Test integration manager workflow with multiple integrations."""
        manager = IntegrationManager()

        # Create mock integrations
        mock_telegram = MagicMock(spec=BaseIntegration)
        mock_telegram.initialize.return_value = True
        mock_telegram.send_signal_safe.return_value = True
        mock_telegram.name = "telegram"

        mock_email = MagicMock(spec=BaseIntegration)
        mock_email.initialize.return_value = True
        mock_email.send_signal_safe.return_value = False  # Simulate failure
        mock_email.name = "email"

        # Register integrations
        manager.register_integration("telegram", mock_telegram)
        manager.register_integration("email", mock_email)

        # Test workflow
        init_results = manager.initialize_all()
        assert init_results["telegram"] == True
        assert init_results["email"] == True

        signal_results = manager.send_signal_to_all({"signal": "BUY"})
        assert signal_results["telegram"] == True
        assert signal_results["email"] == False

        manager.shutdown_all()
        mock_telegram.send_shutdown_message.assert_called_once()
        mock_email.send_shutdown_message.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])


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


class TestEmailIntegration:
    """Test Email integration functionality."""

    @patch('integrations.email_integration.email.smtplib.SMTP')
    def test_email_initialization_success(self, mock_smtp):
        """Test successful Email initialization."""
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server

        config = EmailConfig(
            enabled=True,
            name="Test Email",
            smtp_server="smtp.gmail.com",
            username="test@example.com",
            password="password123",
            recipients=["test@example.com"]
        )

        integration = EmailIntegration(config)
        result = integration.initialize()

        assert result == True
        mock_smtp.assert_called_once_with("smtp.gmail.com")
        mock_server.login.assert_called_once_with("test@example.com", "password123")

    @patch('integrations.email_integration.email.smtplib.SMTP')
    def test_email_initialization_failure(self, mock_smtp):
        """Test Email initialization failure."""
        mock_smtp.side_effect = Exception("SMTP connection failed")

        config = EmailConfig(
            enabled=True,
            name="Test Email",
            smtp_server="invalid.smtp.com",
            username="test@example.com",
            password="password123",
            recipients=["test@example.com"]
        )

        integration = EmailIntegration(config)
        result = integration.initialize()

        assert result == False


class TestEmailIntegration:
    """Test Email integration functionality."""

    @patch('integrations.email_integration.email.smtplib.SMTP')
    def test_email_initialization_success(self, mock_smtp):
        """Test successful Email initialization."""
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server

        config = EmailConfig(
            enabled=True,
            name="Test Email",
            smtp_server="smtp.gmail.com",
            username="test@example.com",
            password="password123",
            recipients=["test@example.com"]
        )

        integration = EmailIntegration(config)
        result = integration.initialize()

        assert result == True
        mock_smtp.assert_called_once_with("smtp.gmail.com", 587)
        mock_server.login.assert_called_once_with("test@example.com", "password123")

    @patch('integrations.email_integration.email.smtplib.SMTP')
    def test_email_initialization_failure(self, mock_smtp):
        """Test Email initialization failure."""
        mock_smtp.side_effect = Exception("SMTP connection failed")

        config = EmailConfig(
            enabled=True,
            name="Test Email",
            smtp_server="invalid.smtp.com",
            username="test@example.com",
            password="password123",
            recipients=["test@example.com"]
        )

        integration = EmailIntegration(config)
        result = integration.initialize()

        assert result == False

    @patch('integrations.email_integration.email.smtplib.SMTP')
    def test_email_send_signal_success(self, mock_smtp):
        """Test successful signal sending via Email."""
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server

        config = EmailConfig(
            enabled=True,
            name="Test Email",
            smtp_server="smtp.gmail.com",
            username="test@example.com",
            password="password123",
            recipients=["test@example.com"]
        )

        integration = EmailIntegration(config)
        integration._initialized = True
        integration.server = mock_server

        # Create SignalData directly
        from integrations.base import SignalData
        signal_data = SignalData(
            signal_type='BUY',
            symbol='EUR/USD',
            price=1.0850
        )

        result = integration.send_signal(signal_data.to_dict())

        assert result == True
        mock_server.send_message.assert_called_once()

    @patch('integrations.email_integration.email.smtplib.SMTP')
    def test_email_send_signal_failure(self, mock_smtp):
        """Test signal sending failure via Email."""
        mock_server = MagicMock()
        mock_server.send_message.side_effect = Exception("Send failed")
        mock_smtp.return_value = mock_server

        config = EmailConfig(
            enabled=True,
            name="Test Email",
            smtp_server="smtp.gmail.com",
            username="test@example.com",
            password="password123",
            recipients=["test@example.com"]
        )

        integration = EmailIntegration(config)
        integration._initialized = True
        integration.server = mock_server

        # Create SignalData directly
        from integrations.base import SignalData
        signal_data = SignalData(signal_type='BUY')

        result = integration.send_signal(signal_data.to_dict())
        assert result == False


class TestIntegrationConfig:
    """Test integration configuration classes."""

    def test_telegram_config_creation(self):
        """Test TelegramConfig creation."""
        config = TelegramConfig(
            enabled=True,
            name="Test Bot",
            bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
            chat_id="987654321"
        )

        assert config.enabled == True
        assert config.name == "Test Bot"
        assert config.bot_token == "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
        assert config.chat_id == "987654321"

    def test_email_config_creation(self):
        """Test EmailConfig creation."""
        config = EmailConfig(
            enabled=True,
            name="Test Email",
            smtp_server="smtp.gmail.com",
            username="test@example.com",
            password="password123",
            recipients=["test@example.com"]
        )

        assert config.enabled == True
        assert config.name == "Test Email"
        assert config.smtp_server == "smtp.gmail.com"
        assert config.username == "test@example.com"
        assert config.password == "password123"


class TestIntegrationWorkflow:
    """Test complete integration workflows."""

    @patch('integrations.telegram.bot.requests.get')
    @patch('integrations.telegram.bot.requests.post')
    def test_complete_telegram_workflow(self, mock_post, mock_get):
        """Test complete Telegram integration workflow."""
        # Mock successful initialization
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"ok": True, "result": {"username": "TestBot"}}
        mock_get.return_value = mock_get_response

        # Mock successful message sending
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {"ok": True}
        mock_post.return_value = mock_post_response

        # Setup integration
        config = TelegramConfig(
            enabled=True,
            name="Test Telegram",
            bot_token="123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
            chat_id="123456789"
        )

        integration = TelegramIntegration(config)

        # Test workflow
        init_result = integration.initialize()
        assert init_result == True

        signal_result = integration.send_signal({"signal": "BUY", "price": 100.0})
        assert signal_result == True

    @patch('integrations.email_integration.email.smtplib.SMTP')
    def test_complete_email_workflow(self, mock_smtp):
        """Test complete Email integration workflow."""
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server

        # Setup integration
        config = EmailConfig(
            enabled=True,
            name="Test Email",
            smtp_server="smtp.gmail.com",
            username="test@example.com",
            password="password123",
            recipients=["test@example.com"]
        )

        integration = EmailIntegration(config)

        # Test workflow
        init_result = integration.initialize()
        assert init_result == True

        # Create SignalData directly
        from integrations.base import SignalData
        signal_data = SignalData(signal_type='BUY', price=100.0)

        signal_result = integration.send_signal(signal_data.to_dict())
        assert signal_result == True

    def test_integration_manager_workflow(self):
        """Test integration manager workflow with multiple integrations."""
        manager = IntegrationManager()

        # Create mock integrations
        mock_telegram = MagicMock(spec=BaseIntegration)
        mock_telegram.initialize.return_value = True
        mock_telegram.send_signal_safe.return_value = True
        mock_telegram.name = "telegram"

        mock_email = MagicMock(spec=BaseIntegration)
        mock_email.initialize.return_value = True
        mock_email.send_signal_safe.return_value = False  # Simulate failure
        mock_email.name = "email"

        # Register integrations
        manager.register_integration("telegram", mock_telegram)
        manager.register_integration("email", mock_email)

        # Test workflow
        init_results = manager.initialize_all()
        assert init_results["telegram"] == True
        assert init_results["email"] == True

        signal_results = manager.send_signal_to_all({"signal": "BUY"})
        assert signal_results["telegram"] == True
        assert signal_results["email"] == False

        manager.shutdown_all()
        mock_telegram.send_shutdown_message.assert_called_once()
        mock_email.send_shutdown_message.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])