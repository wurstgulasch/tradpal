import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add project paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrations.base import IntegrationConfig
from integrations.telegram.bot import TelegramConfig
from integrations.email_integration.email import EmailConfig


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

    @patch.dict(os.environ, {
        'TELEGRAM_BOT_TOKEN': 'env_token',
        'TELEGRAM_CHAT_ID': 'env_chat_id'
    })
    def test_telegram_config_from_env(self):
        """Test TelegramConfig from environment variables."""
        config = TelegramConfig.from_env()

        assert config.enabled == True
        assert config.bot_token == 'env_token'
        assert config.chat_id == 'env_chat_id'

    @patch.dict(os.environ, {
        'EMAIL_USERNAME': 'env_user@example.com',
        'EMAIL_PASSWORD': 'env_password',
        'EMAIL_RECIPIENTS': 'user1@example.com,user2@example.com'
    })
    def test_email_config_from_env(self):
        """Test EmailConfig from environment variables."""
        config = EmailConfig.from_env()

        assert config.enabled == True
        assert config.username == 'env_user@example.com'
        assert config.password == 'env_password'
        assert config.recipients == ['user1@example.com', 'user2@example.com']


