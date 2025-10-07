import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add project paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrations.email_integration.email import EmailIntegration, EmailConfig


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

    @patch('integrations.email_integration.email.smtplib.SMTP')
    def test_email_test_connection_success(self, mock_smtp):
        """Test successful connection test via Email."""
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

        result = integration.test_connection()
        assert result == True
        mock_server.quit.assert_called_once()

    @patch('integrations.email_integration.email.smtplib.SMTP')
    def test_email_test_connection_failure(self, mock_smtp):
        """Test connection test failure via Email."""
        mock_server = MagicMock()
        mock_server.quit.side_effect = Exception("Connection failed")
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

        result = integration.test_connection()
        assert result == False


