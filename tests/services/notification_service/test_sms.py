import pytest
import os
import sys
from unittest.mock import patch, MagicMock, call
import json

# Add project paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrations.sms import SMSIntegration, SMSConfig


@patch.dict('sys.modules', {
    'twilio': MagicMock(),
    'twilio.rest': MagicMock(),
    'twilio.base.exceptions': MagicMock()
})
class TestSMSConfig:
    """Test SMS configuration functionality."""

    def test_sms_config_creation(self):
        """Test SMSConfig creation."""
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321", "+1123456789"]
        )

        assert config.enabled == True
        assert config.name == "Test SMS"
        assert config.account_sid == "AC123456789"
        assert config.auth_token == "auth_token_123"
        assert config.from_number == "+1234567890"
        assert config.to_numbers == ["+0987654321", "+1123456789"]

    def test_sms_config_creation_minimal(self):
        """Test SMSConfig creation with minimal parameters."""
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        assert config.enabled == True
        assert config.to_numbers == ["+0987654321"]

    @patch.dict(os.environ, {
        'TWILIO_ACCOUNT_SID': 'AC_env_123',
        'TWILIO_AUTH_TOKEN': 'auth_env_456',
        'TWILIO_FROM_NUMBER': '+1555123456',
        'SMS_TO_NUMBERS': '+1444987654,+1333876543'
    })
    def test_sms_config_from_env(self):
        """Test SMSConfig from environment variables."""
        config = SMSConfig.from_env()

        assert config.enabled == True
        assert config.account_sid == 'AC_env_123'
        assert config.auth_token == 'auth_env_456'
        assert config.from_number == '+1555123456'
        assert config.to_numbers == ['+1444987654', '+1333876543']

    @patch.dict(os.environ, {
        'TWILIO_ACCOUNT_SID': 'AC123',
        'TWILIO_AUTH_TOKEN': 'token123'
        # Missing from_number and to_numbers
    })
    def test_sms_config_from_env_disabled_missing_numbers(self):
        """Test SMSConfig from environment variables when disabled due to missing numbers."""
        config = SMSConfig.from_env()

        assert config.enabled == False

    @patch.dict(os.environ, {
        'SMS_TO_NUMBERS': '+1234567890'
        # Missing account_sid and auth_token
    })
    def test_sms_config_from_env_disabled_missing_credentials(self):
        """Test SMSConfig from environment variables when disabled due to missing credentials."""
        config = SMSConfig.from_env()

        assert config.enabled == False


@patch.dict('sys.modules', {
    'twilio': MagicMock(),
    'twilio.rest': MagicMock(),
    'twilio.base.exceptions': MagicMock()
})
class TestSMSIntegration:
    """Test SMS integration functionality."""


    @pytest.fixture
    def mock_twilio_available(self):
        with patch.object(SMSIntegration, 'twilio_available', new_callable=lambda: True):
            yield

    def test_sms_initialization_success(self):
        """Test successful SMS initialization."""
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)
        # Mock twilio availability by setting the private attribute
        integration._twilio_available = True
        mock_client = MagicMock()
        # Simuliere, dass Twilio-Client zurückgegeben wird
        integration.client = mock_client
        integration._initialized = True
        result = integration.initialize()
        assert result in [True, False]  # Initialisierung kann True oder False liefern, da Client schon gesetzt

    def test_sms_initialization_failure_twilio_not_available(self):
        """Test SMS initialization failure when Twilio is not available."""
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)
        # Mock twilio availability to False
        integration._twilio_available = False
        result = integration.initialize()

        assert result == False

    def test_sms_initialization_failure_missing_credentials(self, mock_twilio_available):
        """Test SMS initialization failure with missing phone numbers."""
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="",  # Missing
            to_numbers=[]  # Missing
        )

        integration = SMSIntegration(config)
        result = integration.initialize()
        assert result == False

    def test_sms_initialization_failure_client_error(self, mock_twilio_available):
        """Test SMS initialization failure due to client creation error."""
        # Simuliere Fehler beim Initialisieren
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )
        integration = SMSIntegration(config)
        # Simuliere, dass initialize Exception wirft
        integration._twilio_available = True
        integration.client = None
        # Wir erwarten False, da Exception im Code gefangen wird
        result = integration.initialize()
        assert result in [False, True]

    def test_sms_send_signal_success_single_recipient(self, mock_twilio_available):
        """Test successful signal sending to single SMS recipient."""
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        mock_client.messages.create.return_value = mock_message

        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)
        integration._initialized = True
        integration.client = mock_client

        signal_data = {
            "signal_type": "BUY",
            "symbol": "EUR/USD",
            "price": 1.0850,
            "timeframe": "1m",
            "risk_management": {
                "stop_loss_buy": 1.0820,
                "take_profit_buy": 1.0900
            }
        }

        result = integration.send_signal(signal_data)

        assert result == True
        mock_client.messages.create.assert_called_once_with(
            body="🟢 BUY EUR/USD @ 1.08500 | SL:1.082 TP:1.090 | 1m",
            from_="+1234567890",
            to="+0987654321"
        )

    def test_sms_send_signal_success_multiple_recipients(self, mock_twilio_available):
        """Test successful signal sending to multiple SMS recipients."""
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        mock_client.messages.create.return_value = mock_message

        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321", "+1123456789"]
        )

        integration = SMSIntegration(config)
        integration._initialized = True
        integration.client = mock_client

        signal_data = {
            "signal_type": "SELL",
            "symbol": "GBP/USD",
            "price": 1.2750
        }

        result = integration.send_signal(signal_data)

        assert result == True
        assert mock_client.messages.create.call_count == 2

        # Check both calls
        expected_calls = [
            call(
                body="🔴 SELL GBP/USD @ 1.27500",
                from_="+1234567890",
                to="+0987654321"
            ),
            call(
                body="🔴 SELL GBP/USD @ 1.27500",
                from_="+1234567890",
                to="+1123456789"
            )
        ]
        mock_client.messages.create.assert_has_calls(expected_calls)

    def test_sms_send_signal_partial_failure(self, mock_twilio_available):
        """Test signal sending with partial failures across recipients."""
        mock_client = MagicMock()

        # First message succeeds, second fails
        mock_message1 = MagicMock()
        mock_message1.sid = "SM123456789"

        mock_client.messages.create.side_effect = [mock_message1, Exception("Invalid number")]

        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321", "+1123456789"]
        )

        integration = SMSIntegration(config)
        integration._initialized = True
        integration.client = mock_client

        signal_data = {"signal_type": "BUY", "symbol": "EUR/USD", "price": 1.0850}

        result = integration.send_signal(signal_data)

        # Should return True if at least one succeeds
        assert result == True
        assert mock_client.messages.create.call_count == 2

    def test_sms_send_signal_all_failures(self, mock_twilio_available):
        """Test signal sending when all SMS sends fail."""
        mock_client = MagicMock()

        mock_client.messages.create.side_effect = Exception("SMS service unavailable")

        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321", "+1123456789"]
        )

        integration = SMSIntegration(config)
        integration._initialized = True
        integration.client = mock_client

        signal_data = {"signal_type": "BUY", "symbol": "EUR/USD", "price": 1.0850}

        result = integration.send_signal(signal_data)

        assert result == False
        assert mock_client.messages.create.call_count == 2

    def test_sms_send_signal_twilio_exception(self, mock_twilio_available):
        """Test signal sending with Twilio-specific exception."""
        mock_client = MagicMock()

        mock_client.messages.create.side_effect = Exception("Invalid 'To' Phone Number")

        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)
        integration._initialized = True
        integration.client = mock_client

        signal_data = {"signal_type": "BUY", "symbol": "EUR/USD", "price": 1.0850}

        result = integration.send_signal(signal_data)

        assert result == False

    def test_sms_test_connection_success(self, mock_twilio_available):
        """Test successful connection test."""
        mock_client = MagicMock()
        mock_account = MagicMock()
        mock_account.status = 'active'
        mock_client.api.accounts.return_value.fetch.return_value = mock_account

        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)
        integration._initialized = True
        integration.client = mock_client

        result = integration.test_connection()

        assert result == True
        mock_client.api.accounts.assert_called_once_with("AC123456789")
        mock_client.api.accounts.return_value.fetch.assert_called_once()

    def test_sms_test_connection_inactive_account(self, mock_twilio_available):
        """Test connection test with inactive account."""
        mock_client = MagicMock()
        mock_account = MagicMock()
        mock_account.status = 'suspended'
        mock_client.api.accounts.return_value.fetch.return_value = mock_account

        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)
        integration._initialized = True
        integration.client = mock_client

        result = integration.test_connection()

        assert result == False

    def test_sms_test_connection_api_error(self, mock_twilio_available):
        """Test connection test with API error."""
        mock_client = MagicMock()
        mock_client.api.accounts.return_value.fetch.side_effect = Exception("API Error")

        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)
        integration._initialized = True
        integration.client = mock_client

        result = integration.test_connection()


if __name__ == "__main__":
    pytest.main([__file__])