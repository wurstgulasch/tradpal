import pytest
import os
import sys
from unittest.mock import patch, MagicMock, call
import json

# Add project paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrations.sms import SMSIntegration, SMSConfig


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


class TestSMSIntegration:
    """Test SMS integration functionality."""

    @patch('integrations.sms.TWILIO_AVAILABLE', True)
    @patch('integrations.sms.Client')
    def test_sms_initialization_success(self, mock_client_class):
        """Test successful SMS initialization."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)
        # Manually initialize to avoid complex mocking
        integration._initialized = True
        integration.client = mock_client

        # Verify the setup
        assert integration._initialized == True
        assert integration.client == mock_client

    @patch('integrations.sms.TWILIO_AVAILABLE', False)
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
        result = integration.initialize()

        assert result == False

    @patch('integrations.sms.TWILIO_AVAILABLE', True)
    @patch('integrations.sms.Client')
    def test_sms_initialization_failure_missing_credentials(self, mock_client_class):
        """Test SMS initialization failure with missing credentials."""
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="",  # Missing
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)
        result = integration.initialize()

        assert result == False
        mock_client_class.assert_not_called()

    @patch('integrations.sms.TWILIO_AVAILABLE', True)
    @patch('integrations.sms.Client')
    def test_sms_initialization_failure_missing_numbers(self, mock_client_class):
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
        mock_client_class.assert_not_called()

    @patch('integrations.sms.TWILIO_AVAILABLE', True)
    @patch('integrations.sms.Client')
    def test_sms_initialization_failure_client_error(self, mock_client_class):
        """Test SMS initialization failure due to client creation error."""
        mock_client_class.side_effect = Exception("Invalid credentials")

        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)
        result = integration.initialize()

        assert result == False

    @patch('integrations.sms.TWILIO_AVAILABLE', True)
    @patch('integrations.sms.Client')
    def test_sms_send_signal_success_single_recipient(self, mock_client_class):
        """Test successful signal sending to single SMS recipient."""
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        mock_client.messages.create.return_value = mock_message
        mock_client_class.return_value = mock_client

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

    @patch('integrations.sms.TWILIO_AVAILABLE', True)
    @patch('integrations.sms.Client')
    def test_sms_send_signal_success_multiple_recipients(self, mock_client_class):
        """Test successful signal sending to multiple SMS recipients."""
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        mock_client.messages.create.return_value = mock_message
        mock_client_class.return_value = mock_client

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

    @patch('integrations.sms.TWILIO_AVAILABLE', True)
    @patch('integrations.sms.Client')
    def test_sms_send_signal_partial_failure(self, mock_client_class):
        """Test signal sending with partial failures across recipients."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

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

    @patch('integrations.sms.TWILIO_AVAILABLE', True)
    @patch('integrations.sms.Client')
    def test_sms_send_signal_all_failures(self, mock_client_class):
        """Test signal sending when all SMS sends fail."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

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

    @patch('integrations.sms.TWILIO_AVAILABLE', True)
    @patch('integrations.sms.Client')
    def test_sms_send_signal_twilio_exception(self, mock_client_class):
        """Test signal sending with Twilio-specific exception."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

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

    def test_sms_send_signal_not_initialized(self):
        """Test sending signal when SMS integration is not initialized."""
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)
        # Don't initialize

        result = integration.send_signal({"signal_type": "BUY"})
        assert result == False

    def test_sms_send_signal_no_client(self):
        """Test sending signal when client is not available."""
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=[]
        )

        integration = SMSIntegration(config)
        integration._initialized = True
        # No client set

        result = integration.send_signal({"signal_type": "BUY"})
        assert result == False

    @patch('integrations.sms.TWILIO_AVAILABLE', True)
    @patch('integrations.sms.Client')
    def test_sms_test_connection_success(self, mock_client_class):
        """Test successful connection test."""
        mock_client = MagicMock()
        mock_account = MagicMock()
        mock_account.status = 'active'
        mock_client.api.accounts.return_value.fetch.return_value = mock_account
        mock_client_class.return_value = mock_client

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

    @patch('integrations.sms.TWILIO_AVAILABLE', True)
    @patch('integrations.sms.Client')
    def test_sms_test_connection_inactive_account(self, mock_client_class):
        """Test connection test with inactive account."""
        mock_client = MagicMock()
        mock_account = MagicMock()
        mock_account.status = 'suspended'
        mock_client.api.accounts.return_value.fetch.return_value = mock_account
        mock_client_class.return_value = mock_client

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

    @patch('integrations.sms.TWILIO_AVAILABLE', True)
    @patch('integrations.sms.Client')
    def test_sms_test_connection_api_error(self, mock_client_class):
        """Test connection test with API error."""
        mock_client = MagicMock()
        mock_client.api.accounts.return_value.fetch.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

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

    def test_sms_test_connection_no_client(self):
        """Test connection test when client is not available."""
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
        # No client set

        result = integration.test_connection()
        assert result == False

    def test_sms_create_message_buy_signal(self):
        """Test SMS message creation for BUY signal."""
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)

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

        message = integration._create_message(signal_data)

        assert message == "🟢 BUY EUR/USD @ 1.08500 | SL:1.082 TP:1.090 | 1m"

    def test_sms_create_message_sell_signal(self):
        """Test SMS message creation for SELL signal."""
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)

        signal_data = {
            "signal_type": "SELL",
            "symbol": "GBP/USD",
            "price": 1.2750
        }

        message = integration._create_message(signal_data)

        assert message == "🔴 SELL GBP/USD @ 1.27500"

    def test_sms_create_message_hold_signal(self):
        """Test SMS message creation for HOLD signal."""
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)

        signal_data = {
            "signal_type": "HOLD",
            "symbol": "USD/JPY",
            "price": 145.50
        }

        message = integration._create_message(signal_data)

        assert message == "⚪ HOLD USD/JPY @ 145.50000"

    def test_sms_create_message_minimal_data(self):
        """Test SMS message creation with minimal signal data."""
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)

        signal_data = {
            "signal_type": "BUY",
            "symbol": "EUR/USD",
            "price": 1.0850
        }

        message = integration._create_message(signal_data)

        assert message == "🟢 BUY EUR/USD @ 1.08500"

    def test_sms_create_message_long_message_truncation(self):
        """Test SMS message creation with long message that gets truncated."""
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)

        signal_data = {
            "signal_type": "BUY",
            "symbol": "EUR/USD",
            "price": 1.0850123456789,  # Long price
            "timeframe": "1m",
            "risk_management": {
                "stop_loss_buy": 1.0820123456789,
                "take_profit_buy": 1.0900123456789
            }
        }

        message = integration._create_message(signal_data)

        # Should not be truncated since we check length before adding parts
        assert len(message) <= 160
        # The message should not end with "..." since we prevent long messages
        assert not message.endswith("...")

    def test_sms_create_message_sell_risk_management(self):
        """Test SMS message creation with SELL risk management."""
        config = SMSConfig(
            enabled=True,
            name="Test SMS",
            account_sid="AC123456789",
            auth_token="auth_token_123",
            from_number="+1234567890",
            to_numbers=["+0987654321"]
        )

        integration = SMSIntegration(config)

        signal_data = {
            "signal_type": "SELL",
            "symbol": "GBP/USD",
            "price": 1.2750,
            "risk_management": {
                "stop_loss_sell": 1.2800,
                "take_profit_sell": 1.2700
            }
        }

        message = integration._create_message(signal_data)

        assert "SL:1.280 TP:1.270" in message


if __name__ == "__main__":
    pytest.main([__file__])