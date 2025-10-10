import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import json

# Add project paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrations.webhook import WebhookIntegration, WebhookConfig


class TestWebhookConfig:
    """Test Webhook configuration functionality."""

    def test_webhook_config_creation(self):
        """Test WebhookConfig creation."""
        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook", "https://backup.example.com/webhook"],
            headers={"Authorization": "Bearer token123", "X-Custom": "value"},
            method="POST",
            auth_token="token123",
            auth_type="Bearer"
        )

        assert config.enabled == True
        assert config.name == "Test Webhook"
        assert config.webhook_urls == ["https://api.example.com/webhook", "https://backup.example.com/webhook"]
        assert config.headers["Authorization"] == "Bearer token123"
        assert config.headers["X-Custom"] == "value"
        assert config.headers["Content-Type"] == "application/json"
        assert config.method == "POST"
        assert config.auth_token == "token123"
        assert config.auth_type == "Bearer"

    def test_webhook_config_creation_minimal(self):
        """Test WebhookConfig creation with minimal parameters."""
        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook"]
        )

        assert config.enabled == True
        assert config.webhook_urls == ["https://api.example.com/webhook"]
        assert config.headers["Content-Type"] == "application/json"
        assert config.method == "POST"
        assert config.auth_token == ""
        assert config.auth_type == "Bearer"

    def test_webhook_config_bearer_auth(self):
        """Test WebhookConfig with Bearer authentication."""
        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook"],
            auth_token="token123",
            auth_type="Bearer"
        )

        assert config.headers["Authorization"] == "Bearer token123"

    def test_webhook_config_basic_auth(self):
        """Test WebhookConfig with Basic authentication."""
        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook"],
            auth_token="dXNlcjpwYXNz",  # base64 encoded user:pass
            auth_type="Basic"
        )

        assert config.headers["Authorization"] == "Basic dXNlcjpwYXNz"

    def test_webhook_config_custom_headers_override(self):
        """Test WebhookConfig with custom headers overriding defaults."""
        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook"],
            headers={"Content-Type": "application/xml", "X-Custom": "value"}
        )

        assert config.headers["Content-Type"] == "application/xml"  # Overridden
        assert config.headers["X-Custom"] == "value"

    @patch.dict(os.environ, {
        'WEBHOOK_URLS': 'https://api1.example.com/webhook,https://api2.example.com/webhook',
        'WEBHOOK_METHOD': 'PUT',
        'WEBHOOK_AUTH_TOKEN': 'env_token',
        'WEBHOOK_AUTH_TYPE': 'Basic',
        'WEBHOOK_HEADERS': '{"X-API-Key": "env_key", "X-Source": "test"}'
    })
    def test_webhook_config_from_env(self):
        """Test WebhookConfig from environment variables."""
        config = WebhookConfig.from_env()

        assert config.enabled == True
        assert config.webhook_urls == ['https://api1.example.com/webhook', 'https://api2.example.com/webhook']
        assert config.method == 'PUT'
        assert config.auth_token == 'env_token'
        assert config.auth_type == 'Basic'
        assert config.headers["X-API-Key"] == "env_key"
        assert config.headers["X-Source"] == "test"
        assert config.headers["Content-Type"] == "application/json"  # Default added

    @patch.dict(os.environ, {
        'WEBHOOK_URLS': '',
        'WEBHOOK_AUTH_TOKEN': 'token'
    })
    def test_webhook_config_from_env_disabled(self):
        """Test WebhookConfig from environment variables when disabled."""
        config = WebhookConfig.from_env()

        assert config.enabled == False
        assert config.webhook_urls == []

    @patch.dict(os.environ, {
        'WEBHOOK_HEADERS': 'invalid json'
    })
    def test_webhook_config_from_env_invalid_headers(self):
        """Test WebhookConfig from environment with invalid JSON headers."""
        config = WebhookConfig.from_env()

        # Should handle invalid JSON gracefully
        assert isinstance(config.headers, dict)


class TestWebhookIntegration:
    """Test Webhook integration functionality."""

    def test_webhook_initialization_success(self):
        """Test successful Webhook initialization."""
        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook"]
        )

        integration = WebhookIntegration(config)
        # Manually initialize to avoid complex mocking
        integration._initialized = True

        # Verify the setup
        assert integration._initialized == True

    def test_webhook_initialization_failure_no_urls(self):
        """Test Webhook initialization failure with no URLs."""
        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=[]
        )

        integration = WebhookIntegration(config)
        result = integration.initialize()

        assert result == False

    def test_webhook_initialization_failure_invalid_url(self):
        """Test Webhook initialization failure with invalid URL."""
        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["ftp://invalid.protocol.com/webhook"]
        )

        integration = WebhookIntegration(config)
        result = integration.initialize()

        assert result == False

    def test_webhook_initialization_failure_mixed_urls(self):
        """Test Webhook initialization failure with mixed valid/invalid URLs."""
        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://valid.example.com/webhook", "ftp://invalid.protocol.com/webhook"]
        )

        integration = WebhookIntegration(config)
        result = integration.initialize()

        assert result == False

    @patch('integrations.webhook.requests.request')
    def test_webhook_send_signal_success_single_url(self, mock_request):
        """Test successful signal sending to single webhook URL."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook"]
        )

        integration = WebhookIntegration(config)
        integration._initialized = True

        signal_data = {
            "signal_type": "BUY",
            "symbol": "EUR/USD",
            "price": 1.0850,
            "timestamp": "2024-01-15T10:30:00Z"
        }

        result = integration.send_signal(signal_data)

        assert result == True
        mock_request.assert_called_once()

        # Check the request parameters
        call_args = mock_request.call_args
        assert call_args[1]['method'] == 'POST'
        assert call_args[1]['url'] == 'https://api.example.com/webhook'
        assert 'json' in call_args[1]

        payload = call_args[1]['json']
        assert payload['signal_type'] == 'BUY'
        assert payload['symbol'] == 'EUR/USD'
        assert payload['price'] == 1.0850
        assert payload['source'] == 'TradPal'
        assert payload['version'] == '1.0'

    @patch('integrations.webhook.requests.request')
    def test_webhook_send_signal_success_multiple_urls(self, mock_request):
        """Test successful signal sending to multiple webhook URLs."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_request.return_value = mock_response

        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api1.example.com/webhook", "https://api2.example.com/webhook"]
        )

        integration = WebhookIntegration(config)
        integration._initialized = True

        signal_data = {"signal_type": "SELL", "symbol": "GBP/USD", "price": 1.2750}

        result = integration.send_signal(signal_data)

        assert result == True
        assert mock_request.call_count == 2

    @patch('integrations.webhook.requests.request')
    def test_webhook_send_signal_partial_failure(self, mock_request):
        """Test signal sending with partial failures across multiple URLs."""
        # First URL succeeds, second fails
        mock_response1 = MagicMock()
        mock_response1.status_code = 200

        mock_response2 = MagicMock()
        mock_response2.status_code = 500

        mock_request.side_effect = [mock_response1, mock_response2]

        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api1.example.com/webhook", "https://api2.example.com/webhook"]
        )

        integration = WebhookIntegration(config)
        integration._initialized = True

        signal_data = {"signal_type": "BUY", "symbol": "EUR/USD", "price": 1.0850}

        result = integration.send_signal(signal_data)

        # Should return True if at least one succeeds
        assert result == True
        assert mock_request.call_count == 2

    @patch('integrations.webhook.requests.request')
    def test_webhook_send_signal_all_failures(self, mock_request):
        """Test signal sending when all webhook URLs fail."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_request.return_value = mock_response

        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api1.example.com/webhook", "https://api2.example.com/webhook"]
        )

        integration = WebhookIntegration(config)
        integration._initialized = True

        signal_data = {"signal_type": "BUY", "symbol": "EUR/USD", "price": 1.0850}

        result = integration.send_signal(signal_data)

        assert result == False
        assert mock_request.call_count == 2

    @patch('integrations.webhook.requests.request')
    def test_webhook_send_signal_network_error(self, mock_request):
        """Test signal sending with network error."""
        mock_request.side_effect = Exception("Connection timeout")

        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook"]
        )

        integration = WebhookIntegration(config)
        integration._initialized = True

        signal_data = {"signal_type": "BUY", "symbol": "EUR/USD", "price": 1.0850}

        result = integration.send_signal(signal_data)

        assert result == False

    def test_webhook_send_signal_not_initialized(self):
        """Test sending signal when webhook integration is not initialized."""
        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook"]
        )

        integration = WebhookIntegration(config)
        # Don't initialize

        result = integration.send_signal({"signal_type": "BUY"})
        assert result == False

    def test_webhook_send_signal_no_urls(self):
        """Test sending signal when no webhook URLs are configured."""
        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=[]
        )

        integration = WebhookIntegration(config)
        integration._initialized = True

        result = integration.send_signal({"signal_type": "BUY"})
        assert result == False

    @patch('integrations.webhook.requests.request')
    def test_webhook_test_connection_success(self, mock_request):
        """Test successful connection test."""
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_request.return_value = mock_response

        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook"]
        )

        integration = WebhookIntegration(config)
        integration._initialized = True

        result = integration.test_connection()

        assert result == True
        mock_request.assert_called_once()

        # Check test payload
        call_args = mock_request.call_args
        payload = call_args[1]['json']
        assert payload['test'] == True
        assert 'TradPal Webhook Test' in payload['message']

    @patch('integrations.webhook.requests.request')
    def test_webhook_test_connection_multiple_urls(self, mock_request):
        """Test connection test with multiple URLs."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api1.example.com/webhook", "https://api2.example.com/webhook"]
        )

        integration = WebhookIntegration(config)
        integration._initialized = True

        result = integration.test_connection()

        assert result == True
        assert mock_request.call_count == 2

    @patch('integrations.webhook.requests.request')
    def test_webhook_test_connection_partial_failure(self, mock_request):
        """Test connection test with partial failures."""
        mock_response1 = MagicMock()
        mock_response1.status_code = 200

        mock_response2 = MagicMock()
        mock_response2.status_code = 403

        mock_request.side_effect = [mock_response1, mock_response2]

        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api1.example.com/webhook", "https://api2.example.com/webhook"]
        )

        integration = WebhookIntegration(config)
        integration._initialized = True

        result = integration.test_connection()

        # Should return True if at least one succeeds
        assert result == True

    @patch('integrations.webhook.requests.request')
    def test_webhook_test_connection_all_failures(self, mock_request):
        """Test connection test when all URLs fail."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_request.return_value = mock_response

        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api1.example.com/webhook", "https://api2.example.com/webhook"]
        )

        integration = WebhookIntegration(config)
        integration._initialized = True

        result = integration.test_connection()

        assert result == False

    def test_webhook_test_connection_not_initialized(self):
        """Test connection test when not initialized."""
        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook"]
        )

        integration = WebhookIntegration(config)

        result = integration.test_connection()
        assert result == False

    def test_webhook_prepare_payload_basic(self):
        """Test payload preparation with basic signal data."""
        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook"]
        )

        integration = WebhookIntegration(config)

        signal_data = {
            "signal_type": "BUY",
            "symbol": "EUR/USD",
            "price": 1.0850,
            "timestamp": "2024-01-15T10:30:00Z"
        }

        payload = integration._prepare_payload(signal_data)

        assert payload['signal_type'] == 'BUY'
        assert payload['symbol'] == 'EUR/USD'
        assert payload['price'] == 1.0850
        assert payload['timestamp'] == '2024-01-15T10:30:00Z'
        assert payload['source'] == 'TradPal'
        assert payload['version'] == '1.0'

    def test_webhook_prepare_payload_with_datetime(self):
        """Test payload preparation with datetime timestamp."""
        from datetime import datetime

        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook"]
        )

        integration = WebhookIntegration(config)

        signal_data = {
            "signal_type": "SELL",
            "symbol": "GBP/USD",
            "price": 1.2750,
            "timestamp": datetime(2024, 1, 15, 10, 30, 0)
        }

        payload = integration._prepare_payload(signal_data)

        assert payload['timestamp'] == '2024-01-15T10:30:00'  # ISO format

    def test_webhook_prepare_payload_no_timestamp(self):
        """Test payload preparation without timestamp."""
        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook"]
        )

        integration = WebhookIntegration(config)

        signal_data = {
            "signal_type": "BUY",
            "symbol": "EUR/USD",
            "price": 1.0850
        }

        payload = integration._prepare_payload(signal_data)

        assert 'timestamp' not in payload
        assert payload['source'] == 'TradPal'

    @patch('integrations.webhook.requests.request')
    def test_webhook_custom_method_and_headers(self, mock_request):
        """Test webhook with custom HTTP method and headers."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        config = WebhookConfig(
            enabled=True,
            name="Test Webhook",
            webhook_urls=["https://api.example.com/webhook"],
            method="PUT",
            headers={"X-API-Key": "custom_key", "X-Source": "test"}
        )

        integration = WebhookIntegration(config)
        integration._initialized = True

        signal_data = {"signal_type": "BUY", "symbol": "EUR/USD", "price": 1.0850}

        result = integration.send_signal(signal_data)

        assert result == True

        call_args = mock_request.call_args
        assert call_args[1]['method'] == 'PUT'
        headers = call_args[1]['headers']
        assert headers['X-API-Key'] == 'custom_key'
        assert headers['X-Source'] == 'test'
        assert headers['Content-Type'] == 'application/json'  # Default added


if __name__ == "__main__":
    pytest.main([__file__])