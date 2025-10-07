"""
Webhook Integration Configuration Utilities
"""

import os
import json
from .webhook import WebhookConfig, WebhookIntegration


def setup_webhook_integration():
    """Interactive setup for webhook integration"""
    print("🔗 Webhook Integration Setup")
    print("=" * 40)

    # Webhook URLs
    urls_input = input("Webhook URLs (comma-separated): ").strip()
    if not urls_input:
        print("❌ At least one webhook URL is required")
        return None

    webhook_urls = [url.strip() for url in urls_input.split(',') if url.strip()]

    # HTTP Method
    method = input("HTTP Method (default: POST): ").strip().upper()
    if not method:
        method = "POST"

    # Authorization
    auth_token = input("Authorization Token (optional): ").strip()
    auth_type = "Bearer"
    if auth_token:
        auth_type_input = input("Auth Type (Bearer/Basic, default: Bearer): ").strip()
        if auth_type_input.lower() == 'basic':
            auth_type = "Basic"

    # Custom headers
    headers_input = input("Custom Headers (JSON format, optional): ").strip()
    headers = {}
    if headers_input:
        try:
            headers = json.loads(headers_input)
        except json.JSONDecodeError:
            print("⚠️ Invalid JSON format, using default headers")
            headers = {}

    # Create config
    config = WebhookConfig(
        enabled=True,
        name="Webhook Notifications",
        webhook_urls=webhook_urls,
        headers=headers,
        method=method,
        auth_token=auth_token,
        auth_type=auth_type
    )

    print("\n✅ Webhook integration configured!")
    print(f"   URLs: {len(webhook_urls)} configured")
    print(f"   Method: {method}")
    if auth_token:
        print(f"   Auth: {auth_type}")

    return config


def test_webhook_integration(config: WebhookConfig) -> bool:
    """Test webhook integration with a test message"""
    print("🧪 Testing webhook integration...")

    integration = WebhookIntegration(config)

    if not integration.initialize():
        print("❌ Failed to initialize webhook integration")
        return False

    # Create test signal
    test_signal = {
        "timestamp": "2024-01-15T10:30:00Z",
        "symbol": "TEST/USD",
        "timeframe": "1m",
        "signal_type": "TEST",
        "price": 1.0000,
        "indicators": {"test": 1.0},
        "risk_management": {"test": 1.0}
    }

    if integration.send_signal(test_signal):
        print("✅ Test webhook sent successfully!")
        return True
    else:
        print("❌ Failed to send test webhook")
        return False


if __name__ == "__main__":
    # Allow direct testing
    config = setup_webhook_integration()
    if config:
        test_webhook_integration(config)