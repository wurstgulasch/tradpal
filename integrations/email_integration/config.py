"""
Email Integration Configuration Utilities
"""

import os
import getpass
from .email import EmailConfig, EmailIntegration


def setup_email_integration():
    """Interactive setup for email integration"""
    print("📧 Email Integration Setup")
    print("=" * 40)

    # SMTP Server
    smtp_server = input("SMTP Server (default: smtp.gmail.com): ").strip()
    if not smtp_server:
        smtp_server = "smtp.gmail.com"

    # SMTP Port
    smtp_port_input = input("SMTP Port (default: 587): ").strip()
    smtp_port = int(smtp_port_input) if smtp_port_input else 587

    # Credentials
    username = input("Email Username: ").strip()
    password = getpass.getpass("Email Password/App Password: ")

    # Recipients
    recipients_input = input("Recipient Emails (comma-separated): ").strip()
    recipients = [email.strip() for email in recipients_input.split(',') if email.strip()]

    # TLS
    use_tls_input = input("Use TLS? (Y/n): ").strip().lower()
    use_tls = use_tls_input != 'n'

    # Create config
    config = EmailConfig(
        enabled=True,
        name="Email Notifications",
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        username=username,
        password=password,
        recipients=recipients,
        use_tls=use_tls
    )

    print("\n✅ Email integration configured!")
    print(f"   Server: {smtp_server}:{smtp_port}")
    print(f"   Username: {username}")
    print(f"   Recipients: {len(recipients)}")

    return config


def test_email_integration(config: EmailConfig) -> bool:
    """Test email integration with a test message"""
    print("🧪 Testing email integration...")

    integration = EmailIntegration(config)

    if not integration.initialize():
        print("❌ Failed to initialize email integration")
        return False

    # Create test signal
    test_signal = {
        "timestamp": "2024-01-15T10:30:00Z",
        "symbol": "TEST/USD",
        "timeframe": "1m",
        "signal": "TEST",
        "price": 1.0000,
        "indicators": {"test": 1.0},
        "risk_management": {"test": 1.0},
        "confidence": 1.0,
        "reason": "This is a test message from TradPal"
    }

    if integration.send_signal(test_signal):
        print("✅ Test email sent successfully!")
        integration.shutdown()
        return True
    else:
        print("❌ Failed to send test email")
        integration.shutdown()
        return False


if __name__ == "__main__":
    # Allow direct testing
    config = setup_email_integration()
    test_email_integration(config)