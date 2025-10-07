"""
SMS Integration Configuration Utilities
"""

import os
from .sms import SMSConfig, SMSIntegration


def setup_sms_integration():
    """Interactive setup for SMS integration"""
    print("📱 SMS Integration Setup (Twilio)")
    print("=" * 40)

    # Check if twilio is installed
    try:
        import twilio
    except ImportError:
        print("❌ Twilio package not installed.")
        print("   Install with: pip install twilio")
        print("   Then run this setup again.")
        return None

    # Account SID
    account_sid = input("Twilio Account SID: ").strip()
    if not account_sid:
        print("❌ Account SID is required")
        return None

    # Auth Token
    auth_token = input("Twilio Auth Token: ").strip()
    if not auth_token:
        print("❌ Auth Token is required")
        return None

    # From number
    from_number = input("Twilio Phone Number (from): ").strip()
    if not from_number:
        print("❌ From number is required")
        return None

    # To numbers
    to_numbers_input = input("Recipient Phone Numbers (comma-separated): ").strip()
    if not to_numbers_input:
        print("❌ At least one recipient number is required")
        return None

    to_numbers = [num.strip() for num in to_numbers_input.split(',') if num.strip()]

    # Create config
    config = SMSConfig(
        enabled=True,
        name="SMS Notifications",
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
        to_numbers=to_numbers
    )

    print("\n✅ SMS integration configured!")
    print(f"   From: {from_number}")
    print(f"   Recipients: {len(to_numbers)}")

    return config


def test_sms_integration(config: SMSConfig) -> bool:
    """Test SMS integration with a test message"""
    print("🧪 Testing SMS integration...")

    integration = SMSIntegration(config)

    if not integration.initialize():
        print("❌ Failed to initialize SMS integration")
        return False

    # Create test signal
    test_signal = {
        "timestamp": "2024-01-15T10:30:00Z",
        "symbol": "TEST/USD",
        "timeframe": "1m",
        "signal_type": "TEST",
        "price": 1.0000,
        "risk_management": {"stop_loss_buy": 0.99, "take_profit_buy": 1.05}
    }

    if integration.send_signal(test_signal):
        print("✅ Test SMS sent successfully!")
        print("⚠️  Note: This will incur SMS charges from Twilio")
        return True
    else:
        print("❌ Failed to send test SMS")
        return False


if __name__ == "__main__":
    # Allow direct testing
    config = setup_sms_integration()
    if config:
        test_sms_integration(config)