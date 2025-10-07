"""
SMS Integration for TradPal Indicator
Sends trading signals via SMS using Twilio
"""

import os
from typing import Dict, Any, List
from integrations.base import BaseIntegration, IntegrationConfig

try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    Client = None
    TwilioException = Exception


class SMSConfig(IntegrationConfig):
    """Configuration for SMS integration"""

    def __init__(self,
                 enabled: bool = True,
                 name: str = "SMS Notifications",
                 account_sid: str = "",
                 auth_token: str = "",
                 from_number: str = "",
                 to_numbers: List[str] = None):
        super().__init__(enabled=enabled, name=name)
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_numbers = to_numbers or []

    @classmethod
    def from_env(cls) -> 'SMSConfig':
        """Create config from environment variables"""
        to_numbers = []
        if os.getenv('SMS_TO_NUMBERS'):
            to_numbers = [num.strip() for num in os.getenv('SMS_TO_NUMBERS', '').split(',') if num.strip()]

        return cls(
            enabled=bool(os.getenv('TWILIO_ACCOUNT_SID') and os.getenv('TWILIO_AUTH_TOKEN') and to_numbers),
            name="SMS Notifications",
            account_sid=os.getenv('TWILIO_ACCOUNT_SID', ''),
            auth_token=os.getenv('TWILIO_AUTH_TOKEN', ''),
            from_number=os.getenv('TWILIO_FROM_NUMBER', ''),
            to_numbers=to_numbers
        )


class SMSIntegration(BaseIntegration):
    """SMS integration for sending trading signals via Twilio"""

    def __init__(self, config: SMSConfig):
        super().__init__(config)
        self.config: SMSConfig = config
        self.client = None

    def initialize(self) -> bool:
        """Initialize SMS integration"""
        try:
            if not TWILIO_AVAILABLE:
                self.logger.error("Twilio package not installed. Install with: pip install twilio")
                return False

            if not all([self.config.account_sid, self.config.auth_token, self.config.from_number, self.config.to_numbers]):
                self.logger.error("SMS configuration incomplete: account_sid, auth_token, from_number, and to_numbers required")
                return False

            # Initialize Twilio client
            self.client = Client(self.config.account_sid, self.config.auth_token)

            self.logger.info(f"SMS integration initialized for {len(self.config.to_numbers)} recipients")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize SMS integration: {e}")
            return False

    def send_signal(self, signal_data: dict) -> bool:
        """Send trading signal via SMS"""
        if not self.client or not self.config.to_numbers:
            self.logger.warning("SMS client not initialized or no recipients configured")
            return False

        message_body = self._create_message(signal_data)
        success_count = 0

        for to_number in self.config.to_numbers:
            try:
                message = self.client.messages.create(
                    body=message_body,
                    from_=self.config.from_number,
                    to=to_number
                )

                self.logger.info(f"SMS sent successfully to {to_number} (SID: {message.sid})")
                success_count += 1

            except TwilioException as e:
                self.logger.error(f"Twilio error sending SMS to {to_number}: {e}")
            except Exception as e:
                self.logger.error(f"Error sending SMS to {to_number}: {e}")

        # Return True if at least one SMS was sent successfully
        return success_count > 0

    def test_connection(self) -> bool:
        """Test SMS connection"""
        try:
            if not self.client:
                return False

            # Try to get account info
            account = self.client.api.accounts(self.config.account_sid).fetch()
            return account.status == 'active'

        except Exception as e:
            self.logger.error(f"SMS connection test failed: {e}")
            return False

    def _create_message(self, signal_data: dict) -> str:
        """Create SMS message from signal data"""
        signal_type = signal_data.get('signal_type', 'UNKNOWN')
        symbol = signal_data.get('symbol', 'UNKNOWN')
        price = signal_data.get('price', 0)

        # Create concise message (SMS has 160 character limit)
        if signal_type.upper() == 'BUY':
            emoji = '🟢'
        elif signal_type.upper() == 'SELL':
            emoji = '🔴'
        else:
            emoji = '⚪'

        message = f"{emoji} {signal_type.upper()} {symbol} @ {price:.5f}"

        # Add risk info if available
        risk = signal_data.get('risk_management', {})
        if 'stop_loss_buy' in risk or 'stop_loss_sell' in risk:
            sl = risk.get('stop_loss_buy') or risk.get('stop_loss_sell', 0)
            tp = risk.get('take_profit_buy') or risk.get('take_profit_sell', 0)
            if sl and tp:
                message += f" | SL:{sl:.3f} TP:{tp:.3f}"

        # Add timeframe
        timeframe = signal_data.get('timeframe', '')
        if timeframe:
            message += f" | {timeframe}"

        # Ensure message fits in SMS limit
        if len(message) > 160:
            message = message[:157] + "..."

        return message