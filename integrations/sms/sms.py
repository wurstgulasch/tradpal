"""
SMS Integration for TradPal Indicator
Sends trading signals via SMS             except Exception as e:
                # Log error sending SMS
                self.logger.error(f"Error sending SMS to {to_number}: {e}")io
"""

from integrations.base import BaseIntegration


class SMSIntegration(BaseIntegration):
    """SMS integration for sending trading signals via Twilio"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.client = None
        self._twilio_available = None

    @property
    def twilio_available(self):
        """Check if twilio is available"""
        if self._twilio_available is None:
            try:
                import twilio  # noqa: F401
                self._twilio_available = True
            except ImportError:
                self._twilio_available = False
        return self._twilio_available

    def initialize(self) -> bool:
        """Initialize SMS integration"""
        try:
            if not self.twilio_available:
                self.logger.error("Twilio package not installed. Install with: pip install twilio")
                return False

            if not all([self.config.account_sid, self.config.auth_token, self.config.from_number, self.config.to_numbers]):
                self.logger.error("SMS configuration incomplete: account_sid, auth_token, from_number, and to_numbers required")
                return False

            # Import twilio components here
            from twilio.rest import Client
            # Initialize Twilio client
            self.client = Client(self.config.account_sid, self.config.auth_token)

            self.logger.info(f"SMS integration initialized for {len(self.config.to_numbers)} recipients")
            self._initialized = True
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

            except Exception as e:
                # Log error sending SMS
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
                risk_part = f" | SL:{sl:.3f} TP:{tp:.3f}"
                if len(message + risk_part) <= 160:
                    message += risk_part

        # Add timeframe
        timeframe = signal_data.get('timeframe', '')
        if timeframe:
            tf_part = f" | {timeframe}"
            if len(message + tf_part) <= 160:
                message += tf_part

        # Ensure message fits in SMS limit
        if len(message) > 160:
            message = message[:157] + "..."

        return message