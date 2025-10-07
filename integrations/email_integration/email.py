"""
Email Integration for TradPal Indicator
Sends trading signals via email notifications
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from integrations.base import BaseIntegration, IntegrationConfig


class EmailConfig(IntegrationConfig):
    """Configuration for Email integration"""

    def __init__(self,
                 enabled: bool = True,
                 name: str = "Email Notifications",
                 smtp_server: str = "smtp.gmail.com",
                 smtp_port: int = 587,
                 username: str = "",
                 password: str = "",
                 recipients: List[str] = None,
                 use_tls: bool = True):
        super().__init__(enabled=enabled, name=name)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients or []
        self.use_tls = use_tls

    @classmethod
    def from_env(cls) -> 'EmailConfig':
        """Create config from environment variables"""
        return cls(
            enabled=bool(os.getenv('EMAIL_USERNAME') and os.getenv('EMAIL_PASSWORD')),
            name="Email Notifications",
            smtp_server=os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com'),
            smtp_port=int(os.getenv('EMAIL_SMTP_PORT', '587')),
            username=os.getenv('EMAIL_USERNAME', ''),
            password=os.getenv('EMAIL_PASSWORD', ''),
            recipients=os.getenv('EMAIL_RECIPIENTS', '').split(',') if os.getenv('EMAIL_RECIPIENTS') else []
        )


class EmailIntegration(BaseIntegration):
    """Email integration for sending trading signals"""

    def __init__(self, config: EmailConfig):
        super().__init__(config)
        self.config: EmailConfig = config
        self.server: Optional[smtplib.SMTP] = None

    def initialize(self) -> bool:
        """Initialize email connection"""
        try:
            if self.config.use_tls:
                self.server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
                self.server.starttls()
            else:
                self.server = smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port)

            if self.config.username and self.config.password:
                self.server.login(self.config.username, self.config.password)

            self.logger.info(f"Email integration initialized for {self.config.username}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize email integration: {e}")
            return False

    def send_signal(self, signal_data: dict) -> bool:
        """Send trading signal via email"""
        try:
            if not self.server or not self.config.recipients:
                self.logger.warning("Email server not initialized or no recipients configured")
                return False

            subject = self._create_subject(signal_data)
            body = self._create_body(signal_data)

            msg = MIMEMultipart()
            msg['From'] = self.config.username
            msg['To'] = ', '.join(self.config.recipients)
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'html'))

            self.server.send_message(msg)
            self.logger.info(f"Email sent to {len(self.config.recipients)} recipients")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False

    def test_connection(self) -> bool:
        """Test email connection"""
        try:
            # Try to establish a connection
            if self.config.use_tls:
                server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port)

            if self.config.username and self.config.password:
                server.login(self.config.username, self.config.password)

            server.quit()
            return True

        except Exception as e:
            self.logger.error(f"Email connection test failed: {e}")
            return False

    def _create_subject(self, signal_data: dict) -> str:
        """Create email subject line"""
        signal_type = signal_data.get('signal_type', 'UNKNOWN')
        symbol = signal_data.get('symbol', 'UNKNOWN')
        price = signal_data.get('price', 0)

        return f"🚨 TradPal Signal: {signal_type} {symbol} @ {price:.5f}"

    def _create_body(self, signal_data: dict) -> str:
        """Create HTML email body"""
        signal_type = signal_data.get('signal_type', 'UNKNOWN')
        symbol = signal_data.get('symbol', 'UNKNOWN')
        timeframe = signal_data.get('timeframe', 'UNKNOWN')
        price = signal_data.get('price', 0)
        confidence = signal_data.get('confidence', 0)
        reason = signal_data.get('reason', '')

        # Color coding for signals
        if signal_type.upper() == 'BUY':
            color = '#28a745'
            emoji = '🟢'
        elif signal_type.upper() == 'SELL':
            color = '#dc3545'
            emoji = '🔴'
        else:
            color = '#6c757d'
            emoji = '⚪'

        # Indicators section
        indicators = signal_data.get('indicators', {})
        indicators_html = ""
        for key, value in indicators.items():
            if isinstance(value, float):
                indicators_html += f"<li><strong>{key.upper()}:</strong> {value:.4f}</li>"
            else:
                indicators_html += f"<li><strong>{key.upper()}:</strong> {value}</li>"

        # Risk management section
        risk = signal_data.get('risk_management', {})
        risk_html = ""
        for key, value in risk.items():
            if isinstance(value, float):
                risk_html += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value:.2f}</li>"
            else:
                risk_html += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .signal-box {{ border: 2px solid {color}; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .signal-header {{ font-size: 18px; font-weight: bold; color: {color}; }}
                .confidence {{ color: #666; font-style: italic; }}
                .section {{ margin: 15px 0; }}
                .section h3 {{ color: #333; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                ul {{ list-style-type: none; padding: 0; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="signal-box">
                <div class="signal-header">
                    {emoji} {signal_type.upper()} SIGNAL - {symbol}
                </div>
                <div class="confidence">
                    Confidence: {confidence:.1%} | Timeframe: {timeframe}
                </div>
            </div>

            <div class="section">
                <h3>📊 Signal Details</h3>
                <ul>
                    <li><strong>Symbol:</strong> {symbol}</li>
                    <li><strong>Price:</strong> {price:.5f}</li>
                    <li><strong>Timeframe:</strong> {timeframe}</li>
                    <li><strong>Timestamp:</strong> {signal_data.get('timestamp', 'N/A')}</li>
                </ul>
            </div>

            <div class="section">
                <h3>📈 Technical Indicators</h3>
                <ul>
                    {indicators_html}
                </ul>
            </div>

            <div class="section">
                <h3>⚠️ Risk Management</h3>
                <ul>
                    {risk_html}
                </ul>
            </div>

            <div class="section">
                <h3>💡 Analysis</h3>
                <p>{reason}</p>
            </div>

            <hr>
            <p style="color: #666; font-size: 12px;">
                This signal was generated by TradPal Indicator. Always perform your own analysis before trading.
            </p>
        </body>
        </html>
        """

        return html