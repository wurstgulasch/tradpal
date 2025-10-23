"""
TradPal Alert Forwarder Service
Monitors Falco logs and forwards security alerts to notification services
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Set
from pathlib import Path
from datetime import datetime, timedelta

import aiofiles

try:
    from config import AlertConfig, FalcoAlert
except ImportError:
    from .config import AlertConfig, FalcoAlert


class AlertForwarder:
    """Service for forwarding Falco alerts to notification services"""

    def __init__(self, config: AlertConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # State tracking
        self.running = False
        self.last_position = 0
        self.processed_alerts: Set[str] = set()
        self.alert_buffer: List[FalcoAlert] = []
        self.last_batch_time = datetime.now()
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute)

        # Notification services
        self.telegram_integration = None
        self.discord_integration = None
        self.email_integration = None

    async def initialize(self) -> bool:
        """Initialize the alert forwarder"""
        try:
            self.logger.info("Initializing Alert Forwarder...")

            # Initialize notification services
            await self._init_notification_services()

            # Check if Falco log file exists
            log_path = Path(self.config.falco_log_path)
            if not log_path.exists():
                self.logger.warning(f"Falco log file not found: {log_path}")
                self.logger.info("Alert forwarder will wait for log file to be created")

            self.logger.info("Alert Forwarder initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Alert Forwarder: {e}")
            return False

    async def _init_notification_services(self):
        """Initialize notification service integrations"""
        try:
            # Import integrations
            if self.config.telegram_enabled:
                try:
                    from integrations.telegram import TelegramIntegration, TelegramConfig
                    telegram_config = TelegramConfig.from_env()
                    if telegram_config.enabled:
                        self.telegram_integration = TelegramIntegration(telegram_config)
                        await asyncio.get_event_loop().run_in_executor(None, self.telegram_integration.initialize)
                        self.logger.info("Telegram integration initialized")
                    else:
                        self.logger.info("Telegram integration disabled (no config)")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Telegram integration: {e}")

            if self.config.discord_enabled:
                try:
                    from integrations.discord import DiscordIntegration, DiscordConfig
                    discord_config = DiscordConfig.from_env()
                    if discord_config.enabled:
                        self.discord_integration = DiscordIntegration(discord_config)
                        await asyncio.get_event_loop().run_in_executor(None, self.discord_integration.initialize)
                        self.logger.info("Discord integration initialized")
                    else:
                        self.logger.info("Discord integration disabled (no config)")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Discord integration: {e}")

            if self.config.email_enabled:
                try:
                    from integrations.email_integration import EmailIntegration, EmailConfig
                    email_config = EmailConfig.from_env()
                    if email_config.enabled:
                        self.email_integration = EmailIntegration(email_config)
                        await asyncio.get_event_loop().run_in_executor(None, self.email_integration.initialize)
                        self.logger.info("Email integration initialized")
                    else:
                        self.logger.info("Email integration disabled (no config)")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Email integration: {e}")

        except Exception as e:
            self.logger.error(f"Error initializing notification services: {e}")

    async def start(self):
        """Start the alert forwarding service"""
        if self.running:
            self.logger.warning("Alert forwarder is already running")
            return

        self.running = True
        self.logger.info("Starting Alert Forwarder service...")

        try:
            # Send startup notification
            await self._send_startup_notification()

            # Start monitoring loop
            await self._monitoring_loop()

        except Exception as e:
            self.logger.error(f"Error in alert forwarder: {e}")
        finally:
            self.running = False
            await self._send_shutdown_notification()

    async def stop(self):
        """Stop the alert forwarding service"""
        self.logger.info("Stopping Alert Forwarder service...")
        self.running = False

        # Send any remaining buffered alerts
        if self.alert_buffer:
            await self._flush_alert_buffer()

    async def _monitoring_loop(self):
        """Main monitoring loop for Falco logs"""
        self.logger.info(f"Starting Falco log monitoring: {self.config.falco_log_path}")

        while self.running:
            try:
                await self._process_log_file()

                # Handle batched alerts
                if self.config.batch_alerts:
                    await self._check_batch_timeout()

                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _process_log_file(self):
        """Process the Falco log file for new alerts"""
        try:
            log_path = Path(self.config.falco_log_path)

            # Wait for log file to exist
            if not log_path.exists():
                return

            async with aiofiles.open(log_path, 'r', encoding='utf-8') as f:
                # Seek to last known position
                await f.seek(self.last_position)

                async for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Parse alert from log line
                    alert = FalcoAlert.from_log_line(line)
                    if alert and alert.should_forward(self.config):
                        await self._handle_alert(alert)

                # Update position
                self.last_position = await f.tell()

        except Exception as e:
            self.logger.error(f"Error processing log file: {e}")

    async def _handle_alert(self, alert: FalcoAlert):
        """Handle a new security alert"""
        # Create unique alert ID to prevent duplicates
        alert_id = f"{alert.timestamp}_{alert.rule}_{hash(alert.message)}"

        if alert_id in self.processed_alerts:
            return  # Already processed

        self.processed_alerts.add(alert_id)

        # Rate limiting check
        if not self.rate_limiter.allow():
            self.logger.warning(f"Rate limit exceeded, skipping alert: {alert.rule}")
            return

        self.logger.info(f"Processing alert: {alert.rule} ({alert.priority})")

        if self.config.batch_alerts:
            # Add to buffer
            self.alert_buffer.append(alert)

            # Check if buffer should be flushed
            if len(self.alert_buffer) >= self.config.max_batch_size:
                await self._flush_alert_buffer()
        else:
            # Send immediately
            await self._send_alert(alert)

    async def _check_batch_timeout(self):
        """Check if batch timeout has been reached"""
        now = datetime.now()
        if (now - self.last_batch_time).seconds >= self.config.batch_interval:
            if self.alert_buffer:
                await self._flush_alert_buffer()

    async def _flush_alert_buffer(self):
        """Flush all buffered alerts"""
        if not self.alert_buffer:
            return

        self.logger.info(f"Flushing {len(self.alert_buffer)} buffered alerts")

        # Group alerts by priority for batch message
        alerts_by_priority = {}
        for alert in self.alert_buffer:
            priority = alert.priority
            if priority not in alerts_by_priority:
                alerts_by_priority[priority] = []
            alerts_by_priority[priority].append(alert)

        # Send batched alerts
        for priority, alerts in alerts_by_priority.items():
            if len(alerts) == 1:
                # Single alert
                await self._send_alert(alerts[0])
            else:
                # Multiple alerts of same priority
                await self._send_batch_alert(priority, alerts)

        # Clear buffer
        self.alert_buffer.clear()
        self.last_batch_time = datetime.now()

    async def _send_alert(self, alert: FalcoAlert):
        """Send a single alert to all enabled notification services"""
        tasks = []

        if self.telegram_integration:
            tasks.append(self._send_to_telegram(alert))

        if self.discord_integration:
            tasks.append(self._send_to_discord(alert))

        if self.email_integration:
            tasks.append(self._send_to_email(alert))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_batch_alert(self, priority: str, alerts: List[FalcoAlert]):
        """Send a batch of alerts with the same priority"""
        alert_count = len(alerts)
        first_alert = alerts[0]

        # Create batch message
        batch_message = f"🚨 {alert_count} {priority} security alerts detected"

        # Add details for first few alerts
        details = []
        for i, alert in enumerate(alerts[:3]):  # Show first 3 alerts
            details.append(f"• {alert.rule}")

        if alert_count > 3:
            details.append(f"• ... and {alert_count - 3} more")

        batch_message += "\n" + "\n".join(details)
        batch_message += f"\n\n🕐 {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}"

        # Send batch notification
        tasks = []

        if self.telegram_integration:
            tasks.append(self._send_text_to_telegram(batch_message))

        if self.discord_integration:
            embed = {
                "title": f"🚨 Batch Security Alerts",
                "description": f"{alert_count} {priority} alerts detected",
                "color": 0xff0000 if priority == "CRITICAL" else 0xffa500,
                "fields": [
                    {
                        "name": "Alert Details",
                        "value": "\n".join(details[:5]),  # Limit to 5 details
                        "inline": False
                    }
                ],
                "footer": {
                    "text": "TradPal Security Monitoring"
                },
                "timestamp": datetime.now().isoformat()
            }
            tasks.append(self._send_embed_to_discord(embed))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_to_telegram(self, alert: FalcoAlert):
        """Send alert to Telegram"""
        try:
            if self.telegram_integration:
                message = alert.format_for_telegram()
                success = await asyncio.get_event_loop().run_in_executor(
                    None, self.telegram_integration.send_message, message
                )
                if success:
                    self.logger.info("Alert sent to Telegram")
                else:
                    self.logger.error("Failed to send alert to Telegram")
        except Exception as e:
            self.logger.error(f"Error sending to Telegram: {e}")

    async def _send_text_to_telegram(self, message: str):
        """Send text message to Telegram"""
        try:
            if self.telegram_integration:
                success = await asyncio.get_event_loop().run_in_executor(
                    None, self.telegram_integration.send_message, message
                )
                if success:
                    self.logger.info("Batch alert sent to Telegram")
                else:
                    self.logger.error("Failed to send batch alert to Telegram")
        except Exception as e:
            self.logger.error(f"Error sending batch alert to Telegram: {e}")

    async def _send_to_discord(self, alert: FalcoAlert):
        """Send alert to Discord"""
        try:
            if self.discord_integration:
                embed = alert.format_for_discord()
                # Discord integration expects signal_data format, so we create a mock signal
                mock_signal = {
                    "signal_type": alert.priority,
                    "symbol": "SECURITY",
                    "price": 0.0,
                    "indicators": {"alert": alert.message},
                    "risk_management": {"rule": alert.rule},
                    "timestamp": alert.timestamp
                }
                success = await asyncio.get_event_loop().run_in_executor(
                    None, self.discord_integration.send_signal, mock_signal
                )
                if success:
                    self.logger.info("Alert sent to Discord")
                else:
                    self.logger.error("Failed to send alert to Discord")
        except Exception as e:
            self.logger.error(f"Error sending to Discord: {e}")

    async def _send_embed_to_discord(self, embed: Dict):
        """Send embed to Discord"""
        try:
            if self.discord_integration:
                # Create a custom payload for Discord
                import requests
                payload = {
                    "username": "TradPal Security",
                    "embeds": [embed]
                }
                response = requests.post(
                    self.discord_integration.config.webhook_url,
                    json=payload,
                    timeout=10
                )
                if response.status_code == 204:
                    self.logger.info("Batch alert sent to Discord")
                else:
                    self.logger.error(f"Failed to send batch alert to Discord: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error sending batch alert to Discord: {e}")

    async def _send_to_email(self, alert: FalcoAlert):
        """Send alert via email"""
        try:
            if self.email_integration:
                subject = f"TradPal Security Alert: {alert.rule}"
                body = alert.format_for_telegram()  # Reuse Telegram format
                # Email integration would need to be implemented
                self.logger.info("Email alert sending not yet implemented")
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")

    async def _send_startup_notification(self):
        """Send startup notification"""
        startup_message = """
🛡️ *TradPal Alert Forwarder Started*

✅ Security alert monitoring is now active
📊 Monitoring Falco logs for security events
🚨 Will forward alerts to configured notification services

🔧 Configuration:
• Minimum Priority: {self.config.min_priority}
• Batching: {'Enabled' if self.config.batch_alerts else 'Disabled'}
• Rate Limit: {self.config.rate_limit_per_minute} alerts/minute

📱 Active Services:
{telegram_status}
{discord_status}
{email_status}

🔔 You will receive notifications for:
• CRITICAL security events
• HIGH priority alerts
• WARNING and above (configurable)

⏱️ Monitoring started at: {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}
        """.strip()

        telegram_status = "• Telegram: ✅ Active" if self.telegram_integration else "• Telegram: ❌ Disabled"
        discord_status = "• Discord: ✅ Active" if self.discord_integration else "• Discord: ❌ Disabled"
        email_status = "• Email: ✅ Active" if self.email_integration else "• Email: ❌ Disabled"

        startup_message = startup_message.format(
            self=self,
            telegram_status=telegram_status,
            discord_status=discord_status,
            email_status=email_status
        )

        # Send to all active services
        tasks = []
        if self.telegram_integration:
            tasks.append(self._send_text_to_telegram(startup_message))
        if self.discord_integration:
            embed = {
                "title": "🛡️ Alert Forwarder Started",
                "description": "Security alert monitoring is now active",
                "color": 0x28a745,
                "fields": [
                    {
                        "name": "Configuration",
                        "value": f"Min Priority: {self.config.min_priority}\nBatching: {'Yes' if self.config.batch_alerts else 'No'}",
                        "inline": True
                    },
                    {
                        "name": "Active Services",
                        "value": f"Telegram: {'Yes' if self.telegram_integration else 'No'}\nDiscord: {'Yes' if self.discord_integration else 'No'}",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "TradPal Security Monitoring"
                },
                "timestamp": datetime.now().isoformat()
            }
            tasks.append(self._send_embed_to_discord(embed))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_shutdown_notification(self):
        """Send shutdown notification"""
        shutdown_message = """
🛑 *TradPal Alert Forwarder Stopped*

Security alert monitoring has been stopped.
You will no longer receive security notifications.

⏱️ Stopped at: {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}
        """.strip()

        tasks = []
        if self.telegram_integration:
            tasks.append(self._send_text_to_telegram(shutdown_message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class RateLimiter:
    """Simple rate limiter for alert forwarding"""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.alerts_this_minute = 0
        self.current_minute = datetime.now().minute

    def allow(self) -> bool:
        """Check if alert should be allowed based on rate limit"""
        now = datetime.now()

        # Reset counter if minute changed
        if now.minute != self.current_minute:
            self.alerts_this_minute = 0
            self.current_minute = now.minute

        if self.alerts_this_minute < self.max_per_minute:
            self.alerts_this_minute += 1
            return True

        return False