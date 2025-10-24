"""
Alert Management Module for Model Monitoring Service
Handles alert generation, routing, and escalation
"""

import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Callable
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alerts for model monitoring, including generation, routing, and escalation.
    """

    def __init__(self, notification_service_url: str = "http://localhost:8010",
                 alert_cooldown_minutes: int = 60):
        """
        Initialize alert manager.

        Args:
            notification_service_url: URL of the notification service
            alert_cooldown_minutes: Cooldown period between similar alerts
        """
        self.notification_service_url = notification_service_url
        self.alert_cooldown_minutes = alert_cooldown_minutes
        self.alert_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_handlers: Dict[str, Callable] = {}

        # Alert escalation levels
        self.escalation_levels = {
            'info': 1,
            'warning': 2,
            'error': 3,
            'critical': 4
        }

        # Register default alert handlers
        self._register_default_handlers()

    def register_alert_handler(self, alert_type: str, handler: Callable):
        """
        Register a custom alert handler.

        Args:
            alert_type: Type of alert to handle
            handler: Handler function
        """
        self.alert_handlers[alert_type] = handler
        logger.info(f"Registered alert handler for type: {alert_type}")

    def generate_alert(self, model_id: str, alert_type: str, message: str,
                      severity: str = 'warning', metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a new alert.

        Args:
            model_id: Model identifier
            alert_type: Type of alert (drift, performance, etc.)
            message: Alert message
            severity: Alert severity (info, warning, error, critical)
            metadata: Additional alert metadata

        Returns:
            Alert ID
        """
        alert_id = f"{model_id}_{alert_type}_{int(datetime.now().timestamp())}"

        alert = {
            'id': alert_id,
            'model_id': model_id,
            'type': alert_type,
            'message': message,
            'severity': severity,
            'level': self.escalation_levels.get(severity, 2),
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'status': 'active',
            'escalation_count': 0
        }

        # Check for alert cooldown
        if self._is_alert_cooldown_active(model_id, alert_type):
            logger.info(f"Alert cooldown active for {model_id}:{alert_type}, skipping")
            return alert_id

        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history[model_id].append(alert)

        logger.warning(f"Generated alert {alert_id}: {message}")

        # Trigger alert handling
        try:
            # Only create task if we have a running event loop
            import asyncio
            if asyncio.get_event_loop().is_running():
                asyncio.create_task(self._handle_alert(alert))
            else:
                # For testing, run synchronously
                import asyncio
                asyncio.run(self._handle_alert(alert))
        except RuntimeError:
            # No event loop, skip async handling (for testing)
            pass

        return alert_id

    def resolve_alert(self, alert_id: str, resolution: str = "auto-resolved"):
        """
        Resolve an active alert.

        Args:
            alert_id: Alert identifier
            resolution: Resolution description
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert['status'] = 'resolved'
            alert['resolved_at'] = datetime.now().isoformat()
            alert['resolution'] = resolution

            logger.info(f"Resolved alert {alert_id}: {resolution}")

            # Remove from active alerts
            del self.active_alerts[alert_id]

    def get_active_alerts(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get active alerts, optionally filtered by model.

        Args:
            model_id: Optional model identifier filter

        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())

        if model_id:
            alerts = [a for a in alerts if a['model_id'] == model_id]

        return sorted(alerts, key=lambda x: x['level'], reverse=True)

    def get_alert_history(self, model_id: Optional[str] = None,
                         hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get alert history, optionally filtered by model and time.

        Args:
            model_id: Optional model identifier filter
            hours: Hours of history to retrieve

        Returns:
            List of historical alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        all_alerts = []

        for model_alerts in self.alert_history.values():
            for alert in model_alerts:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                if alert_time >= cutoff_time:
                    all_alerts.append(alert)

        if model_id:
            all_alerts = [a for a in all_alerts if a['model_id'] == model_id]

        return sorted(all_alerts, key=lambda x: x['timestamp'], reverse=True)

    def escalate_alert(self, alert_id: str, new_severity: str):
        """
        Escalate an alert to a higher severity level.

        Args:
            alert_id: Alert identifier
            new_severity: New severity level
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            old_severity = alert['severity']

            if self.escalation_levels.get(new_severity, 0) > alert['level']:
                alert['severity'] = new_severity
                alert['level'] = self.escalation_levels[new_severity]
                alert['escalation_count'] += 1
                alert['escalated_at'] = datetime.now().isoformat()

                logger.warning(f"Escalated alert {alert_id} from {old_severity} to {new_severity}")

                # Trigger escalation handling
                asyncio.create_task(self._handle_alert(alert, escalation=True))

    async def _handle_alert(self, alert: Dict[str, Any], escalation: bool = False):
        """
        Handle alert processing and notification.

        Args:
            alert: Alert data
            escalation: Whether this is an escalation
        """
        try:
            # Use custom handler if available
            alert_type = alert['type']
            if alert_type in self.alert_handlers:
                await self.alert_handlers[alert_type](alert)
            else:
                # Use default notification
                await self._send_notification(alert, escalation)

        except Exception as e:
            logger.error(f"Failed to handle alert {alert['id']}: {e}")

    async def _send_notification(self, alert: Dict[str, Any], escalation: bool = False):
        """
        Send alert notification via notification service.

        Args:
            alert: Alert data
            escalation: Whether this is an escalation
        """
        try:
            notification_data = {
                'type': 'model_monitoring_alert',
                'severity': alert['severity'],
                'title': f"Model Alert: {alert['model_id']}",
                'message': alert['message'],
                'metadata': {
                    'alert_id': alert['id'],
                    'model_id': alert['model_id'],
                    'alert_type': alert['type'],
                    'escalation': escalation,
                    'escalation_count': alert.get('escalation_count', 0),
                    **alert.get('metadata', {})
                },
                'channels': self._get_notification_channels(alert['severity'])
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.notification_service_url}/notify",
                    json=notification_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Sent notification for alert {alert['id']}")
                    else:
                        logger.error(f"Failed to send notification: {response.status}")

        except Exception as e:
            logger.error(f"Notification failed for alert {alert['id']}: {e}")

    def _get_notification_channels(self, severity: str) -> List[str]:
        """
        Get notification channels based on severity.

        Args:
            severity: Alert severity

        Returns:
            List of notification channels
        """
        channels = ['log']  # Always log

        if severity in ['warning', 'error', 'critical']:
            channels.extend(['telegram', 'email'])

        if severity == 'critical':
            channels.append('sms')

        return channels

    def _is_alert_cooldown_active(self, model_id: str, alert_type: str) -> bool:
        """
        Check if alert cooldown is active for this model and alert type.

        Args:
            model_id: Model identifier
            alert_type: Alert type

        Returns:
            True if cooldown is active
        """
        if model_id not in self.alert_history:
            return False

        cooldown_cutoff = datetime.now() - timedelta(minutes=self.alert_cooldown_minutes)

        for alert in reversed(self.alert_history[model_id]):
            if alert['type'] == alert_type:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                if alert_time >= cooldown_cutoff and alert['status'] == 'active':
                    return True
                break

        return False

    def _register_default_handlers(self):
        """Register default alert handlers."""
        # Drift alert handler
        async def handle_drift_alert(alert: Dict[str, Any]):
            drift_score = alert['metadata'].get('drift_score', 0)
            threshold = alert['metadata'].get('threshold', 0.1)

            if drift_score > threshold * 2:  # Severe drift
                self.escalate_alert(alert['id'], 'critical')
            elif drift_score > threshold * 1.5:  # Moderate drift
                self.escalate_alert(alert['id'], 'error')

        # Performance alert handler
        async def handle_performance_alert(alert: Dict[str, Any]):
            degradation_pct = alert['metadata'].get('degradation_percentage', 0)

            if degradation_pct > 50:  # Severe degradation
                self.escalate_alert(alert['id'], 'critical')
            elif degradation_pct > 25:  # Moderate degradation
                self.escalate_alert(alert['id'], 'error')

        self.register_alert_handler('drift', handle_drift_alert)
        self.register_alert_handler('performance', handle_performance_alert)

    def get_alert_summary(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get alert summary statistics.

        Args:
            model_id: Optional model identifier filter

        Returns:
            Alert summary
        """
        alerts = self.get_alert_history(model_id, hours=168)  # Last 7 days

        summary = {
            'total_alerts': len(alerts),
            'active_alerts': len(self.get_active_alerts(model_id)),
            'alerts_by_severity': defaultdict(int),
            'alerts_by_type': defaultdict(int),
            'recent_alerts': alerts[:10]  # Last 10 alerts
        }

        for alert in alerts:
            summary['alerts_by_severity'][alert['severity']] += 1
            summary['alerts_by_type'][alert['type']] += 1

        return dict(summary)

    def cleanup_old_alerts(self, days: int = 30):
        """
        Clean up old resolved alerts.

        Args:
            days: Days to keep alerts
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        for model_id in self.alert_history:
            self.alert_history[model_id] = [
                alert for alert in self.alert_history[model_id]
                if datetime.fromisoformat(alert['timestamp']) >= cutoff_date
            ]

        logger.info(f"Cleaned up alerts older than {days} days")

    # Email notification methods (for direct email alerts)
    def send_email_alert(self, alert: Dict[str, Any], recipients: List[str],
                        smtp_config: Dict[str, str]):
        """
        Send email alert directly (fallback method).

        Args:
            alert: Alert data
            recipients: Email recipients
            smtp_config: SMTP configuration
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Model Alert: {alert['model_id']} - {alert['severity'].upper()}"

            body = f"""
Model Monitoring Alert

Model: {alert['model_id']}
Type: {alert['type']}
Severity: {alert['severity']}
Time: {alert['timestamp']}

Message: {alert['message']}

Metadata: {json.dumps(alert.get('metadata', {}), indent=2)}
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(smtp_config['smtp_server'], int(smtp_config['smtp_port']))
            server.starttls()
            server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(msg)
            server.quit()

            logger.info(f"Sent email alert for {alert['id']}")

        except Exception as e:
            logger.error(f"Email alert failed for {alert['id']}: {e}")