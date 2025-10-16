"""
TradPal Alert Forwarder Service
Forwards Falco security alerts to notification services (Telegram, Discord, etc.)
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field


@dataclass
class AlertConfig:
    """Configuration for alert forwarding"""

    # Falco log monitoring
    falco_log_path: str = "/var/log/falco/falco.log"
    falco_namespace: str = "tradpal-security"

    # Alert filtering
    alert_min_priority: str = "WARNING"  # EMERGENCY, ALERT, CRITICAL, ERROR, WARNING, NOTICE, INFO, DEBUG
    alert_enabled_rules: List[str] = None  # None = all rules enabled

    # Notification services
    telegram_enabled: bool = True
    discord_enabled: bool = True
    email_enabled: bool = False

    # Alert batching
    alert_batching: bool = True
    alert_batch_interval: int = 300  # 5 minutes
    alert_max_batch_size: int = 10

    # Rate limiting
    alert_rate_limit: int = 10
    alert_cooldown: int = 60  # seconds

    def __post_init__(self):
        if self.alert_enabled_rules is None:
            self.alert_enabled_rules = []

    @classmethod
    def from_env(cls) -> 'AlertConfig':
        """Create config from environment variables"""
        return cls(
            falco_log_path=os.getenv('FALCO_LOG_PATH', '/var/log/falco/falco.log'),
            falco_namespace=os.getenv('FALCO_NAMESPACE', 'tradpal-security'),
            alert_min_priority=os.getenv('ALERT_MIN_PRIORITY', 'WARNING'),
            alert_enabled_rules=json.loads(os.getenv('ALERT_ENABLED_RULES', '[]')),
            telegram_enabled=bool(os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true'),
            discord_enabled=bool(os.getenv('DISCORD_ENABLED', 'true').lower() == 'true'),
            email_enabled=bool(os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'),
            alert_batching=bool(os.getenv('ALERT_BATCHING', 'true').lower() == 'true'),
            alert_batch_interval=int(os.getenv('ALERT_BATCH_INTERVAL', '300')),
            alert_max_batch_size=int(os.getenv('ALERT_MAX_BATCH_SIZE', '10')),
            alert_rate_limit=int(os.getenv('ALERT_RATE_LIMIT', '10')),
            alert_cooldown=int(os.getenv('ALERT_COOLDOWN', '60'))
        )


class FalcoAlert(BaseModel):
    """Represents a Falco security alert"""

    timestamp: str = Field(..., description="Alert timestamp")
    rule: str = Field(..., description="Falco rule name")
    priority: str = Field(..., description="Alert priority")
    message: str = Field(..., description="Alert message")
    output: str = Field(..., description="Full alert output")
    source: str = Field(default="falco", description="Alert source")
    hostname: Optional[str] = Field(None, description="Host where alert occurred")
    tags: List[str] = Field(default_factory=list, description="Alert tags")

    @classmethod
    def from_log_line(cls, log_line: str) -> Optional['FalcoAlert']:
        """Parse a Falco log line into an alert object"""
        try:
            # Try to parse as JSON first
            if log_line.strip().startswith('{'):
                data = json.loads(log_line)
                return cls(
                    timestamp=data.get('time', data.get('timestamp', datetime.now().isoformat())),
                    rule=data.get('rule', 'unknown'),
                    priority=data.get('priority', 'INFO'),
                    message=data.get('output', log_line),
                    output=data.get('output', log_line),
                    hostname=data.get('hostname'),
                    tags=data.get('tags', [])
                )
            else:
                # Parse plain text log format
                # Example: 2025-10-15T10:30:00.000000000Z rule=Unauthorized API Key Access prio=WARNING
                parts = log_line.split()
                if len(parts) < 3:
                    return None

                timestamp = parts[0] if 'T' in parts[0] else datetime.now().isoformat()
                rule = ""
                priority = "INFO"

                for part in parts[1:]:
                    if part.startswith('rule='):
                        rule = part.split('=', 1)[1]
                    elif part.startswith('prio='):
                        priority = part.split('=', 1)[1]

                return cls(
                    timestamp=timestamp,
                    rule=rule,
                    priority=priority,
                    message=log_line,
                    output=log_line
                )

        except Exception:
            return None

    def should_forward(self, config: AlertConfig) -> bool:
        """Check if this alert should be forwarded based on config"""
        # Check priority
        priority_levels = {
            'EMERGENCY': 8, 'ALERT': 7, 'CRITICAL': 6, 'ERROR': 5,
            'WARNING': 4, 'NOTICE': 3, 'INFO': 2, 'DEBUG': 1
        }

        alert_level = priority_levels.get(self.priority.upper(), 0)
        min_level = priority_levels.get(config.alert_min_priority.upper(), 4)

        if alert_level < min_level:
            return False

        # Check enabled rules
        if config.alert_enabled_rules and self.rule not in config.alert_enabled_rules:
            return False

        return True

    def get_severity_emoji(self) -> str:
        """Get emoji for alert severity"""
        severity_map = {
            'EMERGENCY': '🚨',
            'ALERT': '🚨',
            'CRITICAL': '🔴',
            'ERROR': '🟠',
            'WARNING': '🟡',
            'NOTICE': 'ℹ️',
            'INFO': '📝',
            'DEBUG': '🐛'
        }
        return severity_map.get(self.priority.upper(), '❓')

    def format_for_telegram(self) -> str:
        """Format alert for Telegram"""
        emoji = self.get_severity_emoji()

        message = f"""
{emoji} *FALCO SECURITY ALERT* {emoji}

🚨 *Rule:* {self.rule}
⚠️ *Priority:* {self.priority}
🕐 *Time:* {self.timestamp}

📝 *Details:*
{self.message}

🔍 *Source:* {self.source}
        """.strip()

        if self.hostname:
            message += f"\n🏠 *Host:* {self.hostname}"

        if self.tags:
            message += f"\n🏷️ *Tags:* {', '.join(self.tags)}"

        return message

    def format_for_discord(self) -> Dict[str, Any]:
        """Format alert for Discord embed"""
        emoji = self.get_severity_emoji()

        # Color coding
        color_map = {
            'EMERGENCY': 0xff0000,  # Red
            'ALERT': 0xff4500,      # Orange Red
            'CRITICAL': 0xdc143c,   # Crimson
            'ERROR': 0xff6347,      # Tomato
            'WARNING': 0xffa500,    # Orange
            'NOTICE': 0x4169e1,     # Royal Blue
            'INFO': 0x32cd32,       # Lime Green
            'DEBUG': 0x808080       # Gray
        }

        color = color_map.get(self.priority.upper(), 0x808080)

        embed = {
            "title": f"{emoji} Falco Security Alert",
            "description": f"**Rule:** {self.rule}\n**Priority:** {self.priority}",
            "color": color,
            "fields": [
                {
                    "name": "📝 Details",
                    "value": self.message[:1024],  # Discord limit
                    "inline": False
                },
                {
                    "name": "🔍 Source",
                    "value": self.source,
                    "inline": True
                }
            ],
            "footer": {
                "text": "TradPal Security Monitoring"
            },
            "timestamp": self.timestamp
        }

        if self.hostname:
            embed["fields"].append({
                "name": "🏠 Host",
                "value": self.hostname,
                "inline": True
            })

        if self.tags:
            embed["fields"].append({
                "name": "🏷️ Tags",
                "value": ", ".join(self.tags),
                "inline": True
            })

        return embed