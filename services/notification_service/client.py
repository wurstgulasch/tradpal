#!/usr/bin/env python3
"""
Notification Service Client - Client for interacting with the Notification Service.

Provides methods to:
- Send notifications via different channels
- Manage notification queues
- Get notification statistics
- Configure notification settings
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from config.settings import NOTIFICATION_SERVICE_URL, API_KEY, API_SECRET


class NotificationServiceClient:
    """Client for the Notification Service microservice"""

    def __init__(self, base_url: str = NOTIFICATION_SERVICE_URL):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self) -> None:
        """Initialize the client"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    'X-API-Key': API_KEY,
                    'Content-Type': 'application/json'
                }
            )

    async def close(self) -> None:
        """Close the client"""
        if self.session:
            await self.session.close()
            self.session = None

    async def health_check(self) -> bool:
        """Check if the service is healthy"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    async def send_notification(self, message: str, notification_type: str = "info",
                               priority: str = "normal", channels: Optional[List[str]] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a notification"""
        try:
            payload = {
                'message': message,
                'type': notification_type,
                'priority': priority
            }

            if channels:
                payload['channels'] = channels
            if metadata:
                payload['metadata'] = metadata

            async with self.session.post(f"{self.base_url}/notify", json=payload) as response:
                if response.status in [200, 201]:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Notification send failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            raise

    async def send_signal_notification(self, signal_data: Dict[str, Any],
                                      channels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Send a trading signal notification"""
        try:
            payload = {
                'signal_data': signal_data,
                'channels': channels or ['telegram', 'discord']
            }

            async with self.session.post(f"{self.base_url}/notify/signal", json=payload) as response:
                if response.status in [200, 201]:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Signal notification failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to send signal notification: {e}")
            raise

    async def send_trade_notification(self, trade_data: Dict[str, Any],
                                     channels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Send a trade execution notification"""
        try:
            payload = {
                'trade_data': trade_data,
                'channels': channels or ['telegram', 'email']
            }

            async with self.session.post(f"{self.base_url}/notify/trade", json=payload) as response:
                if response.status in [200, 201]:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Trade notification failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to send trade notification: {e}")
            raise

    async def send_alert_notification(self, alert_data: Dict[str, Any],
                                     channels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Send an alert notification"""
        try:
            payload = {
                'alert_data': alert_data,
                'channels': channels or ['telegram', 'discord', 'email']
            }

            async with self.session.post(f"{self.base_url}/notify/alert", json=payload) as response:
                if response.status in [200, 201]:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Alert notification failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to send alert notification: {e}")
            raise

    async def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        try:
            async with self.session.get(f"{self.base_url}/stats") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}

        except Exception as e:
            self.logger.error(f"Failed to get notification stats: {e}")
            return {}

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get notification queue status"""
        try:
            async with self.session.get(f"{self.base_url}/queue/status") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}

        except Exception as e:
            self.logger.error(f"Failed to get queue status: {e}")
            return {}

    async def clear_queue(self, channel: Optional[str] = None) -> bool:
        """Clear notification queue"""
        try:
            params = {}
            if channel:
                params['channel'] = channel

            async with self.session.delete(f"{self.base_url}/queue", params=params) as response:
                return response.status == 200

        except Exception as e:
            self.logger.error(f"Failed to clear queue: {e}")
            return False

    async def configure_channel(self, channel: str, config: Dict[str, Any]) -> bool:
        """Configure a notification channel"""
        try:
            payload = {
                'channel': channel,
                'config': config
            }

            async with self.session.put(f"{self.base_url}/config/channel", json=payload) as response:
                return response.status == 200

        except Exception as e:
            self.logger.error(f"Failed to configure channel: {e}")
            return False

    async def test_channel(self, channel: str) -> Dict[str, Any]:
        """Test a notification channel"""
        try:
            payload = {'channel': channel}

            async with self.session.post(f"{self.base_url}/test", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Channel test failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to test channel: {e}")
            raise