"""
Webhook Integration for TradPal Indicator
Sends trading signals via HTTP webhooks to external services
"""

import os
import json
import requests
from typing import Dict, Any, Optional, List
from integrations.base import BaseIntegration, IntegrationConfig


class WebhookConfig(IntegrationConfig):
    """Configuration for Webhook integration"""

    def __init__(self,
                 enabled: bool = True,
                 name: str = "Webhook Notifications",
                 webhook_urls: List[str] = None,
                 headers: Dict[str, str] = None,
                 method: str = "POST",
                 auth_token: str = "",
                 auth_type: str = "Bearer"):
        super().__init__(enabled=enabled, name=name)
        self.webhook_urls = webhook_urls or []
        self.headers = headers or {}
        self.method = method.upper()
        self.auth_token = auth_token
        self.auth_type = auth_type

        # Set default headers
        if 'Content-Type' not in self.headers:
            self.headers['Content-Type'] = 'application/json'

        # Add authorization header if token provided
        if self.auth_token:
            if self.auth_type.lower() == 'bearer':
                self.headers['Authorization'] = f'Bearer {self.auth_token}'
            elif self.auth_type.lower() == 'basic':
                self.headers['Authorization'] = f'Basic {self.auth_token}'

    @classmethod
    def from_env(cls) -> 'WebhookConfig':
        """Create config from environment variables"""
        webhook_urls = []
        if os.getenv('WEBHOOK_URLS'):
            webhook_urls = [url.strip() for url in os.getenv('WEBHOOK_URLS', '').split(',') if url.strip()]

        headers = {}
        if os.getenv('WEBHOOK_HEADERS'):
            try:
                headers = json.loads(os.getenv('WEBHOOK_HEADERS', '{}'))
            except json.JSONDecodeError:
                headers = {}

        return cls(
            enabled=bool(webhook_urls),
            name="Webhook Notifications",
            webhook_urls=webhook_urls,
            headers=headers,
            method=os.getenv('WEBHOOK_METHOD', 'POST'),
            auth_token=os.getenv('WEBHOOK_AUTH_TOKEN', ''),
            auth_type=os.getenv('WEBHOOK_AUTH_TYPE', 'Bearer')
        )


class WebhookIntegration(BaseIntegration):
    """Webhook integration for sending trading signals to external services"""

    def __init__(self, config: WebhookConfig):
        super().__init__(config)
        self.config: WebhookConfig = config

    def initialize(self) -> bool:
        """Initialize webhook integration"""
        try:
            if not self.config.webhook_urls:
                self.logger.error("At least one webhook URL is required")
                return False

            # Validate URLs
            for url in self.config.webhook_urls:
                if not url.startswith(('http://', 'https://')):
                    self.logger.error(f"Invalid webhook URL format: {url}")
                    return False

            self.logger.info(f"Webhook integration initialized with {len(self.config.webhook_urls)} URLs")
            self._initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize webhook integration: {e}")
            return False

    def send_signal(self, signal_data: dict) -> bool:
        """Send trading signal via webhooks"""
        if not self.config.webhook_urls:
            self.logger.warning("No webhook URLs configured")
            return False

        success_count = 0

        for url in self.config.webhook_urls:
            try:
                # Prepare payload
                payload = self._prepare_payload(signal_data)

                # Send request
                response = requests.request(
                    method=self.config.method,
                    url=url,
                    json=payload,
                    headers=self.config.headers,
                    timeout=self.config.timeout
                )

                if response.status_code in [200, 201, 202, 204]:
                    self.logger.info(f"Webhook sent successfully to {url}")
                    success_count += 1
                else:
                    self.logger.error(f"Webhook failed for {url}: {response.status_code} - {response.text}")

            except Exception as e:
                self.logger.error(f"Error sending webhook to {url}: {e}")

        # Return True if at least one webhook succeeded
        return success_count > 0

    def test_connection(self) -> bool:
        """Test webhook connections"""
        if not self.config.webhook_urls:
            return False

        success_count = 0

        for url in self.config.webhook_urls:
            try:
                # Send test payload
                test_payload = {
                    "test": True,
                    "message": "TradPal Indicator Webhook Test",
                    "timestamp": "2024-01-01T00:00:00Z"
                }

                response = requests.request(
                    method=self.config.method,
                    url=url,
                    json=test_payload,
                    headers=self.config.headers,
                    timeout=10
                )

                if response.status_code in [200, 201, 202, 204]:
                    success_count += 1
                else:
                    self.logger.warning(f"Test failed for {url}: {response.status_code}")

            except Exception as e:
                self.logger.error(f"Test connection error for {url}: {e}")

        return success_count > 0

    def _prepare_payload(self, signal_data: dict) -> Dict[str, Any]:
        """Prepare payload for webhook"""
        # Add metadata
        payload = signal_data.copy()
        payload['source'] = 'TradPal Indicator'
        payload['version'] = '1.0'

        # Ensure timestamp is ISO format
        if 'timestamp' in payload and hasattr(payload['timestamp'], 'isoformat'):
            payload['timestamp'] = payload['timestamp'].isoformat()

        return payload

    def send_startup_message(self) -> bool:
        """Send a startup message with current configuration"""
        if self.is_test_environment():
            self.logger.info(f"TEST ENVIRONMENT: Skipping startup message for {self.__class__.__name__}")
            return True

        try:
            # Import configuration
            from config.settings import (
                SYMBOL, EXCHANGE, TIMEFRAME, DEFAULT_INDICATOR_CONFIG,
                RISK_PER_TRADE, SL_MULTIPLIER, TP_MULTIPLIER, LEVERAGE_BASE,
                MTA_ENABLED, ADX_ENABLED, ADX_THRESHOLD, FIBONACCI_ENABLED
            )

            # Format indicator configuration
            indicators = []
            if DEFAULT_INDICATOR_CONFIG.get('ema', {}).get('enabled'):
                periods = DEFAULT_INDICATOR_CONFIG['ema'].get('periods', [9, 21])
                indicators.append(f"EMA{periods}")
            if DEFAULT_INDICATOR_CONFIG.get('rsi', {}).get('enabled'):
                period = DEFAULT_INDICATOR_CONFIG['rsi'].get('period', 14)
                indicators.append(f"RSI({period})")
            if DEFAULT_INDICATOR_CONFIG.get('bb', {}).get('enabled'):
                period = DEFAULT_INDICATOR_CONFIG['bb'].get('period', 20)
                indicators.append(f"BB({period})")
            if DEFAULT_INDICATOR_CONFIG.get('atr', {}).get('enabled'):
                period = DEFAULT_INDICATOR_CONFIG['atr'].get('period', 14)
                indicators.append(f"ATR({period})")
            if DEFAULT_INDICATOR_CONFIG.get('adx', {}).get('enabled'):
                indicators.append("ADX")

            indicators_str = ', '.join(indicators) if indicators else 'None'

            # Prepare startup payload
            startup_payload = {
                "event": "bot_started",
                "message": "TradPal Indicator Bot Started",
                "configuration": {
                    "symbol": SYMBOL,
                    "exchange": EXCHANGE,
                    "timeframe": TIMEFRAME,
                    "indicators": indicators_str,
                    "risk_per_trade": f"{RISK_PER_TRADE*100:.1f}%",
                    "stop_loss_multiplier": f"{SL_MULTIPLIER}x ATR",
                    "take_profit_multiplier": f"{TP_MULTIPLIER}x ATR",
                    "base_leverage": f"{LEVERAGE_BASE}x",
                    "multi_timeframe_analysis": MTA_ENABLED,
                    "adx_enabled": ADX_ENABLED,
                    "fibonacci_tp": FIBONACCI_ENABLED
                },
                "status": "active",
                "source": "TradPal Indicator",
                "version": "1.0",
                "timestamp": "2024-01-01T00:00:00Z"  # Will be overridden by datetime
            }

            success_count = 0

            for url in self.config.webhook_urls:
                try:
                    # Add current timestamp
                    payload = startup_payload.copy()
                    from datetime import datetime
                    payload['timestamp'] = datetime.now().isoformat()

                    response = requests.request(
                        method=self.config.method,
                        url=url,
                        json=payload,
                        headers=self.config.headers,
                        timeout=self.config.timeout
                    )

                    if response.status_code in [200, 201, 202, 204]:
                        self.logger.info(f"Startup webhook sent successfully to {url}")
                        success_count += 1
                    else:
                        self.logger.error(f"Startup webhook failed for {url}: {response.status_code} - {response.text}")

                except Exception as e:
                    self.logger.error(f"Error sending startup webhook to {url}: {e}")

            return success_count > 0

        except Exception as e:
            self.logger.error(f"Failed to send startup webhook: {e}")
            return False

    def _send_startup_message(self) -> bool:
        """Internal method for sending startup message - override in subclasses"""
        return self.send_startup_message()