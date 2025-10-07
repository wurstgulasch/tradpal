"""
Base classes and interfaces for TradPal integrations.
All integrations should inherit from these base classes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class IntegrationConfig:
    """Configuration class for integrations"""

    def __init__(self, **kwargs):
        self.enabled = kwargs.get('enabled', True)
        self.name = kwargs.get('name', self.__class__.__name__)
        self.retry_attempts = kwargs.get('retry_attempts', 3)
        self.retry_delay = kwargs.get('retry_delay', 1.0)
        self.timeout = kwargs.get('timeout', 30)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'enabled': self.enabled,
            'name': self.name,
            'retry_attempts': self.retry_attempts,
            'retry_delay': self.retry_delay,
            'timeout': self.timeout
        }

class BaseIntegration(ABC):
    """Abstract base class for all TradPal integrations"""

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the integration. Return True if successful."""
        pass

    @abstractmethod
    def send_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Send a trading signal. Return True if successful."""
        pass

    def send_signal_safe(self, signal_data: Dict[str, Any]) -> bool:
        """Send a trading signal with test environment safety check."""
        if self.is_test_environment():
            self.logger.info(f"TEST ENVIRONMENT: Would send signal via {self.__class__.__name__}, but skipping actual transmission")
            return True  # Return success in test environment
        return self.send_signal(signal_data)

    @abstractmethod
    def test_connection(self) -> bool:
        """Test the integration connection. Return True if working."""
        pass

    def send_startup_message(self) -> bool:
        """Send a startup message (optional)"""
        if self.is_test_environment():
            self.logger.info(f"TEST ENVIRONMENT: Skipping startup message for {self.__class__.__name__}")
            return True
        return self._send_startup_message()

    def send_shutdown_message(self) -> bool:
        """Send a shutdown message (optional)"""
        if self.is_test_environment():
            self.logger.info(f"TEST ENVIRONMENT: Skipping shutdown message for {self.__class__.__name__}")
            return True
        return self._send_shutdown_message()

    def _send_startup_message(self) -> bool:
        """Internal method for sending startup message - override in subclasses"""
        return True

    def _send_shutdown_message(self) -> bool:
        """Internal method for sending shutdown message - override in subclasses"""
        return True

    def is_enabled(self) -> bool:
        """Check if integration is enabled"""
        return self.config.enabled

    def is_test_environment(self) -> bool:
        """Check if we're running in a test environment"""
        return os.getenv('TEST_ENVIRONMENT') == 'true' or 'pytest' in os.sys.argv[0]

    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            'name': self.config.name,
            'enabled': self.config.enabled,
            'initialized': self._initialized,
            'type': self.__class__.__name__,
            'test_environment': self.is_test_environment()
        }

class SignalData:
    """Standardized signal data structure"""

    def __init__(self, **kwargs):
        self.timestamp = kwargs.get('timestamp', datetime.now())
        self.symbol = kwargs.get('symbol', 'EUR/USD')
        self.timeframe = kwargs.get('timeframe', '1m')
        self.signal_type = kwargs.get('signal_type', 'UNKNOWN')  # BUY, SELL, NEUTRAL
        self.price = kwargs.get('price', 0.0)
        self.indicators = kwargs.get('indicators', {})
        self.risk_management = kwargs.get('risk_management', {})
        self.metadata = kwargs.get('metadata', {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'signal_type': self.signal_type,
            'price': self.price,
            'indicators': self.indicators,
            'risk_management': self.risk_management,
            'metadata': self.metadata
        }

    @classmethod
    def from_trading_signal(cls, signal: Dict[str, Any]) -> 'SignalData':
        """Create SignalData from trading signal dictionary"""
        # Determine signal type
        signal_type = 'NEUTRAL'
        if signal.get('Buy_Signal') == 1:
            signal_type = 'BUY'
        elif signal.get('Sell_Signal') == 1:
            signal_type = 'SELL'

        # Extract indicators
        indicators = {
            'ema9': signal.get('EMA9'),
            'ema21': signal.get('EMA21'),
            'rsi': signal.get('RSI'),
            'bb_upper': signal.get('BB_upper'),
            'bb_middle': signal.get('BB_middle'),
            'bb_lower': signal.get('BB_lower'),
            'atr': signal.get('ATR'),
            'adx': signal.get('ADX')
        }

        # Extract risk management
        risk_management = {
            'position_size_percent': signal.get('Position_Size_Percent'),
            'position_size_absolute': signal.get('Position_Size_Absolute'),
            'stop_loss_buy': signal.get('Stop_Loss_Buy'),
            'take_profit_buy': signal.get('Take_Profit_Buy'),
            'stop_loss_sell': signal.get('Stop_Loss_Sell'),
            'take_profit_sell': signal.get('Take_Profit_Sell'),
            'leverage': signal.get('Leverage')
        }

        return cls(
            symbol=signal.get('symbol', 'EUR/USD'),
            timeframe=signal.get('timeframe', '1m'),
            signal_type=signal_type,
            price=signal.get('close', 0.0),
            indicators=indicators,
            risk_management=risk_management,
            metadata={'raw_signal': signal}
        )

class IntegrationManager:
    """Manager class for all integrations"""

    def __init__(self):
        self.integrations: Dict[str, BaseIntegration] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def register_integration(self, name: str, integration: BaseIntegration):
        """Register a new integration"""
        self.integrations[name] = integration
        self.logger.info(f"Registered integration: {name}")

    def unregister_integration(self, name: str):
        """Unregister an integration"""
        if name in self.integrations:
            del self.integrations[name]
            self.logger.info(f"Unregistered integration: {name}")

    def initialize_all(self) -> Dict[str, bool]:
        """Initialize all registered integrations"""
        results = {}
        for name, integration in self.integrations.items():
            if integration.is_enabled():
                try:
                    success = integration.initialize()
                    results[name] = success
                    if success:
                        integration.send_startup_message()
                        self.logger.info(f"Initialized integration: {name}")
                    else:
                        self.logger.error(f"Failed to initialize integration: {name}")
                except Exception as e:
                    self.logger.error(f"Error initializing {name}: {e}")
                    results[name] = False
            else:
                results[name] = True  # Disabled integrations are "successful"
        return results

    def send_signal_to_all(self, signal_data: Dict[str, Any]) -> Dict[str, bool]:
        """Send signal to all enabled integrations"""
        results = {}
        signal = SignalData.from_trading_signal(signal_data)

        for name, integration in self.integrations.items():
            if integration.is_enabled():
                try:
                    success = integration.send_signal_safe(signal.to_dict())
                    results[name] = success
                    if not success:
                        self.logger.warning(f"Failed to send signal via {name}")
                except Exception as e:
                    self.logger.error(f"Error sending signal via {name}: {e}")
                    results[name] = False
            else:
                results[name] = True  # Skip disabled integrations

        return results

    def test_all_connections(self) -> Dict[str, bool]:
        """Test connections for all integrations"""
        results = {}
        for name, integration in self.integrations.items():
            if integration.is_enabled():
                try:
                    success = integration.test_connection()
                    results[name] = success
                    status = "OK" if success else "FAILED"
                    self.logger.info(f"Connection test {name}: {status}")
                except Exception as e:
                    self.logger.error(f"Connection test error {name}: {e}")
                    results[name] = False
            else:
                results[name] = True  # Skip disabled integrations
        return results

    def shutdown_all(self):
        """Shutdown all integrations"""
        for name, integration in self.integrations.items():
            try:
                integration.send_shutdown_message()
                self.logger.info(f"Shutdown integration: {name}")
            except Exception as e:
                self.logger.error(f"Error shutting down {name}: {e}")

    def get_status_overview(self) -> Dict[str, Any]:
        """Get status overview of all integrations"""
        return {
            'total_integrations': len(self.integrations),
            'enabled_integrations': len([i for i in self.integrations.values() if i.is_enabled()]),
            'integrations': {name: integration.get_status() for name, integration in self.integrations.items()}
        }

# Global integration manager instance
integration_manager = IntegrationManager()