"""
Base classes and interfaces for TradPal broker integrations.
All broker integrations should inherit from these base classes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import os
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger(__name__)

class BrokerConfig:
    """Configuration class for broker integrations"""

    def __init__(self, **kwargs):
        self.enabled = kwargs.get('enabled', False)  # Disabled by default for safety
        self.name = kwargs.get('name', self.__class__.__name__)
        self.api_key = kwargs.get('api_key', '')
        self.api_secret = kwargs.get('api_secret', '')
        self.testnet = kwargs.get('testnet', True)  # Use testnet by default
        self.exchange = kwargs.get('exchange', 'binance')
        self.retry_attempts = kwargs.get('retry_attempts', 3)
        self.retry_delay = kwargs.get('retry_delay', 1.0)
        self.timeout = kwargs.get('timeout', 30)
        self.max_position_size_percent = kwargs.get('max_position_size_percent', 1.0)  # 1% max position
        self.min_order_size = kwargs.get('min_order_size', 10.0)  # Minimum order size in USD

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (without sensitive data)"""
        return {
            'enabled': self.enabled,
            'name': self.name,
            'testnet': self.testnet,
            'exchange': self.exchange,
            'retry_attempts': self.retry_attempts,
            'retry_delay': self.retry_delay,
            'timeout': self.timeout,
            'max_position_size_percent': self.max_position_size_percent,
            'min_order_size': self.min_order_size
        }

class OrderResult:
    """Result of an order execution"""

    def __init__(self, **kwargs):
        self.order_id = kwargs.get('order_id', '')
        self.symbol = kwargs.get('symbol', '')
        self.side = kwargs.get('side', '')  # 'buy' or 'sell'
        self.type = kwargs.get('type', 'market')  # 'market', 'limit', etc.
        self.amount = kwargs.get('amount', 0.0)
        self.price = kwargs.get('price', 0.0)
        self.status = kwargs.get('status', 'unknown')  # 'filled', 'partial', 'rejected', etc.
        self.fee = kwargs.get('fee', 0.0)
        self.timestamp = kwargs.get('timestamp', datetime.now())
        self.error = kwargs.get('error', None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'type': self.type,
            'amount': self.amount,
            'price': self.price,
            'status': self.status,
            'fee': self.fee,
            'timestamp': self.timestamp.isoformat(),
            'error': str(self.error) if self.error else None
        }

class Position:
    """Current position information"""

    def __init__(self, **kwargs):
        self.symbol = kwargs.get('symbol', '')
        self.amount = kwargs.get('amount', 0.0)
        self.avg_price = kwargs.get('avg_price', 0.0)
        self.unrealized_pnl = kwargs.get('unrealized_pnl', 0.0)
        self.timestamp = kwargs.get('timestamp', datetime.now())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'amount': self.amount,
            'avg_price': self.avg_price,
            'unrealized_pnl': self.unrealized_pnl,
            'timestamp': self.timestamp.isoformat()
        }

class BaseBroker(ABC):
    """Abstract base class for all TradPal broker integrations"""

    def __init__(self, config: BrokerConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialized = False
        self._positions: Dict[str, Position] = {}

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the broker connection. Return True if successful."""
        pass

    @abstractmethod
    def get_balance(self, currency: str = 'USDT') -> float:
        """Get account balance for specified currency"""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        pass

    @abstractmethod
    def place_order(self, symbol: str, side: str, amount: float,
                   order_type: str = 'market', price: Optional[float] = None) -> OrderResult:
        """Place an order. Return OrderResult."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order. Return True if successful."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str, symbol: str) -> OrderResult:
        """Get status of an order"""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test the broker connection. Return True if working."""
        pass

    def is_enabled(self) -> bool:
        """Check if broker is enabled"""
        return self.config.enabled

    def is_test_environment(self) -> bool:
        """Check if we're running in test environment (always use testnet for safety)"""
        return self.config.testnet or os.getenv('TEST_ENVIRONMENT') == 'true'

    def validate_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> bool:
        """Validate order parameters before execution"""
        if not self.config.enabled:
            self.logger.error("Broker is not enabled")
            return False

        if amount <= 0:
            self.logger.error(f"Invalid amount: {amount}")
            return False

        if side not in ['buy', 'sell']:
            self.logger.error(f"Invalid side: {side}")
            return False

        # Check minimum order size
        if price and amount * price < self.config.min_order_size:
            self.logger.error(f"Order size too small: {amount * price} < {self.config.min_order_size}")
            return False

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get broker status"""
        return {
            'name': self.config.name,
            'enabled': self.config.enabled,
            'initialized': self._initialized,
            'testnet': self.config.testnet,
            'exchange': self.config.exchange,
            'positions_count': len(self._positions)
        }

class BrokerManager:
    """Manager class for broker integrations"""

    def __init__(self):
        self.brokers: Dict[str, BaseBroker] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def register_broker(self, name: str, broker: BaseBroker):
        """Register a new broker"""
        self.brokers[name] = broker
        self.logger.info(f"Registered broker: {name}")

    def unregister_broker(self, name: str):
        """Unregister a broker"""
        if name in self.brokers:
            del self.brokers[name]
            self.logger.info(f"Unregistered broker: {name}")

    def initialize_all(self) -> Dict[str, bool]:
        """Initialize all registered brokers"""
        results = {}
        for name, broker in self.brokers.items():
            if broker.is_enabled():
                try:
                    success = broker.initialize()
                    results[name] = success
                    if success:
                        self.logger.info(f"Initialized broker: {name}")
                    else:
                        self.logger.error(f"Failed to initialize broker: {name}")
                except Exception as e:
                    self.logger.error(f"Error initializing {name}: {e}")
                    results[name] = False
            else:
                results[name] = True  # Disabled brokers are "successful"
        return results

    def get_primary_broker(self) -> Optional[BaseBroker]:
        """Get the primary enabled broker"""
        for broker in self.brokers.values():
            if broker.is_enabled():
                return broker
        return None

    def test_all_connections(self) -> Dict[str, bool]:
        """Test connections for all brokers"""
        results = {}
        for name, broker in self.brokers.items():
            if broker.is_enabled():
                try:
                    success = broker.test_connection()
                    results[name] = success
                    status = "OK" if success else "FAILED"
                    self.logger.info(f"Connection test {name}: {status}")
                except Exception as e:
                    self.logger.error(f"Connection test error {name}: {e}")
                    results[name] = False
            else:
                results[name] = True  # Skip disabled brokers
        return results

    def shutdown_all(self):
        """Shutdown all brokers"""
        for name, broker in self.brokers.items():
            try:
                self.logger.info(f"Shutdown broker: {name}")
            except Exception as e:
                self.logger.error(f"Error shutting down {name}: {e}")

    def get_status_overview(self) -> Dict[str, Any]:
        """Get status overview of all brokers"""
        return {
            'total_brokers': len(self.brokers),
            'enabled_brokers': len([b for b in self.brokers.values() if b.is_enabled()]),
            'brokers': {name: broker.get_status() for name, broker in self.brokers.items()}
        }

# Global broker manager instance
broker_manager = BrokerManager()