"""
CCXT Broker Integration for TradPal
Supports live trading via CCXT library with multiple exchanges
"""

import ccxt
from typing import Dict, Any, Optional, List
import logging
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from .base_broker import BaseBroker, BrokerConfig, OrderResult, Position

logger = logging.getLogger(__name__)

class CCXTBroker(BaseBroker):
    """CCXT-based broker implementation for live trading"""

    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self.exchange = None
        self._initialize_exchange()

    def _initialize_exchange(self):
        """Initialize CCXT exchange instance"""
        try:
            exchange_class = getattr(ccxt, self.config.exchange)
            self.exchange = exchange_class({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True,
                'timeout': self.config.timeout * 1000,  # CCXT uses milliseconds
                'test': self.config.testnet,  # Use testnet/sandbox if available
            })

            # Set sandbox mode for testnet
            if self.config.testnet and hasattr(self.exchange, 'set_sandbox_mode'):
                self.exchange.set_sandbox_mode(True)
                self.logger.info(f"Enabled testnet/sandbox mode for {self.config.exchange}")

        except Exception as e:
            self.logger.error(f"Failed to initialize {self.config.exchange} exchange: {e}")
            self.exchange = None

    def initialize(self) -> bool:
        """Initialize the broker connection"""
        if not self.exchange:
            self._initialize_exchange()

        if not self.exchange:
            return False

        try:
            # Test connection by loading markets
            self.exchange.load_markets()
            self._initialized = True
            self.logger.info(f"Successfully initialized {self.config.exchange} broker")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize broker: {e}")
            return False

    def get_balance(self, currency: str = 'USDT') -> float:
        """Get account balance for specified currency"""
        if not self._initialized:
            self.logger.error("Broker not initialized")
            return 0.0

        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get(currency, {}).get('free', 0.0))
        except Exception as e:
            self.logger.error(f"Failed to fetch balance: {e}")
            return 0.0

    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        if not self._initialized:
            self.logger.error("Broker not initialized")
            return {}

        try:
            # For spot trading, positions are based on balance
            balance = self.exchange.fetch_balance()
            positions = {}

            for currency, data in balance.items():
                if currency != 'info' and float(data.get('total', 0)) > 0:
                    # Convert to position format (simplified for spot)
                    positions[currency] = Position(
                        symbol=currency,
                        amount=float(data.get('total', 0)),
                        avg_price=0.0,  # Not applicable for spot
                        unrealized_pnl=0.0  # Not applicable for spot
                    )

            self._positions = positions
            return positions
        except Exception as e:
            self.logger.error(f"Failed to fetch positions: {e}")
            return {}

    def place_order(self, symbol: str, side: str, amount: float,
                   order_type: str = 'market', price: Optional[float] = None) -> OrderResult:
        """Place an order via CCXT"""
        if not self.validate_order(symbol, side, amount, price):
            return OrderResult(error="Order validation failed")

        if not self._initialized:
            return OrderResult(error="Broker not initialized")

        try:
            # Ensure symbol format (e.g., 'BTC/USDT')
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"

            # Get market info for precision
            market = self.exchange.market(symbol)
            amount_precision = market['precision']['amount']
            price_precision = market['precision']['price']

            # Round amount and price to exchange precision
            if amount_precision > 0:
                amount = float(Decimal(str(amount)).quantize(Decimal('1e-{}'.format(amount_precision)), rounding=ROUND_DOWN))

            if price and price_precision > 0:
                price = float(Decimal(str(price)).quantize(Decimal('1e-{}'.format(price_precision)), rounding=ROUND_DOWN))

            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount
            }

            if price and order_type == 'limit':
                order_params['price'] = price

            # Place the order
            order = self.exchange.create_order(**order_params)

            # Convert to OrderResult
            result = OrderResult(
                order_id=str(order.get('id', '')),
                symbol=symbol,
                side=side,
                type=order_type,
                amount=float(order.get('amount', 0)),
                price=float(order.get('price', 0)),
                status=order.get('status', 'unknown'),
                fee=float(order.get('fee', {}).get('cost', 0)) if order.get('fee') else 0,
                timestamp=datetime.fromtimestamp(order.get('timestamp', time.time()) / 1000) if order.get('timestamp') else datetime.now()
            )

            self.logger.info(f"Order placed: {result.to_dict()}")
            return result

        except Exception as e:
            error_msg = f"Failed to place order: {e}"
            self.logger.error(error_msg)
            return OrderResult(error=error_msg)

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        if not self._initialized:
            self.logger.error("Broker not initialized")
            return False

        try:
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"

            result = self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str, symbol: str) -> OrderResult:
        """Get status of an order"""
        if not self._initialized:
            self.logger.error("Broker not initialized")
            return OrderResult(error="Broker not initialized")

        try:
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"

            order = self.exchange.fetch_order(order_id, symbol)

            result = OrderResult(
                order_id=str(order.get('id', '')),
                symbol=symbol,
                side=order.get('side', ''),
                type=order.get('type', ''),
                amount=float(order.get('amount', 0)),
                price=float(order.get('price', 0)),
                status=order.get('status', 'unknown'),
                fee=float(order.get('fee', {}).get('cost', 0)) if order.get('fee') else 0,
                timestamp=datetime.fromtimestamp(order.get('timestamp', time.time()) / 1000) if order.get('timestamp') else datetime.now()
            )

            return result
        except Exception as e:
            error_msg = f"Failed to fetch order status: {e}"
            self.logger.error(error_msg)
            return OrderResult(error=error_msg)

    def test_connection(self) -> bool:
        """Test the broker connection"""
        if not self._initialized:
            return False

        try:
            # Test by fetching ticker for BTC/USDT
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            return ticker is not None
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False