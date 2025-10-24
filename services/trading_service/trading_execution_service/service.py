"""
TradPal Trading Service - Order Execution
Simplified implementation for unified service consolidation
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class TradingExecutionService:
    """Simplified order execution service"""

    def __init__(self, event_system=None):
        self.event_system = event_system
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.filled_orders: Dict[str, Dict[str, Any]] = {}
        self.cancelled_orders: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False

    async def initialize(self):
        """Initialize the execution service"""
        logger.info("Initializing Trading Execution Service...")
        self.is_initialized = True
        logger.info("Trading Execution Service initialized")

    async def shutdown(self):
        """Shutdown the execution service"""
        logger.info("Trading Execution Service shut down")
        self.is_initialized = False

    async def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Submit an order for execution"""
        if not self.is_initialized:
            raise RuntimeError("Trading execution service not initialized")

        order_id = str(uuid.uuid4())
        order_data = {
            "order_id": order_id,
            "symbol": order["symbol"],
            "side": order["side"],
            "quantity": order["quantity"],
            "order_type": order.get("order_type", "market"),
            "price": order.get("price"),
            "status": "pending",
            "submitted_at": datetime.now().isoformat(),
            "fees": 0.0
        }

        # Store pending order
        self.pending_orders[order_id] = order_data

        # Simulate immediate execution for demo purposes
        await self._execute_order(order_id)

        return order_data

    async def _execute_order(self, order_id: str):
        """Execute a pending order (simplified simulation)"""
        if order_id not in self.pending_orders:
            return

        order = self.pending_orders[order_id]

        # Simulate execution
        executed_order = order.copy()
        executed_order.update({
            "status": "filled",
            "filled_at": datetime.now().isoformat(),
            "filled_quantity": order["quantity"],
            "average_price": order.get("price", 50000.0),  # Simulated price
            "fees": order["quantity"] * order.get("price", 50000.0) * 0.001  # 0.1% fee
        })

        # Move from pending to filled
        del self.pending_orders[order_id]
        self.filled_orders[order_id] = executed_order

        logger.info(f"Order {order_id} executed: {executed_order}")

        # Publish execution event
        if self.event_system:
            await self.event_system.publish("order_executed", executed_order)

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order"""
        if not self.is_initialized:
            raise RuntimeError("Trading execution service not initialized")

        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order["status"] = "cancelled"
            order["cancelled_at"] = datetime.now().isoformat()

            # Move to cancelled
            self.cancelled_orders[order_id] = order
            del self.pending_orders[order_id]

            return {"success": True, "order": order}
        else:
            return {"success": False, "error": "Order not found or already executed"}

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get the status of an order"""
        if not self.is_initialized:
            raise RuntimeError("Trading execution service not initialized")

        # Check all order collections
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]
        elif order_id in self.filled_orders:
            return self.filled_orders[order_id]
        elif order_id in self.cancelled_orders:
            return self.cancelled_orders[order_id]
        else:
            return {"error": "Order not found"}

    async def get_portfolio_positions(self) -> Dict[str, Any]:
        """Get current portfolio positions"""
        if not self.is_initialized:
            raise RuntimeError("Trading execution service not initialized")

        # Aggregate positions from filled orders
        positions = {}
        total_value = 0.0

        for order in self.filled_orders.values():
            symbol = order["symbol"]
            if symbol not in positions:
                positions[symbol] = {
                    "symbol": symbol,
                    "quantity": 0.0,
                    "average_price": 0.0,
                    "total_value": 0.0
                }

            # Update position based on order side
            quantity = order["filled_quantity"]
            price = order["average_price"]

            if order["side"] == "buy":
                positions[symbol]["quantity"] += quantity
                # Update weighted average price
                current_qty = positions[symbol]["quantity"] - quantity
                current_value = current_qty * positions[symbol]["average_price"]
                new_value = quantity * price
                positions[symbol]["average_price"] = (current_value + new_value) / positions[symbol]["quantity"]
            elif order["side"] == "sell":
                positions[symbol]["quantity"] -= quantity

            positions[symbol]["total_value"] = positions[symbol]["quantity"] * positions[symbol]["average_price"]
            total_value += positions[symbol]["total_value"]

        return {
            "positions": list(positions.values()),
            "total_portfolio_value": total_value,
            "total_positions": len([p for p in positions.values() if p["quantity"] != 0]),
            "timestamp": datetime.now().isoformat()
        }

    async def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get order history"""
        if not self.is_initialized:
            raise RuntimeError("Trading execution service not initialized")

        all_orders = []

        # Combine all order types
        for order_dict in [self.filled_orders, self.cancelled_orders, self.pending_orders]:
            for order in order_dict.values():
                if symbol is None or order["symbol"] == symbol:
                    all_orders.append(order)

        # Sort by submission time (most recent first)
        all_orders.sort(key=lambda x: x["submitted_at"], reverse=True)

        return all_orders[:limit]

    async def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate trading performance metrics"""
        if not self.is_initialized:
            raise RuntimeError("Trading execution service not initialized")

        filled_orders = list(self.filled_orders.values())

        if not filled_orders:
            return {"total_trades": 0, "message": "No trades executed yet"}

        # Calculate basic metrics
        total_trades = len(filled_orders)
        total_fees = sum(order.get("fees", 0) for order in filled_orders)

        # Calculate P&L (simplified - assumes current price equals average price)
        total_pnl = 0.0
        winning_trades = 0

        for order in filled_orders:
            if order["side"] == "buy":
                # Assume we sold at the same price (simplified)
                pnl = 0.0  # No unrealized P&L in this simple model
            else:
                pnl = 0.0

            if pnl > 0:
                winning_trades += 1

            total_pnl += pnl

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            "total_trades": total_trades,
            "total_fees": total_fees,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "average_trade_size": sum(order["filled_quantity"] for order in filled_orders) / total_trades,
            "timestamp": datetime.now().isoformat()
        }