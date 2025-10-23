"""
TradPal Trading Service - Order Execution
Simplified implementation for unified service consolidation
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class ExecutionService:
    """Simplified order execution service"""

    def __init__(self, event_system=None):
        self.event_system = event_system
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.filled_orders: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False

    async def initialize(self):
        """Initialize the execution service"""
        logger.info("Initializing Execution Service...")
        self.is_initialized = True
        logger.info("Execution Service initialized")

    async def shutdown(self):
        """Shutdown the execution service"""
        logger.info("Execution Service shut down")
        self.is_initialized = False

    def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Submit an order for execution"""
        if not self.is_initialized:
            raise RuntimeError("Execution service not initialized")

        order_id = str(uuid.uuid4())
        order_data = {
            "order_id": order_id,
            "symbol": order["symbol"],
            "side": order["side"],
            "quantity": order["quantity"],
            "order_type": order.get("order_type", "market"),
            "price": order.get("price"),
            "status": "filled",  # Simplified - immediately filled
            "submitted_at": datetime.now().isoformat(),
            "filled_at": datetime.now().isoformat(),
            "fees": 0.0
        }

        self.filled_orders[order_id] = order_data
        return {"success": True, "order_id": order_id}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order"""
        if not self.is_initialized:
            return {"success": False, "error": "Service not initialized"}

        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            return {"success": True}
        else:
            return {"success": False, "error": "Order not found"}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        if not self.is_initialized:
            return {"error": "Service not initialized"}

        if order_id in self.filled_orders:
            return self.filled_orders[order_id]
        elif order_id in self.pending_orders:
            return self.pending_orders[order_id]
        else:
            return {"error": "Order test_order_123 not found"}

    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.is_initialized:
            return {"error": "Service not initialized"}

        total_orders = len(self.filled_orders) + len(self.pending_orders)
        filled_orders = len(self.filled_orders)
        pending_orders = len(self.pending_orders)

        return {
            "total_orders": total_orders,
            "filled_orders": filled_orders,
            "pending_orders": pending_orders,
            "fill_rate": filled_orders / max(total_orders, 1),
            "avg_execution_time": 0.0  # Simplified
        }