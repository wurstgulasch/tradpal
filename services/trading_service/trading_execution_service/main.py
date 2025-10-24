"""
TradPal Trading Execution Service API
FastAPI application for trade execution and order management
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import asyncio

from service import TradingExecutionService

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TradPal Trading Execution Service",
    description="Specialized service for trade execution and order management",
    version="1.0.0"
)

# Global service instance
execution_service = TradingExecutionService()


# Pydantic models for API requests/responses
class OrderRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Order side (buy/sell)")
    order_type: str = Field(default="market", description="Order type (market/limit/stop)")
    quantity: float = Field(..., description="Order quantity")
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = Field(default="GTC", description="Time in force")

class OrderResponse(BaseModel):
    order_id: str
    status: str
    executed_quantity: float = 0.0
    executed_price: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class PositionInfo(BaseModel):
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percentage: float

class PortfolioResponse(BaseModel):
    positions: List[PositionInfo] = Field(default_factory=list)
    total_value: float
    total_pnl: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Starting Trading Execution Service...")
    await execution_service.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Trading Execution Service...")
    await execution_service.close()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "trading_execution_service"}


@app.post("/order", response_model=OrderResponse)
async def place_order(request: OrderRequest):
    """Place a trading order"""
    try:
        result = await execution_service.place_order(
            request.symbol,
            request.side,
            request.order_type,
            request.quantity,
            request.price,
            request.stop_price,
            request.time_in_force
        )

        return OrderResponse(
            order_id=result["order_id"],
            status=result["status"],
            executed_quantity=result.get("executed_quantity", 0.0),
            executed_price=result.get("executed_price")
        )

    except Exception as e:
        logger.error(f"Order placement failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/order/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order"""
    try:
        result = await execution_service.cancel_order(order_id)
        return result
    except Exception as e:
        logger.error(f"Order cancellation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/order/{order_id}")
async def get_order_status(order_id: str):
    """Get order status"""
    try:
        status = await execution_service.get_order_status(order_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get order status: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")


@app.get("/orders")
async def get_open_orders():
    """Get all open orders"""
    try:
        orders = await execution_service.get_open_orders()
        return {"orders": orders}
    except Exception as e:
        logger.error(f"Failed to get open orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio():
    """Get current portfolio information"""
    try:
        portfolio = await execution_service.get_portfolio()

        positions = [
            PositionInfo(
                symbol=pos["symbol"],
                quantity=pos["quantity"],
                entry_price=pos["entry_price"],
                current_price=pos["current_price"],
                pnl=pos["pnl"],
                pnl_percentage=pos["pnl_percentage"]
            )
            for pos in portfolio["positions"]
        ]

        return PortfolioResponse(
            positions=positions,
            total_value=portfolio["total_value"],
            total_pnl=portfolio["total_pnl"]
        )

    except Exception as e:
        logger.error(f"Failed to get portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/balance")
async def get_account_balance():
    """Get account balance information"""
    try:
        balance = await execution_service.get_account_balance()
        return balance
    except Exception as e:
        logger.error(f"Failed to get balance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/close-position")
async def close_position(symbol: str, quantity: Optional[float] = None):
    """Close a position"""
    try:
        result = await execution_service.close_position(symbol, quantity)
        return result
    except Exception as e:
        logger.error(f"Position closure failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8016,  # Different port for trading execution service
        reload=True
    )