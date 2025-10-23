#!/usr/bin/env python3
"""
Trading Bot Live Service - Live trading execution and monitoring.

This service provides comprehensive live trading capabilities including:
- Real-time signal monitoring and execution
- Order management and position tracking
- Risk management and position sizing
- Performance monitoring and reporting
- Paper trading mode for testing
"""

from .service import TradingBotLiveService, EventSystem
from .client import TradingBotLiveServiceClient

__all__ = [
    "TradingBotLiveService",
    "EventSystem",
    "TradingBotLiveServiceClient"
]