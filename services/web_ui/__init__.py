#!/usr/bin/env python3
"""
Web UI Service - Web interface for trading platform.

This service provides comprehensive web interface capabilities including:
- Dashboard data aggregation and presentation
- Strategy configuration and management
- Live trading monitoring and control
- Backtesting visualization and analysis
- User authentication and session management
- Real-time data streaming
"""

from .api import app
from .service import WebUIService, EventSystem
from .client import WebUIServiceClient, LoginRequest, StrategyConfig, BacktestRequest

__all__ = [
    "app",
    "WebUIService",
    "EventSystem",
    "WebUIServiceClient",
    "LoginRequest",
    "StrategyConfig",
    "BacktestRequest"
]