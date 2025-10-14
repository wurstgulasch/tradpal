"""
Trading Bot Live Service Package

This package contains the live trading bot microservice that handles
real-time trading operations with event-driven architecture.
"""

from .service import TradingBotLiveService, main

__all__ = ['TradingBotLiveService', 'main']