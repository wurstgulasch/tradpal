"""
TradPal Trading Service
Unified trading bot service consolidating live trading, risk management, and AI components
"""

from .orchestrator import TradingServiceOrchestrator

__version__ = "1.0.0"
__all__ = ["TradingServiceOrchestrator"]