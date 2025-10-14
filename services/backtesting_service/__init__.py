"""
Backtesting Service - Microservice for historical trading strategy backtesting.

This module provides the BacktestingService class and supporting components
for running comprehensive backtests of trading strategies.
"""

from .service import BacktestingService, AsyncBacktester
from .api import BacktestingAPI, run_backtesting_service

__all__ = [
    'BacktestingService',
    'AsyncBacktester',
    'BacktestingAPI',
    'run_backtesting_service'
]