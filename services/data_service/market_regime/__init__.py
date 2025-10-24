"""
Market Regime Detection Module

This module provides advanced market regime detection and analysis capabilities
integrated into the TradPal data service.
"""

from .detector import MarketRegimeDetector, MarketRegime, RegimeAnalysis
from .client import MarketRegimeDetectionServiceClient

__version__ = "2.0.0"
__all__ = [
    'MarketRegimeDetector',
    'MarketRegime',
    'RegimeAnalysis',
    'MarketRegimeDetectionServiceClient'
]