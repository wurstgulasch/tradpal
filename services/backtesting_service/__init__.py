# TradPal Backtesting Service
"""
Unified backtesting service for historical analysis, ML training, and strategy optimization.

This service consolidates:
- backtesting_service: Core backtesting engine
- ml_trainer: ML model training
- optimizer: Parameter optimization
- walk_forward_optimizer: Walk-forward analysis
- discovery_service: Parameter discovery
"""

__version__ = "3.0.1"
__author__ = "TradPal Team"
__description__ = "TradPal Backtesting Service - Historical Analysis & ML Training"

# Import main components for easy access
# Temporarily disabled due to FastAPI compatibility issues
# from .main import BacktestingServiceOrchestrator

__all__ = [
    # "BacktestingServiceOrchestrator",
]