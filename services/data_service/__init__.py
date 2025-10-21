# TradPal Data Service
"""
Centralized data management service for the TradPal trading system.

This service consolidates the following previously separate services:
- data_service: Core data fetching and caching
- alternative_data_service: Sentiment, on-chain, and economic data
- market_regime_detection_service: Market regime classification

The data service provides a unified interface for all data operations
with quality validation, caching, and real-time updates.
"""

__version__ = "3.0.1"
__author__ = "TradPal Team"
__description__ = "TradPal Data Service - Centralized data management"

# Import main components for easy access
from .data_sources.service import DataService

__all__ = [
    "DataService",
    # TODO: Add AlternativeDataService and MarketRegimeDetectionServiceClient when implemented
]