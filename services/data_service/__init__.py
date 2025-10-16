"""
Data Service - Centralized time-series data management for TradPal.

This package provides:
- Multi-source data fetching (CCXT, Yahoo Finance)
- Data quality validation and scoring
- Redis caching for performance
- REST API for data access
- Automatic fallback systems
"""

from .service import DataService, DataRequest, DataResponse, DataMetadata
from .service import DataSource, DataProvider, DataQuality, EventSystem
from .client import DataServiceClient

__version__ = "1.0.0"
__all__ = [
    "DataService",
    "DataRequest",
    "DataResponse",
    "DataMetadata",
    "DataSource",
    "DataProvider",
    "DataQuality",
    "EventSystem",
    "DataServiceClient"
]