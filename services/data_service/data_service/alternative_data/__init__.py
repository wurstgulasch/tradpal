#!/usr/bin/env python3
"""
Alternative Data Service - Advanced Data Sources for AI Trading

This service provides comprehensive alternative data collection and processing
for enhanced trading signals and market analysis.

Features:
- Social Media Sentiment Analysis (Twitter, Reddit, News)
- On-Chain Blockchain Metrics
- Economic Indicators and Macro Data
- Real-time Data Processing Pipeline
- Fallback Mechanisms and Data Validation
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np

from config.settings import (
    ALTERNATIVE_DATA_UPDATE_INTERVAL,
    SENTIMENT_DATA_SOURCES,
    ONCHAIN_DATA_SOURCES,
    ECONOMIC_DATA_SOURCES
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Alternative data source types."""
    SENTIMENT = "sentiment"
    ONCHAIN = "onchain"
    ECONOMIC = "economic"
    NEWS = "news"
    SOCIAL = "social"


class SentimentSource(Enum):
    """Sentiment data sources."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    FEAR_GREED_INDEX = "fear_greed_index"


class OnChainMetric(Enum):
    """On-chain metrics."""
    ACTIVE_ADDRESSES = "active_addresses"
    TRANSACTION_VOLUME = "transaction_volume"
    EXCHANGE_FLOWS = "exchange_flows"
    WHALE_MOVEMENTS = "whale_movements"
    NVT_RATIO = "nvt_ratio"
    MINING_DIFFICULTY = "mining_difficulty"


class EconomicIndicator(Enum):
    """Economic indicators."""
    FED_FUNDS_RATE = "fed_funds_rate"
    CPI = "cpi"
    UNEMPLOYMENT_RATE = "unemployment_rate"
    GDP_GROWTH = "gdp_growth"
    PMI = "pmi"


@dataclass
class SentimentData:
    """Sentiment analysis data."""
    symbol: str
    source: SentimentSource
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    volume: int  # Number of analyzed items
    timestamp: datetime
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class OnChainData:
    """On-chain blockchain data."""
    symbol: str
    metric: OnChainMetric
    value: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EconomicData:
    """Economic indicator data."""
    indicator: EconomicIndicator
    value: float
    timestamp: datetime
    source: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AlternativeDataPacket:
    """Complete alternative data packet for a symbol."""
    symbol: str
    sentiment_data: List[SentimentData]
    onchain_data: List[OnChainData]
    economic_data: List[EconomicData]
    fear_greed_index: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ProcessedFeatures:
    """Processed alternative data features for ML models."""
    symbol: str
    sentiment_features: Dict[str, float]
    onchain_features: Dict[str, float]
    economic_features: Dict[str, float]
    composite_features: Dict[str, float]
    timestamp: datetime

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for ML processing."""
        data = {
            **self.sentiment_features,
            **self.onchain_features,
            **self.economic_features,
            **self.composite_features,
            'timestamp': self.timestamp
        }
        return pd.DataFrame([data])

from .client import AlternativeDataService

__all__ = [
    "AlternativeDataService",
    "DataSource", "SentimentSource", "OnChainMetric", "EconomicIndicator",
    "SentimentData", "OnChainData", "EconomicData", "AlternativeDataPacket", "ProcessedFeatures"
]