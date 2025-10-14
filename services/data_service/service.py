#!/usr/bin/env python3
"""
Data Service - Centralized time-series data management for TradPal.

This service provides:
- Unified data fetching from multiple sources (CCXT, Yahoo Finance, etc.)
- Data validation and quality assurance
- Caching with Redis for performance
- Time-series data storage and retrieval
- Metadata management and data lineage
- Automatic fallback systems for data reliability
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator

# Optional imports with fallbacks
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None

try:
    import yfinance as yf
    YAHOO_AVAILABLE = True
except ImportError:
    YAHOO_AVAILABLE = False
    yf = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources."""
    CCXT = "ccxt"
    YAHOO_FINANCE = "yahoo"
    CSV = "csv"
    JSON = "json"


class DataQuality(Enum):
    """Data quality indicators."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


class DataProvider(Enum):
    """Supported data providers."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    YAHOO = "yahoo"
    LOCAL = "local"


@dataclass
class DataMetadata:
    """Metadata for data requests."""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    source: DataSource
    provider: DataProvider
    fetched_at: datetime
    quality_score: float
    quality_level: DataQuality
    record_count: int
    columns: List[str]
    cache_key: str
    checksum: str
    fallback_used: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, (DataSource, DataProvider, DataQuality)):
                data[key] = value.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataMetadata':
        """Create from dictionary."""
        # Convert ISO strings back to datetime
        for key in ['fetched_at', 'start_date', 'end_date']:
            if key in data and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])

        # Convert enum values back to enums
        if 'source' in data:
            data['source'] = DataSource(data['source'])
        if 'provider' in data:
            data['provider'] = DataProvider(data['provider'])
        if 'quality_level' in data:
            data['quality_level'] = DataQuality(data['quality_level'])

        return cls(**data)


class DataRequest(BaseModel):
    """Request model for data fetching."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe (e.g., '1d', '1h', '15m')")
    start_date: str = Field(..., description="Start date in ISO format")
    end_date: str = Field(..., description="End date in ISO format")
    source: Optional[str] = Field("ccxt", description="Preferred data source")
    provider: Optional[str] = Field("binance", description="Preferred data provider")
    use_cache: bool = Field(True, description="Whether to use cached data")
    validate_quality: bool = Field(True, description="Whether to validate data quality")

    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        """Validate date format."""
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use ISO format (YYYY-MM-DDTHH:MM:SS)")

    @validator('timeframe')
    def validate_timeframe(cls, v):
        """Validate timeframe format."""
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        if v not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {v}. Valid options: {valid_timeframes}")
        return v


class DataResponse(BaseModel):
    """Response model for data requests."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cache_hit: bool = False
    processing_time: float = 0.0


class EventSystem:
    """Simple event system for service communication."""

    def __init__(self):
        self._handlers: Dict[str, List[callable]] = {}

    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish an event."""
        if event_type in self._handlers:
            tasks = []
            for handler in self._handlers[event_type]:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(data))
                else:
                    tasks.append(asyncio.to_thread(handler, data))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def subscribe(self, event_type: str, handler: callable):
        """Subscribe to an event."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)


class DataService:
    """
    Centralized data service for time-series financial data.

    Features:
    - Multi-source data fetching with automatic fallbacks
    - Redis caching for performance
    - Data quality validation and scoring
    - Metadata tracking and lineage
    - Async processing for scalability
    """

    def __init__(self, event_system: Optional[EventSystem] = None, redis_url: str = "redis://localhost:6379"):
        self.event_system = event_system or EventSystem()
        self.redis_client = None
        self.cache_ttl = 3600  # 1 hour default

        # Initialize Redis if available
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")

        # Data source priorities for fallbacks
        self.source_priorities = {
            DataSource.CCXT: [DataProvider.BINANCE, DataProvider.COINBASE, DataProvider.KRAKEN],
            DataSource.YAHOO_FINANCE: [DataProvider.YAHOO],
        }

        # Quality thresholds
        self.quality_thresholds = {
            DataQuality.EXCELLENT: 0.95,
            DataQuality.GOOD: 0.85,
            DataQuality.FAIR: 0.70,
            DataQuality.POOR: 0.50,
        }

        logger.info("Data Service initialized")

    def _generate_cache_key(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> str:
        """Generate a unique cache key for data requests."""
        key_data = f"{symbol}:{timeframe}:{start_date.isoformat()}:{end_date.isoformat()}"
        return f"data:{hashlib.md5(key_data.encode()).hexdigest()}"

    def _calculate_data_quality(self, df: pd.DataFrame) -> Tuple[float, DataQuality]:
        """
        Calculate data quality score and level.

        Quality factors:
        - Completeness (no NaN values)
        - Consistency (logical OHLC relationships)
        - Timeliness (data freshness)
        - Volume validity
        """
        if df.empty:
            return 0.0, DataQuality.INVALID

        score = 1.0

        # Check for NaN values
        nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        score -= nan_ratio * 0.3

        # Check OHLC relationships (O <= H, L <= O, C <= H, C >= L, etc.)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df['open'] > df['high']) |
                (df['low'] > df['open']) |
                (df['close'] > df['high']) |
                (df['close'] < df['low'])
            ).sum()
            ohlc_invalid_ratio = invalid_ohlc / len(df)
            score -= ohlc_invalid_ratio * 0.4

        # Check volume validity
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            volume_invalid_ratio = negative_volume / len(df)
            score -= volume_invalid_ratio * 0.2

        # Check for reasonable price ranges
        if 'close' in df.columns:
            price_range = df['close'].max() / df['close'].min()
            if price_range > 1000:  # Unrealistic price movement
                score -= 0.1

        # Determine quality level
        score = max(0.0, min(1.0, score))

        if score >= self.quality_thresholds[DataQuality.EXCELLENT]:
            quality = DataQuality.EXCELLENT
        elif score >= self.quality_thresholds[DataQuality.GOOD]:
            quality = DataQuality.GOOD
        elif score >= self.quality_thresholds[DataQuality.FAIR]:
            quality = DataQuality.FAIR
        elif score >= self.quality_thresholds[DataQuality.POOR]:
            quality = DataQuality.POOR
        else:
            quality = DataQuality.INVALID

        return score, quality

    def _calculate_checksum(self, df: pd.DataFrame) -> str:
        """Calculate checksum for data integrity verification."""
        # Convert to string representation for hashing
        data_str = df.to_json(orient='records', date_format='iso')
        return hashlib.sha256(data_str.encode()).hexdigest()

    async def _fetch_from_ccxt(self, symbol: str, timeframe: str, start_date: datetime,
                              end_date: datetime, provider: DataProvider) -> Optional[pd.DataFrame]:
        """Fetch data from CCXT exchange."""
        if not CCXT_AVAILABLE:
            logger.warning("CCXT not available")
            return None

        try:
            # Initialize exchange
            exchange_class = getattr(ccxt, provider.value, None)
            if not exchange_class:
                logger.warning(f"Exchange {provider.value} not supported by CCXT")
                return None

            exchange = exchange_class()
            exchange.load_markets()

            # Convert timeframe to CCXT format
            ccxt_timeframe = timeframe
            if timeframe == '1d':
                ccxt_timeframe = '1d'
            elif timeframe.endswith('h'):
                ccxt_timeframe = timeframe
            elif timeframe.endswith('m'):
                ccxt_timeframe = timeframe

            # Fetch OHLCV data
            since = int(start_date.timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, ccxt_timeframe, since=since, limit=1000)

            if not ohlcv:
                logger.warning(f"No data returned from {provider.value} for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Filter date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            logger.info(f"Fetched {len(df)} records from {provider.value}")
            return df

        except Exception as e:
            logger.error(f"CCXT fetch failed for {provider.value}: {e}")
            return None

    async def _fetch_from_yahoo(self, symbol: str, timeframe: str, start_date: datetime,
                               end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        if not YAHOO_AVAILABLE:
            logger.warning("Yahoo Finance not available")
            return None

        try:
            # Convert symbol format if needed
            yahoo_symbol = symbol.replace('/', '-')  # BTC/USDT -> BTC-USDT

            # Map timeframe
            yahoo_interval = '1d'
            if timeframe in ['1h', '4h', '1d']:
                yahoo_interval = timeframe if timeframe != '1d' else '1d'

            # Fetch data
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date, interval=yahoo_interval)

            if df.empty:
                logger.warning(f"No data returned from Yahoo for {yahoo_symbol}")
                return None

            # Rename columns to match our format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]

            logger.info(f"Fetched {len(df)} records from Yahoo Finance")
            return df

        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed: {e}")
            return None

    async def _fetch_from_cache(self, cache_key: str) -> Optional[Tuple[pd.DataFrame, DataMetadata]]:
        """Fetch data from cache."""
        if not self.redis_client:
            return None

        try:
            cached_data = self.redis_client.get(cache_key)
            if not cached_data:
                return None

            # Parse cached data
            cache_data = json.loads(cached_data)
            metadata_dict = cache_data['metadata']
            df_json = cache_data['data']

            # Reconstruct DataFrame
            df = pd.read_json(df_json, orient='index')
            df.index = pd.to_datetime(df.index)

            # Reconstruct metadata
            metadata = DataMetadata.from_dict(metadata_dict)

            logger.info(f"Cache hit for key: {cache_key}")
            return df, metadata

        except Exception as e:
            logger.error(f"Cache fetch failed: {e}")
            return None

    async def _store_in_cache(self, cache_key: str, df: pd.DataFrame, metadata: DataMetadata):
        """Store data in cache."""
        if not self.redis_client:
            return

        try:
            # Prepare data for caching
            cache_data = {
                'metadata': metadata.to_dict(),
                'data': df.to_json(orient='index', date_format='iso')
            }

            # Store with TTL
            self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(cache_data))
            logger.info(f"Stored data in cache: {cache_key}")

        except Exception as e:
            logger.error(f"Cache store failed: {e}")

    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """
        Fetch time-series data with automatic fallbacks and caching.

        Args:
            request: DataRequest with fetch parameters

        Returns:
            DataResponse with data, metadata, and status
        """
        start_time = time.time()

        try:
            # Parse dates
            start_date = datetime.fromisoformat(request.start_date)
            end_date = datetime.fromisoformat(request.end_date)

            # Generate cache key
            cache_key = self._generate_cache_key(
                request.symbol, request.timeframe, start_date, end_date
            )

            # Try cache first if enabled
            if request.use_cache:
                cached_result = await self._fetch_from_cache(cache_key)
                if cached_result:
                    df, metadata = cached_result
                    processing_time = time.time() - start_time

                    await self.event_system.publish("data.cache_hit", {
                        "cache_key": cache_key,
                        "symbol": request.symbol,
                        "processing_time": processing_time
                    })

                    return DataResponse(
                        success=True,
                        data={"ohlcv": df.to_dict('index')},
                        metadata=metadata.to_dict(),
                        cache_hit=True,
                        processing_time=processing_time
                    )

            # Determine data sources to try
            sources_to_try = []
            if request.source:
                try:
                    preferred_source = DataSource(request.source)
                    sources_to_try.append((preferred_source, request.provider))
                except ValueError:
                    logger.warning(f"Invalid source: {request.source}")

            # Add fallback sources
            if DataSource.CCXT not in [s[0] for s in sources_to_try]:
                for provider in self.source_priorities[DataSource.CCXT]:
                    sources_to_try.append((DataSource.CCXT, provider))

            if DataSource.YAHOO_FINANCE not in [s[0] for s in sources_to_try]:
                sources_to_try.append((DataSource.YAHOO_FINANCE, DataProvider.YAHOO))

            # Try each source
            df = None
            source_used = None
            provider_used = None
            fallback_used = False

            for source, provider in sources_to_try:
                logger.info(f"Trying source: {source.value}, provider: {provider.value}")

                if source == DataSource.CCXT:
                    df = await self._fetch_from_ccxt(
                        request.symbol, request.timeframe, start_date, end_date, provider
                    )
                elif source == DataSource.YAHOO_FINANCE:
                    df = await self._fetch_from_yahoo(
                        request.symbol, request.timeframe, start_date, end_date
                    )

                if df is not None and not df.empty:
                    source_used = source
                    provider_used = provider
                    if source != DataSource(request.source if request.source else DataSource.CCXT.value):
                        fallback_used = True
                    break

            if df is None or df.empty:
                error_msg = f"No data found for {request.symbol} from any source"
                processing_time = time.time() - start_time

                await self.event_system.publish("data.fetch_failed", {
                    "symbol": request.symbol,
                    "timeframe": request.timeframe,
                    "error": error_msg,
                    "processing_time": processing_time
                })

                return DataResponse(
                    success=False,
                    error=error_msg,
                    processing_time=processing_time
                )

            # Validate and score data quality
            quality_score, quality_level = self._calculate_data_quality(df)

            if request.validate_quality and quality_level == DataQuality.INVALID:
                error_msg = f"Data quality validation failed: {quality_level.value}"
                processing_time = time.time() - start_time

                return DataResponse(
                    success=False,
                    error=error_msg,
                    processing_time=processing_time
                )

            # Create metadata
            metadata = DataMetadata(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=start_date,
                end_date=end_date,
                source=source_used,
                provider=provider_used,
                fetched_at=datetime.now(),
                quality_score=quality_score,
                quality_level=quality_level,
                record_count=len(df),
                columns=list(df.columns),
                cache_key=cache_key,
                checksum=self._calculate_checksum(df),
                fallback_used=fallback_used
            )

            # Cache the result if enabled
            if request.use_cache:
                await self._store_in_cache(cache_key, df, metadata)

            processing_time = time.time() - start_time

            await self.event_system.publish("data.fetch_success", {
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "record_count": len(df),
                "quality_level": quality_level.value,
                "fallback_used": fallback_used,
                "processing_time": processing_time
            })

            return DataResponse(
                success=True,
                data={"ohlcv": df.to_dict('index')},
                metadata=metadata.to_dict(),
                processing_time=processing_time
            )

        except Exception as e:
            error_msg = f"Data fetch failed: {str(e)}"
            processing_time = time.time() - start_time

            logger.error(error_msg)
            await self.event_system.publish("data.fetch_error", {
                "symbol": request.symbol,
                "error": error_msg,
                "processing_time": processing_time
            })

            return DataResponse(
                success=False,
                error=error_msg,
                processing_time=processing_time
            )

    async def get_data_info(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get information about available data for a symbol/timeframe."""
        # This would query cache/metadata store for available date ranges
        # For now, return basic info
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "available_sources": [s.value for s in DataSource],
            "cache_enabled": self.redis_client is not None,
            "quality_thresholds": {q.value: t for q, t in self.quality_thresholds.items()}
        }

    async def clear_cache(self, pattern: str = "*") -> int:
        """Clear cache entries matching pattern."""
        if not self.redis_client:
            return 0

        try:
            # Find keys matching pattern
            keys = self.redis_client.keys(f"data:{pattern}")
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return 0

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        health = {
            "service": "data_service",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }

        # Check Redis
        if REDIS_AVAILABLE:
            try:
                self.redis_client.ping()
                health["components"]["redis"] = "connected"
            except Exception as e:
                health["components"]["redis"] = f"error: {e}"
                health["status"] = "degraded"
        else:
            health["components"]["redis"] = "not_available"

        # Check data sources
        health["components"]["ccxt"] = "available" if CCXT_AVAILABLE else "not_available"
        health["components"]["yahoo_finance"] = "available" if YAHOO_AVAILABLE else "not_available"

        if not CCXT_AVAILABLE and not YAHOO_AVAILABLE:
            health["status"] = "unhealthy"

        return health