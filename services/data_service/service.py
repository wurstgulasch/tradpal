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

import logging
from .data_mesh import DataMeshManager, DataDomain, DataProduct, FeatureSet
from .data_governance import DataGovernanceManager, AccessLevel, AuditEventType

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

        # Initialize Data Mesh components
        self.data_mesh = DataMeshManager(
            redis_url=redis_url,
            event_system=self.event_system
        )

        # Initialize Data Governance components
        self.data_governance = DataGovernanceManager({
            "redis_url": redis_url,
            "max_recent_events": 1000,
            "max_results": 1000
        })

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

        logger.info("Data Service initialized with Data Mesh support")

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

    # Data Mesh Operations

    async def register_data_product(self, name: str, domain: str, description: str,
                                  schema: Dict[str, Any], owners: List[str]) -> Dict[str, Any]:
        """
        Register a new data product in the Data Mesh.

        Args:
            name: Data product name
            domain: Data domain (e.g., 'market_data', 'trading_signals')
            description: Product description
            schema: Data schema definition
            owners: List of data product owners

        Returns:
            Registration result
        """
        try:
            data_domain = DataDomain(domain)
            data_product = DataProduct(
                name=name,
                domain=data_domain,
                description=description,
                schema=schema,
                owners=owners,
                created_at=datetime.now(),
                version="1.0.0"
            )

            result = await self.data_mesh.register_data_product(data_product)

            await self.event_system.publish("data_mesh.product_registered", {
                "product_name": name,
                "domain": domain,
                "owners": owners
            })

            return {
                "success": True,
                "product_id": result,
                "message": f"Data product '{name}' registered successfully"
            }

        except Exception as e:
            error_msg = f"Data product registration failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    async def store_market_data(self, symbol: str, timeframe: str, df: pd.DataFrame,
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store market data in the Data Mesh (TimeSeriesDatabase).

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            df: OHLCV DataFrame
            metadata: Additional metadata

        Returns:
            Storage result
        """
        try:
            result = await self.data_mesh.store_market_data(symbol, timeframe, df, metadata or {})

            await self.event_system.publish("data_mesh.market_data_stored", {
                "symbol": symbol,
                "timeframe": timeframe,
                "record_count": len(df),
                "start_date": df.index.min().isoformat() if not df.empty else None,
                "end_date": df.index.max().isoformat() if not df.empty else None
            })

            return {
                "success": True,
                "message": f"Stored {len(df)} records for {symbol} {timeframe}",
                "details": result
            }

        except Exception as e:
            error_msg = f"Market data storage failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    async def retrieve_market_data(self, symbol: str, timeframe: str,
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Retrieve market data from the Data Mesh.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Retrieved data
        """
        try:
            df, metadata = await self.data_mesh.retrieve_market_data(
                symbol, timeframe, start_date, end_date
            )

            if df is None or df.empty:
                return {
                    "success": False,
                    "error": f"No data found for {symbol} {timeframe}"
                }

            return {
                "success": True,
                "data": {"ohlcv": df.to_dict('index')},
                "metadata": metadata,
                "record_count": len(df)
            }

        except Exception as e:
            error_msg = f"Market data retrieval failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    async def store_ml_features(self, feature_set_name: str, features_df: pd.DataFrame,
                               metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store ML features in the Feature Store.

        Args:
            feature_set_name: Name of the feature set
            features_df: DataFrame with features
            metadata: Feature set metadata

        Returns:
            Storage result
        """
        try:
            feature_set = FeatureSet(
                name=feature_set_name,
                features=list(features_df.columns),
                metadata=metadata,
                created_at=datetime.now(),
                version="1.0.0"
            )

            result = await self.data_mesh.store_ml_features(feature_set, features_df)

            await self.event_system.publish("data_mesh.features_stored", {
                "feature_set": feature_set_name,
                "feature_count": len(features_df.columns),
                "record_count": len(features_df)
            })

            return {
                "success": True,
                "message": f"Stored feature set '{feature_set_name}' with {len(features_df)} records",
                "details": result
            }

        except Exception as e:
            error_msg = f"ML features storage failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    async def retrieve_ml_features(self, feature_set_name: str,
                                  feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Retrieve ML features from the Feature Store.

        Args:
            feature_set_name: Name of the feature set
            feature_names: Specific features to retrieve (optional)

        Returns:
            Retrieved features
        """
        try:
            features_df, feature_set = await self.data_mesh.retrieve_ml_features(
                feature_set_name, feature_names
            )

            if features_df is None or features_df.empty:
                return {
                    "success": False,
                    "error": f"No features found for set '{feature_set_name}'"
                }

            return {
                "success": True,
                "data": {"features": features_df.to_dict('index')},
                "feature_set": feature_set.to_dict() if feature_set else None,
                "record_count": len(features_df)
            }

        except Exception as e:
            error_msg = f"ML features retrieval failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    async def archive_historical_data(self, symbol: str, timeframe: str,
                                     start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Archive historical data to Data Lake.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date for archival
            end_date: End date for archival

        Returns:
            Archival result
        """
        try:
            # First retrieve data from TimeSeriesDatabase
            df, _ = await self.data_mesh.retrieve_market_data(symbol, timeframe, start_date, end_date)

            if df is None or df.empty:
                return {
                    "success": False,
                    "error": f"No data available for archival: {symbol} {timeframe}"
                }

            # Archive to Data Lake
            result = await self.data_mesh.archive_to_data_lake(
                symbol, timeframe, df, start_date, end_date
            )

            await self.event_system.publish("data_mesh.data_archived", {
                "symbol": symbol,
                "timeframe": timeframe,
                "record_count": len(df),
                "date_range": f"{start_date.isoformat()} to {end_date.isoformat()}"
            })

            return {
                "success": True,
                "message": f"Archived {len(df)} records for {symbol} {timeframe}",
                "details": result
            }

        except Exception as e:
            error_msg = f"Data archival failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    async def get_data_mesh_status(self) -> Dict[str, Any]:
        """
        Get comprehensive Data Mesh status.

        Returns:
            Data Mesh health and statistics
        """
        try:
            status = await self.data_mesh.get_status()

            # Add service-specific information
            status.update({
                "service": "data_service",
                "data_mesh_enabled": True,
                "cache_enabled": self.redis_client is not None,
                "supported_domains": [d.value for d in DataDomain],
                "quality_thresholds": {q.value: t for q, t in self.quality_thresholds.items()}
            })

            return status

        except Exception as e:
            error_msg = f"Data Mesh status check failed: {str(e)}"
            logger.error(error_msg)
            return {
                "service": "data_service",
                "status": "error",
                "error": error_msg,
                "data_mesh_enabled": False
            }

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

        # Check Data Mesh components
        try:
            data_mesh_status = await self.get_data_mesh_status()
            health["components"]["data_mesh"] = data_mesh_status.get("status", "unknown")
            if data_mesh_status.get("status") != "healthy":
                health["status"] = "degraded"
        except Exception as e:
            health["components"]["data_mesh"] = f"error: {e}"
            health["status"] = "degraded"

        if not CCXT_AVAILABLE and not YAHOO_AVAILABLE:
            health["status"] = "unhealthy"

        return health

    # Governance Methods

    async def check_data_access(self, user: str, resource_type: str, resource_name: str,
                               access_level: str, purpose: str = "",
                               ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if user has access to a data resource with governance logging.

        Args:
            user: User identifier
            resource_type: Type of resource (data_product, domain, feature_set)
            resource_name: Name of the resource
            access_level: Required access level (read, write, admin)
            purpose: Purpose of access
            ip_address: Client IP address

        Returns:
            Access check result
        """
        try:
            access_level_enum = AccessLevel(access_level.lower())
            has_access, reason = await self.data_governance.check_access_and_log(
                user, resource_type, resource_name, access_level_enum, purpose, ip_address
            )

            return {
                "success": True,
                "has_access": has_access,
                "reason": reason,
                "user": user,
                "resource": f"{resource_type}:{resource_name}",
                "access_level": access_level
            }

        except Exception as e:
            error_msg = f"Access check failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "has_access": False
            }

    async def validate_and_store_data(self, user: str, resource_name: str, df: pd.DataFrame,
                                     resource_type: str = "data_product",
                                     ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate data quality and store with governance logging.

        Args:
            user: User performing the operation
            resource_name: Name of the data resource
            df: DataFrame to validate and store
            resource_type: Type of resource
            ip_address: Client IP address

        Returns:
            Validation and storage result
        """
        try:
            # Check access first
            has_access, reason = await self.data_governance.check_access_and_log(
                user, resource_type, resource_name, AccessLevel.WRITE,
                "data_validation_and_storage", ip_address
            )

            if not has_access:
                return {
                    "success": False,
                    "error": f"Access denied: {reason}",
                    "has_access": False
                }

            # Validate data quality
            quality_result = await self.data_governance.validate_data_and_log(
                user, resource_name, df, resource_type
            )

            # Store data in Data Mesh if quality is acceptable
            if quality_result.quality_level.value in ["excellent", "good", "fair"]:
                # Determine storage method based on resource type
                if resource_type == "data_product" and "market_data" in resource_name:
                    # Parse symbol and timeframe from resource name
                    parts = resource_name.split("_")
                    if len(parts) >= 3:
                        symbol = f"{parts[1]}/{parts[2]}"
                        timeframe = parts[3] if len(parts) > 3 else "1d"
                        storage_result = await self.store_market_data(symbol, timeframe, df)
                    else:
                        storage_result = {"success": False, "error": "Invalid market data resource name format"}
                elif resource_type == "feature_set":
                    storage_result = await self.store_ml_features(resource_name, df, {"validated_by": user})
                else:
                    storage_result = {"success": False, "error": f"Unsupported resource type for storage: {resource_type}"}
            else:
                storage_result = {
                    "success": False,
                    "error": f"Data quality too low: {quality_result.quality_level.value}"
                }

            return {
                "success": storage_result.get("success", False),
                "quality_check": {
                    "score": quality_result.quality_score,
                    "level": quality_result.quality_level.value,
                    "issues": quality_result.issues_found
                },
                "storage": storage_result if storage_result.get("success") else {"error": storage_result.get("error")},
                "user": user,
                "resource": f"{resource_type}:{resource_name}"
            }

        except Exception as e:
            error_msg = f"Data validation and storage failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    async def assign_user_role(self, admin_user: str, target_user: str, role: str,
                              ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Assign a role to a user (admin operation).

        Args:
            admin_user: User performing the assignment
            target_user: User to assign role to
            role: Role to assign
            ip_address: Client IP address

        Returns:
            Role assignment result
        """
        try:
            # Check if admin_user has admin privileges
            has_admin_access, _ = await self.data_governance.check_access_and_log(
                admin_user, "system", "user_management", AccessLevel.ADMIN,
                "role_assignment", ip_address
            )

            if not has_admin_access:
                return {
                    "success": False,
                    "error": "Insufficient privileges for role assignment"
                }

            success = await self.data_governance.assign_user_role(admin_user, target_user, role)

            return {
                "success": success,
                "message": f"Role '{role}' assigned to user '{target_user}'" if success else "Role assignment failed",
                "admin_user": admin_user,
                "target_user": target_user,
                "role": role
            }

        except Exception as e:
            error_msg = f"Role assignment failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": "An internal error has occurred. Please contact support."
            }

    async def create_governance_policy(self, user: str, policy_data: Dict[str, Any],
                                      ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new governance policy.

        Args:
            user: User creating the policy
            policy_data: Policy definition
            ip_address: Client IP address

        Returns:
            Policy creation result
        """
        try:
            from .data_governance import GovernancePolicy

            # Check admin access
            has_admin_access, _ = await self.data_governance.check_access_and_log(
                user, "system", "policy_management", AccessLevel.ADMIN,
                "policy_creation", ip_address
            )

            if not has_admin_access:
                return {
                    "success": False,
                    "error": "Insufficient privileges for policy creation"
                }

            policy = GovernancePolicy(**policy_data)
            success = await self.data_governance.create_governance_policy(policy)

            return {
                "success": success,
                "message": f"Governance policy '{policy.name}' created" if success else "Policy creation failed",
                "policy_name": policy.name,
                "user": user
            }

        except Exception as e:
            error_msg = f"Policy creation failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": "An internal error has occurred. Please contact support."
            }

    async def get_audit_events(self, user: str, filters: Dict[str, Any] = None,
                              ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve audit events with access control.

        Args:
            user: User requesting audit events
            filters: Event filters
            ip_address: Client IP address

        Returns:
            Audit events result
        """
        try:
            # Check access to audit logs
            has_access, _ = await self.data_governance.check_access_and_log(
                user, "system", "audit_logs", AccessLevel.READ,
                "audit_review", ip_address
            )

            if not has_access:
                return {
                    "success": False,
                    "error": "Insufficient privileges to access audit logs"
                }

            filters = filters or {}
            events = await self.data_governance.audit_logger.get_events(
                user=filters.get("user"),
                resource_type=filters.get("resource_type"),
                event_type=AuditEventType(filters.get("event_type")) if filters.get("event_type") else None,
                start_date=datetime.fromisoformat(filters["start_date"]) if filters.get("start_date") else None,
                end_date=datetime.fromisoformat(filters["end_date"]) if filters.get("end_date") else None,
                limit=filters.get("limit", 100)
            )

            return {
                "success": True,
                "events": [event.dict() for event in events],
                "count": len(events),
                "user": user
            }

        except Exception as e:
            error_msg = f"Audit events retrieval failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": "An internal error has occurred. Please contact support."
            }

    async def get_user_permissions(self, requesting_user: str, target_user: str,
                                  ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Get user permissions (admin or self only).

        Args:
            requesting_user: User making the request
            target_user: User whose permissions to retrieve
            ip_address: Client IP address

        Returns:
            User permissions result
        """
        try:
            # Check if user can view permissions (admin or self)
            is_admin, _ = await self.data_governance.check_access_and_log(
                requesting_user, "system", "user_management", AccessLevel.READ,
                "permission_review", ip_address
            )

            if requesting_user != target_user and not is_admin:
                return {
                    "success": False,
                    "error": "Insufficient privileges to view other users' permissions"
                }

            permissions = await self.data_governance.access_control.get_user_permissions(target_user)

            return {
                "success": True,
                "permissions": permissions,
                "requesting_user": requesting_user,
                "target_user": target_user
            }

        except Exception as e:
            error_msg = f"Permissions retrieval failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": "An internal error has occurred. Please contact support."
            }

    async def get_quality_report(self, user: str, resource_name: Optional[str] = None,
                                days: int = 7, ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Get data quality report.

        Args:
            user: User requesting the report
            resource_name: Specific resource to report on
            days: Number of days to include
            ip_address: Client IP address

        Returns:
            Quality report result
        """
        try:
            # Check access to quality reports
            has_access, _ = await self.data_governance.check_access_and_log(
                user, "system", "quality_reports", AccessLevel.READ,
                "quality_review", ip_address
            )

            if not has_access:
                return {
                    "success": False,
                    "error": "Insufficient privileges to access quality reports"
                }

            report = await self.data_governance.quality_monitor.get_quality_summary(
                resource_name, days
            )

            return {
                "success": True,
                "report": report,
                "user": user,
                "resource_filter": resource_name,
                "days": days
            }

        except Exception as e:
            error_msg = f"Quality report generation failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": "An internal error has occurred. Please contact support."
            }

    async def generate_compliance_report(self, user: str, start_date: str, end_date: str,
                                        ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate compliance report for a date range.

        Args:
            user: User requesting the report
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            ip_address: Client IP address

        Returns:
            Compliance report result
        """
        try:
            # Check admin access for compliance reports
            has_access, _ = await self.data_governance.check_access_and_log(
                user, "system", "compliance_reports", AccessLevel.ADMIN,
                "compliance_review", ip_address
            )

            if not has_access:
                return {
                    "success": False,
                    "error": "Insufficient privileges to generate compliance reports"
                }

            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)

            report = await self.data_governance.generate_compliance_report(start, end)

            return {
                "success": True,
                "report": report,
                "user": user,
                "date_range": f"{start_date} to {end_date}"
            }

        except Exception as e:
            error_msg = f"Compliance report generation failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": "An internal error has occurred. Please contact support."
            }

    async def get_governance_status(self) -> Dict[str, Any]:
        """
        Get comprehensive governance status.

        Returns:
            Governance system health and statistics
        """
        try:
            status = await self.data_governance.get_governance_status()

            # Add service-specific information
            status.update({
                "service": "data_service",
                "data_mesh_enabled": True,
                "governance_enabled": True,
                "supported_access_levels": [level.value for level in AccessLevel],
                "supported_audit_events": [event.value for event in AuditEventType]
            })

            return status

        except Exception as e:
            error_msg = f"Governance status check failed: {str(e)}"
            logger.error(error_msg)
            return {
                "service": "data_service",
                "status": "error",
                "error": "An internal error has occurred. Please contact support.",
                "governance_enabled": False
            }