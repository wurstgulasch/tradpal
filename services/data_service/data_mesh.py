#!/usr/bin/env python3
"""
Data Mesh Components for TradPal

Implements Data Mesh Architecture with:
- Time-Series Database (InfluxDB/TimescaleDB) for OHLCV data
- Data Lake (MinIO/S3) for historical data storage
- Feature Store for ML features
- Decentralized data governance
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from enum import Enum
import os

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

# Optional imports for Data Mesh components
try:
    import influxdb_client
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    influxdb_client = None

try:
    import boto3
    from botocore.exceptions import NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    boto3 = None

try:
    import psycopg2
    from psycopg2.extras import execute_values
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class DataDomain(Enum):
    """Data Mesh domains."""
    MARKET_DATA = "market_data"
    TRADING_SIGNALS = "trading_signals"
    PORTFOLIO_DATA = "portfolio_data"
    ML_FEATURES = "ml_features"
    RISK_METRICS = "risk_metrics"


class DataProduct(BaseModel):
    """Data product definition for Data Mesh."""
    name: str
    domain: DataDomain
    version: str
    description: str
    data_schema: Dict[str, Any] = Field(alias="schema")
    owners: List[str]
    tags: List[str] = []
    retention_policy: str = "1y"  # ISO 8601 duration
    update_frequency: str = "1h"  # ISO 8601 duration
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureSet(BaseModel):
    """Feature set for ML feature store."""
    name: str
    version: str
    description: str
    features: List[str]
    entity: str  # e.g., "symbol", "portfolio"
    feature_schema: Dict[str, Any] = Field(alias="schema")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TimeSeriesDatabase:
    """Time-Series Database interface (InfluxDB/TimescaleDB)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.write_api = None
        self.query_api = None

        if INFLUXDB_AVAILABLE:
            try:
                self.client = influxdb_client.InfluxDBClient(
                    url=config.get("url", "http://localhost:8086"),
                    token=config.get("token", ""),
                    org=config.get("org", "tradpal")
                )
                self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
                self.query_api = self.client.query_api()
                logger.info("InfluxDB client initialized")
            except Exception as e:
                logger.error(f"InfluxDB initialization failed: {e}")

    async def write_ohlcv_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """Write OHLCV data to time-series database."""
        if not self.client or not INFLUXDB_AVAILABLE:
            logger.warning("InfluxDB not available")
            return False

        try:
            from influxdb_client import Point

            bucket = self.config.get("bucket", "tradpal-market-data")

            points = []
            for timestamp, row in data.iterrows():
                point = Point("ohlcv") \
                    .tag("symbol", symbol) \
                    .tag("timeframe", timeframe) \
                    .field("open", float(row['open'])) \
                    .field("high", float(row['high'])) \
                    .field("low", float(row['low'])) \
                    .field("close", float(row['close'])) \
                    .field("volume", float(row.get('volume', 0))) \
                    .time(timestamp)

                points.append(point)

            self.write_api.write(bucket=bucket, record=points)
            logger.info(f"Written {len(points)} OHLCV records for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to write OHLCV data: {e}")
            return False

    async def query_ohlcv_data(self, symbol: str, timeframe: str,
                              start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Query OHLCV data from time-series database."""
        if not self.client or not INFLUXDB_AVAILABLE:
            return None

        try:
            bucket = self.config.get("bucket", "tradpal-market-data")

            query = f'''
            from(bucket: "{bucket}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r["_measurement"] == "ohlcv")
                |> filter(fn: (r) => r["symbol"] == "{symbol}")
                |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''

            result = self.query_api.query(query)
            records = []

            for table in result:
                for record in table.records:
                    records.append({
                        'timestamp': record.get_time(),
                        'open': record.values.get('open'),
                        'high': record.values.get('high'),
                        'low': record.values.get('low'),
                        'close': record.values.get('close'),
                        'volume': record.values.get('volume', 0)
                    })

            if records:
                df = pd.DataFrame(records)
                df.set_index('timestamp', inplace=True)
                return df

            return None

        except Exception as e:
            logger.error(f"Failed to query OHLCV data: {e}")
            return None


class DataLake:
    """Data Lake interface (MinIO/S3)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None

        if S3_AVAILABLE:
            try:
                self.client = boto3.client(
                    's3',
                    endpoint_url=config.get("endpoint_url"),
                    aws_access_key_id=config.get("access_key"),
                    aws_secret_access_key=config.get("secret_key"),
                    region_name=config.get("region", "us-east-1")
                )
                logger.info("S3/MinIO client initialized")
            except Exception as e:
                logger.error(f"S3/MinIO initialization failed: {e}")

    async def store_data(self, bucket: str, key: str, data: Union[str, bytes, pd.DataFrame]) -> bool:
        """Store data in data lake."""
        if not self.client or not S3_AVAILABLE:
            logger.warning("S3/MinIO not available")
            return False

        try:
            # Convert data to bytes
            if isinstance(data, pd.DataFrame):
                data_bytes = data.to_parquet()
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_bytes = json.dumps(data).encode('utf-8')

            self.client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data_bytes
            )

            logger.info(f"Stored data in data lake: {bucket}/{key}")
            return True

        except Exception as e:
            logger.error(f"Failed to store data in data lake: {e}")
            return False

    async def retrieve_data(self, bucket: str, key: str) -> Optional[bytes]:
        """Retrieve data from data lake."""
        if not self.client or not S3_AVAILABLE:
            return None

        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            data = response['Body'].read()
            return data

        except Exception as e:
            logger.error(f"Failed to retrieve data from data lake: {e}")
            return None

    async def list_objects(self, bucket: str, prefix: str = "") -> List[str]:
        """List objects in data lake bucket."""
        if not self.client or not S3_AVAILABLE:
            return []

        try:
            response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            objects = response.get('Contents', [])
            return [obj['Key'] for obj in objects]

        except Exception as e:
            logger.error(f"Failed to list objects in data lake: {e}")
            return []


class FeatureStore:
    """Feature Store for ML features."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(config.get("redis_url", "redis://localhost:6379"))
                logger.info("Feature Store Redis client initialized")
            except Exception as e:
                logger.error(f"Feature Store Redis initialization failed: {e}")

    async def store_feature_set(self, feature_set: FeatureSet, features_data: Dict[str, Any]) -> bool:
        """Store feature set in feature store."""
        if not self.redis_client or not REDIS_AVAILABLE:
            logger.warning("Redis not available for feature store")
            return False

        try:
            key = f"features:{feature_set.name}:{feature_set.version}"

            # Store feature set metadata
            metadata_key = f"{key}:metadata"
            self.redis_client.set(metadata_key, feature_set.json())

            # Store features data
            data_key = f"{key}:data"
            self.redis_client.set(data_key, json.dumps(features_data))

            logger.info(f"Stored feature set: {feature_set.name} v{feature_set.version}")
            return True

        except Exception as e:
            logger.error(f"Failed to store feature set: {e}")
            return False

    async def retrieve_feature_set(self, name: str, version: str) -> Optional[Tuple[FeatureSet, Dict[str, Any]]]:
        """Retrieve feature set from feature store."""
        if not self.redis_client or not REDIS_AVAILABLE:
            return None

        try:
            key = f"features:{name}:{version}"

            # Retrieve metadata
            metadata_key = f"{key}:metadata"
            metadata_json = self.redis_client.get(metadata_key)
            if not metadata_json:
                return None

            feature_set = FeatureSet.parse_raw(metadata_json)

            # Retrieve data
            data_key = f"{key}:data"
            data_json = self.redis_client.get(data_key)
            if not data_json:
                return None

            features_data = json.loads(data_json)

            return feature_set, features_data

        except Exception as e:
            logger.error(f"Failed to retrieve feature set: {e}")
            return None

    async def list_feature_sets(self, pattern: str = "*") -> List[str]:
        """List available feature sets."""
        if not self.redis_client or not REDIS_AVAILABLE:
            return []

        try:
            keys = self.redis_client.keys(f"features:{pattern}:metadata")
            feature_sets = []

            for key in keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                # Extract name and version from key
                parts = key_str.split(':')
                if len(parts) >= 3:
                    name = parts[1]
                    version = parts[2]
                    feature_sets.append(f"{name}:{version}")

            return feature_sets

        except Exception as e:
            logger.error(f"Failed to list feature sets: {e}")
            return []


class DataMeshManager:
    """
    Data Mesh Manager coordinating all data mesh components.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize components
        self.tsdb = TimeSeriesDatabase(config.get("timeseries_db", {}))
        self.data_lake = DataLake(config.get("data_lake", {}))
        self.feature_store = FeatureStore(config.get("feature_store", {}))

        # Data products registry
        self.data_products: Dict[str, DataProduct] = {}

        # Domain ownership
        self.domain_owners = {
            DataDomain.MARKET_DATA: ["data_service"],
            DataDomain.TRADING_SIGNALS: ["core_service"],
            DataDomain.PORTFOLIO_DATA: ["trading_bot_live"],
            DataDomain.ML_FEATURES: ["ml_trainer"],
            DataDomain.RISK_METRICS: ["risk_service"]
        }

        logger.info("Data Mesh Manager initialized")

    async def register_data_product(self, product: DataProduct) -> bool:
        """Register a data product in the data mesh."""
        try:
            self.data_products[product.name] = product

            # Store in data lake for persistence
            await self.data_lake.store_data(
                bucket="data-products",
                key=f"{product.name}/{product.version}/metadata.json",
                data=product.json()
            )

            logger.info(f"Registered data product: {product.name} v{product.version}")
            return True

        except Exception as e:
            logger.error(f"Failed to register data product: {e}")
            return False

    async def get_data_product(self, name: str, version: str = "latest") -> Optional[DataProduct]:
        """Get data product from registry."""
        if version == "latest":
            # Find latest version
            candidates = [p for p in self.data_products.values() if p.name == name]
            if candidates:
                return max(candidates, key=lambda p: p.created_at)

        return self.data_products.get(f"{name}:{version}")

    async def store_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """Store market data in data mesh (TSDB + Data Lake)."""
        try:
            # Store in time-series database
            tsdb_success = await self.tsdb.write_ohlcv_data(symbol, timeframe, data)

            # Store in data lake for long-term retention
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            key = f"market-data/{symbol}/{timeframe}/{timestamp}.parquet"
            lake_success = await self.data_lake.store_data("historical-data", key, data)

            success = tsdb_success or lake_success  # At least one should succeed

            if success:
                logger.info(f"Stored market data for {symbol} {timeframe}: TSDB={tsdb_success}, Lake={lake_success}")

            return success

        except Exception as e:
            logger.error(f"Failed to store market data: {e}")
            return False

    async def retrieve_market_data(self, symbol: str, timeframe: str,
                                 start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Retrieve market data from data mesh."""
        try:
            # Try time-series database first (faster for recent data)
            data = await self.tsdb.query_ohlcv_data(symbol, timeframe, start_time, end_time)

            if data is not None:
                logger.info(f"Retrieved market data from TSDB: {len(data)} records")
                return data

            # Fallback to data lake for historical data
            # This would require more complex logic to find the right files
            logger.info("No data found in TSDB, data lake fallback not implemented yet")

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve market data: {e}")
            return None

    async def store_ml_features(self, feature_set: FeatureSet, features_data: Dict[str, Any]) -> bool:
        """Store ML features in feature store."""
        try:
            success = await self.feature_store.store_feature_set(feature_set, features_data)

            if success:
                # Also store in data lake for backup
                await self.data_lake.store_data(
                    bucket="ml-features",
                    key=f"{feature_set.name}/{feature_set.version}/features.json",
                    data=features_data
                )

            return success

        except Exception as e:
            logger.error(f"Failed to store ML features: {e}")
            return False

    async def retrieve_ml_features(self, name: str, version: str) -> Optional[Tuple[FeatureSet, Dict[str, Any]]]:
        """Retrieve ML features from feature store."""
        try:
            return await self.feature_store.retrieve_feature_set(name, version)

        except Exception as e:
            logger.error(f"Failed to retrieve ML features: {e}")
            return None

    async def create_data_product(self, name: str, domain: DataDomain,
                                description: str, schema: Dict[str, Any],
                                owners: List[str]) -> DataProduct:
        """Create a new data product."""
        # Generate version (simple increment)
        existing_versions = [p.version for p in self.data_products.values() if p.name == name]
        version = f"v{len(existing_versions) + 1}"

        product = DataProduct(
            name=name,
            domain=domain,
            version=version,
            description=description,
            data_schema=schema,
            owners=owners
        )

        await self.register_data_product(product)
        return product

    async def list_data_products(self, domain: Optional[DataDomain] = None) -> List[DataProduct]:
        """List available data products."""
        products = list(self.data_products.values())

        if domain:
            products = [p for p in products if p.domain == domain]

        return products

    async def validate_data_access(self, user: str, data_product: str, action: str) -> bool:
        """Validate if user has access to data product."""
        # Simple access control - in production this would be more sophisticated
        product = self.data_products.get(data_product)

        if not product:
            return False

        # Check if user is owner or in allowed domains
        if user in product.owners:
            return True

        # Domain-based access
        user_domains = []
        for domain, owners in self.domain_owners.items():
            if user in owners:
                user_domains.append(domain)

        return product.domain in user_domains

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of data mesh components."""
        health = {
            "service": "data_mesh",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "timeseries_db": "available" if self.tsdb.client else "not_available",
                "data_lake": "available" if self.data_lake.client else "not_available",
                "feature_store": "available" if self.feature_store.redis_client else "not_available"
            },
            "data_products": len(self.data_products)
        }

        # Check if all critical components are available
        critical_components = ["timeseries_db", "data_lake"]
        for component in critical_components:
            if health["components"][component] == "not_available":
                health["status"] = "degraded"
                break

        return health

    async def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive Data Mesh status.

        Returns:
            Data Mesh health and statistics
        """
        try:
            status = await self.health_check()

            # Add additional statistics
            status.update({
                "data_products_count": len(self.data_products),
                "domain_owners": self.domain_owners,
                "supported_domains": [d.value for d in DataDomain]
            })

            return status

        except Exception as e:
            logger.error(f"Data Mesh status check failed: {e}")
            return {
                "service": "data_mesh",
                "status": "error",
                "error": str(e),
                "data_products_count": 0
            }