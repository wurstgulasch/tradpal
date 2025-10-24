"""
Data Mesh Module

Implements Data Mesh Architecture with:
- Time-Series Database (InfluxDB/TimescaleDB) for OHLCV data
- Data Lake (MinIO/S3) for historical data storage
- Feature Store for ML features
- Decentralized data governance
"""

from .data_mesh import (
    DataMeshManager,
    TimeSeriesDatabase,
    DataLake,
    FeatureStore,
    DataDomain,
    DataProduct,
    FeatureSet
)

__all__ = [
    'DataMeshManager',
    'TimeSeriesDatabase',
    'DataLake',
    'FeatureStore',
    'DataDomain',
    'DataProduct',
    'FeatureSet'
]