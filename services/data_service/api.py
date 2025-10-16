#!/usr/bin/env python3
"""
Data Service API - REST API for the Data Service.

Provides endpoints for:
- Data fetching with caching and validation
- Data quality monitoring
- Cache management
- Health checks
"""

import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .service import DataService, DataRequest, DataResponse, EventSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
data_service: Optional[DataService] = None
event_system = EventSystem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global data_service

    # Startup
    logger.info("Starting Data Service API")
    data_service = DataService(event_system=event_system)

    yield

    # Shutdown
    logger.info("Shutting down Data Service API")


# Create FastAPI app
app = FastAPI(
    title="Data Service API",
    description="Centralized time-series data management for TradPal",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FetchDataRequest(BaseModel):
    """Request model for data fetching."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe (e.g., '1d', '1h', '15m')")
    start_date: str = Field(..., description="Start date in ISO format")
    end_date: str = Field(..., description="End date in ISO format")
    source: Optional[str] = Field("ccxt", description="Preferred data source")
    provider: Optional[str] = Field("binance", description="Preferred data provider")
    use_cache: bool = Field(True, description="Whether to use cached data")
    validate_quality: bool = Field(True, description="Whether to validate data quality")


class DataProductRegistration(BaseModel):
    """Request model for data product registration."""
    name: str = Field(..., description="Data product name")
    domain: str = Field(..., description="Data domain (market_data, trading_signals, etc.)")
    description: str = Field(..., description="Product description")
    schema: Dict[str, Any] = Field(..., description="Data schema definition")
    owners: list = Field(..., description="List of data product owners")


class MarketDataStorage(BaseModel):
    """Request model for market data storage."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    data: Dict[str, Any] = Field(..., description="OHLCV data as dict")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Additional metadata")


class FeatureStorage(BaseModel):
    """Request model for ML feature storage."""
    feature_set_name: str = Field(..., description="Name of the feature set")
    features: Dict[str, Any] = Field(..., description="Features data as dict")
    metadata: Dict[str, Any] = Field(..., description="Feature set metadata")


class ArchivalRequest(BaseModel):
    """Request model for data archival."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    start_date: str = Field(..., description="Start date in ISO format")
    end_date: str = Field(..., description="End date in ISO format")


class AccessCheckRequest(BaseModel):
    """Request model for access checking."""
    user: str = Field(..., description="User identifier")
    resource_type: str = Field(..., description="Type of resource (data_product, domain, feature_set)")
    resource_name: str = Field(..., description="Name of the resource")
    access_level: str = Field(..., description="Required access level (read, write, admin)")
    purpose: Optional[str] = Field("", description="Purpose of access")


class DataValidationRequest(BaseModel):
    """Request model for data validation and storage."""
    user: str = Field(..., description="User performing the operation")
    resource_name: str = Field(..., description="Name of the data resource")
    data: Dict[str, Any] = Field(..., description="Data to validate as dict")
    resource_type: Optional[str] = Field("data_product", description="Type of resource")


class RoleAssignmentRequest(BaseModel):
    """Request model for role assignment."""
    admin_user: str = Field(..., description="User performing the assignment")
    target_user: str = Field(..., description="User to assign role to")
    role: str = Field(..., description="Role to assign")


class PolicyCreationRequest(BaseModel):
    """Request model for policy creation."""
    user: str = Field(..., description="User creating the policy")
    policy: Dict[str, Any] = Field(..., description="Policy definition")


class AuditQueryRequest(BaseModel):
    """Request model for audit queries."""
    user: str = Field(..., description="User requesting audit events")
    filters: Optional[Dict[str, Any]] = Field({}, description="Event filters")


class PermissionsQueryRequest(BaseModel):
    """Request model for permissions queries."""
    requesting_user: str = Field(..., description="User making the request")
    target_user: str = Field(..., description="User whose permissions to retrieve")


class QualityReportRequest(BaseModel):
    """Request model for quality reports."""
    user: str = Field(..., description="User requesting the report")
    resource_name: Optional[str] = Field(None, description="Specific resource to report on")
    days: Optional[int] = Field(7, description="Number of days to include")


class ComplianceReportRequest(BaseModel):
    """Request model for compliance reports."""
    user: str = Field(..., description="User requesting the report")
    start_date: str = Field(..., description="Start date in ISO format")
    end_date: str = Field(..., description="End date in ISO format")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Data Service",
        "version": "1.0.0",
        "description": "Centralized time-series data management"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    health_data = await data_service.health_check()
    status_code = 200 if health_data["status"] == "healthy" else 503

    return health_data


@app.post("/data/fetch", response_model=DataResponse)
async def fetch_data(request: FetchDataRequest, background_tasks: BackgroundTasks):
    """
    Fetch time-series data.

    Fetches OHLCV data from various sources with automatic fallbacks,
    caching, and quality validation.
    """
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert to service request
        service_request = DataRequest(**request.dict())

        # Fetch data
        response = await data_service.fetch_data(service_request)

        # Log the request
        logger.info(f"Data fetch request: {request.symbol} {request.timeframe} "
                   f"({request.start_date} to {request.end_date}) - "
                   f"Success: {response.success}, Cache hit: {response.cache_hit}")

        if not response.success:
            raise HTTPException(status_code=400, detail=response.error)

        return response

    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/info", response_model=DataInfoResponse)
async def get_data_info(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe")
):
    """Get information about available data for a symbol/timeframe."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        info = await data_service.get_data_info(symbol, timeframe)
        return DataInfoResponse(**info)
    except Exception as e:
        logger.error(f"Data info request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
async def clear_cache(pattern: str = Query("*", description="Cache key pattern to clear")):
    """Clear cache entries matching the pattern."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        deleted_count = await data_service.clear_cache(pattern)
        return {
            "message": f"Cleared {deleted_count} cache entries",
            "pattern": pattern
        }
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    if not data_service or not data_service.redis_client:
        return {"cache_enabled": False, "stats": {}}

    try:
        # Get Redis info
        info = data_service.redis_client.info()
        keys = data_service.redis_client.keys("data:*")

        return {
            "cache_enabled": True,
            "total_keys": len(keys) if keys else 0,
            "redis_info": {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "uptime_in_days": info.get("uptime_in_days", 0)
            }
        }
    except Exception as e:
        logger.error(f"Cache stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/validate")
async def validate_data_quality(request: FetchDataRequest):
    """
    Validate data quality without storing results.

    Useful for checking data quality before using it in analysis.
    """
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert to service request
        service_request = DataRequest(**request.dict())

        # Fetch data
        response = await data_service.fetch_data(service_request)

        if not response.success:
            return {
                "valid": False,
                "error": response.error,
                "quality_score": 0.0,
                "quality_level": "invalid"
            }

        # Extract quality info from metadata
        metadata = response.metadata
        if metadata:
            return {
                "valid": True,
                "quality_score": metadata.get("quality_score", 0.0),
                "quality_level": metadata.get("quality_level", "unknown"),
                "record_count": metadata.get("record_count", 0),
                "fallback_used": metadata.get("fallback_used", False)
            }

        return {
            "valid": False,
            "error": "No metadata available",
            "quality_score": 0.0,
            "quality_level": "unknown"
        }

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Event handlers for logging
async def handle_data_events(event_data: Dict[str, Any]):
    """Handle data service events."""
    event_type = asyncio.current_task().get_name() if asyncio.current_task() else "unknown"

    if "data.fetch_success" in str(event_type):
        logger.info(f"Data fetch success: {event_data}")
    elif "data.fetch_failed" in str(event_type):
        logger.warning(f"Data fetch failed: {event_data}")
    elif "data.cache_hit" in str(event_type):
        logger.info(f"Cache hit: {event_data}")

# Subscribe to events
event_system.subscribe("data.fetch_success", handle_data_events)
event_system.subscribe("data.fetch_failed", handle_data_events)
event_system.subscribe("data.cache_hit", handle_data_events)


# Data Mesh Endpoints

@app.post("/data-mesh/products/register")
async def register_data_product(request: DataProductRegistration):
    """Register a new data product in the Data Mesh."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await data_service.register_data_product(
            name=request.name,
            domain=request.domain,
            description=request.description,
            schema=request.data_schema,
            owners=request.owners
        )

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Data product registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data-mesh/market-data/store")
async def store_market_data(request: MarketDataStorage):
    """Store market data in the Data Mesh."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert dict data back to DataFrame
        import pandas as pd
        df = pd.DataFrame.from_dict(request.data, orient='index')
        df.index = pd.to_datetime(df.index)

        result = await data_service.store_market_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            df=df,
            metadata=request.metadata
        )

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Market data storage failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data-mesh/market-data/retrieve")
async def retrieve_market_data(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe"),
    start_date: Optional[str] = Query(None, description="Start date in ISO format"),
    end_date: Optional[str] = Query(None, description="End date in ISO format")
):
    """Retrieve market data from the Data Mesh."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        from datetime import datetime
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None

        result = await data_service.retrieve_market_data(symbol, timeframe, start, end)

        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Market data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data-mesh/features/store")
async def store_ml_features(request: FeatureStorage):
    """Store ML features in the Feature Store."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert dict data back to DataFrame
        import pandas as pd
        features_df = pd.DataFrame.from_dict(request.features, orient='index')
        features_df.index = pd.to_datetime(features_df.index)

        result = await data_service.store_ml_features(
            feature_set_name=request.feature_set_name,
            features_df=features_df,
            metadata=request.metadata
        )

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"ML features storage failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data-mesh/features/retrieve")
async def retrieve_ml_features(
    feature_set_name: str = Query(..., description="Name of the feature set"),
    feature_names: Optional[str] = Query(None, description="Comma-separated feature names")
):
    """Retrieve ML features from the Feature Store."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        features_list = feature_names.split(',') if feature_names else None

        result = await data_service.retrieve_ml_features(feature_set_name, features_list)

        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"ML features retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data-mesh/archive")
async def archive_historical_data(request: ArchivalRequest):
    """Archive historical data to Data Lake."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        from datetime import datetime
        start = datetime.fromisoformat(request.start_date)
        end = datetime.fromisoformat(request.end_date)

        result = await data_service.archive_historical_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=start,
            end_date=end
        )

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Data archival failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data-mesh/status")
async def get_data_mesh_status():
    """Get comprehensive Data Mesh status."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        status = await data_service.get_data_mesh_status()
        return status
    except Exception as e:
        logger.error(f"Data Mesh status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Governance API Endpoints

@app.post("/governance/access/check")
async def check_data_access(request: AccessCheckRequest):
    """Check if user has access to a data resource."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await data_service.check_data_access(
            user=request.user,
            resource_type=request.resource_type,
            resource_name=request.resource_name,
            access_level=request.access_level,
            purpose=request.purpose
        )

        if not result["success"]:
            raise HTTPException(status_code=403, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Access check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/data/validate")
async def validate_and_store_data(request: DataValidationRequest):
    """Validate data quality and store with governance."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        import pandas as pd
        df = pd.DataFrame.from_dict(request.data, orient='index')

        result = await data_service.validate_and_store_data(
            user=request.user,
            resource_name=request.resource_name,
            df=df,
            resource_type=request.resource_type
        )

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Validation failed"))

        return result

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/users/assign-role")
async def assign_user_role(request: RoleAssignmentRequest):
    """Assign a role to a user."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await data_service.assign_user_role(
            admin_user=request.admin_user,
            target_user=request.target_user,
            role=request.role
        )

        if not result["success"]:
            raise HTTPException(status_code=403, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Role assignment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/policies/create")
async def create_governance_policy(request: PolicyCreationRequest):
    """Create a new governance policy."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await data_service.create_governance_policy(
            user=request.user,
            policy_data=request.policy
        )

        if not result["success"]:
            raise HTTPException(status_code=403, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Policy creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/audit/events")
async def get_audit_events(request: AuditQueryRequest):
    """Retrieve audit events."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await data_service.get_audit_events(
            user=request.user,
            filters=request.filters
        )

        if not result["success"]:
            raise HTTPException(status_code=403, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Audit events retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/users/permissions")
async def get_user_permissions(request: PermissionsQueryRequest):
    """Get user permissions."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await data_service.get_user_permissions(
            requesting_user=request.requesting_user,
            target_user=request.target_user
        )

        if not result["success"]:
            raise HTTPException(status_code=403, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Permissions retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/quality/report")
async def get_quality_report(request: QualityReportRequest):
    """Get data quality report."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await data_service.get_quality_report(
            user=request.user,
            resource_name=request.resource_name,
            days=request.days
        )

        if not result["success"]:
            raise HTTPException(status_code=403, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Quality report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/compliance/report")
async def generate_compliance_report(request: ComplianceReportRequest):
    """Generate compliance report."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await data_service.generate_compliance_report(
            user=request.user,
            start_date=request.start_date,
            end_date=request.end_date
        )

        if not result["success"]:
            raise HTTPException(status_code=403, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Compliance report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/governance/status")
async def get_governance_status():
    """Get comprehensive governance status."""
    if not data_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        status = await data_service.get_governance_status()
        return status
    except Exception as e:
        logger.error(f"Governance status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )