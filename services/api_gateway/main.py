#!/usr/bin/env python3
"""
TradPal API Gateway

Central entry point for all TradPal services providing:
- Service discovery and routing
- Load balancing
- Authentication and authorization
- Rate limiting
- Request/response transformation
- Centralized monitoring and logging
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import os

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import aiohttp
import jwt
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import uvicorn
from starlette.responses import Response as StarletteResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TradPal API Gateway",
    description="Central API Gateway for TradPal microservices",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


@dataclass
class ServiceInstance:
    """Represents a service instance"""
    url: str
    status: ServiceStatus = ServiceStatus.HEALTHY
    last_health_check: float = field(default_factory=time.time)
    request_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0


@dataclass
class ServiceConfig:
    """Configuration for a service"""
    name: str
    path_prefix: str
    instances: List[ServiceInstance] = field(default_factory=list)
    load_balancing: str = "round_robin"  # round_robin, least_connections, weighted
    rate_limit: Optional[int] = None  # requests per minute
    auth_required: bool = True
    timeout: float = 30.0


class ServiceRegistry:
    """Registry for managing service instances"""

    def __init__(self):
        self.services: Dict[str, ServiceConfig] = {}
        self._current_index: Dict[str, int] = {}

    def register_service(self, config: ServiceConfig):
        """Register a service"""
        self.services[config.name] = config
        self._current_index[config.name] = 0
        logger.info(f"Registered service: {config.name} with {len(config.instances)} instances")

    def unregister_service(self, service_name: str):
        """Unregister a service"""
        if service_name in self.services:
            del self.services[service_name]
            del self._current_index[service_name]
            logger.info(f"Unregistered service: {service_name}")

    def get_service_instance(self, service_name: str) -> Optional[ServiceInstance]:
        """Get next service instance using load balancing"""
        if service_name not in self.services:
            return None

        service = self.services[service_name]
        healthy_instances = [
            instance for instance in service.instances
            if instance.status == ServiceStatus.HEALTHY
        ]

        if not healthy_instances:
            return None

        # Simple round-robin load balancing
        current_index = self._current_index[service_name]
        instance = healthy_instances[current_index % len(healthy_instances)]
        self._current_index[service_name] = (current_index + 1) % len(healthy_instances)

        return instance

    def update_service_health(self, service_name: str, instance_url: str, status: ServiceStatus):
        """Update health status of a service instance"""
        if service_name in self.services:
            for instance in self.services[service_name].instances:
                if instance.url == instance_url:
                    instance.status = status
                    instance.last_health_check = time.time()
                    logger.info(f"Updated {service_name} instance {instance_url} to {status.value}")
                    break

    async def health_check_services(self):
        """Perform health checks on all service instances"""
        for service_name, service in self.services.items():
            for instance in service.instances:
                try:
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                        async with session.get(f"{instance.url}/health") as response:
                            if response.status == 200:
                                self.update_service_health(service_name, instance.url, ServiceStatus.HEALTHY)
                            else:
                                self.update_service_health(service_name, instance.url, ServiceStatus.UNHEALTHY)
                except Exception as e:
                    logger.warning(f"Health check failed for {service_name} {instance.url}: {e}")
                    self.update_service_health(service_name, instance.url, ServiceStatus.UNHEALTHY)


class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self):
        self.requests: Dict[str, List[float]] = {}

    def is_allowed(self, client_id: str, limit: int, window: int = 60) -> bool:
        """Check if request is allowed under rate limit"""
        now = time.time()
        client_requests = self.requests.get(client_id, [])

        # Remove old requests outside the window
        client_requests = [req_time for req_time in client_requests if now - req_time < window]

        if len(client_requests) >= limit:
            return False

        client_requests.append(now)
        self.requests[client_id] = client_requests
        return True


class AuthMiddleware:
    """Authentication and authorization middleware"""

    def __init__(self, jwt_secret: str = "tradpal-secret-key"):
        self.jwt_secret = jwt_secret

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def create_token(self, user_id: str, roles: List[str] = None) -> str:
        """Create JWT token"""
        payload = {
            "user_id": user_id,
            "roles": roles or ["user"],
            "exp": int(time.time()) + 3600,  # 1 hour
            "iat": int(time.time())
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")


# Global instances
service_registry = ServiceRegistry()
rate_limiter = RateLimiter()
auth_middleware = AuthMiddleware()

# Prometheus metrics
REQUEST_COUNT = Counter(
    'api_gateway_requests_total',
    'Total number of requests processed by the API Gateway',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_gateway_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'api_gateway_active_connections',
    'Number of active connections to the API Gateway'
)

SERVICE_HEALTH_STATUS = Gauge(
    'api_gateway_service_health_status',
    'Health status of backend services (1=healthy, 0=unhealthy)',
    ['service_name']
)

RATE_LIMIT_EXCEEDED = Counter(
    'api_gateway_rate_limit_exceeded_total',
    'Total number of rate limit violations',
    ['client_id']
)

AUTH_FAILURES = Counter(
    'api_gateway_auth_failures_total',
    'Total number of authentication failures'
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await initialize_services()
    # Start background health checks
    asyncio.create_task(health_check_loop())


async def initialize_services():
    """Initialize service configurations"""
    # Core Service
    core_service = ServiceConfig(
        name="core_service",
        path_prefix="/api/core",
        instances=[
            ServiceInstance(url="http://core_service:8002"),
        ],
        rate_limit=1000,
        auth_required=True
    )
    service_registry.register_service(core_service)

    # Data Service
    data_service = ServiceConfig(
        name="data_service",
        path_prefix="/api/data",
        instances=[
            ServiceInstance(url="http://data_service:8001"),
        ],
        rate_limit=500,
        auth_required=True
    )
    service_registry.register_service(data_service)

    # Trading Bot Live
    trading_service = ServiceConfig(
        name="trading_bot_live",
        path_prefix="/api/trading",
        instances=[
            ServiceInstance(url="http://trading_bot_live:8003"),
        ],
        rate_limit=100,
        auth_required=True
    )
    service_registry.register_service(trading_service)

    # Backtesting Service
    backtesting_service = ServiceConfig(
        name="backtesting_service",
        path_prefix="/api/backtest",
        instances=[
            ServiceInstance(url="http://backtesting_service:8004"),
        ],
        rate_limit=50,
        auth_required=True
    )
    service_registry.register_service(backtesting_service)

    # Discovery Service
    discovery_service = ServiceConfig(
        name="discovery_service",
        path_prefix="/api/discovery",
        instances=[
            ServiceInstance(url="http://discovery_service:8005"),
        ],
        rate_limit=200,
        auth_required=True
    )
    service_registry.register_service(discovery_service)

    # Notification Service
    notification_service = ServiceConfig(
        name="notification_service",
        path_prefix="/api/notifications",
        instances=[
            ServiceInstance(url="http://notification_service:8006"),
        ],
        rate_limit=300,
        auth_required=True
    )
    service_registry.register_service(notification_service)

    # Risk Service
    risk_service = ServiceConfig(
        name="risk_service",
        path_prefix="/api/risk",
        instances=[
            ServiceInstance(url="http://risk_service:8007"),
        ],
        rate_limit=200,
        auth_required=True
    )
    service_registry.register_service(risk_service)

    # Web UI
    web_ui_service = ServiceConfig(
        name="web_ui",
        path_prefix="/",
        instances=[
            ServiceInstance(url="http://web_ui:8008"),
        ],
        rate_limit=10000,
        auth_required=False  # Public access for web UI
    )
    service_registry.register_service(web_ui_service)

    logger.info("API Gateway services initialized")


async def health_check_loop():
    """Background health check loop"""
    while True:
        await service_registry.health_check_services()
        await asyncio.sleep(30)  # Check every 30 seconds


def get_client_id(request: Request) -> str:
    """Extract client identifier from request"""
    # Use IP address as client identifier (enhance with API keys/tokens in production)
    client_ip = request.client.host if request.client else "unknown"
    return client_ip


def authenticate_request(request: Request) -> Optional[Dict[str, Any]]:
    """Authenticate the request"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split(" ")[1]
    return auth_middleware.verify_token(token)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def gateway_handler(request: Request, path: str):
    """Main gateway handler for routing requests to services"""

    start_time = time.time()

    try:
        # Find matching service
        target_service = None
        remaining_path = ""

        for service_name, service in service_registry.services.items():
            if path.startswith(service.path_prefix.lstrip("/")):
                target_service = service
                remaining_path = path[len(service.path_prefix.lstrip("/")):].lstrip("/")
                break

        if not target_service:
            REQUEST_COUNT.labels(method=request.method, endpoint=path, status="404").inc()
            raise HTTPException(status_code=404, detail="Service not found")

        # Check authentication
        if target_service.auth_required:
            user = authenticate_request(request)
            if not user:
                AUTH_FAILURES.inc()
                REQUEST_COUNT.labels(method=request.method, endpoint=path, status="401").inc()
                raise HTTPException(status_code=401, detail="Authentication required")

        # Check rate limiting
        if target_service.rate_limit:
            client_id = get_client_id(request)
            if not rate_limiter.is_allowed(client_id, target_service.rate_limit):
                RATE_LIMIT_EXCEEDED.labels(client_id=client_id).inc()
                REQUEST_COUNT.labels(method=request.method, endpoint=path, status="429").inc()
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Get service instance
        instance = service_registry.get_service_instance(target_service.name)
        if not instance:
            REQUEST_COUNT.labels(method=request.method, endpoint=path, status="503").inc()
            raise HTTPException(status_code=503, detail="Service unavailable")

        # Build target URL
        target_url = f"{instance.url}/{remaining_path}"
        if request.url.query:
            target_url += f"?{request.url.query}"

        # Prepare headers (remove host header)
        headers = dict(request.headers)
        headers.pop("host", None)

        # Forward request
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=target_service.timeout)) as session:
            # Read request body
            body = await request.body()

            # Forward the request
            async with session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=body
            ) as response:

                # Update instance metrics
                response_time = time.time() - start_time
                instance.request_count += 1
                instance.avg_response_time = (
                    (instance.avg_response_time * (instance.request_count - 1)) + response_time
                ) / instance.request_count

                if response.status >= 400:
                    instance.error_count += 1

                # Update Prometheus metrics
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=target_service.name,
                    status=str(response.status)
                ).inc()
                REQUEST_LATENCY.labels(
                    method=request.method,
                    endpoint=target_service.name
                ).observe(response_time)

                # Return response
                response_body = await response.read()
                return Response(
                    content=response_body,
                    status_code=response.status,
                    headers=dict(response.headers)
                )

    except HTTPException as e:
        # Update metrics for HTTP exceptions
        REQUEST_COUNT.labels(method=request.method, endpoint=path, status=str(e.status_code)).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=path).observe(time.time() - start_time)
        raise
    except Exception as e:
        logger.error(f"Gateway error: {e}")
        REQUEST_COUNT.labels(method=request.method, endpoint=path, status="500").inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=path).observe(time.time() - start_time)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Gateway health check"""
    services_status = {}
    for service_name, service in service_registry.services.items():
        healthy_instances = sum(1 for instance in service.instances if instance.status == ServiceStatus.HEALTHY)
        services_status[service_name] = {
            "total_instances": len(service.instances),
            "healthy_instances": healthy_instances,
            "status": "healthy" if healthy_instances > 0 else "unhealthy"
        }

    overall_status = "healthy" if all(s["status"] == "healthy" for s in services_status.values()) else "unhealthy"

    return {
        "status": overall_status,
        "timestamp": time.time(),
        "services": services_status
    }


@app.get("/services")
async def list_services():
    """List all registered services"""
    return {
        "services": [
            {
                "name": service.name,
                "path_prefix": service.path_prefix,
                "instances": len(service.instances),
                "rate_limit": service.rate_limit,
                "auth_required": service.auth_required
            }
            for service in service_registry.services.values()
        ]
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    # Update service health metrics
    for service_name, service in service_registry.services.items():
        healthy_instances = sum(1 for instance in service.instances if instance.status == ServiceStatus.HEALTHY)
        SERVICE_HEALTH_STATUS.labels(service_name=service_name).set(1 if healthy_instances > 0 else 0)

    return StarletteResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/auth/login")
async def login(credentials: Dict[str, str]):
    """Simple login endpoint (for demonstration)"""
    # This is a simplified example - use proper authentication in production
    username = credentials.get("username")
    password = credentials.get("password")

    if username == "admin" and password == "admin":  # Demo credentials
        token = auth_middleware.create_token(username, ["admin", "user"])
        return {"token": token, "user": username, "roles": ["admin", "user"]}

    raise HTTPException(status_code=401, detail="Invalid credentials")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)