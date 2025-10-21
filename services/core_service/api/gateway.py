# Core Service API Gateway
# Handles routing, authentication, and service discovery

from typing import Dict, Any, Optional
import logging
from fastapi import Request, HTTPException
import httpx

logger = logging.getLogger(__name__)

class APIGateway:
    """API Gateway for service routing and authentication"""

    def __init__(self):
        self.service_registry: Dict[str, Dict[str, Any]] = {}
        self.client = httpx.AsyncClient(timeout=30.0)

    async def initialize(self):
        """Initialize the API gateway"""
        logger.info("Initializing API Gateway...")

        # Register known services (will be expanded)
        self.service_registry = {
            "data_service": {
                "url": "http://localhost:8001",
                "prefix": "/api/data",
                "health_endpoint": "/health"
            },
            "backtesting_service": {
                "url": "http://localhost:8002",
                "prefix": "/api/backtesting",
                "health_endpoint": "/health"
            },
            "trading_bot_service": {
                "url": "http://localhost:8003",
                "prefix": "/api/trading",
                "health_endpoint": "/health"
            }
        }

        logger.info("API Gateway initialized")

    async def register_service(self, service_name: str, service_url: str):
        """Register a new service"""
        self.service_registry[service_name] = {
            "url": service_url,
            "prefix": f"/api/{service_name}",
            "health_endpoint": "/health"
        }
        logger.info(f"Registered service: {service_name} at {service_url}")

    async def get_registered_services(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered services"""
        return self.service_registry.copy()

    async def shutdown(self):
        """Shutdown the API gateway"""
        await self.client.aclose()
        logger.info("API Gateway shut down")

    async def handle_request(self, request: Request, path: str) -> Dict[str, Any]:
        """Handle incoming API requests and route to appropriate services"""

        # Determine target service based on path
        target_service = self._get_target_service(path)

        if not target_service:
            raise HTTPException(status_code=404, detail="Service not found")

        # Build target URL
        service_config = self.service_registry[target_service]
        target_url = f"{service_config['url']}{path}"

        try:
            # Forward request to target service
            response = await self.client.request(
                method=request.method,
                url=target_url,
                headers=dict(request.headers),
                content=await request.body()
            )

            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text
            }

        except Exception as e:
            logger.error(f"Error forwarding request to {target_service}: {e}")
            raise HTTPException(status_code=502, detail="Service unavailable")

    def _get_target_service(self, path: str) -> Optional[str]:
        """Determine which service should handle the request"""

        if path.startswith("/api/data"):
            return "data_service"
        elif path.startswith("/api/backtesting"):
            return "backtesting_service"
        elif path.startswith("/api/trading"):
            return "trading_bot_service"

        return None

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all registered services"""
        health_status = {}

        for service_name, config in self.service_registry.items():
            try:
                health_url = f"{config['url']}{config['health_endpoint']}"
                response = await self.client.get(health_url, timeout=5.0)

                health_status[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                health_status[service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

        return health_status

    async def get_metrics(self) -> Dict[str, Any]:
        """Collect API gateway metrics"""
        return {
            "api_gateway": {
                "registered_services": len(self.service_registry),
                "total_requests": 0,  # TODO: Implement request counting
            }
        }