#!/usr/bin/env python3
"""
TradPal Core Service - Main Orchestrator

Provides centralized API gateway, event handling, security, and core calculations
for the TradPal trading system.

Features:
- API Gateway with service routing and authentication
- Event-driven communication via Redis Streams
- Zero-trust security (mTLS, JWT, secrets management)
- Core trading calculations and indicators
- Health monitoring and metrics
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .api.gateway import APIGateway
from .events.system import EventSystem
from .security.service_wrapper import SecurityService
from .calculations.service import CalculationService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoreService:
    """Main orchestrator service for TradPal"""

    def __init__(self):
        self.app = FastAPI(
            title="TradPal Core Service",
            description="Main orchestrator for TradPal trading system",
            version="1.0.0"
        )

        # Initialize components
        self.api_gateway = APIGateway()
        self.event_system = EventSystem()
        self.security_service = SecurityService()
        self.calculation_service = CalculationService()

        # Setup middleware
        self._setup_middleware()

        # Setup routes
        self._setup_routes()

        # Health status
        self.healthy = False

    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy" if self.healthy else "unhealthy",
                "service": "core_service",
                "components": {
                    "api_gateway": await self.api_gateway.health_check(),
                    "event_system": await self.event_system.health_check(),
                    "security_service": await self.security_service.health_check(),
                    "calculation_service": await self.calculation_service.health_check()
                }
            }

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return await self._collect_metrics()

        # API Gateway routes - delegate to gateway
        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def api_gateway_handler(request: Request, path: str):
            """Delegate requests to appropriate services via API gateway"""
            return await self.api_gateway.handle_request(request, path)

    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all components"""
        metrics = {
            "core_service": {
                "uptime": 0,  # TODO: Implement uptime tracking
                "requests_total": 0,  # TODO: Implement request counting
            }
        }

        # Collect component metrics
        try:
            metrics.update(await self.api_gateway.get_metrics())
            metrics.update(await self.event_system.get_metrics())
            metrics.update(await self.security_service.get_metrics())
            metrics.update(await self.calculation_service.get_metrics())
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

        return metrics

    async def startup(self):
        """Initialize all components"""
        logger.info("🚀 Starting TradPal Core Service...")

        try:
            # Initialize components in order
            await self.security_service.initialize()
            await self.event_system.initialize()
            await self.api_gateway.initialize()
            await self.calculation_service.initialize()

            self.healthy = True
            logger.info("✅ Core Service started successfully")

        except Exception as e:
            logger.error(f"❌ Failed to start Core Service: {e}")
            raise

    async def shutdown(self):
        """Shutdown all components"""
        logger.info("🛑 Shutting down TradPal Core Service...")

        try:
            await self.calculation_service.shutdown()
            await self.api_gateway.shutdown()
            await self.event_system.shutdown()
            await self.security_service.shutdown()

            self.healthy = False
            logger.info("✅ Core Service shut down successfully")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

    async def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the service"""
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)

        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            await self.startup()
            await server.serve()
        except Exception as e:
            logger.error(f"Service error: {e}")
        finally:
            await self.shutdown()


# Global service instance
core_service = CoreService()


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="TradPal Core Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    await core_service.run(host=args.host, port=args.port)


if __name__ == "__main__":
    asyncio.run(main())