"""
TradPal Data Service Orchestrator
Main orchestrator for the unified data service
"""

import logging
from typing import Dict, Any, Optional
import asyncio

# Import simplified components
from .data_sources.service import DataService
from .api.main import app as api_app

logger = logging.getLogger(__name__)


class DataServiceOrchestrator:
    """Main orchestrator for the unified data service"""

    def __init__(self, event_system=None):
        self.event_system = event_system
        self.data_service = DataService(event_system)
        self.api_app = api_app
        self.is_initialized = False

    async def initialize(self):
        """Initialize all data service components"""
        logger.info("Initializing Data Service Orchestrator...")

        try:
            # Initialize core data service
            await self.data_service.initialize()

            # TODO: Initialize alternative data services when implemented
            # TODO: Initialize market regime services when implemented

            self.is_initialized = True
            logger.info("Data Service Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Data Service Orchestrator: {e}")
            raise

    async def shutdown(self):
        """Shutdown all data service components"""
        logger.info("Shutting down Data Service Orchestrator...")

        try:
            # Shutdown core data service
            await self.data_service.shutdown()

            # TODO: Shutdown alternative data services when implemented
            # TODO: Shutdown market regime services when implemented

            self.is_initialized = False
            logger.info("Data Service Orchestrator shut down successfully")

        except Exception as e:
            logger.error(f"Error during Data Service Orchestrator shutdown: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_data = {
            "service": "data_service_orchestrator",
            "status": "healthy" if self.is_initialized else "unhealthy",
            "components": {
                "data_service": {
                    "status": "healthy" if self.data_service.is_initialized else "unhealthy",
                    "available_sources": self.data_service.get_available_sources()
                },
                # TODO: Add health checks for alternative data and market regime services
            },
            "api": {
                "status": "available",
                "endpoints": [
                    "GET /",
                    "GET /health",
                    "GET /data/{symbol}",
                    "GET /data/{symbol}/info",
                    "GET /sources",
                    "POST /data/validate"
                ]
            }
        }

        return health_data

    async def get_service_info(self) -> Dict[str, Any]:
        """Get information about the data service"""
        return {
            "name": "TradPal Data Service",
            "version": "1.0.0",
            "description": "Unified data service for market data, alternative data, and market regime analysis",
            "components": [
                "data_sources",
                "alternative_data",  # TODO: Implement
                "market_regime"      # TODO: Implement
            ],
            "api_port": 8001
        }


# Global orchestrator instance
orchestrator = DataServiceOrchestrator()


async def main():
    """Main entry point for running the data service"""
    try:
        await orchestrator.initialize()

        # Start API server
        import uvicorn
        logger.info("Starting Data Service API on port 8001...")
        uvicorn.run(api_app, host="0.0.0.0", port=8001)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
