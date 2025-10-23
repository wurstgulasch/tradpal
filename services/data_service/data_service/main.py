"""
TradPal Data Service
Main entry point for the unified data service
"""

import logging
from typing import Dict, Any, Optional
import asyncio

# Import unified data service
from .data_sources.service import DataService
from .api.main import app as api_app

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for running the data service"""
    data_service = DataService()

    try:
        await data_service.initialize()

        # Start API server
        import uvicorn
        logger.info("Starting Data Service API on port 8001...")
        uvicorn.run(api_app, host="0.0.0.0", port=8001)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        await data_service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
