#!/usr/bin/env python3
"""
TradPal Alert Forwarder CLI
Command-line interface for the Alert Forwarder service
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.alert_forwarder.config import AlertConfig
from services.alert_forwarder.forwarder import AlertForwarder


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


async def main():
    """Main entry point"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting TradPal Alert Forwarder")

    try:
        # Load configuration
        config = AlertConfig.from_env()
        logger.info(f"Configuration loaded: min_priority={config.min_priority}, batching={config.batch_alerts}")

        # Create and initialize forwarder
        forwarder = AlertForwarder(config)

        if not await forwarder.initialize():
            logger.error("Failed to initialize alert forwarder")
            sys.exit(1)

        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(forwarder.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start forwarder
        await forwarder.start()

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        logger.info("Alert Forwarder stopped")


if __name__ == "__main__":
    asyncio.run(main())