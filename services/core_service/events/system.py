# Core Service Event System Integration
# Provides event handling capabilities within the core service

import logging
from typing import Dict, Any
from .main import EventSystem

logger = logging.getLogger(__name__)

class EventSystemService:
    """Event system service for core service integration"""

    def __init__(self):
        self.event_system: EventSystem = None

    async def initialize(self):
        """Initialize the event system"""
        logger.info("Initializing Event System Service...")
        self.event_system = EventSystem()
        await self.event_system.initialize()
        logger.info("Event System Service initialized")

    async def shutdown(self):
        """Shutdown the event system"""
        if self.event_system:
            # TODO: Implement proper shutdown for EventSystem
            pass
        logger.info("Event System Service shut down")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for event system"""
        try:
            if self.event_system and self.event_system.redis:
                info = await self.event_system.redis.xinfo_stream("tradpal_events")
                return {
                    "status": "healthy",
                    "stream_length": info["length"]
                }
            else:
                return {"status": "unhealthy", "error": "Event system not initialized"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def get_metrics(self) -> Dict[str, Any]:
        """Get event system metrics"""
        return {
            "event_system": {
                "status": "operational" if self.event_system else "offline",
                "stream_length": 0,  # TODO: Implement proper metrics
            }
        }