#!/usr/bin/env python3
"""
Service Template - Blueprint for new TradPal microservices

This template provides the complete structure and patterns required for
new services in the TradPal microservices architecture.

Copy this template to create new services following all Best Practices.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.service_settings import (
    SERVICE_NAME, SERVICE_URL, SERVICE_PORT,
    ENABLE_SERVICE, REQUEST_TIMEOUT
)
from services.infrastructure_service.circuit_breaker_service import (
    get_http_circuit_breaker,
    CircuitBreakerConfig
)
from services.infrastructure_service.event_system_service import (
    EventSystem, Event, EventType, get_event_system
)

logger = logging.getLogger(__name__)


# Pydantic Models for API
class ServiceRequest(BaseModel):
    """Request model for service operations"""
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class ServiceResponse(BaseModel):
    """Response model for service operations"""
    success: bool
    data: Dict[str, Any]
    message: str


# Service Client (MANDATORY PATTERN)
class ServiceNameClient:
    """
    Async client for ServiceName service communication.

    Implements: Circuit Breaker, Zero-Trust Security, Async-First Design
    """

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.circuit_breaker = None
        self.base_url = SERVICE_URL

    async def authenticate(self) -> None:
        """Zero-trust authentication - REQUIRED"""
        # Implement mTLS/JWT authentication
        logger.info("Authenticating with security service...")
        # await security_client.authenticate_service()

    @asynccontextmanager
    async def _get_session(self):
        """Circuit breaker protected session - REQUIRED"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            )

        # Get circuit breaker for this service
        if not self.circuit_breaker:
            self.circuit_breaker = await get_http_circuit_breaker(
                service_name=SERVICE_NAME,
                config=CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=60,
                    expected_exception=aiohttp.ClientError
                )
            )

        try:
            async with self.circuit_breaker:
                yield self.session
        except Exception as e:
            logger.error(f"Circuit breaker triggered: {e}")
            raise

    async def business_method(self, **params) -> Dict[str, Any]:
        """Example business method with error handling"""
        async with self._get_session() as session:
            try:
                async with session.post(
                    f"{self.base_url}/endpoint",
                    json=params
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Service error: {error_text}"
                        )
            except Exception as e:
                logger.error(f"Business method failed: {e}")
                raise

    async def close(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()


# Core Service Logic (OPTIONAL - for complex services)
class ServiceNameService:
    """
    Core business logic for ServiceName.

    Optional: Use only if service has complex internal logic
    that should be separated from the API layer.
    """

    def __init__(self):
        self.event_system: Optional[EventSystem] = None

    async def initialize(self):
        """Initialize service components"""
        self.event_system = await get_event_system()

        # Register event handlers
        self.event_system.register_handler(
            EventType.SERVICE_SPECIFIC_EVENT,
            self._handle_event
        )

        logger.info("ServiceName service initialized")

    async def business_logic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Core business logic implementation"""
        try:
            # Implement your business logic here

            # Publish events for state changes
            await self.event_system.publish_event(Event(
                event_type=EventType.SERVICE_OPERATION_COMPLETED,
                source=SERVICE_NAME,
                data={"result": "success", "data": data}
            ))

            return {"status": "success", "data": data}

        except Exception as e:
            logger.error(f"Business logic failed: {e}")

            # Publish error events
            await self.event_system.publish_event(Event(
                event_type=EventType.SERVICE_OPERATION_FAILED,
                source=SERVICE_NAME,
                data={"error": str(e), "data": data}
            ))

            raise

    async def _handle_event(self, event: Event):
        """Handle incoming events"""
        logger.info(f"Received event: {event.event_type.value}")
        # Implement event handling logic

    async def cleanup(self):
        """Cleanup service resources"""
        if self.event_system:
            await self.event_system.close()


# FastAPI Service (MANDATORY)
app = FastAPI(
    title=f"{SERVICE_NAME} Service",
    description="TradPal microservice following Best Practices",
    version="1.0.0"
)

# Global service instance
service_instance: Optional[ServiceNameService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global service_instance

    if ENABLE_SERVICE:
        service_instance = ServiceNameService()
        await service_instance.initialize()
        logger.info(f"{SERVICE_NAME} service started")
    else:
        logger.warning(f"{SERVICE_NAME} service disabled")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global service_instance

    if service_instance:
        await service_instance.cleanup()
        logger.info(f"{SERVICE_NAME} service shut down")


@app.get("/health")
async def health_check():
    """Health check endpoint - REQUIRED"""
    return {
        "service": SERVICE_NAME,
        "status": "healthy",
        "version": "1.0.0",
        "enabled": ENABLE_SERVICE
    }


@app.post("/endpoint", response_model=ServiceResponse)
async def service_endpoint(request: ServiceRequest):
    """Example service endpoint"""
    if not service_instance:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await service_instance.business_logic(request.data)
        return ServiceResponse(
            success=True,
            data=result,
            message="Operation completed successfully"
        )
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        return ServiceResponse(
            success=False,
            data={},
            message=f"Operation failed: {str(e)}"
        )


# Main Entry Point (MANDATORY)
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {SERVICE_NAME} service on port {SERVICE_PORT}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=True,
        log_level="info"
    )