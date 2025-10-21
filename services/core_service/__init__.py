# TradPal Core Service
"""
Main orchestrator service for the TradPal trading system.

This service consolidates the following previously separate services:
- API Gateway: Service routing and load balancing
- Event System: Redis Streams-based event communication
- Security Service: Zero-trust security (mTLS, JWT, secrets)
- Core Calculations: Technical indicators and trading signals

The core service provides a unified entry point for all trading operations
while maintaining backward compatibility through API gateway routing.
"""

__version__ = "3.0.1"
__author__ = "TradPal Team"
__description__ = "TradPal Core Service - Main orchestrator for trading operations"

# Import main components for easy access
from .main import CoreService
from .api.gateway import APIGateway
from .events.system import EventSystemService
from .security.service_wrapper import SecurityService
from .calculations.service import CalculationService

__all__ = [
    "CoreService",
    "APIGateway",
    "EventSystemService",
    "SecurityService",
    "CalculationService",
]