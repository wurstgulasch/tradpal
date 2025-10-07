"""
TradPal Integrations Package

This package contains all external integrations for the TradPal trading system,
including notification services, messaging platforms, and API connections.
"""

from .base import (
    BaseIntegration,
    IntegrationConfig,
    SignalData,
    IntegrationManager,
    integration_manager
)
from .telegram import TelegramIntegration, TelegramConfig
from .email_integration import EmailIntegration, EmailConfig

__all__ = [
    'BaseIntegration',
    'IntegrationConfig',
    'SignalData',
    'IntegrationManager',
    'integration_manager',
    'TelegramIntegration',
    'TelegramConfig',
    'EmailIntegration',
    'EmailConfig'
]