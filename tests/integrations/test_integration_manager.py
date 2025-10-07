import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import json

# Add project paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrations.base import IntegrationManager, BaseIntegration, IntegrationConfig
from integrations.telegram.bot import TelegramIntegration, TelegramConfig
from integrations.email_integration.email import EmailIntegration, EmailConfig


import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add project paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrations.base import IntegrationManager, BaseIntegration


class TestIntegrationManager:
    """Test the integration manager functionality."""

    def test_integration_manager_initialization(self):
        """Test integration manager initialization."""
        manager = IntegrationManager()
        assert isinstance(manager.integrations, dict)
        assert len(manager.integrations) == 0

    def test_register_integration(self):
        """Test registering an integration."""
        manager = IntegrationManager()

        # Create mock integration
        mock_integration = MagicMock(spec=BaseIntegration)
        mock_integration.name = "test_integration"

        manager.register_integration("test", mock_integration)
        assert "test" in manager.integrations
        assert manager.integrations["test"] == mock_integration

    def test_initialize_all_integrations(self):
        """Test initializing all integrations."""
        manager = IntegrationManager()

                # Create mock integrations
        mock_integration1 = MagicMock(spec=BaseIntegration)
        mock_integration1.initialize.return_value = True
        mock_integration1.is_enabled.return_value = True
        mock_integration1.name = "integration1"

        mock_integration2 = MagicMock(spec=BaseIntegration)
        mock_integration2.initialize.return_value = True
        mock_integration2.is_enabled.return_value = False
        mock_integration2.name = "integration2"

        manager.register_integration("int1", mock_integration1)
        manager.register_integration("int2", mock_integration2)

        results = manager.initialize_all()

        assert results["int1"] == True
        assert results["int2"] == True  # Disabled integrations return True (successful skip)
        mock_integration1.initialize.assert_called_once()
        mock_integration2.initialize.assert_not_called()  # Disabled integrations are not initialized

    def test_send_signal_to_all(self):
        """Test sending signal to all integrations."""
        manager = IntegrationManager()

        # Create mock integrations
        mock_integration1 = MagicMock(spec=BaseIntegration)
        mock_integration1.send_signal_safe.return_value = True
        mock_integration1.is_enabled.return_value = True
        mock_integration1.name = "integration1"

        mock_integration2 = MagicMock(spec=BaseIntegration)
        mock_integration2.send_signal_safe.return_value = False
        mock_integration2.is_enabled.return_value = True
        mock_integration2.name = "integration2"

        manager.register_integration("int1", mock_integration1)
        manager.register_integration("int2", mock_integration2)

        test_signal = {"signal": "BUY", "price": 100.0}
        results = manager.send_signal_to_all(test_signal)

        assert results["int1"] == True
        assert results["int2"] == False
        # The signal gets transformed to SignalData format
        mock_integration1.send_signal_safe.assert_called_once()
        mock_integration2.send_signal_safe.assert_called_once()

    def test_shutdown_all_integrations(self):
        """Test shutting down all integrations."""
        manager = IntegrationManager()

        # Create mock integrations
        mock_integration1 = MagicMock(spec=BaseIntegration)
        mock_integration1.name = "integration1"

        mock_integration2 = MagicMock(spec=BaseIntegration)
        mock_integration2.name = "integration2"

        manager.register_integration("int1", mock_integration1)
        manager.register_integration("int2", mock_integration2)

        manager.shutdown_all()

        # shutdown_all calls send_shutdown_message on each integration
        mock_integration1.send_shutdown_message.assert_called_once()
        mock_integration2.send_shutdown_message.assert_called_once()

    def test_get_status_overview(self):
        """Test getting status overview."""
        manager = IntegrationManager()

        # Create mock integrations
        mock_integration1 = MagicMock(spec=BaseIntegration)
        mock_integration1.name = "integration1"
        mock_integration1.is_enabled.return_value = True
        mock_integration1.get_status.return_value = {"enabled": True, "initialized": True}

        mock_integration2 = MagicMock(spec=BaseIntegration)
        mock_integration2.name = "integration2"
        mock_integration2.is_enabled.return_value = False
        mock_integration2.get_status.return_value = {"enabled": False, "initialized": False}

        manager.register_integration("int1", mock_integration1)
        manager.register_integration("int2", mock_integration2)

        status = manager.get_status_overview()

        assert status["total_integrations"] == 2
        assert status["enabled_integrations"] == 1
        assert len(status["integrations"]) == 2
        assert status["integrations"]["int1"]["enabled"] == True
        assert status["integrations"]["int1"]["initialized"] == True
        assert status["integrations"]["int2"]["enabled"] == False
        assert status["integrations"]["int2"]["initialized"] == False

    def test_test_all_connections(self):
        """Test testing all connections."""
        manager = IntegrationManager()

        # Create mock integrations
        mock_integration1 = MagicMock(spec=BaseIntegration)
        mock_integration1.test_connection.return_value = True
        mock_integration1.is_enabled.return_value = True
        mock_integration1.name = "integration1"

        mock_integration2 = MagicMock(spec=BaseIntegration)
        mock_integration2.test_connection.return_value = False
        mock_integration2.is_enabled.return_value = True
        mock_integration2.name = "integration2"

        manager.register_integration("int1", mock_integration1)
        manager.register_integration("int2", mock_integration2)

        results = manager.test_all_connections()

        assert results["int1"] == True
        assert results["int2"] == False
        mock_integration1.test_connection.assert_called_once()
        mock_integration2.test_connection.assert_called_once()


