"""
Tests for Security Service
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from .service import (
    SecurityService,
    SecurityConfig,
    ServiceCredentials,
    JWTToken
)


class TestSecurityService:
    """Unit tests for security service."""

    @pytest.fixture
    def security_service(self):
        """Create security service for testing."""
        config = SecurityConfig()
        service = SecurityService(config)
        return service

    @pytest.mark.asyncio
    async def test_initialization(self, security_service):
        """Test service initialization."""
        assert not security_service.is_running
        assert security_service.config is not None
        assert security_service.service_credentials == {}
        assert security_service.active_tokens == {}

    @pytest.mark.asyncio
    async def test_service_start_stop(self, security_service):
        """Test service start and stop."""
        await security_service.start()
        assert security_service.is_running

        await security_service.stop()
        assert not security_service.is_running

    @pytest.mark.asyncio
    async def test_health_check(self, security_service):
        """Test health check functionality."""
        await security_service.start()

        health = await security_service.health_check()

        assert health['service'] == 'security_service'
        assert health['status'] == 'healthy'
        assert 'timestamp' in health
        assert 'components' in health
        assert 'active_credentials' in health
        assert 'active_tokens' in health

        await security_service.stop()

        # Test stopped state
        health = await security_service.health_check()
        assert health['status'] == 'stopped'

    @pytest.mark.asyncio
    async def test_jwt_token_generation(self, security_service):
        """Test JWT token generation."""
        await security_service.start()

        token = await security_service.generate_jwt_token(
            service_name="test_service",
            permissions=["read", "write"]
        )

        assert isinstance(token, JWTToken)
        assert token.service_name == "test_service"
        assert token.permissions == ["read", "write"]
        assert token.token is not None
        assert token.expires_at > token.issued_at

        await security_service.stop()

    @pytest.mark.asyncio
    async def test_jwt_token_validation(self, security_service):
        """Test JWT token validation."""
        await security_service.start()

        # Generate token
        token = await security_service.generate_jwt_token(
            service_name="test_service",
            permissions=["read"]
        )

        # Validate token
        validated = await security_service.validate_jwt_token(token.token)

        assert validated is not None
        assert validated.service_name == "test_service"
        assert validated.permissions == ["read"]

        # Test invalid token
        invalid_validated = await security_service.validate_jwt_token("invalid.token.here")
        assert invalid_validated is None

        await security_service.stop()

    @pytest.mark.asyncio
    async def test_secrets_management_local(self, security_service):
        """Test secrets management with local storage."""
        await security_service.start()

        test_data = {"key": "value", "secret": "data"}

        # Store secret
        success = await security_service.store_secret("test/path", test_data)
        assert success

        # Retrieve secret
        retrieved = await security_service.retrieve_secret("test/path")
        assert retrieved == test_data

        # Test non-existent secret
        not_found = await security_service.retrieve_secret("non/existent")
        assert not_found is None

        await security_service.stop()

    @pytest.mark.asyncio
    async def test_mtls_credentials_issuance(self, security_service):
        """Test mTLS credentials issuance."""
        await security_service.start()

        with patch('builtins.open', create=True) as mock_file, \
             patch('json.dump') as mock_json_dump:

            credentials = await security_service.issue_service_credentials("test_service")

            assert isinstance(credentials, ServiceCredentials)
            assert credentials.service_name == "test_service"
            assert credentials.certificate is not None
            assert credentials.private_key is not None
            assert credentials.ca_certificate is not None
            assert credentials.expires_at > credentials.issued_at

            # Check if credentials are stored
            assert "test_service" in security_service.service_credentials

        await security_service.stop()


class TestSecurityServiceIntegration:
    """Integration tests requiring external dependencies."""

    @pytest.mark.asyncio
    async def test_service_lifecycle(self):
        """Test complete service lifecycle."""
        config = SecurityConfig()
        service = SecurityService(config)

        # Start service
        await service.start()
        assert service.is_running

        # Check health
        health = await service.health_check()
        assert health['status'] == 'healthy'

        # Test basic functionality
        token = await service.generate_jwt_token("integration_test")
        assert token is not None

        validated = await service.validate_jwt_token(token.token)
        assert validated is not None

        # Stop service
        await service.stop()
        assert not service.is_running

        # Check stopped health
        health = await service.health_check()
        assert health['status'] == 'stopped'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])