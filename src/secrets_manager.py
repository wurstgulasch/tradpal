#!/usr/bin/env python3
"""
Secrets Management Module für TradPal Indicator
Unterstützt verschiedene Secrets-Backends: Environment, Vault, AWS Secrets Manager
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class SecretsBackend(ABC):
    """Abstract base class for secrets backends."""

    @abstractmethod
    def get_secret(self, key: str, default: Any = None) -> Any:
        """Retrieve a secret value."""
        pass

    @abstractmethod
    def set_secret(self, key: str, value: Any) -> None:
        """Store a secret value."""
        pass

    @abstractmethod
    def list_secrets(self) -> Dict[str, Any]:
        """List all available secrets."""
        pass

class EnvironmentBackend(SecretsBackend):
    """Environment variables secrets backend."""

    def get_secret(self, key: str, default: Any = None) -> Any:
        """Get secret from environment variable."""
        return os.getenv(key, default)

    def set_secret(self, key: str, value: Any) -> None:
        """Set environment variable (not persistent)."""
        os.environ[key] = str(value)

    def list_secrets(self) -> Dict[str, Any]:
        """List environment variables (filtered for tradpal)."""
        return {k: v for k, v in os.environ.items()
                if k.startswith(('TRADPAL_', 'TELEGRAM_', 'VAULT_', 'AWS_'))}

class VaultBackend(SecretsBackend):
    """HashiCorp Vault secrets backend."""

    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize Vault client."""
        try:
            import hvac
            vault_addr = os.getenv('VAULT_ADDR', 'http://localhost:8200')
            vault_token = os.getenv('VAULT_TOKEN')

            if vault_token:
                self.client = hvac.Client(url=vault_addr, token=vault_token)
                if not self.client.is_authenticated():
                    logger.warning("Vault authentication failed")
                    self.client = None
            else:
                logger.warning("VAULT_TOKEN not set")
        except ImportError:
            logger.warning("hvac not installed, Vault support disabled")
        except Exception as e:
            logger.error(f"Vault initialization failed: {e}")

    def get_secret(self, key: str, default: Any = None) -> Any:
        """Get secret from Vault."""
        if not self.client:
            return default

        try:
            # Assume secrets are stored in kv/data/tradpal path
            secret_path = f"kv/data/tradpal/{key.lower()}"
            response = self.client.read(secret_path)

            if response and 'data' in response:
                return response['data']['data'].get(key, default)
        except Exception as e:
            logger.error(f"Failed to retrieve a secret from Vault: {e}")

        return default

    def set_secret(self, key: str, value: Any) -> None:
        """Store secret in Vault."""
        if not self.client:
            raise RuntimeError("Vault client not available")

        try:
            secret_path = f"kv/data/tradpal/{key.lower()}"
            self.client.write(secret_path, data={key: value})
        except Exception as e:
            logger.error(f"Failed to store secret {key} in Vault: {e}")
            raise

    def list_secrets(self) -> Dict[str, Any]:
        """List secrets from Vault."""
        if not self.client:
            return {}

        try:
            response = self.client.list('kv/metadata/tradpal/')
            if response and 'data' in response:
                secrets = {}
                for secret_key in response['data']['keys']:
                    value = self.get_secret(secret_key)
                    if value is not None:
                        secrets[secret_key] = value
                return secrets
        except Exception as e:
            logger.error(f"Failed to list Vault secrets: {e}")

        return {}

class AWSSecretsManagerBackend(SecretsBackend):
    """AWS Secrets Manager backend."""

    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize AWS client."""
        try:
            import boto3
            region = os.getenv('AWS_REGION', 'us-east-1')
            self.client = boto3.client('secretsmanager', region_name=region)
        except ImportError:
            logger.warning("boto3 not installed, AWS Secrets Manager support disabled")
        except Exception as e:
            logger.error(f"AWS Secrets Manager initialization failed: {e}")

    def get_secret(self, key: str, default: Any = None) -> Any:
        """Get secret from AWS Secrets Manager."""
        if not self.client:
            return default

        try:
            secret_name = f"tradpal/{key.lower()}"
            response = self.client.get_secret_value(SecretId=secret_name)

            if 'SecretString' in response:
                secret_data = json.loads(response['SecretString'])
                return secret_data.get(key, default)
        except self.client.exceptions.ResourceNotFoundException:
            logger.debug(f"Secret {key} not found in AWS Secrets Manager")
        except Exception as e:
            logger.error(f"Failed to retrieve secret {key} from AWS: {e}")

        return default

    def set_secret(self, key: str, value: Any) -> None:
        """Store secret in AWS Secrets Manager."""
        if not self.client:
            raise RuntimeError("AWS client not available")

        try:
            secret_name = f"tradpal/{key.lower()}"
            secret_value = json.dumps({key: value})

            # Try to update existing secret
            try:
                self.client.update_secret(SecretId=secret_name, SecretString=secret_value)
            except self.client.exceptions.ResourceNotFoundException:
                # Create new secret
                self.client.create_secret(Name=secret_name, SecretString=secret_value)
        except Exception as e:
            logger.error(f"Failed to store secret {key} in AWS: {e}")
            raise

    def list_secrets(self) -> Dict[str, Any]:
        """List secrets from AWS Secrets Manager."""
        if not self.client:
            return {}

        try:
            secrets = {}
            paginator = self.client.get_paginator('list_secrets')

            for page in paginator.paginate(
                Filters=[{'Key': 'name', 'Values': ['tradpal/']}]
            ):
                for secret in page['SecretList']:
                    secret_name = secret['Name']
                    if secret_name.startswith('tradpal/'):
                        key = secret_name.split('/', 1)[1]
                        value = self.get_secret(key)
                        if value is not None:
                            secrets[key] = value

            return secrets
        except Exception as e:
            logger.error(f"Failed to list AWS secrets: {e}")

        return {}

class SecretsManager:
    """Main secrets management class."""

    def __init__(self, backend: str = 'env'):
        self.backend_name = backend.lower()
        self.backend = self._create_backend()

    def _create_backend(self) -> SecretsBackend:
        """Create the appropriate secrets backend."""
        if self.backend_name == 'vault':
            return VaultBackend()
        elif self.backend_name == 'aws-secretsmanager':
            return AWSSecretsManagerBackend()
        else:
            return EnvironmentBackend()

    def get_secret(self, key: str, default: Any = None) -> Any:
        """Get a secret value."""
        return self.backend.get_secret(key, default)

    def set_secret(self, key: str, value: Any) -> None:
        """Set a secret value."""
        self.backend.set_secret(key, value)

    def list_secrets(self) -> Dict[str, Any]:
        """List all secrets."""
        return self.backend.list_secrets()

    def get_api_credentials(self) -> Dict[str, str]:
        """Get API credentials for exchanges."""
        return {
            'kraken': {
                'api_key': self.get_secret('TRADPAL_API_KEY'),
                'api_secret': self.get_secret('TRADPAL_API_SECRET')
            },
            'binance': {
                'api_key': self.get_secret('BINANCE_API_KEY'),
                'api_secret': self.get_secret('BINANCE_API_SECRET')
            }
        }

    def get_notification_credentials(self) -> Dict[str, str]:
        """Get notification service credentials."""
        return {
            'telegram': {
                'bot_token': self.get_secret('TELEGRAM_BOT_TOKEN'),
                'chat_id': self.get_secret('TELEGRAM_CHAT_ID')
            },
            'discord': {
                'webhook_url': self.get_secret('DISCORD_WEBHOOK_URL')
            },
            'email': {
                'smtp_server': self.get_secret('SMTP_SERVER'),
                'smtp_port': self.get_secret('SMTP_PORT'),
                'username': self.get_secret('SMTP_USERNAME'),
                'password': self.get_secret('SMTP_PASSWORD')
            }
        }

# Global secrets manager instance
secrets_manager = None
_secrets_backend = None  # For backward compatibility with tests

def get_secrets_manager() -> SecretsManager:
    """Get or create global secrets manager instance."""
    global secrets_manager, _secrets_backend

    if secrets_manager is None:
        backend = os.getenv('SECRETS_BACKEND', 'env')
        secrets_manager = SecretsManager(backend)
        _secrets_backend = secrets_manager.backend

    return secrets_manager

def init_secrets_manager():
    """Initialize secrets manager for the application."""
    manager = get_secrets_manager()
    logger.info(f"Secrets manager initialized with backend: {manager.backend_name}")
    return manager

# Convenience functions
def get_secret(key: str, default: Any = None) -> Any:
    """Convenience function to get a secret."""
    return get_secrets_manager().get_secret(key, default)

def set_secret(key: str, value: Any) -> None:
    """Convenience function to set a secret."""
    return get_secrets_manager().set_secret(key, value)

# Alias for backward compatibility
initialize_secrets_manager = init_secrets_manager