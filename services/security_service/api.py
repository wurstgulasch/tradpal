"""
FastAPI REST API for Security Service
Provides endpoints for mTLS credentials, JWT tokens, and secrets management.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

from .service import SecurityService, SecurityConfig, ServiceCredentials, JWTToken


# Pydantic models for API
class IssueCredentialsRequest(BaseModel):
    """Request to issue service credentials."""
    service_name: str = Field(..., description="Name of the service")


class GenerateTokenRequest(BaseModel):
    """Request to generate JWT token."""
    service_name: str = Field(..., description="Name of the service")
    permissions: List[str] = Field(default=["read"], description="Service permissions")


class StoreSecretRequest(BaseModel):
    """Request to store a secret."""
    path: str = Field(..., description="Secret path")
    data: Dict[str, Any] = Field(..., description="Secret data")


class HealthResponse(BaseModel):
    """Health check response."""
    service: str
    status: str
    timestamp: str
    components: Dict[str, bool]
    active_credentials: int
    active_tokens: int


class SecurityAPI:
    """FastAPI application for security service."""

    def __init__(self, config: SecurityConfig, security_service: SecurityService):
        self.config = config
        self.security_service = security_service
        self.logger = logging.getLogger(__name__)

        # Create FastAPI app
        self.app = FastAPI(
            title="TradPal Security Service",
            description="Zero-Trust Security API for mTLS, JWT, and secrets management",
            version="1.0.0"
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Security scheme
        self.security_scheme = HTTPBearer()

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register API routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Get service health status."""
            return await self.security_service.health_check()

        @self.app.post("/credentials/issue", response_model=ServiceCredentials)
        async def issue_credentials(
            request: IssueCredentialsRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security_scheme)
        ):
            """Issue mTLS credentials for a service."""
            try:
                # Validate JWT token
                token_info = await self.security_service.validate_jwt_token(credentials.credentials)
                if not token_info or "admin" not in token_info.permissions:
                    raise HTTPException(status_code=403, detail="Insufficient permissions")

                result = await self.security_service.issue_service_credentials(request.service_name)
                return result
            except Exception as e:
                self.logger.error(f"Failed to issue credentials: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/tokens/generate", response_model=JWTToken)
        async def generate_token(request: GenerateTokenRequest):
            """Generate a JWT token for service authentication."""
            try:
                result = await self.security_service.generate_jwt_token(
                    service_name=request.service_name,
                    permissions=request.permissions
                )
                return result
            except Exception as e:
                self.logger.error(f"Failed to generate token: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/tokens/validate")
        async def validate_token(
            credentials: HTTPAuthorizationCredentials = Depends(self.security_scheme)
        ):
            """Validate a JWT token."""
            try:
                token_info = await self.security_service.validate_jwt_token(credentials.credentials)
                if token_info:
                    return {
                        "valid": True,
                        "service_name": token_info.service_name,
                        "permissions": token_info.permissions,
                        "expires_at": token_info.expires_at.isoformat()
                    }
                else:
                    return {"valid": False}
            except Exception as e:
                self.logger.error(f"Failed to validate token: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/secrets/store")
        async def store_secret(
            request: StoreSecretRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self.security_scheme)
        ):
            """Store a secret."""
            try:
                # Validate JWT token
                token_info = await self.security_service.validate_jwt_token(credentials.credentials)
                if not token_info or "write" not in token_info.permissions:
                    raise HTTPException(status_code=403, detail="Insufficient permissions")

                success = await self.security_service.store_secret(request.path, request.data)
                if success:
                    return {"status": "success", "message": f"Secret stored at {request.path}"}
                else:
                    raise HTTPException(status_code=500, detail="Failed to store secret")
            except Exception as e:
                self.logger.error(f"Failed to store secret: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/secrets/{path:path}")
        async def retrieve_secret(
            path: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security_scheme)
        ):
            """Retrieve a secret."""
            try:
                # Validate JWT token
                token_info = await self.security_service.validate_jwt_token(credentials.credentials)
                if not token_info or "read" not in token_info.permissions:
                    raise HTTPException(status_code=403, detail="Insufficient permissions")

                secret = await self.security_service.retrieve_secret(path)
                if secret:
                    return {"data": secret}
                else:
                    raise HTTPException(status_code=404, detail="Secret not found")
            except Exception as e:
                self.logger.error(f"Failed to retrieve secret: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/credentials")
        async def list_credentials(
            credentials: HTTPAuthorizationCredentials = Depends(self.security_scheme)
        ):
            """List all service credentials."""
            try:
                # Validate JWT token
                token_info = await self.security_service.validate_jwt_token(credentials.credentials)
                if not token_info or "admin" not in token_info.permissions:
                    raise HTTPException(status_code=403, detail="Insufficient permissions")

                return {
                    "credentials": list(self.security_service.service_credentials.keys()),
                    "count": len(self.security_service.service_credentials)
                }
            except Exception as e:
                self.logger.error(f"Failed to list credentials: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/tokens")
        async def list_tokens(
            credentials: HTTPAuthorizationCredentials = Depends(self.security_scheme)
        ):
            """List all active tokens."""
            try:
                # Validate JWT token
                token_info = await self.security_service.validate_jwt_token(credentials.credentials)
                if not token_info or "admin" not in token_info.permissions:
                    raise HTTPException(status_code=403, detail="Insufficient permissions")

                tokens_info = []
                for token_hash, token in self.security_service.active_tokens.items():
                    tokens_info.append({
                        "service_name": token.service_name,
                        "issued_at": token.issued_at.isoformat(),
                        "expires_at": token.expires_at.isoformat(),
                        "permissions": token.permissions
                    })

                return {
                    "tokens": tokens_info,
                    "count": len(tokens_info)
                }
            except Exception as e:
                self.logger.error(f"Failed to list tokens: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def start_server(self, host: str = "0.0.0.0", port: int = 8002):
        """Start the FastAPI server."""
        self.logger.info(f"Starting Security API server on {host}:{port}")
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    def run_server(self, host: str = "0.0.0.0", port: int = 8002):
        """Run the server (blocking)."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )


# Global service instance for dependency injection
_security_service: Optional[SecurityService] = None


def get_security_service() -> SecurityService:
    """Dependency injection for security service."""
    if _security_service is None:
        raise RuntimeError("Security service not initialized")
    return _security_service


def create_security_api(config: SecurityConfig, security_service: SecurityService) -> SecurityAPI:
    """Create security API instance."""
    global _security_service
    _security_service = security_service
    return SecurityAPI(config, security_service)


if __name__ == "__main__":
    # For testing purposes
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    async def main():
        config = SecurityConfig()
        service = SecurityService(config)
        await service.start()

        api = create_security_api(config, service)
        await api.start_server()

    asyncio.run(main())
