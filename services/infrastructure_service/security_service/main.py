#!/usr/bin/env python3
"""
TradPal Security Service

Zero-Trust Security Service providing mTLS authentication, JWT token management,
and secrets management for the microservices architecture.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from .service import SecurityService, SecurityConfig, ServiceCredentials, JWTToken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TradPal Security Service",
    description="Zero-Trust Security Service for TradPal microservices",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global security service instance
security_service: Optional[SecurityService] = None

# Prometheus metrics
CREDENTIALS_ISSUED = Counter(
    'security_service_credentials_issued_total',
    'Total number of mTLS credentials issued',
    ['service_name']
)

TOKENS_GENERATED = Counter(
    'security_service_tokens_generated_total',
    'Total number of JWT tokens generated',
    ['service_name']
)

TOKENS_VALIDATED = Counter(
    'security_service_tokens_validated_total',
    'Total number of JWT tokens validated',
    ['service_name']
)

SECRETS_STORED = Counter(
    'security_service_secrets_stored_total',
    'Total number of secrets stored'
)

SECRETS_RETRIEVED = Counter(
    'security_service_secrets_retrieved_total',
    'Total number of secrets retrieved'
)


@app.on_event("startup")
async def startup_event():
    """Initialize security service on startup"""
    global security_service

    # Load configuration from environment
    config = SecurityConfig(
        enable_mtls=True,
        enable_jwt=True,
        enable_vault=False,  # Can be enabled via environment variables
        jwt_secret_key="tradpal-jwt-secret-key-2024",  # Use environment variable in production
        ca_cert_path="cache/security/ca/ca_cert.pem",
        ca_key_path="cache/security/ca/ca_key.pem",
        vault_url=None,  # Set via environment
        vault_token=None,  # Set via environment
    )

    security_service = SecurityService(config)
    await security_service.start()

    logger.info("Security Service started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global security_service
    if security_service:
        await security_service.stop()


@app.get("/health")
async def health_check():
    """Service health check"""
    if not security_service:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "security_service",
                "error": "Service not initialized"
            }
        )

    health = await security_service.health_check()
    status_code = 200 if health["status"] == "healthy" else 503

    return JSONResponse(status_code=status_code, content=health)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


@app.post("/credentials/{service_name}")
async def issue_credentials(service_name: str):
    """Issue mTLS credentials for a service"""
    try:
        if not security_service:
            raise HTTPException(status_code=503, detail="Security service not available")

        credentials = await security_service.issue_service_credentials(service_name)

        # Update metrics
        CREDENTIALS_ISSUED.labels(service_name=service_name).inc()

        return {
            "service_name": credentials.service_name,
            "certificate": credentials.certificate,
            "private_key": credentials.private_key,
            "ca_certificate": credentials.ca_certificate,
            "issued_at": credentials.issued_at.isoformat(),
            "expires_at": credentials.expires_at.isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to issue credentials for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to issue credentials: {str(e)}")


@app.get("/credentials/{service_name}")
async def get_credentials(service_name: str):
    """Get existing mTLS credentials for a service"""
    try:
        if not security_service:
            raise HTTPException(status_code=503, detail="Security service not available")

        if service_name not in security_service.service_credentials:
            raise HTTPException(status_code=404, detail="Credentials not found")

        credentials = security_service.service_credentials[service_name]

        return {
            "service_name": credentials.service_name,
            "certificate": credentials.certificate,
            "ca_certificate": credentials.ca_certificate,
            "issued_at": credentials.issued_at.isoformat(),
            "expires_at": credentials.expires_at.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get credentials for {service_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get credentials")


@app.delete("/credentials/{service_name}")
async def revoke_credentials(service_name: str):
    """Revoke mTLS credentials for a service"""
    try:
        if not security_service:
            raise HTTPException(status_code=503, detail="Security service not available")

        if service_name in security_service.service_credentials:
            del security_service.service_credentials[service_name]
            # Note: In production, you'd also revoke the certificate in a CRL

        return {"status": "revoked", "service_name": service_name}

    except Exception as e:
        logger.error(f"Failed to revoke credentials for {service_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to revoke credentials")


@app.post("/tokens/{service_name}")
async def generate_token(service_name: str, permissions: Dict[str, Any] = None):
    """Generate JWT token for a service"""
    try:
        if not security_service:
            raise HTTPException(status_code=503, detail="Security service not available")

        perms = permissions.get("permissions", ["read"]) if permissions else ["read"]
        token = await security_service.generate_jwt_token(service_name, perms)

        # Update metrics
        TOKENS_GENERATED.labels(service_name=service_name).inc()

        return {
            "token": token.token,
            "service_name": token.service_name,
            "issued_at": token.issued_at.isoformat(),
            "expires_at": token.expires_at.isoformat(),
            "permissions": token.permissions
        }

    except Exception as e:
        logger.error(f"Failed to generate token for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate token: {str(e)}")


@app.post("/tokens/validate")
async def validate_token(token_data: Dict[str, str]):
    """Validate JWT token"""
    try:
        if not security_service:
            raise HTTPException(status_code=503, detail="Security service not available")

        token = token_data.get("token")
        if not token:
            raise HTTPException(status_code=400, detail="Token required")

        validated_token = await security_service.validate_jwt_token(token)

        if not validated_token:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        # Update metrics
        TOKENS_VALIDATED.labels(service_name=validated_token.service_name).inc()

        return {
            "valid": True,
            "service_name": validated_token.service_name,
            "issued_at": validated_token.issued_at.isoformat(),
            "expires_at": validated_token.expires_at.isoformat(),
            "permissions": validated_token.permissions
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate token: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate token")


@app.put("/secrets/{path:path}")
async def store_secret(path: str, secret_data: Dict[str, Any]):
    """Store a secret"""
    try:
        if not security_service:
            raise HTTPException(status_code=503, detail="Security service not available")

        success = await security_service.store_secret(path, secret_data)

        if success:
            SECRETS_STORED.inc()
            return {"status": "stored", "path": path}
        else:
            raise HTTPException(status_code=500, detail="Failed to store secret")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to store secret at {path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to store secret")


@app.get("/secrets/{path:path}")
async def retrieve_secret(path: str):
    """Retrieve a secret"""
    try:
        if not security_service:
            raise HTTPException(status_code=503, detail="Security service not available")

        secret = await security_service.retrieve_secret(path)

        if secret is None:
            raise HTTPException(status_code=404, detail="Secret not found")

        SECRETS_RETRIEVED.inc()
        return {"path": path, "data": secret}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve secret at {path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve secret")


@app.get("/ca/certificate")
async def get_ca_certificate():
    """Get CA certificate for mTLS setup"""
    try:
        if not security_service or not security_service.ca_certificate:
            raise HTTPException(status_code=503, detail="CA not initialized")

        ca_cert_pem = security_service.ca_certificate.public_bytes(
            encoding=serialization.Encoding.PEM
        ).decode()

        return {
            "ca_certificate": ca_cert_pem,
            "issued_at": security_service.ca_certificate.not_valid_before.isoformat(),
            "expires_at": security_service.ca_certificate.not_valid_after.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get CA certificate: {e}")
        raise HTTPException(status_code=500, detail="Failed to get CA certificate")


@app.get("/services")
async def list_services():
    """List all services with issued credentials"""
    try:
        if not security_service:
            raise HTTPException(status_code=503, detail="Security service not available")

        services = []
        for service_name, creds in security_service.service_credentials.items():
            services.append({
                "service_name": service_name,
                "issued_at": creds.issued_at.isoformat(),
                "expires_at": creds.expires_at.isoformat(),
                "has_certificate": True
            })

        return {"services": services}

    except Exception as e:
        logger.error(f"Failed to list services: {e}")
        raise HTTPException(status_code=500, detail="Failed to list services")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)</content>
<parameter name="filePath">/Users/danielsadowski/VSCodeProjects/tradpal/tradpal/services/infrastructure_service/security_service/main.py