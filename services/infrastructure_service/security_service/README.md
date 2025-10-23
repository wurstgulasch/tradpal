# TradPal Security Service

Zero-Trust Security Service providing mTLS authentication, JWT token management, and secrets management for the TradPal trading system.

## Features

### 🔐 mTLS Authentication
- **Certificate Authority (CA)**: Automated CA creation and management
- **Service Certificates**: Issue X.509 certificates for microservices
- **Mutual TLS**: Secure service-to-service communication
- **Certificate Lifecycle**: Automatic renewal and validation

### 🎫 JWT Token Management
- **Token Generation**: Create JWT tokens with custom permissions
- **Token Validation**: Verify and decode JWT tokens
- **Permission-based Access**: Role-based access control
- **Token Expiration**: Configurable token lifetimes

### 🔑 Secrets Management
- **HashiCorp Vault Integration**: Enterprise-grade secrets storage
- **Local Fallback**: File-based secrets for development
- **Secure Storage**: Encrypted secret storage and retrieval
- **Access Control**: Permission-based secret access

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Trading Bot   │    │   MLOps Service │    │  Notification   │
│                 │    │                 │    │   Service       │
│  ┌────────────┐ │    │  ┌────────────┐ │    │  ┌────────────┐ │
│  │ JWT Token  │ │◄──►│  │ JWT Token  │ │◄──►│  │ JWT Token  │ │
│  └────────────┘ │    │  └────────────┘ │    │  └────────────┘ │
│                 │    │                 │    │                 │
│  ┌────────────┐ │    │  ┌────────────┐ │    │  ┌────────────┐ │
│  │ mTLS Cert  │ │◄──►│  │ mTLS Cert  │ │◄──►│  │ mTLS Cert  │ │
│  └────────────┘ │    │  └────────────┘ │    │  └────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                    ┌─────────────────────┐
                    │ Security Service    │
                    │                     │
                    │  ┌────────────────┐ │
                    │  │ Certificate    │ │
                    │  │ Authority      │ │
                    │  └────────────────┘ │
                    │                     │
                    │  ┌────────────────┐ │
                    │  │ JWT Token      │ │
                    │  │ Manager        │ │
                    │  └────────────────┘ │
                    │                     │
                    │  ┌────────────────┐ │
                    │  │ Secrets        │ │
                    │  │ Manager        │ │
                    │  └────────────────┘ │
                    │                     │
                    │  ┌────────────────┐ │
                    │  │ HashiCorp      │ │
                    │  │ Vault          │ │
                    │  └────────────────┘ │
                    └─────────────────────┘
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Edit configuration as needed
nano .env
```

### Running the Service

```bash
# Start the security service
python -m services.security_service.api

# Or run the demo
python services/security_service/demo.py
```

### Docker Deployment

```bash
# Build the container
docker build -t tradpal/security-service .

# Run the container
docker run -p 8002:8002 tradpal/security-service
```

## API Endpoints

### Health Check
```http
GET /health
```

### Issue mTLS Credentials
```http
POST /credentials/issue
Authorization: Bearer <admin-jwt-token>
Content-Type: application/json

{
  "service_name": "trading_bot_service"
}
```

### Generate JWT Token
```http
POST /tokens/generate
Content-Type: application/json

{
  "service_name": "mlops_service",
  "permissions": ["read", "write"]
}
```

### Validate JWT Token
```http
POST /tokens/validate
Authorization: Bearer <jwt-token>
```

### Store Secret
```http
POST /secrets/store
Authorization: Bearer <jwt-token>
Content-Type: application/json

{
  "path": "api_keys/binance",
  "data": {
    "api_key": "your-api-key",
    "secret_key": "your-secret-key"
  }
}
```

### Retrieve Secret
```http
GET /secrets/api_keys/binance
Authorization: Bearer <jwt-token>
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_MTLS` | Enable mTLS authentication | `true` |
| `ENABLE_JWT` | Enable JWT token management | `true` |
| `ENABLE_VAULT` | Enable HashiCorp Vault integration | `false` |
| `JWT_SECRET_KEY` | JWT signing secret | Auto-generated |
| `JWT_EXPIRATION_HOURS` | JWT token lifetime | `24` |
| `VAULT_URL` | HashiCorp Vault URL | - |
| `VAULT_TOKEN` | HashiCorp Vault token | - |
| `CERT_VALIDITY_DAYS` | Certificate validity period | `365` |

### Certificate Authority

The service automatically creates a Certificate Authority (CA) on first startup:

- **CA Certificate**: `cache/security/ca/ca_cert.pem`
- **CA Private Key**: `cache/security/ca/ca_key.pem`

### Service Credentials

Issued certificates are stored in:
- **Directory**: `cache/security/credentials/`
- **Format**: JSON files with certificate data

### Secrets Storage

Secrets are stored in:
- **Vault**: When enabled, secrets go to HashiCorp Vault
- **Local**: `cache/security/secrets/` (fallback)

## Usage Examples

### Python Client

```python
import httpx
import asyncio
from services.security_service import SecurityService, SecurityConfig

async def main():
    # Initialize service
    config = SecurityConfig()
    service = SecurityService(config)
    await service.start()

    # Issue mTLS credentials
    creds = await service.issue_service_credentials("my_service")
    print(f"Certificate: {creds.certificate[:50]}...")

    # Generate JWT token
    token = await service.generate_jwt_token(
        service_name="my_service",
        permissions=["read", "write"]
    )
    print(f"JWT Token: {token.token}")

    # Store secret
    await service.store_secret("my/secret", {"key": "value"})

    # Retrieve secret
    secret = await service.retrieve_secret("my/secret")
    print(f"Secret: {secret}")

    await service.stop()

asyncio.run(main())
```

### API Client

```python
import httpx

# Generate JWT token
response = httpx.post("http://localhost:8002/tokens/generate", json={
    "service_name": "api_client",
    "permissions": ["read"]
})
token = response.json()["token"]

# Use token for authenticated requests
headers = {"Authorization": f"Bearer {token}"}

# Store a secret
httpx.post("http://localhost:8002/secrets/store",
    headers=headers,
    json={
        "path": "api_keys/test",
        "data": {"key": "secret_value"}
    }
)

# Retrieve a secret
response = httpx.get("http://localhost:8002/secrets/api_keys/test",
    headers=headers
)
secret = response.json()["data"]
```

## Security Best Practices

### Certificate Management
- **Regular Rotation**: Rotate certificates before expiration
- **Secure Storage**: Store private keys securely (HSM recommended for production)
- **Revocation**: Implement certificate revocation lists (CRL)

### Token Security
- **Short Lifetimes**: Use short JWT token lifetimes
- **Refresh Tokens**: Implement refresh token patterns for long sessions
- **Secure Transmission**: Always use HTTPS for token transmission

### Secrets Management
- **Vault Integration**: Use HashiCorp Vault for production deployments
- **Access Logging**: Log all secret access for audit trails
- **Encryption**: Ensure secrets are encrypted at rest and in transit

## Testing

```bash
# Run unit tests
python -m pytest tests.py -v

# Run integration tests
python -m pytest tests.py::TestSecurityServiceIntegration -v

# Run with coverage
python -m pytest tests.py --cov=service --cov-report=html
```

## Monitoring

The service provides health check endpoints and integrates with:

- **Prometheus**: Metrics collection
- **Grafana**: Dashboard visualization
- **ELK Stack**: Log aggregation and analysis

## Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  security-service:
    build: ./services/security_service
    ports:
      - "8002:8002"
    volumes:
      - ./cache/security:/app/cache/security
    environment:
      - ENABLE_VAULT=true
      - VAULT_URL=http://vault:8200
    depends_on:
      - vault
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: security-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: security-service
  template:
    metadata:
      labels:
        app: security-service
    spec:
      containers:
      - name: security-service
        image: tradpal/security-service:latest
        ports:
        - containerPort: 8002
        env:
        - name: ENABLE_VAULT
          value: "true"
        volumeMounts:
        - name: security-storage
          mountPath: /app/cache/security
      volumes:
      - name: security-storage
        persistentVolumeClaim:
          claimName: security-pvc
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.</content>
<parameter name="filePath">/Users/danielsadowski/VSCodeProjects/tradpal_indicator/services/security_service/README.md