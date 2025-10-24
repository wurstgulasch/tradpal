# TradPal API Gateway Service

## Overview
The API Gateway Service provides centralized routing, authentication, and load balancing for all TradPal microservices.

## Features
- **Service Routing**: Centralized API routing to all microservices
- **Authentication**: JWT-based authentication and authorization
- **Load Balancing**: Intelligent load distribution across service instances
- **Rate Limiting**: Configurable rate limiting and throttling
- **Health Monitoring**: Service health checks and monitoring
- **Security**: Request validation and security headers

## API Endpoints

### Authentication
- `POST /auth/verify` - Verify JWT tokens
- `POST /auth/login` - User authentication

### Service Routing
- `GET /api/{service}/{endpoint}` - Route requests to services
- `POST /api/{service}/{endpoint}` - Route requests to services
- `PUT /api/{service}/{endpoint}` - Route requests to services
- `DELETE /api/{service}/{endpoint}` - Route requests to services

### Health & Discovery
- `GET /health` - Gateway health status
- `GET /api/health` - All services health status
- `GET /api/discovery` - Service discovery information

## Configuration
The service uses the following configuration from `config/service_settings.py`:
- `API_GATEWAY_URL`: Service URL (default: http://localhost:8000)

## Dependencies
- fastapi: Web framework
- uvicorn: ASGI server
- aiohttp: Async HTTP client
- python-jose: JWT handling
- redis: Caching and session management

## Usage
```python
from services.infrastructure_service.api_gateway_service.client import APIGatewayClient

async with APIGatewayClient() as client:
    # Authenticate
    auth_result = await client.authenticate(token)

    # Route request
    response = await client.route_request("trading", "/signals", "GET")

    # Check health
    health = await client.get_service_health()
```

## Development
```bash
# Start the service
python main.py

# Run tests
pytest tests/
```

## Architecture
The API Gateway acts as the single entry point for all client requests, providing:
- Request validation and sanitization
- Authentication and authorization
- Service discovery and routing
- Response aggregation and transformation
- Error handling and logging