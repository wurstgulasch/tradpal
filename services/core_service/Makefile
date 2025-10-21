# TradPal Core Service Makefile
# Common development tasks for the core service

.PHONY: help install run test clean docker-build docker-run logs health

# Default target
help:
	@echo "TradPal Core Service - Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  run          - Run the service locally"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean cache and temporary files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run service in Docker"
	@echo "  logs         - Show service logs"
	@echo "  health       - Check service health"
	@echo "  format       - Format code with black"
	@echo "  lint         - Run linting checks"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Run the service locally
run:
	@echo "🚀 Starting core service..."
	python -m services.core_service.main

# Run tests
test:
	@echo "🧪 Running tests..."
	python services/core_service/test_core_service.py

# Clean cache and temporary files
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "✅ Cleanup completed"

# Build Docker image
docker-build:
	@echo "🏗️  Building Docker image..."
	docker build -t tradpal/core-service:latest .
	@echo "✅ Docker image built"

# Run service in Docker
docker-run:
	@echo "🐳 Running service in Docker..."
	docker run --rm -p 8000:8000 \
		-e REDIS_URL=redis://host.docker.internal:6379 \
		-e LOG_LEVEL=DEBUG \
		tradpal/core-service:latest

# Show service logs (when running in Docker)
logs:
	@echo "📋 Showing service logs..."
	docker logs -f tradpal-core-service

# Check service health
health:
	@echo "🏥 Checking service health..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Service not responding"

# Format code
format:
	@echo "🎨 Formatting code..."
	black services/core_service/
	isort services/core_service/
	@echo "✅ Code formatted"

# Run linting
lint:
	@echo "🔍 Running linting checks..."
	flake8 services/core_service/ --max-line-length=88
	mypy services/core_service/
	@echo "✅ Linting completed"

# Development setup
dev-setup: install
	@echo "🔧 Setting up development environment..."
	pre-commit install
	@echo "✅ Development environment ready"

# Run with hot reload (development)
dev:
	@echo "🔄 Starting service with hot reload..."
	uvicorn services.core_service.main:app --reload --host 0.0.0.0 --port 8000

# Run integration tests
test-integration:
	@echo "🔗 Running integration tests..."
	pytest tests/integration/test_core_service.py -v

# Generate API documentation
docs:
	@echo "📚 Generating API documentation..."
	python -c "from services.core_service.main import app; print('API docs available at: http://localhost:8000/docs')"

# Show service status
status:
	@echo "📊 Service Status:"
	@echo "  - Health: $(shell curl -s http://localhost:8000/health | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo 'unknown')"
	@echo "  - Version: $(shell python -c "from services.core_service import __version__; print(__version__)")"
	@echo "  - Uptime: Check logs for startup time"

# Quick test all components
quick-test:
	@echo "⚡ Running quick component tests..."
	python -c "
	import asyncio
	from services.core_service import CoreService
	asyncio.run(CoreService().health_check())
	print('✅ Core service healthy')
	"