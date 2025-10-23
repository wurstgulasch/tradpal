# TradPal Data Service Makefile
# Common development tasks for the data service

.PHONY: help install run test clean docker-build docker-run logs health

# Default target
help:
	@echo "TradPal Data Service - Available commands:"
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
	@echo "  fetch-data   - Test data fetching from sources"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Run the service locally
run:
	@echo "🚀 Starting data service..."
	python -m services.data_service.main

# Run tests
test:
	@echo "🧪 Running tests..."
	python services/data_service/test_data_service.py

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
	docker build -t tradpal/data-service:latest .
	@echo "✅ Docker image built"

# Run service in Docker
docker-run:
	@echo "🐳 Running service in Docker..."
	docker run --rm -p 8001:8001 \
		-e REDIS_URL=redis://host.docker.internal:6379 \
		-e LOG_LEVEL=DEBUG \
		tradpal/data-service:latest

# Show service logs (when running in Docker)
logs:
	@echo "📋 Showing service logs..."
	docker logs -f tradpal-data-service

# Check service health
health:
	@echo "🏥 Checking service health..."
	@curl -s http://localhost:8001/health | python -m json.tool || echo "Service not responding"

# Format code
format:
	@echo "🎨 Formatting code..."
	black services/data_service/
	isort services/data_service/
	@echo "✅ Code formatted"

# Run linting
lint:
	@echo "🔍 Running linting checks..."
	flake8 services/data_service/ --max-line-length=88
	mypy services/data_service/
	@echo "✅ Linting completed"

# Test data fetching
fetch-data:
	@echo "📊 Testing data fetching..."
	curl -s "http://localhost:8001/api/data/fetch/BTC/USDT?timeframe=1h&limit=10" | head -20

# Test alternative data
test-alt-data:
	@echo "📈 Testing alternative data..."
	curl -s "http://localhost:8001/api/data/alternative/sentiment?symbol=BTC" | head -10

# Test market regime
test-regime:
	@echo "🎯 Testing market regime..."
	curl -s "http://localhost:8001/api/data/regime/BTC/USDT" | head -10

# Development setup
dev-setup: install
	@echo "🔧 Setting up development environment..."
	pre-commit install
	@echo "✅ Development environment ready"

# Run with hot reload (development)
dev:
	@echo "🔄 Starting service with hot reload..."
	uvicorn services.data_service.main:app --reload --host 0.0.0.0 --port 8001

# Run integration tests
test-integration:
	@echo "🔗 Running integration tests..."
	pytest tests/integration/test_data_service.py -v

# Generate API documentation
docs:
	@echo "📚 Generating API documentation..."
	python -c "from services.data_service.main import app; print('API docs available at: http://localhost:8001/docs')"

# Show service status
status:
	@echo "📊 Service Status:"
	@echo "  - Health: $(shell curl -s http://localhost:8001/health | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo 'unknown')"
	@echo "  - Version: $(shell python -c "from services.data_service import __version__; print(__version__)")"
	@echo "  - Data Sources: CCXT, Yahoo Finance, Kaggle"
	@echo "  - Alternative Data: Sentiment, On-chain, Economic"
	@echo "  - Market Regime: Clustering, ML Classification"

# Quick test all endpoints
quick-test:
	@echo "⚡ Running quick endpoint tests..."
	@echo "Health check:" && curl -s http://localhost:8001/health | jq '.status' || echo "Failed"
	@echo "Data sources:" && curl -s http://localhost:8001/api/data/sources | jq '.[]' || echo "Failed"
	@echo "Cache stats:" && curl -s http://localhost:8001/api/cache/stats | jq '.cache_hit_rate' || echo "Failed"