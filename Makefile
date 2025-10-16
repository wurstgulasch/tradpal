.PHONY: help setup dev-up dev-down dev-logs dev-build test test-verbose test-coverage test-fast test-integration test-unit backtest train-ml performance-benchmark clean format lint type-check quality-check docs serve-docs

# Development Environment
setup:
	@echo "🚀 Setting up TradPal development environment..."
	./setup_dev.sh

# Check Docker Compose command
DOCKER_COMPOSE := $(shell command -v docker-compose 2>/dev/null || echo "docker compose")
HAS_DOCKER := $(shell command -v docker 2>/dev/null && echo "yes" || echo "no")

# Development Environment
setup:
	@echo "🚀 Setting up TradPal development environment..."
	@if [ "$(HAS_DOCKER)" = "yes" ]; then \
		./setup_dev.sh; \
	else \
		./setup_local.sh; \
	fi

setup-local:
	@echo "🚀 Setting up local development environment..."
	./setup_local.sh

setup-docker:
	@echo "🚀 Setting up Docker development environment..."
	./setup_dev.sh

dev-up:
	@if [ "$(HAS_DOCKER)" = "yes" ]; then \
		echo "🔄 Starting development services..."; \
		$(DOCKER_COMPOSE) -f docker-compose.dev.yml up -d; \
	else \
		echo "❌ Docker not available. Use 'make setup-local' for local development."; \
		exit 1; \
	fi

dev-down:
	@if [ "$(HAS_DOCKER)" = "yes" ]; then \
		echo "🛑 Stopping development services..."; \
		$(DOCKER_COMPOSE) -f docker-compose.dev.yml down; \
	else \
		echo "❌ Docker not available."; \
		exit 1; \
	fi

dev-logs:
	@if [ "$(HAS_DOCKER)" = "yes" ]; then \
		echo "📋 Showing service logs..."; \
		$(DOCKER_COMPOSE) -f docker-compose.dev.yml logs -f; \
	else \
		echo "❌ Docker not available."; \
		exit 1; \
	fi

dev-build:
	@if [ "$(HAS_DOCKER)" = "yes" ]; then \
		echo "🔨 Building development images..."; \
		$(DOCKER_COMPOSE) -f docker-compose.dev.yml build --no-cache; \
	else \
		echo "❌ Docker not available."; \
		exit 1; \
	fi

dev-ui:
	@if [ "$(HAS_DOCKER)" = "yes" ]; then \
		echo "🎨 Starting development with UI..."; \
		$(DOCKER_COMPOSE) -f docker-compose.dev.yml --profile ui up -d; \
	else \
		echo "❌ Docker not available."; \
		exit 1; \
	fi

# Testing
test:
	@echo "🧪 Running all tests..."
	rm -rf cache/*
	pytest tests/ -v --tb=short

test-verbose:
	@echo "🧪 Running tests with verbose output..."
	pytest tests/ -v -s

test-coverage:
	@echo "📊 Running tests with coverage..."
	pytest --cov=services --cov=config --cov=integrations --cov-report=html --cov-report=term-missing

test-fast:
	@echo "⚡ Running fast tests only..."
	pytest -m "not slow" --tb=short

test-integration:
	@echo "🔗 Running integration tests..."
	pytest -m integration -v

test-unit:
	@echo "🧩 Running unit tests..."
	pytest -m unit -v

test-services:
	@echo "🔧 Testing service integrations..."
	pytest tests/test_service_integration.py -v

# Trading Operations
backtest:
	@echo "📈 Running backtest..."
	python main.py backtest --data-source kaggle --start-date 2020-01-01 --end-date 2021-12-31

backtest-all:
	@echo "📊 Running comprehensive backtests..."
	python scripts/test_adaptive_risk.py

train-ml:
	@echo "🤖 Training ML models..."
	python scripts/train_ml_model.py

performance-benchmark:
	@echo "⚡ Running performance benchmarks..."
	python scripts/performance_benchmark.py

# Code Quality
format:
	@echo "🎨 Formatting code..."
	black services/ config/ integrations/ scripts/ tests/

lint:
	@echo "🔍 Linting code..."
	flake8 services/ config/ integrations/ scripts/ tests/

type-check:
	@echo "🔍 Type checking..."
	mypy services/ config/ integrations/ scripts/

quality-check: format lint type-check
	@echo "✅ Code quality checks completed"

# Documentation
docs:
	@echo "📚 Building documentation..."
	cd docs && make html

serve-docs:
	@echo "🌐 Serving documentation..."
	cd docs && python -m http.server 8000

# Maintenance
clean:
	@echo "🧹 Cleaning up..."
	rm -rf .pytest_cache/ htmlcov/ .coverage .mypy_cache/
	rm -rf cache/* logs/* output/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	docker system prune -f

clean-all: clean
	@echo "🧹 Deep cleaning..."
	@if [ "$(HAS_DOCKER)" = "yes" ]; then \
		$(DOCKER_COMPOSE) -f docker-compose.dev.yml down -v; \
		docker system prune -a -f; \
	else \
		echo "Docker not available, skipping container cleanup"; \
	fi

# Health Checks
health-check:
	@echo "🏥 Checking service health..."
	@echo "Data Service:"
	@curl -s http://localhost:8001/health | head -1 || echo "❌ Data Service not responding"
	@echo "Backtesting Service:"
	@curl -s http://localhost:8002/health | head -1 || echo "❌ Backtesting Service not responding"

# Development Helpers
shell-data:
	@if [ "$(HAS_DOCKER)" = "yes" ]; then \
		echo "🐚 Opening shell in data service..."; \
		$(DOCKER_COMPOSE) -f docker-compose.dev.yml exec data-service bash; \
	else \
		echo "❌ Docker not available."; \
		exit 1; \
	fi

shell-backtest:
	@if [ "$(HAS_DOCKER)" = "yes" ]; then \
		echo "🐚 Opening shell in backtesting service..."; \
		$(DOCKER_COMPOSE) -f docker-compose.dev.yml exec backtesting-service bash; \
	else \
		echo "❌ Docker not available."; \
		exit 1; \
	fi

logs-data:
	@if [ "$(HAS_DOCKER)" = "yes" ]; then \
		echo "📋 Data service logs..."; \
		$(DOCKER_COMPOSE) -f docker-compose.dev.yml logs -f data-service; \
	else \
		echo "❌ Docker not available."; \
		exit 1; \
	fi

logs-backtest:
	@if [ "$(HAS_DOCKER)" = "yes" ]; then \
		echo "📋 Backtesting service logs..."; \
		$(DOCKER_COMPOSE) -f docker-compose.dev.yml logs -f backtesting-service; \
	else \
		echo "❌ Docker not available."; \
		exit 1; \
	fi

# Help
help:
	@echo "🚀 TradPal Development Commands"
	@echo "==============================="
	@echo ""
	@echo "🐳 Development Environment:"
	@echo "  setup           - Initial development setup"
	@echo "  dev-up          - Start development services"
	@echo "  dev-down        - Stop development services"
	@echo "  dev-logs        - Show service logs"
	@echo "  dev-build       - Rebuild development images"
	@echo "  dev-ui          - Start with web UI"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test            - Run all tests"
	@echo "  test-verbose    - Run tests with verbose output"
	@echo "  test-coverage   - Run tests with coverage"
	@echo "  test-fast       - Run fast tests only"
	@echo "  test-integration- Run integration tests"
	@echo "  test-unit       - Run unit tests"
	@echo "  test-services   - Test service integrations"
	@echo ""
	@echo "📈 Trading Operations:"
	@echo "  backtest        - Run single backtest"
	@echo "  backtest-all    - Run comprehensive backtests"
	@echo "  train-ml        - Train ML models"
	@echo "  performance-benchmark - Run performance tests"
	@echo ""
	@echo "🔧 Code Quality:"
	@echo "  format          - Format code with black"
	@echo "  lint            - Lint code with flake8"
	@echo "  type-check      - Type check with mypy"
	@echo "  quality-check   - Run all quality checks"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  docs            - Build documentation"
	@echo "  serve-docs      - Serve documentation locally"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  clean           - Clean test artifacts"
	@echo "  clean-all       - Deep clean everything"
	@echo "  health-check    - Check service health"
	@echo ""
	@echo "🐚 Development Helpers:"
	@echo "  shell-data      - Open shell in data service"
	@echo "  shell-backtest  - Open shell in backtesting service"
	@echo "  logs-data       - Show data service logs"
	@echo "  logs-backtest   - Show backtesting service logs"
	@echo ""
	@echo "📖 For more info, see docs/README.md"