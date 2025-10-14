# MLOps Service Management Commands

.PHONY: help install run demo test clean build deploy

# Default target
help: ## Show this help message
	@echo "MLOps Service Management Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing MLOps service dependencies..."
	pip install -r requirements.txt

run: ## Run the MLOps service locally
	@echo "Starting MLOps service..."
	python -m api

run-detached: ## Run the service in background
	@echo "Starting MLOps service in background..."
	nohup python -m api > mlops.log 2>&1 &

demo: ## Run the demo script
	@echo "Running MLOps service demo..."
	python demo.py

test: ## Run tests
	@echo "Running MLOps service tests..."
	pytest tests.py -v

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	pytest --cov=. --cov-report=html tests.py

lint: ## Run linting
	@echo "Running linting..."
	flake8 . --max-line-length=100 --extend-ignore=E203,W503

format: ## Format code with black
	@echo "Formatting code..."
	black . --line-length=100

type-check: ## Run type checking with mypy
	@echo "Running type checking..."
	mypy . --ignore-missing-imports

quality: lint type-check test ## Run all code quality checks

build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t tradpal-mlops-service:latest .

run-docker: ## Run with Docker
	@echo "Running with Docker..."
	docker run -p 8001:8001 -p 5000:5000 tradpal-mlops-service:latest

deploy: ## Deploy to Kubernetes
	@echo "Deploying to Kubernetes..."
	kubectl apply -f k8s-deployment.yaml

clean: ## Clean up generated files
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

setup-env: ## Setup environment file
	@echo "Setting up environment file..."
	cp .env.example .env

health: ## Check service health
	@echo "Checking service health..."
	curl -s http://localhost:8001/health | python -m json.tool || echo "Service not running"

logs: ## Show service logs
	@echo "Showing service logs..."
	tail -f mlops.log

stop: ## Stop background service
	@echo "Stopping background service..."
	pkill -f "python -m api" || echo "No service running"