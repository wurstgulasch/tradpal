# Discovery Service Makefile
# Automates build, test, and deployment tasks

.PHONY: help install test run demo clean docker-build docker-run k8s-deploy k8s-undeploy lint format type-check

# Default target
help: ## Show this help message
	@echo "Discovery Service - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Development setup
install: ## Install Python dependencies
	pip install -r requirements.txt
	pip install -r services/discovery_service/requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -r services/discovery_service/requirements.txt
	pip install pytest pytest-asyncio black isort mypy

# Testing
test: ## Run unit tests
	cd services/discovery_service && python -m pytest tests.py -v

test-coverage: ## Run tests with coverage
	cd services/discovery_service && python -m pytest tests.py --cov=. --cov-report=html

test-integration: ## Run integration tests
	cd services/discovery_service && python -m pytest tests.py -k "integration" -v

# Code quality
lint: ## Run linting
	cd services/discovery_service && python -m flake8 . --max-line-length=100

format: ## Format code with black
	cd services/discovery_service && python -m black . --line-length=100

sort-imports: ## Sort imports with isort
	cd services/discovery_service && python -m isort . --profile=black

type-check: ## Run type checking with mypy
	cd services/discovery_service && python -m mypy . --ignore-missing-imports

quality: lint format sort-imports type-check ## Run all code quality checks

# Running
run: ## Run the service locally
	cd services/discovery_service && python -m api

run-dev: ## Run in development mode with auto-reload
	cd services/discovery_service && uvicorn api:app --reload --host 0.0.0.0 --port 8003

# Demo
demo: ## Run demo script
	cd services/discovery_service && python demo.py

demo-single: ## Run single optimization demo
	cd services/discovery_service && python demo.py --single

demo-multiple: ## Run multiple optimizations demo
	cd services/discovery_service && python demo.py --multiple

demo-events: ## Run event-driven demo
	cd services/discovery_service && python demo.py --events

# Docker
docker-build: ## Build Docker image
	docker build -t tradpal/discovery-service:latest ./services/discovery_service

docker-run: ## Run Docker container
	docker run -p 8003:8003 --name discovery-service tradpal/discovery-service:latest

docker-stop: ## Stop Docker container
	docker stop discovery-service || true
	docker rm discovery-service || true

docker-logs: ## Show Docker container logs
	docker logs -f discovery-service

docker-clean: ## Remove Docker image
	docker rmi tradpal/discovery-service:latest || true

# Kubernetes
k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f services/discovery_service/k8s-deployment.yaml

k8s-undeploy: ## Remove from Kubernetes
	kubectl delete -f services/discovery_service/k8s-deployment.yaml

k8s-status: ## Check Kubernetes deployment status
	kubectl get pods -l app=discovery-service -n tradpal
	kubectl get svc -l app=discovery-service -n tradpal

k8s-logs: ## Show Kubernetes pod logs
	kubectl logs -l app=discovery-service -n tradpal --tail=100 -f

# Cleanup
clean: ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf output/discovery_service_demo_*.json

clean-docker: ## Clean Docker resources
	docker system prune -f
	docker volume prune -f

clean-all: clean clean-docker ## Clean everything

# CI/CD
ci: install-dev quality test ## Run CI pipeline locally

# Performance testing
perf-test: ## Run performance tests
	cd services/discovery_service && python -m pytest tests.py -k "performance" -v --tb=short

# Documentation
docs: ## Generate documentation
	cd services/discovery_service && python -c "import api; print('API documentation available at http://localhost:8003/docs')"

# Health check
health: ## Check service health
	curl -f http://localhost:8003/health || echo "Service not running"

# Development workflow
dev-setup: install-dev quality ## Setup development environment

dev-test: test demo-single ## Run basic tests and demo

# Production deployment
deploy: docker-build k8s-deploy ## Full deployment pipeline

# Monitoring
monitor: ## Monitor service metrics
	@echo "Monitoring Discovery Service..."
	@echo "Health: $(shell curl -s http://localhost:8003/health | jq .status 2>/dev/null || echo 'N/A')"
	@echo "Active optimizations: $(shell curl -s http://localhost:8003/optimize/active | jq length 2>/dev/null || echo 'N/A')"

# Backup
backup: ## Backup service data
	tar -czf backup_discovery_service_$(shell date +%Y%m%d_%H%M%S).tar.gz services/discovery_service/

# Environment info
info: ## Show environment information
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "Docker version: $(shell docker --version 2>/dev/null || echo 'Docker not installed')"
	@echo "Kubectl version: $(shell kubectl version --client --short 2>/dev/null || echo 'Kubectl not installed')"