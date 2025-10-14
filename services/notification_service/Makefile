# Notification Service Makefile

.PHONY: help build run test clean docker-build docker-run docker-stop k8s-deploy k8s-delete lint format type-check

# Default target
help: ## Show this help message
	@echo "Notification Service Management Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development commands
install: ## Install dependencies
	pip install -r requirements.txt

run: ## Run the notification service locally
	python -m uvicorn services.notification_service.api:app --host 0.0.0.0 --port 8001 --reload

run-detached: ## Run the service in background
	python -m uvicorn services.notification_service.api:app --host 0.0.0.0 --port 8001 --reload &

demo: ## Run the demo script
	python services/notification_service/demo.py

# Testing commands
test: ## Run all tests
	pytest services/notification_service/tests.py -v

test-unit: ## Run unit tests only
	pytest services/notification_service/tests.py::TestNotificationService -v

test-integration: ## Run integration tests
	pytest services/notification_service/tests.py::TestNotificationServiceIntegration -v

test-performance: ## Run performance tests
	pytest services/notification_service/tests.py::TestNotificationServicePerformance -v -m performance

test-coverage: ## Run tests with coverage
	pytest services/notification_service/tests.py --cov=services.notification_service --cov-report=html --cov-report=term

# Code quality commands
lint: ## Run linting
	flake8 services/notification_service --max-line-length=100 --extend-ignore=E203,W503

format: ## Format code with black
	black services/notification_service

sort-imports: ## Sort imports with isort
	isort services/notification_service

type-check: ## Run type checking with mypy
	mypy services/notification_service

quality: lint format sort-imports type-check ## Run all code quality checks

# Docker commands
docker-build: ## Build Docker image
	docker build -f services/notification_service/Dockerfile -t tradpal/notification-service:latest .

docker-run: ## Run with Docker Compose
	docker-compose -f services/notification_service/docker-compose.yml up -d

docker-stop: ## Stop Docker Compose services
	docker-compose -f services/notification_service/docker-compose.yml down

docker-logs: ## Show Docker logs
	docker-compose -f services/notification_service/docker-compose.yml logs -f notification-service

docker-clean: ## Clean Docker resources
	docker-compose -f services/notification_service/docker-compose.yml down -v --rmi all

# Kubernetes commands
k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f services/notification_service/k8s-deployment.yaml

k8s-delete: ## Delete from Kubernetes
	kubectl delete -f services/notification_service/k8s-deployment.yaml

k8s-status: ## Check Kubernetes deployment status
	kubectl get pods -l app=notification-service
	kubectl get svc -l app=notification-service

k8s-logs: ## Show Kubernetes logs
	kubectl logs -l app=notification-service --tail=100 -f

# Monitoring commands
monitoring-up: ## Start monitoring stack
	docker-compose -f services/notification_service/docker-compose.yml up -d prometheus grafana

monitoring-down: ## Stop monitoring stack
	docker-compose -f services/notification_service/docker-compose.yml down prometheus grafana

# Utility commands
clean: ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/

clean-logs: ## Clean log files
	rm -f logs/notification_service.log*

# Health check
health: ## Check service health
	curl -f http://localhost:8001/health || echo "Service not running"

# API documentation
docs: ## Open API documentation
	@echo "API documentation available at: http://localhost:8001/docs"
	@echo "Alternative docs at: http://localhost:8001/redoc"

# Environment setup
setup-env: ## Setup environment file
	cp services/notification_service/.env.example services/notification_service/.env
	@echo "Please edit .env file with your configuration"

# Build and deploy shortcuts
build-deploy: docker-build k8s-deploy ## Build and deploy to Kubernetes

full-test: test quality ## Run full test suite and quality checks

# Development server with auto-reload
dev: ## Start development server
	uvicorn services.notification_service.api:app --reload --host 0.0.0.0 --port 8001 --log-level info