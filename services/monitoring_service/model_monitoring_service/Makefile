# Model Monitoring Service Makefile

.PHONY: help install run test clean docker-build docker-run

# Default target
help:
	@echo "Model Monitoring Service"
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  run         - Run the service locally"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean up generated files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run service in Docker"

# Install dependencies
install:
	pip install -r requirements.txt

# Run the service
run:
	python main.py

# Run tests
test:
	python test_monitoring.py

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf models/*.pkl
	rm -rf .pytest_cache

# Docker targets
docker-build:
	docker build -t tradpal/model-monitoring-service .

docker-run:
	docker run -p 8020:8020 \
		-e NOTIFICATION_SERVICE_URL=http://host.docker.internal:8010 \
		-v $(PWD)/models:/app/models \
		tradpal/model-monitoring-service

# Development with auto-reload
dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8020

# Health check
health:
	curl -s http://localhost:8020/health | python -m json.tool

# API documentation
docs:
	@echo "API Documentation: http://localhost:8020/docs"
	@echo "OpenAPI JSON: http://localhost:8020/openapi.json"