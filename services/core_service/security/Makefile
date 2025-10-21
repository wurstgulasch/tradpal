# Security Service Makefile

.PHONY: help install run demo test build clean docker-build docker-run

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install Python dependencies"
	@echo "  run         - Run the security service"
	@echo "  demo        - Run the security service demo"
	@echo "  test        - Run tests"
	@echo "  build       - Build Docker image"
	@echo "  docker-run  - Run Docker container"
	@echo "  clean       - Clean cache and temporary files"

# Install dependencies
install:
	pip install -r requirements.txt

# Run the service
run:
	python -m api

# Run demo
demo:
	python demo.py

# Run tests
test:
	python -m pytest tests.py -v

# Build Docker image
build:
	docker build -t tradpal/security-service .

# Run Docker container
docker-run:
	docker run -p 8002:8002 -v $(PWD)/cache:/app/cache tradpal/security-service

# Clean cache files
clean:
	rm -rf cache/security/*
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true