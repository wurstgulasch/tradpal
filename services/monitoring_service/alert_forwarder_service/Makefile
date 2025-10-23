# TradPal Alert Forwarder Makefile

.PHONY: help install run test clean deploy undeploy logs docker-build docker-run docker-test docker-compose-up docker-compose-down

# Default target
help:
	@echo "TradPal Alert Forwarder Service"
	@echo ""
	@echo "Available targets:"
	@echo "  install           - Install Python dependencies"
	@echo "  run               - Run the alert forwarder locally"
	@echo "  test              - Run unit tests"
	@echo "  test-integration  - Run integration tests"
	@echo "  clean             - Clean up temporary files"
	@echo "  docker-build      - Build Docker image"
	@echo "  docker-run        - Run in Docker container"
	@echo "  docker-test       - Test in Docker container"
	@echo "  docker-compose-up - Start with docker-compose (includes test services)"
	@echo "  docker-compose-down - Stop docker-compose services"
	@echo "  deploy            - Deploy to Kubernetes"
	@echo "  undeploy          - Remove from Kubernetes"
	@echo "  logs              - Show Kubernetes logs"
	@echo "  status            - Show deployment status"
	@echo "  validate          - Validate configuration"
	@echo "  validate-all      - Validate with comprehensive script"
	@echo "  test-alert        - Create test alert file"
	@echo "  help              - Show this help message"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# Run locally
run:
	@echo "🚀 Starting Alert Forwarder..."
	python cli.py run

# Test the service
test:
	@echo "🧪 Running unit tests..."
	python -m pytest tests.py -v

# Integration tests
test-integration:
	@echo "🔗 Running integration tests..."
	python tests.py

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	rm -f *.log *.json
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	docker system prune -f

# Docker targets
docker-build:
	@echo "🏗️ Building Docker image..."
	docker build -t tradpal/alert-forwarder:latest .

docker-run:
	@echo "🐳 Running in Docker container..."
	docker run --rm --name alert-forwarder \
		-v /var/log/falco:/var/log/falco:ro \
		-e TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN} \
		-e TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID} \
		-e DISCORD_WEBHOOK_URL=${DISCORD_WEBHOOK_URL} \
		tradpal/alert-forwarder:latest

docker-test:
	@echo "🧪 Testing in Docker container..."
	docker run --rm -v $(PWD):/app -w /app \
		python:3.10-slim \
		sh -c "pip install pytest pydantic && python -m pytest tests.py -v"

# Docker Compose targets
docker-compose-up:
	@echo "🚀 Starting services with docker-compose..."
	docker-compose up -d
	@echo "⏳ Waiting for services to be ready..."
	sleep 10
	docker-compose logs alert-forwarder

docker-compose-down:
	@echo "🛑 Stopping docker-compose services..."
	docker-compose down -v

# Deploy to Kubernetes
deploy:
	@echo "🚀 Deploying Alert Forwarder to Kubernetes..."
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/deployment.yaml
	@echo "⏳ Waiting for deployment to be ready..."
	kubectl wait --for=condition=available --timeout=300s deployment/alert-forwarder -n tradpal-security

# Undeploy from Kubernetes
undeploy:
	@echo "🛑 Undeploying Alert Forwarder..."
	kubectl delete -f k8s/deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/configmap.yaml --ignore-not-found=true

# Show logs
logs:
	@echo "📋 Alert Forwarder Logs:"
	kubectl logs -n tradpal-security -l app=alert-forwarder -f

# Show status
status:
	@echo "📊 Alert Forwarder Status:"
	@echo ""
	@echo "Deployments:"
	kubectl get deployments -n tradpal-security -l app=alert-forwarder
	@echo ""
	@echo "Pods:"
	kubectl get pods -n tradpal-security -l app=alert-forwarder
	@echo ""
	@echo "ConfigMaps:"
	kubectl get configmaps -n tradpal-security -l app=tradpal
	@echo ""
	@echo "Services:"
	kubectl get services -n tradpal-security -l app=alert-forwarder

# Validate configuration
validate:
	@echo "✅ Validating configuration..."
	python -c "
	import os
	os.environ.setdefault('FALCO_LOG_PATH', '/var/log/falco/falco.log')
	from config import AlertConfig
	try:
		config = AlertConfig.from_env()
		print(f'✅ Config loaded successfully')
		print(f'   Log path: {config.falco_log_path}')
		print(f'   Min priority: {config.alert_min_priority}')
		print(f'   Batching: {config.alert_batching}')
		print(f'   Rate limit: {config.alert_rate_limit}/min')
		print(f'   Telegram: {config.telegram_enabled}')
		print(f'   Discord: {config.discord_enabled}')
	except Exception as e:
		print(f'❌ Config validation failed: {e}')
		exit(1)
	"

# Validate with comprehensive script
validate-all:
	@echo "🔍 Running comprehensive validation..."
	python validate.py

# Create test alert
test-alert:
	@echo "🧪 Creating test alert..."
	@echo '{"output": "Test security alert - manual trigger", "priority": "WARNING", "rule": "Manual Test Alert", "time": "'$$(date -Iseconds)'", "hostname": "test-host", "tags": ["test", "manual"]}' >> /tmp/tradpal_test_alert.json
	@echo "✅ Test alert created: /tmp/tradpal_test_alert.json"