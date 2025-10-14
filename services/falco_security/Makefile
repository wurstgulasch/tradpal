# TradPal Falco Runtime Security Makefile

.PHONY: help deploy undeploy status logs test clean

# Default target
help:
	@echo "TradPal Falco Runtime Security Management"
	@echo ""
	@echo "Available targets:"
	@echo "  deploy    - Deploy Falco to Kubernetes cluster"
	@echo "  undeploy  - Remove Falco from Kubernetes cluster"
	@echo "  status    - Show deployment status"
	@echo "  logs      - Show Falco logs"
	@echo "  test      - Test Falco rules locally"
	@echo "  clean     - Clean up temporary files"
	@echo "  help      - Show this help message"

# Deploy Falco
deploy:
	@echo "🚀 Deploying Falco Runtime Security..."
	./deploy.sh

# Undeploy Falco
undeploy:
	@echo "🛑 Undeploying Falco Runtime Security..."
	kubectl delete namespace tradpal-security --ignore-not-found=true

# Show deployment status
status:
	@echo "📊 Falco Deployment Status:"
	@echo ""
	@echo "Namespaces:"
	kubectl get namespaces | grep tradpal-security || echo "tradpal-security namespace not found"
	@echo ""
	@echo "Pods:"
	kubectl get pods -n tradpal-security 2>/dev/null || echo "No pods found in tradpal-security namespace"
	@echo ""
	@echo "ConfigMaps:"
	kubectl get configmaps -n tradpal-security 2>/dev/null || echo "No configmaps found in tradpal-security namespace"
	@echo ""
	@echo "ServiceAccounts:"
	kubectl get serviceaccounts -n tradpal-security 2>/dev/null || echo "No service accounts found in tradpal-security namespace"

# Show Falco logs
logs:
	@echo "📋 Falco Logs (last 50 lines):"
	kubectl logs -n tradpal-security -l app=falco --tail=50 -f

# Test Falco rules locally (requires falco binary)
test:
	@echo "🧪 Testing Falco Rules..."
	@if command -v falco &> /dev/null; then \
		falco -c falco.yaml -r tradpal_rules.yaml --validate; \
	else \
		echo "Falco binary not found. Using Python validator..."; \
		python3 validate_rules.py; \
	fi

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	rm -f *.log
	rm -rf __pycache__
	find . -name "*.pyc" -delete

# Validate Kubernetes manifests
validate:
	@echo "✅ Validating Kubernetes manifests..."
	kubeconform k8s/ || (echo "Install kubeconform: go install github.com/yannh/kubeconform@latest" && exit 1)

# Update Falco rules
update-rules:
	@echo "🔄 Updating Falco rules..."
	kubectl create configmap tradpal-falco-rules --from-file=tradpal_rules.yaml -n tradpal-security --dry-run=client -o yaml | kubectl apply -f -

# Restart Falco pods
restart:
	@echo "🔄 Restarting Falco pods..."
	kubectl rollout restart daemonset/falco -n tradpal-security