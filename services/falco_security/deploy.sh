#!/bin/bash
# TradPal Falco Runtime Security Deployment Script

set -e

echo "🚀 Deploying Falco Runtime Security for TradPal..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check if we're connected to a Kubernetes cluster
if ! kubectl cluster-info &> /dev/null; then
    print_error "Not connected to a Kubernetes cluster"
    exit 1
fi

print_status "Connected to Kubernetes cluster"

# Create namespace
print_status "Creating tradpal-security namespace..."
kubectl apply -f k8s/namespace.yaml

# Deploy RBAC
print_status "Deploying RBAC resources..."
kubectl apply -f k8s/falco-rbac.yaml

# Deploy ConfigMaps
print_status "Deploying configuration..."
kubectl apply -f k8s/falco-configmaps.yaml

# Deploy Falco DaemonSet
print_status "Deploying Falco DaemonSet..."
kubectl apply -f k8s/falco-daemonset.yaml

# Wait for deployment to be ready
print_status "Waiting for Falco pods to be ready..."
kubectl wait --for=condition=ready pod -l app=falco -n tradpal-security --timeout=300s

# Check deployment status
print_status "Checking deployment status..."
kubectl get pods -n tradpal-security -l app=falco

# Verify Falco is running
print_status "Verifying Falco installation..."
kubectl logs -n tradpal-security -l app=falco --tail=20

print_status "✅ Falco Runtime Security deployment completed successfully!"
print_status ""
print_status "Next steps:"
print_status "1. Monitor Falco logs: kubectl logs -n tradpal-security -l app=falco -f"
print_status "2. Check alerts: kubectl logs -n tradpal-security -l app=falco | grep -i alert"
print_status "3. View Falco metrics: kubectl port-forward -n tradpal-security svc/falco-metrics 9090:9090"
print_warning ""
print_warning "Security Recommendations:"
print_warning "- Regularly review Falco alerts for suspicious activities"
print_warning "- Update Falco rules as new threats emerge"
print_warning "- Monitor resource usage of Falco pods"
print_warning "- Consider integrating with external alerting systems"