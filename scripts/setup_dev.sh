#!/bin/bash
# TradPal Development Environment Setup Script

set -e

echo "🚀 Setting up TradPal Development Environment"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker Compose is available
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    print_error "Docker Compose is not installed. Please install Docker Compose and try again."
    print_status "On macOS: brew install docker-compose"
    exit 1
fi

print_status "Docker Compose available ✓"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p cache logs output data config

# Check if conda environment exists
if ! conda env list | grep -q "tradpal-env"; then
    print_warning "tradpal-env conda environment not found."
    print_status "Creating conda environment..."
    conda env create -f environment.yml --name tradpal-env
    print_success "Conda environment created"
else
    print_status "Conda environment tradpal-env already exists ✓"
fi

# Activate conda environment and install dependencies
print_status "Activating conda environment and installing dependencies..."
eval "$(conda shell.bash hook)"
conda activate tradpal-env

# Install/update pip dependencies
pip install -e .

print_success "Dependencies installed"

# Build Docker images
print_status "Building Docker images..."
$DOCKER_COMPOSE_CMD -f docker-compose.dev.yml build

print_success "Docker images built"

# Start services
print_status "Starting development services..."
$DOCKER_COMPOSE_CMD -f docker-compose.dev.yml up -d

# Wait for services to be healthy
print_status "Waiting for services to be ready..."
sleep 10

# Check service health
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    print_success "Data Service is healthy ✓"
else
    print_warning "Data Service health check failed"
fi

if curl -f http://localhost:8002/health > /dev/null 2>&1; then
    print_success "Backtesting Service is healthy ✓"
else
    print_warning "Backtesting Service health check failed"
fi

print_success "TradPal Development Environment Setup Complete!"
echo ""
echo "🎯 Available Services:"
echo "   📊 Data Service:        http://localhost:8001"
echo "   📈 Backtesting Service: http://localhost:8002"
echo "   🎨 Web UI (optional):   http://localhost:8501"
echo ""
echo "💡 Useful Commands:"
echo "   • Start services:    docker-compose -f docker-compose.dev.yml up -d"
echo "   • Stop services:     docker-compose -f docker-compose.dev.yml down"
echo "   • View logs:         docker-compose -f docker-compose.dev.yml logs -f"
echo "   • Run tests:         make test"
echo "   • Run backtest:      make backtest"
echo ""
echo "📚 Documentation: docs/README.md"
echo ""
echo "Happy coding! 🎉"