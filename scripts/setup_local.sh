#!/bin/bash
# TradPal Local Development Setup Script (No Docker)

set -e

echo "🚀 Setting up TradPal Local Development Environment"
echo "==================================================="

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

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Miniconda/Anaconda first."
    print_status "Download: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_status "Conda is available ✓"

# Check if tradpal-env exists, create if not
if ! conda env list | grep -q "tradpal-env"; then
    print_status "Creating tradpal-env conda environment..."
    conda env create -f environment.yml --name tradpal-env
    print_success "Conda environment created"
else
    print_status "Conda environment tradpal-env already exists ✓"
fi

# Activate environment
print_status "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate tradpal-env

# Install/update pip dependencies
print_status "Installing/updating dependencies..."
pip install -e .

print_success "Dependencies installed"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p cache logs output data config

# Test basic imports
print_status "Testing basic imports..."
python -c "
import sys
sys.path.append('.')
try:
    from services.data_service.service import DataService
    from services.backtesting_service.service import AsyncBacktester
    print('✅ Core services import successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

print_success "Basic imports working"

# Test configuration
print_status "Testing configuration..."
python -c "
import sys
sys.path.append('.')
try:
    from config.settings import get_settings
    settings = get_settings()
    print(f'✅ Configuration loaded: {len(settings)} settings available')
    print(f'   • Symbol: {settings.get(\"symbol\", \"N/A\")}')
    print(f'   • Exchange: {settings.get(\"exchange\", \"N/A\")}')
    print(f'   • Timeframe: {settings.get(\"timeframe\", \"N/A\")}')
except Exception as e:
    print(f'❌ Configuration error: {e}')
    exit(1)
"

print_success "Configuration working"

print_success "TradPal Local Development Environment Setup Complete!"
echo ""
echo "🎯 Ready to develop!"
echo ""
echo "💡 Useful commands:"
echo "   • python main.py backtest --help    # Show backtest options"
echo "   • python -m pytest tests/ -v         # Run tests"
echo "   • python scripts/train_ml_model.py  # Train ML models"
echo ""
echo "📚 Documentation: docs/README.md"
echo ""
echo "Happy coding! 🎉"