# Setup and Installation Guide - TradPal v3.0.1

This guide provides comprehensive setup instructions for TradPal's microservices architecture with Docker Compose, monitoring stack, and performance optimizations.

## 🔧 System Requirements

### Minimum Requirements (Light Profile)
- **OS**: Linux, macOS, Windows 10+
- **CPU**: 2 cores (Intel/AMD/Apple Silicon)
- **RAM**: 4 GB
- **Storage**: 5 GB SSD
- **Docker**: Docker Desktop 4.0+ or Docker Engine 20.10+
- **Python**: 3.10+ (for development)

### Recommended Requirements (Heavy Profile)
- **OS**: Linux (Ubuntu 22.04+) or macOS
- **CPU**: 4+ cores with AVX2 support
- **RAM**: 16 GB+
- **Storage**: 50 GB SSD
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional, for ML acceleration)
- **Docker**: Docker Desktop 4.0+ with Docker Compose V2

## 📦 Installation Methods

### Method 1: Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/wurstgulasch/tradpal.git
cd tradpal_indicator

# Start all services with monitoring stack
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**What's included:**
- ✅ All microservices (core, data, trading, backtesting, etc.)
- ✅ Redis for caching and event streaming
- ✅ Prometheus + Grafana monitoring stack
- ✅ Automatic service discovery
- ✅ Health checks and load balancing

### Method 2: Development Setup with Conda

```bash
# Clone repository
git clone https://github.com/wurstgulasch/tradpal.git
cd tradpal_indicator

# Create conda environment
conda env create -f environment.yml
conda activate tradpal-env

# Install Python dependencies
pip install -r requirements.txt

# Install optional ML dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For GPU support

# Start Redis (required for caching)
docker run -d --name redis-tradpal -p 6379:6379 redis:7-alpine

# Start monitoring stack (optional)
docker-compose -f docker-compose.monitoring.yml up -d
```

### Method 3: Kubernetes Deployment

```bash
# Using kubectl
kubectl apply -f infrastructure/deployment/kubernetes/

# Or using helm
helm install tradpal infrastructure/deployment/helm/tradpal/
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Trading Configuration
TRADING_SYMBOL=BTC/USDT
TRADING_TIMEFRAME=1h
INITIAL_CAPITAL=10000
MAX_POSITION_SIZE=0.1

# API Keys (required for live trading)
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# Database Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# ML Configuration
ML_MODELS_ENABLED=true
GPU_ACCELERATION_ENABLED=true
PYTORCH_DEVICE=cuda  # or cpu

# Performance Settings
MEMORY_OPTIMIZATION_ENABLED=true
PARALLEL_PROCESSING_ENABLED=true
MAX_WORKERS=0  # 0 = auto-detect

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true

# Security
JWT_SECRET_KEY=your_jwt_secret_key
MTLS_ENABLED=true

# Notifications
TELEGRAM_BOT_TOKEN=your_telegram_token
DISCORD_WEBHOOK_URL=your_discord_webhook
```

### Service Configuration

Each service has its own configuration in `config/settings.py`:

```python
# config/settings.py
from typing import Dict, Any

# Performance Profiles
PROFILES = {
    'light': {
        'ml_enabled': False,
        'indicators': ['sma', 'ema', 'rsi'],
        'max_memory': '2GB'
    },
    'heavy': {
        'ml_enabled': True,
        'indicators': ['all'],
        'max_memory': '8GB',
        'gpu_enabled': True
    }
}

# Service Endpoints
SERVICES = {
    'core': 'http://localhost:8001',
    'data': 'http://localhost:8002',
    'trading': 'http://localhost:8003',
    'backtesting': 'http://localhost:8004',
    'mlops': 'http://localhost:8005',
    'risk': 'http://localhost:8006',
    'notification': 'http://localhost:8007'
}
```

## 🚀 First Execution

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# View service logs
docker-compose logs core_service
docker-compose logs data_service

# Access web UI
open http://localhost:8501

# Access Grafana monitoring
open http://localhost:3000  # admin/admin
```

### Using Python Directly

```bash
# Activate environment
conda activate tradpal-env

# Run backtest
python main.py --mode backtest --profile light --start-date 2024-01-01

# Run with web UI
python main.py --mode webui --profile heavy

# Discovery mode (parameter optimization)
python main.py --mode discovery --generations 100 --population 50

# Live trading (paper mode first!)
python main.py --mode live --paper-trading true
```

### Performance Profiles

#### Light Profile
```bash
python main.py --profile light --mode backtest
```
- Minimal resource usage
- Basic indicators only
- No ML/AI features
- Fast execution

#### Heavy Profile
```bash
python main.py --profile heavy --mode backtest
```
- Full ML/AI capabilities
- All indicators and features
- GPU acceleration
- Maximum performance

## 🔍 Verification and Testing

### Health Checks

```bash
# Check all services
curl http://localhost:8000/health  # API Gateway

# Individual services
curl http://localhost:8001/health  # Core Service
curl http://localhost:8002/health  # Data Service
curl http://localhost:8003/health  # Trading Service

# Redis connectivity
docker exec redis-tradpal redis-cli ping
```

### Run Test Suite

```bash
# All tests (490 tests expected to pass)
pytest tests/ -v --tb=short

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Performance tests
pytest tests/performance/ -v
```

### Example Scripts

```bash
# Run performance demo
python examples/demo_performance_features.py

# Test data sources
python scripts/test_data_sources.py

# Benchmark performance
python scripts/performance_benchmark.py
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Docker Compose Issues

**Services won't start:**
```bash
# Check Docker resources
docker system df

# Clean up and restart
docker-compose down -v
docker-compose up --build --force-recreate
```

**Port conflicts:**
```bash
# Check port usage
lsof -i :8000-8007

# Change ports in docker-compose.yml
ports:
  - "8008:8001"  # Change host port
```

#### 2. Memory Issues

**Out of memory in containers:**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Or use light profile
docker-compose --profile light up
```

**Python memory errors:**
```python
# config/settings.py
MEMORY_OPTIMIZATION_ENABLED = True
CHUNK_SIZE = 500000  # Reduce chunk size
```

#### 3. GPU Issues

**CUDA not available:**
```bash
# Install NVIDIA Docker support
# Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Check GPU in container
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi
```

**PyTorch GPU issues:**
```python
# Force CPU usage
PYTORCH_DEVICE = 'cpu'
GPU_ACCELERATION_ENABLED = False
```

#### 4. Network Issues

**Services can't communicate:**
```bash
# Check service discovery
docker-compose exec api_gateway curl http://core_service:8001/health

# Check Docker network
docker network ls
docker network inspect tradpal_indicator_default
```

#### 5. Redis Connection Issues

**Redis not accessible:**
```bash
# Check Redis container
docker-compose logs redis

# Test connection
docker-compose exec redis redis-cli ping

# Reset Redis
docker-compose restart redis
```

### Logs and Debugging

```bash
# View all service logs
docker-compose logs -f

# Individual service logs
docker-compose logs -f core_service

# Application logs
tail -f logs/tradpal.log

# Enable debug logging
export LOG_LEVEL=DEBUG
docker-compose up -d
```

### Performance Troubleshooting

```bash
# Monitor resource usage
docker stats

# Profile Python performance
python -m cProfile main.py --mode backtest

# Memory profiling
python -m memory_profiler examples/demo_performance_features.py
```

## 🔄 Updates and Maintenance

### Update Docker Images

```bash
# Pull latest images
docker-compose pull

# Rebuild and restart
docker-compose up -d --build

# Update with zero downtime
docker-compose up -d --scale core_service=2 --scale data_service=2
docker-compose up -d --scale core_service=1 --scale data_service=1
```

### Backup and Restore

```bash
# Backup Redis data
docker-compose exec redis redis-cli --rdb /backup/redis.rdb

# Backup ML models
cp -r cache/ml_models /backup/

# Restore from backup
docker-compose exec redis redis-cli --rdb /backup/redis.rdb
```

### Database Migration

```bash
# Run database migrations
docker-compose exec core_service python scripts/migrate_database.py

# Validate migration
docker-compose exec core_service python scripts/validate_migration.py
```

## 🌐 Advanced Configuration

### Load Balancing

```yaml
# docker-compose.yml
services:
  core_service:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
```

### High Availability

```yaml
# Multiple Redis instances
services:
  redis_master:
    image: redis:7-alpine
  redis_replica:
    image: redis:7-alpine
    command: redis-server --replicaof redis_master 6379
```

### Monitoring Setup

```bash
# Access Grafana
open http://localhost:3000
# Default credentials: admin/admin

# Import dashboards
# - TradPal Service Metrics
# - ML Model Performance
# - Trading Performance
```

## 📊 Monitoring and Metrics

### Prometheus Metrics

```bash
# Query metrics
curl http://localhost:9090/api/v1/query?query=tradpal_service_up

# Service health metrics
curl http://localhost:8000/metrics
```

### Grafana Dashboards

- **Service Health**: Uptime, response times, error rates
- **Performance**: CPU, memory, disk usage
- **Trading Metrics**: P&L, win rate, drawdown
- **ML Metrics**: Model accuracy, training time, predictions

### Alerting

```yaml
# Alert rules in prometheus/alert_rules.yml
groups:
  - name: tradpal_alerts
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 5m
        labels:
          severity: critical
```

## 🆘 Support and Help

### Getting Help

1. **Documentation**: Check this guide and other docs in `/docs`
2. **GitHub Issues**: Search existing issues or create new ones
3. **Community**: Join Discord/Slack community
4. **Logs**: Collect logs before reporting issues

### Support Information

```bash
# System information
python scripts/collect_system_info.py

# Service status
docker-compose ps

# Recent logs
docker-compose logs --tail=100 > support_logs.txt

# Configuration dump
python scripts/dump_config.py > config_dump.txt
```

### Emergency Commands

```bash
# Stop all trading immediately
docker-compose exec trading_service python scripts/emergency_stop.py

# Reset all positions
docker-compose exec risk_service python scripts/reset_positions.py

# Clear all caches
docker-compose exec redis redis-cli FLUSHALL
```

## 📋 Quick Start Checklist

- [ ] Clone repository
- [ ] Install Docker Desktop
- [ ] Create `.env` file with API keys
- [ ] Run `docker-compose up -d`
- [ ] Access web UI at http://localhost:8501
- [ ] Run test suite: `pytest tests/`
- [ ] Start with paper trading
- [ ] Monitor performance in Grafana

## 🔐 Security Considerations

### API Keys
- Store in `.env` file (never commit)
- Use read-only API keys for live trading
- Rotate keys regularly
- Enable 2FA on exchange accounts

### Network Security
- Use internal Docker networks
- Enable mTLS for service communication
- Configure firewall rules
- Use VPN for remote access

### Data Security
- Encrypt sensitive data at rest
- Use secure connections (HTTPS/WSS)
- Implement audit logging
- Regular security updates

---

**Last Updated**: October 17, 2025
**Version**: v3.0.1
**Test Coverage**: 100% (490 tests passing)
**Docker Images**: Multi-stage builds with security scanning