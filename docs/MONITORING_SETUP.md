# TradPal Monitoring Setup

## Overview

TradPal implements a comprehensive monitoring stack using Prometheus, Grafana, and AlertManager for observability across all microservices. The monitoring system provides real-time metrics, alerting, and visualization for system health and performance.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Services      │    │   Prometheus    │    │    Grafana      │
│   (Metrics)     │───▶│   (Collection)  │───▶│  (Visualization)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │  AlertManager   │
                    │   (Alerting)    │
                    └─────────────────┘
```

## Components

### Prometheus (Port 9090)

**Purpose**: Metrics collection and storage
**Configuration**: `infrastructure/monitoring/prometheus.yml`

Key features:
- Service discovery and metrics scraping
- Time-series database for metrics storage
- Query language (PromQL) for data analysis
- Alerting rules and conditions

### Grafana (Port 3000)

**Purpose**: Metrics visualization and dashboards
**Configuration**: Pre-configured dashboards for TradPal services

Key features:
- Interactive dashboards and panels
- Multiple data source support
- Alerting and notification integration
- User management and permissions

### AlertManager (Port 9093)

**Purpose**: Alert routing and notification
**Configuration**: `infrastructure/monitoring/alertmanager.yml`

Key features:
- Alert grouping and inhibition
- Notification routing (email, Slack, webhook)
- Alert silencing and maintenance
- Integration with external systems

## Service Metrics

### API Gateway Metrics

```python
# Request metrics
REQUEST_COUNT = Counter(
    'api_gateway_requests_total',
    'Total number of requests processed',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_gateway_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

# Service health
SERVICE_HEALTH_STATUS = Gauge(
    'api_gateway_service_health_status',
    'Health status of backend services',
    ['service_name']
)

# Rate limiting
RATE_LIMIT_EXCEEDED = Counter(
    'api_gateway_rate_limit_exceeded_total',
    'Total number of rate limit violations',
    ['client_id']
)
```

### Core Service Metrics

```python
# Indicator calculations
INDICATOR_CALCULATION_TIME = Histogram(
    'core_indicator_calculation_duration_seconds',
    'Time spent calculating indicators',
    ['indicator_type']
)

INDICATOR_CALCULATION_COUNT = Counter(
    'core_indicator_calculation_total',
    'Total number of indicator calculations',
    ['indicator_type', 'status']
)

# Memory usage
MEMORY_USAGE = Gauge(
    'core_memory_usage_bytes',
    'Current memory usage in bytes'
)

# Vectorization performance
VECTORIZATION_TIME = Histogram(
    'core_vectorization_duration_seconds',
    'Time spent on vectorized operations',
    ['operation_type']
)
```

### Trading Bot Metrics

```python
# Trading signals
SIGNAL_GENERATION_COUNT = Counter(
    'trading_signal_generation_total',
    'Total number of trading signals generated',
    ['signal_type', 'confidence_level']
)

SIGNAL_EXECUTION_TIME = Histogram(
    'trading_signal_execution_duration_seconds',
    'Time spent executing trading signals'
)

# Position management
POSITION_COUNT = Gauge(
    'trading_active_positions',
    'Number of active trading positions'
)

POSITION_PNL = Gauge(
    'trading_position_pnl',
    'Current P&L of positions',
    ['position_id']
)

# Risk metrics
PORTFOLIO_VALUE = Gauge(
    'trading_portfolio_value',
    'Current portfolio value'
)

DRAWDOWN_PERCENTAGE = Gauge(
    'trading_drawdown_percentage',
    'Current drawdown percentage'
)
```

### Backtesting Service Metrics

```python
# Backtest execution
BACKTEST_DURATION = Histogram(
    'backtest_execution_duration_seconds',
    'Time spent running backtests',
    ['strategy_name', 'timeframe']
)

BACKTEST_COUNT = Counter(
    'backtest_execution_total',
    'Total number of backtests executed',
    ['status']
)

# Performance metrics
SHARPE_RATIO = Gauge(
    'backtest_sharpe_ratio',
    'Sharpe ratio of backtest results',
    ['backtest_id']
)

MAX_DRAWDOWN = Gauge(
    'backtest_max_drawdown',
    'Maximum drawdown of backtest results',
    ['backtest_id']
)

TOTAL_RETURN = Gauge(
    'backtest_total_return',
    'Total return of backtest results',
    ['backtest_id']
)
```

### Data Service Metrics

```python
# Data fetching
DATA_FETCH_COUNT = Counter(
    'data_fetch_total',
    'Total number of data fetch operations',
    ['source', 'status']
)

DATA_FETCH_DURATION = Histogram(
    'data_fetch_duration_seconds',
    'Time spent fetching data',
    ['source']
)

# Cache performance
CACHE_HIT_RATIO = Gauge(
    'data_cache_hit_ratio',
    'Cache hit ratio (0.0 - 1.0)'
)

CACHE_SIZE = Gauge(
    'data_cache_size_bytes',
    'Current cache size in bytes'
)

# Data quality
DATA_QUALITY_SCORE = Gauge(
    'data_quality_score',
    'Data quality score (0.0 - 1.0)',
    ['source']
)
```

## Docker Compose Configuration

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/config.yml
    command:
      - '--config.file=/etc/alertmanager/config.yml'
      - '--storage.path=/alertmanager'

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  prometheus_data:
  grafana_data:
  redis_data:
```

## Prometheus Configuration

### prometheus.yml

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'api-gateway'
    static_configs:
      - targets: ['api-gateway:8000']
    metrics_path: '/metrics'

  - job_name: 'core-service'
    static_configs:
      - targets: ['core-service:8002']
    metrics_path: '/metrics'

  - job_name: 'data-service'
    static_configs:
      - targets: ['data-service:8001']
    metrics_path: '/metrics'

  - job_name: 'trading-bot'
    static_configs:
      - targets: ['trading-bot:8003']
    metrics_path: '/metrics'

  - job_name: 'backtesting-service'
    static_configs:
      - targets: ['backtesting-service:8004']
    metrics_path: '/metrics'

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
```

### Alert Rules

```yaml
groups:
  - name: tradpal_alerts
    rules:
      - alert: HighRequestLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High request latency detected"
          description: "95th percentile request latency is {{ $value }}s"

      - alert: ServiceDown
        expr: up == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "Service {{ $labels.job }} has been down for 5 minutes"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: LowMemory
        expr: (1 - system_memory_usage / system_memory_total) < 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low memory available"
          description: "Only {{ $value | humanizePercentage }} memory available"
```

## Grafana Dashboards

### Pre-configured Dashboards

1. **System Overview**
   - Service health status
   - Overall system metrics
   - Resource utilization

2. **API Gateway Dashboard**
   - Request throughput and latency
   - Rate limiting violations
   - Service health status

3. **Trading Performance**
   - Signal generation metrics
   - Position P&L tracking
   - Risk metrics (drawdown, Sharpe ratio)

4. **Backtesting Results**
   - Backtest execution times
   - Performance metrics comparison
   - Strategy effectiveness

5. **Data Service Dashboard**
   - Data fetch performance
   - Cache hit ratios
   - Data quality scores

### Custom Dashboard Creation

```json
{
  "dashboard": {
    "title": "TradPal Overview",
    "tags": ["tradpal", "overview"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      }
    ]
  }
}
```

## AlertManager Configuration

### alertmanager.yml

```yaml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@tradpal.com'
  smtp_auth_username: 'alerts@tradpal.com'
  smtp_auth_password: 'your-password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'email-notifications'
  routes:
  - match:
      severity: critical
    receiver: 'critical-notifications'

receivers:
- name: 'email-notifications'
  email_configs:
  - to: 'admin@tradpal.com'
    send_resolved: true

- name: 'critical-notifications'
  email_configs:
  - to: 'admin@tradpal.com'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#alerts'
    send_resolved: true
```

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available
- Ports 3000, 9090, 9093, 6379 available

### Quick Start

1. **Start the monitoring stack**:
```bash
cd infrastructure/monitoring
docker-compose up -d
```

2. **Verify services are running**:
```bash
docker-compose ps
```

3. **Access Grafana**:
   - URL: http://localhost:3000
   - Username: admin
   - Password: admin

4. **Access Prometheus**:
   - URL: http://localhost:9090
   - Query metrics: `up{job="api-gateway"}`

5. **Access AlertManager**:
   - URL: http://localhost:9093

### Service Integration

Each service exposes metrics at `/metrics` endpoint:

```python
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time

# Start metrics server
start_http_server(8000)

# Define metrics
REQUEST_COUNT = Counter('service_requests_total', 'Total requests', ['method'])
REQUEST_LATENCY = Histogram('service_request_latency', 'Request latency', ['method'])

# Use in code
start_time = time.time()
# ... your code ...
REQUEST_LATENCY.labels(method='GET').observe(time.time() - start_time)
REQUEST_COUNT.labels(method='GET').inc()
```

## Troubleshooting

### Common Issues

1. **Grafana not accessible**
   - Check if port 3000 is available
   - Verify Docker container is running
   - Check Grafana logs: `docker-compose logs grafana`

2. **Prometheus not scraping metrics**
   - Verify service endpoints are accessible
   - Check Prometheus configuration
   - Look for scrape errors in Prometheus UI

3. **Alerts not firing**
   - Verify alert rules syntax
   - Check AlertManager configuration
   - Test alert conditions manually in Prometheus

4. **High memory usage**
   - Adjust Prometheus retention time
   - Increase Docker memory limits
   - Implement metric filtering

### Debug Commands

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Query metrics
curl "http://localhost:9090/api/v1/query?query=up"

# Check AlertManager status
curl http://localhost:9093/api/v2/status

# View Grafana logs
docker-compose logs grafana
```

## Performance Optimization

### Prometheus Optimization

1. **Retention Policy**:
```yaml
storage:
  tsdb:
    retention.time: 30d  # Adjust based on disk space
```

2. **Scrape Intervals**:
```yaml
global:
  scrape_interval: 30s  # Increase for less critical metrics
```

3. **Metric Filtering**:
```yaml
metric_relabel_configs:
  - source_labels: [__name__]
    regex: 'go_.*'  # Drop Go runtime metrics
    action: drop
```

### Grafana Optimization

1. **Dashboard Refresh**: Set appropriate refresh intervals
2. **Query Optimization**: Use PromQL best practices
3. **Panel Limits**: Limit data points in panels

### AlertManager Optimization

1. **Grouping**: Group related alerts
2. **Inhibition**: Prevent alert storms
3. **Routing**: Route alerts to appropriate channels

## Security Considerations

### Production Deployment

1. **Authentication**: Enable Grafana authentication
2. **TLS**: Use HTTPS for all endpoints
3. **Network Security**: Restrict access to monitoring ports
4. **Secrets Management**: Store credentials securely

### Access Control

```yaml
# Grafana configuration
[auth]
enabled = true

[auth.anonymous]
enabled = false

[auth.basic]
enabled = true
```

## Integration with CI/CD

### Automated Deployment

```yaml
# .github/workflows/deploy-monitoring.yml
name: Deploy Monitoring
on:
  push:
    branches: [main]
    paths:
      - 'infrastructure/monitoring/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to production
        run: |
          docker-compose -f infrastructure/monitoring/docker-compose.yml up -d
```

### Health Checks

```bash
# Health check script
#!/bin/bash
services=("prometheus" "grafana" "alertmanager")

for service in "${services[@]}"; do
  if ! docker-compose ps $service | grep -q "Up"; then
    echo "Service $service is down"
    exit 1
  fi
done

echo "All monitoring services are healthy"
```

This monitoring setup provides comprehensive observability for the TradPal trading system, enabling proactive monitoring, alerting, and performance optimization.