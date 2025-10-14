# Discovery Service

The Discovery Service provides genetic algorithm optimization for trading indicator parameters to maximize outperformance. It uses evolutionary algorithms to discover optimal indicator combinations and parameter settings across different market conditions.

## Features

- **Genetic Algorithm Optimization**: Uses DEAP library for evolutionary computation
- **Multi-Objective Fitness**: Optimizes for Sharpe ratio, profit factor, and drawdown
- **Cross-Validation**: Prevents overfitting with walk-forward analysis
- **Event-Driven Architecture**: Real-time optimization progress tracking
- **RESTful API**: FastAPI-based endpoints for optimization management
- **Async Processing**: Non-blocking optimization execution
- **Comprehensive Indicator Space**: 25+ predefined indicator combinations

## Architecture

```
Discovery Service
├── service.py          # Core GA optimization engine
├── api.py             # FastAPI REST API
├── demo.py            # Demonstration scripts
├── tests.py           # Unit and integration tests
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container configuration
└── k8s-deployment.yaml # Kubernetes manifests
```

## Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   cd services/discovery_service
   pip install -r requirements.txt
   ```

2. **Run the Service**
   ```bash
   python -m services.discovery_service.api
   ```

3. **Run Demo**
   ```bash
   python services/discovery_service/demo.py
   ```

### Docker

```bash
# Build image
docker build -t tradpal/discovery-service:latest ./services/discovery_service

# Run container
docker run -p 8003:8003 tradpal/discovery-service:latest
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f services/discovery_service/k8s-deployment.yaml
```

## API Endpoints

### Start Optimization
```http
POST /optimize/start
Content-Type: application/json

{
  "optimization_id": "my_optimization_001",
  "symbol": "BTC/USDT",
  "timeframe": "1d",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "population_size": 50,
  "generations": 20,
  "use_walk_forward": true
}
```

### Get Optimization Status
```http
GET /optimize/status/{optimization_id}
```

### List Active Optimizations
```http
GET /optimize/active
```

### Cancel Optimization
```http
POST /optimize/cancel/{optimization_id}
```

### Health Check
```http
GET /health
```

## Optimization Parameters

### Indicator Combinations
The service supports 25+ predefined indicator combinations:

- **Basic**: EMA crossover, RSI divergence
- **Advanced**: Bollinger Bands + RSI, MACD + Stochastic
- **Complex**: Multi-timeframe combinations, volume-based filters
- **ML-Enhanced**: Indicator combinations with ML signal boosting

### Genetic Algorithm Settings
- **Population Size**: 20-100 individuals (default: 50)
- **Generations**: 10-50 iterations (default: 20)
- **Crossover Rate**: 0.8 (fixed)
- **Mutation Rate**: 0.2 (fixed)
- **Tournament Size**: 3 (fixed)

### Fitness Function
Multi-objective optimization:
- **Sharpe Ratio** (40% weight): Risk-adjusted returns
- **Profit Factor** (35% weight): Gross profit / Gross loss
- **Max Drawdown** (25% weight): Maximum peak-to-trough decline

## Example Usage

### Python Client

```python
import httpx
import asyncio

async def run_optimization():
    async with httpx.AsyncClient() as client:
        # Start optimization
        response = await client.post("http://localhost:8003/optimize/start", json={
            "optimization_id": "example_opt",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "population_size": 30,
            "generations": 10
        })

        if response.status_code == 200:
            print("Optimization started successfully")

            # Monitor progress
            while True:
                status_response = await client.get(
                    f"http://localhost:8003/optimize/status/example_opt"
                )
                status = status_response.json()

                if status["status"] == "completed":
                    print(f"Best fitness: {status['best_fitness']}")
                    break
                elif status["status"] == "failed":
                    print(f"Optimization failed: {status['error_message']}")
                    break

                await asyncio.sleep(5)

asyncio.run(run_optimization())
```

### Command Line Demo

```bash
# Run full demo suite
python services/discovery_service/demo.py

# Run specific demo
python services/discovery_service/demo.py --single
python services/discovery_service/demo.py --multiple
python services/discovery_service/demo.py --events
```

## Configuration

### Environment Variables
- `LOG_LEVEL`: Logging level (default: INFO)
- `REDIS_URL`: Redis connection URL for caching
- `MAX_CONCURRENT_OPTIMIZATIONS`: Maximum parallel optimizations (default: 5)

### Genetic Algorithm Tuning
Modify parameters in `service.py`:

```python
# Adjust GA parameters
self.ga_params = {
    'population_size': 50,
    'generations': 20,
    'crossover_prob': 0.8,
    'mutation_prob': 0.2,
    'tournament_size': 3
}
```

## Testing

### Unit Tests
```bash
cd services/discovery_service
python -m pytest tests.py -v
```

### Integration Tests
```bash
# Test with real backtesting service
python -m pytest tests.py::TestDiscoveryService::test_full_optimization_workflow -v
```

### Performance Testing
```bash
# Load testing with multiple concurrent optimizations
python -m pytest tests.py::TestDiscoveryService::test_concurrent_optimizations -v
```

## Monitoring

### Metrics
- Optimization duration and success rate
- Best fitness scores over time
- Population convergence statistics
- API response times

### Health Checks
- Service availability
- Memory and CPU usage
- Active optimization count

## Troubleshooting

### Common Issues

1. **Optimization takes too long**
   - Reduce population size or generations
   - Use smaller date ranges
   - Disable walk-forward analysis

2. **Memory errors**
   - Reduce population size
   - Use lighter indicator combinations
   - Increase container memory limits

3. **Poor optimization results**
   - Increase population size and generations
   - Enable walk-forward validation
   - Check data quality and date ranges

### Debug Mode
Enable detailed logging:
```bash
LOG_LEVEL=DEBUG python -m services.discovery_service.api
```

## Performance Optimization

### Hardware Recommendations
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: SSD for faster data access

### Scaling Strategies
- Horizontal scaling with Kubernetes
- Redis caching for data sharing
- Async processing for concurrent optimizations

## Integration

### Event System
The service integrates with the EventSystem for:
- Optimization lifecycle events
- Progress updates
- Error notifications
- Result broadcasting

### Dependencies
- **Backtesting Service**: For fitness evaluation
- **Data Service**: For historical market data
- **Cache Service**: For optimization result caching

## Contributing

1. Add new indicator combinations in `INDICATOR_COMBINATIONS`
2. Implement custom fitness functions
3. Add new optimization algorithms
4. Improve parallel processing capabilities

## License

MIT License - see project root for details.