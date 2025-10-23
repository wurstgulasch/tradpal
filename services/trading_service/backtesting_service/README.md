# TradPal Backtesting Service

## Overview

The Backtesting Service is a unified microservice that consolidates backtesting, ML training, optimization, and walk-forward analysis functionality. It provides a complete toolkit for strategy development and validation.

## Features

- **Unified Backtesting Engine**: Run backtests for various trading strategies
- **ML Model Training**: Train machine learning models for enhanced strategies
- **Parameter Optimization**: Optimize strategy parameters using grid search, random search, and genetic algorithms
- **Walk-Forward Analysis**: Validate strategy robustness through time-series validation
- **Complete Workflow**: Run end-to-end backtesting workflows with optimization and validation

## Architecture

The service consists of four main components:

- `backtesting/`: Core backtesting engine
- `ml_training/`: Machine learning model training
- `optimization/`: Strategy parameter optimization
- `walk_forward/`: Walk-forward analysis and validation

All components are orchestrated through the `BacktestingServiceOrchestrator`.

## Quick Start

### Installation

```bash
cd services/backtesting_service
pip install -r requirements.txt
```

### Basic Usage

```python
from services.backtesting_service.orchestrator import BacktestingServiceOrchestrator
import pandas as pd

# Initialize orchestrator
orchestrator = BacktestingServiceOrchestrator()
await orchestrator.initialize()

# Sample strategy configuration
strategy = {
    "name": "moving_average_crossover",
    "type": "technical",
    "parameters": {
        "fast_period": 10,
        "slow_period": 20,
        "stop_loss": 0.02,
        "take_profit": 0.05
    }
}

# Sample data (OHLC format)
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Run quick backtest
results = await orchestrator.run_quick_backtest(strategy, data)
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']}")

# Run complete workflow with optimization
workflow_config = {
    "enable_ml": False,
    "enable_optimization": True,
    "enable_walk_forward": True,
    "param_ranges": {
        "fast_period": [5, 10, 15],
        "slow_period": [20, 30, 40]
    }
}

results = await orchestrator.run_complete_backtesting_workflow(strategy, data, workflow_config)
print(f"Final Recommendation: {results['final_recommendation']['overall_rating']}")

await orchestrator.shutdown()
```

### API Usage

Start the FastAPI service:

```bash
python services/backtesting_service/main.py --host 0.0.0.0 --port 8001
```

Available endpoints:

- `GET /health` - Health check
- `POST /backtest` - Run quick backtest
- `POST /optimize` - Optimize strategy parameters
- `POST /ml/train` - Train ML model
- `POST /walk-forward` - Run walk-forward analysis
- `POST /workflow` - Run complete workflow

## Configuration

### Strategy Configuration

```python
strategy_config = {
    "name": "strategy_name",
    "type": "technical|ml_based",
    "parameters": {
        # Strategy-specific parameters
    }
}
```

### Workflow Configuration

```python
workflow_config = {
    "enable_ml": False,
    "enable_optimization": True,
    "enable_walk_forward": True,
    "ml_config": {
        "model_type": "random_forest",
        "train_split": 0.7
    },
    "param_ranges": {
        "param1": [value1, value2, ...],
        "param2": [value1, value2, ...]
    },
    "walk_forward_config": {
        "in_sample_window": 252,
        "out_sample_window": 21,
        "step_size": 21
    }
}
```

## Testing

Run the test suite:

```bash
pytest services/backtesting_service/tests.py -v
```

Run integration tests:

```bash
pytest services/backtesting_service/tests.py::test_service_integration -v
```

## Development

### Adding New Strategies

1. Extend the `BacktestingService` class
2. Add strategy logic in the `run_backtest` method
3. Update parameter validation
4. Add tests for the new strategy

### Adding New Optimization Methods

1. Extend the `OptimizationService` class
2. Implement optimization algorithm
3. Add configuration options
4. Update tests

## Performance Considerations

- Use async/await for all I/O operations
- Implement proper error handling and timeouts
- Consider memory usage for large datasets
- Use pandas vectorized operations for performance

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Machine learning (optional)
- fastapi: API framework (optional)
- uvicorn: ASGI server (optional)
- pytest: Testing framework

## License

See main project LICENSE file.