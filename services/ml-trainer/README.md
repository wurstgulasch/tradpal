# ML Trainer Service

Advanced machine learning training service with support for multiple model types and optimization strategies.

## Features

- **sklearn Models**: Random Forest, Gradient Boosting
- **PyTorch Models**: LSTM, GRU, Transformer
- **AutoML**: Automated hyperparameter optimization with Optuna
- **Ensemble Training**: Combined sklearn + PyTorch models

## Usage

### Basic Training

```bash
# Train Random Forest
python train_service.py --mode sklearn --model-type random_forest

# Train Gradient Boosting
python train_service.py --mode sklearn --model-type gradient_boosting
```

### PyTorch Models

```bash
# Train LSTM
python train_service.py --mode pytorch --model-type lstm

# Train GRU
python train_service.py --mode pytorch --model-type gru

# Train Transformer
python train_service.py --mode pytorch --model-type transformer
```

### AutoML Optimization

```bash
# Optimize Random Forest hyperparameters
python train_service.py --mode automl --model-type random_forest

# Optimize Gradient Boosting hyperparameters
python train_service.py --mode automl --model-type gradient_boosting
```

### Ensemble Training

```bash
# Train both sklearn and PyTorch models
python train_service.py --mode ensemble
```

### Custom Parameters

```bash
python train_service.py \
  --symbol BTC/USDT \
  --timeframe 1h \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --mode pytorch \
  --model-type lstm \
  --lookback-days 365
```

## Options

- `--symbol`: Trading symbol (default: from config)
- `--timeframe`: Chart timeframe (default: from config)
- `--start-date`: Start date for training data (YYYY-MM-DD)
- `--end-date`: End date for training data (YYYY-MM-DD)
- `--lookback-days`: Days of historical data (default: from config)
- `--mode`: Training mode (sklearn, pytorch, automl, ensemble)
- `--model-type`: Model type (depends on mode)

## Requirements

### Core
- pandas
- numpy
- scikit-learn

### Optional
- `torch>=2.0.0` for PyTorch models
- `optuna>=3.0.0` for AutoML

Install with:
```bash
pip install torch optuna
```

## Output

Models are saved to `cache/ml_models/` with:
- Trained model weights
- Feature metadata
- Training history
- Performance metrics

## Configuration

Edit `config/settings.py` to customize:

```python
ML_USE_PYTORCH = True  # Enable PyTorch models
ML_PYTORCH_MODEL_TYPE = 'lstm'  # Default model type
ML_USE_AUTOML = True  # Enable AutoML
ML_AUTOML_N_TRIALS = 100  # Number of optimization trials
```

## Examples

See `examples/advanced_ml_examples.py` for comprehensive usage examples.
