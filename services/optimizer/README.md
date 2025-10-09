# Optimizer Service

Enhanced walk-forward optimization service with advanced overfitting detection metrics.

## Features

- **Walk-Forward Analysis**: Time-series cross-validation
- **Information Coefficient**: Correlation between in-sample and out-of-sample performance
- **Bias-Variance Tradeoff**: Quantitative error decomposition
- **Overfitting Detection**: Multiple metrics for model validation
- **Automated Interpretation**: Human-readable recommendations

## Usage

### Basic Optimization

```bash
# Optimize for Sharpe ratio
python optimize_service.py --metric sharpe_ratio

# Optimize for win rate
python optimize_service.py --metric win_rate

# Optimize for profit factor
python optimize_service.py --metric profit_factor
```

### Custom Parameters

```bash
python optimize_service.py \
  --symbol BTC/USDT \
  --timeframe 1h \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --metric sharpe_ratio \
  --lookback-days 365
```

## Options

- `--symbol`: Trading symbol (default: from config)
- `--timeframe`: Chart timeframe (default: from config)
- `--start-date`: Start date for optimization (YYYY-MM-DD)
- `--end-date`: End date for optimization (YYYY-MM-DD)
- `--lookback-days`: Days of historical data (default: from config)
- `--metric`: Evaluation metric (sharpe_ratio, win_rate, profit_factor, total_return)

## Metrics Calculated

### Standard Metrics
- Average out-of-sample performance
- Standard deviation of performance
- Performance decay (overfitting measure)

### Enhanced Metrics
- **Information Coefficient**: Correlation between IS and OOS performance
- **Bias**: Systematic error (difference between expected and actual)
- **Variance**: Model sensitivity to training data
- **Overfitting Ratio**: IS vs OOS performance gap
- **Consistency Score**: Stability across time periods

## Output

Results saved to `output/walk_forward/` with:
- Optimization summary
- Window-by-window results
- Enhanced metrics
- Automated interpretation

## Interpretation

The service provides automated assessment:

### Overfitting Assessment
- ✅ **Low Overfitting** (ratio < 0.2): Strategy generalizes well
- ⚡ **Moderate Overfitting** (0.2 - 0.5): Monitor closely in live trading
- ⚠️ **High Overfitting** (> 0.5): Not recommended for live use

### Information Coefficient
- ✅ **Strong** (IC > 0.5): Strong predictive relationship
- ⚡ **Moderate** (IC 0.2 - 0.5): Acceptable predictive power
- ⚠️ **Weak** (IC < 0.2): Poor prediction quality

### Consistency Score
- ✅ **High** (> 0.7): Stable across different periods
- ⚡ **Moderate** (0.4 - 0.7): Some variation
- ⚠️ **Low** (< 0.4): Highly variable performance

## Recommendations

Based on metrics, the service provides:
- Risk assessment (NOT RECOMMENDED, USE WITH CAUTION, ACCEPTABLE)
- Specific issues identified (overfitting, weak predictions, inconsistency)
- Suggested actions (simplify model, add regularization, more data)

## Configuration

Edit `config/settings.py` for walk-forward parameters.

## Examples

See `examples/advanced_ml_examples.py` for usage demonstration.
