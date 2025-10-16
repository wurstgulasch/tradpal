# Data Sources Module

This module provides a modular, extensible architecture for fetching financial data from various sources including cryptocurrency exchanges, financial data providers, and datasets.

## Overview

The data sources module implements a factory pattern with the following key components:

- **BaseDataSource**: Abstract base class defining the common interface
- **DataSourceFactory**: Factory for creating and managing data source instances
- **Specific Implementations**: Kaggle, Yahoo Finance, and CCXT data sources

## Architecture

### BaseDataSource

All data sources inherit from `BaseDataSource` which provides:

- Common interface for data fetching (`fetch_historical_data`, `fetch_recent_data`)
- Data validation and quality checks
- Logging and error handling
- Configuration management

### Data Sources

#### KaggleDataSource

**Purpose**: Fetch high-quality historical Bitcoin data from Kaggle datasets.

**Features**:
- Optimized for Bitcoin/USD data with minute-level granularity
- Automatic data quality validation
- Resampling capabilities for different timeframes
- Memory-efficient processing for large datasets

**Configuration**:
```python
config = {
    'dataset': 'bitcoin-historical-data',
    'symbol': 'BTC',
    'quality_checks': True,
    'cache_enabled': True
}
```

#### YahooFinanceDataSource

**Purpose**: Fetch financial data from Yahoo Finance.

**Features**:
- Support for stocks, ETFs, and cryptocurrencies
- Multiple timeframe support (1m, 5m, 1h, 1d, etc.)
- Automatic symbol format conversion
- Configurable timeout and data adjustments

**Configuration**:
```python
config = {
    'timeout': 30,
    'auto_adjust': True,
    'prepost': False
}
```

#### CCXTDataSource

**Purpose**: Fetch real-time and historical data from cryptocurrency exchanges.

**Features**:
- Support for 100+ exchanges via CCXT library
- Real-time and historical OHLCV data
- Exchange-specific optimizations
- Rate limiting and error handling

**Configuration**:
```python
config = {
    'exchange': 'binance',
    'timeout': 30000,
    'enableRateLimit': True
}
```

## Usage

### Basic Usage

```python
from services.data_service.data_sources.factory import DataSourceFactory

# Create a data source (auto-selects best available)
source = DataSourceFactory.create_data_source()

# Or specify a specific source
kaggle_source = DataSourceFactory.create_data_source('kaggle')
yahoo_source = DataSourceFactory.create_data_source('yahoo_finance')
ccxt_source = DataSourceFactory.create_data_source('ccxt')

# Fetch historical data
data = source.fetch_historical_data(
    symbol='BTC/USDT',
    timeframe='1d',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)
```

### Advanced Usage

```python
# With custom configuration
config = {'dataset': 'bitcoin-historical-data', 'quality_checks': True}
source = DataSourceFactory.create_data_source('kaggle', config)

# Fetch recent data
recent_data = source.fetch_recent_data('BTC/USDT', '1h', limit=100)

# Check availability
if source.is_available():
    print(f"Source {source.name} is available")
```

## Data Format

All data sources return data in a consistent pandas DataFrame format:

```python
# Columns: ['open', 'high', 'low', 'close', 'volume']
# Index: DatetimeIndex with timezone information
# Data types: float64 for OHLC, int64 for volume
```

## Error Handling

The module provides comprehensive error handling:

- **ImportError**: Raised when required libraries are not available
- **ValueError**: Invalid configuration or parameters
- **ConnectionError**: Network or API connectivity issues
- **DataQualityError**: Data validation failures

## Testing

The module includes comprehensive unit tests with 100% coverage:

```bash
# Run all data source tests
pytest services/data_service/data_sources/ -v

# Run specific test file
pytest services/data_service/data_sources/test_kaggle.py -v
```

## Configuration

Data sources can be configured via the main TradPal settings or per-instance:

```python
# Global configuration in settings.py
DATA_SOURCES = {
    'kaggle': {
        'enabled': True,
        'default_dataset': 'bitcoin-historical-data'
    },
    'yahoo_finance': {
        'enabled': True,
        'timeout': 30
    },
    'ccxt': {
        'enabled': True,
        'exchange': 'binance'
    }
}
```

## Performance Considerations

- **Caching**: Enable caching for frequently accessed data
- **Chunked Processing**: Large datasets are processed in chunks
- **Async Operations**: Network requests are handled asynchronously
- **Memory Management**: Efficient memory usage for large datasets

## Extending the Module

To add a new data source:

1. Create a new class inheriting from `BaseDataSource`
2. Implement required abstract methods
3. Add the source to `DataSourceFactory.SOURCES`
4. Create comprehensive unit tests
5. Update this documentation

Example:

```python
from .base import BaseDataSource

class MyDataSource(BaseDataSource):
    def __init__(self, config=None):
        super().__init__("My Data Source", config)

    def fetch_historical_data(self, symbol, timeframe, start_date=None, end_date=None, limit=None):
        # Implementation here
        pass

    def fetch_recent_data(self, symbol, timeframe, limit=100):
        # Implementation here
        pass
```

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical operations
- requests: HTTP client for API calls
- kaggle (optional): Kaggle dataset access
- yfinance (optional): Yahoo Finance data
- ccxt (optional): Cryptocurrency exchange data

## Contributing

When contributing to this module:

1. Follow the existing code style and patterns
2. Add comprehensive unit tests
3. Update documentation
4. Ensure backward compatibility
5. Test with all supported data sources