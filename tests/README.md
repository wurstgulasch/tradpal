# TradPal Indicator - Test Suite

## 📁 Test Structure

The test suite mirrors the project structure and is organized as follows:

```
tests/
├── config/                    # Configuration tests
│   └── test_config.py
├── src/                       # Core module tests
│   ├── test_data_fetcher.py
│   ├── test_indicators.py
│   ├── test_output.py
│   └── test_backtester.py
├── integrations/              # Integration tests
│   └── test_integrations.py
├── scripts/                   # Script tests
│   └── test_scripts.py
├── test_error_handling.py     # Comprehensive error handling tests
├── test_edge_cases.py         # Edge cases and special scenarios
├── test_performance.py        # Performance and load tests
└── __init__.py
```

## 🧪 Test Categories

### Unit Tests
- **data_fetcher**: Data retrieval and exchange validation
- **indicators**: Technical indicators (EMA, RSI, BB, ATR, ADX)
- **output**: JSON formatting and file operations
- **backtester**: Historical backtesting functionality
- **config**: Configuration parameters and validation

### Integration Tests
- **integrations**: Telegram/Email notifications
- **scripts**: CLI tools and automation scripts

### Special Tests
- **error_handling**: Comprehensive error handling for all modules
- **edge_cases**: Edge cases and unusual scenarios
- **performance**: Performance benchmarks and load tests

## 🚀 Running Tests

### All Tests
```bash
python run_tests.py
```

### With Verbose Output
```bash
python run_tests.py --verbose
```

### With Coverage Report
```bash
python run_tests.py --coverage
```

### Specific Test Files
```bash
python run_tests.py --test-files tests/src/test_data_fetcher.py tests/src/test_indicators.py
```

### Individual Test Modules
```bash
# Only Data Fetcher Tests
python -m pytest tests/src/test_data_fetcher.py -v

# Only Performance Tests
python -m pytest tests/test_performance.py -v
```

## 📊 Test Coverage

The test suite provides comprehensive coverage for:

- ✅ **120+ individual tests**
- ✅ Unit tests for all core functions
- ✅ Integration tests for external services
- ✅ Error handling for all error scenarios
- ✅ Edge cases for boundary conditions
- ✅ Performance benchmarks
- ✅ Mock-based tests for APIs

## 🛠️ Test Framework

- **pytest**: Modern test framework with extensive features
- **unittest.mock**: Mocking for external APIs and services
- **pandas/numpy**: Data processing in tests
- **tempfile**: Temporary files for I/O tests

## 📈 Quality Metrics

- **Zero Failures**: All tests must pass
- **High Coverage**: >90% code coverage targeted
- **Fast Execution**: Tests run in <30 seconds
- **Reliable**: Deterministic, non-flaky tests

## 🔧 Test Development

### Adding New Tests:
1. Create test file in appropriate subdirectory
2. Follow pytest conventions (test_*.py)
3. Include comprehensive docstrings and comments
4. Mock external dependencies

### Test Naming Conventions:
- `test_function_name()`: Unit tests
- `test_feature_scenario()`: Integration tests
- `TestClassName`: Test classes
- `test_method_name()`: Methods in test classes

## 📋 CI/CD Integration

The tests are optimized for automatic execution in CI/CD pipelines:

- Parallel execution possible
- JUnit/XML output for reporting
- Coverage reports for quality metrics
- Fail-fast on critical errors

---

*Automatically generated for TradPal Indicator v2.0*