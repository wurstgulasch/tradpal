# Contributing to TradPal Indicator

Thank you for your interest in developing TradPal! We welcome contributions from the community.

## 🚀 How can I contribute?

### Types of Contributions

1. **🐛 Bug Reports**: Report bugs
2. **💡 Feature Requests**: Suggest new ideas
3. **📝 Documentation**: Improve documentation
4. **💻 Code Contributions**: Contribute code
5. **🧪 Testing**: Write or run tests
6. **📊 Performance**: Benchmarks and optimizations

### Development Workflow

#### 1. Fork Repository
```bash
git clone https://github.com/wurstgulasch/tradpal.git
cd tradpal_indicator
git checkout -b feature/your-feature-name
```

#### 2. Set up Development Environment
```bash
# Create conda environment (recommended)
conda env create -f environment.yml
conda activate tradpal-env

# Or with pip
pip install -e .[dev,ml,webui]
```

#### 3. Run Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=services --cov-report=html
```

#### 4. Ensure Code Quality
```bash
# Linting
flake8 services/ tests/

# Type checking
mypy services/

# Formatting
black services/ tests/
isort services/ tests/
```

#### 5. Commit and Push
```bash
git add .
git commit -m "feat: Add your feature description"
git push origin feature/your-feature-name
```

#### 6. Create Pull Request
- Go to GitHub and create a PR
- Use one of the [Pull Request Templates](.github/PULL_REQUEST_TEMPLATE/)
- Wait for review

## 📋 Pull Request Guidelines

### Commit Messages
Follow the [Conventional Commits](https://conventionalcommits.org/) format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(ml): Add LSTM model support
fix(backtester): Resolve memory leak in simulation
docs(api): Update parameter descriptions
```

### Code Style

#### Python
- **PEP 8** compliant
- Use **type hints**
- **Docstrings** for all public functions
- **Black** for formatting
- **isort** for import sorting

#### Example:
```python
def calculate_indicator(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate technical indicator.

    Args:
        data: OHLCV DataFrame
        period: Calculation period

    Returns:
        Indicator values as Series
    """
    # Implementation here
    pass
```

### Tests
- Use **pytest** framework
- Unit tests for all new features
- Integration tests for workflows
- Minimum 80% code coverage
- Tests should run fast (< 30 seconds)

#### Test Example:
```python
import pytest
import pandas as pd
from services.data_service.data_sources.liquidation import LiquidationDataSource

def test_liquidation_fallback():
    # Arrange
    source = LiquidationDataSource()

    # Act
    data = source.fetch_recent_data('BTC/USDT', '1h', limit=10)

    # Assert
    assert not data.empty
    assert 'liquidation_signal' in data.columns
    assert 'data_source' in data.columns
```

## 🏗️ Architecture Guidelines

### Modular Structure
```
services/               # Service components
├── data_service/       # Data management
│   └── data_sources/   # Data source implementations
├── core/               # Core calculations
├── backtesting_service/# Backtesting engine
└── ...

scripts/               # Utility scripts
tests/                 # Unit tests
docs/                  # Documentation
```

### Dependencies
- **Core Dependencies**: pandas, numpy, requests
- **ML Dependencies**: pytorch, optuna, scikit-learn
- **Web UI**: streamlit, plotly
- **Dev Tools**: pytest, black, mypy, flake8

### Configuration
- **settings.py**: Central configuration
- **Environment variables**: For secrets
- **Profiles**: light/heavy for performance

## 🔍 Code Review Process

### Reviewer Checklist
- [ ] Code style compliant (PEP 8, Black)
- [ ] Type hints present
- [ ] Docstrings complete
- [ ] Tests present and passing
- [ ] No security issues
- [ ] Performance optimized
- [ ] Documentation updated

### Contributor Checklist
- [ ] All tests passing
- [ ] Code formatted (Black)
- [ ] Imports sorted (isort)
- [ ] Linting without errors
- [ ] Type checking successful
- [ ] Documentation updated
- [ ] Changelog updated

## 🎯 Feature Development

### Ideation
1. Check **Issues** for open features
2. **Discussions** for new ideas
3. **Discord** community for feedback

### Implementation
1. Create **Design Document**
2. Develop **Prototype**
3. Write **Tests**
4. Update **Documentation**
5. Request **Review**

### Example: Adding New Data Source

```python
# services/data_service/data_sources/new_source.py
from .base import BaseDataSource

class NewDataSource(BaseDataSource):
    """New data source implementation."""

    def fetch_recent_data(self, symbol: str, timeframe: str, limit: int = 100):
        # Implementation
        pass

# services/data_service/data_sources/factory.py
from .new_source import NewDataSource

# Add to factory
elif name == 'new_source':
    return NewDataSource(config)

# tests/unit/test_new_source.py
def test_new_data_source():
    # Test implementation
    pass
```

## 📊 Performance Benchmarks

### Targets
- **Backtest Speed**: < 10 seconds for 1 year 1h data
- **Memory Usage**: < 1GB for typical workloads
- **Accuracy**: > 95% signal consistency

### Run Benchmarks
```bash
python scripts/performance_benchmark.py
```

## 🤝 Community Guidelines

### Communication
- Stay **respectful** and **constructive**
- **English** as standard language
- **Issues** for technical discussions
- **Discussions** for general topics

### Code of Conduct
- No discrimination
- No spam posts
- Constructive feedback
- Community-oriented

## 🏆 Rewards

### Contributor Levels
- **🥉 Bronze**: 1-5 merged PRs
- **🥈 Silver**: 6-15 merged PRs
- **🥇 Gold**: 16+ merged PRs
- **💎 Diamond**: Significant contributions

### Hall of Fame
Special mention for outstanding contributions:
- [List of Top Contributors]

## 📞 Support

For questions:
- **GitHub Issues**: For bugs and features
- **GitHub Discussions**: For questions
- **Discord**: For quick support

Thank you for your contribution to TradPal! 🚀