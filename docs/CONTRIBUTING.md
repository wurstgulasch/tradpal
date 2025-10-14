# Contributing to TradPal Indicator

Vielen Dank für Ihr Interesse an der Weiterentwicklung von TradPal! Wir freuen uns über Beiträge aus der Community.

## 🚀 Wie kann ich beitragen?

### Arten von Beiträgen

1. **🐛 Bug Reports**: Fehler melden
2. **💡 Feature Requests**: Neue Ideen vorschlagen
3. **📝 Documentation**: Dokumentation verbessern
4. **💻 Code Contributions**: Code beitragen
5. **🧪 Testing**: Tests schreiben oder ausführen
6. **📊 Performance**: Benchmarks und Optimierungen

### Entwicklungs-Workflow

#### 1. Repository forken
```bash
git clone https://github.com/your-org/tradpal-indicator.git
cd tradpal-indicator
git checkout -b feature/your-feature-name
```

#### 2. Entwicklungsumgebung einrichten
```bash
# Conda-Umgebung erstellen (empfohlen)
conda env create -f environment.yml
conda activate tradpal_env

# Oder mit pip
pip install -e .[dev,ml,webui]
```

#### 3. Tests ausführen
```bash
# Alle Tests
pytest tests/ -v

# Mit Coverage
pytest tests/ --cov=src --cov-report=html
```

#### 4. Code-Qualität sicherstellen
```bash
# Linting
flake8 src/ tests/

# Type checking
mypy src/

# Formatierung
black src/ tests/
isort src/ tests/
```

#### 5. Commit und Push
```bash
git add .
git commit -m "feat: Add your feature description"
git push origin feature/your-feature-name
```

#### 6. Pull Request erstellen
- Gehen Sie zu GitHub und erstellen Sie einen PR
- Verwenden Sie eine der [Pull Request Templates](.github/PULL_REQUEST_TEMPLATE/)
- Warten Sie auf Review

## 📋 Pull Request Guidelines

### Commit Messages
Folgen Sie dem [Conventional Commits](https://conventionalcommits.org/) Format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: Neue Features
- `fix`: Bug fixes
- `docs`: Dokumentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Tests hinzufügen
- `chore`: Maintenance tasks

**Beispiele:**
```
feat(ml): Add LSTM model support
fix(backtester): Resolve memory leak in simulation
docs(api): Update parameter descriptions
```

### Code Style

#### Python
- **PEP 8** konform
- **Type Hints** verwenden
- **Docstrings** für alle öffentlichen Funktionen
- **Black** für Formatierung
- **isort** für Import-Sortierung

#### Beispiel:
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
- **pytest** Framework verwenden
- Unit Tests für alle neuen Features
- Integration Tests für Workflows
- Mindestens 80% Code Coverage
- Tests sollten schnell laufen (< 30 Sekunden)

#### Test Beispiel:
```python
import pytest
import pandas as pd
from src.indicators import calculate_rsi

def test_calculate_rsi():
    # Arrange
    data = pd.DataFrame({
        'close': [100, 110, 105, 115, 120]
    })

    # Act
    result = calculate_rsi(data, period=3)

    # Assert
    assert len(result) == len(data)
    assert not result.isna().all()
```

## 🏗️ Architektur Guidelines

### Modulare Struktur
```
src/                    # Kernmodule
├── indicators.py       # Technische Indikatoren
├── signal_generator.py # Signal-Generierung
├── backtester.py       # Backtesting-Engine
├── ml_predictor.py     # ML-Modelle
└── ...

services/               # Service-Komponenten
├── web_ui/            # Web-Interface
└── ...

scripts/               # Utility-Scripts
tests/                 # Unit-Tests
```

### Abhängigkeiten
- **Kern-Abhängigkeiten**: pandas, numpy, TA-Lib
- **ML-Abhängigkeiten**: pytorch, optuna, scikit-learn
- **Web-UI**: streamlit, plotly
- **Dev-Tools**: pytest, black, mypy, flake8

### Konfiguration
- **settings.py**: Zentrale Konfiguration
- **Umgebungsvariablen**: Für Secrets
- **Profile**: light/heavy für Performance

## 🔍 Code Review Process

### Checkliste für Reviewer
- [ ] Code Style konform (PEP 8, Black)
- [ ] Type Hints vorhanden
- [ ] Docstrings vollständig
- [ ] Tests vorhanden und passing
- [ ] Keine Security Issues
- [ ] Performance optimiert
- [ ] Dokumentation aktualisiert

### Checkliste für Contributors
- [ ] Alle Tests passing
- [ ] Code formatiert (Black)
- [ ] Imports sortiert (isort)
- [ ] Linting ohne Fehler
- [ ] Typprüfung erfolgreich
- [ ] Dokumentation aktualisiert
- [ ] Changelog aktualisiert

## 🎯 Feature Entwicklung

### Ideenfindung
1. **Issues** prüfen für offene Features
2. **Discussions** für neue Ideen
3. **Discord** Community für Feedback

### Implementierung
1. **Design Document** erstellen
2. **Prototype** entwickeln
3. **Tests** schreiben
4. **Dokumentation** aktualisieren
5. **Review** anfordern

### Beispiel: Neue Indikator hinzufügen

```python
# src/indicators.py
def calculate_new_indicator(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate new technical indicator."""
    # Implementation
    pass

# config/settings.py
INDICATORS = {
    'new_indicator': {
        'enabled': True,
        'period': 14
    }
}

# tests/test_indicators.py
def test_calculate_new_indicator():
    # Test implementation
    pass
```

## 📊 Performance Benchmarks

### Ziele
- **Backtest Speed**: < 10 Sekunden für 1 Jahr 1h Daten
- **Memory Usage**: < 1GB für typische Workloads
- **Accuracy**: > 95% Signalkonsistenz

### Benchmarks ausführen
```bash
python scripts/benchmark_performance.py
```

## 🤝 Community Guidelines

### Kommunikation
- **Respektvoll** und **konstruktiv** bleiben
- **Englisch** als Standardsprache
- **Issues** für technische Diskussionen
- **Discussions** für allgemeine Themen

### Verhaltenskodex
- Keine Diskriminierung
- Keine Spam-Posts
- Konstruktives Feedback
- Gemeinschaftsorientiert

## 🏆 Belohnungen

### Contributor Levels
- **🥉 Bronze**: 1-5 merged PRs
- **🥈 Silver**: 6-15 merged PRs
- **🥇 Gold**: 16+ merged PRs
- **💎 Diamond**: Significant contributions

### Hall of Fame
Besondere Erwähnung für außergewöhnliche Beiträge:
- [Liste der Top-Contributors]

## 📞 Support

Bei Fragen:
- **GitHub Issues**: Für Bugs und Features
- **GitHub Discussions**: Für Fragen
- **Discord**: Für schnellen Support

Vielen Dank für Ihren Beitrag zu TradPal! 🚀