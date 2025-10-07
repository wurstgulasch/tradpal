# TradPal Indicator - Test Suite

## 📁 Test Structure

Die Test-Suite spiegelt die Projektstruktur wider und ist wie folgt organisiert:

```
tests/
├── config/                    # Tests für Konfiguration
│   └── test_config.py
├── src/                       # Tests für Kernmodule
│   ├── test_data_fetcher.py
│   ├── test_indicators.py
│   ├── test_output.py
│   └── test_backtester.py
├── integrations/              # Tests für Integrationen
│   └── test_integrations.py
├── scripts/                   # Tests für Skripte
│   └── test_scripts.py
├── test_error_handling.py     # Umfassende Fehlerbehandlungstests
├── test_edge_cases.py         # Grenzfälle und spezielle Szenarien
├── test_performance.py        # Performance- und Lasttests
└── __init__.py
```

## 🧪 Test-Kategorien

### Unit Tests
- **data_fetcher**: Datenabruf und Validierung von Exchanges
- **indicators**: Technische Indikatoren (EMA, RSI, BB, ATR, ADX)
- **output**: JSON-Formatierung und Dateioperationen
- **backtester**: Historische Backtest-Funktionalität
- **config**: Konfigurationsparameter und Validierung

### Integration Tests
- **integrations**: Telegram/Email-Benachrichtigungen
- **scripts**: CLI-Tools und Automatisierungsskripte

### Spezielle Tests
- **error_handling**: Umfassende Fehlerbehandlung für alle Module
- **edge_cases**: Grenzfälle und ungewöhnliche Szenarien
- **performance**: Performance-Benchmarks und Lasttests

## 🚀 Tests Ausführen

### Alle Tests
```bash
python run_tests.py
```

### Mit ausführlicher Ausgabe
```bash
python run_tests.py --verbose
```

### Mit Coverage-Report
```bash
python run_tests.py --coverage
```

### Spezifische Testdateien
```bash
python run_tests.py --test-files tests/src/test_data_fetcher.py tests/src/test_indicators.py
```

### Einzelne Test-Module
```bash
# Nur Data Fetcher Tests
python -m pytest tests/src/test_data_fetcher.py -v

# Nur Performance Tests
python -m pytest tests/test_performance.py -v
```

## 📊 Test-Abdeckung

Die Test-Suite bietet umfassende Abdeckung für:

- ✅ **120+ individuelle Tests**
- ✅ Unit Tests für alle Kernfunktionen
- ✅ Integration Tests für externe Services
- ✅ Error Handling für alle Fehlerszenarien
- ✅ Edge Cases für Grenzfälle
- ✅ Performance Benchmarks
- ✅ Mock-basierte Tests für APIs

## 🛠️ Test-Framework

- **pytest**: Moderne Test-Framework mit umfangreichen Features
- **unittest.mock**: Mocking für externe APIs und Services
- **pandas/numpy**: Datenverarbeitung in Tests
- **tempfile**: Temporäre Dateien für I/O-Tests

## 📈 Qualitätsmetriken

- **Zero Failures**: Alle Tests müssen bestehen
- **High Coverage**: >90% Code-Abdeckung angestrebt
- **Fast Execution**: Tests laufen in <30 Sekunden
- **Reliable**: Deterministische, nicht-flaky Tests

## 🔧 Test-Entwicklung

### Neue Tests hinzufügen:
1. Testdatei im entsprechenden Unterordner erstellen
2. pytest-Konventionen folgen (test_*.py)
3. Umfassende Docstrings und Kommentare
4. Mock externe Dependencies

### Test-Namenskonventionen:
- `test_function_name()`: Unit Tests
- `test_feature_scenario()`: Integration Tests
- `TestClassName`: Test-Klassen
- `test_method_name()`: Methoden in Test-Klassen

## 📋 CI/CD Integration

Die Tests sind für automatische Ausführung in CI/CD-Pipelines optimiert:

- Parallele Ausführung möglich
- JUnit/XML Output für Berichterstattung
- Coverage-Reports für Qualitätsmetriken
- Fail-Fast bei kritischen Fehlern

---

*Automatisch generiert für TradPal Indicator v2.0*