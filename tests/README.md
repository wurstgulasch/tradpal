# TradPal Test Suite

Diese Test-Suite ist nach Best-Practices für Microservices-Architekturen organisiert und bietet eine zentrale Test-Suite im Projekt-Root.

## Struktur

```
tests/
├── conftest.py                    # Zentrale Test-Konfiguration und Fixtures
├── unit/                          # Unit-Tests (isolierte Komponenten)
│   ├── test_*.py                 # Einzelne Funktionen/Klassen
│   └── __init__.py
├── integration/                   # Integration-Tests (Service-Interaktionen)
│   ├── test_*.py                 # Cross-Service Tests
│   └── __init__.py
├── services/                      # Service-spezifische Tests
│   ├── core/                      # Core Service Tests
│   ├── data_service/              # Data Service Tests
│   ├── trading_bot_live/          # Trading Bot Tests
│   ├── backtesting_service/       # Backtesting Tests
│   ├── discovery_service/         # Parameter Optimization Tests
│   ├── risk_service/              # Risk Management Tests
│   ├── notification_service/      # Notification Tests
│   ├── mlops_service/             # ML Operations Tests
│   ├── security_service/          # Security Tests
│   └── web_ui/                    # Web UI Tests
├── e2e/                           # End-to-End Tests (komplette Workflows)
│   ├── test_*.py                 # Vollständige System-Tests
│   └── __init__.py
├── config/                        # Konfigurationstests
└── integrations/                  # Integration-Setup Tests
```

## Test-Typen

### Unit Tests (`tests/unit/`)
- Testen isolierte Komponenten ohne externe Abhängigkeiten
- Verwenden Mocks/Stubs für externe Services
- Schnell und deterministisch
- Fokus: Korrektheit einzelner Funktionen

### Integration Tests (`tests/integration/`)
- Testen Interaktionen zwischen Services
- Können externe Ressourcen verwenden (Datenbanken, APIs)
- Langsamer als Unit-Tests
- Fokus: Service-Kommunikation und Datenfluss

### Service Tests (`tests/services/`)
- Service-spezifische Tests organisiert nach Microservice
- Unit- und Integration-Tests für jeden Service
- Fokus: Service-Logik und -Integration

### End-to-End Tests (`tests/e2e/`)
- Testen komplette Workflows vom Daten-Import bis zur Trade-Ausführung
- Verwenden reale Service-Instanzen (in Containern)
- Sehr langsam, aber höchste Confidence
- Fokus: System als Ganzes funktioniert

## Zentrale Test-Konfiguration

### conftest.py
Die zentrale `conftest.py` im `tests/` Root-Verzeichnis bietet:

- **Gemeinsame Fixtures**: `sample_ohlcv_data`, `mock_redis`, `mock_http_session`, etc.
- **Test Utilities**: `TestUtils` Klasse mit Helper-Funktionen
- **Async Support**: Event loop fixtures für asyncio Tests
- **Test Markers**: Automatische Registrierung von pytest Markern
- **Cleanup**: Automatische Bereinigung temporärer Dateien

### Verfügbare Fixtures
- `sample_ohlcv_data`: Beispiel OHLCV Daten für Tests
- `mock_redis`: Gemockter Redis Client
- `mock_http_session`: Gemockte HTTP Session
- `test_config`: Sichere Test-Konfiguration
- `mock_service_client`: Gemockter Service Client
- `performance_metrics`: Beispiel Performance-Metriken
- `test_utils`: TestUtils Instanz mit Helper-Methoden

## Ausführung

### Alle Tests
```bash
pytest
```

### Nach Test-Typ
```bash
# Nur Unit-Tests (schnell)
pytest -m "unit"

# Nur Integration-Tests
pytest -m "integration"

# Nur Service-Tests
pytest -m "service"

# Nur E2E-Tests (langsam)
pytest -m "e2e"
```

### Nach Verzeichnis
```bash
# Unit-Tests
pytest tests/unit/

# Integration-Tests
pytest tests/integration/

# Service-spezifische Tests
pytest tests/services/core/
pytest tests/services/data_service/

# E2E-Tests
pytest tests/e2e/
```

### Mit Coverage
```bash
pytest --cov=services --cov-report=html
pytest --cov=services --cov-report=term-missing
```

### Performance Tests
```bash
pytest -m "performance" --durations=10
```

### Debug-Modus
```bash
pytest -v -s --tb=long
```

## Best Practices

- **Test-Isolation**: Jeder Test ist unabhängig und kann in beliebiger Reihenfolge ausgeführt werden
- **Descriptive Names**: Test-Methoden beschreiben das erwartete Verhalten (`test_should_calculate_ema_correctly`)
- **Arrange-Act-Assert**: Klare Struktur in jedem Test
- **Mock External Dependencies**: Isolieren von externen Services und APIs
- **Test Data Management**: Verwenden von Test-Fixtures für wiederholbare Daten

## Coverage Ziele

- **Unit Tests**: >90% Coverage
- **Integration Tests**: >80% Coverage
- **Gesamt**: >85% Coverage

## CI/CD Integration

Tests werden automatisch in der CI/CD Pipeline ausgeführt:
- Unit-Tests bei jedem Push
- Integration-Tests nightly
- Performance-Tests weekly