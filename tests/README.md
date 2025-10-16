# TradPal Test Suite

Diese Test-Suite ist nach Best-Practices für Microservices-Architekturen organisiert.

## Struktur

```
tests/
├── unit/                          # Unit-Tests (isolierte Komponenten)
│   ├── test_*.py                 # Einzelne Funktionen/Klassen
│   └── __init__.py
├── integration/                   # Integration-Tests (Service-Interaktionen)
│   ├── test_*.py                 # Cross-Service Tests
│   └── __init__.py
├── services/                      # Service-spezifische Tests
│   ├── core/                      # Core Service Tests
│   │   ├── test_memory_optimization.py
│   │   ├── test_vectorization.py
│   │   └── test_indicators.py
│   ├── data_service/              # Data Service Tests
│   │   ├── test_data_fetcher.py
│   │   └── test_cache.py
│   ├── trading_bot_live/          # Trading Bot Tests
│   ├── backtesting_service/       # Backtesting Tests
│   ├── discovery_service/         # Parameter Optimization Tests
│   ├── risk_service/              # Risk Management Tests
│   ├── notification_service/      # Notification Tests
│   ├── mlops_service/             # ML Operations Tests
│   ├── security_service/          # Security Tests
│   └── web_ui/                    # Web UI Tests
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

## Ausführung

### Alle Tests
```bash
pytest
```

### Nur Unit-Tests
```bash
pytest tests/unit/
```

### Nur Integration-Tests
```bash
pytest tests/integration/
```

### Service-spezifische Tests
```bash
pytest tests/services/core/
pytest tests/services/data_service/
```

### Mit Coverage
```bash
pytest --cov=services --cov-report=html
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