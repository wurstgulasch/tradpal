# TradPal - AI Trading System

TradPal ist ein vollautonomes AI Trading System basierend auf einer vollständigen Microservices-Architektur. Ziel ist die konsistente Outperformance von Buy&Hold und traditionellen Indikatoren durch fortschrittliche ML-Modelle, Ensemble-Methoden und Risikomanagement.

## 🏗️ Projektstruktur

```
tradpal_indicator/
├── services/                    # Microservices-Architektur
│   ├── core/                    # Kernberechnungen & Memory-Optimierung
│   ├── data_service/            # Daten-Management (CCXT, Caching, HDF5)
│   ├── trading_bot_live/        # Live-Trading-Engine mit AI-Modellen
│   ├── backtesting_service/     # Historische Simulation
│   ├── discovery_service/       # ML-Parameter-Optimierung
│   ├── risk_service/            # Risikomanagement & Position-Sizing
│   ├── notification_service/    # Alerts (Telegram, Discord, Email)
│   ├── mlops_service/           # ML-Experiment-Tracking
│   ├── security_service/        # Zero-Trust-Authentifizierung
│   └── web_ui/                  # Streamlit/Plotly Dashboard
├── config/                      # Zentrale Konfiguration
│   ├── settings.py              # Hauptkonfiguration
│   ├── .env                     # Environment-Variablen
│   ├── .env.example             # Beispiel-Konfiguration
│   ├── .env.light               # Light-Profil (ohne KI/ML)
│   └── .env.heavy               # Heavy-Profil (volle Features)
├── data/                        # Daten-Verzeichnisse
│   ├── cache/                   # Cache-Dateien
│   ├── logs/                    # Log-Dateien
│   └── output/                  # Ausgabe-Dateien (Backtests, Reports)
├── infrastructure/              # Infrastruktur & Deployment
│   ├── deployment/              # AWS, Kubernetes Konfigurationen
│   └── monitoring/              # Prometheus, Grafana Setups
├── tests/                       # Test-Suite (nach Best-Practices organisiert)
│   ├── unit/                    # Unit-Tests
│   ├── integration/             # Integration-Tests
│   └── services/                # Service-spezifische Tests
├── scripts/                     # Utility-Scripts für Training/Demos
├── examples/                    # Jupyter-Notebooks und Demos
├── integrations/                # Externe Integrationen
├── docs/                        # Dokumentation
├── main.py                      # Hybrid-Orchestrator mit Service-Clients
└── pyproject.toml               # Python-Projekt-Konfiguration
```

## 🚀 Schnellstart

### Voraussetzungen
- Python 3.10+
- Conda/Minconda
- Git

### Installation

1. **Repository klonen:**
   ```bash
   git clone https://github.com/wurstgulasch/tradpal.git
   cd tradpal_indicator
   ```

2. **Environment einrichten:**
   ```bash
   conda env create -f environment.yml
   conda activate tradpal-env
   ```

3. **Konfiguration:**
   ```bash
   cp config/.env.example config/.env
   # .env Datei anpassen
   ```

4. **Tests ausführen:**
   ```bash
   pytest tests/
   ```

### Verwendung

```bash
# Live-Trading mit Light-Profil (ohne KI)
python main.py --profile light --mode live

# Backtest mit allen Features
python main.py --profile heavy --mode backtest --start-date 2024-01-01

# Performance-Benchmark
python scripts/performance_benchmark.py
```

## 🧪 Test-Organisation

Die Test-Suite folgt Best-Practices für Microservices:

- **Unit-Tests** (`tests/unit/`): Isolierte Komponenten-Tests
- **Integration-Tests** (`tests/integration/`): Service-Interaktionen
- **Service-Tests** (`tests/services/`): Service-spezifische Tests

```bash
# Alle Tests
pytest

# Nur Unit-Tests
pytest tests/unit/

# Service-spezifische Tests
pytest tests/services/core/
```

## 📊 Performance & Benchmarks

Aktuelle Benchmarks zeigen signifikante Verbesserungen:

- **Memory-Optimierung**: 10.25x schneller als traditionelle Methoden
- **Memory-Verbrauch**: Konstant niedrig (~85 MB) unabhängig von Datengröße
- **Test-Coverage**: >90% für alle Services

Detaillierte Benchmarks: [PERFORMANCE_PROFILES.md](PERFORMANCE_PROFILES.md)

## 🏛️ Architektur-Prinzipien

- **Microservices-First**: Jede neue Funktionalität als separater Service
- **Event-Driven**: Apache Kafka/Redis Streams für Service-Kommunikation
- **Zero-Trust-Security**: mTLS, OAuth/JWT, Secrets-Management
- **Observability**: Distributed Tracing, Metrics, Logs
- **Resilience**: Circuit Breaker, Retry-Patterns, Chaos Engineering

## 🔧 Entwicklung

### Neue Features hinzufügen
1. Service in `services/` erstellen
2. Tests in `tests/services/` hinzufügen
3. Dokumentation aktualisieren
4. CI/CD Pipeline erweitern

### Code-Qualität
- **Type-Safety**: Vollständige Type-Hints
- **Testing**: >90% Coverage, Integrationstests
- **Linting**: flake8, mypy
- **Formatting**: black, isort

## 📈 Roadmap 2025

1. **AI-Outperformance**: ML-Modelle die konsistent Benchmarks übertreffen
2. **Service-Optimierung**: Performance, Skalierbarkeit, Reliability
3. **Advanced Features**: Reinforcement Learning, Market Regime Detection
4. **Enterprise-Readiness**: Security, Monitoring, Deployment-Automatisierung

## 🤝 Beitragen

1. Fork das Repository
2. Feature-Branch erstellen (`git checkout -b feature/AmazingFeature`)
3. Tests schreiben
4. Commit machen (`git commit -m 'Add some AmazingFeature'`)
5. Pushen (`git push origin feature/AmazingFeature`)
6. Pull Request erstellen

## 📄 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe [LICENSE](LICENSE) Datei für Details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/wurstgulasch/tradpal/issues)
- **Dokumentation**: [docs/](docs/)
- **Beispiele**: [examples/](examples/)

---

**TradPal v3.0.0** - *Letzte Aktualisierung: Oktober 2025*
