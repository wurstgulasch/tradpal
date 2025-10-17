# Copilot Instructions for AI Agents

## Überblick
**TradPal** ist ein vollautonomes AI Trading System basierend auf einer vollständigen Microservices-Architektur. Ziel ist die konsistente Outperformance von Buy&Hold und traditionellen Indikatoren durch fortschrittliche ML-Modelle, Ensemble-Methoden und Risikomanagement.

**Aktueller Stand (Oktober 2025):** Version 3.0.1 mit kompletter Microservices-Migration und 100% Test-Coverage für implementierte Features (490 Tests bestehen).

## Projektstruktur (STRENG EINHALTEN!)
- `services/`: **Microservices-Architektur** - ALLE neuen Features/Service-Entwicklungen hier implementieren
  - `core/`: Kernberechnungen (Indikatoren, Vektorisierung, Memory-Optimierung)
  - `data_service/`: Daten-Management (CCXT, Kaggle Bitcoin Datasets, Yahoo Finance, Caching, HDF5)
  - `trading_bot_live/`: Live-Trading-Engine mit AI-Modellen
  - `backtesting_service/`: Historische Simulation und Performance-Analyse
  - `discovery_service/`: ML-Parameter-Optimierung und Genetische Algorithmen
  - `risk_service/`: Risikomanagement und Position-Sizing
  - `notification_service/`: Alerts (Telegram, Discord, Email)
  - `mlops_service/`: ML-Experiment-Tracking und Model-Management
  - `security_service/`: Zero-Trust-Authentifizierung
  - `web_ui/`: Streamlit/Plotly Dashboard
- `config/`: Zentrale Konfiguration (settings.py, .env-Dateien)
- `tests/`: Unit- und Integrationstests (pytest)
- `scripts/`: Utility-Scripts für Training/Demos
- `integrations/`: Externe Integrationen (Broker, Notifications)
- `examples/`: Jupyter-Notebooks und Demos
- `docs/`: Dokumentation
- `main.py`: Hybrid-Orchestrator mit Service-Clients

**WICHTIG:** Keine Dateien im Root-Verzeichnis ablegen! Neue Features immer in entsprechenden Services implementieren (Microservice-Architektur beibehalten). Terminals immer in der conda Umgeung `tradpal-env` ausführen. Dokumentation, README und Commit Messages immer auf Englisch und immer aktuell halten. Dokumentation, README und copilot-instructions.md immer mit dem Code synchronisieren.

## Architektur-Prinzipien
- **Microservices-First:** Jede neue Funktionalität als separater Service
- **Event-Driven:** Redis Streams für Echtzeit-Kommunikation zwischen Services
- **API Gateway:** Zentrales Routing, Authentifizierung und Load Balancing
- **Zero-Trust-Security:** mTLS, OAuth/JWT, Secrets-Management
- **Observability:** Prometheus/Grafana Monitoring, verteilte Tracing, Metriken, Logs
- **Resilience:** Circuit Breaker, Retry-Patterns, Health Checks, Chaos Engineering

## Entwicklungsfokus 2025

✅ **Abgeschlossen:**
- Event-Driven Architecture mit Redis Streams
- API Gateway mit Service Discovery und Load Balancing
- Zentralisiertes Monitoring mit Prometheus/Grafana
- Circuit Breaker und Health Check Resilience Patterns

🔄 **In Arbeit:**
1. **AI-Outperformance:** ML-Modelle die konsistent Benchmarks übertreffen
2. **Service-Optimierung:** Performance, Skalierbarkeit, Reliability
3. **Advanced Features:** Reinforcement Learning, Market Regime Detection, Alternative Data
4. **Datenquellen-Erweiterung:** Modulare Datenquellen-Architektur mit Kaggle Bitcoin Datasets für verbessertes Backtesting
5. **Enterprise-Readiness:** Security, Monitoring, Deployment-Automatisierung

## Code-Konventionen
- **Microservice-Struktur:** Services sind unabhängig deploybar
- **Async-First:** asyncio für alle I/O-Operationen
- **Type-Safety:** Vollständige Type-Hints
- **Testing:** >90% Coverage, Integrationstests für Service-Interaktionen
- **Dokumentation:** Englische Docstrings, automatische API-Docs
- **Testing:** Neue Features immer mit Unit-Tests und Integrationstests absichern und im Ordner tests/ ablegen

## Wichtige Workflows
- **Trading Bot:** AI-gestützte Signalgenerierung mit Risikomanagement
- **Backtesting:** Walk-Forward-Analyse mit ML-Modellen und modularen Datenquellen (Kaggle, CCXT, Yahoo Finance)
- **ML-Pipeline:** Optuna-Optimierung, Ensemble-Methoden, SHAP-Explainability
- **Live-Trading:** Paper-Trading → Live-Trading mit Broker-Integration

## Anweisungen für Copilot
- **Microservices-Architektur einhalten:** Neue Features immer in Services/, nie in Root oder src/
- **Service-Clients verwenden:** Für Cross-Service-Kommunikation
- **Async-Patterns:** asyncio für alle Netzwerk-I/O
- **Testing-First:** Unit-Tests vor Implementierung
- **Performance-Optimierung:** Memory-Mapped Files, Chunked Processing, GPU-Training
- **Datenquellen-Architektur:** Verwende modulare Datenquellen (Kaggle, Yahoo Finance, CCXT) für optimale Backtesting-Ergebnisse
- **Security-by-Design:** Zero-Trust-Prinzipien in allen Services
- **Dokumentation:** README und docs/ aktuell halten und auf Englisch schreiben
- **Neue Features:** Immer mit Unit-Tests und Integrationstests absichern und im Ordner tests/ ablegen
- **Root-Verzeichnis sauber halten:** Keine Code-Dateien im Root, nur config/, services/, tests/, scripts/, docs/, integrations/, examples/

## Kritische Entwicklungs-Workflows
- **Environment Setup:** `conda env create -f environment.yml && conda activate tradpal-env`
- **Profile-basierte Ausführung:** `python main.py --profile light --mode live` (light für minimal, heavy für voll)
- **Testing:** `pytest tests/` (Unit-Tests in `tests/unit/`, Integration in `tests/integration/`)
- **Performance Benchmarking:** `python scripts/performance_benchmark.py`
- **ML Training:** `python scripts/train_ml_model.py`
- **Backtesting:** `python main.py --mode backtest --start-date 2024-01-01`

## Projekt-spezifische Patterns
- **Service-Client Pattern:** Jeder Service hat einen async Client mit `authenticate()` für Zero-Trust
- **mTLS Setup:** Services verwenden mutual TLS mit Zertifikaten aus `cache/security/certs/`
- **Async Context Managers:** HTTP Sessions mit `@asynccontextmanager` für sichere Verbindungen
- **Configuration Hierarchy:** `config/settings.py` lädt `.env` basierend auf Profile (light/heavy)
- **Memory Optimization:** `MemoryMappedData`, `RollingWindowBuffer` für große Datasets
- **GPU Acceleration:** Automatische GPU-Erkennung und optimale Device-Auswahl
- **Data Mesh:** Domains wie `market_data`, `trading_signals` mit Governance und Quality Rules
- **Fitness Functions:** Gewichtete Metriken (Sharpe 30%, Calmar 25%, P&L 30%) für Backtesting

## Integration Points
- **API Gateway:** Zentrales Service Routing (Port 8000)
- **Event Service:** Event-Driven Kommunikation (Port 8011)
- **Broker APIs:** CCXT für Exchanges, mit Testnet-Support
- **Notifications:** Telegram/Discord/Email via `notification_service`
- **Data Sources:** CCXT, Yahoo Finance, Funding Rate spezialisiert
- **ML Frameworks:** scikit-learn, PyTorch (optional), Optuna für Hyperparameter
- **Monitoring:** Prometheus/Grafana/AlertManager für alle Services
- **Event Streaming:** Redis Streams für Service-Kommunikation
- **Security:** Vault für Secrets, JWT für Auth, mTLS für Service-to-Service

## Debugging und Troubleshooting
- **Logs:** Alle Logs in `logs/tradpal.log`, rotierend mit 10MB Limit
- **Cache:** ML-Modelle in `cache/ml_models/`, Daten in `cache/`
- **Output:** Backtest-Results in `output/`, Performance in `output/paper_performance.json`
- **Event Streams:** Redis Streams für Event-Debugging und Replay
- **API Gateway:** Zentrales Logging für alle Service-Requests
- **Monitoring:** Prometheus Metrics für alle Services und Events
- **Performance Monitoring:** CPU, Memory, GPU-Nutzung in Echtzeit
- **Error Handling:** Circuit Breaker Pattern für Service-Ausfälle
- **Rate Limiting:** Adaptive Rate Limiting für API-Calls

## Empfehlungen für die Zukunft
1. **Service Mesh:** Istio-Integration für erweiterte Service-Kommunikation
2. **Advanced Event Processing:** Complex Event Processing und Event Correlation
3. **CI/CD-Verbesserungen:** GitHub Actions für automatisierte Tests und Deployments
4. **Docker-Organisation:** Multi-stage Builds für optimierte Service-Container
5. **API-Dokumentation:** OpenAPI/Swagger für alle Service-APIs
6. **Monitoring-Setup:** Erweiterte Alerting-Regeln und Dashboards
7. **Security-Scanning:** Automatisierte Security-Tests und Vulnerability Scanning
8. **Performance-Optimierung:** Erweiterte GPU-Unterstützung und verteiltes Training

*Letzte Aktualisierung: 17.10.2025*