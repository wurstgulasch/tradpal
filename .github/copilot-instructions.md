# Copilot Instructions for AI Agents

## Overview
**TradPal** is a fully autonomous AI trading system based on a complete microservices architecture. The goal is consistent outperformance of Buy&Hold and traditional indicators through advanced ML models, ensemble methods, and risk management.

**Current Status (October 2025):
 **complete microservices consolidation** (4 core services + 14 supporting services) and 98 organized test files with comprehensive coverage. **Trading Service fully consolidated with moderate isolation pattern** - code consolidation with runtime separation for resource management.

## Project Structure (STRICTLY ENFORCE!)
- `services/`: **Microservices Architecture** - ALL new features/service developments go here (18 consolidated services)
  - `core_service/`: Core calculations (indicators, vectorization, memory optimization)
  - `trading_service/`: **Consolidated trading service** - All trading functionality in one service
    - `ml_training_service/`: Specialized ML model training and ensemble methods
    - `reinforcement_learning_service/`: Advanced RL implementation with market regime awareness
    - `market_regime_service/`: Market regime detection and multi-timeframe analysis
    - `risk_management_service/`: Position sizing and portfolio risk assessment
    - `trading_execution_service/`: Order execution and portfolio management
    - `backtesting_service/`: Historical simulation with moderate isolation
    - `backtesting_worker.py`: Dedicated worker process for resource isolation
    - `trading_ai_service/`: AI-powered trading orchestration
    - `trading_bot_live_service/`: Live execution
    - `central_service_client.py`: Unified client for all microservice communication
    - `orchestrator.py`: Coordinates trading operations across specialized services
  - `data_service/`: Data management (CCXT, Kaggle Bitcoin datasets, Yahoo Finance, caching, HDF5)
  - `infrastructure_service/`: **Consolidated infrastructure service** - All platform infrastructure
    - `api_gateway_service/`: API routing & authentication
    - `event_system_service/`: Event-driven communication
    - `security_service/`: Authentication & security
    - `falco_security_service/`: Runtime monitoring
  - `monitoring_service/`: **Consolidated monitoring service** - All monitoring & observability
    - `notification_service/`: Alerts & notifications
    - `alert_forwarder_service/`: Alert processing
    - `mlops_service/`: ML experiment tracking
    - `discovery_service/`: Parameter optimization
  - `ui_service/`: **Consolidated UI service** - All user interfaces
    - `web_ui_service/`: Web interface
- `config/`: Central configuration (modular structure)
  - `settings.py`: Main configuration file (imports from modules + legacy constants for backward compatibility)
  - `core_settings.py`: Core trading and risk management
  - `ml_settings.py`: Machine learning and AI configurations
  - `service_settings.py`: Microservices and data mesh settings
  - `security_settings.py`: Security and authentication
  - `performance_settings.py`: Performance optimization
  - `.env` files: Environment variables for different profiles
- `tests/`: **Centralized test suite** (98 test files organized by best practices)
  - `conftest.py`: **Central test configuration** with fixtures and utilities
  - `unit/`: Unit tests (25+ files) - isolated component testing
  - `integration/`: Integration tests (13+ files) - service interaction testing
  - `services/`: Service-specific tests organized by service
  - `e2e/`: End-to-end tests for complete workflows
  - `config/`: Configuration tests
  - `integrations/`: Integration setup tests
- `scripts/`: Utility scripts for training/demos
- `integrations/`: External integrations (brokers, notifications)
- `examples/`: Jupyter notebooks and demos
- `docs/`: Documentation
- `main.py`: Hybrid orchestrator with service clients

**CRITICAL:** Never place files in root directory! New features always in services/. Terminals always in conda environment `tradpal_env`. Documentation, README, and copilot-instructions.md always synchronized and in english (commit messages as well).

## Architecture Principles
- **Modular Monolith:** Similar to a modular monolith, but each module is a separate service
- **Microservices-Architecture:** create independent, loosely coupled services where it is reasonable to do so
- **Event-Driven:** Redis Streams for real-time service communication
- **API Gateway:** Centralized routing, authentication, and load balancing at port 8000
- **Zero-Trust Security:** mTLS, OAuth/JWT, secrets management
- **Observability:** Prometheus/Grafana monitoring, distributed tracing, metrics, logs
- **Resilience:** Circuit breaker, retry patterns, health checks, chaos engineering
- **Data Mesh:** Modular data sources with governance and quality rules
- **Async-First:** asyncio for all I/O operations
- **Dependency Management:** Centralized catalog with service-specific requirements for true microservice independence

## Development Focus 2025

✅ **Completed:**
- Event-Driven Architecture with Redis Streams
- API Gateway with service discovery and load balancing
- Centralized monitoring with Prometheus/Grafana
- Circuit breaker and health check resilience patterns
- Modular data sources (Kaggle Bitcoin datasets, Yahoo Finance, CCXT)
- Zero-trust security with mTLS and JWT
- **Complete Service Consolidation**: 18 services with modular architecture (4 core + 14 supporting)
- **Centralized Test Suite**: 98 organized test files with comprehensive coverage
- **Modular Service Pattern**: All services follow service.py/client.py/main.py structure
- **Trading Service Moderate Isolation**: Code consolidation with runtime separation for resource management

🔄 **In Progress:**
1. **AI Outperformance**: ML models that consistently outperform benchmarks
2. **Service Optimization**: Performance, scalability, reliability
3. **Advanced Features**: Reinforcement learning, market regime detection, alternative data
4. **Data Sources Expansion**: Additional datasets and real-time feeds
5. **Enterprise Readiness**: Security, monitoring, deployment automation

## Critical Developer Workflows

### Environment Setup
```bash
# One-time setup (Docker-based recommended)
make setup

# Start development services
make dev-up

# Open Web UI (optional)
make dev-ui

# Alternative: Traditional setup
conda env create -f environment.yml
conda activate tradpal_env
cp config/.env.example config/.env
```

### Profile-Based Execution
```bash
# Live trading with light profile (minimal features, no AI/ML)
python main.py --profile light --mode live

# Backtest with all features
python main.py --profile heavy --mode backtest --start-date 2024-01-01 --data-source kaggle

# Backtesting Worker (isolated process for resource management)
python main.py backtesting-worker

# Performance benchmark
python scripts/performance_benchmark.py
```

### Testing Workflows
```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Service-specific tests
pytest tests/services/core_service/
pytest tests/services/ml_training_service/
pytest tests/services/reinforcement_learning_service/
pytest tests/services/market_regime_service/
pytest tests/services/risk_management_service/
pytest tests/services/trading_execution_service/
pytest tests/services/trading_ai_service/
pytest tests/services/backtesting_service/
pytest tests/services/trading_service/backtesting_worker/
pytest tests/services/data_service/
pytest tests/services/api_gateway_service/
pytest tests/services/notification_service/

# Test with coverage
pytest --cov=services --cov-report=html
```

### Docker Development
```bash
# Start monitoring stack
make dev-up

# Check service health
make health-check

# View logs
make logs-data

# Access service shell
make shell-data
```

## Code Conventions and Patterns

### Service Architecture Blueprint (Best Practices)

**STRICTLY ENFORCE this blueprint for ALL new services:**

```
services/{service_name}/
├── __init__.py                    # Package initialization
├── main.py                        # FastAPI service entry point with health checks
├── client.py                      # Async client with circuit breaker & authentication
├── service.py                     # Core business logic (optional, for complex services)
├── requirements.txt               # Service-specific dependencies ONLY
├── README.md                      # Service documentation with API endpoints
├── tests.py                       # Unit tests for service logic
└── test_integration.py           # Integration tests (optional)
```

**Required Implementation Patterns:**

1. **Async-First Design**
   - ALL I/O operations use `asyncio`
   - HTTP clients use `@asynccontextmanager`
   - Circuit breaker integration for ALL external calls

2. **Zero-Trust Security**
   - `await client.authenticate()` on service initialization
   - mTLS certificates from `cache/security/certs/`
   - JWT token validation for API access

3. **Event-Driven Communication**
   - Redis Streams for inter-service communication
   - Event publishing for state changes
   - Event subscription for reactive behavior

4. **Resilience Patterns**
   - Circuit breaker for external dependencies
   - Health checks at `/health` endpoint
   - Graceful shutdown handling

5. **Configuration Management**
   - Service-specific settings in `config/service_settings.py`
   - Environment variable loading via `.env`
   - Lazy loading to reduce memory footprint

6. **Testing Standards**
   - Unit tests in `tests/unit/`
   - Integration tests in `tests/integration/`
   - Async testing with `pytest-asyncio`

7. **Documentation Requirements**
   - README.md with API documentation
   - Inline code documentation
   - Service dependencies clearly stated

**Service Client Pattern (MANDATORY):**
```python
class {ServiceName}Client:
    def __init__(self):
        self.session = None
        self.circuit_breaker = None

    async def authenticate(self) -> None:
        """Zero-trust authentication"""

    @asynccontextmanager
    async def _get_session(self):
        """Circuit breaker protected session"""

    async def {business_method}(self, **params) -> Dict[str, Any]:
        """Business logic with error handling"""
```

**Event Integration Pattern:**
```python
# Publishing events
await event_system.publish_event(Event(
    event_type=EventType.{EVENT_TYPE},
    source=self.service_name,
    data={...}
))

# Subscribing to events
event_system.register_handler(EventType.{EVENT_TYPE}, self._handle_event)
```

### Trading Service Best Practices (Implemented Blueprint)

**✅ VERIFIED: trading_service/ follows ALL patterns above**

**Specialized Service Architecture:**
- **Consolidated Codebase**: All trading logic in one service for cohesion
- **Runtime Separation**: Moderate isolation via dedicated worker processes
- **Event-Driven Orchestration**: Services communicate via Redis Streams
- **Resource Management**: CPU affinity, memory limits, process prioritization

**Service Dependencies (Trading Domain):**
- **Independent Services**: ML Training, RL, Market Regime (no external deps)
- **Orchestrated Services**: Trading AI coordinates specialized services
- **Infrastructure Services**: Data, Security, Event System (shared resources)
- **Moderate Isolation**: Backtesting runs isolated but shares codebase

**Performance Optimization:**
- **GPU Acceleration**: Automatic detection and optimal device selection
- **Memory Mapping**: Large datasets use memory-mapped files
- **Chunked Processing**: Data processed in configurable chunks
- **Async Processing**: All operations are non-blocking

**Quality Assurance:**
- **98 Test Files**: Comprehensive coverage across all services
- **Integration Testing**: Service interaction validation
- **Performance Benchmarking**: Automated performance regression testing
- **Chaos Engineering**: Resilience testing for production readiness

### Service Development Templates

**Available Templates:**
- `services/service_template.py` - Complete service implementation template
- `services/README_TEMPLATE.md` - README documentation template

**Usage:**
```bash
# Create new service from template
cp services/service_template.py services/new_service/main.py
cp services/README_TEMPLATE.md services/new_service/README.md

# Customize for your service requirements
# Follow all patterns in the Service Architecture Blueprint
```

### Async-First Design
- **All I/O operations use asyncio**: Network calls, file operations, service communication
- **Context managers for connections**: `@asynccontextmanager` for HTTP sessions
- **Circuit breaker integration**: All HTTP clients use circuit breaker protection
- **Example**:
```python
async def fetch_data(self, symbol: str) -> Dict[str, Any]:
    async with self.session.get(f"{self.base_url}/data/{symbol}") as response:
        return await response.json()
```

### Zero-Trust Security Pattern
- **mTLS for service-to-service**: Mutual TLS with client certificates
- **JWT authentication**: Token-based API access through API Gateway
- **Service authentication**: `await client.authenticate()` on service initialization
- **Certificate paths**: `cache/security/certs/` for mTLS certificates

### Configuration Hierarchy
- **Modular config files**: `core_settings.py`, `ml_settings.py`, `service_settings.py`
- **settings.py imports all**: Main file imports from modules + legacy constants
- **Environment profiles**: `.env` (unified configuration)
- **Dynamic loading**: Settings loaded from unified .env configuration

### Dependency Management
- **Central catalog**: `dependency_catalog.txt` defines approved package versions
- **Service isolation**: Each service has `requirements.txt` with only needed packages
- **Version consistency**: All services use exact versions from catalog
- **Management scripts**: `python scripts/manage_dependencies.py validate`

### Event-Driven Communication
- **Redis Streams**: Asynchronous communication between services
- **Predefined event types**: `market_data_update`, `trading_signal`, `order_executed`, etc.
- **Publisher-subscriber**: Services publish events, others subscribe
- **Event replay**: Historical events available for backtesting/debugging

### Service Dependencies (Post-Consolidation)
- **Core Service**: Independent (uses event_system for communication)
- **ML Training Service**: Independent (uses data_service for training data)
- **Reinforcement Learning Service**: Independent (uses market_regime_service for regime awareness)
- **Market Regime Service**: Independent (uses core_service for indicators)
- **Risk Management Service**: Independent (uses core_service for calculations)
- **Trading Execution Service**: Depends on data_service, security_service
- **Trading AI Service**: Orchestrates ml_training_service, reinforcement_learning_service, market_regime_service, risk_management_service, trading_execution_service
- **Backtesting Service**: Depends on data_service, security_service, all trading services (moderate isolation via dedicated worker process)
- **Trading Bot Live Service**: Depends on core_service, data_service, all trading services
- **Data Service**: Depends on security_service for authentication
- **API Gateway Service**: Independent routing layer
- **Event System Service**: Independent communication layer
- **Security Service**: Independent authentication layer
- **Falco Security Service**: Depends on security_service
- **Notification Service**: Depends on security_service
- **Alert Forwarder Service**: Depends on notification_service
- **MLOps Service**: Depends on notification_service, ml_training_service
- **Discovery Service**: Depends on backtesting_service, security_service, all ML services
- **Web UI Service**: Depends on api_gateway_service, various monitoring services

## Integration Points

### Data Sources Architecture
- **Modular design**: Factory pattern in `services/data_service/data_sources/factory.py`
- **8+ data sources**: Kaggle Bitcoin datasets, Yahoo Finance, CCXT, sentiment, onchain, volatility
- **Fallback chains**: Liquidation → Volatility → Sentiment → On-chain → Open Interest
- **Quality validation**: OHLC validation, NaN handling, consistent formats

### Broker API Integration
- **CCXT abstraction**: Unified API for 100+ exchanges
- **Testnet support**: Paper trading on exchange testnets
- **Order types**: Market, limit, stop orders
- **Position management**: Real-time P&L, stop-loss, take-profit

### Monitoring Stack
- **Prometheus metrics**: Custom metrics from all services (`/metrics` endpoints)
- **Grafana dashboards**: Trading performance, system health, risk metrics
- **AlertManager**: Automated alerts for system issues
- **Service discovery**: Automatic service registration

### ML Frameworks Integration
- **scikit-learn**: Traditional ML models
- **PyTorch**: Deep learning with GPU acceleration
- **Optuna**: Hyperparameter optimization
- **SHAP**: Model explainability
- **Ensemble methods**: Multiple models combined for better performance

### Security Integration
- **Vault**: Secrets management
- **JWT**: API authentication
- **mTLS**: Service-to-service encryption
- **Audit logging**: Complete activity trails

## Testing Patterns

### Test Organization
- **Unit tests**: `tests/unit/` - Isolated component testing (25+ files)
- **Integration tests**: `tests/integration/` - Service interaction testing (13+ files)
- **Service tests**: `tests/services/` - Service-specific functionality organized by service
- **E2E tests**: `tests/e2e/` - End-to-end tests for complete workflows
- **Config tests**: `tests/config/` - Configuration testing
- **Integration tests**: `tests/integrations/` - Integration setup testing
- **Async testing**: `pytest-asyncio` for async test functions

### Test Coverage Goals
- **100% coverage** for implemented features
- **Integration tests** for service interactions
- **Performance tests** for critical paths
- **Chaos engineering** tests for resilience

## Performance Optimization Patterns

### Memory Management
- **Memory-mapped data**: `MemoryMappedData` for large datasets
- **Chunked processing**: Process data in chunks to manage memory
- **Rolling windows**: `RollingWindowBuffer` for time-series analysis

### GPU Acceleration
- **Automatic detection**: GPU availability checked at startup
- **Optimal device selection**: CUDA vs CPU based on hardware
- **Model optimization**: PyTorch models optimized for inference

### Caching Strategy
- **Redis caching**: Fast access to frequently used data
- **Intelligent TTL**: Different TTL for live vs historical data
- **Cache invalidation**: Automatic invalidation on data updates

## Debugging and Troubleshooting

### Logs and Monitoring
- **Centralized logging**: All logs in `logs/tradpal.log`
- **Structured logging**: JSON format with correlation IDs
- **Log rotation**: 10MB files with 5 backup files
- **Event debugging**: Redis streams for event replay

### Common Issues
- **Service connectivity**: Check mTLS certificates and JWT tokens
- **Memory issues**: Monitor with `psutil`, check for memory leaks
- **Event delays**: Check Redis stream length and consumer groups
- **Performance**: Profile with `cProfile`, check GPU utilization

### Development Tools
- **Health checks**: `GET /health` on all services
- **Metrics endpoints**: `GET /metrics` for Prometheus scraping
- **Debug mode**: Set `LOG_LEVEL=DEBUG` for verbose logging
- **Cache clearing**: `python scripts/manage_cache.py clear`

## Future Development Patterns

### Service Mesh Integration
- **Istio**: Advanced service-to-service communication
- **Traffic management**: Load balancing, circuit breaking, retries
- **Observability**: Distributed tracing, metrics collection

### Advanced Event Processing
- **Complex event processing**: Event correlation and aggregation
- **Event sourcing**: State management through events
- **Event-driven sagas**: Distributed transaction coordination

### AI/ML Advancements
- **Reinforcement learning**: RL agents for trading decisions
- **Market regime detection**: Automatic strategy adaptation
- **Alternative data**: Sentiment, on-chain, economic indicators

## Anweisungen für Copilot

- **Microservices-Architektur einhalten:** Neue Features immer in services/, nie in Root oder src/
- **Modulare Konfiguration verwenden:** Konfigurationseinstellungen in entsprechenden config/ Modulen ablegen (core_settings.py, ml_settings.py, etc.)
- **Legacy-Konstanten in settings.py:** Aktuell hybride Struktur mit modularen Imports + Legacy-Konstanten für Abwärtskompatibilität beibehalten
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
- **Environment Setup:** `conda env create -f environment.yml && conda activate tradpal_env`
- **Unified Execution:** `python main.py --mode live` (all features controlled via .env)
- **Testing:** `pytest tests/` (Unit-Tests in `tests/unit/`, Integration in `tests/integration/`)
- **Performance Benchmarking:** `python scripts/performance_benchmark.py`
- **ML Training:** `python scripts/train_ml_model.py`
- **Backtesting:** `python main.py --mode backtest --start-date 2024-01-01` (or `python main.py backtesting-worker` for isolated execution)

## Projekt-spezifische Patterns
- **Service-Client Pattern:** Jeder Service hat einen async Client mit `authenticate()` für Zero-Trust
- **mTLS Setup:** Services verwenden mutual TLS mit Zertifikaten aus `cache/security/certs/`
- **Async Context Managers:** HTTP Sessions mit `@asynccontextmanager` für sichere Verbindungen
- **Configuration Hierarchy:** `config/settings.py` lädt `.env` basierend auf Profile (light/heavy)
- **Memory Optimization:** `MemoryMappedData`, `RollingWindowBuffer` für große Datasets
- **GPU Acceleration:** Automatische GPU-Erkennung und optimale Device-Auswahl
- **Data Mesh:** Domains wie `market_data`, `trading_signals` mit Governance und Quality Rules
- **Fitness Functions:** Gewichtete Metriken (Sharpe 30%, Calmar 25%, P&L 30%) für Backtesting
- **Moderate Isolation Pattern:** Trading Services konsolidiert aber mit separaten Worker-Prozessen für CPU-intensive Operationen

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
9. **Dynamische Konfiguration:** Lazy-Loading Settings-System implementieren
   - Nur tatsächlich verwendete Settings laden (reduziert Memory-Footprint)
   - Automatische Typkonvertierung für Environment-Variablen
   - Fallback zu sinnvollen Defaults
   - Vereinfacht Erweiterung neuer Konfigurationseinstellungen
   - Migration von Legacy-Konstanten zu dynamischem System planen

## Essential AI Agent Productivity Guide

### Immediate Action Items for New AI Agents

1. **Environment First**: Always run `conda activate tradpal_env` before any terminal commands
2. **Service Location**: All new code goes in `services/`, never in root directory
3. **Blueprint Enforcement**: Copy `services/service_template.py` for new services
4. **Testing Required**: Every feature needs tests in `tests/` before implementation
5. **Async Mandatory**: Use `asyncio` for all I/O operations, never blocking calls

### Critical Commands (Not Obvious from File Inspection)

```bash
# Profile-based execution (controls feature set)
python main.py --profile light --mode live      # Minimal features, fast startup
python main.py --profile heavy --mode backtest  # All AI/ML features enabled

# Isolated backtesting (resource management)
python main.py backtesting-worker               # Dedicated process, no interference

# Service health verification
curl http://localhost:8001/health               # Data service health
curl http://localhost:8002/health               # Core service health

# Performance validation
python scripts/performance_benchmark.py         # Memory/CPU/GPU benchmarks
```

### Architecture Understanding (Key Files to Read)

**Entry Points:**
- `main.py`: Hybrid orchestrator, profile-based execution, service initialization
- `services/trading_service/backtesting_worker.py`: Isolated worker process pattern

**Service Blueprints:**
- `services/core_service/main.py`: FastAPI service with health checks
- `services/core_service/client.py`: Async client with circuit breaker + mTLS
- `services/service_template.py`: Complete service implementation template

**Configuration System:**
- `config/settings.py`: Lazy loading, modular imports, legacy compatibility
- `dependency_catalog.txt`: Version-controlled package versions
- `.env`: Profile-based configuration (light/heavy modes)

**Testing Infrastructure:**
- `tests/conftest.py`: Central fixtures, async test setup, sample data generators
- `tests/unit/test_core_service.py`: Service-specific test patterns

### Project-Specific Patterns (Different from Common Practices)

**Service Communication:**
```python
# NEVER: Direct HTTP calls without circuit breaker
response = requests.get(url)

# ALWAYS: Circuit breaker protected async calls
async with self._get_session() as session:
    async with session.get(url) as response:
        return await response.json()
```

**Configuration Access:**
```python
# NEVER: Direct imports (causes circular dependencies)
from config.core_settings import SYMBOL

# ALWAYS: Lazy loading through main settings
from config.settings import SYMBOL
```

**Data Processing:**
```python
# NEVER: Load entire dataset into memory
data = pd.read_csv('large_file.csv')

# ALWAYS: Memory-mapped or chunked processing
with MemoryMappedData('large_file.h5') as data:
    for chunk in data.chunks(chunk_size=1000):
        process_chunk(chunk)
```

**Service Dependencies:**
```python
# NEVER: Import service directly (tight coupling)
from services.data_service import DataService

# ALWAYS: Use client pattern with authentication
client = DataServiceClient()
await client.authenticate()
result = await client.fetch_data(symbol)
```

### Common Pitfalls to Avoid

1. **Root Directory Pollution**: Never create code files in root - always use `services/`
2. **Synchronous I/O**: Never use `requests` - always `aiohttp` with `asyncio`
3. **Direct Service Imports**: Never import services directly - always use clients
4. **Unprotected HTTP Calls**: Never make HTTP requests without circuit breaker protection
5. **Memory-Inefficient Processing**: Never load large datasets entirely - use memory mapping
6. **Missing Authentication**: Never call services without `await client.authenticate()`
7. **Profile Ignorance**: Never run heavy operations without checking profile settings

### Quality Gates (Always Verify)

**Before Committing:**
- [ ] `pytest tests/` passes all tests
- [ ] `make quality-check` passes (format, lint, type-check)
- [ ] Service follows blueprint (`main.py`, `client.py`, `requirements.txt`)
- [ ] All async I/O operations use proper patterns
- [ ] Circuit breaker protection on all external calls
- [ ] Zero-trust authentication implemented

**Performance Validation:**
- [ ] Memory usage < 85MB baseline for core operations
- [ ] GPU acceleration working if available
- [ ] No memory leaks in long-running processes
- [ ] Backtesting worker isolation maintained

*Last updated: October 25, 2025*