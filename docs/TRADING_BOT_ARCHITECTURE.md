# Trading Bot Architektur: Vollständige Funktionsweise

## Übersicht

Der Trading Bot ist das Herzstück von TradPal - ein vollautonomes AI-Trading-System, das auf einer Microservices-Architektur basiert. Dieser Dokument beschreibt detailliert, wie der Trading Bot funktioniert, welche Funktionen er nutzt und welche Architektur dahintersteckt.

## Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────────────────┐
│                           TRADPAL TRADING BOT                        │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   Signal        │  │   Risk Service  │  │   Data Service  │     │
│  │   Generator     │  │                 │  │                 │     │
│  │                 │  │                 │  │                 │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│             │                       │                       │       │
│             └───────────────────────┼───────────────────────┘       │
│                                     ▼                                 │
│                        ┌─────────────────┐                          │
│                        │   Trading Bot   │                          │
│                        │     Live        │                          │
│                        │                 │                          │
│                        │  ┌────────────┐ │                          │
│                        │  │ Order Mgmt │ │                          │
│                        │  └────────────┘ │                          │
│                        │  ┌────────────┐ │                          │
│                        │  │ Position   │ │                          │
│                        │  │ Tracking   │ │                          │
│                        │  └────────────┘ │                          │
│                        │  ┌────────────┐ │                          │
│                        │  │ Risk Ctrl  │ │                          │
│                        │  └────────────┘ │                          │
│                        └─────────────────┘                          │
│                                 │                                   │
│             ┌───────────────────┼───────────────────┐               │
│             ▼                   ▼                   ▼               │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐       │
│  │   Broker API    │ │   Event System  │ │   Monitoring    │       │
│  │   Integration   │ │   (Redis)       │ │   (Prometheus)  │       │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘       │
├─────────────────────────────────────────────────────────────────────┤
│                    MONITORING & INFRASTRUCTURE                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 1. Kernkomponenten des Trading Bots

### 1.1 Trading Bot Live Service (`services/trading_bot_live/`)

Der zentrale Service für Live-Trading-Operationen mit folgenden Hauptfunktionen:

#### **Trading Session Management**
```python
class TradingSession:
    """Repräsentiert eine aktive Trading-Session"""
    session_id: str
    symbol: str
    strategy: str
    timeframe: str
    capital: float
    risk_per_trade: float
    max_positions: int
    is_active: bool
    paper_trading: bool
```

**Funktionen:**
- **Session-Erstellung**: Initialisiert Trading-Sessions mit Risiko-Parametern
- **Session-Monitoring**: Kontinuierliche Überwachung aktiver Sessions
- **Session-Beendigung**: Sicheres Schließen mit automatischer Positionsauflösung

#### **Order Management System**
```python
@dataclass
class Order:
    """Trading Order Datenstruktur"""
    order_id: str
    symbol: str
    side: OrderSide  # BUY/SELL
    quantity: float
    order_type: OrderType  # MARKET/LIMIT/STOP
    price: Optional[float]
    status: OrderStatus  # PENDING/FILLED/CANCELLED/REJECTED
```

**Order-Arten:**
- **Market Orders**: Sofortige Ausführung zum besten verfügbaren Preis
- **Limit Orders**: Ausführung nur zu einem bestimmten Preis oder besser
- **Stop Orders**: Ausführung bei Erreichen eines Stop-Preises

#### **Position Tracking**
```python
@dataclass
class Position:
    """Offene Position Datenstruktur"""
    position_id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
```

**Position-Management:**
- **Real-time P&L Berechnung**: Kontinuierliche Gewinn/Verlust-Berechnung
- **Stop-Loss/Take-Profit**: Automatische Positionschließung bei Preislevels
- **Position-Updates**: Live-Preis-Updates und Risiko-Neubewertung

#### **Risk Management Engine**
```python
async def _calculate_position_size(self, symbol: str) -> float:
    """Berechnet Positionsgröße basierend auf Risikomanagement"""
    session = self.trading_sessions[symbol]

    # Risiko-basierte Positionsgröße
    risk_amount = session.capital * session.risk_per_trade
    risk_multiplier = 2.0  # Stop-Loss Distanz in Volatilitätseinheiten

    position_size = risk_amount / (volatility * risk_multiplier)
    quantity = position_size / current_price

    return quantity
```

**Risiko-Kontrollen:**
- **Max Drawdown Limits**: Automatische Stopps bei Kapitalverlust-Grenzen
- **Positionsgrößen-Limits**: Begrenzung pro Trade und gesamt
- **Max Trades pro Tag**: Begrenzung der Handelsfrequenz
- **Volatilitäts-Adjustierung**: Dynamische Positionsgrößen basierend auf Marktvolatilität

### 1.2 Event-Driven Kommunikation

Der Trading Bot nutzt ein vollständig event-gesteuertes System:

#### **Event Types**
```python
class TradingEventType(Enum):
    # Trading Events
    TRADING_SESSION_STARTED = "trading.session_started"
    TRADING_SESSION_STOPPED = "trading.session_stopped"
    ORDER_EXECUTED = "trading.order_executed"
    POSITION_OPENED = "trading.position_opened"
    POSITION_CLOSED = "trading.position_closed"
    SIGNAL_GENERATED = "trading.signal_generated"

    # Risk Events
    RISK_TRIGGERED = "trading.risk_triggered"
    EMERGENCY_STOP = "trading.emergency_stop"

    # Performance Events
    PERFORMANCE_UPDATED = "trading.performance_updated"
```

#### **Event Flow Architecture**
```
Signal Generator → Trading Bot → Event System → Andere Services
       ↓               ↓              ↓              ↓
   Signal Event → Order Execution → Trade Event → Risk Assessment
       ↓               ↓              ↓              ↓
   Position Update → P&L Calculation → Performance Update
```

## 2. Integration mit anderen Services

### 2.1 Signal Generator Service (`services/core/`)

**Verantwortlichkeiten:**
- **Technische Analyse**: RSI, MACD, Bollinger Bands, Moving Averages
- **Preisaktions-Analyse**: Support/Resistance, Trendlinien, Chart-Patterns
- **Volatilitäts-Analyse**: ATR, Realized Volatility, Implied Volatility

**Signal-Generierung:**
```python
async def generate_trading_signals(self, market_data: pd.DataFrame) -> List[TradingSignal]:
    """Generiert Trading-Signale aus Markt-Daten"""

    signals = []

    # Technische Indikatoren berechnen
    indicators = self.calculate_technical_indicators(market_data)

    # Signal-Logik anwenden
    for i in range(len(market_data)):
        signal = self.apply_signal_logic(indicators.iloc[i])
        if signal:
            signals.append(signal)

    return signals
```

### 2.2 Risk Service (`services/risk_service/`)

**Risiko-Bewertung:**
- **Value at Risk (VaR)**: Statistisches Risikomaß
- **Expected Shortfall (ES)**: Erwarteter Verlust in Extremfällen
- **Stress Testing**: Historische Szenario-Analyse
- **Portfolio-Optimierung**: Modern Portfolio Theory

**Risiko-Monitoring:**
```python
async def monitor_portfolio_risk(self, portfolio: Portfolio) -> RiskAssessment:
    """Kontinuierliches Risiko-Monitoring"""

    # VaR Berechnung
    var_95 = self.calculate_var(portfolio, confidence=0.95)

    # Stress Tests
    stress_results = await self.run_stress_tests(portfolio)

    # Risk Limits prüfen
    breaches = self.check_risk_limits(portfolio, var_95)

    return RiskAssessment(var_95, stress_results, breaches)
```

### 2.3 Data Service (`services/data_service/`)

**Daten-Management:**
- **Markt-Daten**: OHLCV-Daten von Exchanges (Binance, Kraken, etc.)
- **Alternative Daten**: News Sentiment, Social Media, On-Chain Metriken
- **Historische Daten**: Langfristige Daten für Backtesting
- **Real-time Updates**: Live-Preis- und Volumen-Daten

**Daten-Pipeline:**
```python
class DataPipeline:
    """End-to-End Datenverarbeitung"""

    async def process_market_data(self, raw_data: Dict) -> ProcessedData:
        """Verarbeitet Rohdaten zu analysierbaren Formaten"""

        # Datenvalidierung
        validated_data = self.validate_data(raw_data)

        # Feature Engineering
        features = self.engineer_features(validated_data)

        # Normalisierung
        normalized_data = self.normalize_features(features)

        return ProcessedData(normalized_data)
```

## 3. Trading-Workflow

### 3.1 Signal-generierter Trade

```
1. Markt-Daten Update → Data Service
2. Technische Analyse → Signal Generator
3. Signal-Generierung → Trading Bot
4. Risiko-Prüfung → Risk Service
5. Order-Ausführung → Broker API
6. Position-Tracking → Trading Bot
7. Performance-Update → Monitoring
```

### 3.2 Manueller Trade

```
1. Manuelle Order → Trading Bot API
2. Validierung → Trading Bot
3. Risiko-Prüfung → Risk Service
4. Order-Ausführung → Broker API
5. Position-Erstellung → Trading Bot
6. Event-Publishing → Event System
```

### 3.3 Risiko-getriggerter Stop

```
1. Preis-Update → Trading Bot
2. P&L-Neuberechnung → Trading Bot
3. Risiko-Limits prüfen → Risk Service
4. Stop-Loss Trigger → Trading Bot
5. Position-Schließung → Broker API
6. Verlust-Realisierung → Trading Bot
```

## 4. Performance-Monitoring

### 4.1 Key Performance Indicators (KPIs)

```python
@dataclass
class PerformanceMetrics:
    """Trading Performance Metriken"""
    total_return: float          # Gesamtrendite
    sharpe_ratio: float          # Risk-adjusted Return
    max_drawdown: float          # Maximum Drawdown
    win_rate: float             # Gewinnrate
    profit_factor: float        # Profit/Loss Ratio
    total_trades: int           # Gesamtanzahl Trades
    avg_trade_duration: float   # Durchschnittliche Trade-Dauer
    largest_win: float          # Größter Gewinn
    largest_loss: float         # Größter Verlust
```

### 4.2 Real-time Monitoring

**Dashboard-Metriken:**
- **Portfolio Value**: Aktueller Portfolio-Wert
- **Open Positions**: Anzahl offener Positionen
- **Unrealized P&L**: Unrealisierte Gewinne/Verluste
- **Daily P&L**: Tägliche Performance
- **Risk Metrics**: VaR, Drawdown, Exposure
- **System Health**: Service-Status, Latenz, Fehler-Raten

### 4.3 Performance Analytics

```python
async def calculate_advanced_metrics(self, trades: List[Trade]) -> AdvancedMetrics:
    """Erweiterte Performance-Analyse"""

    # Monte Carlo Simulation für Szenario-Analyse
    monte_carlo_results = self.run_monte_carlo_simulation(trades)

    # Benchmark-Vergleich (Buy & Hold, S&P 500, etc.)
    benchmark_comparison = self.compare_to_benchmarks(trades)

    # Risk-adjusted Returns
    sortino_ratio = self.calculate_sortino_ratio(trades)
    calmar_ratio = self.calculate_calmar_ratio(trades)

    return AdvancedMetrics(
        monte_carlo_results,
        benchmark_comparison,
        sortino_ratio,
        calmar_ratio
    )
```

## 5. Sicherheit und Risiko-Management

### 5.1 Paper Trading Mode

**Sicheres Testen:**
- **Simulierte Orders**: Keine echten Trades
- **Virtuelles Kapital**: Fiktives Startkapital
- **Real-time Preise**: Live-Markt-Daten ohne Risiko
- **Vollständige Funktionalität**: Alle Features verfügbar

### 5.2 Live Trading Safeguards

**Sicherheitsmechanismen:**
- **Bestätigungsanforderung**: Manuelle Genehmigung für Live-Trades
- **Risiko-Limits**: Automatische Stops bei Verlustgrenzen
- **Positionsgrößen-Begrenzung**: Max Exposure pro Trade
- **Emergency Stop**: Sofortige Beendigung aller Aktivitäten

### 5.3 Zero-Trust Security

**Service-to-Service Authentifizierung:**
- **mTLS**: Mutual TLS für sichere Kommunikation
- **JWT Tokens**: Token-basierte API-Autorisierung
- **API Gateway**: Zentralisierte Authentifizierung
- **Audit Logging**: Vollständige Aktivitätsprotokollierung

## 6. Integration mit Brokern

### 6.1 Broker API Abstraktion

```python
class BrokerAdapter(ABC):
    """Abstrakte Broker-API Schnittstelle"""

    @abstractmethod
    async def get_balance(self) -> float:
        """Kontostand abrufen"""

    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult:
        """Order platzieren"""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Order stornieren"""

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Offene Positionen abrufen"""

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Order-Status abrufen"""
```

### 6.2 Unterstützte Broker

- **Binance**: Spot und Futures Trading
- **Kraken**: Krypto-Trading mit niedrigen Gebühren
- **CCXT**: Unified API für 100+ Exchanges
- **Paper Trading**: Simulierter Broker für Tests

## 7. Event-System Architektur

### 7.1 Redis Streams Integration

```python
class EventSystem:
    """Event-gesteuerte Kommunikation"""

    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Event publizieren"""
        await self.redis.xadd(event_type, data)

    async def subscribe(self, event_type: str, handler: Callable):
        """Event abonnieren"""
        # Consumer Group erstellen
        await self.redis.xgroup_create(event_type, self.consumer_group)

        # Events verarbeiten
        while True:
            messages = await self.redis.xreadgroup(
                self.consumer_group,
                self.consumer_name,
                {event_type: '>'}
            )

            for message in messages:
                await handler(message)
```

### 7.2 Event Processing Pipeline

```
Raw Event → Validation → Enrichment → Routing → Processing → Storage
     ↓           ↓           ↓          ↓          ↓          ↓
  Event     Validierung  Kontext-   Service-   Business-  Audit
  Input     Layer       hinzufügen  Routing    Logik     Trail
```

## 8. Monitoring und Observability

### 8.1 Prometheus Metriken

```python
# Trading Bot Metriken
TRADING_SESSIONS_ACTIVE = Gauge('trading_sessions_active', 'Active trading sessions')
ORDERS_EXECUTED_TOTAL = Counter('orders_executed_total', 'Total orders executed')
POSITIONS_OPEN = Gauge('positions_open', 'Currently open positions')
PNL_REALIZED = Counter('pnl_realized', 'Realized P&L')
RISK_VIOLATIONS = Counter('risk_violations', 'Risk limit violations')

# Performance Metriken
SHARPE_RATIO = Gauge('sharpe_ratio', 'Current Sharpe ratio')
MAX_DRAWDOWN = Gauge('max_drawdown', 'Current max drawdown')
WIN_RATE = Gauge('win_rate', 'Current win rate')
```

### 8.2 Grafana Dashboards

**Trading Dashboard:**
- Portfolio Performance Charts
- Risk Metrics Visualisierung
- Trade Execution Timeline
- System Health Indicators

**Operations Dashboard:**
- Service Response Times
- Error Rates und Alerts
- Resource Utilization
- Event Processing Throughput

## 9. Deployment und Skalierung

### 9.1 Docker Containerisierung

```dockerfile
FROM python:3.10-slim

# Trading Bot Service
COPY services/trading_bot_live/ /app/
COPY config/ /app/config/

RUN pip install -r requirements.txt

EXPOSE 8005
CMD ["python", "service.py"]
```

### 9.2 Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-bot-live
spec:
  replicas: 3  # Horizontale Skalierung
  selector:
    matchLabels:
      app: trading-bot-live
  template:
    metadata:
      labels:
        app: trading-bot-live
    spec:
      containers:
      - name: trading-bot
        image: tradpal/trading-bot-live:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### 9.3 Load Balancing

- **API Gateway**: Request Distribution
- **Redis Cluster**: Event System Skalierung
- **Database Sharding**: Performance Data Partitionierung
- **Service Mesh**: Advanced Traffic Management

## 10. Zukunftsentwicklung

### 10.1 Advanced Features Integration

**Reinforcement Learning:**
- RL-basierte Strategie-Optimierung
- Multi-Agent Trading Systeme
- Transfer Learning von Paper zu Live

**Market Regime Detection:**
- Automatische Strategie-Anpassung
- Regime-spezifische Risk-Parameter
- Adaptive Signal-Gewichtung

**Alternative Data:**
- Sentiment-Analyse Integration
- On-Chain Metriken
- Wirtschaftsdaten-Korrelation

### 10.2 Performance Optimierungen

- **GPU Acceleration**: Deep Learning Modelle
- **Memory Mapped Data**: Große Datasets effizient verarbeiten
- **Async Processing**: Nicht-blockierende Operationen
- **Caching Layer**: Redis-basierte Performance-Steigerung

## Zusammenfassung

Der Trading Bot von TradPal ist ein hochkomplexes, event-gesteuertes System, das traditionelle Trading-Funktionen mit modernster AI und Microservices-Architektur kombiniert. Die Kernstärken liegen in:

- **Vollständige Automatisierung**: Vom Signal zur Order-Ausführung
- **Umfassendes Risiko-Management**: Mehrstufige Sicherheitsmechanismen
- **Event-Driven Architecture**: Lose Kopplung und hohe Skalierbarkeit
- **Real-time Processing**: Live-Daten und sofortige Reaktionen
- **Enterprise-Grade Monitoring**: Vollständige Observability

Das System ist designed für professionellen Einsatz mit Fokus auf Sicherheit, Zuverlässigkeit und Performance-Optimierung.</content>
<parameter name="filePath">/Users/danielsadowski/VSCodeProjects/tradpal_indicator/docs/TRADING_BOT_ARCHITECTURE.md