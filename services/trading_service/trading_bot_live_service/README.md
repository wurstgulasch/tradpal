# Trading Bot Live Service

Der Trading Bot Live Service ist ein Microservice für Live-Trading-Operationen mit event-gesteuerter Architektur.

## Übersicht

Dieser Service übernimmt die Kernfunktionalität für Live-Trading:

- **Echtzeit-Datenverarbeitung**: Kontinuierliche Verarbeitung von Markt-Daten
- **Signal-Generierung**: Erzeugung und Validierung von Trading-Signalen
- **Risiko-Management**: Positions-Sizing und Risiko-Kontrolle
- **Order-Ausführung**: Simulierte oder echte Order-Ausführung
- **Performance-Monitoring**: Echtzeit-Überwachung der Performance
- **Event-basierte Kommunikation**: Integration mit anderen Services

## Architektur

Der Service basiert auf der Event-Driven Architecture:

- **Event Bus**: Redis-basierte Pub/Sub-Kommunikation
- **Event Store**: Audit-Trail und Event-Replay
- **Event Handler**: Spezialisierte Handler für verschiedene Event-Typen
- **Async Processing**: Nicht-blockierende, skalierbare Verarbeitung

## Verwendung

### Grundlegende Verwendung

```bash
# Starte den Service im Paper-Trading-Modus (empfohlen für Tests)
python -m services.trading_bot_live.service --symbol BTC/USDT --timeframe 1h --mode paper

# Starte im Live-Trading-Modus (VORSICHT: echte Trades!)
python -m services.trading_bot_live.service --symbol BTC/USDT --timeframe 1h --mode live
```

### Kommandozeilen-Optionen

- `--symbol`: Trading-Symbol (Standard: BTC/USDT)
- `--timeframe`: Timeframe (Standard: 1m)
- `--mode`: Trading-Modus (live oder paper, Standard: paper)

### Programmatische Verwendung

```python
from services.trading_bot_live import TradingBotLiveService

async def run_trading_bot():
    service = TradingBotLiveService('BTC/USDT', '1h')
    await service.initialize()

    try:
        await service.start_trading()
    finally:
        await service.shutdown()

# In asyncio-Kontext ausführen
asyncio.run(run_trading_bot())
```

## Konfiguration

Der Service verwendet die zentrale Konfiguration aus `config/settings.py`:

- `INITIAL_CAPITAL`: Startkapital
- `RISK_PER_TRADE`: Risiko pro Trade (in %)
- `LIVE_TRADING_MAX_DRAWDOWN`: Maximale Drawdown (in %)
- `LIVE_TRADING_MAX_TRADES_PER_DAY`: Maximale Trades pro Tag
- `LIVE_TRADING_MIN_SIGNAL_CONFIDENCE`: Minimale Signal-Konfidenz
- `LIVE_TRADING_AUTO_EXECUTE`: Automatische Order-Ausführung
- `LIVE_TRADING_CONFIRMATION_REQUIRED`: Bestätigung erforderlich

## Event-System

Der Service publiziert verschiedene Events:

- `MARKET_DATA_RECEIVED`: Neue Markt-Daten empfangen
- `SIGNAL_GENERATED`: Trading-Signal generiert
- `TRADE_EXECUTED`: Trade ausgeführt
- `RISK_ASSESSMENT_COMPLETED`: Risiko-Bewertung abgeschlossen
- `POSITION_OPENED/CLOSED`: Position eröffnet/geschlossen

## Sicherheit

- **Paper-Trading-Modus**: Simuliertes Trading ohne reales Geld
- **Risiko-Limits**: Automatische Stops bei Drawdown-Limits
- **Bestätigungsanforderung**: Optionale manuelle Bestätigung für Live-Trades
- **Audit-Trail**: Vollständige Protokollierung aller Aktionen

## Monitoring

Der Service integriert sich mit dem Monitoring-Stack:

- **Prometheus-Metriken**: Performance-Metriken
- **Grafana-Dashboards**: Visuelle Überwachung
- **Health-Checks**: Service-Zustand-Überprüfung
- **Log-Aggregation**: Zentralisierte Logs

## Abhängigkeiten

- Redis (für Event-Streaming)
- Pandas (für Datenverarbeitung)
- TA-Lib (für technische Indikatoren)
- CCXT (für Broker-Integration, optional)

## Deployment

Der Service kann als eigenständiger Microservice deployed werden:

```yaml
# Kubernetes Deployment Beispiel
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-bot-live
spec:
  replicas: 1
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
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: TRADING_MODE
          value: "paper"
```

## Entwicklung

### Tests ausführen

```bash
# Unit-Tests
pytest tests/test_trading_bot_live.py

# Integration-Tests
pytest tests/test_trading_bot_integration.py
```

### Event-System testen

```bash
# Event-System initialisieren
python -c "import asyncio; from src.events import initialize_event_system; asyncio.run(initialize_event_system())"

# Service mit Event-Logging starten
python -m services.trading_bot_live.service --mode paper
```

## Troubleshooting

### Häufige Probleme

1. **Redis-Verbindung fehlgeschlagen**
   - Stelle sicher, dass Redis läuft: `redis-server`
   - Überprüfe REDIS_URL in der Konfiguration

2. **Markt-Daten nicht verfügbar**
   - Überprüfe Internet-Verbindung
   - Überprüfe API-Limits der Exchange
   - Verwende alternative Data-Quellen

3. **Signal-Generierung fehlgeschlagen**
   - Überprüfe TA-Lib Installation
   - Überprüfe Datenqualität
   - Teste Indikator-Berechnungen einzeln

### Logs

Logs werden in `logs/tradpal.log` geschrieben. Für detaillierte Event-Logs:

```bash
tail -f logs/tradpal.log | grep -i "trading_bot\|event"
```