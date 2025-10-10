# Performance Profiles

Das TradPal Indicator System unterstützt verschiedene Performance-Profile, die über den `--profile` Parameter ausgewählt werden können. Jedes Profil ist für unterschiedliche Hardware-Konfigurationen und Anforderungen optimiert.

## Verfügbare Profile

### 🚀 `light` - Ressourcenschonend (Ohne KI/ML)
**Für schwache Hardware (z.B. MacBook Air, Raspberry Pi)**
- ✅ Basis-Indikatoren (EMA, RSI, Bollinger Bands, ATR)
- ✅ Risikomanagement
- ✅ Rate Limiting
- ❌ Machine Learning Features
- ❌ Adaptive Optimization
- ❌ Multi-Timeframe Analysis
- ❌ WebSocket Streaming
- ❌ Monitoring Stack

**Verwendung:**
```bash
python main.py --profile light --mode live
```

### 🔥 `heavy` - Volle Funktionalität (Mit KI/ML)
**Für leistungsstarke Hardware mit allen Features**
- ✅ Alle Indikatoren und Features
- ✅ PyTorch/TensorFlow ML-Modelle
- ✅ Adaptive Optimization
- ✅ WebSocket Streaming
- ✅ Monitoring Stack (Prometheus, Grafana)
- ✅ Parallel Processing
- ✅ Ensemble-Methoden
- ✅ Multi-Timeframe Analysis

**Verwendung:**
```bash
python main.py --profile heavy --mode live
```

## Profile anpassen

Die Profile sind in separaten `.env` Dateien gespeichert:
- `.env.light` - Light Profil (ohne KI/ML)
- `.env.heavy` - Heavy Profil (mit allen Features)

Sie können diese Dateien nach Ihren Bedürfnissen anpassen.

## Empfehlungen

- **MacBook Air / schwache Hardware**: `light`
- **Desktop-PC / Server mit GPU**: `heavy`

## Beispiel-Aufrufe

```bash
# Ressourcenschonender Live-Modus (ohne KI)
python main.py --profile light --mode live

# Volle Funktionalität mit KI/ML
python main.py --profile heavy --mode live

# Backtest mit allen Features
python main.py --profile heavy --mode backtest --start-date 2024-01-01
```

## Technische Details

### Light Profil (.env.light)
```bash
# KI/ML Features deaktiviert
ML_ENABLED=false
ADAPTIVE_OPTIMIZATION_ENABLED=false
MTA_ENABLED=false
MONITORING_STACK_ENABLED=false
PERFORMANCE_MONITORING_ENABLED=false
WEBSOCKET_ENABLED=false
PARALLEL_PROCESSING_ENABLED=false
```

### Heavy Profil (.env.heavy)
```bash
# Alle Features aktiviert
ML_ENABLED=true
ADAPTIVE_OPTIMIZATION_ENABLED=true
MTA_ENABLED=true
MONITORING_STACK_ENABLED=true
PERFORMANCE_MONITORING_ENABLED=true
WEBSOCKET_ENABLED=true
PARALLEL_PROCESSING_ENABLED=true
```

### Validierung
Das System validiert automatisch die Profil-Konfiguration beim Start und warnt bei Inkonsistenzen.