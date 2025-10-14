# Setup und Installation

Diese Anleitung führt Sie durch die Installation und Konfiguration von TradPal Indicator.

## 🔧 Systemvoraussetzungen

### Minimale Anforderungen
- **Python**: 3.10+
- **RAM**: 4GB
- **Speicher**: 2GB frei
- **OS**: Linux, macOS, Windows

### Empfohlene Anforderungen
- **Python**: 3.10+
- **RAM**: 8GB+
- **Speicher**: 10GB+ SSD
- **OS**: Linux (Ubuntu 20.04+)

## 📦 Installation

### Option 1: Conda (Empfohlen)

```bash
# Repository klonen
git clone https://github.com/your-org/tradpal-indicator.git
cd tradpal-indicator

# Conda-Umgebung erstellen
conda env create -f environment.yml
conda activate tradpal_env

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### Option 2: pip

```bash
# Repository klonen
git clone https://github.com/your-org/tradpal-indicator.git
cd tradpal-indicator

# Virtuelle Umgebung erstellen
python -m venv tradpal_env
source tradpal_env/bin/activate  # Linux/macOS
# oder tradpal_env\Scripts\activate  # Windows

# Abhängigkeiten installieren
pip install -e .[dev,ml,webui]
```

### Option 3: Docker

```bash
# Repository klonen
git clone https://github.com/your-org/tradpal-indicator.git
cd tradpal-indicator

# Docker-Image bauen
docker build -t tradpal-indicator .

# Container starten
docker run -p 8501:8501 tradpal-indicator
```

## ⚙️ Konfiguration

### Grundkonfiguration

1. **API-Keys konfigurieren**
```bash
# .env Datei erstellen
cp .env.example .env

# API-Keys hinzufügen
echo "BINANCE_API_KEY=your_api_key" >> .env
echo "BINANCE_API_SECRET=your_api_secret" >> .env
```

2. **Settings anpassen**
```python
# config/settings.py bearbeiten
SYMBOL = 'BTC/USDT'  # Trading-Paar
TIMEFRAME = '1h'     # Zeitrahmen
CAPITAL = 10000      # Startkapital
```

### Performance-Profile

TradPal unterstützt verschiedene Performance-Profile:

#### Light Profile (Ressourcen-schonend)
```bash
python main.py --profile light
```
- Kein ML/AI
- Minimale Indikatoren
- Schnellere Ausführung

#### Heavy Profile (Alle Features)
```bash
python main.py --profile heavy
```
- Alle ML-Modelle
- Maximale Indikatoren
- Umfassende Analyse

## 🚀 Erste Ausführung

### Backtest ausführen
```bash
# Einfacher Backtest
python main.py --mode backtest --symbol BTC/USDT --timeframe 1d

# Mit Web-UI
python main.py --mode webui
```

### Discovery Mode (Genetic Algorithms)
```bash
# Parameter optimieren
python main.py --mode discovery --symbol BTC/USDT --generations 50
```

### Live Trading
```bash
# Paper Trading (empfohlen für Tests)
python main.py --mode paper --symbol BTC/USDT

# Live Trading (Vorsicht!)
python main.py --mode live --symbol BTC/USDT
```

## 🔍 Verifikation der Installation

### Tests ausführen
```bash
# Alle Tests
pytest tests/ -v

# Spezifische Tests
pytest tests/test_indicators.py -v
pytest tests/test_backtester.py -v
```

### Beispiel-Skript ausführen
```python
# examples/demo_performance_features.py ausführen
python examples/demo_performance_features.py
```

## 🐛 Fehlerbehebung

### Häufige Probleme

#### 1. Import-Fehler
```
ModuleNotFoundError: No module named 'talib'
```
**Lösung:**
```bash
# TA-Lib installieren
conda install -c conda-forge ta-lib
# oder
pip install TA-Lib
```

#### 2. Memory-Fehler
```
MemoryError: Unable to allocate array
```
**Lösung:**
- Mehr RAM verwenden
- Light Profile aktivieren
- Datenbereich reduzieren

#### 3. API-Fehler
```
APIError: Invalid API key
```
**Lösung:**
- API-Keys in `.env` überprüfen
- IP-Whitelist bei Exchange prüfen
- Rate-Limits beachten

#### 4. Docker-Probleme
```
docker: Error response from daemon: pull access denied
```
**Lösung:**
- Docker-Image lokal bauen
- Registry-Zugangsdaten prüfen

### Logs überprüfen
```bash
# Logs anzeigen
tail -f logs/tradpal.log

# Debug-Modus aktivieren
export PYTHONPATH=/app
python main.py --debug
```

## 🔄 Updates

### Automatische Updates
```bash
# Repository aktualisieren
git pull origin main

# Abhängigkeiten aktualisieren
pip install -r requirements.txt --upgrade
```

### Manuelle Updates
```bash
# Neue Version herunterladen
wget https://github.com/your-org/tradpal-indicator/releases/latest/download/tradpal-indicator.tar.gz
tar -xzf tradpal-indicator.tar.gz
cd tradpal-indicator

# Installation wiederholen
pip install -e .
```

## 🌐 Netzwerk-Konfiguration

### Proxy-Einstellungen
```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

### Firewall
Stellen Sie sicher, dass folgende Ports offen sind:
- **8501**: Streamlit Web-UI
- **8000**: Prometheus Metrics (optional)

## 📊 Monitoring

### System-Monitoring
```bash
# Ressourcen-Nutzung überwachen
top -p $(pgrep -f tradpal)

# Speicher-Nutzung
free -h
```

### Anwendungs-Monitoring
```bash
# Prometheus-Metrics
curl http://localhost:8000/metrics

# Health-Check
curl http://localhost:8000/health
```

## 🆘 Hilfe

Bei Problemen:
1. **Dokumentation** lesen
2. **GitHub Issues** durchsuchen
3. **Neues Issue** erstellen
4. **Community** kontaktieren

### Support-Informationen sammeln
```bash
# System-Info
python -c "import sys; print(f'Python: {sys.version}')"

# Abhängigkeiten
pip list | grep -E "(pandas|numpy|talib|pytorch)"

# Logs
tail -50 logs/tradpal.log
```