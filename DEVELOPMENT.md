# TradPal Development Quick Start

## 🚀 Schnellstart für Entwickler

### 1. Umgebung aufsetzen
```bash
# Einmalige Einrichtung
make setup

# Oder manuell:
./setup_dev.sh
```

### 2. Services starten
```bash
# Basis-Services starten
make dev-up

# Mit Web-UI
make dev-ui
```

### 3. Entwicklung beginnen
```bash
# Tests ausführen
make test

# Backtest durchführen
make backtest

# Code-Qualität prüfen
make quality-check
```

## 📋 Verfügbare Services

| Service | URL | Beschreibung |
|---------|-----|--------------|
| Data Service | http://localhost:8001 | Daten-Management & APIs |
| Backtesting Service | http://localhost:8002 | Trading-Simulation |
| Web UI | http://localhost:8501 | Dashboard & Monitoring |

## 🛠️ Nützliche Befehle

```bash
# Services verwalten
make dev-up          # Services starten
make dev-down        # Services stoppen
make dev-logs        # Logs anzeigen
make health-check    # Service-Status prüfen

# Entwicklung
make test            # Tests ausführen
make backtest        # Backtest durchführen
make quality-check   # Code-Qualität prüfen

# Wartung
make clean           # Cache & temporäre Dateien löschen
make shell-data      # Shell im Data-Service öffnen
```

## 🔧 Fehlerbehebung

### Services starten nicht?
```bash
# Docker-Container prüfen
docker ps

# Logs anzeigen
make dev-logs

# Services neu starten
make dev-down
make dev-up
```

### Tests schlagen fehl?
```bash
# Abhängigkeiten installieren
pip install -e .

# Cache leeren
make clean

# Tests erneut ausführen
make test
```

## 📚 Weitere Dokumentation

- [Architektur-Übersicht](docs/ARCHITECTURE.md)
- [API-Dokumentation](docs/API.md)
- [Entwicklungsrichtlinien](docs/CONTRIBUTING.md)
- [Performance-Bericht](docs/PERFORMANCE_OUTPERFORMANCE_REPORT.md)

## 🎯 Nächste Schritte

1. **Infrastruktur stabilisieren** ✅ (In Arbeit)
2. Datenquellen erweitern
3. ML-Modelle verbessern
4. Live-Trading integrieren

Happy coding! 🎉