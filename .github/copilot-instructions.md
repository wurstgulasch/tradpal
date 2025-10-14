# Copilot Instructions for AI Agents

## Überblick
Dieses Projekt, **TradPal**, ist ein modulares Trading-Indikator-System, primär optimiert für 1-Minuten-Charts, aber skalierbar auf höhere Timeframes (z. B. 1h, 1d, 1w, 1m). Es kombiniert technische Indikatoren (EMA, RSI, Bollinger Bands, ATR, ADX, Fibonacci-Extensions) mit Multi-Timeframe-Analyse (MTA), historischem Backtesting, Machine-Learning-Integration (PyTorch, Optuna) und genetischen Algorithmen für Signaloptimierung. Ziel ist die Generierung präziser Buy/Sell-Signale mit dynamischem Risikomanagement (Positionsgröße, Leverage, Stop-Loss, Take-Profit). Eine interaktive Web-UI (Streamlit/Plotly) ermöglicht Echtzeit-Überwachung und Backtesting. Das System ist containerisiert (Docker) und unterstützt Integrationen wie Telegram, Discord und Webhooks. Es ist für Kryptowährungen optimiert (via CCXT), mit primärem Fokus auf das Asset `BTC/USDT`, aber erweiterbar auf Forex und Aktien. Das Programm soll möglichst in der Conda-Umgebung `tradpal_env` (Python 3.10+) ausgeführt werden, um Abhängigkeiten wie TA-Lib, PyTorch und Optuna konsistent zu verwalten.
**Aktueller Stand (Oktober 2025):** Version 3.0.0 mit Enterprise-Deployment, erweiterten ML-Funktionen und Monitoring. **NEUER FOKUS:** Entwicklung eines Outperformance-Systems mit ML-Modellen für eigenständige Marktüberlegenheit. Ziel ist ein Bot-System, das Märkte konsistent outperformt. Zukünftig: Modulare Broker-Integration für eigenständige Orderausführung (CCXT-Erweiterung für Live-Trading). Copilot soll bei der Entwicklung experimenteller Features für Outperformance helfen und Best Practices für skalierbare ML-Architekturen vorschlagen.

## Projektstruktur
- `config/settings.py`: Zentrale Konfiguration für Indikator-Parameter, Timeframes, Exchanges, Assets (vorrangig `BTC/USDT`), Risikoparameter und Ausgabeformate. Skalierbare Parameter-Tabellen (z. B. `{'1m': {'ema_short': 9, 'ema_long': 21}, '1d': {'ema_short': 50, 'ema_long': 200}}`).
- `src/`: Kernmodule des Trading-Systems (data_fetcher.py, indicators.py, signal_generator.py, backtester.py, ml_predictor.py, performance.py, audit_logger.py, cache.py, etc.).
- `services/`: Modulare Service-Komponenten (web_ui/, ml_trainer/, core/, etc.).
- `scripts/`: Utility-Scripts und Management-Tools (train_ml_model.py, demo_performance.py, enhanced_backtest.py, test_ml_performance.py, etc.).
- `integrations/`: Module für Telegram, Discord, Email-Notifications und Webhooks (telegram/, discord/, email/, sms/, webhook/).
- `main.py`: Orchestriert Module; Modi: "live", "backtest", "multi-timeframe", "optimize" (genetische Algorithmen), "multi-model", "paper".
- `output/`: JSON-Dateien mit Signalen, Risiko-Parametern, Backtest-Resultaten; Plotly-Charts für Web-UI.
- `tests/`: Unit-Tests (pytest) für Module; Integrationstests für Workflows (629+ Tests).
- `Dockerfile` & `docker-compose.yml`: Container-Setup für Umbrel/Kubernetes; Volumes für Config und Output. Alternative: Ausführung in `tradpal_env` (Conda).
- `.env.light/.env.heavy`: Performance-Profile-Konfigurationen (light für minimal Ressourcen, heavy für alle Features).
- `examples/`: Beispiel-Daten und Demo-Skripte (btc_usdt_backtest.ipynb, ml_training_guide.ipynb, portfolio_management_demo.py, sentiment_analysis_demo.py, shap_integration_demo.py).
- `k8s/`: Kubernetes-Deployment-Manifeste.
- `aws/`: AWS-Deployment-Automatisierung.
- `monitoring/`: Prometheus/Grafana-Monitoring-Stack.
- `docs/`: Zusätzliche Dokumentation (ADVANCED_ML_IMPLEMENTATION.md, IMPLEMENTATION_SUMMARY.md, PERFORMANCE_ENHANCEMENTS.md).
Copilot soll Vorschläge machen, fehlende Verzeichnisse zu ergänzen, z. B. für Community-Beiträge oder automatisierte Builds.

## Wichtige Workflows
- **Daten holen:** CCXT für OHLCV-Daten, vorrangig für `BTC/USDT`; Cache via Redis oder lokal (z. B. HDF5). Multi-Asset/Timeframe-Unterstützung. Erweiterung: Implementiere WebSocket-Support (z. B. via ccxt.pro) für Echtzeit-Daten, um Latenz zu reduzieren. Copilot soll Code-Vorschläge für den Wechsel von REST zu WebSockets machen.
- **Indikatoren berechnen:** EMA, RSI, BB, ATR (TA-Lib); optional ADX (>25 für Trendfilter), Fibonacci (z. B. 161.8% für TP), BB-Bandwidth für Volatilität.
- **Signale generieren:**
- Basis: `Buy = (EMA_kurz > EMA_lang) & (RSI < Oversold_Threshold, z. B. 30) & (close > BB_lower); Sell = (EMA_kurz < EMA_lang) & (RSI > Overbought_Threshold, z. B. 70) & (close < BB_upper)`.
- MTA: Bestätigung durch höheren Timeframe (z. B. 5m für 1m-Trades).
- ML-Enhancer: PyTorch-Modell für Signal-Vorhersage; Optuna für Hyperparameter; Ensemble-Methoden (Random Forest + Gradient Boosting). ML-Enhancer: Ergänze Explainable AI (z. B. SHAP für Feature-Importance in PyTorch-Modellen). Copilot soll Beispiele für SHAP-Integration vorschlagen.
- Genetische Algorithmen: Optimierung von Indikator-Parametern (z. B. EMA-Perioden).
- **Risikomanagement:**
- `Position_Size = (Kapital * Risikoprozent, z. B. 1%) / (ATR * Multiplier, z. B. 1.5)`.
- `SL = close - (ATR * 1–1.5); TP = close + (ATR * 2–3)`.
- Leverage: `min(MAX_LEVERAGE, BASE_LEVERAGE / (ATR / ATR_MEAN))`.
- Erweiterung: ADX für Trade-Dauer; Fibonacci-Levels für TP.
- **Backtesting:** Historische Simulation mit Walk-Forward-Analyse, vorrangig für `BTC/USDT`; Metriken exportieren (CSV/JSON).
- **Web-UI:** Echtzeit-Dashboards, Konfiguration, Backtest-Visualisierung via Streamlit/Plotly.
- **Ausgabe:** JSON mit OHLCV, Indikatoren, Signalen, Risiko-Parametern, Meta-Infos (Timeframe, Backtest-Metriken).
- **Paper-Trading und Portfolio-Management:** Simuliere Trades ohne reales Kapital; tracke Portfolio-Performance über mehrere Assets. Copilot soll Prototypen für Paper-Trading-Modus (z. B. in main.py --mode paper) vorschlagen, inklusive Integration mit CCXT für simulierte Orders.

## Konventionen
- **Modularität:** Lose Kopplung via DataFrames; Dependency Injection für Testbarkeit.
- **Konfiguration:** Zentral in `config/settings.py` oder `.env` für API-Keys, Logins; Standard-Asset: `BTC/USDT`.
- **JSON-Ausgabe:** Erweitert um Warnungen (z. B. "Hohe Volatilität – Leverage reduzieren").
- **Error-Handling:** NaN via `dropna()`/Forward-Fill; robuste API-Fallbacks.
- **Code-Style:** PEP-8, englische Docstrings/Kommentare, Logging via `logging`-Modul.
- **Skalierbarkeit:** Parameter-Dicts in Config; CI/CD via GitHub Actions.
- **Umgebung:** Verwende Conda-Umgebung `tradpal_env` für konsistente Installation (Python 3.10+, TA-Lib, PyTorch, Optuna).
- **Sprache:** Verwende für Dokumentation und Kommentare Englisch; für Issues/PRs Deutsch oder Englisch je nach Zielgruppe. Schreibe Commit Messages auf Englisch.
- **Performance Profiles:** Nutze `.env.light` für ressourcenschonende Konfiguration (kein ML/AI) und `.env.heavy` für alle Features.
- **Sicherheit:** Sensible Daten (API-Keys, Tokens) nie in Git-Repository speichern - nur lokal in untracked .env Dateien. Mache Secrets-Manager (z. B. HashiCorp Vault oder AWS Secrets Manager) standardmäßig, nicht optional. Copilot soll Audits vorschlagen, um sensible Daten aus Git zu entfernen und OAuth/JWT für Web-UI zu implementieren.
- **Community-Konventionen:** Verwende Issue-Templates (.github/ISSUE_TEMPLATE/) für Bugs/Features; fördere PRs mit Guidelines. Copilot soll Vorschläge für Community-Features machen, z. B. ein Wiki oder PyPI-Publikation.

## Datenfluss und Architektur
- **DataFrame-Mutationen:** In-place, mit Kopien für Backtests.
- **Signal-Logik:** Filter (z. B. Volumen > Durchschnitt); MTA via höhere Frames; ML/genetische Algorithmen für Optimierung.
- **JSON-Struktur:** Pro Zeile: OHLCV, Indikatoren, Signale, Risiko-Parameter, Meta-Infos.
- **Web-UI:** Interaktive Dashboards; sichere Authentifizierung ausbauen (z. B. OAuth). Füge Screenshots/GIFs in docs/ hinzu; integriere API-Endpoints (Flask) für externe Tools. Copilot soll Tutorials für UI-Setup vorschlagen.
- **Profile-System:** Automatische Validierung der Profile bei Startup; light/heavy Profile für unterschiedliche Hardware-Anforderungen. Erweitere Validierung mit Tools wie pydantic für .env-Dateien. Copilot soll Optimierungen für Performance vorschlagen, z. B. TensorFlow Lite für ML-Inference auf Edge-Devices.

## Integrationen
- CCXT für Exchanges (Binance, Kraken, etc.), mit Fokus auf `BTC/USDT`; erweiterbar für Forex/Aktien.
- Webhooks/Notifications für Telegram, Discord, Email.
- Container: Docker/Umbrel/Kubernetes; Volumes für Config/Output. Alternative: Conda-Umgebung `tradpal_env`.
- ML: PyTorch/Optuna für Signal-Verbesserung; Ensemble-Methoden; genetische Algorithmen.
- Web-UI: Streamlit/Plotly für Monitoring; API-Endpoint (z. B. Flask) geplant.
- Erweiterung: Integriere Sentiment-Analyse (z. B. via X/Twitter-API für BTC/USDT-News). Copilot soll Code für externe Datenquellen vorschlagen, inklusive Rate-Limiting.

## Beispiele
- **Timeframe ändern:** `settings.py` -> `TIMEFRAME = '1d'`; Parameter skalieren.
- **Asset wechseln:** `settings.py` -> `SYMBOL = 'BTC/USDT'` (Standard).
- **Backtest:** `main.py --mode backtest --start_date 2024-01-01 --symbol BTC/USDT`.
- **MTA:** 1m-Signal mit 5m-EMA-Crossover bestätigen.
- **ML-Training:** `scripts/train_ml_model.py --symbol BTC/USDT --timeframe 1h --start-date 2024-01-01`.
- **Web-UI:** `conda activate tradpal_env; streamlit run services/web_ui/app.py`, dann http://localhost:8501 (Login: admin/admin123).
- **Setup in Conda:** `conda env create -f environment.yml; conda activate tradpal_env; pip install -r requirements.txt`.
- **Profile verwenden:** `python main.py --profile light` für minimal Ressourcen, `python main.py --profile heavy` für alle Features.
- **Multi-Model Backtesting:** `python main.py --mode multi-model --symbol BTC/USDT --timeframe 1d --train-missing --max-workers 4`.
- **Deployment-Beispiel:** `docker build -t tradpal .; docker push wurstgulasch/tradpal:latest` – Copilot soll vollständige Deployment-Skripte vorschlagen.
- **Community-Beispiel:** Erstelle ein Jupyter-Notebook in examples/ für BTC/USDT-Backtests (btc_usdt_backtest.ipynb, ml_training_guide.ipynb).

## Anweisungen für Copilot
- **PRIORITÄTEN (2025):** Fokus auf Outperformance-System Entwicklung - ML-Modelle für konsistente Marktüberlegenheit. Experimentelle Features für bessere Performance (Reinforcement Learning, Market Regime Detection, Advanced Feature Engineering). Vorbereitung für modulare Broker-Integration (CCXT-Erweiterung für Live-Trading). Zweite Priorität: Repository-Aufräumung nach Best Practices und Dokumentations-Reorganisation.
- **Outperformance-Ziele:** Entwickle ML-Modelle, die Buy&Hold und traditionelle Indikatoren konsistent outperformen. Implementiere fortgeschrittene Feature-Engineering, Ensemble-Methoden und Risikomanagement für maximale Sharpe-Ratio und Profit-Faktor.
- **Experimentelle Features:** Schlage neue Ansätze vor wie Reinforcement Learning für Trading, Market Regime Classification, Alternative Data Integration (Social Sentiment, On-Chain-Metrics), und Advanced Risk-Parity-Strategien.
- **Code-Vorschläge:** Fokussiere auf skalierbare ML-Architekturen, Performance-Optimierung und robuste Error-Handling. Implementiere modulare Broker-Adapter für zukünftige Live-Trading-Integration.
- **Tests:** Schreibe Unit-Tests für neue Features; ziele auf >85% Coverage (aktuell 629+ Tests). Erstelle Integrationstests für Outperformance-Workflows.
- **Dokumentation:** Halte docs/ aktuell und reorganisiere README für Übersichtlichkeit. Erstelle separate Dokumentationen für komplexe Features.
- **Sicherheit:** Implementiere sichere Broker-API-Integration mit Key-Management und Rate-Limiting.
- **Lizenz:** MIT-Lizenz ist integriert.
- **Optimierung:** Fokussiere auf ML-Performance (GPU-Training, Memory-Effizienz) und skalierbare Architekturen.
- **Projektstruktur Best Practices:** Halte die modulare Struktur ein - `src/` für Kernlogik, `services/` für Service-Komponenten, `scripts/` für Utilities. Neue Broker-Integrationen nach `integrations/brokers/`.
- **Community und Deployment:** PyPI-Publikation aktiv, automatisierte Releases via GitHub Actions verfügbar.

## Verbesserungsvorschläge
1. **Outperformance-System:** Reinforcement Learning für Trading-Strategien; Market Regime Detection für adaptive Parameter; Alternative Data Integration (On-Chain-Metrics, Social Sentiment); Advanced Feature Engineering für bessere ML-Performance.
2. **Broker-Integration:** Modulare Broker-Adapter-Architektur für CCXT-Erweiterung; Sichere API-Key-Management; Rate-Limiting und Error-Recovery; Paper-Trading-Modus als Übergang.
3. **Experimentelle Features:** Ensemble-Modelle mit Confidence-Weighting; Walk-Forward-Optimierung mit Overfitting-Detection; GPU-Training für PyTorch-Modelle; Real-time Model Updating.
4. **Machine Learning:** SHAP für ML-Explainability integriert; Ensemble-Methoden (Random Forest + Gradient Boosting) verfügbar; LSTM-Modelle für Zeitreihen verfügbar; WebSocket für Echtzeit-ML-Input hinzufügen, in `tradpal_env`.
5. **Sicherheit:** Standard-Login ersetzt; OAuth/JWT hinzugefügt; Disclaimer für Risiken verfügbar; Vault standardmäßig. Entferne sensible Daten aus Git-Repository.
6. **Tests:** Unit-Tests für `ml_trainer.py` verfügbar; Integrationstests für Workflows verfügbar; CI/CD für Releases erweitert (629+ Tests).
7. **Performance:** Datenabruf für `BTC/USDT` optimiert (batchweise API-Calls); ML-Training auf GPU/Cloud auslagerbar; Redis-Caching integriert.
8. **Community:** Issues für bekannte Bugs hinzugefügt; PRs gefördert; PyPI-Publikation verfügbar; Wiki erstellt.
9. **Features:** Sentiment-Analyse integriert (via X/Twitter-Daten für `BTC/USDT`); Paper-Trading-Modus verfügbar; Portfolio-Management für Multi-Assets verfügbar.
10. **Dokumentation:** README mit Screenshots/GIFs verfügbar; Jupyter-Notebook mit Beispiel-Backtests für `BTC/USDT` in `tradpal_env` verfügbar.
11. **Profile-System:** Validierung und Dokumentation der light/heavy Profile verbessert für bessere Benutzerfreundlichkeit.
12. **Deployment:** Releases mit Docker-Images verfügbar; Kubernetes-Deployment automatisiert.
*Letzte Aktualisierung: 14.10.2025*