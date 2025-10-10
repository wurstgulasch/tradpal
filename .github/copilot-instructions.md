# Copilot Instructions for AI Agents

## Überblick
Dieses Projekt, **TradPal Indicator**, ist ein modulares Trading-Indikator-System, primär optimiert für 1-Minuten-Charts, aber skalierbar auf höhere Timeframes (z. B. 1h, 1d, 1w, 1m). Es kombiniert technische Indikatoren (EMA, RSI, Bollinger Bands, ATR, ADX, Fibonacci-Extensions) mit Multi-Timeframe-Analyse (MTA), historischem Backtesting, Machine-Learning-Integration (PyTorch, Optuna) und genetischen Algorithmen für Signaloptimierung. Ziel ist die Generierung präziser Buy/Sell-Signale mit dynamischem Risikomanagement (Positionsgröße, Leverage, Stop-Loss, Take-Profit). Eine interaktive Web-UI (Streamlit/Plotly) ermöglicht Echtzeit-Überwachung und Backtesting. Das System ist containerisiert (Docker) und unterstützt Integrationen wie Telegram, Discord und Webhooks. Es ist für Kryptowährungen optimiert (via CCXT), mit primärem Fokus auf das Asset `BTC/USDT`, aber erweiterbar auf Forex und Aktien. Das Programm soll möglichst in der Conda-Umgebung `tradpal_env` (Python 3.10+) ausgeführt werden, um Abhängigkeiten wie TA-Lib, PyTorch und Optuna konsistent zu verwalten.

**Aktueller Stand (Oktober 2025):** Version 2.5.0 fügt adaptive Optimierung, Ensemble-Methoden (Random Forest + Gradient Boosting) und Walk-Forward-Analyse hinzu. Copilot soll bei der Weiterentwicklung helfen, z. B. durch Verbesserung der ML-Modelle, Sicherheitsmaßnahmen und Community-Funktionen, mit Fokus auf `BTC/USDT`.

## Projektstruktur
- `config/settings.py`: Zentrale Konfiguration für Indikator-Parameter, Timeframes, Exchanges, Assets (vorrangig `BTC/USDT`), Risikoparameter und Ausgabeformate. Skalierbare Parameter-Tabellen (z. B. `{'1m': {'ema_short': 9, 'ema_long': 21}, '1d': {'ema_short': 50, 'ema_long': 200}}`).
- `src/`: Kernmodule des Trading-Systems (data_fetcher.py, indicators.py, signal_generator.py, backtester.py, ml_predictor.py, etc.).
- `services/`: Modulare Service-Komponenten (web_ui.py, ml_trainer.py, optimizer/, core/).
- `scripts/`: Utility-Scripts und Management-Tools (train_ml_model.py, demo_performance.py, manage_integrations.py).
- `integrations/`: Module für Telegram, Discord, Email-Notifications und Webhooks.
- `main.py`: Orchestriert Module; Modi: "live", "backtest", "multi-timeframe", "optimize" (genetische Algorithmen).
- `output/`: JSON-Dateien mit Signalen, Risiko-Parametern, Backtest-Resultaten; Plotly-Charts für Web-UI.
- `tests/`: Unit-Tests (pytest) für Module; Integrationstests für Workflows.
- `Dockerfile` & `docker-compose.yml`: Container-Setup für Umbrel/Kubernetes; Volumes für Config und Output. Alternative: Ausführung in `tradpal_env` (Conda).
- `.env.light/.env.heavy`: Performance-Profile-Konfigurationen (light für minimal Ressourcen, heavy für alle Features).

## Wichtige Workflows
- **Daten holen:** CCXT für OHLCV-Daten, vorrangig für `BTC/USDT`; Cache via Redis oder lokal (z. B. HDF5). Multi-Asset/Timeframe-Unterstützung.
- **Indikatoren berechnen:** EMA, RSI, BB, ATR (TA-Lib); optional ADX (>25 für Trendfilter), Fibonacci (z. B. 161.8% für TP), BB-Bandwidth für Volatilität.
- **Signale generieren:**
  - Basis: `Buy = (EMA_kurz > EMA_lang) & (RSI < Oversold_Threshold, z. B. 30) & (close > BB_lower); Sell = (EMA_kurz < EMA_lang) & (RSI > Overbought_Threshold, z. B. 70) & (close < BB_upper)`.
  - MTA: Bestätigung durch höheren Timeframe (z. B. 5m für 1m-Trades).
  - ML-Enhancer: PyTorch-Modell für Signal-Vorhersage; Optuna für Hyperparameter; Ensemble-Methoden (Random Forest + Gradient Boosting).
  - Genetische Algorithmen: Optimierung von Indikator-Parametern (z. B. EMA-Perioden).
- **Risikomanagement:**
  - `Position_Size = (Kapital * Risikoprozent, z. B. 1%) / (ATR * Multiplier, z. B. 1.5)`.
  - `SL = close - (ATR * 1–1.5); TP = close + (ATR * 2–3)`.
  - Leverage: `min(MAX_LEVERAGE, BASE_LEVERAGE / (ATR / ATR_MEAN))`.
  - Erweiterung: ADX für Trade-Dauer; Fibonacci-Levels für TP.
- **Backtesting:** Historische Simulation mit Walk-Forward-Analyse, vorrangig für `BTC/USDT`; Metriken exportieren (CSV/JSON).
- **Web-UI:** Echtzeit-Dashboards, Konfiguration, Backtest-Visualisierung via Streamlit/Plotly.
- **Ausgabe:** JSON mit OHLCV, Indikatoren, Signalen, Risiko-Parametern, Meta-Infos (Timeframe, Backtest-Metriken).

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
- **Sicherheit:** Sensible Daten (API-Keys, Tokens) nie in Git-Repository speichern - nur lokal in untracked .env Dateien.

## Datenfluss und Architektur
- **DataFrame-Mutationen:** In-place, mit Kopien für Backtests.
- **Signal-Logik:** Filter (z. B. Volumen > Durchschnitt); MTA via höhere Frames; ML/genetische Algorithmen für Optimierung.
- **JSON-Struktur:** Pro Zeile: OHLCV, Indikatoren, Signale, Risiko-Parameter, Meta-Infos.
- **Web-UI:** Interaktive Dashboards; sichere Authentifizierung ausbauen (z. B. OAuth).
- **Profile-System:** Automatische Validierung der Profile bei Startup; light/heavy Profile für unterschiedliche Hardware-Anforderungen.

## Integrationen
- CCXT für Exchanges (Binance, Kraken, etc.), mit Fokus auf `BTC/USDT`; erweiterbar für Forex/Aktien.
- Webhooks/Notifications für Telegram, Discord, Email.
- Container: Docker/Umbrel/Kubernetes; Volumes für Config/Output. Alternative: Conda-Umgebung `tradpal_env`.
- ML: PyTorch/Optuna für Signal-Verbesserung; Ensemble-Methoden; genetische Algorithmen.
- Web-UI: Streamlit/Plotly für Monitoring; API-Endpoint (z. B. Flask) geplant.

## Beispiele
- **Timeframe ändern:** `settings.py` -> `TIMEFRAME = '1d'`; Parameter skalieren.
- **Asset wechseln:** `settings.py` -> `SYMBOL = 'BTC/USDT'` (Standard).
- **Backtest:** `main.py --mode backtest --start_date 2024-01-01 --symbol BTC/USDT`.
- **MTA:** 1m-Signal mit 5m-EMA-Crossover bestätigen.
- **ML-Training:** `services/ml_trainer.py --optimize` für Optuna/genetische Algorithmen in `tradpal_env`.
- **Web-UI:** `conda activate tradpal_env; python services/web_ui.py`, dann http://localhost:8501 (Login: admin/admin123).
- **Setup in Conda:** `conda env create -f environment.yml; conda activate tradpal_env; pip install -r requirements.txt`.
- **Profile verwenden:** `python main.py --profile light` für minimal Ressourcen, `python main.py --profile heavy` für alle Features.

## Anweisungen für Copilot
- **Prioritäten:** Fokussiere auf Verbesserung von ML-Modellen, Sicherheit, Testabdeckung und Community-Funktionen, mit Schwerpunkt auf `BTC/USDT` in `tradpal_env`.
- **Code-Vorschläge:** Schlage konkrete Implementierungen vor (z. B. neue Indikatoren, ML-Filter mit SHAP für Explainability).
- **Tests:** Schreibe Unit-Tests (pytest) für neue Features; ziele auf >80% Coverage.
- **Dokumentation:** Erweitere README.md mit Screenshots, Setup-Guide für `tradpal_env` und Backtest-Beispielen für `BTC/USDT`.
- **Sicherheit:** Vorschläge für sichere Authentifizierung (z. B. OAuth statt admin/admin123) und Disclaimer für Trading-Risiken. Stelle sicher, dass sensible Daten nicht in Git-Repository gelangen.
- **Lizenz:** Schlage eine Open-Source-Lizenz vor (z. B. MIT) und integriere in Repository.
- **Optimierung:** Reduziere Abhängigkeiten (z. B. optionales ML-Modul); verbessere Caching (z. B. Redis).
- **Projektstruktur Best Practices:** Halte die modulare Struktur ein - `src/` für Kernlogik, `services/` für Service-Komponenten, `scripts/` für Utilities. Vermeide Verschachtelung von services/scripts unter src/. Neue Services gehören nach `services/`, neue Scripts nach `scripts/`, neue Kernmodule nach `src/`.
- **Profile-System:** Achte darauf, dass Änderungen an Profilen beide Profile (.env.light und .env.heavy) berücksichtigen und validiert werden.

## Verbesserungsvorschläge
1. **Machine Learning:** Integriere SHAP für ML-Explainability; erweitere Ensemble-Methoden (z. B. XGBoost); teste LSTM-Modelle für Zeitreihen, in `tradpal_env`.
2. **Sicherheit:** Ersetze Standard-Login (admin/admin123); füge OAuth oder JWT hinzu; Disclaimer für finanzielle Risiken. Entferne sensible Daten aus Git-Repository.
3. **Tests:** Unit-Tests für `ml_trainer.py`, Integrationstests für Workflows; CI/CD-Pipeline erweitern.
4. **Performance:** Optimiere Datenabruf für `BTC/USDT` (z. B. batchweise API-Calls); ML-Training auf GPU/Cloud auslagern.
5. **Community:** Füge Issues für bekannte Bugs (z. B. Exchange-Limitierungen) hinzu; lade zu PRs ein; veröffentliche auf PyPI.
6. **Features:** Integriere Sentiment-Analyse (z. B. via X/Twitter-Daten für `BTC/USDT`); Paper-Trading-Modus für risikofreie Tests.
7. **Dokumentation:** README mit Screenshots/GIFs; Jupyter-Notebook mit Beispiel-Backtests für `BTC/USDT` in `tradpal_env`.
8. **Profile-System:** Verbessere Validierung und Dokumentation der light/heavy Profile für bessere Benutzerfreundlichkeit.

*Letzte Aktualisierung: 10.10.2025*