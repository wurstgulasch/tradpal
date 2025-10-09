# Copilot Instructions for AI Agents

## Überblick
Dieses Projekt implementiert ein modulares Trading-Indikator-System, das primär für 1-Minuten-Charts optimiert ist, aber skalierbar auf höhere Timeframes (z. B. 1h, 1d, 1w, 1m) erweitert werden kann. Es basiert auf einer Kombination aus EMA (Exponential Moving Averages), RSI (Relative Strength Index), Bollinger Bands (BB) und ATR (Average True Range) für die Generierung von Buy/Sell-Signalen sowie Risikomanagement-Parameter wie Positionsgröße, Leverage, Stop-Loss (SL) und Take-Profit (TP). Optionale Erweiterungen umfassen ADX (Average Directional Index) für Trendstärke, Fibonacci Extensions für Zielwerte und Volatilitätsindikatoren (z. B. VIX oder BB-Bandwidth) für dynamische Anpassungen. Das System ist containerisiert und für Integration in Dienste wie Telegram-Bots oder Webhooks ausgelegt, mit Fokus auf Backtesting und Multi-Timeframe-Analyse (MTA) für robustere Signale.

## Projektstruktur
- `config/settings.py`: Zentrale Konfiguration (Indikator-Parameter, Timeframes, Exchanges, Assets, Risikoparameter, Ausgabeformate). Enthält skalierbare Parameter-Tabellen für verschiedene Timeframes.
- `src/data_fetcher.py`: Datenabruf mit ccxt von Exchanges; unterstützt multiple Timeframes und Assets (z. B. Forex, Krypto, Aktien).
- `src/indicators.py`: Berechnung technischer Indikatoren (EMA, RSI, BB, ATR; optional ADX, Fibonacci, Volatilitätsmetriken).
- `src/signal_generator.py`: Signalgenerierung, Risikomanagement und MTA-Integration (z. B. höhere Timeframes für Trend-Kontext).
- `src/output.py`: JSON-Ausgabe für Integration; inklusive Backtest-Reports und visueller Exports (z. B. via Matplotlib für Charts).
- `src/backtester.py`: Neu: Modul für historische Backtests mit Pandas/TA-Lib zur Validierung von Strategien.
- `main.py`: Orchestrierung der Module; unterstützt Modi wie "live", "backtest" und "multi-timeframe".
- `output/`: JSON-Dateien mit Signalen, Risiko-Parametern und Backtest-Ergebnissen.
- `Dockerfile` & `docker-compose.yml`: Container-Setup für Umbrel-kompatible Ausführung; inklusive Volume-Mounts für persistente Daten.

## Wichtige Workflows
- **Daten holen:** ccxt für OHLCV-Daten in konfigurierbarem Timeframe (z. B. '1m', '1h') von Exchanges; Multi-Asset-Unterstützung (z. B. 'EUR/USD' für Forex).
- **Indikatoren berechnen:** Manuelle Funktionen für EMA, RSI, BB, ATR; skalierbare Parameter (z. B. EMA 9/21 für 1m, 50/200 für 1d); optionale ADX für Trendfilter (z. B. ADX > 25 für starke Trends) und Fibonacci für TP-Projektionen.
- **Signale generieren:**
  - Basis: Buy = (EMA_kurz > EMA_lang) & (RSI < Oversold_Threshold) & (close > BB_lower); Sell = (EMA_kurz < EMA_lang) & (RSI > Overbought_Threshold) & (close < BB_upper).
  - Anpassung pro Timeframe: Z. B. RSI-Thresholds flexibel (30/70 für 1m, 40/60 für 1d zur Reduzierung von Fehlsignalen).
  - MTA: Signale nur bestätigen, wenn höherer Timeframe (z. B. 5m für 1m-Trades) denselben Trend zeigt.
- **Risikomanagement:**
  - Positionsgröße = (Kapital * Risikoprozent, z. B. 1%) / (ATR * Multiplier, z. B. 1.5).
  - SL = close - (ATR * 1–1.5); TP = close + (ATR * 2–3); Leverage = dynamisch (niedrig bei hohem ATR, z. B. max. 1:10 für 1m).
  - Erweiterung: ADX für Trade-Dauer (länger halten bei ADX > 25); Fibonacci-Levels (z. B. 161.8%) für TP in Trends.
- **Backtesting:** Historische Daten laden, Strategie simulieren; Metriken wie Win-Rate, CAGR, Drawdown berechnen.
- **Ausgabe:** JSON-Dateien in `output/`; Struktur erweitert um Timeframe-spezifische Metriken und Backtest-Results für Integration in Bots.
- **Container:** `docker-compose up` für Ausführung; integrierte TA-Lib und Pandas für Backtests.

## Konventionen
- Modulare Struktur: Jedes Modul hat eine spezifische Verantwortung; lose Kopplung via DataFrames.
- Konfiguration zentral in `config/settings.py`; inklusive Tabellen für Timeframe-spezifische Parameter (z. B. EMA-Perioden skalieren mit Frame-Größe).
- JSON-Ausgabe für API-ähnliche Integration; erweitert um Warnungen (z. B. "Hohe Volatilität – Leverage reduzieren").
- Keine harten Abhängigkeiten; verwende Dependency Injection für Testbarkeit.
- NaN-Handling: dropna() und Forward-Fill vor Berechnungen; robuste Error-Handling für API-Ausfälle.
- Englische Kommentare, Variablennamen und Docstrings; PEP-8-konform.
- sys.path.append() in main.py für relative Imports; Logging mit logging-Modul für Debug.
- Skalierbarkeit: Parameter als Dicts in Config für einfache Anpassung (z. B. {'1m': {'ema_short': 9, 'ema_long': 21}, '1d': {'ema_short': 50, 'ema_long': 200}}).

## Datenfluss und Architektur
- **DataFrame-Mutationen:** Module modifizieren das DataFrame in-place und geben es zurück; Versionierung für Backtests (z. B. Kopien speichern).
- **Signal-Logik:** Erweitert um Filter (z. B. Volumen > Durchschnitt für Bestätigung); MTA-Integration: Höhere Frames laden und Trends vergleichen.
- **Risiko-Formeln:** Position_Size = (CAPITAL * RISK_PER_TRADE) / ATR; Stop_Loss = close - (ATR * SL_MULTIPLIER); Take_Profit = close + (ATR * TP_MULTIPLIER); Leverage = min(MAX_LEVERAGE, BASE_LEVERAGE / (ATR / ATR_MEAN)).
- **JSON-Struktur:** Pro Zeile: OHLCV + Indikatoren (EMA, RSI, BB, ATR, optional ADX/Fib) + Signale (Buy/Sell/Neutral) + Risiko-Parameter (Size, SL, TP, Leverage, Dauer-Estimation); plus Meta-Infos wie Timeframe und Backtest-Metriken.
- **Erweiterungen:** Optionale Module für ADX (Trendstärke), Fibonacci (Extensions für TP) und Volatilitätsfilter (z. B. BB-Bandwidth < Threshold für Range-Märkte).

## Integrationen
- ccxt für Exchange-Daten (unterstützt Binance, Kraken, etc.); erweitert um Multi-Asset (z. B. Krypto, Forex).
- JSON-Ausgabe für Webhooks/Bots; Hooks für Telegram-Notifications bei Signalen.
- Container-ready für Umbrel, Kubernetes oder lokale Docker; Volumes für Config und Output.
- Backtesting-Integration: Verwende Pandas für Simulation; exportiere Reports als CSV/JSON.
- Optionale Erweiterungen: API-Endpoint (z. B. via Flask) für externe Queries; Machine-Learning-Filter (z. B. mit scikit-learn für Signal-Verbesserung).

## Beispiele
- **Timeframe ändern:** `config/settings.py` -> TIMEFRAME = '1d'; Parameter auto-skalieren.
- **Asset wechseln:** `config/settings.py` -> SYMBOL = 'EUR/USD'; Exchange anpassen.
- **Ausgabe lesen:** JSON in `output/signals.json` parsen für Bot; enthält SL/TP/Levarage.
- **Neuer Indikator hinzufügen:** In `indicators.py` implementieren (z. B. ADX), in `signal_generator.py` als Filter verwenden (z. B. if ADX > 25).
- **Backtest ausführen:** `main.py --mode backtest --start_date 2024-01-01` für historische Tests.
- **MTA-Beispiel:** Für 1m-Signal: Überprüfe 5m- oder 15m-Trend mit EMA-Crossover.

## Verbesserungsvorschläge
- **Aktuelle Anpassungen basierend auf Diskussion:** Ich habe Skalierbarkeit für Timeframes integriert (Parameter-Tabellen), MTA hinzugefügt, optionale Indikatoren (ADX, Fibonacci) und dynamisches Leverage/Risikomanagement erweitert, um den besprochenen Anforderungen (Buy/Sell-Signale, Positionsgröße, Dauer, etc.) zu entsprechen. Das macht das System flexibler für Scalping bis Langfrist-Trading.
- **Weitere Verbesserungen:**
  - Füge ein Logging-System hinzu für Audit-Trails (z. B. Signal-Historie).
  - Integriere TA-Lib als Dependency für effizientere Indikator-Berechnungen.
  - Erweitere auf Machine-Learning: Z. B. ein simples Modell zur Signal-Vorhersage basierend auf historischen Daten.
  - Test-Suite: Unit-Tests für Module (z. B. mit pytest) und Integrationstests für Workflows.
  - Performance: Caching für Datenabruf, um API-Limits zu respektieren.
  - Sicherheit: Environment-Variablen für API-Keys in Config.
  - Dokumentation: Erweitere mit einem README.md inklusive Setup-Anleitung und Strategie-Backtest-Beispielen.

---
*Letzte Aktualisierung: 07.10.2025*
