# Model Monitoring Service

Ein dedizierter Service für die Überwachung von ML-Modellen in der TradPal Trading-Plattform. Bietet umfassende Funktionen für Drift-Erkennung, Performance-Tracking und Alert-Management.

## Übersicht

Der Model Monitoring Service ist ein zentraler Bestandteil der TradPal Microservices-Architektur und stellt sicher, dass ML-Modelle kontinuierlich überwacht werden. Er erkennt Daten-Drift, verfolgt Performance-Metriken und generiert automatisierte Alerts bei Problemen.

## Hauptfunktionen

### 🔍 Drift Detection (Drift-Erkennung)
- **Population Stability Index (PSI)** Berechnung für kontinuierliche Features
- **Automatische Baseline-Erstellung** aus Trainingsdaten
- **Konfigurierbare Schwellenwerte** für Alert-Generierung
- **Historische Drift-Analyse** mit Trend-Erkennung

### 📊 Performance Tracking
- **MSE, RMSE, MAE** Metriken für Regressionsmodelle
- **Accuracy, F1-Score, Precision, Recall** für Klassifikationsmodelle
- **Directional Accuracy** für Trading-Modelle
- **Performance-Degradierung** Erkennung mit konfigurierbaren Schwellen

### 🚨 Alert Management
- **Mehrstufige Alert-Eskalation** (Info → Warning → Error → Critical)
- **Automatische Alert-Routing** zu verschiedenen Kanälen (Telegram, Email, SMS)
- **Alert-Cooldown-Mechanismen** zur Vermeidung von Spam
- **Alert-Historie** und Auflösungs-Tracking

## API Endpunkte

### Modell-Management

#### `POST /register`
Registriert ein neues Modell für die Überwachung.

```json
{
  "model_id": "trading_model_v1",
  "baseline_features": {
    "rsi": [30.5, 45.2, 67.8, ...],
    "macd": [0.12, -0.05, 0.33, ...]
  },
  "baseline_metrics": {
    "mse": 0.0234,
    "directional_accuracy": 0.67
  },
  "drift_threshold": 0.1,
  "alert_thresholds": {
    "mse_degradation": 0.2,
    "directional_accuracy_drop": 0.05
  }
}
```

#### `GET /models`
Listet alle registrierten Modelle auf.

#### `DELETE /models/{model_id}`
Entfernt ein Modell aus der Überwachung.

### Monitoring

#### `POST /monitor`
Überwacht eine einzelne Modell-Vorhersage.

```json
{
  "model_id": "trading_model_v1",
  "prediction": 0.0234,
  "actual": 0.0189,
  "features": {
    "rsi": 65.4,
    "macd": 0.12
  },
  "metadata": {
    "confidence": 0.89,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### `GET /status/{model_id}`
Gibt den umfassenden Status eines Modells zurück.

```json
{
  "drift": {
    "current_score": 0.087,
    "threshold": 0.1,
    "drift_detected": false,
    "last_update": "2024-01-15T10:30:00Z"
  },
  "performance": {
    "current_metrics": {
      "mse": 0.0245,
      "directional_accuracy": 0.65
    },
    "baseline_metrics": {
      "mse": 0.0234,
      "directional_accuracy": 0.67
    },
    "degradation_alerts": []
  },
  "alerts": {
    "active_count": 0,
    "summary": {
      "total_alerts": 5,
      "alerts_by_severity": {"warning": 3, "error": 2}
    }
  }
}
```

### Alert-Management

#### `GET /alerts`
Ruft aktive Alerts und Alert-Historie ab.

Parameter:
- `model_id` (optional): Filter nach Modell
- `hours` (optional): Stunden Historie (Standard: 24)

#### `POST /alerts/{alert_id}/resolve`
Löst einen aktiven Alert auf.

```json
{
  "resolution": "Model retrained with fresh data"
}
```

### Wartung

#### `POST /maintenance/cleanup`
Bereinigt alte Monitoring-Daten.

Parameter:
- `days` (optional): Tage aufzubewahren (Standard: 30)

## Verwendung mit Python Client

```python
from services.monitoring_service.model_monitoring_service.client import ModelMonitoringClient

async def monitor_trading_model():
    async with ModelMonitoringClient() as client:
        # Modell registrieren
        await client.register_model(
            model_id="trading_model_v1",
            baseline_features={
                "rsi": [30.5, 45.2, 67.8, 52.1, 71.3],
                "macd": [0.12, -0.05, 0.33, -0.08, 0.21]
            },
            baseline_metrics={
                "mse": 0.0234,
                "directional_accuracy": 0.67
            }
        )

        # Vorhersagen überwachen
        await client.monitor_prediction(
            model_id="trading_model_v1",
            prediction=0.0234,
            actual=0.0189,
            features={"rsi": 65.4, "macd": 0.12}
        )

        # Status abrufen
        status = await client.get_model_status("trading_model_v1")
        print(f"Drift Score: {status['drift']['current_score']}")

        # Alerts prüfen
        alerts = await client.get_alerts(model_id="trading_model_v1")
        if alerts['active_alerts']:
            print("Active alerts found!")
```

## Integration in TradPal

### Service Dependencies
- **Event System Service** (Port 8011): Für Event-basierte Kommunikation
- **Notification Service** (Port 8010): Für Alert-Benachrichtigungen
- **Security Service** (Port 8001): Für Authentifizierung und Autorisierung

### Konfiguration
```python
# config/service_settings.py
MODEL_MONITORING_SERVICE_URL = "http://localhost:8020"
MODEL_MONITORING_ENABLED = True
DRIFT_DETECTION_ENABLED = True
PERFORMANCE_MONITORING_ENABLED = True
```

### Event-basierte Integration
```python
# Integration mit ml_training_service
from services.monitoring_service.model_monitoring_service.client import ModelMonitoringClient

class MLTrainingService:
    def __init__(self):
        self.monitoring_client = ModelMonitoringClient()

    async def after_training(self, model_id: str, baseline_data: dict):
        """Registriert neu trainiertes Modell für Monitoring."""
        await self.monitoring_client.register_model(
            model_id=model_id,
            baseline_features=baseline_data['features'],
            baseline_metrics=baseline_data['metrics']
        )

    async def after_prediction(self, model_id: str, prediction_data: dict):
        """Überwacht Modell-Vorhersagen."""
        await self.monitoring_client.monitor_prediction(
            model_id=model_id,
            **prediction_data
        )
```

## Architektur

### Monitoring-Module

#### `DriftDetector`
- Berechnet PSI für kontinuierliche Features
- Verwaltet Baseline-Statistiken
- Erkennt Drift-Muster über Zeit

#### `PerformanceTracker`
- Verfolgt Performance-Metriken
- Erkennt Performance-Degradierung
- Berechnet Trend-Analysen

#### `AlertManager`
- Generiert und eskaliert Alerts
- Routet Alerts zu Benachrichtigungskanälen
- Managed Alert-Lebenszyklus

### Datenfluss

```
Training Data → Baseline Creation → Model Registration
                                      ↓
Real-time Predictions → Feature Extraction → Drift Detection
                                      ↓
Actual Values → Performance Tracking → Alert Generation
                                      ↓
Notification Service → User Alerts
```

## Alert-Typen

### Drift Alerts
- **Warning**: Drift Score > Threshold
- **Error**: Drift Score > 2x Threshold
- **Critical**: Drift Score > 3x Threshold

### Performance Alerts
- **Warning**: Metrik verschlechtert sich um 10-25%
- **Error**: Metrik verschlechtert sich um 25-50%
- **Critical**: Metrik verschlechtert sich um >50%

## Metriken und Monitoring

### Prometheus Metriken
```
# HELP model_drift_score Current drift score for model
# TYPE model_drift_score gauge
model_drift_score{model="trading_model_v1"} 0.087

# HELP model_performance_mse Current MSE for model
# TYPE model_performance_mse gauge
model_performance_mse{model="trading_model_v1"} 0.0245

# HELP active_alerts_total Number of active alerts
# TYPE active_alerts_total gauge
active_alerts_total{severity="warning"} 2
```

### Health Checks
- `/health`: Service-Gesundheitsstatus
- Component-Status für alle Monitoring-Module
- Letzte Aktivitätszeiten

## Sicherheit

### Zero-Trust Prinzipien
- **mTLS**: Service-to-Service Verschlüsselung
- **JWT Authentication**: API-Zugriffskontrolle
- **Audit Logging**: Vollständige Aktivitätsprotokollierung

### Datenisolierung
- Modell-spezifische Datencontainer
- Verschlüsselte Baseline-Speicherung
- Sichere Alert-Datenübertragung

## Deployment

### Docker
```yaml
# docker-compose.yml
model-monitoring-service:
  build: ./services/monitoring_service/model_monitoring_service
  ports:
    - "8020:8020"
  environment:
    - NOTIFICATION_SERVICE_URL=http://notification-service:8010
  volumes:
    - ./cache/monitoring:/app/models
  depends_on:
    - notification-service
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-monitoring-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-monitoring-service
  template:
    metadata:
      labels:
        app: model-monitoring-service
    spec:
      containers:
      - name: model-monitoring
        image: tradpal/model-monitoring-service:latest
        ports:
        - containerPort: 8020
        env:
        - name: NOTIFICATION_SERVICE_URL
          value: "http://notification-service:8010"
        volumeMounts:
        - name: monitoring-data
          mountPath: /app/models
      volumes:
      - name: monitoring-data
        persistentVolumeClaim:
          claimName: monitoring-pvc
```

## Entwicklung

### Lokaler Start
```bash
cd services/monitoring_service/model_monitoring_service
python main.py
```

### Tests
```bash
# Unit Tests
pytest tests/unit/monitoring/

# Integration Tests
pytest tests/integration/monitoring_service/
```

### Code-Struktur
```
model_monitoring_service/
├── main.py                 # FastAPI Service
├── client.py              # Async Client
├── monitoring/            # Core Monitoring Module
│   ├── __init__.py
│   ├── drift_detector.py
│   ├── performance_tracker.py
│   └── alert_manager.py
├── models/               # Baseline/Model Data
└── requirements.txt      # Dependencies
```

## Troubleshooting

### Häufige Probleme

#### Drift Detection funktioniert nicht
- **Problem**: Baseline-Features nicht korrekt formatiert
- **Lösung**: Sicherstellen, dass Features als Dict[str, List[float]] übergeben werden

#### Performance Alerts werden nicht generiert
- **Problem**: Baseline-Metriken fehlen
- **Lösung**: Baseline-Metriken bei Modell-Registrierung angeben

#### Alerts werden nicht versendet
- **Problem**: Notification Service nicht erreichbar
- **Lösung**: NOTIFICATION_SERVICE_URL überprüfen und Service-Health prüfen

### Logs
```bash
# Service Logs
docker logs model-monitoring-service

# Alert Logs
tail -f logs/model_monitoring_alerts.log
```

### Monitoring Dashboard
- **Grafana**: http://localhost:3000 (Trading Performance Dashboard)
- **Prometheus**: http://localhost:9090 (Metriken abfragen)

## Roadmap

### Phase 1 (Aktuell)
- ✅ Drift Detection mit PSI
- ✅ Performance Tracking
- ✅ Alert Management
- ✅ REST API
- ✅ Async Client

### Phase 2 (Geplant)
- 🔄 Advanced Drift Detection (KS-Test, Chi-Quadrat)
- 🔄 Model Explainability Integration (SHAP)
- 🔄 Automated Model Retraining Triggers
- 🔄 A/B Testing Framework
- 🔄 Model Version Management

### Phase 3 (Zukunft)
- 🔄 Federated Learning Monitoring
- 🔄 Multi-Modal Model Support
- 🔄 Real-time Model Updates
- 🔄 Advanced Anomaly Detection

---

**Service Port**: 8020
**Health Endpoint**: `GET /health`
**API Documentation**: `http://localhost:8020/docs`
**Letzte Aktualisierung**: Oktober 2025