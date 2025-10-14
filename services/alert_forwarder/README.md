# TradPal Alert Forwarder Service

Automatische Weiterleitung von Falco-Sicherheitsmeldungen an Notification-Services (Telegram, Discord, Email).

## Übersicht

Der Alert Forwarder Service überwacht kontinuierlich Falco-Logs und leitet Sicherheitsmeldungen automatisch an konfigurierte Notification-Services weiter. Er unterstützt Alert-Batching, Rate-Limiting und flexible Filteroptionen.

## Features

### 🔄 Alert-Monitoring
- **Echtzeit-Überwachung**: Kontinuierliche Überwachung von Falco-Logdateien
- **Intelligente Parsing**: Automatische Erkennung von JSON und Plain-Text Falco-Logs
- **Duplikat-Filterung**: Verhinderung von doppelten Alert-Weiterleitungen

### 📊 Alert-Verarbeitung
- **Prioritäts-Filterung**: Konfigurierbare Mindestpriorität für Alert-Weiterleitung
- **Batching**: Gruppierung mehrerer Alerts für effizientere Benachrichtigungen
- **Rate-Limiting**: Schutz vor Alert-Spam mit konfigurierbaren Limits

### 📱 Multi-Channel Notifications
- **Telegram**: Formatierte Nachrichten mit Emojis und Markdown
- **Discord**: Rich Embeds mit Farbcodierung nach Schweregrad
- **Email**: Erweiterbar für E-Mail-Benachrichtigungen

## Architektur

```
Falco Logs → Alert Forwarder → Notification Services
     ↓              ↓              ↓
  JSON/Text     Parsing &       Telegram
  Parsing       Filtering       Discord
                Batching        Email
```

## Installation

### Lokale Ausführung

```bash
# Abhängigkeiten installieren
make install

# Service starten
make run
```

### Kubernetes Deployment

```bash
# Deployen
make deploy

# Status überprüfen
make status

# Logs anzeigen
make logs
```

## Konfiguration

### Umgebungsvariablen

```bash
# Falco Konfiguration
FALCO_LOG_PATH=/var/log/falco/falco.log
FALCO_NAMESPACE=tradpal-security

# Alert Filterung
ALERT_MIN_PRIORITY=WARNING          # EMERGENCY, ALERT, CRITICAL, ERROR, WARNING, NOTICE, INFO, DEBUG
ALERT_ENABLED_RULES=                # JSON Array mit erlaubten Regeln (optional)

# Batching
ALERT_BATCHING=true                 # Alerts gruppieren
ALERT_BATCH_INTERVAL=300            # Batch-Intervall in Sekunden
ALERT_MAX_BATCH_SIZE=10             # Maximale Alerts pro Batch

# Rate Limiting
ALERT_RATE_LIMIT=10                 # Max Alerts pro Minute
ALERT_COOLDOWN=60                   # Abklingzeit in Sekunden

# Notification Services
TELEGRAM_ENABLED=true
DISCORD_ENABLED=true
EMAIL_ENABLED=false

# Telegram Konfiguration
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Discord Konfiguration
DISCORD_WEBHOOK_URL=your_webhook_url
```

### Kubernetes Secrets

Für Kubernetes-Deployments müssen die Secrets konfiguriert werden:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: alert-forwarder-secrets
  namespace: tradpal-security
type: Opaque
data:
  telegram-enabled: "dHJ1ZQ=="      # base64: true
  discord-enabled: "dHJ1ZQ=="       # base64: true
  telegram-bot-token: "<base64>"    # Ihr Bot-Token
  telegram-chat-id: "<base64>"      # Ihre Chat-ID
  discord-webhook-url: "<base64>"   # Ihre Webhook-URL
```

## Alert-Formate

### Telegram Format

```
🚨 FALCO SECURITY ALERT 🚨

🚨 Rule: Unauthorized API Key Access
⚠️ Priority: WARNING
🕐 Time: 2025-10-15T10:30:00.000000000Z

📝 Details:
Unauthorized access to sensitive file (file=/etc/api_keys.bin proc=python user=root)

🔍 Source: falco
🏠 Host: tradpal-node-01
🏷️ Tags: filesystem, trading, security
```

### Discord Embed

Rich Embed mit:
- **Titel**: Falco Security Alert mit Emoji
- **Beschreibung**: Regel und Priorität
- **Farbcodierung**: Nach Schweregrad (Rot für CRITICAL, Orange für WARNING, etc.)
- **Felder**: Details, Source, Host, Tags
- **Footer**: TradPal Security Monitoring

## Verwendung

### Lokaler Test

```bash
# Konfiguration validieren
make validate

# Test-Alert erstellen
make test-alert

# Service testen
make test
```

### Produktionsbetrieb

```bash
# Service starten
make run

# Oder Kubernetes
make deploy
make logs
```

### Monitoring

```bash
# Logs überwachen
make logs

# Status prüfen
make status

# Ressourcen-Nutzung
kubectl top pods -n tradpal-security -l app=alert-forwarder
```

## Alert-Regeln

### Standard-Filter

- **Priorität**: Nur WARNING und höher werden weitergeleitet
- **Rate-Limit**: Max. 10 Alerts pro Minute
- **Cooldown**: 60 Sekunden nach Rate-Limit-Erreichung

### Batch-Verarbeitung

- **Intervall**: 5 Minuten
- **Max-Größe**: 10 Alerts pro Batch
- **Gruppierung**: Nach Priorität

## Sicherheit

### Best Practices

1. **Secrets-Management**: Verwenden Sie Kubernetes-Secrets für sensible Daten
2. **Netzwerk-Sicherheit**: Beschränken Sie Pod-Netzwerkzugriffe
3. **Ressourcen-Limits**: Setzen Sie CPU und Memory-Limits
4. **Log-Rotation**: Implementieren Sie Log-Rotation für Falco-Logs

### Häufige Sicherheitsmeldungen

- **Unauthorized API Key Access**: API-Schlüssel-Zugriff überwachen
- **Suspicious Network Connection**: Netzwerkanomalien erkennen
- **Sensitive Data Exfiltration**: Datenlecks verhindern
- **Privileged Container Escape**: Container-Breaks erkennen

## Troubleshooting

### Häufige Probleme

**Keine Alerts werden empfangen:**
```bash
# Logs prüfen
make logs

# Konfiguration validieren
make validate

# Falco-Logs prüfen
kubectl logs -n tradpal-security -l app=falco | tail -20
```

**Rate-Limiting zu aggressiv:**
```bash
# Rate-Limit erhöhen
export ALERT_RATE_LIMIT=20
make run
```

**Batching zu langsam:**
```bash
# Batch-Intervall reduzieren
export ALERT_BATCH_INTERVAL=60
make run
```

### Debug-Modus

```bash
# Debug-Logging aktivieren
export PYTHONPATH=/app
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# ... Debug-Code ...
"
```

## Integration

### Mit Falco

Der Service integriert sich automatisch mit Falco DaemonSets und liest die Logdateien direkt vom Host-System.

### Mit Notification-Services

- **Telegram**: Verwendet bestehende Telegram-Integration
- **Discord**: Verwendet bestehende Discord-Integration
- **Email**: Erweiterbar für SMTP-basierte Benachrichtigungen

## Entwicklung

### Neue Notification-Services hinzufügen

1. Integration in `forwarder.py` implementieren
2. Konfiguration in `config.py` hinzufügen
3. Kubernetes-Secrets erweitern

### Alert-Format anpassen

1. `FalcoAlert.format_for_*()` Methoden erweitern
2. Neue Templates für verschiedene Alert-Typen

## Support

Bei Problemen:

1. Logs überprüfen: `make logs`
2. Konfiguration validieren: `make validate`
3. Falco-Status prüfen: `kubectl get pods -n tradpal-security`

## Lizenz

Teil des TradPal Zero-Trust Security Frameworks.