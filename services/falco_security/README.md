# TradPal Falco Runtime Security

Runtime Security Monitoring für das TradPal Trading-System mit Zero-Trust-Architektur.

## Übersicht

Falco ist ein Runtime-Sicherheits-Monitoring-Tool, das verdächtige Aktivitäten in Containern und auf dem Host-System erkennt. Diese Implementierung bietet spezielle Regeln für Trading-Systeme und integriert sich nahtlos in die bestehende TradPal-Infrastruktur.

## Features

### 🔍 Sicherheitsüberwachung
- **Container-Monitoring**: Überwacht alle Container-Aktivitäten in Echtzeit
- **Filesystem-Schutz**: Erkennt unbefugten Zugriff auf sensible Dateien
- **Netzwerk-Monitoring**: Überwacht verdächtige Netzwerkverbindungen
- **Prozess-Überwachung**: Erkennt unbefugte Prozessausführungen

### 📊 Trading-spezifische Regeln
- **API-Key-Schutz**: Überwacht Zugriffe auf Trading-API-Schlüssel
- **Exchange-Verbindungen**: Erkennt verdächtige Verbindungen zu Kryptobörsen
- **Daten-Exfiltration**: Schützt vor Datenlecks sensibler Trading-Daten
- **Wallet-Schutz**: Überwacht Zugriffe auf Kryptowallet-Dateien

### 🚨 Alert-System
- **Prioritätsbasierte Alerts**: Kritische, Warn- und Info-Alerts
- **JSON-Output**: Strukturierte Logs für SIEM-Systeme
- **Kubernetes-Integration**: Automatische Pod- und Service-Erkennung

## Architektur

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Kubernetes    │    │      Falco       │    │  Notification   │
│    Cluster      │◄──►│   DaemonSet      │◄──►│   Services      │
│                 │    │                  │    │                 │
│ • TradPal Pods  │    │ • Custom Rules   │    │ • Telegram      │
│ • Security Svc  │    │ • eBPF Engine    │    │ • Discord       │
│ • Data Services │    │ • Alert Engine   │    │ • Email         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

### Voraussetzungen

- Kubernetes Cluster (1.19+)
- kubectl Zugriff
- Cluster-Admin Berechtigungen

### Schnellinstallation

```bash
# Repository klonen und in das Verzeichnis wechseln
cd services/falco_security

# Deployment ausführen
make deploy

# Status überprüfen
make status
```

### Manuelle Installation

```bash
# 1. Namespace erstellen
kubectl apply -f k8s/namespace.yaml

# 2. RBAC-Ressourcen deployen
kubectl apply -f k8s/falco-rbac.yaml

# 3. Konfiguration deployen
kubectl apply -f k8s/falco-configmaps.yaml

# 4. Falco DaemonSet deployen
kubectl apply -f k8s/falco-daemonset.yaml

# 5. Auf Bereitschaft warten
kubectl wait --for=condition=ready pod -l app=falco -n tradpal-security --timeout=300s
```

## Konfiguration

### Falco-Konfiguration

Die Hauptkonfiguration befindet sich in `falco.yaml`:

```yaml
# JSON-Output für strukturierte Logs
json_output: true

# Kubernetes-Metadaten einschließen
k8s_meta:
  enabled: true

# Prometheus-Metriken
prometheus_metrics:
  enabled: true
  interval: 60s
```

### Benutzerdefinierte Regeln

Trading-spezifische Regeln in `tradpal_rules.yaml`:

- **Unauthorized API Key Access**: Schützt API-Schlüssel
- **Suspicious Network Connection**: Überwacht Exchange-Verbindungen
- **Sensitive Data Exfiltration**: Verhindert Datenlecks
- **High Priority Security Alert**: Kritische Sicherheitsereignisse

## Verwendung

### Logs überwachen

```bash
# Live-Logs anzeigen
make logs

# Nach Alerts filtern
kubectl logs -n tradpal-security -l app=falco | grep -i alert

# Spezifische Regeln überwachen
kubectl logs -n tradpal-security -l app=falco | grep "tradpal"
```

### Regeln aktualisieren

```bash
# Regeln aktualisieren ohne Neustart
make update-rules

# Falco neu starten
make restart
```

### Metriken überwachen

```bash
# Port-Forwarding für Prometheus-Metriken
kubectl port-forward -n tradpal-security svc/falco-metrics 9090:9090

# Metriken unter http://localhost:9090/metrics anzeigen
```

## Integration mit Notification-Services

Falco-Alerts können automatisch an die bestehenden TradPal Notification-Services weitergeleitet werden:

### Telegram/Discord Integration

```yaml
# Beispiel für Alert-Forwarding
- rule: High Priority Security Alert
  desc: Forward critical alerts to notification services
  condition: evt.priority >= CRITICAL
  output: |
    🚨 CRITICAL SECURITY ALERT 🚨
    Event: %evt.desc
    Process: %proc.name
    User: %user.name
    Container: %container.name
    Time: %evt.time
  priority: CRITICAL
```

### Webhook-Integration

Konfigurieren Sie Webhooks für automatische Alert-Weiterleitung an externe Systeme.

## Sicherheit

### Best Practices

1. **Regelmäßige Updates**: Halten Sie Falco und die Regeln aktuell
2. **Alert-Monitoring**: Überwachen Sie Alerts kontinuierlich
3. **Regel-Anpassung**: Passen Sie Regeln an Ihre spezifischen Anforderungen an
4. **Ressourcen-Monitoring**: Überwachen Sie die Ressourcennutzung von Falco

### Häufige Alerts

- **Unauthorized API Key Access**: Überprüfen Sie Prozess-Berechtigungen
- **Suspicious Network Connection**: Validieren Sie Exchange-Verbindungen
- **Sensitive Data Exfiltration**: Sicherheitsaudit durchführen

## Troubleshooting

### Häufige Probleme

**Falco-Pods starten nicht:**
```bash
kubectl describe pod -n tradpal-security -l app=falco
kubectl logs -n tradpal-security -l app=falco
```

**Falsch-positive Alerts:**
- Regeln in `tradpal_rules.yaml` anpassen
- Ausschlusslisten für bekannte Prozesse hinzufügen

**Hohe Ressourcennutzung:**
- Syscall-Filter anpassen
- Sampling-Rate reduzieren

### Debug-Modus

```bash
# Debug-Logs aktivieren
kubectl set env daemonset/falco -n tradpal-security FALCO_LOG_LEVEL=debug

# Falco neu starten
make restart
```

## Entwicklung

### Lokale Tests

```bash
# Regeln validieren
make test

# Manifeste validieren
make validate
```

### Neue Regeln hinzufügen

1. Regel in `tradpal_rules.yaml` definieren
2. Syntax mit `make test` validieren
3. Regeln mit `make update-rules` aktualisieren

## Support

Bei Problemen oder Fragen:

1. Logs überprüfen: `make logs`
2. Falco-Dokumentation: https://falco.org/docs/
3. GitHub Issues: https://github.com/falcosecurity/falco/issues

## Lizenz

Diese Implementierung ist Teil des TradPal-Systems und unterliegt der MIT-Lizenz.