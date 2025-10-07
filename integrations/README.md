# TradPal Integration System

Das modulare Integrationssystem ermöglicht es, Trading-Signale an verschiedene Plattformen und Dienste zu senden, wie Telegram-Bots, E-Mail, Discord, Webhooks, SMS und mehr.

## Architektur

Das System basiert auf einer Plugin-Architektur mit folgenden Komponenten:

- **`integrations/base.py`**: Basis-Klassen und Integration-Manager
- **`integrations/[name]/`**: Spezifische Integration-Module
- **`integrations/example.py`**: Beispiel für die Verwendung aller Integrationen

## Verfügbare Integrationen

### ✅ Telegram Bot
- **Modul**: `integrations.telegram`
- **Funktionen**: Senden von Trading-Signalen als formatierte Nachrichten
- **Konfiguration**: Bot-Token und Chat-ID erforderlich
- **Status**: Vollständig implementiert

### ✅ E-Mail-Benachrichtigungen
- **Modul**: `integrations.email`
- **Funktionen**: Senden von Trading-Signalen als HTML-E-Mails
- **Konfiguration**: SMTP-Server, Benutzername, Passwort und Empfänger erforderlich
- **Status**: Vollständig implementiert

### ✅ Discord Webhook
- **Modul**: `integrations.discord`
- **Funktionen**: Senden von Signalen an Discord-Kanäle via Webhooks mit Embeds
- **Konfiguration**: Webhook-URL erforderlich
- **Status**: Vollständig implementiert

### ✅ Generic Webhook
- **Modul**: `integrations.webhook`
- **Funktionen**: Senden von Signalen an beliebige HTTP-Endpunkte
- **Konfiguration**: HTTP-URLs, optionale Authentifizierung
- **Status**: Vollständig implementiert

### ✅ SMS (Twilio)
- **Modul**: `integrations.sms`
- **Funktionen**: Senden von Trading-Signalen als SMS
- **Konfiguration**: Twilio Account SID, Auth Token, Telefonnummern
- **Status**: Vollständig implementiert (erfordert `pip install twilio`)

## Schnellstart

### 1. Umgebung einrichten

```bash
# Abhängigkeiten installieren
pip install -r requirements.txt

# Optionale SMS-Integration
pip install twilio

# .env-Datei erstellen
cp .env.example .env
```

### 2. Integrationen konfigurieren

```bash
# Beispiel für alle Integrationen
python integrations/example.py

# Hilfe für Konfiguration
python integrations/example.py --help
```

### 3. Einzelne Integrationen einrichten

#### Telegram
```bash
# Bot bei @BotFather erstellen
# Token und Chat-ID in .env eintragen
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

#### E-Mail
```bash
# SMTP-Konfiguration in .env
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com
```

#### Discord
```bash
# Webhook in Discord-Server erstellen
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN
```

#### Webhook
```bash
# HTTP-Endpunkte konfigurieren
WEBHOOK_URLS=https://api.example.com/webhook,https://webhook.site/test
WEBHOOK_AUTH_TOKEN=your-bearer-token
```

#### SMS (Twilio)
```bash
# Twilio-Konto erstellen
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+1234567890
SMS_TO_NUMBERS=+0987654321,+0987654322
```

## Beispiel-Verwendung

```python
from integrations.base import integration_manager
from integrations.telegram.bot import TelegramIntegration, TelegramConfig
from integrations.email_integration.email import EmailIntegration, EmailConfig

# Integrationen registrieren
telegram_config = TelegramConfig.from_env()
telegram = TelegramIntegration(telegram_config)
integration_manager.register_integration("telegram", telegram)

email_config = EmailConfig.from_env()
email = EmailIntegration(email_config)
integration_manager.register_integration("email", email)

# Alle Integrationen initialisieren
integration_manager.initialize_all()

# Signal senden
signal = {
    "timestamp": "2024-01-15T10:30:00Z",
    "symbol": "EUR/USD",
    "timeframe": "1m",
    "signal_type": "BUY",
    "price": 1.05234,
    "indicators": {"rsi": 65.5, "ema9": 1.05210},
    "risk_management": {"stop_loss_buy": 1.05150, "take_profit_buy": 1.05400}
}

integration_manager.send_signal_to_all(signal)
```

## Neue Integration hinzufügen

1. **Neues Modul erstellen**:
   ```bash
   mkdir -p integrations/[name]
   touch integrations/[name]/__init__.py
   touch integrations/[name]/[name].py
   touch integrations/[name]/config.py
   ```

2. **Basis-Klasse implementieren**:
   ```python
   from integrations.base import BaseIntegration, IntegrationConfig

   class MyIntegrationConfig(IntegrationConfig):
       def __init__(self, enabled=True, name="My Integration", api_key=None):
           super().__init__(enabled=enabled, name=name)
           self.api_key = api_key

   class MyIntegration(BaseIntegration):
       def __init__(self, config: MyIntegrationConfig):
           super().__init__(config)

       def initialize(self) -> bool:
           return True

       def send_signal(self, signal_data: dict) -> bool:
           return True

       def test_connection(self) -> bool:
           return True
   ```

3. **In `integrations/example.py` hinzufügen** für automatische Tests.

## Signal-Format

Alle Integrationen erhalten Signale im standardisierten Format:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "symbol": "EUR/USD",
  "timeframe": "1m",
  "signal_type": "BUY",
  "price": 1.05234,
  "indicators": {
    "ema9": 1.05210,
    "ema21": 1.05195,
    "rsi": 65.5,
    "bb_upper": 1.05350,
    "bb_middle": 1.05225,
    "bb_lower": 1.05100,
    "atr": 0.00050
  },
  "risk_management": {
    "position_size_percent": 2.0,
    "position_size_absolute": 2000.0,
    "stop_loss_buy": 1.05150,
    "take_profit_buy": 1.05400,
    "leverage": 5.0
  },
  "confidence": 0.85,
  "reason": "Strong bullish momentum with RSI above 60"
}
```

## Konfiguration

Integrationen werden über Umgebungsvariablen konfiguriert:

```bash
# .env-Datei
# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Email
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECIPIENTS=recipient@example.com

# Discord
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Webhook
WEBHOOK_URLS=https://api.example.com/webhook
WEBHOOK_AUTH_TOKEN=bearer_token

# SMS (Twilio)
TWILIO_ACCOUNT_SID=account_sid
TWILIO_AUTH_TOKEN=auth_token
TWILIO_FROM_NUMBER=+1234567890
SMS_TO_NUMBERS=+0987654321
```

## Testumgebung

Für Tests wird automatisch `.env.test` verwendet (keine echten Benachrichtigungen):

```bash
# .env.test
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
EMAIL_USERNAME=
# ... alle anderen leer lassen
```

## Fehlerbehebung

### Integration funktioniert nicht

1. **Konfiguration prüfen**:
   ```bash
   python integrations/example.py
   ```

2. **Logs aktivieren**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Verbindung testen**:
   ```python
   integration.test_connection()
   ```

### Import-Fehler

- Stellen Sie sicher, dass optionale Pakete installiert sind
- SMS-Integration erfordert: `pip install twilio`

## Erweiterte Funktionen

- **Multi-Channel**: Gleichzeitiges Senden an mehrere Kanäle
- **Retry-Mechanismen**: Automatische Wiederholung bei Fehlern
- **Rate Limiting**: Respektiert API-Limits verschiedener Dienste
- **Template-System**: Anpassbare Nachrichtenformate
- **Async Support**: Nicht-blockierende Signal-Übertragung

## Roadmap

- [x] Telegram Bot-Integration
- [x] E-Mail-Integration
- [x] Discord Webhook-Integration
- [x] Generic Webhook-Integration
- [x] SMS-Integration (Twilio)
- [ ] Push-Notification-Integration
- [ ] Slack-Integration
- [ ] TradingView Webhook-Integration