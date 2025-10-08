# TradPal Integration System

The modular integration system allows sending trading signals to various platforms and services, such as Telegram bots, email, Discord, webhooks, SMS, and more.

## Architecture

The system is based on a plugin architecture with the following components:

- **`integrations/base.py`**: Base classes and integration manager
- **`integrations/[name]/`**: Specific integration modules
- **`integrations/example.py`**: Example for using all integrations

## Available Integrations

### ✅ Telegram Bot
- **Module**: `integrations.telegram`
- **Features**: Send trading signals as formatted messages
- **Configuration**: Bot token and chat ID required
- **Status**: Fully implemented

### ✅ Email Notifications
- **Module**: `integrations.email`
- **Features**: Send trading signals as HTML emails
- **Configuration**: SMTP server, username, password, and recipients required
- **Status**: Fully implemented

### ✅ Discord Webhook
- **Module**: `integrations.discord`
- **Features**: Send signals to Discord channels via webhooks with embeds
- **Configuration**: Webhook URL required
- **Status**: Fully implemented

### ✅ Generic Webhook
- **Module**: `integrations.webhook`
- **Features**: Send signals to any HTTP endpoints
- **Configuration**: HTTP URLs, optional authentication
- **Status**: Fully implemented

### ✅ SMS (Twilio)
- **Module**: `integrations.sms`
- **Features**: Send trading signals as SMS
- **Configuration**: Twilio Account SID, Auth Token, phone numbers
- **Status**: Fully implemented (requires `pip install twilio`)

## Quick Start

### 1. Set Up Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Optional SMS integration
pip install twilio

# Create .env file
cp .env.example .env
```

### 2. Configure Integrations

```bash
# Example for all integrations
python integrations/example.py

# Help for configuration
python integrations/example.py --help
```

### 3. Set Up Individual Integrations

#### Telegram
```bash
# Create bot with @BotFather
# Add token and chat ID to .env
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

#### Email
```bash
# SMTP configuration in .env
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com
```

#### Discord
```bash
# Create webhook in Discord server
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN
```

#### Webhook
```bash
# Configure HTTP endpoints
WEBHOOK_URLS=https://api.example.com/webhook,https://webhook.site/test
WEBHOOK_AUTH_TOKEN=your-bearer-token
```

#### SMS (Twilio)
```bash
# Create Twilio account
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+1234567890
SMS_TO_NUMBERS=+0987654321,+0987654322
```

## Example Usage

```python
from integrations.base import integration_manager
from integrations.telegram.bot import TelegramIntegration, TelegramConfig
from integrations.email_integration.email import EmailIntegration, EmailConfig

# Register integrations
telegram_config = TelegramConfig.from_env()
telegram = TelegramIntegration(telegram_config)
integration_manager.register_integration("telegram", telegram)

email_config = EmailConfig.from_env()
email = EmailIntegration(email_config)
integration_manager.register_integration("email", email)

# Initialize all integrations
integration_manager.initialize_all()

# Send signal
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

## Adding New Integration

1. **Create new module**:
   ```bash
   mkdir -p integrations/[name]
   touch integrations/[name]/__init__.py
   touch integrations/[name]/[name].py
   touch integrations/[name]/config.py
   ```

2. **Implement base class**:
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

3. **Add to `integrations/example.py`** for automatic tests.

## Signal Format

All integrations receive signals in standardized format:

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

## Configuration

Integrations are configured via environment variables:

```bash
# .env file
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

## Test Environment

For tests, `.env.test` is automatically used (no real notifications):

```bash
# .env.test
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
EMAIL_USERNAME=
# ... leave all others empty
```

## Troubleshooting

### Integration not working

1. **Check configuration**:
   ```bash
   python integrations/example.py
   ```

2. **Enable logs**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Test connection**:
   ```python
   integration.test_connection()
   ```

### Import Errors

- Ensure optional packages are installed
- SMS integration requires: `pip install twilio`

## Advanced Features

- **Multi-Channel**: Simultaneous sending to multiple channels
- **Retry Mechanisms**: Automatic retry on errors
- **Rate Limiting**: Respects API limits of different services
- **Template System**: Customizable message formats
- **Async Support**: Non-blocking signal transmission

## Roadmap

- [x] Telegram Bot Integration
- [x] Email Integration
- [x] Discord Webhook Integration
- [x] Generic Webhook Integration
- [x] SMS Integration (Twilio)
- [ ] Push Notification Integration
- [ ] Slack Integration
- [ ] TradingView Webhook Integration