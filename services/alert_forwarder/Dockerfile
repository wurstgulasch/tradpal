# Alert Forwarder Dockerfile

FROM python:3.10-slim

# Labels
LABEL maintainer="TradPal Team"
LABEL description="Alert Forwarder Service for Falco Security Monitoring"
LABEL version="1.0.0"

# Arbeitsverzeichnis
WORKDIR /app

# System-Abhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Anwendungscode kopieren
COPY config.py forwarder.py cli.py tests.py ./
COPY README.md ./

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; from forwarder import AlertForwarder; print('Health check passed')" || exit 1

# Standardbefehl
CMD ["python", "cli.py", "run"]

# Ports (falls API-Endpunkt hinzugefügt wird)
EXPOSE 8080

# Sicherheit: Nicht-root User
RUN useradd --create-home --shell /bin/bash alertforwarder
USER alertforwarder

# Volumes für Logs und Config
VOLUME ["/var/log/falco", "/app/config"]