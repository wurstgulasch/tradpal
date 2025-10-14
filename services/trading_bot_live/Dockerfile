# Trading Bot Live Service Dockerfile

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY services/trading_bot_live/requirements.txt ./service_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r service_requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY services/trading_bot_live/ ./services/trading_bot_live/

# Create logs directory
RUN mkdir -p logs output

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from services.trading_bot_live.service import TradingBotLiveService; print('Service import OK')" || exit 1

# Default command
CMD ["python", "-m", "services.trading_bot_live.service", "--mode", "paper"]