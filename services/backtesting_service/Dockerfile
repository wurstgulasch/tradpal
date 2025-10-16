# Backtesting Service Dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY services/backtesting_service/requirements.txt ./services/backtesting_service/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r services/backtesting_service/requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY services/backtesting_service/ ./services/backtesting_service/

# Create output directory
RUN mkdir -p output

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the service
CMD ["python", "-m", "services.backtesting_service.api"]