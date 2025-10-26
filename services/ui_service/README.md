# TradPal UI Service

A comprehensive, consolidated user interface service providing web-based access to the TradPal trading platform with interactive dashboards, strategy builders, live charts, and monitoring capabilities.

## Overview

The UI Service delivers a complete web interface ecosystem for TradPal, featuring modern web technologies, real-time data visualization, and intuitive user experiences. Built with Streamlit and Plotly, it provides traders with powerful tools for strategy development, performance monitoring, and system management.

## Architecture

```
UI Service Ecosystem
├── Web UI Service              # Main web interface application
    ├── Authentication System   # User login and session management
    ├── Strategy Builder        # Visual strategy creation tools
    ├── Interactive Controls    # Real-time parameter adjustment
    ├── Live Charts             # Interactive trading charts
    └── Monitoring Dashboard    # Performance and system monitoring
```

## Core Components

### 🌐 Web UI Service
**Purpose**: Complete web-based trading interface with modern UX
- **Streamlit Framework**: Fast, interactive web applications
- **Plotly Charts**: Professional-grade data visualization
- **Flask Authentication**: Secure user management and sessions
- **Real-time Updates**: Live data streaming and auto-refresh
- **Responsive Design**: Works on desktop and mobile devices

### 🔐 Authentication System
**Purpose**: Secure user access and session management
- **User Registration**: Account creation with email validation
- **Secure Login**: Password hashing with Werkzeug security
- **Role-Based Access**: Admin and user permission levels
- **Session Management**: Persistent login sessions with Flask-Login
- **Password Recovery**: Secure password reset functionality

### 🎨 Strategy Builder
**Purpose**: Visual trading strategy development and management
- **Drag-and-Drop Interface**: Intuitive indicator selection
- **Parameter Configuration**: Real-time slider adjustments
- **Preset Strategies**: Ready-to-use trading templates
- **Strategy Persistence**: Save and load custom strategies
- **Backtesting Integration**: Test strategies against historical data
- **Visual Composition**: See strategy logic graphically

### ⚙️ Interactive Controls
**Purpose**: Real-time trading parameter adjustment and configuration
- **Dynamic Sliders**: Live parameter modification
- **Timeframe-Specific Settings**: Different configs per timeframe
- **Organized Categories**: Trend, Momentum, Risk Management tabs
- **Preset Configurations**: Quick access to optimized settings
- **Parameter Validation**: Real-time feedback and constraints
- **Configuration Export**: Save/load settings as JSON

### 📈 Live Charts
**Purpose**: Professional trading charts with technical analysis
- **Interactive Plotly Charts**: Zoom, pan, and detailed hover info
- **Multiple Chart Types**: Candlestick, Line, OHLC formats
- **Technical Indicators**: EMA, Bollinger Bands, RSI, Volume overlays
- **Signal Visualization**: Buy/sell markers on charts
- **Multi-Timeframe Support**: Switch between timeframes seamlessly
- **Auto-Refresh**: Live data updates with configurable intervals

### 📊 Monitoring Dashboard
**Purpose**: Real-time system and performance monitoring
- **Performance Metrics**: Win rate, returns, Sharpe ratio, drawdown
- **System Health**: Component status and connectivity checks
- **Alert Management**: Real-time notifications and warnings
- **Equity Curves**: Portfolio performance visualization
- **Trade Analysis**: Win/loss distribution and monthly returns
- **Component Monitoring**: Data fetcher, indicators, signals status

## Quick Start

### Local Development Setup

1. **Install Dependencies**
   ```bash
   cd services/ui_service/web_ui_service
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   # Copy and edit configuration
   cp .env.example .env
   # Configure database connections, API keys, etc.
   ```

3. **Start the Web Interface**
   ```bash
   # From web_ui_service directory
   streamlit run app.py

   # Or from project root
   streamlit run services/ui_service/web_ui_service/app.py
   ```

4. **Access the Interface**
   - Open browser: `http://localhost:8501`
   - Default credentials: `admin` / `admin123`
   - ⚠️ **Change default password immediately!**

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY services/ui_service/web_ui_service/requirements.txt .
RUN pip install -r requirements.txt

COPY services/ui_service/web_ui_service/ ./web_ui_service/
COPY config/ ./config/

EXPOSE 8501
CMD ["streamlit", "run", "web_ui_service/app.py", "--server.address", "0.0.0.0"]
```

### Production Deployment

```bash
# Build for production
docker build -t tradpal/ui-service:latest ./services/ui_service/web_ui_service

# Run with environment variables
docker run -p 8501:8501 \
  -e STREAMLIT_SERVER_HEADLESS=true \
  -e STREAMLIT_SERVER_PORT=8501 \
  tradpal/ui-service:latest
```

## User Interface Features

### Authentication & User Management

#### User Registration
```python
# Automatic account creation with validation
username = st.text_input("Username")
email = st.text_input("Email")
password = st.text_input("Password", type="password")

if st.button("Register"):
    # Secure registration with password hashing
    register_user(username, email, password)
```

#### Session Management
```python
# Persistent sessions with Flask-Login
@login_required
def protected_page():
    st.write(f"Welcome {current_user.username}!")
```

### Strategy Development

#### Visual Strategy Builder
```python
# Drag-and-drop indicator selection
selected_indicators = st.multiselect(
    "Choose Indicators",
    ["EMA", "RSI", "Bollinger Bands", "MACD", "ADX"]
)

# Real-time parameter adjustment
ema_short = st.slider("EMA Short Period", 5, 50, 9)
ema_long = st.slider("EMA Long Period", 10, 200, 21)
```

#### Preset Strategies
- **Trend Following**: EMA + ADX + ATR
- **Mean Reversion**: Bollinger Bands + RSI + ATR
- **Scalping**: Fast EMA + Short RSI + ATR
- **Custom**: Build your own combinations

### Interactive Trading Controls

#### Parameter Adjustment
```python
# Timeframe-specific settings
timeframe = st.selectbox("Timeframe", ["1m", "5m", "1h", "1d"])

# Dynamic parameter sliders
with st.expander("Trend Indicators"):
    ema_short = st.slider("EMA Short", 5, 50, 9)
    ema_long = st.slider("EMA Long", 10, 200, 21)

with st.expander("Risk Management"):
    stop_loss = st.slider("Stop Loss %", 0.1, 5.0, 1.0)
    take_profit = st.slider("Take Profit %", 0.1, 10.0, 2.0)
```

#### Configuration Management
```python
# Export/import configurations
config = {
    "ema_short": ema_short,
    "ema_long": ema_long,
    "stop_loss": stop_loss
}

st.download_button("Export Config", json.dumps(config))
uploaded_file = st.file_uploader("Import Config")
```

### Advanced Charting

#### Interactive Charts
```python
import plotly.graph_objects as go

# Create candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df['timestamp'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close']
)])

# Add technical indicators
fig.add_trace(go.Scatter(
    x=df['timestamp'],
    y=df['ema_short'],
    name="EMA Short"
))

# Interactive features
fig.update_layout(
    xaxis_rangeslider_visible=False,
    hovermode='x unified'
)
```

#### Real-time Updates
```python
# Auto-refresh functionality
if st.button("Enable Live Updates"):
    st.rerun()

# Manual refresh
if st.button("Refresh Data"):
    # Fetch latest data
    update_charts()
```

### Performance Monitoring

#### Dashboard Metrics
```python
# Key performance indicators
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Win Rate", "68.5%", "2.1%")

with col2:
    st.metric("Total Return", "24.7%", "1.3%")

with col3:
    st.metric("Sharpe Ratio", "1.85", "0.12")

with col4:
    st.metric("Max Drawdown", "-8.2%", "-0.5%")
```

#### System Health
```python
# Component status checks
components = {
    "Data Fetcher": "🟢 Online",
    "Indicators": "🟢 Online",
    "Signal Generator": "🟢 Online",
    "Backtester": "🟡 Warning",
    "Database": "🟢 Online"
}

for component, status in components.items():
    st.write(f"{component}: {status}")
```

## API Integration

### Service Communication

The UI Service integrates with all TradPal microservices:

```python
from services.core_service.client import CoreServiceClient
from services.data_service.client import DataServiceClient
from services.trading_service.client import TradingServiceClient

# Initialize service clients
core_client = CoreServiceClient("http://localhost:8002")
data_client = DataServiceClient("http://localhost:8000")
trading_client = TradingServiceClient("http://localhost:8005")

# Fetch data for charts
ohlcv_data = await data_client.get_ohlcv("BTC/USDT", "1h", limit=100)

# Calculate indicators
indicators = await core_client.calculate_indicators(
    "BTC/USDT", "1h", ohlcv_data, ["ema", "rsi", "bb"]
)

# Generate signals
signals = await core_client.generate_signals(
    "BTC/USDT", "1h", ohlcv_data
)
```

### Real-time Data Streaming

```python
import asyncio
import streamlit as st

async def stream_data():
    """Stream real-time data to UI"""
    while True:
        # Fetch latest data
        latest_data = await data_client.get_latest_price("BTC/USDT")

        # Update UI
        price_placeholder.metric("BTC/USDT", f"${latest_data:.2f}")

        await asyncio.sleep(1)  # Update every second

# Run streaming in background
asyncio.run(stream_data())
```

## Configuration

### Environment Variables

```bash
# Streamlit Configuration
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Authentication
UI_ADMIN_USERNAME=admin
UI_ADMIN_PASSWORD=change_this_password
UI_SESSION_TIMEOUT=3600

# Service Endpoints
CORE_SERVICE_URL=http://localhost:8002
DATA_SERVICE_URL=http://localhost:8000
TRADING_SERVICE_URL=http://localhost:8005
MONITORING_SERVICE_URL=http://localhost:8001

# Database
UI_DATABASE_PATH=users.json
UI_STRATEGIES_PATH=strategies/

# Chart Configuration
UI_CHART_HEIGHT=600
UI_AUTO_REFRESH_INTERVAL=30
UI_MAX_DATA_POINTS=1000
```

### User Configuration

Users can customize their experience:

```json
{
  "theme": "light",
  "default_symbol": "BTC/USDT",
  "default_timeframe": "1h",
  "chart_preferences": {
    "show_volume": true,
    "show_indicators": ["ema", "rsi"],
    "colors": {
      "bullish": "#00ff00",
      "bearish": "#ff0000"
    }
  },
  "notification_settings": {
    "email_alerts": true,
    "sound_enabled": false,
    "alert_thresholds": {
      "large_trade": 10000,
      "unusual_volume": 1000000
    }
  }
}
```

## Security Features

### Authentication Security
- **Password Hashing**: Werkzeug security for password storage
- **Session Security**: Secure session cookies with expiration
- **CSRF Protection**: Cross-site request forgery prevention
- **Rate Limiting**: Login attempt rate limiting

### Data Protection
- **Secure Storage**: Encrypted user data and configurations
- **Access Control**: Role-based permissions for features
- **Audit Logging**: User action logging for security monitoring
- **Data Sanitization**: Input validation and sanitization

### Network Security
- **HTTPS Enforcement**: SSL/TLS encryption in production
- **CORS Configuration**: Cross-origin resource sharing controls
- **IP Whitelisting**: Restrict access to trusted networks
- **API Key Management**: Secure API key storage and rotation

## Performance Optimization

### UI Performance
- **Lazy Loading**: Load components on demand
- **Caching**: Cache frequently accessed data
- **Pagination**: Handle large datasets efficiently
- **WebSocket Streaming**: Real-time data with minimal latency

### Chart Optimization
- **Data Sampling**: Reduce data points for better performance
- **Progressive Loading**: Load chart data incrementally
- **Memory Management**: Efficient memory usage for large datasets
- **GPU Acceleration**: Hardware-accelerated rendering

### Database Optimization
- **Connection Pooling**: Efficient database connections
- **Query Optimization**: Optimized database queries
- **Indexing**: Proper database indexing for fast lookups
- **Caching Layer**: Redis caching for improved performance

## Testing

### Unit Tests
```bash
# Test UI components
pytest services/ui_service/web_ui_service/test_components.py -v

# Test authentication
pytest services/ui_service/web_ui_service/test_auth.py -v
```

### Integration Tests
```bash
# Test service integration
pytest tests/integration/ui_service/ -v
```

### UI Tests
```bash
# Test Streamlit components
pytest tests/ui/ui_components/ -v
```

### Performance Tests
```bash
# Load testing
pytest tests/performance/ui_service/ -v -m performance
```

## Demo Scripts

### Comprehensive UI Demo
```bash
python services/ui_service/web_ui_service/demo.py
```

This demo showcases:
- User authentication flow
- Strategy builder functionality
- Interactive controls usage
- Live chart interactions
- Monitoring dashboard features
- Real-time data updates

### Individual Component Demos
```bash
# Authentication demo
python services/ui_service/web_ui_service/auth_demo.py

# Chart demo
python services/ui_service/web_ui_service/chart_demo.py

# Strategy builder demo
python services/ui_service/web_ui_service/strategy_demo.py
```

## Deployment Patterns

### Development Environment
```bash
# Local development with hot reload
streamlit run app.py --server.headless true --server.runOnSave true
```

### Production Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ui-service
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: ui-service
        image: tradpal/ui-service:latest
        ports:
        - containerPort: 8501
        env:
        - name: STREAMLIT_SERVER_HEADLESS
          value: "true"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Load Balancing
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ui-service-ingress
spec:
  rules:
  - host: trading.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ui-service
            port:
              number: 8501
```

## Troubleshooting

### Common Issues

1. **Streamlit Not Starting**
   ```bash
   # Check port availability
   lsof -i :8501

   # Use different port
   streamlit run app.py --server.port 8502
   ```

2. **Authentication Issues**
   ```bash
   # Reset user database
   rm users.json
   streamlit run app.py
   ```

3. **Chart Not Loading**
   ```bash
   # Update Plotly
   pip install --upgrade plotly

   # Clear cache
   streamlit cache clear
   ```

4. **Performance Issues**
   ```bash
   # Enable caching
   export STREAMLIT_ENABLE_CACHING=true

   # Reduce data points
   export UI_MAX_DATA_POINTS=500
   ```

### Debug Mode
```bash
# Enable debug logging
streamlit run app.py --logger.level debug --server.headless false
```

### Health Checks
```bash
# Check service health
curl http://localhost:8501/health

# Check component status
curl http://localhost:8501/api/v1/status
```

## Customization

### Theming
```python
# Custom theme configuration
st.set_page_config(
    page_title="My Trading Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {color: #1f77b4;}
    .metric-card {background-color: #f0f2f6;}
</style>
""", unsafe_allow_html=True)
```

### Adding Custom Indicators
```python
# Extend strategy builder
custom_indicators = {
    "CUSTOM_INDICATOR": {
        "name": "My Custom Indicator",
        "params": {
            "period": {"type": "slider", "min": 5, "max": 50, "default": 14},
            "multiplier": {"type": "number", "min": 1.0, "max": 5.0, "default": 2.0}
        }
    }
}
```

### Custom Dashboard Widgets
```python
# Add custom monitoring widgets
def custom_metric_widget():
    st.subheader("Custom Metrics")

    # Your custom metrics logic
    custom_value = calculate_custom_metric()

    st.metric(
        label="Custom Metric",
        value=f"{custom_value:.2f}",
        delta="0.5%"
    )
```

## Future Enhancements

- **Mobile App**: React Native mobile application
- **Real-time Collaboration**: Multi-user strategy building
- **AI Assistant**: ChatGPT-style trading assistant
- **Advanced Analytics**: Machine learning insights
- **Social Features**: Strategy sharing and marketplace
- **API Access**: REST API for third-party integrations
- **White-label Solutions**: Customizable branding

## Contributing

1. **UI/UX Standards**: Follow modern web design principles
2. **Performance**: Optimize for speed and responsiveness
3. **Accessibility**: Ensure WCAG compliance
4. **Testing**: Comprehensive test coverage for UI components
5. **Documentation**: Update user guides and API docs

## License

MIT License - see LICENSE file for details.</content>
<parameter name="filePath">/Users/danielsadowski/VSCodeProjects/tradpal/tradpal/services/ui_service/README.md