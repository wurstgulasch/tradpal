# TradPal - Web UI

A comprehensive, interactive web interface for the TradPal trading system featuring authentication, real-time charts, strategy builder, and interactive controls.

## 🎯 Features

### 🔐 Authentication System
- Secure user login and registration
- Session management with Flask-Login
- Role-based access control (admin/user)
- Password hashing with Werkzeug
- User management (create, update, delete accounts)

### 🎨 Strategy Builder
- **Drag-and-drop interface** for indicator selection
- Configure multiple indicators with custom parameters
- Real-time parameter adjustment with sliders
- **Preset strategies** (Trend Following, Mean Reversion, Scalping)
- Save and load custom strategies
- Strategy backtesting integration
- Visual strategy composition

### ⚙️ Interactive Controls
- **Real-time parameter adjustment** with interactive sliders
- Timeframe-specific parameter sets (1m, 5m, 1h, 1d, etc.)
- Organized tabs for different indicator categories:
  - Trend Indicators (EMA, Bollinger Bands)
  - Momentum Indicators (RSI, ADX)
  - Risk Management (ATR, Stop Loss, Take Profit, Leverage)
  - Advanced Settings
- Visual feedback for parameter validation
- Quick preset configurations
- Export/import configuration as JSON
- Live parameter testing

### 📈 Live Charts with Plotly
- **Interactive candlestick charts** with zoom, pan, and hover details
- Multiple chart types (Candlestick, Line, OHLC)
- Real-time indicator overlay:
  - Exponential Moving Averages (EMA)
  - Bollinger Bands
  - RSI with overbought/oversold levels
  - Volume bars
  - Buy/Sell signal markers
- Multi-panel charts with synchronized timeframes
- Signal strength gauge
- Customizable indicator display (toggle on/off)
- Auto-refresh functionality

### 📊 Monitoring Dashboard
- Real-time performance metrics
- System health monitoring
- Active alerts and notifications
- Equity curve visualization
- Win/Loss distribution
- Monthly returns analysis
- Component status checks

## 🚀 Quick Start

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Required packages:
- `streamlit>=1.28.0` - Web UI framework
- `plotly>=5.17.0` - Interactive charts
- `flask>=3.0.0` - Authentication backend
- `flask-login>=0.6.3` - Session management
- `werkzeug>=3.0.0` - Password security

### Running the Web UI

1. **Navigate to the web-ui directory:**
```bash
cd services/web-ui
```

2. **Start the application:**
```bash
streamlit run app.py
```

3. **Access the interface:**
Open your browser and navigate to: `http://localhost:8501`

4. **Default credentials:**
- Username: `admin`
- Password: `admin123`
- ⚠️ **Change these in production!**

### Alternative: Run from project root

```bash
streamlit run services/web-ui/app.py
```

## 📖 User Guide

### Authentication

#### First Time Setup
1. Open the web interface
2. Use default credentials or register a new account
3. Navigate to different sections using the sidebar

#### Changing Password
1. Login to your account
2. Go to user settings (coming soon)
3. Enter current and new password
4. Confirm changes

### Using the Strategy Builder

#### Creating a Custom Strategy
1. Navigate to **"🎨 Strategy Builder"** from the sidebar
2. Select indicators from the left panel
3. Click **"➕ Add"** to add them to your strategy
4. Configure parameters using the sliders
5. Name your strategy
6. Click **"💾 Save Strategy"** to persist

#### Loading Preset Strategies
1. In the Strategy Builder, look for **"⭐ Preset Strategies"**
2. Choose from:
   - **Trend Following**: EMA + ADX + ATR
   - **Mean Reversion**: Bollinger Bands + RSI + ATR
   - **Scalping**: Fast EMA + Short RSI + ATR
3. Customize parameters as needed

#### Testing a Strategy
1. Build or load a strategy
2. Click **"🧪 Test Strategy"**
3. Review performance metrics:
   - Win Rate
   - Total Trades
   - Profit Factor
4. Adjust parameters based on results

### Interactive Controls

#### Adjusting Parameters
1. Navigate to **"⚙️ Interactive Controls"**
2. Select your timeframe (1m, 5m, 1h, etc.)
3. Use sliders to adjust parameters:
   - **Trend Tab**: EMA periods, Bollinger Bands
   - **Momentum Tab**: RSI levels, ADX threshold
   - **Risk Management Tab**: Stop Loss, Take Profit, Leverage
   - **Advanced Tab**: Presets and configuration management

#### Applying Settings
1. Adjust parameters as desired
2. Click **"✅ Apply Settings"**
3. Settings are applied to your active strategy
4. Optionally **"💾 Save as Default"** for the timeframe

#### Using Presets
Quick access to optimized configurations:
- **🏃 Scalping**: Fast indicators, tight stops, higher leverage
- **📈 Trend Following**: Slower indicators, wider stops, moderate leverage
- **💰 Conservative**: Very slow indicators, wide stops, low leverage

### Live Charts

#### Chart Configuration
1. Navigate to **"📈 Live Charts"**
2. Select:
   - **Symbol**: BTC/USDT, ETH/USDT, EUR/USD, etc.
   - **Timeframe**: 1m, 5m, 15m, 1h, 4h, 1d
   - **Chart Type**: Candlestick, Line, or OHLC

#### Indicator Display
Toggle indicators on/off:
- ✅ **EMA**: Shows fast and slow moving averages
- ✅ **RSI**: Displays momentum with overbought/oversold levels
- ✅ **Bollinger Bands**: Volatility bands around price
- ✅ **Volume**: Trading volume bars
- ✅ **Signals**: Buy/Sell markers on the chart

#### Interactive Features
- **Zoom**: Click and drag on chart
- **Pan**: Hold shift and drag
- **Hover**: See detailed information at any point
- **Auto Refresh**: Enable for live updates
- **Reset**: Double-click to reset view

### Monitoring Dashboard

#### Key Metrics
Monitor performance in real-time:
- **Win Rate**: Percentage of winning trades
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline

#### System Status
Check health of system components:
- Data Fetcher: API connectivity
- Indicators: Calculation status
- Signal Generator: Signal production
- Backtester: Testing engine
- Database: Cache status

## 🏗️ Architecture

### Project Structure
```
services/web-ui/
├── app.py                      # Main application entry point
├── auth.py                     # Authentication module
├── strategy_builder.py         # Strategy builder UI component
├── interactive_controls.py     # Interactive controls UI component
├── live_charts.py              # Live charts UI component
├── monitoring_dashboard.py     # Monitoring dashboard
├── users.json                  # User database (auto-generated)
├── strategies/                 # Saved strategies directory
└── README.md                   # This file
```

### Component Overview

#### app.py
- Main Streamlit application
- Routing and navigation
- Session state management
- Component orchestration

#### auth.py
- User authentication
- Password hashing and verification
- User registration and management
- Session handling

#### strategy_builder.py
- Indicator selection interface
- Parameter configuration
- Strategy saving/loading
- Preset strategies

#### interactive_controls.py
- Real-time parameter adjustment
- Timeframe-specific settings
- Configuration presets
- Parameter validation

#### live_charts.py
- Plotly chart creation
- Real-time data visualization
- Indicator overlay
- Signal display

#### monitoring_dashboard.py
- Performance monitoring
- System health checks
- Alert management
- Data visualization

## 🔒 Security

### Best Practices

1. **Change Default Credentials**
   - Immediately change the default admin password
   - Use strong passwords (min 12 characters)

2. **User Management**
   - Regularly audit user accounts
   - Remove inactive users
   - Use appropriate role assignments

3. **Data Protection**
   - `users.json` stores hashed passwords only
   - Keep `users.json` in `.gitignore`
   - Regular backups recommended

4. **Network Security**
   - Use HTTPS in production
   - Configure firewall rules
   - Limit access to trusted IPs

## 🎨 Customization

### Theming
Customize the appearance in `app.py`:
```python
st.set_page_config(
    page_title="Your Custom Title",
    page_icon="🎯",
    layout="wide"
)
```

### Adding Custom Indicators
Edit `strategy_builder.py`:
```python
self.available_indicators = {
    "YOUR_INDICATOR": {
        "name": "Your Indicator Name",
        "params": {
            "param1": {"type": "slider", "min": 1, "max": 100, "default": 14}
        }
    }
}
```

### Custom Presets
Add to `interactive_controls.py`:
```python
presets = {
    'your_preset': {
        'ema_short': 10,
        'ema_long': 30,
        # ... more parameters
    }
}
```

## 🐛 Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Make sure you're in the project root
pip install -r requirements.txt
```

#### Streamlit won't start
```bash
# Check if port 8501 is available
streamlit run app.py --server.port 8502
```

#### Authentication not working
```bash
# Delete users.json and restart
rm services/web-ui/users.json
streamlit run services/web-ui/app.py
```

#### Charts not displaying
```bash
# Ensure plotly is installed
pip install --upgrade plotly
```

### Debug Mode
Enable detailed logging:
```bash
streamlit run app.py --logger.level debug
```

## 📸 Screenshots

### Login Page
![Login Page](../../docs/screenshots/login.png)
*Secure authentication with registration support*

### Strategy Builder
![Strategy Builder](../../docs/screenshots/strategy_builder.png)
*Drag-and-drop interface for building custom strategies*

### Interactive Controls
![Interactive Controls](../../docs/screenshots/interactive_controls.png)
*Real-time parameter adjustment with sliders*

### Live Charts
![Live Charts](../../docs/screenshots/live_charts.png)
*Interactive Plotly charts with technical indicators*

### Monitoring Dashboard
![Monitoring Dashboard](../../docs/screenshots/monitoring.png)
*Real-time performance and system monitoring*

## 🎥 Video Tutorials

Coming soon! Check the main repository for tutorial videos and GIFs.

## 🔄 Updates & Roadmap

### Current Version: 1.0.0

### Planned Features
- [ ] Multi-user collaboration
- [ ] Real-time notifications
- [ ] Mobile-responsive design
- [ ] Dark/Light theme toggle
- [ ] Advanced charting tools
- [ ] Strategy marketplace
- [ ] AI-powered suggestions
- [ ] Export to Excel/PDF
- [ ] Webhook integrations
- [ ] API access

## 📞 Support

For issues specific to the Web UI:
1. Check this README
2. Review troubleshooting section
3. Check main project README
4. Open an issue on GitHub

## 📄 License

Same as main project - see LICENSE file in project root.

---

**Last Updated**: October 2025
**Version**: 1.0.0
**Maintainer**: TradPal Team
