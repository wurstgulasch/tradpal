# Web UI Features Overview

## 🎯 Complete Feature List

### 1. 🔐 Authentication & Security
- **Secure Login System**
  - Password hashing with Werkzeug (bcrypt)
  - Session management with Streamlit
  - Default admin account (username: admin, password: admin123)
  
- **User Management**
  - User registration with password validation
  - Role-based access control (admin/user)
  - User database stored in JSON format
  - Password change functionality
  - Admin user management tools

- **Security Features**
  - Encrypted password storage
  - Session timeout handling
  - Protection against common vulnerabilities
  - Secure cookie handling

### 2. 🎨 Strategy Builder

#### Available Indicators
| Indicator | Parameters | Description |
|-----------|-----------|-------------|
| **EMA** | Short Period (5-50), Long Period (10-200) | Exponential Moving Averages for trend |
| **RSI** | Period (5-30), Oversold (20-40), Overbought (60-80) | Momentum oscillator |
| **BB** | Period (10-50), Std Dev (1-3) | Volatility bands |
| **ATR** | Period (7-30) | Volatility measurement |
| **ADX** | Period (7-30), Threshold (20-40) | Trend strength |
| **MACD** | Fast (5-20), Slow (15-40), Signal (5-15) | Trend and momentum |

#### Strategy Builder Features
- **Drag-and-Drop Interface**: Add indicators with a single click
- **Real-Time Configuration**: Adjust parameters with interactive sliders
- **Visual Feedback**: Instant validation of parameter settings
- **Strategy Presets**:
  - 📈 **Trend Following**: EMA + ADX + ATR (for trending markets)
  - 📉 **Mean Reversion**: BB + RSI + ATR (for ranging markets)
  - ⚡ **Scalping**: Fast EMA + Short RSI + ATR (for quick trades)
- **Save/Load**: Persist custom strategies as JSON files
- **Strategy Testing**: Integrated backtesting with performance metrics
- **Strategy Summary**: JSON export of complete configuration

### 3. ⚙️ Interactive Controls

#### Parameter Categories

**🔹 Trend Indicators Tab**
- EMA Short Period: 5-50 (default: 9)
- EMA Long Period: 10-200 (default: 21)
- Bollinger Bands Period: 10-50 (default: 20)
- BB Standard Deviation: 1.0-3.0 (default: 2.0)
- Real-time EMA cross validation
- Visual BB width indicator

**🔹 Momentum Tab**
- RSI Period: 5-30 (default: 14)
- RSI Oversold Level: 20-40 (default: 30)
- RSI Overbought Level: 60-80 (default: 70)
- ADX Period: 7-30 (default: 14)
- ADX Trend Threshold: 20-40 (default: 25)
- Visual RSI level indicators

**🔹 Risk Management Tab**
- ATR Period: 7-30 (default: 14)
- Stop Loss Multiplier: 0.5-3.0x ATR (default: 1.0)
- Take Profit Multiplier: 1.0-5.0x ATR (default: 2.0)
- Maximum Leverage: 1-20x (default: 10)
- Risk/Reward Ratio Calculator
- Leverage risk warnings

**🔹 Advanced Settings Tab**
- Quick preset configurations
- Configuration export to JSON
- Configuration import from JSON
- Parameter reset functionality

#### Timeframe Support
- 1-minute (1m) - Scalping parameters
- 5-minute (5m) - Day trading parameters
- 15-minute (15m) - Swing trading parameters
- 1-hour (1h) - Position trading parameters
- 4-hour (4h) - Long-term parameters
- 1-day (1d) - Investment parameters

#### Preset Configurations

| Preset | EMA | RSI | SL/TP | Leverage | Use Case |
|--------|-----|-----|-------|----------|----------|
| **Scalping** | 5/13 | 7 (25/75) | 0.8x/1.5x | 15x | Fast trades, tight stops |
| **Trend** | 20/50 | 14 (40/60) | 1.5x/3.0x | 5x | Follow strong trends |
| **Conservative** | 21/55 | 14 (35/65) | 2.0x/4.0x | 3x | Low risk, wide stops |

### 4. 📈 Live Charts with Plotly

#### Chart Types
- **Candlestick**: Traditional OHLC candles with color coding
- **Line Chart**: Simple close price visualization
- **OHLC**: Open-High-Low-Close bars

#### Interactive Features
- **Zoom & Pan**: Click and drag to zoom, shift+drag to pan
- **Hover Details**: See exact values at any point
- **Crosshair**: Synchronized across all panels
- **Reset View**: Double-click to reset zoom
- **Auto-Refresh**: Optional real-time updates (5s, 10s, 30s, 1m)

#### Indicator Overlays
✅ **EMA Lines**
- Fast EMA (orange line)
- Slow EMA (purple line)
- Crossover signals highlighted

✅ **Bollinger Bands**
- Upper band (dashed gray)
- Lower band (dashed gray)
- Shaded area between bands
- Middle band (20-period SMA)

✅ **RSI Panel**
- RSI line (blue)
- Overbought level at 70 (red dashed)
- Oversold level at 30 (green dashed)
- Neutral level at 50 (gray dotted)

✅ **Volume Panel**
- Color-coded bars (green=up, red=down)
- Volume spikes highlighted
- Synchronized with price action

✅ **Signal Markers**
- 🟢 Buy Signals: Green triangles pointing up
- 🔴 Sell Signals: Red triangles pointing down
- Positioned below/above candles
- Clickable for details

#### Additional Visualizations
- **Signal Strength Gauge**: Real-time indicator of signal quality (0-100)
- **Indicator Comparison**: Dual-axis charts for RSI vs ADX
- **Recent Signals Table**: Last 10 signals with timestamp and strength
- **Price Summary**: Current price, 24h high/low, volume

#### Symbol Support
- Cryptocurrencies: BTC/USDT, ETH/USDT, etc.
- Forex pairs: EUR/USD, GBP/USD, etc.
- Stocks: AAPL, TSLA, etc. (if data available)

### 5. 📊 Monitoring Dashboard

#### Key Performance Metrics
- **Win Rate**: Percentage of profitable trades (target: >60%)
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns (target: >1.0)
- **Max Drawdown**: Largest peak-to-trough decline (limit: <20%)

#### Detailed Statistics
- Total Trades Executed
- Profit Factor (total wins / total losses)
- Average Win/Loss sizes
- Largest Win/Loss trades
- Current equity value
- Active positions

#### Performance Charts
1. **Equity Curve**: Portfolio value over time
2. **Win/Loss Distribution**: Histogram of trade outcomes
3. **Monthly Returns**: Bar chart of monthly performance
4. **Drawdown Chart**: Underwater equity curve

#### System Health Monitoring
- ✅ **Data Fetcher**: API connectivity status
- ✅ **Indicators**: Calculation engine status
- ✅ **Signal Generator**: Signal production health
- ✅ **Backtester**: Testing engine availability
- ✅ **Database/Cache**: Storage system status

#### Alert System
- 🚨 **Critical Alerts**: Drawdown exceeds threshold
- ⚠️ **Warnings**: Win rate drops below target
- ℹ️ **Info**: System status changes
- Alert history with timestamps
- Configurable alert thresholds

#### System Resources
- CPU Usage monitoring
- Memory consumption
- Disk space availability
- API rate limit tracking

### 6. 🎯 Navigation & UX

#### Sidebar Navigation
- 📊 Dashboard - Main overview
- 🎨 Strategy Builder - Create strategies
- ⚙️ Interactive Controls - Tune parameters
- 📈 Live Charts - Real-time visualization
- 🔧 System Monitor - Health & performance

#### Quick Settings (Sidebar)
- Theme selection (Dark/Light)
- Auto-refresh rate (5s to 1m)
- User profile display
- Logout button

#### User Experience
- **Responsive Design**: Adapts to screen size
- **Wide Layout**: Maximizes screen real estate
- **Quick Actions**: One-click access to common tasks
- **Persistent State**: Settings saved across sessions
- **Error Messages**: Clear, actionable error descriptions
- **Loading Indicators**: Progress feedback for long operations

## 📊 Data Flow

```
User Input (Web UI)
    ↓
Strategy Configuration
    ↓
Parameter Validation
    ↓
Indicator Calculation (src/indicators.py)
    ↓
Signal Generation (src/signal_generator.py)
    ↓
Risk Management (src/signal_generator.py)
    ↓
Visualization (Plotly Charts)
    ↓
Performance Tracking (Monitoring Dashboard)
```

## 🔒 Security Best Practices

1. **Change Default Credentials**: Immediately update admin password
2. **User Database**: Keep `users.json` in `.gitignore`
3. **HTTPS in Production**: Use SSL/TLS for web traffic
4. **Regular Backups**: Backup user and strategy data
5. **Access Control**: Use firewall rules to limit access
6. **Session Management**: Logout after use
7. **Password Policies**: Enforce strong passwords (min 6 chars, recommend 12+)

## 🎓 Usage Examples

### Example 1: Building a Scalping Strategy
1. Navigate to **Strategy Builder**
2. Add **EMA** indicator
   - Set Short Period: 5
   - Set Long Period: 13
3. Add **RSI** indicator
   - Set Period: 7
   - Set Oversold: 25
   - Set Overbought: 75
4. Add **ATR** indicator
   - Set Period: 7
5. Click **"💾 Save Strategy"**
6. Click **"🧪 Test Strategy"** to backtest

### Example 2: Adjusting Risk Management
1. Navigate to **Interactive Controls**
2. Select **Risk Management** tab
3. Adjust **Stop Loss Multiplier** to 1.5
4. Adjust **Take Profit Multiplier** to 3.0
5. Note the Risk/Reward Ratio: 2.0
6. Click **"✅ Apply Settings"**
7. Optionally **"💾 Save as Default"**

### Example 3: Analyzing Live Charts
1. Navigate to **Live Charts**
2. Select symbol: BTC/USDT
3. Select timeframe: 5m
4. Enable indicators: EMA, RSI, BB
5. Enable **Auto Refresh** for live updates
6. Use mouse to zoom on areas of interest
7. Hover over candles for detailed info
8. Check **Signal Strength Gauge** for entry timing

## 🔄 Workflow Integration

The Web UI seamlessly integrates with existing TradPal features:

- **Backtesting**: Test strategies built in UI using core backtesting engine
- **Live Trading**: Apply UI-configured parameters to live monitoring
- **Strategy Export**: Save strategies as JSON for use in automated systems
- **Performance Tracking**: Monitor real trading performance in dashboard
- **Configuration Sync**: Settings apply across all system components

## 📱 Future Enhancements

Planned features for upcoming releases:

- [ ] Mobile-responsive design
- [ ] Dark/Light theme toggle with persistence
- [ ] Advanced charting tools (trend lines, Fibonacci retracements)
- [ ] Strategy marketplace (share and download strategies)
- [ ] AI-powered parameter suggestions
- [ ] Real-time notifications (browser push, email, SMS)
- [ ] Multi-user collaboration features
- [ ] Advanced portfolio analytics
- [ ] Webhook integrations UI
- [ ] Export reports to PDF/Excel
- [ ] API access for programmatic control

## 🆘 Troubleshooting

### Common Issues

**Issue**: "Module not found" errors
- **Solution**: Install dependencies: `pip install streamlit plotly flask flask-login werkzeug`

**Issue**: Streamlit won't start
- **Solution**: Check port availability: `streamlit run app.py --server.port 8502`

**Issue**: Authentication not working
- **Solution**: Delete `users.json` and restart to recreate default admin

**Issue**: Charts not displaying
- **Solution**: Update plotly: `pip install --upgrade plotly`

**Issue**: Session state errors
- **Solution**: Clear browser cache and restart Streamlit

### Getting Help

1. Check the [Web UI README](README.md)
2. Review [Main Project README](../../README.md)
3. Enable debug logging: `streamlit run app.py --logger.level debug`
4. Open an issue on GitHub with error details

---

**Documentation Version**: 1.0.0  
**Last Updated**: October 2025  
**Maintainer**: TradPal Indicator Team
