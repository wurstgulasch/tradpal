# Web UI User Guide - Step by Step

This guide walks you through using the TradPal Web UI, from first login to creating and testing your first strategy.

## 📋 Table of Contents

1. [First Time Setup](#first-time-setup)
2. [Login and Navigation](#login-and-navigation)
3. [Building Your First Strategy](#building-your-first-strategy)
4. [Adjusting Parameters Interactively](#adjusting-parameters-interactively)
5. [Analyzing Charts](#analyzing-charts)
6. [Monitoring Performance](#monitoring-performance)
7. [Advanced Tips](#advanced-tips)

---

## 1. First Time Setup

### Step 1.1: Install Dependencies

```bash
# Navigate to the project directory
cd /path/to/tradpal_indicator

# Install required packages
pip install streamlit plotly flask flask-login werkzeug

# Or install all project dependencies
pip install -r requirements.txt
```

### Step 1.2: Launch the Web UI

```bash
# Navigate to web-ui directory
cd services/web-ui

# Start the application
streamlit run app.py

# Alternative: Use the quick start script
./start_web_ui.sh
```

### Step 1.3: Access the Interface

Open your web browser and navigate to:
```
http://localhost:8501
```

You should see the login page.

---

## 2. Login and Navigation

### Step 2.1: Login for First Time

On the login page:
- **Username**: `admin`
- **Password**: `admin123`
- Click **"🔑 Login"**

> ⚠️ **Security Note**: Change this password immediately after first login!

### Step 2.2: Create a New User (Optional)

If you want to create a new user account:
1. Click the **"Register"** tab
2. Enter a username (e.g., "trader1")
3. Enter a password (minimum 6 characters)
4. Confirm the password
5. Click **"✨ Register"**
6. Switch back to **"Login"** tab and sign in

### Step 2.3: Navigate the Interface

Once logged in, you'll see the main dashboard with:

**Left Sidebar:**
- 📊 Dashboard - Overview and quick actions
- 🎨 Strategy Builder - Create custom strategies
- ⚙️ Interactive Controls - Adjust parameters
- 📈 Live Charts - View real-time charts
- 🔧 System Monitor - Check system health

**Quick Settings (Sidebar Bottom):**
- Theme selection
- Auto-refresh rate
- User profile
- Logout button

---

## 3. Building Your First Strategy

### Step 3.1: Open Strategy Builder

1. Click **"🎨 Strategy Builder"** in the sidebar
2. You'll see two panels:
   - Left: Available Indicators
   - Right: Your Strategy Canvas

### Step 3.2: Add Indicators

Let's create a simple trend-following strategy:

1. **Add EMA (Exponential Moving Average)**
   - Expand the **"EMA - Exponential Moving Average"** section
   - Click **"➕ Add EMA"**
   - The indicator appears on the right panel
   - Adjust sliders:
     - Short Period: `9` (for fast response)
     - Long Period: `21` (for trend confirmation)

2. **Add RSI (Relative Strength Index)**
   - Expand **"RSI - Relative Strength Index"**
   - Click **"➕ Add RSI"**
   - Adjust sliders:
     - Period: `14` (standard setting)
     - Oversold: `30` (buy zone)
     - Overbought: `70` (sell zone)

3. **Add ATR (Average True Range)**
   - Expand **"ATR - Average True Range"**
   - Click **"➕ Add ATR"**
   - Adjust slider:
     - Period: `14` (for volatility measurement)

### Step 3.3: Configure Strategy

1. **Name Your Strategy**
   - In the strategy canvas, find the "Strategy Name" field
   - Enter: `My First Trend Strategy`

2. **Review Configuration**
   - Scroll down to see the JSON summary
   - Verify all indicators and parameters are correct

### Step 3.4: Save and Test

1. Click **"💾 Save Strategy"** at the bottom
2. You should see: "Strategy saved successfully!"
3. Click **"🧪 Test Strategy"** to run a backtest
4. Review the test results:
   - Win Rate
   - Total Trades
   - Profit Factor

### Step 3.5: Try a Preset (Alternative)

Instead of building from scratch, you can load a preset:

1. In the left panel, find **"⭐ Preset Strategies"**
2. Click one of the presets:
   - **📈 Trend Following**: Good for trending markets
   - **📉 Mean Reversion**: Good for ranging markets
   - **⚡ Scalping**: Good for quick trades
3. The preset loads automatically with optimized parameters
4. Customize as needed

---

## 4. Adjusting Parameters Interactively

### Step 4.1: Open Interactive Controls

1. Click **"⚙️ Interactive Controls"** in the sidebar
2. You'll see tabbed interface with parameter controls

### Step 4.2: Select Timeframe

At the top:
1. Select your trading timeframe from the dropdown:
   - `1m` - For scalping (1-minute charts)
   - `5m` - For day trading (5-minute charts)
   - `1h` - For swing trading (1-hour charts)
   - `1d` - For position trading (daily charts)
2. Parameters automatically adjust to recommended values for that timeframe

### Step 4.3: Adjust Trend Indicators

Click the **"📈 Trend Indicators"** tab:

**EMA Settings:**
1. Drag the **"EMA Short Period"** slider
   - Lower values (5-12) = faster response, more signals
   - Higher values (20-30) = slower response, fewer signals
2. Drag the **"EMA Long Period"** slider
   - Should be 2-3x the short period
3. Watch the validation message:
   - ✅ Green = Good configuration
   - ⚠️ Yellow = Warning (short >= long)

**Bollinger Bands:**
1. Adjust **"BB Period"** (10-50)
   - Standard is 20
2. Adjust **"BB Standard Deviation"** (1.0-3.0)
   - 2.0 captures ~95% of price action
   - Higher = wider bands

### Step 4.4: Tune Momentum Indicators

Click the **"📊 Momentum"** tab:

**RSI Configuration:**
1. Set **"RSI Period"** (5-30)
   - 14 is standard
   - Lower = more sensitive
2. Set **"RSI Oversold Level"** (20-40)
   - 30 is conservative
   - 20 is aggressive
3. Set **"RSI Overbought Level"** (60-80)
   - 70 is conservative
   - 80 is aggressive

Watch the RSI level indicators below the sliders to see your configuration visually.

**ADX Settings:**
1. **"ADX Period"**: 7-30 (14 standard)
2. **"ADX Threshold"**: 20-40
   - 25+ indicates strong trend
   - Below 25 = weak/ranging market

### Step 4.5: Configure Risk Management

Click the **"💰 Risk Management"** tab:

**Stop Loss & Take Profit:**
1. **"Stop Loss Multiplier"**: 0.5-3.0x ATR
   - 1.0 = moderate risk
   - 0.5 = tight stop (more stop-outs)
   - 2.0 = wide stop (ride through volatility)

2. **"Take Profit Multiplier"**: 1.0-5.0x ATR
   - Should be at least 2x your stop loss
   - 2.0 = 2:1 risk/reward
   - 3.0 = 3:1 risk/reward (better)

3. Watch the **Risk/Reward Ratio** metric:
   - ✅ Green if >= 2.0
   - ⚠️ Yellow if < 2.0

**Leverage:**
1. **"Maximum Leverage"**: 1-20x
   - 1-3x = Conservative (✅ Green indicator)
   - 5-10x = Moderate (⚠️ Yellow indicator)
   - 10-20x = Aggressive (❌ Red warning)

### Step 4.6: Apply and Save

1. Click **"✅ Apply Settings"** to use these parameters
2. Click **"💾 Save as Default"** to save for this timeframe
3. Click **"🧪 Test with Backtest"** to validate performance

### Step 4.7: Use Quick Presets

In the **"🎯 Advanced"** tab:

- **🏃 Scalping**: Fast indicators, tight stops (1m-5m timeframes)
- **📈 Trend Following**: Slower indicators, wide stops (1h-1d timeframes)
- **💰 Conservative**: Very slow indicators, very wide stops (4h-1d timeframes)

Click any preset to instantly apply its configuration.

---

## 5. Analyzing Charts

### Step 5.1: Open Live Charts

1. Click **"📈 Live Charts"** in the sidebar
2. You'll see interactive charts with controls at the top

### Step 5.2: Configure Chart Display

**Basic Settings:**
1. **Symbol**: Select trading pair (BTC/USDT, ETH/USDT, EUR/USD)
2. **Timeframe**: Choose chart period (1m, 5m, 15m, 1h, 4h, 1d)
3. **Chart Type**: 
   - Candlestick (recommended)
   - Line (simple)
   - OHLC (traditional bars)

**Indicator Toggles:**
Check/uncheck to show/hide:
- ✅ **EMA**: Moving average lines
- ✅ **RSI**: Momentum panel below chart
- ✅ **Bollinger Bands**: Volatility bands
- ✅ **Volume**: Volume bars panel
- ✅ **Signals**: Buy/Sell markers

### Step 5.3: Interact with Charts

**Navigation:**
- **Zoom In**: Click and drag over area to zoom
- **Pan**: Hold Shift and drag to move view
- **Reset**: Double-click anywhere to reset zoom
- **Hover**: Move mouse over candles to see details

**Reading the Chart:**
- **Green Candles**: Price went up (close > open)
- **Red Candles**: Price went down (close < open)
- **🟢 Green Triangles**: Buy signals (below price)
- **🔴 Red Triangles**: Sell signals (above price)
- **Orange/Purple Lines**: Fast/Slow EMA
- **Gray Dashed**: Bollinger Bands

### Step 5.4: Analyze Multiple Panels

**Main Chart Panel:**
- Price action with candles
- EMA lines (if enabled)
- Bollinger Bands (if enabled)
- Buy/Sell signal markers

**RSI Panel (Below):**
- Blue line showing RSI value (0-100)
- Red line at 70 (overbought)
- Green line at 30 (oversold)
- Gray line at 50 (neutral)

**Volume Panel (Bottom):**
- Green bars = price went up
- Red bars = price went down
- Height = volume magnitude

### Step 5.5: Check Signal Strength

On the right side, you'll see:

**Signal Strength Gauge:**
- Shows current signal quality (0-100)
- Green zone (70-100) = Strong signal
- Yellow zone (30-70) = Moderate signal
- Red zone (0-30) = Weak signal

**Recent Signals Table:**
- Last 10 signals with timestamps
- Signal type (BUY/SELL)
- Signal strength rating

### Step 5.6: Auto-Refresh

1. Check the **"Auto Refresh"** box to enable live updates
2. Adjust refresh rate in sidebar Quick Settings (5s, 10s, 30s, 1m)
3. Uncheck to pause updates

---

## 6. Monitoring Performance

### Step 6.1: Open System Monitor

1. Click **"🔧 System Monitor"** in the sidebar
2. Dashboard loads with real-time metrics

### Step 6.2: Review Key Metrics

**Top Row (Large Metrics):**
- **Win Rate**: % of profitable trades
  - Target: >60% (indicated by color)
- **Total Return**: Overall % gain/loss
- **Sharpe Ratio**: Risk-adjusted performance
  - Target: >1.0 (color-coded)
- **Max Drawdown**: Largest loss from peak
  - Limit: <20% (color-coded)

**Detail Metrics:**
- Total Trades executed
- Profit Factor (wins/losses ratio)
- Average Win/Loss sizes
- Largest Win/Loss trades

### Step 6.3: View Performance Charts

**Equity Curve:**
- Line chart showing portfolio value over time
- Upward slope = profitable
- Downward slope = losing

**Win/Loss Distribution:**
- Histogram showing trade outcome sizes
- Green = wins, Red = losses
- Width shows frequency

**Monthly Returns:**
- Bar chart of performance by month
- Green bars = profitable months
- Red bars = losing months

### Step 6.4: Check System Health

**Component Status:**
Look for status indicators:
- ✅ Green = Healthy
- ⚠️ Yellow = Warning
- ❌ Red = Error

**Components Monitored:**
- Data Fetcher (API connection)
- Indicators (calculation engine)
- Signal Generator (signal production)
- Backtester (testing availability)
- Database (cache status)

### Step 6.5: Monitor Alerts

**Active Alerts Section:**
- 🚨 Critical = Red (immediate attention)
- ⚠️ Warning = Yellow (monitor)
- ℹ️ Info = Blue (informational)

**Alert Types:**
- Win rate dropped below threshold
- Drawdown exceeded limit
- Sharpe ratio too low
- System component issues

### Step 6.6: Track Resources

**System Resources Panel:**
- CPU Usage
- Memory consumption
- Disk space available
- API rate limits

---

## 7. Advanced Tips

### Tip 1: Save Multiple Strategy Configurations

Create different strategies for different market conditions:

1. Build "Trending Market Strategy"
   - Strong trend indicators (EMA, ADX)
   - Save as "Trend_Strategy.json"

2. Build "Ranging Market Strategy"
   - Mean reversion indicators (BB, RSI)
   - Save as "Range_Strategy.json"

3. Switch between them based on current market

### Tip 2: Optimize Parameters by Timeframe

Different timeframes need different settings:

**1-Minute Charts (Scalping):**
- Fast EMAs (5/13)
- Short RSI period (7)
- Tight stops (0.8x ATR)
- High leverage (10-15x)

**1-Hour Charts (Day Trading):**
- Medium EMAs (9/21)
- Standard RSI (14)
- Moderate stops (1.5x ATR)
- Moderate leverage (5-8x)

**1-Day Charts (Position Trading):**
- Slow EMAs (20/50)
- Standard RSI (14)
- Wide stops (2.0x ATR)
- Low leverage (2-3x)

### Tip 3: Use Risk/Reward Ratio

Always aim for at least 2:1 risk/reward:
- If Stop Loss = 1% risk
- Take Profit should be 2% gain (or more)

Formula: TP Multiplier / SL Multiplier >= 2.0

### Tip 4: Backtest Before Live Trading

Before using any strategy live:
1. Build strategy in Strategy Builder
2. Test with historical data
3. Review Win Rate (target: >60%)
4. Review Profit Factor (target: >1.5)
5. Review Max Drawdown (target: <20%)
6. Adjust parameters if needed
7. Re-test until satisfied

### Tip 5: Combine Multiple Indicators

Best strategies use confirmation from multiple sources:
- **Trend**: EMA crossover
- **Momentum**: RSI overbought/oversold
- **Volatility**: Bollinger Bands
- **Strength**: ADX above threshold
- **Risk**: ATR for position sizing

### Tip 6: Export Configurations

Save your configurations:
1. In Interactive Controls, go to Advanced tab
2. Click "📤 Export Config"
3. Copy the JSON
4. Save to file for backup
5. Share with team or use in automation

### Tip 7: Monitor During Live Trading

Keep the Monitoring Dashboard open:
- Watch Win Rate in real-time
- Check for alert notifications
- Monitor system health
- Track equity curve
- Adjust if performance degrades

### Tip 8: Use Auto-Refresh Wisely

For active trading:
- Live Charts: 10-30s refresh
- Monitoring: 30s-1m refresh

For analysis:
- Disable auto-refresh to reduce load
- Manually refresh when needed

### Tip 9: Secure Your Account

Important security practices:
1. Change default admin password immediately
2. Use strong passwords (12+ characters)
3. Create separate user accounts for each person
4. Log out when not using the system
5. Back up your strategies and user data
6. Don't share login credentials

### Tip 10: Keyboard Shortcuts

Speed up your workflow:
- **Ctrl+R**: Refresh browser (reload data)
- **F11**: Full screen mode
- **Double-click chart**: Reset zoom
- **Shift+Drag**: Pan chart view

---

## 🆘 Need Help?

If you encounter issues:

1. **Check Documentation**
   - [Web UI README](README.md)
   - [Features Documentation](FEATURES.md)
   - [Main Project README](../../README.md)

2. **Run Component Tests**
   ```bash
   cd services/web-ui
   python test_components.py
   ```

3. **Enable Debug Mode**
   ```bash
   streamlit run app.py --logger.level debug
   ```

4. **Common Solutions**
   - Clear browser cache
   - Restart Streamlit
   - Reinstall dependencies
   - Check port availability

5. **Get Support**
   - Open GitHub issue
   - Include error messages
   - Describe steps to reproduce

---

## 📚 Learn More

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Chart Types](https://plotly.com/python/)
- [Technical Analysis Basics](https://www.investopedia.com/technical-analysis-4689657)
- [Risk Management Guide](https://www.investopedia.com/trading/risk-management/)

---

**Happy Trading! 📈**

*Remember: Past performance does not guarantee future results. Always use proper risk management and never risk more than you can afford to lose.*

---

**User Guide Version**: 1.0.0  
**Last Updated**: October 2025
