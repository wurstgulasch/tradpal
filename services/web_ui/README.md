# TradPal Web Interface

A modern, comprehensive web interface for the TradPal advanced trading system built with Streamlit.

## 🚀 Features

### 📊 Dashboard Overview
- **Real-time Portfolio Metrics**: Total value, P&L, Sharpe ratio, volatility
- **Asset Allocation Visualization**: Interactive pie charts and bar charts
- **Risk Metrics Display**: VaR, maximum drawdown, diversification ratio
- **Performance Tracking**: Daily/weekly/monthly returns

### 💼 Portfolio Management
- **Add/Remove Assets**: Dynamic portfolio composition
- **Rebalance Portfolio**: Automatic rebalancing with different optimization methods
  - Modern Portfolio Theory (MPT)
  - Risk Parity optimization
  - Minimum Variance optimization
- **Set Target Weights**: Custom allocation percentages
- **Position Monitoring**: Real-time position tracking with P&L

### 📈 Trading Interface
- **Order Execution**: Buy/sell orders for individual assets
- **Position Sizing**: Risk-based and ML-enhanced position sizing
- **Trade History**: Complete trading history and performance
- **Risk Controls**: Configurable risk limits and stop-losses

### 🤖 ML Analytics & Predictions
- **Model Predictions**: Real-time ML predictions for each asset
- **Confidence Scores**: Prediction confidence visualization
- **Market Regime Detection**: Bull/bear market classification
- **Feature Importance**: Understanding model decisions
- **Backtesting Results**: Historical performance analysis

### ⚠️ Risk Management Dashboard
- **Real-time Risk Monitoring**: Portfolio risk metrics
- **Stress Testing**: Scenario analysis and risk simulation
- **Correlation Matrix**: Asset correlation visualization
- **Risk Contribution Analysis**: Individual asset risk breakdown
- **VaR Calculations**: Value at Risk computations

### 📉 Backtesting & Analysis
- **Strategy Backtesting**: Test trading strategies on historical data
- **Performance Comparison**: Compare different strategies
- **Drawdown Analysis**: Maximum drawdown and recovery analysis
- **Risk-Adjusted Returns**: Sharpe ratio, Sortino ratio, etc.
- **Detailed Reports**: Comprehensive backtesting reports

### ⚙️ Settings & Configuration
- **ML Model Configuration**: Adjust model parameters and features
- **Risk Management Rules**: Set risk limits and thresholds
- **Portfolio Optimization**: Configure optimization constraints
- **API Management**: Secure API key configuration
- **System Settings**: General application settings

## 🛠️ Technical Stack

- **Frontend**: Streamlit with custom CSS
- **Visualization**: Plotly for interactive charts
- **Backend Integration**: Direct integration with Python trading modules
- **Real-time Updates**: WebSocket support for live data
- **Security**: Encrypted API key storage
- **Responsive Design**: Mobile and desktop optimized

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TradPal trading system installed
- Required dependencies (see requirements.txt)

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Web Interface**:
   ```bash
   cd /path/to/tradpal
   PYTHONPATH=/path/to/tradpal streamlit run services/web_ui/app.py
   ```

3. **Access the Interface**:
   - Open your browser to `http://localhost:8501`
   - Default credentials: admin/admin (if authentication enabled)

### Configuration

The web interface automatically detects and integrates with:
- `src/multi_asset_portfolio.py` - Portfolio management
- `src/advanced_ml_predictor.py` - ML predictions
- `src/backtester.py` - Backtesting engine
- `config/settings.py` - System configuration

## 📱 Interface Overview

### Navigation
- **Sidebar Navigation**: Easy access to all modules
- **Real-time Status**: System health and connection status
- **Quick Actions**: Fast access to common operations

### Dark Theme
- **Trading-Optimized**: Dark theme designed for long trading sessions
- **Color Coding**: Green for profits, red for losses, orange for warnings
- **High Contrast**: Clear visibility of important information

### Responsive Design
- **Desktop First**: Optimized for trading workstations
- **Mobile Support**: Responsive design for mobile monitoring
- **Tablet Compatible**: Works on tablets and touch devices

## 🔧 Advanced Features

### Real-time Data Updates
- **WebSocket Integration**: Live price feeds
- **Auto-refresh**: Configurable update intervals
- **Background Processing**: Non-blocking data updates

### ML Integration
- **Ensemble Models**: Multiple ML models for robust predictions
- **Feature Engineering**: Advanced technical indicators
- **Model Interpretability**: Explainable AI features

### Risk Analytics
- **Monte Carlo Simulation**: Portfolio stress testing
- **Scenario Analysis**: What-if analysis
- **Risk Parity**: Equal risk contribution optimization

### Performance Analytics
- **Benchmarking**: Compare against market indices
- **Attribution Analysis**: Understand performance drivers
- **Risk Decomposition**: Break down portfolio risk

## 🔒 Security Features

- **API Key Encryption**: Secure storage of exchange API keys
- **Session Management**: Secure user sessions
- **Input Validation**: Comprehensive input sanitization
- **Audit Logging**: Complete action logging

## 📊 API Integration

The web interface integrates with multiple APIs:
- **Exchange APIs**: Live trading via CCXT
- **Market Data**: Real-time price feeds
- **ML Services**: Prediction APIs
- **Database**: Portfolio and trade storage

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH is set correctly
2. **ML Models Not Loading**: Check PyTorch/TensorFlow installation
3. **Database Connection**: Verify database configuration
4. **WebSocket Issues**: Check firewall and network settings

### Logs
- **Application Logs**: `logs/tradpal_web.log`
- **Error Logs**: `logs/errors.log`
- **Audit Logs**: `logs/audit.log`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- **Documentation**: See `/docs` directory
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

**TradPal Web Interface v2.0** - Advanced Trading Made Simple 🚀