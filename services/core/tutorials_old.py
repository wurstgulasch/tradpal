"""
Interactive Tutorials for TradPal Indicator System

Streamlit-based tutorials for system setup, configuration, and usage.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import system components
try:
    from config.settings import SYMBOL, TIMEFRAME, CAPITAL
    from src.data_fetcher import get_data_fetcher
    from src.indicators import calculate_indicators
    from src.signal_generator import generate_signals, calculate_risk_management
    from src.backtester import Backtester, run_walk_forward_backtest
    from src.ml_predictor import LSTMSignalPredictor, is_lstm_available, is_shap_available
    from src.walk_forward_optimizer import get_walk_forward_optimizer
except ImportError as e:
    st.error(f"❌ Failed to import system components: {e}")
    st.stop()


class TutorialManager:
    """
    Manages interactive tutorials for the TradPal Indicator system.
    """
    
    def __init__(self):
        self.tutorials = {
            'getting_started': self.tutorial_getting_started,
            'basic_backtesting': self.tutorial_basic_backtesting,
            'advanced_backtesting': self.tutorial_advanced_backtesting,
            'ml_prediction': self.tutorial_ml_prediction,
            'walk_forward_analysis': self.tutorial_walk_forward_analysis,
            'system_configuration': self.tutorial_system_configuration,
            'performance_monitoring': self.tutorial_performance_monitoring
        }
    
    def run_tutorial(self, tutorial_name: str):
        """Run a specific tutorial."""
        if tutorial_name in self.tutorials:
            self.tutorials[tutorial_name]()
        else:
            st.error(f"❌ Tutorial '{tutorial_name}' not found")
    
    def tutorial_getting_started(self):
        """Getting started tutorial."""
        st.header("🚀 Getting Started with TradPal Indicator")
        st.markdown("""
        Welcome to TradPal Indicator! This tutorial will guide you through the basic setup and first steps.
        """)
        
        # System overview
        st.subheader("📋 System Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Core Features:**
            - Multi-timeframe technical analysis
            - Advanced signal generation
            - Risk management
            - Machine learning predictions
            - Walk-forward optimization
            - Real-time monitoring
            """)
        
        with col2:
            st.markdown("""
            **Supported Assets:**
            - Cryptocurrencies (BTC, ETH, etc.)
            - Forex pairs (EUR/USD, GBP/USD, etc.)
            - Stocks and commodities
            """)
        
        # Prerequisites check
        st.subheader("🔧 Prerequisites Check")
        self.check_prerequisites()
        
        # Quick start
        st.subheader("⚡ Quick Start")
        if st.button("�� Run Quick Demo", type="primary"):
            self.run_quick_demo()
    
    def tutorial_basic_backtesting(self):
        """Basic backtesting tutorial."""
        st.header("📊 Basic Backtesting Tutorial")
        st.markdown("""
        Learn how to perform basic backtesting with historical data and analyze trading performance.
        """)
        
        # Tutorial steps
        steps = [
            "📥 Load Historical Data",
            "📈 Calculate Indicators", 
            "🎯 Generate Signals",
            "💰 Run Backtest",
            "📊 Analyze Results"
        ]
        
        current_step = st.selectbox("Select Tutorial Step:", steps, key="basic_backtest_step")
        
        if current_step == steps[0]:
            self.step_load_data()
        elif current_step == steps[1]:
            self.step_calculate_indicators()
        elif current_step == steps[2]:
            self.step_generate_signals()
        elif current_step == steps[3]:
            self.step_run_backtest()
        elif current_step == steps[4]:
            self.step_analyze_results()
    
    def tutorial_walk_forward_analysis(self):
        """Walk-forward analysis tutorial."""
        st.header("🔄 Walk-Forward Analysis Tutorial")
        st.markdown("""
        Learn about time-series cross-validation and realistic strategy evaluation.
        """)
        
        # Walk-forward explanation
        st.subheader("📚 What is Walk-Forward Analysis?")
        st.markdown("""
        Walk-forward analysis is a time-series cross-validation technique that:
        
        1. **Training Window**: Uses historical data to optimize parameters
        2. **Testing Window**: Evaluates performance on future unseen data  
        3. **Rolling Windows**: Moves forward in time, expanding the training set
        4. **Out-of-Sample Testing**: Ensures realistic performance evaluation
        
        This prevents **overfitting** and provides more reliable strategy assessment.
        """)
        
        # Interactive demo
        if st.button("🎯 Run Walk-Forward Demo", key="wf_demo"):
            self.run_walk_forward_demo()
    
    def tutorial_ml_prediction(self):
        """ML prediction tutorial."""
        st.header("🤖 Machine Learning Prediction Tutorial")
        st.markdown("""
        Learn how to use advanced machine learning models for enhanced signal prediction.
        """)
        
        # Check ML availability
        lstm_available = is_lstm_available()
        shap_available = is_shap_available()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("LSTM Models", "✅ Available" if lstm_available else "❌ Not Available")
        with col2:
            st.metric("SHAP Explanations", "✅ Available" if shap_available else "❌ Not Available")
        
        if not lstm_available:
            st.warning("⚠️ LSTM models require TensorFlow. Install with: `pip install tensorflow>=2.13.0`")
            return
        
        # ML tutorial steps
        ml_steps = [
            "🧠 Train LSTM Model",
            "�� Make Predictions",
            "📋 Explain Predictions",
            "⚖️ Compare with Traditional Signals"
        ]
        
        selected_step = st.selectbox("Select ML Step:", ml_steps, key="ml_step")
        
        if selected_step == ml_steps[0]:
            self.ml_step_train_model()
        elif selected_step == ml_steps[1]:
            self.ml_step_make_predictions()
    
    def check_prerequisites(self):
        """Check system prerequisites."""
        checks = {
            "Python Version": sys.version_info >= (3, 10),
            "Required Packages": self.check_packages(),
            "Configuration Files": os.path.exists("config/settings.py"),
            "Data Directory": os.path.exists("output/"),
            "Log Directory": os.path.exists("logs/")
        }
        
        for check_name, status in checks.items():
            if status:
                st.success(f"✅ {check_name}")
            else:
                st.error(f"❌ {check_name}")
    
    def check_packages(self) -> bool:
        """Check if required packages are installed."""
        required_packages = [
            'pandas', 'numpy', 'streamlit', 'ccxt',
            'plotly', 'ta', 'scikit-learn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            st.warning(f"⚠️ Missing packages: {', '.join(missing_packages)}")
            return False
        
        return True
    
    def run_quick_demo(self):
        """Run a quick demonstration of the system."""
        with st.spinner("🚀 Running quick demo..."):
            try:
                # Create sample data
                dates = pd.date_range(start='2024-01-01', periods=500, freq='1h')
                np.random.seed(42)
                df = pd.DataFrame({
                    'timestamp': dates,
                    'open': 50000 + np.random.normal(0, 1000, 500),
                    'high': 51000 + np.random.normal(0, 1000, 500),
                    'low': 49000 + np.random.normal(0, 1000, 500),
                    'close': 50000 + np.random.normal(0, 1000, 500),
                    'volume': np.random.randint(1000, 10000, 500)
                })
                
                # Calculate indicators and signals
                df = calculate_indicators(df)
                df = generate_signals(df)
                df = calculate_risk_management(df)
                
                # Show results
                st.success("✅ Demo completed successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Points", len(df))
                with col2:
                    st.metric("Buy Signals", int(df['Buy_Signal'].sum()))
                with col3:
                    st.metric("Sell Signals", int(df['Sell_Signal'].sum()))
                
                # Show sample data
                st.subheader("📊 Sample Results")
                st.dataframe(df[['timestamp', 'close', 'EMA9', 'EMA21', 'RSI', 'Buy_Signal', 'Sell_Signal']].head(10))
                
            except Exception as e:
                st.error(f"❌ Demo failed: {e}")
    
    def step_load_data(self):
        """Step: Load historical data."""
        st.subheader("📥 Step 1: Load Historical Data")
        
        symbol = st.selectbox("Select Symbol:", ["BTC/USD", "ETH/USD", "EUR/USD"], key="load_symbol")
        timeframe = st.selectbox("Select Timeframe:", ["1m", "5m", "1h", "1d"], key="load_timeframe")
        limit = st.slider("Data Points:", 100, 5000, 1000, key="load_limit")
        
        if st.button("📥 Load Data", key="load_data_btn"):
            with st.spinner("Loading data..."):
                try:
                    fetcher = get_data_fetcher()
                    data = fetcher.fetch_historical_data(symbol, timeframe, limit=limit)
                    
                    if data is not None and not data.empty:
                        st.success(f"✅ Loaded {len(data)} data points")
                        st.dataframe(data.head())
                        st.session_state['tutorial_data'] = data
                    else:
                        st.error("❌ Failed to load data")
                        
                except Exception as e:
                    st.error(f"❌ Error loading data: {e}")
    
    def step_calculate_indicators(self):
        """Step: Calculate indicators."""
        st.subheader("📈 Step 2: Calculate Technical Indicators")
        
        if 'tutorial_data' not in st.session_state:
            st.warning("⚠️ Please complete Step 1 first")
            return
        
        if st.button("📈 Calculate Indicators", key="calc_indicators_btn"):
            with st.spinner("Calculating indicators..."):
                try:
                    data = st.session_state['tutorial_data']
                    data = calculate_indicators(data)
                    st.session_state['tutorial_data'] = data
                    
                    st.success("✅ Indicators calculated")
                    available_indicators = [col for col in data.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    st.write(f"**Calculated Indicators:** {', '.join(available_indicators[:5])}...")
                    
                    st.dataframe(data[['timestamp', 'close', 'EMA9', 'EMA21', 'RSI', 'BB_upper', 'BB_lower']].head())
                    
                except Exception as e:
                    st.error(f"❌ Error calculating indicators: {e}")
    
    def step_generate_signals(self):
        """Step: Generate signals."""
        st.subheader("🎯 Step 3: Generate Trading Signals")
        
        if 'tutorial_data' not in st.session_state:
            st.warning("⚠️ Please complete previous steps first")
            return
        
        if st.button("🎯 Generate Signals", key="generate_signals_btn"):
            with st.spinner("Generating signals..."):
                try:
                    data = st.session_state['tutorial_data']
                    data = generate_signals(data)
                    st.session_state['tutorial_data'] = data
                    
                    buy_signals = int(data['Buy_Signal'].sum())
                    sell_signals = int(data['Sell_Signal'].sum())
                    
                    st.success("✅ Signals generated")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Buy Signals", buy_signals)
                    with col2:
                        st.metric("Sell Signals", sell_signals)
                    
                    st.dataframe(data[['timestamp', 'close', 'Buy_Signal', 'Sell_Signal']].head(10))
                    
                except Exception as e:
                    st.error(f"❌ Error generating signals: {e}")
    
    def step_run_backtest(self):
        """Step: Run backtest."""
        st.subheader("💰 Step 4: Run Backtest")
        
        if 'tutorial_data' not in st.session_state:
            st.warning("⚠️ Please complete previous steps first")
            return
        
        initial_capital = st.number_input("Initial Capital:", 1000, 100000, 10000, key="backtest_capital")
        
        if st.button("💰 Run Backtest", key="run_backtest_btn"):
            with st.spinner("Running backtest..."):
                try:
                    data = st.session_state['tutorial_data']
                    
                    # Simple backtest simulation
                    backtester = Backtester(initial_capital=initial_capital)
                    results = backtester._calculate_metrics()
                    
                    st.success("✅ Backtest completed")
                    st.session_state['backtest_results'] = results
                    
                    # Show key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Return", f"{results.get('total_pnl', 0):.2f}")
                    with col2:
                        st.metric("Win Rate", f"{results.get('win_rate', 0):.1%}")
                    with col3:
                        st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2f}")
                    
                except Exception as e:
                    st.error(f"❌ Error running backtest: {e}")
    
    def step_analyze_results(self):
        """Step: Analyze results."""
        st.subheader("📊 Step 5: Analyze Results")
        
        if 'backtest_results' not in st.session_state:
            st.warning("⚠️ Please complete Step 4 first")
            return
        
        results = st.session_state['backtest_results']
        
        # Display detailed results
        st.subheader("📈 Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': list(results.keys()),
            'Value': [str(v) for v in results.values()]
        })
        st.dataframe(metrics_df)
        
        # Simple interpretation
        win_rate = results.get('win_rate', 0)
        total_return = results.get('total_pnl', 0)
        max_drawdown = results.get('max_drawdown', 0)
        
        if win_rate > 0.6 and total_return > 0:
            st.success("🎉 Excellent performance! Strategy shows strong potential.")
        elif win_rate > 0.5 and total_return > 0:
            st.info("👍 Good performance. Consider parameter optimization.")
        else:
            st.warning("⚠️ Strategy needs improvement. Consider different parameters.")
    
    def run_walk_forward_demo(self):
        """Run walk-forward analysis demo."""
        st.subheader("🎯 Walk-Forward Analysis Demo")
        
        with st.spinner("Running walk-forward analysis..."):
            try:
                # Create sample data
                dates = pd.date_range(start='2024-01-01', periods=1000, freq='1h')
                np.random.seed(42)
                df = pd.DataFrame({
                    'timestamp': dates,
                    'open': 50000 + np.random.normal(0, 1000, 1000),
                    'high': 51000 + np.random.normal(0, 1000, 1000),
                    'low': 49000 + np.random.normal(0, 1000, 1000),
                    'close': 50000 + np.random.normal(0, 1000, 1000),
                    'volume': np.random.randint(1000, 10000, 1000)
                })
                
                # Add trend
                trend = np.linspace(0, 3000, 1000)
                df['close'] = df['close'] + trend
                
                # Calculate indicators
                df = calculate_indicators(df)
                
                # Run walk-forward analysis
                results = run_walk_forward_backtest(
                    parameter_grid={
                        'ema_short': [9, 12],
                        'ema_long': [21, 26],
                        'rsi_overbought': [70],
                        'rsi_oversold': [30]
                    },
                    evaluation_metric='sharpe_ratio'
                )
                
                if 'error' not in results:
                    st.success("✅ Walk-forward analysis completed")
                    
                    # Show results
                    analysis = results.get('optimization_results', {})
                    final_backtest = results.get('final_backtest', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("OOS Performance", f"{analysis.get('average_oos_performance', 0):.3f}")
                    with col2:
                        st.metric("Robustness Ratio", f"{analysis.get('robustness_ratio', 0):.2%}")
                    with col3:
                        st.metric("Final Win Rate", f"{final_backtest.get('metrics', {}).get('win_rate', 0):.1%}")
                    
                    # Explain concepts
                    st.subheader("📚 Key Concepts")
                    st.markdown("""
                    - **In-Sample**: Performance on training data (often over-optimistic)
                    - **Out-of-Sample**: Performance on unseen future data (realistic)
                    - **Performance Decay**: Difference between IS and OOS performance
                    - **Robustness**: Consistency across different time periods
                    """)
                else:
                    st.error(f"❌ Walk-forward analysis failed: {results['error']}")
                    
            except Exception as e:
                st.error(f"❌ Demo failed: {e}")
    
    def ml_step_train_model(self):
        """ML step: Train LSTM model."""
        st.subheader("🧠 Train LSTM Model")
        
        if not is_lstm_available():
            st.error("❌ LSTM models not available. Please install TensorFlow.")
            return
        
        # Training parameters
        sequence_length = st.slider("Sequence Length:", 10, 100, 20, key="seq_length")
        epochs = st.slider("Training Epochs:", 1, 50, 5, key="train_epochs")
        batch_size = st.selectbox("Batch Size:", [8, 16, 32], key="batch_size")
        
        if st.button("🧠 Train Model", key="train_lstm_btn"):
            with st.spinner("Training LSTM model..."):
                try:
                    # Create sample data
                    dates = pd.date_range(start='2024-01-01', periods=500, freq='1h')
                    np.random.seed(42)
                    df = pd.DataFrame({
                        'timestamp': dates,
                        'open': 50000 + np.random.normal(0, 1000, 500),
                        'high': 51000 + np.random.normal(0, 1000, 500),
                        'low': 49000 + np.random.normal(0, 1000, 500),
                        'close': 50000 + np.random.normal(0, 1000, 500),
                        'volume': np.random.randint(1000, 10000, 500)
                    })
                    
                    # Calculate indicators
                    df = calculate_indicators(df)
                    df = generate_signals(df)
                    
                    # Train model
                    predictor = LSTMSignalPredictor(sequence_length=sequence_length)
                    result = predictor.train_model(df, epochs=epochs, batch_size=batch_size)
                    
                    if result.get('success'):
                        st.success("✅ LSTM model trained successfully!")
                        st.session_state['lstm_model'] = predictor
                        
                        # Show training metrics
                        st.metric("Final Accuracy", f"{result.get('accuracy', 0):.3f}")
                        st.metric("Training Time", f"{result.get('training_time', 0):.1f}s")
                        
                    else:
                        st.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Training error: {e}")
    
    def ml_step_make_predictions(self):
        """ML step: Make predictions."""
        st.subheader("🔮 Make Predictions")
        
        if 'lstm_model' not in st.session_state:
            st.warning("⚠️ Please train a model first (Step 1)")
            return
        
        if st.button("🔮 Make Prediction", key="predict_btn"):
            with st.spinner("Making prediction..."):
                try:
                    predictor = st.session_state['lstm_model']
                    
                    # Create recent data for prediction
                    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
                    np.random.seed(123)
                    df = pd.DataFrame({
                        'timestamp': dates,
                        'open': 50000 + np.random.normal(0, 1000, 100),
                        'high': 51000 + np.random.normal(0, 1000, 100),
                        'low': 49000 + np.random.normal(0, 1000, 100),
                        'close': 50000 + np.random.normal(0, 1000, 100),
                        'volume': np.random.randint(1000, 10000, 100)
                    })
                    
                    df = calculate_indicators(df)
                    
                    # Make prediction
                    prediction = predictor.predict_signal(df)
                    
                    st.success("✅ Prediction made!")
                    
                    # Display result
                    signal = prediction.get('signal', 'UNKNOWN')
                    confidence = prediction.get('confidence', 0)
                    
                    if signal == 'BUY':
                        st.success(f"📈 BUY Signal (Confidence: {confidence:.1%})")
                    elif signal == 'SELL':
                        st.error(f"📉 SELL Signal (Confidence: {confidence:.1%})")
                    else:
                        st.info(f"⏸️ HOLD Signal (Confidence: {confidence:.1%})")
                        
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
    
    # Placeholder methods for other tutorials
    def tutorial_advanced_backtesting(self):
        st.header("🔬 Advanced Backtesting Tutorial")
        st.info("🚧 Advanced backtesting features coming soon!")
    
    def tutorial_system_configuration(self):
        st.header("⚙️ System Configuration Tutorial")
        st.info("🚧 Configuration tutorial coming soon!")
    
    def tutorial_performance_monitoring(self):
        st.header("📈 Performance Monitoring Tutorial")
        st.info("🚧 Monitoring tutorial coming soon!")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="TradPal Indicator Tutorials",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 TradPal Indicator - Interactive Tutorials")
    
    # Sidebar navigation
    st.sidebar.title("📚 Tutorial Navigation")
    
    tutorial_options = {
        "🚀 Getting Started": "getting_started",
        "📊 Basic Backtesting": "basic_backtesting",
        "🔬 Advanced Backtesting": "advanced_backtesting",
        "🤖 ML Prediction": "ml_prediction",
        "🔄 Walk-Forward Analysis": "walk_forward_analysis",
        "⚙️ System Configuration": "system_configuration",
        "📈 Performance Monitoring": "performance_monitoring"
    }
    
    selected_tutorial = st.sidebar.selectbox(
        "Choose Tutorial:",
        list(tutorial_options.keys()),
        key="main_tutorial_select"
    )
    
    # Initialize tutorial manager
    tutorial_manager = TutorialManager()
    
    # Run selected tutorial
    tutorial_manager.run_tutorial(tutorial_options[selected_tutorial])
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 About")
    st.sidebar.markdown("""
    **TradPal Indicator** is a comprehensive trading system featuring:
    
    - Multi-timeframe analysis
    - Advanced ML predictions
    - Risk management
    - Walk-forward optimization
    - Real-time monitoring
    """)


if __name__ == "__main__":
    main()
