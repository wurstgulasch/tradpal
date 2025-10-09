#!/usr/bin/env python3
"""
Live Charts UI Component

Interactive real-time charts using Plotly for visualization.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.data_fetcher import fetch_data
    from src.indicators import calculate_indicators
    from src.signal_generator import generate_signals
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False


class LiveChartsUI:
    """Live Charts User Interface."""
    
    def __init__(self):
        """Initialize Live Charts UI."""
        # Initialize session state
        if 'chart_symbol' not in st.session_state:
            st.session_state.chart_symbol = 'BTC/USDT'
        if 'chart_timeframe' not in st.session_state:
            st.session_state.chart_timeframe = '1m'
        if 'chart_type' not in st.session_state:
            st.session_state.chart_type = 'Candlestick'
        if 'show_indicators' not in st.session_state:
            st.session_state.show_indicators = {
                'EMA': True,
                'RSI': True,
                'BB': True,
                'Volume': True,
                'Signals': True
            }
    
    def render(self):
        """Render the Live Charts UI."""
        st.header("📈 Live Trading Charts")
        st.markdown("Real-time interactive charts with technical indicators")
        
        # Controls
        self.render_chart_controls()
        
        st.divider()
        
        # Main chart
        self.render_main_chart()
        
        st.divider()
        
        # Additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_indicator_chart()
        
        with col2:
            self.render_signal_analysis()
    
    def render_chart_controls(self):
        """Render chart control panel."""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.session_state.chart_symbol = st.selectbox(
                "Symbol",
                ['BTC/USDT', 'ETH/USDT', 'EUR/USD', 'AAPL', 'TSLA'],
                index=0
            )
        
        with col2:
            st.session_state.chart_timeframe = st.selectbox(
                "Timeframe",
                ['1m', '5m', '15m', '1h', '4h', '1d'],
                index=0
            )
        
        with col3:
            st.session_state.chart_type = st.selectbox(
                "Chart Type",
                ['Candlestick', 'Line', 'OHLC'],
                index=0
            )
        
        with col4:
            auto_refresh = st.checkbox("Auto Refresh", value=False)
        
        with col5:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        # Indicator toggles
        st.markdown("#### 📊 Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.session_state.show_indicators['EMA'] = st.checkbox("EMA", value=True)
        with col2:
            st.session_state.show_indicators['RSI'] = st.checkbox("RSI", value=True)
        with col3:
            st.session_state.show_indicators['BB'] = st.checkbox("Bollinger Bands", value=True)
        with col4:
            st.session_state.show_indicators['Volume'] = st.checkbox("Volume", value=True)
        with col5:
            st.session_state.show_indicators['Signals'] = st.checkbox("Signals", value=True)
    
    def render_main_chart(self):
        """Render the main trading chart."""
        st.subheader(f"📊 {st.session_state.chart_symbol} - {st.session_state.chart_timeframe}")
        
        # Get data
        data = self.get_chart_data()
        
        if data is None or data.empty:
            st.warning("No data available. Using sample data for demonstration.")
            data = self.generate_sample_data()
        
        # Calculate indicators if data module is available
        if DATA_AVAILABLE:
            try:
                data = calculate_indicators(data)
                data = generate_signals(data)
            except Exception as e:
                st.warning(f"Could not calculate indicators: {e}")
        
        # Create chart based on type
        fig = self.create_chart(data)
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Current price info
        if not data.empty:
            latest = data.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${latest['close']:.2f}", 
                         f"{((latest['close'] - data.iloc[-2]['close']) / data.iloc[-2]['close'] * 100):.2f}%")
            with col2:
                st.metric("24h High", f"${data['high'].max():.2f}")
            with col3:
                st.metric("24h Low", f"${data['low'].min():.2f}")
            with col4:
                st.metric("Volume", f"{latest.get('volume', 0):.0f}")
    
    def create_chart(self, data):
        """Create the main chart with indicators."""
        # Determine number of subplots
        n_subplots = 1
        if st.session_state.show_indicators['RSI']:
            n_subplots += 1
        if st.session_state.show_indicators['Volume']:
            n_subplots += 1
        
        # Create subplots
        if n_subplots == 1:
            fig = go.Figure()
            row = 1
        else:
            subplot_titles = ["Price"]
            if st.session_state.show_indicators['RSI']:
                subplot_titles.append("RSI")
            if st.session_state.show_indicators['Volume']:
                subplot_titles.append("Volume")
            
            row_heights = [0.6] + [0.2] * (n_subplots - 1)
            
            fig = make_subplots(
                rows=n_subplots,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=subplot_titles,
                row_heights=row_heights
            )
            row = 1
        
        # Add main price chart
        if st.session_state.chart_type == 'Candlestick':
            fig.add_trace(go.Candlestick(
                x=data.index if 'timestamp' not in data.columns else data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ), row=row, col=1)
        elif st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=data.index if 'timestamp' not in data.columns else data['timestamp'],
                y=data['close'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ), row=row, col=1)
        else:  # OHLC
            fig.add_trace(go.Ohlc(
                x=data.index if 'timestamp' not in data.columns else data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ), row=row, col=1)
        
        # Add EMA if enabled
        if st.session_state.show_indicators['EMA'] and 'EMA9' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index if 'timestamp' not in data.columns else data['timestamp'],
                y=data['EMA9'],
                mode='lines',
                name='EMA 9',
                line=dict(color='orange', width=1)
            ), row=row, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index if 'timestamp' not in data.columns else data['timestamp'],
                y=data['EMA21'],
                mode='lines',
                name='EMA 21',
                line=dict(color='purple', width=1)
            ), row=row, col=1)
        
        # Add Bollinger Bands if enabled
        if st.session_state.show_indicators['BB'] and 'BB_upper' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index if 'timestamp' not in data.columns else data['timestamp'],
                y=data['BB_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=True
            ), row=row, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index if 'timestamp' not in data.columns else data['timestamp'],
                y=data['BB_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)'
            ), row=row, col=1)
        
        # Add buy/sell signals if enabled
        if st.session_state.show_indicators['Signals']:
            if 'Buy_Signal' in data.columns:
                buy_signals = data[data['Buy_Signal'] == 1]
                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_signals.index if 'timestamp' not in buy_signals.columns else buy_signals['timestamp'],
                        y=buy_signals['low'] * 0.995,
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(color='green', size=15, symbol='triangle-up')
                    ), row=row, col=1)
            
            if 'Sell_Signal' in data.columns:
                sell_signals = data[data['Sell_Signal'] == 1]
                if not sell_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_signals.index if 'timestamp' not in sell_signals.columns else sell_signals['timestamp'],
                        y=sell_signals['high'] * 1.005,
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(color='red', size=15, symbol='triangle-down')
                    ), row=row, col=1)
        
        # Add RSI subplot if enabled
        if st.session_state.show_indicators['RSI'] and 'RSI' in data.columns:
            row += 1
            fig.add_trace(go.Scatter(
                x=data.index if 'timestamp' not in data.columns else data['timestamp'],
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='blue', width=2)
            ), row=row, col=1)
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=row, col=1)
        
        # Add Volume subplot if enabled
        if st.session_state.show_indicators['Volume'] and 'volume' in data.columns:
            row += 1
            colors = ['red' if data['close'].iloc[i] < data['open'].iloc[i] else 'green' 
                     for i in range(len(data))]
            
            fig.add_trace(go.Bar(
                x=data.index if 'timestamp' not in data.columns else data['timestamp'],
                y=data['volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ), row=row, col=1)
        
        # Update layout
        fig.update_layout(
            height=600 if n_subplots == 1 else 800,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def render_indicator_chart(self):
        """Render indicator comparison chart."""
        st.subheader("📊 Indicator Comparison")
        
        data = self.get_chart_data()
        if data is None or data.empty:
            data = self.generate_sample_data()
        
        if DATA_AVAILABLE:
            try:
                data = calculate_indicators(data)
            except Exception:
                pass
        
        # Create comparison chart
        fig = go.Figure()
        
        if 'RSI' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                name='RSI',
                yaxis='y1'
            ))
        
        if 'ADX' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['ADX'],
                name='ADX',
                yaxis='y2'
            ))
        
        fig.update_layout(
            height=400,
            yaxis=dict(title="RSI", side='left'),
            yaxis2=dict(title="ADX", overlaying='y', side='right'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_signal_analysis(self):
        """Render signal analysis."""
        st.subheader("🎯 Signal Analysis")
        
        # Signal strength gauge
        signal_strength = np.random.uniform(0, 100)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=signal_strength,
            title={'text': "Signal Strength"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Signal summary
        st.markdown("#### Recent Signals")
        signals_data = {
            'Time': ['10:45', '10:30', '10:15'],
            'Type': ['BUY', 'SELL', 'BUY'],
            'Strength': ['Strong', 'Moderate', 'Weak']
        }
        st.dataframe(pd.DataFrame(signals_data), use_container_width=True)
    
    def get_chart_data(self):
        """Get real chart data."""
        if DATA_AVAILABLE:
            try:
                data = fetch_data(limit=100)
                return data
            except Exception as e:
                st.warning(f"Could not fetch real data: {e}")
                return None
        return None
    
    def generate_sample_data(self, periods=100):
        """Generate sample OHLCV data for demonstration."""
        np.random.seed(42)
        dates = pd.date_range(start=datetime.now() - timedelta(hours=periods), periods=periods, freq='1min')
        
        # Generate realistic price data
        base_price = 45000
        returns = np.random.normal(0, 0.002, periods)
        close = base_price * np.cumprod(1 + returns)
        
        # Generate OHLC from close
        open_prices = close * (1 + np.random.uniform(-0.001, 0.001, periods))
        high = np.maximum(open_prices, close) * (1 + np.random.uniform(0, 0.005, periods))
        low = np.minimum(open_prices, close) * (1 - np.random.uniform(0, 0.005, periods))
        volume = np.random.uniform(100, 1000, periods)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        # Calculate simple indicators
        data['EMA9'] = data['close'].ewm(span=9).mean()
        data['EMA21'] = data['close'].ewm(span=21).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_middle'] = data['close'].rolling(window=20).mean()
        data['BB_std'] = data['close'].rolling(window=20).std()
        data['BB_upper'] = data['BB_middle'] + (data['BB_std'] * 2)
        data['BB_lower'] = data['BB_middle'] - (data['BB_std'] * 2)
        
        # Generate some signals
        data['Buy_Signal'] = 0
        data['Sell_Signal'] = 0
        
        # Simple signal logic
        for i in range(21, len(data)):
            if (data['EMA9'].iloc[i] > data['EMA21'].iloc[i] and 
                data['EMA9'].iloc[i-1] <= data['EMA21'].iloc[i-1]):
                data.loc[data.index[i], 'Buy_Signal'] = 1
            elif (data['EMA9'].iloc[i] < data['EMA21'].iloc[i] and 
                  data['EMA9'].iloc[i-1] >= data['EMA21'].iloc[i-1]):
                data.loc[data.index[i], 'Sell_Signal'] = 1
        
        return data
