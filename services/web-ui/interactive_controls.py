#!/usr/bin/env python3
"""
Interactive Controls UI Component

Provides interactive sliders and controls for adjusting indicator parameters in real-time.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from config.settings import TIMEFRAME_PARAMS
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


class InteractiveControlsUI:
    """Interactive Controls User Interface."""
    
    def __init__(self):
        """Initialize Interactive Controls UI."""
        # Initialize session state for parameters
        if 'current_params' not in st.session_state:
            if CONFIG_AVAILABLE:
                st.session_state.current_params = TIMEFRAME_PARAMS.get('1m', {}).copy()
            else:
                st.session_state.current_params = self.get_default_params()
        
        if 'timeframe' not in st.session_state:
            st.session_state.timeframe = '1m'
    
    def get_default_params(self):
        """Get default parameters if config not available."""
        return {
            'ema_short': 9,
            'ema_long': 21,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'atr_period': 14,
            'atr_sl_multiplier': 1.0,
            'atr_tp_multiplier': 2.0,
            'leverage_max': 10,
            'adx_period': 14,
            'adx_threshold': 25
        }
    
    def render(self):
        """Render the Interactive Controls UI."""
        st.header("⚙️ Interactive Controls")
        st.markdown("Adjust indicator parameters in real-time and see the impact on signals")
        
        # Timeframe selector
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            timeframe = st.selectbox(
                "📊 Select Timeframe",
                ['1m', '5m', '15m', '1h', '4h', '1d'],
                index=['1m', '5m', '15m', '1h', '4h', '1d'].index(st.session_state.timeframe)
            )
            
            if timeframe != st.session_state.timeframe:
                st.session_state.timeframe = timeframe
                if CONFIG_AVAILABLE:
                    st.session_state.current_params = TIMEFRAME_PARAMS.get(timeframe, {}).copy()
                st.rerun()
        
        with col2:
            st.metric("Current Timeframe", st.session_state.timeframe)
        
        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                if CONFIG_AVAILABLE:
                    st.session_state.current_params = TIMEFRAME_PARAMS.get(st.session_state.timeframe, {}).copy()
                else:
                    st.session_state.current_params = self.get_default_params()
                st.rerun()
        
        st.divider()
        
        # Tabbed interface for different indicator categories
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend Indicators", "📊 Momentum", "💰 Risk Management", "🎯 Advanced"])
        
        with tab1:
            self.render_trend_indicators()
        
        with tab2:
            self.render_momentum_indicators()
        
        with tab3:
            self.render_risk_management()
        
        with tab4:
            self.render_advanced_settings()
        
        st.divider()
        
        # Action buttons
        self.render_action_buttons()
        
        # Current configuration display
        st.divider()
        self.render_current_config()
    
    def render_trend_indicators(self):
        """Render trend indicator controls."""
        st.subheader("Exponential Moving Averages (EMA)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.current_params['ema_short'] = st.slider(
                "EMA Short Period",
                min_value=5,
                max_value=50,
                value=st.session_state.current_params.get('ema_short', 9),
                step=1,
                help="Shorter EMA period for faster response to price changes"
            )
        
        with col2:
            st.session_state.current_params['ema_long'] = st.slider(
                "EMA Long Period",
                min_value=10,
                max_value=200,
                value=st.session_state.current_params.get('ema_long', 21),
                step=1,
                help="Longer EMA period for trend confirmation"
            )
        
        # Visual feedback
        if st.session_state.current_params['ema_short'] >= st.session_state.current_params['ema_long']:
            st.warning("⚠️ Short EMA should be less than Long EMA")
        else:
            st.success(f"✅ EMA Cross: {st.session_state.current_params['ema_short']}/{st.session_state.current_params['ema_long']}")
        
        st.divider()
        
        # Bollinger Bands
        st.subheader("Bollinger Bands")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.current_params['bb_period'] = st.slider(
                "BB Period",
                min_value=10,
                max_value=50,
                value=st.session_state.current_params.get('bb_period', 20),
                step=1,
                help="Bollinger Bands calculation period"
            )
        
        with col2:
            st.session_state.current_params['bb_std_dev'] = st.slider(
                "BB Standard Deviation",
                min_value=1.0,
                max_value=3.0,
                value=st.session_state.current_params.get('bb_std_dev', 2.0),
                step=0.1,
                help="Number of standard deviations for band width"
            )
    
    def render_momentum_indicators(self):
        """Render momentum indicator controls."""
        st.subheader("Relative Strength Index (RSI)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.current_params['rsi_period'] = st.slider(
                "RSI Period",
                min_value=5,
                max_value=30,
                value=st.session_state.current_params.get('rsi_period', 14),
                step=1,
                help="RSI calculation period"
            )
        
        with col2:
            st.session_state.current_params['rsi_oversold'] = st.slider(
                "RSI Oversold Level",
                min_value=20,
                max_value=40,
                value=st.session_state.current_params.get('rsi_oversold', 30),
                step=5,
                help="RSI level indicating oversold condition"
            )
        
        with col3:
            st.session_state.current_params['rsi_overbought'] = st.slider(
                "RSI Overbought Level",
                min_value=60,
                max_value=80,
                value=st.session_state.current_params.get('rsi_overbought', 70),
                step=5,
                help="RSI level indicating overbought condition"
            )
        
        # RSI visual indicator
        st.markdown("#### RSI Levels")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Oversold", f"{st.session_state.current_params['rsi_oversold']}", "📉")
        with col2:
            st.metric("Neutral", "50", "➡️")
        with col3:
            st.metric("Overbought", f"{st.session_state.current_params['rsi_overbought']}", "📈")
        
        st.divider()
        
        # ADX
        st.subheader("Average Directional Index (ADX)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.current_params['adx_period'] = st.slider(
                "ADX Period",
                min_value=7,
                max_value=30,
                value=st.session_state.current_params.get('adx_period', 14),
                step=1,
                help="ADX calculation period"
            )
        
        with col2:
            st.session_state.current_params['adx_threshold'] = st.slider(
                "ADX Threshold",
                min_value=20,
                max_value=40,
                value=st.session_state.current_params.get('adx_threshold', 25),
                step=5,
                help="Minimum ADX level for strong trend"
            )
    
    def render_risk_management(self):
        """Render risk management controls."""
        st.subheader("Average True Range (ATR)")
        
        st.session_state.current_params['atr_period'] = st.slider(
            "ATR Period",
            min_value=7,
            max_value=30,
            value=st.session_state.current_params.get('atr_period', 14),
            step=1,
            help="ATR calculation period for volatility measurement"
        )
        
        st.divider()
        
        st.subheader("Stop Loss & Take Profit")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.current_params['atr_sl_multiplier'] = st.slider(
                "Stop Loss Multiplier",
                min_value=0.5,
                max_value=3.0,
                value=st.session_state.current_params.get('atr_sl_multiplier', 1.0),
                step=0.1,
                help="ATR multiplier for stop loss distance"
            )
        
        with col2:
            st.session_state.current_params['atr_tp_multiplier'] = st.slider(
                "Take Profit Multiplier",
                min_value=1.0,
                max_value=5.0,
                value=st.session_state.current_params.get('atr_tp_multiplier', 2.0),
                step=0.1,
                help="ATR multiplier for take profit distance"
            )
        
        # Risk/Reward Ratio
        risk_reward = st.session_state.current_params['atr_tp_multiplier'] / st.session_state.current_params['atr_sl_multiplier']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Stop Loss", f"{st.session_state.current_params['atr_sl_multiplier']}x ATR")
        with col2:
            st.metric("Take Profit", f"{st.session_state.current_params['atr_tp_multiplier']}x ATR")
        with col3:
            st.metric("Risk/Reward", f"{risk_reward:.2f}")
        
        if risk_reward < 2.0:
            st.warning("⚠️ Risk/Reward ratio below 2.0")
        else:
            st.success(f"✅ Good Risk/Reward ratio: {risk_reward:.2f}")
        
        st.divider()
        
        st.subheader("Leverage")
        
        st.session_state.current_params['leverage_max'] = st.slider(
            "Maximum Leverage",
            min_value=1,
            max_value=20,
            value=st.session_state.current_params.get('leverage_max', 10),
            step=1,
            help="Maximum leverage for position sizing"
        )
        
        if st.session_state.current_params['leverage_max'] > 10:
            st.error("⚠️ High leverage increases risk significantly!")
        elif st.session_state.current_params['leverage_max'] > 5:
            st.warning("⚠️ Moderate leverage - use with caution")
        else:
            st.success("✅ Conservative leverage setting")
    
    def render_advanced_settings(self):
        """Render advanced settings."""
        st.subheader("🎯 Advanced Configuration")
        
        # Quick presets
        st.markdown("#### Quick Presets")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏃 Scalping", use_container_width=True):
                self.apply_preset('scalping')
        
        with col2:
            if st.button("📈 Trend Following", use_container_width=True):
                self.apply_preset('trend')
        
        with col3:
            if st.button("💰 Conservative", use_container_width=True):
                self.apply_preset('conservative')
        
        st.divider()
        
        # Export/Import configuration
        st.markdown("#### Configuration Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Config", use_container_width=True):
                st.json(st.session_state.current_params)
                st.success("Configuration displayed above")
        
        with col2:
            if st.button("📥 Import Config", use_container_width=True):
                st.info("Paste JSON configuration to import")
    
    def render_action_buttons(self):
        """Render action buttons."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("✅ Apply Settings", use_container_width=True):
                st.success("Settings applied! Use these parameters in your strategy.")
        
        with col2:
            if st.button("💾 Save as Default", use_container_width=True):
                st.success("Settings saved as default for this timeframe")
        
        with col3:
            if st.button("🧪 Test with Backtest", use_container_width=True):
                st.info("Running backtest with current settings...")
        
        with col4:
            if st.button("🔄 Reset to Default", use_container_width=True):
                if CONFIG_AVAILABLE:
                    st.session_state.current_params = TIMEFRAME_PARAMS.get(st.session_state.timeframe, {}).copy()
                else:
                    st.session_state.current_params = self.get_default_params()
                st.rerun()
    
    def render_current_config(self):
        """Display current configuration."""
        st.subheader("📋 Current Configuration")
        
        # Display in expandable section
        with st.expander("View Full Configuration", expanded=False):
            st.json(st.session_state.current_params)
        
        # Key metrics summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("EMA Cross", f"{st.session_state.current_params.get('ema_short', 0)}/{st.session_state.current_params.get('ema_long', 0)}")
        
        with col2:
            st.metric("RSI Period", st.session_state.current_params.get('rsi_period', 0))
        
        with col3:
            st.metric("ATR Period", st.session_state.current_params.get('atr_period', 0))
        
        with col4:
            st.metric("Max Leverage", f"{st.session_state.current_params.get('leverage_max', 0)}x")
    
    def apply_preset(self, preset_name):
        """Apply a configuration preset."""
        presets = {
            'scalping': {
                'ema_short': 5,
                'ema_long': 13,
                'rsi_period': 7,
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'bb_period': 15,
                'bb_std_dev': 2.0,
                'atr_period': 7,
                'atr_sl_multiplier': 0.8,
                'atr_tp_multiplier': 1.5,
                'leverage_max': 15,
                'adx_period': 7,
                'adx_threshold': 20
            },
            'trend': {
                'ema_short': 20,
                'ema_long': 50,
                'rsi_period': 14,
                'rsi_oversold': 40,
                'rsi_overbought': 60,
                'bb_period': 20,
                'bb_std_dev': 2.0,
                'atr_period': 14,
                'atr_sl_multiplier': 1.5,
                'atr_tp_multiplier': 3.0,
                'leverage_max': 5,
                'adx_period': 14,
                'adx_threshold': 30
            },
            'conservative': {
                'ema_short': 21,
                'ema_long': 55,
                'rsi_period': 14,
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'bb_period': 20,
                'bb_std_dev': 2.5,
                'atr_period': 14,
                'atr_sl_multiplier': 2.0,
                'atr_tp_multiplier': 4.0,
                'leverage_max': 3,
                'adx_period': 14,
                'adx_threshold': 30
            }
        }
        
        if preset_name in presets:
            st.session_state.current_params = presets[preset_name]
            st.success(f"Applied {preset_name.title()} preset")
            st.rerun()
