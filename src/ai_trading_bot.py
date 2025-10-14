#!/usr/bin/env python3
"""
TradPal Advanced AI Trading Bot

A comprehensive AI-powered trading bot that integrates all advanced features:
- Sentiment Analysis (Twitter/News/Reddit)
- Multi-Asset Portfolio Management
- Advanced ML Ensemble Models
- Market Regime Detection
- Walk-Forward Optimization
- Kelly Criterion Position Sizing
- Risk Management (VaR, CVaR)
- Real-time Data Integration
- Paper/Live Trading Modes
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.settings import (
    # ML Settings
    ML_ADVANCED_FEATURES_ENABLED, ML_ENSEMBLE_MODELS, ML_MARKET_REGIME_DETECTION,
    # Sentiment Settings
    SENTIMENT_ENABLED, SENTIMENT_WEIGHT,
    # Risk Management
    KELLY_ENABLED, KELLY_FRACTION,
    # Trading Modes
    PAPER_TRADING_ENABLED, LIVE_TRADING_ENABLED,
    # Other settings
    RISK_PER_TRADE, MAX_LEVERAGE
)

# Import core modules
from src.advanced_ml_predictor import AdvancedMLPredictor
from src.sentiment_analyzer import SentimentAnalyzer
from src.portfolio_manager import PortfolioManager
from src.walk_forward_optimizer import WalkForwardOptimizer
from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators
from src.risk_manager import RiskManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ai_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AITradingBot:
    """
    Advanced AI Trading Bot with full feature integration
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AI Trading Bot

        Args:
            config: Bot configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.ml_predictor = None
        self.sentiment_analyzer = None
        self.portfolio_manager = None
        self.walk_forward_optimizer = None
        self.risk_manager = None

        # Bot state
        self.is_active = False
        self.trading_mode = config.get('trading_mode', 'paper')  # 'paper' or 'live'
        self.assets = config.get('assets', ['BTC/USDT'])
        self.timeframe = config.get('timeframe', '1h')

        # Performance tracking
        self.performance_history = []
        self.trade_history = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0

        # Initialize components
        self._initialize_components()

        self.logger.info("AI Trading Bot initialized successfully")

    def _initialize_components(self):
        """Initialize all bot components"""
        try:
            # Initialize cache
            self._market_data_cache = {}
            self._component_cache = {}

            # ML Predictor (lazy load)
            if self.config.get('advanced_ml_enabled', ML_ADVANCED_FEATURES_ENABLED):
                # Defer actual initialization until first use
                self._ml_predictor_configured = True
            else:
                self.ml_predictor = None

            # Sentiment Analyzer (lazy load)
            if self.config.get('sentiment_enabled', SENTIMENT_ENABLED):
                # Defer actual initialization until first use
                self._sentiment_analyzer_configured = True
            else:
                self.sentiment_analyzer = None

            # Portfolio Manager
            if self.config.get('multi_asset_enabled', False):
                self.portfolio_manager = PortfolioManager(
                    assets=self.assets,
                    initial_balance=self.config.get('initial_balance', 10000.0)
                )
                self.logger.info("Portfolio Manager initialized")
            else:
                self.portfolio_manager = None

            # Walk-Forward Optimizer
            if self.config.get('walk_forward_enabled', False):
                self.walk_forward_optimizer = WalkForwardOptimizer()
                self.logger.info("Walk-Forward Optimizer initialized")
            else:
                self.walk_forward_optimizer = None

            # Risk Manager (always initialize)
            self.risk_manager = RiskManager(
                max_drawdown=self.config.get('max_drawdown', 0.1),
                max_daily_loss=self.config.get('max_daily_loss', 0.05),
                kelly_enabled=self.config.get('kelly_enabled', KELLY_ENABLED),
                kelly_fraction=self.config.get('kelly_fraction', KELLY_FRACTION)
            )
            self.logger.info("Risk Manager initialized")

        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise

    def start(self):
        """Start the trading bot"""
        if self.is_active:
            self.logger.warning("Bot is already active")
            return

        self.is_active = True
        self.logger.info("AI Trading Bot started")

        # Main trading loop
        while self.is_active:
            try:
                self._trading_cycle()
                time.sleep(self.config.get('cycle_interval', 60))  # Default 1 minute

            except Exception as e:
                self.logger.error(f"Error in trading cycle: {str(e)}")
                time.sleep(30)  # Wait before retry

    def stop(self):
        """Stop the trading bot"""
        self.is_active = False
        self.logger.info("AI Trading Bot stopped")

    def _trading_cycle(self):
        """Execute one complete trading cycle"""
        cycle_start = datetime.now()

        try:
            # 1. Gather market data
            market_data = self._gather_market_data()

            # 2. Generate signals for all assets
            signals = {}
            for asset in self.assets:
                signal = self._generate_signal(asset, market_data.get(asset, {}))
                signals[asset] = signal

            # 3. Apply risk management
            filtered_signals = self._apply_risk_management(signals)

            # 4. Execute trades
            trades_executed = self._execute_trades(filtered_signals)

            # 5. Update portfolio
            if self.portfolio_manager:
                self.portfolio_manager.update_portfolio(market_data)

            # 6. Log performance
            self._log_performance(cycle_start, trades_executed)

            self.logger.info(f"Trading cycle completed in {(datetime.now() - cycle_start).total_seconds():.2f}s")

        except Exception as e:
            self.logger.error(f"Trading cycle failed: {str(e)}")

    def _gather_market_data(self) -> Dict[str, Dict]:
        """Gather market data for all assets"""
        market_data = {}

        for asset in self.assets:
            try:
                # Check cache first (simple in-memory cache)
                cache_key = f"{asset}_{self.timeframe}"
                if hasattr(self, '_market_data_cache') and cache_key in self._market_data_cache:
                    cached_data, cache_time = self._market_data_cache[cache_key]
                    # Use cache if less than 5 minutes old
                    if (datetime.now() - cache_time).seconds < 300:
                        market_data[asset] = cached_data
                        continue

                # Fetch recent data (reduced limit for performance)
                data = fetch_historical_data(
                    symbol=asset,
                    timeframe=self.timeframe,
                    limit=50  # Reduced from 100 for better performance
                )

                if data is not None and not data.empty:
                    # Calculate indicators
                    data_with_indicators = calculate_indicators(data)

                    # Add sentiment if enabled
                    if self.sentiment_analyzer:
                        sentiment_score = self.sentiment_analyzer.get_sentiment_score(asset)
                        data_with_indicators['sentiment_score'] = sentiment_score

                    market_data[asset] = {
                        'data': data_with_indicators,
                        'current_price': data_with_indicators['close'].iloc[-1],
                        'volume': data_with_indicators['volume'].iloc[-1]
                    }

                    # Cache the data
                    if not hasattr(self, '_market_data_cache'):
                        self._market_data_cache = {}
                    self._market_data_cache[cache_key] = (market_data[asset], datetime.now())

            except Exception as e:
                self.logger.error(f"Error gathering data for {asset}: {str(e)}")

        return market_data

    def _generate_signal(self, asset: str, market_data: Dict) -> Dict[str, Any]:
        """Generate trading signal for a specific asset"""
        try:
            signal = {
                'asset': asset,
                'timestamp': datetime.now(),
                'signal': 'HOLD',
                'confidence': 0.5,
                'price': market_data.get('current_price', 0),
                'reason': 'Default hold signal'
            }

            # Base technical signal
            technical_signal = self._generate_technical_signal(market_data)
            signal.update(technical_signal)

            # ML signal enhancement
            if self.ml_predictor and self.config.get('advanced_ml_enabled', True):
                ml_signal = self._generate_ml_signal(asset, market_data)
                signal = self._combine_signals(signal, ml_signal)

            # Sentiment enhancement
            if self.sentiment_analyzer and self.config.get('sentiment_enabled', True):
                sentiment_signal = self._generate_sentiment_signal(asset, market_data)
                signal = self._combine_signals(signal, sentiment_signal)

            # Market regime adjustment
            if self.config.get('regime_adaptation', True):
                signal = self._apply_regime_adaptation(signal, market_data)

            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal for {asset}: {str(e)}")
            return {
                'asset': asset,
                'timestamp': datetime.now(),
                'signal': 'HOLD',
                'confidence': 0.5,
                'price': 0,
                'reason': f'Error: {str(e)}'
            }

    def _generate_technical_signal(self, market_data: Dict) -> Dict[str, Any]:
        """Generate technical analysis signal"""
        try:
            data = market_data.get('data')
            if data is None or data.empty:
                return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'No data available'}

            # Simple EMA crossover strategy
            ema_short = data['ema_short'].iloc[-1]
            ema_long = data['ema_long'].iloc[-1]
            rsi = data['rsi'].iloc[-1]
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            current_price = data['close'].iloc[-1]

            signal = 'HOLD'
            confidence = 0.5
            reason = 'Neutral conditions'

            # EMA crossover logic
            if ema_short > ema_long and rsi < 70:
                signal = 'BUY'
                confidence = 0.7
                reason = 'EMA crossover + RSI oversold'
            elif ema_short < ema_long and rsi > 30:
                signal = 'SELL'
                confidence = 0.7
                reason = 'EMA crossover + RSI overbought'

            # Bollinger Band confirmation
            if signal == 'BUY' and current_price < bb_lower:
                confidence += 0.1
                reason += ' + BB support'
            elif signal == 'SELL' and current_price > bb_upper:
                confidence += 0.1
                reason += ' + BB resistance'

            return {
                'signal': signal,
                'confidence': min(confidence, 0.95),
                'reason': reason
            }

        except Exception as e:
            self.logger.error(f"Error in technical signal generation: {str(e)}")
            return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'Technical analysis error'}

    def _generate_ml_signal(self, asset: str, market_data: Dict) -> Dict[str, Any]:
        """Generate ML-enhanced signal"""
        try:
            # Lazy load ML predictor
            if not hasattr(self, 'ml_predictor') or self.ml_predictor is None:
                if hasattr(self, '_ml_predictor_configured') and self._ml_predictor_configured:
                    self.ml_predictor = AdvancedMLPredictor()
                    self.logger.info("Advanced ML Predictor initialized (lazy loaded)")
                else:
                    return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'ML disabled'}

            if not self.ml_predictor:
                return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'ML disabled'}

            # Get ML prediction
            ml_result = self.ml_predictor.predict_signal(asset, market_data.get('data'))

            if ml_result:
                return {
                    'signal': ml_result.get('signal', 'HOLD'),
                    'confidence': ml_result.get('confidence', 0.5),
                    'reason': f"ML: {ml_result.get('reason', 'ML prediction')}"
                }

            return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'ML prediction failed'}

        except Exception as e:
            self.logger.error(f"Error in ML signal generation: {str(e)}")
            return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'ML error'}

    def _generate_sentiment_signal(self, asset: str, market_data: Dict) -> Dict[str, Any]:
        """Generate sentiment-enhanced signal"""
        try:
            # Lazy load sentiment analyzer
            if not hasattr(self, 'sentiment_analyzer') or self.sentiment_analyzer is None:
                if hasattr(self, '_sentiment_analyzer_configured') and self._sentiment_analyzer_configured:
                    self.sentiment_analyzer = SentimentAnalyzer()
                    self.logger.info("Sentiment Analyzer initialized (lazy loaded)")
                else:
                    return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'Sentiment disabled'}

            if not self.sentiment_analyzer:
                return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'Sentiment disabled'}

            sentiment_score = market_data.get('data', {}).get('sentiment_score', 0)

            # Convert sentiment to signal
            if sentiment_score > 0.2:
                return {
                    'signal': 'BUY',
                    'confidence': min(abs(sentiment_score) * self.config.get('sentiment_weight', 0.2), 0.3),
                    'reason': f'Sentiment: Bullish ({sentiment_score:.2f})'
                }
            elif sentiment_score < -0.2:
                return {
                    'signal': 'SELL',
                    'confidence': min(abs(sentiment_score) * self.config.get('sentiment_weight', 0.2), 0.3),
                    'reason': f'Sentiment: Bearish ({sentiment_score:.2f})'
                }

            return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'Sentiment neutral'}

        except Exception as e:
            self.logger.error(f"Error in sentiment signal generation: {str(e)}")
            return {'signal': 'HOLD', 'confidence': 0.5, 'reason': 'Sentiment error'}

    def _combine_signals(self, base_signal: Dict, enhancement: Dict) -> Dict:
        """Combine multiple signals using ensemble logic"""
        try:
            # Simple weighted combination
            base_weight = 1.0 - self.config.get('enhancement_weight', 0.3)
            enhancement_weight = self.config.get('enhancement_weight', 0.3)

            # Signal strength based on confidence
            base_strength = base_signal['confidence'] * base_weight
            enhancement_strength = enhancement['confidence'] * enhancement_weight

            # Determine combined signal
            if base_signal['signal'] == enhancement['signal']:
                combined_signal = base_signal['signal']
                combined_confidence = min(base_signal['confidence'] + enhancement['confidence'] * 0.5, 0.95)
            elif base_strength > enhancement_strength:
                combined_signal = base_signal['signal']
                combined_confidence = base_signal['confidence'] * 0.8
            else:
                combined_signal = enhancement['signal']
                combined_confidence = enhancement['confidence'] * 0.8

            return {
                'signal': combined_signal,
                'confidence': combined_confidence,
                'reason': f"{base_signal['reason']} + {enhancement['reason']}"
            }

        except Exception as e:
            self.logger.error(f"Error combining signals: {str(e)}")
            return base_signal

    def _apply_regime_adaptation(self, signal: Dict, market_data: Dict) -> Dict:
        """Apply market regime adaptation to signal"""
        try:
            if not self.config.get('regime_adaptation', True):
                return signal

            # Simple regime detection based on volatility
            data = market_data.get('data')
            if data is not None and len(data) > 20:
                returns = data['close'].pct_change().dropna()
                volatility = returns.std()

                # High volatility = trending regime
                if volatility > 0.02:  # 2% daily volatility threshold
                    if signal['signal'] == 'BUY':
                        signal['confidence'] *= 1.2  # Boost confidence in trending markets
                        signal['reason'] += ' (Trending regime)'
                    elif signal['signal'] == 'SELL':
                        signal['confidence'] *= 1.2
                        signal['reason'] += ' (Trending regime)'
                else:
                    signal['confidence'] *= 0.8  # Reduce confidence in ranging markets
                    signal['reason'] += ' (Ranging regime)'

            return signal

        except Exception as e:
            self.logger.error(f"Error in regime adaptation: {str(e)}")
            return signal

    def _apply_risk_management(self, signals: Dict[str, Dict]) -> Dict[str, Dict]:
        """Apply risk management filters to signals"""
        try:
            if not self.risk_manager:
                return signals

            filtered_signals = {}

            for asset, signal in signals.items():
                # Check confidence threshold
                min_confidence = self.config.get('min_confidence_threshold', 0.6)
                if signal['confidence'] < min_confidence:
                    signal['signal'] = 'HOLD'
                    signal['reason'] += f' (Confidence {signal["confidence"]:.2f} < {min_confidence})'

                # Apply Kelly criterion if enabled
                if self.config.get('kelly_enabled', KELLY_ENABLED) and signal['signal'] != 'HOLD':
                    position_size = self.risk_manager.calculate_kelly_position_size(
                        win_rate=self._calculate_recent_win_rate(asset),
                        avg_win=0.05,  # Assume 5% average win
                        avg_loss=0.03  # Assume 3% average loss
                    )
                    signal['position_size'] = position_size

                # Check drawdown limits
                if self.risk_manager.check_drawdown_limits():
                    signal['signal'] = 'HOLD'
                    signal['reason'] += ' (Drawdown limit reached)'

                filtered_signals[asset] = signal

            return filtered_signals

        except Exception as e:
            self.logger.error(f"Error in risk management: {str(e)}")
            return signals

    def _execute_trades(self, signals: Dict[str, Dict]) -> List[Dict]:
        """Execute trades based on signals"""
        executed_trades = []

        try:
            for asset, signal in signals.items():
                if signal['signal'] == 'HOLD':
                    continue

                # Check trading limits
                if not self._check_trading_limits(asset):
                    continue

                # Calculate position size
                position_size = self._calculate_position_size(asset, signal)

                # Execute trade
                trade = self._execute_single_trade(asset, signal, position_size)

                if trade:
                    executed_trades.append(trade)
                    self.trade_history.append(trade)

                    # Update daily P&L
                    if trade.get('pnl', 0) != 0:
                        self.daily_pnl += trade['pnl']

        except Exception as e:
            self.logger.error(f"Error executing trades: {str(e)}")

        return executed_trades

    def _execute_single_trade(self, asset: str, signal: Dict, position_size: float) -> Optional[Dict]:
        """Execute a single trade"""
        try:
            trade = {
                'timestamp': datetime.now(),
                'asset': asset,
                'side': signal['signal'],
                'price': signal['price'],
                'quantity': position_size,
                'reason': signal['reason'],
                'confidence': signal['confidence']
            }

            if self.trading_mode == 'live' and self.config.get('live_trading_enabled', False):
                # Execute live trade via broker API
                self.logger.info(f"Executing LIVE trade: {trade}")
                # TODO: Implement actual broker API call
                trade['status'] = 'EXECUTED'
                trade['pnl'] = 0  # Will be updated on close

            elif self.trading_mode == 'paper':
                # Execute paper trade
                self.logger.info(f"Executing PAPER trade: {trade}")
                trade['status'] = 'EXECUTED'
                trade['pnl'] = 0  # Paper trade, no real P&L yet

            return trade

        except Exception as e:
            self.logger.error(f"Error executing single trade: {str(e)}")
            return None

    def _calculate_position_size(self, asset: str, signal: Dict) -> float:
        """Calculate position size based on risk management"""
        try:
            base_risk = self.config.get('risk_per_trade', RISK_PER_TRADE)
            current_price = signal['price']

            if self.config.get('kelly_enabled', KELLY_ENABLED) and 'position_size' in signal:
                # Use Kelly-adjusted position size
                kelly_size = signal['position_size']
                position_value = kelly_size * self.config.get('initial_balance', 10000)
            else:
                # Use fixed risk per trade
                position_value = base_risk * self.config.get('initial_balance', 10000)

            # Convert to quantity
            quantity = position_value / current_price

            # Apply leverage if enabled
            leverage = min(self.config.get('max_leverage', 3.0), 5)  # Cap at 5x for safety
            quantity *= leverage

            return quantity

        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.001  # Minimum safe quantity

    def _check_trading_limits(self, asset: str) -> bool:
        """Check if trading limits allow new trades"""
        try:
            # Check max trades per hour
            recent_trades = [t for t in self.trade_history
                           if (datetime.now() - t['timestamp']).total_seconds() < 3600]
            if len(recent_trades) >= self.config.get('max_trades_per_hour', 5):
                return False

            # Check max daily loss
            if self.daily_pnl < -self.config.get('max_daily_loss', 0.05) * self.config.get('initial_balance', 10000):
                return False

            # Check max open positions
            open_positions = len([t for t in self.trade_history if t.get('status') == 'OPEN'])
            if open_positions >= self.config.get('max_open_positions', 3):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking trading limits: {str(e)}")
            return False

    def _calculate_recent_win_rate(self, asset: str) -> float:
        """Calculate recent win rate for Kelly criterion"""
        try:
            recent_trades = [t for t in self.trade_history
                           if t['asset'] == asset and 'pnl' in t
                           and (datetime.now() - t['timestamp']).days <= 30]

            if not recent_trades:
                return 0.5  # Default 50% win rate

            winning_trades = len([t for t in recent_trades if t['pnl'] > 0])
            return winning_trades / len(recent_trades)

        except Exception as e:
            self.logger.error(f"Error calculating win rate: {str(e)}")
            return 0.5

    def _log_performance(self, cycle_start: datetime, trades_executed: List[Dict]):
        """Log performance metrics"""
        try:
            performance_entry = {
                'timestamp': datetime.now(),
                'cycle_duration': (datetime.now() - cycle_start).total_seconds(),
                'trades_executed': len(trades_executed),
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
                'active_positions': len([t for t in self.trade_history if t.get('status') == 'OPEN'])
            }

            self.performance_history.append(performance_entry)

            # Log summary
            self.logger.info(f"Performance: Trades={len(trades_executed)}, "
                           f"Daily P&L=${self.daily_pnl:.2f}, "
                           f"Total P&L=${self.total_pnl:.2f}")

        except Exception as e:
            self.logger.error(f"Error logging performance: {str(e)}")

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            'is_active': self.is_active,
            'trading_mode': self.trading_mode,
            'assets': self.assets,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'total_trades': len(self.trade_history),
            'active_positions': len([t for t in self.trade_history if t.get('status') == 'OPEN']),
            'last_update': datetime.now()
        }

    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades"""
        return sorted(self.trade_history, key=lambda x: x['timestamp'], reverse=True)[:limit]

    def get_performance_history(self, hours: int = 24) -> List[Dict]:
        """Get performance history for the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [p for p in self.performance_history if p['timestamp'] > cutoff]


def create_ai_bot(config: Dict[str, Any]) -> AITradingBot:
    """
    Factory function to create an AI Trading Bot

    Args:
        config: Bot configuration

    Returns:
        Configured AI Trading Bot instance
    """
    return AITradingBot(config)


# Example usage and configuration
if __name__ == "__main__":
    # Example configuration
    bot_config = {
        'trading_mode': 'paper',  # 'paper' or 'live'
        'assets': ['BTC/USDT', 'ETH/USDT'],
        'timeframe': '1h',
        'initial_balance': 10000.0,

        # Feature toggles
        'advanced_ml_enabled': True,
        'sentiment_enabled': True,
        'multi_asset_enabled': True,
        'walk_forward_enabled': False,
        'regime_adaptation': True,
        'kelly_enabled': True,

        # Risk parameters
        'risk_per_trade': 0.01,
        'max_leverage': 3.0,
        'max_drawdown': 0.1,
        'max_daily_loss': 0.05,
        'min_confidence_threshold': 0.6,

        # Trading limits
        'max_trades_per_hour': 5,
        'max_open_positions': 3,
        'cycle_interval': 60,  # seconds

        # Enhancement weights
        'sentiment_weight': 0.2,
        'enhancement_weight': 0.3,
        'kelly_fraction': 0.5
    }

    # Create and start bot
    bot = create_ai_bot(bot_config)

    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()
        print("Bot stopped by user")