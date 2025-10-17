#!/usr/bin/env python3
"""
Example: GPU-Accelerated Trading Model Training

This example demonstrates how to use GPU acceleration for training
advanced neural network models (LSTM, Transformer) for trading predictions.
"""

import asyncio
import logging
import sys
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.core.gpu_accelerator import (
    get_gpu_accelerator, is_gpu_available, create_gpu_lstm_model,
    create_gpu_transformer_model, train_gpu_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUTradingModelTrainer:
    """
    GPU-accelerated trainer for trading prediction models.
    """

    def __init__(self):
        self.gpu_available = is_gpu_available()
        self.gpu = get_gpu_accelerator()

        if self.gpu_available:
            logger.info("🎯 GPU acceleration enabled for model training")
        else:
            logger.warning("⚠️  GPU not available, falling back to CPU")

    def prepare_trading_data(self, data: pd.DataFrame,
                           sequence_length: int = 60,
                           prediction_horizon: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare trading data for neural network training.

        Args:
            data: OHLCV DataFrame
            sequence_length: Number of time steps for input sequences
            prediction_horizon: Number of steps ahead to predict

        Returns:
            Tuple of (input_sequences, target_values)
        """
        logger.info("📊 Preparing trading data for neural network training...")

        # Calculate returns as target
        data = data.copy()
        data['returns'] = data['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        data = data.dropna()

        # Select features
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        if 'rsi_14' in data.columns:
            feature_columns.extend(['rsi_14', 'sma_20', 'ema_20'])

        # Normalize features
        for col in feature_columns:
            if col in data.columns:
                data[col] = (data[col] - data[col].mean()) / data[col].std()

        # Create sequences
        sequences = []
        targets = []

        for i in range(len(data) - sequence_length):
            seq_data = data[feature_columns].iloc[i:i + sequence_length].values
            target = data['returns'].iloc[i + sequence_length - 1]

            sequences.append(seq_data)
            targets.append(target)

        # Convert to tensors
        X = torch.tensor(np.array(sequences), dtype=torch.float32)
        y = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1)

        logger.info(f"Created {len(X)} sequences of length {sequence_length}")
        logger.info(f"Input shape: {X.shape}, Target shape: {y.shape}")

        return X, y

    def create_data_loaders(self, X: torch.Tensor, y: torch.Tensor,
                          batch_size: int = 32, train_split: float = 0.8):
        """Create PyTorch data loaders for training and validation."""

        dataset = torch.utils.data.TensorDataset(X, y)
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        return train_loader, val_loader

    async def train_lstm_model(self, data: pd.DataFrame,
                              sequence_length: int = 60,
                              hidden_size: int = 128,
                              num_layers: int = 2,
                              epochs: int = 50,
                              batch_size: int = 32) -> Dict[str, Any]:
        """
        Train an LSTM model for trading predictions using GPU acceleration.
        """
        logger.info("🚀 Training LSTM Model with GPU Acceleration")
        logger.info(f"Sequence Length: {sequence_length}, Hidden Size: {hidden_size}")
        logger.info(f"Layers: {num_layers}, Epochs: {epochs}")
        logger.info("")

        # Prepare data
        X, y = self.prepare_trading_data(data, sequence_length)
        train_loader, val_loader = self.create_data_loaders(X, y, batch_size)

        # Create model
        input_size = X.shape[2]  # Number of features
        model = create_gpu_lstm_model(input_size, hidden_size, num_layers, 1)

        logger.info(f"Model: {model}")
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train model
        training_results = train_gpu_model(
            model, train_loader, val_loader, epochs, learning_rate=0.001
        )

        # Evaluate final performance
        model.eval()
        device = self.gpu.get_optimal_device()

        with torch.no_grad():
            all_predictions = []
            all_targets = []

            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                predictions = model(batch_x)
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(batch_y.numpy().flatten())

            # Calculate metrics
            predictions = np.array(all_predictions)
            targets = np.array(all_targets)

            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            correlation = np.corrcoef(predictions, targets)[0, 1]

            logger.info("
📈 Final Validation Metrics:"            logger.info(".6f"            logger.info(".6f"            logger.info(".4f"

        return {
            "model": model,
            "model_type": "LSTM",
            "training_history": training_results,
            "validation_metrics": {
                "mse": mse,
                "mae": mae,
                "correlation": correlation
            },
            "config": {
                "sequence_length": sequence_length,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "epochs": epochs,
                "batch_size": batch_size
            }
        }

    async def train_transformer_model(self, data: pd.DataFrame,
                                    sequence_length: int = 60,
                                    num_heads: int = 8,
                                    num_layers: int = 4,
                                    epochs: int = 30,
                                    batch_size: int = 16) -> Dict[str, Any]:
        """
        Train a Transformer model for trading predictions using GPU acceleration.
        """
        logger.info("🚀 Training Transformer Model with GPU Acceleration")
        logger.info(f"Sequence Length: {sequence_length}, Heads: {num_heads}")
        logger.info(f"Layers: {num_layers}, Epochs: {epochs}")
        logger.info("")

        # Prepare data
        X, y = self.prepare_trading_data(data, sequence_length)
        train_loader, val_loader = self.create_data_loaders(X, y, batch_size)

        # Create model
        input_size = X.shape[2]  # Number of features
        model = create_gpu_transformer_model(input_size, num_heads, num_layers, 1)

        logger.info(f"Model: {model}")
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train model (Transformers typically need lower learning rate)
        training_results = train_gpu_model(
            model, train_loader, val_loader, epochs, learning_rate=0.0001
        )

        # Evaluate final performance
        model.eval()
        device = self.gpu.get_optimal_device()

        with torch.no_grad():
            all_predictions = []
            all_targets = []

            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                predictions = model(batch_x)
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(batch_y.numpy().flatten())

            # Calculate metrics
            predictions = np.array(all_predictions)
            targets = np.array(all_targets)

            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            correlation = np.corrcoef(predictions, targets)[0, 1]

            logger.info("
📈 Final Validation Metrics:"            logger.info(".6f"            logger.info(".6f"            logger.info(".4f"

        return {
            "model": model,
            "model_type": "Transformer",
            "training_history": training_results,
            "validation_metrics": {
                "mse": mse,
                "mae": mae,
                "correlation": correlation
            },
            "config": {
                "sequence_length": sequence_length,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "epochs": epochs,
                "batch_size": batch_size
            }
        }

async def example_gpu_trading_model():
    """
    Example of training GPU-accelerated trading models.
    """
    try:
        logger.info("🎯 TradPal GPU Trading Model Training Example")
        logger.info("=" * 60)
        logger.info("")

        # Initialize trainer
        trainer = GPUTradingModelTrainer()

        # Create sample trading data (replace with real data loading)
        logger.info("📊 Generating sample trading data...")
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')

        # Generate realistic price data with trends and volatility
        np.random.seed(42)
        n_days = len(dates)

        # Base price with trend
        base_price = 50000
        trend = np.linspace(0, 20000, n_days)  # Upward trend
        noise = np.random.normal(0, 1000, n_days).cumsum()  # Random walk
        seasonal = 2000 * np.sin(2 * np.pi * np.arange(n_days) / 365)  # Seasonal component

        close_prices = base_price + trend + noise + seasonal

        # Generate OHLCV data
        data = pd.DataFrame(index=dates)
        data['close'] = close_prices

        # Generate high, low, open from close
        volatility = 0.02  # 2% daily volatility
        for i in range(len(data)):
            close = data.iloc[i]['close']
            high = close * (1 + volatility * np.random.uniform(0, 1))
            low = close * (1 - volatility * np.random.uniform(0, 1))
            open_price = close * (1 + volatility * np.random.normal(0, 0.5))

            # Ensure OHLC relationships
            high = max(high, close, open_price)
            low = min(low, close, open_price)

            data.iloc[i, data.columns.get_loc('high')] = high
            data.iloc[i, data.columns.get_loc('low')] = low
            data.iloc[i, data.columns.get_loc('open')] = open_price

        # Generate volume
        data['volume'] = np.random.uniform(1000000, 10000000, n_days)

        logger.info(f"Generated {len(data)} days of sample trading data")
        logger.info(f"Price range: ${data['close'].min():.0f} - ${data['close'].max():.0f}")
        logger.info("")

        # Train LSTM model
        logger.info("🧠 Training LSTM Model...")
        lstm_results = await trainer.train_lstm_model(
            data,
            sequence_length=30,  # 30 days of history
            hidden_size=64,
            num_layers=2,
            epochs=10,  # Reduced for demo
            batch_size=32
        )

        logger.info("✅ LSTM training completed!")
        logger.info("")

        # Train Transformer model
        logger.info("🔄 Training Transformer Model...")
        transformer_results = await trainer.train_transformer_model(
            data,
            sequence_length=30,
            num_heads=4,
            num_layers=2,
            epochs=5,  # Reduced for demo
            batch_size=16
        )

        logger.info("✅ Transformer training completed!")
        logger.info("")

        # Compare results
        logger.info("📊 MODEL COMPARISON:")
        logger.info("=" * 40)

        models = [
            ("LSTM", lstm_results),
            ("Transformer", transformer_results)
        ]

        for name, results in models:
            metrics = results['validation_metrics']
            logger.info(f"{name}:")
            logger.info(".4f"            logger.info(".4f"            logger.info(".4f"            logger.info("")

        # Performance insights
        logger.info("💡 PERFORMANCE INSIGHTS:")
        logger.info("-" * 30)
        logger.info("• LSTM: Good for sequential patterns, faster training")
        logger.info("• Transformer: Better for long-range dependencies, more parameters")
        logger.info("• GPU acceleration provides significant speedup for both")
        logger.info("• Consider ensemble methods combining both architectures")
        logger.info("")

        logger.info("🎉 GPU-accelerated model training example completed!")

        return {
            "lstm_results": lstm_results,
            "transformer_results": transformer_results
        }

    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise

async def main():
    """Main example function."""
    try:
        results = await example_gpu_trading_model()

        # Save models if needed
        logger.info("💾 Models trained and ready for inference!")
        logger.info("Use the trained models for real-time trading predictions.")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())