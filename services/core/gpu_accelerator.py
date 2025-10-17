#!/usr/bin/env python3
"""
GPU Acceleration Support for TradPal

This module provides GPU acceleration capabilities for machine learning models,
neural networks, and computationally intensive operations using CUDA and cuDNN.
"""

import logging
import os
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from contextlib import contextmanager
import pandas as pd

logger = logging.getLogger(__name__)

class GPUAccelerator:
    """
    GPU Acceleration Manager for ML operations and heavy computations.

    Provides automatic GPU detection, memory management, and optimized
    operations for neural networks, matrix computations, and data processing.
    """

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        self.current_device = 0
        self.memory_threshold = 0.8  # Use up to 80% of GPU memory

        if self.cuda_available:
            logger.info(f"✅ GPU acceleration available: {self.device_count} CUDA device(s)")
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  Device {i}: {props.name} ({props.total_memory // 1024**3}GB)")
        else:
            logger.warning("❌ CUDA not available, falling back to CPU")

    def get_optimal_device(self) -> torch.device:
        """Get the optimal device for computations."""
        if self.cuda_available:
            # Select GPU with most available memory
            max_memory = 0
            best_device = 0

            for i in range(self.device_count):
                torch.cuda.synchronize(i)  # Wait for all operations to complete
                memory_info = torch.cuda.mem_get_info(i)
                available_memory = memory_info[0]  # Free memory in bytes

                if available_memory > max_memory:
                    max_memory = available_memory
                    best_device = i

            return torch.device(f'cuda:{best_device}')
        else:
            return torch.device('cpu')

    def get_device_memory_info(self, device_id: Optional[int] = None) -> Dict[str, float]:
        """Get memory information for a specific device."""
        if not self.cuda_available:
            return {"total": 0, "free": 0, "used": 0, "utilization": 0}

        device = device_id if device_id is not None else self.current_device
        total, free = torch.cuda.mem_get_info(device)
        used = total - free
        utilization = used / total if total > 0 else 0

        return {
            "total_gb": total / (1024**3),
            "free_gb": free / (1024**3),
            "used_gb": used / (1024**3),
            "utilization": utilization
        }

    @contextmanager
    def memory_efficient_context(self, device: Optional[torch.device] = None):
        """Context manager for memory-efficient GPU operations."""
        if device is None:
            device = self.get_optimal_device()

        original_device = torch.cuda.current_device() if self.cuda_available else None

        try:
            if self.cuda_available:
                torch.cuda.set_device(device)
                # Enable cuDNN auto-tuner for optimized performance
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True

            yield device

        finally:
            if self.cuda_available and original_device is not None:
                torch.cuda.set_device(original_device)

    def optimize_tensor_operations(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor for GPU operations."""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)

        if self.cuda_available:
            device = self.get_optimal_device()
            tensor = tensor.to(device)

            # Use mixed precision if beneficial
            if tensor.dtype == torch.float32 and tensor.numel() > 1000:
                tensor = tensor.half()  # Convert to float16 for faster computation

        return tensor

    def batch_process_tensors(self, tensors: List[torch.Tensor],
                            batch_size: int = 32) -> List[torch.Tensor]:
        """Process tensors in batches to optimize memory usage."""
        if not self.cuda_available or len(tensors) <= batch_size:
            return tensors

        results = []
        device = self.get_optimal_device()

        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i + batch_size]

            # Process batch on GPU
            with self.memory_efficient_context(device):
                processed_batch = [self.optimize_tensor_operations(t) for t in batch]

            results.extend(processed_batch)

            # Force garbage collection between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

class NeuralNetworkGPU:
    """
    GPU-accelerated neural network operations for trading models.
    """

    def __init__(self, gpu_accelerator: GPUAccelerator):
        self.gpu = gpu_accelerator
        self.device = self.gpu.get_optimal_device()

    def create_lstm_model(self, input_size: int, hidden_size: int,
                         num_layers: int, output_size: int) -> torch.nn.Module:
        """Create GPU-optimized LSTM model."""

        class LSTMModel(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers

                self.lstm = torch.nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=0.2 if num_layers > 1 else 0
                )
                self.fc = torch.nn.Linear(hidden_size, output_size)
                self.dropout = torch.nn.Dropout(0.2)

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

                out, _ = self.lstm(x, (h0, c0))
                out = self.dropout(out[:, -1, :])  # Take last time step
                out = self.fc(out)
                return out

        model = LSTMModel(input_size, hidden_size, num_layers, output_size)

        if self.gpu.cuda_available:
            model = model.to(self.device)

        return model

    def create_transformer_model(self, input_size: int, num_heads: int,
                               num_layers: int, output_size: int) -> torch.nn.Module:
        """Create GPU-optimized Transformer model."""

        class TransformerModel(torch.nn.Module):
            def __init__(self, input_size, num_heads, num_layers, output_size):
                super(TransformerModel, self).__init__()

                self.input_projection = torch.nn.Linear(input_size, 512)
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=512, nhead=num_heads, dim_feedforward=2048,
                    dropout=0.1, batch_first=True
                )
                self.transformer_encoder = torch.nn.TransformerEncoder(
                    encoder_layer, num_layers=num_layers
                )
                self.output_projection = torch.nn.Linear(512, output_size)
                self.dropout = torch.nn.Dropout(0.1)

            def forward(self, x):
                x = self.input_projection(x)
                x = self.transformer_encoder(x)
                x = self.dropout(x.mean(dim=1))  # Global average pooling
                x = self.output_projection(x)
                return x

        model = TransformerModel(input_size, num_heads, num_layers, output_size)

        if self.gpu.cuda_available:
            model = model.to(self.device)

        return model

    def train_model(self, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
                   val_loader: Optional[torch.utils.data.DataLoader] = None,
                   epochs: int = 100, learning_rate: float = 0.001) -> Dict[str, List[float]]:
        """Train neural network model with GPU acceleration."""

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Use mixed precision training if available
        scaler = torch.cuda.amp.GradScaler() if self.gpu.cuda_available else None

        train_losses = []
        val_losses = []

        with self.gpu.memory_efficient_context():
            for epoch in range(epochs):
                model.train()
                epoch_train_loss = 0

                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()

                    if scaler is not None:
                        # Mixed precision training
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_x)
                            loss = criterion(outputs, batch_y)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard training
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                    epoch_train_loss += loss.item()

                avg_train_loss = epoch_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # Validation
                if val_loader is not None:
                    model.eval()
                    epoch_val_loss = 0

                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            batch_x = batch_x.to(self.device)
                            batch_y = batch_y.to(self.device)

                            outputs = model(batch_x)
                            loss = criterion(outputs, batch_y)
                            epoch_val_loss += loss.item()

                    avg_val_loss = epoch_val_loss / len(val_loader)
                    val_losses.append(avg_val_loss)

                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")

        return {"train_losses": train_losses, "val_losses": val_losses}

class GPUMatrixOperations:
    """
    GPU-accelerated matrix operations for technical analysis and data processing.
    """

    def __init__(self, gpu_accelerator: GPUAccelerator):
        self.gpu = gpu_accelerator
        self.device = self.gpu.get_optimal_device()

    def matrix_multiply_gpu(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Perform matrix multiplication on GPU."""
        if not self.gpu.cuda_available:
            return np.dot(a, b)

        with self.gpu.memory_efficient_context():
            a_tensor = torch.tensor(a, dtype=torch.float32).to(self.device)
            b_tensor = torch.tensor(b, dtype=torch.float32).to(self.device)

            result_tensor = torch.mm(a_tensor, b_tensor)
            result = result_tensor.cpu().numpy()

        return result

    def batch_matrix_operations(self, matrices: List[np.ndarray],
                              operation: str = "multiply") -> List[np.ndarray]:
        """Perform batch matrix operations on GPU."""
        if not self.gpu.cuda_available or len(matrices) < 2:
            return matrices

        with self.gpu.memory_efficient_context():
            # Convert to tensors
            tensor_matrices = [torch.tensor(m, dtype=torch.float32).to(self.device)
                             for m in matrices]

            results = []
            for i in range(len(tensor_matrices) - 1):
                if operation == "multiply":
                    result = torch.mm(tensor_matrices[i], tensor_matrices[i + 1])
                elif operation == "add":
                    result = tensor_matrices[i] + tensor_matrices[i + 1]
                else:
                    result = tensor_matrices[i]  # No operation

                results.append(result.cpu().numpy())

        return results

class GPUFeatureEngineering:
    """
    GPU-accelerated feature engineering operations.
    """

    def __init__(self, gpu_accelerator: GPUAccelerator):
        self.gpu = gpu_accelerator
        self.device = self.gpu.get_optimal_device()

    def compute_technical_indicators_gpu(self, data: pd.DataFrame,
                                       indicators: List[str]) -> pd.DataFrame:
        """Compute technical indicators using GPU acceleration."""
        if not self.gpu.cuda_available:
            # Fallback to CPU computation
            return self._compute_indicators_cpu(data, indicators)

        with self.gpu.memory_efficient_context():
            # Convert price data to tensor
            prices = torch.tensor(data[['open', 'high', 'low', 'close', 'volume']].values,
                                dtype=torch.float32).to(self.device)

            result_data = data.copy()

            for indicator in indicators:
                if indicator == 'sma':
                    result_data['sma_20'] = self._gpu_sma(prices[:, 3], 20)  # Close prices
                elif indicator == 'ema':
                    result_data['ema_20'] = self._gpu_ema(prices[:, 3], 20)
                elif indicator == 'rsi':
                    result_data['rsi_14'] = self._gpu_rsi(prices[:, 3], 14)
                elif indicator == 'macd':
                    macd, signal, hist = self._gpu_macd(prices[:, 3])
                    result_data['macd'] = macd
                    result_data['macd_signal'] = signal
                    result_data['macd_hist'] = hist

            return result_data

    def _gpu_sma(self, prices: torch.Tensor, period: int) -> np.ndarray:
        """Compute Simple Moving Average on GPU."""
        weights = torch.ones(period, dtype=torch.float32).to(self.device) / period
        sma = torch.nn.functional.conv1d(
            prices.unsqueeze(0).unsqueeze(0),
            weights.unsqueeze(0).unsqueeze(0),
            padding=period-1
        ).squeeze()
        return sma[period-1:].cpu().numpy()

    def _gpu_ema(self, prices: torch.Tensor, period: int) -> np.ndarray:
        """Compute Exponential Moving Average on GPU."""
        alpha = 2 / (period + 1)
        ema = torch.zeros_like(prices)

        # Compute EMA iteratively (could be optimized further)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

        return ema.cpu().numpy()

    def _gpu_rsi(self, prices: torch.Tensor, period: int) -> np.ndarray:
        """Compute RSI on GPU."""
        # Calculate price changes
        delta = prices[1:] - prices[:-1]
        gain = torch.where(delta > 0, delta, torch.zeros_like(delta))
        loss = torch.where(delta < 0, -delta, torch.zeros_like(delta))

        # Calculate average gain/loss
        avg_gain = torch.zeros_like(gain)
        avg_loss = torch.zeros_like(loss)

        # First average
        avg_gain[period-1] = gain[:period].mean()
        avg_loss[period-1] = loss[:period].mean()

        # Subsequent averages
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period

        rs = avg_gain[period-1:] / (avg_loss[period-1:] + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        return rsi.cpu().numpy()

    def _gpu_macd(self, prices: torch.Tensor, fast_period: int = 12,
                  slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute MACD on GPU."""
        fast_ema = torch.tensor(self._gpu_ema(prices, fast_period), dtype=torch.float32).to(self.device)
        slow_ema = torch.tensor(self._gpu_ema(prices, slow_period), dtype=torch.float32).to(self.device)

        macd = fast_ema - slow_ema
        signal = torch.tensor(self._gpu_ema(macd, signal_period), dtype=torch.float32).to(self.device)
        histogram = macd - signal

        return macd.cpu().numpy(), signal.cpu().numpy(), histogram.cpu().numpy()

    def _compute_indicators_cpu(self, data: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """Fallback CPU computation for technical indicators."""
        # This would use the existing CPU-based indicator calculations
        logger.warning("Using CPU computation for technical indicators")
        return data  # Placeholder

# Global GPU accelerator instance
gpu_accelerator = GPUAccelerator()
neural_net_gpu = NeuralNetworkGPU(gpu_accelerator)
gpu_matrix_ops = GPUMatrixOperations(gpu_accelerator)
gpu_feature_engineering = GPUFeatureEngineering(gpu_accelerator)

def get_gpu_accelerator() -> GPUAccelerator:
    """Get the global GPU accelerator instance."""
    return gpu_accelerator

def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return gpu_accelerator.cuda_available

def get_optimal_device() -> torch.device:
    """Get the optimal compute device."""
    return gpu_accelerator.get_optimal_device()

def create_gpu_lstm_model(input_size: int, hidden_size: int,
                         num_layers: int, output_size: int) -> torch.nn.Module:
    """Create a GPU-accelerated LSTM model."""
    return neural_net_gpu.create_lstm_model(input_size, hidden_size, num_layers, output_size)

def create_gpu_transformer_model(input_size: int, num_heads: int,
                               num_layers: int, output_size: int) -> torch.nn.Module:
    """Create a GPU-accelerated Transformer model."""
    return neural_net_gpu.create_transformer_model(input_size, num_heads, num_layers, output_size)

def train_gpu_model(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
                   val_loader: Optional[torch.utils.data.DataLoader] = None,
                   epochs: int = 100, learning_rate: float = 0.001) -> Dict[str, List[float]]:
    """Train a model with GPU acceleration."""
    return neural_net_gpu.train_model(model, train_loader, val_loader, epochs, learning_rate)