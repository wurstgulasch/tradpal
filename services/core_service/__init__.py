"""
TradPal Core Services

This module contains the core trading and analysis services including:
- GPU acceleration for ML operations
- Parallel processing engine
- Technical analysis and indicators
- Signal generation and risk management
"""

from .gpu_accelerator import (
    GPUAccelerator,
    NeuralNetworkGPU,
    GPUMatrixOperations,
    GPUFeatureEngineering,
    get_gpu_accelerator,
    is_gpu_available,
    get_optimal_device,
    create_gpu_lstm_model,
    create_gpu_transformer_model,
    train_gpu_model
)

from .parallel_engine import ParallelProcessingEngine

__all__ = [
    # GPU Acceleration
    'GPUAccelerator',
    'NeuralNetworkGPU',
    'GPUMatrixOperations',
    'GPUFeatureEngineering',
    'get_gpu_accelerator',
    'is_gpu_available',
    'get_optimal_device',
    'create_gpu_lstm_model',
    'create_gpu_transformer_model',
    'train_gpu_model',

    # Parallel Processing
    'ParallelProcessingEngine',
]