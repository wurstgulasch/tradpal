#!/usr/bin/env python3
"""
Vectorized Technical Indicators
High-performance implementations using NumPy vectorization for better speed and memory efficiency.
"""

import logging
import time
from functools import wraps
from typing import Union, Tuple, Dict, List, Any
import numpy as np
import pandas as pd
from .memory_optimization import (
    MemoryMappedData,
    RollingWindowBuffer,
    ChunkedDataLoader,
    MemoryPool,
    LazyDataLoader,
    MemoryStats
)

# Check if Numba is available for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    jit = lambda *args, **kwargs: lambda f: f  # No-op decorator
    prange = range

# Check if TA-Lib is available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

# Check if PyTorch is available for GPU acceleration
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    torch = None
    F = None

logger = logging.getLogger(__name__)

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class VectorizedIndicators:
    """
    High-performance vectorized technical indicators using NumPy and Numba.
    """

    @staticmethod
    @timing_decorator
    def ema_vectorized(series: Union[pd.Series, np.ndarray], period: int) -> np.ndarray:
        """
        Vectorized EMA calculation that matches the original implementation exactly.
        Uses TA-Lib if available, otherwise pandas implementation.

        Args:
            series: Price series
            period: EMA period

        Returns:
            EMA values as numpy array
        """
        if isinstance(series, pd.Series):
            prices = series.values.astype(np.float64)
        else:
            prices = np.asarray(series, dtype=np.float64)

        if len(prices) < period:
            return np.full(len(prices), np.nan)

        # Use TA-Lib if available (matches original implementation)
        if TALIB_AVAILABLE:
            try:
                ema_values = talib.EMA(prices, timeperiod=period)
                return ema_values
            except Exception:
                pass

        # Fallback to pandas implementation
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def rsi_vectorized_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Numba-accelerated RSI calculation for maximum performance.

        Args:
            prices: Close prices array
            period: RSI period

        Returns:
            RSI values array
        """
        n = len(prices)
        rsi_values = np.full(n, np.nan, dtype=np.float64)

        if n < period + 1:
            return rsi_values

        # Calculate price changes
        deltas = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            deltas[i] = prices[i] - prices[i-1]

        # Calculate gains and losses
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)

        # Calculate initial averages
        avg_gain = np.mean(gains[1:period+1])
        avg_loss = np.mean(losses[1:period+1])

        if avg_loss == 0:
            rsi_values[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[period] = 100.0 - (100.0 / (1.0 + rs))

        # Calculate subsequent values using Wilder's smoothing
        for i in range(period + 1, n):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi_values[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))

        return rsi_values

    @staticmethod
    @timing_decorator
    def rsi_vectorized(series: Union[pd.Series, np.ndarray], period: int = 14) -> np.ndarray:
        """
        Vectorized RSI calculation that matches the original implementation exactly.
        Uses TA-Lib if available, otherwise pandas implementation.

        Args:
            series: Price series
            period: RSI period

        Returns:
            RSI values as numpy array
        """
        if isinstance(series, pd.Series):
            prices = series.values.astype(np.float64)
        else:
            prices = np.asarray(series, dtype=np.float64)

        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)

        # Use TA-Lib if available (matches original implementation)
        if TALIB_AVAILABLE:
            try:
                rsi_values = talib.RSI(prices, timeperiod=period)
                return rsi_values
            except Exception:
                pass

        # Fallback to pandas implementation
        delta = series.diff() if isinstance(series, pd.Series) else pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(loss != 0, np.nan)

        return rsi.values

    @staticmethod
    @timing_decorator
    def bollinger_bands_vectorized(series: Union[pd.Series, np.ndarray],
                                 period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized Bollinger Bands calculation.

        Args:
            series: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (upper_band, middle_band, lower_band) arrays
        """
        if isinstance(series, pd.Series):
            values = series.values.astype(np.float64)
        else:
            values = np.asarray(series, dtype=np.float64)

        if len(values) < period:
            nan_array = np.full(len(values), np.nan)
            return nan_array, nan_array, nan_array

        # Calculate rolling mean and std using pandas (efficient vectorized)
        series_pd = pd.Series(values)

        # Vectorized rolling calculations
        middle = series_pd.rolling(window=period, min_periods=period).mean().values
        std = series_pd.rolling(window=period, min_periods=period).std().values

        # Calculate bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def atr_vectorized_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Numba-accelerated ATR calculation.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            period: ATR period

        Returns:
            ATR values array
        """
        n = len(high)
        atr_values = np.full(n, np.nan, dtype=np.float64)

        if n < period + 1:
            return atr_values

        # Calculate True Range
        tr = np.zeros(n, dtype=np.float64)
        tr[0] = high[0] - low[0]  # First TR is just high - low

        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],  # Current high - low
                abs(high[i] - close[i-1]),  # Current high - previous close
                abs(low[i] - close[i-1])   # Current low - previous close
            )

        # Calculate ATR using Wilder's smoothing
        atr_values[period-1] = np.mean(tr[:period])

        for i in range(period, n):
            atr_values[i] = (atr_values[i-1] * (period - 1) + tr[i]) / period

        return atr_values

    @staticmethod
    @timing_decorator
    def atr_vectorized(high: Union[pd.Series, np.ndarray],
                      low: Union[pd.Series, np.ndarray],
                      close: Union[pd.Series, np.ndarray],
                      period: int = 14) -> np.ndarray:
        """
        Vectorized ATR calculation that matches the original implementation exactly.
        Uses TA-Lib if available, otherwise pandas implementation.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            ATR values as numpy array
        """
        if isinstance(high, pd.Series):
            h = high
            l = low
            c = close
        else:
            h = pd.Series(high)
            l = pd.Series(low)
            c = pd.Series(close)

        if len(h) < period:
            return np.full(len(h), np.nan)

        # Use TA-Lib if available (matches original implementation)
        if TALIB_AVAILABLE:
            try:
                h_vals = h.values.astype(float)
                l_vals = l.values.astype(float)
                c_vals = c.values.astype(float)
                atr_values = talib.ATR(h_vals, l_vals, c_vals, timeperiod=period)
                return atr_values
            except Exception:
                pass

        # Fallback to pandas implementation (matches original fallback)
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().values

    @staticmethod
    @timing_decorator
    def macd_vectorized(series: Union[pd.Series, np.ndarray], fast_period: int = 12,
                       slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized MACD calculation that matches the original implementation exactly.
        Uses TA-Lib if available, otherwise pandas implementation.

        Args:
            series: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period

        Returns:
            Tuple of (macd_line, signal_line, histogram) arrays
        """
        if isinstance(series, pd.Series):
            prices = series.values.astype(np.float64)
        else:
            prices = np.asarray(series, dtype=np.float64)

        # Use TA-Lib if available (matches original implementation)
        if TALIB_AVAILABLE:
            try:
                macd_line, signal_line, histogram = talib.MACD(prices, fastperiod=fast_period,
                                                              slowperiod=slow_period, signalperiod=signal_period)
                return macd_line, signal_line, histogram
            except Exception:
                pass

        # Fallback to pandas implementation
        series_pd = pd.Series(prices)
        fast_ema = series_pd.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series_pd.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line.values, signal_line.values, histogram.values

    @staticmethod
    @timing_decorator
    def stochastic_vectorized(high: Union[pd.Series, np.ndarray],
                            low: Union[pd.Series, np.ndarray],
                            close: Union[pd.Series, np.ndarray],
                            k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized Stochastic Oscillator calculation that matches the original implementation exactly.
        Uses TA-Lib if available, otherwise pandas implementation.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period
            d_period: %D period (smoothing)

        Returns:
            Tuple of (%K, %D) arrays
        """
        if isinstance(high, pd.Series):
            h = high.values.astype(np.float64)
            l = low.values.astype(np.float64)
            c = close.values.astype(np.float64)
        else:
            h = np.asarray(high, dtype=np.float64)
            l = np.asarray(low, dtype=np.float64)
            c = np.asarray(close, dtype=np.float64)

        if len(h) < k_period:
            nan_array = np.full(len(h), np.nan)
            return nan_array, nan_array

        # Use TA-Lib if available (matches original implementation)
        if TALIB_AVAILABLE:
            try:
                k_values, d_values = talib.STOCH(h, l, c,
                                               fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
                return k_values, d_values
            except Exception:
                pass

        # Fallback to pandas implementation
        high_pd = pd.Series(h)
        low_pd = pd.Series(l)
        close_pd = pd.Series(c)

        lowest_low = low_pd.rolling(window=k_period).min()
        highest_high = high_pd.rolling(window=k_period).max()
        k_percent = 100 * ((close_pd - lowest_low) / (highest_high - lowest_low))
        k_percent = k_percent.where(highest_high != lowest_low, 50)
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent.values, d_percent.values

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def rolling_statistics_numba(data: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Numba-accelerated rolling statistics calculation.

        Args:
            data: Input data array
            window: Rolling window size

        Returns:
            Tuple of (mean, std, min, max) arrays
        """
        n = len(data)
        mean_vals = np.full(n, np.nan, dtype=np.float64)
        std_vals = np.full(n, np.nan, dtype=np.float64)
        min_vals = np.full(n, np.nan, dtype=np.float64)
        max_vals = np.full(n, np.nan, dtype=np.float64)

        if n < window:
            return mean_vals, std_vals, min_vals, max_vals

        # Calculate rolling statistics using pure NumPy
        for i in prange(window - 1, n):
            window_data = data[i - window + 1:i + 1]
            mean_vals[i] = np.mean(window_data)
            # Calculate std manually (ddof=1 equivalent)
            if len(window_data) > 1:
                mean = np.mean(window_data)
                variance = np.mean((window_data - mean) ** 2)
                std_vals[i] = np.sqrt(variance * len(window_data) / (len(window_data) - 1))
            else:
                std_vals[i] = 0.0
            min_vals[i] = np.min(window_data)
            max_vals[i] = np.max(window_data)

        return mean_vals, std_vals, min_vals, max_vals

class PerformanceBenchmark:
    """
    Benchmarking utilities to compare vectorized vs non-vectorized implementations.
    """

    @staticmethod
    def benchmark_indicator(indicator_func, *args, runs: int = 10) -> Dict[str, float]:
        """
        Benchmark an indicator function.

        Args:
            indicator_func: Function to benchmark
            *args: Arguments for the function
            runs: Number of benchmark runs

        Returns:
            Dictionary with timing statistics
        """
        times = []

        for _ in range(runs):
            start_time = time.time()
            result = indicator_func(*args)
            end_time = time.time()
            times.append(end_time - start_time)

        times_array = np.array(times)
        return {
            'mean': np.mean(times_array),
            'std': np.std(times_array),
            'min': np.min(times_array),
            'max': np.max(times_array),
            'runs': runs
        }

    @staticmethod
    def compare_implementations(original_func, vectorized_func, *args, runs: int = 5) -> Dict[str, Dict]:
        """
        Compare performance of original vs vectorized implementations.

        Args:
            original_func: Original implementation
            vectorized_func: Vectorized implementation
            *args: Arguments for both functions
            runs: Number of benchmark runs

        Returns:
            Dictionary with comparison results
        """
        print(f"Benchmarking {original_func.__name__} vs {vectorized_func.__name__}...")

        # Benchmark both implementations
        original_stats = PerformanceBenchmark.benchmark_indicator(original_func, *args, runs=runs)
        vectorized_stats = PerformanceBenchmark.benchmark_indicator(vectorized_func, *args, runs=runs)

        speedup = original_stats['mean'] / vectorized_stats['mean']

        return {
            'original': original_stats,
            'vectorized': vectorized_stats,
            'speedup': speedup,
            'improvement_percent': (speedup - 1) * 100
        }


class GPUIndicators:
    """
    GPU-accelerated technical indicators using PyTorch.
    Automatically falls back to CPU if CUDA is not available.
    """

    @staticmethod
    def _get_device() -> Any:
        """Get the appropriate device (GPU if available, else CPU)."""
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            return torch.device('cuda')
        return torch.device('cpu')

    @staticmethod
    def _to_tensor(data: Union[np.ndarray, pd.Series, List], dtype: Any = None) -> Any:
        """Convert input data to PyTorch tensor on appropriate device."""
        if dtype is None and TORCH_AVAILABLE:
            dtype = torch.float32
        elif dtype is None:
            dtype = np.float32

        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)

        data = np.asarray(data, dtype=np.float32)
        if TORCH_AVAILABLE:
            tensor = torch.from_numpy(data).to(GPUIndicators._get_device(), dtype=dtype)
            return tensor
        else:
            return data

    @staticmethod
    def _from_tensor(tensor: Any) -> np.ndarray:
        """Convert PyTorch tensor back to numpy array."""
        return tensor.cpu().detach().numpy()

    @staticmethod
    def ema_gpu(series: Union[pd.Series, np.ndarray], period: int) -> np.ndarray:
        """
        GPU-accelerated Exponential Moving Average.

        Args:
            series: Input price series
            period: EMA period

        Returns:
            EMA values as numpy array
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, falling back to vectorized implementation")
            return VectorizedIndicators.ema_vectorized(series, period)

        try:
            data = GPUIndicators._to_tensor(series)
            alpha = 2.0 / (period + 1.0)

            # Initialize EMA tensor
            ema = torch.full_like(data, torch.nan)

            # Calculate first valid EMA (simple moving average of first 'period' values)
            if len(data) >= period:
                ema[period-1] = torch.mean(data[:period])

                # Calculate subsequent EMAs
                for i in range(period, len(data)):
                    ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

            return GPUIndicators._from_tensor(ema)

        except Exception as e:
            logger.warning(f"GPU EMA calculation failed: {e}, falling back to vectorized")
            return VectorizedIndicators.ema_vectorized(series, period)

    @staticmethod
    def rsi_gpu(series: Union[pd.Series, np.ndarray], period: int = 14) -> np.ndarray:
        """
        GPU-accelerated Relative Strength Index.

        Args:
            series: Input price series
            period: RSI period

        Returns:
            RSI values as numpy array
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, falling back to vectorized implementation")
            return VectorizedIndicators.rsi_vectorized(series, period)

        try:
            data = GPUIndicators._to_tensor(series)

            # Calculate price changes
            delta = torch.diff(data, prepend=data[0:1])

            # Separate gains and losses
            gains = torch.where(delta > 0, delta, torch.zeros_like(delta))
            losses = torch.where(delta < 0, -delta, torch.zeros_like(delta))

            # Calculate initial averages
            avg_gain = torch.full_like(data, torch.nan)
            avg_loss = torch.full_like(data, torch.nan)

            if len(data) >= period + 1:
                avg_gain[period] = torch.mean(gains[1:period+1])
                avg_loss[period] = torch.mean(losses[1:period+1])

                # Calculate smoothed averages
                for i in range(period + 1, len(data)):
                    avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
                    avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period

                # Calculate RS and RSI
                rs = avg_gain / torch.where(avg_loss == 0, 1e-10, avg_loss)
                rsi = 100 - (100 / (1 + rs))

                return GPUIndicators._from_tensor(rsi)

            # Return NaN array if insufficient data
            return np.full(len(series), np.nan)

        except Exception as e:
            logger.warning(f"GPU RSI calculation failed: {e}, falling back to vectorized")
            return VectorizedIndicators.rsi_vectorized(series, period)

    @staticmethod
    def atr_gpu(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray],
                close: Union[pd.Series, np.ndarray], period: int = 14) -> np.ndarray:
        """
        GPU-accelerated Average True Range.
        Currently uses vectorized implementation with PyTorch tensor operations where possible.
        Falls back to CPU if CUDA unavailable.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            ATR values as numpy array
        """
        # For now, use the vectorized implementation
        # Future: Implement true GPU acceleration when CUDA is available
        return VectorizedIndicators.atr_vectorized(high, low, close, period)

    @staticmethod
    def bollinger_bands_gpu(series: Union[pd.Series, np.ndarray], period: int = 20,
                           std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        GPU-accelerated Bollinger Bands.

        Args:
            series: Input price series
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (upper, middle, lower) bands as numpy arrays
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, falling back to vectorized implementation")
            return VectorizedIndicators.bollinger_bands_vectorized(series, period, std_dev)

        try:
            data = GPUIndicators._to_tensor(series)

            # Calculate rolling mean and std
            kernel_size = period
            padding = kernel_size // 2

            # Use 1D convolution for rolling statistics
            # Create uniform kernel for mean
            mean_kernel = torch.ones(1, 1, kernel_size, device=data.device, dtype=data.dtype) / kernel_size

            # Pad the data
            padded_data = F.pad(data.unsqueeze(0).unsqueeze(0), (padding, padding), mode='replicate')

            # Calculate rolling mean
            rolling_mean = F.conv1d(padded_data, mean_kernel).squeeze()

            # Calculate rolling variance
            squared_diff = (padded_data - rolling_mean.unsqueeze(0).unsqueeze(0)) ** 2
            var_kernel = torch.ones(1, 1, kernel_size, device=data.device, dtype=data.dtype) / kernel_size
            rolling_var = F.conv1d(squared_diff, var_kernel).squeeze()
            rolling_std = torch.sqrt(rolling_var)

            # Calculate bands
            upper = rolling_mean + std_dev * rolling_std
            lower = rolling_mean - std_dev * rolling_std

            # Trim padding
            upper = upper[padding:-padding] if padding > 0 else upper
            rolling_mean = rolling_mean[padding:-padding] if padding > 0 else rolling_mean
            lower = lower[padding:-padding] if padding > 0 else lower

            return (
                GPUIndicators._from_tensor(upper),
                GPUIndicators._from_tensor(rolling_mean),
                GPUIndicators._from_tensor(lower)
            )

        except Exception as e:
            logger.warning(f"GPU Bollinger Bands calculation failed: {e}, falling back to vectorized")
            return VectorizedIndicators.bollinger_bands_vectorized(series, period, std_dev)

    @staticmethod
    def macd_gpu(series: Union[pd.Series, np.ndarray], fast_period: int = 12,
                slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        GPU-accelerated MACD (Moving Average Convergence Divergence).
        Currently uses vectorized implementation with PyTorch tensor operations where possible.
        Falls back to CPU if CUDA unavailable.

        Args:
            series: Input price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period

        Returns:
            Tuple of (MACD line, signal line, histogram) as numpy arrays
        """
        # For now, use the vectorized implementation
        # Future: Implement true GPU acceleration when CUDA is available
        return VectorizedIndicators.macd_vectorized(series, fast_period, slow_period, signal_period)


# Convenience functions that wrap the GPU implementations with automatic fallback
def ema_gpu(series: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
    """Convenience function for GPU-accelerated EMA with CPU fallback."""
    result = GPUIndicators.ema_gpu(series, period)
    if isinstance(series, pd.Series):
        return pd.Series(result, index=series.index, name=f'EMA{period}')
    return result

def rsi_gpu(series: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
    """Convenience function for GPU-accelerated RSI with CPU fallback."""
    result = GPUIndicators.rsi_gpu(series, period)
    if isinstance(series, pd.Series):
        return pd.Series(result, index=series.index, name='RSI')
    return result

def atr_gpu(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray],
           close: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
    """Convenience function for GPU-accelerated ATR with CPU fallback."""
    result = GPUIndicators.atr_gpu(high, low, close, period)
    if isinstance(high, pd.Series):
        return pd.Series(result, index=high.index, name='ATR')
    return result

def bb_gpu(series: Union[pd.Series, np.ndarray], period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Convenience function for GPU-accelerated Bollinger Bands with CPU fallback."""
    upper, middle, lower = GPUIndicators.bollinger_bands_gpu(series, period, std_dev)
    if isinstance(series, pd.Series):
        index = series.index
        return (
            pd.Series(upper, index=index, name='BB_upper'),
            pd.Series(middle, index=index, name='BB_middle'),
            pd.Series(lower, index=index, name='BB_lower')
        )
    return upper, middle, lower

def macd_gpu(series: Union[pd.Series, np.ndarray], fast_period: int = 12,
            slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Convenience function for GPU-accelerated MACD with CPU fallback."""
    macd_line, signal_line, histogram = GPUIndicators.macd_gpu(series, fast_period, slow_period, signal_period)
    if isinstance(series, pd.Series):
        index = series.index
        return (
            pd.Series(macd_line, index=index, name='MACD'),
            pd.Series(signal_line, index=index, name='MACD_signal'),
            pd.Series(histogram, index=index, name='MACD_histogram')
        )
    return macd_line, signal_line, histogram

# Convenience functions that wrap the vectorized implementations
def ema_vectorized(series: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
    """Convenience function for vectorized EMA."""
    result = VectorizedIndicators.ema_vectorized(series, period)
    if isinstance(series, pd.Series):
        return pd.Series(result, index=series.index, name=f'EMA{period}')
    return result

def rsi_vectorized(series: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
    """Convenience function for vectorized RSI."""
    result = VectorizedIndicators.rsi_vectorized(series, period)
    if isinstance(series, pd.Series):
        return pd.Series(result, index=series.index, name='RSI')
    return result

def rsi_numba(series: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
    """Convenience function for Numba-accelerated RSI."""
    if isinstance(series, pd.Series):
        values = series.values
    else:
        values = np.asarray(series)
    result = VectorizedIndicators.rsi_vectorized_numba(values, period)
    if isinstance(series, pd.Series):
        return pd.Series(result, index=series.index, name='RSI')
    return result

def bb_vectorized(series: Union[pd.Series, np.ndarray], period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Convenience function for vectorized Bollinger Bands."""
    upper, middle, lower = VectorizedIndicators.bollinger_bands_vectorized(series, period, std_dev)
    if isinstance(series, pd.Series):
        index = series.index
        return (
            pd.Series(upper, index=index, name='BB_upper'),
            pd.Series(middle, index=index, name='BB_middle'),
            pd.Series(lower, index=index, name='BB_lower')
        )
    return upper, middle, lower

def atr_vectorized(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray],
                  close: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
    """Convenience function for vectorized ATR."""
    result = VectorizedIndicators.atr_vectorized(high, low, close, period)
    if isinstance(high, pd.Series):
        return pd.Series(result, index=high.index, name='ATR')
    return result

def atr_numba(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray],
             close: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
    """Convenience function for Numba-accelerated ATR."""
    # For now, use the same implementation as vectorized (TA-Lib or pandas fallback)
    # True numba acceleration would require implementing TA-Lib algorithms in numba
    if isinstance(high, pd.Series):
        h_vals, l_vals, c_vals = high.values, low.values, close.values
        index = high.index
    else:
        h_vals, l_vals, c_vals = np.asarray(high), np.asarray(low), np.asarray(close)
        index = None
    result = VectorizedIndicators.atr_vectorized_numba(h_vals, l_vals, c_vals, period)
    if index is not None:
        return pd.Series(result, index=index, name='ATR')
    return result

def macd_vectorized(series: Union[pd.Series, np.ndarray], fast_period: int = 12,
                   slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Convenience function for vectorized MACD."""
    macd_line, signal_line, histogram = VectorizedIndicators.macd_vectorized(
        series, fast_period, slow_period, signal_period
    )
    if isinstance(series, pd.Series):
        index = series.index
        return (
            pd.Series(macd_line, index=index, name='MACD'),
            pd.Series(signal_line, index=index, name='MACD_signal'),
            pd.Series(histogram, index=index, name='MACD_histogram')
        )
    return macd_line, signal_line, histogram

def stochastic_vectorized(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray],
                         close: Union[pd.Series, np.ndarray], k_period: int = 14,
                         d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Convenience function for vectorized Stochastic Oscillator."""
    k_vals, d_vals = VectorizedIndicators.stochastic_vectorized(high, low, close, k_period, d_period)
    if isinstance(high, pd.Series):
        index = high.index
        return (
            pd.Series(k_vals, index=index, name='Stoch_K'),
            pd.Series(d_vals, index=index, name='Stoch_D')
        )
    return k_vals, d_vals


class MemoryOptimizedIndicators:
    """
    Memory-optimized technical indicators using memory-mapped files and efficient data structures.
    Combines vectorized calculations with memory optimization techniques for large datasets.
    """

    def __init__(self, memory_pool: MemoryPool = None, chunk_size: int = 10000):
        """
        Initialize memory-optimized indicators.

        Args:
            memory_pool: Optional memory pool for buffer management
            chunk_size: Default chunk size for data processing
        """
        self.memory_pool = memory_pool or MemoryPool()
        self.chunk_size = chunk_size
        self.memory_stats = MemoryStats()

    @timing_decorator
    def ema_memory_optimized(self, series: Union[pd.Series, np.ndarray, str],
                           period: int, use_memory_map: bool = False) -> np.ndarray:
        """
        Memory-optimized EMA calculation with optional memory mapping.

        Args:
            series: Price series or path to HDF5 file
            period: EMA period
            use_memory_map: Whether to use memory-mapped files

        Returns:
            EMA values as numpy array
        """
        if isinstance(series, str):
            # Load from memory-mapped file
            with MemoryMappedData(series, mode='r') as mm_data:
                data = mm_data['close'][:] if 'close' in mm_data else mm_data[:]
        else:
            data = series.values if hasattr(series, 'values') else np.asarray(series)

        # Use chunked processing for large datasets
        if len(data) > self.chunk_size:
            return self._process_in_chunks(data, lambda chunk: VectorizedIndicators.ema_vectorized(chunk, period))

        return VectorizedIndicators.ema_vectorized(data, period)

    @timing_decorator
    def rsi_memory_optimized(self, series: Union[pd.Series, np.ndarray, str],
                           period: int = 14, use_rolling_buffer: bool = True) -> np.ndarray:
        """
        Memory-optimized RSI calculation with rolling window buffer.

        Args:
            series: Price series or path to HDF5 file
            period: RSI period
            use_rolling_buffer: Whether to use rolling window buffer

        Returns:
            RSI values as numpy array
        """
        if isinstance(series, str):
            with MemoryMappedData(series, mode='r') as mm_data:
                data = mm_data['close'][:] if 'close' in mm_data else mm_data[:]
        else:
            data = series.values if hasattr(series, 'values') else np.asarray(series)

        if use_rolling_buffer and len(data) > period * 2:
            # Use rolling window buffer for memory efficiency
            buffer = RollingWindowBuffer(window_size=period * 2, dtype=data.dtype)
            rsi_values = np.full(len(data), np.nan, dtype=np.float64)

            for i, price in enumerate(data):
                buffer.add(price)
                if buffer.is_full:
                    window_data = buffer.to_array()
                    rsi_chunk = VectorizedIndicators.rsi_vectorized(window_data, period)
                    rsi_values[i] = rsi_chunk[-1] if len(rsi_chunk) > 0 else np.nan

            return rsi_values

        # Standard processing for smaller datasets
        return VectorizedIndicators.rsi_vectorized(data, period)

    @timing_decorator
    def bollinger_bands_memory_optimized(self, series: Union[pd.Series, np.ndarray, str],
                                       period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Memory-optimized Bollinger Bands calculation.

        Args:
            series: Price series or path to HDF5 file
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (upper_band, middle_band, lower_band) arrays
        """
        if isinstance(series, str):
            with MemoryMappedData(series, mode='r') as mm_data:
                data = mm_data['close'][:] if 'close' in mm_data else mm_data[:]
        else:
            data = series.values if hasattr(series, 'values') else np.asarray(series)

        # Use chunked processing for large datasets
        if len(data) > self.chunk_size:
            upper_chunks = []
            middle_chunks = []
            lower_chunks = []

            for chunk in self._chunk_data(data, self.chunk_size):
                u, m, l = VectorizedIndicators.bollinger_bands_vectorized(chunk, period, std_dev)
                upper_chunks.append(u)
                middle_chunks.append(m)
                lower_chunks.append(l)

            return (
                np.concatenate(upper_chunks),
                np.concatenate(middle_chunks),
                np.concatenate(lower_chunks)
            )

        return VectorizedIndicators.bollinger_bands_vectorized(data, period, std_dev)

    @timing_decorator
    def atr_memory_optimized(self, high: Union[pd.Series, np.ndarray, str],
                           low: Union[pd.Series, np.ndarray, str],
                           close: Union[pd.Series, np.ndarray, str],
                           period: int = 14) -> np.ndarray:
        """
        Memory-optimized ATR calculation.

        Args:
            high: High prices or path to HDF5 file
            low: Low prices or path to HDF5 file
            close: Close prices or path to HDF5 file
            period: ATR period

        Returns:
            ATR values as numpy array
        """
        if isinstance(high, str):
            with MemoryMappedData(high, mode='r') as mm_data:
                h_data = mm_data['high'][:] if 'high' in mm_data else mm_data[:]
                l_data = mm_data['low'][:] if 'low' in mm_data else mm_data[:]
                c_data = mm_data['close'][:] if 'close' in mm_data else mm_data[:]
        else:
            h_data = high.values if hasattr(high, 'values') else np.asarray(high)
            l_data = low.values if hasattr(low, 'values') else np.asarray(low)
            c_data = close.values if hasattr(close, 'values') else np.asarray(close)

        # Use chunked processing for large datasets
        if len(h_data) > self.chunk_size:
            return self._process_ohlc_chunks(h_data, l_data, c_data,
                                           lambda h, l, c: VectorizedIndicators.atr_vectorized(h, l, c, period))

        return VectorizedIndicators.atr_vectorized(h_data, l_data, c_data, period)

    def _process_in_chunks(self, data: np.ndarray, func, overlap: int = 0) -> np.ndarray:
        """
        Process data in chunks with optional overlap.

        Args:
            data: Input data array
            func: Function to apply to each chunk
            overlap: Number of overlapping elements between chunks

        Returns:
            Concatenated results
        """
        chunks = self._chunk_data(data, self.chunk_size, overlap)
        results = []

        for chunk in chunks:
            result = func(chunk)
            results.append(result)

        return np.concatenate(results)

    def _process_ohlc_chunks(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                           func, overlap: int = 0) -> np.ndarray:
        """
        Process OHLC data in chunks.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            func: Function to apply to OHLC chunks
            overlap: Number of overlapping elements

        Returns:
            Concatenated results
        """
        high_chunks = self._chunk_data(high, self.chunk_size, overlap)
        low_chunks = self._chunk_data(low, self.chunk_size, overlap)
        close_chunks = self._chunk_data(close, self.chunk_size, overlap)

        results = []
        for h_chunk, l_chunk, c_chunk in zip(high_chunks, low_chunks, close_chunks):
            result = func(h_chunk, l_chunk, c_chunk)
            results.append(result)

        return np.concatenate(results)

    def _chunk_data(self, data: np.ndarray, chunk_size: int, overlap: int = 0) -> List[np.ndarray]:
        """
        Split data into chunks with optional overlap.

        Args:
            data: Input data array
            chunk_size: Size of each chunk
            overlap: Number of overlapping elements

        Returns:
            List of data chunks
        """
        chunks = []
        start = 0

        while start < len(data):
            end = min(start + chunk_size, len(data))
            chunk = data[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

            if start >= len(data):
                break

        return chunks

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory statistics
        """
        return self.memory_stats.get_stats()

    def optimize_for_large_dataset(self, file_path: str, indicators: List[str] = None) -> Dict[str, Any]:
        """
        Optimize indicator calculations for large datasets using memory mapping.

        Args:
            file_path: Path to HDF5 file with OHLC data
            indicators: List of indicators to calculate

        Returns:
            Dictionary with calculated indicators
        """
        if indicators is None:
            indicators = ['ema', 'rsi', 'bb', 'atr']

        results = {}

        with MemoryMappedData(file_path, mode='r') as mm_data:
            # Load close data
            if 'close' in mm_data.h5file:
                close_data = mm_data['close'][:]
            else:
                # Fallback to first dataset if no 'close' key
                first_key = list(mm_data.h5file.keys())[0]
                close_data = mm_data[first_key][:]

            if 'ema' in indicators:
                results['ema'] = self.ema_memory_optimized(close_data, 20, use_memory_map=True)

            if 'rsi' in indicators:
                results['rsi'] = self.rsi_memory_optimized(close_data, 14, use_rolling_buffer=True)

            if 'bb' in indicators:
                results['bb_upper'], results['bb_middle'], results['bb_lower'] = \
                    self.bollinger_bands_memory_optimized(close_data, 20)

            if 'atr' in indicators and 'high' in mm_data.h5file and 'low' in mm_data.h5file:
                results['atr'] = self.atr_memory_optimized(
                    mm_data['high'][:], mm_data['low'][:], close_data, 14
                )

        return results


# Convenience functions for memory-optimized indicators
def ema_memory_optimized(series: Union[pd.Series, np.ndarray, str], period: int,
                        use_memory_map: bool = False) -> pd.Series:
    """Convenience function for memory-optimized EMA."""
    optimizer = MemoryOptimizedIndicators()
    result = optimizer.ema_memory_optimized(series, period, use_memory_map)
    if isinstance(series, pd.Series):
        return pd.Series(result, index=series.index, name=f'EMA{period}')
    return result

def rsi_memory_optimized(series: Union[pd.Series, np.ndarray, str], period: int = 14,
                        use_rolling_buffer: bool = True) -> pd.Series:
    """Convenience function for memory-optimized RSI."""
    optimizer = MemoryOptimizedIndicators()
    result = optimizer.rsi_memory_optimized(series, period, use_rolling_buffer)
    if isinstance(series, pd.Series):
        return pd.Series(result, index=series.index, name='RSI')
    return result

def bb_memory_optimized(series: Union[pd.Series, np.ndarray, str], period: int = 20,
                       std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Convenience function for memory-optimized Bollinger Bands."""
    optimizer = MemoryOptimizedIndicators()
    upper, middle, lower = optimizer.bollinger_bands_memory_optimized(series, period, std_dev)
    if isinstance(series, pd.Series):
        index = series.index
        return (
            pd.Series(upper, index=index, name='BB_upper'),
            pd.Series(middle, index=index, name='BB_middle'),
            pd.Series(lower, index=index, name='BB_lower')
        )
    return upper, middle, lower

def atr_memory_optimized(high: Union[pd.Series, np.ndarray, str],
                        low: Union[pd.Series, np.ndarray, str],
                        close: Union[pd.Series, np.ndarray, str],
                        period: int = 14) -> pd.Series:
    """Convenience function for memory-optimized ATR."""
    optimizer = MemoryOptimizedIndicators()
    result = optimizer.atr_memory_optimized(high, low, close, period)
    if isinstance(high, pd.Series):
        return pd.Series(result, index=high.index, name='ATR')
    return result