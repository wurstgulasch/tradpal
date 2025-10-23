#!/usr/bin/env python3
"""
Memory Optimization Module
High-performance memory management for large financial datasets and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import h5py
import pickle
import logging
from collections import deque
import gc
import psutil
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

class MemoryStats:
    """Memory usage statistics and monitoring."""

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
            'cpu_percent': process.cpu_percent(interval=0.1)
        }

    @staticmethod
    def log_memory_usage(operation: str = ""):
        """Log current memory usage."""
        stats = MemoryStats.get_memory_usage()
        logger.info(f"Memory usage {operation}: RSS={stats['rss']:.1f}MB, "
                   f"Available={stats['available']:.1f}GB ({stats['percent']:.1f}%)")

    def get_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        return MemoryStats.get_memory_usage()


class MemoryMappedData:
    """
    Memory-mapped data storage for large financial datasets.
    Supports HDF5 format for efficient storage and access.
    """

    def __init__(self, file_path: Union[str, Path], mode: str = 'r'):
        """
        Initialize memory-mapped data storage.

        Args:
            file_path: Path to HDF5 file
            mode: File mode ('r', 'w', 'a')
        """
        self.file_path = Path(file_path)
        self.mode = mode
        self.h5file: Optional[h5py.File] = None
        self._lock = threading.RLock()

        self._open_file()

    def _open_file(self):
        """Open HDF5 file with memory mapping."""
        try:
            self.h5file = h5py.File(self.file_path, self.mode, libver='latest')
            logger.info(f"Opened memory-mapped file: {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to open file {self.file_path}: {e}")
            raise

    def close(self):
        """Close the memory-mapped file."""
        with self._lock:
            if self.h5file:
                self.h5file.close()
                self.h5file = None
                logger.info(f"Closed memory-mapped file: {self.file_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getitem__(self, key: str):
        """Get dataset by key."""
        with self._lock:
            if self.h5file is None:
                raise RuntimeError("File not open")
            return self.h5file[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in the HDF5 file."""
        with self._lock:
            if self.h5file is None:
                return False
            return key in self.h5file

    def store_dataframe(self, df: pd.DataFrame, key: str, compression: str = 'gzip',
                       compression_opts: int = 6):
        """
        Store pandas DataFrame in memory-mapped format.

        Args:
            df: DataFrame to store
            key: HDF5 key/path
            compression: Compression algorithm
            compression_opts: Compression level
        """
        with self._lock:
            if self.h5file is None:
                raise RuntimeError("File not open")

            # Convert datetime index to string for HDF5 compatibility
            df_copy = df.copy()
            if isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy.index = df_copy.index.astype(str)

            # Store as HDF5 table
            self.h5file.create_dataset(
                key,
                data=df_copy.values,
                compression=compression,
                compression_opts=compression_opts
            )

            # Store column names and index
            self.h5file.attrs[f'{key}_columns'] = [col.encode('utf-8') for col in df_copy.columns]
            if hasattr(df_copy.index, 'name') and df_copy.index.name:
                self.h5file.attrs[f'{key}_index_name'] = df_copy.index.name.encode('utf-8')
            self.h5file.attrs[f'{key}_index'] = df_copy.index.astype(str).values.astype('S')

    def load_dataframe(self, key: str) -> pd.DataFrame:
        """
        Load DataFrame from memory-mapped storage.

        Args:
            key: HDF5 key/path

        Returns:
            Loaded DataFrame
        """
        with self._lock:
            if self.h5file is None:
                raise RuntimeError("File not open")

            if key not in self.h5file:
                raise KeyError(f"Key '{key}' not found in file")

            dataset = self.h5file[key]
            data = dataset[:]

            # Reconstruct DataFrame
            columns = []
            for col in self.h5file.attrs[f'{key}_columns']:
                if isinstance(col, bytes):
                    columns.append(col.decode('utf-8'))
                else:
                    columns.append(col)

            index_values = []
            for idx in self.h5file.attrs[f'{key}_index']:
                if isinstance(idx, bytes):
                    index_values.append(idx.decode('utf-8'))
                else:
                    index_values.append(idx)

            df = pd.DataFrame(data, columns=columns, index=index_values)

            # Try to convert index back to datetime
            try:
                df.index = pd.to_datetime(df.index)
            except (ValueError, TypeError):
                pass

            # Restore index name
            if f'{key}_index_name' in self.h5file.attrs:
                index_name_attr = self.h5file.attrs[f'{key}_index_name']
                if isinstance(index_name_attr, bytes):
                    df.index.name = index_name_attr.decode('utf-8')
                else:
                    df.index.name = index_name_attr

            return df

    def create_dataset(self, key: str, data: np.ndarray, **kwargs):
        """
        Create a dataset in the HDF5 file.

        Args:
            key: Dataset key
            data: Data to store
            **kwargs: Additional HDF5 dataset options
        """
        with self._lock:
            if self.h5file is None:
                raise RuntimeError("File not open")

            self.h5file.create_dataset(key, data=data, **kwargs)

    def load_array(self, key: str) -> np.ndarray:
        """
        Load array from memory-mapped storage.

        Args:
            key: HDF5 key/path

        Returns:
            Loaded array
        """
        with self._lock:
            if self.h5file is None:
                raise RuntimeError("File not open")

            if key not in self.h5file:
                raise KeyError(f"Key '{key}' not found in file")

            return self.h5file[key][:]

    def get_chunked_data(self, key: str, chunk_size: int = 10000,
                        start_idx: int = 0, end_idx: Optional[int] = None) -> np.ndarray:
        """
        Load data in chunks to manage memory usage.

        Args:
            key: HDF5 key/path
            chunk_size: Size of each chunk
            start_idx: Starting index
            end_idx: Ending index (None for all remaining)

        Returns:
            Chunked data array
        """
        with self._lock:
            if self.h5file is None:
                raise RuntimeError("File not open")

            dataset = self.h5file[key]
            total_rows = dataset.shape[0]

            if end_idx is None:
                end_idx = total_rows

            # Load data in chunks
            chunks = []
            for i in range(start_idx, min(end_idx, total_rows), chunk_size):
                chunk_end = min(i + chunk_size, end_idx, total_rows)
                chunk = dataset[i:chunk_end]
                chunks.append(chunk)

            return np.concatenate(chunks, axis=0) if chunks else np.array([])


class RollingWindowBuffer:
    """
    Memory-efficient rolling window buffer for technical indicators.
    Uses deque for O(1) append/pop operations.
    """

    def __init__(self, window_size: int, dtype: np.dtype = np.float64):
        """
        Initialize rolling window buffer.

        Args:
            window_size: Size of the rolling window
            dtype: Data type for the buffer
        """
        self.window_size = window_size
        self.dtype = dtype
        self.buffer = deque(maxlen=window_size)
        self._array_cache = None
        self._cache_valid = False

    def append(self, value: Union[float, np.ndarray]):
        """Add value to the buffer."""
        self.buffer.append(float(value))
        self._cache_valid = False

    def add(self, value: Union[float, np.ndarray]):
        """Alias for append method."""
        self.append(value)

    def extend(self, values: Union[List, np.ndarray]):
        """Add multiple values to the buffer."""
        for value in values:
            self.buffer.append(float(value))
        self._cache_valid = False

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self._array_cache = None
        self._cache_valid = False

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.buffer) == self.window_size

    @property
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)

    def to_array(self) -> np.ndarray:
        """Convert buffer to numpy array with caching."""
        if not self._cache_valid or self._array_cache is None:
            self._array_cache = np.array(list(self.buffer), dtype=self.dtype)
            self._cache_valid = True
        return self._array_cache

    def mean(self) -> float:
        """Calculate rolling mean."""
        if not self.buffer:
            return np.nan
        return np.mean(self.to_array())

    def std(self, ddof: int = 1) -> float:
        """Calculate rolling standard deviation."""
        if len(self.buffer) < 2:
            return np.nan
        return np.std(self.to_array(), ddof=ddof)

    def min(self) -> float:
        """Get minimum value in window."""
        if not self.buffer:
            return np.nan
        return np.min(self.to_array())

    def max(self) -> float:
        """Get maximum value in window."""
        if not self.buffer:
            return np.nan
        return np.max(self.to_array())

    def sum(self) -> float:
        """Calculate sum of values in window."""
        if not self.buffer:
            return np.nan
        return np.sum(self.to_array())


class ChunkedDataLoader:
    """
    Chunked data loader for processing large datasets in memory-efficient chunks.
    """

    def __init__(self, chunk_size: int = 10000, max_workers: int = 4):
        """
        Initialize chunked data loader.

        Args:
            chunk_size: Default chunk size
            max_workers: Maximum number of worker threads
        """
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_in_chunks(self, data: Union[np.ndarray, pd.DataFrame],
                         processor_func, *args, **kwargs):
        """
        Process data in chunks using a processor function.

        Args:
            data: Input data
            processor_func: Function to process each chunk
            *args: Additional arguments for processor
            **kwargs: Keyword arguments for processor

        Returns:
            List of processed chunks
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = np.asarray(data)

        total_rows = len(data_array)
        futures = []

        # Submit chunk processing tasks
        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            chunk = data_array[start_idx:end_idx]

            future = self.executor.submit(processor_func, chunk, *args, **kwargs)
            futures.append(future)

        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                results.append(None)

        return results

    def load_chunks(self, file_path: Union[str, Path], dataset_key: str):
        """
        Load data in chunks from memory-mapped file.

        Args:
            file_path: Path to HDF5 file
            dataset_key: Key of the dataset to load

        Yields:
            Data chunks
        """
        with MemoryMappedData(file_path, 'r') as mm_data:
            if dataset_key not in mm_data.h5file:
                raise KeyError(f"Dataset '{dataset_key}' not found in file")

            dataset = mm_data[dataset_key]
            total_rows = dataset.shape[0]

            for start_idx in range(0, total_rows, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, total_rows)
                yield dataset[start_idx:end_idx]

    def load_historical_data(self, file_path: Union[str, Path],
                           symbol: str, timeframe: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical data in chunks from memory-mapped storage.

        Args:
            file_path: Path to data file
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Loaded DataFrame
        """
        with MemoryMappedData(file_path, 'r') as mm_data:
            key = f"{symbol}_{timeframe}"

            if key not in mm_data.h5file:
                raise KeyError(f"Data key '{key}' not found")

            # Load data in chunks
            data = mm_data.get_chunked_data(key, chunk_size=self.chunk_size)
            df = pd.DataFrame(data)

            # Apply date filters if specified
            if start_date or end_date:
                # Assume first column is datetime
                if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
                    if start_date:
                        df = df[df.iloc[:, 0] >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df.iloc[:, 0] <= pd.to_datetime(end_date)]

            return df


class MemoryPool:
    """
    Memory pool for reusable buffers to reduce memory fragmentation.
    """

    def __init__(self, max_size: int = 10, initial_size: int = 0, buffer_size: int = 1024):
        """
        Initialize memory pool.

        Args:
            max_size: Maximum number of buffers to keep
            initial_size: Initial number of buffers to allocate
            buffer_size: Size of each buffer
        """
        self.max_size = max_size
        self.buffer_size = buffer_size
        self.pool = {}
        self._lock = threading.RLock()

        # Pre-allocate buffers if requested
        for _ in range(initial_size):
            self.return_buffer(np.zeros(buffer_size))

    def get_buffer(self, shape: Tuple, dtype: np.dtype = np.float64) -> np.ndarray:
        """
        Get a reusable buffer from the pool.

        Args:
            shape: Buffer shape
            dtype: Data type

        Returns:
            Reusable buffer
        """
        with self._lock:
            # Ensure dtype is np.dtype
            if not isinstance(dtype, np.dtype):
                dtype = np.dtype(dtype)
            key = (shape, dtype)

            if key in self.pool and len(self.pool[key]) > 0:
                buffer = self.pool[key].pop()
                # Clear buffer
                buffer.fill(0)
                return buffer
            else:
                # Create new buffer
                return np.zeros(shape, dtype=dtype)

    def allocate(self, shape: Tuple = (1024,), dtype: np.dtype = np.float64) -> np.ndarray:
        """
        Alias for get_buffer method for compatibility.

        Args:
            shape: Buffer shape (default: (1024,))
            dtype: Data type (default: float64)

        Returns:
            Reusable buffer
        """
        return self.get_buffer(shape, dtype)

    def return_buffer(self, buffer: np.ndarray):
        """
        Return buffer to the pool for reuse.

        Args:
            buffer: Buffer to return
        """
        with self._lock:
            # Ensure dtype is np.dtype
            dtype = buffer.dtype if isinstance(buffer.dtype, np.dtype) else np.dtype(buffer.dtype)
            key = (buffer.shape, dtype)

            if key not in self.pool:
                self.pool[key] = []

            if len(self.pool[key]) < self.max_size:
                self.pool[key].append(buffer)

    def clear(self):
        """Clear all buffers from the pool."""
        with self._lock:
            self.pool.clear()
            gc.collect()

    @property
    def available_buffers(self) -> int:
        """Get number of available buffers in pool."""
        with self._lock:
            return sum(len(buffers) for buffers in self.pool.values())

    def release(self, buffer: np.ndarray):
        """Alias for return_buffer method."""
        self.return_buffer(buffer)


class LazyDataLoader:
    """
    Lazy data loader with LRU caching for historical data.
    """

    def __init__(self, cache_size: int = 100, cache_dir: Optional[Path] = None, max_cache_size: int = 100):
        """
        Initialize lazy data loader.

        Args:
            cache_size: Maximum number of cached items
            cache_dir: Directory for persistent cache
            max_cache_size: Alias for cache_size
        """
        self.cache_size = cache_size if cache_size != 100 else max_cache_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory_cache = {}
        self._access_times = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache(self):
        """Alias for _memory_cache for compatibility."""
        return self._memory_cache

    def load_data(self, file_path: str, key: str) -> pd.DataFrame:
        """
        Load data with caching.

        Args:
            file_path: Path to data file
            key: Data key

        Returns:
            Loaded DataFrame
        """
        cache_key = f"{file_path}:{key}"

        # Check memory cache first
        if cache_key in self._memory_cache:
            self._access_times[cache_key] = time.time()
            return self._memory_cache[cache_key]

        # Check persistent cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{hash(cache_key)}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    self._memory_cache[cache_key] = data
                    self._access_times[cache_key] = time.time()
                    return data
                except Exception:
                    pass  # Fall through to loading

        # Load from source
        with MemoryMappedData(file_path, 'r') as mm_data:
            data = mm_data.load_dataframe(key)

        # Cache in memory
        self._memory_cache[cache_key] = data
        self._access_times[cache_key] = time.time()

        # Cache persistently if directory specified
        if self.cache_dir:
            cache_file = self.cache_dir / f"{hash(cache_key)}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                logger.warning(f"Failed to cache data persistently: {e}")

        # Maintain cache size
        self._evict_old_cache()

        return data

    def save_data(self, key: str, data: Any):
        """
        Save data to cache.

        Args:
            key: Cache key
            data: Data to cache
        """
        cache_key = f"saved:{key}"

        # Cache in memory
        self._memory_cache[cache_key] = data
        self._access_times[cache_key] = time.time()

        # Cache persistently if directory specified
        if self.cache_dir:
            cache_file = self.cache_dir / f"{hash(cache_key)}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                logger.warning(f"Failed to cache data persistently: {e}")

        # Maintain cache size
        self._evict_old_cache()

    def _evict_old_cache(self):
        """Evict least recently used items from cache."""
        if len(self._memory_cache) > self.cache_size:
            # Sort by access time
            sorted_keys = sorted(self._access_times.keys(),
                               key=lambda k: self._access_times[k])

            # Remove oldest items
            to_remove = sorted_keys[:len(sorted_keys) - self.cache_size]
            for key in to_remove:
                del self._memory_cache[key]
                del self._access_times[key]

                # Remove persistent cache if exists
                if self.cache_dir:
                    cache_file = self.cache_dir / f"{hash(key)}.pkl"
                    if cache_file.exists():
                        cache_file.unlink()

    def clear_cache(self):
        """Clear all cached data."""
        self._memory_cache.clear()
        self._access_times.clear()

        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()

    def preload_data(self, file_path: str, keys: List[str]):
        """
        Preload multiple data keys into cache.

        Args:
            file_path: Path to data file
            keys: List of data keys to preload
        """
        for key in keys:
            self.load_data(file_path, key)


# Convenience functions for memory optimization
def create_memory_mapped_store(file_path: Union[str, Path]) -> MemoryMappedData:
    """Create a memory-mapped data store."""
    return MemoryMappedData(file_path, 'w')

def load_memory_mapped_data(file_path: Union[str, Path]) -> MemoryMappedData:
    """Load data from memory-mapped store."""
    return MemoryMappedData(file_path, 'r')

def create_rolling_buffer(window_size: int) -> RollingWindowBuffer:
    """Create a rolling window buffer."""
    return RollingWindowBuffer(window_size)

def create_chunked_loader(chunk_size: int = 10000) -> ChunkedDataLoader:
    """Create a chunked data loader."""
    return ChunkedDataLoader(chunk_size)

def create_memory_pool(max_size: int = 10) -> MemoryPool:
    """Create a memory pool."""
    return MemoryPool(max_size)

def create_lazy_loader(cache_size: int = 100, cache_dir: Optional[Path] = None) -> LazyDataLoader:
    """Create a lazy data loader with caching."""
    return LazyDataLoader(cache_size, cache_dir)