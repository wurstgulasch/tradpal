# Performance optimization settings
# Memory management, caching, GPU acceleration, and parallel processing

import os
from typing import Dict, Any

# Memory Management Configuration
MEMORY_OPTIMIZATION_ENABLED = os.getenv('MEMORY_OPTIMIZATION_ENABLED', 'true').lower() == 'true'  # Enable memory optimization
MEMORY_LIMIT_GB = float(os.getenv('MEMORY_LIMIT_GB', '8.0'))  # Memory limit in GB
MEMORY_WARNING_THRESHOLD = float(os.getenv('MEMORY_WARNING_THRESHOLD', '0.8'))  # Memory warning threshold (80%)
MEMORY_CRITICAL_THRESHOLD = float(os.getenv('MEMORY_CRITICAL_THRESHOLD', '0.9'))  # Memory critical threshold (90%)
MEMORY_CLEANUP_INTERVAL_SECONDS = int(os.getenv('MEMORY_CLEANUP_INTERVAL_SECONDS', '300'))  # Memory cleanup interval

# Memory-Mapped Data Configuration
MEMORY_MAPPED_ENABLED = os.getenv('MEMORY_MAPPED_ENABLED', 'true').lower() == 'true'  # Enable memory-mapped files
MEMORY_MAPPED_CACHE_SIZE_MB = int(os.getenv('MEMORY_MAPPED_CACHE_SIZE_MB', '1024'))  # Memory-mapped cache size
MEMORY_MAPPED_MAX_FILES = int(os.getenv('MEMORY_MAPPED_MAX_FILES', '100'))  # Max memory-mapped files
MEMORY_MAPPED_FILE_SIZE_WARNING_MB = int(os.getenv('MEMORY_MAPPED_FILE_SIZE_WARNING_MB', '500'))  # File size warning threshold

# Rolling Window Buffer Configuration
ROLLING_WINDOW_ENABLED = os.getenv('ROLLING_WINDOW_ENABLED', 'true').lower() == 'true'  # Enable rolling window buffers
ROLLING_WINDOW_MAX_SIZE = int(os.getenv('ROLLING_WINDOW_MAX_SIZE', '10000'))  # Max rolling window size
ROLLING_WINDOW_CHUNK_SIZE = int(os.getenv('ROLLING_WINDOW_CHUNK_SIZE', '1000'))  # Rolling window chunk size
ROLLING_WINDOW_MEMORY_LIMIT_MB = int(os.getenv('ROLLING_WINDOW_MEMORY_LIMIT_MB', '512'))  # Memory limit for rolling windows

# Chunked Processing Configuration
CHUNKED_PROCESSING_ENABLED = os.getenv('CHUNKED_PROCESSING_ENABLED', 'true').lower() == 'true'  # Enable chunked processing
CHUNK_SIZE_DEFAULT = int(os.getenv('CHUNK_SIZE_DEFAULT', '1000'))  # Default chunk size
CHUNK_SIZE_MAX = int(os.getenv('CHUNK_SIZE_MAX', '10000'))  # Maximum chunk size
CHUNK_OVERLAP_SIZE = int(os.getenv('CHUNK_OVERLAP_SIZE', '50'))  # Chunk overlap size for continuity
CHUNK_PROCESSING_WORKERS = int(os.getenv('CHUNK_PROCESSING_WORKERS', '4'))  # Number of chunk processing workers

# Parallel Processing Configuration
PARALLEL_PROCESSING_ENABLED = os.getenv('PARALLEL_PROCESSING_ENABLED', 'true').lower() == 'true'  # Enable parallel processing
MAX_WORKERS_DEFAULT = int(os.getenv('MAX_WORKERS_DEFAULT', '4'))  # Default max workers
MAX_WORKERS = MAX_WORKERS_DEFAULT  # Alias for backward compatibility
MAX_WORKERS_IO = int(os.getenv('MAX_WORKERS_IO', '8'))  # Max workers for I/O operations
MAX_WORKERS_CPU = int(os.getenv('MAX_WORKERS_CPU', '4'))  # Max workers for CPU operations
PARALLEL_THRESHOLD = int(os.getenv('PARALLEL_THRESHOLD', '1000'))  # Minimum data size for parallel processing

# GPU Acceleration Configuration
GPU_ACCELERATION_ENABLED = os.getenv('GPU_ACCELERATION_ENABLED', 'true').lower() == 'true'  # Enable GPU acceleration
GPU_DEVICE_ID = int(os.getenv('GPU_DEVICE_ID', '0'))  # GPU device ID
GPU_MEMORY_FRACTION = float(os.getenv('GPU_MEMORY_FRACTION', '0.8'))  # GPU memory fraction to use
GPU_ALLOW_GROWTH = os.getenv('GPU_ALLOW_GROWTH', 'true').lower() == 'true'  # Allow GPU memory growth
GPU_FORCE_CPU_FALLBACK = os.getenv('GPU_FORCE_CPU_FALLBACK', 'false').lower() == 'true'  # Force CPU fallback if GPU unavailable

# Caching Configuration
CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'  # Enable caching
CACHE_BACKEND = os.getenv('CACHE_BACKEND', 'redis')  # Cache backend ('redis', 'memory', 'disk')
CACHE_TTL_DEFAULT = int(os.getenv('CACHE_TTL_DEFAULT', '3600'))  # Default cache TTL in seconds
CACHE_MAX_SIZE_MB = int(os.getenv('CACHE_MAX_SIZE_MB', '512'))  # Max cache size in MB
CACHE_COMPRESSION_ENABLED = os.getenv('CACHE_COMPRESSION_ENABLED', 'true').lower() == 'true'  # Enable cache compression

# Redis Cache Configuration
REDIS_CACHE_ENABLED = os.getenv('REDIS_CACHE_ENABLED', 'true').lower() == 'true'  # Enable Redis caching
REDIS_CACHE_HOST = os.getenv('REDIS_CACHE_HOST', 'localhost')  # Redis cache host
REDIS_CACHE_PORT = int(os.getenv('REDIS_CACHE_PORT', '6379'))  # Redis cache port
REDIS_CACHE_DB = int(os.getenv('REDIS_CACHE_DB', '1'))  # Redis cache database
REDIS_CACHE_PASSWORD = os.getenv('REDIS_CACHE_PASSWORD', '')  # Redis cache password
REDIS_CACHE_KEY_PREFIX = os.getenv('REDIS_CACHE_KEY_PREFIX', 'tradpal_cache:')  # Redis cache key prefix
REDIS_CACHE_POOL_SIZE = int(os.getenv('REDIS_CACHE_POOL_SIZE', '10'))  # Redis connection pool size

# Disk Cache Configuration
DISK_CACHE_ENABLED = os.getenv('DISK_CACHE_ENABLED', 'true').lower() == 'true'  # Enable disk caching
DISK_CACHE_DIR = os.getenv('DISK_CACHE_DIR', 'cache/disk_cache/')  # Disk cache directory
DISK_CACHE_MAX_SIZE_GB = float(os.getenv('DISK_CACHE_MAX_SIZE_GB', '10.0'))  # Max disk cache size in GB
DISK_CACHE_COMPRESSION_LEVEL = int(os.getenv('DISK_CACHE_COMPRESSION_LEVEL', '6'))  # Disk cache compression level

# Memory Cache Configuration
MEMORY_CACHE_ENABLED = os.getenv('MEMORY_CACHE_ENABLED', 'true').lower() == 'true'  # Enable memory caching
MEMORY_CACHE_MAX_ITEMS = int(os.getenv('MEMORY_CACHE_MAX_ITEMS', '10000'))  # Max memory cache items
MEMORY_CACHE_TTL_DEFAULT = int(os.getenv('MEMORY_CACHE_TTL_DEFAULT', '1800'))  # Memory cache TTL

# Vectorization Configuration
VECTORIZATION_ENABLED = os.getenv('VECTORIZATION_ENABLED', 'true').lower() == 'true'  # Enable vectorization
VECTORIZATION_THRESHOLD = int(os.getenv('VECTORIZATION_THRESHOLD', '1000'))  # Minimum size for vectorization
VECTORIZATION_NUMPY_THRESHOLD = int(os.getenv('VECTORIZATION_NUMPY_THRESHOLD', '100'))  # NumPy vectorization threshold
VECTORIZATION_PANDAS_THRESHOLD = int(os.getenv('VECTORIZATION_PANDAS_THRESHOLD', '1000'))  # Pandas vectorization threshold

# NumPy Configuration
NUMPY_THREADS = int(os.getenv('NUMPY_THREADS', '4'))  # NumPy thread count
NUMPY_FLOAT_PRECISION = os.getenv('NUMPY_FLOAT_PRECISION', 'float64')  # NumPy float precision

# Pandas Configuration
PANDAS_OPTIMIZATION_ENABLED = os.getenv('PANDAS_OPTIMIZATION_ENABLED', 'true').lower() == 'true'  # Enable pandas optimizations
PANDAS_COPY_ON_WRITE = os.getenv('PANDAS_COPY_ON_WRITE', 'true').lower() == 'true'  # Enable copy-on-write
PANDAS_MAX_ROWS_DISPLAY = int(os.getenv('PANDAS_MAX_ROWS_DISPLAY', '1000'))  # Max rows to display
PANDAS_MAX_COLS_DISPLAY = int(os.getenv('PANDAS_MAX_COLS_DISPLAY', '100'))  # Max columns to display

# HDF5 Configuration
HDF5_ENABLED = os.getenv('HDF5_ENABLED', 'true').lower() == 'true'  # Enable HDF5 storage
HDF5_CHUNK_SIZE = int(os.getenv('HDF5_CHUNK_SIZE', '10000'))  # HDF5 chunk size
HDF5_COMPRESSION_LEVEL = int(os.getenv('HDF5_COMPRESSION_LEVEL', '6'))  # HDF5 compression level
HDF5_CACHE_SIZE_MB = int(os.getenv('HDF5_CACHE_SIZE_MB', '128'))  # HDF5 cache size

# Database Performance Configuration
DB_CONNECTION_POOL_SIZE = int(os.getenv('DB_CONNECTION_POOL_SIZE', '10'))  # Database connection pool size
DB_CONNECTION_TIMEOUT = int(os.getenv('DB_CONNECTION_TIMEOUT', '30'))  # Database connection timeout
DB_QUERY_TIMEOUT = int(os.getenv('DB_QUERY_TIMEOUT', '60'))  # Database query timeout
DB_MAX_RETRIES = int(os.getenv('DB_MAX_RETRIES', '3'))  # Database max retries
DB_RETRY_DELAY = float(os.getenv('DB_RETRY_DELAY', '1.0'))  # Database retry delay

# Async I/O Configuration
ASYNC_IO_ENABLED = os.getenv('ASYNC_IO_ENABLED', 'true').lower() == 'true'  # Enable async I/O
ASYNC_IO_MAX_CONCURRENT = int(os.getenv('ASYNC_IO_MAX_CONCURRENT', '100'))  # Max concurrent async operations
ASYNC_IO_TIMEOUT = int(os.getenv('ASYNC_IO_TIMEOUT', '30'))  # Async I/O timeout
ASYNC_IO_BUFFER_SIZE = int(os.getenv('ASYNC_IO_BUFFER_SIZE', '8192'))  # Async I/O buffer size

# Network Performance Configuration
NETWORK_TIMEOUT = int(os.getenv('NETWORK_TIMEOUT', '30'))  # Network timeout in seconds
NETWORK_MAX_RETRIES = int(os.getenv('NETWORK_MAX_RETRIES', '3'))  # Network max retries
NETWORK_RETRY_DELAY = float(os.getenv('NETWORK_RETRY_DELAY', '1.0'))  # Network retry delay
NETWORK_CONNECTION_POOL_SIZE = int(os.getenv('NETWORK_CONNECTION_POOL_SIZE', '10'))  # Network connection pool size

# Monitoring and Profiling Configuration
PERFORMANCE_MONITORING_ENABLED = os.getenv('PERFORMANCE_MONITORING_ENABLED', 'true').lower() == 'true'  # Enable performance monitoring
PERFORMANCE_PROFILING_ENABLED = os.getenv('PERFORMANCE_PROFILING_ENABLED', 'false').lower() == 'true'  # Enable performance profiling
PERFORMANCE_METRICS_INTERVAL = int(os.getenv('PERFORMANCE_METRICS_INTERVAL', '60'))  # Performance metrics interval
PERFORMANCE_LOG_SLOW_QUERIES = os.getenv('PERFORMANCE_LOG_SLOW_QUERIES', 'true').lower() == 'true'  # Log slow queries
PERFORMANCE_SLOW_QUERY_THRESHOLD_MS = int(os.getenv('PERFORMANCE_SLOW_QUERY_THRESHOLD_MS', '1000'))  # Slow query threshold

# Resource Monitoring Configuration
RESOURCE_MONITORING_ENABLED = os.getenv('RESOURCE_MONITORING_ENABLED', 'true').lower() == 'true'  # Enable resource monitoring
RESOURCE_MONITOR_INTERVAL = int(os.getenv('RESOURCE_MONITOR_INTERVAL', '30'))  # Resource monitor interval
RESOURCE_ALERT_CPU_THRESHOLD = float(os.getenv('RESOURCE_ALERT_CPU_THRESHOLD', '0.9'))  # CPU alert threshold
RESOURCE_ALERT_MEMORY_THRESHOLD = float(os.getenv('RESOURCE_ALERT_MEMORY_THRESHOLD', '0.9'))  # Memory alert threshold
RESOURCE_ALERT_DISK_THRESHOLD = float(os.getenv('RESOURCE_ALERT_DISK_THRESHOLD', '0.9'))  # Disk alert threshold

# Performance Optimization Profiles
PERFORMANCE_PROFILES = {
    'light': {
        'description': 'Minimal resource usage profile',
        'memory_limit_gb': 2.0,
        'max_workers': 2,
        'chunk_size': 500,
        'cache_enabled': True,
        'gpu_enabled': False,
        'parallel_processing': False,
        'vectorization': True
    },
    'balanced': {
        'description': 'Balanced performance and resource usage',
        'memory_limit_gb': 4.0,
        'max_workers': 4,
        'chunk_size': 1000,
        'cache_enabled': True,
        'gpu_enabled': True,
        'parallel_processing': True,
        'vectorization': True
    },
    'heavy': {
        'description': 'Maximum performance profile',
        'memory_limit_gb': 16.0,
        'max_workers': 8,
        'chunk_size': 5000,
        'cache_enabled': True,
        'gpu_enabled': True,
        'parallel_processing': True,
        'vectorization': True
    },
    'gpu_optimized': {
        'description': 'GPU-optimized performance profile',
        'memory_limit_gb': 8.0,
        'max_workers': 4,
        'chunk_size': 10000,
        'cache_enabled': True,
        'gpu_enabled': True,
        'parallel_processing': True,
        'vectorization': True,
        'gpu_memory_fraction': 0.9,
        'gpu_allow_growth': False
    }
}

# Performance Benchmarks Configuration
PERFORMANCE_BENCHMARKS_ENABLED = os.getenv('PERFORMANCE_BENCHMARKS_ENABLED', 'true').lower() == 'true'  # Enable performance benchmarks
PERFORMANCE_BENCHMARK_INTERVAL_HOURS = int(os.getenv('PERFORMANCE_BENCHMARK_INTERVAL_HOURS', '24'))  # Benchmark interval
PERFORMANCE_BENCHMARK_DATASET_SIZE = int(os.getenv('PERFORMANCE_BENCHMARK_DATASET_SIZE', '100000'))  # Benchmark dataset size
PERFORMANCE_BENCHMARK_ITERATIONS = int(os.getenv('PERFORMANCE_BENCHMARK_ITERATIONS', '5'))  # Benchmark iterations

# Adaptive Performance Configuration
ADAPTIVE_PERFORMANCE_ENABLED = os.getenv('ADAPTIVE_PERFORMANCE_ENABLED', 'true').lower() == 'true'  # Enable adaptive performance
ADAPTIVE_PERFORMANCE_CHECK_INTERVAL = int(os.getenv('ADAPTIVE_PERFORMANCE_CHECK_INTERVAL', '300'))  # Adaptive check interval
ADAPTIVE_PERFORMANCE_CPU_THRESHOLD = float(os.getenv('ADAPTIVE_PERFORMANCE_CPU_THRESHOLD', '0.8'))  # CPU threshold for adaptation
ADAPTIVE_PERFORMANCE_MEMORY_THRESHOLD = float(os.getenv('ADAPTIVE_PERFORMANCE_MEMORY_THRESHOLD', '0.8'))  # Memory threshold for adaptation

# Performance Metrics Configuration
PERFORMANCE_METRICS_ENABLED = os.getenv('PERFORMANCE_METRICS_ENABLED', 'true').lower() == 'true'  # Enable performance metrics
PERFORMANCE_METRICS_PREFIX = os.getenv('PERFORMANCE_METRICS_PREFIX', 'tradpal_performance_')  # Metrics prefix
PERFORMANCE_METRICS_RETENTION_DAYS = int(os.getenv('PERFORMANCE_METRICS_RETENTION_DAYS', '30'))  # Metrics retention

# Performance Alert Configuration
PERFORMANCE_ALERTS_ENABLED = os.getenv('PERFORMANCE_ALERTS_ENABLED', 'true').lower() == 'true'  # Enable performance alerts
PERFORMANCE_ALERT_EMAILS = os.getenv('PERFORMANCE_ALERT_EMAILS', '').split(',') if os.getenv('PERFORMANCE_ALERT_EMAILS') else []  # Performance alert emails
PERFORMANCE_ALERT_SLACK_WEBHOOK = os.getenv('PERFORMANCE_ALERT_SLACK_WEBHOOK', '')  # Slack webhook for performance alerts

# Performance Optimization Flags
OPTIMIZATION_FLAGS = {
    'numpy_optimization': True,
    'pandas_optimization': True,
    'memory_optimization': True,
    'cache_optimization': True,
    'parallel_optimization': True,
    'gpu_optimization': True,
    'vectorization_optimization': True,
    'async_optimization': True,
    'database_optimization': True,
    'network_optimization': True
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'data_loading_time_ms': 5000,  # Max data loading time
    'indicator_calculation_time_ms': 1000,  # Max indicator calculation time
    'signal_generation_time_ms': 500,  # Max signal generation time
    'backtest_execution_time_ms': 30000,  # Max backtest execution time
    'memory_usage_mb': 1024,  # Max memory usage
    'cpu_usage_percent': 80,  # Max CPU usage
    'disk_io_mb_per_sec': 50,  # Max disk I/O
    'network_latency_ms': 100  # Max network latency
}

# Performance Scaling Configuration
PERFORMANCE_SCALING_ENABLED = os.getenv('PERFORMANCE_SCALING_ENABLED', 'true').lower() == 'true'  # Enable performance scaling
PERFORMANCE_SCALE_UP_THRESHOLD = float(os.getenv('PERFORMANCE_SCALE_UP_THRESHOLD', '0.8'))  # Scale up threshold
PERFORMANCE_SCALE_DOWN_THRESHOLD = float(os.getenv('PERFORMANCE_SCALE_DOWN_THRESHOLD', '0.3'))  # Scale down threshold
PERFORMANCE_MAX_SCALE_FACTOR = int(os.getenv('PERFORMANCE_MAX_SCALE_FACTOR', '4'))  # Max scale factor

# Performance Logging Configuration
PERFORMANCE_LOG_ENABLED = os.getenv('PERFORMANCE_LOG_ENABLED', 'true').lower() == 'true'  # Enable performance logging
PERFORMANCE_LOG_LEVEL = os.getenv('PERFORMANCE_LOG_LEVEL', 'INFO')  # Performance log level
PERFORMANCE_LOG_FILE = os.getenv('PERFORMANCE_LOG_FILE', 'logs/performance.log')  # Performance log file
PERFORMANCE_LOG_MAX_SIZE_MB = int(os.getenv('PERFORMANCE_LOG_MAX_SIZE_MB', '100'))  # Max log file size
PERFORMANCE_LOG_BACKUP_COUNT = int(os.getenv('PERFORMANCE_LOG_BACKUP_COUNT', '5'))  # Log backup count