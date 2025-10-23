# Microservices configuration settings
# Service URLs, feature flags, and inter-service communication

import os
from typing import Dict, Any

# Microservices Configuration
DATA_SERVICE_URL = os.getenv('DATA_SERVICE_URL', 'http://localhost:8001')
CORE_SERVICE_URL = os.getenv('CORE_SERVICE_URL', 'http://localhost:8002')
ML_TRAINER_URL = os.getenv('ML_TRAINER_URL', 'http://localhost:8003')
BACKTESTING_SERVICE_URL = os.getenv('BACKTESTING_SERVICE_URL', 'http://localhost:8004')
TRADING_BOT_LIVE_URL = os.getenv('TRADING_BOT_LIVE_URL', 'http://localhost:8005')
RISK_SERVICE_URL = os.getenv('RISK_SERVICE_URL', 'http://localhost:8006')
NOTIFICATION_SERVICE_URL = os.getenv('NOTIFICATION_SERVICE_URL', 'http://localhost:8007')
WEB_UI_URL = os.getenv('WEB_UI_URL', 'http://localhost:8501')
DISCOVERY_SERVICE_URL = os.getenv('DISCOVERY_SERVICE_URL', 'http://localhost:8008')
MLOPS_SERVICE_URL = os.getenv('MLOPS_SERVICE_URL', 'http://localhost:8009')
OPTIMIZER_URL = os.getenv('OPTIMIZER_URL', 'http://localhost:8010')

# Service Feature Flags
ENABLE_DATA_SERVICE = os.getenv('ENABLE_DATA_SERVICE', 'true').lower() == 'true'
ENABLE_CORE_SERVICE = os.getenv('ENABLE_CORE_SERVICE', 'true').lower() == 'true'
ENABLE_ML_TRAINER = os.getenv('ENABLE_ML_TRAINER', 'true').lower() == 'true'
ENABLE_BACKTESTING = os.getenv('ENABLE_BACKTESTING', 'true').lower() == 'true'
ENABLE_LIVE_TRADING = os.getenv('ENABLE_LIVE_TRADING', 'false').lower() == 'true'
ENABLE_RISK_SERVICE = os.getenv('ENABLE_RISK_SERVICE', 'true').lower() == 'true'
ENABLE_NOTIFICATIONS = os.getenv('ENABLE_NOTIFICATIONS', 'true').lower() == 'true'
ENABLE_WEB_UI = os.getenv('ENABLE_WEB_UI', 'true').lower() == 'true'
ENABLE_DISCOVERY = os.getenv('ENABLE_DISCOVERY', 'true').lower() == 'true'
ENABLE_MLOPS = os.getenv('ENABLE_MLOPS', 'true').lower() == 'true'
ENABLE_OPTIMIZER = os.getenv('ENABLE_OPTIMIZER', 'true').lower() == 'true'

# Service Feature Flags (Legacy Aliases)
ENABLE_ML = ENABLE_MLOPS
ENABLE_BACKTEST = ENABLE_BACKTESTING

# Service URLs (specific service URLs)
ML_TRAINER_SERVICE_URL = MLOPS_SERVICE_URL  # ML Trainer uses MLOps service URL
TRADING_BOT_LIVE_SERVICE_URL = TRADING_BOT_LIVE_URL  # Trading Bot Live service URL
WEB_UI_SERVICE_URL = WEB_UI_URL  # Web UI service URL
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))  # HTTP request timeout in seconds

# UI Configuration
UI_REFRESH_INTERVAL = int(os.getenv('UI_REFRESH_INTERVAL', '30'))  # UI refresh interval in seconds

# Additional constants needed by various modules
ENABLE_MTLS = os.getenv('ENABLE_MTLS', 'true').lower() == 'true'
MTLS_CERT_PATH = os.getenv('MTLS_CERT_PATH', 'cache/security/certs/client.crt')
MTLS_KEY_PATH = os.getenv('MTLS_KEY_PATH', 'cache/security/certs/client.key')
CA_CERT_PATH = os.getenv('CA_CERT_PATH', 'cache/security/ca/ca_cert.pem')
SECURITY_SERVICE_URL = os.getenv('SECURITY_SERVICE_URL', 'http://localhost:8012')
MTA_ENABLED = os.getenv('MTA_ENABLED', 'true').lower() == 'true'
MTA_TIMEFRAMES = ['5m', '15m']  # Available higher timeframes
MAX_BACKTEST_RESULTS = 100
MONITORING_STACK_ENABLED = os.getenv('MONITORING_STACK_ENABLED', 'true').lower() == 'true'
PERFORMANCE_MONITORING_ENABLED = os.getenv('PERFORMANCE_MONITORING_ENABLED', 'true').lower() == 'true'
ADAPTIVE_OPTIMIZATION_ENABLED_LIVE = os.getenv('ADAPTIVE_OPTIMIZATION_ENABLED_LIVE', 'false').lower() == 'true'
DEFAULT_TIMEFRAME = os.getenv('DEFAULT_TIMEFRAME', '1h')  # Default timeframe
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '5'))  # Maximum open positions per symbol
MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '0.1'))  # Maximum drawdown before stopping (10%)
ORDER_TIMEOUT = int(os.getenv('ORDER_TIMEOUT', '30'))  # Order timeout in seconds
POSITION_UPDATE_INTERVAL = int(os.getenv('POSITION_UPDATE_INTERVAL', '60'))  # Position update interval in seconds

# Alternative Data Service Configuration
ALTERNATIVE_DATA_UPDATE_INTERVAL = int(os.getenv('ALTERNATIVE_DATA_UPDATE_INTERVAL', '300'))  # 5 minutes
SENTIMENT_DATA_SOURCES = os.getenv('SENTIMENT_DATA_SOURCES', 'twitter,reddit,news').split(',')  # Sentiment sources
ONCHAIN_DATA_SOURCES = os.getenv('ONCHAIN_DATA_SOURCES', 'glassnode,blockchain_com').split(',')  # On-chain data sources
ECONOMIC_DATA_SOURCES = os.getenv('ECONOMIC_DATA_SOURCES', 'fred,bureau_labor,alpha_vantage').split(',')  # Economic data sources

# Data Mesh Configuration
DATA_MESH_ENABLED = os.getenv('DATA_MESH_ENABLED', 'true').lower() == 'true'  # Enable Data Mesh architecture

# TimeSeries Database Configuration (InfluxDB)
INFLUXDB_ENABLED = os.getenv('INFLUXDB_ENABLED', 'true').lower() == 'true'
INFLUXDB_URL = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN', '')
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG', 'tradpal')
INFLUXDB_BUCKET = os.getenv('INFLUXDB_BUCKET', 'market_data')
INFLUXDB_MEASUREMENT_PREFIX = os.getenv('INFLUXDB_MEASUREMENT_PREFIX', 'ohlcv_')

# Data Lake Configuration (MinIO/S3)
DATA_LAKE_ENABLED = os.getenv('DATA_LAKE_ENABLED', 'true').lower() == 'true'
DATA_LAKE_TYPE = os.getenv('DATA_LAKE_TYPE', 'minio')  # 'minio' or 's3'
DATA_LAKE_ENDPOINT = os.getenv('DATA_LAKE_ENDPOINT', 'http://localhost:9000')
DATA_LAKE_ACCESS_KEY = os.getenv('DATA_LAKE_ACCESS_KEY', 'minioadmin')
DATA_LAKE_SECRET_KEY = os.getenv('DATA_LAKE_SECRET_KEY', 'minioadmin')
DATA_LAKE_BUCKET = os.getenv('DATA_LAKE_BUCKET', 'tradpal-data-lake')
DATA_LAKE_REGION = os.getenv('DATA_LAKE_REGION', 'us-east-1')
DATA_LAKE_ARCHIVE_PREFIX = os.getenv('DATA_LAKE_ARCHIVE_PREFIX', 'archive/market_data/')

# Feature Store Configuration (Redis)
FEATURE_STORE_ENABLED = os.getenv('FEATURE_STORE_ENABLED', 'true').lower() == 'true'
FEATURE_STORE_KEY_PREFIX = os.getenv('FEATURE_STORE_KEY_PREFIX', 'features:')
FEATURE_STORE_METADATA_PREFIX = os.getenv('FEATURE_STORE_METADATA_PREFIX', 'feature_metadata:')
FEATURE_STORE_VERSION_PREFIX = os.getenv('FEATURE_STORE_VERSION_PREFIX', 'feature_versions:')

# Data Product Registry Configuration
DATA_PRODUCT_REGISTRY_ENABLED = os.getenv('DATA_PRODUCT_REGISTRY_ENABLED', 'true').lower() == 'true'
DATA_PRODUCT_KEY_PREFIX = os.getenv('DATA_PRODUCT_KEY_PREFIX', 'data_products:')
DATA_PRODUCT_SCHEMA_PREFIX = os.getenv('DATA_PRODUCT_SCHEMA_PREFIX', 'data_schemas:')

# Data Mesh Governance Configuration
DATA_MESH_GOVERNANCE_ENABLED = os.getenv('DATA_MESH_GOVERNANCE_ENABLED', 'true').lower() == 'true'
DATA_MESH_AUDIT_LOG_ENABLED = os.getenv('DATA_MESH_AUDIT_LOG_ENABLED', 'true').lower() == 'true'
DATA_MESH_AUDIT_KEY_PREFIX = os.getenv('DATA_MESH_AUDIT_KEY_PREFIX', 'audit:')
DATA_MESH_ACCESS_CONTROL_ENABLED = os.getenv('DATA_MESH_ACCESS_CONTROL_ENABLED', 'true').lower() == 'true'

# Data Quality Configuration for Data Mesh
DATA_MESH_QUALITY_ENABLED = os.getenv('DATA_MESH_QUALITY_ENABLED', 'true').lower() == 'true'
DATA_MESH_QUALITY_CHECK_INTERVAL = int(os.getenv('DATA_MESH_QUALITY_CHECK_INTERVAL', '3600'))  # 1 hour
DATA_MESH_QUALITY_ALERT_THRESHOLD = float(os.getenv('DATA_MESH_QUALITY_ALERT_THRESHOLD', '0.8'))

# Data Mesh Performance Configuration
DATA_MESH_BATCH_SIZE = int(os.getenv('DATA_MESH_BATCH_SIZE', '1000'))
DATA_MESH_PARALLEL_WORKERS = int(os.getenv('DATA_MESH_PARALLEL_WORKERS', '4'))
DATA_MESH_CACHE_TTL = int(os.getenv('DATA_MESH_CACHE_TTL', '3600'))  # 1 hour

# Data Mesh Domains
DATA_MESH_DOMAINS = {
    'market_data': {
        'description': 'Real-time and historical market data (OHLCV, volume, etc.)',
        'retention_days': 365 * 2,  # 2 years
        'data_quality_required': True,
        'owners': ['data_team', 'trading_team']
    },
    'trading_signals': {
        'description': 'Generated trading signals and indicators',
        'retention_days': 365,  # 1 year
        'data_quality_required': True,
        'owners': ['trading_team', 'ml_team']
    },
    'ml_features': {
        'description': 'Machine learning features and training data',
        'retention_days': 365,  # 1 year
        'data_quality_required': True,
        'owners': ['ml_team', 'data_team']
    },
    'performance_metrics': {
        'description': 'Trading performance and backtest results',
        'retention_days': 365 * 3,  # 3 years
        'data_quality_required': False,
        'owners': ['trading_team', 'analytics_team']
    },
    'risk_data': {
        'description': 'Risk management data and calculations',
        'retention_days': 365 * 2,  # 2 years
        'data_quality_required': True,
        'owners': ['risk_team', 'trading_team']
    }
}

# Data Mesh Data Products
DATA_MESH_PRODUCTS = {
    'btc_usdt_ohlcv': {
        'domain': 'market_data',
        'description': 'BTC/USDT OHLCV data across all timeframes',
        'schema': {
            'timestamp': 'datetime',
            'open': 'float',
            'high': 'float',
            'low': 'float',
            'close': 'float',
            'volume': 'float',
            'symbol': 'string',
            'timeframe': 'string'
        },
        'owners': ['data_team'],
        'tags': ['crypto', 'btc', 'usdt', 'ohlcv']
    },
    'trading_signals_combined': {
        'domain': 'trading_signals',
        'description': 'Combined trading signals from all strategies',
        'schema': {
            'timestamp': 'datetime',
            'symbol': 'string',
            'signal': 'string',  # 'buy', 'sell', 'hold'
            'confidence': 'float',
            'strategy': 'string',
            'indicators': 'json',
            'metadata': 'json'
        },
        'owners': ['trading_team'],
        'tags': ['signals', 'trading', 'indicators']
    },
    'ml_features_technical': {
        'domain': 'ml_features',
        'description': 'Technical analysis features for ML models',
        'schema': {
            'timestamp': 'datetime',
            'symbol': 'string',
            'sma_20': 'float',
            'sma_50': 'float',
            'rsi': 'float',
            'macd': 'float',
            'macd_signal': 'float',
            'bb_upper': 'float',
            'bb_lower': 'float',
            'atr': 'float',
            'returns': 'float',
            'momentum': 'float'
        },
        'owners': ['ml_team'],
        'tags': ['ml', 'features', 'technical']
    },
    'backtest_performance': {
        'domain': 'performance_metrics',
        'description': 'Backtesting performance results',
        'schema': {
            'backtest_id': 'string',
            'symbol': 'string',
            'timeframe': 'string',
            'start_date': 'datetime',
            'end_date': 'datetime',
            'total_trades': 'int',
            'win_rate': 'float',
            'total_pnl': 'float',
            'sharpe_ratio': 'float',
            'max_drawdown': 'float',
            'profit_factor': 'float'
        },
        'owners': ['trading_team'],
        'tags': ['backtest', 'performance', 'metrics']
    }
}

# Data Governance Configuration
DATA_GOVERNANCE_ENABLED = os.getenv('DATA_GOVERNANCE_ENABLED', 'true').lower() == 'true'  # Enable Data Governance
DATA_GOVERNANCE_AUDIT_ENABLED = os.getenv('DATA_GOVERNANCE_AUDIT_ENABLED', 'true').lower() == 'true'  # Enable audit logging
DATA_GOVERNANCE_ACCESS_CONTROL_ENABLED = os.getenv('DATA_GOVERNANCE_ACCESS_CONTROL_ENABLED', 'true').lower() == 'true'  # Enable access control
DATA_GOVERNANCE_QUALITY_MONITORING_ENABLED = os.getenv('DATA_GOVERNANCE_QUALITY_MONITORING_ENABLED', 'true').lower() == 'true'  # Enable quality monitoring

# Audit Logging Configuration
AUDIT_LOG_RETENTION_DAYS = int(os.getenv('AUDIT_LOG_RETENTION_DAYS', '90'))  # Audit log retention period
AUDIT_LOG_MAX_EVENTS = int(os.getenv('AUDIT_LOG_MAX_EVENTS', '10000'))  # Max audit events in memory
AUDIT_LOG_COMPLIANCE_MODE = os.getenv('AUDIT_LOG_COMPLIANCE_MODE', 'false').lower() == 'true'  # Strict compliance mode

# Access Control Configuration
ACCESS_CONTROL_DEFAULT_POLICY = os.getenv('ACCESS_CONTROL_DEFAULT_POLICY', 'deny')  # 'allow' or 'deny'
ACCESS_CONTROL_CACHE_TTL = int(os.getenv('ACCESS_CONTROL_CACHE_TTL', '300'))  # Access decision cache TTL
ACCESS_CONTROL_MAX_POLICIES = int(os.getenv('ACCESS_CONTROL_MAX_POLICIES', '1000'))  # Max policies per resource type

# Data Quality Monitoring Configuration
QUALITY_MONITOR_CHECK_INTERVAL = int(os.getenv('QUALITY_MONITOR_CHECK_INTERVAL', '3600'))  # Quality check interval (seconds)
QUALITY_MONITOR_ALERT_THRESHOLD = float(os.getenv('QUALITY_MONITOR_ALERT_THRESHOLD', '0.8'))  # Quality alert threshold
QUALITY_MONITOR_MAX_ISSUES = int(os.getenv('QUALITY_MONITOR_MAX_ISSUES', '100'))  # Max issues to track per resource
QUALITY_MONITOR_AUTO_CORRECTION = os.getenv('QUALITY_MONITOR_AUTO_CORRECTION', 'false').lower() == 'true'  # Auto-correct quality issues

# Governance Roles Configuration
GOVERNANCE_ROLES = {
    'data_admin': {
        'description': 'Full access to all data resources and governance functions',
        'permissions': ['*'],
        'max_access_level': 'admin'
    },
    'data_steward': {
        'description': 'Manage data products and quality within assigned domains',
        'permissions': ['data_product:*', 'domain:read', 'quality:*'],
        'max_access_level': 'write'
    },
    'data_consumer': {
        'description': 'Read access to approved data products',
        'permissions': ['data_product:read', 'domain:read'],
        'max_access_level': 'read'
    },
    'ml_engineer': {
        'description': 'Access to ML features and model data',
        'permissions': ['feature_set:*', 'ml_features:*', 'data_product:read'],
        'max_access_level': 'write'
    },
    'trading_service': {
        'description': 'Access for automated trading systems',
        'permissions': ['market_data:read', 'trading_signals:*', 'risk_data:read'],
        'max_access_level': 'write'
    },
    'analyst': {
        'description': 'Read access for data analysis and reporting',
        'permissions': ['data_product:read', 'performance_metrics:read', 'domain:read'],
        'max_access_level': 'read'
    },
    'auditor': {
        'description': 'Access to audit logs and compliance reports',
        'permissions': ['audit_logs:read', 'compliance_reports:read'],
        'max_access_level': 'read'
    }
}

# Governance Policies Configuration
GOVERNANCE_POLICIES = {
    'market_data_policy': {
        'name': 'Market Data Access Policy',
        'description': 'Controls access to real-time market data',
        'resource_type': 'domain',
        'resource_name': 'market_data',
        'rules': {
            'allowed_roles': ['data_admin', 'trading_service', 'analyst'],
            'max_access_level': 'read',
            'time_restrictions': {'weekdays_only': True},
            'rate_limits': {'requests_per_hour': 1000}
        }
    },
    'trading_signals_policy': {
        'name': 'Trading Signals Policy',
        'description': 'Controls access to trading signals and strategies',
        'resource_type': 'domain',
        'resource_name': 'trading_signals',
        'rules': {
            'allowed_roles': ['data_admin', 'trading_service'],
            'max_access_level': 'write',
            'approval_required': True,
            'audit_required': True
        }
    },
    'ml_features_policy': {
        'name': 'ML Features Policy',
        'description': 'Controls access to ML training data and features',
        'resource_type': 'domain',
        'resource_name': 'ml_features',
        'rules': {
            'allowed_roles': ['data_admin', 'ml_engineer', 'data_steward'],
            'max_access_level': 'write',
            'data_classification': 'sensitive',
            'encryption_required': True
        }
    },
    'audit_logs_policy': {
        'name': 'Audit Logs Access Policy',
        'description': 'Controls access to audit logs and compliance data',
        'resource_type': 'system',
        'resource_name': 'audit_logs',
        'rules': {
            'allowed_roles': ['data_admin', 'auditor'],
            'max_access_level': 'read',
            'retention_period_days': AUDIT_LOG_RETENTION_DAYS,
            'immutable': True
        }
    }
}

# Quality Rules Configuration
DATA_QUALITY_RULES = {
    'completeness_check': {
        'name': 'Data Completeness Check',
        'description': 'Ensures critical data fields are not missing',
        'rule_type': 'completeness',
        'parameters': {
            'critical_fields': ['timestamp', 'close', 'volume'],
            'max_missing_ratio': 0.05
        },
        'severity': 'high',
        'enabled': True
    },
    'accuracy_check': {
        'name': 'Data Accuracy Check',
        'description': 'Validates data ranges and logical consistency',
        'rule_type': 'accuracy',
        'parameters': {
            'price_range': {'min': 0.000001, 'max': 1000000},
            'volume_range': {'min': 0, 'max': 1000000000},
            'ohlc_consistency': True
        },
        'severity': 'critical',
        'enabled': True
    },
    'timeliness_check': {
        'name': 'Data Timeliness Check',
        'description': 'Ensures data freshness and timeliness',
        'rule_type': 'timeliness',
        'parameters': {
            'max_age_hours': {'1m': 1, '1h': 24, '1d': 168},  # Max age by timeframe
            'staleness_threshold_minutes': 5
        },
        'severity': 'medium',
        'enabled': True
    },
    'consistency_check': {
        'name': 'Data Consistency Check',
        'description': 'Checks for logical consistency across data points',
        'rule_type': 'consistency',
        'parameters': {
            'duplicate_check': True,
            'sequence_check': True,
            'gap_detection': {'max_gap_minutes': 60}
        },
        'severity': 'medium',
        'enabled': True
    }
}

# Compliance Configuration
COMPLIANCE_ENABLED = os.getenv('COMPLIANCE_ENABLED', 'true').lower() == 'true'  # Enable compliance features
COMPLIANCE_REPORT_INTERVAL = int(os.getenv('COMPLIANCE_REPORT_INTERVAL', '86400'))  # Daily compliance reports
COMPLIANCE_RETENTION_YEARS = int(os.getenv('COMPLIANCE_RETENTION_YEARS', '7'))  # 7-year retention for compliance
COMPLIANCE_AUTO_REPORTING = os.getenv('COMPLIANCE_AUTO_REPORTING', 'true').lower() == 'true'  # Auto-generate reports

# Governance Monitoring Configuration
GOVERNANCE_MONITORING_ENABLED = os.getenv('GOVERNANCE_MONITORING_ENABLED', 'true').lower() == 'true'
GOVERNANCE_ALERT_EMAILS = os.getenv('GOVERNANCE_ALERT_EMAILS', '').split(',') if os.getenv('GOVERNANCE_ALERT_EMAILS') else []
GOVERNANCE_SLACK_WEBHOOK = os.getenv('GOVERNANCE_SLACK_WEBHOOK', '')  # Slack webhook for alerts
GOVERNANCE_METRICS_PREFIX = os.getenv('GOVERNANCE_METRICS_PREFIX', 'tradpal_governance_')  # Prometheus metrics prefix