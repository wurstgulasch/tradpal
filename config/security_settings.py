# Security configuration settings
# Zero-Trust authentication, mTLS, JWT, and secrets management

import os
from typing import Dict, Any

# Zero-Trust Security Configuration
ZERO_TRUST_ENABLED = os.getenv('ZERO_TRUST_ENABLED', 'true').lower() == 'true'  # Enable Zero-Trust architecture

# JWT Configuration
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')  # JWT secret key
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')  # JWT algorithm
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRE_MINUTES', '30'))  # Access token expiry
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv('JWT_REFRESH_TOKEN_EXPIRE_DAYS', '7'))  # Refresh token expiry
JWT_ISSUER = os.getenv('JWT_ISSUER', 'tradpal-system')  # JWT issuer
JWT_AUDIENCE = os.getenv('JWT_AUDIENCE', 'tradpal-services')  # JWT audience

# mTLS Configuration
MTLS_ENABLED = os.getenv('MTLS_ENABLED', 'true').lower() == 'true'  # Enable mutual TLS
MTLS_CERT_PATH = os.getenv('MTLS_CERT_PATH', 'cache/security/certs/')  # Certificate directory
MTLS_CA_CERT = os.path.join(MTLS_CERT_PATH, 'ca.crt')  # CA certificate
MTLS_CLIENT_CERT = os.path.join(MTLS_CERT_PATH, 'client.crt')  # Client certificate
MTLS_CLIENT_KEY = os.path.join(MTLS_CERT_PATH, 'client.key')  # Client private key
MTLS_SERVER_CERT = os.path.join(MTLS_CERT_PATH, 'server.crt')  # Server certificate
MTLS_SERVER_KEY = os.path.join(MTLS_CERT_PATH, 'server.key')  # Server private key
MTLS_CERT_VALIDITY_DAYS = int(os.getenv('MTLS_CERT_VALIDITY_DAYS', '365'))  # Certificate validity period

# OAuth2 Configuration
OAUTH2_ENABLED = os.getenv('OAUTH2_ENABLED', 'false').lower() == 'true'  # Enable OAuth2
OAUTH2_CLIENT_ID = os.getenv('OAUTH2_CLIENT_ID', '')  # OAuth2 client ID
OAUTH2_CLIENT_SECRET = os.getenv('OAUTH2_CLIENT_SECRET', '')  # OAuth2 client secret
OAUTH2_AUTH_URL = os.getenv('OAUTH2_AUTH_URL', '')  # OAuth2 authorization URL
OAUTH2_TOKEN_URL = os.getenv('OAUTH2_TOKEN_URL', '')  # OAuth2 token URL
OAUTH2_REDIRECT_URI = os.getenv('OAUTH2_REDIRECT_URI', '')  # OAuth2 redirect URI

# API Key Configuration
API_KEY_ENABLED = os.getenv('API_KEY_ENABLED', 'true').lower() == 'true'  # Enable API key authentication
API_KEY_HEADER = os.getenv('API_KEY_HEADER', 'X-API-Key')  # API key header name
API_KEY_LENGTH = int(os.getenv('API_KEY_LENGTH', '32'))  # API key length
API_KEY_ROTATION_DAYS = int(os.getenv('API_KEY_ROTATION_DAYS', '90'))  # API key rotation period

# Rate Limiting Configuration
RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'  # Enable rate limiting
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv('RATE_LIMIT_REQUESTS_PER_MINUTE', '60'))  # Requests per minute
RATE_LIMIT_BURST_SIZE = int(os.getenv('RATE_LIMIT_BURST_SIZE', '10'))  # Burst size for rate limiting
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv('RATE_LIMIT_WINDOW_SECONDS', '60'))  # Rate limit window

# IP Whitelisting Configuration
IP_WHITELIST_ENABLED = os.getenv('IP_WHITELIST_ENABLED', 'false').lower() == 'true'  # Enable IP whitelisting
IP_WHITELIST = os.getenv('IP_WHITELIST', '').split(',') if os.getenv('IP_WHITELIST') else []  # Whitelisted IPs

# Secrets Management Configuration (HashiCorp Vault)
VAULT_ENABLED = os.getenv('VAULT_ENABLED', 'false').lower() == 'true'  # Enable HashiCorp Vault
VAULT_URL = os.getenv('VAULT_URL', 'http://localhost:8200')  # Vault server URL
VAULT_TOKEN = os.getenv('VAULT_TOKEN', '')  # Vault authentication token
VAULT_MOUNT_POINT = os.getenv('VAULT_MOUNT_POINT', 'secret')  # Vault mount point
VAULT_PATH_PREFIX = os.getenv('VAULT_PATH_PREFIX', 'tradpal/')  # Vault path prefix
VAULT_TLS_VERIFY = os.getenv('VAULT_TLS_VERIFY', 'true').lower() == 'true'  # Verify TLS certificates

# Encryption Configuration
ENCRYPTION_ENABLED = os.getenv('ENCRYPTION_ENABLED', 'true').lower() == 'true'  # Enable data encryption
ENCRYPTION_KEY_SIZE = int(os.getenv('ENCRYPTION_KEY_SIZE', '256'))  # Encryption key size (bits)
ENCRYPTION_ALGORITHM = os.getenv('ENCRYPTION_ALGORITHM', 'AES-GCM')  # Encryption algorithm
ENCRYPTION_KEY_ROTATION_DAYS = int(os.getenv('ENCRYPTION_KEY_ROTATION_DAYS', '30'))  # Key rotation period

# Audit Logging Configuration
AUDIT_LOG_ENABLED = os.getenv('AUDIT_LOG_ENABLED', 'true').lower() == 'true'  # Enable audit logging
AUDIT_LOG_LEVEL = os.getenv('AUDIT_LOG_LEVEL', 'INFO')  # Audit log level
AUDIT_LOG_FORMAT = os.getenv('AUDIT_LOG_FORMAT', 'json')  # Audit log format ('json' or 'text')
AUDIT_LOG_FILE_PATH = os.getenv('AUDIT_LOG_FILE_PATH', 'logs/security_audit.log')  # Audit log file path
AUDIT_LOG_MAX_SIZE_MB = int(os.getenv('AUDIT_LOG_MAX_SIZE_MB', '100'))  # Max audit log file size
AUDIT_LOG_BACKUP_COUNT = int(os.getenv('AUDIT_LOG_BACKUP_COUNT', '5'))  # Number of backup files

# Security Monitoring Configuration
SECURITY_MONITORING_ENABLED = os.getenv('SECURITY_MONITORING_ENABLED', 'true').lower() == 'true'  # Enable security monitoring
SECURITY_ALERT_EMAILS = os.getenv('SECURITY_ALERT_EMAILS', '').split(',') if os.getenv('SECURITY_ALERT_EMAILS') else []  # Security alert emails
SECURITY_SLACK_WEBHOOK = os.getenv('SECURITY_SLACK_WEBHOOK', '')  # Slack webhook for security alerts
SECURITY_METRICS_PREFIX = os.getenv('SECURITY_METRICS_PREFIX', 'tradpal_security_')  # Prometheus metrics prefix

# Intrusion Detection Configuration
INTRUSION_DETECTION_ENABLED = os.getenv('INTRUSION_DETECTION_ENABLED', 'true').lower() == 'true'  # Enable intrusion detection
INTRUSION_DETECTION_RULES_PATH = os.getenv('INTRUSION_DETECTION_RULES_PATH', 'config/security_rules.json')  # IDS rules file
INTRUSION_DETECTION_LOG_PATH = os.getenv('INTRUSION_DETECTION_LOG_PATH', 'logs/intrusion.log')  # IDS log file
INTRUSION_DETECTION_ALERT_THRESHOLD = int(os.getenv('INTRUSION_DETECTION_ALERT_THRESHOLD', '5'))  # Alert threshold

# Session Management Configuration
SESSION_TIMEOUT_MINUTES = int(os.getenv('SESSION_TIMEOUT_MINUTES', '60'))  # Session timeout
SESSION_MAX_CONCURRENT = int(os.getenv('SESSION_MAX_CONCURRENT', '10'))  # Max concurrent sessions per user
SESSION_CLEANUP_INTERVAL_MINUTES = int(os.getenv('SESSION_CLEANUP_INTERVAL_MINUTES', '15'))  # Session cleanup interval

# Password Policy Configuration
PASSWORD_MIN_LENGTH = int(os.getenv('PASSWORD_MIN_LENGTH', '12'))  # Minimum password length
PASSWORD_REQUIRE_UPPERCASE = os.getenv('PASSWORD_REQUIRE_UPPERCASE', 'true').lower() == 'true'  # Require uppercase
PASSWORD_REQUIRE_LOWERCASE = os.getenv('PASSWORD_REQUIRE_LOWERCASE', 'true').lower() == 'true'  # Require lowercase
PASSWORD_REQUIRE_DIGITS = os.getenv('PASSWORD_REQUIRE_DIGITS', 'true').lower() == 'true'  # Require digits
PASSWORD_REQUIRE_SPECIAL_CHARS = os.getenv('PASSWORD_REQUIRE_SPECIAL_CHARS', 'true').lower() == 'true'  # Require special characters
PASSWORD_MAX_AGE_DAYS = int(os.getenv('PASSWORD_MAX_AGE_DAYS', '90'))  # Password maximum age
PASSWORD_HISTORY_COUNT = int(os.getenv('PASSWORD_HISTORY_COUNT', '5'))  # Password history to prevent reuse

# Multi-Factor Authentication Configuration
MFA_ENABLED = os.getenv('MFA_ENABLED', 'false').lower() == 'true'  # Enable MFA
MFA_METHODS = os.getenv('MFA_METHODS', 'totp,email').split(',')  # Supported MFA methods
MFA_TOTP_ISSUER = os.getenv('MFA_TOTP_ISSUER', 'TradPal')  # TOTP issuer name
MFA_BACKUP_CODES_COUNT = int(os.getenv('MFA_BACKUP_CODES_COUNT', '10'))  # Number of backup codes

# API Security Configuration
API_CORS_ENABLED = os.getenv('API_CORS_ENABLED', 'true').lower() == 'true'  # Enable CORS
API_CORS_ORIGINS = os.getenv('API_CORS_ORIGINS', '*').split(',')  # CORS allowed origins
API_CORS_METHODS = os.getenv('API_CORS_METHODS', 'GET,POST,PUT,DELETE,OPTIONS').split(',')  # CORS allowed methods
API_CORS_HEADERS = os.getenv('API_CORS_HEADERS', 'Content-Type,Authorization,X-API-Key').split(',')  # CORS allowed headers
API_CORS_CREDENTIALS = os.getenv('API_CORS_CREDENTIALS', 'true').lower() == 'true'  # CORS allow credentials

# Security Headers Configuration
SECURITY_HEADERS_ENABLED = os.getenv('SECURITY_HEADERS_ENABLED', 'true').lower() == 'true'  # Enable security headers
SECURITY_HEADERS_HSTS_MAX_AGE = int(os.getenv('SECURITY_HEADERS_HSTS_MAX_AGE', '31536000'))  # HSTS max age
SECURITY_HEADERS_CSP = os.getenv('SECURITY_HEADERS_CSP', "default-src 'self'")  # Content Security Policy
SECURITY_HEADERS_X_FRAME_OPTIONS = os.getenv('SECURITY_HEADERS_X_FRAME_OPTIONS', 'DENY')  # X-Frame-Options
SECURITY_HEADERS_X_CONTENT_TYPE_OPTIONS = os.getenv('SECURITY_HEADERS_X_CONTENT_TYPE_OPTIONS', 'nosniff')  # X-Content-Type-Options

# Data Protection Configuration
DATA_MASKING_ENABLED = os.getenv('DATA_MASKING_ENABLED', 'true').lower() == 'true'  # Enable data masking
DATA_MASKING_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
    'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
}  # Data masking patterns

DATA_ANONYMIZATION_ENABLED = os.getenv('DATA_ANONYMIZATION_ENABLED', 'false').lower() == 'true'  # Enable data anonymization
DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', '2555'))  # Data retention period (7 years for compliance)

# Compliance Configuration
COMPLIANCE_MODE_ENABLED = os.getenv('COMPLIANCE_MODE_ENABLED', 'false').lower() == 'true'  # Enable compliance mode
COMPLIANCE_LOGGING_ENABLED = os.getenv('COMPLIANCE_LOGGING_ENABLED', 'true').lower() == 'true'  # Enable compliance logging
COMPLIANCE_REPORTING_ENABLED = os.getenv('COMPLIANCE_REPORTING_ENABLED', 'true').lower() == 'true'  # Enable compliance reporting

# Security Event Configuration
SECURITY_EVENTS = {
    'authentication_failure': {
        'severity': 'medium',
        'alert_threshold': 5,
        'lockout_duration_minutes': 15,
        'description': 'Failed authentication attempts'
    },
    'unauthorized_access': {
        'severity': 'high',
        'alert_threshold': 1,
        'lockout_duration_minutes': 30,
        'description': 'Unauthorized access attempts'
    },
    'suspicious_activity': {
        'severity': 'high',
        'alert_threshold': 3,
        'lockout_duration_minutes': 60,
        'description': 'Suspicious user activity detected'
    },
    'data_breach_attempt': {
        'severity': 'critical',
        'alert_threshold': 1,
        'lockout_duration_minutes': 0,  # Permanent lockout
        'description': 'Potential data breach detected'
    },
    'rate_limit_exceeded': {
        'severity': 'low',
        'alert_threshold': 10,
        'lockout_duration_minutes': 5,
        'description': 'Rate limit exceeded'
    }
}

# Security Policies Configuration
SECURITY_POLICIES = {
    'password_policy': {
        'name': 'Password Security Policy',
        'description': 'Enforces strong password requirements',
        'rules': {
            'min_length': PASSWORD_MIN_LENGTH,
            'require_uppercase': PASSWORD_REQUIRE_UPPERCASE,
            'require_lowercase': PASSWORD_REQUIRE_LOWERCASE,
            'require_digits': PASSWORD_REQUIRE_DIGITS,
            'require_special_chars': PASSWORD_REQUIRE_SPECIAL_CHARS,
            'max_age_days': PASSWORD_MAX_AGE_DAYS,
            'history_count': PASSWORD_HISTORY_COUNT
        },
        'enforcement': 'strict',
        'enabled': True
    },
    'session_policy': {
        'name': 'Session Management Policy',
        'description': 'Controls user session behavior',
        'rules': {
            'timeout_minutes': SESSION_TIMEOUT_MINUTES,
            'max_concurrent': SESSION_MAX_CONCURRENT,
            'cleanup_interval_minutes': SESSION_CLEANUP_INTERVAL_MINUTES
        },
        'enforcement': 'strict',
        'enabled': True
    },
    'access_policy': {
        'name': 'Access Control Policy',
        'description': 'Controls resource access permissions',
        'rules': {
            'zero_trust_enabled': ZERO_TRUST_ENABLED,
            'mfa_required': MFA_ENABLED,
            'ip_whitelist_enabled': IP_WHITELIST_ENABLED,
            'rate_limiting_enabled': RATE_LIMIT_ENABLED
        },
        'enforcement': 'strict',
        'enabled': True
    },
    'data_policy': {
        'name': 'Data Protection Policy',
        'description': 'Ensures data security and compliance',
        'rules': {
            'encryption_enabled': ENCRYPTION_ENABLED,
            'masking_enabled': DATA_MASKING_ENABLED,
            'retention_days': DATA_RETENTION_DAYS,
            'audit_enabled': AUDIT_LOG_ENABLED
        },
        'enforcement': 'strict',
        'enabled': True
    }
}

# Security Monitoring Thresholds
SECURITY_THRESHOLDS = {
    'failed_login_attempts': {
        'warning': 3,
        'critical': 5,
        'lockout_duration_minutes': 15
    },
    'suspicious_requests': {
        'warning': 10,
        'critical': 25,
        'block_duration_minutes': 30
    },
    'data_access_anomalies': {
        'warning': 50,
        'critical': 100,
        'alert_channels': ['email', 'slack']
    },
    'api_abuse': {
        'warning': 100,
        'critical': 500,
        'block_duration_minutes': 60
    }
}

# Security Alert Configuration
SECURITY_ALERTS_ENABLED = os.getenv('SECURITY_ALERTS_ENABLED', 'true').lower() == 'true'  # Enable security alerts
SECURITY_ALERT_CHANNELS = os.getenv('SECURITY_ALERT_CHANNELS', 'email,slack').split(',')  # Alert channels
SECURITY_ALERT_SEVERITY_FILTER = os.getenv('SECURITY_ALERT_SEVERITY_FILTER', 'medium,high,critical').split(',')  # Alert severity filter

# Security Dashboard Configuration
SECURITY_DASHBOARD_ENABLED = os.getenv('SECURITY_DASHBOARD_ENABLED', 'true').lower() == 'true'  # Enable security dashboard
SECURITY_DASHBOARD_REFRESH_INTERVAL = int(os.getenv('SECURITY_DASHBOARD_REFRESH_INTERVAL', '300'))  # Dashboard refresh interval
SECURITY_DASHBOARD_METRICS_RETENTION_DAYS = int(os.getenv('SECURITY_DASHBOARD_METRICS_RETENTION_DAYS', '30'))  # Metrics retention

# Security Incident Response Configuration
INCIDENT_RESPONSE_ENABLED = os.getenv('INCIDENT_RESPONSE_ENABLED', 'true').lower() == 'true'  # Enable incident response
INCIDENT_RESPONSE_ESCALATION_EMAILS = os.getenv('INCIDENT_RESPONSE_ESCALATION_EMAILS', '').split(',') if os.getenv('INCIDENT_RESPONSE_ESCALATION_EMAILS') else []  # Escalation contacts
INCIDENT_RESPONSE_AUTO_CONTAINMENT = os.getenv('INCIDENT_RESPONSE_AUTO_CONTAINMENT', 'false').lower() == 'true'  # Auto-containment for incidents