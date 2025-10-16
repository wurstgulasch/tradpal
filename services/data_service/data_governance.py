#!/usr/bin/env python3
"""
Data Governance System for TradPal Data Mesh

Implements comprehensive data governance including:
- Access Control and Authorization
- Audit Logging and Compliance
- Data Quality Monitoring
- Policy Management
- Data Lineage Tracking
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from pathlib import Path
import os

import pandas as pd
from pydantic import BaseModel, Field

# Optional imports for governance features
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Access levels for data governance."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"


class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


class AuditEventType(Enum):
    """Types of audit events."""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    POLICY_CHANGE = "policy_change"
    QUALITY_CHECK = "quality_check"
    GOVERNANCE_VIOLATION = "governance_violation"


class GovernancePolicy(BaseModel):
    """Data governance policy definition."""
    name: str
    description: str
    resource_type: str  # "data_product", "domain", "feature_set"
    resource_name: str
    rules: Dict[str, Any]
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"


class AccessRequest(BaseModel):
    """Access request for data resources."""
    user: str
    resource_type: str
    resource_name: str
    access_level: AccessLevel
    purpose: str
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class AuditEvent(BaseModel):
    """Audit event for compliance tracking."""
    event_id: str
    event_type: AuditEventType
    user: str
    resource_type: str
    resource_name: str
    action: str
    details: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class DataQualityRule(BaseModel):
    """Data quality validation rule."""
    name: str
    description: str
    rule_type: str  # "completeness", "accuracy", "consistency", "timeliness"
    parameters: Dict[str, Any]
    severity: str = "medium"  # "low", "medium", "high", "critical"
    enabled: bool = True


class QualityCheckResult(BaseModel):
    """Result of a data quality check."""
    check_id: str
    resource_name: str
    rule_name: str
    quality_score: float
    quality_level: DataQuality
    issues_found: List[str]
    checked_at: datetime = Field(default_factory=datetime.utcnow)
    execution_time: float


class AccessControlManager:
    """Manages access control for data resources."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(config.get("redis_url", "redis://localhost:6379"))
                logger.info("Access Control Redis client initialized")
            except Exception as e:
                logger.error(f"Access Control Redis initialization failed: {e}")

        # In-memory policy cache
        self.policies: Dict[str, GovernancePolicy] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self.role_permissions: Dict[str, Set[str]] = {}

        # Default roles and permissions
        self._initialize_default_permissions()

    def _initialize_default_permissions(self):
        """Initialize default roles and permissions."""
        self.role_permissions = {
            "data_admin": {
                "data_product:*",
                "domain:*",
                "feature_set:*",
                "policy:*"
            },
            "data_steward": {
                "data_product:read",
                "data_product:write",
                "domain:read",
                "quality:check"
            },
            "data_consumer": {
                "data_product:read",
                "domain:read"
            },
            "ml_engineer": {
                "feature_set:*",
                "data_product:read",
                "domain:read"
            },
            "trading_service": {
                "data_product:read",
                "domain:market_data:read",
                "domain:trading_signals:*"
            }
        }

    async def assign_role(self, user: str, role: str) -> bool:
        """Assign a role to a user."""
        if role not in self.role_permissions:
            logger.warning(f"Unknown role: {role}")
            return False

        if user not in self.user_roles:
            self.user_roles[user] = set()

        self.user_roles[user].add(role)

        # Persist to Redis if available
        if self.redis_client:
            key = f"access_control:user_roles:{user}"
            self.redis_client.sadd(key, role)

        logger.info(f"Assigned role '{role}' to user '{user}'")
        return True

    async def revoke_role(self, user: str, role: str) -> bool:
        """Revoke a role from a user."""
        if user not in self.user_roles:
            return False

        self.user_roles[user].discard(role)

        # Persist to Redis if available
        if self.redis_client:
            key = f"access_control:user_roles:{user}"
            self.redis_client.srem(key, role)

        logger.info(f"Revoked role '{role}' from user '{user}'")
        return True

    async def check_access(self, user: str, resource_type: str, resource_name: str,
                          access_level: AccessLevel) -> Tuple[bool, str]:
        """
        Check if user has access to a resource.

        Returns:
            Tuple of (has_access, reason)
        """
        # Get user roles
        user_roles = self.user_roles.get(user, set())

        # Check direct permissions
        for role in user_roles:
            permissions = self.role_permissions.get(role, set())
            required_permission = f"{resource_type}:{resource_name}:{access_level.value}"

            # Check wildcard permissions
            for perm in permissions:
                if (perm == required_permission or 
                    perm == f"{resource_type}:*:{access_level.value}" or 
                    perm == f"{resource_type}:{resource_name}:*" or
                    perm == f"{resource_type}:*" or
                    perm == "*" or
                    (perm.endswith(":*") and perm.startswith(f"{resource_type}:")) or
                    (perm == "data_product:*" and resource_type == "data_product")):
                    return True, f"Access granted via role '{role}'"

        # Check resource-specific policies
        policy_key = f"{resource_type}:{resource_name}"
        if policy_key in self.policies:
            policy = self.policies[policy_key]
            if self._evaluate_policy(policy, user, access_level):
                return True, f"Access granted via policy '{policy.name}'"

        return False, "Access denied: insufficient permissions"

    def _evaluate_policy(self, policy: GovernancePolicy, user: str, access_level: AccessLevel) -> bool:
        """Evaluate a governance policy."""
        rules = policy.rules

        # Check user allowlist
        if "allowed_users" in rules and user in rules["allowed_users"]:
            return True

        # Check role requirements
        if "required_roles" in rules:
            user_roles = self.user_roles.get(user, set())
            if any(role in user_roles for role in rules["required_roles"]):
                return True

        # Check access level requirements
        if "max_access_level" in rules:
            max_level = AccessLevel(rules["max_access_level"])
            if access_level.value <= max_level.value:
                return True

        return False

    async def create_policy(self, policy: GovernancePolicy) -> bool:
        """Create a new governance policy."""
        policy_key = f"{policy.resource_type}:{policy.resource_name}"
        self.policies[policy_key] = policy

        # Persist to Redis if available
        if self.redis_client:
            key = f"access_control:policies:{policy_key}"
            self.redis_client.set(key, policy.json())

        logger.info(f"Created governance policy: {policy.name}")
        return True

    async def get_user_permissions(self, user: str) -> Dict[str, Any]:
        """Get all permissions for a user."""
        roles = self.user_roles.get(user, set())
        permissions = set()

        for role in roles:
            permissions.update(self.role_permissions.get(role, set()))

        return {
            "user": user,
            "roles": list(roles),
            "permissions": list(permissions),
            "policies": [p.name for p in self.policies.values() if self._user_matches_policy(p, user)]
        }

    def _user_matches_policy(self, policy: GovernancePolicy, user: str) -> bool:
        """Check if user matches policy criteria."""
        rules = policy.rules

        if "allowed_users" in rules and user in rules["allowed_users"]:
            return True

        if "required_roles" in rules:
            user_roles = self.user_roles.get(user, set())
            return any(role in user_roles for role in rules["required_roles"])

        return False


class AuditLogger:
    """Comprehensive audit logging for data governance."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(config.get("redis_url", "redis://localhost:6379"))
                logger.info("Audit Logger Redis client initialized")
            except Exception as e:
                logger.error(f"Audit Logger Redis initialization failed: {e}")

        # In-memory event storage (for quick access)
        self.recent_events: List[AuditEvent] = []
        self.max_recent_events = config.get("max_recent_events", 1000)

    async def log_event(self, event: AuditEvent) -> bool:
        """Log an audit event."""
        try:
            # Add to recent events
            self.recent_events.append(event)
            if len(self.recent_events) > self.max_recent_events:
                self.recent_events.pop(0)

            # Persist to Redis if available
            if self.redis_client:
                key = f"audit:events:{event.event_id}"
                self.redis_client.setex(key, 86400 * 30, event.json())  # 30 days retention

                # Add to time-based index
                time_key = f"audit:timeline:{event.timestamp.strftime('%Y%m%d')}"
                self.redis_client.sadd(time_key, event.event_id)

            logger.info(f"Audit event logged: {event.event_type.value} by {event.user}")
            return True

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return False

    async def log_access(self, user: str, resource_type: str, resource_name: str,
                        action: str, success: bool = True, details: Optional[Dict[str, Any]] = None,
                        ip_address: Optional[str] = None) -> bool:
        """Log a data access event."""
        event = AuditEvent(
            event_id=f"access_{datetime.utcnow().timestamp()}_{user}",
            event_type=AuditEventType.DATA_ACCESS,
            user=user,
            resource_type=resource_type,
            resource_name=resource_name,
            action=action,
            details=details or {},
            ip_address=ip_address,
            success=success
        )

        return await self.log_event(event)

    async def log_modification(self, user: str, resource_type: str, resource_name: str,
                             action: str, details: Dict[str, Any],
                             ip_address: Optional[str] = None) -> bool:
        """Log a data modification event."""
        event = AuditEvent(
            event_id=f"mod_{datetime.utcnow().timestamp()}_{user}",
            event_type=AuditEventType.DATA_MODIFICATION,
            user=user,
            resource_type=resource_type,
            resource_name=resource_name,
            action=action,
            details=details,
            ip_address=ip_address
        )

        return await self.log_event(event)

    async def log_policy_change(self, user: str, policy_name: str, action: str,
                               details: Dict[str, Any]) -> bool:
        """Log a policy change event."""
        event = AuditEvent(
            event_id=f"policy_{datetime.utcnow().timestamp()}_{user}",
            event_type=AuditEventType.POLICY_CHANGE,
            user=user,
            resource_type="policy",
            resource_name=policy_name,
            action=action,
            details=details
        )

        return await self.log_event(event)

    async def get_events(self, user: Optional[str] = None, resource_type: Optional[str] = None,
                        event_type: Optional[AuditEventType] = None,
                        start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                        limit: int = 100) -> List[AuditEvent]:
        """Retrieve audit events with filtering."""
        events = self.recent_events.copy()

        # Apply filters
        if user:
            events = [e for e in events if e.user == user]
        if resource_type:
            events = [e for e in events if e.resource_type == resource_type]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]

        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    async def get_user_activity(self, user: str, days: int = 30) -> Dict[str, Any]:
        """Get activity summary for a user."""
        start_date = datetime.utcnow() - timedelta(days=days)
        events = await self.get_events(user=user, start_date=start_date)

        summary = {
            "user": user,
            "period_days": days,
            "total_events": len(events),
            "event_types": {},
            "resource_types": {},
            "daily_activity": {}
        }

        for event in events:
            # Count event types
            et = event.event_type.value
            summary["event_types"][et] = summary["event_types"].get(et, 0) + 1

            # Count resource types
            rt = event.resource_type
            summary["resource_types"][rt] = summary["resource_types"].get(rt, 0) + 1

            # Daily activity
            day = event.timestamp.strftime("%Y-%m-%d")
            summary["daily_activity"][day] = summary["daily_activity"].get(day, 0) + 1

        return summary

    async def export_audit_log(self, start_date: datetime, end_date: datetime,
                              format: str = "json") -> str:
        """Export audit log for a date range."""
        events = await self.get_events(start_date=start_date, end_date=end_date, limit=10000)

        if format == "json":
            return json.dumps([event.dict() for event in events], indent=2, default=str)
        elif format == "csv":
            if not events:
                return "No events found"

            # Create CSV header
            header = "event_id,event_type,user,resource_type,resource_name,action,timestamp,success\n"

            # Create CSV rows
            rows = []
            for event in events:
                row = ",".join([
                    event.event_id,
                    event.event_type.value,
                    event.user,
                    event.resource_type,
                    event.resource_name,
                    event.action,
                    event.timestamp.isoformat(),
                    str(event.success)
                ])
                rows.append(row)

            return header + "\n".join(rows)

        return "Unsupported format"


class DataQualityMonitor:
    """Monitors and validates data quality across the data mesh."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(config.get("redis_url", "redis://localhost:6379"))
                logger.info("Data Quality Monitor Redis client initialized")
            except Exception as e:
                logger.error(f"Data Quality Monitor Redis initialization failed: {e}")

        # Quality rules
        self.quality_rules: Dict[str, DataQualityRule] = {}
        self._initialize_default_rules()

        # Quality check results
        self.check_results: List[QualityCheckResult] = []
        self.max_results = config.get("max_results", 1000)

    def _initialize_default_rules(self):
        """Initialize default data quality rules."""
        default_rules = [
            DataQualityRule(
                name="completeness_check",
                description="Check for missing values in critical columns",
                rule_type="completeness",
                parameters={"max_missing_ratio": 0.05, "critical_columns": ["close", "volume"]},
                severity="high"
            ),
            DataQualityRule(
                name="ohlc_consistency",
                description="Validate OHLC relationships (O <= H, L <= O, C <= H, C >= L)",
                rule_type="consistency",
                parameters={},
                severity="critical"
            ),
            DataQualityRule(
                name="volume_validity",
                description="Check for negative or zero volume values",
                rule_type="accuracy",
                parameters={"allow_zero_volume": False},
                severity="medium"
            ),
            DataQualityRule(
                name="price_reasonableness",
                description="Check for unreasonable price movements",
                rule_type="accuracy",
                parameters={"max_daily_change": 0.5},  # 50% daily change threshold
                severity="high"
            ),
            DataQualityRule(
                name="timeliness_check",
                description="Check data freshness and timeliness",
                rule_type="timeliness",
                parameters={"max_age_hours": 24},
                severity="medium"
            )
        ]

        for rule in default_rules:
            self.quality_rules[rule.name] = rule

    async def add_quality_rule(self, rule: DataQualityRule) -> bool:
        """Add a new quality rule."""
        self.quality_rules[rule.name] = rule

        # Persist to Redis if available
        if self.redis_client:
            key = f"quality:rules:{rule.name}"
            self.redis_client.set(key, rule.json())

        logger.info(f"Added quality rule: {rule.name}")
        return True

    async def check_data_quality(self, resource_name: str, data: pd.DataFrame,
                                resource_type: str = "data_product") -> QualityCheckResult:
        """Perform comprehensive data quality check."""
        start_time = datetime.utcnow()

        issues_found = []
        total_score = 1.0

        # Apply each enabled rule
        for rule in self.quality_rules.values():
            if not rule.enabled:
                continue

            rule_score, rule_issues = await self._apply_quality_rule(rule, data)
            total_score *= rule_score

            if rule_issues:
                issues_found.extend([f"{rule.name}: {issue}" for issue in rule_issues])

        # Determine overall quality level
        if total_score >= 0.95:
            quality_level = DataQuality.EXCELLENT
        elif total_score >= 0.85:
            quality_level = DataQuality.GOOD
        elif total_score >= 0.70:
            quality_level = DataQuality.FAIR
        elif total_score >= 0.50:
            quality_level = DataQuality.POOR
        else:
            quality_level = DataQuality.INVALID

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        result = QualityCheckResult(
            check_id=f"qc_{datetime.utcnow().timestamp()}_{resource_name}",
            resource_name=resource_name,
            rule_name="comprehensive_check",
            quality_score=total_score,
            quality_level=quality_level,
            issues_found=issues_found,
            execution_time=execution_time
        )

        # Store result
        self.check_results.append(result)
        if len(self.check_results) > self.max_results:
            self.check_results.pop(0)

        # Persist to Redis if available
        if self.redis_client:
            key = f"quality:results:{result.check_id}"
            self.redis_client.setex(key, 86400 * 7, result.json())  # 7 days retention

        logger.info(f"Quality check completed for {resource_name}: {quality_level.value} ({total_score:.3f})")
        return result

    async def _apply_quality_rule(self, rule: DataQualityRule, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Apply a specific quality rule to data."""
        issues = []

        if rule.rule_type == "completeness":
            score, rule_issues = self._check_completeness(data, rule.parameters)
            issues.extend(rule_issues)

        elif rule.rule_type == "consistency":
            score, rule_issues = self._check_ohlc_consistency(data)
            issues.extend(rule_issues)

        elif rule.rule_type == "accuracy":
            if "allow_zero_volume" in rule.parameters:
                score, rule_issues = self._check_volume_validity(data, rule.parameters)
                issues.extend(rule_issues)
            elif "max_daily_change" in rule.parameters:
                score, rule_issues = self._check_price_reasonableness(data, rule.parameters)
                issues.extend(rule_issues)
            else:
                score = 1.0

        elif rule.rule_type == "timeliness":
            score, rule_issues = self._check_timeliness(data, rule.parameters)
            issues.extend(rule_issues)

        else:
            score = 1.0

        return score, issues

    def _check_completeness(self, data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check data completeness."""
        issues = []
        max_missing_ratio = params.get("max_missing_ratio", 0.05)
        critical_columns = params.get("critical_columns", ["close"])

        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0

        score = max(0, 1 - (missing_ratio / max_missing_ratio))

        if missing_ratio > max_missing_ratio:
            issues.append(f"Missing data ratio {missing_ratio:.3f} exceeds threshold {max_missing_ratio}")

        # Check critical columns
        for col in critical_columns:
            if col in data.columns:
                col_missing = data[col].isnull().sum()
                if col_missing > 0:
                    issues.append(f"Critical column '{col}' has {col_missing} missing values")

        return score, issues

    def _check_ohlc_consistency(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Check OHLC data consistency."""
        issues = []
        score = 1.0

        required_cols = ["open", "high", "low", "close"]
        if not all(col in data.columns for col in required_cols):
            issues.append("Missing required OHLC columns")
            return 0.0, issues

        # Check OHLC relationships
        invalid_count = 0
        total_count = len(data)

        for idx, row in data.iterrows():
            try:
                o, h, l, c = row["open"], row["high"], row["low"], row["close"]

                if not (l <= o <= h and l <= c <= h):
                    invalid_count += 1
                    issues.append(f"Invalid OHLC at {idx}: O={o}, H={h}, L={l}, C={c}")
            except (TypeError, ValueError):
                invalid_count += 1
                issues.append(f"Invalid OHLC data types at {idx}")

        if total_count > 0:
            invalid_ratio = invalid_count / total_count
            score = max(0, 1 - invalid_ratio * 2)  # More severe penalty

            if invalid_ratio > 0.1:  # More than 10% invalid
                issues.append(f"High OHLC inconsistency ratio: {invalid_ratio:.3f}")

        return score, issues[:10]  # Limit issues to first 10

    def _check_volume_validity(self, data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check volume data validity."""
        issues = []
        allow_zero = params.get("allow_zero_volume", False)

        if "volume" not in data.columns:
            return 1.0, []

        volume_data = data["volume"].dropna()
        invalid_count = 0

        if not allow_zero:
            invalid_count += (volume_data <= 0).sum()
        else:
            invalid_count += (volume_data < 0).sum()

        total_valid = len(volume_data)
        if total_valid > 0:
            invalid_ratio = invalid_count / total_valid
            score = max(0, 1 - invalid_ratio)

            if invalid_ratio > 0.05:
                issues.append(f"Invalid volume ratio: {invalid_ratio:.3f}")
        else:
            score = 0.0
            issues.append("No valid volume data found")

        return score, issues

    def _check_price_reasonableness(self, data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check price movement reasonableness."""
        issues = []
        max_daily_change = params.get("max_daily_change", 0.5)

        if "close" not in data.columns:
            return 1.0, []

        # Calculate daily returns
        if hasattr(data.index, 'date'):
            daily_data = data.groupby(data.index.date)["close"].agg(["first", "last"])
            daily_returns = (daily_data["last"] - daily_data["first"]) / daily_data["first"]
        else:
            # Assume data is already daily
            daily_returns = data["close"].pct_change()

        extreme_changes = daily_returns.abs() > max_daily_change
        extreme_count = extreme_changes.sum()

        total_days = len(daily_returns)
        if total_days > 0:
            extreme_ratio = extreme_count / total_days
            score = max(0, 1 - extreme_ratio)

            if extreme_ratio > 0.1:  # More than 10% extreme days
                issues.append(f"Extreme price movement ratio: {extreme_ratio:.3f}")
        else:
            score = 1.0

        return score, issues

    def _check_timeliness(self, data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check data timeliness."""
        issues = []
        max_age_hours = params.get("max_age_hours", 24)

        if data.empty:
            return 0.0, ["No data available"]

        # Check if data has timestamps
        if hasattr(data.index, 'max'):
            latest_timestamp = data.index.max()
            if isinstance(latest_timestamp, pd.Timestamp):
                age_hours = (datetime.utcnow() - latest_timestamp.to_pydatetime()).total_seconds() / 3600
                score = max(0, 1 - (age_hours / max_age_hours))

                if age_hours > max_age_hours:
                    issues.append(f"Data is {age_hours:.1f} hours old (max allowed: {max_age_hours})")
            else:
                score = 0.5  # Unknown timeliness
                issues.append("Unable to determine data age")
        else:
            score = 0.5
            issues.append("No timestamp information available")

        return score, issues

    async def get_quality_history(self, resource_name: str, days: int = 7) -> List[QualityCheckResult]:
        """Get quality check history for a resource."""
        start_date = datetime.utcnow() - timedelta(days=days)

        # Filter results
        history = [
            result for result in self.check_results
            if result.resource_name == resource_name and result.checked_at >= start_date
        ]

        return sorted(history, key=lambda x: x.checked_at, reverse=True)

    async def get_quality_summary(self, resource_name: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        """Get quality summary for resources."""
        start_date = datetime.utcnow() - timedelta(days=days)

        # Filter results
        results = [
            result for result in self.check_results
            if result.checked_at >= start_date and
            (resource_name is None or result.resource_name == resource_name)
        ]

        if not results:
            return {"message": "No quality checks found in the specified period"}

        summary = {
            "period_days": days,
            "total_checks": len(results),
            "average_score": sum(r.quality_score for r in results) / len(results),
            "quality_distribution": {},
            "resources_checked": set(),
            "issues_summary": {}
        }

        for result in results:
            # Quality distribution
            level = result.quality_level.value
            summary["quality_distribution"][level] = summary["quality_distribution"].get(level, 0) + 1

            # Resources
            summary["resources_checked"].add(result.resource_name)

            # Issues
            for issue in result.issues_found:
                summary["issues_summary"][issue] = summary["issues_summary"].get(issue, 0) + 1

        summary["resources_checked"] = list(summary["resources_checked"])
        return summary


class DataGovernanceManager:
    """
    Central manager for all data governance components.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize components
        self.access_control = AccessControlManager(config)
        self.audit_logger = AuditLogger(config)
        self.quality_monitor = DataQualityMonitor(config)

        # Governance policies
        self.policies: Dict[str, GovernancePolicy] = {}

        logger.info("Data Governance Manager initialized")

    async def check_access_and_log(self, user: str, resource_type: str, resource_name: str,
                                  access_level: AccessLevel, purpose: str = "",
                                  ip_address: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check access and log the attempt.

        Returns:
            Tuple of (has_access, reason)
        """
        has_access, reason = await self.access_control.check_access(
            user, resource_type, resource_name, access_level
        )

        # Log the access attempt
        await self.audit_logger.log_access(
            user=user,
            resource_type=resource_type,
            resource_name=resource_name,
            action=f"{access_level.value}_access",
            success=has_access,
            details={"purpose": purpose, "reason": reason},
            ip_address=ip_address
        )

        return has_access, reason

    async def validate_data_and_log(self, user: str, resource_name: str, data: pd.DataFrame,
                                   resource_type: str = "data_product") -> QualityCheckResult:
        """Validate data quality and log the check."""
        # Perform quality check
        result = await self.quality_monitor.check_data_quality(resource_name, data, resource_type)

        # Log the quality check
        await self.audit_logger.log_event(AuditEvent(
            event_id=f"qc_{datetime.utcnow().timestamp()}_{user}",
            event_type=AuditEventType.QUALITY_CHECK,
            user=user,
            resource_type=resource_type,
            resource_name=resource_name,
            action="quality_check",
            details={
                "quality_score": result.quality_score,
                "quality_level": result.quality_level.value,
                "issues_count": len(result.issues_found),
                "execution_time": result.execution_time
            },
            success=result.quality_level != DataQuality.INVALID
        ))

        return result

    async def create_governance_policy(self, policy: GovernancePolicy) -> bool:
        """Create a new governance policy."""
        # Store policy
        self.policies[policy.name] = policy

        # Create access control policy
        await self.access_control.create_policy(policy)

        # Log policy creation
        await self.audit_logger.log_policy_change(
            user=policy.created_by,
            policy_name=policy.name,
            action="create",
            details={"policy": policy.dict()}
        )

        logger.info(f"Created governance policy: {policy.name}")
        return True

    async def assign_user_role(self, admin_user: str, target_user: str, role: str) -> bool:
        """Assign a role to a user (admin operation)."""
        success = await self.access_control.assign_role(target_user, role)

        if success:
            await self.audit_logger.log_event(AuditEvent(
                event_id=f"role_{datetime.utcnow().timestamp()}_{admin_user}",
                event_type=AuditEventType.POLICY_CHANGE,
                user=admin_user,
                resource_type="user",
                resource_name=target_user,
                action=f"assign_role:{role}",
                details={"role": role, "target_user": target_user}
            ))

        return success

    async def get_governance_status(self) -> Dict[str, Any]:
        """Get comprehensive governance status."""
        user_permissions = await self.access_control.get_user_permissions("system")

        quality_summary = await self.quality_monitor.get_quality_summary(days=1)

        recent_audit_events = await self.audit_logger.get_events(limit=10)

        return {
            "service": "data_governance",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "access_control": {
                    "total_policies": len(self.access_control.policies),
                    "total_users": len(self.access_control.user_roles),
                    "total_roles": len(self.access_control.role_permissions)
                },
                "audit_logging": {
                    "recent_events": len(recent_audit_events),
                    "event_types": list(set(e.event_type.value for e in recent_audit_events))
                },
                "quality_monitoring": {
                    "total_rules": len(self.quality_monitor.quality_rules),
                    "quality_checks_today": quality_summary.get("total_checks", 0),
                    "average_quality_score": quality_summary.get("average_score", 0.0)
                }
            },
            "policies": [p.dict() for p in self.policies.values()],
            "active_roles": list(self.access_control.role_permissions.keys())
        }

    async def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate a compliance report for the specified period."""
        # Get audit events
        events = await self.audit_logger.get_events(
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )

        # Get quality summary
        quality_summary = await self.quality_monitor.get_quality_summary(
            days=(end_date - start_date).days
        )

        # Analyze events
        compliance_stats = {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": (end_date - start_date).days
            },
            "audit_events": {
                "total": len(events),
                "by_type": {},
                "by_user": {},
                "success_rate": 0.0
            },
            "access_attempts": {
                "total": 0,
                "granted": 0,
                "denied": 0
            },
            "data_quality": quality_summary,
            "policy_violations": 0,
            "recommendations": []
        }

        # Analyze events
        successful_events = 0
        for event in events:
            # Count by type
            et = event.event_type.value
            compliance_stats["audit_events"]["by_type"][et] = compliance_stats["audit_events"]["by_type"].get(et, 0) + 1

            # Count by user
            compliance_stats["audit_events"]["by_user"][event.user] = compliance_stats["audit_events"]["by_user"].get(event.user, 0) + 1

            # Track success
            if event.success:
                successful_events += 1

            # Track access attempts
            if event.event_type == AuditEventType.DATA_ACCESS:
                compliance_stats["access_attempts"]["total"] += 1
                if event.success:
                    compliance_stats["access_attempts"]["granted"] += 1
                else:
                    compliance_stats["access_attempts"]["denied"] += 1

            # Track violations
            if event.event_type == AuditEventType.GOVERNANCE_VIOLATION:
                compliance_stats["policy_violations"] += 1

        # Calculate success rate
        if events:
            compliance_stats["audit_events"]["success_rate"] = successful_events / len(events)

        # Generate recommendations
        if compliance_stats["access_attempts"]["denied"] > compliance_stats["access_attempts"]["granted"] * 0.1:
            compliance_stats["recommendations"].append("High access denial rate - review access policies")

        if compliance_stats["policy_violations"] > 0:
            compliance_stats["recommendations"].append("Policy violations detected - review governance policies")

        if quality_summary.get("average_score", 1.0) < 0.8:
            compliance_stats["recommendations"].append("Data quality issues detected - review data validation rules")

        return compliance_stats