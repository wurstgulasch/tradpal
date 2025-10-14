"""
Data Quality Monitor

Monitors data quality across sources and provides warnings/alerts
for quality issues that could affect trading signals.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from config.settings import SYMBOL, TIMEFRAME

logger = logging.getLogger(__name__)

class DataQualityMonitor:
    """Monitors data quality and provides alerts."""

    def __init__(self):
        self.quality_thresholds = {
            'completeness': 95.0,  # Minimum 95% non-null values
            'validity': 98.0,      # Minimum 98% valid OHLC relationships
            'consistency': 90.0,   # Minimum 90% reasonable price movements
            'volume_quality': 80.0,  # Minimum 80% valid volume data
            'overall_quality': 85.0   # Minimum 85% overall quality score
        }

        self.alerts = []

    def monitor_data_quality(self, df: pd.DataFrame, source_name: str) -> Dict:
        """
        Monitor data quality and generate alerts if needed.

        Args:
            df: DataFrame to monitor
            source_name: Name of the data source

        Returns:
            Quality report with alerts
        """
        self.alerts = []  # Reset alerts

        if df.empty:
            self._add_alert('CRITICAL', f"No data available from {source_name}")
            return {'quality_score': 0, 'alerts': self.alerts}

        # Calculate quality metrics
        metrics = self._calculate_quality_metrics(df)

        # Check thresholds and generate alerts
        self._check_thresholds(metrics, source_name)

        # Overall assessment
        overall_score = metrics['overall_score']
        if overall_score < self.quality_thresholds['overall_quality']:
            self._add_alert('WARNING',
                          f"Low overall data quality from {source_name}: {overall_score:.1f}%")

        report = {
            'source': source_name,
            'timestamp': datetime.now(),
            'quality_score': overall_score,
            'metrics': metrics,
            'alerts': self.alerts,
            'recommendations': self._generate_recommendations(metrics)
        }

        return report

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive quality metrics."""
        metrics = {}

        # Completeness
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        completeness_scores = {}
        for col in required_cols:
            if col in df.columns:
                completeness_scores[col] = (1 - df[col].isnull().mean()) * 100
            else:
                completeness_scores[col] = 0

        metrics['completeness'] = sum(completeness_scores.values()) / len(completeness_scores)

        # Validity (OHLC relationships)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['open'] < df['low']) | (df['open'] > df['high']) |
                (df['close'] < df['low']) | (df['close'] > df['high'])
            )
            metrics['validity'] = (1 - invalid_ohlc.mean()) * 100
        else:
            metrics['validity'] = 0

        # Consistency (price movement reasonableness)
        if 'close' in df.columns:
            price_changes = df['close'].pct_change().abs()
            reasonable_changes = (price_changes < 0.5).mean()  # Less than 50% daily change
            metrics['consistency'] = reasonable_changes * 100
        else:
            metrics['consistency'] = 0

        # Volume quality
        if 'volume' in df.columns:
            valid_volume = (df['volume'] > 0).mean()
            metrics['volume_quality'] = valid_volume * 100
        else:
            metrics['volume_quality'] = 0

        # Overall score (weighted average)
        weights = {
            'completeness': 0.3,
            'validity': 0.3,
            'consistency': 0.25,
            'volume_quality': 0.15
        }

        overall_score = sum(metrics[key] * weights[key] for key in weights.keys())
        metrics['overall_score'] = overall_score

        return metrics

    def _check_thresholds(self, metrics: Dict, source_name: str):
        """Check metrics against thresholds and add alerts."""
        threshold_checks = {
            'completeness': ('WARNING', 'Low data completeness'),
            'validity': ('ERROR', 'Invalid OHLC relationships detected'),
            'consistency': ('WARNING', 'Unrealistic price movements'),
            'volume_quality': ('INFO', 'Low volume data quality')
        }

        for metric, (level, message) in threshold_checks.items():
            if metric in metrics and metrics[metric] < self.quality_thresholds[metric]:
                self._add_alert(level, f"{message} from {source_name}: {metrics[metric]:.1f}%")

    def _add_alert(self, level: str, message: str):
        """Add an alert to the alerts list."""
        self.alerts.append({
            'level': level,
            'message': message,
            'timestamp': datetime.now()
        })

        # Log alerts
        if level == 'CRITICAL':
            logger.critical(message)
        elif level == 'ERROR':
            logger.error(message)
        elif level == 'WARNING':
            logger.warning(message)
        else:
            logger.info(message)

    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []

        if metrics['completeness'] < 90:
            recommendations.append("Consider switching to a more reliable data source")
        if metrics['validity'] < 95:
            recommendations.append("Validate OHLC data integrity before using in signals")
        if metrics['consistency'] < 85:
            recommendations.append("Check for data anomalies or gaps in the dataset")
        if metrics['volume_quality'] < 75:
            recommendations.append("Volume data may be unreliable - consider volume-free indicators")

        if not recommendations:
            recommendations.append("Data quality is acceptable")

        return recommendations

    def get_quality_report(self, df: pd.DataFrame, source_name: str) -> str:
        """
        Generate a human-readable quality report.

        Args:
            df: DataFrame to analyze
            source_name: Source name

        Returns:
            Formatted quality report
        """
        report = self.monitor_data_quality(df, source_name)

        output = f"""
📊 **Data Quality Report - {source_name}**
{'='*40}
Overall Quality Score: {report['quality_score']:.1f}%

📈 **Metrics:**
• Completeness: {report['metrics']['completeness']:.1f}%
• Validity: {report['metrics']['validity']:.1f}%
• Consistency: {report['metrics']['consistency']:.1f}%
• Volume Quality: {report['metrics']['volume_quality']:.1f}%

🚨 **Alerts:** {len(report['alerts'])}
"""

        for alert in report['alerts']:
            level_emoji = {
                'CRITICAL': '🚨',
                'ERROR': '❌',
                'WARNING': '⚠️',
                'INFO': 'ℹ️'
            }.get(alert['level'], '❓')
            output += f"{level_emoji} {alert['message']}\n"

        output += f"\n💡 **Recommendations:**\n"
        for rec in report['recommendations']:
            output += f"• {rec}\n"

        return output

# Global monitor instance
_monitor = None

def get_data_quality_monitor() -> DataQualityMonitor:
    """Get or create the global data quality monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = DataQualityMonitor()
    return _monitor

def monitor_data_source(df: pd.DataFrame, source_name: str) -> Dict:
    """
    Convenience function to monitor data quality.

    Args:
        df: DataFrame to monitor
        source_name: Source name

    Returns:
        Quality report
    """
    monitor = get_data_quality_monitor()
    return monitor.monitor_data_quality(df, source_name)