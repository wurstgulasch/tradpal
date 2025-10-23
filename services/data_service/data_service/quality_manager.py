"""
Data Quality Manager for data validation and quality assurance.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataQualityManager:
    """
    Manages data quality validation and monitoring.

    Provides comprehensive data quality checks including completeness,
    consistency, timeliness, and accuracy validation.
    """

    def __init__(self):
        self.quality_thresholds = {
            'completeness': 0.95,
            'consistency': 0.90,
            'timeliness': 0.95,
            'accuracy': 0.85
        }

    def validate_data_quality(self, df: pd.DataFrame, data_type: str = 'market_data') -> bool:
        """
        Validate overall data quality for a DataFrame.

        Args:
            df: DataFrame to validate
            data_type: Type of data (market_data, features, etc.)

        Returns:
            True if data passes quality checks
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for quality validation")
            return False

        try:
            # Calculate quality metrics
            metrics = self.calculate_quality_metrics(df)

            # Check against thresholds
            for metric_name, threshold in self.quality_thresholds.items():
                if metric_name in metrics:
                    if metrics[metric_name] < threshold:
                        logger.warning(f"Quality check failed for {metric_name}: {metrics[metric_name]:.3f} < {threshold}")
                        return False

            # Data type specific validations
            if data_type == 'market_data':
                return self._validate_market_data(df)
            elif data_type == 'liquidation':
                return self._validate_liquidation_data(df)
            elif data_type == 'sentiment':
                return self._validate_sentiment_data(df)
            elif data_type == 'onchain':
                return self._validate_onchain_data(df)

            return True

        except Exception as e:
            logger.error(f"Data quality validation failed: {e}")
            return False

    def calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive quality metrics for a DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        if df.empty:
            return {'completeness': 0.0, 'consistency': 0.0, 'timeliness': 0.0, 'accuracy': 0.0}

        try:
            # Completeness: Percentage of non-null values
            total_cells = df.shape[0] * df.shape[1]
            non_null_cells = df.count().sum()
            metrics['completeness'] = non_null_cells / total_cells if total_cells > 0 else 0.0

            # Consistency: Logical relationships between columns
            consistency_score = 1.0

            # Check OHLC relationships if present
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                invalid_ohlc = (
                    (df['open'] > df['high']) |
                    (df['low'] > df['open']) |
                    (df['close'] > df['high']) |
                    (df['close'] < df['low'])
                ).sum()
                ohlc_consistency = 1.0 - (invalid_ohlc / len(df))
                consistency_score = min(consistency_score, ohlc_consistency)

            # Check volume validity
            if 'volume' in df.columns:
                negative_volume = (df['volume'] < 0).sum()
                volume_consistency = 1.0 - (negative_volume / len(df))
                consistency_score = min(consistency_score, volume_consistency)

            metrics['consistency'] = consistency_score

            # Timeliness: Check for reasonable time gaps
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
                time_diffs = df.index.to_series().diff().dropna()
                median_diff = time_diffs.median()

                # Check for unreasonable gaps (more than 10x median)
                large_gaps = (time_diffs > median_diff * 10).sum()
                timeliness_score = 1.0 - (large_gaps / len(time_diffs))
                metrics['timeliness'] = max(0.0, timeliness_score)
            else:
                metrics['timeliness'] = 0.8  # Default for non-time series

            # Accuracy: Check for reasonable value ranges
            accuracy_score = 1.0

            # Price columns should be positive
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    negative_prices = (df[col] <= 0).sum()
                    if negative_prices > 0:
                        accuracy_score -= (negative_prices / len(df)) * 0.5

            # Volume should be non-negative
            if 'volume' in df.columns:
                negative_volume = (df['volume'] < 0).sum()
                if negative_volume > 0:
                    accuracy_score -= (negative_volume / len(df)) * 0.3

            metrics['accuracy'] = max(0.0, accuracy_score)

        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            # Return conservative defaults
            metrics = {
                'completeness': 0.5,
                'consistency': 0.5,
                'timeliness': 0.5,
                'accuracy': 0.5
            }

        return metrics

    def _validate_market_data(self, df: pd.DataFrame) -> bool:
        """Validate market data specific requirements."""
        required_cols = ['open', 'high', 'low', 'close']

        # Check required columns exist
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns for market data: {required_cols}")
            return False

        # Check for minimum data points
        if len(df) < 10:
            logger.warning("Insufficient data points for market data validation")
            return False

        return True

    def _validate_liquidation_data(self, df: pd.DataFrame) -> bool:
        """Validate liquidation data specific requirements."""
        # Check for signal column
        if 'liquidation_signal' not in df.columns:
            logger.warning("Missing liquidation_signal column")
            return False

        # Check signal values are reasonable
        if not df['liquidation_signal'].between(-3, 3).all():
            logger.warning("Liquidation signals outside expected range [-3, 3]")
            return False

        return True

    def _validate_sentiment_data(self, df: pd.DataFrame) -> bool:
        """Validate sentiment data specific requirements."""
        # Check for sentiment signal
        if 'sentiment_signal' not in df.columns:
            logger.warning("Missing sentiment_signal column")
            return False

        # Check signal values are reasonable
        if not df['sentiment_signal'].between(-2, 2).all():
            logger.warning("Sentiment signals outside expected range [-2, 2]")
            return False

        return True

    def _validate_onchain_data(self, df: pd.DataFrame) -> bool:
        """Validate on-chain data specific requirements."""
        # Check for on-chain signal
        if 'onchain_signal' not in df.columns:
            logger.warning("Missing onchain_signal column")
            return False

        # Check signal values are reasonable
        if not df['onchain_signal'].between(-2, 2).all():
            logger.warning("On-chain signals outside expected range [-2, 2]")
            return False

        return True

    def get_quality_report(self, df: pd.DataFrame, data_type: str = 'market_data') -> Dict[str, Any]:
        """
        Generate detailed quality report.

        Args:
            df: DataFrame to analyze
            data_type: Type of data

        Returns:
            Detailed quality report
        """
        metrics = self.calculate_quality_metrics(df)
        is_valid = self.validate_data_quality(df, data_type)

        report = {
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'record_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'metrics': metrics,
            'overall_quality': 'pass' if is_valid else 'fail',
            'recommendations': []
        }

        # Generate recommendations based on metrics
        if metrics['completeness'] < 0.9:
            report['recommendations'].append("Consider data imputation for missing values")

        if metrics['consistency'] < 0.8:
            report['recommendations'].append("Review data for logical inconsistencies")

        if metrics['timeliness'] < 0.9:
            report['recommendations'].append("Check for data gaps or irregular timestamps")

        if metrics['accuracy'] < 0.8:
            report['recommendations'].append("Validate data ranges and business rules")

        return report

    def detect_data_anomalies(self, df: pd.DataFrame, sensitivity: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect statistical anomalies in the data.

        Args:
            df: DataFrame to analyze
            sensitivity: Z-score threshold for anomaly detection

        Returns:
            List of detected anomalies
        """
        anomalies = []

        try:
            # Check numeric columns for outliers
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if col in df.columns:
                    series = df[col].dropna()
                    if len(series) > 10:  # Need minimum data points
                        mean_val = series.mean()
                        std_val = series.std()

                        if std_val > 0:
                            z_scores = np.abs((series - mean_val) / std_val)
                            outlier_indices = z_scores[z_scores > sensitivity].index

                            for idx in outlier_indices:
                                anomalies.append({
                                    'column': col,
                                    'index': str(idx),
                                    'value': float(series.loc[idx]),
                                    'z_score': float(z_scores.loc[idx]),
                                    'type': 'statistical_outlier'
                                })

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")

        return anomalies

    def compare_data_quality(self, df1: pd.DataFrame, df2: pd.DataFrame,
                           label1: str = 'dataset1', label2: str = 'dataset2') -> Dict[str, Any]:
        """
        Compare quality metrics between two datasets.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            label1: Label for first dataset
            label2: Label for second dataset

        Returns:
            Quality comparison report
        """
        metrics1 = self.calculate_quality_metrics(df1)
        metrics2 = self.calculate_quality_metrics(df2)

        comparison = {
            'datasets': {
                label1: {'record_count': len(df1), 'metrics': metrics1},
                label2: {'record_count': len(df2), 'metrics': metrics2}
            },
            'differences': {}
        }

        # Calculate differences
        for metric in metrics1.keys():
            if metric in metrics2:
                diff = metrics2[metric] - metrics1[metric]
                comparison['differences'][metric] = {
                    'absolute_difference': diff,
                    'relative_difference': diff / metrics1[metric] if metrics1[metric] != 0 else 0,
                    'better_dataset': label2 if diff > 0 else label1
                }

        return comparison