#!/usr/bin/env python3
"""
Data Quality Analysis and Fallback Recommendations

Analyzes data quality from different sources and provides recommendations
for fallback mechanisms and data validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_fetcher import fetch_historical_data
from src.indicators import ema, rsi, bb, atr, adx
from src.data_sources.factory import get_available_data_sources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityAnalyzer:
    """Analyzes data quality across different sources."""

    def __init__(self):
        self.sources = get_available_data_sources()
        self.symbol = 'BTC/USDT'
        self.timeframe = '1d'
        self.days = 365

    def analyze_source_quality(self, source_name: str) -> Dict:
        """Analyze data quality for a specific source."""
        try:
            # Temporarily set data source
            from src.data_fetcher import set_data_source
            set_data_source(source_name)

            # Fetch data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days)

            df = fetch_historical_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=start_date,
                limit=self.days
            )

            if df.empty:
                return {
                    'source': source_name,
                    'status': 'NO_DATA',
                    'records': 0,
                    'quality_score': 0,
                    'issues': ['No data available']
                }

            # Analyze data quality
            quality_metrics = self._calculate_quality_metrics(df)

            return {
                'source': source_name,
                'status': 'OK',
                'records': len(df),
                **quality_metrics
            }

        except Exception as e:
            return {
                'source': source_name,
                'status': 'ERROR',
                'records': 0,
                'quality_score': 0,
                'issues': [str(e)]
            }

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive quality metrics."""
        metrics = {}

        # Basic data completeness
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {
                'quality_score': 0,
                'issues': [f'Missing columns: {missing_cols}'],
                'completeness': 0,
                'validity': 0,
                'consistency': 0
            }

        # Data completeness (non-null values)
        total_cells = len(df) * len(required_cols)
        null_cells = df[required_cols].isnull().sum().sum()
        completeness = 1 - (null_cells / total_cells)

        # Data validity (OHLC relationships)
        valid_ohlc = (
            (df['high'] >= df['low']) &
            (df['open'] >= df['low']) & (df['open'] <= df['high']) &
            (df['close'] >= df['low']) & (df['close'] <= df['high'])
        )
        validity = valid_ohlc.mean()

        # Data consistency (price movements)
        price_changes = df['close'].pct_change().abs()
        reasonable_changes = (price_changes < 0.5).mean()  # Less than 50% daily change
        consistency = reasonable_changes

        # Volume analysis
        volume_valid = (df['volume'] > 0).mean()

        # Indicator calculation success
        indicator_success = self._test_indicators(df)

        # Overall quality score
        quality_score = (
            completeness * 0.3 +
            validity * 0.3 +
            consistency * 0.2 +
            volume_valid * 0.1 +
            indicator_success * 0.1
        ) * 100

        issues = []
        if completeness < 0.95:
            issues.append('.1%')
        if validity < 0.95:
            issues.append('.1%')
        if consistency < 0.9:
            issues.append('Unrealistic price movements detected')
        if volume_valid < 0.8:
            issues.append('Low volume data quality')
        if indicator_success < 0.8:
            issues.append('Indicator calculation issues')

        return {
            'quality_score': round(quality_score, 1),
            'completeness': round(completeness * 100, 1),
            'validity': round(validity * 100, 1),
            'consistency': round(consistency * 100, 1),
            'volume_quality': round(volume_valid * 100, 1),
            'indicator_success': round(indicator_success * 100, 1),
            'issues': issues if issues else ['Data quality acceptable']
        }

    def _test_indicators(self, df: pd.DataFrame) -> float:
        """Test indicator calculations and return success rate."""
        success_count = 0
        total_tests = 0

        try:
            # Test EMA
            total_tests += 1
            ema_short = ema(df['close'], 9)
            if not ema_short.isnull().all():
                success_count += 1

            # Test RSI
            total_tests += 1
            rsi_val = rsi(df['close'], 14)
            if not rsi_val.isnull().all():
                success_count += 1

            # Test Bollinger Bands
            total_tests += 1
            bb_upper, bb_middle, bb_lower = bb(df['close'], 20, 2)
            if not (bb_upper.isnull().all() or bb_middle.isnull().all() or bb_lower.isnull().all()):
                success_count += 1

            # Test ATR
            total_tests += 1
            atr_val = atr(df['high'], df['low'], df['close'], 14)
            if not atr_val.isnull().all():
                success_count += 1

            # Test ADX
            total_tests += 1
            adx_val, di_plus, di_minus = adx(df['high'], df['low'], df['close'], 14)
            if not adx_val.isnull().all():
                success_count += 1

        except Exception as e:
            logger.warning(f"Indicator test failed: {e}")

        return success_count / total_tests if total_tests > 0 else 0

    def compare_sources(self) -> pd.DataFrame:
        """Compare all available data sources."""
        results = []

        for source_name in self.sources.keys():
            if source_name in ['alpha_vantage', 'polygon']:  # Skip not implemented
                continue

            logger.info(f"Analyzing {source_name}...")
            result = self.analyze_source_quality(source_name)
            results.append(result)

        return pd.DataFrame(results)

    def generate_recommendations(self, comparison_df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Find best source
        if not comparison_df.empty:
            best_source = comparison_df.loc[comparison_df['quality_score'].idxmax()]
            recommendations.append(f"🏆 **Beste Datenquelle**: {best_source['source']} (Score: {best_source['quality_score']})")

            # Check for fallback needs
            low_quality_sources = comparison_df[comparison_df['quality_score'] < 70]
            if not low_quality_sources.empty:
                recommendations.append("⚠️ **Fallback benötigt für**: " + ", ".join(low_quality_sources['source'].tolist()))

            # Indicator issues
            if 'indicator_success' in comparison_df.columns:
                indicator_issues = comparison_df[comparison_df['indicator_success'] < 80]
                if not indicator_issues.empty:
                    recommendations.append("🔧 **Indikator-Fallbacks implementieren für**: " + ", ".join(indicator_issues['source'].tolist()))

        # General recommendations
        recommendations.extend([
            "📊 **Datenqualitäts-Monitoring implementieren**: Regelmäßige Validierung aller Datenquellen",
            "🔄 **Automatische Fallbacks**: Yahoo Finance → CCXT bei Qualitätsproblemen",
            "⚡ **Indikator-Caching**: Berechnete Indikatoren cachen zur Performance-Verbesserung",
            "📈 **Qualitäts-Metriken**: Datenqualität in Logs und Monitoring einbeziehen",
            "🛡️ **Validierung erweitern**: Zusätzliche Checks für extreme Werte und Anomalien"
        ])

        return recommendations

def main():
    """Main analysis function."""
    print("🔍 **TradPal Datenqualitäts-Analyse**")
    print("=" * 50)

    analyzer = DataQualityAnalyzer()

    # Compare sources
    comparison = analyzer.compare_sources()

    if comparison.empty:
        print("❌ Keine Datenquellen konnten analysiert werden")
        return

    # Display results
    print("\n📊 **Datenquellen-Vergleich**:")
    print(comparison.to_string(index=False))

    # Generate recommendations
    recommendations = analyzer.generate_recommendations(comparison)

    print("\n💡 **Empfehlungen**:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    print("\n✅ **Analyse abgeschlossen**")

if __name__ == "__main__":
    main()