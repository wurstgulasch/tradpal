#!/usr/bin/env python3
"""
Performance Demo für TradPal Indicator
Vergleicht sequentielle vs. parallele Verarbeitung und zeigt Performance-Verbesserungen.
"""

import time
import logging
import sys
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Projekt-Module importieren
sys.path.append('..')
from src.performance import PerformanceOptimizer
from src.indicators import calculate_indicators
from config.settings import SYMBOL, TIMEFRAME, EXCHANGE

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceDemo:
    """Demo-Klasse für Performance-Vergleiche."""

    def __init__(self):
        self.optimizer = PerformanceOptimizer()
        self.test_data = None

    def generate_test_data(self, num_rows: int = 10000) -> pd.DataFrame:
        """Test-Daten generieren für Performance-Tests."""
        logger.info(f"Generiere {num_rows} Zeilen Test-Daten...")

        # Basis-Preisdaten generieren
        np.random.seed(42)  # Reproduzierbare Ergebnisse
        base_price = 50000
        prices = []
        volumes = []

        for i in range(num_rows):
            # Preis mit realistischer Volatilität
            price_change = np.random.normal(0, 0.002)  # ~0.2% tägliche Volatilität
            base_price *= (1 + price_change)
            prices.append(base_price)

            # Volumen mit Variation
            volume = np.random.lognormal(10, 0.5)  # Log-normal verteiltes Volumen
            volumes.append(volume)

        # OHLCV-Daten erstellen
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=num_rows, freq='1min'),
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'close': prices,
            'volume': volumes
        })

        # NaN-Werte entfernen und Indizes zurücksetzen
        df = df.dropna().reset_index(drop=True)
        self.test_data = df

        logger.info(f"Test-Daten generiert: {len(df)} Zeilen")
        return df

    def benchmark_sequential_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Benchmark für sequentielle Indikator-Berechnung."""
        logger.info("Starte sequentielle Indikator-Berechnung...")

        start_time = time.time()

        # Indikatoren sequentiell berechnen
        result_df = calculate_indicators(df.copy())

        end_time = time.time()
        duration = end_time - start_time

        return {
            'method': 'sequential',
            'duration': duration,
            'rows_processed': len(result_df),
            'indicators_calculated': len([col for col in result_df.columns if col not in df.columns])
        }

    def benchmark_parallel_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Benchmark für parallele Indikator-Berechnung."""
        logger.info("Starte parallele Indikator-Berechnung...")

        start_time = time.time()

        # Indikatoren parallel berechnen
        result_df = self.optimizer.calculate_indicators_parallel(df.copy())

        end_time = time.time()
        duration = end_time - start_time

        return {
            'method': 'parallel',
            'duration': duration,
            'rows_processed': len(result_df),
            'indicators_calculated': len([col for col in result_df.columns if col not in df.columns])
        }

    def benchmark_vectorized_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Benchmark für vektorisierte Indikator-Berechnung."""
        logger.info("Starte vektorisierte Indikator-Berechnung...")

        start_time = time.time()

        # Indikatoren vektorisiert berechnen
        result_df = self.optimizer.calculate_indicators_vectorized(df.copy())

        end_time = time.time()
        duration = end_time - start_time

        return {
            'method': 'vectorized',
            'duration': duration,
            'rows_processed': len(result_df),
            'indicators_calculated': len([col for col in result_df.columns if col not in df.columns])
        }

    def benchmark_memory_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Benchmark für Speicheroptimierung."""
        logger.info("Starte Speicheroptimierung...")

        original_memory = df.memory_usage(deep=True).sum()

        start_time = time.time()

        # DataFrame optimieren
        optimized_df = self.optimizer.optimize_dataframe_memory(df.copy())

        end_time = time.time()
        duration = end_time - start_time

        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100

        return {
            'method': 'memory_optimization',
            'duration': duration,
            'original_memory_mb': original_memory / 1024 / 1024,
            'optimized_memory_mb': optimized_memory / 1024 / 1024,
            'memory_reduction_percent': memory_reduction
        }

    def run_performance_comparison(self, num_rows: int = 5000):
        """Vollständigen Performance-Vergleich durchführen."""
        print("🚀 TradPal Indicator Performance Demo")
        print("=" * 60)

        # Test-Daten generieren
        df = self.generate_test_data(num_rows)

        results = []

        # Verschiedene Methoden benchmarken
        print(f"\n📊 Benchmark mit {num_rows} Datenzeilen")
        print("-" * 60)

        # 1. Sequentielle Verarbeitung
        seq_result = self.benchmark_sequential_indicators(df)
        results.append(seq_result)
        print(".2f"
              ".2f")

        # 2. Parallele Verarbeitung
        par_result = self.benchmark_parallel_indicators(df)
        results.append(par_result)
        speedup_parallel = seq_result['duration'] / par_result['duration']
        print(".2f"
              ".2f")

        # 3. Vektorisierte Verarbeitung
        vec_result = self.benchmark_vectorized_indicators(df)
        results.append(vec_result)
        speedup_vectorized = seq_result['duration'] / vec_result['duration']
        print(".2f"
              ".2f")

        # 4. Speicheroptimierung
        mem_result = self.benchmark_memory_optimization(df)
        results.append(mem_result)
        print(".2f"
              ".2f")

        # Zusammenfassung
        print("\n🏆 Performance-Zusammenfassung")
        print("-" * 60)
        print(".1f")
        print(".1f")
        print(".1f")

        return results

    def demonstrate_real_time_processing(self):
        """Demonstriert Echtzeit-Verarbeitung mit WebSocket-Streaming."""
        print("\n🌐 Echtzeit-Verarbeitung Demo")
        print("-" * 60)

        # Simuliere Echtzeit-Daten-Stream
        print("Simuliere Echtzeit-Daten-Stream...")

        # Kleine Datenmenge für Demo
        df = self.generate_test_data(100)

        # Performance-Monitoring starten
        self.optimizer.start_monitoring()

        # Mehrere Verarbeitungsdurchläufe simulieren
        for i in range(5):
            logger.info(f"Verarbeitungsdurchlauf {i+1}/5")

            # Indikatoren berechnen
            result_df = self.optimizer.calculate_indicators_parallel(df.copy())

            # Kurze Pause simulieren
            time.sleep(0.1)

        # Monitoring stoppen und Bericht erstellen
        report = self.optimizer.stop_monitoring()

        print("Echtzeit-Performance-Bericht:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(f"  Durchschnittliche CPU-Auslastung: {report.get('avg_cpu_percent', 'N/A')}%")
        print(f"  Durchschnittliche Speichernutzung: {report.get('avg_memory_mb', 'N/A')} MB")


def main():
    """Hauptfunktion für die Performance-Demo."""
    demo = PerformanceDemo()

    try:
        # Performance-Vergleich durchführen
        demo.run_performance_comparison()

        # Echtzeit-Verarbeitung demonstrieren
        demo.demonstrate_real_time_processing()

        print("\n✅ Performance-Demo erfolgreich abgeschlossen!")

    except Exception as e:
        logger.error(f"Fehler während der Performance-Demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
