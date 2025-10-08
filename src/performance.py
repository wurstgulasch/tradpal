#!/usr/bin/env python3
"""
Performance Optimization Module für TradPal Indicator
Bietet parallele Verarbeitung, Vektorisierung und Speicheroptimierung.
"""

import time
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
import numpy as np
import psutil
import threading

from config.settings import (
    PERFORMANCE_ENABLED, PARALLEL_PROCESSING_ENABLED, VECTORIZATION_ENABLED,
    MEMORY_OPTIMIZATION_ENABLED, PERFORMANCE_MONITORING_ENABLED,
    MAX_WORKERS, CHUNK_SIZE, PERFORMANCE_LOG_LEVEL
)

# Logging konfigurieren
logging.basicConfig(level=getattr(logging, PERFORMANCE_LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Überwacht System-Performance während der Ausführung."""

    def __init__(self):
        self.start_time = None
        self.cpu_percentages = []
        self.memory_usages = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Starte Performance-Monitoring."""
        if not PERFORMANCE_MONITORING_ENABLED:
            return

        self.start_time = time.time()
        self.cpu_percentages = []
        self.memory_usages = []
        self.monitoring = True

        # Starte Monitoring-Thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance-Monitoring gestartet")

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stoppe Performance-Monitoring und erstelle Bericht."""
        if not self.monitoring:
            return {}

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        end_time = time.time()
        total_duration = end_time - (self.start_time or end_time)

        # Berechne Statistiken
        report = {
            'total_duration': total_duration,
            'avg_cpu_percent': np.mean(self.cpu_percentages) if self.cpu_percentages else 0,
            'max_cpu_percent': np.max(self.cpu_percentages) if self.cpu_percentages else 0,
            'avg_memory_mb': np.mean(self.memory_usages) if self.memory_usages else 0,
            'max_memory_mb': np.max(self.memory_usages) if self.memory_usages else 0,
            'samples_collected': len(self.cpu_percentages)
        }

        logger.info(f"Performance-Monitoring beendet: {report}")
        return report

    def _monitor_loop(self):
        """Monitoring-Schleife für System-Metriken."""
        while self.monitoring:
            try:
                # CPU-Auslastung messen
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_percentages.append(cpu_percent)

                # Speichernutzung messen
                memory = psutil.virtual_memory()
                memory_mb = memory.used / 1024 / 1024
                self.memory_usages.append(memory_mb)

                time.sleep(0.5)  # Alle 0.5 Sekunden messen

            except Exception as e:
                logger.warning(f"Fehler beim Performance-Monitoring: {e}")
                break


class PerformanceOptimizer:
    """Hauptklasse für Performance-Optimierungen."""

    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.max_workers = MAX_WORKERS or multiprocessing.cpu_count()
        self.chunk_size = CHUNK_SIZE

        logger.info(f"PerformanceOptimizer initialisiert: max_workers={self.max_workers}, chunk_size={self.chunk_size}")

    def start_monitoring(self):
        """Starte Performance-Monitoring."""
        self.monitor.start_monitoring()

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stoppe Performance-Monitoring."""
        return self.monitor.stop_monitoring()

    def calculate_indicators_parallel(self, df: pd.DataFrame, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Berechne Indikatoren parallel mit mehreren Threads.

        Args:
            df: DataFrame mit OHLCV-Daten
            chunk_size: Optionale Chunk-Größe für Parallelisierung

        Returns:
            DataFrame mit berechneten Indikatoren
        """
        if not PARALLEL_PROCESSING_ENABLED or not PERFORMANCE_ENABLED:
            # Fallback zur sequentiellen Berechnung
            from src.indicators import calculate_indicators
            return calculate_indicators(df)

        chunk_size = chunk_size or self.chunk_size

        if len(df) < chunk_size * 2:
            # Zu kleine Datenmenge für Parallelisierung
            from src.indicators import calculate_indicators
            return calculate_indicators(df)

        try:
            # DataFrame in Chunks aufteilen
            chunks = self._split_dataframe(df, chunk_size)

            # Parallele Verarbeitung mit ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._calculate_indicators_chunk, chunk)
                    for chunk in chunks
                ]

                # Ergebnisse sammeln
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Fehler bei paralleler Indikator-Berechnung: {e}")
                        # Fallback zur sequentiellen Berechnung für diesen Chunk
                        from src.indicators import calculate_indicators
                        results.append(calculate_indicators(chunk))

            # Ergebnisse zusammenfügen
            if results:
                combined_df = pd.concat(results, ignore_index=True)
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                return combined_df
            else:
                # Fallback
                from src.indicators import calculate_indicators
                return calculate_indicators(df)

        except Exception as e:
            logger.warning(f"Parallele Verarbeitung fehlgeschlagen, verwende sequentielle Berechnung: {e}")
            from src.indicators import calculate_indicators
            return calculate_indicators(df)

    def calculate_indicators_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechne Indikatoren mit vektorisierten Operationen.

        Args:
            df: DataFrame mit OHLCV-Daten

        Returns:
            DataFrame mit berechneten Indikatoren
        """
        if not VECTORIZATION_ENABLED or not PERFORMANCE_ENABLED:
            from src.indicators import calculate_indicators
            return calculate_indicators(df)

        try:
            # Kopie erstellen
            result_df = df.copy()

            # Vektorisierte EMA-Berechnung
            result_df = self._calculate_ema_vectorized(result_df)

            # Vektorisierte RSI-Berechnung
            result_df = self._calculate_rsi_vectorized(result_df)

            # Vektorisierte Bollinger Bands
            result_df = self._calculate_bb_vectorized(result_df)

            # Vektorisierte ATR-Berechnung
            result_df = self._calculate_atr_vectorized(result_df)

            return result_df

        except Exception as e:
            logger.warning(f"Vektorisierte Berechnung fehlgeschlagen, verwende sequentielle Berechnung: {e}")
            from src.indicators import calculate_indicators
            return calculate_indicators(df)

    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimiere DataFrame-Speichernutzung durch Datentyp-Reduzierung.

        Args:
            df: DataFrame zu optimieren

        Returns:
            Optimierter DataFrame
        """
        if not MEMORY_OPTIMIZATION_ENABLED or not PERFORMANCE_ENABLED:
            return df

        try:
            optimized_df = df.copy()

            # Numerische Spalten optimieren
            for col in optimized_df.select_dtypes(include=[np.number]).columns:
                col_min = optimized_df[col].min()
                col_max = optimized_df[col].max()

                # Integer-Typen optimieren
                if optimized_df[col].dtype == 'int64':
                    if col_min >= 0:
                        if col_max < 2**8:
                            optimized_df[col] = optimized_df[col].astype('uint8')
                        elif col_max < 2**16:
                            optimized_df[col] = optimized_df[col].astype('uint16')
                        elif col_max < 2**32:
                            optimized_df[col] = optimized_df[col].astype('uint32')
                    else:
                        if col_min > -2**7 and col_max < 2**7:
                            optimized_df[col] = optimized_df[col].astype('int8')
                        elif col_min > -2**15 and col_max < 2**15:
                            optimized_df[col] = optimized_df[col].astype('int16')
                        elif col_min > -2**31 and col_max < 2**31:
                            optimized_df[col] = optimized_df[col].astype('int32')

                # Float-Typen optimieren
                elif optimized_df[col].dtype == 'float64':
                    # Prüfe, ob float32 ausreicht
                    if (optimized_df[col] % 1 == 0).all():
                        # Integer-Werte als float speichern
                        if col_min >= 0 and col_max < 2**32:
                            optimized_df[col] = optimized_df[col].astype('uint32')
                        else:
                            optimized_df[col] = optimized_df[col].astype('int32')
                    else:
                        # Float-Werte
                        optimized_df[col] = optimized_df[col].astype('float32')

            # Kategorische Spalten optimieren (falls vorhanden)
            for col in optimized_df.select_dtypes(include=['object']).columns:
                if optimized_df[col].nunique() / len(optimized_df) < 0.5:  # Weniger als 50% unique Werte
                    optimized_df[col] = optimized_df[col].astype('category')

            logger.info(f"DataFrame-Speicher optimiert: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB -> {optimized_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            return optimized_df

        except Exception as e:
            logger.warning(f"Speicheroptimierung fehlgeschlagen: {e}")
            return df

    def generate_signals_parallel(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generiere Signale parallel.

        Args:
            df: DataFrame mit Indikatoren

        Returns:
            Liste der generierten Signale
        """
        if not PARALLEL_PROCESSING_ENABLED or not PERFORMANCE_ENABLED:
            from src.signal_generator import generate_signals
            return generate_signals(df)

        try:
            # DataFrame in Chunks aufteilen
            chunks = self._split_dataframe(df, self.chunk_size)

            # Parallele Signal-Generierung
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._generate_signals_chunk, chunk)
                    for chunk in chunks
                ]

                # Ergebnisse sammeln
                all_signals = []
                for future in as_completed(futures):
                    try:
                        signals = future.result()
                        all_signals.extend(signals)
                    except Exception as e:
                        logger.error(f"Fehler bei paralleler Signal-Generierung: {e}")

            # Signale sortieren nach Timestamp
            all_signals.sort(key=lambda x: x.get('timestamp', ''))
            return all_signals

        except Exception as e:
            logger.warning(f"Parallele Signal-Generierung fehlgeschlagen, verwende sequentielle Generierung: {e}")
            from src.signal_generator import generate_signals
            return generate_signals(df)

    def _split_dataframe(self, df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
        """Teile DataFrame in Chunks auf."""
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            chunks.append(chunk)
        return chunks

    def _calculate_indicators_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Berechne Indikatoren für einen Chunk."""
        from src.indicators import calculate_indicators
        return calculate_indicators(chunk)

    def _generate_signals_chunk(self, chunk: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generiere Signale für einen Chunk."""
        from src.signal_generator import generate_signals
        return generate_signals(chunk)

    def _calculate_ema_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vektorisierte EMA-Berechnung."""
        from config.settings import EMA_SHORT, EMA_LONG

        # EMA_SHORT berechnen
        df['ema_short'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()

        # EMA_LONG berechnen
        df['ema_long'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()

        return df

    def _calculate_rsi_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vektorisierte RSI-Berechnung."""
        from config.settings import RSI_PERIOD

        # Preisänderungen berechnen
        delta = df['close'].diff()

        # Gewinne und Verluste
        gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()

        # RS und RSI berechnen
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def _calculate_bb_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vektorisierte Bollinger Bands Berechnung."""
        from config.settings import BB_PERIOD, BB_STD_DEV

        # SMA berechnen
        df['bb_middle'] = df['close'].rolling(window=BB_PERIOD).mean()

        # Standardabweichung berechnen
        df['bb_std'] = df['close'].rolling(window=BB_PERIOD).std()

        # Bollinger Bands berechnen
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * BB_STD_DEV)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * BB_STD_DEV)

        return df

    def _calculate_atr_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vektorisierte ATR-Berechnung."""
        from config.settings import ATR_PERIOD

        # True Range berechnen
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
        df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # ATR berechnen
        df['atr'] = df['true_range'].rolling(window=ATR_PERIOD).mean()

        # Temporäre Spalten entfernen
        df = df.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1)

        return df
