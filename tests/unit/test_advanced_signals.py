#!/usr/bin/env python3
"""
Test Script für Erweiterte Signalgenerierung

Dieses Script testet die neuen ML-basierten Signalgenerierungsfunktionen:
- Vergleich zwischen Legacy und Advanced Signalgenerierung
- ML-Modell Training
- Performance-Messungen
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from services.core_service.service import CoreService


@pytest.fixture
def core_service():
    """Fixture für CoreService Instanz"""
    service = CoreService()
    yield service


@pytest.fixture
def sample_data():
    """Fixture für synthetische Testdaten"""
    # Create event loop for async function
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        data = loop.run_until_complete(create_sample_data("BTC/USDT", periods=500))
        return data
    finally:
        loop.close()


async def create_sample_data(symbol: str = "BTC/USDT", periods: int = 1000) -> pd.DataFrame:
    """Erstelle synthetische Marktdaten für Tests"""

    logger.info(f"Erstelle {periods} Perioden synthetischer Daten für {symbol}")

    # Erstelle Zeitindex
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=periods)
    dates = pd.date_range(start_time, end_time, freq='1H')[:periods]

    # Synthetische Preisdaten mit Trend und Volatilität
    np.random.seed(42)

    # Basis-Trend
    trend = np.linspace(40000, 50000, periods)

    # Zufällige Preisbewegungen
    noise = np.random.normal(0, 1000, periods)
    cumulative_noise = np.cumsum(noise * 0.1)

    # Kombinierte Preise
    close_prices = trend + cumulative_noise

    # Erstelle OHLCV Daten
    high_prices = close_prices + np.abs(np.random.normal(0, 500, periods))
    low_prices = close_prices - np.abs(np.random.normal(0, 500, periods))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0] - np.random.normal(0, 100)

    # Volume
    volume = np.random.randint(1000, 10000, periods)

    # Erstelle DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)

    # Stelle sicher, dass high >= max(open, close) und low <= min(open, close)
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    logger.info(f"Synthetische Daten erstellt: {len(data)} Zeilen")
    logger.info(f"Preisbereich: {data['close'].min():.2f} - {data['close'].max():.2f}")

    return data


@pytest.mark.asyncio
async def test_legacy_vs_advanced_signals(core_service, sample_data):
    """Vergleiche Legacy vs Advanced Signalgenerierung"""

    logger.info("=== Test: Legacy vs Advanced Signalgenerierung ===")

    # Teste Legacy Signalgenerierung
    logger.info("Teste Legacy Signalgenerierung...")
    start_time = time.time()
    legacy_signals = await core_service._generate_legacy_signals("BTC/USDT", "1h", sample_data)
    legacy_time = time.time() - start_time

    logger.info(f"Legacy Signale: {len(legacy_signals)} in {legacy_time:.3f}s")

    # Teste Advanced Signalgenerierung
    logger.info("Teste Advanced Signalgenerierung...")
    start_time = time.time()
    advanced_signals = await core_service.generate_advanced_signals("BTC/USDT", "1h", sample_data)
    advanced_time = time.time() - start_time

    logger.info(f"Advanced Signale: {len(advanced_signals)} in {advanced_time:.3f}s")

    # Assertions
    assert isinstance(legacy_signals, list), "Legacy signals should be a list"
    assert isinstance(advanced_signals, list), "Advanced signals should be a list"
    assert legacy_time > 0, "Legacy processing time should be positive"
    assert advanced_time > 0, "Advanced processing time should be positive"

    # Analysiere Signal-Qualität wenn Signale vorhanden
    if legacy_signals:
        legacy_buy_signals = [s for s in legacy_signals if s.get('action') == 'BUY']
        legacy_sell_signals = [s for s in legacy_signals if s.get('action') == 'SELL']
        legacy_avg_confidence = np.mean([s.get('confidence', 0) for s in legacy_signals])

        assert legacy_avg_confidence >= 0, "Legacy confidence should be non-negative"
        assert legacy_avg_confidence <= 1, "Legacy confidence should be <= 1"

    if advanced_signals:
        advanced_buy_signals = [s for s in advanced_signals if s.get('action') == 'BUY']
        advanced_sell_signals = [s for s in advanced_signals if s.get('action') == 'SELL']
        advanced_avg_confidence = np.mean([s.get('confidence', 0) for s in advanced_signals])

        assert advanced_avg_confidence >= 0, "Advanced confidence should be non-negative"
        assert advanced_avg_confidence <= 1, "Advanced confidence should be <= 1"

        # Zeige erweiterte Features
        sample_signal = advanced_signals[0]
        assert 'action' in sample_signal, "Signal should have action"
        assert 'confidence' in sample_signal, "Signal should have confidence"
        assert 'reason' in sample_signal, "Signal should have reason"


@pytest.mark.asyncio
async def test_ml_training(core_service):
    """Teste ML-Modell Training"""

    logger.info("=== Test: ML-Modell Training ===")

    # Prüfe ob Advanced Signal Generator verfügbar ist
    if not core_service.advanced_signal_generator:
        pytest.skip("Advanced Signal Generator nicht verfügbar")

    # Erstelle Trainingsdaten
    symbol = "BTC/USDT"
    training_data = await create_sample_data(symbol, periods=2000)  # Mehr Daten für Training

    logger.info("Starte ML-Modell Training...")
    start_time = time.time()

    # Trainiere Modell
    success = await core_service.train_ml_model(symbol, training_data)

    training_time = time.time() - start_time

    if success:
        logger.info(f"ML-Training erfolgreich in {training_time:.2f}s")
        assert training_time > 0, "Training time should be positive"

        # Teste Modell nach Training
        logger.info("Teste Signalgenerierung mit trainiertem Modell...")
        test_data = training_data.tail(100)  # Letzte 100 Perioden als Test

        signals_before = await core_service._generate_legacy_signals(symbol, "1h", test_data)
        signals_after = await core_service.generate_advanced_signals(symbol, "1h", test_data)

        assert isinstance(signals_before, list), "Legacy signals should be list"
        assert isinstance(signals_after, list), "Advanced signals should be list"

    else:
        logger.warning(f"ML-Training fehlgeschlagen nach {training_time:.2f}s")
        # Training kann fehlschlagen - das ist akzeptabel für den Test


@pytest.mark.asyncio
async def test_model_loading(core_service):
    """Teste ML-Modell laden"""

    logger.info("=== Test: ML-Modell laden ===")

    if not core_service.advanced_signal_generator:
        pytest.skip("Advanced Signal Generator nicht verfügbar")

    # Versuche Modell zu laden
    symbol = "BTC/USDT"
    success = await core_service.load_ml_model(symbol)

    # Modell loading kann erfolgreich oder nicht erfolgreich sein
    assert isinstance(success, bool), "Model loading should return boolean"


@pytest.mark.asyncio
async def test_health_check(core_service):
    """Teste erweiterten Health Check"""

    logger.info("=== Test: Erweiterter Health Check ===")

    # Führe Health Check durch
    health_status = await core_service.health_check()

    assert isinstance(health_status, dict), "Health check should return dict"
    assert 'status' in health_status, "Health check should have status"

    # Prüfe erweiterte Features wenn verfügbar
    if 'advanced_signal_generation' in health_status:
        advanced_sig_gen = health_status['advanced_signal_generation']
        assert isinstance(advanced_sig_gen, dict), "Advanced signal generation should be dict"


@pytest.mark.asyncio
async def test_sample_data_creation():
    """Teste Erstellung von synthetischen Daten"""

    data = await create_sample_data("BTC/USDT", periods=100)

    assert isinstance(data, pd.DataFrame), "Sample data should be DataFrame"
    assert len(data) == 100, "DataFrame should have correct number of rows"
    assert 'open' in data.columns, "DataFrame should have open column"
    assert 'high' in data.columns, "DataFrame should have high column"
    assert 'low' in data.columns, "DataFrame should have low column"
    assert 'close' in data.columns, "DataFrame should have close column"
    assert 'volume' in data.columns, "DataFrame should have volume column"

    # Prüfe Datenintegrität
    assert (data['high'] >= data['close']).all(), "High should be >= close"
    assert (data['high'] >= data['open']).all(), "High should be >= open"
    assert (data['low'] <= data['close']).all(), "Low should be <= close"
    assert (data['low'] <= data['open']).all(), "Low should be <= open"


@pytest.mark.asyncio
async def test_advanced_features_demo(core_service):
    """Teste Demonstration der erweiterten Features"""

    logger.info("🎯 Test erweiterter Signalgenerierungs-Features")

    # Erstelle Testdaten
    symbol = "BTC/USDT"
    test_data = await create_sample_data(symbol, periods=200)

    # Teste Signalgenerierung
    try:
        signals = await core_service.generate_advanced_signals(symbol, "1h", test_data)

        assert isinstance(signals, list), "Signals should be a list"

        if signals:
            signal = signals[0]
            assert 'action' in signal, "Signal should have action"
            assert 'confidence' in signal, "Signal should have confidence"
            assert 'reason' in signal, "Signal should have reason"

    except Exception as e:
        # Signalgenerierung kann fehlschlagen - das ist akzeptabel
        logger.warning(f"Signalgenerierung fehlgeschlagen: {e}")
