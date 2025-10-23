#!/usr/bin/env python3
"""
Test-Skript für den Alert Forwarder Service.

Dieses Skript testet die Kernfunktionalitäten des Alert Forwarders:
- Alert-Parsing aus verschiedenen Formaten
- Notification-Weiterleitung
- Rate-Limiting und Batching
- Fehlerbehandlung
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from config import AlertConfig, FalcoAlert
from forwarder import AlertForwarder


class TestAlertConfig:
    """Test-Klasse für Alert-Konfiguration."""

    def test_default_config(self):
        """Test Standard-Konfiguration."""
        config = AlertConfig()

        assert config.falco_log_path == "/var/log/falco/falco.log"
        assert config.alert_min_priority == "WARNING"
        assert config.alert_batching is True
        assert config.alert_batch_interval == 300
        assert config.alert_rate_limit == 10
        assert config.telegram_enabled is True
        assert config.discord_enabled is True

    def test_config_from_env(self):
        """Test Konfiguration aus Umgebungsvariablen."""
        env_vars = {
            "FALCO_LOG_PATH": "/custom/log/falco.log",
            "ALERT_MIN_PRIORITY": "ERROR",
            "ALERT_BATCHING": "false",
            "ALERT_RATE_LIMIT": "5",
            "TELEGRAM_ENABLED": "false",
            "DISCORD_ENABLED": "false"
        }

        with patch.dict(os.environ, env_vars):
            config = AlertConfig.from_env()

            assert config.falco_log_path == "/custom/log/falco.log"
            assert config.alert_min_priority == "ERROR"
            assert config.alert_batching is False
            assert config.alert_rate_limit == 5
            assert config.telegram_enabled is False
            assert config.discord_enabled is False

    def test_invalid_priority(self):
        """Test ungültige Priorität."""
        with pytest.raises(ValidationError):
            AlertConfig(alert_min_priority="INVALID")


class TestFalcoAlert:
    """Test-Klasse für Falco-Alert-Parsing."""

    def test_parse_json_alert(self):
        """Test JSON-Alert-Parsing."""
        json_alert = {
            "output": "File below /etc is being accessed for writing (user=root command=vi /etc/hosts file=/etc/hosts)",
            "priority": "WARNING",
            "rule": "Write below etc",
            "time": "2025-10-15T10:30:00.000000000Z",
            "output_fields": {
                "proc.cmdline": "vi /etc/hosts",
                "proc.pid": "12345",
                "fd.name": "/etc/hosts",
                "user.name": "root"
            },
            "hostname": "tradpal-node-01",
            "tags": ["filesystem", "mitre_persistence"]
        }

        alert = FalcoAlert.from_log_line(json.dumps(json_alert))

        assert alert.rule == "Write below etc"
        assert alert.priority == "WARNING"
        assert alert.time.isoformat() == "2025-10-15T10:30:00"
        assert alert.hostname == "tradpal-node-01"
        assert "filesystem" in alert.tags

    def test_parse_text_alert(self):
        """Test Plain-Text-Alert-Parsing."""
        text_alert = "2025-10-15T10:30:00.000000000Z: Warning File below /etc is being accessed (user=root proc=vi file=/etc/hosts)"

        alert = FalcoAlert.from_log_line(text_alert)

        assert alert.rule == "File below /etc is being accessed"
        assert alert.priority == "WARNING"
        assert alert.time.isoformat() == "2025-10-15T10:30:00"
        assert alert.hostname is None

    def test_invalid_json_alert(self):
        """Test ungültiges JSON-Alert."""
        invalid_json = '{"invalid": json}'

        with pytest.raises(json.JSONDecodeError):
            FalcoAlert.from_log_line(invalid_json)

    def test_alert_formatting_telegram(self):
        """Test Telegram-Formatierung."""
        alert = FalcoAlert(
            rule="Unauthorized API Key Access",
            priority="WARNING",
            time="2025-10-15T10:30:00.000000000Z",
            output="Unauthorized access to sensitive file",
            hostname="tradpal-node-01",
            tags=["filesystem", "trading", "security"]
        )

        formatted = alert.format_for_telegram()

        assert "🚨 FALCO SECURITY ALERT 🚨" in formatted
        assert "Unauthorized API Key Access" in formatted
        assert "WARNING" in formatted
        assert "tradpal-node-01" in formatted

    def test_alert_formatting_discord(self):
        """Test Discord-Formatierung."""
        alert = FalcoAlert(
            rule="Suspicious Network Connection",
            priority="CRITICAL",
            time="2025-10-15T10:30:00.000000000Z",
            output="Suspicious outbound connection",
            hostname="tradpal-node-01",
            tags=["network", "security"]
        )

        embed = alert.format_for_discord()

        assert embed["title"] == "🚨 Falco Security Alert"
        assert "CRITICAL" in embed["description"]
        assert embed["color"] == 0xFF0000  # Rot für CRITICAL


class TestAlertForwarder:
    """Test-Klasse für Alert Forwarder."""

    @pytest.fixture
    async def forwarder(self):
        """Fixture für Alert Forwarder."""
        config = AlertConfig(
            falco_log_path="/tmp/test_falco.log",
            alert_rate_limit=5,
            alert_batch_interval=1  # Schnell für Tests
        )

        forwarder = AlertForwarder(config)

        # Mock Notification-Services
        forwarder.telegram_client = AsyncMock()
        forwarder.discord_client = AsyncMock()

        yield forwarder

        # Cleanup
        if os.path.exists(config.falco_log_path):
            os.remove(config.falco_log_path)

    @pytest.mark.asyncio
    async def test_rate_limiting(self, forwarder):
        """Test Rate-Limiting."""
        # Erstelle mehrere Alerts
        alerts = []
        for i in range(10):
            alert = FalcoAlert(
                rule=f"Test Alert {i}",
                priority="WARNING",
                time=f"2025-10-15T10:{30+i:02d}:00.000000000Z",
                output=f"Test output {i}"
            )
            alerts.append(alert)

        # Sende Alerts
        for alert in alerts:
            await forwarder._handle_alert(alert)

        # Prüfe Rate-Limiting
        assert forwarder.rate_limiter.is_rate_limited() is True

        # Warte auf Cooldown
        await asyncio.sleep(forwarder.config.alert_cooldown + 1)
        assert forwarder.rate_limiter.is_rate_limited() is False

    @pytest.mark.asyncio
    async def test_alert_batching(self, forwarder):
        """Test Alert-Batching."""
        forwarder.config.alert_batching = True
        forwarder.config.alert_max_batch_size = 3

        # Erstelle Test-Alerts
        alerts = []
        for i in range(5):
            alert = FalcoAlert(
                rule=f"Batch Alert {i}",
                priority="WARNING",
                time=f"2025-10-15T10:{30+i:02d}:00.000000000Z",
                output=f"Batch output {i}"
            )
            alerts.append(alert)

        # Sende Alerts
        for alert in alerts:
            await forwarder._handle_alert(alert)

        # Warte auf Batch-Verarbeitung
        await asyncio.sleep(forwarder.config.alert_batch_interval + 1)

        # Prüfe Batch-Sending
        assert forwarder.telegram_client.send_message.called
        assert forwarder.discord_client.send_embed.called

    @pytest.mark.asyncio
    async def test_log_monitoring(self, forwarder):
        """Test Log-Monitoring."""
        # Erstelle Test-Log-Datei
        test_alert = {
            "output": "Test security alert",
            "priority": "WARNING",
            "rule": "Test Rule",
            "time": "2025-10-15T10:30:00.000000000Z"
        }

        with open(forwarder.config.falco_log_path, 'w') as f:
            f.write(json.dumps(test_alert) + '\n')

        # Starte Monitoring für kurze Zeit
        monitoring_task = asyncio.create_task(forwarder._monitoring_loop())
        await asyncio.sleep(0.1)  # Kurz warten
        monitoring_task.cancel()

        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

        # Prüfe Alert-Verarbeitung
        assert forwarder.telegram_client.send_message.called or forwarder.discord_client.send_embed.called

    @pytest.mark.asyncio
    async def test_error_handling(self, forwarder):
        """Test Fehlerbehandlung."""
        # Mock fehlerhafte Notification
        forwarder.telegram_client.send_message.side_effect = Exception("Network error")

        alert = FalcoAlert(
            rule="Error Test",
            priority="WARNING",
            time="2025-10-15T10:30:00.000000000Z",
            output="Test error handling"
        )

        # Sollte nicht crashen
        await forwarder._handle_alert(alert)

        # Fehler sollte geloggt werden (prüfen wir durch Mock-Aufruf)
        assert forwarder.telegram_client.send_message.called

    @pytest.mark.asyncio
    async def test_priority_filtering(self, forwarder):
        """Test Prioritäts-Filterung."""
        # Test mit niedriger Priorität
        low_priority_alert = FalcoAlert(
            rule="Low Priority",
            priority="INFO",
            time="2025-10-15T10:30:00.000000000Z",
            output="Low priority alert"
        )

        # Test mit hoher Priorität
        high_priority_alert = FalcoAlert(
            rule="High Priority",
            priority="CRITICAL",
            time="2025-10-15T10:30:00.000000000Z",
            output="High priority alert"
        )

        # Niedrige Priorität sollte gefiltert werden
        await forwarder._handle_alert(low_priority_alert)
        assert not forwarder.telegram_client.send_message.called

        # Hohe Priorität sollte durchgelassen werden
        await forwarder._handle_alert(high_priority_alert)
        assert forwarder.telegram_client.send_message.called


class TestIntegration:
    """Integrationstests."""

    @pytest.mark.asyncio
    async def test_end_to_end(self):
        """End-to-End Test."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as log_file:
            log_path = log_file.name

            # Schreibe Test-Alert
            test_alert = {
                "output": "Integration test alert",
                "priority": "WARNING",
                "rule": "Integration Test",
                "time": "2025-10-15T10:30:00.000000000Z",
                "hostname": "test-host",
                "tags": ["test", "integration"]
            }
            log_file.write(json.dumps(test_alert) + '\n')
            log_file.flush()

            # Erstelle Forwarder
            config = AlertConfig(
                falco_log_path=log_path,
                alert_batching=False,  # Sofort senden für Test
                alert_rate_limit=100   # Hohes Limit für Test
            )

            forwarder = AlertForwarder(config)
            forwarder.telegram_client = AsyncMock()
            forwarder.discord_client = AsyncMock()

            try:
                # Starte Monitoring
                monitoring_task = asyncio.create_task(forwarder._monitoring_loop())
                await asyncio.sleep(0.2)  # Warte auf Verarbeitung
                monitoring_task.cancel()

                try:
                    await monitoring_task
                except asyncio.CancelledError:
                    pass

                # Prüfe Ergebnis
                assert forwarder.telegram_client.send_message.called
                assert forwarder.discord_client.send_embed.called

                # Prüfe Alert-Inhalt
                telegram_call = forwarder.telegram_client.send_message.call_args[0][0]
                assert "Integration Test" in telegram_call
                assert "WARNING" in telegram_call

            finally:
                # Cleanup
                os.unlink(log_path)


if __name__ == "__main__":
    # Einfache Test-Ausführung ohne pytest
    print("🚀 Starte Alert Forwarder Tests...")

    # Test AlertConfig
    print("📋 Teste AlertConfig...")
    config = AlertConfig()
    assert config.falco_log_path == "/var/log/falco/falco.log"
    print("✅ AlertConfig Tests bestanden")

    # Test FalcoAlert
    print("📋 Teste FalcoAlert...")
    json_alert = '{"output": "Test alert", "priority": "WARNING", "rule": "Test Rule", "time": "2025-10-15T10:30:00.000000000Z"}'
    alert = FalcoAlert.from_log_line(json_alert)
    assert alert.rule == "Test Rule"
    assert alert.priority == "WARNING"
    print("✅ FalcoAlert Tests bestanden")

    # Test Formatierung
    telegram_msg = alert.format_for_telegram()
    assert "Test Rule" in telegram_msg
    discord_embed = alert.format_for_discord()
    assert "Falco Security Alert" in discord_embed["title"]
    print("✅ Formatierung Tests bestanden")

    print("🎉 Alle Tests erfolgreich abgeschlossen!")