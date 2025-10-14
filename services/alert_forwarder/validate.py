#!/usr/bin/env python3
"""
Validierungsskript für den Alert Forwarder Service.

Dieses Skript führt eine umfassende Validierung der Alert Forwarder Implementierung durch:
- Code-Syntax und Imports
- Konfiguration und Pydantic-Modelle
- Alert-Parsing und Formatierung
- Integration mit Notification-Services
- Kubernetes-Manifeste
- Docker-Setup
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
except ImportError:
    yaml = None

class ValidationResult:
    """Ergebnis einer Validierung."""

    def __init__(self, name: str, success: bool, message: str = "", details: List[str] = None):
        self.name = name
        self.success = success
        self.message = message
        self.details = details or []

    def __str__(self):
        status = "✅" if self.success else "❌"
        result = f"{status} {self.name}"
        if self.message:
            result += f": {self.message}"
        if self.details:
            result += "\n" + "\n".join(f"   - {detail}" for detail in self.details)
        return result


class AlertForwarderValidator:
    """Validator für den Alert Forwarder Service."""

    def __init__(self, service_path: Path):
        self.service_path = service_path
        self.results: List[ValidationResult] = []

    def validate_all(self) -> bool:
        """Führt alle Validierungen durch."""
        print("🚀 Starte Validierung des Alert Forwarder Services...\n")

        # Basis-Validierungen
        self._validate_file_structure()
        self._validate_python_syntax()
        self._validate_imports()
        self._validate_config_models()
        self._validate_alert_parsing()
        self._validate_notification_formatting()

        # Kubernetes-Validierungen
        self._validate_kubernetes_manifests()

        # Docker-Validierungen
        self._validate_docker_setup()

        # Integration-Tests
        self._validate_integration_tests()

        # Bericht generieren
        return self._generate_report()

    def _validate_file_structure(self):
        """Validiert die Dateistruktur."""
        required_files = [
            "config.py", "forwarder.py", "cli.py", "tests.py",
            "requirements.txt", "README.md", "Makefile",
            "Dockerfile", "docker-compose.yml",
            "k8s/deployment.yaml", "k8s/configmap.yaml"
        ]

        missing_files = []
        for file_path in required_files:
            if not (self.service_path / file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            self.results.append(ValidationResult(
                "Dateistruktur",
                False,
                f"Fehlende Dateien: {', '.join(missing_files)}"
            ))
        else:
            self.results.append(ValidationResult(
                "Dateistruktur",
                True,
                "Alle erforderlichen Dateien vorhanden"
            ))

    def _validate_python_syntax(self):
        """Validiert Python-Syntax."""
        python_files = ["config.py", "forwarder.py", "cli.py", "tests.py"]
        syntax_errors = []

        for file_name in python_files:
            file_path = self.service_path / file_name
            if file_path.exists():
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(file_path)],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    syntax_errors.append(f"{file_name}: {result.stderr.strip()}")

        if syntax_errors:
            self.results.append(ValidationResult(
                "Python-Syntax",
                False,
                "Syntaxfehler gefunden",
                syntax_errors
            ))
        else:
            self.results.append(ValidationResult(
                "Python-Syntax",
                True,
                "Alle Python-Dateien syntaktisch korrekt"
            ))

    def _validate_imports(self):
        """Validiert Python-Imports."""
        try:
            # Temporär zum Pfad hinzufügen
            sys.path.insert(0, str(self.service_path))

            # Test-Imports
            import config
            import forwarder
            import cli

            # Basis-Objekte prüfen
            assert hasattr(config, 'AlertConfig')
            assert hasattr(config, 'FalcoAlert')
            assert hasattr(forwarder, 'AlertForwarder')
            assert hasattr(cli, 'main')

            self.results.append(ValidationResult(
                "Python-Imports",
                True,
                "Alle Module erfolgreich importiert"
            ))

        except Exception as e:
            self.results.append(ValidationResult(
                "Python-Imports",
                False,
                f"Import-Fehler: {str(e)}"
            ))
        finally:
            # Pfad wieder entfernen
            if str(self.service_path) in sys.path:
                sys.path.remove(str(self.service_path))

    def _validate_config_models(self):
        """Validiert Pydantic-Konfigurationsmodelle."""
        try:
            sys.path.insert(0, str(self.service_path))
            from config import AlertConfig, FalcoAlert

            # Test Standard-Konfiguration
            config = AlertConfig()
            assert config.falco_log_path == "/var/log/falco/falco.log"
            assert config.alert_min_priority == "WARNING"

            # Test Konfiguration aus Env
            os.environ['ALERT_MIN_PRIORITY'] = 'ERROR'
            config_env = AlertConfig.from_env()
            assert config_env.alert_min_priority == 'ERROR'

            # Test Alert-Modell
            test_alert_data = {
                "timestamp": "2025-10-15T10:30:00.000000000Z",
                "rule": "Test Rule",
                "priority": "WARNING",
                "message": "Test message",
                "output": "Test output"
            }
            alert = FalcoAlert(**test_alert_data)
            assert alert.rule == "Test Rule"

            self.results.append(ValidationResult(
                "Konfigurationsmodelle",
                True,
                "Pydantic-Modelle funktionieren korrekt"
            ))

        except Exception as e:
            self.results.append(ValidationResult(
                "Konfigurationsmodelle",
                False,
                f"Modell-Validierung fehlgeschlagen: {str(e)}"
            ))
        finally:
            sys.path.remove(str(self.service_path))
            os.environ.pop('ALERT_MIN_PRIORITY', None)

    def _validate_alert_parsing(self):
        """Validiert Alert-Parsing."""
        try:
            sys.path.insert(0, str(self.service_path))
            from config import FalcoAlert

            # Test JSON-Parsing
            json_alert = '{"output": "Test alert", "priority": "WARNING", "rule": "Test Rule", "time": "2025-10-15T10:30:00.000000000Z"}'
            alert = FalcoAlert.from_log_line(json_alert)
            assert alert is not None, "Alert parsing returned None"
            assert alert.rule == "Test Rule"

            # Test Text-Parsing
            text_alert = "2025-10-15T10:30:00.000000000Z rule=Test Rule prio=WARNING Test text alert"
            alert_text = FalcoAlert.from_log_line(text_alert)
            assert alert_text is not None, "Text alert parsing returned None"
            assert alert_text.priority == "WARNING"

            self.results.append(ValidationResult(
                "Alert-Parsing",
                True,
                "JSON und Text-Alert-Parsing funktioniert"
            ))

        except Exception as e:
            self.results.append(ValidationResult(
                "Alert-Parsing",
                False,
                f"Alert-Parsing fehlgeschlagen: {str(e)}"
            ))
        finally:
            sys.path.remove(str(self.service_path))

    def _validate_notification_formatting(self):
        """Validiert Notification-Formatierung."""
        try:
            sys.path.insert(0, str(self.service_path))
            from config import FalcoAlert

            alert = FalcoAlert(
                timestamp="2025-10-15T10:30:00.000000000Z",
                rule="Test Security Alert",
                priority="WARNING",
                message="Test alert message",
                output="Test alert output",
                hostname="test-host",
                tags=["test", "security"]
            )

            # Test Telegram-Formatierung
            telegram_msg = alert.format_for_telegram()
            assert "Test Security Alert" in telegram_msg
            assert "WARNING" in telegram_msg

            # Test Discord-Formatierung
            discord_embed = alert.format_for_discord()
            assert "Falco Security Alert" in discord_embed["title"]
            assert "WARNING" in discord_embed["description"]

            self.results.append(ValidationResult(
                "Notification-Formatierung",
                True,
                "Telegram und Discord-Formatierung funktioniert"
            ))

        except Exception as e:
            self.results.append(ValidationResult(
                "Notification-Formatierung",
                False,
                f"Formatierung fehlgeschlagen: {str(e)}"
            ))
        finally:
            sys.path.remove(str(self.service_path))

    def _validate_kubernetes_manifests(self):
        """Validiert Kubernetes-Manifeste."""
        if yaml is None:
            self.results.append(ValidationResult(
                "Kubernetes-Manifeste",
                False,
                "PyYAML nicht installiert - YAML-Validierung übersprungen"
            ))
            return

        manifests = ["k8s/deployment.yaml", "k8s/configmap.yaml"]
        manifest_errors = []

        for manifest_path in manifests:
            full_path = self.service_path / manifest_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        # YAML kann mehrere Dokumente enthalten
                        documents = list(yaml.safe_load_all(f))

                    for i, data in enumerate(documents):
                        if data is None:
                            continue

                        # Basis-Struktur prüfen
                        if manifest_path.endswith("deployment.yaml"):
                            assert "apiVersion" in data
                            assert "kind" in data
                            if data["kind"] == "Deployment":
                                assert "spec" in data
                            elif data["kind"] == "Service":
                                assert "spec" in data
                        elif manifest_path.endswith("configmap.yaml"):
                            assert "apiVersion" in data
                            assert "kind" in data
                            assert data["kind"] in ["ConfigMap", "Secret"]

                except Exception as e:
                    manifest_errors.append(f"{manifest_path}: {str(e)}")
            else:
                manifest_errors.append(f"{manifest_path}: Datei nicht gefunden")

        if manifest_errors:
            self.results.append(ValidationResult(
                "Kubernetes-Manifeste",
                False,
                "Manifest-Validierung fehlgeschlagen",
                manifest_errors
            ))
        else:
            self.results.append(ValidationResult(
                "Kubernetes-Manifeste",
                True,
                "Alle Manifeste syntaktisch korrekt"
            ))

    def _validate_docker_setup(self):
        """Validiert Docker-Setup."""
        dockerfile_path = self.service_path / "Dockerfile"
        compose_path = self.service_path / "docker-compose.yml"

        docker_errors = []

        # Dockerfile prüfen
        if dockerfile_path.exists():
            with open(dockerfile_path, 'r') as f:
                content = f.read()
                if "FROM python:" not in content:
                    docker_errors.append("Dockerfile: Kein Python-Base-Image")
                if "COPY requirements.txt" not in content:
                    docker_errors.append("Dockerfile: requirements.txt nicht kopiert")
        else:
            docker_errors.append("Dockerfile: Datei nicht gefunden")

        # Docker-Compose prüfen
        if compose_path.exists():
            try:
                with open(compose_path, 'r') as f:
                    if yaml:
                        compose_data = yaml.safe_load(f)
                    else:
                        # Fallback: versuche JSON
                        compose_data = json.load(f)

                if "services" not in compose_data:
                    docker_errors.append("docker-compose.yml: Keine Services definiert")
            except:
                docker_errors.append("docker-compose.yml: Ungültiges Format")
        else:
            docker_errors.append("docker-compose.yml: Datei nicht gefunden")

        if docker_errors:
            self.results.append(ValidationResult(
                "Docker-Setup",
                False,
                "Docker-Validierung fehlgeschlagen",
                docker_errors
            ))
        else:
            self.results.append(ValidationResult(
                "Docker-Setup",
                True,
                "Dockerfile und docker-compose.yml korrekt"
            ))

    def _validate_integration_tests(self):
        """Führt Integrationstests aus."""
        try:
            # Test-Skript ausführen
            test_script = self.service_path / "tests.py"
            if test_script.exists():
                result = subprocess.run(
                    [sys.executable, str(test_script)],
                    capture_output=True, text=True, cwd=self.service_path
                )

                if result.returncode == 0:
                    self.results.append(ValidationResult(
                        "Integrationstests",
                        True,
                        "Alle Tests erfolgreich durchgeführt"
                    ))
                else:
                    self.results.append(ValidationResult(
                        "Integrationstests",
                        False,
                        "Tests fehlgeschlagen",
                        [result.stderr.strip()]
                    ))
            else:
                self.results.append(ValidationResult(
                    "Integrationstests",
                    False,
                    "Test-Skript nicht gefunden"
                ))

        except Exception as e:
            self.results.append(ValidationResult(
                "Integrationstests",
                False,
                f"Test-Ausführung fehlgeschlagen: {str(e)}"
            ))

    def _generate_report(self) -> bool:
        """Generiert Validierungsbericht."""
        print("\n" + "="*60)
        print("📊 ALERT FORWARDER VALIDIERUNGSBERICHT")
        print("="*60)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests

        print(f"\nGesamt Tests: {total_tests}")
        print(f"Erfolgreich: {passed_tests}")
        print(f"Fehlgeschlagen: {failed_tests}")

        print("\n" + "-"*60)
        print("DETAILLIERTE ERGEBNISSE:")
        print("-"*60)

        for result in self.results:
            print(f"\n{result}")

        print("\n" + "="*60)

        if failed_tests == 0:
            print("🎉 ALLE VALIDIERUNGEN ERFOLGREICH!")
            print("Der Alert Forwarder Service ist bereit für den Einsatz.")
            return True
        else:
            print(f"❌ {failed_tests} VALIDIERUNG(EN) FEHLGESCHLAGEN!")
            print("Bitte beheben Sie die oben aufgeführten Probleme.")
            return False


def main():
    """Hauptfunktion."""
    service_path = Path(__file__).parent

    validator = AlertForwarderValidator(service_path)
    success = validator.validate_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()