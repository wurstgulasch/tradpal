# VS Code Test Discovery Fix

## Problem
VS Code zeigt den Fehler "Pytest-Testermittlungsfehler" mit Exit Code 3 an.

## Ursache
VS Code verwendet das `vscode_pytest` Plugin, das eine Umgebungsvariable `TEST_RUN_PIPE` erwartet, die nicht gesetzt wird.

## Lösung

### Schritt 1: VS Code neu starten
1. Schließen Sie VS Code komplett
2. Öffnen Sie das Terminal und führen Sie aus:
   ```bash
   cd /Users/danielsadowski/VSCodeProjects/tradpal/tradpal
   code .
   ```

### Schritt 2: Python Extension aktualisieren (falls nötig)
1. Öffnen Sie VS Code
2. Gehen Sie zu Extensions (Ctrl+Shift+X)
3. Suchen Sie nach "Python"
4. Klicken Sie auf "Update" falls verfügbar

### Schritt 3: Test Discovery manuell auslösen
1. Öffnen Sie die Test-Ansicht in VS Code (Ctrl+Shift+T)
2. Klicken Sie auf "Refresh Tests" (die beiden Pfeile im Kreis)

### Schritt 4: Alternative - Terminal verwenden
Falls VS Code weiterhin Probleme hat, verwenden Sie das Terminal:
```bash
cd /Users/danielsadowski/VSCodeProjects/tradpal/tradpal
python -m pytest --collect-only
```

## Konfiguration
Die VS Code settings.json wurde konfiguriert für:
- Python Interpreter: `/opt/miniconda3/envs/tradpal_env/bin/python`
- Pytest aktiviert
- Korrekte Umgebungsvariablen

## Status
- ✅ Pytest funktioniert im Terminal
- ✅ 351 Tests werden erkannt
- ✅ 23 Import-Fehler verbleibend (normale Tests funktionieren)
- ❌ VS Code Test-Entdeckung hat Plugin-Kompatibilitätsproblem
