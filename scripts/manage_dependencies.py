#!/usr/bin/env python3
"""
TradPal Dependency Management Script

Validates that service requirements.txt files use approved versions
from the dependency catalog.

Usage:
    python scripts/manage_dependencies.py validate    # Check all services
    python scripts/manage_dependencies.py update      # Update service requirements
    python scripts/manage_dependencies.py list        # List all dependencies
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set
import re

class DependencyManager:
    def __init__(self, catalog_path: str = "dependency_catalog.txt"):
        self.catalog_path = Path(catalog_path)
        self.catalog = self._load_catalog()

    def _load_catalog(self) -> Dict[str, str]:
        """Load the dependency catalog."""
        catalog = {}
        if not self.catalog_path.exists():
            print(f"❌ Catalog file not found: {self.catalog_path}")
            sys.exit(1)

        with open(self.catalog_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '==' in line:
                        package, version = line.split('==', 1)
                        catalog[package.strip()] = version.strip()
                    elif '>=' in line:
                        package, version = line.split('>=', 1)
                        catalog[package.strip()] = version.strip()

        return catalog

    def find_service_requirements(self) -> List[Path]:
        """Find all service requirements.txt files."""
        services_dir = Path("services")
        if not services_dir.exists():
            return []

        requirements_files = []
        for service_dir in services_dir.iterdir():
            if service_dir.is_dir():
                req_file = service_dir / "requirements.txt"
                if req_file.exists():
                    requirements_files.append(req_file)

        return requirements_files

    def validate_service_requirements(self, req_file: Path) -> List[str]:
        """Validate a single service requirements.txt file."""
        issues = []

        with open(req_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    if '==' in line:
                        package, version = line.split('==', 1)
                        package = package.strip()
                        version = version.strip()

                        if package not in self.catalog:
                            issues.append(f"Line {line_num}: {package} not in catalog")
                        elif self.catalog[package] != version:
                            catalog_version = self.catalog[package]
                            issues.append(f"Line {line_num}: {package}=={version} != catalog version {catalog_version}")
                    else:
                        issues.append(f"Line {line_num}: {line} - use exact versions (==) not ranges")

        return issues

    def validate_all_services(self) -> bool:
        """Validate all service requirements.txt files."""
        print("🔍 Validating service dependencies...")

        req_files = self.find_service_requirements()
        if not req_files:
            print("❌ No service requirements.txt files found")
            return False

        all_valid = True
        for req_file in req_files:
            service_name = req_file.parent.name
            print(f"\n📦 Checking {service_name}...")

            issues = self.validate_service_requirements(req_file)
            if issues:
                all_valid = False
                print(f"❌ Issues found in {req_file}:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print(f"✅ {service_name} requirements are valid")

        return all_valid

    def list_dependencies(self):
        """List all dependencies in the catalog."""
        print("📋 Dependency Catalog:")
        print("=" * 50)

        for package, version in sorted(self.catalog.items()):
            print(f"{package}=={version}")

    def update_service_requirement(self, req_file: Path, package: str, new_version: str):
        """Update a package version in a requirements.txt file."""
        content = req_file.read_text()

        # Replace the version using string replacement
        old_line_pattern = f"{package}=="
        lines = content.split('\n')
        updated = False

        for i, line in enumerate(lines):
            if line.strip().startswith(old_line_pattern):
                lines[i] = f"{package}=={new_version}"
                updated = True
                break

        if updated:
            new_content = '\n'.join(lines)
            req_file.write_text(new_content)
            print(f"✅ Updated {package} to {new_version} in {req_file}")
        else:
            print(f"❌ Could not find {package} in {req_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/manage_dependencies.py <command>")
        print("Commands: validate, list, update")
        sys.exit(1)

    command = sys.argv[1]
    manager = DependencyManager()

    if command == "validate":
        success = manager.validate_all_services()
        sys.exit(0 if success else 1)

    elif command == "list":
        manager.list_dependencies()

    elif command == "update":
        if len(sys.argv) < 4:
            print("Usage: python scripts/manage_dependencies.py update <package> <version>")
            sys.exit(1)

        package = sys.argv[2]
        version = sys.argv[3]

        # Update catalog
        manager.catalog[package] = version
        with open(manager.catalog_path, 'a') as f:
            f.write(f"{package}=={version}\n")

        print(f"✅ Updated catalog: {package}=={version}")

        # Update all service requirements that use this package
        req_files = manager.find_service_requirements()
        for req_file in req_files:
            manager.update_service_requirement(req_file, package, version)

    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()