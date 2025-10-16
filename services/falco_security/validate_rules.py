#!/usr/bin/env python3
"""
TradPal Falco Rules Test Script
Validates Falco rules syntax and logic
"""

import yaml
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

class FalcoRulesValidator:
    """Validates Falco rules syntax and structure."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate_rules_file(self, file_path: str) -> bool:
        """Validate a Falco rules YAML file."""
        try:
            with open(file_path, 'r') as f:
                rules = yaml.safe_load(f)

            if not isinstance(rules, list):
                self.errors.append(f"Rules file must contain a list of rules, got {type(rules)}")
                return False

            for i, rule in enumerate(rules):
                self.validate_rule(rule, i)

            return len(self.errors) == 0

        except yaml.YAMLError as e:
            self.errors.append(f"YAML parsing error: {e}")
            return False
        except FileNotFoundError:
            self.errors.append(f"Rules file not found: {file_path}")
            return False

    def validate_rule(self, rule: Dict[str, Any], index: int) -> None:
        """Validate a single Falco rule."""
        required_fields = ['rule', 'desc', 'condition', 'output', 'priority']

        # Check required fields
        for field in required_fields:
            if field not in rule:
                self.errors.append(f"Rule {index}: Missing required field '{field}'")

        # Validate priority
        if 'priority' in rule:
            valid_priorities = ['EMERGENCY', 'ALERT', 'CRITICAL', 'ERROR', 'WARNING', 'NOTICE', 'INFO', 'DEBUG']
            if rule['priority'] not in valid_priorities:
                self.errors.append(f"Rule {index}: Invalid priority '{rule['priority']}'. Must be one of {valid_priorities}")

        # Validate condition (basic syntax check)
        if 'condition' in rule:
            condition = str(rule['condition'])
            # Check for common syntax errors
            if 'and' in condition.lower() and 'or' in condition.lower():
                self.warnings.append(f"Rule {index}: Complex condition with both AND and OR - consider using parentheses")

        # Validate output template
        if 'output' in rule:
            output = str(rule['output'])
            # Check for required output fields
            if '%evt.desc' not in output and '%proc.name' not in output:
                self.warnings.append(f"Rule {index}: Output template missing key event information")

        # Validate tags
        if 'tags' in rule:
            if not isinstance(rule['tags'], list):
                self.errors.append(f"Rule {index}: Tags must be a list")
            else:
                valid_tags = ['filesystem', 'network', 'process', 'container', 'security', 'trading', 'anomaly', 'alert']
                for tag in rule['tags']:
                    if tag not in valid_tags:
                        self.warnings.append(f"Rule {index}: Unknown tag '{tag}' - consider adding to valid_tags list")

    def print_report(self) -> None:
        """Print validation report."""
        if self.errors:
            print("❌ Validation Errors:")
            for error in self.errors:
                print(f"  - {error}")
            print()

        if self.warnings:
            print("⚠️  Validation Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
            print()

        if not self.errors and not self.warnings:
            print("✅ All rules validated successfully!")
        elif not self.errors:
            print("✅ No errors found, but some warnings to review.")
        else:
            print("❌ Validation failed. Please fix the errors above.")

def main():
    """Main validation function."""
    print("🛡️  TradPal Falco Rules Validator")
    print("=" * 40)

    rules_file = "tradpal_rules.yaml"

    if not Path(rules_file).exists():
        print(f"❌ Rules file not found: {rules_file}")
        sys.exit(1)

    validator = FalcoRulesValidator()

    if validator.validate_rules_file(rules_file):
        print(f"✅ Successfully validated {rules_file}")
    else:
        print(f"❌ Validation failed for {rules_file}")

    validator.print_report()

    # Exit with error code if there are validation errors
    if validator.errors:
        sys.exit(1)

if __name__ == "__main__":
    main()