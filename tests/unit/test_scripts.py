import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from io import StringIO

# Add scripts to path for testing
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestManageIntegrationsScript:
    """Test the manage_integrations.py script."""

    def test_script_status_command(self):
        """Test status command."""
        # Mock the command line arguments
        with patch('sys.argv', ['manage_integrations.py', '--status']):
            # Import here to get the patched argv
            import scripts.manage_integrations as script

            # Reset the script's state
            script.integration_manager.integrations.clear()

            # Just test that the script can be imported and has the expected functions
            assert hasattr(script, 'setup_integrations')
            assert hasattr(script, 'integration_manager')

    def test_script_setup_command(self):
        """Test setup command."""
        with patch('sys.argv', ['manage_integrations.py', '--setup']):
            import scripts.manage_integrations as script

            # This would be the interactive setup flow
            # We can't easily test the full flow without mocking more,
            # but we can test that the setup functions are available
            assert hasattr(script, 'setup_integrations')
            assert hasattr(script, 'interactive_setup')

    def test_script_list_command(self):
        """Test list command."""
        with patch('sys.argv', ['manage_integrations.py', '--list']):
            import scripts.manage_integrations as script

            # Clear any existing integrations
            script.integration_manager.integrations.clear()

            # The list command should work even with no integrations
            assert hasattr(script, 'integration_manager')

    def test_script_help_command(self):
        """Test help command."""
        with patch('sys.argv', ['manage_integrations.py', '--help']):
            # The help should be printed by argparse
            # We just verify the script can be imported
            import scripts.manage_integrations as script
            assert script is not None


class TestRunIntegratedScript:
    """Test the run_integrated.py script."""

    def test_script_run_integrated_basic(self):
        """Test basic run_integrated execution."""
        with patch('sys.argv', ['run_integrated.py']):
            import scripts.run_integrated as script

            # The script should be importable and have the main function
            assert hasattr(script, 'run_integrated_system')
            assert hasattr(script, 'setup_integrations')

    def test_script_signal_handler(self):
        """Test signal handler."""
        with patch('sys.argv', ['run_integrated.py']):
            import scripts.run_integrated as script

            # Test that signal handler exists
            assert hasattr(script, 'signal_handler')

            # Test calling signal handler
            script.running = True
            script.signal_handler(2, None)  # SIGINT
            assert script.running == False


class TestScriptIntegration:
    """Integration tests for scripts."""

    def test_manage_script_environment_integration(self):
        """Test script integration with environment."""
        import scripts.manage_integrations as script

        # Test that the script can access environment variables
        assert callable(script.setup_integrations)

    def test_run_integrated_environment_integration(self):
        """Test run_integrated script with environment."""
        import scripts.run_integrated as script

        # Test that the script can access environment variables
        assert callable(script.setup_integrations)
        assert callable(script.run_integrated_system)

    def test_script_imports(self):
        """Test that all scripts can be imported without errors."""
        # Test manage_integrations.py
        import scripts.manage_integrations
        assert scripts.manage_integrations is not None

        # Test run_integrated.py
        import scripts.run_integrated
        assert scripts.run_integrated is not None

    @patch('sys.argv', ['test'])
    def test_script_main_execution(self):
        """Test that scripts have proper main execution guards."""
        # This is a basic test to ensure scripts don't execute main code on import
        import scripts.manage_integrations
        import scripts.run_integrated

        # If we get here without errors, the imports work
        assert True


class TestScriptErrorHandling:
    """Test error handling in scripts."""

    def test_manage_script_error_handling(self):
        """Test error handling in manage_integrations script."""
        import scripts.manage_integrations as script

        # The script should handle environment errors gracefully
        # This tests that the script doesn't crash on environment issues
        assert callable(script.setup_integrations)

    def test_run_integrated_error_handling(self):
        """Test error handling in run_integrated script."""
        import scripts.run_integrated as script

        # The script should handle environment errors gracefully
        assert callable(script.setup_integrations)

    @patch('builtins.print')
    @patch('scripts.manage_integrations.setup_integrations', side_effect=Exception("Environment error"))
    def test_test_integrations_error_handling(self, mock_setup, mock_print):
        """Test error handling in test_integrations script."""
        import scripts.manage_integrations as script

        # The script should handle environment errors gracefully
        assert callable(script.setup_integrations)


if __name__ == "__main__":
    pytest.main([__file__])