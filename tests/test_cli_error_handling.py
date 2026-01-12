"""Test CLI error handling improvements."""

import pytest
import sys
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from cli.main import main
from backend.exceptions import (
    CoarchCLIError,
    CoarchConfigError,
    CoarchIndexError,
    CoarchSearchError,
    CoarchValidationError,
    ExitCode,
    handle_cli_error,
    validate_path_exists,
    validate_port,
    validate_limit,
    validate_query,
)


class TestExceptionHierarchy:
    """Test the exception hierarchy works correctly."""

    def test_base_exception(self):
        """Test CoarchCLIError base class."""
        error = CoarchCLIError("Test error")
        assert str(error) == "Error: Test error"
        assert error.exit_code == ExitCode.GENERAL_ERROR

    def test_config_error(self):
        """Test CoarchConfigError specific behavior."""
        error = CoarchConfigError("Config failed", config_path="/tmp/config")
        assert "config_path=/tmp/config" in str(error)
        assert error.exit_code == ExitCode.CONFIGURATION_ERROR
        assert "configuration" in error.suggestion.lower()

    def test_index_error(self):
        """Test CoarchIndexError specific behavior."""
        error = CoarchIndexError("Index failed", repo_path="/repo", files_processed=5)
        assert "repo_path=/repo" in str(error)
        assert "files_processed=5" in str(error)
        assert error.exit_code == ExitCode.INDEXING_ERROR

    def test_search_error(self):
        """Test CoarchSearchError specific behavior."""
        error = CoarchSearchError("Search failed", query="test query", index_path="/index")
        assert "query=test query" in str(error)
        assert "index_path=/index" in str(error)
        assert error.exit_code == ExitCode.SEARCH_ERROR

    def test_validation_error(self):
        """Test CoarchValidationError specific behavior."""
        error = CoarchValidationError("Invalid input", field="port", value=70000)
        assert "field=port" in str(error)
        assert "value=70000" in str(error)
        assert error.exit_code == ExitCode.VALIDATION_ERROR


class TestValidationFunctions:
    """Test input validation functions."""

    def test_validate_path_exists_valid(self):
        """Test path validation with existing path."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise
            validate_path_exists(temp_dir)

    def test_validate_path_exists_invalid(self):
        """Test path validation with non-existent path."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_path_exists("/non/existent/path")
        
        assert exc_info.value.exit_code == ExitCode.VALIDATION_ERROR

    def test_validate_port_valid(self):
        """Test port validation with valid ports."""
        validate_port(8000)
        validate_port(1)
        validate_port(65535)

    def test_validate_port_invalid(self):
        """Test port validation with invalid ports."""
        with pytest.raises(CoarchValidationError):
            validate_port(0)
        
        with pytest.raises(CoarchValidationError):
            validate_port(70000)
        
        with pytest.raises(CoarchValidationError):
            validate_port("not-a-number")

    def test_validate_limit_valid(self):
        """Test limit validation with valid values."""
        validate_limit(10)
        validate_limit(100)
        validate_limit(1)

    def test_validate_limit_invalid(self):
        """Test limit validation with invalid values."""
        with pytest.raises(CoarchValidationError):
            validate_limit(0)
        
        with pytest.raises(CoarchValidationError):
            validate_limit(-5)
        
        with pytest.raises(CoarchValidationError):
            validate_limit(15000)

    def test_validate_query_valid(self):
        """Test query validation with valid queries."""
        validate_query("test query")
        validate_query("function hello_world()")
        validate_query("import os")

    def test_validate_query_invalid(self):
        """Test query validation with invalid queries."""
        with pytest.raises(CoarchValidationError):
            validate_query("")
        
        with pytest.raises(CoarchValidationError):
            validate_query("   ")
        
        # Test very long query
        long_query = "a" * 600
        with pytest.raises(CoarchValidationError):
            validate_query(long_query)


class TestErrorHandling:
    """Test the CLI error handling system."""

    def test_handle_cli_error_custom(self):
        """Test handling of custom Coarch errors."""
        error = CoarchValidationError("Test validation error")
        
        with pytest.raises(SystemExit) as exc_info:
            handle_cli_error(error, verbose=False)
        
        assert exc_info.value.code == ExitCode.VALIDATION_ERROR.value

    def test_handle_cli_error_generic(self):
        """Test handling of generic exceptions."""
        error = ValueError("Generic error")
        
        with pytest.raises(SystemExit) as exc_info:
            handle_cli_error(error, verbose=False)
        
        assert exc_info.value.code == ExitCode.GENERAL_ERROR.value

    def test_handle_cli_error_verbose(self):
        """Test verbose error handling includes debug info."""
        error = RuntimeError("Test error")
        
        with patch('builtins.print') as mock_print:
            with pytest.raises(SystemExit):
                handle_cli_error(error, verbose=True)
        
        # Should print debug information
        assert mock_print.called


class TestCLICommands:
    """Test CLI commands with error handling."""

    def test_index_nonexistent_path(self):
        """Test index command with non-existent path."""
        runner = CliRunner()
        result = runner.invoke(main, ['index', '/non/existent/path'])
        
        # Should exit with validation error
        assert result.exit_code != 0

    def test_search_empty_query(self):
        """Test search command with empty query."""
        runner = CliRunner()
        result = runner.invoke(main, ['search', ''])
        
        # Should exit with validation error
        assert result.exit_code != 0

    def test_search_invalid_limit(self):
        """Test search command with invalid limit."""
        runner = CliRunner()
        result = runner.invoke(main, ['search', 'test', '--limit', '0'])
        
        # Should exit with validation error
        assert result.exit_code != 0

    def test_serve_invalid_port(self):
        """Test serve command with invalid port."""
        runner = CliRunner()
        result = runner.invoke(main, ['serve', '--port', '70000'])
        
        # Should exit with validation error
        assert result.exit_code != 0

    def test_delete_invalid_repo_id(self):
        """Test delete command with invalid repo ID."""
        runner = CliRunner()
        result = runner.invoke(main, ['delete', '-1'])
        
        # Should exit with validation error
        assert result.exit_code != 0


class TestUserFriendlyMessages:
    """Test that error messages are user-friendly."""

    def test_validation_error_message_format(self):
        """Test validation error messages are helpful."""
        error = CoarchValidationError(
            "Port must be between 1 and 65535",
            field="port",
            value=70000,
            suggestion="Use a port number between 1 and 65535"
        )
        
        message = str(error)
        assert "Error:" in message
        assert "field=port" in message
        assert "value=70000" in message
        assert "Suggestion:" in message
        assert "Use a port number" in message

    def test_index_error_context(self):
        """Test index errors include helpful context."""
        error = CoarchIndexError(
            "Failed to read repository",
            repo_path="/tmp/repo",
            files_processed=10,
            suggestion="Check repository permissions and structure"
        )
        
        message = str(error)
        assert "repo_path=/tmp/repo" in message
        assert "files_processed=10" in message
        assert "Check repository permissions" in message

    def test_search_error_context(self):
        """Test search errors include helpful context."""
        error = CoarchSearchError(
            "Index not found",
            query="test function",
            index_path="/tmp/index",
            suggestion="Run 'coarch index <path>' first to create an index"
        )
        
        message = str(error)
        assert "query=test function" in message
        assert "index_path=/tmp/index" in message
        assert "Run 'coarch index" in message


if __name__ == "__main__":
    pytest.main([__file__])