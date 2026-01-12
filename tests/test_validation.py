"""Tests for input validation and sanitization functions."""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock, Mock

# Import validation functions
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.validation import (
        sanitize_path,
        validate_path_exists,
        validate_directory_path,
        validate_file_path,
        validate_port,
        is_port_available,
        validate_limit,
        validate_query,
        validate_language,
        validate_repo_id,
        validate_host,
        validate_config_values,
        validate_startup_config,
        CoarchValidationError,
        CoarchConfigError
    )
except ImportError:
    # Fallback for testing - import from local module
    from validation import (
        sanitize_path,
        validate_path_exists,
        validate_directory_path,
        validate_file_path,
        validate_port,
        is_port_available,
        validate_limit,
        validate_query,
        validate_language,
        validate_repo_id,
        validate_host,
        validate_config_values,
        validate_startup_config,
        CoarchValidationError,
        CoarchConfigError
    )


class TestPathSanitization:
    """Test path sanitization functionality."""

    def test_sanitize_valid_path(self):
        """Test sanitizing a valid path."""
        path = "/home/user/project/src"
        result = sanitize_path(path)
        # On Windows, this will be converted to backslashes
        expected = os.path.normpath(path)
        assert result == expected

    def test_sanitize_relative_path(self):
        """Test sanitizing a relative path."""
        path = "./project/src"
        result = sanitize_path(path)
        expected = os.path.normpath(path)
        assert result == expected

    def test_sanitize_path_with_spaces(self):
        """Test sanitizing a path with spaces."""
        path = "/home/user/My Project/src"
        result = sanitize_path(path)
        expected = os.path.normpath(path)
        assert result == expected

    def test_reject_empty_path(self):
        """Test rejection of empty path."""
        with pytest.raises(CoarchValidationError) as exc_info:
            sanitize_path("")
        assert "Path cannot be empty" in str(exc_info.value)

    def test_reject_path_with_null_bytes(self):
        """Test rejection of path with null bytes."""
        with pytest.raises(CoarchValidationError) as exc_info:
            sanitize_path("/path/with/null\x00byte")
        assert "invalid control characters" in str(exc_info.value)

    def test_sanitize_path_with_traversal(self):
        """Test sanitization of path with directory traversal."""
        sanitized = sanitize_path("/path/../../../etc/passwd")
        # Should normalize the path and remove dangerous traversal
        assert sanitized == os.path.normpath("/etc/passwd")

    def test_sanitize_path_with_variables(self):
        """Test sanitization of path with environment variables."""
        sanitized = sanitize_path("/path/${HOME}/file")
        # Should remove environment variable expansion
        assert "${HOME}" not in sanitized


class TestPathValidation:
    """Test path validation functionality."""

    def test_validate_existing_directory(self, tmp_path):
        """Test validation of existing directory."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        result = validate_directory_path(str(test_dir))
        assert result == str(test_dir)

    def test_validate_nonexistent_directory(self, tmp_path):
        """Test validation of non-existent directory."""
        test_dir = tmp_path / "nonexistent"
        
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_directory_path(str(test_dir))
        assert "does not exist" in str(exc_info.value)

    def test_validate_file_as_directory(self, tmp_path):
        """Test validation of file when directory is expected."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_directory_path(str(test_file))
        assert "Path is not a directory" in str(exc_info.value)

    def test_validate_existing_file(self, tmp_path):
        """Test validation of existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        result = validate_file_path(str(test_file))
        assert result == str(test_file)

    def test_validate_nonexistent_file(self, tmp_path):
        """Test validation of non-existent file."""
        test_file = tmp_path / "nonexistent.txt"
        
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_file_path(str(test_file))
        assert "does not exist" in str(exc_info.value)

    def test_validate_directory_as_file(self, tmp_path):
        """Test validation of directory when file is expected."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_file_path(str(test_dir))
        assert "Path is not a file" in str(exc_info.value)


class TestPortValidation:
    """Test port validation functionality."""

    def test_validate_valid_port(self):
        """Test validation of valid port."""
        result = validate_port(8000, must_be_available=False)
        assert result == 8000

    def test_validate_port_minimum(self):
        """Test validation of minimum port."""
        result = validate_port(1, must_be_available=False)
        assert result == 1

    def test_validate_port_maximum(self):
        """Test validation of maximum port."""
        result = validate_port(65535, must_be_available=False)
        assert result == 65535

    def test_reject_invalid_port_string(self):
        """Test rejection of invalid port string."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_port("invalid", must_be_available=False)
        assert "Port must be an integer" in str(exc_info.value)

    def test_reject_port_too_low(self):
        """Test rejection of port below minimum."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_port(0, must_be_available=False)
        assert "out of range" in str(exc_info.value)

    def test_reject_port_too_high(self):
        """Test rejection of port above maximum."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_port(65536, must_be_available=False)
        assert "out of range" in str(exc_info.value)

    def test_check_port_availability(self):
        """Test port availability checking."""
        # Test an unlikely to be used port
        assert is_port_available(12345) is True

    def test_validate_port_in_use(self):
        """Test validation when port is in use."""
        # Mock the socket creation to simulate port in use
        with patch('socket.socket') as mock_socket_class:
            # Mock socket instance
            mock_socket = Mock()
            mock_socket.connect_ex.return_value = 0  # 0 means port is in use
            mock_socket.__enter__ = Mock(return_value=mock_socket)
            mock_socket.__exit__ = Mock(return_value=None)
            mock_socket_class.return_value = mock_socket
            
            with pytest.raises(CoarchValidationError) as exc_info:
                validate_port(80)
            assert "already in use" in str(exc_info.value)


class TestLimitValidation:
    """Test limit validation functionality."""

    def test_validate_valid_limit(self):
        """Test validation of valid limit."""
        result = validate_limit(10)
        assert result == 10

    def test_validate_limit_minimum(self):
        """Test validation of minimum limit."""
        result = validate_limit(1)
        assert result == 1

    def test_validate_limit_maximum(self):
        """Test validation of maximum limit."""
        result = validate_limit(10000)
        assert result == 10000

    def test_validate_custom_limits(self):
        """Test validation with custom bounds."""
        result = validate_limit(5, min_value=1, max_value=100)
        assert result == 5

    def test_reject_invalid_limit_string(self):
        """Test rejection of invalid limit string."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_limit("invalid")
        assert "Limit must be an integer" in str(exc_info.value)

    def test_reject_limit_below_minimum(self):
        """Test rejection of limit below minimum."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_limit(0, min_value=1)
        assert "below minimum" in str(exc_info.value)

    def test_reject_limit_above_maximum(self):
        """Test rejection of limit above maximum."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_limit(50000, max_value=10000)
        assert "exceeds maximum" in str(exc_info.value)


class TestQueryValidation:
    """Test query validation functionality."""

    def test_validate_valid_query(self):
        """Test validation of valid query."""
        result = validate_query("function hello world")
        assert result == "function hello world"

    def test_validate_query_with_whitespace(self):
        """Test validation of query with leading/trailing whitespace."""
        result = validate_query("  hello world  ")
        assert result == "hello world"

    def test_reject_empty_query(self):
        """Test rejection of empty query."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_query("")
        assert "Query cannot be empty" in str(exc_info.value)

    def test_reject_whitespace_only_query(self):
        """Test rejection of whitespace-only query."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_query("   ")
        assert "Query cannot be empty" in str(exc_info.value)

    def test_reject_query_too_long(self):
        """Test rejection of query that's too long."""
        long_query = "a" * 501
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_query(long_query)
        assert "too long" in str(exc_info.value)

    def test_validate_query_with_custom_length(self):
        """Test validation with custom length limit."""
        short_query = "a" * 50
        result = validate_query(short_query, max_length=50)
        assert result == short_query

    def test_reject_query_with_control_chars(self):
        """Test rejection of query with control characters."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_query("hello\x00world")
        assert "control characters" in str(exc_info.value)


class TestLanguageValidation:
    """Test language validation functionality."""

    def test_validate_valid_language(self):
        """Test validation of valid language."""
        result = validate_language("python")
        assert result == "python"

    def test_validate_language_with_dots(self):
        """Test validation of language with dots."""
        result = validate_language("c++")
        assert result == "c++"

    def test_validate_language_with_underscores(self):
        """Test validation of language with underscores."""
        result = validate_language("objective_c")
        assert result == "objective_c"

    def test_reject_empty_language(self):
        """Test rejection of empty language."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_language("")
        assert "Language cannot be empty" in str(exc_info.value)

    def test_reject_invalid_language_chars(self):
        """Test rejection of language with invalid characters."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_language("python@!")
        assert "Invalid language name" in str(exc_info.value)

    def test_validate_with_supported_languages(self):
        """Test validation with supported languages list."""
        supported = ["python", "javascript", "java"]
        result = validate_language("python", supported_languages=supported)
        assert result == "python"

    def test_reject_unsupported_language(self):
        """Test rejection of unsupported language."""
        supported = ["python", "javascript", "java"]
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_language("rust", supported_languages=supported)
        assert "Unsupported language" in str(exc_info.value)


class TestRepositoryIDValidation:
    """Test repository ID validation functionality."""

    def test_validate_valid_repo_id(self):
        """Test validation of valid repository ID."""
        result = validate_repo_id(1)
        assert result == 1

    def test_validate_repo_id_from_string(self):
        """Test validation of repository ID from string."""
        result = validate_repo_id("42")
        assert result == 42

    def test_reject_invalid_repo_id_string(self):
        """Test rejection of invalid repository ID string."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_repo_id("invalid")
        assert "Repository ID must be an integer" in str(exc_info.value)

    def test_reject_zero_repo_id(self):
        """Test rejection of zero repository ID."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_repo_id(0)
        assert "must be positive" in str(exc_info.value)

    def test_reject_negative_repo_id(self):
        """Test rejection of negative repository ID."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_repo_id(-1)
        assert "must be positive" in str(exc_info.value)


class TestHostValidation:
    """Test host validation functionality."""

    def test_validate_valid_hostname(self):
        """Test validation of valid hostname."""
        result = validate_host("localhost")
        assert result == "localhost"

    def test_validate_valid_ip_address(self):
        """Test validation of valid IP address."""
        result = validate_host("127.0.0.1")
        assert result == "127.0.0.1"

    def test_validate_host_with_dots(self):
        """Test validation of host with dots."""
        result = validate_host("example.com")
        assert result == "example.com"

    def test_reject_empty_host(self):
        """Test rejection of empty host."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_host("")
        assert "Host cannot be empty" in str(exc_info.value)

    def test_reject_invalid_host_chars(self):
        """Test rejection of host with invalid characters."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_host("host@!")
        assert "Invalid host format" in str(exc_info.value)

    def test_reject_invalid_ip_address(self):
        """Test rejection of invalid IP address."""
        with pytest.raises(CoarchValidationError) as exc_info:
            validate_host("999.999.999.999")
        assert "Invalid IP address" in str(exc_info.value)


class TestConfigurationValidation:
    """Test configuration validation functionality."""

    def test_validate_config_values(self):
        """Test validation of configuration values."""
        config = {
            'max_workers': '4',
            'chunk_size': '1000',
            'log_level': 'info'
        }
        
        validation_rules = {
            'max_workers': lambda x: validate_limit(x, min_value=1, max_value=32),
            'chunk_size': lambda x: validate_limit(x, min_value=100, max_value=10000),
            'log_level': lambda x: x.upper() if x.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] else 'INFO'
        }
        
        result = validate_config_values(config, validation_rules)
        assert result['max_workers'] == 4
        assert result['chunk_size'] == 1000
        assert result['log_level'] == 'INFO'

    def test_validate_startup_config(self):
        """Test validation of startup configuration."""
        with patch.dict(os.environ, {
            'COARCH_INDEX_PATH': '/tmp/test_index',
            'COARCH_MAX_WORKERS': '8',
            'COARCH_LOG_LEVEL': 'DEBUG'
        }, clear=False):
            config = validate_startup_config()
            assert 'index_path' in config
            assert 'max_workers' in config
            assert 'log_level' in config
            assert config['log_level'] == 'DEBUG'

    def test_validate_startup_config_invalid_values(self):
        """Test validation of startup configuration with invalid values."""
        with patch.dict(os.environ, {
            'COARCH_MAX_WORKERS': 'invalid'
        }, clear=False):
            with pytest.raises(CoarchConfigError):
                validate_startup_config()


class TestPortAvailability:
    """Test port availability checking."""

    def test_port_availability_success(self):
        """Test successful port availability check."""
        with patch('socket.socket') as mock_socket_class:
            mock_socket = Mock()
            mock_socket.connect_ex.return_value = 1  # Port available (non-zero)
            mock_socket.__enter__ = Mock(return_value=mock_socket)
            mock_socket.__exit__ = Mock(return_value=None)
            mock_socket_class.return_value = mock_socket
            assert is_port_available(8000) is True

    def test_port_availability_failure(self):
        """Test failed port availability check."""
        with patch('socket.socket') as mock_socket_class:
            mock_socket = Mock()
            mock_socket.connect_ex.return_value = 0  # Port in use (zero)
            mock_socket.__enter__ = Mock(return_value=mock_socket)
            mock_socket.__exit__ = Mock(return_value=None)
            mock_socket_class.return_value = mock_socket
            assert is_port_available(8000) is False

    def test_port_availability_exception(self):
        """Test port availability check with exception."""
        with patch('socket.socket', side_effect=Exception("Network error")):
            assert is_port_available(8000) is False


class TestIntegration:
    """Integration tests for validation workflow."""

    def test_validation_chain(self):
        """Test chaining validation operations."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test path validation and sanitization chain
            path = os.path.join(temp_dir, "subdir", "..", "test.txt")
            sanitized = sanitize_path(path)
            
            # Create the file
            with open(sanitized, 'w') as f:
                f.write("test content")
            
            # Validate the file path
            validated_path = validate_file_path(sanitized)
            assert validated_path == sanitized

    def test_validation_with_click_equivalents(self):
        """Test validation functions that would be used with Click."""
        # Test port validation equivalent to Click option
        port = "8080"
        validated_port = validate_port(port)
        assert isinstance(validated_port, int)
        assert validated_port == 8080

        # Test limit validation equivalent to Click option
        limit = "25"
        validated_limit = validate_limit(limit)
        assert isinstance(validated_limit, int)
        assert validated_limit == 25

        # Test query validation equivalent to Click argument
        query = "  function search  "
        validated_query = validate_query(query)
        assert validated_query == "function search"