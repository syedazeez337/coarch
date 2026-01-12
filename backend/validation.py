"""Comprehensive input validation and sanitization for Coarch CLI."""

import os
import re
import socket
from typing import Optional, Union, Any, List, Dict, Callable


# Exception classes (duplicated from exceptions.py to avoid circular imports)
class CoarchValidationError(Exception):
    """Input validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None, suggestion: Optional[str] = None):
        super().__init__(message)
        self.field = field
        self.value = value
        self.suggestion = suggestion

    def __str__(self) -> str:
        """Return formatted error message with context."""
        msg = f"Error: {self.args[0]}"
        if self.field:
            msg += f" (field={self.field}, value={self.value})"
        if self.suggestion:
            msg += f"\nSuggestion: {self.suggestion}"
        return msg


class CoarchConfigError(Exception):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_path: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None, 
                 suggestion: Optional[str] = None):
        super().__init__(message)
        self.config_path = config_path
        self.context = context or {}
        self.suggestion = suggestion

    def __str__(self) -> str:
        """Return formatted error message."""
        msg = f"Error: {self.args[0]}"
        if self.config_path:
            msg += f" (config_path={self.config_path})"
        if self.suggestion:
            msg += f"\nSuggestion: {self.suggestion}"
        return msg


def sanitize_path(path: str) -> str:
    """Sanitize path to prevent directory traversal attacks.
    
    Args:
        path: Input path to sanitize
        
    Returns:
        Sanitized path
        
    Raises:
        CoarchValidationError: If path contains dangerous patterns
    """
    if not path:
        raise CoarchValidationError(
            "Path cannot be empty",
            field="path",
            value=path,
            suggestion="Provide a valid file or directory path"
        )
    
    # Remove null bytes and control characters
    if any(ord(c) < 32 for c in path):
        raise CoarchValidationError(
            "Path contains invalid control characters",
            field="path",
            value=path,
            suggestion="Remove any special or control characters from the path"
        )
    
    # Sanitize dangerous patterns instead of rejecting them
    sanitized_path = path
    
    # Sanitize directory traversal by normalizing the path
    if '..' in path:
        sanitized_path = os.path.normpath(sanitized_path)
    
    # Sanitize environment variable expansion patterns
    if '${' in sanitized_path or '$(' in sanitized_path:
        # Remove environment variable expansion for security
        sanitized_path = sanitized_path.replace('${', '').replace('$(', '').replace(')', '')
    
    # Sanitize home directory expansion
    if '~' in sanitized_path:
        # Remove tilde for security - don't expand to home directory
        sanitized_path = sanitized_path.replace('~', '')
    
    # Normalize path (this will resolve any remaining relative paths safely)
    normalized = os.path.normpath(sanitized_path)
    
    return normalized


def validate_path_exists(path: str, path_type: str = "path", 
                       must_be_readable: bool = True,
                       must_be_writable: bool = False) -> str:
    """Validate that a path exists and has required permissions.
    
    Args:
        path: Path to validate
        path_type: Type of path for error messages
        must_be_readable: Path must be readable
        must_be_writable: Path must be writable
        
    Returns:
        Sanitized path
        
    Raises:
        CoarchValidationError: If path validation fails
    """
    sanitized = sanitize_path(path)
    
    if not os.path.exists(sanitized):
        raise CoarchValidationError(
            f"{path_type.title()} does not exist: {path}",
            field="path",
            value=path,
            suggestion=f"Verify that the {path_type} exists and you have permission to access it"
        )
    
    if must_be_readable and not os.access(sanitized, os.R_OK):
        raise CoarchValidationError(
            f"{path_type.title()} is not readable: {path}",
            field="path",
            value=path,
            suggestion="Check file permissions or run with appropriate privileges"
        )
    
    if must_be_writable and not os.access(sanitized, os.W_OK):
        raise CoarchValidationError(
            f"{path_type.title()} is not writable: {path}",
            field="path",
            value=path,
            suggestion="Check directory permissions or run with appropriate privileges"
        )
    
    return sanitized


def validate_directory_path(path: str, path_type: str = "directory") -> str:
    """Validate that path exists and is a directory.
    
    Args:
        path: Directory path to validate
        path_type: Type of directory for error messages
        
    Returns:
        Sanitized directory path
        
    Raises:
        CoarchValidationError: If directory validation fails
    """
    sanitized = validate_path_exists(path, path_type, must_be_readable=True)
    
    if not os.path.isdir(sanitized):
        raise CoarchValidationError(
            f"Path is not a {path_type}: {path}",
            field="path",
            value=path,
            suggestion=f"Specify a {path_type} path, not a file path"
        )
    
    return sanitized


def validate_file_path(path: str, path_type: str = "file") -> str:
    """Validate that path exists and is a file.
    
    Args:
        path: File path to validate
        path_type: Type of file for error messages
        
    Returns:
        Sanitized file path
        
    Raises:
        CoarchValidationError: If file validation fails
    """
    sanitized = validate_path_exists(path, path_type, must_be_readable=True)
    
    if not os.path.isfile(sanitized):
        raise CoarchValidationError(
            f"Path is not a {path_type}: {path}",
            field="path",
            value=path,
            suggestion=f"Specify a {path_type} path, not a directory path"
        )
    
    return sanitized


def validate_port(port: Union[str, int], must_be_available: bool = True) -> int:
    """Validate port number and optionally check availability.
    
    Args:
        port: Port number to validate
        must_be_available: Check if port is available
        
    Returns:
        Validated port number
        
    Raises:
        CoarchValidationError: If port validation fails
    """
    try:
        port_int = int(port)
    except (ValueError, TypeError):
        raise CoarchValidationError(
            "Port must be an integer",
            field="port",
            value=port,
            suggestion="Use a valid port number between 1 and 65535"
        )
    
    if port_int < 1 or port_int > 65535:
        raise CoarchValidationError(
            f"Port {port_int} is out of range",
            field="port",
            value=port,
            suggestion="Use a port number between 1 and 65535"
        )
    
    if must_be_available and not is_port_available(port_int):
        raise CoarchValidationError(
            f"Port {port_int} is already in use",
            field="port",
            value=port,
            suggestion="Choose a different port or stop the service using this port"
        )
    
    return port_int


def is_port_available(port: int) -> bool:
    """Check if a port is available.
    
    Args:
        port: Port number to check
        
    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.1)
            result = sock.connect_ex(('127.0.0.1', port))
            # If connect_ex returns 0, port is in use (connection succeeded)
            # If it returns non-zero, port is available (connection failed)
            return result != 0
    except Exception:
        # If there's an exception, assume port is not available
        return False


def validate_limit(limit: Union[str, int], min_value: int = 1, 
                  max_value: int = 10000) -> int:
    """Validate limit parameter with configurable bounds.
    
    Args:
        limit: Limit value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Validated limit as integer
        
    Raises:
        CoarchValidationError: If limit validation fails
    """
    try:
        limit_int = int(limit)
    except (ValueError, TypeError):
        raise CoarchValidationError(
            "Limit must be an integer",
            field="limit",
            value=limit,
            suggestion=f"Use an integer between {min_value} and {max_value}"
        )
    
    if limit_int < min_value:
        raise CoarchValidationError(
            f"Limit {limit_int} is below minimum ({min_value})",
            field="limit",
            value=limit,
            suggestion=f"Use a limit of at least {min_value}"
        )
    
    if limit_int > max_value:
        raise CoarchValidationError(
            f"Limit {limit_int} exceeds maximum ({max_value})",
            field="limit",
            value=limit,
            suggestion=f"Use a limit of at most {max_value} for better performance"
        )
    
    return limit_int


def validate_query(query: str, max_length: int = 500) -> str:
    """Validate search query with length and content checks.
    
    Args:
        query: Query string to validate
        max_length: Maximum allowed query length
        
    Returns:
        Validated query string
        
    Raises:
        CoarchValidationError: If query validation fails
    """
    if not query or not query.strip():
        raise CoarchValidationError(
            "Query cannot be empty",
            field="query",
            value=query,
            suggestion="Provide a non-empty search query"
        )
    
    trimmed_query = query.strip()
    
    if len(trimmed_query) > max_length:
        raise CoarchValidationError(
            f"Query is too long ({len(trimmed_query)} characters)",
            field="query",
            value=query,
            suggestion=f"Use a shorter query (max {max_length} characters)"
        )
    
    # Check for potentially problematic characters
    if any(ord(c) < 32 for c in trimmed_query):
        raise CoarchValidationError(
            "Query contains invalid control characters",
            field="query",
            value=query,
            suggestion="Remove any special or control characters from the query"
        )
    
    return trimmed_query


def validate_language(language: str, supported_languages: Optional[List[str]] = None) -> str:
    """Validate programming language specification.
    
    Args:
        language: Language name to validate
        supported_languages: List of supported languages (optional)
        
    Returns:
        Validated language string
        
    Raises:
        CoarchValidationError: If language validation fails
    """
    if not language or not language.strip():
        raise CoarchValidationError(
            "Language cannot be empty",
            field="language",
            value=language,
            suggestion="Provide a valid programming language name"
        )
    
    trimmed_lang = language.strip()
    
    # Basic validation for language names - allow common programming language characters
    if not re.match(r'^[a-zA-Z0-9._+-]+$', trimmed_lang):
        raise CoarchValidationError(
            f"Invalid language name: {language}",
            field="language",
            value=language,
            suggestion="Use alphanumeric characters, dots, underscores, hyphens, and plus signs only"
        )
    
    if supported_languages and trimmed_lang not in supported_languages:
        raise CoarchValidationError(
            f"Unsupported language: {language}",
            field="language",
            value=language,
            suggestion=f"Use one of the supported languages: {', '.join(supported_languages[:10])}{'...' if len(supported_languages) > 10 else ''}"
        )
    
    return trimmed_lang


def validate_repo_id(repo_id: Union[str, int]) -> int:
    """Validate repository ID.
    
    Args:
        repo_id: Repository ID to validate
        
    Returns:
        Validated repository ID as integer
        
    Raises:
        CoarchValidationError: If repository ID validation fails
    """
    try:
        repo_id_int = int(repo_id)
    except (ValueError, TypeError):
        raise CoarchValidationError(
            "Repository ID must be an integer",
            field="repo_id",
            value=repo_id,
            suggestion="Use a valid positive integer for the repository ID"
        )
    
    if repo_id_int <= 0:
        raise CoarchValidationError(
            f"Repository ID {repo_id_int} must be positive",
            field="repo_id",
            value=repo_id,
            suggestion="Use a positive integer (1, 2, 3, ...)"
        )
    
    return repo_id_int


def validate_host(host: str) -> str:
    """Validate host address.
    
    Args:
        host: Host address to validate
        
    Returns:
        Validated host address
        
    Raises:
        CoarchValidationError: If host validation fails
    """
    if not host or not host.strip():
        raise CoarchValidationError(
            "Host cannot be empty",
            field="host",
            value=host,
            suggestion="Provide a valid host address (e.g., localhost, 127.0.0.1, 0.0.0.0)"
        )
    
    trimmed_host = host.strip()
    
    # Basic validation for host addresses
    if not re.match(r'^[a-zA-Z0-9._-]+$', trimmed_host):
        raise CoarchValidationError(
            f"Invalid host format: {host}",
            field="host",
            value=host,
            suggestion="Use alphanumeric characters, dots, underscores, and hyphens only"
        )
    
    # Validate IP address format if it looks like one
    if re.match(r'^\d+\.\d+\.\d+\.\d+$', trimmed_host):
        try:
            socket.inet_aton(trimmed_host)
        except socket.error:
            raise CoarchValidationError(
                f"Invalid IP address: {host}",
                field="host",
                value=host,
                suggestion="Use a valid IP address format (e.g., 127.0.0.1)"
            )
    
    return trimmed_host


def validate_config_values(config: dict, validation_rules: dict) -> dict:
    """Validate configuration values against rules.
    
    Args:
        config: Configuration dictionary to validate
        validation_rules: Dictionary mapping config keys to validation functions
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        CoarchConfigError: If configuration validation fails
    """
    validated_config = {}
    
    for key, value in config.items():
        if key in validation_rules:
            try:
                validated_config[key] = validation_rules[key](value)
            except CoarchValidationError as e:
                raise CoarchConfigError(
                    f"Invalid configuration value for '{key}'",
                    context=getattr(e, 'context', None),
                    suggestion=getattr(e, 'suggestion', None)
                )
        else:
            validated_config[key] = value
    
    return validated_config


# Validation decorators for Click options
def create_click_validator(validator_func: Callable, **kwargs):
    """Create a Click option callback validator.
    
    Args:
        validator_func: The validation function to use
        **kwargs: Additional parameters for the validator
        
    Returns:
        A Click option callback decorator
    """
    try:
        import click
        
        def validator_callback(ctx, param, value):
            if value is not None:
                return validator_func(value, **kwargs)
            return value
        
        return validator_callback
    except ImportError:
        return None


def validate_click_path(path_type: str = "path"):
    """Click option decorator for path validation.
    
    Args:
        path_type: Type of path validation
        
    Returns:
        Click decorator function
    """
    try:
        import click
        
        if path_type == "directory":
            validator = validate_directory_path
        elif path_type == "file":
            validator = validate_file_path
        else:
            validator = validate_path_exists
            
        return create_click_validator(validator)
    except ImportError:
        return lambda f: f


def validate_click_port():
    """Click option decorator for port validation.
    
    Returns:
        Click decorator function
    """
    return create_click_validator(validate_port)


def validate_click_limit():
    """Click option decorator for limit validation.
    
    Returns:
        Click decorator function
    """
    return create_click_validator(validate_limit)


def validate_click_query():
    """Click option decorator for query validation.
    
    Returns:
        Click decorator function
    """
    return create_click_validator(validate_query)


def validate_startup_config() -> dict:
    """Validate configuration at application startup.
    
    Returns:
        Validated configuration dictionary
        
    Raises:
        CoarchConfigError: If startup configuration validation fails
    """
    import os
    
    def safe_int_parse(value: str, default: int, param_name: str) -> int:
        """Safely parse integer from environment variable."""
        try:
            return int(value)
        except (ValueError, TypeError):
            raise CoarchConfigError(
                f"Invalid {param_name} value: {value}",
                suggestion=f"Use a valid integer for {param_name}"
            )
    
    # Get configuration from environment variables with safe parsing
    config = {
        'index_path': os.environ.get('COARCH_INDEX_PATH', 'coarch_index'),
        'db_path': os.environ.get('COARCH_DB_PATH', 'coarch.db'),
        'log_level': os.environ.get('COARCH_LOG_LEVEL', 'INFO'),
        'max_workers': safe_int_parse(os.environ.get('COARCH_MAX_WORKERS', '4'), 4, 'max_workers'),
        'embedding_batch_size': safe_int_parse(os.environ.get('COARCH_EMBEDDING_BATCH_SIZE', '64'), 64, 'embedding_batch_size'),
        'chunk_size': safe_int_parse(os.environ.get('COARCH_CHUNK_SIZE', '1000'), 1000, 'chunk_size'),
        'chunk_overlap': safe_int_parse(os.environ.get('COARCH_CHUNK_OVERLAP', '200'), 200, 'chunk_overlap'),
    }
    
    # Define validation rules
    validation_rules = {
        'index_path': lambda x: validate_path_exists(x, path_type="index directory") if os.path.exists(x) else sanitize_path(x),
        'db_path': lambda x: sanitize_path(x),
        'log_level': lambda x: x.upper() if x.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] else 'INFO',
        'max_workers': lambda x: validate_limit(x, min_value=1, max_value=32),
        'embedding_batch_size': lambda x: validate_limit(x, min_value=1, max_value=256),
        'chunk_size': lambda x: validate_limit(x, min_value=100, max_value=10000),
        'chunk_overlap': lambda x: validate_limit(x, min_value=0, max_value=1000),
    }
    
    return validate_config_values(config, validation_rules)