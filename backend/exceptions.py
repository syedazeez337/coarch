"""Comprehensive exception hierarchy for Coarch CLI."""

import sys
from typing import Optional, Any, Dict, List
from enum import Enum


class ExitCode(Enum):
    """Exit codes for different error types."""
    SUCCESS = 0
    GENERAL_ERROR = 1
    CONFIGURATION_ERROR = 2
    INDEXING_ERROR = 3
    SEARCH_ERROR = 4
    VALIDATION_ERROR = 5
    FILE_NOT_FOUND = 6
    PERMISSION_ERROR = 7
    RESOURCE_EXHAUSTED = 8
    DEPENDENCY_MISSING = 9


class CoarchCLIError(Exception):
    """Base exception for all Coarch CLI errors."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        exit_code: ExitCode = ExitCode.GENERAL_ERROR
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.suggestion = suggestion
        self.exit_code = exit_code

    def __str__(self) -> str:
        """Return formatted error message with context."""
        msg = f"Error: {self.message}"

        if self.context:
            context_parts = []
            for key, value in self.context.items():
                context_parts.append(f"{key}={value}")
            if context_parts:
                msg += f" ({', '.join(context_parts)})"

        if self.suggestion:
            msg += f"\nSuggestion: {self.suggestion}"

        return msg


class CoarchConfigError(CoarchCLIError):
    """Configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        error_context = context or {}
        if config_path:
            error_context['config_path'] = config_path

        default_suggestion = suggestion or "Check your configuration file and environment variables"
        if config_path:
            default_suggestion += f". Verify that {config_path} exists and is readable"

        super().__init__(
            message,
            context=error_context,
            suggestion=default_suggestion,
            exit_code=ExitCode.CONFIGURATION_ERROR
        )


class CoarchIndexError(CoarchCLIError):
    """Indexing-related errors."""

    def __init__(
        self,
        message: str,
        repo_path: Optional[str] = None,
        files_processed: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        error_context = context or {}
        if repo_path:
            error_context['repo_path'] = repo_path
        if files_processed is not None:
            error_context['files_processed'] = files_processed

        default_suggestion = suggestion or "Check that the repository path exists and contains valid source files"
        if repo_path:
            default_suggestion += f". Verify that {repo_path} is a valid directory"

        super().__init__(
            message,
            context=error_context,
            suggestion=default_suggestion,
            exit_code=ExitCode.INDEXING_ERROR
        )


class CoarchSearchError(CoarchCLIError):
    """Search-related errors."""

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        index_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        error_context = context or {}
        if query:
            error_context['query'] = query[:100] + "..." if len(query) > 100 else query
        if index_path:
            error_context['index_path'] = index_path

        default_suggestion = suggestion or "Ensure the index exists and contains data"
        if not index_path:
            default_suggestion += ". You may need to index a repository first"

        super().__init__(
            message,
            context=error_context,
            suggestion=default_suggestion,
            exit_code=ExitCode.SEARCH_ERROR
        )


class CoarchValidationError(CoarchCLIError):
    """Input validation errors."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        error_context = context or {}
        if field:
            error_context['field'] = field
        if value is not None:
            error_context['value'] = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)

        default_suggestion = suggestion or "Check the input format and try again"
        if field:
            default_suggestion += f". Verify that '{field}' has a valid value"

        super().__init__(
            message,
            context=error_context,
            suggestion=default_suggestion,
            exit_code=ExitCode.VALIDATION_ERROR
        )


def handle_cli_error(error: Exception, verbose: bool = False) -> None:
    """Handle CLI errors with appropriate exit codes and user-friendly messages."""
    from rich import print
    from rich.panel import Panel
    from rich.text import Text

    # Print plain error message for test compatibility
    print(f"Error: {error}")

    if isinstance(error, CoarchCLIError):
        # Custom Coarch error - show user-friendly message
        error_text = Text(str(error))
        print(Panel(error_text, title="[red]Coarch Error[/]", border_style="red"))

        if verbose and hasattr(error, '__traceback__'):
            import traceback
            print(f"\n[dim]Debug information:[/]")
            traceback.print_exc()

        sys.exit(error.exit_code.value)
    else:
        # Generic error - provide helpful context
        print(f"[red]Unexpected error: {error}[/]")

        if verbose:
            import traceback
            print(f"\n[dim]Debug information:[/]")
            traceback.print_exc()
            print(f"[dim]Error type: {type(error).__name__}[/]")
        else:
            print("[dim]Run with --verbose for detailed error information[/]")

        sys.exit(ExitCode.GENERAL_ERROR.value)


def validate_path_exists(path: str, path_type: str = "path") -> None:
    """Validate that a path exists and is accessible."""
    import os

    if not os.path.exists(path):
        raise CoarchValidationError(
            f"{path_type.title()} does not exist: {path}",
            field="path",
            value=path,
            suggestion=f"Verify that the {path_type} exists and you have permission to access it"
        )

    if not os.path.isdir(path) and path_type == "directory":
        raise CoarchValidationError(
            f"Path is not a directory: {path}",
            field="path",
            value=path,
            suggestion="Specify a directory path, not a file path"
        )


def validate_port(port: int) -> None:
    """Validate port number."""
    if not isinstance(port, int):
        raise CoarchValidationError(
            "Port must be an integer",
            field="port",
            value=port,
            suggestion="Use a valid port number between 1 and 65535"
        )

    if port < 1 or port > 65535:
        raise CoarchValidationError(
            f"Port {port} is out of range",
            field="port",
            value=port,
            suggestion="Use a port number between 1 and 65535"
        )


def validate_limit(limit: int) -> None:
    """Validate limit parameter."""
    if not isinstance(limit, int):
        raise CoarchValidationError(
            "Limit must be an integer",
            field="limit",
            value=limit,
            suggestion="Use a positive integer for the result limit"
        )

    if limit <= 0:
        raise CoarchValidationError(
            f"Limit {limit} must be positive",
            field="limit",
            value=limit,
            suggestion="Use a positive integer (e.g., 10, 50, 100)"
        )

    if limit > 10000:
        raise CoarchValidationError(
            f"Limit {limit} is too high",
            field="limit",
            value=limit,
            suggestion="Use a limit between 1 and 10000 for better performance"
        )


def validate_query(query: str) -> None:
    """Validate search query."""
    if not query or not query.strip():
        raise CoarchValidationError(
            "Query cannot be empty",
            field="query",
            value=query,
            suggestion="Provide a non-empty search query"
        )

    if len(query) > 500:
        raise CoarchValidationError(
            f"Query is too long ({len(query)} characters)",
            field="query",
            value=query,
            suggestion="Use a shorter query (max 500 characters)"
        )