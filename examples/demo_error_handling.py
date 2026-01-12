#!/usr/bin/env python3
"""Demo script to test the new error handling system."""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '.')

from backend.exceptions import (
    CoarchCLIError,
    CoarchConfigError,
    CoarchIndexError,
    CoarchSearchError,
    CoarchValidationError,
    validate_port,
    validate_limit,
    validate_query,
    handle_cli_error,
)


def demo_validation_errors():
    """Demonstrate validation errors."""
    print("=== Testing Validation Errors ===\n")

    # Test invalid port
    try:
        validate_port(70000)
    except CoarchValidationError as e:
        print(f"Port validation error:\n{e}\n")

    # Test invalid limit
    try:
        validate_limit(-5)
    except CoarchValidationError as e:
        print(f"Limit validation error:\n{e}\n")

    # Test empty query
    try:
        validate_query("")
    except CoarchValidationError as e:
        print(f"Query validation error:\n{e}\n")


def demo_contextual_errors():
    """Demonstrate errors with context."""
    print("=== Testing Contextual Errors ===\n")

    # Test configuration error with context
    try:
        raise CoarchConfigError(
            "Configuration file not found",
            config_path="/tmp/coarch/config",
            suggestion="Run 'coarch init' to create a default configuration"
        )
    except CoarchCLIError as e:
        print(f"Configuration error:\n{e}\n")

    # Test indexing error with context
    try:
        raise CoarchIndexError(
            "Failed to read repository files",
            repo_path="/nonexistent/repo",
            files_processed=0,
            suggestion="Check that the repository exists and you have read permissions"
        )
    except CoarchCLIError as e:
        print(f"Indexing error:\n{e}\n")

    # Test search error with context
    try:
        raise CoarchSearchError(
            "Index database not found",
            query="function hello",
            index_path="/tmp/coarch/index",
            suggestion="Run 'coarch index <path>' first to create an index"
        )
    except CoarchCLIError as e:
        print(f"Search error:\n{e}\n")


def demo_exit_codes():
    """Demonstrate different exit codes."""
    print("=== Testing Exit Codes ===\n")

    errors = [
        (CoarchConfigError("Config error"), "Configuration Error"),
        (CoarchIndexError("Index error"), "Indexing Error"),
        (CoarchSearchError("Search error"), "Search Error"),
        (CoarchValidationError("Validation error"), "Validation Error"),
        (CoarchCLIError("Generic error"), "Generic Error"),
    ]

    for error, name in errors:
        print(f"{name}: Exit code {error.exit_code.value}")


def demo_error_handler():
    """Demonstrate the error handler."""
    print("=== Testing Error Handler ===\n")

    # Test with custom error
    try:
        raise CoarchValidationError(
            "Invalid parameter value",
            field="port",
            value="not-a-number"
        )
    except Exception as e:
        print("Handling custom error:")
        handle_cli_error(e, verbose=False)
        print("This won't print because we exited")


if __name__ == "__main__":
    print("Coarch CLI Error Handling Demo\n")

    demo_validation_errors()
    demo_contextual_errors()
    demo_exit_codes()
    demo_error_handler()

    print("Demo completed successfully!")