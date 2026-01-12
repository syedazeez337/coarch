"""Logging configuration for Coarch."""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import json


LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        extra_data = getattr(record, "extra_data", None)
        if extra_data:
            log_entry["extra"] = extra_data

        return json.dumps(log_entry)


class PlainFormatter(logging.Formatter):
    """Human-readable formatter."""

    def __init__(self, fmt: str = LOG_FORMAT, datefmt: str = LOG_DATE_FORMAT):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.WARNING:
            color = "\033[93m" if record.levelno == logging.WARNING else "\033[91m"
            reset = "\033[0m"
        else:
            color = ""
            reset = ""

        message = super().format(record)
        return f"{color}{message}{reset}"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    verbose: bool = False,
) -> logging.Logger:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        json_format: Use JSON formatting for structured logs
        verbose: Enable verbose (DEBUG) logging
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if verbose:
        numeric_level = logging.DEBUG

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    root_logger.handlers.clear()

    if json_format:
        formatter: Union[JSONFormatter, PlainFormatter] = JSONFormatter(
            LOG_FORMAT, LOG_DATE_FORMAT
        )
    else:
        formatter = PlainFormatter(LOG_FORMAT, LOG_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = False

    logger = logging.getLogger("coarch")
    logger.setLevel(numeric_level)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the coarch namespace."""
    return logging.getLogger(f"coarch.{name}")


class LoggerMixin:
    """Mixin class to add logging capability."""

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_function_call(func):
    """Decorator to log function calls."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.exception(f"{func.__name__} failed: {e}")
            raise

    return wrapper
