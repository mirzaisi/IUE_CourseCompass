"""
Logging Module - Structured logging setup with Rich console support.
====================================================================

Provides centralized logging configuration for the entire application.
Supports Rich console output for beautiful terminal logs and optional
file logging for persistence.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

# Global state for logging configuration
_logging_configured = False
_console = Console()


def setup_logging(
    level: str = "INFO",
    use_rich: bool = True,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_rich: Whether to use Rich console handler for pretty output
        log_file: Optional path to log file
        log_format: Optional custom log format string

    Note:
        This function should be called once at application startup.
        Subsequent calls will be ignored to prevent duplicate handlers.
    """
    global _logging_configured

    if _logging_configured:
        return

    # Parse log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Default format
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers
    root_logger.handlers.clear()

    # Add Rich console handler or standard stream handler
    if use_rich:
        rich_handler = RichHandler(
            console=_console,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
        )
        rich_handler.setLevel(numeric_level)
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(rich_handler)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(numeric_level)
        stream_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(stream_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    _logging_configured = True

    # Log startup message
    logger = get_logger(__name__)
    logger.debug(f"Logging configured: level={level}, rich={use_rich}, file={log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    # Ensure logging is configured with defaults if not already done
    if not _logging_configured:
        setup_logging()

    return logging.getLogger(name)


def get_console() -> Console:
    """
    Get the Rich console instance for direct console output.

    Returns:
        Rich Console instance

    Example:
        >>> console = get_console()
        >>> console.print("[bold green]Success![/bold green]")
    """
    return _console


class LogContext:
    """
    Context manager for temporary log level changes.

    Example:
        >>> with LogContext("DEBUG"):
        ...     logger.debug("This will be shown")
        >>> logger.debug("This might not be shown")
    """

    def __init__(self, level: str, logger_name: Optional[str] = None):
        """
        Initialize log context.

        Args:
            level: Temporary log level
            logger_name: Specific logger to modify (None = root logger)
        """
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.logger_name = logger_name
        self.original_level: Optional[int] = None

    def __enter__(self) -> "LogContext":
        logger = logging.getLogger(self.logger_name)
        self.original_level = logger.level
        logger.setLevel(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        if self.original_level is not None:
            logger = logging.getLogger(self.logger_name)
            logger.setLevel(self.original_level)
