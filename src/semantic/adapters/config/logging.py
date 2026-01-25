"""Structured logging configuration."""

import logging
import sys
from collections.abc import Callable, Sequence
from typing import Any, cast

import structlog
from structlog.typing import EventDict, WrappedLogger


def configure_logging(log_level: str = "INFO", json_output: bool = True) -> None:
    """Configure structured logging."""
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Configure structlog
    processors: Sequence[Callable[[WrappedLogger, str, EventDict], Any]]
    if json_output:
        # Production: JSON output
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Colored console output
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    if name:
        return cast(structlog.BoundLogger, structlog.get_logger(name))
    return cast(structlog.BoundLogger, structlog.get_logger())
