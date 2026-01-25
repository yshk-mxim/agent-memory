"""Structured logging configuration (NEW-6).

Replaces print statements with JSON-formatted structured logs.
"""

import logging
import sys

import structlog


def configure_logging(log_level: str = "INFO", json_output: bool = True) -> None:
    """Configure structured logging with JSON output (NEW-6).

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output JSON format; if False, console format

    Example:
        >>> configure_logging("INFO", json_output=True)
        >>> logger = structlog.get_logger()
        >>> logger.info("model_loaded", model_id="gemma-3-12b", size_mb=4096)
        {"event": "model_loaded", "model_id": "gemma-3-12b", "size_mb": 4096, "timestamp": "2026-01-29T10:30:00Z"}
    """
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Configure structlog
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
    """Get a structured logger instance.

    Args:
        name: Optional logger name (e.g., __name__)

    Returns:
        Structured logger with context binding

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("cache_hit", agent_id="agent_1", tokens=1024)
        {"event": "cache_hit", "agent_id": "agent_1", "tokens": 1024, "logger": "semantic.application"}
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


# Example usage patterns for replacing print statements:
#
# Before (print):
#   print(f"Loading model {model_id}...")
#
# After (structlog):
#   logger.info("model_loading", model_id=model_id)
#
# Before (print):
#   print(f"Cache hit: {agent_id} ({tokens} tokens)")
#
# After (structlog):
#   logger.info("cache_hit", agent_id=agent_id, tokens=tokens)
#
# Before (print debug):
#   if debug:
#       print(f"Block allocation: {n_blocks} blocks for layer {layer_id}")
#
# After (structlog):
#   logger.debug("block_allocation", n_blocks=n_blocks, layer_id=layer_id)
