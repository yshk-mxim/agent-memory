"""Timeout mechanisms for model and cache operations (NEW-3).

Prevents hangs in model loading, cache loading, and extraction.
"""

import functools
import signal
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class TimeoutError(Exception):
    """Raised when an operation exceeds its timeout."""


def timeout(seconds: int) -> Callable[[F], F]:
    """Decorator to add timeout to a function (NEW-3).

    Args:
        seconds: Timeout in seconds

    Returns:
        Decorated function that raises TimeoutError if exceeded

    Example:
        >>> @timeout(30)
        ... def load_model(path):
        ...     # Long operation
        ...     ...

    Notes:
        - Uses SIGALRM signal (Unix/macOS only)
        - Not thread-safe (signal handlers are process-global)
        - For production, consider using multiprocessing.Pool with timeout

    Raises:
        TimeoutError: If function exceeds timeout
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            def _timeout_handler(signum: int, frame: Any) -> None:
                raise TimeoutError(
                    f"{func.__name__}() exceeded timeout of {seconds}s"
                )

            # Set alarm
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# Example usage decorators for common operations
def with_model_load_timeout(seconds: int = 300) -> Callable[[F], F]:
    """Timeout for model loading operations (default 5 min).

    Example:
        >>> @with_model_load_timeout(300)
        ... def load_model(path):
        ...     return mlx_lm.load(path)
    """
    return timeout(seconds)


def with_cache_operation_timeout(seconds: int = 30) -> Callable[[F], F]:
    """Timeout for cache operations (default 30s).

    Example:
        >>> @with_cache_operation_timeout(30)
        ... def extract_cache(uid):
        ...     return batch_gen.extract_cache(uid)
    """
    return timeout(seconds)
