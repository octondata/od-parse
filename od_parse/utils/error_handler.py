"""
Error handling utilities for od-parse.

This module provides consistent error handling and logging across the library.
"""

import functools
import logging
import traceback
from typing import Any, Callable, Type, TypeVar, cast

from ..exceptions import ODParseError, ProcessingError

F = TypeVar("F", bound=Callable[..., Any])


def handle_errors(
    error_map: dict[Type[Exception], Type[ODParseError]] = None,
    default_error: Type[ODParseError] = ProcessingError,
    log_level: int = logging.ERROR,
) -> Callable[[F], F]:
    """
    Decorator for consistent error handling across the library.

    Args:
        error_map: Mapping of source exceptions to od-parse exceptions
        default_error: Default exception type to use if not in error_map
        log_level: Logging level for errors

    Example:
        >>> @handle_errors({
        ...     ValueError: ConfigurationError,
        ...     IOError: FileError
        ... })
        ... def process_file(path: str) -> dict:
        ...     # Function implementation
        ...     pass
    """
    if error_map is None:
        error_map = {}

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except ODParseError:
                # Already a library exception, re-raise
                raise
            except Exception as e:
                logger = logging.getLogger(func.__module__)

                # Get the error type from the map or use default
                error_type = error_map.get(type(e), default_error)

                # Log the error with stack trace
                logger.log(
                    log_level,
                    f"Error in {func.__name__}: {str(e)}",
                    extra={
                        "function": func.__name__,
                        "error_type": error_type.__name__,
                        "original_error": str(e),
                        "stack_trace": traceback.format_exc(),
                    },
                )

                # Raise as library exception
                raise error_type(str(e)) from e

        return cast(F, wrapper)

    return decorator


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: dict[str, Any] = None,
    level: int = logging.ERROR,
) -> None:
    """
    Log an error with consistent formatting and context.

    Args:
        logger: Logger instance to use
        error: Exception to log
        context: Additional context to include in log
        level: Logging level to use
    """
    if context is None:
        context = {}

    error_type = type(error).__name__
    message = str(error)
    stack_trace = traceback.format_exc()

    logger.log(
        level,
        f"{error_type}: {message}",
        extra={
            "error_type": error_type,
            "message": message,
            "stack_trace": stack_trace,
            **context,
        },
    )
