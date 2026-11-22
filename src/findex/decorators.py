"""Timing and utility decorators (Lab 3 M4)."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

timing_logger = logging.getLogger("findex.timing")


def timed(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Log execution time of ``fn`` in milliseconds while preserving metadata."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timing_logger.info("%s took %.2f ms", fn.__name__, elapsed_ms)

    return wrapper
