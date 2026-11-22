"""Timing and utility decorators (Lab 3 M4 & Lab 4 M4)."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable

timing_logger = logging.getLogger("findex.timing")


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Log execution time of ``fn`` in milliseconds at DEBUG level."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timing_logger.debug("%s took %.2f ms", fn.__name__, elapsed_ms)

    return wrapper
