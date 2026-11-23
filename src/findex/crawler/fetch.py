"""Polite async HTTP fetcher with asyncio.timeout, retries, backoff+jitter, and Retry-After."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Final

import httpx

logger = logging.getLogger(__name__)

USER_AGENT: Final[str] = "findex-crawler/0.6 (+https://github.com/Spice1411/findex)"

SleepFn = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Outcome of fetching a single URL (status, payload, bytes, elapsed time, attempts)."""

    url: str
    final_url: str
    status_code: int | None
    content_type: str
    body: str
    num_bytes: int
    elapsed_ms: float
    attempts: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status_code is not None and 200 <= self.status_code < 300


def parse_retry_after(header_value: str | None, *, cap_seconds: float = 60.0) -> float | None:
    """Parse HTTP ``Retry-After`` header (delta-seconds or HTTP-date) into seconds."""
    if not header_value:
        return None
    raw = header_value.strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
        return max(0.0, min(cap_seconds, seconds))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = (dt - datetime.now(UTC)).total_seconds()
        return max(0.0, min(cap_seconds, delta))
    except (TypeError, ValueError, OverflowError):
        return None


def _compute_backoff(
    attempt: int,
    *,
    base_delay: float,
    max_delay: float,
    retry_after: float | None = None,
) -> float:
    """Exponential backoff with jitter, honoring explicit Retry-After when present."""
    if retry_after is not None:
        return retry_after
    exp_delay = min(max_delay, base_delay * (2**attempt))
    jitter = random.uniform(0.0, min(1.0, base_delay))
    return exp_delay + jitter


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    sleep_fn: SleepFn = asyncio.sleep,
) -> FetchResult:
    """Fetch ``url`` with ``asyncio.timeout`` and bounded retries on 429, 5xx, and timeouts.

    Never retries non-429 4xx responses or redirect loops.
    """
    t_start = time.perf_counter()
    total_attempts = 0
    last_status: int | None = None
    last_error: str | None = None
    last_final_url = url

    for attempt in range(max_retries + 1):
        total_attempts = attempt + 1
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url)
        except TimeoutError:
            last_error = f"timeout after {timeout:.1f}s"
            logger.warning(
                "Timeout fetching %s (attempt %d/%d)",
                url,
                total_attempts,
                max_retries + 1,
            )
            if attempt < max_retries:
                delay = _compute_backoff(attempt, base_delay=base_delay, max_delay=max_delay)
                await sleep_fn(delay)
                continue
            break
        except httpx.TimeoutException as exc:
            last_error = f"timeout: {exc}"
            logger.warning(
                "HTTPX timeout fetching %s (attempt %d/%d): %s",
                url,
                total_attempts,
                max_retries + 1,
                exc,
            )
            if attempt < max_retries:
                delay = _compute_backoff(attempt, base_delay=base_delay, max_delay=max_delay)
                await sleep_fn(delay)
                continue
            break
        except httpx.TooManyRedirects as exc:
            last_error = f"redirect loop: {exc}"
            logger.warning("Redirect loop on %s (not retrying): %s", url, exc)
            break
        except httpx.TransportError as exc:
            last_error = f"transport error: {exc}"
            logger.warning(
                "Transport error fetching %s (attempt %d/%d): %s",
                url,
                total_attempts,
                max_retries + 1,
                exc,
            )
            if attempt < max_retries:
                delay = _compute_backoff(attempt, base_delay=base_delay, max_delay=max_delay)
                await sleep_fn(delay)
                continue
            break
        except httpx.HTTPError as exc:
            last_error = f"http error: {exc}"
            logger.warning("HTTP error fetching %s (not retrying): %s", url, exc)
            break

        status = resp.status_code
        last_status = status
        last_final_url = str(resp.url)
        raw_bytes = resp.content
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        content_type = resp.headers.get("content-type", "")

        # Retryable statuses: 429 Too Many Requests and 5xx Server Errors
        if status == 429 or 500 <= status < 600:
            retry_after = parse_retry_after(resp.headers.get("retry-after"))
            last_error = f"HTTP {status}"
            logger.warning(
                "Retryable HTTP %d for %s (attempt %d/%d, retry_after=%s)",
                status,
                url,
                total_attempts,
                max_retries + 1,
                retry_after,
            )
            if attempt < max_retries:
                delay = _compute_backoff(
                    attempt,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    retry_after=retry_after,
                )
                await sleep_fn(delay)
                continue
            return FetchResult(
                url=url,
                final_url=last_final_url,
                status_code=status,
                content_type=content_type,
                body="",
                num_bytes=len(raw_bytes),
                elapsed_ms=elapsed_ms,
                attempts=total_attempts,
                error=last_error,
            )

        # Non-retryable 4xx (e.g. 404, 403, 410)
        if 400 <= status < 500:
            logger.info(
                "Non-retryable HTTP %d for %s (%d bytes, %.1f ms)",
                status,
                url,
                len(raw_bytes),
                elapsed_ms,
            )
            return FetchResult(
                url=url,
                final_url=last_final_url,
                status_code=status,
                content_type=content_type,
                body="",
                num_bytes=len(raw_bytes),
                elapsed_ms=elapsed_ms,
                attempts=total_attempts,
                error=f"HTTP {status}",
            )

        logger.debug(
            "Fetched %s -> %d (%d bytes, %.1f ms, attempts=%d)",
            url,
            status,
            len(raw_bytes),
            elapsed_ms,
            total_attempts,
        )
        return FetchResult(
            url=url,
            final_url=last_final_url,
            status_code=status,
            content_type=content_type,
            body=resp.text,
            num_bytes=len(raw_bytes),
            elapsed_ms=elapsed_ms,
            attempts=total_attempts,
            error=None,
        )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return FetchResult(
        url=url,
        final_url=last_final_url,
        status_code=last_status,
        content_type="",
        body="",
        num_bytes=0,
        elapsed_ms=elapsed_ms,
        attempts=total_attempts,
        error=last_error or "fetch failed",
    )
