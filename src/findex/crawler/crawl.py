"""Async frontier web crawler using asyncio.Queue, TaskGroup, and per-host politeness."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from findex.crawler.fetch import USER_AGENT, FetchResult, SleepFn, fetch
from findex.crawler.parse import canonicalize_url, extract_page_data
from findex.crawler.robots import RobotsCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Page:
    """Single crawled webpage yielded as a stream item."""

    url: str
    title: str
    text: str
    fetched_at: str


@dataclass(slots=True)
class CrawlStats:
    """Live mutable counters updated during an async crawl."""

    pages_fetched: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    bytes_downloaded: int = 0
    skipped_robots: int = 0
    skipped_domain: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_seconds(self) -> float:
        return max(1e-6, time.perf_counter() - self.started_at)

    @property
    def pages_per_second(self) -> float:
        return self.pages_fetched / self.elapsed_seconds


def _host_key(url: str) -> str:
    return urlsplit(url).netloc.lower()


def _hostname(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _is_domain_allowed(url: str, allowed_domains: frozenset[str]) -> bool:
    if not allowed_domains:
        return True
    netloc = _host_key(url)
    host = _hostname(url)
    for allowed in allowed_domains:
        if netloc == allowed or host == allowed or host.endswith(f".{allowed}"):
            return True
    return False


class _HostPoliteness:
    """Enforce per-host concurrency semaphore and minimum inter-request delay."""

    def __init__(self, per_host: int, min_delay: float, sleep_fn: SleepFn) -> None:
        self._per_host = max(1, per_host)
        self._min_delay = max(0.0, min_delay)
        self._sleep = sleep_fn
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._rate_locks: dict[str, asyncio.Lock] = {}
        self._next_allowed: dict[str, float] = {}

    def semaphore(self, host: str) -> asyncio.Semaphore:
        sem = self._sems.get(host)
        if sem is None:
            sem = asyncio.Semaphore(self._per_host)
            self._sems[host] = sem
        return sem

    async def wait_turn(self, host: str, extra_delay: float | None = None) -> None:
        delay = max(self._min_delay, extra_delay or 0.0)
        if delay <= 0.0:
            return
        lock = self._rate_locks.get(host)
        if lock is None:
            lock = asyncio.Lock()
            self._rate_locks[host] = lock

        wait_for = 0.0
        async with lock:
            now = time.monotonic()
            scheduled = self._next_allowed.get(host, now)
            if scheduled > now:
                wait_for = scheduled - now
                self._next_allowed[host] = scheduled + delay
            else:
                self._next_allowed[host] = now + delay
        if wait_for > 0.0:
            await self._sleep(wait_for)


def _format_log_line(res: FetchResult, note: str = "") -> str:
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    status = str(res.status_code) if res.status_code is not None else "ERR"
    detail = note or (res.error or "OK")
    return (
        f"{ts}\t{status}\t{res.num_bytes}B\t{res.elapsed_ms:.1f}ms"
        f"\tattempts={res.attempts}\t{res.url}\t{detail}"
    )


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 200,
    concurrency: int = 10,
    per_host: int = 2,
    min_delay: float = 0.05,
    allowed_domains: Iterable[str] | None = None,
    timeout: float = 10.0,
    user_agent: str = USER_AGENT,
    client: httpx.AsyncClient | None = None,
    stats: CrawlStats | None = None,
    on_log: Callable[[str], None] | None = None,
    respect_robots: bool = True,
    max_retries: int = 3,
    base_retry_delay: float = 0.5,
    sleep_fn: SleepFn = asyncio.sleep,
) -> AsyncIterator[Page]:
    """Stream crawled ``Page`` objects using a bounded ``Queue`` frontier and ``TaskGroup``."""
    if max_pages <= 0:
        return

    live_stats = stats if stats is not None else CrawlStats()
    live_stats.started_at = time.perf_counter()

    canonical_seeds: list[str] = []
    for raw_seed in seeds:
        canon = canonicalize_url(raw_seed)
        if canon is not None:
            canonical_seeds.append(canon)
    if not canonical_seeds:
        return

    if allowed_domains is None:
        domain_set = frozenset(_hostname(s) for s in canonical_seeds if _hostname(s))
    else:
        domain_set = frozenset(d.strip().lower() for d in allowed_domains if d.strip())

    frontier: asyncio.Queue[str] = asyncio.Queue()
    output_queue: asyncio.Queue[Page | None] = asyncio.Queue()
    seen: set[str] = set()
    stop_event = asyncio.Event()
    claimed_pages = 0

    for seed in canonical_seeds:
        if seed not in seen:
            seen.add(seed)
            frontier.put_nowait(seed)
    live_stats.queue_depth = frontier.qsize()

    global_sem = asyncio.Semaphore(max(1, concurrency))
    politeness = _HostPoliteness(per_host=per_host, min_delay=min_delay, sleep_fn=sleep_fn)
    robots = RobotsCache(user_agent=user_agent)

    owns_client = client is None
    active_client = (
        httpx.AsyncClient(
            headers={"User-Agent": user_agent},
            follow_redirects=True,
            max_redirects=10,
        )
        if client is None
        else client
    )

    def _emit_log(line: str) -> None:
        if on_log is not None:
            on_log(line)

    async def _worker() -> None:
        nonlocal claimed_pages
        while not stop_event.is_set():
            url = await frontier.get()
            live_stats.queue_depth = frontier.qsize()
            try:
                if stop_event.is_set():
                    continue

                if not _is_domain_allowed(url, domain_set):
                    live_stats.skipped_domain += 1
                    continue

                if respect_robots:
                    is_ok = await robots.allowed(active_client, url, timeout=min(timeout, 5.0))
                    if not is_ok:
                        live_stats.skipped_robots += 1
                        ts = datetime.now(UTC).isoformat(timespec="seconds")
                        _emit_log(f"{ts}\tROBOTS\t0B\t0.0ms\tattempts=0\t{url}\tdisallowed")
                        continue

                if stop_event.is_set():
                    continue

                host = _host_key(url)
                robots_delay = robots.crawl_delay(url) if respect_robots else None

                async with global_sem, politeness.semaphore(host):
                    if stop_event.is_set():
                        continue
                    await politeness.wait_turn(host, extra_delay=robots_delay)
                    if stop_event.is_set():
                        continue

                    live_stats.in_flight += 1
                    try:
                        res = await fetch(
                            active_client,
                            url,
                            timeout=timeout,
                            max_retries=max_retries,
                            base_delay=base_retry_delay,
                            sleep_fn=sleep_fn,
                        )
                    finally:
                        live_stats.in_flight -= 1

                if not res.ok:
                    live_stats.errors += 1
                    _emit_log(_format_log_line(res))
                    continue

                # Claim a page slot atomically between awaits (no lock needed in single-thread loop)
                if claimed_pages >= max_pages:
                    stop_event.set()
                    continue
                claimed_pages += 1
                live_stats.pages_fetched = claimed_pages
                live_stats.bytes_downloaded += res.num_bytes
                _emit_log(_format_log_line(res, note="OK"))

                final_canon = canonicalize_url(res.final_url)
                if final_canon is not None:
                    seen.add(final_canon)

                is_html = (
                    "html" in res.content_type.lower()
                    or res.content_type == ""
                    or "<html" in res.body[:256].lower()
                )
                if is_html:
                    title, text, links = extract_page_data(res.body, base_url=res.final_url)
                else:
                    title = res.final_url
                    text = res.body
                    links = []

                if claimed_pages < max_pages and not stop_event.is_set():
                    for link in links:
                        if link not in seen and _is_domain_allowed(link, domain_set):
                            seen.add(link)
                            frontier.put_nowait(link)
                    live_stats.queue_depth = frontier.qsize()

                page = Page(
                    url=final_canon or url,
                    title=title or url,
                    text=text,
                    fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
                )
                output_queue.put_nowait(page)

                if claimed_pages >= max_pages:
                    stop_event.set()
            finally:
                frontier.task_done()
                live_stats.queue_depth = frontier.qsize()

    async def _coordinator(workers: list[asyncio.Task[None]]) -> None:
        async def _wait_stop() -> None:
            await stop_event.wait()

        join_task: asyncio.Task[None] = asyncio.create_task(frontier.join())
        stop_task: asyncio.Task[None] = asyncio.create_task(_wait_stop())
        try:
            await asyncio.wait(
                [join_task, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            stop_event.set()
        finally:
            for helper in (join_task, stop_task):
                if not helper.done():
                    helper.cancel()
            for w in workers:
                if not w.done():
                    w.cancel()
            output_queue.put_nowait(None)

    worker_count = max(1, concurrency)
    try:
        async with asyncio.TaskGroup() as tg:
            workers = [tg.create_task(_worker()) for _ in range(worker_count)]
            coord = tg.create_task(_coordinator(workers))
            try:
                while True:
                    item = await output_queue.get()
                    if item is None:
                        break
                    yield item
            finally:
                stop_event.set()
                for w in workers:
                    if not w.done():
                        w.cancel()
                if not coord.done():
                    coord.cancel()
    finally:
        if owns_client:
            await active_client.aclose()
