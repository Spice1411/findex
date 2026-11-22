"""Unit & integration tests for Lab 6 async crawler using httpx.MockTransport (no network)."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path

import httpx
import pytest

from findex.corpus import iter_documents
from findex.crawler.crawl import CrawlStats, Page, crawl
from findex.crawler.fetch import USER_AGENT, fetch, parse_retry_after
from findex.crawler.parse import canonicalize_url, extract_page_data
from findex.crawler.robots import RobotsCache
from findex.index import build_index
from findex.search import search


@pytest.mark.parametrize(
    ("raw", "base", "expected"),
    [
        ("HTTPS://Example.COM:443/Docs/#intro", None, "https://example.com/Docs"),
        ("http://example.com:80/", None, "http://example.com/"),
        ("http://example.com/a/b/?z=1&a=2#frag", None, "http://example.com/a/b?a=2&z=1"),
        (
            "../guide/?b=2&a=1",
            "https://example.com/docs/intro/",
            "https://example.com/docs/guide?a=1&b=2",
        ),
        ("mailto:test@example.com", "https://example.com/", None),
        ("javascript:void(0)", "https://example.com/", None),
    ],
)
def test_canonicalize_url(raw: str, base: str | None, expected: str | None) -> None:
    assert canonicalize_url(raw, base_url=base) == expected


def test_extract_page_data_skips_scripts_and_resolves_links() -> None:
    html = """
    <html>
      <head>
        <title>Async  Crawler</title>
        <style>.hidden { display: none; }</style>
      </head>
      <body>
        <h1>Hello World</h1>
        <script>var ignored = 123;</script>
        <p>Polite <a href="/docs/?b=2&a=1#top">Docs</a> and <a href="mailto:x@y">Mail</a></p>
      </body>
    </html>
    """
    title, text, links = extract_page_data(html, base_url="https://example.com/start")
    assert title == "Async Crawler"
    assert "Hello World" in text
    assert "ignored" not in text
    assert "display" not in text
    assert links == ["https://example.com/docs?a=1&b=2"]


def test_parse_retry_after_seconds_and_invalid() -> None:
    assert parse_retry_after("2.5") == 2.5
    assert parse_retry_after("999", cap_seconds=30.0) == 30.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("not-a-date") is None


def test_fetch_retries_429_with_retry_after_and_sends_honest_ua() -> None:
    calls = 0
    seen_ua = ""
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls, seen_ua
        calls += 1
        seen_ua = request.headers.get("user-agent", "")
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "1.25"}, text="slow down")
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<p>ok</p>")

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, headers={"User-Agent": USER_AGENT}
        ) as client:
            res = await fetch(client, "https://example.com/a", sleep_fn=fake_sleep)
            assert res.ok
            assert res.status_code == 200
            assert res.attempts == 2
            assert res.num_bytes == len(b"<p>ok</p>")
            assert res.elapsed_ms >= 0.0

    asyncio.run(run())
    assert seen_ua == USER_AGENT
    assert sleeps == [1.25]


def test_fetch_does_not_retry_404_but_retries_5xx_up_to_limit() -> None:
    counts = {"404": 0, "503": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/missing":
            counts["404"] += 1
            return httpx.Response(404, text="not found")
        counts["503"] += 1
        return httpx.Response(503, text="service unavailable")

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            r404 = await fetch(
                client, "https://example.com/missing", max_retries=3, sleep_fn=fake_sleep
            )
            assert not r404.ok
            assert r404.status_code == 404
            assert r404.attempts == 1

            r503 = await fetch(
                client, "https://example.com/down", max_retries=3, sleep_fn=fake_sleep
            )
            assert not r503.ok
            assert r503.status_code == 503
            assert r503.attempts == 4

    asyncio.run(run())
    assert counts["404"] == 1
    assert counts["503"] == 4
    assert len(sleeps) == 3


def test_fetch_timeout_and_redirect_loop() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/slow":
            await asyncio.sleep(1.0)
            return httpx.Response(200, text="late")
        raise httpx.TooManyRedirects("redirect loop", request=request)

    async def no_sleep(_delay: float) -> None:
        return None

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            r_slow = await fetch(
                client,
                "https://example.com/slow",
                timeout=0.03,
                max_retries=1,
                sleep_fn=no_sleep,
            )
            assert not r_slow.ok
            assert r_slow.attempts == 2
            assert r_slow.error is not None and "timeout" in r_slow.error

            r_loop = await fetch(
                client,
                "https://example.com/loop",
                max_retries=3,
                sleep_fn=no_sleep,
            )
            assert not r_loop.ok
            assert r_loop.attempts == 1
            assert r_loop.error is not None and "redirect loop" in r_loop.error

    asyncio.run(run())


def test_robots_cache_fetches_once_per_host_and_enforces_rules() -> None:
    robots_hits = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal robots_hits
        if request.url.path == "/robots.txt":
            robots_hits += 1
            return httpx.Response(
                200,
                text="User-agent: *\nDisallow: /private/\nCrawl-delay: 2\n",
            )
        return httpx.Response(404)

    async def run() -> None:
        cache = RobotsCache(user_agent=USER_AGENT)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await cache.allowed(client, "https://example.com/public") is True
            assert await cache.allowed(client, "https://example.com/private/secret") is False
            assert await cache.allowed(client, "https://example.com/another") is True
            assert cache.crawl_delay("https://example.com/public") == 2.0

    asyncio.run(run())
    assert robots_hits == 1


def test_crawl_frontier_dedup_robots_max_pages_and_indexing_pipeline(tmp_path: Path) -> None:
    pages_map = {
        "/": (
            "<title>Home</title>"
            "<a href='/p1#top'>P1</a>"
            "<a href='/p1/'>P1 dup</a>"
            "<a href='/p2?b=2&a=1'>P2</a>"
            "<a href='/private/x'>Secret</a>"
            "<a href='https://other.org/ext'>External</a>"
        ),
        "/p1": (
            "<title>Async Guide</title><p>Event loop and asyncio coroutines</p><a href='/p3'>P3</a>"
        ),
        "/p2": "<title>BM25 Guide</title><p>Inverted index ranking</p><a href='/p4'>P4</a>",
        "/p3": "<title>TaskGroup</title><p>Structured concurrency</p>",
        "/p4": "<title>Extra</title><p>Beyond max pages</p>",
    }
    fetched_urls: list[str] = []
    max_observed_in_flight = 0
    current_in_flight = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal current_in_flight, max_observed_in_flight
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        current_in_flight += 1
        max_observed_in_flight = max(max_observed_in_flight, current_in_flight)
        try:
            await asyncio.sleep(0.01)
            fetched_urls.append(str(request.url))
            html = pages_map.get(request.url.path)
            if html is None:
                return httpx.Response(404, text="missing")
            return httpx.Response(200, headers={"content-type": "text/html"}, text=html)
        finally:
            current_in_flight -= 1

    stats = CrawlStats()
    logs: list[str] = []
    collected: list[Page] = []

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async for page in crawl(
                ["https://example.com/"],
                max_pages=3,
                concurrency=4,
                per_host=2,
                min_delay=0.0,
                client=client,
                stats=stats,
                on_log=logs.append,
            ):
                collected.append(page)

    # Also verify debug=True has zero slow-callback warnings
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    aio_log = logging.getLogger("asyncio")
    aio_log.addHandler(h)
    try:
        asyncio.run(run(), debug=True)
    finally:
        aio_log.removeHandler(h)

    assert len(collected) == 3
    assert stats.pages_fetched == 3
    assert stats.skipped_robots >= 1
    assert max_observed_in_flight <= 2  # per_host=2 respected
    assert all("other.org" not in u for u in fetched_urls)
    assert all("/private/" not in u for u in fetched_urls)
    assert len(fetched_urls) == len(set(fetched_urls))  # no duplicate fetches
    assert "took" not in buf.getvalue()

    # Stream to JSONL -> iter_documents (Lab 1) -> build_index -> search
    jsonl_path = tmp_path / "crawl.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as out:
        for p in collected:
            out.write(json.dumps({"id": p.url, "text": p.text}) + "\n")

    docs = list(iter_documents(jsonl_path))
    assert len(docs) == 3
    idx = build_index(jsonl_path, with_positions=True)
    hits = search(idx, 'asyncio AND "event loop"', k=5)
    assert len(hits) >= 1
