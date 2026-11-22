"""Per-host cached robots.txt policy checker (async, non-blocking)."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import USER_AGENT

logger = logging.getLogger(__name__)


def _origin_for(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


class RobotsCache:
    """Fetch and cache ``/robots.txt`` once per origin using async HTTP."""

    def __init__(self, user_agent: str = USER_AGENT) -> None:
        self._user_agent = user_agent
        self._parsers: dict[str, RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, origin: str) -> asyncio.Lock:
        lock = self._locks.get(origin)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[origin] = lock
        return lock

    async def _ensure_loaded(
        self,
        client: httpx.AsyncClient,
        origin: str,
        *,
        timeout: float = 5.0,
    ) -> RobotFileParser | None:
        if origin in self._parsers:
            return self._parsers[origin]

        async with self._lock_for(origin):
            if origin in self._parsers:
                return self._parsers[origin]

            robots_url = f"{origin}/robots.txt"
            parser: RobotFileParser | None = None
            try:
                async with asyncio.timeout(timeout):
                    resp = await client.get(robots_url)
                if 200 <= resp.status_code < 300:
                    rp = RobotFileParser()
                    rp.set_url(robots_url)
                    rp.parse(resp.text.splitlines())
                    parser = rp
                    logger.info("Loaded robots.txt for %s", origin)
                elif resp.status_code in {401, 403}:
                    rp = RobotFileParser()
                    rp.set_url(robots_url)
                    rp.parse(["User-agent: *", "Disallow: /"])
                    parser = rp
                    logger.info(
                        "robots.txt returned %d for %s (disallow all)",
                        resp.status_code,
                        origin,
                    )
                else:
                    logger.info(
                        "robots.txt returned %d for %s (allow all)",
                        resp.status_code,
                        origin,
                    )
                    parser = None
            except (TimeoutError, httpx.HTTPError) as exc:
                logger.warning("Could not fetch %s (%s); defaulting to allow", robots_url, exc)
                parser = None

            self._parsers[origin] = parser
            return parser

    async def allowed(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        timeout: float = 5.0,
    ) -> bool:
        """Return True if crawling ``url`` is permitted by the host's robots.txt."""
        origin = _origin_for(url)
        parser = await self._ensure_loaded(client, origin, timeout=timeout)
        if parser is None:
            return True
        return parser.can_fetch(self._user_agent, url)

    def crawl_delay(self, url: str) -> float | None:
        """Return the Crawl-delay directive (seconds) if present for ``url``'s origin."""
        origin = _origin_for(url)
        parser = self._parsers.get(origin)
        if parser is None:
            return None
        raw = parser.crawl_delay(self._user_agent)
        if raw is None:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None
