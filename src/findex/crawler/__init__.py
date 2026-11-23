"""Polite async web crawler built with asyncio and httpx (Lab 6)."""

from findex.crawler.crawl import CrawlStats, Page, crawl
from findex.crawler.fetch import USER_AGENT, FetchResult, fetch, parse_retry_after
from findex.crawler.parse import canonicalize_url, extract_page_data
from findex.crawler.robots import RobotsCache

__all__ = [
    "USER_AGENT",
    "CrawlStats",
    "FetchResult",
    "Page",
    "RobotsCache",
    "canonicalize_url",
    "crawl",
    "extract_page_data",
    "fetch",
    "parse_retry_after",
]
