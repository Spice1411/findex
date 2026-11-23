"""URL canonicalization and HTML text/link extraction using stdlib html.parser."""

from __future__ import annotations

import posixpath
import re
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

_SKIP_TAGS = frozenset({"script", "style", "noscript", "svg"})


def canonicalize_url(url: str, base_url: str | None = None) -> str | None:
    """Normalize a URL for deduplication: lowercase scheme/host, no fragment, sorted query."""
    raw = url.strip()
    if not raw:
        return None
    if base_url is not None:
        raw = urljoin(base_url, raw)

    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        return None

    host = (parts.hostname or "").lower()
    if not host:
        return None

    try:
        port = parts.port
    except ValueError:
        return None

    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None

    netloc = f"{host}:{port}" if port is not None else host

    raw_path = parts.path or "/"
    norm_path = posixpath.normpath(raw_path)
    if not norm_path.startswith("/"):
        norm_path = "/" + norm_path
    if len(norm_path) > 1 and norm_path.endswith("/"):
        norm_path = norm_path.rstrip("/")

    query_pairs = sorted(parse_qsl(parts.query, keep_blank_values=True))
    norm_query = urlencode(query_pairs, doseq=True)

    return urlunsplit((scheme, netloc, norm_path, norm_query, ""))


class _PageHTMLParser(HTMLParser):
    """Streaming HTMLParser extracting title, visible text, and anchor hrefs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.hrefs: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if lower == "title":
            self._in_title = True
        elif lower == "a":
            for key, val in attrs:
                if key.lower() == "href" and val:
                    self.hrefs.append(val)

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if lower == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        stripped = data.strip()
        if not stripped:
            return
        if self._in_title:
            self.title_parts.append(stripped)
        else:
            self.text_parts.append(stripped)


def extract_page_data(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """Extract ``(title, plain_text, canonical_links)`` from raw HTML."""
    parser = _PageHTMLParser()
    parser.feed(html)
    parser.close()

    title = re.sub(r"\s+", " ", " ".join(parser.title_parts)).strip()
    body = re.sub(r"\s+", " ", " ".join(parser.text_parts)).strip()
    if title and body:
        full_text = f"{title}\n{body}"
    else:
        full_text = title or body

    links: list[str] = []
    seen_local: set[str] = set()
    for href in parser.hrefs:
        canon = canonicalize_url(href, base_url=base_url)
        if canon is not None and canon not in seen_local:
            seen_local.add(canon)
            links.append(canon)

    return title, full_text, links
