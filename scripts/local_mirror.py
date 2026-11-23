"""Ethical local multi-origin async HTTP mirror for Lab 6 benchmarking & live demo.

Serves 260+ interlinked technical HTML pages across 1..4 local ports with:
- /robots.txt (Allow: /, Disallow: /private/)
- Configurable simulated network latency (default 20 ms)
- Controlled failure routes for testing resilience:
    * /rate-limited   -> returns HTTP 429 with Retry-After: 0.15 on 1st hit, then 200 OK
    * /slow-timeout   -> sleeps 2.5s to trigger asyncio.timeout
    * /redirect-loop  -> 302 cycle (/redirect-loop -> /redirect-loop-b -> /redirect-loop)
    * /private/secret -> disallowed by robots.txt

Usage (standalone demo server for defense):
    uv run python scripts/local_mirror.py --ports 1 --base-port 8765
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

TOPICS = [
    ("asyncio", "Cooperative concurrency uses an event loop with coroutines and await."),
    ("taskgroup", "Structured concurrency via TaskGroup cancels sibling tasks on failure."),
    ("semaphore", "Global and per-host semaphores bound in-flight HTTP requests politely."),
    ("robots", "RobotsCache reads robots.txt per host and obeys Crawl-delay and Disallow."),
    ("backoff", "Transient 429 and 5xx responses retry with exponential backoff and jitter."),
    ("bm25", "Okapi BM25 saturates term frequency with k1 and normalizes document length with b."),
    ("inverted", "An inverted index maps each normalized token to a sorted list of postings."),
    ("streaming", "Streaming JSONL pipelines keep peak memory constant while crawling."),
    ("http", "Reusing a single httpx.AsyncClient connection pool avoids TCP handshake overhead."),
    ("parsing", "URL canonicalization strips fragments, lowercases host, and sorts query params."),
]


class LocalMirrorCluster:
    """Async multi-port HTTP server cluster running entirely on 127.0.0.1."""

    def __init__(
        self,
        *,
        num_servers: int = 4,
        pages_per_server: int = 65,
        latency_sec: float = 0.02,
        base_port: int = 0,
    ) -> None:
        self.num_servers = max(1, num_servers)
        self.pages_per_server = pages_per_server
        self.latency_sec = latency_sec
        self.base_port = base_port
        self.ports: list[int] = []
        self._servers: list[asyncio.Server] = []
        self._rate_limit_hits: dict[int, int] = {}

    def reset_fault_counters(self) -> None:
        self._rate_limit_hits.clear()

    @property
    def seed_url(self) -> str:
        return f"http://127.0.0.1:{self.ports[0]}/"

    async def start(self) -> None:
        self.ports = []
        self._servers = []
        for idx in range(self.num_servers):
            port_req = self.base_port + idx if self.base_port > 0 else 0
            server = await asyncio.start_server(
                lambda r, w, s_idx=idx: self._handle_client(s_idx, r, w),
                host="127.0.0.1",
                port=port_req,
            )
            sock = server.sockets[0]
            actual_port = int(sock.getsockname()[1])
            self.ports.append(actual_port)
            self._servers.append(server)

    async def stop(self) -> None:
        for server in self._servers:
            server.close()
        for server in self._servers:
            await server.wait_closed()
        self._servers.clear()

    def _build_response(
        self,
        status: int,
        reason: str,
        body: str,
        *,
        content_type: str = "text/html; charset=utf-8",
        extra_headers: Sequence[tuple[str, str]] = (),
    ) -> bytes:
        body_bytes = body.encode("utf-8")
        headers = [
            f"HTTP/1.1 {status} {reason}",
            f"Content-Type: {content_type}",
            f"Content-Length: {len(body_bytes)}",
            "Connection: close",
        ]
        for k, v in extra_headers:
            headers.append(f"{k}: {v}")
        raw = "\r\n".join(headers) + "\r\n\r\n"
        return raw.encode("ascii") + body_bytes

    def _render_hub(self, server_idx: int) -> str:
        links: list[str] = []
        for s_i, port in enumerate(self.ports):
            links.append(f'<a href="http://127.0.0.1:{port}/page/0">Host {s_i} Start</a>')
            for p in range(0, min(10, self.pages_per_server)):
                links.append(f'<a href="http://127.0.0.1:{port}/page/{p}">H{s_i} Page {p}</a>')
        if server_idx == 0:
            links.extend(
                [
                    '<a href="/rate-limited">Rate Limited Page</a>',
                    '<a href="/slow-timeout">Slow Timeout Page</a>',
                    '<a href="/redirect-loop">Redirect Loop Page</a>',
                    '<a href="/private/secret">Private Secret (Robots Disallowed)</a>',
                    '<a href="/missing-404">Missing Page</a>',
                ]
            )
        nav = " ".join(links)
        return (
            "<!doctype html><html><head>"
            f"<title>findex Local Mirror Hub {server_idx}</title>"
            "</head><body>"
            f"<h1>Local Documentation Mirror (Node {server_idx})</h1>"
            "<p>Ethical local mirror for asyncio event loop crawler benchmarks and search.</p>"
            f"<nav>{nav}</nav></body></html>"
        )

    def _render_page(self, server_idx: int, page_no: int) -> str:
        topic, desc = TOPICS[(server_idx * 7 + page_no) % len(TOPICS)]
        links: list[str] = []
        for step in (1, 2, 5):
            nxt = page_no + step
            if nxt < self.pages_per_server:
                links.append(f'<a href="/page/{nxt}">Next {nxt}</a>')
        # Also add a duplicate link with fragment and unsorted query to test canonicalization
        if page_no + 1 < self.pages_per_server:
            links.append(f'<a href="/page/{page_no + 1}#section">Section link</a>')
        other_idx = (server_idx + 1) % len(self.ports)
        other_port = self.ports[other_idx]
        links.append(
            f'<a href="http://127.0.0.1:{other_port}/page/{page_no % self.pages_per_server}">'
            f"Cross-host {other_idx}</a>"
        )
        return (
            "<!doctype html><html><head>"
            f"<title>{topic.upper()} Guide — Node {server_idx} Page {page_no}</title>"
            "<style>body { font-family: sans-serif; }</style>"
            "</head><body>"
            f"<h1>{topic.title()} Documentation (Page {page_no})</h1>"
            f"<p>{desc}</p>"
            f"<p>Keyword markers: findex python {topic} node{server_idx} doc{page_no} "
            "event loop coroutine async crawler search engine ranking.</p>"
            "<script>console.log('ignored script text');</script>"
            f"<div>{' '.join(links)}</div>"
            "</body></html>"
        )

    async def _handle_client(
        self,
        server_idx: int,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            req_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not req_line:
                return
            parts = req_line.decode("latin-1", errors="replace").split()
            raw_target = parts[1] if len(parts) >= 2 else "/"
            path = raw_target.split("?", 1)[0]

            # Drain HTTP request headers
            while True:
                hdr = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if not hdr or hdr in {b"\r\n", b"\n"}:
                    break

            if path == "/robots.txt":
                robots_body = "User-agent: *\nDisallow: /private/\nAllow: /\n"
                resp = self._build_response(
                    200, "OK", robots_body, content_type="text/plain; charset=utf-8"
                )
            elif path == "/rate-limited":
                await asyncio.sleep(self.latency_sec)
                hits = self._rate_limit_hits.get(server_idx, 0) + 1
                self._rate_limit_hits[server_idx] = hits
                if hits == 1:
                    resp = self._build_response(
                        429,
                        "Too Many Requests",
                        "Rate limit exceeded. Retry shortly.",
                        content_type="text/plain; charset=utf-8",
                        extra_headers=[("Retry-After", "0.15")],
                    )
                else:
                    html = (
                        "<html><head><title>Recovered After 429</title></head>"
                        "<body><p>Successfully recovered after Retry-After backoff.</p></body>"
                        "</html>"
                    )
                    resp = self._build_response(200, "OK", html)
            elif path == "/slow-timeout":
                await asyncio.sleep(2.5)
                resp = self._build_response(200, "OK", "<html><body>Too slow</body></html>")
            elif path == "/redirect-loop":
                resp = self._build_response(
                    302,
                    "Found",
                    "",
                    extra_headers=[("Location", "/redirect-loop-b")],
                )
            elif path == "/redirect-loop-b":
                resp = self._build_response(
                    302,
                    "Found",
                    "",
                    extra_headers=[("Location", "/redirect-loop")],
                )
            elif path == "/missing-404":
                await asyncio.sleep(self.latency_sec)
                resp = self._build_response(404, "Not Found", "Not found")
            elif path == "/private/secret":
                resp = self._build_response(200, "OK", "<html><body>Secret</body></html>")
            elif path in {"/", ""}:
                await asyncio.sleep(self.latency_sec)
                resp = self._build_response(200, "OK", self._render_hub(server_idx))
            elif path.startswith("/page/"):
                await asyncio.sleep(self.latency_sec)
                tail = path.removeprefix("/page/").rstrip("/")
                if tail.isdigit() and int(tail) < self.pages_per_server:
                    resp = self._build_response(200, "OK", self._render_page(server_idx, int(tail)))
                else:
                    resp = self._build_response(404, "Not Found", "Not found")
            else:
                resp = self._build_response(404, "Not Found", "Not found")

            writer.write(resp)
            await writer.drain()
        except (TimeoutError, ConnectionError, OSError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass


async def _serve_forever(ports: int, base_port: int, pages: int, latency: float) -> None:
    cluster = LocalMirrorCluster(
        num_servers=ports,
        pages_per_server=pages,
        latency_sec=latency,
        base_port=base_port,
    )
    await cluster.start()
    print(f"Ethical local mirror running! Seed URL: {cluster.seed_url}")
    print(f"Active ports: {cluster.ports} | Press Ctrl+C to stop.")
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        await cluster.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ethical local mirror server for Lab 6.")
    parser.add_argument("--ports", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=8765)
    parser.add_argument("--pages", type=int, default=65)
    parser.add_argument("--latency", type=float, default=0.02)
    args = parser.parse_args()
    try:
        asyncio.run(_serve_forever(args.ports, args.base_port, args.pages, args.latency))
    except KeyboardInterrupt:
        print("\nMirror stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
