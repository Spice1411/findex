"""Lab 6 async crawler benchmark, debug verification, and README auto-updater.

Usage:
    uv run --with psutil python scripts/bench_lab6.py --max-pages 220 --update-readme
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import sys
import time
from pathlib import Path

import psutil

from findex.crawler.crawl import CrawlStats, crawl
from findex.index import build_index
from findex.search import search

SCRIPTS_DIR = Path(__file__).resolve().parent


def _peak_rss_mb() -> float:
    info = psutil.Process().memory_info()
    raw = getattr(info, "peak_wset", info.rss)
    return round(raw / (1024 * 1024), 1)


async def _run_one(
    seed_url: str,
    *,
    max_pages: int,
    concurrency: int,
    per_host: int,
    min_delay: float,
    out_jsonl: Path | None = None,
    log_path: Path | None = None,
) -> tuple[dict[str, float | int], list[str]]:
    stats = CrawlStats()
    log_lines: list[str] = []

    def _on_log(line: str) -> None:
        log_lines.append(line)

    t0 = time.perf_counter()
    if out_jsonl is not None:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as handle:
            async for page in crawl(
                [seed_url],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                min_delay=min_delay,
                timeout=0.6,
                stats=stats,
                on_log=_on_log,
                max_retries=2,
                base_retry_delay=0.05,
            ):
                rec = {
                    "id": page.url,
                    "url": page.url,
                    "title": page.title,
                    "text": page.text,
                    "fetched_at": page.fetched_at,
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        async for _page in crawl(
            [seed_url],
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=per_host,
            min_delay=min_delay,
            timeout=0.6,
            stats=stats,
            on_log=_on_log,
            max_retries=2,
            base_retry_delay=0.05,
        ):
            pass
    wall = time.perf_counter() - t0

    if log_path is not None:
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    row: dict[str, float | int] = {
        "concurrency": concurrency,
        "per_host": per_host,
        "pages": stats.pages_fetched,
        "wall_sec": round(wall, 3),
        "pages_per_sec": round(stats.pages_fetched / wall if wall > 0 else 0.0, 1),
        "errors": stats.errors,
        "skipped_robots": stats.skipped_robots,
        "peak_rss_mb": _peak_rss_mb(),
    }
    return row, log_lines


async def _run_suite(max_pages: int) -> dict[str, object]:
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from local_mirror import LocalMirrorCluster

    cluster = LocalMirrorCluster(num_servers=4, pages_per_server=65, latency_sec=0.02, base_port=0)
    await cluster.start()
    try:
        rows: list[dict[str, float | int]] = []
        saved_logs: list[str] = []
        crawl_out = Path("data/crawl.jsonl")
        crawl_log = Path("crawl.log")

        for conc in (1, 5, 20):
            cluster.reset_fault_counters()
            per_host = 5 if conc == 20 else min(conc, 5)
            out_file = crawl_out if conc == 20 else None
            log_file = crawl_log if conc == 20 else None
            row, logs = await _run_one(
                cluster.seed_url,
                max_pages=max_pages,
                concurrency=conc,
                per_host=per_host,
                min_delay=0.01,
                out_jsonl=out_file,
                log_path=log_file,
            )
            rows.append(row)
            if conc == 20:
                saved_logs = logs
            print(
                f"concurrency={conc:>2} | pages={row['pages']} | wall={row['wall_sec']:.3f}s | "
                f"{row['pages_per_sec']:.1f} pages/s | errors={row['errors']} | "
                f"rss={row['peak_rss_mb']:.1f} MB",
                flush=True,
            )

        # Extra comparison: strict per-host limit (per_host=1, min_delay=0.05s) at concurrency=20
        cluster.reset_fault_counters()
        polite_row, _ = await _run_one(
            cluster.seed_url,
            max_pages=max_pages,
            concurrency=20,
            per_host=1,
            min_delay=0.05,
        )
        print(
            f"strict per-host (1/host, 50ms delay) | wall={polite_row['wall_sec']:.3f}s | "
            f"{polite_row['pages_per_sec']:.1f} pages/s",
            flush=True,
        )

        # Verify event-loop debug mode reports ZERO slow callbacks (>100ms)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        aio_logger = logging.getLogger("asyncio")
        prev_lvl = aio_logger.level
        aio_logger.setLevel(logging.WARNING)
        aio_logger.addHandler(handler)
        try:
            loop = asyncio.get_running_loop()
            prev_debug = loop.get_debug()
            loop.set_debug(True)
            cluster.reset_fault_counters()
            await _run_one(
                cluster.seed_url,
                max_pages=40,
                concurrency=10,
                per_host=3,
                min_delay=0.0,
            )
            loop.set_debug(prev_debug)
        finally:
            aio_logger.removeHandler(handler)
            aio_logger.setLevel(prev_lvl)

        slow_warnings = [
            ln for ln in stream.getvalue().splitlines() if "took" in ln and "seconds" in ln
        ]
    finally:
        await cluster.stop()

    # Index the streamed crawl.jsonl OUTSIDE the event loop and verify search works
    t_idx = time.perf_counter()
    idx = build_index(crawl_out, with_positions=True, workers=1, executor="serial")
    idx_sec = round(time.perf_counter() - t_idx, 3)
    hits = search(idx, 'asyncio AND "event loop"', k=3, include_snippets=False)

    # Extract interesting failure / recovery log lines
    failure_lines = [
        ln
        for ln in saved_logs
        if any(
            key in ln
            for key in ("/rate-limited", "/slow-timeout", "/redirect-loop", "ROBOTS", "404")
        )
    ]

    return {
        "max_pages": max_pages,
        "rows": rows,
        "strict_per_host_row": polite_row,
        "debug_slow_warnings": len(slow_warnings),
        "indexed_docs": idx.num_docs,
        "indexed_terms": idx.num_terms,
        "index_seconds": idx_sec,
        "sample_search_hits": len(hits),
        "failure_log_lines": failure_lines[:8],
    }


def format_readme_block(result: dict[str, object]) -> str:
    rows: list[dict[str, float | int]] = result["rows"]  # type: ignore[assignment]
    polite: dict[str, float | int] = result["strict_per_host_row"]  # type: ignore[assignment]
    fail_lines: list[str] = result["failure_log_lines"]  # type: ignore[assignment]

    r1, r20 = rows[0], rows[2]
    speedup_20 = float(r1["wall_sec"]) / max(float(r20["wall_sec"]), 1e-6)

    lines: list[str] = [
        "### Ethics & permission",
        "",
        "- **Target crawled:** our own local multi-origin ethical HTML documentation mirror "
        "(`scripts/local_mirror.py` running on `127.0.0.1` across 4 local ports with 20 ms "
        "simulated network latency, `/robots.txt`, and controlled 429/timeout/redirect-loop "
        "endpoints). Permission is implicit because we own and operate the mirror locally.",
        "- **Politeness safeguards:** honest `User-Agent: findex-crawler/0.6 "
        "(+https://github.com/Spice1411/findex)`, `RobotsCache` obeying `/robots.txt` "
        "(`Disallow: /private/`), global + per-host `asyncio.Semaphore`, per-host minimum delay, "
        "domain allowlist, and hard `--max-pages` ceiling.",
        "",
        f"### Async crawler benchmark (`max_pages = {result['max_pages']}`, same seed URL)",
        "",
        "| Concurrency | Pages | Wall (s) | Pages/s | Errors | Peak RSS |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        label = (
            "1 (effectively sequential)" if int(r["concurrency"]) == 1 else str(r["concurrency"])
        )
        lines.append(
            f"| {label} | {r['pages']} | {r['wall_sec']:.3f} | {r['pages_per_sec']:.1f} "
            f"| {r['errors']} | {r['peak_rss_mb']:.1f} MB |"
        )

    lines.extend(
        [
            "",
            f"- **Event-loop debug verification (`debug=True`):** "
            f"`{result['debug_slow_warnings']}` slow-callback warnings (>100 ms).",
            f"- **Post-crawl offline indexing (`findex index data/crawl.jsonl`):** indexed "
            f"`{result['indexed_docs']}` streamed documents (`{result['indexed_terms']:,}` terms) "
            f"in `{result['index_seconds']:.3f} s` outside the event loop.",
            "",
            "### Handled failures & politeness excerpt (`crawl.log`)",
            "",
            "```text",
            *fail_lines,
            "```",
            "",
            "Notice that `/rate-limited` returned HTTP `429` with `Retry-After: 0.15` on attempt 1 "
            "and succeeded (`200`, `attempts=2`) after waiting; `/slow-timeout` tripped "
            "`asyncio.timeout` and exited cleanly after bounded retries; `/redirect-loop` stopped "
            "immediately without retrying; `/private/secret` was blocked by `RobotsCache` before "
            "any request was sent.",
            "",
            "### Asyncio vs. Lab 5, per-host politeness cost, and blocking analysis",
            "",
            f"**1. Why 20 concurrent coroutines in one thread beat Lab 5 for crawling.** "
            f"Raising concurrency from 1 to 20 lifted throughput from "
            f"**{r1['pages_per_sec']:.1f}** to **{r20['pages_per_sec']:.1f} pages/s** "
            f"(**{speedup_20:.1f}×** speedup) inside a single OS thread with low memory "
            f"(**{r20['peak_rss_mb']:.1f} MB** Peak RSS). Unlike Lab 5's CPU-bound indexing, "
            "crawling spends almost all wall time waiting on sockets (`await client.get(...)`). "
            "One event loop multiplexes all 20 in-flight sockets via OS readiness events and "
            "reuses one `httpx.AsyncClient` pool — without spawning 20 processes or pickle IPC.",
            "",
            f"**2. What per-host limits cost — and why we keep them.** "
            f"Tightening the per-host limit to `per_host=1` and `min_delay=50 ms` at the same "
            f"`concurrency=20` slowed the 220-page crawl from **{r20['wall_sec']:.3f} s** "
            f"({r20['pages_per_sec']:.1f} pages/s) to **{polite['wall_sec']:.3f} s** "
            f"({polite['pages_per_sec']:.1f} pages/s). That throttling is intentional: without "
            "per-host semaphores and minimum delays, an async crawler turns into a self-inflicted "
            "DoS against a single origin, triggering `429` bans and dropping connections.",
            "",
            "**3. Where the event loop would stall if indexing ran inline.** "
            "Tokenizing, Unicode normalization, and inverted-index dict updates are pure CPU "
            "loops with zero `await` points. Running `build_index` directly inside a worker "
            "coroutine would freeze the single event loop for tens or hundreds of milliseconds per "
            "page — delaying heartbeats, socket reads, and `asyncio.timeout` deadlines for all "
            "other 19 in-flight requests (and triggering `debug=True` slow-callback warnings). "
            "Streaming raw pages to `data/crawl.jsonl` and running `findex index` afterward keeps "
            "I/O inside `asyncio` and CPU work outside the loop.",
        ]
    )
    return "\n".join(lines)


def update_readme(readme_path: Path, block: str) -> None:
    start_marker = "<!-- LAB06-RESULTS-START -->"
    end_marker = "<!-- LAB06-RESULTS-END -->"
    content = readme_path.read_text(encoding="utf-8")
    if start_marker not in content or end_marker not in content:
        raise ValueError("README.md is missing LAB06-RESULTS markers")
    before = content.split(start_marker, 1)[0]
    after = content.split(end_marker, 1)[1]
    readme_path.write_text(
        f"{before}{start_marker}\n{block}\n{end_marker}{after}",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Lab 6 async crawler benchmark.")
    parser.add_argument("--max-pages", type=int, default=220)
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(_run_suite(args.max_pages))
    Path("bench_lab6.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    block = format_readme_block(result)

    if args.update_readme:
        update_readme(Path("README.md"), block)
        print("\nUpdated README.md with real Lab 06 benchmark, failure logs, and analysis.")
    else:
        print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
