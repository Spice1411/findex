"""Lab 7 load-testing script: M2 def vs async def + M4 oha benchmark + README updater.

Usage:
    # Local benchmarks only:
    uv run python scripts/bench_lab7.py --index index.bin --update-readme

    # Full benchmark including deployed public URL:
    uv run python scripts/bench_lab7.py --index index.bin \
        --deployed-url https://findex-search.onrender.com --update-readme
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _find_oha() -> str | None:
    local_bin = Path(".tools/oha.exe") if sys.platform == "win32" else Path(".tools/oha")
    if local_bin.exists():
        return str(local_bin.resolve())
    return shutil.which("oha") or shutil.which("oha.exe")


async def _wait_healthy(base_url: str, timeout_sec: float = 25.0) -> None:
    deadline = time.perf_counter() + timeout_sec
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.perf_counter() < deadline:
            try:
                resp = await client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.25)
    raise TimeoutError(f"Server at {base_url} did not become healthy within {timeout_sec}s")


async def _run_concurrent_20(url: str, count: int = 20) -> dict[str, float]:
    """Measure 20 concurrent GET /search requests (Lab 7 M2 decision evidence)."""
    latencies_ms: list[float] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Warm-up
        await client.get(url)

        async def _one() -> None:
            t0 = time.perf_counter()
            resp = await client.get(url)
            resp.raise_for_status()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        t_wall = time.perf_counter()
        async with asyncio.TaskGroup() as tg:
            for _ in range(count):
                tg.create_task(_one())
        wall_ms = (time.perf_counter() - t_wall) * 1000.0

    latencies_ms.sort()
    return {
        "wall_ms": round(wall_ms, 2),
        "p50_ms": round(statistics.median(latencies_ms), 2),
        "p95_ms": round(latencies_ms[int(0.95 * (len(latencies_ms) - 1))], 2),
        "max_ms": round(latencies_ms[-1], 2),
    }


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


async def _fallback_httpx_load(
    url: str,
    *,
    concurrency: int,
    duration_sec: int,
) -> dict[str, float | int | str]:
    """Fallback async load tester if oha binary is not installed."""
    latencies_ms: list[float] = []
    errors = 0
    stop_at = time.perf_counter() + duration_sec
    limits = httpx.Limits(max_connections=concurrency + 10, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(timeout=15.0, limits=limits) as client:

        async def _worker() -> None:
            nonlocal errors
            while time.perf_counter() < stop_at:
                t0 = time.perf_counter()
                try:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        errors += 1
                    else:
                        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                except httpx.HTTPError:
                    errors += 1

        t_start = time.perf_counter()
        async with asyncio.TaskGroup() as tg:
            for _ in range(concurrency):
                tg.create_task(_worker())
        elapsed = max(1e-6, time.perf_counter() - t_start)

    latencies_ms.sort()
    return {
        "tool": "httpx-async",
        "concurrency": concurrency,
        "requests": len(latencies_ms),
        "errors": errors,
        "rps": round(len(latencies_ms) / elapsed, 1),
        "p50_ms": round(_percentile(latencies_ms, 50), 2),
        "p95_ms": round(_percentile(latencies_ms, 95), 2),
        "p99_ms": round(_percentile(latencies_ms, 99), 2),
    }


def _run_oha_load(
    oha_bin: str,
    url: str,
    *,
    concurrency: int,
    duration_sec: int,
) -> dict[str, float | int | str]:
    cmd = [
        oha_bin,
        "-z",
        f"{duration_sec}s",
        "-c",
        str(concurrency),
        "--no-tui",
        "--output-format",
        "json",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"oha failed ({proc.returncode}): {proc.stderr}")

    data = json.loads(proc.stdout)
    summary = data.get("summary", {})
    pcts = data.get("latencyPercentiles", {})
    rps = float(summary.get("requestsPerSec", 0.0))

    def _ms(key: str) -> float:
        return round(float(pcts.get(key, 0.0)) * 1000.0, 2)

    return {
        "tool": "oha",
        "concurrency": concurrency,
        "rps": round(rps, 1),
        "p50_ms": _ms("p50"),
        "p95_ms": _ms("p95"),
        "p99_ms": _ms("p99"),
    }


def _run_load(
    oha_bin: str | None,
    url: str,
    *,
    concurrency: int,
    duration_sec: int,
) -> dict[str, float | int | str]:
    if oha_bin is not None:
        return _run_oha_load(oha_bin, url, concurrency=concurrency, duration_sec=duration_sec)
    return asyncio.run(
        _fallback_httpx_load(url, concurrency=concurrency, duration_sec=duration_sec)
    )


def _start_uvicorn(
    index_path: Path,
    *,
    port: int,
    workers: int,
    async_search: bool,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["INDEX_PATH"] = str(index_path.resolve())
    env["FINDEX_ASYNC_SEARCH"] = "1" if async_search else "0"
    env["LOG_LEVEL"] = "WARNING"
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "findex.web:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--log-level",
        "warning",
    ]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _stop_proc(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def format_readme_block(result: dict[str, object]) -> str:
    m2: dict[str, dict[str, float]] = result["m2_20_concurrent"]  # type: ignore[assignment]
    rows: list[dict[str, float | int | str]] = result["load_rows"]  # type: ignore[assignment]
    tool_used = str(result["tool"])
    deployed_url = str(result.get("deployed_url") or "")

    lines: list[str] = [
        "### M2 — Sync `def` vs. blocking `async def` on `/search` (20 concurrent requests)",
        "",
        "| Route definition | 20-req Wall (ms) | p50 (ms) | p95 (ms) | Max (ms) |",
        "|---|---:|---:|---:|---:|",
        (
            f"| `async def` (blocks event loop) | {m2['async_def']['wall_ms']:.2f} "
            f"| {m2['async_def']['p50_ms']:.2f} | {m2['async_def']['p95_ms']:.2f} "
            f"| {m2['async_def']['max_ms']:.2f} |"
        ),
        (
            f"| `def` (thread-pool offloaded — **kept in production**) "
            f"| {m2['def']['wall_ms']:.2f} | {m2['def']['p50_ms']:.2f} "
            f"| {m2['def']['p95_ms']:.2f} | {m2['def']['max_ms']:.2f} |"
        ),
        "",
        f"### M4 — Load test (`{tool_used}`, local vs. deployed)",
        "",
        "| Target | Route | Concurrency | RPS | p50 (ms) | p95 (ms) | p99 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['target']} | `/search` | {r['concurrency']} "
            f"| {r['rps']} | {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} |"
        )

    lines.extend(
        [
            "",
            "### Connecting the load-test numbers to Labs 5–6",
            "",
            "- **1 worker, `async def` vs. `def`:** BM25 scoring and snippet extraction are pure "
            "synchronous CPU/disk work without `await` points. Inside an `async def` route, each "
            "request freezes the single Uvicorn event loop (the exact Lab 6 blocking bug), so "
            "concurrent clients queue behind one another and p99 spikes. Changing the route to "
            "plain `def` tells FastAPI to offload execution to AnyIO's thread pool so the event "
            "loop stays responsive for HTTP accept/read/write and `/health` probes — though "
            "single-process throughput is still bounded by the CPython GIL (Lab 5).",
            "- **1 worker vs. 4 workers:** Launching `--workers 4` runs four independent Uvicorn "
            "processes, each with its own Python interpreter and its own GIL (Lab 5's "
            "multi-process escape hatch), multiplying available CPU cores and cutting tail latency "
            "(p95/p99) under heavy concurrency at the cost of holding the index in each worker.",
            "- **Deployed service:** "
            + (
                f"Running against the live public endpoint (`{deployed_url}`) adds real TLS + WAN "
                "round-trip latency and free-tier shared-CPU throttling, which raises p50/p99 "
                "compared to localhost while confirming end-to-end availability."
                if deployed_url
                else "Deploy with `Dockerfile` / `render.yaml` and re-run with "
                "`--deployed-url https://...` to record live cloud p50/p95/p99."
            ),
        ]
    )
    return "\n".join(lines)


def update_readme(readme_path: Path, block: str, deployed_url: str | None) -> None:
    content = readme_path.read_text(encoding="utf-8")

    if deployed_url:
        u_start = "<!-- LAB07-URL-START -->"
        u_end = "<!-- LAB07-URL-END -->"
        if u_start in content and u_end in content:
            url_block = (
                f"> **Live Public Demo (Lab 07):** [{deployed_url}]({deployed_url}) · "
                f"[OpenAPI Docs]({deployed_url.rstrip('/')}/docs) · "
                f"[Health Check]({deployed_url.rstrip('/')}/health)"
            )
            before_u = content.split(u_start, 1)[0]
            after_u = content.split(u_end, 1)[1]
            content = f"{before_u}{u_start}\n{url_block}\n{u_end}{after_u}"

    start_marker = "<!-- LAB07-RESULTS-START -->"
    end_marker = "<!-- LAB07-RESULTS-END -->"
    if start_marker not in content or end_marker not in content:
        raise ValueError("README.md is missing LAB07-RESULTS markers")
    before = content.split(start_marker, 1)[0]
    after = content.split(end_marker, 1)[1]
    readme_path.write_text(
        f"{before}{start_marker}\n{block}\n{end_marker}{after}",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Lab 7 FastAPI load tests.")
    parser.add_argument("--index", type=Path, default=Path("data/demo_index.bin"))
    parser.add_argument("--query", type=str, default="python")
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--deployed-url", type=str, default="")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()

    if not args.index.exists():
        print(f"Index not found at {args.index}. Run scripts/build_demo_index.py first.")
        return 1

    oha_bin = _find_oha()
    tool_name = "oha" if oha_bin else "httpx-async"
    print(f"Load-test tool: {tool_name} ({oha_bin or 'built-in fallback'})")

    m2_results: dict[str, dict[str, float]] = {}
    load_rows: list[dict[str, float | int | str]] = []

    configs: list[tuple[str, int, bool, int]] = [
        ("local, 1 worker, `async def`", 1, True, 50),
        ("local, 1 worker, `def`", 1, False, 50),
        ("local, 4 workers", 4, False, 50),
    ]

    for label, workers, is_async, conc in configs:
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        search_url = f"{base_url}/search?q={args.query}&k=10"
        print(f"\nStarting uvicorn ({label}) on port {port}...", flush=True)
        proc = _start_uvicorn(args.index, port=port, workers=workers, async_search=is_async)
        try:
            asyncio.run(_wait_healthy(base_url))
            if workers == 1:
                m2_key = "async_def" if is_async else "def"
                m2_res = asyncio.run(_run_concurrent_20(search_url, count=20))
                m2_results[m2_key] = m2_res
                print(f"  M2 (20 concurrent): {m2_res}", flush=True)

            row = _run_load(oha_bin, search_url, concurrency=conc, duration_sec=args.duration)
            row["target"] = label
            load_rows.append(row)
            print(
                f"  Load (c={conc}, {args.duration}s): RPS={row['rps']} | "
                f"p50={row['p50_ms']}ms | p95={row['p95_ms']}ms | p99={row['p99_ms']}ms",
                flush=True,
            )
        finally:
            _stop_proc(proc)

    deployed_clean = args.deployed_url.strip().rstrip("/")
    if deployed_clean:
        dep_search = f"{deployed_clean}/search?q={args.query}&k=10"
        print(f"\nWarming up deployed service at {deployed_clean}/health ...", flush=True)
        asyncio.run(_wait_healthy(deployed_clean, timeout_sec=60.0))
        dep_row = _run_load(oha_bin, dep_search, concurrency=20, duration_sec=args.duration)
        dep_row["target"] = f"deployed (`{deployed_clean}`)"
        load_rows.append(dep_row)
        print(
            f"  Deployed (c=20, {args.duration}s): RPS={dep_row['rps']} | "
            f"p50={dep_row['p50_ms']}ms | p95={dep_row['p95_ms']}ms | p99={dep_row['p99_ms']}ms",
            flush=True,
        )
    else:
        load_rows.append(
            {
                "target": "deployed (pending `--deployed-url`)",
                "concurrency": 20,
                "rps": "—",
                "p50_ms": "—",
                "p95_ms": "—",
                "p99_ms": "—",
            }
        )

    payload: dict[str, object] = {
        "tool": tool_name,
        "deployed_url": deployed_clean,
        "m2_20_concurrent": m2_results,
        "load_rows": load_rows,
    }
    Path("bench_lab7.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    block = format_readme_block(payload)
    if args.update_readme:
        update_readme(Path("README.md"), block, deployed_url=deployed_clean or None)
        print("\nUpdated README.md with Lab 07 load-test results.")
    else:
        print("\n" + block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
