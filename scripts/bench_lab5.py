"""Lab 5 benchmark runner (pure stdlib; works on standard Python 3.12/3.13 and 3.13t).

Usage:
    uv run python scripts/bench_lab5.py --mode standard --corpus data/
    uv run --no-project --python 3.13t python scripts/bench_lab5.py --mode 313t --corpus data/
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import http.server
import json
import os
import platform
import socketserver
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def is_gil_enabled() -> bool:
    """Return True if the GIL is enabled in the current interpreter."""
    fn = getattr(sys, "_is_gil_enabled", None)
    if callable(fn):
        return bool(fn())
    return True


def get_cpu_model() -> str:
    """Detect a clean CPU model name using only the stdlib."""
    if sys.platform == "win32":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                return str(val).strip()
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def get_peak_rss_mb() -> float:
    """Return Peak RSS of the current process in MiB using OS-native stdlib APIs."""
    if sys.platform == "win32":

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / (1024 * 1024)
        return 0.0

    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024.0


def run_single_config(
    corpus: Path,
    executor: str,
    workers: int,
    runs: int,
    limit: int | None,
) -> dict[str, object]:
    """Execute 1 warmup + `runs` timed runs in an isolated process for accurate Peak RSS."""
    from findex.index import ExecutorName, build_index_profiled

    exec_name: ExecutorName = (
        "threads"
        if executor == "threads"
        else ("processes" if executor == "processes" else "serial")
    )

    # Warm-up run (fills OS file cache and imports modules)
    warm = build_index_profiled(
        corpus,
        limit=limit,
        with_positions=False,
        workers=workers,
        executor=exec_name,
    )
    num_docs = warm.index.num_docs
    num_terms = warm.index.num_terms
    del warm
    gc.collect()

    walls: list[float] = []
    cpus: list[float] = []
    merges: list[float] = []

    for _ in range(runs):
        gc.collect()
        prof = build_index_profiled(
            corpus,
            limit=limit,
            with_positions=False,
            workers=workers,
            executor=exec_name,
        )
        walls.append(prof.wall_seconds)
        cpus.append(prof.cpu_seconds)
        merges.append(prof.merge_seconds)
        del prof

    return {
        "executor": exec_name,
        "workers": workers,
        "runs": runs,
        "docs": num_docs,
        "terms": num_terms,
        "wall_median": round(statistics.median(walls), 4),
        "wall_min": round(min(walls), 4),
        "wall_max": round(max(walls), 4),
        "cpu_median": round(statistics.median(cpus), 4),
        "merge_median": round(statistics.median(merges), 4),
        "peak_rss_mb": round(get_peak_rss_mb(), 2),
        "gil_enabled": is_gil_enabled(),
    }


def _spawn_single_config(
    corpus: Path,
    executor: str,
    workers: int,
    runs: int,
    limit: int | None,
) -> dict[str, object]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-config",
        "--corpus",
        str(corpus),
        "--executor",
        executor,
        "--workers",
        str(workers),
        "--runs",
        str(runs),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed for {executor}({workers}):\n{proc.stdout}\n{proc.stderr}"
        )
    last_line = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
    return json.loads(last_line)


def run_io_benchmark(num_requests: int = 32, delay_sec: float = 0.02) -> list[dict[str, object]]:
    """I/O-bound counter-benchmark: fetch local HTTP endpoints where socket.recv releases GIL."""

    class _DelayedHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            time.sleep(delay_sec)
            body = b'{"status":"ok","payload":"findex io benchmark"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    class _ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    server = _ThreadedServer(("127.0.0.1", 0), _DelayedHandler)
    port = server.server_address[1]
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    url = f"http://127.0.0.1:{port}/doc"

    def _fetch(_idx: int) -> int:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return len(resp.read())

    results: list[dict[str, object]] = []
    base_wall = 0.0
    try:
        for workers in (1, 2, 4, 8):
            t0 = time.perf_counter()
            if workers == 1:
                total_bytes = sum(_fetch(i) for i in range(num_requests))
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    total_bytes = sum(pool.map(_fetch, range(num_requests)))
            wall = time.perf_counter() - t0
            if workers == 1:
                base_wall = wall
            results.append(
                {
                    "workers": workers,
                    "requests": num_requests,
                    "bytes": total_bytes,
                    "wall_sec": round(wall, 3),
                    "speedup": round(base_wall / wall if wall > 0 else 1.0, 2),
                }
            )
    finally:
        server.shutdown()
        server.server_close()

    return results


def run_race_demo(threads: int = 4, per_thread: int = 400_000) -> dict[str, object]:
    """Demonstrate a shared-state race condition vs threading.Lock vs thread-local reduction."""
    expected = threads * per_thread

    # 1. Unsafe shared int (no lock)
    shared_unsafe = 0

    def _inc_unsafe() -> None:
        nonlocal shared_unsafe
        for _ in range(per_thread):
            shared_unsafe += 1

    t0 = time.perf_counter()
    th_list = [threading.Thread(target=_inc_unsafe) for _ in range(threads)]
    for t in th_list:
        t.start()
    for t in th_list:
        t.join()
    unsafe_time = time.perf_counter() - t0

    # 2. Protected with threading.Lock
    shared_locked = 0
    lock = threading.Lock()

    def _inc_locked() -> None:
        nonlocal shared_locked
        for _ in range(per_thread):
            with lock:
                shared_locked += 1

    t0 = time.perf_counter()
    th_list = [threading.Thread(target=_inc_locked) for _ in range(threads)]
    for t in th_list:
        t.start()
    for t in th_list:
        t.join()
    locked_time = time.perf_counter() - t0

    # 3. Lock-free thread-local partial counts + final merge (same pattern as build_partial+merge)
    partials = [0] * threads

    def _inc_local(slot: int) -> None:
        local = 0
        for _ in range(per_thread):
            local += 1
        partials[slot] = local

    t0 = time.perf_counter()
    th_list = [threading.Thread(target=_inc_local, args=(i,)) for i in range(threads)]
    for t in th_list:
        t.start()
    for t in th_list:
        t.join()
    merged_total = sum(partials)
    local_time = time.perf_counter() - t0

    return {
        "gil_enabled": is_gil_enabled(),
        "threads": threads,
        "expected": expected,
        "unsafe_result": shared_unsafe,
        "unsafe_lost": expected - shared_unsafe,
        "unsafe_sec": round(unsafe_time, 4),
        "locked_result": shared_locked,
        "locked_sec": round(locked_time, 4),
        "local_merge_result": merged_total,
        "local_merge_sec": round(local_time, 4),
    }


def _worker_grid(cpu_count: int) -> list[int]:
    grid = {1, 2, 4, 8, cpu_count}
    return sorted(w for w in grid if w >= 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lab 5 concurrency & GIL benchmark suite.")
    parser.add_argument("--corpus", type=Path, default=Path("data"))
    parser.add_argument("--mode", choices=["standard", "313t"], default="standard")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    # Internal isolated worker flags
    parser.add_argument("--single-config", action="store_true")
    parser.add_argument("--executor", choices=["serial", "threads", "processes"], default="serial")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    if args.single_config:
        row = run_single_config(args.corpus, args.executor, args.workers, args.runs, args.limit)
        print(json.dumps(row))
        return 0

    cpu_count = os.cpu_count() or 1
    workers_grid = _worker_grid(cpu_count)
    gil_state = is_gil_enabled()
    py_ver = sys.version.split()[0]

    print(
        f"=== Lab 5 Benchmark ({args.mode}) | Python {py_ver} | "
        f"sys._is_gil_enabled() = {gil_state} | CPUs = {cpu_count} ==="
    )
    print(f"CPU model: {get_cpu_model()}")
    print(f"Workers grid: {workers_grid} | Runs per config: {args.runs}\n")

    rows: list[dict[str, object]] = []
    if args.mode == "standard":
        tasks: list[tuple[str, int]] = [("serial", 1)]
        for w in workers_grid:
            if w > 1:
                tasks.append(("threads", w))
        for w in workers_grid:
            if w > 1:
                tasks.append(("processes", w))
    else:
        tasks = [("serial", 1)]
        for w in workers_grid:
            if w > 1:
                tasks.append(("threads", w))

    for idx, (ex, w) in enumerate(tasks, start=1):
        print(f"[{idx}/{len(tasks)}] Running {ex} (workers={w})...", flush=True)
        row = _spawn_single_config(args.corpus, ex, w, args.runs, args.limit)
        rows.append(row)
        print(
            f"    -> wall={row['wall_median']:.3f}s | cpu={row['cpu_median']:.3f}s | "
            f"merge={row['merge_median']:.3f}s | rss={row['peak_rss_mb']:.1f} MB",
            flush=True,
        )

    payload: dict[str, object] = {
        "mode": args.mode,
        "python_version": py_ver,
        "python_full": sys.version,
        "gil_enabled": gil_state,
        "cpu_model": get_cpu_model(),
        "cpu_count": cpu_count,
        "platform": platform.platform(),
        "runs_per_config": args.runs,
        "rows": rows,
    }

    if args.mode == "standard":
        print("\nRunning I/O-bound counter-benchmark (32 local HTTP requests)...", flush=True)
        io_rows = run_io_benchmark()
        payload["io_benchmark"] = io_rows
        for r in io_rows:
            print(f"    -> threads={r['workers']}: {r['wall_sec']}s ({r['speedup']}x)")
        out_path = args.out or Path("bench_results.json")
    else:
        print("\nRunning free-threaded race condition demo (4 threads x 400k)...", flush=True)
        race = run_race_demo()
        payload["race_demo"] = race
        print(
            f"    -> expected={race['expected']:,} | unsafe={race['unsafe_result']:,} "
            f"(lost {race['unsafe_lost']:,}) | locked={race['locked_result']:,} | "
            f"local+merge={race['local_merge_result']:,}"
        )
        out_path = args.out or Path("bench_313t.json")

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
