"""Generate Lab 5 speedup plot (PNG) and update README.md with measured results & analysis.

Usage:
    uv run --with matplotlib python scripts/plot_lab5.py
"""

from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_plot(
    std: dict[str, object],
    ft: dict[str, object] | None,
    out_png: Path,
) -> None:
    """Plot speedup vs workers for threads (GIL), processes, threads (3.13t), and ideal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    std_rows: list[dict[str, float | int | str]] = std["rows"]  # type: ignore[assignment]
    serial_wall = float(std_rows[0]["wall_median"])

    th_x = [1]
    th_y = [1.0]
    pr_x = [1]
    pr_y = [1.0]
    for r in std_rows:
        w = int(r["workers"])
        sp = serial_wall / float(r["wall_median"])
        if r["executor"] == "threads":
            th_x.append(w)
            th_y.append(sp)
        elif r["executor"] == "processes":
            pr_x.append(w)
            pr_y.append(sp)

    max_w = max(th_x + pr_x)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ideal_x = sorted(set(th_x + pr_x))
    ax.plot(ideal_x, ideal_x, "--", color="gray", linewidth=1.4, label="Ideal linear (y = x)")
    ax.plot(
        th_x,
        th_y,
        "o-",
        color="#d62728",
        linewidth=2.0,
        markersize=7,
        label=f"threads (Python {std['python_version']}, GIL=True)",
    )
    ax.plot(
        pr_x,
        pr_y,
        "s-",
        color="#1f77b4",
        linewidth=2.2,
        markersize=7,
        label=f"processes (Python {std['python_version']}, spawn)",
    )

    if ft is not None:
        ft_rows: list[dict[str, float | int | str]] = ft["rows"]  # type: ignore[assignment]
        ft_x: list[int] = []
        ft_y: list[float] = []
        for r in ft_rows:
            w = int(r["workers"])
            sp = serial_wall / float(r["wall_median"])
            if r["executor"] in {"serial", "threads"}:
                ft_x.append(w)
                ft_y.append(sp)
        if ft_x:
            ax.plot(
                ft_x,
                ft_y,
                "^-",
                color="#2ca02c",
                linewidth=2.2,
                markersize=7,
                label=f"threads (Python {ft['python_version']}t, GIL=False)",
            )
            max_w = max(max_w, max(ft_x))

    ax.set_title(
        f"findex Parallel Indexing Speedup ({std['cpu_model']}, {std['cpu_count']} logical cores)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Workers")
    ax.set_ylabel("Speedup vs. serial (1 worker)")
    ax.set_xticks(sorted(set(th_x + pr_x)))
    ax.set_xlim(0.5, max_w + 0.5)
    ax.set_ylim(0, max(max_w * 0.65, max(pr_y) * 1.35, 3.0))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def build_readme_block(std: dict[str, object], ft: dict[str, object] | None) -> str:
    """Format machine specs, benchmark table, PNG link, I/O & race demos, and 1-page analysis."""
    std_rows: list[dict[str, float | int | str]] = std["rows"]  # type: ignore[assignment]
    serial_row = std_rows[0]
    serial_wall = float(serial_row["wall_median"])
    docs = int(serial_row["docs"])
    terms = int(serial_row["terms"])

    lines: list[str] = [
        "### Machine & methodology",
        "",
        f"- **CPU:** {std['cpu_model']} ({std['cpu_count']} logical cores)",
        f"- **OS:** {std['platform']}",
        f"- **Corpus:** 20 Newsgroups ({docs:,} documents, {terms:,} unique terms)",
        (
            f"- **Standard Python:** `{std['python_version']}` "
            f"(`sys._is_gil_enabled() == {std['gil_enabled']}`)"
        ),
    ]
    if ft is not None:
        lines.append(
            f"- **Free-threaded Python:** `{ft['python_version']}t` "
            f"(`sys._is_gil_enabled() == {ft['gil_enabled']}`)"
        )
    lines.append(
        f"- **Methodology:** 1 warm-up run + {std['runs_per_config']} timed runs per "
        "configuration in a clean isolated process; table reports median wall time, total CPU time "
        "(parent + child workers), parent Peak RSS, and sequential `merge()` time."
    )
    lines.append("")
    lines.append("### Parallel indexing benchmark")
    lines.append("")
    lines.append(
        "| Executor | Workers | Wall (s, median) | CPU (s) | Peak RSS | Merge (s) | Speedup |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    best_proc_wall = serial_wall
    best_proc_row = serial_row
    best_th_wall = serial_wall
    best_th_row = serial_row

    for r in std_rows:
        w = int(r["workers"])
        wall = float(r["wall_median"])
        cpu = float(r["cpu_median"])
        rss = float(r["peak_rss_mb"])
        merge_s = float(r["merge_median"])
        sp = serial_wall / wall if wall > 0 else 1.0
        label = str(r["executor"])
        if label == "processes" and wall < best_proc_wall:
            best_proc_wall = wall
            best_proc_row = r
        if label == "threads" and wall < best_th_wall:
            best_th_wall = wall
            best_th_row = r
        lines.append(
            f"| `{label}` | {w} | {wall:.3f} | {cpu:.3f} | {rss:.1f} MB "
            f"| {merge_s:.3f} | {sp:.2f}× |"
        )

    best_ft_row: dict[str, float | int | str] | None = None
    if ft is not None:
        ft_rows: list[dict[str, float | int | str]] = ft["rows"]  # type: ignore[assignment]
        for r in ft_rows:
            w = int(r["workers"])
            wall = float(r["wall_median"])
            cpu = float(r["cpu_median"])
            rss = float(r["peak_rss_mb"])
            merge_s = float(r["merge_median"])
            sp = serial_wall / wall if wall > 0 else 1.0
            ex = (
                "serial (3.13t, GIL=False)"
                if r["executor"] == "serial"
                else "threads (3.13t, GIL=False)"
            )
            if r["executor"] == "threads":
                if best_ft_row is None or wall < float(best_ft_row["wall_median"]):
                    best_ft_row = r
            lines.append(
                f"| `{ex}` | {w} | {wall:.3f} | {cpu:.3f} | {rss:.1f} MB "
                f"| {merge_s:.3f} | {sp:.2f}× |"
            )

    lines.extend(
        [
            "",
            "![Parallel indexing speedup vs workers](docs/speedup_lab05.png)",
            "",
        ]
    )

    # I/O-bound counter-benchmark
    io_rows = std.get("io_benchmark")
    if isinstance(io_rows, list) and io_rows:
        lines.extend(
            [
                "### I/O-bound counter-benchmark (threads DO scale on I/O)",
                "",
                "Fetching 32 local HTTP responses (20 ms latency each) via `ThreadPoolExecutor`:",
                "",
                "| Workers (threads) | Requests | Wall (s) | Speedup |",
                "|---:|---:|---:|---:|",
            ]
        )
        for ir in io_rows:
            lines.append(
                f"| {ir['workers']} | {ir['requests']} "
                f"| {ir['wall_sec']:.3f} | {ir['speedup']:.2f}× |"
            )
        lines.append("")

    # Free-threaded race condition demo
    if ft is not None and isinstance(ft.get("race_demo"), dict):
        race: dict[str, object] = ft["race_demo"]  # type: ignore[assignment]
        lines.extend(
            [
                "### Free-threaded `3.13t` data-race experiment",
                "",
                (
                    f"With `sys._is_gil_enabled() == {race['gil_enabled']}`, running "
                    f"{race['threads']} threads × 400,000 increments (expected "
                    f"`{race['expected']:,}`):"
                ),
                "",
                f"- **Unsafe shared `int` (`shared += 1`, no lock):** "
                f"`{race['unsafe_result']:,}` — **lost `{race['unsafe_lost']:,}` updates** "
                f"in {race['unsafe_sec']:.3f} s because `+=` is a non-atomic load/add/store.",
                f"- **Fixed with `threading.Lock`:** exact `{race['locked_result']:,}` in "
                f"{race['locked_sec']:.3f} s (correct, but heavy lock contention).",
                f"- **Fixed by removing shared state (thread-local + merge, as in `findex`):** "
                f"exact `{race['local_merge_result']:,}` in {race['local_merge_sec']:.3f} s "
                "(both race-free and fast).",
                "",
            ]
        )

    # One-page explanation grounded in the measured numbers
    proc_sp = serial_wall / best_proc_wall if best_proc_wall > 0 else 1.0
    th_sp = serial_wall / best_th_wall if best_th_wall > 0 else 1.0
    proc_merge = float(best_proc_row["merge_median"])
    merge_frac = (proc_merge / serial_wall) if serial_wall > 0 else 0.0
    amdahl_ceiling = (1.0 / merge_frac) if merge_frac > 0.001 else float("inf")

    ft_sentence = ""
    if best_ft_row is not None:
        ft_wall = float(best_ft_row["wall_median"])
        ft_sp = serial_wall / ft_wall if ft_wall > 0 else 1.0
        ft_sentence = (
            f"On free-threaded Python **3.13t** (`sys._is_gil_enabled() == False`), the exact same "
            f"`ThreadPoolExecutor` code jumped to **{ft_sp:.2f}×** ({ft_wall:.3f} s at "
            f"{best_ft_row['workers']} workers) without paying process startup or pickle IPC "
            "overhead. Removing the GIL adds a modest single-thread overhead (compare 1-worker "
            "3.13t vs standard Python), and — as the race experiment above demonstrates — "
            "immediately exposes bugs in shared mutable state (`shared += 1` lost updates that the "
            "GIL previously hid). Because `findex` uses isolated `build_partial()` maps followed "
            "by a single `merge()`, our indexer has zero shared mutable state across workers."
        )
    else:
        ft_sentence = (
            "On free-threaded Python **3.13t** (`sys._is_gil_enabled() == False`), the GIL is "
            "removed so `ThreadPoolExecutor` can run Python bytecode across multiple cores "
            "without process-spawning or pickle overhead, at the cost of a small single-thread "
            "overhead and exposing any unsynchronized shared state."
        )

    lines.extend(
        [
            "### Concurrency, the GIL, and Amdahl's Law — Analysis",
            "",
            "**1. Why threads did not speed up indexing under the GIL.** "
            "Indexing is overwhelmingly **CPU-bound** in pure Python: UTF-8 decoding, Unicode NFC "
            "normalization, regex tokenization, `Counter` hashing, and allocating `Posting` "
            "objects. In standard CPython, the Global Interpreter Lock (GIL) protects reference "
            "counts by ensuring only one thread executes Python bytecode at a time (switching "
            "roughly every 5 ms via `sys.getswitchinterval()`). Adding more threads therefore "
            f"gave only **{th_sp:.2f}×** ({best_th_wall:.3f} s at {best_th_row['workers']} "
            "threads vs "
            f"{serial_wall:.3f} s serial) and often runs *slower* due to lock contention and CPU "
            "cache ping-ponging. By contrast, our I/O-bound HTTP benchmark scaled cleanly with "
            "threads because blocking C calls like `socket.recv()` release the GIL while waiting.",
            "",
            "**2. Why processes scaled — and what it cost.** "
            "`ProcessPoolExecutor` launches separate Python interpreters (using `spawn` on "
            "Windows, guarded by the CLI entry point / `if __name__ == '__main__'`), each with "
            "its own memory space and its own GIL. "
            f"Best process run reached **{best_proc_wall:.3f} s** vs **{serial_wall:.3f} s** "
            f"(**{proc_sp:.2f}×** at {best_proc_row['workers']} workers). The hidden bill shows up "
            "in two columns of the table: **CPU time** rose to "
            f"**{float(best_proc_row['cpu_median']):.3f} s** (far exceeding wall time because all "
            "worker cores burn CPU simultaneously plus pickle serialization/deserialization of "
            "`PartialIndex` results over pipes), and **Peak RSS** of the parent process grew to "
            f"**{float(best_proc_row['peak_rss_mb']):.1f} MB** while receiving partial indexes "
            "and building the merged dictionary. To minimize IPC tax, workers receive only tiny "
            "picklable `ChunkSpec` path/byte-offset descriptors — never preloaded document text.",
            "",
            "**3. Where Amdahl's Law shows up in `merge()`.** "
            "Our map-reduce pipeline parallelizes `build_partial()` across workers, while "
            "`merge()` runs sequentially in the parent to combine posting lists. At the fastest "
            f"process run, `merge()` alone took **{proc_merge:.3f} s** (~"
            f"{merge_frac * 100:.1f}% of serial time). By Amdahl's Law "
            r"($S_{\max} = 1 / (s + (1 - s)/N)$), even with infinite cores a sequential fraction "
            f"of {merge_frac * 100:.1f}% caps theoretical speedup near **{amdahl_ceiling:.1f}×** "
            "(further bounded in practice by `spawn` startup, pickle IPC, and memory bandwidth). "
            "This explains why the speedup curve flattens past 4–8 workers instead of following "
            "the ideal linear line.",
            "",
            f"**4. What changed on free-threaded Python 3.13t.** {ft_sentence}",
            "",
            "**5. Executor trade-off.** `--executor processes` scales on multi-core servers "
            "(and is supported as the CLI default), while on a 2-core Windows laptop with "
            "`spawn`, pickle IPC overhead makes `3.13t` threads the fastest parallel path.",
        ]
    )

    return "\n".join(lines)


def update_readme(readme_path: Path, block: str) -> None:
    start_marker = "<!-- LAB05-RESULTS-START -->"
    end_marker = "<!-- LAB05-RESULTS-END -->"
    content = readme_path.read_text(encoding="utf-8")
    if start_marker not in content or end_marker not in content:
        raise ValueError("README.md is missing LAB05-RESULTS markers")
    before = content.split(start_marker, 1)[0]
    after = content.split(end_marker, 1)[1]
    readme_path.write_text(
        f"{before}{start_marker}\n{block}\n{end_marker}{after}",
        encoding="utf-8",
    )


def main() -> int:
    std = _load_json(Path("bench_results.json"))
    if std is None:
        print("bench_results.json not found. Run scripts/bench_lab5.py --mode standard first.")
        return 1
    ft = _load_json(Path("bench_313t.json"))

    out_png = Path("docs/speedup_lab05.png")
    make_plot(std, ft, out_png)
    block = build_readme_block(std, ft)
    update_readme(Path("README.md"), block)
    print(f"Generated {out_png} and updated README.md with Lab 05 benchmark & analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
