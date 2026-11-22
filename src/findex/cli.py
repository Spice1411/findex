"""Command-line interface for findex built with Typer and Rich (Lab 4–6)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.live import Live
from rich.markup import escape as escape_rich
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from findex.crawler.crawl import CrawlStats, crawl
from findex.index import ExecutorName, build_index_profiled
from findex.query import QueryParseError
from findex.ranking import ScorerName, make_scorer
from findex.search import search as run_search
from findex.stats import compute_index_stats, compute_stats_lazy
from findex.store import open_index, save_index

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="findex",
    help="Streaming full-text search engine with async crawler, parallel indexing, and BM25.",
    no_args_is_help=True,
)


class ScorerChoice(StrEnum):
    """Supported ranking scorers exposed as CLI options."""

    bm25 = "bm25"
    tfidf = "tfidf"


class ExecutorChoice(StrEnum):
    """Supported parallel execution backends for index construction."""

    serial = "serial"
    threads = "threads"
    processes = "processes"


def configure_logging(verbosity: int) -> None:
    """Configure root logging once at CLI startup; all diagnostics go to stderr."""
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s]: %(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)


def _fail(message: str) -> NoReturn:
    err_console = Console(stderr=True)
    err_console.print(f"[bold red]Error:[/bold red] {escape_rich(message)}")
    raise typer.Exit(code=1)


def _render_crawl_table(stats: CrawlStats, max_pages: int) -> Table:
    table = Table(title="findex Async Crawler (Live)", show_header=False)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")
    table.add_row("Pages fetched", f"{stats.pages_fetched:,} / {max_pages:,}")
    table.add_row("Pages / sec", f"{stats.pages_per_second:.1f}")
    table.add_row("In-flight requests", str(stats.in_flight))
    table.add_row("Queue depth", f"{stats.queue_depth:,}")
    table.add_row("Errors", f"[red]{stats.errors}[/red]" if stats.errors else "0")
    table.add_row("Skipped (robots)", str(stats.skipped_robots))
    table.add_row("Elapsed", f"{stats.elapsed_seconds:.2f} s")
    return table


@app.callback()
def main_callback(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase diagnostic verbosity (-v INFO, -vv DEBUG) on stderr.",
        ),
    ] = 0,
) -> None:
    """findex — streaming full-text search engine CLI."""
    configure_logging(verbose)


@app.command("crawl")
def crawl_cmd(
    seed_url: Annotated[
        str,
        typer.Argument(help="Starting seed URL to crawl (must be permitted by robots.txt)."),
    ],
    max_pages: Annotated[
        int,
        typer.Option("--max-pages", min=1, help="Maximum number of pages to fetch."),
    ] = 500,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-c", min=1, help="Maximum total concurrent requests."),
    ] = 10,
    per_host: Annotated[
        int,
        typer.Option("--per-host", min=1, help="Maximum concurrent requests per host."),
    ] = 2,
    delay: Annotated[
        float,
        typer.Option("--delay", min=0.0, help="Minimum delay (seconds) between requests per host."),
    ] = 0.05,
    timeout: Annotated[
        float,
        typer.Option("--timeout", min=0.1, help="Per-request timeout in seconds."),
    ] = 10.0,
    allow_domain: Annotated[
        list[str] | None,
        typer.Option("--allow-domain", help="Allowed domain/host (repeatable; defaults to seed)."),
    ] = None,
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output JSONL file written streamingly as pages arrive."),
    ] = Path("data/crawl.jsonl"),
    log_file: Annotated[
        Path,
        typer.Option("--log-file", help="Path to per-URL crawl log file."),
    ] = Path("crawl.log"),
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Run asyncio event loop in debug mode to detect blocking."),
    ] = False,
) -> None:
    """Politely crawl web pages asynchronously and stream documents to JSONL."""
    out.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    stats = CrawlStats()
    err_console = Console(stderr=True)
    out_console = Console()

    async def _run_crawl() -> int:
        written = 0
        with (
            out.open("w", encoding="utf-8") as out_handle,
            log_file.open("w", encoding="utf-8") as log_handle,
            Live(
                _render_crawl_table(stats, max_pages),
                console=err_console,
                refresh_per_second=8,
                transient=True,
            ) as live,
        ):

            def _write_log(line: str) -> None:
                log_handle.write(line + "\n")
                live.update(_render_crawl_table(stats, max_pages))

            async for page in crawl(
                [seed_url],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                min_delay=delay,
                allowed_domains=allow_domain,
                timeout=timeout,
                stats=stats,
                on_log=_write_log,
            ):
                record = {
                    "id": page.url,
                    "url": page.url,
                    "title": page.title,
                    "text": page.text,
                    "fetched_at": page.fetched_at,
                }
                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_handle.flush()
                written += 1
                live.update(_render_crawl_table(stats, max_pages))
        return written

    try:
        written_count = asyncio.run(_run_crawl(), debug=debug)
    except OSError as exc:
        _fail(f"I/O error during crawl: {exc}")

    summary = Table(title="Crawl Complete", show_header=False)
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("Seed URL", seed_url)
    summary.add_row("Pages saved", f"{written_count:,}")
    summary.add_row("Throughput", f"{stats.pages_per_second:.1f} pages/s")
    summary.add_row("Errors", str(stats.errors))
    summary.add_row("Skipped (robots)", str(stats.skipped_robots))
    summary.add_row("Output JSONL", str(out))
    summary.add_row("Log file", str(log_file))
    summary.add_row("Elapsed", f"{stats.elapsed_seconds:.2f} s")
    out_console.print(summary)


@app.command("index")
def index_cmd(
    corpus: Annotated[
        Path,
        typer.Argument(help="Path to corpus directory or .jsonl/.txt file."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output path for persisted index."),
    ] = Path("index.bin"),
    positions: Annotated[
        bool,
        typer.Option("--positions", help="Store token positions (enables phrase queries)."),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Index only the first N documents."),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            min=1,
            help="Number of worker threads/processes (default: CPU count).",
        ),
    ] = None,
    executor: Annotated[
        ExecutorChoice,
        typer.Option(
            "--executor",
            "-e",
            help="Execution backend: serial, threads, or processes (default: processes).",
        ),
    ] = ExecutorChoice.processes,
) -> None:
    """Build and persist an inverted index with a Rich progress bar."""
    if not corpus.exists():
        _fail(f"Corpus path not found: {corpus}")

    num_workers = workers if workers is not None else max(1, os.cpu_count() or 1)
    exec_name: ExecutorName = (
        "threads"
        if executor == ExecutorChoice.threads
        else ("processes" if executor == ExecutorChoice.processes else "serial")
    )

    err_console = Console(stderr=True)
    out_console = Console()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.completed:,} docs"),
            TimeElapsedColumn(),
            console=err_console,
            transient=True,
        ) as progress:
            task_id = progress.add_task(
                f"Indexing ({exec_name}, {num_workers}w)...",
                total=limit,
            )

            def _on_doc(total_done: int) -> None:
                progress.update(task_id, completed=total_done)

            profile = build_index_profiled(
                corpus,
                limit=limit,
                with_positions=positions,
                on_document=_on_doc,
                workers=num_workers,
                executor=exec_name,
            )
            save_index(profile.index, out)
    except OSError as exc:
        _fail(f"I/O error while building index: {exc}")

    idx = profile.index
    size_mb = out.stat().st_size / (1024 * 1024)
    logger.info(
        "Indexed %d docs (%d terms) with %s(%d) into %s in %.2fs",
        idx.num_docs,
        idx.num_terms,
        exec_name,
        num_workers,
        out,
        profile.wall_seconds,
    )

    summary = Table(title="Index Built", show_header=False)
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("Executor", f"{exec_name} ({num_workers} workers, {profile.chunks} chunks)")
    summary.add_row("Documents", f"{idx.num_docs:,}")
    summary.add_row("Unique terms", f"{idx.num_terms:,}")
    summary.add_row("Avg doc length", f"{idx.avg_doc_length:.1f} tokens")
    summary.add_row("Positions stored", "yes" if idx.has_positions else "no")
    summary.add_row("Output file", f"{out} ({size_mb:.2f} MB)")
    summary.add_row(
        "Wall / Merge",
        f"{profile.wall_seconds:.2f} s / {profile.merge_seconds:.2f} s",
    )
    summary.add_row("CPU time", f"{profile.cpu_seconds:.2f} s")
    out_console.print(summary)


@app.command("search")
def search_cmd(
    index_path: Annotated[
        Path,
        typer.Argument(help="Path to persisted index file (e.g. index.bin)."),
    ],
    query: Annotated[
        str,
        typer.Argument(help="Boolean/phrase query string."),
    ],
    k: Annotated[
        int,
        typer.Option("--k", "-k", min=1, help="Maximum number of ranked results."),
    ] = 10,
    scorer: Annotated[
        ScorerChoice,
        typer.Option("--scorer", help="Scoring algorithm: bm25 or tfidf."),
    ] = ScorerChoice.bm25,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON lines to stdout for piping."),
    ] = False,
) -> None:
    """Search a persisted index and render ranked results with snippets."""
    if not index_path.exists():
        _fail(f"Index file not found: {index_path}")

    scorer_name: ScorerName = "tfidf" if scorer == ScorerChoice.tfidf else "bm25"

    try:
        with open_index(index_path) as idx:
            results = run_search(
                idx,
                query,
                scorer=make_scorer(scorer_name),
                k=k,
                include_snippets=True,
                highlight_style="markdown" if json_output else "rich",
            )
    except QueryParseError as exc:
        _fail(f"Invalid query syntax: {exc}")
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        _fail(str(exc))

    if json_output:
        for rank, res in enumerate(results, start=1):
            payload: dict[str, int | float | str] = {
                "rank": rank,
                "doc_id": res.doc_id,
                "score": round(res.score, 6),
                "title": res.title,
                "snippet": res.snippet,
            }
            sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return

    out_console = Console()
    if not results:
        out_console.print(f"No matches for query: {query!r}")
        return

    table = Table(
        title=f"Search Results ({scorer_name.upper()}, top {len(results)})",
        show_lines=True,
    )
    table.add_column("#", justify="right", style="bold")
    table.add_column("Doc", justify="right", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Title & Snippet")

    for rank, res in enumerate(results, start=1):
        detail = f"[bold]{escape_rich(res.title)}[/bold]"
        if res.snippet:
            detail = f"{detail}\n{res.snippet}"
        table.add_row(str(rank), str(res.doc_id), f"{res.score:.4f}", detail)

    out_console.print(table)


@app.command("stats")
def stats_cmd(
    target: Annotated[
        Path,
        typer.Argument(help="Path to persisted index file or corpus directory."),
    ],
    top: Annotated[
        int,
        typer.Option("--top", min=1, help="Number of top terms to display."),
    ] = 15,
) -> None:
    """Display corpus or index statistics in Rich tables."""
    if not target.exists():
        _fail(f"Path not found: {target}")

    try:
        if target.is_dir() or target.suffix.lower() in {".jsonl", ".txt"}:
            stats = compute_stats_lazy(target, top_n=top)
        else:
            with open_index(target) as idx:
                stats = compute_index_stats(idx, top_n=top)
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        _fail(str(exc))

    out_console = Console()
    summary = Table(title=f"findex Stats ({stats.mode})", show_header=False)
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("Documents", f"{stats.documents:,}")
    summary.add_row("Total tokens", f"{stats.total_tokens:,}")
    summary.add_row("Vocabulary size", f"{stats.vocab_size:,}")
    avg_len = stats.total_tokens / stats.documents if stats.documents > 0 else 0.0
    summary.add_row("Avg doc length", f"{avg_len:.1f} tokens")
    summary.add_row("Elapsed", f"{stats.elapsed_seconds:.3f} s")
    out_console.print(summary)

    terms_table = Table(title=f"Top-{len(stats.top_terms)} Terms")
    terms_table.add_column("Rank", justify="right", style="bold")
    terms_table.add_column("Term", style="cyan")
    terms_table.add_column("Occurrences", justify="right", style="green")
    for rank, (term, count) in enumerate(stats.top_terms, start=1):
        terms_table.add_row(str(rank), term, f"{count:,}")
    out_console.print(terms_table)
