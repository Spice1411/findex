"""Corpus statistics pipeline and eager-vs-lazy benchmark."""

from __future__ import annotations

import argparse
import gc
import logging
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Aggregated statistics and resource measurements for one pass over a corpus."""

    mode: str
    documents: int
    total_tokens: int
    vocab_size: int
    top_terms: list[tuple[str, int]]
    elapsed_seconds: float
    peak_memory_bytes: int


def compute_stats_lazy(root: Path, limit: int | None = None, top_n: int = 50) -> CorpusStats:
    """Stream documents and tokens in a single pass using generators."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()

    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)

    doc_count = 0
    total_tokens = 0
    counts: Counter[str] = Counter()

    for doc in docs:
        doc_count += 1
        for token in tokenize(doc.text):
            total_tokens += 1
            counts[token] += 1

    elapsed = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return CorpusStats(
        mode="lazy (generators)",
        documents=doc_count,
        total_tokens=total_tokens,
        vocab_size=len(counts),
        top_terms=counts.most_common(top_n),
        elapsed_seconds=elapsed,
        peak_memory_bytes=peak_bytes,
    )


def compute_stats_eager(root: Path, limit: int | None = None, top_n: int = 50) -> CorpusStats:
    """Deliberately eager baseline: load all documents and all token lists into memory."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()

    docs_stream = iter_documents(root)
    if limit is not None:
        docs_stream = islice(docs_stream, limit)

    documents = list(docs_stream)
    tokenized_docs = [list(tokenize(doc.text)) for doc in documents]
    counts: Counter[str] = Counter()
    total_tokens = 0
    for token_list in tokenized_docs:
        total_tokens += len(token_list)
        counts.update(token_list)

    elapsed = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return CorpusStats(
        mode="eager (lists)",
        documents=len(documents),
        total_tokens=total_tokens,
        vocab_size=len(counts),
        top_terms=counts.most_common(top_n),
        elapsed_seconds=elapsed,
        peak_memory_bytes=peak_bytes,
    )


def _format_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def print_stats(stats: CorpusStats) -> None:
    """Print human-readable statistics summary."""
    print(f"Mode:             {stats.mode}")
    print(f"Documents:        {stats.documents:,}")
    print(f"Total tokens:     {stats.total_tokens:,}")
    print(f"Vocabulary size:  {stats.vocab_size:,}")
    print(f"Elapsed time:     {stats.elapsed_seconds:.3f} s")
    print(f"Peak memory:      {_format_mb(stats.peak_memory_bytes)}")
    print(f"\nTop-{len(stats.top_terms)} terms:")
    for rank, (term, count) in enumerate(stats.top_terms, start=1):
        print(f"  {rank:>2}. {term:<20} {count:,}")


def print_comparison_table(eager: CorpusStats, lazy: CorpusStats) -> None:
    """Print a Markdown comparison table ready to paste into README.md."""
    ratio = (
        eager.peak_memory_bytes / lazy.peak_memory_bytes
        if lazy.peak_memory_bytes > 0
        else 0.0
    )
    print("\n--- Copy this table into README.md ---")
    print("| Version | Documents | Total tokens | Peak memory | Elapsed |")
    print("|---|---:|---:|---:|---:|")
    for s in (eager, lazy):
        print(
            f"| {s.mode} | {s.documents:,} | {s.total_tokens:,} "
            f"| {_format_mb(s.peak_memory_bytes)} | {s.elapsed_seconds:.3f} s |"
        )
    print(f"\nMemory reduction: {ratio:.1f}x less peak RAM with generators.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m findex.stats",
        description="Compute corpus statistics using a streaming generator pipeline.",
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("data"),
        help="Path to corpus directory (of .txt/.jsonl files) or .jsonl file (default: data/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N documents (uses itertools.islice)",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Run the eager (list-based) version instead of the lazy generator pipeline",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both eager and lazy versions and print a Markdown benchmark table",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    if args.compare:
        eager_stats = compute_stats_eager(args.root, limit=args.limit)
        lazy_stats = compute_stats_lazy(args.root, limit=args.limit)
        print_stats(lazy_stats)
        print_comparison_table(eager_stats, lazy_stats)
        return 0

    if args.eager:
        stats = compute_stats_eager(args.root, limit=args.limit)
    else:
        stats = compute_stats_lazy(args.root, limit=args.limit)

    print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
