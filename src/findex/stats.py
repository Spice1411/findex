"""Corpus & index statistics pipeline (Lab 1 & Lab 4 CLI stats)."""

from __future__ import annotations

import argparse
import gc
import heapq
import logging
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.models import Index
from findex.tokenize import tokenize

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Aggregated statistics and resource measurements for one pass over a corpus or index."""

    mode: str
    documents: int
    total_tokens: int
    vocab_size: int
    top_terms: list[tuple[str, int]]
    elapsed_seconds: float
    peak_memory_bytes: int


def compute_stats_lazy(root: Path, limit: int | None = None, top_n: int = 20) -> CorpusStats:
    """Stream documents and tokens in a single pass using generators."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()

    docs: Iterable[Document] = iter_documents(root)
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

    logger.info("Computed lazy corpus stats: docs=%d tokens=%d", doc_count, total_tokens)
    return CorpusStats(
        mode="lazy (generators)",
        documents=doc_count,
        total_tokens=total_tokens,
        vocab_size=len(counts),
        top_terms=counts.most_common(top_n),
        elapsed_seconds=elapsed,
        peak_memory_bytes=peak_bytes,
    )


def compute_stats_eager(root: Path, limit: int | None = None, top_n: int = 20) -> CorpusStats:
    """Deliberately eager baseline: load all documents and all token lists into memory."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()

    docs_stream: Iterable[Document] = iter_documents(root)
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


def _term_freq_key(pair: tuple[str, int]) -> int:
    return pair[1]


def compute_index_stats(index: Index, top_n: int = 20) -> CorpusStats:
    """Compute Lab 1 style summary statistics directly from a persisted Index."""
    start = time.perf_counter()
    term_totals = ((term, sum(p.tf for p in plist)) for term, plist in index.items())
    top_terms = heapq.nlargest(top_n, term_totals, key=_term_freq_key)
    elapsed = time.perf_counter() - start
    logger.info("Computed index stats for %r in %.3f s", index, elapsed)
    return CorpusStats(
        mode="persisted index",
        documents=index.num_docs,
        total_tokens=index.total_tokens,
        vocab_size=index.num_terms,
        top_terms=top_terms,
        elapsed_seconds=elapsed,
        peak_memory_bytes=0,
    )


def _format_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def format_stats_text(stats: CorpusStats) -> str:
    """Format human-readable statistics summary."""
    lines = [
        f"Mode:             {stats.mode}",
        f"Documents:        {stats.documents:,}",
        f"Total tokens:     {stats.total_tokens:,}",
        f"Vocabulary size:  {stats.vocab_size:,}",
        f"Elapsed time:     {stats.elapsed_seconds:.3f} s",
    ]
    if stats.peak_memory_bytes > 0:
        lines.append(f"Peak memory:      {_format_mb(stats.peak_memory_bytes)}")
    lines.append(f"\nTop-{len(stats.top_terms)} terms:")
    for rank, (term, count) in enumerate(stats.top_terms, start=1):
        lines.append(f"  {rank:>2}. {term:<20} {count:,}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m findex.stats",
        description="Compute corpus statistics using a streaming generator pipeline.",
    )
    parser.add_argument("root", type=Path, nargs="?", default=Path("data"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eager", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    stats = (
        compute_stats_eager(args.root, limit=args.limit)
        if args.eager
        else compute_stats_lazy(args.root, limit=args.limit)
    )
    print(format_stats_text(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
