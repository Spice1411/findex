"""Inverted index construction and memory study (Lab 2 M1/M4 & Lab 3 M1)."""

from __future__ import annotations

import argparse
import gc
import logging
import shutil
import tempfile
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents, iter_documents_with_offsets
from findex.decorators import timed
from findex.models import (
    CompactArrayPostings,
    CompactIndex,
    DocMeta,
    Index,
    InvertedIndex,
    Posting,
)
from findex.tokenize import tokenize

__all__ = [
    "CompactArrayPostings",
    "CompactIndex",
    "DocMeta",
    "Index",
    "InvertedIndex",
    "Posting",
    "build_compact_index",
    "build_index",
]


class _PlainPosting:
    """Plain class WITHOUT __slots__ (has per-instance __dict__) for memory comparison."""

    def __init__(self, doc_id: int, tf: int) -> None:
        self.doc_id = doc_id
        self.tf = tf


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned[:80]
    return fallback


@timed
def build_index(
    root: Path,
    limit: int | None = None,
    with_positions: bool = False,
) -> Index:
    """Build an Index in a single streaming pass over ``root``."""
    docs = iter_documents_with_offsets(root)
    if limit is not None:
        docs = islice(docs, limit)

    raw_postings: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_num, (doc, byte_offset) in enumerate(docs):
        term_counts: Counter[str] = Counter()
        term_positions: dict[str, list[int]] = defaultdict(list)
        token_count = 0

        for pos, token in enumerate(tokenize(doc.text)):
            token_count += 1
            term_counts[token] += 1
            if with_positions:
                term_positions[token].append(pos)

        doc_lengths[doc_num] = token_count
        doc_meta[doc_num] = DocMeta(
            doc_id=doc_num,
            source_id=doc.doc_id,
            path=doc.path.as_posix(),
            title=_extract_title(doc.text, fallback=doc.doc_id),
            offset=byte_offset,
        )

        for term, tf in term_counts.items():
            pos_tuple = tuple(term_positions[term]) if with_positions else ()
            raw_postings[term].append(Posting(doc_id=doc_num, tf=tf, positions=pos_tuple))

    sorted_postings: dict[str, list[Posting]] = {}
    for term, plist in raw_postings.items():
        plist.sort(key=lambda p: p.doc_id)
        sorted_postings[term] = plist

    return Index(
        postings=sorted_postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        has_positions=with_positions,
    )


def build_compact_index(root: Path, limit: int | None = None) -> CompactIndex:
    """Build an index using array('I') pairs for doc_ids and tfs (zero Posting objects)."""
    docs = iter_documents_with_offsets(root)
    if limit is not None:
        docs = islice(docs, limit)

    raw_pairs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_num, (doc, byte_offset) in enumerate(docs):
        term_counts: Counter[str] = Counter()
        token_count = 0
        for token in tokenize(doc.text):
            token_count += 1
            term_counts[token] += 1

        doc_lengths[doc_num] = token_count
        doc_meta[doc_num] = DocMeta(
            doc_id=doc_num,
            source_id=doc.doc_id,
            path=doc.path.as_posix(),
            title=_extract_title(doc.text, fallback=doc.doc_id),
            offset=byte_offset,
        )
        for term, tf in term_counts.items():
            raw_pairs[term].append((doc_num, tf))

    compact_postings: dict[str, CompactArrayPostings] = {}
    for term, pairs in raw_pairs.items():
        pairs.sort(key=lambda pair: pair[0])
        doc_arr = array("I", (d for d, _ in pairs))
        tf_arr = array("I", (t for _, t in pairs))
        compact_postings[term] = CompactArrayPostings(doc_ids=doc_arr, tfs=tf_arr)

    return CompactIndex(
        postings=compact_postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
    )


def measure_plain_dataclass_build(root: Path, limit: int | None = None) -> tuple[int, float]:
    """Measure build peak memory and time using non-slotted _PlainPosting objects."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()

    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)

    raw: dict[str, list[_PlainPosting]] = defaultdict(list)
    for doc_num, doc in enumerate(docs):
        counts = Counter(tokenize(doc.text))
        for term, tf in counts.items():
            raw[term].append(_PlainPosting(doc_num, tf))

    for plist in raw.values():
        plist.sort(key=lambda p: p.doc_id)

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, elapsed


def measure_slotted_build(
    root: Path,
    limit: int | None = None,
    with_positions: bool = False,
) -> tuple[Index, int, float]:
    """Measure build peak memory and time for slotted frozen dataclass Posting."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    idx = build_index(root, limit=limit, with_positions=with_positions)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return idx, peak, elapsed


def measure_compact_build(root: Path, limit: int | None = None) -> tuple[CompactIndex, int, float]:
    """Measure build peak memory and time for array('I') compact postings."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    cidx = build_compact_index(root, limit=limit)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return cidx, peak, elapsed


def _format_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def run_memory_study(root: Path, limit: int | None = None) -> None:
    """Run Lab 2 M4 memory comparison across plain dataclass, slotted Posting, and array('I')."""
    from findex.store import load_compact_pickle, load_index, save_compact_pickle, save_index

    plain_peak, plain_time = measure_plain_dataclass_build(root, limit=limit)
    slotted_idx, slotted_peak, slotted_time = measure_slotted_build(root, limit=limit)
    compact_idx, compact_peak, compact_time = measure_compact_build(root, limit=limit)

    tmp_dir = Path(tempfile.mkdtemp(prefix="findex_m4_"))
    try:
        slotted_path = tmp_dir / "slotted.pkl"
        compact_path = tmp_dir / "compact.pkl"
        save_index(slotted_idx, slotted_path)
        save_compact_pickle(compact_idx, compact_path)

        slotted_size = slotted_path.stat().st_size
        compact_size = compact_path.stat().st_size

        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        _ = load_index(slotted_path)
        slotted_load_time = time.perf_counter() - t0
        _, slotted_load_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        _ = load_compact_pickle(compact_path)
        compact_load_time = time.perf_counter() - t0
        _, compact_load_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n--- Copy this table into README.md ---")
    print("| Postings representation | Peak memory (build) | Index file size | Load time |")
    print("|---|---:|---:|---:|")
    print(
        f"| `list[Posting]` with plain `@dataclass` / `__dict__` "
        f"| {_format_mb(plain_peak)} ({plain_time:.2f}s) | — | — |"
    )
    print(
        f"| `list[Posting]` with `slots=True` "
        f"| {_format_mb(slotted_peak)} ({slotted_time:.2f}s) "
        f"| {_format_mb(slotted_size)} | {slotted_load_time:.3f} s |"
    )
    print(
        f"| `array('I')` pairs (doc_ids, tfs) — no objects "
        f"| {_format_mb(compact_peak)} ({compact_time:.2f}s) "
        f"| {_format_mb(compact_size)} | {compact_load_time:.3f} s |"
    )
    print(
        f"\nExtra info: slotted load peak={_format_mb(slotted_load_peak)}, "
        f"compact load peak={_format_mb(compact_load_peak)}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m findex.index",
        description="Build and persist an inverted index over a corpus.",
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("data"),
        help="Path to corpus directory or .jsonl file (default: data/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("index.bin"),
        help="Output index file path (default: index.bin)",
    )
    parser.add_argument(
        "--format",
        choices=["pickle", "json"],
        default="pickle",
        help="Persistence format: pickle (default) or json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N documents",
    )
    parser.add_argument(
        "--positions",
        action="store_true",
        help="Record token positions per posting (required for phrase queries)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Measure peak Python RAM using tracemalloc during build",
    )
    parser.add_argument(
        "--memory-study",
        action="store_true",
        help="Run Lab 2 M4 memory study table across representations",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from findex.store import save_index

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s]: %(message)s")
    args = build_parser().parse_args(argv)

    if args.memory_study:
        run_memory_study(args.root, limit=args.limit)
        return 0

    if args.trace:
        gc.collect()
        tracemalloc.start()

    start = time.perf_counter()
    idx = build_index(args.root, limit=args.limit, with_positions=args.positions)
    save_index(idx, args.out, fmt=args.format)
    elapsed = time.perf_counter() - start

    peak = 0
    if args.trace:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    file_size = args.out.stat().st_size
    print(f"Index:             {idx!r}")
    print(f"Avg doc length:    {idx.avg_doc_length:.1f} tokens")
    print(f"Positions stored:  {idx.has_positions}")
    print(f"Format:            {args.format}")
    print(f"Index file size:   {_format_mb(file_size)} ({args.out})")
    print(f"Elapsed time:      {elapsed:.3f} s")
    if args.trace:
        print(f"Peak memory:       {_format_mb(peak)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
