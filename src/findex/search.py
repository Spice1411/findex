"""Boolean search over a persisted inverted index (Lab 2 M2)."""

from __future__ import annotations

import argparse
import gc
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Literal

from findex.index import InvertedIndex, Posting
from findex.store import load_index
from findex.tokenize import tokenize

EngineType = Literal["merge", "set"]


def posting_doc_ids(postings: list[Posting]) -> list[int]:
    """Extract sorted doc_ids from a term's postings list."""
    return [p.doc_id for p in postings]


def intersect_sorted_merge(a: list[int], b: list[int]) -> list[int]:
    """Two-pointer sorted intersection (AND) in O(len(a) + len(b))."""
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        va = a[i]
        vb = b[j]
        if va == vb:
            result.append(va)
            i += 1
            j += 1
        elif va < vb:
            i += 1
        else:
            j += 1
    return result


def union_sorted_merge(a: list[int], b: list[int]) -> list[int]:
    """Two-pointer sorted union (OR) in O(len(a) + len(b))."""
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        va = a[i]
        vb = b[j]
        if va == vb:
            result.append(va)
            i += 1
            j += 1
        elif va < vb:
            result.append(va)
            i += 1
        else:
            result.append(vb)
            j += 1
    while i < len_a:
        result.append(a[i])
        i += 1
    while j < len_b:
        result.append(b[j])
        j += 1
    return result


def difference_sorted_merge(a: list[int], excluded: list[int]) -> list[int]:
    """Two-pointer sorted difference (a AND NOT excluded) in O(len(a) + len(excluded))."""
    if not a:
        return []
    if not excluded:
        return list(a)
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_ex = len(excluded)
    while i < len_a and j < len_ex:
        va = a[i]
        vex = excluded[j]
        if va == vex:
            i += 1
            j += 1
        elif va < vex:
            result.append(va)
            i += 1
        else:
            j += 1
    while i < len_a:
        result.append(a[i])
        i += 1
    return result


def _combine_and(left: list[int], right: list[int], engine: EngineType) -> list[int]:
    if engine == "merge":
        return intersect_sorted_merge(left, right)
    return sorted(set(left) & set(right))


def _combine_or(left: list[int], right: list[int], engine: EngineType) -> list[int]:
    if engine == "merge":
        return union_sorted_merge(left, right)
    return sorted(set(left) | set(right))


def _combine_not(left: list[int], right: list[int], engine: EngineType) -> list[int]:
    if engine == "merge":
        return difference_sorted_merge(left, right)
    return sorted(set(left) - set(right))


def _term_docs(index: InvertedIndex, raw_term: str) -> list[int]:
    tokens = list(tokenize(raw_term))
    if not tokens:
        return []
    docs = posting_doc_ids(index.postings.get(tokens[0], []))
    for extra in tokens[1:]:
        docs = intersect_sorted_merge(docs, posting_doc_ids(index.postings.get(extra, [])))
    return docs


def execute_query(
    index: InvertedIndex,
    query: str,
    engine: EngineType = "merge",
) -> list[int]:
    """Evaluate a Boolean query with implicit AND, explicit AND/OR, and NOT."""
    parts = query.strip().split()
    if not parts:
        return []

    all_doc_ids = sorted(index.doc_meta.keys())
    current: list[int] | None = None
    pending_op: str = "AND"
    negate_next = False

    for token in parts:
        upper = token.upper()
        if upper in {"AND", "OR"}:
            pending_op = upper
            continue
        if upper == "NOT":
            negate_next = True
            continue

        operand_docs = _term_docs(index, token)
        if negate_next:
            operand_docs = difference_sorted_merge(all_doc_ids, operand_docs)
            negate_next = False

        if current is None:
            current = operand_docs
        elif pending_op == "OR":
            current = _combine_or(current, operand_docs, engine=engine)
        else:
            current = _combine_and(current, operand_docs, engine=engine)

        pending_op = "AND"

    if current is None and negate_next:
        return all_doc_ids
    return current or []


def _format_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m findex.search",
        description="Search a persisted inverted index using Boolean operators.",
    )
    parser.add_argument(
        "index_path",
        type=Path,
        help="Path to persisted index file (e.g. index.bin or index.json)",
    )
    parser.add_argument(
        "query",
        type=str,
        help='Boolean query string, e.g. "python memory" or "linux OR windows NOT mac"',
    )
    parser.add_argument(
        "--engine",
        choices=["merge", "set"],
        default="merge",
        help="Evaluation engine: two-pointer merge (default) or Python set operations",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of matching documents to display (default: 10)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()

    index = load_index(args.index_path)
    matched_ids = execute_query(index, args.query, engine=args.engine)

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Engine:       {args.engine}")
    print(f"Query:        {args.query!r}")
    print(f"Total hits:   {len(matched_ids):,}")
    print(f"Elapsed time: {elapsed:.4f} s")
    print(f"Peak memory:  {_format_mb(peak)}")

    for doc_id in matched_ids[: args.limit]:
        meta = index.doc_meta.get(doc_id)
        if meta is not None:
            print(f"  [{doc_id}] {meta.source_id} — {meta.title}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
