"""Ranked & Boolean search over a persisted inverted index (Lab 2–4)."""

from __future__ import annotations

import argparse
import heapq
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

from findex.decorators import timed
from findex.models import DocId, Index, Posting, SearchResult
from findex.query import parse
from findex.ranking import BM25, Scorer, ScorerName, make_scorer
from findex.snippets import HighlightStyle, load_document_text, make_snippet
from findex.store import open_index
from findex.tokenize import tokenize

EngineType = Literal["merge", "set"]

logger = logging.getLogger(__name__)


def posting_doc_ids(postings: list[Posting]) -> list[DocId]:
    """Extract sorted doc_ids from a term's postings list."""
    return [p.doc_id for p in postings]


def intersect_sorted_merge(a: list[DocId], b: list[DocId]) -> list[DocId]:
    """Two-pointer sorted intersection (AND) in O(len(a) + len(b))."""
    result: list[DocId] = []
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


def union_sorted_merge(a: list[DocId], b: list[DocId]) -> list[DocId]:
    """Two-pointer sorted union (OR) in O(len(a) + len(b))."""
    result: list[DocId] = []
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


def difference_sorted_merge(a: list[DocId], excluded: list[DocId]) -> list[DocId]:
    """Two-pointer sorted difference (a AND NOT excluded) in O(len(a) + len(excluded))."""
    if not a:
        return []
    if not excluded:
        return list(a)
    result: list[DocId] = []
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


def _combine_and(left: list[DocId], right: list[DocId], engine: EngineType) -> list[DocId]:
    if engine == "merge":
        return intersect_sorted_merge(left, right)
    return sorted(set(left) & set(right))


def _combine_or(left: list[DocId], right: list[DocId], engine: EngineType) -> list[DocId]:
    if engine == "merge":
        return union_sorted_merge(left, right)
    return sorted(set(left) | set(right))


def _term_docs(index: Index, raw_term: str) -> list[DocId]:
    tokens = list(tokenize(raw_term))
    if not tokens:
        return []
    docs = posting_doc_ids(index.postings.get(tokens[0], []))
    for extra in tokens[1:]:
        docs = intersect_sorted_merge(docs, posting_doc_ids(index.postings.get(extra, [])))
    return docs


def execute_query(
    index: Index,
    query: str,
    engine: EngineType = "merge",
) -> list[DocId]:
    """Evaluate a simple Lab 2 Boolean query with merge or set engine."""
    parts = query.strip().split()
    if not parts:
        return []

    all_doc_ids = sorted(index.doc_meta.keys())
    current: list[DocId] | None = None
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


def _score_key(pair: tuple[DocId, float]) -> float:
    return pair[1]


@timed
def search(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    k: int = 10,
    include_snippets: bool = True,
    highlight_style: HighlightStyle = "markdown",
) -> list[SearchResult]:
    """Parse ``query``, evaluate via cached AST, score matched docs, and return top-k."""
    active_scorer: Scorer = BM25() if scorer is None else scorer
    before_hits = index.cache_info().hits
    matched_tuple = index.evaluate_query_cached(query)
    after_info = index.cache_info()
    status = "HIT" if after_info.hits > before_hits else "MISS"
    logger.info(
        "query=%r cache=%s (hits=%d, misses=%d) matched=%d",
        query,
        status,
        after_info.hits,
        after_info.misses,
        len(matched_tuple),
    )

    if not matched_tuple:
        return []

    matched_set = set(matched_tuple)
    ast_node = parse(query)
    pos_terms = ast_node.positive_terms()

    scores: dict[DocId, float] = defaultdict(float)
    if pos_terms:
        seen_terms = list(dict.fromkeys(pos_terms))
        for term in seen_terms:
            if term not in index:
                continue
            for posting in index[term]:
                if posting.doc_id in matched_set:
                    scores[posting.doc_id] += active_scorer.score(term, posting, index)
    else:
        for doc_id in matched_tuple:
            scores[doc_id] = 0.0

    top_pairs = heapq.nlargest(k, scores.items(), key=_score_key)

    results: list[SearchResult] = []
    for doc_id, score in top_pairs:
        meta = index.doc_meta[doc_id]
        snippet = ""
        if include_snippets:
            full_text = load_document_text(meta)
            snippet = make_snippet(full_text, pos_terms, highlight_style=highlight_style)
        results.append(
            SearchResult(
                score=score,
                doc_id=doc_id,
                title=meta.title,
                snippet=snippet,
            )
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m findex.search",
        description="Search a persisted inverted index with BM25/TF-IDF ranking and snippets.",
    )
    parser.add_argument("index_path", type=Path, help="Path to persisted index file")
    parser.add_argument("query", type=str, help="Query string")
    parser.add_argument(
        "--scorer",
        choices=["bm25", "tfidf"],
        default="bm25",
        help="Ranking scorer: bm25 (default) or tfidf",
    )
    parser.add_argument("-k", "--limit", type=int, default=10, help="Max results (default: 10)")
    parser.add_argument("--no-snippets", action="store_true", help="Disable snippet extraction")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s]: %(message)s")
    args = build_parser().parse_args(argv)
    scorer_name: ScorerName = "tfidf" if args.scorer == "tfidf" else "bm25"

    t0 = time.perf_counter()
    with open_index(args.index_path) as index:
        results = search(
            index,
            args.query,
            scorer=make_scorer(scorer_name),
            k=args.limit,
            include_snippets=not args.no_snippets,
        )
        elapsed = time.perf_counter() - t0
        print(f"Index:        {index!r}")
        print(f"Scorer:       {scorer_name}")
        print(f"Query:        {args.query!r}")
        print(f"Returned:     {len(results)} (top-{args.limit})")
        print(f"Total time:   {elapsed:.4f} s\n")
        for rank, res in enumerate(results, start=1):
            print(f"{rank}. {res}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
