"""Lab 3 evaluation: 3 ranking sanity checks, cache hit demo, and Precision@5 table."""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import Final

from findex.models import Index, Posting
from findex.ranking import BM25, TfIdf
from findex.search import search
from findex.store import open_index

EVAL_QUERIES: Final[list[tuple[str, tuple[str, ...], str]]] = [
    ("nasa OR shuttle OR orbit", ("sci.space",), "Spaceflight & orbital missions"),
    ("clipper OR encryption OR escrow", ("sci.crypt",), "Cryptography & Clipper chip"),
    ("nhl OR hockey OR playoffs", ("rec.sport.hockey",), "Ice hockey & NHL playoffs"),
    ("motorcycle OR helmet OR kawasaki", ("rec.motorcycles",), "Motorcycles & riding gear"),
    ("jpeg OR gif OR polygon", ("comp.graphics",), "Computer graphics formats"),
    ("engine OR transmission OR honda", ("rec.autos",), "Automobiles & drivetrains"),
    ("cancer OR medical OR patient", ("sci.med",), "Medicine & clinical topics"),
    ("motif OR widget OR x11", ("comp.windows.x",), "X Window System programming"),
    ("baseball OR pitcher OR inning", ("rec.sport.baseball",), "Baseball games & stats"),
    ("circuit OR voltage OR amplifier", ("sci.electronics",), "Electronics & circuits"),
]


def _is_relevant(source_id: str, expected_groups: tuple[str, ...]) -> bool:
    return any(group in source_id for group in expected_groups)


def run_sanity_checks(index: Index) -> str:
    """Demonstrate the 3 required BM25/TF-IDF ranking sanity checks on the real corpus."""
    bm25 = BM25(k1=1.5, b=0.75)
    tfidf = TfIdf()
    lines: list[str] = []

    rare_term = ""
    common_term = ""
    for term, plist in index.items():
        if len(term) >= 5 and term.isalpha():
            if not rare_term and 3 <= len(plist) <= 15:
                rare_term = term
            if not common_term and len(plist) >= max(2, int(index.num_docs * 0.25)):
                common_term = term
        if rare_term and common_term:
            break

    if not rare_term or not common_term:
        ordered = sorted(index.items(), key=lambda kv: len(kv[1]))
        rare_term = ordered[0][0]
        common_term = ordered[-1][0]

    rare_post = index[rare_term][0]
    common_post = next(
        (p for p in index[common_term] if p.doc_id != rare_post.doc_id),
        index[common_term][0],
    )
    bm25_rare = bm25.score(rare_term, rare_post, index)
    bm25_common = bm25.score(common_term, common_post, index)
    tfidf_rare = tfidf.score(rare_term, rare_post, index)
    tfidf_common = tfidf.score(common_term, common_post, index)
    lines.append(
        f"1. **Rare term outranks common term:** "
        f"rare `{rare_term}` (`df={index.df(rare_term)}`, doc `{rare_post.doc_id}`) "
        f"scores **BM25={bm25_rare:.4f}** / TF-IDF={tfidf_rare:.4f} vs common `{common_term}` "
        f"(`df={index.df(common_term)}`, doc `{common_post.doc_id}`) "
        f"**BM25={bm25_common:.4f}** / TF-IDF={tfidf_common:.4f}."
    )

    sample_doc_id = next(iter(index.doc_lengths))
    s1 = bm25.score(rare_term, Posting(doc_id=sample_doc_id, tf=1), index)
    s19 = bm25.score(rare_term, Posting(doc_id=sample_doc_id, tf=19), index)
    s20 = bm25.score(rare_term, Posting(doc_id=sample_doc_id, tf=20), index)
    delta_20 = s20 - s19
    pct_gain = (delta_20 / s19 * 100.0) if s19 > 0 else 0.0
    lines.append(
        f"2. **BM25 TF saturation (`k1=1.5`):** for `{rare_term}` in doc `{sample_doc_id}` "
        f"(`len={index.doc_length(sample_doc_id)}`), `tf=1` → **{s1:.4f}**, "
        f"`tf=19` → **{s19:.4f}**, `tf=20` → **{s20:.4f}** "
        f"(20th repetition adds only **+{delta_20:.4f}**, i.e. **+{pct_gain:.2f}%**)."
    )

    chosen_term = rare_term
    short_post: Posting | None = None
    long_post: Posting | None = None
    best_ratio = 0.0
    max_df = max(10, index.num_docs // 5)
    for term, plist in index.items():
        if not term.isalpha() or not (10 <= len(plist) <= max_df):
            continue
        tf1 = [p for p in plist if p.tf == 1]
        if len(tf1) < 2:
            continue
        shortest = min(tf1, key=lambda p: index.doc_length(p.doc_id))
        longest = max(tf1, key=lambda p: index.doc_length(p.doc_id))
        s_len = max(index.doc_length(shortest.doc_id), 1)
        l_len = index.doc_length(longest.doc_id)
        ratio = l_len / s_len
        if ratio > best_ratio and s_len >= 10:
            best_ratio = ratio
            chosen_term = term
            short_post = shortest
            long_post = longest
            if ratio >= 20.0:
                break

    if short_post is not None and long_post is not None:
        s_score = bm25.score(chosen_term, short_post, index)
        l_score = bm25.score(chosen_term, long_post, index)
        lines.append(
            f"3. **Short doc outranks very long doc (`b=0.75`):** for `{chosen_term}` with `tf=1`, "
            f"short doc `{short_post.doc_id}` (`len={index.doc_length(short_post.doc_id)}`) "
            f"scores **BM25={s_score:.4f}** vs long doc `{long_post.doc_id}` "
            f"(`len={index.doc_length(long_post.doc_id)}`) **BM25={l_score:.4f}** "
            f"(avg doc length = {index.avg_doc_length:.1f})."
        )

    return "\n".join(lines)


def run_cache_demo(index: Index, query: str) -> str:
    """Run a query twice and capture @timed (DEBUG) + lru_cache hit log lines."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s]: %(message)s"))

    loggers = [logging.getLogger("findex.timing"), logging.getLogger("findex.search")]
    prev_levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.DEBUG)
        lg.addHandler(handler)

    try:
        index.clear_cache()
        search(index, query, scorer=BM25(), k=5, include_snippets=False)
        search(index, query, scorer=BM25(), k=5, include_snippets=False)
    finally:
        for lg, lvl in zip(loggers, prev_levels, strict=True):
            lg.removeHandler(handler)
            lg.setLevel(lvl)

    return stream.getvalue().strip()


def run_precision_eval(index: Index) -> str:
    """Compute Precision@5 for TF-IDF vs BM25 across 10 hand-labeled queries."""
    tfidf = TfIdf()
    bm25 = BM25(k1=1.5, b=0.75)

    rows: list[str] = [
        "| # | Query | Target group(s) | TF-IDF P@5 | BM25 P@5 |",
        "|---:|---|---|---:|---:|",
    ]
    tfidf_scores: list[float] = []
    bm25_scores: list[float] = []

    for idx_no, (q, groups, _desc) in enumerate(EVAL_QUERIES, start=1):
        res_tfidf = search(index, q, scorer=tfidf, k=5, include_snippets=False)
        res_bm25 = search(index, q, scorer=bm25, k=5, include_snippets=False)

        p5_tfidf = (
            sum(1 for r in res_tfidf if _is_relevant(index.doc_meta[r.doc_id].source_id, groups))
            / 5.0
        )
        p5_bm25 = (
            sum(1 for r in res_bm25 if _is_relevant(index.doc_meta[r.doc_id].source_id, groups))
            / 5.0
        )
        tfidf_scores.append(p5_tfidf)
        bm25_scores.append(p5_bm25)

        group_label = ", ".join(f"`{g}`" for g in groups)
        rows.append(f"| {idx_no} | `{q}` | {group_label} | {p5_tfidf:.2f} | {p5_bm25:.2f} |")

    avg_tfidf = sum(tfidf_scores) / len(tfidf_scores)
    avg_bm25 = sum(bm25_scores) / len(bm25_scores)
    rows.append(f"| **Avg** | **10 queries** | — | **{avg_tfidf:.2f}** | **{avg_bm25:.2f}** |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m findex.eval",
        description="Run Lab 3 sanity checks, cache hit demo, and Precision@5 evaluation.",
    )
    parser.add_argument("index_path", type=Path, help="Path to saved index file")
    args = parser.parse_args(argv)

    with open_index(args.index_path) as index:
        print(run_sanity_checks(index))
        print()
        print(run_cache_demo(index, "(space OR nasa OR shuttle OR orbit) NOT mars"))
        print()
        print(run_precision_eval(index))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
