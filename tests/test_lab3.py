"""Tests for Lab 3: Index data model, open_index, Scorer/BM25, query parser, cache, snippets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from findex.decorators import timed
from findex.index import build_index
from findex.models import Posting, SearchResult
from findex.query import And, Not, Or, Phrase, QueryParseError, Term, parse
from findex.ranking import BM25, Scorer, TfIdf
from findex.search import search
from findex.snippets import make_snippet
from findex.store import open_index, save_index


def _sample_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.jsonl"
    long_filler = " ".join(["filler"] * 120)
    rows = [
        {"id": "sci.space/1", "text": "Python async event loop tutorial for fast servers"},
        {"id": "sci.space/2", "text": "Python await event loop and coroutines"},
        {"id": "comp.lang/3", "text": "Java event loop without python"},
        {"id": "short/4", "text": "Quantum rocket"},
        {"id": "long/5", "text": f"Quantum {long_filler} engine"},
        {"id": "common/6", "text": "Python common common common"},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return corpus


def test_index_data_model_and_mapping(tmp_path: Path) -> None:
    idx = build_index(_sample_corpus(tmp_path), with_positions=True)

    assert isinstance(idx, Mapping)
    assert len(idx) > 0
    assert "python" in idx
    assert "nonexistent_xyz" not in idx
    assert idx.num_docs == 6
    assert idx.avg_doc_length > 0
    assert idx.df("python") == 4
    assert "Index(terms=" in repr(idx)
    assert set(iter(idx)) == set(idx.keys())

    try:
        _ = idx["nonexistent_xyz"]
        raise AssertionError("Expected KeyError")
    except KeyError:
        pass


def test_open_index_context_manager_cleans_up_on_exception(tmp_path: Path) -> None:
    idx = build_index(_sample_corpus(tmp_path), with_positions=True)
    path = tmp_path / "index.bin"
    save_index(idx, path)

    captured_index = None
    try:
        with open_index(path) as loaded:
            captured_index = loaded
            assert not loaded.closed
            assert "python" in loaded
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass

    assert captured_index is not None
    assert captured_index.closed


def test_query_parser_trees_and_operator_overloading() -> None:
    # Precedence: implicit AND binds tighter than OR
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))

    # Parentheses, NOT, and quoted phrase
    parsed = parse('python AND (async OR await) NOT java "event loop"')
    expected = (
        Term("python")
        & (Term("async") | Term("await"))
        & ~Term("java")
        & Phrase(("event", "loop"))
    )
    assert parsed == expected
    assert ~Term("java") == Not(Term("java"))

    # Invalid syntax raises QueryParseError
    for bad in ["", "(a OR b", '"unclosed phrase']:
        try:
            parse(bad)
            raise AssertionError(f"Expected QueryParseError for {bad!r}")
        except QueryParseError:
            pass


def test_phrase_query_requires_adjacent_positions(tmp_path: Path) -> None:
    corpus = tmp_path / "phrase.jsonl"
    rows = [
        {"id": "d0", "text": "The event loop runs callbacks"},
        {"id": "d1", "text": "The event happened inside a loop"},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    idx = build_index(corpus, with_positions=True)

    assert parse('"event loop"').evaluate(idx) == {0}
    assert parse("event loop").evaluate(idx) == {0, 1}


def test_scorers_and_ranking_sanity_checks(tmp_path: Path) -> None:
    idx = build_index(_sample_corpus(tmp_path), with_positions=True)
    bm25 = BM25(k1=1.5, b=0.75)
    tfidf = TfIdf()

    assert isinstance(bm25, Scorer)
    assert isinstance(tfidf, Scorer)

    # 1. Rare term ("rocket", df=1) outranks common term ("python", df=4)
    rare_score = bm25("rocket", idx["rocket"][0], idx)
    common_score = bm25("python", idx["python"][0], idx)
    assert rare_score > common_score

    # 2. BM25 TF saturation: 20th repetition adds almost nothing vs 1st
    s1 = bm25("quantum", Posting(doc_id=3, tf=1), idx)
    s19 = bm25("quantum", Posting(doc_id=3, tf=19), idx)
    s20 = bm25("quantum", Posting(doc_id=3, tf=20), idx)
    assert s20 > s19 > s1
    assert (s20 - s19) < 0.05 * s1

    # 3. Short doc with 1 match outranks very long doc with 1 match
    results = search(idx, "quantum", scorer=bm25, k=5)
    assert [r.doc_id for r in results] == [3, 4]
    assert results[0].score > results[1].score

    # SearchResult ordering works with sorted()
    ordered = sorted(results)
    assert ordered[0].score <= ordered[-1].score
    assert isinstance(ordered[0], SearchResult)


def test_timed_preserves_metadata_and_lru_cache_hits(tmp_path: Path) -> None:
    @timed
    def sample_fn(x: int) -> int:
        """Sample docstring."""
        return x + 1

    assert sample_fn.__name__ == "sample_fn"
    assert sample_fn.__doc__ == "Sample docstring."
    assert sample_fn(2) == 3

    idx = build_index(_sample_corpus(tmp_path), with_positions=True)
    idx.clear_cache()
    _ = search(idx, "python AND (async OR await)", scorer=BM25(), k=5)
    info1 = idx.cache_info()
    assert info1.misses == 1 and info1.hits == 0

    _ = search(idx, "python AND (async OR await)", scorer=BM25(), k=5)
    info2 = idx.cache_info()
    assert info2.misses == 1 and info2.hits == 1


def test_snippets_highlight_terms_within_window() -> None:
    text = ("A " * 80) + "Python async event loop makes concurrency clean" + (" B" * 80)
    snip = make_snippet(text, ["event", "loop"], window=40)
    assert "**event**" in snip
    assert "**loop**" in snip
    assert snip.startswith("...") and snip.endswith("...")
