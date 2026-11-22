"""Tests for query parser AST trees, TF-IDF/BM25 scorers, @timed, cache, and snippets."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from findex.decorators import timed
from findex.index import build_index
from findex.models import Index, Posting, SearchResult
from findex.query import And, Not, Or, Phrase, QueryParseError, Term, parse
from findex.ranking import BM25, Scorer, TfIdf, make_scorer
from findex.search import search
from findex.snippets import make_snippet


def test_parser_precedence_and_tree_equality() -> None:
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))
    assert parse("a b OR c") == Or(And(Term("a"), Term("b")), Term("c"))


def test_parser_parentheses_not_phrase_and_operators() -> None:
    parsed = parse('python AND (async OR await) NOT java "event loop"')
    expected = (
        Term("python") & (Term("async") | Term("await")) & ~Term("java") & Phrase(("event", "loop"))
    )
    assert parsed == expected
    assert ~Term("java") == Not(Term("java"))


@pytest.mark.parametrize("bad_query", ["", "   ", "(a OR b", '"unclosed phrase', '""', "AND a"])
def test_parser_invalid_syntax_raises(bad_query: str) -> None:
    with pytest.raises(QueryParseError):
        parse(bad_query)


def test_phrase_query_matches_adjacent_only(tiny_corpus: Path, tiny_index: Index) -> None:
    assert parse('"event loop"').evaluate(tiny_index) == {0, 1, 2}
    assert parse('"python async"').evaluate(tiny_index) == {0}
    assert parse('"python loop"').evaluate(tiny_index) == set()

    no_pos = build_index(tiny_corpus, with_positions=False)
    with pytest.raises(ValueError, match="--positions"):
        parse('"event loop"').evaluate(no_pos)


def test_scorers_satisfy_protocol_and_factory() -> None:
    bm25 = make_scorer("bm25")
    tfidf = make_scorer("tfidf")
    assert isinstance(bm25, Scorer)
    assert isinstance(tfidf, Scorer)
    assert isinstance(bm25, BM25)
    assert isinstance(tfidf, TfIdf)


def test_sanity_check_rare_term_outranks_common(tiny_index: Index) -> None:
    for scorer in (BM25(k1=1.5, b=0.75), TfIdf()):
        rare = scorer.score("rocket", tiny_index["rocket"][0], tiny_index)
        common = scorer.score("python", tiny_index["python"][0], tiny_index)
        assert rare > common


def test_sanity_check_bm25_tf_saturation(tiny_index: Index) -> None:
    bm25 = BM25(k1=1.5, b=0.75)
    s1 = bm25.score("quantum", Posting(doc_id=3, tf=1), tiny_index)
    s19 = bm25.score("quantum", Posting(doc_id=3, tf=19), tiny_index)
    s20 = bm25.score("quantum", Posting(doc_id=3, tf=20), tiny_index)
    assert s20 > s19 > s1
    assert (s20 - s19) < 0.05 * s1


def test_sanity_check_short_doc_outranks_long_doc(tiny_index: Index) -> None:
    results = search(tiny_index, "quantum", scorer=BM25(k1=1.5, b=0.75), k=5)
    assert [r.doc_id for r in results] == [3, 4]
    assert results[0].score > results[1].score
    assert sorted(results)[0] == results[1]
    assert isinstance(results[0], SearchResult)


def test_timed_preserves_metadata_and_logs_debug(caplog: pytest.LogCaptureFixture) -> None:
    @timed
    def sample_fn(x: int) -> int:
        """Sample docstring."""
        return x * 2

    assert sample_fn.__name__ == "sample_fn"
    assert sample_fn.__doc__ == "Sample docstring."
    with caplog.at_level(logging.DEBUG, logger="findex.timing"):
        assert sample_fn(5) == 10
    assert any("sample_fn took" in rec.message for rec in caplog.records)


def test_lru_cache_hit_on_repeated_query(tiny_index: Index) -> None:
    tiny_index.clear_cache()
    _ = search(tiny_index, "python AND (async OR await)", scorer=BM25(), k=5)
    assert tiny_index.cache_info().misses == 1
    assert tiny_index.cache_info().hits == 0

    _ = search(tiny_index, "python AND (async OR await)", scorer=BM25(), k=5)
    assert tiny_index.cache_info().misses == 1
    assert tiny_index.cache_info().hits == 1


def test_snippets_markdown_and_rich_escaping() -> None:
    text = ("A " * 80) + "NASA [mission-7] space shuttle launch" + (" B" * 80)
    md = make_snippet(text, ["space", "shuttle"], window=40, highlight_style="markdown")
    assert "**space**" in md and "**shuttle**" in md
    assert md.startswith("...") and md.endswith("...")

    rich_snip = make_snippet(text, ["space", "shuttle"], window=40, highlight_style="rich")
    assert "[bold yellow]space[/bold yellow]" in rich_snip
    assert "\\[mission-7]" in rich_snip
