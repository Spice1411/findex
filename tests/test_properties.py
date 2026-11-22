"""Property-based tests with Hypothesis and a slow integration marker (Lab 4 M2)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import build_index, build_index_from_documents
from findex.search import (
    difference_sorted_merge,
    intersect_sorted_merge,
    search,
    union_sorted_merge,
)
from findex.store import load_index, save_index
from findex.tokenize import tokenize

_doc_text_strategy = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ", min_size=1, max_size=40),
    min_size=1,
    max_size=8,
)


@given(st.text(max_size=200))
@settings(max_examples=100)
def test_property_tokenize_never_yields_empty_and_is_casefolded(text: str) -> None:
    for tok in tokenize(text):
        assert tok != ""
        assert tok == tok.casefold()


@given(
    st.lists(st.integers(min_value=0, max_value=200), max_size=50),
    st.lists(st.integers(min_value=0, max_value=200), max_size=50),
)
@settings(max_examples=100)
def test_property_merge_matches_python_set_oracle(a_raw: list[int], b_raw: list[int]) -> None:
    a = sorted(set(a_raw))
    b = sorted(set(b_raw))

    assert intersect_sorted_merge(a, b) == sorted(set(a) & set(b))
    assert union_sorted_merge(a, b) == sorted(set(a) | set(b))
    assert difference_sorted_merge(a, b) == sorted(set(a) - set(b))


@given(_doc_text_strategy)
@settings(max_examples=40)
def test_property_postings_always_strictly_sorted(texts: list[str]) -> None:
    docs = [
        (Document(doc_id=f"d{i}", path=Path(f"d{i}.txt"), text=t), -1) for i, t in enumerate(texts)
    ]
    idx = build_index_from_documents(docs, with_positions=True)

    for plist in idx.values():
        doc_ids = [p.doc_id for p in plist]
        assert doc_ids == sorted(set(doc_ids))
        for posting in plist:
            assert list(posting.positions) == sorted(set(posting.positions))
            assert posting.tf == len(posting.positions)


@given(_doc_text_strategy, st.sampled_from(["pickle", "json"]))
@settings(max_examples=30, deadline=None)
def test_property_save_load_roundtrip(texts: list[str], fmt: str) -> None:
    docs = [
        (Document(doc_id=f"d{i}", path=Path(f"d{i}.txt"), text=t), -1) for i, t in enumerate(texts)
    ]
    idx = build_index_from_documents(docs, with_positions=True)
    chosen = "json" if fmt == "json" else "pickle"

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / (f"idx.{chosen}" if chosen == "json" else "idx.bin")
        save_index(idx, out_path, fmt=chosen)
        loaded = load_index(out_path, fmt=chosen)
        assert loaded == idx


@pytest.mark.slow
def test_slow_large_synthetic_index_roundtrip_and_search(tmp_path: Path) -> None:
    corpus = tmp_path / "synthetic.jsonl"
    rows = [
        {
            "id": f"doc-{i}",
            "text": f"Document {i} discusses python search engine ranking bm25 topic_{i % 25}",
        }
        for i in range(600)
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    idx = build_index(corpus, with_positions=True)
    assert idx.num_docs == 600
    out_path = tmp_path / "large.bin"
    save_index(idx, out_path)
    loaded = load_index(out_path)
    assert loaded == idx

    results = search(loaded, 'python AND "search engine" topic_7', k=10)
    assert len(results) == 10
