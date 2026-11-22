"""Tests for Index data model, Mapping protocol, merge algorithms, and persistence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from findex.index import build_compact_index, build_index
from findex.models import DocMeta, Index, Posting
from findex.search import (
    difference_sorted_merge,
    execute_query,
    intersect_sorted_merge,
    union_sorted_merge,
)
from findex.stats import compute_index_stats, compute_stats_eager, compute_stats_lazy
from findex.store import (
    load_compact_pickle,
    load_index,
    open_index,
    save_compact_pickle,
    save_index,
)


def test_posting_and_docmeta_are_hashable_and_slotted() -> None:
    p1 = Posting(doc_id=1, tf=3)
    p2 = Posting(doc_id=1, tf=3)
    m1 = DocMeta(doc_id=0, source_id="a", path="a.txt", title="Title A")
    m2 = DocMeta(doc_id=0, source_id="a", path="a.txt", title="Title A")

    assert p1 == p2
    assert len({p1, p2}) == 1
    assert len({m1, m2}) == 1
    assert hasattr(Posting, "__slots__")
    assert hasattr(DocMeta, "__slots__")


def test_index_mapping_protocol_and_properties(tiny_index: Index) -> None:
    assert isinstance(tiny_index, Mapping)
    assert len(tiny_index) == tiny_index.num_terms > 0
    assert "python" in tiny_index
    assert "nonexistent_xyz" not in tiny_index
    assert tiny_index.num_docs == 7
    assert tiny_index.total_tokens > 0
    assert tiny_index.avg_doc_length > 0
    assert tiny_index.df("python") == 4
    assert tiny_index.df("nonexistent_xyz") == 0
    assert "Index(terms=" in repr(tiny_index)
    assert set(iter(tiny_index)) == set(tiny_index.keys())
    assert tiny_index != "not an index"

    with pytest.raises(KeyError):
        _ = tiny_index["nonexistent_xyz"]


def test_open_index_cleans_up_on_exception(saved_index_path: Path) -> None:
    captured: Index | None = None
    with pytest.raises(RuntimeError, match="boom"):
        with open_index(saved_index_path) as loaded:
            captured = loaded
            assert not loaded.closed
            assert "python" in loaded
            raise RuntimeError("boom")

    assert captured is not None
    assert captured.closed


@pytest.mark.parametrize("fmt", ["pickle", "json"])
def test_save_and_load_roundtrip(tiny_index: Index, tmp_path: Path, fmt: str) -> None:
    out_path = tmp_path / (f"idx.{fmt}" if fmt == "json" else "idx.bin")
    chosen = "json" if fmt == "json" else "pickle"
    save_index(tiny_index, out_path, fmt=chosen)
    loaded = load_index(out_path, fmt=chosen)
    assert loaded == tiny_index


def test_load_index_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "does_not_exist.bin")


def test_compact_index_build_and_roundtrip(tiny_corpus: Path, tmp_path: Path) -> None:
    cidx = build_compact_index(tiny_corpus)
    assert cidx.num_docs == 7
    assert cidx.num_terms > 0
    path = tmp_path / "compact.pkl"
    save_compact_pickle(cidx, path)
    loaded = load_compact_pickle(path)
    assert loaded.num_docs == cidx.num_docs
    assert loaded.num_terms == cidx.num_terms


def test_merge_primitives_and_boolean_engines(tiny_index: Index) -> None:
    assert intersect_sorted_merge([1, 2, 5, 9], [2, 5, 7, 9, 12]) == [2, 5, 9]
    assert union_sorted_merge([1, 5], [2, 5, 6]) == [1, 2, 5, 6]
    assert difference_sorted_merge([1, 2, 5, 9], [2, 9]) == [1, 5]

    for query in ["python event", "python OR quantum", "event NOT java", "rocket"]:
        assert execute_query(tiny_index, query, engine="merge") == execute_query(
            tiny_index, query, engine="set"
        )


def test_stats_lazy_eager_and_index_agree(tiny_corpus: Path, tiny_index: Index) -> None:
    lazy = compute_stats_lazy(tiny_corpus)
    eager = compute_stats_eager(tiny_corpus)
    from_idx = compute_index_stats(tiny_index)

    assert lazy.documents == eager.documents == from_idx.documents == 7
    assert lazy.total_tokens == eager.total_tokens == from_idx.total_tokens
    assert lazy.vocab_size == eager.vocab_size == from_idx.vocab_size


def test_build_index_respects_limit(tiny_corpus: Path) -> None:
    limited = build_index(tiny_corpus, limit=2)
    assert limited.num_docs == 2
