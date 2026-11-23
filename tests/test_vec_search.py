"""Tests for Lab 8 NumPy int32 scorer, gap compression, memmap, and pytest-benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from findex.index import build_index
from findex.models import Index
from findex.ranking import make_scorer
from findex.search import search as search_lab3
from findex.vec_search import (
    VecScorerName,
    VectorizedIndex,
    decode_gap_varint,
    encode_gap_varint,
    load_memmap,
    save_memmap,
    search_numpy,
)


@pytest.mark.parametrize("scorer_name", ["bm25", "tfidf"])
@pytest.mark.parametrize(
    "query",
    [
        "python",
        "quantum",
        "python AND search",
        "python OR nasa",
        '"event loop"',
    ],
)
def test_numpy_scorer_matches_lab3_ranking_order(
    tiny_index: Index,
    scorer_name: VecScorerName,
    query: str,
) -> None:
    v_idx = VectorizedIndex.from_index(tiny_index)
    lab3_res = search_lab3(
        tiny_index,
        query,
        scorer=make_scorer(scorer_name),
        k=10,
        include_snippets=False,
    )
    np_res = search_numpy(
        tiny_index,
        query,
        scorer=scorer_name,
        k=10,
        include_snippets=False,
        vec_index=v_idx,
    )
    assert [r.doc_id for r in np_res] == [r.doc_id for r in lab3_res]
    for got, exp in zip(np_res, lab3_res, strict=True):
        assert got.score == pytest.approx(exp.score, rel=1e-6, abs=1e-6)


def test_gap_varint_roundtrip_and_compression() -> None:
    arr = np.array([0, 1, 3, 7, 15, 100, 101, 102, 500, 10_000, 10_001], dtype=np.int32)
    encoded = encode_gap_varint(arr)
    decoded = decode_gap_varint(encoded)
    assert decoded.dtype == np.int32
    assert np.array_equal(decoded, arr)
    assert len(encoded) < arr.nbytes


def test_memmap_roundtrip_matches_lab3_bm25(tiny_index: Index, tmp_path: Path) -> None:
    mm_dir = save_memmap(tiny_index, tmp_path / "mm")
    mm_idx = load_memmap(mm_dir)
    assert mm_idx.num_docs == tiny_index.num_docs

    mm_hits = mm_idx.score_terms(["python"], scorer="bm25", k=10)
    lab3_hits = search_lab3(
        tiny_index, "python", scorer=make_scorer("bm25"), k=10, include_snippets=False
    )
    assert [doc_id for doc_id, _ in mm_hits] == [r.doc_id for r in lab3_hits]
    for (_, sc_mm), r_lab3 in zip(mm_hits, lab3_hits, strict=True):
        assert sc_mm == pytest.approx(r_lab3.score, rel=1e-6, abs=1e-6)


def test_benchmark_search_numpy(benchmark: Any, tiny_index: Index) -> None:
    v_idx = VectorizedIndex.from_index(tiny_index)

    def _run_search() -> int:
        hits = search_numpy(
            tiny_index,
            "python OR search",
            scorer="bm25",
            k=5,
            include_snippets=False,
            vec_index=v_idx,
        )
        return len(hits)

    count = benchmark(_run_search)
    assert count >= 1


def test_benchmark_build_index(benchmark: Any, tiny_corpus: Path) -> None:
    def _run_build() -> int:
        idx = build_index(tiny_corpus, with_positions=False, workers=1, executor="serial")
        return idx.num_docs

    n_docs = benchmark(_run_build)
    assert n_docs == 7
