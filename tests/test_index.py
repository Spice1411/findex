"""Tests for Lab 2 inverted index, Boolean search, and persistence."""

from __future__ import annotations

import json
from pathlib import Path

from findex.index import DocMeta, Posting, build_index
from findex.search import execute_query, intersect_sorted_merge, union_sorted_merge
from findex.store import load_index, save_index


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


def test_merge_and_set_engines_match(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"id": "d0", "text": "Python search engine with generators"},
        {"id": "d1", "text": "Inverted index and Boolean search in Python"},
        {"id": "d2", "text": "Hash tables and memory profiling"},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    idx = build_index(corpus)
    assert [p.doc_id for p in idx.postings["python"]] == [0, 1]

    for query in ["python search", "python OR memory", "search NOT index", "memory"]:
        res_merge = execute_query(idx, query, engine="merge")
        res_set = execute_query(idx, query, engine="set")
        assert res_merge == res_set


def test_save_and_load_pickle_and_json(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"id": "doc0", "text": "Alpha beta gamma"}) + "\n",
        encoding="utf-8",
    )
    idx = build_index(corpus, with_positions=True)

    pkl_path = tmp_path / "index.bin"
    json_path = tmp_path / "index.json"
    save_index(idx, pkl_path, fmt="pickle")
    save_index(idx, json_path, fmt="json")

    loaded_pkl = load_index(pkl_path, fmt="pickle")
    loaded_json = load_index(json_path, fmt="json")

    assert loaded_pkl.postings == idx.postings
    assert loaded_json.postings == idx.postings
    assert loaded_pkl.doc_meta == idx.doc_meta
    assert loaded_json.doc_meta == idx.doc_meta


def test_two_pointer_primitives() -> None:
    assert intersect_sorted_merge([1, 2, 5, 9], [2, 5, 7, 9, 12]) == [2, 5, 9]
    assert union_sorted_merge([1, 5], [2, 5, 6]) == [1, 2, 5, 6]
