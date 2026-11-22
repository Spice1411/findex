"""Tests for Lab 5 parallel indexing: build_partial, merge, executors, and error propagation."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from findex.corpus import iter_documents_with_offsets
from findex.index import (
    ChunkSpec,
    JsonlSlice,
    build_index,
    build_index_from_documents,
    build_partial,
    merge,
    plan_chunks,
)
from findex.models import DocMeta, PartialIndex, Posting


def test_serial_build_partial_merge_matches_lab4_builder(txt_corpus_dir: Path) -> None:
    """Lab 5 M1: merge([build_partial(all_paths)]) produces an index identical to Lab 4."""
    lab4_index = build_index_from_documents(
        iter_documents_with_offsets(txt_corpus_dir),
        with_positions=True,
    )
    all_paths = sorted(txt_corpus_dir.rglob("*.txt"))
    lab5_serial = merge([build_partial(all_paths, root=txt_corpus_dir, with_positions=True)])
    assert lab5_serial == lab4_index


@pytest.mark.parametrize("with_positions", [False, True])
def test_serial_threads_processes_identical_on_txt_dir(
    txt_corpus_dir: Path,
    with_positions: bool,
) -> None:
    idx_serial = build_index(
        txt_corpus_dir, with_positions=with_positions, workers=1, executor="serial"
    )
    idx_threads = build_index(
        txt_corpus_dir, with_positions=with_positions, workers=3, executor="threads"
    )
    idx_procs = build_index(
        txt_corpus_dir, with_positions=with_positions, workers=3, executor="processes"
    )

    assert idx_threads == idx_serial
    assert idx_procs == idx_serial


@pytest.mark.parametrize("with_positions", [False, True])
def test_serial_threads_processes_identical_on_jsonl(
    tiny_corpus: Path,
    with_positions: bool,
) -> None:
    idx_serial = build_index(
        tiny_corpus, with_positions=with_positions, workers=1, executor="serial"
    )
    idx_threads = build_index(
        tiny_corpus, with_positions=with_positions, workers=3, executor="threads"
    )
    idx_procs = build_index(
        tiny_corpus, with_positions=with_positions, workers=3, executor="processes"
    )

    assert idx_threads == idx_serial
    assert idx_procs == idx_serial


def test_build_partial_and_chunk_spec_are_picklable(txt_corpus_dir: Path) -> None:
    chunks = plan_chunks(txt_corpus_dir, workers=2, with_positions=True)
    assert len(chunks) == 2
    assert pickle.loads(pickle.dumps(chunks)) == chunks
    assert pickle.loads(pickle.dumps(build_partial)) is build_partial


def test_merge_handles_overlapping_and_out_of_order_partials() -> None:
    p1 = PartialIndex(
        start_doc_id=5,
        postings={"python": [Posting(doc_id=5, tf=1), Posting(doc_id=8, tf=2)]},
        doc_lengths={5: 10, 8: 20},
        doc_meta={
            5: DocMeta(5, "d5", "d5.txt", "D5"),
            8: DocMeta(8, "d8", "d8.txt", "D8"),
        },
        has_positions=False,
    )
    p2 = PartialIndex(
        start_doc_id=1,
        postings={"python": [Posting(doc_id=1, tf=3), Posting(doc_id=6, tf=1)]},
        doc_lengths={1: 12, 6: 15},
        doc_meta={
            1: DocMeta(1, "d1", "d1.txt", "D1"),
            6: DocMeta(6, "d6", "d6.txt", "D6"),
        },
        has_positions=False,
    )
    merged = merge([p1, p2])
    assert [p.doc_id for p in merged["python"]] == [1, 5, 6, 8]


def test_worker_exception_surfaces_loudly_without_hanging(tmp_path: Path) -> None:
    from findex.index import _execute_chunks

    missing = tmp_path / "does_not_exist.jsonl"
    bad_chunk = ChunkSpec(
        start_doc_id=0,
        jsonl_slice=JsonlSlice(path=missing, start_offset=0, end_offset=10, line_start=1),
    )
    valid_file = tmp_path / "ok.txt"
    valid_file.write_text("hello world", encoding="utf-8")
    ok_chunk = ChunkSpec(start_doc_id=1, paths=(valid_file,))

    for executor in ("threads", "processes"):
        with pytest.raises(FileNotFoundError):
            _execute_chunks([ok_chunk, bad_chunk], workers=2, executor=executor)
