"""NumPy-vectorized BM25/TF-IDF scorer, gap compression, and memmap storage (Lab 8 M2)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from findex.models import DocId, DocMeta, Index, Posting, SearchResult
from findex.query import parse
from findex.snippets import HighlightStyle, load_document_text, make_snippet

VecScorerName = Literal["bm25", "tfidf"]


def pack_doc_lengths(index: Index) -> NDArray[np.int32]:
    """Pack per-document token lengths into a dense contiguous ``int32[N]`` array."""
    if not index.doc_meta:
        return np.empty(0, dtype=np.int32)
    n_slots = max(index.num_docs, max(index.doc_meta.keys()) + 1)
    lengths = np.zeros(n_slots, dtype=np.int32)
    for doc_id in index.doc_meta:
        if 0 <= doc_id < n_slots:
            lengths[doc_id] = int(index.doc_length(doc_id))
    return lengths


def pack_postings_int32(postings: list[Posting]) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Convert a postings list into contiguous ``int32`` ``(doc_ids, tfs)`` arrays."""
    if not postings:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    doc_ids = np.array([p.doc_id for p in postings], dtype=np.int32)
    tfs = np.array([p.tf for p in postings], dtype=np.int32)
    return doc_ids, tfs


def encode_gap_varint(doc_ids: NDArray[np.int32]) -> bytes:
    """Delta-encode sorted ``int32`` doc IDs and pack gaps with LEB128 varint bytes."""
    if doc_ids.size == 0:
        return b""
    gaps = np.empty_like(doc_ids, dtype=np.int32)
    gaps[0] = doc_ids[0]
    if doc_ids.size > 1:
        gaps[1:] = np.diff(doc_ids)

    buf = bytearray()
    for val in cast(list[int], gaps.tolist()):
        n = int(val)
        if n < 0:
            raise ValueError("doc_ids must be sorted and non-negative for gap encoding")
        while n >= 0x80:
            buf.append((n & 0x7F) | 0x80)
            n >>= 7
        buf.append(n)
    return bytes(buf)


def decode_gap_varint(payload: bytes) -> NDArray[np.int32]:
    """Decode LEB128 varint gap bytes back to sorted ``int32`` doc IDs via ``np.cumsum``."""
    if not payload:
        return np.empty(0, dtype=np.int32)
    gaps: list[int] = []
    acc = 0
    shift = 0
    for byte in payload:
        acc |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
        else:
            gaps.append(acc)
            acc = 0
            shift = 0
    gap_arr = np.array(gaps, dtype=np.int32)
    return np.cumsum(gap_arr, dtype=np.int32)


@dataclass(slots=True)
class VectorizedIndex:
    """Pre-packed NumPy ``int32`` view of an ``Index`` for zero-loop vectorized scoring."""

    num_docs: int
    avg_doc_length: float
    doc_lengths: NDArray[np.int32]
    doc_ids_by_term: dict[str, NDArray[np.int32]]
    tfs_by_term: dict[str, NDArray[np.int32]]

    @classmethod
    def from_index(cls, index: Index) -> VectorizedIndex:
        """Build contiguous ``int32`` posting and document-length arrays once."""
        lengths = pack_doc_lengths(index)
        doc_ids_map: dict[str, NDArray[np.int32]] = {}
        tfs_map: dict[str, NDArray[np.int32]] = {}
        for term, plist in index.postings.items():
            d_ids, tfs = pack_postings_int32(plist)
            doc_ids_map[term] = d_ids
            tfs_map[term] = tfs
        return cls(
            num_docs=index.num_docs,
            avg_doc_length=index.avg_doc_length,
            doc_lengths=lengths,
            doc_ids_by_term=doc_ids_map,
            tfs_by_term=tfs_map,
        )

    def raw_postings_bytes(self) -> int:
        """Total bytes consumed by raw ``int32`` ``doc_ids`` arrays across all terms."""
        return sum(int(arr.nbytes) for arr in self.doc_ids_by_term.values())

    def compressed_doc_ids_bytes(self) -> int:
        """Total bytes consumed by delta + varint compressed ``doc_ids`` across all terms."""
        return sum(len(encode_gap_varint(arr)) for arr in self.doc_ids_by_term.values())

    def score_term(
        self,
        term: str,
        scores: NDArray[np.float64],
        *,
        scorer: VecScorerName = "bm25",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Add vectorized BM25 or TF-IDF scores for ``term`` into ``scores`` in-place."""
        doc_ids = self.doc_ids_by_term.get(term)
        tfs = self.tfs_by_term.get(term)
        if doc_ids is None or tfs is None or doc_ids.size == 0 or self.num_docs <= 0:
            return
        score_arrays_inplace(
            doc_ids,
            tfs,
            num_docs=self.num_docs,
            avg_doc_length=self.avg_doc_length,
            doc_lengths=self.doc_lengths,
            scores=scores,
            scorer=scorer,
            k1=k1,
            b=b,
        )


def score_arrays_inplace(
    doc_ids: NDArray[np.int32],
    tfs: NDArray[np.int32],
    *,
    num_docs: int,
    avg_doc_length: float,
    doc_lengths: NDArray[np.int32],
    scores: NDArray[np.float64],
    scorer: VecScorerName = "bm25",
    k1: float = 1.5,
    b: float = 0.75,
) -> None:
    """Vectorized BM25 / TF-IDF scoring matching Lab 3 formulas exactly."""
    df = int(doc_ids.shape[0])
    if df == 0 or num_docs <= 0:
        return

    tf_f = tfs.astype(np.float64)
    if scorer == "tfidf":
        # Exact Lab 3 TfIdf: (1 + ln(tf)) * (ln((N + 1) / (df + 1)) + 1)
        idf = math.log((num_docs + 1.0) / (df + 1.0)) + 1.0
        term_scores = (1.0 + np.log(tf_f)) * idf
    else:
        # Exact Lab 3 BM25: idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
        safe_avg = avg_doc_length if avg_doc_length > 0.0 else 1.0
        dl = doc_lengths[doc_ids].astype(np.float64)
        norm = 1.0 - b + b * (dl / safe_avg)
        term_scores = idf * (tf_f * (k1 + 1.0)) / (tf_f + k1 * norm)

    scores[doc_ids] += term_scores


def top_k_numpy(
    cand_doc_ids: NDArray[np.int32],
    cand_scores: NDArray[np.float64],
    k: int,
) -> list[tuple[DocId, float]]:
    """Select top-k ``(doc_id, score)`` pairs in O(n) via ``np.argpartition``."""
    n = int(cand_doc_ids.shape[0])
    if n == 0 or k <= 0:
        return []
    limit = min(k, n)
    if limit < n:
        kth = n - limit
        part_idx = np.argpartition(cand_scores, kth)[kth:]
        top_docs = cast(list[int], cand_doc_ids[part_idx].tolist())
        top_scs = cast(list[float], cand_scores[part_idx].tolist())
    else:
        top_docs = cast(list[int], cand_doc_ids.tolist())
        top_scs = cast(list[float], cand_scores.tolist())

    items: list[tuple[DocId, float]] = [
        (int(d), float(s)) for d, s in zip(top_docs, top_scs, strict=True)
    ]
    items.sort(key=lambda pair: (-pair[1], pair[0]))
    return items


def search_numpy(
    index: Index,
    query: str,
    *,
    scorer: VecScorerName = "bm25",
    k: int = 10,
    include_snippets: bool = True,
    highlight_style: HighlightStyle = "markdown",
    vec_index: VectorizedIndex | None = None,
) -> list[SearchResult]:
    """Run query evaluation + vectorized NumPy ``int32`` scoring + ``np.argpartition`` top-k."""
    matched_tuple = index.evaluate_query_cached(query)
    if not matched_tuple:
        return []

    v_idx = vec_index if vec_index is not None else VectorizedIndex.from_index(index)
    ast_node = parse(query)
    pos_terms = ast_node.positive_terms()

    n_slots = int(v_idx.doc_lengths.shape[0])
    scores = np.zeros(n_slots, dtype=np.float64)
    if pos_terms:
        seen_terms = list(dict.fromkeys(pos_terms))
        for term in seen_terms:
            v_idx.score_term(term, scores, scorer=scorer)

    cand_arr = np.array(matched_tuple, dtype=np.int32)
    cand_scores = scores[cand_arr]
    top_pairs = top_k_numpy(cand_arr, cand_scores, k)

    results: list[SearchResult] = []
    for doc_id, sc in top_pairs:
        meta = index.doc_meta[doc_id]
        snippet = ""
        if include_snippets:
            full_text = load_document_text(meta)
            snippet = make_snippet(full_text, pos_terms, highlight_style=highlight_style)
        results.append(
            SearchResult(
                score=sc,
                doc_id=doc_id,
                title=meta.title,
                snippet=snippet,
            )
        )
    return results


@dataclass(slots=True)
class MemmapIndex:
    """Disk-backed ``np.memmap`` postings store for instant zero-copy startup (Lab 8 Deep M2)."""

    num_docs: int
    avg_doc_length: float
    doc_lengths: NDArray[np.int32]
    all_doc_ids: NDArray[np.int32]
    all_tfs: NDArray[np.int32]
    term_slices: dict[str, tuple[int, int]]
    doc_meta: dict[DocId, DocMeta]

    def score_terms(
        self,
        terms: list[str],
        *,
        scorer: VecScorerName = "bm25",
        k: int = 10,
    ) -> list[tuple[DocId, float]]:
        """Score a list of terms directly from memmapped ``int32`` slices using ``argpartition``."""
        if not terms or self.num_docs <= 0:
            return []
        scores = np.zeros(int(self.doc_lengths.shape[0]), dtype=np.float64)
        for term in dict.fromkeys(terms):
            bounds = self.term_slices.get(term)
            if bounds is None:
                continue
            start, length = bounds
            end = start + length
            doc_ids = self.all_doc_ids[start:end]
            tfs = self.all_tfs[start:end]
            score_arrays_inplace(
                doc_ids,
                tfs,
                num_docs=self.num_docs,
                avg_doc_length=self.avg_doc_length,
                doc_lengths=self.doc_lengths,
                scores=scores,
                scorer=scorer,
            )
        nz = np.flatnonzero(scores).astype(np.int32)
        return top_k_numpy(nz, scores[nz], k)


def save_memmap(index: Index, directory: Path) -> Path:
    """Persist an ``Index`` as flat contiguous ``int32`` files for ``np.memmap`` loading."""
    directory.mkdir(parents=True, exist_ok=True)
    v_idx = VectorizedIndex.from_index(index)

    term_slices: dict[str, list[int]] = {}
    doc_id_chunks: list[NDArray[np.int32]] = []
    tf_chunks: list[NDArray[np.int32]] = []
    cursor = 0
    for term in sorted(v_idx.doc_ids_by_term):
        d_ids = v_idx.doc_ids_by_term[term]
        tfs = v_idx.tfs_by_term[term]
        length = int(d_ids.shape[0])
        term_slices[term] = [cursor, length]
        doc_id_chunks.append(d_ids)
        tf_chunks.append(tfs)
        cursor += length

    all_doc_ids = (
        np.concatenate(doc_id_chunks).astype(np.int32, copy=False)
        if doc_id_chunks
        else np.empty(0, dtype=np.int32)
    )
    all_tfs = (
        np.concatenate(tf_chunks).astype(np.int32, copy=False)
        if tf_chunks
        else np.empty(0, dtype=np.int32)
    )

    v_idx.doc_lengths.tofile(directory / "doc_lengths.int32")
    all_doc_ids.tofile(directory / "doc_ids.int32")
    all_tfs.tofile(directory / "tfs.int32")

    meta_payload = {
        "num_docs": v_idx.num_docs,
        "avg_doc_length": v_idx.avg_doc_length,
        "doc_lengths_len": int(v_idx.doc_lengths.shape[0]),
        "postings_len": int(all_doc_ids.shape[0]),
        "term_slices": term_slices,
        "doc_meta": {
            str(doc_id): {
                "doc_id": m.doc_id,
                "source_id": m.source_id,
                "path": m.path,
                "title": m.title,
                "offset": m.offset,
            }
            for doc_id, m in index.doc_meta.items()
        },
    }
    (directory / "meta.json").write_text(json.dumps(meta_payload), encoding="utf-8")
    return directory


def load_memmap(directory: Path) -> MemmapIndex:
    """Open a ``MemmapIndex`` via ``np.memmap`` without copying postings into Python objects."""
    raw_obj: object = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if not isinstance(raw_obj, dict):
        raise ValueError(f"Invalid memmap metadata in {directory}")
    meta = cast(dict[str, object], raw_obj)

    dl_len = int(cast(int, meta["doc_lengths_len"]))
    post_len = int(cast(int, meta["postings_len"]))

    doc_lengths: NDArray[np.int32] = (
        np.memmap(directory / "doc_lengths.int32", dtype=np.int32, mode="r", shape=(dl_len,))
        if dl_len > 0
        else np.empty(0, dtype=np.int32)
    )
    all_doc_ids: NDArray[np.int32] = (
        np.memmap(directory / "doc_ids.int32", dtype=np.int32, mode="r", shape=(post_len,))
        if post_len > 0
        else np.empty(0, dtype=np.int32)
    )
    all_tfs: NDArray[np.int32] = (
        np.memmap(directory / "tfs.int32", dtype=np.int32, mode="r", shape=(post_len,))
        if post_len > 0
        else np.empty(0, dtype=np.int32)
    )

    raw_slices = cast(dict[str, list[int]], meta["term_slices"])
    term_slices: dict[str, tuple[int, int]] = {
        term: (int(bounds[0]), int(bounds[1])) for term, bounds in raw_slices.items()
    }

    raw_doc_meta = cast(dict[str, dict[str, object]], meta["doc_meta"])
    doc_meta: dict[DocId, DocMeta] = {
        int(k): DocMeta(
            doc_id=int(cast(int, v["doc_id"])),
            source_id=str(v["source_id"]),
            path=str(v["path"]),
            title=str(v["title"]),
            offset=int(cast(int, v["offset"])),
        )
        for k, v in raw_doc_meta.items()
    }

    return MemmapIndex(
        num_docs=int(cast(int, meta["num_docs"])),
        avg_doc_length=float(cast(float, meta["avg_doc_length"])),
        doc_lengths=doc_lengths,
        all_doc_ids=all_doc_ids,
        all_tfs=all_tfs,
        term_slices=term_slices,
        doc_meta=doc_meta,
    )
