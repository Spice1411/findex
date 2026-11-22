"""Parallel & serial inverted index construction (Lab 2–5)."""

from __future__ import annotations

import argparse
import gc
import logging
import multiprocessing
import shutil
import tempfile
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

from findex.corpus import (
    Document,
    iter_documents,
    iter_documents_with_offsets,
    iter_jsonl_slice_with_offsets,
    read_text_safe,
)
from findex.decorators import timed
from findex.models import (
    BuildProfile,
    CompactArrayPostings,
    CompactIndex,
    DocId,
    DocMeta,
    Index,
    InvertedIndex,
    PartialIndex,
    Posting,
)
from findex.tokenize import tokenize

ExecutorName = Literal["serial", "threads", "processes"]

__all__ = [
    "BuildProfile",
    "ChunkSpec",
    "CompactArrayPostings",
    "CompactIndex",
    "DocMeta",
    "ExecutorName",
    "Index",
    "InvertedIndex",
    "JsonlSlice",
    "PartialIndex",
    "Posting",
    "build_compact_index",
    "build_index",
    "build_index_from_documents",
    "build_index_profiled",
    "build_partial",
    "merge",
    "plan_chunks",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class JsonlSlice:
    """Picklable byte-range slice into a single JSONL corpus file (zero loaded text sent)."""

    path: Path
    start_offset: int
    end_offset: int
    line_start: int


@dataclass(frozen=True, slots=True)
class ChunkSpec:
    """Picklable work unit with file paths or a JSONL byte slice and disjoint doc_id base."""

    start_doc_id: DocId
    paths: tuple[Path, ...] = ()
    root: Path | None = None
    jsonl_slice: JsonlSlice | None = None
    with_positions: bool = False


class _PlainPosting:
    """Plain class WITHOUT __slots__ (has per-instance __dict__) for memory comparison."""

    def __init__(self, doc_id: DocId, tf: int) -> None:
        self.doc_id = doc_id
        self.tf = tf


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned[:80]
    return fallback


def _iter_chunk_documents(chunk: ChunkSpec) -> Iterator[tuple[Document, int]]:
    """Read documents assigned to ``chunk`` directly inside the worker."""
    if chunk.jsonl_slice is not None:
        sl = chunk.jsonl_slice
        yield from iter_jsonl_slice_with_offsets(
            sl.path,
            start_offset=sl.start_offset,
            end_offset=sl.end_offset,
            line_start=sl.line_start,
        )
        return

    for file_path in chunk.paths:
        if file_path.suffix == ".jsonl":
            yield from iter_documents_with_offsets(file_path)
            continue
        text = read_text_safe(file_path)
        if text is None:
            continue
        if chunk.root is not None:
            doc_id = file_path.relative_to(chunk.root).as_posix()
        else:
            doc_id = file_path.stem
        yield Document(doc_id=doc_id, path=file_path, text=text), -1


def build_partial(
    target: Sequence[Path] | ChunkSpec,
    *,
    start_doc_id: DocId = 0,
    root: Path | None = None,
    with_positions: bool = False,
) -> PartialIndex:
    """Module-level picklable worker building a PartialIndex from paths or a ChunkSpec."""
    cpu_t0 = time.process_time()

    if isinstance(target, ChunkSpec):
        spec = target
    else:
        spec = ChunkSpec(
            start_doc_id=start_doc_id,
            paths=tuple(target),
            root=root,
            with_positions=with_positions,
        )

    raw_postings: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[DocId, int] = {}
    doc_meta: dict[DocId, DocMeta] = {}

    next_doc_id = spec.start_doc_id
    for doc, byte_offset in _iter_chunk_documents(spec):
        doc_num = next_doc_id
        next_doc_id += 1

        term_counts: Counter[str] = Counter()
        term_positions: dict[str, list[int]] = defaultdict(list)
        token_count = 0

        for pos, token in enumerate(tokenize(doc.text)):
            token_count += 1
            term_counts[token] += 1
            if spec.with_positions:
                term_positions[token].append(pos)

        doc_lengths[doc_num] = token_count
        doc_meta[doc_num] = DocMeta(
            doc_id=doc_num,
            source_id=doc.doc_id,
            path=doc.path.as_posix(),
            title=_extract_title(doc.text, fallback=doc.doc_id),
            offset=byte_offset,
        )

        for term, tf in term_counts.items():
            pos_tuple = tuple(term_positions[term]) if spec.with_positions else ()
            raw_postings[term].append(Posting(doc_id=doc_num, tf=tf, positions=pos_tuple))

    worker_cpu = time.process_time() - cpu_t0
    return PartialIndex(
        start_doc_id=spec.start_doc_id,
        postings=dict(raw_postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        has_positions=spec.with_positions,
        worker_cpu_seconds=worker_cpu,
    )


def _merge_sorted_postings(left: list[Posting], right: list[Posting]) -> list[Posting]:
    """Two-pointer sorted merge of two posting lists ordered by ``doc_id``."""
    out: list[Posting] = []
    i = 0
    j = 0
    len_l = len(left)
    len_r = len(right)
    while i < len_l and j < len_r:
        if left[i].doc_id <= right[j].doc_id:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    if i < len_l:
        out.extend(left[i:])
    if j < len_r:
        out.extend(right[j:])
    return out


def merge(partials: Iterable[PartialIndex]) -> Index:
    """Merge partial indexes into a single Index with strictly sorted posting lists."""
    ordered = sorted(partials, key=lambda part: part.start_doc_id)
    if not ordered:
        return Index(postings={}, doc_lengths={}, doc_meta={}, has_positions=False)

    has_positions = all(part.has_positions for part in ordered)
    doc_lengths: dict[DocId, int] = {}
    doc_meta: dict[DocId, DocMeta] = {}
    merged_postings: dict[str, list[Posting]] = {}

    for part in ordered:
        doc_lengths.update(part.doc_lengths)
        doc_meta.update(part.doc_meta)
        for term, plist in part.postings.items():
            if not plist:
                continue
            existing = merged_postings.get(term)
            if existing is None:
                merged_postings[term] = list(plist)
            elif existing[-1].doc_id < plist[0].doc_id:
                existing.extend(plist)
            else:
                merged_postings[term] = _merge_sorted_postings(existing, plist)

    return Index(
        postings=merged_postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        has_positions=has_positions,
    )


def _scan_jsonl_line_offsets(
    path: Path,
    limit: int | None = None,
) -> tuple[list[tuple[int, int]], int]:
    """Fast binary scan returning ``([(byte_offset, line_no), ...], end_byte_offset)``."""
    offsets: list[tuple[int, int]] = []
    end_offset = 0
    with path.open("rb") as handle:
        line_no = 0
        while True:
            pos = handle.tell()
            raw = handle.readline()
            if not raw:
                end_offset = pos
                break
            line_no += 1
            if not raw.strip():
                continue
            offsets.append((pos, line_no))
            end_offset = handle.tell()
            if limit is not None and len(offsets) >= limit:
                break
    return offsets, end_offset


def _split_jsonl_into_chunks(
    path: Path,
    num_chunks: int,
    start_doc_id: DocId,
    with_positions: bool,
    limit: int | None = None,
) -> tuple[list[ChunkSpec], DocId]:
    offsets, end_byte = _scan_jsonl_line_offsets(path, limit=limit)
    total = len(offsets)
    if total == 0:
        return [], start_doc_id

    actual_chunks = min(max(1, num_chunks), total)
    base_size, remainder = divmod(total, actual_chunks)
    chunks: list[ChunkSpec] = []
    idx = 0
    for chunk_i in range(actual_chunks):
        size = base_size + (1 if chunk_i < remainder else 0)
        lo = idx
        hi = idx + size
        idx = hi

        slice_start_byte, line_start = offsets[lo]
        slice_end_byte = offsets[hi][0] if hi < total else end_byte
        chunks.append(
            ChunkSpec(
                start_doc_id=start_doc_id + lo,
                jsonl_slice=JsonlSlice(
                    path=path,
                    start_offset=slice_start_byte,
                    end_offset=slice_end_byte,
                    line_start=line_start,
                ),
                with_positions=with_positions,
            )
        )
    return chunks, start_doc_id + total


def plan_chunks(
    root: Path,
    workers: int = 1,
    limit: int | None = None,
    with_positions: bool = False,
) -> list[ChunkSpec]:
    """Partition ``root`` into picklable ChunkSpecs with globally unique disjoint doc_id ranges."""
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")
    if limit is not None and limit <= 0:
        return []

    target_chunks = max(1, workers)

    if root.is_file():
        if root.suffix == ".jsonl":
            file_chunks, _ = _split_jsonl_into_chunks(
                root,
                num_chunks=target_chunks,
                start_doc_id=0,
                with_positions=with_positions,
                limit=limit,
            )
            return file_chunks
        return [
            ChunkSpec(
                start_doc_id=0,
                paths=(root,),
                root=None,
                with_positions=with_positions,
            )
        ]

    txt_files = [p for p in sorted(root.rglob("*.txt")) if p.is_file()]
    jsonl_files = [p for p in sorted(root.rglob("*.jsonl")) if p.is_file()]

    remaining = limit
    next_id: DocId = 0
    chunks: list[ChunkSpec] = []

    if txt_files:
        if remaining is not None:
            txt_files = txt_files[:remaining]
            remaining -= len(txt_files)
        if txt_files:
            actual = min(target_chunks, len(txt_files))
            base_size, rem = divmod(len(txt_files), actual)
            pos = 0
            for i in range(actual):
                size = base_size + (1 if i < rem else 0)
                sub = tuple(txt_files[pos : pos + size])
                chunks.append(
                    ChunkSpec(
                        start_doc_id=next_id + pos,
                        paths=sub,
                        root=root,
                        with_positions=with_positions,
                    )
                )
                pos += size
            next_id += len(txt_files)

    for jsonl_path in jsonl_files:
        if remaining is not None and remaining <= 0:
            break
        per_file_chunks = target_chunks if len(jsonl_files) == 1 else 1
        sub_chunks, new_next = _split_jsonl_into_chunks(
            jsonl_path,
            num_chunks=per_file_chunks,
            start_doc_id=next_id,
            with_positions=with_positions,
            limit=remaining,
        )
        added = new_next - next_id
        if remaining is not None:
            remaining -= added
        next_id = new_next
        chunks.extend(sub_chunks)

    return chunks


def _execute_chunks(
    chunks: Sequence[ChunkSpec],
    workers: int,
    executor: ExecutorName,
    on_document: Callable[[int], None] | None = None,
) -> list[PartialIndex]:
    """Execute ``build_partial`` over ``chunks`` using serial, thread, or process pool."""
    if not chunks:
        return []

    partials: list[PartialIndex] = []
    completed_docs = 0

    if executor == "serial" or workers <= 1 or len(chunks) == 1:
        for chunk in chunks:
            part = build_partial(chunk)
            partials.append(part)
            completed_docs += part.num_docs
            if on_document is not None:
                on_document(completed_docs)
        return partials

    max_workers = min(workers, len(chunks))
    if executor == "threads":
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures: list[Future[PartialIndex]] = [
                pool.submit(build_partial, chunk) for chunk in chunks
            ]
            try:
                for fut in as_completed(futures):
                    part = fut.result()
                    partials.append(part)
                    completed_docs += part.num_docs
                    if on_document is not None:
                        on_document(completed_docs)
            except BaseException:
                for fut in futures:
                    fut.cancel()
                raise
        return partials

    mp_ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as pool:
        proc_futures: list[Future[PartialIndex]] = [
            pool.submit(build_partial, chunk) for chunk in chunks
        ]
        try:
            for fut in as_completed(proc_futures):
                part = fut.result()
                partials.append(part)
                completed_docs += part.num_docs
                if on_document is not None:
                    on_document(completed_docs)
        except BaseException:
            for fut in proc_futures:
                fut.cancel()
            raise
    return partials


def build_index_profiled(
    root: Path,
    limit: int | None = None,
    with_positions: bool = False,
    on_document: Callable[[int], None] | None = None,
    workers: int = 1,
    executor: ExecutorName = "serial",
) -> BuildProfile:
    """Build an Index and return wall time, total CPU time, and sequential merge time."""
    wall_t0 = time.perf_counter()
    parent_cpu_t0 = time.process_time()

    chunks = plan_chunks(
        root,
        workers=1 if executor == "serial" else workers,
        limit=limit,
        with_positions=with_positions,
    )
    partials = _execute_chunks(
        chunks,
        workers=workers,
        executor=executor,
        on_document=on_document,
    )

    merge_t0 = time.perf_counter()
    index = merge(partials)
    merge_seconds = time.perf_counter() - merge_t0

    parent_cpu = time.process_time() - parent_cpu_t0
    wall_seconds = time.perf_counter() - wall_t0

    if executor == "processes" and workers > 1 and len(chunks) > 1:
        total_cpu = parent_cpu + sum(part.worker_cpu_seconds for part in partials)
    else:
        total_cpu = parent_cpu

    logger.info(
        "Built %r (executor=%s, workers=%d, chunks=%d, wall=%.3fs, merge=%.3fs)",
        index,
        executor,
        workers,
        len(chunks),
        wall_seconds,
        merge_seconds,
    )
    return BuildProfile(
        index=index,
        wall_seconds=wall_seconds,
        cpu_seconds=total_cpu,
        merge_seconds=merge_seconds,
        chunks=len(chunks),
    )


def build_index_from_documents(
    docs: Iterable[tuple[Document, int]],
    with_positions: bool = False,
    on_document: Callable[[int], None] | None = None,
) -> Index:
    """Pure in-memory index builder accepting any Iterable of ``(Document, offset)`` pairs."""
    raw_postings: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[DocId, int] = {}
    doc_meta: dict[DocId, DocMeta] = {}

    for doc_num, (doc, byte_offset) in enumerate(docs):
        term_counts: Counter[str] = Counter()
        term_positions: dict[str, list[int]] = defaultdict(list)
        token_count = 0

        for pos, token in enumerate(tokenize(doc.text)):
            token_count += 1
            term_counts[token] += 1
            if with_positions:
                term_positions[token].append(pos)

        doc_lengths[doc_num] = token_count
        doc_meta[doc_num] = DocMeta(
            doc_id=doc_num,
            source_id=doc.doc_id,
            path=doc.path.as_posix(),
            title=_extract_title(doc.text, fallback=doc.doc_id),
            offset=byte_offset,
        )

        for term, tf in term_counts.items():
            pos_tuple = tuple(term_positions[term]) if with_positions else ()
            raw_postings[term].append(Posting(doc_id=doc_num, tf=tf, positions=pos_tuple))

        if on_document is not None:
            on_document(doc_num + 1)

    sorted_postings: dict[str, list[Posting]] = {}
    for term, plist in raw_postings.items():
        plist.sort(key=lambda p: p.doc_id)
        sorted_postings[term] = plist

    index = Index(
        postings=sorted_postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        has_positions=with_positions,
    )
    logger.info("Built %r (positions=%s)", index, with_positions)
    return index


@timed
def build_index(
    root: Path,
    limit: int | None = None,
    with_positions: bool = False,
    on_document: Callable[[int], None] | None = None,
    workers: int = 1,
    executor: ExecutorName = "serial",
) -> Index:
    """Build an Index over ``root`` using serial, threads, or processes."""
    profile = build_index_profiled(
        root,
        limit=limit,
        with_positions=with_positions,
        on_document=on_document,
        workers=workers,
        executor=executor,
    )
    return profile.index


def build_compact_index(root: Path, limit: int | None = None) -> CompactIndex:
    """Build an index using array('I') pairs for doc_ids and tfs (zero Posting objects)."""
    docs: Iterable[tuple[Document, int]] = iter_documents_with_offsets(root)
    if limit is not None:
        docs = islice(docs, limit)

    raw_pairs: dict[str, list[tuple[DocId, int]]] = defaultdict(list)
    doc_lengths: dict[DocId, int] = {}
    doc_meta: dict[DocId, DocMeta] = {}

    for doc_num, (doc, byte_offset) in enumerate(docs):
        term_counts: Counter[str] = Counter()
        token_count = 0
        for token in tokenize(doc.text):
            token_count += 1
            term_counts[token] += 1

        doc_lengths[doc_num] = token_count
        doc_meta[doc_num] = DocMeta(
            doc_id=doc_num,
            source_id=doc.doc_id,
            path=doc.path.as_posix(),
            title=_extract_title(doc.text, fallback=doc.doc_id),
            offset=byte_offset,
        )
        for term, tf in term_counts.items():
            raw_pairs[term].append((doc_num, tf))

    compact_postings: dict[str, CompactArrayPostings] = {}
    for term, pairs in raw_pairs.items():
        pairs.sort(key=lambda pair: pair[0])
        doc_arr: array[int] = array("I", (d for d, _ in pairs))
        tf_arr: array[int] = array("I", (t for _, t in pairs))
        compact_postings[term] = CompactArrayPostings(doc_ids=doc_arr, tfs=tf_arr)

    return CompactIndex(
        postings=compact_postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
    )


def measure_plain_dataclass_build(root: Path, limit: int | None = None) -> tuple[int, float]:
    """Measure build peak memory and time using non-slotted _PlainPosting objects."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()

    docs: Iterable[Document] = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)

    raw: dict[str, list[_PlainPosting]] = defaultdict(list)
    for doc_num, doc in enumerate(docs):
        counts = Counter(tokenize(doc.text))
        for term, tf in counts.items():
            raw[term].append(_PlainPosting(doc_num, tf))

    for plist in raw.values():
        plist.sort(key=lambda p: p.doc_id)

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, elapsed


def measure_slotted_build(
    root: Path,
    limit: int | None = None,
    with_positions: bool = False,
) -> tuple[Index, int, float]:
    """Measure build peak memory and time for slotted frozen dataclass Posting."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    idx = build_index(root, limit=limit, with_positions=with_positions)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return idx, peak, elapsed


def measure_compact_build(root: Path, limit: int | None = None) -> tuple[CompactIndex, int, float]:
    """Measure build peak memory and time for array('I') compact postings."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    cidx = build_compact_index(root, limit=limit)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return cidx, peak, elapsed


def _format_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def run_memory_study(root: Path, limit: int | None = None) -> str:
    """Run Lab 2 M4 memory comparison across plain dataclass, slotted Posting, and array('I')."""
    from findex.store import load_compact_pickle, load_index, save_compact_pickle, save_index

    plain_peak, plain_time = measure_plain_dataclass_build(root, limit=limit)
    slotted_idx, slotted_peak, slotted_time = measure_slotted_build(root, limit=limit)
    compact_idx, compact_peak, compact_time = measure_compact_build(root, limit=limit)

    tmp_dir = Path(tempfile.mkdtemp(prefix="findex_m4_"))
    try:
        slotted_path = tmp_dir / "slotted.pkl"
        compact_path = tmp_dir / "compact.pkl"
        save_index(slotted_idx, slotted_path)
        save_compact_pickle(compact_idx, compact_path)

        slotted_size = slotted_path.stat().st_size
        compact_size = compact_path.stat().st_size

        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        _ = load_index(slotted_path)
        slotted_load_time = time.perf_counter() - t0
        _, slotted_load_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        _ = load_compact_pickle(compact_path)
        compact_load_time = time.perf_counter() - t0
        _, compact_load_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    lines = [
        "| Postings representation | Peak memory (build) | Index file size | Load time |",
        "|---|---:|---:|---:|",
        (
            "| `list[Posting]` with plain `@dataclass` / `__dict__` "
            f"| {_format_mb(plain_peak)} ({plain_time:.2f}s) | — | — |"
        ),
        (
            "| `list[Posting]` with `slots=True` "
            f"| {_format_mb(slotted_peak)} ({slotted_time:.2f}s) "
            f"| {_format_mb(slotted_size)} | {slotted_load_time:.3f} s |"
        ),
        (
            "| `array('I')` pairs (doc_ids, tfs) — no objects "
            f"| {_format_mb(compact_peak)} ({compact_time:.2f}s) "
            f"| {_format_mb(compact_size)} | {compact_load_time:.3f} s |"
        ),
        (
            f"\nExtra info: slotted load peak={_format_mb(slotted_load_peak)}, "
            f"compact load peak={_format_mb(compact_load_peak)}."
        ),
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m findex.index",
        description="Build and persist an inverted index over a corpus.",
    )
    parser.add_argument("root", type=Path, nargs="?", default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("index.bin"))
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--positions", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--executor",
        choices=["serial", "threads", "processes"],
        default="serial",
    )
    parser.add_argument("--memory-study", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    from findex.store import save_index

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s]: %(message)s")
    args = build_parser().parse_args(argv)

    if args.memory_study:
        print(run_memory_study(args.root, limit=args.limit))
        return 0

    exec_name: ExecutorName = (
        "threads"
        if args.executor == "threads"
        else ("processes" if args.executor == "processes" else "serial")
    )
    profile = build_index_profiled(
        args.root,
        limit=args.limit,
        with_positions=args.positions,
        workers=args.workers,
        executor=exec_name,
    )
    save_index(profile.index, args.out, fmt=args.format)
    file_size = args.out.stat().st_size

    print(f"Index:             {profile.index!r}")
    print(f"Executor:          {exec_name} (workers={args.workers})")
    print(f"Wall time:         {profile.wall_seconds:.3f} s (merge {profile.merge_seconds:.3f} s)")
    print(f"CPU time:          {profile.cpu_seconds:.3f} s")
    print(f"Index file size:   {_format_mb(file_size)} ({args.out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
