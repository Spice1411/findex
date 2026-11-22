"""Index persistence in pickle and JSON formats plus open_index context manager.

SECURITY NOTE ON PICKLE:
Never call ``pickle.load()`` on data from an untrusted or external source.
A crafted pickle payload can execute arbitrary Python code during unpickling.
Use ``pickle`` only for index files generated locally by your own trusted pipeline,
or prefer a safe data format like JSON / binary buffers for untrusted inputs.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from findex.decorators import timed
from findex.models import CompactIndex, DocMeta, Index, Posting

FormatType = Literal["pickle", "json"]

logger = logging.getLogger(__name__)


class _JsonDocMeta(TypedDict):
    doc_id: int
    source_id: str
    path: str
    title: str
    offset: int


class _JsonPosting(TypedDict):
    doc_id: int
    tf: int
    positions: list[int]


class _JsonIndexPayload(TypedDict):
    has_positions: bool
    doc_lengths: dict[str, int]
    doc_meta: dict[str, _JsonDocMeta]
    postings: dict[str, list[_JsonPosting]]


class _IndexUnpickler(pickle.Unpickler):
    """Compatibility unpickler mapping legacy module paths to findex.models."""

    def find_class(self, module_name: str, global_name: str) -> Any:
        if module_name in {"__main__", "findex.index"} and global_name in {
            "Posting",
            "DocMeta",
            "Index",
            "InvertedIndex",
            "CompactArrayPostings",
            "CompactIndex",
        }:
            module_name = "findex.models"
        return super().find_class(module_name, global_name)


def save_index(index: Index, path: Path, fmt: FormatType | None = None) -> None:
    """Save an Index to disk in ``pickle`` or ``json`` format."""
    chosen: FormatType = fmt or ("json" if path.suffix.lower() == ".json" else "pickle")
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving %r to %s (format=%s)", index, path, chosen)

    if chosen == "pickle":
        with path.open("wb") as handle:
            pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    payload: _JsonIndexPayload = {
        "has_positions": index.has_positions,
        "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
        "doc_meta": {
            str(k): {
                "doc_id": m.doc_id,
                "source_id": m.source_id,
                "path": m.path,
                "title": m.title,
                "offset": m.offset,
            }
            for k, m in index.doc_meta.items()
        },
        "postings": {
            term: [
                {
                    "doc_id": p.doc_id,
                    "tf": p.tf,
                    "positions": list(p.positions) if index.has_positions else [],
                }
                for p in plist
            ]
            for term, plist in index.postings.items()
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


@timed
def load_index(path: Path, fmt: FormatType | None = None) -> Index:
    """Load an Index from disk (``pickle`` or ``json``)."""
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")

    chosen: FormatType = fmt or ("json" if path.suffix.lower() == ".json" else "pickle")
    logger.info("Loading index from %s (format=%s)", path, chosen)

    if chosen == "pickle":
        # Safe only for locally created index artifacts; never unpickle untrusted data.
        with path.open("rb") as handle:
            obj: object = _IndexUnpickler(handle).load()  # nosec B301
        if not isinstance(obj, Index):
            raise TypeError(f"Expected Index in {path}, got {type(obj).__name__}")
        return obj

    with path.open("r", encoding="utf-8") as handle:
        data = cast(_JsonIndexPayload, json.load(handle))

    has_positions = bool(data["has_positions"])
    doc_lengths = {int(k): int(v) for k, v in data["doc_lengths"].items()}
    doc_meta = {
        int(k): DocMeta(
            doc_id=int(m["doc_id"]),
            source_id=str(m["source_id"]),
            path=str(m["path"]),
            title=str(m["title"]),
            offset=int(m["offset"]),
        )
        for k, m in data["doc_meta"].items()
    }
    postings: dict[str, list[Posting]] = {}
    for term, raw_list in data["postings"].items():
        plist: list[Posting] = []
        for entry in raw_list:
            pos = tuple(int(x) for x in entry["positions"]) if has_positions else ()
            plist.append(Posting(doc_id=int(entry["doc_id"]), tf=int(entry["tf"]), positions=pos))
        postings[term] = plist

    return Index(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        has_positions=has_positions,
    )


@contextmanager
def open_index(path: Path, fmt: FormatType | None = None) -> Generator[Index, None, None]:
    """Context manager that loads an Index and guarantees cleanup even on exceptions."""
    index = load_index(path, fmt=fmt)
    try:
        yield index
    finally:
        index.close()


def save_compact_pickle(cidx: CompactIndex, path: Path) -> None:
    """Save a CompactIndex (array('I') postings) via pickle for Lab 2 M4 comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(cidx, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_compact_pickle(path: Path) -> CompactIndex:
    """Load a CompactIndex from a pickle file."""
    with path.open("rb") as handle:
        obj: object = _IndexUnpickler(handle).load()
    if not isinstance(obj, CompactIndex):
        raise TypeError(f"Expected CompactIndex in {path}, got {type(obj).__name__}")
    return obj
