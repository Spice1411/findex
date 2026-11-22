"""Index persistence in pickle and JSON formats (Lab 2 M3).

SECURITY NOTE ON PICKLE:
Never call ``pickle.load()`` on data from an untrusted or external source.
A crafted pickle payload can execute arbitrary Python code during unpickling.
Use ``pickle`` only for index files generated locally by your own trusted pipeline,
or prefer a safe data format like JSON / binary buffers for untrusted inputs.
"""

from __future__ import annotations

import json
import pickle

class _IndexUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "__main__" and name in {"Posting", "DocMeta", "InvertedIndex", "CompactArrayPostings", "CompactIndex"}:
            module = "findex.index"
        return super().find_class(module, name)
from pathlib import Path
from typing import Literal

from findex.index import CompactIndex, DocMeta, InvertedIndex, Posting

FormatType = Literal["pickle", "json"]


def save_index(index: InvertedIndex, path: Path, fmt: FormatType | None = None) -> None:
    """Save an InvertedIndex to disk in ``pickle`` or ``json`` format."""
    chosen: FormatType = fmt or ("json" if path.suffix.lower() == ".json" else "pickle")
    path.parent.mkdir(parents=True, exist_ok=True)

    if chosen == "pickle":
        with path.open("wb") as handle:
            pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    payload = {
        "has_positions": index.has_positions,
        "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
        "doc_meta": {
            str(k): {
                "doc_id": m.doc_id,
                "source_id": m.source_id,
                "path": m.path,
                "title": m.title,
            }
            for k, m in index.doc_meta.items()
        },
        "postings": {
            term: [
                [p.doc_id, p.tf, list(p.positions)] if index.has_positions else [p.doc_id, p.tf]
                for p in plist
            ]
            for term, plist in index.postings.items()
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def load_index(path: Path, fmt: FormatType | None = None) -> InvertedIndex:
    """Load an InvertedIndex from disk (``pickle`` or ``json``)."""
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")

    chosen: FormatType = fmt or ("json" if path.suffix.lower() == ".json" else "pickle")

    if chosen == "pickle":
        # Safe only for locally created index artifacts; never unpickle untrusted data.
        with path.open("rb") as handle:
            obj = _IndexUnpickler(handle).load()  # nosec B301
        if not isinstance(obj, InvertedIndex):
            raise TypeError(f"Expected InvertedIndex in {path}, got {type(obj).__name__}")
        return obj

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    has_positions = bool(data.get("has_positions", False))
    doc_lengths = {int(k): int(v) for k, v in data["doc_lengths"].items()}
    doc_meta = {
        int(k): DocMeta(
            doc_id=int(m["doc_id"]),
            source_id=str(m["source_id"]),
            path=str(m["path"]),
            title=str(m["title"]),
        )
        for k, m in data["doc_meta"].items()
    }
    postings: dict[str, list[Posting]] = {}
    for term, raw_list in data["postings"].items():
        plist: list[Posting] = []
        for entry in raw_list:
            d_id = int(entry[0])
            tf = int(entry[1])
            pos = tuple(int(x) for x in entry[2]) if has_positions and len(entry) > 2 else ()
            plist.append(Posting(doc_id=d_id, tf=tf, positions=pos))
        postings[term] = plist

    return InvertedIndex(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        has_positions=has_positions,
    )


def save_compact_pickle(cidx: CompactIndex, path: Path) -> None:
    """Save a CompactIndex (array('I') postings) via pickle for M4 size/load comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(cidx, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_compact_pickle(path: Path) -> CompactIndex:
    """Load a CompactIndex from a pickle file."""
    with path.open("rb") as handle:
        obj = _IndexUnpickler(handle).load()
    if not isinstance(obj, CompactIndex):
        raise TypeError(f"Expected CompactIndex in {path}, got {type(obj).__name__}")
    return obj
