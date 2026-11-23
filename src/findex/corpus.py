"""Lazy corpus loader (Lab 1 M1, extended with byte offsets & slices for Lab 5)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple, cast

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    """A single text document from the corpus."""

    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Lazily yield Document objects from a directory or a single file."""
    for doc, _ in iter_documents_with_offsets(root):
        yield doc


def iter_documents_with_offsets(root: Path) -> Iterator[tuple[Document, int]]:
    """Lazily yield ``(Document, byte_offset)`` pairs (offset is -1 for plain .txt files)."""
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")

    if root.is_file():
        if root.suffix == ".jsonl":
            yield from _iter_jsonl_with_offsets(root)
        else:
            text = read_text_safe(root)
            if text is not None:
                yield Document(doc_id=root.stem, path=root, text=text), -1
        return

    for file_path in sorted(root.rglob("*.txt")):
        if not file_path.is_file():
            continue
        text = read_text_safe(file_path)
        if text is None:
            continue
        doc_id = file_path.relative_to(root).as_posix()
        yield Document(doc_id=doc_id, path=file_path, text=text), -1

    for jsonl_path in sorted(root.rglob("*.jsonl")):
        if not jsonl_path.is_file():
            continue
        yield from _iter_jsonl_with_offsets(jsonl_path)


def iter_jsonl_slice_with_offsets(
    path: Path,
    start_offset: int,
    end_offset: int,
    line_start: int = 1,
) -> Iterator[tuple[Document, int]]:
    """Stream ``(Document, byte_offset)`` for a byte range ``[start_offset, end_offset)``."""
    with path.open("rb") as handle:
        handle.seek(start_offset)
        line_no = line_start - 1
        while True:
            offset = handle.tell()
            if offset >= end_offset:
                break
            raw_bytes = handle.readline()
            if not raw_bytes:
                break
            line_no += 1
            line = raw_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                raw_obj: object = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON at %s:%d (%s)", path, line_no, exc)
                continue
            if not isinstance(raw_obj, dict):
                logger.warning("Skipping non-object JSON at %s:%d", path, line_no)
                continue
            record = cast(dict[str, object], raw_obj)
            text = str(record.get("text", ""))
            doc_id = str(record.get("id", f"{path.stem}:{line_no}"))
            yield Document(doc_id=doc_id, path=path, text=text), offset


def _iter_jsonl_with_offsets(path: Path) -> Iterator[tuple[Document, int]]:
    """Stream ``(Document, byte_offset)`` from a JSON-lines file in binary mode."""
    size = path.stat().st_size
    yield from iter_jsonl_slice_with_offsets(path, start_offset=0, end_offset=size, line_start=1)


def read_text_safe(path: Path) -> str | None:
    """Read UTF-8 text from ``path``, falling back to replacement mode on encoding errors."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        logger.warning(
            "Invalid UTF-8 in %s (%s); re-reading with errors='replace'",
            path,
            exc,
        )
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as os_exc:
            logger.warning("Could not read %s: %s", path, os_exc)
            return None
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None
