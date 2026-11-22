"""Lazy corpus loader (M1).

Streams documents one at a time using generators so memory stays constant
regardless of total corpus size.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    """A single text document from the corpus."""

    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Lazily yield Document objects from a directory or a single file.

    Supported inputs:
    - Single ``.jsonl`` file: streams one JSON object per line (expects a ``"text"`` field).
    - Single ``.txt`` file: yields one document.
    - Directory: walks lazily with ``rglob`` over ``*.txt`` and ``*.jsonl`` files.

    Files with broken UTF-8 encoding are logged and decoded with replacement characters
    so a single corrupt file never aborts a long indexing run.
    """
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")

    if root.is_file():
        if root.suffix == ".jsonl":
            yield from _iter_jsonl(root)
        else:
            text = _read_text_safe(root)
            if text is not None:
                yield Document(doc_id=root.stem, path=root, text=text)
        return

    for file_path in root.rglob("*.txt"):
        if not file_path.is_file():
            continue
        text = _read_text_safe(file_path)
        if text is None:
            continue
        doc_id = file_path.relative_to(root).as_posix()
        yield Document(doc_id=doc_id, path=file_path, text=text)

    for jsonl_path in root.rglob("*.jsonl"):
        if not jsonl_path.is_file():
            continue
        yield from _iter_jsonl(jsonl_path)


def _iter_jsonl(path: Path) -> Iterator[Document]:
    """Stream one Document per line from a JSON-lines file."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON at %s:%d (%s)", path, line_no, exc)
                continue
            text = str(record.get("text", ""))
            doc_id = str(record.get("id", f"{path.stem}:{line_no}"))
            yield Document(doc_id=doc_id, path=path, text=text)


def _read_text_safe(path: Path) -> str | None:
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
