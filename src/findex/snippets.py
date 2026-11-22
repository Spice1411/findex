"""Snippet extraction (±80-char window) and term highlighting (Lab 3 M4)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from findex.models import DocMeta
from findex.tokenize import TOKEN_RE


def load_document_text(meta: DocMeta) -> str:
    """Load full document text in O(1) via stored byte offset or direct file read."""
    path = Path(meta.path)
    if not path.exists():
        return meta.title

    if meta.offset >= 0 and path.suffix == ".jsonl":
        try:
            with path.open("rb") as handle:
                handle.seek(meta.offset)
                raw = handle.readline()
            record = json.loads(raw.decode("utf-8", errors="replace"))
            return str(record.get("text", meta.title))
        except (OSError, json.JSONDecodeError):
            return meta.title

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return meta.title


def make_snippet(text: str, terms: Sequence[str], window: int = 80) -> str:
    """Extract a ±window character snippet around the best match and highlight terms."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""

    term_set = {t.casefold() for t in terms if t}
    if not term_set:
        return cleaned[: window * 2] + ("..." if len(cleaned) > window * 2 else "")

    best_span: tuple[int, int] | None = None
    best_density = -1

    # Locate token matches in cleaned text
    match_spans: list[tuple[int, int]] = []
    for m in TOKEN_RE.finditer(cleaned):
        tok = m.group(0).casefold().replace("\u2019", "'")
        if tok in term_set:
            match_spans.append((m.start(), m.end()))

    if match_spans:
        for start, end in match_spans:
            lo = start - window
            hi = end + window
            density = sum(1 for s, e in match_spans if s >= lo and e <= hi)
            if density > best_density:
                best_density = density
                best_span = (start, end)
    else:
        # Fallback to plain casefold str.find
        folded = cleaned.casefold()
        for t in term_set:
            idx = folded.find(t)
            if idx != -1:
                best_span = (idx, idx + len(t))
                break

    if best_span is None:
        snippet_raw = cleaned[: window * 2]
        prefix = ""
        suffix = "..." if len(cleaned) > window * 2 else ""
    else:
        win_start = max(0, best_span[0] - window)
        win_end = min(len(cleaned), best_span[1] + window)
        # Snap outward slightly to avoid cutting words in half where convenient
        if win_start > 0:
            space_idx = cleaned.find(" ", win_start, best_span[0])
            if space_idx != -1:
                win_start = space_idx + 1
        if win_end < len(cleaned):
            rspace_idx = cleaned.rfind(" ", best_span[1], win_end)
            if rspace_idx != -1:
                win_end = rspace_idx
        snippet_raw = cleaned[win_start:win_end]
        prefix = "..." if win_start > 0 else ""
        suffix = "..." if win_end < len(cleaned) else ""

    def _highlight(m: re.Match[str]) -> str:
        raw_word = m.group(0)
        norm = raw_word.casefold().replace("\u2019", "'")
        if norm in term_set:
            return f"**{raw_word}**"
        return raw_word

    highlighted = TOKEN_RE.sub(_highlight, snippet_raw)
    return f"{prefix}{highlighted}{suffix}"
