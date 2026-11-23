"""Snippet extraction (±80-char window) and term highlighting (Lab 3, 4 & 7)."""

from __future__ import annotations

import html as html_lib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, cast

from findex.models import DocMeta
from findex.tokenize import TOKEN_RE

HighlightStyle = Literal["markdown", "rich", "html"]


def _escape_rich(text: str) -> str:
    return text.replace("[", "\\[")


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
            raw_obj: object = json.loads(raw.decode("utf-8", errors="replace"))
            if isinstance(raw_obj, dict):
                record = cast(dict[str, object], raw_obj)
                return str(record.get("text", meta.title))
            return meta.title
        except (OSError, json.JSONDecodeError):
            return meta.title

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return meta.title


def make_snippet(
    text: str,
    terms: Iterable[str],
    window: int = 80,
    highlight_style: HighlightStyle = "markdown",
) -> str:
    """Extract a ±window character snippet around the best match and highlight terms."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""

    term_set = {t.casefold() for t in terms if t}
    if not term_set:
        base_snip = cleaned[: window * 2] + ("..." if len(cleaned) > window * 2 else "")
        if highlight_style == "rich":
            return _escape_rich(base_snip)
        if highlight_style == "html":
            return html_lib.escape(base_snip)
        return base_snip

    best_span: tuple[int, int] | None = None
    best_density = -1

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

    if highlight_style == "rich":
        parts: list[str] = []
        cursor = 0
        for m in TOKEN_RE.finditer(snippet_raw):
            parts.append(_escape_rich(snippet_raw[cursor : m.start()]))
            raw_word = m.group(0)
            norm = raw_word.casefold().replace("\u2019", "'")
            escaped_word = _escape_rich(raw_word)
            if norm in term_set:
                parts.append(f"[bold yellow]{escaped_word}[/bold yellow]")
            else:
                parts.append(escaped_word)
            cursor = m.end()
        parts.append(_escape_rich(snippet_raw[cursor:]))
        return f"{prefix}{''.join(parts)}{suffix}"

    if highlight_style == "html":
        html_parts: list[str] = []
        cursor = 0
        for m in TOKEN_RE.finditer(snippet_raw):
            html_parts.append(html_lib.escape(snippet_raw[cursor : m.start()]))
            raw_word = m.group(0)
            norm = raw_word.casefold().replace("\u2019", "'")
            escaped_word = html_lib.escape(raw_word)
            if norm in term_set:
                html_parts.append(f"<mark>{escaped_word}</mark>")
            else:
                html_parts.append(escaped_word)
            cursor = m.end()
        html_parts.append(html_lib.escape(snippet_raw[cursor:]))
        return f"{prefix}{''.join(html_parts)}{suffix}"

    def _highlight_md(m: re.Match[str]) -> str:
        raw_word = m.group(0)
        norm = raw_word.casefold().replace("\u2019", "'")
        if norm in term_set:
            return f"**{raw_word}**"
        return raw_word

    highlighted = TOKEN_RE.sub(_highlight_md, snippet_raw)
    return f"{prefix}{highlighted}{suffix}"
