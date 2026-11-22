"""Streaming Unicode-aware tokenizer (Lab 1 M2)."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*")


def tokenize(text: str) -> Iterator[str]:
    """Lazily yield normalized tokens from ``text`` using ``re.finditer``."""
    if not text:
        return
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in TOKEN_RE.finditer(normalized):
        yield match.group(0).replace("\u2019", "'")
