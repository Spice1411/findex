"""Streaming Unicode-aware tokenizer (M2).

Tokenization policy
-------------------
1. Unicode normalization: NFC (``unicodedata.normalize("NFC", text)``) so canonically
   equivalent characters (e.g. ``"café"`` and ``"cafe\u0301"``) produce identical tokens.
2. Case folding: ``str.casefold()`` instead of ``str.lower()`` for locale-independent
   matching (e.g. ``"Straße"`` -> ``"strasse"``).
3. Apostrophes: internal apostrophes are preserved (``don't``, ``п'ять``), and typographic
   apostrophes (``’``) are normalized to ASCII ``'``. Leading/trailing quotes are stripped.
4. Hyphens: internal hyphens are preserved (``state-of-the-art``, ``f-16``) to keep
   compound terms intact; leading, trailing, and standalone dashes are stripped.
5. Digits: numbers and alphanumeric terms (``2026``, ``3d``) are kept because years,
   identifiers, and version numbers are meaningful search queries.
6. Underscores and punctuation: treated as token separators and discarded.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

# Unicode letters/digits (excluding underscore), optionally joined by internal apostrophes/hyphens.
TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*")


def tokenize(text: str) -> Iterator[str]:
    """Lazily yield normalized tokens from ``text`` using ``re.finditer``."""
    if not text:
        return
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in TOKEN_RE.finditer(normalized):
        yield match.group(0).replace("\u2019", "'")
