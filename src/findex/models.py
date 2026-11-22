"""Core data model: Posting, DocMeta, SearchResult, and Index (Lab 2 & Lab 3 M1)."""

from __future__ import annotations

import functools
from array import array
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Posting:
    """Single document posting for a term: document ID, term frequency, optional positions."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    """Metadata stored per indexed document."""

    doc_id: int
    source_id: str
    path: str
    title: str
    offset: int = -1


@dataclass(frozen=True, slots=True, order=True)
class SearchResult:
    """Ranked search result comparable by score (then doc_id)."""

    score: float
    doc_id: int
    title: str = field(default="", compare=False)
    snippet: str = field(default="", compare=False)

    def __str__(self) -> str:
        header = f"[{self.doc_id}] ({self.score:.4f}) {self.title}"
        if self.snippet:
            return f"{header}\n    {self.snippet}"
        return header


@dataclass(frozen=True, slots=True)
class CompactArrayPostings:
    """Compact postings representation using array('I') buffers instead of Posting objects."""

    doc_ids: array
    tfs: array


@dataclass(slots=True)
class CompactIndex:
    """Memory-lean index storing postings as flat unsigned integer arrays."""

    postings: dict[str, CompactArrayPostings]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]

    @property
    def num_docs(self) -> int:
        return len(self.doc_meta)

    @property
    def num_terms(self) -> int:
        return len(self.postings)


class Index(Mapping[str, list[Posting]]):
    """Pythonic inverted index implementing the Mapping protocol and query caching."""

    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
        has_positions: bool = False,
    ) -> None:
        self._postings = postings
        self._doc_lengths = doc_lengths
        self._doc_meta = doc_meta
        self.has_positions = has_positions
        self._closed = False
        self._cached_eval: Any = None

    # --- Backward-compatible attribute access ---
    @property
    def postings(self) -> dict[str, list[Posting]]:
        return self._postings

    @property
    def doc_lengths(self) -> dict[int, int]:
        return self._doc_lengths

    @property
    def doc_meta(self) -> dict[int, DocMeta]:
        return self._doc_meta

    # --- Mapping / container data model ---
    def __len__(self) -> int:
        return len(self._postings)

    def __contains__(self, term: object) -> bool:
        return term in self._postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self._postings):_}, docs={self.num_docs:_})"

    # --- Computed properties and stats ---
    @property
    def num_docs(self) -> int:
        return len(self._doc_lengths)

    @property
    def num_terms(self) -> int:
        return len(self._postings)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def doc_length(self, doc_id: int) -> int:
        return self._doc_lengths[doc_id]

    def df(self, term: str) -> int:
        plist = self._postings.get(term)
        return len(plist) if plist is not None else 0

    # --- Query evaluation cache (Lab 3 M4) ---
    def _ensure_cache(self) -> Any:
        if self._cached_eval is None:
            from findex.query import parse

            @functools.lru_cache(maxsize=256)
            def _eval_cached(query_str: str) -> tuple[int, ...]:
                node = parse(query_str)
                return tuple(sorted(node.evaluate(self)))

            self._cached_eval = _eval_cached
        return self._cached_eval

    def evaluate_query_cached(self, query_str: str) -> tuple[int, ...]:
        """Return sorted matching doc_ids for ``query_str``, memoized via lru_cache."""
        return self._ensure_cache()(query_str)

    def cache_info(self) -> Any:
        """Expose lru_cache statistics (hits, misses, maxsize, currsize)."""
        return self._ensure_cache().cache_info()

    def clear_cache(self) -> None:
        if self._cached_eval is not None:
            self._cached_eval.cache_clear()

    # --- Resource lifecycle (Lab 3 M1 context manager) ---
    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Release cached query state and mark the index handle as closed."""
        self.clear_cache()
        self._closed = True

    # --- Clean pickle serialization ---
    def __getstate__(self) -> dict[str, object]:
        return {
            "_postings": self._postings,
            "_doc_lengths": self._doc_lengths,
            "_doc_meta": self._doc_meta,
            "has_positions": self.has_positions,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self._postings = state["_postings"]  # type: ignore[assignment]
        self._doc_lengths = state["_doc_lengths"]  # type: ignore[assignment]
        self._doc_meta = state["_doc_meta"]  # type: ignore[assignment]
        self.has_positions = bool(state.get("has_positions", False))
        self._closed = False
        self._cached_eval = None


# Alias kept for Lab 2 compatibility
InvertedIndex = Index
