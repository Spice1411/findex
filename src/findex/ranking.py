"""Scoring protocol and interchangeable TF-IDF / BM25 scorers (Lab 3 M2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from findex.models import Index, Posting


@runtime_checkable
class Scorer(Protocol):
    """Structural protocol for document scoring strategies."""

    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """Classic TF-IDF scorer with sublinear TF scaling: (1 + ln(tf)) * idf."""

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.tf <= 0 or index.num_docs == 0:
            return 0.0
        tf_weight = 1.0 + math.log(posting.tf)
        idf = math.log((index.num_docs + 1.0) / (df + 1.0)) + 1.0
        return tf_weight * idf

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


@dataclass(frozen=True, slots=True)
class BM25:
    """Okapi BM25 scorer with TF saturation (k1) and document-length normalization (b)."""

    k1: float = 1.5
    b: float = 0.75

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        n = index.num_docs
        if df == 0 or posting.tf <= 0 or n == 0:
            return 0.0
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        dl = index.doc_length(posting.doc_id)
        avg_dl = index.avg_doc_length or 1.0
        tf = float(posting.tf)
        norm = 1.0 - self.b + self.b * (dl / avg_dl)
        return idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)
