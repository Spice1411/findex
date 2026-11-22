"""findex — a streaming full-text search engine."""

from findex.corpus import Document, iter_documents
from findex.crawler import (
    CrawlStats,
    FetchResult,
    Page,
    RobotsCache,
    canonicalize_url,
    crawl,
    fetch,
)
from findex.decorators import timed
from findex.models import (
    BuildProfile,
    CacheStats,
    DocMeta,
    Index,
    InvertedIndex,
    PartialIndex,
    Posting,
    SearchResult,
)
from findex.query import And, Not, Or, Phrase, QueryNode, QueryParseError, Term, parse
from findex.ranking import BM25, Scorer, ScorerName, TfIdf, make_scorer
from findex.store import open_index
from findex.tokenize import tokenize

__all__ = [
    "BM25",
    "And",
    "BuildProfile",
    "CacheStats",
    "CrawlStats",
    "DocMeta",
    "Document",
    "FetchResult",
    "Index",
    "InvertedIndex",
    "Not",
    "Or",
    "Page",
    "PartialIndex",
    "Phrase",
    "Posting",
    "QueryNode",
    "QueryParseError",
    "RobotsCache",
    "Scorer",
    "ScorerName",
    "SearchResult",
    "Term",
    "TfIdf",
    "canonicalize_url",
    "crawl",
    "fetch",
    "iter_documents",
    "make_scorer",
    "open_index",
    "parse",
    "timed",
    "tokenize",
]
