"""findex — a streaming full-text search engine."""

from findex.corpus import Document, iter_documents
from findex.decorators import timed
from findex.models import CacheStats, DocMeta, Index, InvertedIndex, Posting, SearchResult
from findex.query import And, Not, Or, Phrase, QueryNode, QueryParseError, Term, parse
from findex.ranking import BM25, Scorer, ScorerName, TfIdf, make_scorer
from findex.store import open_index
from findex.tokenize import tokenize

__all__ = [
    "BM25",
    "And",
    "CacheStats",
    "DocMeta",
    "Document",
    "Index",
    "InvertedIndex",
    "Not",
    "Or",
    "Phrase",
    "Posting",
    "QueryNode",
    "QueryParseError",
    "Scorer",
    "ScorerName",
    "SearchResult",
    "Term",
    "TfIdf",
    "iter_documents",
    "make_scorer",
    "open_index",
    "parse",
    "timed",
    "tokenize",
]
