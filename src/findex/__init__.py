"""findex — a streaming full-text search engine."""

from findex.corpus import Document, iter_documents
from findex.index import DocMeta, InvertedIndex, Posting, build_index
from findex.tokenize import tokenize

__all__ = [
    "DocMeta",
    "Document",
    "InvertedIndex",
    "Posting",
    "build_index",
    "iter_documents",
    "tokenize",
]
