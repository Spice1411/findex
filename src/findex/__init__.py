"""findex — a streaming full-text search engine."""

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize

__all__ = ["Document", "iter_documents", "tokenize"]
