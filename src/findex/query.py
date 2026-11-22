"""Recursive-descent Boolean & phrase query parser and AST nodes (Lab 3 M3).

Grammar:
    or_expr  -> and_expr ('OR' and_expr)*
    and_expr -> not_expr ('AND'? not_expr)*
    not_expr -> 'NOT' not_expr | atom
    atom     -> TERM | PHRASE | '(' or_expr ')'
"""

from __future__ import annotations

from dataclasses import dataclass

from findex.models import DocId, Index
from findex.tokenize import tokenize


class QueryParseError(ValueError):
    """Raised when a query string has invalid syntax."""


class QueryNode:
    """Base AST node supporting Pythonic &, |, ~ composition."""

    def __and__(self, other: QueryNode) -> And:
        return And(self, other)

    def __or__(self, other: QueryNode) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)

    def evaluate(self, index: Index) -> set[DocId]:
        raise NotImplementedError

    def positive_terms(self) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Term(QueryNode):
    """Single normalized term leaf node."""

    value: str

    def evaluate(self, index: Index) -> set[DocId]:
        if self.value not in index:
            return set()
        return {p.doc_id for p in index[self.value]}

    def positive_terms(self) -> tuple[str, ...]:
        return (self.value,)


@dataclass(frozen=True, slots=True)
class Phrase(QueryNode):
    """Exact ordered phrase leaf node requiring positional postings."""

    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> set[DocId]:
        if not self.terms:
            return set()
        if len(self.terms) == 1:
            return Term(self.terms[0]).evaluate(index)
        if not index.has_positions:
            raise ValueError(
                "Phrase queries require positional postings. Rebuild the index with `--positions`."
            )

        posting_maps: list[dict[DocId, tuple[int, ...]]] = []
        candidate_docs: set[DocId] | None = None
        for term in self.terms:
            if term not in index:
                return set()
            plist = index[term]
            doc_set = {p.doc_id for p in plist}
            candidate_docs = doc_set if candidate_docs is None else (candidate_docs & doc_set)
            if not candidate_docs:
                return set()
            posting_maps.append({p.doc_id: p.positions for p in plist})

        matched: set[DocId] = set()
        if candidate_docs is None:
            return matched
        for doc_id in candidate_docs:
            pos_lists = [pm[doc_id] for pm in posting_maps]
            if _has_adjacent_positions(pos_lists):
                matched.add(doc_id)
        return matched

    def positive_terms(self) -> tuple[str, ...]:
        return self.terms


@dataclass(frozen=True, slots=True)
class And(QueryNode):
    """Conjunction of two query subtrees."""

    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[DocId]:
        left_docs = self.left.evaluate(index)
        if not left_docs:
            return set()
        return left_docs & self.right.evaluate(index)

    def positive_terms(self) -> tuple[str, ...]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True, slots=True)
class Or(QueryNode):
    """Disjunction of two query subtrees."""

    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[DocId]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def positive_terms(self) -> tuple[str, ...]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True, slots=True)
class Not(QueryNode):
    """Negation of a query subtree relative to all indexed documents."""

    operand: QueryNode

    def evaluate(self, index: Index) -> set[DocId]:
        return set(index.doc_meta.keys()) - self.operand.evaluate(index)

    def positive_terms(self) -> tuple[str, ...]:
        return ()


def _advance_adjacent(prev_pos: tuple[int, ...], next_pos: tuple[int, ...]) -> tuple[int, ...]:
    """Two-pointer merge finding all p2 in next_pos where (p2 - 1) is in prev_pos."""
    matches: list[int] = []
    i = 0
    j = 0
    len_a = len(prev_pos)
    len_b = len(next_pos)
    while i < len_a and j < len_b:
        target = prev_pos[i] + 1
        val = next_pos[j]
        if target == val:
            matches.append(val)
            i += 1
            j += 1
        elif target < val:
            i += 1
        else:
            j += 1
    return tuple(matches)


def _has_adjacent_positions(pos_lists: list[tuple[int, ...]]) -> bool:
    """Return True if there exists a consecutive chain p, p+1, ..., p+k-1 across pos_lists."""
    current = pos_lists[0]
    for nxt in pos_lists[1:]:
        current = _advance_adjacent(current, nxt)
        if not current:
            return False
    return bool(current)


@dataclass(frozen=True, slots=True)
class _LexToken:
    kind: str  # "LPAREN", "RPAREN", "AND", "OR", "NOT", "PHRASE", "TERM"
    value: str


def _lex(query: str) -> list[_LexToken]:
    tokens: list[_LexToken] = []
    i = 0
    n = len(query)
    while i < n:
        ch = query[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(_LexToken("LPAREN", "("))
            i += 1
            continue
        if ch == ")":
            tokens.append(_LexToken("RPAREN", ")"))
            i += 1
            continue
        if ch == '"':
            end = query.find('"', i + 1)
            if end == -1:
                raise QueryParseError("Unclosed quote in phrase query")
            phrase_raw = query[i + 1 : end]
            tokens.append(_LexToken("PHRASE", phrase_raw))
            i = end + 1
            continue

        start = i
        while i < n and not query[i].isspace() and query[i] not in '()"':
            i += 1
        word = query[start:i]
        if word in {"AND", "OR", "NOT"}:
            tokens.append(_LexToken(word, word))
        else:
            tokens.append(_LexToken("TERM", word))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_LexToken]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> _LexToken | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self, expected_kind: str | None = None) -> _LexToken:
        tok = self._peek()
        if tok is None:
            raise QueryParseError("Unexpected end of query")
        if expected_kind is not None and tok.kind != expected_kind:
            raise QueryParseError(f"Expected {expected_kind}, got {tok.kind} ({tok.value!r})")
        self._pos += 1
        return tok

    def parse(self) -> QueryNode:
        if not self._tokens:
            raise QueryParseError("Empty query")
        node = self._parse_or()
        leftover = self._peek()
        if leftover is not None:
            raise QueryParseError(f"Unexpected token {leftover.value!r}")
        return node

    def _parse_or(self) -> QueryNode:
        node = self._parse_and()
        while (tok := self._peek()) is not None and tok.kind == "OR":
            self._consume("OR")
            right = self._parse_and()
            node = Or(node, right)
        return node

    def _parse_and(self) -> QueryNode:
        node = self._parse_not()
        while (tok := self._peek()) is not None:
            if tok.kind == "AND":
                self._consume("AND")
                right = self._parse_not()
                node = And(node, right)
            elif tok.kind in {"TERM", "PHRASE", "LPAREN", "NOT"}:
                right = self._parse_not()
                node = And(node, right)
            else:
                break
        return node

    def _parse_not(self) -> QueryNode:
        tok = self._peek()
        if tok is not None and tok.kind == "NOT":
            self._consume("NOT")
            return Not(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> QueryNode:
        tok = self._peek()
        if tok is None:
            raise QueryParseError("Expected term, phrase, or '('")
        if tok.kind == "LPAREN":
            self._consume("LPAREN")
            inner = self._parse_or()
            self._consume("RPAREN")
            return inner
        if tok.kind == "PHRASE":
            self._consume("PHRASE")
            words = tuple(tokenize(tok.value))
            if not words:
                raise QueryParseError("Empty phrase quotes")
            return Phrase(words)
        if tok.kind == "TERM":
            self._consume("TERM")
            words = list(tokenize(tok.value))
            if not words:
                raise QueryParseError(f"Term {tok.value!r} produced no indexable tokens")
            node: QueryNode = Term(words[0])
            for extra in words[1:]:
                node = And(node, Term(extra))
            return node
        raise QueryParseError(f"Unexpected token {tok.value!r}")


def parse(query: str) -> QueryNode:
    """Parse a query string into a QueryNode AST."""
    tokens = _lex(query)
    return _Parser(tokens).parse()
