"""Unit tests for the streaming tokenizer and lazy corpus loader."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def test_functions_are_generators() -> None:
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)


def test_mixed_case_and_casefold() -> None:
    assert list(tokenize("PyThOn Straße STRASSE")) == ["python", "strasse", "strasse"]


def test_cyrillic_and_ukrainian() -> None:
    text = "Привіт, Світ! П'ять яблук і м’яч."
    assert list(tokenize(text)) == ["привіт", "світ", "п'ять", "яблук", "і", "м'яч"]


def test_unicode_combining_mark_nfc() -> None:
    composed = "café"
    decomposed = "cafe\u0301"
    assert composed != decomposed
    assert list(tokenize(composed)) == list(tokenize(decomposed)) == ["café"]


def test_apostrophes_policy() -> None:
    text = "Don't stop — 'quoted' word's end'"
    assert list(tokenize(text)) == ["don't", "stop", "quoted", "word's", "end"]


def test_hyphens_policy() -> None:
    text = "state-of-the-art -- f-16 -leading trailing-"
    assert list(tokenize(text)) == ["state-of-the-art", "f-16", "leading", "trailing"]


def test_digits_and_alphanumeric() -> None:
    assert list(tokenize("Python 3.13 released in 2026, v2")) == [
        "python",
        "3",
        "13",
        "released",
        "in",
        "2026",
        "v2",
    ]


def test_punctuation_and_underscores() -> None:
    assert list(tokenize("hello, world! foo_bar... (test)?")) == [
        "hello",
        "world",
        "foo",
        "bar",
        "test",
    ]


def test_empty_and_whitespace() -> None:
    assert list(tokenize("")) == []
    assert list(tokenize("   \n\t  ... --- !!! ")) == []


def test_iter_documents_lazy_and_bad_utf8(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("Hello world", encoding="utf-8")
    (tmp_path / "bad.txt").write_bytes(b"Hello \xff\xfe world")

    gen = iter_documents(tmp_path)
    first = next(gen)
    second = next(gen)
    texts = {first.path.name: first.text, second.path.name: second.text}

    assert texts["good.txt"] == "Hello world"
    assert "Hello" in texts["bad.txt"] and "world" in texts["bad.txt"]


def test_iter_documents_jsonl_in_directory(tmp_path: Path) -> None:
    jsonl = tmp_path / "corpus.jsonl"
    lines = [
        json.dumps({"id": "doc-1", "text": "First post"}),
        json.dumps({"id": "doc-2", "text": "Second post"}),
    ]
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    docs = list(iter_documents(tmp_path))
    assert [d.doc_id for d in docs] == ["doc-1", "doc-2"]
    assert [d.text for d in docs] == ["First post", "Second post"]
