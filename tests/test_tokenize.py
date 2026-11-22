"""Parametrized unit tests for the streaming tokenizer and lazy corpus loader."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def test_functions_are_generators() -> None:
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PyThOn Straße STRASSE", ["python", "strasse", "strasse"]),
        (
            "Привіт, Світ! П'ять яблук і м’яч.",
            ["привіт", "світ", "п'ять", "яблук", "і", "м'яч"],
        ),
        ("café cafe\u0301", ["café", "café"]),
        ("Don't stop — 'quoted' word's end'", ["don't", "stop", "quoted", "word's", "end"]),
        (
            "state-of-the-art -- f-16 -leading trailing-",
            ["state-of-the-art", "f-16", "leading", "trailing"],
        ),
        ("Python 3.13 released in 2026, v2", ["python", "3", "13", "released", "in", "2026", "v2"]),
        ("hello, world! foo_bar... (test)?", ["hello", "world", "foo", "bar", "test"]),
        ("", []),
        ("   \n\t  ... --- !!! ", []),
    ],
)
def test_tokenize_cases(raw: str, expected: list[str]) -> None:
    assert list(tokenize(raw)) == expected


def test_iter_documents_lazy_and_bad_utf8(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("Hello world", encoding="utf-8")
    (tmp_path / "bad.txt").write_bytes(b"Hello \xff\xfe world")

    gen = iter_documents(tmp_path)
    first = next(gen)
    second = next(gen)
    texts = {first.path.name: first.text, second.path.name: second.text}

    assert texts["good.txt"] == "Hello world"
    assert "Hello" in texts["bad.txt"] and "world" in texts["bad.txt"]


def test_iter_documents_jsonl_and_malformed_lines(tmp_path: Path) -> None:
    jsonl = tmp_path / "corpus.jsonl"
    lines = [
        json.dumps({"id": "doc-1", "text": "First post"}),
        "{not valid json",
        json.dumps([1, 2, 3]),
        json.dumps({"id": "doc-2", "text": "Second post"}),
    ]
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    docs = list(iter_documents(tmp_path))
    assert [d.doc_id for d in docs] == ["doc-1", "doc-2"]
    assert [d.text for d in docs] == ["First post", "Second post"]


def test_iter_documents_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_documents(tmp_path / "missing.jsonl"))
