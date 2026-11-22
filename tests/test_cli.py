"""CLI tests using typer.testing.CliRunner (Lab 4–5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "index" in res.stdout
    assert "search" in res.stdout
    assert "stats" in res.stdout


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_cli_index_executors_and_search_table(
    tiny_corpus: Path,
    tmp_path: Path,
    executor: str,
) -> None:
    out_idx = tmp_path / f"cli_index_{executor}.bin"
    res_idx = runner.invoke(
        app,
        [
            "index",
            str(tiny_corpus),
            "--out",
            str(out_idx),
            "--positions",
            "--workers",
            "2",
            "--executor",
            executor,
        ],
    )
    assert res_idx.exit_code == 0
    assert out_idx.exists()
    assert "Index Built" in res_idx.stdout

    res_search = runner.invoke(
        app,
        ["search", str(out_idx), 'python AND (async OR await) "event loop"', "--k", "5"],
    )
    assert res_search.exit_code == 0
    assert "Search Results" in res_search.stdout
    assert "BM25" in res_search.stdout


def test_cli_search_json_stdout_clean_even_with_vv(saved_index_path: Path) -> None:
    res = runner.invoke(
        app,
        [
            "-vv",
            "search",
            str(saved_index_path),
            "nasa OR shuttle",
            "--scorer",
            "tfidf",
            "--json",
        ],
    )
    assert res.exit_code == 0
    lines = [line for line in res.stdout.strip().splitlines() if line]
    assert len(lines) >= 1
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["doc_id"] == 6
    assert "score" in parsed[0] and "snippet" in parsed[0]
    assert "DEBUG" in res.stderr or "INFO" in res.stderr


def test_cli_stats_on_index_and_corpus(saved_index_path: Path, tiny_corpus: Path) -> None:
    res_idx = runner.invoke(app, ["stats", str(saved_index_path), "--top", "5"])
    assert res_idx.exit_code == 0
    assert "Documents" in res_idx.stdout

    res_corpus = runner.invoke(app, ["stats", str(tiny_corpus), "--top", "5"])
    assert res_corpus.exit_code == 0
    assert "Documents" in res_corpus.stdout


def test_cli_missing_index_exits_nonzero_without_traceback(tmp_path: Path) -> None:
    res = runner.invoke(app, ["search", str(tmp_path / "missing.bin"), "python"])
    assert res.exit_code != 0
    combined = res.stderr + res.stdout
    assert "Traceback" not in combined
    assert "Error:" in combined


def test_cli_invalid_query_exits_nonzero_without_traceback(saved_index_path: Path) -> None:
    res = runner.invoke(app, ["search", str(saved_index_path), "(python OR"])
    assert res.exit_code != 0
    combined = res.stderr + res.stdout
    assert "Traceback" not in combined
    assert "Invalid query syntax" in combined
