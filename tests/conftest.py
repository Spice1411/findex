"""Shared pytest fixtures with a small deterministic in-memory/temp corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from findex.index import build_index
from findex.models import Index
from findex.store import save_index


@pytest.fixture(scope="session")
def tiny_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a small deterministic JSONL corpus once per test session."""
    root = tmp_path_factory.mktemp("corpus")
    corpus = root / "corpus.jsonl"
    long_filler = " ".join(["filler"] * 120)
    rows = [
        {"id": "sci.space/1", "text": "Python async event loop tutorial for fast servers"},
        {"id": "sci.space/2", "text": "Python await event loop and coroutines"},
        {"id": "comp.lang/3", "text": "Java event loop without python"},
        {"id": "short/4", "text": "Quantum rocket"},
        {"id": "long/5", "text": f"Quantum {long_filler} engine"},
        {"id": "common/6", "text": "Python common common common"},
        {"id": "sci.space/7", "text": "NASA space shuttle orbit telemetry [mission-7]"},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return corpus


@pytest.fixture()
def tiny_index(tiny_corpus: Path) -> Index:
    """Build a fresh positional Index over the tiny corpus (isolated per test)."""
    return build_index(tiny_corpus, with_positions=True)


@pytest.fixture(scope="session")
def saved_index_path(tiny_corpus: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Persist a session-scoped positional index file for CLI and loader tests."""
    out_dir = tmp_path_factory.mktemp("index_out")
    idx_path = out_dir / "index.bin"
    idx = build_index(tiny_corpus, with_positions=True)
    save_index(idx, idx_path)
    return idx_path
