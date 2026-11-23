"""Tests for Lab 7 FastAPI web API, Jinja2 UI, Pydantic validation, and error handling."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.models import Index
from findex.web.app import create_app, get_optional_index
from findex.web.config import Settings


@pytest.fixture()
def web_client(tiny_index: Index) -> Iterator[TestClient]:
    """TestClient injecting tiny_index via dependency_overrides (no real index file on disk)."""
    application = create_app(Settings(index_path=Path("nonexistent.bin"), allow_unloaded=True))
    application.dependency_overrides[get_optional_index] = lambda: tiny_index
    with TestClient(application, raise_server_exceptions=False) as client:
        yield client


def test_health_ok_and_503_when_unloaded(tiny_index: Index) -> None:
    app_unloaded = create_app(Settings(index_path=Path("nonexistent.bin"), allow_unloaded=True))
    app_unloaded.dependency_overrides[get_optional_index] = lambda: None

    with TestClient(app_unloaded) as client:
        resp_503 = client.get("/health")
        assert resp_503.status_code == 503
        body = resp_503.json()
        assert body["status"] == "unhealthy"
        assert "not loaded" in body["detail"].lower()

        # Also /search returns 503 when index is unloaded
        assert client.get("/search", params={"q": "python"}).status_code == 503

    app_unloaded.dependency_overrides[get_optional_index] = lambda: tiny_index
    with TestClient(app_unloaded) as client:
        resp_200 = client.get("/health")
        assert resp_200.status_code == 200
        assert resp_200.json()["status"] == "ok"
        assert resp_200.json()["num_docs"] == tiny_index.num_docs


def test_search_success_pagination_and_scorers(web_client: TestClient) -> None:
    resp = web_client.get("/search", params={"q": "python", "k": 2, "page": 1, "scorer": "bm25"})
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    data = resp.json()
    assert data["query"] == "python"
    assert data["scorer"] == "bm25"
    assert data["total"] == 4
    assert data["total_pages"] == 2
    assert len(data["results"]) == 2
    assert data["took_ms"] >= 0.0

    # Page 2 with the same scorer returns the remaining 2 distinct documents
    resp_p2 = web_client.get("/search", params={"q": "python", "k": 2, "page": 2, "scorer": "bm25"})
    assert resp_p2.status_code == 200
    data_p2 = resp_p2.json()
    assert data_p2["scorer"] == "bm25"
    assert len(data_p2["results"]) == 2
    ids_p1 = {item["doc_id"] for item in data["results"]}
    ids_p2 = {item["doc_id"] for item in data_p2["results"]}
    assert ids_p1.isdisjoint(ids_p2)

    resp_tfidf = web_client.get("/search", params={"q": "python", "k": 2, "scorer": "tfidf"})
    assert resp_tfidf.status_code == 200
    assert resp_tfidf.json()["scorer"] == "tfidf"
    assert len(resp_tfidf.json()["results"]) == 2


@pytest.mark.parametrize(
    "params",
    [
        {"q": "python", "k": "abc"},
        {"q": "python", "k": 0},
        {"q": "python", "k": 101},
        {"q": "python", "page": 0},
        {"q": "python", "scorer": "unknown"},
        {"q": ""},
        {"q": "(python OR"},
    ],
)
def test_search_validation_returns_422(
    web_client: TestClient,
    params: dict[str, object],
) -> None:
    resp = web_client.get("/search", params=params)
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_get_document_api_and_404(web_client: TestClient) -> None:
    resp = web_client.get("/docs/0")
    assert resp.status_code == 200
    doc = resp.json()
    assert doc["doc_id"] == 0
    assert "Python" in doc["text"]
    assert doc["token_count"] > 0

    missing = web_client.get("/docs/999999")
    assert missing.status_code == 404
    assert "not found" in missing.json()["detail"].lower()


def test_openapi_docs_and_stats(web_client: TestClient) -> None:
    docs_page = web_client.get("/docs")
    assert docs_page.status_code == 200
    assert "swagger" in docs_page.text.lower() or "openapi" in docs_page.text.lower()

    stats = web_client.get("/stats")
    assert stats.status_code == 200
    body = stats.json()
    assert body["num_docs"] == 7
    assert body["num_terms"] > 0
    assert body["has_positions"] is True


def test_web_ui_home_htmx_fragment_document_and_css(web_client: TestClient) -> None:
    home = web_client.get("/", params={"q": "python", "k": 2, "page": 1})
    assert home.status_code == 200
    assert "<mark>" in home.text
    assert "Next" in home.text

    frag = web_client.get("/partials/results", params={"q": "quantum"})
    assert frag.status_code == 200
    assert "<mark>Quantum</mark>" in frag.text

    doc_page = web_client.get("/doc/0", params={"q": "python"})
    assert doc_page.status_code == 200
    assert "Back to results" in doc_page.text
    assert web_client.get("/doc/999999").status_code == 404

    css = web_client.get("/static/style.css")
    assert css.status_code == 200
    assert "text/css" in css.headers.get("content-type", "")


def test_unhandled_exception_returns_clean_500_without_stacktrace(tiny_index: Index) -> None:
    application = create_app(Settings(index_path=Path("nonexistent.bin"), allow_unloaded=True))
    application.dependency_overrides[get_optional_index] = lambda: tiny_index

    @application.get("/boom")
    def _boom() -> dict[str, str]:
        raise RuntimeError("secret internal failure")

    with TestClient(application, raise_server_exceptions=False) as client:
        resp = client.get("/boom")
        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"] == "Internal server error"
        assert "request_id" in body
        assert "secret internal failure" not in resp.text
        assert "Traceback" not in resp.text


def test_settings_reads_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDEX_PATH", "/tmp/custom_index.bin")
    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    cfg = Settings()
    assert cfg.index_path == Path("/tmp/custom_index.bin")
    assert cfg.port == 9090
    assert cfg.log_level == "DEBUG"
