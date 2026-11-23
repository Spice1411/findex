"""FastAPI web service, Jinja2/htmx UI, lifespan index loader, and ASGI middleware (Lab 7)."""

from __future__ import annotations

import logging
import math
import os
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from findex.models import Index
from findex.query import QueryParseError
from findex.ranking import make_scorer
from findex.search import search_paginated
from findex.snippets import load_document_text
from findex.store import load_index
from findex.web.config import Settings
from findex.web.schemas import (
    DocumentResponse,
    HealthResponse,
    SearchHit,
    SearchResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


def get_settings(request: Request) -> Settings:
    """FastAPI dependency returning active Settings from ``app.state``."""
    cfg: object = getattr(request.app.state, "settings", None)
    if isinstance(cfg, Settings):
        return cfg
    return Settings()


def get_optional_index(request: Request) -> Index | None:
    """FastAPI dependency returning the loaded Index, or ``None`` if unavailable."""
    idx: object = getattr(request.app.state, "index", None)
    return idx if isinstance(idx, Index) else None


def get_index(
    index: Annotated[Index | None, Depends(get_optional_index)],
) -> Index:
    """FastAPI dependency requiring a loaded Index (raises HTTP 503 if unloaded)."""
    if index is None:
        raise HTTPException(status_code=503, detail="Search index is not loaded")
    return index


def _uptime(request: Request) -> float:
    started: object = getattr(request.app.state, "started_at", None)
    if isinstance(started, float):
        return round(max(0.0, time.monotonic() - started), 2)
    return 0.0


def _index_size_bytes(request: Request) -> int:
    size: object = getattr(request.app.state, "index_size_bytes", 0)
    return size if isinstance(size, int) else 0


def _run_api_search(
    index: Index,
    q: str,
    k: int,
    scorer: Literal["bm25", "tfidf"],
    page: int,
) -> SearchResponse:
    t0 = time.perf_counter()
    try:
        total, domain_results = search_paginated(
            index,
            q,
            scorer=make_scorer(scorer),
            page=page,
            page_size=k,
            include_snippets=True,
            highlight_style="markdown",
        )
    except QueryParseError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid query syntax: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    took_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    total_pages = math.ceil(total / k) if total > 0 else 0
    return SearchResponse(
        query=q,
        scorer=scorer,
        page=page,
        k=k,
        total=total,
        total_pages=total_pages,
        took_ms=took_ms,
        results=[SearchHit.from_domain(item) for item in domain_results],
    )


def _run_ui_search(
    index: Index,
    q: str,
    k: int,
    scorer: Literal["bm25", "tfidf"],
    page: int,
) -> dict[str, object]:
    t0 = time.perf_counter()
    try:
        total, domain_results = search_paginated(
            index,
            q,
            scorer=make_scorer(scorer),
            page=page,
            page_size=k,
            include_snippets=True,
            highlight_style="html",
        )
        error_msg: str | None = None
    except (QueryParseError, ValueError) as exc:
        total = 0
        domain_results = []
        error_msg = str(exc)

    took_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    total_pages = math.ceil(total / k) if total > 0 else 0
    return {
        "query": q,
        "scorer": scorer,
        "page": page,
        "k": k,
        "total": total,
        "total_pages": total_pages,
        "took_ms": took_ms,
        "error": error_msg,
        "results": [SearchHit.from_domain(item) for item in domain_results],
    }


def create_app(
    settings: Settings | None = None,
    *,
    async_search: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application instance.

    By default ``GET /search`` is registered as a plain ``def`` route so FastAPI
    executes CPU-bound BM25/TF-IDF scoring in its thread pool without blocking the
    event loop. Pass ``async_search=True`` only to benchmark blocking ``async def``.
    """
    cfg = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        application.state.settings = cfg
        application.state.started_at = time.monotonic()
        application.state.index = None
        application.state.index_size_bytes = 0

        idx_path = cfg.index_path
        if idx_path.exists():
            logger.info("Loading search index from %s", idx_path)
            loaded = load_index(idx_path)
            application.state.index = loaded
            application.state.index_size_bytes = idx_path.stat().st_size
            logger.info(
                "Loaded %r (%d bytes)",
                loaded,
                application.state.index_size_bytes,
            )
        elif cfg.allow_unloaded:
            logger.warning(
                "Index path %s does not exist; starting unloaded (/health will return 503)",
                idx_path,
            )
        else:
            raise FileNotFoundError(
                f"INDEX_PATH does not exist: {idx_path}. "
                "Build an index first or set ALLOW_UNLOADED=true."
            )

        try:
            yield
        finally:
            active: object = getattr(application.state, "index", None)
            if isinstance(active, Index):
                active.close()
                logger.info("Closed search index cleanly on shutdown")

    application = FastAPI(
        title="findex Search Engine",
        description="Full-text BM25 & TF-IDF search engine API and Web UI.",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = cfg
    application.state.started_at = time.monotonic()
    application.state.index = None
    application.state.index_size_bytes = 0

    application.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @application.middleware("http")
    async def request_id_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = req_id
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.exception(
                "request_id=%s %s %s -> 500 (%.1f ms)",
                req_id,
                request.method,
                request.url.path,
                elapsed_ms,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": req_id},
                headers={"X-Request-ID": req_id},
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response.headers["X-Request-ID"] = req_id
        logger.info(
            "request_id=%s %s %s -> %d (%.1f ms)",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id: object = getattr(request.state, "request_id", "unknown")
        logger.exception("request_id=%s unhandled error: %s", req_id, exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": str(req_id)},
            headers={"X-Request-ID": str(req_id)},
        )

    # --- JSON API routes ---
    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health_route(
        request: Request,
        index: Annotated[Index | None, Depends(get_optional_index)],
    ) -> Response:
        up = _uptime(request)
        if index is None:
            payload = HealthResponse(
                status="unhealthy",
                num_docs=0,
                uptime_seconds=up,
                detail="Search index is not loaded",
            )
            return JSONResponse(status_code=503, content=payload.model_dump())
        payload = HealthResponse(status="ok", num_docs=index.num_docs, uptime_seconds=up)
        return JSONResponse(status_code=200, content=payload.model_dump())

    @application.get("/stats", response_model=StatsResponse, tags=["system"])
    def stats_route(
        request: Request,
        index: Annotated[Index, Depends(get_index)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> StatsResponse:
        logger.debug("Serving /stats for index %s", settings.index_path)
        return StatsResponse(
            num_docs=index.num_docs,
            num_terms=index.num_terms,
            total_tokens=index.total_tokens,
            avg_doc_length=round(index.avg_doc_length, 2),
            has_positions=index.has_positions,
            index_size_bytes=_index_size_bytes(request),
            uptime_seconds=_uptime(request),
        )

    if async_search:

        @application.get("/search", response_model=SearchResponse, tags=["search"])
        async def search_route_async(
            index: Annotated[Index, Depends(get_index)],
            q: Annotated[str, Query(min_length=1, max_length=200, description="Search query")],
            k: Annotated[int, Query(ge=1, le=100, description="Results per page")] = 10,
            scorer: Annotated[Literal["bm25", "tfidf"], Query(description="Ranking scorer")] = (
                "bm25"
            ),
            page: Annotated[int, Query(ge=1, description="1-based page number")] = 1,
        ) -> SearchResponse:
            return _run_api_search(index, q=q, k=k, scorer=scorer, page=page)

    else:

        @application.get("/search", response_model=SearchResponse, tags=["search"])
        def search_route(
            index: Annotated[Index, Depends(get_index)],
            q: Annotated[str, Query(min_length=1, max_length=200, description="Search query")],
            k: Annotated[int, Query(ge=1, le=100, description="Results per page")] = 10,
            scorer: Annotated[Literal["bm25", "tfidf"], Query(description="Ranking scorer")] = (
                "bm25"
            ),
            page: Annotated[int, Query(ge=1, description="1-based page number")] = 1,
        ) -> SearchResponse:
            return _run_api_search(index, q=q, k=k, scorer=scorer, page=page)

    @application.get("/docs/{doc_id}", response_model=DocumentResponse, tags=["documents"])
    def get_document_api(
        doc_id: int,
        index: Annotated[Index, Depends(get_index)],
    ) -> DocumentResponse:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        full_text = load_document_text(meta)
        return DocumentResponse(
            doc_id=meta.doc_id,
            source_id=meta.source_id,
            title=meta.title,
            token_count=index.doc_length(doc_id),
            text=full_text,
        )

    # --- Server-rendered Jinja2 + htmx Web UI ---
    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui_home(
        request: Request,
        index: Annotated[Index | None, Depends(get_optional_index)],
        q: Annotated[str, Query(max_length=200)] = "",
        k: Annotated[int, Query(ge=1, le=50)] = 10,
        scorer: Annotated[Literal["bm25", "tfidf"], Query()] = "bm25",
        page: Annotated[int, Query(ge=1)] = 1,
    ) -> HTMLResponse:
        cleaned_q = q.strip()
        context: dict[str, object] = {
            "query": cleaned_q,
            "scorer": scorer,
            "k": k,
            "page": page,
            "index_ready": index is not None,
            "num_docs": index.num_docs if index is not None else 0,
            "num_terms": index.num_terms if index is not None else 0,
            "results": [],
            "total": 0,
            "total_pages": 0,
            "took_ms": 0.0,
            "error": None,
        }
        if cleaned_q and index is not None:
            context.update(_run_ui_search(index, q=cleaned_q, k=k, scorer=scorer, page=page))
        return templates.TemplateResponse(request=request, name="index.html", context=context)

    @application.get("/partials/results", response_class=HTMLResponse, include_in_schema=False)
    def ui_results_fragment(
        request: Request,
        index: Annotated[Index | None, Depends(get_optional_index)],
        q: Annotated[str, Query(max_length=200)] = "",
        k: Annotated[int, Query(ge=1, le=50)] = 10,
        scorer: Annotated[Literal["bm25", "tfidf"], Query()] = "bm25",
        page: Annotated[int, Query(ge=1)] = 1,
    ) -> HTMLResponse:
        cleaned_q = q.strip()
        if not cleaned_q or index is None:
            ctx: dict[str, object] = {
                "query": cleaned_q,
                "scorer": scorer,
                "k": k,
                "page": page,
                "results": [],
                "total": 0,
                "total_pages": 0,
                "took_ms": 0.0,
                "error": None if index is not None else "Search index is not loaded.",
            }
        else:
            ctx = _run_ui_search(index, q=cleaned_q, k=k, scorer=scorer, page=page)
        return templates.TemplateResponse(request=request, name="_results.html", context=ctx)

    @application.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def ui_document_view(
        request: Request,
        doc_id: int,
        index: Annotated[Index, Depends(get_index)],
        q: Annotated[str, Query(max_length=200)] = "",
    ) -> HTMLResponse:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        full_text = load_document_text(meta)
        return templates.TemplateResponse(
            request=request,
            name="document.html",
            context={
                "doc_id": meta.doc_id,
                "source_id": meta.source_id,
                "title": meta.title,
                "token_count": index.doc_length(doc_id),
                "text": full_text,
                "back_query": q,
            },
        )

    return application


_ASYNC_MODE = os.environ.get("FINDEX_ASYNC_SEARCH", "").lower() in {"1", "true", "yes"}
app = create_app(async_search=_ASYNC_MODE)
