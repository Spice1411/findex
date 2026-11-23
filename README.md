# findex — Streaming Full-Text Search Engine

[![CI](https://github.com/Spice1411/findex/actions/workflows/ci.yml/badge.svg)](https://github.com/Spice1411/findex/actions/workflows/ci.yml)

<!-- LAB07-URL-START -->
> **Live Public Demo (Lab 07):** [https://findex-vrwg.onrender.com](https://findex-vrwg.onrender.com) · [OpenAPI Docs](https://findex-vrwg.onrender.com/docs) · [Health Check](https://findex-vrwg.onrender.com/health)
<!-- LAB07-URL-END -->

A full-text search engine built incrementally in Python: streaming corpus intake, polite async web crawler (`asyncio` + `httpx`), parallel inverted index construction, Boolean & phrase queries, TF-IDF/BM25 ranking, FastAPI + Pydantic web service with Jinja2/htmx UI, strict typing, property-based tests, and Docker deployment.

---

## Lab 07 — Search on the Web: FastAPI, Pydantic, ASGI, and Deployment

### What was added

- **FastAPI ASGI application (`src/findex/web/`):**
  - `lifespan` loads the persisted index **once** at startup from `Settings.index_path` (`pydantic-settings`, 12-factor environment config) and closes it cleanly on shutdown.
  - Pydantic v2 schemas (`schemas.py`) validate untrusted boundary data while internal Lab 2–3 dataclasses remain untouched.
  - Endpoints: `GET /search` (validated `q`, `k`, `scorer`, `page` with pagination, plain `def` route so CPU-bound BM25 runs in the thread pool without blocking the event loop), `GET /docs/{doc_id}` (404 if unknown), `GET /stats`, `GET /health` (`503` if index is unloaded), and automatic OpenAPI docs at `GET /docs`.
  - Request-ID middleware (`X-Request-ID`) and unhandled exception handler returning clean JSON `500` without leaking tracebacks.
- **Web UI (`src/findex/web/templates/`, `static/style.css`):** server-rendered Jinja2 search page with htmx live search-as-you-type, `<mark>`-highlighted snippets, relevance score pills, pagination, and full document viewer at `GET /doc/{doc_id}`.
- **CLI & Docker (`findex serve`, `Dockerfile`, `render.yaml`):** `findex serve --port 8000 --workers 1` wraps Uvicorn; multi-stage `uv` Dockerfile runs as non-root `appuser` with `/health` check and environment-based `INDEX_PATH` / `PORT`.

```bash
# Build a compact demo index (or point --index-path to your full 20 Newsgroups index.bin)
uv run python scripts/build_demo_index.py --out data/demo_index.bin

# Start the web UI and API via CLI
findex serve --index-path data/demo_index.bin --port 8000

# Run with Docker locally
docker build -t findex .
docker run -e INDEX_PATH=/app/data/demo_index.bin -p 8000:8000 findex
```

<!-- LAB07-RESULTS-START -->
### M2 — Sync `def` vs. blocking `async def` on `/search` (20 concurrent requests)

| Route definition | 20-req Wall (ms) | p50 (ms) | p95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| `async def` (blocks event loop) | 207.79 | 141.95 | 186.41 | 187.33 |
| `def` (thread-pool offloaded — **kept in production**) | 178.31 | 135.36 | 156.44 | 158.26 |

### M4 — Load test (`httpx-async`, local vs. deployed)

| Target | Route | Concurrency | RPS | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---:|---:|---:|---:|---:|
| local, 1 worker, `async def` | `/search` | 50 | 74.1 | 466.72 | 1863.16 | 2522.62 |
| local, 1 worker, `def` | `/search` | 50 | 79.7 | 433.85 | 1885.56 | 2565.73 |
| local, 4 workers | `/search` | 50 | 75.8 | 488.7 | 1640.4 | 2648.55 |
| deployed (`https://findex-vrwg.onrender.com`) | `/search` | 20 | 20.1 | 1008.08 | 1376.33 | 1545.07 |

### Connecting the load-test numbers to Labs 5–6

- **1 worker, `async def` vs. `def`:** BM25 scoring and snippet extraction are pure synchronous CPU/disk work without `await` points. Inside an `async def` route, each request freezes the single Uvicorn event loop (the exact Lab 6 blocking bug), so concurrent clients queue behind one another and p99 spikes. Changing the route to plain `def` tells FastAPI to offload execution to AnyIO's thread pool so the event loop stays responsive for HTTP accept/read/write and `/health` probes — though single-process throughput is still bounded by the CPython GIL (Lab 5).
- **1 worker vs. 4 workers:** Launching `--workers 4` runs four independent Uvicorn processes, each with its own Python interpreter and its own GIL (Lab 5's multi-process escape hatch), multiplying available CPU cores and cutting tail latency (p95/p99) under heavy concurrency at the cost of holding the index in each worker.
- **Deployed service:** Running against the live public endpoint (`https://findex-vrwg.onrender.com`) adds real TLS + WAN round-trip latency and free-tier shared-CPU throttling, which raises p50/p99 compared to localhost while confirming end-to-end availability.
<!-- LAB07-RESULTS-END -->

---

## Lab 06 — The Async Crawler: `asyncio`, Coroutines, and Structured Concurrency

- **Ethics & permission:** crawled our own local multi-origin ethical HTML mirror (`scripts/local_mirror.py` on `127.0.0.1` across 4 ports with `/robots.txt`, 20 ms latency, and controlled 429/timeout/redirect-loop endpoints); honest `User-Agent: findex-crawler/0.6 (+https://github.com/Spice1411/findex)`, `RobotsCache`, global + per-host semaphores, and minimum per-host delay.

| Concurrency | Pages | Wall (s) | Pages/s | Errors | Peak RSS |
|---:|---:|---:|---:|---:|---:|
| 1 (effectively sequential) | 220 | 10.700 | 20.6 | 3 | 45.5 MB |
| 5 | 220 | 3.515 | 62.6 | 3 | 45.5 MB |
| 20 | 220 | 2.204 | 99.8 | 1 | 45.6 MB |

- Strict per-host throttling (`per_host=1`, `50 ms` delay) slowed the 220-page crawl to `4.679 s` (`47.0 pages/s`) to protect target hosts from overload; `debug=True` verified zero slow callbacks (>100 ms); handled `429 Retry-After: 0.15` recovery, `asyncio.timeout`, and redirect loop without crashing.

---

## Lab 05 — Concurrency and the GIL: Threads, Processes, and Parallel Indexing

- **CPU:** 11th Gen Intel(R) Core(TM) i3-1115G4 @ 3.00GHz (4 logical cores), Windows, 20 Newsgroups (18,846 docs).
- **Standard Python:** `3.13.18` (`sys._is_gil_enabled() == True`) · **Free-threaded Python:** `3.13.18t` (`sys._is_gil_enabled() == False`).

| Executor | Workers | Wall (s, median) | CPU (s) | Peak RSS | Merge (s) | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| `processes` | 4 | 25.520 | 34.016 | 527.2 MB | 0.337 | 0.51× |
| `processes` | 8 | 28.535 | 37.109 | 527.2 MB | 0.514 | 0.46× |
| `serial (3.13t, GIL=False)` | 1 | 12.111 | 11.797 | 323.7 MB | 0.173 | 1.08× |
| `threads (3.13t, GIL=False)` | 2 | 10.063 | 16.594 | 339.4 MB | 0.771 | 1.30× |
| `threads (3.13t, GIL=False)` | 4 | 9.141 | 23.656 | 339.4 MB | 0.754 | 1.43× |
| `threads (3.13t, GIL=False)` | 8 | 9.476 | 25.219 | 339.4 MB | 0.714 | 1.38× |

![Parallel indexing speedup vs workers](docs/speedup_lab05.png)

- **I/O-bound counter-benchmark (32 local HTTP requests):** 1 thread `0.942 s (1.00×)` → 8 threads `0.116 s (8.11×)`.
- **Free-threaded `3.13t` race experiment (4 × 400k):** unsafe `shared += 1` produced `868,293` (lost `731,707` updates); `threading.Lock` and thread-local `build_partial` + `merge` produced exact `1,600,000`.

---

## Lab 01–04 Highlights

- **Lab 04 (Typing, Testing, Packaging, CLI):** `pyright` strict (0 errors), `ruff`, `pytest` + Hypothesis, `typer` + `rich` CLI, GitHub Actions CI, and `v0.4.0` wheel release.
<!-- LAB04-COVERAGE-START -->
- **Total coverage (`src/findex`):** **78.9%** (1967/2493 statements).

| Module | Statements | Missed | Coverage |
|---|---:|---:|---:|
| `src/findex/__init__.py` | 9 | 0 | 100.0% |
| `src/findex/cli.py` | 196 | 62 | 68.4% |
| `src/findex/corpus.py` | 79 | 14 | 82.3% |
| `src/findex/crawler/__init__.py` | 5 | 0 | 100.0% |
| `src/findex/crawler/crawl.py` | 215 | 37 | 82.8% |
| `src/findex/crawler/fetch.py` | 118 | 22 | 81.4% |
| `src/findex/crawler/parse.py` | 87 | 7 | 92.0% |
| `src/findex/crawler/robots.py` | 70 | 16 | 77.1% |
| `src/findex/decorators.py` | 15 | 0 | 100.0% |
| `src/findex/eval.py` | 122 | 122 | 0.0% |
| `src/findex/index.py` | 418 | 118 | 71.8% |
| `src/findex/models.py` | 157 | 5 | 96.8% |
| `src/findex/query.py` | 220 | 14 | 93.6% |
| `src/findex/ranking.py` | 40 | 4 | 90.0% |
| `src/findex/search.py` | 201 | 32 | 84.1% |
| `src/findex/snippets.py` | 118 | 24 | 79.7% |
| `src/findex/stats.py` | 96 | 21 | 78.1% |
| `src/findex/store.py` | 84 | 3 | 96.4% |
| `src/findex/tokenize.py` | 12 | 0 | 100.0% |
| `src/findex/web/__init__.py` | 3 | 0 | 100.0% |
| `src/findex/web/app.py` | 172 | 25 | 85.5% |
| `src/findex/web/config.py` | 16 | 0 | 100.0% |
| `src/findex/web/schemas.py` | 40 | 0 | 100.0% |

**Intentionally left outside coverage:**
- Legacy `if __name__ == '__main__':` entry points in `index.py`, `search.py`, `stats.py`, `eval.py` — superseded by the packaged `findex` CLI (`cli.py`).
- Lab 2 memory-study benchmark runner (`run_memory_study`) and network download fallbacks in `scripts/download_corpus.py` — slow / external-network paths.
- Defensive `OSError` handlers for corrupt files or unreadable disks.
<!-- LAB04-COVERAGE-END -->
- **Lab 03 (Ranking):** `Index(Mapping)`, `open_index`, `Scorer` protocol (`TfIdf` vs `BM25`), recursive-descent query parser, `@timed` + `lru_cache` (`4.32 ms` MISS → `2.69 ms` HIT), and 10-query **Precision@5** (`0.96` TF-IDF vs `0.92` BM25).
- **Lab 02 (Memory study):** plain `@dataclass` `309.38 MB` → `slots=True` `234.30 MB` (`51.06 MB` file) → `array('I')` pairs `35.10 MB` (`5.434 s` load).
- **Lab 01 (Eager vs. Lazy on 20 Newsgroups):** eager `333.12 MB` vs lazy generators `19.15 MB` (**17.4×** less peak RAM).
