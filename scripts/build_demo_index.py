"""Build a compact self-contained demo corpus & positional index for Docker / cloud deploy.

Usage:
    uv run python scripts/build_demo_index.py --out data/demo_index.bin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findex.index import build_index
from findex.store import save_index

TOPICS: list[tuple[str, str, str]] = [
    (
        "python",
        "Python Data Model & Protocols",
        (
            "Python dunder methods such as __len__, __getitem__, and __contains__ let custom "
            "containers integrate natively with built-in functions and collections.abc.Mapping. "
            "Structural typing with typing.Protocol allows pyright strict mode to verify "
            "interchangeable BM25 and TF-IDF scorers without brittle inheritance hierarchies."
        ),
    ),
    (
        "fastapi",
        "FastAPI, Pydantic & ASGI Architecture",
        (
            "FastAPI builds on Starlette and Uvicorn ASGI to turn Python type hints into runtime "
            "request validation, automatic HTTP 422 responses, and interactive OpenAPI docs at "
            "/docs. Pydantic validates untrusted boundary data while internal dataclasses stay "
            "lightweight. Plain def routes offload CPU-bound BM25 search to a thread pool so the "
            "event loop never stalls."
        ),
    ),
    (
        "asyncio",
        "Asyncio Event Loop & Structured Concurrency",
        (
            "Cooperative concurrency lets a single OS thread multiplex hundreds of in-flight "
            "network sockets at await points using an event loop. Structured concurrency with "
            "asyncio.TaskGroup guarantees sibling cancellation on failure, while global and "
            "per-host semaphores keep web crawlers polite and respectful of robots.txt."
        ),
    ),
    (
        "bm25",
        "Okapi BM25 & TF-IDF Information Retrieval",
        (
            "Okapi BM25 improves classic TF-IDF with term-frequency saturation (controlled by k1) "
            "so the twentieth repetition of a word adds almost nothing, and document-length "
            "normalization (controlled by b) so concise relevant documents outrank noisy long "
            "pages. Top-k ranking uses heapq.nlargest in O(n log k) time."
        ),
    ),
    (
        "gil",
        "CPython GIL, Multiprocessing & Free-Threaded 3.13t",
        (
            "The Global Interpreter Lock protects CPython reference counts by letting one thread "
            "run Python bytecode at a time. I/O-bound workloads release the GIL during socket "
            "reads, while CPU-bound indexing needs ProcessPoolExecutor or free-threaded Python "
            "3.13t (PEP 703). Amdahl's law bounds speedup by sequential merge time."
        ),
    ),
    (
        "docker",
        "Multi-Stage Docker Containers & 12-Factor Deployment",
        (
            "A multi-stage uv Dockerfile installs locked dependencies from uv.lock in a builder "
            "stage and copies only the slim runtime environment into a non-root container. "
            "Twelve-factor configuration reads INDEX_PATH, PORT, and LOG_LEVEL from environment "
            "variables, while /health probes allow Render, Fly.io, and Railway to verify readiness."
        ),
    ),
    (
        "space",
        "NASA Space Shuttle & Orbital Mechanics",
        (
            "NASA launched space shuttle missions into low Earth orbit to deploy satellites, "
            "service the Hubble telescope, and assemble the International Space Station. "
            "Telemetry pipelines monitor propulsion, thermal tiles, and orbital insertion windows."
        ),
    ),
    (
        "cryptography",
        "Public-Key Cryptography & Clipper Chip History",
        (
            "Modern TLS relies on elliptic-curve key exchange, AES-GCM authenticated encryption, "
            "and cryptographic hash functions. In the 1990s the Clipper chip key-escrow proposal "
            "sparked intense debate on Usenet sci.crypt about privacy and backdoors."
        ),
    ),
    (
        "testing",
        "Pytest Fixtures, Parametrize & Hypothesis Properties",
        (
            "Effective pytest suites hoist I/O out of pure logic, share deterministic fixture "
            "corpora via conftest.py, and swap dependencies with FastAPI dependency_overrides. "
            "Hypothesis property-based tests verify invariants like save/load round-trips and "
            "sorted merge oracles, automatically shrinking counterexamples."
        ),
    ),
    (
        "indexing",
        "Streaming Inverted Index & Positional Phrase Search",
        (
            "A streaming generator pipeline reads documents lazily with constant RAM, normalizes "
            "Unicode NFC text with casefold, and records sorted postings lists. Storing token "
            "positions enables two-pointer adjacent phrase search such as exact 'event loop' "
            "queries and ±80-character highlighted snippets."
        ),
    ),
]

SUBCONTEXTS: list[str] = [
    "Production latency analysis focusing on p50, p95, and p99 percentiles under oha load testing.",
    "Architectural walkthrough comparing single-worker async def, thread-pool def, and 4 workers.",
    "Hands-on tutorial with curl examples, JSON schemas, and clean error handling without stacks.",
    "Memory profiling study contrasting plain dataclasses, slots=True, and compact array('I').",
    "Operational runbook covering structured request-id logging, graceful shutdown, and /health.",
    "Deep-dive engineering note connecting streaming generators, LRU query caches, and snippets.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate demo corpus and build demo index.")
    parser.add_argument("--corpus", type=Path, default=Path("data/demo_corpus.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/demo_index.bin"))
    parser.add_argument("--docs", type=int, default=300)
    args = parser.parse_args()

    args.corpus.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.corpus.open("w", encoding="utf-8") as handle:
        for i in range(args.docs):
            tag, heading, body = TOPICS[i % len(TOPICS)]
            sub = SUBCONTEXTS[(i // len(TOPICS)) % len(SUBCONTEXTS)]
            secondary_tag, _, secondary_body = TOPICS[(i * 3 + 1) % len(TOPICS)]
            title = f"{heading} — Note #{i + 1}"
            text = (
                f"{title}\n\n"
                f"{body} {sub}\n\n"
                f"Related topic ({secondary_tag}): {secondary_body}\n"
                f"Tags: findex {tag} {secondary_tag} guide-{i % 15} release-0.7"
            )
            record = {"id": f"{tag}/doc-{i + 1:03d}", "text": text}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    idx = build_index(args.corpus, with_positions=True, workers=1, executor="serial")
    save_index(idx, args.out)
    size_kb = args.out.stat().st_size / 1024
    print(f"Built demo index {idx!r} -> {args.out} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
