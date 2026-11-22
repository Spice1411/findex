# findex — Streaming Full-Text Search Engine

[![CI](https://github.com/Spice1411/findex/actions/workflows/ci.yml/badge.svg)](https://github.com/Spice1411/findex/actions/workflows/ci.yml)

A full-text search engine built incrementally in Python: streaming corpus intake, inverted index, Boolean & phrase queries, TF-IDF/BM25 ranking, strict typing, property-based tests, and a packaged Typer + Rich CLI.

---

## Installation & CLI (Lab 04)

### Install from wheel or GitHub Release (`v0.4.0`)

```bash
# Install locally from the repository
uv tool install .

# Or build a wheel and run in an isolated environment
uv build
uvx --from dist/findex-0.4.0-py3-none-any.whl findex --help

# Or install directly from the GitHub Release wheel
uv tool install https://github.com/Spice1411/findex/releases/download/v0.4.0/findex-0.4.0-py3-none-any.whl
```

### Commands

```bash
# Build a positional inverted index with a Rich progress bar
findex index data/ --out index.bin --positions

# Ranked BM25 search with Rich table & highlighted snippets
findex search index.bin 'nasa AND (shuttle OR orbit) NOT mars "space shuttle"' --k 5

# Machine-readable JSON Lines output to stdout (diagnostics stay on stderr)
findex -vv search index.bin 'nasa OR shuttle' --scorer tfidf --json

# Pretty corpus / index statistics
findex stats index.bin
```

### Quality checks & test coverage

- **Static typing:** `uv run pyright` (`typeCheckingMode = "strict"` in `pyproject.toml`, 0 errors).
- **Linting & formatting:** `uv run ruff check .` and `uv run ruff format --check .`.
- **Tests:** `uv run pytest` (≥ 30 pytest tests including 4 Hypothesis property-based tests, CLI tests via `CliRunner`, and `@pytest.mark.slow`).

<!-- LAB04-COVERAGE-START -->
- **Total coverage (`src/findex`):** **76.2%** (1076/1413 statements).

| Module | Statements | Missed | Coverage |
|---|---:|---:|---:|
| `src/findex/__init__.py` | 8 | 0 | 100.0% |
| `src/findex/cli.py` | 130 | 11 | 91.5% |
| `src/findex/corpus.py` | 73 | 13 | 82.2% |
| `src/findex/decorators.py` | 15 | 0 | 100.0% |
| `src/findex/eval.py` | 122 | 122 | 0.0% |
| `src/findex/index.py` | 183 | 93 | 49.2% |
| `src/findex/models.py` | 139 | 5 | 96.4% |
| `src/findex/query.py` | 220 | 14 | 93.6% |
| `src/findex/ranking.py` | 40 | 4 | 90.0% |
| `src/findex/search.py` | 192 | 31 | 83.9% |
| `src/findex/snippets.py` | 99 | 20 | 79.8% |
| `src/findex/stats.py` | 96 | 21 | 78.1% |
| `src/findex/store.py` | 84 | 3 | 96.4% |
| `src/findex/tokenize.py` | 12 | 0 | 100.0% |

**Intentionally left outside coverage:**
- Legacy `if __name__ == '__main__':` entry points in `index.py`, `search.py`, `stats.py`, `eval.py`) — superseded by the packaged `findex` CLI (`cli.py`).
- Lab 2 memory-study benchmark runner (`run_memory_study`) and network download fallbacks in `scripts/download_corpus.py` — slow / external-network paths.
- Defensive `OSError` handlers for corrupt files or unreadable disks.
<!-- LAB04-COVERAGE-END -->

---

## Lab 01 — Streams, Not Lists: Iterators, Generators, and Corpus Intake

### Corpus

- **Source:** [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html) (18,846 Usenet posts, 36.28 MB uncompressed, stored locally in `data/corpus.jsonl`).
- **Storage:** `data/` (excluded from Git via `.gitignore`).
- **How to download:** `uv run python scripts/download_corpus.py`

### Tokenizer policy (`src/findex/tokenize.py`)

- **Unicode normalization:** `unicodedata.normalize("NFC", text)` maps canonically equivalent sequences (`"café"` and `"cafe\u0301"`) to the same token.
- **Case folding:** `str.casefold()` for locale-independent matching (`"Straße"` → `"strasse"`).
- **Apostrophes & hyphens:** internal apostrophes (`don't`, `п'ять`, `’` → `'`) and internal hyphens (`state-of-the-art`, `f-16`) are kept; leading/trailing punctuation is stripped.
- **Regex:** `re.finditer(r"[^\W_]+(?:['\u2019-][^\W_]+)*", normalized)` streams tokens lazily.

### Lab 01 Benchmark: Eager vs. Lazy

| Version | Documents | Total tokens | Peak memory | Elapsed |
|---|---:|---:|---:|---:|
| eager (lists) | 18,846 | 5,812,698 | 333.12 MB | 28.557 s |
| lazy (generators) | 18,846 | 5,812,698 | 19.15 MB | 42.375 s |

Memory reduction: **17.4x** less peak RAM with generators.

---

## Lab 02 — The Inverted Index: Dictionaries, Hashing, and Memory

- **`Posting` & `DocMeta`:** `@dataclass(frozen=True, slots=True)` for hashability and lower memory footprint.
- **Persistence (`src/findex/store.py`):** supports `pickle` and `json`. **Security note:** never call `pickle.load()` on untrusted files, as crafted pickle payloads can execute arbitrary code.

| Postings representation | Peak memory (build) | Index file size | Load time |
|---|---:|---:|---:|
| `list[Posting]` with plain `@dataclass` / `__dict__` | 309.38 MB (22.18s) | — | — |
| `list[Posting]` with `slots=True` | 234.30 MB (45.10s) | 51.06 MB | 58.542 s |
| `array('I')` pairs (doc_ids, tfs) — no objects | 296.47 MB (34.57s) | 35.10 MB | 5.434 s |

---

## Lab 03 — Ranking and the Object Model: Dunder Methods, Protocols, Decorators

- **`Index` (`src/findex/models.py`):** `collections.abc.Mapping` with `__len__`, `__contains__`, `__getitem__`, `__iter__`, `__repr__`, `__eq__`, `num_docs`, `avg_doc_length` (`@cached_property`), `df`, `doc_length`, and `open_index()` context manager.
- **Ranking (`src/findex/ranking.py`):** `Scorer` `typing.Protocol` with `TfIdf` and `BM25(k1=1.5, b=0.75)` and `heapq.nlargest`.
- **Query language (`src/findex/query.py`):** recursive-descent parser (`Term`, `Phrase`, `And`, `Or`, `Not` with `&`, `|`, `~`).
- **Decorators, caching & snippets:** `@timed` (with `functools.wraps`), `lru_cache(maxsize=256)`, ±80-char highlighted snippets.

### Ranking sanity checks (real corpus)

1. **Rare term outranks common term:** rare `claus` (`df=12`, doc `0`) scores **BM25=10.1321** / TF-IDF=14.0178 vs common `subject` (`df=18846`, doc `1`) **BM25=0.0000** / TF-IDF=1.0000.
2. **BM25 TF saturation (`k1=1.5`):** for `claus` in doc `0` (`len=339`), `tf=1` → **7.0059**, `tf=19` → **16.8655**, `tf=20` → **16.9317** (20th repetition adds only **+0.0662**, i.e. **+0.39%**).
3. **Short doc outranks very long doc (`b=0.75`):** for `ann` with `tf=1`, short doc `16298` (`len=39`) scores **BM25=8.4382** vs long doc `15531` (`len=5130`) **BM25=0.6374** (avg doc length = 308.4).

### `@timed` and `lru_cache` hit log

```text
INFO [findex.search]: query='(space OR nasa OR shuttle OR orbit) NOT mars' cache=MISS (hits=0, misses=1) matched=1482
INFO [findex.timing]: search took 4.32 ms
INFO [findex.search]: query='(space OR nasa OR shuttle OR orbit) NOT mars' cache=HIT (hits=1, misses=1) matched=1482
INFO [findex.timing]: search took 2.69 ms
```

### Precision@5: TF-IDF vs. BM25 (10 hand-labeled queries)

| # | Query | Target group(s) | TF-IDF P@5 | BM25 P@5 |
|---:|---|---|---:|---:|
| 1 | `nasa OR shuttle OR orbit` | `sci.space` | 1.00 | 1.00 |
| 2 | `clipper OR encryption OR escrow` | `sci.crypt` | 1.00 | 1.00 |
| 3 | `nhl OR hockey OR playoffs` | `rec.sport.hockey` | 1.00 | 1.00 |
| 4 | `motorcycle OR helmet OR kawasaki` | `rec.motorcycles` | 1.00 | 1.00 |
| 5 | `jpeg OR gif OR polygon` | `comp.graphics` | 1.00 | 0.80 |
| 6 | `engine OR transmission OR honda` | `rec.autos` | 0.60 | 0.40 |
| 7 | `cancer OR medical OR patient` | `sci.med` | 1.00 | 1.00 |
| 8 | `motif OR widget OR x11` | `comp.windows.x` | 1.00 | 1.00 |
| 9 | `baseball OR pitcher OR inning` | `rec.sport.baseball` | 1.00 | 1.00 |
| 10 | `circuit OR voltage OR amplifier` | `sci.electronics` | 1.00 | 1.00 |
| **Avg** | **10 queries** | — | **0.96** | **0.92** |
