# findex — Streaming Full-Text Search Engine

A full-text search engine built incrementally in Python.

## Lab 01 — Streams, Not Lists: Iterators, Generators, and Corpus Intake

### Corpus

- **Source:** [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html) (18,846 Usenet posts, 36.28 MB uncompressed, stored locally in `data/corpus.jsonl`).
- **Storage:** `data/` (excluded from Git via `.gitignore`).
- **How to download:**
  ```bash
  uv run python scripts/download_corpus.py
  ```

### Tokenizer policy (`src/findex/tokenize.py`)

- **Unicode normalization:** `unicodedata.normalize("NFC", text)` ensures canonically equivalent sequences (e.g. `"café"` and `"cafe\u0301"`) map to the same token.
- **Case folding:** `str.casefold()` is used instead of `str.lower()` for locale-independent matching (e.g. `"Straße"` → `"strasse"`).
- **Apostrophes:** internal apostrophes are kept (`don't`, `п'ять`), and typographic apostrophes (`’`) are normalized to ASCII `'`. Leading and trailing quotes are stripped.
- **Hyphens:** internal hyphens are preserved (`state-of-the-art`, `f-16`), while standalone, leading, or trailing dashes are stripped.
- **Digits:** numbers and alphanumeric terms (`2026`, `v2`) are preserved.
- **Regex:** `re.finditer(r"[^\W_]+(?:['\u2019-][^\W_]+)*", normalized)` streams matches lazily without creating intermediate lists.

### Lab 01 Benchmark: Eager vs. Lazy

| Version | Documents | Total tokens | Peak memory | Elapsed |
|---|---:|---:|---:|---:|
| eager (lists) | 18,846 | 5,812,698 | 333.12 MB | 28.557 s |
| lazy (generators) | 18,846 | 5,812,698 | 19.15 MB | 42.375 s |

Memory reduction: **17.4x** less peak RAM with generators.

In the eager version, `list(iter_documents(root))` loads every raw document string into RAM at once, and `[list(tokenize(doc.text)) for doc in documents]` allocates millions of individual Python `str` token objects and list pointer arrays simultaneously. In the lazy generator pipeline, only a single document and the current token exist upstream at any moment; memory is bounded by one document plus the `Counter` vocabulary table.

---

## Lab 02 — The Inverted Index: Dictionaries, Hashing, and Memory

### What was added

- **`src/findex/index.py` & `src/findex/models.py`:** `Posting` and `DocMeta` (`@dataclass(frozen=True, slots=True)`), single-pass builder `build_index()` using `defaultdict` and per-document `Counter` (with `--positions`), compact `array('I')` builder `build_compact_index()`, and `python -m findex.index`.
- **`src/findex/search.py`:** Boolean query evaluation (`AND`, `OR`, `NOT`, implicit `AND`) with two-pointer merge (`--engine merge`) and Python `set` operations (`--engine set`).
- **`src/findex/store.py`:** `save_index()` and `load_index()` supporting `pickle` and `json`. **Pickle security warning:** never call `pickle.load()` on untrusted or external files, because unpickling crafted data can execute arbitrary code; use `pickle` only for locally generated index files.

### Memory study & persistence benchmark

| Postings representation | Peak memory (build) | Index file size | Load time |
|---|---:|---:|---:|
| `list[Posting]` with plain `@dataclass` / `__dict__` | 309.38 MB (22.18s) | — | — |
| `list[Posting]` with `slots=True` | 234.30 MB (45.10s) | 51.06 MB | 58.542 s |
| `array('I')` pairs (doc_ids, tfs) — no objects | 296.47 MB (34.57s) | 35.10 MB | 5.434 s |

**Where the bytes went:**
Without `__slots__`, each instance carries a per-instance `__dict__` hash table (~100+ bytes of overhead on top of attribute references), and each `list[Posting]` element is a separate heap object holding 28-byte Python `int` pointers. Adding `frozen=True, slots=True` eliminates the per-instance `__dict__` and cuts build peak memory from 309.38 MB to 234.30 MB while keeping `Posting` and `DocMeta` hashable; `array('I')` pairs store raw 4-byte C unsigned integers contiguously, shrinking the serialized file from 51.06 MB to 35.10 MB and cutting load time from 58.542 s to 5.434 s.

---

## Lab 03 — Ranking and the Object Model: Dunder Methods, Protocols, Decorators

### What was added

- **Pythonic `Index` (`src/findex/models.py`):** inherits from `collections.abc.Mapping[str, list[Posting]]`, implements `__len__`, `__contains__`, `__getitem__`, `__iter__`, `__repr__`, `num_docs`, `avg_doc_length` (`@cached_property`), `doc_length(doc_id)`, `df(term)`, and `close()`.
- **Context manager (`src/findex/store.py`):** `with open_index(path) as ix:` guarantees resource cleanup even when an exception occurs.
- **Swappable ranking (`src/findex/ranking.py`):** `Scorer` `typing.Protocol` with `TfIdf` and `BM25(k1=1.5, b=0.75)` implementations (`score()` and `__call__()`), top-k selection via `heapq.nlargest`, and `SearchResult` (`order=True`, `__str__`).
- **Query language (`src/findex/query.py`):** recursive-descent parser producing `Term`, `Phrase`, `And`, `Or`, `Not` AST nodes with `&`, `|`, `~` operator overloading and positional two-pointer phrase matching.
- **Decorators, caching & snippets (`src/findex/decorators.py`, `src/findex/snippets.py`):** `@timed` (with `functools.wraps`) on `search`, `build_index`, and `load_index`; `@functools.lru_cache(maxsize=256)` on query evaluation; ±80-character snippets with `**highlighted**` terms.

### Usage

```bash
# Build index WITH positions (required for phrase queries)
uv run python -m findex.index data/ --out index.bin --positions

# Ranked BM25 search with parentheses, NOT, phrase, and snippets
uv run python -m findex.search index.bin 'nasa AND (shuttle OR orbit) NOT mars "space"'

# Switch scorer to TF-IDF
uv run python -m findex.search index.bin 'clipper chip encryption' --scorer tfidf -k 5

# Run Lab 3 sanity checks, cache hit demo, and Precision@5 evaluation
uv run python -m findex.eval index.bin --update-readme
```

<!-- LAB03-RESULTS-START -->
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
<!-- LAB03-RESULTS-END -->
