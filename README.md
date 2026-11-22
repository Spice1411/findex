# findex — Streaming Full-Text Search Engine

A full-text search engine built incrementally in Python.

## Lab 01 — Streams, Not Lists: Iterators, Generators, and Corpus Intake

### Corpus

- **Source:** [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html) (~18,846 Usenet posts, ~36.28 MB uncompressed, stored locally in `data/corpus.jsonl`).
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

In the eager version, `list(iter_documents(root))` first loads the raw text of every document into RAM at once, and `[list(tokenize(doc.text)) for doc in documents]` then allocates millions of individual Python `str` token objects and list pointer arrays simultaneously — causing peak memory to grow to several times the raw corpus size on disk. In the lazy generator pipeline, only a single document string and the current token exist upstream at any moment; memory is bounded by one document plus the `Counter` vocabulary table (the sink holding unique terms and their counts), which grows only with the number of distinct words rather than total corpus length.

---

## Lab 02 — The Inverted Index: Dictionaries, Hashing, and Memory

### What was added

- **`src/findex/index.py`:** `Posting` and `DocMeta` (`@dataclass(frozen=True, slots=True)`), single-pass builder `build_index()` using `defaultdict` and per-document `Counter` (with optional `--positions`), compact `array('I')` builder `build_compact_index()`, and `python -m findex.index`.
- **`src/findex/search.py`:** Boolean query evaluation (`AND`, `OR`, `NOT`, implicit `AND`) with two-pointer merge (`--engine merge`, default) and Python `set` operations (`--engine set`), plus `python -m findex.search`.
- **`src/findex/store.py`:** `save_index()` and `load_index()` supporting `pickle` and `json`. **Pickle security warning:** never call `pickle.load()` on untrusted or external files, because unpickling crafted data can execute arbitrary code; use `pickle` only for locally generated index files.

### Usage

```bash
# Build and persist the inverted index
uv run python -m findex.index data/ --out index.bin

# Run Boolean search over the saved index
uv run python -m findex.search index.bin "python memory"
uv run python -m findex.search index.bin "linux OR windows NOT mac" --engine merge
uv run python -m findex.search index.bin "linux AND kernel" --engine set

# Run Lab 2 memory study across representations
uv run python -m findex.index data/ --memory-study
```

### Memory study & persistence benchmark

| Postings representation | Peak memory (build) | Index file size | Load time |
|---|---:|---:|---:|
| `list[Posting]` with plain `@dataclass` / `__dict__` | 309.38 MB (22.18s) | — | — |
| `list[Posting]` with `slots=True` | 234.30 MB (45.10s) | 51.06 MB | 58.542 s |
| `array('I')` pairs (doc_ids, tfs) — no objects | 296.47 MB (34.57s) | 35.10 MB | 5.434 s |

**Where the bytes went:**
Without `__slots__`, each instance carries a per-instance `__dict__` hash table (~100+ bytes of overhead on top of attribute references), and each `list[Posting]` element is a separate heap object holding 28-byte Python `int` pointers. Adding `frozen=True, slots=True` eliminates the per-instance `__dict__` and makes `Posting` and `DocMeta` hashable, while `array('I')` pairs store raw 4-byte C unsigned integers contiguously without creating millions of individual `Posting` and `int` wrapper objects.
