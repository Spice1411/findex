# findex — Streaming Full-Text Search Engine

A full-text search engine built incrementally in Python.

## Lab 01 — Streams, Not Lists: Iterators, Generators, and Corpus Intake

### Corpus

- **Source:** [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html) (~18,846 Usenet posts, ~35 MB of plain text, stored as a single streaming JSON-lines file `data/corpus.jsonl`; falls back to official Python 3.13 text documentation if unreachable).
- **Storage:** `data/` (excluded from Git via `.gitignore`).
- **How to download:**
  ```bash
  uv run python scripts/download_corpus.py
  ```

### Project structure

```txt
findex/
  pyproject.toml
  README.md
  .gitignore
  scripts/
    download_corpus.py       # fetches 20 Newsgroups corpus into data/corpus.jsonl
  src/findex/
    __init__.py
    corpus.py                # Document + lazy iter_documents() generator (.txt & .jsonl)
    tokenize.py              # streaming Unicode-aware tokenize() generator
    stats.py                 # single-pass stats pipeline & eager/lazy benchmark
  tests/
    test_tokenize.py         # tokenizer and generator tests
```

### Tokenizer policy (`src/findex/tokenize.py`)

- **Unicode normalization:** `unicodedata.normalize("NFC", text)` ensures canonically equivalent sequences (e.g. `"café"` and `"cafe\u0301"`) map to the same token.
- **Case folding:** `str.casefold()` is used instead of `str.lower()` for locale-independent matching (e.g. `"Straße"` → `"strasse"`).
- **Apostrophes:** internal apostrophes are kept (`don't`, `п'ять`), and typographic apostrophes (`’`) are normalized to ASCII `'`. Leading and trailing quotes are stripped.
- **Hyphens:** internal hyphens are preserved (`state-of-the-art`, `f-16`), while standalone, leading, or trailing dashes are stripped.
- **Digits:** numbers and alphanumeric terms (`2026`, `v2`) are preserved.
- **Regex:** `re.finditer(r"[^\W_]+(?:['\u2019-][^\W_]+)*", normalized)` streams matches lazily without creating intermediate lists.

### Usage

```bash
# Install dev dependencies (pytest, ruff)
uv sync

# Download the corpus into data/corpus.jsonl
uv run python scripts/download_corpus.py

# Run tests and linter
uv run pytest
uv run ruff check .

# Run single-pass lazy stats on the corpus (or first N documents)
uv run python -m findex.stats data/
uv run python -m findex.stats data/ --limit 100

# Benchmark eager (lists) vs lazy (generators)
uv run python -m findex.stats data/ --compare
```

### Benchmark: Eager vs. Lazy

| Version | Documents | Total tokens | Peak memory | Elapsed |
|---|---:|---:|---:|---:|
| eager (lists) | 18,846 | 5,812,698 | 333.12 MB | 28.557 s |
| lazy (generators) | 18,846 | 5,812,698 | 19.15 MB | 42.375 s |

Memory reduction: **17.4x** less peak RAM with generators.

**Why the numbers differ:**
In the eager version, `list(iter_documents(root))` first loads the raw text of every document into RAM at once, and `[list(tokenize(doc.text)) for doc in documents]` then allocates millions of individual Python `str` token objects and list pointer arrays simultaneously — causing peak memory to grow to several times the raw corpus size on disk. In the lazy generator pipeline, only a single document string and the current token exist upstream at any moment; memory is bounded by one document plus the `Counter` vocabulary table (the sink holding unique terms and their counts), which grows only with the number of distinct words rather than total corpus length.
