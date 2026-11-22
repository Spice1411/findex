"""Download a real text corpus into data/corpus.jsonl using only the Python stdlib."""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

NEWSGROUPS_URLS = [
    "https://ndownloader.figshare.com/files/5975967",
    "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz",
]

PYTHON_DOCS_URLS = [
    "https://docs.python.org/3/archives/python-3.13-docs-text.zip",
    "https://docs.python.org/3/archives/python-3.12-docs-text.zip",
]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def download_file(url: str, dest: Path) -> bool:
    """Download ``url`` to ``dest``, falling back to system ``curl`` if needed."""
    print(f"Trying: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp, dest.open("wb") as out:
            shutil.copyfileobj(resp, out)
        if dest.exists() and dest.stat().st_size > 100_000:
            return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  urllib failed ({exc}); trying system curl...")

    curl_bin = shutil.which("curl") or shutil.which("curl.exe")
    if not curl_bin:
        return False

    try:
        res = subprocess.run(
            [curl_bin, "-fL", "--retry", "2", "-A", USER_AGENT, "-o", str(dest), url],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if res.returncode == 0 and dest.exists() and dest.stat().st_size > 100_000:
            return True
        if res.stderr:
            print(f"  curl failed: {res.stderr.strip()}")
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  curl error: {exc}")
    return False


def convert_newsgroups_tar(archive: Path, out_jsonl: Path) -> int:
    """Stream all posts from 20news-bydate.tar.gz into a single JSON-lines file."""
    count = 0
    with tarfile.open(archive, "r:gz") as tf, out_jsonl.open("w", encoding="utf-8") as out:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            handle = tf.extractfile(member)
            if handle is None:
                continue
            raw = handle.read()
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            record = {"id": member.name, "text": text}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert_python_docs_zip(archive: Path, out_jsonl: Path) -> int:
    """Stream all .txt files from the Python docs zip archive into a JSON-lines file."""
    count = 0
    with zipfile.ZipFile(archive, "r") as zf, out_jsonl.open("w", encoding="utf-8") as out:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            text = zf.read(name).decode("utf-8", errors="replace").strip()
            if not text:
                continue
            record = {"id": name, "text": text}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    out_jsonl = data_dir / "corpus.jsonl"

    if out_jsonl.exists() and out_jsonl.stat().st_size > 1_000_000:
        size_mb = out_jsonl.stat().st_size / (1024 * 1024)
        print(f"Corpus already exists: {out_jsonl} ({size_mb:.2f} MB)")
        return 0

    tmp_tar = data_dir / "_tmp_newsgroups.tar.gz"
    for url in NEWSGROUPS_URLS:
        if download_file(url, tmp_tar):
            print("Extracting 20 Newsgroups into data/corpus.jsonl...")
            docs = convert_newsgroups_tar(tmp_tar, out_jsonl)
            tmp_tar.unlink(missing_ok=True)
            size_mb = out_jsonl.stat().st_size / (1024 * 1024)
            print(f"Done! Corpus: 20 Newsgroups | Docs: {docs:,} | Size: {size_mb:.2f} MB")
            return 0
    tmp_tar.unlink(missing_ok=True)

    tmp_zip = data_dir / "_tmp_pydocs.zip"
    for url in PYTHON_DOCS_URLS:
        if download_file(url, tmp_zip):
            print("Extracting Python docs into data/corpus.jsonl...")
            docs = convert_python_docs_zip(tmp_zip, out_jsonl)
            tmp_zip.unlink(missing_ok=True)
            size_mb = out_jsonl.stat().st_size / (1024 * 1024)
            print(f"Done! Corpus: Python Docs | Docs: {docs:,} | Size: {size_mb:.2f} MB")
            return 0
    tmp_zip.unlink(missing_ok=True)

    print("ERROR: Could not download any corpus archive. Check your internet connection.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
