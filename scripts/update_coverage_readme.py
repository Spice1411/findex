"""Read coverage.json produced by pytest-cov and update Lab 04 coverage section in README.md."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    cov_path = Path("coverage.json")
    readme_path = Path("README.md")
    if not cov_path.exists():
        print("coverage.json not found. Run pytest --cov=findex --cov-report=json first.")
        return 1

    data = json.loads(cov_path.read_text(encoding="utf-8"))
    totals = data["totals"]
    pct = totals["percent_covered"]
    covered = totals["covered_lines"]
    total_lines = totals["num_statements"]

    rows = [
        f"- **Total coverage (`src/findex`):** **{pct:.1f}%** "
        f"({covered}/{total_lines} statements).",
        "",
        "| Module | Statements | Missed | Coverage |",
        "|---|---:|---:|---:|",
    ]
    for filepath, info in sorted(data["files"].items()):
        norm_path = filepath.replace("\\", "/")
        s = info["summary"]
        rows.append(
            f"| `{norm_path}` | {s['num_statements']} | {s['missing_lines']} "
            f"| {s['percent_covered']:.1f}% |"
        )

    rows.extend(
        [
            "",
            "**Intentionally left outside coverage:**",
            "- Legacy `if __name__ == '__main__':` entry points in `index.py`, `search.py`, "
            "`stats.py`, `eval.py`) — superseded by the packaged `findex` CLI (`cli.py`).",
            "- Lab 2 memory-study benchmark runner (`run_memory_study`) and network download "
            "fallbacks in `scripts/download_corpus.py` — slow / external-network paths.",
            "- Defensive `OSError` handlers for corrupt files or unreadable disks.",
        ]
    )

    block = "\n".join(rows)
    start_marker = "<!-- LAB04-COVERAGE-START -->"
    end_marker = "<!-- LAB04-COVERAGE-END -->"
    content = readme_path.read_text(encoding="utf-8")
    before = content.split(start_marker, 1)[0]
    after = content.split(end_marker, 1)[1]
    readme_path.write_text(
        f"{before}{start_marker}\n{block}\n{end_marker}{after}",
        encoding="utf-8",
    )
    print(f"Updated README.md with {pct:.1f}% coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
