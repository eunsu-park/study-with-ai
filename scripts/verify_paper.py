#!/usr/bin/env python3
"""Run mechanical verification checks for a paper before downstream work proceeds.

This script implements the machine-checkable parts of the CLAUDE.md
"Pre-Work Verification Policy". The visual identity check (compare PDF first
page against reading list metadata) still has to be performed by Claude after
this script reports the basic checks pass.

Checks performed:
    1. reading_list.md has an entry for <topic, number> with a non-placeholder title
    2. The expected paper directory exists
    3. The PDF file exists at the canonical path
    4. PDF size is at least MIN_PDF_BYTES (default 100 KB)
    5. PDF starts with the `%PDF` magic bytes (not an HTML paywall page)
    6. The topic's bibliography.bib contains a key matching the dir_name (best-effort)

Output: JSON. Exit code 0 if all checks pass, 1 if any check fails.

Usage:
    python3 scripts/verify_paper.py <topic_alias> <number>
    python3 scripts/verify_paper.py AI 6
    python3 scripts/verify_paper.py Low_SNR_Imaging 11
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_PDF_BYTES = 100 * 1024  # 100 KB
PDF_MAGIC = b"%PDF"


def reading_list_info(topic: str, number: int) -> dict:
    """Call scripts/reading_list.py info to get the canonical entry."""
    result = subprocess.run(
        ["python3", str(ROOT / "scripts" / "reading_list.py"), "info", topic, str(number)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {"_error": result.stderr.strip() or "reading_list.py info failed"}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"_error": f"reading_list.py output not JSON: {exc}"}


def check_paper(topic: str, number: int) -> dict:
    """Run all mechanical checks for one paper. Returns a structured report."""
    report = {
        "topic": topic,
        "number": number,
        "checks": {},
        "all_passed": True,
        "fatal_errors": [],
        "warnings": [],
        "todo_for_claude": [
            "Read the PDF first page and confirm title, first-author surname, and year match the reading_list entry below.",
            "If a DOI is present, run `python3 scripts/bibtex.py lookup <doi>` and confirm CrossRef agrees with the reading_list entry.",
        ],
    }

    info = reading_list_info(topic, number)
    if "_error" in info:
        report["fatal_errors"].append(f"reading_list.py info failed: {info['_error']}")
        report["all_passed"] = False
        return report

    report["entry"] = {
        "title": info.get("title"),
        "authors": info.get("authors"),
        "year": info.get("year"),
        "doi": info.get("doi"),
        "dir_name": info.get("dir_name"),
        "topic_full": info.get("topic"),
        "status": info.get("status"),
    }

    title = info.get("title", "")
    if not title or "PLACEHOLDER" in title.upper():
        report["fatal_errors"].append("Reading list entry has placeholder/empty title")
        report["all_passed"] = False

    citekey = info.get("citekey")
    topic_full = info.get("topic")
    if not citekey or not topic_full:
        report["fatal_errors"].append(
            "citekey missing — paper has no flat papers/ folder (check flatten_mapping.tsv)")
        report["all_passed"] = False
        return report

    paper_dir = ROOT / "papers" / citekey
    pdf_path = paper_dir / f"{citekey}_paper.pdf"
    bib_path = ROOT / "bibliography.bib"

    # 2. Directory check
    report["checks"]["paper_dir_exists"] = paper_dir.is_dir()
    if not paper_dir.is_dir():
        report["fatal_errors"].append(f"Paper directory missing: {paper_dir.relative_to(ROOT)}")
        report["all_passed"] = False
        return report

    # 3. PDF exists
    pdf_exists = pdf_path.is_file()
    report["checks"]["pdf_exists"] = pdf_exists
    if not pdf_exists:
        report["fatal_errors"].append(f"PDF missing: {pdf_path.relative_to(ROOT)}")
        report["all_passed"] = False
        return report

    # 4. PDF size
    pdf_size = pdf_path.stat().st_size
    report["checks"]["pdf_size_bytes"] = pdf_size
    report["checks"]["pdf_size_ok"] = pdf_size >= MIN_PDF_BYTES
    if pdf_size < MIN_PDF_BYTES:
        report["fatal_errors"].append(
            f"PDF too small ({pdf_size} bytes < {MIN_PDF_BYTES}); likely a paywall page"
        )
        report["all_passed"] = False

    # 5. PDF magic bytes
    with pdf_path.open("rb") as fp:
        head = fp.read(4)
    pdf_magic_ok = head.startswith(PDF_MAGIC)
    report["checks"]["pdf_magic_ok"] = pdf_magic_ok
    if not pdf_magic_ok:
        report["fatal_errors"].append(
            f"PDF does not start with %PDF magic bytes (got {head!r}); likely an HTML page saved as .pdf"
        )
        report["all_passed"] = False

    # 6. Bibliography entry (best-effort match — BibTeX keys typically use
    # surnameYYYY (no separator) but we also try the underscore form).
    bib_has_entry = False
    if bib_path.is_file():
        bib_text = bib_path.read_text(encoding="utf-8", errors="replace").lower()
        bib_has_entry = citekey.lower() in bib_text
    report["checks"]["bibliography_has_entry"] = bib_has_entry
    if not bib_has_entry:
        report["warnings"].append(
            f"No entry '{citekey}' in {bib_path.name if bib_path.is_file() else 'bibliography.bib'}; "
            f"run `python3 scripts/bibtex.py generate` to regenerate."
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("topic", help="Topic alias (AI, SP, SW, SO, LRSP, ...) or full topic name")
    parser.add_argument("number", type=int, help="Paper number")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (exit 1 if any warning)",
    )
    args = parser.parse_args()

    report = check_paper(args.topic, args.number)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    failed = not report["all_passed"]
    if args.strict and report["warnings"]:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
