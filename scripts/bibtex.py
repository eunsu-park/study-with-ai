#!/usr/bin/env python3
"""Generate BibTeX bibliography files from reading list DOIs.

Usage:
    python scripts/bibtex.py generate <topic>      # Generate .bib for one topic
    python scripts/bibtex.py generate --all         # Generate .bib for all topics
    python scripts/bibtex.py lookup <doi>           # Lookup single DOI
    python scripts/bibtex.py verify <topic>         # Check DOI coverage
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from paper_dir import _normalize_surname
from reading_list import parse_reading_list, resolve_topic, TOPIC_ALIASES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = Path(__file__).resolve().parent / ".bibtex_cache.json"
USER_EMAIL = "phd.choux@gmail.com"


def _load_cache() -> dict[str, str]:
    """Load cached BibTeX entries."""
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    """Save BibTeX cache."""
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _make_cite_key(entry: dict) -> str:
    """Generate citation key from entry metadata."""
    authors = entry.get("authors", "")
    year = entry.get("year", 0)
    raw = re.split(r",|;|\band\b", authors)
    raw = [a.strip() for a in raw if a.strip()]
    raw = [a for a in raw if not re.match(r"^et\s+al\.?$", a, re.IGNORECASE)]
    if raw:
        surname = _normalize_surname(raw[0])
    else:
        surname = "unknown"
    return f"{surname}{year}"


def fetch_bibtex_crossref(doi: str) -> str | None:
    """Fetch BibTeX from CrossRef via content negotiation."""
    url = f"https://doi.org/{doi}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/x-bibtex",
        "User-Agent": f"StudyWithAI/1.0 (mailto:{USER_EMAIL})",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def fetch_bibtex_arxiv(arxiv_id: str) -> str | None:
    """Fetch BibTeX from arXiv."""
    url = f"https://arxiv.org/bibtex/{arxiv_id}"
    req = urllib.request.Request(url, headers={
        "User-Agent": f"StudyWithAI/1.0 (mailto:{USER_EMAIL})",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def generate_manual_entry(entry: dict, cite_key: str) -> str:
    """Generate a minimal BibTeX entry from reading list metadata."""
    title = entry.get("title", "Unknown Title")
    authors = entry.get("authors", "Unknown")
    year = entry.get("year", 0)
    journal = entry.get("journal", "")
    doi = entry.get("doi", "")

    # Determine entry type
    if any(kw in title.lower() for kw in ("monograph", "book", "(book)")):
        entry_type = "book"
    else:
        entry_type = "misc"

    lines = [f"@{entry_type}{{{cite_key},"]
    lines.append(f"  title = {{{title}}},")
    lines.append(f"  author = {{{authors}}},")
    lines.append(f"  year = {{{year}}},")
    if journal:
        clean_journal = re.sub(r"[*_]", "", journal)
        lines.append(f"  journal = {{{clean_journal}}},")
    if doi and doi != "NO_DOI":
        lines.append(f"  doi = {{{doi}}},")
    lines.append("}")
    return "\n".join(lines)


def _replace_cite_key(bibtex: str, new_key: str) -> str:
    """Replace the citation key in a BibTeX entry."""
    return re.sub(r"(@\w+\{)[^,]+,", rf"\g<1>{new_key},", bibtex, count=1)


def fetch_bibtex(entry: dict) -> str:
    """Fetch or generate BibTeX for a paper entry."""
    cache = _load_cache()
    doi = entry.get("doi", "")
    cite_key = _make_cite_key(entry)

    # Check cache
    if doi and doi in cache:
        return _replace_cite_key(cache[doi], cite_key)

    bibtex = None

    if doi and doi.startswith("arXiv:"):
        arxiv_id = doi.replace("arXiv:", "")
        bibtex = fetch_bibtex_arxiv(arxiv_id)
        time.sleep(0.5)
    elif doi and doi != "NO_DOI":
        bibtex = fetch_bibtex_crossref(doi)
        time.sleep(1)  # CrossRef polite pool

    if bibtex:
        bibtex = _replace_cite_key(bibtex.strip(), cite_key)
        cache[doi] = bibtex
        _save_cache(cache)
        return bibtex

    # Fallback: manual entry
    return generate_manual_entry(entry, cite_key)


def cmd_generate(topic: str | None) -> None:
    """Generate bibliography.bib for a topic or all topics."""
    if topic == "--all":
        topics_to_process = list(TOPIC_ALIASES.values())
    else:
        topic_dir = resolve_topic(topic)
        topics_to_process = [topic_dir.name]

    for topic_name in topics_to_process:
        topic_dir = PROJECT_ROOT / topic_name
        rl_path = topic_dir / "papers" / "reading_list.md"
        if not rl_path.exists():
            print(f"SKIP {topic_name}: no reading_list.md", file=sys.stderr)
            continue

        entries = parse_reading_list(rl_path)
        bib_entries = []
        used_keys: dict[str, int] = {}

        print(f"Generating {topic_name}/papers/bibliography.bib ({len(entries)} papers)...")

        for entry in entries:
            cite_key = _make_cite_key(entry)
            # Handle duplicate keys within topic
            if cite_key in used_keys:
                used_keys[cite_key] += 1
                cite_key = f"{cite_key}{chr(96 + used_keys[cite_key])}"
            else:
                used_keys[cite_key] = 1

            bibtex = fetch_bibtex(entry)
            bibtex = _replace_cite_key(bibtex, cite_key)
            bib_entries.append(f"% Paper #{entry['number']}: {entry['title'][:60]}")
            bib_entries.append(bibtex)
            bib_entries.append("")

            doi = entry.get("doi", "NO_DOI")
            status = "cached" if doi in _load_cache() else "fetched"
            print(f"  #{entry['number']:02d} {cite_key} [{status}]")

        bib_path = topic_dir / "papers" / "bibliography.bib"
        bib_path.write_text("\n".join(bib_entries), encoding="utf-8")
        print(f"  -> {bib_path.relative_to(PROJECT_ROOT)} ({len(entries)} entries)\n")


def cmd_lookup(doi: str) -> None:
    """Lookup BibTeX for a single DOI."""
    if doi.startswith("arXiv:"):
        bibtex = fetch_bibtex_arxiv(doi.replace("arXiv:", ""))
    else:
        bibtex = fetch_bibtex_crossref(doi)

    if bibtex:
        print(bibtex)
    else:
        print(json.dumps({"error": f"Could not fetch BibTeX for {doi}"}))


def cmd_verify(topic: str) -> None:
    """Check DOI coverage for a topic."""
    topic_dir = resolve_topic(topic)
    entries = parse_reading_list(topic_dir / "papers" / "reading_list.md")

    has_doi = [e for e in entries if e.get("doi") and e["doi"] != "NO_DOI"]
    no_doi = [e for e in entries if not e.get("doi") or e["doi"] == "NO_DOI"]

    print(json.dumps({
        "topic": topic_dir.name,
        "total": len(entries),
        "with_doi": len(has_doi),
        "without_doi": len(no_doi),
        "missing": [{"number": e["number"], "title": e["title"][:50]} for e in no_doi],
    }, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="BibTeX bibliography management.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Generate bibliography.bib")
    p_gen.add_argument("topic", nargs="?", default=None,
                       help="Topic name/alias (omit for all topics)")
    p_gen.add_argument("--all", dest="all_topics", action="store_true",
                       help="Generate for all topics")

    p_lookup = sub.add_parser("lookup", help="Lookup BibTeX for a DOI")
    p_lookup.add_argument("doi")

    p_verify = sub.add_parser("verify", help="Check DOI coverage")
    p_verify.add_argument("topic")

    args = parser.parse_args()

    if args.command == "generate":
        if args.all_topics or args.topic is None:
            cmd_generate("--all")
        else:
            cmd_generate(args.topic)
    elif args.command == "lookup":
        cmd_lookup(args.doi)
    elif args.command == "verify":
        cmd_verify(args.topic)


if __name__ == "__main__":
    main()
