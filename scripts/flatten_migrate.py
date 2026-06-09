#!/usr/bin/env python3
"""Flatten per-topic paper directories into a single ``papers/`` folder.

Migrates every ``<Topic>/papers/<NN>_<surname>_<year>/`` directory into a flat
``papers/<citekey>/`` layout, renames the inner files to a citekey stem, and
injects YAML frontmatter (topic/order/status/year/...) into the notes file so
the frontmatter becomes the single source of truth for the generated index.

The citekey follows the BibTeX convention ``<surname><year><keyword>`` (first
significant title word), lowercased, with letter suffixes (a, b, c) on clash.

Dry-run by default; pass ``--execute`` to apply moves. Filesystem moves are used
(not ``git mv``) so that gitignored PDFs move uniformly with tracked files; git
detects the renames on the next ``add``.

Usage:
    python scripts/flatten_migrate.py            # dry-run, prints mapping
    python scripts/flatten_migrate.py --execute  # apply
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from reading_list import parse_reading_list
from paper_dir import _normalize_surname  # noqa: F401 (reused indirectly)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Topic dir -> short tag used in frontmatter and as an Obsidian tag.
TOPIC_TAGS = {
    "Artificial_Intelligence": "artificial-intelligence",
    "Solar_Physics": "solar-physics",
    "Space_Weather": "space-weather",
    "Solar_Observation": "solar-observation",
    "Living_Reviews_in_Solar_Physics": "living-reviews-solar-physics",
    "Low_SNR_Imaging": "low-snr-imaging",
    "Helioseismology_Asteroseismology": "helioseismology-asteroseismology",
    "Heliosphere_Solar_Wind": "heliosphere-solar-wind",
    "Magnetic_Reconnection_Eruption": "magnetic-reconnection-eruption",
    "Plasma_Spectroscopy_Diagnostics": "plasma-spectroscopy-diagnostics",
    "Numerical_MHD_Simulation": "numerical-mhd-simulation",
}

# Paper dir name pattern: NN_surname_year (surname may contain underscores).
_DIR_RE = re.compile(r"^(\d+)_(.+)_(\d{4})$")

# Words skipped when picking the title keyword.
_STOPWORDS = {
    "a", "an", "the", "on", "of", "in", "for", "and", "to", "with", "by",
    "from", "as", "at", "is", "are", "be", "this", "that", "via", "using",
    "their", "its", "new", "der", "die", "das", "und", "uber", "von",
}
_ROMAN_RE = re.compile(r"^[ivxlcdm]+$")


def pick_keyword(title: str) -> str:
    """Pick the first significant word of a title for the citekey.

    Args:
        title: Paper title (English).

    Returns:
        A lowercase alphanumeric keyword, or "" if none qualifies.
    """
    # Strip a leading "Letters on ..." style; just tokenize words.
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", title)
    for w in words:
        token = re.sub(r"[^a-z0-9]", "", w.lower())
        if len(token) < 3:
            continue
        if token in _STOPWORDS or _ROMAN_RE.match(token):
            continue
        return token
    return ""


def make_citekey(surname: str, year: str, keyword: str) -> str:
    """Build a base citekey (no clash suffix).

    Args:
        surname: Surname segment from the dir name (may contain underscores).
        year: 4-digit year string.
        keyword: Title keyword (possibly empty).

    Returns:
        Lowercase citekey like "vaswani2017attention".
    """
    surname_token = re.sub(r"[^a-z0-9]", "", surname.lower())
    return f"{surname_token}{year}{keyword}"


def build_topic_index(topic_dir: Path) -> dict:
    """Map paper number -> reading-list entry for a topic.

    Args:
        topic_dir: Path to a topic directory.

    Returns:
        Dict keyed by paper number; empty if no reading list.
    """
    rl = topic_dir / "papers" / "reading_list.md"
    if not rl.exists():
        return {}
    return {e["number"]: e for e in parse_reading_list(rl)}


def plan_migration() -> list[dict]:
    """Compute the full migration plan across all topics.

    Returns:
        List of plan records (sorted), each describing one paper dir move.
    """
    plans: list[dict] = []
    used: dict[str, int] = {}  # base citekey -> count, for clash suffixes

    for topic_name in sorted(TOPIC_TAGS):
        topic_dir = PROJECT_ROOT / topic_name
        papers_dir = topic_dir / "papers"
        if not papers_dir.is_dir():
            continue
        index = build_topic_index(topic_dir)

        for d in sorted(papers_dir.iterdir()):
            if not d.is_dir():
                continue
            m = _DIR_RE.match(d.name)
            if not m:
                continue
            number = int(m.group(1))
            surname = m.group(2)
            year = m.group(3)

            entry = index.get(number, {})
            title = entry.get("title", "")
            keyword = pick_keyword(title)
            base = make_citekey(surname, year, keyword)

            # Clash suffix (a, b, c, ...)
            n = used.get(base, 0)
            used[base] = n + 1
            citekey = base if n == 0 else f"{base}{chr(ord('a') + n - 1)}"

            plans.append({
                "topic": topic_name,
                "topic_tag": TOPIC_TAGS[topic_name],
                "number": number,
                "old_dir": str(d.relative_to(PROJECT_ROOT)),
                "old_name": d.name,
                "citekey": citekey,
                "new_dir": f"papers/{citekey}",
                "title": title,
                "authors": entry.get("authors", ""),
                "year": int(year),
                "status": entry.get("status", " "),
                "doi": entry.get("doi", ""),
                "matched_reading_list": bool(entry),
            })

    return plans


# Map a status char to a frontmatter status word.
_STATUS_WORD = {"x": "done", "~": "reading", " ": "todo"}


def build_frontmatter(p: dict) -> str:
    """Build a YAML frontmatter block for a paper's notes file.

    Args:
        p: A plan record.

    Returns:
        YAML frontmatter string ending in a trailing newline.
    """
    title = p["title"].replace('"', "'")
    tags = [p["topic_tag"]]
    if p["status"] == "x":
        tags.append("done")
    doi = p.get("doi", "")
    lines = [
        "---",
        f'title: "{title}"',
        f'authors: "{p["authors"]}"',
        f"year: {p['year']}",
        f"topic: {p['topic_tag']}",
        f"order: {p['number']}",
        f"status: {_STATUS_WORD.get(p['status'], 'todo')}",
        f"citekey: {p['citekey']}",
    ]
    if doi and doi.upper() != "NO_DOI":
        lines.append(f"doi: {doi}")
    lines.append(f"tags: [{', '.join(tags)}]")
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + "\n"


def inject_frontmatter(notes_path: Path, p: dict, execute: bool) -> str:
    """Prepend frontmatter to a notes file if absent.

    Args:
        notes_path: Path to the (already moved) notes markdown file.
        p: Plan record.
        execute: Whether to actually write.

    Returns:
        One of "added", "exists", "missing".
    """
    if not notes_path.exists():
        return "missing"
    text = notes_path.read_text(encoding="utf-8")
    if text.lstrip().startswith("---"):
        return "exists"
    if execute:
        notes_path.write_text(build_frontmatter(p) + text, encoding="utf-8")
    return "added"


def apply_one(p: dict) -> None:
    """Move one paper dir and rename its inner files (filesystem move).

    Args:
        p: Plan record.
    """
    old = PROJECT_ROOT / p["old_dir"]
    new = PROJECT_ROOT / p["new_dir"]
    new.parent.mkdir(parents=True, exist_ok=True)
    if new.exists():
        raise FileExistsError(f"target exists: {new}")
    shutil.move(str(old), str(new))

    old_stem = p["old_name"]
    for f in list(new.iterdir()):
        if f.is_file() and f.name.startswith(old_stem):
            suffix = f.name[len(old_stem):]  # e.g. "_notes.md"
            f.rename(new / f"{p['citekey']}{suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Flatten papers into one folder.")
    ap.add_argument("--execute", action="store_true", help="apply (default: dry-run)")
    ap.add_argument("--json", action="store_true", help="emit full plan as JSON")
    args = ap.parse_args()

    plans = plan_migration()

    if args.json:
        print(json.dumps(plans, ensure_ascii=False, indent=2))
        return

    # Clash & sanity report
    citekeys = [p["citekey"] for p in plans]
    dup = {k for k in citekeys if citekeys.count(k) > 1}
    unmatched = [p for p in plans if not p["matched_reading_list"]]
    no_keyword = [p for p in plans if p["citekey"][-1].isdigit()]

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"=== Flatten migration [{mode}] : {len(plans)} paper dirs ===\n")
    by_topic: dict[str, int] = {}
    for p in plans:
        by_topic[p["topic"]] = by_topic.get(p["topic"], 0) + 1
    for t, c in sorted(by_topic.items()):
        print(f"  {t:<38} {c:>3}")
    print()
    print(f"  citekey collisions (suffixed): {len(dup)}")
    print(f"  dirs without reading-list match: {len(unmatched)}")
    print(f"  dirs with no title keyword (citekey ends in year): {len(no_keyword)}")
    print()

    # Sample mapping
    print("--- sample mapping (first 25) ---")
    for p in plans[:25]:
        print(f"  {p['old_dir']:<55} ->  papers/{p['citekey']}")
    print("  ...")

    if unmatched:
        print("\n--- dirs WITHOUT reading-list match (need attention) ---")
        for p in unmatched:
            print(f"  {p['old_dir']}  ->  papers/{p['citekey']}")

    # Write full mapping for review
    map_path = PROJECT_ROOT / "scripts" / "flatten_mapping.tsv"
    with map_path.open("w", encoding="utf-8") as fh:
        fh.write("old_dir\tcitekey\ttopic\tnumber\tstatus\ttitle\n")
        for p in plans:
            fh.write(f"{p['old_dir']}\t{p['citekey']}\t{p['topic']}\t"
                     f"{p['number']}\t{p['status']}\t{p['title']}\n")
    print(f"\nFull mapping written to: {map_path.relative_to(PROJECT_ROOT)}")

    if not args.execute:
        print("\n(DRY-RUN — no files moved. Re-run with --execute to apply.)")
        return

    # Execute
    fm = {"added": 0, "exists": 0, "missing": 0}
    for p in plans:
        apply_one(p)
        notes = PROJECT_ROOT / p["new_dir"] / f"{p['citekey']}_notes.md"
        fm[inject_frontmatter(notes, p, execute=True)] += 1
    print(f"\nMoved {len(plans)} dirs. Frontmatter: "
          f"{fm['added']} added, {fm['exists']} already had, {fm['missing']} no-notes.")


if __name__ == "__main__":
    main()
