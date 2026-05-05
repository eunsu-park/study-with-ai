#!/usr/bin/env python3
"""Paper directory naming, creation, and management.

Usage:
    python scripts/paper_dir.py name <number> <authors_string> <year>
    python scripts/paper_dir.py files <dir_name>
    python scripts/paper_dir.py create <topic> <number> <authors_string> <year>
    python scripts/paper_dir.py archive <paper_dir_path>
"""

import argparse
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_surname(name: str) -> str:
    """Extract and normalize a surname from a full name string.

    Handles prefixes like "Van", "von", "de", multi-part surnames,
    and accented characters.

    Args:
        name: A single author's full name (e.g., "James Van Allen").

    Returns:
        Lowercased, ASCII-only surname (e.g., "van_allen").
    """
    name = name.strip()
    # Remove "et al.", "Jr.", "Sr." suffixes
    name = re.sub(r"\s+(et\s+al\.?|Jr\.?|Sr\.?|III|II|IV)$", "", name, flags=re.IGNORECASE)
    parts = name.split()
    if not parts:
        return "unknown"

    # Known surname prefixes
    prefixes = {"van", "von", "de", "di", "le", "la", "el", "al"}

    # Take last word as surname, then prepend any prefix words before it
    surname_parts = [parts[-1].lower()]
    # Check preceding words for known prefixes
    for p in reversed(parts[:-1]):
        p_lower = p.lower().rstrip(".")
        if p_lower in prefixes:
            surname_parts.insert(0, p_lower)
        else:
            break

    surname = "_".join(surname_parts)

    # Remove accents
    surname = unicodedata.normalize("NFKD", surname)
    surname = "".join(c for c in surname if not unicodedata.combining(c))
    # Keep only alphanumeric and underscores
    surname = re.sub(r"[^a-z0-9_]", "", surname)
    return surname


def make_dir_name(number: int, authors_str: str, year: int) -> str:
    """Generate the standard paper directory name.

    Convention: first author surname only.

    Args:
        number: Paper number (zero-padded to 2 digits).
        authors_str: Full authors string from reading list.
        year: Publication year.

    Returns:
        Directory name like "06_rumelhart_1986".
    """
    # Split authors by comma, semicolon, or "and"
    raw_authors = re.split(r",|;|\band\b", authors_str)
    raw_authors = [a.strip() for a in raw_authors if a.strip()]
    # Remove "et al." entries
    raw_authors = [a for a in raw_authors if not re.match(r"^et\s+al\.?$", a, re.IGNORECASE)]

    if not raw_authors:
        return f"{number:02d}_unknown_{year}"

    surname = _normalize_surname(raw_authors[0])
    return f"{number:02d}_{surname}_{year}"


def make_file_names(dir_name: str) -> dict[str, str]:
    """Generate standard file names for a paper directory.

    Args:
        dir_name: Paper directory name (e.g., "14_mikolov_2013").

    Returns:
        Dict mapping file type to filename.
    """
    return {
        "pdf": f"{dir_name}_paper.pdf",
        "briefing": f"{dir_name}_briefing.md",
        "notes": f"{dir_name}_notes.md",
        "implementation": f"{dir_name}_implementation.ipynb",
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_name(number: int, authors: str, year: int) -> None:
    """Print the directory name for given parameters."""
    name = make_dir_name(number, authors, year)
    print(json.dumps({"dir_name": name}))


def cmd_files(dir_name: str) -> None:
    """Print standard file names for a directory."""
    files = make_file_names(dir_name)
    print(json.dumps(files))


def cmd_create(topic: str, number: int, authors: str, year: int) -> None:
    """Create a paper directory under a topic."""
    # Resolve topic
    from reading_list import resolve_topic
    topic_dir = resolve_topic(topic)

    dir_name = make_dir_name(number, authors, year)
    paper_dir = topic_dir / "papers" / dir_name
    paper_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "ok": True,
        "dir_name": dir_name,
        "path": str(paper_dir),
        "files": make_file_names(dir_name),
    }))


def cmd_archive(paper_dir_path: str) -> None:
    """Move notes.md and implementation.ipynb to archive/ subdirectory."""
    paper_dir = Path(paper_dir_path).resolve()
    if not paper_dir.is_dir():
        sys.exit(json.dumps({"error": f"Directory not found: {paper_dir}"}))

    archive_dir = paper_dir / "archive"
    moved = []

    for pattern in ("*_notes.md", "*_implementation.ipynb"):
        for f in paper_dir.glob(pattern):
            archive_dir.mkdir(exist_ok=True)
            dest = archive_dir / f.name
            shutil.move(str(f), str(dest))
            moved.append(f.name)

    print(json.dumps({"ok": True, "archived": moved}))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper directory naming and management."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_name = sub.add_parser("name", help="Generate directory name")
    p_name.add_argument("number", type=int)
    p_name.add_argument("authors", help="Authors string (quoted)")
    p_name.add_argument("year", type=int)

    p_files = sub.add_parser("files", help="List standard file names")
    p_files.add_argument("dir_name")

    p_create = sub.add_parser("create", help="Create paper directory")
    p_create.add_argument("topic")
    p_create.add_argument("number", type=int)
    p_create.add_argument("authors", help="Authors string (quoted)")
    p_create.add_argument("year", type=int)

    p_archive = sub.add_parser("archive", help="Archive notes and implementation")
    p_archive.add_argument("paper_dir_path")

    args = parser.parse_args()

    if args.command == "name":
        cmd_name(args.number, args.authors, args.year)
    elif args.command == "files":
        cmd_files(args.dir_name)
    elif args.command == "create":
        cmd_create(args.topic, args.number, args.authors, args.year)
    elif args.command == "archive":
        cmd_archive(args.paper_dir_path)


if __name__ == "__main__":
    main()
