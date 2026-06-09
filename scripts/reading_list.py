#!/usr/bin/env python3
"""Parse and update reading_list.md files across study topics.

Usage:
    python scripts/reading_list.py topics
    python scripts/reading_list.py next <topic>
    python scripts/reading_list.py count <topic>
    python scripts/reading_list.py info <topic> <number>
    python scripts/reading_list.py highest <topic>
    python scripts/reading_list.py mark <topic> <number> <status>
    python scripts/reading_list.py add <topic> --title "..." --authors "..." --year YYYY ...
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from paper_dir import make_dir_name

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical topic registry: (display_name, tag, [aliases]). Reading lists live
# at reading_lists/<tag>.md (flat layout — per-topic folders were removed).
TOPICS: list[tuple[str, str, list[str]]] = [
    ("Artificial_Intelligence", "artificial-intelligence", ["ai"]),
    ("Solar_Physics", "solar-physics", ["sp"]),
    ("Space_Weather", "space-weather", ["sw"]),
    ("Solar_Observation", "solar-observation", ["so"]),
    ("Living_Reviews_in_Solar_Physics", "living-reviews-solar-physics", ["lrsp"]),
    ("Low_SNR_Imaging", "low-snr-imaging", ["lowsnr", "lsi"]),
    ("Helioseismology_Asteroseismology", "helioseismology-asteroseismology", ["helio", "ha"]),
    ("Heliosphere_Solar_Wind", "heliosphere-solar-wind", ["hsw"]),
    ("Magnetic_Reconnection_Eruption", "magnetic-reconnection-eruption", ["mre"]),
    ("Plasma_Spectroscopy_Diagnostics", "plasma-spectroscopy-diagnostics", ["psd"]),
    ("Numerical_MHD_Simulation", "numerical-mhd-simulation", ["mhd"]),
]

# Backwards-compatible alias map (alias -> display_name).
TOPIC_ALIASES: dict[str, str] = {a: name for name, _tag, aliases in TOPICS for a in aliases}

_TAG_BY_KEY: dict[str, str] = {}
for _name, _tag, _aliases in TOPICS:
    _TAG_BY_KEY[_name.lower()] = _tag
    _TAG_BY_KEY[_tag.lower()] = _tag
    for _a in _aliases:
        _TAG_BY_KEY[_a.lower()] = _tag
_NAME_BY_TAG: dict[str, str] = {tag: name for name, tag, _a in TOPICS}

# ---------------------------------------------------------------------------
# Topic resolution
# ---------------------------------------------------------------------------


def resolve_tag(name: str) -> str:
    """Resolve a topic name, tag, or alias to its canonical tag.

    Raises:
        SystemExit: If the topic cannot be resolved.
    """
    key = name.lower().replace(" ", "_")
    tag = _TAG_BY_KEY.get(key) or _TAG_BY_KEY.get(name.lower())
    if not tag:
        sys.exit(json.dumps({"error": f"unknown topic '{name}'"}))
    if not _reading_list_path(tag).exists():
        sys.exit(json.dumps({"error": f"reading_lists/{tag}.md not found for '{name}'"}))
    return tag


def display_name(tag: str) -> str:
    """Return the canonical display name for a tag."""
    return _NAME_BY_TAG.get(tag, tag)


def resolve_topic(name: str) -> str:
    """Resolve a topic name/alias to its canonical display name (compat shim)."""
    return display_name(resolve_tag(name))


def _reading_list_path(tag: str) -> Path:
    return PROJECT_ROOT / "reading_lists" / f"{tag}.md"


def citekey_for(name: str, number: int) -> Optional[str]:
    """Look up the flat-papers citekey for (topic, number) via the mapping.

    Reads scripts/flatten_mapping.tsv (keyed by the original topic dir name).

    Returns:
        The citekey string, or None if not found.
    """
    tag = _TAG_BY_KEY.get(name.lower().replace(" ", "_")) or _TAG_BY_KEY.get(name.lower())
    dir_name = _NAME_BY_TAG.get(tag) if tag else None
    mapping = PROJECT_ROOT / "scripts" / "flatten_mapping.tsv"
    if not (dir_name and mapping.exists()):
        return None
    for line in mapping.read_text(encoding="utf-8").splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) >= 4 and cols[2] == dir_name and cols[3] == str(number):
            return cols[1]
    return None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_ENTRY_RE = re.compile(r"^###\s+(\d+)\.\s+(.+)$")
_FIELD_RE = re.compile(r"^-\s+\*\*(\w[\w\s]*)\*\*:\s*(.*)$")
_STATUS_RE = re.compile(r"\[([ x~])\]")


def parse_reading_list(path: Path) -> list[dict]:
    """Parse a reading_list.md into a list of paper entries.

    Args:
        path: Path to reading_list.md.

    Returns:
        List of dicts with keys: number, title, authors, year, why,
        prerequisites, status, journal (optional), raw_status.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    entries: list[dict] = []
    current: Optional[dict] = None
    current_field: Optional[str] = None

    for line in lines:
        entry_match = _ENTRY_RE.match(line)
        if entry_match:
            if current is not None:
                entries.append(current)
            current = {
                "number": int(entry_match.group(1)),
                "title": entry_match.group(2).strip(),
                "authors": "",
                "year": 0,
                "why": "",
                "prerequisites": "",
                "status": " ",
                "raw_status": "[ ]",
            }
            current_field = None
            continue

        if current is None:
            continue

        field_match = _FIELD_RE.match(line)
        if field_match:
            key = field_match.group(1).strip().lower()
            value = field_match.group(2).strip()

            if key == "authors":
                current["authors"] = value
                current_field = "authors"
            elif key == "year":
                year_match = re.search(r"\d{4}", value)
                current["year"] = int(year_match.group()) if year_match else 0
                current_field = "year"
            elif key == "why it matters":
                current["why"] = value
                current_field = "why"
            elif key == "prerequisites":
                current["prerequisites"] = value
                current_field = "prerequisites"
            elif key == "status":
                current["raw_status"] = value.strip()
                status_match = _STATUS_RE.search(value)
                current["status"] = status_match.group(1) if status_match else " "
                current_field = "status"
            elif key == "journal":
                current["journal"] = value
                current_field = "journal"
            elif key == "doi":
                current["doi"] = value
                current_field = "doi"
            else:
                current_field = None
        elif line.startswith("  ") and current_field in ("why", "prerequisites"):
            # Continuation line for multi-line fields
            current[current_field] += " " + line.strip()

    if current is not None:
        entries.append(current)

    return entries



# _make_dir_name removed — use make_dir_name from paper_dir instead.



# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_topics() -> None:
    """List all topics that have a reading list."""
    topics = [name for name, tag, _a in TOPICS
              if _reading_list_path(tag).exists()]
    print(json.dumps({"topics": topics}, ensure_ascii=False))


def cmd_next(topic: str) -> None:
    """Find the next unread paper in a topic."""
    tag = resolve_tag(topic)
    entries = parse_reading_list(_reading_list_path(tag))
    for e in entries:
        if e["status"] == " ":
            result = {
                "number": e["number"],
                "title": e["title"],
                "authors": e["authors"],
                "year": e["year"],
                "dir_name": make_dir_name(e["number"], e["authors"], e["year"]),
                "status": f"[{e['status']}]",
                "prerequisites": e["prerequisites"],
            }
            if "journal" in e:
                result["journal"] = e["journal"]
            print(json.dumps(result, ensure_ascii=False))
            return
    print(json.dumps({"error": "All papers completed", "topic": display_name(tag)}))


def cmd_count(topic: str) -> None:
    """Count papers by status."""
    tag = resolve_tag(topic)
    entries = parse_reading_list(_reading_list_path(tag))
    completed = sum(1 for e in entries if e["status"] == "x")
    in_progress = sum(1 for e in entries if e["status"] == "~")
    total = len(entries)
    print(json.dumps({
        "topic": display_name(tag),
        "completed": completed,
        "in_progress": in_progress,
        "not_started": total - completed - in_progress,
        "total": total,
    }))


def cmd_info(topic: str, number: int) -> None:
    """Get full metadata for a specific paper."""
    tag = resolve_tag(topic)
    entries = parse_reading_list(_reading_list_path(tag))
    for e in entries:
        if e["number"] == number:
            e["dir_name"] = make_dir_name(e["number"], e["authors"], e["year"])
            e["topic"] = display_name(tag)
            e["tag"] = tag
            e["citekey"] = citekey_for(tag, number)
            print(json.dumps(e, ensure_ascii=False))
            return
    print(json.dumps({"error": f"Paper #{number} not found in {display_name(tag)}"}))


def cmd_highest(topic: str) -> None:
    """Get the highest paper number in a topic."""
    tag = resolve_tag(topic)
    entries = parse_reading_list(_reading_list_path(tag))
    highest = max((e["number"] for e in entries), default=0)
    print(json.dumps({"topic": display_name(tag), "highest_number": highest}))


def cmd_mark(topic: str, number: int, status: str) -> None:
    """Change the status of a paper in reading_list.md.

    Args:
        topic: Topic name or alias.
        number: Paper number.
        status: New status character: "x", "~", or " ".
    """
    if status not in ("x", "~", " "):
        sys.exit(json.dumps({"error": f"Invalid status '{status}'. Use 'x', '~', or ' '"}))

    tag = resolve_tag(topic)
    rl_path = _reading_list_path(tag)
    text = rl_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    # Find the entry and its status line
    in_entry = False
    found = False
    for i, line in enumerate(lines):
        entry_match = _ENTRY_RE.match(line)
        if entry_match:
            in_entry = int(entry_match.group(1)) == number
            continue
        if in_entry and "**Status**:" in line:
            # Replace the status
            old_status_match = _STATUS_RE.search(line)
            if old_status_match:
                lines[i] = line[:old_status_match.start(1)] + status + line[old_status_match.end(1):]
                found = True
                break

    if not found:
        sys.exit(json.dumps({"error": f"Paper #{number} status line not found"}))

    rl_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "topic": display_name(tag),
        "number": number,
        "new_status": f"[{status}]",
    }))


def cmd_add(topic: str, title: str, authors: str, year: int,
            why: str, prereqs: str, journal: Optional[str] = None) -> None:
    """Add a new paper to the reading list under 'User-Added Papers'."""
    tag = resolve_tag(topic)
    rl_path = _reading_list_path(tag)
    entries = parse_reading_list(rl_path)
    next_num = max((e["number"] for e in entries), default=0) + 1

    # Build the new entry text
    entry_lines = [
        f"### {next_num}. {title}",
        f"- **Authors**: {authors}",
        f"- **Year**: {year}",
    ]
    if journal:
        entry_lines.append(f"- **Journal**: {journal}")
    entry_lines.extend([
        f"- **Why it matters**: {why}",
        f"- **Prerequisites**: {prereqs}",
        "- **Status**: [ ]",
    ])
    new_entry = "\n".join(entry_lines)

    text = rl_path.read_text(encoding="utf-8")

    # Find or create "User-Added Papers" section
    user_section = "## User-Added Papers / 사용자 추가 논문"
    if user_section in text:
        # Append before the next ## section or at end
        idx = text.index(user_section)
        rest = text[idx + len(user_section):]
        next_section = re.search(r"\n## ", rest)
        if next_section:
            insert_pos = idx + len(user_section) + next_section.start()
            text = text[:insert_pos] + "\n\n" + new_entry + "\n" + text[insert_pos:]
        else:
            text = text.rstrip() + "\n\n" + new_entry + "\n"
    else:
        text = text.rstrip() + "\n\n---\n\n" + user_section + "\n\n" + new_entry + "\n"

    rl_path.write_text(text, encoding="utf-8")

    dir_name = make_dir_name(next_num, authors, year)
    print(json.dumps({
        "ok": True,
        "topic": display_name(tag),
        "number": next_num,
        "dir_name": dir_name,
    }, ensure_ascii=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse and update reading_list.md files."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("topics", help="List all topics")

    p_next = sub.add_parser("next", help="Find next unread paper")
    p_next.add_argument("topic")

    p_count = sub.add_parser("count", help="Count papers by status")
    p_count.add_argument("topic")

    p_info = sub.add_parser("info", help="Get paper metadata")
    p_info.add_argument("topic")
    p_info.add_argument("number", type=int)

    p_highest = sub.add_parser("highest", help="Get highest paper number")
    p_highest.add_argument("topic")

    p_mark = sub.add_parser("mark", help="Change paper status")
    p_mark.add_argument("topic")
    p_mark.add_argument("number", type=int)
    p_mark.add_argument("status", choices=["x", "~", " "],
                        help="New status: x (done), ~ (in progress), ' ' (not started)")

    p_add = sub.add_parser("add", help="Add a new paper")
    p_add.add_argument("topic")
    p_add.add_argument("--title", required=True)
    p_add.add_argument("--authors", required=True)
    p_add.add_argument("--year", type=int, required=True)
    p_add.add_argument("--why", required=True)
    p_add.add_argument("--prereqs", required=True)
    p_add.add_argument("--journal")

    args = parser.parse_args()

    if args.command == "topics":
        cmd_topics()
    elif args.command == "next":
        cmd_next(args.topic)
    elif args.command == "count":
        cmd_count(args.topic)
    elif args.command == "info":
        cmd_info(args.topic, args.number)
    elif args.command == "highest":
        cmd_highest(args.topic)
    elif args.command == "mark":
        cmd_mark(args.topic, args.number, args.status)
    elif args.command == "add":
        cmd_add(args.topic, args.title, args.authors, args.year,
                args.why, args.prereqs, args.journal)


if __name__ == "__main__":
    main()
