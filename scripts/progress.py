#!/usr/bin/env python3
"""Synchronize progress across reading_list.md, README.MD, and WORKFLOW.md.

Usage:
    python scripts/progress.py update <topic> <number>
    python scripts/progress.py status
    python scripts/progress.py verify
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Add scripts/ to path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reading_list import (
    resolve_topic,
    parse_reading_list,
    cmd_mark,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README_PATH = PROJECT_ROOT / "README.MD"
WORKFLOW_PATH = PROJECT_ROOT / "docs" / "WORKFLOW.md"

# Map topic directory names to display names used in README/WORKFLOW
DISPLAY_NAMES: dict[str, str] = {
    "Artificial_Intelligence": "Artificial Intelligence",
    "Solar_Physics": "Solar Physics",
    "Space_Weather": "Space Weather",
    "Solar_Observation": "Solar Observation",
    "Living_Reviews_in_Solar_Physics": "Living Reviews in Solar Physics",
    "Low_SNR_Imaging": "Low-SNR Imaging",
    "Helioseismology_Asteroseismology": "Helioseismology & Asteroseismology",
    "Magnetic_Reconnection_Eruption": "Magnetic Reconnection & Eruption",
    "Heliosphere_Solar_Wind": "Heliosphere & Solar Wind",
    "Plasma_Spectroscopy_Diagnostics": "Plasma Spectroscopy & Diagnostics",
    "Numerical_MHD_Simulation": "Numerical MHD Simulation",
}


# ---------------------------------------------------------------------------
# README.MD editing
# ---------------------------------------------------------------------------

def _update_readme(topic_name: str, completed: int, total: int,
                   paper_number: int, paper_desc: str, paper_year: int) -> None:
    """Update README.MD: increment count and add table row.

    Args:
        topic_name: Topic directory name.
        completed: New completed count.
        total: Total papers.
        paper_number: Number of the newly completed paper.
        paper_desc: Short description for the table row.
        paper_year: Publication year.
    """
    text = README_PATH.read_text(encoding="utf-8")
    display = DISPLAY_NAMES.get(topic_name, topic_name.replace("_", " "))

    # 1. Update the header count: "### <Name> — Paper Reading / 논문 읽기 (N / M)"
    header_re = re.compile(
        rf"(### {re.escape(display)} — Paper Reading / 논문 읽기 \()(\d+)( / )(\d+\+?)\)",
        re.IGNORECASE,
    )
    match = header_re.search(text)
    if match:
        text = text[:match.start(2)] + str(completed) + text[match.end(2):]
        # Refresh match positions after edit
        match = header_re.search(text)

    # 2. Add table row if paper was in the abbreviated range
    # Look for a row like "| 14–40 | (See ...) | ... | — |"
    abbrev_re = re.compile(
        rf"\| (\d+)[–-](\d+\+?) \| \(See.*?\) \|.*?\|.*?\|"
    )
    # Search only within the relevant topic section
    if match:
        section_start = match.start()
        # Find next ### or end
        next_section = re.search(r"\n### ", text[section_start + 1:])
        section_end = section_start + 1 + next_section.start() if next_section else len(text)
        section = text[section_start:section_end]

        abbrev_match = abbrev_re.search(section)
        if abbrev_match:
            range_start = int(abbrev_match.group(1))
            range_end_str = abbrev_match.group(2)

            if paper_number == range_start:
                # Insert new row before the abbreviated row
                new_row = f"| {paper_number} | {paper_desc} | {paper_year} | ✅ Done |\n"
                insert_pos = section_start + abbrev_match.start()
                text = text[:insert_pos] + new_row + text[insert_pos:]

                # Update the range start
                new_range_start = paper_number + 1
                # Re-find the abbreviation row (position shifted)
                text_after = text[insert_pos + len(new_row):]
                abbrev_match2 = abbrev_re.search(text_after)
                if abbrev_match2:
                    old_range = abbrev_match2.group(0)
                    # Check if range is now empty (only one paper left)
                    if str(new_range_start) == range_end_str.rstrip("+"):
                        # Replace range with single number
                        new_range = old_range.replace(
                            f"{range_start}–{range_end_str}",
                            f"{new_range_start}–{range_end_str}",
                        ).replace(
                            f"{range_start}-{range_end_str}",
                            f"{new_range_start}-{range_end_str}",
                        )
                    else:
                        new_range = old_range.replace(
                            f"{range_start}–{range_end_str}",
                            f"{new_range_start}–{range_end_str}",
                        ).replace(
                            f"{range_start}-{range_end_str}",
                            f"{new_range_start}-{range_end_str}",
                        )
                    abs_pos = insert_pos + len(new_row) + abbrev_match2.start()
                    text = (text[:abs_pos] + new_range +
                            text[abs_pos + len(abbrev_match2.group(0)):])

    README_PATH.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# WORKFLOW.MD editing
# ---------------------------------------------------------------------------

def _update_workflow(topic_name: str, completed: int, total: int,
                     paper_number: int, paper_desc: str,
                     next_paper_desc: str | None) -> None:
    """Update WORKFLOW.md: count and next-paper line.

    Args:
        topic_name: Topic directory name.
        completed: New completed count.
        total: Total papers.
        paper_number: Number of completed paper.
        paper_desc: Short description of completed paper (e.g., "#14 Mikolov (2013)").
        next_paper_desc: Description for next paper line, or None if all done.
    """
    if not WORKFLOW_PATH.exists():
        return

    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    display = DISPLAY_NAMES.get(topic_name, topic_name.replace("_", " "))

    # 1. Update header count: "### <Name> — N / M papers"
    header_re = re.compile(
        rf"(### {re.escape(display)} — )(\d+)( / )(\d+\+?)( papers)",
        re.IGNORECASE,
    )
    match = header_re.search(text)
    if match:
        text = text[:match.start(2)] + str(completed) + text[match.end(2):]
        # Also update total if it changed
        text_new = text[:match.start(4)] + str(total) + text[match.end(4):]
        if text_new != text:
            text = text_new

    # 2. Update summary line at top: "- <Topic>: N / M papers"
    summary_re = re.compile(
        rf"(- {re.escape(display)}: )(\d+)( / )(\d+\+?)( papers)",
        re.IGNORECASE,
    )
    s_match = summary_re.search(text)
    if s_match:
        text = text[:s_match.start(2)] + str(completed) + text[s_match.end(2):]
        s_match = summary_re.search(text)
        if s_match:
            text = text[:s_match.start(4)] + str(total) + text[s_match.end(4):]

    # 3. Append to Completed line and update Next paper
    # Find section for this topic
    section_re = re.compile(
        rf"### {re.escape(display)} —",
        re.IGNORECASE,
    )
    sec_match = section_re.search(text)
    if sec_match:
        sec_start = sec_match.start()
        next_sec = re.search(r"\n### ", text[sec_start + 1:])
        sec_end = sec_start + 1 + next_sec.start() if next_sec else len(text)
        section = text[sec_start:sec_end]

        # Append to Completed line
        completed_re = re.compile(r"(\*\*Completed / 완료\*\*:.*?)(\n)")
        c_match = completed_re.search(section)
        if c_match:
            new_completed_line = c_match.group(1) + f", {paper_desc}" + c_match.group(2)
            section = section[:c_match.start()] + new_completed_line + section[c_match.end():]

        # Update Next paper line
        next_re = re.compile(r"\*\*Next paper / 다음 논문\*\*:.*?\n")
        n_match = next_re.search(section)
        if n_match:
            if next_paper_desc:
                new_next = f"**Next paper / 다음 논문**: {next_paper_desc}\n"
            else:
                new_next = "**Next paper / 다음 논문**: All completed! / 전부 완료!\n"
            section = section[:n_match.start()] + new_next + section[n_match.end():]

        text = text[:sec_start] + section + text[sec_end:]

    WORKFLOW_PATH.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_update(topic: str, number: int) -> None:
    """Mark a paper complete and regenerate the README/MOC index.

    In the flat layout the README "Current Progress" section and topic MOCs are
    owned by gen_index.py, so this marks the reading-list status and delegates
    index regeneration there (single source of truth).
    """
    import subprocess
    from reading_list import resolve_tag, display_name, _reading_list_path

    tag = resolve_tag(topic)
    rl_path = _reading_list_path(tag)
    entries = parse_reading_list(rl_path)
    if not any(e["number"] == number for e in entries):
        sys.exit(json.dumps({"error": f"Paper #{number} not found"}))

    cmd_mark(topic, number, "x")

    entries = parse_reading_list(rl_path)
    completed = sum(1 for e in entries if e["status"] == "x")
    total = len(entries)
    next_paper = next((e for e in entries if e["status"] == " "), None)

    subprocess.run(["python3", str(PROJECT_ROOT / "scripts" / "gen_index.py")], check=False)

    print(json.dumps({
        "ok": True,
        "topic": display_name(tag),
        "paper": number,
        "completed": completed,
        "total": total,
        "next_paper": next_paper["number"] if next_paper else None,
    }))


def _iter_reading_lists():
    """Yield (display_name, tag, entries) for every topic reading list."""
    from reading_list import TOPICS, _reading_list_path
    for topic_name, tag, _aliases in TOPICS:
        rl = _reading_list_path(tag)
        if rl.exists():
            yield topic_name, tag, parse_reading_list(rl)


def cmd_status() -> None:
    """Print progress status for all topics (reads reading_lists/)."""
    results = []
    for topic_name, tag, entries in _iter_reading_lists():
        completed = sum(1 for e in entries if e["status"] == "x"
                        or e.get("raw_status", "").lstrip().startswith("✅"))
        in_progress = sum(1 for e in entries if e["status"] == "~")
        next_paper = next(({"number": e["number"], "title": e["title"]}
                           for e in entries if e["status"] == " "), None)
        results.append({
            "topic": topic_name,
            "completed": completed,
            "in_progress": in_progress,
            "total": len(entries),
            "next": next_paper,
        })
    print(json.dumps(results, ensure_ascii=False, indent=2))


def cmd_verify() -> None:
    """Check README index counts agree with the reading lists.

    gen_index.py is the single source for the README "Current Progress" block,
    so this just confirms each topic's "(done / total)" heading matches.
    """
    issues: list[str] = []
    readme_text = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
    for topic_name, tag, entries in _iter_reading_lists():
        done = sum(1 for e in entries if e["status"] == "x"
                   or e.get("raw_status", "").lstrip().startswith("✅"))
        total = len(entries)
        if f"({done} / {total})" not in readme_text:
            issues.append(f"[{topic_name}] README missing '({done} / {total})' "
                          f"— run gen_index.py")
    print(json.dumps({"consistent": len(issues) == 0, "issues": issues},
                     ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize progress tracking files."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_update = sub.add_parser("update", help="Update all 3 files for a completed paper")
    p_update.add_argument("topic")
    p_update.add_argument("number", type=int)

    sub.add_parser("status", help="Show progress for all topics")
    sub.add_parser("verify", help="Check consistency across files")

    args = parser.parse_args()

    if args.command == "update":
        cmd_update(args.topic, args.number)
    elif args.command == "status":
        cmd_status()
    elif args.command == "verify":
        cmd_verify()


if __name__ == "__main__":
    main()
