#!/usr/bin/env python3
"""Merge and rename duplicate paper directories to the first-author-only convention.

Usage:
    python scripts/migrate_dirs.py                # Dry-run (show what would happen)
    python scripts/migrate_dirs.py --execute      # Actually perform the migration
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from paper_dir import make_dir_name
from reading_list import parse_reading_list, resolve_topic, TOPIC_ALIASES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STUDY_MATERIAL_PATTERNS = ["*_notes.md", "*_briefing.md", "*_implementation.ipynb"]


def find_duplicates(topic_dir: Path) -> dict[str, list[Path]]:
    """Find directories sharing the same number prefix under papers/."""
    papers_dir = topic_dir / "papers"
    if not papers_dir.is_dir():
        return {}

    prefix_map: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(papers_dir.iterdir()):
        if d.is_dir() and re.match(r"^\d{2}_", d.name):
            prefix = d.name[:2]
            prefix_map[prefix].append(d)

    return {k: v for k, v in prefix_map.items() if len(v) > 1}


def has_study_materials(d: Path) -> bool:
    """Check if a directory has notes, briefing, or implementation files."""
    for pattern in STUDY_MATERIAL_PATTERNS:
        if list(d.glob(pattern)):
            return True
    # Also check archive subdirectory
    archive = d / "archive"
    if archive.is_dir():
        return True
    return False


def migrate_pair(keeper: Path, empty: Path, correct_name: str,
                 papers_dir: Path, execute: bool) -> list[str]:
    """Merge and rename a duplicate pair.

    Args:
        keeper: Directory with study materials.
        empty: Directory to merge from and delete.
        correct_name: The correct directory name under new convention.
        papers_dir: Parent papers/ directory.
        execute: Whether to actually perform operations.

    Returns:
        List of log messages.
    """
    logs = []
    old_name = keeper.name
    target_dir = papers_dir / correct_name

    # Step 1: Move files from empty dir to keeper (avoid overwriting)
    for f in empty.iterdir():
        if f.is_file():
            dest = keeper / f.name
            if not dest.exists():
                logs.append(f"  MOVE {empty.name}/{f.name} -> {keeper.name}/{f.name}")
                if execute:
                    shutil.move(str(f), str(dest))
            else:
                # Check if the file in keeper is smaller (possibly corrupt)
                if dest.stat().st_size < f.stat().st_size:
                    logs.append(f"  REPLACE {keeper.name}/{f.name} (larger version from {empty.name})")
                    if execute:
                        shutil.move(str(f), str(dest))
                else:
                    logs.append(f"  SKIP {empty.name}/{f.name} (already exists in keeper)")

    # Step 2: Delete empty directory
    logs.append(f"  DELETE dir {empty.name}/")
    if execute:
        shutil.rmtree(str(empty))

    # Step 3: Rename internal files if keeper needs renaming
    if old_name != correct_name:
        for f in keeper.iterdir():
            if f.is_file() and f.name.startswith(old_name):
                new_fname = f.name.replace(old_name, correct_name, 1)
                logs.append(f"  RENAME file {f.name} -> {new_fname}")
                if execute:
                    f.rename(keeper / new_fname)
            elif f.is_dir() and f.name == "archive":
                # Rename files inside archive too
                for af in f.iterdir():
                    if af.is_file() and af.name.startswith(old_name):
                        new_afname = af.name.replace(old_name, correct_name, 1)
                        logs.append(f"  RENAME archive/{af.name} -> archive/{new_afname}")
                        if execute:
                            af.rename(f / new_afname)

        # Step 4: Update internal file references
        for f in (keeper if not execute else papers_dir / correct_name).parent.parent.glob("**/*"):
            pass  # Will handle after rename

        # Update references inside .md and .ipynb files
        for f in keeper.glob("*"):
            if f.is_file() and f.suffix in (".md", ".ipynb"):
                try:
                    content = f.read_text(encoding="utf-8")
                    if old_name in content:
                        new_content = content.replace(old_name, correct_name)
                        logs.append(f"  UPDATE refs in {f.name}")
                        if execute:
                            f.write_text(new_content, encoding="utf-8")
                except (UnicodeDecodeError, PermissionError):
                    pass

        # Step 5: Rename the directory itself
        logs.append(f"  RENAME dir {old_name}/ -> {correct_name}/")
        if execute:
            keeper.rename(target_dir)

    return logs


def main():
    parser = argparse.ArgumentParser(description="Migrate duplicate paper directories.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually perform the migration (default: dry-run)")
    args = parser.parse_args()

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"=== Directory Migration ({mode}) ===\n")

    total_pairs = 0
    total_actions = 0

    for alias, topic_name in sorted(TOPIC_ALIASES.items()):
        topic_dir = PROJECT_ROOT / topic_name
        if not topic_dir.is_dir():
            continue

        papers_dir = topic_dir / "papers"
        rl_path = papers_dir / "reading_list.md"
        if not rl_path.exists():
            continue

        entries = parse_reading_list(rl_path)
        entry_map = {e["number"]: e for e in entries}

        duplicates = find_duplicates(topic_dir)
        if not duplicates:
            continue

        print(f"--- {topic_name} ---")

        for prefix, dirs in sorted(duplicates.items()):
            num = int(prefix)
            entry = entry_map.get(num)
            if not entry:
                print(f"  WARNING: No reading list entry for #{num}, skipping")
                continue

            correct_name = make_dir_name(num, entry["authors"], entry["year"])

            # Determine keeper (has study materials) vs empty
            keeper = None
            empties = []
            for d in dirs:
                if has_study_materials(d):
                    if keeper is None:
                        keeper = d
                    else:
                        # Multiple dirs with materials — pick the one with more files
                        if len(list(d.iterdir())) > len(list(keeper.iterdir())):
                            empties.append(keeper)
                            keeper = d
                        else:
                            empties.append(d)
                else:
                    empties.append(d)

            if keeper is None:
                # No study materials in any — pick the one with more files
                dirs_sorted = sorted(dirs, key=lambda d: len(list(d.iterdir())), reverse=True)
                keeper = dirs_sorted[0]
                empties = dirs_sorted[1:]

            for empty in empties:
                total_pairs += 1
                print(f"\n  #{num}: {keeper.name} (keeper) + {empty.name} (merge) -> {correct_name}")
                logs = migrate_pair(keeper, empty, correct_name, papers_dir, args.execute)
                for log in logs:
                    print(log)
                total_actions += len(logs)

    print(f"\n=== Summary ===")
    print(f"Duplicate pairs processed: {total_pairs}")
    print(f"Actions {'performed' if args.execute else 'planned'}: {total_actions}")

    if not args.execute and total_pairs > 0:
        print(f"\nRe-run with --execute to apply changes.")


if __name__ == "__main__":
    main()
