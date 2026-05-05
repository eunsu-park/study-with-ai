#!/usr/bin/env python3
"""Publish completed paper notes and implementations to external repositories.

Copies _notes.md and _implementation.ipynb to a GitHub repository,
and _notes.md + _paper.pdf to a KnowledgeBase directory.

Usage:
    python scripts/publish.py <topic> [number]   # specific paper(s)
    python scripts/publish.py --all               # all topics, all completed
    python scripts/publish.py --status            # show publish status
    python scripts/publish.py --force <topic>     # overwrite even if unchanged
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "publish_config.json"

# Import shared utilities
sys.path.insert(0, str(SCRIPT_DIR))
from reading_list import TOPIC_ALIASES, parse_reading_list, resolve_topic
from paper_dir import make_dir_name, make_file_names

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config() -> dict:
    """Load publish configuration from publish_config.json.

    Returns:
        Dict with 'github_repo' and 'knowledge_base' paths.

    Raises:
        SystemExit: If config file is missing or invalid.
    """
    if not CONFIG_PATH.exists():
        sys.exit(json.dumps({
            "error": "Config not found",
            "message": f"Create {CONFIG_PATH} with github_repo and knowledge_base paths.",
            "example": {
                "github_repo": "/path/to/github/repo",
                "knowledge_base": "/path/to/knowledge/base",
            },
        }))
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(json.dumps({"error": f"Invalid JSON in config: {e}"}))

    for key in ("github_repo", "knowledge_base"):
        if key not in config:
            sys.exit(json.dumps({"error": f"Missing '{key}' in config"}))
        path = Path(config[key])
        if not path.is_dir():
            sys.exit(json.dumps({
                "error": f"Directory not found for '{key}': {config[key]}",
            }))
    return config


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _find_paper_dir(topic_dir: Path, number: int) -> Path | None:
    """Find the paper directory for a given number under a topic.

    Args:
        topic_dir: Resolved topic directory path.
        number: Paper number.

    Returns:
        Path to the paper directory, or None if not found.
    """
    papers_dir = topic_dir / "papers"
    prefix = f"{number:02d}_"
    for d in sorted(papers_dir.iterdir()):
        if d.is_dir() and d.name.startswith(prefix):
            return d
    return None


def _needs_copy(src: Path, dst: Path, force: bool = False) -> bool:
    """Check if src should be copied to dst.

    Args:
        src: Source file path.
        dst: Destination file path.
        force: If True, always copy.

    Returns:
        True if copy is needed.
    """
    if force:
        return True
    if not dst.exists():
        return True
    return os.path.getmtime(src) > os.path.getmtime(dst)


def publish_paper(
    topic_dir: Path,
    paper_dir: Path,
    config: dict,
    force: bool = False,
) -> dict:
    """Publish a single paper's files to GitHub repo and KnowledgeBase.

    Args:
        topic_dir: Resolved topic directory path.
        paper_dir: Path to the paper directory.
        config: Loaded config dict.
        force: If True, overwrite even if unchanged.

    Returns:
        Dict with publish results for this paper.
    """
    topic_name = topic_dir.name
    dir_name = paper_dir.name
    file_names = make_file_names(dir_name)

    result = {
        "paper": dir_name,
        "topic": topic_name,
        "github": [],
        "knowledge_base": [],
        "skipped": [],
        "missing": [],
    }

    github_repo = Path(config["github_repo"])
    kb_path = Path(config["knowledge_base"])

    # --- GitHub repo: notes + implementation ---
    # Target: repo/Topic/paper_name/
    github_paper_dir = github_repo / topic_name / dir_name
    skip_impl = (paper_dir / ".no_implementation").exists()

    for file_type in ("notes", "implementation"):
        src = paper_dir / file_names[file_type]
        if not src.exists():
            if file_type == "implementation" and skip_impl:
                result["skipped"].append(f"github:{file_names[file_type]} (N/A)")
            else:
                result["missing"].append(file_names[file_type])
            continue

        dst = github_paper_dir / file_names[file_type]
        if _needs_copy(src, dst, force):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            result["github"].append(file_names[file_type])
        else:
            result["skipped"].append(f"github:{file_names[file_type]}")

    # --- KnowledgeBase: notes + pdf ---
    # Target: kb/raw/papers/paper_name/
    kb_paper_dir = kb_path / "raw" / "papers" / dir_name

    for file_type in ("notes", "pdf"):
        src = paper_dir / file_names[file_type]
        if not src.exists():
            if file_type == "notes":
                result["missing"].append(file_names[file_type])
            continue

        dst = kb_paper_dir / file_names[file_type]
        if _needs_copy(src, dst, force):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            result["knowledge_base"].append(file_names[file_type])
        else:
            result["skipped"].append(f"kb:{file_names[file_type]}")

    return result


def get_completed_papers(topic_dir: Path) -> list[dict]:
    """Get all completed papers for a topic.

    Args:
        topic_dir: Resolved topic directory path.

    Returns:
        List of paper entry dicts with status == 'x'.
    """
    rl_path = topic_dir / "papers" / "reading_list.md"
    if not rl_path.exists():
        return []
    entries = parse_reading_list(rl_path)
    return [e for e in entries if e["status"] == "x"]


def get_all_topics() -> list[str]:
    """Get all topic directory names that have a reading list.

    Returns:
        List of topic directory names.
    """
    topics = []
    for d in sorted(PROJECT_ROOT.iterdir()):
        if d.is_dir() and (d / "papers" / "reading_list.md").exists():
            topics.append(d.name)
    return topics


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_publish(topic: str | None, number: int | None, force: bool, all_topics: bool) -> None:
    """Publish papers to external repositories.

    Args:
        topic: Topic name or alias (None if --all).
        number: Specific paper number (None for auto-detect).
        force: Overwrite even if unchanged.
        all_topics: Process all topics.
    """
    config = load_config()
    results = []

    if all_topics:
        topics = get_all_topics()
    else:
        topic_dir = resolve_topic(topic)
        topics = [topic_dir.name]

    for topic_name in topics:
        topic_dir = resolve_topic(topic_name)

        if number is not None:
            paper_dir = _find_paper_dir(topic_dir, number)
            if paper_dir is None:
                results.append({
                    "paper": f"{number:02d}_*",
                    "topic": topic_name,
                    "error": f"Paper #{number} directory not found",
                })
                continue
            r = publish_paper(topic_dir, paper_dir, config, force)
            results.append(r)
        else:
            completed = get_completed_papers(topic_dir)
            if not completed:
                results.append({
                    "topic": topic_name,
                    "message": "No completed papers found",
                })
                continue
            for entry in completed:
                paper_dir = _find_paper_dir(topic_dir, entry["number"])
                if paper_dir is None:
                    results.append({
                        "paper": f"{entry['number']:02d}_*",
                        "topic": topic_name,
                        "error": "Directory not found",
                    })
                    continue
                r = publish_paper(topic_dir, paper_dir, config, force)
                results.append(r)

    # Summary
    total_github = sum(len(r.get("github", [])) for r in results)
    total_kb = sum(len(r.get("knowledge_base", [])) for r in results)
    total_skipped = sum(len(r.get("skipped", [])) for r in results)

    print(json.dumps({
        "results": results,
        "summary": {
            "copied_to_github": total_github,
            "copied_to_kb": total_kb,
            "skipped_unchanged": total_skipped,
        },
    }, indent=2, ensure_ascii=False))


def cmd_status() -> None:
    """Show publish status for all topics."""
    config = load_config()
    github_repo = Path(config["github_repo"])
    kb_path = Path(config["knowledge_base"])

    status = []
    for topic_name in get_all_topics():
        topic_dir = resolve_topic(topic_name)
        completed = get_completed_papers(topic_dir)

        for entry in completed:
            paper_dir = _find_paper_dir(topic_dir, entry["number"])
            if paper_dir is None:
                continue

            dir_name = paper_dir.name
            file_names = make_file_names(dir_name)

            notes_src = paper_dir / file_names["notes"]
            impl_src = paper_dir / file_names["implementation"]
            pdf_src = paper_dir / file_names["pdf"]

            gh_notes = github_repo / topic_name / dir_name / file_names["notes"]
            gh_impl = github_repo / topic_name / dir_name / file_names["implementation"]
            kb_notes = kb_path / "raw" / "papers" / dir_name / file_names["notes"]
            kb_pdf = kb_path / "raw" / "papers" / dir_name / file_names["pdf"]

            paper_status = {
                "topic": topic_name,
                "paper": dir_name,
                "source": {
                    "notes": notes_src.exists(),
                    "implementation": impl_src.exists(),
                    "pdf": pdf_src.exists(),
                },
                "github": {
                    "notes": gh_notes.exists(),
                    "implementation": gh_impl.exists(),
                },
                "knowledge_base": {
                    "notes": kb_notes.exists(),
                    "pdf": kb_pdf.exists(),
                },
                "needs_update": {
                    "github_notes": (
                        notes_src.exists()
                        and _needs_copy(notes_src, gh_notes)
                    ),
                    "github_impl": (
                        impl_src.exists()
                        and _needs_copy(impl_src, gh_impl)
                    ),
                    "kb_notes": (
                        notes_src.exists()
                        and _needs_copy(notes_src, kb_notes)
                    ),
                    "kb_pdf": (
                        pdf_src.exists()
                        and _needs_copy(pdf_src, kb_pdf)
                    ),
                },
            }
            status.append(paper_status)

    print(json.dumps(status, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish completed paper files to GitHub repo and KnowledgeBase.",
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Topic name or alias (e.g., AI, SP, SW, SO, LRSP)",
    )
    parser.add_argument(
        "number",
        nargs="?",
        type=int,
        help="Paper number (omit to auto-detect all completed)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_topics",
        help="Publish all completed papers across all topics",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show publish status for all completed papers",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite even if destination file is up to date",
    )

    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.all_topics:
        cmd_publish(None, None, args.force, all_topics=True)
    elif args.topic:
        cmd_publish(args.topic, args.number, args.force, all_topics=False)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
