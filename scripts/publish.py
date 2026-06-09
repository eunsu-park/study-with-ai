#!/usr/bin/env python3
"""Publish completed paper notes to the Personal Knowledge Base (PKB).

Copies each completed paper's ``_notes.md`` and ``_paper.pdf`` from the flat
``papers/<citekey>/`` folder into ``<knowledge_base>/raw/papers/<citekey>/``.
The PKB root is read from ``scripts/publish_config.json``.

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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "publish_config.json"

sys.path.insert(0, str(SCRIPT_DIR))
from reading_list import (  # noqa: E402
    TOPICS, parse_reading_list, resolve_tag, display_name,
    _reading_list_path, citekey_for, is_done,
)
from paper_dir import make_file_names  # noqa: E402

PAPERS = PROJECT_ROOT / "papers"


def load_config() -> dict:
    """Load publish config; require an existing knowledge_base directory."""
    if not CONFIG_PATH.exists():
        sys.exit(json.dumps({
            "error": "Config not found",
            "message": f"Create {CONFIG_PATH} with a 'knowledge_base' path.",
        }))
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(json.dumps({"error": f"Invalid JSON in config: {e}"}))
    if "knowledge_base" not in config:
        sys.exit(json.dumps({"error": "Missing 'knowledge_base' in config"}))
    if not Path(config["knowledge_base"]).is_dir():
        sys.exit(json.dumps({
            "error": f"knowledge_base directory not found: {config['knowledge_base']}",
        }))
    return config


def _needs_copy(src: Path, dst: Path, force: bool = False) -> bool:
    """Whether src should be copied to dst (missing or newer)."""
    if force or not dst.exists():
        return True
    return os.path.getmtime(src) > os.path.getmtime(dst)


def completed_citekeys(tag: str) -> list[tuple[int, str]]:
    """Return (number, citekey) for completed, non-migrated papers of a topic.

    Migrated cross-references are excluded so each paper publishes once, from
    its home topic (the topic where it physically lives).
    """
    rl = _reading_list_path(tag)
    if not rl.exists():
        return []
    out = []
    seen = set()
    for e in parse_reading_list(rl):
        if not is_done(e):
            continue
        # Skip cross-references ("[x] (migrated from X #N)") so a paper publishes
        # once, from its home topic; ✅-migrated home entries are kept.
        if "migrated from" in e.get("raw_status", ""):
            continue
        ck = citekey_for(tag, e["number"])
        if ck and ck not in seen:
            seen.add(ck)
            out.append((e["number"], ck))
    return out


def publish_paper(citekey: str, topic_name: str, config: dict, force: bool) -> dict:
    """Copy one paper's notes + pdf into the PKB."""
    files = make_file_names(citekey)
    src_dir = PAPERS / citekey
    kb_dir = Path(config["knowledge_base"]) / "raw" / "papers" / citekey

    result = {"paper": citekey, "topic": topic_name,
              "published": [], "skipped": [], "missing": []}

    for file_type in ("notes", "pdf"):
        src = src_dir / files[file_type]
        if not src.exists():
            if file_type == "notes":
                result["missing"].append(files[file_type])
            continue
        dst = kb_dir / files[file_type]
        if _needs_copy(src, dst, force):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            result["published"].append(files[file_type])
        else:
            result["skipped"].append(files[file_type])
    return result


def cmd_publish(topic: str | None, number: int | None, force: bool, all_topics: bool) -> None:
    """Publish completed papers to the PKB."""
    config = load_config()
    results = []

    targets = [(name, tag) for name, tag, _a in TOPICS] if all_topics \
        else [(display_name(resolve_tag(topic)), resolve_tag(topic))]

    for topic_name, tag in targets:
        if number is not None:
            ck = citekey_for(tag, number)
            if not ck:
                results.append({"topic": topic_name,
                                "error": f"Paper #{number} has no flat folder (citekey)"})
                continue
            results.append(publish_paper(ck, topic_name, config, force))
        else:
            papers = completed_citekeys(tag)
            if not papers:
                results.append({"topic": topic_name, "message": "No completed papers"})
                continue
            for _num, ck in papers:
                results.append(publish_paper(ck, topic_name, config, force))

    total = sum(len(r.get("published", [])) for r in results)
    skipped = sum(len(r.get("skipped", [])) for r in results)
    print(json.dumps({
        "results": results,
        "summary": {"published": total, "skipped_unchanged": skipped,
                    "knowledge_base": config["knowledge_base"]},
    }, indent=2, ensure_ascii=False))


def cmd_status() -> None:
    """Show publish status for all completed papers."""
    config = load_config()
    kb = Path(config["knowledge_base"])
    status = []
    for topic_name, tag, _a in TOPICS:
        for _num, ck in completed_citekeys(tag):
            files = make_file_names(ck)
            src_dir = PAPERS / ck
            kb_dir = kb / "raw" / "papers" / ck
            notes_src = src_dir / files["notes"]
            pdf_src = src_dir / files["pdf"]
            status.append({
                "topic": topic_name,
                "paper": ck,
                "source": {"notes": notes_src.exists(), "pdf": pdf_src.exists()},
                "knowledge_base": {
                    "notes": (kb_dir / files["notes"]).exists(),
                    "pdf": (kb_dir / files["pdf"]).exists(),
                },
                "needs_update": {
                    "notes": notes_src.exists() and _needs_copy(notes_src, kb_dir / files["notes"]),
                    "pdf": pdf_src.exists() and _needs_copy(pdf_src, kb_dir / files["pdf"]),
                },
            })
    print(json.dumps(status, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish completed paper files to the Personal Knowledge Base.",
    )
    parser.add_argument("topic", nargs="?", help="Topic name or alias (ai, sp, ...)")
    parser.add_argument("number", nargs="?", type=int,
                        help="Paper number (omit to publish all completed)")
    parser.add_argument("--all", action="store_true", dest="all_topics",
                        help="Publish all completed papers across all topics")
    parser.add_argument("--status", action="store_true",
                        help="Show publish status")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite even if destination is up to date")
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
