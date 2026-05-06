#!/usr/bin/env python3
"""Convert LaTeX `\\(...\\)` and `\\[...\\]` delimiters to `$...$` and `$$...$$`.

Why: Obsidian's MathJax and GitHub Markdown only support `$...$` (inline) and
`$$...$$` (block) delimiters. The standard LaTeX `\\(...\\)` / `\\[...\\]`
delimiters used in many academic templates do not render in those viewers.

Scope:
    - Markdown files (.md): convert in body text, skipping fenced code blocks.
    - Jupyter notebooks (.ipynb): convert inside markdown cells only.

Conservative rules (false-positive guards):
    - Inline `\\(...\\)` is matched only on a single line, and only when the
      opening `\\(` is not preceded by another backslash (so `\\\\(` — typically
      a LaTeX line break followed by literal `(` — is left alone).
    - Block `\\[...\\]` is matched only when both delimiters sit on their own
      lines. This skips citation references like `\\[15\\]` (always inline) and
      LaTeX line-spacing args like `\\\\[6pt]` inside `$$...$$` blocks.
    - Fenced code blocks (```...```) are tokenized out and restored unchanged.

Usage:
    python scripts/fix_latex_delimiters.py              # dry-run (default)
    python scripts/fix_latex_delimiters.py --execute    # apply changes
    python scripts/fix_latex_delimiters.py --diff       # show per-line diff
    python scripts/fix_latex_delimiters.py path/to/file # restrict to one file
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FENCE_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
# Inline code span: backtick(s)...same backtick(s). Skip these so literal
# examples like `\(...\)` in documentation aren't rewritten.
INLINE_CODE_RE = re.compile(r"(`+)(?:(?!`).)+?\1", re.DOTALL)
INLINE_RE = re.compile(r"(?<!\\)\\\(([^\n]*?)(?<!\\)\\\)")
# Block: optional leading indent, \[ on its own line, content, \] on its own line.
# Captures indent so it can be preserved on the $$ delimiters.
BLOCK_RE = re.compile(
    r"^([ \t]*)\\\[[ \t]*\n(.*?)\n[ \t]*\\\]\s*$",
    re.MULTILINE | re.DOTALL,
)

SKIP_DIR_PARTS = {".git", "node_modules", "archive", "__pycache__", ".venv"}


def convert_text(text: str) -> tuple[str, int, int]:
    """Convert delimiters in a markdown text blob.

    Returns:
        (new_text, num_inline_changes, num_block_changes)
    """
    stash: list[str] = []

    def stash_match(match: re.Match[str]) -> str:
        stash.append(match.group(0))
        return f"\x00STASH{len(stash) - 1}\x00"

    # Stash fenced code blocks first (they may contain backticks).
    text = FENCE_RE.sub(stash_match, text)
    # Then stash inline code spans.
    text = INLINE_CODE_RE.sub(stash_match, text)

    def block_repl(match: re.Match[str]) -> str:
        indent, content = match.group(1), match.group(2)
        return f"{indent}$$\n{content}\n{indent}$$"

    new_text, n_block = BLOCK_RE.subn(block_repl, text)
    new_text, n_inline = INLINE_RE.subn(r"$\1$", new_text)

    for i, original in enumerate(stash):
        new_text = new_text.replace(f"\x00STASH{i}\x00", original)

    return new_text, n_inline, n_block


def convert_markdown_file(path: Path, execute: bool, show_diff: bool) -> dict:
    """Convert a markdown file in place (or dry-run)."""
    original = path.read_text(encoding="utf-8")
    new_text, n_inline, n_block = convert_text(original)
    changed = new_text != original
    if changed and execute:
        path.write_text(new_text, encoding="utf-8")
    if changed and show_diff:
        _print_diff(path, original, new_text)
    return {
        "path": path,
        "changed": changed,
        "inline": n_inline,
        "block": n_block,
    }


def convert_notebook_file(path: Path, execute: bool, show_diff: bool) -> dict:
    """Convert markdown cells in a Jupyter notebook (or dry-run)."""
    original_text = path.read_text(encoding="utf-8")
    nb = json.loads(original_text)
    total_inline = 0
    total_block = 0
    cells_changed = 0

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            joined = "".join(source)
        else:
            joined = source
        new_joined, n_inline, n_block = convert_text(joined)
        if new_joined != joined:
            cells_changed += 1
            total_inline += n_inline
            total_block += n_block
            if isinstance(source, list):
                cell["source"] = new_joined.splitlines(keepends=True)
            else:
                cell["source"] = new_joined

    changed = cells_changed > 0
    if changed and execute:
        # Preserve trailing newline if original had one.
        trailing = "\n" if original_text.endswith("\n") else ""
        path.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False) + trailing,
            encoding="utf-8",
        )
    if changed and show_diff:
        # Re-serialize for diffing
        new_text = json.dumps(nb, indent=1, ensure_ascii=False)
        _print_diff(path, original_text, new_text, max_lines=40)

    return {
        "path": path,
        "changed": changed,
        "inline": total_inline,
        "block": total_block,
        "cells_changed": cells_changed,
    }


def _print_diff(path: Path, before: str, after: str, max_lines: int = 60) -> None:
    """Print a minimal context diff, truncated to max_lines."""
    import difflib

    diff = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=str(path),
            tofile=str(path) + " (converted)",
            lineterm="",
            n=1,
        )
    )
    print(f"\n--- diff: {path} ---")
    for line in diff[:max_lines]:
        print(line)
    if len(diff) > max_lines:
        print(f"... ({len(diff) - max_lines} more diff lines suppressed)")


def iter_target_files(roots: list[Path]) -> list[Path]:
    """Yield .md and .ipynb files under given roots, skipping unwanted dirs."""
    targets: list[Path] = []
    for root in roots:
        if root.is_file():
            targets.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".md", ".ipynb"}:
                continue
            if any(part in SKIP_DIR_PARTS for part in path.parts):
                continue
            targets.append(path)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to scan (default: project root)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write changes to disk (default: dry-run only)",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Print unified diff for each changed file",
    )
    args = parser.parse_args()

    roots = args.paths if args.paths else [ROOT]
    files = iter_target_files(roots)

    results = []
    for path in files:
        try:
            if path.suffix == ".ipynb":
                result = convert_notebook_file(path, args.execute, args.diff)
            else:
                result = convert_markdown_file(path, args.execute, args.diff)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR processing {path}: {exc}", file=sys.stderr)
            continue
        results.append(result)

    changed = [r for r in results if r["changed"]]
    print("\n=== Summary ===")
    print(f"Files scanned:   {len(results)}")
    print(f"Files to change: {len(changed)}")
    total_inline = sum(r["inline"] for r in changed)
    total_block = sum(r["block"] for r in changed)
    print(f"Inline replacements: {total_inline}")
    print(f"Block replacements:  {total_block}")

    if changed:
        print("\nChanged files:")
        for r in changed:
            rel = r["path"].relative_to(ROOT) if r["path"].is_absolute() else r["path"]
            extra = ""
            if "cells_changed" in r:
                extra = f", cells={r['cells_changed']}"
            print(f"  {rel}  (inline={r['inline']}, block={r['block']}{extra})")

    if not args.execute and changed:
        print("\nDry-run complete. Re-run with --execute to apply changes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
