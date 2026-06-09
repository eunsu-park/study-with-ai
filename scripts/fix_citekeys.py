#!/usr/bin/env python3
"""Repair degenerate citekeys where the title keyword duplicated the surname.

Some source reading lists held stub titles ("Leighton+ 1962 (...) -> migrated"),
so the flatten step picked the surname as the keyword, yielding citekeys like
``leighton1962leighton``. This recomputes the keyword from the real title stored
in each paper's ``*_notes.md`` frontmatter (skipping surname / reading / notes /
author-list tokens). Papers with no notes (PDF-only) are shortened to
``<surname><year>``.

Updates folder name, inner file names, and ``scripts/flatten_mapping.tsv``.
Dry-run by default; ``--execute`` applies. Re-run ``gen_index.py`` afterwards.
"""

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS = PROJECT_ROOT / "papers"
MAPPING = PROJECT_ROOT / "scripts" / "flatten_mapping.tsv"

_CK_RE = re.compile(r"^([a-z]+?)(\d{4})(.*)$")
_EXTRA_SKIP = {"reading", "notes", "et", "al", "the", "a", "an", "on", "of",
               "in", "for", "and", "to", "with", "by", "from", "as", "via",
               "using", "new", "der", "die", "das", "und", "uber", "von"}
_ROMAN_RE = re.compile(r"^[ivxlcdm]+$")


def read_title(notes: Path) -> str:
    """Extract the frontmatter title from a notes file, else ''."""
    if not notes.exists():
        return ""
    for line in notes.read_text(encoding="utf-8").splitlines()[:20]:
        m = re.match(r"title:\s*[\"']?(.+?)[\"']?\s*$", line)
        if m:
            return m.group(1)
    return ""


def pick_keyword(title: str, surname: str) -> str:
    """Pick a keyword from a title, skipping surname / boilerplate / years."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", title)
    for w in words:
        token = re.sub(r"[^a-z0-9]", "", w.lower())
        if len(token) < 3 or token.isdigit():
            continue
        if token in _EXTRA_SKIP or _ROMAN_RE.match(token):
            continue
        if token == surname or token.startswith(surname) or surname.startswith(token):
            continue
        return token
    return ""


def existing_citekeys() -> set[str]:
    return {p.name for p in PAPERS.iterdir() if p.is_dir()}


def plan() -> list[tuple[str, str]]:
    """Return list of (old_citekey, new_citekey) for degenerate folders."""
    taken = existing_citekeys()
    out = []
    for folder in sorted(PAPERS.iterdir()):
        if not folder.is_dir():
            continue
        ck = folder.name
        m = _CK_RE.match(ck)
        if not m:
            continue
        surname, year, kw = m.group(1), m.group(2), m.group(3)
        if kw != surname:  # only repair the doubled ones
            continue
        title = read_title(folder / f"{ck}_notes.md")
        new_kw = pick_keyword(title, surname) if title else ""
        base = f"{surname}{year}{new_kw}"
        # clash-safe against everything except itself
        cand = base
        n = 0
        while cand in taken and cand != ck:
            cand = f"{base}{chr(ord('a') + n)}"
            n += 1
        if cand != ck:
            taken.discard(ck)
            taken.add(cand)
            out.append((ck, cand))
    return out


def apply(old: str, new: str) -> None:
    folder = PAPERS / old
    newfolder = PAPERS / new
    for f in list(folder.iterdir()):
        if f.is_file() and f.name.startswith(old):
            f.rename(folder / f"{new}{f.name[len(old):]}")
    folder.rename(newfolder)


def update_mapping(pairs: list[tuple[str, str]]) -> None:
    if not MAPPING.exists():
        return
    text = MAPPING.read_text(encoding="utf-8")
    lines = text.splitlines()
    repl = {old: new for old, new in pairs}
    for i, line in enumerate(lines):
        cols = line.split("\t")
        if len(cols) >= 2 and cols[1] in repl:
            cols[1] = repl[cols[1]]
            lines[i] = "\t".join(cols)
    MAPPING.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair doubled citekeys.")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    pairs = plan()
    print(f"=== {len(pairs)} citekeys to repair "
          f"[{'EXECUTE' if args.execute else 'DRY-RUN'}] ===\n")
    for old, new in pairs:
        flag = "" if new[-1].isalpha() and not new[len(re.match(r'[a-z]+', new).group()):][:4].isdigit() else ""
        shortened = " (shortened, no title)" if re.match(rf"^[a-z]+\d{{4}}$", new) else ""
        print(f"  {old:45} ->  {new}{shortened}")
    if not args.execute:
        print("\n(DRY-RUN — re-run with --execute, then run gen_index.py)")
        return
    for old, new in pairs:
        apply(old, new)
    update_mapping(pairs)
    print(f"\nRepaired {len(pairs)} folders + updated mapping. "
          f"Now run: python3 scripts/gen_index.py")


if __name__ == "__main__":
    main()
