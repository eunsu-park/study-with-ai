#!/usr/bin/env python3
"""De-duplicate papers that were stored under two citekeys (same PDF).

Some papers were cross-listed in two topic reading lists, so the flatten step
produced two folders with identical PDFs. This keeps ONE canonical folder per
paper (the better citekey, carrying the richer notes) and removes the redundant
one, then repoints both topics' mapping rows at the survivor so the index lists
the paper under both topics (cross-reference).

Pairs and the canonical citekey are declared explicitly below (decided from a
content-hash + notes-length review). Dry-run by default; ``--execute`` applies.
Re-run ``gen_index.py`` afterwards.
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS = PROJECT_ROOT / "papers"
MAPPING = PROJECT_ROOT / "scripts" / "flatten_mapping.tsv"

# (final_citekey, content_source, redundant_to_delete)
#   final_citekey   : the citekey the surviving folder must end up named
#   content_source  : the folder whose contents to KEEP (richer notes)
#   redundant       : the folder to remove
# When content_source == final_citekey, just delete `redundant`.
# When content_source != final_citekey, delete final_citekey's folder, then
#   rename content_source's folder (and files) to final_citekey.
PAIRS = [
    # final,                  content_source,            redundant
    ("rimmele2011solar",      "rimmele2011solara",       "rimmele2011solar"),
    ("parker1958dynamics",    "parker1958dynamics",      "parker1958parker"),
    ("kaiser2008stereo",      "kaiser2008stereo",        "kaiser2008kaiser"),
    ("fox2016solar",          "fox2016solar",            "fox2016fox"),
    ("muller2020solar",       "muller2020solar",         "muller2020ller"),
    ("depontieu2014interface","depontieu2014interface",  "depontieu2014pontieu"),
    ("woods2012extreme",      "woods2012extreme",        "woods2012woods"),
    ("pesnell2012solar",      "pesnell2012solar",        "pesnell2012pesnell"),
    ("angelopoulos2008themis","angelopoulos2008themis",  "angelopoulos2008themisa"),
    ("charbonneau2010dynamo", "charbonneau2010dynamo",   "charbonneau2010charbonneau"),
    ("rimmele2020daniel",     "rimmele2020daniel",       "rimmele2020rimmele"),
    ("kosugi2007hinode",      "kosugi2007hinode",        "kosugi2007kosugi"),
    ("pulkkinen2007space",    "pulkkinen2007space",      "pulkkinen2007spacea"),
]


def rename_folder(src_ck: str, dst_ck: str) -> None:
    src = PAPERS / src_ck
    for f in list(src.iterdir()):
        if f.is_file() and f.name.startswith(src_ck):
            f.rename(src / f"{dst_ck}{f.name[len(src_ck):]}")
    src.rename(PAPERS / dst_ck)


def update_mapping(removed_to_survivor: dict[str, str]) -> None:
    """Repoint mapping rows whose citekey was removed onto the survivor."""
    if not MAPPING.exists():
        return
    lines = MAPPING.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        cols = line.split("\t")
        if len(cols) >= 2 and cols[1] in removed_to_survivor:
            cols[1] = removed_to_survivor[cols[1]]
            lines[i] = "\t".join(cols)
    MAPPING.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="De-duplicate cross-listed papers.")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    print(f"=== {len(PAIRS)} duplicate pairs "
          f"[{'EXECUTE' if args.execute else 'DRY-RUN'}] ===\n")
    removed_to_survivor: dict[str, str] = {}
    actions = []
    for final, content, redundant in PAIRS:
        if content == final:
            actions.append(("delete", redundant, final))
            removed_to_survivor[redundant] = final
            print(f"  keep {final:30} | delete {redundant}")
        else:
            # delete the (cleaner-name but poorer-content) folder, then
            # rename the richer-content folder to the final citekey.
            actions.append(("replace", redundant, final, content))
            removed_to_survivor[redundant] = final
            print(f"  keep {final:30} | content<-{content} | delete {redundant}")

    if not args.execute:
        print("\n(DRY-RUN — re-run with --execute, then run gen_index.py)")
        return

    for a in actions:
        if a[0] == "delete":
            shutil.rmtree(PAPERS / a[1])
        else:  # replace
            _, redundant, final, content = a
            shutil.rmtree(PAPERS / redundant)   # remove poorer-content folder (== final name)
            rename_folder(content, final)        # promote richer folder to final name
    update_mapping(removed_to_survivor)
    print(f"\nRemoved {len(PAIRS)} duplicate folders + repointed mapping. "
          f"Now run: python3 scripts/gen_index.py")


if __name__ == "__main__":
    main()
