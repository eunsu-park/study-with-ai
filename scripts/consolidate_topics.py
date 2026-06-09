#!/usr/bin/env python3
"""One-shot: consolidate per-topic folders into the flat layout.

Moves each ``<Topic>/papers/reading_list.md`` to ``reading_lists/<tag>.md``,
extracts the Overview + Learning Roadmap from each ``<Topic>/README`` into
``topics/<tag>.md`` (with an AUTO-INDEX marker block that gen_index.py fills),
and relocates the one real cross-topic synthesis note to ``notes/``.

Does NOT delete the topic folders (done separately after verification).
Dry-run by default; ``--execute`` applies.
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# dir_name, tag, title
TOPICS = [
    ("Artificial_Intelligence", "artificial-intelligence", "Artificial Intelligence / 인공지능"),
    ("Solar_Physics", "solar-physics", "Solar Physics / 태양물리학"),
    ("Space_Weather", "space-weather", "Space Weather / 우주기상"),
    ("Solar_Observation", "solar-observation", "Solar Observation / 태양관측"),
    ("Living_Reviews_in_Solar_Physics", "living-reviews-solar-physics", "Living Reviews in Solar Physics / 리빙 리뷰"),
    ("Low_SNR_Imaging", "low-snr-imaging", "Low-SNR Imaging / 저신호대잡음 영상"),
    ("Helioseismology_Asteroseismology", "helioseismology-asteroseismology", "Helioseismology & Asteroseismology / 일진동·성진동학"),
    ("Heliosphere_Solar_Wind", "heliosphere-solar-wind", "Heliosphere & Solar Wind / 태양권·태양풍"),
    ("Magnetic_Reconnection_Eruption", "magnetic-reconnection-eruption", "Magnetic Reconnection & Eruption / 자기재결합·분출"),
    ("Plasma_Spectroscopy_Diagnostics", "plasma-spectroscopy-diagnostics", "Plasma Spectroscopy & Diagnostics / 플라즈마 분광·진단"),
    ("Numerical_MHD_Simulation", "numerical-mhd-simulation", "Numerical MHD Simulation / 수치 MHD 시뮬레이션"),
]

MARKER_START = "<!-- AUTO-INDEX:START -->"
MARKER_END = "<!-- AUTO-INDEX:END -->"


def extract_roadmap(readme: Path) -> str:
    """Return the Overview..Learning-Roadmap slice of a topic README."""
    if not readme.exists():
        return ""
    lines = readme.read_text(encoding="utf-8").splitlines()
    start = end = None
    for i, l in enumerate(lines):
        if start is None and l.startswith("## Overview"):
            start = i
        elif start is not None and l.startswith("## ") and not l.startswith("## Overview") \
                and not l.startswith("## Learning Roadmap"):
            end = i
            break
    if start is None:
        return ""
    body = lines[start:end] if end else lines[start:]
    return "\n".join(body).rstrip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    ex = args.execute

    rl_dir = PROJECT_ROOT / "reading_lists"
    topics_dir = PROJECT_ROOT / "topics"
    notes_dir = PROJECT_ROOT / "notes"
    if ex:
        rl_dir.mkdir(exist_ok=True)
        topics_dir.mkdir(exist_ok=True)
        notes_dir.mkdir(exist_ok=True)

    for dir_name, tag, title in TOPICS:
        tdir = PROJECT_ROOT / dir_name
        # 1) move reading list
        src_rl = tdir / "papers" / "reading_list.md"
        dst_rl = rl_dir / f"{tag}.md"
        print(f"[reading_list] {src_rl.relative_to(PROJECT_ROOT)} -> reading_lists/{tag}.md "
              f"({'ok' if src_rl.exists() else 'MISSING'})")
        if ex and src_rl.exists():
            shutil.move(str(src_rl), str(dst_rl))

        # 2) roadmap -> topics/<tag>.md
        readme = next((p for p in (tdir / "README.md", tdir / "README.MD") if p.exists()), None)
        roadmap = extract_roadmap(readme) if readme else ""
        content = f"# {title} — Topic Map / 주제 지도\n\n{roadmap}\n\n{MARKER_START}\n{MARKER_END}\n"
        print(f"[topic MOC]   topics/{tag}.md  (roadmap {'extracted' if roadmap else 'EMPTY'})")
        if ex:
            (topics_dir / f"{tag}.md").write_text(content, encoding="utf-8")

    # 3) relocate synthesis note
    note = PROJECT_ROOT / "Solar_Physics" / "notes" / "korean_historical_solar_records_series.md"
    print(f"[note] {'move' if note.exists() else 'MISSING'} {note.name} -> notes/")
    if ex and note.exists():
        shutil.move(str(note), str(notes_dir / note.name))

    print("\n" + ("APPLIED." if ex else "DRY-RUN — re-run with --execute"))


if __name__ == "__main__":
    main()
