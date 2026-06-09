#!/usr/bin/env python3
"""Generate the README paper index and per-topic MOC files.

After flattening papers into ``papers/<citekey>/``, this regenerates the
"Current Progress" section of ``README.md`` as topic-grouped tables that link to
each paper's notes/PDF, and writes a Map-of-Content note per topic under
``topics/``.

Data sources (no hand-maintenance needed):
  * ``scripts/flatten_mapping.tsv`` -> (topic, number) -> citekey
  * each ``<Topic>/papers/reading_list.md`` -> authoritative order/status/title

Usage:
    python scripts/gen_index.py            # write README + topics/
    python scripts/gen_index.py --check     # print, do not write
"""

import argparse
from pathlib import Path

from reading_list import parse_reading_list

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAPPING = PROJECT_ROOT / "scripts" / "flatten_mapping.tsv"
README = PROJECT_ROOT / "README.MD"
TOPICS_DIR = PROJECT_ROOT / "topics"

# Display order + bilingual labels + short tag.
TOPICS = [
    ("Artificial_Intelligence", "Artificial Intelligence / 인공지능", "artificial-intelligence"),
    ("Solar_Physics", "Solar Physics / 태양물리학", "solar-physics"),
    ("Space_Weather", "Space Weather / 우주기상", "space-weather"),
    ("Solar_Observation", "Solar Observation / 태양관측", "solar-observation"),
    ("Living_Reviews_in_Solar_Physics", "Living Reviews in Solar Physics / 리빙 리뷰", "living-reviews-solar-physics"),
    ("Low_SNR_Imaging", "Low-SNR Imaging / 저신호대잡음 영상", "low-snr-imaging"),
    ("Helioseismology_Asteroseismology", "Helioseismology & Asteroseismology / 일진동·성진동학", "helioseismology-asteroseismology"),
    ("Magnetic_Reconnection_Eruption", "Magnetic Reconnection & Eruption / 자기재결합·분출", "magnetic-reconnection-eruption"),
    ("Heliosphere_Solar_Wind", "Heliosphere & Solar Wind / 태양권·태양풍", "heliosphere-solar-wind"),
    ("Plasma_Spectroscopy_Diagnostics", "Plasma Spectroscopy & Diagnostics / 플라즈마 분광·진단", "plasma-spectroscopy-diagnostics"),
    ("Numerical_MHD_Simulation", "Numerical MHD Simulation / 수치 MHD 시뮬레이션", "numerical-mhd-simulation"),
]

SECTION_HEADER = "## Current Progress / 현재 진행 상황"
SECTION_END_NEXT = "## Language"  # next top-level heading after the progress block


def load_mapping() -> dict[tuple[str, int], str]:
    """Load (topic, number) -> citekey from the mapping TSV."""
    m: dict[tuple[str, int], str] = {}
    if not MAPPING.exists():
        return m
    for line in MAPPING.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        _old, citekey, topic, number = parts[0], parts[1], parts[2], parts[3]
        try:
            m[(topic, int(number))] = citekey
        except ValueError:
            continue
    return m


import re

# "(migrated from Solar_Physics #13)" -> (source topic dir, source number)
_MIGRATED_RE = re.compile(r"migrated from ([A-Za-z_]+)\s*#(\d+)")


def resolve_citekey(topic_name: str, entry: dict, mapping: dict) -> tuple[str | None, bool]:
    """Resolve a reading-list entry to a paper citekey.

    Phase B topics list papers that physically live under their original
    topic's citekey; follow the "(migrated from <Topic> #N)" marker to find it.

    Returns:
        (citekey or None, is_crossref) where is_crossref means the paper lives
        under a different topic.
    """
    direct = mapping.get((topic_name, entry["number"]))
    if direct:
        return direct, False
    m = _MIGRATED_RE.search(entry.get("raw_status", ""))
    if m:
        src = mapping.get((m.group(1), int(m.group(2))))
        if src:
            return src, True
    return None, False


def is_done(entry: dict) -> bool:
    """Whether an entry counts as completed (incl. ✅ migrated markers)."""
    return entry["status"] == "x" or entry.get("raw_status", "").lstrip().startswith("✅")


def status_label(entry: dict) -> str:
    """Render a status cell."""
    raw = entry.get("raw_status", "").lstrip()
    if entry["status"] == "x":
        return "✅"
    if raw.startswith("✅"):
        return "↗️ migrated"
    if entry["status"] == "~":
        return "🔄"
    return "⬜"


def paper_links(citekey: str | None, prefix: str = "") -> str:
    """Build notes/PDF links for a paper folder, if files exist.

    Args:
        citekey: Paper citekey, or None.
        prefix: Path prefix to prepend (e.g. "../" for files under topics/).
    """
    if not citekey:
        return "—"
    folder = PROJECT_ROOT / "papers" / citekey
    links = []
    notes = folder / f"{citekey}_notes.md"
    pdf = folder / f"{citekey}_paper.pdf"
    impl = folder / f"{citekey}_implementation.ipynb"
    if notes.exists():
        links.append(f"[📝 notes]({prefix}papers/{citekey}/{citekey}_notes.md)")
    if impl.exists():
        links.append(f"[💻 code]({prefix}papers/{citekey}/{citekey}_implementation.ipynb)")
    if pdf.exists():
        links.append(f"[📄 pdf]({prefix}papers/{citekey}/{citekey}_paper.pdf)")
    return " · ".join(links) if links else "—"


def topic_rows(topic_name: str, mapping: dict, prefix: str = "",
               tag: str | None = None) -> tuple[list[str], int, int]:
    """Build table rows for one topic; return (rows, done, total).

    Args:
        topic_name: Topic directory name (the mapping/cross-ref join key).
        mapping: (topic, number) -> citekey.
        prefix: Path prefix for links (e.g. "../" for topics/ MOCs).
        tag: Topic tag; reading list is read from reading_lists/<tag>.md.
    """
    rl = PROJECT_ROOT / "reading_lists" / f"{tag}.md"
    if not rl.exists():
        return [], 0, 0
    entries = parse_reading_list(rl)
    rows = []
    done = 0
    for e in entries:
        if is_done(e):
            done += 1
        citekey, crossref = resolve_citekey(topic_name, e, mapping)
        title = e["title"].replace("|", "\\|")
        if crossref:
            title += " 🔗"
        year = e["year"] or ""
        rows.append(
            f"| {e['number']} | {title} | {year} | {status_label(e)} | {paper_links(citekey, prefix)} |"
        )
    return rows, done, len(entries)


def build_progress_section(mapping: dict) -> tuple[str, dict]:
    """Build the full Current-Progress markdown block + per-topic rows cache."""
    out = [SECTION_HEADER, ""]
    grand_done = grand_total = 0
    cache: dict[str, list[str]] = {}

    # Summary table first
    out.append("| Topic / 주제 | Progress / 진행 |")
    out.append("|---|---|")
    summaries = []
    for topic_name, label, tag in TOPICS:
        rows, done, total = topic_rows(topic_name, mapping, tag=tag)
        cache[topic_name] = rows
        grand_done += done
        grand_total += total
        summaries.append((label, done, total, tag))
        out.append(f"| [{label}](#{tag}) | {done} / {total} |")
    out.append(f"| **Total / 합계** | **{grand_done} / {grand_total}** |")
    out.append("")

    # Per-topic detail tables (explicit anchors for stable cross-links)
    for (topic_name, label, tag), (lbl, done, total, _tag) in zip(TOPICS, summaries):
        rows = cache[topic_name]
        out.append(f'<a id="{tag}"></a>')
        out.append("")
        out.append(f"### {label} ({done} / {total})")
        out.append("")
        if not rows:
            out.append("_No papers migrated yet. / 아직 이관된 논문이 없습니다._")
            out.append("")
            continue
        out.append("| # | Paper / 논문 | Year | Status | Links |")
        out.append("|---|---|---|---|---|")
        out.extend(rows)
        out.append("")

    return "\n".join(out), cache


def splice_readme(progress_md: str) -> str:
    """Replace the Current-Progress block in README, preserving the rest."""
    text = README.read_text(encoding="utf-8")
    start = text.index(SECTION_HEADER)
    end = text.index(f"\n{SECTION_END_NEXT}", start)
    return text[:start] + progress_md + "\n" + text[end + 1:]


MOC_START = "<!-- AUTO-INDEX:START -->"
MOC_END = "<!-- AUTO-INDEX:END -->"


def write_topic_mocs(mapping: dict) -> int:
    """Refresh the auto-index block of each topics/<tag>.md, keeping the
    hand-written roadmap above the markers. Returns count written."""
    TOPICS_DIR.mkdir(exist_ok=True)
    n = 0
    for topic_name, label, tag in TOPICS:
        rows, done, total = topic_rows(topic_name, mapping, prefix="../", tag=tag)
        block = [
            f"**Progress / 진행**: {done} / {total}  ·  "
            f"Source / 원본: [`reading_lists/{tag}.md`](../reading_lists/{tag}.md)",
            "",
        ]
        if rows:
            block.append("| # | Paper / 논문 | Year | Status | Links |")
            block.append("|---|---|---|---|---|")
            block.extend(rows)
        else:
            block.append("_No papers yet. / 아직 논문이 없습니다._")
        index_md = MOC_START + "\n" + "\n".join(block) + "\n" + MOC_END

        path = TOPICS_DIR / f"{tag}.md"
        if path.exists() and MOC_START in path.read_text(encoding="utf-8"):
            text = path.read_text(encoding="utf-8")
            pre = text[: text.index(MOC_START)]
            post = text[text.index(MOC_END) + len(MOC_END):]
            path.write_text(pre + index_md + post, encoding="utf-8")
        else:
            en, ko = label.split(" / ")
            header = f"# {en} — Topic Map / {ko} 주제 지도\n\n"
            path.write_text(header + index_md + "\n", encoding="utf-8")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate README index + topic MOCs.")
    ap.add_argument("--check", action="store_true", help="print, do not write")
    args = ap.parse_args()

    mapping = load_mapping()
    progress_md, _cache = build_progress_section(mapping)

    if args.check:
        print(progress_md)
        return

    README.write_text(splice_readme(progress_md), encoding="utf-8")
    n = write_topic_mocs(mapping)
    print(f"README.MD progress section regenerated.")
    print(f"{n} topic MOC files written to topics/.")


if __name__ == "__main__":
    main()
