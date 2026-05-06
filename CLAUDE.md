# CLAUDE.md

# currentDate
Today's date is 2026-04-02.

# Project Overview
A personal study project for learning independent subjects with AI assistance.
Combines learning notes (Markdown), hands-on code (Jupyter Notebook), and curated paper reading.

| Topic                              | Level                | Status          |
|------------------------------------|----------------------|-----------------|
| Artificial Intelligence            | Beginner → Expert    | **Active**      |
| Solar Physics                      | Beginner → Expert    | **Active**      |
| Space Weather                      | Beginner → Expert    | **Active**      |
| Solar Observation                  | Beginner → Expert    | **Active**      |
| Living Reviews in Solar Physics    | Beginner → Expert    | **Active**      |
| Low-SNR Imaging                    | Beginner → Expert    | **Active**      |
| Helioseismology & Asteroseismology | Beginner → Expert    | **Active**      |
| Magnetic Reconnection & Eruption   | Beginner → Expert    | **Active**      |
| Heliosphere & Solar Wind           | Beginner → Expert    | **Active**      |
| Plasma Spectroscopy & Diagnostics  | Beginner → Expert    | **Active**      |
| Numerical MHD Simulation           | Beginner → Expert    | **Active**      |

# Language Rules
- **Conversation**: Always respond in Korean (한국어)
- **Documents**: Bilingual Korean/English (한/영 병기)
- **Code comments & docstrings**: Google-style English
- **Technical terms**: Use original English terms (do not translate technical jargon)
- **Commit messages**: English

## Bilingual Document Policy (Mandatory)
All Markdown documents (briefing.md, notes.md, etc.) and Jupyter notebook Markdown cells MUST be fully bilingual (한/영 병기). This means:

1. **Section titles**: Must include both Korean and English (e.g., `## 핵심 기여 / Core Contribution`)
2. **Body text**: Every substantive paragraph must have both a Korean version and an English version — not just one language with occasional translations
3. **Tables**: Headers and content should be bilingual where practical
4. **Lists**: Each item should include both Korean and English explanations
5. **Code comments & docstrings**: Exception — these remain English-only (Google style)

A document is **non-compliant** if any section has body text in only one language. When writing or reviewing documents, verify bilingual coverage for every section before considering the document complete.

## Math Notation Policy (Mandatory)
All LaTeX equations in Markdown documents and Jupyter notebook Markdown cells MUST use the dollar-sign delimiters. This is the only style that renders consistently across **VSCode preview, Obsidian, and GitHub**:

- **Inline math**: `$...$` (no spaces inside the delimiters: `$x = 1$`, not `$ x = 1 $`)
- **Block math**: `$$...$$` (each `$$` on its own line, with a blank line before/after)

**Forbidden delimiters** (do not render in Obsidian or GitHub):
- `\(...\)` — inline LaTeX style (use `$...$` instead)
- `\[...\]` — block LaTeX style (use `$$...$$` instead)

Notes:
- LaTeX commands inside math blocks (`\frac`, `\sum`, `\sqrt`, `\boldsymbol`, `\\` line break with `[Npt]` argument, etc.) are unaffected — only the *outer* delimiters are constrained.
- Markdown-escaped square brackets in citation references like `\[15\]` are NOT math and should be left alone.
- The repair script `scripts/fix_latex_delimiters.py` (dry-run by default; `--execute` to apply) auto-converts forbidden delimiters and is safe to re-run any time.

# Pre-Work Verification Policy (Mandatory)

**Before any paper-related work proceeds — briefing, reading-support Q&A, notes, implementation, publishing, or progress updates — you MUST first verify the paper.** No exceptions. If any check fails, STOP and surface the discrepancy to the user before doing anything else. Never write a briefing/notes/code based on assumed-but-unverified paper identity.

## Quick gate (always run first)
```bash
python3 scripts/verify_paper.py <topic_alias> <number>
```
This runs the mechanical checks (paper directory exists, PDF exists, size ≥ 100 KB, valid `%PDF` magic, bibliography entry present) and prints a JSON report. Exit code 0 = mechanical checks passed; 1 = a fatal error (do NOT proceed).

The script intentionally does NOT do the visual identity check — that part requires reading the PDF and comparing to the reading list, which Claude does after the mechanical gate passes (see Component 1 below).

The verification has three components, and all three must pass:

## 1. Paper Identity Verification / 논문 신원 검증
Confirm that the paper you are about to work on is the paper the user intends.
- Run `python3 scripts/reading_list.py info <topic_alias> <number>` to retrieve the canonical title, authors, year, and DOI from the curated reading list.
- Read the **first page** of the downloaded PDF (`<paper_dir>/<paper_name>_paper.pdf`) and extract the actual title and author list.
- Compare:
  - **Title**: PDF title must match (allowing for minor punctuation/casing differences) the reading list title.
  - **First author**: PDF first-author surname must match the directory naming (`{NN}_{surname}_{year}`).
  - **Year**: PDF publication year must match the directory year and the reading list year.
- If any field disagrees, STOP — the wrong PDF may have been downloaded, the reading list entry may be incorrect, or the directory may be misnamed. Report the mismatch and ask the user how to resolve before proceeding.

## 2. Bibliographic Information Verification / 서지정보 검증
Confirm the bibliographic record is complete and consistent.
- Required fields: title, authors (full list), year, journal/venue, DOI (when one exists).
- If a DOI is present in the reading list, run `python3 scripts/bibtex.py lookup <doi>` to fetch the canonical record from CrossRef. Compare against the reading list entry. Discrepancies (wrong year, wrong author order, mistyped journal) must be flagged and corrected in `reading_list.md` BEFORE downstream work.
- If no DOI is available (older papers, books, archived reports), document the source explicitly in the reading list entry and verify the citation against the PDF's own first/last page.
- The per-topic `bibliography.bib` must contain a valid BibTeX entry for the paper. If missing, run `python3 scripts/bibtex.py generate <topic>` to regenerate.

## 3. Downloaded File Verification / 다운로드 파일 검증
Confirm the PDF is the actual paper, not a placeholder, error page, or wrong file.
- File exists at `<paper_dir>/<paper_name>_paper.pdf`.
- File size is reasonable (a usable scientific paper PDF is typically ≥ 100 KB; a tiny file likely is an error page from the publisher).
- File opens as a valid PDF (`file <pdf>` should report `PDF document`; `pdfinfo <pdf>` or equivalent should produce metadata without error).
- First-page text extraction yields readable content (not garbled bytes from an HTML-paywall page saved with `.pdf` extension).
- If the file fails any check, do NOT proceed. Re-download via `python3 scripts/pdf_download.py <doi_or_url> <output_path>` or alert the user that manual acquisition is needed.

## When this policy applies
- **/study, /pre-read**: verify before producing the briefing.
- **/write-notes, /implement**: verify before producing notes or code (the previous skill's verification does NOT count — re-confirm).
- **/drop**: when accepting an inbox PDF, perform identity + file verification before deciding the destination.
- **/publish, /update-progress**: verify the paper still exists and bibliographic info is current before publishing or marking progress.
- **Any direct paper question from the user** ("explain Eq. 7 in paper X", "what does Y conclude?"): verify before answering, so the answer references the correct paper.

This policy supersedes any conflicting instruction in a skill file. If a skill description omits verification, treat verification as an implicit Step 0.

# Directory Structure

Project-level structure:
```
StudyWithAI/
├── CLAUDE.md              # Project rules (this file)
├── README.MD              # Progress tracker
├── docs/
│   ├── WORKFLOW.md        # Detailed workflow documentation
│   ├── IMPROVEMENTS.md    # Project improvement ideas
│   └── MCP_SETUP.md       # MCP server recommendations & setup
├── scripts/               # Shared utility scripts (used by all skills)
│   ├── reading_list.py    # Reading list parse/update CLI
│   ├── paper_dir.py       # Paper directory naming/creation CLI
│   ├── progress.py        # 3-file progress sync CLI
│   ├── pdf_download.py    # PDF download via Unpaywall/arXiv
│   └── templates/         # Document structure templates
├── inbox/                 # PDF drop zone for /drop skill
├── Artificial_Intelligence/
├── Solar_Physics/
├── Space_Weather/
└── Solar_Observation/
```

Each topic directory follows this internal structure:

```
<Topic>/
├── papers/          # Paper reading list and notes
│   ├── reading_list.md   # Curated paper list with status tracking
│   ├── bibliography.bib  # Auto-generated BibTeX file (per topic)
│   └── <paper_name>/     # Per-paper directory (e.g., 01_mcculloch_1943/)
│       ├── <paper_name>_briefing.md        # Pre-reading briefing & Q&A
│       ├── <paper_name>_notes.md           # Reading notes and key insights
│       ├── <paper_name>_implementation.ipynb # Code implementation (if applicable)
│       ├── <paper_name>_paper.pdf          # Downloaded paper PDF
│       └── archive/      # Previous reading cycle's work (if restarting)
├── notes/           # Markdown study notes (theory, concepts)
├── notebooks/       # Jupyter notebooks (practice, analysis)
├── scripts/         # Standalone Python/IDL scripts
├── data/            # Sample data files
└── README.md        # Topic overview and learning roadmap
```

# Content Format
- **Theory & Concepts**: Markdown files (.md) in `notes/`
- **Practice & Analysis**: Jupyter Notebooks (.ipynb) in `notebooks/`
- **Utility code**: Python scripts (.py) or IDL scripts (.pro) in `scripts/`
- **Paper notes**: Markdown + Jupyter in `papers/<paper_name>/`

# Shared Utilities (scripts/)
All skills share common Python CLI tools in `scripts/`. Use these instead of inline parsing:

| Script | Purpose | Key Commands |
|--------|---------|-------------|
| `reading_list.py` | Parse/update reading lists | `next`, `count`, `info`, `highest`, `mark`, `add`, `topics` |
| `paper_dir.py` | Directory naming/creation | `name`, `files`, `create`, `archive` |
| `progress.py` | 3-file progress sync | `update`, `status`, `verify` |
| `pdf_download.py` | PDF download | `<doi_or_url> <output_path>` |
| `bibtex.py` | BibTeX management | `generate [<topic>\|--all]`, `lookup <doi>`, `verify <topic>` |
| `verify_paper.py` | Pre-work verification gate | `<topic_alias> <number>` (returns JSON, exit 0 = pass) |
| `fix_latex_delimiters.py` | LaTeX delimiter repair | `[--execute] [--diff] [paths...]` (default: dry-run) |
| `migrate_dirs.py` | Directory dedup/rename | `[--execute]` (default: dry-run) |

All scripts output JSON for easy parsing. Topic aliases: `AI`, `SP`, `SW`, `SO`, `LRSP`.

### Paper Directory Naming Convention
Format: `{number:02d}_{first_author_surname}_{year}` (e.g., `06_rumelhart_1986`)
- Always use **first author only** (never multi-author names)
- Suffixes (Jr., Sr., III) are removed: "Sheeley Jr." → `sheeley`
- Prefixes (Van, von, de) are preserved: "Van Allen" → `van_allen`
- Use `paper_dir.py` for all directory name generation (single source of truth)
Templates in `scripts/templates/` define standard document structures (briefing, notes, implementation).

# Paper Curation Workflow
Claude curates essential papers for each topic and supports the full reading cycle:

## 1. Curate
- Select historically important and foundational papers
- Organize chronologically into a reading list (`papers/reading_list.md`)
- Each entry includes: title, authors, year, why it matters, prerequisites

## 2. Prepare
Before the user reads each paper, provide:
- **Prerequisites**: Background knowledge needed (math, concepts, prior papers)
- **Context**: Historical significance and what problem the paper solves
- **Key vocabulary**: Important terms and notation used in the paper

## 3. Support
While the user reads:
- Answer questions about the paper
- Explain difficult sections, proofs, or derivations
- Clarify notation and terminology

## 4. Implement
After reading:
- Help implement key algorithms or models from the paper in code
- Create Jupyter notebooks reproducing key results or experiments
- Connect the paper's ideas to modern implementations

## 5. Track
- Update `reading_list.md` with status: `[ ]` not started, `[~]` in progress, `[x]` completed
- Update `README.MD` (project root) — paper count and progress table
- Update `docs/WORKFLOW.md` — progress count
- Maintain notes summarizing key takeaways per paper

# Archive Convention
When restarting a topic from paper #1, existing `notes.md` and `implementation.ipynb` are moved to `<paper_dir>/archive/`. PDFs stay in place. This preserves previous work for reference.

# Study Note Conventions
- Start each note with a YAML-style header: title, date, topic, tags
- Use clear heading hierarchy (H1 for title, H2 for sections, H3 for subsections)
- Include diagrams and equations where appropriate (LaTeX delimiters: `$...$` inline, `$$...$$` block — see Math Notation Policy)
- Add a "Key Takeaways" section at the end
- Include references at the bottom
- All documents are bilingual (Korean/English)

# Notes Quality Standard (Mandatory)
Every `notes.md` file must pass the following quality checklist before being considered complete. When writing or reviewing notes, verify each criterion and fix any deficiencies.

## Required Sections
Every notes.md must include ALL of these sections in order:
1. **Core Contribution / 핵심 기여** — bilingual abstract (1-2 paragraphs each language)
2. **Reading Notes / 읽기 노트** — section-by-section paper walkthrough with page references
3. **Key Takeaways / 핵심 시사점** — 6-8 numbered insights, each with bilingual explanation
4. **Mathematical Summary / 수학적 요약** — all key equations in LaTeX with term-by-term interpretation
5. **Paper in the Arc of History / 역사 속의 논문** — ASCII timeline
6. **Connections to Other Papers / 다른 논문과의 연결** — table linking to series papers
7. **References / 참고문헌**

## Depth Criteria (the "Standalone Test")
The notes must be detailed enough that **someone who has NOT read the original paper** could:
- Understand the paper's core argument and why it matters
- Follow the methodology step by step
- Reproduce key equations from the derivation shown
- Explain the main results with specific numbers
- Place the paper in historical context

## Minimum Detail Requirements
| Criterion | Requirement |
|-----------|------------|
| **Length** | Minimum 350 lines; papers >15 pages should produce 400+ lines |
| **Equations** | Every key equation reproduced in LaTeX with each variable/term explained |
| **Worked examples** | At least one concrete numerical example, trace, or scenario per paper |
| **Original paper coverage** | Every major section of the original paper addressed (not just cherry-picked highlights) |
| **Quantitative results** | Specific numbers from experiments/observations reproduced (not just "good results") |
| **Bilingual completeness** | Every section has substantive content in BOTH Korean and English |
| **Case studies / Figures** | Key figures described or reproduced as ASCII diagrams; important case studies walked through |

## Quality Verification Process
After completing a notes.md file, perform this checklist:

1. **Section completeness**: Are all 7 required sections present?
2. **Standalone test**: Read the notes from scratch — could you explain this paper to a colleague without referring to the original?
3. **Math verification**: Is every key equation present with term-by-term explanation?
4. **Coverage check**: Compare the notes' section headings against the original paper's table of contents — any major sections missing?
5. **Bilingual check**: Scan every section — does each have substantive Korean AND English content?
6. **Length check**: Is the file at least 350 lines? If the source paper is substantial (>15 pages), is it 400+?
7. **Quantitative check**: Are specific numbers (percentages, measurements, parameters) cited rather than vague descriptions?
8. **Math delimiter check**: Are all equations wrapped in `$...$` (inline) or `$$...$$` (block)? No `\(...\)` or `\[...\]` delimiters? (See Math Notation Policy.)

If any check fails, fix the deficiency before marking the paper as completed.

# Reference Style
```
## References
- Author(s), "Title", Journal/Book, Year. [DOI or URL if available]
```

# Coding Conventions
## Python
- Follow PEP 8 style guide
- Use type hints for function signatures
- Add docstrings (Google style, English)
- Preferred libraries: NumPy, SciPy, Matplotlib, PyTorch, scikit-learn
- Use virtual environment (venv or conda)

## IDL
- Use meaningful variable names (avoid single-letter names except loop indices)
- Add header comments to procedures and functions

## Jupyter Notebooks
- Start with a title cell (Markdown H1) and brief description
- Organize: imports → data loading → processing → visualization
- Add Markdown cells between code cells to explain each step
- Clear all outputs before committing

# Progress Tracking
Whenever a paper or study unit is completed, update **all three**:
1. `<Topic>/papers/reading_list.md` — mark the paper status (`[x]` completed)
2. `README.MD` (project root) — update the "Current Progress" section (paper count and table)
3. `docs/WORKFLOW.md` — update progress count

This ensures the project README always reflects the latest progress.

# MCP Servers
See `docs/MCP_SETUP.md` for recommended MCP servers and setup instructions.

# Teaching Style
- Start from fundamentals and progressively advance to expert level
- Use analogies and visual explanations when possible
- Provide step-by-step derivations for equations
- Suggest exercises or practice problems after key concepts
- Connect theory to practical applications
- When explaining papers: summarize the core contribution in one paragraph first, then dive into details
