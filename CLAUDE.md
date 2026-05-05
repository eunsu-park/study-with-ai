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
- Include diagrams and equations where appropriate (LaTeX: `$...$`)
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
