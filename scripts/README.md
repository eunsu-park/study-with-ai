# Shared Utility Scripts / 공유 유틸리티 스크립트

Common CLI tools used by all study skills. All scripts output JSON.

모든 학습 스킬에서 사용하는 공통 CLI 도구입니다. 모든 스크립트는 JSON을 출력합니다.

## Topic Aliases / 토픽 별칭

| Alias | Full Name |
|-------|-----------|
| `AI` | `Artificial_Intelligence` |
| `SP` | `Solar_Physics` |
| `SW` | `Space_Weather` |
| `SO` | `Solar_Observation` |
| `LRSP` | `Living_Reviews_in_Solar_Physics` |

---

## reading_list.py

Parse and update `reading_list.md` files.

```bash
# List all topics with reading lists
python3 scripts/reading_list.py topics

# Find next unread paper
python3 scripts/reading_list.py next AI

# Count papers by status
python3 scripts/reading_list.py count SW

# Get full metadata for a specific paper
python3 scripts/reading_list.py info SP 14

# Get highest paper number
python3 scripts/reading_list.py highest SO

# Mark a paper as completed
python3 scripts/reading_list.py mark AI 14 x      # x=done, ~=in progress, ' '=not started

# Add a new paper
python3 scripts/reading_list.py add AI \
  --title "Paper Title" \
  --authors "Author One, Author Two" \
  --year 2024 \
  --why "Why it matters (bilingual)" \
  --prereqs "Prerequisites (bilingual)"
```

---

## paper_dir.py

Paper directory naming and management.

```bash
# Generate directory name
python3 scripts/paper_dir.py name 14 "Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean" 2013
# → {"dir_name": "14_mikolov_2013"}

# Get standard file names
python3 scripts/paper_dir.py files 14_mikolov_2013
# → {"pdf": "..._paper.pdf", "briefing": "..._briefing.md", ...}

# Create directory under a topic
python3 scripts/paper_dir.py create AI 14 "Tomas Mikolov, Kai Chen" 2013

# Archive notes and implementation to archive/
python3 scripts/paper_dir.py archive path/to/paper_dir
```

---

## progress.py

Synchronize progress across reading_list.md, README.MD, and WORKFLOW.md.

```bash
# Update all 3 files after completing a paper
python3 scripts/progress.py update AI 14

# Show progress for all topics
python3 scripts/progress.py status

# Verify consistency across files
python3 scripts/progress.py verify
```

---

## pdf_download.py

Download academic paper PDFs via Unpaywall API and fallback strategies.

```bash
# Download by DOI
python3 scripts/pdf_download.py "10.1038/323533a0" path/to/paper.pdf

# Download by direct URL
python3 scripts/pdf_download.py "https://arxiv.org/pdf/1301.3781" path/to/paper.pdf
```

---

## Templates / 템플릿

Located in `scripts/templates/`:

| File | Purpose |
|------|---------|
| `briefing_template.md` | Pre-reading briefing structure (8 sections) |
| `notes_template.md` | Reading notes structure with YAML frontmatter (7 sections) |
| `implementation_template.ipynb` | Jupyter notebook skeleton |

Skills read these templates for structural guidance when generating documents.
