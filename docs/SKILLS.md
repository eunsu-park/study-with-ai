# Custom Skills Reference / 커스텀 Skills 레퍼런스

This document describes the custom Claude Code skills created for the Study with AI project.
이 문서는 Study with AI 프로젝트를 위해 생성된 커스텀 Claude Code skills를 설명합니다.

**Location / 위치**: `.claude/skills/<skill-name>/SKILL.md`
**Note / 참고**: `.claude/` is gitignored — skills are local-only and not version controlled.
`.claude/`는 gitignore 처리되어 있으므로 skills는 로컬에서만 관리됩니다.

---

## Overview / 개요

| Skill | Command / 명령어 | Purpose / 목적 |
|---|---|---|
| study | `/study` | Main study workflow — full paper cycle / 메인 학습 워크플로우 — 전체 논문 사이클 |
| pre-read | `/pre-read` | Pre-reading briefing for a paper / 논문 사전 읽기 브리핑 |
| write-notes | `/write-notes` | Write notes.md after reading / 읽기 후 notes.md 작성 |
| implement | `/implement` | Write implementation.ipynb / implementation.ipynb 작성 |
| update-progress | `/update-progress` | Update all tracking files / 모든 추적 파일 업데이트 |
| new-topic | `/new-topic` | Scaffold a new study topic / 새 학습 주제 스캐폴딩 |
| drop | `/drop` | Scan inbox PDFs, classify by topic, add to reading list / 인박스 PDF 스캔, 토픽 분류, 리딩 리스트 추가 |

---

## Skill Details / Skills 상세

### 1. `/study` — Main Study Workflow / 메인 학습 워크플로우

**When to use / 사용 시점**: 논문 학습을 시작하거나 이어서 진행할 때. "진행"이라고 말해도 동일하게 작동합니다.

**What it does / 동작**:
1. 각 토픽의 `reading_list.md`에서 다음 미읽은 논문(`[ ]`) 식별
   Identifies the next unread paper from each topic's `reading_list.md`
2. 여러 토픽에 미읽은 논문이 있으면 사용자에게 선택 요청
   If multiple topics have unread papers, asks user to choose
3. PDF 자동 다운로드 (Unpaywall API, publisher open access 등)
   Downloads PDF automatically
4. 사전 읽기 브리핑 제공 (핵심 기여, 역사적 맥락, 배경지식, 핵심 용어, 수식 미리보기)
   Provides pre-reading briefing
5. 사용자 읽기 대기 및 질문 답변
   Waits for user to read, answers questions
6. `notes.md` 작성 → `implementation.ipynb` 작성 → 진행 추적 파일 업데이트
   Writes notes → implementation → updates tracking

**Usage / 사용법**:
```
/study
/study AI 10
진행
```

---

### 2. `/pre-read` — Pre-Reading Briefing / 사전 읽기 브리핑

**When to use / 사용 시점**: 논문을 읽기 전에 배경지식과 맥락을 파악하고 싶을 때.

**What it does / 동작**:
- **핵심 기여 / Core contribution** — 한 단락 요약
- **역사적 맥락 / Historical context** — 이 논문이 분야 타임라인에서 차지하는 위치
- **필요한 배경지식 / Prerequisites** — 수학, 개념, 선행 논문
- **핵심 용어 / Key vocabulary** — 8-12개 용어 + 직관적 설명
- **수식 미리보기 / Equations preview** — 3-5개 핵심 수식 + 설명
- **읽기 가이드 / Reading guide** — 집중해야 할 섹션, 읽는 순서 추천

**Usage / 사용법**:
```
/pre-read AI 10
/pre-read Solar Physics 5
/pre-read Space Weather 1
```

---

### 3. `/write-notes` — Write Notes / 노트 작성

**When to use / 사용 시점**: 논문을 다 읽은 후 구조화된 노트를 작성할 때.

**What it does / 동작**:
`<Topic>/papers/<NN>_<author>_<year>/notes.md` 파일을 생성합니다.

**Notes structure / 노트 구조**:
1. YAML frontmatter (title, authors, year, journal, tags, status, dates)
2. Core Contribution / 핵심 기여 (한 단락 요약)
3. Reading Notes / 읽기 노트 (섹션별, 핵심 수식 포함)
4. Key Takeaways / 핵심 시사점 (6-8개, 각각 굵은 주장으로 시작)
5. Mathematical Summary / 수학적 요약
6. Paper in the Arc of History / 역사 타임라인 (ASCII)
7. Connections to Other Papers / 다른 논문과의 연결 (테이블)
8. References / 참고문헌

**All content is bilingual (한/영 병기).**

**Usage / 사용법**:
```
/write-notes
```

---

### 4. `/implement` — Write Implementation / 구현 노트북 작성

**When to use / 사용 시점**: 노트 작성 후 논문의 핵심 알고리즘을 코드로 구현할 때.

**What it does / 동작**:
`<Topic>/papers/<NN>_<author>_<year>/implementation.ipynb` 파일을 생성합니다.

**Implementation philosophy / 구현 철학**:
- Part 1: NumPy만으로 핵심 알고리즘 직접 구현 (라이브러리 없이)
  Build core concept from scratch with NumPy only
- Parts 2-4: matplotlib로 핵심 아이디어 시각화 및 수학 시연
  Visualize key ideas and demonstrate math
- Parts 5-6: 논문 실험 재현 (필요 시 축소)
  Reproduce paper's experiments (scaled down)
- Part 7: scikit-learn/PyTorch 등 현대 라이브러리와 비교
  Compare with modern library equivalents
- Final: 다음 논문과의 연결 요약 테이블
  Summary table connecting to next papers

**Skip conditions / 생략 조건**:
- 순수 철학 논문 (예: Turing 1950) / Purely philosophical papers
- 리뷰 논문 (새로운 알고리즘 없음) / Review papers with no novel algorithm

**Usage / 사용법**:
```
/implement
```

---

### 5. `/update-progress` — Update Progress / 진행 업데이트

**When to use / 사용 시점**: 논문 학습을 완료한 후 추적 파일들을 업데이트할 때.

**What it does / 동작**:
세 곳의 파일을 동시에 업데이트합니다:
Updates three files simultaneously:

1. **`<Topic>/papers/reading_list.md`** — `[ ]` → `[x]`로 변경
2. **`README.MD`** (root) — 논문 수 증가 및 테이블 업데이트
3. **`docs/WORKFLOW.md`** — 진행 수 업데이트 및 "다음 논문" 변경

**Usage / 사용법**:
```
/update-progress
```

---

### 6. `/new-topic` — Scaffold New Topic / 새 토픽 스캐폴딩

**When to use / 사용 시점**: 프로젝트에 새로운 학습 주제를 추가할 때.

**What it does / 동작**:
1. 디렉토리 구조 생성: `papers/`, `notes/`, `notebooks/`, `scripts/`, `data/`
   Creates directory structure
2. `README.MD` 작성 (개요, 학습 로드맵, 디렉토리 구조)
   Writes topic README
3. `papers/reading_list.md` 큐레이션 (20-30편, 시간순, 한/영 병기)
   Curates chronological reading list
4. 프로젝트 루트 파일 업데이트 (`README.MD`, `docs/WORKFLOW.md`)
   Updates project root files

**Usage / 사용법**:
```
/new-topic Quantum Computing
/new-topic Plasma Physics
```

---

### 7. `/drop` — Paper Drop & Classify / 논문 드롭 및 분류

**When to use / 사용 시점**: 읽고 싶은 논문 PDF를 `inbox/` 폴더에 넣은 후 자동 분류 및 등록을 할 때.

**What it does / 동작**:
1. `inbox/` 폴더에서 PDF 파일을 스캔
   Scans for PDF files in `inbox/`
2. 각 PDF를 읽어 제목, 저자, 연도, 주제를 분석
   Reads each PDF to analyze title, authors, year, subject
3. 기존 5개 토픽과 비교하여 토픽 분류 → 사용자 확인
   Classifies into existing topics → user confirmation
4. 기존 토픽에 맞지 않으면 `/new-topic` 자동 실행
   Auto-invokes `/new-topic` if no existing topic matches
5. 중복 여부 확인 후 `reading_list.md`에 "User-Added Papers" 섹션으로 추가
   Checks for duplicates, adds to reading list under "User-Added Papers" section
6. PDF를 올바른 디렉토리로 이동 및 이름 변경
   Moves and renames PDF to correct paper directory
7. 완료 후 `/study`로 체이닝 (사용자 선택)
   Chains to `/study` after completion (user's choice)

**Usage / 사용법**:
```
# 1. inbox에 PDF 넣기 / Place PDF in inbox
# 2. 스킬 실행 / Run skill
/drop
```

---

## Recreating Skills / Skills 재생성

Skills are stored locally in `.claude/skills/` and are not version controlled. If lost, they can be recreated from this document or by asking Claude to regenerate them.

Skills는 `.claude/skills/`에 로컬로 저장되며 버전 관리되지 않습니다. 분실 시 이 문서를 참고하거나 Claude에게 재생성을 요청할 수 있습니다.

```bash
# Verify skills are installed / Skills 설치 확인
ls .claude/skills/
# Expected: drop/ implement/ new-topic/ pre-read/ study/ update-progress/ write-notes/
```
