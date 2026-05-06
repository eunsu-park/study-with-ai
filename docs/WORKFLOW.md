# Study Workflow Documentation / 학습 워크플로우 문서

This file documents exactly how this project is conducted between the user and Claude,
so any future session can pick up seamlessly.

이 파일은 사용자와 Claude 사이에서 이 프로젝트가 어떻게 진행되는지 문서화하며,
향후 세션에서 원활하게 이어갈 수 있도록 합니다.

**Last updated / 최종 수정**: 2026-04-25
**Current progress / 현재 진행 상황**:
- Artificial Intelligence: 40 / 69 papers
- Solar Physics: 19 / 60 papers (post-migration; helioseismology/reconnection/heliosphere/diagnostics 이관)
- Space Weather: 81 / 81 papers
- Solar Observation: 61 / 80 papers (post-migration; DEM/atomic data 이관)
- Living Reviews in Solar Physics: 87 / 87 papers
- Low-SNR Imaging: 45 / 45 papers
- Helioseismology & Asteroseismology: 5 / 7 papers (Phase B migrated)
- Magnetic Reconnection & Eruption: 8 / 10 papers (Phase B migrated)
- Heliosphere & Solar Wind: 4 / 6 papers (Phase B migrated)
- Plasma Spectroscopy & Diagnostics: 4 / 20 papers (Phase B migrated)
- Numerical MHD Simulation: 0 / 1 papers (Phase B migrated)

---

## 1. Session Flow (How Each Paper Works) / 세션 흐름 (논문별 진행 방법)

Every paper follows this exact 6-step sequence:
모든 논문은 다음 6단계 순서를 따릅니다:

### Step 1 — Download PDF / PDF 다운로드
Claude downloads the paper PDF automatically using Python/curl.
Claude가 Python/curl을 사용하여 자동으로 논문 PDF를 다운로드합니다.
- Target path / 대상 경로: `<Topic>/papers/<NN>_<author>_<year>/paper.pdf`
- Sources tried in order / 시도 순서: Springer open access, university hosting, Unpaywall API
- The user should NOT need to find or provide PDFs — Claude handles this.
  사용자가 PDF를 찾거나 제공할 필요 없음 — Claude가 처리합니다.

**How Claude finds PDFs / Claude의 PDF 탐색 방법:**
1. Check Unpaywall API: `https://api.unpaywall.org/v2/<DOI>?email=test@test.com`
2. Try publisher open access link from Unpaywall response
3. Try university hosting URLs (known patterns)
4. Download with Python `urllib` using browser-like User-Agent header

### Step 2 — Pre-reading Briefing / 사전 읽기 브리핑
Before the user reads, Claude provides:
사용자가 읽기 전에 Claude가 제공:
- **Core contribution / 핵심 기여** (one paragraph / 한 단락)
- **Historical context / 역사적 맥락** (where this paper fits / 이 논문의 위치)
- **Background knowledge needed / 필요한 배경 지식** (math, concepts, prior papers)
- **Key vocabulary / 핵심 용어** with intuitive explanations / 직관적 설명 포함
- **Equations preview / 수식 미리보기** — the most important formulas introduced early

The briefing and any Q&A during reading are saved to `<paper_dir>/briefing.md` as a Markdown file, so the user can view rendered LaTeX equations in VSCode preview (Cmd+Shift+V). The file is updated incrementally as new Q&A occurs.
브리핑과 읽기 중 Q&A는 `<paper_dir>/briefing.md`에 Markdown 파일로 저장되며, 사용자가 VSCode 미리보기(Cmd+Shift+V)에서 렌더링된 LaTeX 수식을 볼 수 있습니다. 새로운 Q&A가 발생하면 파일이 점진적으로 업데이트됩니다.

### Step 3 — User Reads the Paper / 사용자 논문 읽기
User uploads screenshots of each page.
사용자가 각 페이지의 스크린샷을 업로드합니다.
Claude reads all pages (using the Read tool on the PDF, then processing screenshots).

### Step 4 — Write notes.md / notes.md 작성
After reading, Claude writes a comprehensive `notes.md` file:
읽기 후, Claude가 포괄적인 `notes.md` 파일을 작성합니다:

```
<Topic>/papers/<NN>_<author>_<year>/notes.md
```

**notes.md structure / 구조:**
1. YAML frontmatter (title, authors, year, journal, tags, status, dates)
2. Core Contribution / 핵심 기여 (one-paragraph summary / 한 단락 요약)
3. Reading Notes / 읽기 노트 (section-by-section with key equations / 섹션별, 핵심 수식 포함)
4. Key Takeaways / 핵심 시사점 (numbered list, 6–8 points)
5. Mathematical Summary / 수학적 요약 (complete algorithm in one place)
6. Paper in the Arc of History / 역사적 맥락의 타임라인 (ASCII timeline)
7. Connections to Other Papers / 다른 논문과의 연결 (table)
8. References / 참고문헌

### Step 5 — Write implementation.ipynb / implementation.ipynb 작성
Claude writes a Jupyter notebook:
Claude가 Jupyter 노트북을 작성합니다:

```
<Topic>/papers/<NN>_<author>_<year>/implementation.ipynb
```

**notebook structure / 노트북 구조:**
- Title cell (Markdown H1) + brief description
- Multiple parts (typically 6–8), each with:
  - Markdown explanation cell (with LaTeX equations)
  - Code cell (NumPy/SciPy/scikit-learn/PyTorch)
- Summary table at the end connecting to next papers
- All outputs cleared (no cell output saved)

**Implementation philosophy / 구현 철학:**
- Part 1: Build the core concept from scratch (no libraries) / 라이브러리 없이 핵심 개념 직접 구현
- Parts 2–5: Visualize key ideas, demonstrate math / 핵심 아이디어 시각화, 수학 시연
- Part 6–7: Reproduce the paper's experiments (scaled down) / 논문 실험 재현 (축소)
- Final part: Connection to modern equivalents / 현대 동등 기법과의 연결

### Step 6 — Update Progress Tracking / 진행 상황 업데이트
After completing a paper, update **all three** / 논문 완료 후 **세 곳 모두** 업데이트:
1. `<Topic>/papers/reading_list.md` — change `[ ]` or `[~]` to `[x]`
2. `README.MD` (project root) — increment paper count and update table
3. `docs/WORKFLOW.md` (this file) — update progress count

---

## 2. Current Progress / 현재 진행 상황

### Artificial Intelligence — 40 / 40 papers

All previous work (papers 1–9) has been archived to `archive/` subdirectories within each paper folder. Status reset for a fresh second reading cycle.
이전 작업물(논문 1-9)은 각 논문 폴더 내 `archive/` 하위 디렉토리에 보관되었습니다. 두 번째 읽기 주기를 위해 상태가 리셋되었습니다.

**Completed / 완료**: #1 McCulloch & Pitts (1943), #2 Turing (1950), #3 Rosenblatt (1958), #4 Minsky & Papert (1969), #5 Hopfield (1982), #6 Rumelhart, Hinton & Williams (1986), #7 LeCun et al. (1989), #8 Cortes & Vapnik (1995), #9 Hochreiter & Schmidhuber (1997), #10 LeCun et al. (1998), #11 Breiman (2001), #12 Hinton, Osindero & Teh (2006), #13 Krizhevsky, Sutskever & Hinton (2012), #14 Mikolov (2013), #15 Kingma (2013), #16 Goodfellow (2014), #17 Bahdanau (2014), #18 Kingma (2014), #19 Ioffe (2015), #20 He (2015), #21 Ba (2016), #22 Mnih (2013), #23 Silver (2016), #24 Schulman (2017), #25 Vaswani (2017), #26 Kipf (2017), #27 Zoph (2017), #28 Devlin (2018), #29 Radford (2019), #30 Frankle (2019), #31 al. (2020), #32 Chen (2020), #33 Lewis (2020), #34 al. (2020), #35 Ho (2020), #36 Radford (2021), #37 Jumper (2021), #38 al. (2022), #39 Kaplan (2020), #40 al. (2022)
**Next paper / 다음 논문**: All completed! / 전부 완료!

### Solar Physics — 43 / 43 papers

**Completed / 완료**: #1 Galileo (1613), #2 Fraunhofer (1814), #3 Kirchhoff & Bunsen (1860), #4 Schwarzschild (1906), #5 Hale (1908), #6 Evershed (1909), #7 Hale & Nicholson (1925), #8 Alfvén (1942), #9 Babcock (1961), #10 Leighton (1969), #11 Biermann (1951), #12 Parker (1958), #13 Leighton, Noyes & Simon (1962), #14 Ulrich (1970), #39 Yang (1998), #40 Lee (2004), #41 Lee (2007), #15 Deubner (1975), #42 Yang (2019), #43 Miyake (2012), #16 al. (1996), #17 al. (1998), #18 Giovanelli (1946), #19 Parker (1957), #20 Petschek (1964), #21 Parker (1988), #22 Gosling (1993), #23 Aulanier (1998), #24 Nakariakov (1999), #25 Antiochos (1999), #26 Kliem (2006), #27 Reames (1999), #28 Schrijver (2000), #29 al. (2008), #30 Wijn (2009), #31 Charbonneau (2010), #32 Kopp (2011), #33 Pesnell (2012), #34 al. (2012), #35 Wiegelmann (2012), #36 al. (2012), #37 al. (2016), #38 al. (2021)
**Next paper / 다음 논문**: All completed! / 전부 완료!

### Space Weather — 81 / 81 papers

**Completed / 완료**: #1 Birkeland (1908), #2 Chapman & Ferraro (1931), #3 Chapman & Bartels (1940), #4 Parker (1958), #5 Van Allen et al. (1958), #6 Dungey (1961), #7 Axford & Hines (1961), #8 Akasofu (1964), #9 Ness (1965), #10 McPherron, Russell & Aubry (1973), #11 Burton, McPherron & Russell (1975), #12 Cowley (1982), #13 Richmond & Kamide (1988), #14 Tsyganenko (1989), #15 Gonzalez (1994), #36 Owens (2021), #37 Billcliff (2026), #38 Abduallah (2024), #39 Wang (2026), #16 Fuller-Rowell (1994), #17 Allen (1989), #18 Arge (2000), #19 Baker (2000), #20 Kintner (2007), #21 Odstrcil (2003), #22 Tsurutani (2004), #23 Gopalswamy (2005), #24 Pulkkinen (2007), #25 Kappenman (2010), #40 Lin (2025), #41 Guesmi (2026), #42 Stone (1998 ACE), #43 Ogilvie & Desch (1997 Wind), #44 Burt & Smith (2012 DSCOVR), #45 Goodman (2019 GOES-R), #46 Onsager (1996 GOES EPS), #47 Frank & Craven (1988 DE-1), #48 Anger (1987 Viking), #49 Torr (1995 Polar UVI), #50 Burch (2000 IMAGE), #51 Mende (2000 IMAGE-FUV), #52 Frey (2004 substorm catalog), #53 Paxton (2004 GUVI), #54 Paxton (1992 SSUSI), #55 Carlson (1998 FAST), #56 Mende (2008 THEMIS GBO), #57 Donovan (2006 THEMIS ASI), #58 Greenwald (1995 SuperDARN), #59 Folkestad (1983 EISCAT), #60 Gillies (2019 TREx), #26 Thorne (2010), #27 Angelopoulos (2008), #28 Mauk (2013), #29 al. (2013), #30 Baker (2014), #31 al. (2015), #32 al. (2017), #33 Camporeale (2019), #34 Leka (2019), #35 al. (2020)
**Next paper / 다음 논문**: All completed! / 전부 완료!

### Living Reviews in Solar Physics — 87 / 87 papers

Complete catalog of LRSP journal review articles, read chronologically from 2004 to present.
LRSP 저널 리뷰 논문의 전체 카탈로그를 2004년부터 현재까지 시간순으로 읽습니다.

**Completed / 완료**: #1 Wood (2004), #2 Miesch (2005), #3 Nakariakov & Verwichte (2005), #4 Sheeley (2005), #5 Gizon & Birch (2005), #6 Longcope (2005), #7 Berdyugina (2005), #8 Marsch (2006), #9 Schwenn (2006), #10 Pulkkinen (2007), #11 Haigh (2007), #12 Güdel (2007), #13 Benz (2008), #14 Hall (2008), #15 Howe (2009), #16 Nordlund (2009), #17 Cranmer (2009), #18 Fan (2009), #19 Rieutord (2010), #20 Charbonneau (2010), #21 Ofman (2010), #22 Chen (2011), #23 Rimmele (2011), #24 Rempel (2011), #25 Borrero (2011), #26 Aschwanden (2011), #27 Shibata (2011), #28 Reiners (2012), #29 Webb (2012), #30 Stein (2012), #31 Mackay (2012), #32 Bruno (2013), #33 Potgieter (2013), #34 Lockwood (2013), #35 Owens (2013), #36 Parenti (2014), #37 Penn (2014), #38 Cheung (2014), #39 Reale (2014), #40 Driel-Gesztelyi (2015), #41 Laming (2015), #42 Warmuth (2015), #43 Hathaway (2015), #44 Petrie (2015), #45 Khomenko (2015), #46 Poletto (2015), #47 Houdek (2015), #48 Prieto (2016), #49 Basu (2016), #50 Desai (2016), #51 Iniesta (2016), #52 Usoskin (2017), #53 Brun (2017), #54 Kilpua (2017), #55 Richardson (2018), #56 Narita (2018), #57 Arregui (2018), #58 Gombosi (2018), #59 Zanna (2018), #60 Rincon (2018), #61 Gibson (2018), #62 Rubio (2019), #63 Madjarska (2019), #64 Toriumi (2019), #65 García (2019), #66 Verscharen (2019), #67 Arlt (2020), #68 Petrovay (2020), #69 Pontin (2020), #70 Leenaarts (2020), #71 Charbonneau (2020), #72 Wiegelmann (2021), #73 Christensen-Dalsgaard (2021), #74 Vidotto (2021), #75 Temmer (2021), #76 Fan (2021), #77 Pontin (2022), #78 Cliver (2022), #79 Hanasoge (2022), #80 Jess (2023), #81 Usoskin (2023), #82 Karak (2023), #83 Ramos (2023), #84 Kowalski (2024), #85 Kopp (2025), #86 Veronig (2025), #87 Keppens (2025)
**Next paper / 다음 논문**: All completed! / 전부 완료!

---

## 3. Directory Structure / 디렉토리 구조

```
StudyWithAI/
├── CLAUDE.md                        ← Project rules / 프로젝트 규칙
├── README.MD                        ← Progress tracker / 진행 추적
├── inbox/                           ← PDF drop zone for /drop skill
├── docs/
│   ├── WORKFLOW.md                  ← This file / 이 파일
│   ├── IMPROVEMENTS.md              ← Improvement ideas / 개선 아이디어
│   └── MCP_SETUP.md                 ← MCP server setup / MCP 서버 설정
├── Artificial_Intelligence/
│   ├── README.MD
│   └── papers/
│       ├── reading_list.md          ← 28-paper list / 28편 논문 리스트
│       ├── 01_mcculloch_pitts_1943/
│       │   ├── 01_mcculloch_pitts_1943_paper.pdf
│       │   ├── 01_mcculloch_pitts_1943_briefing.md    ← 사전 브리핑 및 Q&A
│       │   ├── 01_mcculloch_pitts_1943_notes.md       ← 읽기 후 생성
│       │   ├── 01_mcculloch_pitts_1943_implementation.ipynb ← 읽기 후 생성
│       │   └── archive/             ← Previous cycle's work / 이전 주기 작업물
│       └── ...
├── Solar_Physics/
│   ├── README.MD
│   └── papers/
│       ├── reading_list.md          ← 28-paper list / 28편 논문 리스트
│       └── ...
├── Space_Weather/
│   ├── README.MD
│   └── papers/
│       ├── reading_list.md          ← 25-paper list / 25편 논문 리스트
│       └── ...
└── Living_Reviews_in_Solar_Physics/
    ├── README.MD
    └── papers/
        ├── reading_list.md          ← 87+ paper list / 87+편 논문 리스트
        └── ...
```

**Archive convention / 아카이브 규칙:**
When restarting a topic from paper #1, existing `notes.md` and `implementation.ipynb` are moved to `<paper_dir>/archive/`. PDFs stay in place.
토픽을 논문 #1부터 다시 시작할 때, 기존 `notes.md`와 `implementation.ipynb`는 `<paper_dir>/archive/`로 이동됩니다. PDF는 그대로 유지됩니다.

---

## 4. Naming Conventions / 네이밍 규칙

### Paper directories / 논문 디렉토리
```
<two-digit-number>_<first_author_last_name>_<year>/
```
Examples / 예시:
- `07_lecun_1989/`
- `08_cortes_vapnik_1995/`
- `01_galileo_1613/` (Solar Physics)
- `01_birkeland_1908/` (Space Weather)

### Notes YAML frontmatter
```yaml
---
title: "Full Paper Title"
authors: First Author, Second Author, ...
year: YYYY
journal: "Journal Name, Vol. X, pp. Y–Z"
topic: Main Topic / Subtopic
tags: [keyword1, keyword2, ...]
status: completed   # or: in_progress
date_started: YYYY-MM-DD
date_completed: YYYY-MM-DD
---
```

---

## 5. Language Rules / 언어 규칙

- **Conversation with user / 사용자와 대화**: Always in Korean (한국어)
- **Documents / 문서** (notes, markdown files): Bilingual Korean/English (한/영 병기)
- **Code comments & docstrings / 코드 주석**: Google-style English
- **Technical terms / 기술 용어**: Keep original English (do not translate / 번역하지 않음)

---

## 6. Multi-Topic Roadmaps / 다중 주제 로드맵

### Artificial Intelligence — 28 Papers / 인공지능 — 28편

#### Phase 1: Foundations (1943–1969) — Papers 1–4
- #1 McCulloch & Pitts (1943) — First neuron model
- #2 Turing (1950) — Can machines think?
- #3 Rosenblatt (1958) — Perceptron
- #4 Minsky & Papert (1969) — Perceptron limits → AI winter

#### Phase 2: Revival & Classical ML (1982–2001) — Papers 5–11
- #5 Hopfield (1982) — Associative memory
- #6 Rumelhart et al. (1986) — Backpropagation
- #7 LeCun et al. (1989) — CNN / Zip Code
- #8 Cortes & Vapnik (1995) — SVM
- #9 Hochreiter & Schmidhuber (1997) — LSTM
- #10 LeCun et al. (1998) — LeNet-5
- #11 Breiman (2001) — Random Forests

#### Phase 3: Deep Learning Revolution (2006–2016) — Papers 12–19
- #12 Hinton et al. (2006) — Deep Belief Nets
- #13 Krizhevsky et al. (2012) — AlexNet
- #14 Mikolov et al. (2013) — Word2Vec
- #15 Kingma & Welling (2013) — VAE
- #16 Goodfellow et al. (2014) — GAN
- #17 Bahdanau et al. (2014) — Attention
- #18 Kingma & Ba (2014) — Adam
- #19 He et al. (2015) — ResNet

#### Phase 4: Transformer Era (2017–Present) — Papers 20–25
- #20 Vaswani et al. (2017) — Transformer
- #21 Devlin et al. (2018) — BERT
- #22 Radford et al. (2019) — GPT-2
- #23 Dosovitskiy et al. (2020) — ViT
- #24 Brown et al. (2020) — GPT-3
- #25 Ho et al. (2020) — DDPM (Diffusion)

#### Phase 5: Alignment & Frontiers (2022–Present) — Papers 26–28
- #26 Ouyang et al. (2022) — InstructGPT / RLHF
- #27 Kaplan et al. (2020) — Scaling Laws
- #28 Bai et al. (2022) — Constitutional AI

### Solar Physics — 28 Papers / 태양 물리학 — 28편
See `Solar_Physics/papers/reading_list.md` for full details.
전체 내용은 `Solar_Physics/papers/reading_list.md` 참조.

### Space Weather — 25 Papers / 우주 기상 — 25편
See `Space_Weather/papers/reading_list.md` for full details.
전체 내용은 `Space_Weather/papers/reading_list.md` 참조.

### Living Reviews in Solar Physics — 87+ Papers / 태양 물리학 리빙 리뷰 — 87+편

Complete LRSP journal catalog, read chronologically.
LRSP 저널 전체 카탈로그를 시간순으로 읽습니다.

#### Phase 1: Early Reviews (2004–2008) — Papers 1–14
#### Phase 2: Expanding Coverage (2009–2012) — Papers 15–31
#### Phase 3: Comprehensive Reviews (2013–2016) — Papers 32–51
#### Phase 4: Modern Topics (2017–2020) — Papers 52–71
#### Phase 5: Current Frontiers (2021–Present) — Papers 72–87+

See `Living_Reviews_in_Solar_Physics/papers/reading_list.md` for full details.
전체 내용은 `Living_Reviews_in_Solar_Physics/papers/reading_list.md` 참조.

---

## 7. Key Technical Decisions / 주요 기술적 결정

### Implementation style / 구현 스타일
- **From scratch first / 직접 구현 우선**: Every core algorithm is implemented in pure NumPy before using libraries
- **Then validate / 검증**: Compare scratch implementation against sklearn/PyTorch equivalent
- **Visualize math / 수학 시각화**: Every key equation gets a matplotlib visualization
- **Paper experiments / 논문 실험**: Reproduce the paper's own results (scaled down if needed)

### Notes style / 노트 스타일
- Always include the complete mathematical derivation / 완전한 수학적 유도 포함
- Always include a "Paper in the Arc of History" ASCII timeline / 역사 타임라인 포함
- Always include a connections table / 연결 테이블 포함
- Key Takeaways are 6–8 points, numbered, each starting with a bold claim
- Documents are bilingual (Korean/English) / 문서는 한/영 병기

### PDF acquisition / PDF 확보
- Preferred: Springer open access via `link.springer.com/content/pdf/<DOI>.pdf`
- The Unpaywall API (`api.unpaywall.org/v2/<DOI>`) is reliable for finding open access URLs
- Download with Python `urllib` + `ssl._create_unverified_context()` for self-signed certs
- Save directly to `papers/<dir>/paper.pdf`

---

## 8. How to Add Papers via Inbox / 인박스를 통한 논문 추가 방법

The `/drop` skill allows users to add papers by simply placing PDFs in the `inbox/` folder.
`/drop` 스킬을 사용하면 `inbox/` 폴더에 PDF를 넣기만 하면 논문을 추가할 수 있습니다.

**Workflow / 워크플로우:**
1. Place PDF(s) in `inbox/` at the project root / 프로젝트 루트의 `inbox/`에 PDF 넣기
2. Run `/drop` / `/drop` 실행
3. Claude analyzes the PDF, classifies it into a topic, and asks for confirmation
   Claude가 PDF를 분석하고 토픽으로 분류한 후 확인을 요청합니다
4. The paper is added to the topic's `reading_list.md` under "User-Added Papers" section
   논문이 해당 토픽의 `reading_list.md`에 "User-Added Papers" 섹션으로 추가됩니다
5. The PDF is moved to the correct paper directory with proper naming
   PDF가 올바른 네이밍으로 해당 논문 디렉토리로 이동됩니다
6. Optionally chains to `/study` for immediate reading
   선택적으로 `/study`로 연결하여 바로 학습 진행 가능

**Notes / 참고:**
- If the paper doesn't match any existing topic, `/new-topic` is auto-invoked
  기존 토픽에 맞지 않으면 `/new-topic`이 자동 실행됩니다
- Duplicate papers are detected and the user is asked how to handle them
  중복 논문은 감지되며 처리 방법을 사용자에게 확인합니다
- Non-PDF files in inbox are warned and skipped
  inbox의 비-PDF 파일은 경고 후 스킵됩니다

---

## 9. How to Resume in a New Session / 새 세션에서 재개하는 방법

1. Read `CLAUDE.md` (project rules — already loaded automatically)
   `CLAUDE.md` 읽기 (프로젝트 규칙 — 자동 로드)
2. Read this `docs/WORKFLOW.md` file
   이 `docs/WORKFLOW.md` 파일 읽기
3. Check `README.MD` for current progress count
   `README.MD`에서 현재 진행 상황 확인
4. Check each topic's `reading_list.md` for the next paper (first `[ ]` status)
   각 토픽의 `reading_list.md`에서 다음 논문 확인 (첫 번째 `[ ]` 상태)
5. Say "진행" or "/study" — Claude will pick up from the next paper
   "진행" 또는 "/study" 입력 — Claude가 다음 논문부터 이어서 진행

The standard opening sequence when user says "진행" or "/study":
사용자가 "진행" 또는 "/study"를 입력했을 때의 표준 시작 순서:
1. Identify the next unread paper from any topic's reading_list.md
   각 토픽의 reading_list.md에서 다음 미읽은 논문 식별
2. Download its PDF / PDF 다운로드
3. Provide pre-reading briefing / 사전 읽기 브리핑 제공
4. Wait for user to upload screenshots / say they've read it
   사용자가 스크린샷을 업로드하거나 읽었다고 할 때까지 대기
5. Write notes.md + implementation.ipynb / notes.md + implementation.ipynb 작성
6. Update tracking files / 추적 파일 업데이트
