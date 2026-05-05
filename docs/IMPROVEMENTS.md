# Project Improvement Ideas / 프로젝트 개선 아이디어

This document captures ideas for improving the Study with AI project over time.
이 문서는 Study with AI 프로젝트를 지속적으로 개선하기 위한 아이디어를 정리합니다.

**Last updated / 최종 수정**: 2026-04-02

---

## 1. Cross-Topic Connections / 주제 간 연결

**Idea / 아이디어**: Create explicit links between related concepts across the three topics.
세 가지 주제 사이의 관련 개념을 명시적으로 연결합니다.

**Examples / 예시**:
- AI techniques (neural networks, SVMs) are used in Space Weather forecasting
  AI 기법 (신경망, SVM)은 우주기상 예보에 활용됨
- MHD simulations in Solar Physics use numerical methods relevant to AI optimization
  태양 물리학의 MHD 시뮬레이션은 AI 최적화와 관련된 수치 방법 사용
- Time series analysis appears in all three fields
  시계열 분석은 세 분야 모두에 등장

**How to implement / 구현 방법**:
- Add a "Cross-Topic Connections" section in each paper's `notes.md`
  각 논문의 `notes.md`에 "주제 간 연결" 섹션 추가
- Create a `docs/CONNECTIONS.md` mapping file linking related papers across topics
  주제 간 관련 논문을 연결하는 `docs/CONNECTIONS.md` 매핑 파일 생성

---

## 2. Spaced Repetition / 간격 반복 학습

**Idea / 아이디어**: Periodically review completed papers to reinforce understanding.
완료된 논문을 주기적으로 복습하여 이해를 강화합니다.

**How to implement / 구현 방법**:
- After completing each phase, do a "review session" of all papers in that phase
  각 단계 완료 후, 해당 단계의 모든 논문에 대한 "복습 세션" 진행
- Create quick-review Q&A cards for each paper (key questions + answers)
  각 논문에 대한 빠른 복습용 Q&A 카드 생성 (핵심 질문 + 답변)
- Add review dates to YAML frontmatter in notes.md
  notes.md의 YAML frontmatter에 복습 날짜 추가

---

## 3. Concept Maps / 개념 지도

**Idea / 아이디어**: Create visual diagrams showing how papers and concepts connect within each topic.
각 주제 내에서 논문과 개념이 어떻게 연결되는지 보여주는 시각적 다이어그램 생성.

**How to implement / 구현 방법**:
- Use Mermaid diagrams in Markdown for concept maps
  개념 지도에 Mermaid 다이어그램 사용
- Update after each paper is completed
  각 논문 완료 후 업데이트
- Store in `<Topic>/notes/concept_map.md`
  `<Topic>/notes/concept_map.md`에 저장

---

## 4. Exercise Problems / 연습 문제

**Idea / 아이디어**: Create practice problems after each paper or phase to test understanding.
각 논문 또는 단계 이후에 이해도를 테스트하는 연습 문제 생성.

**How to implement / 구현 방법**:
- Add an `exercises.md` file in each paper directory (optional)
  각 논문 디렉토리에 `exercises.md` 파일 추가 (선택)
- Include both conceptual questions and coding challenges
  개념적 질문과 코딩 챌린지 모두 포함
- Phase-end comprehensive exercises in `<Topic>/notebooks/`
  단계별 종합 연습은 `<Topic>/notebooks/`에 배치

---

## 5. Implementation Challenges / 구현 챌린지

**Idea / 아이디어**: Go beyond reproducing paper results — create bonus coding challenges.
논문 결과 재현을 넘어 보너스 코딩 챌린지 생성.

**Examples / 예시**:
- Extend the perceptron to multi-class classification
  퍼셉트론을 다중 클래스 분류로 확장
- Apply Hopfield network to a new pattern recognition problem
  Hopfield 네트워크를 새로운 패턴 인식 문제에 적용
- Implement a solar flare predictor using SVM
  SVM을 사용한 태양 플레어 예측기 구현

**How to implement / 구현 방법**:
- Add challenge notebooks in `<Topic>/notebooks/challenges/`
  `<Topic>/notebooks/challenges/`에 챌린지 노트북 추가

---

## 6. Weekly Reviews / 주간 리뷰

**Idea / 아이디어**: Structured weekly reflection on what was learned.
학습 내용에 대한 구조화된 주간 회고.

**How to implement / 구현 방법**:
- Weekly entry in `<Topic>/notes/weekly/YYYY-WW.md`
  `<Topic>/notes/weekly/YYYY-WW.md`에 주간 항목 작성
- Template: What was learned, what was difficult, what to review, next goals
  템플릿: 학습한 것, 어려웠던 것, 복습할 것, 다음 목표

---

## 7. Progress Dashboard / 진행 대시보드

**Idea / 아이디어**: Enhanced progress tracking with statistics and visual indicators.
통계 및 시각적 표시기를 포함한 향상된 진행 추적.

**How to implement / 구현 방법**:
- Create a Jupyter notebook `notebooks/progress_dashboard.ipynb` that parses all `reading_list.md` files
  모든 `reading_list.md` 파일을 파싱하는 Jupyter 노트북 생성
- Visualize: completion percentage, papers per week, time per paper, phase progress
  시각화: 완료율, 주당 논문 수, 논문당 시간, 단계별 진행
- Auto-generate progress badges for README
  README용 진행 배지 자동 생성

---

## 8. Blog/Summary Writing / 블로그/요약 작성

**Idea / 아이디어**: Write public-facing summaries of learnings to solidify understanding.
학습 내용을 외부에 공유하는 요약을 작성하여 이해를 공고히 합니다.

**How to implement / 구현 방법**:
- After each phase, write a blog-style summary in `docs/blog/`
  각 단계 이후 `docs/blog/`에 블로그 스타일 요약 작성
- Focus on insights and connections, not just paper summaries
  단순 논문 요약이 아닌 통찰과 연결에 집중
- Could be published on a personal blog or GitHub Pages
  개인 블로그 또는 GitHub Pages에 게시 가능

---

## Priority / 우선순위

| # | Idea / 아이디어 | Impact / 영향 | Effort / 노력 | Priority / 우선순위 |
|---|---|---|---|---|
| 1 | Cross-Topic Connections | High | Low | ★★★ |
| 2 | Spaced Repetition | High | Medium | ★★★ |
| 3 | Concept Maps | Medium | Low | ★★☆ |
| 4 | Exercise Problems | Medium | Medium | ★★☆ |
| 5 | Implementation Challenges | Medium | High | ★☆☆ |
| 6 | Weekly Reviews | Low | Low | ★☆☆ |
| 7 | Progress Dashboard | Low | Medium | ★☆☆ |
| 8 | Blog/Summary Writing | Medium | High | ★☆☆ |
