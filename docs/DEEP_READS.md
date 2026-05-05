# Deep Reads / 정독 추천 목록

Line-by-line 으로 음미하며 읽기를 권하는 논문들.
분량(적정 길이), 언어(접근성), 수식·논리 밀도, 역사적 무게를 종합적으로 고려하여 선정합니다.

A curated list of papers worth reading line-by-line.
Selected based on length (concise), language accessibility, density of equations/logic, and historical weight.

---

## 선정 기준 / Selection Criteria

| 기준 | 설명 / Description |
|------|-------------------|
| **분량 / Length** | 너무 길지 않을 것 (4–30 페이지 권장). 리뷰·단행본은 정독보다 통독이 적합 / Not too long (4–30 pages preferred). Reviews and monographs are better skimmed than parsed |
| **언어 / Language** | 영어 접근성, 명료한 문장 / English accessibility, clear prose |
| **수식·논리 밀도 / Density** | 모든 줄·식이 의미를 가질 것 — 한 줄도 흘리지 말아야 할 것 / Every line/equation must carry meaning — nothing skippable |
| **역사적 무게 / Historical weight** | 분야의 결정적 전환점이거나 후속 연구의 원전 / Decisive turning point or canonical source for follow-up work |

---

## 추천 형식 / Format

각 항목은 다음을 포함:
- 토픽, 논문 번호, 제목, 저자, 연도
- 분량 / Length
- 추천 이유 / Why deep-read
- 음미해야 할 핵심 부분 / Sections to focus on

Each entry includes:
- Topic, paper number, title, authors, year
- Length
- Why deep-read
- Sections to focus on

---

## Artificial Intelligence

### #25 Attention Is All You Need
- **저자 / Authors**: Vaswani et al.
- **연도 / Year**: 2017
- **분량 / Length**: ~15 pages
- **추천 이유 / Why deep-read**: 현대 AI의 원전. Transformer 아키텍처의 모든 설계 결정이 이후 LLM·ViT·멀티모달 모델의 기초가 되었음. 짧으면서 한 줄도 흘리지 말아야 할 논문 / The canonical source of modern AI. Every architectural choice in the Transformer became the foundation of subsequent LLMs, ViTs, and multimodal models. Concise yet every line matters
- **음미할 부분 / Focus on**: Scaled dot-product attention 식, multi-head attention 설계, positional encoding 선택, encoder-decoder mask 설계 / Scaled dot-product attention equation, multi-head design, positional encoding choice, encoder-decoder mask design

### #35 Denoising Diffusion Probabilistic Models (DDPM)
- **저자 / Authors**: Ho, Jain, Abbeel
- **연도 / Year**: 2020
- **분량 / Length**: ~25 pages
- **추천 이유 / Why deep-read**: forward/reverse process, ELBO 유도, ε-prediction 변환까지 수학 밀도가 매우 높음. 현대 생성 모델 (Stable Diffusion, DALL-E 2)의 기반 / High mathematical density throughout — forward/reverse process, ELBO derivation, ε-prediction reformulation. Foundation of modern generative models (Stable Diffusion, DALL-E 2)
- **음미할 부분 / Focus on**: Variational bound 유도 (Eq. 3–5), simplified objective (Eq. 14)로의 변환 과정, noise schedule 선택의 의미 / Variational bound derivation (Eq. 3–5), reformulation to simplified objective (Eq. 14), meaning of noise schedule choices

---

## Solar Physics

### #26 Torus Instability
- **저자 / Authors**: Kliem & Török
- **연도 / Year**: 2006
- **분량 / Length**: **4 pages** (PRL)
- **추천 이유 / Why deep-read**: PRL 4페이지 압축미의 정수. 모든 식·문장이 결정적. CME 분출의 정량적 임계조건 (decay index n > 1.5)을 제시한 짧고 강렬한 논문 / Quintessence of PRL's 4-page compactness. Every equation and sentence is decisive. Short, intense paper presenting the quantitative threshold (decay index n > 1.5) for CME eruption
- **음미할 부분 / Focus on**: Decay index 정의, 토러스 평형의 안정성 분석, n > 1.5 임계값 도출 / Decay index definition, stability analysis of toroidal equilibrium, derivation of n > 1.5 threshold

### #25 Magnetic Breakout Model
- **저자 / Authors**: Antiochos, DeVore, Klimchuk
- **연도 / Year**: 1999
- **분량 / Length**: ~10 pages
- **추천 이유 / Why deep-read**: CME 이론의 양대 축 중 하나. 위상학적 논리가 단계별로 명료히 전개되어 한 단락씩 따라갈 가치 / One of the two pillars of CME theory. Topological logic unfolds clearly step by step, worth following paragraph by paragraph
- **음미할 부분 / Focus on**: Quadrupolar field 토폴로지, null point 위 reconnection이 어떻게 overlying field를 제거하는지의 인과 사슬 / Quadrupolar field topology, causal chain of how reconnection above the null point removes the overlying field

---

## Space Weather

### #26 Radiation Belt Wave-Particle Interactions
- **저자 / Authors**: Thorne
- **연도 / Year**: 2010
- **분량 / Length**: 짧은 frontier 리뷰 / Short frontier review (GRL)
- **추천 이유 / Why deep-read**: chorus·EMIC·plasmaspheric hiss의 역할이 응축되어 한 문장도 버릴 게 없음. 방사선대 전자 가속·손실 메커니즘의 결정적 정리 / Roles of chorus, EMIC, and plasmaspheric hiss are condensed — no sentence is dispensable. Decisive summary of radiation belt electron acceleration/loss mechanisms
- **음미할 부분 / Focus on**: 각 파동 모드별 가속/손실 영역, quasi-linear diffusion 계수의 의미 / Acceleration/loss regions for each wave mode, meaning of quasi-linear diffusion coefficients

### #30 Impenetrable Barrier
- **저자 / Authors**: Baker et al.
- **연도 / Year**: 2014 (Nature Letter)
- **분량 / Length**: 매우 짧음 / Very short (Nature Letter)
- **추천 이유 / Why deep-read**: Nature Letter 형식이라 매우 짧고 발견 자체가 명료. 논리 추적이 쉬워 line-by-line 정독에 가장 적합 / Very short Nature Letter format with a clear-cut discovery. Easy to trace the logic — most suitable for line-by-line reading
- **음미할 부분 / Focus on**: 2.8 R_E 경계 관측의 통계적 신뢰도, plasmaspheric hiss와 경계 유지 메커니즘의 인과 관계 / Statistical confidence of the 2.8 R_E boundary observation, causal link between plasmaspheric hiss and boundary maintenance

---

## Solar Observation
*(추가 예정 / To be added)*

---

## Living Reviews in Solar Physics
*(추가 예정 / To be added)*

---

## 사용 / Usage

- 새로운 추천이 추가될 때마다 이 파일을 갱신
- `/study` 진행 시 해당 논문에 도달하면 정독 권유 알림
- 정독 완료 후 짧은 메모(인상적인 부분, 어려웠던 부분)를 항목 아래 남겨도 됨

- Update this file whenever new recommendations are added
- When `/study` reaches one of these papers, prompt for line-by-line reading
- Optional: leave a short note (memorable parts, difficult parts) under each entry after deep reading
