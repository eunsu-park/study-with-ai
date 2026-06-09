# Pre-Reading Briefing: Perceptrons (1969)
# 사전 읽기 브리핑: 퍼셉트론즈 (1969)

**Book**: *Perceptrons: An Introduction to Computational Geometry*
**Authors**: Marvin Minsky, Seymour Papert
**Year**: 1969 (Expanded Edition 1988)
**Publisher**: MIT Press
**Reading scope**: Introduction (Chapter 0, pp. 1–20) — the conceptual core of the entire book

이 책은 232페이지 전체를 읽는 대신, **Introduction (Chapter 0)**을 중심으로 핵심 논증과 주요 정리를 학습합니다. Chapter 0는 나머지 챕터의 모든 핵심 아이디어를 요약한 독립적 에세이입니다.

Rather than reading the entire 232-page book, we focus on **Chapter 0 (Introduction)**, which serves as a self-contained essay summarizing all the key arguments and theorems of the remaining chapters.

---

## 핵심 기여 / Core Contribution

Rosenblatt의 perceptron이 "학습하는 기계"로 큰 흥분을 일으킨 뒤, Minsky와 Papert은 이 흥분에 수학적 냉수를 끼얹었습니다. 그들은 perceptron이 계산할 수 있는 함수와 계산할 수 없는 함수의 경계를 **엄밀하게 증명**했습니다. 핵심 결과: **연결성(connectedness)**과 같은 "전역적(global)" 기하학적 속성은 어떤 지름 제한 perceptron이나 차수 제한 perceptron으로도 계산할 수 없습니다. 반면 **볼록성(convexity)**은 3차의 "국소적(local)" 속성이므로 perceptron으로 계산할 수 있습니다. 이 구분 — local vs global — 이 책의 핵심 통찰이며, 단층 네트워크의 근본적 한계를 보여줍니다. 이 결과는 신경망 연구에 대한 열기를 급격히 식혀 첫 "AI 겨울"을 초래한 것으로 널리 알려져 있습니다.

After Rosenblatt's perceptron generated enormous excitement as a "learning machine," Minsky and Papert poured mathematical cold water on that enthusiasm. They **rigorously proved** the boundary between what perceptrons can and cannot compute. The key result: **global** geometric properties like connectedness cannot be computed by any diameter-limited or order-limited perceptron, while **local** properties like convexity (order 3) can. This local-vs-global distinction is the book's central insight, demonstrating the fundamental limits of single-layer networks. The book is widely credited with dampening enthusiasm for neural networks and triggering the first "AI winter."

---

## 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1960년대는 perceptron에 대한 **과대한 기대**가 팽배했던 시기입니다. Rosenblatt(1958)이 perceptron을 발표한 후, 뉴욕 타임즈는 "해군이 생각하는 기계의 시초를 공개"라고 보도했고, 수백 개의 연구 그룹이 "학습 기계", "적응 네트워크", "자기 조직화 시스템" 실험에 뛰어들었습니다. 그러나 결과는 대부분 실망스러웠고, 작은 문제에서는 잘 작동하지만 **문제가 복잡해지면 급격히 성능이 저하**되는 패턴이 반복되었습니다.

The 1960s were marked by **inflated expectations** about perceptrons. After Rosenblatt's 1958 paper, the New York Times reported "Navy reveals embryo of electronic brain," and hundreds of research groups jumped into experiments with "learning machines" and "adaptive networks." Results were mostly disappointing: machines worked well on simple problems but **deteriorated rapidly** as tasks grew harder.

Minsky와 Papert은 이 상황을 "사이버네틱스의 낭만주의(romanticism of cybernetics)"라고 진단하고, "이제 성숙의 시간이 왔다"며 엄밀한 수학적 분석을 제시합니다.

Minsky and Papert diagnosed this as the "romanticism of cybernetics" and declared "the time has come for maturity" — offering rigorous mathematical analysis in place of speculation.

### 타임라인 / Timeline

```
1943  McCulloch & Pitts ── 최초 인공 뉴런 모델 (논문 #1)
  │
1949  Hebb ── "함께 발화하면 연결 강화" (Hebbian learning)
  │
1950  Turing ── "기계가 학습할 수 있는가?" (논문 #2)
  │
1958  Rosenblatt ── Perceptron: 최초의 학습 가능 신경망 (논문 #3)
  │
1960  Widrow & Hoff ── ADALINE
  │
1962  Rosenblatt ── "Principles of Neurodynamics" (perceptron 이론 확장)
  │
1969  ★ MINSKY & PAPERT ── "PERCEPTRONS" ★
  │     단층 perceptron의 한계를 수학적으로 증명
  │     → 신경망 연구 급감, 첫 "AI 겨울" 시작
  │
  │     ···· 10년 이상의 신경망 암흑기 ····
  │
1982  Hopfield ── 에너지 기반 신경망으로 부활의 시작 (논문 #5)
  │
1986  Rumelhart et al. ── Backpropagation으로 다층 네트워크 학습 (논문 #6)
```

### 저자 배경 / Author Background

**Marvin Minsky (1927–2016)**: MIT AI Lab 공동 창립자, AI 분야의 아버지 중 한 명. 기호주의 AI의 강력한 옹호자.

**Seymour Papert (1928–2016)**: 수학자이자 교육학자. Logo 프로그래밍 언어 창시자. Piaget의 제자로, 구성주의 교육 이론의 선구자.

두 사람 모두 기호주의(symbolic AI) 진영에 있었으며, 연결주의(connectionist) 접근법의 한계를 엄밀하게 밝히려는 동기가 있었습니다. 하지만 그들의 분석은 정치적이 아니라 수학적으로 정당한 것이었습니다.

Both were in the symbolic AI camp with motivation to expose connectionist limitations, but their analysis was mathematically rigorous rather than political.

---

## 필요한 배경 지식 / Prerequisites

### 1. Perceptron 복습 (논문 #3) / Perceptron Review (Paper #3)

Minsky & Papert이 분석하는 "perceptron"은 Rosenblatt의 모델을 수학적으로 정제한 것입니다:

The "perceptron" that Minsky & Papert analyze is a mathematically refined version of Rosenblatt's model:

$$\psi(X) = 1 \iff \sum_{\varphi \in \Phi} a_\varphi \cdot \varphi(X) > \theta$$

- $X$: 입력 패턴 (평면 위의 도형, 점들의 집합). Input pattern (a geometric figure, a set of points on a plane).
- $\Phi = \{\varphi_1, \varphi_2, ..., \varphi_n\}$: **부분 술어(partial predicates)** — 각각 입력의 일부만 검사. Partial predicates — each examines only part of the input.
- $a_\varphi$: 각 부분 술어의 가중치 (계수). Weights (coefficients) for each partial predicate.
- $\theta$: 임계값. Threshold.
- $\psi$: 최종 결정 (0 또는 1). Final decision (0 or 1).

### 2. 술어 (Predicate) / Predicates

수학에서 **술어**란 "참 또는 거짓을 반환하는 함수"입니다. 기하학적 예시:

In mathematics, a **predicate** is a function returning true or false. Geometric examples:

- $\psi_{\text{CIRCLE}}(X) = 1$ 이면 $X$가 원 / if $X$ is a circle
- $\psi_{\text{CONVEX}}(X) = 1$ 이면 $X$가 볼록 도형 / if $X$ is convex
- $\psi_{\text{CONNECTED}}(X) = 1$ 이면 $X$가 연결 도형 / if $X$ is connected

### 3. 국소적 vs 전역적 (Local vs Global) / Local vs Global

이 책의 가장 핵심 구분입니다:

This is the book's most important distinction:

| 개념 / Concept | 의미 / Meaning | 예시 / Example |
|----------------|---------------|----------------|
| **국소적 (Local)** | 입력의 **작은 부분**만 검사해서 결정 가능 | 볼록성: 세 점만 검사하면 됨 |
| **전역적 (Global)** | 입력의 **전체 구조**를 봐야 결정 가능 | 연결성: 모든 부분이 이어져 있는지 확인 필요 |

A property is **local** if it can be determined by examining only small parts of the input. A property is **global** if it requires examining the overall structure.

### 4. 선형 분리 가능성 / Linear Separability

perceptron이 계산할 수 있는 함수는 **선형 분리 가능한(linearly separable)** 함수입니다. 가중합이 임계값을 넘는지로 분류하므로, 고차원 공간에서 **하나의 초평면(hyperplane)**으로 두 클래스를 나눌 수 있어야 합니다.

A perceptron can only compute **linearly separable** functions — classifiable by a single hyperplane in high-dimensional space.

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | Minsky-Papert 정의 / Definition | 직관 / Intuition |
|---|---|---|
| **Predicate ($\psi$)** | 참/거짓을 반환하는 함수 | "이 도형은 원인가?" 같은 질문 / A yes/no question about a figure |
| **Partial predicate ($\varphi$)** | 입력의 일부만 검사하는 술어 | perceptron의 "눈" — 각각 좁은 영역만 봄 / The perceptron's "eyes" — each sees only a small area |
| **Linear predicate** | $\sum a_\varphi \varphi(X) > \theta$ 형태로 계산 가능 | 가중 투표로 결정 가능한 속성 / A property decidable by weighted voting |
| **Order (차수)** | 부분 술어가 의존하는 최대 점의 수 | order 3 = 최대 3개 점만 봄 / Looks at most 3 points |
| **Diameter-limited** | 부분 술어의 지지 집합 크기 제한 | "시야"의 물리적 크기 제한 / Physical size limit on "field of view" |
| **Conjunctively local** | 모든 부분 검사가 만장일치면 참 | "모든 검사를 통과해야 합격" / "Pass only if ALL local tests pass" |
| **$\psi_{\text{CONVEX}}$** | 도형이 볼록한지 판별 | Local (order 3) — perceptron으로 계산 가능 / Computable by perceptron |
| **$\psi_{\text{CONNECTED}}$** | 도형이 연결되어 있는지 판별 | Global — perceptron으로 계산 **불가능** / NOT computable by perceptron |
| **$L(\Phi)$** | perceptron의 **레퍼토리** — 계산 가능한 술어의 집합 | 기계의 능력 범위 / The machine's capability range |

---

## 수식 미리보기 / Equations Preview

### 1. Perceptron의 정의 / Perceptron Definition

$$\psi(X) = 1 \iff \sum_{\varphi \in \Phi} a_\varphi \cdot \varphi(X) > \theta$$

Minsky & Papert의 perceptron은 Rosenblatt의 것과 본질적으로 같지만, **부분 술어($\varphi$)를 명시적으로 분리**합니다. Stage I에서 각 $\varphi$가 독립적으로 입력의 일부를 검사하고, Stage II에서 가중합으로 최종 결정을 내립니다.

Essentially the same as Rosenblatt's, but explicitly separating the computation into Stage I (independent partial predicates) and Stage II (weighted linear combination for final decision).

### 2. 결합적 국소성 / Conjunctive Locality

$$\psi(X) = \begin{cases} 1 & \text{if } \varphi(X) = 1 \text{ for every } \varphi \in \Phi \\ 0 & \text{otherwise} \end{cases}$$

모든 부분 술어가 만장일치로 1이면 참. 이것은 가중합의 특수한 경우입니다: $a_\varphi = -1$, $\theta = -1$로 설정하면 동일합니다.

All partial predicates must unanimously return 1. This is a special case of the weighted sum: set $a_\varphi = -1$ and $\theta = -1$.

### 3. 볼록성은 국소적 (order 3) / Convexity is Local (Order 3)

$$\text{NOT CONVEX}(X) \iff \exists p, q, r: \begin{cases} p \in X \\ q \notin X \\ r \in X \\ q \in \overline{pr} \end{cases}$$

볼록하지 않음을 판정하려면 **세 점만** 검사하면 됩니다: 세그먼트 $\overline{pr}$ 위의 중간 점 $q$가 도형 밖에 있으면 볼록하지 않습니다. 따라서 $\psi_{\text{CONVEX}}$는 order 3으로 conjunctively local합니다.

To test non-convexity, you only need to examine **triplets of points**: if a midpoint $q$ on segment $\overline{pr}$ is outside the figure while $p, r$ are inside, the figure is not convex. Thus $\psi_{\text{CONVEX}}$ is conjunctively local of order 3.

### 4. 연결성은 국소적이지 않음 / Connectedness is NOT Local

**Theorem 0.6.1**: $\psi_{\text{CONNECTED}}$는 어떤 차수로도 conjunctively local이 아닙니다.

**Theorem 0.6.1**: $\psi_{\text{CONNECTED}}$ is not conjunctively local of any order.

증명의 핵심 아이디어 (네 도형 논증):

Core proof idea (four-figure argument):

```
X₀₀ (끊김/disconnected):    X₁₀ (연결/connected):
  ■■■        ■■■               ■■■■■■■■■■■
  
X₀₁ (연결/connected):       X₁₁ (끊김/disconnected):
  ■■■■■■■■■■■               ■■■■■■■■■■■
                               ↑ (gap!)
```

만약 order $k$의 부분 술어가 $X_{00}$(끊김)을 올바르게 거부한다면, $X_{10}$과 $X_{01}$(연결)을 수용하기 위해 왼쪽과 오른쪽 부분의 가중치를 각각 증가시켜야 합니다. 하지만 $X_{11}$에서는 **양쪽 다** 증가하므로, $X_{11}$(끊김)을 잘못 수용하게 됩니다. 이것은 **국소적 정보만으로는 전역적 연결을 결정할 수 없음**을 보여줍니다.

If an order-$k$ perceptron correctly rejects $X_{00}$ (disconnected), it must increase weights for left and right portions separately to accept $X_{10}$ and $X_{01}$ (connected). But for $X_{11}$, **both** increases apply, forcing incorrect acceptance of $X_{11}$ (disconnected). Local information alone cannot determine global connectivity.

### 5. Theorem 0.8: 지름 제한 perceptron의 한계 / Diameter-Limited Perceptrons

$$\text{No diameter-limited perceptron can compute } \psi_{\text{CONNECTED}}$$

이 정리도 같은 네 도형 논증을 사용하지만, "차수" 대신 "지름"으로 제약합니다. 부분 술어의 지지 집합(support set)이 제한된 지름 안에 있으면, 충분히 긴 도형에 대해 왼쪽/오른쪽/중간 그룹으로 나뉘며 같은 모순이 발생합니다.

Uses the same four-figure argument, but constrains by diameter instead of order. With diameter-limited support sets, sufficiently long figures create left/right/middle groups leading to the same contradiction.

---

## 논문의 구조 미리보기 / Paper Structure Preview

| 섹션 / Section | 내용 / Content | 난이도 / Difficulty |
|------|------|--------|
| §0.0–0.1 | **독자와 동기**: 세 종류의 독자, 병렬 계산의 미성숙 / Readers, motivation, immaturity of parallel computation theory | ⭐ 쉬움 / Easy |
| §0.2 | **수학적 전략**: 잘 선택된 특수 사례의 깊은 이해 / Mathematical strategy: deep understanding of well-chosen cases | ⭐ 쉬움 / Easy |
| §0.3 | **사이버네틱스와 낭만주의**: 과대 기대에 대한 비판 / Cybernetics and romanticism: critique of hype | ⭐ 쉬움 / Easy |
| §0.4 | **병렬 계산**: 2단계 구조 (Stage I + Stage II) 정의 / Parallel computation: two-stage architecture | ⭐⭐ 보통 / Medium |
| §0.5 | **기하학적 패턴**: CIRCLE, CONVEX, CONNECTED 정의 / Geometric predicates | ⭐⭐ 보통 / Medium |
| §0.6 | **국소성과 정리 0.6.1**: CONNECTED가 local이 아님의 증명 / Locality and Theorem 0.6.1 proof | ⭐⭐⭐ 핵심! / Key! |
| §0.7 | **다른 국소 개념들**: diameter-limited, order-restricted 정의 / Other notions of local | ⭐⭐ 보통 / Medium |
| §0.8 | **Perceptron 정의 + 정리 0.8**: 지름 제한 perceptron의 한계 증명 / Formal definition + Theorem 0.8 proof | ⭐⭐⭐ 핵심! / Key! |
| §0.9 | **매혹적 측면들의 비판**: 학습, 병렬 계산, 아날로그, 뇌 모델 / Critique of seductive aspects | ⭐⭐ 보통 / Medium |
| §0.10 | **책의 전체 구조**: Part I (대수), Part II (기하), Part III (실용) / Book overview | ⭐ 쉬움 / Easy |

### 읽기 전략 / Reading Strategy

1. **§0.0–0.3** (동기와 철학): 빠르게 읽되, "사이버네틱스의 낭만주의" 비판의 톤을 느끼세요. Read quickly, but feel the tone of the critique.
2. **§0.4–0.6** (핵심 이론): 꼼꼼히 — 특히 **Theorem 0.6.1의 증명**을 한 줄씩 따라가세요. Read carefully — follow the proof of Theorem 0.6.1 line by line.
3. **§0.7–0.8** (perceptron 정의와 정리 0.8): 꼼꼼히 — 5가지 perceptron 유형과 네 도형 논증을 이해하세요. Understand the five perceptron families and the four-figure argument.
4. **§0.9** (비판): 꼼꼼히 — Minsky & Papert이 perceptron 신봉자들을 어떻게 비판하는지 주목. Note how they critique perceptron enthusiasts.

---

## 읽으면서 생각해 볼 질문 / Questions to Consider While Reading

1. CONVEX는 order 3으로 국소적인데, CONNECTED는 왜 어떤 차수로도 국소적이지 않은가? / Why is CONVEX local (order 3) while CONNECTED is not local of any order?
2. 네 도형 논증(four-figure argument)의 **핵심 모순**은 무엇인가? / What is the **core contradiction** in the four-figure argument?
3. Minsky & Papert은 perceptron "학습"의 어떤 점을 비판하는가? / What do they criticize about perceptron "learning"?
4. "국소적 vs 전역적" 구분은 현대 CNN의 수용야(receptive field)와 어떻게 연결되는가? / How does local-vs-global connect to modern CNN receptive fields?
5. 이 비판이 **다층 perceptron**(MLP)에도 적용되는가? 왜 또는 왜 아닌가? / Does this critique apply to multi-layer perceptrons? Why or why not?
6. Minsky & Papert 자신이 인정하는 **비판의 한계**는 무엇인가? / What limitations of their own critique do they acknowledge?
