---
title: "Perceptrons: An Introduction to Computational Geometry"
authors: Marvin Minsky, Seymour Papert
year: 1969
journal: "MIT Press (Book)"
topic: Artificial Intelligence / Computational Geometry
tags: [perceptron, linear separability, connectedness, convexity, local vs global, computational limits, AI winter, parallel computation]
status: completed
date_started: 2026-04-08
date_completed: 2026-04-08
---

# Perceptrons: An Introduction to Computational Geometry (1969)
# 퍼셉트론즈: 계산 기하학 입문 (1969)

**Marvin Minsky, Seymour Papert** — MIT

---

## Core Contribution / 핵심 기여

Rosenblatt의 perceptron(1958)이 "학습하는 기계"로 큰 흥분을 일으킨 후, Minsky와 Papert은 단층 perceptron이 무엇을 계산할 수 있고 무엇을 계산할 수 없는지의 경계를 **엄밀한 수학적 증명**으로 확립했습니다. 핵심 발견: 볼록성(convexity)처럼 **국소적(local)**인 기하학적 속성은 perceptron으로 계산 가능하지만, 연결성(connectedness)처럼 **전역적(global)**인 속성은 어떤 지름 제한/차수 제한 perceptron으로도 계산 불가능합니다. 이 "국소 vs 전역" 구분은 단층 네트워크의 근본적 한계를 밝히며, 신경망 연구에 대한 열기를 급격히 식혀 첫 "AI 겨울"을 초래한 것으로 널리 알려져 있습니다. 동시에 이 책은 병렬 계산 이론의 기초를 놓았으며, 다층 네트워크의 필요성을 이론적으로 정당화했습니다.

After Rosenblatt's perceptron (1958) generated enormous excitement as a "learning machine," Minsky and Papert established, through **rigorous mathematical proofs**, the exact boundary between what single-layer perceptrons can and cannot compute. Key discovery: **local** geometric properties like convexity are computable by perceptrons, but **global** properties like connectedness are not — regardless of diameter or order restrictions. This local-vs-global distinction revealed the fundamental limits of single-layer networks, is widely credited with triggering the first "AI winter," and simultaneously laid foundations for parallel computation theory while theoretically justifying the necessity of multi-layer networks.

---

## Reading Notes / 읽기 노트

### §0.0–0.1: Readers and Real, Abstract, Mythological Computers (pp. 1–3) — 독자와 컴퓨터의 세 종류

**세 부류의 독자 / Three Types of Readers:**

Minsky & Papert은 세 종류의 독자를 상정합니다: (1) 패턴 인식과 학습 기계 전문가, (2) 추상 수학으로서 즐길 독자, (3) **계산의 일반 이론**에 관심 있는 독자. 가장 중요한 것은 세 번째 — 그들의 진정한 목표는 perceptron을 비판하는 것이 아니라, 병렬 계산(parallel computation)의 이론적 기초를 세우는 것입니다.

They address three audiences: (1) pattern recognition / learning machine specialists, (2) abstract mathematicians, and (3) those interested in the **general theory of computation**. The third is most important — their real goal is not to criticize perceptrons per se, but to lay theoretical foundations for parallel computation.

**계산 과학의 미성숙 / Immaturity of Computer Science:**

Minsky & Papert은 당시 계산 과학이 얼마나 미성숙했는지를 강조합니다. 연립 방정식을 푸는 데 $n^3$번의 곱셈이 필요하다는 것을 모두 "알고" 있지만, 아무도 그것을 **증명**하지 못했습니다. "병렬 vs 직렬", "국소 vs 전역", "디지털 vs 아날로그" 같은 용어가 마치 정확한 기술 개념처럼 사용되지만, 실제로는 거의 정의되지 않은 상태입니다.

They emphasize how immature computer science was: everyone "knows" solving linear equations requires $n^3$ multiplications, but nobody can **prove** it. Terms like "parallel vs serial," "local vs global," and "digital vs analog" are used as though they were precise technical concepts, but they are nearly undefined.

이 진단은 논문의 동기를 이해하는 핵심입니다: 모호한 개념을 **엄밀한 수학적 이론**으로 정립하려는 시도입니다.

This diagnosis is key to understanding the paper's motivation: an attempt to turn vague concepts into **rigorous mathematical theory**.

---

### §0.2: Mathematical Strategy (pp. 3–4) — 수학적 전략

**"잘 선택된 특수 사례의 깊은 이해" / "Deep Understanding of Well-Chosen Cases":**

매우 일반적인 이론을 시도하기보다, 잘 선택된 특수한 상황을 **철저히 이해**하는 것이 더 나은 전략이라고 주장합니다. 그들이 선택한 것은 "병렬(parallel)이라고 주장할 수 있는 가장 단순한 기계" — 루프나 피드백 경로가 없으면서도 비자명한 계산을 수행할 수 있는 장치입니다.

Rather than attempting a very general theory, they argue for **thorough understanding** of well-chosen particular situations. Their choice: "the simplest machines with a clear claim to be parallel" — no loops or feedback, yet capable of nontrivial computation.

이것은 과학적 방법론에 대한 강력한 진술입니다: 아직 기본 사례도 이해하지 못하면서 일반 이론을 세우는 것은 "정의만 많고 정리는 없는 공허한 일반성"이 됩니다.

This is a powerful statement about methodology: building general theories before understanding basic cases leads to "vacuous generality with more definitions than theorems."

---

### §0.3: Cybernetics and Romanticism (pp. 4–5) — 사이버네틱스와 낭만주의

**"Rosenblatt의 선구적 작업을 기리며 'perceptron'이라는 이름을 사용" / Acknowledging Rosenblatt's Pioneer Work:**

Minsky & Papert은 Rosenblatt의 선구적 기여를 명시적으로 인정합니다. 하지만 즉시 강한 비판이 따릅니다: perceptron에 관한 대부분의 문헌은 "과학적 가치가 없다(without scientific value)"고 선언합니다.

They explicitly acknowledge Rosenblatt's pioneer contribution, but immediately follow with sharp criticism: most of the literature on perceptrons is "without scientific value."

**"낭만주의의 시대는 끝났다" / "The Time for Romanticism is Over":**

그들의 핵심 주장: 사이버네틱스와 계산 과학은 "낭만적 열정(flourish of romanticism)"으로 시작했고, 이것은 올바랐습니다 — 엄격함을 너무 일찍 요구했다면 발전이 느려졌을 것입니다. 하지만 "이제 성숙의 시간이 왔다(the time has come for maturity)." 추측적 탐구(speculative enterprise)에 **동등하게 상상력 있는 비판 기준(equally imaginative standards of criticism)**을 대응시켜야 합니다.

Their key claim: cybernetics began rightly with "a flourish of romanticism" — demanding rigor too early would have slowed progress. But "the time has come for maturity": speculative enterprise must be matched with "equally imaginative standards of criticism."

이 표현 자체가 매우 정치적으로 섬세합니다: "너희가 틀렸다"가 아니라 "우리 모두 이제 성장할 때가 됐다"입니다.

The phrasing is politically deft: not "you were wrong" but "it's time we all grew up."

---

### §0.4: Parallel Computation (pp. 5–6) — 병렬 계산

**2단계 구조 / The Two-Stage Architecture:**

병렬 계산의 가장 단순한 개념을 형식화합니다:

They formalize the simplest concept of parallel computation:

**Stage I**: 독립적으로 부분 함수 $\varphi_1(X), \varphi_2(X), ..., \varphi_n(X)$를 계산합니다. Compute partial functions $\varphi_1(X), \varphi_2(X), ..., \varphi_n(X)$ independently of one another.

**Stage II**: 결정 함수 $\Omega$가 결과들을 결합하여 최종 값 $\psi(X)$를 얻습니다. A decision function $\Omega$ combines the results to obtain the final value $\psi(X)$.

$$\psi(X) = \Omega(\varphi_1(X), \varphi_2(X), ..., \varphi_n(X))$$

**핵심 관찰 / Key Observation:** 제약이 없으면 이 정의는 의미가 없습니다 — 어떤 계산이든 $\varphi$ 하나를 $\psi$ 자체로 만들고 $\Omega$는 그것을 전달하기만 하면 되기 때문입니다. 의미 있는 이론을 위해 **Stage I과 Stage II 모두에 제약**이 필요합니다.

Without restrictions, this definition is meaningless — any computation $\psi$ could be represented trivially by making one $\varphi$ equal to $\psi$ itself. A meaningful theory requires **restrictions on both stages**.

---

### §0.5: Some Geometric Patterns; Predicates (pp. 5–7) — 기하학적 패턴과 술어

**술어(Predicate)의 정의 / Definition of Predicates:**

평면 $R$ 위의 도형 $X$(점들의 부분집합)에 대해, 술어 $\psi(X)$는 0 또는 1만 반환하는 함수입니다.

For a figure $X$ (a subset of points on the plane $R$), a predicate $\psi(X)$ is a function returning only 0 or 1.

**세 가지 핵심 술어 / Three Key Predicates:**

$$\psi_{\text{CIRCLE}}(X) = \begin{cases} 1 & \text{if } X \text{ is a circle} \\ 0 & \text{otherwise} \end{cases}$$

$$\psi_{\text{CONVEX}}(X) = \begin{cases} 1 & \text{if } X \text{ is a convex figure} \\ 0 & \text{otherwise} \end{cases}$$

$$\psi_{\text{CONNECTED}}(X) = \begin{cases} 1 & \text{if } X \text{ is a connected figure} \\ 0 & \text{otherwise} \end{cases}$$

이 세 가지 중 CONVEX와 CONNECTED의 대비가 책 전체의 핵심입니다. 직관적으로 둘 다 "비슷한 종류"의 기하학적 속성처럼 보이지만, 계산적 관점에서는 **근본적으로 다릅니다**.

The contrast between CONVEX and CONNECTED is the book's central theme. Intuitively they seem like similar geometric properties, but computationally they are **fundamentally different**.

가장 단순한 술어는 특정 점이 $X$에 포함되는지 확인하는 것입니다:

The simplest predicate checks whether a specific point is in $X$:

$$\phi_p(X) = \begin{cases} 1 & \text{if } p \in X \\ 0 & \text{otherwise} \end{cases}$$

---

### §0.6: Conjunctive Locality and Theorem 0.6.1 (pp. 7–9) — 결합적 국소성과 정리 0.6.1

**이것이 책의 가장 핵심적인 섹션입니다. This is the book's most critical section.**

**볼록성의 국소적 판정 / Local Test for Convexity:**

볼록하지 않음(NOT CONVEX)은 다음 조건을 만족하는 **세 점**이 존재하면 됩니다:

A figure fails to be convex if and only if there exist **three points** $p, q, r$ such that:

- $p \in X$ (p는 도형 안에 있음 / p is in the figure)
- $q \notin X$ (q는 도형 밖에 있음 / q is not in the figure)
- $r \in X$ (r은 도형 안에 있음 / r is in the figure)
- $q$는 선분 $\overline{pr}$ 위에 있음 / $q$ lies on the segment $\overline{pr}$

따라서 볼록성은 **세 점의 조(triplets)를 독립적으로 검사**하여 판정할 수 있습니다. 모든 검사가 통과하면(만장일치) 볼록합니다.

Convexity can be tested by independently examining **triplets of points**. If all triplets pass (unanimity), the figure is convex.

**결합적 국소성의 정의 / Definition of Conjunctive Locality:**

$$\psi \text{ is conjunctively local of order } k \iff \psi(X) = \begin{cases} 1 & \text{if } \varphi(X) = 1 \text{ for every } \varphi \in \Phi \\ 0 & \text{otherwise} \end{cases}$$

여기서 각 $\varphi$는 최대 $k$개의 점에만 의존합니다.

Where each $\varphi$ depends on at most $k$ points of $R$.

**예시 / Example:** $\psi_{\text{CONVEX}}$는 order 3으로 conjunctively local합니다. $\psi_{\text{CONVEX}}$ is conjunctively local of order 3.

---

**Theorem 0.6.1: $\psi_{\text{CONNECTED}}$는 어떤 차수로도 conjunctively local이 아니다**

**Theorem 0.6.1: $\psi_{\text{CONNECTED}}$ is not conjunctively local of any order.**

**증명 (귀류법) / Proof (by contradiction):**

$\psi_{\text{CONNECTED}}$가 order $k$로 conjunctively local이라고 가정합니다.

Assume $\psi_{\text{CONNECTED}}$ is conjunctively local of order $k$.

두 도형을 고려합니다:

Consider two figures:

```
Y₀: ■■■■■  (gap)  ■■■■■    ← 끊김 (disconnected) → ψ = 0
Y₁: ■■■■■■■■■■■■■■■■■■■   ← 연결 (connected) → ψ = 1
```

$Y_0$는 끊어져 있으므로 $\psi(Y_0) = 0$입니다. 따라서 어떤 부분 술어 $\varphi_j$가 $\varphi_j(Y_0) = 0$이어야 합니다 (만장일치가 깨져야 하므로).

Since $Y_0$ is disconnected, $\psi(Y_0) = 0$. So some partial predicate $\varphi_j$ must have $\varphi_j(Y_0) = 0$ (unanimity must break).

$\varphi_j$는 최대 $k$개의 점에만 의존합니다. 따라서 도형을 충분히 길게 만들면, 중간에 $\varphi_j$가 의존하는 점이 **하나도 없는** 사각형 $S_m$이 반드시 존재합니다.

$\varphi_j$ depends on at most $k$ points. So if the figure is long enough, there must exist a middle square $S_m$ that contains **none** of $\varphi_j$'s dependency points.

이제 $Y_2$를 만듭니다: $Y_0$에서 $S_m$을 채운 도형입니다. $Y_2$는 연결되어 있으므로 $\psi(Y_2) = 1$이어야 합니다. 하지만 $\varphi_j$의 관점에서 $Y_0$과 $Y_2$는 **동일**합니다 ($S_m$이 $\varphi_j$의 의존 점에 포함되지 않으므로). 따라서 $\varphi_j(Y_2) = \varphi_j(Y_0) = 0$이 되어, $\psi(Y_2) = 0$이 됩니다. **모순!**

Now construct $Y_2$: fill in $S_m$ in $Y_0$. $Y_2$ is connected, so $\psi(Y_2) = 1$. But from $\varphi_j$'s perspective, $Y_0$ and $Y_2$ are **identical** ($S_m$ contains none of $\varphi_j$'s points). So $\varphi_j(Y_2) = \varphi_j(Y_0) = 0$, forcing $\psi(Y_2) = 0$. **Contradiction!**

**증명의 핵심 통찰 / Key Insight of the Proof:** 국소적 부분 술어는 **"볼 수 있는 범위"가 제한**되어 있으므로, 충분히 먼 곳에서 일어나는 변화를 감지할 수 없습니다. 연결성은 본질적으로 **먼 부분들 사이의 관계**이므로, 어떤 국소적 검사로도 포착할 수 없습니다.

Local partial predicates have a **limited "field of view"**, so they cannot detect changes happening far enough away. Connectedness is inherently about **relationships between distant parts**, so no local test can capture it.

---

### §0.7: Other Concepts of Local (pp. 9–10) — 다른 국소성 개념

**결합적 국소성의 한계 / Limitations of Conjunctive Locality:**

"만장일치" 조건은 너무 좁습니다. 더 풍부한 이론을 위해 Stage II의 결정 규칙을 확장해야 합니다.

The "unanimity" requirement is too narrow. For a richer theory, the Stage II decision rule must be extended.

**두 가지 국소 제약 경로 / Two Paths for Locality Constraints:**

1. **차수 제한 (Order-restricted)**: 각 $\varphi$가 의존하는 **점의 수** 제한. Limit the **number of points** each $\varphi$ depends on.
2. **지름 제한 (Diameter-limited)**: 각 $\varphi$가 의존하는 점들 사이의 **물리적 거리** 제한. Limit the **physical distance** between points each $\varphi$ depends on.

Stage II를 "만장일치" 대신 **가중합(weighted sum)** — 즉 가중 투표 — 으로 확장하면, 비로소 perceptron의 정의에 도달합니다.

Extending Stage II from "unanimity" to **weighted sum** — weighted voting — leads to the definition of the perceptron.

---

### §0.8: Perceptrons — Formal Definition and Theorem 0.8 (pp. 10–15) — Perceptron 정의와 정리 0.8

**Perceptron의 형식적 정의 / Formal Definition:**

$$\psi \text{ is linear with respect to } \Phi \iff \psi(X) = 1 \text{ iff } \sum_{\varphi \in \Phi} a_\varphi \cdot \varphi(X) > \theta$$

**정의**: Perceptron은 주어진 부분 술어 집합 $\Phi$에 대해 선형인 **모든 술어**를 계산할 수 있는 장치입니다.

**Definition**: A perceptron is a device capable of computing all predicates which are linear in some given set $\Phi$ of partial predicates.

$\Phi$가 고정되면 가중치 $a_\varphi$와 임계값 $\theta$를 자유롭게 선택할 수 있습니다. $L(\Phi)$는 perceptron의 **레퍼토리** — 계산 가능한 모든 술어의 집합 — 입니다.

Given a fixed $\Phi$, the weights $a_\varphi$ and threshold $\theta$ can be freely chosen. $L(\Phi)$ is the perceptron's **repertoire** — the set of all computable predicates.

**5가지 perceptron 유형 / Five Perceptron Families:**

| 유형 / Type | 제약 / Constraint | 설명 / Description |
|---|---|---|
| **Diameter-limited** | 각 $\varphi$의 지지 집합 지름 제한 / Support set diameter bounded | 물리적 "시야" 크기 제한 / Physical "field of view" limit |
| **Order-restricted** | 각 $\varphi$가 최대 $n$개 점에 의존 / Each $\varphi$ depends on ≤ $n$ points | 복잡도 제한 / Complexity limit |
| **Gamba** | 각 $\varphi$가 모든 점에 의존 가능하지만, 자체가 order 1 perceptron / Each $\varphi$ is itself an order-1 perceptron | 2층 구조 / Two-layer structure |
| **Random** | $\varphi$가 확률 분포로 생성 / $\Phi$ generated by stochastic process | Rosenblatt의 원래 모델 / Rosenblatt's original model |
| **Bounded** | 무한 개의 $\varphi$이지만 계수가 유한 집합에서 옴 / Infinite $\varphi$'s but coefficients from a finite set | 정보 용량 제한 / Information capacity limit |

---

**Theorem 0.8: 지름 제한 perceptron은 $\psi_{\text{CONNECTED}}$를 계산할 수 없다**

**Theorem 0.8: No diameter-limited perceptron can compute $\psi_{\text{CONNECTED}}$.**

**증명 — 네 도형 논증 / Proof — The Four-Figure Argument:**

네 도형을 구성합니다:

Construct four figures:

```
X₀₀: ■■■  (gap)  ■■■     ← 끊김 / disconnected
X₁₀: ■■■■■■■■■  ■■■     ← 연결 / connected (왼쪽 연결)
X₀₁: ■■■  ■■■■■■■■■     ← 연결 / connected (오른쪽 연결)
X₁₁: ■■■■■■■■■■■■■■■   ← 끊김 / disconnected (양쪽 다 연결, 그런데 중간이 두 겹!)
```

(실제로는 $X_{11}$이 끊어진 도형이 되도록 구성합니다.)

부분 술어를 세 그룹으로 나눕니다:

Partition the partial predicates into three groups:

- **Group 1**: 지지 집합이 왼쪽에 있는 $\varphi$들 / Support sets near the left end
- **Group 2**: 지지 집합이 오른쪽에 있는 $\varphi$들 / Support sets near the right end
- **Group 3**: 양쪽 끝 모두에서 먼 $\varphi$들 / Support sets far from both ends

$$\sum a_\varphi \varphi(X) = \underbrace{\sum_{\text{group 1}} a_\varphi \varphi(X)}_{\Sigma_1} + \underbrace{\sum_{\text{group 2}} a_\varphi \varphi(X)}_{\Sigma_2} + \underbrace{\sum_{\text{group 3}} a_\varphi \varphi(X)}_{\Sigma_3} - \theta$$

**논증의 핵심 / Core of the Argument:**

- $X_{00}$ (끊김): $\Sigma_1 + \Sigma_2 + \Sigma_3 - \theta < 0$ (거부해야 함 / must reject)
- $X_{00} \to X_{10}$: 왼쪽만 변함 → Group 1만 영향 → $\Sigma_1$이 $\Delta_1$만큼 증가하여 총합이 양수가 됨. Only left changes → only Group 1 affected → $\Sigma_1$ increases by $\Delta_1$ making total positive.
- $X_{00} \to X_{01}$: 오른쪽만 변함 → Group 2만 영향 → $\Sigma_2$가 $\Delta_2$만큼 증가. Only right changes → only Group 2 affected → $\Sigma_2$ increases by $\Delta_2$.
- $X_{00} \to X_{11}$: **양쪽 다 변함**. 하지만 Group 1은 왼쪽만 보므로 $X_{10}$과 같은 $\Delta_1$을, Group 2는 $X_{01}$과 같은 $\Delta_2$를 봅니다. Group 3는 어느 경우에도 변하지 않습니다. **Both sides change**. But Group 1 sees only the left (same $\Delta_1$ as $X_{10}$), Group 2 sees only the right (same $\Delta_2$ as $X_{01}$). Group 3 is unchanged in all cases.

따라서: Therefore:

$$\text{Score}(X_{11}) = \text{Score}(X_{00}) + \Delta_1 + \Delta_2 > \text{Score}(X_{10}) > 0$$

$X_{11}$은 끊어져 있지만 점수가 양수이므로, perceptron은 이것을 **연결됨**으로 잘못 판정합니다. **모순!**

$X_{11}$ is disconnected but its score is positive, so the perceptron incorrectly classifies it as **connected**. **Contradiction!**

**이 증명이 심오한 이유 / Why This Proof is Profound:** 확률론이나 학습 이론이 아닌 **순수한 대수와 기하의 관계**에서 나옵니다. 지름 제한된 관찰자가 국소적으로 보는 것만으로는 전역적 구조를 결정할 수 없다는 것을 보여줍니다. Minsky & Papert가 강조하듯이, 이것은 생리학적으로도 중요합니다: 수용 세포의 기능이 지름 제한적이라면, 신경시냅스 "합산"만으로는 연결성을 계산할 수 없습니다.

The proof comes from **pure algebra-geometry relationships**, not probability or learning theory. It shows that diameter-limited observers, seeing only local information, cannot determine global structure. As Minsky & Papert note, this is also physiologically significant: if receptor cell functions are diameter-limited, neurosynaptic "summation" alone cannot compute connectedness.

---

### §0.9: Seductive Aspects of Perceptrons (pp. 14–20) — Perceptron의 매혹적 측면들

이 섹션은 perceptron에 대한 **네 가지 "매혹"**을 체계적으로 해체합니다.

This section systematically dismantles **four "seductions"** of perceptrons.

**0.9.1 — 균일한 프로그래밍과 학습 / Homogeneous Programming and Learning:**

Perceptron의 매력: $\Phi$를 고정하고 계수 $(a_1, ..., a_n)$만 조정하면 "프로그래밍"이 균일해지고, 피드백으로 자동 조정하면 "학습"이 됩니다. Perceptron 수렴 정리가 이를 보장합니다.

The perceptron's appeal: fix $\Phi$, adjust only $(a_1, ..., a_n)$ — programming becomes homogeneous, and feedback-driven adjustment becomes "learning." The perceptron convergence theorem guarantees eventual convergence.

Minsky & Papert의 세 가지 비판:

Three critiques:

1. **$L(\Phi)$의 제한된 레퍼토리 / Limited repertoire of $L(\Phi)$**: 직관적으로 단순하고 의미 있는 술어가 어떤 실현 가능한 $L(\Phi)$에도 속하지 않을 수 있습니다. Intuitively simple and meaningful predicates may belong to no practically realizable $L(\Phi)$.
2. **계수의 정보 용량 / Information capacity of coefficients**: 특정 술어에서 최대/최소 계수의 비율이 기하급수적으로 커질 수 있어, 어떤 아날로그 장치로도 저장 불가능합니다. For certain predicates, the ratio of largest to smallest coefficients grows faster than exponentially, making physical storage impossible.
3. **수렴 시간 / Convergence time**: Perceptron 수렴 정리를 인용하는 것은 "공허(vacuous)"합니다 — 유한 상태 장치는 모든 상태를 순환하며 답을 찾을 수 있으므로. 진짜 질문은 **무작위/완전 탐색에 비해 얼마나 빠른가**입니다. 일부 문제에서 수렴 시간이 크기에 대해 **지수적보다 빠르게** 증가합니다. Citing the convergence theorem is "vacuous" since a finite-state device could find the answer by cycling through all states. The real question is **how fast** relative to random/exhaustive search. For some problems, convergence time grows **faster than exponentially** with size.

**0.9.2 — 병렬 계산 / Parallel Computation:**

모든 $\varphi$를 동시에 계산하는 것은 물리적으로 병렬이지만, 대부분의 $\varphi$는 특정 결정에 **무관**합니다. 잘 계획된 순차 과정이 훨씬 적은 총 계산량으로 동일한 결정을 내릴 수 있습니다. $\psi_{\text{CONNECTED}}$의 경우, $100 \times 100$ 격자에서 각 부분 술어가 수백 개의 점을 봐야 하므로, "국소" 함수의 개념 자체가 거의 무의미해집니다.

Computing all $\varphi$'s simultaneously is physically parallel, but most are **irrelevant** to any particular decision. A well-planned sequential process could make the same decision with far less total computation. For $\psi_{\text{CONNECTED}}$ on a $100 \times 100$ retina, each partial predicate must look at hundreds of points — the concept of "local" function becomes nearly meaningless.

**0.9.3 — 아날로그 장치 / Analogue Devices:**

계수의 비율이 크기에 대해 지수적 이상으로 증가할 수 있어, $R$이 20개 이상의 구별 가능한 점을 가지면 **어떤 아날로그 저장 장치도 충분한 정보 용량을 가질 수 없습니다**.

Coefficient ratios can grow faster than exponentially with size, so for retinas with more than ~20 distinguishable points, **no simple analogue storage device has sufficient information capacity**.

**0.9.4 — 뇌 기능 모델 / Models for Brain Function:**

Perceptron의 인기는 뇌를 "느슨하게 조직된, 무작위로 연결된, 비교적 단순한 장치들의 네트워크"로 보는 이미지에서 비롯됩니다. 기억이 특정 위치가 아닌 네트워크 전체에 "분산"되어 있다는 신비주의가 이를 뒷받침합니다. 하지만 실험 결과는 대부분 실망스러웠습니다: 단순 문제에서는 잘 작동하지만 복잡해지면 급격히 성능이 저하됩니다. Minsky & Papert는 성공 사례에서도 **전역적 분산 활동이 아니라 네트워크의 비교적 작은 부분**이 실제 작업을 수행했을 것이라고 의심합니다.

The perceptron's popularity stems from an image of the brain as "a loosely organized, randomly connected network of simple devices" with memories "distributed throughout" rather than localized. But experimental results were mostly disappointing: machines worked well on simple problems but deteriorated rapidly with complexity. Minsky & Papert suspect that even in successful cases, **a relatively small part of the network**, not a global distributed activity, did the actual work.

그들은 이것을 "전체론적(holistic)" 또는 "게슈탈트(Gestalt)" 오해라고 부르며, 이 오해가 공학과 AI 분야를 "유령처럼 따라다닐 것"을 우려합니다.

They call this a "holistic" or "Gestalt" misconception and fear it will "haunt" engineering and AI.

**그러나 — 중요한 유보 / However — A Critical Caveat:**

Minsky & Papert은 기계 학습 자체를 반대하는 것이 아님을 명시합니다: "Exactly the contrary!" 하지만 **의미 있는 학습은 의미 있는 선행 구조를 전제**합니다. 부분 함수 $\varphi$가 과제에 적절히 설계되었을 때 적응적 학습이 유용하지만, 고차 문제를 특별한 설계 없이 "범용 perceptron"에 던지는 것은 도움이 안 됩니다.

They explicitly state they are NOT opposed to machine learning: "Exactly the contrary!" But **significant learning presupposes significant prior structure**. Adaptive learning works when partial functions $\varphi$ are properly designed for the task, but throwing a high-order problem at a "quasi-universal perceptron" with undesigned partial functions offers little hope.

이것은 현대의 **feature engineering**, **architecture design**, 그리고 **inductive bias** 개념의 직접적 선조입니다.

This is a direct ancestor of modern concepts of **feature engineering**, **architecture design**, and **inductive bias**.

---

## Key Takeaways / 핵심 시사점

1. **단층 perceptron에는 증명 가능한 근본적 한계가 있다.** 연결성(connectedness)처럼 전역적 구조를 요구하는 속성은 어떤 지름 제한/차수 제한 perceptron으로도 계산할 수 없습니다. 이것은 의견이 아니라 **수학적 정리**입니다.

   Single-layer perceptrons have **provable fundamental limits**. Properties requiring global structure, like connectedness, cannot be computed by any diameter-limited or order-limited perceptron. This is not opinion — it is a **mathematical theorem**.

2. **"국소 vs 전역"은 계산의 핵심 구분이다.** 볼록성(order 3)은 국소적이므로 perceptron으로 계산 가능하지만, 연결성은 전역적이므로 불가능합니다. 이 구분은 현대 CNN의 수용야(receptive field) 크기 설계, global pooling, skip connection 등의 이론적 기초입니다.

   **"Local vs global" is a fundamental distinction in computation.** Convexity (order 3) is local and perceptron-computable; connectedness is global and not. This distinction underlies modern CNN receptive field design, global pooling, and skip connections.

3. **수렴 정리의 인용은 공허하다 — 진짜 질문은 효율성이다.** Perceptron 수렴 정리는 "답이 존재하면 언젠가 찾는다"를 보장하지만, 유한 상태 장치의 완전 탐색도 마찬가지입니다. 진짜 문제는 **수렴 시간의 스케일링** — 이것이 현대 계산 복잡도 이론의 핵심입니다.

   **Citing convergence theorems is vacuous — the real question is efficiency.** The perceptron convergence theorem guarantees eventual convergence, but so does exhaustive search. The real issue is **scaling of convergence time** — the heart of modern computational complexity theory.

4. **의미 있는 학습에는 의미 있는 선행 구조가 필수적이다.** "범용 학습 기계"에 무작위 부분 함수를 넣고 아무 문제나 던지는 것은 소용없습니다. 부분 함수($\varphi$)가 과제에 맞게 설계되어야 합니다. 이것은 현대의 inductive bias, feature engineering, architecture design의 원형입니다.

   **Significant learning requires significant prior structure.** Throwing any problem at a "universal learning machine" with random partial functions is futile. The partial functions must be designed for the task — the prototype of modern inductive bias, feature engineering, and architecture design.

5. **이 비판은 단층 perceptron에만 적용되며, 다층 네트워크의 필요성을 암시한다.** Minsky & Papert의 증명은 Stage I이 독립적이고 Stage II가 선형인 경우에만 성립합니다. 다층 구조(deep networks)는 이 제약을 깨며, 이것이 #6 Rumelhart et al.(1986) backpropagation의 이론적 정당화입니다.

   **This critique applies only to single-layer perceptrons and implicitly motivates multi-layer networks.** The proofs hold only when Stage I functions are independent and Stage II is linear. Multi-layer (deep) architectures break these constraints — theoretically justifying #6 Rumelhart et al. (1986) and backpropagation.

6. **네 도형 논증은 전역 정보가 국소 관찰로 환원 불가능함을 보여주는 아름다운 증명이다.** 왼쪽 변화와 오른쪽 변화를 독립적으로만 감지할 수 있는 관찰자는, 양쪽이 동시에 변할 때 모순에 빠집니다. 이 논증은 확률론이나 학습 이론이 아니라 순수한 대수-기하 관계에서 나옵니다.

   **The four-figure argument is an elegant proof that global information cannot be reduced to local observations.** An observer that detects left and right changes independently must fall into contradiction when both change simultaneously. This argument comes from pure algebra-geometry relationships, not probability or learning theory.

7. **"AI 겨울"의 원인이 되었지만, 비판 자체는 정당했다.** Minsky & Papert의 결과가 신경망 연구를 10년 이상 위축시킨 것은 사실이지만, 그 수학적 증명은 완전히 올바릅니다. 문제는 연구 커뮤니티가 "단층 perceptron의 한계"를 "신경망 전체의 한계"로 과도하게 일반화한 것입니다.

   **Triggered the "AI winter" but the critique itself was valid.** Their mathematical proofs are entirely correct. The problem was the research community's over-generalization from "limits of single-layer perceptrons" to "limits of all neural networks."

---

## Mathematical Summary / 수학적 요약

### Perceptron Definition / 정의

$$\psi(X) = 1 \iff \sum_{\varphi \in \Phi} a_\varphi \cdot \varphi(X) > \theta$$

### Conjunctive Locality / 결합적 국소성

$$\psi \text{ is conjunctively local of order } k:$$
$$\psi(X) = 1 \iff \forall \varphi \in \Phi: \varphi(X) = 1, \quad \text{each } \varphi \text{ depends on } \leq k \text{ points}$$

### Key Results / 핵심 결과

| Predicate / 술어 | Local? / 국소적? | Perceptron-computable? |
|---|---|---|
| $\psi_{\text{CONVEX}}$ | Yes (order 3) / 예 | Yes / 예 |
| $\psi_{\text{CONNECTED}}$ | No (any order) / 아니오 | No (diameter- or order-limited) / 아니오 |
| $\psi_{\text{CIRCLE}}$ | No / 아니오 | No (diameter-limited) / 아니오 |
| XOR | — | No (order-1 perceptron) / 아니오 |

### Four-Figure Argument Structure / 네 도형 논증 구조

$$\text{Score}(X_{00}) < 0 \quad \text{(disconnected, reject)}$$
$$\text{Score}(X_{10}) = \text{Score}(X_{00}) + \Delta_1 > 0 \quad \text{(connected, accept)}$$
$$\text{Score}(X_{01}) = \text{Score}(X_{00}) + \Delta_2 > 0 \quad \text{(connected, accept)}$$
$$\text{Score}(X_{11}) = \text{Score}(X_{00}) + \Delta_1 + \Delta_2 > \text{Score}(X_{10}) > 0 \quad \text{(disconnected, but forced to accept → contradiction!)}$$

---

## Paper in the Arc of History / 역사적 맥락의 타임라인

```
1943  McCulloch & Pitts ── 최초 인공 뉴런 (고정 가중치, 논리 게이트)
  │
1949  Hebb ── "함께 발화하면 연결 강화"
  │
1950  Turing ── "기계가 학습할 수 있는가?"
  │
1958  Rosenblatt ── Perceptron (최초의 학습 가능 신경망)
  │
1962  Rosenblatt ── "Principles of Neurodynamics"
  │
1969  ★ MINSKY & PAPERT ── "PERCEPTRONS" ★
  │     단층 perceptron의 수학적 한계 증명
  │     CONNECTED ∉ L(Φ) for diameter/order-limited Φ
  │     "사이버네틱스의 낭만주의는 끝났다"
  │
  │     ···· 첫 AI 겨울 (1970s) ····
  │     신경망 연구 급감, 기호주의 AI 지배
  │
1982  Hopfield ── 에너지 기반 네트워크 (부활의 시작)
  │
1986  Rumelhart et al. ── Backpropagation
  │     다층 perceptron으로 Minsky-Papert 한계 극복
  │
1988  Minsky & Papert ── Perceptrons (Expanded Edition)
  │     다층 네트워크에 대한 신중한 인정 추가
  │
1989  LeCun ── CNN (Backprop + 국소 수용야 = local + global 해결)
  │
2017  Vaswani ── Transformer (self-attention으로 global 관계 포착)
```

---

## Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Relationship / 관계 |
|---|---|
| **#1 McCulloch & Pitts (1943)** | M-P 뉴런은 Minsky-Papert이 분석하는 부분 술어($\varphi$)의 기본 단위. M-P neurons are the basic units of the partial predicates ($\varphi$) that Minsky-Papert analyze. |
| **#3 Rosenblatt (1958)** | 직접적 비판 대상. Rosenblatt의 perceptron 흥분에 수학적 냉수를 끼얹음. "statistical separability"의 한계를 공식 증명. Direct critique target. Mathematical cold water on Rosenblatt's excitement. Formally proved limits of "statistical separability." |
| **#5 Hopfield (1982)** | AI 겨울 이후 신경망 부활의 시작. Minsky-Papert 비판을 피해 순환 네트워크/에너지 함수로 우회. Post-winter neural network revival, sidestepping the critique via recurrent networks and energy functions. |
| **#6 Rumelhart et al. (1986)** | 직접적 반론. 다층 네트워크 + backpropagation = 선형 분리 불가능 문제(XOR 등) 해결. Minsky-Papert이 "이것은 단층에만 적용"임을 인정. Direct rebuttal. Multi-layer networks + backpropagation solve linearly inseparable problems. Minsky-Papert acknowledged their critique applied only to single layers. |
| **#7 LeCun (1989)** | CNN의 국소 수용야 + 다층 구조 = "local 관찰 + global 추론" 해결. Minsky-Papert의 "local vs global" 구분을 건축적으로 극복. CNN's local receptive fields + multi-layer architecture = solving "local observation + global reasoning." Architecturally overcomes the local-vs-global distinction. |
| **#8 Cortes & Vapnik (1995)** | SVM의 커널 트릭 = 입력을 고차원으로 매핑하여 선형 분리 가능하게 만듦. Minsky-Papert이 지적한 "레퍼토리 제한"을 다른 방식으로 우회. SVM kernel trick maps inputs to higher dimensions for linear separability — a different way to circumvent the "limited repertoire" problem. |

---

## References / 참고문헌

- Minsky, M. & Papert, S., *Perceptrons: An Introduction to Computational Geometry*, MIT Press, 1969. (Expanded Edition, 1988.)
- Rosenblatt, F., "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain", *Psychological Review*, Vol. 65, No. 6, pp. 386–408, 1958.
- Rosenblatt, F., *Principles of Neurodynamics*, Spartan Books, 1962.
- Hebb, D. O., *The Organization of Behavior*, Wiley, 1949.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J., "Learning Representations by Back-propagating Errors", *Nature*, Vol. 323, pp. 533–536, 1986.
- Bottou, L., "Foreword to the Perceptrons Reprint", 2017. (CC BY-NC-ND 4.0)
