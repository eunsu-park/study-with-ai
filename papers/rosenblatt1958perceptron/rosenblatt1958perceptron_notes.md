---
title: "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
authors: Frank Rosenblatt
year: 1958
journal: "Psychological Review, Vol. 65, No. 6, pp. 386–408"
doi: "10.1037/h0042519"
topic: Artificial Intelligence / Neural Networks
tags: [perceptron, neural networks, learning, statistical separability, pattern recognition, connectionism, generalization]
status: completed
date_started: 2026-04-06
date_completed: 2026-04-07
---

# The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain (1958)
# 퍼셉트론: 뇌에서의 정보 저장 및 조직화를 위한 확률적 모델 (1958)

**Frank Rosenblatt** — Cornell Aeronautical Laboratory

---

## Core Contribution / 핵심 기여

McCulloch & Pitts(1943)가 "뉴런으로 논리 함수를 계산할 수 있다"는 이론적 가능성을 보여주었다면, Rosenblatt은 한 걸음 더 나아가 **"네트워크가 경험으로부터 스스로 학습할 수 있다"**는 것을 확률론적으로 증명했습니다. Perceptron은 감각 입력(S-points), 연상 유닛(A-units), 반응 유닛(R-units)으로 구성된 3층 구조로, A-unit의 "value"(연결 강도)가 강화(reinforcement)에 의해 변화하면서 패턴 분류를 학습합니다. 결정적으로, Rosenblatt은 시스템의 물리적 파라미터(뉴런 수, 연결 수, 임계값 등) 6개만으로 학습 곡선 전체를 예측할 수 있는 수학적 프레임워크를 제시했으며, "같은 클래스 내 유사성 > 다른 클래스 간 유사성"이라는 통계적 분리 가능성(statistical separability) 조건이 충족되면 **이전에 본 적 없는 자극에 대해서도 일반화**할 수 있음을 보였습니다.

While McCulloch & Pitts (1943) demonstrated the theoretical possibility of computing logical functions with neural units, Rosenblatt went further by proving probabilistically that a network can **learn from experience**. The perceptron — a three-layer system of sensory units (S-points), association units (A-units), and response units (R-units) — adjusts its A-unit "values" (connection strengths) through reinforcement, thereby learning pattern classification. Crucially, Rosenblatt provided a mathematical framework predicting entire learning curves from just six physical parameters, and showed that when statistical separability holds (within-class similarity > between-class similarity), the system can **generalize to never-before-seen stimuli**.

---

## Reading Notes / 읽기 노트

### Introduction: Two Theories of Information Storage (pp. 386–388) — 정보 저장의 두 이론

**세 가지 근본 질문 — 논문의 출발점 / Three Fundamental Questions — The Paper's Starting Point:**

Rosenblatt은 지능적 시스템을 이해하기 위해 세 가지 질문을 제시합니다:
1. 물리적 세계의 정보가 어떻게 **감지(sensed)**되는가?
2. 정보가 어떤 형태로 **저장(stored)**되는가?
3. 저장된 정보가 어떻게 인식과 행동에 **영향(influence)**을 미치는가?

첫 번째 질문만 감각 생리학에서 어느 정도 이해되었고, 나머지 두 질문은 아직 미해결이라고 진단합니다. 이것이 이 논문의 범위입니다.

Rosenblatt poses three questions for understanding intelligent systems: (1) How is information sensed? (2) How is it stored? (3) How does stored information influence recognition and behavior? Only the first had been adequately addressed by sensory physiology. The remaining two — how information is stored and how stored information influences behavior — were still wide open. This is the domain Rosenblatt tackles.

**두 학파의 대립 — 표상주의 vs 연결주의 / Two Schools of Thought — Representational vs Connectionist:**

정보 저장에 대해 두 가지 근본적으로 다른 입장이 존재합니다. Two fundamentally different positions exist on information storage:

| 관점 | 기억 방식 | 인식 방식 | 비유 |
|------|-----------|-----------|------|
| **표상주의 (Representational)** | 자극의 코딩된 이미지를 일대일로 저장 | 저장된 이미지와 새 자극을 **비교(matching)** | 사진 앨범에서 같은 사진 찾기 |
| **연결주의 (Connectionist/Empiricist)** | 새로운 연결/경로 형성으로 저장 | 자극이 기존 경로를 **자동 활성화** | 자주 다니면 길이 만들어짐 |

표상주의에서는 기억이 "코드를 해독하면 원래 자극을 복원할 수 있는" 형태로 저장됩니다 — 마치 디지털 사진처럼. McCulloch, Rashevsky 등의 뇌 모델이 이 전통에 속합니다. 인식은 새로운 입력을 저장된 모든 패턴과 **비교**하는 과정입니다.

연결주의에서는 기억이 뉴런들 사이의 **새로운 연결**로 저장됩니다. 정보는 "어떤 응답을 선호하는가"에 담겨 있지, 토포그래피적 표현에 담겨 있지 않습니다. Hebb의 "cell assembly"와 Hull의 "cortical anticipatory goal response"가 이 전통입니다.

In the representational view, memories are stored as coded copies of stimuli — like digital photographs — and recognition requires systematic comparison with incoming patterns. In the connectionist view, memory takes the form of new connections or pathways between neural centers. Information is stored as preferences for particular responses, not as topographic representations. Hebb's "cell assembly" and Hull's "cortical anticipatory goal response" belong to this tradition.

**Rosenblatt의 핵심 주장 — 연결주의 + 확률론 / Rosenblatt's Core Thesis — Connectionism + Probability:**

Rosenblatt은 명확하게 연결주의 입장을 취합니다. 하지만 기존의 연결주의 이론가들(Hebb, Hayek 등)과 차별되는 점이 있습니다: 이전 이론들은 "이런 식으로 작동할 것이다"라는 **정성적 제안**에 그쳤던 반면, Rosenblatt은 **확률론을 분석 언어로 사용**하여 정량적 예측이 가능한 모델을 제시합니다.

왜 확률론인가? 생물학적 신경계에서는 정확한 배선도(wiring diagram)를 알 수 없습니다. 개체마다 연결이 다르고, 대략적인 조직 구조만 특성화할 수 있습니다. Boolean 대수와 기호 논리학은 **정확한 연결 구조**를 전제하므로 이런 상황에 적합하지 않습니다. 확률론은 **"대부분의 배선에서 이 시스템이 이렇게 행동할 확률은 얼마인가?"**를 묻을 수 있게 해줍니다.

Rosenblatt explicitly adopts the connectionist position, but differs crucially from prior connectionists (Hebb, Hayek): while they offered qualitative suggestions, Rosenblatt uses **probability theory as the analytical language** to produce quantitative predictions. Why probability? Biological nervous systems have largely random connectivity at birth. The precise wiring differs between organisms. Therefore, the analytical language must handle systems where "only the gross organization can be characterized." Probability theory, unlike Boolean algebra, is naturally suited for this.

**기존 뇌 모델들의 실패 — Rosenblatt의 진단 / Failure of Previous Brain Models — Rosenblatt's Diagnosis:**

기호 논리 기반 뇌 모델(McCulloch & Pitts 등)의 문제점을 다음과 같이 나열합니다:
- **등가성(equipotentiality) 부재**: 뇌의 일부가 손상되어도 기능이 유지되는 현상을 설명 못함
- **비경제적 연결**: 너무 많은 특정 연결을 요구
- **과도한 동기화 요구**: 뉴런들의 정밀한 동기화가 필요
- **비현실적 자극 특이성**: 특정 뉴런이 발화하려면 매우 특정한 자극이 필요

Rosenblatt은 이런 문제들을 "원칙의 차이(difference in principle)"로 해결해야 한다고 주장합니다 — 단순한 개선이 아니라 근본적으로 다른 접근이 필요합니다. 그의 **통계적 분리 가능성(statistical separability)** 이론이 바로 그 해답입니다.

Rosenblatt lists the shortcomings of symbolic-logic brain models (McCulloch & Pitts, etc.): absence of equipotentiality, uneconomical wiring, excessive synchronization requirements, and unrealistic stimulus specificity. He argues these require a "difference in principle" — not mere refinement, but a fundamentally different approach. His theory of **statistical separability** is that answer.

**다섯 가지 핵심 가정 (Hebb-Hayek 전통) / Five Core Assumptions (Hebb-Hayek Tradition):**

Rosenblatt이 perceptron의 기반으로 삼는 가정:

1. **무작위 초기 연결 / Random initial connectivity**: 태어날 때 가장 중요한 네트워크는 대부분 무작위이며, 최소한의 유전적 제약만 받음. At birth, the most important networks are largely random, subject to minimal genetic constraints.
2. **가소성(plasticity)**: 신경 활동 후 세포의 전달 확률이 장기적으로 변화 (= Hebbian learning의 핵심). After neural activity, transmission probabilities change long-term — the essence of Hebbian learning.
3. **유사 자극 → 같은 반응 / Similar stimuli → same response**: 충분한 자극 노출 후, "유사한" 자극들은 같은 반응 세포에 경로를 형성. After sufficient exposure, similar stimuli form pathways to the same response cells.
4. **강화(reinforcement)**: 양성/음성 강화가 진행 중인 연결 형성을 촉진/억제. Positive/negative reinforcement facilitates or hinders ongoing connection formation.
5. **유사성은 물리적 구조에 의존 / Similarity depends on physical structure**: "유사성"은 자극의 형식적 속성이 아니라, 시스템의 물리적 조직에 의해 결정. Similarity is not an intrinsic attribute of stimuli but depends on the physical organization of the perceiving system.

특히 5번이 깊은 의미를 가집니다: "비슷하다"는 것은 자극 자체의 기하학적 속성이 아니라, **지각 시스템의 구조**에 의해 정의됩니다. 같은 물리적 자극이라도 다른 시스템에서는 다르게 분류될 수 있습니다. 이것은 현대 representation learning의 핵심 통찰 — "좋은 표현을 학습하면 분류가 쉬워진다" — 과 일맥상통합니다.

Assumption 5 is particularly profound: "similarity" is defined not by the geometry of stimuli themselves, but by the **structure of the perceiving system**. The same physical stimulus may be classified differently by different systems. This resonates with the core insight of modern representation learning — good learned representations make classification easy.

---

### The Organization of a Perceptron (pp. 388–392) — Perceptron의 구조

**3+1층 구조 — Photoperceptron / The 3+1 Layer Architecture:**

Rosenblatt이 제시하는 전형적인 photoperceptron(광자극 perceptron)은 다음 구조를 가집니다. The typical photoperceptron (a perceptron responding to optical stimuli) has the following architecture:

```
S-points (감각 유닛)
    │
    ▼  [수렴적, 국소화된 연결 — 윤곽 감지용]
A_I (투사 영역, projection area)
    │
    ▼  [무작위 연결]
A_II (연상 영역, association area)
    │
    ▼  [무작위 연결 + 양방향 피드백]
R-units (반응 유닛)
```

**각 층의 상세 설명 / Detailed Description of Each Layer:**

**1. S-points (Sensory Units / 감각 유닛):**
- 자극에 all-or-nothing으로 반응하는 감각 수용기 (현대의 입력 픽셀). Sensory receptors responding on an all-or-nothing basis (modern equivalent: input pixels).
- 망막(retina)에 배열되며, 자극의 물리적 특성을 감지. Arranged on a retina, detecting physical properties of stimuli.

**2. A_I (Projection Area / 투사 영역):**
- 각 A_I 유닛은 S-points로부터 **국소화된(focalized)** 연결을 받음. Each A_I unit receives **focalized** connections from S-points.
- 중심점에서 거리가 멀어질수록 연결 밀도가 **지수적으로 감소** — 이것은 생물학적 시각 피질의 수용야(receptive field) 조직과 일치. Connection density falls off exponentially with retinal distance — matching biological visual cortex receptive field organization.
- 윤곽(contour) 감지에 중요한 기능을 수행. Serves an important function in contour detection.

**3. A_II (Association Area / 연상 영역):**
- A_I으로부터 **무작위(random)** 연결을 받음 — 이것이 핵심적 차이. Receives **random** connections from A_I — the critical difference.
- A_I과 동일한 활성화 규칙 (임계값 기반 발화). Same activation rules as A_I (threshold-based firing).
- 무작위 연결 덕분에 특정 배선에 의존하지 않는 **통계적** 행동이 가능. Random connectivity enables **statistical** behavior independent of specific wiring.

**4. R-units (Response Units / 반응 유닛):**
- A_II로부터 무작위로 연결된 출력 유닛. Output units randomly connected from A_II.
- 각 R-unit의 **source-set**: 그 R-unit에 연결된 A-unit들의 집합. Each R-unit's **source-set**: the set of A-units transmitting to it.
- **상호 억제적(mutually exclusive)**: 하나의 R-unit이 활성화되면 다른 R-unit들을 억제. **Mutually exclusive**: when one R-unit activates, it inhibits the others.

**피드백 연결의 두 가지 규칙 / Two Feedback Rules:**

R-unit에서 A-unit으로 되돌아가는 피드백에 두 가지 선택지가 있습니다. Two alternatives exist for feedback from R-units back to A-units:

(a) **흥분성 피드백 / Excitatory feedback**: 자기 source-set의 유닛들을 흥분시킴 (해부학적으로 그럴듯). Excites cells in its own source-set (more anatomically plausible).
(b) **억제성 피드백 / Inhibitory feedback**: source-set의 **보완집합**(complement)을 억제 (분석에 유리). Inhibits the complement of its source-set (more analytically tractable).

Rosenblatt은 분석 편의를 위해 주로 규칙 (b)를 사용합니다. 효과는 동일합니다: 우세한 반응이 경쟁자를 억제하여 **winner-take-all** 동역학을 만듭니다. Rosenblatt uses rule (b) for most analyses. The effect is the same: the dominant response suppresses competitors, creating **winner-take-all** dynamics.

**단순화된 3층 모델 (Fig. 2) / Simplified 3-Layer Model:**

실제 분석은 A_I을 생략한 단순화된 모델에서 수행합니다. The actual analysis uses a simplified model omitting A_I:

```
S-points → A-units → R₁, R₂ (상호 억제)
  (망막)     (연상)    (반응)
```

이 모델에서 각 A-unit은 망막의 **무작위 위치**에서 연결을 받습니다. 따라서 "유사성"은 자극의 **겹치는 면적(coincident area)**에 기반합니다 — 윤곽이나 형태가 아니라. Rosenblatt은 이 단순 모델도 놀라울 정도로 유능하다고 주장합니다.

In this model, each A-unit receives connections from **random** retinal positions. Similarity is thus based on **coincident area** of stimuli — not contour or shape. Despite this limitation, Rosenblatt claims it has "quite impressive" capabilities.

**Value — 학습의 핵심 변수 / Value — The Key Learning Variable:**

**Value ($V$)**는 각 A-unit의 출력 강도를 나타내는 변수입니다. 이것이 학습을 통해 변하는 유일한 변수이며, 현대 신경망의 **weight(가중치)**에 해당합니다. Rosenblatt은 이것을 다음과 같이 특성화합니다:
- 진폭(amplitude), 빈도(frequency), 잠복기(latency), 또는 전송 완료 확률
- "상당히 안정적인 특성"이지만 절대적으로 일정하지는 않음
- 활동은 value를 **증가**시키고, 비활동은 value를 (일부 모델에서) **감소**시킬 수 있음

가장 흥미로운 모델은 세포들이 **대사 물질을 두고 경쟁**하는 모델(γ-system)입니다: 활성 세포가 비활성 세포의 비용으로 value를 얻어, 시스템의 **총 value가 일정**하게 유지됩니다.

**Value ($V$)** represents the output potency of each A-unit — the only variable that changes through learning, corresponding to modern neural network **weights**. It may represent amplitude, frequency, latency, or transmission probability. The most interesting model is one where cells **compete for metabolic resources** (γ-system): active cells gain value at the expense of inactive cells, keeping the system's **total value constant**.

---

### Three Learning Systems: α, β, γ (pp. 391–392) — 세 가지 학습 시스템

Rosenblatt은 value 동역학이 다른 세 가지 시스템을 비교합니다. Rosenblatt compares three systems that differ in their value dynamics:

| 특성 | α-System | β-System | γ-System |
|------|----------|----------|----------|
| **강화당 총 value 변화** | $N_{ar}$ (활성 유닛 수에 비례) | $K$ (상수) | $0$ (순 변화 없음) |
| **활성 A-unit의 ΔV** | $+1$ | $K / N_{ar}$ | $+1$ |
| **비활성 A-unit의 ΔV (source-set 외부)** | $0$ | $K / N_A$ | $0$ |
| **비활성 A-unit의 ΔV (dominant set 내부)** | $0$ | $0$ | $-N_{ar} / (N_{Ar} - N_{ar})$ |
| **시스템 평균 value** | 강화 횟수에 비례 증가 | 시간에 비례 증가 | **일정 (보존)** |

여기서:
- $N_{ar}$: source-set 중 활성화된 유닛 수
- $N_{Ar}$: source-set의 전체 유닛 수
- $K$: 임의 상수

**α-system (Uncompensated Gain System / 비보상 이득 시스템):**
가장 단순합니다. 활성 A-unit이 발화할 때마다 +1 value를 얻고, 이 gain을 무한히 보유합니다. 문제: 시스템의 총 value가 계속 증가하여, 신호 대 잡음비(SNR)가 저하됩니다.

The simplest system. Active A-units gain +1 value per firing and retain it indefinitely. Problem: total system value grows without bound, degrading signal-to-noise ratio.

**β-system (Constant Feed System / 일정 공급 시스템):**
각 source-set에 일정한 총 gain($K$)이 할당되고, 활성도에 비례하여 분배됩니다. 문제: α보다 더 나쁜데, 총 value가 계속 증가하면서 작은 통계적 차이가 증폭되어 불안정해집니다.

Each source-set receives a constant total gain ($K$), distributed among active units. Problem: even worse than α — growing total values amplify small statistical differences, causing instability.

**γ-system (Parasitic Gain System / 기생적 이득 시스템):**
활성 세포가 **비활성 세포로부터** value를 흡수합니다 — "기생적" 이득. 결과: source-set의 총 value가 **항상 일정**합니다. 이것이 가장 우수한 이유:
- 총 value가 보존되므로 과도한 value 축적이 없음
- sum-discriminating과 mean-discriminating 시스템의 성능이 동일해짐
- 학습량($n_{sr}$)이 변동하더라도 성능이 안정적

Active cells absorb value from **inactive** cells in their source-set — "parasitic" gain. Result: total source-set value remains **constant**. This is superior because: no runaway value accumulation, sum and mean systems perform identically, and performance is stable regardless of variation in $n_{sr}$.

**현대적 해석 / Modern Interpretation:** γ-system은 현대의 **competitive learning**, **winner-take-all** 네트워크, 그리고 **softmax 정규화**와 본질적으로 같은 원리입니다. 제한된 자원을 두고 뉴런들이 경쟁하면, 관련 있는 연결은 강화되고 관련 없는 연결은 자연스럽게 약해집니다. 또한 **weight decay**나 **normalization** 기법의 원형이기도 합니다.

The γ-system embodies the same principle as modern **competitive learning**, **winner-take-all** networks, and **softmax normalization**. When neurons compete for limited resources, relevant connections strengthen while irrelevant ones weaken — also a precursor to **weight decay** and **normalization** techniques.

---

### Response Phases: Predominant and Postdominant (pp. 392) — 반응 단계

시스템의 반응은 두 단계로 구분됩니다. The system's response is divided into two phases:

**1. 우세 전 단계 (Predominant Phase):**
- 자극이 주어지면 A-unit들의 일부가 활성화됨. Some fraction of A-units responds to the stimulus.
- R-unit들은 아직 비활성 — 아직 "결정"이 내려지지 않은 상태. R-units are still inactive — no "decision" yet.
- 이 단계에서의 활성화 패턴이 자극의 "표현(representation)". The activation pattern constitutes the stimulus representation.

**2. 우세 후 단계 (Postdominant Phase):**
- 하나의 R-unit이 우세해져서 다른 R-unit들을 억제. One R-unit becomes dominant, inhibiting all others.
- 억제된 R-unit의 source-set도 함께 억제됨. The source-sets of suppressed R-units are also inhibited.
- 처음에는 어떤 R-unit이 우세해질지 **무작위**이지만, 학습 후에는 **올바른 반응이 우세해짐**. Initially random, but after learning, the **correct** response tends to dominate.

핵심: 학습이란 "우세 후 단계에서 올바른 R-unit이 이기도록" value 분포를 조정하는 과정입니다. The essence: learning is the process of adjusting the value distribution so that the correct R-unit wins in the postdominant phase.

**두 가지 판별 시스템 / Two Discrimination Systems:**
- **μ-system (Mean-discriminating)**: 입력의 **평균 value**가 가장 큰 R-unit이 우세. The R-unit whose inputs have the greatest **mean value** wins.
- **Σ-system (Sum-discriminating)**: 입력의 **총 value**가 가장 큰 R-unit이 우세. The R-unit whose inputs have the greatest **total value** wins.

대부분의 경우 μ-system이 더 우수합니다 — 평균은 source-set 크기의 무작위 변동에 덜 민감하기 때문. 단, γ-system에서는 둘의 성능이 동일합니다. In most cases the μ-system is superior because means are less sensitive to random variation in source-set sizes. However, in the γ-system, both perform identically.

---

### Analysis of the Predominant Phase: $P_a$ and $P_c$ (pp. 392–394) — 우세 전 단계 분석

**$P_a$ — 활성화 확률 / Activation Probability:**

A-unit이 활성화되려면 흥분성 입력과 억제성 입력의 대수합이 임계값 이상이어야 합니다. An A-unit fires when the algebraic sum of excitatory and inhibitory inputs meets or exceeds the threshold:

$$a = e - i \geq \theta$$

A-unit이 활성화될 확률 $P_a$는:

$$P_a = \sum_{e,i} P(e,i) \cdot \mathbb{1}[e - i \geq \theta]$$

여기서 $P(e,i)$는 $x$개의 흥분성 연결과 $y$개의 억제성 연결 중 각각 $e$개와 $i$개가 자극에 의해 활성화될 확률입니다. $R$(자극이 차지하는 망막 비율)이 주어지면, $e$와 $i$는 각각 이항분포를 따릅니다.

**$P_a$의 주요 특성 (Fig. 4) / Key Properties of $P_a$:**
- $R$이 증가하면 $P_a$ 증가 (자극이 클수록 더 많은 A-unit 활성화). Larger stimuli activate more A-units.
- $\theta$가 증가하면 $P_a$ 감소 (높은 임계값은 활성화를 어렵게 함). Higher thresholds make activation harder.
- 억제 비율($y$)이 증가하면 $P_a$ 감소. More inhibitory connections reduce $P_a$.
- **핵심 / Key**: 흥분과 억제가 대략 균형($x \approx y$)일 때, $P_a$ 곡선이 **평탄화**됨 → 자극 크기에 덜 민감해짐. 이것은 시스템이 $P_a$를 최적값 근처에서 안정적으로 유지하는 데 유리합니다. When excitation roughly balances inhibition ($x \approx y$), the $P_a$ curve **flattens** — making the system less sensitive to stimulus size variations, which helps keep $P_a$ near optimal.

**$P_c$ — 공동 활성화 확률 / Conditional Co-activation Probability:**

$P_c$는 "자극 $S_1$에 반응한 A-unit이 다른 자극 $S_2$에도 반응할 확률"입니다. 이것은 두 자극의 **신경적 유사성**을 측정합니다. $P_c$ is the probability that an A-unit responding to stimulus $S_1$ also responds to $S_2$ — a measure of **neural similarity** between two stimuli.

$$P_c = P(\text{A-unit responds to } S_2 \mid \text{A-unit responds to } S_1)$$

상세한 공식(Equation 2)은 두 자극 사이의 겹침(overlap)을 나타내는 변수들 — $L$(잃는 origin points), $G$(얻는 origin points) — 로 표현됩니다.

**$P_c$의 주요 특성 (Fig. 5, 6) / Key Properties of $P_c$:**
- 두 자극이 완전히 겹치지 않아도($L = 1, G = 1$) $P_c > 0$ — A-unit의 origin points가 무작위이므로, 완전히 다른 자극이라도 일부 A-unit은 우연히 둘 다에 반응. Even for non-overlapping stimuli, $P_c > 0$ because random origin points mean some A-units respond to both by chance.
- 임계값($\theta$)이 높을수록 $P_c$가 급격히 감소 — 높은 임계값은 A-unit을 더 **선택적(selective)**으로 만듦. Higher thresholds sharply reduce $P_c$, making A-units more selective.
- 자극이 동일에 가까워질수록 $P_c → 1$. As stimuli approach identity, $P_c \to 1$.
- **최소값 / Minimum**: $P_{c,\min} = (1-L)^x(1-G)^y$ (임계값이 극도로 높을 때 / when threshold is extremely high).

**직관 / Intuition**: $P_c$는 perceptron이 **다른 자극을 얼마나 구분하는가**의 핵심 지표입니다. $P_c$가 높으면 두 자극을 비슷하게 취급하고(일반화), 낮으면 다르게 취급합니다(판별). $P_c$ is the key indicator of how the perceptron distinguishes stimuli. High $P_c$ means the two stimuli are treated as similar (generalization); low $P_c$ means they are treated as different (discrimination).

---

### Mathematical Analysis of Learning (pp. 394–401) — 학습의 수학적 분석

이것이 논문의 핵심 기여입니다. Rosenblatt은 모든 학습 상황을 하나의 통합 방정식으로 표현합니다. This is the paper's central contribution. Rosenblatt captures all learning situations in a single unified equation.

**두 가지 평가 실험 / Two Types of Evaluation:**

1. **$P_r$ (Probability of correct recall / 정확 재인 확률)**: 학습 시리즈와 **동일한** 자극으로 테스트 — "이전에 본 것을 기억하는가?" Test with the **same** stimuli from the learning series — "does it remember what it has seen?"
2. **$P_g$ (Probability of correct generalization / 정확 일반화 확률)**: 같은 **클래스**의 **새로운** 자극으로 테스트 — "본 적 없는 것도 분류할 수 있는가?" Test with **new** stimuli from the same classes — "can it classify what it has never seen?"

$P_g$가 가능하다는 것이 perceptron의 가장 강력한 주장입니다. That $P_g$ is achievable is the perceptron's most powerful claim.

**통합 학습 방정식 (Equation 4):**

$$P = P(N_{ar} > 0) \cdot \Phi(Z)$$

여기서:

$$P(N_{ar} > 0) = 1 - (1 - P_a)^{N_e}$$

이것은 "source-set에서 최소 하나의 effective A-unit이 활성화될 확률"입니다.
- $N_e$: effective A-unit 수 (두 반응 모두에 연결되지 않은 유닛 — 공통 유닛은 양쪽에 동일하게 기여하므로 판별에 무용)

$$Z = \frac{c_1 n_{sr} + c_2}{\sqrt{c_3^2 n_{sr} + c_4^2}}$$

- $\Phi(Z)$: 표준 정규분포의 누적 분포 함수 (CDF)
- $n_{sr}$: 각 반응에 연관된 자극의 수 (= 학습량)
- $c_1, c_2, c_3, c_4$: 시스템 유형, 환경, 물리적 파라미터에 의존하는 상수

**방정식의 해석 / Interpreting the Equation:**
- $Z > 0 \Rightarrow \Phi(Z) > 0.5$: 랜덤보다 나은 성능 / better than chance
- $Z \to \infty \Rightarrow \Phi(Z) \to 1$: 완벽한 성능 / perfect performance
- $Z = 0 \Rightarrow \Phi(Z) = 0.5$: 랜덤 수준 (일반화 불가) / chance level (no generalization)

**이상적 환경 (Ideal Environment):**

"이상적 환경"은 자극이 무작위 점의 집합인 경우입니다 — 고유한 클래스 구조 없음. 500개 자극은 $R_1$에, 다른 500개는 $R_2$에 임의로 할당됩니다. The "ideal environment" consists of random collections of illuminated points with no class structure. 500 stimuli are arbitrarily assigned to $R_1$ and another 500 to $R_2$.

이 경우 $c_1 = 0$이므로:

$$Z = \frac{c_2}{\sqrt{c_3^2 n_{sr} + c_4^2}}$$

$P_g$의 경우 $c_2 = 0$이기도 하므로 $Z = 0$, 따라서 $P_g = 0.5$ — **일반화가 불가능**합니다! 새로운 자극을 분류할 기반이 없기 때문에 당연합니다.

For $P_g$, $c_2 = 0$ as well, so $Z = 0$ and $P_g = 0.5$ — **generalization is impossible**. This is natural: there is no basis for classifying a stimulus that has never been seen.

$P_r$은 가능합니다: 이전에 본 특정 자극에 대해 value 차이가 축적되므로, 재인은 학습량에 따라 향상됩니다. 하지만 학습한 자극이 너무 많아지면 성능이 다시 랜덤으로 되돌아갑니다(결론 #2).

$P_r$ (recall) is possible: value differences accumulate for specific previously-seen stimuli. However, as the number of learned stimuli grows, performance reverts to chance (Conclusion #2).

**α-system의 학습 상수 (ideal environment, S-system):**

$$c_1 = 0, \quad c_2 = 1 - P_a, \quad c_3 = \sqrt{P_a(1-P_a)}, \quad c_4 = \sqrt{2(1-P_a) + \omega P_a}$$

여기서 $\omega$는 각 A-unit이 연결된 반응의 비율입니다.

**α-system의 학습 상수 (ideal environment, μ-system):**

$$c_3 = 0$$

$c_3 = 0$이 되면서 μ-system이 S-system보다 결정적으로 유리해집니다 — 분모에서 $n_{sr}$에 비례하는 분산 항이 사라지기 때문입니다.

**γ-system의 우수성 / Superiority of the γ-system:**

γ-system에서는 $n_{sr}$이 고정이든 변동이든 동일한 상수를 가지며, S-system과 μ-system의 성능이 같습니다. 이것은 **총 value 보존** 덕분입니다 — 무관한 활동에 의한 잡음 축적이 없으므로 가장 안정적입니다.

In the γ-system, the constants remain identical whether $n_{sr}$ is fixed or variable, and S-system and μ-system performances are equal. This is thanks to **total value conservation** — no noise accumulation from irrelevant activity, making it the most stable system.

---

### Differentiated Environment (pp. 400–402) — 차별화된 환경

**게임 체인저 — $c_1 \neq 0$ / The Game Changer:**

이상적 환경을 사각형, 원, 삼각형 같은 **구별 가능한 클래스**가 있는 환경으로 바꾸면, 모든 것이 달라집니다. $c_1$이 더 이상 0이 아니게 되어, $Z$의 분자에 $n_{sr}$에 비례하는 항이 생깁니다.

Replacing the ideal environment with one containing distinguishable classes (squares, circles, triangles) changes everything. $c_1$ is no longer zero, introducing a term proportional to $n_{sr}$ in the numerator of $Z$:

$$Z = \frac{c_1 n_{sr} + c_2}{\sqrt{c_3^2 n_{sr} + c_4^2}}$$

$n_{sr} \to \infty$이면:

$$Z \to \frac{c_1}{c_3} \cdot \sqrt{n_{sr}} \to \infty$$

따라서 $\Phi(Z) \to 1$로 수렴합니다! 학습을 충분히 하면 성능이 완벽에 가까워집니다. Therefore $\Phi(Z) \to 1$ — with enough training, performance approaches perfection!

**$P_{c\alpha\beta}$의 정의와 일반화의 조건:**

$$P_{c\alpha\beta} = E[P_c \text{ between random } S_\alpha \text{ and } S_\beta]$$

- $P_{c11}$: 같은 클래스(Class 1) 내 자극 쌍의 평균 공동 활성화 확률 → **클래스 내 유사성**
- $P_{c12}$: 다른 클래스(Class 1과 Class 2) 자극 간의 평균 공동 활성화 확률 → **클래스 간 유사성**
- $P_{c1x}$: Class 1과 모든 다른 클래스 자극 간의 평균

**일반화의 필요충분 조건:**

$$P_{c12} < P_a < P_{c11}$$

이 부등식의 의미 / Meaning of the inequality:
- $P_a < P_{c11}$: 같은 클래스의 자극들은 A-unit이 무작위로 활성화되는 것보다 **더 많은** A-unit을 공유 (클래스 내 일관성). Same-class stimuli share more A-units than chance (within-class consistency).
- $P_{c12} < P_a$: 다른 클래스의 자극들은 무작위 수준보다 **더 적은** A-unit을 공유 (클래스 간 구분). Different-class stimuli share fewer A-units than chance (between-class separation).

**현대적 해석 / Modern Interpretation:** 이것은 정확히 **intra-class variance < inter-class variance** 조건입니다. 같은 클래스의 데이터는 feature space에서 가깝고, 다른 클래스의 데이터는 멀어야 합니다. 현대의 contrastive learning(SimCLR, MoCo 등)이 최적화하는 것이 바로 이 조건입니다! This is exactly the **intra-class variance < inter-class variance** condition. Same-class data should be close in feature space, different-class data far apart — precisely what modern contrastive learning (SimCLR, MoCo, etc.) optimizes!

**핵심 결과 (Equation 9):**

$$P_{r,\infty} = P_{g,\infty}$$

충분한 학습 후, **재인 확률과 일반화 확률이 같은 점근값에 수렴**합니다. 즉, 무한히 학습하면, 이전에 본 자극이든 새로운 자극이든 성능 차이가 없습니다. 이것은 매우 강력한 결과입니다: perceptron은 단순히 "외운" 것이 아니라, 클래스의 **본질적 특성**을 포착합니다.

After sufficient learning, **recall probability and generalization probability converge to the same asymptote**. In the limit, it makes no difference whether the test stimulus was seen before or not. This is a powerful result: the perceptron does not merely memorize — it captures the **essential characteristics** of each class.

또한 시스템의 A-unit 수($N_A$)를 늘리면, 점근 한계가 급격히 1에 접근합니다. 수천 개의 A-unit이면 단순한 분류 문제에서 오류가 무시할 수 있는 수준이 됩니다.

As $N_A$ (number of A-units) increases, the asymptotic limit rapidly approaches unity. With several thousand cells, errors become negligible on simple problems.

---

### Binary Response Systems and Bivalent Reinforcement (pp. 402–404) — 이진 응답 시스템과 양가 강화

**이진 코딩 — 확장성 문제의 해결 / Binary Coding — Solving the Scalability Problem:**

반응 수가 증가하면 성능이 급격히 저하됩니다 — 모든 반응이 상호 배타적이기 때문. Rosenblatt의 해결책: **이진 코딩(binary coding)**. 100개 클래스를 100개의 상호 배타적 반응 대신, 7개의 이진 특성(bit)으로 표현합니다 ($2^7 = 128 > 100$). 각 bit는 독립적으로 학습 가능한 반응 쌍에 해당합니다.

As the number of responses increases, performance degrades sharply because all responses are mutually exclusive. Rosenblatt's solution: **binary coding**. Instead of 100 mutually exclusive responses for 100 classes, use 7 binary features (bits) ($2^7 = 128 > 100$). Each bit corresponds to an independently learnable response pair.

이것은 현대의 **multi-label classification**이나 **binary encoding** 접근법의 원형입니다. This is a precursor to modern **multi-label classification** and **binary encoding** approaches.

**양가(Bivalent) 강화 시스템 / Bivalent Reinforcement System:**

이전의 모든 시스템에서는 활성 A-unit의 value가 항상 **양의 방향**으로만 변했습니다. 양가 시스템에서는 **양성(positive)과 음성(negative) 강화**가 가능합니다:

In all previous systems, active A-unit values only changed in the **positive** direction. In bivalent systems, both **positive and negative reinforcement** are possible:

- 양성 강화 + "on" 반응의 활성 유닛 → $+\Delta V$ / Positive reinforcement + active "on" units → $+\Delta V$
- 양성 강화 + "off" 반응의 활성 유닛 → $-\Delta V$ / Positive reinforcement + active "off" units → $-\Delta V$
- 음성 강화 → 위의 반대 / Negative reinforcement → the reverse

이것은 본질적으로 **reward와 punishment**에 해당하며, **trial-and-error learning**을 가능하게 합니다. 현대의 reinforcement learning에서 reward signal이 weight를 양방향으로 조정하는 것과 같은 원리입니다.

This is essentially **reward and punishment**, enabling **trial-and-error learning**. It parallels how reward signals in modern reinforcement learning adjust weights bidirectionally.

Rosenblatt은 이 시스템을 IBM 704 컴퓨터에서 시뮬레이션하여 이론적 예측을 검증했다고 보고합니다. Rosenblatt reports validating theoretical predictions through simulation experiments on the IBM 704 computer at Cornell Aeronautical Laboratory.

---

### Improved Perceptrons and Spontaneous Organization (pp. 404–405) — 개선된 Perceptron과 자발적 조직화

**시간적 패턴 인식 / Temporal Pattern Recognition:**
이전까지의 분석은 "순간 자극(momentary stimulus)" perceptron에 한정되었지만, A-unit의 임계값이 이전 활동에 의해 일시적으로 변하는 모델로 확장하면, 시간적 패턴(속도, 음의 시퀀스 등)도 인식할 수 있습니다.

The analysis so far was limited to "momentary stimulus" perceptrons. By extending to models where A-unit thresholds are temporarily altered by prior activity, the system can also recognize temporal patterns (velocities, sound sequences, etc.).

**윤곽 감지 개선 / Contour Detection Improvement:**
투사 영역(A_I)의 origin points를 공간적으로 제약하면(Fig. 1의 구조), A-unit이 윤곽(contour)에 민감해져 성능이 향상됩니다.

By constraining the spatial distribution of origin points in the projection area (A_I), A-units become sensitive to contours, improving performance.

**자발적 개념 형성 / Spontaneous Concept Formation:**

이것은 논문에서 가장 놀라운 결과 중 하나입니다: A-unit의 value가 크기에 비례하여 감쇠(decay)하면, perceptron은 **명시적 피드백 없이도** 자극의 클래스를 자발적으로 발견합니다. 두 가지 다른 클래스의 자극을 무작위로 보여주고, 모든 반응에 자동 강화(정답/오답 구분 없이)를 적용하면, 시스템은 결국 한 클래스에는 "1", 다른 클래스에는 "0"으로 반응하는 안정 상태에 수렴합니다.

One of the paper's most surprising results: if A-unit values **decay** at a rate proportional to their magnitude, the perceptron **spontaneously discovers** stimulus classes without explicit feedback. When exposed to stimuli from two dissimilar classes with automatic (non-discriminative) reinforcement, the system converges to a stable state where it responds "1" for one class and "0" for the other.

이것은 현대의 **unsupervised clustering**이나 **self-organizing maps**의 원형입니다! 시스템이 외부 레이블 없이도 데이터의 구조를 발견할 수 있다는 것입니다. This is a precursor to modern **unsupervised clustering** and **self-organizing maps** — the system discovers data structure without external labels!

---

### Capabilities and Limits (pp. 404–405) — 능력과 한계

**Perceptron이 할 수 있는 것 / What the Perceptron Can Do:**
- 패턴 인식 (pattern recognition)
- 연상 학습 (associative learning)
- 선택적 주의 (selective attention)
- 선택적 회상 (selective recall)
- 시간적/공간적 패턴 인식 (temporal/spatial pattern recognition)
- 시행착오 학습 (trial-and-error learning)
- 자발적 클래스 인식 (spontaneous class recognition)

**Perceptron이 할 수 없는 것 — Rosenblatt 자신의 한계 인식 / What It Cannot Do — Rosenblatt's Own Acknowledgment:**

> "The limit of the perceptron's capabilities seems to lie in the area of relative judgment, and the abstraction of relationships."

관계 판단(relational judgment)과 관계 추상화(abstraction of relationships)가 한계입니다. 구체적 예:

The limits lie in relational judgment and abstraction of relationships. Concrete examples:

- ✅ "왼쪽의 자극이면 색상을, 오른쪽이면 형태를 이름 붙여라" → 가능 (동시적 조건부 반응). "Name the color if the stimulus is on the left, the shape if on the right" → possible (simultaneous conditional response).
- ❌ "사각형의 왼쪽에 있는 물체를 이름 붙여라" → 불가능 (자극 간 **관계** 파악 필요). "Name the object left of the square" → impossible (requires recognizing **relationships** between stimuli).
- ❌ "원 앞에 나타난 패턴을 지적하라" → 불가능 (시간적 **관계** 파악 필요). "Indicate the pattern that appeared before the circle" → impossible (requires temporal **relationships**).

Rosenblatt은 Goldstein의 뇌 손상 환자와 유사하다고 비유합니다: 구체적이고 절대적인 자극에는 반응할 수 있지만, 관계적이고 추상적인 판단은 할 수 없습니다. Rosenblatt draws an analogy to Goldstein's brain-damaged patients: they can respond to concrete, absolute stimuli but cannot make relational or abstract judgments.

**현대적 해석 / Modern Interpretation:** 이것은 정확히 Minsky & Papert(1969)가 공식적으로 증명하게 될 **선형 분리 불가능성(linear inseparability)** 문제입니다. XOR 문제가 가장 유명한 예인데, 단층 perceptron은 입력 간의 관계(AND와 OR의 조합)를 포착할 수 없습니다. 이 한계의 해결은 #6 Rumelhart et al.(1986)의 backpropagation을 기다려야 합니다. This is precisely the **linear inseparability** problem that Minsky & Papert (1969) would formally prove. The XOR problem is the most famous example — a single-layer perceptron cannot capture relationships between inputs (combinations of AND and OR). Overcoming this limit had to wait for #6 Rumelhart et al. (1986) and backpropagation.

---

### Conclusions and Evaluation (pp. 405–408) — 결론과 평가

**10가지 결론 요약:**

| # | 결론 | 현대적 의의 |
|---|------|-------------|
| 1 | 무작위 환경에서도 특정 자극에 대한 학습 가능 | 기본적인 memorization |
| 2 | 학습 자극 수가 증가하면 성능이 랜덤으로 회귀 | Capacity 한계 |
| 3 | 무작위 환경에서는 일반화 불가 | 구조 없이는 학습 무의미 |
| 4 | 차별화된 환경에서 정확 재인은 점근적으로 향상 | Supervised learning |
| 5 | **$P_r$과 $P_g$가 같은 점근값에 수렴** | 진정한 일반화 |
| 6 | 이진 코딩과 윤곽 감지로 성능 향상 가능 | Feature engineering |
| 7 | 양가 시스템으로 시행착오 학습 가능 | Reinforcement learning |
| 8 | 시간적 패턴도 학습 가능 | Temporal processing |
| 9 | **기억은 분산적** — 일부 세포 제거해도 전체 연상에 약간의 결손만 발생 | Distributed representations |
| 10 | 자발적 클래스 인식 가능, 하지만 관계 인식은 한계 | Unsupervised learning + limits |

**6개의 물리적 파라미터 — 이론의 힘 / Six Physical Parameters — The Power of the Theory:**

Rosenblatt은 perceptron의 모든 학습 행동을 단 6개의 독립적으로 측정 가능한 파라미터로 예측할 수 있다고 주장합니다. Rosenblatt claims that all learning behavior can be predicted from just six independently measurable physical parameters:

| 파라미터 | 의미 |
|----------|------|
| $x$ | A-unit당 흥분성 연결 수 |
| $y$ | A-unit당 억제성 연결 수 |
| $\theta$ | A-unit의 임계값 |
| $\omega$ | A-unit이 연결된 R-unit의 비율 |
| $N_A$ | 시스템의 A-unit 수 |
| $N_R$ | 시스템의 R-unit 수 |

**기존 학습 이론과의 차별점 — 3가지 장점 / Three Advantages Over Previous Learning Theories:**

1. **절약성 (Parsimony)**: 하나의 가설 변수($V$, value)만 도입. 나머지는 이미 물리/생물학에 존재하는 변수. Only one hypothetical variable ($V$, value) is postulated; everything else is already present in physics and biology.
2. **검증 가능성 (Verifiability)**: 기존 학습 이론(Hull, Bush & Mosteller)은 "행동에서 상수를 추출하여 다른 행동을 예측" — 본질적으로 **곡선 맞춤(curve fitting)**. Rosenblatt의 이론은 **물리적 변수에서 행동을 예측** — 독립적 측정으로 검증 가능. Previous theories extracted constants from behavior to predict other behavior — essentially **curve fitting**. Rosenblatt's theory predicts behavior **from physical variables** — testable via independent measurement.
3. **설명력과 일반성 (Explanatory power)**: 특정 유기체나 상황에 한정되지 않음. 물리적 파라미터가 알려진 **어떤 시스템**에도 적용 가능 — 생물학적 뇌든 인공 기계든. Not specific to any organism or situation — applicable to **any system** whose physical parameters are known, whether biological brain or artificial machine.

Rosenblatt은 Hebb(1949)의 작업이 철학적으로 가장 가까운 선행 연구이지만, Hebb는 "이런 종류의 기제가 뇌에 있을 것이다"라는 정성적 제안에 그쳤다고 평가합니다. Perceptron 이론은 **"생리학적 변수에서 학습 곡선을 예측하고, 학습 곡선에서 생리학적 변수를 예측하는" 최초의 양방향 다리**라고 주장합니다.

Rosenblatt acknowledges Hebb (1949) as the closest philosophical predecessor, but notes that Hebb never achieved a model that could actually predict behavior from physiology. The perceptron theory represents **"the first actual completion of a bridge"** between physiology and behavior — predicting learning curves from neurological variables and vice versa.

---

## Key Takeaways / 핵심 시사점

1. **기계는 명시적 프로그래밍 없이도 데이터로부터 학습할 수 있다.** Rosenblatt은 McCulloch-Pitts의 고정 회로를 **적응적 시스템**으로 확장하여, 경험이 연결 강도를 변화시키고 이것이 패턴 분류 능력을 낳는다는 것을 수학적으로 보였습니다. 이것은 전체 machine learning 분야의 존재 정당화입니다.

   Machines can learn from data without explicit programming. Rosenblatt extended the fixed M-P neuron into an adaptive system, mathematically proving that experience-driven weight changes enable pattern classification — the foundational justification for all of machine learning.

2. **확률론은 불완전한 시스템을 분석하는 올바른 언어이다.** Boolean 논리가 정확한 배선을 전제하는 반면, 확률론은 "대부분의 무작위 배선에서 이 시스템이 어떻게 행동할 것인가?"를 물을 수 있습니다. 이것은 현대 statistical learning theory의 직접적 조상입니다.

   Probability theory is the right language for imperfect systems. While Boolean logic presumes exact wiring, probability allows asking "how will this system behave across most random wirings?" — a direct ancestor of modern statistical learning theory.

3. **일반화는 데이터의 구조가 있을 때만 가능하다.** 무작위 환경에서 perceptron은 외울 수는 있지만 일반화할 수 없습니다. $P_{c12} < P_a < P_{c11}$ 조건 — 즉 클래스 내 유사성이 클래스 간 유사성보다 커야 한다는 것 — 은 현대 representation learning과 contrastive learning의 근본 원리입니다.

   Generalization requires structure in the data. The condition $P_{c12} < P_a < P_{c11}$ (intra-class similarity > inter-class similarity) is the fundamental principle behind modern representation learning and contrastive learning.

4. **경쟁적 학습(γ-system)이 단순 축적(α, β)보다 우월하다.** 제한된 자원을 두고 뉴런이 경쟁하면, 관련 연결은 강화되고 무관한 연결은 약화됩니다. 이것은 현대의 weight normalization, dropout, competitive learning의 원형입니다.

   Competitive learning (γ-system) outperforms simple accumulation (α, β). When neurons compete for limited resources, relevant connections strengthen while irrelevant ones weaken — presaging weight normalization, dropout, and modern competitive learning.

5. **기억은 분산적(distributed)으로 저장된다.** 특정 기억이 특정 세포에 국소화되지 않고, 많은 세포에 걸쳐 저장됩니다. 일부 세포를 제거해도 특정 기억이 완전히 사라지지 않고, 모든 기억에 대한 약간의 전반적 결손이 나타납니다. 이것은 현대 신경망의 distributed representation의 원형이며, graceful degradation 속성입니다.

   Memory is distributed, not localized. Removing some cells produces a general deficit across all associations rather than destroying any single memory — the prototype of modern distributed representations and graceful degradation.

6. **Perceptron에는 근본적 한계가 있다 — 관계 추상화 불가.** Rosenblatt 자신이 인정했듯이, perceptron은 자극 간의 관계를 추상화할 수 없습니다. 이것은 단층 구조의 본질적 한계이며, Minsky & Papert(1969)가 공식 증명하고, Rumelhart et al.(1986)의 backpropagation이 다층 구조로 해결하게 됩니다.

   The perceptron has a fundamental limit — it cannot abstract relationships. As Rosenblatt himself acknowledged, single-layer perceptrons cannot recognize relationships between stimuli. This is the limit that Minsky & Papert (1969) formally proved and that Rumelhart et al. (1986) overcame with multi-layer backpropagation.

7. **자발적 개념 형성 — 비지도 학습의 시작.** Value 감쇠(decay)가 있는 perceptron은 명시적 레이블 없이도 자극의 클래스를 자발적으로 발견합니다. 이것은 unsupervised learning, self-organizing maps, 그리고 현대의 contrastive self-supervised learning의 가장 초기 형태입니다.

   Spontaneous concept formation — the birth of unsupervised learning. A perceptron with value decay spontaneously discovers stimulus classes without explicit labels — the earliest form of unsupervised learning, self-organizing maps, and modern self-supervised learning.

---

## Mathematical Summary / 수학적 요약

### Perceptron Architecture / 아키텍처

```
Input: S-points (sensory retina, N_S units)
         │
         ▼  [random connections, x excitatory + y inhibitory per A-unit]
Hidden: A-units (N_A association units, threshold θ)
         │
         ▼  [random connections, mutual inhibition between R-units]
Output: R-units (N_R response units)
```

### A-unit Activation Rule / 활성화 규칙

$$\text{A-unit fires} \iff \sum_{j \in \text{excit}} x_j - \sum_{k \in \text{inhib}} x_k \geq \theta$$

### Value Update Rules / 학습 규칙

**α-system (uncompensated gain):**
$$V_i \leftarrow V_i + 1 \quad \text{if A-unit } i \text{ is active during reinforcement}$$

**β-system (constant feed):**
$$V_i \leftarrow V_i + \frac{K}{N_{ar}} \quad \text{(active units share constant gain } K\text{)}$$

**γ-system (parasitic gain, conserved total value):**
$$V_i \leftarrow V_i + 1 \quad \text{(active)}$$
$$V_j \leftarrow V_j - \frac{N_{ar}}{N_{Ar} - N_{ar}} \quad \text{(inactive in same source-set)}$$
$$\sum_{i \in \text{source-set}} V_i = \text{constant}$$

### Key Probabilities / 핵심 확률

**Activation probability:**
$$P_a = P(e - i \geq \theta) = \sum_{e=0}^{x} \sum_{i=0}^{y} \binom{x}{e} R^e(1-R)^{x-e} \binom{y}{i} R^i(1-R)^{y-i} \cdot \mathbb{1}[e-i \geq \theta]$$

**Co-activation probability:**
$$P_c = P(\text{responds to } S_2 \mid \text{responds to } S_1) \quad \text{(function of overlap } L, G\text{)}$$

### Universal Learning Equation / 통합 학습 방정식

$$P = \left[1 - (1 - P_a)^{N_e}\right] \cdot \Phi\!\left(\frac{c_1 n_{sr} + c_2}{\sqrt{c_3^2 n_{sr} + c_4^2}}\right)$$

### Generalization Condition / 일반화 조건

$$P_{c12} < P_a < P_{c11}$$

### Asymptotic Equivalence / 점근적 동치

$$\lim_{n_{sr} \to \infty} P_r = \lim_{n_{sr} \to \infty} P_g \quad \text{(in differentiated environments)}$$

---

## Paper in the Arc of History / 역사적 맥락의 타임라인

```
1943  McCulloch & Pitts ── 최초 인공 뉴런 (고정 가중치, 논리 연산)
  │
1949  Hebb ── "함께 발화하면 연결 강화" (학습 규칙의 최초 제안, 미구현)
  │
1950  Turing ── "기계가 생각/학습할 수 있는가?" (철학적 정당화)
  │
1956  Dartmouth Conference ── AI 분야 공식 탄생
  │
1957  Rosenblatt ── Mark I Perceptron 하드웨어 구축 (해군 지원)
  │
1958  ★ ROSENBLATT ── "THE PERCEPTRON" ★
  │     최초의 학습 가능 신경망, 확률론적 분석, 일반화 증명
  │
1960  Widrow & Hoff ── ADALINE (Least Mean Squares, 연속 value 조정)
  │
1962  Rosenblatt ── "Principles of Neurodynamics" (perceptron 이론의 확장)
  │
1969  Minsky & Papert ── "Perceptrons" (XOR 등 한계 증명 → 첫 AI 겨울)
  │     ···· 신경망 암흑기 (1970s) ····
  │
1982  Hopfield ── 에너지 기반 신경망 (부활의 시작)
  │
1986  Rumelhart et al. ── Backpropagation (다층 perceptron으로 한계 극복)
  │
2017  Vaswani et al. ── Transformer (perceptron의 후손, 현재 AI의 기초)
```

---

## Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Relationship / 관계 |
|---|---|
| **#1 McCulloch & Pitts (1943)** | 직접적 전신. M-P 뉴런의 고정 가중치를 **학습 가능한 value**로 확장. S-points와 A-units의 활성화 규칙은 M-P 뉴런과 동일 |
| **#2 Turing (1950)** | 철학적 기반. Turing의 "학습하는 어린이 기계" 제안을 구체적 메커니즘으로 실현. Perceptron은 Turing이 예견한 "경험으로 학습하는 기계"의 최초 구현 |
| **Hebb (1949)** | 이론적 영감. "함께 발화하면 연결 강화"라는 Hebbian learning의 정량적 실현. Rosenblatt의 α-system은 Hebb의 학습 규칙을 공식화한 것 |
| **#4 Minsky & Papert (1969)** | 직접적 비판자. Rosenblatt이 인정한 "관계 추상화 불가" 한계를 형식적으로 증명 (XOR 문제). 신경망 연구에 10년 이상의 "겨울"을 초래 |
| **#5 Hopfield (1982)** | 부활의 시작. Perceptron의 "분산 기억"과 "에너지 함수" 개념을 recurrent 네트워크로 발전. 물리학의 에너지 최소화를 신경망에 적용 |
| **#6 Rumelhart et al. (1986)** | Perceptron의 한계를 극복. 다층 네트워크 + backpropagation = 관계 추상화(XOR 등) 해결. Rosenblatt의 3층 구조를 N층으로 확장하고 gradient 기반 학습 도입 |
| **#8 Cortes & Vapnik (1995)** | SVM은 perceptron의 "최적" 버전. 최대 마진으로 선형 분리 → 커널 트릭으로 비선형 확장. Rosenblatt의 statistical separability를 VC 이론으로 엄밀화 |

---

## References / 참고문헌

- Rosenblatt, F., "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain", *Psychological Review*, Vol. 65, No. 6, pp. 386–408, 1958.
- Rosenblatt, F., "The Perceptron: A Theory of Statistical Separability in Cognitive Systems", Cornell Aeronautical Laboratory, Inc. Rep. No. VG-1196-G-1, 1958.
- McCulloch, W. S. & Pitts, W., "A Logical Calculus of the Ideas Immanent in Nervous Activity", *Bull. Math. Biophysics*, Vol. 5, pp. 115–133, 1943.
- Hebb, D. O., *The Organization of Behavior*, Wiley, 1949.
- Hayek, F. A., *The Sensory Order*, Univ. of Chicago Press, 1952.
- Ashby, W. R., *Design for a Brain*, Wiley, 1952.
- Minsky, M. & Papert, S., *Perceptrons*, MIT Press, 1969.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J., "Learning Representations by Back-propagating Errors", *Nature*, Vol. 323, pp. 533–536, 1986.
