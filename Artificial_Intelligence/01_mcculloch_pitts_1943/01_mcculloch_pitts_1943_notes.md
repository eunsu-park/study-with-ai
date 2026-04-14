---
title: "A Logical Calculus of the Ideas Immanent in Nervous Activity"
authors: Warren S. McCulloch, Walter Pitts
year: 1943
journal: "Bulletin of Mathematical Biophysics, Vol. 5, pp. 115–133"
topic: Artificial Intelligence / Foundations of Neural Networks
tags: [neuron model, boolean logic, threshold logic, computational neuroscience, propositional logic, Turing equivalence]
status: completed
date_started: 2026-04-03
date_completed: 2026-04-03
---

# Paper #1: A Logical Calculus of the Ideas Immanent in Nervous Activity

# 논문 #1: 신경 활동에 내재한 아이디어의 논리적 미적분

---

## 1. Core Contribution / 핵심 기여

McCulloch and Pitts demonstrated that biological neurons, simplified as binary threshold units operating in discrete time, can be rigorously described using propositional logic. They proved that any logical function (AND, OR, NOT, and their combinations) can be realized by a network of such neurons (Theorems 1–2), and that networks containing feedback loops (circles) are computationally equivalent to Turing machines — making the nervous system, in principle, a universal computer. This paper established the theoretical foundation for all artificial neural networks.

McCulloch와 Pitts는 이산 시간에서 작동하는 이진 임계값 유닛으로 단순화된 생물학적 뉴런이 명제 논리를 사용하여 엄밀하게 기술될 수 있음을 보여주었습니다. 모든 논리 함수(AND, OR, NOT 및 그 조합)가 이러한 뉴런의 네트워크에 의해 실현될 수 있음을 증명했으며(정리 1–2), 피드백 루프(원)를 포함하는 네트워크가 Turing machine과 계산적으로 동등함을 보여주었습니다 — 신경계가 원리적으로 범용 컴퓨터라는 것입니다. 이 논문은 모든 인공 신경망의 이론적 기초를 확립했습니다.

---

## 2. Reading Notes / 읽기 노트

### Section 1: Introduction / 도입부 (pp. 99–101)

**저자 소개 — 놀라운 협업 / The Authors — A Remarkable Collaboration:**

이 논문은 과학사에서 가장 이례적인 협업 중 하나에서 탄생했습니다. **Warren S. McCulloch** (1898–1969)는 당시 45세의 노련한 신경생리학자이자 철학자로, Yale과 Columbia에서 교육받았으며, 신경계의 본질에 대해 20년 이상 깊이 고민해온 인물이었습니다. 그는 Leibniz의 오랜 꿈 — 정신 활동을 형식적 계산으로 환원하는 것 — 을 실현하고자 했습니다. **Walter Pitts** (1923–1969)는 당시 겨우 18세의 독학 수학 천재로, 가정 형편이 어려워 정규 교육을 거의 받지 못했지만 12세에 Russell과 Whitehead의 *Principia Mathematica*를 독파했다고 전해집니다. Pitts는 시카고 대학 근처를 배회하다가 McCulloch를 만났고, McCulloch는 그를 자신의 집에 들여 함께 연구했습니다. 경험 많은 신경과학자의 생물학적 직관과 젊은 수학자의 형식적 엄밀함이 결합된 이 협업이야말로 이 획기적인 논문을 가능하게 했습니다.

This paper emerged from one of the most unusual collaborations in the history of science. **Warren S. McCulloch** (1898–1969) was a 45-year-old seasoned neurophysiologist and philosopher, educated at Yale and Columbia, who had spent over 20 years pondering the nature of the nervous system. He sought to realize Leibniz's old dream — reducing mental activity to formal computation. **Walter Pitts** (1923–1969) was an 18-year-old self-taught mathematical prodigy who, despite lacking formal education due to a difficult home life, reportedly read through Russell and Whitehead's *Principia Mathematica* at age 12. Pitts was found wandering near the University of Chicago, where McCulloch took him in and they began working together. It was precisely this combination — the biological intuition of an experienced neuroscientist and the formal rigor of a young mathematician — that made this groundbreaking paper possible.

**1943년 신경과학의 상태 / The State of Neuroscience in 1943:**

1943년 당시, 뉴런에 대한 지식은 상당히 제한적이었지만 몇 가지 핵심 사실은 확립되어 있었습니다. Ramón y Cajal (Nobel Prize, 1906)이 뉴런설(neuron doctrine)을 확립하여 신경계가 개별 세포(뉴런)로 구성된다는 것이 알려져 있었고, Sherrington (Nobel Prize, 1932)이 시냅스의 개념을 도입하고 흥분성/억제성 시냅스의 구분을 확립했습니다. Adrian (Nobel Prize, 1932)은 신경 impulse의 "all-or-none" 성질을 실험적으로 입증했습니다. 그러나 시냅스 전달의 화학적 메커니즘(neurotransmitter)은 아직 완전히 이해되지 않았고, 이온 채널의 작동 원리는 Hodgkin & Huxley (1952)에 의해 수학적으로 기술되기 거의 10년 전이었습니다. McCulloch와 Pitts는 이 당시 알려진 사실들 — 특히 all-or-none 법칙과 시냅스 지연 — 을 최대한 활용하여 모델을 구축했습니다.

In 1943, knowledge about neurons was considerable but limited. Ramón y Cajal (Nobel Prize, 1906) had established the neuron doctrine — that the nervous system is composed of discrete cells (neurons). Sherrington (Nobel Prize, 1932) had introduced the concept of the synapse and established the distinction between excitatory and inhibitory synapses. Adrian (Nobel Prize, 1932) had experimentally demonstrated the "all-or-none" property of nerve impulses. However, the chemical mechanism of synaptic transmission (neurotransmitters) was not yet fully understood, and the mathematical description of ion channel function was nearly a decade away (Hodgkin & Huxley, 1952). McCulloch and Pitts built their model by maximally leveraging what was known — especially the all-or-none law and synaptic delay.

**신경생리학적 배경 / Neurophysiological Background:**

논문은 신경계의 기본 구조를 설명하며 시작합니다:

- 신경계는 뉴런의 네트워크이며, 각 뉴런은 세포체(soma)와 축삭(axon)을 가집니다.
  The nervous system is a net of neurons, each having a soma and an axon.
- 시냅스(synapse)는 한 뉴런의 axon과 다른 뉴런의 soma 사이의 연결(adjunction)입니다.
  Synapses are always between the axon of one neuron and the soma of another.
- 각 뉴런에는 **threshold(임계값)**가 있으며, 흥분이 이를 초과해야 impulse가 시작됩니다.
  Each neuron has a threshold which excitation must exceed to initiate an impulse.
- 임펄스는 **"all-or-none"** — 뉴런이 완전히 발화하거나 전혀 발화하지 않습니다.
  The impulse is "all-or-none": fires completely or not at all.
- 발화 후 뉴런은 **불응기(refractory period, ~0.5 ms)** 동안 재발화할 수 없습니다.
  After firing, a neuron enters a refractory period (~0.5 ms).
- **Synaptic delay**: 입력 도달과 뉴런 반응 사이에 > 0.5 ms의 지연이 있습니다.
  There is a synaptic delay of > 0.5 ms between input arrival and neuron response.

**두 가지 시냅스 유형 / Two Types of Synapses:**

| 유형 / Type | 효과 / Effect |
|-------------|-------------|
| Excitatory (흥분성) | 발화를 촉진 — threshold에 양으로 기여 / Promotes firing — contributes positively toward threshold |
| Inhibitory (억제성) | 발화를 절대적으로 차단 — 하나라도 활성화되면 발화 불가 / Absolutely prevents firing — a single active inhibitory synapse blocks firing |

**핵심 통찰 / Key Insight — Neuron as Proposition:**

뉴런의 활동이 all-or-none이므로, 임의의 시점에서 뉴런의 활동은 **명제(proposition)**로 표현할 수 있습니다 (true = 발화, false = 미발화). 신경 활동 간의 관계는 명제 논리의 관계에 대응합니다.

Since neural activity is all-or-none, the activity of any neuron at any moment can be represented as a proposition (true = firing, false = not firing). Relations among neural activities correspond to relations among propositions in logic.

**왜 "all-or-none"이 결정적인가 / Why "All-or-None" Was the Critical Insight:**

"All-or-none" 성질은 이 논문 전체를 가능하게 만든 핵심 연결고리입니다. 만약 뉴런이 아날로그 신호(연속적인 전압 수준)를 전달했다면, 이를 명제 논리로 매핑하는 것은 불가능했을 것입니다. 그러나 뉴런이 "발화한다" 또는 "발화하지 않는다"라는 두 상태만 가진다는 것은 — 이것이 true/false, 1/0, 그리고 Boolean 논리의 세계와 정확히 대응한다는 것을 의미합니다. 이 이진(binary) 성질 덕분에 McCulloch와 Pitts는 Boole의 논리 대수(1854)와 Russell & Whitehead의 *Principia Mathematica* (1910)에서 이미 잘 발달된 수학적 도구를 신경 활동의 분석에 직접 적용할 수 있었습니다. 현대 디지털 컴퓨터 역시 이진 논리에 기반하므로, 이 연결은 "뇌는 일종의 컴퓨터다"라는 아이디어의 첫 번째 형식화이기도 합니다.

The "all-or-none" property is the critical link that made this entire paper possible. If neurons had transmitted analog signals (continuous voltage levels), mapping them to propositional logic would have been impossible. But the fact that a neuron has only two states — "fires" or "does not fire" — means it corresponds exactly to true/false, 1/0, and the world of Boolean logic. Thanks to this binary property, McCulloch and Pitts could directly apply the well-developed mathematical tools from Boole's algebra of logic (1854) and Russell & Whitehead's *Principia Mathematica* (1910) to the analysis of neural activity. Since modern digital computers are also based on binary logic, this connection also represents the first formalization of the idea that "the brain is a kind of computer."

논문은 **facilitation(촉진)**과 **extinction(소멸)**, 그리고 **learning(학습)**의 문제를 인정하지만, 이러한 변화를 겪는 네트워크를 연결과 임계값이 변하지 않는 동등한 가상 네트워크로 대체할 수 있다고 주장합니다.

The paper acknowledges facilitation, extinction, and learning, but argues that nets undergoing these alterations can be substituted by equivalent fictitious nets with unaltered connections and thresholds.

### Section 2: Nets Without Circles / 원이 없는 네트워크 (pp. 101–108)

**다섯 가지 기본 가정 / Five Fundamental Assumptions:**

1. 뉴런의 활동은 "all-or-none" 과정이다.
   The activity of the neuron is an "all-or-none" process.
2. 일정한 고정 수의 시냅스가 잠복 가산 기간 내에 흥분되어야 뉴런을 흥분시킨다 (이 수 = threshold $\theta$).
   A fixed number of synapses must be excited within the period of latent addition to excite a neuron (this number = threshold $\theta$).
3. 신경계 내의 유일한 유의미한 지연은 synaptic delay이다.
   The only significant delay is synaptic delay.
4. 억제성 시냅스의 활동은 그 시점에서 뉴런의 흥분을 **절대적으로** 방지한다.
   The activity of any inhibitory synapse **absolutely** prevents excitation at that time.
5. 네트워크의 구조는 시간에 따라 변하지 않는다 (학습 없음!).
   The structure of the net does not change with time (no learning!).

**표기법 / Notation:**

- 뉴런: $c_1, c_2, \ldots, c_n$
- $N_i(t)$: 뉴런 $c_i$가 시간 $t$에서 발화한다는 명제 / Proposition that neuron $c_i$ fires at time $t$
- $N_i$: 뉴런 $c_i$의 **action**
- Peripheral afferents: 외부 세계로부터의 입력 뉴런 / Input neurons from the external world
- Functor $S$: 시간을 한 단계 이동시키는 연산자 / Operator that shifts time by one step
  - $S(P)(t) \equiv P(t-1)$

**핵심 수식 — 뉴런의 동작 / Key Equation — Neuron Behavior (Eq. 1):**

$$N_i(z_1) \equiv S\left\{\prod_{m=1}^{q} \sim N_{j_m}(z_1) \cdot \sum_{\alpha \in \kappa_i} \prod_{s \in \alpha} N_{i_s}(z_1)\right\}$$

이 수식의 의미:
- $\prod_{m=1}^{q} \sim N_{j_m}$: 모든 inhibitory 입력이 비활성 (하나라도 활성이면 전체가 0)
- $\sum_{\alpha \in \kappa_i} \prod_{s \in \alpha} N_{i_s}$: threshold를 넘는 excitatory 입력 조합 중 하나라도 활성
- $S$: 한 시간 단계 지연 (synaptic delay)

Meaning: The neuron fires if no inhibitory inputs are active AND at least one combination of excitatory inputs exceeds the threshold, all evaluated one time step earlier.

**Temporal Propositional Expression (TPE) / 시간적 명제 표현:**

TPE는 재귀적으로 정의됩니다:
1. $p(z_1)$은 TPE (술어 변수)
2. $S_1$과 $S_2$가 TPE이면, $SS_1$, $S_1 \vee S_2$, $S_1 \cdot S_2$, $S_1 \cdot \sim S_2$도 TPE
3. 그 외에는 TPE가 아님

TPE is defined recursively: (1) a predicate variable is a TPE; (2) applying precession $S$, disjunction, conjunction, or conjoined negation to TPEs yields TPEs; (3) nothing else is a TPE.

**핵심 정리들 / Key Theorems:**

**Theorem 1**: 모든 order 0 (원이 없는) 네트워크는 TPE로 풀 수 있다.
Every net of order 0 can be solved in terms of temporal propositional expressions.

**Theorem 2**: 모든 TPE는 order 0 네트워크로 실현 가능하다.
Every TPE is realizable by a net of order zero.

이 두 정리는 함께 **feed-forward 네트워크와 TPE 사이의 완전한 대응**을 확립합니다.
Together, these two theorems establish a complete correspondence between feed-forward networks and TPEs.

**Theorem 3**: 복합 문장 $S_1$이 TPE인 것은, 모든 구성 명제를 거짓으로 놓았을 때 $S_1$이 거짓인 경우에만 그러하다 (Hilbert 선언 정규형에서 부정된 항만으로 이루어진 항이 없는 경우).
A complex sentence $S_1$ is a TPE if and only if it is false when all its constituents are assumed false.

**구체적 예시: AND 게이트 단계별 분석 / Concrete Example: AND Gate Step-by-Step:**

AND 게이트를 구체적으로 추적해봅시다. 뉴런 $c_3$의 threshold $\theta = 2$이고, 뉴런 $c_1$과 $c_2$로부터 각각 하나씩의 excitatory synapse를 받는다고 합시다. 억제성 입력은 없습니다.

Let us trace through an AND gate concretely. Neuron $c_3$ has threshold $\theta = 2$ and receives one excitatory synapse each from neurons $c_1$ and $c_2$. There are no inhibitory inputs.

| 시간 / Time | $N_1$ | $N_2$ | Excitatory sum at $c_3$ | $\geq \theta = 2$? | $N_3$ (output) |
|-------------|-------|-------|------------------------|---------------------|----------------|
| $t=0$ | 0 | 0 | — | — | — |
| $t=1$ | 1 | 0 | — | — | $0 + 0 = 0 < 2 \to 0$ |
| $t=2$ | 0 | 1 | — | — | $1 + 0 = 1 < 2 \to 0$ |
| $t=3$ | 1 | 1 | — | — | $0 + 1 = 1 < 2 \to 0$ |
| $t=4$ | 1 | 1 | — | — | $1 + 1 = 2 \geq 2 \to 1$ |

$t=1$에서: $c_3$는 $t=0$의 입력을 봅니다 (synaptic delay 때문). $N_1(0)=0, N_2(0)=0$이므로 합은 0이고, $0 < 2$이므로 $N_3(1)=0$입니다.
$t=4$에서: $c_3$는 $t=3$의 입력을 봅니다. $N_1(3)=1, N_2(3)=1$이므로 합은 2이고, $2 \geq 2$이므로 $N_3(4)=1$입니다.

핵심: threshold를 조절하면 같은 2-입력 뉴런이 다른 논리 게이트가 됩니다. $\theta=1$이면 OR 게이트, $\theta=2$이면 AND 게이트입니다. 이것이 McCulloch-Pitts 모델의 우아함입니다 — 구조적으로 동일한 뉴런이 threshold에 따라 다른 논리 함수를 구현합니다.

At $t=1$: $c_3$ sees inputs from $t=0$ (due to synaptic delay). $N_1(0)=0, N_2(0)=0$, so the sum is 0, and $0 < 2$ means $N_3(1)=0$.
At $t=4$: $c_3$ sees inputs from $t=3$. $N_1(3)=1, N_2(3)=1$, so the sum is 2, and $2 \geq 2$ means $N_3(4)=1$.

Key point: by adjusting the threshold, the same 2-input neuron becomes a different logic gate. $\theta=1$ gives OR, $\theta=2$ gives AND. This is the elegance of the McCulloch-Pitts model — structurally identical neurons implement different logical functions depending on the threshold.

**Theorem 2의 직관적 이해 / Intuitive Understanding of Theorem 2:**

Theorem 2가 "모든 TPE는 네트워크로 실현 가능하다"고 말할 때, 이것이 왜 성립하는지 직관적으로 이해해봅시다. 핵심은 **함수적 완전성(functional completeness)**에 있습니다. AND, OR, NOT은 **함수적으로 완전한 기저(functionally complete basis)**를 형성합니다 — 이 세 연산의 조합으로 어떤 Boolean 함수든 표현할 수 있습니다. 예를 들어, XOR은 $(A \cdot \sim B) \vee (\sim A \cdot B)$로, NAND는 $\sim(A \cdot B)$로 표현 가능합니다. McCulloch-Pitts 뉴런은 AND, OR, NOT을 모두 구현할 수 있으므로 (AND: $\theta=2$, OR: $\theta=1$, NOT: 억제성 시냅스), 이 뉴런들을 적절히 조합하면 **임의의 논리 함수**를 구현할 수 있습니다. 이것은 마치 NAND 게이트만으로 어떤 디지털 회로든 만들 수 있다는 현대 디지털 논리의 원리와 동일합니다.

When Theorem 2 says "every TPE is realizable by a net," let's understand intuitively why this holds. The key lies in **functional completeness**. AND, OR, NOT form a **functionally complete basis** — any Boolean function can be expressed as a combination of these three operations. For example, XOR is $(A \cdot \sim B) \vee (\sim A \cdot B)$, and NAND is $\sim(A \cdot B)$. Since McCulloch-Pitts neurons can implement AND, OR, and NOT (AND: $\theta=2$, OR: $\theta=1$, NOT: inhibitory synapse), combining these neurons appropriately can implement **any logical function**. This is the same principle as in modern digital logic, where any digital circuit can be built from NAND gates alone.

**Figure 1 — 기본 논리 게이트 구현 / Basic Logic Gate Implementations (p. 105):**

| 그림 / Figure | 논리식 / Logical Expression | 기능 / Function |
|--------------|---------------------------|----------------|
| (a) | $N_2(t) \equiv N_1(t-1)$ | Simple relay / 단순 전달 |
| (b) | $N_3(t) \equiv N_1(t-1) \vee N_2(t-1)$ | OR gate |
| (c) | $N_3(t) \equiv N_1(t-1) \cdot N_2(t-1)$ | AND gate |
| (d) | $N_3(t) \equiv N_1(t-1) \cdot \sim N_2(t-1)$ | AND-NOT (inhibition) |
| (e) | Heat/cold sensation example | 복합 시간 패턴 / Complex temporal pattern |
| (f) | Relative inhibition replacement | 상대적 억제 대체 |
| (g) | Extinction replacement | 소멸 대체 |
| (h) | Temporal → spatial summation | 시간 합산 → 공간 합산 |
| (i) | Alterable synapse → circle | 가변 시냅스 → 피드백 루프 |

**실용 예시 — 일시적 냉각에 의한 열감 / Practical Example — Heat from Transient Cooling (p. 106):**

차가운 물체를 피부에 잠깐 대면 열감이 느껴지고, 오래 대면 냉감만 느껴지는 착각을 모델링합니다:

$$N_3(t) \equiv N_1(t-1) \vee [N_2(t-3) \cdot \sim N_2(t-2)]$$
$$N_4(t) \equiv N_2(t-2) \cdot N_2(t-1)$$

여기서 $N_1$ = 열 수용체, $N_2$ = 냉 수용체, $N_3$ = 열감 뉴런, $N_4$ = 냉감 뉴런.

**시나리오별 단계 추적 / Step-by-Step Trace by Scenario:**

**시나리오 A: 열 자극만 (hot stimulus only)**
- $N_1$이 계속 발화, $N_2$는 비활성
- $N_3(t) = N_1(t-1) \vee [N_2(t-3) \cdot \sim N_2(t-2)] = 1 \vee [0 \cdot \sim 0] = 1$ → 열감 느낌 ✓
- $N_4(t) = N_2(t-2) \cdot N_2(t-1) = 0 \cdot 0 = 0$ → 냉감 없음 ✓

**시나리오 B: 지속적 냉 자극 (sustained cold stimulus)**
- $N_2$가 $t=0$부터 계속 발화 ($N_2(0)=1, N_2(1)=1, N_2(2)=1, \ldots$)
- $t=3$에서 $N_3$를 점검: $N_3(3) = N_1(2) \vee [N_2(0) \cdot \sim N_2(1)] = 0 \vee [1 \cdot 0] = 0$ → 열감 없음 ✓
- $t=3$에서 $N_4$를 점검: $N_4(3) = N_2(1) \cdot N_2(2) = 1 \cdot 1 = 1$ → 냉감 느낌 ✓
- $N_2$가 연속 발화하므로 $N_2(t-3) \cdot \sim N_2(t-2) = 1 \cdot \sim 1 = 1 \cdot 0 = 0$: 항상 거짓입니다.

**시나리오 C: 일시적 냉 자극 (transient cold — the illusion!)**
- $N_2$가 $t=0$에서만 한 번 발화하고 멈춤: $N_2(0)=1, N_2(1)=0, N_2(2)=0, \ldots$
- $t=3$에서 $N_3$를 점검: $N_3(3) = 0 \vee [N_2(0) \cdot \sim N_2(1)] = 0 \vee [1 \cdot \sim 0] = 0 \vee [1 \cdot 1] = 1$ → **열감 발생!** 🔥
- $t=2$에서 $N_4$를 점검: $N_4(2) = N_2(0) \cdot N_2(1) = 1 \cdot 0 = 0$ → 냉감 없음

핵심 메커니즘: $N_2(t-3) \cdot \sim N_2(t-2)$는 "3단계 전에는 냉 수용체가 활성이었지만 2단계 전에는 비활성이었다"를 감지합니다 — 즉, **냉각의 중단**을 감지합니다. 냉 수용체가 잠깐만 활성화되면 이 조건이 충족되어 열감 뉴런이 발화합니다. 이것은 실제로 관찰되는 신경학적 착각을 네트워크의 시간적 구조만으로 설명합니다 — "지각의 오류"가 신비로운 것이 아니라 네트워크 배선의 자연스러운 결과임을 보여줍니다.

The key mechanism: $N_2(t-3) \cdot \sim N_2(t-2)$ detects "cold receptor was active 3 steps ago but inactive 2 steps ago" — i.e., it detects the **cessation of cooling**. When the cold receptor is active only briefly, this condition is satisfied and the heat neuron fires. This explains an actually observed neurological illusion purely through the temporal structure of the network — showing that "perceptual errors" are not mysterious but natural consequences of network wiring.

**동등성 정리들 / Equivalence Theorems (Theorems 4–7):**

| 정리 / Theorem | 내용 / Content |
|---------------|-------------|
| **Theorem 4** | 상대적 억제와 절대적 억제는 확장된 의미에서 동등하다 / Relative and absolute inhibition are equivalent in the extended sense |
| **Theorem 5** | 소멸(extinction)은 절대적 억제와 동등하다 / Extinction is equivalent to absolute inhibition |
| **Theorem 6** | 촉진(facilitation)과 시간적 합산은 공간적 합산으로 대체 가능하다 / Facilitation and temporal summation can be replaced by spatial summation |
| **Theorem 7** | **가변 시냅스(학습!)는 원(피드백 루프)으로 대체 가능하다** / Alterable synapses (learning!) can be replaced by circles |

Theorem 7은 특히 주목할 만합니다: 학습의 효과를 고정된 구조의 피드백 네트워크로 모델링할 수 있다는 것입니다.

Theorem 7 is particularly remarkable: it suggests that the effect of learning can be modeled by fixed networks with feedback.

### Section 3: Nets with Circles / 원이 있는 네트워크 (pp. 108–113)

**원(Circle)이란? / What Are Circles?**

뉴런의 체인 $c_i \to c_{i+1} \to \ldots \to c_p \to c_i$가 루프를 형성하는 피드백 구조입니다. 활동이 이 루프를 무한정 반향(reverberate)할 수 있어, **기억(memory)**을 도입합니다 — 네트워크가 자신의 과거 상태를 참조할 수 있습니다.

Feedback loops where a chain of neurons forms a cycle. Activity can reverberate around indefinitely, introducing **memory** — the network can reference its own past states.

**왜 원(circle) = 기억(memory)인가 — 직관적 설명 / Why Circles = Memory — Intuitive Explanation:**

원이 없는(feed-forward) 네트워크를 생각해봅시다. 입력이 들어오면 한 방향으로 흘러 출력을 만들고 사라집니다. 네트워크는 이전에 어떤 입력이 들어왔는지 "기억"할 방법이 없습니다 — 현재 입력에만 반응합니다. 이제 뉴런 A → 뉴런 B → 뉴런 A로 이어지는 원(circle)을 상상해봅시다. 한 번 뉴런 A가 발화하면, 그 신호는 B를 거쳐 다시 A로 돌아옵니다. A가 다시 발화하면 또 B를 거쳐 돌아옵니다. 이 활동은 외부 입력이 없어도 무한히 **반향(reverberate)**합니다 — 마치 두 거울 사이에서 빛이 무한히 반사되는 것과 같습니다. 이 반향하는 활동이 바로 **기억**입니다: 과거에 어떤 사건이 일어났다는 정보가 네트워크의 활성 상태로 유지되고 있는 것입니다.

Consider a feed-forward (circle-free) network. Input flows in one direction, produces output, and vanishes. The network has no way to "remember" what input came before — it responds only to current input. Now imagine a circle: neuron A → neuron B → neuron A. Once neuron A fires, the signal travels through B and returns to A. A fires again, travels through B again, and so on. This activity **reverberates** indefinitely without external input — like light bouncing infinitely between two parallel mirrors. This reverberating activity IS **memory**: the information that some past event occurred is maintained as an active state in the network.

**구체적 예시: Flip-Flop 메모리 회로 / Concrete Example: Flip-Flop Memory Circuit:**

두 뉴런 $c_A$와 $c_B$로 구성된 가장 단순한 메모리 회로(flip-flop)를 생각해봅시다:
- $c_A$는 $c_B$로 excitatory 연결, $c_B$는 $c_A$로 excitatory 연결 (threshold 각각 $\theta=1$)
- 외부에서 "set" 입력이 $c_A$를 한 번 발화시키면:

| Time | $N_A$ | $N_B$ | 설명 / Explanation |
|------|-------|-------|-------------------|
| $t=0$ | 1 (set) | 0 | 외부 입력으로 A 발화 / External input fires A |
| $t=1$ | 0 | 1 | A의 신호가 B에 도달 / A's signal reaches B |
| $t=2$ | 1 | 0 | B의 신호가 A에 도달 / B's signal reaches A |
| $t=3$ | 0 | 1 | 반복... / Repeats... |
| $\vdots$ | $\vdots$ | $\vdots$ | 무한히 반향 / Reverberates indefinitely |

이 회로는 **1 bit의 정보를 저장**합니다: "과거 어느 시점에 set 입력이 있었다"는 사실이 A-B 간의 반향 활동으로 유지됩니다. 활동이 멈추면 ("reset") 이 정보는 소실됩니다. 이것은 현대 컴퓨터의 SR latch(Set-Reset latch)와 개념적으로 동일합니다.

This circuit **stores 1 bit of information**: the fact that "a set input occurred at some past time" is maintained as reverberating activity between A and B. When the activity stops ("reset"), this information is lost. This is conceptually identical to the SR latch (Set-Reset latch) in modern computers.

**네트워크의 차수 / Order of a Net:**

네트워크에서 원을 제거하기 위해 필요한 최소 뉴런 집합의 크기입니다. 차수가 높을수록 행동이 더 복잡합니다.

The cardinality of the minimum set of neurons whose removal leaves the net acyclic. Higher order = more complex behavior.

**핵심 결과 / Key Results:**

- **Theorem 8**: 원이 있는 네트워크의 해(solution)는 순환 집합 뉴런에 대한 표현식과 다른 뉴런에 대한 TPE로 구성됩니다.
  The solution of nets with circles consists of expressions for cyclic set neurons plus TPEs for others.
- **Theorems 9–10**: 어떤 행동 클래스("prehensible classes")가 원이 있는 네트워크로 실현 가능한지 특성화합니다.
  Characterize which classes of behavior are realizable by nets with circles.

**Turing 동등성 / Turing Equivalence (p. 113):**

> "every net, if furnished with a tape, scanners connected to afferents, and suitable efferents to perform the necessary motor-operations, can compute only such numbers as can a Turing machine; second, that each of the latter numbers can be computed by such a net."

이것이 논문의 가장 심오한 결과입니다: **McCulloch-Pitts 신경망은 Turing machine과 계산적으로 동등**합니다 — 계산 가능한 모든 것을 계산할 수 있습니다.

This is the paper's most profound result: **McCulloch-Pitts neural networks are computationally equivalent to Turing machines** — they can compute anything that is computable.

**테이프가 왜 필요한가 — 무한 기억의 문제 / Why the Tape Is Needed — The Problem of Unbounded Memory:**

여기서 핵심 질문이 생깁니다: 원(circle)이 이미 기억을 제공하는데, 왜 Turing 동등성을 위해 별도의 "테이프"가 필요할까요? 답은 **유한 vs. 무한 기억**의 차이에 있습니다. 뉴런의 수가 유한한 네트워크는 유한한 수의 원(circle)만 가질 수 있으므로, 유한한 비트만 기억할 수 있습니다 — 이것은 Turing machine이 아니라 **유한 상태 기계(finite state machine)**에 해당합니다. Turing machine의 핵심 특성은 **무한히 긴 테이프** — 즉, 필요한 만큼 무한히 확장 가능한 외부 메모리 — 를 가진다는 것입니다.

McCulloch와 Pitts가 말하는 것은 이것입니다: 유한한 M-P 신경망에 (1) 테이프(무한 외부 메모리), (2) scanner(테이프를 읽어 afferent 뉴런에 입력하는 장치), (3) efferent(테이프에 쓰고 이동하는 모터 출력)를 결합하면, 이 시스템은 정확히 Turing machine이 계산할 수 있는 것만 계산할 수 있고, 그 역도 성립합니다. 즉, 네트워크는 Turing machine의 "유한 제어(finite control)" 부분 역할을 하고, 테이프는 무한 기억을 제공합니다. 현대적 관점에서 보면, 이것은 CPU(유한한 논리 회로)와 RAM(확장 가능한 메모리)의 관계와 정확히 같습니다.

Here a key question arises: if circles already provide memory, why is a separate "tape" needed for Turing equivalence? The answer lies in the difference between **finite vs. unbounded memory**. A network with a finite number of neurons can only have finite circles, thus can only remember finitely many bits — this corresponds to a **finite state machine**, not a Turing machine. The essential characteristic of a Turing machine is an **infinite tape** — external memory that can be extended as far as needed.

What McCulloch and Pitts are saying is this: when a finite M-P neural network is combined with (1) a tape (unbounded external memory), (2) scanners (devices that read the tape and feed input to afferent neurons), and (3) efferents (motor outputs that write to and move the tape), this system can compute exactly what a Turing machine can compute, and vice versa. The network serves as the "finite control" portion of the Turing machine, while the tape provides unbounded memory. In modern terms, this is exactly the relationship between a CPU (finite logic circuits) and RAM (expandable memory).

### Section 4: Consequences / 결과 (pp. 113–115)

**철학적 함의 / Philosophical Implications:**

- **인과성(Causality)의 비가역성**: 현재 상태에서 미래 상태를 계산할 수 있지만, 과거를 완전히 결정할 수는 없습니다 (disjunctive relations 때문).
  Causality is irreciprocal: we can compute future from present, but cannot fully determine the past.
- **원의 재생적 활동**: 시간적 참조를 무한정으로 만들어, 지식은 공간에 대해 불완전하고 시간에 대해 부정확합니다.
  Regenerative activity of circles renders reference to time past indefinite.

**"Mind no longer goes more ghostly than a ghost" — 이 인용의 맥락 / Context of This Quote:**

이 구절은 논문의 마지막 부분에서 등장하며, McCulloch와 Pitts가 반대하고자 했던 오랜 철학적 전통에 대한 선전포고입니다. 원래 문구는 영국 시인 Housman의 "The half of my own soul" 시에서 온 것이 아니라, 당시 팽배했던 **이원론(dualism)** — 정신(mind)과 물질(body)이 근본적으로 다른 종류의 존재라는 데카르트적 관점 — 에 대한 공격입니다. 1943년까지도 많은 철학자와 과학자들은 의식, 사고, 감정 같은 "정신적 현상(psychic events)"이 물리적 과정으로 환원될 수 없다고 믿었습니다 — 정신은 물질 세계와 다른 "유령 같은(ghostly)" 영역에 속한다는 것이었습니다. McCulloch와 Pitts는 이 논문을 통해 정신 활동의 모든 측면 — 논리, 기억, 지각, 심지어 착각 — 이 뉴런 네트워크의 속성으로 **엄밀하게** 기술될 수 있음을 보여줌으로써, 정신을 물리적 세계에 확고히 자리매김시켰습니다.

This passage appears at the end of the paper and is a declaration of war against a long philosophical tradition. It is an attack on **dualism** — the Cartesian view that mind and body are fundamentally different kinds of substance — which was still widely held. Until 1943, many philosophers and scientists believed that "psychic events" such as consciousness, thought, and emotion could not be reduced to physical processes — that the mind belonged to a "ghostly" realm separate from the material world. Through this paper, McCulloch and Pitts placed the mind firmly in the physical world by showing that all aspects of mental activity — logic, memory, perception, even illusions — can be **rigorously** described as properties of neural networks.

- **"정신적 사건(psychic events)"의 네트워크적 기술 / Network Description of "Psychic Events":**

McCulloch와 Pitts가 "psychic events"라고 부르는 것은 신비로운 것이 아니라, 뉴런의 발화 패턴 그 자체입니다. 논문의 논리에 따르면: (1) 뉴런의 활동은 명제로 표현된다; (2) 뉴런들 사이의 관계는 논리식으로 기술된다; (3) 따라서 "내가 열을 느낀다"는 주관적 경험은 "$N_3(t)=1$"이라는 객관적 네트워크 상태에 대응합니다. 환각(hallucination)은 외부 자극 없이 내부 네트워크의 자발적 활동이 지각 뉴런을 발화시키는 것이고, 착각(delusion)은 네트워크 구조가 입력을 잘못 매핑하는 것입니다 (앞서 본 열/냉 착각처럼). 혼란(confusion)은 네트워크가 모호한 상태에 있는 것입니다. 이 모든 "정신 병리"가 네트워크의 정상적 작동의 결과로 설명됩니다.

What McCulloch and Pitts call "psychic events" are not mysterious — they are simply firing patterns of neurons. Following the paper's logic: (1) neural activity is represented as propositions; (2) relations among neurons are described as logical expressions; (3) therefore, the subjective experience "I feel heat" corresponds to the objective network state "$N_3(t)=1$." Hallucination is spontaneous internal network activity firing perceptual neurons without external stimulus. Delusion is the network structure mis-mapping inputs (like the heat/cold illusion above). Confusion is the network being in an ambiguous state. All these "mental pathologies" are explained as consequences of normal network operation.

- **목적적 행동(Purposive behavior) — 항온 조절기로서의 뇌 / Purposive Behavior — The Brain as Thermostat:**

McCulloch와 Pitts가 설명하는 목적적 행동(purposive behavior)은 사이버네틱스(cybernetics)의 핵심 개념인 **부정적 피드백(negative feedback)**과 직접 연결됩니다. 가장 단순한 예로 항온 조절기(thermostat)를 생각해봅시다: (1) 목표 온도(set point)가 재생적 네트워크(circle)에 저장되어 있습니다; (2) 온도 센서(afferent)가 현재 온도를 입력합니다; (3) 네트워크는 목표 온도와 현재 온도의 **차이**를 계산합니다; (4) 이 차이를 줄이는 방향으로 출력(efferent)을 보냅니다 — 너무 추우면 난방을 켜고, 너무 더우면 끕니다. 이 간단한 구조가 "목적(purpose)"을 가진 것처럼 보이는 행동을 산출합니다. McCulloch와 Pitts는 항상성(homeostasis), 식욕(appetite), 심지어 주의(attention)까지도 이 원리로 설명될 수 있다고 주장합니다 — 신비로운 "의지(will)"나 "욕구(desire)" 없이, 순수하게 네트워크 구조의 결과로서. 이 아이디어는 같은 해(1943) Rosenblueth, Wiener, Bigelow가 발표한 "Behavior, Purpose and Teleology"와 깊이 공명하며, 사이버네틱스 운동의 핵심 원리가 됩니다.

The purposive behavior McCulloch and Pitts describe connects directly to **negative feedback**, a core concept of cybernetics. Consider the simplest example, a thermostat: (1) a target temperature (set point) is stored in a regenerative network (circle); (2) a temperature sensor (afferent) inputs the current temperature; (3) the network computes the **difference** between target and current temperature; (4) it sends output (efferent) in the direction that reduces the difference — turns on heating if too cold, turns it off if too warm. This simple structure produces behavior that appears to have "purpose." McCulloch and Pitts argue that homeostasis, appetite, and even attention can be explained by this principle — purely as consequences of network structure, without mysterious "will" or "desire." This idea resonates deeply with Rosenblueth, Wiener, and Bigelow's "Behavior, Purpose and Teleology" published the same year (1943), and becomes a core principle of the cybernetics movement.

---

## 3. Key Takeaways / 핵심 시사점

1. **뉴런을 논리 게이트로 추상화한 최초의 모델**: 생물학적 뉴런을 이진 임계값 유닛으로 단순화함으로써, McCulloch와 Pitts는 신경과학과 수학적 논리 사이의 다리를 놓았습니다. 이 추상화 — 조잡하지만 강력한 — 가 모든 인공 뉴런의 조상입니다. 이전에는 신경계를 수학적으로 기술하려는 시도가 주로 전기 회로 모델(케이블 이론 등)에 머물렀지만, McCulloch와 Pitts는 전기적 세부사항을 과감히 버리고 "정보 처리"의 관점에서 뉴런을 바라본 최초의 인물들이었습니다. 이 관점의 전환이 신경과학을 "배선도 그리기"에서 "알고리즘 분석"으로 끌어올렸고, 궁극적으로 인공지능이라는 분야 자체를 가능하게 했습니다.
   **First model to abstract neurons as logic gates**: By simplifying biological neurons to binary threshold units, McCulloch and Pitts bridged neuroscience and mathematical logic. This abstraction — crude yet powerful — is the ancestor of all artificial neurons. Previous attempts to describe the nervous system mathematically focused mainly on electrical circuit models (cable theory, etc.), but McCulloch and Pitts boldly discarded electrical details and viewed neurons from the perspective of "information processing" for the first time. This shift in perspective elevated neuroscience from "drawing wiring diagrams" to "analyzing algorithms" and ultimately made the field of artificial intelligence itself possible.

2. **범용성(Universality) 증명**: 모든 Boolean 함수는 McCulloch-Pitts 뉴런 네트워크로 계산할 수 있습니다 (Theorems 1–2). 피드백 루프를 추가하면 Turing-complete가 됩니다. 이 결과의 의미를 과소평가하기 어렵습니다: 그것은 원리적으로 뇌가 어떤 계산이든 수행할 수 있다는 것을 수학적으로 증명한 것입니다. 다만 "원리적으로"라는 단서가 중요합니다 — 실제 구현에 필요한 뉴런의 수나 시간 단계의 수는 비현실적으로 클 수 있습니다. 그럼에도 이 존재 증명(existence proof)은 "기계가 생각할 수 있는가?"라는 질문에 대한 최초의 긍정적 이론적 근거를 제공했습니다.
   **Universality proven**: Any Boolean function can be computed by a McCulloch-Pitts network. With feedback loops, these networks become Turing-complete. The significance of this result is hard to overstate: it mathematically proved that, in principle, the brain can perform any computation. The caveat "in principle" is important — the number of neurons or time steps required for actual implementation may be unrealistically large. Nevertheless, this existence proof provided the first positive theoretical basis for the question "can machines think?"

3. **다섯 가지 가정이 한 분야를 탄생시켰다**: all-or-none, 고정 threshold, synaptic delay, 절대적 억제, 고정 구조라는 단순화가 수학을 다루기 쉽게 만들었습니다. 현대 신경망은 이 대부분을 완화합니다 — 특히 고정 구조(학습!). 이것은 과학적 모델링의 교훈이기도 합니다: 현실에서 충분히 많은 것을 추상화(제거)해야 수학적으로 다룰 수 있는 모델이 나오고, 그 모델에서 얻은 통찰이 다시 현실을 이해하는 도구가 됩니다. McCulloch와 Pitts의 다섯 가정은 신경계의 복잡성을 극도로 단순화했지만, 바로 그 단순화 덕분에 범용성이라는 핵심 통찰을 추출할 수 있었습니다.
   **Five assumptions launched a field**: The simplifications made the math tractable. Modern neural networks relax most of these — especially the fixed structure (learning!). This is also a lesson in scientific modeling: one must abstract (remove) enough from reality to produce a mathematically tractable model, and insights from that model then become tools for understanding reality. McCulloch and Pitts' five assumptions drastically simplified the complexity of the nervous system, but it was precisely that simplification that enabled extraction of the key insight of universality.

4. **이 모델에는 학습이 없다**: 가정 5는 구조적 변화를 명시적으로 금지합니다. Theorem 7은 가변 시냅스를 피드백 루프로 대체할 수 있음을 보여주지만, 직접적인 학습 모델은 없습니다. 이 한계는 Hebb (1949)와 Rosenblatt (1958)에 의해 다루어집니다. 흥미롭게도, Theorem 7이 "학습은 피드백 루프로 시뮬레이션 가능하다"고 보여준 것은 학습이 본질적으로 기억(memory)의 한 형태임을 암시합니다 — 과거 경험이 현재 행동에 영향을 미치는 메커니즘으로서. 그러나 이 "시뮬레이션"은 구성적(constructive)이지 않습니다: 어떤 피드백 구조가 원하는 학습 행동을 만드는지를 알아내는 방법을 제공하지 않습니다. 이것이 바로 Rosenblatt의 perceptron 학습 알고리즘과 궁극적으로 backpropagation이 해결한 문제입니다.
   **No learning in this model**: Assumption 5 explicitly forbids structural change. Theorem 7 shows alterable synapses can be replaced by circles, but there is no direct learning model. This limitation would be addressed by Hebb (1949) and Rosenblatt (1958). Interestingly, Theorem 7's demonstration that "learning can be simulated by feedback loops" suggests that learning is essentially a form of memory — a mechanism by which past experience influences present behavior. However, this "simulation" is not constructive: it doesn't provide a method to figure out which feedback structure produces the desired learning behavior. This is precisely the problem solved by Rosenblatt's perceptron learning algorithm and ultimately by backpropagation.

5. **억제(Inhibition)가 근본적이다**: 절대적 억제 가정(가정 4)은 NOT 계산에, 따라서 범용 계산에 결정적입니다. 억제 없이는 단조(monotone) 함수만 계산 가능합니다. 이것을 현대적으로 재해석하면, 신경망에서 음의 가중치(negative weights)가 없으면 네트워크의 표현력이 극도로 제한된다는 것과 같습니다. 생물학적으로도 억제성 뉴런(주로 GABA를 분비하는 interneuron)은 전체 뉴런의 약 20%를 차지하며, 뇌의 정상적 기능에 필수적입니다 — 억제가 실패하면 간질(epilepsy)이 발생합니다.
   **Inhibition is fundamental**: The absolute inhibition assumption is critical for computing NOT and therefore for universal computation. Without it, only monotone functions would be computable. In modern terms, this means that without negative weights, a neural network's expressive power is extremely limited. Biologically, inhibitory neurons (mainly GABA-secreting interneurons) account for about 20% of all neurons and are essential for normal brain function — failure of inhibition leads to epilepsy.

6. **시간은 이산적이다**: 모델은 synaptic delay에 의해 동기화된 이산 시간 단계에서 작동합니다. 이것은 현대 RNN에서 사용되는 이산 시간 단계의 전조입니다. 이산 시간의 도입은 단순한 기술적 편의가 아니라, 뉴런 네트워크를 디지털 논리 회로와 연결시키는 결정적 다리 역할을 했습니다. von Neumann이 이 논문에서 영감을 받아 저장 프로그램 컴퓨터를 설계한 것은 우연이 아닙니다 — 이산 시간에서 작동하는 논리 요소들의 네트워크라는 개념이 디지털 컴퓨터 아키텍처의 핵심이기 때문입니다.
   **Time is discrete**: The model operates in discrete time steps synchronized by synaptic delay. This is a precursor to discrete time steps used in modern recurrent neural networks. The introduction of discrete time was not merely a technical convenience — it served as a critical bridge connecting neural networks to digital logic circuits. It is no coincidence that von Neumann was inspired by this paper to design the stored-program computer — the concept of a network of logic elements operating in discrete time is the core of digital computer architecture.

7. **철학적 대담함**: 논문은 단순히 모델을 제안하는 것이 아니라, 정신이 논리적 기계로 이해될 수 있다는 철학적 주장을 합니다. 1943년에 이것은 급진적이었습니다. 제2차 세계대전이 한창이던 시기에, 당시 지배적이었던 행동주의(behaviorism)는 내적 정신 과정의 논의 자체를 비과학적이라고 배척했고, 이원론(dualism)은 정신을 물질과 별개의 존재로 보았습니다. McCulloch와 Pitts는 이 두 극단 모두를 거부하고, 정신은 실재하며(real) 동시에 물리적(physical)이라고 — 뉴런 네트워크의 속성으로 완전히 기술 가능하다고 — 주장했습니다. 이 입장은 이후 인지과학(cognitive science) 전체의 철학적 기반이 됩니다.
   **Philosophical boldness**: The paper doesn't just propose a model — it makes a philosophical claim that the mind can be understood as a logical machine. This was radical in 1943. In the midst of World War II, the then-dominant behaviorism rejected discussion of internal mental processes as unscientific, while dualism viewed the mind as separate from matter. McCulloch and Pitts rejected both extremes, arguing that the mind is both real and physical — fully describable as properties of neural networks. This position became the philosophical foundation of all of cognitive science.

8. **동등성 정리의 실용적 가치**: 다양한 신경생리학적 가정(상대적/절대적 억제, 소멸, 시간적/공간적 합산)이 동등함을 증명하여, 모델 선택의 자유도를 확보했습니다. Theorems 4–7은 겉보기에 다른 신경 메커니즘들이 계산적으로 동등함을 보여줌으로써, 모델러에게 가장 편리한 가정을 선택할 자유를 줍니다. 이것은 현대 머신러닝에서도 유효한 원칙입니다: 다양한 activation function(sigmoid, tanh, ReLU 등)이 이론적으로 동등한 표현력을 가지며, 실용적 편의에 따라 선택됩니다.
   **Practical value of equivalence theorems**: By proving various neurophysiological assumptions equivalent, the authors secured freedom in model choice. Theorems 4–7 show that seemingly different neural mechanisms are computationally equivalent, giving modelers the freedom to choose the most convenient assumptions. This principle remains valid in modern machine learning: various activation functions (sigmoid, tanh, ReLU, etc.) have theoretically equivalent expressive power and are chosen based on practical convenience.

---

## 4. Mathematical Summary / 수학적 요약

### McCulloch-Pitts Neuron Model / McCulloch-Pitts 뉴런 모델

**뉴런 $c_i$의 동작 / Behavior of neuron $c_i$:**

$$y_i(t) = \begin{cases} 1 & \text{if } \sum_{k \in \text{excitatory}} x_k(t-1) \geq \theta_i \text{ AND } \forall j \in \text{inhibitory}: x_j(t-1) = 0 \\ 0 & \text{otherwise} \end{cases}$$

**논문의 형식적 표기 / Formal notation from the paper (Eq. 1):**

$$N_i(z_1) \equiv S\left\{\prod_{m=1}^{q} \sim N_{j_m}(z_1) \cdot \sum_{\alpha \in \kappa_i} \prod_{s \in \alpha} N_{i_s}(z_1)\right\}$$

### 기본 논리 게이트 구현 / Basic Logic Gate Implementations

| Gate | Threshold $\theta$ | Excitatory inputs | Inhibitory inputs | Expression |
|------|-------------------|-------------------|-------------------|------------|
| Relay | 1 | 1 | 0 | $N_2(t) = N_1(t-1)$ |
| OR | 1 | 2 | 0 | $N_3(t) = N_1(t-1) \vee N_2(t-1)$ |
| AND | 2 | 2 | 0 | $N_3(t) = N_1(t-1) \cdot N_2(t-1)$ |
| AND-NOT | 1 | 1 | 1 | $N_3(t) = N_1(t-1) \cdot \sim N_2(t-1)$ |

### 핵심 정리 요약 / Summary of Key Theorems

| Theorem | Statement (English) | Statement (Korean) |
|---------|--------------------|--------------------|
| 1 | Every net of order 0 is solvable in TPEs | 모든 order 0 네트워크는 TPE로 풀 수 있다 |
| 2 | Every TPE is realizable by a net of order 0 | 모든 TPE는 order 0 네트워크로 실현 가능하다 |
| 3 | $S_1$ is a TPE iff it is false when all atoms are false | $S_1$이 TPE인 것은 모든 원자가 거짓일 때 $S_1$이 거짓인 경우와 동치 |
| 4 | Relative ≡ absolute inhibition | 상대적 억제 ≡ 절대적 억제 |
| 5 | Extinction ≡ absolute inhibition | 소멸 ≡ 절대적 억제 |
| 6 | Temporal summation → spatial summation | 시간적 합산 → 공간적 합산 |
| 7 | Alterable synapses → circles | 가변 시냅스 → 피드백 루프 |
| 8 | Nets with circles: solution via cyclic set + TPE | 원이 있는 네트워크: 순환 집합 + TPE로 해 |
| 9–10 | Characterization of prehensible classes | Prehensible class의 특성화 |
| Turing | M-P nets + tape = Turing machine | M-P 네트워크 + 테이프 = Turing machine |

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1879  Frege — Begriffsschrift (formal logic / 형식 논리학)
  │
1910  Russell & Whitehead — Principia Mathematica
  │
1936  Turing — Turing Machine (computability / 계산 가능성)
  │
1938  Carnap — Logical Syntax of Language
  │
1942  Wiener — Cybernetics ideas forming (사이버네틱스 아이디어 형성)
  │
1943  ★ McCulloch & Pitts — THIS PAPER ★
  │     "뉴런 = 논리 게이트, 신경망 = 범용 컴퓨터"
  │     "Neuron = logic gate, neural net = universal computer"
  │
1945  von Neumann — EDVAC report (influenced by M-P / M-P에 영향받음)
  │
1949  Hebb — The Organization of Behavior (학습 규칙 / learning rule)
  │
1950  Turing — "Computing Machinery and Intelligence" (Paper #2)
  │
1956  Dartmouth Conference — AI 분야 공식 탄생 / Birth of AI as a field
  │
1958  Rosenblatt — Perceptron (Paper #3)
  │     학습 가능한 M-P 뉴런 / Trainable M-P neuron
  │
1969  Minsky & Papert — Perceptrons (Paper #4)
        단층 perceptron의 한계 증명 / Proving single-layer perceptron limits
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Relationship / 관계 |
|-------------|-------------------|
| Turing, "On Computable Numbers" (1936) | M-P 모델의 계산적 동등성은 Turing machine에 직접 연결됩니다. 논문은 신경망이 Turing machine과 동등한 계산 능력을 가짐을 증명합니다. / The computational equivalence of the M-P model directly connects to Turing machines. |
| Hebb, *The Organization of Behavior* (1949) | M-P 모델에 없는 학습 메커니즘을 제공합니다. "함께 발화하는 뉴런은 함께 연결된다"는 Hebbian learning이 가정 5를 완화합니다. / Provides the learning mechanism absent from the M-P model. Hebbian learning relaxes Assumption 5. |
| Turing, "Computing Machinery and Intelligence" (1950) — Paper #2 | M-P가 신경의 수학적 모델을 제공했다면, Turing은 "기계가 생각할 수 있는가?"라는 철학적 질문을 정면으로 다룹니다. / If M-P provided the mathematical model, Turing directly addressed the philosophical question. |
| Rosenblatt, "The Perceptron" (1958) — Paper #3 | M-P 뉴런에 **학습 알고리즘**을 추가하여 최초의 학습 가능한 신경망을 만들었습니다. 가중치를 고정값이 아닌 학습 가능한 파라미터로 전환합니다. / Added a **learning algorithm** to M-P neurons, creating the first trainable neural network. |
| Minsky & Papert, *Perceptrons* (1969) — Paper #4 | 단층 perceptron의 한계(XOR 불가)를 엄밀히 증명하여, 다층 네트워크의 필요성을 밝혔습니다. / Rigorously proved single-layer perceptron limitations (e.g., XOR), motivating multi-layer networks. |
| Rumelhart et al., "Backpropagation" (1986) — Paper #6 | M-P 뉴런을 연속 활성화 함수와 학습 가능한 가중치로 확장하여, 다층 네트워크 학습을 실용화했습니다. / Extended M-P neurons with continuous activations and learnable weights, making multi-layer training practical. |
| von Neumann, EDVAC report (1945) | von Neumann은 M-P 논문에 직접 영향을 받아 저장 프로그램 컴퓨터 아키텍처를 설계했습니다. / von Neumann was directly influenced by this paper in designing the stored-program computer architecture. |

---

## 7. McCulloch-Pitts vs. Modern Neural Networks / M-P 모델과 현대 신경망 비교

| McCulloch-Pitts (1943) | Modern Neural Networks / 현대 신경망 |
|---|---|
| Binary output (0 or 1) / 이진 출력 | Continuous activations (sigmoid, ReLU, etc.) / 연속 활성화 |
| Fixed threshold / 고정 임계값 | Learnable weights and biases / 학습 가능한 가중치와 편향 |
| No learning / 학습 없음 | Backpropagation, gradient descent / 역전파, 경사 하강법 |
| Absolute inhibition / 절대적 억제 | Negative weights / 음의 가중치 |
| Discrete time steps / 이산 시간 | Continuous or batched computation / 연속 또는 배치 계산 |
| Logical equivalence proofs / 논리적 동등성 증명 | Empirical/statistical performance / 경험적/통계적 성능 |
| All weights = 1 / 모든 가중치 = 1 | Variable weights / 가변 가중치 |

---

## 8. References / 참고문헌

- McCulloch, W.S., Pitts, W. (1943). "A Logical Calculus of the Ideas Immanent in Nervous Activity." *Bulletin of Mathematical Biophysics*, 5, 115–133. [DOI: 10.1007/BF02478259]
- Carnap, R. (1938). *The Logical Syntax of Language*. New York: Harcourt-Brace.
- Hilbert, D. and Ackermann, W. (1927). *Grundzüge der Theoretischen Logik*. Berlin: Springer.
- Russell, B. and Whitehead, A.N. (1925). *Principia Mathematica*. Cambridge University Press.
