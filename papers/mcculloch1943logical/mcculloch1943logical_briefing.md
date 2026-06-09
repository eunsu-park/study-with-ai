# Pre-Reading Briefing: McCulloch & Pitts (1943) / 사전 읽기 브리핑

**Paper**: "A Logical Calculus of the Ideas Immanent in Nervous Activity"
**Authors**: Warren McCulloch, Walter Pitts
**Year**: 1943

---

## 1. 핵심 기여 / Core Contribution

Warren McCulloch(신경과학자)와 Walter Pitts(수학자)는 **뉴런의 활동을 명제 논리(propositional logic)로 형식화한 최초의 수학적 모델**을 제시했습니다. 이 논문은 신경망을 "이진 임계값 유닛(binary threshold unit)"으로 추상화하고, 이러한 유닛들의 네트워크가 **모든 논리 함수(AND, OR, NOT 등)를 계산할 수 있음**을 증명했습니다. 이것이 바로 인공 신경망의 이론적 출발점이며, 이후 모든 neural network 연구의 씨앗이 된 논문입니다.

Warren McCulloch (neuroscientist) and Walter Pitts (mathematician) presented **the first mathematical model formalizing neuronal activity as propositional logic**. The paper abstracts neural networks as "binary threshold units" and proves that networks of such units can compute **any logical function (AND, OR, NOT, etc.)**. This is the theoretical starting point of artificial neural networks and the seed of all subsequent neural network research.

---

## 2. 역사적 맥락 / Historical Context

```
1936  Turing Machine (계산 이론의 기초 / Foundation of computation theory)
  │
1943  ★ McCulloch & Pitts ← 지금 읽을 논문 / Paper to read now
  │     "뉴런 = 논리 게이트" 라는 혁명적 아이디어
  │     Revolutionary idea: "neuron = logic gate"
  │
1949  Hebb — "Hebbian Learning" (시냅스 학습 규칙 / Synaptic learning rule)
  │
1950  Turing — "Computing Machinery and Intelligence"
  │
1958  Rosenblatt — Perceptron (최초의 학습 가능한 신경망 / First trainable neural network)
```

- 1940년대 초는 **신경과학**과 **수학적 논리학**이 독립적으로 발전하던 시기였습니다.
  The early 1940s was a period when **neuroscience** and **mathematical logic** were developing independently.
- McCulloch은 뇌가 어떻게 "생각"하는지를, Pitts는 수학적 형식화를 담당했습니다.
  McCulloch focused on how the brain "thinks," while Pitts handled the mathematical formalization.
- 이 논문은 **"뇌의 작동 원리를 수학으로 기술할 수 있다"**는 것을 최초로 보여준 것입니다.
  This paper was the first to show that **"the brain's operating principles can be described mathematically."**

---

## 3. 필요한 배경 지식 / Prerequisites

1. **기초 논리학 / Basic Logic**: AND ($\wedge$), OR ($\vee$), NOT ($\sim$) 연산
2. **명제 논리 / Propositional Logic**: 참/거짓 값을 갖는 명제와 그 조합 / Propositions with true/false values and their combinations
3. **집합론 기초 / Basic Set Theory**: 기본적인 집합 표기법 / Basic set notation
4. **임계값 함수 / Threshold Function**: 입력의 합이 특정 값(threshold)을 넘으면 1, 아니면 0을 출력하는 함수 / A function that outputs 1 if the sum of inputs exceeds a threshold, 0 otherwise

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 직관적 설명 / Intuitive Explanation |
|-------------|-------------------------------------|
| **Neuron (뉴런)** | 논문에서는 "all-or-none" 장치 — 발화(fire)하거나 안 하거나, 이진(binary). In the paper: an "all-or-none" device — it either fires or doesn't, binary. |
| **Threshold (임계값, $\theta$)** | 뉴런이 발화하려면 넘어야 하는 최소 입력 합. The minimum sum of inputs a neuron must receive to fire. |
| **Excitatory input (흥분성 입력)** | 뉴런의 발화를 촉진하는 입력 (양의 가중치). An input that promotes firing (positive weight). |
| **Inhibitory input (억제성 입력)** | 뉴런의 발화를 절대적으로 차단하는 입력 — 하나라도 있으면 발화 불가. An input that absolutely blocks firing — even one active inhibitory input prevents firing. |
| **Net (신경망)** | 뉴런들의 연결 그래프. A connection graph of neurons. |
| **Temporal propositional expression** | 시간 $t$에서의 뉴런 상태를 논리식으로 표현한 것. A logical expression of a neuron's state at time $t$. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 McCulloch-Pitts 뉴런 모델 / McCulloch-Pitts Neuron Model

$$y(t) = \begin{cases} 1 & \text{if } \sum_{i} w_i \cdot x_i(t-1) \geq \theta \text{ and no inhibitory input is active} \\ 0 & \text{otherwise} \end{cases}$$

- $x_i(t-1)$: 시간 $t-1$에서의 입력 뉴런 $i$의 상태 (0 또는 1) / State of input neuron $i$ at time $t-1$ (0 or 1)
- $w_i$: 연결 가중치 (이 논문에서는 모두 1로 동일) / Connection weight (all equal to 1 in this paper)
- $\theta$: 임계값 (threshold)
- 핵심: **시간 지연(time delay)** — 입력은 항상 한 시간 단위 전의 값 / Key: **time delay** — inputs always use values from one time step prior

### 5.2 논리 게이트 구현 예시 / Logic Gate Implementation Examples

- **AND**: $\theta = 2$, 두 개의 흥분성 입력 → 둘 다 1이어야 발화 / Two excitatory inputs → both must be 1 to fire
- **OR**: $\theta = 1$, 두 개의 흥분성 입력 → 하나만 1이어도 발화 / Two excitatory inputs → only one needs to be 1
- **NOT**: 억제성 입력 하나 → 입력이 1이면 출력 차단 / One inhibitory input → blocks output when input is 1

### 5.3 핵심 정리 / Main Theorem

> 충분한 뉴런이 주어지면, McCulloch-Pitts 네트워크는 **시간적 명제 논리의 모든 표현**을 계산할 수 있다.
>
> Given enough neurons, a McCulloch-Pitts network can compute **any expression in temporal propositional logic**.

이것은 사실상 **뉴런 네트워크의 계산적 완전성(computational completeness)**을 증명한 것입니다 — Turing machine과의 연결고리이기도 합니다.

This essentially proves the **computational completeness of neuron networks** — connecting to Turing machines.

---

## 6. 읽기 팁 / Reading Tips

- 논문이 1943년에 쓰여져서 **표기법이 현대와 상당히 다릅니다** — 당황하지 마세요.
  The paper was written in 1943, so **notation differs considerably from modern conventions** — don't be alarmed.
- 핵심은 Section 1–3에 집중되어 있습니다 (뉴런 모델 정의 → 논리 함수 구현 → 정리 증명).
  The core is concentrated in Sections 1–3 (neuron model definition → logic function implementation → theorem proof).
- 수학적 증명이 어려우면 **Figure(그림)와 예시 회로**에 먼저 집중하세요.
  If the mathematical proofs are difficult, **focus on the figures and example circuits first**.
- "이 간단한 모델로 모든 논리를 계산할 수 있다"는 핵심 메시지를 놓치지 마세요.
  Don't miss the core message: "this simple model can compute all logic."

---

## Q&A

### Q1. 신경과학 기초 용어 / Neuroscience Basic Terms

논문을 이해하려면 실제 생물학적 뉴런의 구조를 먼저 알아야 합니다. 논문의 수학적 모델은 아래 구조를 극도로 단순화한 것입니다.

To understand the paper, one must first know the structure of actual biological neurons. The mathematical model in the paper is an extreme simplification of the structure below.

#### Neuron (뉴런 / 신경세포)

뇌의 기본 정보 처리 단위입니다. 인간의 뇌에는 약 860억 개의 뉴런이 있습니다.

The basic information processing unit of the brain. The human brain has approximately 86 billion neurons.

뉴런의 핵심 동작은 **"all-or-none"** 원리입니다:
- 입력 신호의 합이 특정 임계값을 넘으면 → **발화(fire)** (전기 신호 발생)
- 넘지 못하면 → **아무 일도 안 일어남**

The key operation of a neuron follows the **"all-or-none"** principle:
- If the sum of input signals exceeds a threshold → **fire** (electrical signal generated)
- If not → **nothing happens**

이것이 바로 McCulloch & Pitts가 **이진(0 또는 1) 모델**로 추상화한 근거입니다.

This is the basis for McCulloch & Pitts' abstraction into a **binary (0 or 1) model**.

#### Axon (축삭돌기)

뉴런의 **출력 케이블**입니다.

The **output cable** of a neuron.

- 뉴런이 발화하면 전기 신호(action potential)가 axon을 따라 전달됩니다.
  When a neuron fires, an electrical signal (action potential) travels along the axon.
- 길이가 수 미터에 달할 수도 있습니다 (예: 척수에서 발끝까지).
  Can be several meters long (e.g., from spinal cord to toes).
- 끝부분이 여러 갈래로 나뉘어 다른 뉴런들에게 신호를 전달합니다.
  The end branches out to transmit signals to other neurons.

```
[뉴런 세포체] ──── axon (긴 케이블) ────┬── → 다음 뉴런 A
                                         ├── → 다음 뉴런 B
                                         └── → 다음 뉴런 C
```

**논문에서의 대응 / Mapping in paper**: axon = 뉴런의 **출력선(output)**

#### Synapse (시냅스)

두 뉴런 사이의 **연결 지점**입니다.

The **connection point** between two neurons.

- Axon의 끝과 다음 뉴런 사이에는 아주 작은 **틈(synaptic cleft)**이 있습니다.
  There is a tiny **gap (synaptic cleft)** between the end of an axon and the next neuron.
- 전기 신호가 이 틈에 도달하면, **신경전달물질(neurotransmitter)**이 방출되어 다음 뉴런에 신호를 전달합니다.
  When an electrical signal reaches this gap, **neurotransmitters** are released to transmit the signal to the next neuron.
- 시냅스에는 두 종류가 있습니다:

| 종류 / Type | 효과 / Effect | 논문에서의 표현 / Paper representation |
|-------------|--------------|--------------------------------------|
| **Excitatory synapse** (흥분성) | 다음 뉴런의 발화를 촉진 / Promotes firing | 양의 입력 / Positive input |
| **Inhibitory synapse** (억제성) | 다음 뉴런의 발화를 차단 / Blocks firing | **절대적 억제 / Absolute inhibition** |

```
뉴런 A ──axon──┐
               synapse (흥분성, +)
뉴런 B ──axon──┤                    → [뉴런 C] → 발화?
               synapse (억제성, -)
뉴런 D ──axon──┘
```

#### 전체 그림: 생물학 → McCulloch-Pitts 모델 / Full Picture: Biology → M-P Model

```
     [생물학적 뉴런]                    [M-P 모델]
     [Biological Neuron]               [M-P Model]

  dendrite (수상돌기)                    입력 x₁, x₂, ...
    = 입력 수신부          ───→         inputs (0 or 1)
         │                                  │
  cell body (세포체)                    Σ (합산) ≥ θ ?
    = 신호 통합부          ───→         threshold comparison
         │                                  │
  axon (축삭돌기)                       출력 y
    = 출력 전달부          ───→         output (0 or 1)
         │                                  │
  synapse (시냅스)                      가중치 연결
    = 다음 뉴런과의 접점   ───→         weighted connection
```

McCulloch & Pitts의 핵심 통찰은 이 복잡한 생물학적 시스템을 **논리적으로 동등한 최소한의 수학 모델**로 줄인 것입니다.

The core insight of McCulloch & Pitts was reducing this complex biological system to **the minimal mathematical model that is logically equivalent**.

---

### Q2. 논리 연산 가이드 / Logic Operations Guide

#### 기본 논리 연산 3가지 / Three Basic Logic Operations

모든 논리 회로와 McCulloch-Pitts 뉴런 네트워크는 이 세 가지 기본 연산의 조합으로 만들어집니다.

All logic circuits and McCulloch-Pitts neuron networks are built from combinations of these three basic operations.

##### AND (논리곱 / Conjunction)

**"둘 다 참이어야 참" / "True only when both are true"**

| A | B | A AND B |
|---|---|---------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | **1** |

- 기호 / Symbol: $A \wedge B$, or $A \cdot B$
- 일상 예시 / Daily example: "비가 오고(AND) 우산이 없으면 → 젖는다" / "If it rains AND I have no umbrella → I get wet"

**M-P 뉴런 구현 / M-P Neuron Implementation**: threshold $\theta = 2$, 흥분성 입력 2개

```
A (excitatory) ──→ ┌─────────┐
                    │  θ = 2  │──→ output
B (excitatory) ──→ └─────────┘

A=1, B=1 → sum=2 ≥ 2 → fire (1)
A=1, B=0 → sum=1 < 2 → no fire (0)
```

##### OR (논리합 / Disjunction)

**"하나라도 참이면 참" / "True if at least one is true"**

| A | B | A OR B |
|---|---|--------|
| 0 | 0 | 0 |
| 0 | 1 | **1** |
| 1 | 0 | **1** |
| 1 | 1 | **1** |

- 기호 / Symbol: $A \vee B$, or $A + B$
- 일상 예시 / Daily example: "현금이 있거나(OR) 카드가 있으면 → 결제 가능" / "If I have cash OR a card → I can pay"

**M-P 뉴런 구현 / M-P Neuron Implementation**: threshold $\theta = 1$, 흥분성 입력 2개

```
A (excitatory) ──→ ┌─────────┐
                    │  θ = 1  │──→ output
B (excitatory) ──→ └─────────┘

A=0, B=1 → sum=1 ≥ 1 → fire (1)
A=0, B=0 → sum=0 < 1 → no fire (0)
```

##### NOT (부정 / Negation)

**"참이면 거짓, 거짓이면 참" / "Inverts true to false and vice versa"**

| A | NOT A |
|---|-------|
| 0 | **1** |
| 1 | 0 |

- 기호 / Symbol: $\sim A$, or $\neg A$, or $\overline{A}$

**M-P 뉴런 구현 / M-P Neuron Implementation**: inhibitory input + always-on input

```
always-on(=1) ──(excitatory)──→ ┌─────────┐
                                 │  θ = 1  │──→ output
A ─────────────(inhibitory) ──→ └─────────┘

A=0 → no inhibition, 1 ≥ 1 → fire (1)
A=1 → inhibition active → no fire (0)
```

논문에서 inhibitory input은 **절대적(absolute)**입니다 — 하나라도 활성화되면 다른 입력이 아무리 많아도 발화가 차단됩니다.

In the paper, inhibitory input is **absolute** — even one active inhibitory input blocks firing regardless of other inputs.

#### 복합 논리 연산 / Compound Logic Operations

##### NAND (NOT + AND)

**"둘 다 참일 때만 거짓" / "False only when both are true"** — AND의 반대 / opposite of AND

| A | B | A NAND B |
|---|---|----------|
| 0 | 0 | **1** |
| 0 | 1 | **1** |
| 1 | 0 | **1** |
| 1 | 1 | 0 |

- 기호 / Symbol: $\sim(A \wedge B)$
- **특별한 이유 / Special reason**: NAND만으로 모든 논리 연산을 구현할 수 있습니다 (functional completeness). NAND alone can implement all logic operations.

**M-P 뉴런 구현 / M-P Neuron Implementation**: 2-layer network

```
A (excitatory) ──→ ┌─────────┐
                    │  θ = 2  │──(inhibitory)──→ ┌─────────┐
B (excitatory) ──→ └─────────┘                   │  θ = 1  │──→ output
                        AND      always-on(=1) ──→└─────────┘
                                                      NOT
```

##### NOR (NOT + OR)

**"둘 다 거짓일 때만 참" / "True only when both are false"** — OR의 반대 / opposite of OR

| A | B | A NOR B |
|---|---|---------|
| 0 | 0 | **1** |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 0 |

- 기호 / Symbol: $\sim(A \vee B)$

**M-P 뉴런 구현 / M-P Neuron Implementation**: 2 inhibitory inputs

```
always-on(=1) ──(excitatory)──→ ┌─────────┐
A ─────────────(inhibitory) ──→ │  θ = 1  │──→ output
B ─────────────(inhibitory) ──→ └─────────┘

A=0, B=0 → no inhibition → 1 ≥ 1 → fire (1)
A=1, B=0 → inhibition active → no fire (0)
```

##### XOR (배타적 논리합 / Exclusive OR)

**"둘 중 하나만 참일 때 참" / "True when exactly one is true"**

| A | B | A XOR B |
|---|---|---------|
| 0 | 0 | 0 |
| 0 | 1 | **1** |
| 1 | 0 | **1** |
| 1 | 1 | 0 |

- 기호 / Symbol: $A \oplus B$
- 수식 / Formula: $(A \vee B) \wedge \sim(A \wedge B)$ — "OR이면서 AND는 아닌 것" / "OR but not AND"

**M-P 뉴런 구현 / M-P Neuron Implementation**: 단일 뉴런으로는 **불가능** → 다층 네트워크 필요! / **Impossible** with a single neuron → requires multi-layer network!

이 XOR 문제가 나중에 **Minsky & Papert (1969, 논문 #4)**에서 단층 perceptron의 한계로 증명되고, **Rumelhart et al. (1986, 논문 #6)**의 backpropagation으로 해결됩니다.

This XOR problem was later proven as a limitation of single-layer perceptrons by **Minsky & Papert (1969, paper #4)** and resolved by **Rumelhart et al. (1986, paper #6)** with backpropagation.

#### 논문에서의 표현 방식 / Notation in the Paper

McCulloch & Pitts는 이 논리 연산들을 **시간을 포함한 명제 논리(temporal propositional logic)**로 표기합니다.

McCulloch & Pitts express these logic operations using **temporal propositional logic**.

논문의 핵심 표기법 / Key notation in the paper:

$$N_i(t)$$

이것은 "뉴런 $i$가 시간 $t$에서 발화한다"는 명제입니다.

This is the proposition "neuron $i$ fires at time $t$."

| 우리가 아는 표현 / Common expression | 논문의 표현 / Paper's notation | 의미 / Meaning |
|--------------------------------------|-------------------------------|----------------|
| $A$ AND $B$ → $C$ | $N_3(t) \equiv N_1(t-1) \cdot N_2(t-1)$ | 뉴런 1,2가 $t-1$에 발화하면 뉴런 3이 $t$에 발화 / If neurons 1,2 fire at $t-1$, neuron 3 fires at $t$ |
| $A$ OR $B$ → $C$ | $N_3(t) \equiv N_1(t-1) \vee N_2(t-1)$ | 뉴런 1 또는 2가 $t-1$에 발화하면 뉴런 3이 $t$에 발화 / If neuron 1 or 2 fires at $t-1$, neuron 3 fires at $t$ |
| NOT $A$ → $C$ | $N_2(t) \equiv \sim N_1(t-1)$ | 뉴런 1이 $t-1$에 발화하지 않으면 뉴런 2가 $t$에 발화 / If neuron 1 doesn't fire at $t-1$, neuron 2 fires at $t$ |

주목할 점은 **항상 $(t-1)$이 등장**한다는 것입니다. 이것이 McCulloch-Pitts 모델의 핵심 가정 중 하나인 **synaptic delay** — 신호가 시냅스를 거치는 데 정확히 1단위 시간이 걸린다는 것입니다.

Note that **$(t-1)$ always appears**. This is one of the key assumptions of the McCulloch-Pitts model: **synaptic delay** — a signal takes exactly one time unit to cross a synapse.

#### 논문의 핵심 정리와의 연결 / Connection to the Paper's Main Theorem

논문은 이 모든 것을 종합하여 다음을 증명합니다:

The paper synthesizes all of this to prove:

> **Theorem**: 시간적 명제 논리로 표현 가능한 모든 관계는 McCulloch-Pitts 뉴런 네트워크로 실현할 수 있다.
>
> **Theorem**: Any relation expressible in temporal propositional logic can be realized by a McCulloch-Pitts neuron network.

즉, AND, OR, NOT의 조합으로 표현 가능한 **어떤 논리 함수든** 뉴런 네트워크로 만들 수 있다는 것입니다. 이것이 **"신경망은 범용 계산 장치(universal computing device)가 될 수 있다"**는 아이디어의 수학적 근거가 되었습니다.

In other words, **any logical function** expressible as a combination of AND, OR, NOT can be built with a neuron network. This became the mathematical basis for the idea that **"neural networks can be universal computing devices."**

---

### Q3. 논리 연산 종합 요약표 / Logic Operations Summary Table

#### 진리표 종합 / Combined Truth Table

| A | B | AND | OR | NOT A | NAND | NOR | XOR |
|---|---|-----|----|-------|------|-----|-----|
| 0 | 0 |  0  |  0 |   1   |   1  |  1  |  0  |
| 0 | 1 |  0  |  1 |   1   |   1  |  0  |  1  |
| 1 | 0 |  0  |  1 |   0   |   1  |  0  |  1  |
| 1 | 1 |  1  |  1 |   0   |   0  |  0  |  0  |

#### 특성 비교표 / Properties Comparison Table

| 연산 / Op | 기호 / Symbol | 한국어 이름 | 의미 / Meaning | 수식 / Formula | M-P 구현 / M-P Implementation |
|-----------|---------------|-------------|----------------|----------------|-------------------------------|
| **AND** | $A \wedge B$ | 논리곱 | 둘 다 참이면 참 / True if both true | — | $\theta = 2$, excitatory × 2 |
| **OR** | $A \vee B$ | 논리합 | 하나라도 참이면 참 / True if any true | — | $\theta = 1$, excitatory × 2 |
| **NOT** | $\sim A$ | 부정 | 반전 / Invert | — | inhibitory × 1 + always-on |
| **NAND** | $\sim(A \wedge B)$ | 부정 논리곱 | AND의 반대 / NOT of AND | $\sim(A \wedge B)$ | 2-layer: AND → NOT |
| **NOR** | $\sim(A \vee B)$ | 부정 논리합 | OR의 반대 / NOT of OR | $\sim(A \vee B)$ | $\theta = 1$, inhibitory × 2 + always-on |
| **XOR** | $A \oplus B$ | 배타적 논리합 | 하나만 참일 때 참 / True if exactly one true | $(A \vee B) \wedge \sim(A \wedge B)$ | multi-layer required |

#### 핵심 관계도 / Key Relationships

```
기본 연산 (Basic)           복합 연산 (Compound)
─────────────────          ─────────────────────
  AND ──── NOT(AND) ────→ NAND
  OR  ──── NOT(OR)  ────→ NOR
  AND + OR + NOT    ────→ XOR = (A OR B) AND NOT(A AND B)
```

#### 단일 뉴런 구현 가능 여부 / Single Neuron Implementability

| 연산 / Op | 단일 뉴런 가능? / Single neuron? | 이유 / Reason |
|-----------|----------------------------------|---------------|
| AND | Yes | linearly separable |
| OR | Yes | linearly separable |
| NOT | Yes | inhibitory input |
| NAND | No (2 neurons) | NOT(AND) — requires 2 layers |
| NOR | Yes | 2 inhibitory inputs + always-on |
| XOR | **No (multi-layer)** | **not linearly separable** — 이것이 Minsky & Papert (1969)의 핵심 발견 / This is the key finding of Minsky & Papert (1969) |

> **linearly separable (선형 분리 가능)**이란: 2차원 평면에서 직선 하나로 출력이 1인 점과 0인 점을 나눌 수 있는가?
> AND와 OR는 가능하지만, XOR는 불가능합니다 — 이것이 단일 뉴런(= 단층 perceptron)의 근본적 한계입니다.
>
> **Linearly separable** means: can a single straight line in 2D separate the output-1 points from the output-0 points?
> AND and OR can, but XOR cannot — this is the fundamental limitation of a single neuron (= single-layer perceptron).
