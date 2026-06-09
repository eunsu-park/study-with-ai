# Pre-Reading Briefing: The Perceptron (1958)
# 사전 읽기 브리핑: 퍼셉트론 (1958)

**Paper**: *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*
**Author**: Frank Rosenblatt
**Year**: 1958
**Journal**: *Psychological Review*, Vol. 65, No. 6, pp. 386–408
**Pages**: 23 pages

---

## 핵심 기여 / Core Contribution

McCulloch & Pitts(1943)의 뉴런 모델이 "논리적으로 가능하다"는 것을 보여주었다면, Rosenblatt의 perceptron은 **실제로 데이터로부터 학습할 수 있는** 최초의 신경망 모델입니다. 핵심 아이디어는 단순합니다: 네트워크의 연결 강도(value)를 경험에 따라 조정하면, 시스템이 패턴을 자동으로 분류할 수 있게 됩니다. Rosenblatt은 이를 엄밀한 확률론적 프레임워크로 분석하여, 학습 곡선을 물리적 파라미터(뉴런 수, 연결 수, 임계값 등)로부터 **예측**할 수 있음을 보였습니다. 이는 "명시적으로 프로그래밍되지 않고도 기계가 학습한다"는 개념을 처음으로 수학적으로 정당화한 것입니다.

While McCulloch & Pitts (1943) showed that neural networks *could* compute logical functions, Rosenblatt's perceptron was the first model that could actually **learn from data**. The key insight: by adjusting connection strengths ("values") based on experience, a network of simple units can learn to classify patterns automatically. Rosenblatt analyzed this rigorously using probability theory, showing that learning curves can be *predicted* from physical parameters (number of neurons, connections, thresholds). This was the first mathematical justification that machines could learn without being explicitly programmed.

---

## 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1958년, AI라는 분야는 이제 겨우 2살이었습니다(1956 Dartmouth Conference에서 공식 탄생). 초기 AI는 두 갈래로 나뉘었습니다:
- **기호주의 (Symbolicism)**: 논리와 규칙으로 지능을 구현 (McCarthy, Minsky 등)
- **연결주의 (Connectionism)**: 뉴런 네트워크의 학습으로 지능 구현 (Rosenblatt)

Rosenblatt은 연결주의의 최초 챔피언이었습니다. 그의 perceptron은 뉴욕 타임즈에서 "태아의 뇌"에 비유되며 큰 센세이션을 일으켰습니다. 하지만 이 흥분은 11년 후 Minsky & Papert(1969)의 비판으로 급격히 식게 됩니다.

In 1958, AI was barely two years old. Rosenblatt championed the connectionist approach — learning through neural networks — against the dominant symbolic AI approach. His perceptron caused a sensation (the New York Times compared it to an "electronic brain"), but the excitement would be dramatically deflated by Minsky & Papert's critique in 1969.

### Rosenblatt의 배경 / Rosenblatt's Background

Frank Rosenblatt(1928–1971)은 Cornell 대학교에서 심리학 박사 학위를 받았으며, Cornell Aeronautical Laboratory에서 근무했습니다. 그는 심리학자이자 신경과학자의 관점에서 뇌의 정보 처리를 이해하려 했습니다 — 이것이 논문이 *Psychological Review*에 실린 이유입니다.

### 타임라인 / Timeline

```
1943  McCulloch & Pitts — 최초의 인공 뉴런 모델 (논문 #1)
  │
1949  Hebb — "The Organization of Behavior" (Hebbian learning: 함께 발화하면 연결 강화)
  │
1950  Turing — "기계가 생각할 수 있는가?" (논문 #2)
  │
1956  Dartmouth Conference — "AI" 용어 탄생
  │
1957  Rosenblatt — Mark I Perceptron 하드웨어 구현 시작
  │
1958  ★ ROSENBLATT — "THE PERCEPTRON" ★
  │
1960  Widrow & Hoff — ADALINE (적응형 선형 뉴런, LMS 알고리즘)
  │
1969  Minsky & Papert — "Perceptrons" (한계 증명 → AI 겨울, 다음 논문 #4)
  │
1982  Hopfield — 신경망 부활 (논문 #5)
  │
1986  Rumelhart et al. — Backpropagation으로 다층 네트워크 학습 (논문 #6)
```

### 이전 논문들과의 연결 / Connection to Previous Papers

| 논문 | 관계 |
|------|------|
| #1 McCulloch & Pitts (1943) | 논리적 뉴런 모델 → Rosenblatt이 이를 **학습 가능한** 모델로 확장 |
| #2 Turing (1950) | "기계가 학습할 수 있는가?" → Rosenblatt이 **구체적 메커니즘** 제시 |

---

## 필요한 배경 지식 / Prerequisites

### 1. McCulloch-Pitts 뉴런 복습 / M-P Neuron Review

논문 #1에서 배운 모델을 떠올려봅시다:

$$y = \begin{cases} 1 & \text{if } \sum_i w_i x_i \geq \theta \\ 0 & \text{otherwise} \end{cases}$$

McCulloch-Pitts 뉴런은 입력의 가중합이 임계값($\theta$)을 넘으면 발화합니다. 하지만 **가중치($w_i$)가 고정**되어 있었습니다 — 학습이 없었습니다.

Rosenblatt의 핵심 혁신: **가중치(value)를 경험에 따라 변경**할 수 있게 만든 것입니다.

### 2. 기초 확률론 / Basic Probability

Rosenblatt은 논리학 대신 **확률론**을 사용합니다. 필요한 개념:

- **조건부 확률 (Conditional probability)**: $P(A|B)$ — B가 주어졌을 때 A의 확률
- **정규분포 (Normal distribution)**: $\Phi(Z)$ — 표준 정규분포의 누적 분포 함수
  - $Z > 0$ 이면 $\Phi(Z) > 0.5$ (랜덤보다 나음)
  - $Z$가 클수록 성능이 좋아짐
- **이항분포 (Binomial distribution)**: $\binom{n}{k} p^k (1-p)^{n-k}$ — n번 시행에서 k번 성공할 확률

### 3. 기초 선형대수 / Basic Linear Algebra

- **벡터 (Vector)**: 숫자의 목록 $\mathbf{x} = [x_1, x_2, ..., x_n]$
- **내적 (Dot product)**: $\mathbf{w} \cdot \mathbf{x} = \sum_i w_i x_i$
- 내적이 임계값을 넘으면 뉴런이 발화 — 이것이 perceptron의 핵심 연산

### 4. 정보 저장의 두 관점 / Two Views of Information Storage

Rosenblatt이 논문 서두에서 대비하는 두 철학을 이해해야 합니다:

| 관점 | 기억 방식 | 인식 방식 | 비유 |
|------|-----------|-----------|------|
| **표상주의 (Representational)** | 자극의 코딩된 복사본 저장 | 들어오는 자극과 저장된 복사본 비교 | 사진을 찍어두고 비교 |
| **연결주의 (Connectionist)** | 새로운 연결/경로 형성 | 자극이 기존 경로를 자동 활성화 | 자주 다니면 길이 만들어짐 |

Rosenblatt은 **연결주의** 입장을 취합니다. 정보는 "연결의 패턴"으로 저장됩니다.

---

## 핵심 용어 / Key Vocabulary

논문에서 사용하는 용어들은 현대 용어와 다소 다릅니다. 미리 정리해 두겠습니다:

| Rosenblatt의 용어 | 현대 용어 | 의미 |
|-------------------|-----------|------|
| **S-points** (sensory units) | Input layer | 감각 입력을 받는 유닛 (예: 망막의 광수용체) |
| **A-units** (association cells) | Hidden layer neurons | 입력과 출력 사이에서 연관을 형성하는 유닛 |
| **R-units** (response units) | Output layer | 최종 분류 결과를 나타내는 유닛 |
| **Value** ($V$) | Weight / connection strength | A-unit의 출력 강도 — 학습으로 변하는 핵심 변수 |
| **Threshold** ($\theta$) | Activation threshold | 뉴런이 발화하기 위한 최소 입력합 |
| **Source-set** | Receptive field (for outputs) | 특정 R-unit에 연결된 A-unit들의 집합 |
| **Origin points** | Receptive field (for A-units) | 특정 A-unit에 연결된 S-point들의 집합 |
| **Reinforcement** | Training signal | 올바른/틀린 반응에 대한 피드백 |
| **$P_a$** | Activation probability | 자극에 의해 A-unit이 활성화될 확률 |
| **$P_c$** | Co-activation probability | 같은 A-unit이 두 자극 모두에 반응할 조건부 확률 |
| **$P_r$** | Recall accuracy | 이전에 본 자극을 올바르게 분류할 확률 |
| **$P_g$** | Generalization accuracy | **새로운** 자극을 올바르게 분류할 확률 |
| **Ideal environment** | Random/unstructured data | 자극 간 구조적 유사성이 없는 환경 |
| **Differentiated environment** | Structured/clustered data | 자극이 구별 가능한 클래스로 나뉘는 환경 |
| **Statistical separability** | Linear separability (approx.) | 두 클래스의 자극이 통계적으로 구분 가능한 정도 |

---

## 수식 미리보기 / Equations Preview

### 1. Perceptron의 기본 연산 / Basic Perceptron Operation

A-unit은 McCulloch-Pitts 뉴런과 같은 방식으로 작동합니다:

$$a = e - i = \sum_{j \in \text{excitatory}} x_j - \sum_{k \in \text{inhibitory}} x_k$$

$$\text{A-unit fires if } a \geq \theta$$

여기서:
- $e$: 흥분성(excitatory) 입력의 합
- $i$: 억제성(inhibitory) 입력의 합
- $\theta$: 임계값 (threshold)

**직관**: 뉴런은 "찬성표 - 반대표 ≥ 필요 표수"이면 활성화됩니다.

### 2. 활성화 확률 $P_a$ / Activation Probability

$$P_a = \sum_{e,i} P(e,i) \cdot \mathbb{1}[e - i \geq \theta]$$

이것은 "특정 크기의 자극이 주어졌을 때, 임의의 A-unit이 활성화될 확률"입니다. $P_a$는 다음에 의존합니다:
- $R$: 자극이 차지하는 망막 비율
- $x$: 흥분성 연결 수
- $y$: 억제성 연결 수
- $\theta$: 임계값

**직관**: 자극이 커지면(R↑) $P_a$가 증가하고, 임계값이 높으면(θ↑) $P_a$가 감소합니다.

### 3. 핵심 학습 방정식 / The Learning Equation

논문의 가장 중요한 수식 — 모든 학습 곡선을 하나의 방정식으로 통합합니다:

$$P = P(N_{ar} > 0) \cdot \Phi(Z)$$

여기서:

$$P(N_{ar} > 0) = 1 - (1 - P_a)^{N_e}$$

$$Z = \frac{c_1 n_{sr} + c_2}{\sqrt{c_3^2 n_{sr} + c_4^2}}$$

- $P(N_{ar} > 0)$: source-set에서 최소 하나의 유닛이 활성화될 확률
- $\Phi(Z)$: 표준 정규분포 누적 함수 ($Z > 0$이면 $> 0.5$, 즉 랜덤보다 나음)
- $n_{sr}$: 각 반응에 연관된 자극의 수 (= 학습량)
- $c_1, c_2, c_3, c_4$: 시스템 유형과 환경에 따른 상수

**직관**: 학습이 진행될수록($n_{sr}$↑) $Z$가 증가하고, $\Phi(Z)$가 1에 가까워지며, 정확도가 올라갑니다. 단, 이상적 환경에서는 $c_1 = 0$이므로 일반화($P_g$)가 불가능합니다.

### 4. 일반화의 조건 / Condition for Generalization

**차별화된 환경**에서 perceptron이 일반화 가능하려면:

$$P_{c12} < P_a < P_{c11}$$

- $P_{c11}$: **같은** 클래스 내 자극 쌍 간의 공동 활성화 확률 (클래스 내 유사성)
- $P_{c12}$: **다른** 클래스 자극 간의 공동 활성화 확률 (클래스 간 유사성)

**직관**: "같은 클래스의 자극들은 비슷하게 처리되고, 다른 클래스의 자극들은 다르게 처리될 때" 일반화가 가능합니다. 이것은 현대 머신러닝의 intra-class variance < inter-class variance 조건과 동일합니다!

### 5. 세 가지 학습 시스템 비교 / Three Learning Systems

| 시스템 | Value 변화 규칙 | 총 value | 성능 |
|--------|----------------|----------|------|
| **α-system** | 활성 유닛이 +1 value 획득 | 계속 증가 | 보통 |
| **β-system** | 각 source-set에 일정한 총 gain 배분 | 계속 증가 | 가장 낮음 |
| **γ-system** | 활성 유닛이 비활성 유닛으로부터 value 흡수 | **일정** (보존) | **가장 높음** |

**직관**: γ-system은 현대의 **competitive learning** / **winner-take-all** 메커니즘과 유사합니다. 제한된 자원을 두고 경쟁하므로, 관련 없는 연결은 자연스럽게 약해집니다.

---

## 논문의 구조 미리보기 / Paper Structure Preview

| 섹션 | 내용 | 난이도 |
|------|------|--------|
| pp. 386–388 | **철학적 도입**: 정보 저장의 두 관점 (표상주의 vs 연결주의) | ⭐ 쉬움 |
| pp. 388–391 | **Perceptron 구조**: S-points → A-units → R-units 3층 구조 설명 | ⭐⭐ 보통 |
| pp. 391–392 | **세 가지 시스템 (α, β, γ)**: 학습 규칙의 변형 | ⭐⭐ 보통 |
| pp. 392–394 | **$P_a$와 $P_c$ 분석**: 활성화/공동활성화 확률의 수학적 유도 | ⭐⭐⭐ 어려움 |
| pp. 394–400 | **학습 방정식**: 이상적/차별화 환경에서의 학습 곡선 | ⭐⭐⭐ 어려움 |
| pp. 402–404 | **이진 응답 & 개선된 perceptron**: 시간 패턴, 자발적 조직화 | ⭐⭐ 보통 |
| pp. 404–408 | **결론과 평가**: 10가지 결론, 기존 이론과의 비교 | ⭐ 쉬움 |

### 읽기 전략 / Reading Strategy

1. **pp. 386–391** (철학 + 구조)를 꼼꼼히 읽으세요 — 이것이 논문의 핵심 아이디어입니다
2. **pp. 392–400** (수학 분석)은 전체 수식을 따라가지 않아도 됩니다. **결과와 직관**에 집중하세요
3. **pp. 404–408** (결론)을 꼼꼼히 읽으세요 — Rosenblatt 자신이 perceptron의 한계를 인정합니다

---

## 읽으면서 생각해 볼 질문 / Questions to Consider While Reading

1. McCulloch-Pitts 뉴런과 perceptron의 **근본적 차이**는 무엇인가? (힌트: 학습)
2. 왜 Rosenblatt은 Boolean 논리가 아닌 **확률론**을 사용하는가?
3. **이상적 환경** vs **차별화된 환경** — 왜 이 구분이 중요한가?
4. γ-system이 α, β보다 우수한 이유는 무엇인가? (현대적 관점에서 해석해보기)
5. Rosenblatt은 perceptron의 어떤 **한계**를 스스로 인정하는가?
6. 이 논문이 #4 Minsky & Papert(1969)의 비판으로 어떻게 이어지는지 예상해보기
