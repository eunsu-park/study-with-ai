---
title: "Learning Representations by Back-propagating Errors"
authors: David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
year: 1986
journal: "Nature, Vol. 323, pp. 533–536"
doi: "10.1038/323533a0"
topic: Artificial Intelligence / Learning Algorithms
tags: [backpropagation, gradient descent, chain rule, hidden units, internal representation, multi-layer network, sigmoid, momentum, XOR, distributed representation]
status: completed
date_started: 2026-04-09
date_completed: 2026-04-09
---

# Learning Representations by Back-propagating Errors (1986)
# 오류 역전파를 통한 표현 학습 (1986)

**David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams**
UC San Diego; Carnegie-Mellon University

---

## Core Contribution / 핵심 기여

Rumelhart, Hinton & Williams는 **다층 신경망(multi-layer neural network)** 의 모든 층에 대한 가중치 기울기(gradient)를 **연쇄 법칙(chain rule)** 을 체계적으로 적용하여 계산하는 **backpropagation(오류 역전파)** 알고리즘을 기술하고 대중화했습니다. 이 알고리즘의 핵심 통찰은 출력층의 오류 신호를 네트워크를 거꾸로(backward) 전파하면, **숨겨진 유닛(hidden units)** — 입력도 출력도 아닌 중간층 뉴런 — 의 가중치까지 조정할 수 있다는 것입니다. 이로써 hidden units는 과제 수행에 유용한 **내부 표현(internal representation)** 을 명시적 프로그래밍 없이 자동으로 발견하게 됩니다. 이 논문은 Minsky & Papert(1969)가 제기한 "다층 네트워크를 어떻게 학습시키는가?"라는 핵심 문제에 대한 직접적 해답이며, 현대 딥러닝의 근본 학습 알고리즘을 확립했습니다.

Rumelhart, Hinton & Williams described and popularized the **backpropagation** algorithm for computing weight gradients in **multi-layer neural networks** through systematic application of the **chain rule**. The key insight is that propagating output error signals backwards through the network enables adjusting weights of **hidden units** — neurons in intermediate layers that are neither input nor output. This allows hidden units to automatically discover **internal representations** useful for the task, without explicit programming. The paper directly answers the core question raised by Minsky & Papert (1969) — "How do you train multi-layer networks?" — and established the fundamental learning algorithm of modern deep learning.

---

## Reading Notes / 읽기 노트

### Introduction (p.533, col.1) — 문제 정의: 왜 Hidden Units이 필요한가

**Perceptron의 한계와 다층의 필요성 / Perceptron limits and the need for multi-layer:**

신경망의 시냅스(가중치)를 자동으로 조정하는 강력한 규칙을 찾는 것이 오랜 과제였습니다. 입력 유닛이 직접 출력 유닛에 연결된 경우(단층), 학습 규칙을 찾는 것은 비교적 쉽습니다 — 입력-출력 쌍을 반복적으로 제시하며 가중치를 조정하면 됩니다(Rosenblatt의 perceptron convergence theorem). 그러나 "더 흥미로운" 경우는 **hidden units** — 입력도 출력도 아닌 내부 유닛 — 을 도입할 때입니다.

Finding a powerful rule for automatically adjusting neural network synapses (weights) has been a longstanding challenge. When input units connect directly to output units (single layer), finding learning rules is relatively easy — repeatedly present input-output pairs and adjust weights (Rosenblatt's perceptron convergence theorem). But the "more interesting" case is when **hidden units** — internal units that are neither input nor output — are introduced.

**Hidden units이 "흥미로운" 이유 / Why hidden units are "interesting":**

Hidden units이 있으면 과제가 근본적으로 바뀝니다: 목표 출력은 hidden units의 상태를 직접 지정하지 않습니다. Perceptron에서는 입력 연결이 수동으로 고정되므로 "진정한 hidden units"이 아닙니다 — 그 상태가 입력에 의해 완전히 결정됩니다. 학습 절차는 hidden units가 **어떤 상황에서 활성화되어야 하는지** 스스로 결정해야 합니다. 이것은 "hidden units가 무엇을 표현해야 하는지 결정하는 것"과 동일하며, 저자들은 "비교적 단순한 범용 절차가 적절한 내부 표현을 구성할 만큼 강력하다"는 것을 보여주겠다고 선언합니다.

With hidden units, the task changes fundamentally: desired outputs don't directly specify hidden unit states. In perceptrons, input connections are fixed by hand, so they aren't "true hidden units" — their states are completely determined by input. The learning procedure must decide on its own **under what circumstances hidden units should be active**. This is equivalent to "deciding what hidden units should represent," and the authors declare they will show "a general purpose and relatively simple procedure is powerful enough to construct appropriate internal representations."

---

### The Learning Procedure: Forward Pass (p.533, col.2) — 순전파

**네트워크 구조 / Network architecture:**

가장 단순한 형태: 맨 아래 입력층(input layer), 중간의 임의 수의 은닉층(hidden layers), 맨 위 출력층(output layer). 같은 층 내의 연결은 금지되고, 상위 층에서 하위 층으로의 연결도 금지됩니다. 그러나 인접하지 않은 층 사이의 연결("layer skipping")은 허용됩니다. 입력 벡터를 입력 유닛에 제시하면, 각 층의 유닛 상태가 **순차적으로 아래에서 위로** 결정됩니다.

Simplest form: input layer at bottom, any number of hidden layers in the middle, output layer at top. Connections within a layer are forbidden; connections from higher to lower layers are also forbidden. But connections skipping layers are allowed. When an input vector is presented to input units, unit states are determined **sequentially from bottom to top**.

**총 입력 계산 (Eq. 1) / Total input computation:**

$$x_j = \sum_i y_i w_{ji}$$

유닛 $j$의 총 입력 $x_j$는 하위 층에서 연결된 모든 유닛 $i$의 출력 $y_i$에 가중치 $w_{ji}$를 곱하여 합한 것입니다. **Bias 처리의 우아함**: bias를 별도 항으로 다루지 않고, 항상 출력이 1인 추가 유닛을 도입합니다. 이 유닛에 대한 가중치가 곧 bias이며, 다른 가중치와 완전히 동일하게 학습됩니다. 이 트릭은 수학적 표기를 단순화하고, 구현도 깔끔해집니다.

The total input $x_j$ to unit $j$ is the weighted sum of outputs $y_i$ from all connected units $i$ in lower layers, times weights $w_{ji}$. **Elegance of bias handling**: instead of a separate bias term, introduce an extra unit that always outputs 1. Its weight becomes the bias, learned identically to all other weights. This trick simplifies both notation and implementation.

**비선형 활성화 (Eq. 2) / Nonlinear activation:**

$$y_j = \frac{1}{1 + e^{-x_j}}$$

Sigmoid(logistic) 함수. 핵심 속성 두 가지:

The sigmoid (logistic) function. Two key properties:

1. **비선형성(nonlinearity)**: 0~1 범위로 "압축". 선형 함수만 사용하면 다층이 단층과 동일해짐(행렬 곱의 합성은 또 하나의 행렬 곱)

   Squashes to 0–1 range. With only linear functions, multi-layer collapses to single layer (composition of linear maps is linear)

2. **미분의 깔끔함**: $dy_j/dx_j = y_j(1 - y_j)$. 이 성질이 backprop을 실용적으로 만듬 — $y_j$ 자체로 미분을 계산할 수 있어 별도 저장이 불필요

   Clean derivative: $dy_j/dx_j = y_j(1 - y_j)$. Makes backprop practical — derivative computed from $y_j$ itself, no extra storage needed

저자들은 "Eq. 1, 2의 정확한 함수를 사용할 필요는 없다"고 명시합니다 — 유계(bounded) 미분을 가진 어떤 입출력 함수도 가능합니다. 이것은 후에 ReLU, tanh 등의 활성화 함수가 사용될 수 있는 이론적 근거입니다.

The authors state "it is not necessary to use exactly the functions given in equations (1) and (2)" — any input-output function with a bounded derivative works. This provides the theoretical basis for later use of ReLU, tanh, etc.

---

### The Learning Procedure: Error Function and Backward Pass (pp.533–535) — 오류 함수와 역전파

이 섹션이 논문의 **수학적 핵심**입니다.

This section is the **mathematical core** of the paper.

**오류 함수 정의 (Eq. 3) / Error function definition:**

$$E = \frac{1}{2} \sum_c \sum_j (y_{j,c} - d_{j,c})^2$$

모든 학습 케이스 $c$에 대해, 출력 유닛 $j$의 실제 출력 $y_{j,c}$와 목표 출력 $d_{j,c}$의 차이를 제곱하여 합산합니다. $1/2$는 미분 시 2와 상쇄시키기 위한 상수입니다. 이 오류를 최소화하는 것이 학습의 목표입니다.

Sum of squared differences between actual output $y_{j,c}$ and desired output $d_{j,c}$ over all cases $c$ and output units $j$. The $1/2$ cancels the 2 from differentiation. Minimizing this error is the learning objective.

Hopfield(1982)의 에너지 함수 $E$와 직접적 유사성: Hopfield에서는 네트워크 동역학이 $E$를 최소화했지만, 여기서는 **학습 알고리즘(gradient descent)** 이 $E$를 최소화합니다.

Direct analogy to Hopfield's (1982) energy function $E$: in Hopfield, network dynamics minimized $E$; here, the **learning algorithm (gradient descent)** minimizes $E$.

**출력 유닛에 대한 기울기 (Eq. 4–5) / Gradient for output units:**

$$\frac{\partial E}{\partial y_j} = y_j - d_j \quad \text{(Eq. 4)}$$

$$\frac{\partial E}{\partial x_j} = \frac{\partial E}{\partial y_j} \cdot \frac{dy_j}{dx_j} = (y_j - d_j) \cdot y_j(1 - y_j) \quad \text{(Eq. 5)}$$

Eq. 4는 직관적: 오류는 (실제 - 목표). Eq. 5는 chain rule의 첫 적용: sigmoid의 미분 $y_j(1-y_j)$을 곱합니다.

Eq. 4 is intuitive: error is (actual - desired). Eq. 5 is the first chain rule application: multiply by sigmoid derivative $y_j(1-y_j)$.

**$y_j(1-y_j)$의 중요한 의미 / Significance of $y_j(1-y_j)$:**

이 항은 "게이팅" 효과를 가집니다. $y_j \approx 0$ 또는 $y_j \approx 1$ (유닛이 확신을 갖고 있을 때)이면 $y_j(1-y_j) \approx 0$ → 기울기가 거의 0 → 가중치가 거의 변하지 않습니다. $y_j \approx 0.5$ (불확실할 때)이면 $y_j(1-y_j) = 0.25$ (최댓값) → 가장 크게 학습합니다. 이것은 직관적으로 타당합니다: 이미 확신하는 유닛은 변할 이유가 없고, 불확실한 유닛이 가장 많이 배워야 합니다. 그러나 이 성질이 나중에 **기울기 소실(vanishing gradient) 문제**의 원인이 됩니다 — 깊은 네트워크에서 여러 층을 거치며 $y(1-y) \leq 0.25$가 계속 곱해져 기울기가 지수적으로 줄어듭니다.

This term has a "gating" effect. When $y_j \approx 0$ or $y_j \approx 1$ (high confidence): $y_j(1-y_j) \approx 0$ → gradient near 0 → almost no weight change. When $y_j \approx 0.5$ (uncertain): $y_j(1-y_j) = 0.25$ (maximum) → most learning. Intuitively sound: confident units need not change, uncertain units should learn most. However, this property later causes the **vanishing gradient problem** — through many layers of deep networks, repeated multiplication by $y(1-y) \leq 0.25$ exponentially shrinks gradients.

**가중치에 대한 기울기 (Eq. 6) / Gradient for weights:**

$$\frac{\partial E}{\partial w_{ji}} = \frac{\partial E}{\partial x_j} \cdot y_i$$

가중치 $w_{ji}$의 오류 기여도 = (유닛 $j$의 오류 신호) × (유닛 $i$의 출력). 이것은 **Hebbian 학습과의 연결**을 보여줍니다: "오류가 있는 유닛 $j$에 활성화된 유닛 $i$가 연결되어 있으면, 그 연결을 조정하라". Hopfield의 Hebbian 저장 규칙 $T_{ij} \propto V_i V_j$와 구조적으로 유사하지만, 여기서는 상관(correlation) 대신 **오류 기울기**를 사용합니다.

Weight error contribution = (error signal for unit $j$) × (output of unit $i$). This shows a **connection to Hebbian learning**: "if unit $j$ has error and unit $i$ is active, adjust that connection." Structurally similar to Hopfield's Hebbian prescription $T_{ij} \propto V_i V_j$, but using **error gradient** instead of correlation.

**숨겨진 층으로의 역전파 (Eq. 7) — 논문의 가장 중요한 수식 / Backpropagation to hidden layers — the paper's most important equation:**

$$\frac{\partial E}{\partial y_i} = \sum_j \frac{\partial E}{\partial x_j} \cdot w_{ji}$$

**이 한 줄이 전체 논문의 핵심입니다.** 숨겨진 유닛 $i$에 대한 오류를 직접 알 수 없습니다 — 목표 출력이 없으니까요. 하지만 $i$가 연결된 **다음 층의 모든 유닛** $j$의 오류 기울기 $\partial E/\partial x_j$를 이미 알고 있다면, 이를 연결 가중치 $w_{ji}$로 **가중합**하여 $i$의 오류를 역산합니다.

**This single line is the core of the entire paper.** We cannot directly know the error for hidden unit $i$ — there is no target output. But if we already know the error gradients $\partial E/\partial x_j$ for **all next-layer units** $j$ that $i$ connects to, we **back-calculate** $i$'s error by taking the weighted sum using connection weights $w_{ji}$.

**재귀적 적용 / Recursive application:**

Eq. 5와 Eq. 7을 교대로 적용하면 어떤 깊이의 층이든 기울기를 계산할 수 있습니다:

Alternately applying Eq. 5 and Eq. 7 computes gradients for layers of any depth:

1. 출력층: Eq. 4 → Eq. 5로 $\partial E/\partial x_j$ 계산 / Output: compute $\partial E/\partial x_j$ via Eq. 4 → 5
2. 마지막 은닉층: Eq. 7로 $\partial E/\partial y_i$ → Eq. 5로 $\partial E/\partial x_i$ / Last hidden: Eq. 7 → $\partial E/\partial y_i$ → Eq. 5 → $\partial E/\partial x_i$
3. 그 다음 은닉층: 동일 과정 반복... / Next hidden: repeat...
4. 모든 가중치: Eq. 6으로 $\partial E/\partial w_{ji}$ / All weights: Eq. 6 → $\partial E/\partial w_{ji}$

이것이 "back-propagation"이라는 이름의 유래입니다: 오류가 네트워크를 **뒤로** 전파됩니다.

This is the origin of the name "back-propagation": error propagates **backwards** through the network.

---

### Weight Update Rules (p.535) — 가중치 갱신 규칙

**누적 기울기 방식 (Batch gradient descent) / Accumulated gradient:**

논문에서 사용한 방식: 모든 입력-출력 케이스에 대해 $\partial E/\partial w$를 누적한 후, 한꺼번에 가중치를 변경합니다. 이것은 개별 케이스마다 갱신하는 것보다 안정적이지만 느립니다.

The approach used in the paper: accumulate $\partial E/\partial w$ over all input-output cases, then change weights all at once. More stable than per-case updates but slower.

**단순 경사 하강법 (Eq. 8) / Simple gradient descent:**

$$\Delta w = -\varepsilon \frac{\partial E}{\partial w}$$

$\varepsilon$ (learning rate)는 가중치 변화의 크기를 결정합니다. 저자들은 이 방법이 "이차 미분을 사용하는 방법만큼 빠르게 수렴하지 않지만, 훨씬 단순하고 병렬 하드웨어에서 쉽게 구현할 수 있다"고 언급합니다.

$\varepsilon$ determines the magnitude of weight changes. The authors note this "does not converge as rapidly as methods which make use of the second derivatives, but it is much simpler and can easily be implemented by local computations in parallel hardware."

**모멘텀 (Eq. 9) / Momentum:**

$$\Delta w(t) = -\varepsilon \frac{\partial E}{\partial w}(t) + \alpha \Delta w(t-1)$$

$\alpha$는 지수적 감쇠 인자(0~1 사이)로, 이전 갱신 방향의 관성을 유지합니다. 저자들은 이것을 "가속 방법(acceleration method)"이라고 부르며, "현재 기울기를 사용하여 가중치 공간에서의 위치가 아닌 **속도**를 수정하는 것"이라고 설명합니다. 모멘텀의 효과: (1) 평탄한 영역에서 가속 (연속적인 같은 방향의 기울기가 누적됨), (2) 진동 억제 (반대 방향 기울기가 상쇄됨).

$\alpha$ is an exponential decay factor (between 0 and 1) that maintains inertia of previous update direction. The authors call this an "acceleration method" and explain it as "using the current gradient to modify the **velocity** rather than the position in weight-space." Momentum effects: (1) acceleration in flat regions (successive same-direction gradients accumulate), (2) oscillation dampening (opposing gradients cancel).

**Weight decay**: 가중치를 매 갱신마다 0.2%씩 감소시켜 해석을 용이하게 합니다. 큰 가중치가 필요 없으면 0으로 향하도록 유도합니다. 현대의 L2 정규화(weight decay)의 초기 형태입니다.

Weight decay: decrease weights by 0.2% each update for easier interpretation. Pushes unnecessary weights toward 0. An early form of modern L2 regularization.

---

### Example 1: Symmetry Detection (p.534, Fig. 1) — 대칭 감지

**과제 / Task**: 이진 입력 벡터가 좌우 대칭인지 판별. 예: `1 0 1 1 0 1` → 대칭(1), `1 0 0 1 0 1` → 비대칭(0).

Determine whether a binary input vector is left-right symmetric. E.g., `1 0 1 1 0 1` → symmetric (1), `1 0 0 1 0 1` → not symmetric (0).

**구조**: 6개 입력 유닛 → 2개 hidden units → 1개 출력 유닛. 가능한 64개 입력 패턴 전체로 학습. $\varepsilon = 0.1$, $\alpha = 0.9$, 1,425 sweep (sweep = 64개 패턴 전체를 한 번 제시).

Architecture: 6 input → 2 hidden → 1 output. Trained on all 64 possible input patterns. $\varepsilon = 0.1$, $\alpha = 0.9$, 1,425 sweeps.

**학습된 가중치의 놀라운 구조 (Fig. 1) / Amazing structure of learned weights:**

두 hidden units의 가중치를 보면: 입력 벡터의 **중앙을 기준으로 대칭인 위치의 가중치가 크기는 같고 부호만 반대**입니다! 비율은 정확히 1:2:4 (입력 위치 순). 두 hidden unit의 가중치는 부호만 반대입니다.

Looking at the two hidden units' weights: weights at positions **symmetric about the center have equal magnitude but opposite sign**! The ratio is exactly 1:2:4 (by input position). The two hidden units have opposite-sign versions of the same pattern.

**왜 이것이 중요한가 / Why this matters:**

아무도 네트워크에게 "대칭 위치의 비트를 비교하라"고 가르치지 않았습니다. 네트워크가 **스스로** 이 전략을 발견했습니다. 이것이 "내부 표현의 자동 학습(learning representations)"의 첫 번째 구체적 데모입니다. Minsky & Papert(1969)는 대칭 감지가 단층 perceptron으로 불가능함을 보였고 — 이제 backprop이 이를 2개의 hidden units만으로 우아하게 해결합니다.

Nobody taught the network to "compare bits at symmetric positions." The network **discovered** this strategy on its own. This is the first concrete demo of "learning representations." Minsky & Papert (1969) showed symmetry detection is impossible for single-layer perceptrons — now backprop elegantly solves it with just 2 hidden units.

---

### Example 2: Family Trees (pp.534–535, Fig. 2–4) — 가족 관계 학습

**과제 / Task**: 두 개의 동형(isomorphic) 가족 나무(영어 가족 + 이탈리아 가족)에서 "X has-aunt Y", "X has-father Y" 같은 관계를 학습. 정보는 (사람1, 관계, 사람2)의 트리플로 표현됩니다. 네트워크는 처음 두 항이 주어지면 세 번째를 예측해야 합니다.

Learn relationships like "X has-aunt Y" in two isomorphic family trees (English + Italian). Information is expressed as (person1, relationship, person2) triples. The network must predict the third term given the first two.

**네트워크 구조 (Fig. 3)**: 5층 — 24개 입력 유닛(사람 인코딩) + 12개 입력 유닛(관계 인코딩) → 6개 hidden → 12개 중앙 hidden → 6개 hidden → 24개 출력 유닛(사람 예측).

Architecture (Fig. 3): 5 layers — 24 input (person encoding) + 12 input (relationship encoding) → 6 hidden → 12 central hidden → 6 hidden → 24 output (person prediction).

**핵심 결과: 분산 표현의 자발적 등장 (Fig. 4) / Key result: spontaneous emergence of distributed representation:**

Fig. 4는 hidden units의 "receptive fields" — 각 hidden unit이 24명의 사람에 대해 어떤 가중치를 학습했는지 — 를 보여줍니다. 놀라운 관찰:

Fig. 4 shows hidden units' "receptive fields" — what weights each hidden unit learned for the 24 people. Remarkable observations:

- **유닛 1**: 주로 영어/이탈리아 구분에 관심 → **국적(nationality) 감지기** 자동 생성
- **유닛 2**: 세대(generation) 구분에 관심
- **유닛 6**: 가족 내 분기(branch) 구분에 관심

네트워크가 **아무도 가르치지 않은 추상적 특징** — 국적, 세대, 성별 — 을 자동으로 발견했습니다. 더 놀라운 것은 **영어 사람과 이탈리아 대응 인물이 유사한 표현을 학습**했다는 것입니다: 네트워크가 두 가족 나무의 **동형성(isomorphism)** 을 발견하여, 한 가족에서 학습한 관계를 다른 가족에 **일반화(generalize)** 할 수 있었습니다. 학습 시 사용하지 않은 4개의 트리플 중 100개 중 잘 일반화했습니다.

The network automatically discovered **abstract features nobody taught** — nationality, generation, gender. Even more remarkably, **English people and their Italian counterparts learned similar representations**: the network discovered the **isomorphism** between the two family trees, enabling **generalization** of relationships learned from one family to the other. It generalized well on 4 triples not used in training out of 104.

---

### Layered Nets vs Recurrent Nets (p.535, Fig. 5) — 계층형 vs 순환형

**동치 관계 / Equivalence:**

저자들은 순환형(recurrent) 네트워크가 반복적으로 실행되면 **시간적으로 전개된 계층형 네트워크**와 동치임을 보여줍니다 (Fig. 5). 순환 네트워크의 각 시간 단계(time-step)가 계층형 네트워크의 한 층에 대응합니다. 따라서 backpropagation을 순환 네트워크에도 적용할 수 있습니다 — **backpropagation through time (BPTT)** 의 초기 개념.

The authors show that a recurrent network run iteratively is equivalent to a **temporally unfolded layered network** (Fig. 5). Each time-step of the recurrent network corresponds to one layer of the layered network. Therefore backpropagation can be applied to recurrent networks too — an early concept of **backpropagation through time (BPTT)**.

두 가지 기술적 복잡성:
1. 반복형 네트워크에서는 중간층의 forward pass 중 **출력 상태의 이력(history)** 을 저장해야 합니다
2. 다른 층 사이의 대응 가중치가 **동일**해야 하므로, $\partial E/\partial w$를 **모든** 대응 위치에서 평균하여 적용해야 합니다

Two technical complications:
1. In iterative nets, the **history of output states** during forward pass must be stored
2. Corresponding weights between different layers must be **identical**, so $\partial E/\partial w$ must be averaged over **all** corresponding positions

---

### Limitations and Discussion (pp.535–536) — 한계와 논의

**Local minima 문제 / The local minima problem:**

저자들이 솔직하게 인정하는 주요 한계: 오류 곡면(error surface)이 local minima를 가질 수 있으므로, gradient descent가 global minimum을 찾는다는 보장이 없습니다. 그러나 "실제 경험상, 네트워크가 global minimum보다 현저히 나쁜 local minima에 갇히는 경우는 매우 드물다"고 보고합니다. 이것은 수십 년 후에도 여전히 미스터리이며 — 왜 deep learning이 실제로 잘 작동하는지에 대한 이론적 이해는 여전히 활발한 연구 분야입니다.

The main limitation the authors honestly acknowledge: the error surface may have local minima, so gradient descent is not guaranteed to find the global minimum. However, they report that "experience with many tasks shows that the network very rarely gets stuck in poor local minima that are significantly worse than the global minimum." This remains a mystery decades later — theoretical understanding of why deep learning works well in practice is still an active research area.

**추가 차원의 해결책 / Extra dimensions as a fix:**

"연결을 몇 개 추가하면 가중치 공간에 추가 차원이 생기고, 이 차원들이 낮은 차원의 하위공간에서의 poor local minima 주위로 우회 경로를 제공합니다." 이 직관은 현대의 "과잉 매개변수화(overparameterization)"가 최적화를 쉽게 만든다는 이론과 일맥상통합니다.

"Adding a few more connections creates extra dimensions in weight-space and these dimensions provide paths around the barriers that create poor local minima in the lower dimensional subspaces." This intuition aligns with the modern theory that overparameterization makes optimization easier.

**생물학적 타당성 / Biological plausibility:**

"이 학습 절차는 현재 형태로는 뇌에서의 학습의 그럴듯한 모델이 아닙니다." 그러나 저자들은 gradient descent가 가중치 공간에서 흥미로운 내부 표현을 구축할 수 있다는 것이 "더 생물학적으로 타당한 gradient descent 방법을 찾는 것이 가치 있다"고 시사한다고 말합니다.

"The learning procedure, in its current form, is not a plausible model of learning in brains." However, the authors say the fact that gradient descent can construct interesting internal representations suggests "it is worth looking for more biologically plausible ways of doing gradient descent in neural networks."

---

## Key Takeaways / 핵심 시사점

1. **Chain rule이 다층 학습의 열쇠다**: 출력 오류를 chain rule로 역전파하면 어떤 깊이의 hidden layer도 학습시킬 수 있습니다. Eq. 7($\partial E/\partial y_i = \sum_j \partial E/\partial x_j \cdot w_{ji}$)이 이 한 줄의 핵심이며, 이것이 Minsky & Papert(1969)의 도전에 대한 해답입니다.

   **Chain rule is the key to multi-layer learning**: Backpropagating output error via chain rule enables training hidden layers of any depth. Eq. 7 is the single-line core, and this is the answer to Minsky & Papert's (1969) challenge.

2. **Hidden units는 표현을 자동으로 학습한다**: 가장 혁명적인 발견은 hidden units이 과제에 유용한 내부 표현을 **명시적 지시 없이** 발견한다는 것입니다. 대칭 감지기(Fig. 1), 국적/세대/성별 인코딩(Fig. 4) 등이 자동으로 나타납니다. 이것이 논문 제목 "Learning Representations"의 의미입니다.

   **Hidden units automatically learn representations**: The most revolutionary finding is that hidden units discover internal representations useful for the task **without explicit instruction**. Symmetry detectors (Fig. 1), nationality/generation/gender encodings (Fig. 4) emerge automatically. This is what the title "Learning Representations" means.

3. **Sigmoid의 미분 $y(1-y)$가 실용적 backprop을 가능케 한다**: Forward pass에서 이미 계산된 $y$ 값만으로 미분을 구할 수 있어, 추가 메모리/계산이 거의 불필요합니다. 그러나 이 성질이 나중에 vanishing gradient 문제의 원인이 됩니다.

   **Sigmoid derivative $y(1-y)$ makes practical backprop possible**: Derivatives computed from $y$ values already available from forward pass, requiring almost no extra memory/computation. But this property later causes the vanishing gradient problem.

4. **모멘텀은 단순하지만 효과적인 가속기법이다**: $\alpha \Delta w(t-1)$ 항의 추가만으로 평탄 영역 가속과 진동 억제가 가능합니다. 현대 최적화(Adam, SGD with momentum)의 직접적 선조입니다.

   **Momentum is a simple but effective acceleration technique**: Just adding the $\alpha \Delta w(t-1)$ term enables acceleration in flat regions and oscillation dampening. Direct ancestor of modern optimizers (Adam, SGD with momentum).

5. **Bias를 "항상 1을 출력하는 유닛"으로 통일한 트릭**: 별도의 bias 처리를 없애고, 모든 파라미터를 가중치로 통일합니다. 이것은 구현과 수학적 표기 모두를 단순화하는 우아한 아이디어이며, 현대 프레임워크에서도 내부적으로 이 방식을 사용합니다.

   **Trick of unifying bias as a "unit always outputting 1"**: Eliminates separate bias handling, unifying all parameters as weights. An elegant idea that simplifies both implementation and notation, still used internally in modern frameworks.

6. **Local minima는 이론적 한계이지만 실제로는 드물다**: 저자들은 솔직하게 인정하면서도, "과잉 매개변수화가 우회 경로를 제공한다"는 직관을 제시합니다. 이것은 40년이 지난 지금도 딥러닝 이론의 핵심 질문입니다.

   **Local minima are a theoretical limitation but rare in practice**: The authors candidly acknowledge this while offering the intuition that "overparameterization provides bypass paths." This remains a core question in deep learning theory 40 years later.

7. **순환 네트워크도 backprop으로 학습 가능하다**: 시간적으로 전개하면 계층형과 동치 → BPTT의 초기 개념. 이것이 LSTM(1997)과 Transformer(2017)로 이어지는 시퀀스 모델링의 기반을 놓았습니다.

   **Recurrent networks can also be trained with backprop**: Temporal unfolding makes them equivalent to layered nets → early concept of BPTT. This laid the foundation for sequence modeling leading to LSTM (1997) and Transformer (2017).

---

## Mathematical Summary / 수학적 요약

### Backpropagation — Complete Algorithm / 완전한 알고리즘

**입력 / Input:**
- 학습 데이터: $\{(\mathbf{x}^{(c)}, \mathbf{d}^{(c)})\}_{c=1}^C$ (입력-목표 쌍)
- 네트워크 구조: $L$개 층, 가중치 $\mathbf{W}^{(l)}$ ($l = 1, \ldots, L$)
- 학습률 $\varepsilon$, 모멘텀 $\alpha$

**1단계: Forward Pass (순전파)**
```
For each layer l = 1, 2, ..., L:
    x_j^(l) = Σ_i  y_i^(l-1) · w_ji^(l)     [총 입력 / total input]
    y_j^(l) = 1 / (1 + exp(-x_j^(l)))         [sigmoid 활성화 / activation]
```

**2단계: Error Computation (오류 계산)**
$$E = \frac{1}{2} \sum_j (y_j^{(L)} - d_j)^2$$

**3단계: Backward Pass (역전파)**
```
Output layer (l = L):
    δ_j^(L) = (y_j^(L) - d_j) · y_j^(L) · (1 - y_j^(L))

For each hidden layer l = L-1, L-2, ..., 1:
    δ_i^(l) = y_i^(l) · (1 - y_i^(l)) · Σ_j  δ_j^(l+1) · w_ji^(l+1)
```

**4단계: Weight Update (가중치 갱신)**
$$\Delta w_{ji}^{(l)}(t) = -\varepsilon \cdot \delta_j^{(l)} \cdot y_i^{(l-1)} + \alpha \cdot \Delta w_{ji}^{(l)}(t-1)$$

**5단계: 반복 / Repeat** until $E$ converges

---

## Paper in the Arc of History / 역사적 맥락의 타임라인

```
1943    McCulloch-Pitts ────── 이진 뉴런: 계산의 단위 / Binary neuron
          │
1949    Hebb ──────────────── 학습 규칙: "함께 발화 → 함께 연결"
          │                     "Fire together → wire together"
1958    Rosenblatt ────────── Perceptron: 단층 학습 가능 / Single-layer learning
          │
1969    Minsky & Papert ──── XOR 불가 → "다층을 어떻게 학습시키나?"
          │                     Can't do XOR → "How to train multi-layer?"
          │
1974    Werbos ───────────── Backprop 최초 제안 (PhD, 주목 못 받음)
          │                     First backprop proposal (ignored)
1982    Hopfield ─────────── 에너지 함수로 신경망 부활 / Energy revival
          │
1985    Le Cun ───────────── 독립적 backprop 발견 / Independent discovery
          │
  ╔════════════════════════════════════════════════════════════╗
  ║ ★ 1986  Rumelhart, Hinton & Williams ★                    ║
  ║  Nature 게재 → Backprop 대중화 → 딥러닝의 시작             ║
  ║  "Learning Representations by Back-propagating Errors"    ║
  ╚════════════════════════════════════════════════════════════╝
          │
1989    LeCun et al. ──────── Backprop + CNN → 우편번호 인식
          │                     Zip code recognition
1997    Hochreiter & ──────── LSTM: 기울기 소실 해결
        Schmidhuber             Vanishing gradient solved
          │
2006    Hinton et al. ──────── Deep Belief Nets → 딥러닝 부흥
          │
2012    Krizhevsky et al. ──── AlexNet: GPU + 딥 backprop → 빅뱅
          │
2014    Kingma & Ba ────────── Adam: 적응적 학습률 optimizer
          │
2017    Vaswani et al. ──────── Transformer: backprop + attention
          │
2024    Nobel Prize ──────────── Hopfield & Hinton 수상
                                  (Hinton = 이 논문의 공저자)
```

---

## Connections to Other Papers / 다른 논문과의 연결

| 논문 / Paper | 관계 / Relationship |
|---|---|
| **#1 McCulloch & Pitts (1943)** | 뉴런 모델의 기반. Backprop의 forward pass(Eq. 1-2)는 McCulloch-Pitts 뉴런의 직접적 일반화 (이진→연속, 계단→sigmoid) / Basis for neuron model. Forward pass is a direct generalization of M-P neurons |
| **#3 Rosenblatt (1958)** | 단층 perceptron의 학습 규칙 → backprop은 이를 다층으로 일반화. Perceptron 수렴 정리는 단층에서만 유효했고, backprop이 다층의 수렴 가능성을 열음 / Single-layer learning → backprop generalizes to multi-layer |
| **#4 Minsky & Papert (1969)** | **직접적 답변**: XOR, 대칭 감지 등 단층 불가능한 문제를 backprop이 hidden units로 해결. AI 겨울의 원인이 된 도전에 대한 결정적 해답 / **Direct answer**: backprop solves problems impossible for single layer |
| **#5 Hopfield (1982)** | 에너지 함수 $E$ 개념의 재등장. Hopfield: 동역학이 $E$를 최소화, Backprop: 학습이 $E$를 최소화. Hinton은 두 논문 모두에 관여(Boltzmann Machine) / Energy function $E$ concept reappears. Hopfield: dynamics minimize $E$, Backprop: learning minimizes $E$ |
| **#7 LeCun et al. (1989)** | Backprop의 첫 실용적 적용: CNN + backprop → 우편번호 인식. 이 논문의 알고리즘 + 합성곱 구조 = 컴퓨터 비전의 시작 / First practical application: CNN + backprop. This paper's algorithm + convolution = start of computer vision |
| **#9 Hochreiter & Schmidhuber (1997)** | Sigmoid의 $y(1-y) \leq 0.25$ 곱셈이 깊은 네트워크에서 기울기 소실을 유발 → LSTM이 게이팅으로 해결 / Sigmoid's $y(1-y) \leq 0.25$ multiplication causes vanishing gradients → LSTM solves with gating |
| **#18 Kingma & Ba (2014)** | Adam optimizer는 Eq. 9의 모멘텀 + 적응적 학습률. 직접적 후계자 / Adam = Eq. 9's momentum + adaptive learning rate. Direct descendant |

---

## References / 참고문헌

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J., "Learning representations by back-propagating errors," *Nature*, 323, pp. 533–536, 1986.
- Werbos, P. J., "Beyond regression: New tools for prediction and analysis in the behavioral sciences," PhD thesis, Harvard University, 1974.
- Le Cun, Y., "Une procédure d'apprentissage pour réseau a seuil asymétrique," *Proceedings of Cognitiva*, 85, pp. 599–604, 1985.
- Minsky, M. & Papert, S., *Perceptrons: An Introduction to Computational Geometry*, MIT Press, 1969.
- Rosenblatt, F., *Principles of Neurodynamics*, Spartan, 1961.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J., "Learning internal representations by error propagation," in *Parallel Distributed Processing*, Vol. 1, eds. Rumelhart & McClelland, MIT Press, pp. 318–362, 1986.
