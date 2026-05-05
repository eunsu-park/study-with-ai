# Pre-Reading Briefing: Learning Representations by Back-propagating Errors (1986)
# 사전 읽기 브리핑: 오류 역전파를 통한 표현 학습 (1986)

**Authors / 저자**: David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
**Journal / 저널**: *Nature*, Vol. 323, pp. 533–536, 9 October 1986
**Institutions / 소속**: UC San Diego; Carnegie-Mellon University

---

## 핵심 기여 / Core Contribution

Rumelhart, Hinton & Williams는 **다층 신경망(multi-layer neural network)** 의 가중치를 조정하기 위한 **backpropagation(오류 역전파)** 학습 알고리즘을 기술하고 대중화했습니다. 핵심 아이디어: 출력층의 오류를 **연쇄 법칙(chain rule)** 을 통해 네트워크를 거꾸로 전파하여, 모든 층의 가중치에 대한 오류의 기울기(gradient)를 계산합니다. 이로써 **숨겨진 유닛(hidden units)** 이 입력의 유용한 내부 표현(internal representation)을 자동으로 학습할 수 있게 됩니다. 이 논문은 Minsky & Papert(1969)가 지적한 단층 perceptron의 한계를 직접 극복하며, 현대 딥러닝의 **핵심 학습 알고리즘**을 확립했습니다.

Rumelhart, Hinton & Williams described and popularized the **backpropagation** learning algorithm for adjusting weights in **multi-layer neural networks**. The key idea: propagate output error backwards through the network via the **chain rule**, computing the gradient of error with respect to weights at every layer. This enables **hidden units** to automatically learn useful internal representations of the input. The paper directly overcomes the single-layer perceptron limitations identified by Minsky & Papert (1969) and established the **core learning algorithm** of modern deep learning.

---

## 역사적 맥락 / Historical Context

```
1943  McCulloch & Pitts ─── 이진 뉴런 모델 / Binary neuron model
  │
1958  Rosenblatt ────────── Perceptron: 단층 학습 가능 / Single-layer learning
  │
1969  Minsky & Papert ───── 단층의 한계 (XOR 불가) → AI 겨울
  │                         Single-layer limits (no XOR) → AI winter
  │
1974  Werbos ────────────── Backprop 최초 제안 (PhD 논문, 주목 못 받음)
  │                         First backprop proposal (PhD thesis, largely ignored)
  │
1982  Hopfield ──────────── 에너지 함수로 신경망 부활 / Energy function revival
  │
1985  Le Cun ────────────── 독립적으로 backprop 발견
  │                         Independently discovered backprop
  │
  ╔════════════════════════════════════════════════════════════╗
  ║ ★ 1986  Rumelhart, Hinton & Williams ★                    ║
  ║  Nature에 발표 → Backprop 대중화 → 딥러닝의 시작           ║
  ║  Published in Nature → Backprop popularized → DL begins   ║
  ╚════════════════════════════════════════════════════════════╝
  │
1989  LeCun et al. ──────── Backprop → CNN / 우편번호 인식
  │                         Backprop → CNN / zip code recognition
  │
2012  Krizhevsky et al. ──── AlexNet: 딥러닝 빅뱅
                              Deep learning big bang
```

**왜 이 논문이 특별한가 / Why this paper is special:**

1. **Minsky의 도전에 대한 직접적 답변**: Minsky & Papert(1969)는 단층 perceptron이 XOR을 풀 수 없음을 증명했습니다. 다층 네트워크가 이를 풀 수 있다는 것은 알려져 있었지만, **어떻게 학습시키는가**가 미해결 문제였습니다. 이 논문이 바로 그 답입니다.

   Minsky & Papert (1969) proved single-layer perceptrons can't solve XOR. It was known multi-layer networks could, but **how to train them** was unsolved. This paper is that answer.

2. **Nature 게재의 파급력**: Backpropagation 자체는 Werbos(1974)와 Le Cun(1985)이 먼저 발견했지만, Nature에 게재된 이 3.5페이지 논문이 기계학습 커뮤니티 전체에 알고리즘을 대중화시켰습니다.

   Backprop itself was discovered earlier by Werbos (1974) and Le Cun (1985), but this 3.5-page Nature paper popularized the algorithm to the entire ML community.

3. **"표현 학습"의 탄생**: 제목의 "Learning Representations"가 핵심입니다. Hidden units가 유용한 feature를 **자동으로** 발견한다는 개념은 현대 deep learning의 근본 원리입니다.

   "Learning Representations" in the title is key. The concept that hidden units **automatically** discover useful features is the fundamental principle of modern deep learning.

---

## 필요한 배경 지식 / Prerequisites

### 1. 이전 논문에서 알아야 할 것 / From previous papers

| 논문 / Paper | 필요한 개념 / Needed concept |
|---|---|
| #1 McCulloch & Pitts (1943) | 뉴런의 가중합 + 활성화 함수 구조 / Weighted sum + activation function structure |
| #3 Rosenblatt (1958) | 가중치 조정을 통한 학습, perceptron 수렴 정리 / Learning via weight adjustment, convergence theorem |
| #4 Minsky & Papert (1969) | XOR 문제: 단층으로 풀 수 없음 → 다층의 필요성 / XOR problem: unsolvable by single layer → need for multi-layer |
| #5 Hopfield (1982) | 에너지 함수 $E$의 개념 — backprop에서 오류 함수 $E$로 재등장 / Energy function concept — reappears as error function $E$ |

### 2. 미적분학 개념 / Calculus concepts

- **편미분 (Partial derivative)**: $\partial E / \partial w$ — 다른 변수를 고정한 채 $w$의 미소 변화가 $E$에 미치는 영향. Backprop의 핵심 연산입니다.

  $\partial E / \partial w$ — effect of a tiny change in $w$ on $E$ while holding other variables fixed. The core operation of backprop.

- **연쇄 법칙 (Chain rule)**: $\frac{\partial E}{\partial w} = \frac{\partial E}{\partial y} \cdot \frac{\partial y}{\partial x} \cdot \frac{\partial x}{\partial w}$ — 합성 함수의 미분. Backprop은 이것의 **체계적 적용**입니다.

  Derivative of composed functions. Backprop is the **systematic application** of this rule.

- **경사 하강법 (Gradient descent)**: $w \leftarrow w - \varepsilon \frac{\partial E}{\partial w}$ — 오류를 줄이는 방향으로 가중치를 조금씩 이동. $\varepsilon$는 학습률(learning rate).

  Move weights in the direction that reduces error. $\varepsilon$ is the learning rate.

### 3. 선형대수학 / Linear algebra

- **행렬-벡터 곱 (Matrix-vector product)**: 층 간 계산 $x_j = \sum_i y_i w_{ji}$는 행렬 곱 $\mathbf{x} = \mathbf{W}\mathbf{y}$입니다.

  Layer computation $x_j = \sum_i y_i w_{ji}$ is the matrix product $\mathbf{x} = \mathbf{W}\mathbf{y}$.

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 직관적 설명 / Intuitive explanation |
|---|---|
| **Forward pass** | 입력 → 숨겨진 층 → 출력으로 신호가 앞으로 흐름. 각 층에서 가중합 + 활성화 적용 / Signal flows forward from input → hidden → output. Apply weighted sum + activation at each layer |
| **Backward pass** | 출력 오류 → 숨겨진 층 → 입력 방향으로 기울기가 뒤로 흐름. Chain rule의 역방향 적용 / Gradients flow backward from output error → hidden → input direction. Reverse application of chain rule |
| **Hidden units / 숨겨진 유닛** | 입력도 출력도 아닌 **중간층** 뉴런. 네트워크가 스스로 학습하는 내부 표현을 담당 / Neurons in **intermediate layers**, neither input nor output. Responsible for internal representations the network learns on its own |
| **Internal representation** | Hidden units가 발견한 입력 데이터의 유용한 특징 패턴. 명시적으로 프로그래밍하지 않았는데 자동으로 나타남 / Useful feature patterns of input data discovered by hidden units. Appear automatically without explicit programming |
| **Error function $E$** | 실제 출력과 목표 출력의 차이를 측정하는 스칼라. 학습 = $E$를 최소화 / Scalar measuring difference between actual and desired output. Learning = minimizing $E$ |
| **Gradient descent** | $E$가 가장 빠르게 감소하는 방향으로 가중치를 조정. "언덕을 내려가는" 최적화 / Adjust weights in the direction $E$ decreases fastest. "Going downhill" optimization |
| **Learning rate $\varepsilon$** | 한 번의 갱신에서 가중치를 얼마나 크게 바꿀지. 너무 크면 진동, 너무 작으면 느림 / How much to change weights per update. Too large → oscillation, too small → slow |
| **Momentum $\alpha$** | 이전 갱신 방향을 일부 유지하여 학습을 가속. 관성의 효과 / Retain some of previous update direction to accelerate learning. Inertia effect |
| **Sigmoid $\sigma(x)$** | $1/(1+e^{-x})$ — S자 형태의 활성화 함수. 미분 가능하여 chain rule 적용 가능 / S-shaped activation function. Differentiable, enabling chain rule application |
| **Local minimum** | $E$ 지형의 골짜기이지만 가장 깊은 곳(global minimum)이 아닌 곳. Gradient descent가 갇힐 수 있음 / A valley in the $E$ landscape, but not the deepest (global minimum). Gradient descent can get trapped |

---

## 수식 미리보기 / Equations Preview

### 수식 1: Forward Pass — 총 입력 계산 / Total Input Computation

$$x_j = \sum_i y_i w_{ji} \tag{Eq. 1}$$

**직관 / Intuition**: 유닛 $j$의 총 입력은 이전 층 유닛들의 출력 $y_i$를 가중치 $w_{ji}$로 곱하여 합한 것입니다. Bias는 항상 1을 출력하는 추가 유닛으로 처리합니다 — 별도의 bias 항 없이 가중치 하나로 통일!

The total input to unit $j$ is the weighted sum of outputs $y_i$ from the previous layer. Bias is handled as an extra unit that always outputs 1 — unified as just another weight, no separate bias term!

### 수식 2: 활성화 함수 — Sigmoid / Activation Function

$$y_j = \frac{1}{1 + e^{-x_j}} \tag{Eq. 2}$$

**직관 / Intuition**: 총 입력 $x_j$를 0~1 범위로 "압축"합니다. $x_j$가 매우 크면 $\approx 1$, 매우 작으면 $\approx 0$, 0 근처에서 급격히 변합니다. **핵심 속성**: 미분이 매우 깔끔합니다: $dy_j/dx_j = y_j(1 - y_j)$. 이것이 backprop을 실용적으로 만듭니다.

"Squashes" total input $x_j$ to range 0–1. **Key property**: derivative is very clean: $dy_j/dx_j = y_j(1 - y_j)$. This makes backprop practical.

### 수식 3: 오류 함수 / Error Function

$$E = \frac{1}{2} \sum_c \sum_j (y_{j,c} - d_{j,c})^2 \tag{Eq. 3}$$

**직관 / Intuition**: 모든 학습 케이스 $c$에 대해, 실제 출력 $y_{j,c}$와 목표 출력 $d_{j,c}$의 차이를 제곱하여 합산. $1/2$는 미분 시 깔끔함을 위한 상수. Hopfield의 에너지 함수 $E$와 같은 역할 — 최소화해야 할 대상.

Sum of squared differences between actual output $y_{j,c}$ and desired output $d_{j,c}$ over all cases $c$. The $1/2$ is for clean differentiation. Same role as Hopfield's energy function $E$ — the quantity to minimize.

### 수식 4–5: Backward Pass 핵심 — Chain Rule 적용 / Chain Rule Application

**출력층 유닛 / For output units:**
$$\frac{\partial E}{\partial y_j} = y_j - d_j \tag{Eq. 4}$$

$$\frac{\partial E}{\partial x_j} = \frac{\partial E}{\partial y_j} \cdot y_j(1 - y_j) \tag{Eq. 5}$$

**직관 / Intuition**: Eq. 4는 단순 — 출력의 오류는 (실제값 - 목표값). Eq. 5는 chain rule의 첫 적용 — sigmoid의 미분 $y_j(1 - y_j)$를 곱합니다. $y_j$가 0이나 1에 가까우면(확신이 높으면) 미분이 거의 0 → 가중치 변화가 작습니다. $y_j \approx 0.5$이면(불확실하면) 미분이 최대 → 가장 많이 학습합니다.

Eq. 4 is simple — output error is (actual - desired). Eq. 5 applies chain rule — multiply by sigmoid derivative $y_j(1-y_j)$. When $y_j$ is near 0 or 1 (high confidence), derivative is near 0 → small weight change. When $y_j \approx 0.5$ (uncertain), derivative is maximal → most learning.

### 수식 6–7: 가중치 기울기 & 숨겨진 층 전파 / Weight Gradient & Hidden Layer Propagation

$$\frac{\partial E}{\partial w_{ji}} = \frac{\partial E}{\partial x_j} \cdot y_i \tag{Eq. 6}$$

$$\frac{\partial E}{\partial y_i} = \sum_j \frac{\partial E}{\partial x_j} \cdot w_{ji} \tag{Eq. 7}$$

**직관 / Intuition**: **Eq. 6**이 "학습 신호"입니다 — 가중치 $w_{ji}$가 오류에 얼마나 기여하는가 = (유닛 $j$에 대한 오류 기울기) × (유닛 $i$의 출력). **Eq. 7**이 **역전파의 핵심** — 유닛 $i$에 대한 오류를 계산하기 위해, $i$가 연결된 **모든 다음 층 유닛** $j$의 오류 기울기를 가중합합니다. Eq. 5 → Eq. 7 → Eq. 5 → Eq. 7... 을 반복하면 어떤 깊이의 층이든 기울기를 계산할 수 있습니다.

**Eq. 6** is the "learning signal" — how much weight $w_{ji}$ contributes to error = (error gradient for unit $j$) × (output of unit $i$). **Eq. 7** is the **core of backpropagation** — to compute error for unit $i$, sum the error gradients from **all next-layer units** $j$ that $i$ connects to, weighted by $w_{ji}$. Repeating Eq. 5 → Eq. 7 → Eq. 5 → Eq. 7... computes gradients for layers of any depth.

### 수식 8–9: 가중치 갱신 규칙 / Weight Update Rules

**단순 경사 하강법 / Simple gradient descent:**
$$\Delta w = -\varepsilon \frac{\partial E}{\partial w} \tag{Eq. 8}$$

**모멘텀 추가 / With momentum:**
$$\Delta w(t) = -\varepsilon \frac{\partial E}{\partial w}(t) + \alpha \Delta w(t-1) \tag{Eq. 9}$$

**직관 / Intuition**: Eq. 8은 오류가 줄어드는 방향으로 가중치를 $\varepsilon$만큼 이동. Eq. 9는 관성(momentum)을 추가 — 이전 갱신 방향을 $\alpha$ 비율로 유지하여, 평탄한 영역에서 가속하고 진동을 억제합니다. 논문에서 $\varepsilon = 0.1$, $\alpha = 0.9$를 사용했습니다.

Eq. 8 moves weights by $\varepsilon$ in the error-decreasing direction. Eq. 9 adds inertia — retaining $\alpha$ fraction of previous update direction, accelerating through flat regions and dampening oscillations. The paper used $\varepsilon = 0.1$, $\alpha = 0.9$.

---

## 읽기 가이드 / Reading Guide

논문은 Nature 기고 형식으로 3.5페이지(매우 밀도 높음)입니다. 다음 순서로 읽기를 권합니다:

The paper is 3.5 pages in Nature letter format (very dense). Recommended reading order:

1. **첫 2단락** — 문제 정의: 왜 hidden units이 필요한가 / Problem definition: why hidden units are needed
2. **Eq. 1–2** — Forward pass 이해 / Understand forward pass
3. **Eq. 3** — 오류 함수 정의 / Error function definition
4. **Eq. 4–7** — Backward pass의 chain rule 유도 (가장 중요!) / Chain rule derivation for backward pass (most important!)
5. **Eq. 8–9** — 가중치 갱신 규칙 / Weight update rules
6. **Fig. 1** — 대칭 감지 문제: hidden units이 학습한 표현 분석 / Symmetry detection: analyzing what hidden units learned
7. **Fig. 2–4** — 가족 관계 문제: 분산 표현(distributed representation)의 등장 / Family tree task: emergence of distributed representations
8. **마지막 3단락** — 한계와 전망 (local minima, 순환 네트워크와의 관계) / Limitations and outlook

**특히 주의할 점 / Pay special attention to:**
- Eq. 7이 **역전파의 핵심** — 왜 이 한 줄의 수식이 다층 학습을 가능하게 하는지
- Fig. 1의 가중치 패턴 — hidden units이 **스스로** 대칭 감지기가 됨
- Fig. 4의 receptive fields — 네트워크가 **사람의 도움 없이** 의미 있는 특징을 발견

- Eq. 7 is **the core of backprop** — why this single equation enables multi-layer learning
- Weight patterns in Fig. 1 — hidden units **spontaneously** become symmetry detectors
- Receptive fields in Fig. 4 — the network discovers meaningful features **without human help**
