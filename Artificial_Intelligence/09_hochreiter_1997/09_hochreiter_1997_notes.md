---
title: "Long Short-Term Memory"
authors: Sepp Hochreiter, Jürgen Schmidhuber
year: 1997
journal: "Neural Computation, Vol. 9(8), pp. 1735–1780"
doi: "10.1162/neco.1997.9.8.1735"
topic: Artificial Intelligence / Recurrent Neural Networks
tags: [LSTM, vanishing gradient, gating mechanism, constant error carrousel, sequence modeling, RNN, memory cell, long-range dependency]
status: completed
date_started: 2026-04-09
date_completed: 2026-04-09
---

# Long Short-Term Memory
**Sepp Hochreiter & Jürgen Schmidhuber (1997)**

---

## 핵심 기여 / Core Contribution

이 논문은 순환 신경망(RNN)의 근본적 한계인 **기울기 소실(vanishing gradient)** 문제를 해결하는 **LSTM (Long Short-Term Memory)** 아키텍처를 제안합니다. 기존 RNN에서는 오류 신호가 시간을 거슬러 역전파될 때 각 타임스텝마다 $f'(net) \cdot w$가 곱해지며, 이 값이 1보다 작으면 기하급수적으로 소실되고 1보다 크면 폭발합니다. LSTM은 **Constant Error Carrousel (CEC)** — 가중치 1.0의 자기 순환 연결을 가진 선형 유닛을 도입하여, 오류가 시간에 걸쳐 일정하게(소실도 폭발도 없이) 흐르게 합니다. 그러나 CEC만으로는 불충분합니다: 외부의 무관한 입력이 메모리를 교란하거나(input weight conflict), 메모리 출력이 부적절한 시점에 다른 유닛을 교란(output weight conflict)할 수 있기 때문입니다. 이를 해결하기 위해 **곱셈적 게이트 유닛** — 입력 게이트(input gate)와 출력 게이트(output gate) — 을 도입하여, 메모리 셀에 정보를 쓰고 읽는 시점을 네트워크가 학습합니다. 6가지 실험에서 LSTM은 1000 타임스텝 이상의 장기 의존성을 학습하고, BPTT와 RTRL이 완전히 실패하는 복잡한 인공 작업들을 해결합니다. 이 아키텍처는 이후 거의 20년간 텍스트, 음성, 시계열 등 시퀀스 모델링의 지배적 구조가 되었습니다.

This paper proposes the **LSTM (Long Short-Term Memory)** architecture, solving the fundamental **vanishing gradient** problem in recurrent neural networks (RNNs). In conventional RNNs, error signals are multiplied by $f'(net) \cdot w$ at each time step during backpropagation through time — exponentially vanishing when this value is below 1, or exploding when above 1. LSTM introduces the **Constant Error Carrousel (CEC)** — a linear unit with a self-recurrent connection of weight 1.0, allowing error to flow constantly (neither vanishing nor exploding) over time. However, CEC alone is insufficient: irrelevant inputs can disturb memory (input weight conflict), and memory outputs at inappropriate times can disturb other units (output weight conflict). To solve this, **multiplicative gate units** — input gate and output gate — are introduced, enabling the network to learn *when* to write to and read from memory cells. Across six experiments, LSTM learns long-range dependencies exceeding 1000 time steps and solves complex artificial tasks where BPTT and RTRL completely fail. This architecture became the dominant structure for sequence modeling (text, speech, time series) for nearly two decades.

---

## 읽기 노트 / Reading Notes

### Section 1: Introduction — RNN의 문제와 LSTM의 필요성 / The RNN Problem and Need for LSTM

#### 순환 신경망의 약속과 한계 / The Promise and Limits of RNNs

순환 신경망은 피드백 연결을 통해 최근 입력의 표상을 활성화(activation) 형태로 저장할 수 있습니다 — 이것이 "단기 기억(short-term memory)"입니다. 가중치를 천천히 변경하여 저장하는 "장기 기억(long-term memory)"과 대비됩니다. 음성 처리, 비마르코프 제어, 음악 작곡 등 다양한 응용에 잠재력이 있지만, 실제로는 입력과 대응하는 교사 신호 사이의 **최소 시간 지연(minimal time lag)**이 길 때 학습이 극도로 어렵거나 불가능합니다.

Recurrent networks can store representations of recent inputs as activations via feedback connections — this is "short-term memory," contrasted with "long-term memory" stored by slowly changing weights. While promising for speech processing, non-Markovian control, and music composition, in practice learning becomes extremely difficult or impossible when the **minimal time lag** between inputs and corresponding teacher signals is long.

#### 문제의 본질: 오류 신호의 기하급수적 감쇠/폭발 / The Essence: Exponential Decay/Explosion of Error

BPTT ("Back-Propagation Through Time")와 RTRL ("Real-Time Recurrent Learning")에서, 오류 신호가 "시간을 거슬러" 흐를 때 두 가지 중 하나가 발생합니다:

In BPTT and RTRL, when error signals flow "back through time," one of two things happens:

1. **폭발 (Blow up)**: 가중치가 발산하고 학습이 불안정해짐 / Weights diverge, learning becomes unstable
2. **소실 (Vanish)**: 장기 의존성 학습이 극도로 느리거나 불가능 / Long-range dependency learning becomes extremely slow or impossible

논문은 이 문제에 대한 직접적 해결책 — LSTM을 제시합니다. 핵심은 특수 유닛의 내부 상태를 통해 **일정한(constant)** 오류 흐름을 강제하는 것입니다. 계산 복잡도는 타임스텝과 가중치 당 $O(1)$로 BPTT와 동일합니다.

The paper presents a direct solution — LSTM. The key is enforcing **constant** error flow through internal states of special units. Computational complexity is $O(1)$ per time step and weight, same as BPTT.

---

### Section 3: Constant Error Backprop — 기울기 소실의 수학적 분석 / Mathematical Analysis of Vanishing Gradients

#### 3.1 기하급수적 감쇠의 수학 / Mathematics of Exponential Decay

출력 유닛 $k$의 시간 $t$에서의 오류 신호는 $\vartheta_k(t) = f'_k(net_k(t))(d_k(t) - y^k(t))$입니다. 비출력 유닛 $j$의 역전파된 오류 신호는:

Output unit $k$'s error signal at time $t$ is $\vartheta_k(t) = f'_k(net_k(t))(d_k(t) - y^k(t))$. Non-output unit $j$'s backpropagated error signal is:

$$\vartheta_j(t) = f'_j(net_j(t)) \sum_i w_{ij} \vartheta_i(t+1)$$

유닛 $u$에서 유닛 $v$로의 **로컬 오류 흐름(local error flow)**은 $q$ 타임스텝에 걸쳐:

The **local error flow** from unit $u$ to unit $v$ over $q$ time steps:

$$\frac{\partial \vartheta_v(t-q)}{\partial \vartheta_u(t)} = \begin{cases} f'_v(net_v(t-1)) w_{uv} & q = 1 \\ f'_v(net_v(t-q)) \sum_{l=1}^{n} \frac{\partial \vartheta_l(t-q+1)}{\partial \vartheta_v(t)} w_{lv} & q > 1 \end{cases}$$

귀납법으로 전개하면 모든 가능한 경로를 통한 오류 흐름의 합이 됩니다:

Expanding by induction gives the sum of error flow through all possible paths:

$$\frac{\partial \vartheta_v(t-q)}{\partial \vartheta_u(t)} = \sum_{l_1=1}^{n} \cdots \sum_{l_{q-1}=1}^{n} \prod_{m=1}^{q} f'_{l_m}(net_{l_m}(t-m)) w_{l_m l_{m-1}}$$

**핵심 직관**: $n^{q-1}$개의 항 $\prod_{m=1}^{q} f'_{l_m} \cdot w_{l_m l_{m-1}}$이 오류 흐름을 결정합니다.

**Key intuition**: $n^{q-1}$ terms $\prod_{m=1}^{q} f'_{l_m} \cdot w_{l_m l_{m-1}}$ determine the error flow.

**경우 1**: 모든 $m$에 대해 $|f'_{l_m} \cdot w_{l_m l_{m-1}}| > 1.0$이면, 곱이 $q$에 따라 기하급수적으로 **증가** → 오류 폭발, 가중치 발산.

**Case 1**: If $|f'_{l_m} \cdot w_{l_m l_{m-1}}| > 1.0$ for all $m$, the product increases exponentially with $q$ → error explosion, weight divergence.

**경우 2**: 모든 $m$에 대해 $|f'_{l_m} \cdot w_{l_m l_{m-1}}| < 1.0$이면, 곱이 $q$에 따라 기하급수적으로 **감소** → 오류 소실, 학습 불가.

**Case 2**: If $|f'_{l_m} \cdot w_{l_m l_{m-1}}| < 1.0$ for all $m$, the product decreases exponentially with $q$ → error vanishing, no learning.

로지스틱 시그모이드의 경우 $f'_{\max} = 0.25$입니다. 따라서 가중치의 절대값이 4.0 미만이면 (즉 $|w| < 4.0$이면 $0.25 \times 4.0 = 1.0$), 기울기는 **거의 항상 소실**합니다. 가중치를 크게 초기화해도 해결되지 않습니다: $|w| \to \infty$일 때 관련 미분은 가중치보다 "더 빠르게" 0에 수렴하기 때문입니다.

For logistic sigmoid, $f'_{\max} = 0.25$. So if weight magnitudes are below 4.0 (i.e., $0.25 \times 4.0 = 1.0$), gradients **almost always vanish**. Larger initial weights don't help: as $|w| \to \infty$, the relevant derivatives go to zero "faster" than the weights grow.

#### 약한 상계의 공식화 / Weak Upper Bound Formalization

행렬 형태로 Eq. 2를 다시 쓰면:

Rewriting Eq. 2 in matrix form:

$$(W_{uT})^T F'(t-1) \prod_{m=2}^{q-1} (W F'(t-m)) \; W_v \; f'_v(net_v(t-q))$$

여기서 $W$는 가중치 행렬, $F'(t-m)$은 미분의 대각 행렬입니다. $\|W\|_A \cdot f'_{\max} < 1$이면 ($\tau := \frac{n w_{\max}}{4.0} < 1$), 다음이 성립합니다:

Where $W$ is the weight matrix and $F'(t-m)$ is the diagonal matrix of derivatives. If $\|W\|_A \cdot f'_{\max} < 1$ ($\tau := \frac{n w_{\max}}{4.0} < 1$), then:

$$\left|\frac{\partial \vartheta_v(t-q)}{\partial \vartheta_u(t)}\right| \leq n \cdot (\tau)^q$$

**$\tau < 1$이면 오류 흐름은 $q$에 대해 기하급수적으로 감소합니다.** 이것이 기울기 소실의 엄밀한 수학적 증명입니다.

**If $\tau < 1$, error flow decreases exponentially with $q$.** This is the rigorous mathematical proof of vanishing gradients.

#### 3.2 순진한 접근: Constant Error Flow / Naive Approach

기울기 소실을 피하려면, 유닛 $j$의 자기 순환에서 오류가 일정해야 합니다:

To avoid vanishing gradients, error must be constant through unit $j$'s self-recurrence:

$$f'_j(net_j(t)) \cdot w_{jj} = 1.0$$

이를 해결하면: $f_j(x) = x$ (선형/항등 함수), $w_{jj} = 1.0$. 이것이 **Constant Error Carrousel (CEC)**입니다. 유닛 $j$의 활성화는 시간에 걸쳐 변하지 않습니다:

Solving: $f_j(x) = x$ (linear/identity function), $w_{jj} = 1.0$. This is the **Constant Error Carrousel (CEC)**. Unit $j$'s activation doesn't change over time:

$$y_j(t+1) = f_j(net_j(t+1)) = f_j(w_{jj} y^j(t)) = y^j(t)$$

그러나 CEC만으로는 두 가지 근본적 문제가 발생합니다:

However, CEC alone has two fundamental problems:

**1. Input Weight Conflict (입력 가중치 충돌)**: 같은 입력 가중치 $w_{ji}$가 (a) 새 정보를 저장하는 것과 (b) 무관한 입력으로부터 기존 정보를 보호하는 것, 두 가지 상충하는 역할을 동시에 해야 합니다. 학습 초기에는 단기 오류를 줄이기 위해 입력을 받아들이지만, 나중에 장기 오류를 줄이려 할 때 이미 학습된 단기 행동이 간섭합니다.

**1. Input Weight Conflict**: The same input weight $w_{ji}$ must simultaneously (a) store new information and (b) protect existing information from irrelevant inputs — two conflicting roles. Early training reduces short-term errors by accepting inputs, but later, already-learned short-term behavior interferes with reducing long-term errors.

**2. Output Weight Conflict (출력 가중치 충돌)**: 같은 출력 가중치 $w_{kj}$가 (a) 저장된 정보를 다른 유닛에 전달하는 것과 (b) 다른 시점에서 메모리 출력이 다른 유닛을 교란하지 않도록 보호하는 것, 두 가지 상충하는 역할을 합니다.

**2. Output Weight Conflict**: The same output weight $w_{kj}$ must simultaneously (a) transmit stored information to other units and (b) protect other units from memory output at inappropriate times — two conflicting roles.

이 충돌들은 장기 시간 지연일수록 심각해집니다. **문맥에 따라 "쓰기"와 "읽기" 연산을 제어하는 메커니즘이 필요합니다** — 이것이 게이트의 동기입니다.

These conflicts worsen with longer time lags. **A context-sensitive mechanism for controlling "write" and "read" operations is needed** — this motivates the gates.

---

### Section 4: Long Short-Term Memory — LSTM 아키텍처 / The LSTM Architecture

#### Memory Cell과 Gate Units / Memory Cells and Gate Units

CEC의 장점(일정한 오류 흐름)을 유지하면서 입출력 충돌을 해결하기 위해, LSTM은 CEC 주위에 추가적 구조를 구축합니다:

To maintain CEC's advantage (constant error flow) while resolving input/output conflicts, LSTM builds additional structure around the CEC:

- **입력 게이트 (Input Gate, $in_j$)**: 곱셈적 유닛으로, 메모리 셀 $c_j$의 입력 연결로의 오류 흐름을 제어합니다. 무관한 입력으로부터 메모리 내용을 보호합니다.
- **출력 게이트 (Output Gate, $out_j$)**: 곱셈적 유닛으로, 메모리 셀 $c_j$의 출력 연결로의 오류 흐름을 제어합니다. 현재 무관한 메모리 내용이 다른 유닛을 교란하는 것을 방지합니다.

- **Input Gate ($in_j$)**: Multiplicative unit controlling error flow to memory cell $c_j$'s input connections. Protects memory contents from irrelevant inputs.
- **Output Gate ($out_j$)**: Multiplicative unit controlling error flow from memory cell $c_j$'s output connections. Prevents currently irrelevant memory contents from disturbing other units.

#### 수식으로 본 LSTM / LSTM in Equations

**게이트 활성화 / Gate activations**:

$$y^{out_j}(t) = f_{out_j}(net_{out_j}(t)), \quad y^{in_j}(t) = f_{in_j}(net_{in_j}(t))$$

여기서 $f_{out_j}$, $f_{in_j}$는 시그모이드 함수 (범위 $[0, 1]$), net 입력은:

Where $f_{out_j}$, $f_{in_j}$ are sigmoid functions (range $[0, 1]$), with net inputs:

$$net_{out_j}(t) = \sum_u w_{out_j u} y^u(t-1), \quad net_{in_j}(t) = \sum_u w_{in_j u} y^u(t-1)$$

**메모리 셀 내부 상태 업데이트 / Memory cell internal state update**:

$$s_{c_j}(t) = s_{c_j}(t-1) + y^{in_j}(t) \cdot g(net_{c_j}(t)), \quad s_{c_j}(0) = 0$$

이것이 LSTM의 **가장 핵심적인 수식**입니다:
- $s_{c_j}(t-1)$: 이전 상태가 **가중치 1.0으로 그대로 전달** — CEC! 기울기가 이 경로를 통해 소실 없이 흐름
- $y^{in_j}(t) \cdot g(net_{c_j}(t))$: 입력 게이트가 열렸을 때만 ($y^{in_j} \approx 1$) 새 정보가 추가됨. 닫혔을 때 ($y^{in_j} \approx 0$) 기존 상태 보존
- $g$: 입력 압축 함수 (범위 $[-2, 2]$)

This is LSTM's **most crucial equation**:
- $s_{c_j}(t-1)$: Previous state **passes through with weight 1.0** — the CEC! Gradients flow through this path without vanishing
- $y^{in_j}(t) \cdot g(net_{c_j}(t))$: New information added only when input gate is open ($y^{in_j} \approx 1$). When closed ($y^{in_j} \approx 0$), existing state is preserved
- $g$: Input squashing function (range $[-2, 2]$)

**메모리 셀 출력 / Memory cell output**:

$$y^{c_j}(t) = y^{out_j}(t) \cdot h(s_{c_j}(t))$$

- $h$: 출력 스케일링 함수 (범위 $[-1, 1]$)
- 출력 게이트가 열렸을 때만 메모리 내용이 외부에 노출됨

- $h$: Output scaling function (range $[-1, 1]$)
- Memory contents exposed externally only when output gate is open

#### 왜 게이트가 작동하는가 / Why Gates Work

게이트의 핵심은 **곱셈적(multiplicative)** 상호작용입니다:

The key to gates is their **multiplicative** interaction:

- 게이트 값 ≈ 0: $0 \times \text{anything} = 0$ → 정보 완전 차단 / Information completely blocked
- 게이트 값 ≈ 1: $1 \times \text{signal} = \text{signal}$ → 정보 완전 통과 / Information passes through completely

이것은 단순한 가산(additive) 연결과 근본적으로 다릅니다. 가산 연결에서는 신호를 완전히 차단할 방법이 없지만, 곱셈적 게이트는 0을 곱하여 완벽한 차단이 가능합니다. 게이트 자체가 시그모이드 활성화를 가지므로, 네트워크는 학습을 통해 게이트의 열림/닫힘을 조절합니다.

This is fundamentally different from additive connections. With additive connections, there's no way to completely block a signal, but multiplicative gates can achieve perfect blocking by multiplying by 0. Since gates have sigmoid activations, the network learns to control opening/closing through training.

#### LSTM 각 구성 요소 상세 설명 / Detailed Explanation of Each LSTM Component

LSTM의 memory cell을 **금고(vault)**에 비유하면 이해하기 쉽습니다.

LSTM's memory cell is easier to understand when compared to a **vault**.

##### Cell State ($s_{c_j}$) — 금고 안의 내용물 / Contents Inside the Vault

$$s_{c_j}(t) = s_{c_j}(t-1) + y^{in_j}(t) \cdot g(net_{c_j}(t))$$

Cell state는 LSTM의 **장기 기억**입니다. 핵심은 **이전 상태가 가중치 1.0으로 그대로 전달**된다는 것입니다 — 이것이 CEC(Constant Error Carrousel)입니다. 일반 RNN에서는 $h_t = \tanh(W \cdot h_{t-1} + ...)$처럼 매 스텝 비선형 변환을 거치므로, 정보가 반복적으로 "압축"되어 점점 손실됩니다. 반면 cell state는 **컨베이어 벨트**처럼 직선으로 흐르며, 정보가 추가(+)만 될 뿐 변형되지 않습니다. 역전파 시 $\frac{\partial s(t)}{\partial s(t-1)} = 1$이므로, 기울기가 이 경로를 통해 100스텝이든 1000스텝이든 **소실 없이** 흐릅니다.

Cell state is LSTM's **long-term memory**. The key is that **the previous state passes through with weight 1.0** — this is the CEC (Constant Error Carrousel). In standard RNNs, $h_t = \tanh(W \cdot h_{t-1} + ...)$ applies a non-linear transformation every step, causing information to be repeatedly "compressed" and gradually lost. In contrast, cell state flows like a **conveyor belt** — information is only added (+), never transformed. During backpropagation, $\frac{\partial s(t)}{\partial s(t-1)} = 1$, so gradients flow through this path **without vanishing**, whether for 100 or 1000 steps.

##### Input Gate ($in_j$) — 금고의 입구 문지기 / Gatekeeper at the Vault Entrance

$$y^{in_j}(t) = \sigma\left(\sum_u w_{in_j u} \cdot y^u(t-1)\right)$$

입력 게이트는 **"지금 이 정보를 금고에 넣을 것인가?"**를 결정합니다.

The input gate decides: **"Should we store this information in the vault now?"**

- $y^{in_j} \approx 1$ (열림/open): 새 정보가 cell state에 추가됨 / New information is added to cell state
- $y^{in_j} \approx 0$ (닫힘/closed): 새 정보가 차단되고, 기존 cell state가 보존됨 / New information is blocked, existing cell state is preserved

**왜 필요한가?** 논문 Section 3.2의 "input weight conflict" 때문입니다. CEC만 있고 게이트가 없으면, 같은 입력 가중치 $w_{ji}$가 "새 정보를 저장하라"와 "기존 정보를 보호하라"라는 **상충하는 두 역할**을 동시에 해야 합니다. 예를 들어:

**Why is it needed?** Because of the "input weight conflict" in Section 3.2. Without gates, the same input weight $w_{ji}$ must simultaneously serve two **conflicting roles**: "store new information" and "protect existing information." For example:

- 시간 $t=1$에서 중요한 단어 "고양이"를 저장함 / At $t=1$, store the important word "cat"
- 시간 $t=2$~$t=99$에서 무관한 단어들이 계속 들어옴 / At $t=2$–$t=99$, irrelevant words keep arriving
- 게이트 없이는 같은 가중치가 $t=1$에서는 열어야 하고 $t=2$~$t=99$에서는 닫아야 하는데, 고정된 가중치로는 불가능 / Without gates, the same fixed weight would need to open at $t=1$ and close at $t=2$–$t=99$ — impossible with a fixed weight

입력 게이트는 **문맥(현재 입력 + 이전 hidden state)**을 보고 동적으로 열고 닫을 수 있으므로, 이 충돌을 해결합니다.

The input gate resolves this conflict by dynamically opening and closing based on **context (current input + previous hidden state)**.

##### Output Gate ($out_j$) — 금고의 출구 문지기 / Gatekeeper at the Vault Exit

$$y^{out_j}(t) = \sigma\left(\sum_u w_{out_j u} \cdot y^u(t-1)\right)$$

출력 게이트는 **"지금 금고의 내용을 다른 사람에게 보여줄 것인가?"**를 결정합니다.

The output gate decides: **"Should we show the vault's contents to others now?"**

- $y^{out_j} \approx 1$ (열림/open): cell state의 내용이 hidden state로 출력됨 / Cell state contents are output to hidden state
- $y^{out_j} \approx 0$ (닫힘/closed): cell state가 외부에 노출되지 않음 (정보는 금고 안에 안전하게 보존) / Cell state is not exposed externally (information safely preserved inside the vault)

**왜 필요한가?** "output weight conflict" 때문입니다:

**Why is it needed?** Because of the "output weight conflict":

- 금고에 "이 문장의 주어는 '고양이'"라는 정보가 저장되어 있음 / The vault stores "the subject of this sentence is 'cat'"
- $t=50$에서 동사가 나올 때 이 정보가 필요함 (출력 게이트 열림) / At $t=50$ when a verb appears, this info is needed (output gate opens)
- 그러나 $t=2$~$t=49$에서는 이 정보를 출력하면 안 됨 — 다른 유닛의 계산을 교란하기 때문 / But at $t=2$–$t=49$, outputting this would disturb other units' computations
- 출력 게이트가 닫혀 있으면 금고 내용이 **숨겨지므로** 교란이 발생하지 않음 / When output gate is closed, vault contents are **hidden**, preventing disturbance

논문 실험 1(Embedded Reber Grammar)에서 출력 게이트의 중요성이 명확히 드러납니다: 첫 T/P를 저장한 후, 원래 Reber 문법의 쉬운 전이 활성화가 장기 기억을 교란하지 않도록 출력 게이트가 차단합니다.

In Experiment 1 (Embedded Reber Grammar), the importance of output gates is clearly demonstrated: after storing the first T/P, output gates block the easier Reber grammar transition activations from disturbing long-term memories.

##### Squashing Functions $g$와 $h$ — 값의 범위 조절 / Value Range Control

**$g$ (입력 압축, 범위 $[-2, 2]$)**: cell state에 추가될 값을 적절한 범위로 제한합니다. cell state가 무한히 커지는 것을 방지하는 역할을 합니다 (물론 원래 LSTM에는 forget gate가 없어서 cell state가 계속 누적되는 문제가 있었고, 이것이 2000년 forget gate 추가의 동기가 됩니다).

**$g$ (input squashing, range $[-2, 2]$)**: Constrains values added to cell state. Prevents cell state from growing unboundedly (though the original LSTM had no forget gate, so cell state could still accumulate — this motivated the forget gate in 2000).

**$h$ (출력 스케일링, 범위 $[-1, 1]$)**: cell state를 출력하기 전에 스케일링합니다. cell state의 절대값이 크더라도 출력은 $[-1, 1]$ 범위로 제한됩니다. $h'(s)$가 0에 가까워지면 기울기가 사라질 수 있지만, **CEC 내부 경로에서는** 이 함수를 거치지 않으므로 장기 기울기에 영향을 주지 않습니다.

**$h$ (output scaling, range $[-1, 1]$)**: Scales cell state before output. Even if cell state has large absolute values, output is constrained to $[-1, 1]$. While $h'(s)$ approaching 0 could lose gradients, this function is **not on the CEC internal path**, so long-range gradients are unaffected.

##### 전체 정보 흐름 다이어그램 / Complete Information Flow Diagram

```
시간 t에서의 처리 / Processing at time t:

입력 x(t) + 이전 출력 h(t-1)
         │
    ┌────┴────┐────────────┐
    ▼         ▼            ▼
 Input Gate  Cell Input   Output Gate
 σ(W_in·[x,h])  g(W_c·[x,h])  σ(W_out·[x,h])
 [0~1]       [-2~2]       [0~1]
    │         │            │
    └──── × ──┘            │
         │                 │
         ▼                 │
   s(t) = s(t-1) + in·g   │   ← CEC: 이전 상태 + 게이트된 입력
         │                 │      CEC: previous state + gated input
         ▼                 │
       h(s(t))             │
       [-1~1]              │
         │                 │
         └──────── × ──────┘
                   │
                   ▼
              y(t) = out · h(s)   ← 최종 출력 (hidden state)
                                     Final output (hidden state)
```

핵심은 **cell state 경로($s(t-1) \to s(t)$)가 덧셈만으로 구성**되어 있다는 것입니다. 곱셈적 게이트는 이 경로에 "사이드에서" 정보를 추가하거나 출력을 조절할 뿐, 경로 자체를 변형하지 않습니다.

The key is that **the cell state path ($s(t-1) \to s(t)$) consists only of addition**. Multiplicative gates add information "from the side" or regulate output, but never transform the path itself.

##### 2000년 추가된 Forget Gate — 완성된 현대 LSTM / Forget Gate (2000) — Completing Modern LSTM

원래 LSTM에는 정보를 **잊는** 메커니즘이 없어서, cell state가 계속 누적됩니다. Gers et al. (2000)이 **forget gate** $f_j$를 추가하여:

The original LSTM had no mechanism for **forgetting**, so cell state accumulated indefinitely. Gers et al. (2000) added the **forget gate** $f_j$:

$$s_{c_j}(t) = \underbrace{f_j(t) \cdot s_{c_j}(t-1)}_{\text{잊을 만큼 잊고 / forget as needed}} + \underbrace{y^{in_j}(t) \cdot g(net_{c_j}(t))}_{\text{기억할 만큼 기억 / remember as needed}}$$

- $f_j \approx 0$: 이전 상태를 완전히 잊음 / Completely forget previous state
- $f_j \approx 1$: 원래 CEC처럼 이전 상태를 완전히 보존 / Fully preserve previous state (original CEC behavior)

이것이 현대 LSTM의 표준 형태이며, $f_j$를 1에 가깝게 초기화(bias를 양수로)하면 원래 CEC의 장기 기억 능력을 유지하면서도 불필요한 정보를 능동적으로 제거할 수 있습니다.

This is the standard form of modern LSTM. Initializing $f_j$ close to 1 (positive bias) maintains the original CEC's long-term memory while enabling active removal of unnecessary information.

#### Network Topology / 네트워크 토폴로지

LSTM 네트워크는 세 층으로 구성됩니다:

LSTM networks consist of three layers:

1. **입력층 (Input layer)**: 현재 입력을 받음 / Receives current input
2. **은닉층 (Hidden layer)**: 메모리 셀 + 게이트 유닛 (+ 선택적으로 일반 은닉 유닛). 자기 순환적으로 연결 / Memory cells + gate units (+ optionally conventional hidden units). Self-recurrently connected
3. **출력층 (Output layer)**: 예측/분류 출력 / Prediction/classification output

모든 유닛(게이트 제외)은 위 층의 모든 유닛으로 전방 연결됩니다. 게이트와 메모리 셀은 모든 비출력 유닛으로부터 입력 연결을 받습니다.

All units (except gates) have forward connections to all units in the layer above. Gates and memory cells receive input connections from all non-output units.

#### Memory Cell Blocks / 메모리 셀 블록

동일한 입력 게이트와 출력 게이트를 공유하는 $S$개의 메모리 셀은 "크기 $S$의 메모리 셀 블록"을 형성합니다. 블록 아키텍처는 개별 셀보다 약간 더 효율적입니다 (게이트 수가 줄어들므로). 크기 1의 블록은 단순한 메모리 셀입니다.

$S$ memory cells sharing the same input and output gates form a "memory cell block of size $S$." Block architecture is slightly more efficient than individual cells (fewer gates). A block of size 1 is a simple memory cell.

#### 학습 알고리즘: 절단된 기울기 / Learning: Truncated Gradient

LSTM은 RTRL의 변형을 사용하되, **기울기 절단(gradient truncation)**을 적용합니다. "메모리 셀 net 입력" ($net_{c_j}$, $net_{in_j}$, $net_{out_j}$)에 도달한 오류는 더 이상 과거로 전파되지 않습니다 (incoming 가중치를 변경하는 데는 사용됨). 오류는 메모리 셀 **내부**에서만 과거로 전파됩니다 — CEC를 통해 무한히.

LSTM uses a variant of RTRL with **gradient truncation**. Errors arriving at "memory cell net inputs" ($net_{c_j}$, $net_{in_j}$, $net_{out_j}$) are not propagated further back in time (though they do serve to change incoming weights). Errors are propagated back through time only **within** memory cells — indefinitely through the CEC.

이 절단이 왜 작동하는가:
- 오류가 메모리 셀 출력에 도달하면, 출력 게이트 활성화와 $h'$에 의해 스케일링됨
- CEC 안에서는 스케일링 없이 무한히 흐를 수 있음
- 메모리 셀을 떠날 때 입력 게이트 활성화와 $g'$에 의해 한 번 더 스케일링된 후 절단됨
- 이것이 LSTM이 효율적이면서도 장기 의존성을 학습할 수 있는 이유

Why this truncation works:
- When error reaches a memory cell output, it's scaled by output gate activation and $h'$
- Within the CEC, it flows indefinitely without scaling
- When leaving the memory cell, it's scaled once more by input gate activation and $g'$, then truncated
- This is why LSTM is efficient yet capable of learning long-range dependencies

**계산 복잡도**: 가중치 당 타임스텝 당 $O(W)$ — 완전 순환 네트워크에 대한 BPTT와 동일. 공간과 시간 모두에서 **local** (네트워크 크기나 시퀀스 길이에 무관).

**Computational complexity**: $O(W)$ per time step — same as BPTT for fully recurrent nets. **Local** in both space and time (independent of network size or sequence length).

#### Abuse Problem과 해결 / Abuse Problem and Solutions

학습 초기에 네트워크가 메모리 셀을 "남용"할 수 있습니다 — 정보를 저장하지 않고, 활성화를 일정하게 유지하여 bias처럼 사용합니다. 이렇게 "남용된" 셀은 나중에 해제되어 실제 학습에 사용되기까지 오래 걸릴 수 있습니다.

In early training, the network may "abuse" memory cells — using them as bias units by keeping activations constant rather than storing information. Such "abused" cells may take long to be released for actual learning.

해결책: (1) **Sequential network construction** — 오류가 줄어들지 않을 때 새 셀 추가. (2) **Output gate bias** — 출력 게이트 bias를 음수로 초기화하여, 초기에 메모리 셀 활성화를 0 근처로 밀어 남용을 방지.

Solutions: (1) **Sequential network construction** — add new cells when error stops decreasing. (2) **Output gate bias** — initialize output gate bias to negative values, pushing initial memory cell activations near zero to prevent abuse.

---

### Section 5: Experiments — 6가지 실험적 검증 / Six Experimental Validations

#### 실험 1: Embedded Reber Grammar / Experiment 1: Embedded Reber Grammar

**작업**: 확장된 Reber 문법에서 생성된 문자열을 학습. 최소 시간 지연은 9스텝으로, 장기 문제는 아니지만 RNN 벤치마크의 표준입니다.

**Task**: Learn strings from the embedded Reber grammar. Minimal time lag is 9 steps — not a long time lag problem, but a standard RNN benchmark.

**결과**: LSTM은 30번의 시도에서 **항상** 성공 (97-100%). RTRL은 "일부 비율"만 성공, Elman net은 0%, RCC는 50%만 성공. LSTM이 성공적인 시도만 비교해도 더 빠르게 학습합니다.

**Results**: LSTM **always** succeeds in 30 trials (97-100%). RTRL succeeds only "some fraction," Elman net 0%, RCC only 50%. Even comparing only successful trials, LSTM learns faster.

**핵심 관찰**: 출력 게이트의 중요성이 드러남. 첫 번째 T 또는 P를 저장할 때, 원래 Reber 문법의 더 쉬운 전이를 나타내는 활성화가 장기 기억을 교란하지 않아야 합니다. 출력 게이트가 이를 방지합니다.

**Key observation**: The importance of output gates is revealed. When storing the first T or P, activations representing easier transitions of the original Reber grammar must not disturb long-term memories. Output gates prevent this.

#### 실험 2a: 노이즈 없는 시퀀스, 장기 시간 지연 / Experiment 2a: Noise-free Sequences, Long Time Lags

**작업**: $p+1$개의 가능한 입력 기호 중 2개의 매우 유사한 시퀀스를 구분. 마지막 원소를 예측하려면 첫 번째 원소를 $p$ 타임스텝 동안 기억해야 합니다.

**Task**: Distinguish two very similar sequences from $p+1$ possible input symbols. Predicting the final element requires remembering the first element for $p$ time steps.

**결과 (Table 2)**:
- $p = 4$ (짧은 지연): RTRL은 78% 성공, 1,043,000번의 시퀀스 필요
- $p = 10$: RTRL 0% 성공, BPTT 0% 성공 (5,000,000+ 시퀀스에서)
- $p = 100$: **LSTM만 성공** — 100% 성공률, 5,040번의 시퀀스로!
- Neural Sequence Chunker(CH)는 33% 성공, 32,400번 필요

**Results (Table 2)**:
- $p = 4$ (short lag): RTRL 78% success, 1,043,000 sequences needed
- $p = 10$: RTRL 0%, BPTT 0% (at 5,000,000+ sequences)
- $p = 100$: **Only LSTM succeeds** — 100% success, in just 5,040 sequences!
- Neural Sequence Chunker (CH) 33% success, 32,400 needed

#### 실험 2c: 매우 긴 시간 지연 + 많은 노이즈 / Experiment 2c: Very Long Lags + Heavy Noise

**작업**: 실험 2a의 확장. 많은 "방해 기호(distractor symbols)"가 랜덤 위치에 삽입되고, 최소 시간 지연이 최대 1000 타임스텝. **논문에서 가장 어려운 작업**으로, 다른 어떤 RNN 알고리즘도 풀지 못했습니다.

**Task**: Extension of 2a. Many "distractor symbols" at random positions, minimal time lags up to 1000 steps. **The hardest task in the paper** — no other RNN algorithm could solve it.

**결과 (Table 3)**: LSTM은 1000 타임스텝 지연에서도 성공! 학습 시간이 시간 지연에 **비례적으로** 증가하는 놀라운 특성을 보임 (RTRL/BPTT는 기하급수적).

**Results (Table 3)**: LSTM succeeds even at 1000 time step delays! Learning time increases **proportionally** to the time lag — a remarkable property (RTRL/BPTT scale exponentially).

| Time lag $q$ | Input symbols $p$ | 학습 성공까지 시퀀스 수 |
|---|---|---|
| 50 | 50 | 30,000 |
| 100 | 100 | 31,000 |
| 500 | 500 | 38,000 |
| 1,000 | 1,000 | 49,000 |

핵심: 시간 지연이 20배 증가해도 학습 시간은 1.6배만 증가!

Key: Even as time lag increases 20×, learning time increases only 1.6×!

#### 실험 4: Adding Problem / Experiment 4: Adding Problem

**작업**: 시퀀스의 각 원소는 (실수값, 마커)의 쌍. 마커가 표시된 두 원소의 실수값의 합을 시퀀스 끝에서 출력해야 합니다. 분산된 연속값 표상과 장기 저장을 동시에 요구합니다.

**Task**: Each sequence element is a (real-value, marker) pair. Output the sum of real values at the two marked positions at sequence end. Requires distributed continuous-valued representation and long-term storage simultaneously.

**결과**: LSTM은 최소 지연 500에서도 2560개 테스트 시퀀스 중 0~1개만 오류. **다른 어떤 RNN 알고리즘도 이 유형의 문제를 풀지 못했습니다.**

**Results**: LSTM achieves 0-1 errors out of 2560 test sequences even at minimal lag 500. **No other RNN algorithm has solved this type of problem.**

#### 실험 5: Multiplication Problem / Experiment 5: Multiplication Problem

CEC의 내장된 적분 능력(덧셈)이 Adding Problem의 성공 요인일 수 있다는 우려를 해소하기 위한 실험. 합이 아닌 곱을 학습해야 하므로, 본질적으로 비적분적(non-integrative)인 작업입니다. LSTM은 이것도 성공적으로 해결합니다.

Experiment to address the concern that CEC's built-in integration (addition) may be the key to the Adding Problem's success. Requires learning products instead of sums — an inherently non-integrative task. LSTM successfully solves this too.

#### 실험 6: Temporal Order / Experiment 6: Temporal Order

**작업**: 시퀀스 내 널리 분리된 2-3개의 관련 입력의 시간적 순서를 분류. 예: $(X, Y)$ 순서에 따라 4개 클래스 중 하나로 분류.

**Task**: Classify based on temporal order of 2-3 relevant inputs widely separated in a sequence. E.g., classify into one of 4 classes based on order of $(X, Y)$.

**결과**: LSTM은 2560개 테스트 시퀀스 중 1-2개만 오류. 시간적 순서에 대한 정보를 추출하는 능력을 입증합니다.

**Results**: LSTM achieves only 1-2 errors out of 2560 test sequences. Demonstrates ability to extract information about temporal order.

---

### Section 6: Discussion — 한계와 장점 / Limitations and Advantages

#### 한계 / Limitations

1. **LSTM은 강한 시간 지연 문제를 빠르게 풀지만**, 시간 지연이 길어질수록 학습이 느려지는 것은 피할 수 없습니다 (선형적으로).
2. **Forget gate**가 아직 도입되지 않았습니다 (2000년 Gers et al.이 추가). 원래 LSTM은 정보를 영구히 저장하며, 명시적으로 "잊는" 메커니즘이 없습니다.
3. **Abuse problem**: 학습 초기에 메모리 셀이 bias로 사용될 수 있으며, 이를 해소하는 데 시간이 걸립니다.

1. **LSTM solves strong time lag problems quickly, but** learning inevitably slows as time lags increase (linearly).
2. **Forget gate** not yet introduced (added by Gers et al. in 2000). Original LSTM stores information permanently with no explicit "forgetting" mechanism.
3. **Abuse problem**: Memory cells may be used as biases early in training, taking time to resolve.

#### 장점 / Advantages

1. **계산 복잡도**: 가중치 당 $O(1)$ — BPTT와 동일. 공간과 시간 모두 local.
2. **긴 시간 지연**: 1000+ 타임스텝에서도 작동.
3. **노이즈와 방해에 강건**: 많은 distractor symbol이 있어도 학습 가능.
4. **연속값 저장**: 실수값을 장기간 정확히 저장 (Adding, Multiplication 문제).
5. **시간적 순서 추출**: 널리 분리된 입력의 순서를 학습.

1. **Computational complexity**: $O(1)$ per weight — same as BPTT. Local in both space and time.
2. **Long time lags**: Works at 1000+ time steps.
3. **Robust to noise and distractors**: Learns despite many distractor symbols.
4. **Continuous value storage**: Precisely stores real values long-term (Adding, Multiplication problems).
5. **Temporal order extraction**: Learns order of widely separated inputs.

---

## 핵심 시사점 / Key Takeaways

1. **기울기 소실은 구조적 문제이며 구조적 해결이 필요하다**: 학습률 조정, 가중치 초기화, 새로운 활성화 함수 등의 기법은 기울기 소실의 근본 원인 — 반복적인 곱셈에 의한 기하급수적 감쇠 — 을 해결하지 못합니다. LSTM은 아키텍처 자체를 변경하여 (CEC의 가중치 1.0 자기 순환) 오류 흐름의 수학적 성질을 근본적으로 바꿉니다.

   **Vanishing gradients are a structural problem requiring a structural solution**: Techniques like learning rate tuning, weight initialization, or new activation functions don't address the root cause — exponential decay from repeated multiplication. LSTM fundamentally changes error flow's mathematical properties by modifying the architecture itself (CEC's weight-1.0 self-recurrence).

2. **곱셈적 게이팅은 정보 흐름을 제어하는 강력한 메커니즘이다**: 게이트의 핵심 통찰은 $0 \times \text{anything} = 0$이라는 단순한 산술에 있습니다. 시그모이드 게이트는 0(완전 차단)에서 1(완전 통과)까지 연속적으로 조절 가능하며, 이 자체가 미분 가능하므로 역전파로 학습됩니다. 이 아이디어는 이후 GRU, Highway Networks, Transformer의 Attention까지 영향을 미칩니다.

   **Multiplicative gating is a powerful mechanism for controlling information flow**: The key insight of gates lies in simple arithmetic: $0 \times \text{anything} = 0$. Sigmoid gates are continuously adjustable from 0 (full block) to 1 (full pass), and are differentiable so they're learned via backprop. This idea later influences GRU, Highway Networks, and Transformer's Attention.

3. **장기 기억과 단기 처리의 분리가 핵심이다**: LSTM의 이름 자체가 이를 말합니다 — "Long Short-Term Memory." Cell state는 장기 정보를 저장하고 (CEC를 통해), 게이트는 단기적 결정(언제 읽고/쓸지)을 내립니다. 이 분리가 두 시간 스케일을 동시에 처리할 수 있게 합니다.

   **Separation of long-term memory and short-term processing is key**: LSTM's name says it all — "Long Short-Term Memory." Cell state stores long-term information (via CEC), gates make short-term decisions (when to read/write). This separation enables simultaneous handling of both time scales.

4. **효율성과 능력의 양립은 가능하다**: LSTM의 truncated gradient는 장기 기울기를 CEC 내부에서만 보존하고 나머지는 절단합니다. 이것이 "기울기를 잘라도 장기 의존성을 학습할 수 있다"는 놀라운 결과를 낳습니다. 계산 복잡도는 BPTT와 동일한 $O(W)$이면서, BPTT가 실패하는 작업을 해결합니다.

   **Efficiency and capability can coexist**: LSTM's truncated gradient preserves long-range gradients only within the CEC, truncating the rest. This yields the surprising result that "cutting gradients still allows learning long-range dependencies." Complexity is $O(W)$ — same as BPTT — yet solves tasks where BPTT fails.

5. **학습 시간의 선형적 스케일링은 LSTM의 고유한 특성이다**: Table 3에서 시간 지연이 50에서 1000으로 20배 증가할 때, 학습 시간은 30,000에서 49,000으로 약 1.6배만 증가합니다. RTRL/BPTT는 기하급수적으로 스케일링됩니다. 이것은 CEC를 통한 일정한 오류 흐름의 직접적 결과입니다.

   **Linear scaling of learning time is LSTM's unique property**: In Table 3, when time lag increases 20× (50→1000), learning time increases only ~1.6× (30K→49K). RTRL/BPTT scale exponentially. This is a direct consequence of constant error flow through the CEC.

6. **"쓸 때"와 "읽을 때"의 학습은 "무엇을 쓸지"의 학습만큼 중요하다**: Input/output weight conflict 분석은 게이트의 필요성을 동기 부여합니다. 단순히 좋은 표상을 학습하는 것만으로는 부족하며, 언제 저장하고 언제 출력할지를 학습하는 것이 장기 의존성 해결의 열쇠입니다.

   **Learning "when to write/read" is as important as learning "what to write"**: The input/output weight conflict analysis motivates gates. Simply learning good representations is insufficient — learning when to store and when to output is the key to solving long-range dependencies.

7. **LSTM은 1997년 이후 거의 20년간 시퀀스 모델링을 지배했다**: 음성 인식, 기계 번역, 텍스트 생성, 시계열 예측 등 시퀀스 데이터가 관련된 거의 모든 분야에서 LSTM (또는 그 변형인 GRU)이 표준이 되었습니다. 2017년 Transformer가 등장하기 전까지 LSTM의 지위는 흔들리지 않았습니다.

   **LSTM dominated sequence modeling for nearly 20 years after 1997**: Speech recognition, machine translation, text generation, time series forecasting — LSTM (or its variant GRU) became the standard for virtually every field involving sequence data. LSTM's position was unchallenged until the Transformer appeared in 2017.

---

## 수학적 요약 / Mathematical Summary

### LSTM Forward Pass 전체 흐름 / Complete LSTM Forward Pass

**입력 / Input**: 시퀀스 $\mathbf{x}(1), \mathbf{x}(2), \ldots, \mathbf{x}(T)$

**각 타임스텝 $t$에서 / At each time step $t$:**

**Step 1 — Net 입력 계산 / Compute Net Inputs:**
$$net_{in_j}(t) = \sum_u w_{in_j u} y^u(t-1)$$
$$net_{out_j}(t) = \sum_u w_{out_j u} y^u(t-1)$$
$$net_{c_j}(t) = \sum_u w_{c_j u} y^u(t-1)$$

**Step 2 — 게이트 활성화 / Gate Activations:**
$$y^{in_j}(t) = \sigma(net_{in_j}(t)) \quad \text{(input gate, range [0,1])}$$
$$y^{out_j}(t) = \sigma(net_{out_j}(t)) \quad \text{(output gate, range [0,1])}$$

**Step 3 — Cell State 업데이트 / Cell State Update (THE KEY EQUATION):**
$$s_{c_j}(t) = s_{c_j}(t-1) + y^{in_j}(t) \cdot g(net_{c_j}(t))$$

**Step 4 — Cell 출력 / Cell Output:**
$$y^{c_j}(t) = y^{out_j}(t) \cdot h(s_{c_j}(t))$$

**활성화 함수 / Activation Functions:**
- $\sigma$: logistic sigmoid, range $[0, 1]$
- $g$: scaled sigmoid, range $[-2, 2]$
- $h$: scaled sigmoid, range $[-1, 1]$

### 현대 LSTM과의 비교 / Comparison with Modern LSTM

| 구성 요소 / Component | 1997 원본 / Original | 현대 표준 / Modern Standard |
|---|---|---|
| Cell state update | $s_t = s_{t-1} + i_t \cdot g(net_c)$ | $s_t = f_t \odot s_{t-1} + i_t \odot \tilde{c}_t$ |
| Forget gate | **없음** (정보 영구 저장) | $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ |
| Input gate | $\sigma(net_{in})$ | $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ |
| Output gate | $\sigma(net_{out})$ | $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$ |
| Candidate | $g(net_c)$ | $\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$ |
| Peephole | 없음 | 선택적 (cell state → gate 연결) |

현대 LSTM의 핵심 차이는 **forget gate**의 추가입니다 (Gers et al., 2000). 이것은 cell state에 $f_t$를 곱하여 불필요한 정보를 능동적으로 "잊을" 수 있게 합니다.

The key difference in modern LSTM is the addition of the **forget gate** (Gers et al., 2000). This multiplies cell state by $f_t$, enabling active "forgetting" of unnecessary information.

---

## 역사적 맥락의 타임라인 / Paper in the Arc of History

```
1986 ─── Rumelhart et al. ─── Backpropagation
            │         다층 신경망 학습의 시작
            │
1988 ─── Elman ─── Simple Recurrent Network
            │         시퀀스 처리용 RNN
            │
1990 ─── Williams & Zipser ─── RTRL / Werbos ─── BPTT
            │         RNN 학습 알고리즘
            │
1991 ─── Hochreiter ─── Vanishing Gradient 분석 (졸업 논문)
            │         문제의 수학적 규명
            │
1994 ─── Bengio et al. ─── "Learning long-term dependencies is hard"
            │         기울기 소실의 실험적 확인
            │
1995 ─── Cortes & Vapnik ─── SVM (고정 길이 벡터 분류)
            │
     ╔═══════════════════════════════════════════╗
     ║  ★ 1997 ─── Hochreiter & Schmidhuber      ║
     ║       Long Short-Term Memory               ║
     ║       CEC + Gates = 기울기 소실 해결         ║
     ╚═══════════════════════════════════════════╝
            │
2000 ─── Gers et al. ─── Forget Gate 추가 → 현대 LSTM 완성
            │
2009 ─── Graves ─── LSTM으로 필기 인식 → 실용적 응용
            │
2013 ─── Graves (Google) ─── LSTM 음성 인식 → 업계 채택
            │
2014 ─── Cho et al. ─── GRU (LSTM의 경량 변형)
            │         Forget + Input gate를 하나의 Update gate로 통합
            │
2014 ─── Sutskever et al. ─── Seq2Seq with LSTM → 기계 번역
            │
2014 ─── Bahdanau et al. ─── Attention + LSTM
            │         LSTM의 한계를 보완하는 메커니즘
            │
2017 ─── Vaswani et al. ─── Transformer
            │         LSTM을 대체하는 새로운 패러다임
            │
현재 ── LSTM은 여전히 실시간/스트리밍 작업에서 사용됨
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| 논문 / Paper | 연결 / Connection |
|---|---|
| #1 McCulloch & Pitts (1943) | M-P 뉴런은 이진 임계값 유닛. LSTM의 게이트도 시그모이드를 통한 이진적(0/1) 결정을 내림 — M-P 뉴런의 연속화된 버전 / M-P neurons are binary threshold units. LSTM gates also make binary-like (0/1) decisions via sigmoid — a continuous version of M-P neurons |
| #3 Rosenblatt (1958) | Perceptron은 단일 스텝 분류. LSTM은 시퀀스에 걸친 분류 — 시간이라는 차원의 추가 / Perceptron for single-step classification. LSTM for classification over sequences — adding the dimension of time |
| #6 Rumelhart et al. (1986) | Backpropagation이 LSTM 학습의 기반. LSTM은 BP의 한계(기울기 소실)를 아키텍처적으로 해결 / Backpropagation is the basis of LSTM training. LSTM architecturally resolves BP's limitation (vanishing gradients) |
| #7 LeCun et al. (1989) | CNN은 공간적 계층 구조 학습, LSTM은 시간적 의존성 학습. 다른 차원(공간 vs 시간)에서 동일한 원리: 구조화된 연결로 귀납적 편향 도입 / CNN learns spatial hierarchy, LSTM learns temporal dependencies. Same principle in different dimensions: structured connections introduce inductive bias |
| #8 Cortes & Vapnik (1995) | SVM은 고정 길이 입력에 대한 분류. LSTM은 가변 길이 시퀀스를 처리하여 고정 길이 표상으로 변환. 두 방법은 서로 다른 도메인을 지배 / SVM classifies fixed-length inputs. LSTM processes variable-length sequences into fixed-length representations. The two methods dominate different domains |
| #10 LeCun et al. (1998) | LeNet-5의 풀링은 공간적 불변성, LSTM의 게이트는 시간적 선택성. 둘 다 "관련 정보만 통과"시키는 메커니즘 / LeNet-5's pooling provides spatial invariance, LSTM's gates provide temporal selectivity. Both are mechanisms for "passing only relevant information" |
| #17 Bahdanau et al. (2014) | Attention은 LSTM의 한계(고정 길이 hidden state 병목)를 보완. Attention의 아이디어는 LSTM의 게이팅 메커니즘에서 영감 / Attention complements LSTM's limitation (fixed-length hidden state bottleneck). Attention's idea is inspired by LSTM's gating mechanism |
| #20 Vaswani et al. (2017) | Transformer는 순환을 완전히 제거하고 self-attention으로 대체. LSTM의 순차 처리 한계(병렬화 불가)를 해결 / Transformer completely removes recurrence, replacing with self-attention. Resolves LSTM's sequential processing limitation (no parallelization) |

---

## 참고문헌 / References

- Hochreiter, S. & Schmidhuber, J., "Long Short-Term Memory", *Neural Computation*, 9(8), pp. 1735–1780, 1997.
- Hochreiter, S., "Untersuchungen zu dynamischen neuronalen Netzen", Diploma thesis, TU Munich, 1991.
- Bengio, Y., Simard, P. & Frasconi, P., "Learning Long-Term Dependencies with Gradient Descent is Difficult", *IEEE Trans. Neural Networks*, 5(2), pp. 157–166, 1994.
- Gers, F., Schmidhuber, J. & Cummins, F., "Learning to Forget: Continual Prediction with LSTM", *Neural Computation*, 12(10), pp. 2451–2471, 2000.
- Williams, R. & Zipser, D., "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks", *Neural Computation*, 1(2), pp. 270–280, 1989.
- Elman, J., "Finding Structure in Time", *Cognitive Science*, 14, pp. 179–211, 1990.
