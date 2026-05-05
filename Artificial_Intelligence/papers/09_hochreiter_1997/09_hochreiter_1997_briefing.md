---
title: "Long Short-Term Memory — Pre-reading Briefing"
paper: "Long Short-Term Memory"
authors: Sepp Hochreiter, Jürgen Schmidhuber
year: 1997
journal: "Neural Computation, Vol. 9(8), pp. 1735–1780"
type: briefing
date: 2026-04-09
---

# Long Short-Term Memory — 사전 읽기 브리핑 / Pre-reading Briefing

## 핵심 기여 / Core Contribution

이 논문은 순환 신경망(RNN)의 근본적 문제인 **기울기 소실(vanishing gradient)** 및 **기울기 폭발(exploding gradient)**을 해결하는 새로운 아키텍처 **LSTM (Long Short-Term Memory)**을 제안합니다. 핵심 아이디어는 **Constant Error Carrousel (CEC)** — 자기 자신에 대한 가중치 1.0의 순환 연결을 가진 선형 유닛으로, 오류 신호가 시간에 걸쳐 소실도 폭발도 없이 일정하게 흐릅니다. 여기에 **입력 게이트(input gate)**와 **출력 게이트(output gate)**라는 곱셈적 유닛을 추가하여, 메모리 셀에 정보를 쓰고 읽는 시점을 학습합니다. LSTM은 1000 타임스텝 이상의 장기 의존성(long-range dependency)을 학습할 수 있으며, BPTT와 RTRL이 완전히 실패하는 작업에서도 성공합니다. LSTM은 이후 거의 20년간 시퀀스 모델링(텍스트, 음성, 시계열)의 지배적 아키텍처가 되었습니다.

This paper proposes **LSTM (Long Short-Term Memory)**, a novel architecture that solves the fundamental **vanishing/exploding gradient** problems in recurrent neural networks (RNNs). The key idea is the **Constant Error Carrousel (CEC)** — a linear unit with a self-recurrent connection of weight 1.0, through which error signals flow constantly over time without vanishing or exploding. **Input gates** and **output gates** (multiplicative units) are added to learn *when* to write to and read from memory cells. LSTM can learn long-range dependencies exceeding 1000 time steps, succeeding on tasks where BPTT and RTRL completely fail. LSTM became the dominant architecture for sequence modeling (text, speech, time series) for nearly two decades.

---

## 역사적 맥락 / Historical Context

```
1986 ─── Rumelhart et al. ─── Backpropagation
            │         다층 신경망 학습 가능
            │
1988 ─── Elman ─── Simple Recurrent Network (SRN)
            │         시퀀스 처리를 위한 순환 신경망
            │
1989 ─── LeCun ─── CNN (공간적 구조 학습)
            │
1990 ─── Williams & Zipser ─── RTRL / Werbos ─── BPTT
            │         RNN 학습 알고리즘들
            │
1991 ─── Hochreiter ─── 기울기 소실 문제 분석 (졸업 논문)
            │         ★ 문제를 수학적으로 분석한 최초의 연구
            │
1992 ─── Mozer ─── Time constants 접근
            │         장기 의존성 시도 (제한적 성공)
            │
1994 ─── Bengio et al. ─── 기울기 소실에 대한 실험적 확인
            │         "Learning long-term dependencies is hard"
            │
     ╔═══════════════════════════════════════════╗
     ║  ★ 1997 ─── Hochreiter & Schmidhuber      ║
     ║       Long Short-Term Memory (LSTM)        ║
     ║       CEC + Gates = 장기 의존성 학습 해결    ║
     ╚═══════════════════════════════════════════╝
            │
2000 ─── Gers et al. ─── Forget Gate 추가
            │         현대 LSTM의 완성
            │
2014 ─── Cho et al. ─── GRU (LSTM의 경량 변형)
            │
2014 ─── Bahdanau et al. ─── Attention 메커니즘
            │         LSTM의 한계를 보완
            │
2017 ─── Vaswani et al. ─── Transformer
            │         LSTM을 대체하는 새로운 패러다임
```

1990년대 초, RNN은 이론적으로 시퀀스 데이터를 처리할 수 있었지만, 실제로 10스텝 이상의 의존성을 학습하기 어려웠습니다. Hochreiter는 1991년 졸업 논문에서 이 문제를 수학적으로 분석하여, 기울기가 시간에 따라 기하급수적으로 소실되거나 폭발하는 것이 근본 원인임을 밝혔습니다. LSTM은 이 문제에 대한 직접적 해결책으로, 오류 흐름을 "일정하게" 유지하는 아키텍처를 설계했습니다.

In the early 1990s, RNNs could theoretically process sequence data, but struggled to learn dependencies beyond ~10 steps. Hochreiter's 1991 thesis mathematically analyzed this problem, showing that gradients exponentially vanish or explode over time. LSTM is a direct solution, designing an architecture that keeps error flow "constant."

---

## 필요한 배경 지식 / Prerequisites

### 1. 순환 신경망 (RNN) 기초 / Recurrent Neural Network Basics
- **순환 연결 (recurrent connection)**: 이전 타임스텝의 출력이 현재 타임스텝의 입력이 되는 구조
  - $h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b)$
- **시퀀스 처리**: 한 번에 하나의 토큰(문자, 단어, 시간 단계)을 처리
- **은닉 상태 (hidden state)**: 네트워크의 "기억" — 과거 정보를 요약

### 2. Backpropagation Through Time (BPTT) / 시간을 통한 역전파
- RNN을 시간 축으로 "펼쳐서(unfold)" 일반 역전파를 적용
- 시간 $t$에서의 오류를 과거 시간 $t-1, t-2, \ldots$로 전파
- **문제**: 긴 시간에 걸쳐 기울기가 반복적으로 곱해지며 소실/폭발

### 3. 기울기 소실/폭발 문제 / Vanishing/Exploding Gradient Problem
- **소실 (vanishing)**: $|f'(net) \cdot w| < 1$이 반복되면 기울기가 기하급수적으로 0에 수렴 → 장기 의존성 학습 불가
- **폭발 (exploding)**: $|f'(net) \cdot w| > 1$이 반복되면 기울기가 기하급수적으로 증가 → 가중치 발산, 불안정한 학습
- 시그모이드 함수의 경우: $f'_{\max} = 0.25$ (최대 미분값), 가중치가 4 미만이면 거의 항상 소실

### 4. 이전 논문의 개념 / Concepts from Prior Papers
- **Backpropagation (논문 #6)**: LSTM도 기울기 기반 학습을 사용하지만, 구조적으로 기울기 소실을 방지
- **CNN (논문 #7)**: 공간적 구조에 대한 학습. LSTM은 시간적 구조에 대한 학습
- **SVM (논문 #8)**: 고정 길이 벡터를 위한 분류기. LSTM은 가변 길이 시퀀스를 처리

### 5. 수학적 배경 / Mathematical Background
- **미분의 연쇄 법칙 (chain rule)**: $\frac{\partial \vartheta_v(t-q)}{\partial \vartheta_u(t)}$ 형태의 시간을 통한 오류 전파
- **시그모이드 함수**: $\sigma(x) = \frac{1}{1 + e^{-x}}$, 범위 $[0, 1]$ — 게이트의 "열림/닫힘"을 제어
- **행렬 곱의 반복**: $W^q$가 $q$가 클 때 어떻게 행동하는지 — 고유값이 1보다 작으면 소실, 크면 폭발

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 직관적 설명 / Intuitive Explanation |
|---|---|
| **Memory Cell** | LSTM의 핵심 단위. 정보를 장기간 저장할 수 있는 "메모장"과 같음. 자기 순환 연결(CEC)을 통해 정보가 시간에 걸쳐 보존됨 / The core unit of LSTM. Like a "notepad" that can store information long-term. Information is preserved over time via self-recurrent connection (CEC) |
| **Constant Error Carrousel (CEC)** | 메모리 셀의 핵심: 가중치 1.0의 자기 순환 연결. 오류 신호가 이 "회전목마"를 통해 시간에 걸쳐 일정하게(소실/폭발 없이) 흐름 / The heart of the memory cell: self-recurrent connection with weight 1.0. Error signals flow through this "carrousel" constantly over time (no vanishing/exploding) |
| **Input Gate ($in_j$)** | 메모리 셀로의 정보 입력을 제어하는 "문지기". 열리면 새 정보 저장, 닫히면 기존 정보 보호 / "Gatekeeper" controlling information input to the memory cell. Open = store new info, closed = protect existing info |
| **Output Gate ($out_j$)** | 메모리 셀로부터의 정보 출력을 제어. 열리면 저장된 정보를 다른 유닛에 전달, 닫히면 메모리 내용을 숨김 / Controls information output from memory cell. Open = pass stored info to other units, closed = hide memory contents |
| **Cell State ($s_{c_j}$)** | 메모리 셀의 내부 상태. CEC를 통해 시간에 걸쳐 유지됨. $s_{c_j}(t) = s_{c_j}(t-1) + y^{in_j}(t) \cdot g(net_{c_j}(t))$ / The memory cell's internal state, maintained over time via CEC |
| **Gate Activation** | 게이트는 시그모이드 활성화를 사용 → 출력이 0~1. 0이면 완전히 닫힘, 1이면 완전히 열림 / Gates use sigmoid activation → output 0-1. 0 = fully closed, 1 = fully open |
| **Squashing Function $g$** | 메모리 셀 입력을 압축하는 함수. 논문에서는 범위 $[-2, 2]$의 시그모이드 사용 / Compresses memory cell input. Paper uses sigmoid with range $[-2, 2]$ |
| **Squashing Function $h$** | 메모리 셀 출력을 스케일링하는 함수. 범위 $[-1, 1]$. Cell state에서 출력으로의 변환 / Scales memory cell output. Range $[-1, 1]$. Transforms cell state to output |
| **Memory Cell Block** | 동일한 입력/출력 게이트를 공유하는 여러 메모리 셀의 그룹. 효율성을 위해 사용 / A group of memory cells sharing the same input/output gates. Used for efficiency |
| **Truncated Gradient** | LSTM의 효율적 학습을 위해 기울기를 특정 지점에서 절단. CEC 내부에서는 절단해도 장기 기울기가 보존됨 / Gradient truncated at specific points for efficient LSTM training. Long-term gradients preserved within CEC despite truncation |
| **Vanishing Gradient** | 기울기가 시간을 거슬러 전파될 때 기하급수적으로 0에 수렴하는 현상. $|f' \cdot w| < 1$의 반복 곱 / Gradient exponentially approaching 0 when propagated back through time. Repeated multiplication of $|f' \cdot w| < 1$ |

---

## 수식 미리보기 / Equations Preview

### 1. 기울기 소실의 수학적 원인 / Mathematical Cause of Vanishing Gradient

시간 $t$에서 시간 $t-q$로의 오류 역전파 스케일링 팩터:

The scaling factor for error backpropagation from time $t$ to time $t-q$:

$$\frac{\partial \vartheta_v(t-q)}{\partial \vartheta_u(t)} = \sum_{l_1=1}^{n} \cdots \sum_{l_{q-1}=1}^{n} \prod_{m=1}^{q} f'_{l_m}(net_{l_m}(t-m)) w_{l_m l_{m-1}}$$

**핵심**: 각 시간 스텝에서 $f'(net) \cdot w$가 곱해집니다. 시그모이드의 경우 $f'_{\max} = 0.25$이고 가중치가 4 미만이면, 이 곱은 1 미만 → $q$번 반복하면 기하급수적으로 0에 수렴합니다.

**Key**: At each time step, $f'(net) \cdot w$ is multiplied. For sigmoid, $f'_{\max} = 0.25$ and if weights < 4, this product < 1 → after $q$ repetitions, it exponentially approaches 0.

### 2. Constant Error Flow의 조건 / Condition for Constant Error Flow

단일 유닛 $j$의 자기 순환에서 오류가 일정하려면:

For constant error flow through a single unit $j$'s self-recurrence:

$$f'_j(net_j(t)) \cdot w_{jj} = 1.0$$

이를 해결하면 $f_j$는 **선형 함수**(항등 함수)여야 하고, $w_{jj} = 1.0$이어야 합니다. 이것이 CEC입니다.

Solving this requires $f_j$ to be a **linear function** (identity) and $w_{jj} = 1.0$. This is the CEC.

### 3. Memory Cell의 내부 상태 업데이트 / Memory Cell Internal State Update

$$s_{c_j}(t) = s_{c_j}(t-1) + y^{in_j}(t) \cdot g(net_{c_j}(t))$$

- $s_{c_j}(t-1)$: 이전 상태가 **그대로 유지** (CEC의 핵심!)
- $y^{in_j}(t)$: 입력 게이트 활성화 (0이면 새 입력 차단)
- $g(net_{c_j}(t))$: 새 입력을 압축한 값

- $s_{c_j}(t-1)$: Previous state **preserved as-is** (the essence of CEC!)
- $y^{in_j}(t)$: Input gate activation (0 = block new input)
- $g(net_{c_j}(t))$: Squashed new input value

### 4. Memory Cell 출력 / Memory Cell Output

$$y^{c_j}(t) = y^{out_j}(t) \cdot h(s_{c_j}(t))$$

출력 게이트 $y^{out_j}$가 메모리 내용 $h(s_{c_j})$의 출력 여부를 제어합니다.

Output gate $y^{out_j}$ controls whether memory contents $h(s_{c_j})$ are output.

### 5. 게이트 활성화 / Gate Activations

$$y^{in_j}(t) = f_{in_j}(net_{in_j}(t)), \quad y^{out_j}(t) = f_{out_j}(net_{out_j}(t))$$

여기서 $f_{in_j}$, $f_{out_j}$는 시그모이드 함수 (범위 $[0, 1]$):

Where $f_{in_j}$, $f_{out_j}$ are sigmoid functions (range $[0, 1]$):

$$net_{in_j}(t) = \sum_u w_{in_j u} y^u(t-1), \quad net_{out_j}(t) = \sum_u w_{out_j u} y^u(t-1)$$

게이트의 입력은 모든 유닛(입력, 다른 메모리 셀, 게이트)의 이전 출력의 가중합입니다.

Gate inputs are weighted sums of previous outputs from all units (input, other memory cells, gates).

### 6. 일반화 오류 상계 (SVM과의 연결) / Generalization Bound (Connection to SVM)

논문 Eq. 5의 SVM 상계와 대비: LSTM은 이론적 일반화 보장이 아닌, **경험적 성능**으로 검증됩니다. 1000 타임스텝 이상의 time lag에서도 학습 가능함을 실험으로 입증합니다.

In contrast to SVM's theoretical generalization bound (paper #8 Eq. 5), LSTM is validated through **empirical performance**, demonstrating learning capability over 1000+ time step lags.

---

## 논문 구조 미리보기 / Paper Structure Preview

| 섹션 / Section | 내용 / Content |
|---|---|
| 1. Introduction | RNN의 문제점 소개, LSTM의 아이디어 개요 |
| 2. Previous Work | BPTT, RTRL, time-delay nets, Mozer, Bengio 등 기존 접근의 한계 |
| 3. Constant Error Backprop | ★ 기울기 소실의 수학적 분석 (3.1) + 순진한 해결책과 한계 (3.2) |
| 4. Long Short-Term Memory | ★★ LSTM 아키텍처: Memory Cell, CEC, Input/Output Gates |
| 5. Experiments | 6가지 실험: Embedded Reber Grammar, Noise-free/Noisy sequences, 2-sequence, Adding, Multiplication, Temporal Order |
| 6. Discussion | 한계점과 장점 |
| Appendix A | 알고리즘 상세 (A.1) + 오류 흐름 수식 (A.2) |

---

## 읽기 팁 / Reading Tips

1. **Section 3.1이 가장 중요한 동기**: 기울기 소실이 왜 불가피한지 수학적으로 이해하면, LSTM의 설계가 왜 그런 형태인지 자연스럽게 이해됩니다. Eq. 2의 곱 $\prod f'_m \cdot w$이 핵심입니다.

2. **Section 3.2의 순진한 접근**: CEC만으로는 부족한 이유 (input/output weight conflict)를 이해하면, 게이트가 왜 필요한지 명확해집니다.

3. **Figure 1 (p.7)이 LSTM의 모든 것**: 이 다이어그램을 완전히 이해하면 논문의 핵심을 파악한 것입니다. 자기 순환 (1.0), g, h, 입력 게이트, 출력 게이트의 상호작용에 주목하세요.

4. **Table 2-3 (p.14-15)**: LSTM이 BPTT/RTRL 대비 얼마나 우월한지 보여주는 핵심 결과. 특히 p=100 (time lag 100)에서 BPTT/RTRL은 0% 성공률, LSTM은 100%입니다.

1. **Section 3.1 is the most important motivation**: Understanding mathematically why vanishing gradients are inevitable makes LSTM's design naturally understandable. The product $\prod f'_m \cdot w$ in Eq. 2 is the key.

2. **Section 3.2's naive approach**: Understanding why CEC alone is insufficient (input/output weight conflict) clarifies why gates are necessary.

3. **Figure 1 (p.7) is everything about LSTM**: Fully understanding this diagram means grasping the paper's core. Focus on the self-recurrence (1.0), g, h, and the interplay of input/output gates.

4. **Tables 2-3 (pp.14-15)**: Key results showing LSTM's superiority over BPTT/RTRL. At p=100 (time lag 100), BPTT/RTRL have 0% success rate while LSTM has 100%.
