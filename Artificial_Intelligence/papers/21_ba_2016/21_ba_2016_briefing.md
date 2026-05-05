---
title: "Pre-Reading Briefing: Layer Normalization"
paper_id: "21_ba_2016"
topic: Artificial_Intelligence
date: 2026-04-19
type: briefing
---

# Layer Normalization: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. arXiv:1607.06450.
**Author(s)**: Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey Hinton (University of Toronto)
**Year**: 2016

---

## 1. 핵심 기여 / Core Contribution

### 한국어
Layer Normalization(LN)은 Batch Normalization(#18)의 **배치 통계 의존성**이라는 근본적 한계를 해소하기 위해 제안된 정규화 기법입니다. BN은 미니배치 내 샘플들에 걸쳐 평균/분산을 계산하지만, LN은 **단일 샘플 내 모든 뉴런(hidden unit)에 걸쳐** 평균/분산을 계산합니다. 이 단순한 축 변경은 세 가지 중요한 결과를 가져옵니다: (1) 배치 크기에 완전히 독립적이어서 **RNN, online learning, batch size 1** 같은 상황에서도 작동하며, (2) 훈련/추론 시 동작이 완전히 동일해서 BN의 "running statistics" 관리가 불필요하고, (3) 시계열마다 길이가 다른 RNN의 각 time step에 자연스럽게 적용 가능합니다. LN은 시퀀스 모델(특히 Transformer와 현대 LLM)의 사실상 **기본 정규화**가 되었으며, LayerNorm 없이는 GPT·BERT·LLaMA 같은 모델이 학습되지 않습니다.

### English
Layer Normalization (LN) addresses a fundamental limitation of Batch Normalization (paper #18) — its dependence on batch statistics. Where BN averages across samples in a mini-batch, LN averages across all hidden units within a single sample. This simple axis change produces three crucial consequences: (1) complete independence from batch size, enabling **RNNs, online learning, and batch size 1**; (2) identical training and inference behavior, removing the need for BN's running statistics; (3) natural per-timestep application in RNNs with variable-length sequences. LN became the de facto normalization for sequence models — especially Transformers and modern LLMs. GPT, BERT, and LLaMA would not train without it.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
2015년 초 Ioffe & Szegedy의 Batch Normalization(#18)이 CNN 훈련을 14배 가속하며 딥러닝의 표준이 되었습니다. 그러나 **RNN, LSTM, 시퀀스 모델**에 BN을 적용하려 하면 여러 문제가 나타났습니다: (1) 시퀀스 길이가 샘플마다 달라 time step별 통계가 의미를 잃고, (2) 작은 배치에서는 통계가 불안정하며, (3) 훈련과 추론의 통계가 달라 distribution shift가 발생합니다. 몇몇 연구자들은 BN을 RNN에 적용하려 시도했지만(e.g., Laurent et al. 2016), 만족스럽지 못했습니다. 2016년 7월 Hinton 그룹의 이 논문은 **"차원 축을 반대로 돌리자"** 는 단순한 통찰로 이 문제를 해결했습니다.

**English**
In early 2015, Batch Normalization (paper #18) became the standard for CNN training, accelerating convergence by 14×. But applying BN to **RNNs, LSTMs, and sequence models** raised several problems: (1) variable sequence lengths per sample make per-timestep statistics meaningless; (2) small batches yield unstable statistics; (3) train-test statistic mismatch causes distribution shift. Attempts to BN-ify RNNs (e.g., Laurent et al. 2016) were unsatisfying. In July 2016, this paper from Hinton's group solved it with a single idea: **flip the normalization axis**.

### 타임라인 / Timeline

```
2010  Pascanu   — Exploding/vanishing gradients in RNN
2013  Pascanu   — Gradient clipping for RNN
2015  Ioffe     — Batch Normalization (CNN standard)           [#18]
2016  Laurent   — Batch Normalized RNN (partial success)
2016  Cooijmans — Recurrent Batch Normalization
2016  Ba        — Layer Normalization                          ★ THIS PAPER
2016  Salimans  — Weight Normalization (orthogonal alternative)
2016  Ulyanov   — Instance Normalization (style transfer)
2017  Vaswani   — Transformer (uses LayerNorm)                [#25]
2018  Wu        — Group Normalization (bridges LN and BN)
2019  GPT-2     — LayerNorm moved to pre-norm position
2020  Xiong     — Pre-LN vs Post-LN analysis
2019+ LN is universal in Transformers / LLMs (GPT, BERT, LLaMA, Claude)
2023  RMSNorm   — LayerNorm의 단순화 (LLaMA 채택)
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
- **Batch Normalization** (#18): 이 논문의 출발점이자 핵심 비교 대상.
- **RNN/LSTM** (#9): 시퀀스 모델의 구조와 hidden state의 개념.
- **Covariate shift**: Internal covariate shift 개념 (BN 논문에서 도입).
- **Mini-batch SGD**: 배치/샘플 통계의 차이를 이해하기 위해.
- **Invariance analysis**: 간단한 선형대수 — normalization이 어떤 변환에 불변(invariant)인가.
- **Online learning / streaming data**: LN이 해결하는 시나리오들.

### English
- **Batch Normalization (#18)**: the starting point and main comparison target.
- **RNN/LSTM (#9)**: sequence model architecture and hidden states.
- **Covariate shift**: the concept from the BN paper.
- **Mini-batch SGD**: to understand the distinction between batch and sample statistics.
- **Invariance analysis**: light linear algebra — which transformations leave the normalization unchanged.
- **Online learning / streaming data**: scenarios LN addresses.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Layer Normalization (LN)** | 단일 샘플 내 모든 hidden unit에 걸쳐 평균/분산을 계산해 정규화하는 기법. Normalizes across all hidden units within a sample. |
| **Batch Normalization (BN)** | 미니배치 내 샘플들에 걸쳐 평균/분산을 계산하는 기법 (비교 대상). Normalizes across samples in a batch. |
| **Weight Normalization (WN)** | 가중치 벡터의 방향과 크기를 분리해 재매개화하는 기법 (Salimans 2016). Reparameterizes weights into direction and magnitude. |
| **Summed input** $a^l_i$ | 활성화 함수 적용 **전**의 layer input: $a^l_i = \mathbf{w}^l_i{}^\top \mathbf{h}^{l-1}$. Pre-activation input to a neuron. |
| **Gain & bias** $g_i, b_i$ | 정규화 후 학습 가능한 scale/shift 파라미터 (BN의 $\gamma, \beta$에 해당). Learned affine parameters. |
| **Invariance** | 어떤 변환 하에서 결과가 바뀌지 않는 성질. LN은 입력의 **per-sample 스케일**에, BN은 입력의 **feature 스케일**에 불변. The property that the output is unchanged under a given transformation. |
| **Covariate shift** | 훈련/테스트 입력 분포의 변화. *Internal* covariate shift는 네트워크 내부 층 간 분포 변화. Change in input distribution between train/test (internal: between layers). |
| **Minibatch statistics** | 미니배치 내 샘플들로부터 계산된 평균/분산 (BN이 사용). Statistics computed over a mini-batch. |
| **Per-sample statistics** | 단일 샘플의 hidden unit들로부터 계산된 평균/분산 (LN이 사용). Statistics computed over units within one sample. |
| **Running statistics** | BN이 추론 시 사용하는 누적 평균/분산 — LN은 불필요. Running averages kept during BN training; not needed by LN. |
| **Riemannian metric** | 논문 §5의 분석 도구 — 파라미터 공간의 기하학적 구조. Geometric framework used in §5 to analyze parameter-space curvature. |
| **Fisher information matrix** | 확률 모델의 파라미터 공간에서의 자연 곡률 측정. 논문은 LN을 이 관점에서 정당화. Used in the paper's geometric justification of LN. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) Layer Normalization 통계 / LN statistics
단일 샘플의 한 층에서 hidden unit 수를 $H$라 하자. 모든 unit에 걸쳐 평균/분산을 계산:
$$
\mu^l = \frac{1}{H}\sum_{i=1}^{H} a^l_i, \qquad
\sigma^l = \sqrt{\frac{1}{H}\sum_{i=1}^{H}(a^l_i - \mu^l)^2}
$$

$\mu$와 $\sigma$는 **배치 차원이 없다**. 각 샘플마다 독립적으로 계산됨.
$\mu$ and $\sigma$ have **no batch dimension** — they are computed independently for each sample.

### (2) 정규화 및 affine 변환 / Normalization and affine transform
$$
\bar{a}^l_i = \frac{a^l_i - \mu^l}{\sigma^l} \cdot g_i + b_i
$$
$g_i, b_i$는 학습 가능한 scale/shift (BN의 $\gamma, \beta$).
$g_i, b_i$ are learned scale/shift parameters (like $\gamma, \beta$ in BN).

### (3) BN vs LN 비교 / BN vs LN comparison
**Batch Normalization**: for a specific unit $i$ across a mini-batch of size $N$,
$$
\mu^{\text{BN}}_i = \frac{1}{N}\sum_{n=1}^{N} a^{(n)}_i
$$
**Layer Normalization**: for a specific sample, across all $H$ units,
$$
\mu^{\text{LN}} = \frac{1}{H}\sum_{i=1}^{H} a_i
$$
Same formula, flipped axis — one averages across samples (batch dim), the other across units (feature dim).

### (4) RNN에 적용 / Applied to RNN
LSTM/RNN의 time step $t$에서 hidden state $\mathbf{h}_t$를 LN으로 정규화:
$$
\mathbf{a}_t = W_{hh} \mathbf{h}_{t-1} + W_{xh} \mathbf{x}_t
$$
$$
\mathbf{h}_t = f\!\left( \text{LN}(\mathbf{a}_t; \mathbf{g}, \mathbf{b}) \right)
$$
각 time step에 **동일한** LN 파라미터 $\mathbf{g}, \mathbf{b}$ 공유 — BN과 달리 time step별 통계를 따로 추적할 필요 없음.
Same LN parameters $\mathbf{g}, \mathbf{b}$ are shared across time steps — no need to track per-step statistics.

### (5) Invariance 성질 / Invariance properties (Table 1 요지)

| Method | 입력 weight matrix 재스케일 / rescaling | 입력 데이터 재스케일 / rescaling | 입력 weight 재shift |
|---|---|---|---|
| Batch norm | **invariant** (scale) | invariant (per-feature shift) | no |
| Weight norm | **invariant** (scale) | no | no |
| Layer norm | **invariant** (scale) | **invariant** (per-sample rescale) | **invariant** |

LN은 **input의 per-sample 스케일링과 shift에도 불변** — 입력 분포가 샘플마다 달라도 모델이 안정적.
LN is invariant to per-sample rescaling/shifting of the input — the model stays stable under sample-wise distribution differences.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **§1 Introduction**: BN의 RNN 적용 실패를 빠르게 복습. BN의 어떤 성질이 문제가 되는지 명확히 파악.
- **§2 Background**: BN의 수식을 한 번 더 복습. 이 논문은 BN과의 비교가 핵심이므로 여기가 바탕.
- **§3 Layer normalization**: **가장 중요**. 식 (1)~(3)의 단순한 축 변경을 정확히 이해. "왜 이게 통하는가"는 §5에서 설명되지만 §3에서 먼저 직관을 잡을 것.
- **§4 RNN에 적용**: LN이 빛나는 장면. LSTM 게이트에 LN을 어떻게 삽입하는지 수식으로 확인.
- **§5 Analysis**: **스킵해도 되는 이론 섹션**. 읽으면 Riemannian metric/Fisher matrix 관점에서 LN이 왜 좋은지 알 수 있지만 선형대수가 좀 필요.
- **§6 Experiments**: 특히 **Attentive Reader, RNN encoder-decoder, order embeddings, generative modeling** 등 5개 RNN 실험 결과. LN이 수렴 속도와 최종 성능을 모두 개선함.
- **§7 Conclusion**: 한 문단. 끝.

### English
- **§1 Introduction**: quick recap of BN's limitations for RNNs.
- **§2 Background**: BN equations recap — this paper is defined in contrast to BN.
- **§3 Layer normalization**: **most important**. Understand the axis-flip in Eqs. (1)–(3). The "why" comes in §5, but grasp intuition here first.
- **§4 Applied to RNN**: LN's killer app. Note how LN is inserted inside LSTM gates.
- **§5 Analysis**: **skippable theory**. Useful if you want the Riemannian/Fisher justification, but requires some linear algebra.
- **§6 Experiments**: five RNN-centric experiments — Attentive Reader, encoder-decoder, order embeddings, generative modeling, etc. LN improves both convergence and final accuracy.
- **§7 Conclusion**: one paragraph.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
LayerNorm은 2017년 Transformer의 등장과 함께 **NLP/시퀀스 모델의 표준 정규화**가 되었고, 2019년 GPT-2 이후 모든 대형 언어 모델의 표준 위치인 **pre-norm** (각 서브층 앞에 LN, skip 후에는 아무것도 없음 — 이 프로젝트의 ResNet v2 Q&A와 연결)으로 자리잡았습니다.

더 나아가:
- **Transformer 계열**: GPT, BERT, T5, LLaMA, Claude 등 모든 모델이 LayerNorm 사용.
- **Vision Transformer (ViT, 2020)**: CNN이 아닌 Transformer로 비전을 다루면서 BN 대신 LN을 씀.
- **RMSNorm (Zhang & Sennrich 2019)**: LN에서 평균 차감을 생략하고 RMS(root mean square)로만 정규화한 단순화 버전. LLaMA, GPT-NeoX 등이 채택.
- **Group Normalization (Wu & He 2018)**: LN과 BN을 일반화한 형태 — 채널을 그룹으로 묶어 정규화.
- **Instance Normalization (Ulyanov 2016)**: LN의 이웃 — 각 샘플의 각 채널별로 정규화 (style transfer에서).

요약하면, BN이 "CNN 시대의 정규화"였다면 **LN은 "Transformer 시대의 정규화"** 입니다. 현재 사용되는 거의 모든 대형 신경망의 정규화 레이어는 LN이나 그 변형이며, 이 논문이 없었다면 지금의 LLM 생태계는 존재할 수 없었습니다.

### English
LayerNorm became the **standard normalization for NLP / sequence models** after Transformer's rise in 2017, and since GPT-2 (2019) has settled into the **pre-norm** position (LN before each sub-layer, nothing after the skip — connecting to our ResNet v2 Q&A).

Beyond that:
- **Transformer family**: GPT, BERT, T5, LLaMA, Claude — all use LayerNorm.
- **Vision Transformer (ViT, 2020)**: replaces BN with LN when treating vision with Transformers.
- **RMSNorm (Zhang & Sennrich, 2019)**: simplified LN — drops mean subtraction, keeps only RMS scaling. Adopted by LLaMA, GPT-NeoX.
- **Group Normalization (Wu & He, 2018)**: generalizes LN and BN — normalizes over channel groups.
- **Instance Normalization (Ulyanov, 2016)**: LN's neighbor — per-channel per-sample normalization (style transfer).

In short: if BN was the normalization of the **CNN era**, LN is the normalization of the **Transformer era**. Nearly every large modern network uses LN or a variant; today's LLM ecosystem would not exist without this paper.

---

## Q&A

### Q1. 평균/분산을 구하는 단위가 "하나의 데이터 내부"라고 보면 되는가? / Is the statistics computed *within a single data sample*?

**정확히 맞습니다.** LayerNorm의 핵심은 바로 그것입니다 — **평균과 분산은 단일 샘플 내부에서만 계산**되며, 배치(다른 샘플들)와는 전혀 무관합니다. 이 한 가지가 LN과 BN을 가르는 근본적 차이이고, LN의 모든 장점이 여기서 파생됩니다.

Exactly right. The essence of LayerNorm is that **statistics are computed entirely within a single sample**, independent of the rest of the batch. This one axis distinction is what separates LN from BN, and every advantage of LN follows from it.

---

#### ① 구체적 예시: MLP에서 / Concrete example: MLP

배치 크기 $N=4$, hidden unit 수 $H=5$인 한 층의 pre-activation tensor를 보자:

Consider a layer with batch size $N=4$ and $H=5$ hidden units:

```
                  unit 1   unit 2   unit 3   unit 4   unit 5
              ┌────────────────────────────────────────────┐
sample 1      │  a₁₁      a₁₂      a₁₃      a₁₄      a₁₅    │  ← LN averages along THIS row
sample 2      │  a₂₁      a₂₂      a₂₃      a₂₄      a₂₅    │  ← LN averages along THIS row
sample 3      │  a₃₁      a₃₂      a₃₃      a₃₄      a₃₅    │  ← LN averages along THIS row
sample 4      │  a₄₁      a₄₂      a₄₃      a₄₄      a₄₅    │  ← LN averages along THIS row
              └────────────────────────────────────────────┘
                  ↑        ↑        ↑        ↑        ↑
               BN averages along EACH column
```

**BN**: 각 **열**(한 unit의 4개 샘플 값)을 평균 → **5개의 (μ, σ)** 를 얻음 (unit마다 하나씩).
**LN**: 각 **행**(한 샘플의 5개 unit 값)을 평균 → **4개의 (μ, σ)** 를 얻음 (샘플마다 하나씩).

**BN**: averages each **column** (the 4 samples of a unit) → **5 statistics** (one per unit).
**LN**: averages each **row** (the 5 units of a sample) → **4 statistics** (one per sample).

즉 BN과 LN은 **같은 행렬의 평균을 다른 축으로 계산**할 뿐입니다. 축 선택 하나가 모든 차이를 만듭니다.

BN and LN are literally "average over a different axis of the same tensor." That one axis choice drives every difference.

---

#### ② 왜 이 차이가 그렇게 중요한가 / Why this axis matters so much

**한국어**

| 관점 / Aspect | BN (배치 축) | LN (샘플 내부 축) |
|---|---|---|
| **배치 크기 의존** | 필요. 배치 크기 1이면 분산 = 0. | 없음. 배치 크기와 완전히 독립. |
| **샘플 간 상호작용** | 한 샘플의 정규화가 **다른 샘플 값에 의존** → 미묘한 정보 누출. | 없음. 각 샘플은 **자기 안에서만** 정규화됨. |
| **훈련/추론 일치** | 불일치. 추론 시 running mean/var 사용. | 완전 일치. 계산식이 동일. |
| **시퀀스 길이 가변성** | 길이 다르면 time step별 통계가 꼬임. | 각 time step에서 **자기 hidden state만 보면 됨**. |
| **distributed training** | 배치 통계 동기화 필요(cross-GPU). | 불필요. 각 GPU가 독립적으로 계산. |

**English**

| Aspect | BN (batch axis) | LN (within-sample axis) |
|---|---|---|
| **Batch size dependency** | required (batch 1 → variance 0) | none — fully batch-size-independent |
| **Cross-sample coupling** | a sample's output depends on **other samples' values** — subtle information leak | none — each sample normalizes using only its own units |
| **Train/inference parity** | mismatch (running stats at inference) | identical — same formula in both modes |
| **Variable sequence lengths** | per-step statistics break | each time step sees only its own hidden state |
| **Distributed training** | requires cross-GPU stat sync | none — each GPU computes independently |

---

#### ③ CNN에서는 어떻게 달라지는가 / What about CNNs?

**한국어**
CNN의 feature tensor는 shape $(N, C, H, W)$ — 배치, 채널, 공간 세로, 공간 가로.

- **BN**: $(N, H, W)$축으로 평균. 즉 **채널마다 하나의 통계** → $C$개.
- **LN**: $(C, H, W)$축으로 평균. 즉 **샘플마다 하나의 통계** → $N$개. (단일 데이터 내부의 모든 채널·공간 위치를 함께 평균)
- **Instance Norm**: $(H, W)$축으로만 평균 → $N \times C$개. (샘플별·채널별)
- **Group Norm**: 채널을 그룹으로 묶어 $(G, H, W)$축 평균 → $N \times (\text{num groups})$개.

이들은 모두 "어느 축으로 평균 내는가"의 차이일 뿐입니다.

**English**
For a CNN tensor of shape $(N, C, H, W)$:
- **BN**: average over $(N, H, W)$ → $C$ statistics (one per channel)
- **LN**: average over $(C, H, W)$ → $N$ statistics (one per sample, pooling *all* channels and spatial positions within that sample)
- **Instance Norm**: average over $(H, W)$ only → $N \times C$ statistics
- **Group Norm**: average over $(G, H, W)$ within channel groups → $N \times$ groups

They all differ only in **which axes are pooled**.

---

#### ④ RNN/Transformer에서 LN이 빛나는 이유 / Why LN shines in RNN/Transformer

**한국어**
Transformer의 한 토큰 시점에서 hidden state는 shape $(N, H)$ — 여기서 $H$는 d_model (예: 768, 4096).

- LN은 각 토큰마다 **자기 $H$차원 벡터 내부에서** 평균/분산을 구함.
- 토큰이 문장 길이 $T$개 있으면, 결과적으로 $N \times T$개의 통계가 생김 — **모두 독립적으로**.
- 배치 크기가 1이든 1000이든, 시퀀스가 길든 짧든, 정확히 같은 방식으로 작동.

이것이 **LLM의 inference가 batch=1로도 문제없고, training 시 sequence packing이 자유로운** 이유입니다. 만약 Transformer가 BN을 썼다면 시퀀스가 들쭉날쭉한 언어 데이터에서 학습 자체가 불가능했을 것입니다.

**English**
At a single token position in a Transformer, the hidden state has shape $(N, H)$ where $H$ is d_model (e.g., 768, 4096).
- LN normalizes each token **within its own $H$-dim vector**.
- With a sequence of $T$ tokens, you get $N \times T$ independent statistics.
- Works identically whether batch size is 1 or 1000, whether sequence is short or long.

This is why LLM inference works fine at batch=1, and why training with packed/variable-length sequences is trivial. A BN-based Transformer would have been impossible on ragged language data.

---

#### ⑤ 한 줄 요약 / One-line summary

**한국어**
✅ **"LN은 한 샘플의 hidden unit 값들만 보고 평균/분산을 구한다. 다른 샘플은 쳐다보지도 않는다."** — 이 한 문장이 LN의 전부이고, 시퀀스 모델과 LLM 시대가 가능해진 이유입니다.

**English**
✅ **"LN computes mean/variance using only the hidden units of a single sample — never looking at other samples."** That one sentence captures the entirety of LN, and it's why sequence models and modern LLMs exist.

---

### Q2. CNN에서 "샘플 내부"가 정확히 무엇인가? (RGB 4배치 예시) / What exactly is "within a sample" for a CNN? (RGB, batch 4)

**질문**: RGB 3채널 이미지가 batch size 4로 들어올 때, LN이 수행하는 계산은:
- (A) $3 \times 4 = 12$번 (샘플별·채널별)
- (B) $R, G, B$로 묶어서 3번
- (C) 그 외?

**답**: **(C) 4번** — 샘플마다 하나씩. 각 평균은 **해당 샘플의 R, G, B 3채널 × 모든 공간 위치 $(H \times W)$ 를 전부 하나로 pool**해서 계산합니다. R·G·B 채널이 서로 섞여 하나의 통계가 됩니다.

**Answer**: **(C) 4 computations** — one per sample. Each pools **all 3 channels (R, G, B) × all spatial positions $(H \times W)$** of that sample into a single bag. The three color channels are merged into one distribution.

---

#### 네 가지 정규화의 통계 개수 / Statistics count for four norm variants

입력 shape $(N, C, H, W) = (4, 3, H, W)$:

| 기법 / Method | 평균 내는 축 / Pool axes | 통계 개수 / # stats | 각 평균이 대상으로 하는 값 / Values per stat |
|---|---|---|---|
| **Batch Norm** | $(N, H, W)$ | $C = 3$ | $N \cdot H \cdot W$ values (채널별, 배치 전체) |
| **Layer Norm** ✓ | $(C, H, W)$ | $N = 4$ | $C \cdot H \cdot W$ values (한 샘플 전체) |
| **Instance Norm** | $(H, W)$ | $N \cdot C = 12$ | $H \cdot W$ values (샘플별·채널별) |
| **Group Norm ($G$그룹)** | $(C/G, H, W)$ | $N \cdot G$ | $(C/G) \cdot H \cdot W$ values |

사용자가 제시한 옵션 (A) 12번은 **Instance Norm**에 해당하고, (B) 3번은 **Batch Norm**에 해당합니다. 순수 LN은 (C) **4번**입니다.

Option (A) 12 = Instance Norm. Option (B) 3 = Batch Norm. Pure LN = (C) **4**.

---

#### 수식으로 / In equations

LN for a CNN:
$$
\mu_n = \frac{1}{C \cdot H \cdot W} \sum_{c=1}^{C} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{n,c,h,w}
$$
$$
\sigma_n^2 = \frac{1}{C \cdot H \cdot W} \sum_{c,h,w} (x_{n,c,h,w} - \mu_n)^2
$$
$$
\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_n}{\sqrt{\sigma_n^2 + \epsilon}} \cdot g_{c} + b_{c}
$$

- $\mu_n, \sigma_n$: **샘플 $n$마다 하나**. 네 개의 값.
- $g_c, b_c$: **채널마다** 학습 가능한 affine 파라미터 (구현에 따라 $g_{c,h,w}$까지 갈 수도).

$\mu_n, \sigma_n$ live **per sample** (four values). Affine params $g, b$ are learned per channel (or per unit).

---

#### ⚠️ 중요: 왜 CNN에서 LN이 잘 안 쓰이는가 / Important: why LN is rare in CNNs

**한국어**
원 논문 §6.7에서 저자들이 **직접 관찰**합니다: "CNN에 LN을 적용하면 BN보다 성능이 나빠진다."
이유는 위 공식이 설명해줍니다 — 자연 영상에서 R, G, B 채널의 값 분포는 **서로 매우 다릅니다** (예: 하늘 사진에서 B 평균 >> R 평균). 이들을 단일 $\mu_n$으로 섞어 정규화하면 각 채널 고유의 통계 구조가 **파괴**됩니다. 그래서 CNN에서는:
- Batch Norm (배치 통계가 충분할 때 — 분류)
- Group Norm (작은 배치 — detection/segmentation)
- **LN은 거의 쓰지 않음**

반면 **Transformer**에서는 feature 차원이 d_model 하나로 통일되어 있어(예: 4096개의 hidden unit), 이들 사이의 통계 구조가 비교적 균질 → pooling이 의미 있는 평균이 되고, LN이 자연스럽게 작동합니다.

**English**
The original paper §6.7 explicitly reports: "LN underperforms BN on CNNs." The formula above reveals why — R, G, B distributions in natural images are **very different** (e.g., sky photos have $\mu_B \gg \mu_R$). Merging them into a single $\mu_n$ destroys per-channel structure. So in practice:
- **BN** for CNNs with sufficient batch size (classification).
- **GN** for small-batch CNNs (detection/segmentation).
- **LN is rarely used on CNNs.**

In **Transformers**, the feature axis is a single d_model (e.g., 4096 units) whose statistics are comparatively homogeneous — pooling across it is meaningful, so LN thrives.

---

#### ViT의 LN은? / What about Vision Transformer's LN?

**한국어**
ViT가 LN을 쓰는 것을 보고 "그럼 CNN에도 LN이 되는 것 아닌가?" 오해할 수 있지만, ViT는 CNN이 아닙니다. ViT는 이미지를 패치로 잘라 각 패치를 **토큰 벡터(d_model 차원)** 로 만든 뒤 Transformer에 넣습니다. 이 때의 LN은:
- 각 **토큰**의 d_model 차원 벡터 **내부**에서 평균/분산 계산
- 즉 "한 패치 토큰의 d_model개 feature를 pool" — Transformer의 LN과 동일한 사용 방식
- "이미지 전체의 $C \times H \times W$를 pool"하는 것이 아님

**English**
ViT uses LN, but ViT isn't a CNN. ViT tokenizes an image into patches, each a d_model-dim vector, then runs a Transformer. Its LN pools over the **d_model axis of a single token** — identical to Transformer LN, *not* pooling over image $(C, H, W)$.

---

#### 한 문장 정리 / One-line summary

**"LN의 '샘플 내부' = 그 샘플의 모든 feature(채널 포함) 값을 하나의 집합으로 pool해서 평균"** — 따라서 RGB 4배치면 **4번** 계산합니다.

**"LN's 'within a sample' = pool every feature value (including channels) into one bag per sample"** — so RGB with batch 4 → **4 computations**.

