---
title: "Attention Is All You Need"
authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
year: 2017
journal: "Advances in Neural Information Processing Systems (NeurIPS) 30"
doi: "10.48550/arXiv.1706.03762"
topic: Artificial Intelligence / Deep Learning Architectures
tags: [transformer, attention, self-attention, seq2seq, machine-translation, deep-learning]
status: completed
date_started: 2026-04-22
date_completed: 2026-04-23
---

# 25. Attention Is All You Need / 어텐션만 있으면 충분하다

---

## 1. Core Contribution / 핵심 기여

### English

The paper introduces the **Transformer**, a sequence-transduction architecture that **eliminates recurrence and convolutions entirely** and relies solely on attention mechanisms. Prior state-of-the-art models for machine translation (ByteNet, ConvS2S, GNMT) all relied on sequential RNNs or local convolutions to model dependencies, which imposed $O(n)$ sequential computation and made long-range dependencies hard to learn. The Transformer replaces these with **multi-head self-attention**, giving any two tokens an $O(1)$ path length and enabling full parallelization across sequence positions. On WMT 2014 English–German translation it achieves **28.4 BLEU**, outperforming all prior single models (including ensembles) by more than 2 BLEU, while training in only **3.5 days on 8 P100 GPUs** — a fraction of the compute used by competing models. The paper also demonstrates the architecture generalizes beyond translation by applying it to English constituency parsing.

### 한국어

이 논문은 **Transformer**라는 sequence-transduction 아키텍처를 제안한다. 이 모델은 **recurrence와 convolution을 완전히 제거**하고 오직 attention 메커니즘에만 의존한다. 기존 SOTA 기계번역 모델(ByteNet, ConvS2S, GNMT)은 모두 순차 RNN 또는 local convolution에 의존했기에 $O(n)$의 순차 연산이 필요했고 장거리 의존성 학습이 어려웠다. Transformer는 이를 **multi-head self-attention**으로 대체하여 임의의 두 토큰 간 path length를 $O(1)$로 줄이고 시퀀스 전 위치를 완전 병렬화할 수 있다. WMT 2014 영-독 번역에서 **28.4 BLEU**를 기록하며 단일 모델은 물론 앙상블까지 2 BLEU 이상 앞섰고, 학습에는 8대의 P100 GPU로 **단 3.5일**만 소요되었다 — 경쟁 모델 연산량의 극히 일부이다. 본 논문은 또한 영어 구문 분석(constituency parsing)에 적용하여 아키텍처의 범용성도 입증했다.

---

## 2. Reading Notes / 읽기 노트

### Abstract

**English** — The dominant seq2seq models use encoder-decoders with complex RNNs or CNNs plus attention. The paper proposes a simpler architecture, the Transformer, based solely on attention. Experiments on WMT 2014 En→De (28.4 BLEU, +2 BLEU over prior best) and En→Fr (41.8 BLEU, new single-model SOTA) validate the approach, with training efficiency orders of magnitude better.

**한국어** — 지배적인 seq2seq 모델은 복잡한 RNN 또는 CNN 기반 encoder-decoder와 attention의 조합이었다. 저자들은 오직 attention만 사용하는 더 단순한 아키텍처 Transformer를 제안한다. WMT 2014 En→De(28.4 BLEU, 이전 최고 대비 +2 BLEU) 및 En→Fr(41.8 BLEU, 단일 모델 SOTA)에서 검증되었으며 학습 효율은 수십~수백 배 개선되었다.

---

### Section 1: Introduction / 서론

**English** — RNN-based seq2seq (Sutskever 2014, Cho 2014) factorizes $p(y|x) = \prod_t p(y_t | y_{<t}, x)$ by computing hidden states sequentially: $h_t = f(h_{t-1}, x_t)$. This **inherently sequential** computation prevents parallelization within a training example, becoming prohibitive at long sequence lengths due to memory constraints that prevent batching. Attention mechanisms (Bahdanau 2014, Kim 2017) help model long-range dependencies but are almost always combined with recurrent networks. The Transformer **dispenses with recurrence entirely**, relying on attention to draw global dependencies between input and output. It enables significantly more parallelization and reaches SOTA after just 12 hours of training on 8 GPUs.

**한국어** — RNN 기반 seq2seq(Sutskever 2014, Cho 2014)은 $p(y|x) = \prod_t p(y_t | y_{<t}, x)$를 순차적으로 은닉 상태를 계산해 분해한다: $h_t = f(h_{t-1}, x_t)$. 이 **본질적으로 순차적인** 연산은 한 학습 예제 내 병렬화를 막고, 긴 시퀀스에서는 메모리 제약으로 배치 크기까지 제한되어 치명적이 된다. Attention 메커니즘(Bahdanau 2014, Kim 2017)은 장거리 의존성 모델링에 도움이 되나 거의 항상 recurrent network와 결합되어 있었다. Transformer는 **recurrence를 완전히 제거**하고 오직 attention으로 입출력 전역 의존성을 포착한다. 덕분에 병렬화가 훨씬 가능해져 8-GPU에서 12시간 학습 후 SOTA에 도달한다.

---

### Section 2: Background / 배경

**English** — Extended Neural GPU, ByteNet, ConvS2S all try to reduce sequential computation by using convolution-based building blocks. The number of operations needed to relate signals at positions $i$ and $j$ grows with distance: **linearly for ConvS2S, logarithmically for ByteNet**. In the Transformer this reduces to a **constant number of operations**, albeit at the cost of effective resolution (since averaging attention-weighted positions reduces resolution), which is counteracted by **Multi-Head Attention**. Self-attention (intra-attention) has been used successfully for reading comprehension, summarization, and textual entailment. End-to-end memory networks use recurrent attention. The Transformer is the first transduction model relying **entirely on self-attention** without sequence-aligned RNNs or convolution.

**한국어** — Extended Neural GPU, ByteNet, ConvS2S는 모두 convolution 기반 블록으로 순차 연산을 줄이려 했다. 위치 $i, j$의 신호를 연결하는 데 필요한 연산 수는 거리에 따라 증가한다: **ConvS2S에서는 선형, ByteNet에서는 로그**. Transformer는 이를 **상수**로 줄이되, attention-가중 평균으로 인한 유효 해상도 감소의 대가를 치른다 — 이를 **Multi-Head Attention**으로 상쇄한다. Self-attention(intra-attention)은 독해, 요약, 함의(textual entailment) 등에서 성공적으로 사용되어 왔다. End-to-end memory network는 recurrent attention을 쓴다. Transformer는 sequence-aligned RNN이나 convolution 없이 **오직 self-attention에만 의존**한 최초의 transduction 모델이다.

---

### Section 3: Model Architecture / 모델 아키텍처

#### 3.1 Encoder and Decoder Stacks / 인코더와 디코더 스택

**English** — Both encoder and decoder are composed of $N=6$ identical layers.

**Encoder layer** has two sub-layers: (1) multi-head self-attention, (2) position-wise fully connected FFN. Each sub-layer is wrapped in a **residual connection** followed by **layer normalization**:
$$
\text{LayerNorm}(x + \text{Sublayer}(x))
$$
All sub-layers and embeddings produce outputs of dimension $d_{\text{model}}=512$ (required for residual connections).

**Decoder layer** has three sub-layers: (1) **masked** multi-head self-attention, (2) multi-head attention over encoder output (cross-attention), (3) position-wise FFN. The mask prevents positions from attending to subsequent positions, ensuring that predictions for position $t$ depend only on outputs at positions $< t$ — the **autoregressive property**.

**한국어** — Encoder와 decoder 모두 $N=6$개의 동일 layer로 구성된다.

**Encoder layer**는 두 sub-layer로 구성: (1) multi-head self-attention, (2) position-wise FFN. 각 sub-layer는 **residual connection**과 **layer normalization**으로 감싼다:
$$
\text{LayerNorm}(x + \text{Sublayer}(x))
$$
모든 sub-layer와 embedding은 $d_{\text{model}}=512$ 차원의 출력을 생산 (residual 연결을 위해 필수).

**Decoder layer**는 세 sub-layer로 구성: (1) **masked** multi-head self-attention, (2) encoder 출력에 대한 multi-head attention (cross-attention), (3) position-wise FFN. Mask는 위치 $t$의 예측이 $t$ 이전 위치 출력에만 의존하게 하여 **autoregressive 속성**을 보장한다.

---

#### 3.2 Attention / 어텐션

##### 3.2.1 Scaled Dot-Product Attention / 스케일 내적 어텐션

**English** — The core operation:
$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

- $Q \in \mathbb{R}^{n \times d_k}$: queries
- $K \in \mathbb{R}^{m \times d_k}$: keys
- $V \in \mathbb{R}^{m \times d_v}$: values

**Why $\sqrt{d_k}$?** For large $d_k$, dot products grow large in magnitude, pushing softmax into regions of extremely small gradients. Assume $q, k$ are independent with zero mean, unit variance; then $q \cdot k = \sum_i q_i k_i$ has variance $d_k$. Dividing by $\sqrt{d_k}$ keeps variance $O(1)$.

Two commonly used attentions are **additive** (Bahdanau 2014, using a feed-forward network) and **dot-product** (used here). Dot-product is much faster and more space-efficient in practice since it can be implemented using highly optimized matrix multiplication code.

**한국어** — 핵심 연산:
$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

**왜 $\sqrt{d_k}$인가?** $d_k$가 크면 내적 값의 크기가 커져 softmax가 gradient가 매우 작은 영역으로 포화된다. $q, k$가 평균 0, 분산 1의 독립 확률변수라 가정하면 $q \cdot k = \sum_i q_i k_i$의 분산은 $d_k$. $\sqrt{d_k}$로 나누면 분산이 $O(1)$로 유지된다.

Attention에는 **additive** (Bahdanau 2014, FFN 사용)와 **dot-product** (본 논문) 두 종류가 있다. Dot-product는 고도로 최적화된 행렬곱 코드를 쓸 수 있어 실제로 훨씬 빠르고 메모리 효율적이다.

---

##### 3.2.2 Multi-Head Attention / 멀티헤드 어텐션

**English** — Instead of one attention with $d_{\text{model}}$-dimensional Q, K, V, the paper projects them $h$ times with different learned projections to $d_k$, $d_k$, $d_v$ dimensions, performs attention in parallel, then concatenates:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

where $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.

Paper uses $h=8$, $d_k = d_v = d_{\text{model}}/h = 64$. Total computational cost is similar to single-head attention with full dimensionality, but allows the model to jointly attend to information from different representation subspaces at different positions.

**한국어** — $d_{\text{model}}$차원 Q, K, V로 한 번 attention 하는 대신, 서로 다른 학습된 투영으로 $h$번 $d_k, d_k, d_v$ 차원으로 투영하고 병렬로 attention 후 연결한다:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V), \quad \text{MultiHead} = \text{Concat}(\dots) W^O
$$

논문 설정: $h=8$, $d_k = d_v = d_{\text{model}}/h = 64$. 총 연산량은 전체 차원 단일 head attention과 유사하나, 모델이 **서로 다른 표현 부분공간(subspace)에서 다른 위치 정보에 동시에 주의**할 수 있게 된다.

---

##### 3.2.3 Attention in the Transformer / Transformer에서의 세 가지 Attention

**English** — The Transformer uses multi-head attention in three different ways:

1. **Encoder-Decoder (Cross) Attention**: Queries from previous decoder layer, keys/values from encoder output. Every decoder position attends over all encoder positions — the classic seq2seq attention pattern.
2. **Encoder Self-Attention**: All Q, K, V come from the same place (previous encoder layer). Each position can attend to all positions in the previous layer.
3. **Decoder Self-Attention (masked)**: Same, but with masking ($-\infty$ before softmax) to prevent leftward information flow, preserving the autoregressive property.

**한국어** — Transformer는 multi-head attention을 세 가지 방식으로 사용:

1. **Encoder-Decoder (Cross) Attention**: Query는 이전 decoder layer, Key/Value는 encoder 출력. 모든 decoder 위치가 encoder의 모든 위치에 주의 — 고전적 seq2seq attention.
2. **Encoder Self-Attention**: Q, K, V 모두 이전 encoder layer에서. 각 위치가 이전 layer의 모든 위치에 주의 가능.
3. **Decoder Self-Attention (masked)**: 같으나 masking ($-\infty$ before softmax)으로 좌측(미래) 정보 유입 차단, autoregressive 속성 보존.

---

#### 3.3 Position-wise Feed-Forward Network / 위치별 피드포워드 네트워크

**English** — Each layer contains:
$$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2
$$

Applied identically to each position (parameters shared across positions, different across layers). Equivalent to two 1×1 convolutions. Dimensions: $d_{\text{model}}=512$, inner $d_{ff}=2048$. While attention mixes information **across positions**, FFN mixes information **across features within a position**.

**한국어** — 각 layer에 포함:
$$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2
$$

모든 위치에 동일하게 적용 (파라미터는 위치 간 공유, layer 간 별개). 두 개의 1×1 convolution과 등가. 차원: $d_{\text{model}}=512$, 중간 $d_{ff}=2048$. Attention이 **위치 간** 정보를 섞는다면, FFN은 한 **위치 내 feature 간** 정보를 섞는다.

---

#### 3.4 Embeddings and Softmax / 임베딩과 소프트맥스

**English** — Input and output embeddings share the **same weight matrix**, which is also tied to the pre-softmax linear transformation (Press & Wolf 2016, Inan et al. 2016). In the embedding layers, weights are multiplied by $\sqrt{d_{\text{model}}}$ to match the scale of positional encodings.

**한국어** — Input과 output embedding은 **동일한 가중치 행렬**을 공유하며, pre-softmax linear 변환과도 tied되어 있다 (weight tying). Embedding layer에서는 가중치에 $\sqrt{d_{\text{model}}}$을 곱해 positional encoding과 스케일을 맞춘다.

---

#### 3.5 Positional Encoding / 위치 인코딩

**English** — Since there is no recurrence or convolution, the model has no notion of sequence order. Positional encodings (same dim as embeddings) are **added** to the input embeddings. Paper uses sinusoidal:
$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})
$$
$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})
$$

Wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$. Rationale: allows the model to learn to attend by **relative positions** — for any fixed offset $k$, $PE_{pos+k}$ is a linear function of $PE_{pos}$ (specifically, a rotation). Sinusoidal was chosen over learned PE because it may extrapolate to sequence lengths longer than those seen in training.

**한국어** — Recurrence와 convolution이 없으므로 모델에 순서 개념이 없다. Embedding과 같은 차원의 positional encoding을 input embedding에 **더한다**. 논문은 sinusoidal을 사용:
$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}}), \quad PE_{(pos, 2i+1)} = \cos(\dots)
$$

파장은 $2\pi$에서 $10000 \cdot 2\pi$까지 기하급수. 근거: 고정 offset $k$에 대해 $PE_{pos+k}$가 $PE_{pos}$의 선형 함수(회전)이므로 **상대 위치** 기반 attention 학습 용이. 학습된 PE보다 sinusoidal을 선택한 이유는 학습 시 본 적 없는 긴 시퀀스로 extrapolation 가능성.

---

### Section 4: Why Self-Attention / 왜 Self-Attention인가

**English** — Three desiderata motivate using self-attention:

1. **Computational complexity per layer**
2. **Parallelizable computation** (minimum sequential operations required)
3. **Path length between long-range dependencies** — shorter path length makes learning dependencies easier

| Layer Type | Complexity per Layer | Sequential Ops | Max Path Length |
|---|---|---|---|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

Self-attention is faster than recurrent when $n < d$ (common in machine translation — sentence length < representation dim). For very long sequences, "restricted self-attention" limits each query to a neighborhood of size $r$. An **interpretability bonus**: attention heads are visualizable and appear to learn distinct linguistic roles (syntactic structure, coreference, etc. — see appendix figures).

**한국어** — Self-attention을 쓰는 세 가지 이유:

1. **layer당 연산 복잡도**
2. **병렬화 가능한 연산량** (필수 순차 연산의 최소)
3. **장거리 의존성의 path length** — 짧을수록 학습 용이

Self-attention은 $n < d$일 때 recurrent보다 빠르다 (기계번역에서는 보통 문장 길이 < 표현 차원). 매우 긴 시퀀스에는 이웃 $r$개로 제한하는 restricted self-attention 사용. **해석 가능성 보너스**: attention head가 시각화 가능하며 구문 구조, coreference 등 서로 다른 언어학적 역할을 학습하는 것으로 보인다(부록 그림).

---

### Section 5: Training / 학습

**English** — 

- **Data**: WMT 2014 En-De (4.5M sentence pairs, byte-pair encoding, 37K shared vocab), En-Fr (36M sentences, 32K word-piece vocab). Sentences batched by approximate length, ~25K source + 25K target tokens per batch.
- **Hardware**: 8 NVIDIA P100 GPUs. Base model: 0.4 sec/step, 100K steps (12 hours). Big model: 1.0 sec/step, 300K steps (3.5 days).
- **Optimizer**: Adam with $\beta_1=0.9$, $\beta_2=0.98$, $\epsilon=10^{-9}$. Learning rate schedule:
$$
lr = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5})
$$
with $\text{warmup\_steps}=4000$. Increases linearly during warmup, then decreases as inverse square root.
- **Regularization**: (1) Residual dropout $P_{\text{drop}}=0.1$ applied to each sub-layer output before addition, and to embedding+PE sums. (2) **Label smoothing** $\epsilon_{ls}=0.1$ — hurts perplexity but improves accuracy and BLEU.

**한국어** —

- **데이터**: WMT 2014 En-De (4.5M 문장쌍, BPE, 37K 공유 vocab), En-Fr (36M 문장, 32K word-piece vocab). 유사 길이끼리 배치, batch당 ~25K source + 25K target token.
- **하드웨어**: 8대 NVIDIA P100. Base 모델: step당 0.4초, 100K step(12시간). Big 모델: step당 1.0초, 300K step(3.5일).
- **Optimizer**: Adam ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$). Learning rate schedule은 warmup 동안 선형 증가 후 step의 역제곱근으로 감소.
- **Regularization**: (1) Residual dropout $P_{\text{drop}}=0.1$, (2) **Label smoothing** $\epsilon_{ls}=0.1$ — perplexity는 악화되나 accuracy와 BLEU는 향상.

---

### Section 6: Results / 결과

**English** —

**Machine Translation (Table 2):**
- En→De: Transformer (big) = **28.4 BLEU** (new SOTA, +2 over prior best ensembles). Base model also exceeds all prior published single models at a fraction of training cost.
- En→Fr: Transformer (big) = **41.8 BLEU** with 1/4 the training cost of prior SOTA.

**Model Variations (Table 3):** Ablation study on English-German development set (newstest2013):
- (A) Head count $h$: varies performance; single head is 0.9 BLEU worse than 8-head.
- (B) Key dimension $d_k$: reducing hurts; suggests compatibility function is not trivial (dot product may benefit from richer capacity).
- (C) Model size ($d_{\text{model}}, d_{ff}, N$): bigger is better.
- (D) Dropout: very helpful.
- (E) Sinusoidal vs learned positional embedding: nearly identical results.

**English Constituency Parsing (Table 4):** Demonstrates Transformer generalizes — achieves 91.3 F1 (WSJ only) and 92.7 F1 (semi-supervised), outperforming most prior approaches and competitive with RNN-based grammars.

**한국어** —

**기계번역 (Table 2):**
- En→De: Transformer (big) = **28.4 BLEU** (새 SOTA, 이전 앙상블 대비 +2 BLEU). Base 모델도 기존 모든 단일 모델을 훨씬 적은 학습 비용으로 능가.
- En→Fr: Transformer (big) = **41.8 BLEU**, 기존 SOTA 학습 비용의 1/4.

**모델 변형 (Table 3):** newstest2013에서 ablation:
- (A) Head 수 $h$: single head는 8-head 대비 0.9 BLEU 낮음.
- (B) Key 차원 $d_k$: 줄이면 악화.
- (C) 모델 크기: 클수록 좋음.
- (D) Dropout: 매우 유효.
- (E) Sinusoidal vs 학습된 PE: 거의 동일.

**영어 구문 분석 (Table 4):** Transformer가 일반화됨을 입증 — 91.3 F1 (WSJ only), 92.7 F1 (semi-supervised), 기존 대부분 기법을 능가.

---

### Section 7: Conclusion / 결론

**English** — First sequence transduction model based entirely on attention, replacing recurrence in encoder-decoder with multi-head self-attention. Trains significantly faster than RNN/CNN-based models and achieves SOTA on WMT translation. Future work: extend to other modalities (images, audio, video); investigate restricted local attention for large inputs/outputs; make generation less sequential.

**한국어** — Attention에만 전적으로 의존하는 최초의 sequence transduction 모델로 encoder-decoder의 recurrence를 multi-head self-attention으로 대체. RNN/CNN 기반보다 훨씬 빠르게 학습하며 WMT에서 SOTA 달성. 향후 과제: 다른 modality(이미지, 오디오, 비디오)로 확장, 큰 입출력을 위한 restricted local attention 연구, 생성 과정의 순차성 완화.

---

## 3. Key Takeaways / 핵심 시사점

1. **Recurrence is not essential for sequence modeling** — 시퀀스 모델링에 recurrence는 필수가 아니다. The paper overturns a decade of seq2seq orthodoxy by showing that attention alone can model sequential dependencies. RNN의 순차 의존성을 $O(1)$ path length의 self-attention으로 대체 가능함을 입증했다. 이로 인해 딥러닝 전체의 설계 공간이 재편된다.

2. **Parallelization dramatically accelerates training** — 병렬화가 학습을 극적으로 가속한다. Removing the $O(n)$ sequential bottleneck of RNNs unlocks full GPU parallelism, allowing training that took weeks to finish in hours. RNN의 $O(n)$ 순차 병목을 제거하여 GPU를 완전 활용, 주 단위 학습이 시간 단위로 단축된다. 이 특성은 훗날 대규모 모델(GPT, BERT) 학습의 토대가 된다.

3. **Multi-head attention is cheap but expressive** — 멀티헤드 어텐션은 저렴하지만 표현력이 강하다. Splitting dimensions into $h$ heads keeps total compute constant while letting the model attend to different representation subspaces in parallel. 차원을 $h$개 head로 나누면 총 연산량은 유지하면서도 서로 다른 표현 부분공간에 병렬로 주의할 수 있다. 실험적으로 각 head가 구문 구조, coreference 등 서로 다른 언어 관계를 학습함이 관찰된다.

4. **Scaling by $\sqrt{d_k}$ is a necessary engineering fix** — $\sqrt{d_k}$ 스케일링은 필수적인 공학적 보정이다. Without it, large $d_k$ causes softmax saturation and gradient collapse; the fix is mathematically principled (variance control). 없으면 $d_k$가 클 때 softmax 포화로 gradient가 소실된다. 분산 제어라는 수학적 원리에 기반한 해결책이다.

5. **Sinusoidal positional encoding enables relative-position learning** — Sinusoidal PE가 상대 위치 학습을 가능케 한다. Because $PE_{pos+k}$ is a linear (rotational) function of $PE_{pos}$, attention can learn distance-based relations independent of absolute position. 고정 offset $k$에 대해 $PE_{pos+k}$가 $PE_{pos}$의 회전으로 표현되므로 절대 위치 무관하게 거리 기반 관계를 학습 가능. RoPE 등 현대 변형의 직접적 뿌리다.

6. **Residual + LayerNorm makes deep attention stacks trainable** — Residual과 LayerNorm이 깊은 attention 스택을 학습 가능하게 한다. ResNet의 핵심 통찰이 attention 영역에 이식되어 6층, 이후 수십~수백 층 스택이 안정적으로 학습된다. Without these, optimizing a stack of 6 attention+FFN blocks would be unstable. 이 조합은 이후 대규모 모델 스케일링의 안정성 기반이다.

7. **Label smoothing and dropout are critical** — Label smoothing과 dropout이 중요하다. Label smoothing ($\epsilon_{ls}=0.1$) hurts perplexity but improves BLEU — a classic generalization trade-off. Label smoothing은 perplexity를 악화시키지만 BLEU를 개선한다 — 정확성과 일반화의 고전적 trade-off. Dropout 0.1이 attention과 embedding 양쪽에 적용되어 과적합을 크게 억제한다.

8. **The architecture is general, not translation-specific** — 이 아키텍처는 번역 전용이 아닌 범용이다. The constituency parsing experiment hints at the Transformer's generality; within 1–3 years this generality is fully realized in BERT (understanding), GPT (generation), and ViT (vision). 구문 분석 실험은 Transformer의 범용성을 암시하며, 1-3년 내 BERT(이해), GPT(생성), ViT(비전)로 완전히 실현된다. 단일 아키텍처가 언어, 비전, 음성, 멀티모달로 확장되는 현대 foundation model의 출발점.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Core Attention / 핵심 어텐션

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

### 4.2 Multi-Head / 멀티헤드

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

Parameters / 파라미터: $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.

### 4.3 Position-wise FFN

$$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2
$$

$W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$, applied per-position.

### 4.4 Sub-layer Wrapping / 서브레이어 포장

$$
y = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

### 4.5 Positional Encoding / 위치 인코딩

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

### 4.6 Learning Rate Schedule / 학습률 스케줄

$$
lr = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5})
$$

### 4.7 Complexity Comparison / 복잡도 비교

| Layer | Complexity | Sequential | Path Length |
|---|---|---|---|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| Restricted Self-Attention | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

### 4.8 Hyperparameters (Base / Big) / 하이퍼파라미터

| | Base | Big |
|---|---|---|
| $N$ (layers) | 6 | 6 |
| $d_{\text{model}}$ | 512 | 1024 |
| $d_{ff}$ | 2048 | 4096 |
| $h$ (heads) | 8 | 16 |
| $d_k, d_v$ | 64 | 64 |
| $P_{\text{drop}}$ | 0.1 | 0.3 |
| $\epsilon_{ls}$ | 0.1 | 0.1 |
| Params | 65M | 213M |

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1986 ── Rumelhart, Hinton, Williams: Backpropagation
          │
1997 ── Hochreiter & Schmidhuber: LSTM
          │  (long-range dependency via gating)
          │
2013 ── Mikolov: Word2Vec
          │  (distributed word representations)
          │
2014 ── Sutskever: Seq2Seq with RNN encoder-decoder
2014 ── Bahdanau: Attention mechanism (additive) ⭐
          │  (attention as add-on to RNN)
          │
2015 ── Luong: Dot-product attention variants
2016 ── Wu et al.: GNMT (Google Neural Machine Translation)
2016 ── ByteNet, ConvS2S: CNN-based seq2seq
          │
━━━━━ 2017 ── Vaswani et al.: Transformer ⭐⭐⭐ (본 논문) ━━━━━
          │  (attention alone, no recurrence/convolution)
          │
2018 ── Devlin: BERT (encoder-only) ──┐
2018 ── Radford: GPT (decoder-only)  ─┤  Transformer 계열 폭발
2019 ── Raffel: T5 (encoder-decoder) ─┘
          │
2020 ── Dosovitskiy: ViT (비전에 Transformer 적용)
2020 ── Brown: GPT-3 (175B 파라미터 — Transformer scaling)
          │
2022 ── ChatGPT 공개 (Transformer 기반)
2023+ ── Llama, GPT-4, Gemini, Claude 등 — 전 분야 Transformer 기반
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Bahdanau et al. 2014 (#17) | **Attention origin** — Transformer의 cross-attention은 Bahdanau의 additive attention을 dot-product 형태로 일반화 | 직접적 선행 연구 / Direct predecessor |
| Hochreiter & Schmidhuber 1997 (#9) | **LSTM** — Transformer가 대체하려는 recurrent 모델의 원형; 장거리 의존성 문제의 gating 기반 해법 | 대체 대상 / Replaces |
| Ba, Kiros, Hinton 2016 (#21) | **Layer Normalization** — Transformer의 각 sub-layer 안정화에 필수; BatchNorm이 sequential model에 부적합한 문제 해결 | 직접 사용 / Directly used |
| He et al. 2015 (#20) | **ResNet** — Residual connection을 Transformer가 계승 (LayerNorm(x + Sublayer(x)) 형태) | 건축적 기반 / Architectural foundation |
| Mikolov et al. 2013 (#14) | **Word2Vec** — Token embedding의 distributed representation 개념 제공 | 표현 기반 / Representation basis |
| Kipf & Welling 2017 (#26) | **GCN** — 같은 해에 나온 self-attention의 그래프 버전 해석 가능; 이후 attention과 GNN의 관계 연구 활발 | 동시대·관련 / Contemporary related |
| Devlin et al. 2018 (#28) | **BERT** — Transformer encoder-only, pre-training으로 확장. Transformer의 직접적 후속작 | 직접 후속 / Direct successor |
| Dosovitskiy et al. 2020 (#31) | **ViT** — Transformer를 비전에 적용. 이미지 patch를 토큰처럼 취급 | 모달리티 확장 / Modality extension |

---

## 7. References / 참고문헌

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). "Attention Is All You Need". *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate". *ICLR 2015*. [arXiv:1409.0473]
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). "Sequence to Sequence Learning with Neural Networks". *NeurIPS 2014*. [arXiv:1409.3215]
- Cho, K., et al. (2014). "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation". *EMNLP 2014*. [arXiv:1406.1078]
- Luong, M.-T., Pham, H., & Manning, C. D. (2015). "Effective Approaches to Attention-based Neural Machine Translation". *EMNLP 2015*. [arXiv:1508.04025]
- Wu, Y., et al. (2016). "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation". [arXiv:1609.08144]
- Gehring, J., et al. (2017). "Convolutional Sequence to Sequence Learning" (ConvS2S). *ICML 2017*. [arXiv:1705.03122]
- Kalchbrenner, N., et al. (2016). "Neural Machine Translation in Linear Time" (ByteNet). [arXiv:1610.10099]
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). "Layer Normalization". [arXiv:1607.06450]
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition". *CVPR 2016*. [arXiv:1512.03385]
- Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization". [arXiv:1412.6980]
- Press, O., & Wolf, L. (2017). "Using the Output Embedding to Improve Language Models". *EACL 2017*. [arXiv:1608.05859]
- Szegedy, C., et al. (2016). "Rethinking the Inception Architecture for Computer Vision" (label smoothing). *CVPR 2016*. [arXiv:1512.00567]
