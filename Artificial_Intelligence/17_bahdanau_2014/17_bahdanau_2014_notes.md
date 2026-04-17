---
title: "Neural Machine Translation by Jointly Learning to Align and Translate"
authors: Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio
year: 2014
journal: "ICLR 2015"
doi: "arXiv:1409.0473"
topic: Artificial_Intelligence
tags: [attention, sequence-to-sequence, neural-machine-translation, encoder-decoder, alignment, RNN, GRU, BiRNN]
status: completed
date_started: 2026-04-17
date_completed: 2026-04-17
---

# 17. Neural Machine Translation by Jointly Learning to Align and Translate / 정렬과 번역을 동시에 학습하는 신경 기계 번역

---

## 1. Core Contribution / 핵심 기여

이 논문은 sequence-to-sequence 신경 기계 번역에 **attention mechanism**을 최초로 도입했습니다. 기존의 encoder-decoder 아키텍처는 가변 길이의 입력 문장을 하나의 **고정 길이 벡터(fixed-length vector)**로 압축하고, 이 벡터에서 번역을 디코딩했습니다. 저자들은 이 고정 길이 벡터가 성능의 **병목(bottleneck)**이라고 가설을 세우고, 이를 검증했습니다. 특히 Cho et al. (2014b)의 연구에서 문장 길이가 길어질수록 기본 encoder-decoder의 성능이 급격히 하락한다는 실증적 증거가 이미 있었습니다. 이에 대한 해결책으로, 디코더가 각 타깃 단어를 생성할 때마다 소스 문장의 **관련 부분에 동적으로 집중(soft-search)**하는 메커니즘을 제안했습니다. 이 접근법은 Bidirectional RNN 인코더로 소스 문장의 각 위치에 대한 annotation을 생성하고, alignment model이 디코더 상태와 각 annotation 사이의 관련성을 점수화하며, softmax로 정규화된 attention weight를 통해 context vector를 매 디코딩 스텝마다 동적으로 계산합니다. 결과적으로 영어-프랑스어 번역 과제에서 기존 RNN Encoder-Decoder(RNNencdec)를 BLEU 점수 기준 대폭 능가하고, 당시 최첨단 phrase-based SMT 시스템(Moses)에 필적하는 성능을 달성했습니다.

This paper introduced the **attention mechanism** to sequence-to-sequence neural machine translation. The conventional encoder-decoder architecture compressed variable-length input sentences into a single **fixed-length vector** from which translations were decoded. The authors hypothesized — and empirically confirmed — that this fixed-length vector is a performance **bottleneck**, particularly for long sentences (as previously shown by Cho et al., 2014b, where basic encoder-decoder performance degraded rapidly with increasing sentence length). As a solution, they proposed a mechanism that allows the decoder to dynamically **(soft-)search** for relevant parts of the source sentence when generating each target word. The approach uses a Bidirectional RNN encoder to generate annotations for each source position, an alignment model to score relevance between decoder state and each annotation, and softmax-normalized attention weights to dynamically compute a context vector at every decoding step. On the English-to-French translation task, this approach (RNNsearch) significantly outperformed the conventional RNN Encoder-Decoder (RNNencdec) in BLEU score and achieved performance comparable to the state-of-the-art phrase-based SMT system (Moses).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (Section 1) / 서론

저자들은 NMT의 핵심 문제를 제기합니다: 기존 encoder-decoder 모델은 소스 문장 전체를 고정 길이 벡터 하나로 압축해야 하므로, 긴 문장에서 정보 손실이 발생합니다. Cho et al. (2014b)의 선행 연구에서 문장 길이가 증가하면 RNN encoder-decoder의 성능이 급격히 떨어진다는 실증 결과가 이미 보고되었습니다.

The authors identify the core problem: existing encoder-decoder models must compress entire source sentences into a single fixed-length vector, causing information loss for long sentences. Prior work by Cho et al. (2014b) had already shown empirically that RNN encoder-decoder performance deteriorates rapidly as input sentence length increases.

제안하는 해결책의 핵심 특징: 입력 문장을 고정 길이 벡터가 아닌 **벡터의 시퀀스**로 인코딩하고, 디코딩 시 이 벡터들의 **부분 집합(subset)을 적응적으로 선택**합니다.

The key feature of the proposed solution: encode the input sentence as a **sequence of vectors** rather than a fixed-length vector, and **adaptively choose a subset** of these vectors during decoding.

### Part II: Background — Neural Machine Translation (Section 2) / 배경 — 신경 기계 번역

번역은 확률적 관점에서 소스 문장 $\mathbf{x}$가 주어졌을 때 타깃 문장 $\mathbf{y}$의 조건부 확률을 최대화하는 문제입니다:

From a probabilistic perspective, translation is the problem of finding the target sentence $\mathbf{y}$ that maximizes the conditional probability given source sentence $\mathbf{x}$:

$$\arg\max_{\mathbf{y}} p(\mathbf{y} \mid \mathbf{x})$$

#### 2.1 RNN Encoder-Decoder (기존 모델 / Baseline Model)

인코더: 입력 시퀀스 $\mathbf{x} = (x_1, \cdots, x_{T_x})$를 순차적으로 읽어 hidden state를 갱신하고, 최종적으로 context vector $c$를 생성합니다:

The encoder reads input sequence $\mathbf{x} = (x_1, \cdots, x_{T_x})$ sequentially, updating hidden states, and ultimately produces context vector $c$:

$$h_t = f(x_t, h_{t-1}), \quad c = q(\{h_1, \cdots, h_{T_x}\})$$

Sutskever et al. (2014)의 경우 LSTM을 $f$로, $q(\{h_1, \cdots, h_T\}) = h_T$ (마지막 hidden state)를 $c$로 사용했습니다.

In Sutskever et al. (2014), $f$ was an LSTM and $c$ was simply the last hidden state $q(\{h_1, \cdots, h_T\}) = h_T$.

디코더: context vector $c$와 이전 출력들에 기반하여 다음 단어의 확률을 순차적으로 계산합니다:

The decoder sequentially computes the probability of the next word based on $c$ and previous outputs:

$$p(\mathbf{y}) = \prod_{t=1}^{T} p(y_t \mid \{y_1, \cdots, y_{t-1}\}, c)$$

$$p(y_t \mid \{y_1, \cdots, y_{t-1}\}, c) = g(y_{t-1}, s_t, c) \quad \text{(Eq. 3)}$$

여기서 $g$는 비선형 다층 함수이고, $s_t$는 디코더의 hidden state입니다. **핵심 한계**: 모든 디코딩 스텝에서 **동일한 $c$**를 사용합니다.

Here $g$ is a nonlinear multi-layered function and $s_t$ is the decoder's hidden state. **Key limitation**: the **same $c$** is used at every decoding step.

### Part III: Learning to Align and Translate (Section 3) / 정렬과 번역의 동시 학습

이 섹션이 논문의 핵심입니다. 새로운 아키텍처는 두 가지 구성 요소로 이루어져 있습니다: (1) Bidirectional RNN 인코더, (2) 소스 문장을 탐색(search)하면서 디코딩하는 디코더.

This is the core section. The new architecture has two components: (1) a Bidirectional RNN encoder, and (2) a decoder that searches through the source sentence while decoding.

#### 3.1 Decoder with Attention / Attention이 적용된 디코더

기존 Eq. 3과의 핵심 차이점: 고정된 $c$ 대신 **각 타깃 단어 $y_i$마다 고유한 context vector $c_i$**를 사용합니다:

Key difference from Eq. 3: instead of fixed $c$, each target word $y_i$ uses its **own context vector $c_i$**:

$$p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x}) = g(y_{i-1}, s_i, c_i) \quad \text{(Eq. 4)}$$

$$s_i = f(s_{i-1}, y_{i-1}, c_i)$$

**Context vector 계산 / Context vector computation:**

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j \quad \text{(Eq. 5)}$$

**Attention weight 계산 / Attention weight computation:**

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})} \quad \text{(Eq. 6)}$$

$$e_{ij} = a(s_{i-1}, h_j)$$

$a$는 **alignment model**로, 이전 디코더 상태 $s_{i-1}$과 인코더 annotation $h_j$ 사이의 관련성을 점수화하는 feedforward neural network입니다. 전체 모델과 jointly 학습됩니다.

$a$ is the **alignment model**, a feedforward neural network scoring the relevance between previous decoder state $s_{i-1}$ and encoder annotation $h_j$. It is jointly trained with the entire model.

**직관적 해석 / Intuitive interpretation:** $\alpha_{ij}$는 타깃 단어 $y_i$가 소스 단어 $x_j$에서 "번역될" 확률로 이해할 수 있습니다. $c_i$는 이 확률들에 의한 annotation의 기댓값(expected annotation)입니다. 이것은 **soft alignment** 또는 **attention**이라 불립니다.

$\alpha_{ij}$ can be interpreted as the probability that target word $y_i$ is "translated from" source word $x_j$. $c_i$ is the expected annotation under these probabilities. This is called **soft alignment** or **attention**.

중요한 점: 전통적 기계 번역과 달리 alignment가 **latent variable이 아닙니다**. Alignment model이 직접 soft alignment을 계산하므로 gradient가 흐를 수 있고, cost function의 gradient를 통해 alignment model과 전체 번역 모델을 jointly 학습할 수 있습니다.

Importantly, unlike traditional MT, alignment is **not a latent variable**. The alignment model directly computes soft alignment, allowing gradients to flow through. This enables joint training of the alignment model and the entire translation model via backpropagation.

#### 3.2 Encoder: Bidirectional RNN / 인코더: 양방향 RNN

기존 RNN은 $x_1$부터 $x_{T_x}$까지 순방향으로만 읽으므로, annotation $h_j$는 주로 $x_j$ 이전의 단어들에 대한 정보만 담습니다. 이 논문은 **Bidirectional RNN (BiRNN)**을 사용하여 양쪽 맥락을 모두 포착합니다.

A standard RNN reads only forward from $x_1$ to $x_{T_x}$, so annotation $h_j$ primarily contains information about preceding words. This paper uses a **Bidirectional RNN (BiRNN)** to capture context from both directions.

$$h_j = \left[\overrightarrow{h_j}^\top ; \overleftarrow{h_j}^\top\right]^\top \quad \text{(Eq. 7)}$$

- 순방향 RNN $\overrightarrow{f}$: $x_1 \rightarrow x_{T_x}$ 순서로 읽어 $\overrightarrow{h_1}, \cdots, \overrightarrow{h_{T_x}}$ 생성
- 역방향 RNN $\overleftarrow{f}$: $x_{T_x} \rightarrow x_1$ 순서로 읽어 $\overleftarrow{h_1}, \cdots, \overleftarrow{h_{T_x}}$ 생성
- Forward RNN $\overrightarrow{f}$: reads $x_1 \rightarrow x_{T_x}$, producing $\overrightarrow{h_1}, \cdots, \overrightarrow{h_{T_x}}$
- Backward RNN $\overleftarrow{f}$: reads $x_{T_x} \rightarrow x_1$, producing $\overleftarrow{h_1}, \cdots, \overleftarrow{h_{T_x}}$

RNN이 최근 입력을 더 잘 표현하는 경향이 있으므로, $h_j$는 $x_j$ **주변**의 단어들에 집중된 정보를 담게 됩니다.

Since RNNs tend to better represent recent inputs, $h_j$ contains information focused on the words **surrounding** $x_j$.

### Part IV: Experiment Settings (Section 4) / 실험 설정

**데이터**: WMT '14 영어-프랑스어 병렬 코퍼스. Europarl (61M words), news commentary (5.5M), UN (421M), crawled corpora (362.5M). 총 850M words에서 데이터 선별 방법(Axelrod et al., 2011)을 적용하여 348M words로 축소.

**Data**: WMT '14 English-French parallel corpora. Europarl (61M words), news commentary (5.5M), UN (421M), crawled corpora (362.5M). Reduced from 850M to 348M words using data selection (Axelrod et al., 2011).

**어휘**: 각 언어에서 빈도수 상위 30,000개 단어만 사용. 나머지는 [UNK] 토큰으로 대체. 별도의 전처리(소문자화, 스테밍 등) 없음.

**Vocabulary**: Top 30,000 most frequent words per language. Remaining words mapped to [UNK]. No other preprocessing (lowercasing, stemming, etc.).

**모델 비교 / Models compared:**

| 모델 / Model | 설명 / Description |
|---|---|
| RNNencdec-30 | 기본 encoder-decoder, 문장 길이 ≤ 30으로 학습 / Basic encoder-decoder, trained on sentences ≤ 30 words |
| RNNencdec-50 | 기본 encoder-decoder, 문장 길이 ≤ 50으로 학습 / Basic encoder-decoder, trained on sentences ≤ 50 words |
| RNNsearch-30 | Attention 모델, 문장 길이 ≤ 30으로 학습 / Attention model, trained on sentences ≤ 30 words |
| RNNsearch-50 | Attention 모델, 문장 길이 ≤ 50으로 학습 / Attention model, trained on sentences ≤ 50 words |

**아키텍처 세부사항 / Architecture details:**
- Encoder/Decoder hidden units: 1000 (GRU 사용 / using GRU)
- Word embedding dimensionality: 620
- Alignment model hidden units: 1000
- Maxout hidden layer: 500 units (deep output 구현)
- 학습: SGD + Adadelta ($\epsilon = 10^{-6}$, $\rho = 0.95$), minibatch size 80, ~5일 학습
- Training: SGD + Adadelta ($\epsilon = 10^{-6}$, $\rho = 0.95$), minibatch size 80, ~5 days training
- Gradient clipping: $L_2$-norm을 1로 제한 / $L_2$-norm clipped to 1
- 디코딩: beam search 사용 / Decoding: beam search

### Part V: Results (Section 5) / 결과

#### 5.1 Quantitative Results / 정량적 결과

**Table 1 — BLEU scores:**

| Model | All | No UNK |
|---|---|---|
| RNNencdec-30 | 13.93 | 24.19 |
| RNNsearch-30 | 21.50 | 31.44 |
| RNNencdec-50 | 17.82 | 26.71 |
| RNNsearch-50 | 26.75 | 34.16 |
| RNNsearch-50* | 28.45 | 36.15 |
| Moses | 33.30 | 35.63 |

핵심 관찰 / Key observations:
- RNNsearch가 모든 조건에서 RNNencdec을 대폭 능가 (RNNsearch-30: 21.50 vs RNNencdec-30: 13.93)
- RNNsearch-30이 RNNencdec-50보다 높은 BLEU (21.50 vs 17.82) — 짧은 문장으로 학습한 attention 모델이 긴 문장으로 학습한 기본 모델보다 우수
- UNK 없는 문장에서 RNNsearch-50* (36.15)은 Moses (35.63)를 근소하게 능가
- RNNsearch outperforms RNNencdec in all conditions (RNNsearch-30: 21.50 vs RNNencdec-30: 13.93)
- RNNsearch-30 beats RNNencdec-50 (21.50 vs 17.82) — attention model trained on shorter sentences surpasses baseline trained on longer ones
- On sentences without UNK, RNNsearch-50* (36.15) slightly outperforms Moses (35.63)

#### 5.2 Qualitative Analysis / 정성적 분석

**Alignment 시각화 (Figure 3, p.6):**

- 영어-프랑스어 간 alignment는 대체로 **단조(monotonic)**하며, 대각선 패턴을 보입니다.
- 그러나 흥미로운 **비단조(non-monotonic) alignment**도 관찰됩니다:
  - "European Economic Area" → "zone économique européenne" (Figure 3a): 형용사-명사 어순이 영어와 프랑스어에서 다르므로 모델이 정확히 역순으로 attention을 줌
  - "the man" → "l'homme" (Figure 3d): soft alignment이 "the"와 "man" 둘 다에 attention을 줘서 올바른 관사 형태를 선택
- Alignment between English-French is largely **monotonic** with diagonal patterns.
- But interesting **non-monotonic alignments** are observed:
  - "European Economic Area" → "zone économique européenne" (Fig 3a): adjective-noun order differs, model correctly attends in reverse
  - "the man" → "l'homme" (Fig 3d): soft alignment attends to both "the" and "man" to select correct article form

**긴 문장 처리 (Figure 2, Section 5.2.2):**

- RNNencdec의 BLEU 점수는 문장 길이 20 이후 급격히 하락
- RNNsearch-50은 길이 50 이상에서도 성능 저하 없음
- 구체적 사례: 30+ 단어의 긴 문장에서 RNNencdec-50은 ~30단어 후 의미가 벗어나기 시작하지만, RNNsearch-50은 전체 의미를 정확히 보존
- RNNencdec's BLEU drops sharply after sentence length 20
- RNNsearch-50 shows no performance deterioration even at length 50+
- Specific example: for long 30+ word sentences, RNNencdec-50 begins deviating from the source meaning after ~30 words, while RNNsearch-50 preserves the full meaning accurately

### Part VI: Related Work (Section 6) / 관련 연구

**Graves (2013)**: 필기체 합성(handwriting synthesis)에서 유사한 alignment 접근법을 사용했습니다. 그러나 그의 alignment는 location이 **단조 증가(monotonically increasing)**하도록 제약되어 있어, 어순이 재배열되는 기계 번역에는 부적합했습니다. Bahdanau의 방식은 이 제약이 없어 비단조 alignment가 가능합니다.

**Graves (2013)**: Used a similar alignment approach in handwriting synthesis. However, his alignment was constrained so that the location **increases monotonically**, making it unsuitable for machine translation where word order can be rearranged. Bahdanau's approach has no such constraint, allowing non-monotonic alignment.

**계산 비용에 대한 논의**: 이 접근법은 모든 소스 단어에 대해 매 디코딩 스텝마다 attention weight를 계산해야 하므로 $O(T_x \times T_y)$의 비용이 듭니다. 논문은 대부분의 번역 문장이 15-40 단어이므로 이 비용이 심각하지 않다고 언급하지만, 더 긴 시퀀스에서는 한계가 있을 수 있음을 인정합니다.

**Computational cost discussion**: This approach requires computing attention weights for all source words at every decoding step, costing $O(T_x \times T_y)$. The paper notes this is not severe for most translation sentences (15-40 words) but acknowledges potential limitations for longer sequences.

### Part VII: Conclusion (Section 7) / 결론

저자들은 attention 기반 접근법(RNNsearch)이 (1) 기존 encoder-decoder를 문장 길이에 관계없이 대폭 능가하고, (2) phrase-based SMT에 필적하는 성능을 달성했으며, (3) 언어학적으로 타당한 soft alignment을 학습한다는 세 가지 핵심 결과를 요약합니다. 미래 과제로 unknown word 처리 개선을 언급합니다.

The authors summarize three key results: (1) attention-based approach (RNNsearch) significantly outperforms basic encoder-decoder regardless of sentence length, (2) achieves performance comparable to phrase-based SMT, and (3) learns linguistically plausible soft alignments. Future work mentioned includes better handling of unknown words.

### Appendix A: Model Architecture Details / 모델 아키텍처 세부사항

#### A.1.1 GRU (Gated Recurrent Unit)

디코더의 GRU 상태 업데이트:

Decoder GRU state update:

$$s_i = f(s_{i-1}, y_{i-1}, c_i) = (1 - z_i) \circ s_{i-1} + z_i \circ \tilde{s}_i$$

여기서 $\circ$는 element-wise 곱셈이고:

Where $\circ$ is element-wise multiplication and:

- 제안 상태 / Proposed state: $\tilde{s}_i = \tanh(We(y_{i-1}) + U[r_i \circ s_{i-1}] + Cc_i)$
- 업데이트 게이트 / Update gate: $z_i = \sigma(W_z e(y_{i-1}) + U_z s_{i-1} + C_z c_i)$
- 리셋 게이트 / Reset gate: $r_i = \sigma(W_r e(y_{i-1}) + U_r s_{i-1} + C_r c_i)$

Update gate $z_i$는 이전 상태를 얼마나 유지할지, reset gate $r_i$는 이전 상태의 어떤 정보를 초기화할지를 제어합니다. 주목할 점: 기존 GRU와 달리 context vector $c_i$가 게이트 계산에도 포함됩니다.

Update gate $z_i$ controls how much of the previous state to retain; reset gate $r_i$ controls what information from the previous state to reset. Note: unlike standard GRU, the context vector $c_i$ is included in gate computations.

#### A.1.2 Alignment Model

$$a(s_{i-1}, h_j) = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$$

$W_a \in \mathbb{R}^{n \times n}$, $U_a \in \mathbb{R}^{n \times 2n}$, $v_a \in \mathbb{R}^n$. $U_a h_j$는 $i$에 의존하지 않으므로 사전 계산(pre-compute)하여 효율성을 높일 수 있습니다. 이 방식은 후에 **additive attention**이라 불리게 됩니다.

$W_a \in \mathbb{R}^{n \times n}$, $U_a \in \mathbb{R}^{n \times 2n}$, $v_a \in \mathbb{R}^n$. Since $U_a h_j$ does not depend on $i$, it can be pre-computed for efficiency. This approach was later called **additive attention**.

---

## 3. Key Takeaways / 핵심 시사점

1. **고정 길이 벡터 병목의 식별과 해결 / Identification and resolution of the fixed-length vector bottleneck** — 기존 encoder-decoder의 근본적 한계를 명확히 진단하고, 가변 길이 context vector로 해결했습니다. 이는 단순한 아키텍처 개선이 아니라 seq2seq 모델의 패러다임 전환입니다. / Clearly diagnosed the fundamental limitation of existing encoder-decoders and resolved it with variable-length context vectors. This was a paradigm shift, not just an architectural tweak.

2. **Soft alignment의 미분 가능성 / Differentiability of soft alignment** — Alignment를 latent variable이 아닌 직접 계산 가능한 soft 값으로 처리함으로써, 전체 모델을 end-to-end로 backpropagation할 수 있게 만들었습니다. 이것이 "Jointly Learning"의 핵심입니다. / By treating alignment as directly computable soft values rather than latent variables, the entire model can be backpropagated end-to-end. This is the essence of "Jointly Learning."

3. **Bidirectional RNN의 효과적 활용 / Effective use of Bidirectional RNN** — 각 annotation $h_j$가 해당 단어 주변의 양쪽 맥락을 모두 포착하도록 BiRNN을 인코더로 사용했습니다. 이는 순방향만 사용하는 경우보다 훨씬 풍부한 표현을 제공합니다. / Using BiRNN as encoder ensures each annotation $h_j$ captures context from both directions around the word, providing much richer representations than forward-only encoding.

4. **긴 문장에서의 극적인 성능 개선 / Dramatic improvement on long sentences** — Figure 2에서 RNNencdec는 길이 20 이후 성능이 급락하지만, RNNsearch는 길이 50 이상에서도 안정적입니다. Attention이 정보 압축의 부담을 근본적으로 해소했음을 실증합니다. / In Figure 2, RNNencdec's performance drops sharply after length 20, while RNNsearch remains stable even beyond length 50, empirically demonstrating that attention fundamentally relieves the information compression burden.

5. **Neural MT가 SMT에 필적할 수 있음을 실증 / Empirical demonstration that Neural MT can match SMT** — 단일 신경망 모델이 수많은 수작업 컴포넌트로 구성된 Moses에 필적하는 성능을 달성했습니다 (No UNK 조건: 36.15 vs 35.63). 이는 NMT가 SMT를 대체할 가능성을 처음으로 현실적으로 보여주었습니다. / A single neural network achieved performance comparable to Moses, which consists of many hand-engineered components (No UNK: 36.15 vs 35.63). This was the first realistic demonstration that NMT could replace SMT.

6. **Alignment 시각화의 해석 가능성 / Interpretability through alignment visualization** — Figure 3의 attention weight 행렬은 모델이 "어디를 보고 있는지"를 직관적으로 보여줍니다. 이는 black-box 신경망 모델에 해석 가능성을 부여한 초기 사례입니다. / The attention weight matrices in Figure 3 intuitively show "where the model is looking," an early example of providing interpretability to black-box neural networks.

7. **Attention은 독립 layer가 아닌 RNN 내부 컴포넌트 / Attention as an RNN internal component, not a standalone layer** — 이 논문에서 attention은 아직 GRU decoder의 내부 로직에 녹아있으며, 독립적이고 재사용 가능한 모듈로 분리되지 않았습니다. 이 모듈화는 이후 Luong (2015)에서 체계화되고, Vaswani (2017)의 Transformer에서 완성됩니다. / In this paper, attention is still embedded within the GRU decoder's internal logic, not separated as an independent, reusable module. This modularization was later systematized by Luong (2015) and completed in Vaswani's Transformer (2017).

8. **$O(T_x \times T_y)$ 계산 비용의 트레이드오프 / $O(T_x \times T_y)$ computational cost trade-off** — 매 디코딩 스텝마다 모든 소스 위치에 대해 attention을 계산해야 하는 비용이 있습니다. 짧은 문장(15-40 단어)에서는 문제없지만, 긴 문서 수준에서는 한계가 됩니다. 이후 sparse attention, linear attention 등의 연구로 이어졌습니다. / Computing attention over all source positions at every decoding step has a cost. Not a problem for short sentences (15-40 words), but a limitation at document scale. This led to subsequent research on sparse attention, linear attention, etc.

---

## 4. Mathematical Summary / 수학적 요약

### 전체 모델 수식 흐름 / Complete Model Equation Flow

#### 1단계: Bidirectional RNN Encoder / Step 1: Bidirectional RNN Encoder

입력 시퀀스 $\mathbf{x} = (x_1, \ldots, x_{T_x})$에 대해:

For input sequence $\mathbf{x} = (x_1, \ldots, x_{T_x})$:

$$\overrightarrow{h_i} = \begin{cases} (1 - \overrightarrow{z_i}) \circ \overrightarrow{h_{i-1}} + \overrightarrow{z_i} \circ \overrightarrow{\tilde{h}_i} & \text{if } i > 0 \\ 0 & \text{if } i = 0 \end{cases}$$

여기서 / where:
- $\overrightarrow{\tilde{h}_i} = \tanh(\overrightarrow{W}Ex_i + \overrightarrow{U}[\overrightarrow{r_i} \circ \overrightarrow{h_{i-1}}])$ (제안 상태 / proposed state)
- $\overrightarrow{z_i} = \sigma(\overrightarrow{W_z}Ex_i + \overrightarrow{U_z}\overrightarrow{h_{i-1}})$ (업데이트 게이트 / update gate)
- $\overrightarrow{r_i} = \sigma(\overrightarrow{W_r}Ex_i + \overrightarrow{U_r}\overrightarrow{h_{i-1}})$ (리셋 게이트 / reset gate)

역방향도 동일한 구조로 $\overleftarrow{h_i}$를 계산 ($x_{T_x} \rightarrow x_1$ 순서).
Backward states $\overleftarrow{h_i}$ computed similarly in reverse order ($x_{T_x} \rightarrow x_1$).

Annotation은 두 방향의 연결 / Annotation concatenates both directions:

$$h_j = \left[\overrightarrow{h_j} ; \overleftarrow{h_j}\right] \in \mathbb{R}^{2n}$$

#### 2단계: Attention / Step 2: Attention

각 디코딩 스텝 $i$에서 / At each decoding step $i$:

$$e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j) \quad \text{(alignment score)}$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})} \quad \text{(attention weight, softmax 정규화)}$$

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j \quad \text{(context vector)}$$

- $W_a \in \mathbb{R}^{n' \times n}$: 디코더 상태 변환 / decoder state transformation
- $U_a \in \mathbb{R}^{n' \times 2n}$: 인코더 annotation 변환 / encoder annotation transformation
- $v_a \in \mathbb{R}^{n'}$: 스칼라 점수로 사영 / projection to scalar score
- $n' = 1000$ (alignment model hidden units)

#### 3단계: GRU Decoder / Step 3: GRU Decoder

$$s_i = (1 - z_i) \circ s_{i-1} + z_i \circ \tilde{s}_i$$

- $\tilde{s}_i = \tanh(WEy_{i-1} + U[r_i \circ s_{i-1}] + Cc_i)$
- $z_i = \sigma(W_z Ey_{i-1} + U_z s_{i-1} + C_z c_i)$
- $r_i = \sigma(W_r Ey_{i-1} + U_r s_{i-1} + C_r c_i)$

초기 상태 / Initial state: $s_0 = \tanh(W_s \overleftarrow{h_1})$

#### 4단계: 출력 확률 / Step 4: Output Probability

$$p(y_i \mid s_i, y_{i-1}, c_i) \propto \exp(y_i^\top W_o t_i)$$

$$t_i = [\max(\tilde{t}_{i,2j-1}, \tilde{t}_{i,2j})]_{j=1,\ldots,l}^\top \quad \text{(maxout)}$$

$$\tilde{t}_i = U_o s_{i-1} + V_o Ey_{i-1} + C_o c_i$$

Maxout은 인접한 두 값의 최댓값을 취하여 비선형성을 도입합니다 ($l = 500$).

Maxout introduces nonlinearity by taking the max of adjacent pairs ($l = 500$).

### 파라미터 차원 요약 / Parameter Dimension Summary

| 파라미터 / Parameter | 차원 / Dimension | 설명 / Description |
|---|---|---|
| $n$ | 1000 | Hidden layer size |
| $m$ | 620 | Word embedding dimensionality |
| $K_x, K_y$ | 30,000 | Source/target vocabulary size |
| $n'$ | 1000 | Alignment model hidden units |
| $l$ | 500 | Maxout hidden layer size |
| $E$ | $\mathbb{R}^{m \times K}$ | Word embedding matrix |

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1997  ──── Hochreiter & Schmidhuber: LSTM
             │  (vanishing gradient 해결, 장기 의존성 학습 가능)
             ▼
2003  ──── Bengio et al.: Neural Probabilistic Language Model
             │  (단어 분산 표현의 기초)
             ▼
2013  ──── Kalchbrenner & Blunsom: Recurrent Continuous Translation Model
             │  (최초의 neural translation 시도 중 하나)
             │
2013  ──── Graves: Generating Sequences with RNNs
             │  (필기체 합성에서 단조 alignment 사용)
             ▼
2014  ──── Cho et al.: RNN Encoder-Decoder, GRU 제안
        │    (seq2seq의 기본 구조, GRU 게이트 유닛 도입)
        │
2014  ──── Sutskever et al.: Sequence to Sequence with LSTM
        │    (LSTM 기반 seq2seq, SMT에 근접한 성능)
        │
        ▼
2014  ═══ ★ Bahdanau et al.: Attention Mechanism ★  ← 이 논문 / this paper
             │  (고정 길이 벡터 병목 해결, soft alignment 도입)
             ▼
2015  ──── Luong et al.: Effective Approaches to Attention-based NMT
             │  (global/local attention, dot-product attention 체계화)
             ▼
2016  ──── Wu et al.: Google Neural Machine Translation (GNMT)
             │  (attention 기반 NMT의 산업화, SMT 완전 대체)
             ▼
2017  ──── Vaswani et al.: Attention Is All You Need (Transformer)
             │  (RNN 제거, self-attention, Q-K-V 구조, stackable layer)
             ▼
2018+ ──── BERT, GPT, Vision Transformer, ...
             (Transformer 기반 모델의 폭발적 확산)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #6 Rumelhart et al. (1986) — Backpropagation | Attention model이 jointly trained될 수 있는 이유는 전체 모델이 backpropagation으로 미분 가능하기 때문 / Attention can be jointly trained because the entire model is differentiable via backpropagation | 기초 학습 알고리즘 / Foundation training algorithm |
| #9 Hochreiter & Schmidhuber (1997) — LSTM | GRU는 LSTM의 간소화 버전. 이 논문의 RNN에서 GRU를 사용하며, LSTM도 대체 가능하다고 언급 / GRU is a simplified LSTM variant; the paper uses GRU but notes LSTM could substitute | 직접적 선행 연구 / Direct predecessor |
| Cho et al. (2014a) — RNN Encoder-Decoder | 이 논문이 확장하는 기본 모델(RNNencdec). GRU도 이 논문에서 제안됨 / The baseline model (RNNencdec) this paper extends. GRU was also proposed here | 직접적 기반 / Direct foundation |
| Sutskever et al. (2014) — Seq2Seq with LSTM | 동시대의 대안적 seq2seq 접근법. 둘 다 고정 길이 벡터 병목을 가짐 / Contemporary alternative seq2seq approach. Both share the fixed-length bottleneck | 동시대 경쟁 모델 / Contemporary competing model |
| Cho et al. (2014b) — Properties of Encoder-Decoder | 문장 길이 증가 시 encoder-decoder 성능 하락을 실증하여 이 논문의 동기를 제공 / Empirically showed encoder-decoder performance degrades with length, motivating this paper | 직접적 동기 / Direct motivation |
| Graves (2013) — Generating Sequences with RNNs | 필기체 합성에서 유사한 alignment 접근법 사용, 단 단조 증가 제약이 있었음 / Used similar alignment in handwriting synthesis, but constrained to monotonic increase | 관련 선행 연구 / Related prior work |
| #25 Vaswani et al. (2017) — Transformer | Bahdanau attention을 self-attention으로 확장하고, RNN을 완전히 제거하여 독립적인 attention layer로 정립 / Extended Bahdanau attention to self-attention, removed RNNs entirely, established attention as standalone layer | 직접적 후속 연구 / Direct successor |
| Luong et al. (2015) — Attention-based NMT | Bahdanau의 additive attention을 dot-product attention으로 단순화하고, global/local attention 변형을 체계화 / Simplified additive attention to dot-product, systematized global/local variants | 직접적 후속 연구 / Direct successor |

---

## 7. References / 참고문헌

- Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." arXiv:1409.0473. [ICLR 2015]
- Cho, K., van Merrienboer, B., Gulcehre, C., Bougares, F., Schwenk, H., & Bengio, Y. (2014a). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." EMNLP 2014.
- Cho, K., van Merrienboer, B., Bahdanau, D., & Bengio, Y. (2014b). "On the Properties of Neural Machine Translation: Encoder-Decoder Approaches." SSST-8.
- Sutskever, I., Vinyals, O., & Le, Q. (2014). "Sequence to Sequence Learning with Neural Networks." NeurIPS 2014.
- Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation, 9(8), 1735-1780. [DOI: 10.1162/neco.1997.9.8.1735]
- Graves, A. (2013). "Generating Sequences with Recurrent Neural Networks." arXiv:1308.0850.
- Kalchbrenner, N. & Blunsom, P. (2013). "Recurrent Continuous Translation Models." EMNLP 2013.
- Luong, M.-T., Pham, H., & Manning, C. D. (2015). "Effective Approaches to Attention-based Neural Machine Translation." EMNLP 2015.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A., Kaiser, L., & Polosukhin, I. (2017). "Attention Is All You Need." NeurIPS 2017.
- Schuster, M. & Paliwal, K. K. (1997). "Bidirectional Recurrent Neural Networks." IEEE Transactions on Signal Processing, 45(11), 2673-2681.
- Goodfellow, I., Warde-Farley, D., Mirza, M., Courville, A., & Bengio, Y. (2013). "Maxout Networks." ICML 2013.
- Bengio, Y., Ducharme, R., Vincent, P., & Janvin, C. (2003). "A Neural Probabilistic Language Model." JMLR, 3, 1137-1155.
- Axelrod, A., He, X., & Gao, J. (2011). "Domain Adaptation via Pseudo In-Domain Data Selection." ACL-EMNLP 2011.
- Zeiler, M. D. (2012). "ADADELTA: An Adaptive Learning Rate Method." arXiv:1212.5701.
