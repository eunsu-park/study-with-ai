---
title: "Pre-Reading Briefing: Neural Machine Translation by Jointly Learning to Align and Translate"
paper_id: "17_bahdanau_2014"
topic: Artificial_Intelligence
date: 2026-04-17
type: briefing
---

# Neural Machine Translation by Jointly Learning to Align and Translate: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." arXiv:1409.0473. Published at ICLR 2015.
**Author(s)**: Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio
**Year**: 2014 (ICLR 2015)

---

## 1. 핵심 기여 / Core Contribution

이 논문은 **attention mechanism(어텐션 메커니즘)**을 sequence-to-sequence 모델에 최초로 도입한 기념비적인 논문입니다. 기존의 encoder-decoder 아키텍처는 입력 문장 전체를 하나의 고정 길이 벡터(fixed-length vector)로 압축해야 했습니다. 이는 특히 긴 문장에서 정보 손실을 야기하는 심각한 병목(bottleneck)이었습니다. Bahdanau et al.은 디코더가 각 출력 단어를 생성할 때마다 입력 문장의 **관련 부분에 동적으로 집중(soft-search)**할 수 있는 메커니즘을 제안했습니다.

This paper introduced the **attention mechanism** to sequence-to-sequence models — one of the most influential ideas in modern deep learning. The conventional encoder-decoder architecture compressed an entire input sentence into a single fixed-length vector, creating an information bottleneck especially for long sentences. Bahdanau et al. proposed letting the decoder dynamically (soft-)search for relevant parts of the input sentence when generating each target word. This freed the model from the fixed-length constraint, dramatically improving translation quality on long sentences and achieving performance comparable to phrase-based statistical MT systems (Moses). The attention mechanism later became the foundation of the Transformer architecture (Vaswani et al., 2017).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

2014년은 Neural Machine Translation(NMT)이 막 태동하던 시기입니다. 기계 번역의 주류는 여전히 phrase-based statistical machine translation (SMT)이었으며, 대표적인 시스템이 Moses였습니다. SMT는 수많은 개별 컴포넌트(language model, translation model, reordering model 등)를 별도로 튜닝해야 했습니다.

2014 was the dawn of Neural Machine Translation (NMT). The dominant paradigm was still phrase-based statistical machine translation (SMT), exemplified by Moses. SMT required many separately tuned sub-components. Two key papers had just appeared: Cho et al. (2014a) proposed the RNN Encoder-Decoder with GRU units, and Sutskever et al. (2014) showed that an LSTM-based seq2seq model could approach SMT performance. However, both approaches suffered from the fixed-length bottleneck — performance degraded sharply as sentence length increased. Bahdanau's paper directly addressed this fundamental limitation.

### 타임라인 / Timeline

```
1997  Hochreiter & Schmidhuber — LSTM (vanishing gradient 해결)
2003  Bengio et al. — Neural probabilistic language model
2013  Kalchbrenner & Blunsom — Recurrent continuous translation model
2014  Cho et al. — RNN Encoder-Decoder, GRU 제안
2014  Sutskever et al. — Sequence to sequence with LSTM
2014  ★ Bahdanau et al. — Attention mechanism 도입 ← 이 논문
2015  Luong et al. — Global/local attention 변형
2016  Wu et al. — Google Neural Machine Translation (GNMT)
2017  Vaswani et al. — Transformer ("Attention Is All You Need")
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 필수 선행 논문 / Required Prior Papers
- **Paper #6** (Rumelhart et al., 1986): Backpropagation — 신경망 학습의 기본 알고리즘
- **Paper #9** (Hochreiter & Schmidhuber, 1997): LSTM — 시퀀스 모델링의 기초, GRU의 기원

### 개념적 배경 / Conceptual Background

| 개념 / Concept | 설명 / Description |
|---|---|
| RNN (Recurrent Neural Network) | 시퀀스 데이터를 처리하는 신경망. 각 시점의 hidden state가 이전 시점의 정보를 전달 / Neural network for sequential data where hidden states carry information across time steps |
| Encoder-Decoder | 입력 시퀀스를 고정 벡터로 인코딩한 후, 이 벡터에서 출력 시퀀스를 디코딩하는 구조 / Architecture that encodes input to a fixed vector, then decodes output from it |
| GRU (Gated Recurrent Unit) | LSTM의 간소화 버전. Update gate와 reset gate로 정보 흐름 제어 / Simplified LSTM variant with update and reset gates (Cho et al., 2014a) |
| Bidirectional RNN | 순방향과 역방향으로 동시에 시퀀스를 읽어 양쪽 맥락을 모두 포착 / Reads sequence in both directions to capture context from both sides |
| Softmax | 벡터를 확률 분포로 변환하는 함수 / Function that converts a vector into a probability distribution |
| BLEU score | 기계 번역 품질의 표준 평가 지표 / Standard evaluation metric for machine translation quality |

### 수학적 배경 / Mathematical Background
- 확률과 조건부 확률 (conditional probability, $p(\mathbf{y} \mid \mathbf{x})$)
- 행렬 곱셈과 벡터 연산 (matrix multiplication, dot product)
- Softmax 함수와 지수 함수 (exponential function)
- Chain rule과 backpropagation을 통한 gradient 계산

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Attention** | 디코더가 각 출력 단어를 생성할 때 입력 시퀀스의 어떤 부분에 집중할지 결정하는 메커니즘 / Mechanism that determines which parts of the input the decoder focuses on when generating each output word |
| **Alignment model** | 소스 단어 $j$와 타깃 단어 $i$ 사이의 관련성을 점수화하는 함수 $a(s_{i-1}, h_j)$ / Function scoring how well input position $j$ matches output position $i$ |
| **Context vector** ($c_i$) | 각 디코딩 스텝에서 attention weight로 가중합된 인코더 annotation 벡터 / Weighted sum of encoder annotations at each decoding step |
| **Annotation** ($h_j$) | BiRNN 인코더에서 $j$번째 단어에 대한 양방향 hidden state의 연결 / Concatenation of forward and backward hidden states from BiRNN encoder for word $j$ |
| **Soft alignment** | 확률적(연속적) 정렬. 하나의 타깃 단어가 여러 소스 단어에 부분적으로 정렬 가능 / Probabilistic (continuous) alignment where one target word can partially align to multiple source words |
| **Hard alignment** | 전통적 SMT의 이산적 정렬. 각 단어가 정확히 하나의 단어에 대응 / Discrete alignment in traditional SMT where each word maps to exactly one word |
| **RNNsearch** | 이 논문에서 제안한 attention 기반 모델의 이름 / Name of the proposed attention-based model |
| **RNNencdec** | Cho et al. (2014a)의 기본 encoder-decoder 모델 (비교 대상) / Baseline encoder-decoder model (Cho et al., 2014a) |
| **Bidirectional RNN (BiRNN)** | 인코더에서 사용. 순방향/역방향 RNN을 결합하여 각 단어의 양쪽 맥락을 포착 / Used in encoder; combines forward/backward RNNs to capture context from both directions |
| **GRU (Gated Recurrent Unit)** | 이 논문의 RNN에서 사용한 게이트 유닛. Update gate $z_i$와 reset gate $r_i$로 구성 / Gated unit used in the RNN, with update gate $z_i$ and reset gate $r_i$ |
| **Maxout** | 출력 확률 계산에 사용된 활성화 함수. 여러 선형 함수의 최댓값을 취함 / Activation function taking the max over multiple linear functions, used for output probability |
| **Beam search** | 디코딩 시 여러 후보를 동시에 탐색하는 방법 / Decoding method that explores multiple candidate translations simultaneously |

---

## 5. 수식 미리보기 / Equations Preview

### 수식 1: 기본 Encoder-Decoder의 조건부 확률 / Basic Encoder-Decoder Conditional Probability

$$p(\mathbf{y}) = \prod_{t=1}^{T} p(y_t \mid \{y_1, \cdots, y_{t-1}\}, c)$$

기존 모델에서는 고정된 context vector $c$ 하나에 의존합니다. 모든 디코딩 스텝에서 동일한 $c$를 사용합니다.

In the basic model, the entire translation depends on a single fixed context vector $c$, shared across all decoding steps.

### 수식 2: Attention이 적용된 조건부 확률 / Attention-augmented Conditional Probability

$$p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x}) = g(y_{i-1}, s_i, c_i)$$

핵심 변화: 고정된 $c$ 대신 **각 디코딩 스텝 $i$마다 고유한 $c_i$**를 사용합니다.

Key change: instead of a fixed $c$, each decoding step $i$ uses its own context vector $c_i$.

### 수식 3: Context vector 계산 / Context Vector Computation

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$$

$c_i$는 인코더 annotation $h_j$들의 가중합입니다. $\alpha_{ij}$는 attention weight로, 타깃 단어 $i$를 생성할 때 소스 단어 $j$에 얼마나 주목할지를 나타냅니다.

The context vector $c_i$ is a weighted sum of encoder annotations $h_j$. The weights $\alpha_{ij}$ represent how much attention target word $i$ pays to source word $j$.

### 수식 4: Attention weight (Softmax 정규화) / Attention Weight

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

where $e_{ij} = a(s_{i-1}, h_j)$

$e_{ij}$는 alignment score이며, alignment model $a$가 이전 decoder state $s_{i-1}$와 encoder annotation $h_j$의 관련성을 점수화합니다. Softmax를 통해 확률 분포로 정규화됩니다.

$e_{ij}$ is the alignment score computed by alignment model $a$, measuring how well $s_{i-1}$ (previous decoder state) and $h_j$ (source annotation) match. Normalized via softmax to form a probability distribution.

### 수식 5: Alignment model / Alignment Model

$$a(s_{i-1}, h_j) = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$$

단일 hidden layer를 가진 feedforward neural network입니다. $W_a$, $U_a$, $v_a$는 학습 가능한 파라미터입니다. 전체 모델과 함께 jointly trained 됩니다.

A single-layer feedforward network with learnable parameters $W_a$, $U_a$, $v_a$, trained jointly with the entire model. This is sometimes called "additive attention" (vs. Luong's later "dot-product attention").

---

## 6. 읽기 가이드 / Reading Guide

### 읽기 순서 권장 / Recommended Reading Order

1. **Abstract + Section 1 (Introduction)**: 문제 정의 파악 — 고정 길이 벡터의 병목이 왜 문제인지 / Understand the fixed-length bottleneck problem
2. **Section 2 (Background)**: 기존 RNN Encoder-Decoder 복습 (이미 읽은 #9 LSTM 배경 활용) / Review of baseline RNN Encoder-Decoder
3. **Section 3 (Learning to Align and Translate)**: ⭐ **가장 중요한 섹션**. Figure 1을 중심으로 attention 메커니즘 이해 / The core section — understand attention via Figure 1
   - 3.1: Decoder + attention의 수식 전개
   - 3.2: Bidirectional RNN encoder
4. **Section 5 (Results)**: Table 1과 Figure 2, 3에 집중 / Focus on quantitative results and alignment visualizations
5. **Appendix A**: 구현 세부사항이 궁금할 때 참고 / Reference for implementation details

### 주의할 점 / Points to Watch

- **Figure 1** (p.3): 전체 아키텍처를 한눈에 보여주는 핵심 다이어그램입니다. 이 그림을 완전히 이해하는 것이 목표입니다.
- **Figure 3** (p.6): Alignment 시각화. 대각선 패턴이 단조(monotonic) 정렬을, 대각선에서 벗어난 밝은 점이 비단조(non-monotonic) 정렬을 보여줍니다. 형용사-명사 어순 차이(영어 vs 프랑스어)에 주목하세요.
- **Figure 2** (p.5): 문장 길이에 따른 BLEU 점수 변화. RNNencdec은 길이가 길어질수록 급격히 하락하지만, RNNsearch는 안정적입니다.

---

## 7. 현대적 의의 / Modern Significance

이 논문의 attention mechanism은 현대 딥러닝의 **가장 영향력 있는 아이디어 중 하나**가 되었습니다:

This paper's attention mechanism became one of the **most influential ideas** in modern deep learning:

- **Transformer (2017)**: "Attention Is All You Need" — RNN을 완전히 제거하고 attention만으로 구성. Bahdanau attention이 없었다면 Transformer도 없었을 것 / Removed RNNs entirely, using only attention. Bahdanau attention was the direct precursor
- **BERT, GPT 시리즈**: Transformer 기반 언어 모델들의 핵심은 모두 attention / All Transformer-based LMs are built on attention
- **Vision Transformer (ViT)**: 이미지 인식에서도 attention이 CNN을 대체 / Attention replacing CNNs in computer vision
- **Cross-attention**: 멀티모달 모델(텍스트-이미지)에서 서로 다른 모달리티 간의 정보 교환에 사용 / Used for information exchange between modalities
- **"Additive attention" vs "Dot-product attention"**: Bahdanau의 방식은 additive attention이라 불리며, 이후 Luong (2015)의 dot-product attention, Vaswani (2017)의 scaled dot-product attention으로 발전 / Evolved from additive → dot-product → scaled dot-product

논문의 인용 수는 40,000회 이상으로, 딥러닝 역사상 가장 많이 인용된 논문 중 하나입니다.

The paper has over 40,000 citations, making it one of the most cited papers in deep learning history.

---

## Q&A

### Q1: Attention mechanism에 대한 상세 설명 / Detailed Explanation of Attention Mechanism

#### 1. 왜 Attention이 필요한가? / Why is Attention Needed?

기존 encoder-decoder의 구조를 먼저 살펴보겠습니다:

The basic encoder-decoder architecture works as follows:

```
입력 / Input: "The cat sat on the mat"
                ↓
           [Encoder RNN] → 각 단어를 순차적으로 읽음 / reads words sequentially
                ↓
           고정 길이 벡터 c (예: 1000차원) / fixed-length vector c  ← 병목 / bottleneck!
                ↓
           [Decoder RNN] → "Le chat était assis sur le tapis"
```

문제는 **문장이 아무리 길어도** (5단어든 50단어든) 동일한 크기의 벡터 $c$ 하나에 모든 정보를 압축해야 한다는 것입니다. 짧은 문장은 괜찮지만, 긴 문장에서는 앞쪽 단어의 정보가 점점 사라집니다. 논문의 Figure 2에서 RNNencdec의 BLEU 점수가 문장 길이 20 이후 급락하는 것이 바로 이 현상입니다.

The problem is that **no matter how long the sentence is** (5 words or 50 words), all information must be compressed into a single vector $c$ of the same size. Short sentences are fine, but for long sentences, information about earlier words gradually disappears. This is exactly why RNNencdec's BLEU score drops sharply after sentence length 20 in Figure 2 of the paper.

#### 2. Attention의 핵심 아이디어 / Core Idea of Attention

사람이 번역할 때를 생각해 보세요. "European Economic Area"를 "zone économique européenne"으로 번역할 때, **원문 전체를 한 번에 외워서** 번역하지 않습니다. 각 단어를 쓸 때마다 **원문의 해당 부분을 다시 참조**합니다.

Think about how humans translate. When translating "European Economic Area" into "zone économique européenne," you don't **memorize the entire source text at once**. You **refer back to the relevant part** of the source each time you write a word.

Attention은 정확히 이것을 모방합니다 / Attention mimics exactly this:

```
디코더가 "zone"을 생성할 때         → "Area"에 집중
When decoder generates "zone"      → focuses on "Area"

디코더가 "économique"를 생성할 때   → "Economic"에 집중
When decoder generates "économique" → focuses on "Economic"

디코더가 "européenne"을 생성할 때   → "European"에 집중
When decoder generates "européenne" → focuses on "European"
```

#### 3. 수식을 단계별로 따라가기 / Step-by-Step Walkthrough of the Equations

디코더가 $i$번째 타깃 단어를 생성하는 과정입니다:

Here is the process of the decoder generating the $i$-th target word:

##### Step 1: 인코더가 annotation 생성 / Encoder Generates Annotations

Bidirectional RNN이 입력의 각 단어 $x_j$에 대해 annotation $h_j$를 만듭니다:

The Bidirectional RNN creates an annotation $h_j$ for each input word $x_j$:

$$h_j = \left[\overrightarrow{h_j}^\top ; \overleftarrow{h_j}^\top\right]^\top$$

- 순방향 $\overrightarrow{h_j}$는 $x_1 \cdots x_j$까지의 맥락을 담습니다.
  Forward $\overrightarrow{h_j}$ captures context from $x_1 \cdots x_j$.
- 역방향 $\overleftarrow{h_j}$는 $x_j \cdots x_{T_x}$까지의 맥락을 담습니다.
  Backward $\overleftarrow{h_j}$ captures context from $x_j \cdots x_{T_x}$.
- 따라서 $h_j$는 **$x_j$ 주변의 양쪽 문맥**을 모두 포착합니다.
  Thus $h_j$ captures **context from both sides** surrounding $x_j$.

##### Step 2: Alignment score 계산 / Compute Alignment Scores

디코더의 이전 hidden state $s_{i-1}$과 각 인코더 annotation $h_j$ 사이의 **관련성 점수**를 계산합니다:

Compute the **relevance score** between the decoder's previous hidden state $s_{i-1}$ and each encoder annotation $h_j$:

$$e_{ij} = a(s_{i-1}, h_j) = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$$

직관적으로: "디코더가 지금 이 상태($s_{i-1}$)에 있을 때, 소스의 $j$번째 단어($h_j$)가 얼마나 관련 있는가?"를 묻는 것입니다.

Intuitively: "Given the decoder is in state $s_{i-1}$, how relevant is the $j$-th source word ($h_j$)?"

구체적인 예시 / Concrete example:

```
소스 / Source: "The  cat  sat  on  the  mat"
                h₁   h₂   h₃   h₄   h₅   h₆

디코더가 "chat"(고양이)을 생성하려 할 때 (state = s₁):
When decoder tries to generate "chat" (cat) with state = s₁:

  e₂₁ = a(s₁, h₁) = 0.1   ("The" → 약한 관련 / weak relevance)
  e₂₂ = a(s₁, h₂) = 2.8   ("cat" → 강한 관련! / strong relevance!) ★
  e₂₃ = a(s₁, h₃) = 0.3   ("sat" → 약한 관련 / weak relevance)
  e₂₄ = a(s₁, h₄) = 0.0   ("on"  → 거의 무관 / almost irrelevant)
  e₂₅ = a(s₁, h₅) = 0.1   ("the" → 약한 관련 / weak relevance)
  e₂₆ = a(s₁, h₆) = 0.2   ("mat" → 약한 관련 / weak relevance)
```

##### Step 3: Softmax로 정규화 → Attention weights / Normalize via Softmax

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

위의 예시를 softmax에 통과시키면 / Passing the above example through softmax:

```
α₂₁ = 0.04   α₂₂ = 0.62   α₂₃ = 0.05
α₂₄ = 0.04   α₂₅ = 0.04   α₂₆ = 0.05
              ↑
         "cat"에 62% 집중! / 62% focus on "cat"!
```

모든 $\alpha_{ij}$의 합은 항상 1입니다 (확률 분포).
The sum of all $\alpha_{ij}$ is always 1 (probability distribution).

##### Step 4: Context vector 계산 / Compute Context Vector

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$$

각 annotation을 attention weight로 가중합합니다. "cat"에 해당하는 $h_2$가 가장 큰 가중치(0.62)를 받으므로, $c_2$는 주로 $h_2$의 정보를 담게 됩니다.

Each annotation is weighted by its attention weight. Since $h_2$ ("cat") receives the largest weight (0.62), the context vector $c_2$ primarily contains the information from $h_2$.

##### Step 5: 디코더가 다음 단어 생성 / Decoder Generates Next Word

$$p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x}) = g(y_{i-1}, s_i, c_i)$$

디코더는 세 가지를 종합하여 다음 단어를 결정합니다:
The decoder combines three inputs to determine the next word:

1. 이전 출력 $y_{i-1}$ / Previous output $y_{i-1}$
2. 현재 state $s_i$ / Current state $s_i$
3. **이번 스텝 전용 context vector** $c_i$ / **Step-specific context vector** $c_i$

#### 4. Soft Alignment vs Hard Alignment

이 구분이 논문의 중요한 기여입니다:

This distinction is a key contribution of the paper:

| | Hard Alignment (전통 SMT) | Soft Alignment (이 논문) |
|---|---|---|
| 방식 / Method | 각 타깃 단어가 정확히 하나의 소스 단어에 대응 / Each target word maps to exactly one source word | 각 타깃 단어가 **모든** 소스 단어에 확률적으로 대응 / Each target word probabilistically attends to **all** source words |
| 값 / Values | 0 또는 1 (이산적) / 0 or 1 (discrete) | 0~1 사이의 연속 값 / Continuous values between 0 and 1 |
| 학습 / Training | 별도의 alignment 모델 필요 (EM 알고리즘 등) / Separate alignment model needed (e.g., EM algorithm) | 전체 모델과 **jointly** 학습 (backprop 가능!) / Trained **jointly** with entire model (backprop-compatible!) |
| 한계 / Limitation | 1:1 대응이 안 되는 경우 처리 어려움 / Struggles with non-1:1 correspondences | 1:N, N:1, N:M 대응 자연스럽게 처리 / Naturally handles 1:N, N:1, N:M correspondences |

예를 들어 "the man" → "l'homme"에서:
For example, translating "the man" → "l'homme":

- **Hard alignment**: "the" → "l'", "man" → "homme" (하지만 "the"를 "le/la/les/l'" 중 어떤 것으로 번역할지 결정하려면 "man"도 봐야 합니다 / But to decide whether "the" becomes "le/la/les/l'", you need to see "man" too)
- **Soft alignment**: "l'"을 생성할 때 "the"와 "man" **둘 다**에 attention을 줌 → 남성 명사임을 파악하고 올바르게 "l'"을 선택 / When generating "l'", the model attends to **both** "the" and "man" → recognizes it's a masculine noun and correctly selects "l'"

#### 5. 왜 이것이 혁명적인가? / Why is This Revolutionary?

핵심은 **미분 가능(differentiable)**하다는 것입니다. Softmax 기반의 soft attention은 gradient가 흐를 수 있으므로, alignment를 별도로 학습할 필요 없이 번역 성능 최적화와 함께 자동으로 학습됩니다. 이것이 "Jointly Learning to Align and Translate"라는 제목의 의미입니다.

The key is that it is **differentiable**. Softmax-based soft attention allows gradients to flow through, so alignment is learned automatically alongside translation optimization — no separate alignment training needed. This is what the title "Jointly Learning to Align and Translate" means.

이 아이디어가 3년 뒤 Transformer에서 **Self-Attention**으로 발전합니다. Bahdanau attention은 "인코더 → 디코더" 간의 cross-attention이지만, Transformer의 self-attention은 같은 시퀀스 내에서 모든 위치가 서로에게 attention을 주는 것으로 확장한 것입니다.

This idea evolved into **Self-Attention** in the Transformer three years later. Bahdanau attention is cross-attention (encoder → decoder), but the Transformer's self-attention extends this so that every position in the same sequence attends to every other position.

---

### Q2: 이 논문에서 attention layer의 구조가 정립되었는가? / Did This Paper Establish the Attention Layer Architecture?

결론부터 말하면, **이 논문은 attention의 "아이디어"를 정립했지만, 독립적인 "layer"로서의 구조를 정립한 것은 아닙니다.**

In short, **this paper established the "idea" of attention, but did not establish it as an independent, reusable "layer."**

#### 이 논문에서 정립된 것 / What This Paper Established

Bahdanau et al.이 확립한 것은 attention의 **연산 파이프라인**입니다:

What Bahdanau et al. established is the **computational pipeline** of attention:

```
s_{i-1}, h_j  →  alignment score (e_ij)  →  softmax (α_ij)  →  weighted sum (c_i)
```

이 흐름 자체는 이후 모든 attention 변형의 원형이 되었습니다. 하지만 논문에서 이것을 **독립적인 layer**로 분리하지는 않았습니다. Attention 연산이 GRU decoder의 내부 로직에 **녹아들어** 있습니다:

This flow became the prototype for all subsequent attention variants. However, the paper did not separate this as an **independent layer**. The attention computation is **embedded within** the GRU decoder's internal logic:

```
디코더의 한 스텝 / One decoder step:
  1. 이전 state s_{i-1} 으로 alignment score 계산    ← attention 부분 / attention part
  2. softmax → α_ij 계산                           ← attention 부분 / attention part
  3. context vector c_i = Σ α_ij · h_j              ← attention 부분 / attention part
  4. GRU로 s_i = f(s_{i-1}, y_{i-1}, c_i) 계산      ← RNN 부분 / RNN part
  5. 출력 확률 계산                                  ← output 부분 / output part
```

Attention이 decoder RNN의 한 **컴포넌트**이지, 독립적으로 재사용 가능한 모듈이 아닙니다.

Attention is a **component** of the decoder RNN, not an independently reusable module.

#### Attention "Layer"가 정립된 시점 / When the Attention "Layer" Was Established

| 발전 단계 / Stage | 논문 / Paper | 기여 / Contribution |
|---|---|---|
| **개념 도입** / Concept introduction | Bahdanau et al. (2014) ← 이 논문 | Attention 연산의 기본 파이프라인 확립 / Established the basic computational pipeline of attention |
| **변형 체계화** / Systematization | Luong et al. (2015) | Global vs local attention, dot-product attention 등 여러 변형을 정리. Attention을 좀 더 모듈화 / Organized multiple variants; made attention more modular |
| **독립 layer로 정립** / Independent layer | Vaswani et al. (2017) "Attention Is All You Need" | RNN을 완전히 제거하고, attention을 **독립적이고 쌓을 수 있는(stackable) layer**로 정의. Query-Key-Value 구조 도입 / Removed RNNs entirely; defined attention as a **standalone, stackable layer** with Query-Key-Value structure |

Transformer에서 비로소 attention이 이렇게 정의됩니다:

In the Transformer, attention is finally defined as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- **입력 / Input**: Query, Key, Value (명확히 분리된 역할 / clearly separated roles)
- **출력 / Output**: attention이 적용된 벡터 / attention-applied vector
- **특징 / Feature**: 어디에나 끼워 넣을 수 있는 독립 모듈, 여러 층으로 쌓을 수 있음 / A standalone module that can be inserted anywhere and stacked into multiple layers

#### 비유 / Analogy

Bahdanau의 attention은 **"이 요리에 이 양념을 쓰면 맛있다"**를 발견한 것이고, Transformer의 attention layer는 **"이 양념을 표준 규격의 병에 담아서 어떤 요리에든 쓸 수 있게 만든 것"**입니다.

Bahdanau's attention is like **discovering that "this seasoning makes this dish delicious,"** while the Transformer's attention layer is like **"packaging that seasoning in a standardized bottle so it can be used in any dish."**

Reading list #25 "Attention Is All You Need"에서 그 완성된 형태를 보시게 될 것입니다.

You will see the completed form in reading list #25, "Attention Is All You Need."
