---
title: "Pre-Reading Briefing: Attention Is All You Need"
paper_id: "25_vaswani_2017"
topic: Artificial_Intelligence
date: 2026-04-19
type: briefing
---

# Attention Is All You Need: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
**Author(s)**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain / Google Research / University of Toronto)
**Year**: 2017
**arXiv**: 1706.03762

---

## 1. 핵심 기여 / Core Contribution

### 한국어

이 논문은 sequence transduction(예: 기계 번역)을 위한 완전히 새로운 아키텍처 **Transformer**를 제안합니다. 핵심 주장은 간단하고 급진적입니다: **순환(recurrence)도, 합성곱(convolution)도 필요 없다 — 오직 attention만으로 충분하다.** 기존의 RNN 기반 seq2seq 모델은 시퀀스를 순차적으로 처리하기 때문에 병렬화가 본질적으로 어렵고, 긴 의존성(long-range dependency)을 학습하기도 힘들었습니다. Transformer는 self-attention을 사용하여 시퀀스 내 모든 위치 쌍을 상수 시간에 연결하며(path length = O(1)), 학습을 전적으로 병렬화할 수 있게 합니다. WMT 2014 영어→독일어 번역에서 28.4 BLEU, 영어→프랑스어에서 41.8 BLEU로 당시 최고 성능(state-of-the-art)을 달성했으며, 학습 비용도 기존 최고 모델의 일부에 불과했습니다.

구체적 혁신: (1) **Scaled Dot-Product Attention**과 **Multi-Head Attention**, (2) **Positional Encoding**으로 시퀀스 순서 정보 주입, (3) **완전 feed-forward** 인코더-디코더 스택, (4) residual connection + layer normalization.

### English

This paper proposes a radically new architecture called the **Transformer** for sequence transduction (e.g., machine translation). The central claim is simple and bold: **no recurrence, no convolution — attention alone suffices.** RNN-based seq2seq models are inherently sequential and hard to parallelize, and also struggle with long-range dependencies. The Transformer uses self-attention to connect any pair of positions with constant path length O(1), enabling fully parallel training. It achieves 28.4 BLEU on WMT 2014 English→German and 41.8 BLEU on English→French, setting a new state-of-the-art at a fraction of the training cost of prior best models.

Key innovations: (1) **Scaled Dot-Product Attention** and **Multi-Head Attention**, (2) **Positional Encoding** to inject order information, (3) a purely feed-forward encoder-decoder stack, (4) residual connections + layer normalization.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**: 2014~2016년은 신경망 기계 번역(Neural Machine Translation, NMT)의 전성기였습니다. Sutskever et al.(2014)의 seq2seq, Bahdanau et al.(2014)의 additive attention, Luong et al.(2015)의 multiplicative attention이 연달아 등장하며 RNN(LSTM, GRU) 기반 인코더-디코더가 표준이 되었습니다. 2016년에 Google은 GNMT로 번역 서비스를 구글 신경망 번역으로 전환했고, ByteNet(Kalchbrenner et al., 2016), ConvS2S(Gehring et al., 2017) 같은 CNN 기반 번역 모델도 등장했습니다. 그러나 RNN은 시퀀스 길이에 비례하는 순차적 계산 때문에 GPU 병렬화의 한계에 봉착했고, CNN은 먼 거리 의존성을 위해 여러 층을 쌓아야 했습니다. Transformer는 이 두 가지 한계를 동시에 돌파했습니다.

**English**: 2014–2016 was the heyday of Neural Machine Translation. Sutskever et al.'s seq2seq (2014), Bahdanau et al.'s additive attention (2014), and Luong et al.'s multiplicative attention (2015) established RNN (LSTM/GRU) encoder-decoder as the standard. In 2016 Google switched its translation service to GNMT. CNN-based alternatives like ByteNet (2016) and ConvS2S (2017) also emerged. Yet RNNs were fundamentally bottlenecked by sequential computation, and CNNs required many layers to capture long-range dependencies. The Transformer broke both limitations at once.

### 타임라인 / Timeline

```
1997 ─── LSTM (Hochreiter & Schmidhuber)
  │
2014 ─── Seq2Seq (Sutskever et al.)        ← 논문 #17
  │       Additive Attention (Bahdanau et al.) ← 논문 #18
  │
2015 ─── Multiplicative Attention (Luong et al.)
  │
2016 ─── GNMT (Google Neural Machine Translation)
  │       ByteNet, ResNet (residual connections)
  │
2017 ─── ConvS2S (Gehring et al.)
  │   ★  ATTENTION IS ALL YOU NEED ★ ← 이 논문 / this paper
  │
2018 ─── BERT (Devlin et al.) — Transformer 인코더만 사용
  │       GPT (Radford et al.) — Transformer 디코더만 사용
2020 ─── GPT-3 (175B params) — scaling law 검증
2022 ─── ChatGPT
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어

**수학적 배경**:
- 선형대수: 행렬 곱, softmax, 내적(dot product)
- 확률: softmax 확률 분포, cross-entropy loss
- 미적분: gradient, chain rule (이미 알고 있음)

**사전 논문**:
- **논문 #17 (Sutskever et al., 2014)** — seq2seq encoder-decoder 구조
- **논문 #18 (Bahdanau et al., 2014)** — attention mechanism 최초 제안
- **논문 #21 (ResNet, He et al., 2016)** — residual connection

**개념**:
- Word embedding (Mikolov et al., word2vec)
- RNN / LSTM의 한계 (vanishing gradient, 순차 계산)
- Layer Normalization (Ba et al., 2016)
- Dropout (Srivastava et al., 2014)
- Label smoothing (Szegedy et al., 2016)

### English

**Mathematical background**:
- Linear algebra: matrix multiplication, softmax, dot product
- Probability: softmax distribution, cross-entropy loss
- Calculus: gradients, chain rule

**Prior papers**:
- **Paper #17 (Sutskever et al., 2014)** — seq2seq encoder-decoder
- **Paper #18 (Bahdanau et al., 2014)** — first attention mechanism
- **Paper #21 (ResNet, He et al., 2016)** — residual connections

**Concepts**:
- Word embeddings (word2vec)
- Limitations of RNN/LSTM (vanishing gradients, sequential computation)
- Layer Normalization (Ba et al., 2016)
- Dropout, Label smoothing

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Self-Attention** | 한 시퀀스 내의 각 위치가 같은 시퀀스의 모든 위치를 참조하여 자신의 표현을 갱신하는 메커니즘 / Each position in a sequence attends to all positions in the same sequence to update its representation. |
| **Query, Key, Value (Q, K, V)** | Attention의 세 가지 벡터. Query는 "무엇을 찾는가", Key는 "내가 무엇인가", Value는 "실제 전달할 정보". / Three vectors in attention — Query ("what I seek"), Key ("what I am"), Value ("information to pass"). |
| **Scaled Dot-Product Attention** | $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k}) V$. $\sqrt{d_k}$ 스케일링으로 gradient 안정화. / Dot-product attention with $1/\sqrt{d_k}$ scaling for stable gradients. |
| **Multi-Head Attention** | Attention을 $h$개의 병렬 헤드로 나누어 서로 다른 표현 부분공간에서 정보를 집계 / Attention split into $h$ parallel heads, each attending in a different representation subspace. |
| **Positional Encoding** | 순서 정보가 없는 self-attention에 위치 정보를 주입하는 sin/cos 함수 기반 벡터 / Sin/cos-based vectors injecting order information into the otherwise permutation-invariant self-attention. |
| **Encoder-Decoder** | 입력 시퀀스를 인코딩한 후 출력 시퀀스를 디코딩하는 구조. Transformer는 둘 다 $N=6$개 layer로 구성. / Encodes input sequence, then decodes output. Transformer stacks $N=6$ layers on each side. |
| **Masked Self-Attention** | 디코더에서 미래 위치를 보지 못하도록 마스킹하여 autoregressive 특성 유지 / Masks future positions in the decoder to preserve autoregressive property. |
| **Position-wise Feed-Forward Network** | 각 위치에 독립적으로 적용되는 2-layer MLP. 입력 $d_{model}=512$, 은닉 $d_{ff}=2048$. / 2-layer MLP applied independently at each position, with hidden size $d_{ff}=2048$. |
| **Layer Normalization** | 각 샘플 내 feature 차원을 정규화. batch norm과 달리 시퀀스 길이에 무관. / Normalizes across features within each sample; independent of sequence length. |
| **Residual Connection** | $\text{LayerNorm}(x + \text{Sublayer}(x))$. 깊은 네트워크 학습을 안정화. / Stabilizes training of deep networks. |
| **BLEU Score** | Bilingual Evaluation Understudy. 기계 번역 품질의 표준 metric (n-gram precision 기반). / Standard MT quality metric based on n-gram precision. |
| **Label Smoothing** | 정답 label에 확률 질량을 일부 분산시켜 과적합을 방지 ($\epsilon_{ls}=0.1$). / Distributes probability mass off the target label to prevent overconfidence. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

**한국어**: $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$. $QK^T$는 query-key 유사도 행렬 ($n \times m$), softmax로 각 query에 대한 key의 확률 분포를 얻고, 이를 Value에 곱하여 가중합을 계산. $\sqrt{d_k}$로 나누는 이유: $d_k$가 크면 내적 값의 분산이 커져 softmax가 saturated 영역으로 밀려나기 때문.

**English**: $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$. $QK^T$ is the $n \times m$ query-key similarity matrix; softmax converts each row into a probability distribution over keys, then multiplied with $V$ to get a weighted sum. The $\sqrt{d_k}$ scaling prevents the softmax from saturating when $d_k$ is large.

### (2) Multi-Head Attention

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V)
$$

**한국어**: $h=8$개의 헤드를 병렬로 실행. 각 헤드는 $d_k = d_v = d_{model}/h = 64$ 차원으로 사영. 서로 다른 헤드가 서로 다른 관계(문법, 의미, 공참조 등)를 학습한다고 알려져 있음.

**English**: Runs $h=8$ heads in parallel, each projecting to $d_k = d_v = d_{model}/h = 64$. Different heads capture different relational patterns (syntactic, semantic, coreference).

### (3) Positional Encoding

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

**한국어**: 위치 $pos$, 차원 인덱스 $i$. 주기가 $2\pi$부터 $2\pi \cdot 10000$까지 기하적으로 증가하는 sin/cos 함수. 선형 변환으로 상대 위치 $PE_{pos+k}$를 표현할 수 있다는 성질이 핵심.

**English**: Position $pos$, dimension index $i$. Sin/cos with wavelengths forming a geometric progression from $2\pi$ to $2\pi \cdot 10000$. Key property: $PE_{pos+k}$ is a linear function of $PE_{pos}$, enabling the model to attend to relative positions.

### (4) Position-wise Feed-Forward Network

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

**한국어**: 각 위치에 동일하게 적용되는 2-layer MLP. $d_{model}=512$ → $d_{ff}=2048$ → $d_{model}=512$. ReLU 활성화. 논문 parameter의 약 2/3가 여기에 있음.

**English**: A 2-layer MLP applied identically at each position: $d_{model}=512 \to d_{ff}=2048 \to d_{model}=512$ with ReLU. Contains about 2/3 of the model's parameters.

### (5) Learning Rate Schedule

$$
lrate = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5},\ step\_num \cdot warmup\_steps^{-1.5})
$$

**한국어**: Warmup 4000 step까지 선형 증가, 이후 step의 역제곱근으로 감소. Adam optimizer와 함께 사용.

**English**: Linear warmup for 4000 steps, then decay as inverse square root of step number. Used with Adam.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어

**읽기 순서 제안**:

1. **Abstract + §1 Introduction** — 문제 의식과 핵심 주장 파악 (왜 recurrence를 없애고 싶어하는지)
2. **§2 Background** — 기존 접근(ByteNet, ConvS2S)의 한계를 정확히 이해 (특히 "path length" 개념)
3. **§3.1 Encoder and Decoder Stacks** — 전체 아키텍처 그림을 머릿속에 그릴 것 (Figure 1이 핵심). 펜으로 그려보기를 권장
4. **§3.2 Attention** — ⭐ 가장 중요. Scaled Dot-Product Attention → Multi-Head Attention 순서로 이해
5. **§3.3 FFN, §3.4 Embeddings, §3.5 Positional Encoding** — 보조 구성 요소
6. **§4 Why Self-Attention** — Table 1이 핵심 (complexity per layer, path length 비교)
7. **§5 Training** — 하이퍼파라미터와 optimizer (Adam, warmup)
8. **§6 Results** — Table 2, 3, 4에서 BLEU 점수와 ablation study 확인

**체크포인트 질문** (읽으면서 스스로에게):
- Encoder의 각 layer 출력 shape는? ($n \times d_{model} = n \times 512$)
- Decoder의 masked self-attention에서 마스킹은 어떻게 구현되나? ($-\infty$를 softmax 입력에 더함)
- Multi-head가 왜 single-head보다 나은가? (Table 3 참고)
- $\sqrt{d_k}$로 나누지 않으면 어떤 문제가 생기나? (footnote 4)

### English

**Recommended reading order**:

1. **Abstract + §1 Introduction** — grasp the motivation
2. **§2 Background** — understand limitations of prior work, especially "path length"
3. **§3.1 Encoder/Decoder Stacks** — internalize Figure 1 (draw it by hand!)
4. **§3.2 Attention** — ⭐ most important. Scaled Dot-Product → Multi-Head
5. **§3.3–3.5** — supporting components
6. **§4 Why Self-Attention** — Table 1 is key
7. **§5 Training** — hyperparameters, Adam with warmup
8. **§6 Results** — Tables 2–4 for BLEU and ablations

**Checkpoint questions while reading**:
- What is the output shape of each encoder layer? ($n \times 512$)
- How is masking implemented in decoder self-attention? (adding $-\infty$ before softmax)
- Why is multi-head better than single-head? (see Table 3)
- What goes wrong without $\sqrt{d_k}$ scaling? (footnote 4)

---

## 7. 현대적 의의 / Modern Significance

### 한국어

**이 논문은 현대 AI의 출발점이라 해도 과언이 아닙니다.**

- **2018 BERT**: Transformer 인코더만 사용하여 양방향 언어 표현 학습. NLP 벤치마크를 휩쓺.
- **2018 GPT**: Transformer 디코더만 사용하여 autoregressive 생성 모델. → GPT-2, GPT-3, GPT-4, ChatGPT로 이어짐.
- **2020 ViT (Vision Transformer)**: Transformer가 이미지 처리에도 SOTA. CNN 독점 시대 종료.
- **2021 AlphaFold 2**: Transformer 기반 attention이 단백질 구조 예측 혁명.
- **2022~ 멀티모달**: CLIP, DALL·E, Stable Diffusion, Flamingo, GPT-4V 모두 Transformer 기반.
- **Scaling laws** (Kaplan et al., 2020; Chinchilla, 2022): Transformer를 크게 만들수록 성능이 예측 가능하게 증가. → 100B+ 파라미터 LLM의 이론적 근거.

**Solar Physics / Space Weather 연관성**: Transformer 기반 모델이 태양 활동 예측(HelioFM, FlareNet, SolarCNN-Transformer 하이브리드), 자기장 시계열 분석, 대기 시뮬레이션 surrogate model 등에 활발히 적용 중. 본인의 연구 분야에서도 이 논문의 지식은 핵심 도구가 됨.

### English

**This paper is arguably the starting point of modern AI.**

- **2018 BERT**: encoder-only Transformer → dominant in NLP benchmarks.
- **2018 GPT**: decoder-only Transformer → GPT-2/3/4, ChatGPT.
- **2020 ViT**: Transformers conquer computer vision, ending the CNN era.
- **2021 AlphaFold 2**: attention-based protein structure prediction.
- **2022+ Multimodal**: CLIP, DALL·E, Stable Diffusion, Flamingo — all Transformer-based.
- **Scaling laws** (Kaplan 2020, Chinchilla 2022) — predictable scaling motivates 100B+ parameter LLMs.

**Relevance to Solar Physics / Space Weather**: Transformer-based models are actively applied to solar activity forecasting (HelioFM, flare prediction), magnetic time-series analysis, and atmospheric surrogate models. This paper's knowledge is foundational for your research domain.

---

## Q&A

### Q1. 이 논문은 CNN 커널 자체도 attention으로 대체 가능하다고 말하고 있는 것 같다. 초고해상 영상(예: SDO 4096×4096)에도 실질적으로 가능한가?
### Q1. The paper seems to argue that attention can replace CNN kernels. Is this practically feasible for ultra-high-resolution images such as SDO 4096×4096?

**1) 논문이 실제로 주장하는 것 / What the paper actually claims**

Table 1에서 self-attention, recurrent, convolutional layer의 복잡도를 비교합니다 / Table 1 compares the complexity of self-attention, recurrent, and convolutional layers:

| Layer | Complexity per Layer | Max Path Length |
|-------|---------------------|-----------------|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(\log_k n)$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(n/r)$ |

논문의 주장은 "attention이 convolution을 그대로 대체 가능"이 아니라, **"토큰 간 정보 전달 경로(path length)가 $O(1)$이라 long-range dependency 포착에 유리하다"**에 가깝습니다. 저자들도 $n^2$ 항의 부담을 인지하고 "restricted self-attention"(local window)을 언급합니다.

The paper's claim is not "attention fully replaces convolution," but rather **"the path length between any two tokens is $O(1)$, which is favorable for capturing long-range dependencies."** The authors recognize the $n^2$ burden and mention *restricted self-attention* (local window) as a mitigation.

**2) SDO 4096×4096 에 순진하게 적용할 경우 / Naïve application to SDO 4096×4096**

픽셀 단위 토큰화 시 / With pixel-level tokenization:
- $n = 4096 \times 4096 \approx 1.68 \times 10^7$
- Attention matrix $QK^\top$: $n \times n \approx 2.8 \times 10^{14}$ entries
- FP16 기준 약 560 TB / ~560 TB at FP16 — per single head, per single layer

→ 현재 어떤 하드웨어로도 불가능 / **Infeasible on any existing hardware.**

**3) 현실적으로 가능하게 하는 방법 / Practical solutions**

- **(a) Patch embedding (ViT)**: 16×16 패치 → $n=65{,}536$, 여전히 무거움(~8.5 GB/head FP16). 32×32 패치 → $n=16{,}384$, ~540 MB/head → A100급 GPU에서 실용적 / 16×16 patches still heavy; 32×32 patches (~540 MB/head FP16) become feasible on A100-class GPUs.
- **(b) Windowed / hierarchical attention (Swin, NAT)**: window 크기 $w$ 안에서만 attention, $O(w^2 \cdot n)$. 계층적으로 global context 확보. 4k 영상에서 표준적으로 동작 / Local window attention with $O(w^2 \cdot n)$ cost; global context via hierarchical stages. Standard for 4k imagery.
- **(c) Linear attention (Performer, Linformer, Nyströmformer)**: kernel trick으로 $O(n)$ 달성 / Achieves $O(n)$ via kernel approximations.
- **(d) FlashAttention**: IO-aware tiling으로 메모리 절감 (복잡도는 동일) / IO-aware tiling reduces memory footprint (same asymptotic complexity).

**4) 태양물리 도메인에서의 실제 추세 / Current practice in solar physics**

1. **Downsampling** to 512² or 1024² for CNN/ViT (most flare-forecasting work)
2. **Active region crop** (SHARP cutouts, 256–512 px)
3. **Swin / hierarchical ViT** — adopted in recent solar segmentation and magnetogram super-resolution
4. **CNN + attention hybrid** — most practical under limited labeled data

**5) 결론 / Bottom line**

- 순수 pixel-level self-attention은 4096×4096에서 불가능 / Pure pixel-level self-attention is infeasible at 4096×4096.
- Patch + hierarchical/windowed attention 조합이면 충분히 가능하며 실제 사용 중 / Patch + hierarchical/windowed attention makes it tractable and is already in use.
- 제한된 labeled 태양 데이터에서는 inductive bias를 가진 CNN–attention hybrid가 데이터 효율 면에서 유리한 경우가 많음 / With limited labeled solar data, CNN–attention hybrids often win on data efficiency due to their locality/translation-equivariance bias.
- 이 확장은 Vaswani 2017의 범위가 아니며, **ViT (Dosovitskiy 2020)** 와 **Swin Transformer (Liu 2021)** 에서 본격화됨 / This extension is not the scope of Vaswani 2017; it is developed in **ViT (Dosovitskiy 2020)** and **Swin Transformer (Liu 2021)**.

---

### Q2. Attention 레이어(모듈)의 구조와 역할을 자세히 설명해 달라.
### Q2. Describe the structure and role of the attention layer (module) in detail.

**1) 핵심 아이디어 / Core idea**

Attention은 **"출력에 필요한 정보를 입력 시퀀스의 여러 위치에서 가중합으로 골라 가져오는 메커니즘"**입니다. SQL의 `SELECT value FROM table WHERE key = query`의 **연속·미분가능 버전**으로 볼 수 있습니다.
Attention is **"a mechanism that selectively pulls information from multiple input positions via a weighted sum to build an output."** Think of it as a **continuous, differentiable version** of `SELECT value FROM table WHERE key = query`.

**2) Scaled Dot-Product Attention (기본 단위 / basic unit)**

입력 행렬 / Input matrices:
- $Q \in \mathbb{R}^{n \times d_k}$ — "무엇을 찾고 싶은가" / "what am I looking for?"
- $K \in \mathbb{R}^{m \times d_k}$ — "무엇을 갖고 있나"의 색인 / index of "what do I have?"
- $V \in \mathbb{R}^{m \times d_v}$ — 실제 내용 / actual content

수식 / Formula:
$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

단계 / Steps:
1. **Similarity**: $S = QK^\top$ (dot product)
2. **Scaling** by $\sqrt{d_k}$ — $d_k$가 크면 dot product 분산이 커져 softmax가 포화됨(gradient 소실) / prevents softmax saturation when $d_k$ is large.
3. **Softmax** per row — 각 query가 key들에 주는 관심 분포 / attention weights per query.
4. **Weighted sum** $AV$ — value들의 가중 평균 / weighted average of values.

**3) Multi-Head Attention (모듈 수준 / module level)**

왜 여러 head? / Why multiple heads? — 단일 attention은 한 가지 관계만 평균적으로 포착 → 서로 다른 부분공간에서 병렬로 여러 attention을 실행해 다양한 관계(구문, coreference, 의미 유사성 등)를 포착.
A single attention captures only one averaged relation; multiple heads running in parallel in different subspaces capture diverse relations (syntactic, coreference, semantic, ...).

구조 / Structure (논문 설정: $d_{\text{model}}=512$, $h=8$, $d_k=d_v=64$):
- Head별 선형 투영 / Per-head linear projections: $Q_i = XW^Q_i$, $K_i = XW^K_i$, $V_i = XW^V_i$
- 각 head에서 Scaled Dot-Product Attention 실행 / Run scaled dot-product attention per head
- Concatenate → $W^O$로 출력 투영 / Concatenate → project with $W^O$

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \, W^O
$$

- **총 연산량은 single-head와 동일** (head 차원을 $d_{\text{model}}/h$로 분할) / Total FLOPs equal single-head since per-head dim is $d_{\text{model}}/h$.
- **표현력 증가** — head마다 서로 다른 관계 학습 / Each head can learn a distinct relation.

**4) Transformer에서 쓰이는 3가지 변형 / Three variants used in Transformer**

| Variant | Q | K, V | 역할 / Role |
|---|---|---|---|
| Encoder self-attention | encoder prev layer | same as Q | 입력 내 토큰 간 정보 교환 / within-input mixing |
| Decoder masked self-attention | decoder prev layer | same as Q | 출력 토큰 간 정보 교환 + 미래 mask / autoregressive masking |
| Encoder–Decoder (cross) attention | decoder current | encoder final output | 출력이 입력의 어디를 볼지 결정 / decoder attends to encoder |

Masking in decoder self-attention: $S_{ij} = -\infty$ for $j > i$ — 미래 토큰 차단 / blocks future tokens.

**5) 핵심 성질 / Key properties**

1. 가변 길이 처리 / Handles variable lengths ($n, m$).
2. **Permutation-equivariant** — 위치 정보 없음 → positional encoding 필요 / no built-in positional info.
3. **Path length $O(1)$** — 임의 두 토큰이 한 layer 안에서 직접 상호작용 / any two tokens interact in one layer.
4. **완전 병렬화 / Fully parallelizable** — 시간 축 순차 의존 없음 (RNN과 대조) / no sequential dependency.
5. 해석 가능성 / Interpretability — attention weight 시각화 가능 / attention maps are visualizable.

**6) Block 내 관계 / Within the Transformer block**

$$
\begin{aligned}
X' &= \text{LayerNorm}(X + \text{MultiHead}(X)) \\
X'' &= \text{LayerNorm}(X' + \text{FFN}(X'))
\end{aligned}
$$

- Attention = **cross-token mixing** (토큰 간 정보 교환)
- FFN = **cross-feature mixing per token** (채널 간 정보 교환)

Attention이 "누구와 이야기할지"를, FFN이 "이야기한 내용을 어떻게 해석할지"를 담당. / Attention decides *who to talk to*; FFN decides *how to interpret what was said*.

---

### Q3. Query와 Key의 차이는 무엇인가?
### Q3. What is the difference between Query and Key?

**1) 한 줄 요약 / One-line summary**

- **Query**: 내가 **찾고 싶은 것**의 표현 (능동적, 질문하는 쪽) / representation of *what I am looking for* (active, the asker).
- **Key**: 내가 **제공할 수 있는 것**의 표현 (수동적, 색인되는 쪽) / representation of *what I offer*, used as an index (passive, the one being matched).

둘 다 같은 입력 $x$에서 나오지만 **서로 다른 선형 변환** $W^Q$, $W^K$를 거쳐 **역할이 다른 공간**으로 투영됨.
Both come from the same input $x$ but are projected through **different linear maps** $W^Q$, $W^K$ into **spaces with different roles**.

**2) 왜 분리하는가? / Why separate them?**

**(a) 관계의 비대칭성 / Asymmetry of relations**
만약 $Q = K$라면 $S_{ij} = x_i \cdot x_j = S_{ji}$ → 대칭 행렬 → **방향성 있는 관계 학습 불가**.
If $Q = K$, then $S_{ij} = S_{ji}$, a symmetric matrix — **directional relations cannot be learned**.
분리된 $W^Q, W^K$로 $S_{ij} = (x_i W^Q)(x_j W^K)^\top \neq S_{ji}$ → 비대칭 관계 학습 가능.
With distinct $W^Q, W^K$, the score is asymmetric, enabling directional learning.

**(b) Self-similarity 폭주 방지 / Avoid trivial self-attention**
$Q = K = x$이면 자기 자신과의 내적 $\|x_i\|^2$이 항상 커서 자기 자신에게 쏠림 → trivial solution.
With $Q = K$, each token has huge self-similarity $\|x_i\|^2$, collapsing attention onto the token itself.

**(c) 서로 다른 subspace 활용 / Different subspaces**
$W^Q$는 "지금 결정에 필요한 특징", $W^K$는 "어떻게 검색당할지"의 특징으로 투영 — 같은 단어라도 query/key 역할에서 강조하는 속성이 다름.
$W^Q$ projects into features needed *for the current decision*; $W^K$ projects into features describing *how to be retrieved*. The same word can encode different aspects when acting as query vs. key.

**3) DB 비유 / Database analogy**

| 요소 | 전통 DB / Traditional DB | Attention |
|------|-------|-----------|
| Query | 문자열 `"Paris"` | 벡터 $q = xW^Q$ |
| Key | row의 key 컬럼 | 벡터 $k_j = x_j W^K$ |
| 매칭 / Matching | 완전 일치 (binary) | dot product + softmax (soft) |
| 결과 / Result | 일치하는 row의 value | 모든 value의 가중합 / weighted sum |

Query는 *"나는 이런 정보를 원해"*, Key는 *"나는 이런 정보의 열쇠야"*를 인코딩.
Query encodes *"what I want"*; key encodes *"what I am indexed as"*.

**4) Value와의 구분 / Separation from Value**

Key = "나를 **검색하는 기준**" (색인용) / indexing representation.
Value = "나를 검색했을 때 **실제 가져갈 내용**" / actual content returned once matched.
비유: 도서관에서 제목/태그(key)로 찾고, 가져가는 건 책 내용(value). 검색 인덱스와 실제 페이로드의 최적화 목표가 다름.
Analogy: in a library you search by title/tags (key) but take home the book contents (value); the optimal representations for indexing vs. payload differ.

차원 / Dimensionality: $Q, K \in \mathbb{R}^{d_k}$ (내적을 위해 같아야 함 / must match for dot product), $V \in \mathbb{R}^{d_v}$ (일반적으로 독립 / independent in general).

**5) 예시 / Example**

*"The cat sat on the mat because **it** was soft."*
- `it`의 Query: "3인칭 단수 무생물, 'soft'한 것은?" / "3rd-person singular, inanimate, something 'soft'?"
- `mat`의 Key: "명사, 무생물, 표면" / "noun, inanimate, surface"
- `cat`의 Key: "명사, 생물, 동물" / "noun, animate, animal"
→ `it`의 Query와 `mat`의 Key가 더 잘 매칭 → `mat`에 높은 attention. / The `it`-query matches the `mat`-key better → higher attention on `mat`.

같은 `mat`이라도 Key 역할일 때와 (만약 Query가 된다면) Query 역할일 때 표현하는 특징이 다름.
The same token `mat` encodes different features when acting as a key vs. as a query.

**6) 핵심 요지 / Bottom line**

Query와 Key는 같은 재료에서 만들어지지만 **"찾는 자"와 "찾히는 자"라는 비대칭 역할**을 수행하도록 **독립된 선형 변환으로 투영**됨. 이 분리 덕분에 Transformer는 방향성 있고 풍부한 관계를 학습할 수 있음.
Query and key come from the same source but are projected by **independent linear maps** to play the **asymmetric roles of "the searcher" vs. "the searched"**. This separation enables the Transformer to learn directional, rich relations.

---

### Q4. Figure 1의 구조를 자세히 설명하고, 각 모듈의 역할(혹은 기대하는 역할)도 설명해 달라.
### Q4. Explain the structure of Figure 1 in detail, and describe the role (or expected role) of each module.

**1) 전체 구조 / Overall structure**

좌측 **Encoder (N=6)**와 우측 **Decoder (N=6)**가 쌍을 이루는 encoder–decoder 구조. 번역 태스크 기준으로 입력=원문, 출력=번역문.
Left **Encoder (N=6)** and right **Decoder (N=6)** form an encoder–decoder pair. For translation: input=source, output=target.

```
Inputs → Embed ⊕ PE → [Encoder × 6] → Z
                                       ↓ (K,V for cross-attn)
Outputs(shifted) → Embed ⊕ PE → [Decoder × 6] → Linear → Softmax → P(y_t|...)
```

**2) 입력 처리부 / Input processing**

- **Input Embedding**: vocab id → $\mathbb{R}^{d_{\text{model}}=512}$ 벡터. 의미 유사 단어가 가까워지도록 학습. / Lookup table mapping token ids to 512-dim vectors.
- **Positional Encoding**: sinusoidal $PE_{(pos,2i)}=\sin(pos/10000^{2i/d})$, $PE_{(pos,2i+1)}=\cos(\dots)$. Embedding에 **더함**. / Added (not concatenated) to embeddings. 역할: attention은 permutation-equivariant이라 위치 정보가 없으므로 이를 주입 / injects positional info since attention is permutation-equivariant.
- **Output Embedding (shifted right)**: teacher forcing — target 앞에 `BOS` 붙이고 마지막 토큰 제거. / Shifted-right form used for teacher forcing.

**3) Encoder layer (두 sub-layer, residual+LayerNorm) / Encoder layer (two sub-layers with residual+LayerNorm)**

각 sub-layer: $\text{LayerNorm}(x + \text{Sublayer}(x))$.

- **Sub-layer 1: Multi-Head Self-Attention** — $Q=K=V$ 모두 이전 layer 출력, mask 없음. 역할: 입력 시퀀스 내 토큰 간 정보 교환, 문맥에 맞게 표현 갱신. 기대: 구문 의존, 형용사-명사 수식, coreference 등 관계를 여러 head가 분담 학습. / Self-attention mixes info across input tokens; heads are expected to specialize (syntactic deps, coreference, ...).
- **Sub-layer 2: Position-wise FFN** — $\text{FFN}(x)=\max(0, xW_1+b_1)W_2+b_2$, $W_1\in\mathbb{R}^{512\times2048}$. 위치별 독립 적용(= 1×1 conv). 역할: 채널축 비선형 변환으로 attention 결과를 "소화". 기대: 사실 지식/특화 feature 저장. / Per-position 2-layer MLP; feature-axis nonlinear transform that "digests" attention output.
- **Add & Norm**: residual은 gradient highway + identity-first bias, LayerNorm은 활성 분포 안정화 → 깊게 쌓아도 학습 가능. / Residual + LayerNorm enable deep stacking.

**4) Decoder layer (세 sub-layer) / Decoder layer (three sub-layers)**

- **Sub-layer 1: Masked Multi-Head Self-Attention** — causal mask ($S_{ij}=-\infty$ for $j>i$). 역할: 지금까지 생성한 출력 토큰 간 정보 교환, 미래 차단. 필요 이유: autoregressive 생성 일관성 + train/inference 동일 조건 보장. / Prevents peeking at future tokens; maintains autoregressive consistency.
- **Sub-layer 2: Encoder–Decoder (Cross) Attention** — $Q$=decoder 이전 출력, $K,V$=**encoder 최종 출력**, mask 없음. 역할: 번역문 한 토큰을 만들 때 원문의 어느 부분에 주목할지 결정(Bahdanau 2014의 일반화). 기대: 원문–번역문 alignment, 구(phrase) 수준 대응 학습. / Decoder attends to encoder output; learns source–target alignment.
- **Sub-layer 3: Position-wise FFN** — encoder FFN과 동일 구조, 별도 파라미터. / Same structure as encoder FFN, separate parameters.

**5) 출력부 / Output head**

- **Linear**: $W_{\text{out}}\in\mathbb{R}^{d_{\text{model}}\times V}$. 원 논문에서 input/output embedding과 weight tying. / Projects hidden to vocab logits; weight-tied with embeddings.
- **Softmax**: 위치별 다음 토큰 확률분포 $P(y_t|y_{<t},x)$. / Per-position next-token distribution. Loss: cross-entropy with label smoothing $\epsilon=0.1$.

**6) 학습 시 데이터 흐름 / Training data flow**

1. 원문 → Embed+PE → Encoder 6층 → memory $Z$ / Source → encoder → memory $Z$.
2. 타깃(shifted right) → Embed+PE → Decoder 6층 (masked self → cross-attn with $Z$ → FFN) / Target → decoder stack using $Z$.
3. Linear → Softmax → CE loss with target / Linear → softmax → cross-entropy.

**7) 모듈별 역할 요약 / Module role summary**

| 모듈 / Module | 역할 / Role (비유 / analogy) |
|---|---|
| Input Embedding | 단어→좌표 / word → coordinate |
| Positional Encoding | "몇 번째 줄" 태그 / row-number tag |
| Encoder Self-Attn | 원문 내부 관계 정리 / within-source relations |
| Encoder FFN | 정리된 의미 재해석 / feature-wise reinterpretation |
| Masked Self-Attn (Dec) | 번역문 내부 일관성 / within-target consistency |
| Cross-Attn | 원문-번역문 정렬 / source–target alignment |
| Decoder FFN | 번역 문장으로 정제 / target-side refinement |
| Linear + Softmax | 다음 토큰 선택 / next-token selection |
| Residual | 필요 시에만 수정 / identity-first bias |
| LayerNorm | 신호 크기 안정화 / activation stabilization |

**8) 설계 철학 / Design philosophy**

1. 완전 병렬화 — RNN의 시간 축 순차 의존 제거. / Fully parallel, unlike RNN.
2. $O(1)$ path length — 임의 두 토큰 직접 상호작용. / Any two tokens interact in one layer.
3. 모듈 재사용 — 같은 MHA 블록이 Q/K/V 출처만 바꿔 3가지 용도(encoder self / decoder self / cross). / Same MHA block reused for three roles.
4. Residual + LayerNorm으로 깊이 확장성 확보 → 이후 GPT/BERT로 연결. / Enables scaling, paving the way for GPT/BERT.

**9) 이 구조의 유산 / Legacy**

- Encoder-only → BERT 계열 (이해) / BERT family (understanding)
- Decoder-only → GPT 계열 (생성) / GPT family (generation)
- Encoder-Decoder → T5, BART (seq2seq)

---

### Q5. FFN (Position-wise Feed-Forward Network)의 구조와 역할은?
### Q5. What is the structure and role of the FFN (Position-wise Feed-Forward Network)?

**1) 수식과 구조 / Formula and structure**

$$
\text{FFN}(x) = \max(0,\, xW_1 + b_1)\, W_2 + b_2
$$

- $x \in \mathbb{R}^{d_{\text{model}}=512}$
- $W_1 \in \mathbb{R}^{512 \times 2048}$, $W_2 \in \mathbb{R}^{2048 \times 512}$
- 활성 / activation: **ReLU**
- 형태 / shape: 2-layer MLP, hidden $d_{ff}=2048$ = $4 \cdot d_{\text{model}}$

**2) "Position-wise"의 의미 / Meaning of "position-wise"**

같은 $W_1, W_2$를 모든 토큰 위치에 동일 적용 — 토큰별 독립. **Kernel size=1 1D conv와 등가** (논문 명시).
The same weights are applied independently at every position — equivalent to **two 1×1 1D convolutions**.
→ FFN은 **토큰 간 정보를 섞지 않음**, 토큰 **내부**(채널 축)만 변환.
→ FFN does **not** mix across tokens; it only transforms **within** each token (feature axis).

**3) Transformer 블록 내 역할 분담 / Role within the block**

| Module | Mixing axis | 역할 / Role |
|---|---|---|
| Multi-Head Attention | 토큰 간 / cross-token | 문맥 수집 / gather context |
| FFN | 채널 간·토큰 내 / cross-feature, per-token | 문맥 해석 / digest context |

"섞고 → 소화" 패턴을 $N$번 반복 / "mix → digest" repeated $N$ times — Transformer의 본질.

**4) 왜 2-layer인가? / Why 2 layers?**

1-layer linear는 앞뒤 선형 투영에 흡수되어 무의미. 최소 2-layer + 비선형 활성이 있어야 함수 근사 능력(비선형성)이 확보됨.
A single linear layer collapses into surrounding projections; at least 2 layers with nonlinearity are needed to add approximation power.

**5) 왜 $d_{ff} = 4 d_{\text{model}}$인가? / Why expand by 4×?**

- **Overcomplete representation** — ReLU가 뉴런 절반 정도를 끄므로 출력에서 유효 뉴런 수를 맞추려면 중간 폭이 필요 / ReLU sparsifies, so more hidden units maintain effective capacity.
- **파라미터 분배 / Parameter budget**: block 파라미터의 약 2/3이 FFN → 용량이 여기 집중 / FFN holds ~2/3 of block parameters — capacity concentrates here.
- **병목-해제-병목 / Bottleneck release**: residual stream 차원($d_{\text{model}}$) 제약을 잠시 해제 / temporarily relax the residual-stream dimension.

한 블록 파라미터 개략 / Per-block param count: MHA $\approx 4 d_{\text{model}}^2 \approx 1.05\text{M}$, FFN $\approx 2 d_{\text{model}} d_{ff} \approx 2.1\text{M}$.

**6) 현대적 해석 / Modern interpretations**

- **Key-Value Memory view** (Geva 2021): $W_1$ 행 = key, $W_2$ 행 = value, ReLU = soft match → FFN은 학습된 KV memory / FFN behaves as a learned key-value memory.
- **사실 지식 저장 / Factual knowledge**: Attention이 "어디를 볼지"라면 FFN은 "무엇을 아는지"를 담당. 특정 뉴런을 편집해 지식을 수정하는 기법(ROME, MEMIT) / FFN neurons encode factual knowledge; editable by ROME/MEMIT.
- **Superposition**: $d_{ff} > d_{\text{model}}$ 여유로 수많은 feature를 중첩 표현 / extra dim allows superposed feature storage.

**7) Dropout**

원 논문: FFN 출력에 $P=0.1$ dropout → $\text{LayerNorm}(x + \text{Dropout}(\text{FFN}(x)))$. 큰 파라미터로 인한 과적합 억제.
Dropout $P=0.1$ applied after FFN to regularize its large capacity.

**8) 계산량 / Computation**

토큰당 FFN: $2 \cdot d_{\text{model}} \cdot d_{ff} \approx 2.1$ MFLOPs. Attention은 $\sim n \cdot d_{\text{model}}$ (시퀀스 길이 $n$에 비례) — **짧은 시퀀스에서는 FFN 지배, 긴 시퀀스에서는 attention 지배**.
Per-token FFN cost is fixed (~2.1 MFLOPs); attention grows with $n$. **FFN dominates short sequences; attention dominates long ones.**

**9) 이후 진화 / Later variants**

- **GLU / SwiGLU** (LLaMA, PaLM): $(xW_1 \odot \text{Swish}(xW_g)) W_2$ — gating이 ReLU보다 경험적으로 우세 / gated activation outperforms ReLU empirically; de-facto standard in modern LLMs.
- **GELU** (BERT, GPT-2): $x\Phi(x)$ — 부드러운 ReLU / smooth ReLU.
- **Mixture of Experts (MoE)**: 여러 FFN 중 router가 일부만 활성화 → 파라미터 ↑, 연산량 유지 / sparse expert routing decouples parameters from compute (Switch, Mixtral).

**10) 한 줄 요약 / Bottom line**

Attention이 "누구의 정보를 얼마나 가져올지"를 결정한다면, FFN은 "가져온 정보를 각 토큰 내부에서 어떻게 비선형적으로 변환·해석할지"를 담당.
If attention decides *whose information to pull and how much*, FFN decides *how to nonlinearly transform and interpret that information within each token*.

---

### Q6. Figure 1에서 output probabilities 로 가는 신호가 inputs와 outputs 둘 다에서 오는 것으로 보인다. test(추론) 시에는 어떻게 달라지나?
### Q6. In Figure 1 the output probabilities depend on both inputs and outputs — how does this differ at test/inference time?

**1) 혼동의 근원 / Source of confusion**

Figure 1의 "Outputs (shifted right)"는 **학습 시점의 그림**이다. 추론 시에는 타깃이 없으므로 decoder 입력을 **스스로 한 토큰씩 만든다** (autoregressive).
Figure 1 depicts the **training-time** configuration. At inference there is no target, so the decoder input is built **autoregressively**, one token at a time.

**2) 학습 시 — Teacher forcing + Causal mask / Training: teacher forcing with causal mask**

번역 예: 원문 `[I, love, you]`, 타깃 `[Ich, liebe, dich]`.
- Encoder 입력 / Encoder input: `[I, love, you]`
- Decoder 입력 (shifted right) / Decoder input: `[BOS, Ich, liebe, dich]` — **한 번에 전부** / all at once
- 예측 타깃 / Prediction targets: `[Ich, liebe, dich, EOS]`

병렬 처리의 비밀은 **masked self-attention** — attention score 행렬이 하삼각 / lower-triangular:
$$
S_{ij} = -\infty \quad \text{for } j > i
$$

위치 $t$의 예측은 위치 $\le t$의 토큰만 참조하므로 전체를 한 번에 forward해도 수학적으로 autoregressive 생성과 동일.
The prediction at position $t$ depends only on tokens at positions $\le t$, so a single parallel forward pass is mathematically equivalent to sequential generation.

→ RNN seq2seq은 $T$번 순차 forward 필요, Transformer는 학습 시 **1번의 병렬 forward**로 모든 위치 동시 학습 → GPU 효율.
→ Unlike RNNs (T sequential passes), Transformer trains all positions in **one parallel pass**.

**3) 추론 시 — Autoregressive generation / Inference: autoregressive generation**

타깃이 없음 → `BOS`로 시작해 한 토큰씩 생성하며 누적 / Start from `[BOS]` and generate token by token:

```
Step 0: Encoder([I, love, you]) → Z                 (1번만 계산 / computed once)
Step 1: Decoder([BOS])                  → "Ich"
Step 2: Decoder([BOS, Ich])             → "liebe"
Step 3: Decoder([BOS, Ich, liebe])      → "dich"
Step 4: Decoder([BOS, Ich, liebe, dich]) → "EOS" (종료 / stop)
```

**4) 학습 vs 추론 비교 / Training vs inference**

| 항목 / Item | 학습 / Training | 추론 / Inference |
|---|---|---|
| Target 가용성 / Target availability | 있음 (ground truth) | 없음 — 직접 생성 / generated on the fly |
| Decoder 입력 / Decoder input | `[BOS, y_1, …, y_{T-1}]` 한 번에 / all at once | 이전 생성 토큰 누적 / accumulated generated tokens |
| Forward 횟수 / Forward passes | 1번 (병렬) / one (parallel) | $T$번 (순차) / $T$ sequential passes |
| Causal mask | 필요 (정답 누설 방지) / prevents peeking | 필요 (어차피 미래 없음) / still applied |
| Encoder forward | 1번 / once | 1번 재사용 / once, reused |

**5) 실전 최적화 — KV Cache / Practical optimization: KV cache**

매 스텝마다 `[BOS, …, y_{t-1}]` 전체를 다시 forward하면 낭비 — 과거 토큰의 $K, V$는 불변.
Recomputing past $K, V$ at every step is wasteful since they do not change.

**KV cache**: 각 layer에서 과거 $K, V$를 저장하고, 새 토큰 $y_t$의 $Q, K, V$만 계산 → attention은 새 $Q$와 누적 $K, V$로 수행. 스텝당 연산 $O(t) \to O(1)$ (시퀀스 길이에 대해). 현대 LLM inference가 이 덕분에 빠름.
Cache past $K, V$ per layer; compute only the new token's $Q, K, V$. Reduces per-step cost from $O(t)$ to $O(1)$ w.r.t. sequence length. Core to fast modern LLM inference.

**6) Softmax → 토큰 결정 / Turning softmax into tokens**

| 방법 / Method | 설명 / Description | 특성 / Property |
|---|---|---|
| Greedy | argmax | 결정론적, 반복 경향 / deterministic, repetitive |
| Beam search | top-$k$ 부분가설 유지 / keep top-$k$ partials | 품질↑ 다양성↓ / higher quality, less diverse |
| Temperature | logits$/\tau$ | $\tau<1$ 뾰족 / sharper, $\tau>1$ 평평 / flatter |
| Top-$k$ / top-$p$ | 상위 $k$개 / 누적확률 $p$에서 샘플 | 다양성 제어 / diversity control |

원 논문: WMT 번역에서 beam size 4, length penalty 사용.
The paper uses beam search (size 4) with length penalty for WMT translation.

**7) 비유 / Analogy**

- 학습 = 컨닝 허용 시험 — 답안지 전체를 받되 각 문항에서 그 이전 답만 참조(mask), 한 번에 전부 채점.
- 추론 = 실제 시험 — 1번부터 순서대로 풀고, 자기 답을 다음 문항의 입력으로 사용.
- Training = open-book exam where every question may only peek at previous answers (mask); all graded at once.
- Inference = real exam, solved sequentially, each answer fed into the next question.

**8) 핵심 요지 / Bottom line**

Figure 1은 학습 시의 그림이다. 추론 시 decoder 입력은 모델이 스스로 만든 토큰들의 누적이며, 이 피드백 루프가 그림에는 명시되지 않았다. Causal mask 덕분에 학습(병렬)과 추론(순차)이 수학적으로 일치한다.
Figure 1 shows training-time flow. At inference, the decoder input is the model's own accumulated output — an autoregressive feedback loop not drawn in the figure. The causal mask makes parallel training and sequential inference mathematically equivalent.

---

### Q7. Positional Encoding의 역할, 그리고 한 주기가 의미하는 바는?
### Q7. What is the role of positional encoding, and what does "one period" mean?

**1) 왜 필요한가 / Why it is needed**

Self-attention은 permutation-equivariant — 토큰 순서를 바꾸면 출력 순서만 바뀔 뿐 관계는 동일. "개가 사람을 물었다"와 "사람이 개를 물었다"를 구분 못 함. 각 위치에 고유한 벡터를 embedding에 더해 위치 정보를 주입.
Self-attention is permutation-equivariant: reordering tokens only reorders outputs without changing relations. Adding a unique vector per position to the embedding injects positional information.

$$
x'_{pos} = \text{Embedding}(\text{token}_{pos}) + PE_{pos}
$$

**2) 수식 / Formula**

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

같은 $i$끼리 sin/cos **쌍**. / Each $i$ gives a sin/cos **pair**.

**3) 한 주기의 의미 / Meaning of one period**

차원 쌍 $i$의 주기:
$$
T_i = 2\pi \cdot 10000^{2i/d_{\text{model}}}
$$

| $i$ | $2i/d_{\text{model}}$ | 주기 / Period (positions) |
|---|---|---|
| 0 | 0 | $2\pi \approx 6.28$ |
| 64 | 0.25 | ≈ 62.8 |
| 128 | 0.50 | ≈ 628 |
| 192 | 0.75 | ≈ 6,283 |
| 255 | ~1.0 | ≈ 62,832 |

주기 = 그 차원이 값을 한 바퀴 돌려 원점에 복귀하는 위치 간격.
Period = the position interval over which that dimension completes a full cycle.

- 짧은 주기(낮은 $i$): 빠르게 진동 → **미세한(local) 위치 정보** / encodes fine-grained (local) position.
- 긴 주기(높은 $i$): 느리게 변화 → **거친(global) 위치 정보** / encodes coarse (global) position.

비유 / Analogy: **서로 다른 속도로 도는 512개의 시곗바늘** — 초침부터 시침을 넘어 훨씬 느린 "달력 바늘"까지. 이들의 조합이 각 위치의 고유한 지문이 됨. 이진법 표현의 연속 버전과 유사.
A "set of 512 clock hands" rotating at geometrically spaced speeds; their joint state uniquely fingerprints each position — a continuous analogue of a binary place-value representation.

**4) 왜 base 10000? / Why base 10000?**

$i=d_{\text{model}}/2$에서 주기 ≈ 62,832 → 예상 최대 시퀀스 길이보다 훨씬 긴 주기로 aliasing 방지. 너무 작으면 긴 시퀀스에서 위치 혼동, 너무 크면 변별력 저하.
At $i=d_{\text{model}}/2$, period ≈ 62,832 — much longer than expected sequence lengths, avoiding aliasing. Too small causes collisions; too large reduces discriminability.

**5) 왜 sin/cos 쌍? — 핵심 수학적 성질 / Why sin/cos pairs? — key property**

임의의 고정된 offset $k$에 대해 $PE_{pos+k}$는 $PE_{pos}$의 **선형 함수**:
For any fixed offset $k$, $PE_{pos+k}$ is a **linear function** of $PE_{pos}$:

$$
\begin{pmatrix} PE_{pos+k,2i} \\ PE_{pos+k,2i+1} \end{pmatrix}
=
\underbrace{\begin{pmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{pmatrix}}_{\text{rotation by }k\omega_i}
\begin{pmatrix} PE_{pos,2i} \\ PE_{pos,2i+1} \end{pmatrix}
$$

→ **상대 위치가 고정된 회전 행렬로 표현** → attention이 "거리 기반 관계"를 쉽게 학습 ("주어-동사가 3칸 떨어져 있다" 같은 패턴을 pos에 무관하게 포착).
→ **Relative position becomes a fixed rotation**, letting attention learn distance-based patterns invariant to absolute position (e.g., "subject-verb 3 tokens apart").

논문 3.5절의 "attend by relative positions" 주장의 근거. / This is the mechanism behind the paper's claim about relative-position attending.

**6) 왜 더하는가? / Why added (not concatenated)?**

- 파라미터 수 불변 / Keeps $d_{\text{model}}$ unchanged.
- Residual stream 공존 — 의미 벡터와 위치 벡터가 같은 공간에 살고 모델이 차원 분할을 학습 / embedding and PE share the residual stream; model can allocate dimensions between semantics and position.
- 실험적으로 concat과 비슷한 성능, 더 단순 / empirically on par with concat but simpler.

**7) Sinusoidal vs Learned PE**

논문이 둘 다 시도 — 성능 비슷. Sinusoidal 선택 이유 / Chose sinusoidal because:
- 학습 시 본 적 없는 길이로 extrapolation 가능 (공식이라 임의 pos 계산 가능) / extrapolates to unseen lengths by formula.
- 파라미터 절약 / saves parameters.

현실 / In practice: extrapolation은 잘 안 됨 → 후속 변형 등장. / Real extrapolation is limited, motivating later variants.

**8) 현대적 진화 / Modern evolutions**

- **RoPE** (LLaMA, GPT-NeoX): PE를 Q, K에 직접 회전으로 적용 — 상대 위치가 내적에 자연스럽게 반영 / Rotates Q, K; relative position enters the inner product natively. 사실상 현대 LLM 표준 / de-facto standard.
- **ALiBi**: attention score에 거리 비례 bias — 긴 시퀀스 extrapolation에 강함 / linear distance bias on attention scores; strong length extrapolation.
- **T5-style Relative PE** (Shaw 2018): query-key 상대 거리에 학습된 bias / learned relative-distance bias.
- **NoPE**: decoder-only에서는 causal mask만으로 위치가 암묵적으로 학습된다는 증거 / causal mask alone may implicitly encode position in decoder-only models.

**9) 한 줄 요약 / Bottom line**

PE는 "기하급수적으로 다른 속도의 sin/cos 시곗바늘" 집합으로 각 위치에 고유한 지문을 찍고, sin/cos 쌍 구조 덕분에 **상대 위치가 선형(회전)으로 표현**되어 attention이 거리 기반 관계를 쉽게 학습하도록 돕는 장치.
PE is a set of sin/cos "clock hands" at geometrically spaced speeds that give every position a unique fingerprint; the sin/cos pair structure makes **relative positions expressible as fixed rotations**, enabling attention to learn distance-based relations easily.
