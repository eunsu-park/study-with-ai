---
title: "Layer Normalization"
authors: Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey Hinton
year: 2016
journal: "arXiv preprint (NeurIPS 2016 Deep Learning Symposium)"
doi: "arXiv:1607.06450"
topic: Artificial Intelligence / Deep Learning
tags: [layer-normalization, normalization, rnn, lstm, transformer, invariance, batch-size-independence, covariate-shift]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 21. Layer Normalization / 레이어 정규화

---

## 1. Core Contribution / 핵심 기여

Layer Normalization(LN)은 Batch Normalization(BN, #18)의 **배치 통계 의존성**이라는 근본적 한계를 해결하기 위해 제안된 정규화 기법이다. 핵심 통찰은 단순하다 — BN이 "미니배치 내 샘플들에 걸쳐" 평균/분산을 계산하는 반면, LN은 "**단일 샘플 내 모든 hidden unit에 걸쳐**" 평균/분산을 계산한다. 즉 같은 행렬의 평균을 다른 축으로 구할 뿐이지만, 이 축 선택 하나가 네 가지 중요한 결과를 만든다: (1) **배치 크기와 완전히 독립** (배치 크기 1에서도 작동), (2) **훈련/추론 동작이 동일** (running statistics 불필요), (3) **RNN의 각 time step에 자연스럽게 적용** (시퀀스 길이 가변성 문제 해결), (4) **분산 훈련 시 cross-GPU 통계 동기화 불필요**. 논문은 이를 RNN 중심의 5개 실험 (Attentive Reader, skip-thought, 기계 번역 RNN, order embeddings, MNIST MLP 생성모델)에서 입증하며, LN이 BN보다 **더 빠른 수렴과 더 나은 최종 성능**을 보임을 보였다. 2017년 Transformer 이후 LN은 NLP의 표준이 되었고, GPT/BERT/LLaMA/Claude 등 오늘날 모든 대형 언어 모델의 기본 정규화 레이어로 자리잡았다. RMSNorm, GroupNorm, InstanceNorm 등 후속 변형의 원조이기도 하다.

Layer Normalization (LN) addresses the fundamental limitation of Batch Normalization (BN, paper #18) — dependence on batch statistics. The insight is simple: where BN averages across samples in a mini-batch, LN averages **across all hidden units within a single sample**. It's the same tensor averaged over a different axis — but this axis choice produces four crucial consequences: (1) complete independence from batch size (works at batch size 1); (2) identical training and inference behavior (no running statistics needed); (3) natural per-timestep application in RNNs, handling variable sequence lengths; (4) no cross-GPU statistics synchronization in distributed training. The paper validates this with five RNN-centric experiments (Attentive Reader, skip-thought, NMT, order embeddings, generative MNIST MLP), showing LN achieves both **faster convergence and better final accuracy** than BN. After Transformer (2017), LN became the NLP standard, and today it is the default normalization layer of every large language model — GPT, BERT, LLaMA, Claude. LN is also the ancestor of a family of successors: RMSNorm, GroupNorm, InstanceNorm.

---

## 2. Reading Notes / 읽기 노트

### Section 1: Introduction / 도입

**BN의 성공과 한계**
BN(Ioffe & Szegedy 2015)은 CNN 훈련을 극적으로 가속시키며 딥러닝의 표준이 되었다. 그러나 BN은 두 가지 근본적 한계를 가진다:
1. **미니배치 통계에 의존** — 작은 배치에서는 통계가 부정확, 배치 1에서는 분산이 0.
2. **RNN에 적용이 어려움** — 각 time step마다 다른 통계를 별도로 추적해야 하고, 시퀀스 길이가 샘플마다 다르면 time step별 통계가 의미를 잃음.

BN (Ioffe & Szegedy, 2015) dramatically accelerated CNN training and became standard, but has two fundamental limits:
1. **Depends on minibatch statistics** — inaccurate for small batches, variance=0 at batch size 1.
2. **Awkward for RNNs** — requires tracking separate statistics per time step, which breaks with variable-length sequences.

**제안된 해결책: 축을 뒤집기**
LN의 아이디어는 정규화 축을 근본적으로 다르게 선택한다. BN이 "같은 feature 차원의 값들을 배치 축으로 평균"이라면, LN은 "같은 샘플의 모든 feature 값을 feature 축으로 평균"이다. 이 축 변경이 RNN, 배치 크기 1, online learning 모두를 가능하게 한다.

LN chooses a fundamentally different axis. Where BN averages along the batch dimension for a fixed feature, LN averages along the feature dimension for a fixed sample. This axis flip enables RNNs, batch size 1, and online learning.

### Section 2: Background — Batch Normalization / 배치 정규화 복습

논문은 BN 수식을 빠르게 복습한다. Hidden layer의 $i$번째 unit에 대해 summed input $a_i = \mathbf{w}_i^\top \mathbf{h}$를 받는다고 하자.

BN의 정규화 ($x$의 배치 축 통계):
$$
\bar{a}_i = \frac{g_i}{\sigma_i}(a_i - \mu_i)
$$

여기서
$$
\mu_i = \mathbb{E}_{x \sim P(x)}[a_i], \qquad \sigma_i = \sqrt{\mathbb{E}_{x \sim P(x)}[(a_i - \mu_i)^2]}
$$

**핵심**: $\mu_i, \sigma_i$는 **각 unit마다 다르게**, **데이터 분포 $P(x)$ 기대값**으로 정의된다. 실제로는 미니배치에서 근사한다. 따라서 배치 크기와 훈련/추론 괴리 문제가 발생한다.

BN defines per-unit $\mu_i, \sigma_i$ over the data distribution, approximated by minibatch statistics. This causes batch-size dependence and train/inference divergence.

### Section 3: Layer Normalization / 레이어 정규화 (핵심 섹션)

**축을 바꾸자**: $\mu, \sigma$를 배치 축이 아닌 **레이어 내 hidden unit 축**으로 계산.

$$
\mu^l = \frac{1}{H}\sum_{i=1}^{H} a^l_i
$$
$$
\sigma^l = \sqrt{\frac{1}{H}\sum_{i=1}^{H}(a^l_i - \mu^l)^2}
$$

**결정적 차이**: BN과 달리 $\mu, \sigma$가 **샘플마다 독립적**으로 계산된다. 즉 배치의 다른 샘플을 전혀 참조하지 않는다. 따라서:
- 배치 크기 = 1이어도 OK (분산 ≠ 0)
- 훈련/추론 방식이 **완전히 동일**
- Online learning에도 직접 적용 가능

**Decisive difference**: $\mu, \sigma$ are computed independently per sample, never referencing other samples in the batch. Thus:
- Works at batch size 1 (variance ≠ 0)
- Training and inference are **identical**
- Directly applicable to online learning

**Affine 변환**: 정규화 후 학습 가능한 scale/shift를 곱함 (BN과 동일):
$$
h_i = f\!\left(\frac{g_i}{\sigma^l}(a_i - \mu^l) + b_i\right)
$$

$g_i, b_i$는 **unit마다** 하나씩 학습됨 (BN의 $\gamma, \beta$에 해당).

Affine params $g_i, b_i$ are learned per unit (analogous to BN's $\gamma, \beta$).

### Section 4: Layer Normalized Recurrent Neural Networks / LN RNN

RNN의 한 time step에서 summed input은:
$$
\mathbf{a}^t = W_{hh} \mathbf{h}^{t-1} + W_{xh} \mathbf{x}^t
$$

LN은 이를 다음처럼 정규화:
$$
\mathbf{h}^t = f\!\left[ \frac{\mathbf{g}}{\sigma^t} \odot (\mathbf{a}^t - \mu^t) + \mathbf{b} \right]
$$
$$
\mu^t = \frac{1}{H}\sum_{i=1}^{H} a^t_i, \qquad \sigma^t = \sqrt{\frac{1}{H}\sum_{i=1}^{H}(a^t_i - \mu^t)^2}
$$

**핵심 관찰**:
- $\mathbf{g}, \mathbf{b}$는 모든 time step에 걸쳐 **공유**됨 → 시퀀스 길이가 달라도 문제없음.
- Summed input의 스케일이 토큰마다 달라도 LN이 자동으로 맞춰주므로 RNN의 gradient explosion/vanishing을 완화.
- BN과 달리 시퀀스 길이가 가변이어도 각 time step이 독립적으로 자신의 통계를 계산.

Key observation: $\mathbf{g}, \mathbf{b}$ are **shared** across time steps, so variable sequence lengths cause no problem. LN dampens gradient explosion/vanishing by rescaling summed inputs per step.

LSTM의 경우, 4개 게이트에 해당하는 pre-activation $\mathbf{a}^t$ (4H 차원)에 대해 LN 적용. 이후 시그모이드/tanh 비선형성이 따른다. (논문 식 21)

For LSTM, LN is applied to the 4-gate pre-activation $\mathbf{a}^t$ (dim 4H), then sigmoid/tanh follows.

### Section 5: Analysis — Invariance and Geometry / 분석: 불변성과 기하학

논문의 이론 섹션. 세 가지 정규화(BN, WN, LN)가 어떤 변환에 대해 불변(invariant)인가 비교.

#### 5.1 Invariance Table (Table 1) / 불변성 표

| Method | Weight matrix 재스케일 | Weight vector 재스케일 | 데이터 셋 재스케일 | 데이터 셋 재shift |
|---|---|---|---|---|
| Batch norm | invariant | no | invariant | no |
| Weight norm | invariant | **invariant** | no | no |
| Layer norm | invariant | no | **invariant** | **invariant** |

즉 LN은 **입력 데이터의 per-sample 재스케일·재shift**에 모두 강건하다. 입력 분포가 샘플마다 달라도 LN이 자동으로 같은 분포로 맞춰주기 때문.

LN is uniquely robust to **per-sample rescaling and shifting** of inputs — any per-sample affine transformation is absorbed away.

**왜 Per-sample shift invariance가 중요한가**: 예를 들어 어떤 이미지는 전체적으로 밝고 어떤 이미지는 어두울 때, LN은 각 이미지를 자기 통계로 정규화하므로 이 차이가 다음 층에 전달되지 않는다. BN은 per-feature shift만 제거하므로 이 효과가 없다.

Why per-sample shift invariance matters: if one image is globally bright and another dark, LN removes this per-sample bias before passing to the next layer; BN does not.

#### 5.2 Parameter Space and Weight Rescaling Invariance / 파라미터 공간의 곱셈적 불변성

BN, WN, LN 모두 weight $W$에 대한 곱셈적 재스케일에 대해 출력이 불변이다 (어차피 통계로 나누므로). 그러나 LN의 특징은 **샘플 단위 데이터 변환에도 불변**이라는 것.

All three are invariant to multiplicative weight rescaling (trivially, since they divide by statistics), but LN additionally has **per-sample data invariance**.

#### 5.3 Riemannian Metric and Parameter Space Geometry / 리만 메트릭과 파라미터 공간 기하

저자들은 파라미터 공간의 **Fisher information matrix**를 분석한다. 직관적으로, 곡률이 가파르면 gradient 업데이트가 non-uniform해서 학습이 불안정. 정규화는 이 곡률을 부드럽게 하여 안정적 업데이트를 만든다. LN의 경우 weight vector 스케일과 무관한 Riemannian metric이 얻어져 **자연스럽게 스케일-불변 최적화**가 이루어진다.

Authors analyze the Fisher information matrix of the parameter space. Intuitively, steep curvature makes updates non-uniform and unstable. Normalization smooths this curvature. For LN, the resulting Riemannian metric is scale-invariant, yielding **implicit scale-invariant optimization**.

수식 전개는 복잡하지만 결론은: LN은 weight matrix rescaling에 따라 implicit learning rate가 자동으로 조정되는 효과를 준다.

Practical takeaway: LN gives implicit, automatic learning-rate adaptation under weight rescaling.

### Section 6: Experiments / 실험

다섯 가지 RNN-중심 태스크에서 LN vs baseline vs BN(가능한 경우) 비교.

#### 6.1 Order Embeddings for Image-Sentence Retrieval
- 태스크: 이미지와 문장 쌍의 ranking retrieval (Vendrov et al. 2016).
- Base: GRU encoder. LN을 GRU에 삽입.
- **결과** (Table 2): LN은 baseline보다 **더 빠르게 수렴**하고 **더 낮은 R@1, R@5, R@10 error** 달성. 예: R@1 Caption→Image에서 base 37.9% → LN 38.9%.

#### 6.2 Teaching Machines to Read and Comprehend (Attentive Reader)
- 태스크: CNN 뉴스기사 + cloze 질문에서 정답 predict.
- Base model: 복잡한 bidirectional LSTM + attention (Hermann et al. 2015).
- **결과**: LN이 baseline보다 훨씬 **빠른 수렴** (Figure 2). 최종 정확도도 상승.

#### 6.3 Skip-Thought Vectors
- 태스크: 문장 임베딩 학습 (Kiros et al. 2015) — 주어진 문장 다음/이전 문장 generate.
- 대규모 데이터셋 (BookCorpus).
- **결과** (Figure 3, Table 3): LN은 1달 훈련 후에도 baseline 1달 성능을 **능가**. 수렴 속도가 극적으로 빠름. 이는 **큰 데이터셋에서도 LN이 유효**함을 입증.

#### 6.4 Modeling Binarized MNIST with DRAW
- 태스크: DRAW (Gregor et al. 2015) — sequential attention-based image generation.
- **결과** (Table 4): LN 적용 DRAW가 baseline보다 더 낮은 test NLL (82.36 → 82.09).

#### 6.5 Handwriting Sequence Generation
- 태스크: IAM Online handwriting data 생성 (Graves 2013).
- **결과** (Figure 4): LN이 훨씬 빠르게 수렴, 최종 log-likelihood도 높음.

#### 6.6 Permutation Invariant MNIST
- 태스크: MLP로 MNIST 분류 (CNN 구조 사용 안함).
- **결과** (Figure 5): LN vs BN vs baseline. LN이 가장 낮은 test error와 가장 빠른 수렴. **배치 크기 4, 64** 모두에서 LN이 BN보다 우월. 특히 작은 배치에서 BN이 무너지는 반면 LN은 안정적.

#### 6.7 Convolutional Networks / CNN에 적용
- **중요한 negative result**: CNN에 LN을 적용하면 BN보다 성능이 **나쁘다**.
- 저자들의 해석: CNN에서는 각 hidden unit이 local receptive field에 대응하므로 unit 간 통계 구조가 달라 (e.g., R/G/B 채널 분포 차이). 이를 한 평균으로 섞으면 정보 손실.
- **결론**: LN은 RNN/fully-connected에서 최고 성능. CNN에서는 BN이 여전히 우세.

Important negative result: LN underperforms BN in CNNs. Interpretation: in CNNs, different units correspond to different local receptive fields with distinct statistics — mixing them via one mean loses information. LN is best for RNN / fully-connected; BN remains superior for CNNs.

### Section 7: Conclusion / 결론

LN은 BN의 한계(배치 의존성)를 단순한 축 변경으로 해결하며, 특히 RNN/시퀀스 모델에서 효과적이다. 미래 방향으로 CNN에서의 LN 개선을 언급.

LN resolves BN's batch dependency via a simple axis flip and excels in RNNs / sequence models. Future work suggested for CNN variants.

---

## 3. Key Takeaways / 핵심 시사점

1. **축 선택 하나가 모든 차이를 만든다** — BN과 LN은 같은 tensor를 다른 축으로 평균할 뿐이지만, 이 선택이 배치 의존성, 훈련/추론 일치, 시퀀스 모델 적용 가능성 모두를 결정한다. 아키텍처 설계에서 "**어떤 축으로 통계를 공유할 것인가**"가 근본적 질문임을 보여주는 교과서적 예시.
   **One axis choice drives everything** — BN and LN average the same tensor along different axes, but that one choice determines batch dependency, train/inference parity, and RNN applicability. A textbook lesson on how axis-sharing decisions shape architectures.

2. **Batch size independence는 실용적으로 거대한 이점** — Distributed training, online learning, small-batch detection/segmentation, inference-time batch=1 등 현실 시나리오 대부분에서 LN은 "그냥 작동"한다. BN은 각 상황마다 우회책(sync-BN, group BN, 등)을 필요로 한다.
   **Batch-size independence is a huge practical advantage** — LN "just works" in distributed training, online learning, small-batch detection, batch=1 inference. BN needs workarounds (sync-BN, group-BN) for each.

3. **훈련/추론 일치는 실수를 줄이는 엔지니어링 이점** — BN의 "training mode / eval mode" 전환, running statistics 관리, moving average momentum 튜닝 — 이 모든 것이 LN에서는 **존재하지 않는다**. `model.eval()` 호출 잊어서 생기는 버그도 없다.
   **Train/inference parity reduces engineering bugs** — BN's training/eval mode switch, running stats, momentum tuning — none of that exists in LN. No more "forgot to call eval()" bugs.

4. **LN의 invariance 구조는 독특하다** — Table 1에서 LN만이 per-sample input rescaling **과** shifting **둘 다**에 불변이다. 이는 입력 분포가 샘플마다 달라도 모델이 안정적임을 의미 — RNN이 시퀀스마다 다른 스케일을 봐도, 또는 이미지 밝기가 극단적으로 달라도 문제없음.
   **LN's invariance profile is unique** — Table 1 shows LN alone is invariant to **both** per-sample rescaling and shifting. The model is robust to per-sample distribution shifts — variable-scale RNN sequences, extreme brightness differences, etc.

5. **RNN에 정규화를 넣는 올바른 방법** — BN이 RNN에서 고생한 이유는 시간 축 통계 추적이 근본적으로 까다롭기 때문. LN은 "각 time step의 hidden state **자체 내부에서** 정규화"로 이 문제를 우회한다. 이는 사실상 **모든 현대 시퀀스 모델의 기본 설계**가 되었다.
   **The correct way to normalize in RNNs** — BN struggled because tracking stats over time is inherently tricky. LN sidesteps this by normalizing within each timestep's own hidden state — now the default in every modern sequence model.

6. **CNN에서 실패한 것은 "축 선택"의 의미론이 깨지기 때문** — §6.7의 negative result는 중요한 통찰을 준다: 정규화는 "통계적으로 교환 가능한(exchangeable)" 값들을 pool할 때만 효과적. CNN의 feature map에서 R/G/B 채널, 서로 다른 공간 위치는 교환 가능하지 않다. Transformer의 d_model 차원은 비교적 교환 가능해서 LN이 잘 작동. 이 관찰은 **GroupNorm** (Wu & He 2018)의 탄생으로 이어진다.
   **The CNN failure reveals a deep principle** — normalization works only when pooled values are "statistically exchangeable." R/G/B channels and different spatial positions in CNN feature maps are not exchangeable; d_model dimensions in Transformers are. This observation spawned GroupNorm (Wu & He 2018).

7. **Transformer와 LLM 시대의 기반** — Transformer(2017)가 LN을 채택한 순간부터 NLP와 멀티모달 AI 전체가 LN 위에 세워졌다. GPT-2 이후 pre-norm 구조(각 서브층 앞에 LN, skip 후에는 아무것도 없음)는 1000+ 블록 LLM을 가능하게 했다. RMSNorm(2019)은 LN의 mean subtraction을 생략한 더 단순한 변형으로 LLaMA 등이 채택. LN이 없었다면 현재의 LLM 생태계는 **존재할 수 없다**.
   **Foundation of the Transformer/LLM era** — the moment Transformer (2017) adopted LN, all of NLP and multimodal AI was built on it. Pre-norm (GPT-2+) with LN before each sub-layer enables 1000+ block LLMs. RMSNorm (2019) — a simplification dropping mean subtraction — powers LLaMA. The current LLM ecosystem **could not exist** without LN.

8. **정규화 기법은 통일된 시각으로 볼 수 있다** — BN, LN, IN, GN은 모두 "$(N, C, H, W)$ tensor에서 어떤 축 부분집합으로 pool하는가"의 선택으로 통일된다. 이 관점은 후속 연구의 설계 공간을 체계적으로 탐색 가능하게 했다 (e.g., Switchable Norm — 훈련 중 어떤 정규화가 좋은지 학습).
   **Normalization has a unified view** — BN/LN/IN/GN all differ only in "which axis subset of $(N,C,H,W)$ to pool." This view enables systematic design-space exploration (e.g., Switchable Norm learns the axis choice during training).

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Layer Normalization (core formula) / 핵심 수식

For a single sample's pre-activation vector $\mathbf{a} \in \mathbb{R}^H$ at a given layer:

$$
\mu = \frac{1}{H}\sum_{i=1}^{H} a_i
$$

$$
\sigma^2 = \frac{1}{H}\sum_{i=1}^{H}(a_i - \mu)^2
$$

$$
\hat{a}_i = \frac{a_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
y_i = g_i \cdot \hat{a}_i + b_i
$$

- $\mu, \sigma^2$: **scalar per sample** (and per layer). No batch dimension.
- $g_i, b_i \in \mathbb{R}^H$: learned affine parameters.
- $\epsilon$: small constant for numerical stability (e.g., 1e-5).

### 4.2 BN vs LN axis / 축 비교

For a 2D tensor of shape $(N, H)$ (batch × hidden units):

$$
\text{BN}: \quad \mu^{\text{BN}}_i = \frac{1}{N}\sum_{n=1}^{N} a_{n,i} \quad (\text{H stats, one per unit})
$$

$$
\text{LN}: \quad \mu^{\text{LN}}_n = \frac{1}{H}\sum_{i=1}^{H} a_{n,i} \quad (\text{N stats, one per sample})
$$

Same formula, transposed axis.

### 4.3 LN in an LSTM / LSTM에 LN 적용

Standard LSTM gates:
$$
\begin{bmatrix} \mathbf{i} \\ \mathbf{f} \\ \mathbf{o} \\ \mathbf{g} \end{bmatrix} = \begin{bmatrix} \sigma \\ \sigma \\ \sigma \\ \tanh \end{bmatrix}\!\left( W \mathbf{h}^{t-1} + U \mathbf{x}^t \right)
$$

LN-LSTM inserts LN on the pre-activation (논문 식 20–21):
$$
\mathbf{a}^t = \text{LN}\!\left( W \mathbf{h}^{t-1} + U \mathbf{x}^t; \mathbf{g}, \mathbf{b} \right)
$$
Gates and cell update then proceed normally. Key: $\mathbf{g}, \mathbf{b}$ shared across time.

### 4.4 Invariance Properties (Table 1) / 불변성

Let $\phi$ denote a transformation of the input or weights. The method is invariant if output doesn't change under $\phi$.

| $\phi$ | BN | WN | LN |
|---|---|---|---|
| Weight matrix rescale $W \to \alpha W$ | ✓ | ✓ | ✓ |
| Individual weight vector rescale $w_i \to \alpha_i w_i$ | ✗ | **✓** | ✗ |
| Per-dataset rescale $\mathbf{x} \to \alpha \mathbf{x}$ (same for all samples) | ✓ | ✗ | ✓ |
| Per-dataset shift $\mathbf{x} \to \mathbf{x} + \mathbf{c}$ | ✗ | ✗ | — |
| **Per-sample rescale** $\mathbf{x}^{(n)} \to \alpha^{(n)} \mathbf{x}^{(n)}$ | ✗ | ✗ | **✓** |
| **Per-sample shift** $\mathbf{x}^{(n)} \to \mathbf{x}^{(n)} + \mathbf{c}^{(n)}$ | ✗ | ✗ | **✓** |

The last two rows are LN's unique signature.

### 4.5 Normalization Family Unified / 정규화 통일 관점

For tensor $\mathbf{x} \in \mathbb{R}^{N \times C \times H \times W}$ (CNN feature map):

| Method | Pool axes | # stats | Values per stat |
|---|---|---|---|
| **Batch Norm** | $(N, H, W)$ | $C$ | $N \cdot H \cdot W$ |
| **Layer Norm** | $(C, H, W)$ | $N$ | $C \cdot H \cdot W$ |
| **Instance Norm** | $(H, W)$ | $N \cdot C$ | $H \cdot W$ |
| **Group Norm** (G groups) | $(C/G, H, W)$ | $N \cdot G$ | $(C/G) \cdot H \cdot W$ |
| **RMSNorm** | $(C, H, W)$ — but no mean subtraction | $N$ | $C \cdot H \cdot W$ |

RMSNorm drops $\mu$:
$$
y_i = \frac{a_i}{\sqrt{\frac{1}{H}\sum_j a_j^2 + \epsilon}} \cdot g_i
$$

### 4.6 Worked Example: RGB image, batch 4 / RGB 4배치 예시

Input: $(N, C, H, W) = (4, 3, 32, 32)$ — 4 images of RGB 32×32.

LN computes 4 pairs of statistics. For sample $n$:
- Pool all $3 \cdot 32 \cdot 32 = 3072$ values within that sample.
- $\mu_n = \frac{1}{3072}\sum_{c=1}^{3}\sum_{h=1}^{32}\sum_{w=1}^{32} x_{n,c,h,w}$
- Same for $\sigma_n$.
- Normalize every element of that sample by its own $(\mu_n, \sigma_n)$.

So RGB channels are merged into one bag per sample — this is why LN is suboptimal for CNNs (§6.7 of paper, see notes Key Takeaway 6).

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1998  LeCun        — LeNet-5, input normalization              [#10]
2010  Glorot       — Xavier initialization
2013  Hinton       — Dropout (noise-based regularization)
2015  He           — He initialization, PReLU
2015  Ioffe        — Batch Normalization                       [#18]
2015  Srivastava   — Highway Networks (gated skip)
2016  Laurent      — Batch Normalized RNN (partial success)
2016  Cooijmans    — Recurrent Batch Normalization
2016  Ba-Kiros-Hinton — Layer Normalization                   ★ THIS PAPER
2016  Salimans     — Weight Normalization
2016  Ulyanov      — Instance Normalization (style transfer)
2017  Vaswani      — Transformer (LN inside residual)         [#25]
2018  Wu-He        — Group Normalization
2018  He (ResNet v2) — pre-activation residual (connects to pre-norm LN)
2019  Radford      — GPT-2 (pre-norm LN)
2019  Zhang-Sennrich — RMSNorm (simplified LN)
2020  Dosovitskiy  — ViT (LN on patch tokens)
2020  Xiong        — Pre-LN vs Post-LN convergence analysis
2022  Touvron      — LLaMA (RMSNorm + pre-norm)
2024+ 모든 LLM — LN/RMSNorm 기반
```

LN은 "CNN의 정규화"에서 "시퀀스/Transformer/LLM의 정규화"로 분화가 시작된 분기점이다. 같은 "activation 분포 통제"라는 목표를 공유하면서도, 데이터 구조가 다르면 최적 축이 다르다는 통찰을 확립했다.

LN is the branching point where normalization split into "CNN normalization" (BN) and "sequence/Transformer/LLM normalization" (LN). It established that different data structures call for different normalization axes.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#18 Batch Normalization (Ioffe-Szegedy 2015)** | 직접적 동기이자 주된 비교 대상. LN은 BN의 "같은 목표, 다른 축" 해법. | LN 수식 자체가 BN의 축 변형. 논문 전체가 BN과의 대비로 구성됨. |
| **#9 LSTM (Hochreiter-Schmidhuber 1997)** | LN의 주요 적용 대상. §4, §6의 모든 RNN 실험이 LSTM 기반. | LSTM의 gate들 pre-activation에 LN을 삽입하는 구체적 설계 (§4의 식 20-21). |
| **#25 Transformer (Vaswani 2017)** | LN의 가장 성공적 응용. 각 sub-layer에서 residual 전/후에 LN 적용. | 모든 현대 LLM (GPT, BERT, LLaMA, Claude)이 LN 기반. Pre-norm 변형이 사실상 표준. |
| **#20 ResNet (He 2015)** | Residual connection + LN은 Transformer의 기본 결합. ResNet v2의 pre-activation 철학이 Pre-norm LN으로 계승. | "skip 경로는 순수 linear, 비선형성은 내부에"라는 원리가 LN/BN 배치 선택에 영향. |
| **Weight Normalization (Salimans 2016)** | 같은 해에 제안된 대안적 정규화. 가중치 벡터 재매개화. | Table 1의 invariance 비교에서 주요 대조군. WN은 weight-level, LN은 activation-level. |
| **Instance Normalization (Ulyanov 2016)** | LN의 채널별 variant. Style transfer의 표준. | $(N,C,H,W)$에서 pool 축 차이만 있을 뿐 동일 프레임워크. |
| **Group Normalization (Wu-He 2018)** | LN과 IN의 일반화. 채널 그룹별 pool. | CNN에서 BN이 못 하는 작은 배치 상황을 해결. LN-in-CNN 실패에서 배운 교훈을 반영. |
| **RMSNorm (Zhang-Sennrich 2019)** | LN의 단순화 — mean subtraction 생략. | LLaMA 계열 LLM이 채택. LN의 실질적 대체제로 부상. |
| **Xiong et al. (2020) Pre-LN Transformer** | LN 위치의 중요성을 분석. Post-LN은 warm-up 없이 발산, Pre-LN은 안정적. | 이 논문의 후속 영향을 수학적으로 정식화한 핵심 연구. |

---

## 7. References / 참고문헌

- **This paper**: Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. arXiv:1607.06450. (NeurIPS 2016 Deep Learning Symposium)
- Ioffe, S., & Szegedy, C. (2015). *Batch Normalization*. ICML 2015.
- Salimans, T., & Kingma, D. P. (2016). *Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks*. NeurIPS 2016.
- Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016). *Instance Normalization: The Missing Ingredient for Fast Stylization*. arXiv:1607.08022.
- Wu, Y., & He, K. (2018). *Group Normalization*. ECCV 2018.
- Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS 2019. (RMSNorm)
- Xiong, R., et al. (2020). *On Layer Normalization in the Transformer Architecture*. ICML 2020.
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017. (Transformer)
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Laurent, C., Pereyra, G., Brakel, P., Zhang, Y., & Bengio, Y. (2016). *Batch Normalized Recurrent Neural Networks*. ICASSP 2016.
- Cooijmans, T., Ballas, N., Laurent, C., Gülçehre, Ç., & Courville, A. (2017). *Recurrent Batch Normalization*. ICLR 2017.
- Hermann, K. M., et al. (2015). *Teaching Machines to Read and Comprehend*. NeurIPS 2015. (Attentive Reader)
- Kiros, R., et al. (2015). *Skip-Thought Vectors*. NeurIPS 2015.
- Gregor, K., et al. (2015). *DRAW: A Recurrent Neural Network for Image Generation*. ICML 2015.
- Vendrov, I., Kiros, R., Fidler, S., & Urtasun, R. (2016). *Order-Embeddings of Images and Language*. ICLR 2016.
- Touvron, H., et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*. arXiv:2302.13971.
