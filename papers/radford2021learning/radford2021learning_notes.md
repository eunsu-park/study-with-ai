---
title: "Learning Transferable Visual Models From Natural Language Supervision"
authors: [Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever]
year: 2021
journal: "International Conference on Machine Learning (ICML)"
doi: "arXiv:2103.00020"
topic: Artificial_Intelligence
tags: [CLIP, multimodal, contrastive-learning, zero-shot, vision-language, foundation-model, ViT, transformer]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 36. Learning Transferable Visual Models From Natural Language Supervision / 자연어 감독으로 학습한 전이 가능한 시각 모델 (CLIP)

---

## 1. Core Contribution / 핵심 기여

**한국어**
이 논문은 **CLIP (Contrastive Language–Image Pre-training)** 을 제안합니다. CLIP은 인터넷에서 수집한 **4억 개의 (이미지, 텍스트) 쌍** (WIT 데이터셋)을 사용하여 이미지 인코더 $f_v$와 텍스트 인코더 $f_t$를 **대조 학습(contrastive learning)** 으로 동시에 훈련합니다. 학습 목표는 단순합니다: 한 미니배치의 $N$개 (이미지, 텍스트) 쌍 중에서 **어느 이미지가 어느 텍스트와 짝이 맞는지** 예측 — 이는 $N \times N$ 코사인 유사도 행렬에서 대각선이 가장 높아지도록 만드는 대칭적 InfoNCE 손실로 정식화됩니다.

핵심 통찰은 두 가지입니다. 첫째, **자연어가 분류 라벨보다 훨씬 풍부한 감독 신호**입니다. 1000개 ImageNet 클래스에 갇힌 모델과 달리, CLIP은 자연어로 묘사 가능한 모든 시각 개념을 학습할 수 있습니다. 둘째, **사전학습 task로 contrastive(어느 텍스트인가 식별)가 predictive(정확한 텍스트 단어 예측)보다 4배 더 효율적**입니다 (Figure 2). 이 두 결정 덕에 CLIP은 ImageNet의 1.28M 학습 예제를 단 하나도 사용하지 않고 **76.2%의 zero-shot top-1 정확도**를 달성합니다 — 라벨로 학습된 ResNet-50과 동급. 더 놀라운 것은 분포 변화에 대한 견고성입니다: ImageNet-A에서 CLIP은 ResNet-101보다 +74.4%, ImageNet-R에서 +51.2% 높은 정확도를 기록합니다. CLIP은 30개 이상의 데이터셋에서 다양한 task(OCR, geo-localization, action recognition, fine-grained classification)에 자연스럽게 전이됩니다. 가장 큰 모델 ViT-L/14@336px는 256개의 V100 GPU에서 12일간 학습되었고, EfficientNet-L2 NS와 같은 SOTA 비전 모델을 27개 데이터셋의 21개에서 능가합니다.

**English**
This paper introduces **CLIP (Contrastive Language–Image Pre-training)**. CLIP jointly trains an image encoder $f_v$ and a text encoder $f_t$ using **contrastive learning** on **400 million (image, text) pairs** scraped from the internet (the WIT dataset). The training task is deceptively simple: given a mini-batch of $N$ (image, text) pairs, predict **which image goes with which text** — formalized as a symmetric InfoNCE loss that maximizes the diagonal of an $N \times N$ cosine-similarity matrix.

Two insights make this work. First, **natural language is a vastly richer supervision signal than classification labels**: unlike a model locked into 1000 ImageNet classes, CLIP can learn every visual concept describable in language. Second, **a contrastive pre-training objective (which-text-matches) is 4× more efficient than a predictive one (predict-the-exact-words)** (Figure 2). Together, these decisions let CLIP achieve **76.2% zero-shot top-1 on ImageNet without using a single ImageNet training example** — matching the original ResNet-50. Even more strikingly, CLIP is dramatically more **robust to natural distribution shift**: it beats ResNet-101 by +74.4% on ImageNet-A and +51.2% on ImageNet-R. Across 30+ datasets, CLIP transfers to OCR, geo-localization, action recognition, and fine-grained classification — without task-specific labels. The largest model, ViT-L/14@336px, was trained on 256 V100 GPUs for 12 days and outperforms EfficientNet-L2 NS on 21 of 27 datasets.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Motivation (§1) / 서론과 동기

**한국어**
NLP는 task-agnostic 사전학습(GPT, BERT, T5)의 혁명을 겪었지만, 컴퓨터 비전은 여전히 **crowd-labeled ImageNet** 으로 사전학습하는 패러다임에 갇혀 있었습니다. 저자들은 묻습니다: "**웹의 수많은 텍스트로부터 직접 학습하는 scalable pre-training이 컴퓨터 비전에서도 비슷한 돌파구를 만들 수 있을까?**"

기존 자연어 감독 비전 연구는 두 흐름이 있었습니다:
1. **소규모 caption pretraining** (VirTex, ICMLM, ConVIRT — 2019-2020): 효과는 입증했지만 ImageNet에서 11.5% 정도밖에 안 됨.
2. **대규모 약감독** (Mahajan et al. 2018: Instagram 3.5B 이미지 + 해시태그; Kolesnikov et al. 2019: JFT-300M; Sun et al.): 잘 작동했지만 1000~18,291개 클래스로 supervision을 제한 — softmax classifier의 유연성 부족.

저자들의 핵심 차별점: **데이터 규모와 자연어의 표현력 모두를 활용**. 400M 쌍은 GPT-2의 WebText와 비슷한 단어 수.

**English**
NLP has been transformed by task-agnostic pre-training (GPT, BERT, T5), but computer vision was still anchored to pretraining on **crowd-labeled ImageNet**. The authors ask: "Could scalable pre-training methods that learn directly from raw web text produce a similar breakthrough in computer vision?"

Prior natural-language-supervised vision came in two flavors: (i) small-scale caption pretraining (VirTex, ICMLM, ConVIRT) — promising but only 11.5% zero-shot ImageNet; (ii) large-scale weak supervision (Mahajan 2018 with 3.5B Instagram tagged images, Kolesnikov 2019 with JFT-300M) — strong but bottlenecked by static softmax classifiers (1000–18,291 classes), which lacks flexibility for zero-shot. CLIP combines **scale** and **language flexibility**.

### Part II: Approach (§2) / 접근법

#### §2.1 Natural Language Supervision / 자연어 감독

**한국어**
자연어 감독의 두 가지 강점:
1. **확장성**: 1-of-N 라벨링이 필요 없음 — 인터넷의 텍스트가 사실상 무료 감독.
2. **유연성**: 표현이 학습되는 동시에 **언어와 연결**되어, zero-shot 전이가 자연스럽게 가능.

#### §2.2 Creating a Sufficiently Large Dataset / 충분히 큰 데이터셋 만들기

**한국어**
- 기존 데이터셋 검토:
  - MS-COCO, Visual Genome: 고품질이지만 100K 이미지 정도.
  - YFCC100M: 100M 이미지지만 메타데이터 품질이 낮음 (자동 생성 파일명 등). 영어 자연어 description으로 필터링하면 ~15M 이미지로 줄어 ImageNet 규모.
- **WIT (WebImageText) 구축**:
  - 인터넷에서 (이미지, 텍스트) 쌍을 **400M개** 수집.
  - 500,000개 쿼리 (Wikipedia에서 100회 이상 등장한 단어 + 자주 등장하는 bi-gram + WordNet synset 등).
  - 쿼리당 최대 20,000 쌍을 포함하도록 클래스 균형.
  - 단어 수는 GPT-2의 WebText와 비슷.

**English**
Existing datasets are too small (MS-COCO, Visual Genome ≈ 100K images) or too noisy (YFCC100M shrinks to ≈15M after English text filtering — about ImageNet size). The authors built **WIT**: 400M (image, text) pairs scraped from the public internet, balanced across 500K queries (Wikipedia-frequent words/bigrams + WordNet synsets), with up to 20K pairs per query. Total word count is comparable to GPT-2's WebText.

#### §2.3 Selecting an Efficient Pre-Training Method / 효율적 사전학습 방법 선택

**한국어**
저자들은 처음에 VirTex 스타일을 시도: image CNN + transformer language model이 정확한 caption을 예측하도록 학습. 그러나 transformer language model은 학습이 매우 느렸습니다. **Figure 2**가 결정적인 발견을 보여줍니다:
- 63M 파라미터 transformer LM (predictive): 가장 느림.
- Bag-of-Words 예측 (단순화된 predictive): **3배 빠름**.
- **Bag-of-Words contrastive (CLIP)**: 추가로 **4배 빠름** → 총 12배 효율 향상.

직관: 정확한 단어 시퀀스를 예측하는 것은 너무 어려운 task. 한 caption에 대해 다양한 표현이 가능하기 때문입니다. 반면 **"이 이미지에 짝맞는 텍스트가 무엇인가"** 라는 질문은 더 부드러우면서도 강한 감독 신호를 제공합니다.

CLIP의 학습 목표 (배치 크기 $N = 32{,}768$ 사용):
- $N$개 (이미지, 텍스트) 쌍이 미니배치를 구성.
- $N$개 이미지 임베딩 $\{I_i\}_{i=1}^N$ 와 $N$개 텍스트 임베딩 $\{T_j\}_{j=1}^N$ 를 계산.
- $N \times N$ 코사인 유사도 행렬 $S_{ij} = I_i \cdot T_j$ 를 만듦 (L2 정규화 후 내적).
- **대각선 $N$개**는 positive (실제 짝), **off-diagonal $N^2 - N$개**는 negative.
- 대칭 cross-entropy로 손실 계산.

**Figure 3 — CLIP 의사코드** (정확히 Numpy 스타일):

```
# image_encoder - ResNet or Vision Transformer
# text_encoder  - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t             - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I)  # [n, d_i]
T_f = text_encoder(T)   # [n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t) / 2
```

저자들의 단순화 (ConVIRT 대비):
- Image encoder를 ImageNet 가중치로 초기화하지 않음 (scratch).
- Text encoder도 사전학습된 가중치 없음 (scratch).
- Non-linear projection이 아닌 **linear projection** 만 사용.
- ConVIRT의 텍스트 transformation $t_u$ (다중 문장에서 1개 샘플링) 제거 — WIT는 한 문장이 대부분.
- Image augmentation은 random square crop만 사용.
- **Temperature $\tau$를 학습 가능 (log-parameterized scalar)** — 하이퍼파라미터 튜닝 회피.

**English**
The authors initially tried a VirTex-style image CNN + transformer LM that predicts captions exactly. Figure 2 reveals the bottleneck: a 63M-parameter transformer LM is 3× slower than a bag-of-words predictive baseline, which is itself 4× slower than the **bag-of-words contrastive** baseline (CLIP). Predicting exact words is too hard since many phrasings describe the same image; the contrastive task is softer yet still strongly supervised. CLIP uses batch size $N=32{,}768$ and the symmetric InfoNCE loss above. Compared to ConVIRT, CLIP simplifies: train both encoders from scratch, use only a linear projection to the joint space, drop text augmentation, use only random-square-crop image augmentation, and learn the temperature $\tau$ as a log-parameterized scalar (clipped to prevent logits scaling >100).

#### §2.4 Choosing and Scaling a Model / 모델 선택과 스케일링

**한국어**
**Image encoder**, 두 가지 아키텍처:
1. **ResNet-50 변형**: ResNet-D 개선 (He 2019), antialiased rect-2 blur pooling (Zhang 2019), global average pooling을 **attention pooling** (single-layer transformer-style multi-head QKV attention)으로 교체. RN50, RN101, RN50x4, RN50x16, RN50x64 (EfficientNet-style scaling: width, depth, resolution 동시 확대).
2. **Vision Transformer**: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px. 패치+위치 임베딩 결합 후 추가 LayerNorm, 약간 다른 초기화.

**Text encoder**:
- 12-layer, 512-wide, 8-head Transformer — **63M 파라미터**.
- BPE tokenization, 49,152 vocab.
- Max sequence length 76.
- [SOS]…[EOS]로 bracket. [EOS] 위치의 final-layer activation을 텍스트 representation으로 사용. LayerNorm 후 multi-modal embedding space로 linear projection.
- Text encoder는 **width**만 image encoder에 비례하여 확장; depth는 고정 — text encoder 용량에 정확도가 덜 민감했기 때문.

**English**
Image encoder has two families: (1) modified ResNets — RN50, RN101, then EfficientNet-style scaled RN50x4/16/64 — with ResNet-D, antialiased blur pooling, and attention pooling replacing GAP; (2) Vision Transformers — ViT-B/32, B/16, L/14, and L/14@336px (extra fine-tune at higher resolution). The text encoder is a 12-layer, 512-wide, 8-head Transformer (63M params) with BPE 49,152 vocab, max sequence length 76, bracketed by [SOS]/[EOS]; the [EOS] activation is layer-normalized and linearly projected to the joint embedding space. Text encoder is scaled only in width, since CLIP's accuracy proved insensitive to text capacity.

#### §2.5 Training / 학습

**한국어**
- 5 ResNets + 3 ViTs, 모두 **32 epochs**.
- Adam optimizer with decoupled weight decay (Loshchilov & Hutter 2017), cosine LR schedule.
- Initial hyperparameters: grid search + random search + manual tuning on **ResNet-50 baseline trained for 1 epoch**.
- Mini-batch size **32,768** (very large).
- Mixed precision (Micikevicius 2017), gradient checkpointing (Griewank & Walther 2000), half-precision Adam statistics (Dhariwal 2020), half-precision stochastically-rounded text encoder weights.
- Learnable $\tau$ initialized to $\exp(\text{log}(1/0.07)) \approx 14.3$, clipped to prevent logit scaling >100.
- 비용:
  - **RN50x64**: 18 days × 592 V100 GPUs.
  - **ViT-L/14**: 12 days × 256 V100 GPUs.
- "CLIP" = ViT-L/14@336px (한 epoch 추가 fine-tune at 336px).

**English**
All 8 models trained for 32 epochs with Adam + decoupled weight decay + cosine LR. Batch size 32,768 (huge). Mixed precision, gradient checkpointing, and half-precision optimizer state and weights save memory. The temperature $\tau$ is learnable (log-parameterized) and clipped. Largest ResNet (RN50x64): 18 days × 592 V100 GPUs. ViT-L/14: 12 days × 256 V100 GPUs. The default "CLIP" model is ViT-L/14 fine-tuned for one extra epoch at 336×336 resolution (FixRes-style).

### Part III: Experiments — Zero-Shot Transfer (§3.1) / 실험: Zero-Shot 전이

#### §3.1.1–3.1.2 Motivation & Mechanism / 동기와 메커니즘

**한국어**
"Zero-shot"의 의미: CLIP은 **사전학습 분포 외의 데이터셋으로 일반화 평가**. 단순히 unseen 클래스가 아닌, **task 자체에 대한 unseen 분포**를 평가합니다.

**Zero-shot 분류 절차**:
1. 데이터셋의 클래스 이름들을 자연어 prompt로 변환: "a photo of a {label}".
2. 텍스트 인코더로 모든 클래스 embedding $T_1, \ldots, T_K$ 계산 (한 번만 수행 후 캐시).
3. 입력 이미지를 image encoder로 embedding $I$ 계산.
4. 코사인 유사도를 temperature로 스케일하고 softmax:
   $$P(y=k \mid x) = \frac{\exp(\text{sim}(I, T_k)/\tau)}{\sum_{k'=1}^{K} \exp(\text{sim}(I, T_{k'})/\tau)}$$
5. argmax로 분류.

해석: image encoder는 표준 backbone이고, **text encoder는 hypernetwork** (Ha et al. 2016)로서 **자연어 description으로부터 분류기 가중치를 동적으로 생성**합니다.

#### §3.1.3 Initial Comparison to Visual N-Grams / Visual N-Grams와 초기 비교

**한국어**
**Table 1**:
| Dataset | Visual N-Grams | CLIP |
|---|---|---|
| aYahoo | 72.4 | **98.4** |
| ImageNet | 11.5 | **76.2** |
| SUN | 23.0 | **58.5** |

ImageNet에서 +64.7%p, top-5 95% (Inception-V4와 동급).

#### §3.1.4 Prompt Engineering and Ensembling / Prompt 엔지니어링과 앙상블

**한국어**
**문제 1**: Polysemy — 한 단어의 여러 의미가 충돌. 예: ImageNet에는 "construction crane" 과 "crane" (학) 모두 존재. Oxford Pets의 "boxer"는 개 품종이지만 텍스트 인코더는 운동선수로 해석할 수 있음.

**문제 2**: Distribution gap — WIT의 텍스트는 보통 한 단어가 아닌 **완전한 문장** 으로 이미지를 묘사.

**해결 — Prompt 템플릿**:
- 기본: `"a photo of a {label}."` → ImageNet +1.3%p.
- Fine-grained: `"a photo of a {label}, a type of pet."` (Pets), `"a photo of a {label}, a type of food."` (Food101), `"a photo of a {label}, a type of aircraft."` (Aircraft).
- OCR: 텍스트나 숫자를 따옴표로 감쌈.
- Satellite: `"a satellite photo of a {label}."`.

**Prompt ensembling**:
- 80개의 다른 context prompt (예: `"a photo of a big {label}"`, `"a photo of a small {label}"`)의 텍스트 임베딩을 평균.
- 임베딩 공간에서 평균하므로 추론 비용은 단일 prompt와 동일 (한 번 cache).
- ImageNet에서 단일 prompt 대비 추가 +3.5%p.

**총 효과 (Figure 4)**: prompt engineering + ensembling으로 zero-shot 성능 평균 ~5%p 향상 — 4× compute에 해당하는 향상을 무료로 얻음.

**English**
Two issues: **polysemy** (e.g., "boxer" might mean a dog breed or an athlete; "crane" the bird vs. construction equipment) and **distribution gap** (WIT texts are typically full sentences, not single words). Fix with prompt templates: `"a photo of a {label}."` (+1.3% ImageNet). Per-domain templates help further: pet/food/aircraft type qualifiers, quoted text for OCR, "a satellite photo of a {label}." for satellite. **Ensembling 80 prompts** in embedding space (averaged once and cached) adds another +3.5%. Combined: ~5% average gain across 36 datasets — free, since amortized over predictions.

#### §3.1.5 Analysis of Zero-Shot CLIP Performance / Zero-Shot 성능 분석

**한국어**
**Figure 5 — CLIP vs. ResNet-50 linear probe (27개 데이터셋)**: CLIP이 16개에서 승리.

주요 결과 (CLIP zero-shot - ResNet-50 linear probe):

| Dataset | Δ (%) |
|---|---|
| StanfordCars | +28.9 |
| Country211 | +23.2 |
| Food101 | +22.5 |
| Kinetics700 | +14.5 |
| SST2 | +12.4 |
| UCF101 | +7.7 |
| ImageNet | +1.9 |
| OxfordPets | +1.1 |
| **CLIP fails (negative Δ)**: | |
| Birdsnap | -3.2 |
| MNIST | -10.0 |
| FGVCAircraft | -11.3 |
| RESISC45 | -12.5 |
| Flowers102 | -16.6 |
| DTD | -18.2 |
| CLEVRCounts | -18.4 |
| GTSRB | -19.5 |
| KITTI Distance | -34.0 |
| EuroSAT | -37.1 |

**Zero-shot CLIP의 강점**:
- 동영상 인식 (Kinetics700 +14.5%, UCF101 +7.7%): 자연어가 명사 중심 ImageNet 라벨보다 동사도 풍부히 표현.
- Fine-grained 차량/음식 분류 (Cars, Food101): WIT의 다양성.
- General object datasets (CIFAR10, PascalVOC): ImageNet과 비슷.

**Zero-shot CLIP의 약점**:
- **추상적/체계적 task**: counting (CLEVRCounts), distance (KITTI), texture (DTD).
- **고도로 전문적인 도메인**: 위성 이미지 (EuroSAT, RESISC45), 의료 (lymph tumor — PatchCamelyon), 교통 표지 (GTSRB).
- **Out-of-distribution domain**: MNIST가 단지 54%! 자연 이미지에는 손글씨 숫자가 거의 없음.

**Figure 6 — Few-shot 비교**: zero-shot CLIP이 4-shot linear classifier (CLIP feature 위)와 동등. 16-shot 정도 되어야 zero-shot CLIP을 거의 따라잡음.

**Figure 7 — Data efficiency**: zero-shot CLIP과 일치하려면 평균 **20.8 labeled examples per class** 필요. FER2013은 184개, EuroSAT/Flowers102는 1개 미만.

**Figure 8 — Zero-shot vs linear-probe correlation (r=0.82)**: zero-shot은 linear probe보다 평균 10–25점 낮음 — 아직 개선 여지.

**Figure 9 — Compute scaling**: zero-shot 오류율은 model GFLOPs에 log-log linear (RN50→RN50x64로 44× compute 증가).

**English**
Across 27 datasets, zero-shot CLIP beats a fully supervised ResNet-50 linear probe on 16. Big wins on action recognition (Kinetics +14.5, UCF101 +7.7) — language captures verbs, not just nouns. Big losses on **abstract or specialized** tasks: counting, distance, satellite, traffic signs, medical, MNIST. Few-shot: zero-shot CLIP equals a 4-shot linear classifier on its own features and approaches a 16-shot linear classifier across all models — context-less example learning is data-inefficient. Average data efficiency: 20.8 labeled examples per class. Zero-shot performance correlates with linear-probe at r=0.82 but lies 10–25 points below — meaningful headroom remains. Performance scales log-linearly with compute (Figure 9, 44× range).

### Part IV: Representation Learning (§3.2) / 표현 학습

**한국어**
Linear probe 평가: 이미지 인코더의 features를 freeze하고 logistic regression만 학습.

**Figure 10 결과 (12 / 27 데이터셋 평균)**:
- 작은 CLIP 모델(RN50, RN101)은 EfficientNet-B0/B1보다 약함.
- **모든 CLIP 모델이 BiT-M (ImageNet-21K)을 능가**.
- 가장 큰 **ViT-L/14@336px**가 EfficientNet-L2 NS (이전 SOTA)를 12-dataset에서 +2.6%, 27-dataset에서 +5%p로 능가.
- **CLIP-ViT는 CLIP-ResNet보다 ~3배 compute-efficient**.

**Figure 11**: ViT-L/14 vs EfficientNet-L2 NS — CLIP이 27개 중 21개에서 승리. 가장 큰 격차: SST2 (+23.6), Country211 (+22.7), HatefulMemes (+18.8), Cars (+15.9), GTSRB (+14.7). EfficientNet 우세: ImageNet (-3.0), CLEVRCounts (-2.4), CIFAR100 (-1.7), CIFAR10 (-1.2).

**Figure 12**: CLIP의 transfer score는 ImageNet score 대비 **task shift에 더 robust** — ImageNet 전용 모델은 ImageNet에 overfit하는 경향.

**English**
With linear probes (frozen features + logistic regression): small CLIP models underperform EfficientNet-B0/B1, but every CLIP model beats BiT-M on ImageNet-21K. Top model **ViT-L/14@336px outperforms EfficientNet-L2 NS by 2.6% (12-dataset suite) and 5% (27-dataset suite)**. CLIP-ViT is ~3× more compute-efficient than CLIP-ResNet. Best model wins 21 of 27 datasets vs. EfficientNet-L2 NS, with biggest gains on SST2, Country211, HatefulMemes, Cars, GTSRB. Figure 12 shows CLIP features are also more robust to **task shift** than ImageNet-trained features.

### Part V: Robustness to Natural Distribution Shift (§3.3) / 자연 분포 변화 견고성

**한국어**
이 섹션은 논문의 가장 강렬한 결과를 담고 있습니다.

**배경**: ResNet-101은 ImageNet validation에 비해 자연 분포 변화 데이터셋(ImageNetV2, Sketch, Vid, ObjectNet, Adversarial, Rendition)에서 5배 많은 오류. Taori et al. (2020)은 분포 변화 정확도가 ImageNet 정확도의 logit-transform 선형 함수임을 발견.

**Figure 13**: 7개 자연 분포 변화 데이터셋에서 평균 정확도 vs. (class-subsampled) ImageNet 평균.
- ResNet-101 직선과 비교, **CLIP zero-shot 모델이 robustness gap을 75% 줄임**.

**구체적 비교 (ResNet-101 vs CLIP ViT-L/14@336px)**:

| Dataset | ResNet-101 | CLIP zero-shot | Δ |
|---|---|---|---|
| ImageNet | 76.2 | 76.2 | 0 |
| ImageNetV2 | 64.3 | 70.1 | +5.8 |
| ImageNet-R | 37.7 | 88.9 | **+51.2** |
| ObjectNet | 32.6 | 72.3 | +39.7 |
| ImageNet Sketch | 25.2 | 60.2 | +35.0 |
| ImageNet-A | 2.7 | 77.1 | **+74.4** |

**해석**: ImageNet에서 같은 76.2% 모델이 분포 변화에서는 극적으로 다름 — ResNet-101은 ImageNet 분포에 specialized된 spurious correlations에 의존, CLIP은 더 일반적인 representation 학습.

**Figure 14 — Adapt to ImageNet의 역설**: zero-shot CLIP을 ImageNet 라벨로 logistic regression fit하면 ImageNet 정확도가 76.2 → 85.4 (+9.2%p)지만 **평균 robustness가 약간 감소** — 분포 변화 정확도는 향상이 거의 없고 일부에서 감소 (ImageNet-R -4.7).

**Adapt to class shift**: 각 데이터셋에 맞는 prompt로 customize하면 일부에서 +26.9 (Youtube-BB) 향상 — class-name alignment의 중요성.

**핵심 통찰**: **분포 외 견고성은 분포 외 데이터에서의 학습이 아닌, 분포 특정 학습 데이터를 최소화**할 때 발생. ImageNet에 적응하면 그 데이터의 quirks을 학습하기 시작.

**English**
This section delivers CLIP's most striking finding. Taori et al. (2020) showed that on natural distribution shift datasets (ImageNetV2, Sketch, ObjectNet, ImageNet-A, -R, Vid, Youtube-BB), accuracy is well-predicted by a logit-linear function of ImageNet accuracy — and ImageNet-trained models lie *on* this line. **Zero-shot CLIP shrinks the robustness gap by up to 75%**. At the same 76.2% ImageNet accuracy, CLIP beats ResNet-101 by +5.8 on ImageNetV2, +51.2 on ImageNet-R, +39.7 on ObjectNet, +35.0 on Sketch, +74.4 on ImageNet-A. Paradoxically, **adapting CLIP to ImageNet** (logistic regression fit) raises ImageNet by +9.2 but slightly decreases average distribution-shift robustness (e.g., ImageNet-R drops -4.7). The lesson: **effective robustness arises from minimizing distribution-specific training data**, not from training on more distribution-shifted data — an indictment of the ImageNet-fine-tuning paradigm.

### Part VI: Comparison to Human Performance (§4) / 인간 성능과의 비교

**한국어**
Oxford Pets의 37개 품종 분류에서 5명에게 zero-shot, one-shot, two-shot 평가:
- Human zero-shot: ~54%, two-shot에서 ~76%.
- CLIP zero-shot: ~93.5%.
- 흥미롭게도 인간이 가장 어려워한 클래스 ≈ CLIP이 가장 어려워한 클래스 (failure mode가 비슷). 단, 인간은 hard examples에서 prior 지식으로 보완하지만 CLIP은 그렇지 못함.

**English**
On Oxford-IIIT Pets, 5 human raters in zero/one/two-shot settings: zero-shot human ≈54%, jumping to ≈76% with just one example — humans are extremely few-shot data-efficient. CLIP zero-shot is ≈93.5%. Failure modes are correlated between humans and CLIP, but humans uniquely benefit from prior knowledge and reasoning that CLIP cannot exploit.

### Part VII: Data Overlap Analysis (§5) / 데이터 중첩 분석

**한국어**
큰 우려: WIT 400M에 평가 데이터셋이 포함되었을까? 저자들은 duplicate detector로 분석:
- 35개 데이터셋 중, **9개에서 검출된 overlap이 1% 미만**.
- 평균 overlap median 2.2%, 평균 3.2%.
- CLIP의 zero-shot 성능과 overlap 사이에 **체계적 상관관계 없음**.
- Country211에서 가장 큰 영향 (+0.5%p) — 그러나 그 자체로 작은 효과.

**English**
The team built a duplicate detector and found median 2.2% / mean 3.2% overlap across 35 datasets, with no systematic correlation between overlap and zero-shot performance. Largest effect: +0.5% on Country211. CLIP's zero-shot results are not significantly inflated by train/test contamination.

### Part VIII: Limitations (§6) / 한계

**한국어**
1. **Zero-shot이 fully supervised의 천장에 못 미침**: linear probe가 zero-shot 평균 +10–25%p. 1000× compute 추가 필요할 수 있음.
2. **약한 도메인**: 추상적·체계적 task (counting, satellite, fine-grained), 분포 외 (MNIST 88%).
3. **Few-shot이 zero-shot보다 못함** (one/few-shot regime): natural language로 직접 specify하는 것이 example로부터 추론하는 것보다 강함 — **counter-intuitive**.
4. **공정성, 사회적 편향**: 인터넷 텍스트의 편향이 그대로 학습됨.
5. **테스트셋 디자인**: validation 시 zero-shot이 아닌 매개변수 튜닝 같은 모순.
6. **사전학습 비용**: 수백 GPU-day. 환경적, 접근성 문제.

**English**
(1) Zero-shot still trails fully supervised by 10–25 points; closing this might require ~1000× more compute. (2) Weak on abstract/systematic tasks (counting, distance), highly specialized domains (satellite, medical), and out-of-distribution data (MNIST). (3) **Few-shot is worse than zero-shot** in the 1–4-shot regime — natural-language specification beats example-based inference. (4) Inherits social biases from web data. (5) Test-set design tension: zero-shot evaluation must avoid validation-set hyperparameter tuning. (6) Compute cost is prohibitive.

### Part IX: Broader Impacts (§7) / 광범위한 영향

**한국어**
CLIP의 유연한 분류 능력은 **감시(surveillance)** 우려를 증폭. 라벨된 데이터 없이도 임의 클래스로 분류 가능 → 새로운 형태의 자동화된 감시 시스템 구현이 더 쉬워짐. 또한 인종, 성별, 연령에 대한 편향된 association이 임베딩 공간에 학습되어 있음을 보임 (FairFace 분석). 저자들은 이를 솔직히 인정하고 응용 시 주의를 권고.

**English**
CLIP's flexible classification dramatically lowers the barrier to building surveillance systems that recognize arbitrary categories without labeled data. The authors also document biases in CLIP's embeddings (race/gender/age associations on FairFace) and argue for careful study before deployment.

---

## 3. Key Takeaways / 핵심 시사점

1. **Natural language as the universal interface / 자연어는 보편적 인터페이스**
   1000개 라벨로 갇힌 분류 모델 대신 **자연어로 정의한 모든 시각 개념** 으로 zero-shot 분류 가능. 이는 컴퓨터 비전의 GPT-2 모먼트 — vision system이 task-agnostic하게 됩니다. / Instead of locking models to a 1000-class softmax, CLIP enables zero-shot classification of *any* concept describable in natural language. The GPT-2 moment for vision: task-agnostic by design.

2. **Contrastive >> predictive at scale / 대규모에서는 대조 학습이 예측 학습보다 우월**
   정확한 caption 단어를 예측하는 것은 너무 어렵고 느림. **"어느 쌍이 매치되는가"** 는 더 부드럽지만 12× 더 효율적인 감독 신호. / Predicting exact words is too hard; predicting which-pair-matches is 12× faster yet learns equally rich representations.

3. **Scale + simple objective wins / 규모 + 단순한 목표가 승리**
   400M 쌍 + 대칭 InfoNCE의 수십 줄 코드 = SOTA. 정교한 손실 함수, multi-task heads 등이 아닌 **데이터와 컴퓨트 스케일링** 이 핵심 동력. / 400M pairs + ~15 lines of symmetric InfoNCE code = SOTA. Not exotic losses or multi-task heads — scale is the engine.

4. **Zero-shot is more robust than supervised / Zero-shot이 supervised보다 견고**
   ImageNet 76.2% 동급 모델이 ImageNet-A에서 ResNet-101 대비 +74.4%, ImageNet-R에서 +51.2%. **분포 외 견고성은 분포 특정 학습 데이터를 최소화** 할 때 발생. / At equal ImageNet accuracy, zero-shot CLIP beats ResNet-101 by +74.4 on ImageNet-A and +51.2 on ImageNet-R. Robustness comes from minimizing distribution-specific training, not chasing it.

5. **Prompt engineering matters in vision too / Prompt 엔지니어링은 비전에서도 중요**
   `"a photo of a {label}"` 단순 템플릿이 +1.3%p, 도메인별 prompt + 80개 ensemble이 +5%p 추가. 텍스트 인코더는 분류기 가중치를 동적으로 생성하는 hypernetwork. / Simple templates add 1.3%, domain prompts and 80-prompt ensembling add ≈5% more. The text encoder is a hypernetwork that dynamically generates classifier weights.

6. **Few-shot is harder than zero-shot in low-data regime / 저데이터 영역에서 few-shot이 zero-shot보다 어려움**
   1–4 shot에서 example-based learning이 자연어 specification보다 약함 — context-less examples가 ambiguous하기 때문. / In 1–4-shot regimes, learning from examples is *worse* than specifying via language. Examples are ambiguous; language is precise.

7. **ViT > ResNet for CLIP / CLIP에는 ViT가 ResNet보다 좋음**
   ViT-L/14가 RN50x64보다 ~3배 compute-efficient. Vision transformer의 inductive bias가 sufficient data scale에서 더 적합. / ViT-L/14 is ~3× more compute-efficient than RN50x64. ViT's flexible attention works better at this data scale.

8. **Foundation model을 위한 새로운 평가 패러다임 / A new evaluation paradigm for foundation models**
   30+ 데이터셋 zero-shot suite는 task generalization을 측정. 단일 데이터셋 SOTA가 아닌 **wide-task transfer** 가 진정한 capability. / Zero-shot evaluation across 30+ datasets measures task-learning capability, not single-benchmark optimization. CLIP set the template for foundation-model evaluation.

---

## 4. Mathematical Summary / 수학적 요약

### Step 1: Encoders / 인코더

$$I_f = f_v(I) \in \mathbb{R}^{d_i}, \quad T_f = f_t(T) \in \mathbb{R}^{d_t}$$
- $f_v$: image encoder (ResNet variant or ViT). For ResNet, attention pooling replaces GAP; for ViT, an extra LayerNorm is added before the transformer.
- $f_t$: text encoder, 12-layer 512-wide Transformer. Output is the [EOS] token's final-layer activation.

### Step 2: Linear projection to joint embedding / 공동 임베딩으로 선형 사영

$$I_e = \frac{W_i^\top I_f}{\|W_i^\top I_f\|_2}, \quad T_e = \frac{W_t^\top T_f}{\|W_t^\top T_f\|_2} \in \mathbb{R}^{d_e}$$
- $W_i \in \mathbb{R}^{d_i \times d_e}$, $W_t \in \mathbb{R}^{d_t \times d_e}$ — learned linear projections.
- L2 normalize to unit sphere — cosine similarity = inner product.

### Step 3: Pairwise cosine similarity / 쌍별 코사인 유사도

For a mini-batch of $N$ pairs $\{(I^{(i)}, T^{(i)})\}_{i=1}^N$:

$$S_{ij} = I_e^{(i)} \cdot T_e^{(j)} \quad \in [-1, 1]$$
$$\text{logits} = S \cdot \exp(t)$$
- $t$ = learnable scalar; $\exp(t) = 1/\tau$ (temperature inverse).
- $\tau$ initialized to 0.07, clipped so $1/\tau \le 100$ (numerical stability).

### Step 4: Symmetric InfoNCE loss / 대칭 InfoNCE 손실

**Image-to-text direction**: for each row $i$, treat row as logits over $N$ texts, with target index $i$ (the matching pair).

$$\mathcal{L}_{i \to t} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^N \exp(S_{ij}/\tau)}$$

**Text-to-image direction**: for each column $i$, treat column as logits over $N$ images.

$$\mathcal{L}_{t \to i} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^N \exp(S_{ji}/\tau)}$$

**Total loss**:
$$\mathcal{L} = \frac{1}{2}(\mathcal{L}_{i \to t} + \mathcal{L}_{t \to i})$$

Each row/column of $S$ is treated as a softmax classification with $N$ classes, ground truth = diagonal index.

### Step 5: Zero-shot inference / Zero-shot 추론

For a $K$-class classification dataset, build text prompts $\{T^{(k)}\}_{k=1}^K$:
- e.g., $T^{(k)}$ = "a photo of a {class$_k$}".
- (Optional) ensemble 80 prompts: $T_e^{(k)} = \frac{1}{M}\sum_{m=1}^M T_{e,m}^{(k)}$, then re-normalize.

For test image $x$:
$$P(y=k \mid x) = \frac{\exp(\text{sim}(I_e, T_e^{(k)}) / \tau)}{\sum_{k'=1}^K \exp(\text{sim}(I_e, T_e^{(k')}) / \tau)}$$
$$\hat{y} = \arg\max_k P(y=k \mid x)$$

### Worked example: zero-shot inference on a 3-class toy dataset / 3-class 토이 데이터셋 zero-shot 추론 워크스루

**Setup**: classify images as "cat", "dog", or "bird". CLIP image encoder produces $I_e \in \mathbb{R}^{512}$ for the test image.

**Step A**: build text prompts.
```
prompt_1 = "a photo of a cat."
prompt_2 = "a photo of a dog."
prompt_3 = "a photo of a bird."
```

**Step B**: encode each prompt → $T_e^{(1)}, T_e^{(2)}, T_e^{(3)} \in \mathbb{R}^{512}$ (L2-normalized). Cache once.

**Step C**: encode test image → $I_e \in \mathbb{R}^{512}$ (L2-normalized).

**Step D**: compute cosine similarities (e.g., assume):
- $\text{sim}(I_e, T_e^{(1)}) = 0.31$ (cat)
- $\text{sim}(I_e, T_e^{(2)}) = 0.18$ (dog)
- $\text{sim}(I_e, T_e^{(3)}) = 0.05$ (bird)

**Step E**: scale by $1/\tau \approx 100$ (CLIP's learned value):
- logits = (31, 18, 5)

**Step F**: softmax:
$$P(\text{cat}) = \frac{e^{31}}{e^{31} + e^{18} + e^{5}} \approx 1.000$$
$$P(\text{dog}) \approx 2.3 \times 10^{-6}$$
$$P(\text{bird}) \approx 0$$

**Result**: predict "cat". Note how the **temperature scaling** sharpens probabilities — cosine differences of 0.13 become essentially deterministic predictions.

### Parameter count (ViT-L/14) / 파라미터 수

- ViT-L/14 image encoder: ~304M params.
- 12-layer 512-wide text encoder: ~63M params.
- 2 linear projections + 1 temperature scalar: ~negligible.
- **Total ≈ 367M params** (excluding embeddings).

### Training compute / 학습 컴퓨트

ViT-L/14: 12 days × 256 V100 = **3,072 V100-days** ≈ generic order-of-magnitude $10^{22}$ FLOPs.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
2009 ─ ImageNet (Deng et al.) — CV의 지배적 supervised pre-training
        │
2012 ─ AlexNet (paper #13) — CNN deep learning era 시작
        │
2014 ─ Word2Vec / GloVe — text embedding의 선조
        │
2015 ─ ResNet (paper #20) — 152-layer 네트워크
        │
2017 ─ Transformer (paper #25) — attention 기반 architecture
        │
2018 ─ BERT, GPT-1 — NLP self-supervised pre-training
        │
2018 ─ Mahajan et al. — 3.5B Instagram hashtags weak supervision
        │
2019 ─ GPT-2 (paper #29) — zero-shot NLP transfer 입증
        │       │
2019 ─ VirTex, ICMLM — caption-based vision pretraining (small scale)
        │
2020 ─ ViT (paper #31) — vision transformer
        │
2020 ─ SimCLR (paper #32) — contrastive vision pre-training
        │
2020 ─ ConVIRT (Zhang et al.) — contrastive image-text in medical
        │       ★ CLIP의 직접 선조 / Direct predecessor
        │
2021 ─ ★★★ CLIP (this paper) — 400M pairs, zero-shot ImageNet 76.2% ★★★
        │
2021 ─ DALL-E — text-to-image generation
        │       │
2021 ─ ALIGN (Google) — even larger noisy image-text
        │
2022 ─ Stable Diffusion / Imagen — CLIP encoder를 conditioning에 사용
        │
2022 ─ Flamingo (DeepMind), BLIP (Salesforce) — VLM
        │
2023 ─ GPT-4V, LLaVA — large multimodal models
        │
2023+─ SAM (Segment Anything), GroundingDINO — open-vocabulary perception
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#13 Krizhevsky et al. (2012) — AlexNet** | CLIP은 ResNet variants를 image encoder로 사용; CNN 패러다임의 거인의 어깨 / Uses ResNet variants as image encoders; stands on the shoulders of CNN heritage | High — image encoder의 직접 조상 / Direct ancestor for image encoder |
| **#18 Kingma & Ba (2014) — Adam** | CLIP은 decoupled weight decay Adam (AdamW)으로 학습 / Trained with decoupled weight decay Adam | Medium — 실험 도구 / Practical training tool |
| **#20 He et al. (2015) — ResNet** | RN50, RN101, RN50x4/16/64가 CLIP의 image encoder family / RN50 through RN50x64 form CLIP's ResNet family | High — 직접 사용 / Directly used |
| **#25 Vaswani et al. (2017) — Transformer** | Text encoder는 GPT-2 스타일 12-layer transformer; ViT image encoder도 Transformer / Text encoder is a 12-layer GPT-2-style transformer; ViT image encoder is also Transformer | Critical — 양쪽 인코더의 backbone / Backbone of both encoders |
| **#29 Radford et al. (2019) — GPT-2** | Text encoder가 GPT-2 architecture를 따름; "vision의 GPT-2 moment" / Text encoder follows GPT-2 architecture; "the GPT-2 moment for vision" | Critical — 같은 저자, 같은 철학 / Same author, same philosophy |
| **#31 Dosovitskiy et al. (2020) — ViT** | ViT-B/32, ViT-B/16, ViT-L/14가 CLIP의 강력한 image encoder / ViT-B/32, B/16, L/14 are CLIP's strongest image encoders | Critical — 최고 성능 image encoder / Top-performing image encoder |
| **#32 Chen et al. (2020) — SimCLR** | Contrastive learning paradigm; CLIP은 SimCLR의 image-text 일반화 / Contrastive learning paradigm; CLIP is the image-text generalization of SimCLR | Critical — contrastive 손실의 청사진 / Blueprint for contrastive loss |
| **ConVIRT (Zhang et al. 2020)** | CLIP의 직접 선조 — 의료 영역에서의 contrastive image-text / CLIP's direct predecessor — contrastive image-text in medical | Critical — CLIP은 단순화된 ConVIRT / CLIP is a simplified ConVIRT |
| **#19 Ioffe & Szegedy (2015) — BatchNorm** | LayerNorm은 transformer에서 널리 사용 (text encoder와 ViT) / LayerNorm widely used in transformers | Medium — 정규화 도구 / Normalization tool |
| **VirTex (Desai & Johnson 2020)** | CLIP의 초기 baseline; predictive caption objective의 한계를 보여줌 / CLIP's initial baseline; demonstrated limits of predictive caption objective | High — 동기 부여 baseline / Motivating baseline |
| **DALL-E, Stable Diffusion** | CLIP text encoder를 text-to-image 생성의 conditioning에 사용 / Use CLIP's text encoder as conditioning for text-to-image | Critical — CLIP의 가장 영향력 있는 응용 / Most impactful downstream application |
| **#37+ multimodal LLMs (GPT-4V, LLaVA)** | CLIP-style vision encoder를 LLM에 연결 / Connect CLIP-style vision encoders to LLMs | High — 현대 multimodal 패러다임 / Modern multimodal paradigm |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 8748–8763. arXiv:2103.00020.
- Code: https://github.com/OpenAI/CLIP

### Direct predecessors / 직접 선조
- Zhang, Y., Jiang, H., Miura, Y., Manning, C. D., & Langlotz, C. P. (2020). Contrastive learning of medical visual representations from paired images and text (ConVIRT). arXiv:2010.00747.
- Desai, K., & Johnson, J. (2020). VirTex: Learning visual representations from textual annotations. arXiv:2006.06666.
- Bulent Sariyildiz, M., Perez, J., & Larlus, D. (2020). Learning visual representations with caption annotations (ICMLM). *ECCV 2020*.
- Joulin, A., van der Maaten, L., Jabri, A., & Vasilache, N. (2016). Learning visual features from large weakly supervised data. *ECCV 2016*.
- Li, A., Jabri, A., Joulin, A., & van der Maaten, L. (2017). Learning visual n-grams from web data. *ICCV 2017*.
- Sohn, K. (2016). Improved deep metric learning with multi-class N-pair loss objective. *NeurIPS 2016*.
- Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding (InfoNCE). arXiv:1807.03748.

### Architecture / 아키텍처
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale (ViT). *ICLR 2021*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition (ResNet). *CVPR 2016*.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners (GPT-2).

### Training & data / 학습 및 데이터
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
- Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization (AdamW). arXiv:1711.05101.
- Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic gradient descent with warm restarts (cosine schedule). arXiv:1608.03983.
- Micikevicius, P., et al. (2017). Mixed precision training. arXiv:1710.03740.
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units (BPE). *ACL 2016*.

### Robustness & evaluation / 견고성 및 평가
- Taori, R., et al. (2020). Measuring robustness to natural distribution shifts in image classification. *NeurIPS 2020*.
- Recht, B., et al. (2019). Do ImageNet classifiers generalize to ImageNet? (ImageNetV2). *ICML 2019*.
- Hendrycks, D., et al. (2020). The many faces of robustness: A critical analysis of out-of-distribution generalization (ImageNet-R). arXiv:2006.16241.
- Hendrycks, D., et al. (2019). Natural adversarial examples (ImageNet-A). arXiv:1907.07174.
- Wang, H., et al. (2019). Learning robust global representations by penalizing local predictive power (ImageNet Sketch). *NeurIPS 2019*.
- Barbu, A., et al. (2019). ObjectNet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. *NeurIPS 2019*.
- Kornblith, S., Shlens, J., & Le, Q. V. (2019). Do better ImageNet models transfer better? *CVPR 2019*.

### Foundation models & follow-ups / 파운데이션 모델 및 후속 연구
- Brown, T. B., et al. (2020). Language models are few-shot learners (GPT-3). *NeurIPS 2020*.
- Ramesh, A., et al. (2021). Zero-shot text-to-image generation (DALL-E). *ICML 2021*.
- Jia, C., et al. (2021). Scaling up visual and vision-language representation learning with noisy text supervision (ALIGN). *ICML 2021*.
- Rombach, R., et al. (2022). High-resolution image synthesis with latent diffusion models (Stable Diffusion). *CVPR 2022*.
- Alayrac, J. B., et al. (2022). Flamingo: A visual language model for few-shot learning. *NeurIPS 2022*.
- Liu, H., et al. (2023). Visual instruction tuning (LLaVA). *NeurIPS 2023*.
