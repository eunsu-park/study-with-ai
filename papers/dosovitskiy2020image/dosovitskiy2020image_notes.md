---
title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
authors: [Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby]
year: 2020
journal: "International Conference on Learning Representations (ICLR 2021)"
doi: "arXiv:2010.11929"
topic: Artificial_Intelligence
tags: [vision-transformer, ViT, self-attention, image-classification, pretraining, transfer-learning, JFT-300M, inductive-bias, patch-embedding]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 31. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale / 이미지는 16×16 단어와 같다 — 대규모 이미지 인식을 위한 Transformer

---

## 1. Core Contribution / 핵심 기여

**한국어**
이 논문은 **순수 Transformer encoder를 거의 수정 없이 이미지 분류에 직접 적용**할 수 있음을 대규모로 입증한 작업입니다. 핵심 레시피는 단순합니다 — $H \times W \times C$ 이미지를 $P \times P$ 패치 $N = HW/P^2$개로 분할하고, 각 패치를 펼쳐 학습 가능한 선형 투영 $E$로 $D$-차원 토큰으로 변환합니다. BERT처럼 학습 가능한 [class] 토큰을 시퀀스 맨 앞에 붙이고, 1D learnable position embedding을 더해 표준 Transformer encoder에 입력합니다. 그게 전부입니다 — 합성곱도, locality나 translation equivariance 같은 이미지 특화 inductive bias도 거의 없습니다. 핵심 발견은 두 가지입니다. 첫째, **사전학습 데이터가 작을 때(ImageNet 1.3M)는 ViT가 같은 크기의 ResNet보다 약간 나쁩니다** — inductive bias가 부족해 일반화가 떨어지기 때문입니다. 둘째, 그러나 **데이터가 충분히 크면(ImageNet-21k 14M, 특히 JFT-300M)** ViT가 BiT-L, Noisy Student 같은 최첨단 CNN을 능가합니다. ViT-H/14는 ImageNet 88.55%, ImageNet-ReaL 90.72%, CIFAR-10 99.50%, CIFAR-100 94.55%, Oxford-IIIT Pets 97.56%, Oxford Flowers-102 99.68%, VTAB(19 태스크) 77.63%를 달성하며, BiT-L 대비 **약 4배 적은 사전학습 컴퓨트(2.5k vs 9.9k TPUv3-core-days)**만 사용합니다. 한 줄 메시지: **"규모가 inductive bias를 이긴다(scale trumps inductive bias)"**.

**English**
This paper demonstrates, for the first time at convincing scale, that a **pure Transformer encoder applied directly to image classification — with almost no modifications — can match or exceed state-of-the-art CNNs**. The recipe is deceptively simple. An image $x \in \mathbb{R}^{H \times W \times C}$ is split into $N = HW/P^2$ non-overlapping $P \times P$ patches, each flattened and linearly projected by a learnable matrix $E$ to a $D$-dimensional token. A BERT-style learnable [class] token is prepended, a learnable 1D positional embedding is added, and the resulting sequence is fed to a standard Transformer encoder. That is the entire model — no convolutions, no locality bias, no translation equivariance baked into layers. Two key findings. First, when pre-trained on **modest amounts of data (ImageNet, 1.3M images)** ViT trails comparable ResNets, because the missing inductive biases hurt generalization. Second, when pre-trained on **large datasets (ImageNet-21k, 14M; especially JFT-300M, 303M)** ViT overtakes the strongest CNNs (BiT-L, Noisy Student): ViT-H/14 reaches 88.55% on ImageNet, 90.72% on ImageNet-ReaL, 99.50% on CIFAR-10, 94.55% on CIFAR-100, 97.56% on Pets, 99.68% on Flowers-102, and 77.63% on VTAB, while requiring **~4× less pre-training compute (2.5k vs 9.9k TPUv3-core-days)** than BiT-L. The headline thesis: **scale trumps inductive bias** — given enough pre-training data, the absence of CNN-style priors is no longer a handicap.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (§1) / 서론

**한국어**
저자들은 NLP에서 Transformer가 자리 잡은 패러다임을 짚으며 시작합니다 — Vaswani et al.(2017)의 Transformer는 BERT, GPT 시리즈를 거치며 100B+ 파라미터의 모델까지 가능해졌고, 데이터와 모델 크기를 키워도 성능 saturation 신호가 보이지 않았습니다. 그러나 컴퓨터 비전은 여전히 CNN의 영토였고(LeCun 1989, AlexNet 2012, ResNet 2016), self-attention을 도입하려는 시도들 — Wang et al.(2018), Carion et al.(2020)의 CNN 결합, Ramachandran et al.(2019)의 self-attention 대체 — 은 모두 일정한 한계가 있었습니다.

저자들의 도전: **"표준 Transformer를 가능한 한 적게 수정하여 이미지에 직접 적용해보자."** 이미지를 패치로 나누어 NLP의 단어처럼 다루고 supervised image classification으로 학습합니다.

**중간 데이터(ImageNet)에서의 성능**: 강한 정규화 없이 학습하면, ViT는 비슷한 크기의 ResNet보다 몇 %p 낮은 성능을 보입니다. 이는 예상된 결과로, **Transformer는 CNN의 inductive bias(translation equivariance, locality)를 결여**하므로 데이터가 부족할 때 일반화가 어렵습니다.

**대규모 데이터에서의 반전**: 14M~300M 이미지로 사전학습하면 그림이 달라집니다. **"large scale training trumps inductive bias"** — ViT가 사전학습 후 작은 데이터셋으로 fine-tune될 때 SOTA에 도달합니다. 최고 모델은 ImageNet 88.55%, ImageNet-ReaL 90.72%, CIFAR-100 94.55%, VTAB 77.63%.

**English**
The introduction frames the gap. In NLP, the Transformer (Vaswani 2017) plus pre-train-then-fine-tune paradigm (Devlin 2019, Brown 2020) has scaled to 100B+ parameters with no saturation. In vision, CNNs still rule (LeCun 1989, Krizhevsky 2012, He 2016), and attempts to introduce attention either combine it with CNNs (Wang 2018, Carion 2020) or replace convolutions in pieces (Ramachandran 2019, Wang 2020). The authors instead try the **most direct port**: split an image into patches, treat each patch as a token, and feed to a vanilla Transformer.

On mid-scale data (ImageNet) without heavy regularization, ViT is a few points below same-size ResNets — expected, since Transformers lack CNN inductive biases (translation equivariance, locality). But on **larger datasets (14M–300M images)** the picture flips: large-scale training trumps inductive bias. The best ViT model reaches 88.55% (ImageNet), 90.72% (ImageNet-ReaL), 94.55% (CIFAR-100), 77.63% (VTAB).

### Part II: Related Work (§2) / 관련 연구

**한국어**
저자들은 세 갈래의 선행 연구를 정리합니다.

**(a) NLP의 Transformer 사전학습 패러다임**: BERT(denoising self-supervised), GPT(language modeling), T5 등이 대규모 사전학습 후 fine-tuning이 표준임을 확립.

**(b) 이미지에서의 self-attention 시도**:
- **Local self-attention** (Parmar et al. 2018): 각 query 픽셀의 지역 이웃 안에서만 attention
- **Stand-alone self-attention** (Hu 2019, Ramachandran 2019, Zhao 2020): 합성곱을 self-attention으로 완전 대체
- **Sparse Transformer** (Child 2019): 전역 attention의 sparse 근사
- **Axial attention** (Ho 2019, Wang 2020a): 한 축씩 attention
- 대부분 특수한 attention 패턴 때문에 하드웨어 가속이 어려움

**(c) ViT에 가장 가까운 연구**: Cordonnier et al.(2020)은 2×2 패치를 추출해 full self-attention을 적용했지만, 작은 해상도에만 가능했고 대규모 사전학습은 없었습니다. ViT는 (i) **16×16 패치로 중간 해상도 처리**, (ii) **대규모 사전학습 + fine-tune** 두 점에서 차별화됩니다.

**기타 관련**: iGPT(Chen 2020a)는 픽셀 자체를 Transformer로 모델링한 생성 모델이며 ImageNet 72%; ViT는 supervised image classification으로 88%+. BiT(Kolesnikov 2020), Noisy Student(Xie 2020)는 ImageNet-21k/JFT 사전학습의 CNN 강자 — 직접 비교 대상.

**English**
Three threads. **NLP pre-training**: BERT, GPT, T5 — pre-train large + fine-tune. **Self-attention for images**: local attention (Parmar 2018), stand-alone self-attention (Hu/Ramachandran/Zhao 2019–2020), Sparse Transformer (Child 2019), axial attention (Ho 2019, Wang 2020a) — most need specialized attention patterns hard to accelerate. **Closest predecessor**: Cordonnier et al. (2020) — 2×2 patches with full attention, but only at small resolution and without large-scale pre-training. ViT differs by (i) 16×16 patches that handle mid-resolution images and (ii) JFT-300M-scale pre-training. iGPT (Chen 2020a) models pixels autoregressively (72% ImageNet); BiT and Noisy Student are the CNN champions ViT beats.

### Part III: Method — Vision Transformer (§3.1) / 비전 트랜스포머

**한국어**
이 섹션이 논문의 심장입니다. ViT는 4개 식으로 완전히 정의됩니다 (Figure 1).

#### 3.1.1 패치 임베딩 (Eq. 1) / Patch embedding

표준 Transformer는 1D 토큰 시퀀스를 입력받으므로, 2D 이미지를 시퀀스로 변환해야 합니다.

이미지 $x \in \mathbb{R}^{H \times W \times C}$를 $(P, P)$ 크기의 패치들로 reshape:

$$x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}, \qquad N = \frac{HW}{P^2}$$

여기서 $N$은 패치 수이자 Transformer의 입력 시퀀스 길이. 각 패치 $x_p^i$를 학습 가능한 선형 투영 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$로 $D$-차원 임베딩으로 변환. **구현 팁**: 이는 `Conv2d(in_channels=C, out_channels=D, kernel_size=P, stride=P)`와 정확히 동일합니다 (kernel과 stride가 같으므로 겹치지 않는 패치).

#### 3.1.2 [class] 토큰과 위치 임베딩 (Eq. 1) / [class] token and position embedding

BERT의 [CLS] 토큰을 그대로 차용 — 학습 가능한 임베딩 $x_{\text{class}} \in \mathbb{R}^D$를 시퀀스 맨 앞에 붙입니다. 그 후 학습 가능한 1D 위치 임베딩 $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$를 더합니다:

$$z_0 = [\,x_{\text{class}};\ x_p^1 E;\ x_p^2 E;\ \cdots;\ x_p^N E\,] + E_{\text{pos}} \tag{1}$$

**위치 임베딩 선택**: 저자들은 1D learnable, 2D learnable, relative positional embedding을 비교했지만 차이가 미미해 가장 단순한 1D를 채택. 흥미롭게도 학습이 끝나면 위치 임베딩이 행/열 구조를 자동으로 학습합니다 (Figure 7 center).

#### 3.1.3 Transformer encoder (Eq. 2, 3) / 트랜스포머 인코더

표준 pre-LN Transformer encoder를 $L$번 반복:

$$z'_\ell = \mathrm{MSA}(\mathrm{LN}(z_{\ell-1})) + z_{\ell-1}, \qquad \ell = 1, \ldots, L \tag{2}$$

$$z_\ell = \mathrm{MLP}(\mathrm{LN}(z'_\ell)) + z'_\ell, \qquad \ell = 1, \ldots, L \tag{3}$$

**MSA (Multi-Head Self-Attention)**: $h$개의 head, 각각 $D_h = D/h$ 차원. ViT-Base는 $h=12$, $D_h=64$. 표준 scaled dot-product attention.

**MLP block**: $\mathrm{Linear}(D \to 4D) \to \mathrm{GELU} \to \mathrm{Linear}(4D \to D)$. 4× 확장은 BERT/GPT와 동일.

**LayerNorm 위치**: pre-LN(블록 입력에 LN), residual 연결은 블록 출력 후. Wang 2019, Baevski & Auli 2019 따름. 이는 안정적 학습을 보장.

#### 3.1.4 분류 head (Eq. 4) / Classification head

$$y = \mathrm{LN}(z_L^0) \tag{4}$$

마지막 층에서 [class] 위치($i=0$)의 출력을 LayerNorm 한 것이 image representation $y \in \mathbb{R}^D$. 사전학습 시에는 $y$ 위에 **MLP with one hidden layer**(GELU)를, fine-tuning 시에는 **single linear layer** $\mathbb{R}^D \to \mathbb{R}^K$를 붙임 ($K$ = 클래스 수).

**English**
ViT is fully defined by four equations and Figure 1.

**Patches.** Reshape $x \in \mathbb{R}^{H\times W\times C}$ into $N = HW/P^2$ patches $x_p^i \in \mathbb{R}^{P^2\cdot C}$ and project with learnable $E \in \mathbb{R}^{(P^2\cdot C)\times D}$. Equivalent to `Conv2d(C, D, kernel=P, stride=P)`.

**[class] token & positions.** Prepend a learnable $x_{\text{class}}$ and add a 1D learnable $E_{\text{pos}} \in \mathbb{R}^{(N+1)\times D}$ — Eq. (1).

**Encoder.** $L$ pre-LN blocks alternating MSA and MLP — Eqs. (2)–(3). MSA has $h$ heads of dim $D_h = D/h$ (e.g., $h=12$, $D_h=64$ for ViT-B). MLP expands 4× with GELU.

**Head.** Take the final [class]-state, normalize, then attach an MLP-head (pre-train) or linear head (fine-tune) — Eq. (4).

#### 3.1.5 Inductive bias / 귀납적 편향

**한국어**
저자들의 가장 중요한 디자인 철학적 진술입니다. **CNN에서는 locality, 2D neighborhood 구조, translation equivariance가 모든 layer에 baked-in**되어 있습니다. ViT에서는:
- **MLP layer만 local + translation equivariant** (Linear layer)
- **Self-attention layer는 global** — 전역 상호작용 가능
- **2D 구조는 두 곳에서만 사용**: (i) 패치 분할 시, (ii) fine-tuning 시 다른 해상도를 위한 위치 임베딩 보간
- **그 외에는 모두 데이터로부터 학습**해야 함 — 위치 임베딩 초기화 시 2D 정보 없음, 모든 공간 관계는 학습됨

이것이 ViT가 작은 데이터에서 약하지만 큰 데이터에서 빛나는 이유입니다.

**English**
A pivotal design statement. CNNs bake locality, 2D neighborhood, and translation equivariance into every layer. In ViT, only MLP layers are local + equivariant; self-attention is global. 2D structure is injected only at (i) patch extraction and (ii) positional-embedding interpolation when fine-tuning at higher resolution. Everything else — including all spatial relationships — must be learned from data. This explains both ViT's weakness on small data and its triumph on large data.

#### 3.1.6 Hybrid 아키텍처 / Hybrid architecture

**한국어**
대안: 원시 패치 대신 CNN feature map의 패치를 사용. CNN은 stage 4까지 통과한 뒤 그 출력의 1×1 또는 더 큰 패치를 ViT에 입력. 작은 컴퓨트 예산에서는 hybrid가 약간 좋지만 **큰 모델에서는 차이가 사라집니다** (§4.4 결과).

**English**
Alternative: feed ViT with CNN feature-map patches (e.g., from ResNet stage 4) instead of raw image patches. Slightly better at small compute budgets, gap vanishes at scale.

### Part IV: Fine-tuning at Higher Resolution (§3.2) / 더 높은 해상도에서의 fine-tuning

**한국어**
사전학습 후 작은 데이터셋으로 fine-tune할 때, **사전학습보다 큰 해상도**로 fine-tune하면 성능이 향상됩니다 (Touvron 2019, Kolesnikov 2020에서도 확인됨).

**문제**: 패치 크기 $P$를 유지하면 더 큰 해상도에서 패치 수 $N$이 증가 → 시퀀스 길이가 늘어남. ViT는 임의 시퀀스 길이를 처리할 수 있지만, **사전학습된 위치 임베딩의 의미가 깨집니다**.

**해결**: 위치 임베딩을 **원래 이미지에서의 위치에 따라 2D 보간(interpolation)**. 즉 16×16 그리드의 학습된 위치 임베딩을 24×24 그리드로 bicubic 보간. 사전학습 head는 버리고, $D \times K$ feedforward layer를 0으로 초기화하여 새로 부착.

**ImageNet 결과(Table 2)**: ViT-L/16은 512 해상도, ViT-H/14는 518 해상도로 fine-tune; Polyak-Juditsky averaging($\eta=0.9999$) 추가.

**English**
Fine-tuning at higher resolution (Touvron 2019, Kolesnikov 2020) improves transfer. Keep patch size $P$ → sequence length grows. The model handles arbitrary length, but the pre-trained $E_{\text{pos}}$ no longer matches. Solution: 2D-interpolate $E_{\text{pos}}$ according to its original spatial location. This and patch extraction are the only places ViT uses 2D inductive bias. For Table 2, ViT-L/16 is fine-tuned at 512 and ViT-H/14 at 518, with Polyak averaging $\eta=0.9999$.

### Part V: Experimental Setup (§4.1) / 실험 설정

**한국어**

**데이터셋**:
| Dataset | Classes | Images | Role |
|---|---|---|---|
| ILSVRC-2012 (ImageNet) | 1k | 1.3M | small-scale pre-train + downstream |
| ImageNet-21k | 21k | 14M | mid-scale pre-train |
| JFT-300M | 18k | 303M | large-scale pre-train (Google internal) |
| ImageNet-ReaL | 1k | — | cleaned-up ImageNet labels (Beyer 2020) |
| CIFAR-10/100 | 10/100 | 50k+10k | downstream |
| Oxford-IIIT Pets | 37 | 7k | downstream |
| Oxford Flowers-102 | 102 | 8k | downstream |
| **VTAB (19 tasks)** | varied | 1k/task | low-data transfer (Natural/Specialized/Structured) |

**중복 제거**: 사전학습 데이터를 다운스트림 테스트 셋과 중복 제거 (Kolesnikov 2020).

**모델 변형 (Table 1)**:

| Model | Layers $L$ | Hidden $D$ | MLP dim | Heads $h$ | Params |
|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

표기: ViT-L/16 = Large, 패치 크기 16. **시퀀스 길이는 패치 크기의 제곱에 반비례**하므로 작은 패치는 더 비쌉니다 (예: 224 입력에서 P=16이면 196 토큰, P=14이면 256 토큰).

**Baseline CNN**:
- **ResNet (BiT)**: ResNet의 BatchNorm을 GroupNorm으로 교체, standardized convolutions 사용 (Kolesnikov 2020)
- **Hybrid**: ResNet50 backbone(stage 4까지 또는 변형) → 1-pixel patches → ViT

**학습 설정**:
- **Pre-training**: Adam ($\beta_1=0.9, \beta_2=0.999$), batch 4096, weight decay 0.1 (높음 — transfer에 도움), linear warmup + decay
- **Fine-tuning**: SGD with momentum, batch 512
- ViT-L/16, ViT-H/14는 ImageNet에서 각각 512/518 해상도로 fine-tune

**Metrics**:
- **Fine-tuning accuracy**: 다운스트림에서 fine-tune 후 정확도
- **Few-shot accuracy**: frozen representation을 $\{-1, +1\}^K$로 매핑하는 regularized least-squares closed-form 해 — 빠른 평가용

**English**
Datasets: ImageNet (1.3M), ImageNet-21k (14M), JFT-300M (303M) for pre-training; ImageNet/ImageNet-ReaL/CIFAR/Pets/Flowers/VTAB for downstream. Pre-training data de-duplicated against test sets. Three ViT sizes (Table 1) — Base/Large/Huge with 86M/307M/632M params. Notation ViT-L/16 = Large with patch size 16; sequence length scales as $1/P^2$. Baseline = ResNet-BiT (BN → GN, standardized convs); hybrids feed ViT with CNN feature-map patches. Pre-training: Adam, batch 4096, weight decay 0.1. Fine-tuning: SGD-momentum, batch 512. Two metrics: fine-tuning accuracy and few-shot via closed-form ridge regression on frozen features.

### Part VI: Comparison to State-of-the-Art (§4.2) / SOTA 비교

**한국어**
**Table 2 — 주요 결과 (정확도 %)**:

| Dataset | ViT-H/14 (JFT) | ViT-L/16 (JFT) | ViT-L/16 (I21k) | BiT-L (R152x4) | Noisy Student |
|---|---|---|---|---|---|
| **ImageNet** | **88.55** ± 0.04 | 87.76 ± 0.03 | 85.30 ± 0.02 | 87.54 ± 0.02 | 88.4/88.5 |
| **ImageNet-ReaL** | **90.72** ± 0.05 | 90.54 ± 0.03 | 88.62 ± 0.05 | 90.54 | 90.55 |
| **CIFAR-10** | **99.50** ± 0.06 | 99.42 ± 0.03 | 99.15 ± 0.03 | 99.37 ± 0.06 | — |
| **CIFAR-100** | **94.55** ± 0.04 | 93.90 ± 0.05 | 93.25 ± 0.05 | 93.51 ± 0.08 | — |
| **Oxford-IIIT Pets** | **97.56** ± 0.03 | 97.32 ± 0.11 | 94.67 ± 0.15 | 96.62 ± 0.23 | — |
| **Oxford Flowers-102** | 99.68 ± 0.02 | **99.74** ± 0.00 | 99.61 ± 0.02 | 99.63 ± 0.03 | — |
| **VTAB (19 tasks)** | **77.63** ± 0.23 | 76.28 ± 0.46 | 72.72 ± 0.21 | 76.29 ± 1.70 | — |
| **TPUv3-core-days** | 2.5k | 0.68k | 0.23k | 9.9k | 12.3k |

**핵심 관찰**:
1. **ViT-H/14는 거의 모든 데이터셋에서 SOTA** — Flowers-102만 ViT-L/16에 0.06%p 뒤짐
2. **컴퓨트 효율성이 압도적**: ViT-H/14 2.5k vs BiT-L 9.9k vs Noisy Student 12.3k → 최대 5배 적은 사전학습 비용
3. **ViT-L/16 + ImageNet-21k** (공개 데이터)도 BiT-L에 근접 — 공개 데이터셋만으로 약 30일 TPUv3 8-core로 학습 가능

**VTAB 분해 (Figure 2)**: Natural / Specialized / Structured 3개 그룹. ViT-H/14는 Natural과 Structured에서 BiT-R152x4를 능가, Specialized에서는 비슷.

**English**
Table 2 results: ViT-H/14 leads on every dataset except Flowers-102 (where ViT-L/16 wins by 0.06pp), with ImageNet 88.55%, CIFAR-100 94.55%, VTAB 77.63%. Compute: ViT-H/14 took 2.5k TPUv3-core-days vs BiT-L 9.9k and Noisy Student 12.3k — up to 5× cheaper. Even ViT-L/16 pre-trained on the *public* ImageNet-21k beats most baselines and trains in ~30 days on a standard 8-core TPUv3. VTAB breakdown: ViT-H/14 wins Natural and Structured groups, ties on Specialized.

### Part VII: Pre-training Data Requirements (§4.3) / 사전학습 데이터 요구량

**한국어**
이것이 논문의 thesis를 가장 직접 보여주는 섹션입니다. 두 종류의 실험을 수행:

#### 실험 1: 사전학습 데이터셋 크기 변화 (Figure 3)

ViT 모델을 ImageNet, ImageNet-21k, JFT-300M로 사전학습하고 ImageNet으로 fine-tune. weight decay, dropout, label smoothing 세 가지 정규화를 작은 데이터셋에서 최적화.

**관찰**:
- **ImageNet 사전학습 시**: ViT-Large가 ViT-Base보다 **나쁨** — 정규화에도 불구하고 큰 모델이 작은 데이터에 overfit
- **ImageNet-21k 사전학습 시**: 두 모델이 비슷
- **JFT-300M 사전학습 시**: 큰 모델의 진가가 발휘됨; ViT-H/14가 가장 좋음
- BiT 모델(회색 영역)은 ImageNet에서 ViT를 능가하지만 큰 데이터에서 ViT가 추월

#### 실험 2: JFT 부분집합 (9M, 30M, 90M, full 300M) (Figure 4)

JFT-300M의 무작위 부분집합을 사용. 정규화 없이 (이번엔 모델 본질을 보기 위해), 하이퍼파라미터 동일. 컴퓨트 절약을 위해 few-shot linear evaluation 사용.

**관찰**:
- **9M 부분집합**: ViT-B/32가 ResNet50보다 **훨씬 나쁨** — convolutional inductive bias의 도움
- **90M 부분집합**: ViT-B/32가 ResNet50을 능가
- **ResNet152x2 vs ViT-L/16**: 같은 패턴
- **ResNet은 더 빨리 plateau**, ViT는 더 큰 데이터에서 계속 향상

**해석**: convolutional inductive bias는 작은 데이터에 유용하지만, 큰 데이터에서는 직접 학습하는 것이 충분(또는 더 좋음).

**English**
Two experiments form the heart of the paper.

**Experiment 1 (Figure 3):** pre-train ViT on ImageNet, ImageNet-21k, JFT-300M; fine-tune on ImageNet. With ImageNet pre-training, ViT-Large *underperforms* ViT-Base — overfits despite tuned regularization. With ImageNet-21k they match. With JFT-300M, large models shine and ViT-H/14 wins. BiT (shaded region) leads on ImageNet but ViT overtakes at scale.

**Experiment 2 (Figure 4):** train on 9M / 30M / 90M / 300M random JFT subsets, no extra regularization, same hyperparameters, evaluate via few-shot linear. ViT-B/32 (cheaper than ResNet50) is much worse on 9M but better from 90M up; same for ViT-L/16 vs ResNet152x2. ResNets plateau earlier; ViTs keep improving with data.

**Take-away:** convolutional inductive bias helps small data; on large data, learned patterns suffice — even outperform.

### Part VIII: Scaling Study (§4.4) / 스케일링 연구

**한국어**
JFT-300M에서 7개 ResNet, 6개 ViT, 5개 hybrid를 대조 실험. 컴퓨트 vs 평균 전이 정확도 (Figure 5).

**관찰 3가지**:
1. **ViT가 ResNet을 같은 컴퓨트에서 우월**: 동일 성능 도달에 ViT가 약 **2~4배 적은 컴퓨트** 사용
2. **Hybrid는 작은 컴퓨트에서 ViT보다 약간 좋지만 큰 모델에서는 차이 사라짐** — 의외 (CNN local feature가 어떤 크기에서도 도움이 될 줄 예상했으나 그렇지 않음)
3. **ViT는 시도한 범위 내에서 saturation 신호 없음** — 더 큰 스케일이 가능

**English**
Controlled compute-vs-accuracy on JFT (7 ResNets, 6 ViTs, 5 hybrids; Figure 5). Three findings:
1. ViT dominates ResNet at equal compute — uses 2–4× less compute for the same average accuracy.
2. Hybrids beat pure ViT at small compute, but the gap vanishes for large models.
3. ViT does not saturate within the tested range — more scale should help.

### Part IX: Inspecting Vision Transformer (§4.5) / ViT 내부 들여다보기

**한국어**
세 가지 분석 (Figure 7).

**(a) 학습된 패치 임베딩 필터 (Figure 7 left)**: ViT-L/32의 첫 28개 principal component를 보면, 패치 내부 미세 구조의 plausible basis function처럼 생김 — Gabor-like, edge-like 필터. CNN 첫 layer와 비슷한 패턴이 학습으로 등장.

**(b) 위치 임베딩 유사도 (Figure 7 center)**: 학습된 위치 임베딩 사이의 cosine similarity를 시각화. **2D 토폴로지가 자동으로 학습됨** — 가까운 패치는 비슷한 임베딩, 같은 행/열 구조 출현, 큰 그리드에서는 sinusoidal 구조도 보임. 이것이 1D learnable 위치 임베딩이 2D-aware 변형보다 큰 차이가 없는 이유.

**(c) Attention distance (Figure 7 right)**: attention weight로 가중평균한 픽셀 거리. CNN의 receptive field에 대응. 결과:
- **일부 head는 첫 layer부터 전역적으로 attend** — 정보를 전역으로 통합하는 능력 활용
- **다른 head는 낮은 layer에서 small attention distance** — locality에 가까움
- Hybrid 모델에서는 highly localized attention이 덜 두드러짐 → 초반 CNN layer가 비슷한 역할
- **깊이가 깊어질수록 attention distance 증가** → 점진적으로 더 전역적

**Figure 6**: 출력 토큰에서 입력 픽셀로의 attention map — 모델이 분류에 의미 있는 영역을 attend함을 보여줌 (예: 개 얼굴, 비행기 동체).

**English**
Three diagnostics (Figure 7).

**Filters (left):** PCA of learned patch-embedding $E$ shows Gabor/edge-like basis functions — CNN-like first-layer patterns emerge from data alone.

**Positional embeddings (center):** cosine similarity matrix reveals learned 2D topology — closer patches have more similar embeddings, row/column structure appears, sinusoidal patterns for larger grids. Explains why 1D learnable beats hand-crafted 2D variants — the model learns better structure.

**Attention distance (right):** attention-weighted pixel distance, analogous to CNN receptive field. Some heads attend globally even in layer 1; others stay local; in hybrids, localized heads disappear (the CNN already does it). Distance grows with depth.

**Figure 6** shows attention from the output token to the input — semantically relevant regions (dog face, plane fuselage) light up.

### Part X: Self-Supervision (§4.6) / 자기지도

**한국어**
BERT의 masked language modeling을 모방한 **masked patch prediction**을 시도. ViT-B/16를 자기지도 사전학습한 후 ImageNet fine-tune:
- **자기지도 사전학습**: 79.9% (from-scratch 대비 +2%p)
- **지도 사전학습**: ~83.97% (4%p 차이)

여전히 지도 사전학습이 우월하지만 자기지도도 가능성을 보임. Contrastive 사전학습(SimCLR, MoCo, CPC)은 future work로 남김. (이후 BEiT, MAE, DINO가 이 방향으로 큰 성공을 거둠.)

**English**
A preliminary masked-patch-prediction experiment (BERT-style) on ViT-B/16: self-supervised pre-training + ImageNet fine-tuning reaches 79.9% — +2pp over from-scratch but ~4pp behind supervised pre-training. Contrastive variants left as future work. (Later picked up by BEiT, MAE, DINO with great success.)

### Part XI: Conclusion (§5) / 결론

**한국어**
저자들은 **이미지 특화 inductive bias 없이** Transformer를 직접 적용했고(패치 추출 단계만 예외), 대규모 사전학습과 결합할 때 SOTA에 도달함을 보였습니다. 남은 과제: (1) 객체 탐지·분할 등 다른 비전 태스크로의 확장 (DETR과 함께 유망), (2) 자기지도 사전학습 (현재 격차 4%), (3) 더 큰 ViT로의 스케일링.

**English**
Direct, almost-unmodified Transformer for images, plus large-scale pre-training, gives SOTA. Open directions: detection/segmentation; self-supervised pre-training (4% gap); further scaling.

### Part XII: Worked Example — 224×224 Image with 16×16 Patches / 워크스루: 224×224 이미지

**한국어**
ViT-Base/16에 224×224 RGB 이미지를 입력하는 전체 흐름을 추적합니다.

**설정**: $H=W=224$, $C=3$, $P=16$, $D=768$, $L=12$, $h=12$, $D_h=64$, MLP dim 3072.

**Step 1 — Patch count**:
$$N = \frac{HW}{P^2} = \frac{224 \times 224}{16 \times 16} = \frac{50{,}176}{256} = 196 \text{ patches}$$

**Step 2 — Patch flattening**: 각 패치는 $16 \times 16 \times 3 = 768$ 차원 (마침 $D=768$이지만 일반적으로는 다름).

**Step 3 — Linear projection $E$**: $\mathbb{R}^{768} \to \mathbb{R}^{768}$. 결과: $x_p E \in \mathbb{R}^{196 \times 768}$.

**Step 4 — Prepend [class] + add position**:
$$z_0 = [x_{\text{class}}; x_p^1 E; \ldots; x_p^{196} E] + E_{\text{pos}}, \quad z_0 \in \mathbb{R}^{197 \times 768}$$
시퀀스 길이는 197 (= 196 + 1).

**Step 5 — Transformer encoder × 12 layers**: 각 layer의 MSA는 12 head × 64 dim. 각 layer의 입출력 모두 $\mathbb{R}^{197 \times 768}$.

**Step 6 — Classification**: $z_L^0 \in \mathbb{R}^{768}$ → LayerNorm → MLP head (사전학습) 또는 Linear (fine-tuning) → $\mathbb{R}^K$.

**파라미터 수 추정**:
- Patch embedding $E$: $768 \times 768 + 768 \approx 590k$
- Position embedding: $197 \times 768 \approx 151k$
- [class] token: 768
- Encoder block × 12: 각 약 $7M$ (MSA $4 \cdot 768^2 \approx 2.36M$ + MLP $2 \cdot 768 \cdot 3072 \approx 4.72M$ + LN/biases) → 12 × 7M = 84M
- **Total ≈ 86M** (Table 1과 일치).

**컴퓨트 추정 (forward, 단일 이미지)**:
- MSA per block: $\mathcal{O}(N^2 D) = 197^2 \times 768 \approx 30M$ FLOPs
- MLP per block: $\mathcal{O}(N \cdot D \cdot 4D) = 197 \times 768 \times 3072 \approx 465M$ FLOPs
- 12 layers → 약 6 GFLOPs forward pass.

**시퀀스 길이의 의미**: 패치 크기가 16에서 14로 줄면 $N = 256$ 토큰, attention 비용은 $(256/197)^2 \approx 1.69$배 증가.

**English**
Trace ViT-B/16 on a 224×224 RGB image.

- **Patch count:** $N = 224^2/16^2 = 196$.
- **Patch dim:** $16 \times 16 \times 3 = 768$.
- **Projection $E$:** $\mathbb{R}^{768} \to \mathbb{R}^{768}$, output $x_p E \in \mathbb{R}^{196 \times 768}$.
- **Add [class] + pos:** $z_0 \in \mathbb{R}^{197 \times 768}$.
- **12 encoder blocks:** input/output stay $\mathbb{R}^{197 \times 768}$; MSA = 12 heads × 64 dim.
- **Head:** $z_L^0 \in \mathbb{R}^{768}$ → LN → linear → $\mathbb{R}^K$.
- **Parameters:** ~86M, matching Table 1 (12 blocks × ~7M each).
- **Forward cost:** MSA per block $\sim 197^2 \cdot 768 = 30M$ FLOPs; MLP per block $\sim 197 \cdot 768 \cdot 3072 = 465M$ FLOPs; total ~6 GFLOPs.
- **Patch size ↓ from 16 to 14:** $N = 256$, attention cost ×1.69.

---

## 3. Key Takeaways / 핵심 시사점

1. **이미지를 16×16 단어로 다루기 / Treat images as 16×16 words**
   ViT의 단순함이 천재성입니다 — 이미지를 패치 그리드로 자르고 NLP 토큰처럼 다루면, 표준 BERT-스타일 Transformer가 비전에 그대로 작동합니다. 별도 비전용 attention 변형도, 합성곱도 필요 없습니다. / The simplicity is the genius — chop image into patch grid, treat as NLP tokens, and standard BERT-style Transformer works for vision. No specialized attention, no convolutions.

2. **규모가 inductive bias를 이긴다 / Scale trumps inductive bias**
   ImageNet(1.3M)에서는 CNN이 더 좋지만, JFT-300M(303M)에서는 ViT가 SOTA. 충분한 데이터가 주어지면 모델이 locality, translation equivariance를 직접 학습할 수 있고, 그 학습이 hand-crafted bias보다 더 유연합니다. / On ImageNet (1.3M), CNNs win; on JFT-300M (303M), ViT wins. With enough data, the model learns locality and equivariance — and that learning is more flexible than hand-coded biases.

3. **컴퓨트 효율성이 놀라움 / Surprising compute efficiency**
   ViT-H/14의 사전학습 컴퓨트 2.5k TPUv3-core-days는 BiT-L의 9.9k, Noisy Student의 12.3k보다 적습니다 — 큰 모델이 더 적은 자원으로 더 좋은 결과. CNN의 spatial inductive bias가 오히려 비효율의 원인일 수 있음을 시사. / ViT-H/14 trains in 2.5k TPUv3-core-days vs BiT-L's 9.9k and Noisy Student's 12.3k — bigger model, less compute, better result. Suggests CNN's spatial bias can be a compute liability at scale.

4. **위치 임베딩이 2D 토폴로지를 학습한다 / Position embeddings learn 2D topology**
   1D learnable position embedding을 줘도, 학습 후 cosine similarity에서 2D row/column 구조와 (큰 그리드에서) sinusoidal 패턴이 출현. 2D-aware 변형이 큰 이득을 못 주는 이유 — 모델이 이미 충분히 학습. / Despite being 1D learnable, position embeddings end up encoding the 2D grid in their similarity structure. Explains why hand-crafted 2D variants give no clear gain.

5. **[class] 토큰은 BERT 그대로 / The [class] token is borrowed verbatim from BERT**
   학습 가능한 토큰을 시퀀스 맨 앞에 붙이고 그 출력을 분류 표현으로 사용. global average pooling 대안도 있지만 BERT 스타일이 일관된 NLP↔CV 전이를 유지. / A learnable token prepended to the sequence; its post-encoder state is the image representation. GAP works too, but [class] keeps the NLP↔CV recipe symmetric.

6. **High-resolution fine-tuning + position interpolation / 고해상도 fine-tuning과 위치 보간**
   사전학습보다 큰 해상도로 fine-tune하면 성능이 향상 (Touvron 2019 따름); 위치 임베딩은 2D bicubic 보간. 이 한 가지 트릭이 Table 2의 88.55%에 결정적 기여. / Fine-tuning at higher resolution boosts accuracy; pre-trained $E_{\text{pos}}$ is 2D-interpolated. Crucial for the 88.55% headline.

7. **Hybrid의 흥미로운 한계 / The hybrid surprise**
   CNN backbone + ViT는 작은 컴퓨트에서 약간 좋지만 **큰 모델에서 차이 사라짐**. 직관과 반대 — local feature processing이 어떤 크기에서도 도움이 될 줄 예상했으나 그렇지 않음. 큰 ViT는 자체적으로 충분한 표현력. / Hybrids (CNN→ViT) help at small compute but the gap vanishes for large models. Counter-intuitive — convolutional local feature processing isn't a free lunch at scale.

8. **Attention head가 다양한 receptive field를 학습 / Attention heads learn diverse receptive fields**
   첫 layer부터 일부 head는 전역적, 다른 head는 지역적. 깊이에 따라 평균 attention distance가 증가. CNN처럼 "shallow=local, deep=global" 패턴을 데이터로 자율 학습. Hybrid에서는 지역 head가 줄어듦 — CNN이 이미 그 일을 함. / Some heads attend globally from layer 1; others stay local; mean distance grows with depth — CNN-like "shallow=local, deep=global" emerges from data. In hybrids, local heads disappear since the CNN already provides that.

---

## 4. Mathematical Summary / 수학적 요약

### Core forward pass / 핵심 forward 수식 체인

**Step 1 — Patchify + linearly embed (Eq. 1)**
$$x \in \mathbb{R}^{H\times W\times C} \;\Longrightarrow\; x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}, \quad N = \frac{HW}{P^2}$$

**Step 2 — Build input sequence with [class] and positions (Eq. 1)**
$$z_0 = [\,x_{\text{class}};\ x_p^1 E;\ x_p^2 E;\ \cdots;\ x_p^N E\,] + E_{\text{pos}}$$
- $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ (linear projection / 패치 임베딩 행렬)
- $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ (1D learnable / 학습 가능 위치 임베딩)
- $x_{\text{class}} \in \mathbb{R}^D$ (learnable [class] token / 학습 가능 토큰)
- $z_0 \in \mathbb{R}^{(N+1) \times D}$

**Step 3 — Encoder block: pre-LN + MSA + residual (Eq. 2)**
$$z'_\ell = \mathrm{MSA}(\mathrm{LN}(z_{\ell-1})) + z_{\ell-1}$$

**Step 4 — Encoder block: pre-LN + MLP + residual (Eq. 3)**
$$z_\ell = \mathrm{MLP}(\mathrm{LN}(z'_\ell)) + z'_\ell$$
- Repeated for $\ell = 1, \ldots, L$
- MLP $= \mathrm{Linear}(D \to 4D) \circ \mathrm{GELU} \circ \mathrm{Linear}(4D \to D)$

**Step 5 — Image representation (Eq. 4)**
$$y = \mathrm{LN}(z_L^0)$$
- $z_L^0$ = final-layer state at the [class] position
- 사전학습: $\hat{p} = \mathrm{softmax}(\mathrm{MLP}_{\text{head}}(y))$
- Fine-tuning: $\hat{p} = \mathrm{softmax}(W y + b)$ with $W \in \mathbb{R}^{K \times D}$ zero-initialized

### MSA in detail / 멀티헤드 self-attention 상세

For sequence $z \in \mathbb{R}^{n \times D}$ ($n = N+1$) and head $i = 1, \ldots, h$:

$$Q_i = z W_i^Q, \quad K_i = z W_i^K, \quad V_i = z W_i^V \quad (W_i^{Q,K,V} \in \mathbb{R}^{D \times D_h})$$

$$A_i = \mathrm{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{D_h}}\right) \in \mathbb{R}^{n \times n}$$

$$\mathrm{head}_i = A_i V_i \in \mathbb{R}^{n \times D_h}$$

$$\mathrm{MSA}(z) = [\mathrm{head}_1; \ldots; \mathrm{head}_h]\, W^O, \quad W^O \in \mathbb{R}^{h D_h \times D}$$

For ViT-Base: $D=768, h=12, D_h=64$, so $h D_h = D$.

### Patch tokenization / 패치 토큰화

$$x_p^i \in \mathbb{R}^{P^2 \cdot C}, \qquad i = 1, \ldots, N$$
- $P = 16$ (or 14 for ViT-H/14): patch side length
- $P^2 \cdot C = 256 \times 3 = 768$ for $P=16, C=3$
- $N = HW/P^2 = 196$ for $H=W=224$, $P=16$

**구현 트릭 / Implementation trick**
$$\mathrm{PatchEmbed}(x) = \mathrm{Conv2d}(x; \text{kernel}=P, \text{stride}=P, \text{out}=D)$$
이 형식이 reshape + linear과 수학적으로 동등하며 GPU에서 빠릅니다. / Mathematically equivalent to reshape-then-linear, and faster on GPU.

### Resolution change / 해상도 변경

Fine-tuning resolution $H' \times W'$이 사전학습과 다르면:
- 새 패치 수 $N' = H'W'/P^2 \neq N$
- 사전학습된 $E_{\text{pos}}$ ($\sqrt{N} \times \sqrt{N}$ 그리드 형태)를 $\sqrt{N'} \times \sqrt{N'}$로 **2D bicubic 보간**
- [class] 위치의 위치 임베딩은 그대로 유지

### Model variants (Table 1) / 모델 변형

| Symbol | Layers $L$ | Hidden $D$ | MLP dim | Heads $h$ | Head dim $D_h$ | Params |
|---|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | 64 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 64 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 80 | 632M |

**MLP dim = 4D** (BERT 컨벤션과 동일). $D_h = D / h$.

### Sequence length / 시퀀스 길이

| Image size | Patch $P$ | $N$ | $N+1$ |
|---|---|---|---|
| 224×224 | 16 | 196 | 197 |
| 224×224 | 14 | 256 | 257 |
| 384×384 | 16 | 576 | 577 |
| 512×512 | 16 | 1024 | 1025 |
| 518×518 | 14 | 1369 | 1370 |

### ASCII attention map sketch / Attention map 스케치

CLS 토큰이 196 패치에 attend하는 패턴 (ViT-L/16, 마지막 layer, Figure 6 스타일):

```
          col→  1   2   3   4   5   6   7   8   9   10  11  12  13  14
          ┌────────────────────────────────────────────────────────────┐
   row 1  │ .   .   .   .   .   .   .   .   .   .   .   .   .   .  │
   row 2  │ .   .   .   .   ░   ░   .   .   .   .   .   .   .   .  │
   row 3  │ .   .   .   ░   ▒   ▓   ░   .   .   .   .   .   .   .  │
   row 4  │ .   .   ░   ▒   █   █   ▓   ░   .   .   .   .   .   .  │   ← object
   row 5  │ .   .   ░   ▓   █   █   ▓   ░   .   .   .   .   .   .  │     present
   row 6  │ .   .   .   ░   ▒   ▓   ▒   .   .   .   .   .   .   .  │
   row 7  │ .   .   .   .   .   ░   .   .   .   .   .   .   .   .  │
   ...
          └────────────────────────────────────────────────────────────┘
   Legend:  .   < 0.001   ░  0.001-0.005   ▒  0.005-0.02   ▓  0.02-0.05   █  > 0.05
```
Attention from CLS concentrates on semantically meaningful regions (e.g., a dog's face, a plane's body), not uniformly across the grid.

### Positional embedding similarity sketch / 위치 임베딩 유사도

7×7 grid (ViT-L/32 on 224 input, simplified from Figure 7 center):

```
Tile shows cosine similarity of E_pos[(r=4, c=4)] with all (r', c') positions.

         c=1   c=2   c=3   c=4   c=5   c=6   c=7
   r=1  -.05  -.03   .02   .15   .02  -.03  -.05
   r=2  -.03   .05   .15   .35   .15   .05  -.03
   r=3   .02   .15   .35   .65   .35   .15   .02
   r=4   .15   .35   .65  1.00   .65   .35   .15   ← center
   r=5   .02   .15   .35   .65   .35   .15   .02
   r=6  -.03   .05   .15   .35   .15   .05  -.03
   r=7  -.05  -.03   .02   .15   .02  -.03  -.05

Pattern: closer patches → higher similarity; row/column structure visible.
```

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1989 ─ LeCun et al.: Backprop applied to handwritten zip codes (early CNN)
        │
2012 ─ Krizhevsky et al. (paper #13): AlexNet — ImageNet revolution by CNN
        │
2014 ─ Bahdanau et al. (paper #17): Attention for translation
        │
2014 ─ Simonyan & Zisserman: VGG; Szegedy et al.: GoogLeNet
        │
2015 ─ Ioffe & Szegedy (paper #19): BatchNorm
        │
2016 ─ He et al. (paper #20): ResNet — depth without degradation
        │
2017 ─ Vaswani et al. (paper #25): Transformer — "Attention is all you need" (NLP)
        │
2018 ─ Bello et al.: Attention augmented CNN; Devlin et al.: BERT — [CLS] token
        │
2019 ─ Ramachandran et al.: Stand-alone self-attention for vision
        │       Parmar et al.: Image Transformer (local attention, generation)
        │
2019 ─ EfficientNet (Tan & Le); Noisy Student (Xie et al., 2020)
        │
2020 ─ Cordonnier et al.: 2×2 patch self-attention (closest predecessor)
        │       Carion et al.: DETR (Transformer for detection, CNN backbone)
        │       Chen et al.: iGPT (autoregressive pixel Transformer, 72% ImageNet)
        │       Kolesnikov et al.: BiT — large-scale CNN pre-training (BiT-L)
        │       Brown et al.: GPT-3 (175B params, NLP scaling proof)
        │
2020 ─ ★★★ Dosovitskiy et al.: ViT (this paper) ★★★
        │       ─ pure Transformer + JFT-300M → 88.55% ImageNet
        │
2021 ─ Touvron et al.: DeiT — data-efficient ViT (no JFT)
        │       Liu et al.: Swin Transformer — hierarchical window attention
        │       Tolstikhin et al.: MLP-Mixer (no attention either!)
        │       Bao et al.: BEiT — masked image modeling (BERT-style for vision)
        │       Radford et al.: CLIP — ViT vision encoder for vision-language
        │
2021 ─ Caron et al.: DINO — self-supervised ViT, segmentation emerges
        │
2021 ─ Jumper et al. (paper #37): AlphaFold 2 — Transformer + structure
        │
2022 ─ He et al.: MAE — masked autoencoders (random 75% mask, ViT decoder)
        │
2023 ─ Kirillov et al.: SAM — Segment Anything (ViT backbone)
        │       Alayrac et al.: Flamingo, GPT-4V — ViT as vision encoder for LLMs
        │
2024+─ ViT descendants in nearly every multimodal foundation model (Gemini, Claude, GPT-4)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#13 Krizhevsky et al. (2012) — AlexNet** | ViT가 직접 도전한 CNN 패러다임의 시작 / The CNN paradigm ViT directly challenges | High — "CNN을 버려도 된다"의 출발점 / Foundational baseline being replaced |
| **#17 Bahdanau et al. (2014) — Attention** | ViT는 attention의 가장 일반적 형태인 self-attention을 비전에 적용 / ViT applies attention's most general form (self-attention) to vision | High — 개념적 직계 조상 / Conceptual ancestor |
| **#18 Kingma & Ba (2014) — Adam** | ViT 사전학습에 Adam 사용 (β1=0.9, β2=0.999, weight decay 0.1) / Adam used for pre-training (β1=0.9, β2=0.999, wd=0.1) | Medium — 핵심 학습 도구 / Core training tool |
| **#19 Ioffe & Szegedy (2015) — BatchNorm** | ViT는 LayerNorm 사용; baseline ResNet은 BiT 스타일로 BN→GN 교체 / ViT uses LayerNorm; baseline ResNet uses GroupNorm (BiT-style) | Medium — 정규화 선택의 대비 / Normalization design contrast |
| **#20 He et al. (2015) — ResNet** | ViT의 직접적 비교 baseline (BiT는 ResNet 변형) + Transformer의 residual도 ResNet 영감 / Direct baseline (BiT = ResNet variant); Transformer's residual connections come from ResNet | High — baseline + 구조적 영향 / Baseline + structural inheritance |
| **#25 Vaswani et al. (2017) — Transformer** | ViT는 표준 Transformer encoder를 거의 수정 없이 그대로 사용 / ViT uses the standard Transformer encoder almost verbatim | Critical — 직접 부모 / Direct parent architecture |
| **#26 Kipf & Welling (2017) — GCN** | 둘 다 NLP용 패러다임을 다른 도메인(비전/그래프)에 이식 / Both transplant a paradigm into a new domain (vision/graphs) | Medium — 방법론적 유사성 / Methodological parallel |
| **#28 Devlin et al. (2018) — BERT** | [class] 토큰, pre-LN encoder, 사전학습-fine-tuning 레시피 모두 차용 / [class] token, pre-LN encoder, pre-train→fine-tune recipe all borrowed | Critical — NLP 레시피의 직접 이식 / Direct port of NLP recipe |
| **#37 Jumper et al. (2021) — AlphaFold 2** | Transformer 기반 표현 학습이 다른 과학 도메인(단백질 구조)에서도 SOTA / Transformer-based representation learning becomes SOTA in another scientific domain | High — Transformer 일반성의 또 다른 증거 / Another testament to Transformer generality |

---

## 7. References / 참고문헌

### Primary paper / 본 논문
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*. arXiv:2010.11929.
- Code & weights: https://github.com/google-research/vision_transformer

### Direct predecessors / 직접 선조
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *NIPS 2017*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL 2019*.
- Cordonnier, J.-B., Loukas, A., & Jaggi, M. (2020). On the relationship between self-attention and convolutional layers. *ICLR 2020*.

### Self-attention for vision / 비전을 위한 self-attention
- Bello, I., Zoph, B., Le, Q., Vaswani, A., & Shlens, J. (2019). Attention augmented convolutional networks. *ICCV 2019*.
- Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-end object detection with transformers (DETR). *ECCV 2020*.
- Chen, M., Radford, A., Child, R., Wu, J., & Jun, H. (2020). Generative pretraining from pixels (iGPT). *ICML 2020*.
- Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). Generating long sequences with sparse transformers. *arXiv:1904.10509*.
- Hu, H., Zhang, Z., Xie, Z., & Lin, S. (2019). Local relation networks for image recognition. *ICCV 2019*.
- Parmar, N., et al. (2018). Image Transformer. *ICML 2018*.
- Ramachandran, P., et al. (2019). Stand-alone self-attention in vision models. *NeurIPS 2019*.
- Wang, X., Girshick, R., Gupta, A., & He, K. (2018). Non-local neural networks. *CVPR 2018*.

### CNN baselines / CNN 베이스라인
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.
- Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung, J., Gelly, S., & Houlsby, N. (2020). Big Transfer (BiT): General visual representation learning. *ECCV 2020*.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks (AlexNet). *NIPS 2012*.
- LeCun, Y., et al. (1989). Backpropagation applied to handwritten zip code recognition. *Neural Computation*, 1(4), 541–551.
- Xie, Q., Luong, M.-T., Hovy, E., & Le, Q. V. (2020). Self-training with noisy student improves ImageNet classification. *CVPR 2020*.

### Datasets & evaluation / 데이터셋과 평가
- Beyer, L., Hénaff, O. J., Kolesnikov, A., Zhai, X., & van den Oord, A. (2020). Are we done with ImageNet? *arXiv:2006.07159* (ImageNet-ReaL).
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *CVPR 2009*.
- Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Technical report (CIFAR).
- Nilsback, M.-E., & Zisserman, A. (2008). Automated flower classification over a large number of classes (Flowers-102). *ICVGIP 2008*.
- Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and dogs (Pets). *CVPR 2012*.
- Sun, C., Shrivastava, A., Singh, S., & Gupta, A. (2017). Revisiting unreasonable effectiveness of data in deep learning era (JFT-300M). *ICCV 2017*.
- Zhai, X., et al. (2019). The visual task adaptation benchmark (VTAB). *arXiv:1910.04867*.

### Foundations / 기초
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. *arXiv:1607.06450*.
- Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). *arXiv:1606.08415*.
- Ioffe, S., & Szegedy, C. (2015). Batch normalization. *ICML 2015*.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
- Polyak, B. T., & Juditsky, A. B. (1992). Acceleration of stochastic approximation by averaging. *SIAM Journal on Control and Optimization*, 30(4), 838–855.
- Touvron, H., Vedaldi, A., Douze, M., & Jégou, H. (2019). Fixing the train-test resolution discrepancy. *NeurIPS 2019*.
- Wang, Q., et al. (2019). Learning deep transformer models for machine translation. *ACL 2019* (pre-LN evidence).
- Wu, Y., & He, K. (2018). Group normalization. *ECCV 2018*.

### Notable descendants / 주요 후속 연구
- Bao, H., Dong, L., & Wei, F. (2022). BEiT: BERT pre-training of image transformers. *ICLR 2022*.
- Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging properties in self-supervised vision transformers (DINO). *ICCV 2021*.
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners (MAE). *CVPR 2022*.
- Kirillov, A., et al. (2023). Segment anything (SAM). *ICCV 2023*.
- Liu, Z., et al. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. *ICCV 2021*.
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision (CLIP). *ICML 2021*.
- Tolstikhin, I., et al. (2021). MLP-Mixer: An all-MLP architecture for vision. *NeurIPS 2021*.
- Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). Training data-efficient image transformers & distillation through attention (DeiT). *ICML 2021*.
