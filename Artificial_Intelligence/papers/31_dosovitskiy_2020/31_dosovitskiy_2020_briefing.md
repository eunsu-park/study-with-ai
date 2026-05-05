---
title: "Pre-Reading Briefing: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)"
paper_id: "31_dosovitskiy_2020"
topic: Artificial_Intelligence
date: 2026-04-28
type: briefing
---

# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*. arXiv:2010.11929.
**Author(s)**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (Google Research, Brain Team)
**Year**: 2020 (ICLR 2021)

---

## 1. 핵심 기여 / Core Contribution

**한국어**
이 논문은 **순수 Transformer encoder를 이미지 분류에 직접 적용**할 수 있음을 처음으로 대규모로 입증한 작업입니다. 핵심 아이디어는 단순합니다 — 이미지를 16×16 크기의 패치들로 잘라 NLP의 단어 토큰처럼 다루고, 이 패치 시퀀스를 BERT 스타일 [class] 토큰과 1D 학습 가능 위치 임베딩과 함께 표준 Transformer encoder에 입력합니다. 합성곱(convolution)도, locality나 translation equivariance 같은 이미지 특화 inductive bias도 거의 사용하지 않습니다. 그 대신 **JFT-300M처럼 대규모 데이터셋으로 사전학습**하면 이 inductive bias의 부재가 문제가 되지 않으며, ImageNet 88.55%, CIFAR-10 99.50%, VTAB(19 태스크) 77.63%로 BiT-L, Noisy Student 같은 최첨단 CNN을 능가하면서도 사전학습 컴퓨트는 4배 가까이 적게 사용합니다. 핵심 발견은 **"규모가 inductive bias를 이긴다(scale trumps inductive bias)"**는 것입니다.

**English**
This paper demonstrates, for the first time at large scale, that a **pure Transformer encoder can be applied directly to image classification**. The idea is deceptively simple — split an image into 16×16 patches, treat each patch as if it were a word token in NLP, prepend a BERT-style learnable [class] token, add 1D learnable positional embeddings, and feed the resulting sequence into a standard Transformer encoder. No convolutions; almost no image-specific inductive biases (no locality, no translation equivariance baked into layers). The headline finding is that, when **pre-trained on sufficiently large datasets** (ImageNet-21k or JFT-300M), the absence of these biases is not a handicap — Vision Transformer (ViT) reaches 88.55% on ImageNet, 99.50% on CIFAR-10, and 77.63% on VTAB while using ~4× less pre-training compute than state-of-the-art CNNs (BiT-L, Noisy Student). The thesis distilled: **scale trumps inductive bias**.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
2020년 가을, 컴퓨터 비전은 여전히 CNN의 시대였습니다. AlexNet(2012) 이후 ResNet(2016), EfficientNet(2019), BiT(2020), Noisy Student(2020)에 이르기까지 CNN 기반 모델이 모든 주요 벤치마크를 지배했습니다. 한편 NLP에서는 Transformer(2017)와 BERT(2019), GPT-3(2020)를 거쳐 self-attention이 표준이 되었고, "대규모 사전학습 → 다운스트림 fine-tuning" 패러다임이 정착되었습니다. 비전에서도 self-attention을 도입하려는 시도는 많았으나 — Bello et al.(2019)의 attention augmentation, Ramachandran et al.(2019)의 stand-alone self-attention, Cordonnier et al.(2020)의 2×2 패치 attention — 대부분 CNN과 결합하거나 작은 해상도에 머물렀습니다. ViT는 "**CNN을 완전히 버리고 Transformer만 쓰면 어떨까?**"라는 도전을 NLP의 사전학습 레시피와 결합해 답한 첫 작업입니다.

**English**
By late 2020, computer vision was still the empire of CNNs — from AlexNet (2012) through ResNet (2016), EfficientNet (2019), BiT and Noisy Student (2020), every leaderboard was a CNN. Meanwhile in NLP, Transformers (2017), BERT (2019) and GPT-3 (2020) had made self-attention the de-facto architecture, with "large-scale pre-train → fine-tune downstream" the dominant recipe. Several attempts to import attention into vision — Bello et al. (2019, attention augmentation), Ramachandran et al. (2019, stand-alone self-attention), Cordonnier et al. (2020, 2×2 patch attention) — kept CNNs as the backbone or only handled tiny inputs. ViT's bold question: **what if we drop convolutions entirely and apply a vanilla Transformer with the NLP pre-training recipe?**

### 타임라인 / Timeline

```
2012 ─ AlexNet (Krizhevsky et al.) — CNN era begins
2014 ─ Bahdanau et al. attention; VGG; GoogLeNet
2015 ─ ResNet (He et al.) — depth
2017 ─ Transformer (Vaswani et al.) — attention is all you need (NLP)
2018 ─ BERT (Devlin et al.) — pre-train + fine-tune in NLP
2019 ─ Stand-alone self-attention (Ramachandran), Attention augmented CNN (Bello)
2020 ─ DETR (Carion et al.) — Transformer for object detection (with CNN backbone)
2020 ─ iGPT (Chen et al.) — pixel-level Transformer (autoregressive)
2020 ─ BiT (Kolesnikov), Noisy Student (Xie) — large-scale CNN pre-training
2020 ─ Cordonnier et al. — 2×2 patch self-attention (closest predecessor)
2020 ─ ★★★ ViT (this paper) — pure Transformer for images at scale ★★★
2021 ─ Swin Transformer, DeiT, MLP-Mixer, BEiT — explosion of vision Transformers
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
- **Transformer encoder** (paper #25, Vaswani et al. 2017): multi-head self-attention(MSA), MLP block, residual connection, LayerNorm
- **BERT의 [CLS] 토큰 아이디어** (Devlin et al. 2019): 시퀀스 첫 위치에 학습 가능 토큰을 두고 그 출력을 분류기에 사용
- **사전학습-fine-tuning 패러다임**: 대규모 데이터로 사전학습 → 작은 다운스트림 데이터셋으로 fine-tune
- **ResNet (paper #20)**: 비교 베이스라인의 핵심 (BiT가 ResNet의 변형)
- **Inductive bias 개념**: locality, translation equivariance는 CNN에 내재된 가정. ViT에는 거의 없음
- **이미지 데이터셋 규모 감각**: ImageNet 1.3M, ImageNet-21k 14M, JFT-300M 300M
- **VTAB 벤치마크**: 19개의 다양한 분류 태스크 (Natural / Specialized / Structured)
- **Linear projection, positional embedding, GELU, LayerNorm**

**English**
- **Transformer encoder** (paper #25, Vaswani et al. 2017): MSA, MLP, residual + LN
- **BERT [CLS] token** (Devlin 2019): a learnable token whose final state acts as the sequence representation for classification
- **Pre-train then fine-tune paradigm**: large upstream → small downstream
- **ResNet (paper #20)**: backbone for the BiT baseline
- **Inductive bias**: CNNs hard-code locality and translation equivariance; ViT does not
- **Dataset scale**: ImageNet (1.3M), ImageNet-21k (14M), JFT-300M (303M)
- **VTAB**: a 19-task transfer benchmark (Natural / Specialized / Structured)
- **Linear projection, positional embedding, GELU, LayerNorm**

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Patch embedding** | 이미지를 $P \times P$ 패치들로 나누고 각 패치를 $D$-차원 벡터로 선형 투영. 보통 `Conv2d(kernel=stride=P)`로 구현 / Splits image into $P{\times}P$ patches and linearly projects each to $D$ dims; usually realized by `Conv2d(kernel=stride=P)` |
| **[class] token** | BERT처럼 시퀀스 맨 앞에 붙이는 학습 가능 임베딩. encoder 통과 후 그 위치의 출력이 이미지 표현 / A learnable embedding prepended to the sequence; its post-encoder state is the image representation |
| **Position embedding** | 패치 순서 정보를 주입하는 벡터. ViT는 1D learnable 사용 (2D-aware보다 큰 차이 없음) / Vector injecting patch order information; ViT uses 1D learnable (2D-aware variants gave no clear gain) |
| **Multi-Head Self-Attention (MSA)** | $h$개의 attention head를 병렬로 실행, 결과를 결합. ViT-Base는 12 head, head dim 64 / Parallel attention heads concatenated; ViT-Base has 12 heads of dim 64 |
| **MLP block** | LN → Linear($D \to 4D$) → GELU → Linear($4D \to D$). Transformer의 두 번째 sub-layer / The Transformer's second sub-layer with a 4× hidden expansion and GELU |
| **Inductive bias** | 모델 구조에 내재된 가정. CNN: locality + translation equivariance. ViT: 거의 없음 → 데이터로 학습해야 함 / Architectural assumption; CNNs bake in locality + equivariance, ViT learns spatial structure from data |
| **JFT-300M** | Google 내부 데이터셋, 18k 클래스, 303M 이미지. ViT의 최강 결과는 모두 JFT 사전학습 / Internal Google dataset (18k classes, 303M images); ViT's best results all use JFT pre-training |
| **VTAB** | 19-task 전이 학습 벤치마크. Natural/Specialized/Structured 3 그룹, task당 1 000 학습 샘플 / 19-task transfer benchmark, three groups, 1 000 training samples each |
| **Hybrid model** | CNN feature map을 입력으로 받는 ViT. 작은 컴퓨트에서는 약간 우수, 큰 모델에서는 차이 사라짐 / ViT fed by CNN feature maps; slightly better at low compute, gap vanishes at scale |
| **Attention distance** | attention weight로 가중평균한 픽셀 거리. CNN의 receptive field에 대응 / Attention-weighted pixel distance, analogous to CNN receptive field |
| **Fine-tuning at higher resolution** | 사전학습보다 큰 해상도로 fine-tune하면 성능 향상. position embedding은 2D 보간 / Fine-tuning at higher resolution boosts performance; positional embeddings are interpolated in 2D |
| **ViT-B/16, L/16, H/14** | Base/Large/Huge × patch size 16 또는 14. 86M / 307M / 632M 파라미터 / Base/Large/Huge with patch size 16 or 14; 86M / 307M / 632M parameters |

---

## 5. 수식 미리보기 / Equations Preview

**한국어**

**(1) 패치 토큰화와 입력 임베딩 / Patch tokenization and input embedding**

$$z_0 = [\,x_{\text{class}};\ x_p^1 E;\ x_p^2 E;\ \cdots;\ x_p^N E\,] + E_{\text{pos}}, \qquad E \in \mathbb{R}^{(P^2 \cdot C) \times D},\ E_{\text{pos}} \in \mathbb{R}^{(N+1)\times D}$$

이미지 $x \in \mathbb{R}^{H\times W\times C}$를 $N = HW/P^2$개 패치 $x_p^i \in \mathbb{R}^{P^2 \cdot C}$로 나누고 $E$로 $D$-차원으로 선형 투영. 맨 앞에 [class] 토큰을 붙이고 위치 임베딩 $E_{\text{pos}}$를 더해 입력 시퀀스 $z_0 \in \mathbb{R}^{(N+1)\times D}$ 생성.

**(2) MSA 블록 / Multi-head self-attention block**

$$z'_\ell = \mathrm{MSA}(\mathrm{LN}(z_{\ell-1})) + z_{\ell-1}, \qquad \ell = 1, \ldots, L$$

Pre-LN 구조: LayerNorm → MSA → residual.

**(3) MLP 블록 / MLP block**

$$z_\ell = \mathrm{MLP}(\mathrm{LN}(z'_\ell)) + z'_\ell$$

MLP는 두 개의 fully-connected 층, 사이에 GELU.

**(4) 분류 표현 / Image representation**

$$y = \mathrm{LN}(z_L^0)$$

마지막 층의 [class] 위치 출력을 image representation으로 사용. fine-tuning 시 $y$ 위에 단일 linear layer를 붙여 클래스를 예측.

**English**

The four equations form the entire ViT recipe: (1) patches are linearly embedded and concatenated with a [class] token plus a learnable 1D positional embedding; (2)–(3) $L$ Transformer encoder blocks alternate MSA and MLP, each with pre-LayerNorm and a residual connection; (4) the final [class]-token state, normalized, is the image representation that feeds the classification head. Compared with a CNN, the only image-specific inductive biases are the patch grid (Eq. 1) and the optional 2D interpolation of $E_{\text{pos}}$ when fine-tuning at higher resolution.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
1. **§1 Introduction**: 핵심 메시지 두 줄 — "CNN 의존이 필수가 아니다" + "스케일이 inductive bias를 이긴다". ImageNet 88.55% 숫자만 기억해도 충분.
2. **§3.1 Method**: 가장 중요. Figure 1과 식 (1)–(4)를 정확히 이해할 것. 패치 → 선형 투영 → [class]+pos → Transformer → MLP head 흐름.
3. **§3.2 Fine-tuning at higher resolution**: position embedding의 2D 보간 트릭은 실용적으로 중요.
4. **§4.1 Setup**: Table 1 — ViT-B/L/H의 hidden, layer, head, MLP size, 파라미터 수를 메모할 것.
5. **§4.2 Comparison to SOTA**: Table 2 — JFT 사전학습 ViT-H/14가 BiT, Noisy Student를 능가하면서 컴퓨트는 적게 사용한다는 것이 핵심.
6. **§4.3 Pre-training data requirements**: Figure 3, 4가 가장 중요한 그림. ImageNet만으로는 ViT가 ResNet에 미치지 못하지만 JFT-300M에서는 추월. 이것이 논문의 thesis.
7. **§4.5 Inspecting ViT**: position embedding이 2D 토폴로지를 학습하고, 일부 attention head는 낮은 layer에서도 전역적으로 attend.
8. 부록은 큰 그림에 영향이 적음 — 본문 위주로 읽고, 필요시 부록 D(추가 분석), B(학습 디테일)만 선별.

**English**
1. **§1 Introduction**: two key sentences — pure Transformer applied to image patches works; scale beats inductive bias. The 88.55% ImageNet number is the elevator pitch.
2. **§3.1 Method**: most important. Internalize Figure 1 and Eqs. (1)–(4) — patches → linear projection → [class] + pos → Transformer encoder → MLP head.
3. **§3.2 Fine-tuning at higher resolution**: 2D-interpolation of positional embeddings — practically important.
4. **§4.1 Setup**: Table 1 — memorise ViT-B/L/H configs (12/24/32 layers, 768/1024/1280 hidden, 86M/307M/632M params).
5. **§4.2 SOTA**: Table 2 — JFT-pretrained ViT-H/14 beats BiT-L and Noisy Student with substantially less compute.
6. **§4.3 Data requirements**: Figures 3 and 4 are the heart of the paper. ViT trails ResNet on ImageNet pre-training but overtakes on JFT-300M. This is the paper's thesis in pictures.
7. **§4.5 Inspecting ViT**: positional embeddings learn 2D topology; some heads attend globally even in shallow layers.
8. Appendices add depth but not the headline. Skim D (analysis) and B (training details) only as needed; the main body is self-contained.

---

## 7. 현대적 의의 / Modern Significance

**한국어**
ViT는 컴퓨터 비전이 CNN 중심에서 Transformer 중심으로 이동하는 결정적 분기점입니다. 발표 직후 1년 안에 DeiT(데이터 효율 ViT), Swin Transformer(계층적 윈도우 attention), MLP-Mixer, BEiT/MAE(masked image modeling), CLIP/ALIGN(vision-language 사전학습), DINO(self-supervised ViT), SAM(Segment Anything) 등이 쏟아져 나왔고, 이들은 모두 ViT를 backbone으로 사용하거나 직접 확장한 작업입니다. 객체 탐지(DETR 계열), 분할(Mask2Former, SAM), 비디오(ViViT, TimeSformer), 의료 영상, 위성 영상, 멀티모달 모델(Flamingo, GPT-4V, Gemini)에서도 ViT/ViT 변형이 표준이 되었습니다. 더 큰 메시지 — **"올바른 사전학습 데이터만 충분하다면 도메인 특화 inductive bias를 학습이 대체할 수 있다"** — 는 단순한 비전 아키텍처 교체를 넘어 현대 foundation model 시대의 핵심 가설이 되었습니다. 2026년 시점에서 ViT는 이미 "고전"이며, 거의 모든 대규모 멀티모달 시스템의 vision encoder가 ViT 변형입니다.

**English**
ViT marks the inflection point at which computer vision pivots from CNNs to Transformers. Within a year of its release, DeiT (data-efficient ViT), Swin Transformer (hierarchical window attention), MLP-Mixer, BEiT/MAE (masked image modelling), CLIP/ALIGN (vision-language pre-training), DINO (self-supervised ViT), and SAM (Segment Anything) all built directly on or extended ViT. Object detection (DETR family), segmentation (Mask2Former, SAM), video (ViViT, TimeSformer), medical and satellite imaging, and multimodal models (Flamingo, GPT-4V, Gemini) now use ViT or ViT-like backbones as their default visual encoder. The deeper message — **with enough pre-training data, learned representations can replace hand-coded domain inductive biases** — became a foundational hypothesis of the modern foundation-model era, far beyond the architectural swap itself. By 2026, ViT is already a classic; nearly every large multimodal system's vision encoder is some descendant of it.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
