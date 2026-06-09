---
title: "LLNet: A Deep Autoencoder Approach to Natural Low-Light Image Enhancement"
authors: Kin Gwn Lore, Adedotun Akintayo, Soumik Sarkar
year: 2017
journal: "Pattern Recognition, Vol. 61, pp. 650-662"
doi: "10.1016/j.patcog.2016.06.008"
topic: Low_SNR_Imaging
tags: [deep-learning, autoencoder, low-light, denoising, contrast-enhancement, SSDA]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 40. LLNet — A Deep Autoencoder Approach to Natural Low-Light Image Enhancement / 자연 저조도 영상 향상을 위한 심층 오토인코더 접근

---

## 1. Core Contribution / 핵심 기여

본 논문은 **Stacked Sparse Denoising Autoencoder (SSDA)** 를 저조도 영상 향상에 처음 본격 적용한 연구로, 두 가지 변형을 제안한다: (1) **LLNet** — 한 네트워크가 contrast enhancement와 denoising을 동시에 학습; (2) **S-LLNet** — 두 작업을 분리한 두 모듈을 직렬 연결한 staged 변형. 핵심 학습 트릭은 **합성 저조도 데이터 생성**이다. 잘 노출된 표준 영상에 무작위 감마 보정 ($\gamma \sim \mathrm{Unif}(2,5)$)과 Gaussian noise ($\sigma$도 무작위)를 적용해 무한히 많은 (어두움 + 잡음, 깨끗함) 쌍을 만들어 학습한다. 이는 **paired natural low-light data 가 사실상 없다는 문제**를 우회한 핵심 기여이다.

This paper is the first serious application of a **Stacked Sparse Denoising Autoencoder (SSDA)** to low-light image enhancement, with two variants: (1) **LLNet**, a single network that jointly learns contrast enhancement and denoising; and (2) **S-LLNet**, a staged variant with two separately-trained modules in series. The crucial trick is **synthetic low-light data generation**: well-exposed images are darkened with random gamma ($\gamma \sim \mathrm{Unif}(2,5)$) and corrupted with Gaussian noise (random $\sigma$) to manufacture unlimited (dark+noisy, clean) pairs — sidestepping the absence of paired natural low-light data.

학습된 LLNet은 5개 표준 영상 (Bird, Girl, House, Pepper, Town)에서 어두움 + Gaussian noise ($\sigma=18, 25$) 조건에 대해 **HE, CLAHE, GA, HE+BM3D 모두를 PSNR/SSIM으로 능가**하며, 자연 저조도 영상 (셀폰 카메라로 lights-off 촬영)에서도 시각적으로 우월하다. **Figure 7**의 가중치 시각화는 통합 LLNet은 blob-like (contrast 특징), S-LLNet의 denoising 모듈은 noise-like (Gabor 유사) 패턴을 학습함을 보여준다.

The trained LLNet outperforms HE, CLAHE, GA, and HE+BM3D in PSNR/SSIM across five test images (Bird, Girl, House, Pepper, Town) under darkening + Gaussian noise ($\sigma = 18, 25$), and is also visually superior on natural low-light photos (a Nexus 4 cell phone with lights off). Weight-visualisation in Fig. 7 shows that the unified LLNet learns blob-like contrast features, while the S-LLNet denoising module learns noise-like (Gabor-style) features.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Motivation / 도입 (Section 1, p. 1)

- 저조도 + 저가 카메라 (감시, 군사 ISR, 모바일) 환경에서 좋은 영상이 필요. / Low-light + cheap sensors are common in surveillance, military ISR, mobile apps.
- 기존 **HE, CLAHE, gamma adjustment**: 잡음을 함께 증폭. / classical methods amplify noise.
- 기존 **denoising (BM3D, KSVD)**: contrast 향상이 제한적. / classical denoisers do not enhance contrast.
- Deep learning은 영상 분류 (Krizhevsky 2012), denoising (Vincent 2008, Burger 2012)에서 큰 성공을 거뒀지만 **저조도 enhancement에는 미적용**.
- 핵심 기여: SSDA를 LLE에 적용 + **합성 학습 데이터** 생성 절차.

### Part II: Related Work / 관련 연구 (Section 2, pp. 1-2)

- 영상 향상: HE, CLAHE (Pisano 1998), BBHE/QBHE (Kaur 2011), OCTM (Wu 2011), gamma (Gonzalez & Woods 2001).
- 영상 디노이징: BM3D (Dabov 2007), KSVD (Elad & Aharon 2006), tri-state median filter (Chen 1999), denoising AE (Vincent 2008), CNN denoising (Jain & Seung 2008).
- LLNet의 **차별점**: contrast + denoise를 동시에, 합성 darkening + 학습으로 paired data 부재 극복. / unique angle: simultaneous + synthetic darkening.

### Part III: LLNet Architecture / 아키텍처 (Section 3, pp. 2-3)

**Single DA layer:**

$$
h(x) = \sigma(Wx + b), \qquad \hat y(x) = \sigma'(W' h(x) + b')
$$

with sigmoid $\sigma(s) = (1 + \exp(-s))^{-1}$. Inputs $x \in \mathbb{R}^N$ (corrupted), targets $y \in \mathbb{R}^N$ (clean), $h \in \mathbb{R}^K$.

**SSDA structure (Figure 1):**

- 3 DA layers in encoder, 3 in decoder (mirror).
- Input patch: $17 \times 17 = 289$ pixels.
- Hidden sizes: 867 → 578 → **289 (bottleneck)** → 578 → 867 → 289.
- Greedy layer-wise pre-training (30 epochs, lr 0.1 first two layers, 0.01 last) → end-to-end fine-tuning (200 epochs at lr 0.1, then 0.01).

**LLNet vs S-LLNet:**

- LLNet: single SSDA trained on (darkened + noisy → clean).
- S-LLNet: two SSDAs in series, the first trained on (darkened-only → bright), the second on (noisy-only → clean). 더 많은 inference 시간이 들지만 잡음이 클수록 더 우수.

### Part IV: Synthetic Training Data / 합성 학습 데이터 (Section 3, pp. 3-4)

핵심 데이터 생성 절차 / Recipe:

1. 169장의 standard test images에서 patch 추출 — 총 422,500 patches (17×17).
2. 각 patch에 **gamma adjustment**:
   $$ I_{out} = A \cdot I_{in}^{\gamma}, \quad \gamma \sim \mathrm{Unif}(2, 5) $$
   $A$는 최대 픽셀 값으로 결정되는 상수.
3. **Gaussian noise** 추가:
   $$ \sigma = \sqrt{B \cdot (25/255)^2}, \quad B \sim \mathrm{Unif}(0, 1) $$
4. 변환된 patch와 원본 patch가 (input, target) 쌍.
5. 211,250 train + 211,250 validation으로 분할.

이 합성 절차의 직관: 저조도 영상은 (낮은 photon count → noise) + (적은 dynamic range → 어두운 영역의 contrast 손실)이 결합된 것. 감마 어둡힘은 후자를, Gaussian noise는 전자를 모방.

The intuition: a low-light image has both reduced dynamic range (compressed contrast in dark regions) and reduced photon count (noise). Gamma darkening simulates the former; Gaussian noise the latter.

### Part V: Loss Functions / 손실 함수 (Section 3, p. 4)

**Pre-training (per-DA):**

$$
\mathcal{L}_{DA}(\mathcal{D}; \theta) = \frac{1}{N}\sum_{i=1}^{N} \tfrac{1}{2}\|y_i - \hat y(x_i)\|_2^2 + \beta \sum_{j=1}^{K} \mathrm{KL}(\rho \| \hat\rho_j) + \frac{\lambda}{2}(\|W\|_F^2 + \|W'\|_F^2)
$$

with $\hat\rho_j = (1/N)\sum_i h_j(x_i)$ and $\mathrm{KL}(\rho \| \hat\rho_j) = \rho \log(\rho/\hat\rho_j) + (1-\rho)\log\big((1-\rho)/(1-\hat\rho_j)\big)$.

$\rho$: target average activation (sparse). $\beta$, $\lambda$: hyper-params. $\rho$, $\beta$, $\lambda$는 cross-validation으로 결정.

**Fine-tuning (full network):**

$$
\mathcal{L}_{SSDA}(\mathcal{D}; \theta) = \frac{1}{N}\sum_{i=1}^{N} \|y_i - \hat y(x_i)\|_2^2 + \frac{\lambda}{2L}\sum_{l=1}^{2L} \|W^{(l)}\|_F^2
$$

$L$: stack 깊이. 사전 학습으로 sparsity가 이미 인코딩되었기에 KL 항은 제거.

### Part VI: Inference / 추론 (Section 3, p. 4)

- 테스트 영상을 17×17 patch로 분할 (stride 3×3로 overlap).
- 각 patch를 LLNet 통과.
- 겹치는 영역은 평균하여 재조립 → fully overlapped (1×1 stride)는 결과 향상이 미미했다고 보고.

### Part VII: Evaluation / 평가 (Section 4, pp. 4-5)

- **PSNR**: $\mathrm{PSNR} = 10 \log_{10}(\mathrm{MAX}^2 / \mathrm{MSE})$ — 픽셀 레벨 정확성.
- **SSIM** (Wang+ 2004): luminance × contrast × structure — 지각 품질.
- 비교 방법: **HE, CLAHE, GA ($\gamma = 0.3$), HE+BM3D ($\sigma=25$), LLNet, S-LLNet**.

### Part VIII: Results / 결과 (Section 5, pp. 5-9)

**Table 1 (재구성, PSNR / SSIM):**

| Image | Test | HE | CLAHE | GA | HE+BM3D | LLNet | S-LLNet |
|---|---|---|---|---|---|---|---|
| Bird-D | 12.27/0.18 | 11.28/0.62 | 15.51/0.52 | 26.06/0.84 | 11.35/0.71 | 18.43/0.69 | 16.18/0.53 |
| Bird-D+GN18 | 12.56/0.79 | 9.25/0.09 | 14.63/0.13 | 14.71/0.10 | **18.95/0.13** | 17.93/0.55 | 18.60/0.54 |
| Bird-D+GN25 | 12.70/0.13 | 9.04/0.08 | 13.60/0.09 | 12.51/0.08 | 9.72/0.11 | **21.17/0.59** | 22.11/0.61 |
| Town-D | 10.17/0.36 | 17.55/0.79 | 15.04/0.65 | 9.41/0.74 | 17.70/0.76 | 22.47/0.81 | 20.31/0.71 |
| Town-D+GN25 | 10.21/0.14 | 14.22/0.20 | 12.40/0.13 | 13.73/0.17 | 16.62/0.32 | 20.11/0.51 | **24.27/0.61** |

(논문 Table 1의 발췌; 전체 5 영상 × 4 조건 → LLNet/S-LLNet이 darken+noise 조건에서 거의 항상 1, 2위.)

(Excerpt of Table 1; LLNet/S-LLNet take 1st or 2nd place under darken+noise conditions almost always.)

**Key empirical observations / 주요 정량 관찰:**

1. **순수 어둡힘만 (D)**: GA가 강한데, 이는 GA가 정확한 $\gamma$로 어둡힘을 정확히 역변환하기 때문. LLNet은 다양한 $\gamma$를 학습했기에 단일 GA만큼 정밀 역변환은 못 함. / Pure darkening: GA wins because it exactly inverts a known $\gamma$.
2. **어둡힘 + 잡음 (D+GN18, D+GN25)**: LLNet/S-LLNet이 거의 모든 경우 최고. 잡음이 커질수록 S-LLNet이 LLNet을 능가. / S-LLNet pulls ahead at higher noise.
3. **Natural low-light**: LLNet이 HE+BM3D보다 over-amplification 없이 자연스러운 결과.

**Figure 6**: relative patch size $r = d_p / d_i = (w_p^2 + h_p^2)^{1/2} / (w_i^2 + h_i^2)^{1/2}$ vs PSNR/SSIM — patch size에 최적값 존재. PSNR-optimal patch는 부드럽고 SSIM-optimal patch는 더 균형 잡힌 결과.

**Figure 7**: 첫 layer 가중치 시각화. LLNet은 blob-like (contrast), S-LLNet의 contrast 모듈은 거친 blob, denoising 모듈은 noise-like 텍스처.

**Figure 8**: USAF 1951 resolution test chart에서 lights-on (참조) vs lights-off + LLNet — LLNet이 HE보다 detail 보존 우수.

### Part IX: Discussion & Future Work / 논의 (Section 6, pp. 9-10)

- **장점**: paired natural data 불필요, 손작업 hyper-parameter 불필요, GPU 추론 ~0.42 s for 512×512.
- **한계**: Poisson noise / quantisation 미모델링; deblurring 미수행; 단일 modality; 주관적 평가 부재.
- **미래**: Poisson noise 추가, deblurring 통합, foggy/dusty scene 확장, 사용자 평가.

---

## 3. Key Takeaways / 핵심 시사점

1. **합성 학습 데이터가 결정적** / **Synthetic training data is the key** — 잘 노출된 영상에 random gamma + random Gaussian noise를 가해 무한 학습 쌍을 만드는 것이 paired natural data 부재 문제를 해결한다. / Random gamma + random Gaussian noise on bright images solves the paired-data shortage.

2. **단일 네트워크가 contrast + denoise 동시 학습 가능** / **A single network learns both** — 충분한 capacity (3 DA 층)와 합성 학습 데이터만 있으면 두 작업을 함께 처리할 수 있다. / Sufficient capacity + synthetic data lets one network do both tasks.

3. **고잡음에서는 분리(S-LLNet)가 우수** / **Stage-wise wins at high noise** — Gaussian $\sigma=25$ 같은 고잡음 조건에서 S-LLNet이 통합 LLNet을 능가. 작업 분리가 학습을 단순화. / Decoupling tasks simplifies optimisation under heavy noise.

4. **Sparsity (KL) 정규화** / **Sparsity matters during pre-training** — 비대칭 활성화 평균 $\rho$가 작은 hidden representation이 invariance와 일반화에 도움. fine-tune 단계에서는 빠짐. / KL-induced sparsity helps generalisation, dropped in fine-tuning.

5. **Patch size = receptive field 크기 trade-off** / **Patch size trades sharpness vs. denoising** — 큰 patch는 부드러운 (덜 sharp) 결과, 작은 patch는 sharp하지만 잡음 잔존. SSIM-optimal patch가 균형. / Larger patches denoise more but lose sharpness.

6. **HE는 잡음 영상에서 실패** / **HE fails on noisy inputs** — 픽셀 강도 균등화가 잡음을 함께 증폭하여 PSNR 악화. / HE amplifies noise alongside contrast.

7. **자연 저조도 영상에서도 일반화** / **Generalises to natural low-light** — 셀폰 카메라 lights-off 촬영에서 학습 분포 외 영상에도 잘 작동 — 합성 학습 paradigm의 유효성. / Validates the synthetic-to-real strategy.

8. **첫 deep-learning LLE 논문** / **First deep-learning low-light enhancement paper** — Retinex-Net, MBLLEN, Zero-DCE, EnlightenGAN으로 이어지는 흐름의 시발점. / Founding work; later models (Retinex-Net, MBLLEN, Zero-DCE, EnlightenGAN) build on it.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Per-DA forward pass / 단일 DA forward

$$
h(x) = \sigma(Wx + b), \qquad \hat y(x) = \sigma'(W' h(x) + b'),
$$

with $W \in \mathbb{R}^{K \times N}$, $W' \in \mathbb{R}^{N \times K}$, sigmoid $\sigma(s) = (1 + \exp(-s))^{-1}$.

### 4.2 Pre-training loss / 사전 학습 손실

$$
\mathcal{L}_{DA}(\mathcal{D}; \theta) = \underbrace{\frac{1}{N}\sum_{i=1}^{N} \tfrac{1}{2}\|y_i - \hat y(x_i)\|_2^2}_{\text{reconstruction}} + \underbrace{\beta \sum_{j=1}^{K} \mathrm{KL}(\rho \| \hat\rho_j)}_{\text{sparsity}} + \underbrace{\tfrac{\lambda}{2}\big(\|W\|_F^2 + \|W'\|_F^2\big)}_{\text{weight decay}}
$$

where

$$
\hat\rho_j = \frac{1}{N}\sum_{i=1}^{N} h_j(x_i), \qquad \mathrm{KL}(\rho \| \hat\rho_j) = \rho \log\frac{\rho}{\hat\rho_j} + (1-\rho)\log\frac{1-\rho}{1-\hat\rho_j}.
$$

### 4.3 Fine-tuning loss / 미세조정 손실

$$
\mathcal{L}_{SSDA}(\mathcal{D}; \theta) = \frac{1}{N}\sum_{i=1}^{N} \|y_i - \hat y(x_i)\|_2^2 + \frac{\lambda}{2L}\sum_{l=1}^{2L}\|W^{(l)}\|_F^2
$$

### 4.4 Synthetic corruption / 합성 손상

$$
I_{train} = n\big(g(I_{original})\big), \qquad g(I) = A \cdot I^{\gamma}, \;\; \gamma \sim \mathrm{Unif}(2,5),
$$

$$
n(I) = I + \mathcal{N}(0, \sigma^2), \qquad \sigma = \sqrt{B (25/255)^2}, \;\; B \sim \mathrm{Unif}(0,1).
$$

### 4.5 Gamma correction / 감마 보정

For pixel intensities normalised to $[0, 1]$:

- $\gamma > 1$: 영상이 어두워짐 / image darkens.
- $\gamma < 1$: 영상이 밝아짐 / image brightens.
- $\gamma = 1$: 항등 / identity.

Inverse mapping: $I_{rec} = (I_{dark}/A)^{1/\gamma}$.

### 4.6 Worked example / 풀이 예제

가정 / Assume: 정상 픽셀 $I_{orig} = 0.50$, 어둡힘 $\gamma = 3$, $A = 1.0$:

- $I_{dark} = 1.0 \cdot 0.50^3 = 0.125$.
- 잡음 $\sigma = 0.05$ ($B \approx 0.26$): $I_{noisy} = 0.125 + \mathcal{N}(0, 0.05^2) \approx 0.13$ (한 sample).
- LLNet target $y = 0.50$, input $x = 0.13$. 학습은 $\hat y(x) \to 0.50$이 되도록.

For PSNR: with MSE = $(0.50 - 0.49)^2 = 10^{-4}$ and MAX = 1, $\mathrm{PSNR} = 10\log_{10}(1/10^{-4}) = 40$ dB.

### 4.7 Algorithm pseudocode / 의사코드

```
Pre-training (each DA layer l):
   for epoch in 1..30:
       for batch in train:
           x ← input    (corrupted patches at depth l)
           y ← target   (clean patches at depth l)
           h ← σ(Wx + b)
           ŷ ← σ(W'h + b')
           L ← 0.5 * mean(||y - ŷ||²) + β·sum_j KL(ρ || mean_j(h))
                                        + 0.5·λ·(||W||² + ||W'||²)
           SGD step on (W, b, W', b')

Fine-tuning (full SSDA):
   for epoch in 1..200:
       for batch in train:
           x ← darkened+noisy patch
           y ← clean patch
           ŷ ← stack of all layers
           L ← mean(||y - ŷ||²) + (λ/(2L))·sum_l ||W^(l)||²
           SGD step on all weights
```

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1998 ── CLAHE (Pisano+)            local histogram eq. for mammography
2007 ── BM3D (Dabov+)              best classical denoiser of its era
2008 ── Vincent+: denoising AE     learned denoising
2010 ── Loza+: wavelet local       multi-scale low-light
2011 ── KSVD (Elad & Aharon)       dictionary-based denoising
2012 ── Xie+ SSDA                  sparse stacked DA for restoration
2012 ── AlexNet                    deep-learning revolution
   ★ 2017 ── LLNet (this paper)     first SSDA for low-light enhancement
2017 ── LLCNN, MSR-net             CNN successors
2018 ── Retinex-Net (LOL dataset)  paired low/normal data, Retinex decomp.
2018 ── MBLLEN                     multi-branch low-light enhancement
2020 ── Zero-DCE (★ paper #41)     zero-reference curve estimation
2021 ── EnlightenGAN               unpaired GAN-based enhancement
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Vincent+ (2008) — Denoising AE | LLNet의 building block / building block of LLNet | High |
| Xie+ (2012) — SSDA | sparsity 정규화 도입 / introduced sparsity-regularised SDA | Very High |
| Dabov+ (2007) — BM3D | 비교 baseline / benchmark baseline | High |
| Wei+ (2018) — Retinex-Net | LLNet 후속, paired LOL dataset 도입 / direct successor with paired data | Very High |
| Guo+ (2020) — Zero-DCE (★ paper #41) | LLNet의 paired-data 의존을 zero-reference로 해결 / removes paired-data assumption | Very High |
| Krizhevsky+ (2012) — AlexNet | deep-learning 기반 연구의 시대적 배경 / contextual milestone | Medium |
| Jiang+ (2021) — EnlightenGAN | unpaired GAN approach to LLE / unpaired alternative | High |

---

## 7. References / 참고문헌

### Additional context / 추가 맥락

**왜 sparsity가 디노이징에 도움이 되는가?** Hidden representation이 sparse하다는 것은 입력의 작은 변동 (noise) 이 활성화 패턴을 거의 바꾸지 않는다는 뜻이다. 이는 implicit denoising priors처럼 작용한다. KL divergence 정규화는 평균 활성화 $\hat\rho_j$를 작은 target $\rho$로 끌어당겨 이런 sparsity를 유도한다.

**Why does sparsity help denoising?** A sparse hidden representation means small input perturbations (noise) hardly change the activation pattern, acting as an implicit denoising prior. The KL divergence regulariser pulls the mean activation $\hat\rho_j$ toward a small target $\rho$, inducing this sparsity.

**왜 17×17 patch?** 너무 작은 patch는 충분한 contrast 정보를 포착하지 못하고, 너무 큰 patch는 학습 가능한 매개변수 폭증과 inference 시 receptive field 평균화로 sharpness 손실. Figure 6은 PSNR-optimal과 SSIM-optimal patch 크기가 다를 수 있음을 보인다 (PSNR favours smoother, SSIM favours balanced).

**Why 17×17 patch?** Smaller patches miss enough contrast context, larger patches blow up parameter count and oversmooth at inference. Figure 6 shows PSNR-optimal and SSIM-optimal patch sizes can differ — PSNR favours smoother, SSIM favours balanced.

**LLNet vs S-LLNet trade-off**: 통합 LLNet은 단일 네트워크 → 추론 빠름, 매개변수 적음, 그러나 noise level이 다양할 때 contrast와 denoise를 동시에 균형 맞추기 어려움. S-LLNet은 분리된 모듈로 각각 독립 최적화 → 잡음이 클 때 우수하지만 inference 시간 증가.

**LLNet vs S-LLNet trade-off**: integrated LLNet has lower inference cost and fewer parameters but struggles to balance contrast and denoising when noise varies. S-LLNet trains separate modules and wins under heavy noise at the cost of extra inference time.

**LLNet 가중치의 시각적 해석 (Figure 7)**: 통합 LLNet의 첫 layer는 **blob-like** 패턴 — local contrast feature ("이 픽셀이 주변보다 어두운가?"). S-LLNet의 contrast 모듈은 거친 blob, denoising 모듈은 noise-like (Gabor 비슷) — 작업 분리가 학습된 feature의 종류도 분리.

**Visual interpretation of LLNet weights (Fig. 7)**: integrated LLNet's first layer learns blob-like contrast features ("is this pixel darker than its neighbours?"). S-LLNet's contrast module learns coarse blobs, while its denoising module learns noise-like Gabor-style filters — task separation cleanly separates the learned feature types too.

**Generalisation to natural low-light**: LLNet은 합성 darken+noise로만 학습되었지만 실제 셀폰 카메라 lights-off 영상에도 잘 동작 (Figure 5) — synthetic-to-real domain gap이 좁다. 그러나 quantisation, demosaic 인공물 같은 실제 sensor effect는 합성 데이터에 없으므로 일부 한계 존재.

**Generalisation to natural low-light**: LLNet was trained purely on synthetic darken+noise but works on real phone-camera lights-off images (Fig. 5), confirming a small synthetic-to-real gap. Real sensor effects (quantisation, demosaicing artefacts) are absent from the training data, so some limitations remain.

### Comparison summary / 비교 요약

| Aspect | HE | CLAHE | GA | BM3D | LLNet | S-LLNet |
|---|---|---|---|---|---|---|
| Contrast enhancement | ✓ (global) | ✓ (local) | ✓ ($\gamma$) | × | ✓ (learned) | ✓ |
| Denoising | × | × | × | ✓ (best classical) | ✓ (joint) | ✓ (separate) |
| Hyper-parameter free | ✓ | nearly | needs $\gamma$ | needs $\sigma$ | trained | trained |
| Best for | bright background | local contrast | known $\gamma$ | clean denoising | mixed dark+noise | high noise |

### Practical recipe / 실제 사용 레시피

1. **Training data prep**: 169 standard images에서 17×17 patch 422,500개 추출. 각 patch에 random $\gamma \in \mathrm{Unif}(2, 5)$ 적용 후 Gaussian noise ($\sigma \in [0, 25/255]$) 추가. / Extract 17×17 patches and apply random gamma + noise.
2. **Pre-training**: 각 DA layer를 30 epoch greedy하게 학습 (lr 0.1 첫 두 layer, 0.01 마지막 layer). KL sparsity term 포함. / Greedy layer-wise pre-training, 30 epochs each.
3. **Fine-tuning**: 전체 SSDA를 200 epoch end-to-end 학습 (lr 0.1 → 0.01, 검증 향상 0.5% 미만일 때 정지). KL term 제거. / End-to-end fine-tuning, drop the KL term.
4. **Validation split**: 211,250 train + 211,250 val로 분할 (random shuffle). / 50/50 train/val split with random shuffle.
5. **Inference**: 입력 영상을 17×17 patch (stride 3)로 분할 → 각 patch를 LLNet에 통과 → 겹치는 영역 평균. / Patch with stride 3, average overlapping regions.
6. **Hardware**: NVIDIA TITAN X에서 30시간 학습. 추론은 512×512 영상 0.42초 on GPU. / 30 h training, 0.42 s inference for 512×512.

### Open issues / 열린 문제

- **Quantisation noise**: 8-bit quantisation 단계가 합성 학습에 누락 → 실제 cell-phone 영상에는 추가 손실.
- **Poisson noise**: low-photon 영상에서 진짜 noise 분포는 Gaussian이 아니라 Poisson → 후속 모델은 명시적 Poisson 모델링.
- **Deblurring**: 흔들림은 학습되지 않음 → motion blur 영상에는 별도 처리 필요.
- **Subjective evaluation**: 사용자 연구가 없어 정량 PSNR/SSIM과 지각 품질의 일치 여부 불명.
- **Color images**: LLNet은 grayscale 위주 — RGB 적용은 channel별 동일 모델로 단순 확장하나 color cast 가능.

### References / 참고문헌

- Lore, K. G., Akintayo, A., Sarkar, S., "LLNet: A Deep Autoencoder Approach to Natural Low-Light Image Enhancement", *Pattern Recognition* 61, 650-662 (2017). DOI: 10.1016/j.patcog.2016.06.008
- Vincent, P., Larochelle, H., Bengio, Y., Manzagol, P.-A., "Extracting and Composing Robust Features with Denoising Autoencoders", *ICML* (2008).
- Xie, J., Xu, L., Chen, E., "Image Denoising and Inpainting with Deep Neural Networks", *NIPS* (2012).
- Dabov, K., Foi, A., Katkovnik, V., Egiazarian, K., "Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering", *IEEE TIP* 16(8), 2080-2095 (2007).
- Elad, M., Aharon, M., "Image Denoising via Sparse and Redundant Representations Over Learned Dictionaries", *IEEE TIP* 15(12), 3736-3745 (2006).
- Pisano, E., et al., "Contrast Limited Adaptive Histogram Equalization", *J. Digital Imaging* 11(4), 193-200 (1998).
- Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P., "Image Quality Assessment: From Error Visibility to Structural Similarity", *IEEE TIP* 13(4), 600-612 (2004).
- Krizhevsky, A., Sutskever, I., Hinton, G. E., "ImageNet Classification with Deep Convolutional Neural Networks", *NIPS* (2012).
- Wei, C., Wang, W., Yang, W., Liu, J., "Deep Retinex Decomposition for Low-Light Enhancement", *BMVC* (2018).
- Guo, C., et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement", *CVPR* (2020).
