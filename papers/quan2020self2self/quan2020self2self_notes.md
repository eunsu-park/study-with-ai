---
title: "Self2Self with Dropout: Learning Self-Supervised Denoising from a Single Image"
authors: Yuhui Quan, Mingqin Chen, Tongyao Pang, Hui Ji
year: 2020
journal: "IEEE/CVF CVPR 2020, pp. 1890–1898"
doi: "10.1109/CVPR42600.2020.00196"
topic: Low-SNR Imaging / Self-Supervised Deep Denoising
tags: [self2self, dropout, bernoulli-sampling, single-image, ensemble-averaging, partial-convolution, variance-reduction, quan-ji]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 19. Self2Self with Dropout: Learning Self-Supervised Denoising from Single Image / 드롭아웃 기반 단일 영상 자기지도 잡음 제거

---

## 1. Core Contribution / 핵심 기여

### 한국어
Self2Self (S2S)는 **단일 잡음 영상 $\mathbf y = \mathbf x + \mathbf n$** 만으로 deep denoising network를 학습하기 위한 *Bernoulli sampling + dropout ensemble* 프레임워크다. Noise2Self/Noise2Void가 단일 영상 학습 가능성을 제시했으나 BM3D 수준에 못 미친 핵심 원인은 **단일 샘플의 큰 prediction variance**임을 지적하고, 이를 해결하기 위해 dropout을 *학습과 추론 양쪽에서 모두* 사용한다. 핵심 기여:

(i) **Bernoulli-sampled training pair**: 잡음 영상 $\mathbf y$에 Bernoulli 마스크 $\mathbf b$ (확률 $p$, 보통 $p=0.3$)을 적용해 한 쌍 $(\hat{\mathbf y}, \overline{\mathbf y}) = (\mathbf b \odot \mathbf y, (\mathbf 1 - \mathbf b) \odot \mathbf y)$를 만든다. 손실은 마스크된 픽셀 $\overline{\mathbf b}$ 위치에서만 계산:
$$
\min_\theta \sum_m \|\mathcal F_\theta(\hat{\mathbf y}_m) - \mathbf y\|^2_{\overline{\mathbf b}_m}
$$
(ii) **Bernoulli dropout in encoder-decoder**: Encoder는 partial-convolution(PConv, Liu 2018) — 마스크된 픽셀 처리에 적합. Decoder는 모든 conv 층에 element-wise dropout (확률 0.3). Dropout이 **train과 test 양쪽에서 활성** — N2V/N2S의 blind-spot 효과와 ensemble averaging 동시 제공.

(iii) **추론 시 ensemble averaging** (Eq. 9):
$$
\mathbf x^* = \frac{1}{N}\sum_{n=1}^N \mathcal F_{\theta_n}(\mathbf b_{M+n} \odot \mathbf y)
$$
$N=50$회의 forward pass (각자 다른 dropout mask + Bernoulli sample) 평균 → **MSE = bias² + variance**의 variance를 dramatically 줄임.

(iv) **이론적 근거 (Proposition 1)**: 잡음이 zero-mean이면
$$
\mathbb E_{\mathbf n}\sum_m \|\mathcal F_\theta(\hat{\mathbf y}_m) - \mathbf y\|^2_{\overline{\mathbf b}_m} = \sum_m \|\mathcal F_\theta(\hat{\mathbf y}_m) - \mathbf x\|^2_{\overline{\mathbf b}_m} + \sum_m \|\boldsymbol\sigma\|^2_{\overline{\mathbf b}_m}
$$
즉 self-supervised loss = supervised loss + 잡음 분산 (Noise2Self의 Eq. 2와 동일 구조).

(v) **성능 (Table 1, Set9 $\sigma=25$)**: Ours **31.74/0.956** — non-learning BM3D(31.67/0.955) 추월, single-image N2S/N2V(29.51/29.40) +2 dB 우위, dataset-trained N2N(31.33) 추월, dataset-trained DnCNN(31.42)과 동등. **단일 영상 self-supervised로 BM3D를 처음 능가한 방법**.

### English
Self2Self (S2S) trains a deep denoising network from a **single noisy image** by combining (1) Bernoulli sampling of pixel pairs as input/target, (2) dropout in the decoder applied during *both* training and test, and (3) ensemble averaging of $N \approx 50$ stochastic predictions. Identifies single-sample variance as the key bottleneck of N2V/N2S and tackles it via dropout-based ensembling — a Bayesian-network interpretation. Outperforms BM3D and dataset-trained N2N on Set9 (σ=25), the first single-image self-supervised method to do so. Partial convolutions in the encoder handle Bernoulli-masked pixels cleanly.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction / 도입

#### 한국어
- **Goal**: $\mathcal F_\theta: \mathbf y \to \mathbf x$ 학습을 *오직* $\mathbf y$ 자신만으로 수행.
- **MSE 분해**: $\text{MSE} = \text{bias}^2 + \text{variance}$. Single-image regime에서는 *variance*가 폭발한다 — 학습 샘플이 사실상 1개.
- **Blind-spot trick (N2V, N2S)** 은 identity mapping convergence를 막아 *bias*는 잡지만 variance는 미해결.
- **Solution**: *dropout* — 단일 NN을 *수많은 sub-network의 ensemble*로 해석 (Gal & Ghahramani 2016 Bayesian dropout). 다양한 dropout mask 하의 prediction은 어느 정도 statistical independence → 평균으로 variance 감소.
- **Bernoulli sampling**의 두 역할: (a) 마스크된 위치에서만 loss 계산해 identity mapping 회피 (N2V 대안), (b) input과 target 차이를 stochastic하게 만들어 dropout과 함께 randomness 증가.

#### English
Self2Self addresses the *variance* component of MSE in single-image denoising — a problem N2V/N2S only partially solve via blind-spot training (which addresses bias / identity convergence). Dropout, applied in both training and testing, gives the network a Bayesian-ensemble interpretation; averaging $N$ stochastic forward passes reduces prediction variance. Bernoulli pixel sampling generates many input/target pairs from a single image and avoids identity mapping.

---

### Part II: §2 Related Work / 관련 연구

#### 한국어
- **Non-learning**: TV regularisation, BM3D (Dabov 2007), WNNM (Gu 2014). Single-image, no training data — but hand-crafted priors.
- **Supervised on clean/noisy pairs**: DnCNN, FFDNet — 대량 paired data 필요.
- **Pairwise noisy** (Noise2Noise): 두 독립 잡음 측정 필요.
- **N2V (#17), N2S (#18)**: blind-spot or J-invariant — 단일 영상 가능하나 BM3D에 못 미침.
- **DIP** (Ulyanov 2017): random input → CNN → image. Early-stopping 민감, BM3D 미달.

#### English
Survey of non-learning denoisers (BM3D, WNNM), supervised pairs (DnCNN), Noise2Noise, blind-spot single-image (N2V, N2S, Probabilistic-N2V, Laine'19), and Deep Image Prior. S2S positions itself as the first single-image method matching/exceeding BM3D.

---

### Part III: §3 Main Body / 본론

#### 한국어 — §3.1 NN architecture (Fig. 1)

- **Input**: $H \times W \times C$. 첫 layer = Bernoulli dropping (input mask).
- **Encoder (5 EBs)**: 각 EB = PConv + LeakyReLU + 2×2 max-pool stride 2. Channel 수 48 고정. Last EB: PConv + LeakyReLU.
- **Bottleneck**: $H/32 \times W/32 \times 48$.
- **Decoder (5 DBs)**: 첫 4 DBs = upsample(×2) + concat (skip from encoder) + 두 dropout-Conv + LeakyReLU. Channel 96. 마지막 DB는 dropout-Conv 3개 (64, 32, C 채널). All decoder Conv layers have **dropout p=0.3**.
- **Partial convolution**: 마스크된 픽셀의 영향을 정규화. Standard conv는 마스크된 위치에서 feature가 깎여 나옴 → PConv가 마스크 분포를 따라 normalising factor 곱함.

#### English — §3.1
Encoder–decoder UNet with skip connections. Encoder uses partial convolutions (Liu 2018) to handle Bernoulli-masked input cleanly. Decoder applies element-wise dropout ($p=0.3$) in every Conv layer. Encoder fixed at 48 channels, decoder uses 96 channels with output Conv reducing to image channels.

#### 한국어 — §3.2 Training Scheme

**Bernoulli-sampled instance**: $\hat{\mathbf y} = \mathbf b \odot \mathbf y$, $\overline{\mathbf y} = (\mathbf 1 - \mathbf b) \odot \mathbf y$ with $\mathbf b \sim \text{Bern}(p)$, $p = 0.3$.

**Loss** (Eq. 7):
$$
\min_\theta \sum_{m=1}^M \|\mathcal F_\theta(\hat{\mathbf y}_m) - \overline{\mathbf y}_m\|^2_{\overline{\mathbf b}_m}
$$
where $\|\cdot\|^2_{\overline{\mathbf b}} = \|(\mathbf 1 - \mathbf b) \odot \cdot\|^2$. Loss는 *마스크되지 않은(target) 위치*에서만 계산 — N2V/N2S의 blind-spot loss와 동일 정신.

**Proposition 1** (Eq. 8): zero-mean noise 하에
$$
\mathbb E_{\mathbf n}\sum_m \|\mathcal F_\theta(\hat{\mathbf y}_m) - \overline{\mathbf y}_m\|^2_{\overline{\mathbf b}_m} = \sum_m \|\mathcal F_\theta(\hat{\mathbf y}_m) - \mathbf x\|^2_{\overline{\mathbf b}_m} + \sum_m \|\boldsymbol\sigma\|^2_{\overline{\mathbf b}_m}
$$
self-sup loss가 supervised + constant noise variance로 분해 → **N2S 정리(Eq. 2)의 단일 영상 버전**.

**Data augmentation**: input image flipping (horizontal, vertical, diagonal). 4 versions of $\mathbf y$ 사용.

#### English — §3.2
Bernoulli dropout on the input (probability $p=0.3$) generates $\hat{\mathbf y}$; loss is computed only at masked positions $\overline{\mathbf b}$. Proposition 1 mirrors Batson's: under zero-mean noise, the empirical self-supervised loss equals the true supervised loss plus a constant noise variance term. Augmentation: flip horizontally, vertically, diagonally → 4 versions.

#### 한국어 — §3.3 Denoising / Test scheme (Eq. 9)

추론 시 standard practice는 dropout을 OFF 하고 weight를 scale하나, S2S는
$$
\mathbf x^* = \frac{1}{N}\sum_{n=1}^N \mathcal F_{\theta_n}(\mathbf b_{M+n} \odot \mathbf y)
$$
로 *dropout을 끄지 않고* $N=50$ forward pass 평균. 각 forward pass:
- 다른 dropout mask → 다른 sub-network $\mathcal F_{\theta_n}$
- 다른 Bernoulli mask $\mathbf b_{M+n}$ → 다른 input

→ Approximately statistically independent predictions → **variance ↓ $\sim 1/N$**.

#### English — §3.3
At test time, dropout remains active (not scaled-off). $N=50$ forward passes with different dropout masks and different Bernoulli input masks are averaged. Each pass yields an approximately-independent prediction; variance scales $\sim 1/N$. Ablation (Fig. 6): $N \to \infty$ gives saturating PSNR plateau.

---

### Part IV: §4 Experiments / 실험

#### 한국어 — §4.1 Implementation

- All Conv: 3×3, stride 1, zero-padding 2, LeakyReLU(0.1).
- Dropout p = 0.3, Bernoulli sampling p = 0.3.
- Adam, lr = $10^{-5}$, 4.5×10⁵ training steps.
- Test: $N=50$. 1.2 hours / 256×256 image on RTX 2080Ti.
- 데이터셋: Set9 (9 images), BSD68, PolyU (real-world).

#### English — §4.1
Same architecture for all experiments; dropout 0.3, Bernoulli 0.3, Adam lr 1e-5, 4.5×10⁵ steps. Test: 50 forward passes. ~1.2 h per 256×256 image on RTX 2080Ti.

#### 한국어 — §4.2 Blind Gaussian Denoising (AWGN)

| Set9 $\sigma$ | KSVD | PALM-DL | (C)BM3D | DIP* | N2V(1) | N2S(1) | **Ours** | N2V | N2S | N2N | (C)DnCNN |
|----------------|------|---------|---------|------|--------|--------|----------|-----|-----|-----|----------|
| 25             |30.00 |29.84    |31.67    |30.77 |28.12   |29.30   |**31.74** |30.66|30.05|31.33|31.42 |
| 50             |26.50 |26.64    |28.95    |26.01 |27.25   |28.23   |**29.25** |27.81|27.51|28.94|28.84 |
| 75             |24.29 |24.55    |27.36    |24.18 |25.85   |26.64   |**27.61** |25.99|26.49|27.42|27.36 |
| 100            |23.12 |23.18    |26.04    |23.55 |24.67   |25.41   |**26.27** |25.37|25.46|26.45|26.30 |

핵심:
- σ=25: Ours 31.74 > BM3D 31.67 — **첫 single-image self-sup가 BM3D 능가**.
- σ≥50: Ours가 BM3D 대비 +0.3-0.7 dB 일관적 우위.
- N2V(1)/N2S(1) (single-image versions) 대비 +2-3 dB 절대 우위.
- 놀랍게도 dataset-trained N2N도 추월. 저자 해석: dataset noise patterns 다양성이 noise NN에 misleading bias 도입할 수 있음.

#### English — §4.2
Set9 σ=25: Self2Self 31.74 dB beats BM3D 31.67. Dominates N2V(1)/N2S(1) by ~2 dB. Surprisingly outperforms dataset-trained N2N — authors hypothesise: dataset noise diversity can mislead the network when test-image noise distribution differs.

#### 한국어 — §4.3 Real-world denoising (PolyU)
| Metric | CBM3D | TWSC | DIP | N2V | N2S | CDnCNN | Ours |
|--------|-------|------|-----|-----|-----|--------|------|
| PSNR   | 36.98 | 36.10| 36.95|34.08|35.46| **37.55**|37.52 |
| SSIM   | 0.977 |0.963 |0.975|0.954|0.965|**0.983**|0.980 |

Real noise (smartphone)에서도 single-image S2S가 dataset-trained CDnCNN과 거의 동일.

#### English — §4.3
Real-world PolyU dataset: S2S nearly matches dataset-trained CDnCNN (37.52 vs 37.55 dB) — a strong result for a single-image method.

#### 한국어 — §4.4 Salt-and-pepper / Inpainting (Set11)
| Drop ratio | CSC | DIP | Ours |
|-----------|-----|-----|------|
| 50%       |32.97|33.48|**35.14**|
| 70%       |28.44|28.50|**31.06**|
| 90%       |24.34|24.34|**25.91**|

Salt-and-pepper noise는 corrupted pixels 전체 정보 손실. Bernoulli sampling을 *uncorrupted* 픽셀에서만 수행하면 inpainting과 동등.

#### English — §4.4
Salt-and-pepper / impulse noise as inpainting: Bernoulli sample only uncorrupted pixels. S2S beats DIP by 2-3 dB on 50-90% drop.

#### 한국어 — §4.5 Ablation (Set9 σ=25)
| Ablation             | PSNR (dB) | SSIM |
|----------------------|----------|------|
| **Full Ours**         | **31.74**|**0.956**|
| w/o dropout (train+test) | 23.88 | 0.658 |
| w/o ensemble (test=1 pass)| 29.92 | 0.932 |
| w/o Bernoulli sampling   | 23.12 | 0.744 |
| w/o PConv (standard conv)| 31.26 | 0.938 |

해석:
- **Dropout train+test 핵심**: -7.86 dB 손실 (variance 폭발).
- **Test-time ensemble**: -1.82 dB. Train-time dropout 만으로도 29.92 → ensemble이 추가 +1.82.
- **Bernoulli sampling 핵심**: -8.62 dB 손실 (identity mapping convergence).
- **PConv는 minor +0.5 dB**.

#### English — §4.5 Ablation
- Dropout (both stages): essential, -7.86 dB without.
- Test-time ensemble: -1.82 dB (variance reduction).
- Bernoulli sampling: essential, -8.62 dB (avoids identity mapping).
- Partial convolution: minor +0.5 dB contribution.

#### 한국어 — Stability over iterations (Fig. 7)
DIP의 PSNR은 iteration에 매우 민감 (peak 후 급격히 하락). S2S는 충분 iteration 후 평탄 — practitioner가 stopping criterion 걱정 없음. 실용성에서 큰 advantage. 구체적으로 Lena ($\sigma=25$), Baboon ($\sigma=25$), Peppers ($\sigma=50$), Kodim12 ($\sigma=50$) 네 이미지 모두에서 1500 iteration 이후 PSNR이 monotone non-decreasing plateau에 도달. DIP 곡선은 동일 이미지에서 sharp peak 이후 1.5-2 dB hill-down — 즉 stopping iteration 선택이 PSNR에 직접 영향. S2S의 안정성은 *Bernoulli sampling이 학습 분포를 끊임없이 randomise* 하기 때문 — 단일 sample에 over-fit할 통계적 기회가 없음.

#### English — Fig. 7
Unlike DIP (PSNR drops sharply past optimum), S2S PSNR remains stable after sufficient training — no need for early stopping. Across Lena (σ=25), Baboon (σ=25), Peppers (σ=50), Kodim12 (σ=50), all four PSNR curves reach a monotone non-decreasing plateau after ~1500 iterations. DIP curves on the same images peak then drop by 1.5-2 dB — stopping criterion has direct PSNR impact. S2S stability stems from Bernoulli sampling continuously randomising the training distribution; no statistical opportunity to over-fit a single training sample.

#### 한국어 — Computational cost / 계산 비용
$256 \times 256$ 영상 1장 처리에 RTX 2080Ti에서 ~1.2시간. 학습 4.5×10⁵ steps, 추론 50회 forward pass. DIP보다 4-5배 느리나 PSNR 우위 큼. 실시간 응용에는 부적합 — high-quality batch processing 도구. 단일 GPU에서 여러 이미지를 *parallel*로 처리해 throughput 보완 가능.

#### English — Computational cost
~1.2 hours per 256×256 image on RTX 2080Ti. 4.5×10⁵ training steps + 50 test forward passes. 4-5× slower than DIP but PSNR advantage is large. Not suitable for real-time use — a high-quality batch-processing tool. Throughput can be improved by processing multiple images in parallel on a single GPU.

---

## 3. Key Takeaways / 핵심 시사점

1. **Variance is the bottleneck of single-image self-supervised denoising** — N2V/N2S는 *bias* (identity mapping)을 잡지만 *variance*는 미해결. Single training sample → MSE = bias² + Var, Var이 폭발. **Ablation: dropout 없을 때 -7.86 dB**. / Variance, not bias, is the dominant error in single-image regimes — dropout addressing this is the key novelty over N2V/N2S.

2. **Dropout = approximate Bayesian model averaging** — Gal-Ghahramani 2016의 dropout-as-Bayesian 시각으로, 학습된 NN은 무수한 sub-network의 ensemble. Train+test 양쪽에서 dropout 활성화 → $N$ forward 평균이 posterior expectation 근사. / Dropout interpreted as approximate Bayesian inference; train-and-test dropout enables Monte-Carlo ensemble averaging that reduces prediction variance $\sim 1/N$.

3. **Bernoulli sampling = generalised blind spot** — N2V는 single pixel을 mask, S2S는 $\sim 70\%$ 픽셀을 mask. 더 많은 randomness, 더 많은 input/target diversity. Loss는 *visible* (target) 위치에서만. / Bernoulli sampling generalises N2V's single-pixel masking to ~70% of pixels — provides massive randomness in input/target and avoids identity convergence.

4. **Train-time + test-time dropout are both essential** — Ablation: train-only dropout 29.92 dB, train+test dropout 31.74 dB (+1.82). Dropout이 *학습 시 정규화* + *추론 시 ensemble* 두 역할 동시. / Dropout serves dual roles: training-time regularisation against overfitting, and test-time stochastic ensemble for variance reduction.

5. **First single-image self-sup to beat BM3D** — Set9 σ=25: 31.74 vs 31.67 dB. 단일 영상에서 hand-crafted prior (BM3D)가 깨지는 첫 sample. dataset-trained DnCNN (31.42)보다도 약간 높음. / Self2Self is the first single-image self-supervised method to outperform BM3D on Set9 (σ=25), and matches dataset-trained DnCNN.

6. **Stability over iterations vs DIP** — DIP는 PSNR이 iteration 따라 peak-then-drop. S2S는 충분히 iteration하면 plateau에서 안정. **Practical reliability**가 DIP의 가장 큰 약점을 해결. / Unlike DIP (PSNR peaks then degrades), Self2Self shows stable PSNR after sufficient iterations — no early-stopping required, making it deployable.

7. **Partial convolution helps masked-input regions** — Bernoulli-masked pixels에서 standard convolution은 zero contribution을 그대로 propagate → activation magnitude underestimation. PConv는 mask 비율에 따라 normalising → 깨끗한 feature. Minor gain but principled. / Partial convolution normalises feature magnitudes in masked regions where standard convolution underestimates activations — a small but principled improvement (+0.5 dB).

8. **Generalises to inpainting** — Salt-and-pepper noise (50%, 70%, 90% drop)을 Bernoulli-on-uncorrupted-only로 직접 처리. Inpainting과 denoising의 통합 framework. / The Bernoulli-sampling philosophy directly extends to image inpainting: sample only uncorrupted pixels, train on visible pairs. Set11 90% drop: S2S 25.91 dB > DIP 24.34 dB.

9. **$\lambda \to 0$ limit reproduces N2V** — Bernoulli p가 매우 작으면 ($p \to 1/m$) 마스크되는 픽셀이 평균 1개 → N2V의 single-pixel masking. 즉 N2V는 S2S의 *low-mask-rate boundary*. 큰 mask rate가 정보 손실은 크지만 randomness가 ensemble averaging의 bias-cancellation에 기여. / In the limit of very small Bernoulli p, only one pixel is masked per sample — recovering N2V's single-pixel scheme. N2V is the low-mask-rate boundary of S2S; the higher mask rate trades information loss for randomness that benefits ensemble averaging.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Bernoulli sampling / Bernoulli 샘플링
Given noisy image $\mathbf y \in \mathbb R^{H \times W \times C}$ and Bernoulli mask $\mathbf b \sim \text{Bern}(p)^{HWC}$:
$$
\hat{\mathbf y} = \mathbf b \odot \mathbf y, \quad \overline{\mathbf y} = (\mathbf 1 - \mathbf b) \odot \mathbf y
$$
Empirically $p = 0.3$ (30% pixels visible to network as input).

### 4.2 Training loss / 학습 손실 (Eq. 7)
$$
\boxed{\min_\theta \sum_{m=1}^M \|\mathcal F_\theta(\hat{\mathbf y}_m) - \overline{\mathbf y}_m\|^2_{\overline{\mathbf b}_m}}
$$
Loss summed only over positions $\overline{\mathbf b}_m = \mathbf 1 - \mathbf b_m$ (visible target positions).

### 4.3 Proposition 1 (loss decomposition) / 정리 1 (손실 분해, Eq. 8)
For zero-mean noise with per-pixel std $\boldsymbol\sigma$:
$$
\boxed{\mathbb E_{\mathbf n}\sum_m \|\mathcal F_\theta(\hat{\mathbf y}_m) - \overline{\mathbf y}_m\|^2_{\overline{\mathbf b}_m} = \sum_m \|\mathcal F_\theta(\hat{\mathbf y}_m) - \mathbf x\|^2_{\overline{\mathbf b}_m} + \sum_m \|\boldsymbol\sigma\|^2_{\overline{\mathbf b}_m}}
$$
- LHS: empirical self-supervised loss (computable).
- 1st RHS: true supervised loss against clean $\mathbf x$.
- 2nd RHS: noise variance — constant in $\theta$.
→ minimisers coincide.

### 4.4 Test-time ensemble / 추론 ensemble (Eq. 9)
$$
\boxed{\mathbf x^* = \frac{1}{N}\sum_{n=1}^N \mathcal F_{\theta_n}(\mathbf b_{M+n} \odot \mathbf y)}
$$
$\mathcal F_{\theta_n}$ = network with dropout mask $n$, $\mathbf b_{M+n}$ = Bernoulli mask $n$. $N = 50$ in paper.

### 4.5 MSE = bias² + variance / MSE 분해 (Eq. 4)
$$
\text{MSE}(\hat{\mathbf x}) = \underbrace{\|\mathbb E[\hat{\mathbf x}] - \mathbf x\|^2}_{\text{bias}^2} + \underbrace{\mathbb E\|\hat{\mathbf x} - \mathbb E[\hat{\mathbf x}]\|^2}_{\text{variance}}
$$
Ensemble averaging $\mathbf x^* = \frac{1}{N}\sum_n \hat{\mathbf x}_n$ reduces variance $\to 1/N$ under approximate independence:
$$
\text{Var}(\mathbf x^*) \approx \text{Var}(\hat{\mathbf x}_1) / N
$$
### 4.6 Worked numerical example / 수치 예시
Set9 $\sigma=25$ (intensity 0-255). Single forward pass without test ensemble: 29.92 dB (Ablation Table 4).
$$
\text{MSE}_{\text{single}} = 255^2 / 10^{29.92/10} \approx 66.3
$$
Ensemble of $N=50$: 31.74 dB
$$
\text{MSE}_{\text{ens}} \approx 43.6
$$
Variance-reduction ratio: $(66.3 - 43.6)/66.3 \approx 34\%$ of MSE was variance.

If predictions were perfectly independent: $\text{Var}_{\text{ens}} = \text{Var}_{\text{single}}/50$. Empirically the gain saturates around $N \approx 30$ (Fig. 6) → effective independent samples $\approx 30$, correlation among dropout instances $\approx 1 - 30/50 = 40\%$.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
2014  Srivastava — dropout as regularisation
                  ↓
2016  Gal & Ghahramani — dropout = approximate Bayesian inference
                  ↓
2017  Ulyanov DIP — single-image deep prior, no GT, but unstable
                  ↓
2018  Lehtinen Noise2Noise (#16) — clean GT unnecessary
                  ↓
2018  Liu — partial convolution for irregular masks
                  ↓
2018-19 Krull Noise2Void (#17) — single-image, blind-spot CNN, but < BM3D
                  ↓
2019  Batson Noise2Self (#18) — J-invariance theory, < BM3D for single-image
                  ↓
2020  Quan Self2Self (★) — Bernoulli + dropout ensemble, FIRST single-image > BM3D
                  ↓
2021  Huang Neighbor2Neighbor (#20) — sub-sample pairs + N2N loss
                  ↓
2022  Wang Blind2Unblind (#22) — visible blind spots, current SOTA
```

**위치 / Position**: S2S는 *single-image self-supervised denoising의 첫 BM3D-killer*. N2V/N2S가 이론적 가능성을 보였다면 S2S는 dropout-ensemble로 *실용적 우위*를 처음 입증. 이후 모든 single-image self-sup 방법(N2N2, Neighbor2Neighbor)이 S2S를 baseline으로 인용.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문                    | Connection / 연결                                                                                         |
|--------------------------------|---------------------------------------------------------------------------------------------------------|
| #17 Krull 2019 (Noise2Void)    | S2S의 *single-image* 정신은 N2V에서 출발; N2V의 single-pixel masking → S2S의 70% Bernoulli masking으로 일반화. |
| #18 Batson 2019 (Noise2Self)   | Proposition 1은 N2S의 Eq. 2를 Bernoulli-mask 위에서 단일 영상에 specialise. 같은 변분 골격.                    |
| #16 Lehtinen 2018 (N2N)        | Loss 정신 동일 ($\mathbb E \|f(y_1) - y_2\|^2 = \mathbb E \|f(y_1) - x\|^2 + \text{const}$). 차이: S2S는 $y_1, y_2$를 single image의 Bernoulli partition에서 만들어냄. |
| #7 Dabov 2007 (BM3D)           | S2S의 main baseline. Set9 σ=25: 31.74 vs 31.67 — single-image self-sup가 BM3D를 능가하는 첫 사례.            |
| #20 Huang 2021 (Neighbor2Neighbor) | S2S의 직접 후속; Bernoulli random masking → spatially-correlated neighbour sub-sampling. Test ensemble 불필요. |
| #15 Buchholz 2019 (Cryo-CARE)  | Cryo-EM 영상에서 S2S 적용 가능 — clean GT 부재의 prototypical regime.                                       |
| Gal & Ghahramani 2016          | Dropout-as-Bayesian의 직접 motivation. S2S의 test-time dropout은 Bayesian Monte Carlo와 동치.               |
| Liu et al. 2018 (PConv)        | S2S encoder의 building block. Irregular mask에서 standard conv의 normalisation 문제 해결.                   |

---

## 7. References / 참고문헌

- **Primary**: Quan, Y., Chen, M., Pang, T., Ji, H. "Self2Self with Dropout: Learning Self-Supervised Denoising from a Single Image". CVPR 2020, pp. 1890–1898. DOI: 10.1109/CVPR42600.2020.00196.
- **Related**:
  - Krull, A., Buchholz, T.-O., Jug, F. "Noise2Void — Learning denoising from single noisy images". CVPR 2019.
  - Batson, J., Royer, L. "Noise2Self: Blind denoising by self-supervision". ICML 2019.
  - Lehtinen, J. et al. "Noise2Noise: Learning image restoration without clean data". ICML 2018.
  - Ulyanov, D., Vedaldi, A., Lempitsky, V. "Deep image prior". CVPR 2018.
  - Srivastava, N. et al. "Dropout: a simple way to prevent neural networks from overfitting". JMLR 15(1), 2014.
  - Gal, Y., Ghahramani, Z. "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning". ICML 2016.
  - Liu, G. et al. "Image inpainting for irregular holes using partial convolutions". ECCV 2018.
  - Dabov, K. et al. "Image denoising by sparse 3-D transform-domain collaborative filtering" (BM3D). IEEE TIP 16(8), 2007.
