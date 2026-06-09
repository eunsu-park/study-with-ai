---
title: "Low-Light Image Enhancement with Wavelet-based Diffusion Models (DiffLL)"
authors: Hai Jiang, Ao Luo, Songchen Han, Haoqiang Fan, Shuaicheng Liu
year: 2023
journal: "ACM Transactions on Graphics"
doi: "10.1145/3618373"
topic: Low_SNR_Imaging
tags: [diffusion, wavelet, low-light, image-enhancement, DDPM, DWT, conditional]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 31. Low-Light Image Enhancement with Wavelet-based Diffusion Models / 웨이블릿 기반 확산 모델을 이용한 저조도 영상 향상

---

## 1. Core Contribution / 핵심 기여

DiffLL solves the dual practical problems of pixel-domain diffusion models for low-light image enhancement (LLIE) — **slow inference and chaotic content** — by moving the diffusion process into the wavelet domain. The pipeline applies $K=2$ levels of 2D Haar Discrete Wavelet Transform to the low-light image, isolating one global average coefficient $A^K_{low}$ at $1/4^K$ spatial resolution and three sets of high-frequency sub-bands $\{V^k,H^k,D^k\}$. A conditional DDPM (the **Wavelet-based Conditional Diffusion Model, WCDM**) is then trained to denoise $A^K_{low}$ given $A^K_{low}$ as condition, while the high-frequency sub-bands are restored by a lightweight **High-Frequency Restoration Module (HFRM)** that uses cross-attention between the vertical and horizontal bands to refine the diagonal band. A novel training strategy performs forward diffusion *during training* (not only at test time) so that the model learns to denoise from a pre-shaped distribution rather than from arbitrary Gaussian noise — this single trick eliminates content diversity at inference. The final output is reconstructed by inverse DWT. On LOLv1, LOLv2-real, LSRW, and UHD-LL the method sets new SOTA (LOLv1: 26.34 dB / SSIM 0.845, LOLv2-real: 28.86 dB / SSIM 0.876) while being ≥70× faster and using ≥3× less GPU memory than DDIM/Palette/WeatherDiff.

DiffLL은 픽셀 도메인 확산 모델의 두 가지 실용적 문제 — **느린 추론 속도와 무작위 노이즈에서 비롯되는 콘텐츠 혼돈** — 을 *확산을 웨이블릿 도메인으로 옮김*으로써 해결한다. 입력 저조도 이미지에 2D Haar DWT를 $K=2$번 적용하면 평균 계수 $A^K_{low}$는 원본의 $1/4^K=1/16$ 해상도가 되며, 이 작은 텐서에 대해서만 조건부 DDPM(**WCDM**)을 학습한다. 한편 수직·수평·대각 고주파 계수 $V,H,D$는 **HFRM**이 cross-attention으로 복원하는데, $V$와 $H$의 정보가 $D$에 주입되어 대각 디테일을 보강한다. 더 나아가 학습 단계에서 forward diffusion을 명시적으로 수행하는 새로운 학습 전략을 도입해, 추론 시 무작위 가우시안 시작점이 만드는 콘텐츠 다양성을 억제한다. LOLv1·LOLv2-real·LSRW·UHD-LL에서 SOTA(예: LOLv1 26.34 dB)를 달성하면서 DDIM 대비 70배 이상 빠르고 GPU 메모리는 3배 적게 쓴다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Motivation / 도입과 동기 (pp. 1-3)

The paper opens by characterising LLIE as ill-posed: the same low-light input can map to many plausible normal-light outputs, so optimisation- and learning-based methods alike struggle. Existing diffusion-based restoration (Palette, DDRM, WeatherDiff) shows promising perceptual quality but suffers from (i) >10 s inference at 600×400, (ii) artefacts and colour drift caused by the random Gaussian start of the reverse process, and (iii) memory blow-up at 2K/4K resolution. Figure 1 shows a representative case where SCI/SNRNet over-expose, Palette produces colour distortion, and WeatherDiff produces artefacts, while DiffLL preserves natural colour. The contribution list is: (1) WCDM, (2) the new training strategy that combines forward diffusion + denoising, (3) HFRM, and (4) a benchmark on five real-world datasets including UHD-LL.

본 논문은 LLIE를 ill-posed 문제로 정의하며, 기존 확산 기반 복원(Palette, DDRM, WeatherDiff)이 (i) 600×400에서 10초 이상의 추론 시간, (ii) 무작위 가우시안 시작점으로 인한 색감 왜곡과 인공물, (iii) 2K/4K 해상도에서의 메모리 문제를 갖는다고 지적한다. Fig. 1은 대표 사례를 보여 주며, DiffLL은 자연스러운 색감을 유지한다. 4가지 기여(WCDM, 새로운 학습 전략, HFRM, UHD-LL 벤치마크)가 명시된다.

### Part II: 2D Discrete Wavelet Transform Preliminaries / 2D 이산 웨이블릿 변환 (Sec. 3.1, p. 4)

The Haar wavelet transform applied once produces:

$$
\{A^1_{low}, V^1_{low}, H^1_{low}, D^1_{low}\} = \text{2D-DWT}(I_{low})
$$

with each sub-band of size $H/2 \times W/2 \times c$. Iterating $K$ times on the average coefficient yields $A^K_{low}$ of size $H/2^K \times W/2^K \times c$. A crucial *empirical justification* (Fig. 4 of the paper) is presented: if you swap the high-frequency coefficients of a low-light and a normal-light image and inverse-transform, the result still looks essentially the same as the original — the *global illumination* is carried entirely by the average coefficient $A^K$. Replacing the average coefficient, in contrast, drastically changes appearance. This experiment is the conceptual licence for the entire architecture: do diffusion only on $A^K_{low}$.

Haar DWT를 한 번 적용하면 위 식과 같이 4개의 부대역이 생기며 각각은 원본의 절반 크기다. $K$번 반복하면 $A^K_{low}$는 $1/2^K \times 1/2^K$ 크기로 줄어든다. 핵심 실증(Fig. 4)은 *고주파 계수를 저조도/정상광 영상 사이에서 바꿔치기해도 시각적으로 거의 차이가 없고, 평균 계수를 바꿔치기하면 영상이 완전히 변한다*는 것이다. 이는 *전역 조명 정보가 평균 계수에 모두 들어있음*을 의미하며, "평균 계수에서만 확산을 수행하라"는 본 논문의 핵심 결정의 근거가 된다.

### Part III: Conditional Diffusion Model Background / 조건부 확산 모델 배경 (Sec. 3.2, p. 5)

The forward process is the standard variance-preserving SDE discretisation:

$$
q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1}),\qquad q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I)
$$

with the reparametrisation

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon, \quad \bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s).
$$

The reverse process learns Gaussian transitions $p_\theta(x_{t-1}|x_t,\tilde{x}) = \mathcal{N}(\mu_\theta, \sigma_t^2 I)$ with mean

$$
\mu_\theta(x_t,\tilde{x},t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t,\tilde{x},t)\Big),
$$

trained by the simplified objective:

$$
\mathcal{L}_{simple} = \mathbb{E}_{x_0,t,\epsilon}\big[\|\epsilon - \epsilon_\theta(x_t,\tilde{x},t)\|^2\big].
$$

This is standard DDPM/conditional diffusion material and is presented for self-containment.

이 절은 표준 DDPM 자료의 요약이다. forward는 $\beta_t$ 스케줄로 가우시안 노이즈를 누적하고, reverse는 $\epsilon_\theta$를 학습하여 평균 $\mu_\theta$를 산출한다. 학습 손실은 $\|\epsilon - \epsilon_\theta\|^2$이다.

### Part IV: WCDM — Wavelet-based Conditional Diffusion Model / WCDM (Sec. 3.3, p. 5)

Two issues of vanilla conditional diffusion are diagnosed and fixed.

**Issue A: Cost.** With $T=1000$ and small $\beta_t$, the reverse process must take many steps. **Fix:** Operate in wavelet domain — $A^K_{low}$ is $4^K$ times smaller, so forward and reverse are dramatically cheaper. The authors choose $T=200$ (much less than the typical $T=1000$ for image-domain diffusion) and an implicit sampling step $S=10$.

**Issue B: Content diversity.** The reverse process starts from a random $\hat{x}_T \sim \mathcal{N}(0,I)$, so the same input can produce different outputs. **Fix:** During training, perform *both* forward diffusion (to produce $\hat{x}_T$ from the conditioned coefficient $\tilde{x}=A^K_{low}$) *and* the denoising loop — so the model learns that valid starts come from this specific forward distribution, not arbitrary Gaussian noise. This is implemented by adding a content-consistency term to the diffusion loss:

$$
\mathcal{L}_{diff} = \mathbb{E}_{x_0,t,\epsilon \sim \mathcal{N}(0,I)}\Big[\|\epsilon - \epsilon_\theta(x_t,\tilde{x},t)\|^2 + \|\hat{A}^K_{low} - A^K_{high}\|^2\Big] \tag{Eq. 9}
$$

The training is summarised in Algorithm 1 of the paper: in each iteration the model performs both a single $\epsilon_\theta$ gradient step and a full implicit denoising rollout starting from $\hat{x}_T \sim \mathcal{N}(0,I)$ with $S$ steps, then enforces the second L2 term. At inference (Algorithm 2) only the rollout is needed.

**효율 문제.** $T=1000$, 작은 $\beta_t$는 비용이 크다 → 웨이블릿 도메인의 $A^K_{low}$는 $4^K$배 작으므로 forward/reverse가 모두 싸진다. 본 논문은 $T=200$, $S=10$을 사용한다.

**콘텐츠 다양성 문제.** reverse가 무작위 가우시안에서 시작되므로 같은 입력이 다른 출력을 낼 수 있다 → 학습 단계에서 forward+reverse를 동시에 수행해, valid한 시작점은 forward가 만들어낸 분포뿐이라는 점을 모델이 학습하게 한다. 이를 위해 Eq. 9에서 $\|\hat{A}^K_{low}-A^K_{high}\|^2$ 항을 추가한다.

### Part V: HFRM — High-Frequency Restoration Module / HFRM (Sec. 3.4, p. 6)

While $A^K$ carries illumination, $V/H/D$ carry edges. Naive copy of low-light $V/H/D$ leaves blur and noise. The HFRM is a small residual network with three parallel branches (one per sub-band $V,H,D$) connected by **two cross-attention layers** that route information $V \to D$ and $H \to D$. Why? Because Haar's diagonal band $D$ tends to be the noisiest and least informative; injecting $V$ and $H$ — which contain the principal edge information in the two main directions — gives the network material to reconstruct $D$. Three depth-wise separable convolutions extract per-band features, then two cross-attention blocks fuse them, then a progressive-dilation Resblock (dilation rates $d=\{1,2,3,2,1\}$) widens the receptive field without gridding. Output is three refined sub-bands $\hat{V}^k,\hat{H}^k,\hat{D}^k$.

The full image is then reconstructed by repeated 2D-IDWT:

$$
\hat{A}^{k-1}_{low} = \text{2D-IDWT}(\hat{A}^k_{low}, \hat{V}^k_{low}, \hat{H}^k_{low}, \hat{D}^k_{low}). \tag{Eq. 10}
$$

Iterating $k$ from $K$ down to 1 gives the final restored image $\hat{A}^0_{low} = \hat{I}_{low}$.

**HFRM 구조.** 세 개의 입력 분기(V, H, D) → depth-wise separable conv → V→D, H→D 두 cross-attention → 점진적 dilation (1,2,3,2,1) Resblock → 출력. Haar의 대각 성분은 가장 노이지하므로 V/H의 edge 정보로 보강한다는 발상이다. 이후 IDWT로 차례로 합쳐 최종 영상이 나온다.

### Part VI: Network Training and Loss Functions / 학습과 손실 함수 (Sec. 3.5, p. 6)

Three losses are combined:

- **Diffusion loss** $\mathcal{L}_{diff}$ (Eq. 9): noise prediction + content consistency.
- **Detail-preserving loss** $\mathcal{L}_{detail}$ (Eq. 11): MSE between restored and reference high-frequency coefficients plus total-variation regularisation; weights $\lambda_1=0.1$, $\lambda_2=0.01$.
- **Content loss** $\mathcal{L}_{content}$ (Eq. 12): L1 + (1−SSIM) between the restored image and reference.

Total: $\mathcal{L}_{total} = \mathcal{L}_{diff}+\mathcal{L}_{detail}+\mathcal{L}_{content}$ (Eq. 13).

Training uses Adam, learning rate $10^{-4}$ decaying by 0.8 every $5\!\times\!10^3$ iters, batch 12, patch 256×256, on 4× RTX 2080Ti for $1\!\times\!10^5$ iterations. $K=2$, $T=200$, $S=10$ are defaults.

학습은 세 손실의 합을 최소화한다. $\mathcal{L}_{diff}$는 노이즈 예측+콘텐츠 일관성, $\mathcal{L}_{detail}$은 V/H/D 회복+TV 정규화, $\mathcal{L}_{content}$는 L1+SSIM이다. Adam, lr $10^{-4}$, 4×RTX 2080Ti, 100K iter로 학습된다.

### Part VII: Experiments and Ablations / 실험과 절제 연구 (Sec. 4, pp. 7-12)

**Quantitative (Table 1).** On LOLv1 the best baseline (SNRNet) achieves PSNR 24.61 / SSIM 0.842; DiffLL achieves **26.34 / 0.845** — a 1.73 dB jump and improvements on LPIPS (0.217 vs 0.259) and FID (48.11 vs 55.12). On LOLv2-real, DiffLL hits 28.86 dB / 0.876 vs 24.91 / 0.858 for the best baseline (Restormer). On UHD-LL (Table 3), DiffLL trained on LOLv1 still beats UHDFour$_{2\times}$ (a model trained on UHD-LL itself) at 21.36 vs 11.96 PSNR.

**Speed (Table 2).** At 600×400, DiffLL takes 0.157 s / 1.85 GB, while DDIM takes 16.82 s / 5.88 GB — about 100× faster. At 1920×1080, DDIM and Palette OOM; DiffLL succeeds in 1.07 s.

**Low-light face detection (Sec. 4.3, Fig. 9).** Using DiffLL as a preprocessing for the DSFD face detector raises AP from 0.264 (raw input) to **0.385** — better than every other LLIE preprocessing, demonstrating that the gain is not purely cosmetic.

**Ablations (Tables 5-8).** Wavelet scale $K=2$ is the sweet spot; $K=1$ is too costly, $K=3$ loses information. Sampling step $S=10$ is enough — larger $S$ does not help (paper notes this contradicts Palette/GDP, where $S=1000$ is typical, but DiffLL's training strategy means content diversity is already removed). Default HFRM beats two ablations (HFRM-v2 without cross-attn, HFRM-v3 with reverse routing). Removing $\mathcal{L}_{content}$ drops PSNR by 2.49 dB, the biggest single component.

**Quantitative.** LOLv1: 26.34 dB SOTA(이전 24.61). LOLv2-real: 28.86 dB. UHD-LL에서 LOLv1 학습만으로 21.36 dB. **속도.** 600×400에서 0.157 s vs DDIM 16.82 s. 1920×1080에서 DDIM/Palette는 OOM이지만 DiffLL은 1.07 s. **응용.** DiffLL을 face detection 전처리로 쓰면 AP 0.264 → 0.385. **절제.** $K=2$, $S=10$이 최적; HFRM 제거 시 성능 저하; $\mathcal{L}_{content}$가 가장 중요한 손실 항.

### Part VIII: Limitations / 한계 (Sec. 4.5, p. 12)

The authors explicitly note failure in **extreme low-light cases** where the average coefficient itself contains too little signal — diffusion cannot hallucinate well from near-zero input. They also note dependence on paired training data.

저자들은 매우 어두운 장면에서는 평균 계수 자체에 신호가 거의 없어 한계가 있음을 명시한다. 또한 paired 데이터 의존성도 언급된다.

### Part IX: Detailed Walkthrough of Algorithm 1 (Training) / 학습 알고리즘 상세 (Algorithm 1, p. 5)

The training loop, written out step-by-step:

1. Sample a low/normal pair $(I_{low}, I_{high})$ and apply $K=2$ levels of 2D-DWT to both — getting $\tilde{x}=A^K_{low}$ and $x_0=A^K_{high}$.
2. **Forward-diffusion supervision step:** sample $t\sim\mathcal{U}\{1,\ldots,T\}$ and $\epsilon\sim\mathcal{N}(0,I)$. Compute $x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$. Take a single SGD step on $\nabla_\theta\|\epsilon - \epsilon_\theta(x_t,\tilde{x},t)\|^2$. This is the *standard* DDPM gradient.
3. **Denoising rollout step:** sample $\hat{x}_T\sim\mathcal{N}(0,I)$ and execute $S$ implicit denoising iterations with stride $\Delta t = T/S$. Each iteration computes $\hat{x}_{t_{next}}$ from $\hat{x}_t$ via the closed-form deterministic update used in DDIM (Eq. 7 of paper):

$$
\hat{x}_{t_{next}} = \sqrt{\bar{\alpha}_{t_{next}}}\Big(\frac{\hat{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\hat{x}_t,\tilde{x},t)}{\sqrt{\bar{\alpha}_t}}\Big) + \sqrt{1-\bar{\alpha}_{t_{next}}}\epsilon_\theta(\hat{x}_t,\tilde{x},t).
$$

The end of the rollout produces $\hat{x}_0\equiv\hat{A}^K_{low}$. Take a second SGD step on $\nabla_\theta\|\hat{x}_0 - x_0\|^2$.

4. Repeat from 1 until convergence ($10^5$ iterations).

The rollout-during-training step (3) is the unusual ingredient — most diffusion papers train only on (2). This is what teaches the model that valid starts are *not* arbitrary Gaussian samples but specific samples consistent with $\tilde{x}$ as condition.

학습 루프 단계별 정리: ① $(I_{low}, I_{high})$ → $K$-단계 DWT → $(\tilde{x}, x_0)$. ② **Forward 감독 단계**: $t,\epsilon$ 샘플 → $x_t$ 합성 → $\nabla_\theta\|\epsilon-\epsilon_\theta\|^2$로 1회 SGD. ③ **Denoising rollout 단계**: $\hat{x}_T\sim\mathcal{N}(0,I)$에서 $S$회 implicit 샘플링(DDIM-style)으로 $\hat{x}_0$ 생성 → $\nabla_\theta\|\hat{x}_0-x_0\|^2$로 추가 SGD. ④ 수렴까지 반복. 단계 ③(학습 중 rollout)이 이 논문의 차별점이다.

### Part X: Detailed Walkthrough of HFRM Architecture / HFRM 구조 상세 (Fig. 5, p. 6)

Reading the diagram in Figure 5 of the paper carefully:

- **Inputs:** three sub-bands $V^k_{low}$, $H^k_{low}$, $D^k_{low}$, each of shape $H/2^k \times W/2^k \times c$.
- **Stage 1 (per-band feature extraction):** each input goes through one *depth-wise separable convolution* (Chollet 2017) to extract local features. Depth-wise separable convs decouple spatial and channel mixing for efficiency.
- **Stage 2 (cross-attention fusion, two layers):** the first cross-attention layer takes $D$'s features as query, $V$'s features as key/value, and updates $D$. The second cross-attention takes $D$'s features as query, $H$'s features as key/value, and updates $D$ again. The asymmetry (only $V,H$ feed into $D$, never the reverse) is justified by the ablation in Table 6: HFRM-v3 (which routes the reverse) is worse, because $D$ is the noisiest band and has less information to share.
- **Stage 3 (progressive dilation Resblock):** four 3×3 convolutions with dilation rates $\{1,2,3,2,1\}$. The increasing-then-decreasing pattern is borrowed from Hou et al. 2022 and avoids the gridding artefacts that plague constant-dilation networks. The progressive dilation gives a wide effective receptive field while preserving local information.
- **Stage 4 (output projection):** three depth-wise separable convolutions reduce channels to $c$ each, producing $\hat{V}^k$, $\hat{H}^k$, $\hat{D}^k$.
- **Skip connections** are used throughout to preserve high-frequency detail.

HFRM 구조: ① 세 입력 → depth-wise separable conv로 per-band 특징 추출. ② 두 개의 cross-attention 층, $V\to D$ 및 $H\to D$ 방향으로만 정보 주입(역방향은 ablation에서 더 나쁨). ③ 1-2-3-2-1 progressive dilation Resblock(gridding artefact 회피). ④ depth-wise separable conv 3개로 채널 축소 → $\hat{V},\hat{H},\hat{D}$ 출력. skip connection 활용.

### Part XI: Quantitative Walkthrough of Tables 1-8 / 표 1-8 정량 분석

**Table 1 (multi-dataset comparison).** Reading the LOLv1 column: NPE 16.97 dB → SRIE 11.86 → LIME 17.55 → RetinexNet 16.77 → Zero-DCE 14.86 → MIRNet 24.14 → SNRNet (best previous) 24.61 → **DiffLL 26.34**. The improvement of $+1.73$ dB over the previous best is unusually large for a mature benchmark. SSIM jumps 0.842 → 0.845, LPIPS drops 0.259 → 0.217, FID drops 55.12 → 48.11. The same pattern holds on LOLv2-real (24.91 → 28.86) and LSRW (19.28 → 19.28 PSNR, but better SSIM and LPIPS).

**Table 2 (efficiency on 600×400).** DiffLL: 0.157 s, 1.85 GB. DDIM: 16.82 s, 5.88 GB (107× slower, 3.2× more memory). Palette: 168 s (1071× slower). WeatherDiff: 52.7 s. Even at 1920×1080 where most baselines OOM, DiffLL takes 1.07 s in 3.87 GB.

**Table 5 (ablation on $K$ and $S$).** $K=1, S=10$: 26.44 dB but 0.380 s. $K=2, S=10$: 26.34 dB, 0.157 s (default). $K=2, S=20$: 26.25 dB, 0.285 s (more steps don't help). $K=3, S=10$: 25.09 dB, 0.114 s (too much downsampling loses information).

**Table 6 (HFRM ablation).** WCDM only (no HFRM): 21.98 dB / SSIM 0.729. Add HFRM at $k=1$: 23.41 / 0.773 (+1.43 dB). Add HFRM at $k=1,2$ (default): 26.34 / 0.845 (+2.93 dB total). HFRM-v2 (no V/H→D cross-attn): 24.15 / 0.822. HFRM-v3 (reverse routing, D→V/H): 25.63 / 0.824. The default V/H→D routing wins decisively.

**Table 7 (loss ablation).** Default (all three): 26.31 dB / 0.845. w/o $\mathcal{L}_{diff}$: 24.24 / 0.823 (perceptual quality drops, FID jumps 48.11→89.32). w/o $\mathcal{L}_{content}$: 23.82 / 0.808 (biggest single drop, −2.49 dB). w/o $\mathcal{L}_{detail}$: 25.41 / 0.835 (small drop, but high-frequency texture suffers).

**Table 8 (training-strategy ablation).** Vanilla (FD only at training): DDIM 16.52, Palette 11.77, WeatherDiff 17.91, Ours-FD only 18.13 — all bad. With FD+DP (the new strategy): Ours-FD+DP **26.34 dB ± 1×10⁻³**. The variance shrinks by 4 orders of magnitude — quantitative proof that content diversity is gone.

각 표 정량 정리: ① Table 1: LOLv1에서 SOTA 24.61 → 26.34 dB($+1.73$ dB). ② Table 2: DDIM 대비 107배 빠르고 3.2배 적은 메모리. ③ Table 5: $K=2,S=10$이 sweet spot. ④ Table 6: HFRM이 +2.93 dB 기여, V/H→D routing이 정답. ⑤ Table 7: $\mathcal{L}_{content}$가 가장 중요한 손실(없으면 -2.49 dB). ⑥ Table 8: FD+DP 학습 전략이 분산을 4 자릿수 줄임 — 콘텐츠 다양성 제거의 정량적 증거.

### Part XII: Connection to Modern Mobile/Edge LLIE / 모바일·엣지 LLIE와의 연결

DiffLL's release (Sept 2023) coincided with industry interest in on-device generative restoration for smartphone night-mode, AR/VR, and surveillance. Key implications:

- **Wavelet domain is hardware-friendly.** Haar transform reduces to 2×2 sums and differences — extremely cheap on mobile NPUs. SoCs from Qualcomm and Apple can compute multi-level Haar in hardware.
- **Smaller diffusion target = smaller working set.** A 16× reduction in spatial dimension means the U-Net activations fit in on-chip SRAM, removing the DRAM bandwidth bottleneck that dominates pixel-domain diffusion latency on mobile.
- **$S=10$ is reachable in real time.** Ten U-Net forward passes on a 256/16=16 spatial input is feasible at ~10 ms each on modern mobile NPUs — total ~100 ms is acceptable for interactive UX.
- **Cross-resolution generalisation.** As shown in Table 3, training on 256² works for 4K — an enormous practical advantage when capturing/training data at native sensor resolution is infeasible.

The 2024 wave of mobile LLIE papers (and several closed-source vendor implementations) directly cite DiffLL as the architectural template.

DiffLL의 발표(2023.9)는 모바일 야간모드·AR/VR·감시 영상의 *온디바이스 생성 복원* 수요와 맞물린다. ① Haar 변환은 2×2 가산/감산뿐이라 모바일 NPU에서 거의 공짜. ② 16× 공간 축소로 U-Net 활성화가 온칩 SRAM에 들어가 DRAM 대역폭 병목이 사라짐. ③ $S=10$이면 모바일 NPU에서 ~100 ms로 실시간 UX 가능. ④ 256² 학습이 4K에 일반화되어 학습 데이터 수집이 쉬움. 이 네 가지 덕에 DiffLL은 2024년 모바일 LLIE 논문들의 기본 템플릿이 되었다.

### Part XIII: Why Latent Diffusion (Stable Diffusion) Was Rejected / 왜 잠재 확산을 쓰지 않았나 (Sec. 3.3 Discussion, p. 5)

The authors explicitly compare with Latent Diffusion (Rombach 2022). Both achieve spatial-dimension reduction before diffusion. The differences:

| Property | Latent Diffusion (VAE) | DiffLL (DWT) |
|---|---|---|
| Encoder | Learned VAE — has parameters, requires retraining | Linear (Haar) — closed-form, no parameters |
| Lossiness | Encoder is lossy; reconstruction loss in VAE | Lossless (perfect reconstruction up to numerical precision) |
| Domain shift | VAE trained on natural images → must retrain on low-light | None — DWT is universal |
| Memory | VAE encoder/decoder add weights | DWT/IDWT are stateless |
| Cross-resolution | Latent shape fixed by VAE | Any resolution since DWT is recursive |

The conclusion: VAE's information loss is a deal-breaker for restoration (where every pixel matters), and the universality of the wavelet transform means the same trained model generalises across resolutions — which is exactly what Table 3 demonstrates (LOLv1-trained model still wins on UHD-LL).

DiffLL은 잠재 확산(SD)과 비교해 ① VAE는 lossy/학습 필요/재학습 필요, ② DWT는 lossless/closed-form/도메인 무관, ③ DWT는 모든 해상도에 동일 모델 적용 가능. 복원 태스크에서 VAE의 정보 손실은 치명적이며, 표 3의 cross-resolution 일반화 결과가 DWT 선택의 정당성을 입증한다.

---

## 3. Key Takeaways / 핵심 시사점

1. **The wavelet domain is a "free" latent space for diffusion.** Unlike VAE-based latent diffusion (Stable Diffusion), 2D-DWT is a *linear, lossless* transform. You get the spatial-dimension reduction without the encoder's information loss. — 웨이블릿 도메인은 VAE 잠재공간의 *무손실 대안*이다. 같은 차원 축소 이득을 보면서도 인코더가 만드는 정보 손실이 없다.

2. **Global illumination lives in the average coefficient.** The "swap experiment" (Fig. 4) is the conceptual key: swapping high-frequency coefficients changes almost nothing visually, so diffusion can ignore them and focus on $A^K$. — 전역 조명은 거의 전부 평균 계수에 들어 있다. Fig. 4의 계수 교환 실험이 이 사실을 정량적으로 증명한다.

3. **Train with the actual forward distribution, not isotropic Gaussian.** Performing forward diffusion *during training* teaches the model that valid starting points come from a specific data-conditioned distribution, eliminating the random-noise content diversity that plagues other diffusion restorations. — 학습 시에도 forward diffusion을 수행해 valid 시작 분포를 모델에 학습시킴으로써 추론 시 콘텐츠 다양성을 제거한다.

4. **Cross-attention from V/H to D restores diagonal details.** The Haar diagonal band is the noisiest; the symmetry-breaking attention routing (V/H attend into D, never the reverse) is empirically the right direction. — V·H에서 D로 향하는 cross-attention이 대각 디테일을 복원한다. HFRM-v3의 역방향 routing은 더 나쁘다.

5. **Sampling step $S=10$ is enough.** Once the training-time forward-diffusion trick is in place, large $S$ no longer helps — a counter-intuitive but practically valuable result. — 학습 전략이 콘텐츠 다양성을 제거한 덕에 $S=10$이면 충분하다. 이는 GDP·Palette($S\!\sim\!1000$)와의 큰 차이다.

6. **Restoration as a "model preprocessing" can dramatically improve perception.** AP rises from 26.4% to 38.5% on DARK FACE — useful evidence that LLIE has tangible downstream value beyond aesthetics. — 복원 전처리는 인식 성능을 의미 있게 끌어올린다. DARK FACE에서 AP 26.4% → 38.5%.

7. **Generalisation across resolutions.** A model trained on 256×256 patches from LOLv1 still beats 4K-trained models on UHD-LL, suggesting wavelet-domain diffusion learns scale-invariant illumination priors. — 작은 해상도 학습 모델이 4K 평가에서도 작동한다. 웨이블릿 도메인 확산이 *스케일 불변* prior를 학습하는 듯하다.

8. **Three loss terms are not redundant.** Ablation: removing $\mathcal{L}_{content}$ costs 2.49 dB; removing $\mathcal{L}_{diff}$ wrecks LPIPS; removing $\mathcal{L}_{detail}$ harms little but matters at high frequencies. — 세 손실은 직교적 역할을 한다. content가 PSNR을, diff가 perceptual을, detail이 high-frequency를 담당한다.

---

## 4. Mathematical Summary / 수학적 요약

**Haar 2D-DWT (one level)** — given $I \in \mathbb{R}^{H\times W \times c}$:

$$
\{A^1, V^1, H^1, D^1\} = \text{2D-DWT}(I), \qquad A^1, V^1, H^1, D^1 \in \mathbb{R}^{H/2 \times W/2 \times c}.
$$

**$K$-level recursion on the average coefficient:**

$$
\{A^k, V^k, H^k, D^k\} = \text{2D-DWT}(A^{k-1}), \quad k=1,\ldots,K.
$$

**Forward diffusion (DDPM)** with variance schedule $\{\beta_t\}_{t=1}^{T}$:

$$
q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I),\quad
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\quad \bar{\alpha}_t=\prod_{s=1}^t(1-\beta_s).
$$

**Conditional reverse mean:**

$$
\mu_\theta(x_t,\tilde{x},t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,\tilde{x},t)\Big).
$$

**WCDM training objective (Eq. 9):**

$$
\mathcal{L}_{diff} = \mathbb{E}_{x_0,t,\epsilon\sim\mathcal{N}(0,I)}\Big[\|\epsilon-\epsilon_\theta(x_t,\tilde{x},t)\|^2 + \|\hat{A}^K_{low}-A^K_{high}\|^2\Big].
$$

**Detail-preserving loss (Eq. 11):**

$$
\mathcal{L}_{detail} = \lambda_1\sum_{k=1}^K\|\{\hat{V}^k,\hat{H}^k,\hat{D}^k\}-\{V^k_{high},H^k_{high},D^k_{high}\}\|^2 + \lambda_2\sum_{k=1}^K \text{TV}(\{\hat{V}^k,\hat{H}^k,\hat{D}^k\}).
$$

**Content loss (Eq. 12):**

$$
\mathcal{L}_{content} = |\hat{I}_{low}-I_{high}|_1 + (1-\text{SSIM}(\hat{I}_{low},I_{high})).
$$

**Total loss (Eq. 13):**

$$
\mathcal{L}_{total} = \mathcal{L}_{diff} + \mathcal{L}_{detail} + \mathcal{L}_{content}.
$$

**Worked example (numerical trace).** Take a 256×256 RGB image, $K=2$. After 2 levels of Haar DWT, $A^2_{low}$ is 64×64×3 — a 16× reduction in pixels and ~16× reduction in diffusion compute. With $T=200$, $S=10$, the reverse loop performs only 10 calls to $\epsilon_\theta$ on a 64×64×3 input (rather than 1000 calls on 256×256×3 in vanilla DDPM) — a >250× compute saving for the diffusion stage alone. With the HFRM operating on three 128×128×3 inputs (after one IDWT), the heavy lifting still happens at the smallest scale.

각 식의 변수 의미: $A^K_{low}$는 K-단계 DWT의 평균 계수, $\tilde{x}$는 조건 입력(여기서는 $A^K_{low}$ 자체), $\epsilon_\theta$는 노이즈 추정 U-Net, $\hat{A}^K_{low}$는 reverse 끝의 추정값, $A^K_{high}$는 정상광 영상의 K-단계 평균 계수. Worked example: 256×256 입력의 $K=2$ 평균 계수는 64×64에 불과하므로 확산 비용이 16배 줄고, $T=200, S=10$이면 reverse 호출은 10회뿐이다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1989  Mallat — Multiresolution Wavelet Theory
                     │
                     ▼
1995  Donoho — Wavelet Thresholding (paper #02 in this series)
                     │
                     ▼
2017  Lore et al. — LL-Net (first deep LLIE)
                     │
                     ▼
2020  Ho et al. — DDPM (modern diffusion)
                     │
                     ▼
2021  Rombach et al. — Latent Diffusion / Stable Diffusion (VAE latent)
                     │
                     ▼
2022  Saharia et al. — Palette (image-to-image diffusion)
2022  Kawar et al. — DDRM (linear inverse problems with diffusion)
                     │
                     ▼
2023  Özdenizci & Legenstein — WeatherDiff (patch-based conditional)
                     │
                     ▼
2023  *Jiang et al. — DiffLL (this paper, wavelet-domain diffusion for LLIE)*
                     │
                     ▼
2024+  Wavelet/multi-scale diffusion mainstream for restoration (papers #36 ff.)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #02 Donoho 1995 (Wavelet shrinkage) | DWT-based denoising ancestor; replaces hard-threshold with learned diffusion | Foundational — DiffLL inherits the "operate in wavelet domain" strategy |
| #07 Dabov 2007 (BM3D) | Classical patch-based denoising baseline often used as post-processing in LIME (mentioned by DiffLL) | Sets the baseline DiffLL is trying to surpass at much higher quality |
| #25 Kadkhodaie 2021 (stochastic image priors) | Score-based prior for inverse problems | Conceptual sibling: DiffLL's WCDM is a conditional version of this idea on wavelet coefficients |
| #26 Kawar 2022 (DDRM) | Diffusion-based linear inverse solver | DDRM is a direct competitor; DiffLL beats it on LLIE by avoiding pixel-domain diffusion |
| #28 Chung 2023 (DPS / data-consistency diffusion) | Conditional diffusion for inverse problems | DiffLL's content-consistency term in Eq. 9 is a kindred mechanism |
| #41 Guo 2020 (Zero-DCE) | Curve-based unsupervised LLIE | Quantitative baseline in Table 1 — DiffLL improves PSNR by ~12 dB |
| #38 Morgan 2014 (MGN) | Different domain (solar coronagraph) but same philosophy: per-scale normalisation | Cousin work: both decompose image into scales then re-balance |

---

## 7. References / 참고문헌

- Hai Jiang, Ao Luo, Songchen Han, Haoqiang Fan, Shuaicheng Liu, "Low-Light Image Enhancement with Wavelet-Based Diffusion Models," *ACM Transactions on Graphics* 42(6), Article 238, 2023. DOI: 10.1145/3618373
- Ho, J., Jain, A., Abbeel, P., "Denoising Diffusion Probabilistic Models," *NeurIPS* 2020.
- Song, J., Meng, C., Ermon, S., "Denoising Diffusion Implicit Models," *ICLR* 2021.
- Rombach, R., et al., "High-Resolution Image Synthesis with Latent Diffusion Models," *CVPR* 2022.
- Saharia, C., et al., "Palette: Image-to-Image Diffusion Models," *SIGGRAPH* 2022.
- Kawar, B., et al., "Denoising Diffusion Restoration Models," *NeurIPS* 2022.
- Özdenizci, O., Legenstein, R., "Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models," *IEEE TPAMI* 2023.
- Wei, C., et al., "Deep Retinex Decomposition for Low-Light Enhancement (LOL dataset)," *BMVC* 2018.
- Li, J., et al., "Embedding Fourier for Ultra-High-Definition Low-Light Image Enhancement (UHDFour, UHD-LL dataset)," *ICLR* 2023.
- Mallat, S., "A Theory for Multiresolution Signal Decomposition," *IEEE TPAMI* 1989.
- Code: https://github.com/JianghaiSCU/Diffusion-Low-Light
