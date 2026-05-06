---
title: "Denoising Diffusion Restoration Models"
authors: Bahjat Kawar, Michael Elad, Stefano Ermon, Jiaming Song
year: 2022
journal: "Advances in Neural Information Processing Systems (NeurIPS) 35"
doi: "arxiv:2201.11793"
topic: Low-SNR Imaging / Diffusion-Based Inverse Problem Solver
tags: [diffusion-model, ddpm, ddim, ddrm, linear-inverse-problem, svd, spectral-decomposition, posterior-sampling, super-resolution, deblurring, inpainting, colorization, kawar-elad-ermon-song]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 26. Denoising Diffusion Restoration Models (DDRM) / 디노이징 확산 복원 모델

---

## 1. Core Contribution / 핵심 기여

### 한국어
DDRM은 **사전 학습된 unconditional/class-conditional DDPM 디노이저를 단 한 번도 다시 학습하지 않고** 임의의 *선형* 역문제 $y = Hx + z$ (단, $z\sim\mathcal N(0,\sigma_y^2 I)$)를 효율적으로 푸는 *unsupervised posterior sampler* 다. 이 논문은 두 가지 핵심 기술적 결정으로 이를 달성한다.

**(i) Spectral-space (SVD) 분해**: degradation 행렬 $H = U\Sigma V^\top$의 SVD를 이용해 영상을 spectral 좌표 $\bar x = V^\top x$로 변환한다. 이 좌표계에서 각 singular component는 *독립적인 1차원 inverse problem*이 된다 — singular value $s_i$가 0이면 측정값 없음 (pure generative), $s_i > 0$이고 측정 노이즈가 작으면 측정값을 신뢰, 노이즈가 더 크면 diffusion 노이즈에 흡수. 저자는 세 case 각각의 *Gaussian transition*을 명시적으로 정의 (Eqs. 4–5).

**(ii) Variational inference로 학습 목표 도출**: DDRM-specific Markov chain $p_\theta(x_{0:T}\mid y)$의 ELBO를 도출하고 (Theorem 3.2: $\eta=1$, $\eta_b = 2\sigma_y^2/(\sigma_y^2 + \sigma_T^2 s_i^2)$일 때), 이 ELBO가 **DDPM/DDIM의 unconditional 학습 목적과 정확히 동등**함을 증명. 따라서 *기존 DDPM 모델을 그대로* DDRM의 sampler로 사용 가능 — 재학습 불필요.

수치적으로 DDRM은 ImageNet 1K 256×256에서 **단 20 NFE (network function evaluations)** 만으로 4× super-resolution PSNR 26.55 dB / KID 7.22 (DDRM-CC: 6.56) 를 달성, 1500 NFE의 DGP, 100 NFE의 RED, 1000 NFE의 SNIPS를 *모두 능가*하면서 **5×–50× 빠르다** (Table 1, 2). 노이즈 있는 측정 ($\sigma_y = 0.05$)에서는 차이가 더 커진다 (DDRM 25.21 dB vs SNIPS 16.30 dB on 4× SR). 이 논문은 **사전 학습된 거대 diffusion model이 진정한 *universal prior*로 사용 가능함**을 정량적·정성적으로 입증한 결정적 작업이다.

### English
DDRM is the first **efficient, unsupervised, sampling-based linear inverse-problem solver** that uses a *pre-trained unconditional/class-conditional* DDPM as the prior, without any retraining or fine-tuning. Two ingredients make it work:

(1) **Spectral-space decomposition via the SVD** of the degradation matrix $H = U\Sigma V^\top$. In spectral coordinates $\bar x_t = V^\top x_t$, the joint diffusion chain decomposes into *one independent 1-D inverse problem per singular component*. The authors define three Gaussian transitions (Eqs. 4–5) covering: $s_i = 0$ (no measurement, behave like unconditional generation), $\sigma_t < \sigma_y/s_i$ (measurement is noisier than diffusion — replace measurement-info), $\sigma_t \ge \sigma_y/s_i$ (diffusion noise is larger — measurement informs the mean).

(2) **Variational-inference equivalence**: Theorem 3.2 shows that with $\eta = 1$ and a specific $\eta_b$, the DDRM ELBO collapses to the standard DDPM/DDIM denoising-autoencoder objective. Therefore *any* pre-trained DDPM serves as a near-optimal DDRM denoiser — no retraining needed.

Quantitatively, DDRM with 20 NFEs reaches 26.55 dB / KID 7.22 on noiseless ImageNet 1K 4× SR, surpassing DGP (1500 NFE), RED (100 NFE), and SNIPS (1000 NFE) — while being 5×–50× faster (Table 1). With $\sigma_y = 0.05$ noise, DDRM holds 25.21 dB vs SNIPS's 16.30 dB (Table 2). The paper definitively establishes that a single pre-trained diffusion model can be used as a **universal prior** for super-resolution, deblurring, inpainting, and colorization.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction & §2 Background / 서론과 배경

#### 한국어
- 영상 복원 문제: $y = Hx + z$, $z\sim\mathcal N(0,\sigma_y^2 I)$. SR, deblurring, inpainting, colorization 모두 이 형태의 *선형* 문제.
- 두 가지 풀이 흐름:
  1. **Supervised**: $(x, y)$ 쌍으로 end-to-end CNN 학습. 빠르고 정확하지만 *문제 specific* — 새로운 $H$나 $\sigma_y$마다 재학습.
  2. **Unsupervised**: prior $p(x)$를 학습해 두고 추론 시 $H$만 알면 복원 가능. 일반적이지만 느림 — DGP 1500 NFE, SNIPS 1000+ NFE.
- *Iterative methods*에서 NFE는 핵심 비용 — extreme case로 reference [30]은 15,000 NFE 사용.
- DDRM의 약속: **unsupervised이면서 20 NFE 안에 수렴**.
- DDPM 정의 (Eq. 2): forward $q(x_{1:T}\mid x_0)$, reverse $p_\theta(x_{0:T})$, ELBO 학습:
$$
\mathcal L_{\mathrm{DDPM}} = \sum_{t=1}^T \gamma_t\,\mathbb E_{x_0, x_t}\bigl[\|x_0 - f_\theta^{(t)}(x_t)\|_2^2\bigr]. \tag{Eq. 2}
$$
- $f_\theta^{(t)}$: $x_t$로부터 $x_0$ 추정값을 출력 (또는 noise prediction $\epsilon_\theta$로 등가 표현).

#### English
- Linear inverse problems: $y = Hx + z$. The two existing paradigms are supervised (problem-specific, fast) vs unsupervised (general, slow). DDRM seeks the speed of supervised with the generality of unsupervised.
- DDPMs (Eq. 2) provide a powerful unconditional prior, learned by minimizing a sum of denoising losses across noise levels. The key idea of DDRM is to "borrow" this prior without modifying $\theta$.

---

### Part II: §3.1 Variational Objective for DDRM / 변분적 목적함수

#### 한국어
- DDRM은 측정 $y$에 conditioned된 Markov chain $x_T \to x_{T-1} \to \cdots \to x_0$로 정의된다:
$$
p_\theta(x_{0:T}\mid y) = p_\theta^{(T)}(x_T \mid y) \prod_{t=0}^{T-1} p_\theta^{(t)}(x_t\mid x_{t+1}, y).
$$
- $x_0$가 최종 복원 결과; $x_T$는 거의 pure noise (high temperature).
- Factorized variational distribution $q(x_{1:T}\mid x_0, y) = q^{(T)}(x_T\mid x_0, y) \prod_{t=0}^{T-1} q^{(t)}(x_t\mid x_{t+1}, x_0, y)$ — 이는 inference 시 **noise schedule** $0 = \sigma_0 < \sigma_1 < \cdots < \sigma_T$에 따른 *variance-exploding* 형태. (DDPM의 variance-preserving 형태와 fixed linear transformation으로 동치, Appendix B.)
- $q(x_t\mid x_0) = \mathcal N(x_0, \sigma_t^2 I)$로 두면 spectral 좌표에서 각 component가 독립.

#### English
- DDRM defines a Markov chain conditioned on $y$ (Eq. before §3.2). The factorised variational distribution makes spectral-space decomposition tractable.
- The schedule $\sigma_t$ is variance-exploding; equivalence to DDPM's variance-preserving form is shown in Appendix B.

---

### Part III: §3.2 A Diffusion Process for Image Restoration / 영상 복원을 위한 확산 과정

#### 한국어
- **SVD 분해**: $H = U\Sigma V^\top$, $U \in \mathbb R^{m\times m}$, $V \in \mathbb R^{n\times n}$ orthogonal, $\Sigma \in \mathbb R^{m\times n}$ rectangular diagonal.
- $s_1 \ge s_2 \ge \cdots \ge s_m \ge 0$ singular values; $s_i = 0$ for $i \in [m+1, n]$ (즉 $V$의 마지막 $n-m$ columns 은 $H$의 null space).
- Spectral 좌표:
  - $\bar x_t^{(i)} = (V^\top x_t)_i$ — diffusion 변수의 spectral component.
  - $\bar y^{(i)} = (\Sigma^\dagger U^\top y)_i$ — 측정값의 spectral component (i번째 singular value의 norm으로 normalize). $\Sigma^\dagger$: Moore-Penrose pseudo-inverse.
  - **노이즈 표현**: 측정 노이즈 $z\sim\mathcal N(0,\sigma_y^2 I)$의 spectral 표현은 *variance $\sigma_y^2/s_i^2$* 의 1-D Gaussian (i.e., low-singular components are noisier in normalised coordinates).
- 핵심 통찰: *$\bar y^{(i)}$는 Gaussian noise level $\sigma_y/s_i$가 추가된 $\bar x_0^{(i)}$의 측정*. Singular value가 작을수록 noise level이 큼 → noisier components 처리는 더 신중해야 함.

- **Variational distribution** $q^{(t)}(\bar x_t^{(i)}\mid x_0, y)$ (Eq. 4 for $t=T$, Eq. 5 for $t<T$):

  - **Case $s_i = 0$**: 측정 정보 없음 → 표준 DDPM 형식
$$
q^{(t)}(\bar x_t^{(i)}\mid x_{t+1}, x_0) = \mathcal N\bigl(\bar x_0^{(i)} + \sqrt{1-\eta^2}\,\sigma_t \frac{\bar x_{t+1}^{(i)} - \bar x_0^{(i)}}{\sigma_{t+1}},\,\eta^2\sigma_t^2\bigr).
$$
  - **Case $\sigma_t < \sigma_y/s_i$ (diffusion noise smaller than measurement noise)**: diffusion이 더 신뢰할 만함 → measurement에 의존 적게
$$
q^{(t)} = \mathcal N\bigl(\bar x_0^{(i)} + \sqrt{1-\eta^2}\sigma_t \frac{\bar y^{(i)} - \bar x_0^{(i)}}{\sigma_y/s_i},\,\eta^2\sigma_t^2\bigr).
$$
  - **Case $\sigma_t \ge \sigma_y/s_i$ (diffusion noise larger)**: measurement가 더 신뢰할 만함 → measurement를 mean에 직접 융합
$$
q^{(t)} = \mathcal N\bigl((1-\eta_b)\bar x_0^{(i)} + \eta_b \bar y^{(i)},\,\sigma_t^2 - \frac{\sigma_y^2}{s_i^2}\eta_b^2\bigr).
$$
  - $\eta\in(0,1]$: stochastic vs deterministic 제어 hyperparameter (DDIM-style); $\eta_b$: measurement-injection coefficient.
  - 가정 $\sigma_T \ge \sigma_y/s_i$ for all positive $s_i$ → 매우 큰 $\sigma_T$로 충분.

- **Proposition 3.1**: 위 정의로부터 marginal $q(x_t\mid x_0) = \mathcal N(x_0, \sigma_t^2 I)$ 성립 (Appendix C 증명). 이는 unconditional DDPM과 *동일한* marginal 구조 — 그래서 동일 $f_\theta$를 사용 가능.

#### English
- **SVD trick**: in spectral coordinates, the joint $n$-dimensional inverse problem decouples into $n$ independent 1-D inverse problems — one per singular component.
- **Three transition cases** (Eq. 5):
  1. $s_i = 0$: missing information → unconditional generation step (the spectral component drifts according to the diffusion chain alone).
  2. $\sigma_t < \sigma_y/s_i$: diffusion chain is "less noisy" than the spectral measurement at this level — interpolate towards $\bar y^{(i)}$ as if it were a noisy version of $\bar x_0$.
  3. $\sigma_t \ge \sigma_y/s_i$: diffusion is noisier — directly inject the measurement into the mean (with $\eta_b$).
- The **Gaussian-marginal property** (Proposition 3.1) is what allows the conditional chain to use an unconditional DDPM: the per-step transitions integrate to the standard $\mathcal N(x_0, \sigma_t^2 I)$ marginal.

---

### Part IV: §3.3 "Learning" Image Restoration Models / 영상 복원 모델 "학습"

#### 한국어
- DDRM의 model distribution $p_\theta^{(t)}(\bar x_t^{(i)}\mid x_{t+1}, y)$는 $q^{(t)}$와 동일 형식이지만 모르는 $\bar x_0^{(i)}$를 *예측값* $\bar x_{\theta, t}^{(i)} = (V^\top f_\theta(x_{t+1}, t+1))_i$ 로 대체 (Eq. 8).
- 학습 가능한 parameter $\theta$를 ELBO 최대화로 학습할 수도 있지만, 이는 매 (H, σ_y) 조합마다 새 모델 학습 필요 → 비실용적.
- **Theorem 3.2**: 가정 $f_\theta^{(t)}$가 $t \ne t'$에서 weight share 안 함 + $\eta = 1$ + $\eta_b = 2\sigma_t^2/(\sigma_t^2 + \sigma_y^2/s_i^2)$ → DDRM ELBO가 *DDPM/DDIM의 unconditional ELBO (Eq. 2)와 동등*.
- 의미: **사전 학습된 unconditional DDPM이 그 자체로 DDRM의 최적 denoiser**. 별도 학습 불필요.
- 다른 $\eta, \eta_b$ 선택이라도 ELBO는 spectral space에서 *weighted sum-of-squares error*이므로 DDPM이 좋은 근사. 따라서 어떤 $H$에 대해서도 사전 학습된 DDPM 모델로 DDRM 가능.

#### English
- **Theorem 3.2** is the central technical result: for $\eta = 1$ and a specific $\eta_b = 2\sigma_t^2/(\sigma_t^2 + \sigma_y^2/s_i^2)$, the DDRM ELBO reduces *exactly* to the DDPM/DDIM denoising-autoencoder objective. So an unconditional DDPM is an optimal DDRM denoiser; no retraining required.
- For general $\eta, \eta_b$, the ELBO is still a weighted spectral-space sum-of-squares loss, so DDPM remains an excellent approximation.

---

### Part V: §3.4 Accelerated Algorithms / 가속 알고리즘과 §3.5 Memory-Efficient SVD

#### 한국어
- DDPM은 보통 $T = 1000$ steps로 학습하지만, DDIM (Song et al. 2020)처럼 *subset* $T' \ll T$ 만 사용해도 OK → DDRM도 동일.
- Default DDRM: $T' = 20$ steps (uniformly spaced from 1000) → **20 NFE** 만으로 좋은 결과.
- **Memory-efficient SVD (Appendix D)**: $V$를 $n\times n$ matrix로 저장하면 $\Theta(n^2)$ memory — 256×256 영상은 $n = 196{,}608$, $V$는 ~38 GB. 다행히 사용된 모든 $H$ (denoising, inpainting, SR, deblurring, colorization)에서 $V$가 *structured* (e.g., Fourier basis, identity sub-blocks, 3×3 averaging) → $\Theta(n)$ 메모리로 표현 가능.

#### English
- DDRM uses the same time-step subsetting trick as DDIM: 20 of 1000 steps suffice. Per-iteration cost = 1 forward pass through DDPM.
- For all considered $H$ matrices, $V$ has structure that admits $\Theta(n)$ rather than $\Theta(n^2)$ storage — practical for 256×256 and beyond.

---

### Part VI: §5 Experiments / 실험

#### 한국어
- **Datasets**: ImageNet 256×256 (1000 validation images, one per class), CelebA-HQ, LSUN bedrooms/cats, FFHQ.
- **Diffusion models**: pretrained from Ho+ 2020 (LSUN/CelebA) and Dhariwal+Nichol 2021 (ImageNet 256/512).
- **Hyperparameters**: $\eta = 0.85$, $\eta_b = 1$, 20 uniformly spaced timesteps.
- **Baselines**: DGP [38], RED [40], SNIPS [25] for unsupervised; "Baseline" = bicubic up-sampling for SR or blurry input for deblur.
- **Tasks**: 4×/8× block-averaging SR, 9×9 uniform-kernel deblurring (with truncated singular values), grayscale-from-RGB colorization, inpainting (text overlay or 50 % random pixels). Optionally additive Gaussian noise $\sigma_y$.
- **Quantitative (Tables 1, 2)**:

| Method | 4× SR PSNR | 4× SR KID↓ | NFE↓ | Deblur PSNR | Deblur KID↓ | NFE↓ |
|---|---|---|---|---|---|---|
| Baseline | 25.65 | 44.90 | 0 | 19.26 | 38.00 | 0 |
| DGP | 23.06 | 21.22 | 1500 | 22.70 | 27.60 | 1500 |
| RED | 26.08 | 53.55 | 100 | 26.16 | 21.21 | 500 |
| SNIPS | 17.58 | 35.17 | 1000 | 34.32 | 0.87 | 1000 |
| **DDRM** | **26.55** | **7.22** | **20** | **35.64** | **0.71** | **20** |
| **DDRM-CC** | **26.55** | **6.56** | **20** | **35.65** | **0.70** | **20** |

  - DDRM-CC: class-conditional DDPM (same speed). 클래스 라벨 활용해 KID 추가 향상.
  - 노이즈 있는 측정 ($\sigma_y = 0.05$, Table 2): DDRM 25.21 dB / KID 12.43, baselines 모두 KID 40+ 또는 PSNR 폭락. DDRM이 노이즈 있는 측정에서 *더 큰* 우위.
  - **5× 빠름 (DDRM 20 NFE vs RED 100 NFE) ~ 50× 빠름 (vs SNIPS 1000 NFE)**.

- **Qualitative (Figs. 1, 3, 4, 5, 6)**: DDRM은 inpainting, deblurring, colorization에서 *날카롭고 자연스러운* 결과를 생성. 특히 $\sigma_y = 0.05$ 노이즈 있을 때 다른 방법은 noisy artifact를 그대로 가져오지만 DDRM은 깨끗.
- **Out-of-distribution generalization (Fig. 6)**: ImageNet에서 학습된 DDPM이 USC-SIPI 영상 (ImageNet 클래스 아님)에서도 잘 동작 → *진짜* universal prior.

#### English
- DDRM dominates baselines on noiseless ImageNet 1K 4× SR (PSNR 26.55 vs DGP 23.06 / RED 26.08 / SNIPS 17.58) and on deblurring (PSNR 35.64 vs SNIPS 34.32) while using 20 NFE vs 100–1500.
- The advantage *grows* under noisy measurements ($\sigma_y = 0.05$, Table 2): DDRM 25.21 dB SR / 25.45 dB deblur vs all baselines either failing or producing noisy artifacts.
- DDRM works on out-of-distribution images (Fig. 6: USC-SIPI), demonstrating that the learned diffusion prior is *genuinely* universal.

---

## 3. Key Takeaways / 핵심 시사점

1. **Pre-trained diffusion = universal linear-inverse prior — 사전 학습 확산 = 보편적 선형 역문제 prior.**
   - **English**: A single unconditional DDPM serves as the prior for super-resolution, deblurring, inpainting, colorization, and arbitrary Gaussian-noise levels — without retraining. This is the central conceptual achievement.
   - **한국어**: 한 번 학습한 unconditional DDPM이 SR, deblur, inpainting, colorization 모든 문제의 prior로 사용됨 — 재학습 불필요. 가장 핵심적인 개념적 성취.

2. **SVD decomposes joint problem into independent 1-D problems — SVD가 결합 문제를 독립 1차원 문제로 분해.**
   - **English**: In spectral coordinates, the high-dimensional inverse problem becomes $n$ independent 1-D inverse problems — one per singular component. This is what makes the diffusion sampler tractable.
   - **한국어**: spectral 좌표에서 고차원 역문제가 $n$개 독립 1차원 역문제로 분해됨 — 이것이 diffusion sampler를 tractable하게 만드는 비결.

3. **Three transition regimes for spectral components — spectral component의 세 가지 transition 영역.**
   - **English**: Per-component update rule branches on $s_i = 0$ (no measurement, generative step), $\sigma_t < \sigma_y/s_i$ (diffusion is less noisy), and $\sigma_t \ge \sigma_y/s_i$ (diffusion is more noisy). The branching adapts measurement reliability to diffusion noise level.
   - **한국어**: 각 component별 update 규칙이 세 영역으로 분기 — 측정값과 diffusion 노이즈 사이의 신뢰도 trade-off를 해결.

4. **ELBO collapses to DDPM objective — ELBO가 DDPM 목적함수로 환원.**
   - **English**: Theorem 3.2 proves that with specific $\eta, \eta_b$ choices, the DDRM ELBO equals the unconditional DDPM ELBO — so any DDPM is an optimal DDRM denoiser.
   - **한국어**: Theorem 3.2: 특정 $\eta, \eta_b$에서 DDRM ELBO가 unconditional DDPM ELBO와 정확히 같음 — 즉 DDPM이 그대로 최적 denoiser.

5. **20 NFEs suffice (5–50× faster than baselines) — 20 NFE로 충분 (5–50배 빠름).**
   - **English**: DDPM training schedule of 1000 steps is sub-sampled to 20 in DDRM — same DDIM acceleration trick. Per-iteration is one DDPM forward pass + cheap spectral ops.
   - **한국어**: DDPM의 1000 step을 20으로 sub-sample. iteration당 DDPM forward 1회 + 저렴한 spectral 연산.

6. **Robust to measurement noise — 측정 노이즈에 강건.**
   - **English**: With $\sigma_y = 0.05$, DDRM still reaches 25.21 dB on 4× SR while DGP/RED/SNIPS collapse. The measurement noise is *integrated into* the diffusion noise schedule rather than fought separately.
   - **한국어**: $\sigma_y = 0.05$ 노이즈 있는 측정에서도 25.21 dB. 측정 노이즈가 diffusion 노이즈 schedule에 *흡수*되어 자연스럽게 처리됨.

7. **Memory-efficient SVD for structured H — 구조화된 H에 대한 메모리 효율적 SVD.**
   - **English**: $V$ matrix would be $\Theta(n^2)$ for naive storage but is $\Theta(n)$ for all considered $H$ (averaging, Fourier, identity-subset). This makes 256×256 practical.
   - **한국어**: $V$ 행렬은 naive하게는 $\Theta(n^2)$이지만 고려된 모든 $H$ (averaging, Fourier, identity 부분집합)에서 $\Theta(n)$. 256×256 영상에서 실용적.

8. **Out-of-distribution generality — 분포 외 일반화.**
   - **English**: ImageNet-trained DDPM works on USC-SIPI test images outside ImageNet classes (Fig. 6). The diffusion prior captures *general natural image statistics*, not just the training distribution.
   - **한국어**: ImageNet 학습 DDPM이 USC-SIPI 영상 (ImageNet 클래스 아님)에도 잘 동작 (Fig. 6). 학습 분포에 한정되지 않는 *일반적 자연 영상 통계*를 학습함.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Linear inverse problem / 선형 역문제
$$
y = Hx + z,\qquad z \sim \mathcal N(0, \sigma_y^2 I),\qquad H \in \mathbb R^{m\times n}, m \le n. \tag{Eq. 1}
$$
Goal: recover $x \in \mathbb R^n$ from $y \in \mathbb R^m$.

### 4.2 SVD of degradation operator / 분해 행렬 SVD
$$
H = U\Sigma V^\top,\quad U\in\mathbb R^{m\times m},\quad V\in\mathbb R^{n\times n}\text{ orthogonal},\quad \Sigma \text{ rectangular diagonal}. \tag{Eq. 3}
$$
Singular values $s_1 \ge s_2 \ge \cdots \ge s_m \ge 0$, $s_i = 0$ for $i \in [m+1, n]$.

### 4.3 Spectral coordinates / Spectral 좌표
- $\bar x_t = V^\top x_t$, $\bar y = \Sigma^\dagger U^\top y$ (Moore–Penrose pseudo-inverse).
- For $i$ with $s_i > 0$: $\bar y^{(i)} = \bar x_0^{(i)} + n_i$, $n_i \sim \mathcal N(0, \sigma_y^2/s_i^2)$.
- For $i$ with $s_i = 0$: $\bar y^{(i)}$ undefined (no information).

### 4.4 Variational distribution (Eqs. 4–5)
**$t = T$**:
$$
q^{(T)}(\bar x_T^{(i)}\mid x_0, y) = \begin{cases}
\mathcal N(\bar y^{(i)}, \sigma_T^2 - \sigma_y^2/s_i^2) & \text{if } s_i > 0\\
\mathcal N(\bar x_0^{(i)}, \sigma_T^2) & \text{if } s_i = 0.
\end{cases}
$$
**$t < T$** (the three cases of Eq. 5):
$$
q^{(t)}(\bar x_t^{(i)}\mid x_{t+1}, x_0, y) =
\begin{cases}
\mathcal N\!\bigl(\bar x_0^{(i)} + \sqrt{1-\eta^2}\sigma_t \frac{\bar x_{t+1}^{(i)} - \bar x_0^{(i)}}{\sigma_{t+1}},\,\eta^2 \sigma_t^2\bigr) & \text{if } s_i = 0\\[4pt]
\mathcal N\!\bigl(\bar x_0^{(i)} + \sqrt{1-\eta^2}\sigma_t \frac{\bar y^{(i)} - \bar x_0^{(i)}}{\sigma_y/s_i},\,\eta^2 \sigma_t^2\bigr) & \text{if } \sigma_t < \sigma_y/s_i\\[4pt]
\mathcal N\!\bigl((1-\eta_b)\bar x_0^{(i)} + \eta_b \bar y^{(i)},\,\sigma_t^2 - (\sigma_y^2/s_i^2)\eta_b^2\bigr) & \text{if } \sigma_t \ge \sigma_y/s_i.
\end{cases}
$$

### 4.5 Model distribution (Eqs. 7–8)
Replace unknown $\bar x_0^{(i)}$ with predicted $\bar x_{\theta,t}^{(i)} = (V^\top f_\theta(x_{t+1}, t+1))_i$:
$$
p_\theta^{(t)}(\bar x_t^{(i)}\mid x_{t+1}, y) = q^{(t)}\,(\text{but with }\bar x_0^{(i)} \to \bar x_{\theta, t}^{(i)}).
$$

### 4.6 Theorem 3.2 (variational equivalence to DDPM)
Assume $f_\theta^{(t)}$, $f_\theta^{(t')}$ have no shared weights for $t \ne t'$. Set $\eta = 1$ and $\eta_b = 2\sigma_t^2/(\sigma_t^2 + \sigma_y^2/s_i^2)$. Then the DDRM ELBO equals the standard DDPM/DDIM objective Eq. 2:
$$
\mathcal L_{\mathrm{DDRM}} = \mathcal L_{\mathrm{DDPM}} = \sum_{t=1}^T \gamma_t\,\mathbb E_{x_0, x_t}\bigl[\|x_0 - f_\theta^{(t)}(x_t)\|_2^2\bigr].
$$
Consequence: a pre-trained DDPM is a near-optimal DDRM denoiser.

### 4.7 Worked numerical example: 1-D 4× super-resolution / 수치 예시
**Setup**: 1-D signal $x \in \mathbb R^4$, 4× SR with $H = [1/4, 1/4, 1/4, 1/4] \in \mathbb R^{1\times 4}$ (block-averaging). $\sigma_y = 0$ (noiseless).

- SVD: $\Sigma = [1/2, 0, 0, 0]$ (single non-zero singular value $s_1 = 1/2$). $V$ contains $[1/2, 1/2, 1/2, 1/2]$ as first column (mean direction); other 3 columns span the orthogonal complement.
- True signal $x = [0.2, 0.4, 0.6, 0.8]$, mean $= 0.5$.
- Measurement $y = H x = 0.5$.
- Spectral coordinates: $\bar x = V^\top x = [1.0, ?, ?, ?]$ where the first component encodes $\sqrt{4} \cdot \text{mean} = 1.0$. The other 3 components are the "missing information."
- $\bar y^{(1)} = y / s_1 = 0.5 / 0.5 = 1.0$. $\bar y^{(i)}$ undefined for $i > 1$.

**Sampling** (T=2 step toy, $\eta = 1$, $\eta_b = 1$):
- $i = 1$ ($s_1 > 0$): $\sigma_T \ge \sigma_y/s_1 = 0$ trivially → use Case 3. Eq. 5: $\bar x_T^{(1)} = (1-1)\bar x_0^{(1)} + 1\cdot \bar y^{(1)} = 1.0$. Component perfectly determined.
- $i = 2, 3, 4$ ($s_i = 0$): Case 1 — pure DDPM step. $\bar x_T^{(i)} \sim \mathcal N(0, \sigma_T^2)$ and refined by DDPM denoiser.
- After 20 DDPM steps (sub-sampled): $\bar x_0^{(i)}$ for $i > 1$ is sampled consistently with the natural-image prior, conditional on $\bar x_0^{(1)} = 1.0$ (mean fixed).
- Final reconstruction $x = V \bar x_0$ has correct mean (0.5) and plausible high-frequency structure (drawn from prior).

This makes the SVD-decomposition concrete: the measurement constrains a *single* spectral coefficient (the mean), and the diffusion prior fills in the other 3 coefficients.

### 4.8 Algorithmic pseudocode / 알고리즘 의사코드
```
Inputs: pretrained DDPM f_theta, degradation H (with SVD U,Sigma,V),
        measurement y, noise level sigma_y, eta, eta_b, schedule sigma_T > ... > sigma_0.
Compute bar_y = pinv(Sigma) U^T y.
Initialise bar_x_T from q^{(T)}(. | y) (Eq. 4).
For t = T, T-1, ..., 1:
    x_{t+1} = V bar_x_{t+1}.
    pred_x0 = f_theta(x_{t+1}, t+1).
    bar_pred_x0 = V^T pred_x0.
    For each spectral index i:
        Branch on s_i and (sigma_t vs sigma_y/s_i) and sample bar_x_t^{(i)} from Eq. 5.
Return x_0 = V bar_x_0.
```

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
2013      Venkatakrishnan+ — Plug-and-Play Priors (denoiser as proximal op)
2017      Romano-Elad-Milanfar — Regularization by Denoising (RED)
2018      Ulyanov+ — Deep Image Prior (DIP)
2019      Song-Ermon — Score-based generative models (NCSN)
2020      Ho-Jain-Abbeel — DDPM
2020      Song+ — Score-based SDEs (unifying score-based and diffusion)
2020      Song+ — DDIM (deterministic accelerated sampling)
2021      Pan+ — Deep Generative Prior (DGP, GAN-based unsupervised inverse problem)
2021      Kadkhodaie-Simoncelli (paper #25) — denoiser-prior for linear inverse
2021      Choi+ — ILVR (conditional DDPM for SR, special case of DDRM)
2021      Song-Ermon — SNIPS (score-based inverse problem solver, 1000 NFE)
2022 ★★   KAWAR-ELAD-ERMON-SONG — DDRM (THIS PAPER)
                    ↳ SVD-based spectral decomposition + variational ELBO
                    ↳ 20 NFEs, unsupervised, problem-agnostic
                    ↳ ImageNet-trained DDPM works on out-of-distribution
2022      Chung+ — DPS — non-linear inverse problems via approximate posterior
2022      Wang+ — DDNM — null-space decomposition variant
2023      Bansal+ — Cold Diffusion (paper #27) — non-stochastic diffusion
2023      Song+ — Loss-Guided Diffusion — generalised conditional sampling
2023      Saharia+ — Imagen text-to-image (uses DDRM-style conditioning)
```

DDRM은 *score-based / diffusion 기반 unsupervised linear-inverse 풀이의 결정적 표준*. 직전 paper #25 (Kadkhodaie-Simoncelli)는 핵심 아이디어 (denoiser = prior)를 가장 단순한 형태로 제시했고, DDRM은 이를 *대규모 사전학습 DDPM* 및 *임의 H*와 결합 — 실용적 universal solver로 성숙시켰다. 직후 DPS, DDNM은 DDRM을 (i) non-linear (ii) null-space 변형으로 확장.

DDRM defines the standard for **score-based unsupervised linear-inverse solvers**. Paper #25 (Kadkhodaie-Simoncelli) introduced the cleanest form of the denoiser-as-prior idea; DDRM scales it up to large pre-trained DDPMs and arbitrary $H$ via SVD decomposition, turning the idea into a practical universal solver. Subsequent works (DPS, DDNM, RED-Diff, $\Pi$GDM) generalize DDRM to non-linear $H$, null-space refinements, or more efficient noise schedules.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Ho+ (2020)** DDPM | Pre-trained backbone | DDRM uses pre-trained DDPMs unmodified — Theorem 3.2 proves this is near-optimal. Without DDPM, no DDRM. |
| **Song+ (2020)** DDIM | Sampling acceleration | DDIM's idea of sub-sampling time steps (1000 → 20) is what makes DDRM practical at 20 NFEs. The $\eta$ hyperparameter in DDRM directly inherits from DDIM's deterministic-vs-stochastic interpolation. |
| **Song+Ermon (2019)** NCSN | Predecessor sampler | NCSN's annealed Langevin dynamics is the conceptual ancestor of all diffusion samplers used in inverse problems, including DDRM. |
| **Kadkhodaie-Simoncelli (2021)** (paper #25) | Direct conceptual precursor | Paper #25 introduced the cleanest "denoiser-as-prior" formulation and an Algorithm 2 for projected sampling on linear inverse problems — DDRM is the DDPM-scale version with SVD-based decomposition and variational justification. |
| **Choi+ (2021)** ILVR | Special case | ILVR conditions DDPM on SR measurements; Appendix H of DDRM proves ILVR is a *special case* of DDRM under the same diffusion prior. |
| **Song-Ermon (2021)** SNIPS | Closest baseline | SNIPS also performs SVD-based diffusion sampling for inverse problems but requires 1000 NFEs and fails under measurement noise; DDRM's variational derivation gives both speed (20 NFEs) and noise robustness. |
| **Venkatakrishnan+ (2013)** PnP | Algorithmic ancestor | PnP heuristically inserts a denoiser into ADMM; DDRM gives a fully principled posterior-sampling counterpart. |
| **Romano-Elad-Milanfar (2017)** RED | Algorithmic baseline | RED constructs a regularizer from a denoiser; DDRM outperforms it on PSNR and KID (Tables 1, 2). |
| **Pan+ (2021)** Deep Generative Prior | GAN-based competitor | DGP optimizes GAN latent + weights for each test image (1500 NFEs). DDRM is 75× faster and produces sharper natural-image priors. |
| **Chung+ (2022)** DPS | Direct successor | Generalises DDRM to non-linear measurement operators via approximate posterior gradients. |
| **Wang+ (2022)** DDNM | Sister method | Uses null-space decomposition of $H$ for inverse problems with pre-trained diffusion — alternative to DDRM's SVD. |
| **Bansal+ (2023)** Cold Diffusion (paper #27) | Counterexample | Demonstrates that diffusion-style restoration works *without* Gaussian noise — challenging the noise-centric framework underlying DDRM. |
| **Lehtinen+ (2018)** Noise2Noise (paper #16) | Indirect relation | Could in principle be used to train the underlying DDPM without clean targets, giving a fully self-supervised DDRM. |

---

## 7. Failure Modes and Limits / 실패 양상과 한계

### 한국어
1. **Linear-only**: DDRM은 *선형* $H$에만 직접 적용. Phase retrieval, MRI motion compensation 등은 비선형 → DPS 등 후속 연구 필요.
2. **SVD storage**: $V$가 unstructured할 때 $\Theta(n^2)$ 메모리 — 256×256 컬러 영상에서 ~38 GB. 모든 *고려된* $H$는 structured하지만 임의 $H$는 그렇지 않을 수 있음.
3. **Class-conditional ground truth 의존**: DDRM-CC (best results)는 test 영상의 정답 ImageNet 클래스 라벨 사용 → 실세계 응용에서 알 수 없음. Authors가 "unfair advantage"라고 정직하게 표시.
4. **Noise model is Gaussian-only**: 측정 노이즈가 Poisson, impulse 등이면 SVD-spectral의 noise variance 변환이 더 이상 성립 안 함 → 추가 보정 필요.
5. **Pre-trained DDPM 도메인 의존**: ImageNet-trained DDPM은 자연 영상에 OK이지만 medical imaging, microscopy 등 specialized 도메인에서는 새 DDPM 학습 필요.
6. **NFE 절감의 한계**: 20 NFE도 supervised CNN의 1 forward pass보다 느림. Real-time 응용에서는 여전히 supervised 방법이 우월.

### English
1. **Linear-only**: DDRM directly applies only to linear $H$; non-linear problems (phase retrieval, motion compensation) require successors like DPS.
2. **SVD storage**: $\Theta(n^2)$ for unstructured $V$ — ~38 GB at 256×256 RGB. Works for the considered $H$'s because their $V$'s have structure (Fourier, identity sub-blocks, averaging).
3. **DDRM-CC uses ground-truth class labels**: best results in Tables 1–2 use ImageNet class labels per test image — known by the algorithm but typically unknown in real applications. Authors are explicit about this.
4. **Gaussian noise assumption**: the SVD-spectral noise-variance transformation $\sigma_y^2 \to \sigma_y^2/s_i^2$ requires Gaussian additive noise.
5. **Domain match**: ImageNet-trained DDPM transfers to *natural* OOD images but a microscopy or astronomy problem still requires a domain-trained DDPM.
6. **20 NFEs > 1 NFE**: still slower than supervised CNN per-image; real-time applications may prefer supervised.

---

## 8. References / 참고문헌

- Kawar, B., Elad, M., Ermon, S., & Song, J. "Denoising Diffusion Restoration Models" (DDRM), *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2022. [arXiv:2201.11793]
- Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models" (DDPM), *Proc. NeurIPS*, 2020.
- Song, J., Meng, C., & Ermon, S. "Denoising Diffusion Implicit Models" (DDIM), *Proc. ICLR*, 2021.
- Song, Y., & Ermon, S. "Generative modeling by estimating gradients of the data distribution" (NCSN), *Proc. NeurIPS*, 2019.
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. "Score-based generative modeling through stochastic differential equations", *Proc. ICLR*, 2021.
- Kadkhodaie, Z., & Simoncelli, E. P. "Stochastic Solutions for Linear Inverse Problems Using the Prior Implicit in a Denoiser", *Proc. NeurIPS*, 2021. [arXiv:2007.13640]
- Song, Y., & Ermon, S. "Solving Inverse Problems in Medical Imaging with Score-Based Generative Models" (SNIPS), *Proc. ICLR*, 2022.
- Choi, J., Kim, S., Jeong, Y., Gwon, Y., & Yoon, S. "ILVR: Conditioning method for denoising diffusion probabilistic models", *Proc. ICCV*, 2021.
- Pan, X., Zhan, X., Dai, B., Lin, D., Loy, C. C., & Luo, P. "Exploiting Deep Generative Prior for Versatile Image Restoration and Manipulation" (DGP), *Proc. ECCV*, 2020.
- Romano, Y., Elad, M., & Milanfar, P. "The little engine that could: Regularization by Denoising (RED)", *SIAM J. Imaging Sciences*, 10(4), 1804–1844 (2017).
- Venkatakrishnan, S. V., Bouman, C. A., & Wohlberg, B. "Plug-and-play priors for model based reconstruction" (PnP), *Proc. GlobalSIP*, 2013.
- Dhariwal, P., & Nichol, A. "Diffusion models beat GANs on image synthesis", *Proc. NeurIPS*, 2021.
- Chung, H., Kim, J., Mccann, M. T., Klasky, M. L., & Ye, J. C. "Diffusion Posterior Sampling for general noisy inverse problems" (DPS), *Proc. ICLR*, 2023.
- Wang, Y., Yu, J., & Zhang, J. "Zero-shot image restoration using denoising diffusion null-space model" (DDNM), *Proc. ICLR*, 2023.
- Bińkowski, M., Sutherland, D. J., Arbel, M., & Gretton, A. "Demystifying MMD GANs" (KID), *Proc. ICLR*, 2018.
