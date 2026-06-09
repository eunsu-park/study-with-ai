---
title: "Pre-Reading Briefing: Denoising Diffusion Restoration Models (DDRM)"
paper_id: "26_kawar_2022"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Denoising Diffusion Restoration Models (DDRM): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: B. Kawar, M. Elad, S. Ermon, J. Song, *NeurIPS* 35 (2022), pp. 23593-23606, arXiv:2201.11793
**Author(s)**: Bahjat Kawar, Michael Elad, Stefano Ermon, Jiaming Song
**Year**: 2022

---

## 1. 핵심 기여 / Core Contribution

### 한국어
DDRM은 **사전 학습된 unconditional/class-conditional DDPM 을 단 한 번도 다시 학습하지 않고** 임의의 *선형* 역문제 $y = Hx + z$ ($z\sim\mathcal N(0,\sigma_y^2 I)$)를 효율적으로 푸는 *unsupervised posterior sampler* 다. 두 가지 핵심 기술적 결정으로 이를 달성한다: (i) **Spectral-space (SVD) 분해** — degradation $H = U\Sigma V^\top$로 영상을 spectral 좌표로 변환해 각 singular component를 *독립적인 1차원 inverse problem*으로 처리; (ii) **Variational ELBO 동등성 (Theorem 3.2)** — 특정 $\eta, \eta_b$ 선택 시 DDRM ELBO가 unconditional DDPM 학습 목적과 *정확히 동등*하므로 별도 fine-tuning 불필요. 결과: ImageNet 4× SR에서 **20 NFE 만으로** PSNR 26.55 dB / KID 7.22, DGP/RED/SNIPS 대비 5×-50× 빠르며, 노이즈 있는 측정 ($\sigma_y = 0.05$)에서는 차이가 더 커진다.

### English
DDRM is the first **efficient, unsupervised, sampling-based linear inverse-problem solver** that uses a *pre-trained unconditional/class-conditional* DDPM as the prior, with no retraining. Two ingredients: (i) **SVD-based spectral decomposition** of the degradation $H = U\Sigma V^\top$ — in spectral coordinates the joint inverse problem decouples into one independent 1-D inverse problem per singular component; (ii) **Variational equivalence (Theorem 3.2)** — for $\eta = 1$ and a specific $\eta_b$, the DDRM ELBO collapses to the standard DDPM/DDIM denoising objective, so any pre-trained DDPM serves as a near-optimal DDRM denoiser. With **only 20 NFEs**, DDRM reaches 26.55 dB on ImageNet 4× SR — beating DGP (1500 NFE), RED (100 NFE), SNIPS (1000 NFE) — and the gap widens under measurement noise.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**: 2022년 초까지 inverse-problem 해법은 둘 중 하나였다. **Supervised** end-to-end CNN은 빠르고 정확하지만 *문제별 재학습* 필요. **Unsupervised** 방법(DGP, SNIPS, RED)은 일반적이지만 1000-1500 NFE로 매우 느렸다. 동시에 DDPM (Ho 2020)과 DDIM (Song 2021)이 unconditional generation에서 SOTA를 달성하며, "*이 강력한 사전학습 모델을 어떻게 inverse problem에 사용할 것인가*"가 핵심 질문이 되었다. Paper #25 (Kadkhodaie-Simoncelli 2021)가 가장 단순한 답을 제시했지만 BF-CNN 단일 디노이저로 한정되었다. DDRM은 이를 *대규모 사전학습 DDPM*으로 scale up하면서 *임의의 H*에 적용 가능한 universal solver로 성숙시킨다.

**English**: By early 2022 inverse-problem methods split into two camps: supervised end-to-end CNNs (fast but problem-specific) and unsupervised (general but 1000-1500 NFEs). Concurrently DDPM/DDIM achieved SOTA in unconditional generation, raising the question: *how can these powerful pre-trained priors be reused for inverse problems?* Paper #25 (Kadkhodaie-Simoncelli) gave the simplest answer for a single BF-CNN denoiser. DDRM scales the idea to large pre-trained DDPMs and arbitrary linear $H$ via SVD decomposition, becoming the practical universal solver.

### 타임라인 / Timeline

```
2013      Venkatakrishnan+ — PnP Priors (denoiser-as-prox)
2017      Romano+Elad+Milanfar — Regularization by Denoising
2019      Song-Ermon — NCSN (annealed Langevin score matching)
2020      Ho+ — DDPM
2021      Song+ — DDIM (deterministic accelerated sampling)
2021      Kadkhodaie-Simoncelli — denoiser-prior linear inverse (paper #25)
2021      Choi+ — ILVR; Song-Ermon — SNIPS (1000 NFE SVD diffusion)
2022 ★★   KAWAR+ — DDRM (THIS PAPER)
2022      Chung+ — DPS (paper #28); Wang+ — DDNM
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**:
- **DDPM forward/reverse process**: $q(x_t\mid x_0) = \mathcal N(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$, ELBO 학습.
- **DDIM**: deterministic acceleration, time-step subsetting, $\eta$ hyperparameter.
- **SVD**: $H = U\Sigma V^\top$, Moore-Penrose pseudo-inverse $\Sigma^\dagger$.
- **Variance-exploding vs variance-preserving** noise schedules.
- **Variational inference / ELBO** 도출 — Markov chain의 KL 분해.
- **Linear inverse problem 종류**: SR (block averaging), deblurring (convolution), inpainting (mask), colorization (gray-from-RGB).
- **KID/FID**: perceptual generation quality metrics.

**English**:
- **DDPM forward/reverse**: $q(x_t\mid x_0) = \mathcal N(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$, ELBO training.
- **DDIM**: deterministic sampling, time-step subsetting, the $\eta$ hyperparameter.
- **SVD basics**: $H = U\Sigma V^\top$, Moore-Penrose pseudo-inverse $\Sigma^\dagger$.
- **Variance-exploding vs variance-preserving** noise schedules.
- **Variational inference / ELBO** derivation — KL decomposition of Markov chains.
- **Common linear inverse problems**: SR (block averaging), deblurring (convolution), inpainting (mask), colorization (gray-from-RGB).
- **KID / FID** as perceptual generation metrics.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Linear inverse problem | $y = Hx + z$ where $H$ linear, $z \sim \mathcal N(0,\sigma_y^2 I)$. SR, deblur, inpaint 등. / Recover $x$ from a linear noisy measurement. |
| Singular Value Decomposition (SVD) | $H = U\Sigma V^\top$ — degradation을 spectral 좌표로 분해 / Decomposes the degradation operator into orthogonal bases plus a diagonal scale. |
| Spectral coordinates | $\bar x = V^\top x$, $\bar y = \Sigma^\dagger U^\top y$. Component-wise 1차원 문제로 환원 / Per-component 1-D inverse problems after the orthogonal change of basis. |
| Pseudo-inverse $\Sigma^\dagger$ | 0이 아닌 singular value의 역수 / Reciprocal on positive singular values, zero elsewhere. |
| Three transition cases | $s_i = 0$ / $\sigma_t < \sigma_y/s_i$ / $\sigma_t \ge \sigma_y/s_i$ — measurement reliability에 따른 분기 / Branch based on whether diffusion noise or measurement noise dominates. |
| $\eta, \eta_b$ | DDIM-style stochasticity ($\eta$) + measurement-injection coefficient ($\eta_b$). Theorem 3.2가 특정 값에서 ELBO 동치 보장 / Hyperparameters; specific values make the DDRM ELBO equal the DDPM ELBO. |
| NFE | Number of Function Evaluations — sampler step 수. DDRM = 20, DDPM = 1000 / Sampling cost in network forward passes; DDRM uses 20 vs DDPM's 1000. |
| Variance-exploding schedule | $q(x_t\mid x_0) = \mathcal N(x_0, \sigma_t^2 I)$ form with $\sigma_T \to \infty$ / Equivalent to DDPM via fixed linear transformation (Appendix B). |
| ILVR | Iterative Latent Variable Refinement — DDRM의 special case (Appendix H) / Conditional DDPM for SR; a special case of DDRM. |
| Class-conditional (CC) DDPM | 클래스 라벨 사용. DDRM-CC가 best result / Uses ImageNet labels at test time; gives DDRM-CC's best KID. |
| Memory-efficient SVD | $V$의 structure를 활용해 $\Theta(n^2)\to\Theta(n)$ 메모리 / Exploits structure (Fourier, identity sub-blocks) to avoid 38 GB storage. |

---

## 5. 수식 미리보기 / Equations Preview

**핵심 1: 선형 역문제 설정 / Linear inverse problem (Eq. 1)**

$$
y = Hx + z,\qquad z \sim \mathcal N(0, \sigma_y^2 I),\qquad H \in \mathbb R^{m\times n}
$$

**한국어**: SR, deblur, inpaint, colorization 모두 이 단일 형태. DDRM은 $H, \sigma_y$만 알면 동일 코드로 처리.

**English**: SR, deblurring, inpainting, colorization all have this form. DDRM handles them with one code path given $H$ and $\sigma_y$.

**핵심 2: Spectral coordinates / Spectral 좌표 (Eq. 3 변형)**

$$
\bar x_t = V^\top x_t,\qquad \bar y = \Sigma^\dagger U^\top y,\qquad \bar y^{(i)} = \bar x_0^{(i)} + n_i,\quad n_i \sim \mathcal N(0, \sigma_y^2/s_i^2)
$$

**한국어**: $V^\top$ 변환 후 각 component는 독립적인 1차원 noisy observation. *singular value가 작을수록 normalised noise가 커짐.*

**English**: After the $V^\top$ transform each component is an independent 1-D noisy observation; *smaller singular values produce larger normalised noise.*

**핵심 3: 세 가지 transition case / Three transition cases (Eq. 5, schematic)**

$$
q^{(t)}(\bar x_t^{(i)}\mid x_{t+1}, x_0, y) =
\begin{cases}
\mathcal N\bigl(\bar x_0^{(i)} + \sqrt{1-\eta^2}\sigma_t \tfrac{\bar x_{t+1}^{(i)} - \bar x_0^{(i)}}{\sigma_{t+1}},\,\eta^2 \sigma_t^2\bigr) & s_i = 0\\
\mathcal N\bigl(\bar x_0^{(i)} + \sqrt{1-\eta^2}\sigma_t \tfrac{\bar y^{(i)} - \bar x_0^{(i)}}{\sigma_y/s_i},\,\eta^2 \sigma_t^2\bigr) & \sigma_t < \sigma_y/s_i\\
\mathcal N\bigl((1-\eta_b)\bar x_0^{(i)} + \eta_b \bar y^{(i)},\,\sigma_t^2 - \tfrac{\sigma_y^2}{s_i^2}\eta_b^2\bigr) & \sigma_t \ge \sigma_y/s_i
\end{cases}
$$

**한국어**: 측정 정보의 가용성($s_i$)과 노이즈 비교($\sigma_t$ vs $\sigma_y/s_i$)에 따라 spectral component마다 다른 update.

**English**: Three regimes — pure generative when no measurement, gentle conditioning when diffusion noise is small, direct injection when diffusion noise dominates.

**핵심 4: ELBO equivalence / ELBO 동등성 (Theorem 3.2)**

$$
\mathcal L_{\mathrm{DDRM}}\Big|_{\eta=1,\;\eta_b = 2\sigma_t^2/(\sigma_t^2 + \sigma_y^2/s_i^2)} = \mathcal L_{\mathrm{DDPM}} = \sum_{t=1}^T \gamma_t \mathbb E\|x_0 - f_\theta^{(t)}(x_t)\|_2^2
$$

**한국어**: 특정 hyperparameter에서 DDRM의 학습 목적이 unconditional DDPM과 정확히 동일 → 사전학습 DDPM 그대로 사용.

**English**: At specific $\eta, \eta_b$ the DDRM ELBO reduces to DDPM's — so any pre-trained DDPM serves as the optimal DDRM denoiser, no retraining required.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**:
- **§3.1-3.2**: variational distribution 정의와 SVD 분해 — 가장 중요. spectral coordinate 매핑을 paper-and-pencil로 직접 따라가 볼 것.
- **§3.3 (Theorem 3.2)**: 증명은 Appendix C에 있음. 본문에서는 결론과 해석 (왜 사전학습 DDPM 그대로 작동하는가)에 집중.
- **§3.4 (sub-sampling)**: DDIM의 1000→20 step trick이 그대로 적용됨. DDIM 익숙하지 않으면 Song 2021을 먼저 훑어볼 것.
- **§3.5 (memory-efficient SVD)**: 실제 구현에서 가장 중요한 부분. $V$가 $\Theta(n^2)$로 저장되면 256×256에서 38 GB.
- **§5 (experiments)**: Tables 1-2에서 NFE 컬럼과 noisy measurement 결과에 주목. DDRM의 우위는 노이즈가 있을 때 가장 명확.
- **Common stumbling blocks**: (1) variance-exploding ↔ variance-preserving 변환 (Appendix B), (2) Eq. 5의 세 case 분기 조건 시각화, (3) DDRM-CC가 ground-truth label을 사용한다는 점의 "unfair advantage".

**English**:
- **§3.1-3.2**: variational distribution and SVD decomposition — the most important section. Trace the spectral mapping by hand.
- **§3.3 (Theorem 3.2)**: full proof in Appendix C; in the main text focus on the consequence (why pre-trained DDPMs work as-is).
- **§3.4**: DDIM's 1000→20 sub-sampling trick. Skim Song 2021 if DDIM is unfamiliar.
- **§3.5**: practical key — naive $V$ storage is 38 GB at 256×256 RGB.
- **§5 Experiments**: focus on the NFE column and noisy-measurement results in Tables 1-2; DDRM's advantage is widest under noise.
- **Stumbling blocks**: (1) variance-exploding ↔ variance-preserving conversion (Appendix B), (2) the three transition cases of Eq. 5 visualised, (3) DDRM-CC's "unfair advantage" of using ground-truth class labels.

---

## 7. 현대적 의의 / Modern Significance

**한국어**: DDRM은 *score-based / diffusion 기반 unsupervised linear-inverse 풀이의 결정적 표준*이 되었다. 직전 paper #25 (Kadkhodaie-Simoncelli)가 "denoiser = prior" 아이디어를 가장 단순한 형태로 제시했다면, DDRM은 이를 *대규모 사전학습 DDPM* 및 *임의 H*와 결합해 실용적 universal solver로 성숙시켰다. 직후 DPS (paper #28), DDNM, $\Pi$GDM이 (i) non-linear (ii) null-space (iii) pseudoinverse-guided 변형으로 확장하며, DiffPIR (paper #30)은 PnP-HQS 골격에 같은 prior를 끼워넣는다. 의료 영상(MRI/CT), 천체 영상, 마이크로스코피 등 *측정 모델은 명확하지만 데이터는 적은* 도메인에서 사전학습 DDPM을 reuse하는 방법론의 표준이다.

**English**: DDRM established the **standard for score-based unsupervised linear-inverse solvers**. Paper #25 introduced the cleanest form of the denoiser-as-prior idea; DDRM scaled it to large pre-trained DDPMs and arbitrary linear $H$ via SVD, turning it into a practical universal solver. Successors — DPS (paper #28), DDNM, $\Pi$GDM — extend it to (i) non-linear $H$, (ii) null-space refinements, (iii) pseudoinverse-guided variants; DiffPIR (paper #30) plugs the same prior into a PnP-HQS skeleton. In medical imaging (MRI/CT), astronomy, and microscopy — domains where the forward model is known but training data is scarce — DDRM is the canonical method for reusing pre-trained DDPMs.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
