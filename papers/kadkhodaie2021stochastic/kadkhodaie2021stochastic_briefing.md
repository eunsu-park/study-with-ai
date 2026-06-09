---
title: "Pre-Reading Briefing: Stochastic Solutions for Linear Inverse Problems Using the Prior Implicit in a Denoiser"
paper_id: "25_kadkhodaie_2021"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Stochastic Solutions for Linear Inverse Problems Using the Prior Implicit in a Denoiser: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Z. Kadkhodaie & E. P. Simoncelli, *NeurIPS* 34 (2021), arXiv:2007.13640
**Author(s)**: Zahra Kadkhodaie, Eero P. Simoncelli
**Year**: 2021

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 60년 묵은 통계학 항등식 — **Miyasawa (1961) / Tweedie 공식** — 이 현대 딥러닝 디노이저와 결합되면 *영상 사전분포의 score* 를 자동으로 제공함을 보인다. 가우시안 노이즈로 오염된 관측 $y = x + z$의 MMSE 디노이저는 $\hat x(y) = y + \sigma^2 \nabla_y \log p_\sigma(y)$로 표현되며, 따라서 디노이저 잔차 $f(y) = \hat x(y) - y$가 곧 score 추정값이다. 저자들은 이를 (i) 사전에서의 high-probability 샘플링과 (ii) 임의의 선형 역문제(inpainting, super-resolution, deblurring, compressive sensing)에 *동일 알고리즘*으로 적용하며, DIP/DeepRED 대비 비슷하거나 우월한 PSNR을 *2 orders of magnitude 더 빠르게* 달성한다.

### English
The paper revives **Miyasawa's lemma / Tweedie's formula** — a 60-year-old statistical identity — and shows that any modern $L_2$-trained Gaussian denoiser implicitly provides $\nabla_y \log p_\sigma(y)$, the *score of the noisy-image density*. The denoiser residual $f(y) = \hat x(y) - y$ equals $\sigma^2 \nabla_y \log p_\sigma(y)$. The authors leverage this to (i) draw high-probability samples from the implicit prior and (ii) solve arbitrary linear inverse problems (inpainting, super-resolution, deblurring, compressive sensing) with one unified algorithm. Results match or beat DIP/DeepRED while running ~150× faster (9 s vs 1,190 s on 4× Set5 SR).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**: 2020-2021년은 score-based generative model (NCSN, DDPM)이 막 부상하던 시점이다. Plug-and-Play (Venkatakrishnan 2013)은 디노이저를 ADMM의 proximal operator 자리에 끼워 넣어 image restoration을 했지만 *왜* 그것이 작동하는지에 대한 정직한 해석은 부족했다. Vincent (2011)은 score matching ↔ denoising autoencoder의 동치를 보였지만 *학습 목적함수의 minimization 조건*을 통한 간접 도출이었다. 이 논문은 *최적 MMSE 추정값 자체가 score를 정확히 표현*하는 더 직접적인 경로를 제시하며, score-based diffusion의 시대로 가는 결정적 다리를 놓았다.

**English**: In 2020-2021 score-based generative models (NCSN, DDPM) were just emerging. Plug-and-Play methods plugged denoisers into ADMM proximal slots without principled justification, while Vincent's (2011) score-matching ↔ DAE equivalence was an indirect derivation through loss-minimization conditions. This paper provides a more direct route — the *MMSE optimum itself* is the score — bridging classical empirical Bayes to modern diffusion-style restoration.

### 타임라인 / Timeline

```
1956      Robbins — Empirical Bayes framework
1961      Miyasawa — denoiser-score identity for Gaussian noise
2011      Vincent — DAE ↔ score matching equivalence
2013      Venkatakrishnan+ — Plug-and-Play Priors
2019      Song-Ermon — NCSN (annealed Langevin via score matching)
2020      Mohan+ — Bias-free CNN denoisers (BF-CNN)
2020      Ho+ — DDPM
2021 ★★   KADKHODAIE-SIMONCELLI — THIS PAPER
2022      Kawar+ — DDRM (paper #26)
2023      Chung+ — DPS (paper #28)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**:
- **Empirical Bayes 와 Tweedie 공식**: 가우시안 관측의 사후 평균이 score 함수와 어떻게 연결되는지.
- **Score matching**: $\nabla_y \log p(y)$ 학습의 기본. Vincent 2011의 DAE 동치.
- **Langevin dynamics**: stochastic gradient ascent 기반 sampling. annealed (temperature-decreasing) 변형.
- **Gaussian scale-space**: prior의 heat-equation evolution; $\sigma$를 시간으로 해석.
- **Linear measurement matrix**: $M^T x = x^c$, SVD, projection $MM^T$ 와 orthogonal complement $I - MM^T$.
- **DnCNN / BF-CNN denoiser architecture**: residual learning + bias-free 구조.

**English**:
- **Empirical Bayes & Tweedie's formula**: how the posterior mean under Gaussian observation relates to the score.
- **Score matching**: $\nabla_y \log p(y)$ estimation; Vincent's 2011 DAE equivalence.
- **Langevin dynamics**: stochastic gradient ascent for sampling; annealed (temperature-decreasing) variants.
- **Gaussian scale-space**: prior's heat-equation evolution with $t = \sigma^2$.
- **Linear measurement matrix**: $M^T x = x^c$, SVD, projection $MM^T$ vs. complement $I - MM^T$.
- **DnCNN / BF-CNN architecture**: residual learning + bias-free convolutional network.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Miyasawa identity | $\hat x(y) = y + \sigma^2 \nabla_y \log p_\sigma(y)$ — 가우시안 노이즈 하에서 MMSE 디노이저가 score와 연결되는 공식 / Identity linking MMSE denoiser to score under Gaussian noise. |
| Score / 스코어 | $\nabla_y \log p(y)$, 분포 로그의 그래디언트 / Gradient of the log density; "force field" pulling toward modes. |
| Denoiser residual / 디노이저 잔차 | $f(y) = \hat x(y) - y$, score 추정량 (scaled by $\sigma^2$) / Denoiser output minus input; an estimate of $\sigma^2 \nabla_y \log p_\sigma(y)$. |
| BF-CNN | Bias-Free CNN (Mohan+ 2020). 모든 additive bias 제거 → 모든 $\sigma$에 자동 일반화 / Bias-free denoiser homogeneous in input scale; one network for all noise levels. |
| Coarse-to-fine | $\sigma$를 큰 값에서 작은 값으로 점진적 감소시키며 sampling / Iterative descent reducing $\sigma$ from large to small — analogous to simulated annealing. |
| Langevin step | $y \leftarrow y + h\,f(y) + \gamma z$, deterministic + stochastic 결합 step / Deterministic gradient step plus injected noise; one Langevin update. |
| Effective noise variance | $\sigma_t^2 = (1-h_t)^2\sigma_{t-1}^2 + \gamma_t^2$ — denoiser correction 후 남은 노이즈 분산 / Variance remaining after the deterministic correction plus the freshly injected component. |
| Constrained sampling | 측정 부분공간에 conditioning한 sampling. Eq. 9의 직교 분해 / Sampling conditioned on $M^T x = x^c$ via the orthogonal split (Eq. 9). |
| Projection $(I - MM^T)$ | 측정 부분공간에 직교한 성분만 남기는 사영 / Projector onto the orthogonal complement of the measurement subspace. |
| DIP / DeepRED | 비교 baseline. per-image network optimization 방법 / Per-image optimization baselines outperformed by Algorithm 2 in run-time and PSNR. |

---

## 5. 수식 미리보기 / Equations Preview

**핵심 1: Miyasawa-Tweedie 항등식 / Miyasawa-Tweedie identity (Eq. 3)**

$$
\hat x(y) = y + \sigma^2 \nabla_y \log p_\sigma(y)
$$

**한국어**: 가우시안 노이즈 $z\sim\mathcal N(0,\sigma^2 I)$ 하 MMSE 디노이저는 입력에 *score 추정값을 더한 것*으로 정확히 표현된다. 잔차 $f(y) = \hat x(y) - y$는 $\sigma^2 \nabla_y \log p_\sigma(y)$ 자체.

**English**: Under Gaussian noise the MMSE denoiser is *exactly* the input plus $\sigma^2$ times the score. The residual $f(y)$ literally is $\sigma^2 \nabla_y \log p_\sigma(y)$.

**핵심 2: Stochastic ascent on prior / 사전에서의 stochastic ascent (Eq. 5)**

$$
y_t = y_{t-1} + h_t\,f(y_{t-1}) + \gamma_t z_t,\quad z_t \sim \mathcal N(0, I)
$$

**한국어**: 매 step에서 디노이저 잔차로 deterministic 이동 + 약간의 white noise 재주입. $h_t \in [0,1]$ 와 $\gamma_t$가 hyperparameter.

**English**: At each step move along the denoiser residual (deterministic) and inject white noise (stochastic) to escape local minima.

**핵심 3: Adaptive step size / 적응적 step size (Eq. 8)**

$$
\gamma_t^2 = [(1 - \beta h_t)^2 - (1 - h_t)^2]\,\frac{\|f(y_{t-1})\|^2}{N}
$$

**한국어**: 디노이저 잔차의 norm 자체가 noise std의 추정량 → step size를 자동으로 조정. $\beta \in [0,1]$이 stochasticity 제어.

**English**: The denoiser's residual norm self-estimates $\sigma_{t-1}$, giving an adaptive step schedule with no manual tuning.

**핵심 4: Constrained sampling gradient / 제약 샘플링 그래디언트 (Eq. 9)**

$$
\sigma^2 \nabla_y \log p(y\mid x^c) = (I - MM^T) f(y) + M(x^c - M^T y)
$$

**한국어**: 조건부 score가 두 직교 성분으로 분해 — 측정 부분공간에 직교한 prior pull + 측정 부분공간으로의 data pull. 어떤 선형 역문제에도 동일한 디노이저 사용.

**English**: The conditional score splits into two orthogonal pieces: prior gradient on $\ker(M^T)$, and a measurement-consistency pull on $\text{range}(M)$. The same denoiser handles every linear inverse problem; only $M$ changes.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**:
- **§1.2-1.3 (Miyasawa proof)**: Eq. 3-4의 두 줄 증명을 직접 따라가며 다시 유도해 볼 것. 이것이 논문의 수학적 심장.
- **§2 (Algorithm 1)**: Eq. 5-8의 step-size schedule 도출 흐름을 hyperparameter $h_t, \gamma_t, \beta$ 의미와 함께 이해.
- **§3.1 (Eq. 9 derivation)**: $p(y\mid x^c)$의 Bayes factorization과 직교 분해. 이 부분이 가장 까다로우니 Appendix와 함께 천천히.
- **Tables 1-3**: PSNR 수치보다 *run-time*과 *Ours-avg vs Ours single*의 trade-off에 주목.
- **Common stumbling blocks**: (1) BF-CNN이 *왜* 모든 $\sigma$에서 작동하는지 (homogeneity 논변), (2) Eq. 6의 effective noise variance 유도, (3) constrained init $y_0 \sim \mathcal N(0.5(I-MM^T)e + Mx^c, \sigma_0^2 I)$의 의미.

**English**:
- **§1.2-1.3 (Miyasawa proof)**: re-derive Eqs. 3-4 yourself — the two-line proof is the mathematical heart.
- **§2 (Algorithm 1)**: trace the derivation of Eqs. 5-8 with the meaning of $h_t, \gamma_t, \beta$ in mind.
- **§3.1 (Eq. 9 derivation)**: Bayes factorization plus orthogonal split — the trickiest part; consult the appendix.
- **Tables 1-3**: focus on run-time and the *Ours-avg vs Ours single-sample* trade-off, not just PSNR.
- **Stumbling blocks**: (1) why BF-CNN generalises across $\sigma$ (homogeneity argument), (2) deriving Eq. 6's effective noise variance, (3) interpreting the constrained init.

---

## 7. 현대적 의의 / Modern Significance

**한국어**: 이 논문은 *score-based diffusion 시대로의 결정적 다리*다. NCSN/DDPM의 복잡한 학습 손실 (denoising score matching, ELBO) 없이도 *그냥 $L_2$ 디노이저로 score를 얻을 수 있음*을 명시적으로 보였다. 후속 inverse-problem 논문들 — DDRM (paper #26), DPS (paper #28), $\Pi$GDM, RED-Diff, DiffPIR (paper #30) — 이 모두 이 논문의 algorithm skeleton (Langevin step + projected gradient + coarse-to-fine schedule)을 변형한 것이다. 또한 BF-CNN 같은 작고 단일한 denoiser가 거대 사전학습 generative prior와 비슷한 일을 할 수 있다는 점에서, 자원이 제한된 과학 imaging (low-photon microscopy, MRI, helio-imaging) 응용에 직접적 길잡이가 된다.

**English**: This paper is the **decisive bridge to score-based diffusion**. Without invoking elaborate training losses (denoising score matching, ELBO), it shows that *a vanilla $L_2$-trained denoiser already provides the score*. Every later inverse-problem paper — DDRM (paper #26), DPS (paper #28), $\Pi$GDM, RED-Diff, DiffPIR (paper #30) — refines the same algorithmic skeleton (Langevin step + projected gradient + coarse-to-fine schedule). It also shows that a compact denoiser (BF-CNN) can rival massive pre-trained generative priors, making it a direct guide for resource-constrained scientific imaging (low-photon microscopy, MRI, helio-imaging) where training a giant DDPM is impractical.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
