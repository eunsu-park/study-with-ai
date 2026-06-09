---
title: "Denoising Diffusion Models for Plug-and-Play Image Restoration"
authors: Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, Luc Van Gool
year: 2023
venue: "CVPR 2023 NTIRE Workshop"
arxiv: "2305.08995"
topic: Low-SNR Imaging / Plug-and-Play with Diffusion Priors
tags: [diffusion-model, plug-and-play, half-quadratic-splitting, hqs, image-restoration, super-resolution, deblurring, inpainting, diffpir, prox]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 30. Denoising Diffusion Models for Plug-and-Play Image Restoration / Plug-and-Play 이미지 복원을 위한 노이즈 제거 확산 모델

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 고전적인 **Plug-and-Play (PnP) 이미지 복원** 골격에 **사전학습된 확산 모델(diffusion model)을 generative denoiser**로 끼워넣는 단일 프레임워크 **DiffPIR**을 제안한다.

(i) **PnP의 핵심**: variable splitting (HQS or ADMM)으로 데이터 항과 prior 항을 분리: $\hat{\boldsymbol x} = \arg\min \frac{1}{2\sigma_n^2}\|\boldsymbol y - \mathcal H(\boldsymbol x)\|^2 + \lambda \mathcal P(\boldsymbol x)$. ADMM 반복은 데이터 부분 (proximal of forward) + prior 부분 ($\text{prox}_{\mathcal P}$를 *임의의 denoiser*로 교체) 두 단계로 분해.

(ii) **DiffPIR의 변경**: $\text{prox}_{\mathcal P}$를 BM3D/CNN 같은 discriminative denoiser 대신 **diffusion model의 한 단계 reverse step (Tweedie 추정 + 약간의 노이즈 재주입)** 으로 교체. 이로써 denoiser가 generative prior가 됨.

(iii) **통합 알고리즘**: 동일한 코드로 motion deblurring, Gaussian deblurring, super-resolution, inpainting 모두 처리. **NFE ≤ 100**으로 SOTA 달성 — DPS의 1000 NFEs 대비 10× 가속.

(iv) **결과 (Table 1-3)**: FFHQ, ImageNet에서 reconstruction fidelity (PSNR) + perceptual quality (LPIPS, FID) 양쪽에서 DDRM, DPS, PnP-ADMM(BM3D), Restormer 등 모든 방법과 견주거나 능가.

### English
**DiffPIR** plugs a pre-trained denoising-diffusion model into the classical **Plug-and-Play Image Restoration (PnP-IR)** framework, replacing the Gaussian/CNN denoiser with a generative-diffusion proximal step.

(i) **PnP idea**: variable splitting (HQS/ADMM) decomposes $\hat{\boldsymbol x} = \arg\min \frac{1}{2\sigma_n^2}\|\boldsymbol y - \mathcal H(\boldsymbol x)\|^2 + \lambda \mathcal P(\boldsymbol x)$ into a forward-model proximal step and a prior proximal step; PnP replaces $\text{prox}_{\lambda\mathcal P}$ with **any denoiser**.

(ii) **DiffPIR change**: substitute the denoiser by one **reverse-diffusion step**: $\hat{\boldsymbol x}_0$ via Tweedie + (optional) re-noising at level $t-1$. This turns a discriminative denoiser into a generative prior with no retraining.

(iii) **Unified pipeline** for super-resolution, motion/Gaussian deblurring, and inpainting at $\le 100$ NFEs — an order of magnitude faster than DPS while matching or beating quality.

(iv) **Quantitative**: On FFHQ and ImageNet DiffPIR matches/beats DDRM, DPS, PnP-ADMM (BM3D), Restormer in both fidelity (PSNR) and perceptual quality (LPIPS, FID).

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction / 도입

#### 한국어
- **Plug-and-Play의 매력**: 어떤 forward model에도 동일 코드 — denoiser 한 개로 SR, deblur, inpaint, demosaic 모두 가능. Venkatakrishnan et al. (2013)이 도입, Romano et al. (2017) RED가 정형화.
- **고전 PnP의 한계**: BM3D, DnCNN 같은 *discriminative* denoiser는 본질적으로 MMSE 또는 MAP 추정기 — 고품질 *generative* prior가 아님. 결과로 텍스처/세부 부정확, perceptual quality 한계.
- **확산 모델의 역할**: DDPM은 본질적으로 학습된 noise-conditional denoiser의 sequence. 한 단계 step이 곧 high-quality denoiser → PnP의 prior로 직접 사용 가능.
- **이전 시도**:
  - **Score-SDE based inverse (Song et al.)**: projection 기반, noisy 환경 취약.
  - **DDRM (Kawar et al. 2022)**: SVD 기반 spectral 도메인 — linear separable에 한정.
  - **DPS (Chung et al. 2023, paper #28)**: gradient 기반, nonlinear 가능하지만 1000 NFEs.
- **DiffPIR의 위치**: PnP의 모듈성 + diffusion의 generative quality. DPS와 비교해 더 적은 NFEs로 비슷한 품질.

#### English
- PnP-IR's appeal: one denoiser, many forward models (SR, deblur, inpaint, demosaic). Foundational works: Venkatakrishnan 2013, Romano 2017.
- Classical PnP uses discriminative denoisers (BM3D, DnCNN) that are MMSE/MAP estimators rather than generative priors — limits perceptual quality.
- Diffusion models are conditional denoisers by construction; using one reverse step as the PnP $\text{prox}$ inherits their generative quality.
- Compared to Score-SDE-projection (noise-fragile), DDRM (linear-only), and DPS (1000 NFEs), DiffPIR offers PnP modularity, nonlinear generality, and ≤100 NFEs.

---

### Part II: §2 Background / 배경

#### 한국어
- **이미지 복원 일반 형식**: 측정 모델 $\boldsymbol y = \mathcal H(\boldsymbol x_0) + \boldsymbol n$, $\boldsymbol n \sim \mathcal N(0, \sigma_n^2 \boldsymbol I)$. MAP:
  $\hat{\boldsymbol x} = \arg\min_{\boldsymbol x} \frac{1}{2\sigma_n^2}\|\boldsymbol y - \mathcal H(\boldsymbol x)\|^2 + \lambda\mathcal P(\boldsymbol x) \quad (1)$.
- **Half-Quadratic Splitting (HQS)**: 보조 변수 $\boldsymbol z$ 도입,
  $\hat{\boldsymbol x} = \arg\min_{\boldsymbol x, \boldsymbol z} \frac{1}{2\sigma_n^2}\|\boldsymbol y - \mathcal H(\boldsymbol x)\|^2 + \lambda\mathcal P(\boldsymbol z) + \frac{\mu}{2}\|\boldsymbol x - \boldsymbol z\|^2$.
  교대 최소화:
  $\boldsymbol x^{(k+1)} = \arg\min_{\boldsymbol x} \frac{1}{2\sigma_n^2}\|\boldsymbol y - \mathcal H(\boldsymbol x)\|^2 + \frac{\mu}{2}\|\boldsymbol x - \boldsymbol z^{(k)}\|^2 \quad$ (data subproblem)
  $\boldsymbol z^{(k+1)} = \arg\min_{\boldsymbol z} \frac{\mu}{2}\|\boldsymbol x^{(k+1)} - \boldsymbol z\|^2 + \lambda \mathcal P(\boldsymbol z) = \text{prox}_{\lambda\mathcal P/\mu}(\boldsymbol x^{(k+1)})$ (prior subproblem)
- **PnP**: 두 번째 subproblem이 정확히 *Gaussian noise에 대한 denoiser*의 정의. 따라서 임의의 high-quality denoiser $D_{\sigma_k}$로 교체:
  $\boldsymbol z^{(k+1)} = D_{\sigma_k}(\boldsymbol x^{(k+1)})$, $\sigma_k^2 = \lambda/\mu$.
- **확산 모델 검토**: DDPM forward $q(\boldsymbol x_t \mid \boldsymbol x_0) = \mathcal N(\sqrt{\bar\alpha_t}\boldsymbol x_0, (1-\bar\alpha_t)\boldsymbol I)$; reverse $\boldsymbol\epsilon_\theta$ network → $\hat{\boldsymbol x}_0(\boldsymbol x_t, t) = (\boldsymbol x_t - \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon_\theta(\boldsymbol x_t, t))/\sqrt{\bar\alpha_t}$ (Tweedie).

#### English
- MAP framework (1) with data fidelity $\frac{1}{2\sigma_n^2}\|\boldsymbol y - \mathcal H(\boldsymbol x)\|^2$ and prior $\lambda\mathcal P$.
- HQS introduces auxiliary $\boldsymbol z$ and decouples into a quadratic data subproblem and a denoising subproblem $\text{prox}_{\lambda\mathcal P/\mu}$.
- PnP substitutes any denoiser $D_{\sigma_k}$ for the prox; the corresponding noise level is $\sigma_k = \sqrt{\lambda/\mu}$.
- Diffusion-model basics: $\hat{\boldsymbol x}_0$ via Tweedie; conditional denoising at every $t$.

---

### Part III: §3 DiffPIR Method / DiffPIR 방법

#### 한국어
**핵심 매핑**: PnP iteration의 $k$번째 step을 diffusion timestep $t_k$로 매핑. PnP의 noise level $\sigma_k$를 diffusion noise level $\sigma_t = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$로 대응.

**알고리즘 (Algorithm 1)**:

```
Input: y, H, sigma_n, schedule {t_K, ..., t_1}, lambda, zeta
x_T ~ N(0, I) at t_K  (initialise at high noise)
for k = K to 1:
    # Step 1: data-fidelity subproblem (closed form for many H)
    x_hat0_data = argmin_z (1/(2 sigma_n^2)) ||y - H(z)||^2 + (rho_k/2) ||z - x_hat0||^2
                  where x_hat0 = (x_t - sqrt(1-alpha_bar_t) eps_theta(x_t, t)) / sqrt(alpha_bar_t)
                  rho_k chosen so that sigma_t^2 = lambda * sigma_n^2 / rho_k
    # Step 2: re-injection of noise to prepare next-iteration x_t
    x_{t-1} = sqrt(alpha_bar_{t-1}) * x_hat0_data + sqrt(1-alpha_bar_{t-1}-zeta^2) * eps_theta + zeta * eps'
return x_0
```

**해석**:
- Step 1은 **PnP의 data subproblem**과 동일. 많은 forward models($\mathcal H$가 stride conv, blur convolution, mask)에 대해 closed-form (e.g., FFT 도메인 inversion).
- Step 2는 **DDPM ancestral step**의 변형 — $\hat{\boldsymbol x}_0$를 $\boldsymbol x_{t-1}$ 잠재로 다시 noise를 주입. $\zeta \in [0,1]$가 stochasticity 제어 (Song et al. 2021a DDIM 형식).

**Forward model별 closed-form** (§3.2):
- **Inpainting**: $\mathcal H(\boldsymbol x) = \boldsymbol M \odot \boldsymbol x$. Step 1 → 픽셀별 평균. 매우 빠름.
- **Deblurring**: $\mathcal H(\boldsymbol x) = \boldsymbol k * \boldsymbol x$. Step 1 → FFT 도메인에서 Wiener-style 1행 1열 inversion (Wang et al. 2008 trick).
- **Super-resolution (SR×s)**: bicubic downsample. Step 1 → polyphase decomposition + diagonal inversion (Zhang et al. 2021 USRNet trick).
- **Nonlinear**: closed form 없음. 대신 gradient step (DPS와 유사).

#### English
DiffPIR maps the $k$-th PnP iteration onto diffusion timestep $t_k$ with $\sigma_t = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$. Each iteration:
1. **Data subproblem** — closed-form solution for inpainting (per-pixel average), deblurring (FFT/Wiener inversion via Wang 2008), and SR×s (polyphase via Zhang 2021's USRNet trick), starting from the Tweedie estimate $\hat{\boldsymbol x}_0$.
2. **Re-injection** — DDIM-style update with stochasticity parameter $\zeta \in [0,1]$ to produce $\boldsymbol x_{t-1}$.

For nonlinear forward models, the data subproblem is solved by a few gradient steps (DPS-like).

---

### Part IV: §3.3 Schedule and hyperparameters / 스케줄과 하이퍼파라미터

#### 한국어
- **노이즈 스케줄 일치**: 핵심은 $\sigma_t^2 = \lambda \sigma_n^2/\mu_k$의 매칭. 측정 잡음 $\sigma_n$이 작을수록 작은 $\sigma_t$, 즉 큰 $t$가 아닌 작은 $t$에서 시작. 일반적으로 $T_{\text{start}}$를 SNR에 맞게 선택.
- **NFEs**: Number of Function Evaluations = sampler step 수. DDPM 1000 → DiffPIR은 timestep skip을 사용해 100 step 이하. DDIM 식의 deterministic step ($\zeta = 0$) 또는 stochastic ($\zeta > 0$) 선택.
- **$\lambda, \zeta$ 튜닝**: $\lambda$는 prior 강도, $\zeta$는 stochasticity. 작업별 권장값 (Table 4).
- **Warm-up**: 최초 몇 step은 $\hat{\boldsymbol x}_0$ 추정 noisy → low weight. 점진적으로 data fidelity 강화.

#### English
- Match $\sigma_t^2 = \lambda\sigma_n^2/\mu_k$; small measurement noise → start from a small $t$. Typically $T_{\text{start}}$ chosen by SNR.
- Sub-100 NFEs via timestep skipping; DDIM-style updates with $\zeta \in [0,1]$ controlling stochasticity.
- $\lambda$ and $\zeta$ are task-specific (Table 4 of the paper).

---

### Part V: §4 Experiments / 실험

#### 한국어
- **Datasets**: FFHQ-256, ImageNet-256 (1k validation each).
- **Tasks**:
  - SR×4 (bicubic), SR×8.
  - Motion deblur (intensity 0.5), Gaussian deblur ($\sigma_k=2.0$).
  - Box inpainting, freeform inpainting.
- **Noise**: $\sigma_n \in \{0, 0.025, 0.05\}$ (Gaussian, on top of $\mathcal H$).
- **Baselines**: DDRM, DPS, $\Pi$GDM, DnCNN-PnP, BM3D-PnP, Restormer (specialised), USRNet (specialised), Real-ESRGAN.

**Headline numbers (FFHQ, Table 1, partial)**:

| Method                        | SR×4 PSNR↑ / LPIPS↓  | Motion Deblur PSNR↑ / LPIPS↓ | Inpaint PSNR↑ / LPIPS↓ | NFEs |
|-------------------------------|----------------------|-------------------------------|--------------------------|------|
| **DiffPIR (ours)**            | **27.36 / 0.236**    | **27.51 / 0.193**             | **31.20 / 0.060**        | 100  |
| DPS                           | 25.81 / 0.219        | 25.46 / 0.221                 | 28.56 / 0.107            | 1000 |
| DDRM                          | 27.40 / 0.265        | 24.63 / 0.302                 | n/a (mask)               | 20   |
| PnP-ADMM (BM3D)               | 24.77 / 0.426        | 23.18 / 0.510                 | 27.45 / 0.241            | 50   |
| $\Pi$GDM                      | 27.28 / 0.252        | 26.45 / 0.219                 | 29.00 / 0.090            | 100  |

(부분 발췌. DiffPIR은 PSNR과 LPIPS 모두에서 baseline 대부분 능가, DPS 대비 10× 가속.)

- **Qualitative**: motion-deblurred 얼굴 이미지에서 DiffPIR이 DPS보다 텍스처 재현 우수. SR×8에서 specialised USRNet/Real-ESRGAN보다 perceptual quality 비슷, fidelity는 우월.
- **NFE-품질 trade-off (Fig. 6)**: DiffPIR은 50 NFEs에서 이미 high quality, 100 NFEs에서 saturate. DPS는 200 NFEs까지 단조 증가. DiffPIR이 PnP 구조 덕분에 더 빠른 수렴.
- **Real-world test**: 실제 motion-blurred photo에 적용해 generative ability 확인.

#### English
- Datasets and tasks as listed; noise levels $\sigma_n \in \{0, 0.025, 0.05\}$.
- Baselines span spectral (DDRM, $\Pi$GDM), gradient (DPS), classical PnP (BM3D, DnCNN), and specialised solvers (Restormer, USRNet, Real-ESRGAN).
- Headline: DiffPIR is best or co-best on every FFHQ row of Table 1, at 100 NFEs (DPS uses 1000). PSNR ≈ +1.5 dB over DPS on motion deblur, LPIPS lower across all tasks.
- DiffPIR converges in 50 NFEs and saturates by 100; DPS still improving up to 200.

---

### Part VI: §5 Discussion / 토론

#### 한국어
- **Modularity의 가치**: 새로운 forward model 추가 = data-fidelity subproblem만 닫힌 형태 도출. score 모델 재학습 불필요. 이미 학습된 FFHQ-DDPM을 그대로 ImageNet 대비 fine-tune 없이 task-by-task로 재사용.
- **DPS와의 관계**: nonlinear case에서 DiffPIR도 gradient step을 사용 → DPS와 매우 유사. 차이는 (a) HQS-style 변수 분리 구조, (b) re-injection step의 deterministic 옵션 ($\zeta=0$).
- **한계**: closed-form 없는 forward model에는 inner optimisation 필요 → 비용 증가. nonlinear blind problem 미지원. PSNR-LPIPS trade-off 여전히 존재 (perceptual quality 추구 시 PSNR 약간 양보).
- **Future**: latent diffusion (LDM) 결합으로 8× 추가 가속, conditional 학습된 diffusion과의 hybridisation, blind kernel estimation.

#### English
- DiffPIR's modularity allows reusing one pre-trained diffusion model across tasks, requiring only the data-fidelity prox to be derived per forward model.
- Relation to DPS: both reduce to similar gradient updates in the nonlinear case; DiffPIR additionally provides the HQS variable-splitting structure and deterministic $\zeta=0$ option.
- Limitations: forward models without closed-form prox need inner optimisation; PSNR-LPIPS trade-off remains; blind problems out of scope.
- Future: latent-diffusion acceleration, conditional-diffusion hybridisation, blind kernel estimation.

---

## 3. Key Takeaways / 핵심 시사점

### 한국어
1. **Diffusion model = generative denoiser sequence** — DDPM은 noise-conditional denoiser의 시계열. 따라서 PnP의 prox로 한 단계를 직접 끼워 넣을 수 있다.
2. **PnP의 modularity 보존** — DiffPIR은 forward model마다 data-fidelity prox 식만 유도 (FFT, polyphase 등의 잘 알려진 트릭). prior는 단일 사전학습 diffusion 그대로.
3. **HQS의 variable splitting이 노이즈 매칭의 핵심** — PnP의 $\sigma_k$를 diffusion의 $\sigma_t$로 매칭함으로써 PnP iteration ↔ reverse-diffusion timestep 간의 자연스러운 isomorphism.
4. **DPS의 10× 가속** — DPS는 1000 NFEs, DiffPIR은 100 NFEs 이하로 비슷/더 좋은 결과. PnP 구조의 빠른 수렴 + DDIM-style sub-sampling의 결합 효과.
5. **DDIM 형식의 stochasticity 제어** — $\zeta=0$ deterministic 또는 $\zeta>0$ stochastic. 측정 잡음이 클 때 stochasticity가 perceptual quality에 도움.
6. **재학습 불필요** — sample-time 알고리즘만 변경. 산업/임상 환경에서 큰 이점.
7. **PSNR + LPIPS 동시 우위** — 보통은 fidelity↔perceptual의 trade-off가 있지만, DiffPIR은 양 지표에서 모두 SOTA. PnP 구조의 균형성 덕분.

### English
1. **Diffusion is a sequence of conditional denoisers** — naturally a PnP prior with no extra training.
2. **PnP modularity preserved** — only the data-fidelity prox needs derivation per forward model; FFT/polyphase tricks suffice for blur and SR.
3. **HQS variable splitting matches noise levels** — the noise schedule of diffusion matches the PnP shrinkage schedule, giving a clean iteration ↔ timestep isomorphism.
4. **10× speedup over DPS** — PnP structure plus DDIM-style sub-sampling reduces 1000 NFE → 100 NFE while matching or exceeding quality.
5. **DDIM-style stochasticity tuning** with $\zeta\in[0,1]$ trades determinism for perceptual quality.
6. **No retraining** — the diffusion model is reused across tasks, a major industrial/medical benefit.
7. **Best on both PSNR and LPIPS** — usually fidelity and perceptual metrics trade off; DiffPIR wins both.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 MAP formulation / MAP 형식
$$
\hat{\boldsymbol x} = \arg\min_{\boldsymbol x} \frac{1}{2\sigma_n^2}\|\boldsymbol y - \mathcal H(\boldsymbol x)\|_2^2 + \lambda\mathcal P(\boldsymbol x)
$$

### 4.2 Half-quadratic splitting (HQS) / 반제곱 분할
Auxiliary $\boldsymbol z$:
$$
\hat{\boldsymbol x}, \hat{\boldsymbol z} = \arg\min_{\boldsymbol x,\boldsymbol z} \frac{1}{2\sigma_n^2}\|\boldsymbol y - \mathcal H(\boldsymbol x)\|^2 + \lambda \mathcal P(\boldsymbol z) + \frac{\mu}{2}\|\boldsymbol x - \boldsymbol z\|^2.
$$
Alternating updates:
$$
\boldsymbol x^{(k+1)} = \arg\min_{\boldsymbol x} \frac{1}{2\sigma_n^2}\|\boldsymbol y - \mathcal H(\boldsymbol x)\|^2 + \frac{\mu}{2}\|\boldsymbol x - \boldsymbol z^{(k)}\|^2,
$$
$$
\boldsymbol z^{(k+1)} = \text{prox}_{\lambda\mathcal P/\mu}(\boldsymbol x^{(k+1)}) = D_{\sigma_k}(\boldsymbol x^{(k+1)}), \quad \sigma_k = \sqrt{\lambda/\mu}.
$$

### 4.3 Tweedie-based diffusion denoising / Tweedie 기반 확산 디노이징
$$
\hat{\boldsymbol x}_0(\boldsymbol x_t, t) = \frac{1}{\sqrt{\bar\alpha_t}}\big(\boldsymbol x_t - \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon_\theta(\boldsymbol x_t, t)\big).
$$

### 4.4 Closed forms for data subproblem / 데이터 부분문제의 닫힌 형태
**Inpainting** $\mathcal H(\boldsymbol x) = \boldsymbol M \odot \boldsymbol x$:
$$
\boldsymbol x^* = \frac{\boldsymbol M \odot \boldsymbol y / \sigma_n^2 + \mu \boldsymbol z}{\boldsymbol M / \sigma_n^2 + \mu}.
$$
**Deblurring** $\mathcal H(\boldsymbol x) = \boldsymbol k * \boldsymbol x$ (FFT, Wang 2008):
$$
\boldsymbol x^* = \mathcal F^{-1}\!\left[\frac{\overline{\mathcal F(\boldsymbol k)} \mathcal F(\boldsymbol y) / \sigma_n^2 + \mu \mathcal F(\boldsymbol z)}{|\mathcal F(\boldsymbol k)|^2 / \sigma_n^2 + \mu}\right].
$$
**Super-resolution ×s** (polyphase, Zhang 2021): similar diagonal solver in FFT/polyphase basis.

### 4.5 DDIM-style re-injection / DDIM 형식 재주입
$$
\boldsymbol x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat{\boldsymbol x}_0^{\text{data}} + \sqrt{1-\bar\alpha_{t-1}-\zeta^2}\,\boldsymbol\epsilon_\theta(\boldsymbol x_t, t) + \zeta\,\boldsymbol\epsilon', \quad \boldsymbol\epsilon' \sim \mathcal N(0, \boldsymbol I).
$$

### 4.6 Worked example / 워크드 예제: 1차원 디블러링

**설정**: $\boldsymbol x_0 \in \mathbb R^4$, blur kernel $\boldsymbol k = [0.5, 0.5]$ (length 2). $\boldsymbol y = \boldsymbol k * \boldsymbol x_0 + \boldsymbol n$, $\sigma_n = 0.05$. 진실 $\boldsymbol x_0 = (1, -1, 1, -1)^\top$, $\boldsymbol y \approx (0, 0, 0, 0)^\top + (0.05) \cdot \boldsymbol n$ — high-frequency 정보가 거의 다 손실된 측정.

**Initialise** $\boldsymbol x_T \sim \mathcal N(0, \boldsymbol I)$ at $t = T = 100$.

**Iteration $k$ at $t_k$**:
1. Tweedie: $\hat{\boldsymbol x}_0 = (\boldsymbol x_t - \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon_\theta)/\sqrt{\bar\alpha_t}$. 가정 $t_k = 50$, $\bar\alpha_t = 0.6$, $\boldsymbol\epsilon_\theta$가 주는 추정 = $(0.5, -0.6, 0.4, -0.5)$ (대략 $\boldsymbol x_0$의 noisy 근사).
2. Data subproblem (FFT inversion): $\mathcal F(\boldsymbol k) = (1, j, -1, -j)$ (DFT length 4). $|\mathcal F(\boldsymbol k)|^2 = (1, 1, 1, 1)$. $\mu_k = \lambda/\sigma_t^2$, 가정 $\mu_k = 10$.

   $\boldsymbol x^* = \mathcal F^{-1}\big[\tfrac{\overline{\mathcal F(\boldsymbol k)}\mathcal F(\boldsymbol y)/\sigma_n^2 + \mu_k \mathcal F(\hat{\boldsymbol x}_0)}{|\mathcal F(\boldsymbol k)|^2/\sigma_n^2 + \mu_k}\big]$.

   $|\mathcal F(\boldsymbol k)|^2/\sigma_n^2 = 1/0.0025 = 400$. 따라서 분모 $= 400 + 10 = 410$ (모든 frequency).

   분자에서 $\mathcal F(\boldsymbol y) \approx 0$ (대칭 신호 + 작은 noise) → 거의 $10 \cdot \mathcal F(\hat{\boldsymbol x}_0)$. 따라서 $\boldsymbol x^* \approx \tfrac{10}{410} \hat{\boldsymbol x}_0 \approx 0.024 \hat{\boldsymbol x}_0$ — Wiener 강한 shrinkage (high-frequency 영역에서 측정값 의존도 큼).

3. Re-injection: $\boldsymbol x_{t-1} = \sqrt{\bar\alpha_{t-1}}\boldsymbol x^* + \sqrt{1-\bar\alpha_{t-1}-\zeta^2}\boldsymbol\epsilon_\theta + \zeta\boldsymbol\epsilon'$. $\zeta=0$이면 deterministic.

**해석**: 측정값에 신호가 거의 없는 frequency(예: Nyquist 인근)에서는 prior(diffusion)에 거의 의존, 측정값이 신뢰할 만한 frequency에서는 측정값 우위. PnP의 frequency-dependent balance를 자동 수행.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1977 -- HQS introduced (Geman & Yang 1995 popularises;
        roots in Geman-Geman 1984 stochastic relaxation).
   |
2007 -- BM3D (Dabov et al.) — gold-standard non-learned denoiser.
   |
2013 -- PnP Priors (Venkatakrishnan, Bouman, Wohlberg) — denoiser
        as prox in ADMM. Model-based + denoiser modularity.
   |
2017 -- RED (Romano et al.) — formalises denoiser-as-prior with
        explicit regularisation gradient.
2017 -- IRCNN (Zhang et al.) — CNN denoiser inside HQS;
        forerunner of DiffPIR's structure.
   |
2020 -- DDPM (Ho et al.) — diffusion goes mainstream.
2021 -- USRNet / Restormer — task-specialised deep models;
        high quality but per-task retraining.
   |
2022 -- DDRM (Kawar et al.) — spectral-domain diffusion inverse;
        linear only.
2022 -- MCG (Chung et al.) — gradient + projection diffusion inverse.
   |
*** 2023 — DiffPIR (this paper, CVPR-NTIRE) ***
        Plug diffusion into HQS PnP: one prior, all tasks,
        ≤100 NFEs, SOTA on PSNR + LPIPS.
2023 -- DPS (Chung et al., paper #28) — concurrent gradient-based
        nonlinear inverse; 1000 NFEs.
2023 -- $\Pi$GDM (Song et al.) — pseudoinverse-guided diffusion.
   |
2024 -- Latent DiffPIR variants; clinical MRI/CT applications.
```

```
1995 -- HQS (Geman & Yang)
2013 -- PnP Priors (Venkatakrishnan)
2017 -- IRCNN (CNN PnP), RED
2020 -- DDPM
2023 -- DiffPIR: diffusion as PnP prox
2023 -- DPS (paper #28, gradient-based concurrent)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| 관련 논문 / Related paper | 관계 / Relation |
|----------------------|----------------------------------------------------|
| **Venkatakrishnan et al. 2013 (PnP)** — DiffPIR의 직계 조상; denoiser-as-prox의 originator. / Direct ancestor: introduced the denoiser-as-prox principle DiffPIR adopts. |
| **Geman & Yang 1995 (HQS)** — variable splitting의 수학적 골격. / Provides the variable-splitting machinery used by DiffPIR. |
| **Zhang et al. 2017 (IRCNN) / 2021 (USRNet)** — CNN denoiser-PnP의 직계 선조. DiffPIR은 CNN을 diffusion으로 교체. / Direct precursors using CNN denoisers; DiffPIR substitutes a diffusion model. |
| **Ho et al. 2020 (DDPM)** — prior로 사용되는 backbone. 사전학습 모델 그대로. / Backbone diffusion prior used as-is. |
| **Chung et al. 2023 (DPS, paper #28)** — concurrent. 같은 동기, 다른 알고리즘 (PnP-HQS vs. ancestral DDPM + gradient). NFE 차이 큼. / Concurrent; same goal, different mechanism — DiffPIR uses HQS variable splitting at ≤100 NFEs vs DPS's 1000-NFE ancestral sampler. |
| **Daras et al. 2023 (Ambient, paper #29)** — corruption-trained diffusion을 DiffPIR의 prior로 결합 가능. / Composable: an Ambient-trained diffusion can plug into DiffPIR as the generative prior. |
| **Kawar et al. 2022 (DDRM)** — spectral 접근. linear에 한정. DiffPIR은 nonlinear 처리 가능. / Spectral, linear-only counterpart; DiffPIR supports nonlinear via gradient inner-loop. |
| **Donoho-Johnstone 1994 (paper #01)** — wavelet shrinkage = analytic denoiser. PnP의 가장 단순한 인스턴스라 볼 수 있음. / Analytic denoiser; the simplest instance of the PnP idea, generalised by DiffPIR to learned generative priors. |

---

## 7. References / 참고문헌

- Zhu, Y., Zhang, K., Liang, J., Cao, J., Wen, B., Timofte, R., & Van Gool, L. (2023). "Denoising Diffusion Models for Plug-and-Play Image Restoration (DiffPIR)." *CVPR 2023 NTIRE*. arXiv:2305.08995.
- Venkatakrishnan, S. V., Bouman, C. A., & Wohlberg, B. (2013). "Plug-and-Play Priors for Model Based Reconstruction." *GlobalSIP*.
- Romano, Y., Elad, M., & Milanfar, P. (2017). "The Little Engine that Could: Regularization by Denoising (RED)." *SIAM J. Imaging Sciences*, 10(4), 1804-1844.
- Zhang, K., Zuo, W., Gu, S., & Zhang, L. (2017). "Learning Deep CNN Denoiser Prior for Image Restoration (IRCNN)." *CVPR 2017*.
- Zhang, K., Van Gool, L., & Timofte, R. (2021). "Deep Unfolding Network for Image Super-Resolution (USRNet)." *CVPR 2021*.
- Geman, D., & Yang, C. (1995). "Nonlinear Image Recovery with Half-Quadratic Regularization." *IEEE T-IP*, 4(7), 932-946.
- Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*.
- Song, J., Meng, C., & Ermon, S. (2021). "Denoising Diffusion Implicit Models (DDIM)." *ICLR 2021*.
- Chung, H., Kim, J., McCann, M. T., Klasky, M. L., & Ye, J. C. (2023). "Diffusion Posterior Sampling for General Noisy Inverse Problems." *ICLR 2023*. arXiv:2209.14687.
- Kawar, B., Elad, M., Ermon, S., & Song, J. (2022). "Denoising Diffusion Restoration Models." *NeurIPS 2022*.
- Code: https://github.com/yuanzhi-zhu/DiffPIR
