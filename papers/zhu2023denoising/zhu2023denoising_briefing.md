---
title: "Pre-Reading Briefing: Denoising Diffusion Models for Plug-and-Play Image Restoration (DiffPIR)"
paper_id: "30_zhu_2023"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Denoising Diffusion Models for Plug-and-Play Image Restoration (DiffPIR): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Y. Zhu, K. Zhang, J. Liang, J. Cao, B. Wen, R. Timofte, L. Van Gool, *IEEE/CVF CVPR Workshops (NTIRE)* 2023, pp. 1219-1229, arXiv:2305.08995
**Author(s)**: Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, Luc Van Gool
**Year**: 2023

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 고전적 **Plug-and-Play (PnP) 이미지 복원** 골격에 **사전학습된 확산 모델(diffusion model) 을 generative denoiser**로 끼워 넣는 단일 프레임워크 **DiffPIR** 을 제안한다. PnP의 핵심은 variable splitting (HQS or ADMM)으로 데이터 항과 prior 항을 분리하고 prior subproblem $\text{prox}_{\lambda\mathcal P}$ 자리에 *임의의 denoiser*를 끼워 넣는 모듈성. DiffPIR은 이 자리에 **diffusion model의 한 단계 reverse step (Tweedie 추정 + DDIM-style 노이즈 재주입)** 을 넣어 generative prior로 격상시킨다. 결과: motion deblurring, Gaussian deblurring, super-resolution, inpainting을 동일 코드로 처리하며 **NFE ≤ 100**으로 SOTA 달성 — DPS의 1000 NFE 대비 **10× 가속**, PSNR과 LPIPS 양쪽에서 DDRM, DPS, PnP-ADMM(BM3D), Restormer 등을 능가하거나 동등.

### English
**DiffPIR** plugs a pre-trained denoising-diffusion model into the classical **Plug-and-Play Image Restoration (PnP-IR)** framework, replacing the Gaussian/CNN denoiser with a generative-diffusion proximal step. PnP's strength is modularity: variable splitting (HQS/ADMM) decomposes the MAP problem into a data-fidelity prox and a prior prox, and PnP plugs *any denoiser* into the prior slot. DiffPIR substitutes one **reverse-diffusion step** (Tweedie estimate + DDIM-style re-noising) for that prox, elevating it to a generative prior. The result: super-resolution, motion/Gaussian deblurring, and inpainting at $\le 100$ NFEs — an order of magnitude faster than DPS — matching or beating DDRM, DPS, PnP-ADMM (BM3D), Restormer on both fidelity (PSNR) and perceptual quality (LPIPS, FID).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**: PnP-IR은 Venkatakrishnan 등 (2013)이 제안한 모듈식 image restoration 프레임워크다. ADMM/HQS의 prior proximal 자리에 BM3D, DnCNN 같은 임의의 denoiser를 끼워 넣어 한 코드로 SR, deblur, inpaint 모두 처리. Romano 등 (2017) RED 가 정형화. 2017년 IRCNN (Zhang+) 이 CNN denoiser를 사용해 성능을 끌어올렸지만 *discriminative* denoiser는 본질적으로 MMSE/MAP 추정기일 뿐 generative prior가 아니다 — 텍스처/세부 정확도에 한계. 2022-2023년 들어 DDPM이 mainstream이 되며 "확산 모델은 본질적으로 noise-conditional denoiser의 시퀀스" 라는 인식이 퍼졌다. DiffPIR은 자연스러운 다음 단계 — *학습된 generative denoiser를 PnP의 prox로*. DPS (paper #28, concurrent)는 ancestral DDPM에 gradient를 직접 더하는 다른 접근으로, DiffPIR은 *PnP variable splitting 구조*로 더 적은 NFE에 도달.

**English**: PnP-IR was introduced by Venkatakrishnan et al. (2013) as a modular restoration framework: plug *any denoiser* into the prior prox of an ADMM/HQS split, and one codebase handles SR, deblurring, inpainting. Romano et al. (2017, RED) formalised it. Zhang et al. (2017) IRCNN used a CNN denoiser, but discriminative denoisers are MMSE/MAP estimators, not generative priors — limiting texture/detail. By 2022-2023 DDPM was mainstream, and the realisation that "a diffusion model *is* a sequence of conditional denoisers" became pervasive. DiffPIR is the natural next step — *use a learned generative denoiser as the PnP prox*. DPS (paper #28, concurrent) takes a different route — gradient on the ancestral DDPM — while DiffPIR exploits PnP's variable-splitting structure to reach SOTA at fewer NFEs.

### 타임라인 / Timeline

```
1995  — HQS popularised (Geman & Yang)
2007  — BM3D denoiser
2013  — PnP Priors (Venkatakrishnan)
2017  — RED (Romano-Elad-Milanfar); IRCNN (Zhang+)
2020  — DDPM (Ho+)
2021  — DDIM, USRNet, Restormer
2022  — DDRM (paper #26, spectral)
2023 ★★ DiffPIR (THIS PAPER) — diffusion as PnP prox, ≤100 NFE
2023  — DPS (paper #28, concurrent gradient-based)
2023  — ΠGDM, ReSample
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**:
- **MAP framework**: $\hat{\boldsymbol x} = \arg\min \frac{1}{2\sigma_n^2}\|\boldsymbol y - \mathcal H(\boldsymbol x)\|^2 + \lambda\mathcal P(\boldsymbol x)$.
- **Half-Quadratic Splitting (HQS)**: 보조 변수 $\boldsymbol z$로 데이터 항과 prior 항 분리, 교대 최소화.
- **Proximal operator**: $\text{prox}_{\lambda\mathcal P}(\boldsymbol v) = \arg\min_{\boldsymbol x} \frac{1}{2}\|\boldsymbol x - \boldsymbol v\|^2 + \lambda\mathcal P(\boldsymbol x)$.
- **Plug-and-Play 원리**: prior prox = Gaussian denoiser; 임의의 denoiser로 교체.
- **DDPM forward / reverse + Tweedie**: $\hat{\boldsymbol x}_0 = (\boldsymbol x_t - \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon_\theta)/\sqrt{\bar\alpha_t}$.
- **DDIM-style sampling**: $\zeta$ stochasticity parameter, sub-sampling timesteps.
- **FFT 도메인 closed-form**: deblurring (Wang 2008), super-resolution (USRNet/Zhang 2021의 polyphase trick).
- **PSNR vs LPIPS trade-off**: fidelity vs. perceptual quality.

**English**:
- **MAP framework**: $\hat{\boldsymbol x} = \arg\min \frac{1}{2\sigma_n^2}\|\boldsymbol y - \mathcal H(\boldsymbol x)\|^2 + \lambda\mathcal P(\boldsymbol x)$.
- **Half-Quadratic Splitting (HQS)**: introduce auxiliary $\boldsymbol z$, alternate between data and prior subproblems.
- **Proximal operator**: $\text{prox}_{\lambda\mathcal P}(\boldsymbol v) = \arg\min \frac{1}{2}\|\boldsymbol x - \boldsymbol v\|^2 + \lambda\mathcal P(\boldsymbol x)$.
- **Plug-and-Play principle**: the prior prox *is* a Gaussian denoiser; substitute any denoiser.
- **DDPM forward/reverse + Tweedie**: $\hat{\boldsymbol x}_0 = (\boldsymbol x_t - \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon_\theta)/\sqrt{\bar\alpha_t}$.
- **DDIM-style sampling**: $\zeta$ stochasticity parameter, time-step subsetting.
- **FFT-domain closed forms**: deblurring (Wang 2008), super-resolution (USRNet/Zhang 2021 polyphase trick).
- **PSNR vs LPIPS trade-off**: fidelity vs. perceptual quality.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Plug-and-Play (PnP) | denoiser를 ADMM/HQS의 prox 자리에 끼워 넣는 모듈식 프레임워크 (Venkatakrishnan 2013) / Modular framework using any denoiser as the prior proximal operator. |
| Half-Quadratic Splitting (HQS) | 보조 변수로 quadratic coupling, 교대 최소화 / Variable splitting via quadratic coupling; alternating data + denoising subproblems. |
| Proximal operator | $\text{prox}_{\lambda\mathcal P}$ — Gaussian noise removal과 동치 / Defining identity of a Gaussian denoiser; the prior prox in HQS. |
| Data-fidelity subproblem | $\arg\min \frac{1}{2\sigma_n^2}\|\boldsymbol y - \mathcal H(\boldsymbol x)\|^2 + \frac{\mu}{2}\|\boldsymbol x - \boldsymbol z\|^2$. forward model에 대한 closed form 가능 / Quadratic in $\boldsymbol x$; closed-form via FFT/polyphase for many $\mathcal H$. |
| Prior subproblem (denoising) | $\boldsymbol z = D_{\sigma_k}(\boldsymbol x)$, $\sigma_k = \sqrt{\lambda/\mu}$ / Recognised as Gaussian denoising at level $\sigma_k$. |
| DiffPIR substitution | $D_{\sigma_k}$를 한 단계 reverse-diffusion으로 교체 / Replace the denoiser by one diffusion reverse step (Tweedie + re-noising). |
| Tweedie estimate $\hat{\boldsymbol x}_0$ | $(\boldsymbol x_t - \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon_\theta)/\sqrt{\bar\alpha_t}$. score 모델로부터 closed form / Closed-form posterior mean from the trained noise predictor. |
| DDIM-style re-injection | $\boldsymbol x_{t-1} = \sqrt{\bar\alpha_{t-1}}\hat{\boldsymbol x}_0^{\text{data}} + \sqrt{1-\bar\alpha_{t-1}-\zeta^2}\boldsymbol\epsilon_\theta + \zeta\boldsymbol\epsilon'$. $\zeta$ stochasticity / Re-noising step parameterised by $\zeta \in [0,1]$ controlling determinism vs stochasticity. |
| Noise schedule matching | $\sigma_t^2 = \lambda \sigma_n^2/\mu_k$ — PnP shrinkage와 diffusion noise level isomorphism / Matches PnP regularisation strength to diffusion timestep. |
| NFE ≤ 100 | DiffPIR 비용. DPS 1000 vs DiffPIR ≤100 / 10× faster than DPS. |
| Wang 2008 / USRNet trick | deblurring과 SR의 FFT/polyphase closed-form / Standard FFT/polyphase tricks giving closed-form data-prox. |
| Nonlinear $\mathcal H$ | closed-form 없음 → inner gradient step (DPS-like) / No closed form; falls back to gradient inner-loop. |

---

## 5. 수식 미리보기 / Equations Preview

**핵심 1: MAP formulation / MAP 형식 (Eq. 1)**

$$
\hat{\boldsymbol x} = \arg\min_{\boldsymbol x}\,\frac{1}{2\sigma_n^2}\|\boldsymbol y - \mathcal H(\boldsymbol x)\|_2^2 + \lambda\,\mathcal P(\boldsymbol x)
$$

**한국어**: 모든 PnP-IR의 출발점. data-fidelity + prior regularisation의 합.

**English**: Starting point for all PnP-IR — data fidelity plus prior regularisation.

**핵심 2: HQS alternating updates / HQS 교대 갱신**

$$
\boldsymbol x^{(k+1)} = \arg\min_{\boldsymbol x}\,\frac{1}{2\sigma_n^2}\|\boldsymbol y - \mathcal H(\boldsymbol x)\|^2 + \frac{\mu}{2}\|\boldsymbol x - \boldsymbol z^{(k)}\|^2
$$
$$
\boldsymbol z^{(k+1)} = \text{prox}_{\lambda\mathcal P/\mu}(\boldsymbol x^{(k+1)}) = D_{\sigma_k}(\boldsymbol x^{(k+1)}),\quad \sigma_k = \sqrt{\lambda/\mu}
$$

**한국어**: 첫 식은 data subproblem (closed form 가능), 둘째 식은 prior subproblem (Gaussian denoising). PnP는 둘째 식의 $D_{\sigma_k}$를 임의의 denoiser로 교체.

**English**: First equation is the data subproblem (often closed-form); second is the prior subproblem (Gaussian denoising). PnP swaps in any denoiser for $D_{\sigma_k}$.

**핵심 3: Tweedie + DDIM re-injection / Tweedie + DDIM 재주입**

$$
\hat{\boldsymbol x}_0(\boldsymbol x_t, t) = \frac{1}{\sqrt{\bar\alpha_t}}\big(\boldsymbol x_t - \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon_\theta(\boldsymbol x_t, t)\big)
$$
$$
\boldsymbol x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat{\boldsymbol x}_0^{\text{data}} + \sqrt{1 - \bar\alpha_{t-1} - \zeta^2}\,\boldsymbol\epsilon_\theta + \zeta\,\boldsymbol\epsilon'
$$

**한국어**: Tweedie로 $\hat{\boldsymbol x}_0$ 추정 → data subproblem 풀어 $\hat{\boldsymbol x}_0^{\text{data}}$ → DDIM-style re-noising으로 다음 step. $\zeta=0$이면 deterministic.

**English**: Tweedie gives $\hat{\boldsymbol x}_0$ → solve the data subproblem → re-noise via DDIM with stochasticity $\zeta \in [0,1]$ ($\zeta=0$ = deterministic).

**핵심 4: Closed-form for deblurring / 디블러링 닫힌 형태 (Wang 2008)**

$$
\boldsymbol x^* = \mathcal F^{-1}\!\left[\frac{\overline{\mathcal F(\boldsymbol k)} \cdot \mathcal F(\boldsymbol y)/\sigma_n^2 + \mu\,\mathcal F(\boldsymbol z)}{|\mathcal F(\boldsymbol k)|^2/\sigma_n^2 + \mu}\right]
$$

**한국어**: FFT 도메인에서의 Wiener-style 1행 inversion. SR에서는 polyphase trick (Zhang 2021)으로 비슷하게 처리.

**English**: A Wiener-style FFT-domain inversion for blur convolution; SR uses an analogous polyphase trick (Zhang 2021).

---

## 6. 읽기 가이드 / Reading Guide

**한국어**:
- **§2 (Background)**: HQS와 PnP의 기본을 단단히 — 익숙하지 않으면 먼저 Venkatakrishnan 2013과 Zhang 2017 IRCNN 훑어볼 것.
- **§3 (Method)**: Algorithm 1을 한 줄씩 — *PnP iteration ↔ diffusion timestep* 의 isomorphism이 핵심. $\sigma_t^2 = \lambda\sigma_n^2/\mu_k$ 매칭의 의미.
- **§3.2 (Forward-model별 closed form)**: inpainting, deblurring (FFT), SR (polyphase). 이 세 가지 closed-form이 DiffPIR의 NFE 효율의 핵심.
- **§3.3 (Schedule and hyperparameters)**: $T_{\text{start}}$를 SNR에 맞게, $\lambda, \zeta$ 작업별 튜닝. Table 4 참조.
- **§4 (Experiments)**: Table 1에서 NFE 컬럼 + LPIPS 컬럼 동시에 보기. DiffPIR이 *PSNR과 LPIPS 양쪽*에서 우월한 점이 기존 trade-off를 깨는 부분.
- **§5 (Discussion)**: DPS (paper #28)와의 관계 — 같은 동기, 다른 알고리즘. nonlinear case에서 두 방법이 거의 같은 gradient step으로 수렴.
- **Common stumbling blocks**: (1) HQS의 $\mu_k$ schedule이 어떻게 $\sigma_t$와 매칭되는지, (2) inpainting closed form의 mask-weighted 평균, (3) Wang 2008의 FFT inversion 유도 (DC 성분 처리).

**English**:
- **§2 Background**: master HQS and PnP basics — read Venkatakrishnan 2013 and Zhang 2017 IRCNN if unfamiliar.
- **§3 Method**: trace Algorithm 1 line-by-line; the *PnP iteration ↔ diffusion timestep* isomorphism is the central idea. Understand $\sigma_t^2 = \lambda\sigma_n^2/\mu_k$.
- **§3.2 Closed forms per forward model**: inpainting, deblurring (FFT), SR (polyphase). These three are the source of DiffPIR's NFE efficiency.
- **§3.3 Schedule**: choose $T_{\text{start}}$ from SNR; tune $\lambda, \zeta$ per task (Table 4).
- **§4 Experiments**: read NFE and LPIPS columns of Table 1 jointly; DiffPIR breaks the usual PSNR↔LPIPS trade-off.
- **§5 Discussion**: relation to DPS (paper #28) — same goal, different mechanism; in the nonlinear case both reduce to similar gradient updates.
- **Stumbling blocks**: (1) how the $\mu_k$ schedule matches $\sigma_t$, (2) inpainting's mask-weighted closed form, (3) deriving Wang's 2008 FFT inversion (handling DC).

---

## 7. 현대적 의의 / Modern Significance

**한국어**: DiffPIR은 *PnP의 모듈성 + diffusion의 generative quality* 를 결합한 깔끔한 통합이다. 새로운 forward model 추가 = data-fidelity prox 닫힌 형태만 유도; score 모델 재학습 불필요. 실용적으로 의료 영상(MRI 가속, low-dose CT), 산업 검사(motion-blur 제거), 천체관측(deconvolution + SR)에서 사전학습된 generative prior를 그대로 reuse하는 표준 방법이다. 본 reading list의 paper #28 (DPS)와는 *동일 동기, 직교 알고리즘 구조* — DPS는 ancestral DDPM에 gradient를 직접 더하고, DiffPIR은 HQS variable splitting으로 NFE 절감. 또한 paper #29 (Ambient Diffusion)이 *어떻게 corrupted-only 데이터로 score를 학습하는가* 를 다루었다면, DiffPIR은 *그 score를 어떻게 inverse problem 풀이에 사용하는가* 에 답한다 — 두 논문은 자연스럽게 합성 가능. 후속으로 latent-DiffPIR, conditional-diffusion hybridisation, blind kernel estimation 등이 활발하다. NTIRE/CVPR 같은 vision restoration 챌린지의 표준 baseline이 되었다.

**English**: DiffPIR is the clean unification of *PnP's modularity* with *diffusion's generative quality*. Adding a new forward model only requires deriving a data-fidelity prox; the score model is reused. In practice it is the standard route for reusing a pre-trained generative prior in medical imaging (accelerated MRI, low-dose CT), industrial inspection (motion deblur), and astronomical processing (deconvolution + SR). Within this reading list it is the *orthogonal twin* of DPS (paper #28) — same motivation, different algorithmic structure: DPS adds a gradient to the ancestral DDPM sampler, while DiffPIR exploits HQS variable splitting for fewer NFEs. Where Ambient Diffusion (paper #29) answers *how to obtain a score from corrupted-only data*, DiffPIR answers *how to use that score for inverse problem solving* — the two compose naturally. Active successors include latent-DiffPIR, conditional-diffusion hybridisation, and blind kernel estimation. It has become a standard baseline in vision restoration challenges (NTIRE, CVPR).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
