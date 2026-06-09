---
title: "Pre-Reading Briefing: Ambient Diffusion - Learning Clean Distributions from Corrupted Data"
paper_id: "29_daras_2023"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Ambient Diffusion: Learning Clean Distributions from Corrupted Data: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: G. Daras, K. Shah, Y. Dagan, A. Gollakota, A. Klivans, A. G. Dimakis, *NeurIPS* 2023, arXiv:2305.19256
**Author(s)**: Giannis Daras, Kulin Shah, Yuval Dagan, Aravind Gollakota, Adam Klivans, Alexandros G. Dimakis
**Year**: 2023

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 **깨끗한 학습 데이터 없이 손상된(corrupted) 표본만으로 확산 모델을 학습**하여 깨끗한 분포를 복원하는 첫 일반 프레임워크 **Ambient Diffusion** 을 제시한다. 핵심 트릭은 *추가 손상(further corruption)*: 이미 손상된 학습 입력 $\tilde{\boldsymbol x}_0 = \mathcal A_\phi(\boldsymbol x_0)$ 위에 **두 번째 더 강한 손상 마스크** $\tilde{\mathcal A}_\psi$ 를 한 번 더 적용하고, 모델이 이 이중 손상에서 *원래 한 번 손상된* $\tilde{\boldsymbol x}_0$ 를 복원하도록 학습. 이는 Noise2Noise (Lehtinen 2018)의 *generative 일반화*. **Theorem 1** 이 적절한 조건에서 ambient loss의 Bayes-optimal regressor가 $\mathbb E[\tilde{\boldsymbol x}_0 \mid \tilde{\tilde{\boldsymbol x}}_0]$이며, mask-conditioning과 함께 깨끗한 분포의 score를 정확히 복원함을 증명. CelebA에서 **90% 픽셀 결손** 데이터로도 oracle 대비 ~3.4 FID 차이로 분포 학습 성공. MRI fine-tuning에서는 환자 영상 *암기 없이* 분포 학습 — privacy-preserving generative modelling의 길.

### English
**Ambient Diffusion** is the first general framework for **training a diffusion model from only corrupted samples** while still learning the underlying clean distribution. The trick is *further corruption*: at training time apply a **second, stronger corruption** $\tilde{\mathcal A}_\psi$ on top of the already-corrupted sample $\tilde{\boldsymbol x}_0 = \mathcal A_\phi(\boldsymbol x_0)$, and ask the network to predict the **once-corrupted** target from the **twice-corrupted** input — the strict generalisation of Noise2Noise (Lehtinen 2018) to diffusion. **Theorem 1** proves that under technical conditions the Bayes-optimal regressor of the ambient loss equals $\mathbb E[\tilde{\boldsymbol x}_0 \mid \tilde{\tilde{\boldsymbol x}}_0]$, which together with mask-conditioning recovers the clean distribution's score function. With **90% missing pixels** at training time, FID is within ~3.4 of the clean-data oracle on CelebA. MRI fine-tuning avoids memorising patient images — privacy by construction.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**: 2018-2023년 동안 self-supervised denoising (Noise2Noise, Noise2Void, Noise2Self) 은 *회귀 모델*만 다루었지 분포 학습은 못 했다. AmbientGAN (Bora 2018)이 GAN 기반 분포 학습을 시도했지만 학습 불안정성이 컸다. 한편 2022-2023년에 Carlini, Somepalli, Jagielski 등이 *확산 모델이 학습 표본을 그대로 암기*함을 보여 의료/저작권 도메인에서 큰 우려가 제기되었다. Ambient Diffusion은 이 두 흐름을 동시에 해결한다: (i) 깨끗한 데이터가 비싸거나 존재하지 않는 도메인 (블랙홀 EHT, MRI, 천체관측, 저용량 의료) 에서 분포 학습 가능, (ii) 의도적 손상으로 *암기 회피*. 본 reading list의 paper #16 (Noise2Noise)이 회귀로 시작한 길을 generative modeling으로 자연스럽게 이어간다.

**English**: From 2018-2023 self-supervised denoising (Noise2Noise, Noise2Void, Noise2Self) addressed *regression* but not distribution learning. AmbientGAN (Bora 2018) attempted GAN-based distribution learning but suffered from training instability. Concurrently 2022-2023 work (Carlini, Somepalli, Jagielski) showed *diffusion models memorise training samples*, raising serious privacy concerns for medical and copyrighted data. Ambient Diffusion addresses both: (i) distribution learning where clean data is unavailable or expensive (black-hole EHT, MRI, astronomy, low-volume medical), and (ii) memorisation avoidance by design. It naturally extends the regression path of paper #16 (Noise2Noise) to generative modelling.

### 타임라인 / Timeline

```
1977 — EM algorithm (Dempster et al.) — incomplete-data MLE
2008 — Compressed sensing (Candès, Donoho)
2018 — AmbientGAN (Bora et al.) — GAN-based distribution from corruption
2018 — Noise2Noise (paper #16, regression-only)
2019 — Noise2Void / Noise2Self (single-noisy regression)
2020 — DDPM (Ho et al.)
2022 — Carlini et al. — diffusion memorises training data
2023 ★★ AMBIENT DIFFUSION (THIS PAPER)
2023 — SURE-Score (concurrent, Gaussian-noise variant)
2024 — Ambient-style training adopted for radio-astronomy / MRI
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**:
- **확산 모델 학습 손실**: $\mathbb E\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\boldsymbol x_t, t)\|^2$, 또는 동등한 $\boldsymbol x_0$-prediction 형태.
- **Tweedie 공식**: score ↔ posterior mean.
- **Noise2Noise 원리** (paper #16): conditionally independent noise pair → clean signal recovery.
- **Bayes-optimal regression**: squared loss minimizer = conditional expectation.
- **Inpainting / compressed sensing**: $\mathcal A_\phi(\boldsymbol x) = \boldsymbol M_\phi \odot \boldsymbol x$ 또는 $\boldsymbol G_\phi \boldsymbol x$.
- **Mask conditioning**: 모델 입력에 mask를 추가 채널로 주입.
- **Foundation model fine-tuning**: 사전학습 LDM 위에 도메인 적응.
- **Memorisation metrics**: nearest-neighbor distance from generated samples to training set.

**English**:
- **Diffusion training loss**: $\mathbb E\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\boldsymbol x_t, t)\|^2$, or equivalent $\boldsymbol x_0$-prediction.
- **Tweedie's formula**: score ↔ posterior mean.
- **Noise2Noise principle** (paper #16): conditionally independent noise pair recovers clean signal.
- **Bayes-optimal regression**: squared loss minimizer = conditional expectation.
- **Inpainting / compressed sensing**: $\mathcal A_\phi(\boldsymbol x) = \boldsymbol M_\phi \odot \boldsymbol x$ or $\boldsymbol G_\phi \boldsymbol x$.
- **Mask conditioning**: feeding the mask as an additional input channel.
- **Foundation model fine-tuning**: domain adaptation over pre-trained LDM.
- **Memorisation metrics**: nearest-neighbor distance from generated samples to the training set.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Ambient Diffusion | 손상된 데이터만으로 깨끗한 분포 학습하는 diffusion 프레임워크 / Diffusion training that learns the clean distribution from only corrupted samples. |
| Single corruption $\tilde{\boldsymbol x}_0 = \mathcal A_\phi(\boldsymbol x_0)$ | 학습 데이터 자체에 적용된 손상 (e.g., 90% 마스킹) / Corruption already present in the training set, e.g., 90% pixel masking. |
| Further corruption $\tilde{\mathcal A}_\psi$ | 학습 시 추가로 적용하는 더 강한 손상 / Additional, stronger corruption applied at training time. |
| Twice-corrupted input $\tilde{\tilde{\boldsymbol x}}_0$ | 두 번 손상된 모델 입력 / Doubly-corrupted input the network sees. |
| Noise2Noise generalisation | 두 noise instance로 clean signal 추정 → 분포 학습으로 일반화 / Generative analogue of regression-from-two-noisy-copies. |
| Mask conditioning | mask $\boldsymbol M_\psi$를 추가 입력 채널로 / Adding the mask as an extra input channel so the network knows which pixels to predict. |
| Bayes-optimal regressor | squared loss의 최적자 = conditional mean / Squared-loss minimiser equals the conditional expectation. |
| Loss equivalence (Theorem 1) | ambient 최적자 = $\mathbb E[\tilde{\boldsymbol x}_0\mid \tilde{\tilde{\boldsymbol x}}_0]$, mask 합쳐 깨끗한 score 복원 / The ambient optimum aggregates across masks to the clean score. |
| Full pixel support | mask family 전체에서 모든 픽셀이 적어도 가끔 관측 / Across the corruption distribution, every pixel is observed in some sample. |
| Memorisation | diffusion이 학습 표본을 그대로 재생산 (Carlini 2023) / Diffusion replicates training images verbatim — a privacy threat. |
| Privacy by construction | 깨끗한 표본을 본 적 없으므로 정확한 복제 불가능 / Never seeing clean training data prevents exact replication. |
| Compressed sensing variant | $\boldsymbol y = \boldsymbol G \boldsymbol x$, random Gaussian projection / The framework also applies to random Gaussian measurements (§4.4). |

---

## 5. 수식 미리보기 / Equations Preview

**핵심 1: 표준 확산 손실 / Standard diffusion loss**

$$
\mathcal L_{\text{std}}(\theta) = \mathbb E_{\boldsymbol x_0, t, \boldsymbol\epsilon}\Big[\big\|\boldsymbol x_0 - \hat{\boldsymbol x}_{0,\theta}(\boldsymbol x_t, t)\big\|_2^2\Big]
$$

**한국어**: Bayes-optimal solution은 $\hat{\boldsymbol x}_{0,\theta}^* = \mathbb E[\boldsymbol x_0 \mid \boldsymbol x_t]$. 깨끗한 $\boldsymbol x_0$ 표본 필요.

**English**: Bayes-optimal solution is the posterior mean — but requires clean samples $\boldsymbol x_0$.

**핵심 2: Ambient training loss / Ambient 학습 손실**

$$
\mathcal L_{\text{Amb}}(\theta) = \mathbb E_{\tilde{\boldsymbol x}_0, \psi, t, \boldsymbol\epsilon}\Big[\big\|\boldsymbol M_\psi \odot (\tilde{\boldsymbol x}_0 - \hat{\boldsymbol x}_{0,\theta}(\boldsymbol x_t, t, \psi))\big\|_2^2\Big]
$$

where $\boldsymbol x_t = \sqrt{\bar\alpha_t}\,\tilde{\mathcal A}_\psi(\tilde{\boldsymbol x}_0) + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$.

**한국어**: 입력은 이중 손상의 noisy 버전, target은 단일 손상 $\tilde{\boldsymbol x}_0$, 손실은 $\boldsymbol M_\psi$가 *제거한* 영역에서만 측정.

**English**: Input is the noisy doubly-corrupted image; target is the once-corrupted $\tilde{\boldsymbol x}_0$; loss is computed only on pixels removed by $\boldsymbol M_\psi$.

**핵심 3: Loss equivalence (Theorem 1, schematic)**

$$
\hat{\boldsymbol x}_{0,\theta}^*(\boldsymbol x_t, t, \psi) = \mathbb E[\tilde{\boldsymbol x}_0 \mid \boldsymbol x_t, t, \psi]
$$

**한국어**: ambient 손실의 최적자는 단일 손상의 conditional expectation. mask family의 full-pixel support 하에서 marginalise하면 깨끗한 분포의 score를 복원.

**English**: The ambient optimum is the conditional mean of the once-corrupted target; aggregating across the mask family (under full-pixel support) recovers the clean distribution's score.

**핵심 4: Tweedie score recovery / 스코어 복원**

$$
\nabla_{\boldsymbol x_t}\log p_t(\boldsymbol x_t) = \frac{\sqrt{\bar\alpha_t}\,\hat{\boldsymbol x}_{0,\theta}^* - \boldsymbol x_t}{1 - \bar\alpha_t}
$$

**한국어**: 학습된 ambient predictor로부터 깨끗한 분포의 score를 표준 Tweedie 식으로 추출.

**English**: Standard Tweedie identity applied to the learned ambient predictor recovers the clean-distribution score.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**:
- **§1 (Introduction)**: 두 가지 동기 — 깨끗한 데이터 부재 vs. memorisation 회피 — 를 명확히 구분.
- **§3 (Method)**: Algorithm 1을 한 줄씩 따라가며 *추가 손상 $\psi$가 왜 학습 가능하게 만드는지*를 직관적으로 이해. mask-conditioning이 입력에 어떻게 들어가는지 코드 수준에서 시각화.
- **§3.2 (Theorem 1)**: 증명의 두 단계 — Bayes-optimal regression + sufficient-statistic argument. 1-D Bernoulli 마스킹 toy example (notes §4.6)을 paper-and-pencil로 따라가 볼 것.
- **§4 (Experiments)**: Table 1의 $p=0.1$ (90% missing) 결과가 핵심. AmbientGAN과의 격차 (FID 36.4 → 8.85)를 음미. MRI fine-tuning과 NN 거리 분석 부분이 가장 인상적.
- **§4.4 (Compressed sensing)**: 마스킹 외에 random Gaussian projection도 가능 — 적용 범위가 넓음.
- **Common stumbling blocks**: (1) "더 망가뜨려서 학습한다"는 역설적 디자인의 직관 (mask 영역만이 학습 신호 정의), (2) full-pixel support 조건의 의미 (어떤 픽셀이 항상 가려지면 학습 불가), (3) memorisation NN 거리 분석의 해석.

**English**:
- **§1 Introduction**: distinguish the two motivations — clean-data unavailability vs. memorisation avoidance.
- **§3 Method**: trace Algorithm 1 line-by-line; build intuition for *why* further corruption $\psi$ creates the learning signal. Visualise how mask-conditioning enters the network input.
- **§3.2 Theorem 1**: two-step proof — Bayes-optimal regression + sufficiency argument. Walk through the 1-D Bernoulli toy (notes §4.6) by hand.
- **§4 Experiments**: the $p=0.1$ (90% missing) row of Table 1 is the headline. Note the AmbientGAN gap (FID 36.4 → 8.85). The MRI fine-tuning + NN-distance experiment is the most striking.
- **§4.4 Compressed sensing**: random Gaussian projections also work — the framework is broad.
- **Stumbling blocks**: (1) the counter-intuitive design "make it worse to learn" (the loss is defined only on $\psi$-removed pixels), (2) what full-pixel support means (always-masked pixels can't be learned), (3) interpreting NN-distance results for memorisation.

---

## 7. 현대적 의의 / Modern Significance

**한국어**: Ambient Diffusion은 *self-supervised generative modelling의 시발점*이다. 본 reading list의 paper #16 (Noise2Noise)이 연 회귀 길을 분포 학습으로 이어가며, 의료/천체/마이크로스코피 영상의 데이터 가용성 문제와 모델 메모리화 문제를 동시에 해결한다. 후속 연구로는 (a) SURE-Score (Aali 2023, 가우시안 noise 변형), (b) nonlinear corruption 일반화, (c) 차분 프라이버시(DP) 결합 등이 활발하다. DiffPIR (paper #30)이 사용하는 generative prior를 *어떻게 corrupted-only 데이터로 얻을 것인가*에 대한 직접적 답이며, DPS (paper #28)와는 *학습 시 corruption 처리 vs. 추론 시 corruption 역전*의 두 직교 축을 형성 — 두 방법은 합성 가능. EHT 블랙홀 영상, 저용량 의료 영상, 환자 정보 보호가 필요한 도메인에서 제너러티브 모델을 사용하는 *유일한 방법*이라 할 수 있다.

**English**: Ambient Diffusion launches *self-supervised generative modelling*. It carries the regression path opened by paper #16 (Noise2Noise) into distribution learning, simultaneously addressing the data-availability problem in medical/astronomical/microscopy imaging and the memorisation problem in modern diffusion models. Active successors include (a) SURE-Score (Aali 2023, Gaussian-noise variant), (b) nonlinear-corruption generalisations, (c) differential-privacy combinations. It directly answers *how to obtain* the generative prior used by DiffPIR (paper #30) when only corrupted data is available, and forms — together with DPS (paper #28) — two orthogonal axes (corruption at training time vs. corruption at inference time) that compose. For EHT black-hole imaging, low-volume medical imaging, and any domain demanding patient-data privacy, it is essentially the *only* viable route to generative modelling.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
