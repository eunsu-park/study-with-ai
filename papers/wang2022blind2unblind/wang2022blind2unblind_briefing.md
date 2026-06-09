---
title: "Pre-Reading Briefing: Blind2Unblind"
paper_id: "22_wang_2022"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots / 사전 읽기 브리핑

**Paper**: Wang, Z., Liu, J., Li, G., Han, H. "Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots". *IEEE/CVF CVPR 2022*, pp. 2027–2036. DOI: 10.1109/CVPR52688.2022.00207.
**Authors**: Zejin Wang, Jiazheng Liu, Guoqing Li, Hua Han
**Year**: 2022

---

## 1. 핵심 기여 / Core Contribution

### 한국어
Blind2Unblind(B2U)는 **Noise2Void류 blind-spot 디노이저의 정보 손실 문제**를 정면으로 해결한다. N2V는 중심 픽셀을 가린 패치로 학습하므로 blind-spot 픽셀의 정보가 영원히 사용되지 못해 성능 상한이 낮았다. B2U는 **re-visible loss**를 도입한다: $\mathcal L_{\rm rev}=\|h(f_\theta(\Omega_y)) + \lambda \hat f_\theta(y) - (\lambda+1) y\|^2$, 여기서 $h$는 blind-spot 위치만 모아 한 평면으로 사상하는 **global-aware mask mapper**이고 $\hat f_\theta(y)$는 *gradient-stop* non-blind 분기다. Stationary point는 두 분기의 가중 평균 $\tilde x = (h(f_\theta^*) + \lambda \hat f_\theta^*)/(\lambda+1)$. 합성 Gaussian/Poisson, 실세계 SIDD raw-RGB, 형광현미경 FMD 모두에서 R2R(#21)·NBR2NBR(#20)·Self2Self(#19)를 능가하며, 일부 벤치마크에서는 *supervised N2C도 능가*한다.

### English
Blind2Unblind (B2U) directly attacks the **information-loss bottleneck of N2V-style blind-spot denoisers**. By design, N2V hides the centre pixel of every input, so blind-spot information is never used during training, capping performance below clean-target supervision. B2U introduces a **re-visible loss**: $\mathcal L_{\rm rev}=\|h(f_\theta(\Omega_y)) + \lambda \hat f_\theta(y) - (\lambda+1)y\|^2$, where $h$ is a **global-aware mask mapper** that gathers denoised values only at blind-spot positions onto one plane, and $\hat f_\theta(y)$ is a *stop-gradient* non-blind branch. The stationary point is a weighted average $\tilde x = (h(f_\theta^*) + \lambda\hat f_\theta^*)/(\lambda+1)$. On synthetic Gaussian/Poisson, real raw-RGB SIDD, and fluorescence microscopy FMD, B2U beats R2R (#21), NBR2NBR (#20), and Self2Self (#19) — and on several benchmarks even surpasses supervised N2C.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
2018-2021 self-supervised denoising 경쟁은 두 흐름으로 굳어졌다: (a) blind-spot 계열 (N2V, N2S, Laine'19, Self2Self) — 정보 손실의 원천적 한계, (b) data/noise injection 계열 (NBR2NBR, R2R) — blind spot은 피하지만 각자의 가정 (spatial smoothness 또는 known noise). 2022년에는 "blind-spot은 끝났다"라는 공감대가 형성되어 있었다. B2U는 정확히 이 통념을 반박한다: blind spot은 *학습 안정화 도구*이지 *정보 가림막*일 필요가 없다. re-visible loss로 가린 정보를 회복하면 supervised까지 능가할 수 있다.

#### English
By 2021 self-supervised denoising had split into two streams: (a) blind-spot methods (N2V, N2S, Laine'19, Self2Self) suffering inherent information loss, and (b) data/noise-injection methods (NBR2NBR, R2R) that bypass blind spots at the cost of additional assumptions (spatial smoothness or known noise). The community consensus was that blind-spot methods had hit a wall. B2U overturns this view by reframing the blind spot as a *training-stabilisation device* rather than an *information mask*: with the right loss, the lost information can be recovered to surpass even supervised baselines.

### 타임라인 / Timeline

```
2018  Lehtinen N2N (#16) — two noisy captures
2019  Krull N2V (#17) — single-image, blind-spot
2019  Batson N2S (#18) — J-invariance theory
2019  Laine et al. — masked-conv blind-spot network
2020  Quan Self2Self (#19) — dropout ensemble
2021  Pang R2R (#21) — covariance-matched recorruption
2021  Huang Neighbor2Neighbor (#20) — sub-sampling pair
2022 ★ Wang Blind2Unblind — re-visible loss + global mask mapper, beats supervised
2023+ AP-BSN, LG-BPN, PUCA, LAN — asymmetric pixel-shuffle BSN family
```

---

## 3. 필요한 배경 지식 / Prerequisites

#### 한국어
- **N2V blind-spot 메커니즘** (논문 #17) — B2U가 직접 재구축하는 baseline
- **U-Net 아키텍처**, encoder-decoder, skip connection
- **Masked convolution** (Laine 2019) — global mask mapper의 전신
- **Stop-gradient (sg)** 연산 — non-blind 분기 정규화 방식
- **Multi-task loss** 구성 (blind + non-blind term)
- **$\lambda$ annealing schedule** ($\lambda_s\to\lambda_f$)
- **Stationary-point 분석** — convex combination minimiser
- **Receptive field** 개념과 blind-spot의 정보 손실
- **이전 self-sup 논문 맥락**: N2V(#17), N2S(#18), S2S(#19), NBR2NBR(#20), R2R(#21)

#### English
- N2V's blind-spot mechanism (paper #17) — the baseline B2U rebuilds.
- U-Net architecture, encoder-decoder, skip connections.
- Masked convolution (Laine 2019) — predecessor of the global mask mapper.
- Stop-gradient operations for branch regularisation.
- Multi-task loss design (blind + non-blind terms).
- $\lambda$ annealing schedules ($\lambda_s\to\lambda_f$).
- Stationary-point analysis: convex combination as minimiser.
- Receptive field concept and how blind spots lose information.
- Prior self-sup methods: N2V (#17), N2S (#18), S2S (#19), NBR2NBR (#20), R2R (#21).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Blind spot** | N2V에서 가려지는 중심 픽셀 / Masked centre pixel in N2V. |
| **Visible blind spot** | 가려진 픽셀의 정보를 *복원해 사용* / Recovering and using the blind-spot information. |
| **Masked volume $\Omega_y$** | 4가지 마스크의 stacked $4\times W\times H$ 볼륨 / Stack of 4 masked images, one per (i,j) cell offset. |
| **Mask mapper $h(\cdot)$** | blind-spot 위치별 denoised 값을 한 평면에 모음 / Gathers per-pixel denoised values from masked volume to single plane. |
| **Non-blind branch $\hat f_\theta(y)$** | gradient-stop 적용된 비-blind 출력 / Stop-gradient branch on unmasked input. |
| **Re-visible loss $\mathcal L_{\rm rev}$** | blind + non-blind 결합 손실 (Eq. 8) / Combined blind + non-blind loss. |
| **Convex combination minimiser** | $\tilde x = (h(f_\theta^*) + \lambda \hat f_\theta^*)/(\lambda+1)$ / Stationary point is weighted average of branches. |
| **$\lambda$ annealing** | $\lambda_s=2 \to \lambda_f=20$ 점진 증가 / Annealed visible-weight ($\lambda_s=2$ → $\lambda_f=20$). |
| **Regulariser $\eta\mathcal L_{\rm reg}$** | $\eta\|h(f_\theta(\Omega_y))-y\|^2$, 학습 안정화 / Stabilising regulariser keeping blind term in check. |
| **Cell size $s=2$** | 2×2 셀 → 4 마스크 / 2×2 cell → 4 masks (i,j ∈ {0,1}). |
| **Single-forward inference** | $\hat x = f_\theta(y)$, MC 평균 불필요 / Single forward pass; no Monte-Carlo averaging needed. |
| **N2C (Noise-to-Clean)** | supervised baseline (clean GT 사용) / Supervised baseline with clean ground truth. |

---

## 5. 수식 미리보기 / Equations Preview

### 한국어
**N2V baseline (Eq. 1)** — 중심 픽셀 가림 손실:

$$
\mathcal L_{\rm N2V} = \mathbb E_y \|f_\theta(y_{\mathrm{RF}(i)}) - y_i\|_2^2
$$

**Re-visible loss (Eq. 8)** — blind + non-blind 결합:

$$
\mathcal L_{\rm rev} = \mathbb E_y\big\|h(f_\theta(\Omega_y)) + \lambda\,\hat f_\theta(y) - (\lambda+1)\,y\big\|_2^2
$$

(여기서 $\hat f_\theta(y)$는 stop-gradient: $\nabla_\theta \hat f_\theta = 0$.)

**Stationary point (Eq. 9)** — convex 평균:

$$
\tilde x = \frac{h(f_\theta^*(\Omega_y)) + \lambda\,\hat f_\theta^*(y)}{\lambda+1}
$$

**전체 손실 (Eq. 12)** — re-visible + regulariser:

$$
\mathcal L = \mathcal L_{\rm rev} + \eta\,\|h(f_\theta(\Omega_y)) - y\|_2^2
$$

### English
N2V's baseline minimises MSE between the network's blind-spot prediction and the actual noisy centre pixel. The re-visible loss (Eq. 8) replaces this single term with a *combined* objective — its stationary-point analysis shows the optimal denoised image is the *weighted average* of the blind branch $h(f_\theta(\Omega_y))$ and the non-blind branch $\hat f_\theta(y)$. The non-blind branch's stop-gradient prevents direct training on it (which would collapse to identity) but still pulls the optimum toward a richer estimate. The regulariser $\eta\,\|h(f_\theta(\Omega_y))-y\|^2$ stabilises the blind branch during early training when annealed $\lambda$ is small.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **꼭 정독할 부분**: §3 (re-visible loss 유도, Eqs. 1-8 + Eq. 9 stationary point), §4 (global-aware mask mapper의 구조), §5 ablation (특히 Random Mask vs Global Mask 차이, $\lambda$ annealing, $\eta$ ablation).
- **빠르게 훑을 부분**: §2 related work, §6.2 reproducibility 세부.
- **흔한 걸림돌 / Common stumbling blocks**:
  - "stop-gradient 분기가 어떻게 학습에 기여하는가?" — gradient는 blind 분기에서만 오지만, stationary point의 *형태*에 직접 등장 → 학습 후 출력이 두 분기 평균으로 수렴.
  - "왜 random mask는 수렴 실패하고 global mask만 성공하는가?" — global mask는 모든 blind-spot이 한 평면으로 집계되어 gradient가 일관됨; random mask는 cell마다 다른 위치라 gradient 충돌.
  - "$\lambda$ annealing의 역할": 초기에는 blind 항이 학습을 끌고, 학습이 안정된 후 visible 항 비중 증가 → curriculum 학습.
  - "convex combination이 왜 supervised를 능가하는가?": single-image supervised는 noisy *target*에 fit해 noise leakage; B2U는 blind/non-blind 평균이 noise leakage를 막음.
- 동반 자료: Laine 2019 masked-conv 논문, Eq. 8 유도(Eqs. 3-7).

### English
- **Read carefully**: §3 (re-visible loss derivation, Eqs. 1–8 plus Eq. 9 stationary-point analysis), §4 (global-aware mask mapper architecture), §5 ablations on Random vs Global Mask, $\lambda$ annealing, and $\eta$.
- **Skim**: §2 related work, §6.2 reproducibility appendix.
- **Common stumbling blocks**:
  - How can the stop-gradient branch contribute to learning? Gradient flows only through the blind branch, but the *form* of the stationary point includes both — so the network converges to their convex average.
  - Why Random Mask + visible loss diverges while Global Mask + visible converges — the global mapper aggregates all blind-spot pixels onto a coherent plane, keeping gradients consistent.
  - The curriculum role of $\lambda$ annealing: blind term drives early learning ($\lambda_s=2$); the visible term takes over once the network stabilises ($\lambda_f=20$).
  - Why B2U can beat supervised — single-image supervised fits noisy *targets* directly, leaking noise; B2U's blind/non-blind average suppresses this leakage.
- Companion reading: Laine et al. (2019) masked-conv paper; Eqs. 3–7 derivation chain.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
B2U는 **2022년 self-supervised denoising의 SOTA**이자 "blind-spot은 끝났다"라는 통념을 정면으로 반박한 작품이다. 이후 AP-BSN, LG-BPN, PUCA, LAN 같은 asymmetric pixel-shuffle BSN family가 모두 B2U의 *visible blind spot* 아이디어를 발전시킨다. Re-visible loss + global mapper 패턴은 self-supervised inpainting, super-resolution 등 다른 시각 작업으로 확장되고 있다. B2U의 핵심 통찰 — *training stability와 information utilisation은 별개의 axis* — 는 일반 self-supervised learning 설계 원리로 받아들여지고 있다. R2R(#21)은 noise model을 가정하고 NBR2NBR(#20)은 spatial smoothness를 가정하는 데 비해, B2U는 *둘 다* 가정하지 않으면서도 두 방법을 모두 능가한다 — 현재까지 가장 가정-가벼운 SOTA 방법 중 하나다.

### English
B2U represents the SOTA of self-supervised denoising in 2022 and decisively rebuts the "blind spots have hit a wall" narrative. Subsequent work — AP-BSN, LG-BPN, PUCA, LAN, the asymmetric pixel-shuffle BSN family — extends B2U's *visible blind spot* idea. The re-visible-loss + global-mapper pattern has been ported to self-supervised inpainting and super-resolution. B2U's core insight — *training stability and information utilisation are separable axes* — has been adopted as a general principle in self-supervised learning. Whereas R2R (#21) needs a noise model and NBR2NBR (#20) needs spatial smoothness, B2U requires *neither* yet beats both — making it among the most assumption-light SOTA methods to date.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
