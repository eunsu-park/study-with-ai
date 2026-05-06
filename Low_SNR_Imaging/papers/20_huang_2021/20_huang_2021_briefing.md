---
title: "Pre-Reading Briefing: Neighbor2Neighbor"
paper_id: "20_huang_2021"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images / 사전 읽기 브리핑

**Paper**: Huang, T., Li, S., Jia, X., Lu, H., Liu, J. "Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images". *IEEE/CVF CVPR 2021*, pp. 14781–14790. DOI: 10.1109/CVPR46437.2021.01454.
**Authors**: Tao Huang, Songjiang Li, Xu Jia, Huchuan Lu, Jianzhuang Liu
**Year**: 2021

---

## 1. 핵심 기여 / Core Contribution

### 한국어
Neighbor2Neighbor(Nb2Nb)는 **단일 잡음 영상에서 random neighbour sub-sampling으로 pseudo-noisy pair를 생성**하고 Noise2Noise 손실 + regulariser로 학습하는 framework다. N2V/N2S/Self2Self는 모두 *네트워크 구조나 stochastic 추론*에 self-supervision constraint를 부과했지만, Nb2Nb는 *입력 데이터 단계*에서 두 noisy view를 만들어 표준 supervised pipeline을 그대로 쓴다. 핵심은 (i) $2\times 2$ 셀에서 인접 두 픽셀을 무작위로 골라 두 sub-image $g_1(\mathbf y), g_2(\mathbf y)$ 생성, (ii) Theorem 1로 sub-sampled 신호 사이의 *non-zero gap* $\boldsymbol\varepsilon$이 만드는 over-smoothing bias를 정량화하고 *consistency regulariser*로 보정, (iii) 추론은 단일 forward pass. SIDD Benchmark에서 50.76 dB로 supervised N2C(50.60)도 능가했다.

### English
Neighbor2Neighbor (Nb2Nb) builds a *pseudo-noisy pair* from a single noisy image via *random neighbour sub-sampling*, then trains with a Noise2Noise-style loss plus a consistency regulariser. Whereas N2V, N2S, and Self2Self impose self-supervision constraints on the network architecture or stochastic inference, Nb2Nb shifts the burden to *data construction*: any standard denoising network (UNet, DnCNN, RRG) can be used unchanged. Three pillars: (i) partition the image into $2\times 2$ cells, randomly pick two adjacent pixels per cell to form two sub-images $g_1(\mathbf y), g_2(\mathbf y)$; (ii) Theorem 1 quantifies the over-smoothing bias caused by the non-zero signal gap $\boldsymbol\varepsilon$ between sub-images and is cancelled by a consistency regulariser; (iii) inference is a single forward pass on the full image. On SIDD Benchmark Nb2Nb (RRGs) achieves 50.76 dB, beating even supervised N2C (50.60).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
2018-2020년 self-supervised denoising에서 N2V/N2S(blind-spot), S2S(dropout ensemble), Laine'19(probabilistic post-processing), Noisier2Noise/R2R(known noise model)이 각자 다른 가정을 도입했다. 그러나 (a) blind-spot은 정보 손실, (b) dropout ensemble은 추론 50회, (c) 명시적 noise model은 real-world에서 fragile이라는 한계가 있었다. Nb2Nb는 *자연 영상의 spatial smoothness*만 가정해서 노이즈 모델 무관·아키텍처 무관·1회 추론을 동시에 달성한다.

#### English
By 2018-2020 self-supervised denoising had branched into N2V/N2S (blind-spot), Self2Self (dropout ensemble), Laine'19 (probabilistic post-processing), and Noisier2Noise/R2R (known noise model). Each approach traded off in a different direction: blind spots lose information, dropout ensembles need 50 forward passes, explicit noise priors are brittle on real cameras. Nb2Nb's contribution is to assume only *spatial smoothness of natural images* — making it noise-model-free, architecture-agnostic, and single-forward at inference.

### 타임라인 / Timeline

```
2005  Buades NLM (#4) — patch-level spatial self-similarity prior
2018  Lehtinen N2N (#16) — noisy/noisy training
2019  Krull N2V (#17) / Batson N2S (#18) — blind-spot single-image
2019  Laine et al. — probabilistic post-processing (needs noise prior)
2020  Quan Self2Self (#19) — dropout ensemble, beats BM3D
2020  Moran Noisier2Noise / Pang R2R — synthesised pairs (known noise)
2021 ★ Huang Neighbor2Neighbor — neighbour sub-sampling, NOISE-MODEL-FREE
2022  Wang Blind2Unblind (#22) — visible blind spots, current SOTA
```

---

## 3. 필요한 배경 지식 / Prerequisites

#### 한국어
- **Noise2Noise 정리** (Lehtinen 2018, 논문 #16) — 두 noisy 영상이면 clean 없이 학습 가능
- **단일 영상 self-sup의 동기**: N2V/N2S/S2S 핵심 (#17, #18, #19)
- **Conditional independence**: 두 픽셀이 동일 신호 조건 하에서 독립 잡음을 가짐
- **PSNR / SSIM** 평가 지표
- **UNet, RRG (Recursive Residual Group)** 백본
- **자연 영상의 spatial smoothness** — 인접 픽셀이 신호적으로 거의 같다는 가정
- **Stop-gradient / no_grad**: backprop 차단을 통한 학습 안정화
- **SIDD 데이터셋** (real-world raw-RGB smartphone noise)

#### English
- The Noise2Noise theorem (Lehtinen 2018, paper #16): two noisy realisations suffice for clean-target-free training.
- Single-image self-supervision motivation: N2V (#17), N2S (#18), Self2Self (#19).
- Conditional independence: two pixels share clean signal but have independent noise.
- PSNR / SSIM evaluation metrics.
- UNet and RRG (Recursive Residual Group) backbones.
- Spatial smoothness of natural images — neighbouring pixels share the signal almost exactly.
- Stop-gradient / no_grad operations for training stability.
- SIDD real-world smartphone-noise benchmark.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Random neighbour sub-sampler $G=(g_1,g_2)$** | 2×2 셀에서 인접 두 픽셀을 무작위 선택 / Randomly pick two adjacent pixels per 2×2 cell. |
| **Pseudo-noisy pair** | sub-sampled 두 sub-image 쌍 — N2N 학습 가능한 입력/타겟 / A sub-sampled pair acting as N2N training pair. |
| **Signal gap $\boldsymbol\varepsilon$** | $\mathbb E[g_2(\mathbf y)|\mathbf x]-\mathbb E[g_1(\mathbf y)|\mathbf x]$ — 인접 픽셀의 신호 차이 / The non-zero clean-signal difference between sub-sampled pixels. |
| **Reconstruction loss $\mathcal L_{\text{rec}}$** | $\|f_\theta(g_1(\mathbf y))-g_2(\mathbf y)\|^2$, 표준 N2N 손실 / Standard N2N loss on the sub-sampled pair. |
| **Regulariser $\mathcal L_{\text{reg}}$** | $\boldsymbol\varepsilon$이 만드는 over-smoothing 보정항 / Term cancelling the over-smoothing bias from $\boldsymbol\varepsilon$. |
| **Commutativity constraint** | 이상 denoiser는 sub-sampling과 commutative / Optimal denoiser commutes with sub-sampling operator. |
| **Stop-gradient** | $g_1(f_\theta(\mathbf y))$ 계산 시 gradient 차단 / Cut gradient flow in regulariser branch. |
| **$\gamma$ trade-off** | smoothness ↔ noisiness 균형 (synth $\gamma=2$, real $\gamma=1$) / Smoothness-vs-noisiness trade-off weight. |
| **SIDD Benchmark** | real raw-RGB smartphone 잡음 평가 / Real-world raw-RGB benchmark for camera noise. |
| **Theorem 1** | non-zero $\boldsymbol\varepsilon$에서 N2N 손실 분해 / N2N-loss decomposition under non-zero $\boldsymbol\varepsilon$. |
| **Architecture-agnostic** | 어떤 denoiser 백본이든 가능 / Backbone-agnostic — any denoising CNN works. |
| **Single forward inference** | 추론 시 한 번의 forward pass로 충분 / Inference uses a single forward pass on the full image. |

---

## 5. 수식 미리보기 / Equations Preview

### 한국어
**Theorem 1 (Eq. 2)** — non-zero gap $\boldsymbol\varepsilon$에서 N2N 손실 분해:

$$
\mathbb E\|f_\theta(\mathbf y) - \mathbf z\|^2 = \mathbb E\|f_\theta(\mathbf y) - \mathbf x\|^2 - \sigma^2_{\mathbf z} + 2\,\mathbb E\langle f_\theta(\mathbf y) - \mathbf x, \boldsymbol\varepsilon\rangle
$$

**최적 denoiser 제약 (Eq. 5)** — sub-sampling과 commutativity:

$$
g_1(\mathbf y) - g_2(\mathbf y) - \big(g_1(f^*_\theta(\mathbf y)) - g_2(f^*_\theta(\mathbf y))\big) = 0
$$

**전체 손실 (Eq. 7)** — reconstruction + regulariser:

$$
\mathcal L = \underbrace{\|f_\theta(g_1(\mathbf y)) - g_2(\mathbf y)\|^2_2}_{\mathcal L_{\text{rec}}} + \gamma \underbrace{\|f_\theta(g_1(\mathbf y)) - g_2(\mathbf y) - (g_1(f_\theta(\mathbf y)) - g_2(f_\theta(\mathbf y)))\|^2_2}_{\mathcal L_{\text{reg}}}
$$

### English
Theorem 1 says: training on noisy-noisy pairs with a non-zero conditional bias $\boldsymbol\varepsilon$ deviates from the supervised loss by a cross term $2\langle f-\mathbf x,\boldsymbol\varepsilon\rangle$, which causes the network to over-smooth toward $\mathbf x+\boldsymbol\varepsilon/2$. The optimal-denoiser constraint shows that a perfect denoiser commutes (in expectation) with the sub-sampling operator, motivating the consistency regulariser. The total loss combines a standard N2N reconstruction term with this regulariser weighted by $\gamma$ (typically 1-2).

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **꼭 정독할 부분**: §3.2-3.3 (Theorem 1과 sub-sampling으로의 확장), §4.1 (sub-sampler 알고리즘), §5.4 ablation (특히 $\gamma$ 민감도와 SIDD 결과).
- **빠르게 훑을 부분**: §2 related work, §5.1 implementation 세부.
- **흔한 걸림돌 / Common stumbling blocks**:
  - Theorem 1의 cross term 해석: $2\langle f-\mathbf x,\boldsymbol\varepsilon\rangle$이 왜 over-smoothing을 일으키는가 — minimiser가 $\mathbf x+\boldsymbol\varepsilon/2$로 편향.
  - Regulariser의 *commutativity 의미*: 입력 측 sub-sample 차이 = 출력 측 sub-sample 차이.
  - Stop-gradient의 역할: $g_1(f_\theta(\mathbf y))$, $g_2(f_\theta(\mathbf y))$ 분기에서 gradient를 끊는 이유 (학습 안정화).
- 동반 자료: Lehtinen Noise2Noise 논문, SIDD 데이터셋 paper, $\gamma$ ablation Table 3.

### English
- **Read carefully**: §3.2–3.3 (Theorem 1 and extension to single-image sub-sampling), §4.1 (sub-sampler algorithm), §5.4 ablations on $\gamma$ and the SIDD result.
- **Skim**: §2 related work, §5.1 implementation details.
- **Common stumbling blocks**:
  - The cross term in Theorem 1 — why $2\langle f-\mathbf x,\boldsymbol\varepsilon\rangle$ causes the network's optimum to drift to $\mathbf x+\boldsymbol\varepsilon/2$.
  - The geometric meaning of $\mathcal L_{\text{reg}}$: input-side sub-sample difference must equal output-side sub-sample difference.
  - Why gradients are stopped on $g_1(f_\theta(\mathbf y))$, $g_2(f_\theta(\mathbf y))$ — needed for training stability.
- Companion reading: original Noise2Noise paper, SIDD dataset paper, Table 3 $\gamma$ ablation.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
Nb2Nb는 self-supervised denoising 패러다임을 *데이터 측 self-supervision*으로 옮긴 분기점이다. 이후 거의 모든 자기지도 디노이저 — Blind2Unblind(#22), AP-BSN, LG-BPN, PUCA — 가 Nb2Nb를 baseline으로 인용한다. 특히 *noise model 가정 없음*이 real-world 적용에서 결정적 장점이 되어 fluorescence microscopy, low-light photography, autonomous driving 영상 등 noise distribution이 잘 알려지지 않은 도메인에서 채택된다. 또한 SIDD에서 supervised를 능가한 사실은 "self-supervised가 supervised보다 항상 나쁘다"는 통념을 깨뜨렸으며 — supervised pair의 *imperfect* GT(motion, ISP 차이)를 회피하는 self-sup의 이점을 입증했다. CVPR 2021 시점의 정점이며 이후 SOTA 경쟁의 기준이 되었다.

### English
Nb2Nb shifted the self-supervised denoising paradigm to *data-side self-supervision*. Almost every subsequent self-supervised denoiser — Blind2Unblind (#22), AP-BSN, LG-BPN, PUCA — cites Nb2Nb as a baseline. Its *noise-model-free* nature has been decisive for real-world adoption in fluorescence microscopy, low-light photography, and autonomous-vehicle imaging, where the noise distribution is unknown. Beating supervised N2C on SIDD also dispelled the notion that self-supervision is always inferior — it showed that self-sup can sidestep the *imperfect* ground-truth pairs (motion, ISP differences) that plague real-world supervised training. Nb2Nb represented the SOTA inflection point at CVPR 2021 and remains the standard comparison baseline.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
