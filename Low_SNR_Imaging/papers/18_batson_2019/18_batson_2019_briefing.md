---
title: "Pre-Reading Briefing: Noise2Self: Blind Denoising by Self-Supervision"
paper_id: "18_batson_2019"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Noise2Self: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Batson, J., & Royer, L., "Noise2Self: Blind Denoising by Self-Supervision", *Proc. 36th ICML, PMLR 97*, pp. 524–533 (2019). arXiv:1901.11365.
**Author(s)**: Joshua Batson, Loïc Royer
**Year**: 2019

---

## 1. 핵심 기여 / Core Contribution

### 한국어
Noise2Self(N2S)는 **잡음 모델·신호 사전·클린 GT 없이** 단일 잡음 측정으로부터 denoiser를 학습할 수 있는 *일반 프레임워크*다. 핵심 가정은 단 하나 — 측정 차원 $\{1,\dots,m\}$을 분할 $\mathcal J = \{J_1, J_2, \dots\}$로 나눴을 때 **잡음이 분할 사이에서 조건부 독립**이라는 것. 본 논문의 결정적 기여 네 가지:

(i) **J-invariance 정의**: 함수 $f$가 임의 $J \in \mathcal J$에 대해 $f(x)_J$가 입력 $x_J$에 의존하지 않으면 J-invariant — 즉 *자기 자신을 보지 않고 자기 자신을 예측*.

(ii) **자기지도 손실 정리(Proposition 1)**: $x$가 unbiased ($\mathbb E[x|y] = y$)이고 잡음이 $J$ 사이 조건부 독립이면, J-invariant $f$ 에 대해 $\mathbb E\|f(x) - x\|^2 = \mathbb E\|f(x) - y\|^2 + \mathbb E\|x - y\|^2$. 즉 self-supervised loss = supervised loss + noise variance(상수). **클린 GT 없이 진짜 MSE를 최소화**.

(iii) **고전 denoiser의 J-invariant 변형(Donut filter)**: 모든 매개 denoiser $g_\theta$ (median, wavelet thresholding, NLM)에 *donut* 변형 정의 — 마스크 위치를 이웃 평균으로 채우고 $g_\theta$ 적용. self-supervised loss 곡선 최소가 GT loss 최소와 *일치*(Fig. 2).

(iv) **딥러닝 적용**: DnCNN/UNet에 4×4 그리드 마스킹으로 J-invariantisation → Hànzì, ImageNet, CellNet에서 NLM·BM3D 능가, N2N/N2T에 근접(Table 2). *Single-cell RNA-seq* 으로도 일반화 — modality-agnostic 프레임워크.

### English
Noise2Self (N2S) is a **general framework for blind denoising** requiring no noise model, no signal prior, and no clean targets. Its single assumption: the measurement dimensions $\{1,\dots,m\}$ admit a partition $\mathcal J$ such that **noise is conditionally independent across blocks given the signal**. Four contributions:

(i) **J-invariant function class**: $f$ is J-invariant if $f(x)_J$ does not depend on $x_J$ for every $J \in \mathcal J$ — the function predicts each block from its complement.

(ii) **Self-supervised theorem (Proposition 1)**: For unbiased $x$ and J-invariant $f$, $\mathbb E\|f(x)-x\|^2 = \mathbb E\|f(x)-y\|^2 + \mathbb E\|x-y\|^2$. The second term is constant in $f$, so minimising the self-supervised loss is equivalent to minimising the unobservable true MSE.

(iii) **Calibration of classical denoisers (donut trick)**: Any parametric denoiser $g_\theta$ becomes J-invariant by replacing pixels in $J$ with neighbour averages before applying $g_\theta$. The self-supervised-loss curve has the same minimiser as the ground-truth-loss curve — calibration without GT (Fig. 2).

(iv) **Deep learning instantiation**: DnCNN/UNet trained with 4×4 grid masking match N2N/N2T on Hànzì, ImageNet, CellNet (Table 2) and beat NLM/BM3D. The same theorem also calibrates principal-component-regression rank in single-cell RNA-seq — the framework is *modality-agnostic*.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting
**한국어**: 2018–2019년 self-supervised denoising 의 폭발기. Lehtinen+의 Noise2Noise(paper #16)가 깨끗 타겟 요구를 제거했고, Krull+의 Noise2Void(paper #17)는 페어 요구도 제거했지만 *blind-spot 마스킹 트릭*에 의존하는 경험적 방법이었다 — *왜 그것이 작동하는가*의 이론적 정당화는 부재. Batson(Chan Zuckerberg Biohub)과 Royer는 *J-invariance* 라는 partition-based 함수 클래스 추상화로 이 공백을 메웠다 — N2N도, N2V도, donut median filter도, donut NLM도, donut wavelet shrinkage도 *모두* 같은 정리의 특수 사례라는 것. 이는 단순한 통합이 아니라 (a) 고전 알고리즘의 *hyperparameter calibration* 을 GT 없이 가능하게 하고 (Fig. 2 의 빨간 화살표 = self-sup 최소 = GT 최소), (b) image 도메인 밖(single-cell RNA-seq)으로 확장하며, (c) 후속 변종(Self2Self, Neighbor2Neighbor, Blind2Unblind)의 이론적 기반이 되었다. N2V와 *concurrent submission* — N2V가 CVPR 2019, N2S가 ICML 2019.

**English**: 2018–2019 was the explosive growth phase of self-supervised denoising. Lehtinen+'s Noise2Noise (paper #16) had removed the clean-target requirement and Krull+'s Noise2Void (paper #17) the paired-image requirement, but the latter relied on an empirical *blind-spot masking trick* with no theoretical justification for *why* it works. Batson (Chan Zuckerberg Biohub) and Royer filled the gap with the *J-invariance* abstraction — a partition-based function class containing N2N, N2V, donut median, donut NLM, and donut wavelet shrinkage as *special cases*. Beyond mere unification, this enabled (a) hyperparameter calibration of classical denoisers without ground truth (Fig. 2: red-arrow self-sup minimum = GT minimum), (b) extension beyond images to single-cell RNA-seq, and (c) the theoretical foundation for subsequent variants (Self2Self, Neighbor2Neighbor, Blind2Unblind). Concurrent with N2V — N2V appeared at CVPR 2019, N2S at ICML 2019.

### 타임라인 / Timeline
```
1948 ─── Anscombe — Poisson VST (paper #11)
2005 ─── Buades+ — NL-means (paper #4)
2007 ─── Dabov+ — BM3D (paper #7)
2017 ─── Ulyanov+ — Deep Image Prior (training-free single-image)
2018 ─── Lehtinen+ — Noise2Noise (paper #16)
2019 ─── Krull+ — Noise2Void (paper #17, concurrent with N2S)
2019 ★★ Batson-Royer — Noise2Self (THIS PAPER)
2020 ─── Quan+ — Self2Self (Bernoulli dropout extension)
2021 ─── Huang+ — Neighbor2Neighbor (sub-sampled pairs)
2022 ─── Wang+ — Blind2Unblind (visible blind spots)
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **수학 / Math**:
  - Conditional expectation $\mathbb E[Y\mid X]$ 와 conditional independence: $X \perp Y \mid Z$.
  - Iterated expectation: $\mathbb E[Y] = \mathbb E_Z[\mathbb E[Y\mid Z]]$.
  - Partition of a set, indicator function $\mathbf 1_J$, complement $J^c$.
  - L_2 Hilbert-space orthogonality decomposition (Proposition 1 의 cross-term 분해).
- **딥러닝 / Deep learning**:
  - DnCNN, U-Net architecture; convolutional receptive field.
  - Per-pixel MSE loss, mini-batch gradient descent.
  - Bernoulli mask / dropout 의 통계적 효과.
- **통계 / Statistics**:
  - Maximum likelihood estimation, MMSE estimator (paper #14의 conditional-expectation 개념과 연결).
  - Gaussian Process 와 covariance kernel (Proposition 3 의 worst-case bound).
- **Single-cell RNA-seq / 단일세포 RNA-seq** (개념만):
  - mRNA molecule count 측정의 sparsity 와 sub-sampling 잡음.
  - Principal Component Regression rank 선택 문제.
- **선행 논문 / Prior reading (필수 / Required)**:
  - Paper #16 (Noise2Noise) — N2S §2 가 N2N을 J-invariance로 재해석.
  - Paper #17 (Noise2Void) — concurrent work; N2V의 blind-spot이 J-invariant 의 *근사*임을 N2S가 제시.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Partition $\mathcal J$ | 측정 차원 $\{1,\dots,m\}$ 의 분할 $\{J_1, J_2, \dots\}$ — disjoint, union = whole. / Partition of measurement dimensions into disjoint blocks. |
| J-invariance | $f(x)_J$가 $x_J$에 의존하지 않는 함수 — 자기를 보지 않고 자기를 예측. / Function whose output on $J$ does not depend on input on $J$. |
| Self-supervised loss | $\mathcal L_{\rm self}(f) = \mathbb E\|f(x) - x\|^2$ — GT 없이 평가 가능. / Observable loss without GT. |
| Donut filter | classical $g_\theta$의 J-invariant 변형 — 마스크 위치를 이웃 평균으로 채움. / J-invariantised classical denoiser via neighbour-average replacement. |
| Mix-back ($\lambda$-blend) | $\hat y = \lambda f(x) + (1-\lambda) x$, $\lambda^* = \mathrm{Var}(n) / \mathcal L_{\rm self}$. J-invariance 정보 손실 일부 회복. / Optimal affine mixture of denoiser output and raw input recovers some lost information. |
| Bi-cross-validation | Owen-Perry 2009, feature + sample 양쪽 분할로 cross-validate. N2S의 special case. / Owen-Perry's two-way cross-validation; special case of N2S. |
| Hànzì dataset | 한자 문자 + Poisson + Gaussian + Bernoulli 혼합 잡음. 강한 구조 + 강한 잡음. / Chinese-character dataset with mixed Poisson + Gaussian + Bernoulli noise. |
| CellNet | fluorescence microscopy + simulated sCMOS heteroscedastic noise. / Microscopy dataset with simulated sCMOS noise. |
| 4×4 grid masking | partition $|\mathcal J| = 25$, 매 minibatch 마다 한 $J$ 사용. / 25-block partition; one $J$ per mini-batch. |
| Worst-case GP | Proposition 3 — 동일 covariance 하에 GP가 J-invariant 회귀 *가장 어려운* 경우. / GP is the worst case for J-invariant prediction among signals with given covariance. |
| sCMOS noise | fluorescence sensor의 heteroscedastic (signal-dependent variance) 잡음. / Sensor noise model with signal-dependent variance. |
| Probabilistic generalisation | 후속 Probabilistic N2V, DivNoising 의 J-invariance 기반 noise-model 학습. / Follow-ups that learn explicit noise models on top of J-invariance. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 J-invariance 정의 / Definition
$$
f: \mathbb R^m \to \mathbb R^m, \quad f(x)_J \text{ does not depend on } x_J \quad \forall J \in \mathcal J
$$
함수가 자기 위치의 입력 정보를 사용하지 않음. / Function does not see itself.

### 5.2 Self-supervised loss / 자기지도 손실
$$
\mathcal L(f) = \mathbb E \|f(x) - x\|^2
$$
관찰 가능 — GT 불필요. / Observable; no clean ground truth needed.

### 5.3 Main theorem (Proposition 1) / 핵심 정리
가정: $\mathbb E[x|y] = y$, $x_J \perp x_{J^c} \mid y$. $f$ J-invariant.
$$
\boxed{\;\mathbb E\|f(x) - x\|^2 = \mathbb E\|f(x) - y\|^2 + \mathbb E\|x - y\|^2\;}
$$
LHS: 관찰 가능 self-sup loss. RHS 첫째 항: 진짜 MSE. RHS 둘째 항: 잡음 분산 — *상수*. 따라서 **self-sup loss 최소화 = 진짜 MSE 최소화**. / Self-sup-loss minimisation is equivalent to true-MSE minimisation up to a constant noise-variance term.

### 5.4 Optimal J-invariant predictor (Proposition 2) / 최적 예측기
$$
f^*_{\mathcal J}(x)_J = \mathbb E[y_J \mid x_{J^c}]
$$
*complementary* observation block 으로부터의 conditional mean. Bayes posterior 의 partial form. / Conditional mean of signal block given the complementary observation block.

### 5.5 Donut J-invariantisation (Eq. 3) / 도넛 변형
For classical $g_\theta$ and neighbour-average $s(x)$:
$$
f_\theta(x)_J := g_\theta\bigl(\mathbf 1_J \cdot s(x) + \mathbf 1_{J^c} \cdot x\bigr)_J
$$
median radius, wavelet threshold, NLM bandwidth 같은 매개변수를 GT 없이 calibrate. / Calibrate any parametric classical denoiser without ground truth.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
**우선 읽을 부분 / Focus first**:
1. **§4 Theory (Proposition 1, 증명)** — 핵심. cross-term 이 zero-mean conditional independence 로 사라지는 단계 *손으로 따라가기*. 두 가정의 *각각이 어디에서 필요한지* 추적.
2. **Fig. 2** — donut median filter 의 self-sup loss 곡선이 GT loss 곡선과 *수직 거리만큼* 떨어진 채 *형상이 동일* 함을 시각적으로 확인. 빨간 화살표 = self-sup 최소 = GT 최소.
3. **§3.1 Single-cell RNA-seq** — image 도메인 밖에서도 동일 정리 적용. partition 을 *RNA molecule 단위*로 잡으면 PCR rank calibration이 가능. *modality-agnostic* 의 강력한 시연.
4. **Table 2** — N2S ≈ N2N ≈ N2T on Hànzì/CellNet, ImageNet에서 N2T가 +2-3 dB 격차. 자연 영상이 J-invariance 이론적 한계 (block correlation) 에 가장 도전적.

**자주 헷갈리는 지점 / Common stumbling blocks**:
- Proposition 1 의 두 가정 — *unbiasedness* + *conditional independence across $\mathcal J$* — 양쪽이 모두 필요. 가우시안 i.i.d. 가정 *불필요* (이게 N2S의 강점).
- "J-invariant"는 함수 *클래스* 제약이지 *특정* 함수가 아님. 학습은 이 클래스 *내에서* 자유 minimization.
- Donut filter 의 "이웃 평균" $s(x)$는 *다른 J-invariantisation 방법*이 가능함을 의미; 본 논문은 한 가지 구체적 선택.
- N2V (paper #17) 의 blind-spot 마스킹은 J-invariance 의 *근사* — 이웃에서 random sample로 채우면 *그 값이 자기 자신*일 수 있어 strict J-invariance 위배. N2S는 이 문제를 *neighbour-average* (자기 제외) 로 해결.
- $\mathbb E[y_J \mid x_{J^c}]$ 가 *진짜* Bayes posterior $\mathbb E[y\mid x]$ 와 *얼마나 다른지* 는 block correlation 에 의존 (§4.1 Gaussian Process worst-case).

### English
**Focus first**:
1. **§4 Theory (Proposition 1, proof)** — The core. Trace by hand how the cross term vanishes via zero-mean conditional independence. Pinpoint *where each of the two assumptions is used*.
2. **Fig. 2** — Visually confirm that the donut median filter's self-supervised-loss curve has the *same shape* as the GT-loss curve, separated only by a constant vertical offset. Red arrow = self-sup minimum = GT minimum.
3. **§3.1 Single-cell RNA-seq** — Same theorem outside the image domain. Partition over *RNA molecules* enables PC-regression rank calibration. The strongest demonstration of *modality-agnostic* applicability.
4. **Table 2** — N2S ≈ N2N ≈ N2T on Hànzì/CellNet; ImageNet shows the largest N2T-N2S gap of 2-3 dB. Natural images push J-invariance's theoretical limit (block correlation) hardest.

**Common stumbling blocks**:
- Proposition 1's two assumptions — *unbiasedness* + *conditional independence across $\mathcal J$* — *both* are required. *No Gaussian-i.i.d. assumption* (this is N2S's strength).
- "J-invariant" is a *function-class constraint*, not a specific function. Training is free minimisation *within* the class.
- The donut "neighbour-average" $s(x)$ is one specific J-invariantisation; other choices are possible.
- N2V (paper #17)'s blind-spot masking is an *approximation* to J-invariance — replacing a pixel with a random nearby value may copy the pixel itself, violating strict J-invariance. N2S fixes this with *neighbour-average* (self-excluded).
- How much $\mathbb E[y_J \mid x_{J^c}]$ differs from the *true* Bayes posterior $\mathbb E[y\mid x]$ depends on block correlation (§4.1 Gaussian Process worst-case).

---

## 7. 현대적 의의 / Modern Significance

### 한국어
Noise2Self 는 self-supervised denoising 의 *이론적 정점* 이다. N2N과 N2V 가 경험적 / 응용적 발견이었다면 N2S 는 *왜 그것들이 동작하는가* 의 정식 정당화. 후속 발전: Self2Self(2020, Bernoulli dropout) 는 J-invariance를 stochastic ensembling으로; Neighbor2Neighbor(2021)는 spatial decorrelation 이용한 sub-sample N2N 이지만 J-invariance 시각으로 해석 가능; Blind2Unblind(2022)는 visible blind-spot으로 정보 손실 회복; Probabilistic N2V(2020)는 명시적 noise model을 J-invariance 위에 학습. 더 넓게 보면 N2S 의 "partition-based self-supervision" 패턴은 (a) masked autoencoder(MAE 2022)의 patch masking, (b) BERT 의 masked language modeling, (c) score-based diffusion(forward step 의 partition), (d) self-supervised contrastive learning(SimCLR 의 augmentation pair) 모두 같은 *자기 일부 가리기* 원리의 응용이라 볼 수 있다. 모달리티 측면에서도 single-cell RNA-seq, mass spectrometry, SAR 영상 등 다양한 도메인에서 N2S 가 기본 calibration tool로 사용된다.

### English
Noise2Self stands at the **theoretical apex** of self-supervised denoising. While N2N and N2V are empirical/applied discoveries, N2S provides the formal justification for *why* they work. Follow-ups: Self2Self (2020, Bernoulli dropout) extends J-invariance to stochastic ensembling; Neighbor2Neighbor (2021) is a sub-sampled-N2N reinterpreted through J-invariance via spatial decorrelation; Blind2Unblind (2022) recovers lost information using visible blind spots; Probabilistic N2V (2020) learns explicit noise models atop J-invariance. More broadly, N2S's "partition-based self-supervision" pattern unifies (a) masked autoencoders (MAE 2022)'s patch masking, (b) BERT's masked language modelling, (c) score-based diffusion (partition over forward steps), and (d) contrastive self-supervision (SimCLR's augmentation pairs) — all instances of the same *self-information-blocking* principle. Cross-modality, N2S is also a default calibration tool in single-cell RNA-seq, mass spectrometry, and SAR imagery.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
