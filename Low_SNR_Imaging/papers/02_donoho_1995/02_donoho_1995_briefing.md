---
title: "Pre-Reading Briefing: Adapting to Unknown Smoothness via Wavelet Shrinkage"
paper_id: "02_donoho_1995"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Adapting to Unknown Smoothness via Wavelet Shrinkage: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Donoho, D. L., & Johnstone, I. M., "Adapting to Unknown Smoothness via Wavelet Shrinkage", *Journal of the American Statistical Association*, 90(432), 1200–1224 (1995). [DOI: 10.1080/01621459.1995.10476626]
**Author(s)**: David L. Donoho, Iain M. Johnstone
**Year**: 1995

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 paper #1 의 *고정* universal threshold 를 **레벨별(j-별) 데이터 의존 임계값** 으로 확장한 **SureShrink** 를 제시한다. 핵심 아이디어는 각 dyadic resolution level $j$ 에서 wavelet 계수를 독립 다변량 정규 추정 문제로 보고, **Stein의 비편향 위험 추정량(SURE)** 을 최소화하는 threshold $\hat t^S_j$ 를 데이터로부터 직접 선택하는 것이다. 매우 sparse 한 레벨에서는 SURE 가 잡음에 휩쓸리므로 universal $\sqrt{2\log d}$ 로 fallback 하는 **hybrid scheme** 을 도입한다. 그 결과 SureShrink 는 평활도 정도 $\sigma$, 종류 $p$, 양 $C$ 를 *몰라도* 모든 Besov ball $B^\sigma_{p,q}(C)$ 에서 *동시에* 거의 minimax 성능을 달성한다.

### English
This paper extends paper #1 by replacing the *single* universal threshold with **level-dependent, data-driven thresholds** chosen via **Stein's Unbiased Risk Estimate (SURE)** at each dyadic resolution. The core idea is to treat the coefficients at each level $j$ as a multivariate normal estimation problem and pick $\hat t^S_j$ minimising SURE. A **hybrid scheme** falls back to the universal threshold $\sqrt{2\log d}$ on sparse levels where SURE's variance dominates. The resulting SureShrink estimator is *simultaneously* near-minimax over the entire Besov scale $B^\sigma_{p,q}(C)$ — without knowing $\sigma$, $p$, $q$, or $C$. Adaptive linear shrinkers (James-Stein) provably cannot match this on $p<2$ Besov classes, so nonlinearity is essential.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting
Paper #1 (1994) 직후, 통계학자들은 *함수 클래스를 모르는* 상태에서도 minimax rate 를 자동 달성하는 절차를 추구했다. Stein 의 1981 년 unbiased risk identity 가 *데이터로부터 risk 를 비편향 추정* 하는 강력한 도구였지만, soft-threshold 추정기에 적용된 적은 거의 없었다. 한편 Besov body 라는 함수 공간 framework 가 ($p < 2$ 까지 포함하는) 광범위한 평활도 클래스를 통합하고 있었다. 이 논문은 SURE + level-dependent + Besov 라는 세 흐름을 결합해 *데이터 의존 wavelet thresholding* 시대를 열었다.
Right after paper #1 (1994), statisticians sought procedures that attain minimax rates *without knowing the function class*. Stein's 1981 unbiased risk identity was a powerful tool but had rarely been applied to soft-thresholding estimators. Meanwhile, the Besov-body framework had unified $L^p$ smoothness classes (including $p<2$). This paper fused SURE + level-dependence + Besov to launch the *data-adaptive era* of wavelet thresholding.

### 타임라인 / Timeline
```
1981 ─── Stein — unbiased risk identity (SURE)
1985 ─── Nemirovskii — linear methods can't reach minimax on p<2 Besov
1992 ─── Triebel — Theory of Function Spaces II (Besov framework)
1994 ─── Donoho-Johnstone — VisuShrink / RiskShrink (paper #1)
1995 ★★ DONOHO-JOHNSTONE — SureShrink (THIS PAPER)
1995 ─── Donoho-Johnstone-Kerkyacharian-Picard — Wavelet Shrinkage: Asymptopia?
2000 ─── Chang-Yu-Vetterli — BayesShrink (paper #3, GGD prior)
2007 ─── Blu-Luisier — SURE-LET (linear expansion of thresholders)
2018 ─── Lehtinen+ — Noise2Noise (Stein identity in deep learning)
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Paper #1 (필수) / Paper #1 (mandatory)**: VisuShrink, soft thresholding, oracle inequality framework.
- **Stein 의 정리 / Stein's lemma**: $E_\mu\|\hat\mu - \mu\|^2$ 의 비편향 추정 — 약미분 가능 추정기에 대한 기본 항등식.
- **다변량 정규 추정 / Multivariate normal estimation**: $\mathbf{x} = \boldsymbol\mu + \boldsymbol\sigma\mathbf{z}$ 모델, James-Stein 추정기.
- **Besov space 기초 / Besov space fundamentals**: $B^\sigma_{p,q}$ 의 의미 (Sobolev = $p=2$, Hölder = $p=q=\infty$, BV ≈ $p=1$).
- **Minimax rate 정의 / Minimax rates**: $R(N; \mathcal F) = \inf_{\hat f} \sup_{\mathcal F} R$.
- **정렬 알고리즘 / Sorting**: $O(d\log d)$ — SURE 최소화 cost 의 핵심.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| SURE | Stein's Unbiased Risk Estimate — risk 의 비편향 추정량 / Unbiased estimator of MSE for weakly differentiable estimators. |
| Level-dependent threshold | 각 dyadic level $j$ 마다 별도 threshold $t_j$ / Threshold chosen separately per resolution level. |
| Sparsity statistic ($s^2_d$) | $d^{-1}\sum(x_i^2 - 1)$ — 신호 존재 여부 검출 / Detects whether the level contains signal. |
| Hybrid scheme | sparse 면 universal, dense 면 SURE / Universal threshold for sparse levels, SURE for dense levels. |
| Besov ball $B^\sigma_{p,q}(C)$ | $L^p$ 매끄러움 $\sigma$, $L^q$ tail behaviour, 반지름 $C$ / $L^p$-smoothness ball generalising Sobolev/Hölder/BV. |
| Simultaneous adaptation | 모든 $(\sigma, p, q, C)$ 에 대해 단일 절차가 거의 minimax / Single procedure near-minimax over the whole scale. |
| Stein's identity | $E\|\hat\mu - \mu\|^2 = d + E\{\|g\|^2 + 2\nabla\cdot g\}$ for $\hat\mu = x + g$. |
| James-Stein | $(1 - (d-2)/\|x\|^2)_+ x$ — 적응적 *선형* 수축 / Adaptive linear shrinker. |
| Nonlinear shrinkage | 좌표별 비선형 함수 (soft/hard threshold) / Coordinatewise nonlinear thresholding. |
| HaarShrink | Haar 기저 + SureShrink → BV near-minimax / Haar basis variant; near-minimax over BV. |
| Coarse-level fallback | $j < L$ 레벨은 임계화 안 함 (paper #1 동일) / Levels coarser than $L$ are left untouched. |

---

## 5. 수식 미리보기 / Equations Preview

**Stein's identity (Eq. 10)** — 모든 것의 출발점:
$$
E_\mu\|\hat\mu - \mu\|^2 = d + E_\mu\bigl\{\|\mathbf{g}\|^2 + 2\nabla\cdot\mathbf{g}\bigr\}, \qquad \hat\mu(\mathbf{x}) = \mathbf{x} + \mathbf{g}(\mathbf{x})
$$

**SURE for soft thresholding (Eq. 11)** — 핵심 식:
$$
\mathrm{SURE}(t; \mathbf{x}) = d - 2\#\{i: |x_i| \le t\} + \sum_{i=1}^d \min(x_i^2, t^2)
$$
$t$ 에 대해 piecewise quadratic — 정렬된 $|x_i|$ 사이에서만 평가하면 $O(d\log d)$.
Piecewise quadratic in $t$, so optimisation costs only $O(d\log d)$ via sorting.

**Threshold selection (Eq. 12)**:
$$
t^S = \arg\min_{0 \le t \le \sqrt{2\log d}} \mathrm{SURE}(t; \mathbf{x})
$$

**Hybrid scheme (Eq. 14)** — sparse fallback:
$$
\hat\mu^+(\mathbf{x})_i = \begin{cases} \eta_{\sqrt{2\log d}}(x_i) & s^2_d \le \gamma_d \;\;(\text{sparse}\to\text{universal}) \\ \eta_{t^S}(x_i) & s^2_d > \gamma_d \;\;(\text{dense}\to\text{SURE}) \end{cases}
$$
with $s^2_d = d^{-1}\sum(x_i^2 - 1)$, $\gamma_d = (\log d)^{3/2}/\sqrt{d}$.

**Adaptivity (Theorem 1)**:
$$
\sup_{B^\sigma_{p,q}(C)} R(\hat f^*, f) \asymp N^{-2\sigma/(2\sigma+1)}, \quad \text{simultaneously for all } (\sigma, p, q, C)
$$

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **§1 (Setup)**: paper #1 과 동일 모델임을 확인하고, *함수 클래스 미지* 라는 새 조건에 집중.
- **§2 (SureShrink algorithm)**: 핵심. Eq. 11 (SURE 식), Eq. 12 (threshold), Eq. 14 (hybrid), Definition 1 (full algorithm) 을 외울 정도로.
- **§3 (Theorem 1)**: 결과만 받아들이고 증명은 skim. *동시* minimax 라는 의미 — 한 절차가 모든 클래스에서 작동.
- **§4 (vs James-Stein)**: §4.1 의 핵심 메시지 — *비선형성이 본질적*. James-Stein 은 $L^2$ 에서만 좋고 $p<2$ 에서 fail.
- **§5 (Simulations)**: Figure 13 의 4개 panel — pure SURE vs universal vs hybrid 의 *trade-off* 가 한눈에.
- **흔한 걸림돌**: (i) SURE 는 *진짜 risk* 가 아니라 *비편향 추정량* — 무한 분산을 가질 수 있음. 이 때문에 sparse 시 fallback 이 필수. (ii) Besov scale 에서 $p<2$ 가 *특히 중요* — 신호처리의 edge·jump 함수가 여기 속함. (iii) coarse level $j<L$ 은 임계화 안함은 paper #1 과 같음.

### English
- **§1 Setup**: same model as paper #1, but the function class is *unknown* — focus on this new condition.
- **§2 Algorithm**: the core. Memorise Eq. 11 (SURE), Eq. 12 (threshold), Eq. 14 (hybrid), Definition 1 (full algorithm).
- **§3 Theorem 1**: take the conclusion, skim the proof. The word *simultaneously* means a single procedure works on every Besov class.
- **§4 (vs James-Stein)**: §4.1 key message — *nonlinearity is essential*. James-Stein matches the ideal linear shrinker but fails on $p<2$ Besov.
- **§5 Simulations**: Figure 13's four panels visualise the SURE / universal / hybrid trade-off at a glance.
- **Common stumbling blocks**: (i) SURE is an *unbiased estimator* of risk, not the risk itself — its variance can be huge, hence the sparse fallback. (ii) The $p<2$ part of the Besov scale is what makes this paper matter for image-like signals (edges, jumps). (iii) Coarsest levels $j<L$ are not thresholded (same as paper #1).

---

## 7. 현대적 의의 / Modern Significance

### 한국어
SureShrink 는 *데이터 의존 wavelet thresholding* 의 표준 템플릿이 되었다. BayesShrink (paper #3) 는 SURE 객체를 *Bayesian closed-form* 으로 교체하고, SURE-LET (Blu-Luisier 2007) 은 *linear expansion of thresholding functions* 위에서 SURE 를 직접 최소화한다. 더 놀랍게도, **Noise2Noise** (Lehtinen+ 2018) 같은 self-supervised deep denoising 이 같은 Stein identity Eq. 10 위에서 작동한다 — 학습된 네트워크 $f_\theta$ 의 risk 를 *clean target 없이* 비편향 추정. SURE-based loss 는 현대 self-supervised imaging (low-SNR microscopy, astronomy) 에서 활발히 사용된다. Hybrid scheme 의 sparsity-test 도 *학습된 attention gate* 같은 형태로 신경망에 재등장한다.

### English
SureShrink became the standard template for *data-adaptive wavelet thresholding*. BayesShrink (#3) replaces the SURE objective with a Bayesian closed-form; SURE-LET (Blu-Luisier 2007) optimises SURE directly over a *linear expansion of thresholding functions*. More remarkably, self-supervised deep denoising — **Noise2Noise** (Lehtinen+ 2018), Noise2Self, etc. — relies on the same Stein identity (Eq. 10) to estimate network risk *without clean targets*. SURE-based losses are now mainstream in self-supervised low-SNR imaging (microscopy, astronomy). The hybrid scheme's sparsity test reappears in modern networks as a learned gating mechanism.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
