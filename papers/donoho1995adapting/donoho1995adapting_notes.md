---
title: "Adapting to Unknown Smoothness via Wavelet Shrinkage"
authors: David L. Donoho, Iain M. Johnstone
year: 1995
journal: "Journal of the American Statistical Association 90(432), pp. 1200–1224"
doi: "10.1080/01621459.1995.10476626"
topic: Low-SNR Imaging / Wavelet Denoising
tags: [wavelet, sureshrink, sure, stein-unbiased-risk-estimate, level-dependent-threshold, besov-spaces, minimax, smoothness-adaptation, donoho-johnstone, hybrid-threshold]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 2. Adapting to Unknown Smoothness via Wavelet Shrinkage / 미지의 매끄러움에 대한 웨이블릿 수축 적응

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 paper #1의 단일 universal threshold를 **레벨별(j-별) 데이터 의존 임계값**으로 확장한 **SureShrink**를 제시한다. 핵심 아이디어: 각 dyadic resolution level $j$에서 wavelet 계수를 *독립 다변량 정규 추정 문제*로 보고, **Stein의 비편향 위험 추정량(SURE)** 을 최소화하는 threshold $\hat t^S_j$를 데이터로부터 직접 선택한다. 핵심 기여 4가지:
(i) **Level-dependent SURE threshold (Eq. 11–12)**: $\mathrm{SURE}(t; \mathbf x) = d - 2\#\{|x_i| \le t\} + \sum (|x_i| \wedge t)^2$을 $t \in [0, \sqrt{2\log d}]$에서 최소화. $O(d \log d)$ 정렬 알고리즘.
(ii) **Hybrid scheme (Eq. 14)**: 희소(sparse) 레벨에서는 SURE가 잡음에 휩쓸리므로, $s^2_d = d^{-1}\sum(x_i^2 - 1)$이 $\gamma_d = (\log d)^{3/2}/\sqrt d$ 이하이면 universal $\sqrt{2\log d}$로 fallback.
(iii) **Smoothness adaptive (Theorem 1)**: $N \to \infty$에서 $\sup_{B^\sigma_{p,q}(C)} R(\hat f^*, f) \asymp R(N; B^\sigma_{p,q}(C))$ 모든 $p, q \in [1, \infty]$, 모든 $\sigma \in (0, r)$에 대해 *동시에* 거의 minimax 성능. 즉 평활도 정도 $\sigma$, 종류 $p$, 양 $C$를 *몰라도* 알았을 때만큼 좋다.
(iv) **Linear shrinkage 한계 돌파 (Theorem 5)**: James-Stein 같은 적응적 *선형* 수축은 $p < 2$ Besov body에서 minimax rate를 달성 못하지만, SureShrink (비선형) 는 달성. 즉 비선형성이 본질적.

### English
This paper extends paper #1 by replacing the *single* universal threshold with **level-dependent, data-driven thresholds** chosen via **Stein's Unbiased Risk Estimate (SURE)** at each dyadic resolution. Four key contributions:
(i) **SURE-based threshold (Eqs. 11-12)**: minimise $\mathrm{SURE}(t;\mathbf x) = d - 2\#\{|x_i|\le t\} + \sum(|x_i|\wedge t)^2$ over $t \in [0, \sqrt{2\log d}]$; $O(d \log d)$ by sorting $|x_i|$.
(ii) **Hybrid scheme (Eq. 14)**: SURE breaks down when the level is sparse; switch to universal threshold $\sqrt{2\log d}$ when sparsity statistic $s^2_d \le \gamma_d$.
(iii) **Simultaneously minimax over Besov scale (Theorem 1)**: optimal rate over every $B^\sigma_{p,q}(C)$ ball, for *all* $\sigma, p, q$ simultaneously, without knowing them.
(iv) **Beats adaptive linear shrinkage (Theorem 5)**: nonlinearity is essential — adaptive linear methods (James-Stein) fail to attain minimax over $p<2$ Besov bodies, while SureShrink succeeds.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Setup / 설정

#### 한국어
- 모델은 paper #1과 동일: $y_i = f(t_i) + \sigma z_i$, $t_i = (i-1)/N$, $z_i \overset{iid}{\sim} N(0,1)$.
- 핵심 차이: 부드러움 $\sigma$, $C$를 *모름*. 이전 연구 (Stone 1982, Nemirovskii-Polyak-Tsybakov 1985, Nussbaum 1985)는 *고정* 클래스 $\mathcal F$에 대한 minimax. 실용에선 $\mathcal F$ 미지.
- Sobolev $W^m_2(C)$: $L^2$ 매끄러움 클래스. **Besov $B^\sigma_{p,q}(C)$**: $L^p$ 매끄러움까지 일반화 — 불연속을 가진 함수, BV ($p=1$), Hölder ($p=q=\infty$) 등을 포함.
- 비적응 선형 추정기는 $p < 2$ Besov에서 minimax rate 도달 불가능 (Nemirovskii 1985). SureShrink는 가능.

#### English
Same model as paper #1, but assumes the smoothness class is *unknown* (only some scale is known). Besov scale $B^\sigma_{p,q}$ captures $L^p$ smoothness with discontinuities (BV, Hölder etc.). Linear methods cannot attain minimax over $p<2$ Besov classes.

---

### Part II: §2 SureShrink Algorithm / SureShrink 알고리즘

#### 한국어 — §2.1 DWT
- $\mathcal W$: 직교 DWT, $N = 2^J$. Symmlet $N=8$, periodised (또는 boundary-corrected, Cohen-Daubechies-Vial 1993).
- Mallat의 multiresolution: $L$은 가장 거친 레벨; $j = L, L+1, \ldots, J-1$에서 detail 계수 $w_{j,k}$, $k = 0, \ldots, 2^j - 1$.

#### 한국어 — §2.2 Soft thresholding (Eq. 4)
$$
\eta_t(y) = \mathrm{sgn}(y)(|y| - t)_+
$$
*Same as paper #1.* 차이는 $t$가 *level-dependent*라는 것.

#### 한국어 — §2.3 SURE-based threshold selection
**Stein 1981**: 약 미분 가능 $\hat\mu(\mathbf x) = \mathbf x + \mathbf g(\mathbf x)$에 대해
$$
E_\mu \|\hat\mu - \mu\|^2 = d + E_\mu\bigl\{\|\mathbf g\|^2 + 2 \nabla \cdot \mathbf g\bigr\} \quad (10)
$$
이 식의 *unbiased estimator* (구체적 $\mu$ 없이 계산 가능):
$$
\boxed{\;\mathrm{SURE}(t; \mathbf x) = d - 2 \cdot \#\{i: |x_i| \le t\} + \sum_{i=1}^d (|x_i| \wedge t)^2 \quad (11)\;}
$$
이는 soft-threshold 추정량 $\hat\mu^{(t)}_i = \eta_t(x_i)$의 risk $E\|\hat\mu^{(t)} - \mu\|^2$에 대한 비편향 추정량.

**Threshold 선택 (Eq. 12)**:
$$
t^S = \arg\min_{0 \le t \le \sqrt{2\log d}} \mathrm{SURE}(t; \mathbf x)
$$
상한 $\sqrt{2\log d}$는 universal threshold; "이보다 더 큰 임계값은 시각적 의미 없음" + sparsity-fallback의 일관성.

**Computational cost**: SURE는 $t$에 대해 *조각적 이차*. $|x_i|$를 정렬한 뒤 인접 정렬 값 사이의 SURE 값만 계산하면 됨. 각 구간 끝점에서 $O(1)$에 SURE 갱신 가능 → 전체 $O(d \log d)$ (정렬이 dominant).

#### English — §2.3 SURE threshold
Stein's unbiased risk estimate Eq. (11) is computed in $O(d \log d)$ by sorting $|x_i|$ and evaluating SURE between consecutive sorted values (where it's piecewise quadratic). Choose $t^S = \arg\min$ over $[0, \sqrt{2\log d}]$.

#### 한국어 — §2.4 Hybrid scheme (sparse fallback)

**문제**: 매우 sparse 한 레벨 (대부분 $\mu_i = 0$, 극소수 큰 신호 + 잡음)에서는 SURE의 잡음 자체가 진폭이 커서 신호의 진짜 risk shape를 가린다 → $t^S$가 너무 작게 선택됨.

**해결**: 데이터 vector의 sparsity 통계
$$
s^2_d = \frac{1}{d}\sum_{i=1}^d (x_i^2 - 1)
$$
$\mu = 0$이면 $E[s^2_d] = 0$, $s^2_d \approx \mathrm{Var}$ of pure noise. 신호가 들어 있으면 $s^2_d > 0$. 임계 $\gamma_d = (\log d)^{3/2}/\sqrt d$ 또는 더 일반적으로 $\gamma_d = d^{-\gamma}$, $0 < \gamma < 1/2$.

**Hybrid estimator (Eq. 14)**:
$$
\hat\mu^+(\mathbf x)_i = \begin{cases} \eta_{t^F_d}(x_i) & s^2_d \le \gamma_d \quad \text{(sparse → universal threshold)} \\ \eta_{t^S}(x_i) & s^2_d > \gamma_d \quad \text{(dense → SURE)} \end{cases}
$$
$t^F_d = \sqrt{2\log d}$. 두 체제 부드럽게 전환.

#### English — §2.4 Hybrid scheme
When sparsity statistic $s^2_d = d^{-1}\sum(x_i^2 - 1) \le \gamma_d = (\log d)^{3/2}/\sqrt d$, the level is sparse and SURE's noise dominates; switch to universal threshold $\sqrt{2\log d}$. Otherwise use SURE.

#### 한국어 — Definition 1: SureShrink (formal)
$N = 2^J$, $\sigma = 1$. 데이터 $\mathbf x_j = (y_{j,k})_{k=0}^{2^j - 1}$. 추정량:
$$
\hat w^*_{j,k} = \begin{cases} y_{j,k} & j < L \\ (\hat\mu^*(\mathbf x_j))_k & L \le j < J \end{cases} \qquad \hat f^* = \mathcal W^T \hat w^*
$$
$j < L$인 거친 스케일은 임계화 안함 (paper #1과 동일 이유: vanishing moment 부재).

전체 계산량: 각 레벨 $j$에서 $O(2^j \cdot j)$, 총합 $O(N \log N)$. 

#### English — Definition 1
Apply level-by-level adaptive thresholding to each detail level $j \ge L$; coarsest level $j < L$ untouched. Total cost $O(N \log N)$.

---

### Part III: §3 Main Adaptivity Result / 주요 적응성 결과

#### 한국어 — Theorem 1 (Smoothness adaptation)
파동 $\psi$가 $r$ vanishing moments + $r$차 연속 미분, $r > \max(1, \sigma)$일 때 SureShrink는 *동시에 거의 minimax*:
$$
\sup_{B^\sigma_{p,q}(C)} R(\hat f^*, \mathbf f) \asymp R(N; B^\sigma_{p,q}(C)) \asymp N^{-r}, \qquad r = \frac{\sigma}{\sigma + 1/2}
$$
모든 $p, q \in [1, \infty]$, 모든 $C \in (0, \infty)$, 모든 $\sigma_0 < \sigma < r$에 대해. $\sigma_0$은 $p, \gamma_d$에 의존:
- $\gamma_d = (\log d)^{3/2}/\sqrt d \Rightarrow \sigma_0 = \max(1/p, 2(1/p - 1/2)_+)$
- $\gamma_d = d^\gamma$, $0 < \gamma < 1/2 \Rightarrow \sigma_0 = \max(1/p, 2(1/p - 1/2)_+ + \gamma - 1/2)$

#### English — Theorem 1
SureShrink achieves the optimal $N^{-r}$ rate over *every* Besov ball $B^\sigma_{p,q}(C)$ — simultaneously across the entire scale, *without* knowing $\sigma, p, q, C$.

#### 한국어 — Theorem 2 (HaarShrink)
Haar 기저에 SureShrink를 적용하면 *bounded variation* 클래스 $\mathcal V(C)$에서 simultaneously near-minimax. BV는 신호처리에서 자연스러운 클래스 (조각 상수 함수 포함).

#### 한국어 — Theorem 4 (SURE mimics ideal threshold)
$\bar R(\mu) = \inf_t d^{-1} \sum r(t, \mu_i)$을 *ideal threshold risk* (각 좌표마다 최적 threshold). SureShrink hybrid 추정량 $\hat\mu^*$는:
$$
d^{-1} E_\mu \|\hat\mu^* - \mu\|^2 \le \bar R(\mu) + R_F(\mu)\,\mathbf 1\{\tau^2 \le 3\gamma_d\} + c(\log d)^{3/2} d^{-1/2}
$$
$\tau^2 = d^{-1}\sum \mu_i^2$. 즉 $\hat\mu^*$는 $\bar R$을 $(\log d)^{3/2} d^{-1/2}$ 인자만큼 추가 비용으로 모방.

#### English — Theorem 4
The SURE-based hybrid estimator mimics the ideal threshold risk $\bar R(\mu)$ up to a $(\log d)^{3/2}/\sqrt d$ penalty.

---

### Part IV: §4 Comparison with Adaptive Linear Shrinkage / 선형 수축과의 비교

#### 한국어 — §4.1 James-Stein
James-Stein 양 부분 추정량 $\hat\mu^{JS}_i = (1 - (d-2)/\|\mathbf x\|^2)_+ x_i$. Theorem 5: $E_\mu\|\hat\mu^{JS} - \mu\|^2 \le 2 + E_\mu\|\tilde\mu^{IS} - \mu\|^2$ (Eq. 21). 즉 ideal 선형 수축보다 *최대 2배만 나쁨*. 그러나 *비선형* SureShrink는 더 잘 함:
- 부드러운 $L^2$ Sobolev ($p=2$): JS와 SureShrink 비슷.
- 비매끈 $p < 2$ Besov: JS는 minimax rate 못 달성, SureShrink는 달성.

핵심 통찰: 전역 $\|x\|^2$에 의존하는 JS는 *공간 변동성*에 둔감. 좌표별 비선형 SureShrink는 spatial sparsity를 직접 활용.

#### English — §4.1 James-Stein
James-Stein achieves $\le 2\times$ the ideal linear shrinker (Theorem 5) but cannot match SureShrink on $p<2$ Besov classes. The nonlinear, coordinatewise nature of SureShrink is what allows spatial adaptation.

---

### Part V: §5 Simulation Study / 시뮬레이션

#### 한국어
Figure 13: $\mu \in \mathbb R^{1024}$, $\lfloor \varepsilon d \rfloor$개 비영 좌표 (값 $C$), 25 replications.
- (a) SURE 단독: 매우 sparse ($\varepsilon \approx 0.005$)에서 root MSE 거의 $\sqrt{2\log d}$ 인근, dense에서 작아짐.
- (b) Universal $\sqrt{2\log d}$: sparse 우수, dense에서 SURE보다 *훨씬* 나쁨.
- (c) Hybrid ($\hat\mu^*$): 두 영역에서 모두 좋음 — 매우 sparse에선 universal로 fallback, dense에선 SURE.
- (d) Hybrid 변형: 비슷.

이게 SureShrink가 단순 SURE보다 hybrid를 채택한 이유.

#### English — §5
Figure 13 simulations confirm: pure SURE is bad on sparse data, universal is bad on dense data, hybrid is good on both.

---

## 3. Key Takeaways / 핵심 시사점

1. **Level-dependent threshold가 핵심 개선 / Level-dependent threshold is the key improvement** — VisuShrink는 모든 detail level에서 같은 $\sigma\sqrt{2\log n}$을 사용. SureShrink는 각 level $j$마다 $d_j = 2^j$ 계수에 SURE를 minimise → 거친 스케일은 작은 threshold (신호 풍부), 미세 스케일은 큰 threshold (잡음 우세). 이 단순한 변화가 Besov 전 클래스에서 simultaneously minimax를 가능케 함.
   The single global threshold of VisuShrink wastes information; SureShrink picks a different threshold for each dyadic level by minimising Stein's unbiased risk estimator. This unlocks simultaneous near-minimaxity over the entire Besov scale.

2. **SURE는 risk의 비편향 추정량 / SURE is an unbiased estimator of risk** — Stein 1981의 마법: $\mu$를 모르면서도 $E\|\hat\mu - \mu\|^2$를 데이터로부터 비편향으로 추정. soft-threshold에 적용하면 Eq. (11). $t$에 대해 piecewise quadratic이라서 정렬된 $|x_i|$에서 $O(d \log d)$에 최적화 가능.
   Stein's identity gives an unbiased estimator of MSE without knowing the truth — usable as an objective for threshold selection. The piecewise-quadratic structure makes the minimisation $O(d \log d)$.

3. **Sparsity criterion으로 SURE의 약점 보완 / Sparsity criterion patches SURE** — SURE는 dense 데이터에서 잘 동작하지만, 거의 모든 좌표가 $\mu_i = 0$인 sparse 레벨에서는 잡음에 휩쓸려 $t^S$를 너무 작게 선택. Hybrid은 $s^2_d = d^{-1}\sum(x_i^2 - 1)$로 sparsity 검출 → sparse면 universal로 fallback. 이 단순한 검정이 Theorem 1의 $\sigma_0$ 하한을 결정한다.
   SURE breaks down on sparse levels where its variance is large relative to signal; the $s^2_d$ test detects this and falls back to the universal threshold. This switch is what allows simultaneous adaptation across smoothness types.

4. **Besov body로 평활도를 통합 / Besov bodies unify smoothness types** — Sobolev ($p=2$), BV ($p=1$), Hölder ($p=q=\infty$), $L^p$-Sobolev 모두 $B^\sigma_{p,q}$의 특수 경우. $p<2$는 *불연속 허용* — 신호처리·이미징의 edge·jump 함수가 여기 속함. SureShrink가 $p<2$에서도 잘 작동한다는 것이 그 기여의 본질.
   The Besov scale subsumes Sobolev, Hölder, BV; $p<2$ corresponds to functions with edges/jumps. SureShrink covers all of them simultaneously, including $p<2$ where linear methods fail.

5. **비선형성이 본질적 / Nonlinearity is essential** — Theorem 5 + §4.1: James-Stein (적응적 선형 수축)은 $p < 2$ Besov에서 minimax rate 달성 못함. SureShrink (좌표별 비선형 thresholding)은 달성. 이는 신호 처리의 거대 통찰: *공간 변동성을 가진 함수는 비선형 추정기가 본질적이다*.
   Adaptive linear shrinkers (James-Stein) cannot match minimax rates over $p<2$ Besov classes; coordinatewise nonlinear thresholding can. Spatial adaptation requires nonlinearity.

6. **$O(N\log N)$ 알고리즘 / $O(N\log N)$ total cost** — Level $j$의 SURE 최소화는 $O(2^j \cdot j)$. 모든 레벨 합 $O(\sum 2^j j) = O(N \log N)$. DWT 자체도 $O(N)$. 결국 SureShrink는 *FFT-급* 효율성 — 작은 $N$에선 SureShrink가 RiskShrink/VisuShrink보다 약간 비싸지만 점근적으로는 같은 클래스.
   SureShrink runs in $O(N\log N)$ — the same order as a single FFT — making it practical for large-$N$ signals.

7. **VisuShrink/RiskShrink는 SureShrink의 특수 경우 / VisuShrink/RiskShrink are special cases** — 모든 레벨에서 universal $\sqrt{2\log N}$을 강제하면 VisuShrink, minimax $\lambda^*_n$을 강제하면 RiskShrink. SureShrink는 *데이터에 맞춰 자동 선택*. Note: paper의 footnote — SureShrink는 paper #1의 변형들과 *threshold 선택만* 다를 뿐, 수축 비선형성 자체는 동일.
   VisuShrink/RiskShrink are SureShrink's special cases with fixed (universal/minimax) thresholds. SureShrink replaces these with data-driven choices.

8. **Stein lemma는 SURE 외에도 다양하게 사용 / SURE is one face of Stein's lemma** — 향후 BayesShrink, SURE-LET, Sub-band Adaptive (SAS) 등 모두 이 식 (10)에서 출발. Modern denoisers (BLS-GSM, BM3D)도 *암묵적* SURE-style risk를 사용. Deep-learning 시대에도 SURE는 *self-supervised* 학습 (Noise2Noise, SURE-based loss) 의 이론적 기반.
   SURE generalises far beyond thresholding — Bayesian shrinkage, SURE-LET, and even self-supervised deep denoising (Noise2Noise) all rely on Stein's identity.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Stein's identity and SURE / Stein 항등식과 SURE
For $\hat\mu(\mathbf x) = \mathbf x + \mathbf g(\mathbf x)$ weakly differentiable:
$$
E_\mu\|\hat\mu - \mu\|^2 = d + E_\mu\bigl\{\|\mathbf g\|^2 + 2\nabla \cdot \mathbf g\bigr\} \quad (10)
$$
For soft thresholding $\hat\mu^{(t)}_i = \eta_t(x_i)$ ($g_i = -\mathrm{sgn}(x_i)\min(|x_i|, t)$):
$$
\boxed{\;\mathrm{SURE}(t; \mathbf x) = d - 2\#\{i: |x_i| \le t\} + \sum_{i=1}^d \min(x_i^2, t^2) \quad (11)\;}
$$
### 4.2 SURE threshold selection / SURE 임계값 선택
$$
t^S = \arg\min_{0 \le t \le \sqrt{2\log d}} \mathrm{SURE}(t; \mathbf x) \quad (12)
$$
### 4.3 Sparsity statistic and hybrid scheme / Sparsity 통계와 hybrid 방식
$$
s^2_d = \frac{1}{d}\sum_{i=1}^d (x_i^2 - 1), \qquad \gamma_d = \frac{(\log d)^{3/2}}{\sqrt d}
$$
$$
\hat\mu^+(\mathbf x)_i = \begin{cases} \eta_{\sqrt{2\log d}}(x_i) & s^2_d \le \gamma_d \\ \eta_{t^S}(x_i) & s^2_d > \gamma_d \end{cases} \quad (14)
$$
### 4.4 SureShrink algorithm / SureShrink 알고리즘
**Input**: $\mathbf y \in \mathbb R^N$, $N = 2^J$; wavelet (Symmlet 8); coarsest level $L$ (e.g. 5); $\hat\sigma$ via MAD.

**Step 1**: $\mathbf w = \mathcal W \mathbf y$; group into level-bands $\{w_{j,k}\}_k$ for $j = L, \ldots, J-1$.
**Step 2**: For each level $j$:
  a. Standardise: $\mathbf x_j = \mathbf w_j / \hat\sigma$, $d_j = 2^j$.
  b. Compute $s^2_{d_j}$; if $\le \gamma_{d_j}$ → set $t_j = \sqrt{2\log d_j}$.
  c. Else → sort $|x_{j,k}|$ and find $t^S_j = \arg\min \mathrm{SURE}(t)$ on $[0, \sqrt{2\log d_j}]$.
**Step 3**: $\hat w_{j,k} = \hat\sigma \cdot \eta_{t_j}(x_{j,k})$ for $j \ge L$, else $\hat w_{j,k} = w_{j,k}$.
**Step 4**: $\hat{\mathbf f} = \mathcal W^T \hat{\mathbf w}$.

### 4.5 Efficient SURE minimisation / 효율적 SURE 최소화

Sort $|x_{(1)}| \le |x_{(2)}| \le \ldots \le |x_{(d)}|$. On the interval $t \in [|x_{(k-1)}|, |x_{(k)}|]$:
$$
\mathrm{SURE}(t) = d - 2(d - k + 1) + \sum_{i=1}^{k-1} x_{(i)}^2 + (d - k + 1) t^2
$$
This is increasing in $t$ (linear in $t^2$ with positive coefficient $d-k+1$), so minimum on each interval is at the *left endpoint* $|x_{(k-1)}|$. So search reduces to checking $\{|x_{(k)}|: k = 1, \ldots, d\}$ only — $O(d \log d)$ total.

### 4.6 Worked example / 수치 예시
$d = 1024$: universal $\sqrt{2\log 1024} = 3.722$. Sparsity threshold $\gamma_d = (\log 1024)^{3/2}/\sqrt{1024} = (6.93)^{1.5}/32 = 0.571$. For dense level ($s^2_d \gg 0.571$) SureShrink picks something like $t^S \in [1.0, 2.5]$ typically — much smaller than 3.722.

### 4.7 Theorem 1 (Adaptivity)
$$
\sup_{B^\sigma_{p,q}(C)} E\|\hat f^* - \mathbf f\|^2 \asymp N^{-2\sigma/(2\sigma+1)}
$$
simultaneously for all $(\sigma, p, q, C)$ with $\sigma_0 < \sigma < r$, where:
- $\gamma_d = (\log d)^{3/2}/\sqrt d \Rightarrow \sigma_0 = \max(1/p, 2(1/p - 1/2)_+)$
- $\gamma_d = d^\gamma, 0 < \gamma < 1/2 \Rightarrow \sigma_0 = \max(1/p, 2(1/p - 1/2)_+ + \gamma - 1/2)$

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1981 ─── Stein — "Estimation of the mean of a multivariate normal distribution"
          ↳ unbiased risk estimate (SURE) as a tool
1990 ─── Golubev & Nussbaum — adaptive Sobolev minimax (USSR school)
1993 ─── Cohen-Daubechies-Vial — boundary-corrected wavelets
1994 ─── Donoho-Johnstone — VisuShrink/RiskShrink (paper #1)
                            ↳ universal/minimax threshold
1995 ★★ DONOHO-JOHNSTONE — SureShrink (THIS PAPER)
                            ↳ level-dependent SURE threshold
                            ↳ Besov-scale adaptive minimax
1995 ─── Donoho-Johnstone-Kerkyacharian-Picard — Wavelet Shrinkage (asymptopia?)
                            ↳ companion paper, Besov-body theory
2000 ─── Chang-Yu-Vetterli — BayesShrink (paper #3)
                            ↳ Laplace prior gives data-driven threshold
2002 ─── Pesquet-Krim — SURE-LET via wavelet packet
2007 ─── Blu-Luisier — SURE-LET image denoising (orthonormal expansion)
2018+ ─ Stein's identity reborn in deep self-supervised denoising
                            ↳ Noise2Noise, SURE-based losses
```

이 논문은 **'wavelet thresholding의 데이터 적응 시대'를 연다**. paper #1이 *고정* universal/minimax threshold로 시작했다면, SureShrink는 *데이터로 정한* level-dependent threshold로 도약한다 — BayesShrink, SURE-LET, 현대 self-supervised denoiser까지 이어지는 계보의 출발점.

This paper opens the **data-adaptive era of wavelet thresholding**, leading directly to BayesShrink, SURE-LET, and modern self-supervised denoising losses.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Stein (1981)** *Annals of Statistics 9, 1135–1151* | Provenance of SURE | The unbiased risk identity (Eq. 10) is the entire foundation for SureShrink's threshold choice. |
| **Donoho-Johnstone (1994)** *Biometrika* (paper #1) | VisuShrink/RiskShrink | SureShrink is a direct successor — same DWT, same soft-threshold, but data-adaptive level-dependent thresholds. |
| **Donoho-Johnstone-Kerkyacharian-Picard (1995)** | Besov-body asymptopia | Companion paper providing the function-space framework underlying Theorem 1. |
| **Cohen-Daubechies-Vial (1993)** | Boundary-corrected DWT | Used in §2.1 for handling $[0,1]$ intervals. |
| **Chang-Yu-Vetterli (2000)** *IEEE TIP* (paper #3) | BayesShrink | Replaces SURE objective with Bayesian risk under Laplace prior; level-dependent in same spirit. |
| **Donoho (1995)** *IEEE TIT* | Companion: De-noising by soft-thresholding | Theoretical risk inequalities in same model. |
| **Blu & Luisier (2007)** *IEEE TIP* | SURE-LET | Explicitly minimises SURE over a *linear expansion of thresholding functions* — a direct generalisation. |
| **Lehtinen+ (2018)** *Noise2Noise* | Self-supervised deep denoising | The Stein identity Eq. (10) reappears as the theoretical basis for training without clean targets. |
| **Dabov+ (2007)** *IEEE TIP* (paper #7) | BM3D | Uses both hard and Wiener (≈ SURE-style) shrinkage in 3D transform domain. |

---

## 7. References / 참고문헌

- Cohen, A., Daubechies, I., Jawerth, B., & Vial, P., "Multiresolution analysis, wavelets, and fast algorithms on an interval", *C.R. Acad. Sci. Paris Sér. I*, 316, 417–421 (1993).
- Donoho, D. L., & Johnstone, I. M., "Ideal spatial adaptation by wavelet shrinkage", *Biometrika*, 81(3), 425–455 (1994a).
- Donoho, D. L., & Johnstone, I. M., "Adapting to unknown smoothness via wavelet shrinkage", *J. American Statistical Association*, 90(432), 1200–1224 (1995). [DOI: 10.1080/01621459.1995.10476626]
- Donoho, D. L., Johnstone, I. M., Kerkyacharian, G., & Picard, D., "Wavelet shrinkage: asymptopia?", *J. Royal Statistical Society B*, 57(2), 301–369 (1995).
- Mallat, S., "A theory for multiresolution signal decomposition: the wavelet representation", *IEEE PAMI*, 11, 674–693 (1989a, b).
- Nemirovskii, A., "Nonparametric estimation of smooth regression functions", *Soviet Journal of Computer and System Sciences*, 23, 1–11 (1985).
- Stein, C., "Estimation of the mean of a multivariate normal distribution", *Annals of Statistics*, 9, 1135–1151 (1981).
- Stone, C. J., "Optimal global rates of convergence for nonparametric regression", *Annals of Statistics*, 10, 1040–1053 (1982).
- Triebel, H., *Theory of Function Spaces II*, Birkhäuser (1992).
