---
title: "Ideal Spatial Adaptation by Wavelet Shrinkage"
authors: David L. Donoho, Iain M. Johnstone
year: 1994
journal: "Biometrika 81(3), pp. 425–455"
doi: "10.1093/biomet/81.3.425"
topic: Low-SNR Imaging / Wavelet Denoising
tags: [wavelet, soft-thresholding, hard-thresholding, visushrink, riskshrink, universal-threshold, oracle-inequality, minimax, spatial-adaptation, donoho-johnstone]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 1. Ideal Spatial Adaptation by Wavelet Shrinkage / 웨이블릿 수축에 의한 이상적 공간 적응

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 잡음이 섞인 함수 $y_i = f(t_i) + \sigma z_i$를 추정할 때, **웨이블릿 영역에서의 좌표별 비선형 임계화(soft thresholding)** 가 모든 함수에 대해 — 어떤 oracle도 사전에 알지 못한 채 — $2\log n$ 인자 이내의 거의 이상적 공간 적응 성능을 제공함을 증명한다. 핵심 기여는 네 가지다:
(i) **Oracle inequality (Theorem 1)**: 임계값 $\lambda = \varepsilon\sqrt{2\log n}$를 사용한 soft-threshold 추정량 $\hat\theta^* = \eta_S(w, \lambda)$이 모든 $\theta \in \mathbb{R}^n$에 대해 $R(\hat\theta^*, \theta) \le (2\log n + 1)\{\varepsilon^2 + \sum \min(\theta_i^2, \varepsilon^2)\}$를 만족.
(ii) **Universal threshold**: $\lambda^u_n = \sigma\sqrt{2\log n}$는 $\Pr(\max_i |z_i| > \sqrt{2\log n}) \to 0$이라는 가우시안 극단값 사실에서 유도. **VisuShrink**는 이 임계값을 사용한 soft thresholding 추정기.
(iii) **Minimax 최적성 (Theorem 2-3)**: 본 임계 추정기 클래스 안에서 $2\log n$ 인자는 본질적으로 개선 불가능 — $2 - \varepsilon$으로 줄이면 oracle inequality가 깨짐.
(iv) **실용적 수렴률**: 함수 $f$가 조각 다항식, 변지점 스플라인, BV(bounded variation), Hölder 클래스 등 매우 일반적인 부드러움 클래스에 속할 때, RiskShrink/VisuShrink는 $\log^2 n \cdot n^{-r}$의 수렴률을 달성 — 비적응적 선형 추정기의 $n^{-1/2}$ 한계를 결정적으로 돌파.

### English
This paper proves that **coordinatewise nonlinear thresholding (soft thresholding) of empirical wavelet coefficients** mimics the performance of an oracle for spatially-adaptive estimation to within a logarithmic factor $2\log n$, uniformly in the unknown function $f$. Four key contributions:
(i) **Oracle inequality (Theorem 1)**: The soft-threshold estimator $\hat\theta^* = \eta_S(w, \varepsilon\sqrt{2\log n})$ satisfies $R(\hat\theta^*, \theta) \le (2\log n + 1)\{\varepsilon^2 + \sum_i \min(\theta_i^2, \varepsilon^2)\}$ for all $\theta \in \mathbb{R}^n$.
(ii) **Universal threshold**: $\lambda^u_n = \sigma\sqrt{2\log n}$, motivated by $\Pr(\max_i |z_i| > \sqrt{2\log n}) \to 0$ for white noise. **VisuShrink** is the resulting soft-thresholding estimator with this universal threshold (no shrinkage at coarsest levels $j < j_0$).
(iii) **Minimax optimality (Theorems 2-3)**: The $2\log n$ factor cannot be improved — replacing $2$ with $2-\varepsilon$ breaks the oracle inequality.
(iv) **Practical convergence rates**: For piecewise polynomials, variable-knot splines, BV functions, and Hölder classes, RiskShrink/VisuShrink attain $\log^2 n \cdot n^{-r}$ rates, decisively breaking the $n^{-1/2}$ barrier of nonadaptive linear smoothers.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1.1-1.4 Setup and Spatially Adaptive Methods / 설정과 공간 적응 방법

#### 한국어
- **모델**: $y_i = f(t_i) + e_i$, $t_i = i/n$, $e_i \overset{iid}{\sim} N(0, \sigma^2)$. 위험은 $R(\hat f, f) = n^{-1} E\|\hat f - f\|^2_{2,n}$.
- **Spatial adaptation**: $\hat f = T(y, d(y))$로 형식화 — 추정 공식 $T$와 데이터 적응적 파라미터 $d(y)$. CART, Turbo, MARS, 변폭 커널이 모두 이 틀에 들어감.
- **Five examples** (1.2): 조각 상수, 조각 다항식, 변지점 스플라인, 변폭 커널, 변폭 고차 커널.
- **Ideal adaptation with oracles** (1.3): 진실 $f$를 아는 oracle이 최적 $\delta$를 알려준다고 가정 → ideal risk $\mathcal R_{n,\sigma}(T, f) = \inf_\delta R(T(y,\delta), f)$. 이는 도달 불가능한 목표지만 비교 기준.
- **Selective wavelet reconstruction** (1.4): 직교 웨이블릿 변환 $\mathcal W$에서 $w = \mathcal W y$, $w_{j,k} = \theta_{j,k} + \sigma z_{j,k}$. 
  $T_{\rm SW}(y, \delta) = \sum_{(j,k) \in \delta} w_{j,k} W_{j,k}$ — 일부 계수만 살리고 나머지 0. Oracle은 $\delta^* = \{(j,k): \theta_{j,k} \ne 0\}$. 조각 다항식은 $\#\{\theta_{j,k} \ne 0\} \le 2^{j_0} + (J+1-j_0)(2S+1)L$ 만족.
- **Test functions**: Blocks, Bumps, HeaviSine, Doppler — 모두 $n = 2048$, 구간 $[0,1]$. SNR = $\|f\|/\|\sigma z\| = 7$.

#### English
- Model: $y_i = f(t_i) + e_i$ with iid $N(0, \sigma^2)$ noise; risk $n^{-1} E\|\hat f - f\|^2$.
- Spatial-adaptive estimation framework: $\hat f = T(y, d(y))$ — fixed reconstruction formula plus data-adaptive parameter.
- Five reconstruction families illustrate the framework.
- Ideal-oracle benchmark: $\mathcal R_{n,\sigma}(T, f)$ is the infimum risk attainable when the oracle reveals the optimal $\delta$.
- Selective wavelet reconstruction: keep only a subset of coefficients.
- Four test functions Blocks/Bumps/HeaviSine/Doppler at $n = 2048$, SNR 7.

---

### Part II: §2 Decision Theory & Spatial Adaptation / 결정 이론과 공간 적응

#### 한국어 — Theorem 1 (Oracle inequality)

다변량 정규 모델 $w_i = \theta_i + \varepsilon z_i$, $z_i \sim N(0,1)$에서 다음 두 비선형성을 정의:

$$
\eta_H(w, \lambda) = w \cdot \mathbf 1\{|w| > \lambda\} \quad \text{(hard, Eq. 11)}
$$
$$
\eta_S(w, \lambda) = \mathrm{sgn}(w)(|w| - \lambda)_+ \quad \text{(soft, Eq. 12)}
$$
**Theorem 1**: 추정량 $\hat\theta^*_i = \eta_S(w_i, \varepsilon\sqrt{2\log n})$는
$$
E\|\hat\theta^* - \theta\|^2_{2,n} \le (2\log n + 1)\left\{\varepsilon^2 + \sum_{i=1}^n \min(\theta_i^2, \varepsilon^2)\right\} \quad (13)
$$
모든 $\theta \in \mathbb R^n$에 대해 성립.

**해석**: $\sum \min(\theta_i^2, \varepsilon^2)$은 *projection oracle*의 ideal risk (각 좌표마다 $\theta_i^2$ 또는 $\varepsilon^2$ 중 작은 것 선택). 즉 oracle을 $2\log n$ 인자 이내로 모방 가능.

#### English — Theorem 1
Hard threshold $\eta_H$ (Eq. 11) and soft threshold $\eta_S$ (Eq. 12). The soft-threshold estimator with threshold $\varepsilon\sqrt{2\log n}$ satisfies the oracle inequality (13). The right-hand side is essentially the projection oracle's risk plus an extra $\varepsilon^2$ term. So the oracle is mimicked to within $2\log n$.

#### 한국어 — Theorem 2 (Minimax threshold $\lambda^*_n$) and Table 2

$$
\Lambda^*_n = \inf_\lambda \sup_\mu \frac{\rho_{ST}(\lambda, \mu)}{n^{-1} + \min(\mu^2, 1)}, \qquad \lambda^*_n = \arg\min
$$
$\rho_{ST}(\lambda, \mu) = E[\eta_S(\mu + Z, \lambda) - \mu]^2$는 단일 좌표 soft-threshold MSE. 작은 $n$에서 $\lambda^*_n$은 universal $\sqrt{2\log n}$보다 훨씬 작음 (Table 2: $n=256 \to \lambda^*_n = 1.86$ vs $\sqrt{2\log 256} = 3.33$).

#### English — Theorem 2 / Table 2
Minimax threshold $\lambda^*_n$ is much smaller than universal $\sqrt{2\log n}$ for moderate $n$ — e.g., $\lambda^*_n \approx 1.86$ at $n=256$ versus universal $3.33$. RiskShrink uses $\lambda^*_n$; VisuShrink uses universal $\sqrt{2\log n}$.

#### 한국어 — Theorem 3 (Lower bound)
어떤 측정 가능 추정기 $\hat\theta$도 $2\log n$을 $2 - \varepsilon$로 개선할 수 없음. 즉 $2\log n$은 본질적 한계.

#### English — Theorem 3 (Lower bound)
No measurable estimator can replace $2\log n$ by anything smaller than $(2-\varepsilon)\log n$ — the constant 2 is sharp.

#### 한국어 — §2.3-2.4 RiskShrink Definition

$$
\tilde\theta^*_{j,k} = \begin{cases} w_{j,k} & j < j_0 \\ \eta_S(w_{j,k}, \lambda^*_n \sigma) & j_0 \le j \le J \end{cases}
$$
$$
\tilde f^*_n = \mathcal W^T \circ \tilde\theta^* \circ \mathcal W
$$
조밀 스케일 $j < j_0$에서는 수축 안함 (vanishing moment 부재로 $\theta_{j,k}$가 0이 아님).

**Corollary 1**: $R(\tilde f^*_n, f) \le \Lambda^*_n \{\sigma^2/n + \mathcal R_{n,\sigma}(\mathrm{sw}, f)\}$. VisuShrink는 $\Lambda^*_n$ 자리에 $2\log n + 1$를 넣은 형태.

#### English — RiskShrink (§2.3-2.4)
Soft-threshold all detail levels $j_0 \le j \le J$ with threshold $\lambda^*_n \sigma$; leave coarse coefficients $j < j_0$ untouched. Inverse-transform.

---

### Part III: §3 Wavelets vs Piecewise Polynomials / 웨이블릿 vs 조각 다항식

#### 한국어 — Theorem 5
$\mathcal R_{n,\sigma}(\mathrm{sw}, f) \le (C_1 + C_2 J)\mathcal R_{n,\sigma}(\mathrm{PP}(D), f)$ — 웨이블릿 selection이 조각 다항식 fits만큼 강력함을 $\log n$ factor 이내로 보장. 즉 조각 다항식보다 *본질적으로 더 강하지는 않지만 못하지도 않다* — 더 깔끔한 알고리즘으로 같은 결과를 얻음.

#### English
Wavelet oracle is no worse than piecewise-polynomial oracle within a $\log n$ factor. Wavelets are not strictly more powerful, but achieve the same with a cleaner $O(n)$ algorithm and no need to optimise partitions.

---

### Part IV: §4 Discussion and VisuShrink / 토의와 VisuShrink

#### 한국어 — §4.2 VisuShrink (Definition 2)

$$
\check\theta^v_{j,k} = \begin{cases} w_{j,k} & j < j_0 \\ \eta_S(w_{j,k}, \sigma\sqrt{2\log n}) & j_0 \le j \le J \end{cases}, \qquad \check f^v_n = \mathcal W^T \circ \check\theta^v \circ \mathcal W
$$
**핵심 동기**: 가우시안 백색잡음의 *극단값* 성질. $z_i \overset{iid}{\sim} N(0,1)$일 때 $n \to \infty$에서
$$
\Pr\left(\max_i |z_i| > \sqrt{2\log n}\right) \to 0 \quad (31)
$$
따라서 임계값 $\sigma\sqrt{2\log n}$ 위로 올라오는 잡음 계수는 거의 없음 → 진짜 영(zero) 계수는 *높은 확률로 정확히 영*으로 추정. **시각적 효과**: "noise-free" 외관, 작은 fluctuations 제거.

**RiskShrink vs VisuShrink trade-off**:
- RiskShrink: MSE 측면 우수 (작은 $\lambda^*_n$로 진폭 적게 잘림).
- VisuShrink: 시각 측면 우수 (큰 $\sqrt{2\log n}$로 작은 noise blip 완전 제거).

#### English — VisuShrink (Definition 2)
Apply soft thresholding with threshold $\sigma\sqrt{2\log n}$ at all detail levels (keep coarse coefficients). Motivated by Eq. (31): the maximum of $n$ iid $N(0,1)$ noise samples is asymptotically below $\sqrt{2\log n}$. With high probability, every coefficient that should be zero in the noiseless case is exactly estimated as zero — the "noise-free" visual character (Fig. 9).

#### 한국어 — Noise level estimation
$$
\hat\sigma = \mathrm{MAD}\bigl(\{w_{J, k}\}_k\bigr)\bigm/ 0.6745
$$
가장 미세 스케일 $j = J$의 웨이블릿 계수는 거의 모두 잡음만 — 대부분의 신호 에너지는 거친 스케일에 집중. MAD = median absolute deviation. 0.6745는 가우시안 하에서 MAD를 표준편차로 변환하는 인자($\Phi^{-1}(0.75)$).

#### English — Noise level estimation
$\hat\sigma$ is robust-estimated from the median absolute deviation of finest-scale wavelet coefficients, divided by 0.6745. Robust against signal contamination at $j=J$.

#### 한국어 — §4.6 Boundary correction
주기 경계는 $[0,1]$에 적합하지 않음. Cohen et al. (1993)의 boundary-corrected wavelets: $\mathcal W = U \circ P$, $P$는 양 끝 $N+1$, $N$ 샘플의 preconditioning 변환. 결과적으로 모든 oracle inequality가 상수만 약간 바뀐 형태로 유지.

#### English — Boundary correction (§4.6)
Periodic wavelets misbehave at boundaries; replace with Cohen-Daubechies-Vial boundary-corrected wavelets. Risk inequalities hold with slightly different constants.

---

### Part V: §4.4 Numerical Performance / 수치 성능

#### 한국어
Table 3 ($n=2048$, 단일 실현):
| Method | Blocks | Bumps | HeaviSine | Doppler |
|---|---|---|---|---|
| Noisy data | 1.047 | 0.937 | 1.008 | 0.9998 |
| Ideal wavelet | 0.097 | 0.111 | 0.028 | 0.042 |
| Ideal Fourier | 0.370 | 0.375 | 0.062 | 0.200 |
| RiskShrink ($\lambda^*_n$) | 0.395 | 0.496 | 0.059 | 0.152 |
| VisuShrink ($\sqrt{2\log n}$) | 0.874 | 1.058 | 0.076 | 0.324 |

**관찰**:
1. Ideal wavelet은 ideal Fourier보다 4-9배 우수 → 웨이블릿이 공간 변동성 함수에 본질적으로 강함.
2. RiskShrink는 ideal wavelet의 약 4-7배 위 — Theorem 2에서 예측한 $\Lambda^*_n \approx 6.8$ 인자와 일치.
3. VisuShrink의 MSE는 RiskShrink보다 나쁘지만, *시각* 품질은 더 우수 (이론에서 예측).

Table 4 ($n = 256, 512, ..., 8192$에서 10 replications): 부드러운 신호(HeaviSine, Doppler)는 $n$ 증가에 따라 MSE 감소가 가파름. Blocks/Bumps는 더 천천히 감소.

#### English
Table 3 / Table 4 confirm: (i) wavelets dominate Fourier; (ii) RiskShrink within ~6.8× of ideal as predicted by $\Lambda^*_n$; (iii) VisuShrink trades MSE for visual smoothness.

---

## 3. Key Takeaways / 핵심 시사점

1. **임계화는 oracle을 $2\log n$ 이내로 모방한다 / Thresholding mimics the oracle within $2\log n$** — Theorem 1의 식 (13)이 이 논문의 결정적 결과. $R \le (2\log n + 1)\{\varepsilon^2 + \sum \min(\theta^2, \varepsilon^2)\}$는 모든 $\theta$에 대해 성립하므로 함수 $f$에 대한 어떤 가정도 필요 없음. 어떤 데이터가 와도 똑같은 임계기가 (거의) 최적.
   The single inequality (13) characterises the entire paper. It is *uniform in $\theta$* — no smoothness assumption — so a single procedure handles every function.

2. **Universal threshold $\sigma\sqrt{2\log n}$는 가우시안 극단값에서 나온다 / Universal threshold from Gaussian extremes** — $\Pr(\max_i |z_i| > \sqrt{2\log n}) \to 0$이라는 단순한 사실. 따라서 *진짜 영* 계수는 *높은 확률로 정확히 영* → "noise-free" 시각 효과.
   Eq. (31): the maximum of $n$ iid standard normals stays below $\sqrt{2\log n}$ asymptotically; choosing this threshold ensures noise coefficients are killed with high probability.

3. **Hard vs soft thresholding의 trade-off / Hard vs soft threshold trade-off** — 두 임계기 모두 같은 점근 oracle inequality ($\sim 2\log n$ 인자) 만족 (Theorem 4). 차이: hard는 영 근처 불연속 → MSE 위로 들쭉날쭉; soft는 연속이며 진폭을 $\lambda$만큼 *bias* 시킴. 시각적으로는 hard가 sharp edge 보존, soft가 smoother. RiskShrink/VisuShrink는 모두 soft를 선택.
   Both hard and soft achieve the $2\log n$ factor; soft is favoured for continuity and stable risk, at the cost of a $\lambda$-sized bias on each retained coefficient.

4. **Coarse 스케일은 수축하지 않는다 / Coarsest levels are not shrunk** — $j < j_0$의 scaling-function 계수는 vanishing moment가 없어 영 근방으로 클러스터링되지 않음. 일률적으로 임계화하면 신호의 평균/저주파를 깎아냄. 그래서 *detail levels만 임계화*가 표준.
   Scaling-function coefficients at the coarsest scales $j < j_0$ carry low-frequency content and vanishing moments don't apply; thresholding them would erode the signal's mean and large-scale structure.

5. **Noise-level의 robust estimation은 MAD/0.6745 / Robust noise estimation via MAD/0.6745** — 가장 미세한 스케일 $j=J$에서 신호의 vanishing-moment 효과로 거의 잡음만. 그래서 MAD가 robust한 $\sigma$ 추정 → 이상치(즉, 이 스케일에 누설된 신호)에 둔감.
   At the finest scale, signal coefficients are sparse (because of vanishing moments), so MAD/0.6745 robustly estimates the noise standard deviation.

6. **Wavelets 이 piecewise polynomial fits만큼 강하다 / Wavelets are as strong as piecewise polynomial fits** (§3) — Theorem 5: $\mathcal R(\mathrm{sw}, f) \le (C_1 + C_2 J) \mathcal R(\mathrm{PP}(D), f)$. 즉 매개변수 검색·partition 최적화 같은 어려운 문제 없이, 단순한 직교 변환 + 좌표별 임계화가 같은 일을 함 — $O(n)$ 알고리즘.
   Wavelet thresholding achieves what piecewise polynomial fitting does, but with a clean $O(n)$ algorithm and no combinatorial partition search.

7. **$\log^2 n \cdot n^{-r}$ 수렴률은 비적응 선형의 $n^{-1/2}$ 한계를 깬다 / Adaptive rates beat the linear barrier** — 불연속이 있는 함수에서 비적응 선형 추정기는 $n^{-1/2}$에 갇힘 ($L = $ 조각 수). 웨이블릿 임계화는 $\log^2 n \cdot n^{-1}$까지 도달 가능 (BV/Hölder 클래스 등).
   Adaptive thresholding breaks the $n^{-1/2}$ barrier of nonadaptive linear smoothers on non-smooth functions, attaining nearly parametric rates up to log factors.

8. **$2\log n$ 인자는 본질적 한계 / The $2\log n$ factor is essentially sharp** — Theorem 3은 어떤 측정 가능 추정기 시퀀스도 $2\log n$을 $(2-\varepsilon)\log n$으로 줄이지 못함을 증명. 따라서 RiskShrink/VisuShrink는 *상수* 차원에서만 개선 가능, *오더* 차원에서는 최적.
   Theorem 3 proves the constant 2 cannot be improved — $(2-\varepsilon)\log n$ is unattainable for any estimator. Subsequent work (SureShrink, BayesShrink) only improves the constant.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Setting / 설정
$$
y_i = f(t_i) + \sigma z_i, \quad t_i = i/n, \quad z_i \overset{iid}{\sim} N(0,1), \quad i = 1, \ldots, n = 2^{J+1}
$$
$$
R(\hat f, f) = n^{-1} E\|\hat f - f\|^2_{2,n}
$$
### 4.2 Discrete wavelet transform / 이산 웨이블릿 변환
$$
w = \mathcal W y, \qquad y = \mathcal W^T w, \qquad w_{j,k} = \theta_{j,k} + \sigma z_{j,k} \quad (z_{j,k} \overset{iid}{\sim} N(0,1))
$$
**Parseval**: $\|f - \hat f\|^2_{2,n} = \|\theta - \hat\theta\|^2_{2,n}$ → 시간 영역 risk = 웨이블릿 영역 risk.

### 4.3 Thresholding nonlinearities / 임계화 비선형성
$$
\eta_H(w, \lambda) = w \cdot \mathbf 1\{|w| > \lambda\}, \qquad \eta_S(w, \lambda) = \mathrm{sgn}(w) (|w| - \lambda)_+
$$
### 4.4 Universal (VisuShrink) and minimax (RiskShrink) thresholds
$$
\lambda^u_n = \sigma \sqrt{2\log n} \quad \text{(VisuShrink)}
$$
$$
\Lambda^*_n = \inf_\lambda \sup_\mu \frac{\rho_{ST}(\lambda, \mu)}{n^{-1} + \min(\mu^2, 1)}, \qquad \lambda^*_n = \mathrm{arg\,min} \quad \text{(RiskShrink)}
$$
with $\rho_{ST}(\lambda, \mu) = E[\eta_S(\mu + Z, \lambda) - \mu]^2$.

### 4.5 Oracle inequality (Theorem 1) / 오라클 부등식
$$
\boxed{\;E\|\hat\theta^* - \theta\|^2 \le (2\log n + 1)\left\{\varepsilon^2 + \sum_{i=1}^n \min(\theta_i^2, \varepsilon^2)\right\}\;}
$$
where $\hat\theta^*_i = \eta_S(w_i, \varepsilon\sqrt{2\log n})$ and projection oracle risk is $\sum_i \min(\theta_i^2, \varepsilon^2)$.

### 4.6 VisuShrink / RiskShrink algorithm
**Input**: noisy $y \in \mathbb R^n$, wavelet basis (Symmlet $N=8$ recommended), $j_0$ (e.g. 5).
**Step 1**: $w = \mathcal W y$ (forward DWT).
**Step 2**: Estimate noise: $\hat\sigma = \mathrm{median}\{|w_{J,k}|\}_k / 0.6745$.
**Step 3**: For $j_0 \le j \le J$, $0 \le k < 2^j$:
$$
\hat\theta_{j,k} = \begin{cases}
\eta_S(w_{j,k}, \hat\sigma \sqrt{2\log n}) & \text{(VisuShrink)} \\
\eta_S(w_{j,k}, \hat\sigma \lambda^*_n) & \text{(RiskShrink)}
\end{cases}
$$
For $j < j_0$, keep $\hat\theta_{j,k} = w_{j,k}$.
**Step 4**: $\hat f = \mathcal W^T \hat\theta$ (inverse DWT).

### 4.7 Worked numerical example / 수치 예시
For $n = 2048$, $\sigma = 1$:
- $\sqrt{2\log n} = \sqrt{2 \times 7.625} = 3.905$
- VisuShrink threshold $\lambda^u = \sigma \cdot 3.905 = 3.905$
- RiskShrink $\lambda^*_n = 2.414$ (Table 2) → smaller threshold, less aggressive shrinkage.

For HeaviSine (smooth) at $n=2048$, VisuShrink risk $\approx 0.076$, Ideal-wavelet risk $\approx 0.028$. Ratio $0.076/0.028 \approx 2.7$ — much better than the worst-case bound $(2\log 2048 + 1)\sigma^2/n + 1 \approx 16$ suggests.

### 4.8 Risk bound for VisuShrink (from Theorem 7 / Corollary)
$$
R(\check f^v_n, f) \le (2\log n + 1) \left\{\frac{\sigma^2}{n} + \mathcal R_{n,\sigma}(\check{\mathrm{sw}}, f)\right\}
$$
### 4.9 Boundary-corrected DWT / 경계 보정
Replace plain orthogonal $\mathcal W$ with $\mathcal W = U \circ P$ where $P$ is a $2(N+1)$-block preconditioning matrix. All oracle inequalities then hold with constants depending only on the smallest/largest singular values of $P$ — independent of $n$.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1909 ─── Haar — first orthonormal "wavelet" basis (square pulses)
1984 ─── Efroimovich-Pinsker — Fourier-based adaptive estimation (USSR)
1986 ─── Daubechies — compactly supported smooth orthonormal wavelets
1988 ─── Mallat — multiresolution analysis (MRA), pyramid / cascade algorithm
1989-92 ─ Daubechies, Meyer, Coifman — wavelet theory matures (books)
1993 ─── Cohen-Daubechies-Vial — boundary-corrected wavelets on [0,1]
1994 ★★ DONOHO-JOHNSTONE: Ideal Spatial Adaptation by Wavelet Shrinkage
                          ↳ VisuShrink, RiskShrink, oracle inequality
1995 ─── Donoho-Johnstone — Adapting to Unknown Smoothness (SureShrink)
                          ↳ level-dependent SURE-optimal threshold
1995 ─── Donoho — De-Noising by Soft-Thresholding (TIT 41(3))
2000 ─── Chang-Yu-Vetterli — BayesShrink, Laplace prior, Bayesian threshold
2002 ─── Sendur-Selesnick — Bivariate shrinkage with parent coefficient
2005 ─── Buades-Coll-Morel — Non-Local Means (departs from transform domain)
2007 ─── Dabov-Foi-Katkovnik-Egiazarian — BM3D (returns to transform domain
                                          + nonlocal block matching)
2012+ ── Deep-learning denoisers (DnCNN, FFDNet, Restormer ...)
                          ↳ wavelet shrinkage rediscovered as a layer
                            (e.g. wavelet-CNN hybrids).
```

이 논문은 **wavelet denoising의 시작점**이자, oracle inequality라는 통계적 기법을 신호처리에 도입한 분기점이다. SureShrink, BayesShrink, BM3D는 이 임계기 프레임에 점진적 개선을 더한다 (level-dependent threshold, prior, nonlocal grouping).

This paper is **the founding point of wavelet denoising** and introduces the oracle-inequality framework into signal processing. SureShrink, BayesShrink, BM3D refine this template (level-dependent thresholds, priors, nonlocal grouping).

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Daubechies (1988, 1992)** *Comm. Pure Appl. Math.* | Compactly supported orthonormal wavelets | Provides the wavelet basis used; vanishing moments are essential for sparsity. |
| **Cohen-Daubechies-Vial (1993)** | Boundary-corrected wavelets | The CDV preconditioning $P$ used in §4.6 to handle $[0,1]$ boundaries. |
| **Bickel (1983)** | Soft-thresholding in mvariate normal decision theory | Key precedent for using soft thresholding as a near-oracle. |
| **Efroimovich-Pinsker (1984)** | Fourier-domain adaptive estimation | Russian school's parallel approach; this paper shows wavelets dominate Fourier on non-smooth functions. |
| **Donoho-Johnstone (1995)** *JASA* (#2 in this topic) | SureShrink follow-up | Replaces universal threshold with level-dependent SURE-optimal; this paper's RiskShrink minimax threshold is the precursor. |
| **Donoho (1995)** *IEEE TIT* | "De-noising by soft-thresholding" | Companion to this paper, focusing on theoretical risk bounds and Besov-class minimax rates. |
| **Chang-Yu-Vetterli (2000)** *IEEE TIP* (#3) | BayesShrink | Replaces universal threshold with data-driven Bayesian threshold under Laplace prior on coefficients. |
| **Buades-Coll-Morel (2005)** (#4) | Non-Local Means | Moves away from transform-domain thresholding to spatial-domain self-similarity averaging. |
| **Dabov+ (2007)** *IEEE TIP* (#7) | BM3D | Combines transform-domain shrinkage (this paper's heritage) with nonlocal block matching (NLM's heritage). |
| **Mallat (1989)** *IEEE PAMI* | MRA & cascade algorithm | Provides the $O(n)$ DWT algorithm essential to RiskShrink/VisuShrink's practical efficiency. |

---

## 7. References / 참고문헌

- Bickel, P. J., "Minimax estimation of the mean of a normal distribution when the parameter space is restricted", *Annals of Statistics*, 9, 1301–1309 (1981).
- Cohen, A., Daubechies, I., & Vial, P., "Wavelets on the interval and fast wavelet transforms", *Applied and Computational Harmonic Analysis*, 1, 54–81 (1993).
- Daubechies, I., "Orthonormal bases of compactly supported wavelets", *Communications on Pure and Applied Mathematics*, 41, 909–996 (1988).
- Daubechies, I., *Ten Lectures on Wavelets*, SIAM, Philadelphia (1992).
- Donoho, D. L., & Johnstone, I. M., "Ideal spatial adaptation by wavelet shrinkage", *Biometrika*, 81(3), 425–455 (1994). [DOI: 10.1093/biomet/81.3.425]
- Donoho, D. L., "De-noising by soft-thresholding", *IEEE Transactions on Information Theory*, 41(3), 613–627 (1995).
- Donoho, D. L., & Johnstone, I. M., "Adapting to unknown smoothness via wavelet shrinkage", *J. American Statistical Association*, 90(432), 1200–1224 (1995).
- Efroimovich, S. Yu., & Pinsker, M. S., "A learning algorithm for nonparametric filtering", *Avtomatika i Telemekhanika*, 11, 58–65 (1984).
- Mallat, S., "A theory for multiresolution signal decomposition: the wavelet representation", *IEEE PAMI*, 11, 674–693 (1989).
- Meyer, Y., *Ondelettes et Opérateurs* (3 vols), Hermann (1990).
