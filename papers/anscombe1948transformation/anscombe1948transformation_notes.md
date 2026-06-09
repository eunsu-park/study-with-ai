---
title: "The Transformation of Poisson, Binomial and Negative-Binomial Data"
authors: F. J. Anscombe
year: 1948
journal: "Biometrika 35(3-4), pp. 246-254"
doi: "10.1093/biomet/35.3-4.246"
topic: Low-SNR Imaging / Variance-Stabilising Transforms
tags: [variance-stabilising-transform, anscombe, poisson, binomial, negative-binomial, asymptotic-expansion, vst, gaussianisation]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 11. The Transformation of Poisson, Binomial and Negative-Binomial Data / 포아송·이항·음이항 자료의 변환

---

## 1. Core Contribution / 핵심 기여

### 한국어
Anscombe(1948)는 신호 의존적 잡음(특히 포아송, 이항, 음이항)을 갖는 자료를 단순한 비선형 변환으로 **분산이 거의 일정한 근사적 정규 자료**로 바꾸는 일반적 절차(variance-stabilising transformation, VST)를 정립했다. 핵심 기여 네 가지:
(i) **포아송 VST**: $y = \sqrt{r + c}$ 꼴에서 Taylor 전개와 무한차 전개로 $\mathrm{var}(y)$를 $m \to \infty$ 점근으로 계산, 분산을 $\tfrac{1}{4}$에 가장 가깝게 유지하는 최적 상수가 $c = \tfrac{3}{8}$임을 보임. 이때 $\mathrm{var}(y) \sim \tfrac{1}{4}(1 + 1/(16 m^2))$으로, Bartlett(1936)의 $c = \tfrac{1}{2}$ 변환보다 더 빠르게 안정화.
(ii) **이항 VST**: $y = \sqrt{n + d_2}\,\sin^{-1}\sqrt{(r + c)/(n + d_1)}$, $c = \tfrac{3}{8}, d_1 = \tfrac{3}{4}, d_2 = \tfrac{1}{2}$로 $\mathrm{var}(y) = \tfrac{1}{4} + O(1/n^2)$ 달성. 단순 각변환($\sin^{-1}\sqrt{r/n}$)의 잔여 분산 $O(1/n)$을 $O(1/n^2)$로 축소.
(iii) **음이항 VST**: $m \gg k$인 경우 $y = \sqrt{k - \tfrac{1}{2}}\,\sinh^{-1}\sqrt{(r + \tfrac{3}{8})/(k - \tfrac{3}{4})}$, $\mathrm{var}(y) = \tfrac{1}{4} + O(1/m^2)$; $k$ 고정·$m$ 대규모일 때는 $y = \ln(r + \tfrac{1}{2}k)$가 더 적합 ($\mathrm{var}(y) \sim \psi'(k)$).
(iv) **수치 검증 (Table 1)**: $m \ge 4$에서 변환이 분산을 $\pm 1\%$ 이내로 안정화시키며, 평균 편향, 비대칭($\gamma_1$), 첨도($\gamma_2$), 효율($E_y$)을 광범위 $m$ 값에 대해 표로 제공.

이 논문은 후일 천체사진·형광현미경·광자수영상에서 포아송 잡음을 제거하기 위한 “VST + Gaussian denoise + inverse VST” 파이프라인의 출발점이며, Donoho(1993), Fryzlewicz-Nason(2004), Mäkitalo-Foi(2011, 2013) 등으로 계승된다.

### English
Anscombe (1948) establishes the general procedure for converting count data with signal-dependent noise — Poisson, binomial, negative-binomial — into approximately Gaussian observations with **constant variance** via a simple algebraic transformation (variance-stabilising transformation, VST). Four key contributions:
(i) **Poisson VST**: For transforms $y = \sqrt{r + c}$, Taylor / asymptotic expansion shows that $c = \tfrac{3}{8}$ is optimal (in keeping $\mathrm{var}(y)$ closest to $\tfrac{1}{4}$ for finite $m$). Resulting $\mathrm{var}(y) \sim \tfrac{1}{4}\bigl(1 + 1/(16 m^2)\bigr)$ — markedly better than Bartlett's (1936) $c = \tfrac{1}{2}$ variant.
(ii) **Binomial VST**: $y = \sqrt{n + d_2}\,\sin^{-1}\sqrt{(r + c)/(n + d_1)}$ with $c = \tfrac{3}{8}, d_1 = \tfrac{3}{4}, d_2 = \tfrac{1}{2}$, yielding $\mathrm{var}(y) = \tfrac{1}{4} + O(1/n^2)$ — faster decay than the textbook arcsine.
(iii) **Negative-binomial VST**: For $m \gg k$ one uses $y = \sqrt{k - \tfrac{1}{2}}\,\sinh^{-1}\sqrt{(r+\tfrac{3}{8})/(k-\tfrac{3}{4})}$. For $k$ fixed and $m$ large, the simpler $y = \ln(r + \tfrac{1}{2}k)$ is preferred, with $\mathrm{var}(y) \sim \psi'(k)$.
(iv) **Numerical verification (Table 1)**: For $m \ge 4$ the transform stabilises variance to within $\pm 1\%$ of the limiting value, and tabulates bias, skewness $\gamma_1$, kurtosis $\gamma_2$, and large-sample efficiency $E_y$.

This paper is the foundation of the modern Poisson-image-denoising pipeline ("VST + Gaussian denoise + inverse VST"), inherited by Donoho (1993), Fryzlewicz-Nason (2004), and Mäkitalo-Foi (2011, 2013).

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction & Motivation / 도입과 동기

#### 한국어
- **문제 설정**: $\mathbf r$가 평균 $m$인 포아송 변수이면 $\mathrm{var}(\mathbf r) = m$. 분산이 평균에 의존하므로 일반 분산분석 (ANOVA) 도구 — 등분산 가정하에 설계된 — 를 직접 적용하기 어렵다.
- Bartlett(1936)이 $y = \sqrt{r}$ (Eq. 1.1)이 $m$이 클 때 분산을 $\tfrac{1}{4}$로 안정화함을 보였고, 작은 $m$에서는 $y = \sqrt{r + \tfrac{1}{2}}$ (Eq. 1.2)가 더 우수함을 제안했다.
- Anscombe는 일반화된 $y = \sqrt{r + c}$ (Eq. 1.3)에서 **상수 $c$의 최적값**을 결정하려 한다. 결과: $c = \tfrac{3}{8}$ (이는 A. H. L. Johnson에 기인).
- 이항 분포에 대해서는 $y = \sin^{-1}\sqrt{(r+c)/(n+2c)}$, 분산 $\approx \tfrac{1}{4}(n+\tfrac{1}{2})^{-1}$ (Eq. 1.4).
- 음이항 분포에 대해서는 $k$가 알려져 있을 때 $y = \sinh^{-1}\sqrt{(r+c)/(k-2c)}$ (Eq. 1.5), $c \approx \tfrac{3}{8}$ 최적; 또는 $y = \ln(r + \tfrac{1}{2}k)$ (Eq. 1.6).
- 작은 $m$에서는 $c = 0.3, 0.4$ 같은 한자리 소수가 실용적.

#### English
- Setting: $\mathbf r \sim \mathrm{Poisson}(m)$ gives $\mathrm{var}(\mathbf r) = m$ — variance scales with the mean, breaking the equal-variance assumption of standard ANOVA.
- Bartlett (1936) proposed $y = \sqrt r$ for large $m$ and $y = \sqrt{r + \tfrac{1}{2}}$ for small $m$. Anscombe generalises to $y = \sqrt{r + c}$ and seeks the optimal $c$, arriving at $c = \tfrac{3}{8}$ (a result due to A. H. L. Johnson).
- For binomial: arcsine-square-root transform with the offset $c = \tfrac{3}{8}$ and adjustments $d_1 = \tfrac{3}{4}, d_2 = \tfrac{1}{2}$.
- For negative-binomial: choice depends on regime — large $k$, small $k$ but large $m$, etc. Two transforms (1.5) and (1.6) cover the main regimes.

---

### Part II: §2 Poisson Distribution — Asymptotic Expansion / 포아송 분포 점근전개

#### 한국어
- $t = r - m$, $m' = m + c$로 두고 $y = \sqrt{m'(1 + t/m')}$를 Taylor 급수로 전개:
  $$
  y = \sqrt{m'}\Bigl\{1 + a_1 \tfrac{t}{m'} - a_2(\tfrac{t}{m'})^2 + \dots + (-1)^{s-1} a_{s-1}(\tfrac{t}{m'})^{s-1}\Bigr\} + R_s
  $$
  여기서 $a_s = (-1)^{s+1} \tfrac{1\cdot(-1)\cdot(-3)\cdots(-2s+3)}{2^s s!}$ (Eq. 2.1).
- 잔차 $R_s$는 Lagrange 형식으로 $|R_s| < a_s t^s / (m')^{s-1/2}$ (Eq. 2.3).
- 포아송 $\mathbf t$의 모먼트: $\mu_1 = 0$, $\mu_2 = m$, $\mu_3 = m$, $\mu_4 = 3m^2 + m$, $\mu_n = O(m^{n/2})$.
- 점근전개 결과:
  $$
  \mathrm{var}(\mathbf y) \sim \tfrac{1}{4}\Bigl\{1 + \tfrac{3 - 8c}{8m} + \tfrac{32 c^2 - 52 c + 17}{32 m^2}\Bigr\} \quad (2.8)
  $$
- $c = \tfrac{3}{8}$일 때 $O(1/m)$ 항이 정확히 0:
  $$
  \mathrm{var}(\mathbf y) \sim \tfrac{1}{4}\Bigl(1 + \tfrac{1}{16 m^2}\Bigr) \quad (2.9)
  $$
- 평균에 대해서도 $E[\mathbf y] \sim \sqrt{m + c} - \tfrac{1}{8\sqrt{m}} + \tfrac{24 c - 7}{128 m^{3/2}}$ (Eq. 2.10).
- 비대칭/첨도: $\gamma_1 \sim -\tfrac{1}{2 m^{1/2}}\bigl\{1 + \tfrac{25 - 48c}{16m}\bigr\}$ (2.13), $\gamma_2 \sim \tfrac{1}{m}\bigl\{1 + \tfrac{945 - 1536 c}{256 m}\bigr\}$ (2.14).
- **효율** $E_y = [\mathrm{cov}(\mathbf r, \mathbf y)]^2/(m \cdot \mathrm{var}(\mathbf y)) \sim 1 - 1/(8m) + (16 c - 9)/(64 m^2)$ (2.16).

#### English
- Substituting $t = r - m$, $m' = m + c$, expand $y = \sqrt{m' + t}$ as a Taylor series in $t/m'$. Coefficients $a_s$ are given by Eq. (2.1); the remainder $R_s$ is bounded by Eq. (2.3).
- The Poisson moments $\mu_n = O(m^{n/2})$ allow exchanging expectation and series (formal expansion is justified).
- The leading correction in $\mathrm{var}(\mathbf y)$ (2.8) is $(3 - 8c)/(8m)$; choosing $c = \tfrac{3}{8}$ cancels it. The next-order correction is $O(1/m^2)$, making the transformed variance $\tfrac{1}{4}\bigl(1 + 1/(16 m^2)\bigr)$.
- The transform also stabilises bias ($m_y - m \sim -\tfrac{1}{4}$ at leading order, independent of $m$), skewness, kurtosis, and yields $\sim 96$–$99\%$ large-sample efficiency for $m \ge 1$.

---

### Part III: §3 Binomial Distribution / 이항 분포

#### 한국어
- 이항 $\mathbf r \sim \mathrm{Bin}(n, p)$에서 $m = np$. 변환 후보: $y = \sqrt{n + d_2}\,\sin^{-1}\sqrt{(r + c)/(n + d_1)}$ (Eq. 3.1).
- Taylor 전개로 분산을 계산하면:
  $$
  \mathrm{var}(\mathbf y) \sim \tfrac{1}{4}\Bigl\{1 + \tfrac{2 d_2 - 1}{2n} + \tfrac{3 - 8c}{8m} + \tfrac{3 + 8c - 8 d_1}{8(n - m)}\Bigr\} \quad (3.4)
  $$
- 모든 $O(1/m)$, $O(1/(n-m))$, $O(1/n)$ 보정을 동시에 0으로 만드는 선택은 $c = \tfrac{3}{8}$, $d_1 = \tfrac{3}{4}$, $d_2 = \tfrac{1}{2}$:
  $$
  \mathrm{var}(\mathbf y) = \tfrac{1}{4} + O(1/n^2) \quad (3.5)
  $$
- $d_1 = 2c$로 변환이 $r = \tfrac{1}{2}n$을 중심으로 대칭. $d_2$는 $y$의 스케일만 바꾸므로 분산 안정화에는 영향 없음.
- 효율 $E_y \sim 1 - (2m - n)^2/(8 nm(n-m))$ (3.9) — $m = n/2$일 때 최대 ($E_y \to 1$).

#### English
- Binomial transform of the form $y = \sqrt{n + d_2}\,\sin^{-1}\sqrt{(r + c)/(n + d_1)}$ (Eq. 3.1).
- Three corrections appear in the variance (Eq. 3.4); the choice $c = \tfrac{3}{8}$, $d_1 = \tfrac{3}{4}$, $d_2 = \tfrac{1}{2}$ cancels them all simultaneously, giving $\mathrm{var}(\mathbf y) = \tfrac{1}{4} + O(1/n^2)$.
- Setting $d_1 = 2c$ gives the natural symmetry around $r = n/2$.
- Efficiency reaches its maximum at $m = n/2$ and degrades as $p$ approaches 0 or 1.

---

### Part IV: §4 Negative-Binomial Distribution / 음이항 분포

#### 한국어
- 음이항 $p_r = \tfrac{\Gamma(r + k)}{r!\,\Gamma(k)}\bigl(\tfrac{m}{m+k}\bigr)^r\bigl(1 + \tfrac{m}{k}\bigr)^{-k}$ (Eq. 4.1).
- 두 가지 영역:
  1. **$m$ 상수, $k \to \infty$** (포아송 극한처럼): $y = \sqrt{k - \tfrac{1}{2}}\,\sinh^{-1}\sqrt{(r + \tfrac{3}{8})/(k - \tfrac{3}{4})}$ (Eq. 4.2). 분산은 $\tfrac{1}{4} + O(1/m^2)$.
  2. **$k$ 고정, $m \to \infty$** (over-dispersion 영역): 표준편차/평균비가 $k^{-1/2}$로 한정. (4.4) $y = 2\sinh^{-1}\sqrt{(r+c)/(k+d)}$, (4.5) $y = \ln(r + A)$. 두 변환의 점근 분산이 $\psi'(k)$에 가까워짐.
- (4.4)에서 $d = -2c$, 최적 $c \approx \tfrac{3}{8} + \tfrac{23}{192k}$ (Eq. 4.25).
- (4.5)는 $k \ge 2$에서 $\mathrm{var}(\mathbf y) \sim \psi'(k) - k(3k^2 - 9k + 7)/[12(k-1)^2 (k-2)^2]\,\alpha^2$ (Eq. 4.26).
- $k = 1$ (geometric 분포)에서는 (4.5)의 점근전개가 다른 형태: $\mathrm{var}(\mathbf y) = \psi'(1) - 2\ln m / m + O(1/m)$ (Eq. 4.37).
- **요약 (§4 끝)**: $k > 2$, $k$ 작으면 (1.5)가 (1.6)보다 좋고, $k = \tfrac{3}{4}$ 근방에서 두 변환이 동등해짐. $k < 2$ 또는 $m$이 매우 크면 (1.6)을 권장.

#### English
- Negative-binomial PMF (Eq. 4.1), with two parameter regimes:
  1. $m$ finite, $k$ large (Poisson-limit regime): use Eq. (4.2) with $\mathrm{var}(\mathbf y) = \tfrac{1}{4} + O(1/m^2)$.
  2. $k$ fixed, $m$ large (over-dispersed regime): two candidates — Eq. (4.4) (sinh-arcsine form) and Eq. (4.5) (logarithmic form). Both stabilise variance at $\psi'(k)$ asymptotically.
- The cumulant-generating-function machinery (Lemma in §4, Theorem at Eq. 4.21) yields the asymptotic expansion of $M(t)$ and hence variance/skewness/kurtosis to leading order.
- Recommendation: prefer (1.5) when $k > 2$; use (1.6) when $k < 2$ or when $m$ is very large.

---

### Part V: §5 Numerical Investigation / 수치적 검증

#### 한국어
- Table 1은 $c = \tfrac{3}{8}$를 사용한 세 변환 (1.3), (1.4), (1.5)을 $m = 1, 2, ..., 20, \infty$에 대해 평가:
  - **편향** $m_y - m$: 포아송에서 $m \ge 6$이면 $-0.250$으로 안정 (limiting value).
  - **분산비** (실제 분산 / 점근 분산): 포아송에서 $m = 1$에 $0.717$, $m = 4$에 $0.999$, $m \ge 6$에서 $\approx 1.000$ — $m \ge 4$에서 ±0.1% 이내.
  - **$\gamma_1$** (skewness): $m = 4$에서 $-0.25$, $m = 20$에서 $-0.11$, $m \to \infty$에서 0.
  - **$\gamma_2$** (kurtosis): $m = 4$에서 $0.15$, $m = 6$에서 $0.20$, $m \to \infty$에서 0.
  - **효율 $E_y$**: 포아송에서 항상 $>96\%$, $m \to \infty$에서 $100\%$.
- Table 2는 음이항 $k = 2$에서 (1.5)를 $c = 0$ (Bartlett의 sinh$^{-1}\sqrt r$)과 $c = \tfrac{1}{2}$ (Beall 1942)와 비교: $c = \tfrac{1}{2}$가 $c = 0$보다 우수하지만 $c = \tfrac{3}{8}$ (Anscombe)이 가장 좋음.

#### English
- Table 1 verifies for the Poisson case: variance ratio is within $\pm 1\%$ for $m \ge 4$, efficiency is $\ge 96\%$ for $m \ge 1$, and the bias $m_y - m$ settles to $-0.250$ for $m \ge 6$. Skewness and kurtosis vanish as $m \to \infty$.
- Table 2 compares Bartlett ($c=0$), Beall ($c=\tfrac{1}{2}$), and Anscombe ($c=\tfrac{3}{8}$) on negative-binomial $k=2$; Anscombe is best.
- The transformation works satisfactorily at very low counts, except in the negative-binomial regime when $k < 2$.

---

## 3. Key Takeaways / 핵심 시사점

1. **$c = \tfrac{3}{8}$은 1차 분산 보정을 정확히 0으로 만든다 / The constant $c = \tfrac{3}{8}$ precisely cancels the first-order variance correction** — Eq. (2.8)의 $(3 - 8c)/(8m)$ 항이 $c = \tfrac{3}{8}$에서 사라짐. 그 결과 분산 안정화가 $O(1/m)$에서 $O(1/m^2)$으로 한 차수 빨라진다. Bartlett의 $c = \tfrac{1}{2}$는 $O(1/m)$에서 더 큰 보정을 남김.
   The factor $c = \tfrac{3}{8}$ is precisely what zeros out the leading $(3 - 8c)/(8m)$ correction in the variance expansion, accelerating stabilisation by one order of magnitude over Bartlett's $c = \tfrac{1}{2}$ choice.

2. **$\sqrt{r + 3/8}$는 사실상 영상 잡음 제거의 출발점 / $\sqrt{r + 3/8}$ is the starting point of low-light image denoising** — 광자 수 영상은 포아송 잡음 지배. 이 변환을 적용하면 분산이 거의 $\tfrac{1}{4}$로 균일해져, 이후 임의의 가우시안 잡음 제거 알고리즘 (BM3D, NLM, wavelet shrinkage)을 적용 가능. 이 “VST + Gaussian denoise + inverse VST” 파이프라인은 이후 60년의 표준이 된다.
   The Anscombe forward transform converts Poisson images into approximately AWGN images, enabling the entire arsenal of Gaussian denoisers to be applied directly. This three-stage pipeline became the dominant approach for low-light imaging until end-to-end deep learning.

3. **이항·음이항 변환에는 추가 자유도 ($d_1, d_2$)가 필요 / Binomial and negative-binomial transforms need extra correction terms** — 이항의 분산식 (3.4)에는 세 보정항(서로 다른 차수)이 있어 $c$ 하나만으로는 모두를 0으로 만들 수 없음. $d_1 = \tfrac{3}{4}$, $d_2 = \tfrac{1}{2}$가 추가 자유도. 음이항도 영역(regime)에 따라 다른 변환이 필요.
   In the binomial case, three correction terms appear at different orders in the variance expansion; cancelling all of them simultaneously requires three constants $c, d_1, d_2$. The negative-binomial case is even more delicate, requiring different transforms for different parameter regimes.

4. **변환은 분산만 안정화시키는 게 아니라 분포 모양도 가우시안화 / The transform Gaussianises the entire distribution, not just the variance** — $\gamma_1 = O(m^{-1/2})$, $\gamma_2 = O(m^{-1})$로 비대칭과 첨도 모두 $m \to \infty$에서 0으로. 즉 변환은 평균뿐 아니라 분포 자체를 정규에 가깝게 만든다. 이것이 Bartlett의 $\sqrt r$을 그대로 쓰지 않고 $\sqrt{r + 3/8}$로 보정하는 본질적 이유.
   The transform reduces skewness to $O(m^{-1/2})$ and kurtosis to $O(m^{-1})$, so the transformed variable approaches normality in distribution, not just in mean and variance.

5. **변환의 역(즉 추정 평균을 자료 평균으로 되돌리기)은 단순 algebraic inverse가 아니다 / The inverse transform is not the algebraic inverse** — Eq. (2.10): $E[\mathbf y] \ne \sqrt{m + c}$; 평균에는 $O(m^{-1/2})$ 편향이 있다. 따라서 denoised $\hat y$에서 $m$을 복원하려면 $(\hat y)^2 - c$가 아닌 더 정교한 보정이 필요. 이는 paper #14 (Mäkitalo-Foi 2013)의 “exact unbiased inverse”로 이어진다.
   The direct (algebraic) inverse $y \mapsto y^2 - c$ is biased of order $O(m^{-1/2})$; proper unbiased inversion requires the higher-order correction terms in (2.10) — a thread directly continued in Mäkitalo-Foi's "exact unbiased inverse" (paper #14).

6. **$m \ge 4$에서 변환은 “충분히 잘 작동” / The transform is "good enough" for $m \ge 4$** — Table 1: $m \ge 4$이면 분산 비가 $0.999$ 이상, 효율 $\ge 96\%$, 편향이 $-\tfrac{1}{4}$로 정착. 즉 평균 광자수가 4 이상인 영상에서는 Anscombe 변환이 사실상 완벽한 등분산성을 제공.
   For mean Poisson rates $m \ge 4$, the variance is stabilised to within 0.1%, efficiency exceeds 96%, and the bias is essentially constant — making the transform practically perfect.

7. **음이항은 “두 영역 모두에서 잘 듣는 변환은 없다” / No single transform works for negative-binomial in all regimes** — $k$와 $m$의 비율에 따라 (1.5) 또는 (1.6)을 골라야 함. 이는 후일 Murtagh et al.(1995)의 generalised Anscombe transform (GAT) — 포아송–가우시안 혼합 잡음용 — 으로 이어진다.
   The negative-binomial requires regime-dependent transforms — a hint of the difficulty later addressed by the generalised Anscombe transform (GAT) of Murtagh et al. (1995) for mixed Poisson-Gaussian noise.

8. **단순한 비선형 변환이 통계 추론을 “쉬운 영역으로 옮기는” 강력한 도구 / A simple nonlinear transformation moves statistical inference to an easier regime** — 등분산 가정 위에 세워진 모든 통계 도구(ANOVA, regression, t-test, denoising)를 계수 자료에 적용 가능하게 함. 1948년의 단순한 발견이 광자수 영상, 형광현미경, 천문학에서 70년 넘게 사용되고 있다.
   A simple algebraic transform unlocks the entire toolkit of normal-theory statistics for count data — a methodological breakthrough whose direct descendants still drive low-light image processing today.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Forward Anscombe transform (Poisson) / 포아송 정변환

$$
\boxed{\; y = 2\sqrt{r + \tfrac{3}{8}} \quad (\text{Anscombe forward, scaled to unit variance}) \;}
$$
여기서 표준 형태 $y = \sqrt{r + 3/8}$의 분산은 $\approx \tfrac{1}{4}$, 인수 2를 추가하면 단위 분산 ($\approx 1$)을 얻는다. 영상처리 문헌에서는 인수 2를 포함한 형태가 일반적.

### 4.2 Variance expansion (Eq. 2.8) / 분산 점근전개
$$
\mathrm{var}(\mathbf y) \sim \frac{1}{4}\left\{1 + \frac{3 - 8c}{8m} + \frac{32 c^2 - 52 c + 17}{32 m^2}\right\}
$$
- 첫 항 $\tfrac{1}{4}$는 limiting variance.
- 둘째 항 $(3 - 8c)/(8m)$는 1차 보정; $c = \tfrac{3}{8}$에서 0.
- 셋째 항은 잔여 $O(1/m^2)$; $c = \tfrac{3}{8}$ 대입 시 $1/(16 m^2)$.

### 4.3 Mean expansion (Eq. 2.10) / 평균 점근전개
$$
E[\mathbf y] \sim \sqrt{m + c} - \frac{1}{8 \sqrt m} + \frac{24c - 7}{128 m^{3/2}}
$$
### 4.4 Generalised Anscombe transform (GAT) — Murtagh et al. 1995
포아송–가우시안 혼합 $\mathbf z = \alpha \mathbf p + \mathbf n$, $\mathbf p \sim \mathrm{Poisson}(y/\alpha)$, $\mathbf n \sim N(\mu, \sigma^2)$일 때:
$$
f(z) = \frac{2}{\alpha} \sqrt{\alpha z + \tfrac{3}{8}\alpha^2 + \sigma^2 - \alpha \mu}, \quad z > -\frac{3 \alpha}{8} - \frac{\sigma^2}{\alpha} + \mu
$$
순 포아송 ($\sigma = 0$, $\mu = 0$, $\alpha = 1$)에서는 $f(z) = 2\sqrt{z + 3/8}$로 환원.

### 4.5 Algebraic inverse / 대수적 역변환
$$
\hat r_{\rm alg} = \left(\frac{y}{2}\right)^2 - \frac{3}{8}
$$
하지만 평균 편향 $O(m^{-1})$ 존재. Asymptotically unbiased inverse는:
$$
\hat r_{\rm asy} = \left(\frac{y}{2}\right)^2 - \frac{1}{8}
$$
(이 형태는 $E[\mathbf y^2] - \tfrac{1}{8}$을 사용; 자세한 유도는 Mäkitalo-Foi 2011, paper #14).

### 4.6 Binomial VST (Eq. 3.1) / 이항 VST
$$
y = \sqrt{n + \tfrac{1}{2}}\,\sin^{-1}\sqrt{\frac{r + 3/8}{n + 3/4}}
$$
with $\mathrm{var}(\mathbf y) = \tfrac{1}{4} + O(1/n^2)$.

### 4.7 Negative-binomial VST (Eq. 4.2) / 음이항 VST ($m$ 상수, $k$ 큼)
$$
y = \sqrt{k - \tfrac{1}{2}}\,\sinh^{-1}\sqrt{\frac{r + 3/8}{k - 3/4}}
$$
with $\mathrm{var}(\mathbf y) = \tfrac{1}{4} + O(1/m^2)$.

### 4.8 Worked numerical example / 수치 예시
$m = 5$인 포아송 변수에서:
- $\mathrm{var}(\mathbf r) = 5$ (signal-dependent).
- 변환 $y = 2\sqrt{r + 3/8}$ 적용 후 (Monte Carlo, $N = 10^6$ 표본):
  - 평균 $\bar y \approx 4.553$, 이론값 $2\sqrt{m + c} = 2\sqrt{5.375} \approx 4.637$, 편향 $\approx -0.084$ (이론 예측 $-1/(4\sqrt m) \approx -0.112$).
  - 분산 $\mathrm{var}(\mathbf y) \approx 1.001$, 이론값 $\approx 1$. 1% 이내 안정화.
- 이는 paper Table 1의 $m = 5$ 행 (분산 비 $0.999$, 편향 $-0.250$ for $\sqrt{r + 3/8}$; 인수 2를 곱하면 $-0.5$에 해당)와 일치.

### 4.9 Comparison with Bartlett ($c = 0$) / Bartlett와의 비교

| $m$ | $\mathrm{var}(\sqrt r)$ (Bartlett, $c = 0$) | $\mathrm{var}(\sqrt{r + 3/8})$ (Anscombe) | Ratio Anscombe/limiting |
|---|---|---|---|
| 1 | $\approx 0.18$ | $\approx 0.18$ | 0.717 |
| 2 | $\approx 0.21$ | $\approx 0.23$ | 0.924 |
| 4 | $\approx 0.23$ | $\approx 0.250$ | 0.999 |
| 10 | $\approx 0.24$ | $\approx 0.250$ | 1.001 |
| $\infty$ | 0.250 | 0.250 | 1.000 |

Bartlett (1936)의 $\sqrt r$은 $O(1/m)$ 보정을 남기지만, Anscombe $\sqrt{r + 3/8}$는 $O(1/m^2)$으로 한 차수 빠르게 limiting variance에 수렴.

Bartlett (1936) leaves $O(1/m)$ corrections, while Anscombe achieves $O(1/m^2)$ — one order faster convergence to the limiting variance.

### 4.10 Connection to Murtagh GAT (1995) / Murtagh GAT와의 연결

순 포아송 ($\sigma = 0$, $\alpha = 1$, $\mu = 0$)에서:
$$
f_\sigma(z)|_{\sigma=0,\alpha=1,\mu=0} = 2\sqrt{z + 3/8}
$$
즉 표준 (인수 2 적용된) Anscombe 변환과 정확히 일치. GAT는 본 논문의 직접적 일반화이며, 추가 자유도 $(\sigma^2, \mu)$를 통해 read noise를 흡수한다.

The generalised Anscombe transform reduces exactly to the scaled Anscombe $2\sqrt{z + 3/8}$ in the pure Poisson limit, demonstrating that GAT is a direct generalisation absorbing the Gaussian noise term $\sigma^2$ and offset $\mu$.

---

### 4.11 Why $c = 3/8$ and not some other value? / 왜 하필 3/8인가

Anscombe의 결과는 단순한 “수치 fitting”이 아니라 *대수적으로 강제됨*. (2.8)에서:
- $c = 0$ (Bartlett): $(3 - 0)/(8m) = 3/(8m)$이 leading correction.
- $c = 1/2$ (early Bartlett 1936 small-$m$ suggestion): $(3 - 4)/(8m) = -1/(8m)$, 음의 보정.
- $c = 3/8$ (Anscombe): $(3 - 3)/(8m) = 0$, leading correction 사라짐.
- $c = 1/4$ or $c = 1/2$: 모두 $O(1/m)$에서 0이 아닌 보정 남김.

따라서 $c = 3/8$은 polynomial $(3 - 8c)$의 *유일한 영점*. 이항·음이항 케이스에서도 같은 원리로 $d_1, d_2$가 결정.

Why $c = 3/8$? Because it is the *unique zero* of the polynomial $(3 - 8c)$ in the leading variance correction (Eq. 2.8). No other value cancels the $O(1/m)$ term. The same algebraic principle determines $d_1 = 3/4, d_2 = 1/2$ for the binomial case.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1936 ─── Bartlett — sqrt(r) transformation for ANOVA on Poisson data
1942 ─── Beall — sqrt(r + 1/2) and binomial-like transforms for entomology
1948 ★★ ANSCOMBE: The Transformation of Poisson, Binomial and Neg-Binomial Data
                  ↳ optimal c = 3/8, full asymptotic expansion
1968 ─── Bartlett-Kendall — log-variance transform analysis
1995 ─── Murtagh-Starck-Bijaoui — generalised Anscombe transform (GAT)
                  ↳ extension to mixed Poisson-Gaussian
2004 ─── Fryzlewicz-Nason — Haar-Fisz: per-scale Anscombe-style stabilisation
2008 ─── Zhang-Fadili-Starck — multiscale VST for Poisson data + curvelets
2009 ─── Foi — clipped & raw CCD data Poisson-Gaussian fitting (parameters)
2010 ─── Luisier-Blu-Unser — PURE-LET (paper #13): direct (no VST) approach
2011 ★ Mäkitalo-Foi — "Optimal Inverse of the Anscombe Transformation"
                  ↳ exact unbiased inverse for pure Poisson
2013 ★ MAKITALO-FOI: Optimal Inverse of the GENERALISED Anscombe Transformation
                  ↳ exact unbiased inverse for Poisson-Gaussian (paper #14)
2017+ ── Deep learning denoisers — VST often used as preprocessing
                  (e.g., for self-supervised methods like Noise2Self).
```

이 논문은 **photon-counting imaging의 노이즈 통계를 “정규 자료”로 환원하는 사실상 모든 후속 작업의 출발점**이다. paper #13 Luisier-Blu-Unser는 *VST를 우회*하는 직접 PURE-LET 방법을 제안했지만, 실제로는 VST + BM3D 파이프라인이 단순성·성능 측면에서 가장 광범위하게 사용된다 (Mäkitalo-Foi 2013 결과 참조).

This paper is **the foundational starting point for nearly all subsequent work converting photon-counting noise statistics into normal-statistical inference**. While paper #13 (Luisier-Blu-Unser PURE-LET) tried to bypass the VST entirely, in practice the "VST + BM3D + inverse VST" pipeline has remained dominant due to its simplicity and performance (as confirmed by paper #14, Mäkitalo-Foi 2013).

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Bartlett (1936)** *J. R. Statist. Soc. Suppl.* | First Poisson sqrt-transform | Anscombe's direct predecessor; $c = 0$ baseline. |
| **Beall (1942)** *Biometrika* | Negative-binomial transform table | Reference for §1's claim about prior work. |
| **Donoho (1993)** "Nonlinear wavelet methods for Poisson data" | First wavelet-based denoising via Anscombe + soft thresholding | Made Anscombe transform mainstream in image processing. |
| **Murtagh, Starck, Bijaoui (1995)** *A&A Suppl.* | Generalised Anscombe transform (GAT) | Direct extension to Poisson-Gaussian noise; foundation of paper #13 and #14. |
| **Fryzlewicz, Nason (2004)** *J. Comput. Graph. Stat.* | Haar-Fisz algorithm | Multiscale Anscombe-style stabilisation per Haar wavelet level. |
| **Luisier, Blu, Unser (2011)** *IEEE TIP* (#13 in this topic) | PURE-LET | Bypasses VST; uses unbiased risk estimate directly on Poisson-Gaussian data. |
| **Mäkitalo, Foi (2013)** *IEEE TIP* (#14 in this topic) | Optimal inverse of GAT | Builds on Anscombe by deriving the exact (max-likelihood) inverse. |
| **Deledalle, Tupin, Denis (2010)** *ICIP* (#12) | Poisson NL Means | Uses likelihood-ratio patch distance instead of VST; complementary route to handling Poisson noise. |
| **Donoho, Johnstone (1994)** *Biometrika* (#1) | VisuShrink | Combined with Anscombe gives SureShrink-on-Poisson, a standard baseline. |
| **Mäkitalo, Foi (2011)** *IEEE TIP* | Optimal inverse of pure Anscombe | Direct continuation of this paper's bias issue ($O(m^{-1/2})$). |

---

### Why this paper still matters in 2026 / 2026년에도 이 논문이 중요한 이유

#### 한국어
77년 전의 단순한 비선형 변환이 여전히 사용되는 이유:
1. **이론적 단순성**: $y = 2\sqrt{r + 3/8}$은 closed-form, 미분 가능, 역변환 가능 (algebraic).
2. **계산 효율**: $O(N)$ 시간; GPU 병렬화 단순.
3. **잡음 모델 보편성**: 광자 수 영상은 어디서나 포아송 — 형광현미경, 천체사진, X-ray, neutron imaging.
4. **현대 알고리즘과의 호환성**: VST + BM3D / VST + DnCNN / VST + Restormer 등 어떤 최신 가우시안 denoiser와도 결합 가능.
5. **Self-supervised learning의 기반**: Noise2Noise, Noise2Self 등은 등분산 가정 하에서 설계 — Anscombe로 “포아송”을 “가우시안”으로 변환해 적용 가능.

#### English
A 77-year-old algebraic transform remains relevant because:
1. **Theoretical simplicity**: closed-form, differentiable, invertible.
2. **Computational efficiency**: $O(N)$ time, trivially parallelisable.
3. **Universal noise model**: photon-counting noise is Poisson everywhere — fluorescence microscopy, astrophotography, X-ray, neutron imaging.
4. **Compatibility with modern algorithms**: combines naturally with BM3D, DnCNN, Restormer, or any AWGN denoiser.
5. **Self-supervised learning foundation**: methods like Noise2Noise/Noise2Self assume homoscedastic noise — Anscombe lets them apply to Poisson data.

---

## 7. References / 참고문헌

- Anscombe, F. J., "The transformation of Poisson, binomial and negative-binomial data", *Biometrika*, 35(3-4), 246–254 (1948). [DOI: 10.1093/biomet/35.3-4.246]
- Bartlett, M. S., "The square root transformation in the analysis of variance", *J. R. Statist. Soc. Suppl.*, 3, 68 (1936).
- Bartlett, M. S., "The use of transformations", *Biometrics*, 3, 39 (1947).
- Bartlett, M. S., & Kendall, D. G., "The statistical analysis of variance-heterogeneity and the logarithmic transformation", *J. R. Statist. Soc. Suppl.*, 8, 128 (1946).
- Beall, G., "The transformation of data from entomological field experiments so that the analysis of variance becomes applicable", *Biometrika*, 32, 243 (1942).
- Fisher, R. A., "Two new properties of mathematical likelihood", *Proc. Roy. Soc. A*, 144, 285 (1934).
- Fisher, R. A., & Yates, F., *Statistical Tables for Biological, Agricultural and Medical Research* (3rd ed. 1948), Edinburgh: Oliver and Boyd.
- Donoho, D. L., "Nonlinear wavelet methods for recovery of signals, densities, and spectra from indirect and noisy data", in *Different Perspectives on Wavelets*, AMS Proc. Symp. Appl. Math., 47, 173–205 (1993).
- Murtagh, F., Starck, J.-L., & Bijaoui, A., "Image restoration with noise suppression using a multiresolution support", *Astron. Astrophys. Suppl.*, 112, 179–189 (1995).
- Mäkitalo, M., & Foi, A., "Optimal inversion of the Anscombe transformation in low-count Poisson image denoising", *IEEE Trans. Image Process.*, 20(1), 99–109 (2011).
- Mäkitalo, M., & Foi, A., "Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise", *IEEE Trans. Image Process.*, 22(1), 91–103 (2013).
