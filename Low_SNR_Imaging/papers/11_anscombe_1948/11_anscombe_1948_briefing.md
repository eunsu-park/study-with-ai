---
title: "Pre-Reading Briefing: The Transformation of Poisson, Binomial and Negative-Binomial Data (Anscombe Transform)"
paper_id: "11_anscombe_1948"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Anscombe Transform (1948): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Anscombe, F. J., "The transformation of Poisson, binomial and negative-binomial data", *Biometrika* 35(3-4), 246–254 (1948). [DOI: 10.1093/biomet/35.3-4.246]
**Author(s)**: F. J. Anscombe
**Year**: 1948

---

## 1. 핵심 기여 / Core Contribution

### 한국어
Anscombe는 신호 의존적 잡음(특히 Poisson)을 갖는 자료를 단순한 비선형 변환 — 본질적으로 $y = 2\sqrt{r + 3/8}$ — 으로 **분산이 거의 일정한 근사적 정규 자료**로 바꾸는 일반 절차(*variance-stabilising transformation, VST*)를 정립했다. 핵심 발견은 $y = \sqrt{r + c}$에서 점근전개의 *first-order* 분산 보정 $(3 - 8c)/(8m)$의 *유일한 영점*이 $c = 3/8$이라는 것 — 단순한 "수치 fitting"이 아니라 *대수적으로 강제되는* 결과이다. 결과적으로 분산 안정화가 $O(1/m)$에서 $O(1/m^2)$로 한 차수 빨라진다 (Bartlett 1936의 $\sqrt r$보다 우수). 이항·음이항으로의 확장도 같은 *대수적 영점화* 원리로 추가 상수 $d_1, d_2$를 결정한다. 78년 후에도 광자 제한 영상(천체사진, 형광현미경, 저조도 카메라, X-ray)의 표준 전처리 — *VST → Gaussian denoiser → inverse VST* — 의 출발점이다.

### English
Anscombe established the general procedure for converting signal-dependent count data (Poisson, binomial, negative-binomial) into **approximately Gaussian observations with constant variance** via a simple algebraic transform — essentially $y = 2\sqrt{r + 3/8}$ in the Poisson case. The key finding: $c = 3/8$ is the *unique zero* of the leading variance correction $(3 - 8c)/(8m)$ in the asymptotic expansion of $\mathrm{var}(\sqrt{r + c})$ — an algebraically forced result, not a numerical fit — accelerating variance stabilisation from $O(1/m)$ (Bartlett 1936) to $O(1/m^2)$. The same algebraic-zeroing principle determines additional constants $d_1, d_2$ for the binomial and negative-binomial cases. Seventy-eight years later it remains the standard preprocessing for photon-limited imaging (astrophotography, fluorescence microscopy, low-light cameras, X-ray): *VST → Gaussian denoiser → inverse VST*.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
1940년대 통계학은 Fisher의 분산분석(ANOVA)이 농학·생물학 실험에 광범위하게 적용되던 시기였지만, 모든 분산분석 도구가 *등분산성*(homoscedasticity)을 가정해 — Poisson 자료(곤충 수, 박테리아 군집, 방사선 검출)에서는 분산이 평균에 비례 ($\mathrm{var}(r) = m$) 해 직접 사용 불가했다. Bartlett(1936)이 $y = \sqrt r$로 분산을 점근적으로 $\tfrac{1}{4}$로 안정화시킴을 보였고, 작은 $m$에서는 $\sqrt{r + 1/2}$가 더 낫다고 제안했다. Anscombe는 일반화 $\sqrt{r + c}$의 *최적 c*를 점근전개로 결정 — 통계 변환 이론의 결정판. 이 논문이 70년 후 영상처리에서 부활하리라곤 당시 누구도 예상하지 못했다.

#### English
In the 1940s, Fisher's ANOVA was widely applied to agricultural and biological experiments, but all ANOVA tools required *homoscedasticity*. Poisson data (insect counts, bacterial colonies, radiation detection) violated this — variance scales with the mean ($\mathrm{var}(r) = m$). Bartlett (1936) showed that $y = \sqrt r$ asymptotically stabilises variance to $\tfrac{1}{4}$, and proposed $\sqrt{r + 1/2}$ for small $m$. Anscombe generalised to $\sqrt{r + c}$ and pinned down the optimal $c$ via asymptotic expansion — the definitive word in transformation theory. No one in 1948 could have predicted its 70-year-later resurrection in image processing.

### 타임라인 / Timeline

```
1936 ─── Bartlett — sqrt(r) variance stabilisation for Poisson ANOVA
1942 ─── Beall — sqrt(r + 1/2) for entomology
1948 ★★ ANSCOMBE — VST for Poisson/binomial/neg-binomial (THIS PAPER)
1993 ─── Donoho — wavelet denoising via Anscombe + soft thresholding
1995 ─── Murtagh-Starck-Bijaoui — Generalised Anscombe (Poisson + Gaussian)
2004 ─── Fryzlewicz-Nason — Haar-Fisz multiscale stabilisation
2011 ─── Mäkitalo-Foi — exact unbiased inverse of Anscombe
2013 ─── Mäkitalo-Foi — exact unbiased inverse of GAT (paper #14)
2017+ ── Self-supervised denoising (Noise2Noise/Self) often use VST
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
- **Poisson 분포**: PMF $P(r|m) = m^r e^{-m}/r!$, $E[r] = \mathrm{var}(r) = m$
- **이항 분포**: $r \sim \mathrm{Bin}(n, p)$, $\mathrm{var}(r) = np(1-p)$
- **음이항 분포**: PMF (Eq. 4.1), 두 매개변수 $k, m$
- **Taylor 전개와 점근전개**: $f(r) = f(m) + f'(m)(r-m) + \frac{1}{2}f''(m)(r-m)^2 + \cdots$
- **모먼트 (moments)**: 평균, 분산, 비대칭 $\gamma_1$, 첨도 $\gamma_2$
- **Variance-stabilising transformation**: $f$의 도함수가 $1/\sigma(f^{-1})$에 비례하도록 선택
- **Lemma의 Stein/Chen identity** (배경 참고): 후속 PURE 이론의 기반

### English
- **Poisson distribution**: PMF, $E[r] = \mathrm{var}(r) = m$
- **Binomial distribution**: $r \sim \mathrm{Bin}(n, p)$, variance $np(1-p)$
- **Negative-binomial**: PMF (Eq. 4.1), parameters $k, m$
- **Taylor / asymptotic expansion**: $f(r) = f(m) + f'(m)(r-m) + \cdots$
- **Moments**: mean, variance, skewness $\gamma_1$, kurtosis $\gamma_2$
- **Variance-stabilising transformation**: choose $f$ so $f'(\mu) \propto 1/\sigma(\mu)$
- (Background) **Stein/Chen identities** — foundation of PURE/SURE later

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Variance-stabilising transform / 분산 안정화 변환 | 평균-의존 분산을 거의 상수로 만드는 비선형 변환. / Nonlinear transform that renders variance approximately constant. |
| Anscombe transform / Anscombe 변환 | $y = 2\sqrt{r + 3/8}$ — Poisson의 표준 VST. / Standard Poisson VST: $y = 2\sqrt{r + 3/8}$. |
| Asymptotic expansion / 점근전개 | $1/m$의 거듭제곱으로 $\mathrm{var}(y)$의 보정항 전개. / Series in inverse powers of $m$ for $\mathrm{var}(y)$. |
| Bartlett (1936) | $y = \sqrt r$ 변환 — Anscombe의 직접적 전신. / Predecessor with $y = \sqrt r$ leaving $O(1/m)$ correction. |
| Generalised Anscombe (GAT) / 일반화 Anscombe | Murtagh+ 1995 — Poisson + Gaussian read noise: $f(z) = (2/\alpha)\sqrt{\alpha z + 3\alpha^2/8 + \sigma^2 - \alpha\mu}$. / Extension to Poisson-Gaussian noise. |
| Inverse VST / 역 VST | $\hat r = (y/2)^2 - 3/8$ (algebraic) vs unbiased inverse (논문 #14). / Algebraic vs unbiased inverse (paper #14). |
| Bias / 편향 | $E[y] \ne \sqrt{m + c}$ — $O(m^{-1/2})$ 잔여 편향. / Residual $O(m^{-1/2})$ bias in transformed mean. |
| Skewness $\gamma_1$ / 비대칭 | $O(m^{-1/2})$로 0에 수렴. / Decays as $O(m^{-1/2})$. |
| Kurtosis $\gamma_2$ / 첨도 | $O(m^{-1})$로 0에 수렴. / Decays as $O(m^{-1})$. |
| Efficiency $E_y$ / 효율 | $[\mathrm{cov}(r, y)]^2 / (m\,\mathrm{var}(y))$ — 변환의 정보 손실 측도. / Information loss measure of the transform. |
| Arcsine VST / 아크사인 VST | 이항의 VST: $\sin^{-1}\sqrt{(r+c)/(n+d_1)}$. / Binomial VST. |
| sinh-arcsin VST / sinh-아크사인 VST | 음이항의 VST: $\sinh^{-1}\sqrt{(r+c)/(k-d)}$. / Negative-binomial VST. |
| Photon-limited imaging / 광자 제한 영상 | 광자 수가 작아 Poisson noise 지배 (저조도, 형광, X-ray). / Imaging regime where photon counts are small and Poisson noise dominates. |

---

## 5. 수식 미리보기 / Equations Preview

**Forward Anscombe (Poisson)**:
$$
y = 2\sqrt{r + \tfrac{3}{8}}
$$

**분산 점근전개 (Eq. 2.8)**:
$$
\mathrm{var}(\mathbf y) \sim \frac{1}{4}\left\{1 + \frac{3 - 8c}{8m} + \frac{32 c^2 - 52 c + 17}{32 m^2}\right\}
$$
첫 보정항이 $c = 3/8$에서 0이 되어 $\mathrm{var}(\mathbf y) \sim \tfrac{1}{4}\bigl(1 + 1/(16 m^2)\bigr)$ (Eq. 2.9).

**평균 점근전개 (Eq. 2.10)**:
$$
E[\mathbf y] \sim \sqrt{m + c} - \frac{1}{8\sqrt m} + \frac{24 c - 7}{128 m^{3/2}}
$$

**이항 VST (Eq. 3.1)**:
$$
y = \sqrt{n + \tfrac{1}{2}}\,\sin^{-1}\!\sqrt{\frac{r + 3/8}{n + 3/4}}, \quad \mathrm{var}(\mathbf y) = \tfrac{1}{4} + O(1/n^2)
$$

**음이항 VST (Eq. 4.2, $m \gg k$ 영역)**:
$$
y = \sqrt{k - \tfrac{1}{2}}\,\sinh^{-1}\!\sqrt{\frac{r + 3/8}{k - 3/4}}, \quad \mathrm{var}(\mathbf y) = \tfrac{1}{4} + O(1/m^2)
$$

**Generalised Anscombe (Murtagh+ 1995, Poisson + Gaussian)**:
$$
f(z) = \frac{2}{\alpha}\sqrt{\alpha z + \tfrac{3}{8}\alpha^2 + \sigma^2 - \alpha\mu}
$$

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **§1 (도입)**: Bartlett(1936)·Beall(1942)의 선행 결과를 빠르게 정리. Eq. 1.3의 일반 형태 $\sqrt{r + c}$가 출발점.
- **§2 (Poisson 점근전개)**: *논문의 핵심*. Eq. 2.1의 Taylor 계수 $a_s$, Eq. 2.3의 잔차 bound, 그리고 Eq. 2.8의 분산 전개를 차근차근. 첫 항 $\tfrac{1}{4}$는 limiting variance, 두 번째 $(3-8c)/(8m)$의 *영점 c=3/8*이 핵심 발견. *대수적 강제*임을 인지하기.
- **§2 평균/비대칭/첨도**: Eq. 2.10–2.14는 정성적으로 — $E[y]$는 $-1/4$ 정도의 잔여 편향, $\gamma_1, \gamma_2$는 모두 0으로 감.
- **§3 (이항)**: Eq. 3.4의 *세 보정항*이 모두 0이 되도록 $c, d_1, d_2$ 동시 결정. 같은 영점화 원리.
- **§4 (음이항)**: 두 영역 — $k$ 큰 (Poisson 극한)·$k$ 고정 — 에서 Eq. 4.2 vs Eq. 4.5 권장. *유일한 만능 변환은 없다*.
- **§5 (Table 1)**: $m \ge 4$에서 분산비 $\ge 0.999$, 효율 $\ge 96\%$, 편향 안정 — 작은 $m$에서도 사실상 완벽함을 수치 확인.
- **흔한 오해**: "$3/8$은 그냥 경험상 좋은 값"이 아님. Eq. 2.8의 polynomial $(3-8c)$의 *대수적 영점*. 논문이 이 사실을 명시적으로 강조하지는 않지만, 후속 영상처리 문헌의 표준 해석.
- **놓치기 쉬운 점**: 변환 후 *평균이 정확하게 회복되지는 않음* — $O(m^{-1/2})$ 편향이 잔존 (Eq. 2.10). 이 편향이 Mäkitalo-Foi(2011, 논문 #14)의 *exact unbiased inverse* 동기.

### English
- **§1 (introduction)**: quickly reconstruct Bartlett (1936) and Beall (1942) baselines. Eq. 1.3's general $\sqrt{r + c}$ is the starting point.
- **§2 (Poisson asymptotic expansion)**: *the heart of the paper*. Walk slowly through Eq. 2.1's Taylor coefficients $a_s$, Eq. 2.3's remainder bound, and Eq. 2.8's variance expansion. The first term $\tfrac{1}{4}$ is the limiting variance; the *zero* of $(3-8c)/(8m)$ at $c=3/8$ is the central discovery — *algebraically forced*, not fitted.
- **§2 mean/skewness/kurtosis**: read Eq. 2.10–2.14 qualitatively — $E[y]$ has a residual bias near $-1/4$; $\gamma_1, \gamma_2$ both decay.
- **§3 (binomial)**: Eq. 3.4's *three correction terms* are simultaneously zeroed by $c, d_1, d_2$ — same algebraic-zeroing principle.
- **§4 (negative-binomial)**: two regimes — large $k$ (Poisson limit) vs fixed $k$ — give Eq. 4.2 vs Eq. 4.5. *No single universal transform.*
- **§5 (Table 1)**: at $m \ge 4$, variance ratio is ≥ 0.999, efficiency ≥ 96%, bias stable — practically perfect even at small counts.
- **Pitfall**: "$3/8$ is just a numerically good value" — *false*. It is the *algebraic zero* of the polynomial $(3-8c)$ in Eq. 2.8. The paper does not emphasise this explicitly, but it is the standard interpretation in the imaging-processing literature.
- **Easy to miss**: the transform does *not* exactly recover the mean — an $O(m^{-1/2})$ bias remains (Eq. 2.10). This bias motivates Mäkitalo-Foi's (2011, paper #14) *exact unbiased inverse*.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
1948년의 단순한 대수 변환이 2026년에도 사용되는 이유: (i) **이론적 단순성** — closed-form, 미분 가능, 역변환 (algebraic) 가능; (ii) **계산 효율** — $O(N)$, GPU 병렬화 자명; (iii) **잡음 모델 보편성** — 광자 수 영상은 형광현미경, 천체사진(코로나그래프, 은하 deep imaging), X-ray, neutron imaging, photon-counting CT 등에서 보편적으로 Poisson; (iv) **현대 알고리즘과의 호환성** — VST + BM3D / VST + DnCNN / VST + Restormer 어떤 Gaussian denoiser와도 즉시 결합; (v) **Self-supervised learning의 기반** — Noise2Noise/Noise2Self는 등분산 가정이라 Anscombe 전처리로 Poisson 자료에 적용 가능. 직접 후속들: Generalised Anscombe (Murtagh+ 1995, read noise 흡수), Mäkitalo-Foi exact unbiased inverse (2011·2013, 논문 #14), Haar-Fisz multiscale stabilisation (Fryzlewicz-Nason 2004), 그리고 Poisson NL Means(논문 #12)의 *대안*적 likelihood-based 접근.

### English
A 78-year-old algebraic transform persists in 2026 because: (i) **theoretical simplicity** — closed-form, differentiable, algebraically invertible; (ii) **computational efficiency** — $O(N)$, trivially GPU-parallelisable; (iii) **universal noise model** — photon-counting noise is Poisson everywhere — fluorescence microscopy, astrophotography (coronagraphs, galaxy deep imaging), X-ray, neutron imaging, photon-counting CT; (iv) **compatibility with modern algorithms** — VST combines instantly with BM3D, DnCNN, Restormer, or any AWGN denoiser; (v) **self-supervised learning foundation** — methods like Noise2Noise/Noise2Self assume homoscedasticity, so Anscombe lets them apply to Poisson data. Direct descendants: Generalised Anscombe (Murtagh+ 1995, absorbing read noise), Mäkitalo-Foi exact unbiased inverse (2011/2013, paper #14), Haar-Fisz multiscale stabilisation (Fryzlewicz-Nason 2004), and the *alternative* likelihood-based route of Poisson NL Means (paper #12).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
