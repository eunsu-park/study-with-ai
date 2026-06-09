---
title: "Solar Cycle Prediction"
authors: ["Kristóf Petrovay"]
year: 2020
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-020-0022-z"
topic: Living_Reviews_in_Solar_Physics
tags: [solar_cycle, prediction, polar_field, dynamo, precursor, SFT, sunspot_number, Cycle_25]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 68. Solar Cycle Prediction / 태양 주기 예측

---

## 1. Core Contribution / 핵심 기여

### English
This 93-page Living Review (2nd edition, 2020) comprehensively surveys methods for predicting the amplitude (and optionally epoch) of an upcoming solar maximum. Petrovay narrows the scope to **predictions issued no later than shortly after the start of the given cycle**, focusing on the three major method classes: **precursor methods**, **model-based predictions** (SFT and dynamo), and **extrapolation methods** (time-series analysis). The review's central quantitative thesis is that the **polar field precursor has consistently demonstrated skill across every cycle where it has been applied** (Cycles 21, 22, 23, 24), and all early Cycle 25 forecasts cluster in the range ~110-160 (peak SSN v2), indicating that Cycle 25 will be similar to or slightly stronger than the weak Cycle 24 — a broad consensus that stands in marked contrast to the chaotic disagreement seen before Cycle 24. The paper also critically examines the 2015 sunspot-number revision (v2.0), emerging nonaxisymmetric "2×2D" dynamo models, and the nonlinear role of individual "rogue" active regions in controlling the polar field buildup.

### 한국어
93페이지 분량의 이 Living Review(2020년 2판)는 **다가오는 태양 극대기의 진폭(과 시점)을 예측하는 방법론 전체**를 종합적으로 정리한다. Petrovay는 범위를 **해당 주기가 시작된 직후보다 늦지 않게 발표되는 예측**으로 한정하고, 세 가지 주요 방법론에 집중한다: **선행지표(precursor) 방법**, **모델 기반 예측**(SFT 및 다이나모), 그리고 **외삽(extrapolation) 방법**(시계열 분석). 리뷰의 핵심 정량적 주장은 **극자기장 선행지표가 Cycle 21, 22, 23, 24에 걸쳐 일관되게 예측력을 입증**했다는 점이며, Cycle 25에 대한 초기 예측들은 모두 ~110-160 (v2 피크 SSN) 범위로 수렴하여, Cycle 25가 약한 Cycle 24와 유사하거나 약간 강할 것이라는 광범위한 합의를 보여준다 — 이는 Cycle 24 이전에 보였던 혼란스러운 의견 충돌과 극명히 대비된다. 논문은 또한 2015년 흑점 수 수정(v2.0), 새롭게 등장한 비축대칭 "2×2D" 다이나모 모델, 그리고 극자기장 형성을 지배하는 개별 "rogue" 활동 영역의 비선형적 역할을 비판적으로 검토한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Sunspot Number and Its Record / 흑점 수와 그 기록

#### §1.2 Sunspot Number (SSN) / 흑점 수

한국어: Wolf(1850)가 도입한 원래 정의는 다음과 같다:

$$R_Z = k(10g + f) \tag{1}$$

여기서 $g$는 흑점 그룹 수, $f$는 보이는 모든 흑점의 총 개수, $k$는 보정 계수이다. Zürich에서 개정된 값들이 Wolf의 원래 시리즈와 연속성을 유지하도록 $k=0.6$이 체계적으로 적용되었다. 월별 평균의 13개월 이동평균은 다음과 같이 표현된다:

$$R = \frac{1}{24}\left(R_{m,-6} + 2\sum_{i=-5}^{i=5}R_{m,i} + R_{m,6}\right) \tag{2}$$

2015년의 대규모 수정(v2.0)에서는 세 가지 주요 보정이 적용되었다: (a) Locarno drift 수정 (1981년 이후), (b) Waldmeier jump 제거 (1947년), (c) Schwabe-Wolf 전이 수정 (1849-1864, 14% 상향 보정).

English: The original Wolf (1850) definition is $R_Z = k(10g + f)$ where $g$ is the number of sunspot groups, $f$ is the total count of all visible spots, and $k$ is a correction factor. The $k=0.6$ scaling was systematically applied to maintain continuity with Wolf's original series. The 13-month running mean in Eq. (2) is the standard smoothing. The 2015 v2.0 revision applied three major corrections: (a) Locarno drift after 1981, (b) elimination of the 1947 Waldmeier jump, (c) Schwabe-Wolf transition correction (1849-1864, 14% upward).

#### §1.4.3 Waldmeier Effect / Waldmeier 효과

한국어: 강한 주기는 상승 시간이 짧다는 경험적 규칙. Waldmeier의 원래 제안:
$$\log R_{\max}^{(n)} = C_1 - C_2(t_{\max}^{(n)} - t_{\min}^{(n)}) \tag{8}$$

Stix(1972) 등은 무작위 강제된 비선형 진동자로 이 관계를 재현:
$$\log R_{\max}^{(n)} = C_1 + C_2 f \tag{5}$$

여기서 $f \sim (t_{\min}^{(n+1)} - t_{\min}^{(n)})^{-1}$. Kitiashvili & Kosovichev(2009)의 Kleeorin-Ruzmaikin 피드백을 가진 다이나모 모델에서는:
$$R_{\max}^{(n)} = C_1 - C_2(t_{\max}^{(n)} - t_{\min}^{(n)}) \tag{6}$$

English: Stronger cycles have shorter rise times. Waldmeier proposed Eq. (8). Stochastically forced nonlinear oscillators reproduce this (Eq. 5). In Kitiashvili-Kosovichev dynamo models with Kleeorin-Ruzmaikin feedback, Eq. (6) describes the regular regime while chaotic regimes give $R_{\max}^{(n)} \propto 1/(t_{\max}^{(n)} - t_{\min}^{(n)})$ (Eq. 7).

### Part II: Precursor Methods / 선행지표 방법

#### §2.1 Cycle Parameters as Precursors — The Minimax Methods

한국어: Brown(1976)은 이전 극소기의 SSN값과 다음 극대 진폭 사이의 강한 선형 상관을 발견:
$$R_{\max} = 114.3 + 6.1\, R_{\min} \tag{10}$$

상관계수 $r=0.676$ (Cycle 19 제외 시). Cameron & Schüssler(2007)는 더 나은 예측자로 극소 **3년 전**의 활동 수준을 제안:
$$R_{\max} = 79 + 1.52\, R(t_{\min} - 3\text{ years}) \tag{11}$$

상관계수 $r=0.800$. Cameron & Schüssler의 Monte Carlo 분석은 이 방법들이 **진정한 주기간 예측이 아니라** Waldmeier 효과와 주기 중첩의 결합임을 보여준다: 강한 주기는 상승이 빨라서 이전 주기의 활동이 아직 낮은 수준에 이르지 않을 때 극소가 발생한다.

English: Brown (1976) found a strong linear correlation between the previous minimum's SSN and the next maximum (Eq. 10, $r=0.676$). Cameron & Schüssler (2007) showed that activity **3 years before** the minimum is a better predictor (Eq. 11, $r=0.800$). Their Monte Carlo analysis demonstrates that these are **not genuine cycle-to-cycle predictors** but a combination of the Waldmeier effect and cycle overlap: stronger cycles rise faster, so the minimum occurs before the previous cycle's activity has fully decayed.

#### §2.3 Polar Precursor / 극자기장 선행지표

한국어: 이 방법은 Schatten et al.(1978)이 처음 제안했고 **현재 가장 신뢰할 만한 선행지표 방법**으로 평가된다. 물리적 근거는 Babcock-Leighton 메커니즘: 극자기장(poloidal)이 다음 주기 toroidal 장의 seed가 된다는 것. 극자기장 데이터의 세 가지 주요 소스:
- **WSO** (Wilcox Solar Observatory): 1976년부터
- **Mt. Wilson Observatory** (MWO): 1974년부터
- **Kitt Peak/SOLIS**: 1976-2003/2003+

축 쌍극자 계수는 다음과 같이 정의된다:
$$D(t) = \frac{3}{2}\int_0^\pi \bar{B}(\theta,t)\cos\theta\sin\theta\, d\theta \tag{12}$$

반구별 극자기장 분리 형태:
$$D_{NS}(t) = \frac{3}{2}\int_0^{\theta_c}\bar{B}(\theta)\cos\theta\sin\theta\, d\theta + \frac{3}{2}\int_{\pi-\theta_c}^\pi \bar{B}(\theta)\cos\theta\sin\theta\, d\theta \tag{13}$$

**Table 1 데이터** (v2 SSN):

| Cycle | 진폭 (SSN v2) | WSO 최대 | WSO 최소 | $D$ 최대 | $D$ 최소 |
|-------|--------------|----------|----------|----------|----------|
| 21 | 232.9 | (1.07) | 1.05 | 4.19 | 4.10 |
| 22 | 212.5 | 1.31 | 1.28 | 4.23 | 3.98 |
| 23 | 180.3 | 1.13 | 1.00 | 3.96 | 3.01 |
| 24 | 116.4 | 0.63 | 0.52 | 1.95 | 1.33 |
| 25 | ? | >0.72 | <0.72 | >1.93 | <1.93 |
| Fit coef. | | 177.4 | 188.8 | 51.3 | 57.5 |
| Scatter | | 25.9 | 24.8 | 16.9 | 21.9 |
| Cycle 25 forecast | | >102 | <161 | >82 | <133 |

한국어: 선형 회귀 fit: $R_{\max} = c \cdot B_{\text{polar}}$. WSO 최대값 사용 시 Cycle 25 예측 = 102-161. **약 70% 확률로 Cycle 25는 102-133 범위에서 피크**할 것, 즉 Cycle 24와 유사한 수준.

English: The polar precursor method (Schatten et al. 1978) is **currently the most reliable precursor method**. It is grounded in the Babcock-Leighton mechanism: the poloidal field at minimum seeds the next cycle's toroidal field. Direct magnetogram measurements come from WSO (from 1976), MWO (from 1974), and Kitt Peak/SOLIS. The axial dipole coefficient (Eq. 12) and hemisphere-split form (Eq. 13) are the core quantities. From Table 1 with a linear fit to Cycles 21-24, the Cycle 25 prediction is ~102-161 (peak SSN v2), with ~70% probability in the 102-133 range — similar to Cycle 24.

#### §2.3.3 Early Forecasts for Cycle 25 from Polar Precursor Extensions

한국어: Petrovay et al.(2018): 극전환 "rush to the poles" (RTTP)의 속도와 극전환-극대 간격의 상관을 이용 → Cycle 25 진폭 ~130, 2024년 후반 피크 예상. Hawkes & Berger(2018): "helicity flux/input rate"를 사용한 선행지표로 SC25=117 예측. Gopalswamy et al.(2018): 17 GHz microwave emission 대용으로 South 반구 89, North 반구 59 예측.

English: Petrovay et al. (2018) used polar rush-to-the-poles (RTTP) velocity correlations, predicting SC25 amplitude ~130 peaking in late 2024. Hawkes & Berger (2018) proposed helicity input rate as a precursor, yielding SC25=117. Gopalswamy et al. (2018) used 17 GHz microwave emission as proxy, predicting 89 (South) and 59 (North).

#### §2.4 Geomagnetic and Interplanetary Precursors / 지자기 및 행성간 선행지표

한국어: 태양은 두 가지 방식으로 지자기 교란을 유발한다:
- **(a) 물질 방출**: CME와 플레어 입자가 자기권에 충돌 — 흑점 활동과 동시 상관, 예측력 제한적
- **(b) 행성간 자기장과 태양풍 속도 변화**: 코로나 구멍에서 나오는 개방 자속이 지배 — 극자기장과 연결

Ohl(1966): $aa$ 지수의 최소값이 다음 극대 진폭과 상관. Feynman(1982): 연간 $aa$ 지수와 월별 SSN의 상관에서 선형 관계의 최소 포락선을 "sunspot component (a)"로, 초과분을 "interplanetary component (b)"로 분리. (b)가 극자기장의 대리지표로 기능하지만 2003년 "Halloween events" 때처럼 큰 eruption은 false signal을 줄 수 있다.

English: The Sun causes geomagnetic disturbances by (a) material ejections (CMEs, flares) — correlated with sunspot activity with no delay, and (b) interplanetary field/wind speed variations — dominated by open flux from coronal holes, linked to polar field. Ohl (1966) correlated minimum $aa$-index with next cycle amplitude. Feynman (1982) separated the two components using the minimum envelope of the $aa$-SSN relation. However, large eruptions (e.g., 2003 Halloween) can give false signals, leading to the erroneous Bhatt et al. (2009) Cycle 24 forecast of $R_m \sim 150$.

### Part III: Model-Based Predictions / 모델 기반 예측

#### §3.1 Surface Flux Transport (SFT) Models / 표면 자속 수송 모델

한국어: SFT 기본 방정식:
$$\frac{\partial B}{\partial t} = -\Omega(\theta)\frac{\partial B}{\partial\phi} - \frac{1}{R_\odot\sin\theta}\frac{\partial}{\partial\theta}[v(\theta)B\sin\theta] + \frac{\eta}{R_\odot^2\sin\theta}\left[\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial B}{\partial\theta}\right) + \frac{1}{\sin\theta}\frac{\partial^2 B}{\partial\phi^2}\right] - B/\tau + S(\theta,\phi,t) \tag{14}$$

우변 항:
- $-\Omega(\theta)\partial_\phi B$: 미분회전에 의한 자속 이동
- 자오면 순환 $v(\theta)$에 의한 극 방향 수송
- 확산 $\eta$ (일반적으로 500-800 km²/s)
- 감쇠항 $B/\tau$ ($\tau \sim 5-10$ 년 필요; Petrovay & Talafha 2019)
- 소스항 $S$: 활동 영역 출현을 표현

**개별 활동 영역(AR)의 축 쌍극자 기여**:
$$\delta D_{\text{BMR}} = \frac{3}{4\pi R_\odot^2} F d \sin\alpha \sin\theta \tag{15}$$

여기서 $F$는 자속, $d$는 두 극성 간 각거리, $\alpha$는 기울기, $\theta$는 colatitude. **"Rogue AR"**: 이 기여도가 극자기장 총량에 필적할 정도로 큰 AR. Jiang et al.(2015): Cycle 24의 약한 극자기장은 **Non-Hale/non-Joy 방향의 저위도 rogue AR** 때문임을 재현.

**Cycle 25 SFT 예측**: Jiang et al.(2018): 93-159. Upton & Hathaway(2018): ~110 (Cycle 24와 유사).

English: The SFT equation (14) describes photospheric flux transport with differential rotation, meridional flow $v(\theta)$, diffusion $\eta$ (typically 500-800 km²/s), decay term $B/\tau$ (with $\tau \sim 5-10$ yr required per Petrovay & Talafha 2019), and source $S$. An individual active region's contribution to the axial dipole is given by Eq. (15). **Rogue ARs** have contributions comparable to the total polar flux. Jiang et al. (2015) showed that Cycle 24's weak polar field resulted from **low-latitude rogue ARs with non-Hale/non-Joy orientations**. Cycle 25 SFT forecasts: Jiang et al. (2018) 93-159; Upton & Hathaway (2018) ~110.

#### §3.4 Dynamo Models / 다이나모 모델

한국어: 평균장 이론의 기본 다이나모 방정식:
$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{U}\times\mathbf{B} + \alpha\mathbf{B}) - \nabla \times (\eta_T \times \nabla\mathbf{B}) \tag{16}$$

축대칭 가정 및 $\alpha\Omega$ 근사 하의 고전적 다이나모 방정식:
$$\frac{\partial A}{\partial t} = \alpha B - (\mathbf{U}_c\cdot\nabla)A - (\nabla\cdot\mathbf{U}_c)A + \eta_T\nabla^2 A \tag{17}$$
$$\frac{\partial B}{\partial t} = \Omega\frac{\partial A}{\partial x} - (\mathbf{U}_c\cdot\nabla)B - (\nabla\cdot\mathbf{U}_c)B + \eta_T\nabla^2 B \tag{18}$$

**두 가지 주요 범주**:
- **Interface dynamo**: $\alpha$가 대류층 하단 (tachocline 근처)에 집중, 두 층 사이 경계에서 파 전파
- **Flux transport dynamo** (advection-dominated): $\alpha$가 표면 근처 (Babcock-Leighton 기원), 자오면 순환이 toroidal 자속을 적도쪽으로 수송

**중요한 시도들**:
- Dikpati & Gilman(2006, Boulder): Cycle 24 = 150 예측 → 실측 116 (과대평가). Cameron & Schüssler(2007)는 이 예측이 사실상 Waldmeier 효과 기반의 minimax 방법에 불과함을 증명.
- Choudhuri et al.(2007, Surya code): Cycle 24 진폭이 Cycle 23보다 30-35% 낮다고 예측 — 실제로 적중. 그러나 모델이 극자기장을 관측 값으로 설정하는 방식 때문에 사실상 polar precursor.
- **Labonville et al.(2019, 2×2D model)**: SFT 코드와 다이나모 코드를 결합. 데이터 동화로 현재 상태를 구현한 후 AR 무작위 realization들로 앙상블 생성. Cycle 25 피크 = **$89^{+29}_{-14}$**, 2025.3$^{+0.89}_{-1.05}$년 예상.

English: The basic mean-field dynamo equation (16) simplifies under axial symmetry to Eqs. (17)-(18). Two main classes: **interface dynamos** (α concentrated near tachocline) and **flux-transport dynamos** (advection-dominated, with α from Babcock-Leighton at surface). Key attempts: Dikpati & Gilman (2006) Boulder model predicted Cycle 24 = 150 (observed 116, too high); Cameron & Schüssler (2007) showed this was effectively a Waldmeier-based minimax method in disguise. Choudhuri et al. (2007) Surya code predicted 30-35% weaker Cycle 24 — correct, but essentially a polar precursor since polar field is set to observed value. **Labonville et al. (2019) 2×2D model** couples SFT + dynamo with data assimilation and ensemble AR emergence, predicting Cycle 25 peak = $89^{+29}_{-14}$ at 2025.3$^{+0.89}_{-1.05}$.

#### §3.6 The Sun as an Oscillator / 진동자로서의 태양

한국어: Truncated 모델에서 공간 의존성을 완전히 무시하고 $\nabla \sim 1/L$로 근사:
$$\dot A = \alpha B - A/\tau \tag{19}$$
$$\dot B = (\Omega/L) A - B/\tau \tag{20}$$

결합:
$$\ddot B = \frac{D-1}{\tau^2}B - \frac{2}{\tau}\dot B \tag{21}$$

여기서 $D = \alpha\Omega\tau^2/L$은 다이나모 수. $D<1$: 감쇠 선형 진동자; $D>1$: 진동 없음.

van der Pol 진동자 형태 (Nagovitsyn 1997; Lopes & Passos 2009):
$$\ddot B = -\xi + \mu(1-\xi^2)\dot\xi \tag{24}$$

영향 파라미터 $\mu > 0$ → self-excited oscillation. Dalton minimum과 같은 grand minima 재현 가능.

English: Radical simplification ignores all spatial dependence (Eqs. 19-20), yielding an oscillator with dynamo number $D$ (Eq. 21). For $D>1$ there are no oscillations. Adding nonlinear quenching produces **Duffing oscillator** (Paluš & Novotná 1999) or **van der Pol oscillator** (Mininni et al. 2000, 2002; Passos 2012) — self-excited oscillations that can reproduce grand minima.

### Part IV: Extrapolation Methods / 외삽 방법

#### §4.1 Linear Regression and ARMA / 선형 회귀와 ARMA

한국어: 자기회귀 모델 (order $p$):
$$R_n = R_0 + \sum_{i=1}^{p}c_{n-i}R_{n-i} + \epsilon_n$$

ARMA 모델 (AutoRegressive Moving Average):
$$R_n = R_0 + \sum_{i=1}^{p}c_{n-i}R_{n-i} + \epsilon_n + \sum_{i=1}^{q}d_{n-i}\epsilon_{n-i}$$

**Brajša et al.(2009)**: 연간 $R$에 ARMA($p=6, q=6$) 적합 → 2012.0에 90±27 피크 예측. **Hiremath(2008)**: 강제 감쇠 조화진동자로 Cycle 24 진폭 110±10 예측.

English: Linear autoregression is the simplest time-series method. Brajša et al. (2009) fit ARMA(6,6) to annual $R$ values, predicting Cycle 24 maximum at 2012.0 with amplitude 90±27. Hiremath (2008) modeled SSN as a forced damped harmonic oscillator, predicting Cycle 24 = 110±10.

#### §4.2 Spectral Methods / 스펙트럴 방법

한국어: 주요 주기 구조:
- **11-year peak** (and 5.5-year harmonic): 주 태양 주기
- **22-year subharmonic**: Hale cycle / Gnevyshev-Ohl rule
- **Long-period power**: Gleissberg cycle (~80-100년)

방법론:
- **Least-squares (LS) frequency analysis** / Lomb-Scargle periodogram
- **Fourier 분석**: Cole(1973)은 Cycle 21 피크를 60으로 예측했지만 실제는 거의 두 배
- **Maximum Entropy Method (MEM)**: Currie(1973), Kane(2007); Kane은 Cycle 24 피크를 80-101로 예측
- **Singular Spectrum Analysis (SSA)**: Loskutov et al.(2001) → Cycle 24 피크 106-117

저자의 평가: "스펙트럴 예측의 처참한 성능은 흑점 수 시계열이 제한된 수의 고정된 주기 성분의 중첩으로 잘 표현되지 않음을 나타낸다."

English: Key features of the SSN spectrum: 11-year peak with 5.5-year harmonic, 22-year subharmonic (Hale/Gnevyshev-Ohl), and long-period power (Gleissberg ~80-100 yr). Methods include LS periodogram, Fourier analysis, Maximum Entropy Method (MEM), and Singular Spectrum Analysis (SSA). The author notes the dismal performance of spectral predictions: "the sunspot number series cannot be well represented by the superposition of a limited number of fixed periodic components."

#### §4.3.4 Neural Networks / 신경망

한국어: 신경망은 매개변수화된 시그모이드 함수를 가진 임계 논리 유닛들의 집합으로, "역전파 규칙"으로 학습된다. 어떤 다차원 비선형 매핑도 도(degree) 1의 step function들의 조합으로 근사 가능하다. 주요 사례:
- **Calvo et al.(1995)**: 첫 신경망 흑점 예측. Cycle 23 피크 = 166 예측 (실측 121, 크게 빗나감)
- **Uwamahoro et al.(2009)**: 더 보수적 예측
- **Attia et al.(2013)**: Neuro-fuzzy — SC25 = 90.7±8, Cycle 24보다 약간 약한 주기
- **Covas et al.(2019)**: 시공간 신경망으로 butterfly diagram 예측 → SC25 = 57±17 (매우 약함)

English: Neural networks approximate any multidimensional nonlinear mapping as combinations of sigmoid functions, trained via backpropagation. Calvo et al. (1995) predicted Cycle 23 peak = 166 (observed 121, far off). Recent Cycle 25 neural network forecasts: Attia et al. (2013) neuro-fuzzy = 90.7±8; Covas et al. (2019) spatiotemporal NN predicts a very weak cycle at 57±17.

### Part V: Summary and Cycle 25 Consensus / 요약과 Cycle 25 컨센서스

#### §5 Summary Evaluation / 요약 평가

한국어: **Precursor 방법이 Cycle 21-24에 걸쳐 가장 일관되게 뛰어난 예측을 제공**했다. 특히 Schatten et al. 1978 이후의 **polar field precursor는 모든 주기에서 예측 기술을 입증**했다. Cycle 24의 경우, Feynman의 지자기 방법만 잘못되어 150을 예측했다 (실제 121). **모델 기반 예측**은 Cycle 24에 대해 세 가지 예측만 가능했고, 이 중 Choudhuri et al.(2007)과 Jiang et al.(2007)은 본질적으로 polar precursor의 변형이었다. **외삽 방법은 전체적으로 저조한 성능**을 보였다.

English: **Precursor methods have been most consistently successful across Cycles 21-24**. The **polar field precursor (Schatten et al. 1978) has consistently proven its skill in all cycles**. For Cycle 24, only Feynman's geomagnetic method gave a wrong prediction (150 vs. observed 121). Only three model-based predictions were available for Cycle 24, of which Choudhuri et al. (2007) and Jiang et al. (2007) were essentially polar-precursor variants. Extrapolation methods have shown unimpressive overall performance.

#### §6 Table 2: Cycle 25 Forecasts / Cycle 25 예측

| Category | Minimum | Maximum | Peak Amplitude | References |
|----------|---------|---------|---------------|-----------|
| Internal precursors | 2019.9 | 2023.8 | 175 (154-202) | Li et al. (2015) |
| Polar precursor (Table 1) | | | 117±15 | This paper |
| Polar precursor (SoDA) | | 2025.2±1.5 | 120±39 | Pesnell-Schatten (2018) |
| Helicity | | | 117 | Hawkes-Berger (2018) |
| Rush-to-the-poles | 2019.4 | 2024.8 | 130 | Petrovay et al. (2018) |
| SFT | | | 124±31 | Jiang et al. (2018) |
| AFT | 2020.9 | | 110 | Upton-Hathaway (2018) |
| 2×2D dynamo | 2020.5±0.12 | 2027.2±1.0 | **89$^{+29}_{-14}$** | Labonville et al. (2019) |
| Truncated dynamo | 2019-2020 | 2024±1 | 90±15 | Kitiashvili (2016) |
| Wavelet tree | | 2023.4 | 132 | Rigozo et al. (2011) |
| Simplex projection | 2024.0±0.6 | | 103±25 | Singh-Bhargawa (2017) |
| Neuro-fuzzy NN | | 2022 | 90.7±8 | Attia et al. (2013) |
| Spatiotemporal NN | | 2022-2023 | **57±17** | Covas et al. (2019) |
| Cycle 24 (reference) | 2008.9 | 2014.3 | 116 | — |

한국어: 대부분의 예측이 **Cycle 24의 ±20% 범위**에 속하여, Cycle 25는 **Cycle 24와 유사한 약한 주기**가 될 것이라는 공감대가 형성되었다. 다이나모 기반 예측(89, 90)은 SFT와 precursor 예측(~110-130)보다 약간 낮은 경향을 보인다. 신경망 예측(57, 90.7)은 매우 약한 주기를 예측한다.

English: Most forecasts lie within **±20% of Cycle 24**, indicating a **consensus for a cycle similar to Cycle 24 (weak)**. Dynamo-based predictions (89, 90) tend slightly lower than SFT and precursor predictions (~110-130). Neural network predictions (57, 90.7) favor a very weak cycle.

---

## 3. Key Takeaways / 핵심 시사점

1. **Polar field precursor is the gold standard** / **극자기장 선행지표가 황금 표준이다**
   한국어: 1978년 Schatten이 제안한 이래 모든 주기(21, 22, 23, 24)에서 예측 성공을 입증한 유일한 방법. 물리적 근거는 Babcock-Leighton 메커니즘 — 극자기장이 다음 주기 toroidal 장의 seed가 된다. Cycle 25 예측 = **117±15**.
   English: Since Schatten's 1978 proposal, the polar field precursor is the only method that has demonstrated predictive skill in every cycle (21, 22, 23, 24). Its physical basis is the Babcock-Leighton mechanism — polar field at minimum seeds the next cycle's toroidal field. Cycle 25 prediction = **117±15**.

2. **Waldmeier effect underlies many "precursors"** / **Waldmeier 효과가 많은 "선행지표"의 근간**
   한국어: Brown(1976)의 minimax 관계나 Dikpati-Gilman(2006)의 Boulder 다이나모 예측은 모두 **Waldmeier 효과와 주기 중첩의 결합**으로 설명된다. 진짜 cycle-to-cycle 예측이 아니라, 강한 주기가 더 빠르게 상승하여 극소 시점을 결정하기 때문이다 (Cameron & Schüssler 2007).
   English: Brown's (1976) minimax relation and Dikpati-Gilman's (2006) Boulder model prediction are both explainable as **combinations of the Waldmeier effect and cycle overlap**. These are not genuine cycle-to-cycle predictors — stronger cycles rise faster, determining the minimum epoch (Cameron & Schüssler 2007).

3. **Rogue active regions control polar field buildup** / **이상 활동 영역이 극자기장 형성을 지배**
   한국어: Cameron et al.(2013)과 Nagy et al.(2017a)이 보인 바와 같이, 큰 저위도 cross-equatorial AR이 극자기장 축적에 과도한 영향을 미친다. Cycle 24의 이례적으로 약한 극자기장은 **Non-Hale/non-Joy 방향의 저위도 rogue AR** 때문으로 재현됨 (Jiang et al. 2015). 한 개의 rogue AR이 주기 예측을 바꿀 수 있다.
   English: As shown by Cameron et al. (2013) and Nagy et al. (2017a), large low-latitude cross-equatorial ARs have disproportionate effect on polar field buildup. Cycle 24's anomalously weak polar field was reproduced by **low-latitude rogue ARs with non-Hale/non-Joy orientations** (Jiang et al. 2015). A single rogue AR can alter cycle prediction.

4. **Modern Maximum has ended** / **Modern Maximum은 종료되었다**
   한국어: Cycle 17-23은 평균보다 강한 주기들의 시대(Modern Maximum)였다. Cycle 24는 이 평균에서 $-2\sigma$ 아래로 떨어졌고, 이는 120년 전 두 차례의 유사한 intercycle drop을 연상시킨다 — 그 후에 Gleissberg minima가 왔다. **Cycle 25가 Cycle 24 수준에 머물 것**이라는 예측은 새로운 Gleissberg minimum 진입을 시사한다.
   English: Cycles 17-23 were the "Modern Maximum," above average in strength. Cycle 24 dropped $-2\sigma$ below this average, paralleling two similar intercycle drops in the 1800s — which heralded Gleissberg minima. The prediction that **Cycle 25 will stay at Cycle 24 levels** suggests entry into a new Gleissberg minimum.

5. **Model-based predictions are maturing but not yet dominant** / **모델 기반 예측은 성숙 중이지만 아직 주류가 아니다**
   한국어: Labonville et al.(2019)의 2×2D 모델은 SFT + 다이나모 + 데이터 동화를 결합하여 $R=89^{+29}_{-14}$ 예측. 이는 진정한 물리 기반 예측이지만, 자유 매개변수와 "rogue AR" 스토캐스틱 인자 때문에 예측력은 제한적이다. Bushby & Tobias(2007)가 경고한 대로 현재 다이나모 모델은 모두 "illustrative" 수준.
   English: Labonville et al.'s (2019) 2×2D model combines SFT + dynamo + data assimilation, predicting $R=89^{+29}_{-14}$. This is genuine physics-based prediction, but free parameters and "rogue AR" stochasticity limit skill. As Bushby & Tobias (2007) warn, all current dynamo models are merely "illustrative."

6. **Extrapolation methods underperform** / **외삽 방법은 성능 저조**
   한국어: ARMA, Fourier, MEM, SSA, Wavelet 등의 시계열 분석은 Cycle 21-24 예측에서 모두 실패에 가까운 결과를 냈다. 다만 신경망과 SSA 같은 최신 방법은 충분한 검증 기회가 없었으므로 향후 주기를 지켜볼 필요가 있다.
   English: ARMA, Fourier, MEM, SSA, and wavelet time-series methods have largely failed Cycles 21-24 predictions. Modern methods like neural networks and SSA have not had sufficient testing, so future cycles will tell.

7. **Cycle 25 consensus: similar to Cycle 24** / **Cycle 25 컨센서스: Cycle 24와 유사**
   한국어: Table 2의 모든 예측은 **Cycle 24의 ±20%** 범위에 있으며, 이는 Cycle 24 이전 SWPC 패널의 혼란과 극명히 대비된다. Cycle 25 피크 시기는 2023.4-2027.2로 넓게 퍼져 있다. **피크 진폭 범위**: 57 (NN) ~ 175 (internal precursor); 주류 예측은 **100-140**.
   English: All Table 2 predictions lie within **±20% of Cycle 24**, a sharp contrast to the chaotic disagreement before Cycle 24. Peak epochs are spread 2023.4-2027.2. **Peak amplitude range**: 57 (NN) to 175 (internal precursor); mainstream predictions are **100-140**.

8. **Hemispheric asymmetry is predictable** / **반구 비대칭성은 예측 가능하다**
   한국어: Labonville et al.(2019)은 북반구가 남반구보다 20% 강하고 6개월 늦게 피크할 것으로 예측. Iijima et al.(2017): Cycle 24 극소기에 북반구 극자기장이 0에 가까웠음. **2×2D 모델의 반구별 예측**은 관측 가능한 검증 대상이다.
   English: Labonville et al. (2019) predicts the northern hemisphere will peak 20% stronger than the south, 6 months later. Iijima et al. (2017): Cycle 24's north polar field hovered near zero. **Hemispheric predictions from 2×2D models** are observable, testable quantities.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Sunspot Number Definitions / 흑점 수 정의

**Wolf relative sunspot number (Eq. 1)**:
$$R_Z = k(10g + f)$$
- $g$: number of sunspot groups / 흑점 그룹 수
- $f$: total number of visible spots / 총 흑점 수
- $k$: correction factor (nominally 0.6 for Zürich continuation) / 보정 계수

**13-month running mean (Eq. 2)**:
$$R = \frac{1}{24}\left(R_{m,-6} + 2\sum_{i=-5}^{i=5}R_{m,i} + R_{m,6}\right)$$

**Bracewell's alternating series (Eq. 3)**:
$$R_\pm = 100(R_B/83)^{3/2}$$
Implies $\alpha = 2/3$ in $R' = R^\alpha$ transformation. Sign alternates between even/odd cycles (Hale's rule).

### 4.2 Precursor Relations / 선행지표 관계식

**Brown 1976 minimum precursor (Eq. 10)**:
$$R_{\max}^{(n+1)} = 114.3 + 6.1\, R_{\min}^{(n)}, \quad r = 0.676$$

**Cameron-Schüssler minimax3 (Eq. 11)**:
$$R_{\max}^{(n+1)} = 79 + 1.52\, R(t_{\min}^{(n)} - 3\text{ yr}), \quad r = 0.800$$

**Waldmeier effect forms**:
- Original Waldmeier (Eq. 8): $\log R_{\max}^{(n)} = C_1 - C_2(t_{\max}^{(n)} - t_{\min}^{(n)})$
- Stochastic oscillator (Eq. 5): $\log R_{\max}^{(n)} = C_1 + C_2 f$ where $f \sim (t_{\min}^{(n+1)} - t_{\min}^{(n)})^{-1}$
- Kitiashvili-Kosovichev (Eq. 6): $R_{\max}^{(n)} = C_1 - C_2(t_{\max}^{(n)} - t_{\min}^{(n)})$

### 4.3 Polar Field Quantities / 극자기장 물리량

**Axial dipole coefficient (Eq. 12)**:
$$D(t) = \frac{3}{2}\int_0^\pi \bar{B}(\theta,t)\cos\theta\sin\theta\, d\theta$$
- $\bar{B}(\theta,t)$: azimuthally averaged radial field
- Schmidt quasi-normalization (standard in solar physics/geomagnetism)

**Hemispheric separation (Eq. 13)**:
$$D_{NS}(t) = \frac{3}{2}(B_{0,N} - B_{0,S})\int_0^{\theta_c}\bar{f}(\theta)\cos\theta\sin\theta\, d\theta$$

assuming $B = B_0 f(\theta)$ with $f(\theta) = \cos^n\theta$, $n = 8 \pm 1$.

**Linear precursor fit to Table 1 (4 cycles)**:
$$R_{\max}^{(n+1)} \approx c_1 \cdot B_{\text{polar,max}}^{(n)}$$
with fit coefficient 177.4 (for WSO max) and scatter 25.9 (in SSN v2 units).

### 4.4 Active Region Dipole Contribution / 활동 영역 쌍극자 기여

**BMR axial dipole (Eq. 15)**:
$$\delta D_{\text{BMR}} = \frac{3}{4\pi R_\odot^2} F d \sin\alpha \sin\theta$$
- $F$: magnetic flux / 자속
- $d$: angular polarity separation / 양극 간 각거리
- $\alpha$: tilt relative to azimuthal / 방위 방향 기울기
- $\theta$: colatitude

Rogue AR criterion: $\delta D_{\text{BMR}}$ comparable to total polar flux (~$10^{22}$ Mx).

### 4.5 SFT Equation / 표면 자속 수송 방정식

**Complete SFT PDE (Eq. 14)**:
$$\frac{\partial B}{\partial t} = \underbrace{-\Omega(\theta)\frac{\partial B}{\partial\phi}}_{\text{diff. rotation}} \underbrace{-\frac{1}{R_\odot\sin\theta}\frac{\partial}{\partial\theta}[v(\theta)B\sin\theta]}_{\text{meridional flow}} + \underbrace{\frac{\eta}{R_\odot^2\sin\theta}\left[\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial B}{\partial\theta}\right) + \frac{1}{\sin\theta}\frac{\partial^2 B}{\partial\phi^2}\right]}_{\text{diffusion}} \underbrace{- B/\tau}_{\text{decay}} + \underbrace{S(\theta,\phi,t)}_{\text{source}}$$

Typical parameters: $\eta \sim 500$–$800$ km²/s, $\tau \sim 5$–$10$ yr, $v_{\max} \sim 10$–$15$ m/s.

### 4.6 Dynamo Equations / 다이나모 방정식

**Mean-field induction (Eq. 16)**:
$$\frac{\partial\mathbf{B}}{\partial t} = \nabla\times(\mathbf{U}\times\mathbf{B} + \alpha\mathbf{B}) - \nabla\times(\eta_T\nabla\times\mathbf{B})$$

**Axisymmetric $\alpha\Omega$ dynamo (Eqs. 17-18)**:
$$\frac{\partial A}{\partial t} = \alpha B - (\mathbf{U}_c\cdot\nabla)A - (\nabla\cdot\mathbf{U}_c)A + \eta_T\nabla^2 A$$
$$\frac{\partial B}{\partial t} = \Omega\frac{\partial A}{\partial x} - (\mathbf{U}_c\cdot\nabla)B - (\nabla\cdot\mathbf{U}_c)B + \eta_T\nabla^2 B$$

### 4.7 Truncated Dynamo / Oscillator Models

**Simplified dynamo (Eqs. 19-20)**:
$$\dot A = \alpha B - A/\tau, \quad \dot B = (\Omega/L)A - B/\tau$$

**Combined 2nd-order (Eq. 21)**:
$$\ddot B = \frac{D-1}{\tau^2}B - \frac{2}{\tau}\dot B$$
where $D = \alpha\Omega\tau^2/L$ is the dynamo number.

**van der Pol form (Eq. 24)**:
$$\ddot\xi = -\xi + \mu(1-\xi^2)\dot\xi, \quad \mu > 0$$
Self-excited oscillator describing amplitude-modulated cycles.

### 4.8 Time-Series Models

**AR(p)**: $R_n = R_0 + \sum_{i=1}^p c_{n-i}R_{n-i} + \epsilon_n$

**ARMA(p,q)**: $R_n = R_0 + \sum_{i=1}^p c_{n-i}R_{n-i} + \epsilon_n + \sum_{i=1}^q d_{n-i}\epsilon_{n-i}$

Brajša et al. (2009): ARMA(6,6) on annual $R$ → Cycle 24 peak = 90±27 at 2012.0.

### Worked Example: Polar Precursor Calibration / 극자기장 선행지표 보정 예시

Given Table 1 data (WSO field at minimum):
- Cycle 22 predictor: $B_{\min}^{(21)} = 1.05$ G → amplitude 212.5
- Cycle 23 predictor: $B_{\min}^{(22)} = 1.28$ G → amplitude 180.3
- Cycle 24 predictor: $B_{\min}^{(23)} = 1.00$ G → amplitude 116.4

Linear fit $R_{\max} = c \cdot B_{\min}$:
$$c = \frac{\sum B_{\min,i} R_{\max,i}}{\sum B_{\min,i}^2} = \frac{1.05 \cdot 212.5 + 1.28 \cdot 180.3 + 1.00 \cdot 116.4}{1.05^2 + 1.28^2 + 1.00^2}$$
$$c = \frac{223.1 + 230.8 + 116.4}{1.10 + 1.64 + 1.00} = \frac{570.3}{3.74} \approx 152.5$$

Using Cycle 24 minimum $B_{\min}^{(24)} = 0.52$ G:
$$R_{\max}^{(25)} \approx 152.5 \times 0.52 \approx 79.3$$

Using the full 4-cycle fit (coefficient 188.8 from Table 1):
$$R_{\max}^{(25)} \approx 188.8 \times 0.52 \approx 98.2$$

Upper limit with WSO field max: $R_{\max}^{(25)} < 0.72 \times 177.4 \approx 128$, giving the "~102-161" range quoted in §2.3.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1850 ┤ Wolf defines relative sunspot number R_Z
      │
1908 ┤ Hale discovers magnetic fields in sunspots
      │
1913 ┤ Kimura: first spectral prediction of sunspot cycle
      │
1935 ┤ Waldmeier: amplitude-period anti-correlation
      │
1952 ┤ Gleissberg: "each cycle as a closed whole"
      │
1955 ┤ Parker: first migratory dynamo model
      │
1961 ┤ Babcock: surface flux transport mechanism proposed
      │
1969 ┤ Leighton: mathematical formulation of flux-transport dynamo
      │
1976 ┤ Brown: R_max = 114.3 + 6.1 R_min (precursor begins)
      │
1978 ┤ ★ Schatten et al.: POLAR FIELD PRECURSOR PROPOSED
      │
1982 ┤ Feynman: geomagnetic aa-index separation method
      │
1998 ┤ Hoyt-Schatten: Group Sunspot Number (GSN) introduced
      │
2006 ┤ Dikpati-Gilman: flux-transport dynamo Cycle 24 = 150 (failed)
      │
2007 ┤ Choudhuri et al. (Surya): Cycle 24 = 75 (close to correct)
      │
2007 ┤ Cameron-Schüssler: minimax explained as Waldmeier + overlap
      │
2009 ┤ SWPC Panel disagrees widely on Cycle 24 (50-180 range)
      │
2010 ┤ Petrovay Living Review 1st edition
      │
2013 ┤ Cameron et al.: rogue AR plumes in butterfly diagram
      │
2015 ┤ Sunspot Number v2.0 revision released
      │
2015 ┤ Jiang et al.: Cycle 24 weak due to non-Hale rogue ARs
      │
2019 ┤ ★ Labonville et al.: 2×2D dynamo prediction SC25 = 89
      │
2020 ┤ ★ PETROVAY LIVING REVIEW 2ND EDITION (THIS PAPER)
      │
~2024 ┤ Cycle 25 peak expected (consensus: ~100-140)
```

한국어: 이 논문은 polar precursor (1978)의 성공적 입증, 2015년 SSN 수정, 그리고 2×2D 다이나모 모델의 등장이라는 세 가지 주요 이정표 이후의 **종합적 재평가**를 대표한다.

English: This paper represents a **comprehensive re-assessment** after three major milestones: the successful validation of the polar precursor (1978), the 2015 SSN revision, and the emergence of 2×2D dynamo models.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Paper #20: Charbonneau (2010/2020) "Dynamo Models of the Solar Cycle"** | 본 논문은 Charbonneau의 다이나모 리뷰를 §3.4에서 명시적으로 인용하며 dynamo 예측의 물리적 근거를 공유 / This paper explicitly cites Charbonneau's dynamo review in §3.4, sharing physical foundations | **Direct foundational link / 직접적 기초 연결**. 다이나모 모델의 원리가 예측의 근거가 됨 |
| **Paper #43: Hathaway (2015) "The Solar Cycle"** | Hathaway의 전반적 주기 설명이 본 논문의 배경. Figs. 16, 18 (aa index vs SSN) 참조됨 / Hathaway's general cycle review provides backdrop; cited for Figs. 16, 18 | **Background context / 배경 맥락**. 흑점 주기 통계와 지자기 활동 비교 |
| **Schatten, Scherrer, Svalgaard & Wilcox (1978)** | Polar field precursor method의 원전. §2.3의 핵심 / Origin of polar precursor method; core of §2.3 | **Most cited in §2.3 / §2.3의 최다 인용** |
| **Clette et al. (2014, 2016)** | 2015 SSN v2.0 수정의 공식 기술보고서. §1.2.2 전체 근거 / Official technical report on 2015 v2.0 revision; basis of §1.2.2 | **Critical data source / 핵심 데이터 소스** |
| **Jiang et al. (2018)** | SFT 기반 Cycle 25 예측 (93-159). §3.1.6의 핵심 / SFT-based Cycle 25 prediction (93-159); core of §3.1.6 | **Current SFT benchmark / 현재 SFT 벤치마크** |
| **Labonville, Charbonneau, Lemerle (2019)** | 2×2D 다이나모 모델의 Cycle 25 예측 ($89^{+29}_{-14}$). §3.4.2의 highlight / 2×2D dynamo Cycle 25 prediction; highlight of §3.4.2 | **Modern model-based forecast / 현대 모델 기반 예측** |
| **Cameron & Schüssler (2007)** | Dikpati-Gilman 예측이 사실상 minimax 방법임을 증명. §3.4.1과 §2.1에 반복 인용 / Demonstrated Dikpati-Gilman prediction is effectively a minimax method | **Critical re-interpretation / 비판적 재해석** |
| **Muñoz-Jaramillo et al. (2012, 2013)** | 극 facular를 이용한 극자기장 재구성 (1906-2014). §2.3.2의 근거 / Polar faculae reconstruction of polar field (1906-2014); basis of §2.3.2 | **Historical polar field / 역사적 극자기장** |

---

## 7. References / 참고문헌

### Primary Reference / 주 참고문헌
- Petrovay, K., "Solar cycle prediction", *Living Reviews in Solar Physics*, **17**:2 (2020). DOI: [10.1007/s41116-020-0022-z](https://doi.org/10.1007/s41116-020-0022-z)

### Key Cited Works / 주요 인용 문헌

**Precursor methods**:
- Brown, G.M., "What determines sunspot maximum?", *MNRAS* 174, 185-189 (1976).
- Schatten, K.H., Scherrer, P.H., Svalgaard, L., Wilcox, J.M., "Using dynamo theory to predict the sunspot number during solar cycle 21", *Geophys. Res. Lett.* 5, 411-414 (1978).
- Feynman, J., "Geomagnetic and solar wind cycles, 1900-1975", *J. Geophys. Res.* 87, 6153-6162 (1982).
- Ohl, A.I., "Cycle 20 des Sol. I'solaire", *Solnechnye Dannye* 12, 84-85 (1966).
- Cameron, R., Schüssler, M., "Solar cycle prediction using precursors and flux transport models", *ApJ* 659, 801-811 (2007).
- Hawkes, G., Berger, M.A., "Magnetic Helicity as a Predictor of the Solar Cycle", *Sol. Phys.* 293, 109 (2018).
- Petrovay, K., Nagy, M., Gerják, T., Juhász, L., "Precursors of an upcoming solar cycle at high latitudes", *J. Atmos. Sol.-Terr. Phys.* 176, 15-22 (2018).

**Model-based predictions**:
- Dikpati, M., Gilman, P.A., "Simulating and predicting solar cycles using a flux-transport dynamo", *ApJ* 649, 498-514 (2006).
- Choudhuri, A.R., Chatterjee, P., Jiang, J., "Predicting solar cycle 24 with a solar dynamo model", *Phys. Rev. Lett.* 98, 131103 (2007).
- Jiang, J., Cameron, R.H., Schüssler, M., "The cause of the weak solar cycle 24", *ApJ Lett.* 808, L28 (2015).
- Jiang, J., Wang, J.X., Jiao, Q.R., Cao, J.B., "Predictability of the solar cycle over one cycle", *ApJ* 863, 159 (2018).
- Labonville, F., Charbonneau, P., Lemerle, A., "A Dynamo-based Forecast of Solar Cycle 25", *Sol. Phys.* 294, 82 (2019).
- Upton, L.A., Hathaway, D.H., "An Updated Solar Cycle 25 Prediction with AFT", *Geophys. Res. Lett.* 45, 8091-8095 (2018).
- Yeates, A.R., Muñoz-Jaramillo, A., "Kinematic active region formation in a three-dimensional solar dynamo model", *MNRAS* 436, 3366-3379 (2013).

**Sunspot number series**:
- Clette, F., Svalgaard, L., Vaquero, J.M., Cliver, E.W., "Revisiting the Sunspot Number", *Space Sci. Rev.* 186, 35-103 (2014).
- Hoyt, D.V., Schatten, K.H., "Group sunspot numbers: A new solar activity reconstruction", *Sol. Phys.* 181, 491-512 (1998).

**Time-series methods**:
- Brajša, R., Wöhl, H., Hanslmeier, A., et al., "On solar cycle predictions and reconstructions", *A&A* 496, 855-861 (2009).
- Hiremath, K.M., "Prediction of solar cycle 24 and beyond", *Ap&SS* 314, 45-49 (2008).
- Calvo, R.A., Ceccato, H.A., Piacentini, R.D., "Neural network prediction of solar activity", *ApJ* 444, 916-921 (1995).
- Covas, E., Peixinho, N., Fernandes, J., "Neural network forecast of the sunspot butterfly diagram", *Sol. Phys.* 294, 24 (2019).
- Attia, A.F., Ismail, H.A., Basurah, H.M., "A Neuro-Fuzzy modeling for prediction of solar cycles 24 and 25", *Ap&SS* 344, 5-11 (2013).

**Related Living Reviews**:
- Charbonneau, P., "Dynamo models of the solar cycle", *Living Rev. Sol. Phys.* 7, 3 (2010). [Paper #20]
- Hathaway, D.H., "The solar cycle", *Living Rev. Sol. Phys.* 12, 4 (2015). [Paper #43]
- Wang, Y.-M., "Surface Flux Transport and the Evolution of the Sun's Polar Fields", *Space Sci. Rev.* 210, 351-365 (2017).

### License / 라이선스
This paper is Open Access under the Creative Commons Attribution 4.0 International License.
본 논문은 Creative Commons Attribution 4.0 국제 라이선스 하에 Open Access이다.
