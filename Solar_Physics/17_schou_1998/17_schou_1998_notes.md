---
title: "Helioseismic Studies of Differential Rotation in the Solar Envelope by the Solar Oscillations Investigation Using the Michelson Doppler Imager"
authors: Jesper Schou, H. M. Antia, S. Basu, R. S. Bogart, R. I. Bush, S. M. Chitre, J. Christensen-Dalsgaard, M. P. Di Mauro, W. A. Dziembowski, A. Eff-Darwich, D. O. Gough, D. A. Haber, J. T. Hoeksema, R. Howe, S. G. Korzennik, A. G. Kosovichev, R. M. Larsen, F. P. Pijpers, P. H. Scherrer, T. Sekii, T. D. Tarbell, A. M. Title, M. J. Thompson, J. Toomre
year: 1998
journal: "The Astrophysical Journal, 505, 390–417"
doi: "10.1086/306146"
topic: Solar Physics / Helioseismology
tags: [differential rotation, tachocline, helioseismology, SOHO, MDI, inversion, solar interior, convection zone]
status: completed
date_started: 2026-04-17
date_completed: 2026-04-17
---

# 17. Helioseismic Studies of Differential Rotation in the Solar Envelope / MDI를 이용한 태양 외피 차등 회전의 일진학적 연구

---

## 1. Core Contribution / 핵심 기여

이 논문은 SOHO 위성에 탑재된 MDI(Michelson Doppler Imager)의 첫 144일 관측 데이터로부터 태양 내부의 차등 회전 프로파일을 전례 없는 정밀도로 매핑한 획기적인 연구입니다. 7개의 독립적인 역산(inversion) 기법을 적용하고 비교함으로써 결과의 강건성을 검증했습니다. 주요 발견으로는: (1) 대류층에서 표면 차등 회전이 깊이에 거의 무관하게 유지됨, (2) 대류층 하부의 타코클라인(tachocline)에서 차등 회전이 균일 회전으로 급격히 전이됨, (3) 복사층 내부의 거의 균일한 강체 회전(~430 nHz), (4) 근표면 전단층에서의 회전률 변화, (5) 위도 75° 부근의 잠긴 극 제트(submerged polar jet), (6) 비틀림 진동(torsional oscillations)과 일치하는 교대 빠른/느린 회전 대(zonal bands)를 발견했습니다.

This paper is a landmark study that mapped the Sun's internal differential rotation profile with unprecedented precision using the first 144 days of observational data from the MDI (Michelson Doppler Imager) aboard the SOHO spacecraft. By applying and comparing seven independent inversion techniques, the robustness of results was validated. Key findings include: (1) surface differential rotation is maintained nearly independent of depth throughout the convection zone, (2) a sharp transition from differential to uniform rotation at the tachocline at the base of the convection zone, (3) nearly uniform solid-body rotation in the radiative interior (~430 nHz), (4) rotation rate variations in the near-surface shear layer, (5) a submerged polar jet near latitude 75°, and (6) alternating bands of faster and slower rotation consistent with torsional oscillations.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction / 서론 (§1, pp. 390–392)

태양은 중간 질량 주계열성으로서 광범위한 대류 외피를 가집니다. 대류는 열을 전달할 뿐 아니라 각운동량을 재분배하여 차등 회전을 형성합니다. 표면에서는 적도의 항성 회전 주기가 약 25일, 극에서 약 35일로 측정됩니다.

The Sun, as a moderate-mass main-sequence star, has an extensive convective envelope. Convection not only transports heat but also redistributes angular momentum, producing differential rotation. At the surface, the sidereal rotation period is about 25 days at the equator and more than a month near the poles.

대류는 자기장을 생성하고 재건하는데, 이 과정은 대류층 내부와 그 바로 아래의 오버슈팅 영역에서 확립되는 회전 프로파일에 민감하게 의존합니다. 이론적 예측(예: "바나나 셀" 대류)에 따르면 회전 각속도 $\Omega$가 회전축에 평행한 원통 위에서 거의 일정할 것으로 예상되었지만, 일진학 관측은 이와 다른 결과를 보여줬습니다.

Convection builds and rebuilds magnetic fields through magnetohydrodynamic action that depends sensitively on the rotation profile established both within the convection zone and in the overshoot region. Theoretical predictions (e.g., "banana cell" convection) suggested $\Omega$ would be nearly constant on cylinders aligned with the rotation axis, but helioseismic observations revealed a quite different picture.

타코클라인(tachocline)은 대류층 하부에서 차등 회전이 복사층의 균일 회전으로 전이되는 강한 전단(shear) 영역입니다. Gilman, Morrow, & DeLuca (1989)와 Spiegel & Zahn (1992)에 의해 그 역학이 논의되었습니다.

The tachocline is a region of strong shear at the base of the convection zone where differential rotation transitions to the nearly uniform rotation of the radiative interior. Its dynamics were discussed by Gilman, Morrow, & DeLuca (1989) and Spiegel & Zahn (1992).

---

### Part II: SOI-MDI Rotational Splitting Data / SOI-MDI 회전 분리 데이터 (§2, pp. 392–394)

#### 태양 진동 모드와 회전 분리 / Solar Oscillation Modes and Rotational Splitting

태양 진동의 각 모드는 세 양자수 $(n, l, m)$으로 식별됩니다. 구대칭 비회전 태양에서 모드 주파수 $\omega_{nlm}$은 $m$에 무관합니다. 태양의 회전은 이 축퇴를 깨뜨려, 선도 차수(leading order)에서:

Each mode of solar oscillation is identified by three quantum numbers $(n, l, m)$. In a spherically symmetric, nonrotating star, the mode frequency $\omega_{nlm}$ is independent of $m$. Solar rotation lifts this degeneracy, and to leading order:

$$\Delta\omega_{nlm} \equiv \omega_{nlm} - \omega_{nl} = \int_0^R \int_0^{\pi} K_{nlm}(r, \theta)\,\Omega(r, \theta)\,r\,dr\,d\theta \tag{1}$$

여기서 $K_{nlm}(r, \theta)$는 모드 커널(mode kernel)로, 태양의 평균 구대칭 구조의 함수입니다. 커널은 적도에 대해 대칭이므로 분리는 $\Omega(r, \theta)$의 대칭 성분에만 민감합니다.

Here $K_{nlm}(r, \theta)$ is the mode kernel, a function of the mean spherically symmetric structure of the Sun. The kernels are symmetric about the equator, so the splittings are sensitive only to the symmetric component of $\Omega(r, \theta)$.

#### 분리 계수 / Splitting Coefficients

개별 모드 주파수 대신, 멀티플렛 내의 주파수를 직교 다항식으로 전개합니다:

Instead of individual mode frequencies, the frequencies within a multiplet are expanded in orthogonal polynomials:

$$\omega_{nlm}/2\pi = \nu_{nl} + \sum_{j=1}^{j_{\max}} a_j(n, l)\,\mathcal{P}_j^{(l)}(m) \tag{2}$$

여기서 $\mathcal{P}_j^{(l)}(m)$은 Clebsch-Gordan 계수와 관련된 직교 다항식입니다:

Where $\mathcal{P}_j^{(l)}(m)$ are orthogonal polynomials related to Clebsch-Gordan coefficients:

$$\mathcal{P}_j^{(l)}(m) = \frac{l\sqrt{(2l-j)!(2l+j+1)!}}{(2l)!\sqrt{2l+1}}\,C_{j0lm}^{lm} \tag{4}$$

**핵심 속성**: 홀수 $a$-계수($a_1, a_3, a_5, ...$)만 회전에 기여합니다. 짝수 계수($a_2, a_4, ...$)는 구조적 비구면성에서 기인합니다.

**Key property**: Only odd $a$-coefficients ($a_1, a_3, a_5, ...$) contribute to rotation. Even coefficients ($a_2, a_4, ...$) arise from structural asphericity.

홀수 분리 계수와 회전의 관계:

The relation between odd splitting coefficients and rotation:

$$2\pi a_{2s+1}(n, l) = \int_0^R \int_0^{\pi} \sum_m \gamma_{2s+1}(l, m) K_{nlm}(r, \theta)\,\Omega(r, \theta)\,r\,dr\,d\theta \equiv \int_0^R \int_0^{\pi} K_{nls}^{(a)}\,\Omega\,r\,dr\,d\theta \tag{6}$$

#### 데이터 세트 / Data Set

- **관측 기간**: 1996년 5월 9일 ~ 9월 29일 (144일)
- **듀티 사이클**: 95.47%
- **모드 범위**: $l = 1$ ~ 250, $\nu_{nl}$ = 954 ~ 4556 $\mu$Hz
- **분리 계수**: $j_{\max} = 6, 18, 36$ (맞춤에 따라)
- **총 데이터**: 30,648개 분리 계수 ($a_j$, 홀수 $j$, $a_{35}$까지), 2036개 $(n, l)$ 멀티플렛

- **Observation period**: 1996 May 9 to September 29 (144 days)
- **Duty cycle**: 95.47%
- **Mode range**: $l = 1$ to 250, $\nu_{nl}$ = 954 to 4556 $\mu$Hz
- **Splitting coefficients**: $j_{\max} = 6, 18, 36$ (depending on fit)
- **Total data**: 30,648 splitting coefficients ($a_j$, odd $j$, up to $a_{35}$), from 2036 $(n, l)$ multiplets

$a_j$의 오차는 $l$-범위 중간($l \approx 80$, $\nu \approx 3$ mHz)에서 $j = 1$일 때 약 0.2–0.3 nHz이며, $j$가 증가하면 0.4–0.5 nHz까지 증가한 후 $j \approx 35$에서 다시 0.02–0.03 nHz로 감소합니다.

Errors in $a_j$ are about 0.2–0.3 nHz for $j = 1$ in the middle of the $l$-range ($l \approx 80$, $\nu \approx 3$ mHz), increasing to 0.4–0.5 nHz for higher $j$, then dropping back to 0.02–0.03 nHz for $j \approx 35$.

---

### Part III: Inversion Techniques / 역산 기법 (§3, pp. 394–396)

#### 역산의 기본 형식 / Basic Form of Inversion

관측 데이터 $d_i$는 회전률의 가중 평균에 잡음이 더해진 것입니다:

Observed data $d_i$ are weighted averages of the rotation rate plus noise:

$$d_i = \int_0^{\pi/2} \int_0^R K_i(r, \theta)\,\Omega(r, \theta)\,r\,dr\,d\theta + \epsilon_i, \quad i = 1, \ldots, M \tag{7}$$

역산 해는 데이터의 선형 결합으로 표현됩니다:

The inversion solution is expressed as a linear combination of data:

$$\hat{\Omega}(r_0, \theta_0) \equiv \sum_{i=1}^{M} c_i(r_0, \theta_0)\,d_i = \int_0^{\pi/2} \int_0^R \mathscr{K}(r_0, \theta_0, r, \theta)\,\Omega(r, \theta)\,r\,dr\,d\theta + \sum_{i=1}^{M} c_i(r_0, \theta_0)\,\epsilon_i \tag{8}$$

여기서 **평균 커널(averaging kernel)** $\mathscr{K}$가 정의됩니다:

Where the **averaging kernel** $\mathscr{K}$ is defined:

$$\mathscr{K}(r_0, \theta_0, r, \theta) = \sum_{i=1}^{M} c_i(r_0, \theta_0)\,K_i(r, \theta) \tag{9}$$

추정 오차의 분산:

Variance of the estimated error:

$$\sigma[\hat{\Omega}(r_0, \theta_0)]^2 = \sum_{i=1}^{M} [c_i(r_0, \theta_0)\,\sigma_i]^2 \tag{10}$$

#### 7가지 역산 방법 / Seven Inversion Methods

논문은 **두 부류**의 역산 기법을 사용합니다:

The paper uses **two classes** of inversion methods:

**최소자승법 (Least-Squares Methods)**:

| Method | Description / 설명 |
|---|---|
| **2dRLS** | 2차원 정칙화 최소자승법. $(r, \theta)$ 평면에서 쌍일차(bilinear) 함수를 24×100 격자에 맞춤. 2차 도함수에 대한 적분 페널티 사용 / 2D Regularized Least Squares on 24×100 mesh with integral penalty on second derivatives |
| **1.5dRLS** | $\Omega(r, \theta) = \sum_s \Omega_s(r)\psi_{2s}^{(1)}(x)$로 전개 후 반경 방향만 RLS. 반복 개선(iterative refinement) 사용 / Expansion in latitude, independent RLS in radius with iterative refinement |
| **OMD** | 최적 격자 분포(Optimal Mesh Distribution). 1.5dRLS와 유사하나 반경 격자를 최적화 / Similar to 1.5dRLS but with optimal radial mesh distribution |

**국소 평균법 (Localized Averages Methods)**:

| Method | Description / 설명 |
|---|---|
| **2dSOLA** | 2차원 감산적 최적 국소 평균. 평균 커널 $\mathscr{K}$를 목표 함수 $\mathscr{T}$에 근사. 가우시안 목표 사용 / 2D Subtractive Optimally Localized Averages with Gaussian targets |
| **1d×1dOLA** | 근사 2D 곱셈적 OLA. 반경과 위도 방향으로 분리된 국소화 / Approximate 2D multiplicative OLA with separate radial and latitudinal localization |
| **1d×1dSOLA** | 근사 2D 감산적 SOLA. 커널 분리를 활용한 효율적 2D 역산 / Approximate 2D subtractive SOLA using kernel factorization |
| **1.5dSOLA** | 1.5dRLS와 같은 형태의 데이터를 SOLA 방식으로 반경 역산 / SOLA inversion in radius with data in 1.5d form |

모든 방법은 **해상도와 오차 사이의 트레이드오프**를 제어하는 매개변수에 의존합니다.

All methods depend on parameters controlling the **trade-off between resolution and error**.

2dSOLA의 목적 함수:

The 2dSOLA objective function:

$$\int_0^R \int_0^{\pi/2} [\mathscr{T}(r_0, \theta_0, r, \theta) - \mathscr{K}(r_0, \theta_0, r, \theta)]^2\,r\,dr\,d\theta + \lambda \sum_{i=1}^{M} [\sigma_i\,c_i(r_0, \theta_0)]^2 \tag{15}$$

첫째 항은 평균 커널이 목표 함수에 가깝도록, 둘째 항은 오차를 제어합니다. 매개변수 $\lambda$가 둘 사이의 트레이드오프를 조절합니다.

The first term ensures the averaging kernel is close to the target, the second controls the error. Parameter $\lambda$ controls the trade-off.

---

### Part IV: Resolution and Reliability / 해상도와 신뢰성 (§4, pp. 396–399)

#### Hare-and-Hounds 실험 / Hare-and-Hounds Exercise

역산 기법의 신뢰성을 검증하기 위해 **블라인드 테스트**를 수행했습니다. 한 저자(A. G. K. = Kosovichev)가 두 가지 인공 회전 프로파일(test1, test2)을 설계하고, 실제 SOI-MDI 데이터와 같은 모드 세트와 잡음 수준으로 인공 분리 계수를 생성했습니다. 6개의 독립적인 역산 그룹("hounds")이 진짜 프로파일을 모른 채 역산을 수행했습니다.

To validate the reliability of inversion methods, a **blind test** was conducted. One author (A. G. K. = Kosovichev) designed two artificial rotation profiles (test1, test2) and generated artificial splitting coefficients with the same mode set and noise levels as real SOI-MDI data. Six independent inversion groups ("hounds") performed inversions without knowing the true profiles.

**결과**: hounds는 대류층과 근표면 영역에서 주요 특성을 정확히 복원했습니다. 신뢰할 수 있는 역산 영역은 대략 $r_0 \geq 0.5R$, 적도에서 위도 약 60°(0.5R에서)에서 약 80°(표면에서)까지입니다.

**Results**: The hounds correctly recovered the main features in the convection zone and near-surface region. The reliable inversion region is approximately $r_0 \geq 0.5R$, from the equator to about 60° latitude (at 0.5R) to about 80° (at the surface).

#### 평균 커널의 특성 / Properties of Averaging Kernels

Fig. 1은 세 가지 역산 방법(2dRLS, 2dSOLA, 1d×1dSOLA)에 대한 평균 커널을 5개 목표 위치에서 보여줍니다. 대류층 중간의 적도 영역에서는 커널이 목표 위치에 잘 국소화되지만, 심부 내부와 극 근처에서는 국소화가 불량합니다.

Fig. 1 shows averaging kernels for three inversion methods at five target locations. In the equatorial convection zone, kernels are well localized at target positions, but localization is poor in the deep interior and near the poles.

Fig. 2는 2dSOLA 역산에서 평균 커널의 최대값 위치가 목표 위치에서 얼마나 벗어나는지를 보여줍니다. 10% 미만의 이동(blank 영역)은 대류층 중저위도에 집중되어 있습니다.

Fig. 2 shows how far the averaging kernel maximum shifts from the target position in 2dSOLA inversions. Regions with less than 10% shift (blank regions) are concentrated in the low-to-mid latitude convection zone.

---

### Part V: Results — Large-Scale Differential Rotation / 결과 — 대규모 차등 회전 (§5.1, pp. 399–401)

#### 2D 회전 지도 / 2D Rotation Maps

Fig. 3은 4가지 방법(2dRLS, 2dSOLA, 1d×1dSOLA, 1.5dRLS)의 등회전선(iso-rotation contour) 지도를 보여줍니다. Fig. 5는 동일 결과의 컬러 표현입니다. 핵심 관측 결과:

Fig. 3 shows iso-rotation contour maps from four methods. Fig. 5 presents the same results in color. Key observations:

1. **대류층에서 회전은 반경에 거의 무관**: 표면 차등 회전 패턴이 대류층 전체에 걸쳐 유지됨
   - Rotation in the convection zone is roughly independent of radius: the surface differential rotation pattern is maintained throughout

2. **등회전선은 원통면이 아님**: 이론적 예측과 달리, 등회전선은 회전축에 평행한 원통과 일치하지 않음. 저위도에서는 원통 경향이 약간 있으나 전체적으로는 쐐기(wedge) 모양
   - Iso-rotation contours are NOT on cylinders: contrary to theoretical predictions, contours do not align with cylinders parallel to the rotation axis

3. **적도에서 최대 회전률**: $\Omega_{\max}/2\pi = 467.3$–$470.1$ nHz, $r \approx 0.93R$–$0.94R$에서 발견
   - Maximum rotation rate at equator: $\Omega_{\max}/2\pi = 467.3$–$470.1$ nHz, found at $r \approx 0.93R$–$0.94R$

4. **대류층 하부에서 약 430 nHz로 균일하게 전이**: 타코클라인을 통해 복사층의 균일 회전으로 전이
   - Uniform transition to about 430 nHz at base of convection zone: transition through tachocline to uniform rotation

#### 표면 차등 회전 맞춤 / Surface Differential Rotation Fit

$r = 0.995R$에서의 역산 회전률을 3항 표현식으로 맞춤:

The inversion rotation rate at $r = 0.995R$ was fit with a three-term expression:

$$\Omega_s = A + B\cos^2\theta + C\cos^4\theta \tag{17}$$

**Table 2**: 맞춤 계수 (위도 60° 이하) / Fit coefficients (below 60° latitude):

| Method | $A/2\pi$ (nHz) | $B/2\pi$ (nHz) | $C/2\pi$ (nHz) |
|---|---|---|---|
| 2dRLS | 455.8 | −51.2 | −84.0 |
| 2dSOLA | 455.4 | −52.4 | −81.1 |
| 1d×1dSOLA | 455.4 | −54.1 | −75.1 |
| 1.5dRLS | 455.1 | −47.3 | −85.9 |
| Ulrich et al. (1988) | 451.5 | −65.3 | −66.7 |

저위도에서 다른 방법들 간의 일치도가 매우 높으나, 고위도(>60°)에서는 상당한 차이가 존재합니다.

Agreement between different methods is very high at low latitudes, but substantial differences exist at high latitudes (>60°).

#### 반경 방향 절단면 / Radial Cuts (Fig. 7)

Fig. 7은 위도 0°, 30°, 60°, 75°에서의 반경 절단면을 보여줍니다:

Fig. 7 shows radial cuts at latitudes 0°, 30°, 60°, 75°:

- **적도 (0°)**: $\Omega/2\pi \approx 460$ nHz로 대류층 전체에서 거의 일정. 근표면에서 약간의 전단
- **30°**: 유사한 패턴, $\Omega/2\pi \approx 440$–$450$ nHz
- **60°**: 현저한 위도 효과, 약 370–380 nHz. 위도 60° 이상에서 방법 간 차이가 증가
- **75°**: 방법 간 불일치가 심함. 신뢰성이 낮은 영역

- **Equator (0°)**: $\Omega/2\pi \approx 460$ nHz, nearly constant throughout the convection zone. Slight shear near surface
- **30°**: Similar pattern, $\Omega/2\pi \approx 440$–$450$ nHz
- **60°**: Pronounced latitude effect, about 370–380 nHz. Method disagreement increases above 60°
- **75°**: Significant disagreement between methods. Low reliability region

---

### Part VI: Results — Subsurface Rotation Shear / 결과 — 근표면 회전 전단 (§5.2, pp. 401–402)

근표면($r > 0.95R$)에서 반경 방향 전단이 존재합니다:

A radial shear exists near the surface ($r > 0.95R$):

- **적도와 30°**: 표면 아래에서 $\Omega$가 약 10 nHz 증가, 최대값은 $r \approx 0.95R$ 부근
  - $\Omega$ increases about 10 nHz below the surface, maximum near $r \approx 0.95R$
- **60°**: 표면 바로 아래에서 $\Omega$가 약간 감소한 후 증가
  - $\Omega$ slightly decreases just below the surface then increases
- **75°**: 결과가 혼란스러움, 방법 간 수 표준편차의 차이
  - Results confused, differences of several standard deviations between methods

Fig. 9는 2dRLS 결과의 근표면 특성을 보여줍니다:
- Fig. 9a: 근표면 최대값(~0.95R)에서의 $\Omega$ 값과 최소값(~0.99R)에서의 값
- Fig. 9b: 0.995R 부근의 반경 방향 기울기 추정

Fig. 9 shows near-surface features from 2dRLS:
- Fig. 9a: $\Omega$ values at near-surface maximum (~0.95R) and minimum (~0.99R)
- Fig. 9b: Radial gradient estimate near 0.995R

이 근표면 전단은 흑점과 초과립(supergranulation)이 표면보다 빠르게 회전하는 관측(약 3% 차이)과 관련될 수 있습니다.

This near-surface shear may be related to the observation that sunspots and supergranulation rotate about 3% faster than the surface.

---

### Part VII: Results — Alternating Bands / 결과 — 교대하는 빠른/느린 회전 대 (§5.3, pp. 402–403)

3항 맞춤(eq. 17)을 뺀 잔차에서 위도에 따른 교대 패턴이 발견됩니다:

An alternating pattern in latitude is found in residuals after subtracting the three-term fit (eq. 17):

- **크기**: 약 1 nHz (속도로 약 5 m s⁻¹)
- **위도 간격**: 10°–15°
- **깊이**: 약 0.05R(또는 0.01–0.02R)까지 추적 가능
- **비교**: 표면 도플러 관측에서 보고된 **비틀림 진동(torsional oscillations)**과 일치

- **Magnitude**: about 1 nHz (velocity of about 5 m s⁻¹)
- **Latitude spacing**: 10°–15°
- **Depth**: traceable to about 0.05R (or 0.01–0.02R)
- **Comparison**: consistent with **torsional oscillations** reported from surface Doppler observations

이 결과는 Kosovichev & Schou (1997)가 f-mode 데이터에서 처음 보고한 "zonal flows"와 일치합니다.

These results are consistent with "zonal flows" first reported by Kosovichev & Schou (1997) from f-mode data.

---

### Part VIII: Results — Structure of the Tachocline / 결과 — 타코클라인의 구조 (§5.4, pp. 402–404)

대류층 하부에서 위도 의존적 차등 회전이 복사층의 거의 균일한 회전으로 급격히 전이됩니다. 이 전이층의 구조를 정량화하기 위해 오차 함수(error function) 맞춤을 사용합니다:

At the base of the convection zone, latitude-dependent differential rotation sharply transitions to nearly uniform rotation in the radiative zone. To quantify this transition, an error function fit is used:

$$\Omega(r) = C_1 + C_2\,\text{erf}\!\left(\frac{r - r_0}{0.5w}\right) \tag{18}$$

여기서 $r_0$는 전이의 중심 위치, $w$는 전이 폭입니다.

Where $r_0$ is the center position of the transition and $w$ is the transition width.

**타코클라인 매개변수** (2dRLS 및 2dSOLA 결과):

**Tachocline parameters** (from 2dRLS and 2dSOLA):

| Parameter | Value |
|---|---|
| 중심 위치 $r_0$ / Center position | $0.70R$–$0.71R$ (적도 및 고위도 60° / equator and high latitude 60°) |
| 폭 $w$ (적도) / Width (equator) | $\leq 0.05R$ (0과 구분 불가 / indistinguishable from zero) |
| 폭 $w$ (위도 15°) / Width (lat 15°) | $0.05$–$0.1R$ (약간 더 두꺼움 / slightly thicker) |
| 위도 30°, 45° / Lat 30°, 45° | 기울기가 너무 작아 폭 결정 불가 / Gradient too small to determine width |
| 고위도 (60°, 75°) / High lat | 전이가 더 넓은 영역에 걸쳐 발생, 대류층 내부까지 확장 / Transition over broader region, extending into convection zone |

**핵심 발견**: 고위도에서 전이가 대류층 하부가 아닌 대류층 자체 내에서 일어나며, 더 넓은 영역에 걸칩니다. 저자들은 이 넓은 전이를 타코클라인의 일부가 아닌, 대류층 내 차등 회전 패턴의 일부로 해석합니다.

**Key finding**: At high latitudes the transition occurs within the convection zone itself rather than at its base, over a broader region. The authors interpret this broad transition not as part of the tachocline, but as part of the differential rotation pattern in the convective envelope.

---

### Part IX: Results — Polar Rotation and Submerged Jet / 결과 — 극 회전과 잠긴 제트 (§5.5, pp. 404–406)

고위도에서의 주목할 만한 특성들:

Notable features at high latitudes:

1. **극 영역의 느린 회전**: 위도 70° 이상에서 $\Omega$가 3항 맞춤에서 외삽한 값보다 현저히 낮음. 이 비정상적 회전은 태양풍의 빠른 성분이 방출되는 영역과 대략 일치하여 태양 주기와의 인과적 연결을 시사
   - Polar regions rotate slower than the three-term fit extrapolation above 70°. This anomalous rotation roughly coincides with the region from which the fast solar wind emanates, suggesting a causal connection with the solar cycle

2. **잠긴 극 제트 (Submerged Polar Jet)**: 위도 약 75°, $r \approx 0.95R$ 근처에서 주변보다 빠르게 회전하는 국소적 영역. Fig. 8의 75° 절단면에서 가장 두드러짐. 2dRLS에서 가장 뚜렷하나 다른 방법에서는 덜 명확
   - Submerged Polar Jet: A localized region near latitude 75°, $r \approx 0.95R$ rotating faster than surroundings. Most pronounced in the 75° cut of Fig. 8. Clearest in 2dRLS but less clear in other methods

3. **제트의 실재성**: 분리 계수의 조합을 모드 전환점과 비교하여 원시 데이터에서도 증가된 회전률의 신호가 확인됨. 그러나 위도 방향 국소화가 불량하여 제트인지 능선(ridge)인지 불분명
   - Reality of jet: Plotting combinations of splitting coefficients against mode turning points confirms the signal of increased rotation rate in raw data. But poor latitudinal localization makes it unclear whether it is a jet or ridge

---

### Part X: Results — Rotation of the Radiative Interior / 결과 — 복사층 내부의 회전 (§5.6, pp. 405–406)

타코클라인 아래, 복사층 내부에서:

Beneath the tachocline, in the radiative interior:

- **회전률**: 약 430 nHz로 거의 균일 (위도에 무관)
- $r \approx 0.6R$, 위도 ~60° 근처에서 약간의 회전률 증가 증거가 있으나, 2dRLS 외삽의 영향일 가능성
- $r < 0.5R$에서는 신뢰할 수 있는 추정이 불가능 (144일 데이터의 한계)
- 태양 핵의 회전은 매우 낮은 차수($l$)의 p-mode 분리가 필요하며, 현재 지상 관측(LOWL, BiSON, IRIS)의 결과들이 서로 일치하지 않음

- **Rotation rate**: approximately 430 nHz, nearly uniform (independent of latitude)
- Some evidence for slight increase near $r \approx 0.6R$, latitude ~60°, but may be an artifact of 2dRLS extrapolation
- Reliable estimates impossible below $r < 0.5R$ (limitation of 144-day data)
- Core rotation requires very low-degree p-mode splittings; current ground-based results (LOWL, BiSON, IRIS) disagree

---

### Part XI: Discussion / 토론 (§6, pp. 406–410)

#### 대류층 내 회전의 물리적 해석 / Physical Interpretation of Convection Zone Rotation

표면 3항 전개(eq. 17)가 놀랍도록 잘 맞지만, 두 가지 세부 특성이 주목됩니다:

The surface three-term expansion (eq. 17) fits remarkably well, but two detailed features are noteworthy:

1. **위도 의존적 변화가 비교적 매끄러움**: 단순 대류 모델에서 예측하는 원통 회전은 관측되지 않음. $a$-계수 분리 계수 최대 $j = 36$까지 사용한 높은 위도 분해능으로 이를 확실히 배제
   - Latitudinal variation is relatively smooth: the rotation-on-cylinders predicted by simple convection models is not observed. High latitudinal resolution (up to $j = 36$) confidently rules this out

2. **비틀림 진동**: 교대하는 약 1 nHz의 빠른/느린 대는 표면 도플러 관측과 일치하며, GONG으로부터 적도 방향으로 약 20년 주기로 전파됨
   - Torsional oscillations: alternating ~1 nHz faster/slower bands are consistent with surface Doppler observations, propagating equatorward over about 20 years

#### 타코클라인의 역학 / Dynamics of the Tachocline

Spiegel & Zahn (1992)을 따라 타코클라인은 대류적으로 안정한 복사층 상부의 전단층으로 정의됩니다. 타코클라인의 역학에 대한 지배적 이론은 없습니다:

Following Spiegel & Zahn (1992), the tachocline is defined as the shear layer in the convectively stable radiative zone top. There is no dominant theory for tachocline dynamics:

- Spiegel & Zahn (1992): 수평적 등방성 전단 발생 2차원 난류가 점성처럼 작용하여 구면 위에서 균일한 각속도를 강제
- Elliott (1997): 후속 연구로 타코클라인 역학 모델링
- McIntyre (1994): 지구 성층권과의 유사성 — 2차원 난류만으로는 균일 회전을 설명할 수 없으며, 자기장이 필요
- Mestel & Weiss (1987), Gough & McIntyre (1998): 복사층 내부의 강체 회전은 대규모 자기장에 의해서만 제공 가능

- Spiegel & Zahn (1992): Horizontally isotropic shear-generated 2D turbulence acting as viscosity to force uniform angular velocity on spherical surfaces
- McIntyre (1994): Analogy with Earth's stratosphere — 2D turbulence alone cannot explain uniform rotation; a magnetic field is needed
- Mestel & Weiss (1987), Gough & McIntyre (1998): Rigidity of radiative interior can only be provided by a large-scale magnetic field

고위도에서의 넓은 전이는 타코클라인과 대류층 내부 전단의 혼합으로 해석됩니다. 적도 근처에서 타코클라인이 복사층 꼭대기에서 관측되는 반면, 고위도에서는 대류층 자체 내에서의 회전률 변화가 더 지배적입니다.

The broad transition at high latitudes is interpreted as a mixture of the tachocline and shear within the convection zone. While the tachocline is observed at the top of the radiative zone near the equator, the rotation rate change within the convection zone itself dominates at high latitudes.

#### 근표면 전단층의 물리 / Physics of Near-Surface Shear Layer

저위도에서 $r = 0.93R$–$0.95R$에 반경 방향 $\Omega$ 증가의 최대값이 있습니다. 이는 흑점과 초과립이 표면보다 약 3% 빠르게 회전하는 관측과 관련됩니다. 각운동량 보존으로 설명하려는 시도가 있었으나 (Foukal 1975, 1977), 정확한 보존은 불가능합니다 (0.07R 깊이에서 14% 증가를 예측하지만 실제로는 4% 증가만 관측).

At low latitudes, the radial $\Omega$ increase has its maximum at $r = 0.93R$–$0.95R$. This is related to the observation that sunspots and supergranulation rotate about 3% faster than the surface. Attempts to explain this via angular momentum conservation (Foukal 1975, 1977) predict a 14% increase at 0.07R depth, but only 4% is observed.

고위도(~60°)에서는 근표면 전단의 부호가 반전됩니다: $\Omega$가 표면 바로 아래에서 감소한 후 약 $r = 0.99R$에서 최소값을 가집니다.

At high latitudes (~60°), the sign of near-surface shear reverses: $\Omega$ decreases just below the surface, reaching a minimum near $r = 0.99R$.

---

### Part XII: Appendix — Hare and Hounds / 부록 — 블라인드 테스트 (Appendix A, pp. 410–414)

두 인공 프로파일(Fig. 13):

Two artificial profiles (Fig. 13):

- **test1**: 대류층에서 회전이 원통면 위에서 거의 일정. 약한 기울기의 타코클라인
  - Rotation nearly constant on cylinders in the convection zone. Weak-gradient tachocline
- **test2**: 실제 태양 역산 결과(2dRLS)의 선형 보간으로 구성. 더 사실적
  - Constructed by linear interpolation of actual solar inversion (2dRLS). More realistic

Hounds의 주요 결론:

Main conclusions from the hounds:

- test1: 원통 회전 패턴 정확히 복원. 대류-복사층 전이 0.70–0.71R로 정확히 추정. 근표면 전단층 복원. $r < 0.4R$ 정보 없음
- test2: 태양과 매우 유사한 프로파일이라고 올바르게 결론. 타코클라인 ~0.71R에서 복원. 고위도 제트(70°–80°) 검출

---

## 3. Key Takeaways / 핵심 시사점

1. **대류층 차등 회전은 깊이에 거의 무관하다** — 표면에서 관측되는 위도별 차등 회전 패턴(적도 ~460 nHz, 극 ~340 nHz)이 대류층 전체에 걸쳐 유지됩니다. 이는 원통면 위에서 일정한 회전을 예측하는 이론과 모순되며, 각운동량 전달 메커니즘에 대한 재고가 필요함을 의미합니다.
   - Convection zone differential rotation is nearly independent of depth — the surface latitudinal differential rotation pattern (equator ~460 nHz, poles ~340 nHz) is maintained throughout the convection zone. This contradicts theories predicting rotation constant on cylinders and requires rethinking of angular momentum transport.

2. **타코클라인은 매우 얇은 전이층이다** — 적도에서 타코클라인의 폭은 $w \leq 0.05R$ (약 35,000 km 이하)로, 차등 회전이 균일 회전으로 전이되는 급격한 전단층입니다. 이 얇은 구조는 자기 다이나모 이론에서 자기장 생성의 핵심 장소로서의 역할을 뒷받침합니다.
   - The tachocline is a very thin transition layer — at the equator, the tachocline width is $w \leq 0.05R$ (less than about 35,000 km), a sharp shear layer where differential rotation transitions to uniform rotation. This thin structure supports its role as the key site for magnetic field generation in dynamo theory.

3. **복사층은 약 430 nHz로 거의 균일하게 회전한다** — 위도에 무관하게 거의 일정한 회전률을 보이며, 이는 대규모 자기장에 의한 강성 효과로 설명될 수 있습니다.
   - The radiative zone rotates nearly uniformly at about 430 nHz — rotation rate is nearly constant regardless of latitude, possibly explained by the rigidity effect of a large-scale magnetic field.

4. **근표면 전단층의 위도 의존성이 복잡하다** — 저위도에서는 표면 아래 약 0.95R에서 회전률이 최대값을 가지나, 고위도(>60°)에서는 반대로 표면 아래에서 감소합니다. 이 비대칭적 전단은 대류와 회전의 상호작용에 대한 중요한 단서입니다.
   - The near-surface shear layer has complex latitude dependence — at low latitudes, rotation rate peaks at about 0.95R below the surface, but at high latitudes (>60°) it decreases below the surface. This asymmetric shear provides important clues about convection-rotation interaction.

5. **위도 75° 부근에 잠긴 극 제트가 존재할 수 있다** — $r \approx 0.95R$에서 주변보다 빠르게 회전하는 국소적 구조가 발견되었습니다. 원시 데이터에서도 확인되지만 다른 역산 방법에서는 덜 뚜렷하여, 시간 의존성과 비축대칭성을 고려한 추가 관측이 필요합니다.
   - A submerged polar jet may exist near latitude 75° — a localized structure at $r \approx 0.95R$ rotating faster than surroundings was found. Confirmed in raw data but less clear in other inversion methods, requiring additional observations considering time dependence and non-axisymmetry.

6. **비틀림 진동(torsional oscillations)이 내부에서도 확인된다** — 표면에서 관측되던 약 1 nHz 크기의 교대 빠른/느린 회전 대가 약 0.05R 깊이까지 추적됩니다. 이 흐름은 약 10°–15° 위도 간격으로 나타나며, 적도 방향으로 전파됩니다.
   - Torsional oscillations are confirmed in the interior — alternating faster/slower rotation bands of ~1 nHz magnitude observed at the surface are traced to about 0.05R depth. These flows appear at ~10°–15° latitude intervals and propagate equatorward.

7. **7개의 독립적 역산 기법이 강건한 결과를 보장한다** — RLS와 OLA 두 부류의 7가지 방법이 같은 데이터에 적용되었으며, 대류층과 근표면의 주요 결과에서 높은 일치도를 보입니다. Hare-and-hounds 블라인드 테스트로 방법론의 신뢰성이 검증되었습니다.
   - Seven independent inversion techniques guarantee robust results — seven methods from two classes (RLS and OLA) were applied to the same data, showing high agreement for major results in the convection zone and near-surface. Hare-and-hounds blind tests validated methodological reliability.

8. **MDI의 연속 관측이 전례 없는 데이터 품질을 가능하게 했다** — 95.47%의 듀티 사이클, $l = 1$–250의 넓은 모드 범위, $j = 36$까지의 분리 계수가 지상 관측 대비 월등한 위도 분해능과 정밀도를 제공합니다.
   - MDI's continuous observation enabled unprecedented data quality — 95.47% duty cycle, broad mode range $l = 1$–250, splitting coefficients up to $j = 36$ provide vastly superior latitudinal resolution and precision compared to ground-based observations.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 회전 분리의 순방향 문제 / Forward Problem of Rotational Splitting

태양 진동 모드 $(n, l, m)$의 주파수 분리:

Frequency splitting of solar oscillation mode $(n, l, m)$:

$$\Delta\omega_{nlm} = \omega_{nlm} - \omega_{nl} = \int_0^R \int_0^{\pi} K_{nlm}(r, \theta)\,\Omega(r, \theta)\,r\,dr\,d\theta$$

모드 커널의 분리 가능한 형태:

Separable form of mode kernels:

$$K_{nlm} = F_{1nl}(r)\,G_{1lm}(\theta) + F_{2nl}(r)\,G_{2lm}(\theta)$$

여기서 두 번째 항은 $l^2$의 인자만큼 작습니다.

Where the second term is smaller by a factor of order $l^2$.

### 4.2 분리 계수 전개 / Splitting Coefficient Expansion

멀티플렛 주파수를 직교 다항식으로 전개:

Multiplet frequencies expanded in orthogonal polynomials:

$$\omega_{nlm}/2\pi = \nu_{nl} + \sum_{j=1}^{j_{\max}} a_j(n, l)\,\mathcal{P}_j^{(l)}(m)$$

분리 계수와 회전의 관계:

Relation between splitting coefficients and rotation:

$$2\pi a_{2s+1}(n, l) = \int_0^R K_{nlj}^{\prime}(r)\,\Omega_s(r)\,dr$$

여기서 $\Omega(r, \theta) = \sum_s \Omega_s(r)\,\psi_{2s}^{(1)}(\cos\theta)$, $\psi_{2s}^{(1)}(x) = dP_{2s+1}/dx$.

### 4.3 역산 / Inversion

일반 역산 형식:

General inversion form:

$$d_i = \int_0^{\pi/2} \int_0^R K_i(r, \theta)\,\Omega(r, \theta)\,r\,dr\,d\theta + \epsilon_i$$

역산 해:

Inversion solution:

$$\hat{\Omega}(r_0, \theta_0) = \sum_{i=1}^{M} c_i(r_0, \theta_0)\,d_i$$

평균 커널:

Averaging kernel:

$$\mathscr{K}(r_0, \theta_0, r, \theta) = \sum_{i=1}^{M} c_i(r_0, \theta_0)\,K_i(r, \theta)$$

오차 분산:

Error variance:

$$\sigma[\hat{\Omega}]^2 = \sum_{i=1}^{M} [c_i\,\sigma_i]^2$$

### 4.4 2dSOLA 목적 함수 / 2dSOLA Objective Function

$$\min_{c_i} \left\{ \int\int [\mathscr{T} - \mathscr{K}]^2\,r\,dr\,d\theta + \lambda \sum_i [\sigma_i\,c_i]^2 \right\}$$

subject to $\int\int \mathscr{K}\,r\,dr\,d\theta = 1$ (unimodularity).

### 4.5 표면 차등 회전 / Surface Differential Rotation

$$\frac{\Omega(\theta)}{2\pi} = A + B\cos^2\theta + C\cos^4\theta$$

평균 맞춤 값: $A/2\pi \approx 455$ nHz, $B/2\pi \approx -51$ nHz, $C/2\pi \approx -82$ nHz.

### 4.6 타코클라인 프로파일 / Tachocline Profile

$$\Omega(r) = C_1 + C_2\,\text{erf}\!\left(\frac{r - r_0}{0.5w}\right)$$

최적 값: $r_0 = 0.692R \pm 0.05R$, $w = 0.09R \pm 0.04R$.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1960s   Howard & Harvey: 표면 차등 회전의 체계적 관측 확립
        Systematic observation of surface differential rotation established

1975    Deubner: 5분 진동의 p-mode 본성 확인
        Confirms p-mode nature of 5-min oscillations

1984    Duvall & Harvey: 회전 분리(rotational splitting) 첫 측정
        First measurement of rotational splitting

1985    Brown: 내부 차등 회전의 첫 일진학적 증거
        First helioseismic evidence of interior differential rotation

1988    Brown et al.: 타코클라인 개념 도입
        Introduction of tachocline concept

1989    Dziembowski, Goode, & Libbrecht: 역산으로 회전 프로파일 추정
        Rotation profile estimated via inversions

1992    Spiegel & Zahn: 타코클라인 역학 이론 정립
        Tachocline dynamics theory established

1995    SOHO 위성 발사 (MDI 탑재)
        SOHO spacecraft launched (with MDI)

1996    Thompson et al.: GONG 데이터로 내부 회전 매핑
        Interior rotation mapping with GONG data

1996    Christensen-Dalsgaard et al.: 표준 태양 모델 리뷰 (#16)
        Standard solar model review

1997    Kosovichev & Schou: f-mode에서 zonal flows 첫 보고
        First report of zonal flows from f-mode

>>>>    1998    Schou et al.: MDI 데이터로 최고 정밀도 차등 회전 매핑 ◀ THIS PAPER
                Highest-precision differential rotation mapping with MDI data

1998    Gough & McIntyre: 타코클라인의 자기장 역할 이론
        Theory of magnetic field role in tachocline

2000    Howe et al.: 타코클라인의 시간 변화(1.3년 주기) 보고
        Report of tachocline temporal variations (1.3-year period)

2003    Basu & Antia: 타코클라인 구조의 정밀 측정
        Precision measurement of tachocline structure

2010s   Kepler/TESS 시대: 별지진학에서 내부 회전 측정으로 확장
        Kepler/TESS era: extension to interior rotation in asteroseismology
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#9 Babcock (1961)** | 차등 회전이 극향→환향 자기장 변환의 핵심이라는 다이나모 모델 제안 / Proposed dynamo model where differential rotation converts poloidal→toroidal fields | Schou et al.이 이 차등 회전의 내부 구조를 직접 관측하여 Babcock 모델의 물리적 기반 제공 / Schou et al. directly observed the internal structure of this differential rotation, providing physical basis for Babcock's model |
| **#16 Christensen-Dalsgaard et al. (1996)** | 일진학으로 제약된 표준 태양 모델 리뷰 / Review of standard solar model constrained by helioseismology | 같은 일진학 기법을 회전 프로파일에 적용한 직접적 후속 연구 / Direct follow-up applying same helioseismic techniques to rotation profile |
| **Thompson et al. (1996)** | GONG 데이터로 내부 회전 매핑 / Interior rotation mapping with GONG data | Schou et al.은 MDI의 우월한 데이터로 같은 목표를 더 높은 정밀도로 달성 / Schou et al. achieved same goal with higher precision using superior MDI data |
| **Spiegel & Zahn (1992)** | 타코클라인 역학의 이론적 기틀 제공 / Provided theoretical framework for tachocline dynamics | Schou et al.의 관측이 타코클라인의 존재와 구조를 정밀하게 확인 / Schou et al.'s observations precisely confirmed existence and structure of tachocline |
| **Kosovichev & Schou (1997)** | f-mode에서 zonal flows(비틀림 진동) 첫 보고 / First report of zonal flows from f-mode | Schou et al.이 p-mode 포함 전체 역산으로 이를 확인하고 깊이 정보 추가 / Schou et al. confirmed with full inversions including p-modes and added depth information |
| **Gough & McIntyre (1998)** | 타코클라인에서 자기장의 역할 이론화 / Theorized role of magnetic field in tachocline | Schou et al.의 관측(특히 얇은 타코클라인, 균일한 복사층 회전)이 이 이론의 관측적 동기 / Schou et al.'s observations (thin tachocline, uniform radiative rotation) motivated this theory |

---

## 7. References / 참고문헌

- Schou, J., et al., "Helioseismic Studies of Differential Rotation in the Solar Envelope by the Solar Oscillations Investigation Using the Michelson Doppler Imager," ApJ, 505, 390–417, 1998. [DOI: 10.1086/306146]
- Babcock, H. W., "The Topology of the Sun's Magnetic Field and the 22-Year Cycle," ApJ, 133, 572, 1961.
- Christensen-Dalsgaard, J., et al., "The Current State of Solar Modeling," Science, 272, 1286, 1996.
- Thompson, M. J., et al., "Differential Rotation and Dynamics of the Solar Interior," Science, 272, 1300, 1996.
- Spiegel, E. A., & Zahn, J.-P., "The Solar Tachocline," A&A, 265, 106, 1992.
- Kosovichev, A. G., & Schou, J., "Detection of Zonal Shear Flows beneath the Sun's Surface," ApJ, 482, L207, 1997.
- Gough, D. O., & McIntyre, M. E., "Inevitability of a Magnetic Field in the Sun's Radiative Interior," Nature, in press, 1998.
- Brown, T. M., et al., "Inferring the Sun's Internal Angular Velocity from Observed p-Mode Frequency Splittings," ApJ, 343, 526, 1989.
- Duvall, T. L., Jr., et al., "Internal Rotation of the Sun," Nature, 310, 22, 1984.
- Ulrich, R. K., et al., "Solar Rotation Measurements at Mt. Wilson," Sol. Phys., 117, 291, 1988.
- Howe, R., & Thompson, M. J., "On the Use of the Error Function for the Tachocline," MNRAS, 281, 1385, 1996.
- Pijpers, F. P., & Thompson, M. J., "The SOLA Method for Helioseismic Inversion," A&A, 262, L33, 1992.
- Larsen, R. M., & Hansen, P. C., "Efficient Implementation of the SOLA Mollifier Method," A&AS, 121, 587, 1997.
