---
title: "The Electron Density of the Solar Corona"
authors: ["H. C. van de Hulst"]
year: 1950
journal: "Bulletin of the Astronomical Institutes of the Netherlands, Vol. XI, No. 410, pp. 135-149"
doi: "1950BAN....11..135V"
topic: Solar_Observation
tags: [coronagraphy, K-corona, F-corona, Thomson-scattering, electron-density, polarization, pB-inversion, eclipse-photometry]
status: completed
date_started: 2026-04-28
date_completed: 2026-04-28
---

# 62. The Electron Density of the Solar Corona / 태양 코로나의 전자 밀도

---

## 1. Core Contribution / 핵심 기여

### English
This paper establishes the **canonical procedure for self-consistently deriving the coronal electron density profile N(r) from white-light brightness and polarization observations of the K-corona**. van de Hulst (i) cleanly separates the F-corona (zodiacal-light, dust-diffraction component) from the K-corona (free-electron Thomson-scattered component) via three independent methods (residual Fraunhofer-line strength, the assumption of unpolarized F-corona, and direct subtraction); (ii) re-derives, from the Schuster-Minnaert vibration-ellipsoid framework, the line-of-sight integral equations for the tangential and radial brightness components $K_t(x)$ and $K_r(x)$, and writes them in the now-classical form (his Eqs. 17–20) that is **the direct ancestor of every modern pB-inversion**; (iii) presents a rigorously self-consistent "model corona" — a set of tables (Tables 2 and 5A, B) giving brightness, polarization, and electron density for equatorial and polar regions in both minimum and maximum phases, accurate to ~1%. The derivation explicitly avoids Baumbach's approximation $A=B=\tfrac{1}{3}(2A+B)$, which is shown to be in error by up to 18%.

In §5 the equatorial framework is extended to the polar regions through a detour: the spherically asymmetric problem is first solved under isotropic-scattering assumption via the Abel pair (Eqs. 29–30), then anisotropy is restored through the multiplicative factors of Eqs. (33)–(35). The principal new astrophysical result is the discovery of an **electron-density minimum near heliographic latitude β ≈ 70°** (Figure 7B), which van de Hulst interprets as evidence for two distinct agents producing the minimum-phase corona — one acting near the equator and one near the pole — anticipating by decades our understanding of streamer belts and polar coronal holes. The paper closes with a comparison of the "model corona" against observed polarization (Figure 8), revealing a residual mismatch in the outer corona and discussing three candidate causes (sky-light over-correction, intrinsic F-corona polarization, F overestimation).

### 한국어
이 논문은 **K-코로나의 백색광 표면밝기와 편광 관측으로부터 코로나 전자 밀도 분포 N(r)을 자기무모순적(self-consistent)으로 도출하는 표준 절차**를 확립한 고전 논문이다. van de Hulst는 (i) F-코로나(행성간 먼지 회절 성분, 황도광)와 K-코로나(자유전자 톰슨 산란 성분)를 세 가지 독립 방법(Fraunhofer 선 잔류 강도, F-코로나 비편광 가정, 직접 차감)으로 분리하고, (ii) Schuster-Minnaert의 진동 타원체(vibration ellipsoid) 형식에서 시선 적분 방정식 $K_t(x)$, $K_r(x)$를 재유도하여 오늘날 표준이 된 식 (17)–(20) 형태로 정리한다. 이 식들은 **현대 모든 pB 인버전(pB-inversion)의 직접적 조상**이다. (iii) 그 결과는 적도/극 영역, 최소/최대 활동기에 대해 1% 정밀도로 자기무모순적인 표 (Table 2, 5A, 5B)로 제시되며, 이는 "model corona"라 명명된다. 유도 과정에서는 Baumbach의 근사식 $A = B = \tfrac{1}{3}(2A+B)$가 18%의 오차를 가짐을 명시적으로 보이고 이를 회피한다.

§5에서는 적도 형식을 극 영역으로 확장하기 위해 우회 절차를 사용한다. 구대칭이 깨진 문제를 우선 등방 산란 가정 하에 Abel 변환쌍 (Eqs. 29–30)으로 풀고, 이후 식 (33)–(35)의 보정 인자로 비등방성을 회복한다. 본 논문의 새로운 천체물리적 발견은 **헬리오그래픽 위도 β ≈ 70° 부근의 전자 밀도 최솟값**(Figure 7B)이다. van de Hulst는 이를 최소 활동기 코로나가 두 개의 서로 다른 작용 — 적도 영역과 극 영역 — 에 의해 만들어진다는 증거로 해석했고, 이는 수십 년 후 streamer belt와 polar coronal hole의 이해를 예견한 것이다. 논문은 마지막으로 model corona를 관측 편광과 비교(Figure 8)하여 외부 코로나에서의 잔여 불일치를 보이고 세 가지 후보 설명(sky light 과대보정, F-corona 자체 편광, F 과대평가)을 논의한다.

---

## 2. Reading Notes / 읽기 노트

### Part I — §1 Introduction / 서론 (pp. 135–136)

#### English
van de Hulst opens by defining three components of coronal light: (1) **forbidden emission lines** of highly ionised metals contributing only ~0.5% of the integrated visible light; (2) the **continuum from Thomson-scattered photospheric light by free electrons** — the "real" K-corona; (3) an **outer haze that merges into the inner zodiacal light** — the F-corona, attributed to diffraction by interplanetary particles. The paper deals with the second and major part. Reliable photometry is still confined to total-eclipse data. Earlier comprehensive analyses by Schuster (1879), Minnaert (1930), Baumbach (1937) had given a first reference, but Baumbach's compilation is shown to contain several approximations: F-corona was not removed properly; electron densities in the outer parts were biased high; the derivation method was approximate; no attention was paid to polar regions or to solar-cycle changes. The motivation for new accurate values is **radio astronomy**: most solar radio waves between 1 m and 10 m wavelength originate in the corona, and the opacity at these wavelengths is proportional to $N_e^2$, so accurate $N_e(r)$ is required. Rather than entering an open debate on details, van de Hulst chooses to compute "tables giving mutually consistent values of brightness, polarization and electron densities, which fit the available data reasonably well" — the **"model corona"** concept. The investigation was made possible by a March 1948 visit to Lick Observatory, where photometric copies of large-scale eclipse plates from 1893–1932 were obtained.

#### 한국어
van de Hulst는 코로나 빛을 세 성분으로 정의한다: (1) 고전리 금속의 **금지 방출선** — 가시광 적분강도의 약 0.5%만 차지; (2) **자유전자에 의한 광구광의 톰슨 산란 연속체** — 진정한 K-코로나; (3) **외곽으로 갈수록 황도광에 합류하는 헤이즈** — F-코로나로, 행성간 먼지 입자에 의한 회절광으로 해석된다. 본 논문은 두 번째 성분(K-corona)을 주로 다룬다. 신뢰할 만한 측광은 여전히 개기일식 자료에 한정되어 있다. Schuster(1879), Minnaert(1930), Baumbach(1937)의 종합적 작업이 1차 표준이었으나 Baumbach의 자료에는 여러 근사가 포함되어 있다: F-corona가 제대로 제거되지 않았고, 외부 영역의 전자 밀도가 높게 편향되었으며, 유도 방법 자체가 근사였고, 극 영역과 태양주기 변화는 다루어지지 않았다. 새로운 정확한 값을 위한 동기는 **라디오 천문학**이다. 1–10 m 파장의 태양 라디오 방출 대부분은 코로나에서 발생하며, 흡수계수는 $N_e^2$에 비례하므로 정확한 $N_e(r)$이 필수적이다. van de Hulst는 세부 논쟁에 들어가는 대신 "관측을 합리적으로 맞추면서 밝기·편광·전자 밀도가 서로 일관된 표"를 계산하는 길 — **"model corona"** 개념 — 을 택한다. 이 연구는 1948년 3월 Lick Observatory 방문을 계기로 가능해졌으며, 1893–1932년 사이의 대형 일식 사진건판 복사본이 입력 자료가 되었다.

---

### Part II — §2 Separation of F- and K-components / F·K 성분 분리 (pp. 136–137)

#### English
Following Grotrian, van de Hulst denotes by $K$ the K-corona surface brightness and by $F$ the F-corona surface brightness. The chosen unit is $10^{-8}$ times the average surface brightness of the Sun. (Baumbach used $10^{-6}$ times the brightness of disk centre — these differ by a factor 125 in absolute calibration.) Observations measure the combined $F+K$, and the central question is the factor

$$
f \;=\; \frac{K}{F+K} \tag{1}
$$

by which the observed brightness must be multiplied to recover the K-corona alone. Both K and F components are assumed to share the same colour as the integrated Sun (a slight reddening of F, $\propto \lambda^{1/2}$, is acknowledged but ignored). Three methods for $f$ are presented:

**Method 1 — Fraunhofer-line residual.** Fraunhofer lines retain their full strength in the F-corona but are completely obliterated in the K-corona by the huge thermal Doppler broadening of free electrons. Letting $r$ denote the residual intensity at the centre of a Fraunhofer line:

$$
1 - f \;=\; \frac{1 - r_{\text{corona}}}{1 - r_{\text{disk}}} \tag{2}
$$

**Method 2 — Polarization-based.** If the F-corona is assumed unpolarized, and the K-corona's polarization $p_K$ can be computed from Rayleigh's law given an electron-density distribution, then with observed polarization $p$:

$$
f \;=\; \frac{p}{p_K} \tag{3}
$$

**Method 3 — Direct subtraction.** Whenever $F$ can be estimated independently (e.g. by extrapolating zodiacal-light isophotes inward, which are nearly circular within a few solar radii):

$$
f \;=\; 1 - \frac{F}{F+K}
$$

van de Hulst notes that the assumed properties of the F-corona become increasingly uncertain from method 1 to method 3, but consistent results across the three methods support their approximate correctness.

#### 한국어
Grotrian의 표기를 따라, van de Hulst는 K-corona 표면밝기를 $K$, F-corona 표면밝기를 $F$로 표기한다. 단위는 태양 평균 표면밝기의 $10^{-8}$배 (Baumbach가 사용한 디스크 중심 밝기의 $10^{-6}$배와는 절대 캘리브레이션이 인자 125만큼 다르다). 관측은 합 $F+K$를 측정하며, 핵심 질문은 K-corona만을 회복하기 위해 관측에 곱해야 할 인자

$$
f \;=\; \frac{K}{F+K} \tag{1}
$$

이다. K와 F 둘 다 태양 적분광과 같은 색을 가진다고 가정한다(F의 약한 적색화 $\propto \lambda^{1/2}$는 무시). $f$를 결정하는 세 가지 방법:

**방법 1 — Fraunhofer 선 잔류 강도.** Fraunhofer 선은 F-corona에서는 본래 강도를 유지하지만, 자유전자의 거대한 열적 도플러 폭으로 인해 K-corona에서는 완전히 소멸된다. 코로나·디스크의 잔류 강도를 $r$로 표기하면:

$$
1 - f \;=\; \frac{1 - r_{\text{corona}}}{1 - r_{\text{disk}}} \tag{2}
$$

**방법 2 — 편광 기반.** F-corona가 비편광이라 가정하고, K-corona 편광도 $p_K$를 Rayleigh 법칙으로 계산할 수 있다면, 관측 편광도 $p$로부터:

$$
f \;=\; \frac{p}{p_K} \tag{3}
$$

**방법 3 — 직접 차감.** $F$를 독립적으로 추정 가능할 때(예: 황도광 등휘선이 수 태양반경 이내에서 거의 원형이라는 사실을 이용한 외삽). van de Hulst는 방법 1→3로 갈수록 가정의 불확실성이 커지지만 세 결과가 일관되면 정확성을 뒷받침한다고 본다.

---

### Part III — §3 Brightness Distribution / 표면밝기 분포 (pp. 137–141)

#### English
**Absolute intensity.** Two methods exist for absolute calibration: photographic photometry of eclipse plates, and photoelectric/radiometric measurement of total integrated brightness. The latter is more reliable. Eclipse rings (limited inside by the Moon's limb, outside by the photometer field) measure not the total but a partial brightness. The Moon screens about 46% of the K-corona in maximum phase. Nikonov reduced published values of six eclipse expeditions to common values $r=1.03$ and $r=6$. Figure 1 plots total ring brightness ($r=1.03$ to $r=6$, in units $10^{-6}$ of the Sun's brightness) against solar-cycle phase: a clear correlation with activity is seen, with a brightness ratio between maximum and minimum of $Q = 1.7 \pm 0.3$, giving a K-corona scale factor

$$
c \;=\; K_{\max}(r) / K_{\min}(r) \;=\; 1.78 \quad (\text{adopted})
$$

van de Hulst chose $c = 1.78$ for the model. The final Table 1 row totals are: $K_{\max} = 1.213$, $K_{\min} = 0.683$, $K_{\text{pole}} = 0.305$, $F = 0.259$ (units of $10^{-6}$ of the Sun's total brightness). The polar regions in minimum phase are weaker than equatorial regions; van de Hulst adopts a model in which equatorial and polar regions extend over sectors of 0.7 and 0.3 of the solar circumference, so $K'_{\min} = 0.7\,K_{\min}(r) + 0.3\,K_{\text{pole}}(r)$.

**Power-series representation (Eqs. 5–9).** Following Baumbach who fitted

$$
\big[(c^{1/2}\,K_{\min} + F) / a\big] \;=\; 2.56\,r^{-17} + 1.43\,r^{-7} + 0.053\,r^{-2.5} \tag{4}
$$

van de Hulst then disentangled $K_{\min}$, $K_{\max}$, and $F$ separately using Grotrian's $f$ values. With absolute calibration factor $a = 104$ (from Figure 1), the **final formulae** are:

$$
K_{\max}(r) \;=\; 355.6\,r^{-17} + 177.8\,r^{-7} + 0.708\,r^{-2.5} \tag{5}
$$

$$
K_{\min}(r) \;=\; 200.0\,r^{-17} + 100.0\,r^{-7} + 0.398\,r^{-2.5} \tag{6}
$$

$$
F(r) \;=\; \quad\quad\quad\quad\quad\quad\quad\quad 14.86\,r^{-7} + 4.99\,r^{-2.5} \tag{7}
$$

For the polar axis, derived from Bergstrand 1914 and Lick 1900 plates:

$$
K_{\text{pole}}(r) + F(r) \;=\; 191.0\,r^{-17} + 27.45\,r^{-7} + 4.99\,r^{-2.5} \tag{8}
$$

$$
K_{\text{pole}}(r) \;=\; 191.0\,r^{-17} + 12.59\,r^{-7} \tag{9}
$$

Note that $K_{\min}$ has $r^{-17}$ and $r^{-7}$ coefficients exactly half of $K_{\max}$ (because $c^{1/2} \approx 1.33$), and $K_{\text{pole}}$ has the same $r^{-17}$ coefficient as $K_{\min}$ but a much smaller $r^{-7}$ contribution and no $r^{-2.5}$ tail at all. The total brightness of a ring between $r_1$ and $r_2$ is

$$
B \;=\; \sum_n (r_1^{-n+2} - r_2^{-n+2})\,\frac{2 C_n}{n-2} \tag{10}
$$

**Table 2** gives the surface brightness of the model corona in units of $10^{-8}$ of the Sun's average brightness. Sample values: at $r = 1.0$, $K_{\max} = 534.1$, $K_{\min} = 300.4$, $K_{\text{pole}} = 203.6$; at $r = 2.0$, the values drop to $1517$, $852$, $100$ (×$10^{-3}$ in the same unit); at $r = 5$, only $0.0149$, $0.0084$, $0.0002$. The fraction $f = K/(K+F)$ falls from 0.964 at $r=1$ (max) to 0.143 at $r=5$, illustrating how F-corona dominates the outer corona. The radius where $f = 0.50$ is $r = 2.24$ (max), 1.93 (min), and 1.28 (pole).

#### 한국어
**절대 강도.** 절대 보정은 두 방법 — 사진건판 측광과 광전/라디오미터 적분 측광 — 이 있고 후자가 더 신뢰성 있다. 일식 측광 링은 달의 가장자리(내부)와 광도계 시야(외부)로 한정된 부분 밝기를 측정하며, 달은 최대기 K-corona의 약 46%를 가린다. Nikonov는 6개 일식 자료를 $r=1.03$, $r=6$의 공통 값으로 환산했다. Figure 1은 ($r=1.03$–6) 적분 밝기를 태양주기 위상의 함수로 표시하며, 활동도와 명확한 상관을 보인다. 최대/최소 비는 $Q = 1.7 \pm 0.3$이고 K-corona 인자는

$$
c \;=\; K_{\max}(r)/K_{\min}(r) \;=\; 1.78 \quad (\text{채택})
$$

이다. Table 1의 행 총합: $K_{\max}=1.213$, $K_{\min}=0.683$, $K_{\text{pole}}=0.305$, $F=0.259$ (태양 총 밝기의 $10^{-6}$ 단위). 최소 활동기 극 영역은 적도 영역보다 약하므로, van de Hulst는 적도/극 영역이 각각 태양 둘레의 0.7과 0.3을 차지한다는 가중 모델 $K'_{\min}=0.7\,K_{\min}+0.3\,K_{\text{pole}}$을 채택한다.

**멱급수 표현 (Eqs. 5–9).** Baumbach의 fit (Eq. 4)에서 출발해 Grotrian의 $f$ 값으로 $K_{\min}$, $K_{\max}$, $F$를 분리하고, Figure 1에서 결정한 절대 보정 $a=104$를 적용한 **최종 공식**이 식 (5)–(9)이다 (영문 본문 참조). $K_{\min}$의 $r^{-17}$, $r^{-7}$ 계수가 $K_{\max}$의 정확히 절반인 것은 $c^{1/2}\approx 1.33$ 때문이며, $K_{\text{pole}}$은 $K_{\min}$과 같은 $r^{-17}$ 계수를 갖지만 $r^{-7}$ 기여가 훨씬 작고 $r^{-2.5}$ 꼬리는 전혀 없다. 링 적분식 (10)은 멱급수 항별 적분의 직접 결과다.

**Table 2**는 model corona 표면밝기를 $10^{-8}$ 단위로 제시한다. 대표값: $r=1.0$에서 $K_{\max}=534.1$, $K_{\min}=300.4$, $K_{\text{pole}}=203.6$; $r=2.0$에서 $1.517$, $0.852$, $0.100$; $r=5$에서 $0.0149$, $0.0084$, $0.0002$. 비율 $f=K/(K+F)$는 $r=1$ (max)에서 0.964이지만 $r=5$에서 0.143까지 감소하여, 외부 코로나에서 F가 압도적임을 보여준다. $f=0.50$이 되는 반경은 $r=2.24$(max), 1.93(min), 1.28(pole)이다.

---

### Part IV — §4 Polarization and Electron Densities / 편광과 전자 밀도 (pp. 141–144) ★

This is the central section of the paper.

#### English

**Vibration ellipsoid framework.** Let $R = 6.97 \times 10^{10}$ cm be the solar radius, $H$ the mean surface brightness of the Sun, $q$ the limb-darkening coefficient. The brightness of a point on the disk seen at angle $\alpha$ from the normal is $H(1 - q + q\cos\alpha)/(1 - \tfrac{1}{3}q)$. At a coronal point $P$ at distance $rR$ from the Sun's centre, the **density of illumination** integrated over the entire solid angle subtended by the Sun has the form

$$
\pi H \big\{ 2A(r) + B(r) \big\}
$$

where $2A$ is the fraction proportional to the mean square of the electric vector components in any **transversal** direction (perpendicular to the radius from the Sun's centre to $P$), and $B$ is the fraction in the **radial** direction. Because $B$ would be zero if the Sun were a point source, it decreases rapidly with $r$; meanwhile $A \to 1/(2r^2)$ as $r \to \infty$. The full expressions, derived by Schuster and Minnaert via elementary integration with $\sin\gamma = 1/r$, are:

$$
2A + B = \frac{1-q}{1 - q/3}\Big\{2(1 - \cos\gamma)\Big\} + \frac{q}{1 - q/3}\Big\{1 - \frac{\cos^2\gamma}{\sin\gamma}\log\frac{1 + \sin\gamma}{\cos\gamma}\Big\} \tag{11}
$$

$$
2A - B = \frac{1-q}{1 - q/3}\Big\{\tfrac{2}{3}(1 - \cos^3\gamma)\Big\} + \frac{q}{1 - q/3}\Big\{\tfrac{1}{4} + \frac{\sin^2\gamma}{4} - \frac{\cos^4\gamma}{4\sin\gamma}\log\frac{1+\sin\gamma}{\cos\gamma}\Big\} \tag{12}
$$

Minnaert denotes the four expressions in braces by $\tfrac{1}{2}(3C-A)$, $\tfrac{1}{2}(3D-B)$, $\tfrac{1}{2}(C+A)$, $\tfrac{1}{2}(D+B)$. Table 3 tabulates $A$, $B$ for $q = 0$, $q = 1$, $q = 0.75$ (used throughout, corresponding to effective wavelength near 4700 Å). Sample values at $q = 0.75$: $r = 1.0$: $2A+B = 1.6667$, $2A - B = 0.7222$, $A = 0.5972$, $A - B = -0.1250$; $r = 1.5$: $2A+B = 0.4991$, $2A-B = 0.3982$, $A = 0.2243$, $A - B = 0.1739$; $r = 4$: $A = 0.0312$, $A - B = 0.0304$.

**Scattering integral.** With $N(r)$ the electron number density and $\sigma = 0.66 \times 10^{-24}$ cm² the Thomson cross-section, the light scattered per cm³ per second in all directions is

$$
4\pi \overline{J} = \pi H (2A + B)\,\sigma N \tag{13}
$$

In a particular direction making angle $\theta$ with the radius (geometry in Figure 5), the source function for light **vibrating perpendicular to the plane through the Sun's centre** ($J_t$) and **vibrating in that plane** ($J_r$) are:

$$
J_t = \tfrac{3}{8} H N \sigma\,A(r) \tag{14}
$$

$$
J_r = \tfrac{3}{8} H N \sigma\,\big\{A(r)\cos^2\theta + B(r)\sin^2\theta\big\} \tag{15}
$$

The line-of-sight integral over a column of unit cross-section, with $yR$ the distance along the line of sight from the point closest to the Sun and projected radius $x$ (so that $r = \sqrt{x^2 + y^2}$), gives

$$
10^{-8}\,H\,K(x) = \int_{-\infty}^{+\infty} \overline{J}(r)\,R\,dy \tag{16}
$$

Substituting (13) and changing variables to $r$ (so that $dy = r\,dr / \sqrt{r^2 - x^2}$ and the integral runs from $x$ to $\infty$, doubled), this becomes the **central integral equation**:

$$
\boxed{\;K(x) = C \int_x^{\infty} N(r)\,\Big\{(2 - \tfrac{x^2}{r^2})\,A(r) + \tfrac{x^2}{r^2}\,B(r)\Big\}\,\frac{r\,dr}{\sqrt{r^2 - x^2}}\;} \tag{17}
$$

Decomposing into separate polarization directions:

$$
K_t(x) = C \int_x^{\infty} N(r)\,A(r)\,\frac{r\,dr}{\sqrt{r^2 - x^2}} \tag{18}
$$

$$
K_r(x) = C \int_x^{\infty} N(r)\,\Big\{(1 - \tfrac{x^2}{r^2})\,A(r) + \tfrac{x^2}{r^2}\,B(r)\Big\}\,\frac{r\,dr}{\sqrt{r^2 - x^2}} \tag{19}
$$

A useful equation is obtained by subtraction:

$$
\boxed{\;K_t(x) - K_r(x) = C \int_x^{\infty} N(r)\,\big[A(r) - B(r)\big]\,\frac{x^2\,dr}{r\,\sqrt{r^2 - x^2}}\;} \tag{20}
$$

with the constant

$$
C = \tfrac{3}{4} \cdot 10^8\,R\sigma = 3.44 \times 10^{-6}\,\text{cm}^3 \tag{21}
$$

**This is the single most important equation of the paper.** The polarized brightness $pB \equiv K_t - K_r$ on the LHS is, by the assumption that the F-corona is unpolarized, observable directly without the troublesome F/K separation. The integral on the RHS contains $N(r)$ linearly with the kernel $[A(r) - B(r)]\,x^2/r$ (Abel-type, singular as $1/\sqrt{r^2 - x^2}$ at $r \to x$). Inverting this equation determines $N(r)$.

**Solution by successive approximation.** Equations (18) and (20) must be split between $K_t$ and $K_r$ so that both yield the same $N(r)$. Baumbach made an approximate two-step solution: (i) assume isotropy by replacing $A$ and $B$ by $\tfrac{1}{3}(2A+B)$ in (17), giving an electron density too low by 1–18% (factor $b$ in Table 5); (ii) compute the corresponding polarization. Schuster and Minnaert worked from analytical $N(r) \sim r^{-n}$ assumptions; van de Hulst rejects both as insufficient and adopts a successive-approximation method:

First, with a preliminary $p(x)$ guess, compute

$$
K_t = \tfrac{1}{2}(1 + p)\,K, \qquad K_t - K_r = p\,K \tag{22}
$$

and represent each as $\sum h_i x^{-i}$ and $\sum k_s x^{-s}$. Direct integration yields

$$
rCN(r)A(r) = \sum_s \frac{h_s}{a_{s-1}}\,r^{-s} \tag{24a}
$$

$$
rCN(r)\{A(r) - B(r)\} = \sum_s \frac{k_s}{a_{s+1}}\,r^{-s} \tag{24b}
$$

where the constants $a_n$ are the Wallis-type integrals

$$
a_n = \int_0^{\pi/2} \sin^n\varphi\,d\varphi = \frac{\pi}{2^{n+1}} \cdot \frac{n!}{[(n/2)!]^2} \tag{25}
$$

Sample values from Table 4: $a_0 = 1.5708$, $a_1 = 1.0000$, $a_2 = 0.7854$, $a_7 = 0.4571$, $a_{17} = 0.3085$, $a_{35} = 0.2104$. Defining the auxiliary "effective" functions

$$
c_t = \frac{K_t(r)}{rCN(r)A(r)}, \qquad c_v = \frac{K_t(r) - K_r(r)}{rCN(r)\{A(r) - B(r)\}} \tag{26a, 26b}
$$

leads to the relation

$$
\frac{1}{p} + 1 = \frac{2A}{A-B} \cdot \frac{c_t}{c_v} \tag{27}
$$

A short cut (Eqs. 28a, b) corrects the preliminary solution by a small factor $(1+\varepsilon)$, $\varepsilon < 0.05$, achieving $|\varepsilon| < 0.02$ everywhere when $K_t(x)$ has the form $-17.5\,x^{-34} + 132.0\,x^{-17} - 22.5\,x^{-9.7} + 85.2\,x^{-7} + 0.31\,x^{-2.5}$, with the simpler $K_t(x) = 89\,x^{-7} + 0.31\,x^{-2.5}$ sufficing for $r > 4$.

**Numerical results — Table 5.** Table 5A (equatorial, minimum phase) gives $K_t + K_r$, $K_t$, $K_t - K_r$, polarization $p$, $c_t$, $c_v$, $rCN$, $N$ (min and max), and the correction factor $b$. **Sample values from Table 5A**: at $r = 1.03$, $N = 178 \times 10^6$ cm$^{-3}$ (min) and $316 \times 10^6$ cm$^{-3}$ (max); at $r = 1.5$, $N = 8.30 \times 10^6$ (min) and $14.8 \times 10^6$ (max); at $r = 2.0$, $N = 1.58 \times 10^6$ (min) and $2.81 \times 10^6$ (max); at $r = 4.0$, $N = 0.050 \times 10^6$ (min) and $0.090 \times 10^6$ (max). The polarization $p$ rises from 0.181 at $r=1$ to peak ~0.66 near $r = 3$, then falls. The correction factor $b$ (multiplicative correction to Baumbach's isotropy approximation) ranges from 1.02 at $r=1$ to 1.15 at $r = 2.6$, confirming that Baumbach's neglect of anisotropy underestimated $N$ by up to ~18% in the inner corona, ~15% near $r = 2.6$.

Two general remarks:
1. **Polarization is higher in the outer corona.** For very large $r$, the $r^{-7}$ term dominates ($s = 7$): $p = s/(s+2) = 7/9 = 0.78$, and $b = \tfrac{2}{3}(p+1) = 1.18$.
2. **Polarization close to the limb is a little lower** because volume elements along a given line of sight that scatter most strongly polarized radiation lie not in the plane of projection but slightly behind and in front. For a line $x = 1$, an element at $y = 0$, $r = 1$ gives $p = 0.12$, while elements at $y = \pm 0.7$, $r = 1.2$ give the maximum polarization $p = 0.26$. These maxima have less influence the faster the density decreases outward.

#### 한국어

**진동 타원체 형식.** $R = 6.97\times 10^{10}$ cm = 태양 반경, $H$ = 태양 평균 표면밝기, $q$ = limb darkening 계수. 태양 반경 $rR$ 거리의 코로나 점 $P$에서, 태양이 차지하는 입체각 전체에 적분된 **조명 밀도**(density of illumination)는 $\pi H \{2A(r) + B(r)\}$ 형태이며, $2A$는 시선 반경에 대한 **접선** 방향(2축), $B$는 **시선 방향**(1축) 전기장 성분의 평균 제곱에 비례하는 분수이다. 점광원이라면 $B=0$이므로 $B$는 $r$이 커지면 빠르게 감소하고, $A \to 1/(2r^2)$로 점근한다. Schuster-Minnaert의 적분 결과가 식 (11), (12)이며 $\sin\gamma = 1/r$이다. Table 3은 $q=0, 1, 0.75$에 대한 $A, B$ 값을 제공한다. $q = 0.75$ (가시광 4700 Å에 해당하는 표준값)에서: $r=1.0$: $A=0.5972$, $A-B = -0.1250$; $r=1.5$: $A=0.2243$, $A-B = 0.1739$; $r=4$: $A=0.0312$, $A-B = 0.0304$.

**산란 적분.** 전자 수밀도를 $N(r)$, 톰슨 단면적을 $\sigma = 0.66\times 10^{-24}$ cm$^2$로 두면, 단위 체적당 단위 시간당 모든 방향으로 산란되는 빛은 식 (13)이다. 시선과 반경 사이 각 $\theta$를 갖는 특정 방향에 대해, 태양 중심을 지나는 평면에 **수직** 진동 빛($J_t$)과 평면 **내** 진동 빛($J_r$)의 source function은 식 (14), (15)이다.

시선을 따른 적분식 (16)에서 변수 변환 $dy = r\,dr/\sqrt{r^2-x^2}$를 적용하면 본 논문의 **핵심 적분 방정식** (17)을 얻는다 (영문 박스 참조). 편광 성분으로 분해한 식 (18), (19), 그리고 차감으로 얻는 **편광 밝기 식** (20) (영문 두 번째 박스)이 따라온다. 상수 $C = \tfrac{3}{4}\cdot 10^8\,R\sigma = 3.44\times 10^{-6}$ cm$^3$ (식 21).

**식 (20)이 본 논문의 가장 중요한 식이다.** F-corona가 비편광이라는 가정 하에 좌변 $pB \equiv K_t - K_r$은 F/K 분리 없이 직접 관측 가능하며, 우변에 $N(r)$이 선형으로 들어가고 핵 $[A(r)-B(r)]\,x^2/r$은 $r\to x$에서 $1/\sqrt{r^2-x^2}$ 형태로 발산하는 Abel형이다. 이 식을 풀면 $N(r)$이 결정된다.

**연속 근사 풀이.** 식 (18)과 (20)을 $K_t$와 $K_r$로 분할할 때 동일한 $N(r)$을 줘야 한다. Baumbach는 (i) 등방성 가정 $A=B=\tfrac{1}{3}(2A+B)$로 1차 풀이, 이는 1–18% 낮은 $N$을 주며 (Table 5의 인자 $b$로 표시); (ii) 그 $N$으로 편광을 다시 계산하는 2단계 근사를 사용했다. Schuster-Minnaert는 $N(r)\sim r^{-n}$ 해석적 가정으로 작업했다. van de Hulst는 둘 다 충분치 않다고 판단하고 연속 근사 방법을 채택한다.

먼저 $p(x)$ 추정으로 $K_t$, $K_t-K_r$을 식 (22)로 표현하고, 각각을 $\sum h_i x^{-i}$, $\sum k_s x^{-s}$로 적합. 직접 적분으로 식 (24a), (24b)를 얻으며, $a_n$은 식 (25)의 Wallis형 적분이다. Table 4: $a_0=1.5708$, $a_1=1$, $a_2=0.7854$, $a_7=0.4571$, $a_{17}=0.3085$, $a_{35}=0.2104$. 보조 함수 $c_t$, $c_v$ (식 26a, 26b)를 정의하면 식 (27)의 관계가 성립한다. 단축법(식 28a, b)은 $(1+\varepsilon)$, $\varepsilon<0.05$의 작은 인자로 예비 해를 보정해, $K_t(x) = -17.5x^{-34} + 132.0x^{-17} - 22.5x^{-9.7} + 85.2x^{-7} + 0.31x^{-2.5}$ 형태에서 $|\varepsilon|<0.02$ 정확도 달성.

**Table 5 수치 결과.** Table 5A(적도, 최소기) 표본값: $r=1.03$에서 $N = 178\times 10^6$ cm$^{-3}$ (min) / $316\times 10^6$ (max); $r=1.5$에서 $8.30\times 10^6$ / $14.8\times 10^6$; $r=2.0$에서 $1.58\times 10^6$ / $2.81\times 10^6$; $r=4.0$에서 $0.050\times 10^6$ / $0.090\times 10^6$. 편광도 $p$는 $r=1$에서 0.181, $r\sim 3$에서 ~0.66 정점, 그 후 감소. 보정 인자 $b$는 $r=1$에서 1.02, $r=2.6$에서 1.15까지 증가하며, Baumbach의 등방 근사가 내부 코로나에서 $N$을 최대 18% 과소평가했음을 확인한다.

두 가지 일반적 결과:
1. **외부 코로나에서 편광이 더 높다.** $r$이 매우 크면 $r^{-7}$ 항이 지배적($s=7$)이고 $p = s/(s+2) = 7/9 = 0.78$, $b = \tfrac{2}{3}(p+1) = 1.18$.
2. **태양 가장자리 근처에서 편광이 약간 낮다.** 시선 위에서 가장 강하게 편광된 빛을 산란하는 체적요소가 투영면에 있지 않고 약간 앞뒤에 있기 때문. $x=1$ 시선에서 $y=0, r=1$ 요소는 $p=0.12$이지만 $y=\pm 0.7, r=1.2$ 요소는 최대 $p=0.26$. 밀도가 빨리 감소할수록 이 최대값의 영향은 줄어든다.

---

### Part V — §5 Polar Density vs. Heliographic Latitude / 위도에 따른 극 밀도 (pp. 145–147)

#### English
The methods of §4 use spherical symmetry and apply only to the equatorial plane. For the polar regions, where the line of sight passes through different latitudes, a different reduction is needed. The starting input is **micro-photometric tracings on the Floyd plate of the 1900 eclipse**, taken perpendicular to the polar axis at $z = 1.04, 1.08, 1.12$ both north and south, calibrated by tracings in the equatorial regions. Surprisingly, "there is not a big jump where one passes into the strong streamer but only a slight elevation, or just a less steep part in the general decrease of intensity outward." This is consistent with the photometric difference between streamers and inter-streamer background being small — the visual prominence of streamers exaggerates contrast.

The polar inversion proceeds in two steps. **First step — isotropic scattering assumption.** The surface brightness $K$ (with F removed) and source function $J$ are related by

$$
K(x) = 2 \int_x^{\infty} J(v)\,\frac{v\,dv}{\sqrt{v^2 - x^2}} \qquad [\text{units: } k\cdot x^{-s}] \tag{29}
$$

The well-known Abel inversion of this integral equation is

$$
P(x) = -\frac{1}{x}\frac{dK(x)}{dx},\qquad J(v) = \frac{1}{\pi}\int_v^{\infty} P(x)\,\frac{x\,dx}{\sqrt{x^2 - v^2}} \tag{30}
$$

For data representable by a single power of $r$, the coefficient relations are $k = 2j\,a_{s-1}$, $p = s\,k$, $j = p\,a_s/\pi$ (Eq. 31), with the consistency identity $a_n \cdot a_{n-1} = \pi/(2n)$ (Eq. 32). The integral was solved by rough graphical evaluation of (30) for inner parts and by power-series fits for the outer parts. Table 6 gives sample $K$ and $J$ values for the plane $z = 1.08$ at NE and NW orientations: at $x = 0$ (closest approach, $r = 1.08$, $\beta = 90°$), $K = 59$, $J = 125$ (NE) and $94$ (NW); at $x = 0.4$ ($r = 1.15$, $\beta = 70°$), $K = 16$, $J = 18$ (NE) but only $7$ (NW).

**Second step — anisotropy correction.** Comparing equations (13), (16), (29), the source function and electron density are related by

$$
J = 10^8\,R\,\frac{\sigma}{4}\,(2A+B)\,N' \tag{33}
$$

so that

$$
N' = 8.75 \cdot 10^5\,J\,/\,(2A+B) \tag{34}
$$

To restore anisotropy, the integrand of (29) acquires a factor $m_t + m_r$, where

$$
m_t = \tfrac{3}{2}\,\frac{A}{2A+B}, \qquad m_r = \tfrac{3}{2}\,\frac{A\cos^2\theta + B\sin^2\theta}{2A+B} \tag{35}
$$

Numerical integration shows that at $x = 0$, $z = 1.08$ the brightness is reduced to 0.92 (NE) or 0.95 (NW) of its isotropic value. Equivalently, the electron density $N'$ from isotropic scattering must be **multiplied by 1.08 (NE) or 1.05 (NW)** to restore the correct $K$. This corresponds well with the factor 1.08 found from Table 5B.

The integrals further show that the polarization $p$ for the assumed Eq. (30) symmetry would be 0.30 (NE) and 0.30 (NW) compared to 0.30 from Table 5B (spherical symmetry). Thus neither $b$ nor $p$ is sensitive to the breakdown of spherical symmetry.

The ultimate $N$ comes from

$$
N = 9.6 \cdot 10^5\,J \tag{36}
$$

(from Eq. (35) with substitution of $b = 1.07$, $2A+B = 0.97$). After calibration corrections (brightness at $\beta = 0°$ was 9% above the assumed value, at $\beta = 30°$ was 7% below), the final result is shown in **Figure 7A** (surface brightness $K$ at $r = 1.15$ vs. heliographic latitude $\beta$) and **Figure 7B** (electron density $N \times 10^{-7}$ at $r = 1.15$ vs. $\beta$).

**The principal new finding** is a clear "dip" in $K$ near $\beta = 70°$, becoming a pronounced minimum in $N$. Quantitatively: "The dip becomes very pronounced, going down to 10 or 20 per cent of the equatorial density and there is a peak at the pole of about half the equatorial density." Values of $N$ for the polar regions in Table 5B should be increased by ~30% near $r = 1.1$ to bring them into agreement; the correction at larger $r$ is probably smaller because spreading polar rays tend to establish spherical symmetry. The values of $p$ (and $b$) in Table 5B are virtually correct for all $r$.

**Sample values from Table 5B (polar regions in min phase):** $r = 1.03$: $K_t + K_r = 125.9$, $K_t - K_r = 30.0$, $p = 0.238$, $N = 127 \times 10^6$ cm$^{-3}$, $b = 1.04$; $r = 1.5$: $0.93$, $0.51$, $p = 0.555$, $N = 1.41 \times 10^6$; $r = 4.0$: $0.0008$, $0.0006$, $p = 0.748$, $N = 0.004 \times 10^6$.

#### 한국어
§4 방법은 구대칭에 의존하며 적도면에만 적용 가능하다. 극 영역은 시선이 여러 위도를 가로지르므로 다른 환산이 필요하다. 입력 자료는 **1900년 일식의 Floyd 사진건판에 대한 마이크로포토미터 트레이싱**으로, 극축에 수직 방향, $z=1.04, 1.08, 1.12$의 북·남 양쪽, 적도 영역 트레이싱으로 보정. 놀랍게도 "강한 streamer로 진입하는 곳에서 큰 점프가 없고 약간의 상승 또는 외향 감소율의 약간 완만함만 보인다." — streamer와 inter-streamer 배경의 측광 차이는 작고, 시각적 두드러짐이 대비를 과장한다는 것이다.

극 영역 인버전은 두 단계로 진행된다. **1단계 — 등방 산란 가정.** 표면밝기 $K$ (F 제거 후)와 source function $J$의 관계는 식 (29)이며, 이의 잘 알려진 Abel 역변환은 식 (30)이다. 단일 멱 표현 가능한 자료에 대한 계수 관계 $k = 2ja_{s-1}$, $p = sk$, $j = pa_s/\pi$ (식 31)와 항등식 $a_n a_{n-1} = \pi/(2n)$ (식 32). 적분은 내부에서는 그래픽 평가, 외부에서는 멱급수 fit. Table 6은 $z=1.08$ 평면에서 NE/NW 방향의 $K$, $J$ 표본값: $x=0$ ($r=1.08, \beta=90°$)에서 $K=59$, $J=125$ (NE) / $94$ (NW); $x=0.4$ ($r=1.15, \beta=70°$)에서 $K=16$, $J=18$ (NE) / $7$ (NW).

**2단계 — 비등방 보정.** 식 (13), (16), (29)를 비교하면 식 (33), (34)로 source function과 $N'$이 연결된다. 비등방성 회복을 위해 식 (29)의 피적분함수에 $m_t + m_r$ 인자를 곱해야 하며, 정의는 식 (35). 수치 적분 결과 $x=0$, $z=1.08$에서 밝기가 등방값의 0.92 (NE) / 0.95 (NW)로 감소; 등가적으로 $N'$에 **1.08 (NE) 또는 1.05 (NW)를 곱해야** 정확한 $K$ 회복. Table 5B에서 얻은 1.08 인자와 잘 일치한다. 편광도 $p$는 0.30 (양쪽 모두)로 Table 5B의 구대칭 결과 0.30과 같으므로, $b$와 $p$ 모두 구대칭 위반에 둔감함.

최종 $N$은 식 (36)으로 얻으며 ($b=1.07$, $2A+B=0.97$ 대입), 캘리브레이션 보정(평균적으로 $\beta=0°$의 밝기는 9% 위, $\beta=30°$의 밝기는 7% 아래)을 거쳐 **Figure 7A** ($r=1.15$ 표면밝기 $K$ vs. $\beta$)와 **Figure 7B** (전자 밀도 $N$ vs. $\beta$)에 표시.

**주요 새로운 발견**은 $K$가 $\beta\approx 70°$ 부근에서 명확한 "dip"을 보이며, $N$에서는 더욱 두드러진 최솟값을 만든다는 것이다. 정량적으로: "딥은 매우 뚜렷해져 적도 밀도의 10–20%까지 떨어지고, 극에는 적도 밀도의 약 절반 정도의 봉우리가 있다." Table 5B의 극 영역 $N$ 값은 $r = 1.1$ 근처에서 약 30% 증가시켜야 한다. $r$이 클수록 보정은 작아진다(극 광선이 퍼지면서 구대칭에 가까워지므로). $p$(및 $b$)는 모든 $r$에 대해 Table 5B 값이 사실상 정확하다.

**Table 5B 표본값 (최소기 극 영역):** $r=1.03$: $K_t+K_r=125.9$, $K_t-K_r=30.0$, $p=0.238$, $N=127\times 10^6$ cm$^{-3}$, $b=1.04$; $r=1.5$: $0.93, 0.51$, $p=0.555$, $N=1.41\times 10^6$; $r=4.0$: $0.0008, 0.0006$, $p=0.748$, $N=0.004\times 10^6$.

#### Astrophysical interpretation / 천체물리적 해석

#### English
"It is tempting to speculate about the relation between the phenomenon just found and other data on solar activity." After Ludendorff's classical work and Bergstrand's attempts to remove equatorial-streamer projection from the polar regions, the same problem persisted, but Bergstrand's "veil" is largely the F-corona and his formulae are too schematic. Lockyer noted high-latitude prominence parallelism with polar coronal forms; the maxima of these phenomena in the eleven-year cycle are shifted relative to the sunspot maxima similarly. **Latitude distribution of inner-corona electron density (the present finding) shows a fairly high value near the poles and a distinct minimum near $\beta = 70°$, which is "very puzzling"** and unlike that of any other solar phenomenon. Different eclipse photographs should be studied in the same manner.

#### 한국어
"방금 발견된 현상과 다른 태양 활동 자료 사이의 관계를 추측해보고 싶다." Ludendorff의 고전적 연구, Bergstrand의 적도 streamer 투영 제거 시도가 있었지만, Bergstrand의 "veil"은 대부분 F-corona이고 그의 공식은 너무 도식적이다. Lockyer는 고위도 prominence와 극 코로나 형태의 평행성을 지적했고, 11년 주기의 최대값은 sunspot 최대와 비슷하게 시프트되어 있다. **본 논문의 발견 — 내부 코로나 전자 밀도가 극 부근에서 비교적 높고 $\beta\approx 70°$에 명확한 최솟값을 가진다는 사실 — 은 "매우 수수께끼 같으며"** 다른 어떤 태양 현상과도 다르다. 다른 일식 사진들도 같은 방법으로 연구되어야 한다.

---

### Part VI — §6 Comparison with Observed Polarization / 관측 편광과의 비교 (pp. 147–149)

#### English
The model corona of §3–4 was constructed using observations of (a) total intensity and (b) surface brightness. A more stringent test is comparison with (c) observed polarization and (d) Fraunhofer-line depth. **Figure 8** plots observed polarization data (eclipses 1901, 1905, 1908, 1914, 1932, 1934, 1940, 1941, 1943, 1945) against the computed $p_K$ (dashed) and $p$ for combined F+K (full) curves. Datasets include Allen, Cohn, Öhman, Young, Fesenkoff, Hurahata; the photoelectric Hurahata measurements and photographic Öhman determinations get higher weights.

In broad lines the agreement is satisfactory. Both show $p$ rising to ~40% then dropping for very large $r$, with $p$ much lower near the pole. The hypothesis of Öhman and Allen — that the F-component is chiefly responsible for the deviation of observed points from the dashed curves — is well confirmed. Yet the **quantitative agreement is poor in the medium and outer corona ($1.5 < r < 3.0$).**

**Equator regions, inner corona ($r < 1.5$):** good agreement. This is mainly a check on the values of $p_K$ since F has little influence. The model $f$ ranges from 0.87 at $r=1.3$ to 0.70 at $r=1.5$. Measured $p$ via Eq. (3) gives values 1.00 to 0.67, with two points higher than 0.90, in agreement with Minnaert and Baumbach's calculations.

**Equator regions, medium and outer corona ($1.5 < r < 3.0$):** the theoretical curves seem altogether too low. The difference is so strong that the more subtle ~10% difference between $p$ for max vs. min phases seems unimportant. For $r = 2.5$: model $f$ = 0.40 (max), 0.28 (min); $f$ from Fraunhofer-line depth = 0.22–0.35 (Grotrian 1923 minimum), 0.41 (Allen 1940 intermediate); but $f$ from measured $p$ via Eq. (3) is 0.75 (Allen) and 0.51, 0.58 (interpolated from two other sets). The discrepancy between adopted $f$ and polarization-derived $f$ is too large to ignore.

**Three explanations are considered:**
- **(a) Sky-light over-correction.** The only photoelectric measurements (Hurahata) — those that did not need sky-light correction by direct measurement — do not deviate from theoretical curves. Sky light originates from sunlit air outside the eclipse cone, fairly constant over the corona. Hurahata's measurements give sky light = 60·10⁻⁸ Sun = 1/5 of F-corona at $r = 2.5$. If the sky light is polarized by ±50%, the estimated $p$ of combined light may range 23 to 9 per cent, vs. 18 per cent without sky light. The danger that observers consistently over-corrected for sky light (or plate fog) is real, and **this is the most plausible explanation**.
- **(b) Intrinsic F-corona polarization.** Using $p(F+K) = p_K K + p_F F$ with the data $F = 0.53$, $K = 0.65$ at $r = 2.5$, observed $p = 0.54$, computed $p_F$ must equal 0.50. A 50% polarization for a diffraction halo by interplanetary dust is regarded as **impossible**, so this explanation is rejected.
- **(c) F-corona overestimated.** If $F$ were lower and $f$ higher than in the model, $p$ would naturally be larger. To match observed $p$ requires solving Eq. (20) with measured $pB$ and integrating Eq. (18), giving $p_K$ and $K$ separately. Table 7 gives the resulting values: at $r = 2.5$, $K = 0.51$, $p_K = 0.53$, $f = 0.70$ vs. model $f = 0.28$. This implies the actual depth of Fraunhofer lines in the corona at $r = 2.5$ is much lower than measured. This explanation is troubled by a complication: increasing $K$ in the outer corona also makes $K$'s outward gradient shallower, so $p_K$ drops, requiring even more $K$ to match observed $p$.

The author concludes that **(a) sky-light over-correction is the most plausible** explanation, and that the adopted brightness for the F-corona may have to be reduced, especially for the polar regions.

#### 한국어
§3–4의 model corona는 관측의 (a) 총 강도와 (b) 표면밝기로 구성되었다. 더 엄격한 검증은 (c) 관측 편광 및 (d) Fraunhofer 선 깊이와의 비교이다. **Figure 8**은 관측 편광 자료(1901–1945년 9개 일식)와 계산된 $p_K$ (점선) 및 F+K 결합 $p$ (실선) 곡선을 함께 보여준다. 데이터셋: Allen, Cohn, Öhman, Young, Fesenkoff, Hurahata; 광전 Hurahata 측정과 사진 Öhman 측정에 더 높은 가중치.

대체적으로 만족스러운 일치. $p$가 ~40%까지 상승 후 매우 큰 $r$에서 감소, 극 근처에서 훨씬 낮은 $p$. Öhman·Allen의 가설 — 관측점이 점선 곡선에서 벗어나는 주된 원인이 F-성분 — 도 잘 확인된다. 그러나 **중·외부 코로나 ($1.5 < r < 3.0$)에서 정량적 일치는 좋지 않다.**

**적도 영역 내부 코로나 ($r<1.5$):** 좋은 일치. F의 영향이 작아 사실상 $p_K$ 값에 대한 검증. 모델 $f$는 $r=1.3$에서 0.87, $r=1.5$에서 0.70. 측정된 $p$로부터 식 (3)을 통한 $f$ 값은 1.00–0.67, Minnaert·Baumbach 계산과 일치.

**적도 영역 중·외부 코로나 ($1.5 < r < 3.0$):** 이론 곡선이 너무 낮아 보인다. 차이가 크기 때문에 max/min 위상 사이 ~10% 차이는 중요해 보이지 않는다. $r=2.5$에서: 모델 $f = 0.40$(max), 0.28(min); Fraunhofer 깊이로부터 $f = 0.22$–0.35 (Grotrian 1923 최소), 0.41 (Allen 1940 중간); 측정된 $p$로부터 식(3)을 통한 $f = 0.75$ (Allen), 0.51, 0.58. 채택값과 편광 유도값 사이 차이가 너무 크다.

**세 가지 설명:**
- **(a) Sky light 과대보정.** 광전 측정(Hurahata) — sky light를 직접 측정으로 제거 — 만이 이론 곡선에서 벗어나지 않는다. Sky light는 일식 그림자 바깥의 햇빛받는 대기에서 오며 코로나 전역에 거의 일정. Hurahata 측정으로부터 sky light = $60\cdot 10^{-8}$ 태양 = $r=2.5$에서 F-corona의 1/5. Sky light가 ±50% 편광되었다면 결합광 $p$는 23–9%, sky light 무시 시 18%. 관측자들이 sky light(또는 plate fog)를 일관되게 과대보정했을 위험이 실재하며, **이것이 가장 그럴듯한 설명**이다.
- **(b) F-corona 자체 편광.** $p(F+K)=p_K K + p_F F$에서 $r=2.5$의 $F=0.53$, $K=0.65$, 관측 $p=0.54$, 계산 $p_F = 0.50$. 행성간 먼지의 회절 헤일로에 50% 편광은 **불가능**으로 간주되어 기각.
- **(c) F-corona 과소평가.** $F$가 더 작고 $f$가 더 크다면 $p$가 자연히 커진다. 관측 $p$를 맞추려면 측정된 $pB$로 식 (20)을 풀고 식 (18)을 적분해 $p_K$, $K$를 분리해야 한다. Table 7: $r=2.5$에서 $K=0.51$, $p_K=0.53$, $f=0.70$ (모델 0.28). 이는 $r=2.5$에서 실제 Fraunhofer 선 깊이가 측정보다 훨씬 낮음을 의미. 복잡성: 외부 코로나의 $K$ 증가는 외향 기울기를 완만하게 만들어 $p_K$를 낮추므로, 관측 $p$를 맞추려면 $K$를 더 늘려야 한다.

저자는 **(a) sky light 과대보정이 가장 그럴듯한** 설명이라 결론짓고, F-corona의 채택 밝기, 특히 극 영역에서, 줄어들어야 할 가능성을 시사한다.

---

## 3. Key Takeaways / 핵심 시사점

1. **The pB inversion was born here.** / **pB 인버전은 여기서 태어났다.**
   *English:* Equation (20) for $K_t - K_r$ as a line-of-sight integral of $N(r)\,[A(r)-B(r)]\,x^2/(r\sqrt{r^2-x^2})$ is the **direct ancestor of every modern coronal pB inversion** (LASCO, COR1/2, K-Cor, Metis, PUNCH). The polarized brightness $pB \equiv K_t - K_r$ is preferred because it is observable without F/K separation.
   *한국어:* 식 (20) — $K_t - K_r$을 $N(r)[A(r)-B(r)]\,x^2/(r\sqrt{r^2-x^2})$의 시선 적분으로 표현 — 은 **현대 모든 코로나 pB 인버전(LASCO, COR1/2, K-Cor, Metis, PUNCH)의 직접적 조상**이다. 편광 밝기 $pB \equiv K_t - K_r$은 F/K 분리 없이 관측 가능하기에 선호된다.

2. **F-corona dominates the outer corona, not the inner.** / **F-corona는 외부 코로나에서 지배적이며 내부에서는 미약하다.**
   *English:* The fraction $f = K/(K+F)$ falls from 0.96 at $r = 1$ to 0.14 at $r = 5$ (max phase). $f = 0.50$ at $r = 2.24$ (max), 1.93 (min), 1.28 (pole). Beyond $r = 2$, F-corona contamination becomes the dominant source of error in $N_e$ derivations, motivating the use of $pB$ rather than total brightness.
   *한국어:* 비율 $f = K/(K+F)$는 $r=1$에서 0.96, $r=5$에서 0.14 (최대기). $f=0.50$ 반경은 max 2.24, min 1.93, 극 1.28. $r>2$에서 F-corona 오염이 $N_e$ 도출의 지배적 오차원이 되며, 이것이 총 밝기 대신 $pB$ 사용을 정당화한다.

3. **Baumbach's isotropic approximation underestimates $N_e$ by up to 18%.** / **Baumbach의 등방 근사는 $N_e$를 최대 18% 과소평가한다.**
   *English:* Replacing $A(r)$ and $B(r)$ by $\tfrac{1}{3}(2A+B)$ in Eq. (17) is convenient but introduces errors of 1–18% in derived $N(r)$ (factor $b$ in Table 5). The error peaks near $r \approx 2.6$ at ~15% in the equatorial corona and at ~18% in the polar regions for very large $r$. van de Hulst's successive-approximation method (Eqs. 22–28) achieves 1% accuracy.
   *한국어:* 식 (17)에서 $A, B$를 $\tfrac{1}{3}(2A+B)$로 치환하는 것은 편리하지만 도출된 $N(r)$에 1–18% 오차를 만든다 (Table 5의 $b$). 적도에서는 $r\approx 2.6$ 근처에서 ~15%, 극에서는 매우 큰 $r$에서 ~18% 최대 오차. van de Hulst의 연속 근사 방법 (식 22–28)은 1% 정확도를 달성한다.

4. **The model corona is self-consistent to 1%.** / **Model corona는 1% 정밀도로 자기무모순적이다.**
   *English:* The "model corona" is not a fit to observation, but a set of brightness/polarization/density tables (Tables 2, 5A, 5B) whose internal consistency is enforced to better than 1%. Sample inner-corona equatorial densities: $N(1.03) = 178\times 10^6$ cm⁻³ (min), $316\times 10^6$ (max); $N(2.0) = 1.58\times 10^6$ (min). Polar values are typically 50–60% of equatorial.
   *한국어:* "Model corona"는 관측 fit이 아니라, 밝기·편광·밀도 표 (Tables 2, 5A, 5B)의 내부 일관성을 1% 이내로 강제한 결과. 적도 내부 코로나 표본 밀도: $N(1.03) = 178\times 10^6$ cm⁻³ (min), $316\times 10^6$ (max); $N(2.0) = 1.58\times 10^6$ (min). 극 값은 통상 적도의 50–60%.

5. **The K-corona varies by a factor 1.78 over the solar cycle.** / **K-corona는 태양주기에 걸쳐 1.78배 변화한다.**
   *English:* From the cycle-phase analysis of total ring brightness (Figure 1), $K_{\max}/K_{\min} = c = 1.78$. The polar regions in minimum phase contribute only ~70% as much as equatorial regions per unit area in $r^{-17}$ component. F-corona shows no systematic cycle change.
   *한국어:* 총 링 밝기의 주기-위상 분석(Figure 1)에서 $K_{\max}/K_{\min} = c = 1.78$. 최소기 극 영역은 단위 면적당 적도 영역의 약 70%만 기여 ($r^{-17}$ 항 기준). F-corona는 체계적 주기 변화 없음.

6. **Discovery of the $\beta \approx 70°$ electron-density minimum.** / **$\beta\approx 70°$ 전자 밀도 최솟값의 발견.**
   *English:* Figure 7B shows that $N_e$ at $r = 1.15$ has a sharp minimum at heliographic latitude ~70° — only 10–20% of the equatorial value — with a peak at the pole at ~50% of equatorial. This is taken as evidence for **two distinct agents** producing the minimum-phase corona: an equatorial streamer-belt agent (≤ 60°) and a polar agent (≥ 70°). Modern view: the equatorial streamer belt and polar coronal holes are separated by a transition zone first photometrically detected here.
   *한국어:* Figure 7B는 $r=1.15$의 $N_e$가 헬리오그래픽 위도 ~70°에서 적도값의 10–20%까지 떨어지는 날카로운 최솟값을 가지며 극에는 적도값의 ~50% 봉우리를 가짐을 보여준다. 이는 최소기 코로나가 **두 개의 서로 다른 작용** — 적도 streamer belt 작용 (≤60°)과 극 작용 (≥70°) — 으로 만들어진다는 증거. 현대적 관점: 적도 streamer belt와 polar coronal hole 사이의 전이대가 여기서 처음으로 측광적으로 검출.

7. **F-corona properties remain the dominant systematic uncertainty.** / **F-corona 물성이 지배적 체계 오차로 남는다.**
   *English:* The model–observation polarization disagreement at $1.5 < r < 3.0$ is too large to ignore. After ruling out intrinsic F polarization (50% would be physically impossible for a dust diffraction halo), van de Hulst attributes the mismatch to **sky-light over-correction by photographic observers** — pointing forward to the eventual need for space-based coronagraphy.
   *한국어:* $1.5 < r < 3.0$에서의 모델·관측 편광 불일치는 무시할 수 없을 만큼 크다. F의 자체 편광 가능성(먼지 회절 헤일로에 50%는 물리적으로 불가능) 기각 후, van de Hulst는 불일치를 **사진 관측자의 sky light 과대보정**에 돌리며, 이는 결국 우주 기반 coronagraph의 필요성을 예견한다.

8. **A unified observable–theoretical framework.** / **관측·이론의 통합 형식화.**
   *English:* Before this paper, brightness measurements (Baumbach), polarization theory (Schuster, Minnaert), and density inversions were treated piecemeal. van de Hulst's contribution is to insist on a **single self-consistent framework** in which all three observables and derived $N_e(r)$ are mutually compatible — establishing the methodological standard followed by every subsequent coronal photometric analysis.
   *한국어:* 본 논문 이전에는 밝기 측정(Baumbach), 편광 이론(Schuster·Minnaert), 밀도 역변환이 개별적으로 다뤄졌다. van de Hulst의 기여는 세 관측량과 도출된 $N_e(r)$이 서로 호환되는 **단일 자기무모순 형식**을 고집한 것이며, 이후 모든 코로나 측광 분석의 방법론적 표준을 확립했다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 F/K Separation / F·K 분리

| Eq. | Formula | Purpose |
|---|---|---|
| (1) | $f = K/(F+K)$ | The F-removal factor |
| (2) | $1 - f = (1 - r_{\text{cor}})/(1 - r_{\text{disk}})$ | $f$ from Fraunhofer-line residual |
| (3) | $f = p/p_K$ | $f$ from polarimetry (assumes F unpolarized) |
| (—) | $f = 1 - F/(F+K)$ | $f$ by direct subtraction |

### 4.2 Brightness Power-Series Representations / 밝기 멱급수 표현
For corona at $r$ in units of solar radii, brightness in units of $10^{-8} \times$ Sun's average:

| Eq. | Formula |
|---|---|
| (5) | $K_{\max}(r) = 355.6\,r^{-17} + 177.8\,r^{-7} + 0.708\,r^{-2.5}$ |
| (6) | $K_{\min}(r) = 200.0\,r^{-17} + 100.0\,r^{-7} + 0.398\,r^{-2.5}$ |
| (7) | $F(r) = \quad\quad\quad\quad\quad\quad\quad\;\, 14.86\,r^{-7} + 4.99\,r^{-2.5}$ |
| (8) | $K_{\text{pole}}(r) + F(r) = 191.0\,r^{-17} + 27.45\,r^{-7} + 4.99\,r^{-2.5}$ |
| (9) | $K_{\text{pole}}(r) = 191.0\,r^{-17} + 12.59\,r^{-7}$ |

The three terms physically represent: $r^{-17}$ — innermost steep decline (Thomson by densest electrons at the limb); $r^{-7}$ — main K-corona profile; $r^{-2.5}$ — F-corona (zodiacal-light) tail.

### 4.3 Schuster-Minnaert Illumination Functions / 조명 함수
With $\sin\gamma = 1/r$ and limb-darkening $q = 0.75$:

$$
2A(r) + B(r) = \frac{1-q}{1 - q/3}\,\big\{2(1 - \cos\gamma)\big\} + \frac{q}{1 - q/3}\,\bigg\{1 - \frac{\cos^2\gamma}{\sin\gamma}\log\frac{1 + \sin\gamma}{\cos\gamma}\bigg\}
$$

$$
2A(r) - B(r) = \frac{1-q}{1 - q/3}\,\big\{\tfrac{2}{3}(1 - \cos^3\gamma)\big\} + \frac{q}{1 - q/3}\,\bigg\{\tfrac{1}{4} + \tfrac{\sin^2\gamma}{4} - \tfrac{\cos^4\gamma}{4\sin\gamma}\log\frac{1+\sin\gamma}{\cos\gamma}\bigg\}
$$

Asymptotic: $A(r) \to 1/(2r^2)$ as $r \to \infty$, $B \to 0$ rapidly. At $r=1$ (limb), $A - B = -0.125$ (i.e. radial-axis dominates illumination), crossing zero near $r \approx 1.04$ and rising to $A - B \approx 0.20$ for large $r$ (transverse axes dominate).

### 4.4 Central Integral Equations (THE CORE) / 핵심 적분 방정식

$$
\boxed{\;K(x) = C \int_x^{\infty} N(r)\,\Big\{(2 - \tfrac{x^2}{r^2})\,A(r) + \tfrac{x^2}{r^2}\,B(r)\Big\}\,\frac{r\,dr}{\sqrt{r^2 - x^2}}\;} \tag{17}
$$

$$
K_t(x) = C \int_x^{\infty} N(r)\,A(r)\,\frac{r\,dr}{\sqrt{r^2 - x^2}} \tag{18}
$$

$$
K_r(x) = C \int_x^{\infty} N(r)\,\Big\{(1 - \tfrac{x^2}{r^2})\,A(r) + \tfrac{x^2}{r^2}\,B(r)\Big\}\,\frac{r\,dr}{\sqrt{r^2 - x^2}} \tag{19}
$$

$$
\boxed{\;K_t(x) - K_r(x) \;=\; C \int_x^{\infty} N(r)\,\big[A(r) - B(r)\big]\,\frac{x^2\,dr}{r\,\sqrt{r^2 - x^2}}\;} \tag{20}
$$

with

$$
C = \tfrac{3}{4} \cdot 10^8\,R\sigma = 3.44 \times 10^{-6}\,\text{cm}^3 \tag{21}
$$

**Variable glossary / 변수 해설:**
| Symbol | English | 한국어 |
|---|---|---|
| $x$ | projected radius (impact parameter) in solar radii | 투영 반경 (충돌 매개변수), 태양 반경 단위 |
| $r$ | 3-D radial distance, $r = \sqrt{x^2 + y^2}$ | 3차원 방사 거리 |
| $y$ | line-of-sight coordinate in solar radii | 시선 좌표, 태양 반경 단위 |
| $N(r)$ | electron number density (cm⁻³) | 전자 수밀도 |
| $A(r)$ | transverse component of illumination (dimensionless) | 조명의 접선 성분 |
| $B(r)$ | radial component of illumination (dimensionless) | 조명의 시선 성분 |
| $K(x)$ | total surface brightness, $K = K_t + K_r$, in $10^{-8}$ of Sun | 총 표면밝기 |
| $K_t(x)$ | tangential polarization component | 접선 편광 성분 |
| $K_r(x)$ | radial polarization component | 시선 편광 성분 |
| $K_t - K_r$ | polarized brightness $pB$ | 편광 밝기 |
| $C$ | $\tfrac{3}{4}\cdot 10^8 R\sigma = 3.44\times 10^{-6}$ cm³ | 상수 |
| $R$ | solar radius, $6.97 \times 10^{10}$ cm | 태양 반경 |
| $\sigma$ | Thomson cross-section, $0.66 \times 10^{-24}$ cm² | 톰슨 단면적 |
| $q$ | limb-darkening coefficient, 0.75 at 4700 Å | limb darkening 계수 |

### 4.5 Successive Approximation for $N(r)$ / $N(r)$ 연속 근사

$$
K_t = \tfrac{1}{2}(1+p)K, \qquad K_t - K_r = pK \tag{22}
$$

$$
rCN(r)A(r) = \sum_s \frac{h_s}{a_{s-1}}\,r^{-s}, \qquad rCN(r)\{A(r) - B(r)\} = \sum_s \frac{k_s}{a_{s+1}}\,r^{-s} \tag{24a, b}
$$

$$
a_n = \int_0^{\pi/2} \sin^n\varphi\,d\varphi = \frac{\pi}{2^{n+1}} \cdot \frac{n!}{[(n/2)!]^2}, \qquad a_n a_{n-1} = \frac{\pi}{2n} \tag{25, 32}
$$

$$
\frac{1}{p} + 1 = \frac{2A}{A-B} \cdot \frac{c_t}{c_v} \tag{27}
$$

$$
K_t = (1 + \varepsilon p)\,K_t', \qquad K_t - K_r = \big\{1 + \varepsilon(1+p)\big\}(K_t' - K_r') \tag{28a, b}
$$

### 4.6 Polar Region Inversion / 극 영역 인버전

Isotropic-scattering Abel pair:

$$
K(x) = 2 \int_x^{\infty} J(v)\,\frac{v\,dv}{\sqrt{v^2 - x^2}} \tag{29}
$$

$$
P(x) = -\frac{1}{x}\frac{dK(x)}{dx}, \qquad J(v) = \frac{1}{\pi}\int_v^{\infty} P(x)\,\frac{x\,dx}{\sqrt{x^2 - v^2}} \tag{30}
$$

Power-law coefficients: $k = 2j a_{s-1}$, $p = sk$, $j = p a_s/\pi$ (Eq. 31).

Anisotropy correction:

$$
J = 10^8\,R\,\frac{\sigma}{4}(2A + B)\,N', \quad N' = 8.75\cdot 10^5\,\frac{J}{2A + B} \tag{33, 34}
$$

$$
m_t = \tfrac{3}{2}\frac{A}{2A+B}, \quad m_r = \tfrac{3}{2}\frac{A\cos^2\theta + B\sin^2\theta}{2A+B} \tag{35}
$$

$$
N = 9.6\cdot 10^5\,J \tag{36}
$$

### 4.7 Worked Numerical Example / 풀이 예제

Take **$r = 1.5$ in the equatorial plane, minimum phase** as a worked example. From Eq. (6), $K_{\min}(1.5) = 200.0 \cdot 1.5^{-17} + 100.0 \cdot 1.5^{-7} + 0.398 \cdot 1.5^{-2.5}$. Computing each term:
- $1.5^{-17} = 1/(1.5)^{17} \approx 1.52 \times 10^{-3}$, so $200 \cdot 1.52\times 10^{-3} = 0.304$
- $1.5^{-7} = 1/(1.5)^7 \approx 0.0585$, so $100 \cdot 0.0585 = 5.85$
- $1.5^{-2.5} \approx 0.363$, so $0.398 \cdot 0.363 = 0.144$
- Sum: $K_{\min}(1.5) \approx 6.30$ (in $10^{-8}$ Sun; Table 2 gives 6.20 — agreement to ~1.5%)

From Table 5A at $r = 1.5$: $K_t + K_r = 6.20$, $K_t = 4.77$, $K_t - K_r = 3.34$, giving $p = 3.34/6.20 = 0.538$ and $K_r = 1.43$.
The electron density: $N = 8.30 \times 10^6$ cm⁻³ (min). From Table 3 at $q = 0.75$: $A(1.5) = 0.2243$, $A - B = 0.1739$. Verifying via Eq. (20):
$$
K_t - K_r = 3.34 = C \int_{1.5}^{\infty} N(r)[A(r) - B(r)]\,\frac{1.5^2\,dr}{r\sqrt{r^2 - 1.5^2}}
$$
which is the integral form van de Hulst inverts to obtain $N(r)$ self-consistently. The Baumbach correction factor $b(1.5) = 1.13$ — meaning Baumbach's isotropic approximation would have given $N \approx 7.34 \times 10^6$ cm⁻³, ~13% too low.

For the **maximum phase**, multiply by $c = 1.78$: $N_{\max}(1.5) = 14.8 \times 10^6$ cm⁻³ — exactly matching Table 5A.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1879  Schuster        ── Theoretical polarization of K-corona by free electrons
1919  Bergstrand      ── Brightness distribution at 1914 eclipse (equator + polar)
1930  Minnaert        ── Comprehensive polarization theory; A(r), B(r) tables
1934  Grotrian        ── F/K separation via Fraunhofer-line dilution
1937  Baumbach        ── Compilation of N_e(r); coronagraph era (Lyot 1930)
1939  Baumbach        ── Iterative correction for anisotropy (still ~18% off)
1943  Hase / Vashakidze ── New polarization measurements
1947  Allen / van de Hulst ── Improved photometric values (Allen, M.N. 107)
1948  van de Hulst @ Lick ── Micro-photometry of 1893–1932 eclipse plates
═══════════════════════════════════════════════════════════════════════════════
1950  ★ van de Hulst (this paper) ── Self-consistent "model corona" to 1%;
                                     pole vs. equator separation;
                                     pB-inversion equations (17)–(20);
                                     β ≈ 70° N_e minimum discovered
═══════════════════════════════════════════════════════════════════════════════
1957  van de Hulst    ── "Light Scattering by Small Particles" (textbook)
1971  Saito           ── Refinement of equatorial/polar density model
1995  SOHO/LASCO      ── Spaceborne pB observations with C1, C2, C3 begin
2006  STEREO/SECCHI   ── COR1, COR2 — twin-spacecraft 3D pB inversion
2020  Solar Orbiter   ── Metis coronagraph (pB + UV Lyα simultaneous)
2025  Abbo+ (#61)     ── Metis pB inversion using van de Hulst Eq. (20) form
2025  PUNCH (NASA)    ── Wide-field heliospheric pB imaging
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **#61 Abbo et al. 2025 (A&A 702, A254)** | **Direct descendant.** Abbo+25's Eq. (1) is the modern notation of van de Hulst's Eq. (20). Metis pB observations are inverted using the same Abel-type kernel; the limb-darkening parameter $u$ in Abbo+25 corresponds to van de Hulst's $q$. / **직계 후손.** Abbo+25의 식 (1)은 van de Hulst 식 (20)의 현대 표기. Metis pB 관측을 동일한 Abel형 핵으로 인버전하며, $u$ ↔ $q$. | **★★★ Critical** — this paper IS the foundation Abbo+25 builds on |
| **Schuster 1879 (M.N. 40, 35)** | Provides the original Rayleigh-scattering polarization theory for free electrons that van de Hulst extends into the integral-equation framework of §4. The functions $A(r)$, $B(r)$ are originally Schuster's. / 자유전자의 Rayleigh 산란 편광 이론 원조; $A(r), B(r)$ 함수 자체가 Schuster의 것. | ★★ High — theoretical foundation |
| **Minnaert 1930 (Z. f. Ap. 1, 209)** | Derived the explicit closed-form expressions for $2A+B$ and $2A-B$ as functions of $\gamma$ and $q$ — Eqs. (11), (12) of this paper. van de Hulst's tabulated values in Table 3 are computed from Minnaert's formulae. / $2A+B$, $2A-B$의 닫힌 형식 표현 (식 11, 12)을 유도; Table 3은 Minnaert의 식으로 계산. | ★★ High — direct equation source |
| **Baumbach 1937 (Astr. Nachr. 263, 121)** | The compilation against which van de Hulst's "model corona" is benchmarked. Baumbach's $r^{-17} + r^{-7} + r^{-2.5}$ power-law fit (Eq. 4) is the starting form for §3's improved coefficients. Baumbach's isotropic approximation is shown to be 1–18% inaccurate. / van de Hulst가 기준으로 삼은 자료 모음집; Baumbach의 멱급수 형태가 §3의 출발점. Baumbach의 등방 근사는 1–18% 부정확함을 입증. | ★★ High — predecessor reference |
| **Baumbach 1939 (Astr. Nachr. 267, 273)** | Two-step iterative correction for anisotropy; van de Hulst replaces this with the more accurate successive-approximation method of §4. / 비등방성에 대한 두 단계 반복 보정; van de Hulst는 §4의 더 정확한 연속 근사법으로 대체. | ★ Moderate |
| **Bergstrand 1919 (Couronne solaire)** | Provides the brightness distribution for the 1914 eclipse along both equator and polar axis — primary input for Eqs. (8), (9) and Figure 3. / 1914 일식의 적도·극축 밝기 분포 자료 — 식 (8), (9), Figure 3의 1차 입력. | ★ Moderate — observational input |
| **Saito 1971 (Ann. Tokyo Astron. Obs.)** | Refines van de Hulst's equatorial and polar models into widely used analytical fits (the "Saito 1970" or "Saito 1977" form), still cited as the default low-activity coronal density profile. / van de Hulst의 적도·극 모델을 정밀화한 분석적 fit ("Saito 1970/1977"); 저활동기 코로나 밀도 프로파일의 기본값으로 여전히 인용. | ★★ High — direct refinement |
| **Allen 1947 (M.N. 107, 426)** | Allen's photometric measurements provided the absolute calibration cross-check that fixed van de Hulst's factor $a = 104$ in §3. / Allen의 측광이 §3의 절대 보정 인자 $a=104$ 결정의 교차 검증 자료. | ★ Moderate |
| **Lyot 1930 (coronagraph invention)** | Mentioned in §1; coronagraph era enables non-eclipse coronal observation, but did not yet provide reliable absolute photometry by 1950. The integration of Lyot-style coronagraphy with van de Hulst's inversion came later (Newkirk 1965, K-coronameters, LASCO). / §1에 언급; coronagraph 시대를 열었으나 1950년까지 신뢰할 만한 절대 측광 미제공. Lyot식 coronagraph + van de Hulst 인버전 결합은 후대(Newkirk 1965, K-coronameter, LASCO)에. | ★ Moderate |

---

## 7. References / 참고문헌

**Original paper / 원논문:**
- van de Hulst, H. C., "The electron density of the solar corona", *Bulletin of the Astronomical Institutes of the Netherlands*, Vol. XI, No. 410, pp. 135–149, 1950 February 2. [Bibcode: 1950BAN....11..135V]

**Cited within the paper / 본문 인용:**
- Schuster, A., *Monthly Notices of the Royal Astronomical Society*, 40, 35, 1879.
- Minnaert, M., *Zeitschrift für Astrophysik*, 1, 209, 1930.
- Lyot, B., (coronagraph invention, 1930).
- Grotrian, W., *Z. f. Ap.*, 8, 124, 1934.
- Ludendorff, H., *Sitzungsber. d. Preuss. Ak. d. Wiss.*, 10, 185, 1928 and 16, 200, 1934.
- Bergstrand, Ö., *Études sur la distribution de la lumière dans la couronne solaire*, Stockholm, 1919.
- Bergstrand, Ö., *Arkiv f. Mat. Astr. och Fysik*, 22A, 1, 1930; 25A, 4, 1937; 27A, 95, 1936.
- Mitchell, S., *Hdbuch d. Ap.*, 4, 340, 1929 and 7, 398, 1936.
- Lockyer, W. J. S., *M.N.*, 82, 323, 1922; 91, 908, 1931.
- Baumbach, S., *Astronomische Nachrichten*, 263, 121, 1937; 267, 273, 1939.
- Hase, V. Th., *Abastumani Bulletin*, 7, 73, 1943.
- Nikonov, V. B., *Abastumani Bulletin*, 7, 33, 1943.
- Vashakidze, M. A., *Abastumani Bulletin*, 7, 1, 1943.
- Zakharin, K. G., *Abastumani Bulletin*, 3, 72, 1938.
- Öhman, Y., *Stockholm Annaler*, 15, No. 2, 1947.
- Allen, C. W., *M.N.*, 106, 137, 1947; 107, 426, 1947.
- van de Hulst, H. C., *Astrophysical Journal*, 105, 471, 1947.
- van de Hulst, H. C., *Nature*, 163, 24, 1949.
- Hurahata, M., *Japanese J. Astr. Geophys.*, 21, 173, 1947.
- Dyson, F. W. and Woolley, R. v. d. R., *Eclipses of the Sun and Moon*, p. 143, Oxford, 1937.
- Unsöld, A., *Die Naturwissenschaften*, 34, 194, 1947.
- Waldmeier, M. and Müller, H., *Astr. Mitt. Zürich*, Nos. 154 and 155, 1948.
- Denisse, J. F., Thèse, Paris, 1949.
- Nicolet, M., *Ciel et Terre*, 59, 266, 1943.
- Cohn, W. H., *Ap. J.*, 87, 284, 1938.
- Young, R. K., *Lick Obs. Bull.*, 6, 166, 1911.
- Fessenkoff, B., *Russ. Astr. J.*, 12, 309, 1935.
- Dufay, J. and Grouiller, H., *Lyon Publ.*, 2, 129, 1936.
- Johnson, J. J., *P. A. S. P.*, 46, 226, 1934.
- Abetti, G., *Publ. R. Oss. Arcetri*, 56, 53, 1938; Biozzi, M., ibidem 57, 5, 1939.
- Barocas, V., *Ap. J.*, 89, 486, 1939.
- Waldmeier, M., *Z. f. Ap.*, 21, 85, 1941.
- Fracastoro, M. G., *Publ. R. Oss. Arcetri*, 64, 44, 1948.

**Related modern works / 관련 현대 논문:**
- Saito, K., "A non-spherical axisymmetric model of the solar K-corona of the minimum type", *Annals of the Tokyo Astronomical Observatory*, 1971.
- Newkirk, G. Jr., "Structure of the Solar Corona", *Annual Review of Astronomy and Astrophysics*, 5, 213, 1967.
- Brueckner, G. E., et al., "The Large Angle Spectroscopic Coronagraph (LASCO)", *Solar Physics*, 162, 357, 1995.
- Antonucci, E., et al., "Metis: the Solar Orbiter visible light and ultraviolet coronal imager", *A&A*, 642, A10, 2020.
- **Abbo, L., et al., "Solar Orbiter Metis: Coronal electron density inversion and streamer characterization", *A&A*, 702, A254, 2025. — Paper #61 in this study; direct application of van de Hulst's Eq. (20).**
