---
title: "The Five-Minute Oscillations on the Solar Surface"
authors: Roger K. Ulrich
year: 1970
journal: "The Astrophysical Journal, 162, 993–1002"
topic: Solar_Physics
tags: [helioseismology, p-modes, solar oscillations, acoustic waves, resonant cavity]
status: completed
date_started: 2026-04-14
date_completed: 2026-04-14
---

# 14. The Five-Minute Oscillations on the Solar Surface / 태양 표면의 5분 진동

---

## 1. Core Contribution / 핵심 기여

Ulrich는 1962년 Leighton 등이 발견한 태양 표면의 5분 진동에 대해 근본적으로 새로운 이론적 해석을 제시했다. 기존에는 이 진동이 대류에 의해 국지적으로 생성되는 음향파로 여겨졌으나, Ulrich는 이것이 **태양 내부의 공진 캐비티(resonant cavity)에 갇힌 전역적 음향 정상파(trapped standing acoustic waves)**라고 주장했다. 광구의 급격한 밀도 감소가 상부 반사면을, 태양 내부의 음속 증가에 의한 굴절이 하부 반전점을 형성하여, 이 두 경계 사이에서 특정 주파수와 파수 조합만이 정상파로 존재할 수 있다는 것이다. 이 해석의 핵심 예측은 수평 파수-주파수 $(k_h, \omega)$ 다이어그램에서 진동 모드가 이산적인 능선(discrete ridges)을 형성한다는 것이며, 각 능선은 특정 radial order에 대응한다. 또한 에너지 균형 분석을 통해 이 진동이 overstable(자체 여기)하며, 채색층과 코로나 가열에 필요한 에너지를 공급할 수 있음을 보였다.

Ulrich proposed a fundamentally new theoretical interpretation of the 5-minute oscillations discovered by Leighton et al. in 1962. While these oscillations were previously thought to be locally generated acoustic waves driven by convection, Ulrich argued that they are **globally trapped standing acoustic waves in a resonant cavity within the solar interior**. The rapid density decrease at the photosphere forms the upper reflecting surface, while refraction due to increasing sound speed in the solar interior creates the lower turning point. Only specific combinations of frequency and wavenumber can exist as standing waves between these two boundaries. The key prediction of this interpretation is that oscillation modes form **discrete ridges** in the horizontal wavenumber-frequency $(k_h, \omega)$ diagram, with each ridge corresponding to a specific radial order. Through energy balance analysis, he also showed that the oscillations are overstable (self-exciting) and capable of supplying the energy needed to heat the chromosphere and corona.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction / 서론 (p. 993–994)

Ulrich는 먼저 5분 진동의 발견 이후 제안된 기존 해석들을 검토한다. Lighthill(1952)의 난류 메커니즘에서 유도된 Stein(1968)의 스펙트럼은 30–60초 주기에서 파워 피크를 보이며, 관측된 ~300초 주기와 크게 다르다. 이는 Lighthill 메커니즘이 5분 진동의 원인이 될 수 없음을 시사한다.

Ulrich first reviews the existing interpretations proposed since the discovery of the 5-minute oscillations. The spectrum derived by Stein (1968) from Lighthill's (1952) turbulence mechanism shows a power peak at periods of 30–60 seconds, significantly different from the observed ~300-second period. This suggests the Lighthill mechanism cannot be the cause of the 5-minute oscillations.

중요한 관측적 단서로, Gonczi & Roddier(1969)는 진동이 최소 1시간 동안 위상이 유지됨을 보였는데, 이는 전형적인 대류 셀(과립)의 수명인 7–8분보다 훨씬 길다. 보통 인용되는 8–10분의 진동 수명은 실제 수명이 아니라 **맥놀이 주기(beat period)**로 해석되어야 하며, 이는 두 개 이상의 자연 주파수가 존재함을 의미한다. 이것이 바로 Ulrich의 핵심 동기다: 태양 내부의 가변적 음향 특성이 다중 주파수를 만들어내야 한다.

A crucial observational clue came from Gonczi & Roddier (1969), who showed that oscillations maintain phase coherence for at least 1 hour — far longer than the typical convective cell (granule) lifetime of 7–8 minutes. The commonly quoted 8–10 minute oscillation lifetime should be interpreted not as a true lifetime but as a **beat period**, implying the existence of two or more natural frequencies. This is precisely Ulrich's key motivation: the variable acoustic properties of the solar interior must produce multiple frequencies.

Moore & Spiegel(1966)의 선행 연구가 중요한 실마리를 제공한다: 초단열적(superadiabatic) 온도 구배와 복사 에너지 교환이 존재하면 음향파가 overstable해질 수 있다는 것이다. Ulrich는 이 아이디어를 발전시켜 5분 진동이 갇힌 정상파이고, 기본 모드와 처음 세 배음(overtone)의 고유 주파수를 계산한다.

The prior work of Moore & Spiegel (1966) provides an important clue: acoustic waves can become overstable in the presence of a superadiabatic temperature gradient and radiative exchange of energy. Ulrich develops this idea further, showing that the 5-minute oscillations are trapped standing waves and computing eigenfrequencies for the fundamental and first three overtone modes.

### Part II: Trapped Waves / 갇힌 파동 (p. 994–996)

이 섹션은 논문의 물리적 핵심이다. Ulrich는 Whitaker(1963)의 국소 분산 관계를 사용하여 파동 가둠(wave trapping)의 메커니즘을 설명한다:

This section is the physical core of the paper. Ulrich uses Whitaker's (1963) local dispersion relation to explain the wave trapping mechanism:

$$k_z^2 = \frac{\omega^2 - \omega_0^2}{c^2} - k_h^2\left(1 - \frac{N^2}{\omega^2}\right) \tag{1}$$

여기서 $k_z$는 수직 파수, $k_h$는 수평 파수, $\omega_0 = c/(2H)$는 음향 차단 주파수, $N$은 Brunt-Väisälä 주파수다. 파동이 전파하려면 $k_z^2 > 0$이어야 하고, $k_z^2 < 0$인 영역에서는 파동이 evanescent하게 된다(진폭이 지수적으로 감쇠).

Here $k_z$ is the vertical wavenumber, $k_h$ is the horizontal wavenumber, $\omega_0 = c/(2H)$ is the acoustic cutoff frequency, and $N$ is the Brunt-Väisälä frequency. Waves propagate where $k_z^2 > 0$ and become evanescent (exponentially decaying amplitude) where $k_z^2 < 0$.

**상부 반사면(Upper reflecting surface)**: 광구에서 밀도가 급격히 감소하면 스케일 높이 $H$가 작아지고, 따라서 $\omega_0 = c/(2H)$가 커진다. 5분 진동($\omega \approx 0.021$ sec$^{-1}$)은 광구의 $\omega_0$ 최댓값보다 작으므로 $k_z^2 < 0$이 되어 반사된다. Table 1에서 $\omega_0$의 최댓값은 $z = -239$ km(광구 아래)에서 $3.23 \times 10^{-2}$ sec$^{-1}$이다.

**Upper reflecting surface**: As density drops sharply at the photosphere, the scale height $H$ decreases and thus $\omega_0 = c/(2H)$ increases. The 5-minute oscillation ($\omega \approx 0.021$ sec$^{-1}$) is smaller than the maximum $\omega_0$ at the photosphere, so $k_z^2 < 0$ and the wave is reflected. From Table 1, the maximum of $\omega_0$ is $3.23 \times 10^{-2}$ sec$^{-1}$ at $z = -239$ km (below the photosphere).

**하부 반전점(Lower turning point)**: 태양 내부로 들어갈수록 온도가 증가하여 음속 $c$가 커진다. 수평 성분을 가진 파동은 음속이 충분히 커지는 깊이에서 굴절에 의해 다시 표면 쪽으로 되돌아온다. $k_h$가 작을수록(즉 $\ell$이 작을수록) 더 깊이 침투한다.

**Lower turning point**: As temperature increases deeper into the solar interior, the sound speed $c$ increases. Waves with a horizontal component are refracted back toward the surface at the depth where the sound speed becomes sufficiently large. Smaller $k_h$ (i.e., lower $\ell$) means deeper penetration.

복사 에너지 교환을 고려하면 $k_z$가 복소수가 되며, Ulrich는 임계 수평 파수를 다음과 같이 유도한다:

When radiative energy exchange is included, $k_z$ becomes complex, and Ulrich derives the critical horizontal wavenumber:

$$k_h = \frac{\omega^2\omega^2 + \gamma\omega_R^2 - \omega_0^2(1 + \omega_R^2/\omega^2)}{c^2(\omega^2 + \omega_R^2 - N^2)} \tag{3}$$

여기서 $\omega_R$은 복사-상호작용률(radiative-interaction rate)로, Spiegel(1957)이 유도한 공식으로 구해진다:

where $\omega_R$ is the radiative-interaction rate, obtained from the formula derived by Spiegel (1957):

$$\omega_R = \frac{16\sigma T^3 \kappa}{C_P}(1 - \tau_e \cot^{-1}\tau_e) \tag{4}$$

$\tau_e = (\rho\kappa/k)$는 섭동(perturbation)의 유효 광학 두께이다. 광구 근처에서 $\omega_R$은 $\omega_0$와 비슷한 크기가 되어 복사 효과가 중요해진다.

$\tau_e = (\rho\kappa/k)$ is the effective optical thickness of the perturbation. Near the photosphere, $\omega_R$ becomes comparable to $\omega_0$, making radiative effects important.

**Figure 1**은 이 분석의 시각적 요약이다: 세 가지 진동 주기(180초, 200초, 300초)에 대해 반사층 고도 $z$를 수평 파장 $\lambda_h$의 함수로 그린 것이다. 각 곡선의 오른쪽이 진동이 허용되는 영역이며, 점선은 §III의 고유해(eigensolution)의 위치를 나타낸다. 300초 진동의 경우 반사면이 광구(optical depth unity) 부근에 위치하여, 표면에서 관측 가능한 진폭을 가지게 된다.

**Figure 1** is the visual summary of this analysis: it plots the reflecting layer altitude $z$ versus horizontal wavelength $\lambda_h$ for three oscillation periods (180 s, 200 s, 300 s). The permitted region lies to the right of each curve, and dashed lines show the positions of the eigensolutions from §III. For 300-second oscillations, the reflecting surface is located near optical depth unity (the photosphere), giving observable amplitudes at the surface.

### Part III: Modal Analysis / 모달 분석 (p. 996–998)

Ulrich는 질량 플럭스 $j = \rho v$를 사용하여 운동량 및 연속 방정식을 단순화한다. 선형화된 운동 방정식은:

Ulrich simplifies the momentum and continuity equations using the mass flux $j = \rho v$. The linearized equations of motion are:

$$\frac{\partial j}{\partial t} = -\nabla P' + g\rho' \tag{7}$$

$$\frac{\partial \rho'}{\partial t} = -\nabla \cdot j \tag{8}$$

$$\frac{T\partial S'}{\partial t} = -\frac{T}{\rho}j_z\frac{dS}{dz} - C_P\omega_R T' \tag{9}$$

여기서 프라임(')은 섭동량, $S$는 엔트로피다. 식 (9)의 첫째 항 $-(T/\rho)j_z(dS/dz)$는 유체가 수직으로 이동할 때 배경 엔트로피 구배를 가로지르면서 발생하는 엔트로피 변화이고, 둘째 항 $-C_P\omega_R T'$는 복사에 의한 열적 완화(radiative damping)다.

Here primes denote perturbation quantities and $S$ is entropy. The first term in eq. (9), $-(T/\rho)j_z(dS/dz)$, represents entropy change as fluid moves vertically through the background entropy gradient, and the second term $-C_P\omega_R T'$ represents radiative damping (thermal relaxation).

모달 방정식은 $\partial/\partial t = i\omega$, $\partial^2/\partial x^2 + \partial^2/\partial y^2 = -k_h^2$으로 설정하여 얻어진다. $P'$과 $\rho'$를 상태 방정식으로 소거하면, 복소 진폭의 고도 의존성을 나타내는 연립 미분 방정식이 나온다:

The modal equations are obtained by setting $\partial/\partial t = i\omega$ and $\partial^2/\partial x^2 + \partial^2/\partial y^2 = -k_h^2$. Eliminating $P'$ and $\rho'$ through the equation of state yields a system of coupled differential equations for the altitude dependence of complex amplitudes:

$$\frac{\partial P'}{\partial z} = -i\omega j_z - g\rho', \quad \frac{\partial j_z}{\partial z} = -i\omega\rho' - \frac{k_h^2 P'}{i\omega} \tag{10}$$

$$\rho' - \frac{P'}{c^2} = \frac{N^2 j_z}{i\omega g} - \frac{\omega_R}{i\omega}\left(\rho' - \gamma\frac{P'}{c^2}\right) \tag{11}$$

**내부 경계 조건(Interior boundary condition)**: 깊은 내부에서 $N^2 \sim 0$(대류층에서 단열에 가까움)이므로, 경계 조건은:

**Interior boundary condition**: In the deep interior where $N^2 \sim 0$ (nearly adiabatic in the convection zone), the boundary condition is:

$$\frac{1}{j_z}\frac{dj_z}{dz} = \left(\frac{\omega_0^2 - \omega^2}{c^2} + k_h^2\right)^{1/2} - \frac{\omega_0}{c} \tag{12}$$

이는 하부 반전점 아래에서 파동이 지수적으로 감쇠하는 evanescent 해를 선택하는 조건이다.

This selects the evanescent solution that decays exponentially below the lower turning point.

**외부 경계 조건(Outer boundary condition)**: 상부 경계는 더 복잡하다. 온도 최소(temperature minimum) 위에서 다시 온도가 올라가면 전파가 허용되는 영역이 나타난다. Ulrich는 가장 작은 속도 진폭을 가지는 모드를 선택하는데, 이 모드가 충격파(shock) 형성에 의한 감쇠가 가장 적기 때문이다. 이 경계 조건은 고유 주파수를 약 1% 정확도로 결정한다.

**Outer boundary condition**: The upper boundary is more complex. Above the temperature minimum, the temperature rises again and a propagating region reappears. Ulrich selects the mode with smallest velocity amplitude, since this mode is least damped by shock formation. This boundary condition determines eigenfrequencies to about 1% accuracy.

**Figure 2 — $(k_h, \omega)$ 진단 다이어그램**: 이것이 논문의 가장 중요한 결과다. 고유해의 궤적이 모달 번호로 표시된 실선으로 그려져 있다. 이산적인 능선(ridges) 구조가 명확히 보이며, 각 능선은 특정 radial order $n$에 대응한다. Tanenbaum et al.(1969)의 관측 영역(짧은 점선), Frazier(1968)의 주파수(긴 점선), Gonczi & Roddier(1969)의 주파수(짧은 점선)가 함께 표시되어 있다. 이 기존 관측들은 공간 파수의 해상도가 부족하여 이산적 능선을 구별하지 못했지만, 그들의 결과는 분산 곡선(dispersion lines)과 대략 평행한 대각선 능선을 암시하고 있었다.

**Figure 2 — $(k_h, \omega)$ diagnostic diagram**: This is the most important result of the paper. The loci of eigensolutions are plotted as solid lines labeled by modal number. A clear structure of discrete ridges is visible, each corresponding to a specific radial order $n$. The observed region by Tanenbaum et al. (1969) (short dashed), frequencies by Frazier (1968) (long dashed), and Gonczi & Roddier (1969) (short dashed) are also shown. These prior observations lacked spatial wavenumber resolution to distinguish discrete ridges, but their results hinted at diagonal ridges roughly parallel to the dispersion lines.

Ulrich는 공간 분해능이 낮은 관측이 $0.015 < \omega < 0.032$ sec$^{-1}$ 범위에서 무작위적 주파수 피크를 보일 수밖에 없다고 지적한다 — 이것이 Howard(1967)가 관측한 파워 스펙트럼 피크의 무작위 위치를 설명한다.

Ulrich points out that observations with poor spatial resolution are bound to show more or less random frequency peaks in the range $0.015 < \omega < 0.032$ sec$^{-1}$ — this explains the apparently random locations of power-spectrum peaks observed by Howard (1967).

### Part IV: Energy Balance / 에너지 균형 (p. 999–1001)

#### a) 기본 보존 방정식 / The Fundamental Equation of Conservation

외부 경계 조건의 불완전성 때문에 $\omega$의 허수 부분(성장/감쇠율)을 직접 구할 수 없어, Ulrich는 대안적 방법을 사용한다: 고정된 공간 체적에서 한 주기 동안의 에너지 변화를 비교하는 것이다. 세 가지 에너지 플럭스가 존재한다:

Due to the incomplete understanding of the outer boundary condition, the imaginary part of $\omega$ (growth/decay rate) cannot be determined directly, so Ulrich uses an alternative approach: comparing energy changes in a fixed volume over one cycle. Three energy fluxes exist:

$$\text{work} = -\oint_0^{2\pi/\omega} v_z P' dt \tag{13}$$

$$\text{thermal flux} = -\oint_0^{2\pi/\omega} \rho v_z E \, dt \tag{14}$$

$$\text{radiative flux} = -\oint_0^{2\pi/\omega} \int_{\tau}^{\infty} 12\sigma T^3 T' E_2(\tau') d\tau' dt \tag{15}$$

에너지 밀도는 열적(thermal)과 운동(kinetic) 성분의 합이며:

Energy density is the sum of thermal and kinetic components:

$$\text{thermal} = \oint_0^{2\pi/\omega} [\rho E - (\rho E)_0] dt, \quad \text{kinetic} = \oint_0^{2\pi/\omega} \tfrac{1}{2}\rho v^2 dt \tag{16, 17}$$

Landau & Lifshitz(1959)의 결과를 적용하면:

Applying the result of Landau & Lifshitz (1959):

$$\frac{d}{dt}\int_{z_1}^{z_2} (\text{energy density}) \, dz = \text{flux}(z_1) - \text{flux}(z_2) \tag{18}$$

이 식은 **2차(second order)까지 엄밀한 결과**임을 Ulrich는 강조한다.

Ulrich emphasizes that this equation is **rigorous to second order**.

핵심 결과: 온도 최소(temperature minimum)에서의 플럭스가 **음수(negative)**로 나타났다. 이는 에너지가 태양 내부에서 표면으로 순수하게 방출되고 있음을 의미하며, 따라서 진동이 **overstable(자체 여기적)**이다. 물리적 메커니즘은 Souffrin & Spiegel(1967)이 중력파에 대해 논의한 것과 동일하다: 초단열적 온도 구배에서 복사 에너지 교환이 파동의 에너지를 증폭시킨다. 진폭은 20–30주기마다 $e$배 성장해야 하지만, 이 성장을 대류 속도장과의 결합 없이도 설명할 수 있다는 점이 중요하다.

Key result: the flux at the temperature minimum turned out to be **negative**. This means energy is being net-released from the solar interior to the surface, and therefore the oscillations are **overstable (self-exciting)**. The physical mechanism is the same as that discussed by Souffrin & Spiegel (1967) for gravity waves: radiative energy exchange amplifies wave energy in a superadiabatic temperature gradient. The amplitude should grow by a factor of $e$ every 20–30 periods, but the important point is that this growth can be explained without coupling to convective velocity fields.

#### b) 광구 위의 소산 / Dissipation above the Photosphere

진동이 성장하는 것이 관측되지 않으므로, 소산 메커니즘이 존재해야 한다. 온도 최소 위의 영역에서 음파는 자기유체역학적(MHD) 파동으로 변환되어 충격파를 형성하고 빠르게 소산된다(Osterbrock 1961). Ulrich는 관측된 속도 진폭 0.2 km/s를 각 모드에 할당하여 필요 소산율을 계산한다(Table 2).

Since the oscillations are not observed to grow, a dissipation mechanism must exist. Above the temperature minimum, acoustic waves are converted to magnetoacoustic waves, which form shocks and dissipate rapidly (Osterbrock 1961). Ulrich calculates the required dissipation rate by assigning the observed velocity amplitude of 0.2 km/s to each mode (Table 2).

**Table 2**의 결과: 주기 ~300초(5분)이고 가장 큰 에너지를 생산하는 모드들의 소산율이 Athay(1966)가 관측한 에너지 손실률 $5.6 \times 10^6$ ergs cm$^{-2}$ sec$^{-1}$과 잘 일치한다. 이것은 **5분 진동이 채색층과 코로나 가열의 에너지원**일 수 있음을 시사하는 중요한 결과다.

**Table 2** results: the dissipation rate of modes with period ~300 s (5 min) that produce the most energy is in good agreement with the energy loss rate of $5.6 \times 10^6$ ergs cm$^{-2}$ sec$^{-1}$ observed by Athay (1966). This is an important result suggesting that **5-minute oscillations could be the energy source for chromospheric and coronal heating**.

#### c) $\omega_R$의 불확실성 / Uncertainty in $\omega_R$

$\omega_R$의 가장 큰 불확실성은 광학적으로 두꺼운(optically thick) 영역에서의 복사-상호작용률이다. 임의의 스케일 팩터 $f$를 $\omega_R$에 곱하여 ($f = 0.1$에서 $10.0$, 위상각 $-90°$에서 $+90°$) 감도 분석을 수행한 결과, 파워 출력은 대략 $f^{-0.3}$에 비례하며 위상각 $\pm 45°$에서는 사실상 영향을 받지 않는다. $\pm 90°$에서만 overstability가 사라지는데, 이는 온도가 매우 작게 유지되어 $(T, S)$ 및 $(P, V)$ 다이어그램의 루프가 거의 사라지기 때문이다. Ulrich는 Spiegel의 공식이 $f = 50$ 만큼 틀려야 overstability가 제거된다고 주장하며, 이는 비현실적이라고 결론짓는다.

The largest uncertainty in $\omega_R$ is the radiative-interaction rate in optically thick regions. Sensitivity analysis with an arbitrary scale factor $f$ multiplying $\omega_R$ ($f = 0.1$ to $10.0$, phase angles from $-90°$ to $+90°$) showed that power output scales roughly as $f^{-0.3}$ and is virtually unaffected for phase angles of $\pm 45°$. Overstability vanishes only at $\pm 90°$, because $T'$ is kept very small and the loops in the $(T, S)$ and $(P, V)$ diagrams nearly disappear. Ulrich argues that Spiegel's formula would have to be wrong by a factor of 50 to eliminate overstability, which he considers unrealistic.

### Part V: Conclusions / 결론 (p. 1001–1002)

Ulrich는 세 가지 핵심 결론을 제시한다:

Ulrich presents three key conclusions:

1. **5분 진동은 overstable하며 채색층·코로나 가열에 필요한 복사 에너지를 공급할 수 있다.** 이는 진동의 에너지원이 대류 속도장과의 직접적 결합이 아님을 의미한다.

1. **The 5-minute oscillations are overstable and capable of supplying the radiant energy needed for chromospheric and coronal heating.** This means the energy source is not direct coupling with convective velocity fields.

2. **$(k_h, \omega)$ 다이어그램에서 진동은 이산적 능선에 한정되어야 한다.** 이 능선들은 아직 관측으로 확인되지 않았지만(1970년 기준), $k_h$와 $\omega$의 분해능이 부족했기 때문이다.

2. **Oscillations should be confined to discrete lines on the $(k_h, \omega)$ diagram.** These lines have not yet been observationally confirmed (as of 1970), due to insufficient resolution in $k_h$ and $\omega$.

3. **관측 조건**: (a) 공간 분석은 2차원이어야 하고, (b) 파장 8000 km의 진동을 분해하려면 ~60,000 km 직경의 영역을 ~1시간 이상 관측해야 하며, (c) 3000 km 간격의 점들에서 속도 차이를 측정할 수 있어야 한다(4" of arc). 짧은 파장의 진동이 분해하기 쉽지만, 태양 내부 구조에 대한 정보는 긴 파장이 더 많이 제공한다.

3. **Observational requirements**: (a) spatial analysis must be two-dimensional, (b) resolving oscillations at wavelength 8000 km requires observing a region ~60,000 km in diameter for ~1 hour or more, and (c) velocity differences must be measurable at points separated by 3000 km (4" of arc). Short-wavelength oscillations are easier to resolve, but longer wavelengths provide more information about the solar interior structure.

---

## 3. Key Takeaways / 핵심 시사점

1. **5분 진동은 전역적 p-mode 정상파다** — 국지적 대기 현상이 아니라 태양 내부의 공진 캐비티에 갇힌 음향 정상파로서, 전체 태양의 구조와 연결된 전역적 현상이다.
   **5-minute oscillations are global p-mode standing waves** — not local atmospheric phenomena but acoustic standing waves trapped in a resonant cavity within the solar interior, connected to the structure of the entire Sun.

2. **두 경계면이 공진 캐비티를 형성한다** — 상부: 광구의 밀도 급감에 의한 반사, 하부: 음속 증가에 의한 굴절. 이 두 경계 사이에서만 파동이 전파 가능하다.
   **Two boundaries form the resonant cavity** — upper: reflection by rapid density drop at the photosphere, lower: refraction by increasing sound speed. Waves can propagate only between these two boundaries.

3. **$(k_h, \omega)$ 다이어그램에서 이산적 능선 구조가 나타난다** — 이것이 핵심 예측이며, 1975년 Deubner에 의해 관측적으로 확인됨으로써 helioseismology가 본격적으로 시작되었다.
   **Discrete ridge structure appears in the $(k_h, \omega)$ diagram** — this is the key prediction, confirmed observationally by Deubner in 1975, which launched helioseismology as a field.

4. **진동은 overstable(자체 여기적)이다** — 초단열적 온도 구배에서 복사 에너지 교환이 파동 에너지를 증폭시키며, 대류 속도장과의 결합이 필요 없다.
   **Oscillations are overstable (self-exciting)** — radiative energy exchange in a superadiabatic temperature gradient amplifies wave energy, without requiring coupling to convective velocity fields.

5. **5분 진동이 채색층·코로나 가열의 에너지원이 될 수 있다** — 에너지 소산율이 관측된 에너지 손실률($5.6 \times 10^6$ ergs cm$^{-2}$ sec$^{-1}$)과 일치한다.
   **5-minute oscillations could be the energy source for chromospheric/coronal heating** — the dissipation rate matches the observed energy loss rate ($5.6 \times 10^6$ ergs cm$^{-2}$ sec$^{-1}$).

6. **서로 다른 $\ell$ 모드는 태양의 서로 다른 깊이를 탐사한다** — 낮은 $\ell$(긴 수평 파장)은 태양 깊숙이 침투하고, 높은 $\ell$(짧은 수평 파장)은 표면 근처만 탐사한다. 이것이 helioseismology의 핵심 원리다.
   **Different $\ell$ modes probe different depths of the Sun** — low $\ell$ (long horizontal wavelength) penetrates deep, high $\ell$ (short horizontal wavelength) probes only near the surface. This is the fundamental principle of helioseismology.

7. **관측된 8–10분 수명은 맥놀이 주기다** — 여러 고유 주파수가 간섭하여 겉보기 수명을 만들어내며, 이는 다중 모드의 존재를 지지하는 증거다.
   **The observed 8–10 minute lifetime is a beat period** — multiple eigenfrequencies interfere to create an apparent lifetime, supporting the existence of multiple modes.

8. **관측 분해능의 한계가 이전 검증 실패를 설명한다** — $k_h$ 분해능이 부족한 관측은 이산적 능선을 보지 못하고 무작위적 피크를 보게 된다.
   **Observational resolution limitations explain previous failure to verify** — observations with insufficient $k_h$ resolution cannot see discrete ridges and instead show random peaks.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 분산 관계 체계 / Dispersion Relation Framework

파동 가둠의 출발점은 Whitaker(1963)의 국소 분산 관계다:

The starting point for wave trapping is Whitaker's (1963) local dispersion relation:

$$k_z^2 = \frac{\omega^2 - \omega_0^2}{c^2} - k_h^2\left(1 - \frac{N^2}{\omega^2}\right)$$

특성 주파수 정의 / Characteristic frequency definitions:
- 음향 차단 주파수 / Acoustic cutoff frequency: $\omega_0 = c/(2H)$
- Brunt-Väisälä 주파수 / Brunt-Väisälä frequency: $N^2 = -\frac{g}{\rho}\left(\frac{\partial\rho}{\partial S}\right)_P \frac{dS}{dz}$
- 복사 상호작용률 / Radiative interaction rate: $\omega_R = \frac{16\sigma T^3\kappa}{C_P}(1 - \tau_e\cot^{-1}\tau_e)$

### 4.2 전파/반사 조건 / Propagation/Reflection Conditions

| 조건 / Condition | 의미 / Meaning |
|---|---|
| $k_z^2 > 0$ | 파동 전파(propagating) |
| $k_z^2 < 0$ | 파동 evanescent (감쇠/반사) |
| $\omega > \omega_0$ | 수직 전파 가능 (고주파) |
| $\omega < \omega_0$ | 수직 전파 불가 (반사) |

### 4.3 선형화된 운동 방정식 / Linearized Equations of Motion

$$\frac{\partial j}{\partial t} = -\nabla P' + g\rho' \tag{7}$$
$$\frac{\partial \rho'}{\partial t} = -\nabla \cdot j \tag{8}$$
$$\frac{T\partial S'}{\partial t} = -\frac{T}{\rho}j_z\frac{dS}{dz} - C_P\omega_R T' \tag{9}$$

### 4.4 경계 조건 / Boundary Conditions

내부 (evanescent decay) / Interior:
$$\frac{1}{j_z}\frac{dj_z}{dz} = \left(\frac{\omega_0^2 - \omega^2}{c^2} + k_h^2\right)^{1/2} - \frac{\omega_0}{c} \tag{12}$$

외부: 온도 최소 위에서 속도 진폭이 최소인 모드 선택
Exterior: select mode with minimum velocity amplitude above temperature minimum

### 4.5 에너지 균형 / Energy Balance

세 가지 에너지 플럭스의 합:
Sum of three energy fluxes:

$$\text{work} + \text{thermal flux} + \text{radiative flux} = \frac{d}{dt}\int(\text{energy density})\,dz$$

온도 최소에서 플럭스 < 0 → overstable (에너지 방출)
Flux < 0 at temperature minimum → overstable (energy release)

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1952 ─── Lighthill ─── 난류에 의한 음향파 발생 이론
         │                Acoustic wave generation by turbulence
         │
1957 ─── Spiegel ─── 복사 상호작용률 공식 유도
         │              Radiative interaction rate formula
         │
1961 ─── Osterbrock ─── 채색층 가열에서 음향파의 역할
         │                 Role of acoustic waves in chromospheric heating
         │
1962 ─── Leighton, Noyes & Simon ─── ★ 5분 진동 발견
         │                               Discovery of 5-minute oscillations
         │
1966 ─── Moore & Spiegel ─── 초단열 구배에서 음향파의 overstability
         │                      Overstability of acoustic waves in superadiabatic gradient
         │
1968 ─── Frazier ─── 고해상도 진동 관측 (이중 피크 발견)
         │              High-resolution oscillation observations (double peak)
         │
1969 ─── Gonczi & Roddier ─── 진동의 장시간 위상 유지 관측
         │                        Long phase coherence of oscillations
         │
1969 ─── Tanenbaum et al. ─── 1D 파워 스펙트럼 관측
         │                        1D power spectrum observations
         │
1970 ─── ★ ULRICH ─── 전역 p-mode 정상파 해석 (이 논문)
         │                Global p-mode standing wave interpretation (THIS PAPER)
         │
1971 ─── Leibacher & Stein ─── 독립적으로 유사한 해석 제안
         │                        Independently propose similar interpretation
         │
1975 ─── Deubner ─── ★ k-ω 다이어그램 관측으로 Ulrich 예측 확인
         │               k-ω diagram observation confirms Ulrich's prediction
         │
1979 ─── Claverie et al. ─── 전구(whole-disk) 진동 관측
         │                       Whole-disk oscillation observations
         │
1980s── Duvall, Harvey, etc. ─── helioseismology 본격 발전
                                    Full development of helioseismology
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Leighton, Noyes & Simon (1962) — Paper #13 | 5분 진동의 발견. Ulrich 논문의 직접적 동기. / Discovery of 5-min oscillations. Direct motivation for Ulrich's paper. | 이 발견 없이는 Ulrich의 이론이 존재하지 않음. / Ulrich's theory would not exist without this discovery. |
| Moore & Spiegel (1966) | 초단열 구배에서 음향파의 overstability를 처음 제안. / First proposed overstability of acoustic waves in superadiabatic gradient. | Ulrich가 에너지 균형 분석에서 직접 참조한 메커니즘. / Mechanism directly referenced by Ulrich in energy balance analysis. |
| Leibacher & Stein (1971) | Ulrich와 독립적으로 같은 해석 제안. / Independently proposed the same interpretation as Ulrich. | 두 독립적 연구의 수렴이 이론의 신뢰성을 높임. / Convergence of two independent studies enhances the theory's credibility. |
| Deubner (1975) | $k$-$\omega$ 다이어그램에서 이산적 능선을 관측적으로 확인. / Observationally confirmed discrete ridges in the $k$-$\omega$ diagram. | Ulrich의 핵심 예측의 결정적 검증. Helioseismology의 실질적 시작점. / Definitive verification of Ulrich's key prediction. Effective starting point of helioseismology. |
| Lighthill (1952) / Stein (1968) | 난류 음향파 발생 이론 — Ulrich가 기각한 대안적 해석. / Turbulence acoustic wave generation — alternative interpretation rejected by Ulrich. | 예측 주기(30–60초)가 관측(~300초)과 불일치하여 기각됨. / Predicted period (30–60 s) inconsistent with observations (~300 s), hence rejected. |
| Frazier (1968) | 고해상도 관측에서 이중 주파수 피크 발견. / Found double frequency peak in high-resolution observations. | Ulrich의 다중 모드 예측과 일치하는 관측 증거. / Observational evidence consistent with Ulrich's prediction of multiple modes. |

---

## 7. References / 참고문헌

- Ulrich, R. K. (1970). "The Five-Minute Oscillations on the Solar Surface." *The Astrophysical Journal*, 162, 993–1002. [DOI: 10.1086/150350]
- Leighton, R. B., Noyes, R. W., & Simon, G. W. (1962). "Velocity Fields in the Solar Atmosphere. I. Preliminary Report." *The Astrophysical Journal*, 135, 474.
- Moore, D. W., & Spiegel, E. A. (1966). *The Astrophysical Journal*, 143, 871.
- Lighthill, M. J. (1952). "On Sound Generated Aerodynamically." *Proceedings of the Royal Society of London A*, 211, 564.
- Stein, R. F. (1968). *The Astrophysical Journal*, 154, 297.
- Frazier, E. (1968). *Zeitschrift für Astrophysik*, 68, 345.
- Gonczi, G., & Roddier, F. (1969). *Solar Physics*, 8, 225.
- Tanenbaum, A. S., Wilcox, J. M., Frazier, E. N., & Howard, R. (1969). *Solar Physics*, 9, 328.
- Leibacher, J. W., & Stein, R. F. (1971). *Astrophysical Letters*, 7, 191.
- Deubner, F.-L. (1975). "Observations of Low Wavenumber Nonradial Eigenmodes of the Sun." *Astronomy and Astrophysics*, 44, 371.
- Whitaker, W. A. (1963). *The Astrophysical Journal*, 137, 914.
- Spiegel, E. A. (1957). *The Astrophysical Journal*, 126, 202.
- Souffrin, P., & Spiegel, E. A. (1967). *Annales d'Astrophysique*, 30, 985.
- Athay, R. G. (1966). *The Astrophysical Journal*, 146, 223.
- Osterbrock, D. E. (1961). *The Astrophysical Journal*, 134, 347.
- Landau, L. D., & Lifshitz, E. M. (1959). *Fluid Mechanics*. London: Pergamon Press.
