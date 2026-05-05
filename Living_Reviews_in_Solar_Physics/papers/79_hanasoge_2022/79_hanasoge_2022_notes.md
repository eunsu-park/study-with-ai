---
title: "Surface and Interior Meridional Circulation in the Sun"
authors: Shravan M. Hanasoge
year: 2022
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-022-00034-7"
topic: Living_Reviews_in_Solar_Physics
tags: [meridional_circulation, helioseismology, solar_dynamo, flux_transport, time_distance, ring_diagram]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 79. Surface and Interior Meridional Circulation in the Sun / 태양의 표면 및 내부 자오면 순환

---

## 1. Core Contribution / 핵심 기여

**한국어:**
Hanasoge(2022)는 태양 자오면 순환(meridional circulation, MC)에 관한 현대적 이해를 체계적으로 리뷰한다. MC는 표면에서 적도 → 극 방향으로 약 20 m/s로 흐르는 poleward branch와 대류층 내부 어디선가 방향을 반대로 바꾸어 적도 쪽으로 되돌아오는 return flow (추정 2–5 m/s)로 구성된 축대칭(axisymmetric) 흐름이다. 그 진폭은 평균 자전율의 약 1%, 차등자전의 약 7% (∼300 m/s)에 비해 현저히 작으며, 이에 필적하거나 오히려 상회하는 체계적 오차(centre-to-limb shift, $B_0$ 각, $P$ 각, instrumental upside-down effect) 때문에 정확한 inference는 매우 어렵다. 저자는 다음을 통합적으로 논한다: (i) 6가지 표면 측정 기법(흑점/자기요소/도플러/Supergranule/granulation/RDA), (ii) 각운동량 수송 및 thermal-wind 균형을 통한 MC의 이론적 역할, (iii) flux-transport dynamo에서 MC의 페이스메이커 역할, (iv) 3D 수치 시뮬레이션과 평균장 예측의 불일치, (v) time-distance 헬리오지스몰로지의 수학적 토대와 sensitivity kernel을 이용한 역문제, (vi) 주요 체계적 오차들의 기원과 보정 방법, (vii) 단일 셀 대 다중 셀 구조 논쟁과 return-flow 깊이 문제. 결론적으로 현재의 증거는 적어도 $r=0.9\,R_\odot$까지는 단일 셀 poleward 흐름을 지지하지만, 그 아래 심층의 구조와 귀환점(return depth)은 여전히 미해결이다.

**English:**
Hanasoge (2022) provides a comprehensive review of solar meridional circulation (MC), an axisymmetric flow with a ~20 m/s equator-to-pole poleward surface branch and a much slower return flow somewhere in the convection zone. The amplitude is only ~1% of the mean rotation rate and comparable to or smaller than numerous systematical errors (centre-to-limb shift, $B_0$ angle, $P$ angle, instrumental upside-down effect), making accurate inferences exceptionally difficult. The paper integrates: (i) six surface-measurement techniques (sunspots, magnetic elements, Doppler, supergranules, granules, ring-diagram), (ii) the theoretical role of MC in angular-momentum transport and thermal-wind balance, (iii) MC as the pacemaker of the flux-transport dynamo, (iv) disagreement between 3D simulations and mean-field predictions, (v) the mathematical basis of time-distance helioseismology and inverse problems using sensitivity kernels, (vi) the origin and correction of major systematical errors, (vii) the ongoing single-cell vs. multi-cell controversy and the depth of the return flow. Current evidence supports a single-cell poleward flow down to at least $0.9\,R_\odot$; the deep return flow and whether a second cell exists remain open.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Historical Setting / 서론과 역사적 배경 (Section 1)

**한국어:**
- MC 이야기는 흑점(sunspots)과 뗄 수 없이 얽혀 있다. 흑점은 저위도에서 출현하며 그 빈도는 11년 주기로 변동한다. 왜 이 특정 시간 규모인가라는 질문이 해법 모델 전체를 지배한다 (Charbonneau 2020).
- MC의 가장 오래된 이론적 선구는 **Eddington-Sweet(1925)** circulation: 자전이 적도를 벌어지게 하고 극을 편평하게 하여 복사층(radiative zone)에서 대규모 자오면 흐름을 유발. 그러나 그 timescale은 항성 수명 이상이어서 태양 주기 논의에는 무관하다.
- 현대적 대류층(convection zone) 내 MC 논의는 **Lebedinsky(1941) → Kippenhahn(1959, 1963)**의 효과적 이방성 점성(anisotropic viscosity) 모델로 정립. Kippenhahn은 층화된 대류가 유체 운동의 등방성을 파괴해 차등자전과 MC를 동시에 낳는다고 보였다.
- MC는 표면 $\sim 20$ m/s로 적도→극 방향; 질량 보존을 위해 고위도에서 내부로 침강, 깊은 곳 어디에선가 적도 방향으로 되돌아오고, 다시 적도 부근에서 표면으로 상승하여 순환 고리를 닫는다.

**English:**
- The story of MC is intertwined with sunspots (low-latitude, 11-yr cycle) and with the question of what sets the cycle period (Charbonneau 2020).
- The first theoretical anticipation of MC was **Eddington-Sweet (1925)**: rotation causes equatorial bulge and polar flattening, driving a large-scale meridional circulation in the radiative zone. But its timescale is stellar-lifetime-long and irrelevant to the 11-yr cycle.
- The modern convection-zone story begins with **Lebedinsky (1941)** and **Kippenhahn (1959, 1963)**: stratified convection breaks isotropy of fluid motion, generating both differential rotation and MC.
- MC picture: surface ~20 m/s equator→pole; mass conservation forces the loop to close — flow submerges at high latitudes, reverses direction at some depth, drifts equatorward, and rises at the equator.

### Part II: Surface Observations / 표면 관측 (Section 2)

**한국어:**
Fig. 1–2 (p. 4)에서 4가지 기법(sunspots, magnetic elements, helioseismology, direct Doppler)이 비교된다. **북향 속도(northward velocity)**는 저위도에서 poleward로 모두 일치하되 진폭은 기법마다 다르다 ($\sim$5–20 m/s).

1. **Sunspot tracking (§2.1)**: Dyson & Maunder (1913), Tuominen (1942). 저위도 적도방향, 고위도 극방향의 흐름 보고. 그러나 흑점은 플라즈마 속도를 정확히 따르지 않으며, 고위도나 minima에선 흑점 자체가 희소해 신뢰도가 낮다. Ward (1973): 저위도 MC < 1 m/s. Ribes et al. (1985): ~100 m/s의 교대 흐름 보고 — 서로 크게 불일치.
2. **Magnetic-element tracking (§2.2)**: Schroeter & Woehl (1975), Topka et al. (1982), Komm et al. (1993). Hathaway & Rightmire (2010)는 75°까지 MC 측정 가능함을 보임. 소규모 자기요소가 축대칭으로 drift하는 것을 추적. 단점: 자기확산 속도와 플라즈마 속도가 다르고 (Dikpati et al. 2010), 요소의 rootedness 깊이가 위도별로 다를 가능성.
3. **Doppler estimation (§2.3)**: Duvall (1978, 1979) — 첫 현대적 측정. **20 m/s poleward between 10°–50° latitude**. Convective blueshift (centre-to-limb 효과)이 주된 방해 요인. Duvall (1979)의 기여: 동서 도플러로부터 blueshift를 추정해 빼는 방법 도입.
4. **Supergranular waves & tracking (§2.4)**: Gizon et al. (2003)은 TD f-mode 분석으로 supergranule의 분산 관계를 발견. Northward/southward 파가 MC에 의해 Doppler-shift된 주파수 차이 $kV$ ($k$: 남북 파수, $V$: 국소 MC 속도)로 $V$ 추출. Schou (2003)는 Doppler 측정에도 supergranule 분산 기법 적용, 고위도 ~80°까지 MC 확장, **추가 위도 셀의 증거는 없음**을 보고.
5. **Granulation tracking (§2.5)**: Roudier et al. (2018) — supergranule 추적과 일관, 45° 이상에서는 equatorward 흐름 보고. 역시 centre-to-limb 영향을 많이 받음.
6. **Ring-diagram analysis (§2.6)**: Hill (1988) — $15°\times15°$ 패치 위 고차 모드 주파수 shift로 국소 평균류 추정. 여러 patch의 평균으로 저해상도 MC 이미지 구성. Fig. 11과 같이 solar-cycle-dependent한 MC 변화 연구에 유용.

**English:**
Figs. 1–2 compare four techniques; poleward agreement in direction but amplitudes differ by factor ~4.

1. **Sunspot tracking**: Dyson & Maunder (1913), Tuominen (1942), Ward (1973: <1 m/s at low latitudes), Ribes et al. (1985: alternating ~100 m/s). Mutually inconsistent; sunspots don't track plasma well and are scarce at high latitudes and minima.
2. **Magnetic-element tracking**: Schroeter & Woehl (1975), Komm et al. (1993), Hathaway & Rightmire (2010) reach 75°. Issues: magnetic diffusion ≠ plasma velocity (Dikpati et al. 2010); elements rooted at different depths may sample different flows.
3. **Doppler estimation**: Duvall (1978, 1979) established ~20 m/s poleward between 10°–50°. Key: subtract E-W blueshift to obtain N-S MC. Convective blueshift (centre-to-limb) is the main contaminant.
4. **Supergranular waves**: Gizon et al. (2003) — f-mode TD reveals dispersion relation; N/S-going supergranule waves have frequencies shifted by $kV$, giving MC. Schou (2003) finds no evidence of high-latitude additional cell up to 80°.
5. **Granulation tracking**: Roudier et al. (2018), agreement with supergranule tracking; equatorward beyond 45° seen in some data.
6. **RDA**: Hill (1988), $15°\times15°$ patches; coarse but robust for cycle-dependence studies.

### Part III: MC in Global Angular Momentum Balance / 전역 각운동량 균형에서의 MC (Section 3)

**한국어:**
이 섹션은 MC의 **이론적 핵심**이다. 축대칭 각운동량 방정식(Eq. 1)은 다음과 같다:

$$
\nabla\cdot\Big(\rho r\sin\theta\,\langle u'_\phi \mathbf{u}'_m\rangle + \rho\mathbf{u}_m r^2\sin^2\theta\,\Omega + \rho\nu r^2\sin^2\theta\,\nabla\Omega - r\sin\theta B_\phi\mathbf{B}_m - r\sin\theta\,\langle B'_\phi \mathbf{B}'_m\rangle\Big)=0
$$

각 항 해설:
- $\rho r\sin\theta\langle u'_\phi \mathbf{u}'_m\rangle$ — Reynolds stress (난류 각운동량 수송)
- $\rho\mathbf{u}_m r^2\sin^2\theta\,\Omega$ — MC에 의한 각운동량 이류
- $\rho\nu r^2\sin^2\theta\,\nabla\Omega$ — 점성 확산
- $r\sin\theta B_\phi\mathbf{B}_m$ — 평균 Maxwell 응력
- $r\sin\theta\langle B'_\phi \mathbf{B}'_m\rangle$ — 요동 Maxwell 응력

Lorentz 및 점성 항이 작다는 근사(Miesch 2005; Karak et al. 2014)에서, 단위질량당 각운동량 $\mathcal{L}=r^2\sin^2\theta\,\Omega$을 도입하면 anelastic 근사 $\nabla\cdot(\rho\mathbf{u}_m)=0$ 하에:

$$
\nabla\cdot(\rho r\sin\theta\langle u'_\phi \mathbf{u}'_m\rangle)=-\rho\mathbf{u}_m\cdot\nabla\mathcal{L}
$$

**gyroscopic pumping**: Reynolds stress 발산의 부호가 MC 방향을 결정. $\nabla\cdot(\rho r\sin\theta\langle u'_\phi\mathbf{u}'_m\rangle)<0$이면 MC는 회전축에서 멀어지는 방향 (저위도 poleward 표면 흐름). $\nabla\mathcal{L}$은 회전축에서 멀어지는 방향이므로 $\mathcal{L}$의 cylindrical 근사가 자연스럽게 등장한다 (Fig. 3 upper-left).

Fig. 4 (Featherstone & Miesch 2015): Anelastic Spherical Harmonic (ASH) 시뮬레이션에서 저위도에서 표면은 counter-clockwise, 깊은 곳은 clockwise MC. 고위도는 반대. 이 패턴이 "banana cells"와 "ballistic downflows"로부터 유도된다.

**Thermal-wind balance (Eq. 3)**: 자기장과 점성 무시시,

$$
\Omega_0\frac{\partial\Omega}{\partial z}\approx \frac{g}{2 C_p r}\frac{\partial\langle S\rangle}{\partial\theta}
$$

- $z=r\cos\theta$: 자전축 좌표
- $\langle S\rangle$: 자전축 둘레 평균 엔트로피
- 오른편은 baroclinicity (위도 엔트로피 구배)

Taylor-Proudman 상태 ($\partial_z\Omega=0$)의 편차가 baroclinicity에 의해 유지됨. 관측되는 위도 온도 차 $\sim$1.5 K (Kuhn et al. 1998) 및 <2.5 K (Rast et al. 2008).

**Torsional oscillations**: Kosovichev & Schou(1997) — cycle에 따라 10 m/s 수준의 $\Omega$ 변동. MC 변동은 cycle 평균대비 ~2-4 m/s (Gizon & Rempel 2008).

**English:**
This section is the theoretical heart. The axisymmetric angular-momentum equation (Eq. 1) balances five terms; neglecting Lorentz and viscous terms gives:

$$
\nabla\cdot(\rho r\sin\theta\langle u'_\phi\mathbf{u}'_m\rangle)=-\rho\mathbf{u}_m\cdot\nabla\mathcal{L}
$$

This is the **gyroscopic pumping** principle: divergence of Reynolds stress sets MC direction. Fig. 3 shows cylindrical iso-$\mathcal{L}$ surfaces in the Sun; if Reynolds stress were zero, poleward MC would spin up the poles contradicting observation, so Reynolds stresses are essential.

Fig. 4 (Featherstone & Miesch 2015) illustrates from ASH simulations: counter-clockwise upper-cell, clockwise lower-cell at low latitudes.

**Thermal-wind balance (Eq. 3)**: in the deep interior where $\mathbf{B}$ and Reynolds stresses are weak,
$$\Omega_0\,\partial_z\Omega \approx (g/2 C_p r)\,\partial_\theta\langle S\rangle.$$
Departures from Taylor-Proudman (cylindrical $\Omega$) are set by latitudinal entropy gradient. Observed $\sim$1.5 K (Kuhn 1998), <2.5 K (Rast 2008) pole-equator temperature difference.

**Torsional oscillations**: ~10 m/s $\Omega$ variations over the cycle; surface MC varies by 2–4 m/s (Gizon & Rempel 2008).

### Part IV: MC in the Flux-Transport Dynamo / 선속 수송 다이나모에서의 MC (Section 4)

**한국어:**
Babcock(1961), Wang et al.(1991), Choudhuri et al.(1995), Durney(1995): **flux-transport dynamo 모델**. 주요 아이디어:

1. 양극성 활동영역(bipolar active regions)은 Joy's law에 따라 약간 경사진 tilt를 가진다.
2. 선도(leading)/후행(following) 극성 비대칭이 표면 MC에 의해 극 쪽으로 수송되어 이전 주기의 극성 자기장을 상쇄하고 새 주기를 수립.
3. 극성 자기장은 MC 귀환 흐름에 의해 심부로 내려가 대류층 바닥의 magnetic buoyancy 안정 영역에 저장됨.
4. 차등자전이 poloidal → toroidal 선속 변환을 ($\Omega$-효과), 자기 buoyancy로 활동영역이 표면으로 부상, 사이클이 완성.

MC의 역할:
- 빠른 MC ⇒ 짧은 주기, 빠른 poloidal 선속 역전
- 느린 MC ⇒ 긴 주기 (1–2 Gauss poloidal, 1–2 Gauss toroidal — $\sim 2\times10^{22}$ Mx poloidal vs. $\sim 2\times10^{23}$ Mx toroidal; §4 항목 2)
- Maunder minimum(1645–1715) 등 grand minima는 MC의 심각한 약화로 설명 가능 (Karak et al. 2014)
- Hazra et al. (2014): 관측된 표면 MC 속도를 대류층 바닥에서 equatorward로 방향을 맞추면 시뮬레이션에서 주기 재현 가능.
- Cameron & Schüssler (2017): **중요** — induction 방정식은 $\mathbf{v}\times\mathbf{B}$에 의존, 즉 toroidal 선속이 저장되는 영역의 평균 MC가 주기를 결정한다는 통찰.

**English:**
**Flux-transport dynamo**: Babcock (1961), Wang et al. (1991), Choudhuri et al. (1995), Durney (1995):
1. Bipolar active regions have Joy's-law tilt.
2. Surface MC advects leading/following asymmetry poleward, cancelling old polar field and building new-cycle polar field.
3. Polar field is dragged down by the return flow to the stable base of CZ.
4. Differential rotation winds up poloidal into toroidal ($\Omega$-effect); buoyancy lifts it to surface.

MC role: faster MC ⇒ shorter cycle; weak MC ⇒ long cycle (or grand minima, Karak et al. 2014). Cameron & Schüssler (2017): what matters is average MC in the region where toroidal flux is stored.

### Part V: MC in Numerical Simulations / 수치 시뮬레이션에서의 MC (Section 5)

**한국어:**
Gilman & Glatzmaier (1981) 이래로 3D convection 시뮬레이션이 발전. Fig. 5 (Featherstone & Miesch 2015)는 rotation rate을 $2\Omega_0$에서 $0.75\Omega_0$로 바꿔가며 차등자전과 MC 프로파일을 보여줌:

- **약하게 회전하는 시스템** ($Ro\gtrsim 1$): anti-solar rotation (극이 적도보다 빠름), **단일 셀** MC.
- **빠르게 회전하는 시스템** ($Ro\ll 1$): solar-like 차등자전, **다중 셀** MC (위도/반경 모두).

여기서 Rossby number $Ro=\omega_c/\Omega=2\pi/\tau_c\Omega$ ($\tau_c$: 대류 회전시간). 태양은 어느 쪽인가? — 시뮬레이션들이 태양과 유사한 결과를 내기 위해선 $Ro$가 태양값보다 작은 값(즉 더 빠른 자전)을 사용해야 하는 경향. 이것이 현재 시뮬레이션이 **convective conundrum**으로 불리는 미해결 문제의 일부이다.

**Mean-field approaches**: Kitchatinov & Ruediger (1993, 1995), Kitchatinov & Olemskoy (2011). Fig. 6 — anisotropic thermal flux와 Reynolds stress 분포를 손으로 prescribe해서 차등자전과 MC를 얻는다. 결과적으로 예측된 MC 프로파일은 표면에서 15 m/s 극방향, $r/R_\odot\sim 0.77$에서 영점 통과, 깊은 곳에서 약한 적도방향 — **단일 셀** 구조.

**English:**
Gilman & Glatzmaier (1981) onwards. Fig. 5: at $2\Omega_0\to 0.75\Omega_0$, simulation transitions from multi-cell MC + solar-like rotation (fast rotators, $Ro\ll1$) to single-cell MC + anti-solar rotation (slow rotators, $Ro\gtrsim1$).

Mean-field approaches (Kitchatinov & Rüdiger 1993; Kitchatinov & Olemskoy 2011): prescribe anisotropic thermal flux + Reynolds stress → single-cell MC with 15 m/s surface, zero crossing ~0.77 $R_\odot$.

### Part VI: Interior Structure of MC — Helioseismology / 내부 구조 — 헬리오지스몰로지 (Section 6)

**한국어:**
태양 내부는 광학적으로 두꺼워 직접 관측 불가. 헬리오지스몰로지만이 유일한 수단이다.

- **Global-mode oscillations**: MC는 축대칭(m=0)이고 적도에 대해 반대칭이므로, 주파수(frequency)는 MC에 거의 불감(insensitive). Leading order에서는 asymmetric 흐름에만 반응.
- 따라서 MC inference를 위해서는 **local techniques**가 필수:
  - **Time-distance (TD)** (Duvall et al. 1993)
  - **Ring-diagram analysis (RDA)** (Hill 1988)
  - **Fourier-Legendre** (Braun & Fan 1998)
  - **Normal-mode coupling** (Woodard 1989)

**Time-distance 핵심**: 표면 두 지점 사이의 cross-correlation이 파동의 왕복시간을 준다. 흐름이 있으면 순방향/역방향 파의 travel-time이 비대칭 → $\delta\tau$ signature:
$$
\delta\tau = -2\int \hat{\mathbf{n}}\cdot\mathbf{v}/c^2\, ds
$$
(ray 근사) 여기서 $\hat{\mathbf{n}}$은 ray 방향. MC 측정을 위해선 남북(meridional) 방향 two-point correlation이 핵심.

**Data sources**: TON (1995–), GONG (1995–), MDI/SOHO (1996–2011), HMI/SDO (2010–).

**Ring-Diagram Analysis**: HMI는 16 Mpixel 카메라, 이 중 ~10 Mpixel이 유용. RDA의 패치 크기 $15°\times15°$, 깊이 해상도 ~20 Mm까지.

**Normal-mode coupling** (Woodard 2014; Hanasoge et al. 2017): eigenfunctions의 왜곡을 측정. Line-of-sight projection 오차 모델링 가능, 모든 seismic mode 정보를 spectral domain에서 활용 — 전 깊이 접근 가능.

**English:**
Solar interior is optically thick; helioseismology is the only probe. **Global modes** are essentially insensitive to MC (axisymmetric, anti-symmetric) at leading order; MC requires **local techniques**: TD (Duvall et al. 1993), RDA (Hill 1988), Fourier-Legendre (Braun & Fan 1998), normal-mode coupling (Woodard 1989; Hanasoge et al. 2017).

Time-distance: cross-correlations of two-point surface wavefields → travel times asymmetric in presence of flow, $\delta\tau = -2\int\hat{\mathbf{n}}\cdot\mathbf{v}/c^2\,ds$ (ray approximation).

Data: GONG (1995–), MDI/SOHO (1996–2011), HMI/SDO (2010–).

### Part VII: Systematical Errors / 체계적 오차 (Section 7)

**한국어:**
저자는 MC inference의 신뢰성을 깎아먹는 4대 체계 오차를 정리한다.

1. **Centre-to-limb bias (§7.1)**: 가장 크고 가장 nasty한 체계 오차. 디스크 중심에서 림까지 travel-time이 해석 불가능하게 변함. Duvall & Hanasoge (2009)가 공식적으로 제기. 반지름 속도인 $R_\odot/c\approx 2.3\,\text{s}$ 지연이 빛이 림에서 중심까지 이동하는 동안 발생. Zhao et al.(2012): 동서 systematic을 남북 측정에서 빼서 보정. Chen & Zhao (2018): 모드 주파수 의존성, 4 mHz 부근에서 부호 전환.
2. **$B_0$ angle (§7.2)**: 지구 공전궤도가 태양 적도면에 7.25° 기울어짐. 관측 위도가 연주기적으로 변동. 연간 epoch를 3개월씩 나눠 MC가 일정하면 $B_0$ 보정이 제대로 된 것. Zaatri et al.(2006): 고위도 60° 적도방향 흐름이 $B_0$ artifact.
3. **$P$ angle & upside-down (§7.3)**: 태양 자전축에 대한 이미지 회전 오차. MDI는 0.2° 회전 → 자전 속도의 일부가 MC signal에 섞임 (Liang et al. 2017). MDI 카메라는 2003년 이후 3개월마다 상하 뒤집힘 (antenna 고장). Upside-down 데이터의 N-S travel time 신호가 upright와 다름 — 체계적.
4. **Methodological systematics (§7.4)**: Giles (2000) — TD와 global-mode inferences의 차등자전 비교에서 10–20 m/s 편차 존재, 특히 표면층. Jackiewicz et al.(2015)도 유사한 문제 보고. Deep layer ($r/R_\odot<0.8$)에서 Gizon et al.(2020) 에러바 3–4 m/s는 너무 낙관적일 수 있음.

**영어:**

1. **Centre-to-limb bias**: the single most important systematic. Duvall & Hanasoge (2009) first discussed formally — 2.3 s intrinsic lag from limb to centre (distance $R_\odot/c$). Compounded by radiative-transfer asymmetries and height-of-observation changes (Baldner & Schou 2012). Correction: Zhao et al. (2012, 2013) subtract E-W signal from N-S.
2. **$B_0$ angle**: 7.25° ecliptic inclination → annual viewing-angle variation. Zaatri et al. (2006): equatorward flow at >60° is $B_0$ artifact.
3. **$P$ angle & upside-down**: MDI image rotation (0.2°) and periodic flips (antenna failure, post-2003) introduce artifacts.
4. **Methodological systematics**: Giles (2000) differential-rotation comparison shows 10–20 m/s biases, most pronounced in upper 10%; Gizon et al. (2020) tiny deep-interior error bars may be optimistic.

### Part VIII: Active Region Inflows and Solar-Cycle Evolution / 활동영역 유입 흐름 및 태양주기 변화 (Section 8)

**한국어:**
활동영역 주변에 수렴 흐름(inflows)이 존재 (Gizon et al. 2001; Zhao & Kosovichev 2004). 기원 논쟁: 압력 강하(cooling)인가 자기장 집중(convective cells)인가 (Yoshimura 1971).

Fig. 12 (Hathaway & Rightmire 2010): 1995–2010 표면 MC 진폭 $V$ vs. sunspot number. **반상관(anticorrelation)**: MC 진폭이 sunspot minima에서 크고 maxima에서 작음 → ~2 m/s 변동.

Fig. 13 (Komm et al. 2018): 18년 RDA 데이터 — 7.1 Mm 깊이에서 MC 진폭 변화. 자기주기와 강한 상관. 이로써 cycle-dependent MC가 flux-transport dynamo의 자기 되먹임(feedback) 경로 일부임이 시사된다.

**English:**
Active-region inflows (Gizon et al. 2001; Zhao & Kosovichev 2004), attributed to cooling or magnetic-concentration convective cells (Yoshimura 1971).

Fig. 12: surface MC amplitude anticorrelates with sunspot number (Hathaway & Rightmire 2010), ~2 m/s variation.
Fig. 13: 18-yr RDA at 7.1 Mm shows MC strongly cycle-correlated (Komm et al. 2018). Supports cycle feedback in flux-transport dynamo.

### Part IX: Inversions for Interior MC / 내부 MC 역산 (Section 9)

**한국어:**
역문제 모델 (Eq. 5):
$$ d_i = \int_\odot d\mathbf{x}\,K_i(\mathbf{x})\,\psi(\mathbf{x})+\epsilon_i $$
- $d_i$: travel-time 측정값, $\epsilon_i$: realization noise
- $K_i$: sensitivity kernel
- $\psi$: stream function, $\mathbf{v}=\nabla\times(\psi\mathbf{e}_\phi)$ — 자동 질량 보존

Fig. 14 (p. 29): 10°–25° 위도 평균 poleward velocity를 깊이 $r/R_\odot$ 의 함수로 보이는 **4개 그룹 비교**:
- Chen & Zhao (HMI, 2010–2017): cyan, 얕은 depth에서 poleward, 0.9 근처에서 양→음 전환 가능성 (two-cell)
- Jackiewicz et al. (GONG, 2004–2012): 빨강 circles, noisy, multiple sign changes
- Rajaguru & Antia (HMI, 2010–2014): 파랑, **single-cell**
- Gizon et al. (GONG, 1996–2008): 검정 error bars, 넓은 에러

Fig. 15 (p. 30): 4그룹의 2D 반구 inversion — 진폭과 구조가 시각적으로 크게 다름. 원인: 데이터, 분석 방법, centre-to-limb 보정 방식, 역산 정규화의 차이.

Fig. 16 (Rajaguru & Antia 2020, p. 31): **모드 주파수 의존성!** 3.0 mHz 모드 사용: one-cell. 4.0 mHz 모드 사용: two-cell in radius. 즉 Chen & Zhao (2017)의 two-cell 결과는 **centre-to-limb effect의 모드 주파수 의존성의 artifact**일 가능성.

**Cross-equator flow (§9.1)**: $P$ 각 보정이 핵심. Gizon et al. (2020): 적도 교차 흐름 매우 약함. González Hernández et al. (2008): 얕은 표면층에서만 notable한 cross-equator 흐름.

**결론 (Section 10)**: 
- 표면 단일 셀은 확립됨.
- 0.9 $R_\odot$ 정도까지 단일 셀 poleward가 지배적.
- 이하 deep region의 return flow 깊이와 셀 수는 **미해결**.
- 향후 방향: normal-mode coupling + 장기 데이터 + centre-to-limb의 mode-by-mode 처리.

**English:**
Inverse problem (Eq. 5); stream-function formulation enforces mass conservation automatically.

Figs. 14–16 show inversion disagreements among Chen & Zhao, Jackiewicz et al., Rajaguru & Antia, Gizon et al. Rajaguru & Antia (2020, Fig. 16) show that **mode-frequency selection changes single→double cell**, implicating centre-to-limb residuals. Gizon et al. (2020) use all HMI+MDI+GONG, correct all four systematics, and find broad agreement between MDI and GONG — single-cell MC with possibly weak secondary southern-hemisphere cell.

**Conclusion (§10)**: surface single-cell established; single-cell poleward supported down to ~0.9 $R_\odot$; deep-interior return flow and cell count remain open; normal-mode coupling is the most promising future direction.

---

## 3. Key Takeaways / 핵심 시사점

1. **MC는 작지만 결정적이다 / MC is small but pivotal** — 표면 ~20 m/s (차등자전의 1/15), 하지만 각운동량 수송과 flux-transport dynamo의 사이클 주기를 결정. Despite ~1% of rotation, MC governs angular-momentum balance and the 11-yr cycle pacemaker via flux transport.

2. **표면 기법들도 완전히 일치하지 않는다 / Even surface techniques disagree** — Sunspot, 자기요소, 도플러, supergranule, granulation, RDA 6종이 poleward direction은 일치하지만 amplitude는 5–20 m/s로 산재. Six surface techniques agree on direction but scatter by factor ~4 in amplitude, each with distinct systematics.

3. **Gyroscopic pumping이 MC 방향을 결정한다 / Gyroscopic pumping sets MC direction** — Reynolds stress 발산의 부호가 MC가 회전축에서 멀어지는지 가까워지는지 결정. Surface 대류가 angular momentum을 내부로 수송 → upper-CZ에서 poleward MC. Sign of Reynolds-stress divergence determines whether MC moves toward or away from rotation axis.

4. **Centre-to-limb bias가 최대의 적 / Centre-to-limb is the dominant systematic** — 2.3 s intrinsic lag + radiative-transfer + height-of-observation 비대칭 복합 효과. 보정 없이는 km/s MC inferences 산출. Dominant systematic; corrections by Zhao et al. (2012, 2013) now standard.

5. **단일 셀 vs. 다중 셀 논쟁은 모드 주파수에 민감 / Single vs. multi-cell depends on mode frequency** — Rajaguru & Antia (2020): 4 mHz 이상 포함 시 two-cell, 그 미만만 포함 시 one-cell. Centre-to-limb의 주파수 의존성 때문. Rajaguru & Antia (2020) show mode-frequency dependence switches results, pointing to centre-to-limb residuals as the origin of two-cell claims.

6. **Flux-transport dynamo는 MC가 deep에서 equatorward면 작동 / Flux-transport dynamo requires equatorward deep MC** — Hazra et al.(2014)로 확인. Cameron & Schüssler(2017): toroidal flux 저장 영역의 평균 MC가 주기 결정. Works if MC is directed equatorward at base of CZ; what matters is average MC where toroidal flux is stored.

7. **수치 시뮬레이션은 태양과 다른 MC 패턴을 예측 / Simulations predict different MC than the Sun** — Fig. 5: 낮은 $Ro$에서 다중 셀, 높은 $Ro$에서 단일 셀 + anti-solar 회전. 태양은 둘의 경계. Low-$Ro$ simulations: multi-cell + solar-like rotation; high-$Ro$: single-cell + anti-solar. Sun sits at the transition — part of convective conundrum.

8. **심부 return flow 깊이는 미해결 / Deep return-flow depth remains open** — 현재 증거는 $0.9\,R_\odot$까지 단일 셀 poleward를 지지. 하지만 $r/R_\odot < 0.8$에서 SNR 급락, 에러바 3–4 m/s (Gizon et al. 2020)는 너무 낙관적 가능성. Current evidence supports single-cell to $0.9 R_\odot$ but deep return point unresolved; Gizon et al. (2020) error bars may underestimate true uncertainty.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Axisymmetric Meridional Flow / 축대칭 자오면 흐름

자오면 속도장 / Meridional velocity field:
$$
\mathbf{v}_m(r,\theta)=v_r(r,\theta)\hat{\mathbf{e}}_r + v_\theta(r,\theta)\hat{\mathbf{e}}_\theta
$$

**질량 보존 / Mass conservation** (anelastic):
$$
\nabla\cdot(\rho\mathbf{v}_m)=\frac{1}{r^2}\frac{\partial}{\partial r}(r^2\rho v_r)+\frac{1}{r\sin\theta}\frac{\partial}{\partial\theta}(\rho v_\theta \sin\theta)=0
$$

**Stream function formulation / 스트림 함수 공식**:
$$
\mathbf{v}=\nabla\times(\psi(r,\theta)\hat{\mathbf{e}}_\phi)
$$
$$
v_r=\frac{1}{\rho r\sin\theta}\frac{\partial(\psi\sin\theta)}{\partial\theta},\quad v_\theta=-\frac{1}{\rho r}\frac{\partial(r\psi)}{\partial r}
$$

### 4.2 Return Flow Speed from Mass Conservation / 질량 보존을 통한 귀환 흐름 속도

표면 poleward 흐름이 $v_s=20$ m/s이고, return flow가 깊이 $d$에서 일어난다면, 얇은 shell에 대한 질량 유량 연속성은:
$$
\rho_s v_s A_s \approx \rho_r v_r A_r
$$
$A_s, A_r$은 shell 단면적. 대류층 base ($r\approx 0.7 R_\odot$)에서 $\rho_r/\rho_s\sim 10^3$이므로 $v_r\sim 20/1000 = 0.02$ m/s로 비현실적으로 작음. 실제 return flow는 **deep하지 않은 얕은 depth**에서 발생하므로 $\rho_r/\rho_s\sim 5$–10, $v_r\sim 2$–5 m/s로 추정됨.

### 4.3 Travel-Time Perturbation / 왕복시간 섭동

Ray 근사:
$$
\delta\tau(\mathbf{r}_1,\mathbf{r}_2) = -2\int_\Gamma \frac{\hat{\mathbf{n}}\cdot\mathbf{v}(\mathbf{x})}{c^2(\mathbf{x})}\,ds
$$
$\Gamma$: unperturbed ray path, $\hat{\mathbf{n}}$: ray unit tangent.

MC 남북 기하:
$$
\delta\tau_{NS} = \tau(\text{north→south})-\tau(\text{south→north}) = \frac{4}{c^2}\int \hat{\mathbf{n}}_\theta\cdot\mathbf{v}_m\,ds
$$

### 4.4 Sensitivity Kernel / 민감도 커널

Born 근사 (Birch & Kosovichev 2001):
$$
\delta\tau_i = \int_\odot d\mathbf{x}\,\mathbf{K}_i(\mathbf{x})\cdot\mathbf{v}(\mathbf{x})+\epsilon_i
$$
Kernels are finite-width, smooth, and account for wave diffraction/scattering at finite frequency.

### 4.5 Ring-Diagram Analysis / 링 다이어그램 분석

$15°\times15°$ patch 내 고차 $p$-모드 dispersion:
$$
\omega_{obs}(\mathbf{k})=\omega_0(k)+\mathbf{k}\cdot\mathbf{U}_{patch}
$$
$\omega_0$: zero-flow eigenfrequency, $\mathbf{U}_{patch}$: patch-averaged horizontal flow. Ring fit: 3D power spectrum $P(\mathbf{k},\omega)$의 ring center가 $\mathbf{U}$.

### 4.6 Angular Momentum Conservation (Eq. 1)
$$
\nabla\cdot\left(\rho r\sin\theta\langle u'_\phi \mathbf{u}'_m\rangle+\rho\mathbf{u}_m r^2\sin^2\theta\,\Omega+\rho\nu r^2\sin^2\theta\,\nabla\Omega-r\sin\theta B_\phi\mathbf{B}_m-r\sin\theta\langle B'_\phi\mathbf{B}'_m\rangle\right)=0
$$

Anelastic, weak-$\mathbf{B}$ limit:
$$
\nabla\cdot(\rho\mathbf{u}_m\,\mathcal{L}) = -\nabla\cdot(\rho r\sin\theta\langle u'_\phi\mathbf{u}'_m\rangle),\quad \mathcal{L}=r^2\sin^2\theta\,\Omega
$$

### 4.7 Thermal-Wind Balance (Eq. 3)
$$
\Omega_0\frac{\partial\Omega}{\partial z}=\frac{g}{2C_p r}\frac{\partial\langle S\rangle}{\partial\theta}+\mathcal{F},\quad z=r\cos\theta
$$
with forcing
$$
\mathcal{F}=\hat{\boldsymbol\phi}\cdot\langle\nabla\times[(\nabla\times\mathbf{u})\times\mathbf{u}+(4\pi\rho)^{-1}(\nabla\times\mathbf{B})\times\mathbf{B}]\rangle
$$

### 4.8 Inversion Model (Eq. 5)
$$
d_i=\int_\odot d\mathbf{x}\,K_i(\mathbf{x})\psi(\mathbf{x})+\epsilon_i
$$
Regularized solution:
$$
\hat\psi=\arg\min_\psi\left\{\sum_i (d_i-\int K_i\psi)^2/\sigma_i^2 + \lambda\|\mathcal{R}\psi\|^2\right\}
$$
$\mathcal{R}$: smoothness operator (e.g., B-spline), $\lambda$: regularization.

### 4.9 Worked Numerical Example / 수치 예시

표면 poleward 흐름 $v_s=20$ m/s, $r=R_\odot=6.96\times 10^8$ m, 위도 30°에서 극까지의 평균 이동 시간:
$$
\tau_{30\to 90} = \frac{R_\odot (\pi/2-\pi/6)}{v_s}=\frac{R_\odot\cdot\pi/3}{20}\approx \frac{7.3\times 10^8}{20}\approx 3.6\times 10^7\,\text{s}\approx 1.15\,\text{yr}
$$

적도→극→적도 왕복:
$$
\tau_{\text{round}}\sim 11\,\text{yr}\;(\text{for a closed loop with a slow return branch, as in flux-transport dynamo})
$$
이 11년이 태양 주기와 일치한다는 것이 **flux-transport dynamo 핵심 통찰**.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1925 ─── Eddington-Sweet radiative MC (irrelevant, ~t_star)
1941 ─── Lebedinsky: turbulence & rotation coupling
1959 ─── Kippenhahn anisotropic viscosity paper
1961 ─── Babcock solar-cycle topology model
1963 ─── Kippenhahn DR + MC prediction
1974 ─── Durney: MC role in global angular momentum
1978 ─── Duvall: first modern Doppler MC (~20 m/s)
1988 ─── Hill ring-diagram analysis
1991 ─── Wang, Nash, Sheeley flux-transport dynamo
1993 ─── Duvall et al. time-distance helioseismology
1995 ─── Choudhuri et al. flux-transport with deep MC return
                GONG operational
1996 ─── MDI/SOHO launched
1997 ─── Giles et al. first TD interior MC inversion
                Kosovichev & Schou torsional oscillations
2000 ─── Giles PhD thesis: TD large-scale flows
2001 ─── Birch & Kosovichev Born kernels for TD
2005 ─── Miesch Living Reviews on solar dynamics
2008 ─── Gizon & Rempel cycle-correlated MC variation
2009 ─── Duvall & Hanasoge centre-to-limb systematic
2010 ─── HMI/SDO launched
                Hathaway & Rightmire magnetic-element MC to 75°
2012 ─── Zhao et al. centre-to-limb correction method
2013 ─── Zhao et al. first two-cell inversion
2014 ─── Jiang et al. technique comparison review
                Hazra et al. flux-transport with observed MC
2015 ─── Rajaguru & Antia one-cell HMI inversion
                Featherstone & Miesch simulation Ro study
2017 ─── Chen & Zhao two-cell finding
                Cameron & Schüssler MC role clarification
2018 ─── Rincon & Rieutord supergranulation review
2020 ─── Gizon et al. MDI+GONG+HMI single-cell inversion
                Rajaguru & Antia show mode-freq dependence of cells
2021 ─── Braun et al. Fourier-Legendre HMI analysis
2022 ─── Hanasoge Living Reviews (THIS PAPER)
         └── Synthesis: single-cell to 0.9 R_sun; deep remains open
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Paper #2 Stix (2002) "The Sun" textbook | 태양 구조 및 helioseismology 기초 / Provides helioseismology and solar structure foundations | 이 리뷰의 배경 / Foundational textbook for this review |
| Paper #15 (helioseismology review) | Time-distance와 global-mode 기법의 종합 / Time-distance and global-mode method synthesis | Section 6의 직접적 기반 / Direct basis for Section 6 methodology |
| Paper #20 (solar dynamo/cycle paper) | Flux-transport dynamo 모델과 MC의 역할 / Flux-transport dynamo and MC's role | Section 4의 이론적 배경 / Theoretical background for Section 4 |
| Charbonneau (2020) Living Rev Dynamo Models | 현대 dynamo 이론 종합 / Modern dynamo theory synthesis | MC가 어디에 들어맞는지 맥락 / Context for where MC fits |
| Howe (2009) Living Rev Solar Rotation | 차등자전의 체계적 리뷰 / Systematic review of differential rotation | 자전과 MC의 짝 / The rotation side of the rotation-MC pair |
| Miesch (2005) Living Rev Large-scale Dynamics | Mean-field theory 종합 / Mean-field theory synthesis | Eq. 1 각운동량 수송의 이론 전거 / Theoretical source for Eq. 1 angular momentum |
| Gizon & Birch (2005) Living Rev Local Helioseismology | Time-distance, RDA 기초 / TD, RDA methodological foundation | Section 6의 기법 / Techniques in Section 6 |
| Rincon & Rieutord (2018) Living Rev Supergranulation | Supergranule 파동 분산 / Supergranule wave dispersion | Section 2.4의 직접 참조 / Directly referenced in Section 2.4 |

---

## 7. References / 참고문헌

- Hanasoge, S. M., "Surface and interior meridional circulation in the Sun", Living Reviews in Solar Physics, 19:3, 2022. https://doi.org/10.1007/s41116-022-00034-7
- Babcock, H. W., "The topology of the Sun's magnetic field and the 22-year cycle", Astrophys. J., 133, 572, 1961.
- Cameron, R. H. & Schüssler, M., "An update of Leighton's solar dynamo model", Astron. Astrophys., 599, A52, 2017.
- Chen, R. & Zhao, J., "A comprehensive method to measure solar meridional circulation and the centre-to-limb effect using time-distance helioseismology", Astrophys. J., 849, 144, 2017.
- Choudhuri, A. R., Schüssler, M. & Dikpati, M., "The solar dynamo with meridional circulation", Astron. Astrophys., 303, L29, 1995.
- Duvall, T. L., "A study of large-scale solar magnetic and velocity fields", PhD thesis, Stanford Univ., 1978.
- Duvall, T. L., Jefferies, S. M., Harvey, J. W. & Pomerantz, M. A., "Time-distance helioseismology", Nature, 362, 430, 1993.
- Duvall, T. L. & Hanasoge, S. M., "Measuring meridional circulation in the Sun", arXiv:0905.3132, 2009.
- Featherstone, N. A. & Miesch, M. S., "Meridional circulation in solar and stellar convection zones", Astrophys. J., 804, 67, 2015.
- Giles, P. M., Duvall, T. L. & Scherrer, P. H. "A flow of material from the Sun's equator to its poles", Nature, 390, 52, 1997.
- Gizon, L. et al., "Meridional flow in the Sun's convection zone is a single cell in each hemisphere", Science, 368, 1469, 2020.
- Hathaway, D. H. & Rightmire, L., "Variations in the Sun's meridional flow over a solar cycle", Science, 327, 1350, 2010.
- Hazra, G., Karak, B. B. & Choudhuri, A. R., "Is a deep one-cell meridional circulation essential for the flux-transport solar dynamo?", Astrophys. J., 782, 93, 2014.
- Hill, F., "Rings and trumpets — three-dimensional power spectra of solar oscillations", Astrophys. J., 333, 996, 1988.
- Kippenhahn, R., "Differential rotation in stars with convective envelopes", Astrophys. J., 137, 664, 1963.
- Kitchatinov, L. L. & Olemskoy, S. V., "Differential rotation of main-sequence dwarfs", Mon. Not. R. Astron. Soc., 411, 1059, 2011.
- Rajaguru, S. P. & Antia, H. M., "Meridional circulation in the solar convection zone: time-distance helioseismic inferences from four years of HMI/SDO observations", Astrophys. J., 813, 114, 2015.
- Wang, Y.-M., Nash, A. G. & Sheeley, N. R., "Magnetic flux transport on the Sun", Science, 245, 712, 1989.
- Zhao, J. et al., "Detection of equatorward meridional flow and evidence of double-cell meridional circulation inside the Sun", Astrophys. J. Lett., 774, L29, 2013.
- Charbonneau, P., "Dynamo models of the solar cycle", Living Rev. Sol. Phys., 17:4, 2020.
- Miesch, M. S., "Large-scale dynamics of the convection zone and tachocline", Living Rev. Sol. Phys., 2:1, 2005.
- Gizon, L. & Birch, A. C., "Local helioseismology", Living Rev. Sol. Phys., 2:6, 2005.
