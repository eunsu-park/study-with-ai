---
title: "Prominence Oscillations"
authors: [Iñigo Arregui, Ramón Oliver, José Luis Ballester]
year: 2018
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-018-0012-6"
topic: Living_Reviews_in_Solar_Physics
tags: [prominences, filaments, oscillations, MHD_waves, seismology, resonant_absorption, Bayesian_inference, kink_modes, pendulum_model]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 57. Prominence Oscillations / 프로미넌스 진동

---

## 1. Core Contribution / 핵심 기여

**English**: This 154-page *Living Reviews* article by Arregui, Oliver, and Ballester (2018) is a **major revision and expansion** of earlier reviews on solar prominence oscillations (Oliver & Ballester 2002; Arregui et al. 2012). Prominences are cool (~8000 K), dense (n_H ~ 10¹⁰–10¹¹ cm⁻³) clouds of chromospheric plasma suspended in the hot corona (~10⁶ K) by magnetic fields of order 5–30 G. Observations reveal a rich zoology of oscillatory motions. The review systematically classifies them by velocity amplitude into **Large Amplitude Oscillations (LAOs; v > 20 km/s, flare/EUV-wave/jet triggered, affecting the full prominence)** and **Small Amplitude Oscillations (SAOs; v < 3 km/s, flare-unrelated, local)**, documenting for each: observational periods (ranging from ~1 min to several hours), damping times (τ/P ~ 1–10), wavelengths (~3000–75000 km), phase speeds (~5–200 km/s), and polarisations. It then develops the **theoretical framework**: linear ideal-MHD models from simple loaded strings through slab prominences (Joarder & Roberts 1992a,b; Oliver et al. 1992, 1993) to cylindrical flux-tube threads, deriving dispersion relations for the three magnetoacoustic modes (fast, Alfvén, slow) and their spatial structure (internal, external, hybrid modes). Damping mechanisms — non-adiabatic thermal processes (effective for slow modes), ion-neutral collisions (partial ionisation), resonant absorption in the Alfvén continuum (the leading mechanism for kink/fast-mode damping), and wave leakage — are compared against observed τ/P. The climax is **prominence seismology**: using the observed period, damping time, wavelength, and phase speed together with these MHD models to infer otherwise inaccessible plasma and magnetic quantities. For example, standing kink mode interpretation yields B ~ 5–30 G; longitudinal pendulum oscillations (Luna & Karpen 2012) yield dip curvature radii R ~ 40–130 Mm; resonantly-damped thread oscillations yield thread Alfvén speeds v_A ~ 100–150 km/s and transverse inhomogeneity scales l/a ~ 0.2. The review highlights the nascent application of **Bayesian inference** (Arregui et al. 2013, 2014, 2015) for rigorous probabilistic inversion and model comparison.

**한국어**: 154페이지 분량의 *Living Reviews* 논문으로, Arregui, Oliver, Ballester(2018)는 이전 리뷰(Oliver & Ballester 2002; Arregui et al. 2012)를 **대폭 개정·확장**하여 태양 프로미넌스(filament) 진동 연구 전반을 종합 정리한다. 프로미넌스는 뜨거운 코로나(~10⁶ K) 속에 차갑고(~8000 K) 밀도 높은(n_H ~ 10¹⁰–10¹¹ cm⁻³) 채층 기원 플라즈마가 ~5–30 G 자기장에 의해 떠 있는 구조이다. 관측은 다양한 진동 현상을 드러내며, 리뷰는 속도 진폭에 따라 **대진폭 진동(LAO; v > 20 km/s, 플레어·EUV 파·제트 트리거, 프로미넌스 전체 참여)**과 **소진폭 진동(SAO; v < 3 km/s, 플레어 무관, 국소)**로 분류한다. 각 범주별로 관측된 주기(~1 분부터 수 시간까지), 감쇠시간(τ/P ~ 1–10), 파장(~3000–75000 km), 위상속도(~5–200 km/s), 편광을 체계화하고, **이론 프레임**으로 loaded string → slab → 원통 flux tube thread에 이르는 선형 이상 MHD 모델의 분산관계와 공간 구조(internal, external, hybrid 모드)를 전개한다. 감쇠 메커니즘으로는 비단열 열 과정(저속 모드에 효과적), 이온-중성자 충돌(부분 이온화), Alfvén 연속체 **공명흡수**(kink/fast 모드 감쇠 주범), 파동 누출을 비교한다. 클라이맥스는 **프로미넌스 지진학**이다: 관측된 주기·감쇠·파장·위상속도와 MHD 모델을 결합하여 직접 측정 곤란한 물리량을 역산한다. Standing kink 모드 해석으로 B ~ 5–30 G, 종방향 pendulum 진동(Luna & Karpen 2012)으로 dip 곡률반경 R ~ 40–130 Mm, 공명감쇠 thread 진동으로 v_A ~ 100–150 km/s, l/a ~ 0.2 등을 추정한다. **Bayesian 추론**(Arregui et al. 2013, 2014, 2015) 기반의 확률적 역산·모델 비교가 최신 성과로 강조된다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Prominences and Classification / 프로미넌스와 진동 분류 (Sects. 1–2)

**English (Sect. 1, pp. 3–5)**: Quiescent filaments are cool dense clouds along polarity inversion lines, composed of many thin horizontal **threads** (width ~0.3″ ≈ 210 km, length 5″–40″ ≈ 3500–28000 km; Lin et al. 2005, DOT/SST) that are approximately two orders of magnitude denser and cooler than the surroundings. Alternative view (Heinzel & Anzer 2006): threads are a projection effect of small magnetic dips filled with cool plasma, aligned vertically. Prominences host vigorous dynamics: flows 2–35 km/s in H-alpha, up to 200 km/s in EUV lines, counter-streaming flows along field lines. The typical Alfvén speed is v_A ~ 100 km/s and sound speed c_s ~ 11 km/s. Seismology is introduced as the discipline that compares observed oscillations with theoretical models to diagnose physical conditions.

**한국어 (Sect. 1, pp. 3–5)**: 정적 필라멘트(quiescent filament)는 편극성 역전선(polarity inversion line)을 따라 형성되는 차갑고 밀도 높은 구조로, SST/DOT 관측(Lin et al. 2005)에서 폭 ~0.3″ (~210 km), 길이 5″–40″ (~3500–28000 km)의 수많은 수평 **thread**가 관측된다. 내부는 주변보다 밀도 ~100배, 온도 ~1/100 수준이다. 대안적 관점(Heinzel & Anzer 2006)은 thread가 작은 자기장선 dip 다발의 투영 효과라 본다. 프로미넌스는 H-alpha 관측에서 2–35 km/s, EUV에서 최대 200 km/s의 흐름과 자기장선 따라 역방향 counter-streaming을 보인다. 전형적 Alfvén 속도 v_A ~ 100 km/s, 음속 c_s ~ 11 km/s. **지진학(seismology)**은 관측 진동을 이론 모델과 비교하여 물리 조건을 추정하는 기법으로 도입된다.

**English (Sect. 2, p. 5)**: Oscillations are divided into two categories by velocity amplitude. LAOs usually have v > 10–20 km/s and affect the whole prominence or a large fraction; they are **associated with energetic events** (flares, EUV waves, shocks, jets, subflares). SAOs have v < 10 km/s (typically 0.1–3 km/s), are local, often persistent, and lack any obvious flare trigger. Despite intermediate-amplitude cases, these two represent physically distinct phenomena.

**한국어 (Sect. 2, p. 5)**: 진동은 속도 진폭으로 두 범주로 나눈다. LAO는 일반적으로 v > 10–20 km/s, 프로미넌스 전체 또는 큰 부분 참여, **에너지 현상(플레어, EUV파, 충격파, 제트, 서브플레어)**과 연관. SAO는 v < 10 km/s (일반적으로 0.1–3 km/s), 국소, 지속적, 명확한 플레어 트리거 부재. 중간 진폭 사례도 있지만 두 범주는 물리적으로 구분된다.

### Part II: Large Amplitude Oscillations — Observations / 대진폭 진동 관측 (Sect. 3)

**English**: LAOs come in three polarisations. (a) **Vertical oscillations** (Sect. 3.1) are winking filaments: Ramsey & Smith (1966), Hyder (1966) studied 11 winking events with periods 6–40 min, damping times 7–120 min. Shen et al. (2014a) showed chain excitation by a single EUV dome wave (X2.1 flare). Note Fig. 1: in four filaments triggered by the same wave, only F1–F4 oscillated while F5–F8 remained still. (b) **Transverse horizontal** (Sect. 3.2): Kleczek & Kuperus (1969) proposed the alternative interpretation. Hershaw et al. (2011, SOHO/EIT 2005-07-30): two wave trains from two flares drove oscillations lasting 18 h with P ~ 100 min, velocity up to 50 km/s (see Fig. 2). Liu et al. (2012): P = 28 min, τ = 120 min, initial v = 9 km/s with quasi-periodic EUV wave trains from a flux rope cavity. Gosain & Foullon (2012): P = 28 min, τ = 44 min, v = 11 km/s. Xue et al. (2014): P = 19 min, τ = 99 min, v = 2.17 Mm — fundamental standing kink wave. (c) **Longitudinal** (Sect. 3.3): first reported by Jing et al. (2003, 2006). Luna et al. (2014) analysed a filament (36 slits, periods 0.7–0.86 h), fitted with exponentially damped cosine and decaying Bessel function. Bi et al. (2014): P increased from 67–71 min to 80–94 min due to weakening restoring force. Zhang et al. (2017): P evolves with mass drainage.

**한국어**: LAO는 세 가지 편광으로 나뉜다. (a) **수직 진동** (Sect. 3.1): winking filament 현상. Ramsey & Smith (1966), Hyder (1966)가 11개 사례에서 P = 6–40 분, τ = 7–120 분 관측. Shen et al. (2014a)에서 단일 EUV dome wave(X2.1 플레어)가 연쇄적으로 여러 필라멘트 진동 유도 — 단, Fig. 1에서 F1–F4만 진동하고 F5–F8은 반응 없음. (b) **가로(수평) 진동** (Sect. 3.2): Kleczek & Kuperus (1969) 해석. Hershaw et al. (2011)는 SOHO/EIT 2005-07-30 관측에서 18 시간 지속된 진동(P ~ 100 분, v ~ 50 km/s; Fig. 2). Liu et al. (2012): P = 28분, τ = 120분, v = 9 km/s, flux rope cavity 내 EUV wave train. Gosain & Foullon (2012): P = 28분, τ = 44분, v = 11 km/s. Xue et al. (2014): P = 19분, τ = 99분, 진폭 2.17 Mm — 기본 standing kink 파. (c) **종방향 진동** (Sect. 3.3): Jing et al. (2003, 2006) 최초 보고. Luna et al. (2014): 36개 slit 분석, P = 0.7–0.86 h, 지수감쇠 코사인·Bessel 함수 피팅. Bi et al. (2014): P가 67–71분에서 80–94분으로 증가(복원력 약화). Zhang et al. (2017): 질량 배수로 P 변화.

### Part III: Large Amplitude Oscillations — Theory / 대진폭 진동 이론 (Sect. 4)

**English**: Theoretical models for LAOs divide by polarisation. (a) **Vertical** (Sect. 4.1): Hyder (1966) used a Jolly oscillator with viscous damping: $\ddot{r} + (\mu/M)\dot{r} + (K/M)r = 0$ [Eq. (1)], where the restoring force $K$ is magnetic tension. (b) **Transverse** (Sect. 4.2): Kleczek & Kuperus (1969) used damped harmonic oscillator with radiation damping, obtaining
$$P = 4\pi L B^{-1}\sqrt{\pi \rho_p} \quad [\text{Eq. (3)}]$$
(c) **Longitudinal** (Sect. 4.3) — the **Luna & Karpen pendulum model** (2012) is the central result. When an energetic event heats one footpoint, chromospheric evaporation pushes cold plasma along dipped field lines. The restoring force is the projected gravity in the dip, giving
$$P = 2\pi\sqrt{R/g_0} \quad [\text{Eq. (4)}], \qquad B \geq \sqrt{\frac{g_0^2 m n}{4\pi^2}}\, P \quad [\text{Eq. (5)}]$$
Numerical simulations (Luna et al. 2012a, 2016b; Zhang et al. 2012, 2013) confirm gravity-driven longitudinal oscillations with damping attributed to continuous mass accretion (Ruderman & Luna 2016). Vršnak et al. (2007) proposed an alternative: twisted flux rope with azimuthal pressure gradient giving $\ddot{X} = -(2 v_{A\phi}^2/L^2) X$ [Eq. (6)], $P = \sqrt{2}\pi L/v_{A\phi}$.

**한국어**: LAO 이론은 편광별로 나뉜다. (a) **수직**: Hyder (1966) Jolly 진동자 + 점성 감쇠, $\ddot{r}+(\mu/M)\dot{r}+(K/M)r=0$ [식 (1)]. (b) **가로**: Kleczek & Kuperus (1969) 감쇠 조화진동자 + 음파 복사 감쇠, $P = 4\pi L B^{-1}\sqrt{\pi\rho_p}$ [식 (3)]. (c) **종방향**: **Luna & Karpen 진자 모델(2012)**이 핵심. 에너지 사건이 한쪽 자기장선 footpoint를 가열 → 채층 증발 → 차가운 플라즈마가 dip을 따라 흐름. 복원력은 dip 투영 중력:
$$P = 2\pi\sqrt{R/g_0} \quad [\text{식 (4)}], \qquad B \geq \sqrt{\frac{g_0^2 m n}{4\pi^2}}\, P \quad [\text{식 (5)}]$$
수치 시뮬레이션(Luna et al. 2012a, 2016b; Zhang et al. 2012, 2013)이 중력 구동을 확인. 감쇠는 연속 질량 축적(Ruderman & Luna 2016). Vršnak et al. (2007)은 대안으로 꼬인 flux rope와 방위각 압력 구배: $\ddot{X} = -(2 v_{A\phi}^2/L^2) X$ [식 (6)], $P = \sqrt{2}\pi L/v_{A\phi}$.

### Part IV: Small Amplitude Oscillations — Observations / 소진폭 진동 관측 (Sect. 5)

**English**: SAOs are mostly studied via spectroscopy (Doppler, line intensity, line width). Detection success is modest (Harvey 1969: 41% of 68 non-active-region prominences). Simultaneous two-telescope campaigns (Balthasar et al. 1993; Zapiór et al. 2015) have identified coherent signals as genuinely solar. Spectroscopic slit measurements, two-dimensional Dopplergrams, and space data (SUMER, CDS) coexist. The Hinode/SOT high-resolution sample of Hillier et al. (2013) analysed 3436 oscillating features, fitted an attenuated sinusoid with linearly varying period
$$A_0 \exp(t/\tau) \sin\left[\frac{2\pi t}{P_0(1 + C t)} + S\right] \quad [\text{Eq. (8)}]$$
and obtained an impressive power-law correlation $A_0 = 10^{0.13}\,P_0^{0.74}$ and $V = 10^{0.96}\,P_0^{-0.25}$ (Fig. 10). Period histograms (Fig. 9) reveal periods from 50 s to 6000 s with no preferred value. Amplitudes: displacement 19–1400 km, velocity 0.2–23 km/s. Damping: τ/P ~ 1–4 (Molowny-Horas et al. 1999, Terradas et al. 2002). Wavelengths 3000–75000 km. Phase speeds 10–200 km/s. Polarisation analysis (Lin et al. 2009) suggests oscillation planes inclined 42°–59° to the sky plane; Okamoto et al. (2015) found phase differences 90°–180° between displacement and Doppler (indicative of resonant absorption).

**한국어**: SAO는 주로 분광학(Doppler 속도, 선 강도, 선폭)으로 연구된다. 검출 성공률은 보통 수준(Harvey 1969: 68개 비활성 영역 프로미넌스 중 41%). 두 망원경 동시관측(Balthasar et al. 1993; Zapiór et al. 2015)으로 일관된 신호의 태양 기원을 확인. Hinode/SOT 고해상도 샘플(Hillier et al. 2013)에서 3436개 진동 성분을 감쇠 sinusoid + 선형 변화 주기로 피팅:
$$A_0 \exp(t/\tau)\sin\left[\frac{2\pi t}{P_0(1+Ct)} + S\right]$$
거듭제곱 상관 $A_0 = 10^{0.13}\,P_0^{0.74}$, $V = 10^{0.96}\,P_0^{-0.25}$ (Fig. 10). 주기 분포(Fig. 9)는 50 초–6000 초, 뚜렷한 선호 주기 없음. 진폭: 변위 19–1400 km, 속도 0.2–23 km/s. 감쇠: τ/P ~ 1–4 (Molowny-Horas et al. 1999, Terradas et al. 2002). 파장 3000–75000 km, 위상속도 10–200 km/s. 편광(Lin et al. 2009): 진동면이 하늘면에 42°–59° 기울어짐; Okamoto et al. (2015): 변위와 Doppler 사이 90°–180° 위상차(공명흡수 지표).

### Part V: Small Amplitude Oscillations — Theory / 소진폭 진동 이론 (Sect. 6)

**English**: Theory proceeds in four levels. **(6.1) Loaded string**: treating the prominence as mass $M$ on elastic string under gravity gives
$$P = 2\pi (L \tan\theta / g)^{1/2} \quad [\text{Eq. (9)}]$$
yielding 7–24 min for $g = 274$ m/s², $2L = 50,000$ km, $\theta = 3°$–$30°$. For a finite-width prominence (Oliver et al. 1993),
$$P = 2\pi(L x_p)^{1/2}/c_{\text{pro}} \quad [\text{Eq. (12)}]$$
where $c_{\text{pro}}$ is the prominence fast ($c_f = \sqrt{v_A^2 + c_s^2}$), Alfvén ($v_A$), or cusp ($c_T$) speed. With $v_A = 28$ km/s, $c_s = 15$ km/s, $2x_p = 2L/10 = 5000$ km: $P_{\text{fast}} = 26$ min, $P_{\text{Alfvén}} = 30$ min, $P_{\text{slow}} = 63$ min. **(6.2) Slab models** (Joarder & Roberts 1992a,b; Oliver et al. 1992, 1993, 1996): dispersion relations for even/odd modes (Eqs. 17–18), solutions classified as internal, external, and hybrid modes. Skewed-field models (Joarder & Roberts 1993b) yield hybrid Alfvén/slow periods up to 60 min and 5 h. **(6.3) Fine-structure thread (propagating)**: infinitely long cylindrical thread with kink mode
$$\omega_k = k_z \sqrt{\frac{\rho_p v_{Ap}^2 + \rho_c v_{Ac}^2}{\rho_p + \rho_c}} = k_z v_{Ap}\sqrt{\frac{2\zeta}{1+\zeta}} \quad [\text{Eq. (30)}]$$
and period
$$P = \frac{\sqrt{2}}{2}\frac{\lambda}{v_{Ap}}\left(\frac{1+\zeta}{\zeta}\right)^{1/2} \quad [\text{Eq. (31)}]$$
for $\zeta = \rho_p/\rho_c$. Periods span 30 s to a few minutes for typical parameters, matching propagating wave observations (Lin et al. 2007). **(6.4) Thread (standing)**: Joarder et al. (1997), Díaz et al. (2001, 2002, 2003) showed only a few non-leaky modes are supported and kink eigenfunctions decay exponentially outside the thread in cylindrical geometry (Fig. 57). Cylindrical threads are less likely to induce neighbour oscillations than Cartesian threads. **(6.5) Numerical MHD** (Terradas et al. 2013; Luna et al. 2016a): 2-D simulations with impulsive or continuous periodic excitation. **(6.6) Radiative MHD** (Heinzel et al. 2014; Zapiór et al. 2016): couples MHD with radiative transfer for synthesis of H-alpha, H-beta line profiles — MHD modes produce distinctive Doppler, FWHM, and intensity signatures (Fig. 62), offering a route to observational mode identification.

**한국어**: 이론은 네 수준으로 전개된다. **(6.1) Loaded string**: 프로미넌스를 탄성 스트링 위 질량 $M$으로, 중력을 복원력으로 보면
$$P = 2\pi(L\tan\theta/g)^{1/2} \quad [\text{식 (9)}]$$
$g = 274$ m/s², $2L = 50,000$ km, $\theta = 3°–30°$에서 $P = 7–24$ 분. 유한 폭 프로미넌스(Oliver et al. 1993)에서는
$$P = 2\pi(L x_p)^{1/2}/c_{\text{pro}} \quad [\text{식 (12)}]$$
$c_{\text{pro}}$는 fast ($c_f=\sqrt{v_A^2+c_s^2}$), Alfvén ($v_A$), cusp ($c_T$) 속도. $v_A = 28, c_s = 15$ km/s, $2x_p = 5000$ km에서 $P_{\text{fast}} = 26$, $P_{\text{Alfvén}} = 30$, $P_{\text{slow}} = 63$ 분. **(6.2) Slab 모델** (Joarder & Roberts 1992a,b; Oliver et al. 1992, 1993, 1996): 짝/홀 모드 분산관계 [식 (17)–(18)], internal/external/hybrid 모드 분류. Skewed-field (Joarder & Roberts 1993b)에서 hybrid Alfvén/slow 모드가 60분–5시간 주기. **(6.3) Fine-structure thread (전파파)**: 무한 원통 thread의 kink 모드
$$\omega_k = k_z \sqrt{\frac{\rho_p v_{Ap}^2 + \rho_c v_{Ac}^2}{\rho_p + \rho_c}} = k_z v_{Ap}\sqrt{\frac{2\zeta}{1+\zeta}} \quad [\text{식 (30)}]$$
$$P = \frac{\sqrt{2}}{2}\frac{\lambda}{v_{Ap}}\left(\frac{1+\zeta}{\zeta}\right)^{1/2} \quad [\text{식 (31)}]$$
$\zeta = \rho_p/\rho_c$. 전형 파라미터에서 $P = 30$ 초–수 분, Lin et al. (2007) 관측과 일치. **(6.4) Thread (정상파)**: Joarder et al. (1997), Díaz et al. (2001, 2002, 2003)는 몇 개 non-leaky 모드만 지원, kink 고유함수가 원통 밖에서 지수 감쇠(Fig. 57). 원통 thread는 Cartesian보다 이웃 진동 유도 덜함. **(6.5) 수치 MHD** (Terradas et al. 2013; Luna et al. 2016a): 2D 충격·주기 강제. **(6.6) 복사 MHD** (Heinzel et al. 2014; Zapiór et al. 2016): H-alpha, H-beta 선 프로파일 합성 — MHD 모드가 고유한 Doppler, FWHM, 강도 특성을 생성(Fig. 62).

### Part VI: Damping Mechanisms / 감쇠 메커니즘 (Sect. 7)

**English**: The observed damping ratio τ_d/P ~ 1–4 is an important diagnostic. **(7.1) Non-adiabatic thermal** (Terradas et al. 2001, 2005; Soler et al. 2007, 2008, 2009a; Carbonell et al. 2004, 2009): radiative losses (Newton cooling), thermal conduction, heating. Result: slow modes efficiently damped (τ_d/P compatible with observations), **fast modes practically unaffected**. **(7.2) Ion-neutral collisions** (Forteza et al. 2007, 2008): prominence plasma is partially ionised. Collisions damp Alfvén and fast waves; gives damping but ratios don't match observations exactly. **(7.3) Resonant damping in infinite threads** (Arregui et al. 2008a,b; Soler et al. 2009b): transverse inhomogeneity across the thread boundary allows Alfvén continuum coupling. Global kink mode energy transfers to local Alfvén oscillations in the inhomogeneous layer, giving
$$\frac{\tau_d}{P} = \frac{2}{\pi}\frac{R}{l}\frac{\rho_p + \rho_c}{\rho_p - \rho_c} \quad [\text{Eq. (37)}]$$
For ρ_p/ρ_c = 200 and l/R = 0.2, τ_d/P ≈ 3.2 — **in excellent agreement with observations**. Slow mode resonant damping also exists (Soler et al. 2009a). **(7.4) Global prominence oscillations**: Terradas et al. (2016) numerical simulation of 3D density enhancement inside twisted flux rope, global kink energy transferred via resonance at PCTR (prominence-corona transition region). **(7.5) Partial ionisation + resonant damping** (Soler et al. 2009d, 2010a,b): hybrid mechanisms. **(7.6) Finite threads**: Soler et al. (2010a,b) — dense part length L_p, total tube length L; damping ratio rather insensitive to L_p/L but period strongly depends. **(7.7) Flowing threads**: Soler & Goossens (2011). **(7.8) Wave leakage**: Schutgens (1997a,b); Schutgens & Tóth (1999) — quality factor $Q_0 = \pi\tau_d/P$ varies across coronal Alfvén speed.

**한국어**: 관측된 τ_d/P ~ 1–4은 중요 진단. **(7.1) 비단열 열적** (Terradas et al. 2001, 2005; Soler et al. 2007, 2008, 2009a): 복사 손실(Newton cooling), 열전도, 가열. 결과: slow 모드만 효율적 감쇠 (τ_d/P가 관측과 일치), **fast 모드는 거의 영향 없음**. **(7.2) 이온-중성자 충돌** (Forteza et al. 2007, 2008): 부분 이온화 프로미넌스. 충돌이 Alfvén/fast 파 감쇠시키나 정확히 일치하지는 않음. **(7.3) 무한 thread 공명 감쇠** (Arregui et al. 2008a,b; Soler et al. 2009b): thread 경계의 가로 비균일성이 Alfvén 연속체 커플링 유도. Global kink 에너지가 비균일 층의 국소 Alfvén 진동으로 전달:
$$\frac{\tau_d}{P} = \frac{2}{\pi}\frac{R}{l}\frac{\rho_p + \rho_c}{\rho_p - \rho_c} \quad [\text{식 (37)}]$$
ρ_p/ρ_c = 200, l/R = 0.2에서 τ_d/P ≈ 3.2 — **관측과 뛰어난 일치**. Slow 모드 공명 감쇠(Soler et al. 2009a). **(7.4) 전역 프로미넌스 진동**: Terradas et al. (2016) 꼬인 flux rope + 3D 밀도 증강 수치 시뮬레이션, PCTR에서 공명으로 에너지 전달. **(7.5) 부분 이온화 + 공명** (Soler et al. 2009d, 2010a,b). **(7.6) 유한 thread**: Soler et al. (2010a,b) — 차가운 부분 길이 L_p, 총 길이 L; 감쇠비는 L_p/L에 둔감, 주기는 강하게 의존. **(7.7) Flowing thread**: Soler & Goossens (2011). **(7.8) 파동 누출**: Schutgens (1997a,b); Schutgens & Tóth (1999) — 품질 계수 $Q_0 = \pi\tau_d/P$.

### Part VII: Prominence Seismology / 프로미넌스 지진학 (Sect. 8)

**English (Sect. 8.1 LAO Seismology)**: Hyder (1966) fitted winking filaments with damped harmonic oscillator, obtained $B$ = 2–30 G, coronal viscosity coefficient. Isobe & Tripathi (2006): $v_A = 87$ km/s, $B = 9.8$ G for pre-eruption filament using Kleczek & Kuperus (1969) Eq. (3). Gilbert et al. (2008): $B = 30$ G; Gosain & Foullon (2012): $B = 25$ G. Vršnak et al. (2007) with twisted flux-rope: $v_{A\phi} = 100$ km/s, B = 5–15 G (azimuthal), 10–30 G (axial). Luna et al. (2014) with pendulum Eq. (4)–(5): R = 43–66 Mm, $B_{\min} = 14 \pm 8$ G. Bi et al. (2014), Li & Zhang (2012), Zhang et al. (2017) with pendulum: B = 28–55, 15, 15 G respectively. Standing kink mode interpretation (Liu et al. 2012; Xue et al. 2014) with Nakariakov & Verwichte (2005):
$$B_0 = \frac{L}{P}\sqrt{2\mu_0 \rho_0\left(1 + \frac{\rho_e}{\rho_0}\right)} \quad [\text{Eq. (44)}]$$
yields B = 17.6 G (Xue et al. 2014).

**English (Sect. 8.2 Slab Seismology)**: Régnier et al. (2001) used Joarder & Roberts (1993b) skewed-field slab, found 18° field inclination, B from frequency matching of Alfvén, slow-kink, fast-kink, slow-sausage, fast-sausage modes. Pouget et al. (2006) with 16-h CDS/SOHO data detected three filaments' six fundamental modes, inferred inclination, temperature, Alfvén speed simultaneously.

**English (Sect. 8.3–8.4 Thread Seismology)**: Lin et al. (2009) used propagating kink interpretation with $c_{ph} = c_k \approx \sqrt{2}\,v_{Ap}$ for ten swaying threads, obtaining $v_{Ap}$ distributed across threads (B = 0.9–3.5 G for assumed $\rho_p$). Damped thread oscillations (Arregui et al. 2008b; Goossens et al. 2008; Soler et al. 2010a,b): seismology yields $v_{Ap}$, $l/a$ — thin-tube-thin-boundary inversion curve (Fig. 85). For $P = 3$ min, $\tau_d = 9$ min, $\lambda = 3000$ km with density contrast $\zeta \to \infty$: $v_{Ap} \approx 12$ km/s, $l/a \approx 0.21$.

**English (Sect. 8.5 Period-ratio seismology)**: Díaz et al. (2010) — ratio $P_1/2P_2$ of fundamental to first overtone probes longitudinal density structuring. For thread length $2W$ and tube length $2L$:
$$\frac{P_1}{2P_2} = F\left(\frac{W}{L}, \frac{\rho_p}{\rho_c}\right)$$
Lin et al. (2007) observed $P_1/2P_2 = 2.22$ ($P_1 = 16$ min, $P_2 = 3.6$ min) → W/L = 0.12 → L ~ 130,000 km, $v_{Ap} \sim 160$ km/s. Soler et al. (2015): continuous density profiles (Lorentzian, Gaussian, parabolic) give different period ratios; observed ratios favour Lorentzian (most concentrated central density).

**English (Sect. 8.6 Flowing threads)**: Okamoto et al. (2007) — six flowing, oscillating threads analysed by Terradas et al. (2008): one-to-one relation between thread $v_{Ap}$ and coronal $v_{Ac}$; asymptotic thread Alfvén lower limit ~120 km/s for L = 100 Mm. Flow effects shift periods 3–5%.

**English (Sect. 8.7 Bayesian seismology)**:
$$p(\boldsymbol{\theta}|D,M) = \frac{p(D|\boldsymbol{\theta},M) p(\boldsymbol{\theta},M)}{\int p(D|\boldsymbol{\theta},M) p(\boldsymbol{\theta},M)\, d\boldsymbol{\theta}} \quad [\text{Eq. (54)}]$$
Arregui et al. (2014) inferred $v_{Ap}$ and $l/R$ from (P, τ_d) given:
$$P \sim \frac{\sqrt{2}}{2}\frac{\lambda}{v_{Ap}}, \qquad \frac{\tau_d}{P} \sim \frac{2}{\pi}\frac{R}{l} \quad [\text{Eq. (55)}]$$
Marginal posteriors for $v_{Ap}$ and $l/R$ give well-constrained solutions (Fig. 93). Soler et al. (2015, Arregui & Soler 2015): Bayesian model comparison between piecewise, Lorentzian, Gaussian, parabolic density profiles — Lorentzian most probable.

**한국어 (Sect. 8.1 LAO 지진학)**: Hyder (1966) winking filament 감쇠조화진동자 피팅, B = 2–30 G, 코로나 점성 계수. Isobe & Tripathi (2006): $v_A = 87$ km/s, B = 9.8 G. Gilbert et al. (2008): B = 30 G; Gosain & Foullon (2012): B = 25 G. Vršnak et al. (2007) 꼬인 flux rope: $v_{A\phi} = 100$ km/s, B = 5–15 G (방위각), 10–30 G (축). Luna et al. (2014) 진자식 (4)–(5): R = 43–66 Mm, $B_{\min} = 14 \pm 8$ G. Bi et al. (2014), Li & Zhang (2012), Zhang et al. (2017): B = 28–55, 15, 15 G. Standing kink 해석 (Liu et al. 2012; Xue et al. 2014) + Nakariakov & Verwichte (2005) 식 (44) → B = 17.6 G.

**한국어 (Sect. 8.2 Slab 지진학)**: Régnier et al. (2001): Joarder & Roberts (1993b) skewed-field slab, 18° 경사, Alfvén·slow-kink·fast-kink·slow-sausage·fast-sausage 모드 주파수 매칭. Pouget et al. (2006): CDS/SOHO 16시간 데이터로 세 필라멘트의 6개 기본 모드, 경사·온도·Alfvén 속도 동시 추정.

**한국어 (Sect. 8.3–8.4 Thread 지진학)**: Lin et al. (2009): $c_{ph} = c_k \approx \sqrt{2}\,v_{Ap}$로 10개 thread 분석, B = 0.9–3.5 G. 감쇠 thread 진동(Arregui et al. 2008b; Goossens et al. 2008; Soler et al. 2010a,b): $v_{Ap}$, $l/a$ 역산 (Fig. 85). $P = 3$ 분, $\tau_d = 9$ 분, $\lambda = 3000$ km에서 $v_{Ap} \approx 12$ km/s, $l/a \approx 0.21$.

**한국어 (Sect. 8.5 주기비 지진학)**: Díaz et al. (2010): $P_1/2P_2$ → 종방향 밀도 구조. Lin et al. (2007) $P_1/2P_2 = 2.22$ ($P_1 = 16, P_2 = 3.6$ 분) → W/L = 0.12 → L ~ 130,000 km, $v_{Ap} ~ 160$ km/s. Soler et al. (2015): 연속 밀도 프로파일(Lorentzian/Gaussian/parabolic) 비교, 관측은 Lorentzian 선호.

**한국어 (Sect. 8.6 Flowing thread)**: Okamoto et al. (2007) + Terradas et al. (2008): thread $v_{Ap}$ - 코로나 $v_{Ac}$ 일대일 대응, L = 100 Mm에서 $v_{Ap}$ 하한 ~120 km/s. 흐름으로 주기 3–5% 변화.

**한국어 (Sect. 8.7 Bayesian 지진학)**: Bayes 정리 [식 (54)]. Arregui et al. (2014): (P, τ_d) → $v_{Ap}$, $l/R$ [식 (55)]. Marginal posteriors로 해의 확률분포 제공(Fig. 93). Soler et al. (2015) + Arregui & Soler (2015): 밀도 프로파일 모델 비교 — Lorentzian 최대 확률.

---

## 3. Key Takeaways / 핵심 시사점

1. **LAO와 SAO는 물리적으로 구분되는 현상이다 / LAOs and SAOs are physically distinct phenomena** — LAO는 플레어·EUV파·제트 등 외부 에너지 사건으로 촉발되어 v > 20 km/s로 전체 프로미넌스가 흔들리는 반면, SAO는 v < 3 km/s로 국소적·지속적·플레어 무관이며, 아마도 채층/광구 운동이 구동하는 MHD 파동이다. / LAOs require energetic triggers and shake the whole prominence with v > 20 km/s; SAOs are local, persistent, flare-unrelated oscillations with v < 3 km/s probably driven by photospheric/chromospheric motion via MHD waves.

2. **세 편광은 세 MHD 모드에 대응한다 / The three polarisations map to the three MHD modes** — 수직/가로 진동은 fast magnetoacoustic 및 Alfvén kink 모드와, 종방향 진동은 slow (또는 중력-진자) 모드와 대응한다. 이 대응은 감쇠 메커니즘 선택을 결정한다(slow → 열적, kink → 공명흡수). / Vertical and transverse oscillations correspond to fast and kink/Alfvén modes; longitudinal oscillations to slow modes (or gravity pendulum). The mode type dictates which damping mechanism applies.

3. **Pendulum 모델이 종방향 LAO의 주기를 설명한다 / The pendulum model explains LAO longitudinal periods** — $P = 2\pi\sqrt{R/g_0}$ (Luna & Karpen 2012). 관측된 $P = 50$–90 분을 대입하면 자기장선 dip 곡률반경 $R = 40$–130 Mm, 최소 자기장 세기 $B_{\min} \geq g_0 P/\sqrt{4\pi^2/m n}$. 가스 압력·자기장 장력은 부차적 역할. / $P = 2\pi\sqrt{R/g_0}$ with observed periods 50–90 min gives dip curvature radii 40–130 Mm; gravity is the dominant restoring force, with pressure and tension secondary.

4. **공명흡수가 관측된 빠른 감쇠를 설명한다 / Resonant absorption explains the observed rapid damping** — τ_d/P ~ 1–4은 비단열 열 과정만으로는 설명 불가(fast 모드가 거의 감쇠 안 됨). 가로 비균일층 l 두께에서 Alfvén 연속체 공명으로 kink 에너지가 전달되어 $\tau_d/P = (2/\pi)(R/l)(\rho_p+\rho_c)/(\rho_p-\rho_c)$ 달성. / Observed τ_d/P ~ 1–4 cannot come from non-adiabatic thermal effects alone; resonant absorption in the Alfvén continuum within a thin transitional layer of width l yields exactly this range.

5. **Thread의 kink 모드 seismology는 자기장을 제약한다 / Thread kink-mode seismology constrains the magnetic field** — 관측된 위상속도 $c_{ph} \approx \sqrt{2}\,v_{Ap}$ (고밀도 대비)에서 thread Alfvén 속도 그리고 (가정된 $\rho_p$에서) 자기장 세기가 도출된다. 전형적 결과: B = 5–30 G. 종방향 대진폭 진동: B = 14–55 G. / Observed thread phase speeds give thread Alfvén speeds (and B for assumed density): B ~ 5–30 G from kink-mode fits, 14–55 G from pendulum-model longitudinal LAOs.

6. **Bayesian 추론이 역문제에 엄밀한 확률적 틀을 제공한다 / Bayesian inference offers a rigorous probabilistic framework for the inversion** — 관측 불확실성(σ_P, σ_τ)을 prior와 likelihood에 반영하여 parameter posterior를 계산하고 주변화(marginalisation)로 단일 파라미터 추정 및 신뢰 구간을 얻는다. Model comparison(Bayes factor)으로 경쟁 밀도 프로파일(Lorentzian vs. Gaussian vs. parabolic) 비교. / Bayesian methods propagate observational uncertainties into posterior distributions for model parameters and enable model comparison via Bayes factors — Soler et al. (2015) found Lorentzian density profiles favoured over Gaussian/parabolic.

7. **주기비($P_1/2P_2$) 기법이 종방향 밀도 구조를 탐침한다 / Period-ratio seismology probes longitudinal density structure** — 균일 thread는 $P_1/2P_2 = 2$; 프로미넌스 중심에 밀도 집중 시 > 2로 증가. Lin et al. (2007) 관측 $P_1/2P_2 = 2.22$ → W/L = 0.12, L ~ 130 Mm, $v_{Ap}$ ~ 160 km/s. / Departures from $P_1/2P_2 = 2$ diagnose longitudinal density profiles; observed ratios of ~2.22 imply strongly concentrated prominence cores.

8. **복사 MHD 모델이 관측 분광 지표와 MHD 모드를 직접 연결한다 / Radiative MHD models directly link observed spectral indicators to MHD modes** — Heinzel et al. (2014), Zapiór et al. (2016): hS, hF, 1iS, 1iF 모드가 H-alpha, H-beta의 Doppler, FWHM, 강도에 고유한 시간 패턴을 생성하여 관측적 모드 식별의 기반 제공. / The four hybrid/internal MHD modes leave distinctive imprints in Doppler shift, FWHM, and integrated line intensity of H-alpha/H-beta, enabling unambiguous mode identification from multi-diagnostic observations.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 MHD wave speeds / MHD 파동 속도

**Alfvén speed / 알펜 속도:**
$$v_A^2 = \frac{B^2}{\mu_0 \rho} \tag{15}$$

**Sound speed / 음속:**
$$c_s^2 = \frac{\gamma R T}{\tilde{\mu}} \tag{14}$$

**Fast speed (prominence) / 고속 속도:**
$$c_f = \sqrt{v_A^2 + c_s^2} \tag{13}$$

**Cusp (tube) speed / Cusp (튜브) 속도:**
$$c_T = \frac{v_A c_s}{\sqrt{v_A^2 + c_s^2}} \tag{16}$$

**Numerical example / 수치 예**: prominence with $B = 10$ G, $\rho_p = 5\times 10^{-11}$ kg m⁻³, T = 8000 K →
- $v_A = B/\sqrt{\mu_0 \rho} = 10^{-3} / \sqrt{4\pi\times 10^{-7}\times 5\times 10^{-11}} \approx 126$ km/s
- $c_s = \sqrt{\gamma k_B T/m_p} \approx 14$ km/s (γ = 5/3, full ionisation)
- $c_f \approx 127$ km/s, $c_T \approx 14$ km/s

### 4.2 Simple string analogue / 단순 스트링 근사

$$P = 2\pi\left(\frac{L}{g}\tan\theta\right)^{1/2} \tag{9}$$

**Example**: $L = 25,000$ km = $2.5\times 10^7$ m, $g = 274$ m/s², $\theta = 10°$ → $\tan\theta = 0.176$; $P = 2\pi\sqrt{0.176 \times 2.5\times 10^7/274} \approx 2\pi\sqrt{16055}\approx 796$ s ≈ **13 min**.

### 4.3 Loaded string with finite prominence / 유한 폭 loaded string

$$P = 2\pi\frac{(L x_p)^{1/2}}{c_{\text{pro}}} \tag{12}$$

For the fundamental fast, Alfvén, and slow modes, substitute $c_{\text{pro}} = c_f, v_A, c_T$. With $v_A = 28$ km/s, $c_s = 15$ km/s, $L = 50,000$ km, $x_p = 2500$ km:
- $c_f = 31.8$ km/s → $P_{\text{fast}} = 2\pi\sqrt{1.25\times 10^{14}}/31.8\times 10^3 \approx 26$ min
- $c_T = 13.2$ km/s → $P_{\text{slow}} = 63$ min
- $v_A = 28$ km/s → $P_{\text{Alfvén}} = 30$ min

### 4.4 Pendulum period (longitudinal LAO) / 진자 주기 (종방향 대진폭)

$$P = 2\pi\sqrt{\frac{R}{g_0}} \tag{4}$$

**Example**: $R = 100$ Mm = $10^8$ m, $g_0 = 274$ m/s² → $P = 2\pi\sqrt{10^8/274} = 2\pi\sqrt{3.65\times 10^5} \approx 3798$ s ≈ **63 min** ✓ (matches typical LAO longitudinal observations).

Minimum field strength:
$$B \geq \sqrt{\frac{g_0^2 m n}{4\pi^2}}\, P \tag{5}$$
With n = 10¹¹ cm⁻³ = 10¹⁷ m⁻³, m = m_p = 1.67×10⁻²⁷ kg, P = 3798 s:
$B \geq \sqrt{274^2 \times 1.67\times 10^{-27}\times 10^{17}/(4\pi^2)}\times 3798 / \sqrt{\mu_0}$
$\approx 15$ G, consistent with Luna et al. (2014) result $B_{\min} = 14 \pm 8$ G.

### 4.5 Kink mode in infinitely long thread / 무한 thread kink 모드

$$\omega_k = k_z v_{Ap}\sqrt{\frac{2\zeta}{1+\zeta}} \tag{30}$$
$$P_k = \frac{\sqrt{2}}{2}\frac{\lambda}{v_{Ap}}\left(\frac{1+\zeta}{\zeta}\right)^{1/2} \tag{31}$$

**Example**: $\lambda = 3000$ km, $v_{Ap} = 100$ km/s, $\zeta = 200$:
$P_k = (1.414/2)(3000/100)\sqrt{201/200} \approx 0.707\times 30 \times 1.0025 \approx 21.3$ s
For $\lambda = 250,000$ km (fundamental of whole tube) same parameters → $P_k \approx 1770$ s ≈ **30 min**.

### 4.6 Kink period / field inversion (Nakariakov & Verwichte 2005) / kink 주기 역산

$$B_0 = \frac{L}{P}\sqrt{2\mu_0 \rho_0\left(1 + \frac{\rho_e}{\rho_0}\right)} \tag{44}$$

Equivalently:
$$B = \frac{L}{P}\cdot \frac{1}{\sqrt{2}}\cdot\sqrt{\mu_0(\rho_i+\rho_e)} \cdot 2
= L\frac{2\pi}{T}\sqrt{\frac{\mu_0\rho_i(1+\rho_e/\rho_i)}{2}}$$

**Example (Xue et al. 2014)**: $L = ?$ (filament length), $P = 1140$ s, $\rho_0 = 5\times 10^{-11}$ kg m⁻³, $\rho_e/\rho_0 = 1/200$ → B ≈ 17.6 G.

Reference numerical example: $L = 50$ Mm, $P = 30$ min = 1800 s, $\rho_0 = 5\times 10^{-11}$ kg m⁻³ →
$B = (5\times 10^7/1800)\sqrt{2\times 4\pi\times 10^{-7}\times 5\times 10^{-11}\times 1.005}$
$= 2.78\times 10^4 \sqrt{1.26\times 10^{-16}}$
$= 2.78\times 10^4 \times 1.12\times 10^{-8}$ T = $3.1\times 10^{-4}$ T ≈ **3 G**. Larger B requires shorter P or longer L.

### 4.7 Resonant damping ratio (thin tube, thin boundary) / 공명 감쇠비

$$\frac{\tau_d}{P} = \frac{2}{\pi}\frac{R}{l}\frac{\rho_p + \rho_c}{\rho_p - \rho_c} \tag{37}$$

**Example**: $\rho_p/\rho_c = 200$, $l/R = 0.2$:
$\tau_d/P = (2/\pi)(5)(201/199) = 3.19$ ✓ (matches observed range 1–4).

For thread with $\zeta \to \infty$: $\tau_d/P = (2/\pi)(R/l)$. Observed $\tau_d/P = 3$ → $l/R \approx 0.21$ (asymptotic, density-contrast-independent).

### 4.8 Bayesian inversion / 베이지안 역산

$$p(\boldsymbol{\theta}|D,M) = \frac{p(D|\boldsymbol{\theta},M)\,p(\boldsymbol{\theta},M)}{\int p(D|\boldsymbol{\theta},M)\,p(\boldsymbol{\theta},M)\,d\boldsymbol{\theta}} \tag{54}$$

- **posterior** $p(\boldsymbol{\theta}|D,M)$: parameters θ given data D and model M
- **likelihood** $p(D|\boldsymbol{\theta},M)$: probability of observations given parameters
- **prior** $p(\boldsymbol{\theta},M)$: initial belief before data
- **evidence** $\int p(D|\boldsymbol{\theta},M) p(\boldsymbol{\theta},M) d\boldsymbol{\theta}$: normalising integral

For inversion with observables $D = (P, \tau_d)$ and parameters $\boldsymbol{\theta} = (v_{Ap}, l/R)$:
$$P \sim \frac{\sqrt{2}}{2}\frac{\lambda}{v_{Ap}}, \qquad \frac{\tau_d}{P}\sim \frac{2}{\pi}\frac{R}{l} \tag{55}$$
Gaussian likelihood with widths $\sigma_P$, $\sigma_{\tau_d}$:
$$p(D|\boldsymbol{\theta}) = \frac{1}{2\pi\sigma_P\sigma_\tau}\exp\left[-\frac{(P-P_\text{model})^2}{2\sigma_P^2} - \frac{(\tau-\tau_\text{model})^2}{2\sigma_\tau^2}\right]$$
Uniform priors on ($v_{Ap}$, $l/R$) yield marginal posteriors $p(v_{Ap}|D)$, $p(l/R|D)$ with well-defined peaks (see Fig. 93 of paper).

### 4.9 Damping envelope / 감쇠 진동 형태

$$v(t) = v_0 \cos(\omega t + \phi)\exp(-t/\tau_d) \tag{ch5}$$

or Hillier et al. (2013)'s attenuated sinusoid with linearly varying period:
$$\xi(t) = A_0 \exp(t/\tau)\sin\left[\frac{2\pi t}{P_0(1+Ct)} + S\right] \tag{8}$$

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
           Prominence Oscillations — Historical Timeline
                  프로미넌스 진동 — 역사적 타임라인

1930s-40s ────── Early prominence observations (Menzel, Pettit)
                 [prominences as curiosities]

1960 ─────────── Moreton & Ramsey: Moreton wave → filament response
1966 ─────────── Hyder: 11 winking filaments, P = 6-40 min
                 [birth of LAO phenomenology]

1969 ─────────── Harvey: first systematic SAO detection (Doppler)
                 [birth of SAO phenomenology]

1984 ─────────── Roberts, Edwin, Benz: coronal MHD seismology idea
                 [seismology framework born]

1991 ─────────── Yi et al., Yi & Engvold: He I thread oscillations
                 [fine-structure era begins]

1992 ─────────── Joarder & Roberts: slab model, fast/Alfvén/slow modes
                 [modern theoretical foundation]

1995 ─────────── Tandberg-Hanssen: prominence seismology proposed

2002 ─────────── ★ Oliver & Ballester: first Living Reviews
2002 ─────────── Ruderman & Roberts: resonant absorption, kink damping
                 [damping puzzle solved for coronal loops]

2003-06 ──────── Jing et al.: Large Amplitude Longitudinal Oscillations
                 [new LAO class]

2005-07 ──────── Lin, Okamoto: SST/Hinode thread propagating waves
                 [resolved thread-scale oscillations]

2008 ─────────── Soler, Arregui et al.: thread damping mechanisms
2009 ─────────── Goossens et al.: TT-TB resonant inversion scheme
2010 ─────────── Díaz et al.: period ratio seismology

2012 ─────────── ★ Luna & Karpen: pendulum model (P = 2π√(R/g))
                 [LAO longitudinal theory settled]

2014 ─────────── Arregui et al.: Bayesian prominence seismology
                 [probabilistic inversion framework]

2015 ─────────── Soler et al.: Bayesian model comparison
                 (Lorentzian vs Gaussian vs parabolic)

2018 ─────────── ★★ Arregui, Oliver, Ballester: THIS REVIEW
                 [comprehensive synthesis; 154 pages, ~500 refs]

2020+ ────────── Future: DKIST, Solar Orbiter, ASO-S/CHASE
                 thread-scale Bayesian seismology, partial ionisation
```

**English**: This review consolidates a half-century of prominence-oscillation research. The arc moves from simple phenomenology (1960s–1970s) through linear MHD slab theory (1990s), fine-structure thread theory and damping mechanisms (2000s), to probabilistic seismology (2010s). Arregui et al. (2018) bridges prominence seismology with the older, more mature coronal-loop seismology — both employ resonant absorption and Bayesian methods — and sets an agenda for DKIST-era thread-resolved studies.

**한국어**: 이 리뷰는 반세기 프로미넌스 진동 연구를 집약한다. 1960–70년대 현상학 → 1990년대 선형 MHD slab 이론 → 2000년대 fine-structure thread 이론 및 감쇠 메커니즘 → 2010년대 확률적 지진학으로 이어지는 흐름을 정리했다. Arregui et al. (2018)은 프로미넌스 지진학을 더 성숙한 코로나 루프 지진학과 연결(둘 다 공명흡수·Bayesian 기법 사용)하며, DKIST 시대 thread 분해 연구의 방향을 제시한다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Oliver & Ballester (2002), *Solar Phys.* 206, 45 | Previous *Living Reviews* on the same topic / 동일 주제 이전 *Living Reviews* | Direct predecessor; this review updates and expands it / 본 리뷰의 직접 선행, 개정·확장 대상 |
| Joarder & Roberts (1992a,b), *A&A* 256, 264 & 261, 625 | Foundational slab dispersion analysis / 기초 slab 분산 분석 | Equations (17)–(22) of this review derive from their work / 식 (17)–(22)의 기반 |
| Luna & Karpen (2012), *ApJ* 750, L1 | Pendulum model for longitudinal LAOs / 종방향 대진폭 진자 모델 | Cornerstone of Sect. 4.3; P = 2π√(R/g) cited extensively / Sect. 4.3 핵심 |
| Ruderman & Roberts (2002), *ApJ* 577, 475 | Resonant absorption in coronal loops / 코로나 루프 공명흡수 | Damping mechanism imported into prominence seismology / 감쇠 기제의 원천 |
| Goossens et al. (2008), *A&A* 484, 851 | TT-TB inversion scheme / 박형 튜브·박형 경계 역산 기법 | Basis for Eq. (37), (48); Soler et al. applied to threads / 식 (37), (48)의 기반 |
| Lin et al. (2007, 2009), *Solar Phys.* 246, 65; *ApJ* 704, 870 | SST thread swaying observations / SST thread swaying 관측 | Key SAO propagating-wave data for Sect. 8.3 seismology / Sect. 8.3 주요 관측 |
| Okamoto et al. (2007), *Science* 318, 1577 | Hinode flowing + oscillating threads / Hinode 흐름·진동 thread | Basis of Sect. 8.6 Terradas et al. (2008) seismology / Sect. 8.6 기반 |
| Hillier et al. (2013), *ApJ* 779, 16 | 3436 thread oscillations, P-V correlations / 3436개 thread 진동 통계 | Defines modern SAO period/amplitude statistics (Figs. 8–10) / SAO 통계 정의 |
| Arregui et al. (2014), *A&A* 565, A78 | Bayesian thread seismology / 베이지안 thread 지진학 | Sect. 8.7.1 core application; defines Eq. (55) / Sect. 8.7.1의 핵심 적용 |
| Nakariakov & Verwichte (2005), *Living Rev. Sol. Phys.* 2, 3 | Coronal-loop seismology review / 코로나 루프 지진학 리뷰 | Complementary review; provides Eq. (44) and conceptual parallel / 상보적 리뷰; 식 (44) 제공 |
| Tandberg-Hanssen (1995), *The Nature of Solar Prominences* | Textbook on prominences / 프로미넌스 교과서 | First proposed prominence seismology / 프로미넌스 지진학 최초 제안 |
| Labrosse et al. (2010), *Space Sci. Rev.* 151, 243 | Physics of prominences (companion review) / 프로미넌스 물리 (자매 리뷰) | Physical conditions input for seismology / 지진학의 물리 조건 입력 |

---

## 7. References / 참고문헌

**Primary / 주 논문**:
- Arregui, I., Oliver, R., & Ballester, J. L., "Prominence oscillations", *Living Reviews in Solar Physics* 15:3, 2018. DOI: 10.1007/s41116-018-0012-6

**Foundational theory / 기초 이론**:
- Joarder, P. S., & Roberts, B., "The modes of oscillation of a Kippenhahn-Schlüter prominence model", *A&A* 256, 264, 1992a.
- Joarder, P. S., & Roberts, B., "The modes of oscillation of a prominence... transverse magnetic field", *A&A* 261, 625, 1992b.
- Oliver, R., et al., "The influence of the prominence gas pressure on the modes of oscillation of Kippenhahn-Schlüter type prominences", *ApJ* 400, 369, 1992.
- Oliver, R., Ballester, J. L., et al., "Oscillations in a plasma slab embedded in a magnetic atmosphere", *ApJ* 409, 809, 1993.

**Key observational papers / 주요 관측 논문**:
- Hyder, C. L., "Winking filaments and prominence and coronal magnetic fields", *Z. Astrophys.* 63, 78, 1966.
- Harvey, J. W., PhD thesis, Univ. of Colorado, 1969.
- Jing, J., Lee, J., et al., "Periodic motion along a solar filament...", *ApJ* 584, L103, 2003.
- Okamoto, T. J., et al., "Coronal transverse magnetohydrodynamic waves in a solar prominence", *Science* 318, 1577, 2007.
- Lin, Y., Soler, R., et al., "Swaying threads of a solar filament", *ApJ* 704, 870, 2009.
- Hillier, A., Morton, R., & Erdélyi, R., "A statistical study of transverse oscillations in a quiescent prominence", *ApJ* 779, L16, 2013.

**Pendulum and longitudinal LAOs / 진자 및 종방향 대진폭**:
- Luna, M., & Karpen, J., "Large-amplitude longitudinal oscillations in a solar filament", *ApJ* 750, L1, 2012.
- Luna, M., Knizhnik, K., Muglach, K., et al., "Observations and implications of large-amplitude longitudinal oscillations", *ApJ* 785, 79, 2014.

**Damping / 감쇠**:
- Ruderman, M., & Roberts, B., "The damping of coronal loop oscillations", *ApJ* 577, 475, 2002.
- Arregui, I., Andries, J., et al., "MHD seismology of coronal loops using the period and damping of quasi-modes", *A&A* 463, 333, 2007.
- Goossens, M., Arregui, I., et al., "Analytical approximate seismology of transversely oscillating coronal loops", *A&A* 484, 851, 2008.
- Soler, R., Oliver, R., & Ballester, J. L., "Nonadiabatic magnetoacoustic waves in a partially ionized prominence plasma", *ApJ* 699, 1553, 2009.

**Bayesian seismology / 베이지안 지진학**:
- Arregui, I., & Asensio Ramos, A., "Bayesian magnetohydrodynamic seismology of coronal loops", *ApJ* 740, 44, 2011.
- Arregui, I., Asensio Ramos, A., & Díaz, A. J., "Bayesian inference approach to probabilistic prominence seismology", *ApJ Lett.* 765, L23, 2013.
- Arregui, I., Soler, R., & Asensio Ramos, A., "Model comparison for the density structure along solar prominence threads", *ApJ* 811, 104, 2015.

**Context / 맥락**:
- Tandberg-Hanssen, E., *The Nature of Solar Prominences*, Kluwer, 1995.
- Labrosse, N., Heinzel, P., et al., "Physics of solar prominences: I—Spectral diagnostics and non-LTE modelling", *Space Sci. Rev.* 151, 243, 2010.
- Mackay, D. H., Karpen, J. T., et al., "Physics of solar prominences: II—Magnetic structure and dynamics", *Space Sci. Rev.* 151, 333, 2010.
- Parenti, S., "Solar prominences: observations", *Living Rev. Sol. Phys.* 11, 1, 2014.
- Nakariakov, V. M., & Verwichte, E., "Coronal waves and oscillations", *Living Rev. Sol. Phys.* 2, 3, 2005.
- Oliver, R., & Ballester, J. L., "Oscillations in quiescent solar prominences: observations and theory", *Solar Phys.* 206, 45, 2002.

---

## Appendix A: Glossary of Symbols / 기호 용어집

| Symbol / 기호 | Meaning / 의미 |
|---|---|
| $v_A$ | Alfvén speed / 알펜 속도 |
| $c_s$ | Sound speed / 음속 |
| $c_f$ | Fast magnetoacoustic speed / 고속 마그네토음향 속도 |
| $c_T$ | Cusp (tube) speed / cusp (튜브) 속도 |
| $c_k$ | Kink speed / kink 모드 속도 |
| $\rho_p, \rho_i$ | Prominence / internal density / 프로미넌스 내부 밀도 |
| $\rho_c, \rho_e$ | Coronal / external density / 코로나 외부 밀도 |
| $\zeta$ | Density contrast $\rho_p/\rho_c$ / 밀도 대비 |
| $B, B_0$ | Magnetic field strength / 자기장 세기 |
| $L$ | Magnetic tube half-length or filament length / 자기 튜브 반길이 또는 필라멘트 길이 |
| $R$ | Thread radius or dip curvature radius / thread 반지름 또는 dip 곡률반경 |
| $l$ | Transverse inhomogeneity layer thickness / 가로 비균일층 두께 |
| $P$ | Oscillation period / 진동 주기 |
| $\tau_d$ | Damping time / 감쇠 시간 |
| $\lambda$ | Wavelength / 파장 |
| $k_z$ | Axial wavenumber / 축 방향 파수 |
| $\omega$ | Angular frequency / 각주파수 |
| $m$ | Azimuthal wavenumber (m=0 sausage, m=1 kink) / 방위각 파수 |
| $g_0$ | Solar surface gravity 274 m/s² / 태양 표면 중력 |
| $\boldsymbol{\theta}$ | Bayesian parameter vector / 베이지안 파라미터 벡터 |

---

## Appendix B: Representative Parameter Values / 대표 파라미터 값

| Quantity / 물리량 | Typical value / 전형 값 |
|---|---|
| Prominence temperature $T_p$ | 6000–10,000 K |
| Coronal temperature $T_c$ | (1–2) × 10⁶ K |
| Prominence density $n_p$ | 10¹⁰–10¹¹ cm⁻³ |
| Coronal density $n_c$ | 10⁸–10⁹ cm⁻³ |
| Density contrast $\zeta$ | 100–500 |
| Magnetic field strength $B$ | 5–30 G (quiescent), 20–70 G (active region) |
| Prominence Alfvén speed $v_{Ap}$ | 50–200 km/s |
| Sound speed $c_s$ | 11–15 km/s |
| Thread width $2R$ | 100–600 km (~0.1″–0.6″) |
| Thread length $2W$ | 3500–28,000 km |
| Flux tube length $2L$ | 10⁵–2×10⁵ km |
| Short-period SAO | 1–10 min |
| Intermediate-period SAO | 10–40 min |
| Long-period SAO | 40–90 min |
| Ultra-long-period SAO | 1–10 h |
| LAO periods | 6–150 min |
| Damping ratio $\tau_d/P$ | 1–10 (SAO), 1–4 (typical kink) |
| Wavelength $\lambda$ | 3000–75,000 km |
| Phase speed $c_{ph}$ | 5–200 km/s |

---

## Appendix C: Observational Highlights / 관측 하이라이트

**English**: Several representative observational campaigns are cited repeatedly throughout the review:

1. **Shen et al. (2014a)** — SMART + AIA/SDO, X2.1 flare, four winking filaments triggered simultaneously by a dome EUV wave (Fig. 1). Damping indicated viscous/leakage mechanisms.
2. **Hershaw et al. (2011)** — SOHO/EIT 2005-07-30 arched prominence, two wave trains from two flares, periods 86–104 min, velocities 10–50 km/s (Fig. 2).
3. **Luna et al. (2014)** — SDO/AIA longitudinal LAO; 36 slit positions showed near-uniform period 0.7–0.86 h; fitted with Bessel and damped cosine models.
4. **Lin et al. (2007, 2009)** — SST thread swaying, wavelength 3.8″, phase speed 10–40 km/s along threads.
5. **Okamoto et al. (2007, 2015)** — Hinode/SOT + IRIS flowing thread oscillations, periods 135–250 s. 2015 paper reveals 90°–180° phase shift between Hinode displacement and IRIS Doppler signal — signature of resonant absorption.
6. **Hillier et al. (2013)** — Hinode/SOT, 3436 oscillating features, power-law $A \propto P^{0.74}$, $V \propto P^{-0.25}$ (Fig. 10).
7. **Molowny-Horas et al. (1999), Terradas et al. (2002)** — first 2D Doppler maps over (54,000 × 40,000) km region; damping τ = 101–140 min for P = 70 min; wavelength 44,000–75,000 km.

**한국어**: 리뷰에 반복 인용되는 대표 관측 사례:

1. **Shen et al. (2014a)** — SMART + AIA/SDO, X2.1 플레어, EUV dome 파가 네 개 winking 필라멘트를 동시 트리거 (Fig. 1). 감쇠는 점성·누출 기제 시사.
2. **Hershaw et al. (2011)** — SOHO/EIT 2005-07-30 아치형 프로미넌스, 두 플레어의 두 파동 열차, 주기 86–104분, 속도 10–50 km/s (Fig. 2).
3. **Luna et al. (2014)** — SDO/AIA 종방향 LAO; 36개 슬릿 거의 균일 주기 0.7–0.86 h; Bessel·감쇠 코사인 피팅.
4. **Lin et al. (2007, 2009)** — SST thread swaying, 파장 3.8″, 위상속도 10–40 km/s.
5. **Okamoto et al. (2007, 2015)** — Hinode/SOT + IRIS 흐름 thread 진동, 주기 135–250초. 2015년 논문은 변위와 Doppler 사이 90°–180° 위상차 발견 — 공명흡수의 지표.
6. **Hillier et al. (2013)** — Hinode/SOT 3436 진동 성분, 거듭제곱 $A \propto P^{0.74}$, $V \propto P^{-0.25}$ (Fig. 10).
7. **Molowny-Horas et al. (1999), Terradas et al. (2002)** — 최초 2D Doppler 맵 (54,000×40,000 km), 감쇠 τ = 101–140 분 (P = 70 분), 파장 44,000–75,000 km.
