---
title: "Reading Notes — Emslie et al. (2012): Global Energetics of Thirty-Eight Large Solar Eruptive Events"
date: 2026-04-27
topic: Solar_Physics
paper: 34_emslie_2012
authors: "A. G. Emslie, B. R. Dennis, A. Y. Shih, P. C. Chamberlin, R. A. Mewaldt, C. S. Moore, G. H. Share, A. Vourlidas, B. T. Welsch"
journal: "ApJ 759, 71"
year: 2012
doi: "10.1088/0004-637X/759/1/71"
tags: [solar-flare, energetics, CME, SEP, RHESSI, GOES, magnetic-energy, energy-partition, statistical-study]
---

# Global Energetics of Thirty-Eight Large Solar Eruptive Events
# 38개 대형 태양 폭발 사건의 전 에너지 수지

---

## 1. Core Contribution / 핵심 기여

**English.** Emslie et al. (2012) present the first ensemble statistical analysis of energy partitioning across all eight observable components of large solar eruptive events (SEEs). The sample comprises 38 X-class–dominated flares observed between February 2002 and December 2006 — the first five years of RHESSI operation, coincident with maximum coverage by ACE, GOES, SOHO/LASCO, SAMPEX, and SORCE. For each event the authors compute, where data permit: (i) GOES 1–8 Å radiated energy, (ii) total radiated energy from the SXR-emitting plasma (T-rad), (iii) bolometric radiated output, (iv) peak thermal energy of the SXR plasma, (v) energy in flare-accelerated electrons (>20 keV), (vi) energy in flare-accelerated ions (>1 MeV), (vii) CME kinetic energy in both the solar and solar-wind rest frames plus gravitational potential, (viii) energy in interplanetary SEPs, and (ix) free magnetic energy in the active region. Five major conclusions emerge: U_T-rad exceeds U_th^peak by ~order of magnitude (continuous reheating); U_e + U_i ≳ E_bol (particles can power total radiation); U_e ≈ U_i (electron-ion equipartition within factors of a few); U_SEP ≈ few % of U_K^SW (CME shocks moderately efficient at SEP production); E_mag ≳ U_K + U_e + U_i + U_th (magnetic free energy is the universal reservoir). This paper became the de facto reference for solar flare energetics, defining the canonical numbers (~20–30% in CME, ~10–20% in nonthermal particles, ~5% in SEPs, ~10–20% in radiation) cited in nearly all subsequent flare/CME modeling work.

**한국어.** Emslie et al. (2012)는 대형 태양 폭발 사건(SEE)의 관측 가능한 8개 에너지 성분 전체에 대한 최초의 통계적 앙상블 분석을 제시합니다. 샘플은 2002년 2월부터 2006년 12월 사이 — RHESSI 운영 첫 5년간 ACE, GOES, SOHO/LASCO, SAMPEX, SORCE의 관측이 최대로 겹치는 시기 — 의 X급 위주 38개 플레어로 구성됩니다. 각 사건에 대해 데이터가 허용하는 한 다음을 계산합니다: (i) GOES 1–8 Å 복사에너지, (ii) SXR 방출 플라즈마의 총 복사에너지 (T-rad), (iii) 볼로메트릭 복사 출력, (iv) SXR 플라즈마의 피크 열에너지, (v) 플레어 가속 전자(>20 keV) 에너지, (vi) 플레어 가속 이온(>1 MeV) 에너지, (vii) 태양·태양풍 두 좌표계의 CME 운동에너지 + 중력 위치에너지, (viii) 행성간 SEP 에너지, (ix) 활동영역 자유 자기에너지. 5가지 주요 결론: U_T-rad가 U_th^peak를 약 한 자릿수 초과 (지속적 재가열); U_e + U_i ≳ E_bol (입자가 총 복사를 공급 가능); U_e ≈ U_i (수배 이내 등분배); U_SEP ≈ U_K^SW의 수 % (CME 충격파가 SEP 생성에서 적절히 효율적); E_mag ≳ U_K + U_e + U_i + U_th (자유 자기에너지가 보편적 저장소). 이 논문은 태양 플레어 에너지론의 사실상 표준 참조가 되어, 이후 거의 모든 플레어/CME 모델링 연구에서 인용되는 표준 수치(CME에 ~20–30%, 비열적 입자에 ~10–20%, SEP에 ~5%, 복사에 ~10–20%)를 정의했습니다.

---

## 2. Reading Notes / 읽기 노트

### 2.1 Introduction (Section 1, pp. 1–4) / 서론

**English.** The paper opens by framing SEEs (flares + CMEs) as the most energetic events in the solar system, releasing up to 10^32 erg in tens of seconds to minutes. While the *total* energy is well-known, its *partition* across components had only been measured for two events previously — Emslie et al. (2004, 2005) on the 2002 April 21 (X1.5) and July 23 (X4.8) flares, which appear in this 38-event sample as Events #2 and #6. The motivation is twofold: (1) provide *typical* ratios of energy components as constraints on energy-release modeling; (2) identify *outliers* where one component lies far outside the norm. The paper also extends Mewaldt et al. (2008), who first studied the SEP-to-CME-kinetic-energy ratio statistically. The authors caution that components must not be naively summed because energy can transfer chain-wise (e.g., flare-accelerated electrons → thermal plasma → SXR emission), so adding them double-counts. Eleven energy quantities are tabulated (Section 2 list); the first four (SXR, T-rad, Bol, Peak) are related but *not* additive — they describe the same thermal reservoir at different stages.

**한국어.** 논문은 SEE(플레어 + CME)를 태양계에서 가장 에너지가 큰 사건으로 자리매김하며, 수십 초에서 수 분 사이에 최대 10^32 erg를 방출한다고 소개합니다. *총* 에너지는 잘 알려져 있으나, *성분별 분배*는 이전에 단 두 사건 — Emslie et al. (2004, 2005)의 2002년 4월 21일(X1.5)과 7월 23일(X4.8) 플레어, 이 38개 샘플의 사건 #2와 #6 — 에 대해서만 측정되었습니다. 동기는 두 가지: (1) 에너지 방출 모델링의 제약 조건으로서 *전형적* 비율 제공, (2) 한 성분이 평균에서 크게 벗어나는 *이상치* 식별. 또한 Mewaldt et al. (2008) — SEP 대 CME 운동에너지 비율을 최초로 통계 연구한 작업 — 의 확장이기도 합니다. 저자들은 에너지가 연쇄 전달(예: 플레어 가속 전자 → 열 플라즈마 → SXR 방출)되므로 성분들을 단순 합산하면 이중 계산이 됨을 경고합니다. 11개 에너지 양이 표로 정리되며 (Section 2 목록), 처음 4개(SXR, T-rad, Bol, Peak)는 관련 있지만 *덧셈 가능하지 않습니다* — 동일한 열 저장소를 서로 다른 단계에서 기술하기 때문입니다.

### 2.2 Component Energies, Section 2 List (p. 4) / 성분 에너지 목록

**English.** The 11-item enumeration:
1. Radiated energy in GOES 1–8 Å band
2. Total radiated energy from SXR-emitting plasma
3. Total bolometric radiated output
4. Peak thermal energy of SXR plasma
5. Energy in flare-accelerated electrons
6. Energy in flare-accelerated ions
7. CME kinetic energy in Sun rest frame
8. CME kinetic energy in solar-wind rest frame
9. CME gravitational potential energy
10. Energy in SEPs
11. Free (nonpotential) magnetic energy in active region

The paper also explicitly excludes thermal conduction losses and energy in turbulence/directed mass motions, which Doschek et al. (1992) suggested may be comparable to U_th. Of 38 events, only 6 (Events #13, 14, 20, 23, 25, 38) have *all* components measured.

**한국어.** 11개 항목 열거:
1. GOES 1–8 Å 대역 복사에너지
2. SXR 방출 플라즈마의 총 복사에너지
3. 총 볼로메트릭 복사 출력
4. SXR 플라즈마의 피크 열에너지
5. 플레어 가속 전자 에너지
6. 플레어 가속 이온 에너지
7. 태양 정지 좌표계 CME 운동에너지
8. 태양풍 정지 좌표계 CME 운동에너지
9. CME 중력 위치에너지
10. SEP 에너지
11. 활동영역 자유(비포텐셜) 자기에너지

또한 열 전도 손실과 난류·정렬된 질량운동의 에너지(Doschek et al. 1992가 U_th와 비슷할 수 있다고 제안)는 명시적으로 제외됩니다. 38개 사건 중 *모든* 성분이 측정된 것은 6개(사건 #13, 14, 20, 23, 25, 38)뿐입니다.

### 2.3 Section 2.1 — Radiated Energy from Hot Plasma (p. 7) / 고온 플라즈마 복사

**English.** GOES SXR fluxes (1–8 Å, 0.5–4 Å) at 3-s cadence are background-subtracted using the lowest flux in the hour before/after flare. Integration runs from NOAA-listed start time to when 1–8 Å flux falls to 10% of peak. The IDL `rad_loss` procedure in SSW computes the optically-thin radiation loss rate from CHIANTI assuming coronal abundances and Mazzotta et al. (1998) ionization equilibria. Temperature and emission measure come from the GOES two-channel ratio via White et al. (2005) relations, assuming an isothermal plasma. T-rad values include >50% of the SXR-emitted radiation (some events have a "second phase" per Woods+ 2011, Su+ 2011 that can radiate as much as the initial phase).

**한국어.** GOES SXR 플럭스(1–8 Å, 0.5–4 Å)는 3초 간격으로, 플레어 전후 1시간 내 최저 플럭스를 배경으로 빼줍니다. NOAA 시작 시각부터 1–8 Å 플럭스가 피크의 10%로 떨어질 때까지 적분합니다. SSW의 IDL `rad_loss` 절차가 CHIANTI를 사용해 코로나 원소함량과 Mazzotta et al. (1998) 이온화 평형을 가정한 광학적으로 얇은 복사 손실률을 계산합니다. 온도와 방출측도는 GOES 2채널 비율로부터 White et al. (2005) 관계식을 통해 등온 플라즈마 가정 하에 얻습니다. T-rad 값은 SXR 복사의 >50%를 포함합니다 (Woods+ 2011, Su+ 2011에 따르면 "후속 단계"가 초기 단계만큼 복사할 수 있음).

### 2.4 Section 2.2 — Bolometric Irradiance (pp. 7–9) / 볼로메트릭 복사

**English.** Five events have direct TIM (SORCE) measurement (#12, 13, 16, 31, 37); for the rest, FISM (Chamberlin et al. 2007, 2008) provides 1–1900 Å estimates with 1-min cadence using TIMED/SEE and UARS/SOLSTICE inputs. The 1–1900 Å radiated energy is converted to bolometric by an empirical factor of **2.42 ± 0.31**, calibrated against the 5 directly-measured events. A limb-darkening correction up to a factor of ~3 is applied. For the FISM-derived events, the bolometric values agree with TIM to within 40%. Total uncertainty: ±70% (disk center), ±90% (near limb).

**한국어.** 5개 사건은 TIM (SORCE) 직접 측정(#12, 13, 16, 31, 37); 나머지는 FISM (Chamberlin et al. 2007, 2008)이 TIMED/SEE와 UARS/SOLSTICE 입력을 활용해 1분 간격 1–1900 Å 추정을 제공합니다. 1–1900 Å 복사에너지는 직접 측정된 5개 사건에 대해 보정된 경험적 인자 **2.42 ± 0.31**로 볼로메트릭으로 변환됩니다. 광선이 일부 파장에서 광학적으로 두꺼워질 때 ~3배까지의 변연 감광 보정이 적용됩니다. FISM 기반 사건들은 TIM 값과 40% 이내로 일치합니다. 총 불확실성: ±70% (원반 중심), ±90% (변연 부근).

### 2.5 Section 2.3 — Peak Thermal Energy (pp. 9–10) / 피크 열에너지

**English.** RHESSI imaging spectroscopy (single-temperature Maxwellian + double-power-law nonthermal) via OSPEX forward fitting. Temperature T_0(K) and emission measure EM = ∫ n_e^2 dV (cm^-3) are extracted every 20 s.

**Key equation:**
$$U_{\text{th}} = 3 n_e k T_0 f V_{\text{ap}} \simeq 4.14 \times 10^{-16} T_0 \sqrt{EM \times f V_{\text{ap}}} \;\text{erg}$$

V_ap is from RHESSI 3σ-clean (Dennis & Pernak 2009); f is set to unity, justified by Guo et al. (2012) who derived f = 0.20 ×/÷ 3.9 (compatible with f=1 within 1σ logarithmic spread). The peak value of U_th is reported, usually at or near peak GOES flux.

**한국어.** RHESSI 영상 분광(단일 온도 맥스웰 + 이중 거듭제곱 비열) 을 OSPEX 전향 적합으로 수행. 온도 T_0(K)와 방출측도 EM = ∫ n_e^2 dV (cm^-3)를 20초마다 추출합니다.

**핵심 방정식:**
$$U_{\text{th}} = 3 n_e k T_0 f V_{\text{ap}} \simeq 4.14 \times 10^{-16} T_0 \sqrt{EM \times f V_{\text{ap}}} \;\text{erg}$$

V_ap는 RHESSI 3σ-clean (Dennis & Pernak 2009)에서; f는 Guo et al. (2012)의 f = 0.20 ×/÷ 3.9 (1σ 로그 스프레드에서 f=1과 양립) 결과로 정당화하여 1로 설정. 보통 GOES 피크 플럭스 근방에서 U_th의 최댓값이 보고됩니다.

### 2.6 Section 2.4 — Flare-Accelerated Electrons (pp. 10–11) / 플레어 가속 전자

**English.** OSPEX forward-fits an isothermal+nonthermal model to RHESSI Detector #4 spectra. Nonthermal photons assume thick-target bremsstrahlung from a broken power-law electron injection spectrum:

$$F_0(E_0) = A \begin{cases} 0 & E_0 < E_{\min} \\ (E_0/E_p)^{-\delta_1} & E_{\min} \le E_0 < E_b \\ (E_0/E_p)^{-\delta_2}(E_b/E_p)^{\delta_2-\delta_1} & E_b \le E_0 < E_{\max} \\ 0 & E_0 \ge E_{\max} \end{cases}$$

Seven free parameters: A (normalization), E_min, E_max, E_b (break), δ_1, δ_2; pivot E_p = 50 keV; high-energy cutoff E_max = 30 MeV (negligible effect). The dominant uncertainty is E_min: the thermal continuum hides the spectral cutoff, so the *largest* E_min still giving χ²_red ≃ 1 is chosen — making U_e a *lower limit*. Because δ_1 ≳ 4, U_e is extremely sensitive to E_min: lowering E_min by a factor of 2 can raise U_e by an order of magnitude. The fully-ionized cold thick-target assumption (Brown 1973; Kontar et al. 2003) is used; ionization corrections are minor for the lower-energy electrons that dominate U_e.

**한국어.** OSPEX가 RHESSI 검출기 #4 스펙트럼에 등온+비열 모형을 전향 적합합니다. 비열 광자는 꺾인 거듭제곱 전자 주입 스펙트럼에서 두꺼운 표적 제동복사를 가정합니다:

$$F_0(E_0) = A \begin{cases} 0 & E_0 < E_{\min} \\ (E_0/E_p)^{-\delta_1} & E_{\min} \le E_0 < E_b \\ (E_0/E_p)^{-\delta_2}(E_b/E_p)^{\delta_2-\delta_1} & E_b \le E_0 < E_{\max} \\ 0 & E_0 \ge E_{\max} \end{cases}$$

7개 자유 매개변수: A (정규화), E_min, E_max, E_b (꺾임), δ_1, δ_2; 피벗 E_p = 50 keV; 고에너지 컷오프 E_max = 30 MeV (영향 미미). 주요 불확실성은 E_min: 열 연속복사가 스펙트럼 컷오프를 가리므로 χ²_red ≃ 1을 만족하는 *최대* E_min을 선택 — 따라서 U_e는 *하한값*입니다. δ_1 ≳ 4이므로 U_e는 E_min에 매우 민감합니다: E_min을 2배 낮추면 U_e가 한 자릿수 증가할 수 있습니다. 완전 이온화 차가운 두꺼운 표적 가정(Brown 1973; Kontar et al. 2003) 을 사용; U_e를 지배하는 저에너지 전자의 경우 이온화 보정은 작습니다.

### 2.7 Section 2.5 — Flare-Accelerated Ions (pp. 11–13) / 플레어 가속 이온

**English.** Energies derived from RHESSI 2.223 MeV neutron-capture deuterium-formation line fluence (photons cm^-2 time-integral). Sample drawn from Shih (2009) and Shih et al. (2009) — flares with >2σ line detections (4σ upper limits otherwise) plus three 2006 flares (Events #36, 37, 38). The 1 MeV lower threshold is justified by ²⁰Ne lines extending down to ~3 MeV.

Procedure:
1. Fluence corrected for solar-atmosphere attenuation (Hua & Lingenfelter 1987) given heliocentric position
2. Convert to >30 MeV proton energy via Murphy et al. (2007), Shih (2009) factors (chosen because 2.223 MeV line is produced by ions ≳20 MeV nucleon^-1, less spectral-index sensitive)
3. Extrapolate >30 MeV → >1 MeV assuming single power-law index δ = 4 (range 3–5 from Ramaty et al. 1996)
4. Total ion energy = 3 × proton energy (composition correction: ratio (p+α+heavy):p ≈ 2–6, taken as 3)

A spectral-index uncertainty of ±1 produces ±1.5 orders of magnitude uncertainty in total >1 MeV ion energy due to the long lever arm.

Several events are flagged as ion lower limits: #12 (RHESSI missed peak; INTEGRAL coverage); #31 (only 2 min observed; large heliocentric angle); #32, #37 (missed impulsive peak); #36 (RHESSI in shadow until 10:31 UT); #14, 15, 22, 33, 38 (small late-flare miss); #36, 37, 38 (radiation damage by Dec 2006).

**한국어.** RHESSI 2.223 MeV 중성자 포획 중수소 형성선 플루언스(시간적분 광자 cm^-2)에서 에너지 도출. 샘플은 Shih (2009) 및 Shih et al. (2009)에서 — >2σ 선 검출 플레어 (그렇지 않으면 4σ 상한) + 2006년 추가 3개 플레어 (#36, 37, 38). 1 MeV 하한은 ²⁰Ne 선이 ~3 MeV까지 확장된다는 점으로 정당화됩니다.

절차:
1. 일면 위치를 고려한 태양 대기 감쇠 보정 (Hua & Lingenfelter 1987)
2. Murphy et al. (2007), Shih (2009) 인자로 >30 MeV 양성자 에너지로 변환 (2.223 MeV 선이 ≳20 MeV nucleon^-1 이온에서 생성되어 스펙트럼 지수 민감도가 낮음)
3. δ = 4 단일 거듭제곱 가정으로 >30 MeV → >1 MeV 외삽 (Ramaty et al. 1996의 3–5 범위)
4. 총 이온 에너지 = 양성자 × 3 (구성 보정: (p+α+heavy):p ≈ 2–6, 3 채택)

스펙트럼 지수 ±1 불확실성은 긴 외삽 거리 때문에 >1 MeV 이온 총 에너지에 ±1.5 자릿수 불확실성을 생성합니다.

여러 사건이 이온 하한으로 표시: #12 (RHESSI가 피크 놓침; INTEGRAL 보완); #31 (2분만 관측; 큰 일면 각); #32, #37 (충격기 피크 놓침); #36 (10:31 UT까지 그림자); #14, 15, 22, 33, 38 (후기 일부 누락); #36, 37, 38 (2006년 12월까지 방사선 손상).

### 2.8 Section 2.6 — Coronal Mass Ejection (pp. 13–14) / 코로나 질량 방출

**English.** CME masses, kinetic and potential energies follow Vourlidas et al. (2010, 2011) procedure on calibrated LASCO C2/C3 images:
1. Select two LASCO images: pre-event and event
2. Calibrate to mean solar brightness, subtract pre-event from event
3. Excess brightness → excess mass via Thompson scattering, assuming (a) all mass concentrated in plane-of-sky and (b) 90% H + 10% He composition
4. Track total mass and projected position/velocity of leading edge and center of mass

Energies:
$$U_K = \tfrac{1}{2} m V^2, \qquad U_\Phi = G M_\odot m \left( R_\odot^{-1} - r^{-1} \right)$$

Both are *lower bounds*. The plane-of-sky assumption underestimates mass; CMEs ≲40° from sky plane have mass underestimated by ~factor 2; for far-from-sky-plane events, U_K can be up to 8× larger and U_Φ up to 2× larger.

Solar-wind rest-frame KE (column SW): subtract 400 km/s from V before recomputing.

**한국어.** CME 질량·운동에너지·위치에너지는 Vourlidas et al. (2010, 2011) 절차로 보정된 LASCO C2/C3 영상에서:
1. 사건 전 영상과 사건 영상 선택
2. 평균 태양 밝기로 보정, 사건 전 영상을 사건 영상에서 빼기
3. 초과 밝기 → 톰슨 산란을 통한 초과 질량, (a) 모든 질량이 천구면에 집중, (b) 90% H + 10% He 가정
4. 총 질량과 선단·질량중심의 투영 위치/속도 추적

에너지:
$$U_K = \tfrac{1}{2} m V^2, \qquad U_\Phi = G M_\odot m \left( R_\odot^{-1} - r^{-1} \right)$$

둘 다 *하한값*. 천구면 가정이 질량을 과소평가; ≲40° 떨어진 CME는 질량이 ~2배 과소평가되며, 천구면에서 멀리 떨어진 사건은 U_K가 최대 8배, U_Φ가 최대 2배 클 수 있음.

태양풍 정지 좌표계 KE (열 SW): V에서 400 km/s를 빼고 재계산.

### 2.9 Section 2.7 — Solar Energetic Particles (p. 15) / 태양 고에너지 입자

**English.** SEPs are mostly accelerated by CME-driven shocks (exception: Event #1, 2002 Feb 20, possibly flare-accelerated; Chollet et al. 2010). U_SEP measured from electron spectra (~0.035–8 MeV), proton spectra (~0.05–400 MeV nucleon^-1), heavy-ion spectra (~0.05–100 MeV nucleon^-1) using nine instruments:
- ULEIS, EPAM, SIS (ACE)
- PET (SAMPEX)
- EPS (GOES-8/11)
- LET, HET (STEREO, for 2006 events)
- EPHIN (SoHO)

Eleven events have full H/He/heavy-ion spectra fit jointly with Band et al. (1993) double-power-law or Ellison-Ramaty (1985) exponential-cutoff power law. Other events: protons fit alone; He/heavy-ion contributions estimated from event-specific abundances measured by ULEIS/SIS; electrons from EPAM/PET/EPHIN.

**한국어.** SEP는 대부분 CME 충격파에 의해 가속됩니다 (예외: 사건 #1, 2002년 2월 20일, 플레어 가속 가능; Chollet et al. 2010). U_SEP는 9개 측정기를 사용해 전자 스펙트럼(~0.035–8 MeV), 양성자 스펙트럼(~0.05–400 MeV nucleon^-1), 무거운 핵 스펙트럼(~0.05–100 MeV nucleon^-1)에서 측정:
- ULEIS, EPAM, SIS (ACE)
- PET (SAMPEX)
- EPS (GOES-8/11)
- LET, HET (STEREO, 2006년 사건용)
- EPHIN (SoHO)

11개 사건은 H/He/무거운 핵 스펙트럼이 Band et al. (1993) 이중 거듭제곱 또는 Ellison-Ramaty (1985) 지수 컷오프 거듭제곱으로 동시 적합. 기타 사건: 양성자만 적합; He/무거운 핵 기여는 ULEIS/SIS의 사건별 원소함량으로 추정; 전자는 EPAM/PET/EPHIN.

### 2.10 Section 2.8 — Magnetic Free Energy (Mag column) / 자유 자기에너지

**English.** Listed in column 'Mag' of Table 1, ranging from ~110 to ~2900 × 10^30 erg. Estimated from active region observations (area + magnetic flux density), following the heuristic E_mag ~ B² V / 8π. Note: this is *available* free energy estimate, not a formal NLFFF extrapolation (which became routine only with SDO/HMI vector magnetograms post-2010). Aschwanden et al. (2014, 2017) later revisit this same dataset with HMI NLFFF for events overlapping into Cycle 24.

**한국어.** 표 1의 'Mag' 열에 ~110 ~ 2900 × 10^30 erg 범위로 기록. 활동영역 관측(면적 + 자속 밀도)에서 휴리스틱 E_mag ~ B² V / 8π로 추정. 주의: 이는 *가용* 자유 에너지 추정치이며 정식 NLFFF 외삽이 아닙니다 (SDO/HMI 벡터 자기장 관측이 일상화된 2010년 이후에야 가능). Aschwanden et al. (2014, 2017)이 이후 동일 데이터셋을 HMI NLFFF로 Cycle 24 사건과 함께 재분석합니다.

### 2.11 Table 1 — Event List Summary (p. 6) / 사건 목록 표 1 요약

**English.** All energies in units of 10^30 erg. Highlights from the 38 events:
- **Event #1** (2002/02/20, M5.1): smallest; Mag = 1200, but only modest particle/CME signatures
- **Event #6** (2002/07/23, X4.8): the canonical Emslie+ 2005 event; bolometric = 150
- **Event #12** (2003/10/28, X17): the famous "Halloween" flare; bolometric = 362, U_K = 1200 (SW: 850), one of largest in sample
- **Event #16** (2003/11/04, X28): largest GOES class ever; bolometric = 426, U_K not listed (behind-limb-related)
- **Event #25** (2005/01/20, X7.1): high SEP energy (7.8); high U_e (120), unusual proton-rich
- **Event #5** (2002/07/20, X3.3 behind-limb): outlier — bolometric >210 (lower limit), no Mag estimate possible

X-class is dominant; only one C-class (#3, 2002/05/22, C5.0) and several M-class events.

**한국어.** 모든 에너지는 10^30 erg 단위. 38개 사건 중 주요 사례:
- **사건 #1** (2002/02/20, M5.1): 최소; Mag = 1200이지만 입자/CME 신호 보통
- **사건 #6** (2002/07/23, X4.8): Emslie+ 2005 표준 사건; 볼로메트릭 = 150
- **사건 #12** (2003/10/28, X17): 유명한 "Halloween" 플레어; 볼로메트릭 = 362, U_K = 1200 (SW: 850), 샘플 최대 중 하나
- **사건 #16** (2003/11/04, X28): 역대 최대 GOES 등급; 볼로메트릭 = 426, U_K는 미기록 (배면 관련)
- **사건 #25** (2005/01/20, X7.1): 높은 SEP 에너지(7.8); 높은 U_e(120), 이례적 양성자 풍부 사건
- **사건 #5** (2002/07/20, X3.3 배면): 이상치 — 볼로메트릭 >210 (하한), Mag 추정 불가

X급이 지배적; C급은 1개(#3, 2002/05/22, C5.0)만, 나머지는 M급 또는 X급.

---

## 3. Key Takeaways / 핵심 시사점

### Insight 1 — U_T-rad ≫ U_th^peak (Continuous Reheating) / 지속적 재가열

**English.** The total radiated energy from the SXR plasma (T-rad) exceeds the *peak* thermal content of that plasma by ~order of magnitude. Numerically: median T-rad ≈ 17 × 10^30 erg vs. median Peak ≈ 2.4 × 10^30 erg, ratio ≈ 7. This proves the SXR plasma is *continuously reheated* throughout the flare — energy is deposited and radiated multiple times during the gradual phase. This is consistent with sustained reconnection-driven energy release on multi-minute timescales rather than a single impulsive heating pulse.

**한국어.** SXR 플라즈마의 총 복사에너지(T-rad)는 *피크* 열량의 약 한 자릿수를 초과합니다. 수치적으로: T-rad 중앙값 ≈ 17 × 10^30 erg vs. Peak 중앙값 ≈ 2.4 × 10^30 erg, 비율 ≈ 7. 이는 SXR 플라즈마가 플레어 전 기간 동안 *지속적으로 재가열*됨을 증명합니다 — 점진 단계 동안 에너지가 여러 번 주입되고 복사됩니다. 이는 단일 충격 가열 펄스가 아닌 수 분 시간 규모의 지속적 재결합 주도 에너지 방출과 일관됩니다.

### Insight 2 — U_e + U_i ≳ E_bol (Particles Power Radiation) / 입자가 복사를 공급

**English.** The combined energy in flare-accelerated electrons and ions equals or exceeds the bolometric radiated energy. This is the "thick-target" picture validated statistically: nonthermal particle beams precipitate into the chromosphere, deposit their energy via collisional Coulomb losses, heat the chromosphere/transition region, drive chromospheric evaporation that fills coronal loops with hot plasma, which then radiates across all wavelengths (UV, EUV, SXR, optical white-light continuum). The energy budget closes: particles → radiation. Note: this only works because U_e is a lower limit; the true partition could place even more energy in nonthermal particles.

**한국어.** 플레어 가속 전자와 이온의 결합 에너지가 볼로메트릭 복사 에너지와 같거나 초과합니다. 이는 통계적으로 검증된 "두꺼운 표적" 그림입니다: 비열적 입자빔이 채층으로 강하해 충돌 쿨롱 손실로 에너지를 침적시키고 채층/전이층을 가열하며, 채층 증발을 일으켜 코로나 루프를 뜨거운 플라즈마로 채우고, 그 플라즈마가 전 파장(UV, EUV, SXR, 광학 백색광 연속복사)에 걸쳐 복사합니다. 에너지 예산이 닫힙니다: 입자 → 복사. 주의: U_e가 하한값이므로 가능; 실제 분배는 비열적 입자에 더 많은 에너지를 둘 수 있습니다.

### Insight 3 — U_e ≈ U_i (Electron-Ion Equipartition) / 전자-이온 등분배

**English.** Electron and ion energies are comparable, typically within factors of 2–3 — a remarkable result given they are measured by completely independent techniques (electrons: HXR bremsstrahlung continuum 20 keV–1 MeV; ions: 2.223 MeV gamma-ray neutron-capture line, integrating from ~20 MeV/nucleon). This rough equipartition implies a single dominant acceleration mechanism (likely magnetic reconnection–related) that energizes both species to comparable total energies, despite the very different individual particle energies (electrons ~30 keV, ions ~1 MeV — 30× difference per particle, but compensated by larger ion-energy reservoirs).

**한국어.** 전자와 이온 에너지가 비슷합니다 (대개 2–3배 이내) — 완전히 독립적인 방법으로 측정됨을 고려하면 주목할 결과입니다 (전자: HXR 제동복사 연속복사 20 keV–1 MeV; 이온: 2.223 MeV 감마선 중성자 포획선, ~20 MeV/nucleon에서 적분). 이 대략적 등분배는 두 종을 비슷한 총 에너지로 가속하는 단일 지배 메커니즘(자기 재결합 관련일 가능성)을 시사합니다. 개별 입자 에너지는 매우 다르지만(전자 ~30 keV, 이온 ~1 MeV — 입자당 30배 차이), 더 큰 이온 에너지 저장소로 보상됩니다.

### Insight 4 — U_SEP ≈ Few % × U_K^SW (Moderate Shock Efficiency) / 적절한 충격파 효율

**English.** SEP energy is typically a few percent (median ≈ 3%, range 0.1%–10%) of the CME kinetic energy *in the solar-wind rest frame* (which is the kinetic energy actually available for shock acceleration). This means CME-driven shocks are *moderately efficient* SEP accelerators — neither extremely efficient (≳50%) nor negligibly so (≪1%). The few-percent number is now a standard input for SEP transport modeling and space-weather forecasting. Outliers (e.g., Event #25 with high SEP, Event #6 with low) indicate variation in shock geometry, seed populations, or magnetic connection.

**한국어.** SEP 에너지는 *태양풍 정지 좌표계*에서 CME 운동에너지의 수 % (중앙값 ≈ 3%, 범위 0.1%–10%)입니다 (이것이 충격파 가속에 실제 가용한 운동에너지). 이는 CME 충격파가 *적절히 효율적*인 SEP 가속기임을 의미합니다 — 극단적으로 효율적이지도(≳50%) 거의 무시할 만하지도(≪1%) 않습니다. 수 % 수치는 이제 SEP 전파 모델링과 우주날씨 예보의 표준 입력값입니다. 이상치(예: 높은 SEP의 사건 #25, 낮은 사건 #6)는 충격파 기하, 씨앗 입자, 자기 연결의 변동을 나타냅니다.

### Insight 5 — E_mag is the Universal Reservoir / 자유 자기에너지가 보편적 저장소

**English.** The free magnetic energy E_mag is sufficient to power the *sum* of CME kinetic, electron, ion, and thermal energies. Typically E_mag ≳ 2 × (U_K + U_e + U_i + U_th), confirming the basic flare paradigm: stressed (current-carrying) magnetic fields store energy that is released via reconnection into the various downstream channels. No event in the sample violates this constraint, providing strong observational support for the flare standard model. This is *necessary* for energy conservation but does not by itself identify the energy-release mechanism.

**한국어.** 자유 자기에너지 E_mag는 CME 운동에너지, 전자, 이온, 열에너지의 *합*을 공급하기에 충분합니다. 일반적으로 E_mag ≳ 2 × (U_K + U_e + U_i + U_th)로, 기본 플레어 패러다임을 확증합니다: 응력 받은(전류 운반) 자기장이 에너지를 저장하고 재결합으로 다양한 하류 채널로 방출. 샘플의 어떤 사건도 이 제약을 위반하지 않으며, 플레어 표준 모델에 강한 관측적 근거를 제공합니다. 이는 에너지 보존상 *필요* 조건이지만 에너지 방출 메커니즘 자체를 식별하지는 않습니다.

### Insight 6 — Lower-Limit Nature of Energies / 에너지의 하한값 성격

**English.** Most quoted energies are *lower bounds*:
- U_e: chosen with largest E_min giving acceptable χ²; could be 10× higher
- U_i: spectral extrapolation 30 MeV → 1 MeV with assumed δ; could be 30× higher
- U_K, U_Φ: plane-of-sky underestimates by 2–8×
- T-rad: misses second-phase emission, may capture only 50%

The "true" energy partition could thus shift weights significantly toward particles and CME relative to thermal. Reading the paper's conclusions requires constantly remembering this systematic asymmetry.

**한국어.** 인용된 에너지 대부분이 *하한값*:
- U_e: χ² 만족 최대 E_min 선택; 10배까지 클 수 있음
- U_i: 가정된 δ로 30 MeV → 1 MeV 외삽; 30배까지 클 수 있음
- U_K, U_Φ: 천구면 가정으로 2–8배 과소평가
- T-rad: 후속 단계 방출 누락, 50%만 포착할 수 있음

따라서 "실제" 에너지 분배는 가중치를 상당히 입자와 CME 쪽으로 이동시킬 수 있습니다. 논문 결론을 읽을 때 이 체계적 비대칭을 항상 기억해야 합니다.

### Insight 7 — Bolometric Dominance of UV/Optical / UV/광학의 볼로메트릭 지배

**English.** Bolometric radiation is dominated by UV/visible/near-UV continuum, *not* SXR. Conversion factor 2.42 from FISM 1–1900 Å to bolometric implies ~58% of bolometric is in 1–1900 Å while another ~42% extends into visible/IR. The SXR contribution (T-rad) is typically only 10–20% of bolometric. This justifies why "white-light flares" exist and why TIM (a bolometric instrument) sees flare brightening at all — the visible continuum is a major energy channel often overlooked in flare models that focus on SXR.

**한국어.** 볼로메트릭 복사는 UV/가시광/근자외선 연속복사가 지배하며, SXR이 *아닙니다*. FISM 1–1900 Å에서 볼로메트릭으로의 변환 인자 2.42는 볼로메트릭의 ~58%가 1–1900 Å에 있고 다른 ~42%가 가시광/적외선으로 확장됨을 의미합니다. SXR 기여(T-rad)는 일반적으로 볼로메트릭의 10–20%에 불과합니다. 이는 "백색광 플레어"가 존재하고 TIM(볼로메트릭 측정기)이 플레어 밝기 증가를 감지하는 이유를 설명합니다 — 가시광 연속복사는 SXR에 집중하는 플레어 모델에서 종종 간과되는 주요 에너지 채널입니다.

### Insight 8 — Statistical Approach Reveals Outliers / 통계 접근이 이상치 노출

**English.** With 38 events, the paper can identify "outlier" events that fall outside the typical correlations:
- **Event #5** (behind-limb): bolometric ≥210 with no other measured components — useful "calibration" of upper-limit techniques
- **Event #25** (X7.1, 2005/01/20): unusually proton-rich (high U_i, high SEP)
- **Event #1** (M5.1, 2002/02/20): possibly flare-accelerated SEPs (Chollet+ 2010), distinct from CME-shock-accelerated SEPs

Outlier identification is a key advantage of the statistical approach over single-event studies — it points to physical regimes (e.g., shock geometry, magnetic connection) that *vary* across events and motivates targeted theoretical investigation.

**한국어.** 38개 사건으로 일반적 상관관계에서 벗어나는 "이상치" 사건을 식별 가능:
- **사건 #5** (배면): 다른 측정 성분 없이 볼로메트릭 ≥210 — 상한값 기법의 유용한 "보정"
- **사건 #25** (X7.1, 2005/01/20): 이례적 양성자 풍부 (높은 U_i, 높은 SEP)
- **사건 #1** (M5.1, 2002/02/20): 플레어 가속 SEP일 가능성 (Chollet+ 2010), CME 충격파 가속과 구별

이상치 식별은 단일 사건 연구 대비 통계 접근의 핵심 장점 — 사건 간 *변동*하는 물리 영역(예: 충격파 기하, 자기 연결)을 가리키며 표적 이론 조사를 유발합니다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Thermal Energy / 열에너지

$$U_{\text{th}} = 3 n_e k T_0 f V_{\text{ap}} \simeq 4.14 \times 10^{-16} T_0 \sqrt{EM \cdot f V_{\text{ap}}} \;\text{erg}$$

| Symbol | Meaning / 의미 | Units / 단위 | Source / 출처 |
|---|---|---|---|
| n_e | Electron density / 전자 밀도 | cm^-3 | Derived from EM, V_ap |
| k | Boltzmann constant / 볼츠만 상수 | erg/K | 1.38 × 10^-16 |
| T_0 | Plasma temperature / 플라즈마 온도 | K | RHESSI OSPEX fit |
| f | Filling factor / 충전 인자 | – | 1 (assumed) |
| V_ap | Apparent SXR volume / 겉보기 SXR 부피 | cm^3 | RHESSI 3σ-clean |
| EM | Emission measure / 방출 측도 | cm^-3 | RHESSI OSPEX fit |

The simplification uses n_e^2 V = EM and assumes complete ionization (so n_total ≈ 2n_e for fully ionized H). Numerical prefactor 4.14×10^-16 = 3 k × √2 with appropriate unit conversions.

단순화는 n_e^2 V = EM과 완전 이온화(완전 이온화 H에서 n_total ≈ 2n_e)를 사용. 수치 계수 4.14×10^-16 = 3 k × √2의 단위 변환.

### 4.2 Broken Power-Law Electron Spectrum / 꺾인 거듭제곱 전자 스펙트럼

$$F_0(E_0) = \begin{cases} 0 & E_0 < E_{\min} \\ A \left(\dfrac{E_0}{E_p}\right)^{-\delta_1} & E_{\min} \le E_0 < E_b \\ A \left(\dfrac{E_0}{E_p}\right)^{-\delta_2} \left(\dfrac{E_b}{E_p}\right)^{\delta_2-\delta_1} & E_b \le E_0 < E_{\max} \\ 0 & E_0 \ge E_{\max} \end{cases}$$

Total nonthermal electron energy:
$$U_e = \int_{E_{\min}}^{E_{\max}} E_0 \, F_0(E_0) \, dE_0 \times \tau \;\;[\text{erg}]$$

For δ_1 > 2 (typical: δ_1 ≈ 4–6):
$$U_e^{\text{below break}} \approx \frac{A E_p^{\delta_1} E_{\min}^{2-\delta_1}}{\delta_1 - 2} \cdot \tau$$

The strong dependence on E_min^(2-δ_1) is the source of order-of-magnitude uncertainty.

E_min^(2-δ_1)의 강한 의존성이 한 자릿수 불확실성의 원인.

### 4.3 CME Kinetic and Potential Energies / CME 운동·위치 에너지

$$U_K = \frac{1}{2} m V^2$$

$$U_\Phi = G M_\odot m \left( \frac{1}{R_\odot} - \frac{1}{r} \right)$$

| Symbol | Meaning / 의미 | Units / 단위 |
|---|---|---|
| m | CME mass (Thomson scattering) / CME 질량 | g |
| V | Speed (plane-of-sky) / 천구면 속도 | cm/s |
| G | Gravitational constant / 만유인력 상수 | 6.674 × 10^-8 cm^3/(g·s^2) |
| M_⊙ | Solar mass / 태양 질량 | 1.989 × 10^33 g |
| R_⊙ | Solar radius / 태양 반경 | 6.96 × 10^10 cm |
| r | Heliocentric distance at last LASCO frame / 마지막 LASCO 프레임의 일심거리 | cm |

Solar-wind frame: V_SW = V − 400 km/s.
태양풍 좌표계: V_SW = V − 400 km/s.

### 4.4 Bolometric Conversion / 볼로메트릭 변환

$$E_{\text{bol}} = 2.42 \times E_{1-1900\,\text{Å}}^{\text{FISM}} \times C_{\text{LD}}(\theta)$$

where C_LD is the limb-darkening correction, up to ~3 near the limb, ~1 at disk center.

여기서 C_LD는 변연 감광 보정으로, 변연 부근 ~3, 원반 중심 ~1.

### 4.5 Ion Energy from 2.223 MeV Line / 2.223 MeV 선에서 이온 에너지

$$U_i = 3 \times U_p^{>1\,\text{MeV}} = 3 \times U_p^{>30\,\text{MeV}} \times \left(\frac{30}{1}\right)^{\delta - 2} \;\text{(for } \delta = 4\text{)}$$

$$\Rightarrow U_i = 3 \times U_p^{>30\,\text{MeV}} \times 30^2 = 2700 \times U_p^{>30\,\text{MeV}}$$

This explicitly shows the 1.5-orders-of-magnitude lever-arm: factor of 30 in extrapolation, raised to power (δ-2).

이는 1.5 자릿수 외삽 거리의 효과를 명시적으로 보여줍니다: 외삽 인자 30이 (δ-2) 거듭제곱으로.

### 4.6 Worked Numerical Example / 작업 예시

**Event #6 (2002/07/23, X4.8) — the canonical case:**

| Quantity | Value (10^30 erg) | Calculation / 계산 |
|---|---|---|
| GOES SXR | 1.2 | Direct integration |
| T-rad | 19 | rad_loss × duration |
| Bolometric | 150 | TIM/FISM |
| Peak thermal | 2.5 | OSPEX, T = 17 MK, EM = 4×10^49 |
| Electrons | 32 | E_min ≈ 30 keV |
| Ions | 39 | δ = 4, 2.223 MeV line |
| CME KE (Sun) | 260 | m = 2 × 10^16 g, V ≈ 1500 km/s |
| CME KE (SW) | 150 | V_SW ≈ 1100 km/s |
| Pot. energy | 20 | Vourlidas formula |
| SEP | <30 | upper limit (poor coverage) |
| Mag energy | 2000 | active region B²V |

**Energy balance check / 에너지 균형 점검:**
- U_K + U_e + U_i + U_th = 260 + 32 + 39 + 2.5 ≈ 333 × 10^30 erg
- E_mag = 2000 × 10^30 erg
- Ratio E_mag / sum ≈ 6× → consistent with paradigm.
- E_bol / (U_e + U_i) = 150/71 ≈ 2.1 → particles do *not quite* power radiation here, but within factor-2 (and U_e is lower limit).

---

## 5. Paper in the Arc of History / 역사 속의 논문

### 5.1 Timeline / 연표

```
1972 ─── Hudson: first flare energy budget estimates / 최초 플레어 에너지 예산
   │
1992 ─── Doschek: turbulence/mass-motion energy comparable to thermal
   │     난류·질량운동 에너지가 열에너지와 비슷
   │
1997 ─── CHIANTI database (Dere+) — basis for rad_loss
   │     CHIANTI 데이터베이스 — rad_loss 기반
   │
2002 ─── RHESSI launch (Lin+) — enables reliable HXR + γ-ray spectroscopy
   │     RHESSI 발사 — HXR + γ선 분광 가능
   │
2003 ─── Taos workshop → Emslie+ 2004 method foundation
   │     Taos 워크샵 → Emslie+ 2004 방법론 기반
   │
2004 ─── Emslie et al.: 2 events (Apr 21, Jul 23) detailed energy budget
   │     2개 사건 상세 에너지 예산
   │
2005 ─── Emslie et al.: refined Jul 23 with optical/EUV components
   │     Jul 23 광학·EUV 정밀화
   │
2008 ─── Mewaldt+: SEP/CME ratio statistics, motivates extension
   │     SEP/CME 비율 통계
   │
2008 ─── Napa "Solar Activity Cycle 24 Onset" workshop → this paper
   │     Napa 워크샵 → 본 논문
   │
2010 ─── Vourlidas et al.: refined LASCO CME mass procedure
   │     LASCO CME 질량 절차 정밀화
   │
2012 ─── ★ Emslie et al.: 38-event ensemble study (this paper)
   │     ★ 38개 사건 앙상블 연구 (본 논문)
   │
2014 ─── Aschwanden et al.: revisit with HMI NLFFF for Cycle 24 events
   │     HMI NLFFF로 Cycle 24 사건 재분석
   │
2017 ─── Aschwanden et al.: Global Energetics 4-paper series, 173 events
   │     Global Energetics 4부작, 173개 사건
   │
2020s ── Solar Orbiter / PSP enable full-spectrum particle measurements
         Solar Orbiter / PSP가 전 스펙트럼 입자 측정 가능화
```

### 5.2 Significance / 의의

**English.** Emslie+ 2012 is the bridge between single-event case studies (Emslie+ 2004, 2005) and large-N statistical surveys (Aschwanden+ 2014–2017). It established the canonical numbers for flare energy budgets that became the reference point for all subsequent work. Its methodology — integrating GOES, RHESSI, LASCO, ACE, GOES-particle, SAMPEX, STEREO data per event — set the template for multi-instrument energy accounting. The paper's emphasis on *lower-limit* nature of estimates and explicit error budgeting raised methodological standards in the field. The 38-event sample remains a benchmark dataset re-analyzed multiple times since.

**한국어.** Emslie+ 2012는 단일 사건 사례 연구(Emslie+ 2004, 2005)와 대규모 N 통계 조사(Aschwanden+ 2014–2017)를 잇는 다리입니다. 이후 모든 연구의 참조점이 되는 플레어 에너지 예산의 표준 수치를 확립했습니다. GOES, RHESSI, LASCO, ACE, GOES 입자, SAMPEX, STEREO 데이터를 사건별로 통합하는 방법론이 다중 측정기 에너지 회계의 템플릿이 되었습니다. 추정치의 *하한값* 성격을 강조하고 명시적 오차 예산을 제시함으로써 분야의 방법론적 기준을 높였습니다. 38개 사건 샘플은 이후 여러 차례 재분석되는 벤치마크 데이터셋으로 남아 있습니다.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 관련성 | 한국어 |
|---|---|---|
| Hudson (1972) | First-generation flare energy budget concept | 1세대 플레어 에너지 예산 개념 |
| Brown (1973) | Thick-target bremsstrahlung (basis for U_e) | 두꺼운 표적 제동복사 (U_e 기반) |
| Doschek et al. (1992) | Turbulent + mass-motion energy in flares | 플레어의 난류·질량운동 에너지 |
| Lin et al. (2002) | RHESSI mission paper | RHESSI 임무 논문 |
| Emslie et al. (2004) | First single-event energy budget (Events #2, 6) | 최초 단일 사건 에너지 예산 (사건 #2, 6) |
| Emslie et al. (2005) | Refined Jul 23 event w/ optical/EUV | Jul 23 사건 광학·EUV 정밀화 |
| White et al. (2005) | GOES T, EM relations | GOES T, EM 관계식 |
| Woods et al. (2006, 2011) | TIM bolometric measurements | TIM 볼로메트릭 측정 |
| Chamberlin et al. (2007, 2008) | FISM model for non-TIM events | FISM 모형, 비-TIM 사건 |
| Mewaldt et al. (2008) | SEP/CME ratio statistical motivation | SEP/CME 비율 통계 동기 |
| Vourlidas et al. (2010, 2011) | LASCO CME mass/energy procedure | LASCO CME 질량·에너지 절차 |
| Su et al. (2011) | Second-phase contribution to T-rad | T-rad의 후속 단계 기여 |
| Guo et al. (2012) | Filling factor f = 0.20 ×/÷ 3.9 | 충전 인자 f = 0.20 ×/÷ 3.9 |
| Shih (2009), Shih et al. (2009) | RHESSI 2.223 MeV line catalog | RHESSI 2.223 MeV 선 목록 |
| Aschwanden et al. (2014, 2017) | NLFFF revisit, Global Energetics series | NLFFF 재분석, Global Energetics 시리즈 |

---

## 7. References / 참고문헌

- Emslie, A. G., et al. "Global Energetics of Thirty-Eight Large Solar Eruptive Events." *ApJ* 759, 71 (2012). DOI: 10.1088/0004-637X/759/1/71
- Emslie, A. G., et al. "Energy Partition in Two Solar Flare/CME Events." *J. Geophys. Res.* 109, A10104 (2004).
- Emslie, A. G., et al. "Refined Analysis of the 2002 July 23 Flare." *J. Geophys. Res.* 110, A11103 (2005).
- Mewaldt, R. A., et al. "How Efficient are Coronal Mass Ejections at Accelerating Solar Energetic Particles?" *AIP Conf. Proc.* 1039, 111 (2008).
- Vourlidas, A., et al. "Comprehensive Analysis of CME Mass and Energy Properties." *ApJ* 722, 1522 (2010).
- Vourlidas, A., et al. "How Many CMEs Have Flux Ropes? Deciphering the Signatures of Shocks, Flux Ropes, and Prominences in Coronagraph Observations of CMEs." *Sol. Phys.* (2011).
- Woods, T. N., et al. "Total Solar Irradiance Variations: Solar Cycle and Flare Contributions." *J. Geophys. Res.* 111, A10S14 (2006).
- Chamberlin, P. C., et al. "Flare Irradiance Spectral Model (FISM): Daily Component Algorithms and Results." *Space Weather* 5, S07005 (2007).
- Chamberlin, P. C., et al. "Flare Irradiance Spectral Model (FISM): Flare Component Algorithms and Results." *Space Weather* 6, S05001 (2008).
- Brown, J. C. "The Deduction of Energy Spectra of Non-Thermal Electrons in Flares from the Observed Dynamic Spectra of Hard X-Ray Bursts." *Sol. Phys.* 18, 489 (1973).
- White, S. M., et al. "Updated Expressions for Determining Temperatures and Emission Measures from GOES Soft X-Ray Measurements." *Sol. Phys.* 227, 231 (2005).
- Dennis, B. R., & Pernak, R. L. "Hard X-Ray Flare Source Sizes Measured with the RHESSI." *ApJ* 698, 2131 (2009).
- Guo, J., et al. "RHESSI Observations of the Filling Factor in Solar Flares." *ApJ* 755, 32 (2012).
- Lin, R. P., et al. "The Reuven Ramaty High-Energy Solar Spectroscopic Imager (RHESSI)." *Sol. Phys.* 210, 3 (2002).
- Aschwanden, M. J., et al. "Global Energetics of Solar Flares. I–IV." *ApJ* 797, 50 (2014); 836, 17 (2017); etc.
- Shih, A. Y., et al. "RHESSI Observations of the Proportional Acceleration of Relativistic >0.3 MeV Electrons and >30 MeV Protons in Solar Flares." *ApJ* 698, L152 (2009).
- Su, Y., et al. "Two-Stage Energy Release Process of a Confined Flare with Double HXR Peaks." *ApJ* 731, 106 (2011).
