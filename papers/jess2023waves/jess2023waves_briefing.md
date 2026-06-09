---
title: "Pre-Reading Briefing: Waves in the Lower Solar Atmosphere — The Dawn of Next-Generation Solar Telescopes"
paper_id: 80
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Waves in the Lower Solar Atmosphere: The Dawn of Next-Generation Solar Telescopes — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Jess, D. B., Jafarzadeh, S., Keys, P. H., Stangalini, M., Verth, G., & Grant, S. D. T. (2023). *Waves in the Lower Solar Atmosphere: The Dawn of Next-Generation Solar Telescopes.* Living Reviews in Solar Physics, 20:1. DOI: 10.1007/s41116-022-00035-6
**Author(s)**: David B. Jess, Shahin Jafarzadeh, Peter H. Keys, Marco Stangalini, Gary Verth, Samuel D. T. Grant
**Year**: 2023

---

## 1. 핵심 기여 / Core Contribution

This 170-page Living Review is a two-fold comprehensive synthesis of wave activity in the photosphere and chromosphere. First, it is a pedagogical handbook of wave analysis techniques (Fourier, wavelet, EMD/POD/DMD, k-ω, B-ω, phase lag) designed for early-career researchers who must extract oscillatory signatures from modern high-cadence, high-resolution data. Second, it reviews a decade of discoveries on global p-modes, large-scale magnetic structures (sunspots, pores), and small-scale magnetic elements (MBPs, fibrils, spicules), bridging disparate sub-fields that prior reviews (Jess+2015 chromosphere; Khomenko & Collados 2015 sunspots) covered only in isolation. The review closes by setting an agenda for the DKIST/Sunrise-III/Solar-C era, explicitly highlighting which open questions the new facilities can answer.

이 170쪽 분량의 Living Review는 광구-채층 파동 활동에 대한 두 갈래의 종합 리뷰이다. 첫째, Fourier·wavelet·EMD/POD/DMD·k-ω·B-ω·phase-lag 등 현대 고해상도 데이터에서 진동 신호를 추출하는 분석 기법들을 초기 연구자용 핸드북 형식으로 집대성했다. 둘째, 지난 10년간 축적된 발견들 — 전역 p-mode, 대규모 자기 구조물(흑점·포어) 내 파동, 소규모 자기요소(MBP·fibril·spicule) — 을 통합적으로 다룸으로써, 기존 리뷰들(채층만 다룬 Jess+2015, 흑점만 다룬 Khomenko & Collados 2015)이 개별적으로 포착했던 영역들을 하나의 일관된 틀 안에서 연결한다. 마지막 장은 DKIST·Sunrise-III·Solar-C 시대의 연구 의제를 설정하며, 신규 시설이 해결해야 할 열린 질문들을 명시한다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

Wave activity in the solar atmosphere was first reported by Leighton (1960) with the discovery of the ubiquitous 5-minute oscillation, later identified as the global p-mode spectrum (Ulrich 1970; Deubner 1975). The 3-minute chromospheric oscillation in sunspots (Beckers & Tallant 1969) and running penumbral waves (Zirin & Stein 1972) followed. The transition from purely acoustic to MHD interpretations occurred through the 1970s–1980s with Edwin & Roberts (1983) delivering the canonical magnetic cylinder dispersion diagram. The AC/DC dichotomy of coronal heating (Schwarzschild 1948) reframed wave research as a contest for the energy budget. By the 2010s, ground-based adaptive optics + speckle/MOMFBD reconstruction (Wöger+2008) and space-borne Hinode/SDO/IRIS data had pushed wave diagnostics into the sub-arcsecond, sub-minute regime. The review is written at the threshold of the DKIST era (first light 2019) and catalogues what remains to be done.

태양 대기 파동은 Leighton(1960)이 5분 진동을 발견하며 시작됐고, Ulrich(1970)·Deubner(1975)의 k-ω 다이어그램을 통해 전역 p-mode로 재해석됐다. 흑점 3분 진동(Beckers & Tallant 1969)과 running penumbral wave(Zirin & Stein 1972)가 뒤따랐고, 1970-80년대에 순수 음파 해석에서 MHD 해석으로의 전환이 이뤄졌다 — Edwin & Roberts(1983)가 자기 실린더 분산 관계를 정식화했다. Schwarzschild(1948)의 AC/DC 코로나 가열 이분법은 파동 연구를 에너지 예산 경쟁의 장으로 재규정했다. 2010년대에 지상 AO+speckle/MOMFBD(Wöger+2008)와 Hinode/SDO/IRIS 우주관측이 파동 진단을 sub-arcsec·sub-minute 영역으로 확장했다. 본 리뷰는 DKIST(2019 first light)의 문턱에서 지난 성과를 정리하고 향후 과제를 제시한다.

### 타임라인 / Timeline

```
1942 ─ Alfvén: MHD waves predicted
1948 ─ Schwarzschild: AC heating hypothesis
1960 ─ Leighton: 5-min oscillation discovered
1969 ─ Beckers & Tallant: umbral flashes (3-min)
1970 ─ Ulrich: p-modes as resonant cavity modes
1972 ─ Zirin & Stein: running penumbral waves
1975 ─ Deubner: k-ω ridges resolve p-mode spectrum
1981 ─ Spruit: thin flux tube theory
1983 ─ Edwin & Roberts: magnetic cylinder dispersion
2006 ─ Hinode launch; spicule/wave revolution
2010 ─ SDO/AIA/HMI launch
2013 ─ IRIS launch
2015 ─ Jess+2015 chromosphere review; Khomenko & Collados 2015 sunspots
2019 ─ DKIST first light (Haleakalā)
2020 ─ Solar Orbiter launch (PHI instrument)
2023 ─ Jess+2023 (this review)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**Mathematics / 수학**
- Fourier transform and power spectral density; wavelet transform (Morlet, Paul). Essential for Sect. 2.
- Partial differential equations; dispersion relations for linear waves.
- Cylindrical geometry and Bessel functions (for magnetic cylinder eigenmodes in Sect. 2.9.2).
- Fourier 변환과 파워 스펙트럼, wavelet 변환이 2절 전반의 도구이다. 실린더 좌표계에서의 Bessel 함수가 2.9.2절 자기 실린더 고유모드에 등장한다.

**Physics / 물리**
- Magnetohydrodynamics basics: induction equation, frozen-in condition, Alfvén speed, sound speed, plasma-β.
- Three MHD wave modes in a homogeneous unbounded plasma: slow, fast, Alfvén.
- Acoustic cutoff frequency and vertical stratification of the solar atmosphere.
- Mode conversion/transmission across the β=1 equipartition layer.
- MHD 기본: 유도 방정식, 얼어붙음 조건, Alfvén 속도, 음속, plasma-β. 균질·무경계 플라스마의 세 MHD 모드(slow·fast·Alfvén), 음향 차단 주파수와 대기의 수직 층 구조, β=1 등분배층의 모드 변환이 필수이다.

**Solar physics / 태양물리**
- Structure of lower atmosphere: photosphere (T~5800 K), temperature minimum (~4400 K), chromosphere (~10,000 K), canopy.
- Spectral diagnostics: Fe I 6301, Hα, Ca II 8542, Ca II H/K, Mg II h/k, Na I D.
- Magnetic structures: sunspot umbra/penumbra, pores, magnetic bright points (MBPs), plage, network/internetwork.
- 하층 대기 구조 (광구·온도 최소층·채층·캐노피), 분광 진단선, 흑점 umbra/penumbra, pore, MBP, plage, network/internetwork의 이해가 요구된다.

**Instrumentation / 기기**
- Ground-based: DST, SST, GREGOR, DKIST (4-m). Post-focus instruments: HARDcam, ROSA, CRISP, IBIS, ViSP, VBI, DL-NIRSP.
- Space-borne: Hinode/SOT, SDO/AIA/HMI, IRIS, Sunrise (balloon), Solar Orbiter/PHI.
- 지상 DST·SST·GREGOR·DKIST 및 후초점 HARDcam·ROSA·CRISP·IBIS·ViSP·VBI·DL-NIRSP. 우주 Hinode·SDO·IRIS·Sunrise·Solar Orbiter.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Acoustic cutoff ω_c | Frequency below which acoustic waves become evanescent in a stratified atmosphere; ω_c = c_s/(2H) for isothermal case, ~5 mHz photospheric. / 층 구조 대기에서 음파가 증발형이 되는 임계 주파수, 등온 근사 시 ω_c = c_s/(2H), 광구 ~5 mHz. |
| Plasma-β | Ratio of gas to magnetic pressure; β=2μ₀p/B² (SI) or 8πn_H k_B T/B² (cgs). Controls whether acoustic or magnetic forces dominate. / 기체압/자기압 비, 음향·자기압 우열을 결정. |
| Slow magnetoacoustic | MHD mode with pressure/magnetic restoring forces in anti-phase; acoustic-like in low-β. / 압력·자기압 복원력이 반위상, 저β에서 음파적. |
| Fast magnetoacoustic | MHD mode with pressure/magnetic pressures in phase; nearly isotropic. / 압력·자기압 동위상, 거의 등방성. |
| Alfvén wave | Incompressible transverse wave; magnetic tension restoring force; speed v_A = B/√(μ₀ρ). / 비압축성 횡파, 자기 장력 복원력. |
| Kink (m=1) mode | Transverse axi-asymmetric flux tube oscillation; displaces tube axis. / 자속관 축 변위, m=1. |
| Sausage (m=0) mode | Axi-symmetric compressive oscillation; periodic expansion/contraction of cross-section. / m=0 축대칭 압축, 단면 주기 변화. |
| Torsional Alfvén wave | Twisting motion along flux tube axis; m≥0; incompressible. / 자속관 축 비틀림, 비압축성. |
| Umbral flash | Chromospheric brightening caused by upward-propagating slow MHD shocks in umbra; ~3-min period. / 흑점 umbra에서 상승하는 slow MHD 충격파의 채층 밝기 증가, ~3분 주기. |
| Running penumbral wave | Outward-propagating wave train across sunspot penumbra in chromosphere; apparent phase speed 10–40 km/s. / 흑점 penumbra 채층에서 바깥으로 퍼지는 파동열. |
| Mode conversion | Change of wave character (fast↔slow) across β=1 equipartition layer. / β=1 등분배층에서 fast↔slow 변환. |
| Ramp effect | Lowering of effective cutoff frequency by inclined magnetic fields; allows <5 mHz waves to propagate. / 자기장 경사에 의한 차단 주파수 저하. |
| k-ω diagram | 2D power spectrum in wavenumber × frequency; reveals p-mode ridges and trapping regimes. / 파수-주파수 2D 파워, p-mode 능선 가시화. |
| Wave energy flux | F = ρ ⟨v²⟩ c_ph (or c_A for Alfvén); transported power per area. / 단위면적당 파동 전력. |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Acoustic cutoff (isothermal) / 음향 차단 (등온)**
$$\omega_c = \frac{c_s}{2H}, \quad H = \frac{k_B T}{m \bar{g}}$$
Below ω_c, acoustic waves cannot propagate upward; they are evanescent. Photospheric values (T~5800 K, H~150 km) give ω_c/(2π) ≈ 5 mHz. / ω_c 미만 주파수는 상승 전파 불가, 광구 값은 ~5 mHz.

**(2) Dispersion relation (isothermal atmosphere) / 분산 관계**
$$\omega^2 = c_s^2 k_z^2 + \omega_c^2$$
The 5-min solar p-mode (~3 mHz) sits below this cutoff and is trapped inside the Sun. / 5분 p-mode(~3 mHz)는 차단 미만으로 태양 내부에 갇힘.

**(3) Plasma-β / 플라스마 베타**
$$\beta = \frac{2\mu_0 p_0}{B_0^2} = \frac{8\pi n_H k_B T}{B_0^2}$$
Sunspot umbrae (B~3000 G) have β≪1 at photosphere; quiet Sun intergranular lanes can have β>1. / 흑점 umbra β≪1, 조용한 태양 입계선 β>1.

**(4) Tube speed / 관속도**
$$c_T = \frac{c_0 v_A}{\sqrt{c_0^2 + v_A^2}}$$
Lower bound of the slow band for magnetic cylinder trapped modes (Edwin & Roberts 1983). / 자기 실린더 slow 대역의 하한 속도.

**(5) Mode transmission coefficient (fast→slow) / 모드 투과 계수**
$$T = \exp\!\left(-\pi k h_s \sin^2\alpha\right)$$
where α is attack angle between k and B at β=1 layer; T+|C|=1 (energy conservation). / α는 k와 B의 사이각, 에너지 보존 T+|C|=1.

---

## 6. 읽기 가이드 / Reading Guide

- **Sect. 1 (Introduction, pp. 1–8)**: skim — historical narrative; note AC/DC heating dichotomy. / 역사 서술, AC/DC 이분법만 체크.
- **Sect. 2 (Wave analysis tools, pp. 8–67)**: the heart of the "pedagogical" part. Read 2.2–2.3 (Fourier), 2.4 (wavelets), 2.7 (k-ω), 2.8 (resolution), 2.9 (MHD modes in cylinder) carefully. The HARDcam/SuFI case studies (2.1.1, 2.1.2) recur throughout Sect. 2. / 2.2-2.3, 2.4, 2.7-2.9를 정독. 2.1.1·2.1.2 관측 사례가 반복 인용됨.
- **Sect. 3 (Recent studies, pp. 68–117)**: the "review" part. 3.1 global p-modes, 3.2 large-scale structures (sunspots/pores), 3.3 eigenmodes, 3.4 small-scale structures (MBPs/fibrils/spicules). Read selectively based on interest. / 관심에 따라 선별 독해.
- **Sect. 4 (Future directions, pp. 117–128)**: must-read for open questions; covers DKIST first-light instruments, Sunrise-III, Solar-C, FRANCIS. / 열린 질문과 신규 시설 필독.
- **Sect. 5 (Conclusions) + WaLSA team**: note https://www.WaLSA.team/ code repository. / WaLSA 공개 코드 저장소 확인.

**Figures to study / 주요 그림**
Fig. 1 (DST vs DKIST construction), Fig. 36 (NC5 flux tube wave speeds), Fig. 37–38 (magnetic cylinder dispersion), Fig. 40–41 (elliptical eigenmodes), Fig. 50–53 (sunspot mode conversion and resonator), Fig. 56 (body vs surface modes in pores), Fig. 67 (FRANCIS fibre ferrule).

---

## 7. 현대적 의의 / Modern Significance

For the DKIST/Solar-C era, this review is the field's standard reference on (1) how to analyse the torrent of 4-m high-cadence data, (2) which wave-heating questions are scientifically tractable, and (3) which diagnostics (phase-lag, B-ω, POD/DMD) can disambiguate mode conversion from multi-mode superposition. Its emphasis on "Level 2" data products, reproducible analysis via the WaLSA code repository, and explicit methodological pitfalls (Sect. 2.2.1 "common misconceptions") directly shape how PhD students approach wave studies today. The paper also reframes the photosphere–chromosphere coupling problem: rather than treating photospheric drivers, chromospheric cavities, and coronal heating separately, the lower atmosphere is cast as a single, continuously β-varying MHD waveguide where mode conversion and resonance set the energy throughput to the corona.

DKIST·Solar-C 시대에 본 리뷰는 (1) 4-m급 고시간분해능 데이터 분석법, (2) 과학적으로 해결 가능한 파동-가열 질문의 선정, (3) 모드 변환과 다중 모드 중첩을 구별할 진단법(phase-lag·B-ω·POD/DMD)의 표준 참고서 역할을 한다. "Level 2" 데이터 저장소 제안, WaLSA 공개 코드, 2.2.1절의 흔한 오해 정리는 오늘날 박사 과정생의 파동 연구 방법을 직접 규정한다. 또한 본 리뷰는 광구-채층 연계 문제를 재구성한다 — 광구 구동자·채층 공명공동·코로나 가열을 분리하지 않고, 하층 대기 전체를 β가 연속적으로 변하는 단일 MHD 도파관으로 보고 모드 변환과 공명이 코로나로의 에너지 유입을 좌우한다는 관점을 확립했다.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
