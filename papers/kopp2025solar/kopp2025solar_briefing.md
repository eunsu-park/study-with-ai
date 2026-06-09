---
title: "Pre-Reading Briefing: Solar Irradiance Measurements"
paper_id: "85"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Solar Irradiance Measurements: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Kopp, G. "Solar Irradiance Measurements", *Living Reviews in Solar Physics*, 22:1 (2025). DOI: 10.1007/s41116-025-00040-5
**Author(s)**: Greg Kopp (LASP, University of Colorado Boulder)
**Year**: 2025

---

## 1. 핵심 기여 / Core Contribution

본 리뷰는 1978년 이후 거의 반세기에 걸친 우주 탑재 태양복사조도 (solar irradiance) 측정의 역사, 기기 설계, 자료 처리, 그리고 합성 (composite) 레코드를 포괄적으로 정리한다. 총복사조도 (Total Solar Irradiance, TSI)와 분광복사조도 (Spectral Solar Irradiance, SSI) 모두에 대해 측정 요구사항, 현재 역량, 그리고 지구-기후 시스템에 미치는 영향의 맥락에서 미래 개선 방향을 제시한다.

This review comprehensively organizes nearly five decades (since 1978) of space-borne solar-irradiance measurements — their history, instrument designs, data processing, and composite records. For both Total Solar Irradiance (TSI) and Spectral Solar Irradiance (SSI), it discusses measurement requirements, current capabilities, and directions for future improvements in the context of their effects on Earth's climate system. Kopp is the PI of the SORCE/TIM and TSIS-1/TIM instruments that established the currently accepted lower TSI value of 1361 W/m^2.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

18-19세기 Pouillet과 Herschel의 지상 기반 태양상수 측정에서 시작하여, 20세기 초 Abbot의 스미소니언 장기 관측, 1960-70년대 기구/로켓 실험을 거쳐 1978년 NIMBUS-7/ERB의 발사로 연속적 우주 기반 측정 시대가 열렸다. SORCE/TIM (2003)은 기존 값 (~1366 W/m^2) 보다 낮은 새로운 TSI 값 (1361 W/m^2)을 발표하여 학계 논쟁을 촉발했으나, 결국 커뮤니티가 이를 수용하게 되었다.

Starting from Pouillet's and Herschel's ground-based "solar constant" measurements in the 18-19th centuries, through Abbot's Smithsonian long-term program in the early 20th century, to balloon/rocket experiments of the 1960s-70s, the continuous space-based era began with the launch of NIMBUS-7/ERB in 1978. The SORCE/TIM (2003) announced a lower, new TSI value of 1361 W/m^2 (versus the prior ~1366 W/m^2), triggering community debate that was eventually resolved through the TRF (TSI Radiometer Facility) ground calibrations.

### 타임라인 / Timeline

```
1837 - Pouillet의 pyrheliometer, TSI ~1227 W/m^2 (after atmospheric correction)
1881 - Langley, Mt. Whitney, TSI ~2903 W/m^2 (erroneously high)
1902-1962 - Abbot / Smithsonian 지상 프로그램, TSI ~1357 W/m^2
1978 - NIMBUS-7/ERB 발사 (우주 탑재 TSI 레코드 시작)
1980 - SMM/ACRIM-1 (Willson)
1984-1987 - ACRIM Gap (Challenger 사고)
1996 - SoHO/VIRGO (L1, 최장기간 단일 기기)
2003 - SORCE/TIM, 새로운 낮은 TSI 값 1361 W/m^2 발표
2013 - TCTE/TIM (SORCE와 TSIS-1 사이 연결)
2017 - TSIS-1/TIM (ISS 탑재, 최고 정밀도)
2022+ - FY-3E (중국), CTIM (CubeSat)
2025+ - TSIS-2/TIM 발사 예정
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **복사측정학 (Radiometry)**: Watt per square meter, spectral radiance, blackbody emission, Stefan-Boltzmann law
- **태양 자기활동 (Solar magnetic activity)**: 흑점 (sunspots), 광반 (faculae), 11년 태양주기
- **기기 공학 (Instrument engineering)**: Electrical-substitution radiometer (ESR), bolometer, aperture, grating/prism 분광기
- **지구-복사 수지 (Earth energy balance)**: Albedo, greenhouse effect, radiative forcing
- **관련 선행 논문**: Paper #11 Haigh (2007) "The Sun and the Earth's Climate"
- **기초 통계/신호처리**: 장기 트렌드 검출, 계측기 degradation correction, 합성 시계열 (composite) 구성

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| TSI (Total Solar Irradiance) | 1 AU에서 모든 파장에 대해 공간/스펙트럼 적분된 태양 복사력 (W/m^2) / Spatially and spectrally integrated radiant power at 1 AU |
| SSI (Spectral Solar Irradiance) | 파장별 복사조도 (W/m^2/nm) / Wavelength-resolved irradiance |
| Solar constant | 과거 명칭, 실제로는 ~0.1% 변동 / Legacy term; actually varies ~0.1% |
| ESR (Electrical-Substitution Radiometer) | 전기열치환 복사계, TSI 측정의 표준 / Standard TSI detector comparing electrical and radiant heating |
| WRR (World Radiometric Reference) | 1977년 PMOD/WRC 설립, SI 기준보다 0.34% 높음 / 1977-established reference; 0.34% high compared to SI |
| PMOD composite | Fröhlich가 만든 TSI 합성 시계열 / PMOD/WRC TSI composite by Fröhlich |
| ACRIM composite | Willson의 TSI 합성 시계열 / TSI composite by Willson |
| ACRIM Gap | 1989-1992 ACRIM-1과 ACRIM-2 사이 공백 / 3-year gap between ACRIM-1 and ACRIM-2 |
| Faculae / Sunspots | 밝은/어두운 자기활동 영역 / Bright/dark magnetic surface features |
| SATIRE | Spectral And Total Irradiance REconstruction model / Semi-empirical irradiance model |
| Radiative forcing | 복사 강제력 (W/m^2), 기후 반응을 유도하는 양 / Climate-driving flux perturbation |
| TRF (TSI Radiometer Facility) | LASP의 지상 교정 시설 / LASP ground-calibration facility for TSI instruments |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 TSI 정의 / TSI definition
$$
S_0 = \frac{1}{A}\int_0^\infty \frac{dP}{d\lambda}\,d\lambda \approx 1361\ \mathrm{W\,m^{-2}}\quad\text{at 1 AU}
$$
수신 면적 $A$에 대한 파장 적분 플럭스. 현재 IAU 공인값 / Wavelength-integrated flux per unit area; current IAU-accepted value.

### 5.2 회색체 평형 온도 / Gray-body equilibrium temperature
$$
S\cdot e\pi R^2 = 4\pi R^2 \cdot e\sigma T^4 \;\;\Rightarrow\;\; T=\left(\frac{S}{4\sigma}\right)^{1/4}
$$
TSI $S$와 흡수율 $e$가 주어졌을 때 지구 평형 온도; Stefan-Boltzmann 상수 $\sigma$. / Earth equilibrium temperature from Stefan-Boltzmann balance.

### 5.3 태양 출력 (bolometric) / Solar photospheric bolometric output
$$
L_\odot = 4\pi (1\,\mathrm{AU})^2 \cdot S_0 \approx 3.828\times 10^{26}\ \mathrm{W}
$$
광구 플럭스 $6.293\times 10^7$ W/m^2, 유효온도 5772 K. / Photospheric bolometric output, effective $T_\mathrm{eff} = 5772$ K.

### 5.4 SSI ~ TSI 민감도 근사 (Kopp 2024) / SSI-TSI sensitivity
$$
\frac{\Delta SSI}{SSI} \approx \frac{5}{8\lambda}\cdot\frac{\Delta TSI}{TSI}\quad (\lambda\ \text{in microns})
$$
가시/NIR 영역에서 SSI 변동을 더 잘 알려진 TSI로부터 추정. / Estimate SSI variability from better-known TSI in visible/NIR.

### 5.5 측정 요구사항 / Measurement requirements
- Absolute accuracy: $< 0.01\%$
- Stability: $< 0.001\%\ \mathrm{year^{-1}}$

---

## 6. 읽기 가이드 / Reading Guide

1. **Section 1 (pp.3-8)**: 태양복사조도의 정의와 지구 에너지 수지 맥락. 1 AU 기하학, Stefan-Boltzmann 식, 온실효과 간략. / Definitions and Earth-energy-balance context.
2. **Section 2.1-2.3 (pp.8-18)**: 지상/기구/로켓 초기 측정사, 측정 요구사항, 우주 탑재 기기 설계 (ESR, aperture, filter/prism/grating, degradation tracking). / Pre-spacecraft history, requirements, instrument design.
3. **Section 2.4 (pp.18-42)**: NIMBUS-7/ERB, ACRIM-1/2/3, ERBS/ERBE, SoHO/VIRGO, SORCE/TIM, TSIS-1/TIM, CTIM 등 주요 TSI 기기 상세. 각 기기 교정, degradation, 잔존 문제. / Detailed walkthrough of TSI instruments.
4. **Section 2.5 (pp.42-71)**: SSI reference spectra (ATLAS, WHI, SOLSPEC, HSRS) 및 broad-range SSI 시계열 기기 (SBUV, SORCE/SOLSTICE, SORCE/SIM, TSIS-1/SIM). / SSI reference spectra and broadband SSI measurements.
5. **Section 3 (pp.71-78)**: ACRIM/PMOD/RMIB 합성 시계열, 커뮤니티 컨센서스 합성, SOLID SSI 합성. / TSI and SSI composite records.
6. **Section 4 (pp.78-89)**: 분, 시, 27-일 회전주기, 11년 주기, 세기 단위 변동. SATIRE 재구성, Maunder Minimum. / Variability across timescales; SATIRE reconstruction.
7. **Section 5 (pp.89-92)**: 미래 개선 (stability, accuracy, duration, precision) 및 향후 기기 (TRUTHS, CLARREO, Libera). / Future measurement improvements.

읽을 때 주목할 점: (1) 왜 SORCE/TIM이 기존 값을 0.35% 낮췄나? (2) PMOD vs ACRIM composite의 트렌드 차이와 그 의미. (3) SSI의 UV-가시-NIR 변동이 기후에 주는 다른 영향. / Focus on: the SORCE/TIM value shift, PMOD vs ACRIM composite debate, wavelength-dependent climate coupling.

---

## 7. 현대적 의의 / Modern Significance

이 리뷰는 IPCC 보고서 및 기후 모델에 입력되는 태양 강제력 (solar forcing)의 기초 자료 품질을 평가하는 핵심 참고문헌이다. 47년 연속 TSI 레코드는 기후 감도 (~0.1 K / W/m^2) 계산에서 태양 기여도를 정량화할 수 있게 해준다. 미래 TSIS-2, TRUTHS, CLARREO 같은 미션은 $<0.001\%/\text{yr}$ 안정도를 달성하여 장기 세기 단위 태양 변동 검출을 가능하게 할 것이다.

This review is a primary reference for evaluating the quality of solar-forcing inputs to IPCC reports and climate models. The 47-year continuous TSI record allows quantification of the solar contribution to climate sensitivity (~0.1 K per W/m^2). Future missions such as TSIS-2, TRUTHS, and CLARREO aim to achieve the $<0.001\%/\text{yr}$ stability needed to detect secular solar variability on centennial timescales — essential for disentangling solar from anthropogenic forcing.

Paper #11 Haigh (2007)에서 다룬 태양-기후 연결은 본 논문의 복사측정 기반 위에 구축되며, 본 논문은 그 측정 인프라의 최신 (2025) 상태를 제공한다.
The Sun-climate coupling discussed in Paper #11 Haigh (2007) rests on the radiometric foundation reviewed here; this paper provides the state of that infrastructure as of 2025.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
