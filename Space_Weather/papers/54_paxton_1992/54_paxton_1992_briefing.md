---
title: "Pre-Reading Briefing: Special Sensor Ultraviolet Spectrographic Imager (SSUSI): An Instrument Description"
paper_id: "54_paxton_1992"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# Special Sensor Ultraviolet Spectrographic Imager (SSUSI): An Instrument Description: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Paxton, L. J., Meng, C.-I., Fountain, G. H., Ogorzalek, B. S., Darlington, E. H., Gary, S. A., Goldsten, J. O., Kusnierkiewicz, D. Y., Lee, S. C., Linstrom, L. A., Maynard, J. J., Peacock, K., Persons, D. F., and Smith, B. E., "Special Sensor Ultraviolet Spectrographic Imager (SSUSI): An Instrument Description", Proc. SPIE 1745, Instrumentation for Planetary and Terrestrial Atmospheric Remote Sensing, 2-15, 1992. DOI: 10.1117/12.60595
**Author(s)**: Larry J. Paxton, Ching-I. Meng et al. (Johns Hopkins University Applied Physics Laboratory)
**Year**: 1992

---

## 1. 핵심 기여 / Core Contribution

이 논문은 미국 국방기상위성계획 (DMSP) Block 5D3 위성 (S-16부터 S-19)에 탑재될 **Special Sensor Ultraviolet Spectrographic Imager (SSUSI)**의 설계와 운영 개념을 종합적으로 기술한다. SSUSI는 (1) 115-180 nm 원자외선 (FUV) 영역에서 horizon-to-horizon 횡단궤도 스캔을 수행하는 이미징 분광기 (SIS)와, (2) 4278 Å, 6300 Å airglow와 6300 Å 부근 지구 알베도를 측정하는 천저 광도계 시스템 (NPS)으로 구성된 운영용 우주기상 관측 기기이다. 본 논문은 SSUSI의 과학 목표, 광학·전기적 설계, 검출기 특성, 보정 계획을 포괄적으로 정의함으로써 현대 운영용 우주기상 원격탐사의 설계 전범 (canonical reference)이 된다.

This paper provides a comprehensive engineering description of the **Special Sensor Ultraviolet Spectrographic Imager (SSUSI)** scheduled to fly on the DMSP Block 5D3 satellites (S-16 through S-19). SSUSI consists of (1) a **Scanning Imaging Spectrograph (SIS)** that performs horizon-to-horizon cross-track scans in the far-ultraviolet (FUV) range 115-180 nm, and (2) a **Nadir Photometer System (NPS)** that observes airglow at 4278 Angstroms and 6300 Angstroms together with terrestrial albedo near 6300 Angstroms. By defining SSUSI's scientific objectives, optical and electrical design, detector characteristics, and calibration plan, the paper establishes itself as the canonical engineering reference for operational FUV remote sensing of the upper atmosphere and aurora.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1980년대 후반, 우주 기반 원자외선 (FUV) 관측은 단순한 분광 식별 (line identification)에서 정량적 환경 인자 추출 (quantitative environmental parameter retrieval)로 패러다임이 전환되었다. Strickland, Meier 등의 복사 전달 모델 진보로 FUV airglow 강도로부터 O/N₂ 비, F-region 전자밀도 피크, 강수 전자 에너지 플럭스를 계산할 수 있게 되었다. NASA Dynamics Explorer-1 (1981)의 Spin-scan Auroral Imager (SAI)와 POLAR UVI (1996) 같은 과학 임무가 가능성을 입증한 직후, DMSP는 이를 **운영용 (operational)** 미션으로 전환하기로 결정했고, APL (JHU/APL)이 SSUSI 4기를 제작하는 책임을 맡았다.

In the late 1980s, space-based FUV observations underwent a paradigm shift from mere spectral feature identification to **quantitative environmental parameter retrieval**. Advances in radiative transfer modelling by Strickland, Meier and colleagues enabled the inversion of FUV airglow intensities into O/N₂ ratios, F-region peak electron densities, and auroral precipitation energy flux/characteristic energy. Following the proof-of-concept by science missions such as NASA's Dynamics Explorer-1 Spin-Scan Auroral Imager (1981) and the upcoming POLAR UVI (1996), DMSP decided to transition this capability to an **operational** mission, with JHU/APL contracted to build four SSUSI flight units.

### 타임라인 / Timeline

```
1972 -- DMSP Block 5D OLS first flight (visible/IR operational imagery)
1981 -- DE-1 SAI: first global FUV auroral imaging from space
1982 -- HILAT/POLAR BEAR auroral imagers (DoD precursor missions)
1984 -- Meng & Huffman: UV imaging of aurora under sunlight (Ref 12)
1987 -- Meng & Huffman: HILAT auroral remote sensing (Ref 13)
1991 -- Strickland/Meier/Paxton: FUV inversion algorithms mature (Refs 3-9)
1992 -- *** Paxton et al. SSUSI instrument description (THIS PAPER) ***
1995 -- POLAR UVI launch (heritage to SSUSI calibration)
2003 -- DMSP F-16 (S-16) launch with first SSUSI flight unit
2006 -- DMSP F-17 SSUSI launch
2009 -- DMSP F-18 SSUSI launch
2014 -- DMSP F-19 SSUSI launch (final unit; F-19 lost 2016)
2014+ -- SSUSI EDR products feed OVATION-Prime, AMIE, GAIM nowcasts
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **FUV airglow physics / FUV 대기광 물리**: Lyman-alpha (121.6 nm), OI 130.4 nm 공명선 (resonance line), OI 135.6 nm intercombination line, N₂ Lyman-Birge-Hopfield (LBH) bands at 140-180 nm. 각 발광 메커니즘 (photoelectron impact, fluorescence, dissociative recombination)에 대한 기본 이해가 필요하다.
- **Spectrograph optics / 분광기 광학**: Rowland circle geometry, off-axis parabolic telescope, toroidal grating, slit-width vs spectral resolution trade-off.
- **Photon-counting detectors / 광자계수 검출기**: Microchannel plate (MCP) Z-stack, wedge-and-strip anode position-sensitive detector, CsI photocathode, MgF₂ entrance window.
- **Satellite remote sensing geometry / 위성 원격 탐사 기하학**: 830 km circular sun-synchronous polar orbit, cross-track scanning, ground footprint vs scan-angle relation, limb tangent altitude.
- **Operational space weather context / 운영 우주기상 맥락**: DMSP heritage (OLS, SSM/I, SSJ/4, SSIES), what "operational" means (24/7 data delivery, latency, redundancy).
- **Rayleigh as airglow brightness unit / 단위로서의 Rayleigh**: 1 R = 10⁶ photons cm⁻² s⁻¹ column-emission rate, viewed at 4π sr.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| SSUSI | Special Sensor Ultraviolet Spectrographic Imager — DMSP 운영용 FUV 원격탐사 기기 / DMSP operational FUV remote-sensing instrument |
| SIS | Scanning Imaging Spectrograph — horizon-to-horizon 분광 이미징 / horizon-to-horizon spectral imaging subassembly |
| NPS / NPD | Nadir Photometer System — 천저 방향 가시광 광도계 (427.8/630/629.4 nm) / nadir-viewing visible photometers |
| FUV | Far Ultraviolet (115-180 nm) — 대기 흡수와 발광이 강한 영역 / spectral region with strong atmospheric absorption and emission |
| Five colors | 다섯 파장 채널 (121.6, 130.4, 135.6, 140-150, 165-180 nm) / five FUV wavelength bands downlinked |
| LBH bands | N₂ Lyman-Birge-Hopfield bands (140-180 nm), O₂ 흡수에 영향받음 / N₂ LBH bands modulated by O₂ Schumann-Runge absorption |
| Super pixel | 지상 처리에서 공간 픽셀을 합쳐 SNR 확보 / coadded ground-processed spatial bin (200x200 km day, 30x400 km auroral) |
| Rayleigh (R) | $4\pi \times 10^{-6}$ photons cm⁻² s⁻¹ sr⁻¹ — column emission rate / column emission brightness unit |
| Wedge-and-strip anode | MCP 후단의 위치 민감 양극 (3 전극) / 3-electrode position-sensitive anode behind MCP |
| MCP Z-stack | 마이크로채널 플레이트 3장 적층 (이득 ~10⁶) / three-plate MCP stack giving ~10⁶ gain |
| CsI photocathode | FUV 양자효율 ~10% @ 135 nm / FUV photocathode with ~10% QE at 135 nm |
| GLOB | Glare Obstructor — 산란광 차단판 / sunlight glare obstructor on most DMSP slots |
| OCF | Optical Calibration Facility (APL) — SSUSI 보정 시설 / APL's optical calibration facility |
| EDR | Environmental Data Record — 운영용 처리된 환경 인자 / operational geophysical product |
| OLS | Operational Linescan System — DMSP 가시/적외 이미저 / DMSP visible-IR imager (heritage) |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Rayleigh-to-count rate sensitivity / Rayleigh - 계수율 환산**

$$
C\,[\text{counts/s}] \;=\; S_\lambda\,[\text{counts/s/R}] \;\times\; I\,[\text{R}]
$$

각 픽셀의 계수율은 감도 S와 입력 광도 I의 곱이다. 예: 130.4 nm에서 $S = 0.120$ → 1000 R airglow는 120 cps를 만든다.

The pixel count rate is the product of sensitivity $S_\lambda$ and Rayleigh brightness $I$. Example: at 130.4 nm with $S = 0.120$, an airglow of 1000 R produces 120 counts/s.

**(2) Cross-track scan geometry / 횡단궤도 스캔 기하학**

$$
\theta_{\text{nadir}} \in [-72.8^\circ, +61.6^\circ]\quad\text{(no GLOB)}
$$
$$
\theta_{\text{nadir}} \in [-63.2^\circ, +42.4^\circ]\quad\text{(GLOB present)}
$$

Earth-viewing scan은 nadir 기준 비대칭 각도 범위에서 0.8° 스텝으로 156 (또는 132) 픽셀을 만든다.

The Earth-viewing portion of the scan covers the listed nadir angles in 0.8° steps, yielding 156 pixels (or 132 with GLOB).

**(3) Limb tangent altitude / 림 접선 고도**

$$
h_t = \sqrt{(R_E + h_{sat})^2 - R_E^2 \cdot \cos^2(\theta_{\text{horizon}})} - R_E
$$

위성 고도 $h_{sat} = 830$ km, 지구 반지름 $R_E = 6378$ km, $\theta_{\text{horizon}}$ 시준에서 약 520 km 림 위 (-72.8°에서) 접선이 형성된다.

For $h_{sat}=830$ km and $R_E=6378$ km, the geometric tangent altitude at the -72.8° start-of-scan reaches approximately 520 km above the limb.

**(4) Detector saturation criterion / 검출기 포화 조건**

$$
\sum_{\lambda} C_\lambda < 200\,\text{kHz}\quad(\text{maximum input rate})
$$

모든 파장에 대한 총 계수율이 검출기 최대 (200 kHz)를 초과하면 안 된다. 예상 피크는 ~130 kHz.

The summed count rate across all wavelengths must remain below the 200 kHz MCP detector limit; the expected peak is around 130 kHz.

**(5) Footprint / Pixel size / 픽셀 크기**

$$
\Delta x_{\text{cross}} = h_{sat}\,\tan(\Delta\theta_{\text{cross}})/\cos^2(\theta_{\text{nadir}})
$$

천저각 $\theta$가 커질수록 픽셀 footprint는 빠르게 확대된다 (비선형 팽창).

The cross-track footprint dimension grows nonlinearly with off-nadir angle.

---

## 6. 읽기 가이드 / Reading Guide

- **Section 1 (Scientific Objectives, p.2-3)**: 관측 목표, 5개 "color" 채널 선택 동기, Table 1의 Day/Night/Auroral 각 영역 최대·최소 강도를 주의 깊게 읽고 sensitivity 요구가 합리적임을 확인하라.
- **Section 2 (System Description, p.3-9, Figs. 1-4)**: SIS는 imaging mode와 spectrograph mode 두 가지를 가진다. Figure 2/3/4의 스캔 기하학과 픽셀 형성 방식을 시간을 들여 이해할 것.
- **Section 3 (SIS Detector, Table 4)**: MCP + wedge-and-strip anode의 동작 원리와 4×10⁶ 평균 이득의 의미를 이해하라.
- **Section 4 (NPD, Tables 5-6)**: 왜 6300 nm에 두 개의 검출기 (signal 630.0 / background 629.4)가 필요한지 - 알베도 제거 (albedo subtraction)의 논리.
- **Section 5 (Calibration)**: 지상 OCF (deuterium DS-775, Hanovia 901-B1, NIST 추적 기준 검출기)와 비행 중 (HST 표준성 G191-B2B, lunar radiance + LOWTRAN 7) 보정 전략의 차이.

---

## 7. 현대적 의의 / Modern Significance

- SSUSI는 **운영용 우주기상**의 표준 관측 자산이 되었다. 2003년 F-16부터 시작해 F-17 (2006), F-18 (2009)이 현재까지 가동 중이며, 산출물은 OVATION-Prime auroral oval model, AMIE 동기화 동맹, USAF 557th Weather Wing의 통합 우주기상 nowcast에 입력된다.
- SSUSI 데이터는 **F-region 전자밀도 (NmF2)**와 **O/N₂ 비**의 글로벌 지도를 제공하며, GPS 통신 신호 신틸레이션 예보, HF 통신, 위성 항력 모델링에 직접 사용된다.
- TIMED-GUVI (2002), DMSP-SSUSI (2003-), GOLD (2018), ICON (2019) 등의 후속 FUV 임무들은 SSUSI 설계를 직접 계승했다. Paxton 본인이 GUVI PI를 겸직했다.
- 이 논문이 정의한 "5-color FUV" 접근은 데이터 다운링크 대역폭과 정보량을 절충하는 방식의 모범 사례로, 후속 운영 미션 (ICON FUV, GOLD)의 채널 선정에 영향을 주었다.

SSUSI became the cornerstone of operational space-weather remote sensing. Beginning with DMSP F-16 in 2003 and continuing through F-17 (2006) and F-18 (2009), its data products feed the OVATION-Prime auroral oval model, AMIE assimilation, and the USAF 557th Weather Wing's integrated space-weather nowcasts. SSUSI provides global maps of F-region peak electron density (NmF2) and thermospheric O/N₂ ratio that are directly used for GPS-scintillation forecasting, HF-communication assessment, and satellite-drag modelling. Subsequent FUV missions—TIMED-GUVI (2002), GOLD (2018), ICON (2019)—inherit SSUSI's optical and operational concepts (Paxton himself led GUVI). The paper's "five colors" downlink-bandwidth/information trade-off has become a textbook example of operational-mission channel selection.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
