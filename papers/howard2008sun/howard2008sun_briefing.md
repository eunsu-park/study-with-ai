---
title: "Pre-Reading Briefing: Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI)"
paper_id: "50_howard_2008"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: R. A. Howard, J. D. Moses, A. Vourlidas, et al., "Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI)", *Space Science Reviews* 136, 67–115 (2008). DOI: 10.1007/s11214-008-9341-4
**Author(s)**: Russell A. Howard et al. (consortium of NRL, LMSAL, GSFC, Univ. Birmingham, RAL, MPS, CSL, IOTA, IAS)
**Year**: 2008

---

## 1. 핵심 기여 / Core Contribution

이 논문은 NASA의 STEREO (Solar Terrestrial Relations Observatory) 임무에 탑재된 SECCHI 기기 패키지의 종합 설계 명세서이다. SECCHI는 다섯 개의 망원경(EUVI, COR1, COR2, HI-1, HI-2)으로 구성되어 태양 표면(원반)에서 1 AU(지구) 너머까지 끊김 없는 시야로 코로나와 내부 헬리오스피어를 영상화한다. 두 STEREO 위성(Ahead, Behind)에서 동일한 기기를 동시에 운용함으로써, 인류 최초로 태양 활동 영역(특히 CME, Coronal Mass Ejection)을 두 시점에서 동시에 입체적(stereoscopic)으로 관측하고 그 3차원 구조와 운동학을 추적할 수 있게 되었다.

This paper is the comprehensive design specification of the SECCHI instrument suite onboard NASA's STEREO mission. SECCHI consists of five telescopes (EUVI, COR1, COR2, HI-1, HI-2) which together image the corona and inner heliosphere with a continuous, overlapping field of view that extends from the solar disk all the way past 1 AU (Earth). By flying identical packages on two spacecraft (STEREO-A "Ahead" and STEREO-B "Behind") drifting away from Earth in heliocentric orbits, SECCHI provided humanity's first stereoscopic observations of solar features — particularly Coronal Mass Ejections (CMEs) — enabling 3-D reconstruction of their structure and propagation from Sun to Earth.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

CME는 1971년 OSO-7에서 처음 관측된 이래 Skylab/ATM, Solwind, SMM/CP, SOHO/LASCO 등 다섯 임무에 의해 단일 시점(주로 지구 또는 SOHO L1)의 코로나그래프로만 관측되어 왔다. SOHO/LASCO는 1996년 이후 수만 건의 CME를 카탈로그화했지만, 한 시점의 투영 영상만으로는 진짜 3D 속도, 진행 방향, Earth-directed 여부를 모호하게 추정해야 했고("halo CME" 문제) 코로나그래프의 외부 시야(약 30 R☉)와 행성간 in-situ 측정 사이에는 거대한 관측 공백이 존재했다. STEREO/SECCHI는 이 공백 — 영상이 미치지 않던 30–215 R☉ 구간 — 을 헬리오스피어 영상기(HI)로 메우고, 두 위성의 입체 관측으로 투영 모호성을 풀고자 설계된 임무이다.

CMEs were first observed in 1971 with OSO-7 and have been monitored by coronagraphs on five subsequent missions (Skylab/ATM, Solwind, SMM/CP, SOHO/LASCO), but always from a single vantage point (typically Earth/L1). SOHO/LASCO catalogued tens of thousands of events after 1996, yet single-viewpoint projection left the true 3-D speed, propagation direction, and Earth-directedness ambiguous (the "halo CME" problem). Worse, an enormous observational gap existed between the outer coronagraph FOV (~30 R☉) and in-situ planetary measurements. STEREO/SECCHI was designed expressly to bridge that gap with the wide-field Heliospheric Imagers (HI) and to break projection ambiguity through dual-spacecraft stereoscopy.

### 타임라인 / Timeline

```
1939 ──── Lyot: invention of the coronagraph
1971 ──── OSO-7: first CME detected
1973 ──── Skylab/ATM: first imaging coronagraph in space
1979 ──── Solwind P78-1: white-light coronagraph
1980 ──── SMM/CP: improved coronagraph
1995 ──── SOHO/LASCO (C1/C2/C3): the gold-standard CME survey
2003 ──── SMEI on Coriolis: first all-sky heliospheric imager
2006 ──── STEREO-A & STEREO-B launch (Oct 25)
2008 ──── This paper: SECCHI in-flight design overview
2010+ ──── First true 3D CME reconstructions, J-maps, geoeffective tracking
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Thomson scattering / 톰슨 산란**: 자유 전자가 광구에서 온 광자를 산란시켜 K-corona를 만드는 메커니즘. 산란된 빛의 편광도(polarization)는 시선(line-of-sight) 따라 적분된 전자밀도와 산란 각도의 함수.
- **F-corona / Zodiacal light**: 행성간 먼지에 의한 산란광. HI의 가장 큰 배경 신호.
- **Lyot coronagraph 원리**: 외부 또는 내부 차폐(occulter) + Lyot stop으로 광구 회절광을 단계적으로 제거.
- **Polarized brightness (pB)**: 세 각도(-60°, 0°, +60°)의 선형 편광 영상으로부터 K-corona 신호를 F-corona로부터 분리하는 표준 기법. $pB = \sqrt{(I_0 - I_{60})^2 + (I_0 - I_{-60})^2}$ 형태.
- **EUV multilayer mirror**: Mo/Si 다층막에 의한 17.1, 19.5, 28.4, 30.4 nm 협대역 반사. 각 사분면(quadrant)이 다른 파장에 최적화.
- **Elongation angle**: 태양-위성-목표물 사이의 각도. HI 영상에서 가로축으로 사용.

- **Thomson scattering**: Free electrons scatter photospheric photons producing the K-corona; the scattered light's polarization is a line-of-sight integral of electron density weighted by scattering angle.
- **F-corona / zodiacal light**: Light scattered by interplanetary dust — the dominant background in HI fields.
- **Lyot coronagraph principle**: External or internal occulter + Lyot stop progressively eliminate diffracted photospheric light.
- **Polarized brightness (pB)**: Standard technique using three polarization angles (-60°, 0°, +60°) to separate K-corona from F-corona.
- **EUV multilayer mirror**: Mo/Si multilayer thin films providing narrow-band reflection at 17.1, 19.5, 28.4, 30.4 nm; each EUVI quadrant tuned to a different line.
- **Elongation angle**: Sun-spacecraft-target angle; the natural abscissa for HI imagery.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **EUVI** | Extreme UltraViolet Imager. 17.1/19.5/28.4/30.4 nm 4채널 normal-incidence 망원경, FOV ±1.7 R☉. Chromosphere & low corona를 4 emission line으로 영상화. / 4-channel normal-incidence EUV telescope imaging the chromosphere & low corona out to 1.7 R☉. |
| **COR1** | Inner Coronagraph. 내부 차폐(internal occulter) refractive Lyot 코로나그래프, 1.4–4 R☉, 656 nm Hα 협대역. / First space-borne internally occulted refractive Lyot coronagraph; 1.4–4 R☉, 656 nm. |
| **COR2** | Outer Coronagraph. 외부 차폐(external occulter) Lyot 코로나그래프, 2.5–15 R☉, 650–750 nm. / Externally occulted Lyot coronagraph; 2.5–15 R☉, 650–750 nm. |
| **HI-1, HI-2** | Heliospheric Imagers. 광시야 무차폐(shutterless) 카메라. HI-1: 4–24°, HI-2: 19–89° 신연각(elongation), 합쳐서 15–318 R☉ 시야. / Wide-angle shutterless cameras covering 15–318 R☉ between them, viewing the Sun-Earth line. |
| **SCIP** | Sun-Centered Instrument Package. EUVI+COR1+COR2+GT를 담는 광학 벤치. / Optical bench housing EUVI, COR1, COR2 and the Guide Telescope. |
| **GT (Guide Telescope)** | 4-photodiode limb sensor. EUVI의 fine pointing 오류 신호 제공 (±7″, factor-3 attenuation @ 10 Hz). / Limb sensor providing fine pointing error to the EUVI active secondary mirror. |
| **SEB** | SECCHI Electronics Box. RAD750 PowerPC, MIL-STD-1553, SpaceWire, 압축·스케줄링 담당. / Payload controller (RAD750) running flight SW, image compression, scheduling. |
| **pB (polarized brightness)** | 세 편광 각도 영상으로부터 K-corona 신호를 추출. COR1/COR2의 표준 산출물. / Standard data product extracting K-corona from polarization triplets. |
| **ICER / Rice / H-Compress** | 비행 SW가 사용하는 세 압축 알고리즘. ICER는 가변 wavelet 손실 압축 (최대 200×). / Three onboard compression algorithms; ICER offers up to 200× lossy wavelet compression. |
| **Beacon mode** | 504 bps의 저해상도 실시간 우주기상 채널. / 504 bps low-resolution real-time space weather channel. |
| **Stereoscopic baseline** | 두 STEREO 위성이 매년 ~22°씩 멀어지면서 변하는 시차 기저선. / The continually growing parallax baseline (~22°/yr) between A and B. |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Gibson 1973 K-corona model used to estimate COR1 SNR**:
$$
\log_{10}(pB) = -2.65682 - 3.55169\,(R/R_\odot) + 0.459870\,(R/R_\odot)^2
$$
1.4–4 R☉에서 K-corona 편광 휘도를 평균 단면 함수로 근사. SNR 시뮬레이션의 입력 모델. / Empirical functional form valid 1.4–4 R☉ for the polarized brightness, used as an input to COR1 SNR estimates.

**(2) Polarized brightness from three angles** (used by COR1 and COR2):
$$
pB \;=\; \tfrac{2}{3}\sqrt{(I_{0}-I_{60})^{2}+(I_{0}-I_{-60})^{2}+(I_{60}-I_{-60})^{2}}
$$
세 polarization 영상의 차이로부터 K-corona의 편광 신호를 추출. 비편광 F-corona는 상쇄. / Extracts the K-corona polarized signal; the unpolarized F-corona contribution cancels.

**(3) Effective area / count rate prediction (EUVI)**:
$$
N(\lambda) = \int A_{\rm eff}(\lambda)\, F_\odot(\lambda, T)\, d\lambda
$$
$A_{\rm eff}$는 거울×필터×CCD QE의 곱; $F_\odot$은 CHIANTI로 계산된 differential emission measure 분포에 따른 광자속. / Pixel count rate prediction combines effective area with CHIANTI-derived plasma emissivities.

**(4) Fresnel-Kirchhoff diffraction** (HI forward-baffle design rationale):
$$
B/B_\odot \approx \frac{1}{4\pi^2 \alpha^2}
$$
지평선 아래 각거리 $\alpha$에서의 회절 잔광 근사. 5단(knife-edge cascade) 차폐 시스템이 이 함수의 연쇄적 감쇠로 $10^{-13}$ B☉ 수준 거부 달성. / Approximate diffraction profile below a knife edge; the 5-vane cascade composes this rejection multiple times to reach ~10⁻¹³ B☉.

**(5) Onboard SNR enhancement by N-image summation in HI**:
$$
{\rm SNR}_{\rm sum} = {\rm SNR}_{\rm single} \times \sqrt{N}\, \times\, ({\rm 2\!\times\!2\ binning\ factor}\!=\!2)
$$
HI는 무셔터 모드로 50개 영상 합산 + 2×2 비닝 → 약 14× SNR 증가. / HI uses shutterless 50-image summation + 2×2 binning, ~14× SNR boost.

---

## 6. 읽기 가이드 / Reading Guide

이 논문은 50쪽 분량의 hardware overview 논문이다. 다음 순서로 읽기를 권장한다:

This is a 50-page hardware overview paper. Recommended reading flow:

1. **§1 (Introduction)**: SECCHI의 다섯 망원경이 어떤 거리 범위를 어떻게 분담하는지 — Fig.1을 꼭 본다. / Read Fig. 1 carefully — it shows how the five telescopes partition the radial distance from Sun to Earth.
2. **§2 (EUVI)**: Ritchey-Chrétien 광학, 4-quadrant Mo/Si multilayer, GT-driven fine pointing. Table 1과 Fig. 6 (effective area), Fig. 7 (response vs T). / RC optics, 4-quadrant multilayers, GT-driven pointing; focus on Tables 1 & 3.
3. **§3 (COR1)**: 내부 차폐 refractive Lyot — LASCO와 다른 새 설계. Table 4 (perf), Eqn for Gibson model. / Novel internally-occulted refractive Lyot design distinct from LASCO/C1.
4. **§4 (COR2)**: 외부 차폐 Lyot, LASCO/C2-C3 후예지만 더 빠른 광학. Table 5. / External-occulter design, faster optics than LASCO.
5. **§5 (HI)**: 새로운 개념 — Fig. 16, 17, 18을 본다. 5-vane forward baffle + perimeter + interior baffle. SMEI heritage. / New concept; pay attention to baffle cascade & Fresnel diffraction.
6. **§6–§9 (GT, SCIP, mechanisms, electronics)**: 빠르게 훑는다 — 핵심은 RAD750 + SpaceWire + 1553. / Skim engineering details.
7. **§10–§11 (CCDs & flight SW)**: 압축 (ICER, Rice, H-Compress) 부분 자세히. / Compression schemes are scientifically relevant.
8. **§12 (CONOPS)**: synoptic vs campaign 프로그램, beacon mode, 데이터 정책 (open access). / Operations philosophy.

읽으면서 Q: "왜 다섯 개여야 하는가?"를 계속 떠올린다 — 한 망원경의 동적 영역, stray light suppression, signal level은 모두 거리에 따라 극단적으로 변한다.

While reading, hold the question: *why five telescopes?* — dynamic range, stray-light suppression, and signal levels all change by orders of magnitude over the imaged radius.

---

## 7. 현대적 의의 / Modern Significance

SECCHI는 2006년 발사 이후 현대 우주기상학(space weather)의 토대가 되었다. 가장 중요한 유산:

- **3D CME reconstruction**: GCS (Graduated Cylindrical Shell) 모델이 두 위성의 동시 영상에 fit되어 진짜 속도/방향 산출.
- **J-map (time–elongation map)**: HI의 한 행을 시간축으로 쌓아 1 AU까지 CME 추적. 지구 도착 시간 예보의 표준 도구.
- **Stereoscopic active region modeling**: EUVI 듀얼 영상으로 코로나 루프의 3D 재구성 (Aschwanden 등의 연구).
- **DKIST/PUNCH 시대의 기준**: NASA PUNCH (2025–) 임무는 SECCHI의 헬리오스피어 영상 컨셉을 직접 계승.
- **머신러닝 데이터셋**: 16년치 SECCHI/LASCO 데이터는 자동 CME 검출/예보 ML 모델의 학습 데이터로 활용.

SECCHI has been the foundation of modern space-weather forecasting since 2006. Its key legacies:

- **3-D CME reconstruction** via the GCS (Graduated Cylindrical Shell) model fit to simultaneous A/B images, yielding true speeds and directions.
- **J-maps** (time–elongation diagrams) constructed from HI image strips have become the standard tool for tracking CMEs to 1 AU and predicting Earth arrival times.
- **Stereoscopic active-region modeling**: EUVI image pairs enable 3-D loop reconstructions (Aschwanden et al.).
- **DKIST/PUNCH era**: the 2025 NASA PUNCH mission directly inherits the heliospheric imaging concept pioneered here.
- **ML datasets**: 16+ years of SECCHI+LASCO archives now train automated CME detection / arrival-time forecasting models.

---

## Q&A

(읽기 세션 중 추가됨 / Populated during reading session)
