---
title: "Pre-Reading Briefing: Solar Science with the Atacama Large Millimeter/Submillimeter Array"
paper_id: "27"
topic: Solar_Observation
date: 2026-04-23
type: briefing
---

# Solar Science with ALMA — A New View of Our Sun: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Wedemeyer, S., Bastian, T., Brajša, R., et al., "Solar Science with the Atacama Large Millimeter/Submillimeter Array — A New View of Our Sun", *Space Science Reviews*, 2016. DOI: 10.1007/s11214-015-0229-9
**Author(s)**: S. Wedemeyer, T. Bastian, R. Brajša, H. Hudson, G. Fleishman, M. Loukitcheva, et al. (SSALMON collaboration)
**Year**: 2016

---

## 1. 핵심 기여 / Core Contribution

**한국어**: 이 논문은 Atacama Large Millimeter/submillimeter Array (ALMA)를 활용한 태양 관측의 종합 리뷰 논문이다. ALMA는 칠레 Chajnantor 고원(해발 5000 m)에 설치된 66개의 안테나로 구성된 간섭계로, 밀리미터 및 서브밀리미터 파장 영역(35–950 GHz, 파장 8.6 mm–0.3 mm)에서 태양 채층(chromosphere)을 전례 없는 공간·시간·분광 분해능으로 관측할 수 있다. 저자들은 mm/sub-mm 파장 복사의 형성 메커니즘(주로 free-free 흡수), ALMA의 기술 사양(주빔 크기, 기저선, 시야), 그리고 채층 가열, 홍염(prominence), 흑점(sunspot), 태양 플레어 등 광범위한 과학 응용 주제를 포괄적으로 정리한다.

**English**: This paper is a comprehensive review of solar observations with the Atacama Large Millimeter/submillimeter Array (ALMA). ALMA is a 66-antenna interferometer located on the Chajnantor plateau (5000 m altitude) in Chile, capable of observing the solar chromosphere at millimeter and submillimeter wavelengths (35–950 GHz, 8.6 mm–0.3 mm) with unprecedented spatial, temporal, and spectral resolution. The authors systematically review the formation mechanisms of mm/sub-mm radiation (primarily free-free absorption), ALMA's technical specifications (primary beam, baselines, field of view), and the broad range of solar science cases, including chromospheric heating, prominences, sunspots, and solar flares.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**: 태양 채층은 광구(photosphere)와 코로나(corona) 사이의 복잡하고 역동적인 경계층으로, 수십 년간 집중적으로 연구되었지만 여전히 가장 이해가 부족한 영역이다. 기존의 채층 관측은 Ca II, Hα 같은 광학/자외선 분광선에 의존했는데, 이들은 non-LTE 복사전달이 필요해 해석이 매우 어렵다. Mm 파장 영역은 LTE 가정이 유효하고 Rayleigh-Jeans 근사에 의해 휘도 온도(brightness temperature)가 기체 온도와 거의 선형적으로 비례하기 때문에 이상적인 진단 도구로 알려져 있었다. 그러나 BIMA, OVSA, Nobeyama 등 기존 mm/cm 망원경은 분해능이 10″ 수준으로 너무 낮아 잠재력이 제대로 발휘되지 못했다. ALMA의 등장은 mm 파장 태양 관측을 패러다임 수준으로 바꾸는 사건이었다.

**English**: The solar chromosphere — a complex, dynamic interface between the photosphere and corona — has been intensively studied for decades yet remains elusive. Classical chromospheric diagnostics (Ca II, Hα) involve non-LTE radiative transfer and are notoriously hard to interpret. The mm-wavelength regime was long recognized as an ideal diagnostic because LTE applies, and in the Rayleigh-Jeans limit the brightness temperature is nearly linearly proportional to the gas temperature. However, prior mm/cm instruments (BIMA, OVSA, Nobeyama) had resolutions of ~10″, far too coarse to resolve the fine structure of the chromosphere. ALMA's advent represents a paradigm shift in solar mm/sub-mm observing.

### 타임라인 / Timeline

```
1959 ── Kundu's first interferometric solar radio obs.
1975 ── WSRT high-resolution radio imaging (6, 20 cm)
1980s ─ VLA solar observations begin
1990s ─ BIMA, Nobeyama, OVSA, SSRT — 10″ resolution mm/cm
2007 ── Wedemeyer-Böhm 3D radiative transfer predictions
2011 ── Shibasaki review of pre-ALMA solar radio
2013 ── ALMA operation commences (no solar mode yet)
2014 ── First ALMA solar commissioning observations
2015 ── Solar observing modes development; SSALMON formed
2016 ── THIS PAPER — comprehensive review
2016+ ── Cycle 4: regular solar observing expected to begin
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**:
- **전파 천문학 기초**: 전파 간섭계(interferometry), u-v plane, aperture synthesis, primary/synthesized beam 개념
- **복사전달 (Radiative Transfer)**: 광학 깊이 τ, 원천 함수 S_ν, 방사 전달 방정식
- **Planck 함수와 Rayleigh-Jeans 극한**: 저에너지 한계에서의 선형화
- **플라즈마 물리**: free-free 흡수(자유-자유 방출), 제동복사(bremsstrahlung), Gaunt factor
- **태양 대기 구조**: 광구-채층-전이영역-코로나, 온도 최소층, VAL/FAL 모델
- **자기유체역학(MHD) 기초**: Alfvén 파, 자기 재연결, 흑점·홍염 구조
- **이전 논문**: Paper #26 (Bastian 1998, 전파 방출 메커니즘 리뷰) 이해

**English**:
- **Radio astronomy basics**: Interferometry, u-v plane, aperture synthesis, primary/synthesized beam
- **Radiative transfer**: Optical depth τ, source function S_ν, RT equation
- **Planck function & Rayleigh-Jeans limit**: Linearization in the low-energy regime
- **Plasma physics**: Free-free absorption, bremsstrahlung, Gaunt factor
- **Solar atmosphere**: Photosphere-chromosphere-transition-corona, temperature minimum, VAL/FAL models
- **MHD basics**: Alfvén waves, magnetic reconnection, sunspot/prominence structure
- **Prior paper**: Paper #26 (Bastian et al. 1998, solar radio emission review)

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| ALMA | Atacama Large Millimeter/submillimeter Array — 66개 안테나 간섭계 (12-m × 50, 7-m × 12, 12-m TP × 4), 칠레 Chajnantor 5000 m / 66-antenna interferometer in Chile |
| Brightness temperature T_b | 같은 복사 강도를 내는 흑체의 온도. RJ 한계에서 T_b = c²I_ν/(2ν²k) / Temperature of blackbody with same intensity |
| Free-free absorption | 하전 입자와 이온의 쿨롱 산란 시 광자 흡수. χ_ν ∝ n_e²/(T^(3/2) ν²) / Absorption during Coulomb scattering of free charges |
| Rayleigh-Jeans limit | hν ≪ k_B T 극한에서 Planck 함수의 선형 근사 / Low-energy linear limit of Planck function |
| Primary beam | 단일 안테나의 시야 (FWHM ≈ 1.13 λ/D) / Field of view of a single antenna |
| ACA / TP Array | Atacama Compact Array / Total Power — 짧은 기저선과 전체 디스크 측정 담당 / Handles short baselines and absolute flux |
| Gyroresonance emission | 자기장 내 열 전자의 조화 방출; mm 파장에서는 기여 미미 (ALMA 대역 밖) / Thermal electron harmonic emission — negligible at mm |
| Gyrosynchrotron | 상대론적 전자의 비열 방출; 플레어에서 중요 / Non-thermal emission from relativistic electrons; important in flares |
| Contribution function | 특정 높이에서 전체 방출에 대한 기여 함수 / Function describing height contribution to emergent radiation |
| Non-equilibrium H ionization | 시간 의존 수소 이온화 (재결합 시간 효과) / Time-dependent hydrogen ionization |
| SSALMON | Solar Simulations for the ALMA Observatory Network — ALMA 태양 모사 네트워크 / International network coordinating ALMA solar simulations |
| Stokes V / Circular polarisation | 원편광; 자기장 측정의 핵심 / Circular polarization, key to magnetic field measurement |

---

## 5. 수식 미리보기 / Equations Preview

**한국어**:

### (1) Rayleigh-Jeans 한계에서 휘도 온도 / Brightness temperature in RJ limit

$$T_b = \frac{c^2}{2 k_B \nu^2} I_\nu = \frac{\lambda^2}{2 c k_B} I_\lambda$$

mm 파장에서 hν ≪ k_B T이므로 휘도 온도가 강도와 직접 비례. ALMA는 사실상 '선형 온도계' 역할.

### (2) Electron-ion free-free 흡수 계수 / Free-free absorption coefficient

$$\chi_{\text{ions,ff}} \approx 9.78 \times 10^{-3} \frac{n_e}{\nu^2 T^{3/2}} \sum_i Z_i^2 n_i (17.9 + \ln T^{3/2} - \ln \nu) \quad [\text{cm}^{-1}]$$

간단히 χ ∝ n_e²/(T^(3/2) ν²). 낮은 주파수, 저온 고밀도에서 더 큰 불투명도.

### (3) ALMA primary beam / 주빔

$$\theta_{PB} \approx 1.13 \frac{\lambda}{D} \approx 19'' \times \frac{\lambda}{1\,\text{mm}} \quad (D = 12\,\text{m})$$

1 mm에서 약 19″, 3 mm (100 GHz)에서 약 57″.

### (4) 최대 회수 가능 크기 / Maximum recoverable size

$$\vartheta_{\max} = \frac{0.6 \lambda}{L_{\min}}\,\text{rad} = \frac{37100}{L_{\min} \nu}\,\text{arcsec}$$

L_min은 최단 기저선(m), ν는 GHz. 간섭계가 '큰' 구조를 잃는 한계.

### (5) 원편광도와 자기장 / Circular polarization and B

$$\mathcal{P} = \zeta \frac{\nu_B}{\nu} \cos\theta, \qquad \nu_B = 2.8 \times 10^6 \text{Hz} \times B[\text{G}]$$

이로부터 종방향 자기장 B_l 도출. 이는 ALMA의 채층 자기장 측정 기법.

**English**: Same equations — ALMA exploits the Rayleigh-Jeans regime to act as a nearly linear thermometer of the chromosphere; the free-free opacity grows with ν^(-2) and n_e²/T^(3/2); the primary beam scales with wavelength (≈ 19″ at 1 mm); the maximum recoverable angular size depends on the shortest baseline; circular polarization gives the line-of-sight magnetic field.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**:
1. **Sect. 1 (Introduction)**: ALMA의 전반적 능력과 기존 BIMA 영상과의 비교 (Fig. 2 vs. Fig. 1)에 주목.
2. **Sect. 2 (Formation of mm radiation)**: 2.1의 free-free 식 (Eq. 1–9)을 꼼꼼히 읽고 Fig. 4 (1 mm 파장에서 불투명도 vs. 온도)와 Fig. 5 (기여함수 높이)를 이해.
3. **Sect. 2.5 (Polarisation)**: Eq. 12–17 — 원편광도에서 자기장을 추출하는 기법.
4. **Sect. 3 (ALMA 사양)**: 3.2의 주빔, 기저선, 수신기 10개 대역(35–950 GHz) 사양을 표로 정리.
5. **Sect. 4 (Science cases)**: 채층 가열 (4.2), 흑점 (4.3), 플레어 (4.4), 홍염 (4.8), 코로나 rain (4.7.2) 중 관심 주제 선택 읽기.
6. **Figures**: Fig. 5 (contribution functions), Fig. 8 (synthetic ALMA maps), Fig. 10 (brightness temperature histograms), Fig. 13 (AR polarization)은 반드시 이해할 것.

**English**:
1. **Sect. 1**: Note the dramatic improvement over BIMA (Fig. 2 vs. Fig. 1).
2. **Sect. 2**: Work through Eqs. 1–9 for free-free absorption; study Figs. 4–5 for opacity and contribution functions.
3. **Sect. 2.5**: Eqs. 12–17 on polarization-to-B conversion.
4. **Sect. 3**: Tabulate ALMA's 10 receiver bands (35–950 GHz), primary beam, baselines.
5. **Sect. 4**: Pick science cases of interest — chromospheric heating (4.2), sunspots (4.3), flares (4.4), prominences (4.8), coronal rain (4.7.2).
6. **Must-understand figures**: Fig. 5 (contribution functions), Fig. 8 (synthetic maps), Fig. 10 (histograms), Fig. 13 (AR polarization).

---

## 7. 현대적 의의 / Modern Significance

**한국어**: 이 논문 이후 ALMA 태양 관측은 2016년 말 Cycle 4에서 정식으로 시작되었고, 2017년 이후 정기적 태양 campaigns이 진행되어 왔다. 주요 성과는 (1) 채층 온도의 최초 직접 측정 (Shimojo et al. 2017), (2) '채층 모자이크' 영상화 (White et al. 2017), (3) 흑점 위 채층 구조의 mm 파장 매핑, (4) IRIS/SDO와의 동시 관측을 통한 채층 가열 진단, (5) SSALMON을 통한 방법론 표준화이다. 또한 이 논문은 DKIST (2020년대 운영), EST (유럽), Solar-C 등 차세대 태양 관측소와 ALMA를 결합한 다파장 태양물리학의 청사진을 제시한다. 채층 자기장 측정, 플레어의 sub-THz 방출, 코로나 rain과 홍염 열구조 등은 현재도 핵심 연구 주제로 남아 있다.

**English**: Since this paper, ALMA solar observations began officially in Cycle 4 (late 2016), with regular campaigns from 2017. Major results include (1) first direct chromospheric temperature measurements (Shimojo et al. 2017), (2) chromospheric mosaics (White et al. 2017), (3) mm-wave mapping of sunspot chromospheres, (4) coordinated IRIS/SDO+ALMA heating diagnostics, and (5) methodological standardization through SSALMON. The paper also provides a blueprint for multi-wavelength solar physics combining ALMA with DKIST (operational in the 2020s), EST, and Solar-C. Chromospheric magnetometry, flare sub-THz emission, and the thermal structure of coronal rain and prominences remain active frontiers.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
