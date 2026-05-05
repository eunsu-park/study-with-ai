---
title: "Pre-Reading Briefing: The FIELDS Instrument Suite for Solar Probe Plus"
paper_id: "52_bale_2016"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# The FIELDS Instrument Suite for Solar Probe Plus: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: S.D. Bale et al., "The FIELDS Instrument Suite for Solar Probe Plus: Measuring the Coronal Plasma and Magnetic Field, Plasma Waves and Turbulence, and Radio Signatures of Solar Transients", Space Science Reviews, **204**, 49-82 (2016). DOI: 10.1007/s11214-016-0244-5
**Author(s)**: S.D. Bale, K. Goetz, P.R. Harvey, P. Turin, J.W. Bonnell, T. Dudok de Wit, R.E. Ergun, R.J. MacDowall, M. Pulupa, M. Andre, et al. (Berkeley/UMN/LASP/GSFC/IRF/LESIA/QMUL led collaboration)
**Year**: 2016

---

## 1. 핵심 기여 / Core Contribution

**한국어**:
이 논문은 NASA Solar Probe Plus(SPP, 추후 Parker Solar Probe로 명명) 우주선에 탑재될 FIELDS 종합 관측 장비의 과학 목표, 측정 요구사항, 설계, 그리고 운영 계획을 종합 기술한 미션 기기 논문(mission instrument paper)이다. FIELDS는 5개의 전기장 안테나(V1-V5), 2개의 fluxgate magnetometer(MAGi, MAGo), 1개의 search-coil magnetometer(SCM)와, Antenna Electronics Board(AEB), Digital Fields Board(DFB), Time Domain Sampler(TDS), Radio Frequency Spectrometer(RFS), Data Control Board(DCB), Low Noise Power Supply(LNPS) 등의 디지털 처리 보드 스택으로 구성된다. DC부터 20 MHz까지 자기장과 전기장을, 140 dB의 dynamic range로 측정하여, 코로나 가열 문제, 알펜파, MHD 난류, 자기재결합, 충격파, 그리고 type III 전파 폭발 같은 태양풍 가속의 핵심 물리를 인-시추(in situ)로 처음 측정하는 것을 목표로 한다.

**English**:
This paper is the FIELDS instrument-suite reference paper for NASA's Solar Probe Plus (SPP, later renamed Parker Solar Probe), describing the scientific objectives, measurement requirements, hardware design, and mission concept of operations. FIELDS comprises five voltage probes (V1-V5), two fluxgate magnetometers (MAGi, MAGo), one search-coil magnetometer (SCM), and a stack of digital processing electronics: the Antenna Electronics Board (AEB), Digital Fields Board (DFB), Time Domain Sampler (TDS), Radio Frequency Spectrometer (RFS), Data Control Board (DCB), and Low Noise Power Supplies (LNPS). It will measure 3-component magnetic field and electric field from DC to ~20 MHz with 140 dB dynamic range to make the first in situ measurements of coronal heating, Alfven waves, MHD turbulence, magnetic reconnection, shocks, and type III radio bursts in the inner heliosphere down to 9.86 solar radii.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**:
1958년 Eugene Parker가 태양풍 이론을 제시한 이래, 코로나가 어떻게 100만 K 이상으로 가열되며 어떻게 초음속 흐름으로 가속되는지는 태양물리학 최대 미해결 문제로 남아 있었다. Helios 1, 2(1974-1986)는 0.3 AU까지 접근하여 자기장과 플라즈마의 반경 진화를 보여주었으나, 코로나 자체는 한 번도 직접 측정된 적이 없었다. Wind, Polar, STEREO, THEMIS, Van Allen Probes(RBSP) 등에서 다양한 전기장/자기장 측정 기술과 디지털 처리 기법(THEMIS Digital Fields Board, RBSP/EFW Axial Double Probe)이 누적되어 왔다. SPP/FIELDS는 이러한 헤리티지 위에서 코로나 9.86 R_s까지 진입하는 최초의 인간 제작 기기로, 미답 영역의 자기장·전기장·전파를 모두 측정하도록 설계되었다.

**English**:
Since Parker (1958) proposed solar wind theory, how the corona is heated to >1 MK and accelerated to supersonic flow has remained the central unsolved problem of solar physics. Helios 1, 2 (1974-1986) reached 0.3 AU and characterized the radial evolution of B and plasma parameters, but the corona itself was never directly sampled. Heritage instruments on Wind, Polar, STEREO, THEMIS, and Van Allen Probes (RBSP) accumulated the requisite electric/magnetic measurement techniques and digital processing know-how (THEMIS Digital Fields Board, RBSP/EFW Axial Double Probe). SPP/FIELDS, built on this heritage, is the first human instrument to enter the corona to 9.86 R_s, designed to measure DC-to-radio fields in unexplored regions.

### 타임라인 / Timeline

```
1958 ─ Parker: solar wind theory predicted / 태양풍 이론
1974 ─ Helios 1 launch (0.29 AU perihelion) / Helios 1 발사
1976 ─ Voyager radio astronomy (PWS heritage) / Voyager PWS 헤리티지
1995 ─ Wind/WAVES (radio + plasma waves) / Wind WAVES
2006 ─ STEREO/WAVES (Bougeret et al. 2008) / STEREO WAVES
2007 ─ THEMIS Electric Field Instrument (Bonnell+ 2009)
2012 ─ Van Allen Probes EFW (Wygant+ 2013) / RBSP EFW
2016 ─ Bale+ FIELDS instrument paper / 본 논문
2018 ─ Parker Solar Probe launch (Aug 12, 2018, post-paper)
2019 ─ First perihelion ~36 R_s / 첫 근일점
2024 ─ Final perihelion ~9.86 R_s / 최종 근일점
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**:
- **플라즈마 물리 기초**: Debye length λ_D, plasma frequency ω_pe, gyrofrequency Ω_c, inertial length c/ω_p, Larmor radius ρ
- **태양풍 모델**: Parker spiral 자기장, 등방 모형, Sittler-Guhathakurta 코로나 밀도 모형, Sheeley 솔라윈드 속도 모형
- **MHD 난류**: Kolmogorov k^(-5/3) 관성영역 스펙트럼, 소산영역 ~k^(-3), Alfven 파 δE/δB ~ v_A
- **전자기 측정 기법**: double-probe E-field 원리, fluxgate vs search-coil 자력계, 전류 바이어스, Langmuir probe 곡선
- **우주선 공학**: spacecraft charging (Whipple 1981), 광전자 흐름, 플라즈마 wake, 정전기 청결성(EMC), 우주선 부유 전위
- **신호처리**: FFT, polyphase filter bank (PFB, Vaidyanathan 1990), digital filter bank (Cully+ 2008), Bessel low-pass anti-alias filter

**English**:
- **Plasma physics fundamentals**: Debye length λ_D, plasma frequency ω_pe, gyrofrequency Ω_c, inertial lengths c/ω_p, Larmor radius ρ
- **Solar wind models**: Parker spiral magnetic field, density power-laws, Sittler-Guhathakurta coronal density model, Sheeley solar-wind speed model
- **MHD turbulence**: Kolmogorov k^(-5/3) inertial spectrum, dissipation range ~k^(-3), Alfvenic δE/δB ~ v_A
- **EM measurement techniques**: double-probe E-field principle, fluxgate vs search-coil magnetometer, current biasing, Langmuir probe characteristic
- **Spacecraft engineering**: spacecraft charging (Whipple 1981), photoelectron currents, plasma wake, electrostatic cleanliness (EMC), floating potential
- **Signal processing**: FFT, polyphase filter bank (PFB, Vaidyanathan 1990), digital filter banks (Cully+ 2008), Bessel anti-alias filtering

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Double-probe E-field measurement** | 한 쌍의 분리된 sensor 사이의 부유전위 차이를 측정하여 E를 추정하는 기법. E_12 = (V_1 − V_2)/L_12. 우주선 charging 효과를 상쇄. / Differential floating-potential between two separated sensors gives E ≈ ΔV / baseline; cancels common-mode S/C charging |
| **Fluxgate magnetometer (MAG)** | 강자성 코어를 포화시키는 drive coil과 sense coil 쌍으로 DC ~ ~140 Hz의 B를 ±65,536 nT, 16-bit 정밀도로 측정. / Saturable-core sensor for DC-to-Hz magnetic field; 16-bit ranging up to ±65,536 nT |
| **Search-coil magnetometer (SCM)** | 패러데이의 법칙으로 dB/dt를 측정하는 inductive magnetometer. 10 Hz~50 kHz(ELF/VLF)와 1 kHz~1 MHz(LF/MF) 대역. / Inductive sensor: V_out ∝ -dB/dt; covers 10 Hz–50 kHz and 1 kHz–1 MHz |
| **Quasi-thermal noise (QTN)** | 안테나 부근 thermal 전자가 발생시키는 plasma frequency 부근 노이즈 스펙트럼. 정확한 n_e, T_e 추정 가능. / Thermal-electron-induced antenna voltage noise near f_pe; provides accurate n_e, T_e |
| **Type III radio burst** | 비열적 전자빔이 코로나/행성간 매질을 통과하며 plasma frequency에서 방출하는 전파 폭발. 빔의 궤적을 추적. / Non-thermal electron beam radio emission at f_pe; traces beam path |
| **Picket-fence EMC** | 모든 DC-DC 변환기를 150 kHz 정수배의 결정 주파수에서 동기화하여 노이즈를 좁은 주파수 spike로 가두는 청결성 정책. / EMC scheme: all DC-DC converters chop at multiples of 150 kHz, isolating noise into narrow lines |
| **Polyphase filter bank (PFB)** | FFT 전 windowing+다채널 분기로 spectral leakage를 줄이는 고정밀 디지털 스펙트럼 분석기. / Pre-FFT windowing/decomposition that suppresses spectral leakage |
| **NY second** | FIELDS 마스터 클럭 150 kHz의 2^17/150,000 ≈ 0.873813 s 단위. 모든 cadence는 NY second의 2의 거듭제곱 분수. / FIELDS time unit, equal to 2^17/150000 s; all cadences are power-of-two of 1 NYsec |
| **Burst memory (DBM)** | DFB 내 ~3.5초 짜리 6채널 고속 파형을 경쟁적으로 평가·저장하는 quality-ranked buffer 체계. / Competitive ranked storage of 3.5 s, 6-channel high-rate snapshots |
| **Coordinated Burst Signal (CBS)** | DFB·TDS·RFS·SWEAP 입력의 가중합 4 NYsec 카덴스로 흥미 이벤트 동기 트리거. / Weighted-sum 4×/NYsec quality metric triggering coordinated burst captures |
| **Electrostatic barrier** | 광전자 Debye length가 sensor 거리보다 작아 형성되는 비단조 전위 장벽; -25 V로 광전자 탈출을 차단. / Non-monotonic potential barrier near heat shield; ~−25 V; blocks 90% of photoelectron escape |
| **TPS (Thermal Protection System)** | SPP의 탄소 복합재 heat shield. V1-V4 안테나는 그 뒤에 거의 동일선상으로 배치, 끝부분(2 m whip)이 햇빛에 노출. / SPP carbon-composite heat shield; V1-V4 antennas mounted at its base with whips emerging into full sunlight |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Double-probe E-field / 이중 탐침 전기장**:
$$
E_{12} = \frac{V_1 - V_2}{L_{12}}, \quad E_{34} = \frac{V_3 - V_4}{L_{34}}, \quad E_z = V_5 - \frac{V_1+V_2+V_3+V_4}{4}
$$
한국어: V1-V2와 V3-V4는 heat shield 면 안의 두 직교 전기장 성분, V5와 평균값의 차이는 sunward(z-방향) 성분.
English: V1-V2 and V3-V4 give two orthogonal in-shield-plane components; V5 minus mean of V1-V4 yields sunward (z) component.

**(2) Search-coil sensitivity / 서치코일 감도** (Faraday's law):
$$
V_{out}(\omega) = -N A \mu_{eff}\,\frac{dB}{dt} = -j\omega N A \mu_{eff}\,B
$$
한국어: 권선수 N, 단면적 A, 유효 투자율 μ_eff. 응답은 ω에 비례 → 저주파에서 노이즈 한계, 고주파에서 공진(resonance) 후 떨어짐.
English: Output is proportional to ω·B; rises until LC resonance then rolls off (see Fig. 12 ELF/VLF and LF/MF curves).

**(3) Fluxgate range/resolution / 자속계 분해능**:
$$
\Delta B_{LSB} = \frac{B_{range}}{2^{16}} = \frac{2 \times 65{,}536\ {\rm nT}}{65{,}536} = 2\ {\rm nT/LSB}
$$
한국어: 16-bit ADC, ±65,536 nT 범위 → 최대 2 nT/LSB; 자동 ranging으로 다른 범위(±1024, ±4096, ±16,384) 사용 시 분해능이 0.03 nT까지 향상.
English: 16-bit on ±65,536 nT yields 2 nT/LSB; auto-ranging into ±1024 nT improves resolution to ~0.03 nT.

**(4) Turbulence breakpoint scaling / 난류 절단 주파수의 반경 스케일링**:
$$
f_i\,({\rm Hz}) \approx 4.9\,r^{-1.66}, \qquad \delta B^2\,({\rm nT^2/Hz}) \approx 10^{8.1}\,r^{-2}
$$
한국어: r은 R_s 단위. r=10 R_s에서 inertial-range 상한이 ~100 Hz로 이동, 진폭은 ~10^6 nT^2/Hz.
English: r in solar radii; at r=10 R_s the injection-inertial breakpoint shifts to ~100 Hz with amplitude ~10^6 nT²/Hz.

**(5) NY second / NY 초**:
$$
1\,{\rm NYsec} = \frac{2^{17}}{150{,}000\ {\rm Hz}} \approx 0.873813\ {\rm s}, \quad 150{,}000/2^9 = 292.969\ {\rm Sa/NYsec}
$$
한국어: 모든 FIELDS 데이터의 디지털 시간 단위. fluxgate는 256 Sa/NYsec, DFB burst는 150,000 Sa/s.
English: master cadence unit; MAG samples at 256 Sa/NYsec, DFB burst at 150,000 Sa/s.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**:
1. **Sect. 1 Introduction**: FIELDS가 SPP의 4개 기기 중 하나임을 확인하고, 자매 기기(SWEAP, IS⊙IS, WISPR)의 역할을 빠르게 파악.
2. **Sect. 1.1-1.2 Plasma Environment**: Helios 데이터 재분석으로 9.86 R_s에서 |B|=2000 nT, n_e=7000 cm^-3, T_e=85 eV 같은 핵심 플라즈마 매개변수(Table 2)를 머릿속에 새길 것.
3. **Sect. 1.3 Spacecraft Charging**: photoelectron current, electrostatic barrier(Fig. 3), wake formation의 측정 영향을 확실히 이해해야 V1-V4의 forward 위치 선정 이유가 와 닿음.
4. **Sect. 2.1.1-2.1.5 Sensors**: 각 sensor(V1-V4, V5, MAGi/o, SCM)별 물리·기계 설계와 헤리티지를 정리. SCM 응답 곡선(Fig. 12)은 transfer function 구현에 필수.
5. **Sect. 2.2 Electronics**: AEB→DFB→TDS→RFS→DCB의 신호 처리 사슬을 따라가며 어디서 DC vs AC 분리, anti-alias filter, decimation, FFT가 일어나는지 추적.
6. **Sect. 2.2.4 RFS**: PFB로 picket-fence 노이즈를 어떻게 처리하는지 주목 (Fig. 16의 RE02 EMC level).
7. **Sect. 3 Operations**: 90일 궤도(Fig. 18) 내 perihelion science(±6일), playback, burst select 단계 흐름을 파악.

**English**:
1. **Sect. 1 Introduction**: Place FIELDS among the four SPP suites (SWEAP, IS⊙IS, WISPR) and grasp their division of labor.
2. **Sect. 1.1-1.2 Plasma Environment**: Memorize the perihelion plasma parameters in Table 2 (|B|=2000 nT, n_e=7000 cm^-3, T_e=85 eV at 10 R_s).
3. **Sect. 1.3 Spacecraft Charging**: Understand photoelectron currents, the electrostatic barrier (Fig. 3), and wake formation — these justify the V1-V4 forward placement.
4. **Sect. 2.1.1-2.1.5 Sensors**: Catalog each sensor's design and heritage. The SCM response curves (Fig. 12) are essential for implementing the transfer function.
5. **Sect. 2.2 Electronics**: Trace AEB→DFB→TDS→RFS→DCB; locate DC/AC separation, anti-alias filtering, decimation, and FFT stages.
6. **Sect. 2.2.4 RFS**: Note how the polyphase filter bank handles picket-fence supply noise (Fig. 16 vs RE02 EMC level).
7. **Sect. 3 Operations**: Walk through the 90-day orbit cycle (Fig. 18) — perihelion science (±6 days), SSR playback, burst selection.

---

## 7. 현대적 의의 / Modern Significance

**한국어**:
이 논문은 2018년 8월 12일 발사된 Parker Solar Probe의 FIELDS 기기 사양과 거의 일치하며, 이후 PSP의 모든 fields/waves 논문의 기준 인용이 되었다. PSP는 2021년 4월 코로나 (Alfven critical surface 안쪽) 진입에 성공했고, "switchback" 자기장 반전 (Bale et al. 2019, *Nature*), 음향 모드 코로나 가열 흔적, 잠재 type III/II 폭발 영상 등 발견을 만들어냈다. Solar Orbiter RPW(같은 LESIA-Berkeley 헤리티지)와 미래의 Solar Wind Multi-scale Coronal Origins(SWMCO) 후속 미션에도 직접 영향을 주었다. Korean 우주물리학자에게도 KASI/UNIST 등이 PSP/FIELDS 데이터(L2/L3 CDF, ISTP-compliant)를 SPDF/CDAWeb에서 받아 분석할 수 있는 진입점이 된다.

**English**:
This paper essentially matches the as-built configuration of the FIELDS suite on Parker Solar Probe (launched 12 August 2018) and is the canonical citation for every PSP fields/waves study. PSP entered the corona (inside the Alfven critical surface) in April 2021, discovering magnetic switchbacks (Bale et al. 2019, *Nature*), suggestive coronal-heating signatures, and pristine type III/II bursts. The paper directly informed Solar Orbiter RPW (shared Berkeley/LESIA heritage) and the upcoming SWMCO line of missions. For Korean space physicists at KASI/UNIST, it is the doorway to PSP/FIELDS data (ISTP-compliant Level-2/Level-3 CDFs distributed via SPDF/CDAWeb).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
