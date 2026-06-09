---
title: "Pre-Reading Briefing: The WIND Magnetic Field Investigation"
paper_id: "61_lepping_1995"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# The WIND Magnetic Field Investigation: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Lepping, R. P., Acuña, M. H., Burlaga, L. F., Farrell, W. M., Slavin, J. A., Schatten, K. H., Mariani, F., Ness, N. F., Neubauer, F. M., Wang, Y. C., Byrnes, J. B., Kennon, R. S., Panetta, P. V., Scheifele, J., and Worley, E. M., "The WIND Magnetic Field Investigation," *Space Science Reviews*, **71**, 207–229, 1995. DOI: 10.1007/BF00751330
**Author(s)**: R. P. Lepping et al. (NASA/GSFC, NSF, U. Tor Vergata, U. Delaware Bartol, U. Köln, Catholic U. America)
**Year**: 1995

---

## 1. 핵심 기여 / Core Contribution

이 논문은 NASA의 ISTP (International Solar-Terrestrial Physics) 프로그램의 핵심 우주선인 WIND 위성에 탑재된 자기장 탐사장비 (Magnetic Field Investigation, MFI)의 설계, 성능, 과학 목표, 그리고 지상 자료 처리 시스템을 종합적으로 기술한다. MFI는 12 m 붐 (boom) 끝과 그 중간에 장착된 듀얼 (dual) 삼축 (triaxial) 플럭스게이트 (fluxgate) 자력계 시스템으로, ±0.001 nT의 디지털 분해능, <0.006 nT r.m.s.의 잡음 수준, ±4 nT부터 ±65,536 nT까지 8개의 자동 동적 범위 (dynamic range), 그리고 256-점 FFT 처리기 및 256 kbit 스냅샷 메모리를 포함한다. 이 장비는 IMF (interplanetary magnetic field)의 대규모 구조 (sector boundary, magnetic clouds), 중규모 (interplanetary ejecta, plasmoids), 미세 규모 (Alfvén wave, MHD discontinuity), 그리고 운동 규모 (shock ramp)까지 광범위한 시간/공간 스케일을 동시에 분석하도록 설계되었다.

This paper comprehensively describes the design, performance, scientific objectives, and ground data processing system of the Magnetic Field Investigation (MFI) instrument aboard NASA's WIND spacecraft, the heliocentric pillar of the ISTP (International Solar-Terrestrial Physics) program. MFI is a dual triaxial fluxgate magnetometer system, with one sensor mounted at the end of a 12-m boom and a second sensor at approximately 2/3 of the boom length, providing ±0.001 nT digital resolution, <0.006 nT r.m.s. noise, eight automatic dynamic ranges from ±4 nT to ±65,536 nT, an on-board 256-point FFT processor, and a 256 kbit snapshot memory. The instrument is designed to study a vast hierarchy of interplanetary magnetic field phenomena spanning large-scale (sector boundaries, magnetic clouds), meso-scale (interplanetary ejecta, plasmoids), micro-scale (Alfvén waves, MHD discontinuities), and kinetic-scale (shock ramps) features simultaneously.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 초는 ISTP 프로그램이 전성기를 맞이한 시기로, 태양–태양풍–자기권–전리권 연결 (solar-terrestrial coupling)을 다중 위성 (multi-spacecraft)으로 동시에 관측하려는 야심찬 국제 협력 프로그램이 가동되었다. WIND (1994년 발사 예정), POLAR, GEOTAIL (일본 ISAS), CLUSTER (ESA), SOHO (NASA-ESA)가 주요 구성원이었다. WIND는 라그랑주 L1 점 근방에서 IMF와 태양풍의 *upstream monitor* 역할을 맡았다. 이 시점에서 자력계 기술은 Voyager (1977), ISPM/Ulysses (1990), Giotto (1985), Mars Observer (1992)에 이르기까지 GSFC의 Acuña 그룹이 축적한 30년 가까운 경험의 정점에 있었다.

The early 1990s was the heyday of the ISTP program — an ambitious international collaboration designed to observe the solar–solar wind–magnetosphere–ionosphere chain simultaneously with multiple spacecraft. WIND (planned launch 1994), POLAR, GEOTAIL (Japan/ISAS), CLUSTER (ESA), and SOHO (NASA-ESA) were the key members. WIND's role was to serve as the upstream monitor of the IMF and solar wind near the L1 Lagrange point. By this time, magnetometer technology had reached its peak after nearly three decades of GSFC experience accumulated through Voyager (1977), ISPM/Ulysses (1990), Giotto (1985), and Mars Observer (1992).

### 타임라인 / Timeline

```
1971  Ness et al.   : Dual-magnetometer concept (IMP/Explorer-43)
1974  Acuña         : Ring-core fluxgate sensor formalised
1977  Voyager 1/2   : Triaxial fluxgate to outer planets
1981  Burlaga et al.: Discovery of magnetic clouds
1985  Giotto        : Comet Halley fluxgate
1990  Ulysses (ISPM): Polar heliosphere
1992  Mars Observer : Latest dual fluxgate heritage
1993  WIND MFI submitted (16 March 1993)
1994  WIND launch (1 November 1994)
1995  Paper published — instrument description
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Fluxgate magnetometer principle / 플럭스게이트 자력계 원리**: ferromagnetic ring-core saturation, second-harmonic detection, null-feedback operation. Reference: Acuña (1974), Ness (1970).
- **Triaxial sensor geometry / 삼축 센서 기하**: orthogonal three-axis arrangement; spacecraft spin-plane vs. spin-axis components.
- **Coordinate systems / 좌표계**: Spacecraft (S/C) spinning frame, GSE (Geocentric Solar Ecliptic), GSM (Geocentric Solar Magnetospheric).
- **FFT / DSP basics / FFT·DSP 기초**: 256-point FFT, log-spaced frequency bands, μ-law compression, Hanning window, cosine taper.
- **Solar wind structures / 태양풍 구조**: shocks, sector boundaries, magnetic clouds (Burlaga 1991), CMEs, Alfvén waves, tangential/rotational discontinuities.
- **ISTP architecture / ISTP 구조**: WIND, POLAR, GEOTAIL, CLUSTER coordinated mission; Key Parameter (KP) data, CDHF (Central Data Handling Facility).
- **Bit/dynamic range arithmetic / 비트·동적 범위 계산**: 12-bit ADC = 4096 counts, 72 dB dynamic range, eight discrete ranges from ±4 nT to ±65,536 nT (factor of 4 between ranges).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| MFI | Magnetic Field Investigation. WIND의 자기장 탐사 장비. WIND's magnetic field instrument suite. |
| Dual triaxial fluxgate | 12 m 붐의 끝(OB)과 중간(IB)에 장착된 두 개의 3축 플럭스게이트 센서. Two 3-axis fluxgate sensors at boom end (OB, outboard) and mid-boom (IB, inboard). |
| Boom | 우주선 본체에서 자기 잡음을 줄이기 위해 센서를 멀리 두는 12 m astromast 신축 구조. 12 m astromast that places sensors far from spacecraft magnetic noise. |
| Range switching | ±4 nT → ±65,536 nT 8단계 자동 다이나믹 레인지 절환. Eight discrete ranges with automatic switching when output exceeds 7/8 or drops below 1/8 of full scale. |
| Snapshot memory | 256 kbit 트리거 기반 고시간 분해능 메모리(약 165 s 데이터). 256 kbit triggered high-rate buffer (∼165 s of 44 vec/s data). |
| FFT processor | TI320C10 DSP 기반, 256-point FFT, 32 logarithmic spectral channels, μ-law compression. TI320C10 DSP-based 256-point FFT engine producing 32 log-spaced channels with μ-law compression. |
| Key Parameter (KP) | ISTP의 92 s마다 1 vector 의 빠른 dissemination data. Real-time-ish data at 1 vector / 92 s for ISTP-wide situational awareness. |
| GSE / GSM | Geocentric Solar Ecliptic / Solar Magnetospheric coordinates — 표준 우주물리 좌표계. Standard heliospheric/magnetospheric coordinate frames. |
| Magnetic cloud | Burlaga (1981) 정의: 부드럽게 회전하는 강한 B, 낮은 β, 약 12 hr 크기. Burlaga's smooth-rotation, strong-B, low-β interplanetary structure of ∼12 hr duration. |
| Plasmoid | (X-Z)_GSM 평면에서 타원형 단면을 가지는 자기꼬리 자기-플라즈마 구조. Oval cross-section magnetic-plasma structure in the magnetotail; ∼60 R_E in length. |
| SSC | Storm Sudden Commencement. 행성간 충격파의 지상 자기 표현. Sudden geomagnetic onset signaling an interplanetary shock impact. |
| DPU | Digital Processing Unit, 80C86 마이크로프로세서, smart-system 운영 제어. 80C86-based smart microcontroller managing all instrument operations. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) Fluxgate output equation / 플럭스게이트 출력 방정식

$$E_o \;=\; G \,(B_{\text{ambient}} - B_{\text{feedback}}) \;\xrightarrow{\text{null}}\; I_{fb} = \frac{B_{\text{ambient}}}{k_{\text{coil}}}$$

자속이 0이 되도록 피드백 코일 전류가 외부 자기장에 비례하도록 만들어진다. The feedback current is forced to be proportional to the ambient field by nulling the sensed flux.

### (2) Boom magnetic-field separation / 듀얼 자력계 자기장 분리

$$\mathbf{B}_{\text{ambient}} \;=\; \frac{r_{IB}^{3}\,\mathbf{B}_{IB} - r_{OB}^{3}\,\mathbf{B}_{OB}}{r_{IB}^{3} - r_{OB}^{3}}$$

쌍극자 (dipolar) 우주선 자기장 기여가 거리^3 로 떨어진다는 사실을 이용해 두 센서 측정값에서 환경 자기장을 분리한다. Exploits the 1/r³ falloff of the spacecraft dipolar field to algebraically remove S/C contamination from the OB and IB measurements.

### (3) Dynamic range / 동적 범위 (dB)

$$\text{DR} \;=\; 20\log_{10}\!\left(\frac{\pm 65{,}536}{\pm 0.001}\right) \;\approx\; 156\text{ dB},\quad \text{single 12-bit ADC} \to 72\text{ dB}$$

8개 레인지의 전환으로 단일 ADC의 72 dB를 실효적으로 156 dB로 확장한다. Switching among 8 ranges effectively extends the 72 dB native ADC to ∼156 dB end-to-end.

### (4) Coordinate transformation GSE → GSM / 좌표 변환

$$\mathbf{B}_{\text{GSM}} \;=\; R_x(\mu)\,\mathbf{B}_{\text{GSE}}, \qquad \tan\mu \;=\; \frac{Y_{\text{GSE}}^{\text{dipole}}}{Z_{\text{GSE}}^{\text{dipole}}}$$

X-축은 공유, dipole tilt 각 $\mu$ 만큼 X-축 둘레로 회전. Shares the X-axis; differs only by a rotation about X by the dipole-tilt angle μ.

### (5) Magnetic cloud Bz signature / 자기 구름 Bz 신호

$$B_z(t)\;=\;B_0\,\cos\!\left(\pi\,\frac{t-t_0}{\Delta T}\right),\quad \Delta T \approx 12\text{ hr}$$

자기 구름 통과 동안 Bz가 부드럽게 회전하여 약 절반의 기간 동안 강한 음 (southward)을 유지 → 강한 지자기 폭풍 유발. During cloud passage, Bz rotates smoothly so that for roughly half its duration Bz is strongly southward — the principal driver of major geomagnetic storms.

---

## 6. 읽기 가이드 / Reading Guide

논문은 크게 두 줄기로 읽으면 된다: **(A) Science Objectives (Sec 2-3)** 와 **(B) Instrument & Ground Data System (Sec 4-7)**. 처음 읽을 때는 (A)는 Burlaga의 magnetic cloud, Tsurutani의 substorm trigger, Lepping et al. (1992)의 SSC 전파 시간 결과 등 *citation context*를 따라 읽는 것이 좋다. (B)는 Table I (instrument summary), Fig. 1 (block diagram), Fig. 3a/3b (noise), Fig. 4 (range switching), Table II (telemetry modes)를 차례로 보면서 *spec sheet*를 만든다고 생각하면 효과적이다. 특히 (i) 듀얼 자력계의 1/r³ 트릭이 왜 ±0.1 nT 정확도를 가능케 하는지, (ii) snapshot memory의 165 s 버퍼 중 82 s가 *pre-trigger*라는 사실이 왜 shock 연구에 결정적인지, (iii) FFT의 μ-law 압축이 왜 telemetry 절감에 필수인지를 본인의 언어로 정리하라.

The paper has two natural threads: **(A) Science Objectives (Secs 2–3)** and **(B) Instrument & Ground Data System (Secs 4–7)**. On first reading, follow (A) by chasing citation context — Burlaga's magnetic clouds, Tsurutani's substorm triggers, Lepping et al. (1992)'s SSC propagation timing. For (B), proceed sequentially through Table I (instrument summary), Fig. 1 (block diagram), Figs. 3a/3b (noise), Fig. 4 (range switching), and Table II (telemetry modes), treating the section as if you were assembling a spec sheet. Pay particular attention to (i) why the dual-magnetometer 1/r³ trick yields ±0.1 nT cleanliness, (ii) why the 165 s snapshot buffer with 82 s pre-trigger is critical for shock physics, and (iii) why μ-law FFT compression is mandatory for telemetry budget. State each in your own words.

---

## 7. 현대적 의의 / Modern Significance

WIND/MFI는 현재까지(2026년 기준 32년차) 운용 중인 가장 장수한 태양풍 자기장 모니터로, NOAA SWPC의 우주 기상 예보, DSCOVR의 검증 비교 (cross-calibration), Parker Solar Probe와 Solar Orbiter의 배경 IMF 참조에 모두 활용된다. 공간 기상 운영의 두 축인 (1) IMF Bz의 실시간 (real-time) 감시와 (2) 자기 구름 사전 검출 (forewarning)은 본 논문이 정의한 Key Parameter 데이터 흐름의 직계 후손이다. 또한 MFI의 듀얼-자력계 + 듀얼 DPU 완전 이중화 (full redundancy) 설계 철학은 이후 STEREO, Parker Solar Probe FIELDS, Solar Orbiter MAG 등 모든 후속 GSFC 자력계의 표준이 되었다.

WIND/MFI is the longest-running solar-wind magnetic-field monitor still operating (32+ years as of 2026), and is used by NOAA SWPC for space-weather forecasts, by DSCOVR for cross-calibration, and by Parker Solar Probe and Solar Orbiter as the IMF reference. The two pillars of operational space weather — (1) real-time monitoring of IMF Bz and (2) magnetic-cloud forewarning — are direct descendants of the Key Parameter data stream defined in this paper. Furthermore, the MFI design philosophy of dual sensors plus dual DPU full redundancy has become the standard template for all subsequent GSFC magnetometers (STEREO, Parker Solar Probe FIELDS, Solar Orbiter MAG).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
