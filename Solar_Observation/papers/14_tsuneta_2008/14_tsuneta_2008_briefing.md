---
title: "Pre-Reading Briefing: The Solar Optical Telescope for the Hinode Mission: An Overview"
paper_id: "14_tsuneta_2008"
topic: Solar_Observation
date: 2026-04-16
type: briefing
---

# The Solar Optical Telescope for the Hinode Mission: An Overview — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Tsuneta, S. et al., "The Solar Optical Telescope for the Hinode Mission: An Overview", *Solar Physics*, Vol. 249, pp. 167–196, 2008. DOI: 10.1007/s11207-008-9174-z
**Author(s)**: S. Tsuneta, K. Ichimoto, Y. Katsukawa, S. Nagata, M. Otsubo, T. Shimizu, Y. Suematsu, M. Nakagiri, M. Noguchi, T. Tarbell, A. Title, R. Shine, W. Rosenberg, C. Hoffmann, B. Jurcevich, G. Kushner, M. Levay, B. Lites, D. Elmore, T. Matsushita, N. Kawaguchi, H. Saito, I. Mikami, L. D. Hill, J. K. Owens
**Year**: 2008

---

## 1. 핵심 기여 / Core Contribution

이 논문은 Hinode 위성(구 Solar-B)에 탑재된 Solar Optical Telescope(SOT)의 전체적인 개요를 제공합니다. SOT는 우주에 발사된 최대 구경(50 cm)의 태양 광학 망원경으로, 회절 한계 분해능(0.2–0.3 arcsec)을 달성하여 대기 왜곡 없이 광구와 채층의 고해상도 광도측정 및 벡터 자기장 관측을 가능하게 했습니다. SOT는 Optical Telescope Assembly(OTA)와 Focal Plane Package(FPP)로 구성되며, FPP에는 광대역 필터(BFI), 협대역 필터(NFI), Stokes 분광편광측정기(SP)가 포함됩니다. 0.01 arcsec rms 이하의 영상 안정화 시스템과 매우 안정적인 점확산함수(PSF)를 통해 지상 관측에서는 불가능한 연속적이고 균일한 품질의 관측 데이터를 제공합니다.

This paper provides a comprehensive overview of the Solar Optical Telescope (SOT) aboard the Hinode satellite (formerly Solar-B). SOT is the largest aperture (50 cm) solar optical telescope ever launched into space, achieving diffraction-limited resolution (0.2–0.3 arcsec) to enable high-resolution photometric and vector magnetic field observations of the photosphere and chromosphere free from atmospheric distortion. SOT consists of the Optical Telescope Assembly (OTA) and the Focal Plane Package (FPP), which includes broadband (BFI) and narrowband (NFI) filtergraphs, plus a Stokes spectro-polarimeter (SP). With an image stabilization system performing better than 0.01 arcsec rms and a very stable point spread function, SOT provides continuous, uniform-quality observational data impossible from the ground.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 Yohkoh 위성은 X선으로 태양 코로나를 관측하여 자기 재결합이 코로나 가열의 핵심 요소임을 밝혔지만, 코로나 가열의 구체적 메커니즘은 여전히 미해결이었습니다. 지상 망원경(Swedish Solar Telescope, Dunn Solar Telescope 등)이 적응광학으로 0.1 arcsec급 분해능을 달성했지만, 대기 seeing의 한계로 연속적이고 안정적인 편광측정 관측이 불가능했습니다. 태양 자기장의 미세 구조(0.1–0.2 arcsec 규모)를 이해하려면 대기 왜곡이 없는 우주 관측이 필수적이었습니다.

In the 1990s, the Yohkoh satellite observed the solar corona in X-rays and revealed magnetic reconnection as a key ingredient for coronal heating, but the specific mechanisms remained unsolved. Ground-based telescopes (SST, DST, etc.) achieved ~0.1 arcsec resolution with adaptive optics, but atmospheric seeing limited continuous, stable polarimetric observations. Understanding fine-scale solar magnetic structures (0.1–0.2 arcsec) required space-based observations free from atmospheric distortion.

### 타임라인 / Timeline

```
1991        Yohkoh 발사 — X선 태양 관측 / Yohkoh launch — X-ray solar obs.
1995-96     Solar-B/SOT 초기 개념 설계 시작 / SOT concept design begins
1998        TRACE 발사 — EUV 고해상도 코로나 영상 / TRACE launch — EUV imaging
2000        SST 첫 관측 — 지상 0.1" 분해능 달성 / SST first light — 0.1" from ground
2006 Sep    Solar-B 발사, Hinode로 명명 / Solar-B launch, renamed Hinode
2006 Oct    SOT first light — 회절 한계 영상 확인 / SOT first light confirmed
2008        ★ 본 논문 출판 / ★ This paper published
2010        SDO/HMI 발사 — 전일면 자기장 / SDO/HMI — full-disk magnetograms
2020        Solar Orbiter 발사 / Solar Orbiter launch
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 광학 / Optics
- **Gregorian 망원경**: 포물면 주경 + 타원면 부경으로 구성된 반사 망원경. 부경이 주초점 뒤에 위치하여 field stop 설치 가능 / A reflecting telescope with parabolic primary and ellipsoidal secondary mirrors. The secondary is behind the primary focus, allowing a field stop.
- **Strehl ratio**: 실제 PSF 최대값 대 이론적 회절 한계 PSF 최대값의 비율. 0.8 이상이면 "회절 한계" / Ratio of actual PSF peak to theoretical diffraction-limited PSF peak. ≥0.8 is considered "diffraction-limited."
- **Wavefront error (WFE)**: 이상적 파면 대비 실제 파면의 편차. Maréchal criterion: WFE rms < λ/14 ≈ Strehl > 0.8 / Deviation of actual wavefront from ideal. Maréchal criterion: WFE rms < λ/14 gives Strehl > 0.8.

### 편광측정 / Polarimetry
- **Stokes parameters (I, Q, U, V)**: 빛의 편광 상태를 완전히 기술하는 4개 매개변수. I=총 강도, Q,U=선형편광, V=원형편광 / Four parameters fully describing polarization state. I=total intensity, Q,U=linear polarization, V=circular polarization.
- **Zeeman 효과**: 자기장에 의한 스펙트럼선 분리. 선 분리량으로 자기장 세기를, Stokes 프로파일로 자기장 방향을 측정 / Spectral line splitting by magnetic field. Splitting measures field strength; Stokes profiles yield field direction.

### 선수 논문 / Prior Papers in Series
- **Paper #3** (Lyot, 1944): 코로나그래프 원리와 산란광 제어 / Coronagraph principle and stray light control
- **Paper #12** (Kosugi et al., 2007): Hinode 미션 전체 개요 / Hinode mission overview

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| OTA (Optical Telescope Assembly) | 50 cm 구경 Gregorian 망원경 본체. 주경, 부경, CLU, PMU, CTM-TM 포함 / 50 cm aperture Gregorian telescope body including primary/secondary mirrors, CLU, PMU, CTM-TM |
| FPP (Focal Plane Package) | 초점면 기기 패키지. BFI, NFI, SP, correlation tracker 포함 / Focal plane instrument package containing BFI, NFI, SP, and correlation tracker |
| BFI (Broadband Filter Imager) | 광대역 필터 영상기. 6개 파장대, 0.0541"/pixel, 218"×109" FOV / Broadband filter imager. 6 bands, 0.0541"/pixel, 218"×109" FOV |
| NFI (Narrowband Filter Imager) | 협대역 필터 영상기. Lyot 필터 사용, 10개 스펙트럼선, 0.08"/pixel, 328"×164" FOV / Narrowband filter imager using Lyot filter. 10 spectral lines, 0.08"/pixel, 328"×164" FOV |
| SP (Spectro-Polarimeter) | Stokes 분광편광측정기. Fe I 630.15/630.25 nm 이중선, 0.16"×151" 슬릿 / Stokes spectro-polarimeter. Fe I 630.15/630.25 nm dual lines, 0.16"×151" slit |
| PMU (Polarization Modulator Unit) | 편광 변조기. 회전 주기 T=1.6 s의 연속 회전 파장판 (석영+사파이어) / Polarization modulator. Continuously rotating waveplate (quartz+sapphire) with period T=1.6 s |
| CTM-TM (Tip-Tilt Mirror) | 영상 안정화용 경사 거울. piezo 구동, 14 Hz 대역폭 / Tip-tilt mirror for image stabilization. Piezo-driven, 14 Hz bandwidth |
| CT (Correlation Tracker) | 상관 추적기. 580 Hz CCD로 태양립 패턴 추적하여 jitter 검출 / Correlation tracker. Tracks granulation pattern at 580 Hz to detect image jitter |
| HDM (Heat Dump Mirror) | 열 덤프 거울. FOV 밖의 태양광(~1500 solar) 반사하여 우주로 방출 / Heat dump mirror. Reflects unused sunlight (~1500 solar) outside FOV to space |
| CLU (Collimator Lens Unit) | 시준 렌즈 유닛. 6매 렌즈, f=37 cm, 평행광으로 FPP에 전달 / Collimator lens unit. 6 lenses, f=37 cm, delivers parallel light to FPP |
| OBU (Optical Bench Unit) | 광학 벤치 유닛. OTA, FPP, XRT, EIS를 탑재하는 위성 구조체 / Optical bench unit. Satellite structure mounting OTA, FPP, XRT, EIS |
| MDP (Mission Data Processor) | 미션 데이터 처리기. 관측 테이블 실행, 데이터 압축, 명령 제어 / Mission data processor. Executes observation tables, data compression, command control |

---

## 5. 수식 미리보기 / Equations Preview

### 회절 한계 분해능 / Diffraction-Limited Resolution

$$\theta = 1.22 \frac{\lambda}{D}$$

- $\theta$: 각분해능 (radian) / Angular resolution (radian)
- $\lambda$: 파장 / Wavelength
- $D$: 구경 (= 50 cm for SOT) / Aperture diameter
- 500 nm에서: $\theta = 1.22 \times 500 \times 10^{-9} / 0.5 = 1.22 \times 10^{-6}$ rad $\approx 0.25$ arcsec

### Strehl Ratio

$$S = \frac{I_{\text{peak, actual}}}{I_{\text{peak, ideal}}} \approx e^{-(2\pi \sigma / \lambda)^2}$$

- $S$: Strehl ratio (SOT 목표: > 0.8, OTA 달성: > 0.9 at 500 nm)
- $\sigma$: wavefront error rms / 파면 오차 rms
- Maréchal criterion: $S \geq 0.8$ ↔ $\sigma \leq \lambda/14$

### Stokes 편광 변조 / Stokes Polarization Modulation

PMU 회전 주기 $T = 1.6$ s에서 Stokes 매개변수의 변조:
With PMU rotation period $T = 1.6$ s, Stokes parameters are modulated as:

- $Q, U$: 주기 $T/4$에서 정현파 변조 (4 cycles/revolution) / Sinusoidal modulation at period $T/4$
- $V$: 주기 $T/2$에서 변조 (2 cycles/revolution) / Modulation at period $T/2$
- $Q$와 $U$ 사이 위상차: 22.5° / Phase difference between $Q$ and $U$: 22.5°
- 1회전당 16회 샘플링으로 복조 / Demodulated by sampling 16 times per revolution

### Lyot 필터 분광 대역폭 / Lyot Filter Spectral Bandwidth

NFI의 Lyot 필터 대역폭: $\approx 95$ mÅ at 630 nm

$$\text{FWHM} \approx 95 \text{ m\AA} \quad (\text{at } 630 \text{ nm})$$

### SP 감도 / SP Sensitivity

- 종방향 자기장 감도: 1–5 G (시선 방향) / Longitudinal field sensitivity: 1–5 G (line-of-sight)
- 횡방향 자기장 감도: 30–50 G / Transverse field sensitivity: 30–50 G

---

## 6. 읽기 가이드 / Reading Guide

### 집중해서 읽을 부분 / Focus Areas

1. **Section 1 (Introduction)**: Hinode 미션의 과학적 동기와 SOT의 위치 파악. Yohkoh에서 SOT로의 과학적 연결고리 이해 / Scientific motivation and SOT's role. Understand the science link from Yohkoh to SOT.

2. **Section 2 (Science Overview)**: 6개 핵심 과학 주제(코로나 가열, 활동영역, flux tube, 데이터 기반 시뮬레이션, 채층 역학, 자기 재결합) 파악 / Grasp the 6 key science topics. These define what SOT was designed to observe.

3. **Section 4.1–4.2 (OTA Optics & Polarization Modulation)**: 가장 기술적으로 중요한 부분. Gregorian 설계, ULE 거울, CLU, PMU의 작동 원리 / Most technically important. Understand Gregorian design, ULE mirrors, CLU, and PMU operation.

4. **Section 5 (Observing Modes)**: BFI, NFI, SP 각각의 관측 모드와 데이터 산물 이해 / Understand observing modes and data products of BFI, NFI, SP.

5. **Section 6 (Image Stabilization)**: 0.007" 안정성 달성의 핵심 — correlation tracker + tip-tilt mirror 시스템 / Key to achieving 0.007" stability — the CT + tip-tilt mirror system.

### 가볍게 읽을 부분 / Skim

- **Section 4.3 (Optical Testing)**: 지상 시험 과정의 상세 (흥미롭지만 과학적 핵심은 아님) / Ground testing details (interesting but not core science)
- **Section 4.4–4.5 (Structure/Contamination)**: 공학적 세부사항 / Engineering details
- **Section 7 (Control & Data Flows)**: 운용 절차 상세 / Operational procedure details

### 읽기 전략 / Reading Strategy

이 논문은 "개요" 논문이므로, 각 하위 시스템(OTA, FPP, BFI, NFI, SP, image stabilization)의 **핵심 성능 수치**를 표로 정리하면서 읽는 것이 효과적입니다. 특히 Table 1(OTA 사양), Table 2(FG 사양), Table 3(SP 사양), Table 5(image stabilization 사양)에 주목하세요.

This is an "overview" paper, so it's effective to read while tabulating **key performance numbers** for each subsystem (OTA, FPP, BFI, NFI, SP, image stabilization). Pay special attention to Table 1 (OTA specs), Table 2 (FG specs), Table 3 (SP specs), and Table 5 (image stabilization specs).

---

## 7. 현대적 의의 / Modern Significance

Hinode/SOT는 2006년 발사 이후 약 20년간 운용되며 태양 물리학의 여러 분야에 획기적인 기여를 했습니다:

Hinode/SOT has been operating for ~20 years since its 2006 launch, making groundbreaking contributions to multiple areas of solar physics:

- **광구 자기장의 미세 구조**: SOT/SP 데이터는 quiet Sun의 수평 자기장이 수직 자기장보다 우세하다는 발견을 이끌어 "숨겨진 자기장(hidden flux)" 문제를 부각시켰습니다 / **Fine-scale photospheric magnetic fields**: SOT/SP data led to the discovery that horizontal fields dominate vertical fields in the quiet Sun, highlighting the "hidden flux" problem.

- **SDO/HMI와의 상보성**: SDO/HMI(2010)가 전일면 관측을 제공하는 반면, SOT/SP는 여전히 최고의 공간분해능 벡터 자기장 데이터를 제공합니다 / **Complementarity with SDO/HMI**: While SDO/HMI (2010) provides full-disk observations, SOT/SP still delivers the highest spatial resolution vector magnetic field data.

- **채층 역학 연구의 기초**: SOT의 Ca II H 영상과 Hα 관측은 spicule, 채층 제트, 에너지 전달 메커니즘 연구의 기반이 되었습니다 / **Foundation for chromospheric dynamics research**: SOT's Ca II H and Hα observations became the basis for studying spicules, chromospheric jets, and energy transfer mechanisms.

- **차세대 기기 설계의 참조**: DKIST(4 m 지상 망원경), Solar Orbiter/PHI, 그리고 계획 중인 우주 태양 망원경의 설계에 SOT의 경험이 반영되었습니다 / **Reference for next-generation instrument design**: SOT's experience informed the design of DKIST (4 m ground telescope), Solar Orbiter/PHI, and planned space solar telescopes.

- **NFI bubble 문제의 교훈**: 궤도상에서 발견된 Lyot 필터 내 기포 문제는 우주 기기 개발에서 예기치 않은 환경 효과에 대한 중요한 교훈을 제공했습니다 / **Lesson from NFI bubble issue**: The air bubble problem discovered in orbit provided an important lesson about unexpected environmental effects in space instrument development.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
