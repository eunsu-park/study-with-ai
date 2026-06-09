---
title: "Pre-Reading Briefing: WAVES — The Radio and Plasma Wave Investigation on the WIND Spacecraft"
paper_id: "64_bougeret_1995"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# WAVES: The Radio and Plasma Wave Investigation on the WIND Spacecraft — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Bougeret, J.-L., Kaiser, M. L., Kellogg, P. J., Manning, R., Goetz, K., Monson, S. J., Monge, N., Friel, L., Meetre, C. A., Perche, C., Sitruk, L., and Hoang, S., "WAVES: The Radio and Plasma Wave Investigation on the WIND Spacecraft", *Space Science Reviews* **71**, 231–263 (1995). DOI: 10.1007/BF00751331
**Author(s)**: J.-L. Bougeret (Observatoire de Paris-Meudon, France), M. L. Kaiser (NASA/GSFC), P. J. Kellogg (University of Minnesota), and 9 co-authors
**Year**: 1995

---

## 1. 핵심 기여 / Core Contribution

**English.** This paper is the *instrument paper* for WIND/WAVES — the radio and plasma wave investigation on NASA's WIND spacecraft, launched 1 November 1994 as part of the International Solar-Terrestrial Physics (ISTP) program. WAVES provides comprehensive electric-field measurements from a fraction of a Hertz up to ~14 MHz and magnetic-field measurements up to 3 kHz, using three orthogonal electric dipole antennas (Ex = 100 m, Ey = 15 m, Ez = 12 m tip-to-tip) and three orthogonal magnetic search coils. The five receiver subsystems — FFT (DC–10 kHz), TNR (4–256 kHz), RAD1 (20–1040 kHz), RAD2 (1.075–13.825 MHz), and TDS (waveform sampling at up to 120 ksps) — together cover seven orders of magnitude in frequency. Two innovations are emphasized: (1) the first use of onboard neural networks to track the plasma line in the TNR and to select TDS events for telemetry, and (2) a wavelet-transform-like real-time spectral analysis. WAVES enabled remote tracking of type II/III solar radio bursts from ~3 R⊙ to 1 AU, in-situ thermal-noise diagnosis of solar-wind electron density (1–500 cm⁻³) and temperature (10³–10⁶ K), Langmuir-wave waveform capture in the foreshock, and goniopolarimetry of radio sources via spinning-spacecraft modulation.

**Korean.** 이 논문은 1994년 11월 1일 발사된 NASA WIND 위성에 탑재된 전파/플라즈마 파동 관측기 **WAVES**의 *기기 설계 논문*이다. WIND는 국제 태양-지구 물리 프로그램(ISTP)의 핵심 위성이다. WAVES는 1 Hz 미만에서 약 14 MHz까지의 전기장과 3 kHz까지의 자기장을 측정하며, 길이가 다른 세 개의 직교 전기 쌍극자 안테나(Ex = 100 m, Ey = 15 m, Ez = 12 m, tip-to-tip)와 세 개의 직교 자기 서치코일을 사용한다. 다섯 개의 수신기 서브시스템 — FFT(DC–10 kHz), TNR(4–256 kHz), RAD1(20–1040 kHz), RAD2(1.075–13.825 MHz), TDS(최대 120 ksps 파형 샘플링) — 가 합쳐 주파수 7자릿수를 덮는다. 두 가지 혁신이 강조된다: (1) **TNR의 플라즈마 라인 추적**과 **TDS의 이벤트 선별**을 위한 *온보드 신경망*의 최초 우주 응용, (2) 실시간 *웨이블릿 변환 유사* 스펙트럼 분석. WAVES는 태양 II형/III형 전파 폭발을 ~3 R⊙에서 1 AU까지 원격 추적, 열잡음 분광법으로 태양풍 전자 밀도(1–500 cm⁻³)와 온도(10³–10⁶ K)를 in-situ로 결정, 전조 충격(foreshock)에서의 Langmuir 파형 포착, 회전 위성을 활용한 전파원의 *goniopolarimetry*(방향·편파 측정)를 가능케 한다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**English.** By the early 1990s, in-situ plasma-wave instruments had flown on ISEE-3, Helios, Voyager, ICE, Ulysses, and others. The Bougeret group at Meudon, with Kellogg's group at Minnesota and Kaiser's at Goddard, had built the URAP/Ulysses experiment (Stone et al., 1992). WIND was conceived to provide the *ecliptic-plane companion* to Ulysses (then off the ecliptic), feeding into Geospace science and the upcoming SOHO and Cluster missions. The challenge: cover *all* radio/plasma-wave phenomena from the bow shock through the solar wind back to the solar corona, with one instrument suite. WAVES inherited Ulysses heritage (RAD1 receiver design, DC/DC converter) but added wholly new capabilities (FFT, TDS, TNR with neural-network tracking).

**Korean.** 1990년대 초까지 in-situ 플라즈마 파동 관측기는 ISEE-3, Helios, Voyager, ICE, Ulysses 등에 탑재된 바 있었다. Meudon의 Bougeret 그룹, 미네소타의 Kellogg 그룹, 고다드의 Kaiser 그룹은 이미 Ulysses의 URAP 실험(Stone et al., 1992)을 함께 만든 경험이 있었다. WIND는 황도면을 떠난 Ulysses의 *황도면 동반 위성*으로 구상되었으며, Geospace 과학과 곧 발사될 SOHO·Cluster 임무에 입력 데이터를 제공하는 역할을 맡았다. 도전 과제는 활꼴 충격면(bow shock)부터 태양풍을 거쳐 태양 코로나까지 모든 전파/플라즈마 파동 현상을 *하나의 기기군*으로 덮는 것이었다. WAVES는 Ulysses의 유산(RAD1 수신기 설계, DC/DC 컨버터)을 물려받으면서도 FFT, TDS, 신경망 추적 TNR과 같은 완전히 새로운 기능을 추가했다.

### 타임라인 / Timeline

```
1968 ── OGO-5: pioneering plasma-wave satellite (Scarf, Fredricks)
1974 ── Gurnett: AKR — Earth as a radio source (J. Geophys. Res. 79)
1977 ── Voyager 1/2 launched (PWS instrument)
1978 ── ISEE-1/2 launched; ISEE-3 (later ICE)
1985 ── Lacombe et al.: electron plasma waves upstream of bow shock
1990 ── Ulysses launched, URAP experiment (Stone et al. 1992)
1992 ── WIND construction & integration
1993 ── Final calibrations at U. Minnesota (Oct–Dec); paper submitted
1994 ── 1 Nov: WIND launched into ecliptic-plane orbit (apogee 250 R_E)
1995 ── Bougeret et al. published in SSR 71, 231 (this paper)
1995–   First WIND/WAVES discoveries: type II radio splitting, IP type III statistics
2002+ ── WIND moved to L1; still operating > 30 yr later
```

---

## 3. 필요한 배경 지식 / Prerequisites

**English.**
- *Plasma frequency*: $f_p \approx 8980 \sqrt{n_e}$ Hz (n_e in cm⁻³). At 1 AU, n_e ≈ 5–10 cm⁻³ ⇒ f_p ≈ 20–28 kHz.
- *Langmuir waves*: longitudinal electron oscillations at f_p, the plasma line.
- *Type II/III radio bursts*: solar emission at f_p (fundamental) and 2f_p (harmonic) generated by electron beams (III) or shocks (II) propagating outward; observed frequency drift maps source heliocentric distance via the coronal density model.
- *Thermal noise spectroscopy* (Meyer-Vernet & Perche 1989): the open-circuit voltage spectrum on a long dipole shows a broad maximum at f_p whose shape uniquely encodes (n_e, T_e).
- *Goniopolarimetry / direction-finding*: a spinning spacecraft modulates the received power by the antenna pattern; phase information across the spin yields source direction and Stokes parameters.
- *AKR, NTC, ITKR*: Auroral Kilometric Radiation, Non-Thermal Continuum, Isotropic Terrestrial Kilometric Radiation — known terrestrial radio sources from prior ISEE/Polar work.
- *Neural networks (1990s)*: feed-forward back-propagation networks; here implemented as integer-domain lookup tables for embedded use.

**Korean.**
- *플라즈마 주파수*: $f_p \approx 8980 \sqrt{n_e}$ Hz. 1 AU에서 n_e ≈ 5–10 cm⁻³ ⇒ f_p ≈ 20–28 kHz.
- *Langmuir 파*: 플라즈마 주파수 f_p에서의 종방향 전자 진동, 즉 *플라즈마 라인*.
- *II형/III형 전파 폭발*: 전자 빔(III) 또는 충격파(II)가 바깥으로 전파하면서 f_p(기본) 및 2f_p(고조파)에서 방출하는 태양 전파 방출; 관측 주파수의 시간 드리프트가 코로나 밀도 모델을 통해 원천의 태양 중심 거리로 환산된다.
- *열잡음 분광* (Meyer-Vernet & Perche 1989): 긴 쌍극자 안테나의 개방 전압 스펙트럼은 f_p에서 넓은 최대값을 가지며, 그 모양이 (n_e, T_e)를 유일하게 결정한다.
- *Goniopolarimetry / 방향 탐지*: 회전 위성에서는 안테나 패턴이 수신 전력을 변조하며, 자전 주기 동안의 위상 정보가 원천 방향과 Stokes 파라미터를 제공한다.
- *AKR, NTC, ITKR*: 오로라 킬로미터파 복사, 비열 연속체, 등방성 지상 킬로미터파 복사 — 선행 ISEE/Polar 연구에서 알려진 지구 전파원.
- *신경망 (1990년대)*: 순방향 역전파 신경망; 본 논문에서는 임베디드 환경을 위해 *정수 영역 룩업 테이블*로 구현됨.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **WAVES** | The Radio and Plasma Wave Investigation on WIND. 5 receivers + 6 antennas + 2 electronics stacks. / WIND의 전파·플라즈마 파동 관측기. 5개 수신기 + 6개 안테나 + 2개 전자장치 스택. |
| **TNR** (Thermal Noise Receiver) | 4–256 kHz, 5 bands × 32 ch, FIR digital filters. Measures plasma line for n_e, T_e diagnosis. / 4–256 kHz, 5밴드 × 32채널, FIR 디지털 필터; 플라즈마 라인을 측정해 n_e, T_e 진단. |
| **RAD1 / RAD2** | Dual super-heterodyne receivers, 20–1040 kHz / 1.075–13.825 MHz, 256 frequencies each, programmable hop scheduler. / 이중 슈퍼-헤테로다인 수신기, 20–1040 kHz / 1.075–13.825 MHz, 각 256주파수, 프로그램 가능한 hop 스케줄러. |
| **FFT receiver** | DC–10 kHz, 3 sub-bands (low/mid/high), 1024-pt floating-point FFT, 144 dB dynamic range. / DC–10 kHz, 3 서브밴드, 1024점 부동소수점 FFT, 144 dB 동적범위. |
| **TDS** (Time Domain Sampler) | Waveform capture, fast (120 ksps × 2 ch) + slow (7.5 ksps × 4 ch), 1 Mbit memory, neural-net event triage. / 파형 캡처, 빠른(120 ksps × 2채널) + 느린(7.5 ksps × 4채널) 샘플러, 1 Mbit 메모리, 신경망 이벤트 선별. |
| **DPU** (Data Processing Unit) | Sandia SA3300 / NS32016 master processor; flight software fully reloadable in orbit. / 마스터 프로세서; 비행 소프트웨어 전체가 궤도에서 재업로드 가능. |
| **SUM / SEP modes** | RAD1/RAD2 antenna combination: SUM = X+Z (synthetic inclined dipole for direction finding); SEP = X, Z separately. / RAD1/RAD2 안테나 결합 모드: SUM은 방향 탐지용 합성 경사 쌍극자, SEP는 분리. |
| **Plasma line** | The spectral feature near f_p in the thermal-noise spectrum. / 열잡음 스펙트럼에서 f_p 부근에 나타나는 분광 특징. |
| **Goniopolarimetry** | Measuring direction and polarization of a radio source from a spinning spacecraft. / 회전 위성에서 전파원의 방향과 편파를 동시에 측정하는 기법. |
| **AGC** (Automatic Gain Control) | Normalizes receiver input; 70 dB gain variation with 0.375 dB resolution after 8-bit quasi-log compression. / 수신기 입력 정규화; 70 dB 이득 변동, 8비트 의사로그 압축 후 0.375 dB 해상도. |
| **Floating-point ADC** | Mantissa (12 bit) + exponent (2 bit) ⇒ 144 dB dynamic range; used in FFT high/mid bands. / 가수(12bit) + 지수(2bit) ⇒ 144 dB 동적범위; FFT 고/중 대역에 사용. |
| **EMC / ESC** | Electromagnetic / Electrostatic Cleanliness — design discipline to ensure low-noise spacecraft. / 전자기·정전기 청결도 — 저잡음 위성을 위한 설계 원칙. |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Plasma frequency (the cornerstone) / 플라즈마 주파수 (초석)**

$$f_p = \frac{1}{2\pi}\sqrt{\frac{n_e e^2}{\varepsilon_0 m_e}} \approx 8980\sqrt{n_e[\text{cm}^{-3}]}\ \text{Hz}.$$

This sets the *boundary* between propagating EM waves (above f_p) and electrostatic waves (Langmuir, near f_p). Solar wind n_e ≈ 5 cm⁻³ ⇒ f_p ≈ 20 kHz, falling cleanly into the TNR low band. / f_p 위에서는 전자기파가 전파되고, f_p 부근에서는 전기 정적 Langmuir 파가 존재한다. 태양풍 n_e ≈ 5 cm⁻³ ⇒ f_p ≈ 20 kHz, TNR 저주파 밴드에 정확히 위치.

**(2) TNR receiver gain "log law" (Eq. quoted on p. 261)**

$$y = A_2\,\log_{10}\!\Big[10^{(A_1-x)/10} + 10^{-A_4(1/4 -1)}\Big] + A_3.$$

This is the calibration model fit to RAD1, RAD2 receivers. A₁ = saturation, A₂ = log slope, A₃ = offset, A₄ = noise-floor term. Numerical fitting recovers AGC gains and offsets. / RAD1, RAD2 수신기에 적합되는 보정 모델. AGC 이득·오프셋을 수치 적합으로 회수.

**(3) Type III radio drift ↔ heliocentric distance / III형 전파 드리프트 ↔ 태양 중심 거리**

$$f(t) = f_p[n_e(r(t))],\quad r(t) = r_0 + v_b\,t.$$

An electron beam at speed v_b ≈ 0.3 c sweeps outward; the local f_p (set by a coronal density model n_e(r)) decreases as r grows ⇒ characteristic high-to-low frequency drift. WAVES observes this from RAD2 (≥1 MHz, ~few R_⊙) down through RAD1 (≥20 kHz, near 1 AU). / 전자 빔(v_b ≈ 0.3 c)이 바깥으로 전파; 코로나 밀도 모델로 결정된 국지 f_p가 감소 ⇒ 고주파 → 저주파 드리프트. RAD2(≥1 MHz, 수 R_⊙)에서 RAD1(≥20 kHz, 1 AU 근방)까지 관측.

**(4) Goniopolarimetry on spinning spacecraft / 회전 위성에서의 goniopolarimetry**

For an antenna at instantaneous angle φ to source direction:

$$P(\phi) = \tfrac{1}{2}P_0(1+\cos^2\phi) + \frac{1}{2}P_Q\sin^2\phi\cos(2\phi - 2\psi) + \cdots$$

The DC (un-modulated), 2nd-harmonic, and phase of the spin modulation give source direction (θ, ϕ_s) and Stokes (I, Q, U, V). Manning–Fairberg (1980) and Fainberg et al. (1985) developed this for ISEE-3; SUM mode synthesizes the inclined dipole. / 안테나가 원천에 대해 순간 각도 φ를 가질 때, 회전 변조의 DC·2차 고조파·위상이 원천 방향과 Stokes 파라미터를 제공.

**(5) Floating-point ADC dynamic range / 부동소수점 ADC 동적범위**

$$\text{DR}_{\text{theory}}=20\log_{10}(2^{12}\cdot 4^3)\approx 144\ \text{dB}.$$

12-bit mantissa + 2-bit exponent (×1, ×16, ×256, ×4096) ⇒ ~144 dB. Realised value 110 dB (mid) and 128 dB (high). This dynamic range is essential to span thermal noise to bow-shock Langmuir wave amplitudes. / 12bit 가수 + 2bit 지수 ⇒ 144 dB 이론, 실현은 110/128 dB. 열잡음에서 활꼴 충격면 Langmuir 파까지 폭넓은 진폭을 동시 측정하려면 이런 동적범위가 필요.

---

## 6. 읽기 가이드 / Reading Guide

**English.** Read the paper in three passes:

1. **Science overview pass** (§1, §6). Understand *what WIND/WAVES is for* — Plasma Physics, Electron Diagnostics, Magnetosphere Remote Sensing, Interplanetary Remote Sensing. Make a mental map of phenomena → receiver: AKR ⇒ RAD1, type III bursts ⇒ RAD1+RAD2, Langmuir waves ⇒ TNR+TDS, magnetosheath turbulence ⇒ FFT.
2. **Engineering pass** (§2, §3). Treat each subsystem as a *signal-processing chain*: antenna → preamp → analog filter → ADC → digital processing → DPU. Pay attention to *sampling rates* (Table I, V), *bandwidths* (Table III, IV), and *physical mass/power* (Table II). The three FFT bands (low/mid/high) and the two TDS samplers (fast/slow) reflect a deliberate frequency-vs-time-resolution trade.
3. **Innovation pass** (§4.3, §3.3 on FFT). Read the neural-network and floating-point ADC sections carefully — these are the *novel design ideas* of WAVES that distinguish it from ISEE-3, Voyager, and Ulysses heritage.

Skip-light: detailed FIR filter coefficients, stack-level mass breakdown, EMC test history. Linger-heavy: the goniopolarimetry SUM mode (Fig. 5), the photoelectron-resistor compensation scheme (p. 244–245), and the neural-net plasma-line tracker (§4.3).

**Korean.** 논문은 세 단계로 읽기를 권장한다:

1. **과학 개요 패스** (§1, §6). WIND/WAVES가 *무엇을 위한* 기기인지 이해 — 플라즈마 물리, 전자 진단, 자기권 원격 관측, 행성간 원격 관측. *현상 → 수신기* 매핑: AKR ⇒ RAD1, III형 폭발 ⇒ RAD1+RAD2, Langmuir 파 ⇒ TNR+TDS, 자기초프 난류 ⇒ FFT.
2. **공학 패스** (§2, §3). 각 서브시스템을 *신호 처리 체인*으로 본다: 안테나 → 전치증폭기 → 아날로그 필터 → ADC → 디지털 처리 → DPU. 샘플링 속도(표 I, V), 대역폭(표 III, IV), 질량·전력(표 II)에 주목. FFT의 3 밴드(저/중/고)와 TDS의 2 샘플러(빠름/느림)는 *주파수 ↔ 시간 분해능* 절충의 의도적 결과.
3. **혁신 패스** (§4.3 신경망, §3.3 FFT). 신경망과 부동소수점 ADC 부분을 정독 — ISEE-3, Voyager, Ulysses와 차별화되는 *WAVES 고유의 설계 아이디어*.

생략 가능: 세부 FIR 계수, 스택 단위 질량 표, EMC 시험 이력. 정독 권장: 그림 5의 goniopolarimetry SUM 모드, p. 244–245의 광전자-저항 보상 회로, §4.3의 신경망 플라즈마 라인 추적기.

---

## 7. 현대적 의의 / Modern Significance

**English.** Thirty years on, WIND/WAVES remains a *workhorse* of heliophysics:

1. **Type II/III radio bursts**: the WAVES catalog (over 10⁴ events) is the gold standard for solar-energetic-particle SEP forecasting. The combination of WAVES + STEREO/WAVES (2006) + Parker Solar Probe/FIELDS (2018) + Solar Orbiter/RPW (2020) provides multi-spacecraft triangulation of CME shock-driven type II emission, directly mapping shock geometry from 1.5 R⊙ outward.
2. **Langmuir wave physics**: TDS waveforms enabled testing of strong/weak turbulence theories (Bale, Kellogg et al. 1996+), parametric decay verification, and Langmuir collapse studies that informed Cluster/STAFF analyses.
3. **Thermal noise as the standard**: the WIND TNR pipeline (Meyer-Vernet, Issautier, Maksimovic) became the template for in-situ density measurement on Cassini/RPWS, STEREO/WAVES, Parker/FIELDS, Solar Orbiter/RPW.
4. **Onboard intelligence**: the neural-network plasma-line tracker prefigured modern onboard ML — now standard on Solar Orbiter/RPW and PSP/FIELDS for shock detection.
5. **Goniopolarimetry**: the SUM-mode direction-finding template was reused on STEREO/WAVES (allowing the first stereoscopic radio imaging of CMEs) and Solar Orbiter/RPW.

WIND launched 1 Nov 1994 and is still flying at L1 — making the WAVES paper one of the *longest-impacting instrument papers* in space physics.

**Korean.** 30년이 지난 지금도 WIND/WAVES는 태양물리의 *기본 작업 도구*이다:

1. **II형/III형 전파 폭발**: WAVES 카탈로그(10⁴ 이상)는 태양 고에너지 입자(SEP) 예보의 표준이다. WAVES + STEREO/WAVES (2006) + Parker Solar Probe/FIELDS (2018) + Solar Orbiter/RPW (2020) 조합으로 CME 충격파 II형 방출의 다위성 삼각측량이 가능하며, 1.5 R⊙ 이상에서 충격파 형상을 직접 매핑한다.
2. **Langmuir 파 물리**: TDS 파형은 강·약 난류 이론(Bale, Kellogg 등 1996+)의 실험적 검증, 파라메트릭 붕괴, Langmuir 붕괴 연구의 토대를 제공했고 Cluster/STAFF 분석에도 활용되었다.
3. **열잡음을 표준으로**: WIND TNR 파이프라인(Meyer-Vernet, Issautier, Maksimovic)이 Cassini/RPWS, STEREO/WAVES, Parker/FIELDS, Solar Orbiter/RPW의 in-situ 밀도 측정 템플릿이 되었다.
4. **온보드 지능**: 신경망 플라즈마 라인 추적기는 오늘날 표준이 된 *온보드 ML*을 선구했고, Solar Orbiter/RPW, PSP/FIELDS의 충격 탐지에 계승되었다.
5. **Goniopolarimetry**: SUM 모드 방향 탐지 템플릿은 STEREO/WAVES에서 *최초의 입체 전파 CME 영상*을 가능케 했고, Solar Orbiter/RPW로 계승되었다.

WIND는 1994년 11월 1일 발사되어 지금도 L1에서 운영 중 — WAVES 논문은 우주 물리에서 *가장 오래 영향을 미친 기기 논문* 중 하나이다.

---

## Q&A

**Q1. Why three antennas of *different* lengths (100 m / 15 m / 12 m)?**
The longer Ex (100 m) maximizes thermal-noise SNR for low-f plasma diagnostics; but a 100 m antenna is not usable above its full-wave resonance ≈ 3.3 MHz, so the shorter Ey (15 m) is used by RAD2 (1.075–13.825 MHz). The axial Ez (12 m) is rigid for spin-stability reasons and gives the third dimension needed for direction finding. / 긴 Ex는 저주파 열잡음 SNR을 극대화하지만 전파장 공명(약 3.3 MHz) 위로는 사용 불가 ⇒ RAD2는 짧은 Ey(15 m) 사용. 자전축 Ez(12 m)는 안정성을 위해 강성이며 방향 탐지의 세 번째 축을 제공.

**Q2. Why a neural network onboard in 1994?**
Telemetry budget (936 bps low-rate) is the binding constraint. The TNR has 5 bands × 32 ch = 160 channels; only one band can be telemetered in detail. The neural net selects *which* band contains the plasma line in real time, achieving ~85% exact-channel and ~99% within-±1 channel accuracy from Ulysses RAR training data. For TDS, the network ranks waveform "events" by quality so the highest-priority is downlinked. This is *embedded ML* — feed-forward back-prop converted to integer lookup tables. / 텔레메트리 예산(936 bps)이 제약. TNR은 5밴드 × 32채널 = 160채널이지만 한 밴드만 상세 전송 가능. 신경망이 실시간으로 *어느 밴드*에 플라즈마 라인이 있는지 선택 — Ulysses RAR 데이터로 훈련, 정확도 약 85%(완전 일치) / 99%(±1채널). TDS도 파형 품질 순위를 매겨 우선 다운링크. 정수 룩업 테이블로 구현된 *임베디드 ML*.

**Q3. What does "wavelet-transform-like" mean here?**
The TNR digital part uses FIR filters at logarithmically spaced frequencies (32 channels per band), each with a passband 4.4% or 9% of the channel center frequency. This *constant-Q* filter bank is mathematically equivalent to a discretized wavelet transform — short windows at high frequency, long at low frequency. / TNR의 디지털 부분은 로그 등간격 주파수에 FIR 필터(밴드당 32채널)를 두며, 각 통과대역이 중심 주파수의 4.4% 또는 9%이다. 이러한 *상수 Q* 필터 뱅크는 수학적으로 이산화된 웨이블릿 변환과 동등 — 고주파에서 짧은 창, 저주파에서 긴 창.

**Q4. Why does photoemission matter for the antenna?**
In sunlight, the long wire dipole emits photoelectrons; in the spacecraft shadow it does not. The antenna potential can swing 10 V or more per spin, and the antenna-to-plasma resistance changes by ~10×. Without compensation, this would saturate preamps and corrupt low-frequency E-field. The fix: each antenna is biased through a programmable resistor (10 MΩ to 10 GΩ) with a programmable voltage in the range −10 to +10 V (256 steps). The *same* resistors enable in-flight antenna-impedance measurement. / 태양광 하 안테나는 광전자를 방출하지만 그림자에서는 방출하지 않음 ⇒ 안테나 전위가 자전마다 10 V 이상 흔들리고 안테나-플라즈마 저항이 약 10배 변동. 보상 없이는 전치증폭기가 포화되고 저주파 E-필드가 오염됨. 해결책: 각 안테나를 프로그램 가능한 저항(10 MΩ–10 GΩ)과 −10에서 +10 V 전압(256단계)으로 바이어싱; 같은 저항으로 비행 중 안테나 임피던스 측정도 수행.

**Q5. What is the cleanest "killer demonstration" experiment WAVES enables?**
Tracking a *single type III burst* from its onset (RAD2 ~10 MHz, ~3 R⊙) through the corona and solar wind (RAD1, 1 MHz–20 kHz), capturing the *associated* Langmuir wave when the electron beam reaches WIND (TDS waveforms at f_p ≈ 20 kHz), and simultaneously diagnosing the local n_e, T_e from TNR. This *radio-burst → Langmuir → particle* causal chain — predicted theoretically since the 1950s — was first cleanly observed at high cadence by WIND/WAVES + 3D-PLASMA. / 단일 III형 폭발을 그 *시작*(RAD2 ~10 MHz, ~3 R⊙)부터 코로나·태양풍을 거쳐(RAD1, 1 MHz–20 kHz), 전자 빔이 WIND에 도달했을 때의 *동반 Langmuir 파*(TDS 파형, f_p ≈ 20 kHz)까지 추적하고, 동시에 TNR로 국지 n_e, T_e를 진단. *전파 폭발 → Langmuir → 입자*의 이 인과 사슬은 1950년대 이후 이론적으로 예측되었지만, 고시간분해 관측은 WIND/WAVES + 3D-PLASMA가 처음 깨끗이 보여주었다.
