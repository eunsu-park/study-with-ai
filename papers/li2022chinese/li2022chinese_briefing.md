---
title: "Pre-Reading Briefing: The Chinese Hα Solar Explorer (CHASE) mission: An overview"
paper_id: "60_li_2022"
topic: Solar_Observation
date: 2026-04-27
type: briefing
---

# The Chinese Hα Solar Explorer (CHASE) mission: An overview — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: C. Li, C. Fang, Z. Li, M. D. Ding, P. F. Chen, et al., "The Chinese Hα Solar Explorer (CHASE) mission: An overview," *Sci. China Phys. Mech. Astron.* **65**, 289602 (2022). [DOI: 10.1007/s11433-022-1893-3](https://doi.org/10.1007/s11433-022-1893-3)
**Authors**: Chuan Li, Cheng Fang, Zhen Li, MingDe Ding, PengFei Chen, Ye Qiu, et al. (40+ authors, primarily Nanjing University)
**Year**: 2022 (launched October 14, 2021)

---

## 1. 핵심 기여 / Core Contribution

**한국어**
CHASE(Chinese Hα Solar Explorer, 중국명 "Xihe/羲和" — 태양의 여신)는 2021년 10월 14일에 발사된 **중국 최초의 태양 우주 관측 미션**이다. 본 논문은 (1) CHASE의 과학 목표, (2) 핵심 탑재체인 Hα Imaging Spectrograph (HIS) 기기 사양, (3) 데이터 보정/처리 흐름(Level 0 → Level 1), (4) 첫 궤도(on-orbit) 관측 결과를 종합 소개한다. HIS는 두 가지 관측 모드 — **Raster Scanning Mode (RSM)** 로 6559.7–6565.9 Å (Hα)와 6567.8–6570.6 Å (Fe I)에서 0.024 Å 화소 분광 분해능과 1분 시간 분해능으로 풀-Sun 분광 영상을, **Continuum Imaging Mode (CIM)** 로 6689 Å 부근에서 13.4 Å FWHM의 광구 연속체 영상을 — 동시에 제공한다. CHASE는 **우주 기반 풀-Sun Hα 분광 관측의 거의 유일한 사례**(이전엔 일본 SDDI 정도)로, 광구–채층 동역학과 태양 폭발 메커니즘 연구에 새로운 데이터 차원을 연다.

**English**
CHASE (Chinese Hα Solar Explorer, dubbed "Xihe/羲和" — Goddess of the Sun) is **China's first solar space mission**, launched October 14, 2021. This paper provides an overview of (1) CHASE's scientific objectives, (2) its scientific payload — the Hα Imaging Spectrograph (HIS) — instrument design and parameters, (3) the data calibration/processing pipeline (Level 0 → Level 1), and (4) first on-orbit observational results. HIS operates in two simultaneous modes: **Raster Scanning Mode (RSM)** acquires full-Sun spectroscopic images in 6559.7–6565.9 Å (Hα) and 6567.8–6570.6 Å (Fe I) at 0.024 Å pixel spectral resolution and 1-min temporal resolution, while **Continuum Imaging Mode (CIM)** captures photospheric continuum images near 6689 Å with 13.4 Å FWHM. CHASE represents one of the very few **space-based full-Sun Hα spectroscopic platforms** (previously only Japan's SDDI), opening a new observational dimension for photosphere–chromosphere dynamics and solar eruption mechanisms.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어**
Hα 선(6562.8 Å)은 태양 분광 관측의 가장 중요한 광학선 중 하나이다. 선 중심(line center)은 채층(chromosphere) 정보를, 선 날개(far wings)는 광구(photosphere) 정보를 담아, **단일 분광 프로파일로 태양 하부 대기를 연직 단면으로 진단**할 수 있다. 1900년대 초 Hale의 Mount Wilson 분광태양사진(spectroheliograph)부터 시작된 Hα 분광 관측은 100년 넘게 주로 지상 망원경(특히 일본 Hida, 러시아 Kislovodsk 등)에 의존해왔다.

그러나 지상 관측은 (1) 대기 시상(seeing) 한계, (2) 전천 관측 불가, (3) 전천후 불가라는 본질적 제약을 안는다. 우주에서 풀-Sun Hα 분광 관측을 수행한 미션은 일본의 **SDDI (Solar Dynamics Doppler Imager, Hida observatory에 설치, 사실상 지상)** 와 러시아 **Kislovodsk spectroheliograph (분해능 0.16 Å)** 외에는 거의 존재하지 않았다. CHASE는 이 공백을 메우기 위해 추진되었으며, 동시에 **중국이 우주 태양물리 시대로 진입**하는 이정표(milestone) 역할을 한다.

CHASE는 발사 후 곧이어 진행된 (1) FY-3E의 X-선/EUV 영상기(X-EUVI, 2021년 7월), (2) ASO-S (Advanced Space-based Solar Observatory, 2022년 발사 예정)와 함께 중국 태양 우주 미션 군의 일원이다.

**English**
The Hα line (6562.8 Å) is one of the most important optical lines for solar observation. Its line center carries chromospheric information while the far wings carry photospheric information, allowing a **single spectral profile to vertically diagnose the lower solar atmosphere**. Since George Ellery Hale's spectroheliograph at Mount Wilson in the early 1900s, Hα spectroscopic observation has relied for over a century mostly on ground-based telescopes (notably Hida in Japan and Kislovodsk in Russia).

But ground-based observation suffers from intrinsic limits: (1) atmospheric seeing, (2) inability to observe all-day, (3) weather dependence. Before CHASE, the only space-relevant full-Sun Hα spectroscopic platforms were essentially Japan's **SDDI (Solar Dynamics Doppler Imager, mounted at Hida)** and Russia's **Kislovodsk spectroheliograph** (0.16 Å resolution). CHASE was launched to fill this gap and simultaneously marks **China's entry into the era of solar space missions**.

CHASE complements other Chinese solar space missions: (1) FY-3E's X-EUVI (launched July 2021), and (2) ASO-S (Advanced Space-based Solar Observatory, scheduled 2022).

### 타임라인 / Timeline

```
1900s ─── Hale: Mount Wilson spectroheliograph (first full-Sun Hα images)
   │
1960s ─── Ground-based Hα networks expand (Hida, Big Bear, Kislovodsk)
   │
1970s ─── Skylab: first space-based Hα-like observations (limited)
   │
1995  ─── SOHO launched (EUV/UV, but no dedicated Hα spectroscopy)
   │
2006  ─── Hinode launched (SOT NFI Hα filtergrams, not full-Sun spectra)
   │
2010  ─── SDO launched (HMI continuum + AIA EUV; no Hα)
   │
2013  ─── IRIS launched (UV spectroscopy 1300–2800 Å, no Hα)
   │
2016  ─── Hida SDDI installed (full-Sun Hα Doppler, but ground-based)
   │
2020  ─── Solar Orbiter launched (in-situ + remote, no Hα spectroscopy)
   │
2021  ─── ★ CHASE launched (Oct 14) — China's first solar space mission
   │       FY-3E X-EUVI launched (July)
   │
2022  ─── ASO-S launched (planned) — Chinese coordinated solar fleet
   │
2025  ─── Solar Cycle 25 maximum (CHASE designed to cover ascending phase)
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어**
1. **Hα 분광학 / Hα spectroscopy**: 6562.8 Å 선의 형성, 채층-광구 기여 분리, Doppler 효과로 시선속도 추출
2. **태양 하부 대기 / Solar lower atmosphere**: 광구(photosphere) ~5800 K, 온도 최저층, 채층(chromosphere) ~10,000 K, 자기 다발(flux tube) 구조
3. **분광 영상 기법 / Spectroscopic imaging**: raster scanning vs. filtergram vs. Fabry-Perot, 슬릿 분광기 원리
4. **TMA 광학 / Three-Mirror Anastigmat optics**: 비축(off-axis) 3거울 무수차 광학, 우주 망원경 표준
5. **CMOS detector**: 양자효율, 다크 전류, 풀웰(full well), 양자화(ADC bit depth)
6. **분광 보정 / Spectroscopic calibration**: 다크(dark)/플랫필드(flat field)/슬릿 곡률(slit curvature)/슬릿 폭 균일성 보정
7. **태양 동기 궤도 / Sun-synchronous orbit**: ~517 km, ~95분 주기, 태양 정점 시각 일정 유지
8. **태양 활동 현상 / Solar activity phenomena**: 필라멘트(filament), 플레어(flare), CME, Ellerman bomb, 백색광 플레어(white-light flare)

**English**
1. **Hα spectroscopy**: formation of the 6562.8 Å line, separation of chromospheric vs. photospheric contributions, Doppler shifts for line-of-sight velocity
2. **Solar lower atmosphere**: photosphere ~5800 K, temperature minimum, chromosphere ~10,000 K, flux-tube magnetic structure
3. **Spectroscopic imaging techniques**: raster scanning vs. filtergram vs. Fabry-Perot, slit spectrograph principles
4. **TMA optics**: off-axis Three-Mirror Anastigmat — standard for compact space telescopes
5. **CMOS detectors**: quantum efficiency, dark current, full-well capacity, ADC quantization bits
6. **Spectroscopic calibration**: dark, flat-field, slit-curvature, slit-width-uniformity corrections
7. **Sun-synchronous orbit**: ~517 km altitude, ~95 min period, fixed local solar time at ascending node
8. **Solar activity phenomena**: filaments, flares, CMEs, Ellerman bombs, white-light flares

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **CHASE / Xihe (羲和)** | Chinese Hα Solar Explorer; 중국 최초 태양 우주 미션, 2021/10/14 발사. 위성 무게 508 kg, 크기 1210×1210×1350 mm / China's first solar space mission, launched 2021/10/14. Satellite 508 kg, 1210×1210×1350 mm |
| **HIS** | Hα Imaging Spectrograph — CHASE의 과학 탑재체. 무게 54.9 kg, 크기 635×556×582 mm / Scientific payload of CHASE; 54.9 kg, 635×556×582 mm |
| **RSM** | Raster Scanning Mode — 슬릿을 태양 디스크에 가로질러 스캔하여 풀-Sun 분광 영상 획득 / Mode where slit scans across solar disk to acquire full-Sun spectra; 30–60 s cadence |
| **CIM** | Continuum Imaging Mode — 6689 Å 연속체 영상으로 위성 안정성 검증 및 광구 영상 제공 / Continuum imaging at 6689 Å for platform stability verification and photospheric imaging |
| **TMA assembly** | Three-Mirror Anastigmat — 비축 3거울 무수차 광학으로 넓은 시야와 긴 초점거리 동시 달성 / Off-axis three-mirror anastigmat — wide FOV with long focal length |
| **Slit-curvature correction** | 분광 영상에서 슬릿이 휘어 보이는 현상 보정. 비축 거울과 회절격자가 원인 / Correction for apparent slit curvature in spectra; caused by off-axis mirrors and grating |
| **Flat-field correction** | 비네팅, 슬릿/검출기 결함, 슬릿 폭 불균일에 의한 강도 패턴 제거. CHASE는 슬릿 따라 디스크 중심을 이동시키며 획득 / Removes vignetting, slit/detector artifacts, slit-width nonuniformity intensity patterns. CHASE moves disk center along slit to obtain it |
| **Sit-stare spectroscopy** | RSM의 sub-mode 중 하나; 슬릿을 고정하고 < 10 ms 노출로 빠른 분광 시계열 획득 / RSM sub-mode; fixed slit with <10 ms exposure for fast spectral time series |
| **Spectral FWHM (0.072 Å)** | HIS 기기 분광 분해능; 0.024 Å는 화소(pixel) 단위 spacing / HIS instrument spectral resolution; 0.024 Å is per-pixel sampling |
| **Pixel spatial resolution (0.52″)** | HIS 화소 공간 분해능 — Hinode SOT(0.16″)보다 거칠지만 풀-Sun을 1분 안에 스캔 가능 / HIS per-pixel spatial sampling — coarser than Hinode SOT (0.16″) but covers full Sun in 1 min |
| **Sun-as-a-star** | 태양을 점원으로 적분하여 항성 활동 모방, 외계 항성 비교 연구 / Disk-integrated solar spectra mimicking stellar observations for stellar comparison studies |
| **JPEG2000 / Rice compression** | Level 0 raw 데이터(JPEG2000 6:1)와 Level 1 과학 데이터(Rice) 손실/무손실 압축 / Level 0 raw uses lossy JPEG2000 (6:1); Level 1 science data uses lossless Rice |

---

## 5. 수식 미리보기 / Equations Preview

CHASE 논문은 본질적으로 **기기 소개 + 데이터 처리 흐름** 논문이므로 새로운 이론 수식은 거의 없다. 그러나 기기 매개변수와 보정에 사용되는 핵심 관계식들을 정리한다.

CHASE paper is essentially an **instrument description + data pipeline** paper, so it introduces few new theoretical equations. Below are the key relations governing the instrument and calibration.

### (1) Spectral resolution from grating / 격자에서의 분광 분해능

$$
\Delta\lambda \approx \frac{\lambda \cdot d}{f \cdot N \cdot m}
$$

where $d$ = slit width (9 μm), $f$ = focal length (1820 mm), $N$ = grating groove density (1900 lp/mm), $m$ = diffraction order. CHASE delivers FWHM = 0.072 Å with 0.024 Å pixel sampling (Nyquist-oversampled by ~3×).

여기서 $d$ = 슬릿 폭, $f$ = 초점거리, $N$ = 격자 선밀도, $m$ = 회절 차수. CHASE는 0.072 Å FWHM를 0.024 Å 화소로 약 3배 오버샘플링하여 측정.

### (2) Pixel spatial sampling / 화소 공간 샘플링

$$
\theta_{\text{pix}} = \frac{p_{\text{pix}}}{f} \cdot \frac{180 \cdot 3600}{\pi} \quad [\text{arcsec}]
$$

For HIS RSM: $p_{\text{pix}} = 4.6\,\mu\text{m}$, $f = 1820$ mm → $\theta_{\text{pix}} \approx 0.52''$. This matches the value quoted in Table 2.

### (3) Full-Sun raster time / 풀-Sun 래스터 시간

$$
T_{\text{full-Sun}} = \frac{D_{\odot}}{v_{\text{scan}}}
$$

with apparent solar diameter $D_{\odot} \approx 32' = 1920''$ and scan speed $v_{\text{scan}} = 4.6 \pm 0.3$ mm/s in detector plane → ~46 s; designed for 1-min cadence with redundancy. Region-of-interest scanning achieves 30–60 s.

태양 겉보기 지름 $D_{\odot} \approx 32' = 1920''$, 스캔 속도 $v_{\text{scan}} = 4.6 \pm 0.3$ mm/s로 약 46초. 1분 주기로 설계.

### (4) Doppler velocity from line shift / 선 이동에서의 도플러 속도

$$
v_{\text{LOS}} = c \cdot \frac{\Delta\lambda}{\lambda_0}
$$

For Hα ($\lambda_0 = 6562.8$ Å) with 0.024 Å pixel resolution → ~1.1 km/s velocity sensitivity per pixel; sub-pixel centroid fitting can reach ~100 m/s.

Hα 선에서 0.024 Å 화소 → 화소당 ~1.1 km/s 속도 민감도; 부분화소 중심 추정으로 ~100 m/s 도달 가능.

### (5) Data rate and storage / 데이터율과 저장

$$
\text{Data rate} = \frac{N_{\text{spatial}} \cdot N_{\text{spectral}} \cdot \text{bits}}{\text{cadence}}
$$

For RSM: 4608 × 376 × 12 bit per frame, scanned across the disk → ~14.9 GB per uncompressed full-Sun raster. At 6:1 JPEG2000 → ~2.5 GB; daily ground capture ~1.2 Tb/day. Computing system: 6 PB storage, 102.4 Tflops.

---

## 6. 읽기 가이드 / Reading Guide

**한국어**
이 논문은 **기기 개요 논문(instrument paper)** 으로, 다음 4가지 흐름을 따라 읽으면 효율적이다:

1. **§1 Introduction**: Hα 선의 진단 가치와 우주 관측 동기를 파악 (왜 CHASE가 필요한가?)
2. **§2 Scientific objectives**: 4가지 주요 과학 목표 — (2.1) 필라멘트 형성/동역학/카이랄리티, (2.2) 광구·채층 활동, (2.3) 태양-항성 비교 활동, (2.4) Sun-as-a-star 연구
3. **§3 Instrument overview + Tables 1-2**: HIS의 광학 설계, RSM/CIM 두 모드 매개변수를 표로 정리. **Table 1과 Table 2를 반드시 정독.**
4. **§4 Data processing**: Level 0 → Level 1 보정 단계 (다크 → 슬릿 곡률 → 플랫필드 → 좌표/파장/강도 보정). 각 단계가 왜 필요한지 이해.
5. **§5 First results**: 첫 궤도 관측 결과 (필라멘트, 플레어 등의 사례). 정량적 수치 (속도, 강도)에 주목.

**자주 등장하는 reference 번호** — \[15\]는 이전 CHASE 리뷰(Li 2019), \[39\]는 동반 논문 Liu et al.(HIS 상세 설계), \[41\]은 Qiu et al.(보정 절차 상세). 본 논문은 의도적으로 이들 동반 논문에 디테일을 위임하므로, 깊이 들어가려면 그쪽도 참고.

**주의할 점**:
- Table 1 (instrument)과 Table 2 (observational modes)의 차이를 헷갈리지 말 것 — 전자는 **하드웨어 사양**, 후자는 **관측 모드 파라미터**.
- 분광 분해능은 두 숫자가 등장: **FWHM 0.072 Å** (실제 라인 폭)과 **0.024 Å pixel resolution** (샘플링 간격). FWHM이 진정한 분해능.
- "spatial resolution"은 HIS pixel sampling인 0.52″를 말하지만, 실제 광학적 회절 한계는 별개로 계산해야 함.

**English**
This is an **instrument overview paper**. Read it in this order:
1. §1 Introduction: why Hα and why a space mission?
2. §2 Scientific objectives: four sub-goals (filaments / lower-atmosphere dynamics / solar–stellar comparison / Sun-as-a-star)
3. §3 Instrument overview + **Tables 1–2** (read carefully): optical design, two-mode parameters
4. §4 Data processing: Level 0 → Level 1 (dark → slit-curvature → flat → coordinate/wavelength/intensity)
5. §5 First results: on-orbit examples — focus on quantitative numbers

**Distinguish the two resolution numbers**: FWHM 0.072 Å is the true resolution; 0.024 Å is per-pixel sampling.
**References to companion papers**: \[15\] earlier CHASE review, \[39\] Liu et al. detailed HIS design, \[41\] Qiu et al. detailed calibration.

---

## 7. 현대적 의의 / Modern Significance

**한국어**
1. **풀-Sun Hα 분광의 우주 시대 개막**: SDO/AIA가 EUV 영역에서 한 일을 Hα에서 수행. 태양 활동의 광구–채층 결합을 처음으로 우주에서 풀-Sun으로 시계열 추적 가능.
2. **태양–항성 비교(stellar physics) 가능**: Sun-as-a-star Hα 프로파일이 외계 항성 슈퍼플레어, 항성 CME 연구의 직접 비교 대상이 됨. Kepler/TESS 시대에 매우 중요.
3. **다중 미션 연계**: SDO (HMI/AIA), IRIS (UV), Solar Orbiter (in-situ + remote), Hinode와 협업 관측. 광구 자기장 + EUV 코로나 + Hα 채층의 3차원 진단 체인 완성.
4. **솔라 사이클 25 상승기 커버**: 2021–2024 (3년 미션) — 사이클 25 최대기에 진입하는 시점이라 플레어/CME 풍부. 데이터 가치 극대화.
5. **중국 우주 태양물리의 출발점**: ASO-S (2022)와 함께 중국 태양 미션 군의 핵심 자산. 향후 미션(SUNDIAL 등)의 기술적 기반.
6. **새로운 데이터 차원 = 새로운 과학**: 매 1분마다 풀-Sun에서 모든 화소가 0.072 Å 분광 프로파일을 가진다 — 이전엔 존재하지 않던 데이터 큐브. 머신러닝/패턴 인식 기반 새로운 분석 기법의 출발점.

**English**
1. **Opens the space era of full-Sun Hα spectroscopy**: what SDO/AIA did for EUV, CHASE does for Hα — first-ever space-based time-resolved tracking of photosphere–chromosphere coupling across the full solar disk.
2. **Enables solar–stellar comparison**: Sun-as-a-star Hα profiles directly comparable to exoplanet-host stars studied by Kepler/TESS — superflare, stellar CME research.
3. **Multi-mission synergy**: SDO (HMI/AIA), IRIS (UV), Solar Orbiter, Hinode coordinated observation enables 3D photosphere–chromosphere–corona diagnostic chain.
4. **Covers Solar Cycle 25 rise (2021–2024)**: 3-year mission lifetime hits cycle 25 maximum — flares and CMEs abundant, maximizing scientific yield.
5. **Foundation for Chinese space solar physics**: alongside ASO-S (2022), CHASE anchors China's solar fleet and informs future missions.
6. **A new data dimension = new science**: every minute, every pixel of the full Sun yields a 0.072 Å spectral profile — a data cube that did not exist before, prime for ML/pattern-recognition analysis.

---

## Q&A

### Q1. CHASE/HIS가 생산하는 영상의 dimension, depth, cadence는? / Image dimensions, bit depth, cadence?

논문 Table 1 (instrument)과 Table 2 (observational modes)를 종합하면 다음과 같다.

#### RSM (Raster Scanning Mode) — Hα 분광 영상

| 항목 / Item | 값 / Value | 비고 / Note |
|---|---|---|
| Detector array | **4608 × 376** | spatial(4608) × spectral(376) |
| Pixel size | 4.6 μm | CMOS |
| Bit depth (ADC) | **12 bit** | full-well 14.5 k e⁻ |
| Spatial sampling | **0.52″ / pixel** | |
| Spectral resolution | **FWHM 0.072 Å**, 0.024 Å/pixel | Hα 부근 / near Hα |
| Passband | Hα 6559.7–6565.9 Å, Fe I 6567.8–6570.6 Å | 두 윈도우 동시 / two windows simultaneously |
| Exposure / step | **< 10 ms** | |
| Full-Sun cadence | **60 s** (실제 ~46 s + 여유) | 1-min cadence by design |
| ROI / Sit-stare cadence | 30–60 s (ROI), < 10 ms (sit-stare) | RSM sub-modes |

- **데이터 큐브 / Data cube**: 풀-Sun 한 라스터 ≈ 4608 × N_scan × 376 × 12 bit ≈ **14.9 GB uncompressed**

#### CIM (Continuum Imaging Mode) — 광구 연속체 영상

| 항목 / Item | 값 / Value |
|---|---|
| Detector array | **5120 × 5120** |
| Pixel size | 4.5 μm |
| Bit depth (ADC) | **10 bit** (full-well 12 k e⁻) |
| Spatial sampling | 0.52″ / pixel |
| Center λ | **6689 Å**, FWHM 13.4 Å (순수 연속체 / pure continuum) |
| Exposure | < 5 ms |
| Frame rate | **1 fps** (cadence 1 s) |

#### 광학·텔레메트리 / Optics & Telemetry
- HIS 광학 FOV: **40′ × 40′**, 유효 구경 180 mm, f = 1820 mm, F/10.1
- 송신율 / Transmission: **300 Mbps**, 지상 수신 ~1.2 Tb/day (압축 후)
- Level 0: JPEG2000 6:1 손실 압축 / Level 1: Rice 무손실 압축
- 처리 시스템 / SSDC-NJU: 6 PB 저장, 102.4 Tflops

#### TL;DR
- **RSM**: 4608 × 376 × 12-bit, 0.52″/pixel · 0.072 Å FWHM, **풀-Sun 1분 cadence**
- **CIM**: 5120² × 10-bit, 0.52″/pixel, **1 fps 풀-Sun 광구 영상**

---

### Q2. 실제 현재 릴리스되고 있는 데이터 스펙은? / Currently released data specs?

조사일 / As of: 2026-04-27. 출처: SSDC-NJU 포털, Qiu et al. (2022) 보정 동반 논문, CNSA Service Rules.

#### 데이터 릴리스 현황 / Release Status

| 항목 / Item | 내용 / Details |
|---|---|
| Data host | **SSDC-NJU** — `https://ssdc.nju.edu.cn` |
| CHASE portal | `https://ssdc.nju.edu.cn/NdchaseSatellite` ("CHASE卫星" / "羲和号") |
| Public level | **Level 1 FITS** (community release) |
| Usage policy | CNSA "Service Rules for the CHASE Satellite Data" 공식 문서. 등록 후 다운로드 / registration required |
| Operation status | 지속 운영 중 (Solar Cycle 25 상승기·최대기 커버) / actively operating, covering Cycle 25 rise + maximum |
| Read routines | **IDL + Python** routines 제공 by CHASE team / provided |

#### Level 1 정의 / Level 1 Definition

Level 1 = 보정 완료 과학 데이터. Qiu et al. (2022) [arXiv:2205.06075](https://arxiv.org/abs/2205.06075)에 보정 절차 정의:

1. **Dark correction** — digital offset, read noise, dark current
2. **Slit-curvature correction** — off-axis mirror + grating 보정
3. **Flat-field correction** — vignetting, slit-width nonuniformity, detector artifacts
4. **Wavelength calibration** — 파장 가변 레이저 기반 / via tunable laser
5. **Intensity calibration** — flux-conserving normalization
6. **Coordinate transformation** — detector frame → helioprojective

Higher-level products (Doppler, LOS velocity maps)는 사용자 후처리. CHASE 팀이 IDL/Python read routine 제공.

#### 데이터 차원·포맷 / Dimensions & Format

- 형식 / Format: **FITS multi-extension HDU**
- RSM 풀-Sun raster (Level 0, uncompressed): ~14.9 GB
- 실제 Level 1 FITS shape, header keywords, 파일 크기는 SSDC-NJU 데이터 매뉴얼(중국어 우선)에서 확인 필요. 영문 명세서가 공개 페이지에 적게 노출됨.
- Recommended: 작은 raster 하나를 받은 뒤 `astropy.io.fits.info()`로 직접 구조 확인

#### 미확인 / Unverified

- Level 1.5 / Level 2 등 상위 제품의 공식 릴리스 여부 — 검색 결과 명시 자료 없음. 사용자 후처리가 표준 워크플로우.
- 데이터 latency (관측 → 공개) 및 검색 UI 세부 — 포털 매뉴얼 참조 필요.
- 2024–2025년 추가 campaign / coordinated observation 데이터셋 — 별도 announcement 채널 확인 필요.

#### 출처 / Sources
- [Solar Science Data Center of NJU](https://ssdc.nju.edu.cn/)
- [CHASE 卫星 / Xihe satellite portal](https://ssdc.nju.edu.cn/NdchaseSatellite)
- [Service Rules for the CHASE Satellite Data — CNSA](https://www.cnsa.gov.cn/english/n6465645/n6465648/c10373923/content.html)
- [Qiu et al. (2022), Calibration procedures for the CHASE/HIS science data, arXiv:2205.06075](https://arxiv.org/abs/2205.06075)
- [Li et al. (2022), CHASE mission overview, arXiv:2205.05962](https://arxiv.org/abs/2205.05962)
- [Chinese Hα Solar Explorer — Wikipedia](https://en.wikipedia.org/wiki/Chinese_H-alpha_Solar_Explorer)

