---
title: "Pre-Reading Briefing: Operational Uses of the GOES Energetic Particle Detectors"
paper_id: "46_onsager_1996"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# Operational Uses of the GOES Energetic Particle Detectors: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Onsager, T. G., R. Grubb, J. Kunches, L. Matheson, D. Speich, R. Zwickl, and H. Sauer, "Operational uses of the GOES energetic particle detectors," *Proc. SPIE* **2812**, 281–290 (1996). DOI: 10.1117/12.254075
**Author(s)**: T. G. Onsager, R. Grubb, J. Kunches, L. Matheson, D. Speich, R. Zwickl, H. Sauer (NOAA Space Environment Center; Sauer also at CIRES, University of Colorado)
**Year**: 1996

---

## 1. 핵심 기여 / Core Contribution

이 논문은 GOES-8/9 우주환경 모니터(SEM)에 탑재된 에너지 입자 센서(EPS)와 고에너지 양성자/알파 입자 검출기(HEPAD)의 설계, 데이터 처리 절차, 그리고 운영적 사용 방법을 종합적으로 기술한 NOAA SEC의 기술 보고서입니다. 0.7 MeV에서 900 MeV에 이르는 양성자, 4–500+ MeV 알파 입자, 그리고 0.6 MeV 이상의 전자 플럭스를 정지궤도(geosynchronous orbit)에서 실시간으로 측정하여, 알림(alerts), 경보(warnings), 사후 분석, 그리고 장기 우주 기후 연구의 기준 데이터로 활용하는 운영 체계 전반을 다룹니다.

This paper is a comprehensive technical report from NOAA SEC describing the design, data processing pipeline, and operational uses of the Energetic Particle Sensor (EPS) and High Energy Proton and Alpha Detector (HEPAD) on the GOES-8/9 Space Environment Monitor. It documents how proton fluxes from 0.7 MeV to 900 MeV, alpha particles 4–500+ MeV, and >0.6 MeV electrons measured at geosynchronous orbit are converted in real time into the alerts, warnings, post-event analyses, and long-term space-climate datasets that anchor U.S. civilian and military space-weather operations.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 중반은 GOES 위성이 회전형(spinning)에서 3축 안정형(non-spinning, body-fixed)으로 전환되던 시기였습니다. GOES-8(1994년 발사)부터 EPS/HEPAD는 고정된 시야(서쪽, HEPAD는 지구 반대 방향)를 갖게 되었고, 이는 SEP(Solar Energetic Particle) 사건의 초기 비등방성(anisotropy) 측정에 새로운 운영적 도전을 가져왔습니다. 동시에 NOAA SEC는 위성 산업, 우주 비행사 안전(Space Shuttle, 곧 ISS), 통신/항법 사용자에 대한 실시간 우주환경 서비스를 확장하던 단계였습니다.

The mid-1990s marked the transition of GOES from spinning to three-axis-stabilized (body-fixed) spacecraft. Starting with GOES-8 (launched 1994) the EPS/HEPAD pointed in fixed directions (westward; HEPAD anti-Earthward), introducing a new operational challenge for capturing the highly anisotropic onset of solar energetic particle (SEP) events. Simultaneously, NOAA SEC was scaling up real-time space-environment services for satellite operators, astronaut safety (Space Shuttle, soon ISS), and communications/navigation users.

### 타임라인 / Timeline

```
1975 ─────── 1986 ─────── 1988 ─────── 1990 ─────── 1994 ─────── 1996 ─────── 2000s
  │            │            │            │            │            │            │
GOES-1       Stevens       Shea          Baker       GOES-8      Onsager     GOES-N/O/P
SEM era    sc charging    SEP profiles  LPF MeV-e    launch      this paper   improved EPS
begins                    review                     (3-axis)    (operations)
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **정지궤도 환경 / Geostationary environment**: L≈6.6 R_E에서 외부 복사대(outer radiation belt)와 자기권계면(magnetopause)이 가까이 있는 지점.
- **3가지 입자 모집단 / Three particle populations**: (1) 자기권 포획 입자(trapped), (2) 태양 기원 입자(SEP), (3) 은하 우주선(GCR).
- **Detector physics**: silicon surface barrier detector, Cherenkov PMT, coincidence/anti-coincidence logic, geometric factor.
- **Particle Flux Unit (pfu)**: 1 particle/(cm² s sr) — NOAA SEC가 알림 임계값을 표현하는 단위.
- **Solar wind coupling**: high-speed streams → MeV electron acceleration in outer belt (Paulikas & Blake 1979).
- **SEP physics 기초**: flare/CME 가속, 행성간 충격파 재가속, Parker spiral magnetic connectivity.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| EPS (Energetic Particle Sensor) | 망원경(telescope) + 3개의 돔(dome) 모듈로 구성된 0.7–900 MeV 양성자, 4–500 MeV 알파, >0.6 MeV 전자 측정 장비 / Telescope + three dome modules covering 0.7–900 MeV protons, 4–500 MeV alphas, >0.6 MeV electrons |
| HEPAD | 고에너지 양성자(330–700+ MeV)/알파(2560–3400+ MeV) 검출기. Si 망원경 + Cherenkov PMT / High-energy proton/alpha detector using Si telescope + Cherenkov PMT |
| Geometric factor (G, cm² sr) | 등방 플럭스를 카운트율로 변환하는 검출기 기하학적 인자 / Detector aperture factor converting isotropic flux to count rate |
| Channel response factor | 1차 에너지 영역에서의 평탄 스펙트럼 응답(cm² sr MeV) / Flat-spectrum response over the primary energy range |
| Secondary response | 1차 에너지 범위 외부의 입자가 차폐를 뚫고 만드는 부수적 카운트 / Counts from particles outside the primary energy/aperture penetrating the shielding |
| pfu (Particle Flux Unit) | 1 particle/(cm² s sr) — alerts/warnings 임계값 단위 / Standard unit for SEC thresholds |
| PCA (Polar Cap Absorption) | >10 MeV 양성자가 극지 D-영역을 이온화하여 HF 통신을 흡수시키는 현상 / Ionization of polar D-region by >10 MeV protons absorbing HF radio |
| Deep dielectric charging | >1 MeV 전자가 위성 내부 절연체에 침투해 ESD를 유발하는 현상 / Internal charging of dielectrics by >1 MeV electrons leading to ESD |
| Anisotropy onset | SEP 사건 초기에 자기력선 평행 방향으로 강하게 정렬된 플럭스 / Field-aligned beamed flux at SEP onset |
| Effective heliospheric potential | GCR 변조(modulation)를 단일 매개변수로 표현해 장기 누적 선량을 추정 / Single-parameter description of GCR modulation for long-term dose estimates |
| Ten-day low-pass filter | EPS 양성자 채널의 우주선 배경(cosmic-ray background)을 추정하는 10일 최소값 필터 / Ten-day minimum filter used to estimate cosmic-ray background for proton channels |

---

## 5. 수식 미리보기 / Equations Preview

**1. Flux from count rate (등방 가정 / isotropic):**
$$ J(E) \;=\; \frac{C}{G \cdot \Delta E} \quad \text{[particles/(cm}^2\,\text{s sr MeV)]} $$
여기서 $C$는 카운트율, $G \cdot \Delta E$는 채널 응답 인자(cm² sr MeV) / Here $C$ is count rate and $G\cdot\Delta E$ is the channel response factor.

**2. Background-corrected count rate:**
$$ C_\text{corr}(t) \;=\; C(t) \;-\; \min_{\tau\in[t-10\text{d},\,t]} C(\tau) $$
10일 최소값을 제거하여 GCR 배경을 빼는 저역 통과(low-pass) 필터 / Subtracts the rolling 10-day minimum as a GCR-background low-pass filter.

**3. Secondary response correction (power-law assumption):**
$$ C_\text{primary} \;=\; C_\text{total} \;-\; \sum_k G_k \int_{E_k^{lo}}^{E_k^{hi}} J_\text{secondary}(E)\,dE $$
인접 채널 간 멱함수 스펙트럼을 가정해 부수 카운트를 추정·제거 / Power-law spectrum between adjacent channels estimates secondary counts to subtract.

**4. NOAA SEC alert thresholds (운영 / operations):**
$$ J_{>10\text{ MeV}} > 10\text{ pfu},\quad J_{>100\text{ MeV}} > 1\text{ pfu},\quad J_{>2\text{ MeV electrons}} > 10^{3}\text{ pfu} $$

**5. Effective electron geometric factor (스펙트럼 종속 / spectrum-dependent):**
$$ G_\text{eff} \;=\; G_\text{eff}\!\left(\gamma\right),\quad \gamma = -\frac{d\ln J}{d\ln E} $$
인접 채널로부터 스펙트럼 기울기 $\gamma$를 추정해 광대역 전자 채널의 유효 G를 계산 / Spectral slope from neighboring channels yields effective G for wide-band electron channels.

---

## 6. 읽기 가이드 / Reading Guide

- **§1 Instrument Description**: Figure 1–3과 Table 1을 함께 보세요. EPS telescope vs. dome vs. HEPAD의 에너지 범위/시야/검출 원리를 표로 정리하면 좋습니다.
  Read alongside Figures 1–3 and Table 1; tabulate EPS telescope vs. dome vs. HEPAD by energy range, FOV, and detection principle.
- **§2 Data Processing**: 두 가지 핵심 보정(10일 저역 필터 배경 제거, 부수 응답 차감)을 단계별로 추적하세요. P7부터 시작해 아래로 내려가는 반복적 보정 순서가 중요합니다.
  Trace the two key corrections (10-day low-pass background, secondary-response subtraction). The iterative order from P7 downward matters.
- **§3 Particle Environment**: Figure 4의 7 패널을 실제로 보면서 (a) 일주변동(trapped electrons), (b) 10/20일 SEP, (c) 배경 레벨을 식별하세요. §3.1과 §3.2의 운영적 시사점에 주목.
  Inspect Figure 4's seven panels for diurnal variation, the 20 Oct 1995 SEP event, and quiet background. Note operational implications in §3.1, §3.2.
- **§4 NOAA SEC Operations**: Table 2의 알림/경보 임계값이 어떻게 정해졌는지 (PCA, 사용자 폴링, 역사적 관행) 이유를 메모하세요.
  Note rationale for thresholds in Table 2 (PCA, user polling, historical practice).
- **§5 Summary**: 정지궤도 모니터링이 단일 위성 운영을 넘어 ionosphere/대기/우주인 보호로 확장되는 점에 주목.
  Note how geosynchronous monitoring extends beyond single-spacecraft operations to ionosphere, atmosphere, and human spaceflight.

---

## 7. 현대적 의의 / Modern Significance

이 1996년 SPIE 논문은 오늘날 SWPC(Space Weather Prediction Center, 구 SEC)가 발행하는 S-스케일(SEP scale, S1–S5)과 R/G 스케일의 직접적 운영 토대를 형성합니다. >10 MeV proton flux 10 pfu 임계값은 현재도 S1 ("Minor")의 정의로 유지되고, GOES-R 시리즈(GOES-16/17/18, 2016+)의 SGPS/EHIS 측정 또한 본 논문이 정립한 채널 구조와 보정 파이프라인을 계승합니다. 또한 deep dielectric charging 알림(>2 MeV electrons > 1000 pfu)은 현재 위성 운영자(Intelsat, SES 등)의 일상 운영 결정에 사용됩니다.

This 1996 SPIE paper is the operational foundation for today's SWPC space-weather scales: the >10 MeV proton flux 10 pfu threshold still defines S1 ("Minor"), and the GOES-R series (GOES-16/17/18, 2016+) SGPS/EHIS instruments inherit the channel structure and correction pipeline established here. The deep-dielectric-charging alert (>2 MeV electrons >1000 pfu) remains a routine operational input used by satellite operators (Intelsat, SES, etc.). The paper is widely cited whenever GOES energetic-particle data are used in modern statistical or machine-learning SEP forecasting work.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)

**Q1**: Why is HEPAD pointed anti-Earthward while EPS points westward? / 왜 HEPAD는 지구 반대 방향, EPS는 서쪽인가?
**A1**: EPS는 서쪽 시야로 자기권의 비등방성(노치/local-time effect)을 잘 보고, HEPAD는 GCR/고에너지 SEP가 거의 등방이라 지구 반대 방향이 지구 알베도(albedo) 입자 오염을 줄입니다. / EPS westward captures magnetospheric anisotropy at local time; HEPAD anti-Earthward minimizes Earth-albedo contamination for the near-isotropic GCR/high-energy SEP populations.

**Q2**: 왜 10일 저역 필터인가? / Why a 10-day low-pass for background?
**A2**: 양성자 SEP 사건의 전형적 지속시간은 수일이며, 10일은 평균적으로 두 사건 사이의 조용한 기간을 포착하기에 충분히 길고, GCR 변조 시간 척도(태양 자전 27일, Forbush 감소 며칠)보다 짧아 GCR 배경의 시간 종속성을 따라갈 수 있습니다. / SEP events last a few days; 10 days reliably brackets quiet intervals between events while remaining short enough to track GCR background variations (27-day solar rotation, Forbush decreases).

**Q3**: 비등방성이 omnidirectional flux에 어떻게 영향을 미치나? / How do anisotropies bias the omnidirectional flux estimate?
**A3**: GOES-8 이후 위성은 비회전이므로 검출기가 빔 방향을 향하면 과대평가, 빗나가면 과소평가됩니다. 따라서 SEP 초기 30–60분간 절대값보다는 시간 변화율과 인접 채널 간 일관성에 더 무게를 둡니다. / Body-fixed GOES-8+ overestimates if pointed into the beam, underestimates if off it; analysts weight temporal evolution and inter-channel consistency over absolute values during the first 30–60 minutes.
