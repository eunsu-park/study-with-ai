---
title: "Pre-Reading Briefing: The Solar Orbiter Heliospheric Imager (SoloHI)"
paper_id: "58_howard_2020"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# The Solar Orbiter Heliospheric Imager (SoloHI): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Howard, R. A. et al., "The Solar Orbiter Heliospheric Imager (SoloHI)", A&A 642, A13 (2020). DOI: 10.1051/0004-6361/201935202
**Author(s)**: R. A. Howard, A. Vourlidas, R. C. Colaninno, C. M. Korendyke, S. P. Plunkett, et al. (NRL-led consortium)
**Year**: 2020

---

## 1. 핵심 기여 / Core Contribution

SoloHI는 ESA/NASA Solar Orbiter 임무에 탑재된 단일 광시야(40°×40°) 백색광 망원경으로, Sun 중심으로부터 5°–45° elongation 영역의 외부 corona와 inner heliosphere를 Thomson-scattered 광으로 영상화한다. STEREO/SECCHI HI-1을 계승하면서도 (1) 단일 망원경 + 더 큰 시야, (2) custom CMOS Active Pixel Sensor(APS) 4-die 모자이크(3968×3968) 채택, (3) ecliptic 밖(>30° inclination) 시점 + 0.28 AU 근일점이라는 두 축에서 처음으로 heliosphere를 3차원 관측한다. 이를 위해 4단(F1–F4 + I0) forward baffle, 9개 interior baffle, AE1/AE2 light-trap baffle로 구성된 다단 stray-light 차폐 시스템이 핵심이다.

SoloHI is a single, wide-field (40°×40°) white-light telescope on the ESA/NASA Solar Orbiter that images Thomson-scattered light from the outer corona and inner heliosphere over elongations of 5°–45° from Sun centre. Building on STEREO/SECCHI HI-1, the paper presents three innovations: (1) a single telescope with twice the inner-FOV of HI-1, (2) a custom 4-die CMOS Active Pixel Sensor mosaic (3968×3968 pixels), and (3) the first heliospheric imager flown on an out-of-ecliptic, close-perihelion (0.28 AU) trajectory, enabling 3D heliospheric reconstruction. The instrument's stray-light suppression — F1–F4 + I0 forward baffles, nine interior baffles, and AE1/AE2 aperture-trap baffles — is the central engineering achievement that allows recording faint coronal signals 10⁻¹³ B☉ next to the Sun.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1995년 SOHO/LASCO가 정지 시점에서 corona를 안정적으로 관측하기 시작했고, 2006년 STEREO/SECCHI는 두 위성 시점으로 corona–heliosphere 연결을 처음 영상화했다. 그러나 두 임무 모두 (i) ecliptic 평면 안에 머물렀고, (ii) 1 AU 근방에서만 관측했으므로 inner heliosphere의 3D 구조와 0.5 AU 이내 진화는 간접적으로만 추정되었다. Solar Orbiter(2020 발사)와 Parker Solar Probe(2018 발사)는 이 한계를 동시에 극복하기 위한 차세대 encounter 임무이고, SoloHI는 그중 유일한 광시야 heliospheric imager다.

SOHO/LASCO (1995) provided the first stable, single-vantage white-light coronagraphy, and STEREO/SECCHI (2006) added a second viewpoint that permitted CME triangulation. Both missions were limited to (i) the ecliptic plane and (ii) ~1 AU heliocentric distance, so the 3D structure of the inner heliosphere and CME evolution inside 0.5 AU were inferred only indirectly. Solar Orbiter (launched Feb 2020) and Parker Solar Probe (launched 2018) are the two complementary encounter missions designed to break these limits, and SoloHI is the sole wide-field heliospheric imager among Solar Orbiter's ten instruments.

### 타임라인 / Timeline

```
1974 ──── Helios (ZL photometer, in-situ + photometric inner heliosphere)
1980s ─── Solwind, SMM C/P (early space coronagraphs)
1995 ──── SOHO/LASCO C1/C2/C3 (coronagraph, 30 R☉)
2006 ──── STEREO/SECCHI HI-1, HI-2 (first heliospheric imagers, ecliptic, 1 AU)
2018 ──── Parker Solar Probe / WISPR (closer perihelion, ecliptic)
2020 ──── Solar Orbiter / SoloHI ◄── this paper (out-of-ecliptic, 0.28 AU)
2020+ ─── joint SoloHI+WISPR+LASCO multi-vantage 3D reconstruction
```

---

## 3. 필요한 배경 지식 / Prerequisites

수학/물리 / Math & Physics
- Thomson scattering 단면적과 K-corona 강도 식 (B/B☉ 단위, B☉ ≈ mean solar disk brightness)
- F-corona (zodiacal dust scattering) vs K-corona (electron scattering) 분리 개념
- Fraunhofer 회절 이론과 baffle edge로부터의 회절 광 감쇠
- Cosine⁴(또는 cos³θ) vignetting 법칙
- Optical 시스템의 plate scale, F/#, focal length 관계

기기/관측 / Instrumentation & Observation
- CCD vs CMOS APS 차이 (read noise, radiation tolerance, pinned photodiode 5T pixel)
- Snap shutter vs rolling shutter 동작
- Correlated double sampling (CDS) 신호 처리
- 다단 baffle stray-light 억제 원리 (forward, interior, light-trap)
- Bidirectional reflectance distribution function (BRDF)

이전 논문 / Prior Papers
- Eyles et al. 2009 (STEREO HI 계기 설명)
- Howard et al. 2008 (SECCHI 슈트 개요)
- Vourlidas et al. 2016 (PSP/WISPR — SoloHI의 sister 기기)
- Müller et al. 2013, 2020 (Solar Orbiter 임무 개요)
- Vourlidas & Howard 2006 (Thomson surface 이론)

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Elongation (ε) | 우주선에서 본 Sun 중심과 시선 사이 각거리. SoloHI: 5.4°–44.9° / Angular distance from Sun centre as seen from spacecraft; SoloHI inner cutoff at 5.4°, outer at 44.9° |
| K-corona | 자유전자가 광구 빛을 Thomson scatter한 성분. 편광되어 있고 1/r² 약하게 감소 / Free-electron Thomson-scattered light, polarised |
| F-corona / Zodiacal Light (ZL) | 행성간 먼지가 산란한 광구 빛. 5 R☉ 이상에서 신호의 dominant 성분 / Dust-scattered photospheric light; dominant signal above 5 R☉ |
| AUeq (1 AU equivalent) | 0.28 AU에서 측정한 각/공간 분해능을 1 AU 거리로 환산한 값 (HI-1 등과 비교용) / FOV/resolution rescaled to 1 AU for cross-instrument comparison |
| APS (Active Pixel Sensor) | 픽셀당 트랜지스터를 갖는 CMOS 이미저. CCD 대비 고온/방사선 환경에서 안정적, 단점은 read noise·QE·column pattern noise / CMOS imager with per-pixel transistors; better radiation tolerance |
| 5T pixel | 5-transistor pinned-photodiode pixel: reset, gain control, transfer gate, source follower, row select / Pixel architecture of SoloHI APS |
| Pinwheel mosaic | 4개의 die를 회전 배치하여 read-out side가 바깥(= solar disk 반대쪽)에 오도록 한 구성 / Four buttable die rotated so read-out edges face outward |
| Forward baffle (F1–F4, I0) | Sun 직접광·반사광·회절광이 망원경 입사구에 도달하지 못하도록 막는 5단 baffle 시스템 / Multi-stage baffles blocking direct/reflected/diffracted sunlight |
| Light trap baffle (AE1, AE2) | F4와 interior baffle을 통과한 잔여 회절광·반사광을 흡수하는 2단 baffle / Aperture-plane baffles trapping residual stray light |
| Field of regard (FOR) | 광학적 FOV는 아니지만 spacecraft 부품이 들어와서는 안 되는 더 큰 각도 영역 / Angular zone larger than FOV that must remain free of spacecraft intrusions |
| Thomson surface | 우주선–Sun 선분을 지름으로 하는 구면. 이 표면 근처 전자가 가장 많이 산란 / Sphere with the spacecraft–Sun line as diameter; locus of strongest scattering |
| AU (perihelion 0.28 AU) | Solar Orbiter 최소 근일점 거리 (지구 거리의 28%) / Solar Orbiter minimum perihelion |
| Boresight | 광축 방향. SoloHI는 Sun 중심에서 25° 동쪽 (anti-ram) / Optical axis; SoloHI points 25° east of Sun centre |

---

## 5. 수식 미리보기 / Equations Preview

(1) Streamer blob radial speed model (Fig. 1):
$$V^2(r) = V_0^2\left[1 - e^{-(R-R_1)/R_0}\right]$$
- $V_0=298.3$ km/s, $r_0=8.1\,R_\odot$, $r_1=2.8\,R_\odot$ (LASCO blob fit) — describes the asymptotic acceleration profile of slow-wind blobs.

(2) Scene brightness (Thomson surface integral):
$$B(\varepsilon) = \int_{\rm LOS} G(\theta)\, n_e(s)\, ds$$
- where $G(\theta)$ is the Thomson scattering geometry factor and the integral is along the line of sight at elongation $\varepsilon$. SoloHI inverts this for $n_e$.

(3) Diffracted irradiance after N baffles:
$$I_N \approx I_0 \prod_{k=1}^{N} \alpha_k, \quad \alpha_k \sim 10^{-3}$$
- Each baffle attenuates by ~3 orders of magnitude → 5 baffles required to reach $10^{-13}\,B_\odot$ at outer FOV.

(4) Vignetting law:
$$T(\theta) = \cos^3\theta \times V_{\rm baffle}(\theta)$$
- Natural cos³ falloff plus forward-baffle vignetting from 5.4° to 8.8°.

(5) Photometric S/N requirement:
$$\frac{S}{N} \geq 5 \;(\text{simple known target}); \quad \geq 30 \;(\text{complex unknown})$$
- Drives integration-time / image-summing strategy.

---

## 6. 읽기 가이드 / Reading Guide

1차 읽기(맥락) / First pass — Read §1 (Introduction), §2.5 (SoloHI unique science), §3 (Instrument overview), §6 (Summary). Goal: catch the why.
- §1: Solar Orbiter 전체 임무 목표 + SoloHI 위치
- §2.5: out-of-ecliptic 시점 + 0.28 AU 근일점이 만드는 새로운 과학
- §3: 핵심 instrument parameter Table 1, design philosophy

2차 읽기(설계) / Second pass — Read §4.1–§4.3, §4.6. Goal: how the photons are caught.
- §4.1–§4.2: optical design (5-element lens, 500–850 nm, 25° boresight)
- §4.3: stray-light rejection (forward/interior/light-trap/peripheral baffles, Figs. 10–12)
- §4.6: APS detector (4-die pinwheel, 5T pixel, 32% QE, 5.8 e⁻ read noise)

3차 읽기(운영) / Third pass — Read §2.1–§2.4, §5. Goal: what science programs.
- §2.1–§2.4: four science questions (solar wind, transients, SEPs, 3D heliosphere)
- §5: observing program, calibration, data products (Levels 0–3)

읽으면서 점검할 것 / While reading, check:
- Why is the boresight 25° from Sun centre and not on Sun? (anti-ram + heat-shield geometry)
- How does the FOV in R☉ change as Solar Orbiter moves from 0.88 AU to 0.28 AU?
- Why use a CMOS APS rather than a CCD (radiation, mass, complexity)?
- Why use 4 die in pinwheel rather than a single monolithic 4k×4k sensor?

---

## 7. 현대적 의의 / Modern Significance

SoloHI는 LASCO–SECCHI가 보여준 단일/이중 시점 coronagraphy 시대를 종결하고, multi-vantage + close-perihelion + out-of-ecliptic이라는 세 자유도를 모두 갖춘 첫 imager다. WISPR(PSP)·Metis(Solar Orbiter)·LASCO와 결합하면 같은 CME를 4–5개 시점에서 동시에 촬영할 수 있어, CME mass·속도·3D 형상에 대한 inversion 불확실성이 크게 줄어든다. 또한 30°+ 황도면 경사에서 본 F-corona는 zodiacal dust 분포의 3D 단층촬영을 가능하게 하며, 혜성·금성 dust ring·SEP 가속 충격파의 직접 영상까지 SoloHI의 응용 영역에 들어온다. 본 논문은 이러한 차세대 multi-vantage heliospheric imaging 시대의 reference instrument paper다.

SoloHI closes the era of single- or dual-vantage coronagraphy (LASCO, SECCHI) and opens the era of multi-vantage + close-perihelion + out-of-ecliptic imaging — the first instrument with all three degrees of freedom. Combined with WISPR (PSP), Metis, and LASCO, the same CME can be imaged from 4–5 simultaneous viewpoints, dramatically reducing inversion uncertainties in CME mass, speed, and 3D geometry. Imaging the F-corona from > 30° ecliptic inclinations enables 3D tomography of zodiacal dust, while comets, the Venus dust ring, and SEP-driving shocks all enter SoloHI's scientific reach. This paper is the reference instrument description for that next-generation multi-vantage heliospheric imaging era.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
