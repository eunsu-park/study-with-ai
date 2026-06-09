---
title: "Pre-Reading Briefing: The Wide-Field Imager for Solar Probe Plus (WISPR)"
paper_id: "53_vourlidas_2016"
topic: Solar_Observation
date: 2026-04-25
type: briefing
---

# The Wide-Field Imager for Solar Probe Plus (WISPR): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Vourlidas, A., Howard, R.A., Plunkett, S.P., et al., "The Wide-Field Imager for Solar Probe Plus (WISPR)", Space Science Reviews, 204, 83–130 (2016). DOI: 10.1007/s11214-014-0114-y
**Author(s)**: Angelos Vourlidas, Russell A. Howard, Simon P. Plunkett, Clarence M. Korendyke, Arnaud F.R. Thernisien, Dennis Wang, Nathan Rich, Michael T. Carter, Damien H. Chua, Dennis G. Socker, Mark G. Linton, Jeff S. Morrill, Sean Lynch, Adam Thurn, Peter Van Duyne, Robert Hagood, Greg Clifford, Phares J. Grey, Marco Velli, Paulett C. Liewer, Jeffrey R. Hall, Eric M. DeJong, Zoran Mikic, Pierre Rochus, Emanuel Mazy, Volker Bothmer, Jens Rodmann
**Year**: 2016 (online 2015)

---

## 1. 핵심 기여 / Core Contribution

이 논문은 NASA Solar Probe Plus(SPP, 후일 Parker Solar Probe로 개명) 미션에 탑재되는 유일한 영상기기 WISPR(Wide-field Imager for Solar PRobe)의 과학 목표, 광학·기계·전자 설계, 그리고 운용 계획을 종합적으로 기술한 instrument paper이다. WISPR은 두 개의 망원경(Inner: 13.5°–53°, Outer: 50°–108°)으로 결합하여 95° radial × 58° transverse FOV를 가지며, 2K×2K APS CMOS 검출기로 일면(perihelion) 9.86 R⊙ 부근의 코로나·태양풍 구조를 in-situ 관측 직전·직후에 동시 촬영한다. 이로써 SPP가 직접 통과하는 플라즈마 구조와 대규모 코로나의 연결을 처음으로 관측적으로 잇는 'local' 헬리오스피어 영상기 역할을 수행한다.

This paper presents the comprehensive instrument description of WISPR, the sole imager aboard NASA's Solar Probe Plus (SPP, later renamed Parker Solar Probe) mission. WISPR comprises two wide-field telescopes (Inner: 13.5°–53°; Outer: 50°–108°) that together provide a 95° radial × 58° transverse field of view, imaging the corona and solar wind in white light using radiation-hardened 2K×2K APS CMOS detectors. By riding to within 9.86 R⊙ of Sun center at perihelion, WISPR is the first heliospheric imager to operate as a "local" imager — its line-of-sight Thomson-scattering geometry passes through and beyond the spacecraft, providing the crucial visual context that links the SPP in-situ FIELDS/SWEAP/ISIS measurements to the large-scale coronal structures from which the solar wind is born.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

태양풍(solar wind)이라는 개념은 Parker(1958)의 이론과 Snyder et al.(1963)의 직접 측정 이래 60년 동안 정립되어 왔지만, 우주선이 진입할 수 있었던 가장 가까운 거리는 Helios 미션의 0.3 AU(약 65 R⊙)에 머물러 있었다. 이 영역은 태양풍이 이미 가속·진화를 마친 후이기에, 가속 메커니즘·코로나 가열 문제는 원격 영상(LASCO, SECCHI/COR2, HI1/HI2)과 분광(UVCS) 관측에 의지해 추정만 가능했다. 1990–2010년대 STEREO/SECCHI HI 영상기는 0.1–1 AU의 헬리오스피어를 1 AU에서 처음 영상화했으나, in-situ 측정과 영상이 같은 위치에서 동시에 이뤄진 적이 없었다.

The concept of the solar wind has been firmly established since Parker (1958) and the in-situ confirmation by Snyder et al. (1963), yet for six decades the closest spacecraft approach was Helios at 0.3 AU (~65 R⊙). At those distances the wind has already accelerated and evolved, leaving the heating and acceleration mechanisms accessible only through remote sensing — coronagraphs (LASCO, SECCHI/COR2), heliospheric imagers (STEREO/HI1, HI2), and UV spectroscopy (UVCS). Although the STEREO/SECCHI Heliospheric Imagers (HI1, HI2) extended white-light imaging out to 1 AU, no mission had ever combined in-situ plasma sampling with simultaneous local imaging from the same vantage point inside the Alfvén surface.

### 타임라인 / Timeline

```
1958  Parker: solar wind theoretical prediction
1963  Snyder et al.: in-situ solar wind detection (Mariner 2)
1974-76 Helios 1/2: closest approach 0.29 AU; first F-corona/dust photometry
1995  SOHO/LASCO: continuous coronagraph imaging (C1/C2/C3)
1995  SOHO/UVCS: UV spectroscopy of outer corona
2003  Vourlidas & Howard: Thomson surface concept introduced
2006  Vourlidas & Howard: proper Thomson-scattering treatment for wide FOV
2006  STEREO launch; SECCHI: COR1/COR2/HI1/HI2 (1 AU heliospheric imaging)
2010  Viall et al.: 5-h periodic density structures in HI1
2014  Solar Orbiter design (SoloHI heritage)
2014  WISPR Preliminary Design Review (PDR)
2015  This paper (Vourlidas et al. 2016 print)
2018  Parker Solar Probe launch (renamed from SPP)
```

---

## 3. 필요한 배경 지식 / Prerequisites

- Thomson scattering by free electrons (Minnaert; Billings 1966; Howard & Tappin 2009): scattering geometry, Thomson surface concept, B/B☉ surface brightness units. / 자유전자에 의한 Thomson 산란 — 산란 기하·Thomson surface 개념·B/B☉ 단위.
- White-light coronagraphy concepts: K-corona vs F-corona, signal-to-noise, stray-light suppression. / 백색광 코로나그래프 — K/F-corona 분리, S/N, 산란광 억제.
- Heliospheric imager geometry: elongation angle ε, line-of-sight integral, conversion to heliocentric distance for given spacecraft location. / 헬리오스피어 영상기 기하 — 이격각 ε, LOS 적분, 우주선 위치에서의 거리 변환.
- Optical lens design basics: refractive multi-element lens, F-number, vignetting, lens BSDF. / 굴절 다요소 렌즈 설계 — F#, vignetting, BSDF.
- APS (Active Pixel Sensor) CMOS detectors: read noise, full-well, radiation tolerance, vs CCD trade-offs. / APS CMOS — 읽기 잡음·full-well·내방사선성·CCD 대비 장단점.
- Baffle theory and diffraction profiles for coronagraphs (Socker et al. 2000). / 코로나그래프 baffle 이론과 회절 프로파일.
- Parker Solar Probe (SPP) mission profile: 24 perihelia, Venus gravity assists, 9.86 R⊙ minimum perihelion, ram-side instrument mounting. / PSP 미션 — 24회 근일점, 9.86 R⊙ 최소 근일점, ram-side 장착.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| WISPR | Wide-field Imager for Solar PRobe Plus — SPP의 유일 영상기, 두 개 망원경(Inner/Outer) 결합. The sole imager on SPP, two-telescope system covering 95°×58° FOV. |
| Elongation (ε) | 우주선에서 본 태양 중심으로부터의 각거리. 시야 안의 픽셀이 보는 LOS 방향. Angular distance from Sun center as seen from spacecraft; defines pixel LOS. |
| Thomson surface | 주어진 LOS에서 자유전자의 산란 효율이 최대가 되는 위치 — Sun-observer 직각 평면(정확히는 구면). Locus of maximum Thomson-scattering efficiency: a sphere with the Sun-observer line as diameter. |
| B/B☉ | 표면밝기를 평균 태양 디스크 밝기로 정규화한 단위. Surface brightness normalized to mean solar disk brightness; standard coronagraph unit. |
| Stray light | 의도하지 않은 광로(태양 림 회절, 우주선 산란, 먼지 충돌 후 BSDF 변화)로 검출기에 도달한 빛. Unwanted light reaching the detector via diffraction at heat-shield edge, scatter from S/C structures or dust-damaged optics. |
| Baffle (F-baffle, A1) | 산란광·회절광을 차단하기 위한 불투명 차단판; WISPR은 forward(F1–F3), interior(I1–I7), peripheral(aperture hood)의 3중 baffle. Opaque vanes that intercept stray rays; WISPR uses three baffle systems (forward, interior, peripheral). |
| F-corona | 행성간 먼지(zodiacal)에 의해 전방 산란된 태양광; 1 AU에선 K-corona를 압도(>4 R⊙). Thomson + dust scattering composite; F dominates over K beyond ~4 R⊙ at 1 AU. |
| K-corona | 자유전자 Thomson 산란 성분; 코로나 구조(streamer, CME) 추적의 핵심. Free-electron Thomson scattering component carrying coronal-structure information. |
| APS CMOS | Active Pixel Sensor — 픽셀별 증폭기 내장으로 CCD 대비 방사선 내성 우수, 셔터 없는 글로벌 리셋. Per-pixel amplifier gives high radiation tolerance, shutterless readout. |
| UFOV | Unobstructed Field Of View — 우주선·열차폐가 침범하지 않아야 하는 직사광 보호 영역. The clear angular region around boresight that must remain free of direct sunlight. |
| Heat-shield shadow line | 열차폐 가장자리에서 정의되는 직사광 안전선; WISPR은 이 선 아래에 머물러야 함. Solar exclusion zone of 8.07° at 9.5 R⊙ perihelion (6.07° disk + 2° offpoint). |
| BSDF | Bidirectional Scattering Distribution Function — 광학 표면(렌즈, baffle)의 산란 특성. Optical surface scattering characterization for stray-light modeling. |
| 5/50/95 % LOS contour | 누적 Thomson 신호의 5 %·50 %·95 %가 들어오는 LOS 길이 등고선. Cumulative Thomson-scattering brightness contours along the line of sight. |

---

## 5. 수식 미리보기 / Equations Preview

(1) Thomson scattering brightness for a point on the LOS (Howard & Tappin 2009, 단순 형식):

$$
B(\chi) \;\propto\; N_e(r)\,\Bigl[(1-u)\,B_T(\chi)\;+\;u\,B_R(\chi)\Bigr]
$$

with χ the scattering angle (angle Sun–scatter–observer), $N_e$ the local electron density, and $B_T, B_R$ the tangential/radial geometric factors that peak at χ = 90° (Thomson surface). / χ는 산란각, $N_e$ 전자밀도; $B_T, B_R$은 χ=90°(Thomson surface)에서 최대.

(2) Elongation–distance relation (uniform spherical geometry):

$$
r \;=\; d\,\sin\varepsilon \quad\text{(Thomson-surface distance from Sun for elongation } \varepsilon \text{)}
$$

with $d$ the spacecraft–Sun distance. WISPR이 9.86 R⊙ 근일점에서 ε=13.5°이면 r ≈ 2.3 R⊙. / 우주선이 9.86 R⊙ 근일점에 있을 때 ε=13.5°→r≈2.3 R⊙.

(3) Vignetting envelope for wide-field lens:

$$
T(\theta) \;\propto\; \cos^4\theta
$$

with θ the angle from the optical axis (boresight) — natural cos⁴θ falloff in throughput for wide-angle refractive lenses. / 넓은 시야 굴절 렌즈에서 광학축으로부터 각도 θ에 따른 자연 vignetting.

(4) Plate scale conversion:

$$
\Delta\theta_{\text{pix}} \;=\; \frac{\text{pixel size}}{f}\;\;\;[\text{rad}]
$$

For WISPR Inner (f = 28 mm, 10 µm pixels): 357 µrad ≈ 73.6 arcsec ≈ 1.23 arcmin/pix. / WISPR Inner: 28 mm 초점거리·10 µm 픽셀 → 1.2 arcmin/pix.

(5) Stray-light requirement scaling (~ 1 / d² for direct, ~ 1 / d for diffracted):

$$
B_{\text{stray}}/B_\odot \;\le\; \begin{cases}1.4\times10^{-11} & \text{(at 9.86 } R_\odot, \text{ inner)}\\1.8\times10^{-12} & \text{(at 0.25 AU, outer)}\end{cases}
$$

WISPR 요구사항의 핵심 두 점. / WISPR stray-light requirement at the two extremes of the orbit.

---

## 6. 읽기 가이드 / Reading Guide

- §1 (서론·과학 목표) 빠르게 — SPP 미션 맥락과 9개 과학 질문(L-1 objectives + 2 unique). 표 1·2를 메모. / Skim §1 and note Tables 1–2 (SRTM).
- §2 (Overview) — Fig. 9–10로 instrument 구조와 IDPU/CIE 위계 파악. Table 3 instrument characteristics 암기 수준으로. / Internalize Table 3 numbers.
- §3 (Design) — §3.1 광학·baffle은 본 논문의 핵심. Fig. 14(렌즈), Fig. 15(side view + shadow line), Fig. 16(diffraction)을 주의 깊게. §3.3.1 APS detector 사양(Table 5)도 중요. / §3.1 optics and baffles is the heart of the paper.
- §4 (Operations, Data Products) — 실제 사용 흐름: Table 7 observing program, Table 9 data product levels. / Practical mission-operations layer.
- §5 (Summary) — 한 페이지 요약. / One-page wrap.
- 추천 순서: Abstract → §1.2–1.3 → Table 1 → Table 3 → §3.1 → Fig. 15 + Fig. 17 → Tables 7–9 → §5. / Suggested reading order above.

특히 다음 그림에 시간을 더 투자: Fig. 1(상상도), Fig. 2(LOS Thomson 누적), Fig. 7(F·K·노이즈 vs ε), Fig. 13(BOL/EOL 산란광), Fig. 15(side view), Fig. 17(2-telescope 산란광 개선). / Spend extra time on Figs. 1, 2, 7, 13, 15, 17.

---

## 7. 현대적 의의 / Modern Significance

WISPR은 PSP가 2018년 발사 후 24회 근일점 동안 실제로 코로나 안에서 영상을 보낸 최초이자 유일한 카메라이다. 이 논문은 그 이전에 출판된 instrument paper로서, PSP 시대의 모든 WISPR 관측 논문(예: Howard et al. 2019 Nature; Hess et al. 2020 streamer 관측; Korendyke et al. 2024 calibration)이 인용하는 사양·과학목표의 원전이다. SPP가 Alfvén 표면을 통과하면서(2021) 얻은 첫 in-situ 관측은 이 논문이 약속한 "local imaging + in-situ"의 시너지를 정확히 실현했고, 코로나 streamer/HCS의 미세구조, dust-free zone의 실증, sungrazing comet의 고해상도 관측 등 PSP의 가장 인상적인 결과들이 모두 WISPR에서 나왔다. 더 나아가, WISPR의 SoloHI(2020 Solar Orbiter) 자매 기기와 함께 다중 시점 헬리오스피어 영상이 가능해져, CME의 3D 재구성과 우주 기상 예보 정확도 향상에 직접 기여한다.

WISPR is the only camera ever to image the corona from inside it. After PSP's 2018 launch, every WISPR science result (e.g., the streamer "switchback" context, dust-free zone constraints, sungrazer comet imaging, coronal loop fine structure) traces its specifications and science framing to this 2016 paper. PSP's crossing of the Alfvén surface in 2021 fulfilled the precise promise made here: simultaneous local imaging and in-situ measurement of the inner corona. Combined with its sister instrument SoloHI on Solar Orbiter (2020), WISPR enables multi-vantage heliospheric reconstruction, advancing both heliophysics and operational space-weather forecasting.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
