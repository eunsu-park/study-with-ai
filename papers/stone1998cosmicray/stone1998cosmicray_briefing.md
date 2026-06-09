---
title: "Pre-Reading Briefing: The Cosmic-Ray Isotope Spectrometer for the Advanced Composition Explorer"
paper_id: "68_stone_1998_cris"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# The Cosmic-Ray Isotope Spectrometer for the Advanced Composition Explorer: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Stone, E. C. et al., "The Cosmic-Ray Isotope Spectrometer for the Advanced Composition Explorer", Space Science Reviews 86, 285-356, 1998. DOI: 10.1023/A:1005075813033
**Author(s)**: E. C. Stone, C. M. S. Cohen, W. R. Cook, A. C. Cummings, B. Gauld, B. Kecman, R. A. Leske, R. A. Mewaldt, M. R. Thayer, B. L. Dougherty, R. L. Grumm, B. D. Milliken, R. G. Radocinski, M. E. Wiedenbeck, E. R. Christian, S. Shuman, H. Trexel, T. T. von Rosenvinge, W. R. Binns, D. J. Crary, P. Dowkontt, J. Epstein, P. L. Hink, J. Klarmann, M. Lijowski, M. A. Olevitch
**Year**: 1998

---

## 1. 핵심 기여 / Core Contribution

CRIS는 ACE 미션의 6개 분광기 중 갈락틱 우주선(Galactic Cosmic Rays, GCR)의 동위원소 조성을 측정하기 위해 설계된 핵심 기기로, ~50–500 MeV/nucleon 에너지 범위에서 Z = 2–30 원소의 동위원소를 ≲0.25 amu 분해능으로 식별한다. 이전 우주 임무(ISEE-3, Ulysses, CRRES, SAMPEX)보다 한 차원 더 큰 ~250 cm² sr의 기하학적 인자(geometrical factor)와 Si(Li) 검출기 4개 스택 + Scintillating Optical Fiber Trajectory(SOFT) 호도스코프의 새로운 조합을 통해 2년 동안 5×10⁶개의 정지 중원자핵을 수집할 수 있는 통계 능력을 달성한다.

CRIS, one of six spectrometers on the ACE mission, is the primary instrument for measuring the elemental and isotopic composition of galactic cosmic rays (GCRs) over the ~50–500 MeV/nucleon energy range with mass resolution ≲0.25 amu for nuclei from Z = 2 to Z = 30. By combining four stacks of Si(Li) detectors with a Scintillating Optical Fiber Trajectory (SOFT) hodoscope, CRIS achieves a geometrical factor of ~250 cm² sr — an order of magnitude larger than previous space-borne isotope spectrometers (ISEE-3, Ulysses, CRRES, SAMPEX) — enabling the collection of ~5×10⁶ stopping heavy nuclei over two years of solar minimum operation.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 후반에 이르러 우주선 동위원소 측정은 ISEE-3 (1978), Voyager (1977), CRRES (1990), Ulysses (1990) 등의 임무를 통해 누적되어 왔으나, 통계적 정확도 부족으로 ⁵⁹Ni/⁵⁸Ni, ⁵⁷Co/⁵⁶Fe, ²²Ne/²⁰Ne 등 핵심 비율의 신호 검증이 불가능했다. ACE는 1997년 8월 25일 발사되어 L1 라그랑주 점에서 광범위한 에너지 영역(<1 keV/nuc 태양풍부터 ~500 MeV/nuc GCR까지)을 통합 측정하기 위해 설계되었으며, CRIS는 그 중 가장 높은 에너지 10년대(highest decade)를 담당한다.

By the late 1990s, cosmic-ray isotope measurements had accumulated through ISEE-3 (1978), Voyager (1977), CRRES (1990), and Ulysses (1990), but insufficient statistics prevented decisive measurements of crucial ratios such as ⁵⁹Ni/⁵⁸Ni, ⁵⁷Co/⁵⁶Fe, and ²²Ne/²⁰Ne. ACE, launched 25 August 1997 to the L1 Lagrangian point, was designed to span the full energy range (<1 keV/nuc solar wind through ~500 MeV/nuc GCRs) with six high-resolution spectrometers and three monitors, of which CRIS covers the highest energy decade.

### 타임라인 / Timeline

```
1952 ──── Rossi: 'High-Energy Particles' (energy loss theory)
1976 ──── Fisher et al.: Isotopic composition Z=5–26
1978 ──── Soutoul, Cassé, Juliusson: time-delay clocks (⁵⁹Ni)
1979 ──── Stone & Wiedenbeck: GCR source isotope abundances
1985 ──── Meyer / Breneman & Stone: FIP fractionation
1990 ──── HEAO-3-C2 (Engelmann et al.): elemental composition
1990 ──── Hubert et al.: Range-Energy tables 2.5–500 MeV/nuc
1990 ──── Ulysses HET / CRRES launches
1993 ──── Cook et al.: custom analog VLSI for ACE
1996 ──── Allbritton et al.: large-diameter Si(Li) detectors
1997 ──── ACE launch (25 August), CRIS turn-on (27 August)
1998 ──── ★ THIS PAPER ★ — CRIS instrument description
2003 ──── Wiedenbeck et al.: ⁵⁹Co excess (acceleration delay confirmed)
2014 ──── Binns et al.: ²²Ne/²⁰Ne Wolf-Rayet origin
2025+ ──── CRIS still operating after ~28 years
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **dE/dx vs E technique** — Bethe-Bloch 에너지 손실 공식과 잔여 에너지(residual energy)에서 입자 종 식별 / Identifying particle species from energy loss versus residual energy using the Bethe-Bloch ionization formula
- **Range-Energy relation** — 입자가 정지할 때까지의 사정거리 R(E/M) ∝ M/Z² · (E/M)^a, a≈1.7 / Particle range-to-stopping integrated over 1/(dE/dx)
- **Bohr/Landau fluctuations** — 얇은 흡수체에서 에너지 손실의 통계적 변동 / Statistical width of energy loss distribution in thin absorbers
- **Multiple Coulomb scattering** — 경로 길이 불확실성을 만드는 누적 산란 / Cumulative angular deflection causing path-length uncertainty
- **Si(Li) detector** — 리튬 드리프팅으로 보상한 두꺼운 실리콘 검출기 / Lithium-drifted silicon detector with thick depletion layer
- **Scintillating optical fibers** — 폴리스티렌 코어 + 형광 도펀트(BPBD/DPOPOP), 200 µm 단면적 / Plastic fiber with fluorescence dyes for charged-particle tracking
- **Image-intensified CCD readout** — MCP 광전증배관 + CCD 이미지 처리 / Microchannel plate gain + CCD pixel readout
- **GCR propagation models** — Leaky-box model, secondary/primary ratios (B/C, ¹⁰Be/⁹Be) / Cosmic ray confinement and fragmentation in the ISM
- **Solar minimum modulation** — 태양풍이 GCR 강도를 약화시키는 정도의 11년 주기 / Solar-cycle modulation of GCR intensity at Earth

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| CRIS | Cosmic-Ray Isotope Spectrometer — ACE의 GCR 동위원소 측정기 / The galactic cosmic-ray isotope spectrometer on ACE |
| SOFT | Scintillating Optical Fiber Trajectory hodoscope — 궤적 측정용 형광 광섬유 호도스코프 / Trajectory hodoscope of scintillating fibers |
| Si(Li) | Lithium-drifted silicon detector — 두께 ~3 mm, 직경 10 cm / Lithium-compensated silicon, 3 mm × 10 cm |
| Hodoscope | 위치 측정용 검출기 어레이(여기선 3 xy 평면 + trigger) / Detector array for position measurement |
| Mass resolution σ_M | 동위원소 질량 측정의 표준편차(amu 단위) / Standard deviation of mass determination in atomic mass units |
| Geometrical factor AΩ | 검출기 수용 면적 × 입체각 (cm² sr) / Detector acceptance area × solid angle |
| Image intensifier | MCP 기반 광전 증폭관, S-20 광음극 / MCP-based photon multiplier with S-20 photocathode |
| GCRS | Galactic Cosmic-Ray Source composition — 가속 전 모재의 조성 / Source composition before propagation |
| FIP effect | First Ionization Potential 분별효과 / Element fractionation by first ionization potential |
| Propagation clock | β⁻ 붕괴로 측정하는 GCR 갇힘시간 (¹⁰Be, ²⁶Al) / Confinement time clock via β decay |
| Acceleration delay clock | 전자포획 동위원소(⁵⁹Ni, ⁵⁷Co)로 측정하는 합성-가속 시간차 / EC isotope clock between nucleosynthesis and acceleration |
| Range-Energy ℛ_{Z,M}(ε) | 핵이 ε 만큼 에너지로 정지하기까지의 g cm⁻² 단위 사정거리 / Particle range integrated to stop |
| Dead layer | 검출기 표면의 비활성 영역(~60 µm) / Inactive surface layer of detector |

---

## 5. 수식 미리보기 / Equations Preview

**Equation 1 — Range-energy fundamental relation / 사정거리-에너지 기본 관계식**:
$$\mathcal{R}_{Z,M}(E/M) - \mathcal{R}_{Z,M}(E'/M) = L$$
침투한 두께 L을 사정거리의 차로 표현한다. 이것이 ΔE-E' 기법의 근간. / Penetrated thickness L equals the difference of ranges; foundation of the ΔE-E' technique.

**Equation 2 — Mass from power-law range / 파워-로 사정거리에서 유도한 질량**:
$$M \simeq (k/Z^2 L)^{1/(a-1)} (E^a - E'^a)^{1/(a-1)}$$
ℛ ∝ k(M/Z²)(E/M)^a, a ≈ 1.7로 근사할 때 질량의 명시적 표현. / Explicit expression for the mass given ℛ ∝ k(M/Z²)(E/M)^a with a ≈ 1.7.

**Equation 3 — Charge solution / 전하 표현식**:
$$Z \simeq \left(\frac{k}{L(2+\epsilon)^{(a-1)}}\right)^{1/(a+1)} (E^a - E'^a)^{1/(a+1)}$$
M/Z = 2+ε 가정 하에 Z를 풀어낸다. / With M/Z ≈ 2+ε, solve for Z.

**Equation 7 — Bohr/Landau variance / 보어/란다우 분산**:
$$\left(\frac{d\sigma_{\Delta E}^2}{dx}\right)_\text{Landau} = Z^2 \frac{(0.396 \text{ MeV})^2}{\text{g cm}^{-2}} \frac{Z_m}{A_m} \frac{\gamma^2 + 1}{2}$$
얇은 흡수체에서 에너지 손실 분산의 단위경로당 증가율. / Per-path-length variance of energy loss in thin absorbers.

**Equation 11 — Multiple scattering variance / 다중 산란 분산**:
$$\frac{d\sigma_{\delta\theta}^2}{dx} \simeq \left(\frac{Z}{M} \frac{0.0146}{\beta^2 \gamma}\right)^2 \frac{1}{X_0}$$
실리콘 X₀ = 21.82 g cm⁻². 경로 길이 오차의 주된 원인. / In silicon X₀ = 21.82 g cm⁻²; the main path-length error source.

---

## 6. 읽기 가이드 / Reading Guide

- **Section 2 (과학목표 / Scientific Objectives)** — 4가지 동위원소 그룹(primary, acceleration delay clock, propagation clock, reacceleration clock)의 천체물리학적 의미를 먼저 잡아야 후속 측정 요구가 와닿는다. / Grasp the four isotope groups before instrument requirements make sense.
- **Section 3 (설계 요구사항 / Design Requirements)** — 식 (1)–(3)의 ΔE-E' 기법 직관을 반드시 이해할 것. CRIS의 모든 설계 결정이 여기서 파생된다. / Understand the ΔE-E' technique; all design choices flow from it.
- **Section 4 (센서 시스템)** — SOFT(궤적)와 Si(Li) 스택(에너지)의 역할 분담을 파악. Figure 6, 11이 핵심 도면이다. / SOFT does trajectory; Si(Li) stack does energy. Figures 6 and 11 are the key drawings.
- **Section 5 (전자장치 / Electronics)** — Forth로 프로그램된 RTX2010, 32 PHA 채널, Actel A1020 FPGA 등 1990년대 우주 등급 디지털 설계의 우수한 예. / RTX2010 + Forth + Actel FPGAs — excellent example of 1990s space digital design.
- **Section 8 (기대 성능)** — 그림 18, 19, 20이 임무 수확량을 설명. ⁵⁶Fe 1.4×10⁵ events/yr 같은 기대값이 제시된다. / Figures 18, 19, 20 quantify expected yields (e.g., ⁵⁶Fe at 1.4×10⁵ events/yr).
- **Appendix A (질량 분해능)** — 모든 노이즈 소스를 quadrature로 합산하는 'error budget'의 우수 예시. 처음 읽으면 어렵지만 한 번 이해하면 다른 분광기에도 적용 가능. / The mass-resolution error budget is an excellent template applicable to other spectrometers.

---

## 7. 현대적 의의 / Modern Significance

CRIS는 1997년 발사 후 28년이 지난 현재(2026년)에도 여전히 운용 중이며, ACE 미션의 가장 오래된 우주선 분광기 중 하나로 자리잡았다. CRIS 데이터는 ⁵⁹Co 초과를 통한 acceleration delay (Wiedenbeck et al. 2003), ²²Ne/²⁰Ne의 Wolf-Rayet 별 기원 (Binns et al. 2008), ⁶⁰Fe의 GCR 검출(Binns et al. 2016) 등 천체물리학적으로 중요한 발견을 가능하게 했다. 또한 SOFT 호도스코프 방식은 후속 미션(예: NUCLEON, ISS-CREAM)에서 단순화된 변형으로 채택되었다. 현재 IMAP(Interstellar Mapping and Acceleration Probe, 2025+) 미션에 탑재되는 IS⊙IS는 CRIS의 핵심 기법(ΔE-E' + 궤적 호도스코프)을 21세기 부품으로 재구현한다.

CRIS, launched in 1997, remains operational in 2026 — nearly three decades later — making it one of the longest-lived cosmic-ray spectrometers ever flown. CRIS data has produced landmark astrophysical results: ⁵⁹Co excess confirming an acceleration delay (Wiedenbeck et al. 2003), the ²²Ne/²⁰Ne ratio establishing a Wolf-Rayet stellar wind contribution (Binns et al. 2008), and the first detection of GCR ⁶⁰Fe (Binns et al. 2016). The SOFT hodoscope concept has propagated to subsequent missions (NUCLEON, ISS-CREAM) in simplified form, and the upcoming IMAP IS⊙IS instrument (2025+) re-implements the same ΔE-E'-plus-trajectory paradigm with modern components.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
