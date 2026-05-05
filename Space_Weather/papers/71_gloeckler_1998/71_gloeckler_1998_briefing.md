---
title: "Pre-Reading Briefing: Investigation of the Composition of Solar and Interstellar Matter Using Solar Wind and Pickup Ion Measurements with SWICS and SWIMS on the ACE Spacecraft"
paper_id: "71_gloeckler_1998"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# SWICS and SWIMS on ACE: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Gloeckler, G., Cain, J., Ipavich, F. M., Tums, E. O., Bedini, P., Fisk, L. A., Zurbuchen, T. H., Bochsler, P., Fischer, J., Wimmer-Schweingruber, R. F., Geiss, J., and Kallenbach, R., "Investigation of the Composition of Solar and Interstellar Matter Using Solar Wind and Pickup Ion Measurements with SWICS and SWIMS on the ACE Spacecraft", *Space Science Reviews* **86**, 497–539 (1998). DOI: 10.1023/A:1005036131689
**Author(s)**: G. Gloeckler, J. Cain, F. M. Ipavich, E. O. Tums, P. Bedini, L. A. Fisk, T. H. Zurbuchen, P. Bochsler, J. Fischer, R. F. Wimmer-Schweingruber, J. Geiss, R. Kallenbach
**Year**: 1998

---

## 1. 핵심 기여 / Core Contribution

이 논문은 1997년 발사된 ACE(Advanced Composition Explorer) 위성에 탑재된 두 개의 정밀 질량 분광계, **SWICS(Solar Wind Ion Composition Spectrometer)** 와 **SWIMS(Solar Wind Ions Mass Spectrometer)** 의 설계, 측정 원리, 성능, 그리고 과학적 목표를 종합적으로 기술한다. SWICS는 정전기 편향(E/Q) → 후가속(post-acceleration, ≤30 kV) → 비행 시간(TOF, 10 cm) → 잔여 에너지(SSD) 측정의 5단계 기법으로 H부터 Fe까지의 모든 주요 태양풍 이온의 **질량 M, 전하 상태 Q, 입사 에너지 E, 속도 V** 를 동시에 결정한다. SWIMS는 SWICS와 같은 정전기 편향(E/Q) 후 **하모닉 정전기 포텐셜** 안에서 TOF만으로 M/Q*≈M (대부분 Q*=1)을 측정해 M/ΔM > 100의 고질량 분해능을 제공한다. 이 두 기기가 함께 태양풍의 원소·동위원소·전하 상태 조성, 픽업 이온의 분포 함수, 그리고 성간 물질의 특성을 1 AU 황도면에서 전례 없는 정밀도로 매핑한다.

This paper comprehensively describes the design, measurement principles, performance, and scientific goals of two high-precision mass spectrometers, **SWICS (Solar Wind Ion Composition Spectrometer)** and **SWIMS (Solar Wind Ions Mass Spectrometer)**, aboard the ACE (Advanced Composition Explorer) spacecraft launched in 1997. SWICS uses a five-stage technique—electrostatic deflection (E/Q) → post-acceleration (up to 30 kV) → time-of-flight (TOF, 10 cm) → residual energy (SSD)—to simultaneously determine the **mass M, charge state Q, incident energy E, and velocity V** of every major solar wind ion from H through Fe. SWIMS, after the same E/Q analysis, measures TOF in a **harmonic electrostatic potential** so that τ depends only on M/Q* ≈ M (since most ions emerge from the carbon foil as Q*=1), achieving high mass resolution M/ΔM > 100. Together the two instruments map solar wind elemental, isotopic, and charge-state composition, pickup-ion distribution functions, and the local interstellar medium with unprecedented precision in the 1 AU ecliptic.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1990년대 후반은 태양풍 기원과 가속 메커니즘에 대한 의문이 깊어지던 시기였다. SOHO/CDS·SUMER 등 원격 분광 관측은 코로나의 온도와 밀도 프로파일을 보여주었지만, **현장(in-situ) 조성 측정**과 결합되지 않으면 가속 영역의 경계 조건을 확정할 수 없었다. SWICS는 이미 Ulysses(1990 발사)에서 운용되었고, 그 비행 예비기(flight spare)를 개량해 ACE에 탑재한 것이다. SWIMS는 WIND/SMS의 MASS와 SOHO/CELIAS의 MTOF 기술을 결합한 후속 기기이다. ACE의 L1 라그랑주점 궤도는 자기권 들어가기 직전의 태양풍을 지속적으로 감시하기에 이상적이었고, ISTP(International Solar Terrestrial Physics) 프로그램의 핵심 측정 자산이었다.

The late 1990s was a period of deepening questions about solar wind origin and acceleration mechanisms. SOHO/CDS and SUMER remote spectroscopy revealed coronal temperature and density profiles, but without **in-situ composition measurements**, boundary conditions in the acceleration region remained unconstrained. SWICS had already operated on Ulysses (launched 1990), and its flight spare was improved for ACE. SWIMS combined the MASS technique of WIND/SMS with the MTOF technique of SOHO/CELIAS. ACE's L1 Lagrangian orbit was ideal for monitoring the solar wind just before it enters the magnetosphere and was a core measurement asset of the ISTP program.

### 타임라인 / Timeline

```
1968 ─ Hundhausen et al.: freeze-in concept introduced
1972 ─ Geiss et al.: Apollo 16 SWC foil — solar wind isotopes
1977 ─ Gloeckler: TOF-vs-E technique design (UMd TR 77-043)
1985 ─ Möbius et al.: discovery of interstellar He+ pickup ions (AMPTE/SULEICA)
1990 ─ Ulysses launch, SWICS first flight; Hamilton et al. HMRS technique
1992 ─ Gloeckler et al.: Ulysses SWICS instrument paper
1995 ─ SOHO launch (CELIAS-MTOF); Geiss et al. inner-source pickup ions
1997 Aug ─ ACE launch
1998 ─ THIS PAPER (Space Sci. Rev. 86)
2007 ─ Zurbuchen & von Steiger reviews using SWICS/SWIMS data
```

---

## 3. 필요한 배경 지식 / Prerequisites

**물리 / Physics**
- 정전기 편향 분석기(electrostatic analyzer, ESA)와 E/Q 선택 원리
- 비행 시간(TOF) 측정과 이차 전자(secondary electron) 생성
- 박막(carbon foil) 통과 시의 에너지 손실, 핵 결손(nuclear defect), 산란
- 마이크로채널 플레이트(MCP, chevron stack) 동작
- 고체상 검출기(solid-state detector, SSD)에서 잔여 에너지 측정
- 하모닉 포텐셜에서의 진동(SWIMS의 V ∝ z²)

**플라즈마 / Plasma**
- 태양풍 매개변수(speed 300–800 km s⁻¹, density, temperature)
- Frozen-in 코로나 전자 온도와 이온화 평형
- 픽업 이온 형성: vsw × B 전기장에 의한 가속, ring → shell distribution
- FIP(First Ionization Potential) 효과
- 단열 냉각(adiabatic cooling) in expanding solar wind

**수학 / Math**
- 운동량과 에너지의 단순 보존 법칙
- 시간-속도 관계: V = d/τ
- 가우스 잡음 결합(quadrature sum) — 분해능 계산

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **SWICS** | Solar Wind Ion Composition Spectrometer — ESA + post-accel + TOF + SSD; 5단계로 M, Q, E를 결정 / 5-stage instrument identifying M, Q, E |
| **SWIMS** | Solar Wind Ions Mass Spectrometer — ESA + 하모닉 TOF; M/ΔM>100 / harmonic-potential TOF achieving M/ΔM>100 |
| **TOF** | Time-of-Flight; 시작-정지 검출기 사이 비행 시간 τ / time τ between start and stop detectors |
| **E/Q** | 에너지 대 전하 비율 (keV e⁻¹); ESA 통과 조건 / energy-per-charge selected by ESA |
| **Post-acceleration** | SWICS 비행 시간 시스템 진입 직전 ≤30 kV로 이온을 가속 / accelerates ions by ≤30 kV before TOF |
| **MCP** | Microchannel Plate (chevron); 이차 전자 검출 / detects secondary electrons |
| **Carbon foil** | 1.5 μg cm⁻² 박막; start signal 발생, 또한 SWIMS에서 전하 평형 / thin foil providing start signal and charge equilibration |
| **Triple coincidence** | start–stop–energy 3중 일치; 배경 10⁻²⁰배 감소 / TCR — start-stop-SSD coincidence reducing background |
| **Freeze-in temperature** | 이온화 상태가 고정되는 코로나 온도; 예: O⁷⁺/O⁶⁺ 비율로 결정 / corona electron temperature where charge state is frozen, derived e.g. from O⁷⁺/O⁶⁺ |
| **Pickup ion** | 성간 또는 내부 중성 원자가 광이온화/전하교환 후 vsw×B로 가속된 이온; ring distribution / interstellar or inner-source neutral, ionized and gyrating, forming ring at V=Vsw |
| **FIP effect** | 1차 이온화 퍼텐셜 ≤10 eV 원소가 광구 대비 코로나에서 4배 정도 풍부 / low-FIP elements enhanced in corona vs photosphere |
| **HMRS** | High Mass Resolution Spectrometer — SWIMS의 하모닉 포텐셜 부분 / harmonic-potential mass analyzer of SWIMS |

---

## 5. 수식 미리보기 / Equations Preview

**(1) SWICS ion identification / SWICS 이온 식별**
$$M = 2\left(\frac{\tau}{d}\right)^{2}\frac{E_\text{meas}}{\alpha},\qquad \frac{M}{Q}\approx 2\left(\frac{\tau}{d}\right)^{2}U_a$$

여기서 d=10 cm, τ=비행 시간, E_meas=SSD가 측정한 잔여 에너지, α=핵 결손 보정 인수, U_a=후가속 전위. 시간과 에너지의 두 측정만으로 M과 Q를 모두 분리할 수 있다는 것이 핵심이다. / d=10 cm, τ=time of flight, E_meas=residual energy, α=nuclear defect, U_a=post-accel; two measurements yield both M and Q.

**(2) SWIMS harmonic potential mass / SWIMS 하모닉 질량**
$$\tau \propto \sqrt{M/Q^{*}}$$

V ∝ z² 인 하모닉 포텐셜에서 단순조화진동의 주기는 입사 속도·각도와 무관하고 오직 M/Q*에만 의존. Q*=1인 경우 τ ∝ √M. / In V ∝ z² the SHO period depends only on M/Q*, not on entry velocity or angle.

**(3) Ion speed / 이온 속도**
$$V_\text{ion}=438\sqrt{(E/Q)/(M/Q)}\ \text{km s}^{-1}$$

E/Q는 keV e⁻¹, M/Q는 amu e⁻¹. ESA 단계와 결합해 입자별 속도 벡터를 도출한다. / E/Q in keV e⁻¹, M/Q in amu e⁻¹ — speed for each PHA event.

**(4) Charge-state ratio → freeze-in T_e / 전하 상태비 → 동결 온도**
$$\frac{n_{O^{7+}}}{n_{O^{6+}}}=\frac{C_{6}(T_e)}{R_{7}(T_e)}\approx \exp\!\left(-\frac{\Delta E_{67}}{kT_e}\right)$$

코로나 ~1.5 R_⊙ 부근에서 동결되는 전자 온도를 측정. / freeze-in T_e of the corona near 1.5 R_⊙.

**(5) Pickup-ion ring → shell injection / 픽업 이온 링 → 셸 주입**
$$f(v)\propto \frac{1}{v^{2}}\,\delta(v-v_{sw})\quad\xrightarrow{\text{adiabatic}}\quad f(w)\propto w^{-3/2}\Theta(1-w),\;w=V/V_{sw}$$

vsw×B 가속 후 단열 냉각으로 V<Vsw 영역이 채워지는 분포(Vasyliunas-Siscoe). / after adiabatic cooling, the cooled distribution fills V<Vsw with the Vasyliunas–Siscoe form.

---

## 6. 읽기 가이드 / Reading Guide

- **Section 1–2**: 배경과 10가지 과학 목표를 빠르게 훑되, (1)–(10)의 항목별 우선순위가 어떻게 검출기 설계로 환원되는지 주목. / Background and 10 scientific objectives — note how each maps to a detector requirement.
- **Section 3.1 SWICS**: Figure 1·2·5를 함께 보며 5단계(collimator → ESA → post-accel → TOF → SSD) 흐름을 손에 익히기. 식 (1)을 직접 유도해보라. / Read Figs. 1, 2, 5 together; rederive equation (1) by hand.
- **Section 3.2 SWIMS**: 하모닉 포텐셜 V ∝ z²에서 등시성(isochronism) 결과를 받아들이는 것이 핵심. Figure 10의 비대칭 트랩 구조와 finned hyperbola의 백그라운드 억제를 이해. / Accept the isochronism in V ∝ z² and understand how the finned hyperbola suppresses neutral background.
- **Section 3.3 S³DPU**: 보드 위 분류(on-board classification)가 텔레메트리 한계를 어떻게 해결하는지(matrix box) 살피기. / How on-board classification handles telemetry limits.
- **Section 5**: 분해능 표(Table VI)와 Figure 13의 M-vs-M/Q 산포도를 번갈아 보면 실제 비행 데이터의 의미가 분명해짐. / Compare Table VI and Fig. 13 to interpret real in-flight data.
- **Section 6 Browse**: 12분 단위로 (1) He²⁺·O⁶⁺ 속도/온도, (2) freeze-in T_e from O⁷⁺/O⁶⁺, (3) Fe/O FIP 비, (4) He/O 비가 매일 산출되는 것을 확인. / Routine 12-min browse parameters.

---

## 7. 현대적 의의 / Modern Significance

ACE/SWICS와 SWIMS가 산출한 데이터는 그 이후 25년 이상의 태양풍 조성 연구의 기준 표준이 되었다. **freeze-in 온도(O⁷⁺/O⁶⁺, C⁶⁺/C⁵⁺)** 는 fast/slow 태양풍의 코로나 기원을 진단하는 핵심 변수로 자리잡았고(Geiss 1995; Zurbuchen et al. 2002), Fe/O FIP 비는 active region wind 식별에 사용된다. SWICS 픽업 이온 측정은 내부 source(0.1 AU 부근의 먼지 증발) 발견(Geiss 2000; Schwadron 2000)과 연결되며, 이는 향후 Parker Solar Probe(2018)와 Solar Orbiter(2020)의 SWA/HIS 같은 후속 기기 설계에 직접적인 영감을 주었다.

The data products from ACE/SWICS and SWIMS have become the reference standard of solar wind composition research for over 25 years. **Freeze-in temperatures (O⁷⁺/O⁶⁺, C⁶⁺/C⁵⁺)** are now essential diagnostics of fast/slow solar wind coronal origin (Geiss 1995; Zurbuchen et al. 2002), and Fe/O FIP ratio is used to identify active-region wind. The pickup-ion measurements led to the discovery of an inner source (~0.1 AU dust-evaporation neutrals; Geiss 2000; Schwadron 2000), and directly inspired follow-on instruments such as SWA/HIS on Solar Orbiter (2020) and SWEAP/SPAN-Ion on Parker Solar Probe (2018).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
