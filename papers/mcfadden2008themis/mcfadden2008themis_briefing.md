---
title: "Pre-Reading Briefing: The THEMIS ESA Plasma Instrument and In-flight Calibration"
paper_id: "80_mcfadden_2008"
topic: Space_Weather
date: 2026-04-25
type: briefing
---

# The THEMIS ESA Plasma Instrument and In-flight Calibration: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: McFadden, J. P., Carlson, C. W., Larson, D., Ludlam, M., Abiad, R., Elliott, B., Turin, P., Marckwordt, M., & Angelopoulos, V. (2008). The THEMIS ESA Plasma Instrument and In-flight Calibration. *Space Science Reviews*, 141, 277–302. DOI: 10.1007/s11214-008-9440-2
**Author(s)**: J. P. McFadden, C. W. Carlson, D. Larson, M. Ludlam, R. Abiad, B. Elliott, P. Turin, M. Marckwordt, V. Angelopoulos
**Year**: 2008

---

## 1. 핵심 기여 / Core Contribution

This paper documents the THEMIS Electrostatic Analyzer (ESA) plasma instrument — a pair of "top-hat" hemispherical analyzers measuring electrons (a few eV–30 keV) and ions (a few eV–25 keV) on each of five identical THEMIS probes — and details the multi-month ground and in-flight calibration campaign that brought the ten sensors into mutual agreement before the substorm science phase. The novelty lies less in the instrument concept (inherited from FAST, Carlson & McFadden 1998) than in the **rigorous cross-calibration methodology** that exploits the close-formation flight of the five spacecraft to harmonize relative efficiencies, energy-dependent leakage corrections, dead-time, and absolute sensitivity using the magnetosheath, the solar wind, and Wind/SWE comparisons.

본 논문은 THEMIS의 정전기 분석기(ESA) 플라즈마 측기 — 5대의 동일 위성에 탑재된 한 쌍의 "탑햇(top-hat)" 반구형 분석기로 전자(수 eV–30 keV)와 이온(수 eV–25 keV)의 3차원 분포함수를 측정함 — 의 설계와, 자기권 폭발(서브스톰) 과학 운영 전에 10개 센서를 상호 일치시키기 위해 수개월간 수행된 지상 및 비행 중 교정 작업을 정리한다. 핵심 기여는 측기 개념의 신규성보다는, 5개 위성이 가까운 편대 비행 상태인 동안 자기초(magnetosheath), 태양풍, Wind/SWE 비교 자료를 이용해 상대 효율, 에너지 의존 누설장 보정, 데드타임, 절대 감도를 일관되게 맞춘 **교차 교정 방법론의 엄밀함**에 있다.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

THEMIS (Time History of Events and Macroscale Interactions during Substorms) was a NASA MIDEX mission launched on 17 February 2007 with five identical probes designed to resolve the "where and when" of substorm onset by aligning along the magnetotail. After a 7-month coast phase the probes were placed in highly elliptical orbits (apogee ~11.8 R_E for inner probes, ~19.6 to ~31.6 R_E for outer probes). The plasma instrument's heritage traces back to the Carlson 1983 top-hat ESA, the FAST plasma instrument (Carlson et al. 2001), and the Cluster mission's plasma sensors. Cluster — launched in 2000 — was THEMIS's only multi-spacecraft predecessor, but Cluster's electron and ion sensors were built by separate teams with different formats, complicating cross-calibration.

THEMIS는 자기권 폭발(서브스톰)의 발생 지점과 시각을 결정하기 위해 설계된 NASA MIDEX 5위성 미션으로, 2007년 2월 17일 발사되었다. 7개월의 합체 비행 후, 안쪽 3기는 약 11.8 R_E, 바깥쪽 2기는 약 19.6, 31.6 R_E의 원지점을 갖는 고타원 궤도에 배치되어 자기꼬리(magnetotail) 정렬을 가능하게 했다. ESA 플라즈마 측기의 계보는 Carlson(1983)의 탑햇 ESA, FAST 측기(Carlson et al. 2001), 그리고 Cluster 미션 플라즈마 센서로 이어진다. 2000년 발사된 Cluster는 THEMIS 이전 유일한 다위성 미션이었으나, 전자/이온 센서를 서로 다른 팀이 제작해 자료 형식과 분해능이 달랐기에 교차 교정이 까다로웠다.

### 타임라인 / Timeline

```
1958 ─ Faraday cup plasma probes (Explorer / Lunik)
1983 ─ Carlson et al.: top-hat ESA concept (RSI 55, 67)
1984 ─ Goruganthu & Wilson: MCP electron efficiency
1995 ─ Ogilvie et al.: Wind/SWE solar-wind reference
1997 ─ Cluster plasma instruments (CIS, PEACE)
2001 ─ Carlson et al.: FAST plasma instrument
2007 ─ THEMIS launch (5 probes, 17 Feb)
2008 ─ THIS PAPER + Angelopoulos 2008 mission overview
2015 ─ MMS plasma instruments (FPI) launch
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **Top-hat electrostatic analyzer geometry**: ions/electrons of energy E pass between concentric hemispheres biased to V via the relation E/q = k·V (k = analyzer constant). The 180°×6° FOV plus spacecraft spin gives 4π coverage every 3 s spin.
  - 탑햇 정전기 분석기에서 입자는 동심 반구 사이를 통과하며 E/q = k·V (k는 분석기 상수)에 따라 선택된다. 180°×6° FOV에 위성 회전(주기 3초)을 결합해 4π 입체각을 매 회전마다 덮는다.
- **Geometric factor and counts → flux**: differential flux j (cm⁻² s⁻¹ sr⁻¹ eV⁻¹) = count rate / (G·E), where G is the geometric factor in cm² sr E. For THEMIS, G_iESA ≈ 0.0061, G_eESA ≈ 0.0066 cm² sr E (in-flight values).
  - 기하 인자 G와 차분 플럭스 j = (계수율)/(G·E)의 관계. THEMIS 비행 중 값은 이온 0.0061, 전자 0.0066 cm² sr E.
- **MCP (Microchannel Plate) detection**: chevron MCPs at ~−2 kV produce ~2×10⁶ e⁻ per particle; gain depends on incidence energy and angle (Goruganthu & Wilson 1984; Gao et al. 1984).
  - MCP는 −2 kV 바이어스에서 입자 1개당 ~2×10⁶ 전자를 생성하며 입사 에너지·각도에 따라 효율이 변한다.
- **Plasma moments**: density n = ∫f d³v, velocity u = (1/n)∫v f d³v, pressure tensor P_ij = m∫(v_i−u_i)(v_j−u_j) f d³v. THEMIS computes these on board in 3-s spin cadence.
  - 플라즈마 모멘트(밀도, 속도, 압력)를 분포함수 f의 적분으로 정의. THEMIS는 3초 회전 주기로 온보드에서 계산.
- **Spacecraft potential correction**: photoelectrons and ambient electrons charge the spacecraft to Φ_sc; the EFI Langmuir probe gives Φ_sensor and one applies Φ_sc = −A·(Φ_sensor + Φ_offset) with A ≈ 1.15, Φ_offset ≈ 1.0 V.
  - 위성 전위 보정. EFI Langmuir 프로브로 측정한 Φ_sensor에 척도 인자 A와 오프셋을 적용해 실제 위성 전위를 추정.
- **Solar wind / magnetosheath plasma regimes**: solar wind n ≈ 5–20 cm⁻³, T ≈ 10 eV; magnetosheath 5×–20× compressed and heated to ~100 eV.
  - 태양풍과 자기초 플라즈마의 전형적 밀도·온도 범위.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Top-hat ESA** | 동심 반구 위에 톱햇 형 입구가 얹힌 정전기 분석기. 180°×6° 시야 / Spherical hemispherical analyzer with a top-hat aperture giving 180°×6° instantaneous FOV |
| **Analyzer constant (k)** | E/q = k·V_inner의 비례 상수. eESA k=7.9, iESA k=6.2 / Proportionality between selected energy/charge and inner-hemisphere voltage |
| **Energy resolution ΔE/E** | 통과 대역 폭. 분석기 자체 ~17%(e), ~18%(i); 측정용 32% (로그 스윕) / FWHM bandwidth of the energy passband |
| **Geometric factor (G)** | cm² sr E 단위. G = (계수율)/(차분 플럭스·E) / Sensor étendue × energy bandwidth |
| **Anode** | MCP 후단 전하 수집 전극. eESA 8개(22.5°), iESA 16개(5.6°) / Charge-collection segments behind MCP |
| **MCP gain** | 입자 1개당 출력 전하 (~2×10⁶ e⁻) / Charge multiplication per incident particle |
| **Dead time** | 펄스 처리 회복 시간. ~170 ns (전자), ~30 ns (검출기) / Recovery time of preamp + MCP |
| **Spacecraft potential Φ_sc** | 위성 charging으로 인한 전위. THEMIS는 EFI Langmuir로 추정 / Floating potential of spacecraft relative to plasma |
| **Leakage field** | 분석기 출구 그리드를 통해 침투하는 MCP 전위 영향 / Field leaking through exit grid affecting low-energy response |
| **Relative efficiency** | 양극별·룩 방향별 감도 보정 인자 (~unity ±10%) / Per-anode sensitivity factor |
| **Absolute calibration** | Wind/SWE 등 외부 표준과의 절대 비교로 결정한 감도 / Overall sensitivity tied to external "standard candle" |
| **Magnetosheath** | 활대충격(bow shock) 뒤 아음속 압축·가열된 태양풍 플라즈마 영역 / Compressed shocked solar-wind plasma downstream of bow shock |

---

## 5. 수식 미리보기 / Equations Preview

### Eq. (1) Spacecraft potential / 위성 전위
$$\Phi_{\rm sc} = -A\,(\Phi_{\rm sensor} + \Phi_{\rm offset})$$

위성 전위는 Langmuir 센서 측정값에 척도 인자 A(~1.15)와 오프셋(~1.0 V)을 곱해 얻는다. 이 보정 없이는 광전자가 전자 밀도를 크게 오염시킨다. / Spacecraft potential is reconstructed from the Langmuir probe reading scaled by A ≈ 1.15 and offset Φ_offset ≈ 1.0 V; without this, photoelectrons inflate electron densities.

### Eq. (3) THA/THB potential mapping / THA·THB 전위 매핑
$$\Phi_{\rm THA} = \Phi_{\rm THB} = 0.49\,\Phi_{\rm THD} + 1.22$$

EFI 미배치 위성 THA·THB의 전위를 THD로부터 회귀로 추정. 6월 22일 EFI 가드 전압 변경 후에는 식 (4) Φ_THA = 0.8·Φ_THD를 사용. / Empirical regression to estimate Φ on probes lacking deployed Langmuir sensors during the early mission, replaced by Eq. (4) after EFI guard-voltage change.

### Differential flux / 차분 플럭스
$$j(E,\Omega) = \frac{C(E,\Omega)}{G\,E\,\Delta t\,\varepsilon_{\rm eff}}$$

여기서 C는 계수, G는 기하 인자, ε_eff(E) = ε_MCP(E)·ε_grid(E)·ε_anode는 누적 효율. / Counts converted to differential energy flux via geometric factor and effective efficiency chain.

### Plasma moments / 플라즈마 모멘트
$$n = \sum_{i,j,k} f_{ijk}\,\Delta^3 v_{ijk}, \quad \mathbf{u} = \frac{1}{n}\sum f\,\mathbf{v}\,\Delta^3 v$$

THEMIS는 위성 전위 보정을 포함한 모멘트를 온보드에서 계산하는 최초의 미션. / On-board moments including spacecraft-potential correction were a THEMIS first.

### Velocity ↔ Energy / 속도-에너지 변환
$$v = \sqrt{\frac{2E}{m}},\qquad d^3 v = v^2\,dv\,d\Omega = \frac{2E}{m}\sqrt{\frac{2E}{m}}\,\frac{dE}{E}\,d\Omega$$

차분 플럭스에서 분포함수로 변환할 때 사용. / Used to convert differential flux j(E,Ω) into the velocity distribution f(v).

---

## 6. 읽기 가이드 / Reading Guide

**Section 1 (Introduction & 1.1–1.3)** — Skim Table 1 carefully: it lists all the numeric specs you will need for any quantitative work. Note the difference between "predicted" and "in-flight" geometric factors (~25–40% lower in flight). Figures 1–4 are mostly hardware photos; speed-read.
**섹션 1**: Table 1을 정독. 예측 vs 비행 중 기하 인자의 차이(약 25–40% 감소)에 주목. 그림 1–4는 하드웨어 사진이므로 빠르게.

**Section 2.1 (Spacecraft potential)** — Important nuance: THEMIS's "scale factor" A and "offset" approach is a *Langmuir → spacecraft → plasma* potential chain that is not identical to FAST or Cluster recipes. Eq. (1) is the master formula.
**섹션 2.1**: 척도 인자와 오프셋 접근법이 핵심. 식 (1)이 핵심 공식.

**Section 2.2 (Energy-dependent efficiency)** — The discovery that ion sensors had ~40% extra sensitivity below 100 eV due to leakage fields is the most subtle technical lesson; see Fig. 11c.
**섹션 2.2**: 100 eV 이하 이온 감도 ~40% 상승의 원인이 출구 그리드 누설장이라는 발견이 가장 미묘한 기술적 교훈. Fig. 11c 확인.

**Section 2.3–2.4 (Dead time + relative anode)** — Fig. 12 demonstrates dead time correction with magnetosheath data: pre-correction Ni/Ne > 1.0–1.3, post-correction ~0.9. Fig. 13 shows the iterative anode-efficiency fit converging to <10% residual.
**섹션 2.3–2.4**: Fig. 12로 데드타임 보정의 효과(보정 전 1.0–1.3 → 보정 후 0.9). Fig. 13은 양극 효율 반복 적합으로 잔차 <10% 수렴.

**Section 2.5 (Cross-calibration)** — The heart of the paper. Magnetosheath pairs each spacecraft's ion ↔ electron sensors via Ni/Ne, then chains electron sensors across spacecraft. Note that the calibration is anchored to THC (June 28, 2007) and propagated.
**섹션 2.5**: 논문의 핵심. 자기초에서 같은 위성의 이온/전자 센서를 짝짓고, 다음 전자 센서끼리 위성 간 비교. 기준은 THC 2007-06-28.

**Section 2.6 (Absolute calibration)** — Wind/SWE comparison reveals THEMIS electron G was *underestimated by ~40%* (factor 1/0.7) pre-flight; this 40% factor is the single biggest correction.
**섹션 2.6**: Wind/SWE 비교로 발견된 ~40% 절대 보정. 가장 큰 보정 요소.

---

## 7. 현대적 의의 / Modern Significance

THEMIS plasma instrumentation set the template for subsequent multi-spacecraft missions: MMS Fast Plasma Investigation (FPI, Pollock et al. 2016) inherited the top-hat geometry but pushed cadence to 30 ms; ARTEMIS (the two THEMIS probes redirected to lunar orbit in 2009) continues to use the ESAs as space-weather/lunar-wake probes. The cross-calibration recipe — anchor one sensor, chain across spacecraft via magnetosheath Ni/Ne, then absolutely scale to upstream Wind/SWE — is now standard practice in MMS, Cluster reanalyses, and the upcoming HelioSwarm and MUSE missions.

THEMIS 플라즈마 측기는 후속 다위성 미션의 표준이 되었다. MMS의 FPI(Pollock et al. 2016)는 탑햇 기하를 이어받으면서 측정 주기를 30 ms까지 단축했고, ARTEMIS(2009년 달 궤도로 재배치된 두 THEMIS 위성)는 우주환경·달 후류 연구용 ESA를 계속 사용 중이다. "한 센서를 기준으로 정한 뒤 자기초 Ni/Ne로 위성 간 연결, 마지막에 Wind/SWE로 절대 스케일" 방식의 교차 교정 절차는 현재 MMS, Cluster 재해석, 그리고 HelioSwarm·MUSE 미션 등에서 표준 방법으로 자리 잡았다.

The paper's deeper lesson is that **multi-spacecraft science is impossible without identical instruments and cross-calibration plans budgeted from day one**. Cluster's heterogeneous plasma payloads required years of ad-hoc cross-calibration; THEMIS's identical sensors compressed this to a few months. Future missions (MMS, MAVEN, Solar Orbiter SWA) explicitly cite this lesson.
이 논문의 더 깊은 교훈은 **다위성 과학은 동일한 측기와 사전 계획된 교차 교정 없이는 불가능**하다는 점이다. Cluster의 이종 플라즈마 측기는 다년간의 즉흥 교정을 요했지만, THEMIS의 동일 센서는 이를 수개월로 단축했다. 후속 미션들(MMS, MAVEN, Solar Orbiter SWA)은 이 교훈을 명시적으로 인용한다.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)

### Q1. 왜 자기초가 교차 교정에 이상적인가? / Why is the magnetosheath ideal for cross-calibration?

**A**: 자기초 플라즈마는 (i) 밀도가 충분히 높아 통계적 잡음이 작고, (ii) 분포가 거의 등방·맥스웰적이라 모멘트 계산이 안정적이며, (iii) 이온/전자가 모두 분석기 에너지 창(수 eV–수 keV)에 들어오고, (iv) Φ_sc가 상대적으로 작음(5–6 V)이고, (v) 알파 입자 비율이 태양풍보다 작다. 자기권은 차가운 성분 누락 위험이, 태양풍은 좁은 빔 폭 때문에 24-bin 모드에서 이온 밀도가 과소평가되는 문제가 있다.
**A**: The magnetosheath plasma (i) has high enough density to reduce counting noise, (ii) is nearly isotropic Maxwellian so moment integrals converge, (iii) places both species inside the energy window, (iv) keeps Φ_sc small (5–6 V), and (v) has lower alpha contamination than solar wind. The magnetosphere risks missing cold populations; the solar wind underestimates ion density due to narrow beam width when sweep mode has only 24 angular bins.

### Q2. 누설장(leakage field) 보정이 왜 이온에만 ~40%인가? / Why is the leakage-field correction ~40% for ions only?

**A**: MCP 앞면이 −2 kV로 바이어스되어 분석기 출구 그리드 안쪽으로 전기장이 누설된다. 분석기 시뮬레이션이 출구 그리드를 "이상적 차폐"로 가정해 이 효과를 무시했었다. 저에너지 이온은 이 누설장에 가속되어 그리드 와이어를 피해 더 잘 통과하므로 100 eV 이하에서 감도가 ~45% 증가하고 ~180 eV 부근에서 e-folding 한다. 전자도 비슷한 효과가 있을 수 있으나 MCP 전압이 −450 V 정도로 낮고, 이차 전자 생성과 상쇄되어 알짜 효과는 측정적으로 평탄하다.
**A**: The MCP front face is biased at −2 kV, leaking field through the exit grid. Simulations had assumed the exit grid was an ideal screen. Low-energy ions get focused around grid wires by this leakage, raising effective transmission from ~90% toward ~100% below ~10 eV and increasing the analyzer geometric factor by ~30% at low energies with an e-folding scale of ~180 eV. Electrons see only a ~−450 V MCP bias and the leakage effect is roughly compensated by secondary-electron production from the hemisphere, making the net energy dependence flat.

### Q3. 절대 교정에서 왜 Wind/SWE를 표준 캔들로 쓰는가? / Why is Wind/SWE the absolute calibration standard?

**A**: THEMIS는 고주파 플라즈마 진동(plasma frequency) 측정 기기가 없어 밀도의 절대 표준을 자체적으로 갖지 못한다. Wind/SWE Faraday cup은 1995년부터 안정적으로 동작하며 알파 입자를 분리해 양성자 밀도를 직접 측정하기에, 태양풍에서 THEMIS가 Wind와 함께 측정되는 시간대를 골라 비교했다. 비교 결과 THEMIS 전자 G가 ~40% 과소평가되어 있었음(즉 실제 G는 1/0.7 ≈ 1.43배). 5번의 2개월 비교에서 일관된 ~10% 재현성.
**A**: THEMIS lacks a high-frequency plasma-wave receiver, so it cannot lock its density scale to the upper-hybrid line. Wind/SWE Faraday cups (Ogilvie et al. 1995) provide stable, alpha-corrected proton densities. Comparing five intervals over two months, THEMIS electron densities were ~30% low, implying G was *underestimated by ~40%* (factor 1/0.7 ≈ 1.43). Repeat measurements were consistent at the ~10% level.
