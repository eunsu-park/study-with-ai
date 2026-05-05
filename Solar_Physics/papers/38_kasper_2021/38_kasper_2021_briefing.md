---
title: "Parker Solar Probe Enters the Magnetically Dominated Solar Corona — Briefing"
authors: "J. C. Kasper et al."
year: 2021
journal: "Physical Review Letters 127, 255101"
doi: "10.1103/PhysRevLett.127.255101"
date: 2026-04-27
topic: Solar_Physics
tags: [PSP, Alfven_critical_surface, sub-Alfvenic, corona, plasma_beta, switchbacks]
---

# Pre-Reading Briefing / 사전 브리핑

## 1. Why This Paper Matters / 이 논문이 중요한 이유

**English.** This is the paper announcing the historic first crossing of the Alfvén critical surface by a human-made spacecraft. On 28 April 2021, during its eighth perihelion (E8), Parker Solar Probe (PSP) descended to ~19.7 R☉ and for ~5 hours flew through plasma that was magnetically dominated (β ≪ 1) and sub-Alfvénic (M_A < 1). For the first time, in situ measurements were taken from inside the solar corona, ending six decades of speculation that began with Parker's 1958 prediction of a supersonic wind.

**한국어.** 본 논문은 인공 우주선이 Alfvén 임계면(Alfvén critical surface)을 처음으로 통과했음을 발표한 역사적인 논문이다. 2021년 4월 28일 8차 근일점(E8) 동안 Parker Solar Probe(PSP)는 약 19.7 R☉까지 하강하여 약 5시간 동안 자기적으로 우세한(β ≪ 1) 그리고 sub-Alfvénic(M_A < 1) 플라즈마 안을 비행하였다. 이로써 태양 코로나 내부에서의 최초 in situ 측정이 이루어졌으며, Parker의 1958년 초음속 태양풍 예측 이후 60년에 걸친 추측이 종지부를 찍었다.

## 2. Core Claims / 핵심 주장

**English.**
1. Three sub-Alfvénic intervals (I1, I2, I3) were identified during E8, with I1 being the most robust (~5 h, M_A ≈ 0.79, 18.4–19.8 R☉).
2. Magnetic energy density exceeded both ion and electron pressure energy density (β < 1) in I1.
3. Magnetic field mapping (PFSS) ties the source of I1 to a slow flow above an expanding pseudostreamer.
4. The unusually low density (2–5× lower than typical scaling) is attributed to suppressed reconnection at the pseudostreamer base.
5. The turbulent power spectrum below the Alfvén surface shows a 1/f range with a clear break at f_sc ≈ 2×10⁻³ Hz and a -3/2 inertial range slope, with mild power enhancement at the break.

**한국어.**
1. E8에서 세 개의 sub-Alfvénic 구간(I1, I2, I3)이 식별되었으며 I1이 가장 견고함(~5 h, M_A ≈ 0.79, 18.4–19.8 R☉).
2. I1에서 자기 에너지 밀도가 이온·전자의 압력 에너지 밀도를 모두 초과하였다(β < 1).
3. 자기장 매핑(PFSS)은 I1의 원천을 확장하는 의사스트리머(pseudostreamer) 위의 느린 흐름과 연결시킨다.
4. 통상적인 스케일링보다 2–5배 낮은 밀도는 의사스트리머 바닥에서 자기 재결합 억제로 설명된다.
5. Alfvén 임계면 아래의 난류 전력 스펙트럼은 f_sc ≈ 2×10⁻³ Hz에서 깨지는 1/f 영역과 -3/2 관성 영역 기울기를 보이며, break 부근에 미약한 에너지 증가가 있다.

## 3. Prerequisites / 사전 지식

| Concept | Korean | Why needed |
|---------|--------|------------|
| Alfvén speed v_A = B/√(μ₀ρ) | Alfvén 속도 | Defines the critical surface |
| Plasma beta β = 2μ₀p/B² | 플라즈마 베타 | Magnetic vs thermal dominance |
| Alfvén Mach number M_A = v/v_A | Alfvén Mach 수 | M_A < 1 ⇒ sub-Alfvénic |
| Parker spiral & solar wind expansion | Parker 나선 | Background v(r), B(r) |
| PFSS model | PFSS 모델 | Maps PSP to photospheric source |
| Switchbacks | 스위치백 | Background phenomenon being explained |

## 4. Key Vocabulary / 핵심 어휘

- **Alfvén critical surface (r_A)** / Alfvén 임계면: locus where v_r = v_A.
- **Sub-Alfvénic flow** / sub-Alfvénic 흐름: M_A < 1, magnetically connected to Sun.
- **Pseudostreamer** / 의사스트리머: magnetic structure separating two coronal holes of same polarity (no current sheet).
- **Quasi-separatrix layer (QSL)** / 준-분리면: thin region of large field-line connectivity gradient.
- **HCS** / 태양권 전류층: heliospheric current sheet, polarity reversal boundary.
- **PFSS** / Potential Field Source Surface: potential-field model up to ~1.5–2 R☉.
- **SWEAP / FIELDS** / SWEAP·FIELDS: PSP's plasma & magnetic field instrument suites.

## 5. Q&A — Anticipated Questions / 예상 질문 및 답변

**Q1. What does "magnetically dominated" mean precisely? / "자기적으로 우세하다"의 정확한 의미?**
- En: β ≪ 1, i.e. magnetic pressure B²/(2μ₀) dominates plasma thermal pressure n k_B T, and additionally magnetic energy density exceeds bulk kinetic energy density (½ρv²), i.e. M_A < 1.
- 한: β ≪ 1, 즉 자기 압력 B²/(2μ₀)이 열 압력 n k_B T를 압도하며, 또한 자기 에너지 밀도가 운동 에너지 밀도(½ρv²)를 초과(M_A < 1)하는 상태.

**Q2. Why is r_A important for solar physics? / r_A가 태양물리에서 왜 중요한가?**
- En: It marks the boundary where torques can extract solar angular momentum, sets the mass-loss rate, and divides regions where magnetic forces vs. inertia dictate dynamics. It's also the natural site predicted for switchback formation and turbulent cascade onset.
- 한: 태양으로부터 각운동량을 추출할 수 있는 한계, 질량 손실률 결정, 그리고 자기력이 관성보다 우세한 영역과 그 반대 영역을 구분짓는 경계. 또한 스위치백 형성과 난류 캐스케이드 시작의 자연스러운 위치로 예측된다.

**Q3. Why was I1 (low density) so unusual? / I1의 저밀도가 왜 이례적인가?**
- En: Compared to the Wind-derived empirical scaling n_p ≈ 10⁴·⁸⁵ v_r⁻⁰·⁵⁴ at 1 AU, the I1 density is 2–5× lower. This combination of normal v_r but anomalously low n drives ρ down, raising v_A and reducing M_A below 1.
- 한: Wind 기반 경험식 n_p ≈ 10⁴·⁸⁵ v_r⁻⁰·⁵⁴ (1 AU)와 비교했을 때 I1의 밀도는 2–5배 낮다. 정상적인 v_r과 이례적으로 낮은 n의 조합이 ρ를 낮추어 v_A를 높이고 M_A를 1 미만으로 떨어뜨린다.

**Q4. Pseudostreamer connection — what's the physical picture? / 의사스트리머 연결의 물리적 그림?**
- En: PFSS mapping shows I1's footpoint sits at the boundary of a southern coronal hole extension, on a pseudostreamer/QSL. Reconnection at pseudostreamer bases is thought to inject mass; if suppressed, the wind there is starved of plasma → low ρ → high v_A → sub-Alfvénic.
- 한: PFSS 매핑은 I1의 발자취가 남쪽 코로나 홀 확장의 경계, 의사스트리머/QSL 위에 있음을 보인다. 의사스트리머 기저의 재결합은 질량 주입원으로 여겨지며, 이것이 억제되면 그 영역의 태양풍은 플라즈마가 빈약 → 낮은 ρ → 높은 v_A → sub-Alfvénic.

**Q5. What does the turbulence spectrum tell us? / 난류 스펙트럼이 의미하는 것?**
- En: The presence of a 1/f range that becomes clearer below r_A supports the view that 1/f is generated near the Sun (energy-containing range), and the slightly enhanced power at the break could be a parametric instability/inverse cascade signature predicted by Chandran (2018).
- 한: r_A 아래에서 1/f 영역이 더 뚜렷해진다는 것은 1/f 영역이 태양 근처에서 생성된다는 견해를 뒷받침하며, break 부근의 미약한 에너지 증가는 Chandran(2018)이 예측한 매개변수 불안정성/역 캐스케이드의 흔적일 수 있다.

**Q6. Why does I2 reach M_A=0.49 yet matter less? / I2의 M_A=0.49가 더 깊은데도 덜 중요한 이유?**
- En: I2 is short and adjacent to the HCS, so it could be a transient HCS-related dip rather than a steady sub-Alfvénic stream. I1, being far from HCS and ~5 h long, is unambiguously a steady stream.
- 한: I2는 짧고 HCS 근처에 있어 정상상태의 sub-Alfvénic 스트림이라기보다 HCS 관련 일시적 하강일 가능성이 있다. 반면 I1은 HCS에서 멀고 약 5시간 지속되어 명백한 정상 스트림이다.

## 6. Reading Roadmap / 읽기 가이드

**English.** PRL letters are dense. Suggested reading order:
1. Abstract → Fig. 1 (multi-panel overview): orient to the three intervals.
2. Table I + intro paragraphs on r_A and v_A.
3. "Observations of sub-Alfvénic solar wind" section paired with Fig. 1d-e (β and M_A panels).
4. Fig. 2 (turbulence spectrum) + Discussion paragraph on 1/f and -3/2.
5. Fig. 3 (PFSS mapping) + "Solar surface source" section.
6. Fig. 4 + comparison-with-prediction paragraph.

**한국어.** PRL 형식은 매우 압축적이다. 권장 읽기 순서:
1. 초록 → Fig. 1 다중 패널: 세 구간의 윤곽 파악.
2. Table I + r_A·v_A 정의 단락.
3. "Observations of sub-Alfvénic solar wind" 절 + Fig. 1d-e(β, M_A 패널).
4. Fig. 2 (난류 스펙트럼) + 1/f, -3/2에 대한 논의.
5. Fig. 3 (PFSS 매핑) + "Solar surface source" 절.
6. Fig. 4 + 예측과의 비교 단락.
