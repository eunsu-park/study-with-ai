---
title: "A New, Lower Value of Total Solar Irradiance: Evidence and Climate Significance"
authors: "Greg Kopp, Judith L. Lean"
year: 2011
journal: "Geophysical Research Letters"
doi: "10.1029/2010GL045777"
date_briefed: 2026-04-27
topic: Solar_Physics
tags: [TSI, SORCE, TIM, radiometry, solar_constant, climate]
---

# Pre-Reading Briefing / 사전 읽기 브리핑
## Kopp & Lean (2011) — A New, Lower Value of Total Solar Irradiance

---

## 1. Why This Paper Matters / 이 논문이 중요한 이유

**English.** For three decades, space-borne radiometers measuring Total Solar Irradiance (TSI) — the radiative flux from the Sun integrated over all wavelengths at 1 AU — disagreed with one another at the ~5 W/m² level. Different instruments (ACRIM I/II/III, ERBE, VIRGO) reported absolute values clustered around 1365–1366 W/m², while the SORCE/TIM (Total Irradiance Monitor), launched in 2003, persistently reported a much lower value near 1361 W/m². The discrepancy mattered: TSI is the single largest external energy input driving Earth's climate system, and a 4–5 W/m² absolute uncertainty translates into systematic offsets in climate-model boundary conditions and in any inferred long-term solar trend. Kopp & Lean (2011) marshal evidence — laboratory cryogenic-radiometer comparisons, the new TRF (TSI Radiometer Facility), and instrument-design diagnostics — to argue that the TIM value (~1360.8 W/m²) is the correct one and that older instruments suffered from internally scattered light biasing them high. They then translate the new absolute value into climate-relevant quantities.

**한국어.** 30년 동안 총 태양 복사 조도(Total Solar Irradiance, TSI) — 1 AU에서 모든 파장에 걸쳐 적분된 태양 복사속 — 를 측정한 우주 복사계들은 서로 약 5 W/m² 수준에서 일치하지 않았습니다. 서로 다른 기기들(ACRIM I/II/III, ERBE, VIRGO)은 1365–1366 W/m² 부근의 절대값을 보고했지만, 2003년에 발사된 SORCE/TIM(Total Irradiance Monitor)은 약 1361 W/m²의 훨씬 낮은 값을 지속적으로 보고했습니다. 이 불일치는 중요했습니다: TSI는 지구 기후 시스템을 구동하는 가장 큰 외부 에너지 입력이며, 4–5 W/m²의 절대 불확실성은 기후 모델 경계 조건 및 장기 태양 추세 추론에 체계적 편차를 만듭니다. Kopp & Lean(2011)은 실험실 극저온 복사계 비교, 새로운 TRF(TSI Radiometer Facility), 기기 설계 진단을 통해 TIM 값(~1360.8 W/m²)이 정확하며 이전 기기들은 내부 산란광에 의해 높게 편향되었음을 주장합니다. 그런 다음 이 새로운 절대값을 기후 관련 양으로 환산합니다.

---

## 2. Historical Context / 역사적 맥락

**English.** The "solar constant" was estimated from the ground (Langley, Abbot) at ~1370–1380 W/m² before the satellite era. Beginning with Nimbus-7/HF (1978) and ACRIM-I on SMM (1980), continuous space-based TSI monitoring was established, but each instrument carried independent absolute calibration referenced to ground tests. Cross-calibration during overlapping mission lifetimes was used to splice composite records (PMOD, ACRIM, IRMB), but the absolute level remained unsettled. SORCE/TIM (2003–) introduced a phase-sensitive detection scheme with a precision aperture in front of the instrument view-limiting baffles — opposite to legacy designs — significantly reducing scattered-light contamination. The 4–5 W/m² gap persisted until the TRF (deployed at LASP, Boulder) finally allowed end-to-end vacuum comparison of full instruments against an absolute cryogenic radiometer under solar-like illumination.

**한국어.** "태양 상수"는 위성 시대 이전에 지상에서(Langley, Abbot) 약 1370–1380 W/m²로 추정되었습니다. Nimbus-7/HF(1978)와 SMM의 ACRIM-I(1980)로 시작하여 우주 기반 TSI 연속 모니터링이 확립되었지만, 각 기기는 지상 시험에 참조된 독립적인 절대 보정값을 가지고 있었습니다. 중첩 임무 기간 동안 교차 보정을 통해 합성 기록(PMOD, ACRIM, IRMB)을 만들었지만, 절대 수준은 미해결 상태로 남았습니다. SORCE/TIM(2003–)은 시야 제한 배플 앞에 정밀 조리개를 배치하는 위상 민감 검출 방식 — 기존 설계와 반대 — 을 도입하여 산란광 오염을 크게 줄였습니다. LASP(Boulder)에 구축된 TRF가 마침내 태양 유사 조명 하에서 절대 극저온 복사계와 전체 기기를 진공에서 종단 간 비교할 수 있게 되기까지 4–5 W/m²의 격차는 지속되었습니다.

---

## 3. Prerequisites / 사전 지식

**English.**
- **Radiometry basics**: irradiance (W/m²), inverse-square law, the "solar constant" definition at 1 AU.
- **Active cavity radiometers**: electrical-substitution principle, cone receiver, blackened cavity, thermal balance against a heater.
- **Aperture geometry**: view-limiting vs. precision aperture; why placing the precision aperture in front matters for scattered-light rejection.
- **PMOD vs. ACRIM composites**: two competing TSI time-series reconstructions over solar cycles 21–24.
- **Climate forcing concept**: radiative forcing ΔF (W/m²), Earth's disk-averaged absorption (1−A)/4 factor, equilibrium climate sensitivity λ.

**한국어.**
- **복사 측정 기초**: irradiance(W/m²), 역제곱 법칙, 1 AU에서의 "태양 상수" 정의.
- **능동 공동 복사계(active cavity radiometer)**: 전기적 치환 원리, 원뿔형 수신기, 흑화 공동, 히터 대비 열 평형.
- **조리개 기하학**: view-limiting vs. precision aperture; 산란광 제거를 위해 정밀 조리개를 앞쪽에 배치하는 것이 왜 중요한가.
- **PMOD vs. ACRIM 합성**: 태양주기 21–24에 걸친 두 경쟁하는 TSI 시계열 재구성.
- **기후 강제력 개념**: 복사 강제력 ΔF(W/m²), 지구 원반 평균 흡수의 (1−A)/4 인수, 평형 기후 민감도 λ.

---

## 4. Key Vocabulary / 핵심 어휘

| Term / 용어 | Meaning / 의미 |
|---|---|
| **TSI** | Total Solar Irradiance: spatially integrated, all-wavelength solar radiative flux at 1 AU / 1 AU에서 공간 적분된 전 파장 태양 복사속 |
| **TIM** | Total Irradiance Monitor (SORCE instrument) / SORCE 탑재 복사계 |
| **TRF** | TSI Radiometer Facility — vacuum end-to-end comparison facility at LASP / LASP의 진공 종단 비교 시설 |
| **Cryogenic radiometer** | NIST-traceable absolute reference, ~0.01% accuracy / NIST 추적성 절대 기준 |
| **Scattered light** | Diffuse reflection inside the instrument adding spurious signal / 기기 내부 확산 반사로 인한 신호 |
| **Precision aperture** | Tightly machined opening defining the entering solar beam area / 입사 태양 빔 면적을 정의하는 정밀 조리개 |
| **PMOD composite** | Fröhlich's TSI composite (default for IPCC) / Fröhlich의 TSI 합성 |
| **ACRIM composite** | Willson's alternative composite (gap-bridging differs) / Willson의 대안적 합성 |

---

## 5. Reading Questions / 읽기 질문

**English.**
1. Why does scattered light bias legacy radiometers HIGH rather than low?
2. What is the new "best" TSI value at solar minimum, and what is its 1-σ uncertainty?
3. How does the lower TSI change Earth's effective temperature T_eff?
4. Does this paper change estimates of the *Maunder Minimum* solar forcing? Why or why not?
5. What does the TRF demonstrate about each historical instrument when retro-tested?

**한국어.**
1. 왜 산란광은 기존 복사계를 낮은 쪽이 아니라 높은 쪽으로 편향시키는가?
2. 태양 극소기에서 새로운 "최적" TSI 값과 1-σ 불확실성은 무엇인가?
3. 더 낮은 TSI가 지구 유효 온도 T_eff에 어떻게 영향을 미치는가?
4. 이 논문이 *Maunder Minimum* 태양 강제력 추정치를 변경하는가? 왜 그런가/그렇지 않은가?
5. TRF는 각 역사적 기기에 대한 사후 시험에서 무엇을 보여주는가?

---

## 6. Expected Takeaways / 예상 핵심 결론

**English.** After reading you should be able to (i) cite 1360.8 ± 0.5 W/m² as the modern solar-minimum TSI, (ii) explain the *front-aperture* design advantage of TIM, (iii) describe how the offset is a constant absolute shift (not a trend) and therefore does NOT change cycle-to-cycle climate forcing variability, and (iv) compute the new disk-averaged forcing 1360.8/4 ≈ 340.2 W/m² and the corresponding shift in Earth's equilibrium temperature.

**한국어.** 읽은 후 다음을 할 수 있어야 합니다: (i) 현대 태양 극소기 TSI로 1360.8 ± 0.5 W/m²를 인용, (ii) TIM의 *전면 조리개(front-aperture)* 설계 이점 설명, (iii) 오프셋이 추세가 아닌 일정한 절대 이동이므로 주기 간 기후 강제력 변동성을 변경하지 *않음*을 기술, (iv) 새로운 원반 평균 강제력 1360.8/4 ≈ 340.2 W/m² 및 지구 평형 온도의 해당 이동을 계산.
