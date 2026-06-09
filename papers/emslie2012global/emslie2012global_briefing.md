---
title: "Pre-reading Briefing — Emslie et al. (2012): Global Energetics of Thirty-Eight Large Solar Eruptive Events"
date: 2026-04-27
topic: Solar_Physics
paper: 34_emslie_2012
tags: [solar-flare, energetics, CME, SEP, RHESSI, GOES, magnetic-energy]
---

# Pre-reading Briefing / 사전 학습 브리핑
## Emslie, Dennis, Shih, Chamberlin, Mewaldt, Moore, Share, Vourlidas, Welsch (2012, ApJ 759, 71)

---

## 1. Why This Paper Matters / 이 논문이 중요한 이유

**English.** Solar eruptive events (SEEs) are the most energetic phenomena in the heliosphere, releasing up to 10^32 erg in tens of seconds to minutes. While prior work had analyzed individual flares (e.g., Emslie et al. 2004, 2005 on the 2002 April 21 and July 23 events), no statistical study existed quantifying *how* this enormous energy budget partitions across thermal plasma, accelerated electrons, accelerated ions, CME bulk kinetic energy, gravitational potential, solar energetic particles (SEPs), and bolometric radiation. Emslie+ 2012 delivered the first such ensemble study — 38 large RHESSI-era events from 2002–2006 — establishing the canonical energy-partition framework still cited in flare/CME modeling.

**한국어.** 태양 폭발 현상(SEE)은 태양권에서 가장 에너지가 큰 현상으로, 수십 초에서 수 분 사이에 최대 10^32 erg를 방출합니다. Emslie et al. (2004, 2005)이 2002년 4월 21일과 7월 23일 사건을 개별적으로 분석한 바 있지만, 이 거대한 에너지 예산이 열적 플라즈마, 가속 전자, 가속 이온, CME 운동에너지, 중력 위치에너지, 태양고에너지입자(SEP), 그리고 복사 등으로 *어떻게 분배되는지* 통계적으로 정량화한 연구는 없었습니다. Emslie+ 2012는 RHESSI 시대(2002–2006)의 38개 대형 사건을 종합 분석하여, 오늘날까지 플레어/CME 모델링에 인용되는 표준 에너지 분배 프레임워크를 확립한 최초의 통계 연구입니다.

---

## 2. Historical Context / 역사적 맥락

**English.** This paper is the culmination of an effort started at the 2003 Taos ACE/RHESSI/WIND workshop (Emslie+ 2004, 2005) and extended at the 2008 Napa "Solar Activity during Cycle 24 Onset" meeting. It builds on Mewaldt et al. (2008) — the first SEP/CME energy-ratio statistical study — but expands to all eight observable energy components. The paper appeared just before SDO/HMI vector magnetograms became routinely used for NLFFF magnetic-energy estimates, which Aschwanden+ later (2014, 2017) revisited the same dataset with.

**한국어.** 이 논문은 2003년 Taos ACE/RHESSI/WIND 워크샵(Emslie+ 2004, 2005)에서 시작되어 2008년 Napa "Solar Cycle 24 시작기 활동" 회의에서 확장된 연구의 결정체입니다. SEP/CME 에너지 비율 통계 연구의 효시인 Mewaldt et al. (2008)을 기반으로, 관측 가능한 8개 에너지 성분 모두로 확장한 것입니다. 이 논문은 SDO/HMI 벡터 자기장 관측이 NLFFF 자기에너지 추정에 본격 사용되기 직전에 발표되었으며, 이후 Aschwanden+ (2014, 2017)가 동일 데이터셋을 재분석하게 됩니다.

---

## 3. Key Concepts to Know / 알아두어야 할 핵심 개념

### 3.1 Energy Components / 에너지 성분

| Component / 성분 | Symbol | What it is / 정의 | Instrument / 관측기 |
|---|---|---|---|
| GOES 1–8 Å radiated | E_SXR | Soft X-ray narrow-band radiation / 연 X선 협대역 복사 | GOES XRS |
| Total SXR plasma radiated | E_T-rad | Bolometric loss from hot plasma / 고온 플라즈마의 전 파장 복사 손실 | GOES + CHIANTI |
| Bolometric radiated | E_bol | Total irradiance increase / 전 파장 적분 복사 | SORCE/TIM, FISM |
| Peak thermal energy | U_th | 3 n_e k T_0 f V_ap / 최고시점 열에너지 | RHESSI imaging spectroscopy |
| Flare-accelerated electrons | U_e | >20 keV nonthermal electrons / >20 keV 비열적 전자 | RHESSI HXR |
| Flare-accelerated ions | U_i | >1 MeV ions / >1 MeV 이온 | RHESSI 2.223 MeV n-capture line |
| CME kinetic | U_K | (1/2) m V^2 (Sun rest frame) / 태양 정지 좌표계 | LASCO |
| CME potential | U_Φ | GM_⊙ m [R_⊙^-1 - r^-1] / 중력위치에너지 | LASCO |
| SEP energy | U_SEP | Interplanetary energetic particles / 행성간 SEP | ACE, GOES, SAMPEX, STEREO |
| Magnetic free energy | E_mag | Nonpotential B-field energy / 비포텐셜 자기에너지 | Active region area + B |

### 3.2 Key Equations / 핵심 방정식

**Thermal energy / 열에너지:**
$$U_{\text{th}} = 3 n_e k T_0 f V_{\text{ap}} \simeq 4.14 \times 10^{-16} T_0 \sqrt{EM \times f V_{\text{ap}}} \;\text{erg}$$

where f ≈ 1 (filling factor), V_ap is the apparent SXR volume from RHESSI 3σ-clean imaging.
여기서 f ≈ 1 (충전인자), V_ap는 RHESSI 3σ-clean 영상에서 구한 겉보기 SXR 부피.

**Nonthermal electron spectrum (broken power-law) / 비열적 전자 스펙트럼 (꺾인 거듭제곱):**
$$F_0(E_0) = A (E_0/E_p)^{-\delta_1} \;\;\text{for}\;\; E_{\min} \le E_0 < E_b$$
$$F_0(E_0) = A (E_0/E_p)^{-\delta_2}(E_b/E_p)^{\delta_2-\delta_1} \;\;\text{for}\;\; E_b \le E_0 < E_{\max}$$

E_min is the dominant uncertainty — a small change shifts U_e by an order of magnitude.
E_min이 주된 불확실성으로, 작은 변화에도 U_e가 한 자릿수 변할 수 있습니다.

### 3.3 Methodological Choices / 방법론적 선택

**English.**
- **GOES SXR:** Background subtraction → integrate flux from start time to 10% of peak. IDL `rad_loss` in SSW with CHIANTI.
- **Bolometric:** Direct TIM measurement for 5 events; otherwise FISM (1–1900 Å) × 2.42 ± 0.31 conversion factor.
- **Peak thermal:** RHESSI OSPEX isothermal+nonthermal forward fit, V_ap from RHESSI 3σ-clean (Dennis & Pernak 2009), f = 1.
- **Electrons:** OSPEX `thick2` thick-target bremsstrahlung, fit broken power-law, take *largest* E_min giving χ²_red ≃ 1 → these are *lower limits*.
- **Ions:** RHESSI 2.223 MeV neutron-capture line fluence → proton energy >30 MeV → extrapolate to >1 MeV with δ = 4. Total ion energy = 3 × proton energy (accounting for α and heavy ions).
- **CME:** LASCO Thomson-scattering excess mass (assumes plane-of-sky, 90% H + 10% He). U_K = (1/2) m V², U_Φ = GM_⊙ m[R_⊙^-1 − r^-1]. Solar-wind frame: subtract 400 km/s from V.
- **SEPs:** Spectra from ULEIS, EPAM, PET, EPS, SIS, LET, HET, EPHIN — fit Band function or Ellison-Ramaty exponential cutoff.

**한국어.**
- **GOES SXR:** 배경 제거 → 시작 시각부터 피크의 10%까지 플럭스 적분. SSW의 IDL `rad_loss` + CHIANTI 사용.
- **볼로메트릭:** 5개 사건은 TIM 직접 측정; 나머지는 FISM (1–1900 Å) × 2.42 ± 0.31 변환 계수.
- **피크 열에너지:** RHESSI OSPEX 등온+비열 전향적합, V_ap는 RHESSI 3σ-clean (Dennis & Pernak 2009), f = 1.
- **전자:** OSPEX `thick2` 두꺼운 표적 제동복사, 꺾인 거듭제곱 적합, χ²_red ≃ 1을 만족하는 *최대* E_min 사용 → *하한값*.
- **이온:** RHESSI 2.223 MeV 중성자 포획선 플루언스 → 30 MeV 이상 양성자 에너지 → δ = 4로 1 MeV까지 외삽. 총 이온에너지 = 양성자 에너지 × 3 (α 및 무거운 핵 고려).
- **CME:** LASCO 톰슨산란 초과 질량 (천구면 가정, 90% H + 10% He). U_K = (1/2) m V², U_Φ = GM_⊙ m[R_⊙^-1 − r^-1]. 태양풍 좌표계: V에서 400 km/s 차감.
- **SEP:** ULEIS, EPAM, PET, EPS, SIS, LET, HET, EPHIN 스펙트럼 → Band 함수 또는 Ellison-Ramaty 지수 컷오프 적합.

---

## 4. Major Conclusions (Preview) / 주요 결론 (미리보기)

**English.**
1. **U_T-rad ≈ 10 × U_th^peak** — radiation from SXR plasma exceeds peak thermal content by ~order of magnitude (continuous reheating).
2. **U_e + U_i ≳ E_bol** — energy in flare-accelerated particles is sufficient to power total radiation across all wavelengths.
3. **U_e ≈ U_i** — electron and ion energies are comparable (within factors of a few).
4. **U_SEP ≈ few % × U_K (solar-wind frame)** — SEPs typically a few percent of CME kinetic energy.
5. **E_mag ≳ U_K + U_e + U_i + U_th** — available free magnetic energy is sufficient to power CME, particles, and thermal plasma.

**한국어.**
1. **U_T-rad ≈ 10 × U_th^peak** — SXR 플라즈마의 복사가 피크 열량을 약 한 자릿수 초과 (지속적 재가열).
2. **U_e + U_i ≳ E_bol** — 가속 입자 에너지가 전 파장 총 복사를 공급하기에 충분.
3. **U_e ≈ U_i** — 전자와 이온 에너지가 비슷 (수배 이내).
4. **U_SEP ≈ 수 % × U_K (태양풍 좌표계)** — SEP는 CME 운동에너지의 수 %.
5. **E_mag ≳ U_K + U_e + U_i + U_th** — 가용 자유 자기에너지가 CME, 입자, 열 플라즈마 모두를 공급하기에 충분.

---

## 5. Vocabulary / 핵심 어휘

| Term / 용어 | Meaning / 의미 |
|---|---|
| SEE | Solar Eruptive Event = flare + CME / 플레어 + CME 동반 사건 |
| Bolometric | Integrated over all wavelengths / 전 파장 적분 |
| RHESSI | Ramaty High Energy Solar Spectroscopic Imager / 라마티 고에너지 분광영상기 |
| OSPEX | Object SPectral EXecutive (SSW package) / SSW 분광 적합 패키지 |
| Thick-target | Bremsstrahlung where electrons fully stop / 전자가 완전히 멈추는 제동복사 |
| FISM | Flare Irradiance Spectral Model / 플레어 복사 스펙트럼 모형 |
| TIM | Total Irradiance Monitor (SORCE) / 총 복사 모니터 |
| Filling factor f | Fraction of V_ap actually emitting / 실제 복사하는 부피 비율 |
| 2.223 MeV line | Neutron-capture deuterium-formation line / 중성자 포획 중수소 형성선 |
| Band function | Double power-law w/ exponential roll / 이중 거듭제곱 + 지수 롤오프 |

---

## 6. Reading Strategy / 읽기 전략

**English.** Section 2 (methods, 11 subsections) is the most important — it defines how each component is measured. Skim Section 2 carefully on first pass to understand instrument-specific assumptions (especially E_min for electrons, plane-of-sky for CMEs, spectral index extrapolation for ions). Section 3 (scatter plots) is where the *physics* lives — focus on which components correlate. Section 4 summarizes the energy-budget picture; Section 5 discusses outliers (Events #5 behind-limb, #12 INTEGRAL crosscheck).

**한국어.** Section 2 (방법론, 11개 소절)이 가장 중요합니다 — 각 성분을 어떻게 측정했는지 정의합니다. 첫 통독 시 Section 2를 주의 깊게 읽어 측정기별 가정을 이해하세요 (특히 전자의 E_min, CME의 천구면 가정, 이온의 스펙트럼 지수 외삽). Section 3 (산점도)이 *물리학*이 드러나는 부분 — 어떤 성분들이 상관관계를 보이는지에 주목하세요. Section 4는 에너지 예산 그림을 요약하며, Section 5는 이상치(#5 배면 사건, #12 INTEGRAL 교차검증)를 다룹니다.

---

## 7. Pre-reading Q&A / 사전 학습 질의응답

**Q1. Why are U_T-rad and U_th not summable?**
**A.** U_T-rad is the *cumulative* radiation lost by the SXR plasma over the entire flare. U_th is the *instantaneous peak* thermal content. Adding them double-counts the same energy reservoir — radiation is the *output* of the thermal reservoir.

**Q1. U_T-rad와 U_th는 왜 합산이 안 되나요?**
**A.** U_T-rad는 SXR 플라즈마가 플레어 전 기간 동안 *누적* 방출한 복사량이며, U_th는 *순간 최댓값*입니다. 둘을 더하면 동일한 에너지 저장소를 이중계산하게 됩니다 — 복사는 열저장소의 *출력*입니다.

**Q2. Why is E_min so critical for electron energy?**
**A.** Because electron spectra are steep (δ₁ ≳ 4), most of the energy is in low-energy electrons. Lowering E_min from 30 keV to 10 keV can multiply U_e by an order of magnitude. The thermal continuum hides the low-energy cutoff in the photon spectrum, so E_min is observationally bounded only from above.

**Q2. 왜 E_min이 전자 에너지에 그렇게 결정적인가요?**
**A.** 전자 스펙트럼이 가팔라(δ₁ ≳ 4) 대부분 에너지가 저에너지 전자에 있기 때문입니다. E_min을 30 keV에서 10 keV로 낮추면 U_e가 한 자릿수 증가할 수 있습니다. 열 연속복사가 광자 스펙트럼의 저에너지 컷오프를 가리므로, E_min은 관측상 위쪽 경계만 정해집니다.

**Q3. Why subtract 400 km/s from CME speed for SEP comparison?**
**A.** SEPs are accelerated by the CME-driven shock relative to the *ambient solar wind* (~400 km/s). The kinetic energy *available* for shock acceleration is in the solar-wind rest frame, not the Sun rest frame. This typically reduces U_K by 20–40%.

**Q3. SEP 비교 시 왜 CME 속도에서 400 km/s를 빼나요?**
**A.** SEP는 *주변 태양풍*(~400 km/s) 대비 CME 충격파에 의해 가속되기 때문입니다. 충격파 가속에 *가용한* 운동에너지는 태양 정지 좌표계가 아닌 태양풍 정지 좌표계에 있습니다. 일반적으로 U_K가 20–40% 감소합니다.

**Q4. Why is f = 1 (filling factor)?**
**A.** Guo et al. (2012) used HXR imaging spectroscopy on 22 events to derive a logarithmic mean f = 0.20 ×/÷ 3.9. This is consistent with f = 1 within the (logarithmic) uncertainty, and Emslie+ 2004 also adopted f = 1 for consistency. Lower f reduces U_th by √f.

**Q4. 충전인자를 왜 1로 두었나요?**
**A.** Guo et al. (2012)이 22개 사건의 HXR 분광영상에서 로그 평균 f = 0.20 ×/÷ 3.9를 도출했습니다. 이는 (로그) 불확실성 범위 내에서 f = 1과 일관되며, Emslie+ 2004도 일관성을 위해 f = 1을 채택했습니다. f가 작아지면 U_th가 √f만큼 감소합니다.

**Q5. What is the dominant uncertainty in CME energy?**
**A.** The plane-of-sky assumption underestimates mass; for CMEs ≲40° from sky plane, mass is underestimated by a factor of 2; for far-from-sky-plane events, U_K can be 8× larger. Hence quoted U_K values are *lower bounds*.

**Q5. CME 에너지의 주된 불확실성은?**
**A.** 천구면 가정이 질량을 과소평가합니다 — 천구면에서 ≲40° 떨어진 CME는 질량이 약 2배 과소평가되며, 천구면에서 멀리 떨어진 사건은 U_K가 8배까지 클 수 있습니다. 따라서 인용된 U_K는 *하한값*입니다.

---

## 8. Connections / 연결 고리

| Paper / 논문 | Connection / 관련성 |
|---|---|
| Emslie et al. (2004, 2005) | First case studies (Events #2, #6) / 첫 사례 연구 (사건 #2, #6) |
| Mewaldt et al. (2008) | First SEP/CME ratio statistics / 최초 SEP/CME 비율 통계 |
| Aschwanden et al. (2014, 2017) | NLFFF revisit of same dataset / 동일 데이터 NLFFF 재분석 |
| Su et al. (2011) | Second-phase radiation contribution / 후속 단계 복사 기여 |
| Woods et al. (2006, 2011) | TIM bolometric measurements / TIM 볼로메트릭 측정 |
| Vourlidas et al. (2010, 2011) | LASCO CME mass procedure / LASCO CME 질량 절차 |

---

## References / 참고문헌
- Emslie et al., "Global Energetics of Thirty-Eight Large Solar Eruptive Events", ApJ 759, 71 (2012). DOI: 10.1088/0004-637X/759/1/71
- Emslie et al., "Energy Partition in Two Solar Flare/CME Events", JGR 109, A10104 (2004).
- Emslie et al., "Refined Analysis of the 2002 July 23 Flare", JGR 110, A11103 (2005).
- Mewaldt et al., "How Efficient are Coronal Mass Ejections at Accelerating Solar Energetic Particles?", AIP Conf. Proc. 1039, 111 (2008).
- Vourlidas et al., "Comprehensive Analysis of Coronal Mass Ejection Mass and Energy Properties", ApJ 722, 1522 (2010).
- Woods et al., "Solar Irradiance Reference Spectra (SIRS) for the 2008 Whole Heliosphere Interval", GRL 36, L01101 (2006).
- Chamberlin et al., "Flare Irradiance Spectral Model (FISM)", Space Weather 5, S07005 (2007); 6, S05001 (2008).
