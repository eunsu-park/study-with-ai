---
title: "Pre-Reading Briefing: The FIP and Inverse FIP Effects in Solar and Stellar Coronae"
paper_id: "41"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# The FIP and Inverse FIP Effects in Solar and Stellar Coronae: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: J. Martin Laming, "The FIP and Inverse FIP Effects in Solar and Stellar Coronae", *Living Reviews in Solar Physics*, Vol. 12, No. 2, 2015. DOI: 10.1007/lrsp-2015-2
**Author(s)**: J. Martin Laming (Naval Research Laboratory, Code 7684)
**Year**: 2015

---

## 1. 핵심 기여 / Core Contribution

이 리뷰 논문은 태양과 별의 코로나에서 관측되는 원소 존재비 이상 현상(abundance anomalies), 즉 First Ionization Potential (FIP) effect와 Inverse FIP effect를 체계적으로 정리하고, 이들을 **ponderomotive force**(크로모스피어를 통과하거나 반사되는 MHD 파동에 의한 2차 비선형 힘)로 통일적으로 설명하는 모델을 제시한다. 저 FIP 원소(Mg, Si, Fe 등 FIP < 10 eV)는 태양 코로나에서 광구 대비 약 3-4배 증가하며, 고 FIP 원소(O, Ne, Ar)는 상대적으로 감소한다. Ponderomotive force는 이온에만 작용하므로(중성자는 무영향) 이온-중성자 분리를 자연스럽게 유도한다.

This review paper systematically consolidates observations of elemental abundance anomalies in solar and stellar coronae — the First Ionization Potential (FIP) effect and Inverse FIP effect — and presents a unifying model based on the **ponderomotive force** (a time-averaged nonlinear force arising from MHD waves propagating through, or reflecting from, the chromosphere). Low-FIP elements (Mg, Si, Fe with FIP < 10 eV) are enhanced by a factor of ~3-4 in the solar corona relative to photospheric values, while high-FIP elements (O, Ne, Ar) are not or slightly depleted. Because the ponderomotive force acts only on ions (not on neutrals), it produces a natural ion-neutral fractionation mechanism that intimately connects coronal abundances to coronal heating physics.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

FIP 효과는 Pottasch (1963)가 초기 UV/X-ray 분광학 시대에 Mg, Si, Fe가 광구보다 저 코로나에서 훨씬 풍부함을 발견하면서 처음 제기되었다. 1980년대 중반 Meyer (1985a,b)의 영향력 있는 리뷰로 본격적인 주목을 받았고, 2000년대 이후 SOHO, Hinode, ACE, Ulysses 관측으로 태양풍과 코로나 전체에 걸친 상세한 패턴이 드러났다. 동시에 EUVE, Chandra, XMM-Newton으로 항성 코로나 존재비가 측정되면서, 후기형(M dwarf) 별에서는 반대 부호의 Inverse FIP 효과가 발견되어 통일적 모델이 요구되었다.

The FIP effect was first noted by Pottasch (1963) during the early era of solar UV/X-ray spectroscopy, who found Mg, Si, Fe substantially enriched in the low corona relative to the photosphere. The influential reviews of Meyer (1985a,b) crystallized community interest in the mid-1980s. From the 2000s, SOHO, Hinode, ACE and Ulysses yielded detailed patterns across the corona and solar wind. Simultaneously, EUVE, Chandra and XMM-Newton began measuring stellar coronal abundances; the discovery of Inverse FIP in M dwarfs (Wood & Linsky 2010) demanded a unifying physical model.

### 타임라인 / Timeline

```
1963: Pottasch — first hint of Mg/Si/Fe enrichment in corona
1985: Meyer — influential FIP reviews, "FIP effect" terminology
1992-99: Ulysses — fast vs slow wind FIP difference
1994: Antiochos — thermoelectric driving model
1997-2000: SOHO era — coronal spectroscopy of FIP
1999: Schwadron et al. — ion-cyclotron wave heating model
2004: Laming — ponderomotive force model (first paper)
2008: Avrett & Loeser — modern empirical chromosphere
2010: Wood & Linsky — Inverse FIP trend with spectral type
2012: Laming — updated ponderomotive model, mode conversion
2015: Laming — this comprehensive review
```

---

## 3. 필요한 배경 지식 / Prerequisites

**수학 / Mathematics**:
- 벡터 미적분, 편미분 방정식, 라그랑지안 역학 / Vector calculus, PDEs, Lagrangian mechanics
- WKB 근사 / WKB approximation for waves in inhomogeneous media
- 플라즈마 파동 분산 관계 / Plasma wave dispersion relations

**물리 / Physics**:
- 자기유체역학 (MHD): Alfvén waves, fast/slow magnetosonic waves
- Chromospheric physics: temperature structure, ionization balance (Saha equation, charge exchange)
- Plasma-β = 8πnkT/B²; mode conversion at β ≈ 1 layer
- Coronal heating theories: nanoflares, wave heating

**천문 / Astronomy**:
- 태양 크로모스피어/코로나 구조, solar wind types (fast vs slow)
- 분광학으로 존재비 측정 (emission measure, line ratios)
- Stellar types (F, G, K, M dwarfs), X-ray luminosity diagnostics

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| FIP (First Ionization Potential) | 원자를 1가 이온으로 만드는 에너지. 저 FIP(<10 eV): Fe(7.9), Mg(7.6), Si(8.2). 고 FIP: O(13.6), Ne(21.6), Ar(15.8) / Energy to singly ionize an atom. Low-FIP (<10 eV) vs high-FIP dichotomy. |
| Ponderomotive Force | 진동 전자기장 내 이온에 작용하는 시간 평균 비선형 힘 F = (q²/4mω²)∇⟨δE²⟩. 이온 질량과 부호에 독립적 / Time-averaged nonlinear force on ions in oscillating EM fields; mass-independent for ω ≪ Ω_i |
| FIP Fractionation Factor f_Z | 코로나에서 원소 Z가 광구 대비 증가한 비율 (≡ (X/H)_cor / (X/H)_phot) / Ratio of coronal to photospheric abundance ratio |
| Inverse FIP Effect | 후기형 별(M dwarfs)에서 관측되는 반대 효과: 저 FIP 원소가 오히려 감소 / Opposite effect in M dwarfs: low-FIP elements depleted relative to high-FIP |
| Alfvén Wave | 자기장에 평행하게 전파하는 횡파, 속도 V_A = B/√(4πρ) / Transverse MHD wave propagating along B with speed V_A = B/√(4πρ) |
| Mode Conversion | β ≈ 1 층에서 음향파(acoustic) ↔ fast-mode 변환 / Acoustic ↔ fast-mode conversion at plasma β=1 layer |
| Chromospheric Evaporation | 코로나에서 하방으로 전달된 열에너지가 크로모스피어 물질을 상승시키는 현상 / Upward flow of chromospheric material driven by downward heat conduction from the corona |
| Charge Exchange | 양성자와 중성 원자 사이의 전자 교환 반응 (H⁺ + O → H + O⁺). O의 이온화 상태를 H에 결부시킴 / Proton-neutral electron exchange; locks O ionization to H |
| Plasma β | 열압력/자기압력 비율, β = 8πp/B². Mode conversion은 β=1에서 발생 / Ratio of thermal to magnetic pressure; mode conversion at β=1 |
| Slow Solar Wind | 저속 태양풍 (<500 km/s), 닫힌 코로나 구조 기원, 강한 FIP 효과 / Low-speed wind from closed coronal structures, strong FIP effect |
| Fast Solar Wind | 고속 태양풍 (>600 km/s), 코로나홀 기원, 약한 FIP 효과 / High-speed wind from coronal holes, weak FIP effect |
| Coronal Hole | 열린 자기장 영역, 고속 태양풍의 근원 / Open magnetic field region, source of fast wind |
| WKB Approximation | 파장 ≪ 배경 스케일 길이 가정. 크로모스피어에서는 성립하지 않음 / Assumes wavelength ≪ background scale; fails in chromosphere |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Ponderomotive force on ion (uniform B):**
$$F_i = \frac{q_i^2}{4 m_i (\Omega_i^2 - \omega^2)} \frac{d \delta E_p^2}{dz}$$

저주파 극한 ω ≪ Ω_i (이온 사이클로트론 주파수)에서 힘은 이온 질량 m_i에 거의 무관. 이것이 질량 비의존적 FIP 분별의 핵심. / In low-frequency limit ω ≪ Ω_i, the force is essentially mass-independent — the key to mass-independent FIP fractionation.

**(2) Non-uniform B (Alfvén wave):**
$$F_i = \frac{m_i c^2}{4} \frac{d}{dz}\left[\frac{\delta E_p(z_i)^2}{B(z_i)^2}\right]$$

이 형태가 실제 크로모스피어에서 적용된다. / This is the form used in the realistic chromosphere.

**(3) Fractionation integral:**
$$\frac{\rho_s(z_u)}{\rho_s(z_l)} = \exp\left\{ 2 \int_{z_l}^{z_u} \frac{\xi_s \, a \, \nu_{\text{eff}}}{\nu_{si} v_s^2}\, dz \right\}$$

ξ_s = 이온화 분율, a = 포데로모티브 가속도, v_s = 열속도, ν_si/ν_sn = 이온-수소/중성-수소 충돌률 / ξ_s = ionization fraction, a = ponderomotive acceleration, v_s = thermal speed, ν_si/ν_sn = ion-H / neutral-H collision rates.

**(4) Alfvén wave energy flux and density:**
$$U = \frac{\rho \, \delta v^2}{2} + \frac{\delta B^2}{8\pi} = \frac{\rho}{4}(I_+^2 + I_-^2), \quad F_\pm = \frac{\rho}{4} I_\pm^2 V_A$$

여기서 I_± = δv ± δB/√(4πρ) = Elsässer 변수. / I_± are Elsässer variables for waves propagating in ∓z directions.

**(5) Inverse FIP condition:**
$$a = \frac{\delta v^2}{f_R}\left\{ (f_R - 1)\left(-\frac{1}{8H_B} - \frac{1}{4H_B}\right) + \frac{c_S^2(z_{\beta=1}) V_A^2}{(V_A^2 + c_S^2)^2 f_R}\left(-\frac{c_S^2}{8H_D} - \frac{c_S^2}{4H_B} - \frac{V_A^2}{2H_B}\right)\right\}$$

첫 항(음수)이 우세하면 하방 ponderomotive 가속도 → Inverse FIP / When the first (negative) term dominates, downward ponderomotive acceleration → Inverse FIP. Requires |H_D| < |H_B|/6 for V_A ≫ c_S.

---

## 6. 읽기 가이드 / Reading Guide

**구조 / Structure**:
1. Sections 1-2: 서론 및 광구 존재비 기준 값 / Introduction and photospheric abundance baselines
2. Section 3: 태양 FIP 효과 관측 / Observational overview of solar FIP
3. Section 4: 항성 FIP/Inverse FIP 관측 (Wood-Linsky 관계) / Stellar FIP and Inverse FIP (Wood-Linsky relation)
4. Section 5: 초기 이론 모델들 (확산, 열전기, 이온 사이클로트론 공명) / Early theoretical attempts
5. Section 6: Ponderomotive force 모델 상세 — **핵심 섹션** / The ponderomotive force model in depth
6. Section 7: 닫힌/열린 자기장에서 결과, inverse FIP 모델링 / Closed/open field results, inverse FIP modeling
7. Section 8: 결론과 향후 연구 / Conclusions and future work

**읽기 순서 권장 / Recommended order**:
- 먼저 Section 3과 4로 관측 현상을 이해 / Start with Sections 3 & 4 for observational grounding
- Section 6.2의 ponderomotive force 유도 꼼꼼히 / Work through Section 6.2 derivation carefully
- Section 7은 구체적 loop 계산 예제 / Section 7 provides worked numerical examples
- Equations (4), (5), (22), (35) 이 네 식이 가장 중요 / Equations (4), (5), (22), (35) are the critical ones

**주의점 / Watch-outs**:
- WKB가 크로모스피어에서 실패함을 의식 / Be aware WKB fails in chromosphere; full transport integration needed
- 질량 비의존성 (ion-mass-independent)의 물리적 의미를 놓치지 말 것 / Don't miss the physical significance of mass-independence
- β=1 mode conversion이 inverse FIP의 열쇠 / β=1 mode conversion is the key to inverse FIP

---

## 7. 현대적 의의 / Modern Significance

**태양 연구 / Solar research**:
- Solar Orbiter / Parker Solar Probe가 채집한 태양풍 존재비 데이터 해석에 핵심적 / Central to interpreting Solar Orbiter / Parker Solar Probe in-situ abundance data
- Solar wind origin tracing: FIP 패턴은 태양풍 근원을 식별하는 "fingerprint" / FIP patterns serve as solar wind source fingerprints
- 코로나 가열 문제와 직결: ponderomotive force가 존재한다면 Alfvén wave heating이 동시에 일어나야 함 / Directly ties to coronal heating: wave amplitudes for FIP imply concurrent Alfvén-wave heating

**항성 연구 / Stellar astrophysics**:
- Inverse FIP (M dwarfs)은 강한 자기장에서 β=1 층이 깊숙이 내려간 경우를 의미 / Inverse FIP in M dwarfs signals a deep β=1 layer in strong-field stellar chromospheres
- 태양 aging과 항성 활동 진화의 일관된 해석 틀 / Consistent framework for solar aging and stellar activity evolution
- 외계 행성 host-star의 성간풍 조성 예측에 응용 / Useful for predicting stellar wind composition of exoplanet host stars

**우주 날씨 / Space weather**:
- 태양풍 조성 변동이 magnetosphere 반응 및 행성 대기 탈출에 영향 / Solar wind composition variability affects magnetospheric coupling and atmospheric escape

**향후 / Future**:
- MHD turbulence full-wave 시뮬레이션 (Dahlburg et al. 2012)이 존재비 예측의 다음 단계 / Full-wave MHD turbulence simulations are the next frontier
- Hinode/EIS, Solar Orbiter/SPICE, MUSE에서 시공간 분해 FIP 관측 / Spatially & temporally resolved FIP mapping

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
