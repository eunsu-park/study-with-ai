---
title: "Pre-Reading Briefing: Effects of the March 1989 Solar Activity"
paper_id: "17_allen_1989"
topic: Space_Weather
date: 2026-04-17
type: briefing
---

# Effects of the March 1989 Solar Activity: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Allen, J., Sauer, H., Frank, L., & Reiff, P. (1989). "Effects of the March 1989 Solar Activity." *Eos, Transactions American Geophysical Union*, 70(46), 1479-1488.
**Author(s)**: Joe Allen, Herb Sauer, Loren Frank, Patricia Reiff
**Year**: 1989

---

## 1. 핵심 기여 / Core Contribution

1989년 3월 대자기폭풍은 현대 우주기상 역사에서 가장 중요한 사건 중 하나입니다. 이 논문은 1989년 3월 6-19일 동안 발생한 일련의 태양 활동과 그로 인한 지자기 폭풍의 영향을 체계적으로 기록한 최초의 종합적인 보고서입니다. 가장 극적인 영향은 3월 13일 캐나다 Hydro-Québec 전력망의 완전 붕괴로, 600만 명이 9시간 동안 정전을 경험했습니다. 이 사건은 우주기상이 현대 기술 인프라에 실질적이고 치명적인 위협이 될 수 있음을 전 세계에 각인시켰습니다.

The March 1989 geomagnetic superstorm is one of the most significant events in modern space weather history. This paper provides the first comprehensive report documenting the series of solar activities during March 6-19, 1989, and the resulting geomagnetic storm impacts. The most dramatic consequence was the complete collapse of the Hydro-Québec power grid on March 13, leaving 6 million people without electricity for 9 hours. This event demonstrated to the world that space weather poses a real and catastrophic threat to modern technological infrastructure.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1989년은 태양 활동 주기 22의 극대기에 해당하는 시기였습니다. 태양 활동은 약 11년 주기로 변하며, 극대기에는 태양 플레어와 코로나 질량 방출(CME)의 빈도가 크게 증가합니다.

1989 fell during the maximum of Solar Cycle 22. Solar activity varies on an approximately 11-year cycle, and during solar maximum, the frequency of solar flares and coronal mass ejections (CMEs) increases dramatically.

이전 대자기폭풍들—1859년 Carrington Event, 1921년 뉴욕 철도 화재, 1972년 미 해군 기뢰 사건—은 기록되었지만, 현대적인 전력망과 통신 인프라가 취약하다는 것이 실제로 입증된 것은 1989년이 처음이었습니다.

Previous great geomagnetic storms — the 1859 Carrington Event, the 1921 New York railroad fires, the 1972 US Navy mine incident — were documented, but 1989 was the first time that modern power grids and communication infrastructure were proven vulnerable in practice.

### 타임라인 / Timeline

```
1859 ── Carrington Event: 최초의 기록된 대자기폭풍
         First recorded great geomagnetic storm
1921 ── New York Railroad Storm: 전신 화재 유발
         Caused telegraph fires
1957 ── IGY (International Geophysical Year): 체계적 관측 시작
         Systematic observations begin
1972 ── August storm: 미 해군 자기 기뢰 오작동
         US Navy magnetic mines malfunction
1975 ── Burton et al.: Dst 예측 공식 (Paper #11)
         Dst prediction formula
1989 ── March superstorm: Hydro-Québec 전력망 붕괴 ◀ THIS PAPER
         Hydro-Québec grid collapse
1994 ── Gonzalez & Tsurutani: 자기폭풍 정의 체계화 (Paper #15)
         Geomagnetic storm classification formalized
2003 ── Halloween storms: 위성/항공 피해 대규모 기록
         Satellite/aviation impacts extensively documented
```

---

## 3. 필요한 배경 지식 / Prerequisites

### A. 지자기 유도 전류 (GIC) / Geomagnetically Induced Currents

자기폭풍 동안 지구 자기장이 급격히 변하면, 패러데이 유도 법칙에 의해 지표면에 전기장이 유도됩니다. 이 전기장은 긴 도체(송전선, 파이프라인, 해저 케이블)에 준직류(quasi-DC) 전류를 흐르게 합니다. 이것이 GIC입니다.

During a geomagnetic storm, rapid changes in Earth's magnetic field induce electric fields at the surface via Faraday's law of induction. These electric fields drive quasi-DC currents through long conductors (power lines, pipelines, submarine cables). These are GICs.

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$

GIC가 변압기에 유입되면 자기 코어가 반주기 포화(half-cycle saturation)되어, 고조파 발생, 무효전력 소비 증가, 과열 및 손상이 일어납니다.

When GIC enters a transformer, it causes half-cycle saturation of the magnetic core, leading to harmonic generation, increased reactive power consumption, overheating, and potential damage.

### B. Dst 지수 / Dst Index

환전류(ring current)의 세기를 나타내는 지수로, 적도 부근 자기장 관측소들의 수평 성분 교란을 평균한 값입니다. 자기폭풍이 강할수록 Dst는 더 음의 값을 가집니다 (Paper #11 Burton et al. 참조).

The Dst index measures ring current intensity by averaging horizontal magnetic field perturbations at equatorial stations. Stronger storms produce more negative Dst values (see Paper #11 Burton et al.).

- Moderate storm: -50 to -100 nT
- Intense storm: -100 to -250 nT
- Super storm: < -250 nT
- **March 1989 storm: Dst ≈ -589 nT** (역사적으로 가장 강한 폭풍 중 하나)

### C. 전력망 기초 / Power Grid Basics

교류(AC) 전력 시스템은 발전, 송전, 배전의 세 단계로 구성됩니다. 고압 송전선은 수백 km에 걸쳐 전기를 전달하며, 변압기가 전압을 변환합니다. GIC는 이 변압기에서 문제를 일으킵니다.

AC power systems consist of three stages: generation, transmission, and distribution. High-voltage transmission lines carry electricity over hundreds of kilometers, with transformers converting voltage levels. GIC causes problems at these transformers.

캐나다 퀘벡은 특히 취약한데, (1) 높은 위도(오로라 전류대에 가까움), (2) 캐나다 순상지의 높은 지반 저항률(GIC 증가), (3) 긴 송전선이 원인입니다.

Québec was particularly vulnerable due to (1) high latitude (near the auroral electrojet), (2) high ground resistivity of the Canadian Shield (amplifies GIC), and (3) long transmission lines.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Coronal Mass Ejection (CME) / 코로나 질량 방출 | 태양 코로나에서 대량의 플라즈마와 자기장이 방출되는 현상. Massive ejection of plasma and magnetic field from the solar corona. |
| Geomagnetically Induced Current (GIC) / 지자기 유도 전류 | 자기폭풍 시 지표면에 유도되는 준직류 전류. Quasi-DC currents induced at Earth's surface during geomagnetic storms. |
| Dst Index / Dst 지수 | 환전류 세기를 나타내는 지자기 교란 지수. Geomagnetic disturbance index measuring ring current intensity. |
| Half-cycle Saturation / 반주기 포화 | GIC가 변압기 코어를 AC 주기의 절반 동안 포화시키는 현상. GIC saturating transformer cores during half of the AC cycle. |
| Reactive Power / 무효전력 | 전력계통에서 에너지를 저장/방출하지만 일을 하지 않는 전력 성분. Power component that stores/releases energy but does no work. |
| Solar Proton Event (SPE) / 태양 양성자 사건 | 태양 플레어에서 방출된 고에너지 양성자가 지구에 도달하는 현상. High-energy protons from solar flares reaching Earth. |
| Auroral Electrojet / 오로라 전류제트 | 극지방 전리층에 흐르는 강한 전류. Strong electric current flowing in the polar ionosphere. |
| Static Var Compensator (SVC) / 정지형 무효전력 보상기 | 전력계통의 무효전력을 보상하는 전력전자 장치. Power electronic device that compensates reactive power in power systems. |
| Single-Event Upset (SEU) / 단일 사건 오류 | 고에너지 입자가 전자기기의 비트를 뒤집는 현상. High-energy particle flipping a bit in electronic equipment. |
| Kp Index / Kp 지수 | 3시간 간격의 행성 자기 활동 지수 (0-9). Three-hourly planetary magnetic activity index (0-9). |
| Polar Cap Absorption (PCA) / 극관 흡수 | 태양 양성자가 극지 전리층을 이온화하여 HF 전파를 흡수하는 현상. Solar protons ionizing polar ionosphere, absorbing HF radio waves. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 패러데이 유도 법칙 / Faraday's Law of Induction

$$\oint \mathbf{E} \cdot d\mathbf{l} = -\frac{d\Phi_B}{dt}$$

지표면의 유도 전기장은 자기 플럭스의 시간 변화율에 비례합니다. $dB/dt$가 클수록 GIC도 커집니다.

The induced electric field at the surface is proportional to the time rate of change of magnetic flux. Larger $dB/dt$ produces stronger GIC.

### 5.2 GIC 크기 추정 / GIC Magnitude Estimation

$$\text{GIC} \approx \frac{E \cdot L}{R}$$

여기서 $E$는 지표 전기장 (V/km), $L$은 도체 길이 (km), $R$은 접지 저항 (Ω)입니다.

Where $E$ is the surface electric field (V/km), $L$ is the conductor length (km), and $R$ is the grounding resistance (Ω).

### 5.3 Burton 방정식 (Paper #11) / Burton Equation

$$\frac{dDst^*}{dt} = Q(t) - \frac{Dst^*}{\tau}$$

환전류의 주입($Q$)과 감쇠($\tau$)를 통해 Dst를 예측합니다. 1989년 3월 폭풍에서 $Q$가 극도로 컸습니다.

Predicts Dst through ring current injection ($Q$) and decay ($\tau$). During the March 1989 storm, $Q$ was extremely large.

### 5.4 자기폭풍 분류 기준 (Paper #15) / Storm Classification (Gonzalez & Tsurutani)

$$\text{Intense storm: } Dst_{\min} < -100 \text{ nT}$$
$$\text{Super storm: } Dst_{\min} < -250 \text{ nT}$$

1989년 3월 폭풍의 Dst ≈ -589 nT는 super storm 범주를 훨씬 초과합니다.

The March 1989 storm's Dst ≈ -589 nT far exceeds the super storm threshold.

---

## 6. 읽기 가이드 / Reading Guide

### 읽기 순서 추천 / Recommended Reading Order

1. **Introduction / 서론** — 1989년 3월 태양 활동의 개요를 파악하세요. 어떤 활동 영역(Active Region)에서 폭발이 시작되었는지 주목하세요.
   Get an overview of March 1989 solar activity. Note which Active Region initiated the eruptions.

2. **Solar Activity Summary / 태양 활동 요약** — 3월 6일부터 19일까지의 플레어, CME 시퀀스를 따라가세요. X-class 플레어의 빈도에 주목하세요.
   Follow the flare/CME sequence from March 6-19. Note the frequency of X-class flares.

3. **Geomagnetic Effects / 지자기 영향** — Dst, Kp 지수의 변화를 추적하세요. Dst가 -589 nT까지 떨어지는 급격한 주상(main phase)에 주목하세요.
   Track Dst and Kp index changes. Focus on the rapid main phase drop to -589 nT.

4. **Technological Impacts / 기술적 영향** — 가장 중요한 섹션입니다. Hydro-Québec 정전, 위성 이상, 통신 장애 등을 자세히 읽으세요.
   The most critical section. Read carefully about the Hydro-Québec blackout, satellite anomalies, and communication disruptions.

5. **Conclusions & Lessons / 결론 및 교훈** — 이 사건이 우주기상 정책과 예보에 미친 영향을 파악하세요.
   Understand the impact on space weather policy and forecasting.

### 주의 깊게 읽을 부분 / Pay Close Attention To

- **Hydro-Québec 붕괴 시퀀스**: SVC 트립 → 무효전력 부족 → 전압 붕괴 → 전체 정전의 연쇄 과정
  The cascade: SVC trip → reactive power deficit → voltage collapse → total blackout
- **시간 규모**: CME 발생에서 전력망 붕괴까지 걸린 시간 (경보 여유 시간의 한계)
  Time scales from CME eruption to grid collapse (limits of warning time)
- **다중 영향**: 전력 외에 위성, 항공, 통신, 파이프라인 등 다양한 분야의 피해
  Multi-sector impacts: satellites, aviation, communications, pipelines beyond just power

---

## 7. 현대적 의의 / Modern Significance

이 논문은 현대 우주기상 분야의 "기원 사건(origin event)"으로 간주됩니다:

This paper is considered the "origin event" of the modern space weather discipline:

1. **우주기상 예보 체계 구축 촉진 / Catalyzed space weather forecasting**: 이 사건 이후 미국 SWPC(Space Weather Prediction Center)의 역할이 대폭 강화되었고, 실시간 우주기상 경보 시스템이 확대되었습니다.
   After this event, NOAA's SWPC role was significantly enhanced, and real-time space weather alert systems were expanded.

2. **전력망 취약성 연구 활성화 / Power grid vulnerability research**: GIC에 대한 체계적 연구가 시작되었고, 변압기 보호 장비(GIC 차단기)가 개발되었습니다.
   Systematic GIC research began, and transformer protection equipment (GIC blockers) was developed.

3. **위성 설계 기준 강화 / Satellite design standards**: 방사선 차폐 기준이 강화되고, 우주기상 조건에 따른 위성 운영 프로토콜이 수립되었습니다.
   Radiation shielding standards were strengthened, and satellite operations protocols based on space weather conditions were established.

4. **국제 협력 강화 / International cooperation**: 우주기상에 대한 국제적 관심이 높아져, 이후 ISES(International Space Environment Service) 등의 협력이 강화되었습니다.
   International interest in space weather increased, leading to enhanced cooperation through organizations like ISES.

5. **2012년 Carrington급 CME 근접 통과**: 2012년 7월 23일 Carrington급 CME가 지구 궤도를 근접 통과했습니다. 만약 1주일 일찍 발생했다면 1989년보다 훨씬 심각한 피해를 초래했을 것입니다. 1989년 사건의 교훈이 없었다면 대비가 전무했을 것입니다.
   A Carrington-class CME narrowly missed Earth on July 23, 2012. Had it occurred one week earlier, the damage would have been far worse than 1989. Without the lessons of 1989, preparedness would have been nonexistent.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
