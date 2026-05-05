# Pre-reading Briefing: Surface Evolution of the Sun's Magnetic Field
# 사전 읽기 브리핑: 태양 자기장의 표면 진화

**Paper**: Sheeley, N. R., Jr. (2005)
**Journal**: *Living Reviews in Solar Physics*, **2**, 5
**DOI**: 10.12942/lrsp-2005-5

---

## 핵심 기여 / Core Contribution

이 논문은 Babcock (1961)과 Leighton (1964)의 초기 모델부터 2000년대 초반의 다중 태양주기 시뮬레이션까지, **flux-transport model(자기 플럭스 수송 모델)**의 40년 발전사를 저자 본인의 경험을 중심으로 서술한 역사적 리뷰입니다. 태양 표면에서 자기 플럭스가 어떻게 진화하는지 — 흑점군에서 기원하여 초과립 확산으로 퍼지고, 차등 회전으로 전단되며, 자오면 흐름으로 극지방으로 수송되어 궁극적으로 극성 자기장을 역전시키는 과정 — 을 기술하는 이 모델이 어떻게 관측과의 반복적 비교를 통해 정련되어 왔는지를 보여줍니다. 특히 자오면 흐름(meridional flow)의 역할이 처음에는 불확실했지만, 현재는 극성 역전과 다이나모 자체에 핵심적이라는 인식의 변화를 추적합니다.

This paper is a historical review tracing the 40-year development of the **flux-transport model** from the early models of Babcock (1961) and Leighton (1964) to multi-sunspot-cycle simulations of the early 2000s, told from the author's personal experience. It shows how the model describing the surface evolution of magnetic flux — originating in sunspot groups, spreading via supergranular diffusion, being sheared by differential rotation, and transported poleward by meridional flow to ultimately reverse the polar fields — has been refined through iterative comparison with observations. It traces the shift in understanding regarding meridional flow, initially uncertain but now recognized as central to polar field reversal and the dynamo itself.

---

## 역사적 맥락 / Historical Context

```
1952  Babcock & Babcock — 태양 자기장 일상 관측 시작
         Daily solar magnetic field observations begin
  |
1957–58  극성 자기장 역전 최초 관측 (남극 1957, 북극 1958)
           First polar field reversal observed (S-pole 1957, N-pole 1958)
  |
1961  Babcock — "The Topology of the Sun's Magnetic Field and the 22-Year Cycle"
         태양 자기장 위상 구조와 22년 주기 모델
  |
1963  Leighton — 초과립 확산에 의한 flux transport 아이디어 착상
         Leighton conceives flux transport via supergranular diffusion
  |
1964  Leighton — flux-transport 모델 최초 발표 (확산 + 차등 회전)
         First publication of flux-transport model (diffusion + differential rotation)
  |
1969  Leighton — 자오면 흐름 없이 태양주기 재현 시도 (magneto-kinematic model)
         Sunspot cycle reproduction without meridional flow
  |
1970s  "Dark ages" — 관측 결과가 모델과 불일치, 지하 자기장 가설 대두
         Observations inconsistent with model, subsurface field hypotheses arise
  |
1977  Mosher — 확산률 200-400 km²/s 추정, 자오면 흐름 ~3 m/s 추정
         Estimated diffusion rate 200-400 km²/s, meridional flow ~3 m/s
  |
1981  NRL 그룹 시뮬레이션 시작 (Sheeley, Boris, DeVore)
         NRL group begins simulations
  |
1983  Sheeley et al. — 유효 확산률 730±250 km²/s 측정
         Effective diffusion rate 730±250 km²/s measured
  |
1985  DeVore et al. — cycle 21 대부분 시뮬레이션 성공
         Successful simulation of most of cycle 21
  |
1987  Sheeley et al. — 자오면 수송이 준강체 회전의 원인임을 발견
         Meridional transport found to cause quasi-rigid rotation
  |
1989  Wang et al. — 자오면 흐름 10-20 m/s 필요 확인
         Meridional flow 10-20 m/s confirmed necessary
  |
1991  Wang et al. — flux-transport 다이나모 제안 (지하 역흐름 ~1 m/s)
         Flux-transport dynamo proposed (subsurface return flow ~1 m/s)
  |
2002  Wang et al. — 100년 시뮬레이션, 가변 자오면 흐름(±6 m/s) 필요
         100-year simulation, variable meridional flow (±6 m/s) needed
  |
>>> 2005  Sheeley — 이 리뷰 논문 <<<
           This review
```

---

## 필요한 배경 지식 / Prerequisites

### 태양 물리 기초 / Solar Physics Basics
- **태양주기 / Sunspot cycle**: ~11년 주기의 흑점 수 변동, ~22년 자기 주기 / ~11-year sunspot number variation, ~22-year magnetic cycle
- **Hale의 법칙 / Hale's law**: 각 반구에서 쌍극 흑점군의 선행 극성이 주기마다 반전 / Leading polarity of bipolar groups reverses each cycle in each hemisphere
- **Joy의 법칙 / Joy's law**: 흑점군이 약간 기울어져 후행 극성이 극에 더 가까움 / Sunspot groups are slightly tilted, trailing polarity closer to pole
- **극성 자기장 역전 / Polar field reversal**: 태양 극지의 자기장이 ~11년마다 극성이 바뀜 / Polarity of solar polar magnetic field reverses ~every 11 years

### 수송 과정 / Transport Processes
- **초과립 확산 / Supergranular diffusion**: 초과립 대류 셀(~30,000 km 크기)에 의한 자기 플럭스의 random walk
  Random walk of magnetic flux by supergranular convection cells (~30,000 km)
- **차등 회전 / Differential rotation**: 적도가 극보다 빠르게 회전 (~25일 vs ~35일)
  Equator rotates faster than poles (~25 days vs ~35 days)
- **자오면 흐름 / Meridional flow**: 적도에서 극으로 향하는 표면 흐름 (~10-20 m/s)
  Surface flow from equator to poles (~10-20 m/s)

### 수학 / Mathematics
- 확산 방정식 기초 / Basics of diffusion equations
- 구면 좌표계 / Spherical coordinates

---

## 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Flux-transport model** / 플럭스 수송 모델 | 태양 표면에서 자기 플럭스의 진화를 확산, 차등 회전, 자오면 흐름의 세 과정으로 기술하는 모델. / Model describing surface magnetic flux evolution through three processes: diffusion, differential rotation, meridional flow. |
| **Supergranular diffusion** / 초과립 확산 | 초과립 대류 셀의 수명과 크기에 의해 결정되는 유효 확산 계수. ~500-600 km² s⁻¹. / Effective diffusion coefficient determined by lifetime and size of supergranular cells. ~500-600 km² s⁻¹. |
| **Meridional flow** / 자오면 흐름 | 적도에서 극 방향으로의 대규모 표면 흐름. ~10-20 m s⁻¹. 극성 역전과 다이나모에 핵심. / Large-scale surface flow from equator to poles. ~10-20 m s⁻¹. Central to polar reversal and dynamo. |
| **Doublet source** / 쌍극 소스 | 양극과 음극이 쌍을 이루는 이상적 쌍극 자기 영역. 시뮬레이션의 입력 소스. / Idealized bipolar magnetic region with paired positive and negative poles. Input sources for simulations. |
| **Polar faculae** / 극 백반 | 극지방에서 관측되는 밝은 점. 강한 자기 플럭스 집중의 지표. / Bright points observed in polar regions. Indicators of concentrated magnetic flux. |
| **Butterfly diagram** / 나비 다이어그램 | 흑점 출현 위도를 시간에 따라 그린 도표. 주기 시작에는 고위도, 끝에는 저위도. / Plot of sunspot emergence latitude vs time. High latitude at cycle start, low at end. |
| **Quasi-rigid rotation** / 준강체 회전 | 대규모 자기장 패턴이 차등 회전에도 불구하고 거의 균일하게 회전하는 현상. 자오면 수송의 결과. / Large-scale field patterns rotating nearly uniformly despite differential rotation. Result of meridional transport. |
| **Source-surface model** / 소스-표면 모델 | 광구 자기장에서 코로나 자기장을 외삽하는 포텐셜 장 모델. / Potential-field model extrapolating coronal field from photospheric field. |
| **Open flux** / 열린 플럭스 | 태양에서 행성간 공간으로 뻗어나가는 자기 플럭스. 태양풍 구조와 직결. / Magnetic flux extending from Sun into interplanetary space. Directly linked to solar wind structure. |

---

## 수식 미리보기 / Equations Preview

이 논문은 역사적 서술 위주로, 명시적 수식이 거의 없습니다. 그러나 논문이 기반하는 **flux-transport 방정식** (Leighton 1964)을 미리 이해하면 도움됩니다:

This paper is primarily narrative with few explicit equations. However, understanding the underlying **flux-transport equation** (Leighton 1964) is helpful:

$$\frac{\partial B_r}{\partial t} = -\frac{1}{R\sin\theta}\frac{\partial}{\partial\theta}(v_\theta B_r \sin\theta) - \frac{1}{R\sin\theta}\frac{\partial}{\partial\phi}(v_\phi B_r) + \frac{\eta}{R^2}\left[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial B_r}{\partial\theta}\right) + \frac{1}{\sin^2\theta}\frac{\partial^2 B_r}{\partial\phi^2}\right] + S(\theta, \phi, t)$$

여기서 / Where:
- $B_r$: 태양 표면의 방사 자기장 성분 / Radial magnetic field at solar surface
- $v_\theta$: 자오면 흐름 속도 (~10-20 m/s 극방향) / Meridional flow speed (~10-20 m/s poleward)
- $v_\phi$: 차등 회전 속도 / Differential rotation speed
- $\eta$: 초과립 확산 계수 (~500-600 km² s⁻¹) / Supergranular diffusion coefficient
- $S(\theta, \phi, t)$: 새로운 흑점군 출현 (소스 항) / New sunspot group emergence (source term)
- $R$: 태양 반지름 / Solar radius
- $\theta, \phi$: 여위도, 경도 / Colatitude, longitude

### 핵심 매개변수 값 / Key Parameter Values

| 매개변수 / Parameter | 논문의 값 / Value in paper | 비고 / Notes |
|---|---|---|
| 확산 계수 $\eta$ | 500-600 km² s⁻¹ | Leighton 원래 값 770-1540에서 하향 조정 / Reduced from Leighton's original |
| 자오면 흐름 $v_\theta$ | 10-25 m s⁻¹ | 초기에 불확실, 현재 ~10-20 m/s로 확립 / Initially uncertain, now established |
| 자오면 흐름 변동 | ±6 m s⁻¹ (주기별) | 활동 주기에 빠르고, 비활동 주기에 느림 / Faster in active cycles, slower in inactive |
| Doublet 소스 플럭스 | ≥ 0.1 × 10²¹ Mx | 시뮬레이션 입력 임계값 / Simulation input threshold |

### 핵심 물리적 메커니즘 / Key Physical Mechanism

극성 역전의 메커니즘은 다음과 같습니다:
The polar field reversal mechanism works as follows:

1. 흑점군이 Joy's law 기울기로 출현 → 후행 극성이 극에 더 가까움
   Sunspot groups emerge with Joy's law tilt → trailing polarity closer to pole
2. 초과립 확산이 양극과 음극을 분산시킴
   Supergranular diffusion disperses both polarities
3. 차등 회전이 패턴을 전단시킴
   Differential rotation shears the patterns
4. 자오면 흐름이 후행 극성 플럭스를 극 방향으로 수송
   Meridional flow transports trailing polarity flux poleward
5. 후행 극성이 기존 극성 자기장을 상쇄하고 역전시킴
   Trailing polarity cancels and reverses existing polar field
6. 선행 극성은 적도 부근에서 반대 반구의 선행 극성과 상쇄됨
   Leading polarity cancels near equator with opposite hemisphere's leading polarity

---

## 논문 구조 안내 / Paper Structure Guide

| 섹션 / Section | 시기 / Era | 핵심 내용 / Content |
|---|---|---|
| §1 The Beginning | 1963-1969 | Leighton의 아이디어 탄생, 초기 모델, Sheeley 박사과정 |
| §2 The 1970s | 1970s | "Dark ages" — Mosher의 작은 확산률, 자오면 흐름 논쟁 |
| §3 Early Simulations | 1980s초 | NRL 그룹 시뮬레이션 시작, DeVore 합류 |
| §4 The Era of Enlightenment | 1980s-90s | 자오면 흐름 확인, 준강체 회전 설명, flux-transport dynamo |
| §5 The Australian School | 1990s-2000s | Wilson 그룹의 비판과 독립 시뮬레이션 |
| §6 Simulations Over Many Cycles | 2000s | 열린 플럭스, 100년 시뮬레이션, 가변 흐름 속도 |
| §7 Epilogue | — | 역사적 성찰, Babcock/Ward 참조 |

---

## 읽기 전략 / Reading Strategy

이 논문은 수식이나 이론적 유도보다는 **과학적 발견의 역사적 과정**에 초점을 맞춘 1인칭 서술 논문입니다. 다음에 주목하세요:

This paper focuses on the **historical process of scientific discovery** rather than equations or theoretical derivations. It's a first-person narrative. Pay attention to:

1. **매개변수 값의 변천사**: 확산률과 자오면 흐름 속도가 시간에 따라 어떻게 조정되었는지
   How diffusion rate and meridional flow speed were adjusted over time

2. **관측과 시뮬레이션의 상호작용**: Figure 3가 핵심 — 수송 없음 vs 확산만 vs 확산+흐름의 극적 차이
   Observation-simulation interplay: Figure 3 is key — dramatic differences between no transport vs diffusion only vs diffusion+flow

3. **과학적 논쟁의 해결 과정**: 특히 준강체 회전과 자오면 흐름의 필요성에 관한 논쟁
   How scientific debates were resolved, especially about quasi-rigid rotation and meridional flow necessity
