---
title: "Response of the Thermosphere and Ionosphere to Geomagnetic Storms"
authors: Timothy J. Fuller-Rowell, Mihail V. Codrescu, Robert G. Roble, Arthur D. Richmond
year: 1994
journal: "Journal of Geophysical Research"
doi: "10.1029/93JA02015"
topic: Space_Weather
tags: [thermosphere, ionosphere, geomagnetic-storm, composition, O/N2-ratio, TIGCM, satellite-drag, F-region]
status: completed
date_started: 2026-04-16
date_completed: 2026-04-16
---

# 16. Response of the Thermosphere and Ionosphere to Geomagnetic Storms / 지자기 폭풍에 대한 열권-전리층 반응

---

## 1. Core Contribution / 핵심 기여

이 논문은 NCAR의 결합 열권-전리층 대순환 모델(Coupled Thermosphere-Ionosphere Model, CTIM)을 사용하여 지자기 폭풍 동안 열권과 전리층의 반응을 체계적으로 시뮬레이션한 연구입니다. 4개의 서로 다른 UT 시작 시간(1200, 1800, 2400, 0600 UT)에 동일한 폭풍을 적용하여, 폭풍 반응의 **UT 의존성**과 **경도 의존성**을 규명했습니다. 핵심 발견은 (1) 고위도 줄 가열이 열권 온도를 400 K 이상 상승시키고, (2) 이로 인한 대규모 순환이 적도 방향 바람 서지(wind surge)를 구동하며 위상 속도 ~600–700 m/s로 전파되고, (3) 발산 바람장이 N₂가 풍부한 "조성 돌출(composition bulge)"을 생성하여 중위도까지 이동시키며, (4) 이 조성 변화가 F2층 전자 밀도의 "음의 폭풍 효과"의 주된 원인이라는 것입니다. 또한 야간에는 배경풍과 폭풍풍이 모두 적도 방향이어서 조성 돌출이 쉽게 확장되지만, 주간에는 배경풍이 극 방향이라 확장이 억제된다는 주야간 비대칭성을 밝혔습니다.

This paper systematically simulated the thermosphere-ionosphere response to geomagnetic storms using the NCAR Coupled Thermosphere-Ionosphere Model (CTIM). By applying identical storms at four different UT start times (1200, 1800, 2400, and 0600 UT), the study elucidated the **UT dependence** and **longitude dependence** of storm responses. Key findings include: (1) high-latitude Joule heating raises thermospheric temperatures by over 400 K; (2) the resulting large-scale circulation drives equatorward wind surges propagating at phase velocities of ~600–700 m/s; (3) the divergent wind field generates an N₂-enriched "composition bulge" that migrates to middle latitudes; and (4) this composition change is the primary cause of the "negative storm effect" in F2-layer electron density. The paper also revealed a critical day-night asymmetry: on the nightside, both background and storm-driven winds are equatorward, allowing easy bulge expansion, while on the dayside, the poleward background wind restricts equatorward penetration of the composition disturbance.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction / 서론 (pp. 3893–3894)

논문은 지자기 폭풍 시 전리층 F-영역의 전자 밀도 변화에 대한 기존 연구를 정리하면서 시작합니다. Prölss [1980]와 Rishbeth [1991]의 리뷰를 언급하며, 상부 대기의 관점에서 폭풍이 수시간~하루에 걸친 자기권 에너지 입력의 급증임을 설명합니다.

The paper opens by reviewing prior work on ionospheric F-region density changes during geomagnetic storms, referencing reviews by Prölss [1980] and Rishbeth [1991]. From the upper atmosphere perspective, a storm represents a period of substantial increase in magnetospheric energy input over several hours to a day.

폭풍 시 상부 대기에 영향을 미치는 주요 메커니즘들:

Key mechanisms affecting the upper atmosphere during storms:

- **대류 전기장 증가 / Enhanced convection electric field**: 강도와 공간 범위 확대 → Foster et al. [1986], Knipp et al. [1989]
- **오로라 강수 증가 / Enhanced auroral precipitation**: 오로라 타원이 저위도까지 확장, 지상에서도 관측 가능 → Allen et al. [1989]
- **F-영역 밀도 증가 / F-region density increase**: 오로라 강수에 의한 직접적 이온화 증가
- **열권 팽창 / Thermospheric expansion**: 줄 가열 → 온도 상승 → 위성 항력 증가
- **중성 조성 변화 / Neutral composition changes**: O/N₂ 비율 변화 → 전리층 밀도 변화

### Part II: Model Description / 모델 설명 (pp. 3894–3895)

모델은 CTIM (Coupled Thermosphere-Ionosphere Model)으로, Fuller-Rowell et al. [1987]과 Quegan et al. [1982]에 기반합니다. 주요 특성:

The model is the CTIM, based on Fuller-Rowell et al. [1987] and Quegan et al. [1982]. Key characteristics:

- **자기일관적 결합 / Self-consistent coupling**: 열권의 중성 대기와 전리층의 이온/전자가 상호작용
- **압력 좌표계 / Pressure coordinate system**: 15개 압력면 사용, 고도 ~80 km에서 ~500 km 이상
- **해상도 / Resolution**: 위도 2°, 경도 18° (= 20개 경도 격자)
- **입력 조건 / Input forcing**: 
  - Cross-polar-cap potential drop: 배경 45 kV → 폭풍 최대 130 kV (Figure 1)
  - 활동도 레벨: 배경 S55 (Kp 약 3), 폭풍 레벨 10 (12시간 지속)
  - 전기장은 Foster et al. [1986]을 기반으로 하되 30% 증강, 폭풍 시 추가 25% 증강

**모델의 한계점 / Model limitations** (p. 3895):
- 전기장의 저위도 관통(equatorward penetration)은 포함되지 않음
- 진동적으로 여기된 분자질소(vibrationally excited N₂*)의 효과 미포함
- 오로라 타원의 위도 범위가 실제보다 약간 좁을 수 있음 (~10° 정도)

### Part III: Polar Thermosphere Response / 극지역 열권 반응 (pp. 3895–3897)

고위도에서의 반응이 가장 극적입니다:

The most dramatic response occurs at high latitudes:

**줄 가열 패턴 (Figure 2)**:
- 폭풍 6시간 후(GS1, 1800 UT) 줄 가열률 패턴을 북극에서 본 극좌표 도면
- 최대 가열률: 50 mW m⁻² 이상
- 줄 가열 패턴은 자기극을 중심으로 하며, dawn sector에서 강함
- "열권을 통해 plows" → 지구 자전에 따라 가스를 가열하며 이동

**온도 변화 (Figure 3)**:
- 12시간 폭풍 후(GS1, 2400 UT) 압력면 12 (~300 km 고도)에서의 온도 증가
- **최대 온도 증가: 400 K 이상** (고위도 자정 sector)
- 50° 위도까지 유사한 크기의 온도 증가
- 바람 변화: dusk sector에서 600 m/s, 극관 위 400 m/s, dawn sector 250 m/s

**중성풍 변화**:
- dawn/dusk 비대칭성: dusk sector에서 풍속이 더 큼 — Fuller-Rowell [1985]에서 이미 설명됨
- 패턴이 자기극과 지리극의 offset에 의해 dusk 쪽으로 치우침
- 극지방 온도 증가 → 대규모 압력 경도 → 전지구 순환 변화

### Part IV: Global Dynamical Response / 전지구 동역학적 반응 (pp. 3897–3904)

이 섹션이 논문의 핵심입니다.

This section is the heart of the paper.

#### 바람 서지 (Wind Surge)

폭풍 시 고위도 에너지 주입은 적도 방향 바람 서지를 발생시킵니다:

Storm-time high-latitude energy input generates equatorward wind surges:

- **위상 속도 / Phase velocity**: ~600–700 m/s (대규모 중력파와 유사)
- **실제 풍속 / Actual wind speed**: ~100 m/s
- **전파 특성**: 양 극지방에서 출발, 저위도로 전파, 적도에서 수렴
- 서지 뒤에 약 600 m/s⁻¹의 전지구 순환이 형성 (Plate 2)

서지 전파와 순환 변화는 같은 현상의 두 측면입니다. 전파면은 적도에서 반대 반구로 관통합니다.

The surge propagation and circulation change are two facets of the same phenomenon. The wave front penetrates into the opposite hemisphere.

#### 자오선 풍(Meridional Wind) — Plates 2–5

**GS1 (1200 UT 시작) — Plate 2**:
- (a) 3시간: 고위도에서 적도 방향 풍속 증가 시작
- (b) 6시간: 서지가 중위도에 도달, 풍속 최대 ~400 m/s (고위도)
- (c) 12시간: 서지가 적도를 통과하여 반대 반구까지 도달
- (d) 18시간: 폭풍 종료 후 바람이 빠르게 회복

**GS3 (2400 UT 시작) — Plate 3**:
- GS1과 동일한 폭풍이지만 UT 시작 시간만 다름
- 반응 패턴은 "현저하게 유사(remarkably similar)" — **UT 의존성이 제한적**
- 이는 대기의 관성이 UT 변화보다 동역학을 지배함을 시사

**4개 경도 단면 (Plates 4–5)**:
- 0°, 90°, 180°, 270° 경도에서의 자오선 풍 시간 이력
- 각 경도에서 자기극까지의 거리가 다르므로 반응 강도가 다름
- **가장 강한 반응**: 자기극에 가장 가까운 야간 sector
- **가장 약한 반응**: 자기극에서 가장 먼 주간 sector

#### 동서풍(Zonal Wind) — Plate 6

- 0° 경도 단면에서 동서풍 변화
- 서향풍(westward) 500 m/s 이상이 저위도에서 나타남
- 코리올리 효과에 의해 적도 방향 자오선 풍이 서향으로 전환
- **각운동량 보존이 동서풍 크기를 제한**: 더 긴 폭풍이라도 자오선 풍이 무한히 유지되지 않음

#### 순환의 두 단계 / Two Phases of Circulation

1. **1단계 (First phase)**: 서지 전파 — 대규모 중력파 특성, 빠른 적도 방향 이동
2. **2단계 (Second phase)**: 코리올리 효과에 의한 동서풍 증가 → 자오선 압력 경도와 균형 → 새로운 지오스트로픽 균형 도달

폭풍 12시간 후 (Plates 4b 참조), 순환은 적도 방향 이동의 단순 증가가 아니라 모든 위도와 경도에서 새로운 평형을 향해 재조정됩니다.

After 12 hours of storm (see Plates 4b), the circulation is not simply an increase in equatorward flow but a readjustment toward a new equilibrium at all latitudes and longitudes.

### Part V: Global Neutral Composition Response / 전지구 중성 조성 반응 (pp. 3901–3906)

**이 논문의 가장 중요한 결과**입니다.

This is the **most important result** of the paper.

#### 평균 분자량(Mean Molecular Mass, m) 변화 — Plate 1

Plate 1은 6개의 극좌표 도면으로, 압력면 12 (~300 km)에서 평균 분자량 변화의 시간 진화를 보여줍니다:

Plate 1 shows six polar coordinate plots depicting the temporal evolution of mean molecular mass change at pressure level 12 (~300 km):

- **(a) 6시간**: 고위도에서 m 증가(= N₂ 풍부) 시작, 자정 sector에 집중
- **(b) 12시간**: "조성 돌출"이 뚜렷하게 형성, 50°N까지 확장
- **(c) 18시간**: 폭풍 종료 후에도 조성 변화 지속, 야간에서 주간으로 이동
- **(d) 24시간**: 돌출이 주간으로 이동하면서 약화, 야간 쪽은 회복 시작
- **(e) 36시간**: 회복 진행, 돌출이 주간에 잔류
- **(f) 48시간**: 거의 배경 상태로 회복

핵심 물리:

Key physics:

$$\text{발산 바람} \rightarrow \text{압력면 상의 상승류(upwelling)} \rightarrow \text{무거운 성분(N}_2\text{) 상승} \rightarrow m \uparrow \rightarrow \text{O/N}_2 \downarrow$$

#### 조성 돌출의 일주 변동 / Diurnal Variation of Composition Bulge

**이 논문의 핵심 통찰 중 하나**:

One of the paper's key insights:

- **야간**: 배경 자오선 풍 = 적도 방향 → 폭풍 바람과 같은 방향 → 조성 돌출이 쉽게 적도로 이동
- **주간**: 배경 자오선 풍 = 극 방향 → 폭풍 바람과 반대 방향 → 조성 돌출의 적도 방향 이동이 억제
- 결과: **조성 돌출이 지구와 함께 회전하지 않고**, 배경풍과 폭풍풍의 상호작용에 따라 **local time sector에 따른 일주 변동**을 보임

- **Nightside**: background meridional wind = equatorward → same direction as storm wind → composition bulge easily migrates equatorward
- **Dayside**: background meridional wind = poleward → opposite to storm wind → equatorward migration restricted
- Result: the composition bulge **does not simply corotate with Earth** but exhibits a **diurnal variation** driven by the interaction of background and storm winds

#### 4개 경도 단면에서의 m 변화 — Plates 7, 10

**GS1 폭풍 구동기 (Plate 7)**: 0°, 90°, 180°, 270° 경도에서 24시간 동안의 m 변화
- 모든 경도에서 초기 m 증가(고위도)는 발산 바람에 의한 상승류의 결과
- 조성 돌출이 형성되면 수평 바람에 의해 수송 가능
- 검은 실선: m 증가/감소의 경계
- 야간 sector (270° 경도)에서 가장 깊이 적도로 침투

**GS1 회복기 (Plate 10)**: 24–48시간, 폭풍 종료 후
- 배경풍(주간 극방향)이 돌출을 다시 극으로 밀어냄
- 회복은 점진적 (24–48시간 소요)
- 조성 변화가 이온 밀도 변화(음의 폭풍 효과)를 **폭풍 종료 후에도 지속시킴**

### Part VI: Ionospheric Response / 전리층 반응 (pp. 3906–3912)

#### NmF2 변화 — Plates 8, 9, 11

전리층 F2 피크 이온 밀도($N_mF2$)의 변화는 열권 조성과 바람의 복합 효과입니다:

The change in F2 peak ion density ($N_mF2$) is a combined effect of thermospheric composition and winds:

**양의 폭풍 효과 (Positive storm effect)**:
- 적도 방향 바람 → F층을 높은 고도로 밀어 올림 → 재결합률 감소 → 밀도 증가
- 주로 폭풍 초기, **주간** sector에서 발생
- 야간에는 이온 밀도가 이미 낮아 양의 효과 탐지 어려움
- 예: 180° 경도 sector에서 처음 6시간 동안 적도 방향 바람이 양의 반응 생성

**음의 폭풍 효과 (Negative storm effect)**:
- N₂ 증가(조성 돌출) → $\text{O}^+ + \text{N}_2 \rightarrow \text{NO}^+ + \text{N}$ 반응 증가 → 밀도 감소
- **주간**에서 가장 두드러짐 (태양 EUV에 의한 이온 생성이 있어야 조성 변화가 밀도에 영향)
- 주간에 조성 돌출이 도달하면 강한 음의 위상이 나타남

**GS1 (Plate 8)**: 4개 경도에서 12시간 동안의 $N_mF2$ 변화
- 0° 경도: 음/양의 위상 혼재, 복잡한 패턴
- 90° 경도: 주간 양의 위상 후 음의 위상으로 전환
- 180° 경도: 명확한 양의 위상 (처음 6시간)
- 270° 경도: 야간 sector, 강한 음의 위상

**GS1 회복기 (Plate 11)**: 24–48시간
- **회복기에도 음의 위상 지속**: 조성 변화가 폭풍 종료 후 최소 24시간 잔류
- 일부 경도에서 양의 위상 잔류 — 이전 downwelling에 의한 m 감소 영역

**UT 의존성** (GS1 vs. GS3 비교):
- Plate 8 (GS1, 1200 UT 시작) vs. Plate 9 (GS3, 2400 UT 시작)
- 동역학적 반응은 유사하지만, 어떤 경도 sector가 폭풍 구동기에 야간/주간인지에 따라 전리층 반응이 다름
- 이는 **전리층 폭풍 효과의 경도/UT 의존성**을 설명

#### Figure 4: NmF2 비율의 시간-위도 변화

0° 경도에서 교란/정상 $N_mF2$ 비율의 자연 로그값:
- 1보다 큰 값(밝은 색) = 양의 위상
- 1보다 작은 값(어두운 색) = 음의 위상
- 최대 감소: ~45%까지 (약 0.55배)
- 양/음 위상의 공간적 패턴이 local time에 강하게 의존

### Part VII: Conclusion / 결론 (pp. 3912–3914)

저자들은 이 상대적으로 단순한 이상화된 폭풍 시뮬레이션에서도 전리층 반응이 매우 복잡함을 강조합니다:

The authors emphasize that even for these relatively simple idealized storm simulations, the ionospheric response is remarkably complex:

1. **바람 서지**: 대규모 중력파 특성, 위상 속도 600–700 m/s, 풍속 ~100 m/s
2. **조성 돌출**: 지구와 함께 회전하지 않음, 배경풍과의 상호작용이 일주 변동 결정
3. **음의 폭풍 효과**: 주로 조성 변화(O/N₂ 감소)에 기인, 주간에 가장 강함
4. **양의 폭풍 효과**: 적도 방향 바람에 의한 F층 상승에 기인
5. **UT/경도 의존성**: 자기극-지리극 오프셋이 경도에 따른 반응 차이를 유발
6. **회복**: 조성 변화가 폭풍 종료 후 24–48시간 지속 → 장기적 전리층 효과

모델 결과는 Prölss [1980, 1993]의 관측적 폭풍 시나리오를 잘 지지하지만, 세부 수정 사항도 있습니다: 경도/UT 의존성의 출현이 단순한 "일반적 시나리오"보다 실제가 더 복잡함을 시사합니다.

The model results support the observational storm scenario of Prölss [1980, 1993] but with refinements: the emergence of longitude/UT dependence suggests reality is more complex than a simple "general scenario."

---

## 3. Key Takeaways / 핵심 시사점

1. **줄 가열이 열권 온도를 400 K 이상 상승시킨다** — 12시간 폭풍 동안 고위도 ~300 km에서 온도가 배경 대비 400 K 이상 상승하며, 이것이 모든 후속 반응의 근본 원인입니다. Joule heating raises thermospheric temperatures by over 400 K at high latitudes (~300 km altitude) during a 12-hour storm, and this is the root cause of all subsequent responses.

2. **바람 서지는 대규모 중력파의 특성을 가진다** — 위상 속도 600–700 m/s, 실제 풍속 ~100 m/s로 전파되는 적도 방향 서지는 파동과 순환 변화의 이중 성격을 가집니다. 반대 반구까지 관통합니다. The equatorward wind surge propagates at phase velocities of 600–700 m/s with actual wind speeds of ~100 m/s, having a dual character of wave propagation and circulation change, penetrating into the opposite hemisphere.

3. **조성 돌출은 지구와 함께 회전하지 않는다** — 이것이 이 논문의 가장 혁신적인 통찰입니다. N₂ 풍부 영역은 배경풍(주간: 극방향, 야간: 적도방향)과 폭풍풍의 상호작용에 따라 local time에 따른 일주 변동을 보입니다. The N₂-enriched composition bulge does not corotate with Earth — this is the paper's most innovative insight. Its position is modulated by the interaction of background winds (daytime poleward, nighttime equatorward) and storm winds.

4. **O/N₂ 비율 감소가 음의 폭풍 효과의 주된 원인이다** — 발산 바람장에 의한 상승류(upwelling)가 압력면에서 N₂를 상방 수송하여 O/N₂ 비율을 감소시키고, 이것이 F2층의 O⁺ 손실률을 높여 전자 밀도를 감소시킵니다. The decreased O/N₂ ratio, caused by upwelling through pressure surfaces due to divergent wind fields, is the primary driver of the negative ionospheric storm effect.

5. **양의 폭풍 효과는 바람에 의한 F층 상승에 기인한다** — 적도 방향 자오선 풍이 F층을 높은 고도로 밀어 올려 재결합률을 감소시킵니다. 주로 폭풍 초기, 주간 sector에서 발생합니다. Positive storm effects arise from equatorward meridional winds pushing the F-layer to higher altitudes where recombination is slower, occurring mainly during the early storm phase in the daytime sector.

6. **전리층 폭풍 효과는 강한 UT/경도 의존성을 보인다** — 자기극과 지리극의 오프셋(~11.5°)으로 인해, 같은 폭풍이라도 어떤 경도가 야간/주간에 있는지에 따라 전리층 반응이 크게 다릅니다. Ionospheric storm effects show strong UT/longitude dependence due to the ~11.5° offset between magnetic and geographic poles; the same storm produces very different responses depending on which longitude is in the night/day sector.

7. **조성 변화는 폭풍 종료 후 24–48시간 지속된다** — 회복기에도 조성 돌출이 잔류하여 음의 전리층 효과가 지속됩니다. 이는 전리층 예보에서 회복기 예측의 어려움을 설명합니다. Composition changes persist for 24–48 hours after storm cessation, causing lingering negative ionospheric effects during recovery, explaining the difficulty of forecasting ionospheric recovery.

8. **각운동량 보존이 자오선 순환을 제한한다** — 코리올리 효과에 의해 적도 방향 바람이 동서풍으로 전환되면서, 극단적 폭풍 입력에서도 자오선 풍의 크기에 이론적 상한이 존재합니다. Conservation of angular momentum limits meridional circulation: the Coriolis effect converts meridional to zonal winds, imposing a theoretical upper bound on meridional wind magnitudes even under extreme storm input.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 줄 가열률 / Joule Heating Rate

$$Q_J = \sigma_P |\mathbf{E} + \mathbf{V}_n \times \mathbf{B}|^2$$

- $\sigma_P$: Pedersen 전도도 — 오로라 강수에 의해 증강됨 / Pedersen conductivity, enhanced by auroral precipitation
- $\mathbf{E}$: 대류 전기장 — 폭풍 시 cross-polar-cap potential이 45 kV → 130 kV로 증가 / Convection electric field, cross-polar-cap potential increases from 45 kV to 130 kV during storm
- $\mathbf{V}_n$: 중성풍 속도 / Neutral wind velocity
- $\mathbf{B}$: 지구 자기장 / Earth's magnetic field

이 식에서 $\mathbf{V}_n \times \mathbf{B}$ 항은 중성풍이 이온과 다른 속도로 움직일 때의 줄 가열 기여를 나타냅니다. 폭풍 시 $E$가 크게 증가하면 $Q_J$가 1차수 이상 증가합니다.

### 4.2 연속 방정식과 상승류 / Continuity Equation and Upwelling

압력 좌표계에서 연속 방정식:

$$\frac{\partial \bar{m}}{\partial t} = -\mathbf{V}_h \cdot \nabla_p \bar{m} - \dot{p} \frac{\partial \bar{m}}{\partial p} + D$$

- $\bar{m}$: 평균 분자량 / Mean molecular mass
- $\mathbf{V}_h$: 수평 바람 / Horizontal wind
- $\dot{p}$: 연직 속도 (압력 좌표) / Vertical velocity in pressure coordinates
- $D$: 분자 확산 항 / Molecular diffusion term

발산 수평 바람($\nabla \cdot \mathbf{V}_h > 0$) → 연속성에 의해 $\dot{p} < 0$ (상승류) → 아래에서 N₂ 풍부한 공기 상승 → $\bar{m}$ 증가

Divergent horizontal wind → upwelling by continuity → N₂-rich air from below rises → $\bar{m}$ increases

### 4.3 스케일 높이와 조성 / Scale Height and Composition

각 성분의 스케일 높이:

$$H_i = \frac{k_B T}{m_i g}$$

- $H_O = \frac{k_B T}{16 \, m_u \cdot g} \approx 50 \text{ km}$ (T = 1000 K에서 / at T = 1000 K)
- $H_{N_2} = \frac{k_B T}{28 \, m_u \cdot g} \approx 29 \text{ km}$

온도가 T에서 T + ΔT로 증가하면:

$$\frac{n_{N_2}(z, T+\Delta T)}{n_{N_2}(z, T)} = \exp\left(\frac{z}{H_{N_2}(T)} - \frac{z}{H_{N_2}(T+\Delta T)}\right)$$

무거운 N₂의 밀도가 가벼운 O보다 더 크게 증가 → O/N₂ 비율 감소

Heavier N₂ density increases more than lighter O → O/N₂ ratio decreases

### 4.4 F2층 이온 밀도 평형 / F2-Layer Ion Density Equilibrium

F2 피크에서의 근사적 평형:

$$N_mF2 \propto \frac{q}{L} = \frac{n(O) \cdot I \cdot \sigma_O}{k_1 \cdot n(N_2) + k_2 \cdot n(O_2)} \propto \frac{n(O)}{n(N_2)}$$

여기서 / where:
- $q$: 이온 생성률 ∝ $n(O)$ / Ion production rate ∝ $n(O)$
- $L$: 손실률 ∝ $n(N_2)$ / Loss rate ∝ $n(N_2)$
- $k_1 \approx 1.2 \times 10^{-12}$ cm³/s (O⁺ + N₂ 반응 계수 / reaction coefficient)

따라서 O/N₂ 비율 감소 → $N_mF2$ 감소 (음의 폭풍 효과)

### 4.5 바람에 의한 F층 상승 / Wind-Induced F-Layer Uplift

적도 방향 자오선 풍 $V_n$에 의한 F층 높이 변화:

$$\frac{dh_F}{dt} \approx V_n \sin I \cos I$$

- $I$: 자기 복각 / Magnetic dip angle
- $h_F$: F층 피크 높이 / F-layer peak height

높은 고도에서는 $n(N_2)$가 작으므로 손실률이 감소 → $N_mF2$ 증가 (양의 폭풍 효과)

At higher altitudes, $n(N_2)$ is smaller, so the loss rate decreases → $N_mF2$ increases (positive storm effect)

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1966    Jacchia — 폭풍 시 열권 밀도 증가 경험 모델 / Empirical model of storm-time thermospheric density increase
        ↓
1972    Woodman et al. — 전리층 전기장 관통 관측 / Observations of ionospheric electric field penetration
        ↓
1977    Bowman — 폭풍 시 F층 높이 변화 / Storm-time F-layer height changes
        ↓
1980    Prölss — 전리층 폭풍 효과 종합 리뷰 / Comprehensive review of ionospheric storm effects
        ↓
1980    Dickinson et al. — 최초 NCAR TGCM / First NCAR Thermospheric GCM
        ↓
1982    Rishbeth & Garriott — O/N₂ 가설 제안 / O/N₂ hypothesis proposed
        ↓
1985    Fuller-Rowell — dawn/dusk 비대칭성 설명 / Dawn/dusk asymmetry explained
        ↓
1986    Foster et al. — 폭풍 시 전기장 패턴 관측 / Storm-time electric field patterns observed
        ↓
1987    Fuller-Rowell et al. — CTIM 개발 / CTIM development
        ↓
1987    Roble et al. — TIGCM 개발 / TIGCM development
        ↓
1988    Richmond & Kamide — AMIE 기법 (Paper #13) / AMIE technique
        ↓
1991    Rishbeth — 폭풍 시 F-영역 리뷰 / F-region storm review
        ↓
1993    Prölss — 전리층 폭풍의 일반적 시나리오 / General scenario of ionospheric storms
        ↓
★ 1994  Fuller-Rowell et al. — THIS PAPER: 열권-전리층 폭풍 반응 시뮬레이션
        ↓
1994    Gonzalez et al. — 지자기 폭풍 분류 (Paper #15) / Geomagnetic storm classification
        ↓
1995    Buonsanto — 전리층 폭풍 리뷰 / Ionospheric storm review
        ↓
2004    Meier et al. — GUVI O/N₂ 관측 검증 / GUVI O/N₂ observational verification
        ↓
2010s   TIE-GCM, WACCM-X — 차세대 결합 모델 / Next-generation coupled models
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #13 Richmond & Kamide (1988) — AMIE | AMIE 기법이 제공하는 고위도 전기장/강수 패턴이 이 모델의 입력 조건의 관측적 기반. Richmond은 공저자 / AMIE provides the observational basis for high-latitude input. Richmond is a co-author | 직접적 선행 연구 / Direct predecessor |
| #15 Gonzalez et al. (1994) — Storm classification | 이 논문이 시뮬레이션하는 "지자기 폭풍"의 정량적 정의(Dst 기준)를 확립 / Established quantitative definition of the "geomagnetic storm" this paper simulates | 정의 및 분류 체계 / Definition framework |
| Prölss (1980, 1993) | 전리층 폭풍 효과의 관측적 "일반 시나리오" — 이 논문이 시뮬레이션으로 검증하고 세부 수정 / Observational "general scenario" of ionospheric storm effects — verified and refined by this simulation | 관측적 검증 대상 / Observational validation target |
| #11 Burton et al. (1975) — Ring current | Dst 지수와 태양풍 파라미터의 관계 — 폭풍 강도의 정량화에 사용 / Dst-solar wind relationship for storm intensity quantification | 폭풍 강도 측정 / Storm intensity measurement |
| #8 Akasofu (1964) — Substorms | 오로라 서브스톰의 형태학이 이 논문의 오로라 강수 패턴 입력의 기초 / Auroral substorm morphology underlies the precipitation pattern inputs | 오로라 패턴 기반 / Auroral pattern basis |
| Roble et al. (1987) — TIGCM | CTIM의 자매 모델, 동일한 물리를 공유. 두 모델이 독립적으로 유사한 결과를 재현 / Sister model sharing same physics; independent reproduction of similar results | 모델 상호 검증 / Model cross-validation |
| Foster et al. (1986) | 폭풍 시 전기장 패턴의 관측적 기반 — 이 논문의 전기장 입력에 직접 사용 / Observational basis for storm electric field patterns — directly used as input | 입력 데이터 / Input data |

---

## 7. References / 참고문헌

- Fuller-Rowell, T. J., M. V. Codrescu, R. G. Roble, and A. D. Richmond, "How Does the Thermosphere and Ionosphere React to a Geomagnetic Storm?", *J. Geophys. Res.*, 99(A3), 3893–3914, 1994. [DOI: 10.1029/93JA02015]
- Prölss, G. W., "Magnetic storm associated perturbations of the upper atmosphere: Recent results obtained by satellite-borne gas analyzers", *Rev. Geophys. Space Phys.*, 18, 183–202, 1980.
- Rishbeth, H., "F-region storms and thermospheric dynamics", *J. Geomag. Geoelectr.*, 43, suppl., 513–524, 1991.
- Fuller-Rowell, T. J., and D. Rees, "A three-dimensional time-dependent global model of the thermosphere", *J. Atmos. Sci.*, 37, 2545–2567, 1980.
- Fuller-Rowell, T. J., D. Rees, S. Quegan, R. J. Moffett, and G. J. Bailey, "Interactions between neutral thermospheric composition and the polar ionosphere using a coupled ionosphere-thermosphere model", *J. Geophys. Res.*, 92, 7744–7748, 1987.
- Roble, R. G., E. C. Ridley, A. D. Richmond, and R. E. Dickinson, "A coupled thermosphere/ionosphere general circulation model", *Geophys. Res. Lett.*, 15, 1325–1328, 1988.
- Foster, J. C., J. M. Holt, R. G. Musgrove, and D. S. Evans, "Ionospheric convection associated with discrete levels of particle precipitation", *Geophys. Res. Lett.*, 13, 656–659, 1986.
- Richmond, A. D., and Y. Kamide, "Mapping electrodynamic features of the high-latitude ionosphere from localized observations: Technique", *J. Geophys. Res.*, 93(A6), 5741–5759, 1988.
- Gonzalez, W. D., et al., "What is a geomagnetic storm?", *J. Geophys. Res.*, 99(A4), 5771–5792, 1994.
- Prölss, G. W., "Common origin of positive ionospheric storms at middle latitudes and the geomagnetic activity effect at low latitudes", *J. Geophys. Res.*, 98, 5981–5991, 1993.
- Allen, J. H., C. C. Abston, and L. D. Morris, "Magnetograms at geomagnetic observatories", report UAG-95, World Data Center A, 1989.
- Buonsanto, M. J., "Ionospheric Storms — A Review", *Space Sci. Rev.*, 88, 563–601, 1999.
