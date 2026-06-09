---
title: "Geomagnetic Storms and Their Impacts on the U.S. Power Grid (Meta-R-319)"
authors: John G. Kappenman
year: 2010
journal: "Metatech Corporation (prepared for Oak Ridge National Laboratory)"
doi: "NO_DOI (Meta-R-319, ORNL Subcontract 6400009137)"
topic: Space_Weather
tags: [GIC, power-grid, geomagnetic-storm, transformer-saturation, 1989-Quebec, Carrington, extreme-space-weather, infrastructure-risk]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 25. Geomagnetic Storms and Their Impacts on the U.S. Power Grid / 지자기 폭풍과 미국 전력망에 대한 영향

---

## 1. Core Contribution / 핵심 기여

### English
Kappenman's Meta-R-319 report is the **first full-scale engineering assessment quantifying the threat that extreme geomagnetic storms pose to the continental U.S. power grid**. It synthesizes a decade of Metatech modeling into a single coupled framework — (1) a data-assimilating geomagnetic storm environment model built on ~1 minute cadence magnetometer vector data, (2) a family of 1-D layered-Earth ground conductivity models (18 profiles, tiled across CONUS) that converts dB/dt into surface geoelectric field **E**, (3) a continental-scale EHV circuit model of 345/500/765 kV substations (~69,500 line-miles, 2,146 transformers), and (4) a transformer half-cycle-saturation and thermal model. The framework is validated against the 13–14 March 1989 storm — reproducing both the Hydro-Québec collapse (92 seconds, ~480 nT/min, ~1.5 V/km) and 200+ reported U.S. anomalies in four substorm intervals (7:40–8:00, 10:50–12:00, 21:20–22:30, 0:30–2:00 UT). The framework is then extended to **Carrington-class scenarios** parameterized by the May 1921 benchmark (~5,000 nT/min, ~20 V/km observed on the Stockholm–Toreboda rail circuit). For a 4,800 nT/min westward-electrojet footprint (~120° longitude × 5–10° latitude) centered on 50° geomagnetic latitude — a "1-in-100 year" event — simulation predicts cumulative MVAR demand >100,000 MVAR (10× the March 1989 peak), system collapse across PJM, ECAR, SERC, NPCC, NE-Quad, and the Pacific Northwest, and ≥300 permanently damaged EHV transformers. Because EHV transformers are custom-built with 12–24 month lead times and world annual production is lower than the simulated loss, long-duration blackouts (months to years) affecting >130 million people become plausible.

### 한국어
Kappenman의 Meta-R-319 보고서는 **극한 지자기 폭풍이 미국 본토 전력망에 미치는 위협을 최초로 본격적으로 정량화한 엔지니어링 평가**다. Metatech가 10년 이상 축적한 모델들을 하나의 결합 프레임워크로 통합: (1) 1분 단위 자력계 벡터 데이터를 동화(assimilation)한 지자기 폭풍 환경 모델, (2) 18개 프로파일로 구성된 1-D 층상 지층 전도도 모델 패밀리(미 대륙 전역을 타일링), (3) 345/500/765 kV 변전소 기반의 대륙급 EHV 회로 모델(송전선 약 69,500 마일, 변압기 2,146기), (4) 변압기 반-주기 포화 및 열 모델. 이 프레임워크는 1989년 3월 13–14일 폭풍에 대해 검증되었다 — Hydro-Québec 붕괴(92초, ~480 nT/min, ~1.5 V/km)와 미국에서 관측된 200여 건의 이상(7:40–8:00, 10:50–12:00, 21:20–22:30, 0:30–2:00 UT의 4개 substorm 구간)을 모두 재현. 이어서 1921년 5월 폭풍(~5,000 nT/min, Stockholm–Toreboda 철도 통신선에서 관측된 ~20 V/km)을 벤치마크로 **Carrington급 시나리오**로 확장했다. 50° 지자기 위도 중심의 4,800 nT/min 서향 electrojet 풋프린트(~120° 경도 × 5–10° 위도) — "100년 재현주기" 이벤트 — 시뮬레이션은 누적 MVAR 수요 >100,000 MVAR(1989년 3월 피크의 10배), PJM·ECAR·SERC·NPCC·북동 Quad·태평양 북서부에 걸친 계통 붕괴, 그리고 **≥300기의 EHV 변압기 영구 손상**을 예측한다. EHV 변압기는 주문 제작이고 리드타임 12–24개월이며 전 세계 연간 생산량이 시뮬레이션 손실보다 적기 때문에, 수개월~수년의 장기 정전(1.3억 명 이상 영향)이 물리적으로 가능한 시나리오가 된다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Section 1 — U.S. Power Grid Model for Geomagnetic Storm Threat Environments / 지자기 폭풍 위협 환경에 대한 미국 전력망 모델

**모델 구조 / Model architecture (p. 1-1 to 1-3)**

**한국어**: 이 보고서의 뼈대는 네 개의 결합 모델이다:
1. 지자기 폭풍 환경 모델 (Section 1.1)
2. 지층 전도도 모델 → 지전장 계산 (Section 1.2)
3. 미국 EHV 전력망 회로 모델 (Section 1.3)
4. 변압기 및 AC 전력망 성능 모델 (Section 1.4)

Figure 1-1은 1933–2008년간 태양 흑점 주기와 대형 폭풍(Ap > 50)의 발생을 겹쳐 그린다. 중요한 관찰: **큰 폭풍은 solar max에만 국한되지 않는다** — 예를 들어 1986년 2월 폭풍은 Cycle 21/22 minimum에서 발생. Figure 1-2는 미국 고전압 송전망이 50년간 **10배 성장**했음을 보여준다 (Cycle 19 → Cycle 22). 즉, "안테나"(지전장에 결합되는 송전선)가 기하급수적으로 커진 반면, 우주기상은 변하지 않았다. 이 비대칭성이 취약성 증가의 근본 원인이다.

**English**: The report's backbone is four coupled models. Figure 1-1 (1933–2008 sunspot + Ap overlay) establishes that **large storms are not confined to solar max** — the February 1986 storm, for example, occurred at Cycle 21/22 minimum. Figure 1-2 shows the U.S. HV transmission network grew nearly **tenfold** between Cycle 19 and Cycle 22. The "antenna" coupling to the geoelectric field has expanded dramatically while space weather has not, producing the asymmetric rise in vulnerability that this report exists to quantify.

**§1.1 Storm Environment Model / 폭풍 환경 모델 (p. 1-4 to 1-6)**

**한국어**: 기존의 "단순 평면파 모델"은 미 대륙 규모에서는 부적절하다. 전리층 electrojet(북극 지역 고도 ~100 km, 전류 ~백만 암페어)이 폭풍 중 강화·이동하며 발생시키는 지자기 교란은 시공간적으로 복잡하다. Metatech는 **data-assimilation 접근**을 취한다: 북미 전역의 자력계 벡터 데이터(Figure 1-3의 May 10, 1992 09:10 UT 예시)를 이용해 electrojet 구조를 보간(Figure 1-4). 자력계는 보통 1분 cadence이나, 일부 이벤트는 1–3초 데이터도 가용.

**English**: The "simple plane-wave" assumption fails at continental scales. Auroral electrojet currents (~100 km altitude, ~1 million amps) intensify and move during storms, producing spatio-temporally complex ground disturbances. Metatech's solution is data assimilation: magnetometer vector observations across North America (Figure 1-3 at 09:10 UT on May 10, 1992) are interpolated to produce a gridded environment (Figure 1-4). Cadence is typically 1 minute; some events provide 1–3 second data.

**§1.2 Ground Conductivity & Geoelectric Field / 지층 전도도 및 지전장 (p. 1-7 to 1-11)**

**한국어**: 지전장 계산에 필요한 전도도 심도는 **300 km 이상**. 저주파 폭풍 성분(0.00001 Hz ~ 0.3 Hz)이 깊게 침투하기 때문이다. 표층 전도도는 위치별 편차가 3–5 orders of magnitude로 크지만, 심부는 비교적 균질. 1-D layered-Earth 모델 18개를 CONUS에 타일링(Figure 1-6의 색 지도). **핵심 결과(Figure 1-5)**: 동일한 2400 nT/min electrojet 위협 아래, 가장 반응성 높은 지층(SOS2)은 **~16 V/km**, 가장 둔한 지층(BCA2)은 ~2 V/km → **7배 이상의 편차**. 검증: (1) 1993년 11월 4일 미네소타 폭풍의 관측 E vs 계산 E 파형 비교(Figure 1-7), (2) 1998년 5월 4일 메인 주 Chester substation의 GIC 측정 vs 시뮬레이션(Figure 1-8, ~60초 cadence 차이 감안 시 상당한 일치).

**English**: Ground conductivity must be known to **>300 km depth** because the low-frequency content (0.00001–0.3 Hz) penetrates deeply. Shallow conductivity varies by 3–5 orders of magnitude laterally, but deeper layers are more uniform. Eighteen 1-D profiles tile CONUS (Figure 1-6). **Key result (Figure 1-5)**: under the same 2400 nT/min electrojet threat, the most responsive profile (SOS2) produces ~16 V/km, the least responsive (BCA2) ~2 V/km — a >7× range. Validation comes from (1) a Minnesota storm on Nov 4, 1993 (Fig 1-7) and (2) Chester, Maine GIC measurements vs simulation during May 4, 1998 (Fig 1-8).

지전장 계산의 핵심 수식 (주파수 영역, 평면파 근사):

$$
E_x(\omega) = Z(\omega)\,H_y(\omega), \qquad Z(\omega) = \sqrt{\frac{i\omega\mu_0}{\sigma_{\text{eff}}(\omega)}}
$$

- $Z(\omega)$: 주파수 의존 표면 임피던스 (Ω). 저전도(암반) 지역일수록 $|Z|$가 크고, 따라서 같은 자기장 변동에서도 더 큰 E를 생성. / Frequency-dependent surface impedance; resistive shields produce larger E for the same dB/dt.
- $\sigma_{\text{eff}}$: 층상 구조를 반영한 유효 전도도.

**§1.3 U.S. Grid Circuit Model / 미국 전력망 회로 모델 (p. 1-12 to 1-19)**

**한국어**: CONUS 모델은 **345 kV 이상만** 포함 (저전압은 GIC 흐름이 미미). Figure 1-9/1-10: 345 kV는 45,626 line-miles (64%), 500 kV는 23,812 (33%), 765 kV는 2,064 (3%). 765 kV는 Illinois, Ohio, Indiana, West Virginia, 업스테이트 뉴욕에 집중. 변압기 수(Figure 1-11): 345 kV 1,501기, 500 kV 587기, 765 kV 58기 — **총 2,146기**. Figure 1-12는 송전선 저항이 전압과 함께 감소함을 보여준다: 115 kV ≈ 0.1 Ω/mile → 765 kV ≈ 0.01 Ω/mile (10배 감소). Figure 1-14는 평균 선로 길이가 전압과 함께 증가: 69 kV 14.5 miles → 765 kV **64.7 miles**. 즉 **고전압 = 긴 선로 + 낮은 저항 → 높은 GIC**. 이중 곱.

**English**: Only ≥345 kV is modeled because lower-voltage segments carry negligible GIC. The continental picture: 2,146 transformers, ~69,500 line-miles. Two compounding trends push GIC into the highest voltages: resistance drops ~10× from 115 kV (~0.1 Ω/mile) to 765 kV (~0.01 Ω/mile) (Fig 1-12), while average line length rises from 14.5 miles at 69 kV to **64.7 miles at 765 kV** (Fig 1-14). Long + low-resistance = more GIC per unit geoelectric field.

직렬 커패시터(series capacitors)는 DC-like GIC를 **차단**한다. 동부 미국 grid에는 2개만 있음; WECC(서부)는 500 kV 선로의 ~55%, 345 kV의 ~25%가 직렬 보상 → 서부가 구조적으로 덜 취약. / Series capacitors block DC-like GIC. The eastern grid has only two series-compensated lines; WECC has ~55% (500 kV) / ~25% (345 kV), making the western grid structurally less vulnerable.

**선로 유도 전압 / Induced line voltage**:

$$
V_{\text{line}} = \int_L \mathbf{E}\cdot d\boldsymbol{\ell} \approx E L \cos\theta
$$

**GIC 회로 방정식 (simplified) / Simplified GIC circuit**:

$$
I_{\text{GIC}} = \frac{V_{\text{line}}}{R_{\text{line}} + R_{\text{wind,1}} + R_{\text{wind,2}} + R_{\text{gnd,1}} + R_{\text{gnd,2}}}
$$

리액턴스는 무시 (GIC는 준-DC). 실제 모델은 continental-scale Kirchhoff nodal 해석. / Reactances ignored since GIC is quasi-DC; the actual model solves Kirchhoff's laws over the whole grid.

**§1.4 Transformer & AC Performance Model / 변압기 및 AC 성능 모델 (p. 1-20 to 1-29)**

**한국어**: 이 섹션이 본 보고서의 "심장"이다. GIC가 변압기에 들어오면 권선에 DC 바이어스가 걸려 자속이 한쪽으로 편향 → 매 AC 사이클의 **반주기마다 철심이 포화(half-cycle saturation)** → 그 결과:
1. **무효전력(MVAR) 흡수 폭증**: Figure 1-17에서 500 kV 단상 변압기는 100 A/phase GIC에서 ~170 MVAR, 3상 3-legged core 형은 ~40 MVAR — **단상이 4배 더 취약**.
2. Figure 1-18: 동일 GIC에서 765 kV 단상은 345 kV 단상의 **2배** MVAR. 즉 **고전압이 더 치명적**.
3. **변압기 설계 인구통계**: 345 kV는 15% 단상 / 85% 3상 (Fig 1-19), 500 kV는 66% 단상 / 34% 3상 (Fig 1-20), 765 kV는 **97% 단상** (Fig 1-21). 따라서 765 kV는 거의 전량 취약형.
4. BPA(태평양 북서부) 검증: BPA 500 kV의 97%가 단상 → 보고서의 전체 미국 66% 추정치가 오히려 **낙관적일 수 있음** (Fig 1-22).
5. **고조파(harmonics) 주입**: Figure 1-24는 정상 500 kV 여자전류(< 1 A); Figures 1-25/26/27은 GIC 5/25/100 A/phase에서 여자전류가 85 A, 290 A, **>800 A**로 폭증 (총 왜곡률 >200%, THD 93–96%). Figure 1-28: 정상 부하 300 A에 GIC 150 A/phase가 걸리면 총 전류 파형 왜곡 274%.

**English**: This is the heart of the report. DC bias from GIC tips the core's magnetization curve asymmetrically, saturating the core on one half of each AC cycle. Consequences:
1. **MVAR demand spikes**: single-phase 500 kV at 100 A/phase GIC → ~170 MVAR; 3-leg 3-phase → ~40 MVAR. Single-phase designs are ~4× worse (Fig 1-17).
2. Higher kV compounds it: 765 kV single-phase draws **2× more** MVAR than 345 kV single-phase at the same GIC (Fig 1-18).
3. **Design population** (Figs 1-19 to 1-21): 345 kV = 15% single-phase, 500 kV = 66% single-phase, **765 kV = 97% single-phase**. The vulnerable design dominates precisely where voltages are highest.
4. BPA check: 97% of BPA's 500 kV is single-phase, suggesting the report's 66% national estimate is **if anything optimistic**.
5. **Harmonic injection**: normal 500 kV excitation current is <1 A (Fig 1-24). At 5 / 25 / 100 A/phase GIC, excitation current peaks at 85 / 290 / **>800 A** respectively (Figs 1-25–27). Total waveform distortion exceeds 200%; THD reaches 93–96% (Fig 1-28).

**반-주기 포화 물리식 / Half-cycle saturation physics**:

$$
B(t) = B_{\text{AC}}\sin(\omega t) + B_{\text{DC}},\quad B_{\text{DC}} \propto \frac{N \cdot I_{\text{GIC}}}{\mathcal{R}_{\text{core}}}
$$

포화 상태에서 국부 과열의 누적 열 방정식 / Cumulative hot-spot heating under saturation:

$$
T_{\text{hot}}(t) = T_0 + \int_0^t \frac{P_{\text{leak}}(\tau)}{C_{\text{th}}}\,d\tau
$$

$P_{\text{leak}}$은 포화로 코어 밖으로 나간 누설 자속이 tie-plate·탱크 벽에서 유도하는 와전류 손실. GIC 크기의 거의 2제곱에 비례. $C_{\text{th}}$는 열용량. 기름 순환 냉각이 따라가지 못하면 분 단위로 >200°C 가능.

**§1.5 Evolving Vulnerability (p. 1-29 to 1-31)**

**한국어**: **결정적 수치들**: 1989년 3월 Québec 붕괴 → ~480 nT/min (지역 최대), 지전장 ~1.5 V/km. **하지만 같은 폭풍의 유럽 측**에서는 Stockholm 근처에서 **~2,000 nT/min** 관측 (2배 이상). 1972년 8월 4일 AT&T L4 telecom cable 장애(일리노이) → E ≥ 7 V/km 추정. 1982년 7월 13–14일 중부 스웨덴 → **2,700 nT/min**, Stockholm–Toreboda 철도 회로에서 9.1 V/km 관측. **1921년 5월 폭풍**(같은 Swedish 회로) → **~20 V/km**. "이 강도는 1989년을 precipitate한 수준의 **10배**." → 과거 재현주기에 근거해 **~5,000 nT/min**이 물리적으로 가능하다는 것이 본 보고서의 extreme 시나리오 근거. N-1 기준의 한계: 기상재해와 달리 GMD는 **near-simultaneous multipoint** 장애를 초래 → 기존 설계 철학이 깨진다.

**English**: Key numbers. March 1989 Québec: ~480 nT/min, ~1.5 V/km; **Europe side of the same storm**: ~2,000 nT/min near Stockholm. August 4, 1972 AT&T L4 telecom cable failure (Illinois): E ≥ 7 V/km. July 13–14, 1982 mid-Sweden: **2,700 nT/min**, 9.1 V/km on the Stockholm–Toreboda rail circuit. **May 1921, same circuit: ~20 V/km** — "ten times larger than the levels that precipitated the March 1989 power-system impacts." This, plus historical return-period reasoning, underwrites the report's extreme-scenario choice of ~5,000 nT/min. Standard N-1 design assumes localized contingencies; GMD produces **near-simultaneous multipoint failures**, defeating that philosophy.

### Part II: Section 2 — March 13–14, 1989 Forensic Analysis / 1989년 3월 폭풍 포렌식 분석

**Section opener (p. 2-1 to 2-4)**: 02:44 EST, 미·캐나다 국경에서 electrojet impulse가 발생. 92초 내 Hydro-Québec 전역 붕괴. Figure 2-1의 2:43–2:46 EST 4프레임 시퀀스는 **ground-level geomagnetic intensification**이 4분 동안 동서로 퍼지는 모습을 보여준다. **Regional GIC Index (RGI)**: dB/dt(nT/min)를 지역별 GIC 위협 프록시로 사용. Figure 2-3은 48시간 NY/NE/Canada RGI plot — 02:45 EST 피크 ~400 nT/min(Québec 붕괴 유발), 16:00–22:00 EST 사이 여러 피크에서 동부 미국 이상 발생. Figure 2-4의 지역별 최대 RGI: 남 매니토바 892 nT/min @ 7:45 UT(3/13), 북 뉴욕 556 nT/min @ 1:21 UT(3/14), 메릴랜드/버지니아 353 @ 22:12, 미시시피 461 @ 1:20 UT, 태평양 북서부 381, 콜로라도 332. **미국 전역이 위협 환경에 노출됨을 보여주는 관측 근거**.

**English**: At 02:44 EST, an electrojet impulse erupted along the U.S./Canada border. Hydro-Québec collapsed completely in 92 seconds. Figure 2-1's 2:43–2:46 EST sequence shows ground-level geomagnetic intensification spreading east-west over four minutes. The **Regional GIC Index (RGI)** uses local dB/dt (nT/min) as a proxy for regional GIC threat. Figure 2-3 is the 48-hour NY/NE/Canada RGI plot — 02:45 EST spike ~400 nT/min (trigger of Québec collapse); multiple later spikes during 16:00–22:00 EST drove eastern U.S. anomalies. Peak RGIs by region (Fig 2-4): southern Manitoba 892 nT/min @ 7:45 UT, northern NY 556 @ 1:21 UT, MD/VA 353 @ 22:12, Mississippi 461 @ 1:20 UT, Pacific NW 381, Colorado 332. The storm blanketed the continent.

**§2.1 Québec Collapse (p. 2-5 to 2-13)**: GIC가 변압기에 들어오면 네 가지 serious impact를 유발:
1. 여자전류 첨두가 반주기에 치솟아 과열·내부 손실 증가
2. 시스템 전체에 짝수·홀수 고조파 주입 → capacitor bank, SVC, protective relay 오동작
3. 무효전력 수요 급증 → 전압 조절 문제 → 전압 붕괴
4. **누설 자속(stray flux)으로 인한 국부 과열 → 영구 손상**

Québec의 취약성 요인(모두 최악): 북위(aurora 근처), 저전도 암반, 735 kV 초고전압(높은 GIC), **단상 변압기 집중**, **SVC에 의존하는 전압 조절**. 붕괴 시퀀스(Figure 2-7/2-9/2-10):
- 7:43 UT: storm intensity 가속 시작
- 7:44:16 UT: Chibougamau SVC #12 trip
- 7:44:19 UT: Chibougamau SVC #11 trip
- 7:44:33–46 UT: Albanel/Nemiscau 4 SVC trips
- 7:45:16 UT: LaVerendrye SVC #2 (7번째 SVC) trip → **모든 7개 SVC 상실**
- 7:45:25 UT: 5개 735 kV James Bay tie line이 out-of-step relay로 trip (9,500 MW 발전 상실)
- 7:45:31 UT: Arnaud-Churchill Falls lines trip
- **7:45:49 UT: 전체 붕괴**

정상 상태에서 붕괴까지 **92초**. Figure 2-7: 7:39–7:46 UT에 변압기 평균 GIC가 ~1 → 12 A/phase, 시스템 MVAR 수요가 ~0 → 1,600 MVAR로 상승. **Table 2-1**: Québec 복구 — 02:45 collapse, 07:00 25% (5,000 MW), 09:00 48%, 13:00 83% (17,500 MW). **관측**: 대부분의 미국 발전은 증기 터빈이므로 미국 재시작은 훨씬 어렵다.

**English**: GIC in a transformer produces four impacts: (1) excitation current peaks in half-cycles causing heating; (2) even/odd harmonics cascade through the grid, misoperating capacitor banks, SVCs, and relays; (3) MVAR demand spikes cause voltage collapse; (4) stray flux causes localized overheating → **permanent damage**. Québec's vulnerability was maximal: high latitude, resistive igneous rock, 735 kV EHV, dominant single-phase transformers, reliance on SVCs. The collapse sequence (Figs 2-9, 2-10) ran from 7:44:16 UT (first Chibougamau SVC trip) to **7:45:49 UT (full collapse)** — 92 seconds from normal state. Figure 2-7 tracks average GIC rising 1 → 12 A/phase and system MVAR 0 → 1,600 MVAR in 7 minutes. Table 2-1: Québec restored 83% load (17,500 MW) by 13:00 — a hydro-dominant grid's quick restart, which most of the U.S. (steam-electric dominated) cannot replicate.

**§2.2 U.S. Grid Impacts — 4 substorm intervals (p. 2-14 to 2-28)**:

| Interval (UT) | 특징 / Features | 미국 임팩트 / U.S. Impact |
|---|---|---|
| **7:40–8:00** | Québec 붕괴 구간; 남 매니토바/북 미네소타 최대(~892 nT/min); Manitoba 500 kV substation에서 synchronous condenser의 MVAR가 **420 MVAR** 증가; Fargo SVC trip | HQ-NY 765 kV tie 상실(1,949 MW 수입 손실); US는 minimum load 상태라 비교적 소규모 |
| **10:50–12:00** | westward electrojet 재강화, 강도는 이전의 20–50% 수준이나 footprint가 동부 US로 확장; Figure 2-14의 11:26 UT sim | Fig 2-15: 6:06–6:30 EST에 NIMO, PJM, Va. Power, APS 등에서 capacitor bank 다수 스위칭 |
| **21:20–22:30** | **가장 심각한 US 구간**. eastward electrojet가 중위도 US로 내려옴(Fig 2-18의 22:00 UT sim). 미국 전체 피크 **부하 시간**과 겹침 | Fig 2-16: LA까지 확장된 40여 건 이상(capacitor, voltage, transformer, converter, alarm). Fig 2-19/20/21/22 시뮬: PJM ~940 amps·1,550 MVAR; SERC 1,060·1,780; ECAR 2,420·2,200; WECC 1,700·2,100. 총 미국 **GIC ~8,200 amps, MVAR ~8,100** (Fig 2-25) |
| **0:30–2:00 UT (3/14)** | 서향 electrojet 강화 + Florida panhandle, 미국-멕시코 국경 지역에도 교란. 461 nT/min @ Bay St. Louis, 미시시피 | NE-MAPP-Midwest 지역 저녁 피크부하 시 다수 이벤트(Fig 2-23). NYISO 피크 GIC 890 amps, 950 MVAR @ 1:17 UT |

**Figure 2-25의 의미**: 4개 구간 누적 GIC & MVAR sum은 **21:20–22:30 UT가 최대** (3/14의 0:30–2:00 UT가 그 다음). 이것은 직관과 다름 — Québec을 collapse시킨 7:40–8:00 구간은 미국 측에서 상대적으로 약함. **teaching point**: Québec 붕괴는 Québec의 취약성(단상·SVC 의존)이 결합한 결과이지, 단일 "극대" 폭풍 구간이 전국을 동시 파괴한 것이 아니다.

**§2.3 Transformer Internal Heating / 변압기 내부 과열 (p. 2-29 to 2-34)**:

**Salem Nuclear Plant GSU (N.J., 500 kV 연결)**: Figure 2-33의 사진은 22 kV 저압 권선의 melting과 탄화된 종이 절연. 원래 3,000 A 정격 권선이 녹을 정도의 과열. Figure 2-34: Salem GIC 시뮬레이션 — 4개 substorm 구간 중 **21:44 UT에 90 A/phase 피크**. 이 변압기 위치가 남부 mid-lat이라 동부 electrojet 강화 시간에 최대 노출. Fortunate한 점: 22 kV 저압 권선에서 손상이 국한됨(500 kV 주 권선은 생존). 사용 중단은 폭풍 다음날(3/14).

**Meadowbrook 500 kV / Allegheny Power (버지니아)**: 전 세계 몇 안 되는 **GIC + 탱크벽 온도 직접 측정** 사례. 1989 폭풍에서 변압기 탱크 외부 페인트 4곳이 blistered → 추정 열 400°C. 1992년 5월 10일(훨씬 작은 폭풍)에 직접 계측(Fig 2-35): neutral GIC ~60 A, **hot-spot 온도 top-oil 대비 ~10분 내 175°C 급상승**. Top-oil 온도는 거의 변화 없음 — **국부 hot-spot이 벌크 온도보다 먼저 한계에 도달**.

**제어된 DC 주입 테스트(Fig 2-36)**: 370-MVA 3상 변압기에 12.5 A → 75 A DC 주입. 75 A 주입 시 "top of tie plate" 위치가 3분 만에 85°C까지 급등. 핵심 발견: **GIC 크기 > GIC 지속시간** (큰 GIC가 짧아도 위험; 작은 GIC는 오래 가도 상대적 안전).

**광범위한 후속 피해**: 1989 이후 2년 이내에 미국 원자력 발전소 11곳에서 GSU 변압기 장애 보고. 1994년 4월 3일 Zion Nuclear (Chicago 근교) GSU 대형 파괴 → 변압기 탱크 파열, 수천 갤런 기름 유출 + 대형 화재. 4월 5일 Braidwood, 4월 15일 Powerton에서도 동일 operator 변압기 장애 — "우연의 일치"로 보기 어려움. **Transformer 교체 난이도**: 600 MVA+ 변압기는 신규 제조 리드타임 12개월+, 이송·재조립에 수주. 지금 없는 spare 변압기는 다시 사오기 어려움.

**§2.3 English summary**: The Salem Nuclear GSU (500 kV, N.J.) lost low-voltage (22 kV) windings to overheating; Figure 2-34 shows a 90 A/phase peak at 21:44 UT. The Meadowbrook 500 kV transformer (Allegheny Power, VA) is a rare direct-measurement case: external tank paint blistered (~400°C) after March 1989; during a smaller May 10, 1992 storm (Fig 2-35) neutral GIC ~60 A produced a **175°C hot-spot rise in ~10 minutes** while top-oil barely moved. Controlled DC injection tests (Fig 2-36): a step from 12.5 A to 75 A produced 85°C tie-plate temperature in three minutes. **Magnitude matters more than duration.** Eleven nuclear plant GSU failures in two years after March 1989 plus the April 1994 Zion/Braidwood/Powerton cluster reinforce the causal link.

### Part III: Section 3 — Extreme Storm Threat Assessment / 극한 폭풍 위협 평가

**§3.1 Storm climatology / 폭풍 기상학 (p. 3-1 to 3-13)**:

**한국어 핵심**: 폭풍 severity의 가장 의미 있는 지표는 **dB/dt**이다 (Ap, Dst, Kp는 planetary average로 위도·국부 정보를 잃음).
- March 1989: Dst ~-600, North America 피크 dB/dt ~900 nT/min, 유럽 측 BFE ~2,000 nT/min
- August 1972: North America에서 ~2,200 nT/min, J.M. Stuart Station(Ohio) 변압기 neutral GIC가 **>100 A로 off-scale**
- July 1982: 중부 스웨덴 ~2,700 nT/min, Stockholm–Toreboda 9.1 V/km
- May 1921: Stockholm–Toreboda **~20 V/km** (기록된 최대 geo-potential), Dst 추정 ~-1,000, dB/dt 추정 ~5,000 nT/min
- 1859 Carrington: Dst ~-1,760 (1989의 ~3배; Tsurutani et al.), 1921과 유사한 dB/dt 가능성

**결론적 재현주기**: 2,000+ nT/min급 North America 이벤트는 과거 30년에 최소 3회 → 1-in-10 year. ~5,000 nT/min급 (May 1921)이 다시 오면 **~1-in-100 year**로 평가.

**§3.1 English**: The authoritative severity metric is **dB/dt**, not planetary averages like Ap/Kp/Dst. Historical benchmarks: March 1989 ~900 nT/min (NA) but ~2,000 nT/min (Europe); August 1972 ~2,200 nT/min (NA, with a >100 A off-scale GIC at J.M. Stuart, Ohio); July 1982 ~2,700 nT/min (central Sweden, 9.1 V/km); **May 1921 ~5,000 nT/min, 20 V/km** — the largest known geo-potential measurement. 1859 Carrington Dst ~-1,760 (Tsurutani et al.), ~3× larger than March 1989. The report places 2,000+ nT/min events at ~1-in-10 year return, ~5,000 nT/min at **~1-in-100 year**.

**Figures 3-8/3-9 (IMF coupling)**: July 15–16, 2000 이벤트 예시로 **coupling efficiency**가 3–85%로 변동함을 보여준다. "Perfect Storm"은 태양 ejecta의 크기 + IMF Bz southward + 지속시간의 드문 convergence. X22+ 플레어(2001/4/2)는 1989보다 **30배 크지만** 지구 방향이 아니어서 소규모 영향. "solar flare magnitude ≠ storm severity."

**§3.2 Extreme scenarios / 극한 시나리오 (p. 3-14 to 3-28)**:

**핵심 manipulation**: July 13–14, 1982 이벤트의 observed 23:54 UT electrojet 구조(Fig 3-11)를 가져와서 120° 경도 회전 + 5–10° 남향 stretch (Figs 3-12, 3-13) → **"1982년 구조를 북미 위에 그대로 얹으면 어떻게 될까?"** 시뮬레이션:
- No shift (Fig 3-14/15): 미국 GIC/MVAR 피크 ~11,000 MVAR (March 1989의 8,000 대비 **37% 더 큼**)
- 120° longitude shift (Fig 3-17): 16,000 MVAR (**March 1989의 2배**)
- 120° long + 5° lat shift (Fig 3-18/19): **32,000 MVAR** (4배)
- 120° long + 10° lat shift (Fig 3-20/21): 일부 plateau

**계단식 시나리오** (2,400 / 3,600 / 4,800 nT/min, 40°/45°/50°/55° 지자기 위도 배치). Figure 3-23 — **Comparison of US Peak MVAR Demand Increases**:
- March 1989 기준: ~10,000 MVAR
- 2,400 @ 50°: ~55,000 MVAR (1-in-30 year)
- 4,800 @ 50°: **~110,000 MVAR** (1-in-100 year, Carrington-like) → **1989의 10배+**

**Figures 3-25/3-26**: 100년 시나리오 50° 및 45° 위도 붕괴 footprint 지도. 50°: 북동부 + 태평양 북서부. 45°: 남부·서부로 확장. **45°가 overall disturbance energy는 낮아도 outage region이 더 크다** — 고전압 기반의 intrinsic susceptibility 때문.

**Low-latitude risk**: 남 일본 11월 6, 2001 폭풍(Fig 3-27)에서도 중규모 GIC 관측. Fig 3-28: Dst vs GIC 선형 trend → Dst -1,700 시나리오 추정. 남아공 Eskom grid가 2003년 10월 폭풍에서 **14 기 400 kV 변압기 상실** — 저위도도 장시간 노출 시 위험.

**§3.2 At-risk transformer count (p. 3-27 to 3-29)**: Salem GSU 경험(~950 amp-min 누적 노출 → 대파)을 **damage threshold 프록시**로 사용. 2,400 nT/min × 3/13/89 보조 substorm 조합을 전국에 적용한 시뮬레이션 결과 — **~216 EHV 변압기**가 Salem 수준을 초과하는 GIC 누적 노출을 받을 것. Figure 3-29의 미국 지도가 이들의 위치 분포를 보여준다(주로 동부).

### Part IV: Section 4 — At-Risk EHV Transformers and Damage Estimates / 취약 EHV 변압기와 피해 추정

**§4.0 Overview (p. 4-1 to 4-3)**: 최악 시나리오에서 "**순식간에 70%+ 전력 공급 상실**, 2003년 8월 북미 대정전보다 몇 배 큰 blackout". 진짜 문제는 **복구 시간**: 변압기 영구 손상으로 수개월~수년. Eskom (남아공, 지자기 위도 -27 ~ -34°)은 2003년 10월말-11월초 **100 nT/min 미만** 교란에서 400 kV 변압기 15기 상실 — 저위도조차 취약함을 증명.

**EHV age data**: Figure 4-2의 ECAR 지역 데이터 — 가중 평균 사용연수 **>30년** (설계 수명 ~40년). 노후화가 GIC 취약성을 더 높인다.

**§4.1 Heating thresholds (p. 4-4 to 4-10)**:
- **Meadowbrook 5/10/92** (Fig 4-3): neutral GIC ~60 A에서 탱크 hot-spot 175°C (10분 내).
- **dB/dt at Fredericksburg for 5/10/92** (Fig 4-4): peak **~55 nT/min**. **극한 시나리오는 이것의 ~100배** → GIC도 ~100배 예상.
- **Controlled DC test** (Fig 4-5, 370 MVA, Hydro-Québec): 12.5 A → 75 A DC 주입 시 top-of-tie-plate 85°C, 3분 만에 급등.
- **New Zealand SSC failure** (Fig 4-6, Nov 6, 2001): Sudden Storm Commencement의 급격한 rise가 변압기 즉시 고장 유발. **SSC는 E3 HEMP와 유사한 onset**.
- **Salem GSU dB/dt/GIC** (Fig 4-7/4-8): FRD 피크 dB/dt ~470 nT/min @ 21:44 UT; GIC 피크 ~90 A/phase. 지속 시간은 < 2분이었지만 영구 손상 유발.
- **Girgis et al. analytical** (Fig 4-9): Salem-class design은 30 A/phase 이상이면 부하 0 만으로도 영구 손상 위험.
- **Hurlet table** (Fig 4-10): 최적화 미비 설계는 100 A GIC에서 **0분 버팀(즉시 실패)**; 50 A → 3분; 25 A → 33분. 최적화된 설계는 100 A에서 23분, 50/25 A는 >2시간.

**§4.2 Impacts under 4800 nT/min @ 50° lat (p. 4-11 to 4-17)**:

두 임계치 비교:
- **90 A/phase 임계** (보수적, NAS/FEMA 분석에 사용): Table 4-1/4-2/4-3
- **30 A/phase 임계** (Salem GSU 실측 기반, 덜 보수적)

| Class | 90 A 임계 / 90A threshold | 30 A 임계 / 30A threshold | 증가 / Increase |
|---|---|---|---|
| **345 kV** | 214기, 156,423 MVA (20.0%) | 677기, 400,476 MVA (51.2%) | 216% in MVA |
| **500 kV** | 137기, 131,353 MVA (28.4%) | 285기, 245,317 MVA (53.0%) | 108% in MVA |
| **765 kV** | 17기, 20,881 MVA (31.8%) | 49기, 49,876 MVA (76.0%) | 141% in MVA |
| **합계 / Total at 30A** | — | **~1,011 transformers** | — |

Figure 4-11 지도: 4,800 nT/min @ 50°에서 30 A/phase 이상 GIC 노출 변압기들 — **대부분 동부**(IL, IN, OH, PA, NY, NJ, MD). Figure 4-12의 GIC ranking curve: 상위 200 EHV 변압기에서 March 1989는 < 200 A/phase, **극한 시나리오는 최대 1,800 A/phase** — 10배.

**NE-Quad (북동 미국)의 GSU 집중**: 30 A 임계 기준 128,000 MVA의 GSU가 at-risk (NE-Quad GSU의 **~63%**). 전체 NE-Quad 발전 용량의 ~30% 해당. Figure 4-13(30 A): Coal 43%, Nuclear 38%, NG 9%, Oil 8%. **원자력 GSU 상실은 특히 치명적** (~92%의 NE-Quad 원자력이 장기 off-service). Figure 4-14(90 A): Nuclear 47%, Coal 30%.

**§4.3 Replacement logistics (p. 4-18 to 4-22)**:

**한국어**: 300–400 MVA급 EHV 변압기 주문 리드타임 **15개월+**. 큰 용량은 수개월 추가. 운송 자체가 별개의 문제:
- 중량 1,000,000 lb+, 특수 trailer(300 ft, 19 axles, 6 drivers)
- 해외 수송 시 수주 해양 운송
- 항만·고속도로·지자체 허가 필요
- 현장 도착 후 bushing 재장착, 오일 채우기, 진공, 펌프 장착, relay 배선 → 숙련팀 수일
- 교통신호 철거·재설치, 교량 load-bearing 인증 → 6개월 전 notice 관행
- 대용량 ridge 통과에 **Schnabel car**(특수 철도차량) 필요

**STEP program** (NERC Spare Transformer Equipment Program): ~170기 commitment하나 **765 kV 미포함, GSU 미포함** → 정확히 본 보고서가 걱정하는 범주 제외. **EPRI/DHS의 Rapidly Deployable Transformer 이니셔티브**(Stiegemeier & Girgis 2006)는 테러 공격 대응용으로 개발되었으나 GSU 미포함 — GMD 시나리오에는 불충분.

Figure 4-16: 뉴욕·뉴잉글랜드·펜실베니아 지역 변압기의 **권선 구성·전압비 다양성** (345 GSU 17%, 500/230 42%, 500 GSU 16%, 500/345 10%, 345/115 7% 등) → "공유 spare" 전략이 거의 작동하지 않는 이유.

### Part V: Appendix A1 — Disturbance Impact Criteria (p. A1-1 to A1-12)

기존 **N-1** 설계 기준: 단일 contingency(선로 1개, 변압기 1개)에 내성. Québec에서는 **N-7이 92초 내**에 발생 → N-1 설계 철학으로는 방어 불가. 위협 강도에 따른 회복 능력 확률 분석(Figs A1-3~A1-5): 피크 부하 시 시스템이 가장 취약. Fig A1-8: Three Mile Island에서 **GIC 왜곡된 capacitor bank current** — 정상 사인파 대비 수십배 THD → capacitor bank 보호 회로 오동작 유도.

**English**: The N-1 design criterion (withstand any single contingency) fails against GMD. Québec experienced **N-7 in 92 seconds**. Peak-load periods compound the risk (Fig A1-2). Harmonic injection distorts capacitor bank currents (Fig A1-8, TMI example), tripping protection incorrectly and removing reactive support when most needed.

---

## 3. Key Takeaways / 핵심 시사점

1. **공간적 dB/dt가 모든 것을 결정한다 / dB/dt is the master variable**
   - GIC, E-field, transformer saturation, 시스템 붕괴 연쇄가 모두 dB/dt에서 출발. Ap/Kp/Dst는 planetary 평균이라 지역적 위협을 숨긴다. / All downstream effects — GIC magnitude, E-field, saturation severity, system collapse — trace to local dB/dt. Planetary indices (Ap/Kp/Dst) average out the very localization that matters for grid impact.

2. **"March 1989은 mild한 사건"이라는 reframing / March 1989 was the mild case**
   - Québec 붕괴는 ~480 nT/min에서 발생. 같은 폭풍 유럽 측 2,000 nT/min, 1982년 2,700 nT/min, **1921년 ~5,000 nT/min, 20 V/km**. 본 보고서는 1921-level recurrence(~100년 재현)를 진지한 공학 가정으로 삼는다. / Québec collapsed at ~480 nT/min. The same storm hit ~2,000 nT/min in Europe; 1982 reached 2,700; **1921 reached ~5,000 nT/min and 20 V/km**. The report treats a 1921-level recurrence (~100-year return) as an engineering baseline, not science fiction.

3. **전력망의 취약성은 우주기상이 아니라 인프라가 만든다 / Vulnerability is driven by infrastructure, not by the Sun**
   - 1950s → 2000s 사이 미국 HV 송전망 10배 성장. 평균 선로 길이와 kV가 함께 증가했고 저항은 감소. 동일 폭풍이 1950년과 2010년에 가한 임팩트는 거의 완전히 다른 사건이다. / The U.S. HV network grew ~10× between the 1950s and 2000s. Line lengths and operating voltages went up while resistances went down. The same storm produces fundamentally different impacts on the 1950 grid and the 2010 grid — vulnerability is an infrastructure story.

4. **변압기 반-주기 포화는 이중 위협이다 / Half-cycle saturation poses a dual threat**
   - (a) MVAR 수요 폭증 → 시스템 전압 붕괴 (Québec의 즉각적 mechanism), (b) 누설 자속 → tie-plate 국부 과열 → 영구 손상 (Salem, Zion, 원자력 GSU 연쇄). 두 경로는 독립적이고 상보적이다. / Saturation causes both (a) massive MVAR demand that crashes grid voltages (the Québec mechanism) and (b) stray-flux heating that destroys transformers permanently (the Salem mechanism). They are independent, complementary failure paths.

5. **단상·고전압 설계의 희귀한 취약성 / Single-phase high-kV is the uniquely vulnerable combination**
   - 765 kV 변압기의 97%가 단상, 500 kV는 66% (BPA는 97%). 단상은 3상 3-leg 대비 동일 GIC에서 4배 MVAR 흡수, 765 kV는 345 kV 대비 2배. 미국 grid에서 "가장 중요한" 장비가 "가장 취약"하다. / 97% of 765 kV transformers are single-phase (66% at 500 kV, 97% for BPA's 500 kV). Single-phase draws 4× the MVAR of 3-leg 3-phase at equal GIC; 765 kV draws 2× of 345 kV. The most critical hardware is also the most vulnerable.

6. **복구 불가능성이 핵심 / Irreparability is the core risk**
   - 300 MVA+ 변압기 리드타임 ≥12개월, 물리적 운송 조차 수주. **세계 연간 EHV 변압기 생산량이 본 보고서의 예측 손실(300+기)보다 적다.** 손실 후 "단지 새 것을 사면 된다"는 해결책이 존재하지 않는다. / EHV transformers have 12–24 month lead times; even transport takes weeks. **World annual production is less than the report's projected loss of 300+ units.** Post-failure, there is no "just buy new ones" option.

7. **N-1 설계 원칙의 근본적 파탄 / N-1 design philosophy fundamentally fails**
   - 전통적 N-1은 "단일 contingency를 어디서든 견딘다"를 의미. GMD는 **대륙 전체에 동시-correlated multipoint failure**를 부과 → N-1은 수학적으로 불가. Québec의 92초 N-7 붕괴가 증명. 새로운 "GMD-ready" 설계 기준이 필요. / N-1 assumes localized, time-separated contingencies. GMD produces **continental-scale, simultaneous, correlated multipoint failures**, which N-1 cannot accommodate by construction. Québec's N-7 in 92 seconds is the proof. A new "GMD-ready" criterion is required — and the report is the foundational document arguing for it.

8. **Salem은 dry-run이었다 / Salem was a warning shot**
   - 1989년 Salem GSU는 파괴되었으나 22 kV 저전압 권선에만 국한되어 "운 좋게" 사용 중단이 다음날까지 지연. 4,800 nT/min 시나리오에서는 **216~1,011기의 Salem-급 노출이 동시 발생**. 1989년의 "lucky outcome"이 재현된다는 가정은 이제 성립하지 않는다. / The Salem GSU failed in 1989 but damage luckily confined to 22 kV windings. Under a 4,800 nT/min scenario, **216–1,011 transformers receive Salem-class exposure simultaneously**. Assuming 1989's lucky outcome replicates across that population is statistically indefensible.

---

## 4. Mathematical Summary / 수학적 요약

### A. From dB/dt to Geoelectric Field / dB/dt에서 지전장까지

**Faraday 유도 / Faraday induction**:
$$
\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}
$$

**1-D 층상 지층의 표면 임피던스 / Plane-wave surface impedance**:
$$
E_x(\omega) = Z(\omega)\,H_y(\omega), \qquad Z(\omega) = \sqrt{\frac{i\omega\mu_0}{\sigma_{\text{eff}}(\omega)}}
$$

- $Z(\omega)$: 표면 임피던스(Ω). Section 1.2의 18 profiles는 각각 다른 $\sigma(z)$를 갖는다.
- $\mu_0$: 진공 투자율 ($4\pi \times 10^{-7}$ H/m)
- $\sigma_{\text{eff}}(\omega)$: 주파수별 유효 전도도 (층상 구조의 가중치)

**Figure 1-5의 핵심 수치**: 2400 nT/min electrojet에서 SOS2 profile → ~16 V/km, BCA2 → ~2 V/km. **7배 차이**는 지층 불확실성이 전체 위협 평가의 주요 오차 원천임을 의미.

### B. Line-Level GIC / 선로 수준 GIC

$$
V_{\text{line}} = \int_L \mathbf{E}\cdot d\boldsymbol{\ell} \approx E\,L\cos\theta
$$

$$
I_{\text{GIC}} = \frac{V_{\text{line}}}{R_{\text{line}} + R_{\text{wind,1}} + R_{\text{wind,2}} + R_{\text{gnd,1}} + R_{\text{gnd,2}}}
$$

- $L$: 선로 길이 (miles/km). 765 kV 평균 64.7 miles → 1 V/km 지전장에서 ~100 V 이상.
- $\theta$: E field 방향과 선로 방향의 각도.
- $R$: DC 저항 (Ω). 345 kV 이상은 < 0.1 Ω/mile → 수백 암페어 GIC 가능.

### C. Transformer Half-Cycle Saturation / 변압기 반-주기 포화

**DC 바이어스 하의 자속 / Flux under DC bias**:
$$
B(t) = B_{\text{AC}}\sin(\omega t) + B_{\text{DC}}, \qquad B_{\text{DC}} \propto \frac{N \cdot I_{\text{GIC}}}{\mathcal{R}_{\text{core}}}
$$

- $B_{\text{DC}}$가 포화 자속 $B_{\text{sat}}$을 초과하는 반주기에서 자화전류가 기하급수적으로 증가.

**여자전류 관찰치 / Measured excitation current peaks** (500 kV 단상):
- 정상 / Normal: < 1 A peak
- 5 A/phase GIC: ~85 A peak (Fig 1-25)
- 25 A/phase GIC: ~290 A peak (Fig 1-26)
- 100 A/phase GIC: **>800 A peak** (Fig 1-27), THD 93–96%

### D. MVAR Demand / 무효전력 수요

경험적 선형 관계 (Fig 1-17, 500 kV):
$$
\Delta Q_{1\phi}(I_{\text{GIC}}) \approx 1.7\,I_{\text{GIC}}\ \text{MVAR/(A/phase)}
$$
$$
\Delta Q_{3\phi,\text{3-leg}}(I_{\text{GIC}}) \approx 0.4\,I_{\text{GIC}}\ \text{MVAR/(A/phase)}
$$

kV에 따른 scaling (단상, Fig 1-18):
$$
\Delta Q_{765}/\Delta Q_{345} \approx 2
$$

시스템 전체 MVAR:
$$
\Delta Q_{\text{system}} = \sum_{i \in \text{xformers}} \Delta Q_i(I_{\text{GIC},i})
$$

**March 1989**: 피크 8,100 MVAR (21:44 UT, Fig 2-25). **4800 nT/min @ 50° lat 시나리오**: ~110,000 MVAR (**13배**).

### E. Transformer Hot-Spot Heating / 변압기 국부 과열

$$
T_{\text{hot}}(t) = T_0 + \int_0^t \frac{P_{\text{leak}}(\tau) - P_{\text{cool}}(T_{\text{hot}}(\tau))}{C_{\text{th}}}\,d\tau
$$

- $P_{\text{leak}} \propto I_{\text{GIC}}^n$ (n ≈ 1.5–2.0, 설계 의존)
- $P_{\text{cool}}$: 기름 순환 냉각 제거율 (bulk oil temperature의 느린 변화)
- $C_{\text{th}}$: 국부 열 용량

**Meadowbrook 실측 (1992/5/10)**: 60 A neutral GIC → 175°C rise in 10 min (top-oil nearly unchanged). / DC 테스트 (Fig 2-36): 12.5 A → 75 A 스텝 in 370-MVA transformer → top-of-tie-plate 85°C in 3 min.

**Hurlet table / 시간 내구력 / Time withstand (Fig 4-10, un-optimized design)**:
| GIC (amps neutral) | Withstand time |
|---|---|
| 100 | **0 min (instant failure)** |
| 50 | 3 min |
| 25 | 33 min |

최적화 설계 / Optimized design:
| 100 A | 23 min |
| 50 A | > 2 h |
| 25 A | > 2 h |

### F. Damage Threshold Criteria / 피해 임계치

- **Conservative (NAS/FEMA)**: 90 A/phase GIC
- **Salem-class (Meta-R-319)**: 30 A/phase GIC
- **최소 관측 실패 (Eskom 2003)**: 2 A/phase 3-leg 3-phase (Price 2002 analysis)

**4800 nT/min @ 50° lat 시나리오의 at-risk count**:
| Voltage | 90 A threshold | 30 A threshold |
|---|---|---|
| 345 kV | 214 units (20% of MVA) | 677 units (51% of MVA) |
| 500 kV | 137 units (28%) | 285 units (53%) |
| 765 kV | 17 units (32%) | 49 units (76%) |
| **Total** | **368** | **1,011** |

### G. Return Period / 재현주기

- 2,000 nT/min급 이벤트: 과거 30년에 3회 (1972/8, 1982/7, 1989/3) → ~1-in-10 year
- 2,400 nT/min 120° longitudinal footprint가 North America 위에 center될 확률 (~1/3) → **~1-in-30 year**
- **~5,000 nT/min (May 1921 급) @ North America**: **~1-in-100 year**

### H. 예시 계산 / Worked Example — 765 kV GIC under Carrington-class E field

**전제 / Assumptions**:
- $E$ = 10 V/km (4800 nT/min, 대표 지층)
- $L$ = 100 km (765 kV 평균의 약 1.5배 — 미 동부 long-distance interconnection)
- $\theta = 0$ (E와 선로 정렬)
- $R_{\text{total}}$ = 0.3 Ω (선로 0.1 + 변압기 권선 2×0.08 + 접지 2×0.02)

**결과 / Result**:
$$
V = 10 \times 100 \times 1 = 1000\ \text{V}
$$
$$
I_{\text{GIC}} = \frac{1000}{0.3} \approx 3333\ \text{A (total line)}
$$
$$
I_{\text{GIC}}/\text{phase} = \frac{3333}{3} \approx 1111\ \text{A/phase}
$$

Figure 4-12의 top-200 ranking curve와 **일치** (최대 ~1,800 A/phase). 해당 765 kV 변압기의 MVAR: ~2×1,111×1.7 ≈ **~3,800 MVAR** (단상 가정, 765 kV scaling factor 2 반영) — 단일 변압기가 중형 발전소 용량의 무효전력을 흡수.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1847 ──── First reports of telegraph anomalies from auroral activity
            │
1859 Sep ── Carrington Event: telegraph fires; estimated Dst ~-1,760
            │
1921 May ── "New York Railroad Storm" — Stockholm–Toreboda 20 V/km
            │            (largest instrumented geoelectric field)
1940 Mar ── First U.S. power-system impacts (115 kV era, McNish/Davidson)
            │
1972 Aug ── AT&T L-4 cable failure; Stuart, OH transformer off-scale >100 A
            │
1982 Jul ── 2,700 nT/min in mid-Sweden; 9.1 V/km measured
            │
1989 Mar ── Hydro-Québec blackout (9h, 6M customers); Salem GSU damaged
            │            ★ Benchmark forensic event for this paper
1990s ──── First coupled Metatech modeling (Kappenman et al. EOS 1997,
            │   CIGRE 2002, NATO-ASI 2001) validated against Nov 1993,
            │   May 1998, Oct 1991 events
2003 Oct ── Malmö blackout (Sweden); Eskom loses 14× 400 kV transformers
            │
2008 ──── NAS "Severe Space Weather Events" report — $1–2T estimate
            │
2010 Jan ── ★ Kappenman Meta-R-319 (THIS REPORT)
            │
2011–12 ── Kappenman JASTP companion papers; JASON report
            │
2013 May ── FERC Order 779: directs NERC to develop GMD reliability standards
            │
2014 Jun ── NERC TPL-007-1 (Stage 1 GMD vulnerability assessment) approved
            │
2016 ──── NERC TPL-007-2 (Stage 2); benchmark set at 8 V/km (1-in-100 yr)
            │   — substantially lower than Kappenman's ~20 V/km scenarios
            │   → ongoing "over-hype vs under-hype" controversy
2017 ──── Pulkkinen et al. ExPRE statistical model; Oughton et al.
            │   economic re-estimates; Love et al. extreme-value analysis
2019+ ──── TPL-007-4; SWFO-L1 planning; G5 Gannon Storm (May 2024)
            │   re-validates coupled-modeling approach
```

**이 보고서의 위치**: forensic science(1989 Québec 사건 분석) → engineering risk assessment(2010 Meta-R-319) → regulatory action(FERC 779, NERC TPL-007)의 **중간 다리**. Kappenman 이전에는 "GMD는 fleeting operational disturbance"로 치부되었고, Kappenman 이후에는 "GMD는 permanent infrastructure damage risk"로 재프레이밍되었다.

**This report's position**: the pivot point between forensic science (1989 Québec analysis) and regulatory action (FERC 779 / NERC TPL-007). Before Kappenman, the prevailing frame was "GMD = transient operational nuisance." After Kappenman, "GMD = permanent infrastructure damage risk" — a reframe that directly enabled federal mandate.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Paper #17 (GIC foundational, e.g., Boteler/Pirjola)** | GIC 유도 물리의 기초 이론. 본 보고서는 이를 대륙 규모로 **engineering scale-up**. / Provides the underlying induction physics that Kappenman scales up to continental engineering analysis. | 선행 prerequisite — Kappenman의 출발점 |
| **Paper #24 (Bolduc 2002 / Québec 1989 forensic)** | 1989 Québec 붕괴의 상세 forensic. Section 2.1이 이 논문의 결과를 **Metatech 모델로 재현**하며 모델 타당성을 검증. / The Québec blackout forensic paper; Section 2.1 is the validation test that Kappenman's model must pass. | 본 보고서가 replicate하는 benchmark event |
| **Tsurutani et al. 2003 (Carrington Event analysis)** | 1859 사건의 Dst ~-1,760 추정. Kappenman의 "1-in-100 year ~5,000 nT/min" 시나리오가 이 작업과 연결됨. / Provides the Carrington-class Dst estimate that underpins Kappenman's extreme scenario. | Extreme scenario calibration |
| **NAS 2008 "Severe Space Weather Events" report** | $1–2T 경제 피해 추정. Kappenman이 engineering backbone을 제공; NAS가 economic framing. 짝 문서. / NAS provides economic framing; Kappenman provides the engineering substantiation. They are paired documents driving U.S. policy. | 정책 창(policy window)의 쌍둥이 문서 |
| **Pulkkinen et al. 2017 (ExPRE)** | Kappenman 프레임워크를 **통계적으로 재계산**. 8 V/km (100년) 벤치마크로 수렴 — Kappenman의 20 V/km보다 낮음. "how extreme is extreme?" 논쟁의 한 축. / Reassesses Kappenman's extremes statistically, converging on a lower 8 V/km benchmark (now NERC's TPL-007 standard). | 비판적 계승자 |
| **Love et al. 2016 (Extreme-value statistics of B̂)** | 지자기 관측기록으로 return-period 통계. Kappenman의 ~1-in-100 year 추정을 empirical하게 검증/수정. / Empirical return-period statistics on geomagnetic observatories; tests Kappenman's return-period claims. | 재현주기 검증 |
| **Oughton et al. 2017 (Economic impact)** | Kappenman의 engineering 추정을 macroeconomic model과 결합. 미국 GDP 대비 daily loss $6B–42B. / Wraps Kappenman's transformer-loss estimates in a macroeconomic model, producing daily GDP loss figures of $6–42B. | 경제적 연장 |
| **NERC TPL-007 standard (2016+)** | Kappenman 보고서의 **직접적 규제 출력**. 모든 미국 BPS 사업자는 8 V/km 벤치마크에 대한 취약성 평가 의무. / The direct regulatory output of Kappenman (via FERC Order 779). Mandates 8 V/km benchmark vulnerability assessments. | 정책 산출물 |
| **EMP Commission reports (2004/2008)** | "natural GMD ≈ E3 HEMP" 프레임. Kappenman Section A4.4는 이 등가를 정량 비교. / Establishes the GMD ≈ E3 equivalence; Kappenman's Appendix A4.4 quantifies the comparison. | Dual-threat 관점 |

---

## 7. References / 참고문헌

### 본 보고서의 주요 인용 / Primary references cited in the report

**Section 2 (March 1989 analysis)**:
- NERC Disturbance Analysis Working Group, "The 1989 System Disturbances: March 13, 1989 Geomagnetic Disturbance," 1990 (Ref 2-1).
- Larose, D. "The Hydro-Québec System Blackout of March 13, 1989," IEEE Special Publication 90TH0291-5 PWR, 1989 (Ref 2-2).
- Kappenman, J.G. "An Introduction to Power Grid Impacts and Vulnerabilities from Space Weather," NATO-ASI, Kluwer, 2001 (Ref 2-4).
- Gattens, P., "Application of a Transformer Performance Analysis System," SE Electric Exchange, 1992 (Ref 2-5).
- Kappenman, J.G. "Geomagnetic Storms and Their Impact on Power Systems: Lessons Learned from Solar Cycle 22...," IEEE Power Eng. Review, May 1996 (Ref 2-7).
- Walling, R.A. & Kahn, A.H. "Solar-Magnetic Disturbance Impact on Power System Performance and Security," EPRI TR-100450, 1992 (Ref 2-8).

**Section 3 (Extreme scenarios)**:
- Carovillano, R.L. & Siscoe, G.L. "Energy and Momentum Theorems in Magnetospheric Processes," Rev. Geophys. Space Phys., 11, 289, 1973 (Ref 3-1).
- Tsurutani, B.T., Gonzalez, W.D., Lakhina, G.S., Alex, S. "The Extreme Magnetic Storm of September 1–2, 1859," JGR, 2002 (Ref 3-3).
- Anderson, C.W., Lanzerotti, L.J., Maclennan, C.G. "Outage of the L-4 System and the Geomagnetic Disturbances of August 4, 1972," Bell System Tech J, 53, 1817, 1974 (Ref 3-5).
- Boteler, D.H. & Van Beek, J.G. "Mapping the March 13, 1989 Magnetic Disturbance...," Solar-Terrestrial Predictions IV, 1992 (Ref 3-6).
- Davidson, W.F. "The Magnetic Storm of March 24, 1940 – Effects in Power Systems," EEI Bulletin, 1940 (Ref 3-8).
- Kappenman, J.G. "Great Geomagnetic Storms and Extreme Impulsive Geomagnetic Field Disturbance Events — An Analysis of Observational Evidence including the Great Storm of May 1921," Adv. Space Res., 2005 (Ref 3-4b), doi:10.1016/j.asr.2005.08.055.

**Section 4 (At-risk transformers)**:
- Gaunt, C.T. & Coetzee, G. "Transformer failures in regions incorrectly considered to have low GIC-risk," IEEE Power Tech, 2007 (Ref 4-1).
- Price, P.R. "Geomagnetically Induced Current Effects on Transformers," IEEE Trans. Power Delivery, 17(4), 1002–1008, October 2002 (Ref 4-2).
- Girgis, R.S. & Ko, C.D. "Calculation Techniques and Results of Effects of GIC Currents as Applied to Two Large Power Transformers," IEEE Trans. Power Delivery, 7(2), April 1992 (Ref 4-5).
- Hurlet, P. & Berthereau, F. "Impact of geomagnetic induced currents on power transformer design," MATPOST'07, Lyon, 2007 (Ref 4-6).
- Stiegemeier, C.L. & Girgis, R. "Rapidly Deployable Recovery Transformers," IEEE Power and Energy, 4(2), March/April 2006 (Ref 4-7).

### Canonical citation / 정식 인용

Kappenman, J. G. (2010). *Geomagnetic Storms and Their Impacts on the U.S. Power Grid* (Meta-R-319). Metatech Corporation, prepared for Oak Ridge National Laboratory under subcontract 6400009137, January 2010.

### Related follow-on work / 후속 관련 연구

- Kappenman, J.G. "Low-frequency protection concepts for the electric power grid: GIC & E3 HEMP mitigation," Metatech Meta-R-322, 2010.
- Pulkkinen, A., Bernabeu, E., Thomson, A., et al. "Geomagnetically induced currents: Science, engineering, and applications readiness," Space Weather, 15, 828–856, 2017.
- Love, J.J., Rigler, E.J., Pulkkinen, A., Riley, P. "On the lognormality of historical magnetic-storm intensity statistics," Geophys. Res. Lett., 42, 6544–6553, 2015.
- Oughton, E.J., Skelton, A., Horne, R.B., Thomson, A.W.P., Gaunt, C.T. "Quantifying the daily economic impact of extreme space weather due to failure in electricity transmission infrastructure," Space Weather, 15(1), 65–83, 2017.
- NERC, Reliability Standard TPL-007-4 (Transmission System Planned Performance for Geomagnetic Disturbance Events), 2019.
