---
title: "Modeling 3-D Solar Wind Structure"
authors: Dušan Odstrčil
year: 2003
journal: "Advances in Space Research, Vol. 32, No. 4, pp. 497–506"
doi: "10.1016/S0273-1177(03)00332-6"
topic: Space_Weather
tags: [ENLIL, MHD, solar_wind, CME, WSA, heliosphere, TVDLF, AMR, NOAA_SWPC, space_weather_forecasting, streamer_belt, CIR]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 21. Modeling 3-D Solar Wind Structure / 3차원 태양풍 구조 모델링

---

## 1. Core Contribution / 핵심 기여

**한국어**
이 논문은 NOAA Space Environment Center에서 개발된 3차원 자기유체역학(MHD) 코드 **ENLIL**의 우주기상 응용 능력을 4개 주제로 종합 시연한 정본 리뷰이다. 저자(Odstrčil)는 (1) **수치 기법** — Tóth & Odstrčil(1996)의 modified high-resolution **TVDLF** scheme + Tóth(2000)의 **$\nabla\cdot\mathbf{B}=0$ 보존** + MPI 병렬화 + Paramesh AMR — 을 정리한 뒤; (2) **CME 동역학**으로 over-pressured plasmoid가 streamer belt와 만나 forward+reverse shock과 CIR을 형성하며 변형되는 3D 시뮬레이션(Fig. 1, 2)을 제시; (3) **Ambient solar wind 검증**으로 1995 declining-phase에 수정된 Wang–Sheeley–Arge 경험식 ($V_R = V_0 + V_1\sin\theta/F^Z$, $V_0=150, V_1=1500, Z=0.6$)을 inner boundary 조건으로 사용해 L1 in-situ 관측을 정량 재현(Fig. 3, 4); (4) **Coupled coronal-heliospheric** 모델링으로 SAIC + CU/CIRES-NOAA/SEC 합작 시스템에서 magnetic flux rope CME가 코로나에서 행성간으로 전이하는 모습(Fig. 5, 6)과 향후 **STEREO Heliospheric Imager**의 white-light 영상을 LOS 적분으로 합성(Fig. 7-9); (5) **Near-Earth simulation**으로 1996-05-14~18 이벤트에서 WIND 위성을 driver로 IMP-8/Interball L1 위치의 $B_z$를 예측 비교(Fig. 10). 이 모든 작업이 5년 후(2008) NOAA SWPC가 정식 운영 모델로 채택한 **WSA-ENLIL+Cone 파이프라인**의 직접적 청사진이 되었으며, 현재 KASI/KSWC, ESA SSA, JAXA 등 모든 주요 우주기상 운영기관의 baseline forecasting 시스템이다.

**English**
This paper is the canonical 2003 demonstration of NOAA SEC's 3-D MHD code **ENLIL** across four space-weather applications. Odstrčil (1) lays out the numerics — modified high-resolution **TVDLF** (Tóth & Odstrčil 1996), **$\nabla\cdot\mathbf{B}=0$ field-interpolated central differencing** (Tóth 2000), MPI domain decomposition, and **Paramesh AMR** (MacNeice et al. 2000); (2) shows how an **over-pressured plasmoid CME** interacts with the streamer belt to produce forward+reverse shock pairs that interlock with CIRs (Fig. 1–2); (3) validates the **ambient solar wind** in 1995 with a tuned **Wang–Sheeley–Arge** driver $V_R = V_0 + V_1\sin\theta/F^Z$ ($V_0{=}150$, $V_1{=}1500$, $Z{=}0.6$) reproducing both Ulysses high-latitude fast wind and L1 in-situ observations (Fig. 3–4); (4) demonstrates **coupled coronal-heliospheric MHD** with a magnetic flux rope tracked from the Sun into interplanetary space (Fig. 5–6), and previews **STEREO Heliospheric Imager** observations via line-of-sight Thomson-scattering synthesis (Fig. 7–9); (5) performs **near-Earth Cartesian-grid simulation** of the 14–18 May 1996 event, using WIND as upstream driver to predict $B_z$ at IMP-8 and Interball (Fig. 10). The blueprint laid out here became, almost unchanged five years later, NOAA SWPC's **operational WSA-ENLIL+Cone pipeline** (2008–present), which today anchors space-weather forecasting at NOAA, KASI/KSWC, ESA SSA, and JAXA.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (p.497) / 도입

**한국어** 우주기상 연구·예보는 다양한 시간·공간 척도에서 동시 발생하는 복잡한 현상의 사슬. CME와 행성간 충격파가 비정상적 지자기 폭풍의 주된 원인 (Gosling et al. 1991). 지구로 오는 길에 구조화된 태양풍과 다른 transient들과 상호작용하여 geo-effectiveness가 크게 변화 (Burlaga 1987; Tsurutani 1999). 이를 정확히 모사하려면 (a) **사전 존재하는 구조화된 ambient**, (b) **현실적 transient launch**, (c) **거대 공간 도메인에서 다중 구조 추적**의 세 가지가 동시에 필요하다.

**English** Space weather involves a complex multi-scale phenomenon chain. CMEs and IP shocks cause non-recurrent storms; their geo-effectiveness is modified by interactions with structured ambient wind. Realistic simulation requires (a) pre-existing structured ambient, (b) realistic transient launch, (c) tracking of multiple structures over huge spatial domains.

### Part II: Numerical Model (p.497–498) / 수치 모델

**한국어**
ENLIL이 푸는 ideal MHD 시스템 (식 1–4 + 두 추적자 식):
$$\partial_t \rho + \nabla\cdot(\rho\mathbf{V}) = 0$$
$$\partial_t(\rho\mathbf{V}) + \nabla\cdot(\rho\mathbf{V}\mathbf{V}) = -\nabla P + \nabla\cdot(\mathbf{B}\mathbf{B}/\mu) + \rho GM_S/r^2$$
$$\partial_t E + \nabla\cdot(E\mathbf{V}) = -p\nabla\cdot\mathbf{V}$$
$$\partial_t\mathbf{B} = \nabla\times(\mathbf{V}\times\mathbf{B})$$
$$\partial_t\rho_c + \nabla\cdot(\rho_c\mathbf{V}) = 0,\quad \partial_t\rho_p + \nabla\cdot(\rho_p\mathbf{V}) = 0$$

여기서 $P$ = thermal $p$ + magnetic $B^2/2\mu$, $E = p/(\gamma-1)$ (thermal energy), state $p = 2nkT$, $\rho = mn$. **중요한 설계 선택**: total energy 대신 **thermal energy 방정식**을 사용 — smooth profile 보장하나 강한 shock의 속도/진폭에 inaccuracy 가능. $\rho_c, \rho_p$는 각각 CME 물질과 자기 polarity 추적자.

**수치 기법**:
- **Modified high-resolution TVDLF** (Tóth & Odstrčil 1996), 명시적 인공 점성 없음, shock·discontinuity 밖에서 2차 정확도, oscillation 없는 단조 보존.
- **$\nabla\cdot\mathbf{B}=0$**: field-interpolated central-difference (Tóth 2000), round-off 수준 유지.
- **Dimensional splitting** for 효율, multi-dimensional도 가능.
- **MPI domain decomposition** — 3-D 도메인을 작은 slab으로 나눠 병렬 처리, boundary data MPI 교환.
- **Paramesh AMR** (MacNeice et al. 2000) — fine structure 고해상도화 (Odstrčil et al. 2002c).

**Geometry**: Cartesian or spherical, 1/2/3-D 모두 가능.

**English** Standard ideal MHD with gravity, plus two passive tracers. Key design choice: thermal-energy equation (smoother profiles, but loses some shock-strength accuracy). Numerics combine TVDLF + ∇·B cleaning + MPI + Paramesh AMR.

### Part III: Global 3-D Interactions (p.498–500) / 전역 3-D 상호작용

**한국어** Hundhausen(1993): 대부분 CME가 coronal streamer belt 근처에서 발사 → 자전하는 구조화된 wind에서 진화. Odstrčil & Pizzo(1999a,b)의 **Case 1** 시나리오 — CME(over-pressured plasmoid)를 tilted, 밀집한 streamer belt 안으로 발사. 본 논문 Fig. 1–2가 Case 1의 두 azimuthal slice (적도 위 7.5°, 적도 아래 9.75°).

**Fig. 1 (적도 위 7.5°)**: 12일 후 CME. 초기 속도가 slow stream의 2배 = fast stream과 동일. CME가 streamer belt를 통과하며 momentum 산실 → 감속. 한편 northern fast stream의 선두가 slow streamer belt를 만나 **shock-pair structure (forward + reverse)** 형성하며 **CIR(Co-rotating Interaction Region)** 발달. 감속한 CME가 CIR의 leading edge에 점차 따라잡힘 → CME가 **slow streamer belt와 fast coronal hole flow 사이에 "샌드위치"**, CIR 구조 안에 trap. 결과: CME와 CIR shock 간 **stand-off distance ≈ 0**, CME-driven shock의 stand-off가 CME 앞에서 변하다가 점진적으로 forward CIR shock과 합병.

**Fig. 2 (적도 아래 9.75°)**: 같은 시각, 다른 slice. CME가 **선행 southern fast stream의 trailing rarefaction region**에 침투. CME의 forward shock이 약화되고 pressure wave로 전이; **reverse shock**이 CME leading edge 근처에 생겨 CME 물질을 거슬러 전파, forward shock보다 좁고 강함. **3 AU에서의 시간 프로파일**: $V_R$ ~600 km/s 점프 후 감속, $N$ 짧은 spike, $P$ enhancement.

**핵심 결론**: 단순 over-pressured plasmoid 발사로도 **위치마다 매우 다른 transient 특성**을 만들 수 있다. CME의 초기 모양·밀도 분포가 모든 차원에서 왜곡되고, CME-driven shock의 inclination/strength/stand-off가 변한다. **Geo-effectiveness는 CME의 streamer belt 대비 발사 위치 + 지구의 pulse 중심선 대비 위치에 의해 결정**.

**English** Following Odstrčil & Pizzo (1999a,b) Case 1: a plasmoid launched into a tilted, dense streamer belt produces dramatically different signatures at different azimuthal slices because of CME-CIR interaction. The CME gets trapped between slow streamer-belt flow and fast coronal-hole flow, the CME-driven shock merges with the forward CIR shock, and the trailing rarefaction erodes the following CIR. Conclusion: **even a uniform plasmoid produces complex, position-dependent geo-effectiveness** — the launch location relative to the streamer belt is decisive.

### Part IV: Ambient Solar Wind (p.500–502) / 배경 태양풍

**한국어** Ambient 구조 지식이 (a) CME 전파 맥락 + (b) 재발성(recurrent) 지자기 폭풍 예보에 필수. Global 3-D MHD가 적절한 시간 의존 inner boundary를 받으면 Earth의 태양풍 파라미터 예측 가능. 현재 그런 조건의 직접 측정이 부족 → 다양한 근사 사용. **Potential Field Source-Surface (PFSS)** 모델이 inner boundary 도출에 활용 (Usmanov 1993; Odstrčil et al. 1998; Linker 1999; Riley 2001).

**1995 검증** (declining-phase, 태양 극소기 근처):
- Wilcox Solar Observatory 광구 자기장 (http://wso.stanford.edu) → PFSS → source surface ($R = 21.5\,R_S = 0.1$ AU)에서 radial 자기장 추출.
- **수정된 Arge & Pizzo (2000)** 속도식: $V_R = V_0 + V_1\sin\theta/F^Z$, 원본 ($V_0=285, V_1=575, Z=1/1.7$)이 Wang & Sheeley(1990) 5단계 이산 속도값보다 개선. $F$ = 광구↔source surface(2.5 $R_S$) 자기력선 expansion factor.
- **Azimuthal magnetic field**: $B_\phi = -B_r\sin\theta\,V_\text{rot}/V_r$ (Parker spiral), $V_\text{rot}$ = 27.2753-day 자전 속도. $B_\theta$, $V_\theta$, $V_\phi$는 0 가정.
- **Density**: 일정 momentum flux로 $\rho$ 도출. **Temperature**: 일정 total pressure로 $T$ 도출.

**Fig. 3 (원본 daily-updated WSA)**: Top panel — source surface에서 derived velocity (CR 1891-1894). Bottom — 지구에서 시뮬레이션(실선) vs 관측(굵은 점). 일치 양호하나 **고위도에서 속도 너무 느림** (Ulysses 관측 대비), **저위도 coronal hole이 가장 빠른 wind** 제공 (불일치).

**Fig. 4 (수정 WSA: $V_0=150, V_1=1500, Z=0.6$, clip 275–625 km/s)**: slow/fast 전이 더 sharp, 고위도 속도 더 균일하고 큼 → Ulysses 관측과 더 일치. 지구에서의 일치(bottom)도 향상됨 → source-surface 모델 추가 정교화 동기 부여 (Arge et al. 2002).

**English** PFSS-based source surface at 21.5 R_S provides ENLIL's inner boundary. 1995 validation tunes the Arge–Pizzo velocity law to ($V_0{=}150$, $V_1{=}1500$, $Z{=}0.6$) to match Ulysses high-latitude fast wind. The tuned model improves agreement at Earth (Fig. 4 vs Fig. 3) and motivates further source-surface model refinement.

### Part V: Coupled Coronal and Heliospheric Simulations (p.502–503) / 코로나-행성간 결합

**한국어** 우주기상은 다중 시·공간 척도 → 특화된 모델들을 결합. **SAIC San Diego의 코로나 코드** + **CU/CIRES-NOAA/SEC의 ENLIL**을 합쳐 **2-D MHD merging** 시연 (Odstrčil et al. 2002b). 본 논문은 **3-D 결합**의 첫 결과 미리보기 (Odstrčil et al. 2002d).

**Fig. 5 (적도 평면 위에서 본 magnetic flux rope, $t=210$h)**: fast-moving flux rope이 태양에서 model interface boundary를 거쳐 heliosphere로 확장. 동반된 IP shock이 flux rope 앞 plasma density를 압축. 좌: $r^2$로 normalized number density (cm⁻³); 우: equatorial 평면의 radial flow velocity. 격자 간격 4 $R_S$ × 8°.

**Fig. 6 (옆에서 본 magnetic flux rope, $t=220$h)**: flux rope이 radial 압축, plasma density 축적. 왜곡된 shock front은 slower solar wind와의 상호작용 + heliospheric current sheet 주변 구조 때문. **South-north IMF 성분**이 (a) shock compression, (b) field line draping, (c) ejected flux rope topology에서 발생 (Detman et al. 1991; Odstrčil & Pizzo 1999c). Flux rope 후행 (heliosphere) 및 그 아래(코로나) post-eruptive reconnection signature (Riley et al. 2002).

**STEREO Heliospheric Imager 합성 (Fig. 7–9)**: 3-D 시뮬레이션 결과를 LOS 적분 (Hundhausen 1993의 white-light scattering)으로 합성. 3개 viewpoint(STEREO-B, Earth, STEREO-A) × 2개 시각.
- **Fig. 7 (Total brightness)**: 매우 낮은 contrast, heliospheric 구조 식별 어려움.
- **Fig. 8 (Normalized gradient)**: 균질 large-scale 구조에 적합 (예: streamer belt 투영).
- **Fig. 9 (Running difference, 10시간 간격)**: 강한 배경 잡음과 비균질 구조에서도 transient 식별 가능. 코로나 영상의 분 단위 cadence보다 훨씬 느린 cadence — heliospheric structure가 large spatial scale에서 천천히 확장하기 때문.

**3가지 결론**:
1. **Total brightness**가 streamer belt flow 투영에 따라 north/south 반구 전체에서 강화될 수 있다.
2. **고위도 fast stream 안에서 자유 팽창하는 over-pressured CME**는 slow stream belt flow로 전파하는 compressive CME front보다 식별 어렵다.
3. **Velocity 구조가 큰 ambient에서 CME shape distortion**이 일부 경우 식별 가능.

**English** SAIC coronal model + ENLIL coupled (Odstrčil et al. 2002b,d). Fig. 5–6 show flux-rope CME extending from Sun into heliosphere with shock compression, draping, and post-reconnection signatures. Fig. 7–9 synthesize STEREO Heliospheric Imager white-light brightness from three viewpoints — running-difference imaging works best for transient detection in the noisy heliospheric environment.

### Part VI: Near-Earth Solar Wind Simulation (p.503–505) / 근지구 시뮬레이션

**한국어** 거대 IP 거리에서는 ambient의 비균질성이 커 standard AMR refine/de-refine criteria가 사실상 작동 불가 → **Cartesian geometry**로 high-resolution 시뮬레이션.

**1996-05-14~18 이벤트 사례** (3개 위성 동시 관측): WIND가 지구로부터 가장 멀리(60 $R_E$) — driver로 사용. IMP-8 + Interball이 자기권 upstream → 비교 대상. CDAWeb (http://cdaweb.gsfc.nasa.gov)이 데이터 출처.

**4개 모델 비교**:
- **Model 0**: WIND 관측을 평균 태양풍 속도로 다른 위성 위치로 단순 shift.
- **Model 1**: 구조가 Sun-Earth line에 수직 평면이라고 가정.
- **Model 2a**: ambient solar wind의 보조 heliospheric 시뮬레이션에서 inclination 도출.
- **Model 2b**: IMP-8/Interball 관측에서 temporal shift 도출 (실시간 응용 불가).

**Fig. 10 (IMP-8 & Interball $B_z$ 비교)**: 1996-05-15 13–19 UT. 여러 모델 결과와 관측. 흥미로운 결과: **Model 2a가 IMP-8에 대해 약 15시간에 더 좋은 예측 / 16시간에 더 나쁜 예측**, Interball에는 그 반대. Model 2b 결과는 2a와 비슷하나 shift 적음. Model 0과 1은 태양풍 속도 변동성이 작고 위성 간 radial separation이 작아 비슷.

**핵심 통찰**: WIND가 이번 이벤트에서 지구에 더 가까웠고 큰 변동성도 측정 안 됨. 그럼에도 시뮬레이션은 **단일 위성 관측 사용의 어려움**을 보여준다 — 예측 정확도가 시간·위치에 따라 변하므로 태양풍 구조는 inclination 뿐 아니라 **substantial spatial inhomogeneities**를 가진다.

**English** Cartesian-geometry near-Earth ENLIL simulation of 14–18 May 1996 event: WIND drives the inner boundary at 60 R_E upstream; IMP-8 and Interball serve as predicted points. Four models tested. Even with relatively quiet conditions, model performance varies by time and location — single-spacecraft drivers are intrinsically limited.

### Part VII: Conclusions (p.505) / 결론

**한국어** 본 리뷰는 (a) 태양풍 구조 조사를 위한 수치 모델링 응용, (b) 향후 원격 + in-situ 관측 모델링 지원에 적합한 도구 개발의 최근 결과를 정리. **3-D 태양풍 구조의 수치 모델링은 흐름의 강한 비균질성과 거대 공간 도메인 때문에 도전적인 작업**. **Ambient에서 transient 전파의 현실적 시뮬레이션을 위한 개선된 수치 모델은 아직 개발 필요** — 더 정교한 관측 및 이론 모델 통합 요구. (겸손하지만 정확한 마무리.)

**English** Numerical modeling of 3-D solar-wind structure is challenging due to inhomogeneity and huge spatial domains. Improved models for realistic transient propagation in ambient still need development — requiring more sophisticated observations and theoretical models.

---

## 3. Key Takeaways / 핵심 시사점

1. **CME geo-effectiveness는 발사 위치에 결정적으로 의존한다 / CME geo-effectiveness depends decisively on launch position.**
   동일한 over-pressured plasmoid라도 streamer belt 안/밖, 인접 fast stream 위/아래에 따라 forward shock 강도, reverse shock 형성, CME-CIR merging 양상이 완전히 달라진다 (Fig. 1 vs Fig. 2). **현 운영 forecasting의 첫 번째 자유도가 cone model의 위치 추정 정확도**인 이유. / Even a uniform plasmoid produces dramatically different transients depending on its position relative to the streamer belt — the *first* free parameter in operational cone-model forecasts is launch location.

2. **WSA 경험식의 파라미터 튜닝이 운영 정확도의 큰 부분을 결정 / WSA parameter tuning drives much of operational accuracy.**
   원본 ($V_0=285, V_1=575, Z=1/1.7$) → 수정 ($V_0=150, V_1=1500, Z=0.6$, clip 275-625) 단순 변경이 Ulysses 일치도와 L1 일치도 모두 개선 (Fig. 3 vs Fig. 4). **물리 기반 모델조차 inner boundary tuning이 결정적**. / The simple Arge–Pizzo parameter retune dramatically improves both Ulysses and L1 agreement; even a physics-based model is dominated by inner-boundary tuning.

3. **Inner boundary 21.5 $R_S$ (0.1 AU)가 표준이 된 데에는 실용적 이유가 있다 / The 21.5 R_S inner boundary became standard for practical reasons.**
   PFSS 가정이 source surface 위에서만 유효 (radial field) + Alfvén critical surface 너머에서 super-magnetosonic flow 보장 + 코로나 코드의 outer boundary와 자연스러운 매칭. **이 선택이 ENLIL의 영구 architectural 결정**. / Choice of 21.5 R_S inner boundary respects PFSS validity, super-magnetosonic flow regime, and coronal-code coupling — a permanent architectural decision.

4. **Coupled coronal-heliospheric 모델이 운영 forecasting의 다음 fronteir / Coupled coronal-heliospheric is the next frontier.**
   2003년 시점에 SAIC + CU/CIRES-NOAA/SEC가 시연. flux rope 자기 구조가 Sun→1 AU 보존되어야 $B_z$ 예측 가능. **이것이 EUHFORIA, CORHEL의 후속 작업이 추구한 방향**. / Coupled coronal+heliospheric models (the 2003 demo) are required to track flux-rope topology from Sun to 1 AU — the direction taken by EUHFORIA and CORHEL.

5. **단일 위성 driver의 한계는 줄어들지 않는다 / Single-spacecraft drivers remain fundamentally limited.**
   1996-05 사건의 4개 모델 비교 (Fig. 10): WIND을 driver로 사용해도 IMP-8/Interball $B_z$ 예측이 시간·공간마다 다르게 빗나감. **태양풍은 simple inclination이 아니라 substantial spatial inhomogeneity**. **2024 Gannon storm에도 동일 한계**. / Multi-spacecraft 1996 case shows that even with a clean WIND driver, predictions at IMP-8/Interball miss differently at different times — solar wind has substantial spatial inhomogeneity, not a simple tilted plane structure.

6. **Thermal-energy 방정식 선택은 trade-off / Thermal-energy equation choice is a trade-off.**
   Total energy 대신 thermal energy를 쓰면 smooth profile은 보장되나 강한 shock의 속도/진폭 부정확. **현재 EUHFORIA, GAMERA 등은 total energy + entropy fix 채택** — 이 trade-off가 후속 코드의 차별화 지점. / Using thermal energy ensures smooth profiles but loses some shock-strength accuracy; modern successor codes (EUHFORIA, GAMERA) take different choices, distinguishing themselves on this trade-off.

7. **STEREO Heliospheric Imager의 운영 가치는 합성 영상으로 이미 예언됨 / STEREO HI's operational value was foretold by synthetic imagery.**
   Fig. 7-9의 3개 viewpoint × 3개 imaging 기법 비교 (total brightness / normalized gradient / running difference)가 2006년 STEREO 발사 후 J-map 시대의 운영 영상 처리 방법론을 정확히 예측. / The three-imaging-technique demo (total/normalized-gradient/running-difference) accurately previewed the post-2006 J-map era's operational image-processing pipeline.

8. **운영 우주기상의 baseline은 22년간 거의 변하지 않았다 / The operational space-weather baseline has barely changed in 22 years.**
   2003 ENLIL → 2008 NOAA SWPC operational → 2026 현재 KASI/KSWC, ESA SSA, JAXA 모두 WSA-ENLIL+Cone. EUHFORIA·CORHEL이 부분적 개선이지만 **운영 표준은 여전히 ENLIL**. 이는 (a) 안정성, (b) 검증된 트랙 레코드, (c) CCMC 인프라 때문. / 22 years on, WSA-ENLIL+Cone remains the operational baseline at NOAA SWPC, KASI/KSWC, ESA SSA, and JAXA. Newer codes are partial improvements but operational standard is unchanged due to stability, validated track record, and CCMC infrastructure.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Ideal MHD 시스템 (보존형) / Conservation-form ideal MHD

| Eq. | 식 | 의미 |
|---|---|---|
| (1) | $\partial_t \rho + \nabla\cdot(\rho\mathbf{V}) = 0$ | 연속 / continuity |
| (2) | $\partial_t(\rho\mathbf{V}) + \nabla\cdot(\rho\mathbf{V}\mathbf{V}) = -\nabla P + \nabla\cdot(\mathbf{B}\mathbf{B}/\mu) + \rho GM_S/r^2$ | 운동량 + Lorentz + 중력 |
| (3) | $\partial_t E + \nabla\cdot(E\mathbf{V}) = -p\nabla\cdot\mathbf{V}$ | thermal energy ($E = p/(\gamma-1)$) |
| (4) | $\partial_t\mathbf{B} = \nabla\times(\mathbf{V}\times\mathbf{B})$ | Faraday/induction |
| — | $\partial_t\rho_c + \nabla\cdot(\rho_c\mathbf{V}) = 0$ | CME 추적자 |
| — | $\partial_t\rho_p + \nabla\cdot(\rho_p\mathbf{V}) = 0$ | polarity 추적자 |

여기서 $P$ = thermal $p$ + magnetic $B^2/2\mu$, state equation $p = 2nkT$, $\rho = mn$ ($m$ = proton mass).

### 4.2 Inner boundary at 21.5 $R_S$ (PFSS + WSA)

**Wang–Sheeley–Arge tuned wind speed**:
$$\boxed{\;V_R(\theta) = V_0 + \frac{V_1\sin\theta}{F^Z}\;}$$
- 원본 (Arge & Pizzo 2000): $V_0=285$, $V_1=575$, $Z=1/1.7\approx 0.588$
- 본 논문 수정: $V_0=150$, $V_1=1500$, $Z=0.6$, clip $\in[275, 625]$ km/s
- $F$ = 광구↔source surface(2.5 $R_S$) magnetic flux tube의 areal expansion factor

**Parker-spiral azimuthal field at source surface**:
$$B_\phi = -B_r\sin\theta\,\frac{V_\text{rot}}{V_r}$$
$V_\text{rot}$ = 27.2753-day 자전. $B_\theta = 0$, $V_\theta = V_\phi = 0$ 가정.

**밀도/온도**: 일정 momentum flux로 $\rho$, 일정 total pressure로 $T$.

### 4.3 수치 기법 / Numerical methods

| Component | Choice | Reference |
|---|---|---|
| Shock-capturing | Modified high-resolution **TVDLF** | Tóth & Odstrčil (1996) |
| ∇·B = 0 | Field-interpolated central-difference | Tóth (2000) |
| Mesh | Fixed or adaptive (Paramesh AMR) | MacNeice et al. (2000) |
| Parallelization | MPI domain decomposition (3-D slabs) | — |
| Time-stepping | Explicit, dimensional splitting | — |
| Energy variable | Thermal (not total) | trade-off vs shock fidelity |

### 4.4 Worked example: WSA tuning → wind speed

**적도 ($\theta=90°$), expansion factor $F=10$**:
- 원본: $V_R = 285 + 575/10^{0.588} = 285 + 148 = 433$ km/s
- 수정: $V_R = 150 + 1500/10^{0.6} = 150 + 376 = 526$ km/s

**$F=2$ (코로나 홀, 작은 expansion)**:
- 원본: $V_R = 285 + 575/2^{0.588} = 285 + 383 = 668$ → clip 625 km/s
- 수정: $V_R = 150 + 1500/2^{0.6} = 150 + 989 = 1139$ → clip 625 km/s

**$F=50$ (큰 expansion, slow streamer belt)**:
- 원본: $V_R = 285 + 575/50^{0.588} = 285 + 60 = 345$ km/s
- 수정: $V_R = 150 + 1500/50^{0.6} = 150 + 134 = 284$ → clip 275 km/s

→ 수정판이 fast/slow contrast를 더 sharp하게 만든다 (Ulysses의 고위도 700+ km/s 관측에 더 부합).

### 4.5 Trace 변수 (CME / polarity tracers)

본 논문이 추가한 두 passive scalar 추적자:
$$\partial_t\rho_c + \nabla\cdot(\rho_c\mathbf{V}) = 0\quad\text{(CME injected mass tracer)}$$
$$\partial_t\rho_p + \nabla\cdot(\rho_p\mathbf{V}) = 0\quad\text{(magnetic-polarity tracer)}$$

**용도 / Purpose**:
- $\rho_c$: 시뮬레이션 시각화에서 ICME 물질이 ambient와 어떻게 섞이는지 추적 (Fig. 1, 2의 색상이 이것).
- $\rho_p$: heliospheric current sheet의 진화와 자기장 polarity 분포를 시각화.

이 둘은 dynamics에 영향 없는 **passive markers** — 분석/가시화 전용. 운영 forecasting에서도 Earth가 ICME 물질에 실제 잠겼는지 판단할 때 중요한 진단 변수.

### 4.6 ENLIL 격자/도메인 설계 / Grid & domain design

| 항목 | 본 논문 시점(2003) | 현재 운영(2026) |
|---|---|---|
| Inner boundary | 0.1 AU (21.5 $R_S$) | 0.1 AU (변경 없음) |
| Outer boundary | 5 AU 또는 2 AU | 1.7 AU (Earth 중심), 2 AU (행성용) |
| Geometry | spherical (typical) / Cartesian (near-Earth) | spherical 표준 |
| Grid | uniform 또는 Paramesh AMR | 일반적으로 fixed mesh, 256×60×180 정도 |
| Time-stepping | explicit, CFL 제한 | 동일 |
| 1 run 시간 (2003) | ~수시간 (병렬) | ~10–30분 (현대 클러스터) |

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1958 ─ Parker: 태양풍 예측 (Astrophys. J. 128)
         |
1965 ─ Brackbill & Barnes: ∇·B = 0 첫 수치 처리
         |
1985 ─ Pizzo: 2-D MHD CIR 시뮬레이션
         |
1990 ─ Wang & Sheeley: V ∝ 1/F empirical relation
         |
1991 ─ Detman et al.: 3-D MHD draping around plasmoids
1991 ─ Gosling et al.: ICME-storm 인과관계 정립
         |
1993 ─ Hundhausen: SMM CME observations 종합
1993 ─ Usmanov: 첫 global 3-D MHD heliosphere
         |
1994 ─ Odstrčil: solar wind streams + small structures (J. Geophys. Res. 99)
         |
1996 ─ Tóth & Odstrčil: TVDLF for MHD (J. Comput. Phys. 128)
         |
1999 ─ Odstrčil & Pizzo (Cases 1, 2): 3-D CME in structured wind
1999 ─ Linker et al.: Whole Sun Month coronal MHD
         |
2000 ─ Tóth: ∇·B = 0 constraint scheme (J. Comput. Phys. 161)
2000 ─ MacNeice et al.: Paramesh AMR (Comput. Phys. Commun. 126)
2000 ─ Arge & Pizzo: real-time WSA improvement (J. Geophys. Res. 105)
         |
2001 ─ Riley et al.: empirically-driven coronal MHD
         |
2002 ─ CISM 설립 (NSF/STC)
2002 ─ Odstrčil et al.: coronal-heliospheric 2-D MHD merging
         |
★ 2003 ─ Odstrčil (현재 논문): ENLIL 종합 + 4개 응용
         |
2003 ─ Halloween storms → 운영 우주기상 수요 폭발
         |
2006 ─ STEREO 발사 → Fig. 7-9 합성 영상의 실측
         |
2008 ─ NOAA SWPC: WSA-ENLIL 운영 채택
         |
2010s ─ ENLIL+Cone 표준화; CCMC 호스팅
         |
2018 ─ EUHFORIA (KU Leuven) — 유럽판 ENLIL 후속
2018 ─ Parker Solar Probe 발사
         |
2020 ─ Solar Orbiter 발사 → ENLIL inner boundary 검증 강화
         |
2024 ─ Gannon storm (G5) — WSA-ENLIL+Cone forecast의 직접적 운영
         |
2025+─ DAGGER, ML emulator 등이 ENLIL을 ground truth로 학습
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Parker (1958)** "Dynamics of the interplanetary gas and magnetic fields" *Astrophys. J.* 128 | 태양풍 자체의 존재 예측 — ENLIL의 전제 | 모든 heliospheric MHD의 출발점 / Foundational solar-wind theory |
| **Wang & Sheeley (1990)** "Solar wind speed and coronal flux-tube expansion" *Astrophys. J.* 355 | $V \propto 1/F$ 경험식의 원본 — WSA의 W·S | ENLIL inner boundary의 결정적 component |
| **Arge & Pizzo (2000)** "Improvement in the prediction of solar wind conditions using near-real time solar magnetic field updates" *J. Geophys. Res.* 105 | Daily-updated WSA (=A); 본 논문에서 파라미터 수정 | Operational WSA의 직접 전임 |
| **Tóth & Odstrčil (1996)** "Comparison of FCT and TVDLF schemes for HD/MHD" *J. Comput. Phys.* 128 | ENLIL의 수치 핵심 (TVDLF) | 수치 기법의 단일 source |
| **Tóth (2000)** "$\nabla\cdot\mathbf{B}=0$ constraint in shock-capturing MHD" *J. Comput. Phys.* 161 | Field-interpolated CD scheme | 자기장 보존의 표준 |
| **MacNeice et al. (2000)** "Paramesh: parallel AMR community toolkit" *Comput. Phys. Commun.* 126 | ENLIL의 AMR 적층 | 병렬 격자 정련 인프라 |
| **Odstrčil & Pizzo (1999a,b,c)** "3-D CME propagation in structured solar wind" *J. Geophys. Res.* 104 | 본 논문 §3 (Fig. 1, 2)의 원본 데이터 | Global 3-D CME 동역학의 base series |
| **Linker et al. (1999)** "MHD modeling of solar corona during whole sun month" *J. Geophys. Res.* 104 | SAIC 코로나 코드 (현재 MAS) | 본 논문 §5 결합의 코로나 측 |
| **Hundhausen (1993)** "Sizes and locations of CMEs: SMM observations" *J. Geophys. Res.* 98 | CME-streamer belt 관계의 관측적 기반 | 본 논문 §3의 가정의 출처 |
| **Riley et al. (1997, 2001, 2002)** 시리즈 | 2-D CME 시뮬, empirically-driven coronal MHD, post-eruption reconnection | 본 논문 §4-5의 직접적 자매 작업 |
| **#18 (Hundhausen 1972 CME 또는 동등 SP CME 리뷰)** | CME 물리의 기초 — ENLIL의 입력 가정 | 선행 논문(prerequisite) |
| **Burlaga et al. (1987)** "Compound streams, magnetic clouds, major storms" *J. Geophys. Res.* 92 | CME-CME / CME-CIR 상호작용의 관측적 정립 | 본 논문 §3 결론의 관측적 모티브 |
| **Detman et al. (1991)** "3-D MHD numerical study of IP magnetic draping" *J. Geophys. Res.* 96 | 본 논문 §5의 magnetic draping 인용 | 자기 draping의 선구 시뮬레이션 |

---

## 7. References / 참고문헌

**MHD 시뮬레이션 인프라 / MHD simulation infrastructure**
- Tóth, G., & Odstrčil, D. (1996). "Comparison of some flux corrected transport and total variation diminishing numerical schemes for HD and MHD problems," *J. Comput. Phys.*, **128**, 82–100.
- Tóth, G. (2000). "The $\nabla\cdot\mathbf{B}=0$ constraint in shock-capturing magnetohydrodynamic codes," *J. Comput. Phys.*, **161**, 605–652.
- MacNeice, P., Olson, K. M., Mobarry, C., de Fainchtein, R., & Parker, C. (2000). "Paramesh: A parallel adaptive mesh refinement community toolkit," *Comput. Phys. Commun.*, **126**, 330–354.

**WSA 모델 / WSA model**
- Wang, Y.-M., & Sheeley, N. R. Jr. (1990). "Solar wind speed and coronal flux-tube expansion," *Astrophys. J.*, **355**, 726–732.
- Arge, C. N., & Pizzo, V. J. (2000). "Improvement in the prediction of solar wind conditions using near-real time solar magnetic field updates," *J. Geophys. Res.*, **105**, 10,465–10,480.
- Arge, C. N., Odstrčil, D., Pizzo, V. J., & Mayer, L. (2002). "Improved method for specifying solar wind speed near the Sun," in *Proc. Solar Wind 10*.

**3-D CME 전파 / 3-D CME propagation**
- Odstrčil, D., & Pizzo, V. J. (1999a). "Three-dimensional propagation of CMEs in a structured solar wind flow 1. CME launched within the streamer belt," *J. Geophys. Res.*, **104**, 483–492.
- Odstrčil, D., & Pizzo, V. J. (1999b). "Three-dimensional propagation of CMEs in a structured solar wind flow 2. CME launched adjacent to the streamer belt," *J. Geophys. Res.*, **104**, 493–503.
- Odstrčil, D., & Pizzo, V. J. (1999c). "Distortion of interplanetary magnetic field by 3-D CMEs in a structured solar wind," *J. Geophys. Res.*, **104**, 28,225–28,239.
- Odstrčil, D. (1994). "Interactions of solar wind streams and related small structures," *J. Geophys. Res.*, **99**, 17,653–17,671.

**코로나-행성간 결합 / Coronal-heliospheric coupling**
- Linker, J. A., Mikic, Z., Biesecker, D. A., et al. (1999). "MHD modeling of the solar corona during whole sun month," *J. Geophys. Res.*, **104**, 9809–9830.
- Riley, P., Linker, J. A., & Mikic, Z. (2001). "Empirically-driven global MHD model of the solar corona and inner heliosphere," *J. Geophys. Res.*, **106**, 15,889–15,901.
- Riley, P., Linker, J., Mikic, Z., Odstrčil, D., Pizzo, V. J., & Webb, D. F. (2002). "Evidence for post-eruption reconnection associated with CMEs in the solar wind," *Astrophys. J. Lett.*, **578**, 972–978.
- Odstrčil, D., Linker, J. A., Lionello, R., et al. (2002b). "Merging of coronal and heliospheric numerical 2-D MHD models," *J. Geophys. Res.*, **107**(A12), 1493.

**관측·CME 물리 / Observations & CME physics**
- Hundhausen, A. J. (1993). "Sizes and locations of coronal mass ejections: SMM observations from 1980 and 1984–1989," *J. Geophys. Res.*, **98**, 13,177–13,200.
- Burlaga, L. F., Behannon, K. W., & Klein, L. W. (1987). "Compound streams, magnetic clouds, and major geomagnetic storms," *J. Geophys. Res.*, **92**, 5725–5734.
- Gosling, J. T., McComas, D. J., Phillips, J. L., & Bame, S. J. (1991). "Geomagnetic activity associated with Earth passage of IP shock disturbances and CMEs," *J. Geophys. Res.*, **96**, 7831–7839.
- Tsurutani, B. T., Kamide, Y., Arballo, J. K., Gonzales, W. D., & Lepping, R. P. (1999). "Interplanetary causes of great and superintense magnetic storms," *Phys. Chem. Earth (C)*, **24**, 101–105.
- Detman, T. R., Dryer, M., Yeh, T., Han, S. M., Wu, S. T., & McComas, D. J. (1991). "A time-dependent, 3-D MHD numerical study of IP magnetic draping around plasmoids in the solar wind," *J. Geophys. Res.*, **96**, 9531–9540.

**선구 3-D MHD / Pioneer 3-D MHD**
- Usmanov, A. V. (1993). "A global numerical 3-D MHD model of the solar wind," *Sol. Phys.*, **145**, 377–396.
- Riley, P., Gosling, J. T., & Pizzo, V. J. (1997). "A 2-D simulation of the radial and latitudinal evolution of a solar wind disturbance driven by a fast, high-pressure CME," *J. Geophys. Res.*, **102**, 14,677–14,685.

**기준 인용 (이 논문)**
- Odstrčil, D. (2003). "Modeling 3-D solar wind structure," *Adv. Space Res.*, **32**(4), 497–506. DOI: 10.1016/S0273-1177(03)00332-6
