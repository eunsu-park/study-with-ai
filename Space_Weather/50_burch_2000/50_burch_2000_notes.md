---
title: "IMAGE Mission Overview"
authors: J. L. Burch
year: 2000
journal: "Space Science Reviews"
doi: "10.1023/A:1005245323115"
topic: Space_Weather
tags: [IMAGE, magnetosphere, ENA, EUV, FUV, RPI, MIDEX, imaging, ring_current, plasmasphere]
status: completed
date_started: 2026-04-25
date_completed: 2026-04-25
---

# 50. IMAGE Mission Overview / IMAGE 임무 개요

---

## 1. Core Contribution / 핵심 기여

**EN**: Burch (2000) is the lead overview paper for the *Space Science Reviews* special issue introducing **IMAGE — the first satellite mission ever dedicated to imaging Earth's magnetosphere**. Until 2000, magnetospheric physics was almost entirely an *in situ* discipline: Polar, Geotail, Wind, Equator-S and their predecessors flew through plasma regions and reported point-by-point measurements, leaving the global structure to be reconstructed statistically over months. IMAGE inverts this. From a 1000 km × 7 R$_E$ highly elliptical polar orbit, an instrument suite of eight cameras spread across **three imaging modalities** — neutral atom imaging (NAI: LENA, MENA, HENA covering 10 eV to 500 keV), photon imaging (FUV at 121.6/135.6/140–190 nm and EUV at 30.4 nm), and active radio sounding (RPI, 3 kHz–3 MHz) — produces *global, simultaneous*, two-minute-cadence snapshots of the plasmasphere, ring current, near-Earth plasma sheet, polar cusp, and aurora.

**KR**: Burch (2000)는 *Space Science Reviews* 특별호의 머리말 격 개요 논문으로서 **자기권 영상화에 전용으로 설계된 최초의 위성 임무 IMAGE**를 소개한다. 2000년 이전까지 자기권물리학은 거의 전적으로 *in situ*(현장 측정) 학문이었다 — Polar, Geotail, Wind, Equator-S와 그 이전 위성들은 플라즈마 영역을 통과하며 점 단위 측정을 보고했고, 전역 구조는 수개월에 걸쳐 통계적으로 재구성해야 했다. IMAGE는 이를 뒤집는다. 1000 km × 7 R$_E$ 고이심률 극궤도에서, 세 가지 영상 모달리티 — 중성원자 영상(NAI: LENA, MENA, HENA가 10 eV–500 keV를 분담), 광자 영상(FUV 121.6/135.6/140–190 nm, EUV 30.4 nm), 능동 라디오 사운딩(RPI, 3 kHz–3 MHz) — 에 걸친 8개 카메라가 *전역·동시·2분 주기*로 플라즈마권·환전류·근지구 플라즈마 시트·극 커스프·오로라의 스냅샷을 생성한다.

**EN (cont.)**: Beyond hardware, the paper crystallises three conceptual contributions: (1) a *global imaging paradigm* for magnetospheric physics, mirroring the way astrophysics has used photons since Galileo; (2) a *24-hour completely open data policy* with no proprietary intervals, level-1 browse products on the web within one day, which became the template for later NASA missions and for the modern Heliophysics System Observatory; and (3) a *real-time 44 kb s$^{-1}$ broadcast* downlink usable by any 6 m dish, making IMAGE the first science mission to deliberately serve operational space-weather forecasting. Operationally, the paper also commits to a 100 % duty cycle, two-minute spin-period image cadence (one-minute for RPI, ten-minute integrations for EUV), and a single Central Instrument Data Processor (CIDP) for the entire payload.

**KR (계속)**: 하드웨어를 넘어, 본 논문은 세 가지 개념적 기여를 결정화한다: (1) 자기권물리학을 위한 *전역 영상화 패러다임* — 갈릴레오 이래 천체물리학이 광자를 사용해온 방식과 동일; (2) 사유 기간 없이 level-1 브라우즈 자료를 24시간 안에 웹에 공개하는 *완전 개방 데이터 정책* — 후속 NASA 임무들과 현대 Heliophysics System Observatory의 표본; (3) 6 m 직경의 어떤 안테나로도 수신 가능한 *실시간 44 kb s$^{-1}$ 방송* 다운링크 — 운영 우주기상 예보에 의도적으로 봉사한 최초의 과학 임무. 운영적으로 본 논문은 100 % 듀티사이클, 2분 회전주기에 맞춘 영상 주기(RPI 1분, EUV 10분 적분), 그리고 전 탑재체를 단일 중앙기기데이터처리기(CIDP)로 통합하는 설계를 약속한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Mission Concept / §1 서론과 임무 개념 (pp. 1–3)

**EN**: Burch opens with a candid concession: 35+ years of in situ data have produced excellent statistical pictures, but they fundamentally cannot resolve **dynamics on the substorm/storm time scales** (minutes to hours) because a single spacecraft samples one trajectory while *both* time and space vary along that trajectory. The driving question is stated bluntly: *"How does the magnetosphere respond globally to the changing conditions in the solar wind?"* He notes that astrophysics, solar physics, and even ionospheric physics already enjoy the global perspective; magnetospheric physics is the laggard. IMAGE addresses three concrete sub-questions: (1) dominant plasma injection mechanisms on substorm/storm scales, (2) the directly driven response to solar wind changes, and (3) where/how plasmas are energised, transported, and lost.

**KR**: Burch는 솔직한 인정으로 시작한다 — 35년 이상의 *in situ* 데이터는 훌륭한 통계적 그림을 만들어냈지만, 단일 위성이 하나의 궤적을 샘플링할 때 시간과 공간이 함께 변하기 때문에 **서브스톰·자기폭풍 시간 스케일(분∼시간)의 동역학**을 본질적으로 분해할 수 없다. 그는 추진 질문을 단호히 제시한다: *"자기권은 변화하는 태양풍 조건에 대해 전역적으로 어떻게 반응하는가?"* 그는 천체물리·태양물리·전리권물리는 이미 전역 시각을 누리고 있으나 자기권물리학만이 뒤처져 있음을 지적한다. IMAGE는 세 가지 구체적 하위 질문을 다룬다: (1) 서브스톰/폭풍 스케일의 우세한 플라즈마 주입 메커니즘, (2) 태양풍 변화에 대한 직접 구동 반응, (3) 플라즈마가 어디서·어떻게 에너지화·수송·소실되는가.

**EN**: The instrument suite is enumerated: NAI over 10 eV–500 keV (split into LENA/MENA/HENA for technological reasons — no single detector can span 4.7 decades in energy), FUV at 121–190 nm (split into SI for spectrographic line discrimination, WIC for wideband electron-aurora imaging, and GEO for geocoronal Lyman-α needed to deconvolve the neutral atom images), EUV at 30.4 nm for plasmaspheric He$^+$ resonance scattering, and RPI for radio plasma imaging over 0.1–10$^5$ cm$^{-3}$ density range. The IMAGE satellite is launched into an elliptical orbit at 90° inclination, **apogee 7 R$_E$, perigee 1000 km** — Figure 1 shows the orbit. The line of apsides initially sits at 40° latitude and precesses over the pole during the two-year nominal mission, returning to 40°. This precession is the key to surveying both hemispheres' polar regions during one mission lifetime.

**KR**: 기기 묶음이 열거된다: NAI 10 eV–500 keV (4.7 자릿수 에너지 범위는 단일 검출기로 불가능하므로 LENA/MENA/HENA로 분할), FUV 121–190 nm (선 분광 분리를 위한 SI, 광대역 전자 오로라용 WIC, 그리고 중성원자 영상 디컨볼루션에 필요한 지구코로나 Lyman-α용 GEO), EUV 30.4 nm는 플라즈마권 He$^+$ 공명 산란용, RPI는 0.1–10$^5$ cm$^{-3}$ 밀도 범위에 걸친 라디오 플라즈마 영상. IMAGE 위성은 90° 경사 타원궤도, **원지점 7 R$_E$, 근지점 1000 km**로 발사된다 (그림 1). 장축선은 초기에 위도 40°에 위치하며 2년 명목 임무 동안 극을 넘어 다시 40°로 세차한다. 이 세차가 한 임무 수명 동안 양 반구의 극지를 모두 조사하는 핵심이다.

### Part II: Science Objectives / §2 과학 목표 (pp. 3–7)

#### §2.1 Solar-wind plasma injection / 태양풍 플라즈마 주입

**EN**: The magnetopause was known to be in continuous motion with a mixed boundary layer, but the **relative global importance of magnetic reconnection vs. diffusive entry** remained unsettled. Reconnection produces sharp gradients across a variable-thickness boundary; diffusion produces smooth gradients across a uniform-thickness layer. Single in situ crossings cannot distinguish these because they conflate boundary thickness with crossing geometry. Onsager et al. (1993) modelled cusp ions as quasi-steady reconnection signatures; Lockwood & Smith (1992) modelled the same data as pulsed reconnection. IMAGE solves this with the RPI plasmagram (Figure 2 — a simulated range-vs-frequency plot showing characteristic signatures from cusp, magnetopause, polar cap, plasmapause, ionosphere, plasmasphere core), supported by NAI cusp imaging and FUV electron/proton aurora maps. When IMAGE is at high altitude with the cusp in the RPI field, it provides high-time-resolution shape images of the cusp; FUV maps the cusp footprint in the ionosphere; at low altitudes, NAI gives an alternative cusp view that cross-validates the high-altitude picture.

**KR**: 자기권계면은 가변 두께 경계층과 함께 연속적으로 운동하는 것으로 알려져 있었으나, **자기 재결합 대 확산성 진입의 상대적 전역 중요도**는 미정이었다. 재결합은 가변 두께 경계에 걸친 급격한 기울기를 만들고, 확산은 균일 두께 층에 걸친 부드러운 기울기를 만든다. 단일 *in situ* 통과로는 경계 두께와 통과 기하가 혼재하므로 두 시나리오를 구분할 수 없다. Onsager 등(1993)은 커스프 이온을 준정상 재결합 신호로 모형화했고, Lockwood & Smith (1992)는 동일 데이터를 펄스형 재결합으로 모형화했다. IMAGE는 RPI 플라즈마그램(그림 2 — 거리 vs 주파수 모의 플롯, 커스프·자기권계면·극관·플라즈마경계·전리권·플라즈마권 핵의 특성 신호 표시)으로 이를 해결하며, NAI 커스프 영상과 FUV 전자·양성자 오로라 지도가 보완한다. 고도가 높을 때 RPI 시야에 커스프가 들어오면 커스프 형상을 고시간분해능으로 영상화하고, FUV는 전리권의 커스프 발자국을 사상하며, 저고도에서는 NAI가 또 다른 커스프 시점을 제공해 고고도 그림과 교차 검증한다.

#### §2.2 Ionospheric plasma injection / 전리권 플라즈마 주입

**EN**: The ionosphere is a major plasma source for the magnetosphere, but the **source localisation has been controversial for 15 years**. Moore et al. (1985) argued for an intense localised cusp source; Shelley et al. (1985) described a diffuse and extensive source spanning the entire auroral oval. The controversy persists because statistical surveys (months of data) cannot resolve the *minute-scale* spatial/temporal features. Figure 3 shows a simulated LENA image of charge-exchanged O$^+$ outflow as viewed from above the northern polar regions with the noon meridian at the bottom — colour-coded oxygen atom flux on the left, raw instrument counts on the right. With several-minute cadence, LENA delivers ion outflow flux and composition (H$^+$ vs O$^+$) down to 10 eV as a function of magnetospheric activity simultaneously characterised by FUV (auroral activity) and RPI (magnetopause boundary location).

**KR**: 전리권은 자기권의 주요 플라즈마 공급원이지만 **공급원 위치 논쟁은 15년간 지속**되어 왔다. Moore 등(1985)은 강한 국지적 커스프 공급원을 주장했고, Shelley 등(1985)은 오로라 타원 전체에 걸친 분산되고 확장된 공급원을 기술했다. 통계 조사(수개월 데이터)는 *분 단위* 공간·시간 특징을 해상하지 못해 논쟁이 지속된다. 그림 3은 북극 위에서 정오 자오선이 아래 중앙에 오도록 본 전하교환 O$^+$ 유출의 LENA 모의 영상이다 (좌: 색상화된 산소 원자 플럭스, 우: 원시 기기 계수). 수분 주기로 LENA는 자기권 활동의 함수로서 이온 유출 플럭스와 조성(H$^+$ 대 O$^+$)을 10 eV까지 측정하며, 이때 활동도는 FUV(오로라 활동)와 RPI(자기권계면 위치)로 동시 특징지어진다.

#### §2.3 Plasmaspheric dynamics / 플라즈마권 동역학

**EN**: The plasmapause is traditionally interpreted as the boundary between closed and open convection trajectories — when the open/closed boundary moves inward, filled flux tubes become entrained and form long sunward-drawn tails (Rasmussen et al. 1993). But observations require a more complicated picture: detached high-density regions (Chappell 1974), low-density "biteouts" suggesting precipitation loss comparable to convection loss, nightside plasmapause steepening with $K_p$ (Chappell et al. 1970), and rapid radial motion across broad MLT sectors. **No progress in 20+ years** because no images of plasmaspheric ions existed. Figure 4 shows a simulated EUV image at 30.4 nm from 7 R$_E$ over the north pole — a bright doughnut of resonantly scattered solar EUV from He$^+$, with a slot-like dark region (Earth's shadow) at the upper left. EUV will deliver *2-D line-of-sight* He$^+$ column densities at 2–10 minute cadence; RPI identifies internal density structures (biteouts, closely wrapped tails, field-aligned features) hidden in the integrated EUV; NAI ring-current images identify ring current–plasmasphere interactions; FUV anchors the whole sequence to magnetospheric activity through auroral morphology.

**KR**: 플라즈마경계는 전통적으로 닫힌 대류 궤도와 열린 대류 궤도의 경계로 해석된다 — 열린/닫힌 경계가 안쪽으로 이동하면 채워진 자속관이 휩쓸려 들어가 태양 방향으로 길어진 꼬리를 형성한다(Rasmussen 등 1993). 그러나 관측은 더 복잡한 그림을 요구한다: 분리된 고밀도 영역(Chappell 1974), 강수 손실이 대류 손실과 비교될 만큼 큼을 시사하는 저밀도 "바이트아웃", $K_p$ 증가에 따른 야간 측 플라즈마경계 가팔라짐(Chappell 등 1970), 넓은 MLT 섹터에 걸친 급속한 반경 운동. **20여 년간 진전이 없었던** 이유는 플라즈마권 이온의 영상이 없었기 때문이다. 그림 4는 북극 위 7 R$_E$에서 본 30.4 nm EUV 모의 영상 — He$^+$의 공명 산란 태양 EUV가 만든 밝은 도넛, 좌상단에는 슬롯 모양 어두운 영역(지구 그림자). EUV는 2–10분 주기로 *2차원 시선* He$^+$ 컬럼 밀도를 제공하고, RPI는 적분 EUV에 가려진 내부 밀도 구조(바이트아웃, 가까이 감긴 꼬리, 자기력선 정렬 구조)를 식별하며, NAI 환전류 영상은 환전류-플라즈마권 상호작용을 식별하고, FUV는 오로라 형태로 전체 시퀀스를 자기권 활동도에 정박한다.

#### §2.4 Ring current injection, build-up, and decay / 환전류 주입·축적·소멸

**EN**: To characterise the **respective roles of ionospheric injection, in situ acceleration, and earthward transport** during substorm injections, IMAGE needs *composition-resolved* images of plasma sheet and ring current ions across the energy range from cold ionospheric (~eV) to ring current (~hundreds keV). Storm energy initially residing in the ring current is lost first to drift across the magnetopause, then to charge exchange with the geocorona (the dominant loss for stably trapped ions), then to pitch-angle diffusion via wave–particle interactions producing precipitation. By correlated study of FUV (auroral images), EUV+RPI (plasmasphere), and NAI (hot plasma), all source and loss processes of the ring current can be assessed *globally and quantitatively*. Figure 5 simulates ring current injection in the nightside magnetosphere: 1.7 keV H$^+$ flux (left), neutral H flux from charge exchange (centre), MENA image counts for a 120 s exposure (right). NAI images are line-of-sight integrals deconvolvable into equatorial pitch-angle distributions — Roelof (2000) and Perez (2000) develop two complementary deconvolution techniques.

**KR**: 서브스톰 주입 동안 **전리권 주입, *in situ* 가속, 지구방향 수송 각각의 역할**을 특징짓기 위해 IMAGE는 차가운 전리권(~eV)에서 환전류(~수백 keV)에 이르는 에너지 범위에서 *조성 분해* 영상이 필요하다. 초기에 환전류에 머무는 폭풍 에너지는 먼저 자기권계면을 가로지르는 표류로 손실되고, 이어서 지구코로나와의 전하교환(안정 포획 이온의 우세 손실 메커니즘), 그리고 파동-입자 상호작용을 통한 피치각 확산이 강수를 만들어 손실된다. FUV(오로라), EUV+RPI(플라즈마권), NAI(고온 플라즈마)의 상관 연구로 환전류의 모든 공급·손실 과정을 *전역적·정량적*으로 평가할 수 있다. 그림 5는 야간 측 환전류 주입 모의: 1.7 keV H$^+$ 플럭스(좌), 전하교환에 의한 중성 H 플럭스(중), MENA의 120 s 노출 영상 계수(우). NAI 영상은 시선 적분이며 적도 피치각 분포로 디컨볼루션 가능 — Roelof (2000)와 Perez (2000)이 두 가지 상보적 기법을 제시한다.

### Part III: Science Payload / §3 과학 탑재체 (pp. 7–10)

**EN**: All eight instruments communicate with the spacecraft through a single payload computer — the **Central Instrument Data Processor (CIDP)** — for commanding and data downlink. Figure 6 shows the payload layout (top view of the spinning octagonal deck): HENA, MENA, LENA, EUV at the periphery for unobstructed sky views; FUV-WIC, FUV-SI, FUV-GEO clustered centrally; RPI X- and Y-axis couplers and deployers around the perimeter to anchor the long crossed-dipole antennas. Figure 7 is a photograph of the integrated payload before installation. Table I summarises the measurement objectives of NAI, EUV, FUV, and RPI:

| Instrument | Energy / Wavelength | FOV | Spatial / Spectral / Angular Resolution | Image Time |
|---|---|---|---|---|
| **NAI** (LENA 10–300 eV, MENA 1–30 keV, HENA 30–500 keV) | 10 eV – 500 keV | 90° × 90° (ring current at apogee) | 4°×4° to 8°×8° (energy/mass dependent); ΔE/E ≤ 0.8 above 1 keV; H/O species separation | 2 min (resolve substorms) |
| **EUV** | 30.4 nm | 90° × 90° (plasmasphere from apogee) | 0.1 R$_E$ from apogee | 2–10 min (plasmaspheric processes) |
| **FUV** (WIC, SI, GEO) | 121.6, 135.6 nm; 140–190 nm | ≥ 16° aurora; 60° geocorona | 70 km (WIC), 90 km (SI); Δλ 0.2 nm near 121.6 nm (separate cold geocoronal H from hot proton precipitation; reject 130.4 nm; select 135.6 nm electron aurora) | 2 min |
| **RPI** | 3 kHz – 3 MHz radio sounding | n/a (omni) | 500 km spatial; 0.1 – 10$^5$ cm$^{-3}$ density | 1 min |

**KR**: 8개 기기 모두 단일 탑재체 컴퓨터인 **CIDP(중앙기기데이터처리기)**를 통해 위성과 통신한다(명령·다운링크). 그림 6은 회전하는 팔각형 데크의 상면도이다: HENA, MENA, LENA, EUV는 시야 차단을 피하기 위해 외곽에 배치되고, FUV-WIC, FUV-SI, FUV-GEO는 중앙에 집중되며, RPI X·Y축 커플러와 디플로이어는 둘레에 분산되어 긴 십자 다이폴 안테나를 고정한다. 그림 7은 통합된 탑재체의 설치 전 사진이다. 표 I은 NAI, EUV, FUV, RPI의 측정 목표를 위 표와 같이 요약한다.

### Part IV: Mission Operations and Data Analysis / §4 임무 운영과 데이터 분석 (pp. 10–11)

**EN**: After launch, **deployment of RPI antennas and instrument activation takes ~1 month**. Once science operations begin, all instruments run with a 100 % duty cycle. Time resolution is set by the **2-minute spin period** for all instruments except RPI (1 min) and EUV (5-spin integration → 10 min). Mode changes and command uploads are limited to once per week. The prime data downlink is *store-and-forward* to the NASA Deep Space Network once per orbit (~13 hours). For forecasting, the full **44 kb s$^{-1}$ data rate is broadcast in real time** and can be received by any 6 m dish; Japan's Communications Research Center is one such station.

**KR**: 발사 후 **RPI 안테나 전개와 기기 활성화에 약 1개월**이 소요된다. 과학 운영 시작 후 모든 기기는 100 % 듀티사이클로 작동한다. 시간 분해능은 **2분 회전주기**로 RPI(1분)와 EUV(5회전 적분 → 10분)를 제외한 모든 기기에 적용된다. 모드 변경과 명령 업로드는 주 1회로 제한된다. 주요 데이터 다운링크는 궤도당 1회(~13시간) NASA 딥스페이스 네트워크로의 *저장-전달* 방식이다. 예보를 위해 **44 kb s$^{-1}$ 전 데이터율이 실시간 방송**되며, 어떤 6 m 직경 안테나로도 수신 가능하다 — 일본의 통신연구센터(CRC)가 그러한 수신국 중 하나이다.

**EN**: A central pillar is the **completely open data set**: within 24 hours of acquisition by SMOC (Science and Mission Operations Center) at GSFC, level-1 browse products are online via the WWW. Browse products include orbital plot, RPI sky map, EUV (He$^+$), FUV-SI proton aurora, FUV-WIC electron aurora, and LENA/MENA/HENA NAI images, displayed in GIF format and viewable as movies (Figure 8 — sample browse product layout). The full science set sits at SMOC for two months, then archives at the National Space Science Data Center (NSSDC). All data are stored in **Universal Data Format (UDF)** allowing single-program plotting across all instruments (Gurgiolo 1999).

**KR**: 핵심 기둥은 **완전 개방 데이터셋**: GSFC의 SMOC(과학·임무운영센터)가 자료를 획득한 지 24시간 안에 level-1 브라우즈 산출물이 WWW로 공개된다. 브라우즈 산출물에는 궤도 플롯, RPI 하늘지도, EUV(He$^+$), FUV-SI 양성자 오로라, FUV-WIC 전자 오로라, LENA/MENA/HENA NAI 영상이 포함되며, GIF 포맷으로 표시되고 동영상으로 볼 수 있다(그림 8 — 표본 브라우즈 산출물 배치). 전체 과학 데이터셋은 SMOC에 2개월 거주 후 국립우주과학자료센터(NSSDC)에 보관된다. 모든 데이터는 **Universal Data Format(UDF)**로 저장되어 단일 프로그램으로 모든 기기 데이터를 함께 플로팅할 수 있다(Gurgiolo 1999).

### Part IV.b: Browse-Product Pipeline Walk-through / 브라우즈 산출물 파이프라인 워크스루

**EN**: Figure 8 illustrates the level-1 browse layout that becomes available within 24 hours of acquisition: the *upper row* shows orbit (IMAGE position relative to the Sun direction), RPI sky map (echo direction colour-coded by minimum range, in km looking up — the sample shows a feature at ~1000 km extending to ~10000 km), and Dst (the upper-right panel intentionally shows historical Dst — real-time Dst is delayed; IMP-8 IMF data fills in when available). The *middle row* shows EUV at 30.4 nm (a doughnut-shaped He$^+$ image with elevation/azimuth axes in degrees), FUV-SI 121.6 nm proton aurora image (counts/spin), and FUV-SI 135.6 nm electron aurora image (counts/spin) with image effective area $A_{\text{eff}} \approx 1.50$ cm$^2$ and SS ≈ 1.34 keV. The *bottom row* shows LENA, MENA, and HENA neutral atom images (1.5 keV H, 30.6 keV H respectively) with log counts colour-bars. A web user can scroll through 13-hour orbit-by-orbit movies and quickly identify storm onset, plasmaspheric erosion events, or auroral substorm expansions.

**KR**: 그림 8은 자료 획득 후 24시간 내에 공개되는 level-1 브라우즈 배치를 예시한다: *상단 행*은 궤도(태양 방향 대비 IMAGE 위치), RPI 하늘지도(최소 거리 km, 위에서 본 모습 — 표본은 ~1000 km에서 ~10000 km까지 확장된 특징을 보여줌), Dst(우상단 패널은 역사적 Dst를 의도적으로 보여줌 — 실시간 Dst는 지연되며, IMP-8 IMF 데이터가 가용할 때 채워짐). *중간 행*은 30.4 nm EUV(고도/방위각 축의 도넛형 He$^+$ 영상), FUV-SI 121.6 nm 양성자 오로라(회전당 계수, 유효면적 ≈ 1.50 cm$^2$, SS ≈ 1.34 keV), FUV-SI 135.6 nm 전자 오로라(유효면적 ≈ 0.50 cm$^2$, SS ≈ 23.00 keV). *하단 행*은 LENA(50 eV O$^+$, 채널 4), MENA(1.5 keV H), HENA(30.6 keV H) 중성원자 영상으로 로그 계수 컬러바를 사용한다. 웹 사용자는 궤도당 13시간 분량의 동영상을 스크롤하여 폭풍 시작, 플라즈마권 침식, 오로라 서브스톰 팽창을 신속히 식별할 수 있다.

### Part V: Science Team and Summary / §5–6 (pp. 11–14)

**EN**: The science team is large and international (§5): co-investigators from SwRI (PI Burch, MENA lead Pollock, NAI lead Young), GSFC (Green RPI science lead, Moore LENA lead/modelling), APL (Mitchell HENA lead), Berkeley (Mende FUV lead), Arizona (Sandel EUV lead), UMass-Lowell (Reinisch RPI instrument lead), plus contributions from Bern, Paris-Meudon, Maryland, Rutherford-Appleton, Calgary, Rice, Logica, Raytheon, Liège, Los Alamos, ISAS Japan, and others. Participating scientists add inversion specialists (Roelof, Perez), modellers (Fok, Wilson, Schulz), and education/public outreach coordinators (Reiff, Taylor, Odenwald). Section 6 (Summary) reiterates the launch window (winter 1999–2000), 7 R$_E$ apogee / 1000 km perigee, two-year nominal mission, 100 % duty cycle, 44 kb s$^{-1}$ data rate, 24-hour data release, and emphasises that **critical tests of pulsed-vs-steady reconnection, detached plasma regions, substorm injections, and ring current injection/decay** become possible for the first time, and that IMAGE will provide global context for the contemporary in situ missions Cluster 2 and Polar.

**KR**: 과학팀은 대규모 국제 팀이다(§5): 공동연구자는 SwRI(PI Burch, MENA 책임 Pollock, NAI 책임 Young), GSFC(RPI 과학 책임 Green, LENA 책임/모델링 Moore), APL(HENA 책임 Mitchell), Berkeley(FUV 책임 Mende), Arizona(EUV 책임 Sandel), UMass-Lowell(RPI 기기 책임 Reinisch), 그리고 Bern, Paris-Meudon, Maryland, Rutherford-Appleton, Calgary, Rice, Logica, Raytheon, Liège, Los Alamos, ISAS Japan 등에서의 기여로 구성된다. 참여 과학자에는 역변환 전문가(Roelof, Perez), 모델러(Fok, Wilson, Schulz), 교육·대중확산 코디네이터(Reiff, Taylor, Odenwald)가 추가된다. §6(요약)은 발사 기간(1999–2000년 겨울), 7 R$_E$ 원지점/1000 km 근지점, 2년 명목 임무, 100 % 듀티사이클, 44 kb s$^{-1}$ 데이터율, 24시간 자료 공개를 재언급하며, **펄스형-정상형 재결합, 분리 플라즈마 영역, 서브스톰 주입, 환전류 주입/소멸의 결정적 검증**이 처음으로 가능해지고, IMAGE가 동시대 *in situ* 임무인 Cluster 2 및 Polar에 전역 맥락을 제공할 것임을 강조한다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Imaging is the missing dimension of magnetospheric physics / 영상화는 자기권물리학에서 빠져 있던 차원이다** — *EN*: For 35+ years magnetospheric physics relied on point in situ measurements, generating excellent statistics but blurring time vs. space. IMAGE delivers the global snapshot that astrophysics has had since Galileo. *KR*: 35년 이상 자기권물리학은 점 측정에 의존하여 통계는 우수했지만 시간-공간 분리가 흐릿했다. IMAGE는 천체물리학이 갈릴레오 이래 누려온 전역 스냅샷을 자기권에 가져온다.

2. **Three modalities, one mission / 세 가지 모달리티, 하나의 임무** — *EN*: NAI (10 eV – 500 keV neutral atoms), photons (FUV 121–190 nm + EUV 30.4 nm), and active radio sounding (RPI 3 kHz – 3 MHz) cover hot, warm, and cold plasma populations simultaneously — no prior mission combined all three. *KR*: NAI(10 eV – 500 keV 중성원자), 광자(FUV 121–190 nm + EUV 30.4 nm), 능동 라디오 사운딩(RPI 3 kHz – 3 MHz)이 뜨거운·따뜻한·차가운 플라즈마를 동시에 다룬다 — 세 모달리티를 모두 결합한 임무는 IMAGE 이전에 없었다.

3. **Charge-exchange ENA imaging is line-of-sight tomography / 전하교환 ENA 영상은 시선 단층촬영이다** — *EN*: Each NAI pixel is a path-integral $\int n_H \sigma_{cx} j_{ion}\, d\ell$. With multiple viewing geometries from a precessing orbit, one can deconvolve to recover the 3-D ion distribution; Roelof (2000) and Perez (2000) provide complementary inversions. *KR*: 각 NAI 픽셀은 경로 적분 $\int n_H \sigma_{cx} j_{ion}\, d\ell$이다. 세차궤도의 다양한 시야 기하로부터 디컨볼루션하여 3차원 이온 분포를 복원할 수 있으며, Roelof (2000)과 Perez (2000)이 상보적 역변환을 제공한다.

4. **Highly elliptical polar orbit is the geometry enabler / 고이심률 극궤도가 기하학적 핵심이다** — *EN*: 7 R$_E$ apogee gives wide-angle global view; 90° inclination with apsides precessing 40°→90°→40° in two years lets IMAGE survey both polar regions; the ~13-hour orbit maximises high-altitude dwell time. *KR*: 7 R$_E$ 원지점은 광시야 전역 시점을, 90° 경사와 2년에 걸친 40°→90°→40° 장축선 세차는 양 극지 조사를, ~13시간 궤도는 고고도 체류시간 최대화를 제공한다.

5. **Pulsed vs. steady reconnection becomes a falsifiable hypothesis / 펄스 대 정상 재결합이 반증 가능한 가설이 된다** — *EN*: Onsager (1993) and Lockwood-Smith (1992) modelled identical cusp data with opposite reconnection paradigms. RPI plasmagrams + FUV cusp footprint + NAI cusp images jointly resolve which is dominant. *KR*: Onsager (1993)와 Lockwood-Smith (1992)는 동일한 커스프 데이터를 정반대의 재결합 패러다임으로 모형화했다. RPI 플라즈마그램 + FUV 커스프 발자국 + NAI 커스프 영상이 결합되어 어느 쪽이 우세한지를 결정한다.

6. **Operational space weather is born here / 운영 우주기상이 여기서 태어났다** — *EN*: The 44 kb s$^{-1}$ real-time broadcast and 24-hour-open level-1 products made IMAGE the first NASA science mission deliberately useful to forecasters; this directly anticipates today's NOAA SWFO architecture. *KR*: 44 kb s$^{-1}$ 실시간 방송과 24시간 개방 level-1 산출물 덕분에 IMAGE는 예보자에게 의도적으로 유용한 최초의 NASA 과학 임무가 되었다 — 오늘날의 NOAA SWFO 아키텍처의 직접적 전조이다.

7. **Spin-imaging discipline drives the cadence / 회전영상 규율이 주기를 정한다** — *EN*: The 2-minute spin period sets the natural cadence; almost every instrument's 2-minute integration rolls up to "resolve substorm development". RPI (1 min) and EUV (10 min) deviate for SNR reasons, never operational ones. *KR*: 2분 회전주기가 자연스러운 주기를 정하며, 거의 모든 기기의 2분 적분은 "서브스톰 전개 분해"로 통합된다. RPI(1분)와 EUV(10분)는 SNR 이유로 벗어나며 운영적 이유가 아니다.

8. **Cross-instrument inversion is the science multiplier / 기기 간 역변환이 과학 승수효과를 낸다** — *EN*: A LENA O$^+$ outflow image makes sense only when paired with FUV auroral context, RPI plasmapause location, and NAI ring current state. The CIDP + UDF + open-data architecture exists precisely so that a single user can plot all eight datastreams on one frame. *KR*: LENA O$^+$ 유출 영상은 FUV 오로라 맥락, RPI 플라즈마경계 위치, NAI 환전류 상태와 결합되어야만 의미가 있다. CIDP + UDF + 개방 데이터 아키텍처는 정확히 단일 사용자가 8개 데이터 스트림을 한 프레임에 플로팅할 수 있도록 존재한다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Charge-exchange ENA emissivity / 전하교환 ENA 방출률

The fundamental forward model for any line-of-sight ENA observation is:

$$
j_{\text{ENA}}(\hat{n}, E) \;=\; \int_{\text{LOS}} n_H(\mathbf{r})\, \sigma_{cx}(E)\, j_{\text{ion}}(\mathbf{r}, E)\, d\ell
$$

| Symbol | Meaning |
|---|---|
| $j_{\text{ENA}}(\hat n, E)$ | ENA differential flux at IMAGE in direction $\hat n$ (atoms cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$) |
| $n_H(\mathbf r)$ | geocoronal hydrogen density (cm$^{-3}$); typical model: $n_H(r) = n_0 (R_E/r)^{3}$ with $n_0 \sim 10^4$ cm$^{-3}$ |
| $\sigma_{cx}(E)$ | charge-exchange cross-section (cm$^2$); for $\text{H}^+ + \text{H}$ near 1 keV $\sigma_{cx} \approx 1.5\times10^{-15}$ cm$^2$, falling at higher $E$ |
| $j_{\text{ion}}(\mathbf r, E)$ | parent ion flux at point $\mathbf r$ (the unknown to be recovered) |
| $d\ell$ | path element along the line of sight |

**EN**: The recovery problem is ill-posed (Fredholm integral of the first kind). Roelof (2000) uses pitch-angle anisotropy and adiabatic invariance to constrain the solution; Perez (2000) employs a maximum-entropy approach. Both are detailed in companion papers.
**KR**: 복원 문제는 부적정(제1종 프레드홀름 적분)이다. Roelof (2000)는 피치각 이방성과 단열불변량으로 해를 제약하며, Perez (2000)는 최대 엔트로피 기법을 채용한다. 두 방법 모두 동반 논문에서 상술된다.

### 4.2 EUV resonance scattering brightness / EUV 공명 산란 광도

For an optically thin plasmaspheric He$^+$ cloud illuminated by solar 30.4 nm:

$$
4\pi I = g \cdot N_{\text{He}^+}, \qquad
g = \frac{\pi e^2}{m_e c}\, f_{30.4}\, \pi F_\odot(30.4\,\text{nm})
$$

| Symbol | Meaning |
|---|---|
| $4\pi I$ | column-integrated brightness (Rayleigh; 1 R = $10^6/4\pi$ photons cm$^{-2}$ s$^{-1}$ sr$^{-1}$) |
| $g$ | $g$-factor (photons s$^{-1}$ ion$^{-1}$); $\sim 1.5\times10^{-6}$ at solar minimum, $\sim 7\times10^{-6}$ at maximum |
| $N_{\text{He}^+}$ | column density of He$^+$ along the line of sight (cm$^{-2}$) |
| $f_{30.4}$ | oscillator strength of the 30.4 nm He$^+$ transition (dimensionless) |
| $F_\odot(30.4)$ | solar EUV photon flux at 30.4 nm (photons cm$^{-2}$ s$^{-1}$ Å$^{-1}$) |

**EN**: A typical bright plasmaspheric column near $L = 3$ has $N_{\text{He}^+} \sim 10^{12}$ cm$^{-2}$ giving $4\pi I \sim 1$ Rayleigh — easily detectable by the EUV imager.
**KR**: $L = 3$ 근처의 전형적 플라즈마권 밝은 컬럼은 $N_{\text{He}^+} \sim 10^{12}$ cm$^{-2}$이며 $4\pi I \sim 1$ 레일리를 산출 — EUV 영상기로 충분히 검출 가능하다.

### 4.3 RPI radio sounding equation / RPI 라디오 사운딩 방정식

The plasma cutoff frequency:

$$
f_p [\text{Hz}] \;\approx\; 8980\,\sqrt{n_e[\text{cm}^{-3}]} \;=\; 8.98 \times 10^3\,\sqrt{n_e}
$$

Range to reflecting layer from echo time-of-flight $\tau$ (in vacuum approximation):

$$
R = \frac{c\,\tau}{2}
$$

**EN**: RPI sweeps 3 kHz – 3 MHz; the echo at frequency $f$ arrives from the layer where $f_p(\mathbf r) = f$. Plotting echo range vs. frequency yields the **plasmagram** (Figure 2). Refraction in plasma density gradients introduces corrections beyond the vacuum formula but is well-handled by ray-tracing.
**KR**: RPI는 3 kHz – 3 MHz 범위를 스윕하며, 주파수 $f$의 에코는 $f_p(\mathbf r) = f$인 층에서 도착한다. 에코 거리 vs 주파수 플롯이 **플라즈마그램**(그림 2)을 만든다. 플라즈마 밀도 기울기에서의 굴절은 진공 공식 너머의 보정을 도입하지만, 광선 추적으로 잘 다룰 수 있다.

### 4.4 Orbital geometry / 궤도 기하

Highly elliptical orbit parameters:

$$
r_a = R_E + 7 R_E = 8 R_E,\quad r_p = R_E + 1000\,\text{km} \approx 1.157 R_E
$$
$$
a = \frac{r_a + r_p}{2} \approx 4.58 R_E,\quad e = \frac{r_a - r_p}{r_a + r_p} \approx 0.748
$$
$$
T = 2\pi \sqrt{\frac{a^3}{\mu_E}} \approx 13.3\ \text{hr},\quad \mu_E = G M_E
$$

**EN**: Plug in $\mu_E = 3.986\times 10^{14}\,\text{m}^3 \text{s}^{-2}$ and $a = 4.58\,R_E = 2.92\times 10^7$ m to verify $T \approx 49,500\,\text{s} \approx 13.7$ hr — consistent with Burch's "downlink once per orbit (~13 hours)".
**KR**: $\mu_E = 3.986\times 10^{14}\,\text{m}^3 \text{s}^{-2}$와 $a = 4.58\,R_E = 2.92\times 10^7$ m를 대입하면 $T \approx 49,500\,\text{s} \approx 13.7$시간 — Burch가 말한 "궤도당 1회 다운링크 (~13시간)"와 일치.

### 4.5 Spin-period sampling / 회전주기 샘플링

For a 2-minute spin period, the angular rate is $\omega = 2\pi/120\,\text{s} = 0.0524\,\text{rad/s} = 3°\,\text{s}^{-1}$. A 90° FOV camera builds a half-sky image once per spin. The Nyquist condition for resolving $\Delta\theta$ angular features requires the integration time per pixel to satisfy $\tau_{\text{pixel}} \le \Delta\theta / \omega$ — for $4°$ pixels and $3°$ s$^{-1}$, $\tau_{\text{pixel}} \le 1.33$ s. NAI counting statistics and scientific cadence considerations together set the 2-minute aggregate image.

**EN**: This is why "2 min image time" appears throughout Table I — it is not arbitrary but emerges from the spin-imaging geometry combined with substorm-development science requirements.
**KR**: 표 I 전반에 "2분 영상 시간"이 등장하는 이유 — 임의가 아니라 회전영상 기하와 서브스톰 전개 과학 요구가 결합되어 자연 발생한 값이다.

### 4.6 Worked Example: ENA flux from a 1.7 keV proton ring current / 작업 예제: 1.7 keV 양성자 환전류 ENA 플럭스

**EN**: Consider Figure 5's reference scenario — a 1.7 keV H$^+$ ring current with peak flux $j_{\text{ion}} = 10^{6}$ (cm$^2$ s sr keV)$^{-1}$ at $L = 4$. We estimate the ENA flux IMAGE-MENA would observe along a line of sight tangent to $L = 4$.

Geocoronal H density at $L = 4$ ($r = 4 R_E$): using the optically thin Chamberlain model $n_H(r) = n_0 (R_E/r)^3$ with $n_0 \approx 10^4$ cm$^{-3}$ at the exobase, $n_H(4 R_E) \approx 10^4/64 \approx 156$ cm$^{-3}$.

Charge-exchange cross-section at 1.7 keV: $\sigma_{cx} \approx 1.4 \times 10^{-15}$ cm$^2$ (interpolated from Lindsay & Stebbings tables).

Path length through the ring current (from a tangent geometry with $L = 4$ and ring current half-width $\sim 1 R_E$): $\ell \approx 2\sqrt{(2L) \cdot 1\,R_E} \cdot R_E \approx 2 \cdot 2.83\,R_E \approx 5.66\,R_E \approx 3.6 \times 10^9$ cm.

Plug in:
$$
j_{\text{ENA}} \approx n_H \cdot \sigma_{cx} \cdot j_{\text{ion}} \cdot \ell
\approx 156 \cdot 1.4{\times}10^{-15} \cdot 10^{6} \cdot 3.6{\times}10^{9}
\approx 7.9 \times 10^{2}\ \text{(cm}^2\text{ s sr keV)}^{-1}
$$

For a 120 s exposure with MENA's effective area $A_{\text{eff}} \approx 0.1$ cm$^2$, pixel solid angle $\Omega \approx (4° \times 4°) = 4.9 \times 10^{-3}$ sr, and energy bandpass $\Delta E \approx 0.34$ keV (ΔE/E ≈ 0.2):

$$
N_{\text{counts}} = j_{\text{ENA}} \cdot A_{\text{eff}} \cdot \Omega \cdot \Delta E \cdot \tau
\approx 790 \cdot 0.1 \cdot 4.9{\times}10^{-3} \cdot 0.34 \cdot 120
\approx 16\ \text{counts/pixel}
$$

This matches the order of magnitude of the simulated MENA image counts in Figure 5 (legend reaches ~ a few tens). The exercise demonstrates that the mission design provides usable SNR per 2-minute image even at $L = 4$ ring current intensities.

**KR**: 그림 5의 기준 시나리오 — $L = 4$에서 피크 플럭스 $j_{\text{ion}} = 10^{6}$ (cm$^2$ s sr keV)$^{-1}$의 1.7 keV H$^+$ 환전류 — 를 고려하자. $L = 4$ 접선 시선에서 IMAGE-MENA가 관측할 ENA 플럭스를 추정한다.

$L = 4$ ($r = 4 R_E$)에서의 지구코로나 H 밀도: 광학적 박형 Chamberlain 모형 $n_H(r) = n_0 (R_E/r)^3$, 외기권저면 $n_0 \approx 10^4$ cm$^{-3}$ 사용 → $n_H(4 R_E) \approx 156$ cm$^{-3}$.

1.7 keV에서 전하교환 단면적: $\sigma_{cx} \approx 1.4 \times 10^{-15}$ cm$^2$ (Lindsay & Stebbings 표에서 보간).

환전류 통과 경로 ($L = 4$ 접선 기하, 환전류 반폭 ~1 $R_E$): $\ell \approx 3.6 \times 10^9$ cm.

대입하면 $j_{\text{ENA}} \approx 7.9 \times 10^{2}$ (cm$^2$ s sr keV)$^{-1}$이다.

MENA 유효면적 $A_{\text{eff}} \approx 0.1$ cm$^2$, 픽셀 입체각 $\Omega \approx 4.9 \times 10^{-3}$ sr, 에너지 통과대역 $\Delta E \approx 0.34$ keV, 노출 120 s에서:

$$
N_{\text{counts}} \approx 16\ \text{counts/pixel}
$$

이는 그림 5의 모의 MENA 영상 계수(범례가 수십 정도) 자릿수와 일치한다. 본 연습은 임무 설계가 $L = 4$ 환전류 강도에서도 2분 영상당 사용 가능한 SNR을 제공함을 보여준다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1958 ─ Van Allen belts (Explorer 1) — radiation environment discovered
        밴 앨런대 (Explorer 1) — 방사선 환경 발견
1959 ─ Bernstein: first ENA charge-exchange detection
        Bernstein: 최초의 ENA 전하교환 검출
1968 ─ ATS-5: first synchronous-orbit plasma measurements
        ATS-5: 최초의 정지궤도 플라즈마 측정
1970 ─ Chappell: plasmaspheric bulge morphology (J. Geophys. Res. 75)
        Chappell: 플라즈마권 융기 형태
1981 ─ DE-1 SAI: first global UV auroral images from space
        DE-1 SAI: 우주에서의 최초 전역 UV 오로라 영상
1985 ─ Moore vs Shelley controversy — localised vs diffuse ionospheric outflow
        Moore 대 Shelley 논쟁 — 국지적 대 분산형 전리권 유출
1989 ─ Roelof: first ring-current ENA images from ISEE-1
        Roelof: ISEE-1로부터 최초의 환전류 ENA 영상
1992 ─ Lockwood & Smith: pulsed reconnection model
        Lockwood & Smith: 펄스형 재결합 모형
1993 ─ Onsager: quasi-steady reconnection cusp ion model
        Onsager: 준정상 재결합 커스프 이온 모형
1995 ─ Polar launch — high-resolution but still single-point in situ
        Polar 발사 — 고해상도이지만 여전히 단일점 *in situ*
1996 ─ IMAGE selected as first MIDEX mission
        IMAGE가 최초 MIDEX 임무로 선정
1998 ─ Carlson et al.: SuperDARN/FAST cusp/auroral microphysics
        Carlson 등: SuperDARN/FAST 커스프·오로라 미세물리
1999 ─ Cluster II launch (4-spacecraft formation)
        Cluster II 발사 (4기 편대비행)
2000 ─ ★ This paper — IMAGE Mission Overview
        ★ 본 논문 — IMAGE 임무 개요
2000 ─ IMAGE launch (25 March 2000, Vandenberg, Delta II)
        IMAGE 발사 (2000년 3월 25일, 반덴버그, Delta II)
2001 ─ First IMAGE plasmaspheric plume images (Sandel et al.)
        최초의 IMAGE 플라즈마권 플룸 영상 (Sandel 등)
2003 ─ "Halloween storms" — IMAGE's most studied event
        "Halloween 폭풍" — IMAGE 가장 많이 연구된 사건
2005 ─ IMAGE communications loss (December 2005)
        IMAGE 통신 두절 (2005년 12월)
2008 ─ TWINS-A,B launched (ENA imaging legacy of IMAGE)
        TWINS-A,B 발사 (IMAGE의 ENA 영상 유산)
2018 ─ IMAGE signal recovered by amateur radio astronomer
        아마추어 무선천문가가 IMAGE 신호 회복
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Chapman & Ferraro (1931) | First magnetopause concept | IMAGE finally images globally what Chapman-Ferraro predicted theoretically |
| Dungey (1961) | Open magnetosphere reconnection model | IMAGE's RPI+FUV+NAI cusp suite tests Dungey-cycle plasma flow globally |
| Frank et al. (1986) — DE-1 SAI | First global UV auroral imaging | Direct technological ancestor of IMAGE's FUV-WIC instrument |
| Moore et al. (1985) | Localised cusp outflow source | One side of the controversy IMAGE-LENA was designed to settle |
| Shelley et al. (1985) | Diffuse auroral oval outflow source | Other side of the same controversy — IMAGE LENA composition imaging discriminates |
| Lockwood & Smith (1992) | Pulsed reconnection model | IMAGE RPI plasmagram + FUV cusp footprint cadence directly tests this |
| Onsager et al. (1993) | Quasi-steady reconnection model | Counter-hypothesis to Lockwood-Smith, tested by same IMAGE observations |
| Roelof (2000), Perez (2000) | NAI image deconvolution | Companion papers in same SSR-91 issue providing the inversion math |
| Mende et al. (2000) | FUV instrument system design | Detailed instrument paper for what Burch overviews in §3 |
| Reinisch et al. (2000) | RPI investigation paper | Detailed RPI paper; Burch overview in §2.1 and Figure 2 |
| Sandel et al. (2000) | EUV imager design | Companion paper; Sandel later produced first plasmaspheric plume images |
| Goldstein et al. (2003, 2005) | IMAGE plasmaspheric plume science | Major scientific results enabled by Burch's mission concept |
| Mitchell et al. (2003) | IMAGE/HENA storm-time ring current | Direct realisation of Figure 5's ring current imaging concept |
| McComas et al. (2009) — TWINS | Successor ENA imaging mission | Direct technological descendant; IMAGE legitimised the paradigm |
| Carruthers Geocorona Observatory (2025+) | Future geocorona imager | Builds on IMAGE GEO instrument heritage |

---

## 7. References / 참고문헌

- **Burch, J. L.** (2000). "IMAGE Mission Overview." *Space Science Reviews* **91**, 1–14. DOI: 10.1023/A:1005245323115. *(this paper)*
- Burley, R. J. et al. (2000). "The IMAGE Science and Mission Operations Center." *Space Sci. Rev.* **91**, 483–496.
- Chappell, C. R. et al. (1970). "The Morphology of the Bulge Region of the Plasmasphere." *J. Geophys. Res.* **75**, 3848–3861.
- Chappell, C. R. (1974). "Detached Plasma Regions in the Magnetosphere." *J. Geophys. Res.* **79**, 1861–1870.
- Fuselier, S. A. et al. (2000). "Overview of the IMAGE Science Objectives and Mission Phases." *Space Sci. Rev.* **91**, 51–66.
- Gibson, W. C. et al. (2000). "The IMAGE Observatory." *Space Sci. Rev.* **91**, 15–50.
- Gurgiolo, C. (2000). "The IMAGE High-Resolution Data Set." *Space Sci. Rev.* **91**, 461–481.
- Lockwood, M. & Smith, M. F. (1992). "The Variation of Reconnection Rate at the Dayside Magnetopause and Cusp Ion Precipitation." *J. Geophys. Res.* **97**, 14,841–14,848.
- Mende, S. B. et al. (2000). "Far Ultraviolet Imaging From the IMAGE Spacecraft: 1. System Design." *Space Sci. Rev.* **91**, 243–270.
- Mitchell, D. G. et al. (1999). "The High Energy Neutral Atom (HENA) Imager for the IMAGE Mission." *Space Sci. Rev.* **91**, 67–112.
- Moore, T. E. et al. (2000). "The Low Energy Neutral Atom Imager for IMAGE." *Space Sci. Rev.* **91**, 155–195.
- Moore, T. E. et al. (1985). "Superthermal Ion Signatures of Auroral Acceleration Processes." *J. Geophys. Res.* **90**, 1611–1618.
- Onsager, T. G. et al. (1993). "Model of Magnetosheath Plasma in the Magnetosphere: Cusp and Mantle Particles at Low Altitudes." *Geophys. Res. Lett.* **20**, 479–482.
- Perez, J. D. et al. (2000). "Deconvolution of Energetic Neutral Atom Images of the Earth's Magnetosphere." *Space Sci. Rev.* **91**, 421–436.
- Pollock, C. J. et al. (2000). "The Medium Energy Neutral Atom (MENA) Imager for the IMAGE Mission." *Space Sci. Rev.* **91**, 113–154.
- Rasmussen, C. E. et al. (1993). "A Two-dimensional Model of the Plasmasphere: Refilling Time Constants." *Planet. Space Sci.* **41**, 35–44.
- Reinisch, B. W. et al. (2000). "The Radio Plasma Imager Investigation on the IMAGE Spacecraft." *Space Sci. Rev.* **91**, 319–359.
- Roelof, E. C. & Skinner, A. J. (2000). "Extraction of Ion Distributions from Magnetospheric ENA and EUV Images." *Space Sci. Rev.* **91**, 437–459.
- Sandel, B. R. et al. (2000). "The Extreme Ultraviolet Imager Investigation for the IMAGE Mission." *Space Sci. Rev.* **91**, 197–242.
- Shelley, E. G. et al. (1985). "Circulation of Energetic Ions of Terrestrial Origin in the Magnetosphere." *Adv. Space Res.* **5**, 401–410.

### Supplementary references for cross-checks / 교차검증 보충 참고문헌

- Lindsay, B. G. & Stebbings, R. F. (2005). "Charge transfer cross sections for energetic neutral atom data analysis." *J. Geophys. Res.* **110**, A12213. — provides the σ_cx(E) tables used in §4.6.
- Chamberlain, J. W. (1963). "Planetary coronae and atmospheric evaporation." *Planet. Space Sci.* **11**, 901–960. — provides the geocoronal H density model.
- Goldstein, J. & Sandel, B. R. (2005). "The global pattern of evolution of plasmaspheric drainage plumes." *AGU Monograph* **159**, 1–22. — major IMAGE EUV scientific result.
- Pollock, C. J. et al. (2003). "First medium energy neutral atom (MENA) images of Earth's magnetosphere during substorm and storm-time." *Geophys. Res. Lett.* **28**, 1147–1150. — first scientific publication realising Burch's Figure 5 concept.
