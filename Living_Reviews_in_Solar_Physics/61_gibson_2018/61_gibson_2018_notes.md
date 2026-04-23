---
title: "Solar Prominences: Theory and Models — Fleshing out the magnetic skeleton"
authors: ["Sarah E. Gibson"]
year: 2018
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-018-0016-2"
topic: Living_Reviews_in_Solar_Physics
tags: [prominence, filament, flux-rope, sheared-arcade, MHD, magnetic-dip, CME, thermal-nonequilibrium, coronal-cavity, chirality]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 61. Solar Prominences: Theory and Models — Fleshing out the magnetic skeleton / 태양 홍염: 이론과 모델 — 자기 골격에 살을 붙이다

---

## 1. Core Contribution / 핵심 기여

**English**: Gibson's 2018 *Living Reviews* chapter synthesizes the theoretical and modeling literature on solar prominences — cool (~7,500 K), dense chromospheric plasma suspended in the hot (~1 MK) solar corona by magnetic forces. The unifying metaphor is that of a "magnetic skeleton": the slowly evolving magnetic structure (the bones — dips in sheared arcades, helical turns of flux ropes, or fully detached spheromaks) supports the prominence mass, while thermodynamic processes (radiative cooling, heating localized at footpoints, thermal nonequilibrium) and plasma dynamics (flows, Rayleigh–Taylor instabilities, draining, and eventual eruption) add the "flesh and blood". Gibson carefully distinguishes the two historical paradigms — Kippenhahn–Schlüter (KS) dipped-arcade models and Kuperus–Raadu (KR) flux-rope models — and shows that at close range they look similar (both have local current sheets whose Lorentz force balances gravity), but at the global-topology level the flux rope is irreducibly non-potential, possesses wrapping field lines about a central axis, and hosts free magnetic energy available for a coronal mass ejection. The review surveys observational diagnostics (chirality of filament barbs, sigmoids, coronal cavities, bullseye flows, sigmoidal separatrices) that can distinguish topologies, and closes with 3D MHD simulations (Xia & Keppens 2016; Fan 2017) that now self-consistently treat magnetism plus conduction, radiation, and coronal heating to grow a prominence from a coronal atmosphere.

**한국어**: Gibson의 2018년 *Living Reviews* 챕터는 태양 홍염 — 뜨거운 코로나(~1 MK) 속에 자기력으로 매달린 차갑고(~7,500 K) 조밀한 채층 플라즈마 — 에 대한 이론 및 모델링 문헌을 종합한다. 핵심 은유는 "자기 골격(magnetic skeleton)"으로, 느리게 진화하는 자기 구조(뼈대 — sheared arcade의 dip, flux rope의 나선 감김, 혹은 완전히 분리된 spheromak)가 홍염 질량을 받치고, 열역학적 과정(복사 냉각, 풋포인트 국소 가열, 열 비평형)과 플라즈마 동역학(유동, Rayleigh–Taylor 불안정성, 배수, 궁극적 폭발)이 "살과 피"를 더한다. Gibson은 두 역사적 패러다임 — Kippenhahn–Schlüter(KS) dipped arcade와 Kuperus–Raadu(KR) flux rope — 를 주의 깊게 구분한다. 근접 시점에서는 둘 다 중력을 상쇄하는 Lorentz 힘을 가진 국소 전류 시트를 갖고 비슷해 보이지만, 전역 위상 수준에서 flux rope는 환원 불가능하게 비포텐셜이며, 중심축을 감는 자기장선을 가지고, CME에 쓰일 수 있는 자유 자기 에너지를 저장한다. 리뷰는 chirality(필라멘트 barb 손대칭성), sigmoid, 코로나 cavity, bullseye 유동, sigmoidal separatrix 등 위상 구분 관측 진단을 개관하고, 자기장과 전도·복사·코로나 가열을 자체 일관적으로 처리하여 코로나 대기에서 홍염을 성장시키는 3D MHD 시뮬레이션(Xia & Keppens 2016; Fan 2017)으로 마무리한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction — Why prominences are surprising / 왜 홍염이 놀라운가 (pp. 1–3)

**English**: Prominences are observed in true-color eclipse images (Fig. 1) as "discordantly clumpy and bright, indeed, pink" — a striking contrast to the pearly-white streamers. Their central mysteries are three: (i) how can cool, dense plasma survive in a hot, tenuous corona? (ii) where does it come from? (iii) how does it disappear? The answers all involve the magnetic field, which must provide both gravitational support and thermal insulation. Gibson frames the review as a focused treatment of the non-erupting, magnetohydrostatic quasi-equilibrium — the "magnetic skeleton" — in §§2–3, then relaxes both the assumption of stasis and of force-freeness in §4.

**한국어**: 홍염은 개기일식 진짜색상 영상(Fig. 1)에서 "뭉친 듯 밝고, 실제로 분홍색"으로 보이며, 진주빛 백색 streamer와 강한 대조를 이룬다. 중심 수수께끼는 셋이다: (i) 차갑고 조밀한 플라즈마가 뜨겁고 희박한 코로나 속에서 어떻게 생존하는가? (ii) 어디에서 오는가? (iii) 어떻게 사라지는가? 답은 모두 자기장에 있으며, 자기장은 중력 지지와 열 차단을 동시에 제공해야 한다. Gibson은 §§2–3에서 비폭발·자기정수역학적 준평형 — "자기 골격" — 에 집중하고, §4에서 정적/force-free 가정을 모두 완화한다.

### Part II: The prominence magnetic skeleton / 홍염의 자기 골격 (Sect. 2, pp. 3–10)

#### 2.1 Prominence bones: magnetic dips / 자기 dip

**English**: Prominence spines are vertical sheets 20°–35° inclined above a PIL. Early Zeeman observations (Rust 1967; Leroy 1977, 1978) found predominantly horizontal fields with magnitudes large enough that magnetic pressure dominates thermal pressure (low β). Since a low-β equilibrium is force-free, a downward magnetic pressure gradient requires an upward magnetic tension force. Together with longevity and stability, this establishes a fundamental building block: **the cool prominence mass is supported within dipped magnetic fields** (though Sect. 4.2.1 revisits this).

**한국어**: 홍염 spine은 PIL 위 20°–35° 기울어진 수직 시트이다. 초기 Zeeman 관측(Rust 1967; Leroy 1977, 1978)은 주로 수평 방향의, 자기 압력이 열 압력을 압도하는(low β) 크기의 자기장을 발견했다. Low-β 평형이 force-free이므로, 아래로 향하는 자기 압력 구배는 위로 향하는 자기 장력을 요구한다. 장수명·안정성과 결합하여, 근본 구성 블록이 성립한다: **차가운 홍염 질량은 dip된 자기장 안에 지지된다**(§4.2.1에서 재검토).

#### 2.2 Early prominence models (KS & KR) / 초기 홍염 모델

**English**: **Kippenhahn–Schlüter (1957, KS)** is a 2.5D infinitely-thin vertical sheet of dense material on edge above a solar PIL, supported within an arcade of dipped horizontal field lines (Fig. 2a). A quadrupolar potential arcade can contain dips (Fig. 2b); as mass accrues, a current sheet forms at the dip center and the upward Lorentz force balances gravity:
$$\rho g = B_x(x=0)\,[B_z]/\mu_0$$
where $[B_z]$ is the jump of $B_z$ across the sheet. In the limit of vanishing mass the sheet vanishes and the configuration reverts to its potential dipped arcade.

**Kuperus–Raadu (1974, KR)** treats the prominence as a current filament at height $h$ above the photosphere surrounded by closed field lines (Fig. 3). Induced (image) currents at $-h$ repel the filament upward, while a background horizontal field adds a downward confining force. In the limit of vanishing mass this becomes a **force-free flux rope** with field-aligned currents. A horizontal background field component parallel to the rope is needed to avoid magnetic pinch destabilization.

Critical insight: zooming into the KS and KR configurations near the prominence yields similar local dips; the **distinction lies in global topology** — KR has a rope detached from the photosphere, KS's dips sit in an arcade anchored in bipolar photospheric flux.

**한국어**: **KS(1957)**: 2.5D 무한히 얇은 수직 질량 시트가 수평 dipped arcade 위에 서 있는 모델(Fig. 2a). 쿼드러폴 포텐셜 arcade는 dip을 포함할 수 있고(Fig. 2b), 질량이 쌓이면 dip 중심에 전류 시트가 형성되어 위 방향 Lorentz 힘이 중력을 상쇄:
$$\rho g = B_x(x=0)\,[B_z]/\mu_0$$
질량 → 0 극한에서 시트가 사라지고 포텐셜 dipped arcade로 복귀.

**KR(1974)**: 홍염을 광구 위 높이 $h$의 전류 필라멘트로 본다. 이미지 전류($-h$)가 위로 밀고, 배경 수평장이 아래로 누른다(Fig. 3). 질량 → 0 극한에서 **force-free flux rope**가 된다. Rope 축에 평행한 배경장 성분이 자기 pinch 불안정성 방지에 필요.

핵심 통찰: 홍염 근방만 보면 KS와 KR의 국소 dip이 비슷하다. **차이는 전역 위상에 있다** — KR은 광구에서 분리된 rope, KS의 dip은 양극 광구 자속에 닻을 내린 arcade에 놓인다.

#### 2.3 Inverse vs. normal magnetic fields / Inverse vs. normal

**English**: If the field in the dip points from (+) to (–) photospheric polarity, it is *normal* (same as a potential arcade). If it points from (–) to (+), it is *inverse*. Observations (Leroy et al. 1984) found 75–90% of prominences have inverse configuration. A common but misleading shortcut equates KS=normal, KR=inverse. Malherbe & Priest (1983) showed both topologies admit both polarity senses (Fig. 4): inverse arcades and normal flux ropes are possible. The more meaningful 3D dichotomy is **sheared arcade vs. flux rope**: an arcade (dipped and sheared) is topologically equivalent to a potential arcade; a flux rope has field lines wrapping a central axis, topologically unobtainable from the potential field.

**한국어**: Dip의 장이 (+)→(–)이면 *normal*, (–)→(+)이면 *inverse*. 관측(Leroy et al. 1984)에서 75–90%가 inverse. KS=normal, KR=inverse라는 지름길은 오해의 소지. Malherbe & Priest(1983)는 양 위상 모두 양 극성을 허용함을 보였다(Fig. 4). 더 의미 있는 3D 이분법은 **sheared arcade vs. flux rope**: arcade(dipped+sheared)는 포텐셜 arcade와 위상 동등; flux rope는 중심축을 감는 장선을 가져 포텐셜에서 얻을 수 없음.

#### 2.4 Sheared-arcade vs. flux-rope models / Sheared arcade vs. flux rope

**Sheared-arcade models (§2.4.1)**:
**English**: 2.5D bipolar dipped arcades can be potential (Démoulin & Forbes 1992) or linear force-free (Amari & Aly 1990; Démoulin et al. 1992). But **in 2.5D, imposing shear on an arcade cannot create top-of-arcade dips** — lines expand and flatten (Klimchuk 1990). Antiochos et al. (1994) showed that **in 3D, shearing a bipolar arcade can create a force-free dipped region in the middle** of the sheared portion (the overlying arcade compresses the sheared center, Fig. 5a). Such 3D sheared-arcade dips are mostly *inverse* (Aulanier et al. 2002).

**Flux-rope models (§2.4.2)**:
**English**: A flux rope is any structure where flux wraps a central axis. This creates topologically distinct sub-systems — QSLs at field-line-linkage gradients, BPSS at tangential photospheric touches (Titov & Démoulin 1999). Formation mechanisms: photospheric twisting (Priest et al. 1989), reconnection of sheared lines (van Ballegooijen & Martens 1989), or emergence of pre-twisted flux from below (Rust & Kumar 1994). Analytic models evolved from 2.5D force-free (Ridgway et al. 1991; Forbes & Isenberg 1991) to 3D cylindrical (Titov–Démoulin 1999) to spheromak (Lites & Low 1997; Gibson & Low 1998). 3D NLFF ropes typically possess 1–2 turns about the central axis; >~2 turns risk kink instability and eruption (Rust & Kumar 1996; Kliem et al. 2004).

**Comparison (§2.4.3)**:
**English**: Because pressure scale heights are much smaller in cool prominence plasma than in coronal plasma, mass fills only the lowest portion of each dip, producing **bead-like condensations** that form a sheet. Gibson notes that "painting the dips" in a sheared arcade, a cylindrical flux rope, or even a twisted spheromak all yield **sheet-like prominence structures** (Fig. 6) — topology alone is not easy to infer from prominence shape.

**한국어**:
**Sheared arcade(§2.4.1)**: 2.5D에서 shear로는 arcade 꼭대기 dip을 못 만든다(Klimchuk 1990). 3D에서는 Antiochos et al.(1994)이 sheared 중심부에 dip이 형성됨을 보였다(Fig. 5a). 대부분 *inverse*(Aulanier et al. 2002).

**Flux rope(§2.4.2)**: 중심축을 감는 flux. QSL, BPSS 같은 위상 구조 수반. 형성: 광구 twisting, sheared line 재결합, 사전 twisted flux 방출. 해석적 모델: 2.5D → Titov–Démoulin 3D → spheromak. 축 주위 1–2 회전이 전형, >~2 회전은 kink 불안정성/폭발 위험.

**비교(§2.4.3)**: 차가운 홍염 플라즈마의 척도 높이는 코로나보다 훨씬 작아, 질량은 각 dip의 가장 낮은 부분만 채워 **구슬 같은 응축**이 시트 형태로 배열. Sheared arcade, 원통형 flux rope, 심지어 twisted spheromak 모두 "dip에 색칠"하면 **시트형 홍염 구조**가 나온다(Fig. 6) — 형태만으로 위상 추론은 어렵다.

### Part III: Beyond bones — the prominence in context / 뼈를 넘어: 맥락 속의 홍염 (Sect. 3, pp. 11–20)

#### 3.1 Filament channels and chirality / 필라멘트 채널과 chirality

**English**: Filament *channels* are regions where chromospheric fibrils align on either side of a PIL (Fig. 7a); arguably more fundamental than filaments themselves. Chirality (handedness) is diagnosed by fibril orientation viewed from the positive-polarity side: fibrils extending right → **dextral**, left → **sinistral**. Filament *barbs* bear right/left consistently with the channel chirality, and associate with parasitic opposite-polarity patches. Aulanier & Démoulin (1998) built a linear force-free rope with periodic boundary and parasitic polarities that produced dips organized like barbs, extending from a central sheet (Fig. 8). Overlying coronal loops show opposite-skew chirality (right-handed dextral filament → left-skewed overlying loops; Martin & McAllister 1996). Chirality rules (dextral dominant in northern hemisphere, sinistral in southern) are naturally explained by helical flux ropes (Gibson & Low 2000) where tops of winding lines skew oppositely to bottoms relative to the PIL.

**한국어**: **필라멘트 채널**은 PIL 양쪽에서 채층 fibril이 정렬된 영역(Fig. 7a). (+)극성 쪽에서 보아 fibril이 오른쪽 → **dextral**, 왼쪽 → **sinistral**. 필라멘트 **barb**는 채널 chirality와 일관되게 좌/우로 뻗고, 기생 반대 극성 patch와 연관된다. Aulanier & Démoulin(1998)은 주기적 경계와 기생 극성을 가진 선형 force-free rope에서 barb 같은 dip을 생성(Fig. 8). Overlying 코로나 loop은 반대 skew chirality(Martin & McAllister 1996). Chirality 규칙(북반구 dextral 우세, 남반구 sinistral)은 나선형 flux rope로 자연스럽게 설명(Gibson & Low 2000) — 감긴 장선의 위/아래가 PIL 대비 반대 skew를 가짐.

#### 3.2 Sigmoids and separatrices / 시그모이드와 separatrix

**English**: Soft X-ray **sigmoids** are S- or inverse-S-shaped active regions prone to eruption (Rust & Kumar 1996; Canfield et al. 1999). Two classes: (i) diffuse S-regions of sheared loops, (ii) sharp sigmoidal current sheets. Filaments often sit within sigmoids (Gibson et al. 2002; Pevtsov 2002). Overlying arcades opposite to filament skew implies — for flux ropes — that the SXR sigmoid is formed from field lines tracing the *bottom* of the helix. Various models explain sigmoid formation: magnetic flux emergence + diffusive evolution building sheared J-loops that reconnect into an erupting rope + transient sigmoid (van Ballegooijen & Martens 1989; Amari et al. 2000; Moore et al. 2001; Kusano 2005), or a pre-existing rope kinking downward (Kliem et al. 2004). QSLs (Priest & Démoulin 1995) are expected current-sheet locations; sigmoid formation near a BPSS ("bald patch") is the archetype, with interchange reconnections between J-type arcade and S-type rope lines producing the sharp sigmoidal loop (Fig. 9b). A sigmoid that survives a partial eruption (Fig. 11) is explained by rope bifurcation — the BPSS part survives because it lies below the filament (Gibson & Fan 2006b).

**한국어**: Soft X-ray **시그모이드**는 S 또는 역-S 모양의 활동 영역으로 폭발 경향(Rust & Kumar 1996). 두 유형: (i) sheared loop들의 확산된 S 영역, (ii) 또렷한 sigmoidal 전류 시트. 필라멘트가 시그모이드 내부에 자주 위치. Overlying arcade가 필라멘트와 반대 skew라는 사실은 flux rope 기준에서 SXR 시그모이드가 나선의 *아래쪽* 장선으로 구성됨을 시사. 모델: 자속 방출 + 확산 진화로 sheared J-loop 형성 → 재결합하여 폭발 rope + transient 시그모이드(van Ballegooijen & Martens 1989 등), 또는 기존 rope이 아래로 kink(Kliem et al. 2004). QSL(Priest & Démoulin 1995)이 전류 시트 형성 위치로 예측됨; BPSS("bald patch") 근처 시그모이드 형성이 표준. 부분 폭발 후 시그모이드가 살아남는 것(Fig. 11)은 rope 이분화로 설명 — BPSS 부분이 필라멘트 아래에 있어 생존.

#### 3.3 Cavities and flux surfaces / Cavity와 자속 표면

**English**: Coronal **cavities** are dark elliptical regions around prominences seen at multiple wavelengths (radio → white light → EUV → SXR; Fig. 12). They are long-lived (days–months for polar crowns), tunnel-like or croissant-shaped in 3D (Gibson et al. 2010; Karna et al. 2015), with density depletion of ~30–50% relative to surrounding streamers. Cavities may contain a hot central core (Fig. 13a) with "horn"-like brightenings connecting prominence to cavity (Régnier et al. 2011; Schmit & Gibson 2013), and bullseye line-of-sight flow patterns (Bąk-Stęślicka et al. 2016). These properties are naturally explained by a **pre-eruption magnetic flux rope** (Low 1996, 2001; Linker et al. 2003; Fan & Gibson 2006) whose nested toroidal flux surfaces produce the elliptical cavity boundary, swirling flows, and bullseye pattern (Fig. 13e, Fig. 14). **Teardrop-shaped cavities** are especially predictive of eruption (Forland et al. 2013; Karna et al. 2015a; Fig. 15) — the teardrop morphology arises when an X-line forms beneath a flux rope, effectively a "lit fuse" for tether-cutting → eruption. Coronal polarimetry (CoMP; Tomczyk et al. 2008) shows **lagomorphic ("rabbit-head")** linear polarization consistent with flux rope topology (Bąk-Stęślicka et al. 2013); this rules out spheromak for most cavities.

**한국어**: 코로나 **cavity**는 홍염 주변의 다파장 암흑 타원 영역(Fig. 12). 장수명(극 crown은 수일–수개월), 3D에서 터널 또는 크루아상 모양, 주변 streamer 대비 ~30–50% 밀도 결핍. 중심 코어(Fig. 13a), 홍염–cavity 연결 horn, bullseye 시선속도 유동 패턴을 포함. 이 특성은 **사전 폭발 flux rope**(Low 1996 등)로 자연스럽게 설명 — nested 토로이달 flux surface가 타원 경계, 선회 유동, bullseye를 만듦. **눈물방울 모양** cavity는 폭발 예측자(Fig. 15) — rope 아래 X-line이 tether-cutting 점화기 역할. CoMP 편광으로 **lagomorphic("토끼 머리")** 선편광이 flux rope 위상과 부합(Bąk-Stęślicka et al. 2013); 대부분 cavity에서 spheromak 배제.

### Part IV: Adding flesh and blood — dynamics and thermodynamics / 살과 피 — 동역학과 열역학 (Sect. 4, pp. 21–28)

#### 4.1 Prominence dynamics / 홍염 동역학

**English**: Even quiescent prominences are intrinsically dynamic, with flows of a few–tens of km/s — sub-sonic and sub-Alfvénic (coronal $v_A \sim$ 100s km/s). Cool material streams along the spine, moves helically in "solar tornadoes", and rises in plume-like hot bubbles from below (Berger et al. 2011). Vertical flows occur in vertical "hedgerow" structures despite horizontal fields; Hα Doppler shifts (Schmieder et al. 2010) show the velocity is not purely vertical. Mechanisms: (a) local KS sheet out of equilibrium → constant-speed descent (Low & Petrie 2005); (b) gravity-driven sagging fields reconnecting (Lerche & Low 1980); (c) tangential-discontinuity reconnections passing mass downward (Chae 2010); (d) tangled horizontal fields supporting mass against free-fall (van Ballegooijen & Cranmer 2010, Fig. 17d). **Plume upflows** are interpreted as Magnetic Rayleigh–Taylor (MRT) instability between a coronal bubble and overlying dense prominence (Ryutova et al. 2010; Hillier et al. 2011, 2012a,b; Fig. 18); mass accelerates downward in dense blobs punctuated by shocks. Alternatively, bubbles may be arcade-type gaps below dipped flux-rope fields perturbed by parasitic bipoles (Dudík et al. 2012).

**한국어**: 정상(quiescent) 홍염도 본질적으로 동적 — 수 km/s~수십 km/s 유동, 아음속·아알벤 속도. Spine을 따라 유동, "태양 tornado"에서 나선 유동, "bubble" 상승 유동(Berger et al. 2011). 수평장임에도 수직 hedgerow에서 수직 유동. 기구: (a) 국소 KS 시트의 비평형 → 등속 하강, (b) 중력으로 처진 장선 재결합, (c) 접선 불연속 재결합, (d) 얽힌 수평장 지지. **Plume 상승 유동**은 MRT 불안정성 — 코로나 bubble vs. overlying 홍염(Fig. 18). Dudík et al.(2012) 대안: bubble은 기생 bipole로 교란된 linear force-free rope 하부 arcade 간극.

**Dynamic skeleton**: field lines themselves may move, be broken by reconnection, or deform under gravity. The frozen-in condition may break down via diffusion of neutral atoms (Gilbert et al. 2002; Khomenko et al. 2014) or spontaneously (Low et al. 2012). Flows may also directly evidence dynamic rearrangement of the skeleton — Okamoto et al. (2016) observed a small part of prominence spine rotating, interpreted as magnetic reconnection between flux systems (Fig. 16).

#### 4.2 Prominence thermodynamics / 홍염 열역학

**4.2.1 Beyond equilibrium — Thermal Nonequilibrium (TNE)**:
**English**: TNE (Antiochos & Klimchuk 1991; Antiochos et al. 1999, 2000) has footpoint-localized heating that causes catastrophic cooling and condensation along coronal loops — no static equilibrium is reached. Karpen et al. (2001) showed even **flat, long field lines** can host steady-state condensations; dips may not be *necessary*. Luna et al. (2012) implemented TNE along multiple field lines in a 3D sheared arcade (Fig. 19a); Schmit et al. (2013) did similar work on a 3D flux rope, giving clues to prominence-cavity connection. Gunár & Mackay (2015) developed a "Whole-Prominence Fine Structure" (WPFS) approach that imposes a semi-empirical T profile (with prominence-corona transition region) and solves hydrostatic balance along field lines in a 3D NLFF model — synthetic Hα, submillimeter, and He II 304 emission can be compared to observations (Fig. 19b).

**4.2.2 Beyond force-free**:
**English**: Plasma β is not homogeneous: Anzer & Heinzel (2007) showed β could be order unity for massive prominences with B ~ a few G. Hillier & van Ballegooijen (2013) mass-loaded a force-free 2.5D rope and relaxed to a new MHD equilibrium: increasing mass or β deformed field lines, **pulling the rope axis down**. Prominence-like sheet morphology occurs when gravity is balanced by added magnetic tension from deformation; β ≈ 0.1 (moderate), but still low. Terradas et al. (2015) found in 3D sheared-arcade that **low-β plasma supports detached prominences**, while **high-β** extends mass down to the photosphere like hedgerows. Terradas et al. (2016) found in a 3D force-free flux rope (TDm) that mass pushes rope axis down, dips deepen, rope twist provides support against gravity in low-β; **MRT instability was suppressed** by rope structure, and the prominence was organized horizontally by Kelvin–Helmholtz shear rather than vertically. Tentative conclusion: horizontal-structure prominences may match flux rope, vertical-structure ones may match sheared arcade.

**4.3 The Full Monty — 3D MHD prominence formation** (pp. 27–28):
**English**: **Xia & Keppens (2016)** built the first full 3D MHD prominence-in-flux-rope: start with isothermal rope in MHD equilibrium, add hydrostatic atmosphere with chromosphere, solve MHD with energy equation, apply footpoint heating → condense prominence via TNE (Fig. 20a). Result: fragmented, highly dynamic prominence with blobs and threads forming continuously in rope dips, dragging field lines into vertical hedgerows. Plasma replenished by chromospheric evaporation. Surrounded by coronal cavity. **Fan (2017)** imposed time-varying electric field at lower boundary of a quasi-steady coronal streamer (spherical coordinates), emerging a confined flux rope, then forming a cold dense prominence via runaway radiative instability (Fig. 20c). Both simulations: prominences dragged field downward, **significant departure from force-freeness** in a still-low-β plasma, mass in magnetic dips. Both produced morphologically similar, observationally realistic prominence-cavity systems despite different setups.

**한국어 — §4 요약**: 정지(static) 가정과 force-free 가정을 모두 약화해야 한다. TNE는 dip 유무 상관없이 풋포인트 가열로 응축을 생성(Karpen et al. 2001). 질량이 실리면 β가 moderate가 될 수 있고, 자기장이 변형되어 rope 축이 내려앉는다. 3D 완전 MHD 시뮬레이션(Xia & Keppens 2016; Fan 2017)은 자체 일관적으로 홍염-cavity 시스템을 재현하며, 중력이 장을 누른 low-β, 하지만 비-force-free 상태가 핵심이다.

### Part IVa: Figure walkthroughs / 주요 그림 해설

**Figure 2 (KS arcade variants)**: Panel (a) summarizes the original KS concept — a vertical mass sheet (hatched) standing in an arcade of dipped horizontal field lines with current $\mathbf{J}$ pointing out of the page. Panel (b) shows a quadrupolar *potential* arcade where dips arise from the natural geometry of four polarities; this is a stable, current-free configuration. Panel (c) shows how, as prominence mass accumulates, a localized current sheet forms at the center of the dips, generating an upward Lorentz force that exactly balances gravity. Importantly, in the zero-mass limit, this current sheet vanishes and (c) reverts to (b).

**English → 한국어**: Panel (a)는 원본 KS 개념 — dipped arcade 내 수직 질량 시트(음영), 전류 $\mathbf{J}$는 페이지 밖으로. (b)는 4극 *포텐셜* arcade의 자연적 dip. (c)는 질량이 쌓이며 dip 중심에 형성된 국소 전류 시트가 위로 향하는 Lorentz 힘을 만들어 중력과 상쇄. 질량 → 0이면 전류 시트 소멸, (c)→(b).

**Figure 3 (KR flux-rope variants)**: Panel (a) shows the Kuperus–Raadu concept — a current filament at height $h$ above the photosphere with image current at $-h$; repulsion lifts the filament. Panel (b) shows the same filament embedded in a background potential field with horizontal component, which supplies the confining downward force. Panel (c) (Pneuman 1983) adds an X-line beneath the prominence and a surrounding helmet streamer — this is the canonical modern picture of a quiescent prominence with overlying cavity.

**한국어**: (a)는 KR 개념 — 광구 위 $h$의 전류 필라멘트, 이미지 전류는 $-h$; 척력이 필라멘트를 들어올림. (b)는 배경 포텐셜장 내 수평 성분이 confining 힘 제공. (c)는 홍염 아래 X-line과 주변 helmet streamer — 현대 표준 quiescent 홍염 그림.

**Figure 4 (Malherbe–Priest normal/inverse matrix)**: Four panels arrange the two topologies (arcade vs. flux rope) against the two polarity senses (normal vs. inverse). Crucially, (d) "inverse arcade" is marked with "????" — reflecting the ambiguity Gibson discusses, since an arcade dipped from a quadrupolar potential field may be called normal (same direction as potential) or inverse (opposite to potential extrapolated from bipolar PIL only).

**한국어**: 4 패널이 두 위상(arcade vs. flux rope)과 두 극성(normal vs. inverse)을 배치. (d) "inverse arcade"에 "????" 표기 — 4극 포텐셜장 dipped arcade가 normal인지 inverse인지 모호함을 반영.

**Figure 6 ("Painting the dips")**: Demonstrates that regardless of skeleton topology (sheared arcade, cylindrical flux rope, spheromak), the locus of dipped field lines — once "painted" with prominence plasma — produces a sheet-like structure mimicking observations. This is a fundamental point: **prominence shape alone cannot distinguish skeleton topology**; additional diagnostics (chirality, cavity structure, polarimetry, sigmoids) are required.

**한국어**: 골격 위상(sheared arcade, 원통 flux rope, spheromak)에 상관없이 dip된 장선을 "색칠"하면 시트형 구조가 나와 관측을 흉내냄. 핵심: **홍염 형태만으로 골격 위상 구분 불가**; chirality, cavity, 편광, 시그모이드 등 추가 진단 필요.

**Figure 10 (sigmoid modeling)**: BBSO Hα filament and Yohkoh SXR sigmoid from the same region (2001). Panel (c) shows the flux rope model (purple = BPSS field lines) overlaid with current-sheet-intersecting field lines (red/orange = J-type arcade that reconnects with S-type rope). Panel (d) paints the dipped field centers up to a prominence scale height (brown) — showing the filament naturally forms within the sigmoid architecture.

**한국어**: BBSO Hα 필라멘트와 Yohkoh SXR 시그모이드(같은 영역, 2001). (c)는 flux rope 모델(보라=BPSS)과 전류 시트 교차 장선(빨강=J-arcade). (d)는 dipped 장 중심을 홍염 척도 높이까지 색칠(갈색) — 시그모이드 구조 내에서 필라멘트가 자연스럽게 형성됨을 보여줌.

**Figure 18 (MRT in prominences)**: Hillier et al. (2012) numerical simulation — KS model perturbed by a low-density/high-temperature region at its base, RT instability produces upward plumes (b,c,d) while dense blobs accelerate downward (e,f). Panel (g) overlays field lines showing reconnection at tangential discontinuities.

**한국어**: Hillier et al.(2012) 수치 시뮬레이션 — KS 모델이 바닥의 저밀도·고온 영역에 의해 교란, RT 불안정성이 상승 plume(b,c,d)과 하강 조밀 blob(e,f)을 생성. (g)는 접선 불연속에서의 재결합 장선.

**Figure 20 (Full Monty simulations)**: Left column: Xia & Keppens (2016) 3D MHD with dense prominence plasma in flux rope dips, synthetic 171 Å emission reproducing vertical hedgerows and cavity morphology. Right column: Fan (2017) spherical-coordinate simulation, synthetic white light and 171 Å of prominence-cavity system embedded in a coronal streamer.

**한국어**: 왼쪽: Xia & Keppens(2016) 3D MHD, flux rope dip 내 조밀 홍염 플라즈마, 합성 171 Å 방출이 수직 hedgerow와 cavity 형태 재현. 오른쪽: Fan(2017) 구면좌표 시뮬레이션, 코로나 streamer 내 홍염-cavity 계의 합성 백색광·171 Å.

### Part IVb: Worked example — KS-to-KR distinction as seen through cavity polarimetry / 실례 — Cavity 편광을 통한 KS–KR 구분

**English**: Consider a quiescent prominence with a surrounding cavity observed by CoMP at Fe XIII 1074.7 nm in linear polarization. If the underlying skeleton is a sheared arcade, field lines lie in vertical-plane-like sheets above the PIL; the integrated Stokes Q/U along a line of sight traversing the cavity yields a linear polarization pattern dominated by the orientation of these sheets. If instead the skeleton is a flux rope with nearly horizontal axis along the PIL, the wrapping field produces a distinctive **lagomorphic ("rabbit-head")** pattern of linear polarization — two "ears" at the top flanks where the field turns around the axis, dipping to a notch where the axis itself has zero perpendicular-to-LOS projection. Bąk-Stęślicka et al. (2013) performed forward-modeling with an analytic rope and showed the lagomorphic signature is a direct fingerprint of flux-rope topology. Observationally, lagomorphic patterns are seen in the majority of surveyed cavities, supporting the flux-rope paradigm for *quiescent* prominences.

**Numerical estimate**: with prominence $B_{\text{prom}}\sim 10$ G embedded in a cavity of $B_{\text{cav}}\sim 5$ G and coronal ambient of $\sim 1$ G, the Stokes Q/U magnitudes scale as $\sim B^2$ with LOS integration length $\sim 200$ Mm for cavity → signal-to-noise well within CoMP's 10$^{-5}$ relative polarization sensitivity.

**한국어**: CoMP가 Fe XIII 1074.7 nm에서 quiescent 홍염 주변 cavity를 선편광 관측한다고 하자. 바닥 골격이 sheared arcade이면 PIL 위 수직 시트형 장선이 Stokes Q/U를 지배한다. Flux rope (축이 PIL에 거의 수평)이면 감싸는 장이 **lagomorphic("토끼 머리")** 선편광 패턴을 만든다 — 축 주위에서 장이 도는 상부 양쪽에 두 "귀", LOS 수직 성분이 0인 축 자체에서 홈. Bąk-Stęślicka et al.(2013)의 분석적 rope 순방향 모델링은 lagomorphic 신호가 flux rope 위상의 직접적 지문임을 보였다. 관측적으로 대부분 cavity에서 lagomorphic 패턴이 보여, *quiescent* 홍염에 대해 flux rope 패러다임을 지지.

**수치 추정**: 홍염 $B\sim 10$ G, cavity $\sim 5$ G, 배경 $\sim 1$ G, LOS 적분 $\sim 200$ Mm → CoMP 상대 편광 감도 $10^{-5}$에 충분한 S/N.

### Part V: Conclusions / 결론 (Sect. 5, pp. 28–29)

**English**: Gibson's three conclusions:
(1) Both sheared arcade and flux rope models are consistent with basic prominence observations; **spheromak is inconsistent with most cavity linear polarimetry**.
(2) Some observations strongly imply flux rope — cavities with nested toroidal structures and flows, non-erupting sigmoidal loops, filaments/sigmoids surviving partial eruption. A **spectrum** from sheared arcade → rope with BPSS → rope with X-line may map onto stability.
(3) The assumptions underlying the skeleton (dips, force-freeness) must be examined in light of full thermodynamics; **numerical simulations are essential** because they treat both simultaneously.

Personal reflection: the **elephant and three blind men** — any model fitting particular observations is suspect; future progress lies in data-driven 3D MHD-thermodynamics simulations with comprehensive observations.

**한국어**: (1) Sheared arcade와 flux rope 모두 기본 관측과 일관; **spheromak은 대부분 cavity 선편광과 부합하지 않음**. (2) 일부 관측은 flux rope을 강하게 시사 — nested 토로이달 cavity, 비폭발 sigmoidal loop, 부분 폭발 생존 필라멘트. 안정성 위상 스펙트럼: sheared arcade → BPSS 있는 rope → X-line 있는 rope. (3) Dip 필요성, force-free 가정은 완전 열역학 맥락에서 재검토 필요; **수치 시뮬레이션**이 둘을 동시에 다루는 유일한 방법. "코끼리와 세 눈먼 사람" 비유 — 특정 관측에만 맞는 모델은 의심스럽다.

---

## 3. Key Takeaways / 핵심 시사점

1. **The magnetic skeleton metaphor unifies the field / 자기 골격 은유가 분야를 통합한다** — Bones (magnetic structure: dips, ropes, separatrices, QSLs) provide long-lived support; flesh (plasma mass, flows) and blood (condensation, heating, eruption) are dynamic. This cleanly separates the two timescales and physics regimes that govern prominences. / 뼈(자기 구조: dip, rope, separatrix, QSL)가 장수명 지지를 제공하고, 살(질량, 유동)과 피(응축, 가열, 폭발)가 동역학을 담당. 홍염을 지배하는 두 시간 척도·물리 영역을 명확히 분리.

2. **Low plasma β makes force-balance magnetic / 저 플라즈마 β가 힘 균형을 자기력화한다** — Rust's 1967 measurement of B~5–10 G with p~0.1 dyne/cm² yields $\beta \approx 2\mu_0 p / B^2 \ll 1$; prominence equilibrium is therefore force-free to first order and any downward pressure gradient demands upward magnetic tension (i.e., a dip). This single observation motivated the KS and KR models. / Rust(1967)의 B~5–10 G, p~0.1 dyne/cm² 측정으로 $\beta \ll 1$이 확립. 홍염 평형은 1차적으로 force-free이며, 아래 방향 압력 구배는 위 방향 자기 장력(즉 dip)을 요구. 이 단일 관측이 KS·KR 모델의 근거.

3. **KS and KR are locally similar but globally different / KS와 KR은 국소적으론 유사, 전역적으론 다르다** — Zoom into a prominence and both show a current sheet whose Lorentz force supports gravity. Step back: KS is anchored bipolar (topologically potential-arcade), KR is a detached current filament (topologically flux rope). The **flux rope stores free magnetic energy**; the arcade does not. / 홍염 근방에서 둘 다 Lorentz 힘이 중력 상쇄하는 전류 시트를 보인다. 전체를 보면 KS는 양극성 anchored(포텐셜-arcade 위상 동등), KR은 분리된 전류 필라멘트(flux rope 위상). **Flux rope은 자유 자기 에너지 저장**, arcade는 저장하지 않는다.

4. **Chirality and cavity polarimetry favor flux ropes / Chirality와 cavity 편광은 flux rope을 지지한다** — Dextral/sinistral rules, barb orientation, and opposite-skew overlying arcades are naturally explained by helical ropes. Teardrop-shaped cavities with nested bullseye flows and lagomorphic linear polarization from CoMP rule out spheromaks for most cases and support flux-rope topology for quiescent cavities. / Dextral/sinistral 규칙, barb 방향, 반대-skew overlying arcade는 나선형 rope으로 자연스럽게 설명. 눈물방울 cavity + bullseye 유동 + CoMP lagomorphic 선편광은 대부분 cavity에서 spheromak 배제하고 flux rope 위상을 지지.

5. **TNE decouples prominence formation from dips / TNE는 홍염 형성을 dip에서 분리한다** — Thermal Nonequilibrium with footpoint-localized heating (Antiochos & Klimchuk 1991; Karpen et al. 2001) produces catastrophic cooling even along long flat field lines. "Do we need dips?" → "No." for steady-state formation. Dips are sufficient, not necessary. / 풋포인트 국소 가열에 의한 TNE는 긴 평평한 장선에서도 파국 냉각 생성. "Dip이 필요한가?" → "아니오." Dip은 충분조건이지 필요조건이 아니다.

6. **Mass loading breaks force-freeness / 질량 적재가 force-free를 깨뜨린다** — Hillier & van Ballegooijen (2013), Terradas et al. (2015, 2016), and full 3D MHD (Xia & Keppens 2016; Fan 2017) all show that accumulated mass deforms magnetic fields (especially pulling flux rope axes down) and pushes local plasma β to moderate values. Realistic prominences are neither static nor fully force-free. / 누적된 질량이 자기장을 변형(특히 rope 축을 아래로 당김)시키고 국소 β를 moderate로 밀어 올린다. 현실적 홍염은 정적이지도 완전 force-free도 아니다.

7. **MRT instability animates the plume-and-bubble cycle / MRT 불안정성이 plume-bubble 순환을 이끈다** — Heavy prominence over lighter coronal bubble → Magnetic Rayleigh–Taylor interchange → downflows in dense blobs and upflows in tenuous plumes, completing the "magnetothermal convection" cycle (Berger et al. 2011; Hillier 2018). / 가벼운 코로나 bubble 위 무거운 홍염 → MRT interchange → 조밀 blob의 하강 유동과 희박 plume의 상승 유동이 "자기열 대류" 순환을 완성.

8. **The future is data-driven 3D MHD-plus-thermodynamics / 미래는 자료 기반 3D MHD+열역학이다** — Idealized homogeneous boundaries must yield to mixed boundary conditions with inserted flux ropes (Aulanier et al. 1999; van Ballegooijen 2004) and explicit energy equation treatment. Xia & Keppens (2016) and Fan (2017) exemplify this frontier; DKIST and next-gen polarimetry will provide the constraints. / 이상화된 균질 경계는 flux rope 삽입 혼합 경계 조건(van Ballegooijen 2004)과 에너지 방정식 포함 처리에 자리를 내줘야 한다. Xia & Keppens(2016)와 Fan(2017)이 최전선을 예시하고, DKIST와 차세대 편광 장비가 제약을 제공할 것.

9. **Free magnetic energy reservoirs bridge prominences and CMEs / 자유 자기 에너지 저장이 홍염과 CME를 잇는다** — The non-potentiality that supports the prominence via twist or shear is simultaneously the reservoir of free magnetic energy that can drive eruption. A quasi-equilibrium flux rope can persist for days, but a perturbation (tether-cutting reconnection, flux emergence, torus instability onset) can unleash ~10$^{31}$–$10^{32}$ erg in minutes. Teardrop cavity morphology is an empirical predictor — the underlying X-line is the physical mechanism. / 홍염을 지지하는 비포텐셜성(twist 또는 shear)이 동시에 폭발을 이끄는 자유 자기 에너지 저장. 준평형 flux rope이 수일 지속되다가 교란(tether-cutting 재결합, 자속 방출, torus 불안정성 시작)으로 수 분 만에 ~10$^{31}$–$10^{32}$ erg 방출. 눈물방울 cavity 형태가 경험적 예측자 — 아래의 X-line이 물리적 기구.

10. **The spectrum-of-models picture resolves the debate / 모델 스펙트럼 관점이 논쟁을 해소한다** — Gibson's (conclusion #2) central insight is that sheared arcade ↔ BPSS flux rope ↔ X-line flux rope ↔ spheromak may not be discrete alternatives but a continuum with observable transition markers. Stability is lost at different points along this spectrum for active region vs. quiescent prominences. / Gibson의 결론 #2 핵심 통찰: sheared arcade ↔ BPSS flux rope ↔ X-line flux rope ↔ spheromak은 분리된 대안이 아니라 관측 전이 지표를 가진 연속체일 수 있다. 활성 영역과 quiescent 홍염에서 안정성 상실 지점이 다르다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Plasma β and force-free condition / 플라즈마 β와 force-free 조건

$$\beta = \frac{p}{B^2/(2\mu_0)} = \frac{2\mu_0 n k_B T}{B^2}$$

For a prominence ($n \sim 10^{10}$–$10^{11}$ cm$^{-3}$, $T \sim 7{,}500$ K, $B \sim 10$ G):
$$p \approx n k_B T \approx 10^{10} \cdot 1.38\times10^{-16} \cdot 7500 \approx 10^{-2}\text{ dyne cm}^{-2}$$
$$B^2/8\pi \approx 100/(8\pi) \approx 4\text{ dyne cm}^{-2} \Rightarrow \beta \sim 2.5\times10^{-3} \ll 1$$

Hence to leading order $\mathbf{J}\times\mathbf{B} = 0$, equivalently:
$$\nabla\times\mathbf{B} = \alpha(\mathbf{r})\,\mathbf{B}, \qquad \mathbf{B}\cdot\nabla\alpha = 0$$
$\alpha$ is constant along each field line.

### 4.2 KS sheet: force balance and analytic form / KS 시트: 힘 균형과 해석해

Vertical momentum for a stationary, isothermal, infinitely thin vertical sheet with horizontal field $B_x$ and jump $[B_z]$ across the sheet:
$$\rho g = \frac{B_x(x=0)\,[B_z]}{\mu_0}$$
The upward Lorentz force per unit area equals the weight per unit area.

A self-consistent isothermal 2.5D solution (Kippenhahn & Schlüter 1957; schematic):
$$B_x = B_{x0} = \text{const}, \qquad B_z(x) = B_{z,\infty}\,\tanh(x/H)$$
$$p(x) = p_0\,\text{sech}^2(x/H), \qquad \rho(x) = \frac{p(x)}{k_B T / m_p}$$
with
$$H = \frac{B_{x0} B_{z,\infty}}{\mu_0 \rho_0 g} = \frac{2 k_B T}{m_p g}\cdot\frac{B_{x0} B_{z,\infty}}{2\mu_0 p_0}$$
Numerical sanity check: for $T=7500$ K, $g = 274$ m/s², $2k_B T/m_p g \approx 0.47$ Mm (hydrostatic scale); with $B_{x0}=10$ G, $B_{z,\infty}=5$ G, $p_0=10^{-2}$ dyne/cm², the magnetic/thermal ratio ~200 so $H \sim$ tens of Mm, much larger than the hydrostatic scale — the sheet is magnetically supported.

### 4.3 Kuperus–Raadu (KR): line current + image current / KR: 선 전류와 이미지 전류

Line current $I$ at height $h$, image $-I$ at $-h$. Mutual force per unit length (SI):
$$F_{\text{image}} = \frac{\mu_0 I^2}{4\pi h}\quad\text{(upward, repulsive)}$$

Background horizontal field $B_{\text{bg}}$ exerts downward force per unit length $I B_{\text{bg}}/c$ (cgs) on the line current (sign chosen so the KR geometry confines the filament). Gravity pulls mass per unit length $m$ down:
$$\frac{\mu_0 I^2}{4\pi h} - I B_{\text{bg}} - m g = 0 \quad\text{(force-free equilibrium height)}$$

In the limit $m \to 0$ this reduces to a force-free flux rope equilibrium with a specific axis height set by $I$ and $B_{\text{bg}}$.

### 4.4 Sheared arcade: linear force-free, periodic boundary / Sheared arcade: 선형 force-free

For a 2.5D arcade with translational invariance in $y$ (PIL direction) and photospheric $B_z(x,0) = B_0\cos(kx)$:
$$\mathbf{B} = \left(-\frac{B_0\,k_\perp}{k}\sin(kx)e^{-k_\perp z},\;\frac{B_0\,\alpha}{k}\sin(kx)e^{-k_\perp z},\;B_0\cos(kx)e^{-k_\perp z}\right)$$
$$k_\perp = \sqrt{k^2 - \alpha^2}, \qquad \nabla\times\mathbf{B} = \alpha\mathbf{B}$$
Shear is controlled by $\alpha$; $\alpha=0$ is potential, $\alpha\to k$ brings arcade toward open configuration.

### 4.5 Titov–Démoulin (TDm) and Gibson–Low flux rope / TDm와 Gibson–Low flux rope

**TDm (1999)**: Toroidal current ring of radius $R$ at submerged depth $d$ below photosphere, plus two subphotospheric magnetic charges $\pm q$ providing background strapping field $B_{\text{bg}}$. Rope self-inductance and background field give equilibrium condition (torus instability threshold when decay index $n_{\text{crit}} = -d\ln B_{\text{bg}}/d\ln R \approx 1.5$).

**Gibson–Low (1998)**: Analytic magnetostatic prominence model. Spherical coordinates $(r,\theta,\phi)$ with stretched-sphere transformation $r \to r - a$ produces a detached spheromak flux rope with prescribed pressure/density that satisfies full (non-force-free) MHD equilibrium:
$$\mathbf{J}\times\mathbf{B} - \nabla p - \rho\nabla\Phi = 0$$
The solution supports a prominence-like sheet of dense plasma in a helical flux rope embedded in a coronal cavity.

### 4.6 Thermal Nonequilibrium (TNE) condensation criterion / TNE 응축 조건

Along a 1D coronal loop with arc length $s$, the hydrodynamic energy equation balances thermal conduction, radiative cooling, and (localized) heating:
$$\rho c_p \frac{\partial T}{\partial t} = -\rho c_p v\,\partial_s T + \partial_s(\kappa_\parallel \partial_s T) - n^2\Lambda(T) + H(s,t)$$

If heating $H$ is **strongly concentrated near footpoints** (≲0.3 loop length scale), a radiative runaway occurs when radiative losses exceed heating plus conduction at the top of the loop. Approximate cooling criterion (Karpen et al. 2001): the heating skew parameter $\zeta = H_{\text{footpoint}}/H_{\text{total}}$ must exceed a threshold depending on loop geometry and background heating for condensations to form. Numerically, TNE 1D simulations produce cold dense blobs ($T\sim10^4$ K, $n \sim 10^{10}$–$10^{11}$ cm$^{-3}$) on timescales of hours, repeating quasi-periodically.

### 4.7 Free magnetic energy / 자유 자기 에너지

$$E_{\text{mag}} = \int \frac{B^2}{2\mu_0}\,dV, \quad E_{\text{pot}} = \min_{\mathbf{B}|_{\partial V}=\mathbf{B}_{\text{obs}}} \int \frac{B^2}{2\mu_0}\,dV$$
$$E_{\text{free}} = E_{\text{mag}} - E_{\text{pot}} \geq 0$$

A potential arcade has $E_{\text{free}} = 0$; a flux rope *cannot* be constructed from boundary data alone without adding twist, so $E_{\text{free}} > 0$. Numerical estimate for a typical active-region filament with $B\sim100$ G, volume $V\sim(100\text{ Mm})^3$: $E_{\text{mag}} \sim 10^{32}$–$10^{33}$ erg, $E_{\text{free}}$ a few $\times 10^{31}$–$10^{32}$ erg — enough to power a major CME ($\sim 10^{32}$ erg).

### 4.8 Magnetic Rayleigh–Taylor (MRT) growth rate / MRT 성장률

For heavy plasma above light in a magnetized atmosphere, interchange modes perpendicular to $\mathbf{B}$ grow at
$$\omega^2 = -g k \frac{\rho_1 - \rho_2}{\rho_1 + \rho_2} + \frac{2 (\mathbf{k}\cdot\mathbf{B})^2}{\mu_0(\rho_1+\rho_2)}$$
Magnetic tension stabilizes modes *parallel* to $\mathbf{B}$ but not *perpendicular* (interchange). With prominence B ~ 10 G, $\rho\sim 10^{-14}$ g/cm³, $g=2.74\times10^4$ cm/s²: for $k=1/(1\text{ Mm})$, $\omega \sim 10^{-3}$ s$^{-1}$ → characteristic time ~1000 s, consistent with observed bubble rise times (~10$^3$ s).

### 4.9 Cool-plasma formation criterion (radiative runaway) / 냉각 플라즈마 형성 조건

The onset of TNE condensation is governed by the competition between radiative losses $n^2\Lambda(T)$ and heating $H$ at the loop top. A necessary condition for runaway cooling along an idealized 1D loop of length $L$ with footpoint-concentrated heating $H(s)=H_0 e^{-s/s_H}$ is:
$$\frac{n_{\text{top}}^2\,\Lambda(T_{\text{top}})}{H(L/2)} \gg 1,\qquad s_H \ll L$$
For coronal parameters ($n\sim10^9$ cm$^{-3}$, $T\sim 10^6$ K, $\Lambda\sim 10^{-22}$ erg cm³/s, $L\sim 100$ Mm), required heating skew is $s_H/L \lesssim 0.1$–$0.3$. TNE then produces cool condensations at $T\sim10^4$ K on timescales of hours. This mechanism works **with or without magnetic dips** — flat long loops suffice (Karpen et al. 2001).

### 4.10 Kippenhahn–Schlüter sheet thickness — numerical evaluation / KS 시트 두께 수치 평가

Using the KS self-similar solution with prominence parameters:
- $B_{x0} = 10$ G, $B_{z,\infty}=5$ G (horizontal and asymptotic transverse field)
- $p_0 = 0.1$ dyne cm$^{-2}$ (central thermal pressure, order of Rust 1967)
- $g_\odot = 2.74\times 10^4$ cm s$^{-2}$

Characteristic sheet half-thickness:
$$H = \frac{B_{x0}\, B_{z,\infty}}{4\pi p_0\,(g/c_s^2)}\quad\text{(schematic cgs form)}$$
With isothermal sound speed $c_s = \sqrt{k_B T/m_p}\approx 8$ km/s at $T=7500$ K, hydrostatic scale height $c_s^2/g \approx 230$ km. Magnetic-to-thermal ratio $B^2/(8\pi p_0) = 100/(8\pi\cdot 0.1) \approx 40$, yielding $H\sim 10^4$ km = 10 Mm — the sheet is **much thicker** than the thermal scale, consistent with the observed ~10$^3$–$10^4$ km prominence sheet widths.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1957  ├─ Kippenhahn & Schlüter: 2.5D dipped arcade (KS model)              [foundation]
1967  ├─ Rust: B ~ 5-10 G, plasma β << 1 → force-free prominence          [observational anchor]
1974  ├─ Kuperus & Raadu: inverse-polarity flux rope with image currents   [rival paradigm]
1982  ├─ Low: "invisible man" cavity around prominence
1983  ├─ Malherbe & Priest: polarity senses vs. topology
1989  ├─ Leroy et al.: 75-90% of prominences are inverse
1991  ├─ Antiochos & Klimchuk: Thermal Nonequilibrium for prominences
1994  ├─ Antiochos et al.: 3D sheared-arcade dipped fields                 [3D turning point]
1998  ├─ Titov & Démoulin: analytic 3D flux-rope model (TDm)               [benchmark model]
1998  ├─ Aulanier & Démoulin: linear FF rope with barbs from parasitic
1998  ├─ Gibson & Low: spheromak prominence model                          [non-force-free magnetostatic]
2001  ├─ Karpen et al.: "No" — TNE does not require dips
2003  ├─ Linker et al.: MHD simulations of prominence/cavity from rope
2006  ├─ Fan & Gibson; Gibson & Fan: BPSS sigmoids, partial eruption
2010  ├─ Berger et al.; Ryutova et al.; Hillier: MRT plumes
2012  ├─ Luna et al.: 3D TNE in sheared arcade
2016  ├─ Xia & Keppens: full 3D MHD prominence-in-flux-rope                [Full Monty I]
2017  ├─ Fan: 3D MHD in spherical coords with eruption                     [Full Monty II]
2018  ◄── Gibson (this review) — synthesizes 60 years of theory & models
2019+ ── DKIST era, data-driven MHD with thermodynamics
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Paper #36: Parenti (2014), "Solar Prominences: Observations"** (Living Reviews) | Direct observational companion to this theoretical review; Gibson explicitly references Parenti for mass, temperature, velocity, and morphology data / 이 이론 리뷰의 직접적 관측 동반 논문; Gibson이 질량·온도·속도·형태 자료를 Parenti로 명시 참조 | **High** — the two reviews together form the complete prominence physics picture / 이 둘이 홍염 물리 전체 그림을 완성 |
| Kippenhahn & Schlüter (1957, Zs. Astrophys.) | The foundational dipped-arcade analytic prominence model; KS model discussed throughout §2 / 기초 dipped arcade 해석 모델 | **Foundational** — the starting point for all prominence theory |
| Kuperus & Raadu (1974, A&A) | The rival flux-rope paradigm with image currents; KR model discussed throughout §2 / 이미지 전류 flux rope 모델 | **Foundational** — the alternative paradigm |
| Antiochos & Klimchuk (1991) + Antiochos et al. (1999, 2000) | Thermal Nonequilibrium — prominence formation without dips; §4.2.1 / Dip 없이 홍염 형성하는 TNE | **Central** to the "beyond equilibrium" discussion |
| Titov & Démoulin (1999, A&A 351:707) | Analytic 3D current-carrying flux-rope equilibrium (TDm); reference model for torus instability; §2.4.2 / 3D 해석적 flux rope (TDm); torus 불안정성 기준 | **High** — the benchmark flux-rope model |
| Gibson & Low (1998, 2000, 2006) | Spheromak prominence + chirality explanations + BPSS sigmoids; Sections 2–3 / Spheromak + chirality + BPSS | **Author's own series** — foundational for §3 |
| Xia & Keppens (2016, ApJ 823:22) + Fan (2017, ApJ 844:26) | The two "Full Monty" 3D MHD prominence-in-flux-rope simulations; §4.3 / 두 완전 3D MHD 시뮬레이션 | **State-of-the-art** — exemplify the future direction |
| Berger et al. (2011); Hillier et al. (2011, 2012); Hillier (2018) | MRT instability in prominences; §4.1 / 홍염 MRT 불안정성 | **High** for dynamics section |
| Mackay et al. (2010, *Space Sci Rev*) + Labrosse et al. (2010) | Earlier comprehensive prominence reviews that Gibson updates and extends / Gibson이 갱신·확장한 선행 리뷰 | **High** — establishes the review baseline |
| Engvold & Vial (2015, *Solar Prominences* book) | Companion volume of chapters (Ballester, Gibson, Gopalswamy, Kucera, Labrosse, Martin, Parenti, Webb) covering the broader prominence field / 홍염 분야 전반을 다루는 동반 서적 | **High** — contextual framework |
| Arregui et al. (2018, *LRSP*) | Prominence oscillations review — complements Gibson's dynamics section / 홍염 진동 리뷰 — 동역학 절을 보완 | **Medium** for §4.1 |
| Titov & Démoulin topology papers (1999 onward) | BPSS/QSL formalism underlying Gibson's sigmoid and chirality discussion / Gibson의 시그모이드·chirality 논의 기저의 BPSS/QSL 형식론 | **High** for §3 |
| Low (1996, 2001, 2018 "Coronal Magnetism" review) | Big-picture context for coronal magnetic equilibria; Gibson recommends Low (2018) for broader context / 코로나 자기 평형의 큰 그림 | **Medium** — philosophical framing |

---

## 7. References / 참고문헌

- Gibson, S. E., "Solar Prominences: Theory and Models — Fleshing out the magnetic skeleton", *Living Reviews in Solar Physics*, 15:7 (2018). DOI: [10.1007/s41116-018-0016-2](https://doi.org/10.1007/s41116-018-0016-2)
- Kippenhahn, R., Schlüter, A., "Eine Theorie der solaren Filamente", *Zeitschrift für Astrophysik*, 43, 36 (1957).
- Kuperus, M., Raadu, M. A., "The Support of Prominences Formed in Neutral Sheets", *Astronomy & Astrophysics*, 31, 189 (1974).
- Rust, D. M., "Measurements of the Magnetic Field in Quiescent Solar Prominences", *Astrophysical Journal*, 150, 313 (1967). DOI: [10.1086/149334](https://doi.org/10.1086/149334)
- Leroy, J.-L., Bommier, V., Sahal-Bréchot, S., "New data on the magnetic structure of quiescent prominences", *Astronomy & Astrophysics*, 131, 33 (1984).
- Malherbe, J. M., Priest, E. R., "Current sheet models for solar prominences", *Astronomy & Astrophysics*, 123, 80 (1983).
- Antiochos, S. K., Klimchuk, J. A., "A model for the formation of solar prominences", *Astrophysical Journal*, 378, 372 (1991). DOI: [10.1086/170437](https://doi.org/10.1086/170437)
- Antiochos, S. K., Dahlburg, R. B., Klimchuk, J. A., "The magnetic field of solar prominences", *Astrophysical Journal Letters*, 420, L41 (1994). DOI: [10.1086/187158](https://doi.org/10.1086/187158)
- Antiochos, S. K., MacNeice, P. J., Spicer, D. S., Klimchuk, J. A., "The dynamic formation of prominence condensations", *Astrophysical Journal*, 512, 985 (1999). DOI: [10.1086/306804](https://doi.org/10.1086/306804)
- Titov, V. S., Démoulin, P., "Basic topology of twisted magnetic configurations in solar flares", *Astronomy & Astrophysics*, 351, 707 (1999).
- Gibson, S. E., Low, B. C., "A time-dependent three-dimensional magnetohydrodynamic model of the coronal mass ejection", *Astrophysical Journal*, 493, 460 (1998). DOI: [10.1086/305107](https://doi.org/10.1086/305107)
- Aulanier, G., Démoulin, P., "3-D magnetic configurations supporting prominences. I.", *Astronomy & Astrophysics*, 329, 1125 (1998).
- Karpen, J. T., Antiochos, S. K., Hohensee, M., Klimchuk, J. A., MacNeice, P. J., "Are Magnetic Dips Necessary for Prominence Formation?", *Astrophysical Journal Letters*, 553, L85 (2001). DOI: [10.1086/320497](https://doi.org/10.1086/320497)
- Luna, M., Karpen, J. T., DeVore, C. R., "Formation and evolution of a multi-threaded solar prominence", *Astrophysical Journal*, 746, 30 (2012). DOI: [10.1088/0004-637X/746/1/30](https://doi.org/10.1088/0004-637X/746/1/30)
- Berger, T. E., Slater, G., Hurlburt, N., et al., "Quiescent Prominence Dynamics Observed with the Hinode Solar Optical Telescope. I.", *Astrophysical Journal*, 716, 1288 (2010). DOI: [10.1088/0004-637X/716/2/1288](https://doi.org/10.1088/0004-637X/716/2/1288)
- Hillier, A., Isobe, H., Shibata, K., Berger, T., "Numerical Simulations of the Magnetic Rayleigh–Taylor Instability in the Kippenhahn–Schlüter Prominence Model", *Astrophysical Journal*, 746, 120 (2012). DOI: [10.1088/0004-637X/746/2/120](https://doi.org/10.1088/0004-637X/746/2/120)
- Xia, C., Keppens, R., "Formation and Plasma Circulation of Solar Prominences", *Astrophysical Journal*, 823, 22 (2016). DOI: [10.3847/0004-637X/823/1/22](https://doi.org/10.3847/0004-637X/823/1/22)
- Fan, Y., "MHD Simulations of the Eruption of Coronal Flux Ropes under Coronal Streamers", *Astrophysical Journal*, 844, 26 (2017). DOI: [10.3847/1538-4357/aa7a56](https://doi.org/10.3847/1538-4357/aa7a56)
- Parenti, S., "Solar Prominences: Observations", *Living Reviews in Solar Physics*, 11:1 (2014). DOI: [10.12942/lrsp-2014-1](https://doi.org/10.12942/lrsp-2014-1) *(Paper #36, observational companion)*
- Mackay, D. H., Karpen, J. T., Ballester, J. L., Schmieder, B., Aulanier, G., "Physics of Solar Prominences: II — Magnetic Structure and Dynamics", *Space Science Reviews*, 151, 333 (2010). DOI: [10.1007/s11214-010-9628-0](https://doi.org/10.1007/s11214-010-9628-0)
- Hillier, A., "The magnetic Rayleigh–Taylor instability in solar prominences", *Reviews of Modern Plasma Physics*, 2, 1 (2018). DOI: [10.1007/s41614-017-0013-2](https://doi.org/10.1007/s41614-017-0013-2)
- Low, B. C., "Coronal Mass Ejections, Magnetic Flux Ropes, and Solar Magnetism", *Journal of Geophysical Research*, 106, 25141 (2001). DOI: [10.1029/2000JA004015](https://doi.org/10.1029/2000JA004015)
- Low, B. C., Hundhausen, J. R., "Magnetostatic structures of the solar corona. II — The magnetic topology of quiescent prominences", *Astrophysical Journal*, 443, 818 (1995). DOI: [10.1086/175572](https://doi.org/10.1086/175572)
- Linker, J. A., Mikić, Z., Lionello, R., et al., "Flux cancellation and coronal mass ejections", *Physics of Plasmas*, 10, 1971 (2003). DOI: [10.1063/1.1563668](https://doi.org/10.1063/1.1563668)
- Fan, Y., Gibson, S. E., "Numerical simulations of three-dimensional coronal magnetic fields resulting from the emergence of twisted magnetic flux tubes", *Astrophysical Journal*, 609, 1123 (2004). DOI: [10.1086/421238](https://doi.org/10.1086/421238)
- van Ballegooijen, A. A., Martens, P. C. H., "Formation and eruption of solar prominences", *Astrophysical Journal*, 343, 971 (1989). DOI: [10.1086/167766](https://doi.org/10.1086/167766)
- Démoulin, P., Priest, E. R., "The importance of magnetic fluctuations for solar flares", *Solar Physics*, 144, 283 (1993).
- Bąk-Stęślicka, U., Gibson, S. E., Fan, Y., et al., "The magnetic structure of solar prominence cavities: New observational signature revealed by coronal magnetometry", *Astrophysical Journal Letters*, 770, L28 (2013). DOI: [10.1088/2041-8205/770/2/L28](https://doi.org/10.1088/2041-8205/770/2/L28)
- Terradas, J., Soler, R., Luna, M., Oliver, R., Ballester, J. L., "Morphology and dynamics of solar prominences from 3D MHD simulations", *Astrophysical Journal*, 799, 94 (2015). DOI: [10.1088/0004-637X/799/1/94](https://doi.org/10.1088/0004-637X/799/1/94)

### Appendix: Cross-reference to Paper #36 (Parenti 2014 — observational companion) / 부록: Paper #36 교차 참조

**English**: Paper #36 (Parenti 2014, "Solar Prominences: Observations") supplies the observational numbers that anchor every theoretical claim in Gibson (2018). Key linkages:

| Gibson 2018 theoretical claim / 이론적 주장 | Parenti 2014 observational anchor / Parenti 관측 근거 |
|---|---|
| Low plasma β (≪1) motivates force-free prominence / 저 β가 force-free 홍염 동기 | $T\sim 7{,}500$ K, $n\sim 10^{10}$–$10^{11}$ cm$^{-3}$, $B\sim 5$–$30$ G (Parenti §3) |
| Prominences are intrinsically dynamic / 본질적 동적 | $v\sim$ few–tens km/s, counterstreaming, plumes (Parenti §4) |
| Sheet-like morphology regardless of skeleton / 골격 무관 시트 형태 | Observed vertical threads 100–300 km wide (Parenti §2) |
| Prominence-corona transition region / 홍염-코로나 전이 영역 | Steep T gradient over $\sim 1000$ km (Parenti §3.3) |
| Chirality rules / Chirality 규칙 | Dextral dominant in northern hemisphere (Parenti §2.4) |

Together, the two reviews form the complete picture: Parenti describes what is seen, Gibson explains why it looks that way.

**한국어**: Paper #36은 Gibson의 모든 이론적 주장을 뒷받침하는 관측 수치를 제공. Parenti는 "무엇이 보이는가", Gibson은 "왜 그렇게 보이는가"를 설명 — 둘이 완전한 그림을 이룬다.

### Bilingual glossary of model acronyms / 모델 약어 대역 용어집

- **KS (Kippenhahn–Schlüter)** / KS 모델 — 2.5D dipped arcade with current-sheet support. / 전류 시트 지지 2.5D dipped arcade.
- **KR (Kuperus–Raadu)** / KR 모델 — 2.5D current-filament flux rope with image currents. / 이미지 전류 2.5D 전류-필라멘트 flux rope.
- **TDm (Titov–Démoulin)** / TDm 모델 — Analytic 3D current-carrying flux rope equilibrium (1999). / 3D 해석적 전류 운반 flux rope 평형.
- **TNE (Thermal Nonequilibrium)** / 열 비평형 — Footpoint-localized heating → radiative runaway → cool condensations. / 풋포인트 국소 가열 → 복사 runaway → 냉각 응축.
- **QSL (Quasi-Separatrix Layer)** / 준-separatrix 층 — Steep gradient in field-line connectivity; current-sheet nursery. / 장선 연결성의 가파른 구배; 전류 시트 형성지.
- **BPSS (Bald-Patch Separatrix Surface)** / BPSS — Separatrix from tangential photospheric touches. / 광구 접선 접촉에서 뻗는 separatrix 표면.
- **MRT (Magnetic Rayleigh–Taylor)** / 자기 RT — Interchange instability driving prominence plumes/bubbles. / 홍염 plume/bubble을 이끄는 interchange 불안정성.
- **NLFF (Nonlinear Force-Free)** / 비선형 force-free — Force-free with spatially varying α. / 공간에 따라 α가 변하는 force-free.
- **WPFS (Whole-Prominence Fine Structure)** / 전체 홍염 미세 구조 — Gunár & Mackay 3D NLFF + hydrostatic approach. / Gunár & Mackay의 3D NLFF + 정역학 접근.
- **CoMP** / CoMP — Coronal Multichannel Polarimeter for infrared polarization of cavities. / 코로나 cavity의 적외 편광 관측 장비.
