---
title: "Flare Observations"
authors: Arnold O. Benz
year: 2008
journal: "Living Rev. Solar Phys., 5, 1"
topic: Living Reviews in Solar Physics / Solar Flares
tags: [solar flares, hard X-rays, soft X-rays, RHESSI, magnetic reconnection, particle acceleration, Neupert effect, chromospheric evaporation, energy budget, bremsstrahlung, gyrosynchrotron, CSHKP model, Transit-Time Damping]
status: completed
date_started: 2026-04-13
date_completed: 2026-04-13
---

# Flare Observations — Arnold O. Benz (2008)

---

## 핵심 기여 / Core Contribution

이 리뷰는 태양 플레어 관측의 현대적 종합이다. RHESSI, Yohkoh, TRACE, SOHO 등의 우주 미션이 밝혀낸 플레어의 다파장 관측 결과를 체계적으로 정리하며, 에너지 방출(자기 재결합), 에너지 전달(입자 가속과 채층 증발), 그리고 에너지 배분(에너지 수지)을 관측적 증거에 기반하여 논의한다. 특히 RHESSI의 hard X-ray 영상 분광 관측이 가져온 새로운 발견들 — coronal source와 footpoint의 관계, 양성자와 전자의 가속 위치 차이, soft-hard-soft 행동의 정량화 — 을 강조한다. 표준 플레어 모델(CSHKP)을 소개하되 그 한계를 명확히 지적하고, 가속 메커니즘으로서 Transit-Time Damping이 관측과 가장 잘 부합함을 보인다.

This review is a modern synthesis of solar flare observations. It systematically organizes multi-wavelength results from space missions (RHESSI, Yohkoh, TRACE, SOHO), discussing energy release (magnetic reconnection), energy transport (particle acceleration and chromospheric evaporation), and energy partition (energy budget) based on observational evidence. It highlights new discoveries from RHESSI's hard X-ray imaging spectroscopy — the coronal source–footpoint relationship, different acceleration sites for protons and electrons, and quantification of soft-hard-soft behavior. While introducing the standard flare model (CSHKP), it clearly identifies its limitations and shows that Transit-Time Damping best matches observations as an acceleration mechanism.

---

## 읽기 노트 / Reading Notes

### 1. Introduction — 플레어의 정의와 개요 / Flare Definition and Overview

**플레어의 정의 / Definition of a flare:**
1859년 Carrington과 Hodgson이 백색광에서 처음 관측한 이후, "flare"라는 용어는 다소 모호하게 사용되어 왔다. 현대적 정의는 "자기 재결합에 의한 자기 에너지의 갑작스런 방출(sudden release of magnetic energy by reconnection)"이다. 그러나 이 정의는 관측적이라기보다 이론적이며, 실제로 태양 대기의 다른 밝아짐 현상(자기 플럭스 방출, 충격파 소산 등)과 구별하기 어렵다. 관측적으로는 "전자기 스펙트럼 전체에 걸친 수 분 시간 규모의 밝아짐"으로 정의하는 것이 적절하다.

Since Carrington and Hodgson's first white-light observation in 1859, the term "flare" has been used rather loosely. The modern definition is "a sudden release of magnetic energy by reconnection." However, this is more theoretical than observational, making it difficult to distinguish from other brightening phenomena (magnetic flux expulsion, shock wave dissipation, etc.). Observationally, it is better defined as "a brightening across the electromagnetic spectrum on a timescale of minutes."

**플레어의 4단계 / Four phases of flares (Figure 2):**

| 단계 / Phase | 시간 규모 / Timescale | 주요 방출 / Key Emissions | 물리 과정 / Physical Process |
|---|---|---|---|
| **Preflare** | 수 분 / Few minutes | SXR, EUV | 코로나 플라즈마 서서히 가열 / Coronal plasma slowly heats |
| **Impulsive** | 3–10 분 / 3–10 min | HXR, microwave, EUV | 에너지 대부분 방출, 입자 가속, HXR footpoint 출현 / Most energy released, particle acceleration, HXR footpoints appear |
| **Flash** | 5–20 분 / 5–20 min | H$\alpha$, SXR peak | H$\alpha$ 급격 증가, 열적 방출 최대 / Rapid H$\alpha$ increase, thermal emission peaks |
| **Decay** | 수 시간 / Hours | SXR, metric radio | 코로나 원래 상태로 복귀, post-flare 루프 성장 / Corona returns to original state, post-flare loops grow |

**중요 관점 / Key perspective:**
플레어는 순수한 코로나 현상이 아니다. 코로나와 채층이 상호작용하는 시스템으로 이해해야 한다. 대부분의 관측 가능한 현상(H$\alpha$, 백색광, footpoint HXR)은 에너지 방출의 2차적 반응이다.

Flares are not purely coronal phenomena. They must be understood as a system where corona and chromosphere interact. Most observable phenomena (H$\alpha$, white light, footpoint HXR) are secondary responses to the primary energy release.

**CME와의 관계 / Relation to CMEs:**
CME는 플레어의 단순한 결과가 아니다. CME는 자체적인 자기적 구동력(magnetic driver)을 가지며, 오히려 CME가 재결합 조건을 만들어 플레어를 유발할 수 있다. 둘은 독립적이지만 연관된 과정이다.

A CME is not simply an explosive result of a flare. CMEs have their own magnetic driver and may even lead to reconnection conditions, causing a flare. They are independent but related processes.

---

### 2. Energy Release in a Coupled Solar Atmosphere — 결합된 태양 대기에서의 에너지 방출

#### 2.1 광구 자기장 배위 / Photospheric Field Configuration

플레어는 태양 어디서든 발생할 수 있지만, 대규모 플레어는 활동 영역(active region)에서 선호적으로 발생한다. 관측적 특징:

Flares can occur anywhere on the Sun, but large flares preferentially occur in active regions. Observational features:

- **$\delta$ configuration**: 반대 극성의 자기장이 하나의 반암(penumbra) 안에 공존 → 대규모 플레어의 필수 조건 / Opposite polarities coexist within one penumbra → necessary condition for large flares
- **Magnetic shear**: 플레어 위치의 자기장이 강하게 전단(shear)됨. 그러나 이것만으로는 충분 조건이 아님 / Magnetic field at flare sites is strongly sheared, but this alone is not sufficient
- **Emerging flux**: 새로운 자기 플럭스의 부상이 대규모 플레어의 추가 필요조건 / Emergence of new magnetic flux is an additional requirement for large flares
- **Separatrix / Quasi-separatrix layers**: 플레어가 자주 발생하는 위치는 서로 다른 자기 루프 시스템의 분리면(separatrix) 또는 cusp 형태의 코로나 구조 / Flares frequently occur at the separatrix between different magnetic loop systems or cusp-shaped coronal structures

**핵심 결론**: 플레어 에너지는 광구 아래에서 기원한 자기장에 저장되어 있으며, 자유 자기 에너지(free magnetic energy = 현재 자기장 - potential field의 에너지 차이), 즉 코로나의 전류에 의해 유지되는 에너지가 방출되는 것이다.

**Key conclusion**: Flare energy is stored in the magnetic field originating from below the photosphere. What is released is the free magnetic energy (difference between actual field and potential field energy), maintained by electric currents in the corona.

#### 2.2 Hard X-ray 기하학 / Geometry of Hard X-ray Emissions

**Bremsstrahlung의 두 모델 / Two bremsstrahlung models:**

1. **Thin target**: 전자빔이 플라즈마를 통과하면서 에너지를 거의 잃지 않는 경우. 단일 충돌에 의한 순간 방출. 코로나 source에 해당 / Electron beam passes through plasma with negligible energy loss. Instantaneous emission from single collision. Corresponds to coronal source
2. **Thick target** (Brown, 1971): 전자가 밀도 높은 채층에서 완전히 정지할 때까지 모든 충돌에 의한 총 방출. Footpoint에 해당. Power-law index가 thin target보다 2 더 flat / Total emission from all collisions until electron stops in dense chromosphere. Corresponds to footpoints. Power-law index is 2 flatter than thin target

**RHESSI가 밝혀낸 핵심 관측 사실들:**

RHESSI의 영상 분광 능력은 플레어 물리학에 혁명적 진전을 가져왔다:

- 플레어 hard X-ray의 95% 이상($\gtrsim 150$ keV)이 고도 $\lesssim 2500$ km(채층)에서 방출 → thick target 모델 지지 / Over 95% of flare hard X-rays ($\gtrsim 150$ keV) originate at altitudes $\lesssim 2500$ km (chromosphere) → supports thick target model
- Footpoint는 움직인다 — 재결합이 진행됨에 따라 에너지 방출 위치가 이동 / Footpoints move — energy release site shifts as reconnection progresses
- Footpoint 사이의 분리는 cusp 위의 재결합과 일치 → flare ribbon과 대응 / Footpoint separation matches reconnection above cusp → corresponds to flare ribbons
- Coronal source는 주로 열적(thermal) — 비열적 부분은 약함 / Coronal source is primarily thermal — non-thermal component is weak
- 코로나와 footpoint의 비열적 방출은 시간적으로 잘 상관됨 → 코로나-채층 사이의 강한 결합 / Non-thermal emission from corona and footpoints is well correlated in time → strong corona–chromosphere coupling

#### 2.3 Return Current / 반환 전류

비열적 전자와 이온의 flux가 채층 방향으로 불균형하면, 전하와 전류 중성(neutrality)을 유지하기 위해 반환 전류(return current)가 형성된다. 이 전류는 주로 열적 전자가 자기장선을 따라 위로 자유롭게 이동하여 구성된다.

When the flux of non-thermal electrons and ions toward the chromosphere is unbalanced, a return current forms to maintain charge and current neutrality. This current is primarily composed of thermal electrons moving freely upward along field lines.

반환 전류의 관측적 증거: 두 footpoint의 non-thermal power-law index가 동일(2.7±0.1)한 반면, coronal source는 5.6±0.1로 차이가 큼 → 반환 전류의 전기장이 footpoint 스펙트럼을 flat하게 만드는 효과로 해석 가능.

Observational evidence: both footpoints have identical non-thermal power-law indices (2.7±0.1), while the coronal source shows 5.6±0.1 → interpretable as the return current's electric field flattening footpoint spectra.

반환 전류는 "flare particle number problem"도 해결한다: HXR에서 추정되는 precipitating electron 수가 매우 많아 코로나 가속 영역이 곧 고갈될 것이나, 반환 전류가 밀도를 유지해 준다.

The return current also solves the "flare particle number problem": the number of precipitating electrons inferred from HXR is so large that the coronal acceleration region would soon be depleted, but the return current maintains its density.

#### 2.4 Neupert Effect / Neupert 효과

Neupert (1968)가 처음 발견한 경험적 관계:

$$F_{SXR}(t) \propto \int_{t_0}^{t} F_{HXR}(t') \, dt' \quad \Longleftrightarrow \quad \frac{d}{dt} F_{SXR}(t) \propto F_{HXR}(t)$$

**물리적 해석 / Physical interpretation:**
- HXR = 비열적 전자의 순간적 에너지 입력률 (bremsstrahlung) / Instantaneous energy input rate from non-thermal electrons
- SXR = 가열된 플라즈마의 누적 열 에너지 (thermal emission) / Accumulated thermal energy of heated plasma
- 따라서 SXR은 HXR의 시간 적분에 비례 / Therefore SXR is proportional to the time integral of HXR

**한계**: 이 관계는 냉각(전도나 복사)이 무시 가능할 때만 성립하는 근사이다. 실제로 플레어의 약 절반에서는 SXR과 HXR의 상대적 타이밍이 Neupert 관계를 위반한다 — 특히 preflare 가열이 있는 경우. 이는 비열적 전자만이 SXR 플라즈마의 유일한 가열원이 아닐 수 있음을 시사한다.

**Limitation**: This relation is an approximation valid only when cooling (conduction or radiation) is negligible. In practice, about half of flares violate the Neupert timing relation — especially when preflare heating exists. This suggests non-thermal electrons may not be the sole heating source for SXR plasma.

#### 2.5 Standard Flare Model (CSHKP) / 표준 플레어 모델

표준 모델(Carmichael–Sturrock–Hirayama–Kopp–Pneuman)의 시나리오:

The standard model (CSHKP) scenario:

1. **에너지 방출**: 코로나 고고도에서 자기 재결합 발생 → 자기 에너지가 운동 에너지와 열로 변환 / **Energy release**: Magnetic reconnection occurs at high coronal altitude → magnetic energy converts to kinetic energy and heat
2. **입자 가속**: 재결합 영역에서 전자와 이온이 가속됨 (가속 메커니즘 자체는 모델에 포함되지 않음) / **Particle acceleration**: Electrons and ions accelerated in reconnection region (acceleration mechanism itself not part of model)
3. **에너지 전달**: 가속된 입자가 자기장선을 따라 채층으로 precipitate → bremsstrahlung으로 HXR 방출 (footpoint) / **Energy transport**: Accelerated particles precipitate along field lines to chromosphere → emit HXR via bremsstrahlung (footpoints)
4. **채층 가열**: 입자가 채층에 에너지를 deposite → 온도 급상승 → 고온 플라즈마가 루프를 따라 코로나로 팽창 ("evaporation") / **Chromospheric heating**: Particles deposit energy in chromosphere → temperature surges → hot plasma expands along loop into corona ("evaporation")
5. **SXR 방출**: 증발된 플라즈마가 SXR로 관측 / **SXR emission**: Evaporated plasma observed in SXR

**표준 모델이 설명하는 것:**
- Neupert effect (SXR ∝ 누적 HXR)
- HXR footpoint가 SXR 루프의 발 부분에 위치
- Coronal HXR source가 SXR 루프 위에 위치 (Masuda, 1994)
- 비열적 전자 에너지 > 열 에너지
- 채층 증발의 blue-shifted spectral lines

#### 2.6 Chromospheric Evaporation / 채층 증발

가속된 전자(또는 이온)가 코로나에서 밀도 높은 채층으로 precipitate하면, Coulomb 충돌로 에너지를 잃고 채층 플라즈마를 가열한다. 온도가 급상승하면 초과압력이 형성되어 고온 플라즈마가 자기장을 따라 양방향으로 팽창한다.

When accelerated electrons (or ions) precipitate from the corona into the dense chromosphere, they lose energy through Coulomb collisions and heat the chromospheric plasma. The temperature surge creates overpressure, causing hot plasma to expand along the magnetic field in both directions.

관측적 증거 / Observational evidence:
- Ca XIX의 blue-shifted line (Antonucci et al., 1982): 20 MK 플라즈마가 300–400 km/s로 상승 / 20 MK plasma rising at 300–400 km/s
- 루프를 따른 밀도 프로파일 증가 (Liu et al., 2006): 시간에 따라 루프가 채워짐 / Density profile increasing along loop over time
- SOHO/CDS: 230 km/s의 upflow velocity / 230 km/s upflow velocity

**"Evaporation"은 오해를 부르는 용어**: 실제로는 상전이(phase transition)가 아니라 MHD 과정에 가까운 폭발적 팽창이다. 상승하는 고온 플라즈마와 하강하는 저온 플라즈마는 운동량 보존에 의해 동일한 모멘타를 가진다.

**"Evaporation" is a misnomer**: It is not a phase transition but an explosive expansion closer to an MHD process. The upflowing hot plasma and downflowing cool plasma have equal momenta due to momentum conservation.

**두 종류의 증발 / Two types of evaporation:**
1. **Explosive evaporation**: 비열적 전자 flux $> 3 \times 10^{10}$ erg cm$^{-2}$ s$^{-1}$ → 초음속 팽창 / Non-thermal electron flux $> 3 \times 10^{10}$ erg cm$^{-2}$ s$^{-1}$ → supersonic expansion
2. **Gentle evaporation**: 비열적 전자 flux $< 3 \times 10^{10}$ erg cm$^{-2}$ s$^{-1}$ → 65 km/s 정도의 완만한 upflow. Preflare 단계와 post-flare 단계에서 관측 / Non-thermal electron flux below threshold → gentle upflow of ~65 km/s. Observed in preflare and post-flare phases

#### 2.7 표준 모델의 한계 / Deviations from Standard Model

표준 모델은 모든 관측을 설명하지 못한다:

The standard model does not explain all observations:

1. **Footpoint 없는 플레어**: 일부 플레어는 코로나에서만 HXR이 관측되고 footpoint가 없음. 루프 밀도가 매우 높아 가속된 전자가 코로나에서 이미 충돌하여 에너지를 잃기 때문 / Some flares show HXR only in the corona without footpoints. The loop density is so high that accelerated electrons lose energy through collisions already in the corona
2. **Neupert effect 위반**: 약 절반의 hard X-ray event에서 SXR이 HXR보다 먼저 나타남 (preheating). 비열적 전자만이 가열의 유일한 원인이 아닐 수 있음 / In about half of HXR events, SXR appears before HXR. Non-thermal electrons may not be the sole heating agent
3. **Coronal source의 온도 구조**: loop-top이 footpoint보다 일반적으로 더 뜨거움 — 에너지가 열적 전도만으로 전달되지 않음을 시사 / Loop-top is generally hotter than footpoints — suggests energy is not transported solely by thermal conduction
4. **Standard model의 수정**: 일부 코로나 입자는 에너지를 매우 적게 받아 Maxwellian 분포를 유지 (가열에 해당). 에너지 방출률이 높아지면 다른 입자들은 충분한 에너지를 얻어 비열적 분포를 발달시킴 / Some coronal particles gain so little energy that they maintain their Maxwellian distribution (corresponding to heating). When energy release rate increases, other particles gain enough energy to develop non-thermal distributions

---

### 3. Flare Geometry — 플레어 기하학

#### 3.1 코로나 자기장 기하학 / Coronal Magnetic Field Geometry

**CSHKP 기하학**: 2차원 모델로, 다리(legs)에서 조여진(pinched) 자기 루프를 포함. 루프의 꼭대기가 재결합으로 plasmoid로 방출된다. 가장 좋은 증거는 SXR에서 관측되는 수직 cusp 형태의 구조와 시간에 따른 cusp 성장이다.

**CSHKP geometry**: A 2D model involving a magnetic loop pinched at its legs. The loop-top is ejected as a plasmoid through reconnection. The best evidence comes from vertical cusp-shaped structures observed in SXR and their growth over time.

**관측적 증거:**
- X-point 위와 아래로 고온 플라즈마의 수평 유입(inflow) 관측 (Yokoyama et al., 2001)
- 고온 plasma blob의 방출 (ejection) — 전파에서 drifting pulsating structure로 관측
- Two ribbon 구조 → H$\alpha$, EUV, X-ray에서 모두 관측

**Two-loop (상호작용) 모델 / Two-loop (interaction) model:**
평행하지 않은 두 루프가 만나서 재결합. 이 경우 기하학이 닫혀(closed) 있어 ejecta가 행성간 공간으로 전파되지 않음 → "compact flare"에 적합.

Two non-parallel loops meet and reconnect. In this case the geometry is closed, so ejecta don't propagate to interplanetary space → suitable for "compact flares."

**핵심 결론**: 1-loop과 2-loop 시나리오 모두 관측적 증거가 있으며, 같은 플레어에서 둘 다 동시에 일어날 수 있다. 자기 토폴로지는 dipolar, tripolar, quadrupolar, 또는 nullpoint geometry 등으로 다양하게 분류된다.

**Key conclusion**: Both 1-loop and 2-loop scenarios have observational evidence, and both can occur simultaneously in the same flare. Magnetic topologies are classified as dipolar, tripolar, quadrupolar, or nullpoint geometries.

#### 3.2–3.3 Coronal HXR Sources와 ITTT 모델

**Coronal hard X-ray source의 특성:**
- 일반적으로 soft X-ray 루프의 꼭대기(loop-top) 위에 위치 / Generally located above the soft X-ray loop-top
- 대부분 thermal source와 공간적으로 일치 (Masuda, 1994의 above-the-looptop source는 예외적) / Mostly cospatial with thermal source (Masuda's above-the-looptop source is exceptional)
- 정지해 있지 않음 — 약 1000 km/s의 상향 운동 보고 / Not always stationary — upward motions of ~1000 km/s reported
- Pre-flare 단계에서도 HXR 방출 가능 / Can emit HXR even in pre-flare phase

**ITTT (Intermediate Thin-Thick Target) 모델** (Wheatland & Melrose, 1995):
coronal source가 에너지에 따라 intermediate target 역할을 함:
- 저에너지 전자 → 코로나에서 thick target (에너지를 모두 잃음)
- 고에너지 전자 → 코로나를 통과하여 채층에서 thick target
- 임계 에너지 $E_c$ 경계

The coronal source acts as an intermediate target depending on energy: low-energy electrons stop in the corona (thick target), high-energy electrons pass through to the chromosphere (thick target there), with a critical energy $E_c$ boundary.

Battaglia & Benz (2007)의 RHESSI 데이터 검증: 단순 ITTT 모델만으로는 coronal과 footpoint source의 비열적 스펙트럼 관계를 완전히 설명할 수 없음. 전기장에 의한 non-collisional energy loss나 wave turbulence에 의한 가속도 필요.

RHESSI data verification by Battaglia & Benz (2007): Simple ITTT alone cannot fully explain the non-thermal spectral relationship between coronal and footpoint sources. Non-collisional energy loss by electric fields or acceleration by wave turbulence also needed.

#### 3.4 입자 가속 위치 / Location of Particle Acceleration

가속된 입자는 자기장을 따라 전파하면서 다양한 전파 방출을 생성하여 가속 환경의 기하학을 추적할 수 있다:

Accelerated particles propagate along magnetic fields, generating various radio emissions that trace the geometry of the acceleration environment:

- **Gyrosynchrotron** (센티미터파): 비상대론적~준상대론적 전자의 비간섭 방출. 루프 전체를 추적 / Incoherent emission from mildly relativistic electrons. Traces the entire loop
- **Type III radio burst** (미터~데시미터파): 전자빔이 코로나를 따라 전파할 때 플라즈마 주파수에서 방출. 시간에 따른 주파수 drift로 전자빔의 경로 추적 가능 / Emitted at plasma frequency as electron beams propagate through corona. Frequency drift traces electron beam path
- **Narrowband spikes** (데시미터파): 가속 영역 근처에서 직접 방출. Type III burst보다 약간 높은 주파수에서 시작 → 가속 영역이 Type III 출발점 근처임을 시사 / Direct emission near acceleration region. Start at slightly higher frequency than Type III → acceleration region near Type III departure point

가속 영역의 고도: HXR, 입자 관측, 간섭 전파 관측 결과를 종합하면 약 6000–90,000 km로 추정되며, 이는 서로 다른 가속 과정을 시사한다.

Acceleration region altitude: combining HXR, particle events, and coherent radio observations, estimated at ~6,000–90,000 km, suggesting different acceleration processes.

#### 3.5 Energetic Ions / 에너지 이온

감마선(0.8–20 MeV)은 가속된 이온이 원자핵과 충돌하여 방출. 핵심 관측:

Gamma-rays (0.8–20 MeV) are emitted by accelerated ions colliding with nuclei. Key observations:

- **2.223 MeV 중수소 포획선**: 가장 강한 감마선 라인. 가속 양성자(10 MeV 이상)가 채층 핵과 충돌 → 중성자 생성 → 중성자가 열화(thermalize) 후 수소에 포획되어 중수소 형성 시 방출 / Strongest gamma-ray line. Accelerated protons (>10 MeV) collide with chromospheric nuclei → produce neutrons → neutrons thermalize and are captured by hydrogen to form deuterium, emitting this line
- **RHESSI의 놀라운 발견**: 2.223 MeV 선(이온 가속 추적)의 footpoint와 non-thermal continuum(전자 가속 추적)의 footpoint가 **일치하지 않음**! → 양성자와 전자가 서로 다르게 가속되거나, 서로 다른 루프에서 기원함 / RHESSI's surprising discovery: footpoints of 2.223 MeV line (ion acceleration tracer) and non-thermal continuum (electron acceleration tracer) **do not coincide**! → Protons and electrons are accelerated differently or originate from different loops

이 발견은 표준 모델에 대한 근본적 도전이다. 단일 가속 메커니즘이 전자와 이온을 동시에 설명해야 하는데, 공간적 분리는 이것이 간단하지 않음을 보여준다.

This discovery is a fundamental challenge to the standard model. A single acceleration mechanism should explain both electrons and ions, but spatial separation shows this is not straightforward.

---

### 4. Energy Budget — 에너지 수지

#### 4.1 비열적 전자 에너지 / Non-thermal Electron Energy

$$E_{\text{kin}} = \int_{\varepsilon_{\min}}^{\varepsilon_{\max}} F(\varepsilon) \, \varepsilon \, d\varepsilon$$

Power-law 분포 $F(\varepsilon) \propto \varepsilon^{-\delta}$에서 $\delta > 2$이므로, $E_{\text{kin}}$은 저에너지 컷오프 $\varepsilon_{\min}$에 강하게 의존한다. 이로 인해 비열적 전자 에너지의 정확한 측정은 어렵다.

For a power-law distribution with $\delta > 2$, $E_{\text{kin}}$ depends strongly on the low-energy cutoff $\varepsilon_{\min}$, making accurate measurement of non-thermal electron energy difficult.

RHESSI의 1 keV 스펙트럼 분해능으로 ~10 keV까지 전자 에너지 분포 재구성이 가능해졌으나, albedo effect(채층에서의 X-ray 반사), free-bound emission, pulse pile-up 등의 효과가 저에너지 컷오프 측정을 왜곡한다. 현재 보고된 20–40 keV의 low-energy turnover는 미확인 상한값이다.

RHESSI's 1 keV spectral resolution enabled reconstruction of electron energy distributions down to ~10 keV, but albedo effect, free-bound emission, and pulse pile-up distort low-energy cutoff measurements. The currently reported 20–40 keV low-energy turnover is an unconfirmed upper limit.

#### 4.2 열 에너지 / Thermal Energy

$$E_{\text{th}} = \frac{3}{2} \sum_{\alpha} \int n_\alpha k_B T_\alpha \, dV \approx 3 k_B T \sqrt{MV}$$

여기서 $M$은 emission measure, $V$는 부피. 열적 플라즈마는 대부분 증발된 채층 물질로 구성. 코로나의 emission measure는 impulsive phase 동안 크게 증가한다.

Where $M$ is the emission measure and $V$ is volume. Thermal plasma is mostly composed of evaporated chromospheric material. Coronal emission measure increases greatly during the impulsive phase.

#### 에너지 수지 종합 / Energy Budget Summary (Table 1)

| 에너지 형태 / Energy Mode | M-class flare (erg) | X-class flare (erg) |
|---|---|---|
| Non-thermal electrons / 비열적 전자 | $10^{30}$ | $10^{31}$–$10^{32}$ |
| Non-thermal ions / 비열적 이온 | — | $< 4 \times 10^{31}$ |
| Thermal hot plasma / 열적 고온 플라즈마 | $10^{30}$ | $10^{31}$ |
| Total radiated / 총 복사 | — | $\sim 10^{32}$ |
| Kinetic CME / CME 운동 에너지 | — | $\sim 10^{32}$ |

**핵심 발견**: $T > 10$ MK 플레어에서 비열적 전자 에너지가 열 에너지보다 1–10배 크다! 이는 가열이 에너지 손실 과정이 아님을 의미한다 — 코로나 온도까지의 가열은 에너지를 보존하는 과정이다.

**Key finding**: In flares with $T > 10$ MK, non-thermal electron energy is 1–10 times larger than thermal energy! This means heating is not an energy-loss process — heating to coronal temperatures is an energy-conserving process.

총 플레어 에너지 = 백색광 + 적외선이 77%를 차지하고, UV + SXR은 23%에 불과하다. 즉, soft X-ray에서 측정하는 에너지는 전체의 일부일 뿐이다.

Total flare energy = white light + infrared accounts for 77%, while UV + SXR is only 23%. The energy measured in soft X-rays is only a fraction of the total.

#### 4.3–4.4 파동 에너지와 기타 에너지 / Wave Energy and Other Energies

자기 재결합은 자기 에너지를 열, 비열적 입자, 파동, 운동의 형태로 방출한다. MHD 파동은 관측되었으나 (EUV와 X-ray에서의 코로나 루프 진동), 에너지는 전체 플레어 에너지 방출의 $\sim 10^{-6}$에 불과하여 코로나 가열에 크게 기여하지 않는다.

Magnetic reconnection releases energy as heat, non-thermal particles, waves, and motion. MHD waves have been observed (coronal loop oscillations in EUV and X-rays), but their energy is only ~$10^{-6}$ of total flare energy release, not significantly contributing to coronal heating.

이온의 비열적 에너지는 전자와 비슷한 크기 ($\sim 10^{31}$ erg for X-class).

Non-thermal energy in ions is comparable to that in electrons (~$10^{31}$ erg for X-class).

#### 4.5 코로나 에너지 입력: 나노플레어 / Energy Input to Corona: Nanoflares

정상 플레어 외에 매우 작은 플레어-유사 현상(nanoflare)이 quiet corona에서 빈번하게 발생한다:

Beyond regular flares, very small flare-like events (nanoflares) frequently occur in the quiet corona:

- 가장 큰 나노플레어: $\sim 10^{26}$ erg / Largest nanoflares: ~$10^{26}$ erg
- 가장 작은 관측 가능 이벤트 (TRACE): $\sim 10^{23}$ erg / Smallest observable events (TRACE): ~$10^{23}$ erg
- 발생 빈도: 태양 전체에서 $> 10^{24}$ erg 이벤트가 초당 ~300개 / Occurrence rate: ~300 events with $> 10^{24}$ erg per second over the whole Sun
- 관측된 나노플레어의 총 에너지 = 관측된 코로나 영역 복사 에너지의 약 12% / Total energy of observed nanoflares ≈ 12% of radiated energy of observed coronal area

결론: 나노플레어가 코로나 가열에 기여할 가능성은 있지만, 정량적 평가는 아직 불가능하다. SXR이나 EUV 방출만으로는 전체 에너지 입력을 측정할 수 없기 때문이다.

Conclusion: Nanoflares may contribute to coronal heating, but quantitative assessment is not yet possible because SXR or EUV emission alone cannot measure total energy input.

---

### 5. Signatures of Energy Release — 에너지 방출의 시그니처

#### 5.1 Coronal HXR 시그니처 / Coronal Hard X-ray Signatures

Coronal source의 운동학적 특성은 에너지 방출 과정의 시그니처를 제공한다:

The kinematic properties of coronal sources provide signatures of the energy release process:

- Centroid 위치가 에너지가 높을수록 위로 이동 → 재결합이 높은 고도에서 발생함을 시사 / Centroid position moves upward with increasing energy → suggests reconnection occurs at high altitude
- Elongated tongue 구조가 loop-top에서 위로 뻗음 / Elongated tongue structures stretch upward from loop-top
- Source 고도는 HXR 시작부터 감소하다가 peak time에 상승 → 재결합 X-point의 위치 변화와 일치 / Source altitude decreases from HXR start, then rises at peak time → consistent with change in reconnection X-point location

#### 5.2 Soft-Hard-Soft (SHS) 행동 / Soft-Hard-Soft Behavior

대다수 플레어의 비열적 X-ray 스펙트럼은 시간에 따라 soft → hard → soft로 변한다:

The non-thermal X-ray spectrum of most flares evolves soft → hard → soft over time:

- 초기(rise): 스펙트럼이 steep (soft) / Initial rise: spectrum is steep (soft)
- 최대(peak): 스펙트럼이 flat (hard) — 가장 효율적 가속 / Peak: spectrum is flat (hard) — most efficient acceleration
- 감쇠(decay): 다시 steep (soft) / Decay: steep again (soft)

**정량적 관계 (Grigis & Benz, 2004):**

$$\gamma = A \cdot F(E_0)^{-\alpha}$$

여기서 $E_0 = 35$ keV, rise phase에서 $\alpha = 0.121 \pm 0.009$, decay phase에서 $\alpha = 0.172 \pm 0.012$.

이 관계는 가속 이론에 대한 매우 엄격한 제약 조건이다: 높은 flux일수록 더 flat한 스펙트럼을 만드는 가속 과정이 필요하다. 많은 가속 이론이 이 행동을 보이지만, 정량적으로 맞추기는 어렵다.

This relation is a very stringent constraint on acceleration theories: an acceleration process that produces a harder spectrum at higher flux is needed. Many theories show this behavior, but matching it quantitatively is difficult.

#### 5.3 전파 방출 / Radio Emissions from the Acceleration Region

플레어 관련 전파 방출의 분류:

Classification of flare-related radio emissions:

| 유형 / Type | 주파수 범위 / Frequency | 메커니즘 / Mechanism | 의미 / Significance |
|---|---|---|---|
| Gyrosynchrotron | $> 1$ GHz (cm–mm) | 비간섭, 준상대론적 전자 / Incoherent, mildly relativistic electrons | 자기장 내 전자 분포 추적 / Traces electron distribution in magnetic field |
| Type III | metric–decimetric | 간섭, 전자빔의 플라즈마 방출 / Coherent, plasma emission by electron beam | 가속 영역 위치와 전자빔 경로 추적 / Traces acceleration region and beam path |
| Narrowband spikes | decimetric (800–2000 MHz) | 간섭, 가속 영역 근처 직접 방출 / Coherent, direct emission near acceleration | 에너지 방출 위치의 가장 직접적 추적자 / Most direct tracer of energy release location |
| Type IV | metric–decimetric | 다양 (plasmoid, trapped electrons) / Various | Late-phase 가속과 관련 / Related to late-phase acceleration |

**중요한 제약**: C5.0 이상 대규모 플레어의 15%에서 간섭 전파 방출이 없음 (electron beam emitting at very high altitude 제외). 이는 가속이 "gentle"하여 Maxwellian에서 크게 벗어나지 않음을 시사하며, Transit-Time Damping을 지지한다.

**Important constraint**: 15% of flares > C5.0 show no coherent radio emission (except electron beams at very high altitude). This suggests acceleration is "gentle," not departing significantly from Maxwellian, supporting Transit-Time Damping.

---

### 6. Acceleration Processes — 가속 과정

자기 에너지의 운동 에너지로의 변환은 관측적으로 두 단계로 나뉜다:

The conversion of magnetic to kinetic energy is observationally separated into two phases:

1. **1차 (bulk energization)**: Impulsive phase에서 코로나 열 에너지(~0.1 keV)가 2자릿수 이상 증가 → 1초 이내 / In impulsive phase, coronal thermal energy (~0.1 keV) increases by over two orders of magnitude within less than 1 second
2. **2차**: 충격파나 CME에 의해 발생 — 태양 고에너지 입자(SEP) 생성에 더 중요 / Initiated by shocks or CMEs — more important for solar energetic particle production

**가속의 시간 제약**: 입자 가속이 효율적이려면 가속 시간 $<$ 충돌 시간 $\tau_{\text{coll}}$이어야 한다:

$$\tau_{\text{coll}} = 3.1 \times 10^{-20} \frac{v_T^3}{n_e} \approx 0.31 \left(\frac{v_T}{10^{10} \text{ cm/s}}\right)^3 \left(\frac{10^{11} \text{ cm}^{-3}}{n_e}\right) \text{ s}$$

코로나 밀도 $n_e \sim 10^{10}$–$10^{12}$ cm$^{-3}$에서 충돌 시간은 $\sim 1$초 이하 → 가속 가능.

At coronal densities $n_e \sim 10^{10}$–$10^{12}$ cm$^{-3}$, collision time is $\sim 1$ second or less → acceleration is feasible.

#### 6.1 세 가지 가속 이론 / Three Acceleration Theories

| 메커니즘 / Mechanism | 원리 / Principle | 장점 / Advantages | 문제점 / Problems |
|---|---|---|---|
| **Stochastic (TTD)** | MHD 파동의 자기장 성분이 입자를 운동량 공간에서 확산시킴 (Fokker–Planck). Cerenkov 공명 근처 입자가 반사 / Magnetic component of MHD waves diffuses particles in momentum space. Particles near Cerenkov resonance are mirrored | "Gentle" 가속 → 간섭 전파 방출 부재와 일치. Soft-hard-soft 행동 재현 가능 / "Gentle" acceleration consistent with lack of coherent radio emission. Can reproduce soft-hard-soft behavior | 직접 관측 증거 없음. 아직 가설 단계 / No direct observational evidence. Still hypothetical |
| **DC electric field** | 전류 시트 내 전기장에 의한 직접 가속 / Direct acceleration by electric field in current sheets | 재결합과 자연스럽게 연결 / Naturally connected to reconnection | 가속 가능한 전자 수가 제한적. 대규모 정상 전기장은 속도 공간 불안정성 유발 → 전파 방출 예상 (항상 관측되지는 않음) / Number of acceleratable electrons limited. Large-scale stationary fields would cause velocity-space instabilities → expected radio emission (not always observed) |
| **Shock acceleration** | 행성간 공간의 충격파에서의 1차/2차 Fermi 가속 / First/second order Fermi acceleration at shocks | 높은 에너지(100 MeV 이온 $< 1$초)로 빠른 가속 가능 / Can achieve high energies quickly (100 MeV ions in $< 1$ s) | 플레어와 직접 관련된 충격파 관측이 드물고 논쟁 중 / Shock observations directly related to flares are rare and disputed |

#### 6.2 관측과의 비교 / Comparing Theories with Observations

관측은 세 이론 중 어느 하나를 확실히 확인하기보다는 배제하는 데 더 효과적이다:

Observations are more effective at excluding theories than confirming any one:

1. **플라즈마 주파수 근처 고주파 파동**: 확률적 가속의 driver로 배제 가능 — 이들은 decimeter radio wave로 결합하여 모든 플레어에서 관측되어야 하지만 그렇지 않음 / High-frequency waves near plasma frequency can be excluded as stochastic acceleration driver — they would couple to decimeter radio waves and should be present in every flare
2. **TTD (Transit-Time Damping)**: 가정하는 turbulence 주파수가 플라즈마 주파수보다 훨씬 낮아 전파 방출을 유발하지 않음 → 15%의 radio-quiet 플레어와 일치 / TTD's postulated turbulence frequency is far below plasma frequency, not causing radio emission → consistent with 15% radio-quiet flares
3. **전류 시트 내 DC 전기장**: 가속 가능한 입자 수 부족. 대규모 정상 전기장은 속도 공간 왜곡 → 전파 방출 예상이나 항상 관측되지 않음 / DC field in current sheets: insufficient number of acceleratable particles. Large-scale fields would distort velocity space → expected radio emission not always observed
4. **충격파**: 플레어 직접 관련 증거가 부족하고 논쟁 중이나, 재결합 outflow의 bow shock 등은 가능 / Shock: insufficient direct evidence for flares, disputed, but bow shock of reconnection outflow is possible

**최종 결론**: TTD가 현재 관측과 가장 잘 부합하지만, 직접적 확인은 아직 없다. 저주파·고진폭 turbulence에서 기원하는 요동 전기장에 의한 가속도 대안으로 존재한다.

**Final conclusion**: TTD best matches current observations, but direct confirmation is still lacking. Acceleration by fluctuating electric fields from low-frequency, high-amplitude turbulence (kinetic Alfven waves) is also a viable alternative.

---

### 7. Conclusions — 결론

Benz의 핵심 결론들:

1. 플레어는 코로나의 연속적 역학의 극단적 현상으로 보아야 한다. 나노플레어도 유사한 과정을 보인다 / Flares should be viewed as extreme phenomena of continuous coronal dynamics. Nanoflares show similar processes
2. 자기 재결합은 보편적으로 받아들여지지만, 어떻게 에너지가 비열적 입자로 효율적으로 변환되는지는 여전히 미해결 / Magnetic reconnection is universally accepted, but how energy is efficiently converted to non-thermal particles remains unsolved
3. 각 플레어는 개별적 특성을 가지며, 관측이 단순한 시나리오대로 상관되지 않음 → 코로나 역학의 자유도가 많음 / Each flare has individual features, and observations don't correlate as predicted by simple scenarios → coronal dynamics has many degrees of freedom
4. 코로나 20,000 km 이하 (transition region 포함)는 아직 제대로 탐사되지 않은 영역이며, 이곳에서 대부분의 코로나 가열이 일어날 가능성이 있다 / The corona below 20,000 km (including transition region) is not yet well explored, and most coronal heating may occur there
5. 향후 다파장 동시 영상 관측이 플레어 물리학의 이해를 크게 증진시킬 것이다 / Future simultaneous multi-wavelength imaging will greatly advance understanding of flare physics

---

## 핵심 시사점 / Key Takeaways

1. **플레어의 에너지 원천은 자유 자기 에너지이다**: 광구 아래에서 기원한 자기장의 잠재 에너지(potential field)를 초과하는 부분 — 코로나 전류에 의해 유지됨 — 이 재결합으로 방출된다. 자기장의 전단(shear)과 새로운 플럭스의 부상(emerging flux)이 에너지 축적의 핵심이다. / **Free magnetic energy is the source**: The energy stored in excess of the potential field — maintained by coronal currents — is released via reconnection. Magnetic shear and emerging flux are key to energy accumulation.

2. **표준 모델(CSHKP)은 유용하지만 불완전하다**: 재결합→입자 가속→채층 가열→증발의 시나리오는 많은 관측을 설명하지만, footpoint 없는 플레어, Neupert effect 위반, preheating 현상 등은 표준 모델의 수정이 필요함을 보여준다. / **Standard model (CSHKP) is useful but incomplete**: The reconnection→acceleration→heating→evaporation scenario explains many observations, but flares without footpoints, Neupert effect violations, and preheating require modifications.

3. **비열적 전자의 에너지가 열 에너지보다 크다**: 이것은 가열이 에너지 "손실"이 아님을 의미한다. 전자의 비열적 에너지에서 열 에너지로의 변환은 에너지를 보존하는 과정이며, 원래의 자기 에너지에서 비열적 입자로의 변환이 주요 에너지 채널이다. / **Non-thermal electron energy exceeds thermal energy**: This means heating is not energy "loss." The conversion from non-thermal to thermal is energy-conserving; the primary channel is from magnetic to non-thermal particles.

4. **양성자와 전자는 서로 다르게 가속된다**: RHESSI가 발견한 감마선(이온 추적)과 HXR(전자 추적) footpoint의 공간적 불일치는 가속 메커니즘에 대한 근본적 재고를 요구한다. 단일 가속 메커니즘이 둘 다를 설명할 수 없을 가능성이 높다. / **Protons and electrons are accelerated differently**: RHESSI's discovery of spatial mismatch between gamma-ray (ion tracer) and HXR (electron tracer) footpoints demands fundamental rethinking of acceleration mechanisms.

5. **Transit-Time Damping이 가장 유력한 가속 메커니즘이다**: "Gentle" 가속 특성이 15%의 radio-quiet 플레어와 일치하며, soft-hard-soft 행동의 정량적 재현이 가능하다. 그러나 아직 직접적 관측 확인은 없다. / **Transit-Time Damping is the most favored acceleration mechanism**: Its "gentle" characteristic matches 15% radio-quiet flares, and it can quantitatively reproduce soft-hard-soft behavior. But direct observational confirmation is still lacking.

6. **에너지 수지에서 백색광과 적외선이 77%를 차지한다**: SXR만으로 플레어 에너지를 측정하면 전체의 ~23%만 보는 것이다. 총 플레어 에너지는 총 태양 복사조도(TSI) 증가로 측정해야 가장 정확하다. / **White light and infrared account for 77% of total flare energy**: Measuring flare energy from SXR alone captures only ~23%. Total flare energy is most accurately measured from total solar irradiance (TSI) enhancement.

7. **나노플레어는 코로나 가열의 잠재적 후보이나 정량화가 불가능하다**: quiet corona에서 초당 수백 개의 나노플레어가 관측되지만, 복사 에너지만으로는 총 에너지 입력을 평가할 수 없다. 파동, 운동, 직접 가열 등의 에너지 배분이 정규 플레어에서조차 불확실하기 때문이다. / **Nanoflares are potential candidates for coronal heating but cannot be quantified**: Hundreds of nanoflares per second are observed in quiet corona, but total energy input cannot be assessed from radiated energy alone, since energy partition into waves, motion, and direct heating is uncertain even for regular flares.

8. **플레어의 관측적 복잡성은 코로나의 높은 자유도를 반영한다**: 동일한 물리 과정(재결합)이 다양한 자기 토폴로지(1-loop, 2-loop, dipolar, tripolar, quadrupolar)에서 발생하므로, 각 플레어는 고유한 관측적 특성을 가진다. 이것이 단순한 "표준 모델"이 모든 플레어를 설명할 수 없는 근본적 이유이다. / **Observational complexity reflects the high degrees of freedom of the corona**: The same physical process (reconnection) occurs in various magnetic topologies, so each flare has unique observational characteristics. This is the fundamental reason a simple "standard model" cannot explain all flares.

---

## 수학적 요약 / Mathematical Summary

### 핵심 물리 수식 / Key Physics Equations

**1. Neupert Effect:**

$$F_{SXR}(t) \propto \int_{t_0}^{t} F_{HXR}(t') \, dt' \quad \Longleftrightarrow \quad \frac{d}{dt} F_{SXR}(t) \propto F_{HXR}(t)$$

**2. Non-thermal electron energy (power-law):**

$$E_{\text{kin}} = \int_{\varepsilon_{\min}}^{\varepsilon_{\max}} F(\varepsilon) \, \varepsilon \, d\varepsilon, \quad F(\varepsilon) \propto \varepsilon^{-\delta}, \quad \delta > 2$$

**3. Thermal energy:**

$$E_{\text{th}} = 3 k_B T \sqrt{MV}$$

**4. Thick target photon spectrum:**

비열적 전자의 power-law index $\delta$ → bremsstrahlung photon spectral index $\gamma = \delta - 1$
Non-thermal electron power-law index $\delta$ → bremsstrahlung photon spectral index $\gamma = \delta - 1$

**5. Plasma frequency (Type III 전파 방출):**

$$\omega_p = \sqrt{\frac{4\pi e^2 n_e}{m_e}}$$

**6. Soft-hard-soft quantitative relation:**

$$\gamma = A \cdot F(E_0)^{-\alpha}, \quad \alpha \approx 0.12 \text{ (rise)}, \quad 0.17 \text{ (decay)}$$

**7. Collision time (가속 효율 제약):**

$$\tau_{\text{coll}} = 3.1 \times 10^{-20} \frac{v_T^3}{n_e}$$

**8. Fokker–Planck (stochastic acceleration):**

$$\frac{\partial f(\mathbf{p})}{\partial t} = \left( \frac{1}{2} \sum_{i,j} \frac{\partial}{\partial p_i} \frac{\partial}{\partial p_j} D_{ij} - \sum_i \frac{\partial}{\partial p_i} F_i \right) f(\mathbf{p})$$

---

## 역사적 맥락의 타임라인 / Paper in the Arc of History

```
1859  Carrington & Hodgson — 최초 백색광 플레어 관측 / First white-light flare
  |
1908  Hale — 태양 흑점 자기장 발견 / Sunspot magnetic fields
  |
1942  Hey — 군사 레이더에서 태양 전파 방출 발견 / Solar radio emission from radar
  |
1958  Peterson & Winckler — 최초 hard X-ray 관측 / First hard X-ray observation
  |
1964  Carmichael — CSHKP 모델 시작 / Beginning of CSHKP model
  |
1968  Neupert — Neupert effect 발견 / Discovery of Neupert effect
  |
1971  Brown — Thick target 모델 / Thick target model
  |
1974  Hirayama — Evaporating flare model
  |
1976  Kopp & Pneuman — CSHKP 모델 완성 / CSHKP model completed
  |
1981  Hoyng et al. — HXR footpoint 최초 발견 / First HXR footpoint discovery
  |
1991  Yohkoh 발사 — SXR/HXR 영상 분광 시대 시작 / Imaging spectroscopy era
  |
1994  Masuda et al. — loop-top HXR source 발견 / Above-the-looptop source
  |
1995  SOHO 발사 — EUV/coronagraph 관측 / EUV/coronagraph observations
  |
1998  TRACE 발사 — 고해상도 EUV 영상 / High-resolution EUV imaging
  |
2002  RHESSI 발사 — HXR 영상 분광의 혁명 / HXR imaging spectroscopy revolution
  |
  ★ 2008  Benz — "Flare Observations" LRSP 리뷰 ★
  |
2006  Hinode 발사 — 광구 자기장 초고해상도 / Ultra-high-res photospheric fields
  |
2010  SDO 발사 — 다파장 전면 태양 관측 / Multi-wavelength full-disk observations
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| 논문 / Paper | 연결 / Connection |
|---|---|
| LRSP #6 Longcope (2005) | 자기 토폴로지와 separatrix — 플레어 위치의 자기 구조적 이해 / Magnetic topology and separatrix — structural understanding of flare locations |
| LRSP #8 Marsch (2006) | 태양풍 입자의 운동론적 물리학 — 파동-입자 상호작용의 다른 맥락 / Kinetic physics of solar wind particles — wave-particle interaction in different context |
| LRSP #9 Schwenn (2006) | CME 관측 — 플레어-CME 관계의 관측적 증거 / CME observations — observational evidence for flare-CME relationship |
| LRSP #10 Pulkkinen (2007) | 우주 기상에서의 플레어 영향 — SEP와 지자기 폭풍 / Flare effects in space weather — SEP and geomagnetic storms |
| LRSP #3 Nakariakov & Verwichte (2005) | 코로나 진동 — 플레어의 MHD 파동과 에너지 전달 / Coronal oscillations — MHD waves and energy transport in flares |
| LRSP #5 Gizon & Birch (2005) | 일진학 — 플레어 에너지의 내부 기원 이해 / Helioseismology — understanding internal origin of flare energy |
| LRSP #14 Hall (2008, next) | 항성 채층 활동 — 태양 플레어의 항성 맥락 확장 / Stellar chromospheric activity — extending solar flares to stellar context |

---

## References / 참고문헌

- Benz, A.O., "Flare Observations", *Living Rev. Solar Phys.*, **5**, 1 (2008). DOI: 10.12942/lrsp-2008-1
- Brown, J.C., "The Deduction of Energy Spectra of Non-Thermal Electrons in Flares from the Observed Dynamic Spectra of Hard X-Ray Bursts", *Solar Phys.*, **18**, 489 (1971).
- Carmichael, H., "A Process for Flares", *The Physics of Solar Flares*, NASA, p. 451 (1964).
- Grigis, P.C., Benz, A.O., "The spectral evolution of impulsive solar X-ray flares", *Astron. Astrophys.*, **426**, 1093 (2004).
- Hirayama, T., "Theoretical Model of Flares and Prominences", *Solar Phys.*, **34**, 323 (1974).
- Kopp, R.A., Pneuman, G.W., "Magnetic reconnection in the corona and the loop prominence phenomenon", *Solar Phys.*, **50**, 85 (1976).
- Masuda, S., et al., "A Loop-Top Hard X-Ray Source in a Compact Solar Flare as Evidence for Magnetic Reconnection", *Nature*, **371**, 495 (1994).
- Neupert, W.M., "Comparison of Solar X-Ray Line Emission with Microwave Emission during Flares", *Astrophys. J. Lett.*, **153**, L59 (1968).
- Sturrock, P.A., "Model of the High-Energy Phase of Solar Flares", *Nature*, **211**, 695 (1966).
- Wheatland, M.S., Melrose, D.B., "Interpreting YOHKOH Hard and Soft X-Ray Flare Observations", *Solar Phys.*, **158**, 283 (1995).
