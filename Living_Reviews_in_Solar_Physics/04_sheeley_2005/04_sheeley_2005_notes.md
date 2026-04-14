---
title: "Surface Evolution of the Sun's Magnetic Field: A Historical Review of the Flux-Transport Mechanism"
authors:
  - Neil R. Sheeley Jr.
year: 2005
journal: "Living Reviews in Solar Physics, 2, 5"
topic: Living Reviews in Solar Physics
tags:
  - flux-transport
  - solar magnetic field
  - supergranular diffusion
  - meridional flow
  - differential rotation
  - polar field reversal
  - solar cycle
  - magneto-kinematic model
  - coronal holes
  - solar dynamo
status: completed
date_started: 2026-04-08
date_completed: 2026-04-08
---

# Surface Evolution of the Sun's Magnetic Field: A Historical Review of the Flux-Transport Mechanism
## 태양 자기장의 표면 진화: Flux-Transport 메커니즘의 역사적 리뷰

---

## 핵심 기여 / Core Contribution

이 논문은 태양 표면 자기장의 진화를 설명하는 flux-transport 메커니즘의 40년 역사(1963-2005)를 1인칭 시점에서 서술한 역사적 리뷰이다. 1963년 Robert Leighton이 supergranulation에 의한 자기 플럭스의 random-walk diffusion과 Joy의 법칙에 의한 bipolar region의 기울어짐을 결합하여 극 자기장 역전을 설명한 아이디어에서 시작하여, 1980년대 NRL(Naval Research Laboratory)에서의 수치 시뮬레이션 개발, meridional flow의 역할 발견, 그리고 2000년대의 다주기 시뮬레이션에 이르기까지의 발전 과정을 추적한다. 핵심적으로, 태양 표면 자기장의 대규모 패턴(준강체 회전, 극 자기장 역전, 코로나 홀 진화)이 세 가지 표면 과정 -- supergranular diffusion ($\eta \approx 500$-$600$ km$^2$ s$^{-1}$), differential rotation, 그리고 poleward meridional flow ($v \approx 10$-$25$ m s$^{-1}$) -- 만으로 정량적으로 재현될 수 있음을 보여준 연구 프로그램의 기록이다.

This paper is a first-person historical review tracing 40 years (1963-2005) of the flux-transport mechanism that explains the evolution of the Sun's surface magnetic field. Starting from Robert Leighton's 1963 idea that random-walk diffusion by supergranulation combined with the Joy's law tilt of bipolar magnetic regions could reverse the Sun's polar fields, it traces the development through the 1980s numerical simulations at NRL, the discovery of meridional flow's role, and multi-cycle simulations of the 2000s. The central achievement documented is that the large-scale patterns of the solar surface magnetic field -- quasi-rigid rotation, polar field reversal, coronal hole evolution -- can be quantitatively reproduced by just three surface processes: supergranular diffusion ($\eta \approx 500$-$600$ km$^2$ s$^{-1}$), differential rotation, and poleward meridional flow ($v \approx 10$-$25$ m s$^{-1}$).

---

## 읽기 노트 / Reading Notes

### 1. 시작 (1963-1969) / The Beginning (1963-1969)

#### 1.1 Leighton의 핵심 아이디어 / Leighton's Key Idea

1963년 9월 16일, Robert Leighton은 IAU Symposium 22 참석 후 Sheeley에게 전화를 걸어 flux-transport 아이디어를 설명했다. Leighton의 통찰은 supergranulation이 자기 플럭스를 sunspot group으로부터 무작위 보행(random walk)으로 이동시킬 수 있으며, 이것이 궁극적으로 극 자기장을 역전시킬 수 있다는 것이었다. 이 아이디어는 태양 물리학에서 가장 중요한 개념적 도약 중 하나가 되었다.

On September 16, 1963, Robert Leighton called Sheeley after attending IAU Symposium 22 to describe his flux-transport idea. Leighton's insight was that supergranulation could transport magnetic flux from sunspot groups via random walk, ultimately reversing the polar fields. This idea became one of the most important conceptual leaps in solar physics.

#### 1.2 극 자기장 역전의 물리학 / Physics of Polar Field Reversal

극 자기장 역전의 물리학은 놀라울 정도로 우아하다. Bipolar magnetic region(쌍극자 자기 영역)은 Joy의 법칙에 따라 기울어져서 나타나는데, 선행 극성(leading polarity)은 적도 쪽에, 후행 극성(trailing polarity)은 극 쪽에 위치한다. Supergranular diffusion이 이 플럭스를 사방으로 퍼뜨리면, 적도 쪽으로 간 선행 극성 플럭스는 반대 반구의 선행 극성 플럭스와 상쇄된다. 반면 후행 극성 플럭스는 극 방향으로 이동하여 기존의 극 자기장을 상쇄하고, 결국 새로운 극성으로 대체한다.

The physics of polar field reversal is remarkably elegant. Bipolar magnetic regions emerge tilted according to Joy's law, with the leading polarity closer to the equator and the trailing polarity closer to the pole. Supergranular diffusion spreads this flux in all directions: the leading-polarity flux migrating equatorward cancels with leading-polarity flux from the opposite hemisphere, while the trailing-polarity flux migrates poleward, canceling the existing polar field and eventually replacing it with the new polarity.

각 doublet(쌍극자)의 극에 대한 순 기여는 다음과 같이 추정된다:

The net polar contribution per doublet is estimated as:

$$\text{Net contribution per doublet} \sim \frac{F \cdot \Delta\theta}{4\pi R^2}$$

여기서 $F$는 doublet의 플럭스 강도, $\Delta\theta$는 Joy의 법칙에 의한 기울기 각도, $R$은 태양 반경이다. 한 주기당 약 $10^3$개의 sunspot group이 나타나므로, 이 누적 효과가 극 자기장을 역전시키기에 충분하다는 것이 Leighton의 계산이었다.

Here, $F$ is the flux strength of the doublet, $\Delta\theta$ is the tilt angle from Joy's law, and $R$ is the solar radius. Leighton calculated that approximately $10^3$ sunspot groups per cycle is sufficient to reverse the polar fields through this cumulative effect.

#### 1.3 Sheeley의 관측적 확인 / Sheeley's Observational Confirmation

Sheeley는 1905-1964년 Mount Wilson Observatory 데이터를 분석하여 극 faculae(명점)이 sunspot minimum에 나타나고 sunspot maximum에서 역전됨을 확인했다. 이는 Leighton의 이론적 예측과 정확히 일치하는 관측적 증거였다.

Sheeley analyzed Mount Wilson Observatory data from 1905-1964, confirming that polar faculae appeared at sunspot minimum and reversed at sunspot maximum. This was observational evidence exactly matching Leighton's theoretical predictions.

#### 1.4 Leighton의 이론적 발전 / Leighton's Theoretical Development

- **Leighton (1964)**: 자기 플럭스의 supergranular diffusion과 differential rotation을 결합한 flux-transport 모델을 발표했다. 이것이 flux-transport 방정식의 첫 번째 공식적 정립이었다.
- **Leighton (1969)**: Butterfly diagram(나비 도표)을 재현하는 magneto-kinematic model을 개발했다. 중요한 점은 이 모델이 meridional flow **없이** 작동했다는 것인데, 그 대신 angular velocity의 음의 반경 기울기(negative radial gradient)를 필요로 했다. 그러나 이 가정은 나중에 helioseismology(태양진동학)에 의해 부정되었다.
- **Schatten et al. (1972)**: Mount Wilson magnetogram을 사용한 최초의 수치 시뮬레이션을 수행했다.

- **Leighton (1964)**: Published the flux-transport model combining supergranular diffusion with differential rotation. This was the first formal formulation of the flux-transport equation.
- **Leighton (1969)**: Developed a magneto-kinematic model reproducing the butterfly diagram. Critically, this model worked WITHOUT meridional flow, but required a negative radial gradient of angular velocity -- an assumption later shown incorrect by helioseismology.
- **Schatten et al. (1972)**: Performed the first numerical simulation using Mount Wilson magnetograms.

---

### 2. 1970년대 -- "암흑기" / The 1970s -- "Dark Ages"

#### 2.1 관측적 도전 / Observational Challenges

1970년대는 flux-transport 모델에 대한 심각한 의문이 제기된 시기였다. Mosher (1977)는 diffusion rate을 200-400 km$^2$ s$^{-1}$로 추정했는데, 이는 Leighton의 770-1540 km$^2$ s$^{-1}$보다 훨씬 낮았다. 또한 Mosher는 약 3 m/s의 meridional flow를 발견했는데, 이는 Leighton의 순수 diffusion 모델에 대한 "명백한 타격(apparent blow)"으로 여겨졌다.

The 1970s were a period when serious questions were raised about the flux-transport model. Mosher (1977) estimated the diffusion rate at only 200-400 km$^2$ s$^{-1}$, much lower than Leighton's 770-1540 km$^2$ s$^{-1}$. Mosher also found meridional flow of approximately 3 m/s, which was seen as an "apparent blow" to Leighton's pure diffusion model.

#### 2.2 지하 자기장 패러다임 / Subsurface Field Paradigm

이 시기에 태양 관측은 가상의 지하 자기장(hypothetical subsurface fields)의 관점에서 해석되었다. 광구 자기장의 준강체 회전(quasi-rigid rotation)은 지하에서의 회전에 기인한 것으로 여겨졌다. Sheeley는 이 시기를 "태양 물리학이 진정으로 이해의 '암흑기'에 들어섰다"고 표현했다. 이 패러다임은 10년 이상 지속되며 flux-transport 모델의 발전을 지연시켰다.

During this period, solar observations were interpreted in terms of hypothetical subsurface fields. The quasi-rigid rotation of photospheric fields was attributed to subsurface rotation. Sheeley described this as a time when "solar physics had truly entered the 'dark ages' of understanding." This paradigm persisted for over a decade, delaying the development of the flux-transport model.

---

### 3. 초기 시뮬레이션 (1980년대) / Early Simulations (1980s)

#### 3.1 NRL에서의 시뮬레이션 시작 / Simulation Begins at NRL

1981년 1월, NRL(Naval Research Laboratory)의 Jay Boris가 추측 대신 직접 시뮬레이션을 해보자고 제안했다. 이로 인해 Sheeley, Boris, DeVore가 NRL에서 flux-transport 코드를 개발하기 시작했다. 이것은 이론적 추측에서 수치적 검증으로의 패러다임 전환이었다.

In January 1981, Jay Boris at NRL suggested direct simulation instead of speculation. This led Sheeley, Boris, and DeVore to begin developing a flux-transport code at NRL. This represented a paradigm shift from theoretical speculation to numerical verification.

#### 3.2 관측 데이터 입력 / Observational Data Input

Kitt Peak magnetogram(1976-1981)에서 bipolar magnetic region을 측정하여 doublet list를 구성했다. Pole 강도 $\geq 0.1 \times 10^{21}$ Mx인 doublet만 포함시켰다. 이 관측된 source term을 직접 시뮬레이션에 입력함으로써, 모델의 예측력을 실제 데이터와 비교할 수 있게 되었다.

Bipolar magnetic regions were measured on Kitt Peak magnetograms (1976-1981) to construct a doublet list. Only doublets with pole strengths $\geq 0.1 \times 10^{21}$ Mx were included. By directly inputting these observed source terms into the simulation, the model's predictive power could be compared against actual data.

#### 3.3 주요 결과 / Key Results

- **Sheeley et al. (1983)**: 유효 diffusion rate을 $730 \pm 250$ km$^2$ s$^{-1}$로 결정했다. 이는 Leighton의 원래 추정치에 가까웠고, Mosher의 낮은 값이 과소추정이었음을 시사했다.
- 이후 연구에서 meridional flow를 포함하면 diffusion rate이 약 500-600 km$^2$ s$^{-1}$로 감소함을 발견했다. 즉, meridional flow가 일부 역할을 담당하면 diffusion만으로 설명해야 할 양이 줄어드는 것이다.
- **DeVore et al. (1985)**: 태양주기 21(cycle 21)의 대부분을 성공적으로 시뮬레이션했다.

- **Sheeley et al. (1983)**: Determined the effective diffusion rate as $730 \pm 250$ km$^2$ s$^{-1}$. This was close to Leighton's original estimate, suggesting Mosher's lower value was an underestimate.
- Subsequent studies found that including meridional flow reduced the required diffusion rate to approximately 500-600 km$^2$ s$^{-1}$. That is, when meridional flow handles part of the transport, the amount needing explanation by diffusion alone decreases.
- **DeVore et al. (1985)**: Successfully simulated most of solar cycle 21.

#### 3.4 핵심적 발견: 섹터 패턴의 기원 / Critical Finding: Origin of Sector Patterns

이 시뮬레이션에서 나온 가장 중요한 발견 중 하나는 sector pattern(섹터 패턴)이 flux-transport 매개변수의 세부사항에 둔감하다는 것이었다. 이는 대규모 자기장의 평균 패턴이 활동 영역(active region)에 뿌리를 두고 있으며, 지하의 원시 자기장(subsurface primordial fields)에서 기원하는 것이 아님을 의미했다. 이 결과는 1970년대의 지하 자기장 패러다임에 대한 직접적인 반증이었다.

One of the most important findings from these simulations was that the sector pattern was insensitive to the details of flux-transport parameters. This meant the mean-field pattern of the large-scale field was rooted in active regions, NOT in subsurface primordial fields. This result was a direct refutation of the 1970s subsurface field paradigm.

---

### 4. 계몽의 시대 (1980-90년대) / The Era of Enlightenment (1980s-90s)

#### 4.1 준강체 회전의 설명 -- "수영하는 오리" 비유 / Quasi-rigid Rotation Explained -- "Swimming Duck" Analogy

**Sheeley et al. (1987)**은 flux transport의 meridional 성분이 대규모 자기장 패턴의 준강체 회전을 일으킨다는 것을 보여주었다. 이를 설명하기 위해 유명한 "수영하는 오리(swimming duck)" 비유(Figure 2)를 사용했다:

**Sheeley et al. (1987)** showed that the meridional component of flux transport causes quasi-rigid rotation of large-scale field patterns. They used the famous "swimming duck" analogy (Figure 2) to explain this:

강물이 중앙에서 빠르고 가장자리에서 느리게 흐르는 것처럼(differential rotation), 오리가 강을 가로질러 수영하면(meridional flow) 오리의 궤적은 강물 속도에 관계없이 일정한 각속도로 회전하는 것처럼 보인다. 마찬가지로, 자기 플럭스가 적도에서 극으로 meridional flow에 의해 이동하면, 높은 위도에서의 느린 differential rotation이 중요해지면서 전체 패턴이 준강체적으로 회전하는 것처럼 보인다.

Like a river flowing faster in the center and slower at the edges (differential rotation), when a duck swims across the river (meridional flow), the duck's trajectory appears to rotate at a constant angular velocity regardless of the water speed. Similarly, when magnetic flux is transported from the equator to the pole by meridional flow, the slower differential rotation at higher latitudes becomes important, making the overall pattern appear to rotate quasi-rigidly.

이것은 핵심적인 통찰이었다: **준강체 회전은 지하 자기장이 아니라 표면 운동(diffusion + meridional flow)만으로 완전히 설명된다.**

This was a crucial insight: **quasi-rigid rotation is explained entirely by surface motions (diffusion + meridional flow), with no subsurface fields needed.**

#### 4.2 Meridional Flow의 확정 / Confirmation of Meridional Flow

**Wang et al. (1989a)**는 태양주기 21 동안의 poleward surge(극 방향 급등)를 재현하기 위해 증가된 eruption rate과 가속된 flow가 필요하다는 것을 발견했다. 이는 meridional flow가 실제로 존재함을 의미했고, 이 시점에서 연구팀은 meridional flow에 대한 "if present(만약 존재한다면)" 한정어를 공식적으로 삭제했다. 이것은 단순한 표현의 변화가 아니라, 과학적 확신의 전환점이었다.

**Wang et al. (1989a)** found that enhanced eruption rates and accelerated flows were needed to match poleward surges during cycle 21, implying meridional flow IS present. At this point, the "if present" qualifier for meridional flow was officially dropped. This was not merely a change in phrasing but a turning point in scientific confidence.

독립적으로, **Topka et al. (1982)**도 poleward Hα filament migration으로부터 같은 결론에 도달했다.

Independently, **Topka et al. (1982)** reached the same conclusion from poleward Hα filament migration.

#### 4.3 Flux-Transport Dynamo / Flux-Transport Dynamo

**Wang & Sheeley (1991)**는 지하 약 1 m/s의 return flow와 약 10 km$^2$ s$^{-1}$의 turbulent diffusion을 가진 flux-transport dynamo 모델을 개발하여 butterfly diagram을 재현했다. 이 모델에서 meridional flow는 표면에서는 극 방향으로, 지하(대류층 바닥)에서는 적도 방향으로 흐르는 순환 셀을 형성한다. 표면의 poleward flow는 후행 극성 플럭스를 극으로 운반하고, 지하의 equatorward return flow는 자기장을 적도 쪽으로 되가져와 새로운 주기의 활동 영역을 생성하는 데 기여한다.

**Wang & Sheeley (1991)** developed a flux-transport dynamo model with subsurface return flow of approximately 1 m/s and turbulent diffusion of approximately 10 km$^2$ s$^{-1}$, reproducing the butterfly diagram. In this model, meridional flow forms a circulation cell flowing poleward at the surface and equatorward at the base of the convection zone. The surface poleward flow carries trailing-polarity flux to the poles, while the subsurface equatorward return flow brings magnetic field back toward the equator to generate new-cycle active regions.

이 모델은 후에 **Choudhuri et al. (1995)**와 **Dikpati & Charbonneau (1999)**에 의해 더욱 발전되었다.

This model was later further developed by **Choudhuri et al. (1995)** and **Dikpati & Charbonneau (1999)**.

#### 4.4 코로나 및 태양풍으로의 확장 / Extension to Corona and Solar Wind

**Wang et al. (1988)**은 flux-transport 모델을 potential-field source-surface 모델과 결합하여 코로나 홀의 진화와 태양풍 속도를 예측했다. 이로써 flux-transport 모델은 단순히 광구 자기장을 설명하는 것을 넘어, 태양 대기와 태양풍의 구조까지 예측하는 포괄적인 도구가 되었다.

**Wang et al. (1988)** coupled the flux-transport model to a potential-field source-surface model to predict coronal hole evolution and solar wind speed. This extended the flux-transport model beyond merely explaining the photospheric field to become a comprehensive tool for predicting solar atmospheric and solar wind structure.

#### 4.5 Stenflo 논쟁과 해결 / The Stenflo Debate and Resolution

**Stenflo (1989a,b, 1992)**는 flux-transport 모델에 반대하는 주장을 펼쳤다. 그러나 discrete random walk 접근법이 이 불일치를 해결했다: 소규모 feature는 Snodgrass differential rate을 보여주고, 대규모 feature는 준강체 회전율을 보여준다는 것이다. 핵심은 관측하는 공간 스케일에 따라 회전율이 달라 보인다는 것이며, 이 두 가지 관측이 모두 같은 물리적 메커니즘(diffusion + meridional flow)에서 자연스럽게 나온다는 것이다.

**Stenflo (1989a,b, 1992)** argued against the flux-transport model. However, the discrete random walk approach resolved the discrepancy: small-scale features show the Snodgrass differential rate, while large-scale features show the quasi-rigid rate. The key point is that the observed rotation rate depends on the spatial scale being examined, and both observations emerge naturally from the same physical mechanism (diffusion + meridional flow).

---

### 5. 호주 학파 (1990-2000년대) / The Australian School (1990s-2000s)

#### 5.1 반대에서 수용으로 / From Opposition to Acceptance

**Wilson et al. (1990+)**은 처음에 flux-transport 모델이 작동할 수 없다고 주장했다. 그들은 표면 자기장이 지하의 toroidal field에 연결되어 있다고 보았다. 그러나 자체 시뮬레이션을 시작한 후, 그들의 강조점은 모델에 반대하는 것에서 모델을 활용하는 것으로 점차 전환되었다. 이 과정은 과학에서 패러다임 전환의 전형적인 사례를 보여준다.

**Wilson et al. (1990+)** initially argued the flux-transport model couldn't work, claiming surface fields were linked to subsurface toroids. However, after beginning their own simulations, their emphasis gradually shifted from arguing against the model to exploiting it. This process illustrates a classic case of paradigm shift in science.

#### 5.2 Ephemeral Region의 역할 / Role of Ephemeral Regions

흥미로운 발견은 ephemeral region(수명이 짧은 소규모 자기 영역)의 역할에 관한 것이었다. 태양주기 21 동안 전체 source의 85%가 ephemeral region이었고, 이들이 총 플럭스의 50%를 차지했다. 그러나 이들을 시뮬레이션에서 제외해도 평균 자기장 패턴에는 거의 영향이 없었다.

An interesting finding concerned the role of ephemeral regions (short-lived small-scale magnetic regions). During cycle 21, 85% of all sources were ephemeral regions, accounting for 50% of the total flux. However, excluding them from simulations had little effect on the mean-field pattern.

이것은 **Harvey (1994)**에 의해 정량적으로 확인되었는데, ephemeral region의 dipole 기여는 active region의 dipole 기여의 약 1/6에 불과했다. 그 이유는 ephemeral region이 무작위에 가까운 방향으로 나타나기 때문에, 그들의 순 dipole 기여(net dipole contribution)가 크게 상쇄되기 때문이다. 대규모 자기장 패턴을 결정하는 것은 Joy의 법칙에 따라 체계적으로 기울어진 활동 영역이다.

This was quantitatively confirmed by **Harvey (1994)**, who found that the dipole contribution of ephemeral regions was only about 1/6 that of active regions during cycle 21. The reason is that ephemeral regions emerge with nearly random orientations, so their net dipole contributions largely cancel. It is the active regions, systematically tilted according to Joy's law, that determine the large-scale magnetic field pattern.

---

### 6. 다주기 시뮬레이션 (2000년대) / Simulations Over Many Cycles (2000s)

#### 6.1 Open Flux의 세기 변화 / Open Flux Variation

**Lockwood et al. (1999)**는 20세기 동안 태양의 open flux가 두 배로 증가했다고 보고했다. **Wang et al. (2000a,b)**은 open flux가 total dipole moment의 진화를 따른다는 것을 보여주었다. 이는 flux-transport 모델이 태양권 자기장의 장기적 변동까지 설명할 수 있는 잠재력을 가짐을 의미했다.

**Lockwood et al. (1999)** reported that the Sun's open flux doubled during the 20th century. **Wang et al. (2000a,b)** showed that open flux follows the evolution of the total dipole moment. This implied the flux-transport model had the potential to explain long-term variations in heliospheric magnetic fields.

#### 6.2 Source Flux, Flow Speed, 그리고 Polar Field의 비직관적 관계 / Counter-intuitive Relationship Between Source Flux, Flow Speed, and Polar Field

이 시기에 발견된 가장 중요하고 비직관적인 결과 중 하나는 다음과 같다:

One of the most important and counter-intuitive results discovered during this period is:

- **더 강한 source flux** (더 활발한 태양주기) → **더 강한 극 자기장**이 예상되지만...
- **더 빠른 meridional flow** (diffusion에 비해 상대적으로) → **더 약한 극 자기장**

- **Stronger source flux** (more active solar cycle) → **stronger polar field** is expected, BUT...
- **Faster meridional flow** (relative to diffusion) → **WEAKER polar field**

이 두 번째 관계가 비직관적인 이유를 설명하면: meridional flow가 빠르면 자기 플럭스가 극으로 빨리 도달하지만, 동시에 diffusion이 작용할 시간이 줄어든다. Diffusion은 반대 극성의 선행 플럭스를 적도에서 상쇄시키는 핵심 과정인데, 이 과정이 충분히 일어나기 전에 플럭스가 극으로 운반되면, 극에 도달하는 순 불균형 플럭스(net unbalanced flux)가 감소한다. 즉, flow가 빠르면 선행/후행 극성의 분리가 덜 효과적으로 이루어지고, 결과적으로 극 자기장이 약해진다.

The reason the second relationship is counter-intuitive: faster meridional flow delivers magnetic flux to the poles more quickly, but simultaneously reduces the time for diffusion to act. Diffusion is the key process that cancels opposite-polarity leading flux at the equator. If flux is transported to the poles before this cancellation is sufficiently complete, the net unbalanced flux arriving at the poles decreases. In other words, faster flow makes the separation of leading/trailing polarities less effective, resulting in a weaker polar field.

**이 관계는 왜 강한 주기 다음에 약한 주기가 와도 극 자기장 역전이 보존되는지를 이해하는 열쇠이다.** 강한 주기는 많은 source flux를 생성하지만, 동시에 빠른 flow를 동반하여 극 자기장이 지나치게 강해지는 것을 방지한다.

**This relationship is the key to understanding why polar field reversal is preserved even when a weak cycle follows strong ones.** Strong cycles produce abundant source flux but simultaneously come with faster flow, preventing the polar field from becoming excessively strong.

#### 6.3 100년 시뮬레이션 / 100-Year Simulation

**Wang et al. (2002a)**은 주기별로 ±6 m/s의 가변 meridional flow를 사용한 100년 시뮬레이션을 수행했다. 활발한 주기(active cycle)에서는 빠른 flow, 비활발한 주기(inactive cycle)에서는 느린 flow를 적용했으며, 이는 flux-transport dynamo 이론(Dikpati & Charbonneau 1999)과 일치했다. 이 시뮬레이션은 20세기의 관측된 자기장 변동을 성공적으로 재현했다.

**Wang et al. (2002a)** performed a 100-year simulation with variable meridional flow of ±6 m/s cycle-to-cycle. Fast flow was applied during active cycles and slow flow during inactive cycles, consistent with flux-transport dynamo theory (Dikpati & Charbonneau 1999). This simulation successfully reproduced the observed magnetic field variations of the 20th century.

#### 6.4 미해결 문제들 / Remaining Challenges

여러 연구에서 모델의 한계가 드러났다:

Several studies revealed limitations of the model:

- **Schrijver et al. (2002)**: 시뮬레이션에서 극 자기장이 역전되지 않는 경우를 발견했다. 약 5년의 유한한 플럭스 수명(finite flux lifetime)을 도입하여 이를 해결하려 했다. 이는 자기장이 무한히 지속되지 않고 점진적으로 소멸됨을 의미한다.
- **Lean et al. (2002)**: 점진적으로 강해지는 주기가 약한 주기에 의해 중단될 때 역전이 일어나지 않음을 발견했다.
- **Mackay et al. (2002a,b)**: 모델에서 open flux가 sunspot minimum에 peak를 보이는 반면, 관측에서는 maximum 후 1-2년에 peak를 보이는 불일치를 발견했다. 이를 맞추려면 doublet 강도를 3배 증가시키고, diffusion을 500 km$^2$ s$^{-1}$로 줄이고, flow를 약 25 m/s로 증가시켜야 했다.

- **Schrijver et al. (2002)**: Found cases where simulated polar fields did not reverse. Introduced a finite flux lifetime of approximately 5 years to address this, implying magnetic fields do not persist indefinitely but gradually decay.
- **Lean et al. (2002)**: Found non-reversal when progressively stronger cycles were interrupted by a weaker one.
- **Mackay et al. (2002a,b)**: Found that open flux peaked at sunspot minimum in the model versus 1-2 years after maximum in observations. Matching required increasing doublet strengths by a factor of 3, reducing diffusion to 500 km$^2$ s$^{-1}$, and increasing flow to approximately 25 m/s.

---

### 7. 에필로그 / Epilogue

#### 7.1 역사적 맥락 / Historical Context

Sheeley는 이 논문이 기술적 세부사항이 아닌 역사적 서사임을 강조한다. 심사 과정에서 Rabin et al. (1991)의 리뷰와 Csada (1949, 1955)의 differential rotation 이론이 추가적으로 언급되었다.

Sheeley emphasizes that this paper is a historical narrative, NOT technical details. During review, Rabin et al. (1991) review and Csada (1949, 1955) differential rotation theory were additionally referenced.

#### 7.2 아이디어의 선구자들 / Precursors of the Idea

**Babcock (1961)**과 그의 아버지 **H. W. Babcock (1955)**은 후행 극성의 poleward migration에 대해 추측한 바 있다. 그러나 Leighton의 기여는 random-walk diffusion과 Joy의 법칙 기울기를 결합한 것이 설득력 있는 물리적 메커니즘을 제공했다는 점에서 차별화된다. Babcock 부자의 추측은 정성적이었지만, Leighton은 정량적으로 이 과정이 극 자기장을 역전시키기에 충분함을 보여주었다.

**Babcock (1961)** and his father **H. W. Babcock (1955)** had speculated about poleward migration of trailing polarity. However, Leighton's contribution is distinguished by providing a convincing physical mechanism combining random-walk diffusion with Joy's law tilt. The Babcocks' speculation was qualitative, but Leighton showed quantitatively that this process was sufficient to reverse the polar fields.

#### 7.3 모델의 진화 요약 / Summary of Model Evolution

Flux-transport 모델은 40년에 걸쳐 점진적으로 발전했다:

The flux-transport model evolved gradually over 40 years:

| 연도 / Year | 구성요소 / Components |
|---|---|
| 1963 | Diffusion + Differential rotation |
| 1983 | + Meridional flow |
| 2002 | + Variable flow speed (cycle-to-cycle) |

각 단계에서 새로운 물리적 요소가 추가되면서 모델의 설명력과 예측력이 향상되었다.

At each stage, new physical elements were added, improving the model's explanatory and predictive power.

---

## 핵심 요점 / Key Takeaways

1. **Flux-transport 메커니즘의 세 가지 핵심 요소**: 태양 표면 자기장의 대규모 진화는 supergranular diffusion ($\eta \approx 500$-$600$ km$^2$ s$^{-1}$), differential rotation, 그리고 poleward meridional flow ($v \approx 10$-$25$ m s$^{-1}$)의 세 가지 표면 과정으로 정량적으로 설명된다. / The large-scale evolution of the solar surface magnetic field is quantitatively explained by three surface processes: supergranular diffusion, differential rotation, and poleward meridional flow.

2. **극 자기장 역전은 Joy의 법칙에 의존한다**: Bipolar region의 체계적 기울기가 후행 극성을 극으로 우선적으로 운반하게 하여, 한 주기당 약 $10^3$개의 sunspot group으로 극 자기장 역전이 가능하다. / Polar field reversal depends on Joy's law: systematic tilt of bipolar regions preferentially transports trailing polarity to the poles, with approximately $10^3$ sunspot groups per cycle sufficient for reversal.

3. **준강체 회전은 지하 자기장 없이 설명된다**: 1970년대의 지하 자기장 패러다임은 잘못되었으며, "수영하는 오리" 메커니즘(meridional flow + diffusion)이 대규모 패턴의 준강체 회전을 완전히 설명한다. / Quasi-rigid rotation is explained without subsurface fields: the 1970s subsurface field paradigm was incorrect, and the "swimming duck" mechanism (meridional flow + diffusion) fully explains quasi-rigid rotation of large-scale patterns.

4. **Meridional flow 속도와 극 자기장의 비직관적 관계**: 더 빠른 meridional flow는 직관과 달리 더 약한 극 자기장을 생성하는데, 이는 diffusion에 의한 선행/후행 극성 분리가 불충분해지기 때문이다. 이 관계가 태양주기 간 극 자기장 역전의 강건성을 보장한다. / Faster meridional flow counter-intuitively produces weaker polar fields because leading/trailing polarity separation by diffusion becomes insufficient. This relationship ensures robustness of polar field reversal across solar cycles.

5. **Ephemeral region은 대규모 패턴에 무시할 수 있다**: 전체 source의 85%, 총 플럭스의 50%를 차지하지만, 무작위 방향으로 나타나기 때문에 순 dipole 기여가 미미하여 평균 자기장 패턴에 거의 영향을 미치지 않는다. / Ephemeral regions are negligible for large-scale patterns: despite being 85% of sources and 50% of total flux, their random orientations result in minimal net dipole contribution.

6. **모델은 코로나 구조와 태양풍까지 예측한다**: Flux-transport 모델과 potential-field source-surface 모델의 결합은 코로나 홀 진화와 태양풍 속도를 성공적으로 예측하여, 모델의 적용 범위를 광구에서 태양권까지 확장했다. / The model predicts coronal structure and solar wind: coupling with the potential-field source-surface model successfully predicts coronal hole evolution and solar wind speed, extending the model's scope from photosphere to heliosphere.

7. **과학적 진보에서의 교훈: 수치 시뮬레이션의 힘**: 1970년대의 "암흑기"를 끝낸 것은 새로운 관측이 아니라, 1981년 NRL에서의 직접 수치 시뮬레이션 결정이었다. 추측 대신 시뮬레이션이라는 접근은 10년 이상의 논쟁을 해결했다. / The lesson for scientific progress: the "dark ages" of the 1970s were ended not by new observations but by the 1981 NRL decision to directly simulate. The approach of simulation instead of speculation resolved over a decade of debate.

8. **미해결 문제**: Open flux의 peak timing 불일치, 유한한 플럭스 수명의 필요성, 그리고 주기별 가변 meridional flow의 물리적 기원은 아직 완전히 해결되지 않았다. / Remaining issues: the peak timing mismatch of open flux, the need for finite flux lifetime, and the physical origin of cycle-variable meridional flow remain incompletely resolved.

---

## 수학적 요약 / Mathematical Summary

### Flux-Transport 방정식 / The Flux-Transport Equation

태양 표면의 radial 자기장 $B_r(\theta, \phi, t)$의 시간 진화를 기술하는 flux-transport 방정식은 다음과 같다 (Leighton 1964에서 최초 정립; 이 논문의 모든 시뮬레이션의 기초):

The flux-transport equation describing the time evolution of the radial magnetic field $B_r(\theta, \phi, t)$ on the solar surface is (first formulated in Leighton 1964; the foundation of all simulations in this paper):

$$\frac{\partial B_r}{\partial t} = -\frac{1}{R \sin\theta} \frac{\partial}{\partial \theta}(v_\theta B_r \sin\theta) - \frac{1}{R \sin\theta} \frac{\partial}{\partial \phi}(v_\phi B_r) + \frac{\eta}{R^2}\left[\frac{1}{\sin\theta} \frac{\partial}{\partial \theta}\left(\sin\theta \frac{\partial B_r}{\partial \theta}\right) + \frac{1}{\sin^2\theta} \frac{\partial^2 B_r}{\partial \phi^2}\right] + S(\theta, \phi, t)$$

여기서 각 항의 의미는:

Where each term represents:

| 항 / Term | 물리적 의미 / Physical Meaning |
|---|---|
| $\dfrac{\partial B_r}{\partial t}$ | Radial 자기장의 시간 변화율 / Time rate of change of the radial magnetic field |
| $-\dfrac{1}{R\sin\theta}\dfrac{\partial}{\partial\theta}(v_\theta B_r \sin\theta)$ | **Meridional flow**에 의한 수송: poleward flow ($v_\theta$)가 플럭스를 극 방향으로 이동 / Transport by meridional flow: poleward flow ($v_\theta$) carries flux toward the poles |
| $-\dfrac{1}{R\sin\theta}\dfrac{\partial}{\partial\phi}(v_\phi B_r)$ | **Differential rotation**에 의한 수송: 위도별 차등 회전($v_\phi$)이 플럭스를 경도 방향으로 전단 / Transport by differential rotation: latitude-dependent rotation ($v_\phi$) shears flux in longitude |
| $\dfrac{\eta}{R^2}[\cdots]$ | **Supergranular diffusion**: 소규모 대류에 의한 자기 플럭스의 무작위 보행 확산 / Supergranular diffusion: random-walk spreading of magnetic flux by small-scale convection |
| $S(\theta, \phi, t)$ | **Source term**: 새로운 bipolar magnetic region의 출현 (관측 데이터에서 입력) / Source term: emergence of new bipolar magnetic regions (input from observational data) |

### 핵심 매개변수 (최종값) / Key Parameters (Final Values)

| 매개변수 / Parameter | 값 / Value | 비고 / Notes |
|---|---|---|
| Supergranular diffusion rate ($\eta$) | $500$-$600$ km$^2$ s$^{-1}$ | 초기 추정 $730 \pm 250$ → meridional flow 포함 시 감소 / Initial estimate $730 \pm 250$ → reduced with meridional flow |
| Meridional flow speed ($v$) | $10$-$25$ m s$^{-1}$ poleward | 극 방향, 위도에 따라 변화 / Poleward, varies with latitude |
| Flow speed variation | $\pm 6$ m s$^{-1}$ cycle-to-cycle | 활발한 주기: 빠른 flow / Active cycles: faster flow |
| Doublet threshold | $\geq 0.1 \times 10^{21}$ Mx | 시뮬레이션 입력 source의 최소 pole 강도 / Minimum pole strength for simulation input sources |
| Subsurface return flow (dynamo) | $\sim 1$ m s$^{-1}$ | Flux-transport dynamo 모델에서의 값 / Value in flux-transport dynamo model |
| Subsurface turbulent diffusion (dynamo) | $\sim 10$ km$^2$ s$^{-1}$ | Flux-transport dynamo 모델에서의 값 / Value in flux-transport dynamo model |

### Diffusion Rate 추정의 역사 / History of Diffusion Rate Estimates

| 연구 / Study | $\eta$ (km$^2$ s$^{-1}$) | 비고 / Notes |
|---|---|---|
| Leighton (1964) | 770-1540 | 최초 이론적 추정 / First theoretical estimate |
| Mosher (1977) | 200-400 | 과소추정 (1970년대 "암흑기") / Underestimate ("dark ages") |
| Sheeley et al. (1983) | $730 \pm 250$ | NRL 시뮬레이션 (meridional flow 미포함) / NRL simulation (without meridional flow) |
| 이후 연구 / Later studies | 500-600 | Meridional flow 포함 시 / With meridional flow included |

---

## 논문의 역사적 위치 / Paper in the Arc of History

```
1955    1961    1963    1964    1969    1972    1977    1981    1983
  |       |       |       |       |       |       |       |       |
  H.W.    H.D.    Leighton Leighton Leighton Schatten Mosher  Boris   Sheeley
  Babcock Babcock phone   flux-    magneto- et al.  low η   suggests et al.
  polar   solar   call to  transport kinematic first   200-400 direct  η=730
  field   cycle   Sheeley  equation  model    numerical        sim at  ±250
  obs.    model   (idea!)  (η+Ω)   (no v_m) sim              NRL
  |       |       |       |       |       |       |       |       |
  v       v       v       v       v       v       v       v       v
──┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼──
  |                                                               |
  |<--- 선구적 아이디어 / Precursor ideas --->|<-- "암흑기" -->|<-- NRL 시작
  |                                           |  "Dark Ages"  |   NRL begins


1985    1987    1988    1989    1991    1998    1999    2000    2002    2005
  |       |       |       |       |       |       |       |       |       |
  DeVore  Sheeley Wang    Wang    Wang &  van B.  Lockwood Wang   Wang    Sheeley
  et al.  et al.  et al.  et al.  Sheeley et al.  et al.  et al. et al.  THIS
  cycle   "duck"  PFSS    drop    flux-   filament open    open   100-yr  REVIEW
  21 sim  analogy coupling "if    transport channels flux   flux   sim
                          present" dynamo         doubled follows variable
                                                         dipole  flow
  |       |       |       |       |       |       |       |       |       |
  v       v       v       v       v       v       v       v       v       v
──┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼──
  |                                       |                               |
  |<---- "계몽의 시대" / Era of Enlightenment ---->|<-- 다주기 / Multi-cycle ->|
```

---

## 다른 논문과의 연결 / Connections to Other Papers

| 논문 / Paper | 관계 / Relationship | 설명 / Description |
|---|---|---|
| Babcock (1961) | 선행 연구 / Precursor | 태양 자기 주기 모델; 후행 극성의 poleward migration 추측 / Solar magnetic cycle model; speculated poleward migration of trailing polarity |
| Leighton (1964) | 기초 이론 / Foundation | Flux-transport 방정식의 최초 정립 (diffusion + differential rotation) / First formulation of the flux-transport equation |
| Leighton (1969) | 확장 / Extension | Meridional flow 없는 magneto-kinematic butterfly diagram model / Magneto-kinematic butterfly diagram model without meridional flow |
| Schatten et al. (1972) | 최초 수치화 / First numerical | Mount Wilson magnetogram을 사용한 최초 수치 시뮬레이션 / First numerical simulation using Mount Wilson magnetograms |
| Wang, Nash & Sheeley (1989) | PFSS 결합 / PFSS coupling | Flux-transport + potential-field source-surface model → 코로나 홀, 태양풍 예측 / Corona and solar wind prediction |
| Wang & Sheeley (1991) | Dynamo 확장 / Dynamo extension | Flux-transport dynamo with subsurface return flow → butterfly diagram 재현 / Reproduced butterfly diagram with subsurface return flow |
| Choudhuri, Schüssler & Dikpati (1995) | 발전 / Development | Flux-transport dynamo 모델의 독립적 발전 / Independent development of flux-transport dynamo |
| van Ballegooijen et al. (1998) | 응용 / Application | 수평 자기장 성분으로 확장 → filament channel 모델 / Extended to horizontal field → filament channel model |
| Dikpati & Charbonneau (1999) | 발전 / Development | Flux-transport dynamo의 체계적 연구; 가변 flow 속도와의 연결 / Systematic flux-transport dynamo study; connection to variable flow speed |
| Wang, Lean & Sheeley (2002) | 장기 시뮬레이션 / Long-term sim | 100년 시뮬레이션; 주기별 가변 meridional flow / 100-year simulation with cycle-variable meridional flow |
| Schrijver, DeRosa & Title (2002) | 한계 탐색 / Limits explored | 극 자기장 미역전 사례; 유한 플럭스 수명(~5년) 도입 / Non-reversal cases; finite flux lifetime (~5 years) |
| Mackay, Priest & Lockwood (2002) | 문제 제기 / Challenge | Open flux peak timing 불일치 발견; 매개변수 조정 필요성 제기 / Found open flux peak timing mismatch; raised need for parameter adjustment |

---

## 참고문헌 / References

- Babcock, H.D., "The Sun's Polar Magnetic Field", *Astrophys. J.*, **130**, 364, 1959.
- Babcock, H.W., "The Topology of the Sun's Magnetic Field and the 22-Year Cycle", *Astrophys. J.*, **133**, 572-587, 1961.
- Choudhuri, A.R., Schüssler, M. & Dikpati, M., "The solar dynamo with meridional circulation", *Astron. Astrophys.*, **303**, L29-L32, 1995.
- DeVore, C.R., Sheeley Jr., N.R. & Boris, J.P., "Simulations of Magnetic-Flux Transport on the Sun", *Solar Phys.*, **102**, 41-49, 1985.
- Dikpati, M. & Charbonneau, P., "A Babcock-Leighton Flux Transport Dynamo with Solar-Like Differential Rotation", *Astrophys. J.*, **518**, 508-520, 1999.
- Harvey, K.L., "The Solar Magnetic Cycle", in *Solar Surface Magnetism* (eds. R.J. Rutten & C.J. Schrijver), 347-363, Kluwer, 1994.
- Lean, J., Wang, Y.-M. & Sheeley Jr., N.R., "The Effect of Increasing Solar Activity on the Sun's Total and Open Magnetic Flux During Multiple Cycles", *Geophys. Res. Lett.*, **29**, 2224, 2002.
- Leighton, R.B., "Transport of Magnetic Fields on the Sun", *Astrophys. J.*, **140**, 1547-1562, 1964.
- Leighton, R.B., "A Magneto-Kinematic Model of the Solar Cycle", *Astrophys. J.*, **156**, 1-26, 1969.
- Lockwood, M., Stamper, R. & Wild, M.N., "A doubling of the Sun's coronal magnetic field during the past 100 years", *Nature*, **399**, 437-439, 1999.
- Mackay, D.H., Priest, E.R. & Lockwood, M., "The Evolution of the Sun's Open Magnetic Flux: I. A Single Bipole", *Solar Phys.*, **207**, 291-308, 2002a.
- Mosher, J.M., *The Magnetic History of Solar Active Regions*, Ph.D. thesis, Caltech, 1977.
- Schatten, K.H., Leighton, R.B. & Howard, R., "Simulated Solar Magnetic Fields", *Solar Phys.*, **26**, 283-291, 1972. [Note: often cited as Schatten et al. 1972 though Sheeley references this early numerical work]
- Schrijver, C.J., DeRosa, M.L. & Title, A.M., "What Is Missing from Our Understanding of Long-Term Solar and Heliospheric Activity?", *Astrophys. J.*, **577**, 1006-1012, 2002.
- Sheeley Jr., N.R., "Surface Evolution of the Sun's Magnetic Field: A Historical Review of the Flux-Transport Mechanism", *Living Rev. Solar Phys.*, **2**, 5, 2005. [DOI: 10.12942/lrsp-2005-5]
- Sheeley Jr., N.R., Boris, J.P. & DeVore, C.R., "Simulations of the Mean Solar Magnetic Field during Sunspot Cycle 21", *Astrophys. J.*, **272**, 739-751, 1983.
- Sheeley Jr., N.R., Nash, A.G. & Wang, Y.-M., "The Origin of Rigidly Rotating Magnetic Field Patterns on the Sun", *Astrophys. J.*, **319**, 481-502, 1987.
- Stenflo, J.O., "On the Origin of Solar Cycle Periodicity", *Astron. Astrophys.*, **210**, 403-409, 1989a.
- van Ballegooijen, A.A., Cartledge, N.P. & Priest, E.R., "Magnetic Flux Transport and the Formation of Filament Channels on the Sun", *Astrophys. J.*, **501**, 866-881, 1998.
- Wang, Y.-M., Nash, A.G. & Sheeley Jr., N.R., "Magnetic Flux Transport on the Sun", *Science*, **245**, 712-718, 1989a.
- Wang, Y.-M. & Sheeley Jr., N.R., "Magnetic Flux Transport and the Sun's Dipole Moment: New Twists to the Babcock-Leighton Model", *Astrophys. J.*, **375**, 761-770, 1991.
- Wang, Y.-M., Lean, J. & Sheeley Jr., N.R., "The Long-Term Variation of the Sun's Open Magnetic Flux", *Geophys. Res. Lett.*, **27**, 505-508, 2000a.
- Wang, Y.-M., Lean, J. & Sheeley Jr., N.R., "Role of a Variable Meridional Flow in the Secular Evolution of the Sun's Polar Fields and Open Flux", *Astrophys. J. Lett.*, **577**, L53-L57, 2002a.
- Wang, Y.-M., Sheeley Jr., N.R. & Nash, A.G., "A New Solar Cycle Model Including Meridional Circulation", *Astrophys. J.*, **335**, 726-732, 1988.
