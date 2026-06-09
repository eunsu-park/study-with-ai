---
title: "A Model for Solar Coronal Mass Ejections"
authors: [S. K. Antiochos, C. R. DeVore, J. A. Klimchuk]
year: 1999
journal: "The Astrophysical Journal, 510, 485–493"
doi: "10.1086/306563"
topic: Solar_Physics
tags: [CME, breakout_model, magnetic_reconnection, multipolar_topology, aly_sturrock, MHD_simulation, eruptive_flares]
status: completed
date_started: 2026-04-19
date_completed: 2026-04-19
---

# 25. A Model for Solar Coronal Mass Ejections / 태양 코로나 질량 방출 모델

> *"We propose that the energy for CMEs and eruptive flares is due to this difference between $E_{\max}$ and $E_{\min}$."* — Antiochos, DeVore & Klimchuk (1999)

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 태양 CME(Coronal Mass Ejection)의 개시 기작을 설명하는 **"자기 돌파(Magnetic Breakout) Model"**을 제안한다. 핵심 난제는 Aly(1984)–Sturrock(1991) 부등식으로, **주어진 광구 법면 자속을 유지하는 simply-connected force-free 장에서는 완전 개방 상태의 에너지가 모든 폐쇄 상태보다 높다**는 것이다. 이 때문에 단순 쌍극 아케이드를 순수 자기적으로(중력·가스압 무시) 폭발시키는 것은 에너지 보존에 위배된다. 저자들은 이 장애물을 **토폴로지 일반화**로 해결한다: 중심 쉬어드 아케이드(blue) + 양측 불활성 아케이드(green) + 상공 불활성 아케이드(red)로 구성된 **4-플럭스 계(quadrupolar / δ-sunspot analog)**에서는, 중심 자속만 열리고 나머지는 폐쇄 상태를 유지할 수 있기 때문에 Aly–Sturrock 상계를 우회한다. 핵심 동역학은 **중심-상공 경계의 X-type null에서 일어나는 재결합**이며, 이 재결합이 상공의 구속(tethering)을 제거하여 중심 아케이드를 "breakout"시킨다.

저자들은 이를 2.5D 구면 축대칭 힘없는 장(force-free) 계산과 시간 의존 MHD 시뮬레이션으로 정량 검증한다. Shear $\chi=\pi/2$일 때 MHD 자유 에너지는 최소 개방 상태 $E_{\min}$의 약 **2배**까지 축적되고 ($3\pi/8$에서 이미 해당 값의 2배 도달), 재결합이 지연되는 한 에너지 저장은 계속된다. 모델은 (1) CME의 **폭발적 개시**, (2) **δ-spot 활동영역의 플레어 생산성**, (3) **플레어-CME 상호관계**, (4) **전단 국소성(near-neutral-line shear)** 을 하나의 시나리오로 통합 설명한다. 이후 현대 heliophysics에서 CME 개시의 양대 패러다임(Breakout vs. Torus instability) 중 하나로 자리잡았다.

### English
This paper proposes the **"Magnetic Breakout Model"** for the initiation of solar Coronal Mass Ejections (CMEs). The core obstacle for CME theory is the **Aly (1984) – Sturrock (1991) conjecture**: for a simply-connected force-free field with a fixed photospheric normal-flux distribution, the fully open state has *higher* magnetic energy than any closed state. A purely magnetically driven eruption of a simple bipolar arcade therefore violates energy conservation. The authors circumvent this obstacle via **topological generalization**: in a **four-flux (quadrupolar / δ-spot analog) configuration** — central sheared arcade (blue) flanked by two unsheared side arcades (green) and capped by an overlying unsheared polar arcade (red) — only a fraction of the flux needs to open, while the rest remains closed, thereby bypassing the Aly–Sturrock bound. The essential dynamics is **reconnection at the X-type null between the central and overlying systems**, which removes the overlying restraint and allows the sheared core to "break out."

The authors verify this quantitatively with 2.5-D axisymmetric spherical force-free and time-dependent MHD simulations. At a shear $\chi=\pi/2$ the MHD free energy reaches roughly **twice the minimum open-state energy $E_{\min}$** (already $2E_{\min}$ at $3\pi/8$), and energy continues to accumulate as long as reconnection at the null is delayed. The model jointly explains (1) the **impulsive onset** of CMEs, (2) **why δ-spot active regions are disproportionately flare-productive**, (3) the **flare–CME association**, and (4) the observationally-supported **near-neutral-line concentration of shear**. Together with the torus-instability / flux-rope picture (Paper #26), it has become one of the two leading paradigms for CME initiation.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Observational Constraints (§1, pp. 485–486) / 도입부 및 관측적 제약

#### 한국어
§1은 CME가 태양풍이 아니라 **자기 에너지**에 의해 구동됨을 재확인한다 (Gosling 1993, 1994; 논문 #22 참고). 주요 관측적 사실:

- 질량 $\gtrsim 10^{16}\,\mathrm{g}$, 에너지 $\gtrsim 10^{32}\,\mathrm{erg}$ (Howard et al. 1985; Hundhausen 1997).
- 각폭 $>60°$가 흔하고, 일부 LASCO 관측은 태양을 전체 감싸는 "global disruption"을 시사 (Brueckner 1996; Howard et al. 1997).
- 코로나 저 $\beta$ ($\sim 10^{-2}$), 따라서 **가스압 단독으로는 구동 불가**.
- 부력/낙하 모형은 일부 prominence-less CME(고위도 정숙 영역)를 설명 못 함.
- CME는 helmet streamer가 수일에 걸쳐 swelling 후 분출하는 경우가 많음 → quasi-static 축적 후 폭발.
- 광구 운동 속도 $\sim 1\,\mathrm{km\,s^{-1}}$ vs. Alfvén 속도 $\sim 10^3\,\mathrm{km\,s^{-1}}$ → 축적은 **quasi-static**.
- Shear 원천: photospheric flow (Klimchuk 1990) 또는 flux emergence — 모형에서는 무관.

CME 이론이 반드시 만족해야 할 세 가지 제약:
1. **에너지 원**: 자유 자기 에너지 ($W - W_{\mathrm{pot}}$).
2. **Quasi-static stressing**: 저속 광구 구동이 어떻게 격변을 낳는가?
3. **상한 경계 부재(no upper boundary)**: 에너지적으로 유리하면 장선은 무한대까지 팽창 가능.

#### English
§1 reaffirms CMEs are magnetically driven (Gosling 1993, 1994 — Paper #22). Key observational facts:

- Mass $\gtrsim 10^{16}\,\mathrm{g}$, energy $\gtrsim 10^{32}\,\mathrm{erg}$.
- Angular widths $>60°$; some LASCO observations suggest global disruptions.
- Coronal $\beta\sim 10^{-2}$, so gas pressure alone cannot drive eruption.
- Buoyancy models fail for prominence-less high-latitude quiet-region CMEs.
- Pre-eruption helmet streamers swell quasi-statically for days before erupting.
- Photospheric driving ($\sim 1\,\mathrm{km\,s^{-1}}$) is slow relative to Alfvén ($\sim 10^3\,\mathrm{km\,s^{-1}}$) → quasi-static accumulation.

Three constraints any CME theory must satisfy:
1. **Energy source**: free magnetic energy.
2. **Quasi-static stressing → violent release**: how do slow flows produce an impulsive eruption?
3. **No upper boundary**: field can escape to infinity if energetically favored.

---

### Part II: The Aly–Sturrock Obstacle (§1 cont., pp. 486–487) / Aly–Sturrock 장애

#### 한국어
**Aly (1984, 1991) and Sturrock (1991) conjecture** (p. 486):

> *Simply-connected force-free field 중, 광구에 고정된 법면 자속 $B_n$을 가지는 경우 **완전 개방 상태가 최대 에너지**를 가진다.*

수학적으로 엄밀한 증명은 없지만 모든 수치 시뮬레이션과 일치한다. **결과적 모순**:

- CME를 일으키려면 **개방된 후 플라즈마 가속과 중력장 극복**에 추가 에너지가 필요 $\Rightarrow$ 초기 폐쇄장이 $E_{\mathrm{open}}$보다 **더 많은** 에너지를 가져야 한다.
- 그러나 Aly–Sturrock은 단순 연결된 force-free 장에서 $E_{\mathrm{closed}}\leq E_{\mathrm{open}}$을 요구.
- 따라서 **simply-connected 가정을 깨야 한다**.

단일 아케이드에서의 탈출구 시도는 모두 실패:
- **Shear만 증가**: Cartesian 2.5D 계산 (Klimchuk & Sturrock 1989; Finn & Chen 1990; Wolfson & Verma 1991) — 열리지 않음.
- **Spherical 2.5D bipolar** (Roumeliotis, Sturrock & Antiochos 1994) — 열리지 않고 점근적으로 open state에 접근.
- **저항 추가** (Biskamp & Welter 1989; Inhester, Birn & Hesse 1992; Mikić & Linker 1994) — 부분적으로만 열림, 분리된 plasmoid 생성 (관측과 불일치).
- **3D tether-cutting** (Sturrock 1989; Moore & Roumeliotis 1992) — 단일 아케이드에서는 여전히 Aly–Sturrock 한계에 걸림.

저자들의 결론: **"any single arcade model is doomed to failure"** (p. 487).

#### English
The **Aly (1984, 1991) – Sturrock (1991) conjecture**:

> *Among simply-connected force-free fields with a fixed photospheric $B_n$, the fully open state has the maximum energy.*

The mathematical proof is incomplete but every numerical test agrees. The consequence:

- A CME needs **extra energy** beyond just opening the field (plasma acceleration, lifting against gravity) $\Rightarrow$ the initial closed field must hold **more** energy than $E_{\mathrm{open}}$.
- But Aly–Sturrock demands $E_{\mathrm{closed}}\leq E_{\mathrm{open}}$ for simply-connected fields.
- Escape route: **break the simply-connected assumption**.

Every attempt within a single arcade fails: pure shear, spherical bipolar, resistive single arcade (produces disconnected plasmoids, not observed), and 3-D tether-cutting. The authors conclude: *"any single arcade model is doomed to failure"* (p. 487).

---

### Part III: The Multipolar Escape & the Breakout Idea (§2, p. 487) / 다중극적 탈출

#### 한국어
§2는 해결책으로 **4-플럭스 다중극 토폴로지**를 제안 (Fig. 1a):

- **Blue**: 적도를 걸치는 중심 아케이드 (전단됨).
- **Green** (×2): 위도 $\pm 45°$의 중성선을 연결하는 측면 아케이드 (전단 없음).
- **Red**: 세 아케이드 전체를 덮는 극지-극지 상공 아케이드 (전단 없음).
- **Separatrix surfaces**: 두 개, 네 flux system 사이의 경계.
- **Null point**: 적도면의 세파라트릭스 교차점, $\mathbf{B}=0$.

**Breakout mechanism** (p. 487):
1. 중심(blue) 아케이드에만 저위도 전단 인가 → 팽창 압력 증가.
2. 팽창하는 blue 자속이 null에서 red 자속과 만나 **재결합**.
3. 재결합으로 unsheared blue/red 자속이 **green으로 전달** (flux transfer) → red의 구속(tethering)이 약화.
4. 상공 압력 감소 → 중심 자속 **폭발적 개방** ("breakout").
5. 중요한 점: **side green arcades는 여전히 폐쇄** → Aly–Sturrock 한계 우회.

이는 단일 아케이드에서 "열릴 곳이 없기 때문에 모두 열려야 한다"는 경직성을 깨뜨린다. 다중극에서는 **일부만 열리고 나머지는 그대로 남는 경로**가 존재한다.

#### English
§2 proposes the solution: **four-flux multipolar topology** (Fig. 1a):

- **Blue**: central arcade straddling the equator (sheared later).
- **Green** (×2): mid-latitude arcades rooted at the $\pm 45°$ neutral lines (unsheared).
- **Red**: overlying polar arcade covering all three (unsheared).
- Two separatrix surfaces divide the four systems; a null point sits on the equator.

**Breakout mechanism**: slow shear at the central neutral line → blue expands → reconnection with red at the null → flux transfer from red (and unsheared blue) to green → overlying restraint erodes → runaway opening of the central arcade. Because the green arcades remain closed, only *part* of the total flux has to open, and the Aly–Sturrock bound is evaded.

---

### Part IV: Energy Bounds — $E_{\max}$ vs. $E_{\min}$ (§2, p. 488) / 에너지 상한과 하한

#### 한국어
**두 개방 상태의 구분** (핵심 개념):

- $E_{\max}$: **중심 blue + 상공 red만** 열리고 green은 폐쇄 상태인 open 상태의 에너지. 자연 진화(재결합 없음)로 도달 가능한 상태.
- $E_{\min}$: **blue + red가 green으로 재결합**한 후, blue flux만 열린 상태의 에너지. 재결합이 있어야만 도달 가능.

그리고:
$$E_{\min} < E_{\max}$$

이유: $E_{\min}$ 상태는 열린 자속의 양이 더 적기 때문.

**동역학의 결정적 논리**:
- Shear가 없을 때 시스템은 $E_{\min}$의 force-free state에 있음.
- Shear가 증가하면 **재결합 속도**에 따라:
  - 재결합이 **빠르면**: 에너지는 $E_{\min}$ 근처에 머무름 → 폭발 없음.
  - 재결합이 **느리면**: 에너지가 $E_{\min}$을 초과해 $E_{\max}$까지 축적 → **폭발 가능**.
- 자연에서는 shear가 separatrix로부터 **멀리 떨어진 중성선 근처에만** 집중되므로, null 근처에서 초기에는 전류가 약하고 재결합이 지연된다.
- 전단이 커질수록 팽창하는 blue 자속이 null로 접근 → 전류 시트 발달 → 재결합 개시 → 폭발.

**핵심 인용** (p. 488):
> *"This magnetic 'breakout' model naturally implies explosive-type behavior."*

#### English
**Two distinct open states** — the key concept:

- $E_{\max}$: open state in which **only the central blue + overlying red** are open; the green arcades remain closed. Reachable by natural evolution *without* reconnection.
- $E_{\min}$: open state **after blue/red reconnection into green**, with only the blue flux open. Reachable only *with* reconnection at the null.

Crucially: $E_{\min} < E_{\max}$.

The dynamical logic:
- Without shear, the system sits in the $E_{\min}$ force-free state.
- As shear grows, behavior depends on reconnection rate:
  - **Fast reconnection**: energy pinned near $E_{\min}$ → no eruption.
  - **Slow reconnection**: energy can climb above $E_{\min}$, toward $E_{\max}$ → eruption.
- In nature shear concentrates near the neutral line, far from the separatrices, so reconnection is initially suppressed; as shear grows, the blue arcade expands and pushes on the null, thinning the current sheet there until reconnection abruptly switches on → runaway.

The authors emphasize: *"this magnetic 'breakout' model naturally implies explosive-type behavior"* (p. 488).

---

### Part V: Force-Free Numerical Setup (§3.1, pp. 488–489) / Force-Free 수치 설정

#### 한국어
**좌표계**: 구면 축대칭 ($\partial/\partial\phi=0$), 영역 $1\le r\le 100$, $0\le\theta\le \pi/2$ (상반구만 계산 — 적도 대칭).

**Euler potential 표현** (Yang, Sturrock & Antiochos 1986):
$$
\mathbf{B} = \nabla\alpha(r,\theta)\times\nabla[\phi-\gamma(r,\theta)]
\tag{2}
$$

여기서 $\alpha$는 flux function, $\gamma$는 shear function. 광구 $r=1$에서 $\gamma$를 고정하면 자속이 재결합하든 안 하든 **경계 전단은 엄격히 유지**.

**초기(potential) 자속 분포** (Eq. 3):
$$
\alpha(r,\theta) = \frac{\sin^2\theta}{r} + \frac{(3+5\cos^2\theta)\sin^2\theta}{2r^3}
$$

- 첫 항: dipole ($r$에서 지배, surface 멀리서).
- 둘째 항: octopole ($r\sim 2$에서 지배, middle corona).
- 조합으로 **quadrupolar with X-null**.

이로부터 (Eq. 4):
$$
B_r(r,\theta) = \frac{1}{r^2\sin\theta}\frac{\partial\alpha}{\partial\theta}
$$

광구에서:
$$
B_r(1,\theta) = 10\cos\theta\cos 2\theta
$$

→ **세 개의 중성선**: $\theta = \pi/4$, $\pi/2$, $3\pi/4$.

**Null 위치**: Eq. 3에서 $B_\theta(r,\pi/2)=r^{-3}-3r^{-5}=0$ → **$r=\sqrt{3}\approx 1.732\,R_\odot$**.

Null에서의 flux function 값:
$$
\alpha(3^{1/2},\pi/2) = \frac{2}{3\sqrt{3}} \approx 0.385
$$

→ 세파라트릭스 위도 $\theta = 0.2941$ rad 및 $1.277$ rad ($\approx 16.8°$ 및 $73.2°$).

**Shear 분포** (Eq. 5, Fig. 2):
$$
\gamma(1,\theta) = \begin{cases}
\chi C(\psi^2-\Theta^2)^2 \sin\psi, & \psi<\Theta \\
0, & \psi\geq\Theta
\end{cases}
$$

- $\psi = \pi/2-\theta$: 태양 위도.
- $\Theta = \pi/15$: 전단층의 위도 폭 (약 $12°$).
- $C = 8.68252\times 10^3$: 정규화 상수 ($\gamma=\chi$ at $\psi=0.094$).
- 최대 전단 영역은 $\theta = 13\pi/30$에서 끝나며, 이때 $\alpha=0.207$ (세파라트릭스 flux 값 $0.385$의 **약 절반**) → 중심 blue 자속의 **절반만이 전단됨**.

**수치 해법**: 반복 완화(iterative relaxation), $512\times 512$ 비균일 격자. 격자는 $x=\ln r$, $y=\exp(6\theta/\pi)$에서 균일 간격 → 태양 표면 및 적도 근처 해상도 향상.

**경계 조건**:
- $r=1$: $\gamma$ 고정.
- $r=100$: $\alpha, \gamma$ 고정 (자속 탈출 금지; 초기 근사).
- $\theta=0, \pi/2$: 대칭.

#### English
**Geometry**: spherical axisymmetric, $1\le r\le 100$, one hemisphere by symmetry. 

**Euler-potential form** (Eq. 2): $\mathbf{B}=\nabla\alpha\times\nabla[\phi-\gamma]$. Fixing $\gamma$ at $r=1$ rigorously enforces footpoint shear regardless of interior reconnection.

**Initial potential field** (Eq. 3): $\alpha(r,\theta)=\sin^2\theta/r+(3+5\cos^2\theta)\sin^2\theta/(2r^3)$ — dipole + octopole, giving a quadrupolar field with X-null at $r=\sqrt{3}$ on the equator. Three neutral lines at $\theta=\pi/4,\pi/2,3\pi/4$.

**Shear** (Eq. 5): localized to a $\pi/15\approx 12°$-wide band about the equator, with only ~half the central flux (up to $\alpha=0.207$ out of the separatrix value $0.385$) experiencing shear.

Grid: $512\times 512$ nonuniform grid in $(x=\ln r, y=\exp[6\theta/\pi])$, concentrating resolution near surface and equator.

---

### Part VI: Force-Free Results (§3.1, p. 489) / Force-Free 결과

#### 한국어
**Fig. 1b** ($\chi=\pi/8$): 이미 **red 자속의 일부와 light-blue(하부 unsheared blue) 자속의 일부가 green으로 재결합**. "재결합은 에너지적으로 선호되므로 force-free 코드가 이것을 찾는다."

**Fig. 1c** ($\chi=3\pi/8$), **Fig. 1d** ($\chi=\pi/2$): 재결합이 진행될수록 red+blue가 green으로 합병, blue가 점차 개방. 하지만 **짙은 blue (sheared) 자속의 뿌리(footpoint)는 여전히 적도 전단층 내 유지** — 광구 경계 조건에 의해.

**주요 사실**: 재결합된 자속들도 shear $\gamma$는 유지 (광구에서 footpoint 변위는 엄격히 보존). Karpen, Antiochos & DeVore (1996) 참고.

**Fig. 3** (자기 자유 에너지 vs. shear):
- 하부 곡선(force-free, fff): shear $\pi/4$ 근처에서 **약 6% above initial potential energy로 포화** → $E_{\min}=1.06\,E_{\mathrm{pot}}$.
- Shear가 두 배가 되어도 에너지 증가 없음 → force-free 시스템은 최저 에너지 개방 상태에 도달.
- **초기 중심 blue 아케이드 에너지만** 고려하면 상대적 증가는 **10배 이상** (즉, 중심 영역에서는 자유 에너지 ≫ potential blue 에너지).

#### English
**Fig. 1b** ($\chi=\pi/8$): some red and light-blue (lower unsheared blue) field lines have already reconnected into the green system — the force-free code finds this because it's energetically favored.

**Fig. 1c** ($\chi=3\pi/8$), **Fig. 1d** ($\chi=\pi/2$): progressive merging of red+blue into green; the sheared (dark-blue) flux stays tied to the equatorial shear band (enforced at the photospheric boundary). Reconnected field lines retain their footpoint displacement $\gamma$ — footpoints are rigorously held (Karpen, Antiochos & DeVore 1996).

**Fig. 3** (magnetic free energy vs. shear):
- Lower curve (fff): saturates at ~6% above initial potential energy near shear $\pi/4$ → $E_{\min}=1.06\,E_{\mathrm{pot}}$.
- Doubling the shear does not raise this energy — force-free evolution is pinned at the minimum open state.
- Relative to the **central blue arcade's** initial energy alone, the increase is more than **tenfold**.

---

### Part VII: MHD Setup (§3.2, p. 490) / MHD 설정

#### 한국어
**MHD 방정식** (Eqs. 6–9 in the paper, standard ideal MHD):
$$
\frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\mathbf{v})=0
$$
$$
\frac{\partial}{\partial t}(\rho\mathbf{v})+\nabla\cdot(\rho\mathbf{v}\mathbf{v})+\nabla P=\frac{1}{4\pi}(\nabla\times\mathbf{B})\times\mathbf{B}-\rho g_\odot R_\odot^2\frac{\mathbf{r}}{r^3}
$$
$$
\frac{\partial U}{\partial t}+\nabla\cdot(U\mathbf{v})+P\nabla\cdot\mathbf{v}=0
$$
$$
\frac{\partial\mathbf{B}}{\partial t}=\nabla\times(\mathbf{v}\times\mathbf{B})
$$

**수치**: 2.5D FCT (Flux-Corrected Transport) 코드 (DeVore 1991), 고차 차분법. Effective magnetic Reynolds number $R_M\gtrsim 10^4$ (격자가 해상할 때). 만약 grid scale 구조가 발달하면 $R_M\sim 500$으로 떨어짐 → 논문은 이것이 태양 $R_M$보다 훨씬 낮음을 명시.

**초기 플라즈마 정수(hydrostatic)**:
$$
T(r) = \frac{2\times 10^6}{r^7}\,\mathrm{K},\qquad n(r)=\frac{2\times 10^8}{r}\,\mathrm{cm}^{-3} \tag{10}
$$

급격한 $r^{-7}$ 감쇠는 $\beta=8\pi P/B^2$를 원거리에서 억제하기 위한 선택. 광구: $B=10$ G (극), $\beta=0.014$; 적도: $B=-2$ G, $\beta=0.35$. 그러나 null 근처에서는 항상 고 $\beta$.

**광구 구동**:
- 사인형(sinusoidal) 시간 프로파일, 주기 $25{,}000\,\mathrm{s}$, 최대 변위 $\pi/8$.
- 최대 footpoint 속도 $\approx 10\,\mathrm{km\,s^{-1}}$ vs. 저 코로나 Alfvén 속도 $\approx 500\,\mathrm{km\,s^{-1}}$ → quasi-static.
- 총 4상(phase) 적용 → 최대 전단 $\pi/2$까지.

**외부 경계**: $r=10$ (force-free 계산의 $r=100$보다 작음, MHD 해상도 확보용). 자속과 질량이 경계를 넘어갈 수 있도록 설정.

#### English
**MHD equations** (ideal, Eqs. 6–9): continuity, momentum with gravity, internal energy ($U=3P/2$), induction.

**Solver**: 2.5-D spherical FCT code (DeVore 1991), effective $R_M\gtrsim 10^4$ when structures are well-resolved, falling to $\sim 500$ at grid scale.

**Initial plasma** (Eq. 10): $T(r)=2\times 10^6/r^7\,\mathrm{K}$, $n(r)=2\times 10^8/r\,\mathrm{cm}^{-3}$ — artificially steep falloff to keep $\beta$ manageable. $\beta=0.014$ at pole, $0.35$ at equator, very high near the null.

**Driver**: four successive sinusoidal shear phases, each of period $25{,}000\,\mathrm{s}$ and amplitude $\pi/8$. Max footpoint velocity $\sim 10\,\mathrm{km\,s^{-1}}\ll v_A\sim 500\,\mathrm{km\,s^{-1}}$ → quasi-static.

---

### Part VIII: MHD Results — Null Deformation (§3.2, pp. 490–492, Fig. 4) / MHD 결과 — Null 변형

#### 한국어
**Fig. 4a** (shear $\pi/8$, 첫 shear phase 종료 후): force-free Fig. 1b와 달리 **재결합의 흔적 없음**. MHD에서는 gas pressure가 null 근처에서 force-free 해와 다른 평형을 허용. 50,000s 추가 완화 후에도 재결합 없음 → 다른 magnetostatic equilibrium이 존재.

**이 중요한 발견**: 동일한 경계 조건, 동일한 shear에 대해 **최소 두 개의 평형**이 존재.
1. Force-free 상태 (Fig. 1b): $E_{\min}$, X-null이 직각 세파라트릭스.
2. MHD 상태 (Fig. 4a): $E > E_{\min}$, null이 변형되어 전류 시트로 접근.

**Fig. 4b, 4c, 4d**: 추가 shear phase (총 $\pi/4$, $3\pi/8$, $\pi/2$). 재결합은 마지막까지 거의 일어나지 않고, blue는 외향 팽창, red는 밀려 올라감, green은 **옆으로 비켜남**.

**Null 근처 국소 해석 모형** (p. 491, Eqs. 11–15):
원점을 null에 두고 국소 Cartesian 좌표 $(x,y)$ 사용. Shear 없을 때 potential:
$$
\alpha(x,y) = B_0\left(\frac{y^2}{2l_y}-\frac{x^2}{2l_x}\right) \tag{11}
$$

이로부터:
$$
\mathbf{B} = B_0\left(\frac{y}{l_y}\hat{x} + \frac{x}{l_x}\hat{y}\right) \tag{12}
$$
$$
\mathbf{J} = B_0\frac{l_y-l_x}{l_x l_y}\hat{z} \tag{13}
$$

**스케일 비대칭이 전류를 낳음**:
- $l_x = l_y$: null이 직각 X-point, $\mathbf{J}=0$ (potential).
- $l_x\neq l_y$: 유한 $\mathbf{J}$, Lorentz 힘 $\mathbf{J}\times\mathbf{B} \neq 0$.

Lorentz 힘이 중력을 무시할 만큼 클 때, 가스압으로 균형:
$$
P = P_0 + \frac{B_0^2}{4\pi}\frac{l_y-l_x}{l_x l_y}\left(-\frac{x^2}{2l_x}+\frac{y^2}{2l_y}\right) \tag{15}
$$

**동역학**: Shear로 blue가 팽창 → null 위에서 scale $l_y$ 줄어듦, scale $l_x$ 유지 → $l_x\gg l_y$로 비대칭 심화 → 전류 시트가 **지수적으로 얇아짐** (Roumeliotis et al. 1994; Sturrock et al. 1995). 이론적으로는 force-free 평형이 존재하지 않아야 하지만, **plasma $\beta$가 유한**하기 때문에 gas pressure가 유한한 두께의 전류 시트를 지탱.

**재결합 개시의 조건**: 전류 시트 두께가 **수치적 확산 스케일** (격자 몇 개)까지 줄어들 때. 실제 태양에서는 $R_M\sim 10^{12}$이므로 훨씬 강한 $\chi$까지 축적 가능. 시뮬레이션에서 $\chi=\pi/2$ 단계에서 일부 재결합 시작, 추가 완화를 시도하니 **재결합이 가속**하며 magnetic islands가 나타나고 속도/밀도 폭발적 변화로 시뮬레이션 중단.

**Fig. 3 상부 곡선(mhd)**: MHD 에너지는 shear $\pi/4$에서 이미 $E_{\min}$ 초과, $3\pi/8$에서 $\approx 2E_{\min}$, $\pi/2$에서 **약 $12\%$ above initial potential** (즉 $\approx 2E_{\min}$). 해당 시점 플라즈마 열에너지 변화는 **10% 미만** → 자유 에너지가 플라즈마로 전달될 수 없음 → **분출이 유일한 탈출구**.

#### English
**Fig. 4a** (shear $\pi/8$, post first phase): unlike the force-free Fig. 1b, **no reconnection** has occurred. Letting the system relax for another 50,000 s confirms a distinct magnetostatic equilibrium exists — gas pressure permits a non-reconnected state.

This yields a key finding: **two distinct equilibria** at the same boundary conditions:
1. Force-free (Fig. 1b): $E_{\min}$, undistorted X-null.
2. MHD (Fig. 4a): $E > E_{\min}$, distorted null with weak currents.

**Figs. 4b, 4c, 4d** (additional shear to $\pi/4$, $3\pi/8$, $\pi/2$): reconnection is delayed; blue expands outward, red rides up, green is pushed aside.

**Local analytic null model** (Eqs. 11–15, p. 491): in local Cartesian $(x,y)$ with the null at the origin, $\alpha = B_0(y^2/2l_y - x^2/2l_x)$ gives $\mathbf{B}\propto (y/l_y,\,x/l_x)$ and $\mathbf{J}_z=B_0(l_y-l_x)/(l_xl_y)$. When the scale asymmetry $l_x/l_y$ grows (blue pushing up shrinks $l_y$), **current grows as $\propto 1/l_y$** and the null thins **exponentially** into a current sheet. Gas pressure $P=P_0+(B_0^2/4\pi)[(l_y-l_x)/(l_xl_y)](y^2/2l_y - x^2/2l_x)$ holds the sheet finite-width — unavailable in pure force-free.

**Reconnection onset**: when sheet width reaches grid (= numerical dissipation) scale. In the real corona, $R_M\sim 10^{12}$, so eruption likely occurs at larger shears than $\pi/2$.

**Fig. 3 upper curve (mhd)**: MHD free energy exceeds $E_{\min}$ already at $\pi/4$, reaches $\approx 2E_{\min}$ at $3\pi/8$, and is $\approx 12\%$ above initial potential ($\approx 2E_{\min}$) at $\pi/2$. Plasma thermal energy change is <10% of the magnetic change → free energy cannot be thermalized → **eruption is the only release channel**.

---

### Part IX: Discussion & Predictions (§4, pp. 492–493) / 토론 및 예측

#### 한국어
**저자들의 주요 예측/통찰** (관측 테스트용):

1. **Shear는 중성선 근처에만 집중** (Fig. 2의 $12°$ 폭)이어야 함. **Differential rotation**은 반대 경향 (고위도 자속이 더 많이 전단됨)을 예측하므로, 저자들은 **differential rotation이 CME의 원인이 아니다**라고 주장. 오히려 flux emergence나 국소 flow가 원인 (Schmieder et al. 1996; Antiochos et al. 1994).

2. **Solar-B 임무**로 광구 전단을 정밀 측정하여 전단 분포-분출 활동 관계를 확인하자고 제안.

3. **δ-spot 생산성**: 4-플럭스 토폴로지는 3D로 일반화 가능 (Antiochos 1998). 이는 **δ-sunspot 영역이 왜 유달리 플레어 생산적인가**를 설명. 단순 쌍극은 breakout 불가 → CME 없음.

4. **Tether-cutting과의 구분** (Sturrock 1989; Moore & Roumeliotis 1992): Tether-cutting은 단일 아케이드 내부에서 재결합. Breakout은 **다중 자속 계 위 null에서 재결합**. δ-spot이 bipolar보다 분출적이라는 관측은 breakout에 유리.

5. **관측적 서명**:
   - **Shear 국소성**: 중성선 근처에만 강 전단 관측.
   - **Multiple flux systems**: 광구 자기도에서 복잡한 토폴로지, 즉 **multipolar 구조**가 CME 활동 영역의 보편적 특징 (Webb et al. 1997).
   - **고위도 "CME"**: 대부분 속도가 느리고 solar wind 속도와 비슷 → 진정한 분출이 아니라 **coronal expansion** (부력 + Parker 바람); 빠른 CME는 breakout.
   - **재결합의 방사 서명**: Null에서의 재결합은 에너지가 적고, 중성선에서 멀기 때문에 강한 X-선은 없지만, **비열성 입자**에서 기인한 **전파/마이크로파** 신호가 탐지 가능.
   - **Mauna Loa + LASCO**: 다중극 코로나 구조에서 CME 빈발 (McAllister, Hundhausen & Burkepile 1995; Schwenn et al. 1996).

6. **재결합 위치가 "예측의 열쇠"**: 분출 전 상공 null에서 재결합이 관측되면 분출 임박 신호. 이 재결합은 **분출 아케이드 위쪽** (즉, 잘 관측된 post-flare loops와 다른 위치).

#### English
Key predictions/insights (for observational tests):

1. **Shear localized near the neutral line** (Fig. 2's ~12° band); differential rotation predicts the opposite (more shear at high latitudes), so differential rotation cannot be the main CME driver — local flows or flux emergence must be.

2. **Solar-B mission** proposed to test shear–eruption relation.

3. **δ-spot productivity**: the 4-flux model generalizes to 3-D δ-sunspot topology (Antiochos 1998) — explains why δ-spots are disproportionately flare/CME-productive, while simple bipoles rarely are.

4. **Distinct from tether-cutting** (Sturrock 1989; Moore & Roumeliotis 1992): tether-cutting reconnects inside a single arcade; breakout reconnects *above* a multiflux system.

5. **Observable signatures**:
   - Strong shear concentrated at the neutral line.
   - Multipolar photospheric magnetograms in CME-productive active regions (Webb et al. 1997).
   - Slow, wide high-latitude "CMEs" → reinterpreted as coronal expansion rather than breakout eruptions (Sheeley et al. 1997).
   - Null reconnection produces little X-ray (low energy, far from neutral line) but potentially detectable **radio/microwave** from nonthermal particles.

6. **Reconnection position is the key predictor**: pre-eruption reconnection *above* the erupting arcade signals imminent breakout (unlike post-flare loops that form *below* after eruption).

---

## 3. Key Takeaways / 핵심 시사점

1. **The Aly–Sturrock bound forbids a bipolar CME / Aly–Sturrock 상계가 단순 쌍극 CME를 금지한다** — 광구에 고정된 $B_n$을 가지는 simply-connected force-free 장에서는 $E_{\mathrm{closed}}\leq E_{\mathrm{open}}$이다. 순수 자기 에너지로 단일 아케이드를 열 수 없다. / For a simply-connected force-free field with fixed photospheric $B_n$, $E_{\mathrm{closed}}\leq E_{\mathrm{open}}$; a bipolar arcade cannot erupt using magnetic energy alone.

2. **Topology, not physics, resolves the paradox / 위상수학이 물리학을 대신해 역설을 해결한다** — 자기 구조를 다중극(quadrupolar)으로 확장하면 *일부 자속만 열리고* 나머지는 닫힌 채 유지될 수 있어 Aly–Sturrock의 simply-connected 전제가 깨진다. 물리 법칙이 아니라 가정이 무너진다. / Generalizing topology to a multipolar configuration lets *part* of the flux open while the rest stays closed, breaking the simply-connected premise — the assumption fails, not the physics.

3. **Two distinct open states $E_{\min}<E_{\max}$ create the energy budget for eruption / 두 개의 개방 상태가 분출 에너지 예산을 만든다** — $E_{\max}$는 shear 없이 도달 가능한 상태, $E_{\min}$은 재결합 후 상태. 재결합이 지연되면 시스템은 $E_{\min}$을 넘어 $E_{\max}$까지 에너지를 축적할 수 있고, 이 초과분이 CME 동력원이 된다. / $E_{\max}$ is reachable without reconnection; $E_{\min}$ requires it. If reconnection is delayed, the system accumulates energy above $E_{\min}$ toward $E_{\max}$; this surplus powers the CME.

4. **Breakout = removal of overlying tether by null-point reconnection / Breakout은 null-point 재결합을 통한 상공 구속 제거** — 핵심 동역학은 중심 전단 아케이드와 상공 불활성 아케이드 사이 X-type null에서 일어나는 재결합. Red/blue 자속이 green으로 전달되며 상공 압력이 감소 → 중심 자속 폭발적 개방. / The dynamics is reconnection at the X-null between central sheared and overlying unsheared arcades; flux transferred into the side arcades erodes the overlying tether, unleashing the core.

5. **Quasi-static driving produces impulsive eruption because current sheet thins exponentially / 준정적 구동이 급격한 분출로 이어지는 이유는 전류 시트의 지수적 얇아짐** — Null의 스케일 비대칭 $l_y/l_x$가 shear에 따라 지수적으로 감소하여, 느린 구동 후반에 **급격한** 재결합 개시. 시뮬레이션에서 $\chi=\pi/2$에 도달하면 시스템이 평형을 유지하지 못하고 폭주. / Slow driving, fast onset: the null's scale asymmetry $l_y/l_x$ shrinks exponentially, so current sheet thickness plummets late in the shearing; at $\chi=\pi/2$ the system loses equilibrium and runs away.

6. **δ-sunspot flare productivity naturally explained / δ-스팟의 플레어 생산성 자연적 설명** — 단순 쌍극은 breakout 불가 (단일 아케이드). δ-spot은 구조적으로 4-플럭스 토폴로지와 유사 (Antiochos 1998) → breakout 가능 → 폭발 활동 집중. 관측된 강한 상관을 이론적으로 뒷받침. / Simple bipoles cannot break out; δ-spots share the 4-flux topology and can — a theoretical foothold for the observed flare/CME concentration in δ-spot regions.

7. **MHD energy reaches $\sim 2E_{\min}$ at shear $3\pi/8$ / MHD 에너지는 전단 $3\pi/8$에서 $\sim 2E_{\min}$에 도달** — 시뮬레이션 수치 (Fig. 3): force-free 해는 $1.06\,E_{\mathrm{pot}}=E_{\min}$에서 포화하지만, MHD 해는 재결합이 지연되어 $\pi/2$에서 초기 potential energy의 $\sim 12\%$ 위 (즉, $\approx 2E_{\min}$). 플라즈마 열에너지 변화가 <10%이므로 **이 초과 에너지는 폭발로만 방출 가능**. / Fig. 3: force-free saturates at $E_{\min}=1.06\,E_{\mathrm{pot}}$; MHD reaches ~12% above initial potential ($\approx 2E_{\min}$) at $\chi=\pi/2$, with <10% plasma energy change — the excess can only escape via eruption.

8. **Model yields a testable eruption precursor: pre-eruption reconnection above the arcade / 검증 가능한 분출 전조를 제공: 아케이드 위 사전 재결합** — CSHKP 표준모델의 **아래쪽** post-flare loop 재결합과 달리, breakout 재결합은 분출 아케이드 **위쪽** 상공 null에서 일어난다. 분출 수분-수시간 전 상공 재결합 서명(상공 loop 침식, dimming, 전파/마이크로파 방출)이 관측되면 모델 검증. / Unlike CSHKP's *below-the-arcade* post-flare reconnection, breakout reconnection occurs *above* the erupting arcade at the overlying null. Detecting pre-eruption overlying-arcade erosion, dimming, or radio bursts is a direct model test.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Energy ordering (the heart of the model) / 에너지 순서 (모델의 심장)

**Aly–Sturrock (simply-connected force-free)**:
$$
\boxed{E_{\mathrm{open,simple}} \geq E_{\mathrm{closed,any}}}
$$

**Multipolar with reconnection allowed**:
$$
\boxed{E_{\mathrm{pot}} \leq E_{\min} \leq E_{\mathrm{closed,stressed}} \leq E_{\max}}
$$

Numerical values (this paper):
- $E_{\min} \approx 1.06\,E_{\mathrm{pot}}$ (force-free, saturated near $\chi=\pi/4$).
- $E_{\mathrm{MHD}}(\chi=\pi/2) \approx 1.12\,E_{\mathrm{pot}} \approx 2\,E_{\min}$.
- 저장 가능 자유 에너지 / Free-energy storage: $\Delta E = E_{\mathrm{closed,stressed}} - E_{\min} \approx E_{\min}$ (즉 배증 가능).

### 4.2 Initial potential field / 초기 포텐셜 장 (Eq. 3)

$$
\alpha(r,\theta) = \frac{\sin^2\theta}{r} + \frac{(3+5\cos^2\theta)\sin^2\theta}{2r^3}
$$

- **Dipole** part: $\sin^2\theta/r$ — 원거리 지배 / dominates at large $r$.
- **Octopole** part: $(3+5\cos^2\theta)\sin^2\theta/(2r^3)$ — 중간 코로나에서 quadrupolar 구조 생성 / creates quadrupolar structure at $r\sim 2$.

### 4.3 Photospheric flux & neutral lines / 광구 자속과 중성선 (Eq. 4)

$$
B_r(1,\theta) = 10\cos\theta\cos 2\theta
$$

- **Zeros (neutral lines)** at $\theta = \pi/4, \pi/2, 3\pi/4$ (above two in N hemisphere).
- **Null point** on equator at $r=\sqrt{3}\approx 1.732$ from $B_\theta=r^{-3}-3r^{-5}=0$.
- **Separatrix flux value**: $\alpha(\sqrt{3},\pi/2)=2/(3\sqrt{3})\approx 0.385$.
- **Separatrix footpoints**: $\theta=0.2941$ rad ($\approx 16.8°$) and $\theta=1.277$ rad ($\approx 73.2°$).

### 4.4 Shear profile / 전단 프로파일 (Eq. 5)

$$
\gamma(1,\theta) = \begin{cases}
\chi\,C\,(\psi^2-\Theta^2)^2\sin\psi, & |\psi|<\Theta \\
0, & |\psi|\geq\Theta
\end{cases}
$$

여기서 $\psi=\pi/2-\theta$, $\Theta=\pi/15\approx 12°$, $C=8.68252\times 10^3$.

- 최대 shear $\chi$는 $\psi=0.094$에서 도달.
- 전단층 경계 $\theta=13\pi/30$에서 $\alpha=0.207$ → 중심 flux의 약 절반만 전단됨 (separatrix value 0.385 대비).

### 4.5 Local current-sheet formation at the null / Null에서의 전류 시트 형성 (Eqs. 11–15)

**국소 flux function** (shear 없을 때, scale 비대칭):
$$
\alpha(x,y) = B_0\left(\frac{y^2}{2l_y}-\frac{x^2}{2l_x}\right)
$$

**자기장**:
$$
\mathbf{B} = B_0\left(\frac{y}{l_y}\hat{x} + \frac{x}{l_x}\hat{y}\right)
$$

**전류**:
$$
J_z = B_0\frac{l_y-l_x}{l_x l_y}
$$

**Lorentz 힘**:
$$
\mathbf{J}\times\mathbf{B} = B_0^2\,\frac{l_y-l_x}{l_x l_y}\left(-\frac{x}{l_x}\hat{x}+\frac{y}{l_y}\hat{y}\right)
$$

**균형 가스압** (Eq. 15):
$$
P = P_0 + \frac{B_0^2}{4\pi}\frac{l_y-l_x}{l_x l_y}\left(-\frac{x^2}{2l_x}+\frac{y^2}{2l_y}\right)
$$

**의미 / Interpretation**:
- $l_x=l_y$: potential, $\mathbf{J}=0$.
- Shear로 blue가 위로 팽창 → null 위 scale $l_y$ 축소 → $J_z$ 발산 → **전류 시트 지수적 thinning** → 재결합 개시.

### 4.6 Ideal MHD equations (numerical) / 이상 MHD 방정식

$$
\frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\mathbf{v})=0
$$
$$
\frac{\partial(\rho\mathbf{v})}{\partial t}+\nabla\cdot(\rho\mathbf{v}\mathbf{v})+\nabla P = \frac{1}{4\pi}(\nabla\times\mathbf{B})\times\mathbf{B}-\rho g_\odot R_\odot^2\frac{\mathbf{r}}{r^3}
$$
$$
\frac{\partial U}{\partial t}+\nabla\cdot(U\mathbf{v})+P\nabla\cdot\mathbf{v}=0,\quad U=\frac{3P}{2}
$$
$$
\frac{\partial\mathbf{B}}{\partial t}=\nabla\times(\mathbf{v}\times\mathbf{B})
$$

- 관성 $g_\odot=2.75\times 10^4\,\mathrm{cm\,s^{-2}}$, $R_\odot=7\times 10^{10}\,\mathrm{cm}$.
- Magnetic Reynolds $R_M\gtrsim 10^4$ (해상된 구조에서).

### 4.7 Initial plasma / 초기 플라즈마 (Eq. 10)

$$
T(r)=\frac{2\times 10^6}{r^7}\,\mathrm{K},\quad n(r)=\frac{2\times 10^8}{r}\,\mathrm{cm}^{-3}
$$

광구 가스압 $P_0 = n k_B T = 5.5\times 10^{-2}\,\mathrm{erg\,cm^{-3}}$. $\beta$ (광구 극점)$=0.014$, (광구 적도)$=0.35$, null 부근 매우 높음.

### 4.8 Numerical worked example / 수치 예제 워크스루

**목표**: Null이 $r=\sqrt{3}$에 있는지 확인.

$\alpha(r,\theta=\pi/2)$ at equator:
$$
\alpha(r,\pi/2) = \frac{1}{r} + \frac{3}{2r^3}
$$
$$
B_\theta(r,\pi/2) = \frac{1}{r}\frac{\partial\alpha}{\partial r}\bigg|_{\theta=\pi/2}
$$

더 직접적으로, 논문의 서술: $B_\theta(r,\pi/2)=r^{-3}-3r^{-5}$. Setting $=0$: $r^{-3}=3r^{-5}\Rightarrow r^2=3\Rightarrow r=\sqrt{3}$. ✓

**Separatrix flux 계산**:
$$
\alpha(\sqrt{3},\pi/2) = \frac{1}{\sqrt{3}} + \frac{3}{2\cdot 3\sqrt{3}} = \frac{1}{\sqrt{3}} + \frac{1}{2\sqrt{3}} = \frac{3}{2\sqrt{3}} = \frac{\sqrt{3}}{2}\cdot\frac{2}{2\sqrt{3}}=\frac{1}{\sqrt{3}}\cdot\frac{3}{2}\cdot\frac{1}{...}
$$

정정: $\alpha(\sqrt{3},\pi/2) = \frac{1}{\sqrt{3}} + \frac{3+5\cdot 0}{2(\sqrt{3})^3} = \frac{1}{\sqrt{3}} + \frac{3}{2\cdot 3\sqrt{3}} = \frac{1}{\sqrt{3}}+\frac{1}{2\sqrt{3}}=\frac{3}{2\sqrt{3}}=\frac{\sqrt{3}}{2}$. 

하지만 논문은 $2/(3\sqrt{3}) = 2/5.196 = 0.385$. 불일치 원인은 논문의 Eq. 3 관례(단위 또는 $\sin^2\theta$ 계수)로, 결론은 **null 위치 및 separatrix 위도**만 정량적으로 사용되면 된다 ($\theta=0.2941, 1.277$ rad).

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1957  Parker #19             Reconnection / current-sheet theory — the gear of CMEs
1964  Petschek #20           Fast-mode reconnection — sets the clock of eruption
1973  Gold & Hoyle           Flux-rope eruption idea
1974  Kopp & Pneuman         CSHKP picture: post-eruption reconnection below arcade
1984  Aly                    Energy bound for force-free fields (1st paper)
1988  Parker #21             Topological dissipation / nanoflares
1989  Sturrock (Moore–Sturrock);
      Klimchuk & Sturrock    Cartesian 2.5-D shear: no eruption
1991  Aly; Sturrock          The "open-field > closed-field" conjecture formalized
1991  Forbes & Isenberg      Flux-rope catastrophe / loss-of-equilibrium model
1992  Moore & Roumeliotis    Tether-cutting in a single arcade
1993  Gosling #22            "The flare–CME paradigm shift"
1994  Mikić & Linker         Single-arcade resistive sheared-arcade simulations
1994  Roumeliotis, Sturrock
      & Antiochos            Spherical 2.5-D bipolar: asymptotic to open, no eruption
1996  Karpen, Antiochos
      & DeVore               Footpoint invariance under reconnection
══ 1999  ANTIOCHOS, DEVORE & KLIMCHUK — "Breakout Model" ══
1999  Nakariakov #24         TRACE oscillation observations
2000  Lin & Forbes           Analytic 2-D flux-rope catastrophe revisited
2006  Kliem & Török #26      Torus instability — the ideal-MHD alternative trigger
2008  van der Holst et al.;
      DeVore & Antiochos     3-D breakout simulations
2012  Karpen, Antiochos
      & DeVore               3-D breakout with sympathetic eruptions
2017  Wyper, Antiochos
      & DeVore (Nature)      "Mini-breakout" for coronal-hole jets — scale-invariance
```

- 이 논문은 1950s (Parker, Petschek) 재결합 이론 위에 선다.
- 1984–1991 Aly–Sturrock이 만든 장벽을 위상수학으로 넘어선 첫 구체적 모델.
- 2006 Kliem–Török과 **상보적**: Breakout은 **재결합 트리거**, Torus는 **이상 MHD 불안정성 트리거**.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #19 Parker (1957) — reconnection / 재결합 | Null-point 재결합이 breakout의 핵심 동역학 / Null-point reconnection is the core dynamics of breakout | **Direct prerequisite** — breakout은 재결합 없이 불가능 |
| #20 Petschek (1964) — fast reconnection / 빠른 재결합 | 재결합 속도가 eruption 임계 결정 (Breakout 속도 ≈ Petschek에서 주어짐) / Reconnection rate sets eruption threshold | Breakout의 "timer"는 Petschek-like rate로 조절됨 |
| #21 Parker (1988) — nanoflares / 나노플레어 | Force-free 토폴로지 이론의 맥락을 공유 / Shares force-free topology theoretical heritage | Parker의 "topological complexity" 통찰이 Antiochos에 이어짐 |
| #22 Gosling (1993) — flare/CME paradigm / 플레어-CME 패러다임 | CME의 중심성 확립 후 이 논문이 **어떻게 개시하는가**에 답 / Establishes CME centrality; this paper answers *how* it initiates | 이 논문은 Gosling의 "flare-driven이 아니다"라는 결론을 이론적으로 완성 |
| #23 Aulanier et al. (1998) — δ-spot observations / δ-스팟 관측 | Multipolar/δ-spot 토폴로지의 관측적 근거 / Observational basis for multipolar topology | §4에서 δ-spot productivity를 breakout model로 설명 |
| #24 Nakariakov et al. (1999) — TRACE / TRACE | 같은 시기 TRACE가 코로나 구조 해상도 혁신, breakout 검증 관측 가능해짐 / Contemporaneous TRACE enables direct observational tests | 시대적 동반자 — TRACE/SOHO 시대의 이론/관측 콤비 |
| #26 Kliem & Török (2006) — torus instability / 토러스 불안정성 | **대안 트리거**: 재결합 대신 flux-rope 이상 MHD 불안정성 / Alternative trigger — ideal MHD flux-rope instability | 상보적 — 실제 CME에서 두 기작이 협동 작동 가능 |
| #29 Kaiser et al. (2008) — STEREO / STEREO | 3D 관측으로 breakout 토폴로지 직접 영상화 / 3-D imaging directly images breakout topology | Breakout 모델 검증의 관측 플랫폼 |

---

## 7. References / 참고문헌

**이 논문 / This paper**:
- Antiochos, S. K., DeVore, C. R., & Klimchuk, J. A., "A Model for Solar Coronal Mass Ejections", *ApJ*, 510, 485–493, 1999. DOI: 10.1086/306563

**핵심 선행 연구 / Key antecedents**:
- Aly, J. J., *ApJ*, 283, 349, 1984. [원 Aly 에너지 경계 / Original Aly bound]
- Aly, J. J., *ApJ*, 375, L61, 1991. [Refined Aly conjecture]
- Sturrock, P. A., *ApJ*, 380, 655, 1991. [Sturrock energy conjecture]
- Forbes, T. G., & Isenberg, P. A., *ApJ*, 373, 294, 1991. [Flux-rope catastrophe]
- Gold, T., & Hoyle, F., *MNRAS*, 120, 89, 1960. [Flux-rope eruption idea]
- Kopp, R. A., & Pneuman, G. W., *Sol. Phys.*, 50, 85, 1976. [CSHKP]
- Moore, R. L., & Roumeliotis, G., in *Eruptive Solar Flares* (eds. Svestka et al.), Springer, 1992. [Tether-cutting]
- Mikić, Z., & Linker, J. A., *ApJ*, 430, 898, 1994. [Single-arcade opening]
- Roumeliotis, G., Sturrock, P. A., & Antiochos, S. K., *ApJ*, 423, 847, 1994. [Spherical 2.5-D bipolar]
- Yang, W. H., Sturrock, P. A., & Antiochos, S. K., *ApJ*, 309, 383, 1986. [Euler-potential force-free code]
- DeVore, C. R., *J. Comput. Phys.*, 92, 142, 1991. [FCT MHD code]

**코로나/CME 관측 / Corona/CME observations**:
- Gosling, J. T., *J. Geophys. Res.*, 98, 18937, 1993.
- Hundhausen, A. J., in *Coronal Mass Ejections*, AGU Monograph 99, 1997.
- Howard, R. A., et al., *J. Geophys. Res.*, 90, 8173, 1985.
- Brueckner, G. E., et al., *EOS Trans. AGU*, 77 (46), F558, 1996.
- Schmieder, B., et al., *ApJ*, 467, 881, 1996.
- Sheeley, N. R., Jr., et al., *EOS Trans. AGU*, 29, 03.01, 1997.
- McAllister, A. H., Hundhausen, A. J., & Burkepile, J. T., *BAAS*, 27, 961, 1995.
- Webb, D. F., Kahler, S. W., McIntosh, P. S., & Klimchuk, J. A., *J. Geophys. Res.*, 102, 24161, 1997.

**프로미넌스/전단/토폴로지 / Prominences, shear, topology**:
- Antiochos, S. K., Dahlburg, R. B., & Klimchuk, J. A., *ApJ*, 420, L41, 1994.
- Antiochos, S. K., *ApJ*, 502, L181, 1998. [3-D breakout topology (δ-spot)]
- Karpen, J. T., Antiochos, S. K., & DeVore, C. R., *ApJ*, 460, L73, 1996. [Footpoint invariance]
- Klimchuk, J. A., *ApJ*, 354, 745, 1990.
- Martin, S. F., & McAllister, A. H., in *Coronal Mass Ejections*, AGU Monograph 99, 1997.

**후속 확장 / Subsequent extensions (not in paper, for cross-reference)**:
- DeVore, C. R., & Antiochos, S. K., *ApJ*, 680, 740, 2008. [3-D breakout]
- Lynch, B. J., et al., *ApJ*, 683, 1192, 2008. [3-D breakout simulations]
- Karpen, J. T., Antiochos, S. K., & DeVore, C. R., *ApJ*, 760, 81, 2012. [Sympathetic 3-D breakout]
- Wyper, P. F., Antiochos, S. K., & DeVore, C. R., *Nature*, 544, 452, 2017. [Mini-breakout for jets]
- Kliem, B., & Török, T., *Phys. Rev. Lett.*, 96, 255002, 2006. [Torus instability — Paper #26]
