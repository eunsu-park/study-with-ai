---
title: "Torus Instability"
authors: "Bernhard Kliem, Tibor Török"
year: 2006
journal: "Physical Review Letters"
doi: "10.1103/PhysRevLett.96.255002"
topic: Solar_Physics
tags: [torus_instability, CME, flux_rope, decay_index, MHD_instability, ideal_MHD, hoop_force]
status: completed
date_started: 2026-04-27
date_completed: 2026-04-27
---

# Torus Instability — Reading Notes / 읽기 노트

> Kliem, B., & Török, T., "Torus instability," *Phys. Rev. Lett.* **96**, 255002 (2006). 4 pages, 3 figures.

---

## 1. Core Contribution / 핵심 기여

저-베타(low-β) 자기 플라즈마에서 Shafranov 평형을 이루는 토로이달 전류 고리의 팽창 안정성을 다시 분석하여, **외부 포텐셜 자기장 $B_{\rm ex}\propto R^{-n}$ 의 감쇠 지수 $n$ 이 임계값 $n_{\rm cr}=3/2-1/(4c_0)$ 를 넘으면 고리가 자유 팽창 불안정** 임을 보였다. 여기서 $c_0=\ln(8R_0/b_0)-2+l_i/2$ 는 종횡비와 전류 분포에 약하게 의존하는 상수이며, 코로나에서 흔한 $R_0/b_0\sim 10$, $l_i=1/2$ 에서 $n_{\rm cr}\approx 1.4$ 이다. Bateman (1978) 의 고전적 결과 $n_{\rm cr}=3/2$ 는 외부 자속이 보존된다는 가정($\Psi_{\rm ex}={\rm const}$) 의 한계로 회복된다. 더 나아가 footpoint 가 광구에 묶여 전류가 보존되는($I={\rm const}$) 시나리오와 자유 팽창 시나리오 모두에 대해 가속 프로파일과 점근 속도를 해석적으로 유도하고, 결과를 spheromak 분출 실험 및 빠른/느린 CME 의 핵심 관측 특성과 정량적으로 비교했다.

The Letter re-analyzes the expansion stability of a toroidal current ring in Shafranov equilibrium within a low-β magnetized plasma and shows that **a freely expanding ring is unstable when the decay index $n$ of the external poloidal field $B_{\rm ex}\propto R^{-n}$ exceeds $n_{\rm cr}=3/2-1/(4c_0)$**, where $c_0=\ln(8R_0/b_0)-2+l_i/2$ depends weakly on aspect ratio and current profile. For coronal values $R_0/b_0\sim 10$ and $l_i=1/2$, $n_{\rm cr}\approx 1.4$. Bateman's (1978) classical result $n_{\rm cr}=3/2$ is recovered in the limit of conserved external flux ($\Psi_{\rm ex}={\rm const}$, $c_0\to\infty$). The authors derive the acceleration profile and asymptotic velocity analytically for both the freely expanding case and the photosphere-anchored constant-current case, and show qualitative agreement with spheromak expansion experiments and the essential properties of solar CMEs, unifying the apparently disparate fast and slow CME populations.

---

## 2. Reading Notes / 읽기 노트 (Section by Section)

### 2.1 Introduction & Bateman recap (p. 1) / 도입 및 Bateman 복습

저자들은 토로이달 전류 고리(plasma ring)의 평형이 **Shafranov(1966)** 에 의해 정립되었음을 상기시킨다. 이 평형은 (i) **hoop force** (전류 채널의 곡률에서 비롯되는 $\nabla(L^{-1})$ 항, $\propto I^2/R$), (ii) bent channel 의 **net pressure gradient** (내부 자기 압력의 비대칭) 둘 다 반지름 방향 외향이며, 이를 (iii) 외부 포텐셜 자기장 $B_{\rm ex}$ 가 만드는 인장 로렌츠 힘 ($-IB_{\rm ex}/(\pi b^2)\cdot 2\pi R$) 이 균형 잡는 구도이다. **Bateman (1978)** 은 perturbation $dR>0$ 가 있을 때, $B_{\rm ex}\propto R^{-n}$ 가 hoop force ($\propto R^{-1}$) 보다 빠르게 줄어들면 균형이 깨져 고리가 팽창한다는 것을 직관적으로 보였고 이를 정식화하여 $n>3/2$ 라는 조건을 얻었다.

The authors recall that the equilibrium of a plasma current ring was established by **Shafranov (1966)**. The equilibrium balances (i) the **hoop force** arising from the channel's curvature ($\propto I^2/R$, originating in $-\nabla L^{-1}$ of the self-inductance), (ii) the **net pressure gradient** of the bent current channel (internal magnetic pressure asymmetry across the curvature), both of which point radially outward, against (iii) the restoring Lorentz force $-IB_{\rm ex}/(\pi b^2)\cdot 2\pi R$ from the external poloidal field. **Bateman (1978)** observed that on perturbation $dR>0$, if $B_{\rm ex}\propto R^{-n}$ decreases faster than the hoop force ($\propto R^{-1}$), equilibrium fails and the ring expands; he formalized this into the criterion $n>3/2$.

핵심 코멘트:
- TI 는 **헬리컬 킹크(kink)** 와 달리 toroidal 자기장 성분이 있어도 안정화되지 않는다. (kink 는 $B_T$ 가 있으면 억제됨.)
- 융합 장치는 작은 $n$ 과 벽의 영상 전류로 TI 를 억제하지만, 천체 플라즈마(특히 태양 코로나, spheromak 분출 실험)에서는 이 안정화가 없을 수 있다.
- Titov & Démoulin (1999) 는 line-tied flux rope 에서 $n>2$ 라는 추정치를 제시했고, 그 후 TI 자체는 거의 재검토되지 않았다.

Key remarks:
- Unlike the **helical kink**, the TI is *not* stabilized by a toroidal field component (the kink is suppressed when a force-free toroidal $B_T$ is present, because the hoop force still points outward).
- Fusion devices suppress the TI through small $n$ and image currents in conducting walls; astrophysical plasmas (notably the solar corona and spheromak expansion experiments) lack such stabilization.
- Titov & Démoulin (1999) estimated $n>2$ for a line-tied flux rope; the TI itself had been little revisited since.

### 2.2 Setup of the model / 모형 설정

Kliem & Török 은 **두 가지 일반화 시나리오** 를 다룬다.

(A) **Freely expanding ring**: laboratory plasmas, spheromak expansion, and CMEs in the relevant initial stage. 큰 종횡비 한계 ($R\gg b$) 에서 hoop force 와 외부 자기장 인장력만 남기고 중력·압력·toroidal 외부장은 무시한다. (저-베타 가정으로 정당화.)

(B) **Expanding ring with fixed total current $I=I_0$**: 광구에 묶인 발(footpoint anchoring) 의 효과를 포착. CME 초기 단계에서 reconnection 으로 인한 흐름이 충분치 않을 때 $I$ 가 거의 일정하게 유지될 수 있음.

Two generalized scenarios are considered: (A) a **freely expanding ring** (laboratory, spheromak, and the relevant initial stage of CMEs) and (B) an expanding ring with **fixed total current $I=I_0$**, which captures footpoint anchoring at the photosphere during the early CME phase before reconnection releases the constraint. In the large aspect-ratio limit ($R\gg b$), only the hoop force and the restoring Lorentz force from $B_{\rm ex}$ are retained; gravity, plasma pressure, and any toroidal external field are neglected, justified for low-β plasmas.

**힘 균형 (Eq. 1) / Force balance (Eq. 1)**:

$$\boxed{\;\rho_m\frac{d^2 R}{dt^2}=\frac{I^2}{4\pi^2 b^2 R^2}\Big(L+\frac{\mu_0 R}{2}\Big)-\frac{I\,B_{\rm ex}(R)}{\pi b^2}\;}$$

여기서 $L=\mu_0 R[\ln(8R/b)-2+l_i/2]$ 는 ring 의 자체-인덕턴스. $l_i=1/2$ 는 균일 전류 분포 가정.

**Flux conservation (Eq. 2) / 자속 보존 (식 2)**:

$$\Psi=\Psi_I+\Psi_{\rm ex}=LI-2\pi\int_0^R B_{\rm ex}(r)\,r\,dr={\rm const}$$

이상-MHD 에서 봉입 자속은 보존되어야 하므로, 외부장 프로파일 $B_{\rm ex}(R)$ 가 주어지면 $I(R)$ 가 결정된다.

**Ansatz**: $B_{\rm ex}(R)=\hat B R^{-n}$ for $R\geq R_0$ (활성 영역 위쪽 영역에서). 이것은 표준 자기 쌍극자 또는 다중 극자가 만드는 외부 포텐셜장의 거듭제곱 근사이다.

The ansatz $B_{\rm ex}(R)=\hat B R^{-n}$ for $R\geq R_0$ is the standard power-law approximation of an external potential field above a magnetic dipole or higher multipole. With this profile, flux conservation (Eq. 2) determines $I(R)$:

$$I(R)=\frac{c_0 R_0 I_0}{cR}\left\{1+\frac{c_0+1/2}{2c_0}\frac{1}{2-n}\left[\left(\frac{R}{R_0}\right)^{2-n}-1\right]\right\},\quad n\neq 2.\tag{3}$$

여기 $c=L/(\mu_0 R)$, $c_0=c|_{R=R_0}=\ln(8R_0/b_0)-2+l_i/2$.

### 2.3 The reduced ODE & instability condition (Eq. 4–5) / 축약 ODE 와 불안정 조건

$\rho=R/R_0$, $\tau=t/T$ 로 무차원화. 시간 스케일

$$T=\left(\frac{c_0+1/2}{4}\frac{b_0^2}{B_{\rm eq}^2/\mu_0\rho_{m0}}\right)^{1/2}=\frac{(c_0+1/2)^{1/2}}{2}\frac{b_0}{V_{Ai}}$$

는 minor radius 에 대한 **하이브리드 Alfvén 시간** 이다.

$c(R)\approx c_0$ 가정 (logarithmic 변화이므로 좋은 근사) 아래 식 (1) 은

$$\boxed{\;\frac{d^2\rho}{d\tau^2}=\frac{c_0^2}{(c_0+1/2)c}\rho^{-2}\!\left[1+\frac{c_0+1/2}{c_0}\frac{\rho^{2-n}-1}{2(2-n)}\right]\!\!\left\{\frac{c+1/2}{c}\!\left[1+\frac{c_0+1/2}{c_0}\frac{\rho^{2-n}-1}{2(2-n)}\right]-\frac{c_0+1/2}{c_0}\rho^{2-n}\right\}\;}\tag{4}$$

이 된다.

평형 ($\rho=1$) 부근에서 작은 변위 $\epsilon=\rho-1\ll 1$ 에 대해 좌변은 $\ddot\epsilon$, 우변은 $\epsilon$ 에 비례한다. **불안정 조건** $d(d^2\rho/d\tau^2)/d\rho|_{\rho=1}>0$ 에서

$$\boxed{\;n>n_{\rm cr}=\frac{3}{2}-\frac{1}{4c_0}\;}\tag{5}$$

이 도출된다. $c_0\to\infty$ 한계 ($\Psi_{\rm ex}={\rm const}$) 에서 Bateman 의 $n_{\rm cr}=3/2$ 회복.

**선형 단계 해 (Eq. 6) / Linear-stage solution (Eq. 6)**:

$$\epsilon(\tau)=\frac{v_0 T/R_0}{(n-n_{\rm cr})^{1/2}}\sinh\!\left((n-n_{\rm cr})^{1/2}\tau\right)$$

성장률 $\gamma=(n-n_{\rm cr})^{1/2}/T$. 거의 지수적이다 (작은 $\epsilon$ 동안).

The growth rate is $\gamma=(n-n_{\rm cr})^{1/2}/T$ in dimensional form, i.e. roughly the inverse of the hybrid Alfvén time multiplied by the supercriticality $(n-n_{\rm cr})^{1/2}$.

**점근 속도 (Eq. 7) / Asymptotic velocity (Eq. 7)**:

$$v_\infty=\left[\Big(\frac{v_0 T}{R_0}\Big)^2+\frac{2(2n-3+1/(2c_0))(n-1+1/(4c_0))}{(2n-3)(n-1)}\right]^{1/2}\approx [(v_0 T/R_0)^2+2]^{1/2},\;n>3/2.\tag{7}$$

차원화하면 $\sim\sqrt{2}(R_0/b_0)V_{Ai}$, 코로나 내측 알펜 속도 ($\sim 10^3$ km/s) 와 같은 정도. For $n_{\rm cr}<n<3/2$ 에서는 점근 속도가 발산하는데, 이는 큰 $\rho$ 영역에서 무시한 압력·외부 toroidal flux 의 누적 효과 때문이다.

### 2.4 Acceleration profiles & figures (Fig. 1) / 가속 프로파일과 그림 1

Fig. 1 은 $c(R)=c_0$, $R_0/b_0=10$, $v_0 T/R_0=0.005$, $l_i=1/2$ 에서 $n=1.5,\,2,\,3,\,4$ 의 $a(\rho)=d^2\rho/d\tau^2$ 와 $\rho(\tau)$, $v(\tau)$ 곡선을 보여준다.

- 가속도는 $\rho>n_{\rm cr}$ 부근에서 최대치를 가지며, $n$ 이 클수록 더 강하고 빠르게 증가.
- $n\gtrsim 2$ 에서는 $\rho\sim 2$ 이후 가속도가 빠르게 감소; $n\to n_{\rm cr}$ 부근에서는 천천히 감소.
- 결과적으로 팽창 $\rho(\tau)-1$ 은 $n\gtrsim 2$ 에서 "지수 $\to$ 선형" 패턴, $n\approx n_{\rm cr}$ 에서는 일정 가속도 곡선에 가깝다.

Figure 1 (with $c(R)=c_0$, $R_0/b_0=10$, $v_0 T/R_0=0.005$, $l_i=1/2$) shows the acceleration $a(\rho)=d^2\rho/d\tau^2$ and the time histories $\rho(\tau)$, $v(\tau)$ for $n=1.5,\,2,\,3,\,4$. The acceleration peaks at $\rho$ slightly above $n_{\rm cr}$, rises strongly with $n$, and decreases rapidly for $n\gtrsim 2$ but slowly for $n\to n_{\rm cr}$. As a consequence, the expansion $\rho(\tau)-1$ has an "exponential-to-linear" character for $n\gtrsim 2$ and a near-constant-acceleration profile for $n$ close to $n_{\rm cr}$.

이 $n$-의존성은 CME rise profile 의 관측 양상과 정확히 일치: **fast CMEs** 는 활성 영역 가까이에서 빠른 가속과 함께 알펜 속도($\sim 10^3$ km/s) 까지 $h\lesssim R_\odot/3$ 안에서 도달; **slow CMEs** 는 거의 일정 가속도로 $h\sim 30 R_\odot$ 까지 천천히 가속해 탈출 속도 정도($10^2$ km/s) 에 도달.

This matches CME rise profiles: **fast CMEs** reach $\sim 10^3$ km/s within $h\lesssim R_\odot/3$ above the photosphere with sharp acceleration (corresponding to $n\gtrsim 3$ at $h>D/2$ for bipolar regions with sunspot separation $D$), while **slow CMEs** accelerate roughly uniformly out to $h\sim 30 R_\odot$ before reaching $\sim 10^2$ km/s (corresponding to $n$ close to $n_{\rm cr}$, as expected for filaments far from active regions where $B\propto h^{-3/2}$).

### 2.5 The line-tied case (Eq. 8–10) / 라인-타잉 경우

발이 광구에 고정되어 있으면 reconnection 없이는 ring 전류가 방출되지 못하므로 $I(R)=I_0$ 이 더 적절. (3) 에서 $I(R)=I_0$ 로 치환하면

$$\boxed{\;\frac{d^2\rho}{d\tau^2}=\frac{1}{2(c_0+1/2)}+\frac{(2n-3)c_0+1/2}{2(n-2)(c_0+1/2)}\rho^{-1}-\frac{2n-3}{2(n-2)}\rho^{1-n}\;}\tag{8}$$

임계 감쇠 지수는

$$\boxed{\;n_{\rm cr}=\frac{3}{2}-\frac{1}{2(c_0+1)}\;}\tag{9}$$

자유 팽창과 거의 같다. 대신 가속도 프로파일이 더 강하게 증폭되며 더 큰 $\rho$ 영역까지 가속이 유지된다. CME 의 일부 관측이 가속 구간이 더 넓다는 사실과 일치 (Fig. 2).

For the line-tied case ($I=I_0$), substituting into the full equation gives Eq. (8). The critical decay index, Eq. (9), is only slightly smaller than in the freely expanding case; the dominant new effect is a **stronger amplification of the acceleration** that persists to larger $\rho$, in better agreement with some CME observations showing extended acceleration phases.

**Aspect-ratio evolution (Eq. 10) / 종횡비 진화 (식 10)**:

$$\frac{b(R)}{R}=\frac{8}{\exp\!\big\{c_0\rho^{-1}+\frac{c_0+1/2}{2(n-2)}\rho^{-1}(1-\rho^{2-n})+2-l_i/2\big\}},\;n\neq 2$$

Fig. 3 의 결과: $\rho\sim 10^1$–$10^2$ 에서 $b\sim R$ 이 되어 minor radius 가 major radius 만큼 빨리 팽창 (overexpansion) — CME 의 cavity 와 "three-part structure" (밝은 leading edge, 어두운 cavity, 안쪽 코어) 와 정합.

The minor radius eventually overexpands to $b\sim R$ at $\rho\sim 10^1$–$10^2$, consistent with the cavity structure of CMEs and the observed three-part morphology.

### 2.6 Application to fast/slow CMEs and δ-spots (p. 3, col. 2) / 빠른·느린 CME 와 δ 흑점 응용

- Bipolar 활성 영역에서 sunspot 분리 거리 $D\sim R_\odot/10$. **$h>D/2$ 부터 $n>3/2$, $h\gtrsim D$ 에서 $n\approx 3$.** 따라서 fast CME 는 활성 영역 위에서 짧은 가속 구간으로 알펜 속도까지 도달 가능.
- 활성 영역에서 멀리 떨어진 quiescent prominence 에서는 large-scale 자기장이 거의 $B\propto h^{-3/2}$ — 이미 코로나 낮은 곳부터 임계에 가깝다 → slow, near-uniform 가속.
- **$\delta$-spot**: 한 흑점 안에 반대 극성이 있어 quadrupolar. 가까운 두 극성 쌍이 만드는 외부장은 $h$ 에 따라 매우 빠르게 감소($n>3$). 동시에 강한 자기장 → 높은 알펜 속도($\sim$ 수 $10^3$ km/s) → **가장 강력한 분출이 이 영역에서 우선 발생** (Sammis, Tang, Zirin 1980 의 통계와 정합).

Fast CMEs originate from active regions where $D\sim R_\odot/10$: $n>3/2$ already at $h>D/2$ and approaches $n\approx 3$ at $h\gtrsim D$. Slow CMEs come from prominences far from active regions where the large-scale dependence $B\propto h^{-3/2}$ already dominates low in the corona. In **$\delta$-spot regions** (quadrupolar, with opposite-polarity pairs packed within a single sunspot), $n>3$ occurs very low in very strong fields — explaining why the most powerful eruptions originate preferentially from such regions (Sammis, Tang, & Zirin 1980).

### 2.7 Comparison with spheromak experiments / Spheromak 분출 실험 비교

Yee & Bellan (2000) 의 실험에서 spheromak-like torus 가 $\rho\lesssim 2$ 까지 거의 일정 속도로 팽창하는 것을 보였다. $B_{\rm ex}=0$ ($\Psi=L_0 I_0$) 가정으로 단순화한 식

$$\frac{d^2\rho}{d\tau'^2}=(c+1/2)c^{-2}\rho^{-2},\quad \tau'=(\pi/c_0)(b_0/\bar V_{Ai})t$$

에서 점근 팽창 속도 $((c+1/2)/c^2)^{1/2}R_0/T'\sim 5$–$16$ km/s 가 관측치 $\sim 5$ km/s 와 정합. ($\bar B\sim 300$ G, $N\sim 10^{15}$–$10^{16}$ cm$^{-3}$).

The simplified $B_{\rm ex}=0$ limit reproduces the observed near-constant expansion at $\rho\lesssim 2$ in spheromak experiments (Yee & Bellan 2000), with the asymptotic velocity $\sim 5$–$16$ km/s in acceptable agreement with the measured $\sim 5$ km/s for representative $\bar B\sim 300$ G and $N\sim 10^{15}$–$10^{16}$ cm$^{-3}$.

### 2.8 Conclusions / 결론

저자들은 TI 가 (i) catastrophe (Forbes & Isenberg/Priest), (ii) helical kink (Török & Kliem 2005) 와 함께 CME 메커니즘의 한 축이며, **medium-scale 팽창 ($\rho\lesssim 10^2$) 을 지배** 하고, fast/slow CME 의 **통일된 기술** 을 제공하며, three-part 구조와 spheromak 실험까지 자연스럽게 설명한다고 결론짓는다.

The authors conclude that the TI is one viable CME mechanism alongside catastrophe and helical kink, that it governs the medium-scale ($\rho\lesssim 10^2$) expansion, that it provides a unified description of fast and slow CMEs and a possible explanation of their three-part morphology, and that the same instability has occurred in spheromak expansion experiments.

---

## 3. Key Takeaways / 핵심 시사점

1. **임계 감쇠 지수 $n_{\rm cr}\approx 1.5$** — 외부 포텐셜 자기장이 $R^{-n}$ 로 감소하는 단순 가정 아래, 자유 팽창 토로이달 전류 고리는 $n>3/2-1/(4c_0)$ 일 때 이상-MHD 적으로 불안정. 이는 관측가능한 단일 진단량으로 즉시 응용 가능. / The simple criterion $n>3/2-1/(4c_0)\approx 1.5$ yields a single observable diagnostic (decay index) that maps directly onto magnetogram-based extrapolations.

2. **Hoop force ($\propto R^{-1}$) vs. tension ($\propto R^{1-n}$) 의 경쟁** — TI 는 본질적으로 두 멱법칙(power-law) 사이의 단순한 경쟁. $n=1$ (균일 외부장) 이면 외부장이 더 천천히 감소해 안정; $n>3/2$ 면 외부장이 더 빨리 감소해 불안정. / The TI reduces to a competition between two power-law forces with the hoop force ($\sim R^{-1}$) and the restoring tension ($\sim R^{1-n}$); the threshold appears at $n=3/2$ from the second-derivative analysis.

3. **이상-MHD 자속 보존이 $I(R)$ 을 결정** — Bateman 의 $\Psi_{\rm ex}={\rm const}$ 가정은 부분적이며, Kliem & Török 은 총 봉입 자속 $\Psi=\Psi_I+\Psi_{\rm ex}$ 를 일관되게 보존시켜 $I(R)$ 를 도출. 그 결과 $n_{\rm cr}$ 가 $1/(4c_0)$ 만큼 살짝 줄어듦. / Conserving total flux (rather than just $\Psi_{\rm ex}$) generates the small correction $-1/(4c_0)$ to Bateman's value and yields $I(R)$ self-consistently.

4. **Line-tying (footpoint anchoring)** 는 $n_{\rm cr}$ 를 거의 바꾸지 않지만 가속 프로파일을 강하게 증폭하고 가속 구간을 넓힘. / Line tying does not significantly change $n_{\rm cr}$ but strongly amplifies the acceleration and extends the radial range over which it acts, matching certain extended-acceleration CMEs.

5. **Fast vs. slow CME 통일** — fast CMEs 는 $\delta$-spot/active region 의 가파른 외부장 감소($n\gtrsim 3$, $V_{Ai}\sim 10^3$ km/s) 로, slow CMEs 는 $B\propto h^{-3/2}$ 의 quiescent prominence 영역으로 자연스럽게 설명. / The TI provides a unified mechanism: fast CMEs from steep-decay active regions, slow CMEs from large-scale ($n\sim 3/2$) regions far from active regions.

6. **Three-part 구조 자연 설명** — 식 (10) 의 $b/R$ 진화는 $\rho\gtrsim 10$ 에서 minor radius 가 major radius 만큼 빨리 팽창함을 보여줌, CME 의 어두운 cavity + leading edge 와 일치. / Eq. (10) predicts that the minor radius overexpands to $b\sim R$ at $\rho\sim 10$–$100$, naturally producing the three-part CME morphology (leading edge / cavity / core).

7. **$\delta$-spot 우선 발생 설명** — 가까운 반대극성 쌍 → 매우 빠른 외부장 감소 → 강한 fast CME, 동시에 강한 자기장으로 알펜 속도 큼. Sammis–Tang–Zirin (1980) 의 통계와 정합. / Quadrupolar $\delta$-spot regions combine steep decay ($n>3$ low in the corona) and high Alfvén speeds, explaining why the most powerful eruptions originate there.

8. **헬리컬 킹크와의 차별성** — TI 는 toroidal field 성분이 있어도 안정화되지 않음. 따라서 force-free flux rope 에 대해서도 적용 가능; CME 메커니즘에서 kink 와 별개의, 보다 보편적 메커니즘. / Unlike the helical kink, the TI is not stabilized by a toroidal field, so it applies even to force-free flux ropes — a separate, more universal eruption channel.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Self-inductance of a circular ring / 원형 고리의 자체-인덕턴스

$$L=\mu_0 R\!\left[\ln\!\frac{8R}{b}-2+\frac{l_i}{2}\right]\equiv \mu_0 R\,c(R)$$

- $\ln(8R/b)$: 외부 자기 에너지 부분 (Biot–Savart 적분).
- $-2$: 토러스 곡률 보정.
- $l_i/2$: 내부 자기 에너지 ($l_i=1/2$ for uniform current density).
- $c(R)\approx c_0$: $R$ 에 대해 로그적으로만 변화하므로 좋은 근사.

The ring's self-inductance combines the external Biot–Savart logarithm $\ln(8R/b)$, the toroidal-curvature correction $-2$, and the internal energy contribution $l_i/2$ ($l_i=1/2$ for a uniform current density). It varies only logarithmically with $R$, justifying $c(R)\approx c_0$.

### 4.2 Force balance / 힘 균형

$$\rho_m\frac{d^2 R}{dt^2}=\underbrace{\frac{I^2}{4\pi^2 b^2 R^2}\!\left(L+\frac{\mu_0 R}{2}\right)}_{\text{hoop force (outward)}}-\underbrace{\frac{I\,B_{\rm ex}(R)}{\pi b^2}}_{\text{tension (inward)}}$$

- Hoop force ratio expansion: $\frac{1}{R}\frac{\partial L}{\partial R}+\frac{\mu_0}{2R}\sim \frac{\mu_0}{R}[\ln(8R/b)-1+l_i/2]$.
- $-\nabla\cdot$ pressure of bent channel + curvature self-force $=$ first term.
- Restoring force (per unit length $2\pi R$): $J\times B_{\rm ex}=I/(\pi b^2)\cdot B_{\rm ex}$ acting on the cross-section.

### 4.3 Flux conservation / 자속 보존

$$\Psi_I=LI,\qquad \Psi_{\rm ex}=-2\pi\!\int_0^R B_{\rm ex}(r)r\,dr$$

For $B_{\rm ex}=\hat B R^{-n}$ (valid for $R\geq R_0$):

$$\Psi_{\rm ex}(R)-\Psi_{\rm ex}(R_0)=-2\pi\hat B\!\int_{R_0}^R r^{1-n}\,dr=-\frac{2\pi\hat B}{2-n}(R^{2-n}-R_0^{2-n})$$

Combining with $\Psi=LI-\Psi_{\rm ex}={\rm const}=L_0 I_0-\Psi_{{\rm ex},0}$ gives Eq. (3) above.

### 4.4 Linearized stability and growth rate / 선형 안정성 및 성장률

Define $\epsilon=\rho-1\ll 1$. From Eq. (4):

$$\frac{d^2\epsilon}{d\tau^2}=(n-n_{\rm cr})\,\epsilon\;+\;{\cal O}(\epsilon^2)$$

with $n_{\rm cr}=3/2-1/(4c_0)$. **Solution / 해**:

$$\epsilon(\tau)=\frac{\dot\epsilon_0}{\sqrt{n-n_{\rm cr}}}\sinh\!\big(\sqrt{n-n_{\rm cr}}\,\tau\big),\qquad \dot\epsilon_0=v_0 T/R_0$$

성장률 / Growth rate:

$$\gamma=\frac{\sqrt{n-n_{\rm cr}}}{T}\quad\text{with}\quad T\approx\frac{(c_0+1/2)^{1/2}}{2}\frac{b_0}{V_{Ai}}$$

코로나 값 $b_0\sim 10^4$ km, $V_{Ai}\sim 10^3$ km/s → $T\sim$ 수십 초 → $\gamma^{-1}\sim$ 분~수 분. (관측된 CME 가속 시간 스케일과 정합.)

For coronal $b_0\sim 10^4$ km and $V_{Ai}\sim 10^3$ km/s, $T\sim$ tens of seconds, giving $\gamma^{-1}\sim$ minutes — matching observed CME acceleration timescales.

### 4.5 Critical decay index / 임계 감쇠 지수

**Free expansion** (Eq. 5): $\;n_{\rm cr}=\dfrac{3}{2}-\dfrac{1}{4c_0}$.

**Line-tied / fixed current** (Eq. 9): $\;n_{\rm cr}=\dfrac{3}{2}-\dfrac{1}{2(c_0+1)}$.

| $R_0/b_0$ | $c_0$ | $n_{\rm cr}^{\rm free}$ | $n_{\rm cr}^{\rm tied}$ |
|---:|---:|---:|---:|
| 5 | 1.69 | 1.352 | 1.314 |
| 10 | 2.38 | 1.395 | 1.352 |
| 20 | 3.08 | 1.419 | 1.378 |
| 50 | 3.99 | 1.437 | 1.400 |

(With $l_i=1/2$. For $R_0/b_0=10$, $n_{\rm cr}\approx 1.40$.)

### 4.6 Decay index from a magnetic dipole / 자기 쌍극자에서의 감쇠 지수

대부분의 활성 영역은 광구 표면 아래에 묻힌 두 개의 반대 극성 단일 극(monopole) 또는 그 합으로 근사된다. 두 점전하 $\pm q$ at $(\pm D/2, 0, -d)$ 의 수평 자기장은

$$B_x(0,0,h)=\frac{q}{2\pi}\!\left[\frac{D/2}{((D/2)^2+(h+d)^2)^{3/2}}-\frac{-D/2}{((D/2)^2+(h+d)^2)^{3/2}}\right]\propto \frac{D}{((D/2)^2+(h+d)^2)^{3/2}}$$

$h\gg D$ 한계에서 $B\propto h^{-3}$, 즉 $n\to 3$. $h\ll D$ 에서 $n\to 0$. 따라서 $n=3/2$ 가 되는 임계 높이 $h_{\rm cr}$ 은 항상 존재하며 $D/2\lesssim h_{\rm cr}\lesssim D$ 정도. / In the dipole limit $h\gg D$, $n\to 3$; near the photosphere $n\to 0$. The critical height $h_{\rm cr}$ where $n=3/2$ always lies between $D/2$ and $D$.

### 4.7 Asymptotic velocity / 점근 속도

$$v_\infty^2=\Big(\frac{v_0 T}{R_0}\Big)^2+\frac{2(2n-3+1/(2c_0))(n-1+1/(4c_0))}{(2n-3)(n-1)}\xrightarrow{n>3/2}\;\big(v_0 T/R_0\big)^2+2$$

차원화: $v_\infty\to\sqrt{2}(R_0/b_0)V_{Ai}$. 이는 코로나 빠른 CME 의 $\sim 10^3$ km/s 와 일치 (with $R_0/b_0\sim 10$, $V_{Ai}\sim 10^2$ km/s in the lower corona, or $R_0/b_0\sim 3$, $V_{Ai}\sim 10^3$ km/s in stronger fields).

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1966 ──── Shafranov: tokamak equilibrium of toroidal current ring
            │
1978 ──── Bateman: TI condition n>3/2 (fixed external flux)
            │
1991 ──── Forbes & Isenberg: catastrophe model for CME onset
            │
1999 ──── Titov & Démoulin: line-tied flux rope, n>2 estimate
            │
2000 ──── Yee & Bellan: spheromak expansion experiment
            │
2005 ──── Török & Kliem: helical kink instability of coronal flux rope
            │
══════ 2006: Kliem & Török, "Torus instability" PRL 96, 255002 ══════
            │
2010 ──── Démoulin & Aulanier: combined kink+TI thresholds
            │
2010s ─── Liu, Zuccarello, Cheng, ...: statistical confirmation n≈1.3–1.7
            │
2017 ──── Jiang et al.: data-driven MHD simulations show TI at threshold
            │
2020s ─── Operational space-weather: decay-index-based eruption forecasts
```

이 논문은 핵융합(Bateman) → 천체물리 (CME) → 우주기상 운영(operational forecasting) 으로 이어지는 사슬에서 결정적 연결고리. CME 발생의 이상-MHD 메커니즘 4 대 패러다임 (catastrophe, breakout, kink, **TI**) 의 한 축을 정의했고, 광구 자기장 기반 외삽으로 즉시 시험 가능한 임계 진단을 제공함으로써 이론과 관측·예측을 잇는 다리 역할을 했다.

This paper is the decisive bridge in the chain fusion → astrophysics → operational space weather. It defined one of the four ideal-MHD paradigms of CME onset (alongside catastrophe, breakout, and kink) and supplied the first directly-testable diagnostic: the decay index from a potential-field extrapolation, immediately deployable on routine magnetograms.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 |
|---|---|
| **Bateman (1978), MHD Instabilities** | Original TI derivation with $\Psi_{\rm ex}={\rm const}$; this paper recovers $n_{\rm cr}=3/2$ in the $c_0\to\infty$ limit and generalizes to consistent flux conservation. / 원래 TI 유도, $c_0\to\infty$ 한계로 회복. |
| **Forbes & Priest (1995); Forbes (2000) — catastrophe** | Alternative (loss-of-equilibrium) eruption mechanism; TI provides a distinct ideal-MHD onset criterion that can occur even when no catastrophe exists. / 대안 메커니즘, 균형 상실 모형. |
| **Török & Kliem (2005), ApJ 630** | Companion paper on helical kink instability of a coronal flux rope; TI applies even when kink is stabilized by toroidal field. / 자매 논문, kink 와 상호 보완. |
| **Titov & Démoulin (1999), A&A 351** | Line-tied flux-rope model, earlier $n>2$ estimate that this paper supersedes with $n_{\rm cr}\approx 3/2$. / line-tied flux rope, $n>2$ 추정을 갱신. |
| **Démoulin & Aulanier (2010), ApJ 718** | Numerical mapping of combined kink+TI threshold in the parameter plane; refines $n_{\rm cr}$ for various rope shapes. / 매개변수 평면에서 임계값 정밀화. |
| **Yee & Bellan (2000), Phys. Plasmas 7** | Spheromak expansion experiment; this paper compares against the freely expanding limit and obtains quantitative agreement. / spheromak 실험과 정량 비교. |
| **Liu (2008), ApJ 679 / Zuccarello et al. (2014), ApJ 795** | Observational confirmation: $n\approx 1.3$–$1.7$ at heights of eruption onset for many CMEs. / 관측적 확증. |

---

## 7. References / 참고문헌

1. B. Kliem & T. Török, "Torus instability," *Phys. Rev. Lett.* **96**, 255002 (2006). DOI: 10.1103/PhysRevLett.96.255002
2. V. D. Shafranov, "Plasma equilibrium in a magnetic field," *Rev. Plasma Phys.* **2**, 103 (1966).
3. G. Bateman, *MHD Instabilities* (MIT Press, Cambridge, 1978).
4. T. G. Forbes, "A review on the genesis of coronal mass ejections," *J. Geophys. Res.* **105**, 23153 (2000).
5. V. S. Titov & P. Démoulin, "Basic topology of twisted magnetic configurations in solar flares," *Astron. Astrophys.* **351**, 707 (1999).
6. T. Török & B. Kliem, "Confined and ejective eruptions of kink-unstable flux ropes," *Astrophys. J.* **630**, L97 (2005).
7. J. Yee & P. M. Bellan, "Taylor relaxation and λ decay of unbounded, freely expanding spheromaks," *Phys. Plasmas* **7**, 3625 (2000).
8. P. Démoulin & G. Aulanier, "Criteria for flux rope eruption: non-equilibrium versus torus instability," *Astrophys. J.* **718**, 1388 (2010).
9. R. Liu, "Magnetic flux ropes in the solar corona: structure and evolution toward eruption," *Res. Astron. Astrophys.* **20**, 165 (2020).
10. F. Zuccarello, D. B. Seaton, M. Mierla, et al., "Observational evidence of torus instability as trigger mechanism for CMEs," *Astrophys. J.* **795**, 175 (2014).
11. B. Vršnak et al., "Magnetic configuration and dynamics of erupting prominences," *Astron. Astrophys.* **396**, 673 (2002).
12. R. M. MacQueen & R. R. Fisher, "The kinematics of solar inner coronal transients," *Solar Phys.* **89**, 89 (1983).
13. I. Sammis, F. Tang, & H. Zirin, "The dependence of large flare occurrence on the magnetic structure of sunspots," *Astrophys. J.* **540**, 583 (2000).
14. A. W. Hood & E. R. Priest, "The kink instability of a solar coronal loop as the cause of solar flares," *Geophys. Astrophys. Fluid Dyn.* **17**, 297 (1981).
15. J. Chen & J. Krall, "Acceleration of CMEs in flux-rope models," *J. Geophys. Res.* **108**, 1410 (2003).
16. S. C. Hsu & P. M. Bellan, "Experimental identification of the kink instability as a poloidal flux amplification mechanism for coaxial gun spheromak formation," *Phys. Rev. Lett.* **90**, 215002 (2003).

---

## Appendix A. Numerical example / 부록 A. 수치 예제

**Setting / 설정**: $R_0=10^5$ km (above an active region), $b_0=10^4$ km, $l_i=1/2$, $V_{Ai}=10^3$ km/s in the rope.

(i) **Inductance constant / 인덕턴스 상수**: $c_0=\ln(8\cdot 10)-2+0.25=\ln 80-1.75=4.382-1.75=2.63$.

(ii) **Critical decay index (free) / 임계 감쇠 지수 (자유)**: $n_{\rm cr}=1.5-1/(4\cdot 2.63)=1.5-0.0951=1.405$.

(iii) **Hybrid Alfvén time / 하이브리드 알펜 시간**: $T=(c_0+0.5)^{1/2}/2\cdot b_0/V_{Ai}=(3.13)^{1/2}/2\cdot 10=8.85$ s.

(iv) **Growth rate at $n=2$**: $\gamma=\sqrt{2-1.405}/T=\sqrt{0.595}/8.85=0.0871$ s$^{-1}$, i.e. $\gamma^{-1}\approx 11.5$ s — fast acceleration consistent with impulsive CME onset.

(v) **Asymptotic velocity / 점근 속도**: $v_\infty\approx \sqrt{2}\cdot R_0/T=1.414\cdot 10^5/8.85=1.6\times 10^4$ km/s normalized; with $v_0\to 0$ the dimensional value is $v_\infty\approx \sqrt{2}\cdot(R_0/b_0)\cdot V_{Ai}=14\cdot 10^3/\sqrt{2}\approx 10^4$ km/s — overshoots fast-CME velocities, indicating that pressure pile-up and external $B_{\rm tor}$ (neglected here) regulate the eventual speed.

This worked example illustrates how a small change in $R_0/b_0$ shifts $n_{\rm cr}$ between 1.35 and 1.44, and how the same model yields growth times of seconds-to-minutes typical of impulsive CME acceleration. / 이 예시는 $R_0/b_0$ 의 작은 변화가 $n_{\rm cr}$ 을 1.35–1.44 사이에서 이동시키며, 같은 모형이 초~분 단위의 성장 시간을 산출함을 보여준다.

---

## Appendix B. Observational decay-index measurement / 부록 B. 관측적 감쇠 지수 측정

**Procedure / 절차**:
1. Obtain photospheric line-of-sight magnetogram (e.g., SDO/HMI). / 광구 시선 자기장 자료 획득.
2. Compute potential field extrapolation $\mathbf B(\mathbf r)=-\nabla\phi$ with $\nabla^2\phi=0$, boundary $\partial\phi/\partial z|_{z=0}=B_z$. / 포텐셜 자기장 외삽.
3. Identify polarity inversion line (PIL); choose representative point. / 극성 반전선 위 점 선택.
4. Compute decay index along height: $n(h)=-d\ln|B_h|/d\ln h$ where $B_h$ is the horizontal field component perpendicular to PIL. / 수평장 성분의 감쇠 지수 계산.
5. Find $h_{\rm cr}$ where $n(h_{\rm cr})=1.5$. Compare with measured filament/flux-rope height. / 임계 높이를 측정 위치와 비교.

For most active regions, $h_{\rm cr}\sim 30$–$70$ Mm; eruptive cases tend to have flux ropes at $h\geq h_{\rm cr}$ (Liu 2008; Zuccarello 2014). / 대부분 활성 영역에서 임계 높이는 30–70 Mm; 분출 사건은 플럭스 로프가 임계 높이를 넘은 경우가 많다.
