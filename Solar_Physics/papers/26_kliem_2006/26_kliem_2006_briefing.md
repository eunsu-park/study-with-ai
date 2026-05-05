---
title: "Pre-Reading Briefing: Torus Instability"
paper_id: "26_kliem_2006"
topic: Solar_Physics
date: 2026-04-27
type: briefing
---

# Torus Instability: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: B. Kliem & T. Török, "Torus instability," Phys. Rev. Lett. 96, 255002 (2006).
**Author(s)**: Bernhard Kliem, Tibor Török
**Year**: 2006
**DOI**: 10.1103/PhysRevLett.96.255002

---

## 1. 핵심 기여 / Core Contribution

이 논문은 토러스(torus, 도넛) 모양의 전류 고리가 저-베타(low-β) 자기 플라즈마 환경에서 갖는 팽창 불안정성(expansion instability)을 분석하여, "토러스 불안정(Torus Instability, TI)" 의 임계 조건을 정량적으로 유도한다. 결정적 결과는 외부 포텐셜 자기장 $B_{\rm ex}\propto R^{-n}$ 에서 임계 감소 지수가 $n_{\rm cr}=3/2-1/(4c_0)$ ($c_0=\ln(8R/b)-2+l_i/2$ 의 상수)로 주어지며, 자유 팽창 고리의 경우 약 $n_{\rm cr}\approx 1.5$ 라는 것이다. 이 임계값은 코로나 질량 방출(CME, Coronal Mass Ejection) 의 빠르고 느린 두 부류를 통합적으로 설명하고, $\delta$-점(델타 흑점) 영역에서 가장 강력한 분출이 일어나는 이유를 자연스럽게 설명한다.

This Letter analyzes the expansion instability of a toroidal current ring in a low-β magnetized plasma and derives the quantitative criterion for the "Torus Instability (TI)". The decisive result is that an external poloidal field $B_{\rm ex}\propto R^{-n}$ destabilizes the ring when the decay index exceeds $n_{\rm cr}=3/2-1/(4c_0)$, which reduces to the canonical $n_{\rm cr}\approx 1.5$ for a freely expanding ring. The criterion provides a unified description of fast and slow CMEs and naturally explains the preferred occurrence of the most powerful eruptions in $\delta$-spot regions where the overlying field decays most rapidly with height.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

1970년대 토카막(tokamak) 융합 연구는 Shafranov 평형(equilibrium)을 통해 토로이달 전류 고리가 유지되는 조건을 정립했고, Bateman(1978)은 외부 포텐셜 자기장이 충분히 빠르게 감소하면 후프 힘(hoop force)이 인장 로렌츠 힘(restoring Lorentz force)을 압도하여 고리가 팽창 불안정해짐을 보여 $n>3/2$ 라는 조건을 유도했다. 그러나 핵융합 장치에서는 작은 감소 지수와 벽의 영상 전류로 TI가 억제되었기에 후속 연구가 거의 없었다. 한편 천체물리에서는 1990년대 이후 spheromak 분출 실험과 CME 관측을 통해 도넛형 자기 구조(플럭스 로프, flux rope)의 분출이 핵심 문제로 떠올랐고, "균형 상실(catastrophe)" 모형(Forbes & Priest)과 헬리컬 킹크 불안정성(helical kink, Török & Kliem 2005)이 후보 메커니즘으로 제시되었다. Kliem & Török(2006)은 Bateman의 결과를 일반화하여 태양 코로나에 직접 적용함으로써 CME 발생 메커니즘 논쟁의 새로운 축을 열었다.

In the 1970s tokamak fusion research established the Shafranov equilibrium, and Bateman (1978) showed that a toroidal ring is unstable to expansion when an external poloidal field decreases sufficiently rapidly with major radius, deriving $n>3/2$. In fusion devices the TI was suppressed by small decay indices and image currents in conducting walls, so the result was largely forgotten. Meanwhile, by the 1990s astrophysical work on spheromak expansion experiments and CME observations had focused attention on toroidal flux-rope eruptions, with the catastrophe model (Forbes & Priest) and the helical kink instability (Török & Kliem 2005) emerging as candidate eruption triggers. Kliem & Török (2006) generalized Bateman's analysis to the solar corona, opening a new axis in the CME-trigger debate by providing a clean ideal-MHD criterion based on the height profile of the ambient field.

### 타임라인 / Timeline

- **1966** — Shafranov, *Rev. Plasma Phys.* 2: tokamak equilibrium of a toroidal current ring
- **1978** — Bateman, *MHD Instabilities*: TI condition $n>3/2$ for fixed external flux
- **1991** — Forbes & Isenberg / Forbes & Priest: catastrophe model of CME onset
- **1999** — Titov & Démoulin: line-tied flux-rope model (threshold $n>2$ estimated)
- **2000** — Forbes, *JGR* 105: review of CME models
- **2005** — Török & Kliem, *ApJ* 630: helical kink instability of a coronal flux rope
- **2006** — **Kliem & Török, *PRL* 96: torus instability, $n_{\rm cr}\approx 1.5$**
- **2010** — Démoulin & Aulanier: combined kink+TI threshold mapping
- **2010s** — Observational confirmations: $n\approx 1.3$–$1.7$ at eruption sites (Liu, Zuccarello, ...)

---

## 3. 필요한 배경 지식 / Prerequisites

- **수학 / Math**: 벡터 미적분, 로그 미분(logarithmic derivative), 1차원 ODE 안정성 분석, 점근 전개
- **전자기학 / Electromagnetics**: 비오-사바르(Biot–Savart) 법칙, 자속(magnetic flux) 보존, 자기 인덕턴스(inductance)
- **MHD**: ideal MHD, 자기 장력(magnetic tension)·압력, 플럭스 로프 평형, low-β 플라즈마 가정
- **선행 논문 / Prior papers**: Shafranov 평형(논문 #23 시리즈), Forbes & Priest catastrophe(논문 #25 계열), Bateman(1978) 4장
- **개념 / Concepts**: Hoop force(굽힌 전류 채널의 자체 팽창력), poloidal/toroidal 분리, 감쇠 지수(decay index) $n=-d\ln B/d\ln h$, "라인 타잉(line-tying)" 효과

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Torus instability (TI) / 토러스 불안정 | 외부 자기장이 충분히 빠르게 약해질 때 토로이달 전류 고리가 자유 팽창하는 이상-MHD 불안정 / Ideal-MHD expansion instability of a toroidal current ring when the external field weakens sufficiently fast |
| Hoop force / 후프 힘 | 굽혀진 전류 채널의 자체-인덕턴스 감소 경향에서 비롯되는 외향 로렌츠 힘 ($\propto I^2/R$) / Outward Lorentz self-force from the channel's tendency to lower its self-inductance |
| Decay index $n$ / 감쇠 지수 | $n=-d\ln B_{\rm ex}/d\ln R$, 외부 포텐셜 자기장이 높이에 따라 얼마나 빠르게 약해지는가의 척도 / Logarithmic slope of the external field with respect to height/major radius |
| Shafranov equilibrium / 샤프라노프 평형 | 후프 힘 + 토로이달 압력 vs. 외부 자기장 인장력의 균형으로 유지되는 토로이달 전류 고리 평형 / Toroidal ring equilibrium balancing hoop and tire-tube forces against the external field |
| Flux rope / 플럭스 로프 | 트위스트된 자기력선 다발이 만드는 도넛형 자기 구조; CME 코어의 핵심 모형 / Twisted bundle of field lines forming a toroidal magnetic structure modeling the CME core |
| Aspect ratio $R/b$ / 종횡비 | 큰 반지름 $R$ 과 작은 반지름 $b$ 의 비; "큰 종횡비 한계"는 hoop force 만 남도록 단순화 / Ratio of major to minor radius; "large aspect ratio limit" simplifies the dynamics |
| Internal inductance $l_i$ / 내부 인덕턴스 | 전류 분포의 비균질성에 따른 인덕턴스 보정($l_i=1/2$ for uniform) / Correction depending on the radial profile of the current density |
| Line tying / 라인 타잉 | 광구(photosphere)에 발이 고정되어 자기력선이 자유롭게 움직이지 못하게 하는 효과 / Photospheric anchoring constraint that pins field-line footpoints |
| $\delta$-spot / 델타 흑점 | 한 흑점 내부에 반대 극성이 가깝게 공존하는 4극(quadrupolar) 활성 영역; 강하고 빠르게 감소하는 자기장 / Quadrupolar active region with opposite-polarity pairs packed within a single sunspot, hosting steep field decay |
| Self-similar expansion / 자가-유사 팽창 | $b/R$ 가 거의 일정하게 유지되며 진행되는 팽창; spheromak 실험에서도 관측 / Expansion in which $b/R$ stays approximately constant, also seen in spheromak experiments |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Force balance in the major radius direction / 큰 반지름 방향 힘 균형**

$$\rho_m\frac{d^2 R}{dt^2}=\frac{I^2}{4\pi^2 b^2 R^2}(L+\mu_0 R/2)-\frac{I B_{\rm ex}(R)}{\pi b^2}$$

좌변은 관성, 우변 첫 항은 hoop force, 둘째 항은 외부 포텐셜 자기장에 의한 인장 로렌츠 힘이다. The left side is inertia; on the right, the first term is the hoop force, the second is the restoring Lorentz force due to the external poloidal field.

**(2) Conserved enclosed flux assumption / 봉입 자속 보존 가정**

$$\Psi=\Psi_I+\Psi_{\rm ex}=LI-2\pi\int_0^R B_{\rm ex}(r)r\,dr$$

이상-MHD 에서 봉입 자속은 보존되므로 $I(R)$ 는 외부장 프로파일에 의해 결정된다. Ideal-MHD requires $\Psi$ to be invariant during a perturbation, fixing $I(R)$ in terms of the prescribed $B_{\rm ex}$.

**(3) Critical decay index for free expansion / 자유 팽창 임계 감소 지수**

$$n_{\rm cr}=\frac{3}{2}-\frac{1}{4c_0},\qquad c_0=\ln(8R_0/b_0)-2+l_i/2$$

전형적 코로나 값($R_0/b_0\sim 10$, $l_i=1/2$)에서 $n_{\rm cr}\approx 1.4$–$1.5$. For coronal aspect ratios $R_0/b_0\sim 10$ with $l_i=1/2$, $n_{\rm cr}\approx 1.4$–$1.5$.

**(4) Exponential growth at threshold / 임계 부근 지수 성장**

$$\epsilon(\tau)=\frac{v_0 T/R_0}{(n-n_{\rm cr})^{1/2}}\sinh\!\left((n-n_{\rm cr})^{1/2}\tau\right),\quad \epsilon\ll 1$$

성장률 $\gamma=(n-n_{\rm cr})^{1/2}/T$. The growth rate scales as $\gamma=(n-n_{\rm cr})^{1/2}/T$ where $T$ is the hybrid Alfvén time of the minor radius.

**(5) Asymptotic expansion velocity / 점근 팽창 속도**

$$v_\infty\approx\big[(v_0 T/R_0)^2+2\big]^{1/2},\qquad n>3/2$$

차원적으로 $\sqrt{2}(R_0/b_0)V_{Ai}$ 정도의 알펜 속도(Alfvén velocity) 스케일을 갖는다. Dimensionally this is $\sim\sqrt{2}(R_0/b_0)V_{Ai}$, comparable to the inner-corona Alfvén speed of $\sim 10^3$ km/s for fast CMEs.

---

## 6. 읽기 가이드 / Reading Guide

1. **도입과 Bateman 결과 복습 / Intro & Bateman recap (p. 1, col. 1)** — Shafranov 평형과 $n>3/2$ 조건의 출발점. Read carefully: this sets the canonical TI baseline.
2. **두 가지 시나리오 / Two scenarios (p. 1, col. 2)** — (a) "freely expanding ring" (실험·우주), (b) "constant total current" (CME 초기, 라인 타잉). Note that the canonical $n_{\rm cr}=3/2$ result is recovered in the limit $c_0\to\infty$.
3. **수식 (1)–(4) 유도 / Equations (1)–(4)** — flux conservation 가정($\Psi_{\rm ex}={\rm const}$ vs. consistent treatment) 의 함의를 추적. Track how the assumption $c(R)={\rm const}$ simplifies $\rho$-evolution to a clean ODE.
4. **그림 1–3 / Figures** — (Fig. 1) acceleration profile vs. $\rho$, (Fig. 2) line-tied case with maximum amplification, (Fig. 3) overexpansion of $b/R$ explaining CME "three-part" cavity.
5. **CME 응용 / Application to CMEs (p. 3, col. 2)** — fast CMEs ($n>3/2$ already at $h\lesssim D/2$) vs. slow CMEs (filaments far from active regions); $\delta$-spot quadrupolar steep decay.
6. **결론 / Conclusion** — TI as one of three CME mechanisms (catastrophe, helical kink, TI), and unification of fast/slow CMEs.

---

## 7. 현대적 의의 / Modern Significance

이 4쪽짜리 PRL 논문은 출간 이후 태양물리 분야에서 가장 많이 인용된 이론 논문 중 하나가 되었다. "감쇠 지수 $n\approx 1.5$" 라는 단순 임계는 관측자에게 즉시 적용 가능한 진단 도구를 제공했고, SDO/HMI·SOLIS 의 광구 자기장으로부터 외부 포텐셜장을 외삽한 뒤 활성 영역 위 높이에 따른 $n(h)$ 를 계산하는 표준 절차가 자리잡았다. Liu(2008), Zuccarello et al.(2014), 등 수많은 통계 연구가 분출된 사건의 임계 높이에서 $n\approx 1.3$–$1.7$ 임을 확인했다. TI 는 또한 헬리컬 킹크 불안정성 (#25), Forbes-Priest 균형 상실 모델 (#23/24), magnetic breakout (#27) 과 함께 CME 발생의 4 대 패러다임을 이루며, 우주기상 예측(space-weather forecasting)에서 분출 가능성 추정의 핵심 지표로 사용된다.

This four-page PRL has become one of the most cited theoretical papers in solar physics. The simple criterion "decay index $n\approx 1.5$" gave observers an immediately deployable diagnostic: extrapolate a potential field from photospheric magnetograms (SDO/HMI, SOLIS), then compute $n(h)$ above an active region. Statistical studies (Liu 2008; Zuccarello et al. 2014; and many others) have repeatedly found $n\approx 1.3$–$1.7$ at the heights where flux ropes lose stability. Together with the helical kink (#25), the Forbes–Priest catastrophe (#23/24), and magnetic breakout (#27), the TI forms one of the four dominant paradigms of CME initiation and is now a routine ingredient in space-weather forecasting tools that estimate eruption likelihood.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
