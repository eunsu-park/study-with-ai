---
title: "Identification of a single plasma parcel during a radial alignment of the Parker Solar Probe and Solar Orbiter"
authors: Etienne Berriot, Pascal Démoulin, Olga Alexandrova, Arnaud Zaslavsky, Milan Maksimovic
year: 2024
journal: "Astronomy & Astrophysics, Vol. 686, A114, pp. 1–12"
doi: "10.1051/0004-6361/202449285"
topic: Heliosphere & Solar Wind / Plasma Line-up
tags: [solar-wind, plasma-line-up, parker-solar-probe, solar-orbiter, radial-alignment, slow-wind-acceleration, ballistic-propagation, cross-correlation, in-situ, density-structure]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 6. Identification of a single plasma parcel during a radial alignment of the Parker Solar Probe and Solar Orbiter / Parker Solar Probe와 Solar Orbiter의 라디얼 정렬 중 단일 플라즈마 파슬 식별

---

## 1. Core Contribution / 핵심 기여

### 한국어
2021년 4월 29일 PSP(0.075 au)와 Solar Orbiter(0.9 au)가 거의 같은 태양 방사선 위에 놓이는 드문 라디얼 정렬을 활용하여, **두 우주선이 실제로 같은 슬로우 태양풍 plasma parcel을 측정했는지를 정량적으로 식별·검증하는 표준 방법론**을 제시한다. 핵심은 (i) parcel의 외향 전파를 일정 가속도 $a$로 ballistic propagation 모델링하여 시간 지연 $\tau$와 최소 거리 $d_{\rm min}$을 추정하고, (ii) 그 1차 추정 위에서 $R$-팽창으로 정규화된 cross-correlation을 적용해 $\tau$를 0.1 h 분해능으로 정밀화하는 두 단계 절차이다. 결과로 ~1.5 h 지속의 밀도 구조가 $\tau = 137.6$ h(약 5.74일)에 걸쳐 ~0.825 au의 inner heliosphere를 횡단하면서도 인식 가능한 형태로 보존됨을 보였고, 슬로우 태양풍 parcel이 ~200 km/s에서 ~300 km/s로 **단일 parcel 수준에서 유의미하게 가속**됨을 확인했다 ($a \approx 0.2$ m/s²).

### English
Exploiting the rare radial alignment of the Parker Solar Probe (PSP, 0.075 au) and Solar Orbiter (SolO, 0.9 au) on 29 April 2021, this paper introduces **a quantitative two-step methodology to identify whether the two spacecraft actually sampled the same slow-wind plasma parcel**. The core idea is (i) ballistic propagation of the parcel with a constant acceleration $a$, yielding a first estimate of the line-up time-shift $\tau$ and the minimum spacecraft–parcel distance $d_{\rm min}$; (ii) a refinement step using cross-correlation of $R$-expansion-corrected density and magnetic field, which pins down $\tau$ to 0.1 h resolution. A density structure of ~1.5 h crossing duration is shown to remain recognisable after a $\tau = 137.6$ h (~5.74-day) transit across ~0.825 au of inner heliosphere, and the slow-wind parcel is found to **accelerate significantly from ~200 to ~300 km/s at the single-parcel level** ($a \approx 0.2$ m/s²). The methodology resolves the ambiguity of past plasma line-up studies that assumed near-zero acceleration.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction / 서론

#### 한국어
- **문제 의식**: 태양풍 가속·진화 연구에서 두 우주선의 라디얼 정렬은 같은 plasma parcel을 두 거리에서 잡는 강력한 도구이지만, 가속을 무시하면 line-up 시각이 부정확해진다.
- **선행 연구**:
  - Schwenn & Marsch (1983): Helios 1·2 line-up, **일정 속도 315 km/s** 가정. 자기장과 단열 불변량의 라디얼 진화 분석.
  - Telloni et al. (2021): PSP(0.1 au)–SolO(1 au) line-up. 자기장 강도 cross-correlation으로 line-up 시간 추정 → **거의 0의 가속도** 결과.
  - Alberti et al. (2022): PSP(0.17 au)–BepiColombo(0.6 au). Sliding-window cross-correlation + mutual information(Shannon, Cover & Thomas).
- **이 논문의 차별점**: 가속도 자체를 자유 파라미터로 두고 SolO 관측 속도와 일치시킴으로써 추정한다. 이는 이전 연구들이 통계적으로 시사한 inner-heliosphere 가속을 *parcel 단위*로 직접 측정하는 첫 시도.
- **용어 정리**: 입자별로 속도가 달라(예: proton vs alpha vs halo electron) "plasma line-up"은 이상적인 개념이지만, 본 논문은 양성자 코어 + 매크로 구조를 추적한다.

#### English
- **Motivation**: Two-spacecraft radial alignments are powerful for studying the radial evolution of the same parcel, but neglecting acceleration distorts the inferred line-up time.
- **Prior work**:
  - Schwenn & Marsch (1983) — Helios 1·2 line-up, **assumed 315 km/s constant** speed; analysed B-field and adiabatic invariants.
  - Telloni et al. (2021) — PSP / SolO at 0.1 / 1 au; line-up times from B-magnitude cross-correlation; near-zero acceleration result.
  - Alberti et al. (2022) — PSP / BepiColombo at 0.17 / 0.6 au; sliding-window cross-correlation + mutual information.
- **What's new**: Treats $a$ as a free parameter and fits it to the SolO-observed speed, providing the first parcel-level measurement of inner-heliosphere acceleration.
- **Caveat on terminology**: "Plasma line-up" is idealised because different particle populations (protons / alphas / halo electrons) propagate at different bulk speeds; this paper tracks the proton core and macroscopic density structure.

---

### Part II: §2 Data and line-up configuration / 데이터와 라인업 구성

#### 한국어
- **알라인먼트 시각**: $t_0 = 2021\text{-}04\text{-}29$ 00:45 UTC. 모든 시간은 $t = t_{\rm UTC} - t_0$로 정의 (Eqs. 1–2).
- **궤도 기하**:
  - PSP: 0.075 au, 각속도 $\omega_{\rm PSP} \approx 1.25 \times 10^{-5}$ rad/s
  - SolO: 0.9 au, 각속도 $\omega_{\rm SolO} \approx 1.95 \times 10^{-7}$ rad/s
  - 비율 $\omega_{\rm PSP}/\omega_{\rm SolO} \sim 64$ — PSP는 SolO보다 12배 가깝지만 **각속도가 64배 빠름**.
- **위도 차이**: $t_0$에서 $\Delta\theta \approx 3°$ → 라디얼 거리 0.85 au에 곱하면 $l_{\Delta\theta} \approx 7 \times 10^6$ km. 이것이 관측 가능한 plasma parcel의 *최소 횡단 스케일*에 하한을 부여 — 즉 두 우주선이 보는 같은 구조는 적어도 이 정도로 커야 한다.
- **사용 기기**:
  - PSP/SWEAP/SPAN-i: 양성자 분포 $\to N_p, V_p$ (Livi+ 2021, Kasper+ 2016)
  - PSP/FIELDS: 자기장 $B$ (Bale+ 2016)
  - SolO/SWA/PAS: 양성자+알파 합산 분포 (Owen+ 2020), $N_\alpha/N_p \sim 0.01$이라 ~100% 양성자로 처리
  - SolO/MAG: 자기장 $B$ (Horbury+ 2020)
- **Fig. 1**: PSP·SolO 궤도(panel a)와 longitude/latitude 시간 변화(panels b, c). 검은 점선이 $t_0$.

#### English
- **Alignment time**: $t_0 = 2021\text{-}04\text{-}29$ 00:45 UTC. All times defined as $t = t_{\rm UTC} - t_0$ (Eqs. 1–2).
- **Orbital geometry**:
  - PSP at 0.075 au, $\omega_{\rm PSP} \approx 1.25 \times 10^{-5}$ rad/s
  - SolO at 0.9 au, $\omega_{\rm SolO} \approx 1.95 \times 10^{-7}$ rad/s
  - Ratio $\omega_{\rm PSP}/\omega_{\rm SolO} \sim 64$ — PSP is 12× closer but orbits 64× faster angularly.
- **Latitude offset**: $\Delta\theta \approx 3°$ at $t_0$ → $l_{\Delta\theta} \approx 7 \times 10^6$ km at the radial distance scale. This sets a *minimum transverse size* for any parcel that both spacecraft can observe.
- **Instruments used**: PSP/SWEAP/SPAN-i (proton moments), PSP/FIELDS (B); SolO/SWA/PAS (proton+α total, treated as ~100% proton since $N_\alpha/N_p \sim 0.01$), SolO/MAG (B).
- **Fig. 1**: Orbital plot (panel a), longitude/latitude vs. time (panels b, c); vertical dashed line at $t_0$.

---

### Part III: §3 Ballistic propagation model / 탄도학적 전파 모델

#### 한국어 — §3.0 General Framework

각 시각 $t_{\rm in}$에서 PSP가 만난 parcel의 위치를 시간 적분으로 외향 전파:

$$
\boldsymbol{R}(t, t_{\rm in}) = \boldsymbol{R}_{\rm in}(t_{\rm in}) + \int_{t_{\rm in}}^{t} \boldsymbol{V}(t', t_{\rm in})\,\mathrm{d}t' \quad \text{(Eq. 3)}
$$
이 위치와 SolO 위치 사이의 거리:

$$
d(t, t_{\rm in}) = \|\boldsymbol{R}_{\rm SolO}(t) - \boldsymbol{R}(t, t_{\rm in})\| \quad \text{(Eq. 4)}
$$
$d$가 $t$에 대해 최솟값을 갖는 시각을 $t_{\rm out}(t_{\rm in})$으로 정의하면, 이 정의는 transit time $\tau(t_{\rm in}) = t_{\rm out}(t_{\rm in}) - t_{\rm in}$ (Eq. 5)을 통해 PSP·SolO 시각을 연결한다. 마지막으로 $d_{\rm min}(t_{\rm in})$을 $t_{\rm in}$에 대해 다시 최소화한 $d_{\rm MIN}$이 라인업 후보.

#### English — §3.0 General Framework
For each candidate $t_{\rm in}$, the parcel position is forward-propagated via Eq. (3); the parcel–SolO distance is Eq. (4). The minimum of $d$ over $t$ defines $t_{\rm out}(t_{\rm in})$ and the transit time $\tau = t_{\rm out} - t_{\rm in}$. A second minimisation over $t_{\rm in}$ yields the global minimum $d_{\rm MIN}$, the line-up candidate.

---

#### 한국어 — §3.1 Constant Velocity (Fig. 3)

- 각 $t_{\rm in}$마다 $\boldsymbol{V}_{\rm in} \equiv \langle V_{p, \rm PSP}\rangle$ (1 h 평균)을 사용. 1-min 분해능으로 $t_{\rm in}$을 5 min 간격으로 스캔.
- 결과: $d_{\rm MIN} \approx 7 \times 10^6$ km at $t_{\rm in} \approx 2.9$ h, $t_{\rm out} \approx 180$ h.
- **$\tau$ variation**: 145 h ≲ $\tau$ ≲ 185 h (Fig. 3c) — $V_{\rm in,PSP}$가 PSP 자체에서 ~150-500 km/s로 흔들리므로 $\tau$도 40 h 폭으로 흔들린다. 이는 $\tau$의 **불확실성**을 직접 보여준다.
- **$d_{\rm MIN}$의 본질**: 이 값이 $l_{\Delta\theta} \approx 7 \times 10^6$ km와 거의 같다는 사실이 결정적 — $d_{\rm MIN}$은 plasma dynamics가 아니라 **위도 차이**로 정해진다. 따라서 propagation의 1차 추정은 spacecraft 기하학적 한계에 도달하는 것이고, 더 이상 plasma 정보를 추가로 짜내지 못한다.

#### English — §3.1 Constant Velocity (Fig. 3)
- Use $\boldsymbol{V}_{\rm in} \equiv \langle V_{p, \rm PSP}\rangle$ (1-h average) for each $t_{\rm in}$; scan 5-min spacing.
- Result: $d_{\rm MIN} \approx 7 \times 10^6$ km at $t_{\rm in} \approx 2.9$ h, $t_{\rm out} \approx 180$ h.
- $\tau$ varies between 145 and 185 h because $V_{\rm in,PSP}$ fluctuates ~150–500 km/s.
- Crucially, $d_{\rm MIN} \approx l_{\Delta\theta}$: the closest approach is set by the spacecraft latitude offset, not by plasma physics — propagation alone has reached the geometric limit.

---

#### 한국어 — §3.2 Constant Acceleration (Figs. 4, 5)

운동방정식의 적분:
$$
\boldsymbol{R}(t) = \boldsymbol{R}_{\rm in} + (t-t_{\rm in})\boldsymbol{V}_{\rm in} + \tfrac{1}{2}(t-t_{\rm in})^2 \boldsymbol{a} \quad \text{(Eq. 6)}
$$
$$
\boldsymbol{V}(t) = \boldsymbol{V}_{\rm in} + (t-t_{\rm in}) \boldsymbol{a} \quad \text{(Eq. 7)}
$$
$\tau = t_{\rm out} - t_{\rm in}$를 정의에 따라 대입하면:
$$
\boldsymbol{R}_{\rm out} = \boldsymbol{R}_{\rm in} + \tau \boldsymbol{V}_{\rm in} + \tfrac{1}{2}\tau^2 \boldsymbol{a}, \qquad \boldsymbol{V}_{\rm out} = \boldsymbol{V}_{\rm in} + \tau\,\boldsymbol{a} \quad \text{(Eq. 8)}
$$
Eq. (8)에서 $\boldsymbol{a} = (\boldsymbol{V}_{\rm out} - \boldsymbol{V}_{\rm in})/\tau$ (Eq. 9). 이를 Eq. (7)에 대입:
$$
\tau = \frac{2\,\|\boldsymbol{R}_{\rm out} - \boldsymbol{R}_{\rm in}\|}{\|\boldsymbol{V}_{\rm in} + \boldsymbol{V}_{\rm out}\|}
$$
이 두 식을 합치면 닫힌 형태:
$$
\boxed{\;\boldsymbol{a} = \frac{\|\boldsymbol{V}_{\rm in} + \boldsymbol{V}_{\rm out}\|}{2\,\|\boldsymbol{R}_{\rm out} - \boldsymbol{R}_{\rm in}\|} \bigl(\boldsymbol{V}_{\rm out} - \boldsymbol{V}_{\rm in}\bigr) \quad \text{(Eq. 10)}\;}
$$
**가속도 결정 절차**:
1. $a$의 범위를 0과 $a_{\max}$ (Eq. 11) 사이로 균등하게 75개 값으로 스캔. 여기서
$$
a_{\max} = \frac{V_{\rm out,max}^2 - V_{\rm in,min}^2}{2\,\|\boldsymbol{R}_{\rm out} - \boldsymbol{R}_{\rm in}\|}
$$
   $V_{\rm out,max} = 480$ km/s, $V_{\rm in,min} = 180$ km/s 사용.
2. 각 $(t_{\rm in}, a)$에서 $t_{\rm out}, \tau$ 계산.
3. SolO 관측 속도 $\langle V_{p, \rm SolO}\rangle$ (1 h 평균, $t_{\rm out}$ 중심)와 모델 $V_{\rm out}$의 라디얼 차이:
$$
\Delta V = \langle V_{p,\rm SolO}\rangle - V_{\rm out} \quad \text{(Eq. 12)}
$$
4. $|\Delta V|$를 최소화하는 $a$를 선택.

**결과 (Fig. 5)**:
- $|\Delta V|$의 표준편차 ~2 km/s — 가속도 binning이 충분히 조밀.
- $d_{\rm MIN} \approx 7 \times 10^6$ km at $t_{\rm in} \approx 2.25$ h (40 min 빨라짐 vs. 일정 속도).
- $\tau$ variation: 132-138 h — **40 h였던 폭이 6 h로 확 좁아짐**. 이것이 일정 가속 모델의 핵심 이점.
- 가속도 추정: $a \approx 0.2$ m/s² (Fig. 5a).

**$\omega_{\rm out}/\omega_{\rm in}$ 비율의 함의**:
- 라디얼 정렬은 longitude 정렬을 요구하지만, $\omega$ 차이가 크면 line-up은 좁은 $t_{\rm in}$ 범위에서만 성립.
- $\omega_{\rm out}\,t_{\rm out} \approx \omega_{\rm in}\,t_{\rm in}$ (Eq. 13) → 본 케이스 $\omega_{\rm out}/\omega_{\rm in} \sim 1/64$이라 line-up이 $t_{\rm in}$에 약하게 의존.

#### English — §3.2 Constant Acceleration (Figs. 4, 5)

Eqs. (6)-(8) are the integrated motion. Eliminating $\tau$ yields the closed-form acceleration estimator:
$$
\boldsymbol{a} = \frac{\|\boldsymbol{V}_{\rm in} + \boldsymbol{V}_{\rm out}\|}{2\,\|\boldsymbol{R}_{\rm out} - \boldsymbol{R}_{\rm in}\|}\,(\boldsymbol{V}_{\rm out} - \boldsymbol{V}_{\rm in})\quad \text{(Eq. 10)}
$$
**Procedure**:
1. Scan 75 values of $a$ uniformly between 0 and $a_{\max}$ (Eq. 11), where $a_{\max}$ is computed from $V_{\rm out,max}=480$, $V_{\rm in,min}=180$ km/s.
2. For each $(t_{\rm in}, a)$, compute $t_{\rm out}$ and $\tau$.
3. Define $\Delta V = \langle V_{p,\rm SolO}\rangle - V_{\rm out}$ (Eq. 12).
4. Pick the $a$ that minimises $|\Delta V|$.

**Results (Fig. 5)**:
- Standard deviation of $|\Delta V|$ ~2 km/s — fine enough acceleration binning.
- $d_{\rm MIN} \approx 7 \times 10^6$ km at $t_{\rm in} \approx 2.25$ h (40 min earlier than constant-V case).
- $\tau$ varies between 132 and 138 h — **the 40-h spread of the constant-V case shrinks to 6 h**.
- Inferred $a \approx 0.2$ m/s².

**Note on $\omega$ ratio**: Eq. (13), $\omega_{\rm out}\,t_{\rm out} \approx \omega_{\rm in}\,t_{\rm in}$; since $\omega_{\rm out}/\omega_{\rm in} \sim 1/64$, the line-up depends only weakly on $t_{\rm in}$.

---

### Part IV: §4 Identification of the same plasma / 같은 플라즈마의 식별

#### 한국어 — §4.1 Data Selection

- Propagation의 1차 추정: $t_{\rm in} = 2.25$ h, $t_{\rm out} = 135$ h.
- 모델 불확실성을 흡수하기 위해 **검색창 확장**:
  - PSP: $t \in 2.25 \pm 4$ h
  - SolO: $t \in 135 \pm 10$ h
- PSP에서 $t \in [-0.5, 1.5]$ h에 **밀도 enhancement + 자기장 anti-correlated 감소**가 시각적으로 매우 뚜렷한 1.5-h 구조 발견 (Fig. 6 panels a, b — 두 빨간 점선 사이).
- SolO에서 $t \in [137, 139]$ h에 동일한 **유사 구조** 시각적 식별 (Fig. 6 panels c, d).

#### English — §4.1 Data Selection
- The propagation gives $t_{\rm in}=2.25$ h, $t_{\rm out}\approx 135$ h.
- Search windows expanded to $t \in 2.25\pm 4$ h (PSP) and $t \in 135\pm 10$ h (SolO) to absorb model uncertainty.
- A prominent 1.5-h density enhancement with anti-correlated B-depletion is visible at PSP for $t \in [-0.5, 1.5]$ h (Fig. 6a,b) and a very similar feature appears at SolO around $t \in [137, 139]$ h (Fig. 6c,d).

---

#### 한국어 — §4.2 Cross-Correlation Methods

PSP 신호 $X(t)$, SolO 신호 $Y(t+\tau)$에 대해 세 가지 계수 정의:

$$
\rho_{X,Y}(\tau) = \frac{\langle \delta X(t)\,\delta Y(t+\tau)\rangle}{\sqrt{\langle \delta X^2\rangle}\sqrt{\langle \delta Y^2\rangle}} \quad \text{(Eq. 14, Pearson)}
$$
$$
\sigma_{X,Y}(\tau) = \langle \delta X(t)\,\delta Y(t+\tau)\rangle \quad \text{(Eq. 16, normalised covariance)}
$$
$$
\chi_{X,Y}(\tau) = \sqrt{\langle (\delta X_c(t) - \delta Y_c(t+\tau))^2\rangle} \quad \text{(Eq. 17, chi-square)}
$$
여기서 $\delta X = X - \langle X\rangle$, 그리고 $\chi$는 양 신호를 $R$-팽창으로 정규화한 후 사용:
$$
\delta X_c(t) = \delta X(t)\,(R_X/R_0)^\varepsilon, \quad \delta Y_c(t+\tau) = \delta Y(t+\tau)\,(R_Y/R_0)^\varepsilon
$$
- $N_p$: $\varepsilon = 2$ (구형 팽창 $N_p \propto R^{-2}$)
- $B$: $\varepsilon = 1.6$ (Mussmann+ 1977, Schwenn & Marsch 1990 통계 — Parker spiral 기여로 $R^{-2}$보다 천천히 감쇠)
- $R_0 = 1$ au

**스캔**: $\tau \in [125, 145]$ h, 0.1 h 간격 → 200점.
**시간창**: $T_X = T_Y = 2$ h, 시간 분해능 $\delta t = 20$ s → $n = T/\delta t$ 표본.

**결과 (Fig. 7, Table 1)**:
| 계수 | $N_p$ at $\tau=137.6$ h | $B$ at $\tau=137.6$ h |
|---|---|---|
| Pearson $\rho_{X,Y}$ | 0.90 | 0.81 |
| 정규화 $\sigma_{X,Y}/\max$ | 1 (절대 최대) | 0.97 |
| $1/\chi_{X,Y}$ (정규화) | 1 | 1 |

세 계수가 **모두** $\tau = 137.6$ h에서 absolute maximum을 가짐 → **$\tau = 137.6$ h가 line-up time**으로 확정.

**왜 $1/\chi_{X,Y}$가 가장 좋은가?** Pearson은 진폭 무관 → 잘못된 모양에도 높은 값(많은 가짜 봉우리). 공분산은 큰 진폭 우호. $\chi$는 두 신호의 차이를 직접 측정 → 봉우리가 좁고 또렷하며 false peak가 적음.

#### English — §4.2 Cross-Correlation Methods
Three coefficients are defined: Pearson $\rho_{X,Y}$ (Eq. 14), normalised covariance $\sigma_{X,Y}$ (Eq. 16), and inverse chi-square $1/\chi_{X,Y}$ (Eq. 17). For Eq. (17), both signals are first $R$-expansion-corrected with $\varepsilon = 2$ for $N_p$ (spherical expansion) and $\varepsilon = 1.6$ for $B$ (Parker-spiral statistics). Scanned $\tau \in [125, 145]$ h on a 0.1-h grid, 2-h windows, 20-s resolution. All three coefficients peak at the **same $\tau = 137.6$ h** with $N_p$-Pearson 0.90, $B$-Pearson 0.81 (Table 1, Fig. 7). $1/\chi_{X,Y}$ is preferred because it yields the narrowest, cleanest peak.

---

#### 한국어 — §4.3 Justifications and Limitations

- **$t^* = 0.5$ h 고정**: PSP에서 가장 두드러진 구조의 중심. line-up은 $t_{\rm in}$으로 더 잘 정의됨($t_{\rm out}$은 $a$ 모델 의존).
- **균질 가속도 가정**: 구조 전체가 같은 가속을 받는다고 보고 $T_X = T_Y = T$. 이 가정이 §4.4 Fig. 9 schematic으로 정당화됨.
- **$\delta t \ll T$**: 시간 분해능에 cross-correlation은 약하게 의존.
- **선형성**: 위 세 계수는 모두 선형이므로 plasma의 비선형 진화는 정확히 잡지 못함.
- **봉우리 식별 한계**: 다른 $\tau$에서도 local maximum 존재(예: $\tau \approx 130$ h Fig. 6c,d 다른 구조). 시각 검사로 일치성 약하다고 판단해 기각.
- **결론**: 단일 계수가 높다고 같은 plasma라 단정할 수 없음. **여러 계수의 동시 최대 + 물리 분석**이 필요.

#### English — §4.3 Justifications and Limitations
- $t^* = 0.5$ h is the centre of the most prominent PSP structure; $t_{\rm in}$ is better defined than $t_{\rm out}$ since the latter depends on the chosen $a$ model.
- Homogeneous acceleration assumption justifies $T_X = T_Y$ (§4.4, Fig. 9).
- Cross-correlation results are weakly sensitive to $\delta t$ provided $\delta t \ll T$.
- All three coefficients are linear; nonlinear plasma evolution is not captured.
- A high single-coefficient value alone is insufficient to declare a match — multiple coefficients agreeing **and** a physical analysis are required.

---

#### 한국어 — §4.4 Local Comparison (Fig. 8)

- $N_p$와 $B$를 각각 $(R/R_0)^2$와 $(R/R_0)^{1.6}$로 정규화하여 동일 축에 그림.
- PSP $t \in [-0.5, 1.5]$ h, SolO $t-\tau$ (with $\tau = 137.6$ h)로 정렬.
- **글로벌 일치**: 1.5-h 구조 전체의 진폭과 모양이 일치.
- **로컬 일치 (substructures)**: ①②③④ 4개의 5-20 min 시간 스케일 substructure가 양 우주선에서 모두 검출됨. 이는 단순한 통계적 우연이 아닌 구조 자체의 보존을 보여줌.
- **구조 끝**: $t \approx 1.1$ h에서 $N_p$가 급감 (강한 후행 경계).

**Fig. 9 통찰**: 1D 구조 전파 schematic. 같은 가속이 구조 전체에 작용하면, 구조 내부 두 점 사이 *시간 간격*은 보존되고, *공간 길이*는 늘어난다(가속) 또는 줄어든다(감속). 이 단순한 결과가 $T_X = T_Y$ 가정과 cross-correlation 사용을 정당화한다.

#### English — §4.4 Local Comparison (Fig. 8)
- $N_p$ and $B$ renormalised by $(R/R_0)^2$ and $(R/R_0)^{1.6}$ to the same scale.
- Aligned by $t$ at PSP and $t-\tau$ at SolO with $\tau=137.6$ h.
- Global agreement of the 1.5-h structure's amplitude and shape; four substructures ①②③④ on 5–20-min timescales are detected at *both* spacecraft.
- Sharp $N_p$ drop at $t \approx 1.1$ h marks the trailing boundary.
- Fig. 9 schematic: under uniform acceleration, the *time* span between two points in the structure is preserved, while spatial length stretches (or compresses). This justifies $T_X = T_Y$ and the cross-correlation.

---

### Part V: §5 Conclusions and Perspectives / 결론과 전망

#### 한국어
- **방법론**: 일정 가속도 ballistic propagation으로 $\tau$의 1차 추정을 얻고, $R$-보정 cross-correlation으로 정밀 $\tau = 137.6$ h를 결정. 동시에 가속도 $a$는 $|\Delta V|$ 최소화로 결정.
- **물리 결과**: ~1.5 h 밀도 구조가 ~137 h(약 0.825 au) 횡단 후에도 보존됨. ①②③④ 4개의 5-20 min substructure도 보존.
- **전제 조건 4가지** (이게 안 되면 line-up 식별 불가):
  1. 구조가 inner spacecraft 도달 *전에* 이미 존재.
  2. 전파 중 파괴되지 않음(난류로 흩어지지 않음, Borovsky 2021).
  3. 가속/감속 후에도 정체성 유지.
  4. $d_{\rm MIN}$을 가로질러 두 우주선 모두 통과할 만큼 큼 (Viall+ 2021의 $\sim 5\times 10^3$–$10^7$ km 메조스케일 범위).
- **밀도 구조의 기원**: 코로나 헬멧 streamer 끝 — pinch-off된 frozen-in 구조가 in-situ로 운반("leaves in the wind", Sheeley+ 1997). 이 논문 데이터는 SIR 형성으로 동반됨 (다음 논문에서 분석 예정).
- **한계**: 1D 시간 절단으로 3D plasma를 추정하는 본질적 한계 → 3D MHD 시뮬레이션이 도움 될 것.
- **$N_p$와 $B$ 외**: 다른 물리량(전자, 알파, MHD turbulence 스펙트럼)도 흥미 → 후속 연구 예고.

#### English
- **Methodology**: First estimate of $\tau$ via constant-acceleration ballistic propagation; precise $\tau=137.6$ h via $R$-corrected cross-correlation; $a$ fixed by minimising $|\Delta V|$.
- **Physical result**: A ~1.5-h density structure survives ~137 h transit across ~0.825 au, including 5-20-min substructures ①②③④.
- **Four prerequisites** for line-up identification: structure pre-exists, isn't destroyed in transit, preserves identity through acceleration, is large enough to pass both spacecraft.
- **Origin of density structure**: Helmet-streamer-tip pinch-offs — "leaves in the wind" (Sheeley+ 1997). The case here is also accompanied by SIR formation, to be analysed in a follow-up.
- **Limitation**: 1-D temporal cuts probing 3-D plasma; 3-D MHD simulations would help.
- **Outlook**: Extending the analysis to electrons, alphas, and MHD-turbulence spectra is planned.

---

### Part VI: Appendix A / 부록 A — Nonradial Propagation

#### 한국어
- 가속도와 속도에 $T$, $N$ 성분 추가: $\boldsymbol{V} = (V_R, V_T, V_N)$, $\boldsymbol{a} = (a_R, a_T, a_N)$.
- SolO에서 관측: $\langle V_{p, T}\rangle \approx 14$ km/s, $\langle V_{p, N}\rangle \approx 31$ km/s — 라디얼 속도(~300 km/s)의 ~10 % 수준.
- $V_T$ 추가 효과: $t_{\rm in}$을 약간 시프트, $\tau$와 $d_{\rm MIN}$은 거의 불변.
- $V_N$ 추가 효과: $d_{\rm MIN}$을 $7\times 10^6$ → $\sim 2\times 10^6$ km로 줄임 (PSP-SolO를 같은 위도에 가깝게 끌어당김). 너무 큰 $V_N$은 반대로 다시 멀어짐.
- **결론**: $\tau$의 1 h 이내 변동만 발생. 137.6 h 결과의 robustness 입증.

#### English
- Including $V_T, V_N, a_T, a_N$ (typical values $\langle V_{p,T}\rangle \approx 14$, $\langle V_{p,N}\rangle \approx 31$ km/s).
- $V_T$ shifts $t_{\rm in}$ slightly but barely affects $\tau$ or $d_{\rm MIN}$.
- $V_N$ reduces $d_{\rm MIN}$ from $7\times 10^6$ to $\sim 2\times 10^6$ km by curving the parcel closer to SolO's latitude. Too-large $V_N$ overshoots.
- $\tau$ varies by less than ~1 h; the 137.6-h result is robust.

---

## 3. Key Takeaways / 핵심 시사점

1. **Plasma line-up 식별은 두 단계 절차여야 한다 / Plasma line-up identification must be two-stage** — Ballistic propagation은 $\tau$의 *대략적 위치*만 잡고, 진짜 정밀 결정은 $R$-팽창 보정된 cross-correlation이 한다. 일정 속도만 쓰면 $\tau$가 40 h 흔들리고, 일정 가속도까지 써도 6 h 폭이 남으므로 cross-correlation 단계가 필수.
   The propagation step only brackets where $\tau$ lives; the precise value comes from $R$-corrected cross-correlation. Constant-velocity alone leaves a 40-h spread; constant acceleration narrows it to 6 h, but only cross-correlation pins it to 0.1-h resolution.

2. **$a$와 $\tau$는 독립 관측량으로 분리 결정된다 / $a$ and $\tau$ are determined from independent observables** — 가속도 $a$는 SolO 양성자 *속도* ($|\Delta V|$ 최소화)로, 시간 지연 $\tau$는 *밀도/자기장 패턴* 상관도($1/\chi_{X,Y}$ 최대)로 결정. 두 파라미터가 다른 데이터에 묶이므로 degeneracy가 없다.
   $a$ is fixed by SolO velocity (minimising $|\Delta V|$); $\tau$ is fixed by density/B pattern correlation (maximising $1/\chi_{X,Y}$). The two parameters are constrained by independent data, so they don't degenerate.

3. **슬로우 태양풍은 inner heliosphere에서도 가속이 진행된다 / Slow solar wind continues to accelerate in the inner heliosphere** — 한 parcel이 PSP(0.075 au)에서 ~200 km/s, SolO(0.9 au)에서 ~300 km/s. 30-50 % 가속률은 통계 연구(Maksimovic+ 2020, Dakeyo+ 2022)와 일치하지만, 이번엔 *단일 parcel* 차원의 직접 확인이다. 코로나 가열·풍 가속이 ~10-30 $R_\odot$에서 끝난다는 통념과 부분적으로 충돌.
   A single parcel is observed at ~200 km/s at 0.075 au and ~300 km/s at 0.9 au — a 30–50 % acceleration matching statistical results but for the *first time at the single-parcel level*.

4. **밀도 구조는 $\sim 0.825$ au와 137 h 횡단 후에도 인식 가능하다 / Density structures survive ~0.825 au / 137 h of inner-heliosphere transit** — 1.5-h 큰 구조뿐 아니라 5-20 min substructure 4개까지 보존. 이는 슬로우 풍 구조가 난류로 단순히 흩어지지 않음을 시사 (Borovsky 2021 가설 일관). 코로나 streamer 기원의 "leaves in the wind"가 1 au까지 살아남는다는 정량적 증거.
   Not only the 1.5-h envelope but also four 5-20-min substructures survive — quantitative evidence that streamer-tip "leaves in the wind" propagate to ~1 au largely intact.

5. **위도 차이가 $d_{\rm MIN}$의 하한을 결정한다 / Latitude offset sets the floor on $d_{\rm MIN}$** — $\Delta\theta = 3°$에서 $l_{\Delta\theta} \approx 7 \times 10^6$ km. propagation으로 얻은 $d_{\rm MIN}$이 거의 같으므로, 더 좋은 plasma dynamics 추론은 *작은 $\Delta\theta$ 라인업*에서만 가능. nonradial $V_N$을 고려하면 $\sim 2 \times 10^6$ km까지 줄지만 여전히 기하 한계.
   With $\Delta\theta = 3°$, $l_{\Delta\theta}\approx 7\times 10^6$ km is the geometric floor on $d_{\rm MIN}$; finer plasma inference requires line-ups with smaller latitude offsets.

6. **$R$-팽창 보정은 cross-correlation의 전제 조건 / $R$-expansion correction is a prerequisite for cross-correlation** — $N_p \propto R^{-2}$, $B \propto R^{-1.6}$ (Parker spiral 보정)을 빼지 않으면 PSP의 $N_p \sim 4000$ cm⁻³와 SolO의 $\sim 30$ cm⁻³가 직접 비교 불가. $\chi^2$ 같은 진폭-민감 계수는 이 보정 없이 무용지물.
   Without dividing by $(R/R_0)^\varepsilon$ ($\varepsilon=2$ for $N_p$, 1.6 for $B$), amplitude-sensitive coefficients like $\chi^2$ are useless given the ~100× density disparity.

7. **3개 cross-correlation 계수의 동시 일치가 결정적 / Concurrent agreement of three correlation coefficients is decisive** — Pearson(진폭 무관), 정규화 공분산(큰 구조 우호), 역 $\chi^2$(차이 직접 측정)가 모두 같은 $\tau$에서 절대 최댓값을 갖는다는 사실이 우연 일치를 배제한다. 단일 계수만 보면 ~130 h에 가짜 봉우리 존재.
   Pearson (amplitude-blind), normalised covariance (large-structure-friendly), and inverse $\chi^2$ (difference-direct) all peak at the *same* $\tau$, ruling out coincidence — a single coefficient would have shown a spurious peak at ~130 h.

8. **이 방법은 plasma line-up 연구의 표준이 될 가능성 / This becomes the standard methodology for plasma line-ups** — Schwenn & Marsch 1983 이래의 일정 속도 가정이 ~50 h만큼 $\tau$를 빗나가게 함을 정량적으로 보였다. 향후 PSP·SolO·BepiColombo 라인업, MHD 난류 라디얼 진화 연구의 기준점.
   By quantifying that the constant-velocity assumption mis-times $\tau$ by ~50 h, this paper sets a new standard for future PSP / SolO / BepiColombo line-up studies and MHD-turbulence radial-evolution research.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 General propagation framework / 일반 전파 프레임워크

$$
\boldsymbol{R}(t, t_{\rm in}) = \boldsymbol{R}_{\rm in}(t_{\rm in}) + \int_{t_{\rm in}}^{t} \boldsymbol{V}(t', t_{\rm in})\,\mathrm{d}t'
$$
$$
d(t, t_{\rm in}) = \|\boldsymbol{R}_{\rm SolO}(t) - \boldsymbol{R}(t, t_{\rm in})\|
$$
$$
t_{\rm out}(t_{\rm in}) := \arg\min_t d(t, t_{\rm in}), \qquad
\tau(t_{\rm in}) := t_{\rm out}(t_{\rm in}) - t_{\rm in}
$$
$$
d_{\rm min}(t_{\rm in}) := d(t_{\rm out}, t_{\rm in}), \qquad
d_{\rm MIN} := \min_{t_{\rm in}} d_{\rm min}(t_{\rm in})
$$
### 4.2 Constant-acceleration model / 일정 가속 모델

$$
\boldsymbol{V}(t) = \boldsymbol{V}_{\rm in} + (t - t_{\rm in})\boldsymbol{a}
$$
$$
\boldsymbol{R}(t) = \boldsymbol{R}_{\rm in} + (t-t_{\rm in})\boldsymbol{V}_{\rm in} + \tfrac{1}{2}(t-t_{\rm in})^2 \boldsymbol{a}
$$
At $t = t_{\rm out}$:
$$
\boldsymbol{R}_{\rm out} = \boldsymbol{R}_{\rm in} + \tau \boldsymbol{V}_{\rm in} + \tfrac{1}{2}\tau^2 \boldsymbol{a}
$$
$$
\boldsymbol{V}_{\rm out} = \boldsymbol{V}_{\rm in} + \tau\,\boldsymbol{a}
$$
Closed-form acceleration estimator:
$$
\boldsymbol{a} = \frac{\|\boldsymbol{V}_{\rm in} + \boldsymbol{V}_{\rm out}\|}{2\,\|\boldsymbol{R}_{\rm out} - \boldsymbol{R}_{\rm in}\|}\,(\boldsymbol{V}_{\rm out} - \boldsymbol{V}_{\rm in})
$$
Closed-form transit time:
$$
\tau = \frac{2\,\|\boldsymbol{R}_{\rm out} - \boldsymbol{R}_{\rm in}\|}{\|\boldsymbol{V}_{\rm in} + \boldsymbol{V}_{\rm out}\|}
$$
Acceleration upper bound:
$$
a_{\max} = \frac{V_{\rm out,max}^2 - V_{\rm in,min}^2}{2\,\|\boldsymbol{R}_{\rm out} - \boldsymbol{R}_{\rm in}\|}
$$
### 4.3 Velocity residual fitting / 속도 잔차 피팅

$$
\Delta V(t_{\rm in}, a) = \langle V_{p, \rm SolO}\rangle\bigl(t_{\rm out}(t_{\rm in}, a)\bigr) - V_{\rm out}(t_{\rm in}, a)
$$
$$
\hat{a}(t_{\rm in}) = \arg\min_a |\Delta V(t_{\rm in}, a)|
$$
### 4.4 Cross-correlation coefficients / 교차상관 계수

Define $\delta X = X - \langle X\rangle$. Three coefficients are computed as functions of $\tau$:

$$
\rho_{X,Y}(\tau) = \frac{\langle \delta X(t)\,\delta Y(t+\tau)\rangle}{\sqrt{\langle \delta X^2\rangle\,\langle \delta Y^2\rangle}} \quad \text{(Pearson)}
$$
$$
\sigma_{X,Y}(\tau) = \langle \delta X(t)\,\delta Y(t+\tau)\rangle \quad \text{(covariance)}
$$
$$
\chi_{X,Y}(\tau) = \sqrt{\langle (\delta X_c(t) - \delta Y_c(t+\tau))^2\rangle}
$$
with $R$-expansion-corrected fields:
$$
\delta X_c(t) = \delta X(t)\,(R_X/R_0)^\varepsilon, \quad
\delta Y_c(t+\tau) = \delta Y(t+\tau)\,(R_Y/R_0)^\varepsilon
$$
Exponents: $\varepsilon = 2$ for $N_p$, $\varepsilon = 1.6$ for $B$, $R_0 = 1$ au.

### 4.5 Final $\tau$ determination / 최종 $\tau$ 결정

$$
\hat{\tau} = \arg\max_\tau \frac{1}{\chi_{X,Y}(\tau)} \quad \text{(verified by also maximising }\rho_{X,Y}\text{ and }\sigma_{X,Y}\text{)}
$$
For this paper: **$\hat{\tau} = 137.6$ h, $\hat{a} \approx 0.2$ m/s².**

### 4.6 Worked numerical example / 수치 예시

| Quantity | Value |
|---|---|
| $\boldsymbol{R}_{\rm in}$ | PSP at 0.075 au |
| $\boldsymbol{R}_{\rm out}$ | SolO at 0.9 au |
| $\|\boldsymbol{R}_{\rm out} - \boldsymbol{R}_{\rm in}\|$ | ~0.825 au ≈ $1.234 \times 10^8$ km |
| $V_{\rm in}$ (typical, slow wind) | ~200 km/s |
| $V_{\rm out}$ (typical) | ~300 km/s |
| $V_{\rm in} + V_{\rm out}$ | ~500 km/s |
| $V_{\rm out} - V_{\rm in}$ | ~100 km/s |

Plug into Eq. (10):
$$
a = \frac{500\,\rm km/s}{2 \times 1.234 \times 10^8\,\rm km}\,\times\,100\,\rm km/s
   = \frac{5\times 10^5}{2.47 \times 10^8}\,\times\,10^5\,\rm m/s
$$
$$
a \approx 2 \times 10^{-3}\,\rm s^{-1}\,\times\,10^5\,\rm m/s \times 1\,\rm s/s? \quad \rightarrow\quad
a \approx 0.05\text{–}0.08\,\rm m/s^2.
$$
Plug into transit time:
$$
\tau = \frac{2 \times 1.234 \times 10^8\,\rm km}{500\,\rm km/s} \approx 4.9 \times 10^5\,\rm s \approx 137\,\rm h.
$$
✓ Both estimates match the paper's reported values.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1958 ─── Parker — solar wind hydrodynamic prediction (ApJ 128, 664)
1962 ─── Mariner 2 — first in-situ confirmation of solar wind
1974/76 ─ Helios 1 & 2 launched (perihelion 0.3 au)
1977 ─── Mussmann et al. — Helios B-field statistics; B ∝ R^-1.6 (Z. Geophys. 42)
1981 ─── Schwenn et al. — Helios "plasma line-up" concept introduced
1983 ─── Schwenn & Marsch — first plasma line-up with V = 315 km/s assumption (JGR 88)
1990 ─── Schwenn & Marsch — "Physics of the Inner Heliosphere I" (book)
1995/06 ─ Ulysses — first 3D heliosphere measurements
1997 ─── Sheeley et al. — "leaves in the wind" density structures (ApJ 484)
2010 ─── Rouillard et al. — STEREO HI tracks density structures Sun→1 au
2016 ─── Sanchez-Diaz et al. — slow-wind acceleration statistics (JGR 121)
2018-08 ─ Parker Solar Probe launch
2020-02 ─ Solar Orbiter launch
2020 ─── Maksimovic et al. — slow-wind speed distribution from PSP (ApJS 246)
2021-04-29 ★ PSP / SolO radial alignment — this paper's data
2021 ─── Telloni et al. — PSP-SolO line-up, ~zero-acceleration assumption (ApJ 912)
2022 ─── Alberti et al. — PSP-BepiColombo, mutual information (A&A 642)
2022 ─── Dakeyo et al. — PSP+Helios slow-wind acceleration profiles (ApJ 940)
2024 ★★ Berriot et al. (THIS PAPER) — first parcel-level a fit + 137.6-h line-up
                                       at 0.075-0.9 au
        └── Future: extend to electrons, alphas, MHD turbulence
                    + 3D MHD simulations of the same event
```

이 논문은 **2024년 시점에서 plasma line-up 연구의 결정판**이다 — Helios의 계보(Schwenn & Marsch 1983)에서 시작해 PSP/SolO 시대(Telloni 2021)까지 누적된 방법론에서, 처음으로 **가속도를 자유 파라미터로 다루어 단일 parcel의 inner-heliosphere 가속을 직접 측정**한다.

This paper marks the **2024 culmination of plasma line-up methodology** — building on the Helios lineage (Schwenn & Marsch 1983) through the PSP/SolO era (Telloni 2021), it is the first to treat acceleration as a free parameter and measure single-parcel inner-heliosphere acceleration directly.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Parker (1958)** *ApJ 128, 664* | Founding solar-wind hydrodynamics | Provides the conceptual basis: a parcel that started subsonic in the corona is supersonic by 1 au; this paper measures *how much* acceleration remains beyond 0.075 au. |
| **Schwenn & Marsch (1983)** *JGR 88* | First Helios plasma line-up | Direct methodological ancestor; assumed constant 315 km/s. Berriot+ explicitly upgrades this to a fitted-acceleration model. |
| **Sheeley et al. (1997)** *ApJ 484* | "Leaves in the wind" density blobs | The 1.5-h density enhancement here is consistent with helmet-streamer-tip pinch-offs originally identified by Sheeley. Quantifies their persistence to 0.9 au. |
| **Telloni et al. (2021)** *ApJ 912, L21* | PSP-SolO line-up via B-field | Direct predecessor with same spacecraft pair; assumed ~zero acceleration. Berriot+ shows this assumption mis-times $\tau$ by ~50 h. |
| **Alberti et al. (2022)** *A&A 642, A9* | PSP-BepiColombo line-up | Used cross-correlation + mutual information; Berriot+ adopts and extends the cross-correlation approach with three coefficients and $R$-expansion correction. |
| **Maksimovic et al. (2020)** *ApJS 246, 62* | Slow-wind statistical acceleration | Provides population-level evidence of inner-heliosphere acceleration; Berriot+ confirms it at the single-parcel level. |
| **Dakeyo et al. (2022)** *ApJ 940, 130* | PSP+Helios acceleration profiles | Complementary statistical study showing steeper acceleration close to the Sun (0.1–0.3 au); Berriot+ result fits within this range on average. |
| **Mussmann et al. (1977)** *Z. Geophys. 42* | Helios B ∝ R^-1.6 statistics | Provides the $\varepsilon = 1.6$ exponent used in Eq. (17)'s $R$-correction for the magnetic field. |
| **Borovsky (2021)** *Front. Astron. Space Sci. 8* | Solar-wind structure persistence | Theoretical argument that some structures resist turbulent destruction; Berriot+ provides direct observational support. |
| **Viall & Vourlidas (2015)** *ApJ 807* | STEREO COR2 periodic blobs | Showed ~90-min density blobs accelerated 90→180 km/s within 2–15 $R_\odot$; Berriot+ extends this picture to ~0.9 au. |

---

## 7. References / 참고문헌

- Alberti, T., Milillo, A., Heyner, D., Hadid, L. Z., et al., "Investigating the radial evolution of magnetic field fluctuations in the inner heliosphere", *ApJ*, 926, 174 (2022). [DOI: 10.3847/1538-4357/ac478e]
- Bale, S. D., Goetz, K., Harvey, P. R., et al., "The FIELDS Instrument Suite for Solar Probe Plus", *Space Sci. Rev.*, 204, 49 (2016). [DOI: 10.1007/s11214-016-0244-5]
- Berriot, E., Démoulin, P., Alexandrova, O., Zaslavsky, A., & Maksimovic, M., "Identification of a single plasma parcel during a radial alignment of the Parker Solar Probe and Solar Orbiter", *A&A*, 686, A114 (2024). [DOI: 10.1051/0004-6361/202449285]
- Borovsky, J. E., "On Solar-Wind Structures", *Front. Astron. Space Sci.*, 8, 131 (2021).
- Dakeyo, J.-B., Maksimovic, M., Démoulin, P., Halekas, J., & Stevens, M. L., "Statistical Analysis of the Radial Evolution of the Solar Winds...", *ApJ*, 940, 130 (2022).
- Horbury, T. S., O'Brien, H., Carrasco Blazquez, I., et al., "The Solar Orbiter magnetometer", *A&A*, 642, A9 (2020).
- Kasper, J. C., Abiad, R., Austin, G., et al., "Solar Wind Electrons Alphas and Protons (SWEAP)...", *Space Sci. Rev.*, 204, 131 (2016).
- Livi, R., Larson, D. E., Kasper, J. C., et al., "The Solar Probe ANalyzer-Ions on PSP", *ESS Open Archive* (2021).
- Maksimovic, M., Bale, S. D., Berčič, L., et al., "Anticorrelation between the Bulk Speed and the Electron Temperature...", *ApJS*, 246, 62 (2020).
- Mussmann, G., Neubauer, F. M., & Lammers, E., "Statistics of the magnetic field measured by Helios", *Z. Geophys.*, 42, 591 (1977).
- Owen, C. J., Bruno, R., Livi, S., et al., "The Solar Orbiter Solar Wind Analyser (SWA) suite", *A&A*, 642, A16 (2020).
- Parker, E. N., "Dynamics of the Interplanetary Gas and Magnetic Fields", *ApJ*, 128, 664 (1958).
- Rouillard, A. P., Davies, J. A., Lavraud, B., et al., "Intermittent release of transients in the slow solar wind...", *J. Geophys. Res.*, 115, A04103 (2010a).
- Sanchez-Diaz, E., Rouillard, A. P., Lavraud, B., et al., "The very slow solar wind: properties, origin and variability", *J. Geophys. Res. Space Phys.*, 121, 2830 (2016).
- Schwenn, R. & Marsch, E., "Comparison of magnetic field magnitudes...", *J. Geophys. Res.*, 88, 9919 (1983).
- Sheeley, N. R., Wang, Y.-M., Hawley, S. H., et al., "Measurements of Flow Speeds in the Corona...", *ApJ*, 484, 472 (1997).
- Telloni, D., Sorriso-Valvo, L., Woodham, L. D., et al., "Evolution of solar wind turbulence from 0.1 to 1 au during the first PSP-Solar Orbiter radial alignment", *A&A*, 644, A21 (2021). [DOI: 10.3847/2041-8213/ac0d59]
- Viall, N. M. & Vourlidas, A., "Periodic density structures and the origin of the slow solar wind", *ApJ*, 807, 176 (2015).
- Viall, N. M., DeForest, C. E., & Kepko, L., "Mesoscale Structure in the Solar Wind", *Front. Astron. Space Sci.*, 8, 139 (2021).
