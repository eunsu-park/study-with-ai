---
title: "Thermal Diagnostics with AIA on SDO: A Validated Method for DEM Inversions"
authors: [Cheung, Boerner, Schrijver, Testa, Chen, Peter, Malanushenko]
year: 2015
journal: "The Astrophysical Journal"
doi: "10.1088/0004-637X/807/2/143"
topic: Solar_Observation
tags: [DEM, AIA, SDO, sparse_inversion, basis_pursuit, thermal_diagnostics, active_region]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 30. Thermal Diagnostics with AIA on SDO: A Validated Method for DEM Inversions / SDO/AIA를 이용한 열 진단: DEM 역산의 검증된 방법

---

## 1. Core Contribution / 핵심 기여

Cheung et al. (2015)은 SDO/AIA의 6개 협대역 EUV 채널에서 미분 방출 측도(DEM) 분포를 역산하는 "희소(sparse) DEM inversion" 방법을 제시한다. 전통적 DEM 역문제는 Fredholm 제1종 적분방정식이며, AIA의 $m=6$개 채널로는 $n\gg 6$개의 온도 bin을 제약해야 하므로 근본적으로 부정(underdetermined)이다. 저자들은 이 문제를 basis pursuit(기저 추구) 선형계획 문제로 재정식화한다: L1-norm을 최소화하되 DEM 양성($\vec{x}\ge 0$), 관측값 허용오차 $\vec\eta$ 내 만족이라는 제약을 부과한다. 이 정식화는 (1) 양성 해를 보장하고, (2) Occam의 면도날(가장 parsimonious한 해 선호)을 자연스럽게 실현하며, (3) simplex 방법으로 초당 $10^4$개 이상 해를 계산할 수 있다. 저자들은 3가지 thermal model(Gaussian DEMs, AR 11158의 NLFFF quasi-steady loops, 시간종속 MHD 시뮬레이션)으로 방법을 엄격히 검증하고, 이어 실제 AR 11158 관측에 적용하여 두 번의 solar rotation에 걸친 thermal evolution을 보인다. 마지막으로 Hinode/XRT Be-thin 채널을 추가한 joint AIA+XRT inversion이 고온 DEM 정확도를 크게 향상시킴을 보인다.

Cheung et al. (2015) introduce a "sparse DEM inversion" method for recovering the differential emission measure distribution from SDO/AIA's six narrowband EUV channels. The DEM inverse problem is a Fredholm integral equation of the first kind; with only $m=6$ AIA channels but $n\gg 6$ temperature bins, it is fundamentally underdetermined. The authors recast it as a basis-pursuit linear program: minimize the L1 norm of the DEM subject to positivity ($\vec{x}\ge 0$) and tolerance bands $\vec\eta$ around the data. This formulation (1) guarantees positive-semidefinite solutions, (2) naturally realizes Occam's razor by preferring parsimonious solutions, and (3) admits a fast simplex solution (>10^4 solutions/second). The method is validated against three classes of thermal models of increasing realism: Gaussian log-normal DEMs, quasi-steady loop atmospheres on an NLFFF extrapolation of AR 11158, and a time-dependent compressible MHD simulation of AR corona formation. Application to real AR 11158 data during its flaring and decay phases reveals clear thermal-structure evolution. Adding the Hinode/XRT Be-thin channel markedly improves high-temperature DEM recovery, demonstrating the method's extensibility to joint multi-instrument inversions.

---

## 2. Reading Notes / 읽기 노트

### Part I: The DEM Inverse Problem / DEM 역문제 (§2, pp. 1–3)

**Forward problem.** AIA의 각 EUV 채널 $i$에서 관측되는 노출시간-정규화 pixel 값 $y_i$ (DN s$^{-1}$ pixel$^{-1}$)은 코로나 플라스마의 DEM 분포와 온도 응답 함수의 적분으로 주어진다:

$$
y_i = \int_0^\infty K_i(T)\, \mathrm{DEM}(T)\, dT \quad\text{(Eq. 1)}
$$

여기서 $K_i(T)$는 온도 응답 함수(DN cm$^5$ s$^{-1}$ pixel$^{-1}$)이고, DEM은 $\mathrm{DEM}(T)\,dT = \int_0^\infty n_e^2(T)\,dz$로 정의된다. AIA는 `aia_get_response(/temp,/evenorm,/chiantifix)` (SolarSoft) 루틴과 CHIANTI 7.1.3, Feldman-Grevesse-Landi coronal abundances로 $K_i(T)$를 계산한다. `chiantifix` 키워드는 94 Å 채널의 누락된 transition을 보정한다(Boerner et al. 2014).

The observed AIA pixel rate in channel $i$ is Eq. (1), where $K_i(T)$ is the temperature response (DN cm$^5$ s$^{-1}$ pixel$^{-1}$). DEMs are computed using CHIANTI 7.1.3 with coronal abundances and pressure $p/k = 10^{15}$ K cm$^{-3}$. The `chiantifix` flag corrects missing lines in the 94 Å channel.

**Passbands used.** AIA의 7개 EUV 채널 중 304 Å는 광학적으로 두꺼워 CHIANTI로 모델링 불가하므로 6개 채널 (94, 131, 171, 193, 211, 335 Å)만 사용. 이로부터 $m=6$.

Of AIA's seven EUV channels, 304 Å is typically optically thick under CHIANTI's coronal-equilibrium assumption, so only six channels are used for DEM inversion, giving $m=6$.

**Craig & Brown's four pathologies (1976).** 이 Fredholm 제1종 문제의 병적 성격:
1. 주어진 $\vec y$에 대해 해가 없을 수도 있다.
2. 해가 있어도 유일하지 않을 수 있다.
3. 해가 있어도 안정적이지 않다($\vec y$의 작은 변화 → DEM의 큰 변화).
4. 해가 양(positive-semidefinite)이 아닐 수 있다.

Craig & Brown (1976) enumerated four pathologies: (1) non-existence, (2) non-uniqueness, (3) instability, (4) non-positivity. Any inversion scheme must confront these.

**Matrix formulation (§2.2).** Quadrature로 이산화하면 $\vec y = \mathbf{D}\vec x$ (Eq. 2). Dirac-delta basis + 3세트의 truncated Gaussian basis (width $a=0.1, 0.2, 0.6$)를 결합하여 총 $l=84$ basis 함수를 사용. 온도 격자는 $\log T/K \in [5.5, 7.5]$, $\Delta\log T = 0.1$, $n=21$ bin. 이에 따라 $\mathbf{D}$ 행렬은 $6\times 84$ 크기로 심하게 underdetermined.

After discretizing on $n=21$ log-T bins over $\log T/K \in [5.5, 7.5]$ with $\Delta\log T = 0.1$, and using $l=84$ basis functions (21 Dirac deltas + three sets of truncated Gaussians of widths 0.1, 0.2, 0.6), we get $\vec y = \mathbf{D}\vec x$ with $\mathbf{D}$ of size $6\times 84$ — severely underdetermined.

### Part II: Methods Based on χ² and Regularization / χ² 최소화 기반 기존 방법 (§2.3, pp. 3–4)

**Reduced χ² minimization.** 대부분의 기존 방법은 다음을 최소화:
$$
\chi^2(\vec x) = \sum_{i=1}^m \left(\frac{y_i - \sum_j K_{ij} x_j}{\delta y_i}\right)^2 \quad\text{(Eq. 6)}
$$

이 방법은 overdetermined($n<m$) 시스템에는 적합하나, AIA의 underdetermined($m<n$) 문제에서는 overfitting과 다중해가 문제가 된다. 해결책: Lagrange multiplier 기반 정규화 $\chi^2 + F(\vec x)$ 추가.

Most existing schemes minimize Eq. (6). For underdetermined systems, overfitting occurs; standard remedy is adding a regularization term $F(\vec x)$ via Lagrange multipliers.

**Smoothness (2nd-order) regularization.** $F(\vec x) = \lambda\sum(x_{i-1}-2x_i+x_{i+1})^2$. Phillips (1962), Craig & Brown (1986), Monsignori Fossi & Landini (1991), Hubeny & Judge (1995).

**Zeroth-order regularization.** $F(\vec x) = \lambda\|\vec x\|_2^2$. Hannah & Kontar (2012), Plowman et al. (2013). SVD로 직접 풀 수 있으나, χ² 최소화 방법은 일반적으로 positive-semidefinite 해를 보장하지 못한다(음수 DEM bin 발생 → post hoc 수정 필요).

Zeroth-order regularization (L2 norm penalty) with SVD is used by Hannah & Kontar (2012) and Plowman et al. (2013); however, χ² schemes generally do not guarantee positivity and require post hoc negativity correction.

**Parametric and MCMC methods.** 
- Aschwanden & Boerner (2011): DEM을 Gaussian 형태로 강제하고 χ² 파라미터 추정.
- Kashyap & Drake (1998), Testa et al. (2012): MCMC로 DEM 사후분포 샘플링. 불확도 추정 가능하나 초당 수 개 해만 가능.

Parametric inversions (Gaussians, splines, power laws) constrain the DEM shape. MCMC (Kashyap & Drake 1998; applied to AIA by Testa et al. 2012) samples the posterior but is too slow for full-cadence map production.

### Part III: The Sparse Inversion Method / 희소 역산 방법 (§3, pp. 4–5)

**Core idea.** 압축 센싱(Candès & Tao 2006)에 착안. Underdetermined 시스템에서 참된 해가 희소(sparse)하면 L0 norm 최소화로 복구 가능. 하지만 L0은 조합폭발이므로 L1 norm으로 근사(이것이 basis pursuit, Chen, Donoho & Saunders 1998).

Inspired by compressed sensing: if the true signal is sparse, it can be recovered by L0 minimization. L0 is NP-hard; basis pursuit (L1 minimization) is the convex relaxation.

**L0 vs L1 vs LP1 formulation.**

$$
\text{L0: } \min\|\vec x\|_0 \ \text{ s.t. } \ \mathbf{D}\vec x = \vec y \quad\text{(Eq. 7)}
$$

$$
\text{L1: } \min\|\vec x\|_1 \ \text{ s.t. } \ \mathbf{D}\vec x = \vec y \quad\text{(Eq. 8)}
$$

노이즈와 양성 조건을 고려하여 실제 풀리는 형태:

$$
\boxed{
\text{LP1: } \min \sum_{j=1}^n x_j \ \text{ s.t. } \ 
\begin{cases}
\mathbf{D}\vec x \le \vec y + \vec\eta \\
\mathbf{D}\vec x \ge \max(\vec y - \vec\eta,\, 0) \\
\vec x \ge 0
\end{cases}
} \quad\text{(Eqs. 9–12)}
$$

$\vec x \ge 0$ 덕분에 $\|\vec x\|_1 = \sum x_j$로 단순해지고, 전체 문제는 표준 선형계획(LP). IDL의 `simplex` 함수(Numerical Recipes 10.8)로 해결.

Because $\vec x\ge 0$, $\|\vec x\|_1$ is simply the sum. Tolerance bands $\vec\eta$ (per-channel AIA uncertainty from `aia_bp_estimate_error`) handle noise. The LP is solved by the IDL `simplex` routine.

**Tolerance $\vec\eta$.** `aia_bp_estimate_error` (Solarsoft)로 계산. 광자 카운팅 통계, read noise, 압축/양자화 반올림, 다크 빼기 오차 포함.

The per-channel tolerance $\eta_i$ comes from `aia_bp_estimate_error`, which incorporates photon-counting statistics, read noise, compression/quantization, and dark-subtraction errors.

**Why L1?** 저자들이 언급한 세 가지 장점:
1. Parsimony / Occam's razor — basis component 수 최소화.
2. Positivity — 자연스럽게 $x_j\ge 0$ 유지.
3. 속도 — LP은 simplex로 빠르게 풀림(초당 $10^4$+).

L1 is motivated by: (1) parsimony (Occam's razor — minimizing the number of basis components), (2) natural positivity, (3) fast LP solution.

### Part IV: Validation — Log-Normal DEMs / 검증 (1): 로그 정규 분포 (§4.1, pp. 5–7)

**Validation DEM.** Gaussian in $\log T$ (Eq. 13):
$$
\xi(T, T_c, \sigma) = \frac{\mathrm{EM}_0}{\sigma\sqrt{2\pi}}\exp\!\left[-\frac{(\log T - \log T_c)^2}{2\sigma^2}\right]
$$

파라미터 범위: $\log T_c/K \in [5.5, 7.0]$, $\sigma \in [0, 0.8]$, $\mathrm{EM}_0 = 10^{29}$ cm$^{-5}$.

**Synthetic observations.** $K_i(T)$와 $\xi$를 convolve하여 $y_i$ 합성 → 각 model에 대해 5000개 noise realization 생성 → 각각 inversion.

For each model, synthetic count rates are computed and 5000 noisy realizations are generated ($y_{ij} + \alpha_j e_j$ with $\alpha_j \sim \mathcal{N}(0,1)$); the inversion is performed on every member of the ensemble.

**Fidelity metrics (Eqs. 14–16).**
$$
\mathrm{EM} = \sum_j \mathrm{EM}_j, \quad
\log T_\mathrm{EM} = \mathrm{EM}^{-1}\sum_j \mathrm{EM}_j \log T_j,\quad
W_\mathrm{EM}^2 = \mathrm{EM}^{-1}\sum_j \mathrm{EM}_j (\log T_j - \log T_\mathrm{EM})^2
$$
0차(total EM), 1차(EM-weighted log-T), 2차(thermal width) 모멘트.

Zeroth, first, and second moments of the EM distribution are used as fidelity metrics.

**Results.**
- Total EM recovered to within **10–20%** over parameter space.
- $\log T_\mathrm{EM}$ recovered to within **0.2 log T/K** error.
- $W_\mathrm{EM}$ recovered to within **0.2 log T/K** error.
- 95% confidence intervals from joint PDFs (Fig. 3) lie tightly along the diagonal.

**Systematic bias.** Fig. 4의 하단 패널이 드러내는 유일한 체계적 문제: $T_c/K = 10^{6.7}$, $\sigma = 0.7$ 조건에서 $\log T/K \gtrsim 6.6$ 영역의 EM이 체계적으로 과소추정됨. 즉 hot & broad DEM에서 고온 tail 복원이 어렵다. Testa et al. (2012)의 MCMC 결과와 일치.

The only systematic bias: for hot & broad DEMs ($T_c\sim 10^{6.7}$, $\sigma\sim 0.7$), EM above $\log T/K \sim 6.6$ is consistently underestimated, consistent with Testa et al. (2012).

**Comparison with Guennou et al. (2012b).** Guennou 등의 χ²-min (unregularized) 방식은 $\sigma=0.7$에서 $T_c$를 잘못 복원하는 systematic bias 있었으나 sparse 방법은 이 문제가 없다.

Unlike the χ²-minimization approach of Guennou et al. (2012b), which mis-locates $T_c$ for broad DEMs, the sparse method preserves peak temperature even for broad DEMs.

### Part V: Validation — NLFFF Quasi-Steady Loops / 검증 (2): NLFFF 준정상 loop (§4.2, pp. 7–8)

**Model setup.** Malanushenko & Schrijver (2015, in prep)의 AR 11158 3D 모델. AIA loop feature를 맞추는 NLFFF 외삽 (Malanushenko et al. 2014) → >7000개 flux tube로 분해. 각 튜브에 1D quasi-steady atmosphere (Schrijver & van Ballegooijen 2005) 적용. Volumetric heating rate $\epsilon \propto \Phi L^{-2.5} B(s)^{-0.5}$; total heating $\sum E_H = 5\times 10^{32}$ erg s$^{-1}$.

The Malanushenko & Schrijver (2015) AR 11158 model is an NLFFF extrapolation matched to observed loop features, decomposed into >7000 flux tubes, each carrying a 1D quasi-steady atmosphere (Schrijver & van Ballegooijen 2005). Total heating $5\times 10^{32}$ erg s$^{-1}$; two variants: Model A ($\beta=0$) and Model B ($\beta=2$) differing in $E_H\propto L^{-1.5}\Phi[B_1^{\beta-1}f(B_1)+B_2^{\beta-1}f(B_2)]$ with $f(B_\mathrm{base}) = \exp\{-(B_\mathrm{base}/500\,\mathrm{G})^2\}$.

**Results (Figs. 6, 7).** Fan loop ($\log T_\mathrm{EM}\sim 5.9$)과 core AR loop ($\log T_\mathrm{EM}\sim 6.45$)의 구분이 inversion과 ground truth 모두에서 뚜렷. 전체 EM과 $\log T_\mathrm{EM}$은 정확히 복원. Thermal width $W_\mathrm{EM}$은 상대적으로 덜 정확.

For both Models A and B, inverted total EM and EM-weighted log-T match ground truth. Core AR loops at $\log T_\mathrm{EM}\sim 6.45$ are clearly distinguished from cooler fan loops at $\log T_\mathrm{EM}\sim 5.9$. The thermal width $W_\mathrm{EM}$ is less reliably recovered (as already seen in §4.1 for broad DEMs).

### Part VI: Validation — Time-Dependent MHD / 검증 (3): 시간종속 MHD (§4.3, pp. 8–10)

**Model setup.** Chen et al. (2014)의 fully-compressible MHD AR formation 시뮬레이션, Pencil code (Brandenburg & Dobler 2002; Bingert & Peter 2011). $147.5\times 73.7$ Mm$^2$ horizontal, 50 Mm vertical. Rempel & Cheung (2014)의 flux emergence 시뮬레이션에서 광구 MHD 변수를 bottom boundary로 주입 → 코로나 loop 자발 형성.

The validation uses Chen et al. (2014)'s Pencil-code MHD simulation of AR formation with bottom-boundary MHD variables fed from Rempel & Cheung (2014)'s flux emergence run. Simulation domain $147.5\times 73.7\times 50$ Mm$^3$.

**DEM computation.** Line-of-sight integration으로 top-down 및 side view에서 DEM cube 합성 → AIA response 적용으로 synthetic 6-channel 이미지 → sparse inversion 수행 (이 경우는 노이즈 없이).

Line-of-sight DEMs were computed for top-down and side views; six AIA synthetic images were produced and sparse inversion was applied without noise.

**Results (Figs. 8–10).** 전체 EM과 $\log T_\mathrm{EM}$ 잘 복원. $W_\mathrm{EM}$ 덜 정확. Core loop (total EM $<10^{27}$ cm$^{-5}$) 영역에서 상대 오차가 order unity까지 올라갈 수 있음. Fig. 10의 4개 DEM profile 예시에서 inversion이 복잡한 DEM 형태(cool fan loop single peak, footpoint peak+broad tail, hot core loop peak+tail, mid-temperature single peak)를 잘 구분함을 확인.

Sparse inversion recovers total EM and $\log T_\mathrm{EM}$ accurately; $W_\mathrm{EM}$ less so. Pixels with total EM $<10^{26}$ cm$^{-5}$ fall below AIA sensitivity and are excluded. DEM profiles at four sampled positions (Fig. 10) show recoverable distinctions between cool fan loop peaks, core loop double-peaked distributions, and broad warm distributions.

### Part VII: Application to AR 11158 / AR 11158 실제 관측 적용 (§5, pp. 10–14)

**Target.** AR 11158은 Cycle 24 최초의 X급 플레어(X2.2, 2011-02-15)를 일으킨 활동영역. 2011-02-15 22:00 (X-flare 20시간 후, 이후 M-flare 2개와 다수 C-flare 계속); 1개월 후 decay phase 비교.

AR 11158 produced the first X-class flare (X2.2) of Cycle 24. The paper applies sparse DEM inversion at 2011-02-15 22:00 UT (>20 h after the X-flare peak, with ongoing M and C flares) and one solar rotation later.

**Results (Figs. 11, 12).**
- $\log T/K \in [5.75, 6.05]$: cool fan loop이 활동영역 주변부에 위치, 흑점 바깥쪽을 향해 fanning.
- $\log T/K \in [6.05, 6.35]$, $[6.35, 6.65]$: high-EM core loop이 반대 극성을 연결.
- $\log T/K \in [6.65, 6.95]$: flare-still 영역에서 high-EM 구조 잔존.
- 1개월 후 (Fig. 12): 동일 AR이 decay phase 진입, $\log T/K > 6.6$에서 EM이 **2–3 orders of magnitude** 감소. 자기장 약화와 shear flow 부재로 에너지 주입 감소.

Five temperature-bin EM maps at 2011-02-15 22:00 show cool fan loops ($\log T\in[5.75, 6.05]$) at the AR periphery, high-EM core loops at mid-temperatures ($\log T\in[6.35, 6.65]$) connecting opposite polarities, and residual hot material ($\log T\in[6.65, 6.95]$). One solar rotation later, EM above $\log T/K\sim 6.6$ drops by 2–3 orders of magnitude during the AR decay phase.

### Part VIII: Speed and Joint AIA+XRT / 속도와 AIA+XRT 결합 (§5.1, §6, pp. 10, 14–15)

**Computational speed.** 2.6 GHz Intel Core i7 (MacBook Pro), 단일 IDL thread: $m=6$, $n=21$, $l=84$에서 초당 $>10^4$ inversion 해. 응답 행렬 초기화 (Solarsoft `aia_get_response`로 구성, 디스크 I/O 지배)는 1회성, 많은 관측벡터에 amortize 가능.

Benchmarked on a 2.6 GHz Intel Core i7: >10^4 inversions/second with $m=6$, $n=21$, $l=84$. Setup dominated by `aia_get_response` I/O is one-time.

**Joint AIA+XRT inversion (§6).** Hinode/XRT의 Be-thin 채널 (response peak $\log T/K\sim 7.0$) 추가. `make_xrt_temp_resp`로 XRT response 생성. 5000-member noisy ensemble로 검증.

Augmenting AIA with XRT's Be-thin channel (peak response $\log T/K\sim 7.0$) and repeating the log-normal validation shows:
- Hot+broad DEM bias mostly removed (Fig. 13 vs Fig. 2).
- Joint PDFs (Fig. 14) tighten the 95% confidence interval in all three metrics.
- Even cool regime ($\log T \lesssim 6.3$) improves.

광범위한 온도에서 개선. 실제 AIA-XRT 동시 inversion 적용은 instrument 간 intercalibration이 선행되어야 하며 future work.

The improvement spans all temperatures. Application to real simultaneous AIA–XRT data requires careful cross-calibration and is reserved for future work (see also Hanneman & Reeves 2014).

### Part IX: Discussion and Caveats / 논의 및 한계 (§7, pp. 13–16)

**Appropriate use cases.** AIA 단독 inversion은 cool fan loop vs warm core loop과 같은 거친 thermal 구조 구분에 충분. 그러나 고온 영역($\log T \gtrsim 6.6$)의 DEM 기울기 측정(예: coronal heating 이론 검증)에는 부적합. 이 경우 XRT나 EIS 데이터 추가 필요.

AIA-only inversions suffice for gross thermal classification (cool fan vs. warm core) but are inadequate for measuring high-T DEM slopes (relevant to coronal heating theories). Joint inversions with XRT or EIS are recommended.

**Overfitting avoidance.** 3가지 model class 검증 → 단일 DEM 형태에 "tuning"하는 overfitting 회피. 이는 기존 논문들에서 논의되지 않은 강조점.

Testing on three DEM model classes mitigates "tuning to a specific DEM shape" — a form of methodological overfitting.

### Part X: Appendix — Quadrature Scheme / 부록: Quadrature 구성 (pp. 17–20)

**Basis dictionary.** Eq. (A1): $\mathrm{DEM}(\log T) = \sum_{k=1}^{l} b_k(\log T) x_k$.
- Dirac-delta basis: $b_k^\mathrm{Dirac}(\log T_j) = \delta_{jk}$ (21 columns → identity matrix).
- Gaussian basis (Eq. A6): $b_k^a(\log T_j) = \exp[-(\log T_j - \log T_k)^2/a^2]$ if $|\log T_j - \log T_k|\le 1.8a$ else 0. Widths $a=0.1, 0.2, 0.6$ → 21+21+21 columns.
- **Combined**: $\mathbf{B} = (\mathbf{B}^\mathrm{Dirac} | \mathbf{B}^{a=0.1} | \mathbf{B}^{a=0.2} | \mathbf{B}^{a=0.6})$ → 84 columns (Eq. A8).

The dictionary $\mathbf{B}$ has 84 columns: 21 Dirac deltas + 3×21 truncated Gaussians of widths 0.1, 0.2, 0.6. Dictionary matrix $\mathbf{D} = \mathbf{K}\mathbf{B}$ has dimensions $6\times 84$ — highly overcomplete.

**Why non-normalized Gaussians?** Gaussian basis는 peak=1로 고정(총합 normalization 없음). 이는 broad Gaussian 해가 multiple narrow Gaussian 해보다 L1-norm 관점에서 "저비용"이 되어, broad DEM 선호 bias 도입. 만약 normalize된 Gaussian을 쓰면 inversion은 broad 대신 여러 narrow Gaussian 합으로 해를 expressed → Figs. 16–17처럼 부정확한 결과.

Gaussian bases are NOT normalized by their integrals (max value = 1 regardless of width). This introduces a preference for broad solutions over many narrow ones, avoiding spurious isothermal-component solutions (Figs. 16–17).

**Why $\log T/K \ge 5.5$?** 하한을 $\log T/K = 5.0$ 또는 5.2로 확장하면 전이영역 온도에서 spurious EM 증강 발생(Fig. 18), 동시에 core loop 온도($\log T/K \in [6.4, 6.6]$)에서 EM 결손. 저자들은 경험적으로 $\log T/K \ge 5.5$ 채택.

Extending the lower bound to $\log T/K=5.0$ or 5.2 introduces spurious transition-region EM and artifactual EM deficits in core loops. Empirically, the range is restricted to $\log T/K \ge 5.5$.

---

## 3. Key Takeaways / 핵심 시사점

1. **Sparse inversion reframes DEM retrieval as an LP / 희소 역산은 DEM 역문제를 LP 문제로 재정의** — L0→L1→LP1 relaxation으로 Craig & Brown (1976)의 4가지 병적 성격 중 존재(tolerance band), 유일성(sparsity prior), 안정성(LP duality), 양성(bound constraint)을 한꺼번에 해결. 전통적 χ² 방법이 필요로 했던 post hoc 음수 수정이 불필요.
   L0→L1→LP1 relaxation simultaneously addresses all four Craig–Brown pathologies in a single formulation. No post-hoc negativity corrections are required.

2. **Computational throughput enables routine DEM map production / 계산 처리량이 일상적 DEM 맵 생산을 가능케 함** — 2.6 GHz CPU에서 초당 $>10^4$ 해. AIA는 12초마다 $\sim 10^6$ 관측벡터 생성하므로 subsampling+multithreading으로 실시간 DEM map 생산이 실현 가능. MCMC(초당 수 개)는 비현실적.
   At >10^4 inversions/second, the method scales to AIA's data rate (~10^6 observation vectors per 12 s) with subsampling and multithreading, unlike MCMC (few/s).

3. **Three-tier validation prevents methodological overfitting / 3단계 검증이 방법론적 overfitting을 방지** — Gaussian log-normal, NLFFF quasi-steady loops, 시간종속 MHD 세 가지 물리 모델로 검증 → 단일 DEM 형태에 튜닝된 bias 배제. 이는 DEM inversion 논문에서 드물게 엄격한 검증.
   Validation across three physically distinct model classes is unusually rigorous and rules out tuning bias.

4. **AIA alone resolves gross thermal structure but not high-T slopes / AIA 단독으로는 거친 구조 구분 가능, 고온 기울기는 불가** — Cool fan loop (log T ~5.9) vs warm core loop (log T ~6.45) 구별은 정확. 그러나 $\log T/K \gtrsim 6.6$ DEM의 기울기 측정은 체계적 bias 있어, coronal heating frequency test(Warren et al. 2012)에 AIA alone 부적합.
   AIA alone distinguishes cool fan from warm core structures but has systematic bias in the high-T slope — inadequate for heating-frequency tests without XRT/EIS.

5. **Joint AIA+XRT inversion is a natural extension / AIA+XRT 결합 역산은 자연스러운 확장** — XRT Be-thin ($\log T/K\sim 7.0$) 추가만으로 hot+broad DEM bias 거의 제거, 95% 신뢰구간 크게 좁힘. 방법론적으로 다중 장비 DEM 연구의 표준 프레임워크 제시.
   Adding XRT's Be-thin channel removes the hot-broad bias and tightens confidence intervals — a blueprint for multi-instrument DEM studies.

6. **Dictionary design matters — unnormalized Gaussians favor broad solutions / 사전행렬 설계가 중요 — 정규화되지 않은 Gaussian은 broad 해 선호** — Peak=1로 유지된 Gaussian basis는 L1-norm 관점에서 broad DEM을 "저비용"으로 만들어 spurious isothermal 성분 해 방지. 작은 설계 결정이 큰 해석적 결과를 가져옴.
   The choice to leave Gaussian bases unnormalized (max = 1) is a small design decision with large interpretive consequences: it biases solutions toward broad DEMs over isothermal components.

7. **AR 11158 analysis demonstrates thermal evolution / AR 11158 분석이 열적 진화 실증** — 하나의 AR를 flaring phase(Feb 15)와 decay phase(1 rotation later) 두 번 관측해 $\log T/K > 6.6$에서 EM이 2–3 orders 감소하는 것을 정량화. 단순 이미지로는 보이지 않는 thermal energy budget 변화를 DEM map이 드러냄.
   The two-epoch AR 11158 analysis quantifies a 2–3 orders-of-magnitude drop in high-T EM from flaring to decay phase — a thermal diagnostic invisible in raw EUV images.

8. **The sparse code is now a community tool / 희소 코드는 커뮤니티 표준 도구가 됨** — `aia_sparse_em_init`/`aia_sparse_em_solve`로 SolarSoft에 포함. Nanoflare heating 검증, flare thermal structure, AR thermal evolution 등 수많은 후속 연구의 backbone. 이 논문은 단순히 방법 제안에 그치지 않고 solar physics DEM 연구의 표준을 바꿈.
   The code has become a standard SolarSoft tool (`aia_sparse_em_init`/`aia_sparse_em_solve`), underpinning nanoflare-heating tests, flare thermal structure studies, and AR thermal evolution — this paper shifted the community standard.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Forward Model / 정방향 모델

**Continuous form:**
$$
y_i = \int_0^\infty K_i(T)\, \mathrm{DEM}(T)\, dT
$$
- $y_i$ [DN s$^{-1}$ pixel$^{-1}$]: exposure-normalized signal in channel $i$.
- $K_i(T)$ [DN cm$^5$ s$^{-1}$ pixel$^{-1}$]: temperature response.
- $\mathrm{DEM}(T)\,dT = \int n_e^2(T)\,dz$ [cm$^{-5}$]: differential emission measure.

**Discretized form (Dirac basis, $n$ bins):**
$$
y_i = \sum_{j=1}^n K_{ij}\,\mathrm{EM}_j, \quad \mathrm{EM}_j = \int_{\log T_j}^{\log T_j + \Delta\log T} \mathrm{DEM}(\log T)\,d\log T
$$

**Matrix form with dictionary:**
$$
\vec y = \mathbf{D}\vec x, \quad \mathbf{D} = \mathbf{K}\mathbf{B}, \quad \mathbf{B}\in\mathbb{R}^{n\times l}
$$

For AIA: $\mathbf{D}\in\mathbb{R}^{6\times 84}$ (severely underdetermined).

### 4.2 Inversion Schemes Compared

**(a) χ² minimization (standard).**
$$
\chi^2(\vec x) = \sum_{i=1}^m \left(\frac{y_i - \sum_j K_{ij} x_j}{\delta y_i}\right)^2
$$

**(b) Tikhonov (smoothness) regularized:**
$$
\min_{\vec x} \ \chi^2(\vec x) + \lambda \sum_i (x_{i-1}-2x_i+x_{i+1})^2
$$

**(c) Zeroth-order regularized (Hannah & Kontar 2012; Plowman et al. 2013):**
$$
\min_{\vec x} \ \chi^2(\vec x) + \lambda \|\vec x\|_2^2
$$
SVD-solvable, no positivity guarantee.

**(d) Parametric (Gaussian, Aschwanden & Boerner 2011):**
$$
\mathrm{DEM}(T; T_c, \sigma, \mathrm{EM}_0) = \frac{\mathrm{EM}_0}{\sigma\sqrt{2\pi}}\exp[-(\log T-\log T_c)^2/(2\sigma^2)]
$$
then minimize $\chi^2$ over $(T_c, \sigma, \mathrm{EM}_0)$.

**(e) MCMC (Kashyap & Drake 1998).** Metropolis–Hastings sampling of DEM posterior.

**(f) THIS PAPER — Sparse LP1:**
$$
\boxed{
\begin{aligned}
\min_{\vec x}\quad & \sum_{j=1}^n x_j \\
\text{s.t.}\quad & \mathbf{D}\vec x \le \vec y + \vec\eta \\
& \mathbf{D}\vec x \ge \max(\vec y - \vec\eta, 0) \\
& \vec x \ge 0
\end{aligned}
}
$$
where $\eta_i$ = per-channel AIA uncertainty from `aia_bp_estimate_error`.

### 4.3 Validation DEMs

**Log-normal Gaussian (Eq. 13):**
$$
\xi(T, T_c, \sigma) = \frac{\mathrm{EM}_0}{\sigma\sqrt{2\pi}}\exp\!\left[-\frac{(\log T-\log T_c)^2}{2\sigma^2}\right]
$$
with $\mathrm{EM}_0 = 10^{29}$ cm$^{-5}$, $\log T_c/K\in[5.5,7.0]$, $\sigma\in[0,0.8]$.

**Relation between DEM and $\xi$:**
$$
\mathrm{DEM}(T) = \ln 10^{-1}\, \xi(T)
$$

### 4.4 Fidelity Metrics (Moments of EM)

$$
\mathrm{EM} = \sum_j \mathrm{EM}_j \qquad\text{(0th moment)}
$$
$$
\log T_\mathrm{EM} = \mathrm{EM}^{-1}\sum_j \mathrm{EM}_j \log T_j \qquad\text{(1st moment)}
$$
$$
W_\mathrm{EM}^2 = \mathrm{EM}^{-1}\sum_j \mathrm{EM}_j (\log T_j - \log T_\mathrm{EM})^2 \qquad\text{(2nd moment)}
$$

### 4.5 Dictionary Construction (Appendix A)

**Dirac-delta basis (21 columns):**
$$
b_k^\mathrm{Dirac}(\log T_j) = \begin{cases} 1 & \text{if } \log T_j = \log T_k \\ 0 & \text{otherwise} \end{cases}
$$

**Truncated Gaussian basis of width $a$ (21 columns each for $a\in\{0.1, 0.2, 0.6\}$):**
$$
b_k^a(\log T_j) = \begin{cases} \exp\!\left[-(\log T_j - \log T_k)^2/a^2\right] & \text{if } |\log T_j - \log T_k|\le 1.8a \\ 0 & \text{otherwise} \end{cases}
$$

**Combined basis (84 columns):**
$$
\mathbf{B} = (\mathbf{B}^\mathrm{Dirac} \,|\, \mathbf{B}^{a=0.1} \,|\, \mathbf{B}^{a=0.2} \,|\, \mathbf{B}^{a=0.6})
$$

Note: Gaussians are NOT normalized by integral; their max value is 1 for all widths.

### 4.6 AIA Passband Peak Temperatures

| Channel | Dominant Ion | $\log T_\mathrm{peak}/K$ |
|---|---|---|
| 94 Å | Fe XVIII | 6.85 (hot) + Fe X (6.05, cool) |
| 131 Å | Fe XX/XXIII + Fe VIII | 7.05 + 5.6 |
| 171 Å | Fe IX | 5.85 |
| 193 Å | Fe XII/XXIV | 6.20 + 7.25 |
| 211 Å | Fe XIV | 6.30 |
| 335 Å | Fe XVI | 6.45 |

이 이중 피크 구조가 6-채널로 넓은 온도범위를 cover하게 해주지만, 동시에 degenerate한 inversion 해를 만들 수 있어 sparsity prior가 필요한 이유이기도 하다.

These double-peaked response functions both cover a wide temperature range and create inversion degeneracies — a key motivation for the sparsity prior.

### 4.7 Numerical Example — Isothermal Source

Suppose a purely isothermal column at $\log T/K = 6.30$ (Fe XIV peak) with $\mathrm{EM} = 10^{28}$ cm$^{-5}$. Using $K_{211}(6.30)\approx 10^{-24}$ DN cm$^5$ s$^{-1}$ pixel$^{-1}$ (from Fig. 1):
$$
y_{211} = K_{211}(6.30)\times \mathrm{EM} = 10^{-24} \times 10^{28} = 10^{4}\ \mathrm{DN\, s^{-1}\,pixel^{-1}}
$$
즉 $\sim 10^4$ DN/s의 신호. 다른 채널들도 유사 계산으로 Fe XIV peak 근처 온도에서 211 Å가 dominant함을 확인. Inversion은 이 6개 값 $(y_{94},\ldots,y_{335})$ + tolerance $(\eta_{94},\ldots,\eta_{335})$로 LP1을 풀어 Dirac basis 단일 성분 $x_{k: T_k=10^{6.3}}$ 해를 복원한다.

For an isothermal column at $\log T/K=6.30$ with EM $=10^{28}$ cm$^{-5}$, the 211 Å channel receives $\sim 10^4$ DN/s/pixel. The inversion recovers a single Dirac component at $T = 10^{6.3}$ K.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
 1953 ── Courant & Hilbert: Fredholm integral equations
            │
 1962 ── Phillips: regularization of ill-posed problems
            │
 1976 ── Craig & Brown: DEM inversion pathologies
            │
 1986 ── Craig & Brown book: "Inverse Problems in Astronomy"
            │
 1992 ── Monsignori Fossi & Landini: smoothness regularization
            │
 1995 ── Hubeny & Judge: max entropy DEM
            │
 1998 ── Chen, Donoho & Saunders: basis pursuit (atomic decomposition)
            │                    Kashyap & Drake: MCMC DEM
            │
 2006 ── Candès & Tao: compressed sensing theory
            │
 2010 ── SDO launch: AIA full-disk 12-s EUV
            │
 2011 ── Aschwanden & Boerner: Gaussian parametric AIA DEM
            │
 2012 ── Hannah & Kontar: zeroth-order regularized AIA DEM
       ── Testa et al.: AIA+MCMC 3D thermal diagnostics
       ── Guennou et al.: systematic test of AIA 6-channel DEM
            │
 2013 ── Plowman, Kankelborg & Martens: fast SVD+positivity DEM
            │
 2014 ── Chen et al.: 3D MHD AR formation
       ── Rempel & Cheung: flux emergence simulation
            │
╔══════════╪══════════════════════════════════════════════════╗
║ 2015 ── Cheung et al. (THIS PAPER): sparse/basis-pursuit    ║
║         DEM inversion with 3-tier validation + AIA+XRT      ║
╚══════════╪══════════════════════════════════════════════════╝
            │
 2018+ ── Widespread adoption: sparse method as SolarSoft standard
            │  (nanoflare heating, flare thermal studies,
            │   AR thermal evolution, joint AIA-XRT/EIS)
            │
 2020+ ── ML-based DEM inversion benchmarks use this method
            │
 2025+ ── MUSE multi-slit extension of sparse DEM methodology
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Lemen et al. 2012 (AIA instrument paper) / Boerner et al. 2012 | Provides instrument description, response functions, calibration on which this paper builds / AIA 기기 설명 및 응답 함수 기본 제공 | Essential prerequisite — defines $K_i(T)$ used here / $K_i(T)$ 정의 필수 선행 |
| Craig & Brown 1976 | Identifies the four pathologies that this paper's LP1 formulation resolves / 본 논문이 해결하는 4대 병적 성격 지적 | Foundational — frames the problem / 문제 구조 설정의 근간 |
| Candès & Tao 2006; Chen, Donoho & Saunders 1998 | Compressed sensing + basis pursuit foundation for the sparse method / 압축 센싱과 basis pursuit 수학적 기반 | Methodological backbone / 방법론의 수학적 뼈대 |
| Hannah & Kontar 2012 | Competing zeroth-order-regularized DEM inversion for AIA; compared throughout / 본 논문과 비교되는 대표적 경쟁 방법 | Direct methodological comparison / 직접적 방법 비교 |
| Plowman, Kankelborg & Martens 2013 | Another fast χ² + positivity AIA DEM method; discussed in §2.3 / 또 다른 빠른 AIA DEM 기법 | Contemporary alternative / 동시대 대안 |
| Aschwanden & Boerner 2011 | Parametric Gaussian DEM from AIA; forms the functional form used in §4.1 validation / AR DEM 파라메트릭 기법 | Comparison point + validation DEM form / 비교 대상 + 검증 DEM 형태 |
| Testa et al. 2012 | MCMC AIA DEM reliability study; informs validation design / MCMC AIA DEM 신뢰성 분석 | Motivates validation approach / 검증 접근법 동기 |
| Guennou et al. 2012a,b | Systematic χ² AIA DEM test; identified the hot-broad DEM bias addressed here / 6-채널 한계 체계적 분석 | Sets baseline systematic biases / baseline systematic 제공 |
| Chen et al. 2014; Rempel & Cheung 2014 | MHD simulation used as §4.3 validation truth / §4.3 검증에 사용된 MHD 시뮬레이션 | Provides ground-truth DEMs / 검증용 ground truth |
| Malanushenko et al. 2014; Malanushenko & Schrijver 2015 | NLFFF + quasi-steady loop model used as §4.2 validation / §4.2 검증 모델 | Ground-truth for quasi-steady loops / quasi-steady loop 검증 |
| Golub et al. 2007; Kosugi et al. 2007 | Hinode/XRT and the instrument; source of the Be-thin channel used in §6 / Hinode/XRT 기기 | Enables joint DEM inversion / 결합 역산의 기반 |
| Warren et al. 2012 | Coronal heating frequency via high-T DEM slopes — case where AIA-only is insufficient / 고온 DEM 기울기 기반 nanoflare 검증 | Scientific use case limiting AIA-only / AIA 단독 한계 사례 |
| Paper #28 (Rempel & Cheung 2014 or earlier simulation paper, if in reading list) | Same MHD framework producing the AR used in §4.3 / §4.3의 AR MHD 모델 생성 | Shared computational framework / 공통 계산 기반 |

---

## 7. References / 참고문헌

- Cheung, M. C. M., Boerner, P., Schrijver, C. J., Testa, P., Chen, F., Peter, H., & Malanushenko, A., "Thermal Diagnostics with the Atmospheric Imaging Assembly onboard the Solar Dynamics Observatory: A Validated Method for Differential Emission Measure Inversions," ApJ, 807, 143 (2015). [DOI: 10.1088/0004-637X/807/2/143]
- Lemen, J. R., et al., "The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)," Sol. Phys., 275, 17 (2012).
- Boerner, P., et al., "Initial Calibration of the Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)," Sol. Phys., 275, 41 (2012).
- Boerner, P. F., et al., "Photometric and Thermal Cross-calibration of Solar EUV Instruments," Sol. Phys., 289, 2377 (2014).
- Craig, I. J. D., & Brown, J. C., "Fundamental Limitations of X-ray Spectra as Diagnostics of Plasma Temperature Structure," A&A, 49, 239 (1976).
- Craig, I. J. D., & Brown, J. C., Inverse Problems in Astronomy: A Guide to Inversion Strategies for Remotely Sensed Data (1986).
- Phillips, D. L., "A Technique for the Numerical Solution of Certain Integral Equations of the First Kind," J. ACM, 9, 84 (1962).
- Candès, E., & Tao, T., "Near-optimal Signal Recovery from Random Projections: Universal Encoding Strategies?" IEEE Trans. Inf. Theory, 52, 5406 (2006).
- Chen, S. S., Donoho, D. L., & Saunders, M. A., "Atomic Decomposition by Basis Pursuit," SIAM J. Sci. Comput., 20, 33 (1998).
- Hannah, I. G., & Kontar, E. P., "Differential Emission Measures from the Regularized Inversion of Hinode and SDO Data," A&A, 539, A146 (2012).
- Plowman, J., Kankelborg, C., & Martens, P., "Fast Differential Emission Measure Inversion of Solar Coronal Data," ApJ, 771, 2 (2013).
- Aschwanden, M. J., & Boerner, P., "Solar Corona Loop Studies with the Atmospheric Imaging Assembly: I. Cross-sectional Temperature Structure," ApJ, 732, 81 (2011).
- Testa, P., De Pontieu, B., Martinez-Sykora, J., Hansteen, V., & Carlsson, M., "Testing Differential Emission Measure Measurements with AIA," ApJ, 758, 54 (2012).
- Guennou, C., et al., "On the Accuracy of the Differential Emission Measure Diagnostics of Solar Plasmas. I, II," ApJS, 203, 25, 26 (2012a,b).
- Kashyap, V., & Drake, J. J., "Markov Chain Monte Carlo Reconstruction of Emission Measure Distributions: Application to Solar Extreme-Ultraviolet Spectra," ApJ, 503, 450 (1998).
- Landi, E., Young, P. R., Dere, K. P., Del Zanna, G., & Mason, H. E., "CHIANTI — an Atomic Database for Emission Lines. XIII. Soft X-Ray Improvements and Other Changes," ApJ, 763, 86 (2013).
- Chen, F., Peter, H., Bingert, S., & Cheung, M. C. M., "A Model for the Formation of Loops within a Corona," A&A, 564, A12 (2014).
- Rempel, M., & Cheung, M. C. M., "Numerical Simulations of Active Region Scale Flux Emergence," ApJ, 785, 90 (2014).
- Malanushenko, A., Schrijver, C. J., DeRosa, M. L., & Wheatland, M. S., "Using the Parameterized Magnetic Field Approximation for NLFFF Extrapolation," ApJ, 783, 102 (2014).
- Golub, L., et al., "The X-Ray Telescope (XRT) for the Hinode Mission," Sol. Phys., 243, 63 (2007).
- Warren, H. P., Winebarger, A. R., & Brooks, D. H., "A Systematic Survey of High-Temperature Emission in Solar Active Regions," ApJ, 759, 141 (2012).
- Dantzig, G. B., Orden, A., & Wolfe, P., "The Generalized Simplex Method for Minimizing a Linear Form under Linear Inequality Restraints," Pacific J. Math., 5, 183 (1955).
- Press, W. H., Flannery, B. P., Teukolsky, S. A., & Vetterling, W. T., Numerical Recipes: The Art of Scientific Computing, Cambridge Univ. Press (1986).
- Schrijver, C. J., & van Ballegooijen, A. A., "Is the Quiet-Sun Corona a Quasi-Steady, Force-Free Environment?" ApJ, 630, 552 (2005).
