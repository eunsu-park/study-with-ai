---
title: "Multi-frame blind deconvolution with linear equality constraints"
authors: [Löfdahl, M. G.]
year: 2002
journal: "Proc. SPIE 4792-21, Image Reconstruction from Incomplete Data II"
doi: "10.1117/12.451791"
topic: Solar_Observation
tags: [MFBD, phase-diversity, PDS, Shack-Hartmann, image-restoration, wavefront-sensing, MOMFBD]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 32. Multi-frame Blind Deconvolution with Linear Equality Constraints / 선형 등식 제약조건이 있는 다중프레임 블라인드 디컨볼루션

---

## 1. Core Contribution / 핵심 기여

**English.** Löfdahl (2002) re-casts the Phase-Diverse Speckle (PDS) image restoration problem as plain Multi-Frame Blind Deconvolution (MFBD) supplemented by a set of **Linear Equality Constraints (LECs)** on the wavefront expansion coefficients $\alpha_{jm}$. In the simplest form the algorithm maximises a regularised Gaussian-noise likelihood whose object dependence has been analytically eliminated by the Wiener-filter estimate. The key observation is that the difference between PD, PDS, plain MFBD, and even Shack–Hartmann wavefront sensing amounts entirely to *which linear relations hold between the $\alpha_{jm}$ in different channels*. These relations can be collected into a matrix $\mathbf{C}$, and the constrained optimisation $\min L$ subject to $\mathbf{C}\boldsymbol\alpha = \mathbf{d}$ is transformed into an *unconstrained* optimisation in a lower-dimensional parameter $\boldsymbol\beta$ via a null-space basis $\mathbf{Q}_2$ obtained from a QR (or SVD) factorisation of $\mathbf{C}^{\top}$. Thus a single code, whose gradient/Hessian machinery is the MFBD one (block-diagonal, channel-independent), handles every data-collection scheme simply by loading a different $\mathbf{C}$. Löfdahl further lists a menu of extensions — different phase parameterisations per frame, multiple objects, joint treatment of different wavelengths, variable diversity count — each of which just adds rows to $\mathbf{C}$. This formulation is the theoretical skeleton of what would later become the MOMFBD package (van Noort, Rouppe van der Voort, Löfdahl 2005), the standard post-processing pipeline of the Swedish Solar Telescope, GREGOR, and DKIST.

**Korean.** Löfdahl(2002)은 위상 다양성 스펙클(PDS) 영상 복원 문제를 일반 다중프레임 블라인드 디컨볼루션(MFBD)에 파면 확장 계수 $\alpha_{jm}$에 대한 **선형 등식 제약조건(LEC)** 집합을 더한 것으로 재해석한다. 가장 단순한 형태로 이 알고리즘은 가우시안 잡음 가정하의 정규화된 우도를 최대화하며, 물체 의존성은 Wiener 필터 추정으로 해석적으로 제거되어 있다. 핵심 관찰은 PD, PDS, 일반 MFBD, 심지어 Shack–Hartmann 파면 센싱의 차이가 오직 *서로 다른 채널의 $\alpha_{jm}$ 사이에 어떤 선형 관계가 성립하는가*에 달려 있다는 것이다. 이 관계를 행렬 $\mathbf{C}$에 모으면 제약조건 최적화 $\min L\ \text{s.t.}\ \mathbf{C}\boldsymbol\alpha = \mathbf{d}$가 $\mathbf{C}^{\top}$의 QR(또는 SVD) 분해로 얻은 영공간 기저 $\mathbf{Q}_2$를 통해 낮은 차원의 $\boldsymbol\beta$에 대한 *무제약* 최적화로 바뀐다. 따라서 MFBD(블록 대각, 채널 독립)용 경사도/헤시안 기계를 가진 하나의 코드가 $\mathbf{C}$만 바꿔 실으면 모든 데이터 수집 방식을 다룰 수 있다. Löfdahl은 또한 프레임별 다른 위상 모수화, 복수 물체, 다파장 공동 처리, 가변 다이버시티 개수 등 각 확장이 $\mathbf{C}$에 행을 추가하는 것에 불과하다는 메뉴를 제시한다. 이 수식화는 훗날 MOMFBD 패키지(van Noort, Rouppe van der Voort, Löfdahl 2005)가 되는 이론적 골격이며, 스웨덴 태양 망원경(SST), GREGOR, DKIST의 표준 후처리 파이프라인이다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Motivation (Sec. 1) / 서론과 동기

**English.** The paper opens by framing an old challenge: single-frame blind deconvolution is ill-posed — the convolution equation $d = f * s + n$ leaves the object $f$ and PSF $s$ jointly undetermined. Multi-frame methods add leverage by assuming a *common object* across frames, but even this is not enough without prior structure on the PSFs. One way to add structure is to constrain each PSF to come from a physical pupil: $P_j = A_j e^{i\phi_j}$ with $\phi_j$ expanded in a finite basis $\{\psi_m\}$. Löfdahl points out that although such parameterised MFBD *works* (Refs. 2, 3), adding more physical information — such as an *intentionally defocused* channel paired with each in-focus channel — works better (Ref. 4 Tyler et al. 1998). PD is this diversity trick with one focus/defocus pair; PDS combines many such pairs across time (short exposures through seeing). The paper's thesis is that all these variants share one underlying algorithm — differ only by *constraints on* $\alpha_{jm}$.

**Korean.** 논문은 오래된 문제로 시작한다: 단일 프레임 블라인드 디컨볼루션은 부적절(ill-posed)하다 — 컨볼루션 방정식 $d = f * s + n$에서 물체 $f$와 PSF $s$는 공동으로 미정 상태다. 다중 프레임 방법은 프레임 전체에 걸친 *공통 물체*를 가정해 지렛대를 얻지만, PSF에 대한 사전 구조 없이는 여전히 부족하다. 구조를 더하는 한 방법은 각 PSF를 물리적 동공에서 오도록 제약하는 것이다: $P_j = A_j e^{i\phi_j}$, $\phi_j$는 유한 기저 $\{\psi_m\}$로 전개. Löfdahl은 이런 모수화된 MFBD가 *작동*은 하지만(참조 2, 3), 더 많은 물리 정보 — 예컨대 각 초점 채널에 *의도적으로 초점을 벗어난* 채널을 짝지음 — 를 더하면 더 잘 작동한다고 지적한다(참조 4 Tyler 등 1998). PD는 이 다이버시티 기법의 1초점/1디포커스 쌍 버전이고, PDS는 시간에 걸쳐(시잉 통과 단시간 노출) 여러 쌍을 결합한 것이다. 논문의 주장은 이 모든 변형이 하나의 기저 알고리즘을 공유하며 — 오직 $\alpha_{jm}$에 대한 *제약*만 다르다는 것이다.

### Part II: Forward Model and ML Error Metric (Sec. 2.1) / 전방 모델과 ML 오차 척도

**English.** The imaging model is isoplanatic (a single PSF per frame, valid across the field of view) with additive Gaussian white noise. Each frame $j \in \{1,\ldots,J\}$ has a complex pupil

$$P_j = A_j \exp\{i\phi_j\}, \qquad s_j = |\mathcal{F}^{-1}\{P_j\}|^2,$$

and the data, in the Fourier domain, satisfy $D_j = F\cdot S_j + N_j$. The phase $\phi_j$ is expanded as

$$\phi_j = \theta_j + \sum_{m=1}^{M} \alpha_{jm}\psi_m,$$

where $\theta_j$ holds any **known** piece (e.g. the defocus of the diverse channel) and $\{\psi_m\}$ are typically Karhunen–Loève (KL) modes — the orthogonal basis that diagonalises the Kolmogorov covariance of atmospheric phase. For solar work one usually truncates to $M \approx 15{-}40$ KL modes, much fewer than Zernikes required for the same RMS residual.

Gaussian noise makes the Wiener-filter object estimate available in closed form:

$$F = \frac{1}{Q}\sum_j S_j^{*} D_j, \qquad Q = \gamma_{\rm obj} + \sum_j |S_j|^2.$$

Substituting this back into the log-likelihood yields a metric in the phase parameters *alone*:

$$L(\boldsymbol\alpha) = \sum_u\!\left[\sum_j|D_j|^2 - \frac{\left|\sum_j D_j^{*}S_j\right|^2}{Q}\right] + \frac{\gamma_{\rm wf}}{2}\sum_m\frac{1}{\lambda_m}\sum_j|\alpha_{jm}|^2. \quad (6)$$

The regulariser $\gamma_{\rm obj}$ in $Q$ stabilises object perturbations; $\gamma_{\rm wf}/\lambda_m$ penalises implausibly large mode coefficients (where $\lambda_m$ is the expected KL-mode variance under Kolmogorov statistics). Löfdahl stresses that this metric is *pure MFBD* — phase diversity enters only through $\theta_j$ or, in the new formulation, through the LEC matrix $\mathbf{C}$.

**Korean.** 영상 모델은 등평면(한 프레임에 하나의 PSF, 시야 전체에 유효)이며 가산적 가우시안 백색 잡음을 가정한다. 각 프레임 $j \in \{1,\ldots,J\}$는 복소 동공을 가진다

$$P_j = A_j \exp\{i\phi_j\}, \qquad s_j = |\mathcal{F}^{-1}\{P_j\}|^2,$$

데이터는 푸리에 공간에서 $D_j = F\cdot S_j + N_j$를 만족한다. 위상 $\phi_j$는

$$\phi_j = \theta_j + \sum_{m=1}^{M} \alpha_{jm}\psi_m$$

로 확장되며, $\theta_j$는 **알려진** 부분(예: 다이버시티 채널의 디포커스)을 담고, $\{\psi_m\}$은 보통 Karhunen–Loève(KL) 모드 — 대기 위상의 Kolmogorov 공분산을 대각화하는 직교 기저. 태양 관측에서는 보통 $M \approx 15{-}40$ KL 모드로 절단하며, 같은 RMS 잔차를 얻는 데 필요한 Zernike 모드 수보다 훨씬 적다.

가우시안 잡음 덕에 Wiener 필터 물체 추정을 폐쇄형으로 얻는다:

$$F = \frac{1}{Q}\sum_j S_j^{*} D_j, \qquad Q = \gamma_{\rm obj} + \sum_j |S_j|^2.$$

이를 대입하면 위상 모수 *만*의 척도를 얻는다(식 6 위). $Q$의 $\gamma_{\rm obj}$는 물체 섭동을 안정화하고, $\gamma_{\rm wf}/\lambda_m$은 비현실적으로 큰 모드 계수에 벌점을 부과한다($\lambda_m$은 Kolmogorov 통계하 KL 모드의 기대 분산). Löfdahl은 이 척도가 *순수한 MFBD* 임을 강조한다 — 위상 다양성은 오직 $\theta_j$를 통해서, 또는 새 수식화에서는 LEC 행렬 $\mathbf{C}$를 통해서만 들어간다.

### Part III: Gradient and Hessian for Classical PD (Sec. 2.2) / 고전 PD용 경사도와 헤시안

**English.** For traditional PD, all channels share a single $\boldsymbol\alpha = (\alpha_1,\ldots,\alpha_M)$ (the common phase). The Newton-type update solves

$$\mathbf{A}^{\rm PD}\cdot\delta\boldsymbol\alpha - \mathbf{b}^{\rm PD} \simeq 0. \quad (7)$$

The gradient components are Euclidean inner products $\langle\cdot,\cdot\rangle$ of a real-space expression with the basis function $\psi_m$:

$$b_m^{\rm PD} = \left\langle -2\sum_{j=1}^{J}\mathrm{Im}\bigl[P_j^{*}\mathcal{F}\bigl\{p_j\,\mathrm{Re}\!\bigl[\mathcal{F}^{-1}\{F^{*}D_j - |F|^2 S_j\}\bigr]\bigr\}\bigr], \psi_m \right\rangle + \gamma_{\rm wf}\frac{\alpha_m}{\lambda_m}. \quad (8)$$

Here $p_j = \mathcal{F}^{-1}\{P_j\}$ is the coherent amplitude, and the quantity in the inner bracket is the part of the gradient that links image residuals back to pupil-plane phase. An approximate (Gauss–Newton-like) Hessian $\mathbf{A}^{\rm PD}$ is obtained by treating $Q$ as fixed (Vogel, Chan, Plemmons 1998); its elements involve a double sum over channel pairs $j, j'$ (Eqs. 9–10).

**Korean.** 고전 PD에서는 모든 채널이 하나의 $\boldsymbol\alpha = (\alpha_1,\ldots,\alpha_M)$(공통 위상)을 공유한다. 뉴턴형 갱신은 식 (7)을 푼다. 경사도 성분은 실공간 표현과 기저 함수 $\psi_m$의 유클리드 내적이다(식 8). 여기서 $p_j = \mathcal{F}^{-1}\{P_j\}$는 결맞음 진폭이고, 내부 대괄호의 양은 영상 잔차를 동공면 위상과 연결하는 경사도의 핵심이다. 근사(가우스–뉴턴형) 헤시안 $\mathbf{A}^{\rm PD}$은 $Q$를 고정으로 취급해 얻으며(Vogel, Chan, Plemmons 1998), 그 원소는 채널 쌍 $j, j'$에 대한 이중 합을 포함한다(식 9–10).

### Part IV: Gradient and Hessian for plain MFBD (Sec. 2.3) / 일반 MFBD용 경사도와 헤시안

**English.** For plain MFBD, phases in *different* channels are assumed independent. Löfdahl derives the gradient/Hessian by starting from the PD expressions and **dropping any cross-$j$ couplings**. Formally each independent variable $\alpha_m$ is now split into $J$ independent variables $\alpha_{jm}$, and because a sum of $J$ identical $d\alpha_{jm}$ is $\sqrt{J}$ times one $d\alpha_m$, the gradient must be multiplied by $\sqrt{J}$ and the Hessian by $J$. Arranging the coefficients lexicographically,

$$\boldsymbol\alpha = (\alpha_{11}, \alpha_{12}, \ldots, \alpha_{JM})^{\top} \in \mathbb{R}^{N},\quad N = JM,$$

the MFBD gradient components are

$$b_{jm}^{\rm MFBD} = \left\langle -2\sqrt{J}\,\mathrm{Im}\bigl[P_j^{*}\mathcal{F}\{p_j\,\mathrm{Re}[\mathcal{F}^{-1}\{F^{*}D_j - |F|^2 S_j\}]\}\bigr], \psi_m \right\rangle + \gamma_{\rm wf}\frac{\alpha_{jm}}{\lambda_m}, \quad (13)$$

and the $N\times N$ Hessian is **block-diagonal** with $J$ blocks of size $M\times M$ (Eq. 14). Each block is given by Eq. (15) with the auxiliary

$$V_{ij} = \frac{D_i}{Q}\mathcal{F}\{\mathrm{Im}[p_j^{*}\mathcal{F}^{-1}\{\psi_{m'}P_j\}]\}. \quad (16)$$

Block-diagonality is crucial: it makes MFBD numerically cheap compared with PD even when $J$ is large, because one solves $J$ decoupled $M\times M$ systems rather than a single $JM\times JM$ system.

**Korean.** 일반 MFBD에서는 *서로 다른* 채널의 위상이 독립이라고 가정한다. Löfdahl은 PD 식에서 **모든 교차-$j$ 결합을 제거**해 경사도/헤시안을 유도한다. 형식적으로 각 독립 변수 $\alpha_m$은 이제 $J$개의 독립 변수 $\alpha_{jm}$으로 분할되며, $J$개의 동일한 $d\alpha_{jm}$의 합은 하나의 $d\alpha_m$의 $\sqrt{J}$배이므로 경사도에 $\sqrt{J}$, 헤시안에 $J$를 곱해야 한다. 계수를 사전식으로 배열하면 $\boldsymbol\alpha \in \mathbb{R}^N$, $N = JM$이고, MFBD 경사도(식 13)와 **블록 대각** $N\times N$ 헤시안(식 14)이 얻어진다. 각 블록은 보조량 $V_{ij}$(식 16)를 사용한 식 (15)로 주어진다. 블록 대각성은 중요하다: $J$가 커도 단일 $JM\times JM$ 시스템이 아니라 $J$개의 분리된 $M\times M$ 시스템을 풀면 되므로 PD보다 수치적으로 저렴하다.

### Part V: The Heart of the Paper — Linear Equality Constraints (Sec. 3.1) / 선형 등식 제약조건 (논문의 핵심)

**English.** The constraints take the form

$$\mathbf{C}\cdot\boldsymbol\alpha - \mathbf{d} = 0, \quad (17)$$

with $\mathbf{C} \in \mathbb{R}^{N_C\times N}$ and $N_C < N$. All solutions form an affine subspace

$$\boldsymbol\alpha = \bar{\boldsymbol\alpha} + \mathbf{Q}_2\boldsymbol\beta, \quad (18)$$

where $\bar{\boldsymbol\alpha}$ is any particular solution and the $N' = N - N_C$ columns of $\mathbf{Q}_2$ are an orthogonal basis of the null space of $\mathbf{C}$. A clever simplification: any known phase differences are folded into $\theta_j$, so $\mathbf{d}\equiv 0$ and the trivial particular solution $\bar{\boldsymbol\alpha} = 0$ is always available. The null-space basis is obtained from the QR factorisation of $\mathbf{C}^{\top}$:

$$\mathbf{C}^{\top} = \mathbf{Q}\mathbf{R}, \qquad \mathbf{Q} = [\mathbf{Q}_1\ \mathbf{Q}_2], \quad (20)$$

where $\mathbf{Q}_2$ is the last $N'$ columns of $\mathbf{Q}$. The constrained normal equations in $\boldsymbol\alpha$ become *unconstrained* ones in $\boldsymbol\beta$ by left-multiplying by $\mathbf{Q}_2^{\top}$ and substituting $\boldsymbol\alpha = \mathbf{Q}_2\boldsymbol\beta$:

$$\mathbf{Q}_2^{\top}\mathbf{A}^{\rm MFBD}\mathbf{Q}_2\cdot\delta\boldsymbol\beta - \mathbf{Q}_2^{\top}\mathbf{b}^{\rm MFBD} \simeq 0. \quad (21)$$

Once $\boldsymbol\beta$ is found, $\boldsymbol\alpha = \mathbf{Q}_2\boldsymbol\beta$ recovers the full coefficient vector. Löfdahl introduces a two-index labelling $j = k + (t-1)K$ with $k\in\{1,\ldots,K\}$ (simultaneous diversity channels) and $t\in\{1,\ldots,T\}$ (time/atmospheric realisations). Special cases: $K = 1$ gives MFBD; $T = 1$ gives PD; $K = T = 1$ gives vanilla BD.

**Korean.** 제약조건은 식 (17) 형태다. $\mathbf{C} \in \mathbb{R}^{N_C\times N}$, $N_C < N$. 모든 해는 아핀 부분공간(식 18)을 이루며, $\bar{\boldsymbol\alpha}$는 특수해, $\mathbf{Q}_2$의 $N' = N - N_C$개 열은 $\mathbf{C}$의 영공간의 직교 기저다. 기발한 단순화: 알려진 위상 차이는 $\theta_j$에 흡수되므로 $\mathbf{d}\equiv 0$이고, 자명한 특수해 $\bar{\boldsymbol\alpha} = 0$을 항상 사용할 수 있다. 영공간 기저는 $\mathbf{C}^{\top}$의 QR 분해(식 20)에서 얻으며, $\mathbf{Q}_2$는 $\mathbf{Q}$의 마지막 $N'$개 열이다. $\boldsymbol\alpha$에 대한 제약 정규방정식은 $\mathbf{Q}_2^{\top}$를 좌측 곱하고 $\boldsymbol\alpha = \mathbf{Q}_2\boldsymbol\beta$를 대입하면 $\boldsymbol\beta$에 대한 *무제약* 방정식이 된다(식 21). $\boldsymbol\beta$를 찾으면 $\boldsymbol\alpha = \mathbf{Q}_2\boldsymbol\beta$로 전체 계수 벡터가 복원된다. Löfdahl은 이중 지수 표지 $j = k + (t-1)K$를 도입하며, $k\in\{1,\ldots,K\}$는 동시 다이버시티 채널, $t\in\{1,\ldots,T\}$는 시간/대기 실현이다. 특수 경우: $K = 1$은 MFBD, $T = 1$은 PD, $K = T = 1$은 순수 BD.

### Part VI: Worked cases — PD, PDS, SH (Secs. 3.2–3.5) / 응용 사례 — PD, PDS, SH

**English.**

**(a) PD with fully known phase differences (Sec. 3.2).** For a single-realisation PD data set ($T=1$, $K$ diversity channels), the constraint $\alpha_{1m} = \alpha_{km}$ for all $k > 1$ and all modes $m$ yields a block-form

$$\mathbf{C}^{\rm PD'} = \begin{bmatrix}\mathbf{I}_M & -\mathbf{I}_M & & \\ \vdots & & \ddots & \\ \mathbf{I}_M & & & -\mathbf{I}_M\end{bmatrix}, \qquad \mathbf{Q}_2 = \frac{1}{\sqrt{K}}\begin{bmatrix}\mathbf{I}_M \\ \vdots \\ \mathbf{I}_M\end{bmatrix}. \quad (23)$$

This null-space basis is *extremely sparse* — the $K$ channels are identical copies of $\boldsymbol\beta/\sqrt{K}$. Löfdahl notes that $\mathbf{Q}_2^{\top}\mathbf{b}^{\rm MFBD} = \mathbf{b}^{\rm PD}$ and $\mathbf{Q}_2^{\top}\mathbf{A}^{\rm MFBD}\mathbf{Q}_2 = \mathbf{A}^{\rm PD}$ (up to an inconsequential constant in the regulariser), proving the LEC machinery correctly reduces to classical PD.

**(b) PD with partly unknown differences (Sec. 3.3).** Typically tilt modes (image-registration differences between channels) are *not* known. Remove those constraints from Eq. (22) and add instead $\sum_k \alpha_{km} = 0$ for $m\in\{\text{tilts}\}$ so the common tilt cannot run away. The resulting $\mathbf{Q}_2$ is less sparse and less "pretty", but it still works (Fig. 2).

**(c) PDS — full case (Sec. 3.4).** $J = KT$, but the true unknown count is $TM$ (plus 2$(K{-}1)$ per-channel registrations), because the same object and the same $K$-channel relative structure recurs across $T$ seeing realisations. The constraint matrix (Eq. 25) has three kinds of rows: (i) tilts sum to zero *globally*, (ii) the *inter-channel* tilt difference is fixed across all $t$, (iii) non-tilt modes are identical across $k$ within each $t$.

**(d) Shack–Hartmann (Sec. 3.5).** Revelation: SH is just MFBD where the different channels have different **amplitude masks** $A_j$ rather than different phases — each sub-image samples a different part of the pupil. With the same phase $\phi$ underlying all sub-images, the constraint matrix is identical to $\mathbf{C}^{\rm PD'}$ for $J=$ number of microlenses. Local tilts (registration of sub-images) can be either calibrated into $\theta_j$ or estimated as part of the phase expansion. This unification allows SH and a conventional high-resolution image to be processed jointly, which conventional SH analysis cannot do.

**Korean.**

**(a) 완전히 알려진 위상 차이의 PD (3.2절).** 단일 실현 PD 데이터($T=1$, $K$개 다이버시티 채널)에서 제약 $\alpha_{1m} = \alpha_{km}$(모든 $k>1$, 모든 $m$)은 블록 형태(식 23)를 산출한다. 이 영공간 기저는 *극도로 성기며* — $K$개 채널이 $\boldsymbol\beta/\sqrt{K}$의 동일 복사본이다. Löfdahl은 $\mathbf{Q}_2^{\top}\mathbf{b}^{\rm MFBD} = \mathbf{b}^{\rm PD}$와 $\mathbf{Q}_2^{\top}\mathbf{A}^{\rm MFBD}\mathbf{Q}_2 = \mathbf{A}^{\rm PD}$(정규화 항의 사소한 상수 차이 제외)를 확인해 LEC 기계가 고전 PD로 정확히 환원됨을 증명한다.

**(b) 부분적으로 알려지지 않은 차이의 PD (3.3절).** 보통 틸트 모드(채널 간 영상 정합 차이)는 알려지지 않는다. 식 (22)에서 해당 제약을 제거하고 대신 $m\in\{\text{tilts}\}$에 대해 $\sum_k \alpha_{km} = 0$을 추가해 공통 틸트가 발산하지 않게 한다. 결과의 $\mathbf{Q}_2$는 덜 성기고 덜 "예쁘지만" 여전히 작동한다(그림 2).

**(c) PDS 완전형 (3.4절).** $J = KT$지만, 진짜 미지수 개수는 $TM$(+ 채널당 $2(K{-}1)$개 정합)이다 — $T$개 시잉 실현 전체에서 같은 물체와 같은 $K$ 채널 상대 구조가 반복되기 때문. 제약 행렬(식 25)은 세 종류의 행: (i) 틸트의 *전역* 합이 영, (ii) *채널 간* 틸트 차이가 모든 $t$에서 고정, (iii) 비-틸트 모드가 각 $t$ 내에서 $k$에 걸쳐 동일.

**(d) Shack–Hartmann (3.5절).** 계시: SH는 채널마다 위상이 아니라 **진폭 마스크** $A_j$가 다른 MFBD일 뿐이다 — 각 부-영상은 동공의 다른 부분을 표본한다. 모든 부-영상의 바닥에 같은 위상 $\phi$가 있으므로 제약 행렬은 $J$=마이크로렌즈 수의 $\mathbf{C}^{\rm PD'}$과 동일하다. 국소 틸트(부-영상 정합)는 $\theta_j$로 보정하거나 위상 전개의 일부로 추정할 수 있다. 이 통합은 SH와 전통 고해상도 영상의 공동 처리를 가능케 하며, 이는 재래식 SH 분석이 할 수 없는 일이다.

### Part VII: Extensions (Sec. 4) / 확장

**English.** Löfdahl outlines four relaxations, each adding new applications at the cost of only a few more constraint rows:

1. **Different $M$ per frame (Sec. 4.1).** Useful when an SH sub-image needs local tilts in its expansion while the accompanying high-resolution image does not. Just use a single index $n = m + \sum_{j'<j} M_{j'}$.

2. **Different objects per scene (Sec. 4.2).** Introduce a set index $s\in\{1,\ldots,S\}$ with separate objects $F_s$; the metric becomes $L = \sum_s L_s$. This is the *multi-object* generalisation — crucial for solar magnetograms where two Zeeman components produce nearly-identical images with a faint polarisation signal, and jointly registered/deconvolved frames beat cross-correlation alignment. It is also essential when combining broad-band and narrow-band filter channels on the same telescope.

3. **Different wavelengths (Sec. 4.3).** Wavefronts scaled by $1/\lambda$. With PDS in several wavelengths, add the LEC $\lambda_1(\alpha_{111m} - \alpha_{1t1m}) = \lambda_s(\alpha_{s11m} - \alpha_{st1m})$ to enforce a wavelength-independent aberration profile — valid when the optical train is quasi-achromatic.

4. **Different $K_s$ per wavelength (Sec. 4.4).** Enables "PDS at $\lambda_1$ + MFBD at $\lambda_2$" scenarios (Paxman & Seldin 1999; Tritschler & Schmidt 2002). Wavelength-dependent effective PSFs are estimated jointly so that relative camera differences need not be pre-calibrated.

**Korean.** Löfdahl은 네 가지 완화를 설명하며, 각각 제약 행 몇 개만 추가하면 새로운 응용을 연다:

1. **프레임별 다른 $M$ (4.1절).** SH 부-영상은 전개에 국소 틸트가 필요하지만 동반 고해상도 영상은 아닐 때 유용. 단일 지수 $n = m + \sum_{j'<j} M_{j'}$만 쓰면 된다.

2. **장면별 다른 물체 (4.2절).** 집합 지수 $s\in\{1,\ldots,S\}$와 별도 물체 $F_s$를 도입; 척도는 $L = \sum_s L_s$가 된다. 이것이 *다물체* 일반화 — 두 Zeeman 성분이 거의 동일한 영상에 희미한 편광 신호를 더하는 태양 자기도에서 결정적이며, 공동 정합/디컨볼루션된 프레임이 상호 상관 정렬을 이긴다. 같은 망원경에서 광대역과 협대역 필터 채널을 결합할 때도 필수.

3. **다파장 (4.3절).** 파면은 $1/\lambda$로 스케일된다. 여러 파장의 PDS에서 LEC $\lambda_1(\alpha_{111m} - \alpha_{1t1m}) = \lambda_s(\alpha_{s11m} - \alpha_{st1m})$를 추가해 파장 독립 수차 프로파일을 강제 — 광학계가 준무색 수차(achromatic)일 때 유효.

4. **파장별 다른 $K_s$ (4.4절).** "$\lambda_1$에서 PDS + $\lambda_2$에서 MFBD" 시나리오 지원(Paxman & Seldin 1999; Tritschler & Schmidt 2002). 파장 의존 유효 PSF가 공동으로 추정되어 카메라 상대 차이를 사전 보정할 필요가 없다.

### Part VIII: Practical Details — Per-set diagnostics (Sec. 3.4 tail) / 실용: 세트별 진단

**English.** Not every PDS snapshot is good (one can land on a moment of strong seeing). Löfdahl suggests computing a *per-set* metric (Eq. 26):

$$L_t = \sum_u\left[\sum_k|D_{tk}|^2 - \frac{|\sum_k D_{tk}^{*}S_{tk}|^2}{Q_t}\right],$$

skipping the wavefront regulariser and summing only over $k$ at fixed $t$. Large $L_t$ flags a seeing realisation whose wavefront has been inverted poorly — a good diagnostic for discarding bad frames or down-weighting them.

**Korean.** 모든 PDS 스냅샷이 좋은 것은 아니다(강한 시잉 순간에 걸릴 수 있다). Löfdahl은 *세트별* 척도 계산을 제안한다(식 26): 파면 정규화 항을 생략하고 고정 $t$의 $k$에 대해서만 합산. 큰 $L_t$는 파면이 잘못 역변환된 시잉 실현을 표시 — 나쁜 프레임을 버리거나 가중치를 낮추는 데 좋은 진단.

### Part IX: Discussion (Sec. 5) / 토론

**English.** The formulation has been validated on PD, PDS, and MFBD data from the Swedish Vacuum Solar Telescope and on simulations. Löfdahl flags one real challenge: *the null-space matrix is not unique*. Sparse, block-regular $\mathbf{Q}_2$ (as in Fig. 1) are much faster and easier to interpret than the irregular ones produced by his QR code when unknown-difference tilt constraints are mixed in (Fig. 2). Developing algorithms that return a maximally sparse null-space basis is marked as a priority for future work — a problem of general interest in constrained linear algebra.

**Korean.** 수식화는 스웨덴 진공 태양 망원경(SVST)의 PD, PDS, MFBD 데이터와 시뮬레이션에서 검증되었다. Löfdahl은 실제적 도전 하나를 지적한다: *영공간 행렬은 유일하지 않다*. 성기고 블록 규칙적인 $\mathbf{Q}_2$(그림 1)는 그의 QR 코드가 알려지지 않은 차이의 틸트 제약을 섞었을 때 산출하는 불규칙한 것(그림 2)보다 훨씬 빠르고 해석하기 쉽다. 최대한 성긴 영공간 기저를 반환하는 알고리즘의 개발은 향후 연구의 우선순위로 표시되며 — 제약 선형대수 일반에서도 관심 있는 문제다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Unification is the message** — **통합이 메시지다.**
   **English.** The paper's real product is not a new algorithm but a new *viewpoint*: PD, PDS, MFBD, SH WFS, and their hybrids are all the same MFBD optimization subject to different linear-equality constraint matrices $\mathbf{C}$. One code base, driven by data (the $\mathbf{C}$ matrix), replaces a family of custom solvers.
   **Korean.** 이 논문의 진짜 산물은 새 알고리즘이 아니라 새 *관점*이다: PD, PDS, MFBD, SH WFS와 그 혼합은 모두 서로 다른 선형 등식 제약 행렬 $\mathbf{C}$에 따른 같은 MFBD 최적화다. 데이터($\mathbf{C}$ 행렬)가 구동하는 하나의 코드 기반이 전용 솔버 군을 대체한다.

2. **Object elimination via Wiener filter is the workhorse** — **Wiener 필터를 통한 물체 소거가 핵심.**
   **English.** Closed-form substitution $F = \sum_j S_j^{*}D_j / Q$ removes the high-dimensional unknown object from the metric, leaving only $\approx JM$ wavefront coefficients to estimate. This scales far better than joint $(F, \alpha)$ optimization.
   **Korean.** 폐쇄형 대입 $F = \sum_j S_j^{*}D_j / Q$은 고차원 미지 물체를 척도에서 제거하고, $\approx JM$개의 파면 계수만 남긴다. 이는 $(F, \alpha)$ 공동 최적화보다 훨씬 잘 스케일된다.

3. **Block-diagonal Hessian in MFBD is a big computational win** — **MFBD의 블록 대각 헤시안은 큰 계산적 이득.**
   **English.** Channel independence yields $J$ decoupled $M\times M$ systems instead of one $JM\times JM$ system; cost drops from $(JM)^3$ to $J\cdot M^3$. For $J = 100$, $M = 30$ this is a factor-$10^4$ saving per Newton step.
   **Korean.** 채널 독립성은 단일 $JM\times JM$ 시스템 대신 $J$개의 분리된 $M\times M$ 시스템을 낳으며, 비용은 $(JM)^3$에서 $J\cdot M^3$으로 감소. $J = 100$, $M = 30$에서 뉴턴 단계당 $10^4$배의 절약.

4. **Null-space parameterization is the right way to handle equality constraints** — **영공간 모수화는 등식 제약을 다루는 올바른 방법.**
   **English.** Substituting $\boldsymbol\alpha = \mathbf{Q}_2\boldsymbol\beta$ converts a constrained optimization into an unconstrained one with fewer variables. QR/SVD factorization of $\mathbf{C}^{\top}$ builds $\mathbf{Q}_2$ once; subsequent Newton steps just multiply by $\mathbf{Q}_2$ — cheap when $\mathbf{Q}_2$ is sparse.
   **Korean.** $\boldsymbol\alpha = \mathbf{Q}_2\boldsymbol\beta$ 대입은 제약 최적화를 변수 수가 적은 무제약 최적화로 변환. $\mathbf{C}^{\top}$의 QR/SVD 분해로 $\mathbf{Q}_2$를 한 번 만들면 이후 뉴턴 단계는 $\mathbf{Q}_2$ 곱셈만 수행 — $\mathbf{Q}_2$가 성기면 저렴.

5. **KL modes are the "right" basis for atmospheric phase** — **KL 모드가 대기 위상에 "올바른" 기저.**
   **English.** Karhunen–Loève modes diagonalise the Kolmogorov covariance, so the regulariser $\sum_m \alpha_m^2/\lambda_m$ becomes a diagonal quadratic form. About 15–40 KL modes typically suffice for solar imaging, a large reduction from Zernike truncation.
   **Korean.** Karhunen–Loève 모드는 Kolmogorov 공분산을 대각화하므로, 정규화 항 $\sum_m \alpha_m^2/\lambda_m$이 대각 이차 형식이 된다. 태양 관측에서는 보통 15–40개의 KL 모드로 충분하며, 이는 Zernike 절단보다 크게 감소한 수.

6. **SH is a different-amplitude, same-phase MFBD** — **SH는 같은 위상, 다른 진폭 MFBD.**
   **English.** Treating Shack–Hartmann sub-images as MFBD channels with different $A_j$ but a shared $\phi$ exploits *blurring* information inside each sub-aperture that the conventional local-tilt estimate throws away. This was demonstrated in simulation with two microlenses (Löfdahl, Duncan & Scharmer 1998).
   **Korean.** Shack–Hartmann 부-영상을 다른 $A_j$와 공유 $\phi$를 가진 MFBD 채널로 다루면, 재래식 국소 틸트 추정이 버리는 각 부-동공 내 *블러링* 정보를 활용한다. 이는 마이크로렌즈 두 개의 시뮬레이션에서 입증되었다(Löfdahl, Duncan & Scharmer 1998).

7. **Per-set diagnostics $L_t$ identify bad seeing realisations** — **세트별 진단 $L_t$가 나쁜 시잉 실현을 식별.**
   **English.** Computing the metric restricted to one atmospheric realisation separates the good moments from the bad ones post-hoc, enabling adaptive weighting or frame selection.
   **Korean.** 한 대기 실현으로 제한된 척도 계산은 사후에 좋은 순간과 나쁜 순간을 분리해 적응적 가중치나 프레임 선택을 가능케 한다.

8. **The extensions anticipate all of modern solar post-processing** — **확장은 현대 태양 후처리의 모든 것을 예고.**
   **English.** Multi-object, multi-wavelength, variable-$K$, simultaneous magnetograms, and joint SH+broadband processing — every strategy used today at SST/CRISP/CHROMIS and being prepared for DKIST was sketched in Sec. 4 as "add a row to $\mathbf{C}$". This modularity is why MOMFBD (2005) could be built on this foundation.
   **Korean.** 다물체, 다파장, 가변 $K$, 동시 자기도, 공동 SH+광대역 처리 — 오늘날 SST/CRISP/CHROMIS에서 사용되고 DKIST를 위해 준비되는 모든 전략이 4절에서 "$\mathbf{C}$에 행을 추가"로 스케치되어 있다. 이 모듈성이 MOMFBD(2005)가 이 기반 위에 세워질 수 있었던 이유.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Forward Model / 전방 모델

**Pupil and PSF / 동공과 PSF:**

$$
P_j(\mathbf{r}) = A_j(\mathbf{r})\,\exp\{i\phi_j(\mathbf{r})\}, \qquad s_j(\mathbf{x}) = \left|\mathcal{F}^{-1}\{P_j\}(\mathbf{x})\right|^2.
$$

- $\mathbf{r}$: pupil-plane coordinate / 동공면 좌표
- $\mathbf{x}$: image-plane coordinate / 영상면 좌표
- $A_j$: binary aperture mask (1 inside pupil, 0 outside) / 이진 조리개 마스크
- $\phi_j$: wavefront phase in radians / 라디안 단위 파면 위상
- OTF: $S_j(\mathbf{u}) = \mathcal{F}\{s_j\}(\mathbf{u})$ / 광학 전달 함수

**Image equation (Fourier domain) / 영상 방정식 (푸리에 영역):**

$$
D_j(\mathbf{u}) = F(\mathbf{u})\cdot S_j(\mathbf{u}) + N_j(\mathbf{u}),
$$

with $N_j$ zero-mean complex Gaussian white noise. Switch to summation over $\mathbf{u}$ for the finite-pixel realisation.

### 4.2 Wavefront Expansion / 파면 전개

$$
\phi_j(\mathbf{r}) = \theta_j(\mathbf{r}) + \sum_{m=1}^{M}\alpha_{jm}\,\psi_m(\mathbf{r}),
$$

where $\{\psi_m\}$ are Karhunen–Loève (KL) modes for Kolmogorov statistics, with variances

$$
\lambda_m = \mathbb{E}[|\alpha_{jm}|^2] \propto m^{-\mu}, \quad \mu \approx 11/6 \text{ for high modes}.
$$

### 4.3 Wiener Object Estimate / Wiener 물체 추정

Given OTFs $S_j$, the MMSE object estimator under Gaussian noise is

$$
\boxed{\; F = \frac{1}{Q}\sum_{j=1}^{J} S_j^{*}\,D_j, \qquad Q = \gamma_{\rm obj} + \sum_{j=1}^{J}|S_j|^2 \;}.
$$

- $\gamma_{\rm obj}$: object-side regulariser, chosen $\sim 1/\mathrm{SNR}$. / 물체 측 정규화 항, $\sim 1/\mathrm{SNR}$로 선택.

### 4.4 Löfdahl Metric / Löfdahl 척도

Substituting the Wiener estimate back gives

$$
\boxed{\; L(\boldsymbol\alpha) = \sum_{\mathbf{u}}\left[\sum_{j}|D_j|^2 - \frac{\left|\sum_j D_j^{*}S_j\right|^2}{Q}\right] + \frac{\gamma_{\rm wf}}{2}\sum_{m=1}^{M}\frac{1}{\lambda_m}\sum_{j=1}^{J}|\alpha_{jm}|^2 \;}.
$$

The first square bracket is the *data-misfit* term (real-valued, $\geq 0$); the second is the KL-weighted Tikhonov prior on the wavefront.

### 4.5 Gradient and Hessian / 경사도와 헤시안

Let $p_j = \mathcal{F}^{-1}\{P_j\}$. The PD gradient (Eq. 8) has scalar components

$$
b_m^{\rm PD} = \left\langle -2\sum_j \mathrm{Im}\bigl[P_j^{*}\mathcal{F}\{p_j\mathrm{Re}[\mathcal{F}^{-1}\{F^{*}D_j - |F|^2 S_j\}]\}\bigr],\ \psi_m\right\rangle + \gamma_{\rm wf}\frac{\alpha_m}{\lambda_m}.
$$

For MFBD, the coefficient index splits $m \to (j, m)$ and the gradient picks up a $\sqrt{J}$:

$$
b_{jm}^{\rm MFBD} = \sqrt{J}\cdot\bigl(\text{PD-like term with only } j\text{-th channel}\bigr) + \gamma_{\rm wf}\frac{\alpha_{jm}}{\lambda_m}.
$$

The Hessian becomes block-diagonal with $J$ blocks of size $M\times M$:

$$
\mathbf{A}^{\rm MFBD} = \mathrm{blkdiag}(\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_J), \qquad \mathbf{A}_j\in\mathbb{R}^{M\times M}.
$$

### 4.6 Constrained Optimization via Null Space / 영공간을 통한 제약 최적화

Given constraints $\mathbf{C}\boldsymbol\alpha = \mathbf{d}$ with $\mathbf{C}\in\mathbb{R}^{N_C\times N}$ and $N = JM$, $N' = N - N_C$:

$$
\mathbf{C}^{\top} = \mathbf{Q}\mathbf{R},\quad \mathbf{Q} = [\mathbf{Q}_1\ \mathbf{Q}_2],\quad \mathbf{Q}_2\in\mathbb{R}^{N\times N'}.
$$

$\mathbf{Q}_2$ is an orthonormal basis of $\ker\mathbf{C}$: $\mathbf{C}\mathbf{Q}_2 = 0$ and $\mathbf{Q}_2^{\top}\mathbf{Q}_2 = \mathbf{I}_{N'}$.

Parameterise all feasible $\boldsymbol\alpha$ as

$$
\boldsymbol\alpha = \bar{\boldsymbol\alpha} + \mathbf{Q}_2\boldsymbol\beta,\qquad \boldsymbol\beta\in\mathbb{R}^{N'}.
$$

With $\mathbf{d}=0$, pick $\bar{\boldsymbol\alpha}=0$. Substituting into the (linearised) MFBD normal equations and left-multiplying by $\mathbf{Q}_2^{\top}$ gives the **reduced Newton system**

$$
\boxed{\; \mathbf{Q}_2^{\top}\mathbf{A}^{\rm MFBD}\mathbf{Q}_2\cdot\delta\boldsymbol\beta = \mathbf{Q}_2^{\top}\mathbf{b}^{\rm MFBD}.\;}
$$

Solve for $\delta\boldsymbol\beta$, recover $\delta\boldsymbol\alpha = \mathbf{Q}_2\delta\boldsymbol\beta$, update, iterate to convergence.

### 4.7 Example Constraint Matrices / 제약 행렬 예

**PD (fully known differences), $K$ channels, $M$ modes:**

$$
\mathbf{C}^{\rm PD'} = \begin{bmatrix}\mathbf{I}_M & -\mathbf{I}_M & & \\ & \mathbf{I}_M & -\mathbf{I}_M & \\ & & \ddots & \ddots\\ & & & \mathbf{I}_M & -\mathbf{I}_M\end{bmatrix}\in\mathbb{R}^{(K-1)M\times KM}, \qquad \mathbf{Q}_2 = \frac{1}{\sqrt{K}}\begin{bmatrix}\mathbf{I}_M\\ \vdots\\ \mathbf{I}_M\end{bmatrix}.
$$

Only $M$ free parameters; the $K$-fold redundancy is collapsed.

**PDS with unknown tilts:** Union of PD rows (per $t$) and tilt constraints that enforce a *constant inter-channel tilt* plus *zero global tilt sum*. Total constraints $N_C^{\rm PDS}$ grows linearly with $T$.

### 4.8 Numerical Scales / 수치 규모

Representative figures for a modern SST configuration:
- Field of view: $\sim 50''\times 50''$ at 0.041''/pix → $\sim 1220\times 1220$ pixels.
- $M = 15$–40 KL modes (most solar papers use 30–36). / 대부분의 태양 논문은 30–36개 사용.
- $K = 2$ (one in-focus + one defocused) for PD; $T = 20$–100 burst frames for PDS. / PD에서 $K=2$; PDS에서 $T = 20$–100.
- Defocus magnitude: $\sim 1$ wave peak-to-valley ($\theta_j = \pi$ rad for the defocus channel).
- Regulariser $\gamma_{\rm wf}\sim 10^{-3}$–$10^{-2}$, $\gamma_{\rm obj}\sim 10^{-3}$ (data-dependent).
- Strehl ratio expected after restoration: $0.3$–$0.7$ on good seeing, raw $\sim 0.05$–$0.15$ through $r_0 = 10$–$15$ cm atmosphere.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
Ground-based high-resolution solar imaging — a timeline / 지상 고해상도 태양 관측 타임라인
============================================================================
1917  Hartmann — original Hartmann screen test for telescope figures
1970  Labeyrie — speckle interferometry: diffraction info survives seeing
1971  Knox–Thompson — phase closure from pairs of short-exposure spectra
1976  Fried — Kolmogorov r_0; coherence length of atmospheric phase
1977  Weigelt — speckle masking; bispectrum phase recovery
1982  Gonsalves — phase diversity proposed for AO (focused + defocused pair)
1990  Roddier N. — Zernike/KL simulation of Kolmogorov turbulence   ← Ref. 13
1990  Shack–Hartmann WFS deployed in astronomical AO
1992  Paxman–Schulz–Fienup — joint object/aberration ML; PDS theory  ← Refs. 5, 9
1993  Schulz — MFBD framework for astronomy                          ← Ref. 1
1994  Löfdahl & Scharmer — PD adapted to solar photosphere            ← Ref. 6
1996  Paxman et al. — PDS evaluated on solar images                  ← Ref. 7
1997  Schulz, Stribling, Miller — MFBD on Hubble Space Telescope     ← Ref. 2
1998  Vogel, Chan, Plemmons — fast PD with regularisation            ← Ref. 11
1998  Tyler et al. — comparison: PDS > MFBD > single-frame BD       ← Ref. 4
1998  Löfdahl, Duncan, Scharmer — PD for DM control                 ← Ref. 17
2000  Scharmer, Shand, Löfdahl et al. — workstation solar AO         ← Ref. 16
2000  Löfdahl, Scharmer, Wei — DM calibration via PD                 ← Ref. 10
2001  Löfdahl, Berger, Seldin — multi-wavelength PDS, hours-long     ← Ref. 22
2002  *** THIS PAPER — Löfdahl, MFBD-LEC unified formulation ***
2002  SST (1m, La Palma) first light — 0.1" diffraction limit
2002  Tritschler & Schmidt — PD for sunspot photometry              ← Ref. 24
2005  van Noort, Rouppe van der Voort, Löfdahl — MOMFBD package (full impl.)
2008  CRISP spectropolarimeter at SST — routine MOMFBD post-processing
2012  GREGOR 1.5m first light — MOMFBD adopted
2013  ROSA at Dunn Solar Telescope — speckle + MOMFBD pipelines
2014  Hinode/SOT — 50 cm space telescope; PD-based calibration
2019  DKIST first light (4m) — largest solar telescope; MOMFBD-class needed
2022+ SUNRISE III balloon mission — MFBD restoration standard
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Gonsalves (1982) "Phase retrieval and diversity in adaptive optics" | Original PD paper — this paper generalises it to the multi-frame + constraint setting / 원래의 PD 논문 — 이 논문은 이를 다중프레임 + 제약 설정으로 일반화 | Foundational; PD is a special case (T=1) of MFBD–LEC / 기초; PD는 MFBD–LEC의 특수 경우(T=1) |
| Paxman, Schulz & Fienup (1992) "Joint estimation of object and aberrations by using phase diversity" | Introduces the ML metric with object-eliminated form / 물체가 제거된 ML 척도 도입 | Direct ancestor of Eq. (6); Löfdahl adopts its regularised version / 식 (6)의 직접 조상 |
| Schulz (1993) "Multi-frame blind deconvolution of astronomical images" | Coined MFBD for astronomy / 천문학에서 MFBD 용어 도입 | The MFBD half of MFBD–LEC / MFBD–LEC의 MFBD 절반 |
| Löfdahl & Scharmer (1994) "Wavefront sensing and image restoration from focused and defocused solar images" | First PD application to solar photosphere / 태양 광구에 PD 최초 적용 | Previous paper by same author; establishes Gaussian-noise assumption for low-contrast granulation / 같은 저자의 이전 논문; 저대비 입상에 대한 Gaussian 잡음 가정 정립 |
| Vogel, Chan, Plemmons (1998) "Fast algorithms for phase diversity-based blind deconvolution" | Regularised metric with $\gamma_{\rm obj}, \gamma_{\rm wf}$ / 정규화 척도 | Source of Löfdahl's two-regulariser form (Eq. 6) / Löfdahl의 두-정규화 형식 출처 |
| Roddier (1990) "Atmospheric wavefront simulation using Zernike polynomials" | KL modes and variances $\lambda_m$ / KL 모드와 분산 | Justifies $\gamma_{\rm wf}/\lambda_m$ weighting / $\gamma_{\rm wf}/\lambda_m$ 가중치 정당화 |
| Paxman, Seldin & Löfdahl (1996) "Evaluation of phase-diversity techniques for solar-image restoration" | PDS evaluation on solar data / 태양 데이터에서 PDS 평가 | Defines the regime in which MFBD–LEC will operate / MFBD–LEC가 작동할 체제 정의 |
| Tyler et al. (1998) "Comparison of image reconstruction algorithms using adaptive optics" | Benchmarks PDS > MFBD > single-frame / PDS > MFBD > 단일-프레임 벤치마크 | Motivates why adding constraints (PDS) beats plain MFBD / 제약 추가(PDS)가 일반 MFBD를 이기는 이유를 동기화 |
| van Noort, Rouppe van der Voort, Löfdahl (2005) "Solar image restoration by use of multi-frame blind de-convolution with multiple objects and phase diversity" (MOMFBD) | Full software implementation of this paper's theory / 이 논문 이론의 완전한 소프트웨어 구현 | Direct successor; MOMFBD is still the standard SST pipeline / 직계 후속; MOMFBD는 여전히 표준 SST 파이프라인 |
| Löfdahl, Berger & Seldin (2001) "Two dual-wavelength sequences…" | Multi-wavelength solar PDS / 다파장 태양 PDS | Worked example of Sec. 4.3's extension / 4.3절 확장의 작업된 사례 |
| Tritschler & Schmidt (2002) "Sunspot photometry with phase diversity" | PD applied to sunspots / 흑점에 PD 적용 | Example of mixed-$K_s$ extension (Sec. 4.4) / 혼합 $K_s$ 확장(4.4절)의 예 |

---

## 7. References / 참고문헌

- Löfdahl, M. G. (2002). "Multi-frame blind deconvolution with linear equality constraints." *Proc. SPIE* **4792-21**. DOI: 10.1117/12.451791. arXiv: physics/0209004.
- Schulz, T. J. (1993). "Multi-frame blind deconvolution of astronomical images." *JOSA A* **10**, 1064–1073.
- Paxman, R. G., Schulz, T. J., Fienup, J. R. (1992). "Joint estimation of object and aberrations by using phase diversity." *JOSA A* **9**, 1072–1085.
- Gonsalves, R. A. (1982). "Phase retrieval and diversity in adaptive optics." *Optical Engineering* **21**, 829–832.
- Löfdahl, M. G., Scharmer, G. B. (1994). "Wavefront sensing and image restoration from focused and defocused solar images." *A&A Suppl. Ser.* **107**, 243–264.
- Paxman, R. G., Seldin, J. H., Löfdahl, M. G., Scharmer, G. B., Keller, C. U. (1996). "Evaluation of phase-diversity techniques for solar-image restoration." *ApJ* **466**, 1087–1099.
- Vogel, C. R., Chan, T. F., Plemmons, R. J. (1998). "Fast algorithms for phase diversity-based blind deconvolution." *Proc. SPIE* **3353**.
- Tyler, D. W., Ford, S. D., Hunt, B. R., et al. (1998). "Comparison of image reconstruction algorithms using adaptive optics instrumentation." *Proc. SPIE* **3353**, 160–171.
- Roddier, N. (1990). "Atmospheric wavefront simulation using Zernike polynomials." *Optical Engineering* **29**, 1174–1180.
- Löfdahl, M. G., Berger, T. E., Seldin, J. H. (2001). "Two dual-wavelength sequences of high-resolution solar photospheric images…" *A&A* **377**, 1128–1135.
- Löfdahl, M. G., Scharmer, G. B., Wei, W. (2000). "Calibration of a deformable mirror and Strehl ratio measurements by use of phase diversity." *Applied Optics* **39**, 94–103.
- Tritschler, A., Schmidt, W. (2002). "Sunspot photometry with phase diversity. I." *A&A* **382**, 1093–1105.
- Kahaner, D., Moler, C., Nash, S. (1989). *Numerical Methods and Software*, Prentice Hall.
- Engl, H. W., Hanke, M., Neubauer, A. (1996). *Regularization of Inverse Problems*, Kluwer.
- Löfdahl, M. G., Duncan, A. L., Scharmer, G. B. (1998). "Fast phase diversity wavefront sensing for mirror control." *Proc. SPIE* **3353**, 952–963.
- Scharmer, G. B., Shand, M., Löfdahl, M. G., Dettori, P. M., Wei, W. (2000). "A workstation based solar/stellar adaptive optics system." *Proc. SPIE* **4007**, 239–250.
- van Noort, M., Rouppe van der Voort, L., Löfdahl, M. G. (2005). "Solar image restoration by use of multi-frame blind deconvolution with multiple objects and phase diversity." *Solar Physics* **228**, 191–215. (The MOMFBD paper)
- Rimmele, T. R., Marino, J. (2011). "Solar adaptive optics." *Living Reviews in Solar Physics* **8**, 2.
