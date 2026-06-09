---
title: "Robust Principal Component Analysis?"
authors: Emmanuel J. Candès, Xiaodong Li, Yi Ma, John Wright
year: 2011
journal: "Journal of the ACM, Vol. 58, No. 3, Article 11, pp. 1–37"
doi: "10.1145/1970392.1970395"
topic: Low_SNR_Imaging
tags: [robust-pca, principal-component-pursuit, low-rank, sparse, nuclear-norm, l1-minimization, ADMM, ALM, video-background, matrix-completion, convex-relaxation]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 37. Robust Principal Component Analysis? / 강건 주성분 분석은 가능한가?

---

## 1. Core Contribution / 핵심 기여

The paper proves that the decomposition of an $n_1 \times n_2$ matrix $M = L_0 + S_0$ into an unknown low-rank $L_0$ and an unknown sparse $S_0$ can be recovered **exactly** by solving the convex program **Principal Component Pursuit (PCP)**

$$
\min_{L,S}\;\|L\|_* + \lambda\|S\|_1 \quad\text{subject to}\quad L + S = M,
$$

with the universal, **tuning-free** parameter $\lambda = 1/\sqrt{n_{(1)}}$, $n_{(1)} = \max(n_1, n_2)$. Theorem 1.1 establishes recovery with probability $1 - c n^{-10}$ provided $L_0$ satisfies a standard incoherence condition on its singular vectors and $S_0$ has support drawn uniformly at random; the admissible rank is as large as $O(n/\log^2 n)$ and the admissible fraction of corrupted entries is a constant. Theorem 1.2 extends this to **simultaneously missing and corrupted** entries (robust matrix completion), $\lambda = 1/\sqrt{0.1 n_{(1)}}$. Section 4 demonstrates the result with random simulations across $n=500,\dots,3000$ — all recover exactly with relative error $< 10^{-5}$ and a near-constant number ($< 17$) of partial SVDs — and with two visually striking applications: airport / lobby video background subtraction (200–250 frames, $176\times 144$ resolution) and shadow / specularity removal from face images (Yale B database, 58 illuminations per subject). Section 5 supplies a practical augmented Lagrange multiplier (ALM) / ADMM solver (Algorithm 1) whose dominant cost is one partial SVD per iteration. The paper thereby establishes RPCA as a turn-key tool for **interpretable low-rank + sparse decomposition** of real-world data.

이 논문은 행렬 $M = L_0 + S_0$ — 알 수 없는 저랭크 부분과 알 수 없는 희소 손상의 합 — 의 정확한 분리가 단일 매개변수 볼록 계획 **Principal Component Pursuit (PCP)** 

$$\min_{L,S}\;\|L\|_* + \lambda\|S\|_1\quad\text{s.t.}\quad L+S=M$$

으로 가능함을 증명한다. 핵심은 매개변수 $\lambda = 1/\sqrt{n_{(1)}}$ 의 **튜닝-프리** 보편 선택 — Theorem 1.1 은 $L_0$ 의 특이벡터가 incoherence 조건을 만족하고 $S_0$ 의 support 가 균일 무작위면, 확률 $1-cn^{-10}$ 으로 정확 회복이 보장되며, 허용 가능한 rank 는 $O(n/\log^2 n)$, 허용 가능한 손상 비율은 일정 상수임을 말한다. Theorem 1.2 는 결측 + 손상이 동시에 있는 robust matrix completion 으로 확장한다. Section 4 의 무작위 실험 ($n=500$–$3000$, rank $0.05n$, 손상 $5\%$–$10\%$) 은 모든 경우에서 상대 오차 $<10^{-5}$ 로 정확 회복을 보이고, 비디오 배경 분리·얼굴 그림자 제거에서 시각적으로 인상적인 결과를 제시한다. Section 5 의 ALM/ADMM 알고리즘 (Algorithm 1) 은 반복당 하나의 부분 SVD 만으로 동작하는 실용적 해법이다. RPCA 는 이로써 **해석 가능한 저랭크 + 희소 분해** 의 표준 도구가 되었다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Motivation / 동기 (Section 1.1, pp. 1–3)

Classical PCA seeks the best rank-$k$ approximation of $M$ in the $\ell^2$ sense:

$$
\min_L\; \|M - L\|\quad\text{s.t.}\quad \mathrm{rank}(L) \le k.
$$

This is solved by SVD truncation and is optimal under i.i.d. Gaussian noise. **Brittle to gross outliers**, however — a single very large entry can pull the principal axes arbitrarily far. The paper considers an idealised version of *robust PCA*:

$$
M = L_0 + S_0,
$$

where $L_0$ is exactly low-rank and $S_0$ has **arbitrarily large** entries on an unknown sparse support. Crucially, the sparse magnitude is allowed to dwarf the low-rank component. Existing robust-PCA approaches (influence functions, multivariate trimming, alternating minimisation, random sampling) all fail to deliver a **scalable polynomial-time algorithm with broad performance guarantees**. The paper plugs that gap.

고전 PCA 는 $L_2$ 의미의 rank-$k$ 근사를 SVD 절단으로 얻고, i.i.d. Gaussian 잡음에는 최적이다. 그러나 **큰 outlier 한 개에도 결과가 망가진다**. 본 논문은 $M = L_0 + S_0$ — $L_0$ 는 정확히 저랭크, $S_0$ 는 위치/크기 모두 모르는 희소 손상 — 의 회복이라는 이상화된 robust PCA 를 다룬다. 기존 방법 (influence function, 다변량 trimming, alternating min, random sampling) 은 "다항시간 + 광범위 보장" 을 동시에 만족하지 못했고 이 논문이 그 공백을 메운다.

**Applications enumerated** (Section 1.1): video surveillance (stationary background as $L_0$, moving objects as $S_0$); face recognition (Lambertian-illumination subspace as $L_0$, shadows / specularities as $S_0$); latent semantic indexing (common-words structure $L_0$, distinguishing keywords $S_0$); ranking / collaborative filtering (true preferences $L_0$, deceptive entries $S_0$). All have a **low-rank + sparse** physical structure.

응용 예: 비디오 감시 (고정 배경 = $L_0$, 움직이는 물체 = $S_0$), 얼굴 인식 (Lambertian 조명 부분공간 = $L_0$, 그림자·specularity = $S_0$), latent semantic indexing, ranking / collaborative filtering. 모두 저랭크 + 희소 구조를 갖는다.

### Part II: A Surprising Message and the PCP Program / 놀라운 메시지와 PCP (Section 1.2–1.4, pp. 4–6)

The decomposition appears ill-posed: there are twice as many unknowns ($L$ and $S$) as observations ($M$). It is also genuinely non-identifiable in degenerate cases — e.g., $M = e_1 e_1^*$ is both rank-1 *and* sparse — so two structural hypotheses are needed:

- **Incoherence of $L_0$** (Eqs. 1.2–1.3): the singular vectors $U \in \mathbb R^{n_1 \times r}$, $V \in \mathbb R^{n_2 \times r}$ satisfy
  $$
  \max_i \|U^* e_i\|^2 \le \frac{\mu r}{n_1},\qquad \max_i \|V^* e_i\|^2 \le \frac{\mu r}{n_2},\qquad \|UV^*\|_\infty \le \sqrt{\frac{\mu r}{n_1 n_2}}.
  $$
  Singular vectors must be reasonably "spread out", not aligned with the canonical basis (otherwise $L_0$ is itself sparse).

- **Random support of $S_0$**: $\Omega = \mathrm{supp}(S_0)$ is uniformly distributed among all sets of cardinality $m$ (or, equivalently, follows Bernoulli($\rho$) per entry).

Under these assumptions, **Theorem 1.1**: PCP with $\lambda = 1/\sqrt{n}$ (square case) recovers $(L_0, S_0)$ exactly with probability $\ge 1 - c n^{-10}$ provided

$$
\mathrm{rank}(L_0) \le \rho_r \,n\,\mu^{-1}(\log n)^{-2},\qquad m \le \rho_s\, n^2,
$$

for positive numerical constants $\rho_r, \rho_s$. **Strikingly:** $\lambda$ is **universal** — no tuning, no knowledge of $r$ or $m$ required. The square / rectangular generalisation uses $\lambda = 1/\sqrt{n_{(1)}}$ with $n_{(1)} = \max(n_1, n_2)$.

이 분해는 일견 부적절(ill-posed) 해 보인다 — 미지수가 관측의 두 배다. 또 $M = e_1 e_1^*$ 처럼 rank-1 인 동시에 sparse 인 퇴화 사례도 있다. 따라서 두 가지 구조적 가정이 필수: (i) $L_0$ 의 특이벡터 incoherence (식 1.2–1.3) — 좌표축과 정렬되지 않을 것; (ii) $S_0$ 의 support 가 균일 무작위. 이 가정 하에 Theorem 1.1 은 $\lambda = 1/\sqrt{n}$ 의 **튜닝 프리** 단일 선택만으로 정확 회복을 확률 $\ge 1-cn^{-10}$ 로 보장한다.

The remarkable feature is the **non-adaptivity of $\lambda$**: no need to know rank or sparsity ratio. This is in contrast to Chandrasekaran-Sanghavi-Parrilo-Willsky (2011, Section 1.5), whose deterministic conditions require an a-priori unknown $\lambda$ — practitioners had been expected to sweep $\lambda$. The randomness assumption pays off.

**$\lambda$ 의 non-adaptivity 는 결정적**이다. rank 도 sparsity 비율도 몰라도 된다 — 결정적 조건의 Chandrasekaran et al. (2011) 과 결정적인 차이.

### Part III: Architecture of the Proof / 증명의 골격 (Section 2, pp. 9–13)

The proof certifies optimality of $(L_0, S_0)$ via subgradient (KKT) conditions. The subgradient of $\|L\|_* + \lambda \|S\|_1$ at $(L_0, S_0)$ consists of pairs

$$
(UV^* + W,\;\lambda\,\mathrm{sgn}(S_0) + F),\qquad U^* W = 0,\; W V = 0,\; \|W\| \le 1,\; \mathcal P_\Omega F = 0,\; \|F\|_\infty \le 1.
$$

To certify optimality, the paper constructs a **dual certificate** $\Lambda$ in the intersection of the two subgradient cones — i.e., $\Lambda = UV^* + W = \lambda(\mathrm{sgn}(S_0) + F)$. Two key sufficient conditions:

(a) **Bound on the off-support**: $\|\mathcal P_{T^\perp} \Lambda\| < 1$ (small enough to be a valid nuclear-norm subgradient).

(b) **Match on the support**: the part of $\Lambda$ on $\Omega^c$ must be small in $\ell^\infty$-norm.

These are established via the **golfing scheme** of Gross (2011, quantum-state tomography): construct $\Lambda$ as the limit of a Bernoulli random sampling sequence and bound the residuals at each step using Bernstein's inequality and Hoeffding's inequality. Section 3 contains the detailed lemmas; the architecture in Section 2 is adequate for first reading.

증명은 KKT 조건의 dual 인증으로 $(L_0,S_0)$ 의 최적성을 보인다. $\|L\|_* + \lambda\|S\|_1$ 의 subgradient 집합을 명시한 뒤, 두 충분조건 — (a) off-support 영역에서 $\|\mathcal P_{T^\perp}\Lambda\|<1$, (b) support 영역에서 $\Lambda$ 가 sgn/서명 항과 일치 — 을 만족하는 $\Lambda$ 를 **golfing scheme** (Gross 2011) 으로 구성한다. Bernstein/Hoeffding 부등식이 각 단계에서 잔차 bound 를 제공한다.

### Part IV: Algorithms (ALM / ADMM) / 알고리즘 (Section 5, pp. 28–30)

The PCP problem is non-smooth but admits closed-form proximal updates for each variable. The augmented Lagrangian:

$$
\ell(L, S, Y) = \|L\|_* + \lambda\|S\|_1 + \langle Y,\; M-L-S\rangle + \frac{\mu}{2}\|M-L-S\|_F^2. \tag{5.1}
$$

Two key proximal operators:

- **Soft-thresholding** $\mathcal S_\tau[x] = \mathrm{sgn}(x)\max(|x|-\tau,0)$, the proximal of $\tau\|\cdot\|_1$.
- **Singular-value thresholding** $\mathcal D_\tau(X) = U\, \mathcal S_\tau(\Sigma)\, V^*$ where $X = U\Sigma V^*$, the proximal of $\tau\|\cdot\|_*$ (Cai-Candès-Shen 2010).

Then alternating-direction updates (Algorithm 1):

$$
\boxed{\;\begin{aligned}
L_{k+1} &= \mathcal D_{\mu^{-1}}(M - S_k - \mu^{-1} Y_k), \\
S_{k+1} &= \mathcal S_{\lambda \mu^{-1}}(M - L_{k+1} + \mu^{-1} Y_k), \\
Y_{k+1} &= Y_k + \mu(M - L_{k+1} - S_{k+1}).
\end{aligned}\;}
$$

The initial $S_0 = Y_0 = 0$, $\mu > 0$. **Termination criterion**: $\|M - L - S\|_F \le \delta\|M\|_F$ with $\delta = 10^{-7}$. Recommended $\mu = n_1 n_2 / (4 \|M\|_1)$ (heuristic from Lin-Chen-Ma 2010). Typically converges in fewer than 50 iterations under "Inexact ALM" (continuation) or in $\sim 700$ iterations for the exact version on real video data.

PCP 는 비매끄럽지만 변수마다 닫힌 형태의 proximal 업데이트가 가능하다. 핵심은 두 연산자 — soft-thresholding $\mathcal S_\tau$ 와 singular-value thresholding $\mathcal D_\tau$ — 그리고 ALM 의 세 줄 업데이트. 부분 SVD 가 반복당 한 번 필요한 것이 지배 비용이며, 무작위 행렬 실험에서 $n=3000$ 까지 SVD 횟수 $<17$ 회로 수렴했다.

### Part V: Numerical Experiments / 수치 실험 (Section 4, pp. 21–27)

**(A) Exact recovery from varying rank/sparsity (Section 4.1, Table 1).** Random $L_0 = X Y^*$, $X, Y \in \mathbb R^{n\times r}$ with i.i.d. $\mathcal N(0, 1/n)$ entries, $S_0$ Bernoulli $\pm 1$ on a uniform random support. Tested $n \in \{500, 1000, 2000, 3000\}$ with $\mathrm{rank}(L_0) = 0.05n$ and $\|S_0\|_0 \in \{0.05 n^2, 0.10 n^2\}$. **Result**: in **every case**, PCP recovers the exact rank, exact $\ell_0$ count of $S_0$, and relative error $\|L - L_0\|_F / \|L_0\|_F \le 2.5 \times 10^{-6}$. Number of partial SVDs needed: 15–17 in all cases — nearly constant in dimension. Total time: 3 s ($n=500$) to 191 s ($n=3000$) on a Mac Pro.

**(A) 정확 회복 실험.** $n=500$–$3000$ 에서 rank $0.05n$, 손상 $5\%$/$10\%$ 의 무작위 행렬 모두에서 PCP 가 정확한 rank·sparsity 를 회복하고 상대 오차 $<10^{-5}$, 부분 SVD 횟수가 차원과 무관하게 거의 일정 (15–17 회). 매우 강력한 실증.

**(B) Phase transition (Section 4.2, Fig. 1).** $400 \times 400$ matrices, sweep $r$ and $\rho$, 10 trials per cell. Three diagrams plotted: random-sign $S_0$, coherent-sign $S_0 = \mathcal P_\Omega \mathrm{sgn}(L_0)$, and pure matrix completion. The recovery region in the $(r, \rho)$ plane is **broad** for all three; for incoherent $L_0$, signs of $S_0$ matter little (Lemma 2.4).

**(B) Phase transition (Fig. 1).** 회복 영역이 $(r,\rho)$ 평면에서 매우 넓다 — 무작위 부호 vs. coherent 부호의 차이는 작다.

**(C) Background modelling from video (Section 4.3, Figs. 2–3).** Airport-lobby video: 200 frames, $176\times 144$ pixels, stacked into $M\in\mathbb R^{25344\times 200}$. Solve PCP with $\lambda = 1/\sqrt{n_1}$. **806 iterations, 43 minutes** on a 2.33 GHz Core 2 Duo. $\hat L$ correctly recovers the static background; $\hat S$ correctly captures moving pedestrians. Lobby video with three drastic illumination changes (250 frames, $168\times 120$): converges in 561 iterations / 36 min; the low-rank component captures the multiple lighting modes; the sparse part captures motion. **Beats** the prior alternating-minimisation $m$-estimator method (De La Torre & Black 2003) despite using less prior knowledge.

**(C) 비디오 배경 분리.** 공항 로비 200 프레임, $176\times 144$. $M$ 은 $25344\times 200$ 행렬. PCP 가 정적 배경을 $\hat L$ 로, 보행자 움직임을 $\hat S$ 로 깨끗이 분리. 조명이 급변하는 lobby 비디오 (250 프레임) 도 동일하게 처리. De La Torre & Black (2003) 의 대안 방법보다 사전 정보 적게 쓰면서도 더 좋은 결과.

**(D) Face recognition (Section 4.4, Fig. 4).** Yale B database, 58 illuminations per subject, $192\times 168$ image. PCP with $\lambda = 1/\sqrt{n_1}$ runs in 642 iterations / 685 s. $\hat L$ produces shadow- and specularity-free face approximations; $\hat S$ exposes the cast-shadows and specular highlights as outliers. Useful for face-alignment / training-data conditioning.

**(D) 얼굴 인식.** Yale B 의 58 조명 이미지에서 그림자 / specularity 가 sparse 항으로 빠지고 $\hat L$ 은 깨끗한 Lambertian 얼굴 추정. Face recognition 학습 데이터 정제에 직접 유용.

### Part VI: Discussion and Extensions / 논의와 확장 (Section 6, p. 30)

The paper closes with three forward-looking remarks: (1) the $L_0$ low-rank assumption can be relaxed to "approximately low-rank" $M = L_0 + S_0 + N_0$, where $N_0$ is i.i.d. Gaussian — a unifying noisy/outlier observation model; (2) the analysis of `low-rank + sparse + dense-noise` parallels "noisy compressed sensing" and matrix-completion stability results; (3) for very-large-scale problems, scalable / online algorithms are an open direction. These hints catalysed a literature on noisy RPCA, online RPCA, deep RPCA, and fast non-convex variants.

논의는 세 갈래의 후속 연구를 시사한다: (i) "근사 저랭크 + sparse + Gaussian 잡음" 으로의 확장, (ii) 잡음 있는 compressed sensing / matrix completion 안정성과의 연결, (iii) 초대규모 / 온라인 / 비볼록 변형. 이것들이 이후 noisy RPCA, deep RPCA, online RPCA, non-convex RPCA 의 시작점이 된다.

---

## 3. Key Takeaways / 핵심 시사점

1. **A one-line convex program solves a non-trivial separation.** $\min \|L\|_* + \lambda\|S\|_1$ s.t. $L+S=M$ — every additional ingredient (incoherence, random support) is a *checkable* hypothesis, not a numerical knob. / **한 줄짜리 볼록 계획** 이 자명하지 않은 분리를 해결한다 — 가정은 모두 검증 가능한 구조 조건.

2. **The penalty $\lambda = 1/\sqrt{n_{(1)}}$ is universal.** No tuning, no knowledge of rank or sparsity. Theoretically derived (not empirical), and works on real images / videos. / **$\lambda$ 의 보편 선택** — 튜닝 불필요, rank·sparsity 모름. 이론적 결과, 실험적 확인.

3. **The convex relaxation is tight under broad randomness.** Rank up to $O(n/\log^2 n)$ and constant fraction of corruptions are recoverable with high probability — far beyond what robust statistics had achieved. / **볼록 완화가 광범위한 무작위 모형에서 tight.** Rank $O(n/\log^2 n)$ 까지, 손상 비율 일정 상수까지 회복 가능.

4. **ADMM with closed-form proximals is exquisite for this problem.** Every inner step has a closed form: $L$-update is SVD-thresholding, $S$-update is soft-thresholding, $Y$-update is a vector add. Just **one partial SVD per iteration**. / **ADMM 의 닫힌 형태 proximal** — $L$ 은 SVD-thresholding, $S$ 는 soft-thresholding, $Y$ 는 벡터 갱신. 반복당 하나의 부분 SVD 만 필요.

5. **Number of SVDs is dimension-independent.** Random-matrix experiments at $n=500\dots 3000$ show 15–17 SVDs suffice — "constant work" in the sense of iteration counts, only SVD cost grows. / **SVD 횟수는 차원에 거의 무관** ($n=500$–$3000$ 모두 15–17 회).

6. **Background-subtraction is the canonical visual demonstration.** A static scene gives the low-rank $L$; foreground motion concentrates in the sparse $S$. The paper's airport / lobby / face examples are the founding case studies that subsequent surveillance work relies on. / **비디오 배경 분리 = 정전적 시각적 데모.** 후속 감시·MRI·hyperspectral 응용의 기반.

7. **Robust matrix completion: missing + corrupted handled together.** Theorem 1.2's $\lambda = 1/\sqrt{0.1 n_{(1)}}$ extends PCP to `Y = P_Ω(L+S)`, unifying matrix completion and outlier rejection. / **결측 + 손상 동시 회복** — Theorem 1.2 의 robust matrix completion.

8. **Solar / astronomy connection.** For coronagraph time series (LASCO / SECCHI / Metis), the **K-corona** is intrinsically low-rank (slow-varying static structure) while CMEs / streamers / cosmic rays are sparse — RPCA gives a physically interpretable separation, contrasted with monolithic background-subtraction or running-difference. / **태양물리 응용**: 코로나그래프 시계열에서 K-corona (저랭크) vs. CME/streamer (희소) 분리.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Norms used / 사용된 놈

For $M \in \mathbb R^{n_1 \times n_2}$ with SVD $M = U\Sigma V^*$, $\sigma_1 \ge \dots \ge \sigma_{r}$:

$$
\|M\| = \sigma_1\;\;\text{(operator)},\qquad \|M\|_F = \sqrt{\sum_i \sigma_i^2}\;\;\text{(Frobenius)},\qquad \|M\|_* = \sum_i \sigma_i\;\;\text{(nuclear)},
$$

$$
\|M\|_1 = \sum_{ij}|M_{ij}|\;\;(\ell_1\;\text{vectorised}),\qquad \|M\|_\infty = \max_{ij}|M_{ij}|.
$$

### 4.2 Principal Component Pursuit / 주성분 추적

$$
\boxed{\;\min_{L,S}\;\|L\|_* + \lambda\|S\|_1\quad\text{s.t.}\quad L + S = M, \qquad \lambda = \frac{1}{\sqrt{n_{(1)}}}\;}
$$

Tuning-free, convex, and admits scalable ADMM/ALM solvers.

### 4.3 Theorem 1.1 (Square case) / 주정리 (정방행렬)

Suppose $L_0 = U\Sigma V^*$ obeys

$$
\max_i \|U^* e_i\|^2 \le \frac{\mu r}{n}, \qquad \max_i \|V^* e_i\|^2 \le \frac{\mu r}{n}, \qquad \|UV^*\|_\infty \le \sqrt{\frac{\mu r}{n^2}},
$$

and $\mathrm{supp}(S_0)$ is uniformly random with $|\mathrm{supp}(S_0)| = m$. Then with probability $\ge 1 - cn^{-10}$, PCP with $\lambda = 1/\sqrt n$ exactly recovers $(L_0, S_0)$ provided

$$
\boxed{\;\mathrm{rank}(L_0) \le \rho_r\, n\, \mu^{-1}(\log n)^{-2},\qquad m \le \rho_s\, n^2\;}
$$

for positive numerical constants $\rho_r, \rho_s$.

### 4.4 Subgradient characterization / 서브그래디언트

$$
\partial \|S_0\|_1 = \big\{\mathrm{sgn}(S_0) + F : \mathcal P_\Omega F = 0,\;\|F\|_\infty \le 1\big\},
$$

$$
\partial \|L_0\|_* = \big\{UV^* + W : U^* W = 0,\; W V = 0,\;\|W\| \le 1\big\}.
$$

### 4.5 Augmented Lagrangian / 증강 라그랑지안

$$
\ell(L, S, Y) = \|L\|_* + \lambda\|S\|_1 + \langle Y,\; M - L - S\rangle + \frac{\mu}{2}\|M - L - S\|_F^2.
$$

### 4.6 Proximal operators / Proximal 연산자

**Soft thresholding (entry-wise)**:

$$
\mathcal S_\tau[x] = \mathrm{sgn}(x)\,\max(|x| - \tau, 0).
$$

**Singular-value thresholding**:

$$
\mathcal D_\tau(X) = U\,\mathcal S_\tau(\Sigma)\, V^*\quad\text{where}\quad X = U\Sigma V^*.
$$

These are the proximal operators of $\tau\|\cdot\|_1$ and $\tau\|\cdot\|_*$ respectively.

### 4.7 ADMM updates (Algorithm 1) / ADMM 갱신식

$$
\boxed{\;
\begin{aligned}
L_{k+1} &= \mathcal D_{\mu^{-1}}\!\left(M - S_k - \mu^{-1} Y_k\right), \\
S_{k+1} &= \mathcal S_{\lambda \mu^{-1}}\!\left(M - L_{k+1} + \mu^{-1} Y_k\right), \\
Y_{k+1} &= Y_k + \mu\big(M - L_{k+1} - S_{k+1}\big).
\end{aligned}\;}
$$

Initialise $S_0 = Y_0 = 0$, $\mu > 0$. Recommended $\mu = n_1 n_2 / (4\|M\|_1)$ (Lin-Chen-Ma 2010). Stopping rule $\|M - L - S\|_F \le 10^{-7}\|M\|_F$.

### 4.8 Robust matrix completion / 결측 + 손상 회복

$$
\min_{L, S}\;\|L\|_* + \lambda\|S\|_1 \quad\text{s.t.}\quad \mathcal P_{\Omega_\text{obs}}(L + S) = Y,\qquad \lambda = \frac{1}{\sqrt{0.1\,n_{(1)}}}.
$$

**Theorem 1.2.** With probability $\ge 1 - c n^{-10}$, PCP exactly recovers $L_0$ provided $\mathrm{rank}(L_0) \le \rho_r\, n\, \mu^{-1}(\log n)^{-2}$ and per-entry corruption probability $\tau \le \tau_s$.

### 4.9 Worked numerical example / 작동 예제

For $n = 500$, $\mathrm{rank}(L_0) = 25$, $\|S_0\|_0 = 12500$ (5%): PCP returns $\hat L$ with $\mathrm{rank}(\hat L) = 25$, $\|\hat S\|_0 = 12500$, $\|\hat L - L_0\|_F / \|L_0\|_F = 1.1\times 10^{-6}$, using **16 partial SVDs** in $\sim 3$ s (Table 1, top row). For $n=3000$ with $\mathrm{rank}(L_0)=150$ and 10% corruption: $\|\hat L - L_0\|_F / \|L_0\|_F = 2.5 \times 10^{-6}$ in 191 s with 16 SVDs.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
 1901  Pearson — Principal Component Analysis (PCA)
   │
 1933  Hotelling — PCA for psychometrics
   │
 1981  Hampel-Rousseeuw-Stahel — robust statistics: influence functions
   │
 2002  Fazel — nuclear norm as convex relaxation of rank
   │
 2006  Donoho; Candès-Romberg-Tao — compressed sensing
   │
 2009  Candès & Recht — Exact matrix completion via convex optimisation
   │
 2009  Lin-Chen-Ma — Inexact ALM for RPCA (algorithmic seed)
   │
 2010  Recht-Fazel-Parrilo — guarantees for nuclear-norm rank minimisation
   │
 2010  Cai-Candès-Shen — Singular Value Thresholding for matrix completion
   │
 2011  Chandrasekaran-Sanghavi-Parrilo-Willsky — deterministic sparse + low-rank
   │
 2011 ★ Candès-Li-Ma-Wright — Robust PCA via Principal Component Pursuit
        (THIS PAPER, JACM 58:11). Universal λ = 1/√n; tuning-free.
   │
 2014  Netrapalli et al. — non-convex RPCA (faster heuristic)
   │
 2017  Wang et al. — RPCA on LASCO C2 to separate K-corona / CMEs
   │
 2020+ Lamy et al. — RPCA on Metis / SECCHI; deep RPCA; LISTA-RPCA
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Candès & Recht (2009) — Exact matrix completion | Same incoherence framework, same nuclear-norm tool; PCP generalises to corrupted entries. / 같은 incoherence, nuclear-norm 도구. | High; methodological parent. |
| Chandrasekaran et al. (2011) — Deterministic decomposition | Concurrent paper with deterministic sufficient conditions and unknown $\lambda$; this paper gives random-model results with universal $\lambda$. / 결정적 vs. 무작위 결과. | High; direct contrast. |
| Donoho (2006) — Compressed sensing | $\ell_1$ as convex envelope of $\ell_0$; PCP combines that with nuclear-norm relaxation. / sparsity ↔ $\ell_1$. | High; conceptual background. |
| Cai-Candès-Shen (2010) — SVT for matrix completion | Provides the singular-value thresholding operator $\mathcal D_\tau$ that powers Algorithm 1's $L$-update. / $L$-update 의 핵심 연산자. | Direct algorithmic input. |
| Lin-Chen-Ma (2010) — Inexact ALM | Gives the ALM/ADMM solver and the heuristic $\mu = n_1 n_2 / (4\|M\|_1)$ used here. / 알고리즘 인용. | Direct algorithmic input. |
| Wang et al. (2017) — RPCA on LASCO | First major application to coronagraph time series for K-corona vs. CME separation. / 본 논문의 코로나 응용. | Solar-physics linkage. |
| Druckmüller (2013) — NAFE / Morgan-Druckmüller (2014) MGN (#38) | Alternative single-image enhancement; RPCA is the multi-frame complement. / 단일 영상 enhancement 의 보완. | Topical. |
| Starck-Fadili-Murtagh (2007) #36 — UWT | Sparse-in-wavelet view of the same data; PCP works in the *pixel/raw* domain. / wavelet sparsity vs. pixel sparsity. | Methodological cousin. |
| Bouwmans et al. (2017) — RPCA review | Comprehensive review of RPCA variants and applications. / 후속 종합 리뷰. | Reference. |
| He-Sun-Wright (2020) — non-convex RPCA | Faster non-convex alternatives traded off for stronger guarantees. / 비볼록 후속. | Modern follow-up. |

---

## 7. References / 참고문헌

- Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). *Robust Principal Component Analysis?* Journal of the ACM, **58**(3), Article 11, 1–37. DOI: 10.1145/1970392.1970395
- Candès, E. J. & Recht, B. (2009). *Exact matrix completion via convex optimization.* Foundations of Computational Mathematics, **9**(6), 717–772.
- Chandrasekaran, V., Sanghavi, S., Parrilo, P. A., & Willsky, A. S. (2011). *Rank-sparsity incoherence for matrix decomposition.* SIAM Journal on Optimization, **21**(2), 572–596.
- Cai, J.-F., Candès, E. J., & Shen, Z. (2010). *A singular value thresholding algorithm for matrix completion.* SIAM Journal on Optimization, **20**(4), 1956–1982.
- Lin, Z., Chen, M., & Ma, Y. (2010). *The augmented Lagrange multiplier method for exact recovery of corrupted low-rank matrices.* arXiv:1009.5055.
- Donoho, D. L. (2006). *Compressed sensing.* IEEE Trans. Inf. Theory, **52**(4), 1289–1306.
- Fazel, M. (2002). *Matrix Rank Minimization with Applications.* Ph.D. thesis, Stanford University.
- Recht, B., Fazel, M., & Parrilo, P. A. (2010). *Guaranteed minimum-rank solutions of linear matrix equations via nuclear norm minimization.* SIAM Review, **52**(3), 471–501.
- Gross, D. (2011). *Recovering low-rank matrices from few coefficients in any basis.* IEEE Trans. Inf. Theory, **57**(3), 1548–1566.
- De La Torre, F. & Black, M. J. (2003). *A framework for robust subspace learning.* International Journal of Computer Vision, **54**(1–3), 117–142.
- Bouwmans, T., Sobral, A., Javed, S., Jung, S. K., & Zahzah, E.-H. (2017). *Decomposition into low-rank plus additive matrices for background/foreground separation: a review for a comparative evaluation with a large-scale dataset.* Computer Science Review, **23**, 1–71.
- Wang, X., Bullock, D. M., Gan, Q., Wang, Y., & Lugaz, N. (2017). *Robust PCA on coronagraph time series.* Space Weather (workshop reports / unpublished applications).
- Lamy, P., Floyd, O., Boclet, B., Wojak, J., Gilardy, H., & Barlyaeva, T. (2020). *Coronal mass ejections: a global view from the LASCO archive.* Space Science Reviews, **216**, 39.

---

## Appendix A. Mini-derivation of the proximal updates / 부록: Proximal 업데이트의 작은 유도

The augmented Lagrangian (Eq. 5.1) is

$$
\ell(L, S, Y) = \|L\|_* + \lambda\|S\|_1 + \langle Y, M-L-S\rangle + \frac{\mu}{2}\|M - L - S\|_F^2.
$$

**$L$-update** (fix $S, Y$): minimise

$$
\|L\|_* - \langle Y, L\rangle + \frac{\mu}{2}\|M - L - S\|_F^2 = \|L\|_* + \frac{\mu}{2}\|L - (M - S + Y/\mu)\|_F^2 + \text{const}.
$$

The minimum of $\|L\|_* + \frac{1}{2\tau}\|L - X\|_F^2$ is $\mathcal D_\tau(X)$ (Cai-Candès-Shen 2010). Setting $\tau = 1/\mu$ and $X = M - S + Y/\mu$ gives

$$
L^+ = \mathcal D_{1/\mu}(M - S + Y/\mu) = \mathcal D_{1/\mu}(M - S - (- Y/\mu)).
$$

The paper's update writes this as $L^+ = \mathcal D_{\mu^{-1}}(M - S - \mu^{-1}Y)$ — note the sign convention on $Y$ (here we used $Y$ with the usual gradient-of-Lagrangian sign).

**$S$-update** (fix $L, Y$): minimise

$$
\lambda\|S\|_1 + \frac{\mu}{2}\|S - (M - L + Y/\mu)\|_F^2.
$$

By the standard $\ell_1$-prox lemma the answer is

$$
S^+ = \mathcal S_{\lambda/\mu}(M - L + Y/\mu).
$$

**$Y$-update**: gradient ascent on the dual,

$$
Y^+ = Y + \mu\big(M - L^+ - S^+\big).
$$

$\ell(L,S,Y)$ 의 $L,S$ 에 대한 부분 최소화는 각각 nuclear-norm proximal ($\mathcal D_{1/\mu}$) 과 $\ell_1$-norm proximal ($\mathcal S_{\lambda/\mu}$) 의 닫힌 형태로 나오며, $Y$ 에 대해선 잔차에 비례하는 그래디언트 상승. 이 세 줄이 Algorithm 1 의 본체다.

---

## Appendix B. Choosing $\mu$ and convergence intuition / 부록: $\mu$ 선택과 수렴 직관

$\mu$ 가 크면 등식 제약 $L+S = M$ 이 빠르게 만족되지만 nuclear/$\ell_1$ 항의 영향이 약해져 일찍 멈추고, $\mu$ 가 작으면 정규화 항이 잘 작용하지만 잔차가 천천히 줄어든다. Lin-Chen-Ma (2010) 의 **continuation/inexact ALM** 은 $\mu$ 를 작은 값에서 시작해 $\rho > 1$ 비율로 증가시키며 가운데 길을 간다 — 본 논문 표 1 의 SVD 횟수 ≤ 17 회는 이 "Inexact ALM" 변형을 사용한 결과. 우리 구현은 단순한 고정 $\mu$ 만 사용하므로 실제로는 더 많은 반복이 필요하다.

If $\mu$ is too large the equality constraint is enforced quickly but the regularisers have little weight; if too small, the dual ascent is slow. The Lin-Chen-Ma "Inexact ALM" continuation $\mu_{k+1} = \rho \mu_k$ (ρ > 1) is what gives the paper's < 17 SVDs at $n = 3000$. Our toy implementation uses a single fixed $\mu$ for clarity, accepting more iterations.
