---
title: "Gray and Color Image Contrast Enhancement by the Curvelet Transform"
authors: Jean-Luc Starck, Fionn Murtagh, Emmanuel J. Candès, David L. Donoho
year: 2003
journal: "IEEE Transactions on Image Processing, 12(6), 706-717"
doi: "10.1109/TIP.2003.813140"
topic: Low_SNR_Imaging
tags: [curvelet, ridgelet, contrast-enhancement, multiscale, edge-detection]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 33. Gray and Color Image Contrast Enhancement by the Curvelet Transform / Curvelet 변환에 의한 흑백/컬러 영상 대비 강화

---

## 1. Core Contribution / 핵심 기여

본 논문은 곡선형 에지(curvilinear edge)를 다중스케일·다중방향에서 비등방적으로 표현하는 *curvelet transform* 을 contrast enhancement 에 본격적으로 적용한 첫 실용 논문이다. 핵심 알고리듬은 (1) 영상에 디지털 curvelet 변환을 적용하여 다중스케일 ridgelet 계수 $w_{j,k}$ 를 얻고, (2) 각 계수에 *잡음 인지형* 비선형 매핑 $y_c(|w_{j,k}|,\sigma_j)$ 를 곱한 뒤, (3) 역변환으로 영상을 복원한다. 매핑함수의 파라미터 $c$ (잡음 임계, $K$-시그마), $m$ (포화 임계), $p$ (비선형 강도), $s$ (동적범위 압축) 가 강조의 모양을 제어한다. 평가에는 Lena, 위성영상, Kodak 컬러 영상이 사용되었으며 정량지표로 (i) 시뮬레이션 막대(bar) 영상의 *Canny edge recovery rate*, (ii) Markov-Potts segmentation 의 marginal-density 충실도가 사용되었다. 결과적으로 잡음을 포함한 곡선/에지 풍부 영상에서 wavelet 기반 강조 (Velde, 1999) 와 Multiscale Retinex 를 분명히 능가했다 (예: SNR=2 에서 wavelet 54.77% vs new curvelet 73.91%).

This paper is the first practical application of the *curvelet transform* — a multiscale, multi-directional, anisotropic representation of curvilinear edges — to contrast enhancement. The pipeline is: (1) take a digital curvelet transform to obtain multiscale ridgelet coefficients $w_{j,k}$, (2) multiply each coefficient by a *noise-aware* nonlinear mapping $y_c(|w_{j,k}|,\sigma_j)$, (3) reconstruct via the inverse curvelet transform. Mapping parameters $c$ (noise threshold, $K$-sigma), $m$ (saturation), $p$ (nonlinearity), $s$ (dynamic-range compression) shape the response. Lena, satellite and Kodak color images are used for evaluation; quantitative metrics include (i) Canny edge-recovery rate on a simulated bar image and (ii) marginal-density / Markov-Potts segmentation fidelity. Curvelet enhancement clearly outperformed wavelet-based enhancement (Velde 1999) and Multiscale Retinex on noisy edge-rich images (e.g. at SNR=2 wavelet recovers 54.77% of edges vs the new curvelet recovers 73.91%).

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (Sec. I, pp. 706-707) / 도입

논문은 영상 가시화의 두 큰 줄기를 정리한다. 첫 번째는 *Retinex* 계열로, 인간 시각의 색 항상성 (color constancy) 을 모사하기 위해 Land (1986) 가 제안한 single-scale retinex (SSR)

$$
R_i(x,y) = \log I_i(x,y) - \log\bigl(F(x,y)*I_i(x,y)\bigr)
$$

에서 출발한다 (Eq. 1). 여기서 $F$ 는 가우시안, $*$ 는 합성곱이다. Jobson 등 (1996) 이 이를 다중스케일로 확장하여 Multiscale Retinex (MSR) 를 정의했다 (Eq. 2-4):

$$
R_{\text{MSR}_i} = \sum_{j=1}^N w_j R_{i,j},\quad R_{i,j}=\log I_i - \log(F_j*I_i),\quad F_j(x,y)=K\exp(-r^2/c_j^2)
$$

권장값은 $N=3$, $c_j\in\{15,80,250\}$ pixel, $w_j=1/N$. 두 번째 줄기는 wavelet 기반 강조로, Velde (1999) 가 dyadic wavelet transform 의 두 방향 ($G^{(h)}, G^{(v)}$) gradient 크기 $G_{j,k}=\sqrt{(w^{(h)}_{j,k})^2+(w^{(v)}_{j,k})^2}$ 에 다음 비선형 함수를 적용했다 (Eq. 5):

$$
y(x)=\begin{cases} (m/c)^p & |x|<c\\ (m/|x|)^p & c\le|x|<m\\ 1 & |x|\ge m\end{cases}
$$

저자들은 "wavelet basis 가 *고도로 비등방인 요소* (정렬, 시트, sheets in cubes) 표현에는 부적절하다" 고 지적하며 ridgelet/curvelet 으로의 도약을 동기화한다.

The introduction surveys two prior families. First, the *Retinex* line — Land's (1986) single-scale retinex (SSR, Eq. 1) modeling human color constancy, extended by Jobson et al. (1996) to multiscale (MSR, Eqs. 2-4) with three Gaussian scales $c_j\in\{15,80,250\}$. Second, the wavelet-based enhancement of Velde (1999) which applies the piecewise function above to dyadic-wavelet gradient magnitudes. The authors then motivate ridgelets/curvelets by noting that wavelet bases are not adapted to *highly anisotropic features* (alignments, sheets in cubes).

### Part II: Curvelet Transform (Sec. II, pp. 707-709) / Curvelet 변환

**Ridgelet (II-A).** 1-D wavelet 함수 $\psi:\mathbb R\to\mathbb R$ 가 admissibility $\int|\hat\psi(\xi)|^2|\xi|^{-1}d\xi<\infty$ 와 $\int_0^\infty |\hat\psi(\xi)|^2\xi^{-2}d\xi=1$ 을 만족할 때, 2-D ridgelet 은 Eq. 7

$$
\psi_{a,b,\theta}(x_1,x_2) = a^{-1/2}\,\psi\!\left(\frac{x_1\cos\theta+x_2\sin\theta-b}{a}\right)
$$

로 정의된다. 이는 직선 $x_1\cos\theta+x_2\sin\theta=\text{const}$ 상에서는 상수이고 그 직선에 수직 방향으로는 wavelet 인 함수다. Ridgelet 계수는 $\mathcal R_f(a,b,\theta)=\int\overline{\psi_{a,b,\theta}(x)}f(x)dx$ 이며 정확한 재구성식은 Eq. 8

$$
f(x) = \int_0^{2\pi}\!\int_{-\infty}^\infty\!\int_0^\infty \mathcal R_f(a,b,\theta)\psi_{a,b,\theta}(x)\,\frac{da}{a^3}\,db\,\frac{d\theta}{4\pi}.
$$

핵심 통찰: ridgelet 분석은 Radon 영역에서의 wavelet 분석과 등가다 — Radon 변환 $Rf(\theta,t)=\iint f(x_1,x_2)\delta(x_1\cos\theta+x_2\sin\theta-t)dx_1dx_2$ 의 angular slice 마다 1-D wavelet 을 적용하면 ridgelet 변환이 된다. 디지털 구현은 polar grid 를 통한 FFT-based Radon 으로 수행되며, $n\times n$ 영상은 $2n\times2n$ ridgelet 배열을 만들어 redundancy 4.

**Curvelet (II-B).** 영상을 $b\times b$ 로 부드럽게 중첩 분할 (overlap $b\times b/2$) 하여 *국소 ridgelet* 을 적용하면 multiscale ridgelet pyramid 가 된다. Curvelet 은 이 dictionary 의 부분집합으로, scale $j$ 에서 길이 $\sim 2^{-j/2}$, 폭 $\sim 2^{-j}$, 즉 $\text{width}\approx\text{length}^2$ 의 비등방 스케일링 법칙을 따른다. 알고리듬은 (i) à trous wavelet 분해

$$
I(x,y) = c_J(x,y)+\sum_{j=1}^J w_j(x,y),
$$

(ii) 각 $w_j$ 를 분할하고 ridgelet 적용, 으로 진행된다. 기본 블록 크기는 $b_{\min}=16$ 이며 dyadic subband 마다 두 배씩 증가한다. 이 구현은 redundant ($16J+1$) 하고 안정적·역가역적이다.

**Ridgelet (II-A).** Given a 1-D wavelet $\psi$ satisfying admissibility, the bivariate ridgelet (Eq. 7) is constant along $x_1\cos\theta+x_2\sin\theta=\text{const}$ and wavelet-like transverse to it. Coefficients are $\mathcal R_f(a,b,\theta)=\int\overline{\psi_{a,b,\theta}}f$, with exact reconstruction Eq. 8. The key insight: ridgelet analysis = wavelet analysis on Radon slices. Digital implementation uses FFT on a polar grid; an $n\times n$ image produces a $2n\times 2n$ ridgelet array (redundancy 4).

**Curvelet (II-B).** Smoothly overlapping $b\times b$ block partitioning + local ridgelet on each block builds the multiscale-ridgelet pyramid. Curvelets are a subset following the anisotropic scaling $\text{width}\approx\text{length}^2$. Algorithm: (i) à trous wavelet pyramid producing $w_j$, (ii) partition each $w_j$ and apply local ridgelet. Default $b_{\min}=16$, doubles every other dyadic scale. The transform is redundant ($16J+1$), stable, exactly invertible.

### Part III: Contrast Enhancement (Sec. III, pp. 709-711) / 대비 강화

논문의 핵심 기여 — Eq. 10 의 *잡음 인지형* 4-구간 비선형 매핑:

$$
y_c(x,\sigma)=\begin{cases}
1 & x<c\sigma\\[4pt]
\dfrac{x-c\sigma}{c\sigma}\Bigl(\dfrac{m}{c\sigma}\Bigr)^p+\dfrac{2c\sigma-x}{c\sigma} & c\sigma\le x<2c\sigma\\[8pt]
\Bigl(\dfrac{m}{x}\Bigr)^p & 2c\sigma\le x<m\\[6pt]
\Bigl(\dfrac{m}{x}\Bigr)^s & x\ge m
\end{cases}
$$

각 구간의 의미:
- $x<c\sigma$ — 잡음 영역 (보통 $c\ge 3$): 곱셈 인자 1, 즉 *증폭 안 함* (noise preservation).
- $c\sigma\le x<2c\sigma$ — 전이 구간: noise→signal 의 부드러운 보간.
- $2c\sigma\le x<m$ — 신호 강조 구간: $(m/x)^p$ 가 *작은* $x$ 에 *큰* 인자를, *큰* $x$ 에 *작은* 인자를 부여 — 약한 에지가 가장 많이 증폭됨.
- $x\ge m$ — 동적범위 압축 구간: $s>0$ 으로 강한 에지를 *약간 부드럽게*.

권장 운용: (a) noise std $\sigma$ 추정 → $K_m\sigma=m$ 로 잡음 대비 SNR 단위로 $K_m$ 를 사용자가 지정 (예: $c=3$, $K_m=10$ 이면 SNR 3-10 의 계수를 강조), 또는 (b) 각 밴드에서 max coefficient $M_c$ 의 분수 $m=lM_c$ ($l<1$) 로 자동 설정.

**알고리듬 단계 (Sec. III, p. 710):**
1. 입력 영상에서 잡음 표준편차 $\sigma$ 추정 (MAD on finest wavelet scale).
2. Curvelet 변환 → bands $\{w_j\}$ 와 각 band 의 noise std $\sigma_j$ 계산.
3. 각 band 에 대해 $M_j=\max|w_j|$ 계산, 모든 계수 $w_{j,k}\to y_c(|w_{j,k}|,\sigma_j)\cdot w_{j,k}$.
4. 역 curvelet 변환으로 재구성.

**컬러 영상 확장:** 영상을 LUV 공간으로 변환하여 세 채널 $L,u,v$ 각각에 curvelet 변환을 적용. 위치 $k$ 에서 *결합* gradient norm

$$
e = \sqrt{c_L^2+c_u^2+c_v^2}
$$

을 계산하고, 모든 채널에 동일한 $y_c(e,\sigma)$ 를 곱해 색 일관성을 유지한다. 강조 후 픽셀 값이 $[0,255]$ 를 넘어가면 gain/offset clipping (k-sigma) 으로 클리핑.

The heart of the paper is the noise-aware four-piece nonlinear mapping (Eq. 10). Region semantics: below $c\sigma$ is *noise* (multiplier 1, do nothing); $[c\sigma,2c\sigma]$ is the smooth transition; $[2c\sigma,m]$ is the *enhancement* range where $(m/x)^p$ assigns the largest factor to the smallest signal coefficients (faint edges boosted most); above $m$ is the *dynamic-range compression* range with $s$. The user chooses $K_m$ (so $m=K_m\sigma$, e.g. $K_m=10$, $c=3$) or sets $m=lM_c$ from band maximum. Color extension uses LUV space and a *joint* gradient norm $e=\sqrt{c_L^2+c_u^2+c_v^2}$ to keep color consistency.

### Part IV: Evaluation (Sec. IV, pp. 712-715) / 평가

**Edge detection (IV-B).** 시뮬레이션 영상: 6 개 막대(rectangle, 20×150 pixel, $30^\circ$ 기울기), 강도 $\{1,2,3,4,5,8\}$, 가산 가우시안 잡음 $\sigma=1$. 각 막대의 SNR 이 다르므로 *recovered edge percentage* 를 SNR 의 함수로 그릴 수 있다. Canny edge detector 를 (i) wavelet-enhanced, (ii) curvelet-enhanced with Velde's $y$, (iii) curvelet-enhanced with new $y_c$ 위에서 실행한 결과 (Fig. 10):

| Method | Edge recovery (Fig. 9 image) |
|---|---|
| Wavelet enhancement | **54.77%** |
| Curvelet w/ Velde's $y$ | 64.66% |
| Curvelet w/ new $y_c$ | **73.91%** |

새 매핑 함수는 모든 SNR 구간에서 wavelet 보다 우수하고, 특히 SNR≈2 의 가장 약한 막대에서 18 pp 이상의 격차를 만든다.

**Marginal density / segmentation (IV-C).** 512×512 Lena 영상에 대해 5-component Gaussian + Markov-Potts 모형 (3×3 neighborhood) 으로 분할 (Fig. 12-15). Histogram equalization 은 marginal density 가 거의 균등 분포로 망가져서 (Fig. 11 top right) 분류 정보 소실; wavelet enhancement 도 일부 평탄 영역의 정보를 잃는다; *curvelet enhancement 만이* original marginal density 의 모양을 충실히 보존 (Fig. 11 bottom right). Pseudo-likelihood 정보 척도 값은 각각 0.72/0.72/0.63/0.73 (original/HE/wavelet/curvelet).

**주관적 평가 (IV-A).** 가시 비교 (Fig. 5 Lena, Fig. 6 satellite harbor, Fig. 7 outdoor color, Fig. 8 indoor color): curvelet 결과는 정렬된 작은 특징을 잘 보존하고 색 충실도가 좋다. MSR 은 outdoor 컬러 (Fig. 7) 에서 grayness tendency 가 보임.

**Edge detection (IV-B).** Simulated 6-bar image (20×150 pixels, $30^\circ$ tilt, intensities $\{1,2,3,4,5,8\}$, Gaussian noise $\sigma=1$). Canny edge recovery: wavelet 54.77%, curvelet+Velde 64.66%, curvelet+new $y_c$ 73.91%. New mapping wins at every SNR. **Segmentation (IV-C)** uses 5-component Gaussian + Markov-Potts on Lena. Histogram equalization destroys the marginal density; wavelet smooths some flat regions; curvelet alone preserves the original density shape (pseudo-likelihood 0.72/0.72/0.63/0.73).

### Part V: Conclusion (Sec. V, p. 716) / 결론

세 결론: (1) curvelet/wavelet 강조 함수는 영상 잡음을 잘 다룬다 (잡음 영역에서 곱셈 인자 1). (2) curvelet 으로 *잡음이 있는* contour 의 검출이 wavelet 보다 좋다. (3) 잡음이 없는 영상에서는 강조 함수가 Velde 의 함수와 유사해져 wavelet 과 큰 차이가 없다. 즉 curvelet 의 우위는 *low-SNR regime* 에 한정된다.

Three conclusions: (1) the enhancement functions handle noise well (multiplier 1 on the noise band); (2) curvelet beats wavelet for *noisy* contour detection; (3) on noise-free images the function reduces to Velde's, so curvelet's edge over wavelet is small. The curvelet advantage is thus a *low-SNR* effect.

---

## 3. Key Takeaways / 핵심 시사점

1. **곡선 표현은 wavelet basis 보다 curvelet 이 효율적이다 / Curvelets are more efficient than wavelets for curves** — 비등방 스케일링 $\text{width}\approx\text{length}^2$ 가 $C^2$ 곡선의 sparsest representation 을 보장한다 (Donoho 의 "optimal nonlinear approximation" 결과). This anisotropic scaling guarantees the sparsest representation of $C^2$ curves (Donoho's optimal nonlinear-approximation result).

2. **잡음 임계 $c\sigma$ 가 노이즈 증폭을 차단 / The noise threshold $c\sigma$ blocks noise amplification** — 잡음 영역에서 $y_c=1$ 로 곱해 *증폭 없음*. 이는 enhancement 가 denoising 을 *암묵적으로* 포함하게 한다. With $y_c=1$ on the noise band, no amplification occurs — enhancement implicitly performs denoising.

3. **약한 에지가 가장 많이 강조된다 / Faint edges are boosted the most** — 신호 영역의 $(m/x)^p$ 가 작은 $x$ 에 큰 인자를 부여 — 약한 신호 ($x$ 작음) → 큰 multiplier; 강한 신호 ($x\to m$) → multiplier 1. Faintest features see the largest multiplier; strongest features see ~1.

4. **동적범위 압축 파라미터 $s$ 의 도입이 핵심 신규성 / Adding the dynamic-range parameter $s$ is the key novelty** — Velde 의 함수는 강한 에지에서 multiplier 가 1; 새 $y_c$ 는 $s>0$ 으로 강한 에지를 약간 *부드럽게* 만들어 동시에 *동적 범위 압축* 을 수행. Velde's function leaves strong edges untouched; the new $y_c$ uses $s>0$ to softly compress them, integrating dynamic-range compression.

5. **컬러는 LUV + joint gradient norm 으로 처리 / Color via LUV + joint gradient norm** — 채널별로 독립 처리하면 색감이 깨진다 → $e=\sqrt{c_L^2+c_u^2+c_v^2}$ 로 *동일* multiplier 를 모든 채널에 적용. Independent per-channel processing breaks color; using a joint norm keeps a single multiplier across channels.

6. **정량 평가 — Canny edge recovery 와 Markov-Potts segmentation / Quantitative evaluation via Canny edge recovery and Markov-Potts segmentation** — 주관적 비교를 넘어 (i) edge-recovery rate vs SNR curve, (ii) marginal-density 충실도로 객관 측정. 실용 알고리듬에서 본받아야 할 표준. Beyond visual comparison, two objective metrics are introduced as a template for future work.

7. **저-SNR 한정 우위 / Advantage limited to low SNR** — Conclusion (3): noise-free 영상에서는 curvelet 의 이점이 작다. 따라서 본 논문이 가장 큰 가치를 갖는 응용은 *intrinsically noisy* 영상 — 의료영상, 위성영상, **태양 코로나그래프** 등. The curvelet edge is largest precisely on intrinsically noisy data — medical imaging, satellite, **solar coronagraphs**.

8. **재사용 가능한 noise-aware enhancement curve 디자인 패턴 / Reusable design pattern for noise-aware enhancement curves** — Eq. 10 의 *4-piece* 구조 (noise / transition / signal / saturation) 는 wavelet, curvelet 외 다른 sparse 변환 (shearlet, contourlet) 에도 직접 이식 가능. The four-piece curve design generalizes to other sparse transforms (shearlets, contourlets).

---

## 4. Mathematical Summary / 수학적 요약

**Single-scale retinex (Eq. 1):**

$$
R_i(x,y)=\log I_i(x,y)-\log\!\bigl(F(x,y)*I_i(x,y)\bigr)
$$

— $I_i$ 는 $i$ 번째 채널의 영상 강도; $F$ 는 가우시안 surround; 출력은 잔차 contrast.

**MSR (Eq. 2-4):**

$$
R_{\text{MSR}_i}=\sum_{j=1}^N w_j\bigl[\log I_i-\log(F_j*I_i)\bigr],\quad F_j(r)=K e^{-r^2/c_j^2}
$$

— $N=3$, $c_j\in\{15,80,250\}$, $w_j=1/N$.

**Velde's wavelet enhancement (Eq. 5):**

$$
y(x)=\begin{cases}(m/c)^p & |x|<c\\ (m/|x|)^p & c\le|x|<m\\ 1 & |x|\ge m\end{cases}
$$

— $p\in[0,1]$ 비선형성, $c$=잡음 임계, $m$=포화 상한.

**À trous decomposition (paper, Sec. II-B):**

$$
I(x,y) = c_J(x,y) + \sum_{j=1}^J w_j(x,y)
$$

— $c_J$ 는 coarse 잔여, $w_j$ 는 scale $2^{-j}$ 의 detail.

**Bivariate ridgelet (Eq. 7) and reconstruction (Eq. 8):**

$$
\psi_{a,b,\theta}(x_1,x_2)=a^{-1/2}\psi\!\left(\frac{x_1\cos\theta+x_2\sin\theta-b}{a}\right)
$$

$$
f(x)=\int_0^{2\pi}\!\!\int_{-\infty}^\infty\!\!\int_0^\infty \mathcal R_f(a,b,\theta)\,\psi_{a,b,\theta}(x)\,\frac{da}{a^3}\,db\,\frac{d\theta}{4\pi}
$$

**Radon link (Eq. 9):**

$$
Rf(\theta,t)=\iint f(x_1,x_2)\,\delta(x_1\cos\theta+x_2\sin\theta-t)\,dx_1\,dx_2
$$

— Ridgelet = 1-D wavelet on slices of $Rf$.

**Curvelet anisotropy law:**

$$
\text{width}\approx\text{length}^2\qquad(\text{at scale }j:\ \text{length}\sim 2^{-j/2},\text{width}\sim 2^{-j})
$$

**Noise-aware curvelet enhancement (Eq. 10) — paper's central equation:**

$$
y_c(x,\sigma)=\begin{cases}
1 & x<c\sigma\\[3pt]
\dfrac{x-c\sigma}{c\sigma}\Bigl(\dfrac{m}{c\sigma}\Bigr)^p+\dfrac{2c\sigma-x}{c\sigma} & c\sigma\le x<2c\sigma\\[8pt]
\bigl(m/x\bigr)^p & 2c\sigma\le x<m\\[3pt]
\bigl(m/x\bigr)^s & x\ge m
\end{cases}
$$

**Color enhancement gradient norm:**

$$
e=\sqrt{c_L^2+c_u^2+c_v^2}\,,\qquad (\tilde c_L,\tilde c_u,\tilde c_v)=\bigl(y_c(e,\sigma)c_L,\,y_c(e,\sigma)c_u,\,y_c(e,\sigma)c_v\bigr)
$$

**Algorithmic pseudo-code (Sec. III, p. 710).** The full pipeline at a glance:

```
procedure CURVELET_CONTRAST_ENHANCE(I, c, m_factor, p, s)
    σ ← MAD(I_finest_wavelet_band) / 0.6745       # noise std estimate
    {w_j}, c_J ← À_TROUS_DECOMPOSE(I, J=4)        # multiscale pyramid
    for each detail band w_j do
        σ_j ← σ * normalization_factor(j)         # band-wise noise std
        partition w_j into overlapping b×b blocks
        for each block B do
            R_B ← LOCAL_RIDGELET_FORWARD(B)       # FFT-Radon then 1-D wavelet
            for each ridgelet coefficient r in R_B do
                m ← m_factor * σ_j
                r ← r * y_c(|r|, σ_j; c, m, p, s)  # Eq. 10
            B' ← LOCAL_RIDGELET_INVERSE(R_B)
        reassemble blocks → w_j_enhanced
    I_enhanced ← c_J + Σ w_j_enhanced
    return clip(I_enhanced, [0, 255])
end procedure
```

The block size $b_{\min}=16$ (paper default) is doubled at every other dyadic subband, maintaining the curvelet width-length scaling.

**의사코드 (Sec. III, p. 710).** 전체 파이프라인을 한눈에 — 잡음 std 추정, à trous 분해, 각 detail band 의 블록 ridgelet, Eq. 10 매핑, 역변환, 재구성, 클리핑. 블록 크기 $b_{\min}=16$ 은 dyadic 한 단계마다 두 배가 된다.

**Why noise threshold = $K\sigma$ with $K\ge 3$ ?** Donoho-Johnstone (1994) 의 universal threshold $\sqrt{2\log N}$ 가 sample size $N\sim 10^4$ 에서 약 4.3 인 것을 고려하면 $c=3$ 은 *간소화* 된 실용적 선택이다. False-positive 측면에서 가우시안 가정 하 $P(|x|>3\sigma)\approx 0.27\%$, 영상 한 장 ($512^2\approx 2.6\times 10^5$ 픽셀) 에서 약 700 false positive — 강조 함수의 multiplier 가 1 인 것은 이 false positive 가 *증폭되지 않음* 을 보장한다.

For a Gaussian background with $c=3$, $P(|x|>3\sigma)\approx 0.27\%$ (~700 false positives in a 512² image). Critically, since $y_c=1$ in this band, those false positives are *not amplified* — the design tolerates them.

**Worked numerical example (paper Fig. 1 caption).** $m=30$, $c=3$, $p=0.5$, $\sigma=1$ (so noise band is $|x|<3$, saturation at $x=30$). Then for an edge with $|x|=4$ (just above the noise floor):

$$
y_c(4,1)=(30/4)^{0.5}\approx 2.74
$$

i.e. the coefficient is multiplied by 2.74. For $|x|=20$:

$$
y_c(20,1)=(30/20)^{0.5}\approx 1.22
$$

i.e. the strong edge is barely amplified. The faint edge gains $2.74/1.22\approx 2.25\times$ more than the strong one — this is the *contrast equalization* principle.

**Second worked example — color image at coordinate $k$.** $\sigma=2$, $c=3$, $m=20$, $p=0.5$. LUV coefficients $(c_L,c_u,c_v)=(8,3,2)$. Joint norm $e=\sqrt{8^2+3^2+2^2}=\sqrt{77}\approx 8.77$. Since $2c\sigma=12 > 8.77 > c\sigma=6$, we are in the *transition* region:

$$
y_c(8.77,2)=\frac{8.77-6}{6}\Bigl(\frac{20}{6}\Bigr)^{0.5}+\frac{12-8.77}{6}=0.462\cdot 1.826+0.538\approx 1.38
$$

All three channels are multiplied by the same factor 1.38, preserving the chromatic ratio $L:u:v=8:3:2\to 11.04:4.14:2.76$. This *single-multiplier-per-pixel* rule is what keeps color fidelity across the enhancement.

**LUV joint enhancement 예시.** 채널별 계수 (8,3,2), $\sigma=2$, $c=3$, $m=20$, $p=0.5$. 결합 norm $e\approx 8.77$ 은 transition 구간 ($6 < e < 12$) 에 속하여 multiplier 1.38. 세 채널 모두 동일 multiplier 로 곱해져 색비율이 유지된다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1986 Land — Retinex theory of color constancy
   │
1989 Mallat — dyadic wavelet image processing
   │
1996 Jobson, Rahman, Woodell — Multiscale Retinex
   │
1999 Velde — wavelet-based contrast enhancement
   │
1999 Candès — Ridgelets (harmonic analysis)
   │
2000 Candès & Donoho — first curvelet construction
   │
2002 Starck-Candès-Donoho — curvelet image denoising (TIP)
   │
2003 ★ THIS PAPER — curvelet contrast enhancement (TIP)
   │
2006 Candès-Demanet-Donoho-Ying — Fast Discrete Curvelet (FDCT)
   │
2008 Easley-Labate-Lim — Shearlets (alternate anisotropic dictionary)
   │
2014 Morgan & Druckmüller — MGN coronal enhancement
   │
2018+ Deep-learning sparse representations (curvelet-init nets)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Candès & Donoho 2000 (curvelet construction) | 본 논문이 적용하는 변환의 수학적 정의 / Mathematical definition of the transform applied here | High — without this, the algorithm has no basis |
| Velde 1999 (wavelet enhancement) | Eq. 10 이 일반화하는 직접적 선조 함수 / Direct ancestor of Eq. 10 | High — paper's enhancement curve is built on Velde's |
| Starck, Candès, Donoho 2002 (curvelet denoising, TIP) | 동일 변환을 *denoising* 에 적용한 직전 논문 / Companion paper applying same transform to denoising | High — denoising and enhancement are two faces of the same coefficient-shrinkage idea |
| Jobson et al. 1996 (MSR) | 본 논문이 비교하는 baseline / Baseline compared in evaluation | Medium — different philosophy (illumination model) but compared head-to-head |
| Candès-Demanet-Donoho-Ying 2006 (FDCT) | 본 논문 알고리듬의 *현대적* 후속 / Modern successor algorithm | High — practitioners today use FDCT, not the 2003 implementation |
| Morgan & Druckmüller 2014 (MGN) | 본 논문의 *noise-aware enhancement curve* 아이디어를 태양 코로나에 직접 적용한 후속 / Coronal application of the noise-aware enhancement-curve idea | High — direct lineage to Low_SNR_Imaging topic |
| Donoho 1995 (wavelet shrinkage) | 잡음 임계 $c\sigma$ 의 통계적 정당성 / Statistical foundation of the $c\sigma$ noise threshold | Medium — provides the K-sigma rationale |

---

## 6.5 Practical Application Notes / 실무 적용 노트

**For solar coronagraph imaging (LASCO C2, K-Cor, Metis):** The enhancement curve maps directly onto faint streamer/CME detection. Practitioners typically (a) preprocess with running-difference or temporal-median background subtraction, (b) apply à trous + ridgelet, (c) use parameters $c=3$, $m=K_m\sigma$ with $K_m=8\text{–}15$, $p\in[0.3,0.5]$ to avoid amplifying granulation/F-corona residuals while bringing out CME fronts. The 2003 method has been largely supplanted by FDCT-based pipelines and Multi-scale Gaussian Normalization (MGN, Morgan & Druckmüller 2014), but the *parameter philosophy* — separate noise band, controlled boost in the signal range, soft compression of the saturation range — is unchanged.

**태양 코로나 영상 (LASCO C2, K-Cor, Metis) 적용:** 일반적 작업 흐름은 (a) running-difference / temporal-median 배경 차감, (b) à trous + ridgelet, (c) $c=3$, $K_m=8\text{–}15$, $p\in[0.3,0.5]$ — granulation/F-corona 잔차를 증폭하지 않으면서 CME front 를 가시화. 2003 년 방법은 FDCT 와 MGN (Morgan & Druckmüller 2014) 에 의해 대부분 대체되었으나 *파라미터 철학* (noise band 분리, signal 강조, saturation 압축) 은 그대로 유지된다.

**Sensitivity to parameter choice (sweep observation).** From the implementation notebook's $K_m$ sweep:

| $K_m$ | Visual effect / 시각 효과 | Risk / 위험 |
|---|---|---|
| 5 | Mild boost; conservative | Faint edges undertreated |
| 10 (paper default) | Balanced; clear edge enhancement | — |
| 20 | Strong contrast; faint structures pop | Amplifies coupled noise patterns |
| 40 | Saturation-driven look; "HDR-like" | Posterization, loss of mid-tone fidelity |

The recommended operating regime for low-SNR data is $K_m\in[8,15]$, $c=3$, $p=0.5$.

**파라미터 민감도.** $K_m=10$ 이 논문 기본값이며 저-SNR 데이터에서 권장 범위는 $K_m\in[8,15]$, $c=3$, $p=0.5$.

**Failure modes / 실패 모드.**
1. **Block boundary artifacts / 블록 경계 인공물** — overlap+window blending 이 부족하면 grid artifact 발생. 해결: 블록 크기의 1/3~1/2 overlap, cosine-bell window.
2. **Color desaturation / 색 채도 소실** — channel-independent processing 시 발생. 해결: joint LUV gradient norm.
3. **Halo around strong edges / 강한 에지 주변 halo** — $s$ 가 너무 작으면 dynamic range 압축 실패로 over-shoot. 해결: $s\in[0.4,0.7]$.
4. **Noise pattern lock-in / 잡음 패턴 고정** — $c$ 가 너무 작으면 (예: 1.5) 잡음을 신호로 오해석. 해결: $c\ge 3$ 유지.

---

## 7. References / 참고문헌

- J.-L. Starck, F. Murtagh, E. J. Candès, D. L. Donoho, "Gray and Color Image Contrast Enhancement by the Curvelet Transform," *IEEE Trans. Image Process.*, 12(6), 706-717, 2003. DOI: 10.1109/TIP.2003.813140
- E. J. Candès, "Harmonic analysis of neural networks," *Appl. Comput. Harmon. Anal.*, 6, 197-218, 1999.
- E. J. Candès and D. L. Donoho, "Curvelets — A surprisingly effective nonadaptive representation for objects with edges," in *Curve and Surface Fitting: Saint-Malo 1999*, Vanderbilt Univ. Press, 2000.
- J. L. Starck, E. J. Candès, D. L. Donoho, "The curvelet transform for image denoising," *IEEE Trans. Image Process.*, 11(6), 670-684, 2002.
- D. J. Jobson, Z. Rahman, G. A. Woodell, "A multi-scale retinex for bridging the gap between color images and the human observation of scenes," *IEEE Trans. Image Process.*, 6(7), 965-976, 1997.
- K. V. Velde, "Multi-scale color image enhancement," *Proc. ICIP*, vol. 3, pp. 584-587, 1999.
- E. Land, "Recent advances in retinex theory," *Vis. Res.*, 26(1), 7-21, 1986.
- E. J. Candès, L. Demanet, D. L. Donoho, L. Ying, "Fast discrete curvelet transforms," *Multiscale Model. Simul.*, 5, 861-899, 2006.
- H. Morgan and M. Druckmüller, "Multi-scale Gaussian normalization for solar image processing," *Solar Phys.*, 289, 2945-2955, 2014.
- J. L. Starck, F. Murtagh, A. Bijaoui, *Image Processing and Data Analysis: The Multiscale Approach*, Cambridge Univ. Press, 1998.
