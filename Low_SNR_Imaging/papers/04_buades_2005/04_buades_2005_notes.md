---
title: "A Non-Local Algorithm for Image Denoising"
authors: Antoni Buades, Bartomeu Coll, Jean-Michel Morel
year: 2005
journal: "IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Vol. 2, pp. 60–65"
doi: "10.1109/CVPR.2005.38"
topic: Low-SNR Imaging / Spatial-Domain Denoising
tags: [non-local-means, nlm, patch-similarity, self-similarity, method-noise, buades-coll-morel, spatial-domain, weighted-averaging]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 4. A Non-Local Algorithm for Image Denoising / 영상 노이즈 제거를 위한 비국소 알고리즘

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 영상 노이즈 제거 분야에 두 가지 결정적 기여를 한다.

(A) **Method noise**: 노이즈 제거 알고리즘 $D_h$의 *품질*을 정량적으로 평가하는 새로운 척도. **노이즈가 거의 없는 영상** $u$에 알고리즘을 적용했을 때의 차이 $u - D_h u$를 *method noise*로 정의 (Definition 1). 좋은 알고리즘은 method noise가 백색잡음처럼 보여야 하며, 구조(에지·텍스처)가 method noise에 남으면 알고리즘이 *그 구조를 부수고 있다*는 의미. 이 통찰로 Gaussian, anisotropic, TV, neighborhood 필터의 *수학적 한계*를 명시적으로 보임 (Theorems 1-3).

(B) **Non-Local Means (NL-means)**: 영상의 **자기 유사성(self-similarity)**을 활용한 새로운 알고리즘. 픽셀 $i$의 추정값은 *영상 전체*에서 $i$의 주변 patch $\mathcal N_i$와 비슷한 patch $\mathcal N_j$를 가진 모든 픽셀 $j$의 가중 평균:
$$
\boxed{\;NL[v](i) = \sum_{j \in I} w(i, j)\,v(j), \quad w(i,j) = \frac{1}{Z(i)} \exp\left(-\frac{\|v(\mathcal N_i) - v(\mathcal N_j)\|^2_{2, a}}{h^2}\right)\;}
$$
$\mathcal N_k$는 $k$ 중심의 $7\times 7$ patch, $\|\cdot\|_{2, a}$는 가우시안-가중 $L^2$ 거리 ($a$는 patch내 가우시안 분산), $h \approx 10\sigma$는 필터 파라미터.

핵심 통찰: 자연 영상은 *비국소 redundancy*가 풍부 (직선·곡선·flat 영역·텍스처가 영상 곳곳에 *비슷한 모양으로* 나타남). 픽셀 자체가 아니라 *patch*를 비교함으로써 잡음에 강건하면서 에지·텍스처를 보존.

(C) **Conditional expectation 일관성 (Theorem 4)**: stationarity 조건 하에 NL-means가 *조건부 기댓값* $E[Y_i | X_i = v(\mathcal N_i \setminus \{i\})]$에 수렴 → MSE 의미에서 *최적*.

### English
Two decisive contributions:

(A) **Method noise**: A novel quality measure for denoising algorithms — apply $D_h$ to a *nearly noiseless* image $u$ and inspect the difference $u - D_h u$. Good algorithms produce method noise resembling white noise; structures (edges, texture) appearing in the method noise reveal what the algorithm is destroying. Theorems 1–3 use this to expose the limitations of Gaussian filtering, anisotropic diffusion, total-variation minimisation, and neighborhood filtering.

(B) **Non-Local Means (NL-means)**: Exploits the **self-similarity** of natural images. The estimate at pixel $i$ is a weighted average over the *entire image* of pixels whose surrounding patches $\mathcal N_j$ resemble $\mathcal N_i$:
$$
NL[v](i) = \sum_j w(i,j) v(j), \qquad w(i,j) = \frac{1}{Z(i)}\exp\bigl(-\|v(\mathcal N_i) - v(\mathcal N_j)\|^2_{2,a}/h^2\bigr)
$$
with patch size $7\times 7$, Gaussian-weighted $L^2$ distance, and $h \approx 10\sigma$.

Key insight: natural images have abundant *non-local redundancy* — lines, curves, flat regions, textures recur throughout the image. Comparing whole patches (not pixel intensities) is robust to noise while preserving edges and texture.

(C) **Consistency theorem (Thm 4)**: Under stationarity, NL-means converges to the conditional expectation $E[Y_i | X_i]$ — MSE-optimal.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction / 서론

#### 한국어
- 모델: $v(i) = u(i) + n(i)$, $n \overset{iid}{\sim} N(0, \sigma^2)$. 목표: $u$ 복원.
- 기존 방법은 모두 *averaging*: 가우시안 (Gabor 1960), anisotropic (Perona-Malik), neighborhood (Yaroslavsky 1985 / SUSAN / bilateral), TV-min (Rudin-Osher-Fatemi), 주파수 (empirical Wiener), 웨이블릿 (paper #1).
- 모두 *국소(local)* 평균. NL-means는 *비국소* — 영상 전체에서 비슷한 patch를 찾음.

#### English
Standard methods all average locally; NL-means averages non-locally over patches that match in structure.

---

### Part II: §2 Method Noise / 방법 잡음

#### 한국어 — Definition 1
**Method noise** of denoiser $D_h$ on image $u$: $u - D_h u$. 좋은 $D_h$에서는 method noise = white noise. 구조가 보이면 $D_h$가 그 구조를 잘못 평탄화한 것.

이 단순 도구가 paper의 강력한 분석 무기.

#### 한국어 — Theorem 1 (Gaussian filtering, Gabor 1960)
$$
u - G_h * u = -h^2 \Delta u + o(h^2)
$$
- Method noise = $-h^2 \Delta u$ (라플라시안). flat 영역 ($\Delta u \approx 0$)에선 small, edge·texture ($\Delta u$ 큼)에선 큼 → **edge·texture 파괴**. Gaussian의 본질적 한계.

#### 한국어 — Theorem 2 (Anisotropic, Perona-Malik)
$$
u - AF_h u = -\tfrac{1}{2} h^2 |Du|\,\mathrm{curv}(u) + o(h^2)
$$
- Method noise = curvature × |gradient|. 직선 edge ($\mathrm{curv} = 0$)에선 0, 휘어진 edge·texture에선 큼. **직선은 보존, 곡선·texture는 파괴**.

#### 한국어 — Theorem 3 (TV minimisation, ROF)
$$
u - TVF_\lambda u = -\frac{1}{2\lambda}\,\mathrm{curv}(TVF_\lambda(u))
$$
- Method noise = curvature of denoised image. 직선 edge 보존, 곡선·detail은 $\lambda$에 따라 over-smooth.

이 셋 모두 *국소 differential operator*에 기반 → **공통 한계**: edge나 texture 같은 *고주파 구조*를 평탄화.

#### 한국어 — §2.4 Neighborhood filter (Yaroslavsky)
$$
YNF_{h,\rho} u(\mathbf x) = \frac{1}{C(\mathbf x)}\int_{B_\rho(\mathbf x)} u(\mathbf y) e^{-|u(\mathbf y) - u(\mathbf x)|^2 / h^2}\,d\mathbf y \quad (2)
$$
중심 픽셀과 *grey level이 비슷한* 주변 픽셀들 평균. SUSAN, bilateral filter도 같은 계열. **단점**: 단일 픽셀 비교는 잡음에 약함, 구조 정보 무시.

#### English — §2 Method noise highlights
- **Method noise** $u - D_h u$ reveals algorithm bias.
- **Gaussian**: method noise = $-h^2\Delta u$ → wipes edges/textures.
- **Anisotropic / TV**: method noise contains curvature → preserves straight edges, blurs curved ones.
- **Neighborhood filters** (Yaroslavsky, SUSAN, bilateral): use single-pixel intensity similarity → noise-sensitive.

All these have *local-differential-operator* origins; their method noise *contains structure* (edges, textures) — that's what they erase.

---

### Part III: §3 NL-means Algorithm / NL-means 알고리즘

#### 한국어
**핵심 식**:
$$
NL[v](i) = \sum_{j \in I} w(i,j) v(j), \quad w(i,j) = \frac{1}{Z(i)} \exp\left(-\frac{\|v(\mathcal N_i) - v(\mathcal N_j)\|^2_{2,a}}{h^2}\right)
$$
- $\mathcal N_k$: pixel $k$ 중심의 정사각 patch (보통 $7\times 7$).
- $\|v(\mathcal N_i) - v(\mathcal N_j)\|^2_{2,a}$: 가우시안 가중 $L^2$ patch 거리. $a$는 patch 내 가우시안 분산 (보통 patch 반지름 정도).
- $h$: 필터링 강도. 잡음 std $\sigma$의 ~10배 권장.
- $Z(i) = \sum_j \exp(-\ldots)$: 정규화 (가중치 합 1).
- $w(i, i)$는 자기 자신과의 거리(=0)에서 $\exp(0) = 1$ → 가장 큰 가중치 (실제 구현에선 $w(i,i)$를 따로 처리해 다른 patch들의 평균에 너무 큰 영향 주는 것을 방지).

**핵심 사실** (Buades+ 다른 논문 [2]):
$$
E\|v(\mathcal N_i) - v(\mathcal N_j)\|^2_{2,a} = \|u(\mathcal N_i) - u(\mathcal N_j)\|^2_{2,a} + 2\sigma^2
$$
즉 잡음이 있어도 patch 거리의 *순서*가 보존됨 → NL-means가 잡음에 강건.

#### English
The NL-means estimate at pixel $i$ is a weighted average over the entire image, with weights based on patch similarity in a $L^2$-Gaussian-weighted sense. Patch comparison (vs. single-pixel) makes the algorithm robust to noise. The expected patch distance equals the noiseless distance plus $2\sigma^2$, preserving the ordering of similarities.

---

### Part IV: §4 Consistency Theorem / 일관성 정리

#### 한국어 — Theorem 4 (Conditional Expectation)

$V$가 stationary mixing field, $NL_n$을 $\{V(i), V(\mathcal N_i \setminus \{i\})\}_{i=1}^n$에 적용한 NL-means라면:
$$
|NL_n(j) - r(j)| \to 0 \quad a.s.
$$
$r(i) = E[Y_i | X_i = v(\mathcal N_i \setminus \{i\})]$. 즉 NL-means는 *조건부 기댓값* 추정량으로 수렴.

**Theorem 5**: $V = U + N$, $N$ 독립 잡음 이라면 (i) $E[V(i)|X_i = x] = E[U(i)|X_i = x]$, (ii) 이 조건부 기댓값이 $E[(U(i) - g(V(\mathcal N_i\setminus\{i\})))^2]$을 최소화 → MSE 최적.

#### English
NL-means converges (a.s.) to the conditional expectation $E[Y_i | V(\mathcal N_i)]$, which under independence of signal and noise minimises MSE — so NL-means is *MSE-optimal* in the limit of infinite redundancy.

---

### Part V: §5 Experiments / 실험

#### 한국어
**구현 매개변수**: search window $S \times S = 21 \times 21$ (full-image 대신 효율 위해), patch $7 \times 7$, $h = 10\sigma$. 복잡도 $\sim 49 \times 441 \times N^2$ per image.

**Method noise 비교 (Fig. 4)**: Gaussian/anisotropic/TV/neighborhood 모두 method noise에 *영상 구조*가 보임 → 알고리즘이 그 구조를 흐림. NL-means는 method noise가 *거의 white noise*.

**MSE 비교 (Table 1)**, $\sigma = 20$:
| Image | GF | AF | TVF | YNF | NL |
|---|---|---|---|---|---|
| Lena | 120 | 114 | 110 | 129 | **68** |
| Baboon | 507 | 418 | 365 | 381 | **292** |

NL-means가 모든 경쟁자보다 ~30-50% 좋음.

**시각 (Fig. 5)**: NL-means는 texture와 edge 모두 잘 보존. Gaussian/anisotropic은 texture 흐림, TV는 staircase artifact.

#### English
With 21×21 search and 7×7 patch, NL-means halves the MSE of competing local methods on Lena and Baboon at $\sigma=20$ (Table 1). Visually, NL-means preserves texture and edges far better. Method noise (Fig. 4) confirms NL-means is the only filter whose residual contains no visible structure.

---

## 3. Key Takeaways / 핵심 시사점

1. **Method noise는 알고리즘 비교의 보편적 도구 / Method noise is a universal diagnostic** — 이 논문이 처음 도입. $u - D_h u$에 구조가 보이면 $D_h$가 그 구조를 파괴 중. paper에서 Gaussian/anisotropic/TV/Yaroslavsky의 method noise 식을 정리적으로 도출, NL-means만 *구조 없음* 시각 확인.
   The paper introduces method noise as a universal diagnostic: any structure visible in the residual reveals what the algorithm wrongly smooths.

2. **자기 유사성이 자연 영상의 핵심 / Self-similarity is the dominant prior of natural images** — 직선·곡선·flat·texture 모두 영상 내 *비국소* 어딘가에 *유사 patch*가 있음. NL-means는 이 redundancy를 직접 활용. 이 통찰이 BM3D, Restormer, SwinIR 등 후속 모든 기법의 토대.
   Natural images contain rich non-local redundancy; NL-means is the first algorithm to exploit it explicitly. This insight underpins BM3D and modern transformer-based denoisers.

3. **Patch가 pixel보다 강건 / Patch matching is noise-robust** — 단일 픽셀 비교는 잡음 $\sigma$에 직접 노출. Patch $7\times 7 = 49$차원 비교는 $\sigma$의 $\sqrt{49} = 7$배 큰 신호 컨텍스트 → 잡음 강건. $E\|\Delta v\|^2 = \|\Delta u\|^2 + 2\sigma^2$로 잡음 항이 거리 비교에 *균일*하게 더해져 *순서*가 보존됨.
   Comparing 49-dim patches is far more robust than single-pixel comparison; the noise contribution $2\sigma^2$ is a constant added to all distances, preserving ordering.

4. **국소 미분 연산자는 본질적 한계 / Local differential operators have fundamental limits** — Theorems 1-3: Gaussian/anisotropic/TV의 method noise는 모두 $\Delta u, |Du|\mathrm{curv}, \mathrm{curv}$ 같은 *국소 미분*. 따라서 edge·curve·texture에서 항상 method noise 큼 → 항상 흐림. NL-means는 비미분, 비국소 → 이 한계를 우회.
   Local differential filters always fail on edges/textures because their method noise is itself a derivative; non-local averaging escapes this.

5. **MSE 최적성 / MSE optimality** — Theorem 4-5: stationarity 가정 하에 NL-means가 *조건부 기댓값* $E[U_i | V(\mathcal N_i)]$에 수렴. 이는 MSE를 최소화하는 *Bayes optimal* 추정량. 즉 영상이 충분히 reduundant하면 NL-means는 최적.
   Under stationarity, NL-means converges to the Bayes-optimal conditional expectation — *the* MSE-optimal estimator.

6. **$h \approx 10\sigma$가 경험 최적 / $h \approx 10\sigma$ is empirically optimal** — Patch 거리 $\|v(\mathcal N_i) - v(\mathcal N_j)\|^2_{2,a}$는 $\sim 2\sigma^2$ 배경. $h^2 \sim (10\sigma)^2 = 100\sigma^2$ → exponent 인자가 잡음 vs 신호 차이 비율 $\sim 50$배. 너무 작은 $h$는 매칭 patch 못 찾고, 너무 크면 모든 patch가 비슷해 보임. 10× 가 sweet spot.
   Filtering parameter $h \approx 10\sigma$ balances selectivity (small $h$: few matches) vs. averaging (large $h$: everything averaged). The exponent kernel becomes a soft top-$k$ selector.

7. **Search window는 효율과 redundancy의 trade-off / Search window trades efficiency for redundancy** — 이론은 *영상 전체* 검색을 요구. 실용적 $21\times 21$ search window는 $O(N^2)$ → $O(S^2 N)$로 비용 감소 (10000× 가속). 작은 영상에선 거의 무손실, 큰 영상에선 redundancy를 일부 놓침. 이는 NL-means의 가장 큰 *실용 비용*.
   Restricting search to a 21×21 window cuts cost from $O(N^2)$ to $O(S^2 N)$; the cost is missing some non-local matches in large images, the main practical limitation.

8. **NL-means가 BM3D의 직접 조상 / NL-means is the direct ancestor of BM3D** — paper #7 BM3D는 NL-means의 *block matching* 단계와 wavelet thresholding (paper #1-3)의 *transform-domain shrinkage*를 결합. NL-means의 *patch group* 발견이 BM3D의 핵심 step. Modern transformer denoisers (Restormer, SwinIR)에서 *self-attention*도 NL-means weights의 일반화.
   BM3D combines NL-means' block-matching with wavelet shrinkage; transformer self-attention is a learned generalisation of NL-means weights.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Image model and method noise / 영상 모델과 방법 잡음
$$
v(i) = u(i) + n(i), \quad n \overset{iid}{\sim} N(0, \sigma^2)
$$
$$
\text{Method noise}(D_h, u) = u - D_h u
$$
### 4.2 Theorems 1-3 (limits of local methods)
$$
u - G_h * u = -h^2 \Delta u + o(h^2) \quad \text{(Gaussian, Thm 1)}
$$
$$
u - AF_h u = -\tfrac{1}{2} h^2 |Du|\,\mathrm{curv}(u) + o(h^2) \quad \text{(anisotropic, Thm 2)}
$$
$$
u - TVF_\lambda u = -\frac{\mathrm{curv}(TVF_\lambda u)}{2\lambda} \quad \text{(TV, Thm 3)}
$$
### 4.3 NL-means (the key algorithm) / NL-means

$$
\boxed{\;NL[v](i) = \sum_j w(i,j)\,v(j), \quad w(i,j) = \frac{1}{Z(i)}\exp\left(-\frac{\|v(\mathcal N_i) - v(\mathcal N_j)\|^2_{2,a}}{h^2}\right)\;}
$$
$$
Z(i) = \sum_j \exp\left(-\frac{\|v(\mathcal N_i) - v(\mathcal N_j)\|^2_{2,a}}{h^2}\right)
$$
$$
\|v(\mathcal N_i) - v(\mathcal N_j)\|^2_{2,a} = \sum_{k} g_a(k) (v(i+k) - v(j+k))^2
$$
where $g_a$ is a Gaussian kernel of variance $a$ over the patch offsets $k$.

### 4.4 Patch distance and noise robustness / Patch 거리와 잡음 강건성
$$
E\|v(\mathcal N_i) - v(\mathcal N_j)\|^2_{2,a} = \|u(\mathcal N_i) - u(\mathcal N_j)\|^2_{2,a} + 2\sigma^2
$$
### 4.5 Algorithm parameters / 알고리즘 파라미터
| Parameter | Typical value | Role |
|---|---|---|
| Patch size | $7 \times 7$ ($P=3$ radius) | Larger = more context, less noise sensitivity |
| Search window | $21 \times 21$ ($S=10$ radius) | Larger = more matches, slower |
| Filter strength $h$ | $\approx 10\sigma$ | Larger = more averaging, more blur |
| Patch Gaussian $a$ | $\approx P/2$ | Center weighting |

### 4.6 Algorithm pseudocode / 알고리즘 의사코드
```
For each pixel i in image:
    Z = 0
    accumulator = 0
    For each j in search window centered at i:
        d = || v(N_i) - v(N_j) ||^2_{2,a}
        w = exp(-d / h^2)
        Z += w
        accumulator += w * v(j)
    NL_v[i] = accumulator / Z
```
Cost: $O(N \cdot S^2 \cdot P^2)$ for image of size $N$, search $S\times S$, patch $P\times P$.

### 4.7 Consistency (Theorems 4-5)
Under stationarity:
$$
NL_n(j) \xrightarrow{a.s.} r(j) = E[Y_j | X_j = v(\mathcal N_j \setminus \{j\})]
$$
For $V = U + N$ with independent noise:
$$
E[V_i | X_i] = E[U_i | X_i] = \arg\min_g E[(U_i - g(V(\mathcal N_i)))^2]
$$
### 4.8 Worked numerical example / 수치 예시
Lena, $\sigma = 20$, Table 1: NL-means MSE = 68; nearest competitor (TV) MSE = 110. Ratio 0.62 → NL-means PSNR is about $10\log_{10}(110/68) \approx 2.1$ dB *better*. Strong margin.

For typical patch ($P=3$, $7\times 7 = 49$ pixels): noise contribution to patch distance is $2\sigma^2 \cdot 49 = 39200$ at $\sigma=20$. Signal patch distance for matching patches is much smaller; for non-matching, much larger. Threshold $h^2 = 100\sigma^2 = 40000$ → matching patches get $w \approx \exp(-1)$, non-matching $w \approx 0$.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1960 ─── Gabor — Gaussian smoothing
1985 ─── Yaroslavsky — neighborhood filtering
1990 ─── Perona-Malik — anisotropic diffusion
1992 ─── Rudin-Osher-Fatemi — total variation regularisation
1994 ─── Donoho-Johnstone — VisuShrink (paper #1, transform domain)
1995 ─── SUSAN filter (Smith-Brady), Bilateral filter (Tomasi-Manduchi 1998)
                            ↳ all neighborhood-style, single-pixel comparison
1999 ─── Efros-Leung — texture synthesis by non-parametric sampling
                            ↳ key idea: copy patches from elsewhere in image
2000 ─── Chang-Yu-Vetterli — BayesShrink (paper #3)
2003 ─── Ordentlich+ — discrete universal denoiser (DUDE)
                            ↳ also non-local in spirit
2005 ★★ BUADES-COLL-MOREL — Non-Local Means (THIS PAPER)
                            ↳ first explicit non-local averaging for denoising
                            ↳ method-noise diagnostic
2006 ─── Mairal-Sapiro-Elad — K-SVD dictionary learning (sparse codes)
2007 ─── Dabov+ — BM3D (paper #7)
                            ↳ block-matching (NL-means) + 3D transform (paper #1-3)
2009 ─── Mairal-Bach-Ponce-Sapiro — non-local sparse models
2012+ ── Deep learning era: DnCNN (2017), Restormer (2022)
                            ↳ self-attention is learned NL-means weights
```

이 논문은 **non-local 시대를 연 결정적 분기점**. 모든 후속 patch-based 방법 (BM3D, K-SVD non-local, transformer denoiser)이 NL-means의 직접 후예.

This paper opens the **non-local era** of image denoising. Every subsequent patch-based method (BM3D, K-SVD non-local, transformer-based denoisers) is a descendant.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Yaroslavsky (1985)** | Neighborhood filter | NL-means' immediate predecessor; uses pixel intensity instead of patch similarity. NL-means' patch-based weighting is the key generalisation. |
| **Perona-Malik (1990)** *PAMI* | Anisotropic diffusion | Theorem 2 derives its method noise; shown to over-smooth curved edges. |
| **Rudin-Osher-Fatemi (1992)** | Total variation | Theorem 3 derives its method noise; explains TV's staircasing. |
| **Tomasi-Manduchi (1998)** | Bilateral filter | Same neighborhood-filter family as Yaroslavsky; pixel-intensity weighted. NL-means generalises to patch-intensity. |
| **Efros-Leung (1999)** *ICCV* | Patch-based texture synthesis | Inspired NL-means' patch comparison; both rely on natural image self-similarity. |
| **Donoho-Johnstone (1994)** *Biometrika* (paper #1) | VisuShrink | Different paradigm (transform-domain shrinkage); paper #7 BM3D unifies them. |
| **Chang-Yu-Vetterli (2000)** *IEEE TIP* (paper #3) | BayesShrink | Subband-adaptive Bayesian threshold; complementary transform-domain approach. |
| **Mairal-Sapiro-Elad (2006)** | K-SVD denoising | Dictionary-based sparse coding; another patch-based approach. |
| **Dabov+ (2007)** *IEEE TIP* (paper #7) | BM3D | Direct successor; combines NL-means' block-matching with wavelet thresholding. |
| **Vaswani+ (2017)** "Attention is All You Need" | Transformer self-attention | Soft attention with learned similarity weights = learned NL-means; modern denoisers (Restormer, SwinIR) directly inherit. |

---

## 7. References / 참고문헌

- Alvarez, L., Lions, P.-L., & Morel, J.-M., "Image selective smoothing and edge detection by nonlinear diffusion (II)", *J. Numerical Analysis*, 29, 845–866 (1992).
- Buades, A., Coll, B., & Morel, J.-M., "A non-local algorithm for image denoising", *Proc. IEEE CVPR*, 2, 60–65 (2005). [DOI: 10.1109/CVPR.2005.38]
- Buades, A., Coll, B., & Morel, J.-M., "On image denoising methods", *CMLA Tech Report 2004-15* (2004).
- Donoho, D. L., "De-noising by soft-thresholding", *IEEE TIT*, 41, 613–627 (1995).
- Efros, A., & Leung, T., "Texture synthesis by non-parametric sampling", *Proc. ICCV*, 2, 1033–1038 (1999).
- Perona, P., & Malik, J., "Scale space and edge detection using anisotropic diffusion", *IEEE PAMI*, 12, 629–639 (1990).
- Rudin, L., Osher, S., & Fatemi, E., "Nonlinear total variation based noise removal algorithms", *Physica D*, 60, 259–268 (1992).
- Smith, S., & Brady, J., "SUSAN — A new approach to low level image processing", *IJCV*, 23(1), 45–78 (1997).
- Tomasi, C., & Manduchi, R., "Bilateral filtering for gray and color images", *Proc. ICCV*, 839–846 (1998).
- Yaroslavsky, L., *Digital Picture Processing — An Introduction*, Springer (1985).
