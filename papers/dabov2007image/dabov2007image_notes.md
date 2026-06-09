---
title: "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering"
authors: Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, Karen Egiazarian
year: 2007
journal: "IEEE Transactions on Image Processing 16(8), pp. 2080–2095"
doi: "10.1109/TIP.2007.901238"
topic: Low-SNR Imaging / Block-Matching + Transform-Domain Denoising
tags: [bm3d, block-matching, collaborative-filtering, 3d-transform, hard-thresholding, wiener-filtering, aggregation, two-step, dabov-foi-katkovnik-egiazarian]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 7. Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering / 희소 3-D 변환영역 협력 필터링에 의한 영상 잡음 제거

---

## 1. Core Contribution / 핵심 기여

### 한국어
**BM3D**는 NLM (paper #4)의 block-matching과 wavelet thresholding (paper #1-3)의 transform-domain shrinkage를 결합한 **2-단계 알고리즘**으로, 2007 시점 image denoising의 *state-of-the-art*. Lena 512×512, σ=25에서 **32.08 dB** PSNR — 이전 모든 기법 (BLS-GSM, K-SVD, exemplar-based, NLM)을 +0.5-1.5 dB로 능가.

**핵심 아이디어** (Fig. 3 알고리즘 흐름):
- **Step 1 (basic estimate via hard thresholding)**:
  1. **Grouping (block matching)**: 각 reference block $Z_{x_R}$에 대해 $N_S \times N_S$ search window 내에서 $L^2$ 거리 $d(Z_{x_R}, Z_x) < \tau^{ht}_{\text{match}}$ 만족하는 *유사 블록*들을 stacking → 3-D 그룹 $\mathbf Z_{S_{x_R}^{ht}}$.
  2. **Collaborative hard-thresholding**: 3-D 분리형 변환 (2-D Bior1.5 + 1-D Haar) → hard threshold $\lambda_{3D} \sigma$ → 역변환 → block-wise 추정 $\hat{\mathbf Y}^{ht}$.
  3. **Aggregation**: 모든 reference block에 대해 반복 수행 → 같은 픽셀에 다수 추정값 → 분산 역수 가중 평균 (weight = 1/(σ² × #retained coefficients)).

- **Step 2 (final estimate via empirical Wiener filtering)**:
  4. **Re-grouping**: 잡음이 줄어든 *basic estimate*로 BM 재수행 (더 정확).
  5. **Collaborative Wiener filtering**: basic estimate를 pilot signal로 → Wiener shrinkage 계수 $W = |\mathcal T_{3D}(\hat Y^{basic})|^2 / (|\mathcal T_{3D}(\hat Y^{basic})|^2 + \sigma^2)$ → 잡음 영상의 3-D 변환 계수에 element-wise 곱.
  6. **Aggregation**: weight = σ⁻² × $\|W\|_2^{-2}$.

**왜 동작하는가**: (i) 자연 영상은 *self-similarity* 풍부 → 3-D 그룹이 매우 sparse한 3-D 변환 표현 가짐, (ii) 3-D 변환은 *intra-fragment* 상관 (NLM의 patch 비교) + *inter-fragment* 상관 (block 사이 유사성)을 동시에 활용, (iii) hard threshold 단계는 잡음이 큰 영역에서도 robust한 basic estimate 제공, (iv) Wiener 단계는 그 basic estimate를 pilot으로 *최적 shrinkage*.

### English
**BM3D** combines **block-matching** (from paper #4, NLM) with **transform-domain shrinkage** (from papers #1-3) in a **two-step algorithm**, achieving state-of-the-art denoising as of 2007 (Lena σ=25 → 32.08 dB, beating all priors by 0.5-1.5 dB):

- **Step 1**: Block-match → form 3-D groups of similar patches → 3-D transform → hard-threshold → inverse 3-D transform → aggregate. Yields basic estimate.
- **Step 2**: Re-block-match on basic estimate → 3-D transform of both basic and noisy groups → Wiener shrinkage using basic as pilot → inverse → aggregate. Yields final estimate.

**Why it works**: Natural images have strong self-similarity → 3-D groups (stacked similar patches) are highly sparse in the 3-D transform domain, exploiting both intra-patch and inter-patch correlations. Two-step refinement uses the basic estimate to improve both grouping accuracy and Wiener pilot.

---

## 2. Reading Notes / 읽기 노트

### Part I: §I-II Overview and Conceptual Framework / 개요와 개념 프레임워크

#### 한국어 — §II.A-D Grouping and Collaborative Filtering

- **Grouping** (paper와 일반론): partitioning (K-means, vector quantization) vs **matching**. Matching은 disjoint를 요구 안 함 — 같은 패치가 여러 그룹에 속할 수 있음 → 더 풍부한 redundancy.
- **Block matching (BM)**: video compression의 motion estimation에서 차용. $d$-차원 fragment를 $d+1$-차원 group에 stack.
- **Collaborative filtering**: 그룹 내 $n$ 조각으로부터 *각각의* 추정값 생성 (단순 평균이 아님). 핵심: 그룹 내 *다른* 조각들이 *각* 조각의 잡음 제거에 기여.
- **Why transform-domain shrinkage?** Natural-image patches는 *intra* 상관 (각 patch의 공간 구조) + *inter* 상관 (group 내 patch들의 유사성) 둘 다 가짐. 2-D 변환은 *intra*만, 1-D 변환 along stack direction은 *inter*만 활용. 분리형 3-D 변환 = 2-D × 1-D 으로 둘 다 활용.

#### English — §II.A-D
Grouping by matching (vs partitioning) lets the same patch be in multiple groups. Collaborative filtering produces an estimate *for each grouped patch*, not a single average. Separable 3-D transform exploits intra-patch + inter-patch correlations simultaneously.

---

### Part II: §III Algorithm — The two-step BM3D / 알고리즘

#### 한국어 — Observation model
$$
z(x) = y(x) + \eta(x), \quad \eta \overset{iid}{\sim} N(0, \sigma^2)
$$
- $Z_x$: $z$에서 위치 $x$ 기준 $N_1 \times N_1$ block.
- $\mathbf Z_S$: 3-D array, $S$ 좌표들의 block stacking.

#### 한국어 — §III.A.1 Steps 1ai/1aii: Hard-thresholding step

**Block-distance** (Eq. 4):
$$
d(Z_{x_R}, Z_x) = \frac{\bigl\|\Upsilon\bigl(\mathcal T^{ht}_{2D}(Z_{x_R})\bigr) - \Upsilon\bigl(\mathcal T^{ht}_{2D}(Z_x)\bigr)\bigr\|^2_2}{(N^{ht}_1)^2}
$$
여기서 $\Upsilon$는 hard-threshold 연산자 ($\lambda_{2D} \sigma$). 즉 *prefilter된 변환 계수의 $L^2$ 거리*. 잡음 영향 줄임.

**Group set**:
$$
S^{ht}_{x_R} = \{x \in X: d(Z_{x_R}, Z_x) \le \tau^{ht}_{\text{match}}\} \quad (5)
$$
**3-D 변환 + Hard threshold + 역변환** (Eq. 6):
$$
\hat{\mathbf Y}^{ht}_{S^{ht}_{x_R}} = \mathcal T^{ht-1}_{3D}\bigl(\Upsilon\bigl(\mathcal T^{ht}_{3D}(\mathbf Z_{S^{ht}_{x_R}})\bigr)\bigr)
$$
$\mathcal T^{ht}_{3D} = \mathcal T^{ht}_{2D} \otimes \mathcal T_{1D}$ (분리형), 보통 2-D Bior1.5 (paper #1-3과 호환) + 1-D Haar.

**Aggregation weights** (Eq. 10):
$$
w^{ht}_{x_R} = \begin{cases} 1/(\sigma^2 N^{x_R}_{\text{har}}) & N^{x_R}_{\text{har}} \ge 1 \\ 1 & \text{otherwise} \end{cases}
$$
$N^{x_R}_{\text{har}}$ = retained (nonzero) hard-threshold 계수 수. *Sparser* group → *fewer* retained coefs → *higher* weight (less variance in estimate).

#### 한국어 — §III.A.2 Steps 2ai/2aii: Wiener filtering step

**Re-grouping** (Eq. 7) using basic estimate:
$$
S^{wie}_{x_R} = \left\{x \in X : \frac{\|\hat Y^{basic}_{x_R} - \hat Y^{basic}_x\|^2_2}{(N^{wie}_1)^2} < \tau^{wie}_{\text{match}}\right\}
$$
$\hat Y^{basic}$이 잡음 적어 grouping이 더 정확.

**Two groups**:
- $\hat{\mathbf Y}^{basic}_{S^{wie}_{x_R}}$: basic estimate에서 stack
- $\mathbf Z_{S^{wie}_{x_R}}$: noisy image에서 stack (같은 좌표)

**Wiener shrinkage coefficients** (Eq. 8):
$$
\mathbf W_{S^{wie}_{x_R}} = \frac{|\mathcal T^{wie}_{3D}(\hat{\mathbf Y}^{basic}_{S^{wie}_{x_R}})|^2}{|\mathcal T^{wie}_{3D}(\hat{\mathbf Y}^{basic}_{S^{wie}_{x_R}})|^2 + \sigma^2}
$$
이는 표준 empirical Wiener — basic estimate가 *pilot* (true energy의 추정).

**Filtering** (Eq. 9):
$$
\hat{\mathbf Y}^{wie}_{S^{wie}_{x_R}} = \mathcal T^{wie-1}_{3D}\bigl(\mathbf W_{S^{wie}_{x_R}} \cdot \mathcal T^{wie}_{3D}(\mathbf Z_{S^{wie}_{x_R}})\bigr)
$$
**Wiener weights** (Eq. 11):
$$
w^{wie}_{x_R} = \sigma^{-2} \|\mathbf W_{S^{wie}_{x_R}}\|^{-2}_2
$$
#### English — §III Algorithm
Step 1 (basic): block-match in noisy image (with prefilter), 3-D transform, hard-threshold, inverse, aggregate. Step 2 (final): re-block-match using basic estimate, form basic+noisy groups, Wiener-shrink noisy using basic as pilot, inverse, aggregate.

---

### Part III: §IV Fast and efficient realization / 빠른 구현

#### 한국어 — Reductions
1. **Step size** $N_{\text{step}} > 1$ (보통 3): reference block 수를 $|X|/N_{\text{step}}^2$로 줄임.
2. **Group size cap** $N_2$ (보통 16-32): 거리 작은 상위 $N_2$개만.
3. **Predictive search BM**: 이전 BM 결과 근방에서만 small $N_{PR} \times N_{PR}$ (보통 5×5) 검색, periodic하게 $N_S \times N_S$ full 검색.
4. **Separable transforms**: 2-D = 1-D ⊗ 1-D, 3-D = 2-D ⊗ 1-D.
5. **Precompute** all sliding-block 2-D transforms once (각 reference block 처리시 재사용).
6. **Kaiser window** in aggregation: 경계 효과 (특히 2-D DCT) 줄임.

**Complexity**: $O(|X|)$ (모든 파라미터 fixed). 256×256 영상 σ ≤ 40 normal profile: ~4 sec on 2007 hardware.

#### English — §IV
Sliding step $>1$, group size cap $N_2$, predictive-search BM, separable transforms, pre-computed sliding 2-D transforms, Kaiser window. Total complexity $O(|X|)$ — linear in image size.

---

### Part IV: §V-VI Color extension and Experiments / 컬러 확장과 실험

#### 한국어 — §V Color BM3D (CBM3D)
RGB → opponent color space (luminance + chrominance). Block-matching은 *luminance only* (chrominance는 잡음에 약함). 그룹화 후 모든 채널에 같은 BM 결과 적용 → collaborative filter 각 채널에 따로.

#### 한국어 — §VI Experiments

**Table III** (PSNR, dB), 6 grayscale test images, σ ∈ {2, 5, 10, 15, 20, 25, 30, 35, 50, 75, 100}:

| σ | C.man | House | Peppers | Lena | Barbara | Boats |
|---|---|---|---|---|---|---|
| 5 | 38.29 | 39.83 | 38.12 | 38.72 | 38.31 | 37.28 |
| 10 | 34.18 | 36.71 | 34.68 | 35.93 | 34.98 | 33.92 |
| 15 | 31.91 | 34.94 | 32.70 | 34.27 | 33.11 | 32.14 |
| 20 | 30.48 | 33.77 | 31.29 | 33.05 | 31.78 | 30.88 |
| 25 | 29.45 | 32.86 | 30.16 | 32.08 | 30.72 | 29.91 |
| 50 | 25.84 | 29.37 | 26.41 | 28.86 | 27.17 | 26.64 |

**Fig. 4 비교 baselines** (BM3D, FSP+TUP BLS-GSM, BLS-GSM, exemplar-based, K-SVD, pointwise SA-DCT):
- BM3D는 모든 σ ∈ [10, 25]에서 *모든* 영상에서 +0.3-1.5 dB 우수.

**Table II — transform sensitivity** (Lena, σ=25):
- $\mathcal T^{ht}_{2D}$ 선택 (Haar, Db, Bior1.x, WHT, DCT, DST): 31.93-32.08 dB로 0.15 dB 변동 → 거의 영향 없음.
- $\mathcal T_{1D}$ 선택: Haar 32.08 vs DC-only 30.65 vs DC+rand 31.88 → **1-D 변환이 그룹 내 inter-fragment 상관 활용에 결정적**. DC만으로는 부족, full Haar가 최적.

#### English — §VI
BM3D outperforms all 2007-era baselines (BLS-GSM, K-SVD, NLM-exemplar, SA-DCT) by 0.3-1.5 dB across noise levels and images. Performance is robust to 2-D transform choice (DCT/Haar/Bior all within 0.15 dB) but the 1-D transform along stack direction matters: Haar far outperforms DC-only.

---

## 3. Key Takeaways / 핵심 시사점

1. **Block matching + transform shrinkage의 결합이 핵심 / Joining BM with transform shrinkage is the key** — NLM (paper #4)은 *spatial-domain averaging*만, wavelet shrinkage (paper #1-3)는 *single-image transform*만 사용. BM3D는 둘을 결합: BM으로 그룹을 찾고, *그룹 내* 3-D 변환에서 shrinkage. 자연 영상 self-similarity + 변환 sparsity 둘 다 활용.
   BM3D combines block-matching (NLM heritage) with transform-domain shrinkage (wavelet heritage); the synergy of these two complementary approaches is what unlocks SOTA performance.

2. **Two-step refinement이 통계적 우월성 / Two-step refinement is statistically superior** — Step 1 hard-threshold는 robust but biased basic estimate. Step 2 Wiener는 그 basic을 *pilot signal*로 써서 *최적 shrinkage* 수행. Wiener는 진짜 신호 분산 $\|Y\|^2$에 의존하는데, basic estimate가 그 추정을 제공.
   Step 1 gives a robust basic estimate; Step 2 uses it as the Wiener pilot, achieving statistical optimality through MMSE shrinkage.

3. **3-D 변환의 *inter-fragment* correlation이 본질 / Inter-fragment correlation is essential** — Table II의 핵심 발견: 2-D 변환 종류는 거의 무관 (0.15 dB 변동), 1-D 변환은 결정적 (DC-only vs Haar +1.5 dB 차이). 이는 *그룹 내 patch들 사이의* 유사성이 sparsity의 진짜 원천임을 의미.
   The crucial sparsity in 3-D groups comes from *inter-fragment* correlation (similarity between stacked patches), not intra-fragment. This is why 1-D transform along stack direction matters far more than the 2-D transform choice.

4. **Aggregation weight ∝ 1/variance / Variance-inverse aggregation** — Sparser group = fewer retained transform coefficients = lower estimation variance → higher weight in averaging. 이는 통계적으로 *최적*: independent 추정자들의 inverse-variance weighted average is BLUE (best linear unbiased estimator).
   Inverse-variance aggregation weights (∝ 1/σ²·N_har) are the statistically optimal way to combine independent block-wise estimates.

5. **Block matching robustness / 블록 매칭의 강건성** — Hard-threshold pre-filter (Eq. 4)가 잡음의 distance metric 영향 줄임. $\sigma$가 클 때 block matching이 깨질 수 있는데, prefiltered distance가 이를 완화. Step 2에서는 더 깨끗한 basic estimate로 더 정확한 BM.
   The hard-threshold prefilter in the distance metric (Eq. 4) preserves block-matching accuracy at high noise; Step 2's basic-estimate-based BM is even more reliable.

6. **$O(|X|)$ 복잡도 / $O(|X|)$ complexity** — 모든 파라미터 fixed (search window, group size, etc.). NLM처럼 $O(N^2)$가 아닌 *linear*. Practical: 256×256 4 sec, 512×512 ~16 sec on 2007 hardware. 오늘날에는 < 1 sec.
   With fixed parameters, BM3D runs in linear time — practical for real-time use today.

7. **2007 SOTA / 2007 state-of-the-art** — Table III와 Fig. 4: BLS-GSM, K-SVD, NLM-exemplar, SA-DCT를 모두 능가. 결과 모든 영상·σ에서 *우수* — *우월성이 robust*. 이후 10년간 BM3D는 *학습 없는* (non-learned) 방법의 표준.
   BM3D dominated all non-learned denoising approaches from 2007 to ~2017 when deep learning (DnCNN, FFDNet) finally beat it. Even today, BM3D remains a classic baseline.

8. **Deep learning이 BM3D를 *겨우* 추월 / Deep learning only narrowly surpasses BM3D** — DnCNN (2017) vs BM3D Lena σ=25: ~32.43 dB vs 32.08 dB (+0.35 dB). 딥러닝의 추가 PSNR 이득은 작음. 실용적으로는 *no-training* 장점이 BM3D를 여전히 가치 있게 함 (생체 의료·천문 등 학습 데이터 부족 분야).
   Deep learning beats BM3D by only ~0.35 dB; in domains lacking training data (medical, astrophysical), BM3D's training-free property keeps it relevant.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Observation model
$$
z(x) = y(x) + \eta(x), \quad \eta \sim \mathcal N(0, \sigma^2 I)
$$
2-D blocks: $Z_x \in \mathbb R^{N_1 \times N_1}$. 3-D group: $\mathbf Z_S$ of shape $N_1 \times N_1 \times |S|$.

### 4.2 Step 1 — Basic estimate (hard thresholding)
**Block matching** (Eq. 4-5):
$$
d(Z_{x_R}, Z_x) = \frac{\|\Upsilon(\mathcal T^{ht}_{2D} Z_{x_R}) - \Upsilon(\mathcal T^{ht}_{2D} Z_x)\|^2}{N_1^2}, \quad S^{ht}_{x_R} = \{x: d \le \tau^{ht}_{\text{match}}\}
$$
**Collaborative hard thresholding** (Eq. 6):
$$
\hat{\mathbf Y}^{ht}_{S} = \mathcal T^{ht}_{3D}{}^{-1}\bigl(\Upsilon\bigl(\mathcal T^{ht}_{3D} \mathbf Z_S\bigr)\bigr), \quad \Upsilon(c) = c \cdot \mathbf 1\{|c| > \lambda_{3D}\sigma\}
$$
**Aggregation weight** (Eq. 10):
$$
w^{ht}_{x_R} = 1/(\sigma^2 N^{x_R}_{\text{har}}) \quad \text{if } N_{\text{har}} \ge 1, \text{ else } 1
$$
**Aggregation** (Eq. 12):
$$
\hat y^{basic}(x) = \frac{\sum_{x_R} \sum_{x_m \in S^{ht}_{x_R}} w^{ht}_{x_R} \hat Y^{ht, x_R}_{x_m}(x)}{\sum_{x_R} \sum_{x_m} w^{ht}_{x_R} \chi_{x_m}(x)}
$$
### 4.3 Step 2 — Final estimate (Wiener filtering)

**Wiener block matching** (Eq. 7):
$$
S^{wie}_{x_R} = \{x: \|\hat Y^{basic}_{x_R} - \hat Y^{basic}_x\|^2 / N_1^2 < \tau^{wie}_{\text{match}}\}
$$
**Wiener shrinkage** (Eq. 8-9):
$$
\mathbf W_S = \frac{|\mathcal T^{wie}_{3D} \hat{\mathbf Y}^{basic}_S|^2}{|\mathcal T^{wie}_{3D} \hat{\mathbf Y}^{basic}_S|^2 + \sigma^2}, \quad \hat{\mathbf Y}^{wie}_S = \mathcal T^{wie}_{3D}{}^{-1}(\mathbf W_S \cdot \mathcal T^{wie}_{3D} \mathbf Z_S)
$$
**Wiener weight** (Eq. 11):
$$
w^{wie}_{x_R} = 1/(\sigma^2 \|\mathbf W_S\|^2_2)
$$
### 4.4 Default parameters (Normal Profile)

| Step | $\mathcal T_{2D}$ | $\mathcal T_{1D}$ | $N_1$ | $N_2$ | $N_{\text{step}}$ | $N_S$ | $\lambda$ or $\tau$ |
|---|---|---|---|---|---|---|---|
| 1 (HT) | Bior1.5 | Haar | 8 (12 if σ>40) | 16 | 3 (4) | 39 | $\lambda_{3D} = 2.7$ (2.8), $\tau^{ht}=2500$ (5000) |
| 2 (Wiener) | DCT | Haar | 8 (11) | 32 | 3 (6) | 39 | $\tau^{wie} = 400$ (3500) |

### 4.5 BM3D algorithm (pseudocode)
```
def bm3d(z, sigma):
    # Step 1: hard thresholding
    y_basic = zeros_like(z); weight_buffer = zeros_like(z)
    for x_R in reference_block_locations:
        # 1. Block matching
        S = find_similar_blocks_hard(z, x_R, threshold=tau_ht)
        # 2. Form 3D group, apply 3D transform
        Z_S = stack(z[S])  
        c = T_3D(Z_S)
        # 3. Hard threshold
        c_ht = c * (abs(c) > lambda_3D * sigma)
        # 4. Inverse transform
        Y_S_hat = T_3D_inv(c_ht)
        # 5. Aggregate with inverse-variance weights
        N_har = count_nonzero(c_ht)
        w = 1.0 / (sigma**2 * max(N_har, 1))
        for x_block, Y_block in zip(S, Y_S_hat):
            y_basic[x_block] += w * Y_block
            weight_buffer[x_block] += w
    y_basic /= weight_buffer
    
    # Step 2: Wiener filtering
    y_final = zeros_like(z); weight_buffer = zeros_like(z)
    for x_R in reference_block_locations:
        S = find_similar_blocks_wiener(y_basic, x_R, threshold=tau_wie)
        Z_S = stack(z[S]); Y_basic_S = stack(y_basic[S])
        c_basic = T_3D(Y_basic_S); c_noisy = T_3D(Z_S)
        # Wiener shrinkage
        W = abs(c_basic)**2 / (abs(c_basic)**2 + sigma**2)
        Y_S_hat = T_3D_inv(W * c_noisy)
        # Aggregate with Wiener weights
        w = 1.0 / (sigma**2 * sum(W**2))
        for x_block, Y_block in zip(S, Y_S_hat):
            y_final[x_block] += w * Y_block
            weight_buffer[x_block] += w
    y_final /= weight_buffer
    return y_final
```

### 4.6 Worked numerical example / 수치 예시
Lena σ=25, Normal Profile: PSNR 32.08 dB. MSE = $255^2/10^{32.08/10} \approx 40.3$ → average error $\sqrt{40.3} \approx 6.3$ gray levels (cf. noise std 25). MSE reduction ratio: $25^2/40.3 \approx 15.5\times$ — BM3D removes 94% of the noise variance.

For C.man σ=10 → 34.18 dB: MSE $\approx 24.8$, error $\approx 5.0$, MSE reduction $100/24.8 = 4.0\times$ — 75% noise variance removed.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1985 ─── Yaroslavsky — neighborhood filter
1989 ─── Mallat — wavelet MRA
1994 ─── Donoho-Johnstone — VisuShrink (paper #1, transform thresholding)
1995 ─── Donoho-Johnstone — SureShrink (paper #2)
1996 ─── Simoncelli-Adelson — Bayesian wavelet coring
2000 ─── Chang-Yu-Vetterli — BayesShrink (paper #3)
2003 ─── Portilla+ — BLS-GSM (Gaussian-scale-mixture prior)
2005 ─── Buades-Coll-Morel — Non-Local Means (paper #4, spatial-domain)
2006 ─── Aharon-Elad-Bruckstein — K-SVD (learned dictionary)
2006 ─── Foi+ — Pointwise SA-DCT (shape-adaptive DCT)
2007 ★★ DABOV-FOI-KATKOVNIK-EGIAZARIAN — BM3D (THIS PAPER)
                          ↳ block matching + 3-D transform shrinkage
                          ↳ two-step (hard-threshold + Wiener)
                          ↳ +0.5-1.5 dB over all priors
2010 ─── Mairal-Bach-Ponce-Sapiro — non-local sparse models (LSSC)
2012 ─── Maggioni+ — V-BM4D (paper #9, video extension)
2013 ─── Maggioni+ — BM4D (paper #10, volumetric extension)
2017 ─── Zhang+ — DnCNN (deep CNN denoising) — narrowly beats BM3D
2018 ─── Zhang+ — FFDNet (multi-noise CNN)
2022 ─── Zamir+ — Restormer (transformer denoising)
                          ↳ self-attention is essentially learned BM3D weights
```

**위치**: BM3D는 *2007 시점의 결정판*. NLM의 nonlocal idea와 wavelet shrinkage의 transform-domain idea를 결합. 이후 10년간 (~2017까지) 모든 *non-learned* denoising의 표준 baseline. Deep learning에 의해 좁은 차이로 추월되었지만, *training-free* 장점으로 의료·천문 분야에서 여전히 사용.

This paper marks the apex of non-learned image denoising. It bridged the spatial-domain (NLM) and transform-domain (wavelets) traditions, and held the SOTA for ~10 years until deep learning narrowly surpassed it.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Donoho-Johnstone (1994)** *Biometrika* (paper #1) | Hard/soft thresholding | BM3D's Step 1 hard-thresholding directly inherits this framework, lifted to 3-D groups. |
| **Donoho-Johnstone (1995)** *JASA* (paper #2) | SureShrink | BM3D's threshold $\lambda_{3D}\sigma$ is fixed (2.7), simpler than SureShrink's data-driven choice — works because 3-D sparsity is much higher. |
| **Chang-Yu-Vetterli (2000)** *IEEE TIP* (paper #3) | BayesShrink, MDL | Wiener shrinkage in Step 2 has the same Bayesian flavour as BayesShrink. |
| **Buades-Coll-Morel (2005)** *CVPR* (paper #4) | Non-Local Means | The block-matching step inherits NLM's nonlocal patch-similarity idea; BM3D upgrades the spatial averaging to transform-domain shrinkage. |
| **Portilla+ (2003)** *IEEE TIP* | BLS-GSM | Direct competitor in 2007; BM3D beats by 0.3-1 dB. |
| **Aharon-Elad-Bruckstein (2006)** *IEEE TSP* | K-SVD | Learned-dictionary alternative; BM3D is faster (no learning), comparable PSNR. |
| **Foi+ (2007)** | SA-DCT | Shape-adaptive DCT denoising; complementary, BM3D is generally better. |
| **Maggioni+ (2012)** *IEEE TIP* (paper #9) | V-BM4D | Direct video extension: 3-D blocks → 4-D groups with motion-compensated trajectories. |
| **Maggioni+ (2013)** *IEEE TIP* (paper #10) | BM4D | Direct volumetric extension: 3-D cubes instead of 2-D blocks. |
| **Zhang+ (2017)** *IEEE TIP* | DnCNN | Deep CNN that finally surpasses BM3D by ~0.3 dB; uses residual learning. |
| **Restormer (2022), Vaswani+ (2017)** | Self-attention | Self-attention can be viewed as a learned generalisation of BM3D's similarity weights. |

---

## 7. References / 참고문헌

- Aharon, M., Elad, M., & Bruckstein, A., "K-SVD: an algorithm for designing overcomplete dictionaries for sparse representation", *IEEE TSP*, 54(11), 4311–4322 (2006).
- Buades, A., Coll, B., & Morel, J.-M., "A non-local algorithm for image denoising", *Proc. IEEE CVPR*, 2, 60–65 (2005).
- Chang, S. G., Yu, B., & Vetterli, M., "Adaptive wavelet thresholding for image denoising and compression", *IEEE TIP*, 9(9), 1532–1546 (2000).
- Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K., "Image denoising by sparse 3-D transform-domain collaborative filtering", *IEEE TIP*, 16(8), 2080–2095 (2007). [DOI: 10.1109/TIP.2007.901238]
- Donoho, D. L., & Johnstone, I. M., "Ideal spatial adaptation by wavelet shrinkage", *Biometrika*, 81, 425–455 (1994).
- Foi, A., Katkovnik, V., & Egiazarian, K., "Pointwise shape-adaptive DCT for high-quality denoising and deblocking of grayscale and color images", *IEEE TIP*, 16(5), 1395–1411 (2007).
- Mallat, S., "A theory for multiresolution signal decomposition: the wavelet representation", *IEEE PAMI*, 11, 674–693 (1989).
- Portilla, J., Strela, V., Wainwright, M. J., & Simoncelli, E. P., "Image denoising using scale mixtures of Gaussians in the wavelet domain", *IEEE TIP*, 12(11), 1338–1351 (2003).
- Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L., "Beyond a Gaussian denoiser: residual learning of deep CNN for image denoising", *IEEE TIP*, 26(7), 3142–3155 (2017).
