---
title: "Nonlocal Transform-Domain Filter for Volumetric Data Denoising and Reconstruction"
authors: Matteo Maggioni, Vladimir Katkovnik, Karen Egiazarian, Alessandro Foi
year: 2013
journal: "IEEE Transactions on Image Processing 22(1), pp. 119–133"
doi: "10.1109/TIP.2012.2210725"
topic: Low-SNR Imaging / Volumetric Denoising
tags: [bm4d, volumetric-denoising, mri-denoising, ct-denoising, 4d-collaborative-filtering, rician-noise, vst, variance-stabilising-transform, iterative-reconstruction, k-space]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 10. Nonlocal Transform-Domain Filter for Volumetric Data Denoising and Reconstruction (BM4D) / 볼륨 데이터 노이즈 제거와 재구성을 위한 비국소 변환영역 필터

---

## 1. Core Contribution / 핵심 기여

### 한국어
**BM4D**는 BM3D (paper #7)의 *볼륨 데이터*(3-D) 확장. BM3D가 2-D image patch를 처리하는 데 반해 BM4D는 $L \times L \times L$ **3-D cube** $C^z_{x_R}$를 기본 단위로 함. 핵심 기여:

(A) **3-D cubes + 4-D groups**: 잡음 볼륨에서 $L \times L \times L$ cube 추출, 비국소 검색으로 *상호 유사 cubes* stacking → 4-D group. 차원: 3-D (cube의 voxel 공간) × 1-D (비국소 stack).

(B) **4-D collaborative filtering**:
$$
\mathcal T^{ht}_{4D} = (3\text{-D bior 1.5 wavelet}) \otimes (\text{1-D Haar along stack})
$$
$$
\mathcal T^{wie}_{4D} = (3\text{-D DCT}) \otimes (\text{1-D Haar along stack})
$$
Hard threshold (Step 1) + Wiener (Step 2), BM3D와 동일.

(C) **MRI denoising application**: T1 BrainWeb phantom ($181 \times 217 \times 181$ voxel) + Gaussian / **Rician** noise 모델 (Eq. 13). Rician은 magnitude MR data의 자연적 noise distribution.

(D) **Variance-Stabilising Transform (VST)** for Rician (Eq. 14):
$$
\hat y = \mathrm{VST}^{-1}\bigl(\mathrm{BM4D}(\mathrm{VST}(z, \sigma), \sigma_{\mathrm{VST}}), \sigma\bigr)
$$
Forward VST → Gaussian-distributed → BM4D → inverse VST. BM4D 알고리즘 자체는 *수정 불필요*.

(E) **Iterative reconstruction from incomplete measurements** (§IV): k-space (Fourier) 또는 Radon 측정의 일부만 있는 경우, BM4D를 *regularizer* operator로 iteratively 적용. Compressed sensing analog. Shepp-Logan + BrainWeb 둘 다 검증.

**성과 (Table II, BrainWeb σ=15%)**:
- BM4D: PSNR 30.82, SSIM 0.91 — Gaussian noise.
- 비교: OB-NLM3D 28.61/0.82, OB-NLM3D-WM 29.68/0.85, ODCT3D 29.35/0.86, PRI-NLM3D 28.99/0.85.
- BM4D가 모든 σ에서 +1.3 dB까지 우수.
- Rician noise: VST + BM4D가 PRI-NLM3D보다 우수.

**Implementations**: Matlab/C 공개 (cs.tut.fi/~foi/GCF-BM3D/), 11분으로 BrainWeb phantom denoise (2013 hardware).

### English
**BM4D** extends BM3D from 2-D images to 3-D volumes (cubes). 4-D groups (3-D cubes stacked along nonlocal dim) processed via separable 4-D transform (3-D bior wavelet + 1-D Haar) with hard-thresholding (Step 1) and Wiener (Step 2). Validated on MRI denoising (BrainWeb, OASIS): up to 1.3 dB better than OB-NLM3D, ODCT3D, PRI-NLM3D. VST handles Rician noise. Iterative reconstruction extension for incomplete k-space data.

---

## 2. Reading Notes / 읽기 노트

### Part I: §II Algorithm / 알고리즘

#### 한국어 — Setup
$$
z(x) = y(x) + \eta(x), \quad x \in X \subset \mathbb Z^3, \quad \eta \overset{iid}{\sim} \mathcal N(0, \sigma^2)
$$
#### 한국어 — §II.B.1 Hard-thresholding stage

**Cube and group**:
- Reference cube $C^z_{x_R}$: $L \times L \times L$ voxels (보통 $L = 4$).
- **Photometric distance** (Eq. 2):
$$
d(C^z_{x_i}, C^z_{x_j}) = \frac{\|C^z_{x_i} - C^z_{x_j}\|^2_2}{L^3}
$$
**No prefiltering** (BM3D와 차이) — 작은 cube 크기와 voxel-level smoothness가 grouping 강건성 보장.
- **Group set** (Eq. 3): $S^z_{x_R} = \{x_i: d(C^z_{x_R}, C^z_{x_i}) \le \tau^{ht}_{\text{match}}\}$.
- **4-D group** (Eq. 4): $\mathbf G^z_{S^z_{x_R}} = \bigsqcup_{x_i \in S^z_{x_R}} C^z_{x_i}$.

**4-D transform** (Eq. 5-6):
$$
\hat{\mathbf G}^{ht}_{S} = \mathcal T^{ht-1}_{4D}\bigl(\Upsilon^{ht}\bigl(\mathcal T^{ht}_{4D} \mathbf G^z_S\bigr)\bigr)
$$
$\mathcal T^{ht}_{4D}$ = 3-D bior 1.5 + 1-D Haar (along stack). $\Upsilon^{ht}$ = hard threshold with $\sigma \lambda_{4D}$.

**중요**: DC term은 임계화 안함 — group의 mean value 보존 (Section §II.B.1).

#### 한국어 — §II.B.2 Aggregation

**Weight** (Eq. 8):
$$
w^{ht}_{x_R} = 1/(\sigma^2 N^{ht}_{x_R})
$$
$N^{ht}_{x_R}$ = retained nonzero coefficients. Sparser group → higher weight.

**Estimate** (Eq. 7):
$$
\hat y^{ht}(x) = \frac{\sum_{x_R}\sum_{x_i \in S} w^{ht}_{x_R} \hat C^y_{x_i}(x)}{\sum_{x_R}\sum_{x_i} w^{ht}_{x_R}\chi_{x_i}(x)}
$$
#### 한국어 — §II.B.2 Wiener-filtering stage

같은 BM3D 흐름:
1. Re-grouping with basic estimate $\hat y^{ht}$.
2. Wiener shrinkage coefficients (Eq. 10):
$$
\mathbf W_{S^{\hat y^{ht}}_{x_R}} = \frac{|\mathcal T^{wie}_{4D} \hat{\mathbf G}^{ht}_{S}|^2}{|\mathcal T^{wie}_{4D} \hat{\mathbf G}^{ht}_{S}|^2 + \sigma^2}
$$
3. Filtered group (Eq. 11): element-wise multiply with noisy group's 4-D spectrum, inverse transform.
4. Aggregate with Wiener weight $w^{wie}_{x_R} = \sigma^{-2}\|\mathbf W\|^{-2}_2$ (Eq. 12).

**Key transforms**:
- $\mathcal T^{wie}_{4D}$: 3-D DCT + 1-D Haar.

#### 한국어 — Parameter settings (Table I)

| Parameter | HT Normal | HT Modif | Wiener Normal | Wiener Modif |
|---|---|---|---|---|
| Cube size $L$ | 4 | 4 | 4 | 5 |
| Group size $M$ | 16 | 32 | 32 | 32 |
| Step $N_{\text{step}}$ | 3 | 3 | 3 | 3 |
| Search cube $N_S$ | 11 | 11 | 11 | 11 |
| Similarity $\tau_{\text{match}}$ | 2.9 | 24.6 | 0.4 | 6.7 |
| Shrinkage $\lambda_{4D}$ | 2.7 | 2.8 | (Wiener) | (Wiener) |

"Modified" profile은 더 큰 cubes + larger group으로 high noise ($\sigma > 15\%$)에서 더 좋음.

#### English — §II Algorithm
3-D cubes ($L=4$) → block matching with no prefiltering → 4-D groups → 3-D bior wavelet × 1-D Haar transform → hard threshold → inverse → aggregate. Wiener stage: re-group with basic estimate, 3-D DCT × 1-D Haar, Wiener shrink, aggregate.

---

### Part II: §III Denoising Experiments / 노이즈 제거 실험

#### 한국어 — Test data
- **BrainWeb T1 phantom**: $181 \times 217 \times 181$ voxels, 1mm isotropic.
- **OASIS phantom**: $256 \times 256 \times 128$ real MR data.
- **Noise types**: Gaussian (Eq. 1) or **Rician** (Eq. 13):
$$
z(x) = \sqrt{(c_r y(x) + \sigma\eta_r(x))^2 + (c_i y(x) + \sigma\eta_i(x))^2}
$$
Magnitude of complex MR signal — Rician.

#### 한국어 — VST framework (Eq. 14)
$$
\hat y = \mathrm{VST}^{-1}(\mathrm{BM4D}(\mathrm{VST}(z, \sigma), \sigma_{\mathrm{VST}}), \sigma)
$$
VST: forward Anscombe-style → makes noise approximately Gaussian. Inverse VST handles bias correction.

#### 한국어 — Comparison filters
- **OB-NLM3D**: optimised blockwise nonlocal means (Coupé+ 2008)
- **OB-NLM3D-WM**: + wavelet mixing
- **ODCT3D**: oracle-based 3-D DCT
- **PRI-NLM3D**: prefiltered rotationally invariant nonlocal means

#### 한국어 — Results (Table II, BrainWeb)
| σ % | Noisy | OB-NLM3D | OB-NLM3D-WM | ODCT3D | PRI-NLM3D | **BM4D** |
|---|---|---|---|---|---|---|
| 5 | 26.02 | 34.73 | 35.01 | 34.89 | 35.51 | **35.95** |
| 11 | 19.17 | 30.32 | 29.68 | 30.90 | 30.40 | **32.28** |
| 15 | 16.48 | 28.61 | 28.18 | 29.35 | 28.99 | **30.82** |
| 19 | 14.42 | 27.28 | 27.55 | 28.18 | 28.40 | **29.70** |

BM4D가 모든 σ에서 우수. SSIM도 BM4D가 가장 높음.

**Visual** (Fig. 4, BrainWeb σ=15%): BM4D가 OB-NLM 변종들보다 *less over-smoothing* + *better edge preservation*. PRI-NLM3D, ODCT3D와는 비슷한 visual quality.

#### English — §III
On BrainWeb (T1 MR phantom) at σ=5-19%, BM4D consistently outperforms OB-NLM3D, OB-NLM3D-WM, ODCT3D, PRI-NLM3D by up to 1.3 dB PSNR. VST handles Rician noise without algorithm modification.

---

### Part III: §IV Iterative Reconstruction / 반복 재구성

#### 한국어
**Setting**: 측정 $z = \mathcal A y + \eta$, $\mathcal A$: k-space subsampling (MRI) 또는 Radon (CT). $\mathcal A$이 *severe undersampling*이라 직접 inversion 불가능.

**알고리즘** (반복):
1. $\hat y^{(0)} = \mathcal A^* z$ (zero-fill 추정)
2. For $k = 1, 2, \ldots$:
   a. $\hat y^{(k)}_{\text{filtered}} = \mathrm{BM4D}(\hat y^{(k-1)}, \sigma_k)$ (regularization)
   b. $\hat y^{(k)} = \hat y^{(k)}_{\text{filtered}} + \mathcal A^*(z - \mathcal A \hat y^{(k)}_{\text{filtered}})$ (data consistency)
   c. $\sigma_k$ decreasing schedule

**Equivalent to compressed sensing** with BM4D as sparsity-inducing regularizer. Similar to NESTA, FISTA.

**Test cases** (§IV.B-C):
- 3-D Shepp-Logan + Radon trajectories (radial, spiral, log-spiral, limited angle, spherical) — Fig. 4 bottom row.
- BrainWeb phantom + radial/spiral k-space sampling.
- BM4D 반복으로 noise + aliasing artifacts 동시 제거. Aliasing pattern과 noise를 *둘 다* nonlocal redundancy로 해결.

#### English — §IV
BM4D as a regularizer in iterative reconstruction from incomplete k-space (MRI) or Radon (CT) data. Tested on Shepp-Logan and BrainWeb with radial / spiral / limited-angle / spherical sampling. Removes both noise and aliasing artifacts.

---

## 3. Key Takeaways / 핵심 시사점

1. **3-D cubes는 2-D blocks의 자연 일반화 / 3-D cubes naturally generalise 2-D blocks** — $L = 4$인 cube는 BM3D의 $N_1 = 8$ block보다 *작지만 voxel 수는 비슷* ($4^3 = 64$ vs $8^2 = 64$). 작은 cubes로 spatial precision + group sparsity 둘 다 확보.
   3-D cubes ($L=4$, $4^3=64$ voxels) match BM3D's 2-D blocks ($8^2=64$) in element count but with finer spatial precision; this preserves edges in volumetric data.

2. **No prefiltering needed (BM3D와 차이) / No prefiltering** — BM3D는 거리 측정시 hard-threshold prefilter 사용 (Eq. 4 of paper #7). BM4D는 raw cube distance 직접 사용. 이유: 작은 cube + voxel-level natural smoothness가 잡음 매칭 강건.
   Smaller cubes + voxel-level smoothness (in MRI/CT) makes prefiltering unnecessary; raw cube distance suffices for grouping.

3. **VST framework는 BM4D 알고리즘 변경 없이 Rician 처리 / VST handles Rician without algorithm changes** — Eq. (14): forward VST (Gaussian-stabilizing) → BM4D → inverse VST. BM4D는 자기 자신이 *Gaussian assumption*을 기반으로 하지만, VST가 transparent layer 역할을 함. MRI magnitude 영상 (Rician)에 그대로 적용 가능.
   The VST-BM4D combination handles non-Gaussian (Rician) noise without modifying BM4D — VST stabilises, BM4D denoises, inverse VST restores.

4. **Modified profile이 high-noise에서 우수 / Modified profile better at high noise** — Modified profile은 larger cubes (5×5×5) + larger groups (32) + 더 큰 thresholds ($\lambda_{4D} = 2.8$ vs 2.7). σ ≥ 15%에서 PSNR 0.5-1 dB 추가 이득. 큰 cubes로 잡음 통계적 평균화, 큰 그룹으로 redundancy 더 활용.
   The modified profile ($L=5$, larger groups) gains 0.5-1 dB at high noise by leveraging more averaging at the cost of slightly less spatial precision.

5. **MRI denoising에서 SOTA / SOTA in MRI denoising** — OB-NLM3D, OB-NLM3D-WM, ODCT3D, PRI-NLM3D 등 nonlocal MR denoising 표준 baselines를 +0.3-1.3 dB 능가. Visual: edge preservation + flat region smoothness 모두 우수. SSIM도 가장 높음.
   BM4D dominates 2013-era MR denoisers (NLM-3D variants, ODCT3D) by up to 1.3 dB while preserving edges and SSIM.

6. **Iterative reconstruction = compressed sensing / Iterative reconstruction is compressed sensing** — §IV의 BM4D regularization과 data consistency 반복은 nonlinear iterative reconstruction 표준 framework. BM4D가 nonlocal+transform-domain prior 역할. NESTA/FISTA와 유사.
   BM4D-as-regulariser in iterative MRI/CT reconstruction is a compressed-sensing approach: the BM4D operator implicitly imposes a nonlocal sparsity prior in 4-D space.

7. **K-space subsampling pattern과 BM4D의 시너지 / Synergy between k-space patterns and BM4D** — radial, spiral, log-spiral, limited-angle, spherical 모두 검증. BM4D가 sampling pattern에 따라 다른 *aliasing artifact*를 생성해도 모두 nonlocal redundancy로 처리. 따라서 BM4D는 sampling pattern에 *robust*.
   BM4D handles radial, spiral, log-spiral, limited-angle, spherical k-space sampling — its nonlocal redundancy is sampling-pattern-agnostic.

8. **Computational complexity / 연산 복잡도** — BrainWeb phantom denoising 11 min on 2.66 GHz / 8 GB (single-threaded Matlab/C). 본 논문 시점에서 비현실적이지 않음 (*offline* MR denoising 용도). GPU 가속이나 deep learning 대체 시 1초 미만. $N_S = 11, N_{\text{step}} = 3$이 PSNR vs 시간 sweet spot (Table III).
   BM4D's 11-minute runtime per phantom is reasonable for offline scientific imaging; modern hardware brings this to <1 sec.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Volumetric model / 볼륨 모델
$$
z(x) = y(x) + \eta(x), \quad x \in X \subset \mathbb Z^3, \quad \eta \sim \mathcal N(0, \sigma^2)
$$
### 4.2 Cube and group / 큐브와 그룹
$L \times L \times L$ cube $C^z_{x_R}$. Photometric distance (Eq. 2):
$$
d(C^z_{x_i}, C^z_{x_j}) = \|C^z_{x_i} - C^z_{x_j}\|^2_2 / L^3
$$
4-D group (Eq. 4): $\mathbf G^z_{S} = \bigsqcup_{x_i \in S} C^z_{x_i}$, $S = \{x_i: d \le \tau\}$.

### 4.3 Hard-threshold stage / 강한 임계화 단계
$$
\hat{\mathbf G}^{ht}_{S} = \mathcal T^{ht-1}_{4D}\bigl(\Upsilon^{ht}\bigl(\mathcal T^{ht}_{4D} \mathbf G^z_S\bigr)\bigr)
$$
$\mathcal T^{ht}_{4D}$ = 3-D bior 1.5 ⊗ 1-D Haar. $\Upsilon^{ht}(c) = c \cdot \mathbf 1\{|c| > \sigma \lambda_{4D}\}$, DC preserved.

### 4.4 Aggregation (Eq. 7-8)
$$
w^{ht}_{x_R} = 1/(\sigma^2 N^{ht}_{x_R}), \quad \hat y^{ht}(x) = \text{weighted average}
$$
### 4.5 Wiener stage / Wiener 단계
$$
\mathbf W_S = \frac{|\mathcal T^{wie}_{4D} \hat{\mathbf G}^{ht}_S|^2}{|\mathcal T^{wie}_{4D} \hat{\mathbf G}^{ht}_S|^2 + \sigma^2}
$$
$$
\hat{\mathbf G}^{wie}_S = \mathcal T^{wie-1}_{4D}\bigl(\mathbf W_S \cdot \mathcal T^{wie}_{4D} \mathbf G^z_S\bigr), \quad w^{wie}_{x_R} = \sigma^{-2}\|\mathbf W\|^{-2}_2
$$
$\mathcal T^{wie}_{4D}$ = 3-D DCT ⊗ 1-D Haar.

### 4.6 VST for Rician noise (Eq. 13-14)
$$
z(x) = \sqrt{(c_r y(x) + \sigma\eta_r)^2 + (c_i y(x) + \sigma\eta_i)^2}, \quad \eta_r, \eta_i \sim \mathcal N(0, 1)
$$
$$
\hat y = \mathrm{VST}^{-1}\bigl(\mathrm{BM4D}(\mathrm{VST}(z, \sigma), \sigma_{\mathrm{VST}}), \sigma\bigr)
$$
### 4.7 Iterative reconstruction
For incomplete measurements $z = \mathcal A y + \eta$:
```
y_hat = A^*(z)              # zero-fill
for k = 1, 2, ...:
    y_hat = BM4D(y_hat, σ_k)        # regularization
    y_hat = y_hat + A^*(z - A·y_hat) # data consistency
    decrease σ_k
```

### 4.8 Worked numerical example
BrainWeb σ=15% → BM4D PSNR 30.82 dB. Compared to:
- OB-NLM3D 28.61 dB → BM4D +2.21 dB
- OB-NLM3D-WM 28.18 dB → BM4D +2.64 dB  
- ODCT3D 29.35 dB → BM4D +1.47 dB
- PRI-NLM3D 28.99 dB → BM4D +1.83 dB

MSE reduction: $10^{(30.82 - 28.99)/10} \approx 1.52\times$ better than PRI-NLM3D.

For Rician at σ=11%: VST+BM4D 30.90 dB vs PRI-NLM3D 29.71 dB — VST framework adds 1.19 dB.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1992 ─── Healy-Weaver — first wavelet MR denoising
1995 ─── Nowak — Rician bias correction in wavelet MR denoising
2005 ─── Buades-Coll-Morel — NLM (paper #4)
2006 ─── Coupé+ — OB-NLM3D (block-wise NLM for MR)
2007 ─── Dabov+ — BM3D (paper #7)
2010 ─── Manjón-Coupé+ — PRI-NLM3D (rotation-invariant)
2010 ─── Manjón-Coupé+ — ODCT3D
2012 ─── Maggioni+ — V-BM4D (paper #9, video)
2013 ★★ MAGGIONI-KATKOVNIK-EGIAZARIAN-FOI — BM4D (THIS PAPER)
                          ↳ volumetric extension of BM3D
                          ↳ MRI / CT denoising SOTA
                          ↳ iterative reconstruction extension
2017 ─── Manjón-Coupé+ — adaptive multi-resolution NLM3D
2018+ ── Deep learning MRI denoisers (DeepMRI, NLRRPCA-deep)
                  BM4D remains common baseline + used in scarce-data domains
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Dabov+ (2007)** *IEEE TIP* (paper #7) | BM3D | Direct ancestor; BM4D extends 2-D blocks to 3-D cubes. |
| **Buades-Coll-Morel (2005)** (paper #4) | NLM | Underlies the nonlocal grouping mechanism. |
| **Coupé+ (2008)** | OB-NLM3D | Direct 2013-era competitor; BM4D dominates by 1-2 dB. |
| **Manjón-Coupé+ (2010)** | PRI-NLM3D, ODCT3D | Strong MRI denoising baselines surpassed by BM4D. |
| **Maggioni+ (2012)** *IEEE TIP* (paper #9) | V-BM4D | Sister paper: video extension. Same 4-D framework, different 4th dimension semantics. |
| **Foi+ (2009)** | VST for Rician | Provides the variance-stabilising transform integrated in BM4D. |
| **Compressed sensing (Donoho 2006)** | Iterative reconstruction | §IV uses BM4D as a sparsity regulariser in iterative MR/CT reconstruction. |
| **Modern deep MRI denoisers** (2018+) | DeepMRI, NLRPCA | Replace BM4D with learned alternatives; BM4D remains training-free baseline. |

---

## 7. References / 참고문헌

- Coupé, P., Yger, P., Prima, S., Hellier, P., Kervrann, C., & Barillot, C., "An optimized blockwise nonlocal means denoising filter for 3-D magnetic resonance images", *IEEE TMI*, 27(4), 425–441 (2008).
- Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K., "Image denoising by sparse 3-D transform-domain collaborative filtering", *IEEE TIP*, 16(8), 2080–2095 (2007).
- Foi, A., "Noise estimation and removal in MR imaging: the variance-stabilization approach", *IEEE ISBI* (2011).
- Maggioni, M., Boracchi, G., Foi, A., & Egiazarian, K., "Video denoising, deblocking, and enhancement through separable 4-D nonlocal spatiotemporal transforms", *IEEE TIP*, 21(9), 3952–3966 (2012).
- Maggioni, M., Katkovnik, V., Egiazarian, K., & Foi, A., "Nonlocal transform-domain filter for volumetric data denoising and reconstruction", *IEEE TIP*, 22(1), 119–133 (2013). [DOI: 10.1109/TIP.2012.2210725]
- Manjón, J. V., Coupé, P., Buades, A., Collins, D. L., & Robles, M., "New methods for MRI denoising based on sparseness and self-similarity", *Medical Image Analysis*, 16(1), 18–27 (2012).
- BrainWeb phantom: https://brainweb.bic.mni.mcgill.ca
- BM4D Matlab/C: http://www.cs.tut.fi/~foi/GCF-BM3D/
