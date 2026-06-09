---
title: "Video Denoising, Deblocking, and Enhancement Through Separable 4-D Nonlocal Spatiotemporal Transforms"
authors: Matteo Maggioni, Giacomo Boracchi, Alessandro Foi, Karen Egiazarian
year: 2012
journal: "IEEE Transactions on Image Processing 21(9), pp. 3952–3966"
doi: "10.1109/TIP.2012.2199324"
topic: Low-SNR Imaging / Video Denoising
tags: [v-bm4d, bm3d-extension, video-denoising, spatiotemporal-volumes, 4d-transform, motion-trajectory, nonlocal-filter, deblocking, enhancement, alpha-rooting]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 9. Video Denoising via Separable 4-D Nonlocal Spatiotemporal Transforms (V-BM4D) / 분리형 4-D 비국소 시공간 변환을 통한 비디오 노이즈 제거

---

## 1. Core Contribution / 핵심 기여

### 한국어
**V-BM4D**는 BM3D (paper #7)의 *비디오 확장*. 핵심 아이디어는 *공간 redundancy*와 *시간 redundancy*를 *별도로 활용*:

(A) **Spatiotemporal volumes (시공간 볼륨)**: $N \times N$ 공간 블록을 motion trajectory를 따라 $h^- + h^+ + 1$개 프레임에 걸쳐 stacking → 3-D volume $V_z(\mathbf x_0, t_0)$. 즉 *움직이는 객체*를 따라가는 *물리적으로 의미 있는* 3-D 데이터.

(B) **4-D Groups**: 비국소 검색으로 *유사한 volumes*들을 stacking → 4-D structure $G_z(\mathbf x_0, t_0)$. 차원: 2-D (공간) × 1-D (시간) × 1-D (비국소). 이는 BM3D의 3-D group을 한 차원 확장.

(C) **4-D collaborative filtering**: 분리형 4-D 변환 $\mathcal T_{4D} = \mathcal T_{2D} \otimes \mathcal T_{1D}^{(t)} \otimes \mathcal T_{1D}^{(\text{nonlocal})}$. Hard threshold (Step 1) + Wiener filtering (Step 2), BM3D와 동일 구조.

(D) **Motion vector tracking with smoothness penalty** (Eq. 7):
$$
\delta^b(B_z(\mathbf x_i, t_i), B_z(\mathbf x_j, t_i \pm 1)) = \frac{\|B_z(\mathbf x_i, t_i) - B_z(\mathbf x_j, t_i\pm 1)\|^2}{N^2} + \gamma_d \|\hat{\mathbf x}_i(t_i\pm 1) - \mathbf x_j\|_2
$$
$\gamma_d$: 위치 예측 페널티 (motion smoothness). 평탄 영역·균일 텍스처에서 잡음만 따라가는 trajectory 방지.

(E) **Adaptive parameters**: $\sigma$에 의존하는 quadratic functions (Eq. 9-11)으로 $\gamma_d, \tau_{\text{traj}}, \tau_{\text{match}}$ 자동 결정. 8개 test sequence + Nelder-Mead 최적화로 fit.

(F) **Extensions**:
- **Deblocking** (§IV): MPEG/H.264 압축 artifacts에 V-BM4D 적용. $\sigma(\text{bpp}, q)$ bivariate function (Eq. 12)으로 적응 $\sigma$ 추정.
- **Enhancement** (§V): hard-thresholding 단계에서 alpha-rooting ($\bar\phi_i = \mathrm{sgn}(\phi_i)|\phi_0||\phi_i/\phi_0|^{1/\alpha}$, Eq. 13)으로 sharpening + denoising 동시 수행.
- **Color** (§VI.D): luminance trajectory + 채널별 4-D collaborative filtering.

**성과 (PSNR)**:
- σ=20에서 *Tennis* 32.04 dB, *Salesman* 33.02 dB, *Foreman* 35.81 dB (paper Table II).
- BM3D를 *frame-by-frame* 적용하는 baseline 대비 +0.5-2 dB 우수 (시간적 redundancy 활용).
- 비교: VBM3D (이전 비디오 BM3D), c-bm4d (color BM4D)도 능가.

### English
**V-BM4D** extends BM3D to video. Key novelty: separately exploit *spatial* and *temporal* redundancy via **spatiotemporal volumes** (3-D blocks along motion trajectories) grouped into **4-D structures**. A separable 4-D transform (2-D spatial × 1-D temporal × 1-D nonlocal) enables collaborative filtering with hard-thresholding + Wiener. Motion-smoothness penalty (Eq. 7) avoids noise-driven trajectories. PSNR improves over frame-by-frame BM3D by 0.5-2 dB.

---

## 2. Reading Notes / 읽기 노트

### Part I: §I-II Algorithm / 알고리즘

#### 한국어 — Setup
$z(\mathbf x, t) = y(\mathbf x, t) + \eta(\mathbf x, t)$, $\eta \sim \mathcal N(0, \sigma^2)$.
$B_z(\mathbf x_0, t_0)$: $(\mathbf x_0, t_0)$ 기준 $N \times N$ block.

#### 한국어 — §II.B Spatiotemporal volumes
**Trajectory** (Eq. 2): $\text{Traj}(\mathbf x_0, t_0) = \{(\mathbf x_j, t_0 + j): j = -h^-, \ldots, h^+\}$.

**Volume** (Eq. 3): $V_z(\mathbf x_0, t_0) = \{B_z(\mathbf x_i, t_i): (\mathbf x_i, t_i) \in \text{Traj}\}$.

핵심: $\mathbf x_j$는 *motion trajectory* 따라 이동 — 단순 정적 위치 아님. 이 trajectory가 motion vector $\mathbf v(\mathbf x_i, t_i) = \mathbf x_i - \mathbf x_{i-1}$의 concatenation.

#### 한국어 — §II.C Grouping
**Volume distance** $\delta^v$ (Eq. 8): $L^2$ norm of *time-synchronous* volume difference, length-normalized.

**Group** (Eq. 4): $G_z(\mathbf x_0, t_0) = \{V_z(\mathbf x_i, t_i): \delta^v < \tau_{\text{match}}\}$. 4-D structure.

#### 한국어 — §II.D Collaborative filtering
$d+1 = 4$ 차원 변환 + shrinkage + 역변환. 분리형:
$$
\mathcal T_{4D} = \mathcal T_{2D} \otimes \mathcal T_{1D}^{(t)} \otimes \mathcal T_{1D}^{(\text{nl})}
$$
- $\mathcal T_{2D}$: 2-D 공간 변환 (Bior 1.5 또는 DCT).
- $\mathcal T_{1D}^{(t)}$: 1-D 시간 변환 (보통 DCT) — *intra-volume temporal correlation* 활용.
- $\mathcal T_{1D}^{(\text{nl})}$: 1-D 비국소 변환 (보통 Haar) — *inter-volume similarity* 활용.

#### 한국어 — §II.E Aggregation
Eq. (5): convex combination with adaptive weights ∝ 1/total variance, BM3D와 동일.

#### English — §II
4-D groups = stacks of 3-D volumes (motion-tracked blocks) of mutually similar trajectories. Separable 4-D transform = 2-D spatial × 1-D temporal × 1-D nonlocal. Hard-threshold (Step 1) + Wiener (Step 2), as in BM3D.

---

### Part II: §III Implementation / 구현

#### 한국어 — Motion tracking
- **Location prediction** (Eq. 6): $\hat{\mathbf x}_i(t+1) = \mathbf x_i + \gamma_p \mathbf v(\mathbf x_i, t_i)$, $\gamma_p \in [0, 1]$. 부드러운 motion 가정.
- **Penalised similarity** (Eq. 7): $\delta^b$에 $\gamma_d \|\hat{\mathbf x}_i(t\pm 1) - \mathbf x_j\|$ 추가 → trajectory가 motion-smooth하도록 강제.
- **Adaptive search neighborhood**: $N_{PR}$이 motion velocity $\mathbf v$에 따라 적응. 빠른 객체 → 큰 search.
- **Trajectory truncation**: $\tau_{\text{traj}}$ 임계 초과시 trajectory 끝 — occlusion, scene change 처리.

#### 한국어 — Two-stage filtering
같은 BM3D 흐름:
1. **Hard threshold stage**: $\hat G^{ht}_y = \mathcal T_{4D}^{-1}(\Upsilon^{ht}(\mathcal T_{4D} G_z))$.
2. **Wiener stage**: basic estimate를 pilot으로 4-D Wiener.

#### 한국어 — Adaptive parameters (Eq. 9-11)
8개 test sequence + Nelder-Mead로 $(\gamma_d, \tau_{\text{traj}}, \tau_{\text{match}})$을 $\sigma$의 quadratic function로 fit.

#### English — §III
Motion vectors via penalised similarity (motion smoothness via $\gamma_d$ term). Adaptive search neighborhood scales with motion velocity. Two-stage filtering. Adaptive parameters as quadratic in $\sigma$.

---

### Part III: §IV-V Extensions / 확장

#### 한국어 — Deblocking
MPEG/H.264 압축 artifacts (blocking, ringing, mosquito noise)를 *additive Gaussian noise*로 모델링. $\sigma(\text{bpp}, q)$ (Eq. 12) bivariate function으로 적응 $\sigma$ 추정. 8개 sequence에서 fit.

#### 한국어 — Enhancement (sharpening)
**Alpha-rooting** (Eq. 13): $\bar\phi_i = \mathrm{sgn}(\phi_i)|\phi_0|^{1-1/\alpha}|\phi_i|^{1/\alpha}$, $\alpha > 1$. 작은 계수를 큰 계수에 비해 *상대적*으로 강조 → high-frequency 강조 → sharpening. Hard-threshold 단계에서만 적용.

V-BM4D + alpha-rooting → 동시 denoising + sharpening + flicker 억제.

#### English
Deblocking by treating compression artifacts as Gaussian noise; adaptive $\sigma(\text{bpp}, q)$. Enhancement via alpha-rooting on hard-threshold stage coefficients.

---

### Part IV: §VI Experiments / 실험

#### 한국어 (간략)
- **Test sequences**: Tennis, Foreman, Coastguard, Salesman, Flower Garden, Bicycle, Bus, Miss America. 모두 grayscale CIF (352×288) 또는 QCIF (176×144).
- **Settings**: $\sigma \in \{5, 10, 15, 20, 25, 30, 50, 70\}$, σ-adaptive parameters via Eq. 9-11.
- **Comparison**: VBM3D (Dabov+ 2007 video extension), single-image BM3D applied frame-by-frame, RNLM (recursive NLM), KSVD-3D.
- **PSNR**: V-BM4D는 모든 σ, 모든 sequence에서 +0.3-2 dB 우수.
- **Visual** (§VI subjective + MOVIE index): V-BM4D가 motion blur와 flicker 모두 적게 보임.

#### English
V-BM4D outperforms VBM3D, frame-by-frame BM3D, RNLM, and KSVD-3D by 0.3-2 dB across the standard 8 sequences. Visual quality (MOVIE index) is also superior, with less motion blur and flicker.

---

## 3. Key Takeaways / 핵심 시사점

1. **Spatiotemporal volume이 V-BM4D의 핵심 발명 / Spatiotemporal volume is V-BM4D's central invention** — 단순 3-D block이 아니라 *motion trajectory를 따라가는* 3-D structure. 이는 *동일한 객체의 시간 변화*를 정확히 추적 → temporal redundancy 보존.
   The spatiotemporal volume — 3-D structure following motion — is V-BM4D's defining contribution: it captures *temporal redundancy* of the same object in motion.

2. **4-D group은 spatial + temporal redundancy 분리 활용 / 4-D groups separately exploit spatial vs temporal redundancy** — 4-D 변환 = 2-D 공간 × 1-D 시간 × 1-D 비국소. 각 차원이 다른 종류의 correlation을 활용 → group이 매우 sparse.
   The 4-D group transform decomposes into spatial × temporal × nonlocal — three separable correlations, all exploited at once.

3. **Motion smoothness penalty가 잡음 강건성의 핵심 / Motion smoothness penalty is key to noise robustness** — Eq. 7의 $\gamma_d$ 항이 trajectory가 motion vector field와 일치하도록 강제. 평탄 영역에서 잡음만 따라가는 ill-conditioned trajectory 방지.
   The $\gamma_d$ penalty regularises motion vectors against noise-driven trajectories in flat regions, dramatically improving robustness.

4. **Trajectory length가 가변적 / Variable trajectory length** — $\tau_{\text{traj}}$ 초과시 trajectory 끝남 → occlusion, scene change, illumination change 자연스럽게 처리. 모든 volumes는 같은 길이 (group 동질성 위해) — sub-volume extraction $\mathcal E_{L_0}$ 사용.
   Trajectories truncate when block matching fails (occlusion, scene change), then sub-volumes of common length $L_0$ are extracted to ensure homogeneous 4-D groups.

5. **Adaptive parameters via Eq. 9-11 / σ-adaptive parameters** — 9 coefficient의 quadratic polynomial로 $(\gamma_d, \tau_{\text{traj}}, \tau_{\text{match}})$을 $\sigma$의 함수로 fit. Pre-trained — 사용자 개입 없음. 8 test sequences로 robust.
   Pre-fit quadratic parameters (Eq. 9-11) make V-BM4D fully automatic across noise levels.

6. **Deblocking은 압축 artifacts를 Gaussian noise로 모델링 / Deblocking treats compression artifacts as noise** — MPEG/H.264 blockiness, ringing, mosquito noise 모두 $\eta$로 추상화. $\sigma(\text{bpp}, q)$ (Eq. 12)로 자동 추정 → 사용자 개입 없이 V-BM4D를 deblocker로 사용.
   Compression artifacts are treated as additive Gaussian noise; $\sigma$ is estimated from bitrate and quantisation parameter.

7. **Alpha-rooting은 sharpening + denoising 결합 / Alpha-rooting combines sharpening with denoising** — Hard-threshold 단계에서만 적용 (Eq. 13). $\alpha > 1$으로 작은 계수를 큰 계수에 비해 강조. Wiener 단계에서는 미적용 (잡음 증폭 위험).
   Alpha-rooting in the hard-threshold stage simultaneously denoises and sharpens; restricted to Step 1 to avoid noise amplification in Step 2's Wiener.

8. **Frame-by-frame BM3D 대비 시간 redundancy의 가치 / Value of temporal redundancy** — Frame-by-frame BM3D는 같은 비디오 frame을 *독립적*으로 처리 → 시간 correlation 무시. V-BM4D는 그 redundancy 활용으로 +0.5-2 dB 추가 PSNR. 비디오 denoising에서 motion-aware 처리는 본질적.
   Frame-by-frame BM3D wastes temporal redundancy; motion-aware V-BM4D recovers 0.5-2 dB by exploiting it. Motion-awareness is essential for video denoising.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Setup / 모델
$$
z(\mathbf x, t) = y(\mathbf x, t) + \eta(\mathbf x, t), \quad \eta \overset{iid}{\sim} \mathcal N(0, \sigma^2)
$$
### 4.2 Spatiotemporal volume / 시공간 볼륨
$$
\text{Traj}(\mathbf x_0, t_0) = \{(\mathbf x_j, t_0 + j): j = -h^-, \ldots, h^+\}
$$
$$
V_z(\mathbf x_0, t_0) = \{B_z(\mathbf x_i, t_i): (\mathbf x_i, t_i) \in \text{Traj}\}
$$
### 4.3 Motion-smoothness penalised similarity (Eq. 7)
$$
\delta^b(B_i, B_j) = \frac{\|B_z(\mathbf x_i, t_i) - B_z(\mathbf x_j, t_i\pm 1)\|^2}{N^2} + \gamma_d \|\hat{\mathbf x}_i(t_i\pm 1) - \mathbf x_j\|_2
$$
$$
\hat{\mathbf x}_i(t+1) = \mathbf x_i + \gamma_p \mathbf v(\mathbf x_i, t_i) \quad (6)
$$
### 4.4 4-D group (Eq. 4)
$$
G_z(\mathbf x_0, t_0) = \{V_z(\mathbf x_i, t_i): \delta^v(V_z(\mathbf x_0, t_0), V_z(\mathbf x_i, t_i)) < \tau_{\text{match}}\}
$$
4-D structure: $N \times N \times L_0 \times M$ (M = group size).

### 4.5 4-D collaborative filtering
$$
\mathcal T_{4D} = \mathcal T_{2D}^{\text{spatial}} \otimes \mathcal T_{1D}^{\text{temporal}} \otimes \mathcal T_{1D}^{\text{nonlocal}}
$$
$$
\hat G^{ht}_y = \mathcal T_{4D}^{-1}(\Upsilon^{ht}(\mathcal T_{4D} G_z))
$$
Wiener step uses $\hat G^{ht}_y$ as pilot.

### 4.6 Adaptive parameters (Eq. 9-11)
$$
\gamma_d(\sigma) = 0.0005\sigma^2 - 0.0059\sigma + 0.0400
$$
$$
\tau_{\text{traj}}(\sigma) = 0.0047\sigma^2 + 0.0676\sigma + 0.4564
$$
$$
\tau_{\text{match}}(\sigma) = 0.0171\sigma^2 + 0.4520\sigma + 47.9294
$$
### 4.7 Aggregation (Eq. 5)
$$
\hat y(\mathbf x, t) = \frac{\sum_{(\mathbf x_0, t_0)} \sum_{(\mathbf x_i, t_i) \in \text{Ind}} w_{(\mathbf x_0, t_0)} \hat V_y(\mathbf x_i, t_i)(\mathbf x, t)}{\sum_{(\mathbf x_0, t_0)} \sum w_{(\mathbf x_0, t_0)}\chi_{(\mathbf x_i, t_i)}(\mathbf x, t)}
$$
Weights ∝ 1/(σ²·N_har) (HT) or σ⁻²‖W‖₂⁻² (Wiener).

### 4.8 Alpha-rooting (Eq. 13)
$$
\bar\phi_i = \begin{cases} \mathrm{sgn}(\phi_i)|\phi_0||\phi_i/\phi_0|^{1/\alpha} & \phi_0 \neq 0 \\ \phi_i & \text{otherwise}\end{cases}
$$
$\alpha > 1$ for sharpening.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1989-94 ─ MPEG: motion compensation in video compression
2005 ─── Buades-Coll-Morel — NLM (paper #4)
2007 ─── Dabov+ — BM3D (paper #7)
2007 ─── Dabov-Foi-Egiazarian — VBM3D (video BM3D, naive temporal extension)
2009 ─── Maggioni+ — V-BM3D conference paper (motion-aware)
2010 ─── Mairal+ — non-local sparse models LSSC
2012 ★★ MAGGIONI-BORACCHI-FOI-EGIAZARIAN — V-BM4D (THIS PAPER)
                          ↳ spatiotemporal volumes + 4-D groups
                          ↳ deblocking, enhancement, color extensions
2013 ─── Maggioni+ — BM4D (volumetric, paper #10)
2017 ─── Tassano+ — DVDnet (deep video denoising) — surpasses V-BM4D
2018+ ── FastDVDnet, EDVR, BasicVSR — modern deep video methods
                  V-BM4D remains baseline + used in scientific imaging.
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Dabov+ (2007)** *IEEE TIP* (paper #7) | BM3D | Direct ancestor; V-BM4D extends BM3D's 2-D blocks to 3-D motion-tracked volumes. |
| **Buades-Coll-Morel (2005)** (paper #4) | NLM | The nonlocal grouping mechanism is shared with V-BM4D's 4-D grouping step. |
| **Dabov-Foi-Egiazarian (2007)** | VBM3D (naive) | First video extension; V-BM4D's predecessor. V-BM4D adds proper motion tracking. |
| **MPEG/H.264 standards** | Motion compensation | Borrows motion-vector-based block matching for trajectory construction. |
| **Mairal+ (2010)** | LSSC | Alternative video denoising via group-sparse coding; V-BM4D is competitive without dictionary learning. |
| **Maggioni+ (2013)** *IEEE TIP* (paper #10) | BM4D (volumetric) | Sister extension: BM4D for 3-D volumes (CT/MRI), V-BM4D for video. Same 4-D group framework. |
| **Tassano+ (2017)** *DVDnet* | Deep video denoising | Modern alternative; V-BM4D remains training-free baseline. |

---

## 7. References / 참고문헌

- Buades, A., Coll, B., & Morel, J.-M., "A non-local algorithm for image denoising", *Proc. IEEE CVPR*, 2, 60–65 (2005).
- Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K., "Image denoising by sparse 3-D transform-domain collaborative filtering", *IEEE TIP*, 16(8), 2080–2095 (2007).
- Dabov, K., Foi, A., & Egiazarian, K., "Video denoising by sparse 3D transform-domain collaborative filtering", *Proc. EUSIPCO* (2007).
- Maggioni, M., Boracchi, G., Foi, A., & Egiazarian, K., "Video denoising, deblocking, and enhancement through separable 4-D nonlocal spatiotemporal transforms", *IEEE TIP*, 21(9), 3952–3966 (2012). [DOI: 10.1109/TIP.2012.2199324]
- Mairal, J., Bach, F., Ponce, J., Sapiro, G., & Zisserman, A., "Non-local sparse models for image restoration", *Proc. IEEE ICCV* (2009).
- Tassano, M., Delon, J., & Veit, T., "DVDnet: A fast network for deep video denoising", *Proc. IEEE ICIP* (2019).
