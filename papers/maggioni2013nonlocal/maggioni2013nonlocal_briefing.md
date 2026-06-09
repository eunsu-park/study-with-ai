---
title: "Pre-Reading Briefing: Nonlocal Transform-Domain Filter for Volumetric Data Denoising and Reconstruction (BM4D)"
paper_id: "10_maggioni_2013"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# BM4D (Maggioni+ 2013): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Maggioni, M., Katkovnik, V., Egiazarian, K., & Foi, A., "Nonlocal transform-domain filter for volumetric data denoising and reconstruction", *IEEE Trans. Image Process.* 22(1), 119–133 (2013). [DOI: 10.1109/TIP.2012.2210725]
**Author(s)**: Matteo Maggioni, Vladimir Katkovnik, Karen Egiazarian, Alessandro Foi
**Year**: 2013

---

## 1. 핵심 기여 / Core Contribution

### 한국어
BM4D는 BM3D(논문 #7)의 **3-D 볼륨 데이터 일반화** — 2-D 패치 대신 $L \times L \times L$ **3-D cube** ($L = 4$ 표준)를 기본 단위로, 비국소 검색으로 모은 cubes를 stacking하여 **4-D group**을 형성한다. 분리형 4-D 변환(3-D bior 1.5 wavelet $\otimes$ 1-D Haar) + hard-threshold + Wiener의 BM3D 2단계 구조를 그대로 계승한다. 핵심 응용 두 가지: (1) **MRI denoising** — BrainWeb / OASIS phantom에서 OB-NLM3D, ODCT3D, PRI-NLM3D를 +0.3–1.3 dB 능가. Rician magnitude noise는 *Variance-Stabilising Transform*(논문 #11의 Anscombe 후예)로 전처리하여 BM4D 알고리즘 자체를 수정하지 않고 처리. (2) **Iterative reconstruction** — k-space (MRI) 또는 Radon (CT) subsampled 측정에서 BM4D를 *prior regulariser*로 반복 적용 (compressed sensing의 한 형태). Shepp-Logan, BrainWeb 모두에서 noise + aliasing artifact 동시 제거.

### English
BM4D is the **volumetric generalisation of BM3D** (paper #7) — replacing 2-D patches with $L \times L \times L$ **3-D cubes** ($L = 4$ default) and stacking similar cubes via non-local search into **4-D groups**. The separable 4-D transform (3-D bior-1.5 wavelet $\otimes$ 1-D Haar) plus the BM3D two-step (hard threshold + Wiener) carries over directly. Two flagship applications: (1) **MRI denoising** — beats OB-NLM3D, ODCT3D, PRI-NLM3D by 0.3–1.3 dB on BrainWeb/OASIS phantoms; Rician magnitude noise is handled by a *Variance-Stabilising Transform* (Anscombe descendant, see paper #11) without modifying BM4D itself. (2) **Iterative reconstruction** — applies BM4D as a *prior regulariser* in alternation with data consistency for sub-sampled k-space (MRI) or Radon (CT) measurements (compressed-sensing style). Validated on Shepp-Logan and BrainWeb with radial/spiral/limited-angle/spherical sampling — removes noise *and* aliasing.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
2010년대 초 의료영상 denoising은 *blockwise NLM* 계열(Coupé+ 2008 OB-NLM3D, Manjón+ 2010 PRI-NLM3D, ODCT3D)이 주도했다. 이들은 NLM(논문 #4)을 3-D voxel에 직접 확장한 형태로 patch 평균에 기반했다. 한편 2007 BM3D(논문 #7)는 자연영상에서 SOTA였고, 2012 V-BM4D(논문 #9)는 BM3D의 4-D 확장 청사진을 비디오로 보여줬다. BM4D는 그 *볼륨 분기*에 해당 — V-BM4D와 같은 4-D collaborative-filtering framework이지만 4번째 차원의 *의미*가 다르다 (V-BM4D: 시간; BM4D: 비국소). 또한 2013은 compressed sensing(CS)이 임상 MRI에 안착한 시기였고, BM4D-based iterative reconstruction은 BM4D를 CS regulariser로 사용하는 첫 결정판이다.

#### English
By the early 2010s medical-image denoising was dominated by *blockwise NLM* variants (Coupé+ 2008 OB-NLM3D, Manjón+ 2010 PRI-NLM3D, ODCT3D) — all 3-D voxel extensions of NLM (paper #4) using patch averaging. Meanwhile BM3D (paper #7) had set 2007 SOTA on natural images, and V-BM4D (paper #9, 2012) showed the 4-D extension blueprint for video. BM4D is the *volumetric branch* of that same framework — same 4-D collaborative filtering, but the 4th dimension means *non-local* rather than *temporal*. 2013 was also the year compressed-sensing MRI matured clinically; BM4D-as-CS-regulariser is the definitive 4-D entry in that line.

### 타임라인 / Timeline

```
1948 ─── Anscombe — VST for Poisson/binomial (paper #11)
2005 ─── Buades-Coll-Morel — NLM (paper #4)
2007 ─── Dabov+ — BM3D (paper #7)
2008 ─── Coupé+ — OB-NLM3D (block-wise NLM for MR)
2010 ─── Manjón-Coupé+ — PRI-NLM3D, ODCT3D
2011 ─── Foi — Rician VST for MRI
2012 ─── Maggioni+ — V-BM4D (paper #9, video)
2013 ★★ MAGGIONI-KATKOVNIK-EGIAZARIAN-FOI — BM4D (THIS PAPER)
2018+ ── Deep MRI denoisers (DeepMRI, NLRPCA, Self-supervised)
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
- **BM3D** (논문 #7) — *필수*: cube/group, collaborative filtering, two-step
- **3-D voxel data** 구조 (MRI/CT volumes)
- **Rician noise model**: magnitude MR data의 자연 분포 $z = \sqrt{(c_r y + \sigma\eta_r)^2 + (c_i y + \sigma\eta_i)^2}$
- **Variance-Stabilising Transform** (논문 #11): Rician → 근사 Gaussian
- **Compressed sensing & iterative reconstruction**: $\hat y^{(k+1)} = \hat y^{(k)} + \mathcal A^*(z - \mathcal A \hat y^{(k)})$ + regulariser
- **k-space (Fourier)와 Radon transform**: MR/CT 측정 모델

### English
- **BM3D** (paper #7) — essential: cubes/groups, collaborative filtering, two-step
- **3-D voxel data** structures (MRI/CT volumes)
- **Rician noise model** for magnitude MR images
- **Variance-Stabilising Transform** (paper #11): Rician → approximately Gaussian
- **Compressed sensing & iterative reconstruction** with regulariser
- **k-space (Fourier) and Radon transforms** as forward operators

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| 3-D cube / 3-D 큐브 | $L \times L \times L$ voxel block ($L=4$). BM3D 2-D 패치의 3-D 일반화. / $L\times L\times L$ voxel block — volumetric analogue of BM3D's 2-D patch. |
| 4-D group / 4-D 그룹 | 비국소 유사 cubes를 stacking. 차원: 3-D voxel × 1-D 비국소. / Stack of similar cubes — dimensions: 3-D voxel × 1-D non-local. |
| Photometric distance / 광도 거리 | $\|C_i - C_j\|^2 / L^3$. BM3D와 달리 prefilter 없음. / Raw $\ell^2$ distance over cubes — no prefilter (unlike BM3D). |
| 4-D collaborative filter / 4-D 협력 필터 | $\mathcal T^{ht}_{4D}$ = 3-D bior 1.5 ⊗ 1-D Haar; $\mathcal T^{wie}_{4D}$ = 3-D DCT ⊗ 1-D Haar. / Separable 4-D transform; bior+Haar (HT), DCT+Haar (Wiener). |
| DC preservation / DC 보존 | Group의 mean coefficient는 임계화하지 않음 — 평균값 보존. / The DC coefficient is not thresholded, preserving the group mean. |
| Rician noise / Rician 잡음 | Magnitude MR signal의 noise 분포: $z = \sqrt{(c_r y + \sigma\eta_r)^2 + (c_i y + \sigma\eta_i)^2}$. / Noise distribution of magnitude MR images. |
| Variance-Stabilising Transform / 분산 안정화 변환 | Rician/Poisson → 근사 Gaussian. BM4D 알고리즘 변경 없이 비-Gaussian noise 처리. / Maps non-Gaussian noise to approximately Gaussian; BM4D unchanged. |
| Iterative reconstruction / 반복 재구성 | $\hat y^{(k+1)} = \mathrm{BM4D}(\hat y^{(k)}, \sigma_k) + \mathcal A^*(z - \mathcal A \cdot)$ — CS-style. / CS-style alternation between BM4D regulariser and data consistency. |
| BrainWeb phantom / BrainWeb 팬텀 | 표준 T1 MR 합성 데이터 ($181 \times 217 \times 181$ voxel). / Standard synthetic T1 MR dataset. |
| Modified profile / 수정 프로파일 | High-noise용 큰 cube ($L=5$) + 큰 group + 큰 threshold. / High-noise variant: larger cubes, groups, thresholds. |
| OB-NLM3D / OB-NLM3D | Coupé+ 2008 block-wise NLM for MR — BM4D의 주요 비교 대상. / Coupé+ 2008 block-wise NLM — main BM4D competitor. |
| Aliasing artifact / 앨리어싱 아티팩트 | k-space subsampling에서 발생하는 wraparound — BM4D-CS가 noise와 함께 제거. / Wraparound artifacts from k-space subsampling, removed jointly with noise. |

---

## 5. 수식 미리보기 / Equations Preview

**볼륨 모델 / Volumetric observation model**:
$$
z(x) = y(x) + \eta(x), \quad x \in X \subset \mathbb Z^3, \quad \eta \overset{iid}{\sim} \mathcal N(0, \sigma^2)
$$

**Photometric cube distance (Eq. 2)**:
$$
d(C^z_{x_i}, C^z_{x_j}) = \frac{\|C^z_{x_i} - C^z_{x_j}\|^2_2}{L^3}
$$

**4-D collaborative hard-thresholding (Eq. 5–6)**:
$$
\hat{\mathbf G}^{ht}_S = \mathcal T^{ht-1}_{4D}\bigl(\Upsilon^{ht}\bigl(\mathcal T^{ht}_{4D}\,\mathbf G^z_S\bigr)\bigr), \quad \mathcal T^{ht}_{4D} = (\text{3-D bior 1.5}) \otimes (\text{1-D Haar})
$$

**Wiener stage (Eq. 10–11)**:
$$
\mathbf W_S = \frac{|\mathcal T^{wie}_{4D} \hat{\mathbf G}^{ht}_S|^2}{|\mathcal T^{wie}_{4D} \hat{\mathbf G}^{ht}_S|^2 + \sigma^2}, \quad \hat{\mathbf G}^{wie}_S = \mathcal T^{wie-1}_{4D}\bigl(\mathbf W_S \cdot \mathcal T^{wie}_{4D}\,\mathbf G^z_S\bigr)
$$

**VST framework for Rician (Eq. 13–14)**:
$$
\hat y = \mathrm{VST}^{-1}\bigl(\mathrm{BM4D}(\mathrm{VST}(z, \sigma), \sigma_{\mathrm{VST}}), \sigma\bigr)
$$

**Iterative reconstruction**:
$$
\hat y^{(k+1)} = \hat y^{(k)}_{\text{filtered}} + \mathcal A^*\bigl(z - \mathcal A\,\hat y^{(k)}_{\text{filtered}}\bigr), \quad \hat y^{(k)}_{\text{filtered}} = \mathrm{BM4D}(\hat y^{(k)}, \sigma_k)
$$

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **§II.A–B (cube + 4-D group)**: BM3D를 *읽었다고 가정*하고 *어떻게 차원이 +1되는지*만 정확히 이해하면 충분. Eq. 2 photometric distance에 *prefilter가 없다*는 점이 BM3D(Eq. 4)와의 미묘한 차이 — 작은 cube + voxel-level smoothness가 보장.
- **§II.B.1 (HT stage)**: *DC term은 임계화하지 않는다*는 한 줄을 놓치지 말 것 — group mean 보존이 핵심.
- **§II.B.2 (Wiener stage)**: BM3D Step 2와 *완전히 동일 구조*. Pilot으로 basic estimate 사용.
- **§II.C (parameters, Table I)**: Normal vs Modified profile의 차이를 정리. high-noise(σ>15%)에서 Modified profile($L=5$, group 32)이 +0.5–1 dB.
- **§III (denoising experiments)**: Table II의 BrainWeb σ=15% 결과 (BM4D 30.82 vs PRI-NLM 28.99)에서 BM4D의 우위 확인. Fig. 4의 시각 비교 — BM4D가 over-smoothing 없이 edge 보존.
- **§III.B (Rician + VST)**: Eq. 13의 Rician PMF와 Eq. 14의 VST 사용 — *BM4D 자체는 수정되지 않음*. VST가 *transparent layer* 역할.
- **§IV (iterative reconstruction)**: 첫 읽기에서는 알고리즘의 *큰 그림*만: BM4D regulariser + data consistency 반복. CS와의 등가성 인지.
- **흔한 오해**: BM4D ≠ V-BM4D. V-BM4D(논문 #9)는 4번째 차원이 *시간*, BM4D는 *비국소*. Spatiotemporal volume 개념은 BM4D에 없음.

### English
- **§II.A–B (cube + 4-D group)**: assuming BM3D is fresh, focus solely on *how a dimension is added*. Note that Eq. 2's photometric distance has *no prefilter* (unlike BM3D Eq. 4) — the small cube + voxel-level smoothness suffices.
- **§II.B.1 (HT stage)**: do not miss the line stating *the DC term is not thresholded* — group-mean preservation is key.
- **§II.B.2 (Wiener stage)**: structurally identical to BM3D Step 2; basic estimate as pilot.
- **§II.C (parameters, Table I)**: organise Normal vs Modified profiles; Modified ($L=5$, group 32) wins +0.5–1 dB at $\sigma > 15\%$.
- **§III (denoising experiments)**: confirm Table II BrainWeb σ=15% (BM4D 30.82 vs PRI-NLM 28.99). Fig. 4: BM4D preserves edges without over-smoothing.
- **§III.B (Rician + VST)**: Eq. 13 Rician PMF, Eq. 14 VST recipe — *BM4D itself is unchanged*; VST is a transparent layer.
- **§IV (iterative reconstruction)**: on first read, grasp only the big picture: BM4D regulariser + data consistency in alternation; equivalent to CS reconstruction.
- **Pitfall**: BM4D ≠ V-BM4D. In V-BM4D (paper #9) the 4th dimension is *time*; in BM4D it is *non-local*. There is no spatiotemporal-volume notion in BM4D.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
BM4D는 (i) MRI/CT denoising의 *non-deep* baseline의 사실상 표준이며, (ii) cryo-EM tomogram, fMRI 시계열, 3-D microscopy(light-sheet, confocal Z-stack), spectral data cube(IFU spectroscopy) 등 *볼륨 과학영상* 전반에서 그대로 사용된다. (iii) Compressed-sensing MRI와 deep PnP-ADMM 프레임워크의 *prior*로 가장 자주 사용되는 비-deep regulariser 중 하나. (iv) VST + BM4D 조합은 *Poisson* 볼륨 영상(저광량 fluorescence 3-D, photon-counting CT)에도 그대로 확장 가능 — 논문 #11(Anscombe)·#14(Mäkitalo-Foi)의 이론으로 보강. (v) 2018+ deep MRI denoisers(DeepMRI, NLRPCA-deep, self-supervised Noise2Noise variants)가 PSNR을 추월했지만, BM4D는 *training-free + closed-form + plug-and-play* 세 가지 장점으로 학습 데이터가 부족하거나 도메인 shift가 큰 임상/과학 환경에서 여전히 표준이다.

### English
BM4D is (i) effectively the standard *non-deep* baseline for MRI/CT denoising; (ii) used as-is across volumetric scientific imaging — cryo-EM tomograms, fMRI time-series, 3-D microscopy (light-sheet, confocal Z-stacks), spectral data cubes (IFU spectroscopy); (iii) one of the most-used non-deep regularisers inside compressed-sensing MRI and deep PnP-ADMM frameworks; (iv) VST + BM4D extends naturally to *Poisson* volumetric imaging (low-light fluorescence 3-D, photon-counting CT), buttressed by the theory of papers #11 (Anscombe) and #14 (Mäkitalo-Foi); (v) although 2018+ deep MRI denoisers (DeepMRI, NLRPCA, self-supervised Noise2Noise variants) surpass it in PSNR, BM4D's *training-free + closed-form + plug-and-play* triad keeps it the default in clinical and scientific settings with scarce training data or large domain shifts.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
