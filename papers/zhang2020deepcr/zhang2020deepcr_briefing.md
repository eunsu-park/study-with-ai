---
title: "Pre-Reading Briefing: deepCR"
paper_id: "24_zhang_2020"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# deepCR: Cosmic Ray Rejection with Deep Learning / 사전 읽기 브리핑

**Paper**: Zhang, K., Bloom, J. S. "deepCR: Cosmic Ray Rejection with Deep Learning". *Astrophysical Journal (ApJ)*, Vol. 889, No. 1, 24 (2020). DOI: 10.3847/1538-4357/ab3fa6.
**Authors**: Keming Zhang, Joshua S. Bloom
**Year**: 2020

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 **HST ACS/WFC 영상**에서 우주선(CR)을 검출·복원하는 딥러닝 framework **deepCR**를 제안한다. 두 모듈로 구성: (i) **deepCR-mask** — 입력 영상에서 픽셀별 CR 확률 맵을 출력하는 modified U-Net (segmentation), (ii) **deepCR-inpaint** — 마스크된 픽셀을 채우는 또 다른 U-Net. 훈련 데이터는 HST ACS/WFC F606W 16개 visit에서 *AstroDrizzle* median stacking으로 생성된 ground-truth CR mask이다. 0.5% FPR에서 deepCR-2-32은 extragalactic 98.7%, globular cluster 99.5%, resolved galaxy 91.2% TPR을 달성 — L.A.Cosmic(#23)의 69.5%/73.9%/53.4%를 *압도*. 추론 속도도 GPU에서 약 45-90× 빠름. 공개 PyPI 패키지 `deepCR`로 제공되며 inpainting MSE는 L.A.Cosmic 대비 5~20× 우수하다.

### English
The paper presents **deepCR**, a deep-learning framework for cosmic-ray (CR) identification and replacement in HST ACS/WFC imaging, composed of two modules: (i) **deepCR-mask** — a modified U-Net producing a per-pixel CR probability map (image segmentation), (ii) **deepCR-inpaint** — a second U-Net filling pixel values at masked positions. Training data comes from 16 HST ACS/WFC F606W visits, with ground-truth CR masks built via *AstroDrizzle* median-stacking. At 0.5% false-positive rate, deepCR-2-32 reaches **98.7%/99.5%/91.2% TPR** across extragalactic / globular cluster / resolved galaxy fields — versus L.A.Cosmic's 69.5%/73.9%/53.4% (paper #23). It is also ~45-90× faster on a GPU. The framework ships as the open-source PyPI package `deepCR`, with inpainting MSE 5–20× better than the best non-neural baseline.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
2001년부터 2019년까지 L.A.Cosmic(논문 #23)이 HST 단일 노출 CR 제거의 표준이었으나, 그 *수작업 통계 feature* (Laplacian + fine-structure ratio)에는 한계가 있었다 — 특히 resolved galaxy 같은 복잡 구조의 영상에서 false positive 폭증. 한편 2015년 U-Net의 등장으로 의료 영상 segmentation에서 CNN이 전통 알고리즘을 대체했으며, 2018년 Sedaghat-Mahabal이 천문학에서도 U-Net을 transient detection에 처음 사용했다. deepCR은 *L.A.Cosmic의 통계 feature를 학습된 CNN feature로 대체*하면 어떤 일이 일어나는지를 보여주는 결정적 사례다 — 답은 30+ percentage point 개선이다.

#### English
From 2001 to 2019, L.A.Cosmic (paper #23) was the standard for HST single-exposure CR rejection, but its *hand-crafted statistical features* (Laplacian + fine-structure ratio) faltered on complex fields, especially resolved galaxies where false positives exploded. Meanwhile, U-Net (2015) was sweeping medical-image segmentation, replacing classical algorithms; in 2018 Sedaghat & Mahabal first used U-Net in astronomy for transient detection. deepCR is the decisive case showing what happens when L.A.Cosmic's statistical features are replaced by *learned* CNN features — a 30+ percentage-point improvement.

### 타임라인 / Timeline

```
1995  Salzberg — decision-tree CR classifier (early ML)
2000  Rhoads — linear-filter CR detection
2001  van Dokkum L.A.Cosmic (#23) — 19-year standard
2009  Deng+ — ImageNet (deep-learning era)
2015  Ronneberger+ — U-Net (medical-image segmentation)
2017  Shelhamer+ — Fully Convolutional Networks
2018  Sedaghat-Mahabal — U-Net for transient detection in astronomy
2020 ★ Zhang & Bloom — deepCR: U-Net beats L.A.Cosmic by 30+ pp
2020+ deepCR adopted by HST/JWST/Roman pipelines as supplement
2022+ Hybrid classical/deep CR pipelines (deepCR + L.A.Cosmic ensemble)
```

---

## 3. 필요한 배경 지식 / Prerequisites

#### 한국어
- **L.A.Cosmic 알고리즘** (논문 #23) — deepCR이 추월하려는 baseline
- **U-Net 아키텍처** (Ronneberger 2015): encoder-decoder + skip connection
- **Image segmentation**, binary cross-entropy loss
- **Image inpainting**, MSE loss
- **AstroDrizzle pipeline** (HST 표준 mosaicking): multi-exposure CR mask 생성
- **HST ACS/WFC** 검출기 specs (read-noise, gain, F606W 필터)
- **ROC curve** 평가 (TPR vs FPR)
- **Data augmentation**: noise scaling, mask sampling
- **PyTorch / segmentation 모델 학습** 일반 흐름
- **천문학적 CR 특성**: HST trapped-proton, terrestrial muon, instrumental cosmic ray

#### English
- L.A.Cosmic algorithm (paper #23) — the baseline deepCR aims to surpass.
- U-Net architecture (Ronneberger 2015): encoder-decoder + skip connections.
- Image segmentation and binary cross-entropy loss.
- Image inpainting and MSE loss.
- AstroDrizzle pipeline (HST standard mosaicking) for multi-exposure CR masks.
- HST ACS/WFC detector specs (read-noise, gain, F606W filter).
- ROC curve evaluation (TPR vs FPR).
- Data augmentation: noise scaling, mask sampling.
- General PyTorch / segmentation-model training pipeline.
- Astronomical CR phenomenology: HST trapped protons, terrestrial muons, instrumental events.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **deepCR-mask** | CR 확률 맵을 출력하는 segmentation U-Net / Segmentation U-Net producing per-pixel CR probability. |
| **deepCR-inpaint** | 마스크된 픽셀 값 복원하는 U-Net / Inpainting U-Net filling masked pixels. |
| **deepCR-D-N** | depth $D$, base channel $N$ 변형 표기 / Variant naming: depth $D$, base channels $N$. |
| **AstroDrizzle** | HST 다중 노출 mosaicking 도구 (ground-truth 생성) / HST multi-exposure tool used for ground-truth masks. |
| **F606W** | HST ACS/WFC 광대역 V-band 필터 / HST ACS/WFC broad V-band filter. |
| **ROC curve** | TPR vs FPR plot, threshold 변경에 따라 / TPR vs FPR plot across thresholds. |
| **TPR / FPR** | true/false positive rate / True / false positive rate. |
| **Modified U-Net** | boundary 보존을 위해 표준 U-Net 수정 / U-Net modified to preserve boundary pixels. |
| **Inpainting mask $M_I$** | 인페인팅 손실을 적용할 영역 / Region where inpainting loss is computed. |
| **Bad-pixel / saturation mask** | backprop 제외 영역 / Regions excluded from backprop. |
| **Exposure-time augmentation** | sky scaling으로 짧은 노출 시뮬레이션 / Sky-background scaling to simulate variable exposure. |
| **PyPI `deepCR`** | 공개 패키지 — `pip install deepCR` / Open-source PyPI package. |

---

## 5. 수식 미리보기 / Equations Preview

### 한국어
**픽셀 모델 (영상 형성)**:

$$
n = (f_{\rm star} + f_{\rm sky})\cdot t_{\rm exp} + n_{\rm CR}
$$

**Mask loss (Eq. 1)** — binary cross-entropy:

$$
\mathcal L_{\rm F} = \mathbb E\big[M\log(1 - F(X)) + (1-M)\log F(X)\big]
$$

**Inpaint loss (Eq. 2)** — MSE on inpainting mask only:

$$
\mathcal L_{\rm G} = \mathbb E\Big[\big(G(X, M_I)\odot M_I\odot (1-M) - X\odot M_I\odot (1-M)\big)^2\Big]
$$

**Exposure-time augmentation (Eqs. 3-4)**:

$$
n' = n + \alpha f_{\rm sky}\,t_{\rm exp} = \!\left(\frac{f_{\rm star}}{1+\alpha} + f_{\rm sky}\right)\!(1+\alpha)\,t_{\rm exp} + n_{\rm CR}
$$

**추론 파이프라인** — mask → inpaint chain:

$$
X_{\rm clean} = X\odot(1-M_{\rm pred}) + G_\phi(X\odot(1-M_{\rm pred}), M_{\rm pred})\odot M_{\rm pred}
$$

### English
The pixel model captures the additive nature of source flux, sky background, and CR contribution. The mask loss is standard binary cross-entropy with the convention $F\to 1$ for non-CR pixels (so $1-F(X)$ in the first log). The inpaint loss is MSE computed only inside the inpainting mask AND outside actual CRs ($1-M$ factor), since CR pixels have no clean ground truth. Exposure-time augmentation rescales sky background to mimic shorter/longer exposures during training. Inference chains the two modules: mask → zero out CRs → inpaint → composite.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **꼭 정독할 부분**: §2 (deepCR-mask와 deepCR-inpaint의 분리 학습 결정과 modified U-Net 이유), §3 (AstroDrizzle ground-truth 생성, 데이터 증강 Fig. 2), §4 ROC 결과 (특히 Table 2 0.05% / 0.5% FPR 비교).
- **빠르게 훑을 부분**: §1 introduction 일부, §3 Table 1 visit 목록 세부.
- **흔한 걸림돌 / Common stumbling blocks**:
  - "왜 mask와 inpaint를 분리 학습하는가?" — 두 작업은 본질적으로 다르며 (분류 vs 회귀), 분리하면 각자 최적 손실(BCE vs MSE)을 사용 가능.
  - "Eq. 2에서 $(1-M)$은 왜 필요한가?" — CR 픽셀은 ground-truth 픽셀 값이 없으므로 (CR이 가린 게 GT) 손실에서 제외.
  - "왜 weighted L1 대신 MSE를 채택했나?" — weighted L1은 sky-background 픽셀(전체의 90%+)에 과도한 gradient를 주어 *어려운* 별/은하 픽셀이 학습되지 않음.
  - "Boundary 처리": 표준 U-Net은 92픽셀 잘라냄, deepCR은 *입력=출력 크기*가 되도록 수정 — 천문학에서 boundary 데이터 손실은 받아들일 수 없음.
  - "왜 $\sigma_{\rm rn}$, gain 같은 노이즈 모델이 명시적으로 안 들어가는가?" — 학습 데이터에서 학습됨; supervised 방식.
- 동반 자료: Ronneberger U-Net 원논문, AstroDrizzle 문서, deepCR PyPI README.

### English
- **Read carefully**: §2 (decoupled training of deepCR-mask and deepCR-inpaint, plus the modified U-Net rationale), §3 (AstroDrizzle ground-truth construction, data augmentation Fig. 2), §4 ROC results — especially Table 2 at 0.05% / 0.5% FPR.
- **Skim**: parts of §1, §3 Table 1 visit list.
- **Common stumbling blocks**:
  - Why mask and inpaint are decoupled — segmentation vs reconstruction are fundamentally different tasks; decoupling lets each use its optimal loss (BCE vs MSE).
  - Why $(1-M)$ appears in Eq. 2 — CR pixels have no clean GT (the CR replaced it), so they must be excluded from the loss.
  - Why MSE beats weighted L1 — weighted L1 over-weights sky-background pixels (>90% of total), leaving harder star/galaxy pixels under-learnt.
  - Boundary handling — standard U-Net crops 92 boundary pixels; deepCR modifies the architecture to keep input=output size since astronomy can't discard boundary data.
  - Why noise model parameters ($\sigma_{\rm rn}$, gain) don't appear explicitly — they're absorbed into the learned weights via supervised training.
- Companion reading: Ronneberger U-Net paper, AstroDrizzle documentation, deepCR PyPI README.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
deepCR은 **천문학 영상 처리에서 deep learning이 19년 표준 알고리즘(L.A.Cosmic)을 결정적으로 추월**한 대표 사례다. PyPI 공개와 사전학습 모델 제공으로 즉시 채택 가능 — `pip install deepCR`로 한 줄 적용. JWST·Roman·Euclid·LSST 같은 차세대 망원경 파이프라인에서도 deepCR-style 모델이 검토·통합되고 있으며, 일부는 *deepCR + L.A.Cosmic ensemble*로 두 알고리즘의 장점을 결합한다. 더 넓게는 — 천문학의 *전통적 통계 feature → 학습된 deep feature* 전환을 가속화한 작품이다. self-supervised 디노이저(논문 #16-22)와 직접 비교하면, deepCR은 *supervised* (AstroDrizzle ground-truth 사용) 방식이라는 점에서 다르다. 향후 self-supervised CR detection (예: Noise2Noise을 이용한 이중 노출 학습)이 새로운 연구 방향으로 떠오르고 있다.

### English
deepCR is the canonical case of **deep learning decisively surpassing a 19-year standard algorithm (L.A.Cosmic) in astronomical image processing**. Its PyPI release with pretrained models enables drop-in adoption — one line of `pip install deepCR`. Next-generation pipelines (JWST, Roman, Euclid, LSST) are integrating deepCR-style models, with some combining them as *deepCR + L.A.Cosmic ensembles* to merge strengths. More broadly, deepCR accelerated astronomy's transition from *hand-crafted statistical features → learned deep features*. Compared with the self-supervised denoising track (papers #16-22), deepCR is *supervised* (uses AstroDrizzle ground-truth masks). Self-supervised CR detection — e.g., Noise2Noise-style training on double exposures — is an emerging research direction inspired partially by this paper's success.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
