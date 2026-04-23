---
title: "CMEGNets: A self-supervised framework for coronal mass ejection detection & region segmentation"
authors: Besma Guesmi, Jinen Daghrir, David Moloney, Jose Luis Espinosa-Aranda, Elena Hervas-Martin
year: 2026
journal: "Advances in Space Research, Vol. 77, pp. 7455–7483"
doi: "10.1016/j.asr.2026.01.061"
topic: Space_Weather / Modern Space Weather & Machine Learning
tags: [CME, LASCO, self-supervised-learning, SimCLR, U-Net, Mahalanobis, segmentation, anomaly-detection, space-weather-forecasting]
status: completed
date_started: 2026-04-20
date_completed: 2026-04-20
---

# 41. CMEGNets: A Self-Supervised Framework for Coronal Mass Ejection Detection & Region Segmentation / 코로나 질량 방출 탐지 및 영역 분할을 위한 자기지도학습 프레임워크

---

## 1. Core Contribution / 핵심 기여

### 한국어
**CMEGNets**는 SOHO/LASCO C2 코로나그래프 영상에서 **수동 어노테이션 없이** 코로나 질량 방출(CME)을 탐지하고 픽셀 단위로 분할하는 **자기지도학습(Self-Supervised Learning, SSL) 프레임워크**이다. 두 단계로 설계되었다: (1) **SimCLR 기반 대조학습**으로 ResNet-18 백본을 약 230만 장의 라벨 없는 LASCO 데이터에 사전학습하고, (2) **Mahalanobis 거리 기반 이상치 탐지**로 "quiet Sun" 기준선에 대한 통계적 편차를 계산해 **의사 마스크(pseudo-mask)**를 생성한 뒤, 약 1,000장 규모의 soft-label 집합으로 **경량 U-Net 분할 헤드**를 지도학습한다. LASCO C2 벤치마크에서 **CME/비-CME 분류 정확도 99.34%, 분할 Dice 계수 95.33% (테스트 시 98%)**를 달성하며, 수동 어노테이션 비용을 **80% 이상 절감**하고 기존 supervised baseline 대비 **recall 94.7%, faint CME 18% 추가 탐지**, 파라미터 수 **<12M**으로 onboard 위성 배포도 가능하다. CDAW(735 events), SEEDS(550 events) 두 카탈로그에서 평가된 모든 이벤트를 **100% 재현**하였다.

### English
**CMEGNets** is a **self-supervised learning (SSL) framework** that detects and pixel-level segments coronal mass ejections (CMEs) in SOHO/LASCO C2 coronagraph imagery **without any manual annotation**. It comprises two stages: (1) a **SimCLR-based contrastive pre-training** stage that trains a ResNet-18 backbone on ~2.3M unlabelled LASCO frames, and (2) a **Mahalanobis-distance anomaly detector** that computes statistical deviation from a "quiet Sun" baseline to produce **pseudo-masks**, which are then used to fine-tune a **lightweight U-Net segmentation head** on ~1,000 soft-labelled images. On the LASCO C2 benchmark it achieves **99.34% classification accuracy and 95.33% Dice (98% on held-out test sets)**, cuts annotation effort by **over 80%**, recovers **18% more faint CMEs** than a supervised baseline at **94.7% recall**, and fits in **<12M parameters** — small enough for onboard satellite deployment. CMEGNets **100% recovers** all events in the CDAW (735 events) and SEEDS (550 events) catalogues across three solar-maximum intervals.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction & Motivation / 서론 및 동기 (§1, pp. 7455–7456)

#### 한국어
CME는 코로나에서 대량의 플라즈마와 자기장이 방출되는 현상으로, **지자기 폭풍·위성 장애·전력망 정전·항공기 항법 교란·우주비행사 피폭**의 주 원인이다. LASCO 같은 백색광 코로나그래프 영상에서 CME를 자동으로 식별하는 것이 핵심 과제이지만, 다음과 같은 근본적 어려움이 있다.

1. **형태의 다양성**: narrow (<20°), normal (20–120°), partial-halo (120–360°), halo (360°) 네 가지 각폭 분류가 있고, 3-part 구조(bright front / dark cavity / bright core)가 자주 불완전함
2. **관측 잡음**: 산란광(stray light), vignetting, streamer, 우주선(cosmic ray) hit가 edge-detection 기반 알고리즘을 교란
3. **라벨 병목**: 사실상 표준인 **CDAW 카탈로그**(Gopalswamy et al., 2009)는 30년간 사람이 수동 큐레이션 — 노동집약적·관측자 편차(inter-observer variability)·faint/partial-halo 누락 문제

저자는 이 문제를 **라벨 없이 학습 가능한 SSL + 물리기반 이상치 탐지**로 돌파한다. 세 가지 기여를 명시한다:
- (1) Rotation-equivariant contrastive pre-training on unlabelled LASCO C2/C3
- (2) Mahalanobis-distance pseudo-mask generation using a "quiet Sun" baseline
- (3) Lightweight supervised U-Net achieving >99% accuracy & 95% Dice with <12M parameters

#### English
CMEs — massive eruptions of coronal plasma and magnetic flux — drive geomagnetic storms, satellite failures, power-grid outages, aviation navigation disturbances, and astronaut radiation exposure. Automatically identifying CMEs in white-light coronagraph imagery (e.g., LASCO) is the core task, but is hampered by (1) **morphological variability** — narrow/normal/partial-halo/halo classes, often with incomplete 3-part structure (bright front / dark cavity / bright core); (2) **observational noise** — stray light, vignetting, streamers, and cosmic-ray hits that derail edge-detection algorithms; (3) a **labelling bottleneck** — the *de facto* reference **CDAW catalogue** (Gopalswamy et al., 2009) has been hand-curated for 30 years, incurring inter-observer variability and missing faint/partial-halo events.

The authors attack this via **label-free SSL + a physics-guided anomaly detector**, with three named contributions: (1) rotation-equivariant contrastive pre-training on unlabelled LASCO C2/C3; (2) Mahalanobis pseudo-masks from a quiet-Sun baseline; (3) a lightweight supervised U-Net hitting >99% accuracy / 95% Dice with <12M parameters.

---

### Part II: Related Work — Arc from Hand-Crafted to Deep Learning / 수공예 특징에서 딥러닝으로의 전개 (§2, pp. 7456–7457)

| 세대 / Era | 대표 작업 / Representative | 핵심 / Core | 한계 / Limits |
|---|---|---|---|
| Hand-crafted (1990s–2000s) | CACTus (Robbrecht 2009, Hough), SEEDS (Olmedo 2008), ARTEMIS (Boursier 2009), CORIMP (Byrne 2012) | Edge/contour heuristics | faint·partial-halo 누락 |
| ELM/SVM (mid-2010s) | Zhang et al. 2017 | Extreme Learning Machine | manual descriptor 의존 |
| Supervised CNN (2019–) | Nguyen 2019, CAMEL (Wang 2019), CAMEL II (Shan 2024 3D) | CNN on CDAW labels | 라벨 편향 계승 |
| Transformer (2021–) | SegFormer (Xie 2021), TransCME (Yang 2025) | Attention-based seg. | 여전히 manual mask |
| Anomaly/Background (2019–) | Qiang 2019 (background learning) | bkg-deviation flags | label-noise 취약 |
| **SSL (2026, 본 논문)** | **CMEGNets** | **SimCLR + Mahalanobis + U-Net** | **라벨 병목 해결** |

#### 한국어
저자는 특히 **"라벨 의존도"가 최대 차별화 축**임을 강조한다. 기존 CNN/Transformer 방법은 모두 CDAW 라벨에 의존하여 (i) 편향을 상속하고 (ii) 희소·미약한 이벤트에서 일반화 실패가 잦다. CMEGNets의 핵심 차별점은 **pretext task + anomaly prior의 조합**이다.

#### English
The key axis of differentiation is **degree of label dependence**. Every prior CNN/Transformer method inherits CDAW biases and struggles with faint events. CMEGNets escapes this via a **pretext task + anomaly prior** combination.

---

### Part III: Data Pipeline / 데이터 파이프라인 (§3.1, pp. 7457–7458)

#### 한국어
- **데이터 소스**: 1995년 발사 이후 SOHO 아카이브에서 200,000+장의 LASCO C2 이미지 수집. CDAW 카탈로그와 cross-reference하여 CME/non-CME 구분.
- **기간 분리**: 태양 주기별 샘플링
  - 2001-07-01 ~ 2001-09-30 (Cycle 23 최대기)
  - 2019-01 ~ 2020-12 (Cycle 24 감쇠기)
  - 2021-06-01 ~ 2021-11-30 (Cycle 24–25 최소기)
  - 2022-01 ~ 2024-04-30 (Cycle 25 상승/최대 근접)
- **전처리**:
  - **FITS Level-0.5** → JSOC `render_image` (linear stretch btw DATAMIN/DATAMAX) → 512×512 8-bit PNG (Hapgood et al. 1997)
  - **BM3D 디노이징** (Dabov et al. 2007) — readout/photon shot noise 억제
- **Noise 성분 구분** (이해 깊음):
  1. 확률적 detector noise → BM3D로 제거
  2. impulsive 우주선 hit → 제거 대상이나 BM3D 범위 외
  3. structured quasi-static 배경 (F-corona, streamer) → 제거 대상 아님 (별도 모델링 필요)
  4. 기기 artefact (vignetting, stray light) → calibration pipeline에서 처리

#### English
- **Sources**: 200k+ LASCO C2 frames from the SOHO archive (1995–), cross-referenced with CDAW.
- **Temporal sampling**: Cycle 23 max (Jul–Sep 2001), Cycle 24 decline (2019–2020), Cycle 24–25 min (Jun–Nov 2021), Cycle 25 rise/near-max (Jan 2022–Apr 2024).
- **Preprocessing**: FITS Level-0.5 → JSOC `render_image` linear stretch → 512×512 8-bit PNG; **BM3D** denoising to suppress detector/shot noise.
- **Noise taxonomy** (important): (1) stochastic detector noise (addressed by BM3D), (2) cosmic-ray hits (not addressed — flagged as future work), (3) structured backgrounds (F-corona, streamers — explicitly not removed), (4) instrumental artefacts (left to the standard calibration pipeline).

---

### Part IV: CMEGNets Framework / CMEGNets 프레임워크 (§3.2, pp. 7458–7461)

#### 한국어
Fig. 1의 2-stage 구조를 정확히 읽는 것이 핵심이다.

**Stage 1 — Classification pipeline.**
```
Unlabelled LASCO
   → Denoising (BM3D)
   → SimCLR-training on ResNet-18 backbone
   → Head (projection, train-time only)
   → Classification via cosine distance vs references
```

**Stage 2 — Segmentation pipeline.**
```
Test image
   → ResNet-18 (frozen, layers 1–3)
   → Patch features fᵢ ∈ ℝ³
   → Compare to quiet-Sun Gaussian (μᵢ, Σᵢ) via Mahalanobis
   → Per-pixel CME Activity Score
   → Threshold → pseudo-mask
   → Fine-tune U-Net with pseudo-masks
```

**Voting 매커니즘 (Fig. 1 우측).**
테스트 임베딩을 CME/non-CME 레퍼런스 임베딩 집합과 (i) **mean cosine distance**와 (ii) **mean Mahalanobis distance** 두 척도로 비교. 두 점수를 투표로 결합하여 최종 라벨 결정 — 단일 척도의 실패 모드(단일 척도로는 streamer edge 등을 오분류)를 보완.

#### English
Fig. 1 is the critical roadmap.

**Stage 1 (classification)**: Unlabelled LASCO → BM3D denoising → SimCLR-pretrained ResNet-18 → projection head → cosine-distance classification against a reference embedding bank.

**Stage 2 (segmentation)**: Frozen ResNet-18 (layers 1–3) → 3-channel patch feature $f_i$ → Mahalanobis distance to a quiet-Sun Gaussian $(\mu_i, \Sigma_i)$ → activity heat map → threshold pseudo-mask → supervised U-Net on the pseudo-labels.

**Voting** (right side of Fig. 1): The test embedding is compared to CME/non-CME reference banks via **mean cosine** and **mean Mahalanobis** distance; the two are voted to cover each other's failure modes.

---

#### §3.2.1 LASCO Image Classification (SimCLR detail) / 이미지 분류

**한국어.** ResNet-18 백본에 두 가지 feature 추출:
- **Global feature** $h \in \mathbb{R}^{512}$ — layer4 뒤 global average pool
- **Spatial feature map** $7\times7\times512$ — pooling 직전

**Projection head** $g$는 2-layer MLP (512 → 512 → 128) + ReLU. SimCLR 훈련 중에만 사용, 추론 시 버린다.

**NT-Xent loss**는 같은 이미지의 서로 다른 augmentation을 positive pair로, 배치의 나머지를 negative로 밀어낸다:
$$\ell(i,j) = -\log \frac{\exp\bigl(\operatorname{sim}(z_i, z_j)/\tau\bigr)}{\sum_{k=1}^{2N}\mathbf{1}_{[k\neq i]}\exp\bigl(\operatorname{sim}(z_i, z_k)/\tau\bigr)}$$

여기서 $\operatorname{sim}(u,v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$ 는 코사인 유사도, $\tau$는 temperature (=0.5), $2N$은 배치 내 augmented 샘플 총 수.

**훈련 하이퍼파라미터 (§4.2)**:
- Epochs: 50 (cosine annealed over 200), SGD lr=0.06, momentum=0.9, weight decay=5×10⁻⁴, 입력 256×256
- **NN 검색 정성 검증**: 3-NN retrieval이 평균 92% 정확 — 임베딩 품질 확인

**Reference-based 분류** (식 6):
$$d_{\cos}(z_t, z_r) = 1 - \frac{z_t \cdot z_r}{\|z_t\|_2 \|z_r\|_2}$$
$$\text{Classification} = \begin{cases} \text{CME} & \text{if } d_{\cos}(z_t, z_c) \le d_{\cos}(z_t, z_{nc}) \\ \text{non-CME} & \text{otherwise}\end{cases}$$

Reference set: **20 halo + 20 partial-halo + 20 normal = 60 CME + 20 quiet-Sun non-CME frames**.

**Fig. 2 UMAP 검증**: 3D UMAP(metric='cosine')에서 CME와 non-CME가 명확히 분리된 두 cluster로 나타남 — modest overlap은 UMAP 1 ≈ 2–3, UMAP 3 ≈ 8–9 구간에서 관측 (faint/edge 케이스).

**Supervised MLP classifier (§4.2)**: frozen backbone 위에 경량 MLP. 2,800 labelled (1,500 non-CME + 1,300 CME), 224×224, Adam lr=1e-5, 5-fold CV, 최대 83 epoch (early stop patience=10). **평균 정확도 99.34% / F1 99.34% / training loss 0.008**. Cycle 23 최대기 hold-out (720 frame) 평가에서 95.6% 정확도 / 95.4% F1; Cycle 25 상승기 (759 frame)에서 99.34% 정확도/ F1.

**English.** The SimCLR stage uses ResNet-18 with two feature outputs (global $h \in \mathbb{R}^{512}$ and spatial $7\times7\times512$). A 2-layer MLP projection head maps $h \mapsto z \in \mathbb{R}^{128}$ during training only. NT-Xent loss (above) with $\tau=0.5$ pulls augmented positive pairs together and pushes negatives apart. Training: 50 epochs cosine-annealed over 200, SGD lr=0.06, momentum 0.9, weight-decay 5e-4, 256×256 inputs. Qualitative 3-NN retrieval averages 92% accuracy. Reference-based inference uses equation (6) with a 60-CME + 20-quiet-Sun bank. A supervised MLP head on frozen features trained on 2,800 labelled samples yields **99.34% accuracy / F1**. Hold-out tests: **95.6% (Cycle 23 max) / 99.34% (Cycle 25 rise)**.

---

#### §3.2.2 CME Region Detection (Mahalanobis) / CME 영역 탐지

**한국어.** 핵심 통찰: **CME = 정상 분포에 대한 anomaly**로 모델링.

1. **Quiet Sun feature 분포 구축** (non-CME 1,000장):
   - ResNet-18의 첫 3 layer feature 추출 (d=3 channel)
   - 각 패치 위치 $i$ (3×3 pixel 근방)에서 N개 샘플 $\{f_i^j\}$ 수집
   - 다변량 Gaussian fit:
     $$\mu_i = \frac{1}{N}\sum_{j=1}^N f_i^j, \quad \Sigma_i = \frac{1}{N-1}\sum_{j=1}^N (f_i^j - \mu_i)(f_i^j - \mu_i)^\top$$

2. **테스트 프레임의 Mahalanobis 거리**:
   $$d_M(f_{\text{test},i}, \mu_i) = \sqrt{(f_{\text{test},i} - \mu_i)^\top \Sigma_i^{-1} (f_{\text{test},i} - \mu_i)}$$
   거리 큰 패치 = CME 후보.

3. **CME Activity Map 생성**: 패치별 $d_M$을 픽셀 단위로 집계 → heat map → threshold → binary pseudo-mask.

**물리적 직관**: F-corona, K-corona, streamer 등은 "준정상" 구조이므로 Gaussian에 포함됨. CME는 시공간적으로 분포에서 크게 벗어나 anomaly로 포착. 이 방식은 **"quiet Sun"을 모델링 → 편차를 검출**하는 이중 전략: (i) frame-level 분류(CME 존재 여부) + (ii) pixel-level localization(어디?).

**English.** The core insight: **treat CME as anomaly against normal distribution**.
- Build a per-patch multivariate Gaussian $(\mu_i, \Sigma_i)$ from ~1,000 non-CME frames.
- Compute Mahalanobis distance for test patches; large $d_M$ = CME candidate.
- Aggregate into a pixel-wise heat map → threshold → binary mask.

Physically, F-corona, K-corona, and streamers are "quasi-stationary" and absorbed into the Gaussian. A transient CME front deviates sharply — flagged automatically. The approach simultaneously answers "**is there a CME?**" (frame) and "**where?**" (pixel).

---

#### §3.2.3 Pseudo-Labelled Dataset & Supervised U-Net / 의사 라벨 데이터셋과 지도학습 U-Net

**한국어.** SSL 단계에서 생성한 마스크를 "soft-label"로 사용해 12,800+ binary-classification 라벨셋과 ~1,000장 segmentation 라벨셋 (2019-Q1, 2021-Q1 최소기 non-CME + manually verified CME) 구축.

**U-Net 훈련 (§4.3)**:
- 구조: 대칭 encoder–decoder + skip connections (Ronneberger et al. 2015)
- 입력: 256×256 grayscale
- Batch 16, Adam lr=1×10⁻⁵, 80 epochs, 1,000장 훈련셋
- **Mean Dice 98%** on 60+ held-out frames (전체 벤치마크는 95.33%)

**왜 U-Net을 추가하나?** Mahalanobis만으로는 **non-CME 분포만** 모델링하므로 CME의 **명시적 형태 prior가 없어** 넓고 broad한 anomaly map이 나온다. U-Net이 이를 **sharper·morphologically coherent**한 마스크로 정제 — 실시간 배포 시 coherent contour가 필요한 downstream (각폭·속도·leading-edge tracking)에 필수.

**English.** Pseudo-masks from the SSL stage yield ~12,800 classification labels and ~1,000 segmentation labels. A standard U-Net (Ronneberger 2015) on 256×256 greyscale, batch 16, Adam lr=1e-5, 80 epochs → **Dice 98%** on held-out frames. U-Net is necessary because the Mahalanobis approach models **only the non-CME distribution** without a CME shape prior, yielding over-broad anomaly maps; U-Net enforces morphological coherence essential for downstream angular-width/speed estimation.

---

### Part V: Experimental Results / 실험 결과 (§4, pp. 7461–7464)

#### §4.1 Evaluation Metrics / 평가 지표

Table 1 정리 (bilingual):

| Metric | Equation | 의미 / Meaning |
|---|---|---|
| Accuracy | $(TP+TN)/(TP+TN+FP+FN)$ | 전체 정분류율 |
| Precision | $TP/(TP+FP)$ | 예측 CME 중 실제 CME 비율 (FP 측정) |
| Recall (Sensitivity) | $TP/(TP+FN)$ | 실제 CME 중 탐지 비율 (missed event 측정) |
| F1 | $2 \cdot P \cdot R / (P + R)$ | P, R의 조화평균 |
| IoU | $TP/(TP+FP+FN)$ | 분할 overlap 정량 |
| Dice | $2 \cdot TP / (2 \cdot TP + FP + FN)$ | overlap 기반 유사도 (0–1) |
| MDtB | $\frac{1}{N}\sum_i d(\partial M_P, \partial M_g)$ | 예측 mask와 GT mask의 경계 평균 거리 |

#### §4.2 Classification Results / 분류 성능 (Table 2)

| Model | TP | TN | FP | FN | Accuracy | F1 | MDtB |
|---|---|---|---|---|---|---|---|
| LASCO C2 (rising phase, Cycle 25) | 381 | 373 | 2 | 3 | 99.34% | 99.34% | 4e-6 |
| LASCO C2 (Cycle 23 max, 2001) | 362 | 345 | 5 | 8 | 95.33% | 95.40% | 1.5e-5 |

**한국어.** Cycle 25 상승기 대비 Cycle 23 최대기 성능이 4%p 하락 — 높은 배경 난류 환경에서의 **일반화 검증**. 그럼에도 95%대는 유지되며, **faint CME 15% 추가 탐지** 유지.

**English.** Cycle 23-max performance drops ~4 pp vs Cycle 25-rise — a generalization stress test under high background turbulence. Still >95%; recovers 15% more faint events than supervised baseline.

#### §4.3 CME Region Segmentation / 영역 분할

- **Qualitative (Fig. 3)**: 10가지 CME morphology (복합 limb, narrow, normal, halo, partial-halo, multiple) 모두에서 CDAW contour와 높은 일치.
- **Quantitative**: Dice 95% (벤치마크), 98% (held-out 60장 테스트).
- **비교 방법**: CAMEL (LeNet + PCA + graph-cut)은 noisy·경계 부정확; SegFormer (Transformer, ADE20K pretrained)은 domain-specific finetune 없이 faint front 놓침. CMEGNets만 **통합적·coherent** 마스크 생성.

#### §4.4 Baseline Brightness-Thresholding vs CMEGNets (Fig. 4)

**한국어.** 단순 밝기 thresholding은 8개 케이스에서 **3가지 실패 모드**를 일관되게 보임:
1. 밝은 core만 포착, diffuse structure 놓침
2. coherent CME front를 disconnected 조각들로 fragment
3. background feature를 over-segment

CMEGNets의 deep feature + multivariate Gaussian이 local+global context를 결합 → morphologically coherent mask 생성.

**English.** Naive brightness thresholding exhibits three consistent failure modes: (1) captures only bright core, misses diffuse structure; (2) fragments coherent CME fronts; (3) over-segments background features. CMEGNets's deep features + multivariate Gaussian capture local+global context.

#### §4.5 Temporal Consistency (Fig. 5 and Appendix A, Fig. A.5)

**한국어.** 2023-11-22 및 2023-11-24 CME 이벤트에 대해 12분 cadence로 10+ frame의 시간 시리즈 분할을 수행. CMEGNets는 **확장→감쇠** 국면 전반에 걸쳐 coherent mask를 유지하는 반면, thresholding은 faint 후반 단계에서 분절·소실. 이는 **CME 속도·도착 시간 추정**에 필수.

**English.** On CMEs of 2023-11-22 and 2023-11-24 at 12-min cadence (10+ frames), CMEGNets preserves coherent masks through expansion and fade; thresholding fragments late-stage faint emission. Critical for speed and arrival-time estimation.

---

### Part VI: Use Case — Real-Time Space Weather / 실시간 우주기상 활용 (§5, p. 7464)

#### 한국어
- **실시간 LASCO 스트림**에 통합 → 영상 업로드 후 수초 내 frame-level 분류 + preliminary pixel-level seg.
- 산출 mask는 **ENLIL**(Odstreil 2003), **EUHFORIA**(Pomoell & Poedts 2018) 같은 헬리오스피어 전파 코드의 입력으로 활용 가능.
- 파라미터 <12M + 경량 supervised head → **onboard satellite processor** 배포 가능. Frame-wise 구조 → 텔레메트리 중단에도 stateless 복원.
- **운영 분석가 시나리오**: CMEGNets 알림 + mask → ENLIL/EUHFORIA 동적 실행 → 도착 시각·최대 속도 추정 → NOAA/ESA 경보.

#### English
Integrates into real-time LASCO streams with sub-second latency. Masks feed ENLIL (Odstreil 2003) / EUHFORIA (Pomoell & Poedts 2018) for geomagnetic-storm propagation. <12M-param lightweight head enables **onboard deployment**. Frame-wise design is telemetry-interruption resilient. Analyst workflow: alert → mask → ENLIL/EUHFORIA → ETA/peak-speed → NOAA/ESA warning.

---

### Part VII: Discussion & Limitations / 논의 및 한계 (§6, pp. 7465–7467)

#### 한국어
**논문이 명시하는 한계**:
1. **UMAP overlap**: ~UMAP₁=2–3, UMAP₃=8–9 구간의 modest overlap — 미약·occulter 주변·strong streamer 케이스. 정량적 분리도(silhouette 등)는 미측정.
2. **Noise 성분 부분 처리**: BM3D는 stochastic noise만. 우주선, F-corona/streamer, vignetting은 미처리.
3. **Segmentation 평가의 시간 축 편향**: 현 평가는 temporal alignment(±15 min)만 사용, spatial overlap threshold 부재 → streamer-inclusive mask도 true positive로 집계됨.
4. **Shock sheath와 ejecta core 구분 부재**: 현 마스크는 shock와 ejecta body를 하나로 묶음 — 차기 버전에서 **two-stage segmentation**으로 분리 예정.
5. **Catalogue-to-catalogue 비교**: CDAW/SEEDS/CACTus는 event 정의·threshold 상이 → 1:1 matching 대신 **detection coverage**만 평가.
6. **Solar maximum 일반화**: 2001 pilot에서 92.3% recall (rising phase 대비 2.4%p 하락) — 향후 1999–2003 전체로 확장 예정.

**기여 요약 (논문 §7)**:
1. Solar maximum에서 **CDAW 735 + SEEDS 550 = 전체 이벤트 100% recovery** (265 공통)
2. Annotation engine 역할 — 200,000+ labelled 이미지 자동 생성
3. 11M 파라미터의 경량 supervised net으로 99.34% 달성, 라벨의 6%만 사용
4. 다양한 morphology(narrow jet ~ halo) 전반에서 >95% Dice 유지
5. CMEGNets가 720 "hour-bin"의 **추가 CME-positive 시각**을 탐지 (CDAW-only 68 + SEEDS-only 652 구간) → 기존 카탈로그가 놓친 faint/단기 이벤트 시사

#### English
**Stated limitations**: (1) modest UMAP overlap at ~UMAP₁=2–3, UMAP₃=8–9 (faint, occulter-edge, streamer cases; no silhouette score reported); (2) only stochastic noise removed (cosmic-ray, F-corona, vignetting untreated); (3) evaluation uses ±15-min temporal matching only, no spatial overlap threshold; (4) no shock/ejecta distinction — planned for a two-stage follow-up; (5) catalogue-to-catalogue comparison uses detection-coverage only; (6) 92.3% recall at Cycle 23 max (2.4 pp drop vs rising phase).

**Paper conclusion summary**: (1) **100% event recovery** across CDAW (735) + SEEDS (550) with 265 shared; (2) functions as an annotation engine (>200,000 auto-labelled images); (3) 11M-param supervised head at 99.34% accuracy using only 6% of labels; (4) >95% Dice across morphologies; (5) flags 720 additional CME-positive hour-bins beyond each catalogue alone, suggesting missed faint/short-lived events.

---

## 3. Key Takeaways / 핵심 시사점

1. **Label-free 우주기상 ML의 실운영 수준 입증** — CMEGNets는 SSL+anomaly detection 조합이 실제 과학/운영 지표(Dice 95%, Recall 94.7%, Annotation ↓80%)에서 supervised 방법과 동등 이상임을 **처음으로 실증**했다. / **First operational-grade demonstration** that SSL+anomaly detection can match or exceed supervised methods on real metrics in space weather.

2. **"Anomaly = event" prior의 물리적 정당화** — CME는 본질적으로 **정상 코로나 분포에 대한 통계적 편차**이므로 Mahalanobis-distance anomaly detection이 physically motivated. 이는 태양 플레어·오로라·전리층 disturbance 등 **"정상 기준선이 알려진 이벤트"**에 일반화 가능한 템플릿이다. / The **"anomaly = event" prior is physically grounded** — CMEs are statistical deviations from quiescent corona; Mahalanobis is physically motivated and generalizes to flares/auroras/ionospheric disturbances where a quiet baseline is known.

3. **Layer-depth와 task의 매칭** — Classification은 layer-4의 high-level semantic features가, Segmentation은 layer 1–3의 low-level texture features가 최적. **단일 backbone의 다른 depth가 서로 다른 downstream에 사용**되는 설계 패턴은 SSL의 generality를 극대화한다. / **Layer-depth-to-task matching** — layer-4 semantic features serve classification, layer 1–3 low-level texture features serve segmentation — a canonical SSL pattern for reusing one backbone.

4. **Physics-guided 샘플링의 중요성** — quiet-Sun baseline을 **solar-minimum phase (2019-Q1, 2021-Q1)**에서만 샘플링한 것이 핵심. 물리적으로 CME가 희박한 시기를 골라 "normal" 분포를 오염 없이 구축 → anomaly detector의 신뢰도 확보. / **Physics-guided sampling is crucial** — the quiet-Sun baseline is drawn only from solar-minimum intervals to build an uncontaminated "normal" distribution, boosting anomaly-detector reliability.

5. **3-NN retrieval이 SSL 품질의 실용적 지표** — 92% top-3 semantic consistency라는 metric은 linear probing보다 쉽고 빠르며, 여전히 representation quality를 잘 반영한다. 소규모 SSL 프로젝트에서 **저비용 검증 수단**으로 바로 활용 가능. / **3-NN retrieval as a practical SSL-quality proxy** — 92% top-3 semantic consistency is a cheap, fast alternative to linear probing that still tracks representation quality.

6. **Dual classifier architecture (reference + MLP) = robustness** — reference-based(제로샷) + MLP(지도학습) + voting 조합은 각 방법의 약점(reference set 크기/라벨 편향)을 상쇄. 이는 **semi-supervised loop의 실전 설계 패턴**이다. / **Dual-classifier architecture with voting** (zero-shot reference + supervised MLP) neutralizes each method's failure modes — a practical semi-supervised design pattern.

7. **Lightweight + onboard 배포 현실성** — <12M 파라미터 + frame-wise stateless 구조는 Proba-3/Vigil 같은 **차세대 미션의 onboard processor에 즉시 배포 가능**한 수준. ML 모델의 **"compute budget for space-qualified hardware"**를 의식한 설계는 과학 ML의 필수 요건이 되고 있다. / **<12M params + frame-wise stateless design is onboard-deployable**, illustrating that space-ML must account for compute budgets of radiation-hardened processors.

8. **Catalogue 완전성에 대한 패러다임 전환** — CMEGNets가 720 추가 hour-bin을 flag한 것은 **CDAW가 완전한 ground truth가 아니라는 통계적 증거**. 이는 "supervised model vs catalogue"에서 "SSL model ± catalogue 교차검증"으로 **과학 검증 패러다임이 진화**해야 함을 시사한다. / **Paradigm shift for catalogue completeness** — the 720 extra hour-bins statistically show CDAW is *not* complete ground truth; the scientific validation paradigm must move from "supervised vs catalogue" to "SSL ± catalogue cross-validation".

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 SimCLR Pre-training Stack / SimCLR 사전학습 스택

**Feature extraction** (ResNet-18 backbone):
$$h = f_{\text{ResNet-18}}(x), \qquad x \in \mathbb{R}^{256\times256\times1}, \quad h \in \mathbb{R}^{512}$$

**Projection head** (train-time only, 2-layer MLP, discarded after):
$$z = g(h) = W_2\,\sigma(W_1 h + b_1) + b_2, \qquad z \in \mathbb{R}^{128},\ \sigma=\text{ReLU}$$

**Cosine similarity**:
$$\operatorname{sim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

**NT-Xent contrastive loss** (for augmented positive pair $(i,j)$):
$$\ell(i,j) = -\log \frac{\exp\bigl(\operatorname{sim}(z_i, z_j)/\tau\bigr)}{\displaystyle\sum_{k=1}^{2N}\mathbf{1}_{[k\neq i]}\exp\bigl(\operatorname{sim}(z_i, z_k)/\tau\bigr)}$$

**Total loss** (batch of $N$ images, $2N$ augmentations):
$$\mathcal{L}_{\text{SimCLR}} = \frac{1}{2N}\sum_{\text{positive pairs }(i,j)} \bigl(\ell(i,j) + \ell(j,i)\bigr)$$

**Hyperparameters**: $\tau=0.5$, batch of images at 256×256 after BM3D, SGD lr=0.06, momentum=0.9, weight-decay=5×10⁻⁴, 50 epochs (cosine schedule annealed over 200).

---

### 4.2 Reference-Based Classification / 참조 기반 분류

**Cosine distance**:
$$d_{\cos}(z_t, z_r) = 1 - \frac{z_t \cdot z_r}{\|z_t\|_2 \|z_r\|_2}$$

**Decision rule** (논문 식 6의 인과 해석):
$$\hat{y} = \begin{cases} \text{CME} & \text{if}\ d_{\cos}(z_t, z_c) \le d_{\cos}(z_t, z_{nc}) \\ \text{non-CME} & \text{otherwise}\end{cases}$$

where $z_c = \frac{1}{|R_c|}\sum_{r \in R_c} z_r$ is the **mean CME reference embedding** (over 60 frames: 20 halo + 20 partial + 20 normal) and $z_{nc}$ is analogously the mean non-CME embedding (20 quiet-Sun frames).

---

### 4.3 Mahalanobis-Based Anomaly Scoring / Mahalanobis 이상치 스코어

For each spatial patch location $i$ (3×3-pixel area), using the first-3-layer feature $f_i \in \mathbb{R}^3$:

**Gaussian fit on non-CME set** (N ≈ 1,000 frames):
$$\mu_i = \frac{1}{N}\sum_{j=1}^N f_i^j, \qquad \Sigma_i = \frac{1}{N-1}\sum_{j=1}^N (f_i^j - \mu_i)(f_i^j - \mu_i)^\top$$

**Test anomaly score**:
$$d_M(f_{\text{test},i}, \mu_i) = \sqrt{(f_{\text{test},i} - \mu_i)^\top \Sigma_i^{-1}(f_{\text{test},i} - \mu_i)}$$

**CME Activity Map** $\to$ thresholding at $d_M > \theta$ $\to$ binary pseudo-mask $M$.

---

### 4.4 Supervised U-Net Segmentation / 지도학습 U-Net

Training objective combines cross-entropy and Dice terms (standard for segmentation):
$$\mathcal{L}_{\text{U-Net}} = \lambda_1 \mathcal{L}_{\text{CE}} + \lambda_2 (1 - \text{Dice})$$
$$\text{Dice}(A, B) = \frac{2 \lvert A \cap B \rvert}{\lvert A \rvert + \lvert B \rvert}$$

Evaluation metrics:
$$\text{IoU} = \frac{TP}{TP + FP + FN}, \quad \text{Dice} = \frac{2 TP}{2 TP + FP + FN}, \quad \text{MDtB} = \frac{1}{N}\sum_i d(\partial M_P^{(i)}, \partial M_g^{(i)})$$

---

### 4.5 End-to-End Pipeline Diagram / 전체 파이프라인

```
INFERENCE (per LASCO C2 frame):
┌──────────────────────────────────────────────────────────────┐
│  x ∈ ℝ^{512×512}                                             │
│        │                                                      │
│        ▼   BM3D denoise                                       │
│  x̃                                                            │
│        │                                                      │
│        ▼   ResNet-18 (frozen)                                 │
│  h ∈ ℝ^{512}   ┬────────────┐                                │
│        │       │            │                                 │
│        ▼       ▼            ▼                                 │
│  MLP clsf   d_cos vs     layers 1–3 → patch feats            │
│  "CME?"     references   → Mahalanobis vs (μᵢ, Σᵢ)            │
│        │       │            │                                 │
│        └───┬───┘            ▼                                 │
│            │          heatmap → U-Net → mask ∈ {0,1}^{256²}   │
│         Voting                  │                             │
│            ▼                    ▼                             │
│        Label                Binary Mask                      │
│            │                    │                             │
│            └────────────────────┘                             │
│                     ▼                                         │
│     downstream (angular width, speed, ENLIL/EUHFORIA)        │
└──────────────────────────────────────────────────────────────┘
```

---

### 4.6 Worked Numerical Example / 수치 예제

**Scenario**: 2024-04-19 08:48:26 UTC, LASCO C2 frame (known CDAW event).
- After BM3D denoising, ResNet-18 produces $h$ with $\|h\|_2 \approx 11.3$.
- Cosine distance to CME reference mean: $d_{\cos}(z_t, z_c) = 0.18$.
- Cosine distance to non-CME reference mean: $d_{\cos}(z_t, z_{nc}) = 0.67$.
- Decision: $0.18 < 0.67$ → **CME**.
- Patch-wise $d_M$ values in SE quadrant reach 8–15 (vs baseline typical 1–3) → thresholded → connected region of ~3,200 pixels → U-Net refines to smooth contour of 2,700 pixels.
- Compared with CDAW CPA/angular-width: CMEGNets contour matches within ±5° CPA.

(Illustrative; exact numbers as reported in paper Figs. 3–5 and Table 3.)

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1971    OSO-7 최초 CME 관측 / First CME observation
1973    Skylab coronagraph
1995  ──→ SOHO/LASCO 발사 / SOHO launched (C1/C2/C3)
1997    Hapgood et al. — FITS 표준
2004    Yashiro et al. — LASCO CME 통계
2007    Dabov et al. — BM3D
2008    Olmedo et al. — SEEDS (edge-based detection)
2009    Gopalswamy et al. — CDAW 카탈로그 표준화
2009    Robbrecht et al. — CACTus (Hough transform)
2012    Byrne et al. — CORIMP
2013    Vourlidas et al. — CME geoeffectiveness review
2015    Ronneberger et al. — U-Net
2016    He et al. — ResNet
2017    Zhang et al. — ELM on CME
2019    Nguyen et al. — CNN CME detection
2019    Qiang et al. — background-learning anomaly
2019    Wang et al. — CAMEL (CNN + PCA + graph-cut)
2020    Chen et al. — SimCLR (대조학습 표준)
2021    Xie et al. — SegFormer (Transformer seg.)
2022    Pricopi et al. — geoeffectiveness ML
2024    Shan et al. — CAMEL II (3D CNN cataloguing)
2025    Yang et al. — TransCME (CNN-Transformer hybrid)
2026 ★  CMEGNets (본 논문) — label-free SSL + Mahalanobis + U-Net
```

**맥락 / Context.** CMEGNets는 **CDAW(2009) 수동 표준 시대**에서 출발해 **딥러닝 자동화(2019–2025)**를 거쳐 **SSL 자율화(2026)**로 넘어가는 전환점에 위치한다. 이후 과제: (1) multi-view (C3, STEREO/COR, PROBA-3 ASPIICS) 융합, (2) shock/ejecta 분리, (3) downstream kinematic·geoeffectiveness 예측의 end-to-end 학습.

**Context.** CMEGNets sits at the inflection from **manual CDAW standard (2009)** through **supervised deep learning (2019–2025)** to **self-supervised autonomy (2026)**. Open fronts: (1) multi-view fusion (C3, STEREO/COR, PROBA-3 ASPIICS); (2) shock/ejecta separation; (3) end-to-end kinematic/geoeffectiveness prediction.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Ronneberger et al. (2015) — U-Net** | Segmentation head 구조로 직접 차용 / Used directly as the segmentation head. | ★★★★★ CMEGNets Stage 2의 핵심 부품 / Core component of Stage 2. |
| **Chen et al. (2020) — SimCLR** | Pretext contrastive task 방법론 차용 / Methodology of pretext contrastive learning. | ★★★★★ CMEGNets Stage 1의 학습 원리 / Learning principle of Stage 1. |
| **He et al. (2016) — ResNet** | Backbone feature extractor / Backbone architecture. | ★★★★☆ 512-d feature의 출처 / Source of 512-d features. |
| **Gopalswamy et al. (2009) — CDAW 카탈로그** | 벤치마크 참조 라벨 / Benchmark reference labels. | ★★★★★ 검증 기준 (735 events 100% recovery). |
| **Olmedo et al. (2008) — SEEDS** | Benchmark 비교 대상 / Benchmark comparator. | ★★★★☆ 550 events 중 100% recovery. |
| **Robbrecht et al. (2009) — CACTus** | Hand-crafted 자동 탐지의 전통 / Traditional automated detection. | ★★★☆☆ Limit-of-prior-art comparison. |
| **Nguyen et al. (2019) — CNN CME** | 지도학습 CNN의 라벨 병목 사례 / Supervised CNN's label-bottleneck case. | ★★★☆☆ 차별화 축의 baseline / Baseline for label-dependence differentiation. |
| **Wang et al. (2019) — CAMEL** | CNN+PCA+graph-cut 기반 seg. / CNN+PCA+graph-cut segmentation. | ★★★☆☆ 정성 비교 대상. |
| **Xie et al. (2021) — SegFormer** | Transformer seg. 비교 / Transformer seg. comparator. | ★★☆☆☆ 도메인 적응 없이 faint front 놓침. |
| **Shan et al. (2024) — CAMEL II** | 3D CNN cataloguing | ★★★☆☆ 여전히 라벨 의존하는 최근 접근. |
| **Yang et al. (2025) — TransCME** | CNN-Transformer 하이브리드 / Hybrid detector. | ★★★☆☆ Mask 품질 ratchet을 위로 당긴 경쟁자. |
| **Odstreil (2003) — ENLIL** | Downstream 헬리오스피어 전파 모델 / Downstream propagation model. | ★★★★☆ Mask를 입력으로 사용할 수 있는 응용. |
| **Pomoell & Poedts (2018) — EUHFORIA** | Downstream propagation model | ★★★★☆ Operational pipeline 소비자. |
| **Pricopi et al. (2022) — Geoeffectiveness ML** | Geoeffectiveness 예측 downstream / Downstream geoeffectiveness. | ★★★☆☆ CMEGNets output의 잠재 소비자. |
| **Lin et al. (2025, SW #40) — 3D reconstruction** | Dual-viewpoint CME 재구성 / Dual-viewpoint 3D reconstruction. | ★★★☆☆ Multi-view 확장의 자연스러운 파트너. |

---

## 7. References / 참고문헌

- Guesmi, B., Daghrir, J., Moloney, D., Espinosa-Aranda, J. L., & Hervas-Martin, E. (2026). "CMEGNets: A self-supervised framework for coronal mass ejection detection & region segmentation", *Advances in Space Research*, 77, 7455–7483. [DOI: 10.1016/j.asr.2026.01.061]
- Boursier, Y., et al. (2009). "The ARTEMIS catalog of LASCO coronal mass ejections", *Solar Physics*, 257, 125–147.
- Brueckner, G. E., et al. (1995). "The Large Angle Spectroscopic Coronagraph (LASCO)", *Solar Physics*, 162, 357.
- Byrne, J. P., et al. (2012). "Automatic detection and tracking of coronal mass ejections. II. Multiscale filtering of coronagraph images", *ApJ*, 752, 145.
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). "A simple framework for contrastive learning of visual representations (SimCLR)", *ICML*.
- Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. (2007). "Image denoising by sparse 3-D transform-domain collaborative filtering (BM3D)", *IEEE Trans. Image Process.*, 16, 2080.
- Gopalswamy, N., et al. (2009). "The SOHO/LASCO CDAW CME catalog", *Earth Moon Planets*, 104, 295.
- Hapgood, M. A., et al. (1997). "Space physics coordinate transformations: a user guide", *Planet. Space Sci.*, 40, 711.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep residual learning for image recognition (ResNet)", *CVPR*.
- McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection", *arXiv:1802.03426*.
- Odstreil, D. (2003). "Modeling 3-D solar wind structure (ENLIL)", *Adv. Space Res.*, 32, 497.
- Olmedo, O., Zhang, J., Wechsler, H., Poland, A., & Borne, K. (2008). "Automatic detection and tracking of coronal mass ejections in coronagraph time series (SEEDS)", *Solar Physics*, 248, 485.
- Pomoell, J., & Poedts, S. (2018). "EUHFORIA: European heliospheric forecasting information asset", *J. Space Weather Space Clim.*, 8, A35.
- Robbrecht, E., Berghmans, D., & Van der Linden, R. A. M. (2009). "Automated LASCO CME catalog for solar cycle 23: are CMEs scale-invariant? (CACTus)", *ApJ*, 691, 1222.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional networks for biomedical image segmentation", *MICCAI*.
- Shan, H., et al. (2024). "CAMEL II: a 3D CNN cataloguing system for LASCO CMEs", *ApJ Suppl. Ser.*
- Vourlidas, A., Balmaceda, L. A., Stenborg, G., & Dal Lago, A. (2017). "Multi-viewpoint coronal mass ejection catalog based on STEREO COR2 observations", *ApJ*, 838, 141.
- Wang, Z., et al. (2019). "CAMEL: a CNN-based framework for CME automatic detection", *ApJ*, 877, 81.
- Xie, E., et al. (2021). "SegFormer: simple and efficient design for semantic segmentation with Transformers", *NeurIPS*.
- Yang, Y., et al. (2025). "TransCME: a hybrid CNN-Transformer pipeline for LASCO CME segmentation and tracking", *Solar Physics*.
- Yashiro, S., et al. (2004). "A catalog of white light coronal mass ejections observed by SOHO", *JGR: Space Physics*, 109, A07105.
