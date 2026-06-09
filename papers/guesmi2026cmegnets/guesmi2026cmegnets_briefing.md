---
title: "Pre-Reading Briefing: CMEGNets — A Self-Supervised Framework for CME Detection & Region Segmentation"
paper_id: "41_guesmi_2026"
topic: Space_Weather
date: 2026-04-20
type: briefing
---

# CMEGNets: A Self-Supervised Framework for Coronal Mass Ejection Detection & Region Segmentation — Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Guesmi, B., Daghrir, J., Moloney, D., Espinosa-Aranda, J. L., & Hervas-Martin, E. (2026). *CMEGNets: A self-supervised framework for coronal mass ejection detection & region segmentation*. **Advances in Space Research, 77**, 7455–7483. https://doi.org/10.1016/j.asr.2026.01.061
**Author(s)**: Besma Guesmi, Jinen Daghrir, David Moloney, Jose Luis Espinosa-Aranda, Elena Hervas-Martin (Ubotica Technologies; DCU Alpha, Dublin; Universidad de Castilla-La Mancha, Ciudad Real)
**Year**: 2026

---

## 1. 핵심 기여 / Core Contribution

### 한국어
CMEGNets는 LASCO C2/C3 코로나그래프 영상에서 **사람 손으로 만든 마스크 없이** 코로나 질량 방출(CME)을 **탐지(classification)하고 픽셀 단위로 분할(segmentation)하는 자기지도학습(Self-Supervised Learning, SSL) 프레임워크**이다. 파이프라인은 두 단계로 구성된다. (1) **SimCLR 기반 대조학습**으로 약 230만 장의 라벨 없는 LASCO 데이터에서 ResNet-18 백본을 사전학습하여 인스턴스 판별(instance discrimination) 능력을 학습하고, (2) **Mahalanobis 거리 기반 의사 마스크(pseudo-mask)** 생성 — "quiet Sun" 기준선에 대한 통계적 이상치(anomaly)를 탐지 — 로 경량 U-Net 분할 헤드를 준지도(semi-supervised) 방식으로 미세조정한다. LASCO C2 벤치마크에서 **CME/비-CME 분류 99% 정확도, 분할 95% Dice 계수**를 달성하였고 수동 어노테이션 비용을 **80% 이상 절감**한다. CDAW·CACTus·SEEDS 같은 기존 수동 카탈로그를 **대체할 수 있는 실시간 우주기상 파이프라인의 원형**이다.

### English
CMEGNets is a **self-supervised learning (SSL) framework** that **detects and pixel-segments coronal mass ejections (CMEs) in LASCO C2/C3 coronagraph imagery without any hand-drawn masks**. The pipeline has two stages: (1) a **SimCLR contrastive pre-training** stage that trains a ResNet-18 backbone on ~2.3M unlabelled LASCO frames to learn instance discrimination, and (2) a **Mahalanobis-distance pseudo-mask generator** — flagging statistical anomalies against a "quiet Sun" baseline — that supervises a lightweight U-Net segmentation head in a semi-supervised loop. On the LASCO C2 benchmark, CMEGNets achieves **99% classification accuracy (CME vs non-CME) and a 95% Dice coefficient for segmentation**, while cutting annotation effort by **over 80%**. It is a prototype for **real-time space-weather pipelines** that can supplant expert-curated catalogues like CDAW, CACTus, and SEEDS.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어.** CME는 1970년대 Skylab/OSO-7 코로나그래프로 처음 포착된 이래 우주기상의 핵심 관측 대상이 되었다. 1995년 SOHO 위성의 **LASCO (Large Angle Spectrometric Coronagraph)** 발사 이후 C1/C2/C3 세 시야각에서 30년에 걸쳐 200,000장 이상의 백색광 영상이 축적되었고, 이로부터 NASA/가톨릭대학교가 공동 운영하는 **CDAW CME 카탈로그**(Gopalswamy et al., 2009)가 사실상 표준 참조 자료가 되었다. 그러나 CDAW는 **사람이 일일이 프레임을 검토하여 CME를 잡아내는 방식**이므로 (i) 노동집약적이고 (ii) 관측자 간 편차(inter-observer variability)가 크며 (iii) 미약하거나 부분 할로(partial-halo) CME를 놓치기 쉽다. 이에 따라 1990년대 말부터 자동 탐지 알고리즘이 개발되었다 — **CACTus** (Robbrecht et al., 2009, Hough 변환 기반), **SEEDS** (Olmedo et al., 2008, 분할 기반), **ARTEMIS** (Boursier et al., 2009), **CORIMP** (Byrne et al., 2012). 2010년대 딥러닝의 등장으로 CNN·U-Net·Transformer 기반 방법(Nguyen 2019, Shan 2024의 CAMEL II, Yang 2025의 TransCME 등)이 대거 등장했으나, 모두 **CDAW 라벨에 의존**하여 라벨 편향(label bias)과 희소 이벤트 문제를 피할 수 없었다. CMEGNets는 이 **"라벨 병목"을 SSL·대조학습·이상치 탐지 기법으로 돌파**하려는 2025–2026년 시점의 시도이다.

**English.** CMEs have been a flagship space-weather target since their first detection by Skylab/OSO-7 coronagraphs in the 1970s. After **LASCO** (Large Angle Spectrometric Coronagraph) launched on SOHO in 1995, its C1/C2/C3 detectors have accumulated 30+ years of ~200,000 white-light frames, spawning the **CDAW CME catalogue** (Gopalswamy et al., 2009) — the *de facto* reference maintained by NASA and the Catholic University of America. But CDAW is **manually curated frame-by-frame**: (i) labor-intensive, (ii) subject to inter-observer variability, and (iii) prone to missing faint or partial-halo CMEs. Automated detectors since the late 1990s — **CACTus** (Hough-transform based), **SEEDS** (segmentation based), **ARTEMIS**, **CORIMP** — partially alleviated this. The deep-learning wave of the 2010s produced CNN/U-Net/Transformer pipelines (Nguyen 2019; Shan's CAMEL II 2024; Yang's TransCME 2025), but all **lean on CDAW labels** and inherit their biases. CMEGNets (2025–2026) represents the attempt to **break the labelling bottleneck** using SSL, contrastive learning, and anomaly detection.

### 타임라인 / Timeline

```
1971    OSO-7 최초 CME 관측 / First CME detected by OSO-7
1973    Skylab 코로나그래프
1995    SOHO/LASCO 발사 / SOHO/LASCO launched (C1/C2/C3)
2004    Yashiro et al. — LASCO CME 통계
2008    Olmedo et al. — SEEDS 자동 탐지
2009    Gopalswamy et al. — CDAW 카탈로그
2009    Robbrecht et al. — CACTus
2012    Byrne et al. — CORIMP
2015    Ronneberger et al. — U-Net (바이오-의료 분할의 표준)
2019    Nguyen et al. — CNN 기반 CME 분류
2020    Chen et al. — SimCLR (대조학습의 이정표)
2022    Pricopi et al. — CME geoeffectiveness ML
2024    Shan et al. — CAMEL II (3D CNN)
2025    Yang et al. — TransCME (Transformer)
2026 *  CMEGNets (본 논문) — SSL로 라벨 의존성 제거
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
1. **태양물리 / 우주기상 기초**
   - 태양 코로나 구조, K-corona의 Thomson 산란, 코로나그래프(occulter, F/K/E-corona) 원리
   - CME의 3단계 형태(bright front, dark cavity, bright core)와 각폭(angular width) 분류(narrow <20°, normal 20–120°, partial halo 120–360°, halo 360°)
   - LASCO C1(1.1–3 R⊙), C2(1.5–6 R⊙), C3(3.7–30 R⊙) 시야각 차이
   - CDAW / CACTus / SEEDS 카탈로그의 목적과 한계
2. **딥러닝 기반 지식**
   - CNN의 합성곱·풀링·활성화 구조, ResNet 잔차 연결
   - **U-Net** 인코더-디코더 스킵 연결 구조 (Ronneberger et al., 2015)
   - Softmax, cross-entropy, Dice loss, IoU/Dice 평가지표
3. **자기지도학습 (SSL) / 대조학습**
   - **SimCLR**(Chen et al., 2020): positive/negative pair, data augmentation, projection head, **NT-Xent loss**
   - 임베딩 공간의 코사인 유사도·거리 개념
   - 표현학습(representation learning)과 downstream task의 관계
4. **통계·이상치 탐지**
   - 다변량 가우시안과 **Mahalanobis 거리** $D_M(x,\mu)=\sqrt{(x-\mu)^{\top}\Sigma^{-1}(x-\mu)}$
   - 커널 밀도 추정, anomaly score 개념
5. **영상 처리 / FITS 파이프라인**
   - FITS Level-0.5 → PNG 변환(JSOC `render_image`), linear stretch, DATAMIN/DATAMAX
   - BM3D(Block-Matching 3D) 노이즈 제거
   - UMAP 차원 축소(2D/3D 시각화)

### English
1. **Solar & space-weather basics** — corona morphology, Thomson-scattered K-corona, coronagraph design (occulter, F/K/E components); three-part CME morphology (bright front / dark cavity / bright core); angular width classes (narrow, normal, partial halo, halo); LASCO C1/C2/C3 fields of view (1.1–3 / 1.5–6 / 3.7–30 R⊙); purpose and limits of CDAW/CACTus/SEEDS.
2. **Deep-learning fundamentals** — CNN layers, ResNet residual connections, **U-Net** encoder–decoder with skip connections, cross-entropy/Dice loss, IoU/Dice metrics.
3. **Self-supervised & contrastive learning** — **SimCLR**: positive/negative pairs, data augmentations, projection head, **NT-Xent loss**; cosine similarity in embedding space; the link between representation learning and downstream tasks.
4. **Statistics & anomaly detection** — multivariate Gaussian and **Mahalanobis distance** $D_M(x,\mu)=\sqrt{(x-\mu)^{\top}\Sigma^{-1}(x-\mu)}$; anomaly scoring.
5. **Image processing / FITS pipeline** — FITS Level-0.5 → 8-bit PNG via JSOC `render_image`, linear stretch between DATAMIN/DATAMAX; **BM3D** denoising; UMAP for 2D/3D visualisation.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **CME (Coronal Mass Ejection)** | 태양 코로나에서 plasma와 자기장이 대량으로 분출되는 현상. 지구 자기폭풍의 주원인. / Massive eruption of plasma and magnetic flux from the corona; the main driver of geomagnetic storms. |
| **LASCO C2 / C3** | SOHO 위성의 백색광 코로나그래프(1.5–6 R⊙ / 3.7–30 R⊙ 시야). / White-light coronagraphs onboard SOHO covering 1.5–6 R⊙ and 3.7–30 R⊙ fields of view. |
| **CDAW Catalogue** | NASA/CUA가 공동 운영하는 수동 큐레이션 CME 카탈로그 — 사실상 표준 참조. / Manually curated CME catalogue — the *de facto* reference. |
| **Halo / Partial-halo CME** | 오컬터 디스크를 360°/120–360° 감싸는 CME; 지구 방향 폭발 가능성이 큼. / CME whose emission surrounds the occulter 360° / 120–360°; likely Earth-directed. |
| **SSL (Self-Supervised Learning)** | 라벨 없이 pretext task로 표현을 학습하는 방법. / Learning representations without labels via pretext tasks. |
| **SimCLR / NT-Xent Loss** | 같은 이미지의 서로 다른 augmentation을 positive pair로 묶고 나머지는 negative로 밀어내는 대조학습 프레임워크 및 정규화된 temperature-scaled cross-entropy 손실. / Contrastive framework pulling augmented views of the same image together, pushing others apart, optimised by normalised temperature-scaled cross-entropy. |
| **Pseudo-Mask / Pseudo-Label** | 모델이 자체 생성한 가짜 라벨 — 수동 라벨 대신 학습 신호로 사용. / Model-generated labels used in place of manual annotations. |
| **Mahalanobis Distance** | 공분산을 반영한 다변량 거리 — "정상(quiet Sun)" 분포로부터의 이상도를 측정. / Covariance-aware multivariate distance — used here to score anomalousness against a quiet-Sun baseline. |
| **U-Net** | 인코더·디코더·스킵 연결로 픽셀 단위 분할을 수행하는 표준 CNN. / Standard CNN for pixel-level segmentation with encoder/decoder and skip connections. |
| **Instance Discrimination** | 각 이미지를 고유 클래스처럼 취급하여 임베딩을 학습하는 pretext task. / Pretext task treating each image as its own class for embedding learning. |
| **Dice Coefficient** | 분할 평가지표, $\mathrm{Dice}=\tfrac{2\lvert A\cap B\rvert}{\lvert A\rvert+\lvert B\rvert}$, 0–1 범위. / Segmentation metric, 0–1. |
| **BM3D** | Block-Matching 3D 필터링 — 코로나그래프 노이즈 제거에 사용. / State-of-the-art denoising used to clean LASCO C2 frames. |

---

## 5. 수식 미리보기 / Equations Preview

### (1) ResNet-18 백본의 글로벌 feature 추출 / Global feature extraction
$$h = f_{\text{ResNet-18}}(x), \quad x \in \mathbb{R}^{H \times W \times C},\ h \in \mathbb{R}^{512}$$
입력 이미지 $x$ (512×512 grayscale, $C=1$)를 ResNet-18로 통과시켜 512차원 semantic feature vector $h$를 얻는다. 또한 pooling 직전의 $7\times7\times512$ spatial feature map도 보관하여 segmentation에 재사용.
Pass input $x$ (512×512 grayscale) through ResNet-18; extract a 512-d semantic vector $h$ and a $7\times7\times512$ spatial map for downstream segmentation.

### (2) Projection head / 투영 헤드
$$z = g(h) \in \mathbb{R}^{128}$$
2-layer MLP (ReLU + normalize)로 feature를 128차원 임베딩 $z$로 투영. 학습 후에는 버리고 $h$만 사용한다.
A 2-layer MLP with ReLU projects features into a 128-d embedding $z$; discarded after training — only $h$ is used at inference.

### (3) 코사인 유사도 / Cosine similarity
$$\operatorname{sim}(z_0, z_1) = \frac{z_0 \cdot z_1}{\lVert z_0\rVert_2 \, \lVert z_1\rVert_2}$$
두 임베딩 간 각도 기반 유사도. SimCLR 학습의 핵심 척도.
Angle-based similarity between embeddings — the central measure for SimCLR training.

### (4) NT-Xent contrastive loss
$$\ell(i,j) = -\log \frac{\exp\bigl(\operatorname{sim}(z_i, z_j)/\tau\bigr)}{\sum_{k=1}^{2N}\mathbf{1}_{[k\neq i]}\exp\bigl(\operatorname{sim}(z_i, z_k)/\tau\bigr)}$$
$\tau$는 temperature, $N$은 batch size, $2N$은 augmented pair 포함 총 샘플 수. positive pair $(i,j)$를 가깝게, 나머지를 멀리 민다.
NT-Xent loss — pulls positive pair $(i,j)$ together, pushes others apart, normalised by temperature $\tau$ across a batch of $2N$ augmented samples.

### (5) 코사인 거리 (CME vs non-CME 판단) / Cosine distance for classification
$$d_{\cos}(z_t, z_r) = 1 - \frac{z_t \cdot z_r}{\lVert z_t\rVert_2 \lVert z_r\rVert_2}$$
$$\text{Classification} = \begin{cases} \text{CME} & \text{if } d_{\cos}(z_t, z_c) > d_{\cos}(z_t, z_{nc}) \\ \text{non-CME} & \text{otherwise}\end{cases}$$
(주의: 논문 식 (6)의 부등호 방향은 원문 그대로 유지 — 가까운 쪽이 같은 클래스여야 하므로, 실제 판정은 $d$가 **작은** 쪽을 선택해야 함. 읽으며 확인 필요)
Test embedding $z_t$ is compared to reference embeddings $z_c$ (CME) and $z_{nc}$ (non-CME); the test frame is classified by the nearer reference. *(Verify inequality direction while reading.)*

### (6) Mahalanobis 거리 기반 이상치 탐지 / Anomaly scoring for pseudo-masks
$$D_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x}-\boldsymbol{\mu})^{\top}\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}$$
quiet Sun 분포 $(\mu, \Sigma)$로부터의 거리가 큰 픽셀/패치가 CME 후보.
Pixels/patches far from the quiet-Sun distribution (mean $\mu$, covariance $\Sigma$) are flagged as CME candidates → becomes the pseudo-mask.

### (7) Dice 계수 / Dice coefficient (segmentation metric)
$$\mathrm{Dice}(A, B) = \frac{2\lvert A \cap B\rvert}{\lvert A\rvert + \lvert B\rvert}$$
예측 마스크 $A$와 정답 마스크 $B$의 겹침 비율. 논문에서 **0.95** 달성.
Overlap fraction between predicted and ground-truth masks — reported at **0.95**.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
1. **Abstract + §1 Introduction** — CME 탐지 문제와 라벨 병목, CMEGNets가 어떻게 이를 돌파하는지 세 가지 기여(SSL backbone, Mahalanobis pseudo-mask, lightweight U-Net)를 머릿속에 고정하라. 핵심 숫자: **99% 정확도, 95% Dice, 80% 어노테이션 절감, 94.7% recall, 18% 더 많은 faint CME 탐지**.
2. **§2 Related Work** — CACTus/SEEDS/ARTEMIS/CORIMP (전통) → Nguyen, CAMEL, TransCME (딥러닝) 흐름을 정리. 본 논문이 **라벨 의존 vs SSL** 축에서 차별화됨을 파악.
3. **§3.1 Data** — LASCO C2 이미지 200,000+ 장, FITS→PNG 변환·BM3D 노이즈 제거 파이프라인. **Solar Cycle 23/24/25에 걸친 시간대** 샘플링 전략에 주목 (Jul 2001 / 2019–2020 / Jun–Nov 2021 / Jan 2022–Apr 2024).
4. **§3.2 Framework** — Fig. 1 다이어그램을 먼저 파악. (1) binary classifier(frame-level) → (2) segmentation(pixel-level) 2단계 구조. Fig. 2의 UMAP 시각화로 CME/non-CME 임베딩 분리 확인.
5. **§3.2.1 LASCO image classification** — SimCLR 설정, ResNet-18 backbone, projection head, NT-Xent loss, reference set(halo 20 + partial 20 + normal 20 / quiet-Sun non-CME set)의 구성 방식.
6. **§3.2.2 ~ § Segmentation** — Mahalanobis pseudo-mask 생성 로직, small expert-verified set에 의한 U-Net fine-tuning, semi-supervised loop의 반복 구조.
7. **§ Experiments / Results** — benchmark protocol, baseline 비교(supervised baseline 대비 **94.7% recall, +18% faint CME**), ablation, qualitative examples(halo / partial-halo / faint 케이스).
8. **§ Discussion & Limitations** — false-positive 원인(streamer 혼동, occulter edge), 일반화(C3, STEREO/COR로 전이), geoeffectiveness 예측 downstream 가능성.

### English
1. **Abstract + §1 Introduction** — Lock in the three contributions (SSL backbone, Mahalanobis pseudo-mask, lightweight U-Net) and the headline numbers: **99% accuracy, 95% Dice, >80% annotation reduction, 94.7% recall, +18% faint-CME detection**.
2. **§2 Related Work** — Trace the arc CACTus/SEEDS/ARTEMIS/CORIMP → Nguyen/CAMEL/TransCME, and spot how CMEGNets is differentiated on the **label-dependence vs SSL** axis.
3. **§3.1 Data** — 200k+ LASCO C2 frames; FITS→PNG conversion; BM3D denoising; sampling strategy spanning Cycles 23, 24, 25.
4. **§3.2 Framework** — Read Fig. 1 first. Two stages: binary classifier → segmentation. Fig. 2's UMAP confirms CME/non-CME separation in embedding space.
5. **§3.2.1 Classification** — SimCLR setup, ResNet-18, projection head, NT-Xent; note the reference-set design (20 halo + 20 partial + 20 normal CMEs + 20 quiet-Sun non-CME frames).
6. **§3.2.2 – Segmentation** — Mahalanobis pseudo-mask logic, U-Net fine-tuning with a small expert-verified set, semi-supervised iterative loop.
7. **§ Experiments / Results** — baselines, ablations, and the qualitative halo/partial-halo/faint-CME examples.
8. **§ Discussion & Limitations** — streamer confusions, occulter-edge artefacts, generalisation to C3 and STEREO/COR, downstream geoeffectiveness forecasting.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
- **우주기상 관점**: NOAA SWPC·ESA SSA 같은 운영 예보 센터는 실시간 CME 탐지·속도·질량 추정을 요구한다. CMEGNets처럼 라벨 없이 학습하는 파이프라인은 (i) 30년 LASCO 아카이브를 **재처리**하여 CDAW에서 놓친 이벤트를 찾고, (ii) **실시간 스트림**에 즉시 적용 가능하며, (iii) **차세대 미션(PROBA-3 ASPIICS, ESA Vigil)** 같은 라벨 없는 신규 자료원에 **zero-shot 전이**할 수 있다.
- **머신러닝 관점**: 천체물리 분야에서 SSL/대조학습·이상치 탐지를 **실제 운영 지표**(Dice 0.95, >80% 어노테이션 절감)로 입증한 대표 사례. 라벨 희소 영역(태양 플레어, 오로라, 극지방 기상 등) 전반의 방법론적 모범.
- **과학적 함의**: CDAW 편향에서 벗어난 **일관된 마스크** 생성은 각폭 분포, 질량·에너지 통계, CME-태양풍 관계, CME-geoeffectiveness 모델링(Pricopi 2022, Shan 2024 계열)의 하류 연구에 **재현 가능한 기반**을 제공한다.

### English
- **Space-weather side**: Operational centres (NOAA SWPC, ESA SSA) demand real-time CME detection, speed, and mass. Pipelines like CMEGNets can (i) **re-process** 30 years of LASCO to recover CMEs CDAW missed, (ii) drop directly into real-time streams, and (iii) transfer **zero-shot** to next-generation, unlabelled instruments such as **PROBA-3 ASPIICS** and ESA Vigil.
- **Machine-learning side**: An operational demonstration (Dice 0.95, >80% annotation reduction) of SSL/contrastive + anomaly detection in astrophysics — a methodological template for other label-scarce domains (solar flares, auroras, polar weather).
- **Scientific consequence**: Bias-free masks enable **reproducible downstream statistics** — angular width distributions, mass/energy spectra, CME-solar-wind coupling, and CME-geoeffectiveness models (Pricopi 2022; Shan 2024 line).

---

## Q&A

### Q1. UMAP이 뭔가요? / What is UMAP?

**한 줄 정의 / One-line definition.**
**고차원 데이터를 2D/3D로 내려서 시각화하는 비선형 차원축소 기법** — "비슷한 것은 가깝게, 다른 것은 멀게"라는 이웃 관계(neighbor structure)를 보존한다. / A nonlinear dimensionality-reduction technique that projects high-dimensional data into 2D/3D while preserving **local neighbor structure**.

**왜 논문에 등장하나 / Why it appears in this paper.**
CMEGNets의 SimCLR이 학습한 임베딩은 **512차원**(ResNet-18 출력) 또는 **128차원**(projection head 출력)이다. 이 공간에서 CME 프레임과 non-CME 프레임이 실제로 **잘 분리되는지** 눈으로 확인하고 싶을 때 UMAP으로 2D 평면에 투영해서 본다 (논문 Fig. 2). "두 클래스가 뚜렷한 구름(cluster)으로 나뉘면 표현학습이 잘 됐다"는 **정성적 검증 도구**이다.
The embeddings learned by SimCLR live in **512-d** (ResNet-18 output) or **128-d** (projection head). To *visually* confirm that CME vs non-CME frames separate in that space, UMAP projects them to 2-D (Fig. 2) — a qualitative sanity check that representation learning worked.

**직관 / Intuition.**
1. 각 점에 대해 k-nearest neighbors를 찾아 "내 이웃과 얼마나 가까운지"를 기록 (fuzzy simplicial set).
2. 고차원에서의 이웃 관계를 최대한 닮게끔 저차원 점 배치를 최적화 (cross-entropy 최소화).
3. 결과: 가까이 있는 점들끼리 뭉쳐 **cluster**로 보인다.
Build a fuzzy simplicial complex from k-nearest neighbors in high-D, then optimise a low-D layout whose neighborhood structure matches the high-D one (cross-entropy minimisation). Nearby points form visual **clusters**.

**t-SNE와의 차이 / vs t-SNE.**

| | **UMAP** | **t-SNE** |
|---|---|---|
| 속도 / Speed | **빠름** (수십만 점 OK) | 느림 |
| 전역 구조 / Global structure | 비교적 보존 | 잘 망가짐 |
| 이론 / Theory | 리만 기하 + 대수 위상 | 확률 분포 매칭 |
| 재현성 / Reproducibility | 보통 | 낮음 (시드 민감) |

최근 ML 논문은 거의 UMAP을 기본값으로 쓴다. / Modern ML papers default to UMAP.

**사용 예시 / Usage.**
```python
import umap
reducer = umap.UMAP(n_components=2, metric='cosine', n_neighbors=15)
embedding_2d = reducer.fit_transform(features_512d)  # (N, 512) → (N, 2)
```
논문에서는 `metric='cosine'`을 사용 — 임베딩 벡터의 **방향(각도)**에 민감하도록. / The paper uses `metric='cosine'` so angular differences dominate.

**주의점 / Caveats.**
- **축에 의미 없음** — "UMAP 1 = 2.5"는 해석 불가. 오직 **상대적 거리와 군집 구조**만 본다. / Axes are not interpretable — only relative distance/clusters matter.
- **cluster 크기·간격**은 과장될 수 있음. / Cluster size and gap distances can be distorted.
- **정성 시각화 도구**이지 정량 분류기가 아님 — 그래서 논문도 "modest overlap" 같은 표현을 쓴다. / A qualitative visualisation tool, not a quantitative classifier.

### Q2. 논문의 네트워크 구조와 알고리즘 흐름 / Network architecture & algorithm flow

#### 전체 파이프라인 (Fig. 1) / Overall pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 0: 데이터 전처리                        │
│   FITS (Level-0.5) → JSOC render_image (linear stretch)         │
│   → 512×512 8-bit PNG → BM3D denoising                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │ (약 230만 장 unlabelled)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│   STAGE 1A: SimCLR 대조학습 (ResNet-18 backbone 사전학습)       │
│                                                                 │
│   x ──[random augmentation 2회]──→ x₀, x₁                       │
│        │                                                        │
│        ▼                                                        │
│   ResNet-18 fθ  ──→  h ∈ ℝ⁵¹² (global)                          │
│                  ↓                                              │
│                  +  7×7×512 (spatial, for segmentation)         │
│        │                                                        │
│        ▼                                                        │
│   Projection head g (MLP 512→512→128, ReLU) → z ∈ ℝ¹²⁸          │
│        │                                                        │
│        ▼                                                        │
│   NT-Xent loss (τ=0.5) — positive pair 당김, negative 밀어냄    │
└──────────────────────┬──────────────────────────────────────────┘
                       │  (projection head g 버림, backbone만 보관)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│   STAGE 1B: 이진 분류 (CME / non-CME)                           │
│                                                                 │
│   두 가지 방식 병행 → Voting:                                   │
│                                                                 │
│   (i) Reference-based (zero-shot)                               │
│       테스트 z_t 를 reference 임베딩 z_c(CME), z_nc(non-CME)와  │
│       코사인 거리 비교 → 가까운 쪽으로 분류                     │
│       Reference: 20 halo + 20 partial + 20 normal = 60 CME      │
│                  + quiet-Sun non-CME set                        │
│                                                                 │
│   (ii) Supervised MLP head                                      │
│        ResNet-18(frozen) → 경량 MLP                             │
│        훈련: 2,800 labelled (1,500 non-CME + 1,300 CME)         │
│        224×224, Adam lr=1e-5, 5-fold CV, early stop(10)         │
│        → 정확도 99.34%, F1 99.34%                               │
└──────────────────────┬──────────────────────────────────────────┘
                       │  CME로 분류된 프레임만 통과
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│   STAGE 2A: Mahalanobis 의사 마스크 생성 (pseudo-label)         │
│                                                                 │
│   "quiet Sun" 기준선 구축:                                      │
│     - 1,000 non-CME 영상 (2019-Q1 + 2021-Q1, 태양활동 최소기)   │
│     - ResNet-18 첫 3 layer 의 feature map 추출                  │
│     - 각 3×3 패치 위치 i 에 대해:                               │
│       μᵢ = (1/N) Σ fᵢʲ,   Σᵢ = (1/(N-1)) Σ (fᵢʲ-μᵢ)(fᵢʲ-μᵢ)ᵀ   │
│                                                                 │
│   테스트 프레임에 대해:                                         │
│     D_M(f_test,i, μᵢ) = √((f-μᵢ)ᵀ Σᵢ⁻¹ (f-μᵢ))                 │
│                                                                 │
│   픽셀별 D_M → heat map → threshold → binary pseudo-mask        │
└──────────────────────┬──────────────────────────────────────────┘
                       │  pseudo-mask가 U-Net 훈련 신호로
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│   STAGE 2B: U-Net 지도학습 세그멘테이션                         │
│                                                                 │
│   U-Net 인코더-디코더 + skip connection                         │
│   훈련: ~1,000 이미지, 256×256 grayscale, batch=16,             │
│         lr=1e-5, 80 epochs                                      │
│   → 정교한 픽셀 단위 CME mask (Dice 95%, 테스트 Dice 98%)       │
└─────────────────────────────────────────────────────────────────┘
```

#### 각 네트워크 상세 / Network-by-network breakdown

**① ResNet-18 Backbone (feature extractor).**
- 입력 / Input: 512×512 grayscale (SimCLR 단계에서는 256×256으로 resize).
- 구조 / Architecture: He et al. 2016의 표준 ResNet-18 — 4개의 residual block(layer1–4).
- 두 종류의 출력 / Two outputs:
  - **Global vector** $h \in \mathbb{R}^{512}$ (layer4 뒤 global average pooling) — 분류용 / for classification.
  - **Spatial map** $7\times7\times512$ (pooling 직전) — 각 패치가 국소 패턴(모서리, 텍스처, 밝기 변화)을 인코딩, 분할용 / for segmentation.
- Segmentation에서는 **첫 3 layer**(layer1–3)의 d=3 채널 feature만 사용 — 저수준 texture가 코로나 구조에 더 유용 / Only the first 3 layers' d=3 channel features are used for segmentation because low-level textures map better to coronal structure.

**② Projection Head (SimCLR-only).**
- 구조 / Architecture: 2-layer MLP (512 → 512 → 128) + ReLU + L2 normalise.
- 역할 / Role: NT-Xent 손실 계산을 위한 저차원 투영 / Low-dim projection for NT-Xent.
- 사후 처리 / Post-training: **SimCLR 학습 후 버림** — 추론 시에는 raw 512-d $h$ 또는 spatial feature만 사용 / Discarded after pre-training.

**③ MLP Classifier Head (supervised classifier).**
- 입력 / Input: frozen ResNet-18의 512-d feature.
- 훈련 설정 / Training setup:
  | 항목 / Item | 값 / Value |
  |---|---|
  | 이미지 / Images | 2,800 labelled (1,500 non-CME + 1,300 CME) |
  | 해상도 / Resolution | 224×224 |
  | 옵티마이저 / Optimiser | Adam, lr = 1×10⁻⁵ |
  | 검증 / Validation | 5-fold CV |
  | Epoch | 최대 83 (early stop patience=10) |
  | 결과 / Result | 정확도 99.34%, F1 99.34%, training loss 0.008 |

**④ U-Net Segmentation Head.**
- 구조 / Architecture: 표준 U-Net (Ronneberger 2015) — 대칭 인코더·디코더 + skip connection.
- 훈련 설정 / Training setup:
  | 항목 / Item | 값 / Value |
  |---|---|
  | 이미지 / Images | ~1,000 (pseudo-mask로 학습) |
  | 해상도 / Resolution | 256×256 grayscale |
  | Batch | 16 |
  | lr | 1×10⁻⁵ |
  | Epoch | 80 |
  | 결과 / Result | 테스트 60+ 이미지에서 Mean Dice **98%** (벤치마크 95%) |

#### Semi-supervised iterative loop / 반복 구조

CMEGNets의 영리한 점은 **세 방법이 서로를 부트스트랩**한다는 것이다. / The clever bit: three methods bootstrap each other.

```
    SimCLR backbone (라벨 0개)
           │
           ▼
    Reference-based 분류 (라벨 60장)
           │
           ▼  → 대량 라벨 예측
    MLP classifier (라벨 2,800장, 부분적으로 SSL이 생성)
           │
           ▼
    Mahalanobis pseudo-mask (라벨 0, quiet-Sun 통계만)
           │
           ▼  → 1,000장 mask 생성
    U-Net (의사-라벨만으로 지도학습) → 고품질 최종 mask
```

**핵심 트릭 / Key trick**: "이상한 것(anomaly) = CME"라는 **물리적 사전지식(physics-guided prior)**을 통계(Mahalanobis)로 주입 → 라벨 없이도 학습 신호 생성. / A physics-guided prior — "anomalous = CME" — is injected statistically via Mahalanobis, generating supervision without labels.

#### 추론 시 흐름 (실시간 운영) / Inference flow (real-time operation)

```
LASCO C2 새 프레임 도착 (12분 cadence)
    ↓
ResNet-18 forward pass → h(512-d) + spatial map(7×7×512)
    ↓
MLP classifier → "CME?" (Y/N) — 수 ms
    ↓ (Y인 경우)
U-Net forward pass → 픽셀 mask (256×256) — 수십 ms
    ↓
Downstream: 각폭·질량·속도·전파방향 추정 → ENLIL/EUHFORIA 입력
```

논문은 **전체 추론 파이프라인의 파라미터가 <12M**이라고 명시 — onboard satellite 배포도 가능한 수준. / Total inference-pipeline parameters are <12M, enabling onboard satellite deployment.

#### 핵심 설계 포인트 / Core design choices

1. **Denoising(BM3D) → SSL 순서**: raw LASCO는 노이즈가 심해 contrastive learning이 artefact를 학습할 위험이 있음 → 먼저 깨끗하게 만든 뒤 학습. / Denoise before SSL — raw LASCO noise could be latched onto by contrastive learning.
2. **Projection head는 버린다 / Drop the projection head**: SimCLR의 정석. projection은 손실을 위한 **learning trick**이고 downstream은 backbone feature가 더 풍부 / Standard SimCLR practice.
3. **Classification vs Segmentation feature depth**:
   - Classification: layer4의 **high-level semantic** (CME 개념) / high-level semantics.
   - Segmentation: layer1–3의 **low-level texture** (픽셀 경계) / low-level texture.
4. **Voting (Fig. 1)**: Mean cosine distance + Mean Mahalanobis distance 앙상블 → 단일 메트릭의 실패 모드 보완 / Ensembling cosine + Mahalanobis covers single-metric failure modes.

