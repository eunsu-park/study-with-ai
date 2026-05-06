---
title: "deepCR: Cosmic Ray Rejection with Deep Learning"
authors: Keming Zhang, Joshua S. Bloom
year: 2020
journal: "Astrophysical Journal (ApJ), 889, 24"
doi: "10.3847/1538-4357/ab3fa6"
topic: Low-SNR Imaging / Cosmic-Ray Detection
tags: [cosmic-ray, deep-learning, u-net, hst, image-segmentation, inpainting, deepcr, lacosmic]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 24. deepCR: Cosmic Ray Rejection with Deep Learning / deepCR: 심층학습 기반 우주선 제거

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 **HST ACS/WFC 영상**에서 우주선(CR)을 검출·복원하는 딥러닝 프레임워크 **deepCR**를 제안한다. 핵심 구성:
- **deepCR-mask**: 입력 영상을 받아 픽셀별 CR 확률 맵을 출력하는 modified U-Net (segmentation).
- **deepCR-inpaint**: 예측된 CR 마스크 위치의 픽셀 값을 재구성하는 또 다른 U-Net (inpainting).
- 두 모듈은 *독립* 학습되고 inference 시 mask → inpaint 순서로 결합.

훈련 데이터는 HST ACS/WFC F606W 16개 visit (3가지 카테고리: extragalactic field, globular cluster, resolved galaxy)에서 *AstroDrizzle*의 median-stacking을 사용해 생성된 **ground-truth CR mask**. ROC 분석 결과, **0.5% FPR에서 deepCR-2-32은 extragalactic 98.7%, globular cluster 99.5%, resolved galaxy 91.2% TPR**을 달성 — L.A.Cosmic(paper #23)의 69.5%/73.9%/53.4%를 *압도*. 추론 속도도 GPU에서 90× 빠름. 공개 PyPI 패키지 `deepCR`로 제공되며 inpainting MSE는 L.A.Cosmic 대비 5~20× 우수.

### English
The paper presents **deepCR**, a deep-learning framework for cosmic-ray (CR) identification and replacement in HST ACS/WFC imaging. Two complementary modules:
- **deepCR-mask**: a modified U-Net producing a per-pixel CR probability map (image segmentation).
- **deepCR-inpaint**: a second U-Net that fills in pixel values at masked positions (inpainting).
The two networks are trained independently and chained at inference (mask → inpaint).

Training data is built from HST ACS/WFC F606W exposures across 16 visits in three field categories (extragalactic, globular cluster, resolved galaxy), with ground-truth CR masks derived from AstroDrizzle median stacks. ROC analysis at 0.5% false positive rate shows **deepCR-2-32 reaches 98.7%/99.5%/91.2% TPR** for the three categories — versus L.A.Cosmic's 69.5%/73.9%/53.4%. deepCR-mask is up to 6.5× faster on CPU and 90× faster on a single GPU. Inpainting MSE is 5–20× better than the best non-neural baseline. The framework is shipped as the open-source PyPI package `deepCR`.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction / §1 서론

#### 한국어
- CCD 등 solid-state 검출기는 우주선·radioactivity·지상 muon 등에 의해 픽셀이 *과도 충전* → "cosmic ray" artefact. HST는 magnetosphere 내 trap된 양성자·전자 때문에 특히 심각.
- 다중 노출 stacking이 가장 표준적이지만 (a) 전이 천체, (b) 슬릿분광, (c) 오프셋된 노출 정렬 잔차 등 부적합 상황 다수.
- 단일 노출 방법: linear filtering (Rhoads 2000), median filtering (IRAF xzap), Laplacian edge detection (van Dokkum 2001 = L.A.Cosmic, paper #23), histogram analysis (Pych 2004). Farage & Pimbblet 2005에서 L.A.Cosmic이 최고로 평가됨. 그러나 HST under-sampled PSF에서는 여전히 false detection 다수.
- 본 논문 제안: ML 기법(특히 CNN U-Net)로 *학습된* CR 검출. ImageNet 분류 (Deng+ 2009), semantic segmentation (Shelhamer+ 2017), inpainting (Lehtinen+ 2018) 등 CV 분야의 발전을 활용.

#### English
- Solid-state detectors are bombarded by charged particles (terrestrial, instrumental, and cosmic), each producing localised excess charge. HST is especially vulnerable due to trapped protons in Earth's radiation belts.
- Multi-exposure stacking is the standard but fails for (a) transient/variable sources, (b) long-slit spectra, (c) misaligned dithered frames.
- Single-exposure baselines (Rhoads 2000 / IRAF xzap / van Dokkum 2001 L.A.Cosmic / Pych 2004): L.A.Cosmic ranked highest by Farage & Pimbblet 2005, but still suffers many false detections on HST under-sampled PSFs.
- The paper proposes a CNN-based learnt CR detector exploiting CV advances (ImageNet, semantic segmentation, inpainting).

### Part II: §2 Model architecture / §2 모델 아키텍처

#### 한국어
- 두 독립 U-Net: **deepCR-mask**와 **deepCR-inpaint**.
- 기본 U-Net (Ronneberger+ 2015) 구조: encoder-decoder, skip connection으로 high-level semantic + low-level edge feature 결합.
- *수정점*: 표준 U-Net은 경계 92픽셀을 버림 → astronomy에서는 boundary 데이터를 잃음. deepCR은 *segmentation map과 입력 영상 동일 크기*가 되도록 수정. 약간의 boundary loss 증가가 있지만 천체 데이터 보존 우선.
- 변형 표기: **deepCR-D-N**: depth $D$, base channels $N$. 본문은 deepCR-2-4와 deepCR-2-32 평가; deepCR-3-32도 inpainting에 사용. deepCR-4-64도 시도했으나 deepCR-2-32 대비 큰 개선 없음.
- mask Loss: binary cross-entropy
$$
\mathcal L_{\rm F} = \mathbb E[M\log(1 - F(X)) + (1-M)\log F(X)],\quad (1)
$$
where $F$ is the mask output (note: paper writes $1-F(X)$ inside the first log so that $F\to 1$ for non-CR pixels — sign convention).
- inpaint Loss: MSE only on inpainting mask region $M_I$:
$$
\mathcal L_{\rm G} = \mathbb E[(G(X, M_I)\circ M_I\circ(1-M) - X\circ M_I\circ(1-M))^2],\quad (2)
$$
즉 *CR이 아닌 영역에서만* MSE 계산 (CR 영역은 ground truth가 없으므로).
- L1 + per-pixel noise weighting도 시도했으나 sky background 픽셀에 과도한 penalty → MSE를 채택.

#### English
- Two independent U-Nets: **deepCR-mask** and **deepCR-inpaint**.
- Standard U-Net (Ronneberger+ 2015) modified so that segmentation/inpainting map matches input dimension (boundary pixels are no longer cropped) — this trades a small boundary accuracy hit for retaining astronomical data near the edges.
- Variants labelled **deepCR-D-N** = depth $D$, base channels $N$. Main results use deepCR-2-32 (mask) and deepCR-3-32 (inpaint). deepCR-4-64 tried but no significant gain over deepCR-2-32.
- mask loss: binary cross-entropy (Eq. 1).
- inpaint loss: MSE on inpainting mask only, masked away from CR pixels (Eq. 2). L1 weighted by noise was rejected because it over-penalises sky-background pixels (which are the easiest predictions but the most numerous, dominating gradients).

### Part III: §3 Data and augmentation / §3 데이터와 증강

#### 한국어
- 데이터: HST ACS/WFC F606W의 16개 visit (Table 1). 카테고리: extragalactic field (5), globular cluster (4), resolved galaxy (7). 각 visit당 3~6 노출.
- ground-truth mask 생성: AstroDrizzle 파이프라인이 CR-free median 영상과 derivative median 영상으로 정렬 잔차를 보정하면서 CR 마스크 생성. 두 단계 통과 (S/N 임계 5와 1.5; 기본값 3.5와 3보다 더 엄격).
- 영상을 256×256 stamp로 자름 — 훈련 8190 stamp, validation 1638 stamp, test 3360 stamp. test set은 *다른 target field*에서 추출하여 일반화 평가.
- 데이터 증강 (Fig. 2): 데이터셋의 다른 stamp로부터 1~9개 CR mask를 샘플링해 *inpainting mask*에 추가 → 다양한 mask 밀도 학습. *bad pixel mask*와 *saturation mask*로 부정확한 ground truth 영역은 backprop에서 제외.
- 노출 시간 증강: $n' = (1+\alpha)\cdot n$ 형식의 sky background scaling (Eq. 3-4). 짧은 노출(100s) test에서 augmentation으로 ~2% 검출률 향상.

#### English
- Data: HST ACS/WFC F606W from 16 visits (Table 1) covering extragalactic field, globular cluster, resolved galaxy. 3–6 exposures per visit.
- Ground-truth CR masks built from the AstroDrizzle pipeline: a two-pass S/N threshold (5 then 1.5; tighter than the default 3.5/3) on aligned median + derivative median images.
- 256×256 stamps: 8190 training, 1638 validation, 3360 test. Test stamps drawn from *different target fields* to assess generalisation.
- Augmentation (Fig. 2): inpainting masks built by sampling 1–9 CR masks from other dataset stamps; bad-pixel and saturation masks exclude unreliable regions from backprop.
- Exposure-time augmentation $n' = (1+\alpha)\cdot n$ (Eqs. 3–4) by scaling sky background; helps short-exposure (100 s) tests by ~2%.

### Part IV: §4 Results / §4 결과

#### 한국어
- 평가: ROC curves (TPR vs FPR) of deepCR-mask vs L.A.Cosmic on three field categories.
- L.A.Cosmic의 `objlim` 매개변수는 fine-tuned: extragalactic = 2, globular cluster = 3.5, resolved galaxy = 5. 이는 권장 default 4-5보다 작음.
- ROC 결과 (Fig. 4): deepCR이 모든 FPR 영역에서 L.A.Cosmic 압도. 특히 resolved galaxy처럼 천체 구조 복잡한 경우 차이가 큼.
- **Table 2**: 0.05% FPR에서 deepCR-2-32 vs L.A.Cosmic
  - extragalactic: 88.5% vs 57.3%
  - globular cluster: 93.3% vs 58.3%
  - resolved galaxy: 75.2% vs 33.8%
  - 0.5% FPR에서: 98.7%/99.5%/91.2% vs 69.5%/73.9%/53.4%.
- 속도: deepCR-2-32 256×256 100 stamps에서 4-core CPU 7.9s, V100 GPU 0.2s. L.A.Cosmic 9.0s. → CPU 1.1×, GPU 45× 빠름.
- 더 작은 deepCR-2-4: CPU 1.4s, GPU 0.1s, 그러나 TPR 약간 낮음 (82.2/83.9/56.2%).
- inpainting: median replacement 대비 MSE 5× (resolved galaxy) ~ 20× (globular cluster) 낮음.
- 결과 시각화 (Fig. 3): deepCR이 길쭉한 muon track, 작은 단일픽셀 CR, 천체와 겹친 CR 모두 깔끔히 식별.

#### English
- Evaluation by ROC curves (TPR vs FPR) of deepCR-mask vs L.A.Cosmic across three field categories.
- L.A.Cosmic's `objlim` was fine-tuned per category (extragalactic=2, globular=3.5, galaxy=5; tighter than the default 4–5).
- deepCR dominates the ROC across all FPRs (Fig. 4). Largest gap on resolved galaxy fields where complex astronomical structure confuses L.A.Cosmic.
- **Table 2** at 0.05% FPR: deepCR-2-32 vs L.A.Cosmic — 88.5/57.3, 93.3/58.3, 75.2/33.8 (%). At 0.5% FPR: 98.7/69.5, 99.5/73.9, 91.2/53.4 (%).
- Runtime on 100 stamps of 256×256: deepCR-2-32 CPU 7.9 s / V100 GPU 0.2 s; L.A.Cosmic 9.0 s. ~1.1× CPU and ~45× GPU speedup; deepCR-2-4 reaches 0.1 s GPU.
- Inpainting MSE 5× to 20× lower than median replacement (best non-neural).
- Visual examples (Fig. 3) show clean detection of long muon tracks, single-pixel CRs, and CRs overlapping with sources.

### Part V: §5 Discussion / §5 논의

#### 한국어
- **L.A.Cosmic 매개변수 fine-tuning의 한계**: `objlim`을 데이터셋별로 조정해도 deepCR을 따라잡지 못함. 이는 *학습된 feature*가 *수작업 통계 feature*보다 풍부함을 시사.
- **AstroDrizzle 마스크의 한계와 deepCR의 우위**: drizzle 자체도 정렬 잔차 때문에 작은 CR을 놓침. deepCR은 학습 중 이런 *체계적 누락*을 일부 보상하도록 학습.
- **CR이 천체를 가리는 경우**: deepCR-inpaint가 인근 픽셀로부터 자연스럽게 *내삽*. 별/은하의 PSF 모델을 명시적으로 알 필요 없음.
- **일반화**: 본 모델은 ACS/WFC F606W에 학습. 다른 필터/검출기에서는 fine-tune 권장. 그러나 architecture는 동일.
- **재현 가능성**: PyPI 패키지 + 사전학습 모델 공개로 재현이 쉬움. 새로운 telescope/instrument에 fine-tune 가능.

#### English
- **Limits of L.A.Cosmic fine-tuning**: even after per-category `objlim` tuning, L.A.Cosmic cannot match deepCR — *learnt features* are richer than *hand-crafted statistical features*.
- **AstroDrizzle mask imperfection vs deepCR's advantage**: drizzle itself misses small CRs due to alignment residuals; deepCR partially compensates for these systematic omissions by learning from many samples.
- **CRs overlapping sources**: deepCR-inpaint naturally interpolates from neighbouring pixels — no explicit PSF model is needed.
- **Generalisation**: pretrained on ACS/WFC F606W; fine-tuning is recommended for other filters/instruments, but the architecture is unchanged.
- **Reproducibility**: PyPI package + pretrained models make adoption straightforward; the same code base supports fine-tuning on new telescope/instrument data.

---

## 3. Key Takeaways / 핵심 시사점

1. **U-Net이 CR 검출에 최적 / U-Net is the right architecture for CR detection** — Skip connection이 high-level (CR vs source 구분)과 low-level (정확한 edge 위치) feature를 모두 보존. Fully convolutional이라 입력 크기 자유.
   U-Net's skip connections preserve both high-level (CR vs source classification) and low-level (precise edge location) features, while its full-convolutional design accepts arbitrary input sizes.

2. **Mask와 inpaint를 분리 학습 / Decouple masking and inpainting** — 두 작업은 본질적으로 다름 (segmentation vs reconstruction). 분리 학습으로 각 작업 최적화 가능. 추론 시 chain.
   Splitting into segmentation + inpainting modules lets each network optimise its own loss; the outputs are simply chained at inference.

3. **Ground truth가 필요하지만 AstroDrizzle로 자동 생성 / Ground truth is automated via AstroDrizzle** — 다중 노출 median stacking이 *deep learning을 위한 supervisory signal*을 제공. 단일 노출 검출기를 학습하기 위해 다중 노출을 *training-time only* 사용.
   Ground-truth masks come from AstroDrizzle median stacks — multi-exposure stacking provides the supervisory signal *only at training time*; deepCR runs single-exposure at inference.

4. **MSE > weighted L1 for inpainting / 인페인팅에는 MSE가 weighted L1보다 좋음** — sky-background 픽셀이 압도적으로 많아 weighted L1은 background에 과도한 가중치 부여. MSE는 전체에 균등 가중치 → 별/은하 같은 *어려운* 픽셀이 공정히 학습.
   Weighted L1 over-penalises easy sky-background pixels (which dominate by sheer count); MSE distributes gradient more evenly so harder pixels (stars/galaxies) get learnt.

5. **0.5% FPR에서 99% TPR / 99% TPR at 0.5% FPR** — 사실상 supervised baseline. resolved galaxy처럼 어려운 분야에서도 91% TPR — L.A.Cosmic 대비 +38%p.
   Practically supervised-level performance: 99% TPR at 0.5% FPR. Even on the hardest category (resolved galaxies) deepCR is +38 percentage points over L.A.Cosmic.

6. **GPU에서 45-90× 가속 / 45-90× speedup on GPU** — single forward pass의 수치적 cost가 L.A.Cosmic의 반복 median filtering보다 적음. 실시간 파이프라인에 적합.
   A single forward pass beats L.A.Cosmic's iterative median filtering, enabling real-time pipelines.

7. **Boundary 처리 차이 / Boundary handling differs from vanilla U-Net** — 천문학 데이터는 boundary 정보를 못 버리므로 segmentation map이 입력과 동일 크기가 되도록 수정. 약간의 boundary accuracy 손실 vs 90+ pixels 데이터 손실의 trade-off.
   Astronomy can't discard boundary pixels, so deepCR modifies U-Net to produce same-size segmentation; pays a small boundary-accuracy cost for retaining the data.

8. **Open-source PyPI 패키지 / Open-source PyPI package** — `pip install deepCR`. pretrained 모델 제공. drop-in API로 누구나 사용 가능. 이는 *재현 가능성*을 위한 결정적 설계 선택.
   Released as the `deepCR` PyPI package with pretrained models — a deliberate choice for reproducibility, drop-in adoption, and benchmark baselines.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Setting / 설정
$$
n = (f_{\rm star} + f_{\rm sky})\cdot t_{\rm exp} + n_{\rm CR},
$$
$n$ = pixel count (e$^-$), $f_{\rm star}, f_{\rm sky}$ = source/sky flux (e$^-$/s), $n_{\rm CR}$ = CR contribution.

### 4.2 deepCR-mask / 마스크 모듈
$F_\theta: \mathbb R^{H\times W} \to [0,1]^{H\times W}$. Modified U-Net (depth=2, base channels=32).
$$
M_{\rm pred}(p,q) = \mathbf 1\{F_\theta(X)(p,q) > \tau\},
$$
$\tau$ is the decision threshold (operated as a knob in ROC analysis).

### 4.3 Mask loss (Eq. 1) / 마스크 손실
$$
\mathcal L_{\rm F} = \mathbb E\big[M\log(1 - F(X)) + (1-M)\log F(X)\big].
$$
(Standard binary cross-entropy with the convention $F\to 1$ for non-CR pixels.)

### 4.4 deepCR-inpaint / 인페인팅 모듈
$G_\phi: (\mathbb R^{H\times W}, \{0,1\}^{H\times W}) \to \mathbb R^{H\times W}$. Input is image with CR pixels zeroed and a binary mask channel.

### 4.5 Inpaint loss (Eq. 2) / 인페인팅 손실
$$
\mathcal L_{\rm G} = \mathbb E\big[\big(G(X, M_I)\odot M_I \odot (1-M) - X\odot M_I \odot (1-M)\big)^2\big].
$$
Compute MSE only on pixels that are (a) inside the inpainting mask $M_I$ AND (b) NOT actual CR ($1-M$).

### 4.6 Training augmentation (Eqs. 3–4) / 훈련 증강
$$
n = (f_{\rm star}+f_{\rm sky})t_{\rm exp} + n_{\rm CR}, \quad n' = n + \alpha f_{\rm sky} t_{\rm exp} = \!\left(\frac{f_{\rm star}}{1+\alpha} + f_{\rm sky}\right)\!(1+\alpha)t_{\rm exp} + n_{\rm CR}.
$$
Adding $\alpha f_{\rm sky} t_{\rm exp}$ effectively rescales $t_{\rm exp}$ and reduces source flux contrast, mimicking variable sky background.

### 4.7 Inference pipeline / 추론 파이프라인
1. $P = F_\theta(X)$ (probability map).
2. $M_{\rm pred} = \mathbf 1\{P > \tau\}$ (binary mask).
3. $X_{\rm masked} = X \odot (1 - M_{\rm pred})$ (zero out CR pixels).
4. $\hat X = G_\phi(X_{\rm masked}, M_{\rm pred})$ (inpaint).
5. Output: $X_{\rm clean} = X\odot(1-M_{\rm pred}) + \hat X \odot M_{\rm pred}$.

### 4.8 Worked numerical example / 수치 예시
HST ACS/WFC pixel: gain $g\approx 1.0$ e$^-$/ADU, read-noise $\sigma_{\rm rn}\approx 4.4$ e$^-$, sky $\langle f_{\rm sky}\rangle\approx 0.1$ e$^-$/s, $t_{\rm exp}=500$ s. Background $\approx 50$ e$^-$, noise std $\approx \sqrt{50+4.4^2}\approx 8.4$ e$^-$.
- A typical CR deposits 1000–10000 e$^-$ over 1–10 px. SNR = $1000/8.4\approx 119\sigma$. L.A.Cosmic detects easily; deepCR also.
- A faint CR of 50 e$^-$ over 1 px → SNR $\approx 6\sigma$. L.A.Cosmic detection probability ~95%, but produces $\approx 0.5\%$ false positive rate. deepCR-2-32 at same FPR: 99% TPR.
- For resolved galaxies the false positive rate cost is steep: the "spiral arm" structure can have $\nabla^2 I$ comparable to faint CRs; deepCR uses learnt features (not just $\nabla^2$) so it discriminates much better — hence the +38 percentage point gain.

### 4.9 ROC table at 0.5% FPR (Table 2)
| Category | deepCR-2-32 TPR | deepCR-2-4 TPR | L.A.Cosmic TPR |
|---|---|---|---|
| extragalactic | 98.7% | 94.0% | 69.5% |
| globular cluster | 99.5% | 96.2% | 73.9% |
| resolved galaxy | 91.2% | 80.6% | 53.4% |

### 4.10 Runtime (100 stamps of 256×256)
| Method | CPU [s] | GPU (V100) [s] |
|---|---|---|
| deepCR-2-4 | 1.4 | 0.1 |
| deepCR-2-32 | 7.9 | 0.2 |
| L.A.Cosmic | 9.0 | n/a |

### 4.11 Architecture diagram (text version) / 아키텍처 텍스트 다이어그램
```
                  Input (1, 256, 256)
                        |
                +-------+--------+
                |                |
              Conv 32        skip down
                |                |
            Maxpool 2x          ...
                |
              Conv 64
                |
            Maxpool 2x
                |
              Conv 128 (bottleneck)
                |
              Upconv 64 + concat skip
                |
              Upconv 32 + concat skip
                |
              Conv 1x1 -> sigmoid -> output mask
```
This is depth-2 (two encoder blocks before bottleneck), base channels 32 — i.e., "deepCR-2-32".

### 4.12 Decision rule / 판정 규칙
$$
M_{\rm pred}(p,q) = \mathbf 1\{F_\theta(X)(p,q) > \tau\}.
$$
The threshold $\tau$ controls the operating point on the ROC curve. Default $\tau=0.5$ for balanced TPR/FPR; lower $\tau$ raises TPR at the cost of FPR (useful for catastrophic-event detectors).

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1980 ─── Marr-Hildreth — Edge detection theory
1995 ─── Salzberg+ — Decision-tree CR classifier (early ML)
2000 ─── Rhoads — Linear-filter CR detector
2001 ★ van Dokkum — L.A.Cosmic (paper #23)
2004 ─── Pych — Histogram-based CR detection
2005 ─── Farage-Pimbblet — L.A.Cosmic ranked best
2009 ─── Deng+ — ImageNet (deep learning era begins)
2015 ─── Ronneberger+ — U-Net (medical image segmentation)
2017 ─── Shelhamer+ — Fully Convolutional Networks for semantic segmentation
2018 ─── Lehtinen+ — Noise2Noise (training on noisy targets)
2018 ─── Sedaghat-Mahabal — U-Net for transient detection in astronomy
2020 ★★ ZHANG-BLOOM: deepCR (this paper)
              ↳ U-Net beats L.A.Cosmic by 30+ pp at fixed FPR
              ↳ open-source PyPI package
2020+ ── deepCR adopted by HST/JWST/Roman pipelines as supplement to L.A.Cosmic
2022+ ── Hybrid classical/deep CR pipelines (deepCR + L.A.Cosmic ensemble)
2024+ ── Diffusion-based CR removal experiments
```

이 논문은 **"L.A.Cosmic이 19년간 표준이었지만 이제 deep learning이 추월할 시간"** 임을 보여주었고, astronomical image processing 분야의 CNN 채택을 가속화한 주요 사례 중 하나다.

This paper showed that **after 19 years of L.A.Cosmic's reign, deep learning can decisively surpass it** and was a key catalyst for CNN adoption in astronomical image processing.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **van Dokkum (2001)** *L.A.Cosmic* (paper #23) | The benchmark deepCR is built to surpass. | Direct ROC comparison; deepCR uses L.A.Cosmic for both validation and ROC plotting. |
| **Ronneberger et al. (2015)** *U-Net* | Backbone architecture. | deepCR's modified U-Net keeps the input size while using the standard encoder-decoder + skip-connection design. |
| **Shelhamer et al. (2017)** *Fully Convolutional Networks* | Founding paper for semantic segmentation. | deepCR-mask is exactly such a fully-convolutional segmentation model. |
| **Lehtinen et al. (2018)** *Noise2Noise* | Training on noisy targets. | deepCR-inpaint trains on noisy median targets — Lehtinen+ showed this is comparable to clean targets. |
| **Sedaghat & Mahabal (2018)** *Effective image differencing with CNN* | U-Net in astronomy precedent. | First major astronomy U-Net paper; deepCR cites it as proof U-Net is well-suited to astronomy. |
| **Murtagh & Adorf (1991)** *Neural networks for CR* | Early ML attempt. | Predecessor; deepCR shows modern CNNs vastly improve on these classical neural methods. |
| **Salzberg et al. (1995)** *Decision tree CR classifier* | Earlier ML CR detector. | Predecessor compared in §1; deepCR's gain over decision-trees mirrors Modern-CV vs classical-CV gap. |
| **Pang et al. (2021)** *R2R* (paper #21) | Self-supervised denoising philosophy. | Different but related: R2R does noise removal without ground truth; deepCR does CR removal *with* ground truth from AstroDrizzle. |
| **Wang et al. (2022)** *Blind2Unblind* (paper #22) | Self-supervised denoising. | Another self-supervised baseline; deepCR is fully *supervised* via AstroDrizzle masks. |

---

### 6.1 Practical usage of the deepCR PyPI package / deepCR PyPI 패키지 실용 사용법

#### 한국어
```python
from deepCR import deepCR
mdl = deepCR(mask='ACS-WFC-F606W-2-32',
             inpaint='ACS-WFC-F606W-3-32',
             device='cuda')
mask, cleaned = mdl.clean(noisy_image, threshold=0.5,
                          inpaint=True, segment=True)
```
- `threshold`: 결정 임계값. 낮추면 TPR↑, FPR↑.
- `segment=True`: 큰 영상을 256×256 stamp로 자동 분할 → 메모리 효율.
- `inpaint=True`: 마스크 + 인페인팅 모두 수행.
- 사전학습 모델: `ACS-WFC-F606W-2-32` (default), `ACS-WFC-F606W-2-4` (가벼움).

#### English
```python
from deepCR import deepCR
mdl = deepCR(mask='ACS-WFC-F606W-2-32',
             inpaint='ACS-WFC-F606W-3-32',
             device='cuda')
mask, cleaned = mdl.clean(noisy_image, threshold=0.5,
                          inpaint=True, segment=True)
```
- `threshold`: decision threshold. Lower → higher TPR and higher FPR.
- `segment=True`: split large images into 256×256 stamps automatically (memory efficiency).
- `inpaint=True`: run both masking and inpainting modules.
- Pretrained models: `ACS-WFC-F606W-2-32` (default), `ACS-WFC-F606W-2-4` (lightweight).

### 6.2 Fine-tuning to a new instrument / 새 기기로 fine-tuning

#### 한국어
1. *충분한* 다중 노출 데이터 수집 (≥3 exposures per visit).
2. AstroDrizzle로 ground-truth CR 마스크 생성.
3. 256×256 stamp 추출 (paper의 `Roman` 등 새 미션도 같은 방식).
4. 사전학습 가중치 로드 후 lr=1e-4로 짧게 fine-tune.
5. 목표 기기의 영상에서 ROC 측정, threshold 조정.

#### English
1. Collect *enough* multi-exposure data (≥3 exposures per visit).
2. Generate ground-truth CR masks via AstroDrizzle.
3. Extract 256×256 stamps (the same recipe applies to Roman, JWST, Euclid, etc.).
4. Load pretrained weights and fine-tune briefly at lr=1e-4.
5. Measure ROC on target-instrument data and tune threshold.

### 6.3 Outlook for next-generation telescopes / 차세대 망원경 전망

#### 한국어
- **JWST**: NIR 검출기는 우주선 분포가 HST와 다름 (lower-energy events with telegraph-like jumps). deepCR-style 모델을 JWST 데이터로 재학습 가능하나, JWST의 *up-the-ramp* 데이터 모델 (Williamson+ 2018)이 이미 CR을 효과적으로 제거하므로 단일 노출 검출의 우선순위는 낮음.
- **Roman Space Telescope**: WFI (Wide Field Instrument)는 HST ACS 후속격으로 deepCR 직접 적용 가능. Calibration 파이프라인에 통합 검토.
- **Euclid**: VIS 채널은 0.1″ 픽셀로 잘 표본화 → L.A.Cosmic만으로도 충분할 수 있음. NISP 분광은 다중 노출 사용.
- **지상 측광 (LSST)**: 30s 노출 수만 회 → CR 처리 자동화 필수. deepCR-style 모델이 candidate.

#### English
- **JWST**: NIR detectors have a different CR distribution from HST (lower-energy events causing telegraph-like jumps). deepCR could be retrained on JWST data, though the up-the-ramp readout model (Williamson+ 2018) already mitigates CRs, lowering the priority for single-exposure detection.
- **Roman Space Telescope**: WFI is a direct HST ACS successor — deepCR applies directly; integration into the calibration pipeline is plausible.
- **Euclid**: VIS pixels at 0.1″ are well-sampled, so L.A.Cosmic may suffice; NISP spectroscopy uses multi-exposure stacking.
- **Ground-based photometry (LSST)**: tens of thousands of 30 s exposures demand automated CR handling — deepCR-style models are strong candidates.

---

## 7. References / 참고문헌

- Zhang, K., & Bloom, J. S., "deepCR: Cosmic Ray Rejection with Deep Learning", *ApJ*, 889, 24 (2020). [DOI: 10.3847/1538-4357/ab3fa6]
- van Dokkum, P. G., "Cosmic-Ray Rejection by Laplacian Edge Detection", *PASP*, 113, 1420 (2001).
- Ronneberger, O., Fischer, P., & Brox, T., "U-Net: Convolutional Networks for Biomedical Image Segmentation", *MICCAI 2015*.
- Shelhamer, E., Long, J., & Darrell, T., "Fully Convolutional Networks for Semantic Segmentation", *IEEE PAMI*, 39, 640–651 (2017).
- Lehtinen, J., et al., "Noise2Noise: Learning Image Restoration without Clean Data", *ICML 2018*.
- Hack, W., et al., "AstroDrizzle: A Redesigned MultiDrizzle", *Astronomical Data Analysis Software and Systems XXI*, 461, 233 (2012).
- Salzberg, S., et al., "Decision Trees for Automated Identification of Cosmic-Ray Hits in HST Images", *PASP*, 107, 1 (1995).
- Murtagh, F., & Adorf, H.-M., "Detecting Cosmic Ray Hits Using Neural Networks", *Astronomy from Large Databases II*, 51 (1991).
- Sedaghat, N., & Mahabal, A., "Effective Image Differencing with Convolutional Neural Networks", *MNRAS*, 476, 5365 (2018).
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L., "ImageNet: A Large-Scale Hierarchical Image Database", *CVPR 2009*.
- Code: https://github.com/profjsb/deepCR ; PyPI: `pip install deepCR`.
