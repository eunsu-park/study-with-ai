---
title: "Pre-Reading Briefing: L.A.Cosmic"
paper_id: "23_van_dokkum_2001"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Cosmic-Ray Rejection by Laplacian Edge Detection (L.A.Cosmic) / 사전 읽기 브리핑

**Paper**: van Dokkum, P. G. "Cosmic-Ray Rejection by Laplacian Edge Detection". *Publications of the Astronomical Society of the Pacific (PASP)*, 113, No. 789, 1420–1427 (2001). DOI: 10.1086/323894.
**Author**: Pieter G. van Dokkum
**Year**: 2001

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 **단일 CCD 노출**에서 우주선(cosmic ray, CR)을 검출/제거하는 **L.A.Cosmic** 알고리즘을 제안한다. 핵심 통찰은: *대기·광학에 의해 번지지 않은* CR은 천체보다 *훨씬 날카로운 에지*를 가진다는 점이다. 따라서 (i) **2× 서브샘플링 + Laplacian 컨볼루션**으로 sharp edge를 검출하고, (ii) **Poisson + read-noise 모델**로 임계값을 데이터 적응적으로 정의하며, (iii) **fine-structure image $\mathcal F$**의 비율 $\mathcal L^+/\mathcal F$로 별/은하 같은 *대칭적* 점광원과 *비대칭적* CR을 구분한다. 임의의 모양·크기 CR을 처리하고 *under-sampled PSF* (HST WFPC2 같은 경우)에서도 매개변수 $f_{\rm lim}$ 조정만으로 동작. 시뮬레이션에서 227개 CR 중 222개(98%) 검출, 별 오검출은 단 0.2%. 2001년부터 2020년까지 거의 모든 HST 단일 노출 처리 파이프라인의 *de-facto 표준*이 되었다.

### English
The paper introduces **L.A.Cosmic**, an algorithm to detect and remove cosmic rays in a *single* CCD exposure. The key insight: cosmic rays — unlike astronomical sources — are not smeared by the atmosphere or optics, so they exhibit *markedly sharper edges*. The algorithm exploits this with three pillars: (i) **2× sub-sampling + Laplacian convolution** detects sharp CR edges while smoothly-sampled stars stay weak, (ii) a **Poisson + read-noise model** sets a data-adaptive threshold, and (iii) a **fine-structure image** $\mathcal F$ provides a second discriminator via the ratio $\mathcal L^+/\mathcal F$ — large for asymmetric CRs, small for symmetric point sources. The method handles CRs of arbitrary shape/size and works on under-sampled PSFs (HST WFPC2) by tuning $f_{\rm lim}$. On a synthetic test field, it detects 222/227 CRs (98%) with only 0.2% stellar false positives. From 2001 to ~2020 it has been the *de-facto standard* for HST single-exposure CR rejection.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
1990년대 HST 시대에 CCD 영상에서 CR 제거는 두 갈래로 나뉘었다: (a) **다중 노출 median stacking** — 정적 장면은 잘 작동하지만 (b) 변광·이동 천체, (c) 슬릿분광에서 sky/object spectrum의 시간 변동, (d) 노출 사이 시상 변동에서는 부적합. 단일 노출 방법으로는 IRAF COSMICRAYS (median filtering), QZAP, PSF-matched filtering, 신경망 분류기(Salzberg+ 1995), 선형 필터(Rhoads 2000)가 있었으나 모두 *PSF가 잘 표본화되어야* 동작하는 한계 — HST WFPC2의 under-sampled PSF에서 false positive 폭증. 2001년 van Dokkum이 2× 서브샘플링 + Laplacian + fine-structure ratio라는 *기하학적 통찰*로 이 문제를 해결한다.

#### English
In the 1990s HST era, CR removal had two branches: (a) **multi-exposure median stacking** worked well for static scenes but failed for (b) variable/moving sources, (c) long-slit spectroscopy with time-variable sky lines, and (d) seeing-variable exposures. Single-exposure tools (IRAF COSMICRAYS median filtering, QZAP, PSF-matched filtering, Salzberg+ 1995's neural classifier, Rhoads 2000's linear filter) all assumed PSF was well-sampled — a fatal limitation for HST WFPC2's under-sampled PSF, where false positives blew up. In 2001 van Dokkum solved this with a geometric insight: 2× sub-sampling + Laplacian convolution + fine-structure ratio.

### 타임라인 / Timeline

```
1980  Marr-Hildreth — theory of edge detection (Laplacian zero-crossings)
1992  Gonzalez & Woods — Laplacian as standard image-processing tool
1995  Salzberg — decision-tree CR classifier
2000  Rhoads — linear-filter CR detector (PSP-sampled only)
2001 ★ van Dokkum L.A.Cosmic — 2× sub-sample + Laplacian + fine-structure
2004  Pych — histogram-based CR detection
2005  Farage & Pimbblet — L.A.Cosmic ranked highest among all methods
2012  astropy.lacosmic — Python wrapper
2020  Zhang & Bloom deepCR (#24) — U-Net successor (uses L.A.Cosmic as baseline)
```

---

## 3. 필요한 배경 지식 / Prerequisites

#### 한국어
- **이산 Laplacian 연산자** $\nabla^2$ 와 그 $3\times 3$ 커널
- **PSF (Point Spread Function)** 와 영상 컨볼루션
- **Marr-Hildreth zero-crossing edge detection** (1980)
- **Median filter** ($M_3, M_5, M_7$)
- **Poisson photon noise + Gaussian read-noise** 모델
- **CCD detector 기초**: gain $g$ (e$^-$/ADU), read-noise $\sigma_{\rm rn}$
- **PSF sampling** 개념: well-sampled vs critically-sampled vs under-sampled (HST WFPC2)
- **Significance map (SNR)** 개념과 임계값 처리
- **iterative algorithm** 설계

#### English
- The discrete Laplacian operator $\nabla^2$ and its $3\times 3$ kernel.
- PSF (Point Spread Function) and image convolution.
- Marr-Hildreth zero-crossing edge detection (1980).
- Median filtering (sizes $3, 5, 7$).
- Poisson photon noise + Gaussian read-noise model.
- CCD detector basics: gain $g$ (e$^-$/ADU) and read-noise $\sigma_{\rm rn}$.
- PSF sampling concepts: well-sampled vs critically-sampled vs under-sampled (HST WFPC2).
- Significance / SNR maps and threshold-based detection.
- Iterative algorithm design.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Cosmic ray (CR)** | 검출기 픽셀에 직접 들어오는 고에너지 입자, sharp edge / Sharp-edged particle hit on CCD pixels. |
| **2× sub-sampling** | 픽셀을 4개로 복제, 인접 CR 픽셀 간섭 방지 / Pixel replication by 4× to decouple adjacent-CR cross-patterns. |
| **Discrete Laplacian** | $3\times 3$ kernel, 평균 0, edge 강조 / Mean-zero $3\times 3$ kernel emphasising edges. |
| **Significance image $S$** | $\mathcal L^+/(f_s N)$, 잡음 단위 SNR / Per-pixel SNR after Laplacian. |
| **Noise model $N$** | $g^{-1}\sqrt{g(M_5\circ I)+\sigma_{\rm rn}^2}$, Poisson + read-noise / Poisson + read-noise composite. |
| **Fine-structure image $\mathcal F$** | $(M_3 I) - ((M_3 I) M_7)$, small-scale 부드러운 구조 / Small-scale smooth-structure image. |
| **$\mathcal L^+/\mathcal F$ ratio** | CR-vs-source 식별 비율 / Discrimination ratio between CR and source. |
| **$\sigma_{\rm lim}, f_{\rm lim}$** | 두 임계값: SNR과 ratio / Two thresholds — significance and ratio. |
| **PSF sampling** | well/critically/under-sampled의 3 단계 / Sampling regimes; under-sampled HST WFPC2 needs $f_{\rm lim}\approx 5$. |
| **HST WFPC2** | under-sampled HST instrument (FWHM≈1.3 px) / Under-sampled HST camera. |
| **Iterative peeling** | 큰 CR을 외곽부터 점진적 제거 / Large CRs peeled outward layer-by-layer. |
| **L.A.Cosmic** | "Laplacian Cosmic-ray identification" — algorithm 이름 / Algorithm name (Laplacian + Cosmic). |

---

## 5. 수식 미리보기 / Equations Preview

### 한국어
**이산 Laplacian 커널 (Eq. 4)**:

$$
\nabla^2 f = \frac{1}{4}\begin{pmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{pmatrix}
$$

**잡음 모델 (Eq. 10)** — Poisson + read-noise:

$$
N_{i,j} = g^{-1}\sqrt{\,g\,(M_5\circ I)_{i,j} + \sigma_{\rm rn}^2\,}
$$

**유의도 영상 (Eq. 11)** — Laplacian SNR:

$$
S_{i,j} = \frac{\mathcal L^+_{i,j}}{f_s\,N_{i,j}}, \qquad f_s = 2
$$

**Fine-structure image (Eq. 14)**:

$$
\mathcal F = (M_3\circ I) - \big((M_3\circ I)\circ M_7\big)
$$

**CR 판별 기준** — 두 임계값 결합:

$$
\text{CR pixel} \iff S'_{i,j} > \sigma_{\rm lim} \;\wedge\; \mathcal L^{+}_{i,j}/\mathcal F_{i,j} > f_{\rm lim}
$$

### English
The $3\times 3$ Laplacian kernel highlights sharp edges (mean-zero, removes smooth structure). The noise model combines Poisson photon noise (gain-converted) with Gaussian read-noise, evaluated on a 5×5-median-smoothed image so the threshold adapts to local count rate. The significance map $S$ is essentially per-pixel Laplacian SNR. The fine-structure image $\mathcal F$ preserves small-scale smooth features (point sources have $\mathcal F\sim 50$ e$^-$, CRs have $\mathcal F\to 0$). A pixel is declared CR only if both criteria hold: high significance AND high $\mathcal L^+/\mathcal F$ ratio.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **꼭 정독할 부분**: §2 (Laplacian의 직관과 Eq. 3-4), §3.1 (basic procedure 6단계), §3.2 (sampling-flux removal과 fine-structure $\mathcal F$의 역할), §4.2 (HST WFPC2 결과와 $f_{\rm lim}=5$의 의미).
- **빠르게 훑을 부분**: §1 introduction 일부, §3.3 추가 기능, §4.3 spectroscopy 적용.
- **흔한 걸림돌 / Common stumbling blocks**:
  - "왜 2× 서브샘플링이 필수인가?" — 직접 컨볼루션은 인접 CR 픽셀의 negative cross-pattern이 *서로의 신호를 약화*시킴 (Eq. 12 참조). 서브샘플링이 이를 방지.
  - "$f_s = 2$가 어디서 오는가?" — sub-sampling factor (2×2 = 4픽셀 → block-average로 1픽셀로 환원). Laplacian 잡음 증가율 보정.
  - "fine-structure $\mathcal F$의 의미": $M_3$은 작은 sharp 노이즈 제거, $M_7$은 더 큰 구조 추출, 차이는 *작은 부드러운 구조* (=별 코어). CR은 이런 구조 없음 → ratio 큼.
  - "PSF 표본화에 따라 $f_{\rm lim}$이 달라지는 이유": well-sampled 별은 $\mathcal F$가 크므로 ratio 작음, under-sampled 별은 spike-like → $\mathcal F$ 작음 → ratio가 CR과 비슷해짐 → 더 큰 $f_{\rm lim}$ 필요.
- 동반 자료: Marr-Hildreth 1980 edge detection, astroscrappy/lacosmic Python 구현.

### English
- **Read carefully**: §2 (Laplacian intuition and Eqs. 3–4), §3.1 (the 6-step basic procedure), §3.2 (sampling-flux removal and the role of $\mathcal F$), §4.2 (HST WFPC2 results and why $f_{\rm lim}=5$).
- **Skim**: parts of §1, §3.3 extra features, §4.3 spectroscopic application.
- **Common stumbling blocks**:
  - Why 2× sub-sampling is essential — direct convolution lets adjacent CR pixels' negative cross-patterns suppress each other (see Eq. 12). Sub-sampling decouples them.
  - Where $f_s = 2$ comes from — the sub-sampling factor (2×2 → block-average back to 1 pixel) plus Laplacian-noise correction.
  - Meaning of $\mathcal F$ — $M_3$ removes small sharp noise, $M_7$ extracts larger smooth structure; the difference captures *small smooth structure* (star cores). CRs have no such structure → ratio is large.
  - Why $f_{\rm lim}$ depends on PSF sampling — well-sampled stars have large $\mathcal F$ (ratio small); under-sampled stars become spike-like with small $\mathcal F$ (ratio close to CRs), demanding higher $f_{\rm lim}$.
- Companion reading: Marr & Hildreth (1980); astroscrappy / lacosmic Python implementations.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
L.A.Cosmic은 발표 후 **20년간 거의 모든 HST 단일 노출 처리 파이프라인의 표준 도구**였으며, 현재 `astroscrappy`(Curtis McCully, C-extension), `ccdproc.cosmicray_lacosmic`(AstroPy), `lacosmic` 등 다수 Python 구현이 활발히 사용된다. deepCR(논문 #24, 2020) 같은 deep-learning 후속 연구도 L.A.Cosmic을 *표준 baseline*으로 삼아 ROC 비교를 수행한다. 알고리즘의 *training-free + dependency-light + 해석 가능* 특성은 GPU·대용량 데이터셋 없는 소규모 관측에서 여전히 강력하다. JWST·Roman·Euclid 차세대 망원경 파이프라인에서도 deep-learning 모델과 *상보적*으로 사용된다 (예: deepCR + L.A.Cosmic ensemble). 더 넓게는 — wavelet thresholding(논문 #1)과 같은 *변환→임계화→역변환* 패턴의 천문학 영상판이며, 같은 철학이 sparse coding, dictionary learning, low-rank decomposition 등 현대 영상 처리 전반에 흐른다.

### English
L.A.Cosmic was the **de-facto standard for HST single-exposure CR rejection from 2001 to ~2020**, and remains heavily used today via `astroscrappy` (Curtis McCully, C-extension), `ccdproc.cosmicray_lacosmic` (AstroPy ecosystem), and `lacosmic` (pure-Python). Deep-learning successors like deepCR (paper #24, 2020) all benchmark against L.A.Cosmic on ROC plots. Its *training-free, dependency-light, interpretable* design keeps it indispensable on small surveys without GPUs or labelled datasets. Next-generation pipelines (JWST, Roman, Euclid) use it *complementarily* with deep models (e.g., deepCR + L.A.Cosmic ensemble). More broadly, L.A.Cosmic is the astronomical analogue of the *transform → threshold → invert* template (cf. wavelet shrinkage, paper #1) — a philosophy that pervades modern image processing through sparse coding, dictionary learning, and low-rank methods.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
