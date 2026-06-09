---
title: "Cosmic-Ray Rejection by Laplacian Edge Detection"
authors: Pieter G. van Dokkum
year: 2001
journal: "Publications of the Astronomical Society of the Pacific (PASP), 113, 1420–1427"
doi: "10.1086/323894"
topic: Low-SNR Imaging / Cosmic-Ray Detection
tags: [cosmic-ray, ccd, laplacian, edge-detection, hst, wfpc2, single-exposure, classical-algorithm]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 23. Cosmic-Ray Rejection by Laplacian Edge Detection (L.A.Cosmic) / Laplacian 에지 검출에 의한 우주선 제거

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 **단일 CCD 노출**에서 우주선(cosmic ray, CR)을 안정적으로 검출/제거하는 **L.A.Cosmic** 알고리즘을 제안한다. 핵심 통찰은: *대기·광학에 의해 번지지 않은* CR은 천체보다 *훨씬 날카로운 에지*를 가진다는 점. 따라서 **2× 서브샘플링 + Laplacian 컨볼루션**으로 에지 위치(zero-crossing)를 찾고, **Poisson + read-noise 모델로 정의된 임계값**으로 CR 후보를 추출한 뒤, **fine-structure image $\mathcal F$로 대칭성**을 검사하여 별/은하 같은 *대칭적* 점광원과 *비대칭적* CR을 구분한다. 임의의 모양과 크기의 CR을 처리하고, *under-sampled PSF* (HST WFPC2 같은 경우)에서도 매개변수 $f_{\rm lim}$ 조정만으로 동작. 시뮬레이션에서 500 stars + 100 galaxies + 227 CRs 중 222개(98%) 검출, 단 1개(0.2%) 별 오검출.

### English
The paper introduces **L.A.Cosmic**, a robust algorithm for detecting and removing cosmic rays in a *single* CCD exposure. The key insight: cosmic rays — unlike astronomical sources — are not smeared by the atmosphere or optics, so they exhibit **markedly sharper edges**. The algorithm exploits this by:
(i) **2× sub-sampling** the image and convolving with the discrete Laplacian kernel — sharp CR edges produce strong signals while smoothly-sampled stars do not;
(ii) thresholding against a **noise model** built from Poisson + read-noise statistics; and
(iii) using a **fine-structure image $\mathcal F$** to discriminate *symmetric* point sources from *asymmetric* CRs via the ratio $\mathcal L^{+}/\mathcal F$.
The method handles CRs of arbitrary shape and size, works on both well-sampled and under-sampled data (HST WFPC2 with $f_{\rm lim}\approx 5$), and on a synthetic test field detected 222/227 CRs (98%) with only 1/500 stellar false positives (0.2%). It became the de-facto standard for HST single-exposure CR rejection through the 2010s.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction / §1 서론

#### 한국어
- 다중 노출 CR 제거(예: median stack)는 가장 단순하지만 (a) 변광/이동 천체, (b) 슬릿분광에서 sky/object spectrum이 시간 변동, (c) 노출 사이 시상 변동이 클 때 부적절. 단일 노출 CR 제거가 필요한 시나리오 다수.
- 기존 단일 노출 방법: median filtering (QZAP), PSF-matched filtering, neural network (Salzberg+ 1995), nearest-neighbour interpolation (IRAF COSMICRAYS). 모두 *PSF가 필터 크기보다 잘 표본화되어야* 동작하는 한계.
- 본 논문 목표: **임의 크기/모양 CR + 임의 PSF 표본화**에서 동작하는 단일 노출 알고리즘.

#### English
- Multi-exposure CR rejection (median stacks) is simplest but breaks down for (a) variable or moving sources, (b) long-slit spectroscopy with time-variable sky lines, (c) variable seeing between exposures. Single-exposure CR rejection is essential.
- Existing single-exposure methods (median filtering / QZAP, PSF-matched filtering, neural-net classifiers, nearest-neighbour interpolation in IRAF COSMICRAYS) all assume the PSF is well-sampled relative to the filter — they fail on under-sampled HST data.
- Goal: a single-exposure algorithm that works for arbitrary CR shapes and arbitrary PSF sampling.

### Part II: §2 The Laplacian / §2 Laplacian

#### 한국어
- 2D Laplacian: $\nabla^2 f = \partial^2 f/\partial x^2 + \partial^2 f/\partial y^2$. 영상의 에지에서 zero-crossing을 가짐 (Marr-Hildreth 1980 검출자).
- 2D Gaussian $f(x,y) = e^{-r^2/(2\sigma^2)}$의 Laplacian (Eq. 3): zero-crossing at $r = \pm\sqrt{2}\sigma$. CR은 매우 날카로운 에지 → 작은 컨볼루션 커널이 적합. 논문은 다음 $3\times 3$ kernel 사용 (Eq. 4):
$$
\nabla^2 f = \tfrac{1}{4}\begin{pmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{pmatrix}.
$$
- 평균값은 0 (smooth structure 제거).

#### English
- The 2D Laplacian highlights edges via zero-crossings (Marr-Hildreth detector). For a Gaussian PSF the Laplacian zero-crossings sit at $r = \pm\sqrt 2\,\sigma$, so a small kernel is optimal for the very sharp CR edges. The paper uses the canonical 3×3 discrete Laplacian (Eq. 4) with mean zero, which removes smoothly varying structure.

### Part III: §3 Implementation / §3 구현

#### 한국어 — §3.1 Basic procedure
1. **2× 서브샘플링 (Eq. 5)**: $I^{(2)}_{i,j} = I_{\lceil(i+1)/2\rceil, \lceil(j+1)/2\rceil}$. 각 픽셀을 4개로 복제. 이 단계가 핵심: 직접 컨볼루션은 인접 CR 픽셀이 negative cross-pattern으로 *서로의 신호를 약화*시킴. 서브샘플링은 이를 방지.
2. **Laplacian 컨볼루션 (Eq. 6)**: $\mathcal L^{(2)} = \nabla^2 f \circ I^{(2)}$.
3. **양수 부분만 보존 (Eq. 7)**: $\mathcal L^{(2+)} = \max(\mathcal L^{(2)}, 0)$ — 음수 cross-pattern 제거.
4. **블록 평균 → 원해상도 (Eq. 8)**: $\mathcal L^{+}_{i,j} = \tfrac{1}{4}(\mathcal L^{(2+)}_{2i-1,2j-1} + \cdots)$.
5. **잡음 모델 (Eq. 10)**: $N = g^{-1}\sqrt{g (M_5 \circ I) + \sigma_{\rm rn}^2}$ — Poisson(전자 단위) + read noise. $M_5$는 5×5 median.
6. **유의도 (Eq. 11)**: $S = \mathcal L^{+}/(f_s N)$, $f_s$는 서브샘플링 인자(=2). $S > \sigma_{\rm lim}$ (예: $5\sigma$) 픽셀이 CR 후보.

#### English — §3.1 Basic procedure
1. **2× sub-sample** (Eq. 5): replicate each pixel 4×. Without this step, direct convolution lets the negative cross-patterns of adjacent CR pixels suppress each other's signal.
2. **Laplacian convolve** (Eq. 6): $\mathcal L^{(2)} = \nabla^2 f \circ I^{(2)}$.
3. **Keep positives only** (Eq. 7): $\mathcal L^{(2+)} = \max(\mathcal L^{(2)}, 0)$, removing the cross-pattern artefacts.
4. **Block-average back to original resolution** (Eq. 8).
5. **Noise model** (Eq. 10): $N = g^{-1}\sqrt{g(M_5\circ I)+\sigma_{\rm rn}^2}$ — Poisson + read-noise; $M_5$ is a $5\times 5$ median (smooth estimate of expected counts).
6. **Significance** (Eq. 11): $S = \mathcal L^+/(f_s N)$ with $f_s=2$. Pixels with $S>\sigma_{\rm lim}$ (e.g. $5\sigma$) are CR candidates.

#### 한국어 — Detection probability (§3.1, Eq. 12)
단일 픽셀 CR의 검출 확률은 background noise에 의해 결정. Laplacian 자체가 잡음을 $\sqrt{5/4}$ 배 키움. $5\sigma$ threshold 적용 시:
- $4\sigma$ 피크: 5%만 CR로 표시 (낮은 검출률)
- $5\sigma$ 피크: 50%
- $6\sigma$ 피크: 95%

여러 픽셀이 연결된 큰 CR은 픽셀-간 변동이 적어 Laplacian 응답이 더 작음. 한계는 $S_{i,j} \sim N^{-1}_{i,j}(1 - n_{i,j}/4)(I_{i,j} - B_{i,j})$ where $n_{i,j}$ = adjacent CR pixel count. 큰 CR은 *반복 적용*으로 외곽부터 점진적으로 제거.

#### English — Detection probability
Single-pixel CR detection is noise-limited; the Laplacian raises the noise by a factor $\sqrt{5/4}$. With a $5\sigma$ threshold, the detection rate is $\sim 5\%$ at $4\sigma$, $\sim 50\%$ at $5\sigma$, $\sim 95\%$ at $6\sigma$. Connected multi-pixel CRs have weaker Laplacian responses (Eq. 12 shows the $1-n_{i,j}/4$ factor); large CRs are removed iteratively, peeling outward layers each pass.

#### 한국어 — §3.2 Sampling flux removal
실제 천체는 PSF로 *부드럽게 표본화*되므로 Laplacian이 작지만 0이 아님. 이 "sampling flux"가 별/은하에서 CR로 오검출되는 원인.

대응책:
1. 대형 구조 제거: $S' = S - (S \circ M_5)$, $5\times 5$ median 적용 후 차감.
2. **Fine-structure image $\mathcal F$ (Eq. 14)**: $\mathcal F = (M_3 \circ I) - ((M_3 \circ I) \circ M_7)$ — $3\times 3$ median에서 $7\times 7$ median 빼기. $\mathcal F$는 *작은 스케일 부드러운 구조* (별)는 보존, CR은 응답 작음.
3. CR 판별 기준 (Eq. 14): $S' > \sigma_{\rm lim}$ AND $\mathcal L^+/\mathcal F > f_{\rm lim}$. $f_{\rm lim}$은 PSF 표본화에 따라 조정 — well-sampled는 $f_{\rm lim}=2$, HST WFPC2 같은 under-sampled는 $f_{\rm lim}\approx 5$.

#### English — §3.2 Sampling flux removal
Astronomical objects produce small but nonzero Laplacian responses ("sampling flux"). Two-step removal:
1. Subtract the large-scale structure: $S' = S - (S\circ M_5)$.
2. Build a **fine-structure image** $\mathcal F = (M_3\circ I) - ((M_3\circ I)\circ M_7)$ that preserves small-scale smooth features (point sources) but stays small for CRs.
3. Decision: pixel is CR if $S' > \sigma_{\rm lim}$ AND $\mathcal L^+/\mathcal F > f_{\rm lim}$. Default $f_{\rm lim}=2$ for well-sampled data; HST WFPC2 (under-sampled) needs $f_{\rm lim}\approx 5$.

### Part IV: §3.3 Additional features and §4 Examples / §3.3 추가 기능과 §4 예제

#### 한국어
- **반복 적용**: 첫 패스 후 식별된 CR 픽셀의 *이웃*은 더 낮은 임계값을 적용 → 큰 CR의 *희미한 가장자리*까지 점진 제거.
- **CR 픽셀 대체**: median of surrounding "good" 픽셀로 대체.
- **장슬릿 분광 옵션**: sky 라인과 object 스펙트럼을 적합/제거 후 Laplacian 컨볼루션 → 스펙트럼 영상에 특화.
- **연산 비용**: 800×800 픽셀, Sun UltraSparc 1 (200 MHz)에서 iter당 65초 (2001년 기준). 현대 하드웨어/numpy로 1초 미만.

- §4.1 Well-sampled artificial: 500 stars / 100 galaxies / 227 CRs ($\geq 5\sigma$). L.A.Cosmic이 222/227 CR 검출 (98%), 1/500 stars (0.2%) 오검출, galaxies 0/100 오검출.
- §4.2 HST WFPC2: undersampled PSF에서도 $f_{\rm lim}=5$ 설정으로 잘 동작. MS 1053-04 클러스터 ($z=0.58$) WFPC2 영상 5750 CR 중 5638 (98.1%) 검출.
- §4.3 Spectroscopic: Keck LRIS 1800s 슬릿 스펙트럼에서 fringe pattern 제거 + emission line 보존.

#### English
- **Iterative application**: after the first pass, neighbouring pixels of detected CRs use a lower threshold so large CRs get peeled inward.
- **CR replacement**: median of surrounding "good" pixels.
- **Long-slit spectroscopy**: optional fitting and subtraction of sky lines and object spectrum before Laplacian convolution.
- **Speed**: $\sim 65$ s/iteration on 800×800 image on a Sun UltraSparc 1 (200 MHz, 2001); modern numpy makes it sub-second.

- §4.1 well-sampled synthetic test: 500 stars + 100 galaxies + 227 CRs ($\geq 5\sigma$). L.A.Cosmic detects 222/227 (98%), with 1/500 stellar (0.2%) and 0/100 galactic false positives.
- §4.2 HST WFPC2 with $f_{\rm lim}=5$ works well despite undersampled PSF; on a MS 1053-04 ($z=0.58$) WFPC2 field 5638/5750 CRs detected (98.1%).
- §4.3 Spectroscopic Keck LRIS 1800 s slit spectrum: removes fringes while preserving emission lines.

### Part V: §3.2 PSF-sampling-dependent threshold $f_{\rm lim}$ / PSF 표본화에 따른 $f_{\rm lim}$

#### 한국어
- 핵심 통찰: 잘 표본화된 별과 critically-sampled 별은 $\mathcal L^+/\mathcal F$ 비율이 다름. Fig. 4 (paper)는 $f_{\rm lim}$이 별의 FWHM의 함수로 어떻게 변하는지 보여줌.
- well-sampled (FWHM ≥ 2.5 px): $\mathcal L^+/\mathcal F \lesssim 1$. 따라서 $f_{\rm lim}=2$ default가 안전.
- HST WFPC2 (FWHM ≈ 1.3 px): 별의 비율이 ~3까지 올라감 → $f_{\rm lim}=5$ 필요.
- 직관: PSF가 잘 표본화될수록 *별의 fine structure*가 살아있어 $\mathcal F$가 크고, ratio가 작아짐. Under-sampling은 별을 spike-like로 만들어 $\mathcal F$가 작아지고 ratio가 커짐 — CR과 구분이 어려워지므로 더 큰 $f_{\rm lim}$ 필요.

#### English
- Key insight: well-sampled and critically-sampled stars give different $\mathcal L^+/\mathcal F$ ratios. Fig. 4 of the paper plots this ratio versus stellar FWHM.
- Well-sampled (FWHM ≥ 2.5 px): $\mathcal L^+/\mathcal F \lesssim 1$, so the default $f_{\rm lim}=2$ is safe.
- HST WFPC2 (FWHM ≈ 1.3 px): the stellar ratio rises to ~3, requiring $f_{\rm lim}=5$.
- Intuition: better sampling preserves the *fine structure* of stars (large $\mathcal F$, small ratio); under-sampling makes stars spike-like (small $\mathcal F$, large ratio), bringing them closer to CRs in the ratio space and demanding a higher $f_{\rm lim}$ to discriminate.

---

## 3. Key Takeaways / 핵심 시사점

1. **CR과 천체의 차이는 에지의 날카로움 / CR vs source: edge sharpness** — CR은 대기·광학에 번지지 않으므로 PSF보다 훨씬 날카로움. Laplacian은 sharp edge 검출자 → CR과 source 분리에 직접 사용.
   The defining feature of cosmic rays vs astronomical sources is *edge sharpness*: CRs are not smeared by the PSF, so a Laplacian (a sharp-edge detector) cleanly separates them.

2. **2× 서브샘플링은 알고리즘의 핵심 / 2× sub-sampling is essential** — 직접 컨볼루션은 인접 CR 픽셀이 *서로의 음의 cross-pattern*을 통해 검출 신호를 줄임. 서브샘플링은 이를 방지.
   Without 2× sub-sampling, the negative cross-patterns of adjacent CR pixels suppress each other's signal. Sub-sampling decouples them.

3. **잡음 모델은 Poisson + read-noise / Noise model is Poisson + read-noise** — Eq. 10. 임의의 영상에 대해 *현실적 잡음 추정*을 해주므로 임계값이 데이터에 자동 적응. 단, gain $g$와 read-noise $\sigma_{\rm rn}$은 사전에 알아야 함.
   The noise model $N = g^{-1}\sqrt{g\langle I\rangle + \sigma_{\rm rn}^2}$ adapts the threshold to the local count rate; gain and read-noise must be pre-known.

4. **Fine-structure image가 별/CR 구분의 두 번째 안전장치 / Fine-structure image is the second discriminator** — 잘 표본화된 별은 $\mathcal L^+ \sim \mathcal F$ → 비율 작음. CR은 $\mathcal F\approx 0$ → 비율 큼. Threshold $f_{\rm lim}$은 PSF 표본화에 따라 조정 (well-sampled $\to 2$, under-sampled $\to 5$).
   The fine-structure image $\mathcal F$ provides the second criterion. Well-sampled stars give $\mathcal L^+ / \mathcal F \sim 0.7$, undersampled stars $\sim 1.8$, CRs $\geq 21$. The threshold $f_{\rm lim}$ adapts to PSF sampling.

5. **반복 적용으로 큰 CR 처리 / Iterative peeling for large CRs** — 한 패스로 큰 CR의 외곽 픽셀만 검출. 반복에서 *이미 식별된 CR의 이웃*에 낮은 임계값 적용 → 점차 안쪽으로 진행. 큰 CR도 처리 가능.
   Large CRs are peeled iteratively: each pass detects the outer rim, and neighbours of detected pixels use a lowered threshold in the next pass.

6. **HST WFPC2 같은 under-sampled 데이터에 강함 / Robust on under-sampled HST data** — 다른 단일 노출 알고리즘이 무너지는 under-sampled regime에서 $f_{\rm lim}\approx 5$로 정상 작동. 이것이 본 알고리즘의 *대표적 실용 가치*.
   The algorithm shines on under-sampled HST WFPC2 data, where other single-exposure methods fail; this is L.A.Cosmic's flagship use case.

7. **임의 모양/크기 CR 처리 / Handles CRs of arbitrary shape and size** — Eq. 12의 $1 - n_{i,j}/4$ 인자가 큰 CR의 검출률을 *낮추지만*, 반복 적용이 보완. 결과적으로 길쭉한 muon track도 검출.
   The $1 - n_{i,j}/4$ factor reduces detection probability for connected multi-pixel CRs, but the iterative peeling compensates — long muon tracks are eventually fully removed.

8. **고전 알고리즘이지만 deep learning 시대에도 baseline / Classical yet still a deep-learning baseline** — deepCR (Zhang & Bloom 2020, paper #24) 같은 최신 deep CR detector도 L.A.Cosmic을 표준 비교 대상으로 사용. 여전히 *fast, dependency-free, no-training* 기준.
   Even modern deep CR detectors like deepCR (paper #24) use L.A.Cosmic as their standard baseline; it remains the reference for "fast, training-free, dependency-light" CR rejection.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Setting / 설정
$$
I_{i,j} = B_{i,j} + S_{i,j} + R_{i,j} + C_{i,j},
$$
$B$: smooth sky background; $S$: source flux (PSF-convolved); $R$: read+shot noise; $C$: cosmic-ray contribution (sharp, asymmetric).

### 4.2 Discrete Laplacian kernel (Eq. 4)
$$
\nabla^2 f = \frac{1}{4}\begin{pmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{pmatrix}.
$$
### 4.3 2× sub-sampling (Eq. 5)
$$
I^{(2)}_{i,j} = I_{\lceil(i+1)/2\rceil,\,\lceil(j+1)/2\rceil}, \quad i,j = 1,\ldots, 2n.
$$
### 4.4 Positive Laplacian and re-binning (Eqs. 6–8)
$$
\mathcal L^{(2)} = \nabla^2 f \circ I^{(2)}, \quad \mathcal L^{(2+)} = \max(\mathcal L^{(2)}, 0),
$$
$$
\mathcal L^{+}_{i,j} = \tfrac{1}{4}\big(\mathcal L^{(2+)}_{2i-1,2j-1} + \mathcal L^{(2+)}_{2i-1,2j} + \mathcal L^{(2+)}_{2i,2j-1} + \mathcal L^{(2+)}_{2i,2j}\big).
$$
### 4.5 Noise model (Eq. 10)
$$
N_{i,j} = g^{-1}\sqrt{\,g\,(M_5\circ I)_{i,j} + \sigma_{\rm rn}^2\,}.
$$
$g$ = gain (e$^-$/ADU), $\sigma_{\rm rn}$ = read-noise (e$^-$), $M_5$ = $5\times 5$ median filter.

### 4.6 Significance image (Eq. 11)
$$
S_{i,j} = \frac{\mathcal L^{+}_{i,j}}{f_s\,N_{i,j}}, \qquad f_s = 2.
$$
### 4.7 Sampling-flux removal (Eq. 13)
$$
S'_{i,j} = S_{i,j} - (S \circ M_5)_{i,j}.
$$
### 4.8 Fine-structure image (Eq. 14)
$$
\mathcal F = (M_3\circ I) - ((M_3\circ I)\circ M_7).
$$
### 4.9 CR criterion
$$
\text{CR pixel} \iff S'_{i,j} > \sigma_{\rm lim} \;\wedge\; \mathcal L^{+}_{i,j}/\mathcal F_{i,j} > f_{\rm lim}.
$$
Defaults: $\sigma_{\rm lim}=5$, $f_{\rm lim}=2$ (well-sampled) or $5$ (HST WFPC2).

### 4.10 Worked numerical example / 수치 예시
Take a small image with sky background $B = 100$ e$^-$, gain $g=1$ e$^-$/ADU, read-noise $\sigma_{\rm rn}=5$ e$^-$.
Inject a 1-pixel CR with intensity $C=300$ e$^-$ above background and a star with peak 250 e$^-$ FWHM 2.5 px.
- Star Laplacian: $\nabla^2 f \approx (4\cdot 250 - 4\cdot 200) /4 = 50$ e$^-$ at peak, but block-averaged down further.
- CR Laplacian (after 2× sub-sample): $\nabla^2 f \approx (4 \cdot 400 - 4 \cdot 100)/4 = 300$ — full CR signal.
- Noise: $N = \sqrt{100 + 25} = 11.2$ e$^-$. Significance: $S_{\rm CR} = 300/(2\cdot 11.2)=13.4\sigma$ — easily above $5\sigma$.
- Star $S\approx 50/(2\cdot 11.2)\approx 2.2\sigma$ — below threshold.
- Fine-structure: star $\mathcal F\approx 250 - 200 = 50$, CR $\mathcal F\approx 300 - 300 = 0$ → ratio $\mathcal L^+/\mathcal F \to \infty$ for CR, $\sim 1$ for star → easily separated by $f_{\rm lim}=2$.

### 4.11 Detection-probability table (from §3.1) / 검출 확률 표
| CR peak (in $\sigma$) | Marked as CR (5σ threshold) |
|---|---|
| $4\sigma$ | $\sim 5\%$ |
| $5\sigma$ | $\sim 50\%$ |
| $6\sigma$ | $\sim 95\%$ |
| $7\sigma$ | $\gtrsim 99\%$ |

Rule of thumb: $5\sigma$ threshold misses ~half of $5\sigma$ CRs but virtually no $6\sigma$+ CRs.

### 4.12 Asymptotic behaviour for connected CRs (Eq. 12) / 연결된 CR
$$
S_{i,j} \sim N^{-1}_{i,j}(1 - n_{i,j}/4)(I_{i,j} - B_{i,j}),
$$
where $n_{i,j}\in\{0,1,2,3,4\}$ is the count of adjacent CR pixels. Worst case $n=2$ on corners gives factor $1/2$; iteration recovers the rest.

### 4.13 Algorithm pseudocode / 알고리즘 의사코드
```
Input: image I (counts in e-), gain g, read noise sigma_rn,
       sigclip (default 5), objlim (default 2 or 5), n_iter (default 4)
mask_total = zeros(I.shape)
For k = 1, ..., n_iter:
    I_2 = upsample_2x(I)                          # Eq. 5
    L_2 = convolve(I_2, laplacian_kernel)         # Eq. 6
    L_2_pos = max(L_2, 0)                          # Eq. 7
    L_pos = block_average_2x(L_2_pos)              # Eq. 8
    N = (1/g) * sqrt( g * median_filter(I, 5) + sigma_rn^2 )   # Eq. 10
    S = L_pos / (2 * N)                            # Eq. 11
    S_prime = S - median_filter(S, 5)              # Eq. 13
    F = median_filter(I, 3) - median_filter(median_filter(I, 3), 7)  # Eq. 14
    new_mask = (S_prime > sigclip) AND (L_pos / F > objlim)
    new_mask &= ~mask_total                        # only newly found pixels
    if new_mask.sum() == 0: break
    I = where(new_mask, median_filter(I, 5), I)    # replace
    mask_total |= new_mask
Return I, mask_total
```

### 4.14 Comparison to a Gaussian-PSF Laplacian / 가우시안 PSF Laplacian과 비교
For a 2D Gaussian PSF $f = \exp(-r^2/2\sigma^2)$, $\nabla^2 f$ has zero crossings at $r=\pm\sqrt 2\sigma$. The kernel in Eq. 4 has zero crossings at radii ~$\sqrt 2$ pixel. So:
- For a star with PSF-$\sigma$ = 1.5 px (FWHM ≈ 3.5 px), the kernel zero-crossings are well inside the PSF — the Laplacian sees a "smooth" star.
- For a single-pixel CR, the Laplacian kernel is matched in scale — full response.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1980 ─── Marr-Hildreth — Theory of edge detection (zero-crossings of Laplacian-of-Gaussian)
1992 ─── Gonzalez & Woods — Digital Image Processing (Laplacian as standard tool)
1994 ─── Windhorst, Franklin, Neuschaefer — Multi-exposure CR removal for HST
1995 ─── Salzberg+ — Decision-tree CR classifier
1997 ─── Fruchter & Hook — Drizzle algorithm (multi-exposure stacking)
2000 ─── Rhoads — Linear filtering CR detector
2001 ★★ VAN DOKKUM: L.A.Cosmic
                 ↳ single-exposure CR rejection via Laplacian + sub-sampling + fine-structure
2004 ─── Pych — Histogram-based CR detection
2005 ─── Farage & Pimbblet — Compare CR rejection methods (L.A.Cosmic ranked highest)
2012 ─── astropy.lacosmic — Python wrapper
2016 ─── Desai+ — DECam CR rejection survey
2020 ★ ZHANG-BLOOM — deepCR (paper #24 in this study)
                 ↳ U-Net beats L.A.Cosmic on HST ACS, but uses L.A.Cosmic as baseline
2022+ ── Hybrid classical/deep CR pipelines for JWST, Roman, Euclid
```

이 알고리즘은 **2001년부터 2020년까지 거의 모든 HST 단일 노출 데이터 처리 파이프라인의 표준 도구**였다. deepCR 등장 후에도 baseline으로 살아있으며, training-free·해석 가능성·의존성 없음 측면에서 여전히 강력하다.

This algorithm has been **the de-facto standard for HST single-exposure CR rejection from 2001 through ~2020**. Even after deepCR, L.A.Cosmic remains the standard baseline thanks to its training-free, interpretable, dependency-light design.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Marr & Hildreth (1980)** *Proc. R. Soc. Lond. B* | Foundational theory of zero-crossings as edge detectors. | L.A.Cosmic's Laplacian convolution is a direct instance of Marr-Hildreth detection applied to CR identification. |
| **Salzberg et al. (1995)** *PASP* | Decision-tree CR classifier, ML predecessor. | Compared in §1; L.A.Cosmic outperforms it on HST WFPC2. |
| **Rhoads (2000)** *PASP* | Linear-filter CR detection. | Predecessor that L.A.Cosmic improves on by adding sub-sampling and fine-structure discrimination. |
| **Fruchter & Hook (1997, 2002)** *Drizzle* | Multi-exposure dithered imaging. | Complementary multi-exposure approach; L.A.Cosmic targets the single-exposure case Drizzle cannot handle. |
| **Zhang & Bloom (2020)** *deepCR* (paper #24) | Deep-learning successor. | Uses L.A.Cosmic as the standard baseline; deepCR achieves higher TPR at same FPR but at the cost of training data and GPU. |
| **astroscrappy / lacosmic** *Python ports* | Modern reproductions. | `astropy.lacosmic`, `astroscrappy` (Curtis McCully) provide drop-in implementations used across solar/space-physics pipelines. |
| **Donoho-Johnstone (1994)** *VisuShrink* (paper #1) | Both share the philosophy of *transform → threshold by noise model → invert*. | L.A.Cosmic's Laplacian + thresholding mirrors the wavelet + soft-thresholding template, applied to CR rather than denoising. |

---

### 6.1 Practical parameter cheat-sheet / 실용 매개변수 치트시트

#### 한국어
| 데이터 / Data | $\sigma_{\rm lim}$ | $f_{\rm lim}$ | $n_{\rm iter}$ | 비고 / Note |
|---|---|---|---|---|
| Well-sampled CCD | 5 | 2 | 4 | Default |
| HST WFPC2 (under-sampled) | 4.5 | 5 | 4 | larger $f_{\rm lim}$ |
| HST ACS (better sampled) | 5 | 3 | 3 | intermediate |
| Long-slit spectrum | 5 | 2 | 4 | with sky/object subtract |
| Short exposure (low S/N) | 4 | 2 | 5 | more iterations |

#### English
Same table — defaults are conservative; tune via the L^+/F histogram of a known star field.

### 6.2 Common failure modes / 흔한 실패 사례

#### 한국어
- **별이 critically sampled에 가까울 때**: $\mathcal L^+/\mathcal F$가 별과 CR이 비슷 → false positive 증가. 해결: $f_{\rm lim}$ 상향, 또는 별 마스크 사전 제거.
- **flat-fielding 잔차**: dust speck 같은 잔차가 sharp edge를 가져 CR로 오인. 해결: bad-pixel mask 적용.
- **확장된 천체 (e.g., 은하 nucleus)**: nucleus의 sharp 코어가 CR-like. 해결: $f_{\rm lim}$ 조정 + post-processing 매뉴얼 검토.

#### English
- **Stars near critically sampled**: stars and CRs share similar $\mathcal L^+/\mathcal F$ → more false positives. Fix: raise $f_{\rm lim}$ or mask out star catalog beforehand.
- **Flat-field residuals**: dust specks etc. produce sharp edges and are flagged as CRs. Fix: apply a bad-pixel mask.
- **Extended objects (galaxy nuclei)**: sharp galaxy cores look CR-like. Fix: tune $f_{\rm lim}$ and review post-processed regions manually.

### 6.3 Modern Python implementations / 최신 Python 구현

#### 한국어
- `astroscrappy` (Curtis McCully): 가장 빠른 C-extension 구현. `pip install astroscrappy`. `astroscrappy.detect_cosmics()`로 한 줄 호출.
- `ccdproc.cosmicray_lacosmic`: AstroPy ecosystem의 표준 wrapper. CCDData 객체와 통합.
- `lacosmic` (조시 블룸 group): 순수 Python, 교육·검증 용도.
- 본 study 노트북: `numpy + scipy`만으로 reference 구현 제공.

#### English
- `astroscrappy` (Curtis McCully): fastest C-extension port. `pip install astroscrappy`; one-line API `astroscrappy.detect_cosmics()`.
- `ccdproc.cosmicray_lacosmic`: AstroPy ecosystem wrapper that integrates with CCDData objects.
- `lacosmic` (pure-Python from Bloom group): for educational/validation use.
- This study notebook: provides a reference implementation in pure `numpy + scipy`.

---

## 7. References / 참고문헌

- van Dokkum, P. G., "Cosmic-Ray Rejection by Laplacian Edge Detection", *PASP*, 113, 1420–1427 (2001). [DOI: 10.1086/323894]
- Marr, D., & Hildreth, E., "Theory of Edge Detection", *Proc. R. Soc. Lond. B*, 207, 187–217 (1980).
- Salzberg, S., Chandar, R., Ford, H., Murthy, S. K., & White, R., "Decision Trees for Automated Identification of Cosmic-Ray Hits in Hubble Space Telescope Images", *PASP*, 107, 1–10 (1995).
- Rhoads, J. E., "Cosmic-Ray Rejection by Linear Filtering of Single Images", *PASP*, 112, 703–710 (2000).
- Fruchter, A. S., & Hook, R. N., "Linear Reconstruction of the Hubble Deep Field", *PASP*, 114, 144–152 (2002).
- Pych, W., "A Fast Algorithm for Cosmic-Ray Removal from Single Images", *PASP*, 116, 148–153 (2004).
- Farage, C. L., & Pimbblet, K. A., "Evaluation of Cosmic Ray Rejection Algorithms on Single-Shot Exposures", *PASA*, 22, 249–256 (2005).
- Zhang, K., & Bloom, J. S., "deepCR: Cosmic Ray Rejection with Deep Learning", *ApJ*, 889, 24 (2020).
- Gonzalez, R. C., & Woods, R. E., *Digital Image Processing*, Addison-Wesley, 1992.
- Code: Curtis McCully's `astroscrappy` (Python C-extension), `astropy.lacosmic` wrappers.
