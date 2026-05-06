---
title: "A Wavelet Packets Equalization Technique to Reveal the Multiple Spatial-Scale Nature of Coronal Structures"
authors: G. Stenborg, P. J. Cobelli
year: 2003
journal: "Astronomy & Astrophysics"
doi: "10.1051/0004-6361:20021687"
topic: Low_SNR_Imaging
tags: [wavelet, à-trous, wavelet-packets, coronagraph, LASCO, image-enhancement, multiresolution]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 32. A Wavelet Packets Equalization Technique to Reveal the Multiple Spatial-Scale Nature of Coronal Structures / 코로나 구조의 다중 공간 스케일을 드러내는 웨이블릿 패킷 평활화 기법

---

## 1. Core Contribution / 핵심 기여

Stenborg & Cobelli (2003) deliver a complete *multiresolution image-processing pipeline* tailored to white-light coronagraph data. The pipeline is built from four well-known but rarely combined tools: (i) the 2D **à trous wavelet transform** with a $B_3$-spline scaling kernel, (ii) **wavelet-packet** refinement that recursively decomposes each first-level wavelet plane into sub-scales, (iii) **local hard-thresholding** with per-scale, per-pixel noise variance estimated from the data via the Anscombe transform, and (iv) **interactive weighted recomposition** in which the user assigns weights $\alpha_{i,j}$ to each (scale, sub-scale) plane. The combined effect is *frequency equalisation* of a coronagraph image: faint loops, CME cores, prominence filaments, and other low-contrast structures that are buried under the radial gradient and noise are recovered, while the user retains full control over which spatial scales to highlight or suppress. The technique is demonstrated on LASCO C1 (Fe XIV inner-corona loops, 1998-05-21), C2 (CME on 2002-08-13, prominence on 1998-06-02), and C3 (CME on 2002-07-04). It has since become a standard tool in heliophysics and underlies the later MGN filter.

본 논문은 백색광 코로나그래프 영상에 특화된 완전한 다중분해(multiresolution) 영상처리 파이프라인을 제시한다. 핵심 도구는 (i) $B_3$-스플라인 커널을 사용하는 **2D à trous 웨이블릿 변환**, (ii) 1차 분해 결과를 다시 분해하는 **웨이블릿 패킷**, (iii) Anscombe 변환을 통해 *데이터 자체에서 추정한 국소 노이즈 분산*에 기반한 **국소 hard-threshold** 잡음 제거, (iv) 사용자가 가중치 $\alpha_{i,j}$를 정해 스케일별 강도를 조절하는 **대화형 가중 재합성**의 네 단계다. 이는 일종의 *주파수 평활화(equalization)* 효과를 내며, radial gradient와 noise에 가려져 보이지 않던 loop, CME 내부 구조, prominence filament 같은 미세 구조를 드러낸다. LASCO C1·C2·C3의 다양한 사례로 그 위력을 입증하며, 이후 MGN(2014) 등 후속 코로나 영상처리 도구의 직계 조상이 된다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction and Motivation / 도입과 동기 (Sec. 1, p. 1185)

The authors begin from a heliophysics motivation: precise determination of CME *onset times* requires sharp tracking of close-to-limb structures, which in turn requires unambiguous detection. Coronagraph exposure times are long enough that fast-moving features smear over several pixels; the authors quote the displacement formula

$$
d = \frac{v\,\tau_e}{r_s\cdot 700\;\text{km}}\;\text{pixels}
$$

with $v$ the plane-of-sky speed, $\tau_e$ the exposure time, and $r_s$ the spatial resolution per pixel. For an 800 km/s feature at the equator with a 20 s exposure on a 5.6 arcsec/pixel coronagraph, $d \approx 4$ pixels — the leading edge is smeared and contrast is lost. Faint diffuse features may not be discernible at all. The motivation for *enhanced imaging* is thus operational, not just aesthetic.

CME onset 시간을 정확히 결정하려면 limb 근처 구조를 선명히 추적해야 한다. 위 식은 노출시간 동안 구조가 번지는 픽셀 수를 준다. 800 km/s, 20 s 노출, 5.6 arcsec/pixel이면 $d\approx4$ pixel이 번져 contrast가 떨어진다. 따라서 영상 향상은 미적 문제가 아니라 *측정 정확도*의 문제다.

### Part II: Wavelet Transform Background / 웨이블릿 변환 배경 (Sec. 2.1, p. 1186)

The Continuous Wavelet Transform (CWT) is reviewed as the *time/scale-localised* alternative to Fourier:

$$
W_c f(a,b) = \frac{1}{\sqrt{a}}\int_{-\infty}^{\infty} f(x)\,\psi^*\!\Big(\frac{x-b}{a}\Big)\,dx \tag{Eq. 2}
$$

with $\psi$ the *mother wavelet*, $a>0$ the scale, $b$ the position. The CWT's three properties — linearity, translation covariance, and dilation covariance — are listed (Eqs. 3-5). Crucially the CWT is *invertible* (Eq. 6), so each "wavelet plane" can be re-decomposed without losing information — the property that makes wavelet packets possible.

CWT는 Fourier와 달리 *시간/스케일 동시 국소화*를 제공한다. mother wavelet $\psi$를 평행이동·확대축소하여 얻는 기저로 신호를 분해하며, 선형성·평행이동 공변성·확대축소 공변성·가역성을 갖는다. 각 wavelet plane을 다시 분해해도 정보가 보존되므로 wavelet packet이 가능하다.

### Part III: 2D à trous Algorithm / 2D à trous 알고리즘 (Sec. 2.2, p. 1187)

The à trous ("with holes") algorithm is the discrete, *redundant*, *isotropic* wavelet transform of choice for astronomy. Unlike Mallat's pyramidal DWT, à trous does not down-sample — every wavelet plane has the same size as the input image, so spatial registration with the original is trivial. The recursion is:

$$
c_i(k,l) = \sum_{m,n} h(m,n)\,c_{i-1}(k+2^{i-1}m,\,l+2^{i-1}n) \tag{Eq. 10}
$$

$$
w_i(k,l) = c_{i-1}(k,l) - c_i(k,l) \tag{Eq. 11}
$$

with $h$ the discrete low-pass filter associated with the chosen scaling function. The authors choose the cubic $B_3$-spline, leading to the separable 5×5 mask

$$
\frac{1}{256}\begin{pmatrix}
1 & 4 & 6 & 4 & 1 \\
4 & 16 & 24 & 16 & 4 \\
6 & 24 & 36 & 24 & 6 \\
4 & 16 & 24 & 16 & 4 \\
1 & 4 & 6 & 4 & 1
\end{pmatrix}.
$$

The reconstruction identity is exact:

$$
c_0(k,l) = c_p(k,l) + \sum_{i=1}^{p} w_i(k,l) \tag{Eq. 12}
$$

so the original image equals the smoothed (continuum) array $c_p$ plus all wavelet planes — a finite, additive, lossless decomposition. The key advantages of à trous in astronomy: (1) isotropy (no preferred direction), (2) redundancy (so structures persist across scales even when buried in noise), (3) preservation of resolution at every scale.

à trous 변환은 *down-sampling 없이* 각 스케일 평면이 원본과 같은 크기이므로 천문 영상에 적합하다. $B_3$-스플라인을 scaling function으로 선택하면 위 5×5 마스크가 나오며, $c_i$는 $i$-단계 평활화, $w_i = c_{i-1}-c_i$가 detail이다. 식(12)는 *정확한 가역성*을 보장한다. 등방성·중복성 덕분에 구조는 여러 스케일에 걸쳐 동시에 나타나며 noise에 강하다.

### Part IV: Multi-Level Wavelet-Packet Decomposition / 다단계 웨이블릿 패킷 분해 (Sec. 2.3, p. 1188)

This is the paper's *novelty axis*: rather than truncate or threshold at the finest scale, *split* each wavelet plane $w^{(0)}_i$ once more, producing $\{w^{(0,1)}_i,\ldots,w^{(0,p_{0,i})}_i\}$ — a second decomposition level. Each new level is itself an additive decomposition (Eq. 13 of paper). The labelling scheme uses superscripts equal to the decomposition level: $w^{(0,1)}_i$ means "second-level subscale of first-level plane $i$." The scheme is a tree (Fig. 1 of paper) — full at all branches if every $w_i$ is re-decomposed. The reason this matters: in coronagraph data, the higher-frequency scales are noise-dominated, and naive thresholding throws away signal with the noise. Splitting these scales further before thresholding lets the user preserve signal at intermediate sub-scales while still rejecting the noisiest sub-bands. This is the *equalisation* mechanism.

이 절이 본 논문의 핵심 *차별성*이다. 가장 미세한 스케일에서 단순 truncate/threshold 대신, $w^{(0)}_i$ 각각을 다시 분해해 $\{w^{(0,1)}_i,\ldots\}$의 두 번째 레벨을 만든다. 이 트리(Fig. 1)는 깊이만큼 더 세밀한 주파수 분해를 제공한다. 코로나그래프 데이터에서 높은 주파수 스케일은 noise가 지배적이므로 단순 threshold가 신호를 버리는데, 한 번 더 쪼갠 뒤 *서브-스케일 단위로* threshold하면 신호를 살릴 수 있다.

### Part V: Reconstruction of Structures (the four-step pipeline) / 구조 재구성 (Sec. 2.4, p. 1188)

This is where the entire algorithm comes together. Four numbered steps:

**Step 1 — Anscombe variance stabilisation.** For Poisson-dominated noise (CCD photon counting):

$$
t[I(x,y)] = 2\sqrt{I(x,y)+\tfrac{3}{8}} \tag{Eq. 14}
$$

makes the noise approximately Gaussian with $\sigma=1$, so the rest of the pipeline can use Gaussian threshold theory.

**Step 2 — Local noise variance estimation.** Compute the wavelet transform of a synthetic image of unit-variance Gaussian noise and read off the per-scale noise stdev $\hat{\sigma}^{(\cdot)}_j$. For the *spatial* dependence, take an $N\times N$ neighbourhood (the paper uses $N=5$) at each pixel and compute the local stdev $\sigma_1(k,l)$ at the finest first-level scale. Then the per-scale, per-pixel local stdev is the product:

$$
\sigma^{(\cdot)}_m(k,l) = \hat{\sigma}^{(\cdot)}_m \cdot \sigma_1(k,l) \tag{Eq. 16}
$$

This local-and-per-scale model is more flexible than a single global $\sigma$ and is the key to handling the wide dynamic range of coronal images (where noise itself varies with brightness).

**Step 3 — Hard-threshold in wavelet space.**

$$
W^{(\cdot)}_m(k,l) = \begin{cases}
0, & |w^{(\cdot)}_m(k,l)| < k\,\sigma^{(\cdot)}_m(k,l) \\
w^{(\cdot)}_m(k,l), & |w^{(\cdot)}_m(k,l)| \ge k\,\sigma^{(\cdot)}_m(k,l)
\end{cases}
\tag{Eq. 15}
$$

The authors use $k=3$, the $3\sigma$ rule, retaining coefficients with 99.7% confidence of being signal.

**Step 4 — Weighted recomposition.**

$$
\mathcal{I} = \sum_{i=0}^{p_0}\sum_{j=0}^{p_{0,i}} \alpha_{i,j}\,W^{(0,i)}_j \tag{Eq. 17}
$$

where $\alpha_{i,j}$ are user-chosen weights. Setting all $\alpha_{i,j}=1$ recovers the (denoised) input exactly. Setting different weights *equalises* the spectrum: under-weight the dominant low-frequency continuum and over-weight intermediate scales, and faint structure pops out. The "reconstruction strategy" — the choice of weight vector — is the user's only tuning knob, and Sec. 3 shows several effective strategies for different scientific goals.

핵심 4단계 파이프라인:
1. **Anscombe 변환** (식 14)으로 Poisson noise를 가우시안화.
2. **국소 노이즈 분산 추정** — 단위분산 시뮬레이션으로 per-scale $\hat{\sigma}$를 얻고, 영상의 finest scale에서 $N\times N=5\times 5$ 이웃의 국소 표준편차 $\sigma_1(k,l)$을 곱해 식(16) 만든다.
3. **국소 hard-threshold** (식 15) — $k=3$이면 99.7% 신뢰. 
4. **가중 재합성** (식 17) — 사용자가 $\alpha_{i,j}$로 스케일별 강도를 조절. 이것이 *equalization*의 본체다.

### Part VI: Applications / 응용 (Sec. 3, pp. 1189-1192)

Four cases demonstrate the method's flexibility — each requires a *different* weight vector $\{\alpha_{i,j}\}$.

**Sec. 3.1: LASCO-C1 Fe XIV green-line loops (1998-05-21 14:33 UT).** A 2-level decomposition with 8 first-level scales and 4 sub-scales each. Three reconstruction strategies are demonstrated (Fig. 2):
- Upper-right: $\alpha_{0,j}=1$ for $j=0\ldots4$ (all continuum) and $\alpha_{i,j}=4$ for $i=1\ldots4$, $j=1\ldots4$ (boost first 4 high-frequency scales by 4×) — gives sharper boundaries.
- Bottom-left: like classical unsharp masking — all scales weight 4, continuum weight 1.
- Bottom-right: only first-level scales 2-4 are kept with weights $\alpha_{2,1}=10$, $\alpha_{2,2}=7$, $\alpha_{3,2}=7$ — aggressive high-frequency emphasis, structures appear noticeably sharper.

**Sec. 3.2: CME on 2002-08-13 in LASCO-C2 field of view.** 8 first-level scales with 3 sub-scales. The first reconstruction (Fig. 3, second column) is essentially unsharp masking; the second (third column) zeroes the continua $\alpha_{0,j}=0$ for all $j$ and emphasises high-frequency, revealing internal twisted filaments inside the CME and a previously invisible streamer pushed northwards by the bow shock.

**Sec. 3.3: Prominence on 1998-06-02 in LASCO-C2.** 50 first-level scales (!) with 3 sub-scales each, reconstructed using only the first 25 first-level scales and excluding the continuum. Despite the original level-0.5 image showing only a faint blob, the processed image (Fig. 4 right) reveals the prominence's clear filamentary structure and even diffraction rings around the occulter.

**Sec. 3.4: CME on 2002-07-04 in LASCO-C3.** Despite C3's lower resolution, internal CME structure is recovered using a "stress all wavelet scales relative to the continuum" strategy with weights mostly 3 and a few selected weights of 6.

In each case the authors emphasise that the *choice* of $\alpha_{i,j}$ is interactive and motivated by the science question — the technique is not a black box.

네 응용 사례 모두 *다른* 가중치 벡터 $\{\alpha_{i,j}\}$를 사용한다. (i) Fe XIV 내부 코로나 loop는 8 first-level × 4 sub-scale 분해 후 세 가지 재합성. (ii) 2002-08-13 CME는 continuum을 0으로, 고주파를 강조 → 내부 twisted filament 노출. (iii) 1998-06-02 prominence는 50 first-level × 3 sub-scale의 극단 분해, 처음 25개만 사용 → filament 구조와 diffraction ring까지 보임. (iv) C3 CME는 모든 스케일을 상대적으로 강조하는 전략. 핵심은 *과학 질문에 따라 사용자가 가중치를 결정*한다는 점이다.

### Part VII: Comparison with Standard Image Enhancement Tools / 표준 영상 향상 도구와의 비교

A useful framing is to compare the Stenborg pipeline with three other classes of enhancement:

| Tool | Principle | Strength | Weakness vs Stenborg |
|---|---|---|---|
| Unsharp masking | Subtract Gaussian blur | Simple, fast | Single scale, amplifies noise, poor on diffuse features |
| Histogram equalisation | Stretch intensity histogram | Good display contrast | Spatially-blind, can introduce artefacts |
| à trous shrinkage (Starck 1997) | Wavelet hard-threshold then sum | Multi-scale, preserves resolution | Single decomposition level (no packets), no equalization control |
| Wavelet packets (this paper) | Two-level decomposition + per-pixel local noise + interactive weights | Multi-scale, multi-sub-scale, user-controlled, statistically grounded | Computationally heavier, requires parameter choices |

The single-step unsharp mask is recoverable as a *special case* of the Stenborg pipeline (set all $\alpha_{i,j}=4$ except continuum=1 — exactly the bottom-left case in Fig. 2). This is a strong argument for the framework: it *generalises* simpler methods rather than competing with them.

표준 향상 도구와의 비교: ① 단순 unsharp masking은 본 논문의 한 특수 경우(continuum=1, 나머지=4)로 회수된다. ② histogram equalisation은 공간 정보를 무시한다. ③ Starck의 단일 수준 wavelet shrinkage는 packet의 더 세밀한 분해를 결여한다. Stenborg 프레임워크는 이들을 *일반화*하므로 경쟁이 아니라 포섭이다.

### Part VIII: Computational Considerations and Implementation Notes / 계산상 고려사항과 구현 노트

A faithful implementation involves several decisions:

- **Edge handling.** The à trous filter is convolution with a 5×5 mask at increasing dilation. At iteration $i$, the effective spacing is $2^{i-1}$, so the filter footprint is $5\cdot 2^{i-1}$ pixels — at $i=8$ this is 640 pixels. Mirror or zero-pad boundaries appropriately; LASCO's circular field of view also needs occulter masking.
- **Memory.** A redundant transform with 8 first-level scales × 4 sub-scales × float32 stores 32 image-sized arrays. For a 1024² image that is 128 MB — manageable but not free. Out-of-core processing or scale-by-scale streaming may be needed for SDO/AIA-class data (4096²).
- **Noise simulation.** $\hat{\sigma}_j$ depends only on the wavelet kernel and the decomposition tree, not on the image — so it can be pre-computed and tabulated once.
- **$\alpha_{i,j}$ tuning.** The interactive nature is a feature, not a bug, but for batch processing of long time series one usually fixes a reconstruction strategy. Inspecting the wavelet planes individually (with imshow on each $W^{(0,i)}_j$) is the only reliable way to choose weights.

구현 노트: ① edge 처리 — à trous는 확장 간격 $2^{i-1}$로 5×5 합성곱이므로 $i=8$이면 footprint 640 pixel. mirror 또는 zero pad, LASCO의 원형 시야는 occulter 마스킹 필요. ② 메모리 — 1024² 영상에서 8×4=32개 평면 × float32 ≈ 128 MB. SDO/AIA 4096²급은 streaming 필요. ③ 노이즈 시뮬레이션은 영상 무관하므로 pre-compute 가능. ④ $\alpha_{i,j}$ 튜닝은 wavelet plane 시각화로 결정한다.

### Part VII: Summary and Conclusions / 요약 및 결론 (Sec. 4, p. 1192)

The authors enumerate the algorithm in 5 steps:
1. Multi-level wavelet decomposition via à trous.
2. Local noise variance estimation per scale.
3. Threshold determination via statistical confidence levels.
4. Noise reduction by local hard-thresholding.
5. Interactive weighted recomposition.

They emphasise the *adaptability* — the same machinery handles widely different structures (loops, CMEs, prominences, streamers) with appropriate weight choices. Coexisting structures of widely different intensity can be treated on equal footing because of the local-statistics machinery. The technique's complement to a tracking algorithm is announced as forthcoming (Stenborg et al. 2002, in prep.).

저자들은 알고리즘을 5단계로 정리한다: à trous 다단계 분해 → 국소 노이즈 분산 추정 → 통계적 임계 결정 → hard-threshold 잡음제거 → 가중 재합성. 같은 도구가 loop, CME, prominence, streamer 모두에 가중치만 바꿔 적용된다는 점이 강조된다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Wavelet packets give finer frequency control than plain wavelets.** Re-decomposing each detail plane lets you keep signal at intermediate frequencies that plain wavelet thresholding would discard along with noise. — 플레인을 한 번 더 쪼개면 단순 wavelet thresholding이 노이즈와 함께 버리던 중간 주파수 신호를 살릴 수 있다.

2. **The à trous algorithm is the right discrete wavelet for astronomy.** Redundancy + isotropy + same-size planes mean structures are spatially registered and persist across scales — exactly what diffuse, rotationally-symmetric coronal images need. — à trous는 redundant·isotropic·same-size이므로 코로나처럼 등방적이고 다중 스케일이 중첩된 영상에 최적이다.

3. **Estimate noise from the data, not from a model.** The $\hat{\sigma}_j \cdot \sigma_1(k,l)$ factorisation lets per-scale and per-pixel noise be measured *from the image itself*, handling the wide dynamic range that breaks any single global $\sigma$. — 데이터에서 노이즈를 직접 추정해야 한다. per-scale × per-pixel 분해가 광범위한 동적 범위 문제를 해결한다.

4. **Equalisation, not just enhancement.** The user-chosen $\alpha_{i,j}$ weights perform frequency *equalisation* — coexisting structures with vastly different intensities are placed on equal visual footing, like a graphic-equaliser for an image. — *향상*이 아니라 *평활화(equalisation)*가 본질이다. 서로 다른 밝기의 구조를 시각적으로 동등하게 만든다.

5. **Anscombe is the missing link for Poisson data.** The transform $t[I]=2\sqrt{I+3/8}$ maps Poisson statistics to (approximately) Gaussian-with-$\sigma=1$, letting all subsequent Gaussian-based tools apply. — Anscombe 변환은 photon-counting CCD 데이터를 Gaussian-friendly로 만들어 모든 후속 Gaussian 도구를 사용할 수 있게 해준다.

6. **The 3$\sigma$ hard-threshold ($k=3$) is a principled default.** It corresponds to 99.7% confidence and rarely needs tuning if the local-stdev model is accurate. — $k=3$는 99.7% 신뢰의 통계적 디폴트이며 국소 stdev 모델이 정확하면 거의 손볼 필요가 없다.

7. **Different scientific questions require different weight vectors.** The paper's four case studies all use radically different $\{\alpha_{i,j}\}$ — there is no universal "best" weight set. — 과학적 질문마다 가중치 벡터를 다르게 정해야 한다. 보편적 최적값은 없다.

---

## 4. Mathematical Summary / 수학적 요약

**Continuous wavelet transform (definition):**

$$
W_c f(a,b) = \frac{1}{\sqrt{a}}\int f(x)\,\psi^*\!\left(\frac{x-b}{a}\right)dx,\qquad a>0.
$$

**à trous recursion** with $B_3$-spline filter $h$ ($5\times5$ mask above):

$$
c_i(k,l) = \sum_{m,n} h(m,n)\,c_{i-1}(k+2^{i-1}m,\,l+2^{i-1}n).
$$

**Detail plane (wavelet plane):**

$$
w_i(k,l) = c_{i-1}(k,l) - c_i(k,l).
$$

**Reconstruction identity** ($p$-level decomposition):

$$
c_0(k,l) = c_p(k,l) + \sum_{i=1}^{p} w_i(k,l).
$$

**Wavelet-packet two-level expansion:**

$$
c_0 = \sum_{i=0}^{p_0}\sum_{j=0}^{p_{0,i}} w^{(0,i)}_j.
$$

**Anscombe transform:**

$$
t[I(x,y)] = 2\sqrt{I(x,y)+\tfrac{3}{8}}.
$$

**Local stdev factorisation:**

$$
\sigma^{(\cdot)}_m(k,l) = \hat{\sigma}^{(\cdot)}_m \cdot \sigma_1(k,l)
$$

with $\hat{\sigma}^{(\cdot)}_m$ from a unit-variance Gaussian simulation passed through the same wavelet transform, and $\sigma_1(k,l)$ from an $N\times N$ neighbourhood of the actual image's first-scale plane.

**Hard-threshold:**

$$
W^{(\cdot)}_m(k,l) = w^{(\cdot)}_m(k,l)\cdot \mathbf{1}\!\left[|w^{(\cdot)}_m(k,l)| \ge k\,\sigma^{(\cdot)}_m(k,l)\right],\quad k=3.
$$

**Weighted recomposition:**

$$
\mathcal{I}(k,l) = \sum_{i=0}^{p_0}\sum_{j=0}^{p_{0,i}} \alpha_{i,j}\,W^{(0,i)}_j(k,l).
$$

**Worked example (1D toy).** Take a 1D signal $f = c_0$ with three superposed structures: a smooth ramp $r(x)$ (low-freq), a 4-pixel loop bump $\ell(x)$ (mid-freq), and white noise $n(x)$. After 3-level à trous: $c_3$ ≈ ramp, $w_1$ ≈ noise + loop edges, $w_2$ ≈ loop body, $w_3$ ≈ broad envelope. Hard-threshold at $3\sigma$ kills $w_1$'s noise (the loop edge survives because edge amplitude > $3\sigma$). Reweight $\alpha_3=0$ (drop ramp), $\alpha_1=1$, $\alpha_2=4$ (boost loop body). Result: a clean loop image stripped of the ramp — a 1D analogue of what the paper does to LASCO loops. 

**Smearing displacement (Sec. 1):**

$$
d = \frac{v\,\tau_e}{r_s\cdot 700\;\text{km}}\;\text{pixels}.
$$

Plug in $v=800$ km/s, $\tau_e=20$ s, $r_s=5.6$ arcsec/pix → $d \approx 4$ pixels.

각 변수의 의미: $h$는 $B_3$-spline 저주파 필터(5×5), $c_i$는 $i$-단계 평활화 영상, $w_i$는 detail, $\hat{\sigma}_m$은 unit-variance noise 시뮬레이션의 per-scale 응답, $\sigma_1(k,l)$은 영상에서 직접 추정한 first-scale 국소 표준편차, $\alpha_{i,j}$는 사용자 지정 가중치. Worked example는 위와 같이 1D 시뮬레이션으로 검증할 수 있다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1948  Anscombe — variance stabilisation                    
                     │                                      
                     ▼                                      
1984  Grossmann & Morlet — continuous wavelets             
                     │                                      
                     ▼                                      
1989  Mallat — multiresolution / pyramidal DWT              
                     │                                      
                     ▼                                      
1990  Holschneider — à trous algorithm                      
1991  Wickerhauser — wavelet packets                        
1994  Donoho & Johnstone — wavelet shrinkage / hard-threshold
                     │                                      
                     ▼                                      
1995  SOHO/LASCO operations begin (data flood)             
1997  Starck, Murtagh — astronomical à trous                
                     │                                      
                     ▼                                      
2003  *Stenborg & Cobelli (this paper) — combine all of the above for coronal imaging*
                     │                                      
                     ▼                                      
2006  Morgan, Habbal & Woo — NRGF (complementary radial filter)
2014  Morgan & Druckmüller — MGN (descendant of this work) 
2020+ MGN/NRGF combined preprocessing for STEREO, PSP/WISPR, Solar Orbiter/Metis
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #02 Donoho 1995 (wavelet shrinkage) | Source of the hard-threshold idea adopted in Eq. 15 | Direct mathematical ancestor — provides the per-scale thresholding framework |
| #11 Anscombe 1948 | Variance-stabilising transform used in Eq. 14 | Indispensable preprocessing for Poisson-dominated CCD data |
| #33 Starck 2003 (astronomical multiscale) | Companion methodology applied to the same broad problem | Sister paper — both build on à trous for astronomy |
| #34 Yashiro 2004 (CME catalog) | Relies on coronagraph image processing for CME detection | The kind of downstream science that requires Stenborg-quality images |
| #35 Morgan 2006 (NRGF) | Different filter (radial standardisation) for the same data | Complementary tool — NRGF removes radial gradient, Stenborg recovers fine structure |
| #36 Starck 2007 (MGA tutorial) | Modern review covering wavelet packets, curvelets, ridgelets | Places this paper inside the broader multiscale geometric analysis landscape |
| #38 Morgan 2014 (MGN) | Direct descendant — generalises the per-scale equalisation idea | MGN is the "modern industrial" version of Stenborg's pipeline |

---

## 7. References / 참고문헌

- G. Stenborg & P. J. Cobelli, "A wavelet packets equalization technique to reveal the multiple spatial-scale nature of coronal structures," *Astronomy & Astrophysics* 398, 1185-1193 (2003). DOI: 10.1051/0004-6361:20021687
- F. J. Anscombe, "The transformation of Poisson, binomial and negative-binomial data," *Biometrika* 15, 246 (1948).
- Brueckner, G. E., et al., "The Large Angle Spectroscopic Coronagraph (LASCO)," *Solar Physics* 162, 357 (1995).
- Donoho, D. L. & Johnstone, I. M., "Adapting to Unknown Smoothness via Wavelet Shrinkage," Stanford Tech. Report (1994).
- Holschneider, M. & Tchamitchian, P., "Les ondelettes," in: P. G. Lemarié (ed.), Springer-Verlag (1990).
- Wickerhauser, M. V., "INRIA Lectures on Wavelet Packets Algorithms" (Yale University, New Haven, 1991).
- Murtagh, F., Starck, J.-L., & Bijaoui, A., *A&AS* 112, 179 (1995).
- Shensa, M. J., "The discrete wavelet transform: wedding the à trous and Mallat algorithms," *IEEE Trans. Signal Process.* 40, 2464 (1992).
- Starck, J.-L., Siebenmorgen, R., & Gredel, R., *ApJ* 482, 1011 (1997).

### Appendix A: Worked 1D toy with code-style pseudo-instructions / 부록 A: 1D 토이 의사 코드

```
# 1) build signal: ramp + loop bump + noise
x = linspace(0, 1, 256)
ramp  = 1 - x                                    # low-frequency
loop  = exp(-((x-0.5)/0.04)**2) * 0.4            # mid-frequency (4-pixel bump-equiv)
noise = gauss(0, 0.05, 256)                     # high-frequency
f0    = ramp + loop + noise

# 2) Anscombe (skip for Gaussian); not needed here

# 3) à trous decomposition with 1D B3-spline (1,4,6,4,1)/16
c0 = f0
for i in 1..p:
    c[i] = atrous_filter(c[i-1], step=2**(i-1))
    w[i] = c[i-1] - c[i]
# now: c0 = c[p] + sum_i w[i]   (exact identity)

# 4) Local hard-threshold with k=3
for i in 1..p:
    sigma_i = std_local(w[i], window=5) * sigma_hat[i]
    W[i] = where(abs(w[i]) >= 3*sigma_i, w[i], 0)

# 5) Weighted recomposition
alpha[continuum] = 0       # drop ramp
alpha[1] = 1               # keep first wavelet plane
alpha[2] = 4               # boost loop body
alpha[3] = 0
result = alpha[continuum]*c[p] + sum_i alpha[i]*W[i]
```

The result is the loop bump with ramp removed and most noise suppressed — the 1D analogue of what the paper does to LASCO loops. The same structure (à trous → local-noise threshold → weighted sum) is exactly what is implemented in the accompanying Jupyter notebook on a 2D synthetic coronagraph.

위 의사 코드는 본 논문의 1D 단순화이다. 동일한 구조(à trous → 국소 임계 → 가중 합)가 동반 Jupyter 노트북에 2D 합성 코로나그래프 영상으로 구현되어 있다.

### Appendix B: Common Pitfalls / 부록 B: 흔한 함정

- **Forgetting Anscombe.** If applied to Poisson data without the variance-stabilising transform, the noise model is wrong and the threshold $k\sigma$ over-suppresses bright regions (where Poisson $\sigma\propto\sqrt{I}$).
- **Single global $\sigma$.** Using one $\sigma$ for the whole image breaks down on coronagraph data because the dynamic range is $\sim 10^4$. The local $\sigma_1(k,l)$ is essential.
- **Choosing $\alpha_{i,j}$ blindly.** Without inspecting the wavelet planes, one cannot know which scales contain the structure of interest. Always visualise $W^{(0,i)}_j$ before reweighting.
- **Confusing wavelet packets with discrete wavelet packets.** Stenborg uses the redundant à trous variant; classical wavelet packets in signal processing use down-sampled, orthogonal versions. The naming is the same but the math differs.

흔한 함정: ① Poisson 데이터에 Anscombe 누락 — 임계가 밝은 영역을 과도하게 억제. ② 영상 전체에 단일 $\sigma$ 사용 — 동적 범위 $10^4$를 감당 못함. ③ wavelet plane을 안 보고 $\alpha_{i,j}$ 결정 — 어느 스케일에 신호가 있는지 모르면 가중치 의미 없음. ④ "wavelet packets" 용어 혼동 — Stenborg는 redundant à trous 버전, 신호처리 교과서의 down-sampled orthogonal 버전과 다름.
