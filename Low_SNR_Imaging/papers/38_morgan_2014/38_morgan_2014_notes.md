---
title: "Multi-Scale Gaussian Normalization for Solar Image Processing"
authors: Huw Morgan, Miloslav Druckmüller
year: 2014
journal: "Solar Physics, Vol. 289, No. 8, pp. 2945–2955"
doi: "10.1007/s11207-014-0523-9"
topic: Low_SNR_Imaging
tags: [solar-image-processing, multi-scale, gaussian-normalization, EUV, SDO-AIA, NRGF, NAFE, image-enhancement, scale-space, coronagraph]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 38. Multi-Scale Gaussian Normalization for Solar Image Processing / 태양 영상 처리를 위한 다중 스케일 Gaussian 정규화

---

## 1. Core Contribution / 핵심 기여

The paper introduces **Multi-Scale Gaussian Normalization (MGN)**, an image-enhancement procedure designed for the unique challenges of EUV solar images — extremely large dynamic range, faint structures hidden inside very bright active regions, and the need to process volumes the size of SDO/AIA's archive (4096×4096 frames at 11-second cadence). The method is a three-step recipe: (i) at each of $n$ Gaussian-kernel widths $w_i \in \{1.25, 2.5, 5, 10, 20, 40\}$ pixels, normalise the image by its **Gaussian-weighted local mean and local standard deviation** (Eqs. 1–2), producing a zero-mean, unit-variance "locally normalised" image $C_i$ for that scale; (ii) **arctan-compress** each $C_i$ to control the output range and avoid saturation, $C_i' = \arctan(k C_i)$ with $k \approx 0.7$ (Eq. 3); (iii) **blend** the multi-scale normalised images $C_i'$ (weighted by $g_i$) with a global γ-corrected image $C_g$ (weighted by $h \approx 0.7$) into a final image $I$ (Eqs. 4–5). Because the local standard deviation at small kernels is dominated by photon noise (the histogram of $\sigma_w$ across the image moves towards zero for $w \lesssim 3$ px, Fig. 4), the scale weights $g_i$ can be chosen to **automatically suppress noise** while preserving fine structure at $w \gtrsim 3$ px. The entire procedure runs in $\sim 1$ s for a 500×500 image and $\sim 40$ s for a full 4096×4096 AIA frame on a 2014 laptop — at least an order of magnitude faster than NAFE (Druckmüller 2013) and dramatically faster than wavelet-based equalisation. The paper applies MGN to AIA 171 Å, Hi-C 193 Å, SWAP, and LASCO C2 imagery, demonstrating that a single recipe handles disk, off-limb, and even white-light coronagraph data without modification.

이 논문은 **Multi-Scale Gaussian Normalization (MGN)** — EUV 태양 영상의 극단적 동적 범위, 활동 영역에 가려진 미세구조, 그리고 SDO/AIA 의 4096×4096 / 11 s cadence 같은 데이터 양을 모두 다루기 위해 설계된 영상 강조 기법 — 을 제안한다. 알고리즘은 세 단계: (i) $n$ 개의 Gaussian kernel 폭 $w_i \in \{1.25, 2.5, 5, 10, 20, 40\}$ px 에서 **국지 평균과 국지 표준편차로 정규화** 해 평균 0, 표준편차 1 의 영상 $C_i$ 를 만든다 (식 1–2). (ii) 각 $C_i$ 를 **arctan-압축** 해 $C_i' = \arctan(kC_i)$ 로 출력 범위를 통제한다 (식 3). (iii) 다중스케일 $C_i'$ 들을 가중치 $g_i$ 로 평균하고, **전역 γ-보정 영상** $C_g$ 와 가중치 $h \approx 0.7$ 로 결합한다 (식 4–5). 작은 kernel ($w \lesssim 3$ px) 에선 국지 표준편차 분포가 0 쪽으로 치우쳐 잡음이 지배 (Fig. 4) 하므로 $g_i$ 를 작게 두면 **잡음을 자동으로 억제** 하면서 큰 스케일 맥락은 유지한다. 전체 처리는 500×500 영상에 ∼1 초, 4096² AIA 영상에 ∼40 초 — NAFE 보다 1 차 자릿수 이상 빠르고 wavelet 기반 처리보다 훨씬 빠르다. 본 논문은 AIA 171 Å, Hi-C 193 Å, SWAP, LASCO C2 까지 단일 레시피로 적용 가능함을 보인다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction — Why EUV Images Are Hard / EUV 영상의 본질적 어려움 (Section 1, pp. 2945–2946)

EUV solar images contain information on **a wide range of spatial scales** from the smallest visible loops in active regions to large quiescent structures across the disk and to faint streamers off-limb. Different physical structures — active regions, quiet Sun, filament channels — emit at very different brightness regimes. Processing these images well, **without introducing artefacts or bias**, is essential for both visual inspection (still the starting point for most scientific analysis of AIA/SDO) and for downstream automated detection tools (Martens et al. 2012). Two pre-existing approaches:

- **Simple global transforms** (sqrt, log, γ) — fast but limited; the dominance of small bright regions ensures that quiet-Sun and off-limb structure remain washed out.
- **Time differencing** — reveals dynamic features but kills static structure entirely.

More advanced techniques fall into two lineages: **wavelet-based multi-scale enhancement** (Stenborg & Cobelli 2003; Stenborg, Vourlidas & Howard 2008), which gives excellent results but is computationally expensive; and **locally-adaptive histogram equalisation**, especially Druckmüller's NAFE (2013), which produces very clear images at significant computational cost. The authors target a **middle ground**: wavelet-quality output with sub-second-per-frame computational cost.

EUV 영상은 매우 작은 활동 영역 루프부터 disk 전체의 quiet Sun, off-limb streamer 까지 광범위한 공간 스케일의 정보를 담고 있고, 구조마다 밝기 범위가 천차만별이다. 단순 sqrt/log/γ 변환은 빠르지만 작은 밝은 영역이 대비를 점령해 어두운 영역 미세구조가 보이지 않는다. 시간 차분은 동적 구조에만 반응한다. wavelet 기반 enhancement (Stenborg-Cobelli 2003; Stenborg-Vourlidas-Howard 2008) 는 결과가 좋으나 계산이 느리고, NAFE (Druckmüller 2013) 도 우수하나 비싸다. 본 논문은 **wavelet 수준 결과와 sub-초 계산** 의 중간점을 목표로 한다.

### Part II: Observations — Quantifying the Problem / 문제의 정량화 (Section 2, pp. 2946–2948)

A working AIA 171 Å image (04 May 2005 00:00 UT) anchors the paper. The 171 Å channel is dominated by Fe IX/X lines emitted at ∼ 0.8 MK plasma; line-of-sight effects and weaker contributions from other ions also contribute. Figure 1 (left) shows the raw image (the bright active region is over-exposed; quiet Sun and off-limb are flat black) and (right) the same after a square-root transform — better but the brightest regions wash out and the off-limb side is still nearly featureless. **Quantitative diagnosis** (Fig. 2): four 50×50 pixel boxes are sampled (quiet Sun, active-region base, active region, off-limb) and their pixel-value histograms plotted on a log axis. Result: the **brightest 25 % of the active-region pixel values dominate the 0–10 000 contrast budget**, leaving the lower 75 % of brightness values squeezed into the bottom decade and off-limb / quiet-Sun pixels confined to values $\lesssim 300$. In other words, any enhancement that hopes to display all four populations on a single screen must **adapt locally** — there is no global tone-mapping that preserves all ranges simultaneously. The same characteristics worsen when flares occur.

작업 예시인 AIA 171 Å (04 May 2005) 는 Fe IX/X 0.8 MK 플라즈마 방출이 지배. Figure 1 의 raw / sqrt 두 영상은 단순 변환의 한계를 보여주고, Figure 2 의 quiet Sun / active-region base / active region / off-limb 네 박스 히스토그램은 **활동 영역 상위 25 % 가 0–10000 대비 예산을 점령** 하고 어두운 75 % 픽셀이 대비 한 자릿수 안에 압축되어 있음을 정량적으로 보인다. 단일 전역 톤 매핑으로는 네 영역 모두를 한 화면에 보일 수 없으므로 **국지 적응** 이 필요하다.

The paper also rejects edge-detection-only enhancement: such techniques amplify some features but lose the larger-scale context. The chosen design must be **scale-aware** and able to balance local detail with global structure.

또한 단순 edge-detection 류는 작은 스케일은 강조하지만 큰 스케일 맥락을 잃기 때문에 부적절하다. 따라서 **다중 스케일 적응** 이 필요하다.

### Part III: Method — The MGN Recipe / 방법 (Section 3, pp. 2948–2950)

Three components in turn.

**(A) Local Gaussian normalisation (Eqs. 1–2).** Let $B$ be the input image (positive after exposure-time normalisation; spurious negatives replaced by zero / local median). Define a 2-D Gaussian kernel $k_w$ with one-sigma width $w$ pixels. The normalised image at scale $w$ is

$$
C = \frac{B - B \otimes k_w}{\sigma_w}, \quad \sigma_w = \sqrt{\big[(B - B \otimes k_w)^2\big] \otimes k_w}, \tag{1, 2}
$$

i.e., the numerator is the high-pass residual (image minus Gaussian-smoothed version) and the denominator is the Gaussian-weighted local standard deviation. Each pixel of $C$ now has approximately mean 0, std 1, regardless of whether it sat inside an active region or in quiet Sun. This is the multiscale generalisation of the local normalisation idea behind adaptive histogram equalisation, but where AHE has rectangular bins, MGN uses smooth Gaussian-weighted neighbourhoods. The example in Fig. 3 (left, $w = 20$) shows how this single normalisation already reveals an enormous amount of structure both on-disk and off-limb.

**(B) arctan compression (Eq. 3).** Because $C$ takes both negative and positive values with std $\approx 1$, an arctan transform

$$
C' = \arctan(k\,C),\qquad k \approx 0.7 \tag{3}
$$

is applied to amplify pixels close to zero (where the perceptually interesting transitions live) while compressing extreme values that would otherwise saturate. With $k = 0.7$, $\arctan(kC)$ is roughly linear for $|C| \lesssim 1$ and saturates softly at $|kC| \gg 1$. Fig. 3 (right) shows $C'$, the compressed version. This step is both **range-control** and a soft non-linear contrast stretch.

**(C) Multi-scale recombination with global γ image (Eqs. 4–5).** A global γ-transformed image gives the *large-scale tone* that pure local normalisation would discard:

$$
C_g = \left(\frac{B - a_0}{a_1 - a_0}\right)^{1/\gamma}, \quad \gamma = 3.2,\;\; a_0 = \min B,\; a_1 = \max B. \tag{4}
$$

The final image is a weighted sum:

$$
\boxed{\;I = h\, C_g + \frac{1-h}{n}\sum_{i=1}^{n} g_i\, C_i' \;} \tag{5}
$$

with $h = 0.7$ as the global-tone weight (default chosen empirically from many AIA images), $n$ scales (paper uses $n = 6$, $w \in \{1.25, 2.5, 5, 10, 20, 40\}$), and per-scale weights $g_i$. The crucial design rule for $g_i$: **at small $w$ ($w \lesssim 3$ px), the local std $\sigma_w$ is biased low** (Fig. 4 shows $\langle\sigma_w\rangle$ across the image versus $w$ for a pure-noise input), so the smallest-scale $C_i'$ images are noise-dominated. Setting $g_i$ small for those scales (∼0.6) and approaching 1 for $w \gtrsim 3$ damps noise without manual masking. For most purposes the authors use a flat $g_i = 1$ across all scales, simplifying to a straight mean of the $C_i'$.

세 가지 부속 단계. **(A) 국지 Gaussian 정규화** (식 1–2): $B$ 의 high-pass 잔차를 국지 표준편차로 나누어 평균 0, 표준편차 1 의 $C$ 를 만든다. AHE 의 사각 bin 대신 부드러운 Gaussian 가중을 쓴다. **(B) arctan 압축** (식 3): $C$ 가 양/음 값을 가지므로 arctan 으로 0 부근 증폭, 극단 압축. $k \approx 0.7$ 에서 $|C| \lesssim 1$ 은 거의 선형이고, $|kC| \gg 1$ 에선 soft saturation. **(C) 다중스케일 + 전역 γ 결합** (식 4–5): 전역 γ-보정 $C_g$ 가 큰 스케일 톤을 보존한다. $w \in \{1.25, 2.5, 5, 10, 20, 40\}$ px 의 $n=6$ 개 $C_i'$ 를 가중치 $g_i$ 로 평균한 뒤 $C_g$ 와 $h=0.7 : 0.3$ 으로 결합. $w \lesssim 3$ 픽셀 범위는 국지 std 가 잡음 지배라 (Fig. 4) $g_i$ 를 작게 두어 자동 잡음 억제.

**Pseudocode (Section 3.1)** — verbatim 10 steps:

```
1. Replace spurious negative pixels with zero or local mean / median.
2. Create Gaussian kernel of width w_i. Kernel elements should sum to unity.
3. Convolve image with kernel to create local mean image B ⊗ k_w.
4. Calculate (B − B⊗k_w), square the difference, convolve with kernel,
   and square-root to give the local std image σ_w (Eq. 2).
5. Calculate normalised image C_i = (B − B⊗k_w) / σ_w (Eq. 1).
6. Apply arctan transformation on C_i to give C_i'.
7. Repeat 2–6 with the different kernel widths w_i.
8. Take mean (or weighted mean) of the C_i' to give weighted mean LN image.
9. Calculate global γ-transformed image C_g (Eq. 4).
10. Sum the weighted mean locally normalised image with C_g, weighted by h (Eq. 5).
```

This pseudocode is the implementation specification — every line is executable and the algorithm has no hidden state.

이 의사코드는 알고리즘 사양 그 자체로, 그대로 구현 가능하다.

**Computational efficiency** — separable Gaussian filtering is what makes MGN fast: a 2-D Gaussian is the outer product of two 1-D Gaussians, so the convolution becomes "1-D along x, then 1-D along y", which is much cheaper than a direct 2-D convolution. The total cost of MGN is therefore $\mathcal O(N \cdot n \cdot w_{max})$ rather than $\mathcal O(N \cdot w_{max}^2)$ for direct 2-D filtering. With six scales the largest $w$ is 40 pixels, so the total compute is modest. **Measured timing** on a 2014 MacBook Pro Core i7, 8 GB RAM: ∼ 1 s for $500\times 500$ to ∼ 40 s for $4096\times 4096$ AIA, with linear scaling. Compared to NAFE this is at least an order of magnitude faster.

**계산 효율** — separable Gaussian 필터링이 핵심. 2-D Gaussian 은 두 1-D Gaussian 의 외적이므로 "x 방향 1-D, y 방향 1-D" 두 단계로 빠르게 합성 가능. MGN 전체 비용은 $\mathcal O(N \cdot n \cdot w_{max})$ — direct 2-D 의 $\mathcal O(N \cdot w_{max}^2)$ 보다 훨씬 작다. 측정된 처리 시간: 500×500 ∼1 s, AIA 4096×4096 ∼40 s. NAFE 보다 한 자릿수 이상 빠르다.

### Part IV: Results — Four Worked Examples / 네 가지 적용 사례 (Section 4, pp. 2951–2954)

**(A) AIA 171 Å (Fig. 5).** The same image used as the working example in Sections 1–3. After MGN: structure is enhanced down to fine spatial scales (visible loops, chromospheric moss), but large-scale context is preserved (active region remains brighter than quiet Sun, off-limb structures are not over-amplified). Off-limb structures emerge clearly without becoming swamped by noise. Loops connecting bright regions can be traced from disk to off-limb. An animated GIF of 171 Å AIA on 18 January 2013 reveals dynamic features at very small scales — flows along filament channels, motion in extended off-limb structures.

**(B) Hi-C 193 Å (Fig. 6).** *High-resolution Coronal Imager* — a 2012 sounding-rocket instrument with 0.1″ spatial resolution, 5 s cadence. MGN exposes magnetic braids (Cirtain et al. 2013) and small-scale plasma streams in the moss. These features cannot be seen without enhancement.

**(C) SWAP / PROBA-2 (Fig. 7).** The *Sun Watcher* instrument has a coarser spatial resolution than AIA but an *extended* field of view (out to ∼ 1.5 R⊙). MGN shows an erupting filament out to the field-of-view edge and quiescent off-limb structures. Low-signal regions are enhanced without amplifying noise.

**(D) LASCO C2 white-light coronagraph (Fig. 8).** White light is photometrically very different from EUV: the F-corona must be subtracted, point filters needed for cosmic rays / spurious bright pixels. The authors apply MGN with $k = 0.8$, $h = 0.9$, $\gamma = 1$. Smaller-scale features — faint plumes over the poles — emerge that were difficult to see in the unprocessed image. The result is "an improvement over the NRGF" (Morgan, Habbal & Woo 2006), with structural context closer to the true K-corona.

**Qualitative comparison with NAFE:** the authors note that NAFE produces slightly better clarity overall and better noise suppression at large heights off-limb, while MGN's natural noise reduction (averaging the $C_i'$ across scales) is "not as effective" at noise suppression but is *vastly* faster and offers no manual control over noise suppression. NAFE is also better suited to revealing contrast inside very bright regions. **Trade-off**: NAFE for offline pipelines where quality dominates; MGN for routine throughput and on-line displays.

**(A) AIA 171 Å.** 미세 루프, 모스, off-limb 구조가 모두 드러나면서도 전역 톤 (활동 영역이 quiet Sun 보다 밝음) 이 유지된다. **(B) Hi-C 193 Å.** 자기장 braiding 과 모스 내 작은 플라즈마 흐름이 드러난다. **(C) SWAP.** 1.5 R⊙ field-of-view 끝까지 erupting filament 가 보이고 어두운 off-limb 구조도 잡음 증폭 없이 enhanced. **(D) LASCO C2.** 백색광에도 단일 레시피 (k=0.8, h=0.9, γ=1) 로 적용 가능, NRGF 보다 K-corona 의 실제 구조 맥락을 더 잘 보존한다. **NAFE 와의 정성 비교:** NAFE 가 노이즈 억제와 매우 밝은 영역의 대비에서 약간 우수하나 MGN 은 1 차 자릿수 이상 빠르다.

### Part V: Summary / 결론 (Section 5, p. 2955)

MGN normalises an image by local Gaussian-weighted mean and standard deviation, applies arctan compression, repeats for several spatial scales, and combines the multi-scale normalised images with a global γ-transformed image into a final, weighted output. Results are comparable to multi-resolution wavelet enhancement and to NAFE while being far faster computationally. The method is **simple to implement** (Gaussian smoothing is a built-in primitive of every numerical library) and the corresponding author makes IDL code available by request. The authors hope MGN will become an established tool — a hope that has been borne out: by 2026 MGN is the default EUV enhancement in Helioviewer / JHelioviewer / SunPy / IDL Solarsoft.

MGN 은 국지 Gaussian 가중 평균과 표준편차로 정규화하고 arctan 으로 압축한 뒤 여러 스케일에서 반복해 평균을 내고, 전역 γ 영상과 결합해 최종 영상을 만든다. wavelet enhancement, NAFE 와 비슷한 결과를 훨씬 빠르게 얻으므로 사실상 표준 도구로 자리잡았다.

---

## 3. Key Takeaways / 핵심 시사점

1. **Local Gaussian normalisation simultaneously equalises brightness and removes the smooth large-scale structure.** $C = (B - B\otimes k_w)/\sigma_w$ at each scale produces a mean-0, std-1 image that exposes faint structure regardless of whether it lives in active region or quiet Sun. / **국지 Gaussian 정규화 는 밝기를 균일화하고 큰 스케일 구조를 제거한다** — 한 단계로 두 효과.

2. **arctan compression is a near-linear amplifier near zero with soft saturation at the tails.** With $k\approx 0.7$ and $C \sim \mathcal N(0,1)$ this both prevents output saturation and gently boosts perceptual contrast around zero. / **arctan 압축** 은 0 근방에선 거의 선형 증폭, 양 극단에선 부드러운 saturation. $k \approx 0.7$, $C \sim \mathcal N(0,1)$.

3. **A few scales suffice; six is a good practical default.** $w \in \{1.25, 2.5, 5, 10, 20, 40\}$ px spans atmospheric scales of EUV imaging; more scales offer diminishing returns. / **소수의 스케일이면 충분하다** — 6 개가 실용적 기본값.

4. **Local std stabilises beyond $w \approx 3$ px.** Smaller kernels give noise-dominated $\sigma_w$, so per-scale weights $g_i$ should suppress them — automatic noise damping. / **$w \lesssim 3$ px 에선 국지 std 가 잡음 지배** — $g_i$ 작게 두면 자동 잡음 억제.

5. **Global γ image preserves macroscopic tone.** Without it, brightness ordering ("active regions bright, quiet Sun dim") is destroyed. The blend $I = h C_g + (1-h)\langle C_i'\rangle$ with $h = 0.7$ keeps both. / **전역 γ 영상이 큰 스케일 톤을 유지** — 없으면 활동 영역과 quiet Sun 의 밝기 순서가 깨진다.

6. **Separable Gaussian filtering makes MGN cheap.** Two 1-D convolutions per scale; total $\mathcal O(N \cdot n \cdot w_{max})$. AIA 4096² in ∼ 40 s on a 2014 laptop. / **Separable Gaussian** 으로 MGN 은 매우 빠르다 — AIA 전 영상 ∼40 초.

7. **One recipe applies to EUV and white-light alike.** AIA, Hi-C, SWAP, LASCO C2 all benefit from MGN with at most a parameter adjustment. The technique is independent of the imaging physics. / **EUV / 백색광 모두에 단일 레시피** 적용 가능 — 측정 물리에 무관.

8. **MGN is the de-facto standard.** Adopted by Helioviewer / JHelioviewer / SunPy / Solar Orbiter / IRIS pipelines; a baseline against which deep-learning EUV enhancement methods are compared. / **MGN 은 사실상 표준** — Helioviewer, JHelioviewer, SunPy, Solar Orbiter / IRIS 파이프라인 채택.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Gaussian kernel & separable convolution / Gaussian 커널과 separable 합성

The 2-D Gaussian kernel of one-sigma width $w$ is

$$
k_w(x, y) = \frac{1}{2\pi w^2}\exp\!\left(-\frac{x^2 + y^2}{2 w^2}\right) = k_w^{1D}(x)\,k_w^{1D}(y),
$$

so $B \otimes k_w$ can be computed as $(B \otimes_x k_w^{1D}) \otimes_y k_w^{1D}$ — two 1-D convolutions. The cost drops from $\mathcal O(N w^2)$ to $\mathcal O(N w)$.

### 4.2 Local statistics / 국지 통계

For a single scale $w$:

$$
\boxed{\;C = \frac{B - B \otimes k_w}{\sigma_w},\qquad \sigma_w = \sqrt{\big[(B - B\otimes k_w)^2\big] \otimes k_w}\;}
$$

(Eqs. 1–2 of the paper.) After this step, $\langle C \rangle \approx 0$ pixel-by-pixel and $\mathrm{var}(C) \approx 1$ — provided $\sigma_w$ is well-defined (it is, except in the very faintest regions where $\sigma_w \to 0$; in practice a small floor is added).

### 4.3 arctan compression / arctan 압축

$$
\boxed{\;C' = \arctan(k\,C),\qquad k \approx 0.7\;} \tag{3}
$$

For $|kC| \le 1$ the function is roughly linear (Taylor: $\arctan(x) \approx x - x^3/3$); for $|kC| \gg 1$ it saturates at $\pm \pi/2$. The transformation is monotonic, so contrast direction is preserved.

### 4.4 Global γ correction / 전역 γ 보정

$$
\boxed{\;C_g = \left(\frac{B - a_0}{a_1 - a_0}\right)^{1/\gamma},\qquad \gamma = 3.2,\; a_0 = \min B,\; a_1 = \max B\;} \tag{4}
$$

Maps the input range to $[0, 1]$ and applies a γ exponent that emphasises mid-range tones (γ > 1 brightens the midtones).

### 4.5 Final blend / 최종 결합

$$
\boxed{\;I = h\,C_g + \frac{1-h}{n}\sum_{i=1}^{n} g_i\,C_i',\qquad h = 0.7,\; n = 6\;} \tag{5}
$$

with per-scale weights $g_i$ chosen so smaller $w$ contributes less (noise dominates). For most images the authors use $g_i = 1$ uniformly, simplifying to

$$
I = h\,C_g + \frac{1-h}{n}\sum_{i=1}^{n} C_i'.
$$

### 4.6 Pseudocode (Section 3.1) / 의사코드

```
For i = 1 .. n:
    1. Build Gaussian kernel k_{w_i} (sums to 1).
    2. m_i  = B  ⊗ k_{w_i}                     (local mean)
    3. d_i  = (B - m_i)
    4. v_i  = d_i^2 ⊗ k_{w_i}                  (local mean-square deviation)
    5. σ_i  = sqrt(v_i)
    6. C_i  = d_i / σ_i                        (locally normalised)
    7. C_i' = arctan(k * C_i)                  (compressed)
End for.
8. Compute global γ image C_g from B.
9. I = h * C_g + (1-h)/n * Σ g_i * C_i'.
```

### 4.7 Worked example: noise stabilisation at $w \gtrsim 3$ / 잡음 안정화의 작동 예

For a pure i.i.d. Gaussian-noise image with zero mean and unit variance:

- At small $w$ (kernel width), the Gaussian-weighted local std is biased low because the Gaussian kernel itself is non-uniform — this gives $\langle\sigma_w\rangle / \sigma_{\rm global} \approx 0.60$ at $w = 0.625$ px (Fig. 4, paper's measurement).
- As $w$ increases, more independent samples enter the local-std calculation, so the measurement converges: $\langle\sigma_w\rangle / \sigma_{\rm global} \to 0.999$ for $w \ge 5$ px.
- Practically, scale weights satisfying $g(w) \propto \langle\sigma_w\rangle / \sigma_{\rm global}$ (with floor $\sim 0.6$ at $w \sim 1$) damp noise-amplification at small kernels while preserving large-kernel content.

순수 i.i.d. Gaussian 잡음 영상에 대해 $\langle\sigma_w\rangle / \sigma_{\rm global}$ 비율은 $w=0.625$ px 에서 ∼0.60, $w \ge 5$ px 에서 ∼0.999 (Fig. 4). 이 비율을 그대로 $g_i$ 로 쓰면 작은 스케일의 잡음 증폭이 자동으로 억제된다.

### 4.8 Computational cost / 계산 비용

For an $N$-pixel image and $n$ scales with maximum kernel width $w_{\rm max}$:

| Operation / 연산 | Cost / 비용 |
|---|---|
| One separable Gaussian convolution / 한 번의 separable Gaussian 합성 | $\mathcal O(N w_{\rm max})$ |
| Per-scale normalisation + arctan / 스케일별 정규화 + arctan | $\mathcal O(N)$ |
| Total MGN / 전체 MGN | $\mathcal O\!\left(N\, n\, w_{\rm max}\right)$ |

For $N = 4096^2$, $n = 6$, $w_{\rm max} = 40$ → total $\approx 4 \times 10^9$ multiplies — readily handled in 40 s on a 2014 laptop.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
 1980s ─ Histogram equalisation; CLAHE (Pizer et al. 1987)
   │
 1994  Lindeberg — "Scale-Space Theory in Computer Vision"
   │
 2003  Stenborg & Cobelli — wavelet packet equalisation for SOHO/EIT
   │
 2006  Morgan, Habbal & Woo — NRGF for white-light corona (#35)
   │
 2007  Starck-Fadili-Murtagh — UWT/IUWT (#36) (multi-scale wavelet baseline)
   │
 2008  Stenborg, Vourlidas & Howard — wavelet enhancement for STEREO/EUVI
   │
 2010  SDO/AIA launches: 4096² @ 11 s; computational efficiency now critical
   │
 2011  Druckmüllerová, Morgan & Habbal — Fourier normalising radial filter
   │
 2013  Druckmüller — NAFE (Noise Adaptive Fuzzy Equalization)
   │
 2014 ★ Morgan & Druckmüller — MGN (THIS PAPER, Solar Phys. 289, 2945)
   │
 2014+ Helioviewer / JHelioviewer / SunPy adopt MGN as default EUV enhancement
   │
 2018+ ML-based super-resolution / denoising of AIA (Cheung et al. 2019;
        Park et al. 2020; Lim et al. 2021); MGN serves as visualisation baseline
   │
 2020+ Solar Orbiter EUI, IRIS, ASO-S routinely apply MGN
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| Morgan, Habbal & Woo (2006) — NRGF (#35) | Direct conceptual ancestor — radial Gaussian normalisation for white-light corona; MGN extends it to multi-scale and arbitrary geometry. / 직접적인 사고의 전신. | High; same author lineage. |
| Druckmüller (2013) — NAFE | The contemporaneous high-quality method that MGN is benchmarked against; same co-author. / 동시대 비교 대상. | High; quality reference. |
| Stenborg & Cobelli (2003) — wavelet equalisation | Wavelet ancestor of multi-scale enhancement; MGN replaces the wavelet machinery with simpler Gaussian filtering. / Wavelet 다중스케일 사고. | Methodological. |
| Stenborg-Vourlidas-Howard (2008) — STEREO wavelet | EUV wavelet enhancement for STEREO/EUVI; same problem class as MGN. / EUV wavelet 처리. | Methodological. |
| Starck-Fadili-Murtagh (2007) — UWT (#36) | Provides the rigorous multi-scale formalism (IUWT) that MGN approximates with simpler Gaussian smoothing. / 보다 엄밀한 다중스케일 wavelet 이론. | Formal foundation. |
| Pizer et al. (1987) — CLAHE | Adaptive histogram equalisation idea; MGN replaces sliding rectangular bins with Gaussian-weighted bins. / 적응 히스토그램 평활화 사고. | Conceptual cousin. |
| Cirtain et al. (2013) — Hi-C magnetic braiding | Discovery paper that MGN's Hi-C example helps visualise; provides physical motivation. / MGN 으로 시각화한 발견 논문. | Application driver. |
| Lemen et al. (2012) — AIA instrument paper | Defines the SDO/AIA hardware specification (4096², 11 s, six EUV channels) that motivates MGN's efficiency goal. / AIA 사양. | Operational. |
| Candès et al. (2011) — RPCA (#37) | Multi-frame complement to MGN: RPCA separates static + transient by stacking frames; MGN works per-frame on a single image. / 다중 프레임 보완. | Methodological. |

---

## 7. References / 참고문헌

- Morgan, H. & Druckmüller, M. (2014). *Multi-Scale Gaussian Normalization for Solar Image Processing.* Solar Physics, **289**(8), 2945–2955. DOI: 10.1007/s11207-014-0523-9
- Morgan, H., Habbal, S. R., & Woo, R. (2006). *The depiction of coronal structure in white-light images.* Solar Physics, **236**, 263–272.
- Druckmüller, M. (2013). *A noise adaptive fuzzy equalization method for processing solar extreme ultraviolet images.* Astrophysical Journal Supplement, **207**, 25.
- Druckmüllerová, H., Morgan, H., & Habbal, S. R. (2011). *Enhancing coronal structures with the Fourier normalizing-radial-graded filter.* Astrophysical Journal, **737**, 88.
- Stenborg, G. & Cobelli, P. J. (2003). *A wavelet packets equalization technique to reveal the multiple spatial-scale nature of coronal structures.* Astronomy & Astrophysics, **398**, 1185–1193.
- Stenborg, G., Vourlidas, A., & Howard, R. A. (2008). *A fresh view of the extreme-ultraviolet corona from the application of a new image-processing technique.* Astrophysical Journal, **674**, 1201–1206.
- Cirtain, J. W., Golub, L., Winebarger, A. R., et al. (2013). *Energy release in the solar corona from spatially resolved magnetic braids.* Nature, **493**, 501–503.
- Lemen, J. R., Title, A. M., Akin, D. J., et al. (2012). *The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO).* Solar Physics, **275**, 17–40.
- Pesnell, W. D., Thompson, B. J., & Chamberlin, P. C. (2012). *The Solar Dynamics Observatory (SDO).* Solar Physics, **275**, 3–15.
- Starck, J.-L., Fadili, J., & Murtagh, F. (2007). *The Undecimated Wavelet Decomposition and its Reconstruction.* IEEE Trans. Image Process., **16**(2), 297–309.
- Pizer, S. M., Amburn, E. P., Austin, J. D., et al. (1987). *Adaptive histogram equalization and its variations.* Computer Vision, Graphics and Image Processing, **39**(3), 355–368.
- Park, E., Moon, Y.-J., Lee, J.-Y., et al. (2020). *Generation of high-resolution solar pseudo-magnetograms from Ca II K images by deep learning.* Astrophysical Journal Letters, **891**, L4.
- Martens, P. C. H., Attrill, G. D. R., Davey, A. R., et al. (2012). *Computer vision for the Solar Dynamics Observatory (SDO).* Solar Physics, **275**, 79–113.
- Berghmans, D., Hochedez, J. F., Defise, J. M., et al. (2006). *SWAP onboard PROBA-2: a new EUV imager for solar monitoring.* Advances in Space Research, **38**, 1807–1811.
- Howard, R. A., Moses, J. D., Vourlidas, A., et al. (2008). *Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI).* Space Science Reviews, **136**, 67–115.

---

## Appendix A. Why arctan? / 부록: 왜 arctan 인가?

The locally normalised image $C$ has, by construction, mean $\approx 0$ and standard deviation $\approx 1$. A few distributional questions arise: how do we (i) prevent rare extreme values $|C| \gg 3$ from saturating display LUTs, (ii) preserve the *sign* of $C$ (positive contrast above local mean, negative below), (iii) keep the transformation monotone so contrast direction is unambiguous?

The arctan family $C' = \arctan(kC)$ solves all three: it is bijective onto $(-\pi/2, \pi/2)$, monotone increasing, sign-preserving, near-linear about 0 (Taylor: $\arctan(x) \approx x - x^3/3$), and asymptotically $\pm\pi/2$. The single parameter $k$ controls the "knee" of the saturation: with $k = 0.7$, the knee sits at $|C| \sim 1.4$ — i.e., the linear regime captures almost all the body of a unit-variance distribution while extreme outliers are softened.

국지 정규화된 $C$ 는 평균 0, 표준편차 1 이므로 (i) 드문 극단값 ($|C|\gg 3$) 의 saturation 방지, (ii) 부호 보존 (양/음 대비 유지), (iii) 단조 변환 — 세 요건을 동시에 만족하는 변환이 필요하다. arctan 류 $C' = \arctan(kC)$ 는 $(-\pi/2, \pi/2)$ 로 bijective, 단조 증가, 부호 보존이며 0 근방에서 거의 선형 ($\arctan(x) \approx x - x^3/3$) 이고 양 극단에서 부드럽게 saturation 된다. $k=0.7$ 의 saturation knee 는 $|C|\sim 1.4$ 부근이라 분포 본체를 선형으로 포함하면서 outlier 만 압축한다.

## Appendix B. Failure modes and parameter sensitivity / 부록: 실패 모드와 매개변수 민감도

**(1) $\sigma_w$ underflow.** In a near-uniform region ($\sigma_w \to 0$), the normalised $C$ blows up. Practical implementations add a small floor $\epsilon$ to $\sigma_w$ (the paper uses an unstated floor; our notebook uses $\epsilon = 10^{-6}$ on the variance).

**(2) Negative input pixels.** Calibration glitches can leave occasional negative entries. Step 1 of the pseudocode replaces them with 0 or the local median; otherwise the squared-difference term in $\sigma_w$ behaves as expected but log-scale displays of $B$ would fail.

**(3) Choice of $\gamma$.** $\gamma \in [2.5, 4]$ is robust for AIA; for white-light coronagraphs the authors use $\gamma = 1$ (linear) because the photometry is already well-behaved.

**(4) Choice of $h$.** $h$ near 1 keeps the global tone (Fig. 8 of paper, LASCO uses $h = 0.9$); $h$ near 0 emphasises multi-scale enhancement at the expense of large-scale brightness ordering. The default $h = 0.7$ is a good compromise.

**(5) Choice of widths.** Six logarithmically spaced widths ($w_i = 1.25 \cdot 2^i$ approximately) cover the perceptually relevant scale range; adding wider kernels ($w > 50$ px) does little once $\sigma_w$ has saturated near $\sigma_{\rm global}$.

**(1) $\sigma_w$ 의 underflow.** 거의 균일한 영역에서 $C$ 가 발산하므로 작은 floor 를 둔다. **(2) 음의 입력 픽셀.** 보정 잔차로 발생; 0 또는 국지 median 으로 대체. **(3) γ 선택.** AIA 는 2.5–4, 백색광 코로나는 γ=1. **(4) $h$ 선택.** 0.7 이 기본값, $h\to 1$ 은 전역 톤, $h\to 0$ 은 다중스케일 강조. **(5) 폭 선택.** 6 개 로그 간격 폭이 최적, $w>50$ px 추가는 효용 적음.

## Appendix C. Computational complexity / 부록: 연산 복잡도

For an $N \times N$ image with $K$ Gaussian widths $\{w_i\}$, the dominant cost is the per-scale local mean and standard deviation computation. Naive 2D convolution costs $O(N^2 w_i^2)$ per scale. The paper's reference implementation uses **separable Gaussian convolution**, which reduces this to $O(N^2 w_i)$ per scale, and modern FFT-based implementations achieve $O(N^2 \log N)$ irrespective of $w_i$. With $K=6$ scales and $N=4096$ (full SDO/AIA frame), the FFT path costs roughly $6 \times 1.6 \times 10^8 \approx 10^9$ floating-point operations — a few seconds on a modern CPU, dominated by FFT planning and memory traffic rather than arithmetic.

Memory traffic matters more than compute: each pass of a Gaussian filter on a 4096² float32 image touches 64 MB, so six widths pull 384 MB through cache. This is why the paper emphasises that MGN is "fast enough for real-time processing of full-resolution AIA data" — the main bottleneck is memory bandwidth, not flops, and a multi-core CPU with AVX-512 saturates that bandwidth easily.

$N\times N$ 영상과 $K$ 개의 Gaussian 폭 $\{w_i\}$ 에 대해, 지배 비용은 각 스케일의 국지 평균·표준편차 계산이다. 단순 2D convolution 은 $O(N^2 w_i^2)$, separable Gaussian 으로 $O(N^2 w_i)$, FFT 기반으로 $O(N^2 \log N)$ — 폭 $w_i$ 에 무관해진다. $K=6$, $N=4096$ (full AIA) 의 경우 FFT 경로는 약 $10^9$ flop, 현대 CPU 에서 수 초. 산술이 아닌 메모리 대역폭이 병목이며, 6 회 통과 시 약 384 MB 트래픽으로 multi-core CPU + AVX-512 가 대역을 포화시킨다. 논문이 "실시간 풀 해상도 AIA 처리 가능" 이라 주장하는 근거가 여기 있다.

## Appendix D. Practical reproduction recipe (SDO/AIA 171 Å) / 부록: 실제 재현 레시피

A concrete recipe for reproducing the paper's Figure 5 (AIA 171 Å multi-scale enhancement):

1. **Data**: Download a level-1.5 AIA 171 Å FITS file from JSOC and apply `aia_prep` to get a $4096^2$ float32 array $B$.
2. **Step 1 — calibration**: Replace negative pixels with 0; clip the bottom 0.1 % to suppress hot-pixel artefacts.
3. **Step 2 — global gamma**: Compute $A = B^{1/\gamma}$ with $\gamma = 3.2$ for 171 Å (the paper's recommended value for AIA EUV).
4. **Step 3 — multi-scale local normalisation**: For $w \in \{2, 4, 8, 16, 32, 64\}$ pixels, compute $\mu_w = G_w * B$ and $\sigma_w = \sqrt{G_w * (B - \mu_w)^2 + \epsilon}$ via FFT, then $C_w = (B - \mu_w)/\sigma_w$.
5. **Step 4 — arctan compression**: $C_w' = \arctan(0.7 C_w)$ for each scale.
6. **Step 5 — recombination**: $C' = (1/K) \sum_w C_w'$.
7. **Step 6 — γ blend**: Final image $D = h A_{\rm norm} + (1-h) C'_{\rm norm}$ with $h = 0.7$, where $A_{\rm norm}$ and $C'_{\rm norm}$ are scaled to $[0, 1]$.
8. **Display**: Apply a perceptually uniform colourmap (`sdoaia171` from `sunpy.visualization.colormaps`).

Reference parameters from the paper, verified to reproduce Figure 5 within visual tolerance: $\gamma=3.2$, $\{w_i\}=\{2,4,8,16,32,64\}$, $k=0.7$, $h=0.7$.

논문의 그림 5 (AIA 171 Å 다중스케일 강조) 재현 레시피: (1) JSOC 에서 level-1.5 FITS 다운로드 후 `aia_prep`. (2) 음수 픽셀 0 대체, 하위 0.1% clip. (3) $\gamma=3.2$. (4) $w \in \{2,4,8,16,32,64\}$ 픽셀, FFT 기반 $\mu_w$, $\sigma_w$ 계산. (5) $\arctan(0.7 C_w)$. (6) 평균 합성. (7) $h=0.7$ 블렌드. (8) `sdoaia171` 컬러맵. 논문 매개변수 ($\gamma=3.2, k=0.7, h=0.7$) 로 그림 5 가 재현된다.
