---
title: "The Depiction of Coronal Structure in White-Light Images (NRGF)"
authors: Huw Morgan, Shadia Rifai Habbal, Richard Woo
year: 2006
journal: "Solar Physics"
doi: "10.1007/s11207-006-0113-6"
topic: Low_SNR_Imaging
tags: [coronagraph, NRGF, white-light, LASCO, image-enhancement, normalisation, polar-coordinates]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 35. The Depiction of Coronal Structure in White-Light Images / 백색광 영상에서의 코로나 구조 묘사 (NRGF)

---

## 1. Core Contribution / 핵심 기여

Morgan, Habbal & Woo (2006) introduce the **Normalising Radial Graded Filter (NRGF)** — a one-line algorithm that revolutionised coronagraph image display. The K-corona brightness drops by a factor of ~$10^4$ between $1\,R_\odot$ and $3\,R_\odot$, so any single linear stretch can show only a thin slice of the coronal field of view. The NRGF replaces each pixel of a coronagraph image with its *azimuthal z-score*: subtract the mean intensity at that heliocentric distance and divide by the standard deviation at that distance. The result is an image where structure at every height is on equal footing, allowing simultaneous depiction of low corona, mid corona, and outer corona in one frame. The paper additionally introduces a procedure for subtracting a **time-stable unpolarized background** from LASCO C2 *total*-brightness images — which is the long-term average of (total brightness − polarized brightness) — enabling NRGF-quality results to be produced from the high-cadence total-brightness stream rather than the rare polarized-brightness observations. The combined pipeline is demonstrated on a 6-hour LASCO C2 sequence of a CME on 2001-01-07 and reveals a small dark cavity, a heart-shaped filamentary outer loop, and faint twists trailing the central cavity — features invisible in standard processing.

본 논문은 코로나그래프 영상 표시를 혁신한 한 줄 알고리즘 **NRGF (Normalising Radial Graded Filter)**를 도입한다. K-corona 밝기는 1 $R_\odot$에서 3 $R_\odot$ 사이 약 $10^4$배 떨어지므로 단일 stretch로는 전체 시야를 보여줄 수 없다. NRGF는 각 픽셀을 *방위각 평균을 뺀 후 방위각 표준편차로 나눈* z-score로 대체한다. 결과는 모든 높이의 구조가 동등하게 보이는 영상이다. 또한 LASCO C2 *총 밝기(tB)* 영상에서 사용할 시간-안정적인 *편광되지 않은 배경* 빼기 절차를 함께 제시하는데, 이는 (tB - pB)의 장기 평균으로 만들어진다. 두 절차를 결합해 2001-01-07 CME의 6시간 LASCO C2 시퀀스를 처리하면 작은 어두운 cavity, 하트 모양의 필라멘트 외곽 loop, 중심부를 따르는 미세한 twist 등 표준 처리로는 보이지 않던 구조가 드러난다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction — Why the Radial Gradient Defeats Standard Display / 방사형 기울기가 표준 표시를 무력화하는 이유 (Sec. 1, p. 2)

The introduction frames the problem in physical terms. The K-corona brightness $B$ scales roughly as a power-law with heliocentric distance: $B \propto r^{-\alpha}$ with $\alpha \sim 5\!-\!7$ in the inner corona, dropping by a factor of $\sim\!10^4$ from the limb to $3\,R_\odot$. Hardware Radial Graded Filters (RGFs) — physical neutral-density wedges — have been used since Newkirk & Harvey (1968), and multi-exposure compositing (Guillermier & Koutchmy 1999) is another solution, but software approaches in the LASCO era (running mean subtraction, monthly background division, etc.) introduce artifacts and east/west asymmetries. The motivation is simple: white-light coronal images are routinely used to (i) compare with potential-field source-surface and MHD models, (ii) place in-situ measurements (Solar Wind, ACE, STEREO) in their large-scale context, and (iii) study CMEs morphologically — and they should produce a *true picture*, not a tool-distorted one.

K-corona 밝기는 거리의 거듭제곱으로 떨어져 limb에서 3 $R_\odot$까지 $10^4$배 감소한다. Newkirk & Harvey(1968)의 하드웨어 RGF나 다중 노출 합성은 옛 해법이고, LASCO 시대의 SW 후처리(running mean, monthly bg division)는 인공물이 많다. 코로나 영상은 PFSS·MHD 모델과 비교, in-situ 측정의 large-scale 맥락 제공, CME 형태 분석에 쓰이므로 *왜곡 없는 정확한 표시*가 필수다.

### Part II: The NRGF Algorithm / NRGF 알고리즘 (Sec. 2, p. 3)

The whole filter is one equation:

$$
I'(r,\phi) = \frac{I(r,\phi) - \langle I(r)\rangle_\phi}{\sigma(r)_\phi}
$$

where $I(r,\phi)$ is the original intensity at heliocentric distance $r$ and position angle $\phi$, $\langle I(r)\rangle_\phi$ is the azimuthal mean over all $\phi$ at that $r$, and $\sigma(r)_\phi$ is the azimuthal standard deviation at that $r$. The ingredients are entirely classical statistics — this is just *z-scoring each annulus* — but the implication is profound. Figure 1(a) of the paper shows three latitudinal $pB$ profiles at $1.5,2.3,5.0\,R_\odot$ at solar maximum, which on a linear axis are dominated by the radial drop and look almost flat at $5.0\,R_\odot$. Figure 1(b) shows the same three profiles after each is normalised: now they are visually comparable, sharing a dynamic range of $\pm 3$ standard deviations.

**Implementation note.** The image is first re-sampled into polar coordinates $(r,\phi)$. At each $r$, the empirical mean $\bar{I}_r$ and stdev $\sigma_r$ are computed over the annulus, then every pixel in that annulus is standardised. The output is then re-sampled back to Cartesian (or kept polar for analysis).

The same procedure is applied to images from three instruments — EIT (304 Å disk), MLSO MKIII/MKIV ($1.15\!-\!2.4\,R_\odot$), and LASCO C2 ($2.3\!-\!6.0\,R_\odot$). A useful side-effect: because every $r$ is normalised independently to mean 0 and stdev 1, the boundaries between instruments are visually seamless without any smoothing or interpolation — equation 1 *forgives* instrumental discontinuities in low-signal regions. Figure 2 of the paper shows the resulting composite for solar minimum (1997-01-18) and solar maximum (2000-12-03).

NRGF는 한 줄 식이다 — 각 반경 $r$에서 방위각 평균 $\langle I(r)\rangle_\phi$를 빼고 방위각 표준편차 $\sigma(r)_\phi$로 나눈 z-score로 픽셀을 대체한다. Fig. 1(a) → (b)에서 보듯, 동적 범위가 $\pm 3\sigma$로 평탄화되어 모든 높이의 구조가 동등하게 표시된다. 구현은 극좌표 변환 → annulus별 통계 → 표준화 → 직교좌표 복원 순. EIT(304 Å), MLSO, LASCO C2 세 기기를 결합해도 *경계가 매끄럽다* — 각 $r$이 독립적으로 정규화되기 때문이다.

### Part III: Removing the Unpolarized Background from Total-Brightness Images / 총 밝기 영상에서 편광되지 않은 배경 제거 (Sec. 3, pp. 4-6)

The NRGF on its own works on $pB$ data, but $pB$ measurements are infrequent (a few sequences per day on LASCO C2) compared to $tB$ which is taken at high cadence. To get NRGF-quality images at high cadence, the F-corona, stray light, and other unpolarized contributions to $tB$ must be removed first. The authors construct an *unpolarized background* as the long-term average of $(tB - pB)$:

$$
B_{unpol}(r,\phi) = \big\langle\, I_{tB}(r,\phi;t) - I_{pB}(r,\phi;t)\,\big\rangle_{t \in \text{Carrington rotation}}
$$

so each $tB$ image is corrected by

$$
I_{tB}^{\text{corr}}(r,\phi) = I_{tB}(r,\phi) - B_{unpol}(r,\phi)
$$

before NRGF is applied. The key empirical claim, supported by Figure 3 (left), is that **$B_{unpol}$ is virtually unchanged between solar minimum (1997 Jan) and solar maximum (2000 Dec)** above the equator and at the North pole, with a slightly steeper falloff at the poles. This time-stability is what makes a long-term average viable — it does *not* wash out time-varying features because the time-varying components are almost entirely in the polarized brightness. Figure 3 (right) maps $\log_{10}$ of this background as a contour plot in the LASCO C2 field.

**Validation (Fig. 4).** The authors compare $pB$ (dashed) with $tB$ processed by their procedure (solid) at $3.5\,R_\odot$ for both solar minimum and maximum. The top row shows the two profiles after background subtraction; bottom row shows them normalised to mean 0 / stdev 1. The profiles match excellently at solar minimum, less perfectly at solar maximum (where the corona is more dynamic and the background is less stable), but in both cases match far better than the standard LASCO Quick Look output (bottom row), which exhibits artefacts: the QL east streamer has a "secondary peak" that does not exist in $pB$, and the QL west streamer is far too bright relative to the east. This is direct evidence that *standard tools introduce structure that is not real*.

NRGF만으로는 빈도 낮은 $pB$ 영상에 한정된다. 자주 찍히는 $tB$를 NRGF로 처리하려면 F-corona·stray light 같은 편광되지 않은 기여를 먼저 빼야 한다. 저자들은 $(tB - pB)$의 *장기 평균*을 unpolarized background로 정의한다. Fig. 3 left가 보여주듯 이 background는 1997 (solar min)과 2000 (solar max) 사이에서 거의 변하지 않으므로 장기 평균이 valid하다 — 시간 변화 성분은 거의 모두 polarized 부분에 있기 때문이다. Fig. 4의 검증: $pB$와 (background 뺀) $tB$가 잘 일치하며, 표준 LASCO Quick Look은 동/서 streamer 비대칭과 가짜 secondary peak 같은 인공물을 만들어낸다.

### Part IV: NRGF + Background Subtraction Applied to a CME / CME에 적용한 NRGF + 배경 빼기 (Sec. 4, pp. 8-9)

Figure 5 of the paper shows six LASCO C2 images of a CME propagating through the C2 field over $\sim$6 hours starting on 2001-01-07 02:00 UT. Each image is processed by background subtraction followed by NRGF. The NRGF mean and stdev profiles are *averaged from observations before and after* the CME passage (not during), to keep the normalisation reference time-stable and prevent the CME itself from contaminating the statistics.

What appears in the processed sequence:
- A **small dark central cavity** — likely the magnetic flux rope / void that conventional theory predicts.
- A **heart-shaped filamentary outer loop** — the CME's leading edge, seen with internal structure rather than as a smooth front.
- **Faint twists** trailing the central cavity, suggestive of helically-organised field — in five of six images.
- All of this at *uniform contrast across the entire $2.2\!-\!6.1\,R_\odot$ field*, without time-difference imaging or unsharp masking.

The authors emphasise that conventional time-difference (running difference) imaging shows *changes* but distorts morphology because of the moving subtraction reference. The NRGF approach shows *absolute* structure at every frame, which is what one needs for morphological and 3D-reconstruction studies.

CME 적용. Fig. 5는 2001-01-07 02:00 UT부터 6시간 LASCO C2 시퀀스를 NRGF + 배경 빼기로 처리한 6 컷이다. 통계 참조는 CME 지나가기 *전후*의 영상에서 평균으로 만들어 통계가 시간-안정적이도록 한다. 처리 영상에서 작은 중심 cavity, 하트 모양 외곽 loop의 필라멘트 구조, 중심 cavity를 따르는 미세한 twist가 *전 시야에 균일한 대비로* 드러난다. 표준 running difference는 *변화*만 보여주지만 NRGF는 매 프레임의 *절대* 형태를 보여준다.

### Part V: Conclusions and Implications for Coronal Topology / 결론과 코로나 위상학에의 함의 (Sec. 5, pp. 8-9)

The conclusions section is unusually substantive. The authors argue that NRGF-processed images *reframe the basic question* of coronal structure: standard LASCO Quick Look images suggest streamers are *solid blocks* with sharp boundaries on a flat background, but NRGF images show streamers as *bundles of myriad filaments* embedded in a fainter sea of finer structure that is itself non-homogeneous. This raises the question: where exactly is the boundary of a streamer? The authors propose that streamers may simply be regions where filament density is higher, not regions with categorically different physics. This connects with prior radio (Woo & Habbal 1997) and density-profile (Woo & Habbal 1997a, Habbal & Woo 2001) evidence for the highly filamentary corona.

The closing remark looks forward to STEREO (then about to launch) and notes that NRGF will become a standard tool for that mission's white-light coronagraph data — a prediction that proved correct.

결론은 단순한 "잘 보인다"가 아니라 *코로나 구조 자체에 대한 질문을 다시 짠다*. 표준 LASCO QL은 streamer를 균일한 덩어리로 보이게 하지만, NRGF는 streamer가 *수많은 필라멘트의 다발*임을 드러낸다 — streamer의 경계 정의 자체가 모호해진다. 이는 라디오·밀도 프로파일에서의 기존 증거(Woo & Habbal 1997 등)와 일치한다. 곧 발사될 STEREO에 NRGF가 표준 도구가 될 것이라는 예측이 본문 끝에 등장한다(이후 사실로 입증됨).

### Part VI: Implementation Details and Computational Considerations / 구현 세부사항과 계산상의 고려 (synthesis from Secs. 2-4)

While the paper presents only a single equation, a faithful implementation involves several practical decisions:

- **Polar resampling.** Choose a polar grid with $N_\phi$ angular bins (typical: 360 or 720) and $N_r$ radial bins (typical: $\sim 200$ for LASCO C2). The choice of $N_\phi$ controls the smoothness of the azimuthal mean — too few and you lose structure, too many and you introduce noise.
- **Sun-centre determination.** The Sun centre must be known sub-pixel — LASCO's pointing model gives this, but for ground-based data limb-fitting may be needed.
- **Occulter mask.** The internal occulter and pylons must be masked before computing the azimuthal statistics, otherwise the "zero" pixels behind them would bias $\bar{I}_r$ down and inflate $\sigma_r$.
- **Boundary effects.** At the inner and outer edges of the field of view, fewer angular samples may be available; the authors do not discuss this, but typical implementations either truncate the radial range or weight by valid-pixel count.
- **Display stretch.** The output $I'$ is roughly in $[-3,3]$ standard deviations; a final clipping to $[-3,3]$ and linear remapping to $[0,255]$ is standard for display.

본 논문은 한 줄 식이지만, 실제 구현에는 여러 결정이 필요하다 — 극좌표 resampling 격자 크기 ($N_\phi\sim360,N_r\sim200$), 태양 중심 sub-pixel 추정, occulter/pylon 마스킹(통계 오염 방지), 시야 경계 처리, 표시용 $[-3,3]\sigma$ clipping과 $[0,255]$ 매핑 등.

### Part VII: Why z-Scoring Each Annulus Works / annulus z-score가 잘 작동하는 이유

A deeper question: *why* should standardising every annulus produce visually pleasing, scientifically meaningful images? Three reasons:

1. **The radial gradient is well-modelled by the azimuthal mean.** If the K-corona is approximately axisymmetric, $\langle I(r)\rangle_\phi \approx I_{\text{spherical}}(r)$, the mean is the natural separator of "background" from "structure." Subtracting $\langle I(r)\rangle_\phi$ approximates what a hardware RGF would do.
2. **The structure amplitude scales with the radial brightness.** The relative contrast of streamers, plumes, etc. is roughly preserved at each height — $b(r)\propto a(r)$ is a reasonable approximation. Dividing by $\sigma(r)_\phi$ undoes this height-dependent amplitude, putting all heights on equal footing in $\sigma$ units.
3. **Human visual perception is logarithmic in intensity.** Z-scoring is a near-linear approximation to log-stretching that uses *empirical* log scaling computed from the data itself, rather than assuming a fixed exponent.

This is why NRGF works without any tuning: it uses the data itself to determine the right "slot" for every height.

왜 annulus 단위 표준화가 잘 작동하는가? ① K-corona는 거의 축대칭이므로 방위각 평균이 *배경*을 잘 근사한다. ② 구조의 진폭이 반경에 따른 밝기에 비례해 변하므로($b(r)\propto a(r)$), $\sigma$로 나누면 높이별 진폭이 정규화된다. ③ 인간 시각이 강도에 대해 로그적이므로, z-score는 *데이터에서 결정된* 경험적 로그-스트레치의 선형 근사다. 따라서 NRGF는 별도 튜닝 없이 작동한다 — 데이터가 스스로 옳은 표시 슬롯을 결정한다.

### Part VIII: Limitations and What NRGF Does Not Do / 한계와 NRGF가 못하는 것

It is important to know what NRGF is *not*:

- **Not a denoising filter.** Random noise is preserved (and possibly amplified at high $r$ where $\sigma_r$ is small). For denoising, combine with Stenborg-Cobelli (#32) wavelet processing or a modern deep denoiser.
- **Not a deconvolution.** Point-spread function blur and exposure-time motion smearing are not addressed.
- **Not photometric.** Output values $I'$ are in standard-deviation units, not physical brightness. Quantitative photometry must use the original $I$.
- **Sensitive to large transients.** A single bright CME contaminates the azimuthal statistics during its passage; the authors mitigate this by computing reference statistics from images *outside* the CME window (Sec. 4).
- **Assumes approximate axisymmetry.** Strongly asymmetric coronas (e.g., during major flux-rope eruptions) may have weight residual from the inhomogeneous mean.

NRGF가 *못하는 것*: 잡음 제거(오히려 외곽에서 증폭될 수 있음 — 32번 논문이나 딥 디노이저와 결합 필요), deconvolution(PSF/노출 번짐 미해결), 측광(출력은 표준편차 단위로, 물리량이 아님), 큰 transient 대응(CME 중 통계 오염), 비축대칭 코로나 처리. 사용 시 한계를 분명히 인식해야 한다.

### Part IX: Detailed Analysis of Figure 4 — Quantitative Validation / Fig. 4 정량 검증 상세

Figure 4 of the paper is the most important quantitative result and deserves a careful walkthrough. The figure shows latitudinal profiles at $3.5\,R_\odot$ for two epochs (1997-01-18 solar minimum, 2000-12-02 solar maximum) and three processing schemes:

- **Top row (background-subtracted, raw amplitude):** $pB$ (dashed) and $tB - B_{unpol}$ (solid) at 1997 (left) and 2000 (right). Solar minimum: profiles match almost perfectly except in the low-signal "polar" regions (90° and 270° position angle). Solar maximum: pB and tB profiles agree on the *shape* and *peak positions* but the absolute amplitudes differ — meaning that the residual $tB - pB - B_{unpol}$ is non-zero (transient activity contaminates the background subtraction), but only by a few percent.

- **Middle row (both normalised to mean 0, stdev 1):** the agreement is now excellent at both epochs. *Structure* is identical between the pB and processed-tB images, differing only in absolute amplitude. This validates the entire pipeline: NRGF + background subtraction makes a tB image visually equivalent to a pB image *in structural terms*.

- **Bottom row (pB vs LASCO Quick Look standard processing):** the LASCO QL output deviates dramatically from pB. The 1997 east streamer (~270°) appears with a phantom secondary peak that does not exist in pB. The 2000 west streamer (~90°) is too bright. The 2000 east streamer at ~270° appears completely missing from QL — but is plainly there in pB.

The implication is direct: **studies that used LASCO QL images to characterise streamer morphology may have to be revisited.** The background division and running-mean subtraction in QL processing produce visible structure that is artefactual.

Fig. 4 정량 검증: ① 위 행(배경 빼고 원래 진폭)에서 solar min은 거의 완벽히 일치, solar max는 진폭이 약간 다르지만 모양은 같다 — transient 오염은 수 % 수준. ② 중간 행(둘 다 mean 0/stdev 1로 정규화) — 구조는 동일하다는 결정적 증거. ③ 아래 행은 LASCO Quick Look이 *가짜* secondary peak와 누락 streamer를 만들어낸다는 사실을 보여줌 — QL 영상으로 한 streamer 형태 연구는 *재검토*가 필요할 수 있다.

### Part X: Connection to Statistical Normalisation in Deep Learning / 딥러닝의 통계 정규화와의 연결

In hindsight, NRGF can be read as a domain-specific instance of the family of *normalisation operators* that became central to modern deep learning. The mapping is:

| Operation | Description | NRGF analogue |
|---|---|---|
| Batch Norm (Ioffe-Szegedy 2015) | $z = (x-\mu_{\text{batch}})/\sigma_{\text{batch}}$ per channel | NRGF normalises per *radius* — analogous to per-channel statistics |
| Layer Norm (Ba 2016) | per-sample feature normalisation | NRGF can be read as per-image, per-radius layer norm |
| Group Norm (Wu-He 2018) | per group of channels | NRGF normalises angular pixels in each radial "group" |
| Instance Norm (Ulyanov 2016) | per-sample, per-channel | Closest analogue to NRGF |

The deeper lesson — *use the data itself to compute the right normalising statistics* — is identical in both domains. NRGF (2006) anticipated by ~9 years the BatchNorm philosophy that revolutionised neural network training, in the much narrower domain of coronal images.

NRGF는 회고적으로 보면 딥러닝의 정규화 연산자(BatchNorm, LayerNorm, GroupNorm, InstanceNorm) 가족의 *도메인-특정 사례*다. *데이터 자체에서 정규화 통계를 계산하라*는 깊은 교훈은 동일하다. NRGF(2006)는 BatchNorm(2015)의 철학을 코로나 영상이라는 좁은 도메인에서 9년 앞서 적용했다 — 알고리즘적 단순성이 시대 정신을 앞지른 사례.

### Part XI: Practical Tips for Reproducing NRGF / NRGF 재현을 위한 실용 팁

A practitioner reproducing NRGF should:

1. Convert image to polar coordinates with sub-pixel-accurate Sun centre.
2. Mask occulter, pylons, saturated stars, and any hot pixels *before* computing $\bar{I}_r$ and $\sigma_r$.
3. For low-cadence pB data, apply NRGF directly. For high-cadence tB data, first build the unpolarized background by averaging $(tB - pB)$ over $\geq$ one Carrington rotation.
4. For CME studies, compute the NRGF reference statistics from images immediately before/after the CME passage, not during.
5. After NRGF, optionally clip to $[-3, 3]$ and remap linearly to display range. For *publication*, retain the actual NRGF values and let the journal handle the stretch.
6. Combine with Stenborg-Cobelli wavelet processing or MGN for a maximum-quality result.

Modern open-source implementations: SunPy's `sunpy.image.coalignment` and the standalone `sunkit_image.enhance.nrgf` function. The original IDL implementation is also available as `nrgf.pro` in SolarSoft.

NRGF 재현을 위한 실용 팁: ① sub-pixel 태양 중심으로 극좌표 변환. ② occulter·pylon·saturated star 마스킹 후 통계 계산. ③ pB 데이터엔 직접 적용, tB엔 (tB-pB)의 1 Carrington rotation 이상 평균을 빼기 먼저. ④ CME 연구 시 통계는 CME 통과 *전후* 영상에서 계산. ⑤ $[-3,3]$ clipping은 표시용에만 적용, 출판용은 원래 값 보존. ⑥ Stenborg-Cobelli wavelet 또는 MGN과 결합. 오픈소스 구현은 SunPy의 `sunkit_image.enhance.nrgf`, IDL은 SolarSoft `nrgf.pro`.

### Part XII: Common Misconceptions about NRGF / NRGF에 대한 흔한 오해

Several misconceptions appear regularly in literature using NRGF-processed images:

1. **"NRGF brightens the corona"** — false. NRGF *standardises*: it makes faint structure *visible* by removing the dominating mean, but adds no actual signal. If $\sigma(r)_\phi=0$ at some radius, NRGF is undefined (a perfectly axisymmetric corona would map to a NaN-everywhere image).

2. **"NRGF removes noise"** — false. Random noise contributes to $\sigma(r)_\phi$, so it is *partially* compensated, but high-frequency noise is preserved (and can be visually amplified at low-signal heights). For genuine denoising, combine with Stenborg-Cobelli or modern deep denoisers.

3. **"NRGF preserves intensity ratios"** — false. The output is in standard-deviation units that vary with radius, so a streamer that is 2× brighter than a coronal hole at $2\,R_\odot$ may *not* show as 2× brighter in the NRGF output if $\sigma(r)$ differs at the relevant radii.

4. **"NRGF works on EUV disk images"** — partially true. The radial-symmetry assumption holds reasonably for limb regions but breaks down on the disk where active regions create strong azimuthal asymmetry. On-disk, alternative filters (e.g., MGN) are usually preferred.

5. **"Higher $N_\phi$ is always better"** — false. For typical LASCO data $N_\phi=720$ is sufficient; larger $N_\phi$ adds little useful information and increases noise in the per-annulus statistics.

NRGF에 대한 흔한 오해: ① "코로나를 밝게 만든다" — 틀림, *표준화*만 함. ② "잡음을 제거한다" — 틀림, 일부 보정될 뿐 기본적으로 보존. ③ "강도 비를 보존한다" — 틀림, $\sigma(r)$이 반경마다 달라 비율이 바뀜. ④ "EUV disk 영상에 작동한다" — 부분 사실, limb은 OK이지만 활동 영역이 있는 disk는 MGN 등이 더 적합. ⑤ "$N_\phi$가 클수록 좋다" — 틀림, 720이면 충분.

### Part XIII: A Numerical Example — Step-by-Step NRGF on a 64×64 Toy Image / 64×64 토이 이미지의 단계별 NRGF

To make the algorithm concrete, here is a complete worked example. Consider a 64×64 image where:

- The Sun centre is at $(32, 32)$.
- The radial profile is $a(r) = e^{-r/8}$ (steep drop, decade in 18 pixels).
- A "streamer" pattern is added: $s(\phi) = \cos(2\phi)$ (two-fold symmetry).
- The full image is $I(r,\phi) = a(r)\big[1 + 0.3\,s(\phi)\big] + \text{Gaussian noise with } \sigma=0.005$.

Steps:

1. Convert each $(x,y)$ to $(r,\phi)$ where $r=\sqrt{(x-32)^2+(y-32)^2}$, $\phi=\arctan_2(y-32, x-32)$.
2. Bin pixels into 20 radial bins of width 1 pixel from $r=2$ to $r=22$.
3. For each bin: compute $\bar{I}_r = \frac{1}{N_r}\sum_{(x,y)\in \text{bin}} I(x,y) \approx a(r)$ (since $\langle s\rangle = 0$).
4. Compute $\sigma(r)_\phi = \sqrt{\frac{1}{N_r}\sum (I-\bar{I}_r)^2} \approx 0.3\cdot a(r)/\sqrt{2}$ (the streamer contribution dominates for small noise).
5. Replace each pixel: $I'(r,\phi) = (I(r,\phi)-\bar{I}_r)/\sigma(r)_\phi \approx \cos(2\phi)/0.707/0.3 \cdot 0.3 = \cos(2\phi)\cdot \sqrt{2}$.

The result $\cos(2\phi)\cdot\sqrt{2}$ is independent of $r$ — the radial gradient has been *exactly* removed and the streamer pattern is now uniformly visible. This toy calculation matches what NRGF achieves on real LASCO data.

64×64 토이 영상의 단계별 NRGF: 위와 같이 $a(r)=e^{-r/8}$ 방사형 감소와 $\cos(2\phi)$ streamer 패턴, 작은 noise를 합성한 영상에 NRGF를 적용하면, 결과는 정확히 $\cos(2\phi)\cdot\sqrt{2}$로 — 반경에 *독립적*인 streamer 패턴만 남는다. 이것이 NRGF의 정수다.

---

## 3. Key Takeaways / 핵심 시사점

1. **The radial gradient is a display problem, not an information problem.** Coronal images contain detail at every height; the issue is that linear stretching can only show one decade at a time. Standardising each annulus removes the display problem in one line of code. — 방사형 기울기는 *표시* 문제이지 *정보* 문제가 아니다. NRGF의 한 줄 표준화가 표시 문제를 해결한다.

2. **Per-annulus z-scoring is the simplest possible RGF.** The NRGF requires no calibration, no model, no fitted profile — just empirical mean and stdev of each ring. — annulus 단위 z-score는 가장 단순한 RGF 구현이다. 보정·모델·프로파일 fit 없이 경험적 통계만 쓴다.

3. **The unpolarized background is time-stable.** This non-obvious empirical fact is what makes high-cadence $tB$ data NRGF-able: subtract a once-built long-term unpolarized background and you have effectively a $pB$-quality image. — 편광되지 않은 배경이 시간 변화에 거의 무관하다는 실증적 사실이 high-cadence $tB$를 $pB$ 수준으로 끌어올려 준다.

4. **Standard tools introduce false structure.** Figure 4 shows the LASCO QL images have artefacts that flip the relative brightness of east/west streamers and create a phantom secondary peak. Image-processing artefacts can corrupt physics inference. — 표준 도구는 가짜 구조를 만든다. LASCO Quick Look은 streamer 동/서 비대칭과 phantom secondary peak를 만든다 — 영상처리 인공물이 물리 해석을 망가뜨린다.

5. **Reference statistics should come from outside the structure of interest.** In the CME example, the mean/stdev are averaged from before and after the CME, not during — the structure being studied must not contaminate its own normalisation reference. — 통계 참조는 관심 구조 *바깥*에서 가져와야 한다. CME 예제에서 통계 평균은 CME 통과 전후에서 산출되며 CME 자체는 자기 정규화 참조에 들어가지 않는다.

6. **Streamers are filament bundles, not solid blocks.** NRGF reframes the very picture of coronal morphology, with implications for magnetic-field topology and the source regions of the slow solar wind. — streamer는 균일 덩어리가 아니라 *필라멘트 다발*이다. 이는 코로나 자기장 위상학과 slow solar wind 발원지 이해에 직접적 함의를 갖는다.

7. **Boundary smoothness across instruments comes for free.** Independent normalisation of each annulus means EIT + MLSO + LASCO C2 composites have no visible boundary even without smoothing or interpolation. — 각 annulus 독립 정규화 덕에 EIT + MLSO + LASCO 합성에서 경계가 자동으로 매끄러워진다.

8. **Simple beats complex when the simple thing is the right thing.** This 8-page paper, with one equation, has been cited thousands of times and become a standard heliophysics tool — a reminder that algorithmic novelty is not always required. — 8 페이지·식 한 줄의 이 논문이 수천 번 인용되어 표준 도구가 되었다 — 알고리즘적 화려함이 항상 필요하지는 않다는 교훈.

---

## 4. Mathematical Summary / 수학적 요약

**The NRGF (Eq. 1 of paper):**

$$
I'(r,\phi) = \frac{I(r,\phi) - \langle I(r)\rangle_\phi}{\sigma(r)_\phi}
$$

with the azimuthal mean and stdev:

$$
\langle I(r)\rangle_\phi = \frac{1}{N_\phi}\sum_{\phi=0}^{2\pi} I(r,\phi),\qquad
\sigma(r)_\phi = \sqrt{\frac{1}{N_\phi}\sum_{\phi=0}^{2\pi}\big(I(r,\phi) - \langle I(r)\rangle_\phi\big)^2}.
$$

**Coordinate transform** (Cartesian $\to$ polar):

$$
r = \sqrt{x^2+y^2},\qquad \phi = \arctan_2(y,x).
$$

**Unpolarized background construction:**

$$
B_{unpol}(r,\phi) = \frac{1}{T}\sum_{t=1}^{T}\big[\,I_{tB}(r,\phi;t) - I_{pB}(r,\phi;t)\,\big]
$$

over a time window of one or several Carrington rotations. The empirical observation $B_{unpol}^{(1997)} \approx B_{unpol}^{(2000)}$ (Fig. 3 left) justifies using a long average.

**Background-subtracted total brightness:**

$$
I_{tB}^{\text{corr}}(r,\phi;t) = I_{tB}(r,\phi;t) - B_{unpol}(r,\phi).
$$

**Combined pipeline (paper-style total brightness processing):**

$$
I'_{tB}(r,\phi;t) = \frac{I_{tB}^{\text{corr}}(r,\phi;t) - \langle I_{tB}^{\text{corr}}(r;t)\rangle_\phi}{\sigma_\phi[\,I_{tB}^{\text{corr}}(r;t)\,]}.
$$

**Worked example.** Consider an idealised 1D radial profile at fixed time $t$: $I(r,\phi) = a(r) + b(r)\cdot s(\phi)$, where $a(r) = a_0 e^{-r/r_0}$ is the steep radial drop and $s(\phi)$ encodes a streamer pattern with unit amplitude. Then $\langle I(r)\rangle_\phi = a(r)$ (since $\langle s\rangle=0$) and $\sigma(r)_\phi = b(r)$. The NRGF gives

$$
I'(r,\phi) = \frac{a(r) + b(r)s(\phi) - a(r)}{b(r)} = s(\phi),
$$

i.e., the radial part is *exactly* removed and what is left is the structure $s(\phi)$ with unit amplitude — independent of how steep $a(r)$ is. This is the analytic toy that explains why NRGF works.

**Smearing not removed** — the NRGF does *not* deconvolve the PSF nor remove exposure-time motion blur (cf. Stenborg & Cobelli 2003 paper #32 for a complementary deblurring tool).

각 변수: $I(r,\phi)$는 $(r,\phi)$ 픽셀의 원시 강도, $\langle I(r)\rangle_\phi$, $\sigma(r)_\phi$는 반경 $r$의 annulus에서 방위각 평균/표준편차, $B_{unpol}$은 (tB-pB)의 장기 시간 평균이다. Worked example: 분리 가능한 모델 $I=a(r)+b(r)s(\phi)$에서 NRGF는 정확히 $s(\phi)$를 복원한다 — 이것이 한 줄 알고리즘이 그토록 잘 작동하는 이유의 분석적 설명이다.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1950  van de Hulst — pB observations of K-corona
                     │
                     ▼
1968  Newkirk & Harvey — hardware Radial Graded Filter
                     │
                     ▼
1995  SOHO/LASCO operations begin (white-light corona at high cadence)
1997  Woo & Habbal — filamentary corona in radio
1999  Guillermier & Koutchmy — multi-exposure HDR coronal compositing
                     │
                     ▼
2003  Stenborg & Cobelli — wavelet packets for coronal structure (#32)
                     │
                     ▼
2006  *Morgan, Habbal & Woo (this paper) — NRGF*
                     │
                     ▼
2014  Morgan & Druckmüller — MGN (Multi-Scale Gaussian Normalisation)
                     │
                     ▼
2020+ STEREO, PSP/WISPR, Solar Orbiter/Metis, Aditya-L1/VELC — NRGF/MGN standard
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #32 Stenborg & Cobelli 2003 | Complementary tool — wavelet packets recover small-scale structure while NRGF removes the radial gradient | Often used together: Stenborg+NRGF for the highest-quality coronal images |
| #34 Yashiro 2004 (CDAW CME catalog) | NRGF is a standard preprocessing step for CME identification and morphology in this catalog | Operational dependence — much of the CME literature post-2006 relies on NRGF-processed images |
| #38 Morgan & Druckmüller 2014 (MGN) | Direct generalisation: MGN replaces the single-scale annulus normalisation with multi-scale Gaussian normalisations | MGN is the modern industrial-strength descendant of this paper |
| #07 Dabov 2007 (BM3D) | Both are denoising/enhancement tools; NRGF is *display* normalisation rather than statistical denoising | Conceptual contrast — NRGF is content-preserving (no smoothing) whereas BM3D smooths |
| #22 Wang 2022 (low-light enhancement) | Both standardise per-region statistics to reveal hidden structure, in solar vs natural-image domains | Conceptual cousin — the per-region z-score idea recurs in deep learning normalisation layers |
| #44 Ma 2022 / #45 Xu 2022 (modern LLIE) | NRGF predates these by 16 years yet the philosophy ("normalise locally, see globally") is identical | Long-arc influence on image normalisation in low-SNR settings |

---

## 7. References / 참고문헌

- Morgan, H., Habbal, S. R., & Woo, R., "The Depiction of Coronal Structure in White Light Images," *Solar Physics* 236, 263-272 (2006). DOI: 10.1007/s11207-006-0113-6
- Brueckner, G. E., et al., "The Large Angle Spectroscopic Coronagraph (LASCO)," *Solar Physics* 162, 357 (1995).
- Espenak, F., "Last Total Solar Eclipse of the Millennium," *ASP Conf. Ser.* 205, 101 (2000).
- Fisher, R. R., et al., "MLSO Coronagraph Description," *Applied Optics* 20, 1094 (1981).
- Guillermier, P., & Koutchmy, S., *Total Eclipses: Science, Observations, Myths and Legends* (Springer, 1999).
- Habbal, S. R., & Woo, R., *ApJ Letters* 549, L253 (2001).
- Newkirk, G. J., & Harvey, J., "Coronal Polarization Brightness," *Solar Physics* 3, 321 (1968).
- Woo, R., & Habbal, S. R., *Geophys. Res. Lett.* 24, 1159 (1997).
- Woo, R., & Habbal, S. R., *ApJ Letters* 474, L139 (1997).
- Morgan, H., & Druckmüller, M., "Multi-scale Gaussian Normalization for Solar Image Processing," *Solar Physics* 289, 2945 (2014). [descendant work]
- van de Hulst, H. C., *Bull. Astron. Inst. Netherlands* 11, 135 (1950).

### Appendix A: Pseudo-Code for NRGF / 부록 A: NRGF 의사 코드

```
# Inputs: image I[Ny, Nx], Sun centre (cx, cy), N_r radial bins
# Output: I_prime, the NRGF-processed image

# 1) Build per-pixel polar coordinates
for each pixel (x, y):
    dx = x - cx
    dy = y - cy
    r[x, y] = sqrt(dx*dx + dy*dy)
    phi[x, y] = atan2(dy, dx)

# 2) Mask occulter, pylons, saturated pixels, hot pixels
mask = build_mask(I)             # 1 for valid, 0 for invalid

# 3) Bin pixels into N_r radial annuli
r_min, r_max = 2.0, 6.0          # in solar radii (LASCO C2 example)
bins = linspace(r_min, r_max, N_r + 1)

# 4) For each annulus, compute mean and stdev of valid pixels
for k in 0 .. N_r-1:
    pixels_in_bin = where((r >= bins[k]) and (r < bins[k+1]) and mask)
    mu[k]    = mean(I[pixels_in_bin])
    sigma[k] = std(I[pixels_in_bin])

# 5) Standardise every pixel
for each pixel (x, y):
    if mask[x, y] = 0: continue
    k = digitize(r[x, y], bins)
    I_prime[x, y] = (I[x, y] - mu[k]) / sigma[k]

# 6) Optional: clip to [-3, 3] for display
I_display = clip(I_prime, -3, 3)
```

The whole algorithm fits in ~20 lines. The accompanying Jupyter notebook contains a faithful Python/NumPy implementation that runs in a few hundred milliseconds on a 256×256 synthetic coronagraph image.

위 의사 코드는 NRGF의 전체이다 — 약 20줄. 동반 Jupyter 노트북에 256×256 합성 코로나그래프 영상에 대한 충실한 Python/NumPy 구현이 들어 있으며, CPU에서 수백 ms 안에 실행된다.

### Appendix B: NRGF as a Differentiable Layer / 부록 B: 미분 가능한 NRGF 레이어

Modern deep learning frameworks (PyTorch, JAX) make it trivial to embed NRGF as a *differentiable preprocessing layer* in a network. Every operation — polar resampling (via `grid_sample`), mean and stdev along an axis, broadcasting subtraction/division — has a gradient. This means a CNN trained for CME segmentation or active region detection can be trained *end-to-end* with NRGF as the first layer, and the network can learn its own corrections to NRGF (e.g., better masking).

Possible architectures:

- **NRGF-then-CNN:** apply NRGF as a fixed first layer, train a CNN on the output. Simple, used in production CME catalogs (CACTUS-NRGF).
- **Learned per-annulus stats:** replace fixed mean/stdev with learned per-radius bias and gain. The NRGF prior is preserved as a special case (bias=$-\mu$, gain=$1/\sigma$).
- **Multi-scale NRGF:** apply NRGF at multiple radial bin sizes (analogous to MGN's multi-scale Gaussian) as parallel input channels.

미분 가능한 NRGF 레이어: 극좌표 resampling, mean/stdev, 빼기/나누기 모두 grad가 있으므로 PyTorch/JAX에서 NRGF를 *미분 가능한 첫 층*으로 삽입 가능. CME 분할이나 활동영역 검출 CNN을 NRGF + CNN 구조로 end-to-end 학습할 수 있다. 가능한 아키텍처: ① 고정 NRGF + CNN (CACTUS-NRGF 스타일), ② 학습 가능한 per-annulus bias/gain (NRGF가 특수 경우), ③ 다중 스케일 NRGF (MGN 스타일).
