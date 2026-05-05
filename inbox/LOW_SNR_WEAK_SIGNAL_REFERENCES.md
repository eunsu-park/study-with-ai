# LOW_SNR_WEAK_SIGNAL_REFERENCES.md — Paper-cited algorithms for the strong-noise / weak-signal regime / 강한 잡음·약한 신호 regime 알고리즘 논문 목록

> External-facing handout extracted from [`REGIME_ANALYSIS.md`](REGIME_ANALYSIS.md) (Tier A–E) and [`ALGORITHMS.md`](ALGORITHMS.md). Lists every algorithm tagged with the `low-snr-weak-signal` regime token in the project's `Domain provenance` column **and** carrying a citable peer-reviewed paper or refereed conference reference. Textbook-only entries (sigma-clipping, multi-frame median rejection, temporal-median frame stacking, weighted-average stacking) are deliberately excluded from the body — they are listed in §"Excluded entries" with a one-line note.
>
> [`REGIME_ANALYSIS.md`](REGIME_ANALYSIS.md) (Tier A–E) 와 [`ALGORITHMS.md`](ALGORITHMS.md) 에서 추출한 **외부 전달용** 핸드아웃. 프로젝트의 `Domain provenance` 컬럼에 `low-snr-weak-signal` regime 토큰이 부여된 알고리즘 중 **인용 가능한 peer-reviewed 논문 또는 refereed 학회 레퍼런스를 가진 항목** 만 모은 것. 교과서 인용만 있는 항목 (sigma-clipping, multi-frame median rejection, temporal-median 프레임 스택, weighted-average stacking) 은 §"Excluded entries" 에 한 줄 메모와 함께 분리.
>
> Last updated / 최종 수정: 2026-05-01 — extracted from project audit dated 2026-04-30 (4 parallel verification agents on 181 algorithm rows; see `ALGORITHMS.md` Changelog).

---

## English

### Purpose / scope

LOLIPOP's *private target data* sits in the **strong noise + weak signal** regime (per-pixel SNR ≲ a few; signal buried under a comparable-or-larger background; no clean ground truth available). Most published natural-image denoising assumes the opposite regime (strong signal + weak Gaussian noise). The `low-snr-weak-signal` token enumerates algorithms that are *designed for*, *robust in*, or *adaptable to* this regime; this file delivers their full citations.

41 algorithms are listed below. They are organised by Tier following [`REGIME_ANALYSIS.md`](REGIME_ANALYSIS.md):

- **Tier A** — designed for the regime (must-try): 12 entries.
- **Tier B** — robust at low SNR (classical baselines): 11 entries.
- **Tier C** — diffusion-based restoration (with physics-aware likelihood): 4 entries.
- **Tier D** — faint-feature signal-enhancement (post-denoise visualisation): 8 entries.
- **Tier E** — low-light DL (paired or unpaired): 7 entries (DiffLL also counts under Tier C; counted once in the unique total of 41).

DOIs are provided where assigned; arXiv identifiers are given as a fallback or supplement when a DOI is not in the project's audited references.

---

### Tier A — Designed for the regime / regime 에 직접 설계 (must-try)

Self-supervised DL (no clean ground truth required), Poisson-mixed classical methods, multi-frame classical denoisers.

#### Self-supervised DL family

1. **Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., & Aila, T. 2018.** "Noise2Noise: Learning image restoration without clean data." *Proc. 35th International Conference on Machine Learning (ICML)*, PMLR 80, 2965–2974. arXiv: [1803.04189](https://arxiv.org/abs/1803.04189). — slug `noise2noise`.
2. **Krull, A., Buchholz, T.-O., & Jug, F. 2019.** "Noise2Void — Learning denoising from single noisy images." *IEEE/CVF CVPR 2019*, 2129–2137. arXiv: [1811.10980](https://arxiv.org/abs/1811.10980). DOI: [10.1109/CVPR.2019.00223](https://doi.org/10.1109/CVPR.2019.00223). — slug `noise2void`.
3. **Batson, J., & Royer, L. 2019.** "Noise2Self: Blind denoising by self-supervision." *Proc. 36th International Conference on Machine Learning (ICML)*, PMLR 97, 524–533. arXiv: [1901.11365](https://arxiv.org/abs/1901.11365). — slug `noise2self`.
4. **Quan, Y., Chen, M., Pang, T., & Ji, H. 2020.** "Self2Self with dropout: Learning self-supervised denoising from single image." *IEEE/CVF CVPR 2020*, 1890–1898. DOI: [10.1109/CVPR42600.2020.00196](https://doi.org/10.1109/CVPR42600.2020.00196). — slug `self2self`.
5. **Huang, T., Li, S., Jia, X., Lu, H., & Liu, J. 2021.** "Neighbor2Neighbor: Self-supervised denoising from single noisy images." *IEEE/CVF CVPR 2021*, 14781–14790. arXiv: [2101.02824](https://arxiv.org/abs/2101.02824). DOI: [10.1109/CVPR46437.2021.01454](https://doi.org/10.1109/CVPR46437.2021.01454). — slug `neighbor2neighbor`.
6. **Wang, Z., Liu, J., Li, G., & Han, H. 2022.** "Blind2Unblind: Self-supervised image denoising with visible blind spots." *IEEE/CVF CVPR 2022*, 2027–2036. arXiv: [2203.06967](https://arxiv.org/abs/2203.06967). DOI: [10.1109/CVPR52688.2022.00207](https://doi.org/10.1109/CVPR52688.2022.00207). — slug `blind2unblind`.
7. **Pang, T., Zheng, H., Quan, Y., & Ji, H. 2021.** "Recorrupted-to-Recorrupted: Unsupervised deep learning for image denoising." *IEEE/CVF CVPR 2021*, 2043–2052. DOI: [10.1109/CVPR46437.2021.00208](https://doi.org/10.1109/CVPR46437.2021.00208). — slug `r2r`.

#### Poisson / Poisson-Gaussian classical

8. **Anscombe, F. J. 1948.** "The transformation of Poisson, binomial and negative-binomial data." *Biometrika* 35(3/4), 246–254. DOI: [10.1093/biomet/35.3-4.246](https://doi.org/10.1093/biomet/35.3-4.246). — slug `anscombe`.
9. **Mäkitalo, M., & Foi, A. 2013.** "Optimal inversion of the generalized Anscombe transformation for Poisson–Gaussian noise." *IEEE Transactions on Image Processing* 22(1), 91–103. DOI: [10.1109/TIP.2012.2202675](https://doi.org/10.1109/TIP.2012.2202675). — slug `gen-anscombe-bm3d` (used in conjunction with BM3D, entry 14).
10. **Luisier, F., Blu, T., & Unser, M. 2011.** "Image denoising in mixed Poisson–Gaussian noise" (PURE-LET). *IEEE Transactions on Image Processing* 20(3), 696–708. DOI: [10.1109/TIP.2010.2073477](https://doi.org/10.1109/TIP.2010.2073477). — slug `pure-let`.
11. **Deledalle, C.-A., Denis, L., & Tupin, F. 2010.** "Poisson NL means: Unsupervised non-local means for Poisson noise." *Proc. IEEE ICIP 2010*, 801–804. DOI: [10.1109/ICIP.2010.5653394](https://doi.org/10.1109/ICIP.2010.5653394). — slug `poisson-nlm`.

#### Multi-frame classical (paper-cited; sigma-clipping / median rejection / median stack / weighted stack are textbook and listed in §"Excluded entries")

12. **Maggioni, M., Boracchi, G., Foi, A., & Egiazarian, K. 2012.** "Video denoising, deblocking, and enhancement through separable 4-D nonlocal spatiotemporal transforms" (V-BM4D). *IEEE Transactions on Image Processing* 21(9), 3952–3966. DOI: [10.1109/TIP.2012.2199324](https://doi.org/10.1109/TIP.2012.2199324). — slug `v-bm4d`.

---

### Tier B — Robust at low SNR / 저-SNR robust (classical baselines)

Patch-based and transform-domain denoisers historically dominant in faint-signal regimes; outlier / cosmic-ray rejection methods with citations.

#### Patch-based

13. **Buades, A., Coll, B., & Morel, J.-M. 2005.** "A non-local algorithm for image denoising." *Proc. IEEE CVPR 2005*, vol. 2, 60–65. DOI: [10.1109/CVPR.2005.38](https://doi.org/10.1109/CVPR.2005.38). — slug `nlm`.
14. **Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. 2007.** "Image denoising by sparse 3-D transform-domain collaborative filtering" (BM3D). *IEEE Transactions on Image Processing* 16(8), 2080–2095. DOI: [10.1109/TIP.2007.901238](https://doi.org/10.1109/TIP.2007.901238). — slug `bm3d`.
15. **Maggioni, M., Katkovnik, V., Egiazarian, K., & Foi, A. 2013.** "Nonlocal transform-domain filter for volumetric data denoising and reconstruction" (BM4D). *IEEE Transactions on Image Processing* 22(1), 119–133. DOI: [10.1109/TIP.2012.2210725](https://doi.org/10.1109/TIP.2012.2210725). — slug `bm4d`.

#### Transform-domain (wavelets / curvelets / contourlets / shearlets)

16. **Donoho, D. L., & Johnstone, I. M. 1994.** "Ideal spatial adaptation by wavelet shrinkage" (VisuShrink). *Biometrika* 81(3), 425–455. DOI: [10.1093/biomet/81.3.425](https://doi.org/10.1093/biomet/81.3.425). — slug `wavelet-visushrink`.
17. **Donoho, D. L., & Johnstone, I. M. 1995.** "Adapting to unknown smoothness via wavelet shrinkage" (SureShrink). *Journal of the American Statistical Association* 90(432), 1200–1224. DOI: [10.1080/01621459.1995.10476626](https://doi.org/10.1080/01621459.1995.10476626). — slug `wavelet-sureshrink`.
18. **Chang, S. G., Yu, B., & Vetterli, M. 2000.** "Adaptive wavelet thresholding for image denoising and compression" (BayesShrink). *IEEE Transactions on Image Processing* 9(9), 1532–1546. DOI: [10.1109/83.862633](https://doi.org/10.1109/83.862633). — slug `wavelet-bayesshrink`.
19. **Candès, E. J., Demanet, L., Donoho, D., & Ying, L. 2006.** "Fast discrete curvelet transforms." *Multiscale Modeling & Simulation* 5(3), 861–899. DOI: [10.1137/05064182X](https://doi.org/10.1137/05064182X). — slug `curvelet-threshold`.
20. **Do, M. N., & Vetterli, M. 2005.** "The contourlet transform: an efficient directional multiresolution image representation." *IEEE Transactions on Image Processing* 14(12), 2091–2106. DOI: [10.1109/TIP.2005.859376](https://doi.org/10.1109/TIP.2005.859376). — slug `contourlet`.
21. **Easley, G., Labate, D., & Lim, W.-Q. 2008.** "Sparse directional image representations using the discrete shearlet transform." *Applied and Computational Harmonic Analysis* 25(1), 25–46. DOI: [10.1016/j.acha.2007.09.003](https://doi.org/10.1016/j.acha.2007.09.003). — slug `shearlet`.

#### Outlier / cosmic-ray rejection (paper-cited)

22. **van Dokkum, P. G. 2001.** "Cosmic-ray rejection by Laplacian edge detection" (L.A.Cosmic). *Publications of the Astronomical Society of the Pacific* 113(789), 1420–1427. DOI: [10.1086/323894](https://doi.org/10.1086/323894). arXiv: [astro-ph/0108003](https://arxiv.org/abs/astro-ph/0108003). — slug `la-cosmic`.
23. **Zhang, K., & Bloom, J. S. 2020.** "deepCR: Cosmic ray rejection with deep learning." *Astrophysical Journal* 889(1), 24. DOI: [10.3847/1538-4357/ab3fa6](https://doi.org/10.3847/1538-4357/ab3fa6). arXiv: [1907.09500](https://arxiv.org/abs/1907.09500). — slug `deepcr`.

---

### Tier C — Diffusion-based restoration / 확산 모형 기반 복원 (physics-aware likelihood 필수)

Pretrained-diffusion priors plugged into inverse-problem solvers. **Caveat:** unconditional generative diffusion can hallucinate prior-consistent structure absent from the data; for science-grade photometry pair with a known forward model and validate via injection-recovery.

24. **Kawar, B., Elad, M., Ermon, S., & Song, J. 2022.** "Denoising diffusion restoration models" (DDRM). *NeurIPS 2022* 35, 23593–23606. arXiv: [2201.11793](https://arxiv.org/abs/2201.11793). — slug `ddrm`.
25. **Chung, H., Kim, J., McCann, M. T., Klasky, M. L., & Ye, J. C. 2023.** "Diffusion posterior sampling for general noisy inverse problems" (DPS). *Proc. ICLR 2023*. arXiv: [2209.14687](https://arxiv.org/abs/2209.14687). — slug `dps`.
26. **Zhu, Y., Zhang, K., Liang, J., Cao, J., Wen, B., Timofte, R., & Van Gool, L. 2023.** "Denoising diffusion models for plug-and-play image restoration" (DiffPIR). *IEEE/CVF CVPR Workshops (NTIRE) 2023*, 1219–1229. arXiv: [2305.08995](https://arxiv.org/abs/2305.08995). — slug `diffpir`.
27. **Jiang, H., Luo, A., Fan, H., Han, S., & Liu, S. 2023.** "Low-light image enhancement with wavelet-based diffusion models" (DiffLL). *ACM Transactions on Graphics* 42(6), Article 238 (SIGGRAPH Asia 2023). DOI: [10.1145/3618373](https://doi.org/10.1145/3618373). arXiv: [2306.00306](https://arxiv.org/abs/2306.00306). — slug `diffll` (also Tier E).

---

### Tier D — Faint-feature signal-enhancement / 약한 신호 강조 (post-denoise)

Visualisation / detection-prep enhancement methods that **amplify both signal and residual noise** — apply *after* a Tier-A or Tier-B denoiser.

28. **Starck, J.-L., Murtagh, F., Candès, E. J., & Donoho, D. L. 2003.** "Gray and color image contrast enhancement by the curvelet transform." *IEEE Transactions on Image Processing* 12(6), 706–717. DOI: [10.1109/TIP.2003.813140](https://doi.org/10.1109/TIP.2003.813140). — slug `curvelet-enhance`. (Cited by both `atrous-wavelet` and `curvelet-enhance` rows in `ALGORITHMS.md`; for the strict à-trous / undecimated wavelet construction, the more canonical citation is entry 29.)
29. **Starck, J.-L., Fadili, J., & Murtagh, F. 2007.** "The undecimated wavelet decomposition and its reconstruction." *IEEE Transactions on Image Processing* 16(2), 297–309. DOI: [10.1109/TIP.2006.887733](https://doi.org/10.1109/TIP.2006.887733). — recommended canonical citation for slug `atrous-wavelet`.
30. **Morgan, H., Habbal, S. R., & Woo, R. 2006.** "The depiction of coronal structure in white-light images" (NRGF). *Solar Physics* 236(2), 263–272. DOI: [10.1007/s11207-006-0113-6](https://doi.org/10.1007/s11207-006-0113-6). — slug `nrgf`.
31. **Morgan, H., & Druckmüller, M. 2014.** "Multi-scale Gaussian normalization for solar image processing" (MGN). *Solar Physics* 289(8), 2945–2955. DOI: [10.1007/s11207-014-0523-9](https://doi.org/10.1007/s11207-014-0523-9). — slug `mgn`.
32. **Stenborg, G., & Cobelli, P. J. 2003.** "A wavelet packets equalization technique to reveal the multiple spatial-scale nature of coronal structures." *Astronomy & Astrophysics* 398(3), 1185–1193. DOI: [10.1051/0004-6361:20021687](https://doi.org/10.1051/0004-6361:20021687). — slug `atrous-coronagraph`.
33. **Patel, R., Majumdar, S., Pant, V., & Banerjee, D. 2022.** "SiRGraF — A simple radial gradient filter for batch-processing of coronagraph images." *Solar Physics* 297, 27. DOI: [10.1007/s11207-022-01957-y](https://doi.org/10.1007/s11207-022-01957-y). — slug `sirgraf`.
34. **Yashiro, S., Gopalswamy, N., Michalek, G., St. Cyr, O. C., Plunkett, S. P., Rich, N. B., & Howard, R. A. 2004.** "A catalog of white light coronal mass ejections observed by the SOHO spacecraft." *Journal of Geophysical Research* 109, A07105. DOI: [10.1029/2003JA010282](https://doi.org/10.1029/2003JA010282). — anchor citation for slug `temporal-median-bg` (CDAW catalog as the closest single-paper anchor; the technique itself is community practice).
35. **Candès, E. J., Li, X., Ma, Y., & Wright, J. 2011.** "Robust Principal Component Analysis?" *Journal of the ACM* 58(3), Article 11, 1–37. DOI: [10.1145/1970392.1970395](https://doi.org/10.1145/1970392.1970395). — slug `rpca`.

---

### Tier E — Low-light DL / 저조도 딥러닝 (paired or unpaired)

Built for low-light *natural-image* enhancement; transfer to scientific data depends on noise-model and signal-statistic overlap. Mostly architectural / loss-function references rather than drop-in solutions.

36. **Lore, K. G., Akintayo, A., & Sarkar, S. 2017.** "LLNet: A deep autoencoder approach to natural low-light image enhancement." *Pattern Recognition* 61, 650–662. DOI: [10.1016/j.patcog.2016.06.008](https://doi.org/10.1016/j.patcog.2016.06.008). — slug `llnet`.
37. **Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., Yang, M.-H., & Shao, L. 2020.** "Learning enriched features for real image restoration and enhancement" (MIRNet). *ECCV 2020*, LNCS 12370, 492–511. DOI: [10.1007/978-3-030-58595-2_30](https://doi.org/10.1007/978-3-030-58595-2_30). — slug `mirnet`.
38. **Ma, L., Ma, T., Liu, R., Fan, X., & Luo, Z. 2022.** "Toward fast, flexible, and robust low-light image enhancement" (SCI). *IEEE/CVF CVPR 2022*, 5637–5646. arXiv: [2204.10137](https://arxiv.org/abs/2204.10137). DOI: [10.1109/CVPR52688.2022.00555](https://doi.org/10.1109/CVPR52688.2022.00555). — slug `sci`.
39. **Xu, X., Wang, R., Fu, C.-W., & Jia, J. 2022.** "SNR-aware low-light image enhancement." *IEEE/CVF CVPR 2022*, 17714–17724. DOI: [10.1109/CVPR52688.2022.01719](https://doi.org/10.1109/CVPR52688.2022.01719). — slug `snr-aware`.
40. **Guo, C., Li, C., Guo, J., Loy, C. C., Hou, J., Kwong, S., & Cong, R. 2020.** "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement" (Zero-DCE). *IEEE/CVF CVPR 2020*, 1780–1789. DOI: [10.1109/CVPR42600.2020.00185](https://doi.org/10.1109/CVPR42600.2020.00185). — slug `zero-dce`.
41. **Jiang, Y., Gong, X., Liu, D., Cheng, Y., Fang, C., et al. 2021.** "EnlightenGAN: Deep light enhancement without paired supervision." *IEEE Transactions on Image Processing* 30, 2340–2349. DOI: [10.1109/TIP.2021.3051462](https://doi.org/10.1109/TIP.2021.3051462). — slug `enlighten-gan`.

(*DiffLL* — Jiang et al. 2023 — appears under Tier C as entry 27 and is also a Tier-E low-light DL method; counted once.)

---

### Excluded entries / 본 목록 제외 항목

These four `low-snr-weak-signal`-tagged algorithm rows have **textbook-only** references in [`ALGORITHMS.md`](ALGORITHMS.md) and are intentionally excluded from the numbered body. They are still relevant in the regime; the standard textbook anchors are listed for transparency.

- **σ-clipping** — slug `sigma-clip` — *Howell, S. B. 2006.* `Handbook of CCD Astronomy` (2nd ed.), Cambridge University Press.
- **Multi-frame median rejection** — slug `multiframe-median-reject` — *Howell, S. B. 2006.* `Handbook of CCD Astronomy` (2nd ed.).
- **Temporal median (frame stack)** — slug `temporal-median` — *Gonzalez, R. C., & Woods, R. E. 2018.* `Digital Image Processing` (4th ed.), Pearson.
- **Weighted-average stacking** — slug `weighted-stack` — *Howell, S. B. 2006.* `Handbook of CCD Astronomy` (2nd ed.).

---

## 한국어

### 본 문서의 목적·범위

LOLIPOP 의 *비공개 타겟 데이터* 는 **강한 잡음 + 약한 신호** regime 에 위치 (per-pixel SNR ≲ 수 단위, 신호가 비슷하거나 큰 배경에 묻혀 있음, clean ground truth 부재). 대부분의 publish 된 자연 이미지 denoising 은 정반대 regime (강한 신호 + 약한 Gaussian 잡음) 을 가정. 본 regime 에 *설계되었거나 / robust 하거나 / 적응 가능한* 알고리즘에 `low-snr-weak-signal` 토큰이 부여되어 있고, 본 파일은 그 항목들의 완전한 인용을 제공.

총 41 개 알고리즘이 본 문서에 등재. [`REGIME_ANALYSIS.md`](REGIME_ANALYSIS.md) 의 Tier 분류를 따름:

- **Tier A** — regime 에 직접 설계 (must-try): 12 항목
- **Tier B** — 저-SNR robust (classical baseline): 11 항목
- **Tier C** — 확산 모형 기반 복원 (physics-aware likelihood 필수): 4 항목
- **Tier D** — 약한 신호 강조 (post-denoise): 8 항목
- **Tier E** — 저조도 딥러닝 (paired / unpaired): 7 항목 (DiffLL 은 Tier C 와 중복; unique 합계 41 에서는 1 회 카운트)

DOI 는 발급된 경우 표기, 미발급 또는 프로젝트 audit references 에 미기록인 경우 arXiv ID 를 fallback / 보조로 제공.

### 영문 섹션 표가 정본

위 영문 섹션의 41 개 entry 에 모든 서지 정보 + DOI 가 그대로 기재되어 있습니다. 한국어 섹션은 사용 안내만 담고 있고, 영문의 각 entry 를 그대로 인용하시면 됩니다.

### 외부 전달 시 참고 사항

- 본 파일은 [`REGIME_ANALYSIS.md`](REGIME_ANALYSIS.md) (regime 정의 + Tier 카드 + 미션 cross-reference + 미해결 질문) 에서 파생된 외부 핸드아웃입니다. 알고리즘 선택 의사결정의 *근거* 는 `REGIME_ANALYSIS.md` 본문에, *완전한 catalog 표* 는 [`ALGORITHMS.md`](ALGORITHMS.md) 에 있습니다.
- DOI 가 표기된 항목은 출판사 페이지 / 학회 proceedings / 저널 페이지로 직접 연결됩니다. arXiv ID 는 preprint 접근용입니다.
- 교과서로만 인용된 4 항목 (sigma-clipping, multi-frame median rejection, temporal-median frame stack, weighted-average stacking) 은 영문 섹션 §"Excluded entries" 에 분리.
- 본 문서의 인용은 모두 프로젝트 내부 audit (2026-04-30, 4 개 verification agent 병렬, 181 행 검증) 후 [`ALGORITHMS.md`](ALGORITHMS.md) §References 에 등록된 항목에서 추출. 해당 audit 에 미포함이었던 일부 학회 paper DOI (CVPR 시리즈) 는 IEEE Xplore proceedings 의 표준 DOI 를 추가 표기.

---

## References / 참고문헌

> Per CLAUDE.md §5, every documentation file must close with a dedicated References section. The body above *is* the references list (41 entries + 4 textbook anchors), so this section records only the **synthesis sources** that informed the regime characterization itself, not the per-algorithm citations already given.
>
> CLAUDE.md §5 에 따라 모든 문서 파일은 별도 References 섹션을 닫는 위치에 두어야 합니다. 본 파일의 본문 자체가 references 목록 (41 항목 + 교과서 앵커 4) 이므로, 본 섹션에서는 regime 정의 자체에 영향을 준 **synthesis 출처** 만 기록합니다 (개별 알고리즘 인용은 위에 이미 있음).

### Synthesis sources for the regime characterization / regime 정의 종합 출처

- **Buchholz, T.-O., Jordan, M., Pigino, G., Jug, F. 2019.** "Cryo-CARE: Content-aware image restoration for cryo-electron tomography." *Proc. IEEE ISBI 2019*, 502–506. DOI: [10.1109/ISBI.2019.8759519](https://doi.org/10.1109/ISBI.2019.8759519). arXiv: [1810.05420](https://arxiv.org/abs/1810.05420). **(Cryo-EM noise-regime origin reference for the Noise2Noise lineage.)**
- **Daras, G., Shah, K., Dagan, Y., Gollakota, A., Klivans, A., Dimakis, A. G. 2023.** "Ambient diffusion: Learning clean distributions from corrupted data." *NeurIPS 2023*. arXiv: [2305.19256](https://arxiv.org/abs/2305.19256). **(Diffusion training without clean ground truth.)**
- **Kadkhodaie, Z., Simoncelli, E. P. 2021.** "Stochastic solutions for linear inverse problems using the prior implicit in a denoiser." *NeurIPS 2021* 34. arXiv: [2007.13640](https://arxiv.org/abs/2007.13640). **(Score-based posterior sampling theory.)**
- **Bansal, A., Borgnia, E., Chu, H.-M., Li, J. S., Kazemi, H., Huang, F., Goldblum, M., Geiping, J., Goldstein, T. 2023.** "Cold diffusion: Inverting arbitrary image transforms without noise." *NeurIPS 2023*. arXiv: [2208.09392](https://arxiv.org/abs/2208.09392). **(Generalised diffusion to non-Gaussian degradation.)**

### Linked project artifacts / 연결 프로젝트 문서

- [`REGIME_ANALYSIS.md`](REGIME_ANALYSIS.md) — regime 정의, Tier A–E 카드, 미션 cross-reference, 미해결 질문.
- [`ALGORITHMS.md`](ALGORITHMS.md) — 알고리즘 마스터 인덱스 + 완전한 References 섹션 (45 개 `low-snr-weak-signal`-tagged 행 + 그 외 ~140 행).
- [`../MISSIONS.md`](../MISSIONS.md) — anchor 미션 (SSL pretraining 용).
- [`../MISSIONS_CANDIDATES.md`](../MISSIONS_CANDIDATES.md) — FUV 오로라 미션 (domain-adaptive finetuning 용).

---

## Changelog / 변경 기록

### 2026-05-01 — Initial extraction

Created in response to a request for an external-facing handout listing the paper-cited algorithms tagged `low-snr-weak-signal` in [`ALGORITHMS.md`](ALGORITHMS.md). 41 algorithms listed in the body across Tiers A–E (DiffLL counted once across Tier C and Tier E); 4 textbook-only entries (`sigma-clip`, `multiframe-median-reject`, `temporal-median`, `weighted-stack`) moved to §"Excluded entries". Citations sourced from the [`ALGORITHMS.md`](ALGORITHMS.md) §References section as audited on 2026-04-30; CVPR proceedings DOIs that were not in the audited references (Self2Self, Neighbor2Neighbor, Blind2Unblind, R2R, SCI, SNR-aware, Zero-DCE conference DOIs) added from the standard IEEE Xplore proceedings page. The handout intentionally trims [`REGIME_ANALYSIS.md`](REGIME_ANALYSIS.md)'s decision-context (open user-side questions, suggested transfer-learning recipe) so that the file delivers only the bibliography.

본 파일은 [`ALGORITHMS.md`](ALGORITHMS.md) 의 `low-snr-weak-signal` 토큰 부여 행 중 paper-cited 항목을 외부 전달용으로 추출한 핸드아웃. Tier A–E 에 걸쳐 41 개 알고리즘 본문 등재 (DiffLL 은 Tier C / E 양쪽에 등재되나 unique 카운트는 1); 교과서만 인용된 4 항목 (`sigma-clip`, `multiframe-median-reject`, `temporal-median`, `weighted-stack`) 은 §"Excluded entries" 로 분리. 인용은 [`ALGORITHMS.md`](ALGORITHMS.md) §References (2026-04-30 audit) 에서 추출하였으며, audit 미포함이었던 일부 CVPR proceedings DOI (Self2Self, Neighbor2Neighbor, Blind2Unblind, R2R, SCI, SNR-aware, Zero-DCE 등) 는 IEEE Xplore proceedings 표준 DOI 로 보충. [`REGIME_ANALYSIS.md`](REGIME_ANALYSIS.md) 의 의사결정 맥락 (사용자측 미해결 질문, 권장 transfer-learning recipe) 은 본 핸드아웃에서 의도적으로 생략 — 본 파일은 서지 정보만 전달.
