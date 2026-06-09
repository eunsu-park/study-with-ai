# Low-SNR Imaging Paper Reading List / 저신호잡음비 영상 논문 읽기 목록

A curated list of landmark papers on **image restoration and faint-signal enhancement in the low signal-to-noise regime**. Initial 41-paper extraction from the LOLIPOP project's `LOW_SNR_WEAK_SIGNAL_REFERENCES.md` (audited 2026-04-30), augmented with 4 synthesis sources for theoretical foundations. Organized by methodological cluster (corresponding to LOLIPOP Tier A–E with reordering for didactic flow), chronologically within each cluster.

저신호잡음비 regime 에서의 **영상 복원과 약한 신호 강조**에 관한 주요 논문을 정리. 초기 41편은 LOLIPOP 프로젝트의 `LOW_SNR_WEAK_SIGNAL_REFERENCES.md` (2026-04-30 audit) 에서 추출하고, 이론적 기반을 위해 synthesis source 4편을 추가. 방법론적 군집 (LOLIPOP Tier A–E 와 대응, 학습 흐름에 맞게 재배열) 으로 묶고 군집 내 시대순 정렬.

---

## Phase 1: Classical Patch-Based and Transform-Domain Denoising / 고전 패치·변환 영역 denoising (1994–2013)

### 1. Ideal Spatial Adaptation by Wavelet Shrinkage (VisuShrink)
- **Authors**: David L. Donoho, Iain M. Johnstone
- **Year**: 1994
- **Journal**: *Biometrika*, Vol. 81, No. 3, pp. 425–455
- **DOI**: 10.1093/biomet/81.3.425
- **Why it matters**: Wavelet 영역에서 universal threshold ($\sigma\sqrt{2\log N}$) 로 잡음 계수를 영(0)으로 보내는 **wavelet shrinkage 의 효시**. 모든 후속 wavelet/curvelet/shearlet shrinkage 의 출발점. / The **founding paper of wavelet shrinkage denoising** — sets noise coefficients to zero via a universal threshold $\sigma\sqrt{2\log N}$ in the wavelet domain. Starting point for all subsequent wavelet/curvelet/shearlet shrinkage methods.
- **Prerequisites**: Discrete wavelet transform basics, Gaussian noise model / 이산 웨이블릿 변환 기초, Gaussian 잡음 모델
- **Status**: [x]

### 2. Adapting to Unknown Smoothness via Wavelet Shrinkage (SureShrink)
- **Authors**: David L. Donoho, Iain M. Johnstone
- **Year**: 1995
- **Journal**: *Journal of the American Statistical Association*, Vol. 90, No. 432, pp. 1200–1224
- **DOI**: 10.1080/01621459.1995.10476626
- **Why it matters**: VisuShrink 의 한계(과도한 평탄화)를 **Stein's Unbiased Risk Estimate(SURE)** 로 적응적 임계 선택으로 극복. 부드러움 정도가 미지일 때도 minimax 수준의 위험률 보장. / Adapts VisuShrink to **Stein's Unbiased Risk Estimate (SURE)** for sub-band-wise threshold selection, achieving minimax-near risk even when smoothness is unknown.
- **Prerequisites**: Paper #1; Stein's lemma, unbiased risk estimation / 논문 #1; Stein 정리, unbiased risk 추정
- **Status**: [x]

### 3. Adaptive Wavelet Thresholding for Image Denoising and Compression (BayesShrink)
- **Authors**: S. Grace Chang, Bin Yu, Martin Vetterli
- **Year**: 2000
- **Journal**: *IEEE Transactions on Image Processing*, Vol. 9, No. 9, pp. 1532–1546
- **DOI**: 10.1109/83.862633
- **Why it matters**: **베이지안 prior(Generalized Gaussian)** 로 sub-band 별 임계를 closed-form 으로 도출. 자연 영상 통계에 적합한 sub-band-adaptive 임계로 SureShrink 보다 더 자연스러운 결과. / Derives sub-band thresholds in **closed form under a Bayesian generalized-Gaussian prior**. Sub-band-adaptive thresholds match natural-image statistics, improving over SureShrink for natural scenes.
- **Prerequisites**: Papers #1, #2; generalized Gaussian distribution, Bayesian inference / 논문 #1, #2; 일반화 가우시안 분포, 베이지안 추론
- **Status**: [x]

### 4. A Non-Local Algorithm for Image Denoising (NLM)
- **Authors**: Antoni Buades, Bartomeu Coll, Jean-Michel Morel
- **Year**: 2005
- **Journal**: *Proc. IEEE CVPR 2005*, Vol. 2, pp. 60–65
- **DOI**: 10.1109/CVPR.2005.38
- **Why it matters**: 영상의 **자기 유사성(self-similarity)** 을 활용하는 첫 본격 알고리즘 — 비국소 패치 평균. 패치 기반 denoising 의 패러다임을 열었고, 이후 BM3D, NLM-Poisson 등의 직접 토대. / **First major algorithm to exploit image self-similarity** via non-local patch averaging. Opens the patch-based denoising paradigm; direct precursor to BM3D and Poisson NLM.
- **Prerequisites**: Patch representation, kernel-weighted averaging / 패치 표현, 커널 가중 평균
- **Status**: [x]

### 5. The Contourlet Transform: An Efficient Directional Multiresolution Image Representation
- **Authors**: Minh N. Do, Martin Vetterli
- **Year**: 2005
- **Journal**: *IEEE Transactions on Image Processing*, Vol. 14, No. 12, pp. 2091–2106
- **DOI**: 10.1109/TIP.2005.859376
- **Why it matters**: Wavelet 의 isotropic 한계를 보완하는 **방향성 다중해상도 표현**. Laplacian pyramid + directional filter bank 로 곡선·엣지를 sparse 하게 표현하여 약한 엣지 신호의 임계 처리에 강점. / Provides **directional multiresolution representation** complementing isotropic wavelets. Laplacian pyramid + directional filter bank yields sparse representation of curves/edges — useful for thresholding faint edge signals.
- **Prerequisites**: Paper #1; Laplacian pyramid, directional filter banks / 논문 #1; Laplacian 피라미드, 방향성 필터뱅크
- **Status**: [x]

### 6. Fast Discrete Curvelet Transforms
- **Authors**: Emmanuel J. Candès, Laurent Demanet, David L. Donoho, Lexing Ying
- **Year**: 2006
- **Journal**: *Multiscale Modeling & Simulation*, Vol. 5, No. 3, pp. 861–899
- **DOI**: 10.1137/05064182X
- **Why it matters**: 곡선·**curvilinear discontinuity** 의 sparse 표현을 위한 anisotropic scaling (parabolic) wavelet. Wrapping/USFFT 두 가지 fast 구현으로 wavelet 보다 우수한 곡선 신호 임계처리 성능. 코로나 ray 등 곡선 구조 처리에 사용. / Anisotropic (parabolic) scaling for **sparse representation of curvilinear discontinuities**. Two fast implementations (wrapping, USFFT) outperform wavelets on curve features — used for coronal-ray-like structures.
- **Prerequisites**: Paper #1; anisotropic scaling, parabolic frame / 논문 #1; 비등방 스케일링, parabolic frame
- **Status**: [x]

### 7. Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering (BM3D)
- **Authors**: Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, Karen Egiazarian
- **Year**: 2007
- **Journal**: *IEEE Transactions on Image Processing*, Vol. 16, No. 8, pp. 2080–2095
- **DOI**: 10.1109/TIP.2007.901238
- **Why it matters**: NLM 의 patch averaging 을 **3D 변환 영역 협력 필터링**으로 발전 — 유사 패치들을 쌓아 3D 변환·hard threshold·역변환·aggregation. **15년 이상 자연 영상 denoising SOTA** 의 자리를 지킨 고전. 2단계(hard threshold + Wiener)로 PSNR 한계에 근접. / Extends NLM patch averaging via **3D transform-domain collaborative filtering** — stacks similar patches, 3D transform, hard-threshold, inverse, aggregation. **Held SOTA for >15 years** on natural-image denoising; two-step (hard + Wiener) reaches near-PSNR-bound.
- **Prerequisites**: Paper #4; 3D DCT/wavelet, Wiener filtering / 논문 #4; 3D DCT/웨이블릿, Wiener 필터
- **Status**: [x]

### 8. Sparse Directional Image Representations using the Discrete Shearlet Transform
- **Authors**: Glenn Easley, Demetrio Labate, Wang-Q Lim
- **Year**: 2008
- **Journal**: *Applied and Computational Harmonic Analysis*, Vol. 25, No. 1, pp. 25–46
- **DOI**: 10.1016/j.acha.2007.09.003
- **Why it matters**: Curvelet 의 affine group 기반 변형 — **shear matrix 로 모든 방향을 cone 별로 균등 sampling**. 디지털 격자에 친화적이고 fast 구현이 깔끔. 약한 방향성 구조 임계처리에 사용. / An affine-group variant of curvelets — uses **shear matrices to sample all orientations uniformly per cone**. Lattice-friendly with clean fast implementation; used for thresholding faint directional structures.
- **Prerequisites**: Papers #1, #6; affine group representation theory / 논문 #1, #6; affine group 표현론
- **Status**: [x]

### 9. Video Denoising, Deblocking, and Enhancement Through Separable 4-D Nonlocal Spatiotemporal Transforms (V-BM4D)
- **Authors**: Matteo Maggioni, Giacomo Boracchi, Alessandro Foi, Karen Egiazarian
- **Year**: 2012
- **Journal**: *IEEE Transactions on Image Processing*, Vol. 21, No. 9, pp. 3952–3966
- **DOI**: 10.1109/TIP.2012.2199324
- **Why it matters**: BM3D 의 **시공간 비국소 확장** — motion-compensated patch trajectory + 4D 변환. 비디오·시계열 영상 (코로나그래프 동영상, 시계열 미세영상) 에 직접 적용 가능한 다중 프레임 denoiser. / **Spatiotemporal non-local extension of BM3D** — motion-compensated patch trajectories + 4D transform. Directly applicable multi-frame denoiser for video / time-series imaging (coronagraph movies, microscopy time-lapse).
- **Prerequisites**: Paper #7; optical flow / motion estimation, 4D separable transforms / 논문 #7; 광류/모션 추정, 4D 분리형 변환
- **Status**: [x]

### 10. Nonlocal Transform-Domain Filter for Volumetric Data Denoising and Reconstruction (BM4D)
- **Authors**: Matteo Maggioni, Vladimir Katkovnik, Karen Egiazarian, Alessandro Foi
- **Year**: 2013
- **Journal**: *IEEE Transactions on Image Processing*, Vol. 22, No. 1, pp. 119–133
- **DOI**: 10.1109/TIP.2012.2210725
- **Why it matters**: BM3D 의 **3D 볼륨 일반화** (예: 의료 CT, cryo-EM tomogram, 분광 데이터큐브). 큐브 패치 + 4D 변환 협력 필터링으로 볼륨 데이터의 SOTA 클래식 baseline. / **3D volumetric generalization of BM3D** (medical CT, cryo-EM tomograms, spectral data cubes). Cube patches + 4D collaborative filtering — SOTA classical baseline for volumetric data.
- **Prerequisites**: Paper #7; volumetric data structure / 논문 #7; 볼륨 데이터 구조
- **Status**: [x]

---

## Phase 2: Photon-Statistics-Aware Denoising / 광자 통계 기반 denoising (1948–2013)

### 11. The Transformation of Poisson, Binomial and Negative-Binomial Data (Anscombe Transform)
- **Authors**: Frank J. Anscombe
- **Year**: 1948
- **Journal**: *Biometrika*, Vol. 35, No. 3/4, pp. 246–254
- **DOI**: 10.1093/biomet/35.3-4.246
- **Why it matters**: $f(x) = 2\sqrt{x + 3/8}$ 변환으로 **Poisson 데이터를 근사적으로 분산 1 의 정규분포로 안정화**. 78년 후에도 광자-제한 영상의 표준 전처리: 변환→Gaussian denoiser→역변환. / The transform $f(x) = 2\sqrt{x + 3/8}$ **stabilizes Poisson data to approximately unit-variance Gaussian**. After 78 years still the standard pretreatment for photon-limited imaging: transform → Gaussian denoiser → inverse.
- **Prerequisites**: Poisson distribution, variance-stabilizing transformations / Poisson 분포, 분산 안정화 변환
- **Status**: [x]

### 12. Poisson NL Means: Unsupervised Non-local Means for Poisson Noise
- **Authors**: Charles-Alban Deledalle, Florence Tupin, Loïc Denis
- **Year**: 2010
- **Journal**: *Proc. IEEE ICIP 2010*, pp. 801–804
- **DOI**: 10.1109/ICIP.2010.5653394
- **Why it matters**: NLM 의 patch 유사성 측도를 **Generalized Likelihood Ratio (GLR) for Poisson** 로 교체 — Anscombe + Gaussian NLM 보다 광자 부족 영역에서 더 정확. SAR·X-ray 영상에 사용. / Replaces NLM's patch similarity with the **Generalized Likelihood Ratio (GLR) for Poisson** — more accurate than Anscombe + Gaussian NLM in photon-starved regions. Used for SAR / X-ray imaging.
- **Prerequisites**: Papers #4, #11; likelihood ratio test / 논문 #4, #11; 우도비 검정
- **Status**: [x]

### 13. Image Denoising in Mixed Poisson–Gaussian Noise (PURE-LET)
- **Authors**: Florian Luisier, Thierry Blu, Michael Unser
- **Year**: 2011
- **Journal**: *IEEE Transactions on Image Processing*, Vol. 20, No. 3, pp. 696–708
- **DOI**: 10.1109/TIP.2010.2073477
- **Why it matters**: **Poisson + Gaussian 혼합 잡음의 unbiased risk estimator(PURE)** 를 도출하고, linear expansion of thresholds(LET) 기반 폐형 솔버 제공. 실제 카메라 잡음 모델(샷+읽기)에 직접 부합. / Derives the **Poisson–Gaussian unbiased risk estimator (PURE)** and provides a closed-form solver via linear expansion of thresholds (LET). Matches realistic sensor noise models (shot + read).
- **Prerequisites**: Papers #2, #11; risk-unbiased estimation, linear expansions / 논문 #2, #11; risk-unbiased 추정, 선형 확장
- **Status**: [x]

### 14. Optimal Inversion of the Generalized Anscombe Transformation for Poisson–Gaussian Noise
- **Authors**: Markku Mäkitalo, Alessandro Foi
- **Year**: 2013
- **Journal**: *IEEE Transactions on Image Processing*, Vol. 22, No. 1, pp. 91–103
- **DOI**: 10.1109/TIP.2012.2202675
- **Why it matters**: **Generalized Anscombe(Poisson+Gaussian)** 변환 후 BM3D 와 같은 Gaussian denoiser 를 적용한 뒤 정확한 역변환 — 단순 algebraic 역변환이 아닌 conditional expectation 기반 최적 역변환. **Anscombe-BM3D 파이프라인의 마지막 퍼즐**. / Provides **optimal (conditional-expectation-based) inverse for the generalized Anscombe (Poisson + Gaussian) transform** after Gaussian denoising — not simple algebraic inversion. The **final piece of the Anscombe-BM3D pipeline**.
- **Prerequisites**: Papers #7, #11, #13; conditional expectation, MMSE estimator / 논문 #7, #11, #13; 조건부 기댓값, MMSE 추정기
- **Status**: [x]

---

## Phase 3: Self-Supervised Deep Learning Denoising / 자기지도 딥러닝 denoising (2018–2022)

### 15. Cryo-CARE: Content-Aware Image Restoration for Cryo-Electron Tomography
- **Authors**: Tim-Oliver Buchholz, Mareike Jordan, Gaia Pigino, Florian Jug
- **Year**: 2019
- **Journal**: *Proc. IEEE ISBI 2019*, pp. 502–506
- **DOI**: 10.1109/ISBI.2019.8759519
- **Why it matters**: **Cryo-EM regime 에서 Noise2Noise 적용의 효시 case study** — 두 개의 독립 노출에서 학습. 본 reading list 의 regime 정의(강한 잡음 + 약한 신호 + clean GT 부재) 가 그대로 cryo-EM 에서 작동함을 입증한 논문. / **Origin case study of Noise2Noise applied to the cryo-EM regime** — trained on two independent exposures. Demonstrates that the strong-noise/weak-signal regime extends to cryo-EM imaging.
- **Prerequisites**: Cryo-EM tomography basics, U-Net architecture / Cryo-EM 단층 촬영 기초, U-Net 구조
- **Status**: [x]

### 16. Noise2Noise: Learning Image Restoration without Clean Data
- **Authors**: Jaakko Lehtinen, Jacob Munkberg, Jon Hasselgren, Samuli Laine, Tero Karras, Miika Aittala, Timo Aila
- **Year**: 2018
- **Journal**: *Proc. 35th ICML*, PMLR 80, pp. 2965–2974 (arXiv: 1803.04189)
- **Why it matters**: **Clean ground truth 없이도 잡음 영상 쌍만으로 denoiser 를 학습**할 수 있음을 충격적으로 입증 — L2 loss 의 평균 성질을 이용. 본 reading list regime 의 직접 동기. 이후 모든 self-supervised denoising 의 시발점. / Stunningly demonstrates that **denoisers can be trained from noisy-noisy pairs alone, without clean ground truth** — exploits the mean property of L2 loss. The direct motivation for this reading list's regime; starting point of all self-supervised denoising.
- **Prerequisites**: U-Net / CNN regression, L2 vs L1 loss properties / U-Net/CNN 회귀, L2 vs L1 손실 성질
- **Status**: [x]

### 17. Noise2Void — Learning Denoising from Single Noisy Images
- **Authors**: Alexander Krull, Tim-Oliver Buchholz, Florian Jug
- **Year**: 2019
- **Journal**: *IEEE/CVF CVPR 2019*, pp. 2129–2137
- **DOI**: 10.1109/CVPR.2019.00223
- **Why it matters**: Noise2Noise 의 **단일 영상 확장** — blind-spot network 구조로 입력 픽셀이 자기 자신을 보지 못하게 함으로써 동일 이미지 내에서 학습 가능. 잡음 쌍을 모을 수 없는 경우의 표준. / **Single-image extension of Noise2Noise** — blind-spot network architecture prevents input pixels from seeing themselves, enabling training within a single image. Standard when noisy pairs cannot be acquired.
- **Prerequisites**: Paper #16; blind-spot network design / 논문 #16; blind-spot 네트워크 설계
- **Status**: [x]

### 18. Noise2Self: Blind Denoising by Self-Supervision
- **Authors**: Joshua Batson, Loïc Royer
- **Year**: 2019
- **Journal**: *Proc. 36th ICML*, PMLR 97, pp. 524–533 (arXiv: 1901.11365)
- **Why it matters**: Noise2Void 와 독립적으로 **J-invariant function 이론**으로 self-supervised denoising 을 정형화. 잡음 모델을 가정하지 않는 일반적 프레임워크. / Independently formalizes self-supervised denoising via **J-invariant function theory**. General framework not assuming a specific noise model.
- **Prerequisites**: Paper #16; J-invariance, conditional independence / 논문 #16; J-invariance, 조건부 독립
- **Status**: [x]

### 19. Self2Self with Dropout: Learning Self-Supervised Denoising from Single Image
- **Authors**: Yuhui Quan, Mingqin Chen, Tongyao Pang, Hui Ji
- **Year**: 2020
- **Journal**: *IEEE/CVF CVPR 2020*, pp. 1890–1898
- **DOI**: 10.1109/CVPR42600.2020.00196
- **Why it matters**: 단일 영상 자기지도를 **Bernoulli dropout + 다수 추론 ensemble** 로 구현. Noise2Void 의 blind-spot 제약 없이 더 깊은 네트워크 사용 가능. / Implements single-image self-supervision via **Bernoulli dropout + many-inference ensemble**. Avoids Noise2Void's blind-spot constraint, allowing deeper networks.
- **Prerequisites**: Papers #17, #18; dropout regularization, MC ensemble / 논문 #17, #18; dropout 정규화, MC 앙상블
- **Status**: [x]

### 20. Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images
- **Authors**: Tao Huang, Songjiang Li, Xu Jia, Huchuan Lu, Jianzhuang Liu
- **Year**: 2021
- **Journal**: *IEEE/CVF CVPR 2021*, pp. 14781–14790
- **DOI**: 10.1109/CVPR46437.2021.01454
- **Why it matters**: 입력에서 인접 픽셀 두 개를 **sub-sample 하여 'pseudo-pair'** 를 만들고 Noise2Noise 손실로 학습. blind-spot 없이도 단일 영상 학습 가능, 더 빠르고 깔끔한 결과. / Sub-samples adjacent pixel pairs to create **'pseudo-noisy pairs'** trained with Noise2Noise loss. Achieves single-image training without blind spots — faster and cleaner.
- **Prerequisites**: Papers #16, #17; sub-sampling strategies / 논문 #16, #17; 서브샘플링 전략
- **Status**: [x]

### 21. Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising (R2R)
- **Authors**: Tongyao Pang, Huan Zheng, Yuhui Quan, Hui Ji
- **Year**: 2021
- **Journal**: *IEEE/CVF CVPR 2021*, pp. 2043–2052
- **DOI**: 10.1109/CVPR46437.2021.00208
- **Why it matters**: 단일 잡음 영상에 **추가 잡음을 두 가지 다르게 주입** 해 한 쌍을 인공적으로 만들고 Noise2Noise 학습. 알려진 잡음 모델만 있으면 작동, 이론적 보장 명확. / Injects **two different additional noise realizations** into a single noisy image to synthetically create a pair, then applies Noise2Noise training. Works given a known noise model with clear theoretical guarantees.
- **Prerequisites**: Papers #16, #20; noise model parameterization / 논문 #16, #20; 잡음 모델 파라미터화
- **Status**: [x]

### 22. Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots
- **Authors**: Zejin Wang, Jiazheng Liu, Guoqing Li, Hua Han
- **Year**: 2022
- **Journal**: *IEEE/CVF CVPR 2022*, pp. 2027–2036
- **DOI**: 10.1109/CVPR52688.2022.00207
- **Why it matters**: blind-spot 의 정보 손실 문제(blind spot 자체에서 신호도 잃음) 를 **visible blind spot + global perception network** 로 해결. Noise2Void 계열 SOTA. / Solves the information-loss problem of blind spots (signal also lost at the blind spot) via **visible blind spots + global perception network**. SOTA in the Noise2Void family.
- **Prerequisites**: Papers #17, #20; receptive field design / 논문 #17, #20; 수용 영역 설계
- **Status**: [x]

---

## Phase 4: Outlier and Cosmic-Ray Rejection / 이상치 및 cosmic-ray 제거 (2001–2020)

### 23. Cosmic-Ray Rejection by Laplacian Edge Detection (L.A.Cosmic)
- **Authors**: Pieter G. van Dokkum
- **Year**: 2001
- **Journal**: *Publications of the Astronomical Society of the Pacific*, Vol. 113, No. 789, pp. 1420–1427
- **DOI**: 10.1086/323894
- **Why it matters**: **CCD 영상의 cosmic-ray 제거 표준** — Laplacian 컨볼루션으로 sharp edge 의 우주선 흔적을 검출, 별·은하 같은 PSF-convolved 신호와 분리. 천문 관측 파이프라인의 사실상 표준. / **Standard cosmic-ray rejection for CCD images** — uses Laplacian convolution to detect sharp-edged cosmic-ray traces and separate them from PSF-convolved astrophysical sources. De-facto standard in astronomical pipelines.
- **Prerequisites**: Discrete Laplacian operator, PSF convolution / 이산 Laplacian 연산자, PSF 컨볼루션
- **Status**: [x]

### 24. deepCR: Cosmic Ray Rejection with Deep Learning
- **Authors**: Keming Zhang, Joshua S. Bloom
- **Year**: 2020
- **Journal**: *The Astrophysical Journal*, Vol. 889, No. 1, 24
- **DOI**: 10.3847/1538-4357/ab3fa6
- **Why it matters**: **U-Net 기반 cosmic-ray segmentation** — L.A.Cosmic 보다 정확하고 빠르며 underdense field 에서도 안정. 모델 가중치 공개로 즉시 사용 가능. / **U-Net-based cosmic-ray segmentation** — more accurate and faster than L.A.Cosmic, robust in underdense fields. Public model weights enable drop-in use.
- **Prerequisites**: Paper #23; U-Net for segmentation, dataset augmentation / 논문 #23; segmentation U-Net, 데이터셋 증강
- **Status**: [x]

---

## Phase 5: Diffusion-Based Inverse-Problem Restoration / 확산 모형 기반 역문제 복원 (2021–2023)

### 25. Stochastic Solutions for Linear Inverse Problems Using the Prior Implicit in a Denoiser
- **Authors**: Zahra Kadkhodaie, Eero P. Simoncelli
- **Year**: 2021
- **Journal**: *NeurIPS 2021*, Vol. 34 (arXiv: 2007.13640)
- **Why it matters**: **사전훈련된 denoiser 의 score(∇log p) 가 자연 영상의 prior 를 암묵적으로 인코딩**한다는 핵심 관찰. score-based posterior sampling 의 이론적 토대 — 본 phase 의 모든 후속 알고리즘이 이 원리를 활용. / Foundational observation that **a pretrained denoiser's score (∇log p) implicitly encodes a natural-image prior**. Theoretical basis for score-based posterior sampling — every subsequent algorithm in this phase exploits this principle.
- **Prerequisites**: Score matching, Langevin dynamics, Tweedie's formula / 스코어 매칭, Langevin 동역학, Tweedie 공식
- **Status**: [x]

### 26. Denoising Diffusion Restoration Models (DDRM)
- **Authors**: Bahjat Kawar, Michael Elad, Stefano Ermon, Jiaming Song
- **Year**: 2022
- **Journal**: *NeurIPS 2022*, Vol. 35, pp. 23593–23606 (arXiv: 2201.11793)
- **Why it matters**: 사전훈련 diffusion model 을 **선형 역문제(deblur, super-resolution, inpainting, denoising)** 의 posterior sampler 로 변환 — SVD 분해로 측정값에 일관된 reverse process 도출. 별도 fine-tuning 불필요. / Converts pretrained diffusion models into **posterior samplers for linear inverse problems** (deblur, super-resolution, inpainting, denoising) — SVD decomposition yields measurement-consistent reverse process. No fine-tuning required.
- **Prerequisites**: Paper #25; DDPM/DDIM, SVD of measurement operator / 논문 #25; DDPM/DDIM, 측정 연산자 SVD
- **Status**: [x]

### 27. Cold Diffusion: Inverting Arbitrary Image Transforms without Noise
- **Authors**: Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie S. Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, Tom Goldstein
- **Year**: 2023
- **Journal**: *NeurIPS 2023* (arXiv: 2208.09392)
- **Why it matters**: Diffusion 의 stochastic 잡음을 **결정론적 변형(blur, mask, snow, downsampling 등)** 으로 일반화 — Gaussian noise 가정이 깨지는 regime 에서도 diffusion-style restoration 가능함을 보임. 비-Gaussian degradation 응용의 이론적 길잡이. / Generalizes diffusion's stochastic noise to **deterministic transforms (blur, mask, snow, downsampling)** — shows diffusion-style restoration works in regimes where Gaussian noise doesn't apply. Theoretical guide for non-Gaussian degradation applications.
- **Prerequisites**: Paper #25; DDPM training objective, fixed-point iteration / 논문 #25; DDPM 학습 목적함수, fixed-point 반복
- **Status**: [x]

### 28. Diffusion Posterior Sampling for General Noisy Inverse Problems (DPS)
- **Authors**: Hyungjin Chung, Jeongsol Kim, Michael T. McCann, Marc L. Klasky, Jong Chul Ye
- **Year**: 2023
- **Journal**: *Proc. ICLR 2023* (arXiv: 2209.14687)
- **Why it matters**: DDRM 의 한계(선형 측정만)를 극복 — **비선형 forward operator + non-Gaussian noise** 에 대해서도 manifold-constrained posterior sampling 가능. 의료 영상·과학 데이터의 일반적 forward model 에 직접 적용. / Lifts DDRM's linearity restriction — enables manifold-constrained posterior sampling for **nonlinear forward operators + non-Gaussian noise**. Directly applicable to general forward models in medical imaging and scientific data.
- **Prerequisites**: Paper #26; manifold gradient projection, automatic differentiation / 논문 #26; manifold gradient projection, 자동 미분
- **Status**: [x]

### 29. Ambient Diffusion: Learning Clean Distributions from Corrupted Data
- **Authors**: Giannis Daras, Kulin Shah, Yuval Dagan, Aravind Gollakota, Adam Klivans, Alexandros G. Dimakis
- **Year**: 2023
- **Journal**: *NeurIPS 2023* (arXiv: 2305.19256)
- **Why it matters**: **Clean ground truth 없는 데이터로 diffusion model 학습** — Noise2Noise 의 generative 일반화. 본 reading list regime 에서 score-based prior 를 자체 데이터로 학습할 수 있게 함. Self-supervised diffusion 의 시발점. / **Trains diffusion models without clean ground truth** — generative generalization of Noise2Noise. Enables learning score-based priors from in-domain corrupted data — starting point of self-supervised diffusion.
- **Prerequisites**: Papers #16, #25, #26; ambient sampler theory / 논문 #16, #25, #26; ambient sampler 이론
- **Status**: [x]

### 30. Denoising Diffusion Models for Plug-and-Play Image Restoration (DiffPIR)
- **Authors**: Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, Luc Van Gool
- **Year**: 2023
- **Journal**: *IEEE/CVF CVPR Workshops (NTIRE) 2023*, pp. 1219–1229 (arXiv: 2305.08995)
- **Why it matters**: 고전 **Plug-and-Play (PnP) 프레임워크** 의 denoiser 를 사전훈련 diffusion model 로 교체 — HQS/ADMM 분할법에 diffusion sampling 을 endpoint 로 끼워 넣음. 다양한 IR 작업을 동일 prior 로 처리. / Replaces the denoiser in classical **Plug-and-Play (PnP)** with a pretrained diffusion model — embeds diffusion sampling as the endpoint within HQS/ADMM splitting. Single prior handles diverse IR tasks.
- **Prerequisites**: Paper #26; HQS/ADMM splitting, plug-and-play prior / 논문 #26; HQS/ADMM 분할, plug-and-play prior
- **Status**: [x]

### 31. Low-Light Image Enhancement with Wavelet-Based Diffusion Models (DiffLL)
- **Authors**: Hai Jiang, Ao Luo, Haoqiang Fan, Songchen Han, Shuaicheng Liu
- **Year**: 2023
- **Journal**: *ACM Transactions on Graphics*, Vol. 42, No. 6, Article 238 (SIGGRAPH Asia 2023)
- **DOI**: 10.1145/3618373
- **Why it matters**: Diffusion 을 **wavelet 분해 영역에서 수행**해 저조도 enhancement 의 속도·품질 동시 개선. Tier C(diffusion) 와 Tier E(low-light) 의 교집점 — 두 가지 모두에 분류됨. / Performs diffusion **in wavelet-decomposed domain** for simultaneous speed and quality improvements in low-light enhancement. Intersection of Tier C (diffusion) and Tier E (low-light) — categorized in both.
- **Prerequisites**: Papers #1, #26; wavelet-domain neural networks / 논문 #1, #26; wavelet 영역 신경망
- **Status**: [x]

---

## Phase 6: Faint-Signal Enhancement (Solar / Astronomy) / 약한 신호 강조 (태양·천문 특화) (2003–2022)

> ⚠️ Caveat: Tier D 알고리즘은 **신호와 잡음을 함께 증폭**하므로 반드시 Phase 1–5 의 denoiser 적용 후 시각화/검출 단계에서 사용. / These algorithms **amplify both signal and residual noise** — apply *after* a Phase 1–5 denoiser, for visualization / detection-prep only.

### 32. A Wavelet Packets Equalization Technique to Reveal the Multiple Spatial-Scale Nature of Coronal Structures
- **Authors**: G. Stenborg, P. J. Cobelli
- **Year**: 2003
- **Journal**: *Astronomy & Astrophysics*, Vol. 398, No. 3, pp. 1185–1193
- **DOI**: 10.1051/0004-6361:20021687
- **Why it matters**: **코로나그래프 영상에서 다중 공간 스케일 구조를 wavelet packets equalization** 으로 동시 가시화. 강한 K-corona 배경에 묻힌 streamer·loop·CME 구조 분리에 사용. / Reveals multi-scale coronal structures via **wavelet packets equalization on coronagraph images**. Separates streamers / loops / CME structures buried in the strong K-corona background.
- **Prerequisites**: Paper #1; wavelet packet decomposition, K-corona / 논문 #1; wavelet packet 분해, K-corona
- **Status**: [x]

### 33. Gray and Color Image Contrast Enhancement by the Curvelet Transform
- **Authors**: Jean-Luc Starck, Fionn Murtagh, Emmanuel J. Candès, David L. Donoho
- **Year**: 2003
- **Journal**: *IEEE Transactions on Image Processing*, Vol. 12, No. 6, pp. 706–717
- **DOI**: 10.1109/TIP.2003.813140
- **Why it matters**: **Curvelet 계수의 비선형 매핑(coefficient enhancement curve)** 으로 약한 곡선 신호를 선택적으로 증폭. Tier B 의 curvelet shrinkage(잡음 제거) 와 한 짝 — denoise 후 enhancement. / Selectively amplifies faint curvilinear signals via **non-linear mapping of curvelet coefficients (enhancement curves)**. Companion to Tier-B curvelet shrinkage — apply after denoising.
- **Prerequisites**: Paper #6; non-linear coefficient mapping / 논문 #6; 비선형 계수 매핑
- **Status**: [x]

### 34. A Catalog of White Light Coronal Mass Ejections Observed by the SOHO Spacecraft (CDAW Catalog)
- **Authors**: S. Yashiro, N. Gopalswamy, G. Michalek, O. C. St. Cyr, S. P. Plunkett, N. B. Rich, R. A. Howard
- **Year**: 2004
- **Journal**: *Journal of Geophysical Research*, Vol. 109, A07105
- **DOI**: 10.1029/2003JA010282
- **Why it matters**: **시간적 중앙값 배경 차감(temporal-median background)** 에 의한 CME 검출의 community-standard 적용 사례. CDAW catalog 는 LASCO C2/C3 데이터의 사실상 표준 CME 카탈로그 — 본 reading list 에서 'temporal-median-bg' 기법의 anchor citation. / Anchor citation for the **temporal-median background subtraction** technique applied to CME detection. The CDAW catalog is the de-facto standard CME catalog for LASCO C2/C3 — used as the reference for the 'temporal-median-bg' technique in the LOLIPOP audit.
- **Prerequisites**: SOHO/LASCO C2/C3 instruments, CME morphology / SOHO/LASCO C2/C3 기기, CME 형태학
- **Status**: [x]

### 35. The Depiction of Coronal Structure in White-Light Images (NRGF)
- **Authors**: Huw Morgan, Shadia R. Habbal, Richard Woo
- **Year**: 2006
- **Journal**: *Solar Physics*, Vol. 236, No. 2, pp. 263–272
- **DOI**: 10.1007/s11207-006-0113-6
- **Why it matters**: **Normalizing Radial Graded Filter (NRGF)** — 코로나그래프 영상의 동경(radial) 방향 1/r⁴–1/r⁶ 강도 감쇠를 표준화하여 outer corona 의 약한 구조를 균등하게 가시화. SOHO/LASCO 이래 코로나그래프 시각화의 사실상 표준. / **Normalizing Radial Graded Filter (NRGF)** — standardizes the 1/r⁴–1/r⁶ radial intensity falloff in coronagraph images, equalizing visibility of faint outer-corona structures. De-facto standard for coronagraph visualization since SOHO/LASCO.
- **Prerequisites**: Coronal Thomson scattering geometry, K-corona radial profile / 코로나 Thomson 산란 기하, K-corona 동경 프로파일
- **Status**: [x]

### 36. The Undecimated Wavelet Decomposition and Its Reconstruction (à-trous)
- **Authors**: Jean-Luc Starck, Jalal Fadili, Fionn Murtagh
- **Year**: 2007
- **Journal**: *IEEE Transactions on Image Processing*, Vol. 16, No. 2, pp. 297–309
- **DOI**: 10.1109/TIP.2006.887733
- **Why it matters**: **à-trous (with holes) 알고리즘** 의 정형화 — translation-invariant 한 redundant wavelet 분해. 천문 영상 처리(MultiResolution Analysis on the Sphere, MR/1, ISAP) 의 표준 도구. NRGF·MGN 과 결합해 코로나 영상 다중스케일 처리에 사용. / Formalizes the **à-trous (with holes) algorithm** — a translation-invariant redundant wavelet decomposition. Standard tool in astronomical image processing (MR/1, ISAP). Used with NRGF / MGN for multi-scale coronal-image processing.
- **Prerequisites**: Paper #1; redundant frame, B-spline scaling / 논문 #1; 잉여 프레임, B-spline 스케일링
- **Status**: [x]

### 37. Robust Principal Component Analysis? (RPCA)
- **Authors**: Emmanuel J. Candès, Xiaodong Li, Yi Ma, John Wright
- **Year**: 2011
- **Journal**: *Journal of the ACM*, Vol. 58, No. 3, Article 11, pp. 1–37
- **DOI**: 10.1145/1970392.1970395
- **Why it matters**: 영상/비디오 데이터를 **저랭크 배경(L) + 희소 전경(S)** 으로 분해하는 convex relaxation (Principal Component Pursuit). 코로나그래프 시계열에서 **정적 K-corona(저랭크) + 동적 CME/streamer(희소)** 분리에 직접 적용. / Convex relaxation (Principal Component Pursuit) decomposing image/video data into **low-rank background (L) + sparse foreground (S)**. Direct application: separating **static K-corona (low-rank) from dynamic CME/streamer (sparse)** in coronagraph time series.
- **Prerequisites**: SVD, nuclear-norm minimization, ADMM / SVD, nuclear-norm 최소화, ADMM
- **Status**: [x]

### 38. Multi-Scale Gaussian Normalization for Solar Image Processing (MGN)
- **Authors**: Huw Morgan, Miloslav Druckmüller
- **Year**: 2014
- **Journal**: *Solar Physics*, Vol. 289, No. 8, pp. 2945–2955
- **DOI**: 10.1007/s11207-014-0523-9
- **Why it matters**: **다중 스케일 Gaussian 표준화** — 여러 스케일의 Gaussian filter 결과를 결합해 동시에 약한 미세구조와 큰 동적 범위 구조를 가시화. SDO/AIA·STEREO·Solar Orbiter EUV 영상의 사실상 표준 enhancement. NRGF 의 EUV 시대 후속. / **Multi-scale Gaussian normalization** — combines results across many Gaussian scales to simultaneously reveal faint fine structure and large-dynamic-range structure. De-facto standard EUV enhancement for SDO/AIA, STEREO, Solar Orbiter. EUV-era successor to NRGF.
- **Prerequisites**: Paper #35; Gaussian scale-space / 논문 #35; Gaussian scale-space
- **Status**: [x]

### 39. SiRGraF: A Simple Radial Gradient Filter for Batch-Processing of Coronagraph Images
- **Authors**: Ritesh Patel, Sankar Majumdar, Vaibhav Pant, Dipankar Banerjee
- **Year**: 2022
- **Journal**: *Solar Physics*, Vol. 297, 27
- **DOI**: 10.1007/s11207-022-01957-y
- **Why it matters**: **단순 동경 기울기 필터(radial gradient)** 로 배치 처리에 최적화된 코로나그래프 enhancement. NRGF·MGN 보다 단순하고 빠름 — STEREO/COR2, PSP/WISPR 대량 처리에 적합. / **Simple radial gradient filter** optimized for batch processing of coronagraph images. Simpler and faster than NRGF / MGN — suited for large-volume STEREO/COR2 and PSP/WISPR processing.
- **Prerequisites**: Papers #35, #38; radial intensity gradient / 논문 #35, #38; 동경 강도 기울기
- **Status**: [x]

---

## Phase 7: Low-Light Deep Learning / 저조도 딥러닝 (2017–2023)

> 본 phase 의 알고리즘은 **저조도 자연 영상**에 설계됨. 과학 데이터로의 전이는 잡음 모델·신호 통계 일치 여부에 의존 — drop-in 솔루션 아님. / These algorithms target **natural low-light images**. Transfer to scientific data depends on noise-model / signal-statistic overlap — not drop-in solutions.

### 40. LLNet: A Deep Autoencoder Approach to Natural Low-Light Image Enhancement
- **Authors**: Kin Gwn Lore, Adedotun Akintayo, Soumik Sarkar
- **Year**: 2017
- **Journal**: *Pattern Recognition*, Vol. 61, pp. 650–662
- **DOI**: 10.1016/j.patcog.2016.06.008
- **Why it matters**: **저조도 enhancement 의 첫 본격 deep learning 시도** — stacked sparse denoising autoencoder. 합성 저조도 데이터(감마 보정 + Gaussian noise) 로 학습. 이후 모든 저조도 DL 모델의 baseline. / **First major deep-learning approach to low-light enhancement** — stacked sparse denoising autoencoder trained on synthetic low-light data (gamma + Gaussian noise). Baseline for all subsequent low-light DL models.
- **Prerequisites**: Stacked autoencoder, gamma correction / 적층 오토인코더, 감마 보정
- **Status**: [x]

### 41. Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (Zero-DCE)
- **Authors**: Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, Runmin Cong
- **Year**: 2020
- **Journal**: *IEEE/CVF CVPR 2020*, pp. 1780–1789
- **DOI**: 10.1109/CVPR42600.2020.00185
- **Why it matters**: **참조 영상 없이(zero-reference) 비참조 손실(non-reference loss)** 만으로 픽셀별 enhancement curve 를 추정. 대량의 unpaired 저조도 영상에서 즉시 학습 — 데이터 수집 부담 최소화. / Estimates pixel-wise enhancement curves using **zero-reference (non-reference) losses** alone. Trains on unpaired low-light images at scale — minimal data-collection burden.
- **Prerequisites**: Paper #40; non-reference quality losses (exposure, color constancy) / 논문 #40; 비참조 품질 손실 (노출, 색 항상성)
- **Status**: [x]

### 42. Learning Enriched Features for Real Image Restoration and Enhancement (MIRNet)
- **Authors**: Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, Ling Shao
- **Year**: 2020
- **Journal**: *ECCV 2020*, LNCS 12370, pp. 492–511
- **DOI**: 10.1007/978-3-030-58595-2_30
- **Why it matters**: **다중 해상도 병렬 처리 + 선택적 커널 attention** 의 통합 IR 백본 — 저조도 enhancement, denoising, deblurring 등 다중 작업에 단일 구조로 SOTA. / Unified IR backbone combining **multi-resolution parallel streams + selective-kernel attention** — single architecture achieves SOTA on low-light enhancement, denoising, deblurring.
- **Prerequisites**: Multi-scale architecture, attention mechanisms / 다중 스케일 구조, attention 메커니즘
- **Status**: [x]

### 43. EnlightenGAN: Deep Light Enhancement without Paired Supervision
- **Authors**: Yifan Jiang, Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, et al.
- **Year**: 2021
- **Journal**: *IEEE Transactions on Image Processing*, Vol. 30, pp. 2340–2349
- **DOI**: 10.1109/TIP.2021.3051462
- **Why it matters**: **GAN 기반 unpaired 저조도 enhancement** — global-local discriminator 와 self-regularization 으로 paired 데이터 없이 학습. CycleGAN 계보의 저조도 응용. / **GAN-based unpaired low-light enhancement** — global-local discriminator + self-regularization enable training without paired data. Low-light specialization in the CycleGAN lineage.
- **Prerequisites**: GAN training, CycleGAN concept / GAN 학습, CycleGAN 개념
- **Status**: [x]

### 44. Toward Fast, Flexible, and Robust Low-Light Image Enhancement (SCI)
- **Authors**: Long Ma, Tengyu Ma, Risheng Liu, Xin Fan, Zhongxuan Luo
- **Year**: 2022
- **Journal**: *IEEE/CVF CVPR 2022*, pp. 5637–5646
- **DOI**: 10.1109/CVPR52688.2022.00555
- **Why it matters**: **Self-Calibrated Illumination (SCI)** — cascaded illumination estimator 에 self-calibration 모듈을 더해 **0.5 ms** 수준의 추론 속도와 일반화 능력 동시 확보. 모바일·실시간 응용 가능. / **Self-Calibrated Illumination (SCI)** — adds a self-calibration module to a cascaded illumination estimator, achieving **~0.5 ms** inference and strong generalization. Mobile / real-time capable.
- **Prerequisites**: Retinex decomposition, knowledge distillation / Retinex 분해, 지식 증류
- **Status**: [x]

### 45. SNR-Aware Low-Light Image Enhancement
- **Authors**: Xiaogang Xu, Ruixing Wang, Chi-Wing Fu, Jiaya Jia
- **Year**: 2022
- **Journal**: *IEEE/CVF CVPR 2022*, pp. 17714–17724
- **DOI**: 10.1109/CVPR52688.2022.01719
- **Why it matters**: **픽셀별 SNR 추정 후 SNR-conditioned attention** 으로 영역별 enhancement 강도 조절. 본 reading list 의 regime 정의(SNR ≲ a few)에 가장 직접적으로 부합하는 DL 구조 — 강한 신호 영역과 약한 신호 영역을 분리해 처리. / Estimates **per-pixel SNR + SNR-conditioned attention** to modulate enhancement intensity per region. Most directly aligned DL architecture with this reading list's regime definition (SNR ≲ a few) — separates treatment of strong-signal and weak-signal regions.
- **Prerequisites**: Papers #40, #42; per-pixel SNR estimation, transformer attention / 논문 #40, #42; 픽셀별 SNR 추정, 트랜스포머 attention
- **Status**: [x]

---

## Cross-References / 교차 참조

- Tier-D 5편 (#35 NRGF, #36 à-trous, #37 RPCA, #38 MGN, #39 SiRGraF) 은 Solar_Observation Phase 7 (Calibration & Cross-Cutting Techniques) 와 도메인이 겹침. 본 토픽에 풀 entry, Solar_Observation 에는 cross-reference 만 두는 정책. / The five Tier-D entries (#35–#39) overlap with Solar_Observation Phase 7. Full entries in this topic; Solar_Observation maintains only cross-references.
- Tier-A self-supervised DL 7편 (#16–#22) 은 Artificial_Intelligence 의 미래 "Computer Vision: Image Restoration" 단원과 연결 가능. / Tier-A self-supervised DL entries (#16–#22) are candidates for a future Artificial_Intelligence "Computer Vision: Image Restoration" module.

## Excluded Entries / 제외 항목

These four `low-snr-weak-signal`-tagged algorithms in the LOLIPOP audit have **textbook-only** references and are intentionally excluded from the numbered reading list (textbook anchors retained for reference):

- **σ-clipping** — Howell, S. B. 2006. *Handbook of CCD Astronomy* (2nd ed.).
- **Multi-frame median rejection** — Howell, S. B. 2006.
- **Temporal median (frame stack)** — Gonzalez & Woods 2018. *Digital Image Processing* (4th ed.).
- **Weighted-average stacking** — Howell, S. B. 2006.

## Source / 출처

Initial 41 entries extracted from `LOW_SNR_WEAK_SIGNAL_REFERENCES.md` (LOLIPOP project, audited 2026-04-30, 4 verification agents on 181 algorithm rows). Synthesis sources (#15 Cryo-CARE, #25 Kadkhodaie & Simoncelli, #27 Cold Diffusion, #29 Ambient Diffusion) added from the same file's "Synthesis sources for the regime characterization" section.

초기 41편은 LOLIPOP 프로젝트 `LOW_SNR_WEAK_SIGNAL_REFERENCES.md` (2026-04-30 audit, 4 verification agent, 181 알고리즘 행) 에서 추출. Synthesis source 4편 (#15 Cryo-CARE, #25 Kadkhodaie & Simoncelli, #27 Cold Diffusion, #29 Ambient Diffusion) 은 동일 파일의 "Synthesis sources for the regime characterization" 섹션에서 추가.

## Legend / 범례

- `[ ]` not started / 시작 전
- `[~]` in progress / 진행 중
- `[x]` completed / 완료
