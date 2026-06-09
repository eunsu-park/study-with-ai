# Low-SNR Imaging / 저신호대잡음 영상 — Topic Map / 주제 지도

## Overview / 개요
A study track on **image restoration and faint-signal enhancement in the low signal-to-noise regime** — where per-pixel SNR is ≲ a few and the signal is buried in comparable-or-larger background noise, often without clean ground truth. Covers classical denoising (Anscombe-stabilized BM3D, NLM, wavelet/curvelet/shearlet shrinkage), self-supervised deep learning (Noise2Noise/Void/Self family), diffusion-based inverse-problem solvers (DDRM, DPS), faint-signal visual enhancement (NRGF, MGN, à-trous coronagraph filtering), cosmic-ray rejection (L.A.Cosmic, deepCR), and low-light deep learning (LLNet, MIRNet, Zero-DCE, EnlightenGAN). Born from the LOLIPOP project's algorithm survey, but the techniques generalise to coronagraph/auroral/cryo-EM/microscopy data and any photon-starved scientific imaging.

**저신호잡음비 regime에서의 영상 복원과 약한 신호 강조**를 다루는 학습 트랙. per-pixel SNR ≲ 수 단위, 신호가 비슷하거나 큰 배경 잡음에 묻혀 있고 clean ground truth 가 없는 경우가 표적. 고전 denoising (Anscombe-안정화 BM3D, NLM, wavelet/curvelet/shearlet shrinkage), 자기지도 딥러닝 (Noise2Noise/Void/Self 계열), diffusion 기반 역문제 풀이 (DDRM, DPS), 약한 신호 시각화 (NRGF, MGN, à-trous coronagraph), cosmic-ray rejection (L.A.Cosmic, deepCR), 저조도 딥러닝 (LLNet, MIRNet, Zero-DCE, EnlightenGAN) 을 망라. LOLIPOP 프로젝트의 알고리즘 서베이에서 출발했지만, 코로나그래프·오로라·cryo-EM·현미경·광자 부족 과학 영상 전반으로 일반화 가능.

## Learning Roadmap / 학습 로드맵

### Phase 1 (Tier B basics): Classical Patch-Based and Transform-Domain Denoising / 고전 패치·변환 영역 denoising (1994–2013)
- Wavelet shrinkage thresholding (VisuShrink, SureShrink, BayesShrink) / wavelet shrinkage 임계법
- Curvelet, contourlet, shearlet — directional sparsity / 방향성 희소성
- Non-local means (NLM) and BM3D / 비국소 평균 및 BM3D
- BM4D / V-BM4D for volumetric and video data / 볼륨 및 비디오 데이터

### Phase 2 (Tier A — Poisson stats): Photon-Statistics-Aware Denoising / 광자 통계 기반 denoising (1948–2013)
- Anscombe / generalized Anscombe variance-stabilizing transforms / 분산 안정화 변환
- Optimal Anscombe inversion for Poisson–Gaussian / Poisson–Gaussian 최적 역변환
- PURE-LET (unbiased risk for mixed noise) / 혼합 잡음 PURE-LET
- Poisson NL means / Poisson NLM

### Phase 3 (Tier A — DL): Self-Supervised Deep Learning Denoising / 자기지도 딥러닝 denoising (2018–2022)
- Noise2Noise lineage (training without clean ground truth) / clean GT 없는 학습
- Blind-spot networks (Noise2Void, Noise2Self) / 블라인드 스팟 네트워크
- Self-supervised single-image (Self2Self, Neighbor2Neighbor, Blind2Unblind, R2R) / 단일 영상 자기지도
- Domain reference: Cryo-CARE (cryo-EM regime origin) / Cryo-CARE (regime 기원 reference)

### Phase 4 (Tier B — astronomy): Outlier and Cosmic-Ray Rejection / 이상치 및 cosmic-ray 제거 (2001–2020)
- L.A.Cosmic — Laplacian edge detection / Laplacian 엣지 검출
- deepCR — CNN-based cosmic-ray rejection / CNN 기반 cosmic-ray 제거

### Phase 5 (Tier C): Diffusion-Based Inverse-Problem Restoration / Diffusion 모형 기반 역문제 복원 (2022–2023)
- DDRM, DPS — pretrained-prior posterior sampling / 사전훈련 prior 사후표본
- DiffPIR — plug-and-play with diffusion / Diffusion plug-and-play
- DiffLL — wavelet-domain low-light diffusion / Wavelet 영역 저조도 diffusion
- Caveats: prior hallucination, injection-recovery validation / Prior hallucination, injection-recovery 검증

### Phase 6 (Tier D): Faint-Signal Enhancement (Solar/Astronomy) / 약한 신호 강조 (태양·천문 특화) (2003–2022)
- à-trous / undecimated wavelet transform / à-trous 웨이블릿 변환
- NRGF, MGN — coronal radial filtering / 코로나 동경 필터링
- SiRGraF — radial gradient filter / 동경 기울기 필터
- Wavelet-packets equalization for coronagraphs / 코로나그래프 wavelet packets 균등화
- Robust PCA — low-rank + sparse decomposition / 저랭크 + 희소 분해

### Phase 7 (Tier E): Low-Light Deep Learning / 저조도 딥러닝 (2017–2023)
- LLNet, MIRNet — paired supervised low-light enhancement / 쌍 지도 저조도 강화
- Zero-DCE, EnlightenGAN — zero-reference / unpaired / zero-reference / 비쌍 학습
- SCI, SNR-aware — fast / SNR-conditioned architectures / 빠른 SNR-조건화 구조

<!-- AUTO-INDEX:START -->
**Progress / 진행**: 45 / 45  ·  Source / 원본: [`reading_lists/low-snr-imaging.md`](../reading_lists/low-snr-imaging.md)

| # | Paper / 논문 | Year | Status | Links |
|---|---|---|---|---|
| 1 | Ideal Spatial Adaptation by Wavelet Shrinkage (VisuShrink) | 1994 | ✅ | [📝 notes](../papers/donoho1994ideal/donoho1994ideal_notes.md) · [💻 code](../papers/donoho1994ideal/donoho1994ideal_implementation.ipynb) · [📄 pdf](../papers/donoho1994ideal/donoho1994ideal_paper.pdf) |
| 2 | Adapting to Unknown Smoothness via Wavelet Shrinkage (SureShrink) | 1995 | ✅ | [📝 notes](../papers/donoho1995adapting/donoho1995adapting_notes.md) · [💻 code](../papers/donoho1995adapting/donoho1995adapting_implementation.ipynb) · [📄 pdf](../papers/donoho1995adapting/donoho1995adapting_paper.pdf) |
| 3 | Adaptive Wavelet Thresholding for Image Denoising and Compression (BayesShrink) | 2000 | ✅ | [📝 notes](../papers/chang2000adaptive/chang2000adaptive_notes.md) · [💻 code](../papers/chang2000adaptive/chang2000adaptive_implementation.ipynb) · [📄 pdf](../papers/chang2000adaptive/chang2000adaptive_paper.pdf) |
| 4 | A Non-Local Algorithm for Image Denoising (NLM) | 2005 | ✅ | [📝 notes](../papers/buades2005nonlocal/buades2005nonlocal_notes.md) · [💻 code](../papers/buades2005nonlocal/buades2005nonlocal_implementation.ipynb) · [📄 pdf](../papers/buades2005nonlocal/buades2005nonlocal_paper.pdf) |
| 5 | The Contourlet Transform: An Efficient Directional Multiresolution Image Representation | 2005 | ✅ | [📝 notes](../papers/do2005contourlet/do2005contourlet_notes.md) · [💻 code](../papers/do2005contourlet/do2005contourlet_implementation.ipynb) · [📄 pdf](../papers/do2005contourlet/do2005contourlet_paper.pdf) |
| 6 | Fast Discrete Curvelet Transforms | 2006 | ✅ | [📝 notes](../papers/candes2006fast/candes2006fast_notes.md) · [💻 code](../papers/candes2006fast/candes2006fast_implementation.ipynb) · [📄 pdf](../papers/candes2006fast/candes2006fast_paper.pdf) |
| 7 | Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering (BM3D) | 2007 | ✅ | [📝 notes](../papers/dabov2007image/dabov2007image_notes.md) · [💻 code](../papers/dabov2007image/dabov2007image_implementation.ipynb) · [📄 pdf](../papers/dabov2007image/dabov2007image_paper.pdf) |
| 8 | Sparse Directional Image Representations using the Discrete Shearlet Transform | 2008 | ✅ | [📝 notes](../papers/easley2008sparse/easley2008sparse_notes.md) · [💻 code](../papers/easley2008sparse/easley2008sparse_implementation.ipynb) · [📄 pdf](../papers/easley2008sparse/easley2008sparse_paper.pdf) |
| 9 | Video Denoising, Deblocking, and Enhancement Through Separable 4-D Nonlocal Spatiotemporal Transforms (V-BM4D) | 2012 | ✅ | [📝 notes](../papers/maggioni2012video/maggioni2012video_notes.md) · [💻 code](../papers/maggioni2012video/maggioni2012video_implementation.ipynb) · [📄 pdf](../papers/maggioni2012video/maggioni2012video_paper.pdf) |
| 10 | Nonlocal Transform-Domain Filter for Volumetric Data Denoising and Reconstruction (BM4D) | 2013 | ✅ | [📝 notes](../papers/maggioni2013nonlocal/maggioni2013nonlocal_notes.md) · [💻 code](../papers/maggioni2013nonlocal/maggioni2013nonlocal_implementation.ipynb) · [📄 pdf](../papers/maggioni2013nonlocal/maggioni2013nonlocal_paper.pdf) |
| 11 | The Transformation of Poisson, Binomial and Negative-Binomial Data (Anscombe Transform) | 1948 | ✅ | [📝 notes](../papers/anscombe1948transformation/anscombe1948transformation_notes.md) · [💻 code](../papers/anscombe1948transformation/anscombe1948transformation_implementation.ipynb) · [📄 pdf](../papers/anscombe1948transformation/anscombe1948transformation_paper.pdf) |
| 12 | Poisson NL Means: Unsupervised Non-local Means for Poisson Noise | 2010 | ✅ | [📝 notes](../papers/deledalle2010poisson/deledalle2010poisson_notes.md) · [💻 code](../papers/deledalle2010poisson/deledalle2010poisson_implementation.ipynb) · [📄 pdf](../papers/deledalle2010poisson/deledalle2010poisson_paper.pdf) |
| 13 | Image Denoising in Mixed Poisson–Gaussian Noise (PURE-LET) | 2011 | ✅ | [📝 notes](../papers/luisier2011image/luisier2011image_notes.md) · [💻 code](../papers/luisier2011image/luisier2011image_implementation.ipynb) · [📄 pdf](../papers/luisier2011image/luisier2011image_paper.pdf) |
| 14 | Optimal Inversion of the Generalized Anscombe Transformation for Poisson–Gaussian Noise | 2013 | ✅ | [📝 notes](../papers/makitalo2013optimal/makitalo2013optimal_notes.md) · [💻 code](../papers/makitalo2013optimal/makitalo2013optimal_implementation.ipynb) · [📄 pdf](../papers/makitalo2013optimal/makitalo2013optimal_paper.pdf) |
| 15 | Cryo-CARE: Content-Aware Image Restoration for Cryo-Electron Tomography | 2019 | ✅ | [📝 notes](../papers/buchholz2019cryocare/buchholz2019cryocare_notes.md) · [💻 code](../papers/buchholz2019cryocare/buchholz2019cryocare_implementation.ipynb) · [📄 pdf](../papers/buchholz2019cryocare/buchholz2019cryocare_paper.pdf) |
| 16 | Noise2Noise: Learning Image Restoration without Clean Data | 2018 | ✅ | [📝 notes](../papers/lehtinen2018noise2noise/lehtinen2018noise2noise_notes.md) · [💻 code](../papers/lehtinen2018noise2noise/lehtinen2018noise2noise_implementation.ipynb) · [📄 pdf](../papers/lehtinen2018noise2noise/lehtinen2018noise2noise_paper.pdf) |
| 17 | Noise2Void — Learning Denoising from Single Noisy Images | 2019 | ✅ | [📝 notes](../papers/krull2019noise2void/krull2019noise2void_notes.md) · [💻 code](../papers/krull2019noise2void/krull2019noise2void_implementation.ipynb) · [📄 pdf](../papers/krull2019noise2void/krull2019noise2void_paper.pdf) |
| 18 | Noise2Self: Blind Denoising by Self-Supervision | 2019 | ✅ | [📝 notes](../papers/batson2019noise2self/batson2019noise2self_notes.md) · [💻 code](../papers/batson2019noise2self/batson2019noise2self_implementation.ipynb) · [📄 pdf](../papers/batson2019noise2self/batson2019noise2self_paper.pdf) |
| 19 | Self2Self with Dropout: Learning Self-Supervised Denoising from Single Image | 2020 | ✅ | [📝 notes](../papers/quan2020self2self/quan2020self2self_notes.md) · [💻 code](../papers/quan2020self2self/quan2020self2self_implementation.ipynb) · [📄 pdf](../papers/quan2020self2self/quan2020self2self_paper.pdf) |
| 20 | Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images | 2021 | ✅ | [📝 notes](../papers/huang2021neighbor2neighbor/huang2021neighbor2neighbor_notes.md) · [💻 code](../papers/huang2021neighbor2neighbor/huang2021neighbor2neighbor_implementation.ipynb) · [📄 pdf](../papers/huang2021neighbor2neighbor/huang2021neighbor2neighbor_paper.pdf) |
| 21 | Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising (R2R) | 2021 | ✅ | [📝 notes](../papers/pang2021recorruptedtorecorrupted/pang2021recorruptedtorecorrupted_notes.md) · [💻 code](../papers/pang2021recorruptedtorecorrupted/pang2021recorruptedtorecorrupted_implementation.ipynb) · [📄 pdf](../papers/pang2021recorruptedtorecorrupted/pang2021recorruptedtorecorrupted_paper.pdf) |
| 22 | Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots | 2022 | ✅ | [📝 notes](../papers/wang2022blind2unblind/wang2022blind2unblind_notes.md) · [💻 code](../papers/wang2022blind2unblind/wang2022blind2unblind_implementation.ipynb) · [📄 pdf](../papers/wang2022blind2unblind/wang2022blind2unblind_paper.pdf) |
| 23 | Cosmic-Ray Rejection by Laplacian Edge Detection (L.A.Cosmic) | 2001 | ✅ | [📝 notes](../papers/vandokkum2001cosmicray/vandokkum2001cosmicray_notes.md) · [💻 code](../papers/vandokkum2001cosmicray/vandokkum2001cosmicray_implementation.ipynb) · [📄 pdf](../papers/vandokkum2001cosmicray/vandokkum2001cosmicray_paper.pdf) |
| 24 | deepCR: Cosmic Ray Rejection with Deep Learning | 2020 | ✅ | [📝 notes](../papers/zhang2020deepcr/zhang2020deepcr_notes.md) · [💻 code](../papers/zhang2020deepcr/zhang2020deepcr_implementation.ipynb) · [📄 pdf](../papers/zhang2020deepcr/zhang2020deepcr_paper.pdf) |
| 25 | Stochastic Solutions for Linear Inverse Problems Using the Prior Implicit in a Denoiser | 2021 | ✅ | [📝 notes](../papers/kadkhodaie2021stochastic/kadkhodaie2021stochastic_notes.md) · [💻 code](../papers/kadkhodaie2021stochastic/kadkhodaie2021stochastic_implementation.ipynb) · [📄 pdf](../papers/kadkhodaie2021stochastic/kadkhodaie2021stochastic_paper.pdf) |
| 26 | Denoising Diffusion Restoration Models (DDRM) | 2022 | ✅ | [📝 notes](../papers/kawar2022denoising/kawar2022denoising_notes.md) · [💻 code](../papers/kawar2022denoising/kawar2022denoising_implementation.ipynb) · [📄 pdf](../papers/kawar2022denoising/kawar2022denoising_paper.pdf) |
| 27 | Cold Diffusion: Inverting Arbitrary Image Transforms without Noise | 2023 | ✅ | [📝 notes](../papers/bansal2023cold/bansal2023cold_notes.md) · [💻 code](../papers/bansal2023cold/bansal2023cold_implementation.ipynb) · [📄 pdf](../papers/bansal2023cold/bansal2023cold_paper.pdf) |
| 28 | Diffusion Posterior Sampling for General Noisy Inverse Problems (DPS) | 2023 | ✅ | [📝 notes](../papers/chung2023diffusion/chung2023diffusion_notes.md) · [💻 code](../papers/chung2023diffusion/chung2023diffusion_implementation.ipynb) · [📄 pdf](../papers/chung2023diffusion/chung2023diffusion_paper.pdf) |
| 29 | Ambient Diffusion: Learning Clean Distributions from Corrupted Data | 2023 | ✅ | [📝 notes](../papers/daras2023ambient/daras2023ambient_notes.md) · [💻 code](../papers/daras2023ambient/daras2023ambient_implementation.ipynb) · [📄 pdf](../papers/daras2023ambient/daras2023ambient_paper.pdf) |
| 30 | Denoising Diffusion Models for Plug-and-Play Image Restoration (DiffPIR) | 2023 | ✅ | [📝 notes](../papers/zhu2023denoising/zhu2023denoising_notes.md) · [💻 code](../papers/zhu2023denoising/zhu2023denoising_implementation.ipynb) · [📄 pdf](../papers/zhu2023denoising/zhu2023denoising_paper.pdf) |
| 31 | Low-Light Image Enhancement with Wavelet-Based Diffusion Models (DiffLL) | 2023 | ✅ | [📝 notes](../papers/jiang2023lowlight/jiang2023lowlight_notes.md) · [💻 code](../papers/jiang2023lowlight/jiang2023lowlight_implementation.ipynb) · [📄 pdf](../papers/jiang2023lowlight/jiang2023lowlight_paper.pdf) |
| 32 | A Wavelet Packets Equalization Technique to Reveal the Multiple Spatial-Scale Nature of Coronal Structures | 2003 | ✅ | [📝 notes](../papers/stenborg2003wavelet/stenborg2003wavelet_notes.md) · [💻 code](../papers/stenborg2003wavelet/stenborg2003wavelet_implementation.ipynb) · [📄 pdf](../papers/stenborg2003wavelet/stenborg2003wavelet_paper.pdf) |
| 33 | Gray and Color Image Contrast Enhancement by the Curvelet Transform | 2003 | ✅ | [📝 notes](../papers/starck2003gray/starck2003gray_notes.md) · [💻 code](../papers/starck2003gray/starck2003gray_implementation.ipynb) · [📄 pdf](../papers/starck2003gray/starck2003gray_paper.pdf) |
| 34 | A Catalog of White Light Coronal Mass Ejections Observed by the SOHO Spacecraft (CDAW Catalog) | 2004 | ✅ | [📝 notes](../papers/yashiro2004catalog/yashiro2004catalog_notes.md) · [💻 code](../papers/yashiro2004catalog/yashiro2004catalog_implementation.ipynb) · [📄 pdf](../papers/yashiro2004catalog/yashiro2004catalog_paper.pdf) |
| 35 | The Depiction of Coronal Structure in White-Light Images (NRGF) | 2006 | ✅ | [📝 notes](../papers/morgan2006depiction/morgan2006depiction_notes.md) · [💻 code](../papers/morgan2006depiction/morgan2006depiction_implementation.ipynb) · [📄 pdf](../papers/morgan2006depiction/morgan2006depiction_paper.pdf) |
| 36 | The Undecimated Wavelet Decomposition and Its Reconstruction (à-trous) | 2007 | ✅ | [📝 notes](../papers/starck2007undecimated/starck2007undecimated_notes.md) · [💻 code](../papers/starck2007undecimated/starck2007undecimated_implementation.ipynb) · [📄 pdf](../papers/starck2007undecimated/starck2007undecimated_paper.pdf) |
| 37 | Robust Principal Component Analysis? (RPCA) | 2011 | ✅ | [📝 notes](../papers/candes2011robust/candes2011robust_notes.md) · [💻 code](../papers/candes2011robust/candes2011robust_implementation.ipynb) · [📄 pdf](../papers/candes2011robust/candes2011robust_paper.pdf) |
| 38 | Multi-Scale Gaussian Normalization for Solar Image Processing (MGN) | 2014 | ✅ | [📝 notes](../papers/morgan2014multiscale/morgan2014multiscale_notes.md) · [💻 code](../papers/morgan2014multiscale/morgan2014multiscale_implementation.ipynb) · [📄 pdf](../papers/morgan2014multiscale/morgan2014multiscale_paper.pdf) |
| 39 | SiRGraF: A Simple Radial Gradient Filter for Batch-Processing of Coronagraph Images | 2022 | ✅ | [📝 notes](../papers/patel2022sirgraf/patel2022sirgraf_notes.md) · [💻 code](../papers/patel2022sirgraf/patel2022sirgraf_implementation.ipynb) · [📄 pdf](../papers/patel2022sirgraf/patel2022sirgraf_paper.pdf) |
| 40 | LLNet: A Deep Autoencoder Approach to Natural Low-Light Image Enhancement | 2017 | ✅ | [📝 notes](../papers/lore2017llnet/lore2017llnet_notes.md) · [💻 code](../papers/lore2017llnet/lore2017llnet_implementation.ipynb) · [📄 pdf](../papers/lore2017llnet/lore2017llnet_paper.pdf) |
| 41 | Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (Zero-DCE) | 2020 | ✅ | [📝 notes](../papers/guo2020zeroreference/guo2020zeroreference_notes.md) · [💻 code](../papers/guo2020zeroreference/guo2020zeroreference_implementation.ipynb) · [📄 pdf](../papers/guo2020zeroreference/guo2020zeroreference_paper.pdf) |
| 42 | Learning Enriched Features for Real Image Restoration and Enhancement (MIRNet) | 2020 | ✅ | [📝 notes](../papers/zamir2020learning/zamir2020learning_notes.md) · [💻 code](../papers/zamir2020learning/zamir2020learning_implementation.ipynb) · [📄 pdf](../papers/zamir2020learning/zamir2020learning_paper.pdf) |
| 43 | EnlightenGAN: Deep Light Enhancement without Paired Supervision | 2021 | ✅ | [📝 notes](../papers/jiang2021enlightengan/jiang2021enlightengan_notes.md) · [💻 code](../papers/jiang2021enlightengan/jiang2021enlightengan_implementation.ipynb) · [📄 pdf](../papers/jiang2021enlightengan/jiang2021enlightengan_paper.pdf) |
| 44 | Toward Fast, Flexible, and Robust Low-Light Image Enhancement (SCI) | 2022 | ✅ | [📝 notes](../papers/ma2022toward/ma2022toward_notes.md) · [💻 code](../papers/ma2022toward/ma2022toward_implementation.ipynb) · [📄 pdf](../papers/ma2022toward/ma2022toward_paper.pdf) |
| 45 | SNR-Aware Low-Light Image Enhancement | 2022 | ✅ | [📝 notes](../papers/xu2022snraware/xu2022snraware_notes.md) · [💻 code](../papers/xu2022snraware/xu2022snraware_implementation.ipynb) · [📄 pdf](../papers/xu2022snraware/xu2022snraware_paper.pdf) |
<!-- AUTO-INDEX:END -->
