---
title: "Machine Learning in Solar Physics"
authors: [Andrés Asensio Ramos, Mark C. M. Cheung, Iulia Chifu, Ricardo Gafeira]
year: 2023
journal: "Living Reviews in Solar Physics"
doi: "10.1007/s41116-023-00038-x"
topic: Living_Reviews_in_Solar_Physics
tags: [machine_learning, deep_learning, solar_physics, CNN, UNet, GAN, flare_prediction, Stokes_inversion, NLFFF, super_resolution]
status: completed
date_started: 2026-04-23
date_completed: 2026-04-23
---

# 83. Machine Learning in Solar Physics / 태양물리학에서의 머신러닝

---

## 1. Core Contribution / 핵심 기여

**English.** This review is the first Living Review dedicated to machine learning (ML) and deep learning (DL) in solar physics. It opens with the standard ML taxonomy — supervised, unsupervised, and reinforcement learning — and then builds up, layer by layer, from classical linear methods (PCA, fuzzy c-means, k-means, Hermite bases, RVMs, compressed sensing) to the full modern deep-learning stack (fully-connected networks, CNNs, RNN/LSTM, attention/Transformer, GNNs, activation functions, loss functions, backpropagation, initialization, regularization, residual blocks, specialized hardware). It surveys unsupervised deep models (SOM, t-SNE, autoencoders) and the generative-model zoo (GANs, VAEs, normalizing flows, DDPMs) before devoting the bulk of the paper to solar applications: segmentation of coronal holes, sunspots, and granulation; classification of solar events; flare and CME prediction with HMI/SHARP features evaluated via TSS/HSS; Stokes-profile inversion accelerated by up to 10⁵×; neural-field (NeF) 3-D coronal reconstruction and NLFFF extrapolation; image deconvolution (MOMFBD learners); and image-to-image models including EUV/magnetogram translation, velocity estimation (DeepVel, DeepVelU), super-resolution (Enhance), denoising (Noise2Noise, denoising GANs), desaturation, and farside imaging (FarNet, FarNet-II).

**한국어.** 본 논문은 태양물리학에 적용된 머신러닝(ML)·딥러닝(DL) 기법을 전면적으로 정리한 최초의 Living Review이다. 지도·비지도·강화학습의 분류 체계에서 시작하여 PCA·퍼지 c-means·k-means·Hermite 기저·RVM·압축센싱 같은 고전 선형 방법을 거쳐, 완전연결망·CNN·RNN/LSTM·어텐션/Transformer·GNN, 활성함수·손실함수·역전파·가중치 초기화·정규화·잔차블록·전용 하드웨어까지 현대 딥러닝 스택 전반을 체계적으로 소개한다. 또한 SOM·t-SNE·오토인코더 등 비지도 DL과 GAN·VAE·Normalizing Flow·DDPM 등 생성모델 네 가지 계열을 다룬 뒤, 태양물리 응용 — 코로나홀·흑점·granulation 분할, 태양 이벤트 분류, SHARP 기반 플레어·CME 예측(TSS/HSS 평가), Stokes 인버전 10⁵배 가속, 뉴럴필드(NeF) 기반 NLFFF·3D 코로나 재구성, MOMFBD 가속 디컨볼루션, EUV·자기도 번역, 속도장 추정(DeepVel/DeepVelU), 초해상도(Enhance), 디노이징(Noise2Noise, GAN 기반), desaturation, Farside 영상화(FarNet, FarNet-II) 등 — 을 폭넓게 검토한다.

---

## 2. Reading Notes / 읽기 노트

### Part I: Foundations (§1 Introduction, §1.1–1.3) / 기초 개념

**한국어.** 태양물리학은 관측 기반 과학으로, 실험 조건을 바꿀 수 없고 오직 관측만 가능하다. 20세기까지는 데이터가 작아 물리모델에 강한 귀납적 편향을 주는 *생성모델* $p(x,y)$가 주류였다. 그러나 SDO(Galvez et al. 2019: 6.5 TB 다운로드 필요), DKIST, EST의 도래로 하루 관측 데이터가 PB급에 이르러 *판별모델* $p(y|x)$ 기반 딥러닝이 필수가 되었다. ML은 지도학습(입력 $\mathbf{X}$와 타깃 $\mathbf{Y}$ 쌍으로 $f: X\to Y$ 학습), 비지도학습(타깃 없이 $\mathbf{X}$의 구조 탐색), 강화학습(에이전트의 누적 보상 극대화)으로 분류된다. 태양물리의 대부분 응용은 앞 두 부류에 속하며, 강화학습은 Nousiainen et al. 2022의 적응광학 응용 정도다.

**English.** Solar physics is observational. Pre-2000s datasets were small, so generative models $p(x,y)$ with strong physical priors were natural. With SDO (Galvez et al. 2019 requires 6.5 TB to download), DKIST, and EST pushing data rates to PB/day scales, the field has shifted to discriminative models $p(y|x)$ learned from data. ML subdivides into supervised learning (learn $f$ from pairs $(\mathbf{X},\mathbf{Y})$), unsupervised learning (structure in $\mathbf{X}$ alone), and reinforcement learning (cumulative-reward maximization). Reinforcement learning has had almost no solar-physics traction to date (Nousiainen et al. 2022 on adaptive optics is the exception).

Key supervised-learning setup (Eq. 1):
$$
f^{\#} = \arg\min_f L(f(\mathbf{X}), \mathbf{Y})
$$
For Gaussian residuals, the canonical loss is MSE:
$$
L_{\text{MSE}} = \mathbb{E}_{(x,y)\sim\mathcal{D}}\!\left[(y - f(x))^2\right]
$$

The example running throughout §1 is Stokes inversion: observe $\mathbf{x} = (I,Q,U,V)(\lambda)$ of size $4N_\lambda$; infer $\mathbf{y} = (B, \phi, \theta, T, \ldots)$. Classical iterative inversion gives pairwise mappings per pixel — inefficient. Supervised ML learns a *global* mapping $f: X \to Y$ once, applied in constant time per sample.

### Part II: Dimensionality and Linear Models (§2–§4) / 차원과 선형 모델

**한국어.** 고차원 공간에서는 단위 초구 vs. 초입방체 부피비가
$$
\frac{V_{\text{hypersphere}}}{V_{\text{hypercube}}} = \frac{\pi^d}{d\,2^{d-1}\,\Gamma(d/2)}
$$
로 급속히 0에 수렴한다(차원의 저주). 그러나 실제 데이터는 저차원 매니폴드에 위치하는 경우가 많다. Asensio Ramos et al. 2007c는 최대우도 추정으로 Fe I 스펙트럼 라인의 Stokes 프로파일이 실제로 $d \ll N_\lambda$의 내재 차원에 있음을 확인.

**PCA.** 관측행렬 $\mathbf{O} \in \mathbb{R}^{N_{\text{obs}} \times 4N_\lambda}$의 SVD:
$$
\mathbf{O} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\star
$$
top-$r$ 특이벡터로 재구성한 $\mathbf{O}_{\text{denoised}} = (\mathbf{O}\mathbf{V}')(\mathbf{V}')^\star$는 가우시안 노이즈를 효율적으로 제거. Skumanich & López Ariste (2002): Stokes I의 1·2·3번째 특이벡터가 각각 평균, 속도, 자기 splitting의 물리적 의미를 지님.

**k-means & spectral clustering.** 코로나홀 분할(Verbeeck et al. 2014 SPoCA), 스펙트럼 분류. Benvenuto et al. 2018은 희소성 제약 선형모델 + k-means 변형을 결합한 하이브리드로 플레어 분류 개선.

**Compressed sensing (§4.3).** $\ell_1$ 정규화로 희소 신호 복원. Asensio Ramos & López Ariste 2010, Cheung et al. 2019의 MUSE 해체.

**English.** High-dimensional data suffers the curse of dimensionality (Eq. 2 above), yet most solar data live on low-dimensional manifolds. Asensio Ramos et al. (2007c) quantified this via maximum-likelihood intrinsic-dimension estimation for Fe I Stokes profiles.

PCA emerges from the SVD of the stacked observation matrix. The first few singular vectors often admit physical interpretation (Skumanich & López Ariste 2002 showed mean profile, velocity, and magnetic splitting for Stokes I). Denoising is a truncated reconstruction. Other linear methods include Hermite-function bases (§4.1), relevance vector machines (§4.2), and compressed sensing (§4.3) for sub-Nyquist spectropolarimetric sampling and MUSE-type spectral decomposition (Cheung et al. 2019).

### Part III: Deep Neural Networks — Architectures and Training (§5) / 심층 신경망

**한국어.** 단일 뉴런은 $y_i = f\!\left(\sum_j w_j x_j + b_i\right)$로 McCulloch-Pitts (1943, #1)의 일반화. $L$-층 네트워크는
$$
\mathbf{y} = f^{(L)}_{\boldsymbol{\theta}^{(L)}} \circ \cdots \circ f^{(1)}_{\boldsymbol{\theta}^{(1)}}(\mathbf{x})
$$
Cybenko 1988의 보편근사 정리는 "충분한 뉴런이 있으면 어떤 연속함수도 근사 가능"을 보장한다.

**CNN (Eq. 25).** 2D 입력 $X \in \mathbb{R}^{C\times N\times N}$에 대해
$$
O_i = K_i * X + b_i, \qquad i=1,\ldots,M
$$
FCN 대비 파라미터 폭증 문제(FCN: $N = \sum N_i N_{i-1}$)를 가중치 공유로 해결. Shift-invariance(평행이동 불변) 확보. Pooling(maxpool, $N_{\text{sub}} \times N_{\text{sub}}$)로 공간해상도 축소.

**Transformer (Eq. 26–27).**
$$
\mathbf{V} = \mathbf{W}_V\mathbf{X}, \quad \mathbf{Q} = \mathbf{W}_Q\mathbf{X}, \quad \mathbf{K} = \mathbf{W}_K\mathbf{X}
$$
$$
\mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$
Transformer는 아직 태양물리에 본격 적용되지 않음 — 큰 블루오션(§7.10.6 Brown et al. 2022 예외).

**손실/옵티마이저.** 일반 손실은 $L(\mathbf{x},\mathbf{y},\{\boldsymbol{\theta}^{(i)}\},\{\mathbf{y}^{(i)}\})$. SGD 업데이트:
$$
\boldsymbol{\theta}_{i+1} = \boldsymbol{\theta}_i - \eta\,\nabla L_j(\boldsymbol{\theta}_i), \qquad j \in \{1,\ldots,n\}
$$
역전파는 체인룰 자동화 — Jacobian 행렬의 곱:
$$
\frac{\partial L}{\partial \boldsymbol{\theta}^{(1)}} = \frac{\partial L}{\partial \mathbf{u}}\cdot\frac{\partial \mathbf{u}}{\partial \mathbf{v}}\cdot\frac{\partial \mathbf{v}}{\partial \boldsymbol{\theta}^{(1)}}
$$
활성함수: tanh/sigmoid(포화 → vanishing gradient), ReLU $=\max(0,x)$(He et al. 2015의 Kaiming init), ELU.

**Bag-of-tricks (§5.4).** Xavier/Kaiming 초기화, augmentation(회전/반사), 가중치 감쇠 $L_{\text{reg}} = L + \lambda\|\boldsymbol{\theta}\|^2$, dropout, Batch Normalization
$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y_i = \gamma\hat{x}_i + \beta
$$
ResNet의 residual mapping $y = x + R(x)$ — 깊은 네트워크 학습 가능하게 함. 그리고 GPU/TPU — 단정밀도·반정밀도 가속.

**English.** A single neuron generalizes McCulloch-Pitts (1943). An $L$-layer net is a composition $f^{(L)}\circ\cdots\circ f^{(1)}$; Cybenko (1988) guarantees universal approximation. CNNs introduce weight sharing and pooling, drastically reducing parameter count while injecting shift-invariance. The Transformer attention mechanism (Vaswani et al. 2017) is a data-dependent reweighting of values via query-key similarities — still largely unexploited in solar physics. Training uses SGD with step $\eta$ on mini-batches; backprop computes gradients as products of layer Jacobians. Modern tricks: Xavier/Kaiming initialization, augmentation, weight decay (L2), dropout, batch normalization, residual blocks, and GPU/TPU acceleration (often half precision).

### Part IV: Unsupervised Deep Learning and Generative Models (§6) / 비지도 DL과 생성모델

**한국어.** *SOM*(Asensio Ramos 2007a — Mn I 분류), *t-SNE*(Panos & Kleint 2020 — Mg II 플레어 영역 구분), *Mutual information 추정*(Panos et al. 2021 — IRIS Mg II·C II 강결합 발견), *오토인코더*(Socas-Navarro 2005a의 AANN; Sadykov et al. 2021은 Mg II를 27배 압축; Díaz Baso et al. 2022는 Stokes I 인코딩 → 베이지안 인버전).

**생성모델 4대천왕 (Fig 6).**
- **GAN** (Goodfellow 2014): 생성자 $G(\mathbf{z})$ vs. 판별자 $D(\mathbf{x})$의 min-max.
  $$\min_G \max_D \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))]$$
  Kim et al. 2019: STEREO EUV → 자기도, Hale 극성 법칙 재현.
- **VAE** (Kingma & Welling 2014): ELBO
  $$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\mathrm{KL}}(q(z|x)\|p(z))$$
  Panos et al. 2021은 VAE로 Mg II quiet-Sun 학습 → flaring profiles 아웃라이어 검출.
- **Normalizing Flow** (Dinh 2014): 가역 $\mathbf{X} = f(\mathbf{Z})$로 밀도 변환
  $$q_{\mathbf{X}}(\mathbf{x}) = q_{\mathbf{Z}}(\mathbf{z})\prod_{i=1}^M \left|\det\!\left(\frac{\partial f_i(\mathbf{y}_i)}{\partial \mathbf{y}_i}\right)\right|^{-1}$$
  Díaz Baso et al. 2022: Bayesian Stokes 인버전 가속.
- **DDPM** (Ho et al. 2020): 노이즈 추가 → 역과정 학습. 아직 태양 응용 없음.

**English.** SOM, t-SNE, AEs, and mutual-information networks provide unsupervised structure. Four generative families dominate modern unsupervised DL. GANs train generator vs. discriminator adversarially (Kim et al. 2019 generated farside magnetograms obeying Hale's law). VAEs optimize the evidence lower bound (ELBO) and give a tractable latent posterior (Panos et al. 2021 used them as outlier detectors for flaring profiles). Normalizing flows use invertible transforms with tractable Jacobians (Díaz Baso et al. 2022 for accelerated Bayesian inversion). DDPMs are dominant in image generation elsewhere but still unapplied to solar data.

### Part V: Applications of Supervised Deep Learning (§7) / 지도 딥러닝의 응용

#### §7.1 Segmentation / 분할

**한국어.** Illarionov & Tlatov (2018)은 AIA/SDO 193 Å에서 U-Net으로 CH를 검출. 2385개의 Kislovodsk 수동 바이너리 맵으로 학습. Binary cross-entropy loss:
$$
L = -\sum_i y_i \log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)
$$
CHIMERA/SPoCA 비교 시 CH 면적 변화 상관계수 0.76. Baek et al. 2021은 SSD와 Faster R-CNN으로 2010-2019 데이터(12hr cadence, 흑점/CH 5085장, 프로미넌스 2926장) 학습. Díaz Castillo et al. 2022는 granulation 분할.

**English.** Illarionov & Tlatov (2018) pioneered U-Net for coronal-hole detection on 2385 Kislovodsk binary maps, achieving 0.76 correlation vs. CHIMERA. Baek et al. (2021) applied SSD and Faster R-CNN to 5085 CH images, 4383 sunspots, and 2926 prominences spanning 2010–2019 at 12-hour cadence.

**Uncertainty issue (Reiss et al. 2021):** 9 automatic CH detection methods (ASSA-CH, CHIMERA, CHORTLE, CNN193, CHRONNOS, SPoCA-CH, SYNCH) compared — results consistent at CH center but diverge at boundaries. Mean intensity, magnetic field, open flux all differ significantly. There is no community-agreed ground truth for "CH".

#### §7.2 Classification / 분류

**한국어.** Armstrong & Fletcher (2019): quiet-Sun, prominence, filament, sunspot, flare-ribbon 5-class CNN, 정확도 ≈99.9%. *Transfer learning*으로 파장 간 학습 가능. MacBride et al. (2021)은 Ca II 8542 Å 라인을 FCN으로 5-class 분할(순수 흡수 0 → 순수 방출 4).

**English.** Armstrong & Fletcher (2019) reached ≈99.9% accuracy on 5-class classification (quiet Sun, prominence, filament, sunspot, flare ribbon) and demonstrated transfer learning across wavelengths. MacBride et al. (2021) classified Ca II 8542 Å profile shapes into 5 sub-types via an FCN.

#### §7.3 Flare Prediction / 플레어 예측 (core section)

**한국어.** 문제 정의: "시각 $t_0$의 특징 $\mathbf{x}$로 $[t_0, t_0+\Delta t]$ 내 M/X-class 플레어 발생 여부 예측" (보통 $\Delta t = 24$ h).

**Physics-based baselines.** Falconer (2001), Leka & Barnes (2003)은 벡터 자기도의 current/twist/PIL 길이 등으로 선형 판별. Bobra & Couvidat (2015)이 HMI/SHARP + $n$-fold cross-validation으로 패러다임 확립.

**Metrics (Table 2).** Contingency table (TP/FP/FN/TN):
- Recall = TP/(TP+FN), Precision = TP/(TP+FP)
- False Alarm Rate FAR = FP/(FP+TN)
- **TSS** = Recall − FAR ∈ [−1,1], class-imbalance robust
- **HSS**$_1$ = Recall × (2 − Precision$^{-1}$)
- **HSS**$_2 = \tfrac{\text{TP}+\text{TN}-E}{\text{TP}+\text{TN}+\text{FP}+\text{FN}-E}$

**Climatological baseline.** X-class 기저율 $p = 0.0001$일 때 Accuracy $= p^2 = 10^{-8}$(터무니없이 낮아 보이지만), Rate Correct $= p^2 + (1-p)^2 \approx 0.9802$(너무 관대). TSS = 0은 불편성(unbiased)을 드러냄.

**Deep learning approaches.**
- Huang et al. (2018): MDI/HMI LoS magnetogram → CNN 바이너리 분류. 중간층이 PIL-민감 공간 필터를 학습함을 해석.
- Nishizuka et al. (2018): "hand-engineered" 피처 + CNN.
- Liu et al. (2019): 25개 SHARP + 15개 flare history → LSTM.
- Yi et al. (2023): 강화학습으로 CNN 훈련 — 희귀 이벤트에 유리.
- Grad-CAM (Yi et al. 2021): PIL 영역에 강한 attention. Panos et al. (2023): Mg II 스펙트럼의 triplet emission·flow·broadening이 사전 특징.

**English.** Flare prediction is posed as binary classification: given features $\mathbf{x}$ at $t_0$, does an M/X-class flare occur within $\Delta t$ (typically 24 h)? Bobra & Couvidat (2015) standardized the HMI/SHARP dataset with $n$-fold cross-validation. Evaluation centers on Recall, Precision, FAR, and the imbalance-robust True Skill Statistic (TSS = Recall − FAR). For the climatological X-class base rate ($p=10^{-4}$): Accuracy $= p^2 \approx 10^{-8}$ (misleadingly low), Rate Correct $\approx 0.98$ (misleadingly high), TSS $= 0$ exposing the unbiased truth. Deep models (Huang 2018 CNN, Liu 2019 LSTM on SHARP + flare history, Yi 2023 RL-trained CNN) now dominate. Grad-CAM analyses confirm CNNs attend to polarity inversion lines — the physically expected feature.

#### §7.5 Heliosphere & Space Weather / 태양권·우주기상

**한국어.** Torres et al. 2022: FCN으로 10 MeV 이상 SEP flux ≥10 cm⁻²s⁻¹sr⁻¹ 예측. Upendran et al. 2020: SDO/AIA EUV 시계열 → GoogLeNet 기반 전이학습으로 L1 태양풍 속도 예측. Brown et al. 2022: 30min 샘플링 + attention-based → RMSE 크게 개선. Bernoux et al. 2022: AIA 193 Å → 확률적 Kp(평균+표준편차) 예측.

**English.** Torres et al. (2022) used FCNs for SEP flux prediction (>10 MeV, ≥10 cm⁻²s⁻¹sr⁻¹). Upendran et al. (2020) and Brown et al. (2022) predicted L1 solar wind speed from SDO/AIA EUV sequences, with transfer learning from GoogLeNet and attention mechanisms improving performance at 30-min cadence. Bernoux et al. (2022) produced probabilistic Kp (mean ± σ) from AIA 193 Å.

#### §7.6 Solar Cycle Prediction / 태양주기 예측

**한국어.** Nandy (2021)은 SC24에 대한 77건·SC25에 대한 37건의 예측을 분석 — *어떤 ML 모델도 진폭을 정확히 예측하지 못했다*. SC25 peak SSN 예측들(Fig 12):
- Li et al. 2021 (auto-regressive NN + LSTM): SSN ≈ 175
- Prasad et al. 2022: +20% vs. SC24, peak 2023-08, SSN 171.9 ± 3.4
- Wang et al. 2021b: peak SSN ≈ 114 in 2023
- Bizzarri et al. 2022 (Bayesian multistep NN): −14% vs. SC24, peak 2024-mid
- Benson et al. 2020 (WaveNet + LSTM): peak SSN 106 ± 19.75
- Okoh et al. 2018 (regression+NN): 112.1 ± 17.2, January 2025 ±6mo
- Dani & Sulistiani 2019 (4 ML): lin reg 159.4 ± 22.3 (Sep 2023); RF 110.2 ± 12.8 (Dec 2024); RBF 95.5 ± 21.9 (Dec 2024); SVM 93.7 ± 23.2 (Jul 2024)
- Covas et al. 2019 (FCN): lowest amplitude

**English.** Nandy (2021) reviewed 77 SC24 and 37 SC25 predictions — *no ML model successfully captured the cycle amplitude*. SC25 predictions (Fig 12) span 60 to 175 peak SSN with peak times ranging from 2022 to late 2025, a striking illustration of ML epistemic uncertainty on nonlinear physical systems.

#### §7.7 Stokes Inversion / 스토크스 인버전

**한국어.** Carroll & Staude (2001): 최초의 ML Stokes 인버전, FCN + Milne-Eddington. Carroll & Kopf (2008): MHD 시뮬 기반 FCN으로 depth-stratified $T, v, \mathbf{B}$ 추론. 

**SICON (Asensio Ramos & Díaz Baso 2019).** Hinode SP용 CNN. Key result:
- 512×512 map을 GPU 1장에서 **200 ms**에 inversion (SIR 대비 ≈10⁵배 가속)
- CNN은 Hinode PSF를 deconvolve하는 효과(우주관측기는 PSF 일정)
- gas pressure, Wilson depression 같은 고전 어려운 양도 예측 가능

**SynthIA (Higgins et al. 2022).** HMI/SDO로 Hinode/SOT-SP급 벡터 자기도 생성. full-disk 적용.

**Milic & Gafeira (2020).** 1D CNN으로 3개 depth의 $T, v, \mathbf{B}$ inversion. SNAPI로 synthetic training → 실측 적용. $\sim 10^5$배 속도.

**Non-LTE inversion.** Chappell & Pereira (2022) SunnyNet: LTE→non-LTE departure coefficients 매핑(수소 원자). Vicente Arévalo et al. (2022): Ca II 흡수계에 GNN. 10³배 가속.

**Uncertainty (§7.7.2).** Osborne et al. (2019): Invertible Neural Network(INN) — forward $y=f(x)$와 inverse $x=g(y,z)$($z$는 잠재) 동시학습. RADYN 모델에서 $T, n_e, v$ 사후확률 추정. Díaz Baso et al. (2022): Normalizing Flow로 Fe I·Ca II 인버전 posterior를 in-situ 생성.

**English.** ML Stokes inversion dates to Carroll & Staude (2001). The modern landmark is SICON (Asensio Ramos & Díaz Baso 2019): a CNN inverting a **512×512 Hinode/SP map in 200 ms on an off-the-shelf GPU**, a ≈10⁵× speed-up versus SIR. Side benefit: the CNN deconvolves the (constant) Hinode PSF and predicts quantities notoriously hard for classical inversion (gas pressure, Wilson depression). Higgins et al. (2022) scaled this idea to SynthIA, emulating Hinode/SOT-SP quality across the HMI full disk. For non-LTE lines, Chappell & Pereira (2022) use a CNN to map LTE populations to non-LTE departure coefficients (hydrogen), and Vicente Arévalo et al. (2022) use GNNs for Ca II departure coefficients, accelerating inversions by 10³. Uncertainty quantification uses INNs (Osborne 2019) and normalizing flows (Díaz Baso 2022) to produce full posteriors.

#### §7.8 3D Coronal Reconstruction — Neural Fields / 코로나 3D 재구성

**한국어.** 뉴럴필드(NeF)는 $\mathbb{R}^3 \to \mathbb{R}^d$의 MLP 함수:
$$
\log N_e(\mathbf{x}) = \phi_n \circ \phi_{n-1} \circ \cdots \circ \phi_0(\mathbf{x}), \quad \phi_i = \sigma(\mathbf{W}_i\mathbf{x}_i + \mathbf{b}_i)
$$
고주파 스펙트럴 바이어스 문제 → Fourier feature mapping 또는 **SIREN**(Sitzmann 2020): $\phi_i = \sin(\omega_i(\mathbf{W}_i\mathbf{x}_i + \mathbf{b}_i))$.

**Jarolim et al. 2022 — NLFFF via NeF.** 코로나 자기장 $\mathbf{B}(\mathbf{x})$를 NeF로 표현. 손실:
$$
L_{\text{ff}} = \frac{\|(\nabla \times \mathbf{B}) \times \mathbf{B}\|^2}{\|\mathbf{B}\|^2 + \epsilon}, \qquad L_{\text{div}} = \|\nabla \cdot \mathbf{B}\|^2
$$
공간미분은 자동미분으로 계산 → 경계조건과 함께 최적화. **PINN의 정확한 예.**

**Bintsi et al. 2022.** 단일 관측 ray-tracing 데이터만으로 전체 3D 코로나 방출성질 재구성. 32개 관측이면 적도 영역 충분.

**Rahman et al. 2023.** GAN으로 자기도→전자밀도 $N_e(\mathbf{x})$ 매핑(MAS 시뮬 학습).

**English.** Neural fields (NeFs) parameterize 3D fields via coordinate-MLPs. Spectral bias (Rahaman 2019) motivates Fourier features or sinusoidal activations (SIREN, Sitzmann 2020). Jarolim et al. (2022) provide a classic **physics-informed** use: represent $\mathbf{B}(\mathbf{x})$ as a NeF, add force-free and solenoidal loss terms computed via autodiff, optimize jointly with boundary conditions — this is a textbook PINN. Bintsi et al. (2022) recover 3D emission properties of the whole corona from EUV ray-tracing; Rahman et al. (2023) use GANs for magnetogram→density mapping.

#### §7.9 Image Deconvolution / 영상 디컨볼루션

**한국어.** Asensio Ramos et al. (2018): MOMFBD 결과를 타깃으로 supervised CNN 학습. 학습 후 7-frame burst of 1k×1k 이미지를 **5 ms**에 deconvolve. Asensio Ramos & Olspert (2021): unsupervised 확장, 웨이브프론트도 추정.

**English.** Asensio Ramos et al. (2018) trained a fully convolutional deep net on MOMFBD outputs; after training, it deconvolves 7-frame 1k×1k bursts in **5 ms**. The follow-up (2021) went unsupervised and also estimates the wavefront, useful for AO diagnostics.

#### §7.10 Image-to-Image Models / 영상-영상 변환

**7.10.1 Synthetic data.** Szenicer et al. (2019): AIA 7-channel → EVE MEGS-A line irradiance via CNN. DEM baseline 대비 RMSE 개선. Kim et al. (2019): cGAN으로 STEREO EUV → magnetogram. Salvatelli et al. (2019, 2021, 2022): U-Net·cGAN으로 AIA 채널 간 변환.

**7.10.2 Velocity — DeepVel / DeepVelU.**
- *DeepVel* (Asensio Ramos et al. 2017): quiet-Sun 시뮬 학습, 2-frame 입력, pixel-level horizontal velocity. LCT가 놓치는 수백 km 스케일 vortex 탐지.
- Tremblay et al. (2018): SDO/HMI에서 DeepVel > LCT at small scales.
- *DeepVelU* (Tremblay & Attie 2020): U-Net 기반, multiscale velocity(granular → supergranular).

**7.10.3 Super-resolution — Enhance (Díaz Baso & Asensio Ramos 2018).**
- HMI 0.5″/pix → **0.25″/pix (×2 spatial, ×4 pixel)** — effectively mapping HMI quality toward Hinode/DKIST-like sampling.
- Cross-check: Hinode images degraded to 0.5″ and enhanced back match originals. Continuum contrast improved by ≈×2.
- 자기도도 super-resolve되나 아티팩트 가능.
- Dou et al. (2022): HMI → MDI resolution 다운스케일 GAN + 실제 MDI → 초해상 GAN (2단계 GAN).

**7.10.4 Denoising.**
- Díaz Baso et al. (2019): Noise2Noise 접근(Lehtinen 2018) — 같은 위치 두 noise 실현만으로 denoiser 학습. CRISP/SST 편광계 적용, 시스템 아티팩트 제거.
- Park et al. (2020): conditional GAN, SDO/HMI 단일 자기도 → 앞뒤 10개 평균 타깃. Noise STD **×4.6 감소**(단, 약간의 blur 수반).

**7.10.5 Desaturation (Yu et al. 2022).** SDO/AIA의 플레어에 의한 포화를 partial convolution U-Net + PatchGAN으로 inpainting.

**7.10.6 Farside imaging.**
- Kim et al. (2019): STEREO EUV → farside 자기도. Polarity 재현 가능.
- *FarNet* (Felipe & Asensio Ramos 2019): helioseismic holography + nearside 자기도 probability를 CNN으로 결합. 약한 AR도 검출.
- Broock et al. (2021): FarNet이 표준 대비 **≈47% 더 많은 참 양성**.
- *FarNet-II* (Broock et al. 2022): attention + ConvLSTM(Shi 2015)로 시간적 일관성 개선.

**English.** Image-to-image models span synthetic data generation (AIA→MEGS-A irradiance via CNN; Szenicer 2019), velocity estimation (DeepVel with 2-frame input, DeepVelU U-Net for multiscale flows), super-resolution (Enhance mapping HMI 0.5″/pix to 0.25″/pix, a ≈×4 pixel-count increase with continuum contrast improved ≈×2 vs. Hinode benchmarks), denoising (Noise2Noise for CRISP; conditional GANs reducing SDO/HMI magnetogram noise STD by ≈×4.6), flare desaturation (inpainting), and farside detection (FarNet-II with attention+ConvLSTM improving weak AR detection by ≈47%).

### Part VI: Outlook (§8) / 전망

**한국어.** 현재는 ML을 *역문제를 매개변수화하는 편리한 방법*으로 사용하지만, 본 리뷰는 다음 단계를 **물리법칙-내재 DL**(physics-informed DL)로 제시한다. PINN·NeF·INR·GNN이 MHD 가속·자기장 외삽·관측 동화의 표준이 될 전망. Transformer·DDPM은 태양 이미지·스펙트럼 사전확률로 부상. 강화학습은 적응광학·관측 계획에 유망.

**English.** The paper argues the next phase is *physics-informed deep learning*: PINNs, NeFs, INRs, and GNNs integrated with MHD solvers, NLFFF extrapolation, and data assimilation. Transformers and DDPMs are emerging priors; RL will matter for adaptive optics and observation planning.

---

## 3. Key Takeaways / 핵심 시사점

1. **The ML-solar coupling is now structural, not cosmetic / ML-태양물리 결합은 이제 필수 구조이다.** — *English:* With DKIST (80 GB/hr) and EST pushing PB-class daily data rates, classical pixel-by-pixel inversion and manual feature detection are physically impossible. ML isn't a "bonus" — it is the only scalable pipeline. *한국어:* DKIST(80 GB/hr)와 EST로 인해 하루 PB급 관측이 도래하며, 기존의 픽셀 단위 인버전이나 수동 특징 검출은 물리적으로 불가능해졌다. ML은 보조가 아니라 유일한 확장 가능한 파이프라인이다.

2. **Stokes inversion is the flagship success (10⁵× speed-up) / 스토크스 인버전은 대표 성공사례.** — *English:* SICON inverts 512² Hinode/SP maps in **200 ms** vs. minutes-to-hours for SIR. The CNN also deconvolves PSF and predicts gas pressure/Wilson depression — side benefits classical codes cannot offer. *한국어:* SICON은 Hinode/SP 512² 맵을 **200 ms**에 처리(SIR은 수분~수시간). PSF 디컨볼루션과 gas pressure/Wilson depression 예측이라는 부가 이점까지 제공.

3. **TSS is the right flare-prediction metric / TSS가 플레어 예측의 올바른 지표.** — *English:* Accuracy and Rate Correct are dominated by class imbalance (climatological X-flare model: Rate Correct ≈ 0.98, TSS = 0). HSS partially corrects. TSS = Recall − FAR is imbalance-invariant and is the community-preferred metric. *한국어:* Accuracy·Rate Correct는 클래스 불균형에 지배됨(X-플레어 기후평균: Rate Correct ≈ 0.98이나 TSS = 0). HSS는 부분 보정. TSS는 불균형 불변으로 커뮤니티 표준.

4. **Neural Fields realize physics-informed DL / 뉴럴필드가 PINN의 대표 실현.** — *English:* Jarolim et al. (2022) represent $\mathbf{B}(\mathbf{x})$ as a coordinate MLP and enforce force-free + solenoidal conditions via autodiff loss terms — a textbook PINN. NeFs have strong implicit biases favoring smooth fields, enabling sub-pixel coronal reconstruction. *한국어:* Jarolim 등은 $\mathbf{B}(\mathbf{x})$를 MLP로 나타내고 force-free와 div-free 조건을 자동미분 손실로 부과 — 전형적 PINN. NeF는 부드러운 필드에 강한 내재 편향이 있어 서브픽셀 코로나 재구성에 유리.

5. **Generative models are underutilized in solar physics / 생성모델은 태양물리에서 아직 미활용.** — *English:* VAEs, normalizing flows, and especially DDPMs have transformed general computer vision but have minimal solar-physics footprint. Opportunity area: learned priors for ill-posed inverse problems (inversion, deconvolution, farside imaging). *한국어:* VAE·Normalizing Flow·DDPM은 일반 CV를 변혁했으나 태양물리에선 미미. 역문제(인버전·디컨볼루션·Farside)의 학습 사전확률로 큰 기회.

6. **Segmentation lacks ground truth / 분할은 정답의 부재가 문제.** — *English:* Reiss et al. (2021) showed nine automated CH methods produce strongly divergent boundaries. Without a community-agreed CH definition, supervised learning just encodes the labeling school's bias. A curated multi-observer training set (a "solar ImageNet") is a pressing need. *한국어:* Reiss 등은 9개 CH 자동 검출법의 경계가 크게 발산함을 보였다. CH 정의의 합의가 없으면 지도학습은 라벨링 학파의 편향을 재생산할 뿐. 다자 관측자 학습셋(태양 ImageNet)이 긴급.

7. **Solar Cycle 25 reveals ML epistemic limits / SC25 예측이 ML의 인식론적 한계를 드러냄.** — *English:* The 12+ SC25 peak-SSN predictions in Fig 12 span ~60 to ~175 and peak times from 2022 to 2025. Nandy (2021) notes **no ML model** correctly predicted SC25 amplitude. Nonlinear chaotic physics outruns data-driven priors trained on ≈25 cycles. *한국어:* SC25 peak SSN 예측은 ~60부터 ~175까지, peak 시점도 2022~2025로 광범위 산재. Nandy는 **어떤 ML 모델도** SC25 진폭을 제대로 맞추지 못했다고 평가. 25개 주기뿐인 데이터로는 비선형 혼돈계 예측에 한계.

8. **Transfer learning and explainability are now first-class citizens / 전이학습과 설명가능성은 이제 필수.** — *English:* Armstrong & Fletcher (2019) showed a CNN trained on one wavelength transfers to another. Grad-CAM (Yi et al. 2021) verifies flare-prediction CNNs attend to PILs, validating physical priors. Bernoux et al. (2022) output probabilistic Kp — moving from point predictions to calibrated posteriors. *한국어:* Armstrong & Fletcher는 파장 간 전이학습을 시연. Grad-CAM은 플레어 CNN이 PIL에 주목함을 검증(물리적 사전확률과 일치). Bernoux 등은 확률적 Kp 예측으로 점예측에서 교정된 사후분포로 이동.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Supervised-learning loss function / 지도학습 손실함수

The generic form (Eq. 1):
$$
f^{\#} = \arg\min_f L(f(\mathbf{X}), \mathbf{Y})
$$
- $f: X \to Y$ parametrized by $\boldsymbol{\theta}$ (NN weights/biases).
- Canonical MSE under Gaussian noise:
$$
L = \mathbb{E}_{(x,y)\sim\mathcal{D}}\!\left[(y - f(x; \boldsymbol{\theta}))^2\right]
$$
- Binary cross-entropy (segmentation / classification):
$$
L_{\text{BCE}} = -\sum_i y_i \log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)
$$

### 4.2 Gradient descent and backpropagation / 경사하강과 역전파

SGD step with learning rate $\eta$ (Eq. 32):
$$
\boldsymbol{\theta}_{i+1} = \boldsymbol{\theta}_i - \eta \nabla L(\boldsymbol{\theta}_i)
$$
Mini-batch variant (Eq. 33):
$$
\boldsymbol{\theta}_{i+1} \approx \boldsymbol{\theta}_i - \frac{\eta}{B}\sum_{k=1}^B \nabla L_{jk}(\boldsymbol{\theta}_i)
$$
Two-layer backprop gradient (Eq. 37):
$$
\frac{\partial L}{\partial \boldsymbol{\theta}^{(1)}} = \frac{\partial L}{\partial \mathbf{u}}\cdot \frac{\partial \mathbf{u}}{\partial \mathbf{v}}\cdot \frac{\partial \mathbf{v}}{\partial \boldsymbol{\theta}^{(1)}}
$$
Three-layer (Eq. 43):
$$
\frac{\partial L}{\partial \boldsymbol{\theta}^{(1)}} = \frac{\partial L}{\partial \mathbf{u}}\cdot \frac{\partial \mathbf{u}}{\partial \mathbf{v}}\cdot\frac{\partial \mathbf{v}}{\partial \mathbf{w}}\cdot\frac{\partial \mathbf{w}}{\partial \boldsymbol{\theta}^{(1)}}
$$
— a chain of Jacobians computed by reverse-mode autodiff (PyTorch, JAX).

### 4.3 Convolution + pooling / 컨볼루션과 풀링

For input $X \in \mathbb{R}^{C\times N\times N}$ (Eq. 25):
$$
O_i = K_i * X + b_i, \quad i = 1, \ldots, M
$$
where $K_i \in \mathbb{R}^{C\times K\times K}$ is the $i$-th kernel tensor.

Maxpool with window $N_{\text{sub}} \times N_{\text{sub}}$:
$$
O[i,j] = \max_{(p,q)\in W_{i,j}} X[p,q]
$$

### 4.4 Attention and Transformer / 어텐션과 Transformer

Given input $\mathbf{X}$ (Eqs. 26–27):
$$
\mathbf{V} = \mathbf{W}_V \mathbf{X}, \quad \mathbf{Q} = \mathbf{W}_Q \mathbf{X}, \quad \mathbf{K} = \mathbf{W}_K \mathbf{X}
$$
$$
\mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$
The softmax normalization gives a probability distribution over positions; scaling by $\sqrt{d_k}$ prevents saturation as $d_k$ grows.

### 4.5 GAN min-max objective / GAN의 min-max 목적함수

$$
\min_G \max_D \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))]
$$
- $G: \mathcal{Z} \to \mathcal{X}$ generator; $D: \mathcal{X} \to [0,1]$ discriminator.
- Nash equilibrium: $D^*(x) = \tfrac{1}{2}$ for all $x$; $G$ matches $p_{\text{data}}$.

### 4.6 VAE ELBO / VAE의 ELBO

For latent $z$, encoder $q_\phi(z|x)$, decoder $p_\theta(x|z)$, prior $p(z) = \mathcal{N}(0,I)$:
$$
\log p_\theta(x) \geq \mathcal{L}_{\text{ELBO}}(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}\!\left[\log p_\theta(x|z)\right] - D_{\mathrm{KL}}\!\left(q_\phi(z|x)\,\|\,p(z)\right)
$$

### 4.7 Normalizing flow change of variables / 정규화 흐름 변수변환

With invertible $\mathbf{X} = f(\mathbf{Z})$, inverse $\mathbf{Z} = g(\mathbf{X})$ (Eq. 48):
$$
q_{\mathbf{X}}(\mathbf{x}) = q_{\mathbf{Z}}(g(\mathbf{x}))\left|\det\!\left(\frac{\partial g(\mathbf{x})}{\partial \mathbf{x}}\right)\right|
$$
Compositional (Eq. 50):
$$
q_{\mathbf{X}}(\mathbf{x}) = q_{\mathbf{Z}}(\mathbf{z})\prod_{i=1}^M \left|\det\!\left(\frac{\partial f_i(\mathbf{y}_i)}{\partial \mathbf{y}_i}\right)\right|^{-1}
$$

### 4.8 U-Net (architecture summary) / U-Net 구조 요약

- Encoder: repeated $[\text{Conv}(3\!\times\!3) \to \text{BN} \to \text{ReLU}]\times 2 \to \text{MaxPool}(2\!\times\!2)$, channel doubling at each stage $C \to 2C \to 4C \to 8C \to 16C$.
- Decoder: bilinear upsample + concatenate with skip connection + conv block, channel halving.
- Output: $1\times 1$ conv to target channel count (1 for binary segmentation).
- Loss: BCE (Eq. 51).

### 4.9 Batch normalization and L2 regularization / 배치 정규화와 L2 정규화

BN (Eq. 47):
$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta
$$
L2 weight decay (Eq. 46):
$$
L_{\text{reg}} = L + \lambda\|\boldsymbol{\theta}\|^2
$$

### 4.10 Physics-informed neural network loss / 물리정보 신경망 손실

For a general PDE $\mathcal{N}[u](\mathbf{x}) = 0$ with boundary conditions $\mathcal{B}[u](\mathbf{x}) = g(\mathbf{x})$:
$$
L_{\text{PINN}} = \underbrace{\frac{1}{N_d}\sum_{i=1}^{N_d}\|u_\theta(\mathbf{x}_i) - u_i^{\text{obs}}\|^2}_{\text{data loss}} + \lambda_r \underbrace{\frac{1}{N_r}\sum_{j=1}^{N_r}\|\mathcal{N}[u_\theta](\mathbf{x}_j)\|^2}_{\text{PDE residual}} + \lambda_b \underbrace{\frac{1}{N_b}\sum_{k=1}^{N_b}\|\mathcal{B}[u_\theta](\mathbf{x}_k) - g(\mathbf{x}_k)\|^2}_{\text{boundary}}
$$

NLFFF specialization (Eq. 58, Jarolim et al. 2022):
$$
L_{\text{ff}} = \frac{\|(\nabla \times \mathbf{B}) \times \mathbf{B}\|^2}{\|\mathbf{B}\|^2 + \epsilon}, \qquad L_{\text{div}} = \|\nabla \cdot \mathbf{B}\|^2
$$
Total loss: $L = L_{\text{data}} + \lambda_{\text{ff}} L_{\text{ff}} + \lambda_{\text{div}} L_{\text{div}} + \lambda_{\text{bc}} L_{\text{bc}}$.

### 4.11 Worked numerical examples / 수치 예시

**(a) Super-resolution ×4 (Enhance)**
- HMI native pixel: 0.5″ → target 0.25″ (2× per axis, **4× pixel count**).
- Continuum RMS contrast improvement: **≈×2** vs. unenhanced HMI.
- Benchmark: Hinode images degraded to 0.5″ then enhanced back, compared against originals.
- DKIST equivalent: 0.03″/pix, so Enhance closes only part of the gap — a DKIST-era super-resolution pipeline remains to be designed.

**(b) Flare prediction skill scores — worked case**
Suppose over 1000 days, base rate of M+ flares = 0.05 (50 events).
- Climatological forecast ($p = 0.05$):
  - Expected TP = $p \cdot 50 = 2.5$, FP = $p \cdot 950 = 47.5$
  - Recall = $2.5/50 = 0.05$, FAR = $47.5/950 = 0.05$
  - **TSS = Recall − FAR = 0**
- An ML model with Recall = 0.80, FAR = 0.10:
  - TSS = 0.80 − 0.10 = **0.70** (excellent)
- Another model memorizing imbalance: always "No flare":
  - Accuracy = $950/1000 = 0.95$ (misleading)
  - Recall = 0, TSS = 0

**(c) Stokes inversion speedup**
- Classical SIR: ~0.1–1 s/pixel → 512² map = 26214 s = 7.3 hours (low end 2.6 hours).
- SICON (Asensio Ramos & Díaz Baso 2019): 512² map in **200 ms**.
- Speedup = $(2.6 \times 3600)/0.2 \approx 4.7 \times 10^4$; reviews cite $\sim 10^5$ for SIR upper-bound.
- Milic & Gafeira 2020: separate 1D CNN with $\sim 10^5$ speedup over SNAPI.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1943 ── #1 McCulloch-Pitts ─┐
                             │
1957 ── Rosenblatt           │
                             │
1986 ── #6 Rumelhart (BP) ───┤
                             │ ML foundations
1988 ── Cybenko              │ (data-poor era)
                             │
1998 ── LeCun CNN            │
                             │
2001 ── Carroll & Staude FCN Stokes (first ML-solar inv.)
                             │
2009 ── ImageNet             │
                             │
2010 ── SDO launch           │
2012 ── AlexNet              │
                             │
2014 ── VAE, GAN, Adam       │
        Bobra & Couvidat     │ Deep-learning era
2015 ── U-Net, ResNet        │ (solar ML accelerates)
                             │
2017 ── Transformer          │
        DeepVel              │
2018 ── Enhance              │
        Illarionov CH U-Net  │
2019 ── SICON (10⁵× speedup)│
2020 ── DDPMs; DeepVelU      │
2021 ── Jarolim NeF NLFFF    │
2022 ── Díaz Baso NF inv.    │
                             │
2023 ── ★ This review ★       │ Consolidation
                             │
Future ─ PINN-MHD synergy    │
        DDPM solar priors    │ (#84 Kowalski stellar
        RL adaptive optics    │  flares 2024 next)
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| #1 McCulloch & Pitts 1943 | Original neuron model; Eq. 20 is its direct generalization | Foundation — every architecture in §5 descends from this |
| #6 Rumelhart et al. 1986 | Backpropagation (Eq. 32–45); enables all DNN training in §5.3 | Indispensable training algorithm still used universally |
| #52 Hathaway 2015 (Solar Cycle LRSP) | Reviewed in §7.6; SC predictions rely on ML SSN forecasts | Direct subject — Fig 12 summarizes 12+ ML SC25 predictions |
| #71 Charbonneau 2020 (Dynamo) | Nonlinear dynamo physics ML tries to surrogate | Sanity check on why ML fails on SC25 — chaotic system |
| #78 Cranmer (wind) | Solar wind driving; §7.5 covers Upendran 2020 L1 wind prediction | Complementary — physics vs. ML approaches to wind |
| #82 Green CMEs | CME physics; §7.3 flare prediction often conflated with CME prediction | Methodology overlap — CNN on SHARP for both |
| Bobra & Couvidat 2015 | HMI/SHARP + n-fold CV established flare-prediction benchmark | Gold-standard dataset for ML flare work |
| Asensio Ramos & Díaz Baso 2019 (SICON) | CNN Stokes inversion; 10⁵× speedup; full §7.7.1 landmark | Most cited applied-ML-solar result in the review |
| Jarolim et al. 2022 (NeF NLFFF) | Physics-informed neural field for coronal extrapolation | Concrete PINN use — Eq. 58 |
| Ronneberger et al. 2015 (U-Net) | Architecture in Fig 7; basis for Illarionov CH, Díaz Castillo granulation | Workhorse for every segmentation task in §7.1 |
| Vaswani et al. 2017 (Transformer) | Eq. 26–27 attention; Brown et al. 2022 attention-based wind predictor | Emerging but largely untapped in solar physics |
| He et al. 2016 (ResNet) | Residual mapping $y = x + R(x)$ (§5.4.5); used in every deep CNN | Enables very deep networks (100+ layers) |

---

## 7. References / 참고문헌

- Asensio Ramos, A., Cheung, M. C. M., Chifu, I., & Gafeira, R. (2023). Machine learning in solar physics. *Living Reviews in Solar Physics*, 20:4. https://doi.org/10.1007/s41116-023-00038-x
- Asensio Ramos, A., & Díaz Baso, C. J. (2019). Stokes inversion based on convolutional neural networks. *A&A*, 626, A102.
- Asensio Ramos, A., Requerey, I. S., & Vitas, N. (2017). DeepVel: Deep learning for the estimation of horizontal velocities. *A&A*, 604, A11.
- Asensio Ramos, A., de la Cruz Rodríguez, J., & Pastor Yabar, A. (2018). Real-time, multiframe, blind deconvolution of solar images. *A&A*, 620, A73.
- Bobra, M. G., & Couvidat, S. (2015). Solar flare prediction using SDO/HMI vector magnetic field data. *ApJ*, 798, 135.
- Broock, E. G., Felipe, T., & Asensio Ramos, A. (2021). Performance of solar far-side active region neural detection. *A&A*, 652, A132.
- Broock, E. G., Asensio Ramos, A., & Felipe, T. (2022). FarNet-II: An improved solar far-side active region detection method. *A&A*, 667, A132.
- Carroll, T. A., & Staude, J. (2001). The inversion of Stokes profiles with artificial neural networks. *A&A*, 378, 316–326.
- Cybenko, G. (1988). Approximation by superpositions of a sigmoidal function. *Math. Control Signals Systems*.
- Díaz Baso, C. J., & Asensio Ramos, A. (2018). Enhancing SDO/HMI images using deep learning. *A&A*, 614, A5.
- Díaz Baso, C. J., de la Cruz Rodríguez, J., & Danilovic, S. (2019). Solar image denoising with CNNs. *A&A*, 629, A99.
- Díaz Baso, C. J., Asensio Ramos, A., & de la Cruz Rodríguez, J. (2022). Bayesian Stokes inversion with normalizing flows. *A&A*, 659, A165.
- Goodfellow, I., et al. (2014). Generative adversarial networks.
- Higgins, R. E. L., et al. (2022). SynthIA: A synthetic inversion approximation for the Stokes vector fusing SDO and Hinode. *ApJS*, 259, 24.
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS.
- Illarionov, E. A., & Tlatov, A. G. (2018). Segmentation of coronal holes in solar disc images with a convolutional neural network. *MNRAS*.
- Jarolim, R., et al. (2022). Neural field extrapolation of the solar coronal magnetic field.
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. ICLR.
- LeCun, Y., & Bengio, Y. (1998). Convolutional networks for images, speech, and time series.
- McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity.
- Milic, I., & Gafeira, R. (2020). Machine learning approach to the Milne-Eddington inversion. *A&A*, 644, A129.
- Nandy, D. (2021). Progress in solar cycle predictions: Sunspot cycles 24–25 in perspective. *Solar Phys.*, 296, 54.
- Reiss, M. A., et al. (2021). Unveiling the uncertainty in coronal hole boundary maps. *ApJ*, 913, 28.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation.
- Sitzmann, V., et al. (2020). Implicit neural representations with periodic activation functions (SIRENs). NeurIPS.
- Szenicer, A., et al. (2019). A deep learning virtual instrument for monitoring extreme UV solar spectral irradiance.
- Tremblay, B., & Attie, R. (2020). Inferring plasma flows at granular and supergranular scales with DeepVelU. *Front. Astron. Space Sci.*
- Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
- Yi, K., et al. (2021/2023). Deep learning and reinforcement learning for solar flare prediction.
