---
title: "Pre-Reading Briefing: Machine Learning in Solar Physics"
paper_id: "83"
topic: Living_Reviews_in_Solar_Physics
date: 2026-04-23
type: briefing
---

# Machine Learning in Solar Physics: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Asensio Ramos, A., Cheung, M. C. M., Chifu, I., & Gafeira, R. (2023). Machine learning in solar physics. *Living Reviews in Solar Physics*, 20:4. https://doi.org/10.1007/s41116-023-00038-x
**Author(s)**: Andrés Asensio Ramos, Mark C. M. Cheung, Iulia Chifu, Ricardo Gafeira
**Year**: 2023

---

## 1. 핵심 기여 / Core Contribution

**한국어.** 본 리뷰는 태양물리학에서 머신러닝(ML)과 딥러닝(DL)의 광범위한 적용을 체계적으로 정리한 최초의 Living Review이다. 지도·비지도·강화학습의 기본 개념에서 시작해 PCA, k-means, 퍼지 클러스터링 같은 고전 선형 방법부터 CNN, U-Net, GAN, VAE, Normalizing Flow, Neural Field(NeF)까지 현대 DL 아키텍처를 개괄한다. 응용 측면에서는 코로나 홀·흑점·AR의 분할(segmentation)과 분류, 플레어 예측(SHARP 기반 CNN·LSTM과 TSS/HSS 지표), Stokes 프로파일 인버전 가속(10⁵배 속도 향상), NLFFF 외삽을 위한 NeF, 이미지 초해상도(HMI→Hinode급, 0.5″→0.25″)와 디컨볼루션(DeepVel, Enhance), 합성 자기도·Farside 영상(FarNet-II), 우주기상(Kp, 태양풍 속도)과 태양주기 SC25 예측까지 폭넓게 다룬다.

**English.** This is the first comprehensive Living Review systematically mapping machine learning (ML) and deep learning (DL) onto solar physics. It starts from the core trichotomy of supervised/unsupervised/reinforcement learning, walks through the classical linear toolkit (PCA, k-means, fuzzy clustering, compressed sensing, RVMs), then covers modern deep architectures (FCNs, CNNs, RNN/LSTMs, attention/Transformers, GNNs, U-Nets, GANs, VAEs, normalizing flows, DDPMs, and neural fields). On the applications side it surveys segmentation and classification of solar images, flare prediction (HMI/SHARP CNN and LSTM models, TSS/HSS evaluation, class-imbalance issues), Stokes-profile inversion acceleration (≈10⁵ speed-up, SICON 512×512 maps in 200 ms), 3D coronal reconstruction via neural fields (NLFFF via force-free + solenoidal NeF losses), image-to-image models (DeepVel/DeepVelU, Enhance super-resolution, Noise2Noise denoising, desaturation GANs, FarNet/FarNet-II), and heliospheric/space-weather predictions (Kp, solar wind, SC25 SSN).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

**한국어.** 20세기 말까지 태양물리 데이터는 크기가 작아 *물리 모델 기반의 생성적 추론*(강한 귀납적 편향 + 소규모 데이터)이 주류였다. 2010년대 SDO(2010), Hinode, IRIS, GOES/SHARP, 그리고 DKIST·EST 같은 차세대 관측기가 도입되면서 하루 관측량이 PB 수준으로 폭증했고, 사람이 일일이 조사하기 불가능한 규모가 되었다. 동시에 ImageNet(2009), AlexNet(2012), GPU, TensorFlow(2015)·PyTorch(2019)·JAX(2018)로 대표되는 *딥러닝 혁명*이 진행되었다. 이 두 흐름이 만나 태양물리는 생성적 모델에서 *판별적(discriminative) 모델* — 즉 $p(y|x)$를 직접 학습 — 으로 전환되고 있다. 본 리뷰는 2023년 시점에서 이 전환의 전경을 집대성한다.

**English.** Before the 2010s solar data volumes were modest; research relied on *physics-driven generative models* that compensated for small data with strong inductive biases. SDO (2010), Hinode, IRIS, and the upcoming DKIST/EST pushed the community into the PB-per-day regime, making manual inspection impossible. Simultaneously the deep-learning revolution — ImageNet (2009), AlexNet (2012), GPUs/TPUs, and frameworks like TensorFlow, PyTorch, and JAX — matured. The intersection is driving solar physics toward *discriminative models* that directly learn $p(y|x)$ from observations. This review captures a snapshot of that transition as of 2023.

### 타임라인 / Timeline

```
1940s ── McCulloch-Pitts neuron (#1)
1957 ─── Rosenblatt perceptron
1980s ── Backpropagation (Rumelhart 1986, #6)
1988 ─── Cybenko universal approximation theorem
1998 ─── LeCun CNN / LeNet
2001 ─── Carroll & Staude: FCN Stokes inversion (first ML solar inv.)
2005 ─── Socas-Navarro: autoencoders (AANN) for spectropolarimetry
2010 ─── SDO launch, GPU deep learning takes off
2014 ─── Bobra & Couvidat HMI/SHARP flare prediction
2015 ─── U-Net (Ronneberger), ResNet (He)
2017 ─── Transformer (Vaswani); DeepVel (Asensio Ramos)
2018 ─── Enhance super-resolution; Illarionov & Tlatov CH U-Net
2019 ─── SICON Stokes inversion (200 ms/512², ×10⁵)
2020 ─── DDPMs; DeepVelU multiscale velocities
2021 ─── NeF NLFFF extrapolation (Jarolim); FarNet-II for farside
2023 ─── ★ This review consolidates a decade of progress ★
Future ── RL for adaptive optics; DDPM priors for inversions;
          Transformer attention for spectrograms
```

---

## 3. 필요한 배경 지식 / Prerequisites

**한국어.** (1) 선형대수: SVD, 고유값분해, 행렬미분. (2) 확률·통계: 베이즈 정리, 사후분포, 최대우도(MLE)/MAP, 가우시안 노이즈 가정. (3) 최적화: 경사하강, 확률적경사하강(SGD), Adam, 체인룰. (4) 신호처리: 컨볼루션, 푸리에 기저, 주파수 응답. (5) 태양물리 배경 — Stokes IQUV 편광분광, 자기광학효과(Zeeman/Hanle), HMI SHARP 데이터 형식, 플레어 GOES 등급, NLFFF 외삽, 활동영역·흑점·코로나홀 구조. (6) Python/NumPy/Matplotlib(가능하면 PyTorch·scikit-learn) 실습 경험.

**English.** (1) *Linear algebra*: SVD, eigendecomposition, matrix calculus. (2) *Probability*: Bayes' theorem, priors/posteriors, MLE/MAP, Gaussian noise models. (3) *Optimization*: gradient descent, SGD, Adam, chain rule. (4) *Signal processing*: convolution, Fourier bases. (5) *Solar physics domain*: Stokes IQUV spectropolarimetry, Zeeman/Hanle effects, HMI SHARP parameters, GOES flare classes, NLFFF extrapolation, active-region/sunspot/coronal-hole morphology. (6) Practical *Python/NumPy* familiarity, ideally with scikit-learn or PyTorch.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Supervised / Unsupervised / RL | $(X, Y)$ 입출력 쌍 학습 / 분포 $p(X)$만 학습 / 보상 누적 최대화. 태양물리 응용 대부분은 앞 두 부류. / Pairs vs. density vs. reward maximization. |
| Stokes inversion / 스토크스 인버전 | 관측된 편광 프로파일 $(I,Q,U,V)$로부터 $B, \theta, \phi, T, v$ 등 대기변수를 역추정. ML이 SIR 대비 10⁵배 가속. / Infer atmospheric parameters from polarimetric profiles; ML offers ≈10⁵× speed-up over SIR. |
| SHARP | Space-weather HMI Active Region Patch. 자기 twist·current·flux 등 수십 개의 "hand-engineered" 피처 포함. / HMI active-region patch data with flare-predictive magnetic features. |
| TSS / HSS | True Skill Statistic = Recall − FAR (클래스 불균형 robust); Heidke Skill Score는 랜덤 기준 상대 성능. / Recall minus false-alarm rate; Heidke score measures gain over random. |
| U-Net | 인코더-디코더 + skip-connection CNN; 의료·태양 분할의 표준. / Encoder-decoder CNN with skips, standard for dense segmentation. |
| GAN / VAE / NF / DDPM | 생성 모델 4대천왕: 적대적, ELBO, 역변환, 확산 역과정. / Four generative paradigms. |
| NeF / INR / CBR | Neural Field: 좌표→필드값 매핑(MLP). NLFFF·코로나 3D에 사용. / Coordinate-based MLP mapping $\mathbf{x} \mapsto f(\mathbf{x})$; used for NLFFF, coronal reconstruction. |
| PINN | 물리 정보 신경망: 손실함수에 PDE 잔차 항을 추가. / Physics-informed NN augmenting data loss with PDE residual. |
| Grad-CAM / XAI | 설명가능 AI: CNN 예측을 입력 영역에 투영 → PIL이 하이라이트됨을 확인. / Gradient-weighted class activation mapping for interpretability. |
| Transfer Learning | 타 도메인에서 사전학습된 가중치를 재사용. Armstrong & Fletcher 2019이 파장 간 전이를 시연. / Reuse of pretrained weights; demonstrated across solar wavelengths. |

---

## 5. 수식 미리보기 / Equations Preview

**(1) Supervised loss / 지도학습 손실.**
$$
f^{\#} = \arg\min_f L\big(f(\mathbf{X}), \mathbf{Y}\big), \qquad L_{\text{MSE}} = \mathbb{E}[(y - f(x))^2]
$$

**한국어.** 예측과 관측의 불일치를 스칼라로 압축한 뒤 파라미터 $\theta$에 대해 최소화. MSE는 가우시안 노이즈 가정에서 자연스럽다.
**English.** Scalar loss over $\theta$; MSE follows from a Gaussian residual assumption.

**(2) Neuron + backprop / 뉴런과 역전파.**
$$
y_i = f\!\left(\sum_j w_j x_j + b_i\right), \qquad
\frac{\partial L}{\partial \theta^{(\ell)}} = \frac{\partial L}{\partial \mathbf{u}}\,\frac{\partial \mathbf{u}}{\partial \mathbf{v}}\,\frac{\partial \mathbf{v}}{\partial \theta^{(\ell)}}
$$

**한국어.** 체인룰로 야코비안을 곱해 전 층의 그래디언트를 얻는다. 자동미분(PyTorch/JAX)이 이를 구현.
**English.** Backprop = product of Jacobians computed via reverse-mode autodiff.

**(3) Convolutional layer / 컨볼루션 층.**
$$
O_i = K_i * X + b_i, \qquad i = 1,\ldots,M
$$

**한국어.** 가중치 공유로 파라미터 수 급감, shift-invariance 확보.
**English.** Weight sharing → drastic parameter reduction and shift invariance.

**(4) Attention / 어텐션.**
$$
\mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

**한국어.** 시퀀스 요소 간 관련도를 데이터-의존적으로 재가중. 태양 스펙트로그램·플레어 예측에서 부상 중.
**English.** Data-dependent reweighting; emerging in spectrogram and flare applications.

**(5) Neural field for NLFFF / NLFFF용 뉴럴필드.**
$$
L_{\text{ff}} = \frac{\|(\nabla\times\mathbf{B})\times\mathbf{B}\|^2}{\|\mathbf{B}\|^2 + \epsilon}, \qquad
L_{\text{div}} = \|\nabla\cdot\mathbf{B}\|^2
$$

**한국어.** Force-free + solenoidal 조건을 자동미분으로 강제해 $\mathbf{B}(\mathbf{x})$를 MLP로 외삽.
**English.** Force-free and divergence-free residuals backpropagated through a coordinate MLP $\mathbf{B}(\mathbf{x})$.

---

## 6. 읽기 가이드 / Reading Guide

**한국어.**
1. **§1–3 (지도/비지도/차원·PCA)**: 기초. 빠르게 훑되 PCA-기반 Stokes 인버전 lookup table(§3.1.3)은 꼼꼼히. 
2. **§4 (선형 지도)**: Hermite 함수, RVM, 압축센싱은 참고용. 
3. **§5 (DNN)**: 아키텍처(FCN/CNN/RNN/Attention/GNN)와 Training(loss/SGD/backprop)을 중점. Bag-of-tricks(§5.4)은 실무에 유용. 
4. **§6 (비지도 DL)**: SOM/t-SNE/오토인코더/GAN/VAE/NF/DDPM — 각 모델의 *생성 원리*를 그림 6과 함께 이해. 
5. **§7 (지도 DL 응용 — 핵심!)**: 분할→분류→플레어(§7.3은 가장 길고 실무적)→스페이스 웨더→Stokes 인버전→NeF→디컨볼루션→이미지-이미지. 
6. **§8 (미래)**: 짧지만 향후 연구 방향 정리. 표/그림은 특히 Fig 6(생성모델 비교), Fig 8(CH 검출 9종), Fig 11(불균형-TSS), Fig 12(SC25 예측 분산)을 주의.

**English.**
1. *§1–3 (Supervised/unsupervised, dimensionality, PCA)* — basics; give the lookup-table inversion subsection a close read.
2. *§4 (Linear supervised)* — skim RVM/CS unless you need them.
3. *§5 (DNNs)* — focus on architectures and training; the "bag-of-tricks" section is practical.
4. *§6 (Unsupervised DL)* — learn the generative principle of each model family using Fig 6.
5. *§7 (Applications)* — the meat of the paper. Flare prediction (§7.3) is the longest and most operationally relevant. Stokes inversion (§7.7) and image-to-image (§7.10) showcase the largest performance wins.
6. *§8 (Outlook)* — short but flags the open frontiers.
Key figures: Fig 6 (generative model zoo), Fig 8 (CH detection disagreements), Fig 11 (TSS vs. imbalance), Fig 12 (SC25 prediction spread).

---

## 7. 현대적 의의 / Modern Significance

**한국어.** DKIST(0.03″, 80 GB/hr)와 EST가 2020년대 중반 풀가동에 접어들면 전통적 픽셀 단위 인버전은 사실상 불가능해진다. 본 리뷰가 정리한 CNN 인버전(SICON: 200 ms/512²), 이미지 초해상도(Enhance), 뉴럴필드 NLFFF(Jarolim 2022), 그리고 DDPM 사전확률이 현실적 파이프라인의 필수 요소가 되고 있다. 또한 플레어 예측은 NOAA/SWPC 운영에 실제 투입되고 있으며, SC25 예측은 머신러닝 모델의 불확실성을 드러냈다(Fig 12의 큰 분산). 향후 5년 내 *물리법칙-내재형 딥러닝*(physics-informed NN, NeF, INR)이 시뮬레이션 가속·역문제·관측자료 동화의 표준이 될 가능성이 높다.

**English.** As DKIST (0.03″, 80 GB/hr) and EST reach full operation, pixel-by-pixel classical inversion becomes infeasible and the techniques catalogued here — SICON-style CNN inversion (200 ms per 512² map), super-resolution (Enhance), coordinate-based NeF NLFFF, and generative priors — move from demonstrations to essential pipeline components. Operational flare forecasting already runs at NOAA/SWPC, and the SC25 prediction landscape (Fig 12) is a cautionary tale about ML epistemics. Over the next five years *physics-informed* deep learning (PINNs, NeFs, INRs with Fourier features/SIRENs) will likely become standard for simulation acceleration, inverse problems, and data assimilation in solar physics.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
