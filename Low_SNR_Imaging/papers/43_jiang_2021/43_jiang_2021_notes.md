---
title: "EnlightenGAN: Deep Light Enhancement without Paired Supervision"
authors: Yifan Jiang, Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, Xiaohui Shen, Jianchao Yang, Pan Zhou, Zhangyang Wang
year: 2021
journal: "IEEE Transactions on Image Processing, vol. 30, pp. 2340-2349"
doi: "10.1109/TIP.2021.3051462"
topic: Low_SNR_Imaging
tags: [low-light-enhancement, GAN, unpaired, attention, self-regularization, perceptual-loss]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 43. EnlightenGAN: Deep Light Enhancement without Paired Supervision / EnlightenGAN: 쌍 없는 학습으로 저조도 영상 향상

---

## 1. Core Contribution / 핵심 기여

EnlightenGAN은 **저조도/정상광 짝지어진 데이터(paired data) 없이도** 학습 가능한 GAN 기반 저조도 영상 향상 모델로, CycleGAN과 달리 **단방향(one-path) 매핑**만 학습한다 — cycle-consistency를 사용하지 않으므로 학습 비용이 크게 줄고 안정성이 향상된다. 핵심 세 가지 혁신은 (i) 공간적으로 변하는 조명을 다루기 위한 **global-local discriminator** 구조 (전체 이미지용 relativistic LSGAN + 랜덤 패치용 LSGAN PatchGAN), (ii) 입력의 휘도 채널 $I_Y$ 로부터 $1-I_Y$ 로 만들어지는 **self-regularized attention map**을 generator의 모든 레벨에 곱해 어두운 영역을 더 많이 향상시키는 메커니즘, (iii) ground-truth가 없으므로 입력과 출력 사이 VGG-feature 거리를 최소화하는 **self feature preserving (SFP) loss**다. 학습은 914장 저조도 + 1016장 정상광 unpaired 이미지로 진행되며, NIQE 점수와 인간 주관 평가, 다섯 개 공개 저조도 데이터셋에서 RetinexNet/LIME/SRIE/NPE/LLNet/CycleGAN을 일관되게 능가한다. 또한 BBD-100k 야간 운전 데이터셋에 도메인 적응(EnlightenGAN-N)이 가능함을 보여 unpaired 학습의 실용적 일반화를 입증한다.

EnlightenGAN is the first practical GAN-based low-light enhancement model that requires **no paired low/normal-light training data**, and unlike CycleGAN it uses only a **one-path mapping** — no cycle-consistency — which dramatically reduces training cost and stabilizes optimization. Three innovations make this work: (i) a **global-local discriminator** combining a relativistic LSGAN over the full image with an LSGAN PatchGAN over random crops to handle spatially-varying illumination; (ii) a **self-regularized attention map** built from the input luminance channel as $1 - I_Y$, multiplied into every level of the U-Net generator so that darker regions receive proportionally more enhancement; and (iii) a **self feature preserving (SFP) loss** that, lacking ground truth, instead constrains the VGG-feature distance between the input and the enhanced output, preserving content while changing only illumination. Trained on 914 low-light + 1016 normal-light unpaired images, the model beats RetinexNet, LIME, SRIE, NPE, LLNet, and CycleGAN on five public benchmarks (NIQE, human study) and adapts to the BBD-100k night-driving dataset (EnlightenGAN-N), demonstrating real-world generalization.

---

## 2. Reading Notes / 읽기 노트

### Part I: Introduction (pp. 1-2) / 서론

문제 정의: 저조도 사진은 대비가 낮고 noise(특히 ISO noise)가 크며 가시성이 떨어진다. 인간 시각과 자율주행·생체인식 등 모든-시간(all-day) 컴퓨터 비전 시스템 모두에 문제가 된다 (p.1).

Problem statement: low-light photos suffer from low contrast, high ISO noise, and poor visibility — a challenge for both human perception and all-day computer-vision systems including autonomous driving and biometrics (p.1).

저자들이 paired supervision의 본질적 한계를 명확히 짚는다 (p.1):

The authors identify three fundamental limits of paired supervision (p.1):

1. **노출 시간/ISO를 바꿔 같은 장면을 두 번 찍는 것은 매우 어렵고 비현실적** — 카메라 고정·움직이는 객체 없음 등 제약 / Hard to capture corrupted and clean version of same scene simultaneously
2. **합성 저조도는 photo-realistic하지 않다** — 실제 저조도 이미지에 적용하면 artifact / Synthesized corruption is rarely photo-realistic
3. **일반적으로 unique/well-defined ground-truth가 없다** — 새벽-자정 노출은 모두 "정상광"으로 볼 수 있다 / There may be no unique ground-truth normal-light image

저자들의 해법: CycleGAN 영감 + paired-free GAN 설계 + dual D + self-regularization (p.2).

The fix: CycleGAN-inspired one-path GAN with dual discriminator and self-regularization (p.2).

**Notable innovations claimed (p.2):**

- 첫 unpaired training for low-light / 첫 unpaired 저조도 향상
- Global-local D for spatially-varying illumination
- Self-regularized attention + self feature preserving loss
- Comprehensive benchmark across visual quality / NIQE / human study

### Part II: Related Works (pp. 2-3) / 관련 연구

**Paired datasets** (p.2): LoL [Wei+ 2018]은 500쌍에 그치고 노출시간 변화로 만든 인위적 쌍이라 한계가 있다. HDR 다중 노출 fusion은 후처리 목적이 아니다.

**Traditional approaches** (p.2): AHE, Retinex (Land), multi-scale Retinex (Jobson 1997), bi-log transformation (Fu 2014), weighted variational model, LIME (illumination map estimation, Guo+ 2017), joint denoise-enhance (Ren).

**Deep learning approaches** (p.2): LLNet (stacked AE, Lore+ 2017), RetinexNet (Wei+ 2018), HDR-Net (bilateral grid, Gharbi+), SID (raw-domain learning, Chen+ 2018).

**Adversarial learning** (p.3): Pix2Pix paired translation, CycleGAN unpaired translation. EnlightenGAN의 차별점: **one-path GAN, no cycle-consistency, lightweight, unpaired** (p.3 마지막 단락).

### Part III: Method Overview (p. 3, Fig 2) / 방법 개요

전체 구조 (Fig 2):

Overall architecture (Fig 2):

1. 입력 저조도 이미지 $I^L$
2. 휘도 채널 $I_Y$ 추출 → 정규화 $I_Y \in [0,1]$ → attention map $A = 1 - I_Y$
3. **Attention-guided U-Net generator**: 8 conv blocks, 각 블록은 두 개의 $3\times3$ conv + BN + LeakyReLU; upsample은 bilinear+conv (deconv 아님 → checkerboard 회피); 각 레벨 feature에 resize된 attention map을 element-wise 곱
4. 출력 $G(I^L)$ → **global discriminator** (전체 이미지) + **local discriminator** (5개 random crop)
5. **Self feature preserving loss**: $\mathcal L_{SFP}(I^L) = \|\phi_{5,1}(I^L) - \phi_{5,1}(G(I^L))\|_2^2 / (W H)$

### Part IV: Global-Local Discriminator (pp. 3-4) / 전역-지역 판별기

**Global discriminator**는 **relativistic discriminator** (Jolicoeur-Martineau 2018)를 사용하되 sigmoid를 LSGAN least-squares로 대체:

The global discriminator uses a **relativistic discriminator** (Jolicoeur-Martineau 2018) but replaces the sigmoid with LSGAN least-squares form:

$$
D_{Ra}(x_r, x_f) = \sigma\!\big(C(x_r) - \mathbb E_{x_f \sim \mathbb P_{\text{fake}}}[C(x_f)]\big),
$$

$$
D_{Ra}(x_f, x_r) = \sigma\!\big(C(x_f) - \mathbb E_{x_r \sim \mathbb P_{\text{real}}}[C(x_r)]\big).
$$

**Global D 손실 (LSGAN-relativistic, p.4 식 (3)-(4)):**

$$
\mathcal L_D^{\text{Global}} = \mathbb E_{x_r}\big[(D_{Ra}(x_r, x_f) - 1)^2\big] + \mathbb E_{x_f}\big[D_{Ra}(x_f, x_r)^2\big],
$$

$$
\mathcal L_G^{\text{Global}} = \mathbb E_{x_f}\big[(D_{Ra}(x_f, x_r) - 1)^2\big] + \mathbb E_{x_r}\big[D_{Ra}(x_r, x_f)^2\big].
$$

**Local discriminator**: PatchGAN, 5개 random crop 사용. LSGAN 손실 (p.4 식 (5)-(6)):

Local discriminator: PatchGAN over 5 random crops, plain LSGAN (eqs (5)-(6)):

$$
\mathcal L_D^{\text{Local}} = \mathbb E_{x_r}\big[(D(x_r) - 1)^2\big] + \mathbb E_{x_f}\big[D(x_f)^2\big],
$$

$$
\mathcal L_G^{\text{Local}} = \mathbb E_{x_f}\big[(D(x_f) - 1)^2\big].
$$

이 dual D 구조는 spatial-varying illumination에 대응하기 위함이다 — global만 쓰면 어두운 배경의 작은 밝은 영역을 잘못 학습한다 (Fig 3 ablation Row 3에서 "Without local D"가 색 왜곡을 보임).

The dual-D structure compensates for spatially-varying illumination — a global-only D mishandles small bright regions in dark backgrounds (Fig 3 row 3 "Without local discriminator" exhibits color distortion).

### Part V: Self Feature Preserving Loss (p. 4) / 자기 특징 보존 손실

기존 perceptual loss (Johnson+ 2016)는 **출력과 ground-truth** 사이 VGG distance다. unpaired 환경에선 GT가 없으므로 EnlightenGAN은 **입력과 출력** 사이 VGG distance를 사용한다 (p.4):

The classical perceptual loss measures VGG distance between **output and ground truth**. In an unpaired setting there is no GT, so EnlightenGAN measures VGG distance between **input and output** (p.4):

$$
\mathcal L_{SFP}(I^L) = \frac{1}{W_{i,j}H_{i,j}} \sum_{x=1}^{W_{i,j}}\sum_{y=1}^{H_{i,j}} \big(\phi_{i,j}(I^L) - \phi_{i,j}(G(I^L))\big)^2,
$$

여기서 $\phi_{i,j}$는 ImageNet pre-trained VGG-16의 $i$번째 maxpool 이후 $j$번째 conv 이후 feature ($i=5, j=1$). 이 손실은 enhancement 전후로 **content/구조를 보존**하도록 강제한다 — 동기는 "VGG classification은 입력의 픽셀 강도 변화에 둔감하다"는 경험적 관찰 (p.4).

where $\phi_{i,j}$ is the feature after the $j$-th conv following the $i$-th maxpool of an ImageNet-pretrained VGG-16 (defaults $i=5, j=1$). The loss preserves content/structure across enhancement, motivated by the empirical fact that VGG classification is largely invariant to global pixel-intensity rescaling (p.4).

Local 영역에도 동일하게 정의된 $\mathcal L_{SFP}^{\text{Local}}$ 적용. VGG feature 앞에 instance normalization을 두어 학습 안정화 (p.4 마지막 단락).

The same form applies to local crops as $\mathcal L_{SFP}^{\text{Local}}$. Instance normalization is added before the VGG features for training stability (p.4 last paragraph).

### Part VI: Total Loss (p. 4, 식 (8)) / 전체 손실

$$
\boxed{\;\text{Loss} = \mathcal L_{SFP}^{\text{Global}} + \mathcal L_{SFP}^{\text{Local}} + \mathcal L_G^{\text{Global}} + \mathcal L_G^{\text{Local}}.\;}
$$

가중치 없이 단순 합. global 분기가 전반적 톤·색을, local 분기가 부분 over/under exposure를, SFP가 구조 보존을 담당하는 균형 (p.4).

A simple unweighted sum: global terms govern tonal balance, local terms suppress over/under-exposure, SFP preserves structure (p.4).

### Part VII: U-Net Generator with Self-Regularized Attention (pp. 4-5) / 자기-정규화 attention U-Net

**Attention map 생성 (p.5):**

1. RGB 입력 → illumination channel $I$ 추출 (Y of YCbCr 또는 max(R,G,B))
2. $I$를 $[0,1]$로 정규화
3. element-wise $A = 1 - I$ → 어두운 픽셀일수록 attention 값 큼

**Generator**: 8 conv blocks, 각 블록 = $3\times3$ conv + BN + LeakyReLU 두 번. Upsample은 **bilinear + conv** (deconvolution 대신; checkerboard artifact 방지, p.5).

**Attention 적용**: 각 레벨 feature map에 attention map을 해당 해상도로 resize 후 element-wise 곱.

**Generator**: 8 convolutional blocks (two 3×3 conv + BN + LeakyReLU each). Upsampling uses **bilinear + conv** rather than deconvolution to avoid checkerboard artifacts (p.5). Attention is resized to each level's resolution and element-wise multiplied with the feature map.

이 attention은 학습되지 않고 입력에서 직접 만들어지는 self-regularization이다 — 외부 supervision이 없는 상황에서 어두운 영역을 더 많이 향상시키도록 유도한다 (p.5).

The attention is **not learned** — it is computed deterministically from the input as a form of self-regularization, biasing the network to enhance darker regions more (p.5).

### Part VIII: Dataset & Implementation (p. 5) / 데이터셋과 구현

- **Training set (unpaired)**: 914 low-light + 1016 normal-light from various datasets (no pairs needed)
- 이미지 600×400 PNG로 정규화
- **Test sets (5)**: NPE, LIME, MEF, DICM, VV (모두 prior 연구에서 사용된 표준)
- **Optimizer**: Adam, batch size 32
- **LR**: $1\text{e-}4$ for first 100 epochs, linearly decays to 0 over next 100 epochs
- 총 학습 시간 ≈ 3 hours on 3× 1080Ti
- LoL의 50쌍을 hold-out validation으로 사용 (training에는 사용 안 함)

### Part IX: Ablation Study (p. 6, Fig 3) / Ablation 연구

Fig 3 다섯 행:

Five rows of Fig 3:

1. Input (저조도 입력)
2. Attention map ($1 - I$ 시각화)
3. Without local D — 부분적 색 왜곡, 부적절한 노출
4. Without attention (vanilla U-Net) — 색 왜곡 또는 under-enhancement
5. Full EnlightenGAN — 시각적으로 가장 균형 잡힌 결과

→ local D와 attention 두 컴포넌트 모두 필수 / Both local D and attention are necessary.

### Part X: Quantitative Comparison (pp. 6-7, Tab I) / 정량 비교

**NIQE (lower is better, p.6 Tab I):**

| Image set | LLNet | CycleGAN | RetinexNet | LIME | SRIE | NPE | **EnlightenGAN** |
|---|---|---|---|---|---|---|---|
| MEF | 4.845 | 3.782 | 4.149 | 3.720 | 3.475 | 3.524 | **3.232** |
| LIME | 4.940 | 3.276 | 4.420 | 4.155 | 3.788 | 3.905 | **3.719** |
| NPE | 4.78 | 4.036 | 4.485 | 4.268 | 3.986 | 3.953 | 4.113 |
| VV | 4.446 | 3.343 | 2.602 | **2.489** | 2.850 | 2.524 | 2.581 |
| DICM | 4.809 | 3.560 | 4.200 | 3.846 | 3.899 | 3.760 | **3.570** |
| **All** | 4.751 | 3.554 | 3.920 | 3.629 | 3.650 | 3.525 | **3.385** |

5개 세트 중 3개에서 1위, 전체 평균에서 1위.

EnlightenGAN wins 3 out of 5 sets and the overall average.

### Part XI: Human Subjective Evaluation (p. 6-7, Fig 5) / 인간 주관 평가

23 testing images × 9 subjects × 5 methods, pair-wise ranking via Bradley-Terry. EnlightenGAN이 23장 중 10장에서 1위, 8장에서 2위 (Fig 5 histogram).

23 testing images × 9 subjects × 5 methods, pairwise ranking via Bradley-Terry model. EnlightenGAN ranks #1 on 10/23 images and #2 on 8/23 (Fig 5 histogram).

### Part XII: Domain Adaptation on BBD-100k (pp. 7-8, Fig 6) / 도메인 적응

- 950 night-time photos from BBD-100k (Berkeley Deep Driving) → low-light training set
- 1016 normal-light from original assembly → normal-light side
- **EnlightenGAN-N**: 50 hold-out testing → AHE/LIME/CycleGAN/EnlightenGAN과 시각 비교 (Fig 6)
- 결과: original EnlightenGAN은 unseen 도메인에서 artifact가 있지만 EnlightenGAN-N은 brightness↑ + noise 억제 균형 / Easy adaptation thanks to unpaired setting.

### Part XIII: Pre-Processing for Classification (p. 8) / 분류 전처리

ExDark dataset (low-light object classification, 7363 images, 12 classes):

Top-1 accuracy with ResNet-50:
- Raw input: **22.02 %**
- After EnlightenGAN: **23.94 %** (+1.92)
- After LIME: 23.32
- After AHE: 23.04

Top-5: 39.46 % → 40.92 % with EnlightenGAN.

→ low-light 향상이 high-level 비전 task에도 도움이 된다는 부수 증거 (p.8).

Provides side evidence that unpaired enhancement helps downstream classification (p.8).

---

## 3. Key Takeaways / 핵심 시사점

1. **Unpaired = realism + scalability** — paired GT는 본질적으로 잘 정의되지 않는다. unpaired로 가면 데이터 수집이 쉬워지고 도메인 일반화가 좋아진다.
   **Unpaired enables realism + scale** — paired GT for low-light isn't well-defined; going unpaired both eases data collection and improves real-world generalization.

2. **One-path > cycle-consistent for enhancement** — CycleGAN의 cycle은 색 매핑에는 유용하지만 저조도→정상광은 information-asymmetric (정보를 *추가*하는 방향)이라 한 방향만으로 충분하다. 이로써 학습 시간 ½ 이하.
   **One-path beats cycle-consistent here** — CycleGAN's cycle helps color mapping, but low-to-normal-light is information-asymmetric (adding information). One direction suffices and trains in <½ the time.

3. **Global-local discriminator** — spatially-varying illumination에 단일 D는 부족. Global이 전반 톤, Local이 작은 밝은/어두운 영역을 책임진다. ablation에서 local 제거 시 색 왜곡 (Fig 3).
   **Global-local discriminator** — a single D cannot handle spatially-varying illumination; the global D governs overall tone while the local PatchGAN policies small over/under-exposed regions.

4. **Self-regularized attention from $1-I$** — 학습 불필요, 입력에서 결정적으로 계산. 어두운 픽셀에 더 큰 가중치를 줘서 over-enhancement of bright regions를 방지.
   **Self-regularized attention from $1-I$** — deterministic, no learning required, gives darker pixels higher weight and prevents over-enhancement of bright areas.

5. **Self feature preserving (SFP) loss** — paired perceptual loss를 unpaired로 변형. 입력과 출력의 VGG feature를 닫게 만들어 content를 보존하고 illumination만 바꾼다.
   **Self feature preserving loss** — adapts the perceptual loss to the unpaired regime by closing the VGG-feature gap between **input and output**, preserving structure while changing illumination.

6. **Relativistic + LSGAN combination** — 안정적이고 high-quality. relativistic이 "real이 fake 평균보다 더 진짜"라는 비교 기준을, LSGAN이 saturation 없는 그래디언트를 제공.
   **Relativistic + LSGAN hybrid** — relativistic provides a comparative criterion ("real more real than mean fake"); LSGAN provides non-saturating gradients. Together they stabilize training.

7. **Domain adaptability** — paired 모델과 달리, unpaired 학습 모델은 새로운 도메인 (night driving)에 같은 unpaired 절차로 빠르게 적응 (EnlightenGAN-N).
   **Domain adaptability** — unlike paired models, EnlightenGAN can be re-trained on a new domain (e.g. night driving) using the same unpaired recipe (EnlightenGAN-N).

8. **Bilinear+conv > deconv upsample** — checkerboard artifact 회피. 모든 현대 generator가 따라야 할 디테일.
   **Bilinear-then-conv upsample > deconv** — avoids checkerboard artifacts; a default detail every modern generator should adopt.

---

## 4. Mathematical Summary / 수학적 요약

### Self-regularized attention map / 자기-정규화 attention

Given input $I^L$ in RGB:

$$
I_Y = \text{normalize}(\text{Y of YCbCr}(I^L)) \in [0,1], \qquad A = 1 - I_Y.
$$

Per level $\ell$ of the U-Net:
$$
F_\ell \;\leftarrow\; F_\ell \odot \mathrm{Resize}(A,\, H_\ell\times W_\ell).
$$

### Relativistic LSGAN losses (Global) / 상대적 LSGAN 손실 (전역)

$$
D_{Ra}(x_r, x_f) = \sigma\!\big(C(x_r) - \mathbb E_{x_f \sim \mathbb P_{\text{fake}}}[C(x_f)]\big).
$$

$$
\mathcal L_D^{\text{Global}} = \mathbb E_{x_r}[(D_{Ra}(x_r, x_f) - 1)^2] + \mathbb E_{x_f}[D_{Ra}(x_f, x_r)^2].
$$

$$
\mathcal L_G^{\text{Global}} = \mathbb E_{x_f}[(D_{Ra}(x_f, x_r) - 1)^2] + \mathbb E_{x_r}[D_{Ra}(x_r, x_f)^2].
$$

### Local LSGAN PatchGAN losses / 지역 손실

$$
\mathcal L_D^{\text{Local}} = \mathbb E_{x_r \sim \mathbb P_{\text{real-patches}}}[(D(x_r) - 1)^2] + \mathbb E_{x_f \sim \mathbb P_{\text{fake-patches}}}[D(x_f)^2].
$$

$$
\mathcal L_G^{\text{Local}} = \mathbb E_{x_f}[(D(x_f) - 1)^2].
$$

### Self feature preserving loss / 자기 특징 보존 손실

$$
\mathcal L_{SFP}(I^L) = \frac{1}{W_{i,j}H_{i,j}} \sum_{x,y} \big(\phi_{i,j}(I^L)(x,y) - \phi_{i,j}(G(I^L))(x,y)\big)^2.
$$

with $\phi_{i,j}$ = VGG-16 feature at $i=5, j=1$.

### Total objective / 전체 목표

$$
\mathcal L_{\text{total}} = \mathcal L_{SFP}^{\text{Global}} + \mathcal L_{SFP}^{\text{Local}} + \mathcal L_G^{\text{Global}} + \mathcal L_G^{\text{Local}}.
$$

### LSGAN motivation / 동기

Vanilla GAN의 sigmoid-CE는 saturating gradient 문제가 있다. LSGAN의 quadratic은 분류기 출력이 1에서 멀어질수록 더 강한 신호를 준다:

Vanilla GAN's sigmoid-CE saturates when $D$ is confident. LSGAN's quadratic loss produces a stronger gradient the further $D$'s output is from the target:

$$
\mathcal L_{\text{LSGAN}} = \mathbb E[(D-1)^2] \quad\text{vs}\quad \mathbb E[\log D].
$$

### Relativistic insight / 상대적 판별 통찰

Standard GAN: $D$ asks "is $x$ real?" → easily fooled by moderately-good fakes.

Relativistic GAN: $D$ asks "is $x_r$ more real than the *average* fake?" → forces realism to be relative, leading to more useful gradients (Jolicoeur-Martineau 2018).

### Effective receptive field of generator / generator의 수용 영역

8개 conv block + 다운샘플 4단계 → 입력 픽셀에 대한 ERF는 약 $\approx 64\times64$ 영역에 해당 (대략적). attention map이 모든 레벨에 곱해지므로 ERF는 attention에 의해 추가 변조됨.

8 conv blocks + 4 downsampling steps → ERF roughly $\sim 64\times 64$ in the input. The attention map further modulates the effective response per level.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1977 Retinex theory (Land)
         ▼ illumination = reflectance × light / 조명 = 반사 × 빛
1997 Multi-Scale Retinex (Jobson)
         ▼ classical low-light tool / 고전 저조도 도구
2014 Generative Adversarial Nets (Goodfellow)
         ▼ generative learning / 생성 학습
2017 Pix2Pix (Isola) — paired translation / 쌍 기반 변환
2017 CycleGAN (Zhu) — unpaired translation, cycle-consistency
2017 LSGAN (Mao) — least-squares, no saturation
2017 LLNet (Lore) — first deep low-light AE
2018 Relativistic D (Jolicoeur-Martineau)
2018 RetinexNet (Wei) — paired Retinex CNN, LoL dataset
2018 SID / Learning to See in the Dark (Chen) — raw paired
══════════════════════════════════════════════
2021 ★ EnlightenGAN (Jiang) — first unpaired one-path GAN (THIS PAPER)
══════════════════════════════════════════════
2020 Zero-DCE (Guo) — reference-free, no GAN
2021 RUAS (Liu) — Retinex-inspired unrolled
2022 SCI (Ma) — self-calibrated illumination
2022 LLFlow (Wang) — normalizing-flow-based enhancement
```

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| CycleGAN (Zhu+ 2017) | Unpaired translation의 직접적 영감, EnlightenGAN은 cycle을 제거 / Direct unpaired-translation ancestor; EnlightenGAN drops the cycle | High |
| LSGAN (Mao+ 2017) | EnlightenGAN의 모든 adversarial loss 형식 / Loss form for both global and local D | High |
| Relativistic D (Jolicoeur-Martineau 2018) | Global D 구조 / Global discriminator structure | High |
| Pix2Pix / PatchGAN (Isola+ 2017) | Local D의 PatchGAN 디자인 / Local D's patch-discrimination design | High |
| Perceptual / VGG loss (Johnson+ 2016) | SFP loss의 출발점 (paired→unpaired 변형) / Origin of SFP loss (adapted from paired) | High |
| RetinexNet (Wei+ 2018) | Paired baseline 비교 대상, LoL 데이터셋 출처 / Paired baseline; LoL dataset origin | High |
| LIME (Guo+ 2017) | Classical baseline, NIQE 비교 / Classical baseline in NIQE comparison | Medium |
| MIRNet (Zamir+ 2020, paper #42) | 같은 task (저조도)를 paired CNN으로 / Same task, paired CNN route | Medium |
| Zero-DCE (Guo+ 2020) | 동일 paradigm (no paired GT)의 다음 세대 / Next-generation no-GT enhancement | Medium |
| U-Net (Ronneberger+ 2015) | Generator backbone | Medium |

---

## 7. References / 참고문헌

- Jiang, Y. et al. "EnlightenGAN: Deep Light Enhancement without Paired Supervision." *IEEE TIP*, vol. 30, pp. 2340-2349, 2021. DOI: 10.1109/TIP.2021.3051462
- Goodfellow, I. et al. "Generative Adversarial Nets." *NeurIPS 2014*.
- Mao, X. et al. "Least Squares Generative Adversarial Networks." *ICCV 2017*. (LSGAN)
- Jolicoeur-Martineau, A. "The Relativistic Discriminator: A Key Element Missing from Standard GAN." *arXiv:1807.00734*, 2018.
- Zhu, J.-Y. et al. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." *ICCV 2017*. (CycleGAN)
- Isola, P. et al. "Image-to-Image Translation with Conditional Adversarial Networks." *CVPR 2017*. (Pix2Pix / PatchGAN)
- Johnson, J., Alahi, A., Fei-Fei, L. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution." *ECCV 2016*.
- Wei, C. et al. "Deep Retinex Decomposition for Low-Light Enhancement." *BMVC 2018*. (RetinexNet, LoL dataset)
- Guo, X., Li, Y., Ling, H. "LIME: Low-Light Image Enhancement via Illumination Map Estimation." *IEEE TIP*, 26(2):982-993, 2017.
- Land, E. H. "The Retinex Theory of Color Vision." *Scientific American*, 237(6):108-129, 1977.
- Jobson, D. J. et al. "A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes." *IEEE TIP*, 6(7):965-976, 1997.
- Lore, K. G. et al. "LLNet: A Deep Autoencoder Approach to Natural Low-Light Image Enhancement." *Pattern Recognition*, 61:650-662, 2017.
- Chen, C. et al. "Learning to See in the Dark." *CVPR 2018*. (SID)
- Mittal, A., Soundararajan, R., Bovik, A. "Making a 'Completely Blind' Image Quality Analyzer." *IEEE Signal Processing Letters*, 20(3):209-212, 2013. (NIQE)
- Ronneberger, O., Fischer, P., Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI 2015*.
- GitHub: https://github.com/VITA-Group/EnlightenGAN
