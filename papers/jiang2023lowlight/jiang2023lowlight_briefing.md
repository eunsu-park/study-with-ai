---
title: "Pre-Reading Briefing: Low-Light Image Enhancement with Wavelet-based Diffusion Models (DiffLL)"
paper_id: "31_jiang_2023"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Low-Light Image Enhancement with Wavelet-based Diffusion Models (DiffLL): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Hai Jiang, Ao Luo, Songchen Han, Haoqiang Fan, Shuaicheng Liu. "Low-Light Image Enhancement with Wavelet-based Diffusion Models." *ACM Transactions on Graphics*, Vol. 42, No. 6, Article 238 (Dec 2023). DOI: 10.1145/3618373.
**Author(s)**: Hai Jiang, Ao Luo, Songchen Han, Haoqiang Fan, Shuaicheng Liu
**Year**: 2023

---

## 1. 핵심 기여 / Core Contribution

DiffLL은 저조도(low-light) 이미지 향상을 위해 **2D 이산 웨이블릿 변환(DWT)을 통해 영상 차원을 4배씩 줄인 뒤, 그 평균(저주파) 계수에서만 조건부 확산(diffusion) 모델을 학습**시키는 새로운 프레임워크이다. 기존 픽셀-도메인 확산 모델이 갖는 (i) 막대한 추론 비용과 (ii) 무작위 가우시안 시작점에서 비롯되는 색감 왜곡/혼돈된 콘텐츠 문제를 동시에 해결하며, 학습 시 forward+denoising 모두를 수행하는 새로운 *학습 전략*과, 고주파 계수의 수직/수평 정보로 대각 성분을 보강하는 **HFRM(High-Frequency Restoration Module)**을 제안한다.

DiffLL is a low-light image enhancement (LLIE) framework that performs **conditional diffusion only on the low-frequency average coefficient produced by repeated 2D Discrete Wavelet Transform (DWT)**, shrinking spatial dimension by a factor of $4^K$ before the costly denoising loop. It additionally introduces (i) a *training-time forward+denoising strategy* that suppresses content diversity during inference and (ii) a **High-Frequency Restoration Module (HFRM)** that uses cross-attention between vertical and horizontal sub-bands to refine the diagonal high-frequency coefficient. The result is state-of-the-art PSNR/SSIM/LPIPS/FID on LOL-v1, LOL-v2-real, LSRW, and UHD-LL benchmarks while being **>70× faster** than the previous diffusion baseline (DDIM).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

저조도 영상 향상 연구는 (1) 히스토그램 평활화·Retinex 같은 *수공 prior* 시대 → (2) RetinexNet, MIRNet, EnlightenGAN 등 *learning-based* 시대 → (3) 2022~2023년 확산 모델(diffusion) 시대로 발전해 왔다. Palette(SIGGRAPH'22), DDRM(ICLR'22), WeatherDiff(TPAMI'23) 같은 확산 기반 복원 모델은 PSNR/SSIM에서는 강력했으나 색감 왜곡과 추론시간(>10초/2K) 문제로 실용성이 떨어졌다. DiffLL은 *웨이블릿 영역에서의 확산*이라는 발상으로 이 두 약점을 동시에 공격한 첫 사례다.

LLIE evolved from (1) hand-crafted priors (HE, Retinex, LIME) to (2) learned mappings (RetinexNet 2018, EnlightenGAN 2021, MIRNet 2020, SNRNet 2022) to (3) diffusion-based restoration in 2022-2023 (Palette, DDRM, WeatherDiff). Diffusion models offered strong perceptual quality but suffered from chaotic content from random Gaussian starts and >10 s inference at 600×400. DiffLL is the first to push diffusion into the wavelet domain to fix both at once.

### 타임라인 / Timeline

```
1989  DWT (Mallat)         ── multiresolution
2010  HE/Retinex era       ── hand-crafted priors
2017  LL-Net (Lore)        ── first deep LLIE
2018  RetinexNet           ── decomposition-based deep model
2020  DDPM (Ho)            ── modern diffusion arrives
2021  EnlightenGAN, MIRNet ── unpaired/learned LLIE
2022  Palette, DDRM        ── diffusion for restoration
2023  WeatherDiff, GDP     ── conditional diffusion ↑
2023  *DiffLL (this paper)*── wavelet-domain diffusion for LLIE
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **2D Discrete Wavelet Transform (Haar)**: 입력을 $A$(평균), $V$(수직), $H$(수평), $D$(대각) 네 부대역으로 분해하며 공간 차원이 절반이 된다. / Splits an image into one low-frequency $A$ and three high-frequency $V/H/D$ sub-bands at half resolution.
- **DDPM/DDIM**: 가우시안 노이즈 추가 forward process $q(x_t|x_{t-1})$와 학습된 $\epsilon_\theta(x_t,t)$로 역과정을 수행하는 score-based 모델. / Forward $q(x_t|x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\beta_t I)$ and learned reverse $p_\theta$.
- **Conditional diffusion**: 조건 $\tilde{x}$를 입력 채널에 concat 하여 $\epsilon_\theta(x_t,\tilde{x},t)$를 학습. / Conditioning input concatenated to noise estimator.
- **U-Net + Cross-Attention**: HFRM의 핵심 building block. / Backbone of $\epsilon_\theta$ and the HFRM.
- **PSNR/SSIM/LPIPS/FID**: 정량 평가 지표 / quantitative metrics for distortion (PSNR, SSIM) and perception (LPIPS, FID).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| WCDM | Wavelet-based Conditional Diffusion Model — 평균 계수에서만 동작하는 조건부 확산 / diffusion that operates only on $A^K_{low}$ |
| HFRM | High-Frequency Restoration Module — $V,H,D$ 계수를 cross-attention으로 복원 / cross-attention module restoring $V/H/D$ |
| 2D-DWT / 2D-IDWT | 2D 이산 웨이블릿 변환과 역변환(Haar) / 2D forward and inverse Haar wavelet transform |
| Average coefficient $A^k_{low}$ | $k$-단계 DWT 후의 저주파(전역 조명) 성분 / low-frequency global-illumination coefficient at level $k$ |
| High-frequency coefficients $V,H,D$ | 수직·수평·대각 디테일 / vertical, horizontal, diagonal details |
| $K$ (wavelet scale) | DWT 반복 횟수, 기본값 2 / number of DWT applications, default 2 |
| $T$ (diffusion steps) | 확산 총 스텝 수, 기본 200 / total diffusion steps, default 200 |
| $S$ (sampling step) | implicit sampling 횟수, 기본 10 / number of implicit sampling iterations |
| Content diversity | 무작위 노이즈로 인해 같은 입력에 다른 출력이 나오는 현상 / output randomness from random Gaussian start |
| Forward diffusion in training | 학습 시에도 forward diffusion을 수행해 추론 시 안정성을 확보 / FD performed during training to suppress randomness at test time |
| HFRM cross-attention | $V$와 $H$의 정보를 $D$에 주입 / cross-attention injects $V/H$ info into $D$ |
| LOL/LSRW/UHD-LL | 평가용 paired 저조도 데이터셋 / paired LLIE benchmark datasets |

---

## 5. 수식 미리보기 / Equations Preview

(1) Haar DWT 분해 / Haar decomposition:

$$
\{A^1_{low}, V^1_{low}, H^1_{low}, D^1_{low}\} = \text{2D-DWT}(I_{low})
$$

(2) Forward diffusion (DDPM):

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\,\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I)
$$

(3) Reparametrised marginal $q(x_t|x_0)$:

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
$$

(4) DiffLL training objective on $A^K_{low}$ (Eq. 9 of paper):

$$
\mathcal{L}_{diff} = \mathbb{E}_{x_0,t,\epsilon}\left[\,\|\epsilon - \epsilon_\theta(x_t,\tilde{x},t)\|^2 + \|\hat{A}^K_{low}-A^K_{high}\|^2\,\right]
$$

(5) Total loss (Eq. 13):

$$
\mathcal{L}_{total} = \mathcal{L}_{diff} + \mathcal{L}_{detail} + \mathcal{L}_{content}
$$

각 수식은 (1) DWT가 차원을 줄여 확산이 작은 텐서에 적용된다는 것, (2) 노이즈 추정 네트워크 $\epsilon_\theta$가 평균 계수만 처리한다는 것, (3) content loss(L1+SSIM)와 detail loss(MSE+TV)가 결합되어 전반적 충실도와 디테일 보존을 동시에 강제한다는 것을 보여준다.

These equations show: (1) DWT shrinks the diffusion target; (2) only $A^K_{low}$ is denoised; (3) the auxiliary content+detail terms suppress drift and recover high-frequency details lost by diffusion alone.

---

## 6. 읽기 가이드 / Reading Guide

- **Sec. 3.1–3.2 (DWT + DDPM 복습)**: 빠르게 훑어도 됨, 표준 자료. / Quick skim — standard background.
- **Sec. 3.3 WCDM**: Algorithm 1의 *training-time forward diffusion* 부분과 Eq. 9의 두 번째 항을 정확히 이해할 것. / Pay close attention to the FD-during-training trick and the second term of Eq. 9.
- **Sec. 3.4 HFRM**: Fig. 5 구조도(Depth Conv → cross-attn → dilation Resblock)를 따라가며 *왜 V, H를 D에 cross-attend* 하는지 이해. / Understand why $V,H$ attend into $D$.
- **Sec. 4.4 Ablations**: $K=2$, $S=10$이 왜 sweet spot인지, HFRM 제거 시 PSNR이 얼마나 떨어지는지 (Table 6) 확인. / Look at the $K$, $S$, and HFRM ablation tables.
- **Limitation**: 극단적 어두운 장면에서는 여전히 한계가 있음을 언급. / Authors acknowledge failure in extreme low-light cases.

---

## 7. 현대적 의의 / Modern Significance

DiffLL은 *latent diffusion*(Stable Diffusion)의 정신 — "확산은 저차원 잠재 공간에서 하라" — 을 **VAE 대신 무손실(linear) 변환인 웨이블릿**으로 재해석한 작품으로, 2024–2025년의 모바일/에지용 LLIE 모델 설계에 직접적인 영감을 주었다. 2K 영상에서 1초 안에 끝나는 확산 LLIE는 사실상 DiffLL이 처음이며, low-light face detection 같은 하류 태스크에서 26.4%→38.5% AP 향상을 보임으로써 *학습된 향상이 단지 미적 개선이 아니라 인식 성능에도 기여*함을 보여 주었다.

DiffLL reinterprets the latent-diffusion idea using a **lossless linear** transform (wavelet) instead of a VAE, removing the encoder-induced information loss. It is the first diffusion-based LLIE practical at 2K resolution (~1 s) and demonstrates that LLIE-as-preprocessing pushes downstream face-detection AP from 26.4% (raw) to 38.5%. This blueprint — "do diffusion in the cheapest invertible representation" — directly informs subsequent mobile/edge restoration research.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
