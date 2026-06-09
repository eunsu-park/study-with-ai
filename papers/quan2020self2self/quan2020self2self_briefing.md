---
title: "Pre-Reading Briefing: Self2Self with Dropout"
paper_id: "19_quan_2020"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Self2Self with Dropout: Learning Self-Supervised Denoising from Single Image / 사전 읽기 브리핑

**Paper**: Quan, Y., Chen, M., Pang, T., Ji, H. "Self2Self with Dropout: Learning Self-Supervised Denoising from a Single Image". *IEEE/CVF CVPR 2020*, pp. 1890–1898. DOI: 10.1109/CVPR42600.2020.00196.
**Authors**: Yuhui Quan, Mingqin Chen, Tongyao Pang, Hui Ji
**Year**: 2020

---

## 1. 핵심 기여 / Core Contribution

### 한국어
Self2Self(S2S)는 **단일 잡음 영상 $\mathbf y = \mathbf x + \mathbf n$만을 사용**하여 deep denoising network를 학습하는 자기지도 framework다. 핵심 통찰은 Noise2Void/Noise2Self가 BM3D를 못 따라잡는 진짜 원인이 *bias*가 아닌 *prediction variance*라는 점이다. 이를 해결하기 위해 (i) Bernoulli 마스크($p=0.3$)로 input/target 픽셀 쌍을 생성하고, (ii) decoder의 모든 conv 층에 dropout을 *학습과 추론 양쪽 모두*에 활성화하며, (iii) 추론 시 $N=50$회 forward pass를 *평균*해 variance를 $1/N$로 줄인다. Set9 σ=25에서 31.74 dB로 BM3D(31.67)·dataset-trained N2N(31.33)을 모두 능가하며, **단일 영상 self-supervised로 BM3D를 처음 추월한 방법**이다.

### English
Self2Self (S2S) trains a deep denoiser from **only a single noisy image** $\mathbf y = \mathbf x + \mathbf n$ via a Bernoulli-sampling + dropout-ensemble framework. The paper identifies that prior single-image self-supervised methods (N2V, N2S) under-perform BM3D not because of bias but because of *prediction variance* — a single training sample blows up the variance term in MSE = bias² + variance. The fix is: (i) Bernoulli pixel masks ($p=0.3$) form input/target pairs, (ii) dropout is applied in every decoder conv layer and stays *active during both training and inference*, (iii) at test time, $N=50$ forward passes are averaged to cut variance ~$1/N$. Achieves 31.74 dB on Set9 σ=25 — the *first* single-image self-supervised method to surpass BM3D and even dataset-trained DnCNN.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
2018-2019년 self-supervised denoising은 Noise2Noise(2018, 두 잡음 캡처 필요), Noise2Void(2019, 단일 영상 + blind-spot), Noise2Self(2019, J-invariance 이론)로 빠르게 발전했지만 단일 영상 모드에서는 모두 BM3D(2007)에 못 미쳤다. 동시에 Gal & Ghahramani(2016)의 "dropout = approximate Bayesian inference" 해석이 자리 잡았고, Liu et al.(2018)의 partial convolution이 마스크된 입력 처리 도구로 등장했다. S2S는 이 두 흐름을 결합해 단일 영상 self-sup가 hand-crafted prior를 추월하는 변곡점을 만든다.

#### English
Between 2018 and 2019 self-supervised denoising advanced rapidly — Noise2Noise (1800, two noisy captures), Noise2Void (2019, single image with blind spot), Noise2Self (2019, J-invariance theory) — yet none matched BM3D (2007) in single-image mode. Simultaneously, Gal & Ghahramani's (2016) "dropout-as-approximate-Bayesian" interpretation matured, and Liu et al. (2018) introduced partial convolutions for masked-input handling. S2S combines these threads and marks the inflection point where a single-image self-supervised method finally surpasses hand-crafted priors.

### 타임라인 / Timeline

```
2007  Dabov BM3D — non-learning denoising baseline
2014  Srivastava — dropout as regularisation
2016  Gal & Ghahramani — dropout = Bayesian approximation
2017  Ulyanov DIP — single-image deep prior (unstable, < BM3D)
2018  Lehtinen N2N (#16) — clean GT unnecessary
2019  Krull N2V (#17) / Batson N2S (#18) — single-image, < BM3D
2020 ★ Quan Self2Self — Bernoulli + dropout ensemble, FIRST > BM3D
2021  Huang Neighbor2Neighbor (#20) — successor: sub-sampling pairs
```

---

## 3. 필요한 배경 지식 / Prerequisites

#### 한국어
- **MSE 분해**: $\text{MSE}=\text{bias}^2 + \text{variance}$ — 본 논문의 출발점
- **Dropout regularisation** (Srivastava 2014) 및 **dropout-as-Bayesian** (Gal-Ghahramani 2016)
- **Encoder-decoder UNet**, skip connection, 2×2 max-pooling
- **Partial convolution** (Liu 2018): 마스크 비율로 정규화하는 conv
- **Bernoulli sampling**, conditional independence
- **Noise2Void/Noise2Self의 핵심 아이디어** (논문 #17, #18)
- **이전 논문 맥락**: BM3D(#7), N2N(#16), N2V(#17), N2S(#18)

#### English
- MSE decomposition: $\text{MSE}=\text{bias}^2+\text{variance}$ — the paper's starting motivation.
- Dropout regularisation (Srivastava 2014) and dropout-as-Bayesian inference (Gal & Ghahramani 2016).
- UNet encoder-decoder with skip connections, 2×2 max-pooling.
- Partial convolution (Liu 2018) — convolution with mask-aware normalisation.
- Bernoulli sampling and conditional independence.
- Familiarity with N2V (#17) and N2S (#18) — S2S generalises their masking schemes.
- Prior context: BM3D (#7), N2N (#16).

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Bernoulli sampling** | 잡음 영상 $\mathbf y$에 확률 $p=0.3$의 마스크 적용해 input/target 분할 / Apply a Bernoulli mask ($p=0.3$) to split $\mathbf y$ into input/target pixel sets. |
| **Blind spot** | N2V에서 중심 픽셀을 가리는 정규화 — S2S에선 70% 마스킹으로 일반화 / N2V's masked centre pixel; S2S generalises to ~70% masking. |
| **Partial convolution (PConv)** | 마스크 비율에 따라 정규화하는 conv 변형 / Convolution that normalises by the unmasked-pixel ratio (Liu 2018). |
| **Test-time dropout** | 추론 시에도 dropout 활성 → MC ensemble / Dropout kept active at inference for Monte-Carlo averaging. |
| **Ensemble averaging** | $N=50$ forward pass를 평균해 variance↓ / Average $N=50$ stochastic forwards to reduce variance ~$1/N$. |
| **Bayesian dropout** | dropout = posterior 근사 (Gal-Ghahramani 2016) / Dropout interpreted as approximate Bayesian inference. |
| **Variance term** | MSE의 두 성분 중 하나, S2S가 표적으로 삼음 / One of the two MSE components — S2S's primary target. |
| **Deep Image Prior (DIP)** | 단일 영상, random input, 조기 종료 의존 (Ulyanov 2017) / Single-image deep prior with early-stopping dependence. |
| **Identity mapping** | 학습이 $\mathcal F(\mathbf y)=\mathbf y$로 무너지는 실패 모드 / Failure mode where the network simply outputs the input. |
| **J-invariance** | N2S의 partition-invariance 개념 / Partition-invariance underlying Noise2Self. |
| **Proposition 1** | 자기지도 손실 = 지도 손실 + 잡음 분산 (S2S 정리) / S2S's theorem: self-sup loss = supervised loss + noise variance. |
| **Set9 / PolyU** | 평가 데이터셋 (synthetic Gaussian / real-world smartphone) / Synthetic vs real-world evaluation benchmarks. |

---

## 5. 수식 미리보기 / Equations Preview

### 한국어
**Bernoulli 샘플링** — 입력/타겟 분할:

$$
\hat{\mathbf y} = \mathbf b \odot \mathbf y, \quad \overline{\mathbf y} = (\mathbf 1 - \mathbf b)\odot \mathbf y, \quad \mathbf b\sim\text{Bern}(p),\ p=0.3
$$

**학습 손실 (Eq. 7)** — 마스크된 위치에서만 손실 계산:

$$
\min_\theta \sum_{m=1}^M \|\mathcal F_\theta(\hat{\mathbf y}_m) - \overline{\mathbf y}_m\|^2_{\overline{\mathbf b}_m}
$$

**Proposition 1 (Eq. 8)** — 자기지도 손실 = 지도 손실 + 잡음 분산:

$$
\mathbb E_{\mathbf n}\sum_m \|\mathcal F_\theta(\hat{\mathbf y}_m)-\overline{\mathbf y}_m\|^2_{\overline{\mathbf b}_m} = \sum_m \|\mathcal F_\theta(\hat{\mathbf y}_m)-\mathbf x\|^2_{\overline{\mathbf b}_m} + \sum_m \|\boldsymbol\sigma\|^2_{\overline{\mathbf b}_m}
$$

**추론 ensemble (Eq. 9)** — $N=50$ 평균:

$$
\mathbf x^* = \frac{1}{N}\sum_{n=1}^N \mathcal F_{\theta_n}(\mathbf b_{M+n}\odot \mathbf y)
$$

### English
The Bernoulli sampling equation defines two complementary pixel sets from one noisy image. The training loss penalises predictions only at *masked* positions, mirroring N2V's blind-spot intuition but at 70% rather than 1-pixel scale. Proposition 1 shows the training objective decomposes into the true supervised loss plus a $\theta$-independent noise variance — same structure as N2S Eq. 2. The test-time ensemble averages $N=50$ stochastic forward passes (each with a different dropout mask and Bernoulli sample) to reduce prediction variance.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **꼭 정독할 부분**: §1 Introduction (variance vs bias 구분), §3.2 Loss + Proposition 1, §3.3 test ensemble의 motivation, §4.5 Ablation Table 4 (각 component 기여도).
- **빠르게 훑을 부분**: §2 Related work, §4.1 implementation 세부.
- **흔한 걸림돌 / Common stumbling blocks**:
  - "왜 추론 시 dropout을 끄지 않는가?" — Bayesian MC ensemble을 위해 *반드시* 켜둬야 함.
  - "Bernoulli mask와 N2V blind-spot의 차이" — N2V는 1픽셀, S2S는 ~70% — 더 큰 randomness.
  - "Partial convolution이 왜 필요한가" — masked 영역에서 standard conv는 activation을 underestimate.
- 동반 자료: Gal-Ghahramani 2016 ICML 논문, partial convolution 원논문(Liu 2018), Set9/PolyU 데이터셋 설명.

### English
- **Read carefully**: §1 (variance-vs-bias framing), §3.2 (loss + Proposition 1), §3.3 (motivation for keeping dropout on at test), §4.5 ablation (per-component contribution).
- **Skim**: §2 related work, §4.1 implementation minutiae.
- **Common stumbling blocks**:
  - Why dropout stays *on* at test — required for the Monte-Carlo Bayesian-ensemble interpretation.
  - The conceptual jump from N2V's single-pixel mask to S2S's 70% Bernoulli mask — more randomness, more diversity.
  - Why partial convolution is necessary — standard conv underestimates activation magnitude in masked regions.
- Companion reading: Gal & Ghahramani (2016) on dropout-as-Bayesian, Liu et al. (2018) on partial convolutions.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
S2S는 *현대적 self-supervised denoising 스택의 조상*이다. 후속 작업 — Neighbor2Neighbor(#20)는 random Bernoulli 대신 spatially-structured sub-sampling으로 inference cost를 줄였고, Recorrupted-to-Recorrupted(#21)는 noise injection 변환으로 N2N 등가 학습을 단일 영상에 가능케 했으며, Blind2Unblind(#22)는 visible blind spot으로 정보 손실을 회복했다. 모두 S2S를 직접 baseline으로 인용한다. 또한 dropout-as-MC-ensemble의 실용적 가치는 cryo-EM, fluorescence microscopy, 천체 영상 등 single-shot 저-SNR 도메인에서 여전히 적용된다. "단일 영상 + 알려지지 않은 잡음 + 학습 가능 prior"라는 가장 어려운 조합이 BM3D를 능가할 수 있음을 처음 증명한 작품이다.

### English
Self2Self is the ancestor of the modern self-supervised denoising stack. Direct successors — Neighbor2Neighbor (#20) replaces random Bernoulli with structured sub-sampling for cheaper inference; Recorrupted-to-Recorrupted (#21) uses noise-injection transforms to enable N2N-equivalent training from a single image; Blind2Unblind (#22) recovers blind-spot information via a re-visible loss — all cite S2S as a primary baseline. The dropout-as-MC-ensemble pattern remains practically useful in cryo-EM, fluorescence microscopy, and astronomical single-shot imaging where ground-truth pairs are unavailable. S2S is the first proof that the hardest combination — *single image + unknown noise + learnable prior* — can surpass BM3D.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
