---
title: "Pre-Reading Briefing: Generative Adversarial Nets"
paper_id: "16_goodfellow_2014"
topic: Artificial Intelligence
date: 2026-04-16
type: briefing
---

# Generative Adversarial Nets: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio, "Generative Adversarial Nets," *NIPS 2014*, 2014.
**Author(s)**: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
**Year**: 2014

---

## 1. 핵심 기여 / Core Contribution

이 논문은 **적대적 과정(adversarial process)**을 통해 생성 모델을 학습하는 완전히 새로운 프레임워크를 제안합니다. 두 개의 신경망—데이터를 생성하는 Generator(G)와 진짜/가짜를 구별하는 Discriminator(D)—이 서로 경쟁하며 동시에 학습합니다. 이 아이디어는 게임 이론의 minimax 게임으로 공식화되며, 이론적으로 G가 실제 데이터 분포를 완벽히 복원하고 D가 항상 1/2을 출력하는 유일한 균형점이 존재함을 증명합니다. 기존의 Boltzmann machines, VAE 등과 달리 Markov chain이나 근사 추론 없이 backpropagation만으로 학습할 수 있다는 혁신적 장점을 제시합니다.

This paper proposes an entirely new framework for training generative models through an **adversarial process**. Two neural networks—a Generator (G) that produces data and a Discriminator (D) that distinguishes real from fake—are trained simultaneously in competition. The idea is formalized as a minimax game from game theory, and the authors prove that a unique equilibrium exists where G perfectly recovers the real data distribution and D outputs 1/2 everywhere. Unlike Boltzmann machines, VAEs, and other approaches, this framework requires no Markov chains or approximate inference—only backpropagation. This simplicity and elegance launched an entirely new subfield of deep learning.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

2014년은 deep learning의 **판별 모델(discriminative models)**이 이미 큰 성공을 거둔 시기입니다. AlexNet(2012, Paper #13)이 ImageNet에서 압도적 성능을 보였고, word2vec(2013, Paper #14)이 언어 표현을 혁신했습니다. 그러나 **생성 모델(generative models)**은 여전히 어려운 문제였습니다:

2014 was a time when deep learning had already achieved great success with **discriminative models**. AlexNet (2012, Paper #13) dominated ImageNet, and word2vec (2013, Paper #14) revolutionized language representations. However, **generative models** remained challenging:

- **RBM / DBM / DBN** (Paper #12, Hinton 2006): Markov chain Monte Carlo (MCMC) 기반으로 학습이 느리고 mixing 문제가 있었음 / MCMC-based training was slow with mixing problems
- **VAE** (Kingma & Welling, 2013, Paper #15): Reparameterization trick으로 학습 가능했지만 생성된 이미지가 blurry했음 / Trainable via reparameterization trick but generated blurry images
- **Denoising Autoencoders / GSN**: Markov chain을 정의하는 간접적 방식 / Indirect approach defining a Markov chain

Goodfellow의 핵심 통찰은 이러한 복잡한 확률적 기계를 피하고, 판별 모델의 성공(backprop + dropout + piecewise linear units)을 생성 모델에 그대로 활용할 수 있다는 것이었습니다.

Goodfellow's key insight was to sidestep all these complex probabilistic machines and directly leverage the success of discriminative models (backprop + dropout + piecewise linear units) for generative modeling.

### 타임라인 / Timeline

```
1986  Rumelhart et al. — Backpropagation (Paper #6)
  │
2006  Hinton et al. — Deep Belief Nets / RBM pretraining (Paper #12)
  │
2012  Krizhevsky et al. — AlexNet, ImageNet breakthrough (Paper #13)
  │
2013  Kingma & Welling — VAE (Paper #15)
  │
2014  ★ Goodfellow et al. — Generative Adversarial Nets ← 지금 읽는 논문
  │
2014  Radford et al. — DCGAN (convolutional GAN)
  │
2017  Arjovsky et al. — Wasserstein GAN (학습 안정성 개선)
  │
2018  Karras et al. — Progressive GAN → StyleGAN (고해상도 얼굴 생성)
  │
2020+ Diffusion Models가 GAN을 대체하기 시작
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 수학적 배경 / Mathematical Background

| 개념 / Concept | 필요 수준 / Level | 설명 / Description |
|---|---|---|
| **확률 분포 / Probability distributions** | 중급 / Intermediate | $p_{\text{data}}(x)$, $p_g(x)$, $p_z(z)$ 등 확률밀도함수 이해 / Understanding PDFs |
| **기댓값 / Expectation** | 중급 / Intermediate | $\mathbb{E}_{x \sim p}[f(x)]$ 표기법과 의미 / Notation and meaning |
| **KL Divergence** | 중급 / Intermediate | $KL(p \| q) = \mathbb{E}_p[\log \frac{p}{q}]$, 두 분포 간 비대칭 거리 / Asymmetric distance between distributions |
| **Jensen-Shannon Divergence** | 기초 / Basic | $JSD(p \| q) = \frac{1}{2}KL(p \| m) + \frac{1}{2}KL(q \| m)$, $m = \frac{p+q}{2}$ / Symmetric version of KL |
| **Minimax 게임 / Minimax games** | 기초 / Basic | $\min_G \max_D V(G,D)$ 형식의 최적화 / Two-player zero-sum game optimization |
| **Backpropagation** | 중급 / Intermediate | Paper #6에서 다룸 / Covered in Paper #6 |
| **MLP (Multilayer Perceptrons)** | 중급 / Intermediate | Papers #6, #13에서 다룸 / Covered in Papers #6, #13 |

### 선수 논문 / Prerequisite Papers

- **Paper #6 (Rumelhart 1986)**: Backpropagation — GAN 학습의 기반 / Foundation of GAN training
- **Paper #13 (Krizhevsky 2012)**: Deep CNN — 판별 모델의 성공을 이해하기 위해 / To understand discriminative model success

### 개념적 배경 / Conceptual Background

**위조지폐범과 경찰 비유 / Counterfeiter-Police Analogy**:
- Generator = 위조지폐범: 진짜처럼 보이는 가짜 화폐를 만들려 함 / Counterfeiter: tries to produce fake currency that looks real
- Discriminator = 경찰: 진짜와 가짜를 구별하려 함 / Police: tries to distinguish real from fake
- 경쟁이 계속되면 위조지폐범의 실력이 점점 좋아져서 결국 구별 불가능해짐 / Competition drives the counterfeiter to improve until fakes are indistinguishable

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Generator (G)** | 잠재 공간 $z$에서 데이터 공간 $x$로의 매핑 함수. 가짜 데이터를 생성 / Mapping from latent space $z$ to data space $x$. Produces fake data |
| **Discriminator (D)** | 입력 $x$가 진짜 데이터인지 생성된 가짜인지 판별하는 함수. 출력은 $[0,1]$ 확률 / Function that judges whether input $x$ is real or generated. Outputs probability in $[0,1]$ |
| **Adversarial training** | G와 D가 서로 적대적으로 경쟁하며 동시에 학습하는 과정 / Process where G and D compete against each other during simultaneous training |
| **Minimax game** | $\min_G \max_D V(G,D)$: G는 V를 최소화, D는 V를 최대화하려는 2인 게임 / Two-player game where G minimizes and D maximizes the value function V |
| **Latent space ($z$)** | Generator에 입력되는 노이즈 벡터의 공간. 보통 가우시안 또는 균등 분포에서 샘플링 / Space of noise vectors fed to the Generator. Typically sampled from Gaussian or uniform distribution |
| **$p_{\text{data}}$** | 실제 학습 데이터의 확률 분포 / Probability distribution of the real training data |
| **$p_g$** | Generator가 생성하는 데이터의 확률 분포. 학습 목표는 $p_g = p_{\text{data}}$ / Probability distribution of generated data. Training goal is $p_g = p_{\text{data}}$ |
| **Nash Equilibrium** | G와 D 모두 더 이상 개선할 수 없는 균형 상태. $p_g = p_{\text{data}}$이고 $D(x) = 1/2$ / Equilibrium where neither G nor D can improve. $p_g = p_{\text{data}}$ and $D(x) = 1/2$ |
| **Mode collapse** | G가 다양한 출력 대신 소수의 모드만 생성하는 실패 현상 (논문에서 "Helvetica scenario"로 언급) / Failure mode where G produces only a few modes instead of diverse outputs (called "Helvetica scenario" in the paper) |
| **Parzen window estimate** | 생성된 샘플로부터 확률밀도를 추정하는 비모수적 방법. 논문의 정량적 평가에 사용됨 / Non-parametric density estimation from generated samples. Used for quantitative evaluation in the paper |

---

## 5. 수식 미리보기 / Equations Preview

### 수식 1: GAN 목적함수 / GAN Objective Function

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

- 이것이 논문의 핵심 수식입니다 / This is the central equation of the paper
- **$D(x)$**: D가 실제 데이터 $x$를 "진짜"로 판단할 확률 → D는 이를 1에 가깝게 만들고 싶음 / Probability D assigns to real data being "real" → D wants this close to 1
- **$D(G(z))$**: D가 생성된 가짜 데이터를 "진짜"로 판단할 확률 → D는 이를 0에, G는 1에 가깝게 만들고 싶음 / Probability D assigns to fake data being "real" → D wants 0, G wants 1
- D 입장: 두 항 모두 최대화 (진짜는 맞추고, 가짜는 거부) / D's view: maximize both terms
- G 입장: 두 번째 항 최소화 ($D(G(z))$를 1에 가깝게) / G's view: minimize second term (make $D(G(z))$ close to 1)

### 수식 2: 최적 Discriminator / Optimal Discriminator

$$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

- G가 고정되었을 때 D의 최적 해 / Optimal D for a fixed G
- $p_{\text{data}} = p_g$이면 $D^*(x) = 1/2$ (구별 불가) / When $p_{\text{data}} = p_g$, $D^*(x) = 1/2$ (cannot distinguish)

### 수식 3: Jensen-Shannon Divergence와의 연결 / Connection to JSD

$$C(G) = -\log(4) + 2 \cdot JSD(p_{\text{data}} \| p_g)$$

- 최적 D를 대입하면 G의 비용함수는 JSD에 비례 / Substituting optimal D, G's cost is proportional to JSD
- $p_g = p_{\text{data}}$일 때 $JSD = 0$이므로 $C(G) = -\log 4$가 전역 최솟값 / When $p_g = p_{\text{data}}$, $JSD = 0$, so $C(G) = -\log 4$ is the global minimum

### 수식 4: 실전 트릭 — Generator 목적함수 / Practical Trick — Generator Objective

학습 초기에 $\log(1 - D(G(z)))$는 gradient saturation 문제가 있어, 실제로는:

In early training, $\log(1 - D(G(z)))$ suffers from gradient saturation, so in practice:

$$\text{G를 학습할 때 / When training G:} \quad \max_G \mathbb{E}_{z \sim p_z}[\log D(G(z))]$$

$\log(1-D(G(z)))$를 최소화하는 대신 $\log D(G(z))$를 최대화 → 동일한 고정점, 더 강한 gradient / Instead of minimizing $\log(1-D(G(z)))$, maximize $\log D(G(z))$ → same fixed point, stronger gradients

---

## 6. 읽기 가이드 / Reading Guide

### 논문 구조 / Paper Structure (9 pages)

| 섹션 / Section | 페이지 / Pages | 중요도 / Priority | 읽기 팁 / Reading Tips |
|---|---|---|---|
| **1. Introduction** | 1 | ★★★ | 위조지폐범 비유에 집중. 기존 생성 모델의 한계를 파악 / Focus on counterfeiter analogy. Understand limitations of existing generative models |
| **2. Related work** | 2 | ★★ | RBM, DBN, NCE, GSN 등 기존 방법과의 차이를 빠르게 파악. 세부 사항은 건너뛰어도 됨 / Quickly grasp differences from existing methods. Details can be skimmed |
| **3. Adversarial nets** | 2-3 | ★★★ | **가장 중요한 섹션**. Eq. 1 (minimax 목적함수)과 Algorithm 1을 완전히 이해. Figure 1의 (a)→(d) 진행 과정 숙지 / **Most important section**. Fully understand Eq. 1 and Algorithm 1. Study Figure 1 progression |
| **4. Theoretical Results** | 3-5 | ★★★ | Proposition 1 (최적 D), Theorem 1 (전역 최적), Proposition 2 (수렴) 증명을 따라가기. KL→JSD 변환 핵심 / Follow proofs of Proposition 1, Theorem 1, Proposition 2. KL→JSD transformation is key |
| **5. Experiments** | 5-6 | ★★ | Table 1의 Parzen window 결과, Figure 2의 생성 샘플 확인 / Check Parzen window results in Table 1, generated samples in Figure 2 |
| **6. Advantages & disadvantages** | 7 | ★★★ | Table 2 비교표 중요. "Helvetica scenario" (mode collapse) 개념 / Table 2 comparison is important. "Helvetica scenario" concept |
| **7. Conclusions** | 7-8 | ★★ | 5가지 미래 확장 방향: conditional GAN, semi-supervised 등 / 5 future extensions: conditional GAN, semi-supervised, etc. |

### 핵심 집중 포인트 / Key Focus Points

1. **Eq. 1과 Algorithm 1**: 이 두 가지가 GAN의 전부입니다. 완전히 이해하세요 / These two are the entirety of GANs. Understand them completely
2. **Figure 1 (a)→(d)**: 학습 과정의 직관적 이해. 녹색(pg)이 검은색(pdata)에 수렴하고, 파란색(D)이 1/2로 평탄해지는 과정 / Intuitive understanding of training. Green ($p_g$) converges to black ($p_{\text{data}}$), blue (D) flattens to 1/2
3. **실전 트릭**: $\log(1-D(G(z)))$ → $\log D(G(z))$ 변경의 이유 (gradient saturation) / Practical trick: why switch objectives (gradient saturation)
4. **Table 2**: 다른 생성 모델과의 체계적 비교 / Systematic comparison with other generative models

---

## 7. 현대적 의의 / Modern Significance

GAN은 딥러닝 역사에서 가장 영향력 있는 아이디어 중 하나로, 발표 후 폭발적인 후속 연구를 촉발했습니다:

GANs are one of the most influential ideas in deep learning history, triggering an explosion of follow-up research:

- **이미지 생성 / Image generation**: DCGAN (2015), Progressive GAN (2017), StyleGAN (2018-2021) — 포토리얼리스틱 얼굴 생성 / Photorealistic face generation
- **이미지 변환 / Image translation**: Pix2Pix (2016), CycleGAN (2017) — 스타일 변환, 도메인 변환 / Style transfer, domain adaptation
- **초해상도 / Super-resolution**: SRGAN (2017) — 저해상도 이미지 고해상도 복원 / Low-to-high resolution restoration
- **텍스트-이미지 / Text-to-image**: StackGAN (2017) — 텍스트 설명으로부터 이미지 생성 / Generate images from text descriptions
- **의료 / Medical**: 희소 의료 데이터 증강, 합성 의료 이미지 생성 / Augmenting scarce medical data, synthetic medical imaging
- **학습 안정성 개선 / Training stability**: WGAN (2017), Spectral Normalization (2018) — mode collapse와 학습 불안정성 해결 / Addressing mode collapse and training instability

2020년대에 들어 diffusion models (DALL-E 2, Stable Diffusion, Midjourney)가 이미지 생성에서 GAN을 상당 부분 대체했지만, GAN의 적대적 학습 개념은 여전히 많은 분야에서 활용됩니다. 무엇보다 "두 네트워크의 경쟁을 통한 학습"이라는 패러다임 자체가 AI 연구의 사고방식을 근본적으로 바꿨습니다.

In the 2020s, diffusion models (DALL-E 2, Stable Diffusion, Midjourney) have largely replaced GANs for image generation, but the adversarial training concept remains widely used. Most importantly, the paradigm of "learning through competition between two networks" fundamentally changed how AI researchers think about generative modeling.

---

## Q&A

### Q1: Figure 1의 노이즈와의 연결 — Generator는 노이즈를 어떻게 활용하는가?

**질문**: Figure 1의 $z$ 공간과 $x$ 공간의 관계가 불명확. 노이즈 샘플링이 왜 중요한가?

**답변**: Generator $G$는 단순한 분포(균등/가우시안)의 노이즈 $z$를 복잡한 데이터 분포 $x$로 **변환하는 함수**입니다. Figure 1의 화살표($z \rightarrow x$ 매핑)가 조밀한 곳은 $p_g$의 밀도가 높고, 희소한 곳은 낮습니다. 노이즈는 **다양성의 원천** — 각 $z$ 샘플이 서로 다른 데이터 포인트를 생성합니다. G의 학습은 이 매핑 함수의 파라미터 $\theta_g$를 조정해서 $p_g \rightarrow p_{\text{data}}$로 만드는 과정입니다.

**Answer**: Generator $G$ is a function that **transforms** noise $z$ from a simple distribution into complex data distribution $x$. The arrows in Figure 1 (mapping $z \rightarrow x$) show density: dense arrows = high $p_g$, sparse = low $p_g$. Noise provides **diversity** — each $z$ sample produces a different data point. Training adjusts $\theta_g$ so that $p_g \rightarrow p_{\text{data}}$.

### Q2: cGAN/Pix2Pix에서는 노이즈 없이 입력 이미지가 제공되는데?

**질문**: Pix2Pix는 노이즈 대신 입력 이미지를 제공. 그러면 입력 이미지 분포와 타겟 이미지 분포 사이의 매핑이 G의 역할인가? 다양성 문제가 있고 paired 데이터가 많이 필요하지 않을까?

**답변**: 세 가지 모두 정확한 지적.

1. Pix2Pix의 G는 조건부 분포 $p(x_{\text{output}} | x_{\text{input}})$를 학습 — 원래 GAN이 "무에서 유를 창조"한다면 Pix2Pix는 "도메인 변환"
2. 노이즈가 사실상 무시되어 다양성이 제한되지만, 결정론적 매핑이 목표인 태스크에서는 문제가 아님
3. Paired 데이터가 많이 필요 → 이것이 CycleGAN (unpaired, cycle consistency loss 활용)의 동기

**Answer**: All three intuitions are correct. (1) Pix2Pix's G learns conditional distribution $p(x_{\text{out}} | x_{\text{in}})$. (2) Diversity is limited but not problematic for deterministic mapping tasks. (3) Need for paired data motivated CycleGAN's unpaired approach with cycle consistency loss.

### Q3: 과학 데이터에서 Pix2Pix vs CycleGAN vs Diffusion 비교

**질문**: 입출력 매핑이 명확한 과학 분야에서는 Pix2Pix가 유리한가?

**답변**: 맞음. 과학 데이터는 "물리적 정확성"이 최우선이고 다양성보다 결정론적 정답을 요구하므로 Pix2Pix가 가장 실용적인 첫 선택. CycleGAN은 paired 데이터가 없을 때 유용하나 hallucination 위험. Diffusion은 불확실성 정량화가 가능하다는 고유 장점이 있음 (동일 입력에 대해 여러 번 샘플링 → 평균 + 표준편차 → 픽셀별 신뢰 구간).

**Answer**: Correct. Scientific data demands physical accuracy over diversity, making Pix2Pix the most practical first choice. CycleGAN is useful without paired data but risks hallucination. Diffusion models uniquely enable uncertainty quantification (multiple samples → mean + std → per-pixel confidence intervals).

### Q4: Diffusion의 불확실성 정량화 + 자기장 맵 denoising에의 적용

**질문**: (1) 불확실성 정량화의 의미? (2) 자기장 맵 denoising에 Pix2Pix 경험이 있는데 Diffusion은 어떨까?

**답변**:

(1) 불확실성 맵은 각 픽셀마다 예측의 신뢰도를 제공. 예: quiet Sun 영역은 불확실성 낮음(±2G), 활동 영역이나 림 근처는 높음(±25-30G). 후속 분석(자유 에너지 계산, 플레어 예측)에서 오차 전파 가능.

(2) Diffusion denoising의 핵심 장점: 극성 반전선(PIL) 근처의 약한 자기장처럼 노이즈가 물리적 해석을 바꿀 수 있는 영역에서 "여기는 확신이 없다"고 알려줄 수 있음. 기존 Pix2Pix 결과를 baseline으로 유지하면서 조건부 DDPM으로 비교 연구가 가능.

**Answer**: (1) Uncertainty maps provide per-pixel confidence — low near quiet Sun (±2G), high near active regions/limb (±25-30G), enabling error propagation in downstream analysis. (2) For magnetogram denoising, diffusion's key advantage is flagging uncertain regions (e.g., near polarity inversion lines). Could compare conditional DDPM against existing Pix2Pix baseline.
