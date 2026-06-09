---
title: "Pre-Reading Briefing: Noise2Void — Learning Denoising from Single Noisy Images"
paper_id: "17_krull_2019"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Noise2Void: Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Krull, A., Buchholz, T.-O., & Jug, F., "Noise2Void — Learning Denoising from Single Noisy Images", *Proc. IEEE/CVF CVPR 2019*, pp. 2129–2137.
**Author(s)**: Alexander Krull, Tim-Oliver Buchholz, Florian Jug
**Year**: 2019

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 **단 한 장의 노이즈 영상만으로** denoising 신경망을 학습할 수 있음을 보였다. Noise2Noise(N2N, paper #16) 가 *깨끗한 타겟*의 필요성을 제거했다면, Noise2Void(N2V) 는 *두 독립 노이즈 측정 페어*의 필요성마저 제거한다. 핵심 도구는 **blind-spot network** — receptive field 에서 *중심 픽셀*만 인위적으로 가리는 네트워크. 일반 CNN은 입력 = 타겟 = 같은 영상으로 학습하면 *항등 함수*를 학습한다. 중심 픽셀을 가리면 그 trivial 해가 사라지고, 두 가정 — (i) 신호의 spatial dependence, (ii) 잡음의 conditional pixel-wise independence + zero-mean — 하에 신경망은 $\mathbb E[s_i \mid \mathbf x_{\setminus i}]$ (이웃에서 신호 추정) 으로 수렴한다. 효율적 구현을 위해 patch 안에 *N=64* 픽셀을 random masking 후 그 위치에서만 손실 계산 — 별도 blind-spot 아키텍처 설계 없이 표준 U-Net 그대로 사용. BSD68/$\sigma=25$에서 N2V 27.71 dB vs N2N 28.86 dB, BM3D 28.59 dB — N2V는 BM3D보다 약간 낮지만 *cryo-TEM, CTC-MSC/N2DH 등 페어조차 얻을 수 없는* 데이터에 적용 가능. 이후 Noise2Self, Self2Self, Neighbor2Neighbor, Probabilistic N2V 등의 시발점.

### English
The paper trains a denoising network using **a single noisy image** — no clean targets, no paired noisy acquisitions. Noise2Noise (N2N, paper #16) removed the clean-target requirement; Noise2Void (N2V) removes the paired-image requirement. The mechanism is a **blind-spot network**: a CNN whose receptive field at output pixel $i$ excludes the input value $x_i$. Trained with input = target = same noisy image, an ordinary CNN would learn the identity; with a blind spot, the network is forced to predict $x_i$ from its surroundings. Under two assumptions — (i) spatial dependence of signal pixels, (ii) conditional pixel-wise independence and zero-mean of noise — the optimum is $\mathbb E[s_i\mid\mathbf x_{\setminus i}]$. Practically, instead of designing a structurally blind-spot CNN, the authors mask $N=64$ random pixels per training patch and compute the loss only at those positions. On BSD68 with $\sigma=25$ Gaussian noise, N2V scores 27.71 dB vs N2N 28.86 dB and BM3D 28.59 dB — slightly below BM3D but applicable in cryo-TEM, Cell Tracking Challenge MSC/N2DH and any other modality where neither clean targets nor noisy pairs exist. Inference is 30–50× faster than BM3D. N2V seeds an entire family (Noise2Self, Self2Self, Neighbor2Neighbor, Probabilistic N2V).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting
**한국어**: 2018년 N2N 발표 직후, *살아있는 시료* (live cell, single-particle EM) 처럼 *동일 시료의 두 독립 측정* 자체가 불가능한 도메인이 남아 있었다. CARE 같은 supervised 방법, N2N 같은 N2N-paired 방법 모두 작동 못하는 영역. Krull, Buchholz, Jug(MPI-CBG 그룹)는 *cryo-CARE(paper #15)* 의 같은 저자들로, cryo-EM 데이터의 다양성을 직접 다루던 중 "단일 영상 학습"의 필요를 강하게 느꼈다. 동시기 Batson-Royer(Noise2Self, paper #18)도 *J-invariance* 라는 일반 이론 프레임워크로 같은 문제를 다뤘다 — 두 논문은 *concurrent submission* 으로 N2V가 CVPR 2019, N2S가 ICML 2019에 등장. 핵심 차이: N2V는 *공학적 마스킹 트릭*으로 표준 U-Net을 그대로 쓰고, N2S는 *J-invariant 함수 클래스* 로 더 넓은 추상화. 두 논문이 함께 *single-image self-supervised denoising* 시대를 열었다.

**English**: Right after N2N (2018), domains remained where the *paired-noisy* requirement still failed: live cells, single-particle EM, and any modality where two independent measurements of the same specimen are impossible. CARE-style supervised and N2N-style paired methods both broke down there. Krull, Buchholz, Jug (MPI-CBG; the same group as Cryo-CARE, paper #15) faced these data daily and pushed for a single-image solution. Concurrently, Batson-Royer (Noise2Self, paper #18) attacked the same problem from a general *J-invariance* framework — N2V appeared at CVPR 2019, N2S at ICML 2019 as concurrent submissions. The key difference: N2V is an *engineering masking trick* that lets standard U-Nets be reused, whereas N2S is a *function-class abstraction*. Together they opened the single-image self-supervised denoising era.

### 타임라인 / Timeline
```
2005 ─── Buades+ — NL-means (paper #4; single-image, hand-crafted)
2007 ─── Dabov+ — BM3D (paper #7; single-image, hand-crafted)
2016 ─── van den Oord+ — PixelCNN (masked-receptive-field generative model)
2017 ─── Zhang+ — DnCNN; Weigert+ — CARE (both supervised)
2018 ─── Lehtinen+ — Noise2Noise (paper #16, paired noisy)
2018 ─── Ulyanov+ — Deep Image Prior (single-image, training-free)
2019 ─── Buchholz+ — Cryo-CARE (paper #15, N2N for cryo-EM)
2019 ★★ Krull+ — Noise2Void (THIS PAPER)
2019 ─── Batson-Royer — Noise2Self (paper #18, J-invariance theory)
2020 ─── Krull+ — Probabilistic N2V; Quan+ — Self2Self
2021 ─── Huang+ — Neighbor2Neighbor; many follow-ups
```

---

## 3. 필요한 배경 지식 / Prerequisites

- **수학 / Math**:
  - Conditional expectation $\mathbb E[Y\mid X]$ 의 *iterated* 형태와 conditional independence 의 의미.
  - Conditional pixel-wise independence: $p(\mathbf n\mid \mathbf s) = \prod_i p(n_i \mid s_i)$.
  - 신호의 spatial dependence: $p(s_i \mid s_j) \ne p(s_i)$ for nearby $j$.
- **딥러닝 / Deep learning**:
  - U-Net (Ronneberger+ 2015), CNN의 receptive field 개념.
  - Masked convolution / blind-spot network (PixelCNN 의 causal mask 와 유사).
  - Stratified sampling, patch-based training.
- **선행 논문 / Prior reading (필수 / Required)**:
  - Paper #16 (Noise2Noise) — 본 논문이 *직접* 확장하는 출발점.
  - Paper #15 (Cryo-CARE) — 같은 저자들의 N2N 응용; N2V는 cryo-EM 페어조차 없을 때의 대안.
- **확률 / Probability**:
  - Zero-mean conditional noise 의 의미: $\mathbb E[n_i \mid s_i] = 0$.
  - Marginalisation over $s_i$ when $n_i$ is conditionally independent of $\mathbf x_{\setminus i}$.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Blind-spot network | Receptive field 에서 중심 픽셀을 제외한 CNN — 항등 함수 학습 방지. / CNN whose receptive field excludes the central input pixel; prevents identity learning. |
| Receptive field | 출력 픽셀의 예측에 영향을 주는 입력 픽셀들의 집합. / Set of input pixels that influence a given output pixel. |
| Masking scheme | Patch 안 N개 픽셀을 이웃 값으로 덮어쓴 후 그 위치에서만 손실 계산. / Replace N patch pixels with neighbour values; compute loss only at those positions. |
| Stratified sampling | 균등 분포 + 클러스터 회피 위치 선택. / Uniform-but-non-clustered position sampling. |
| Conditional pixel-wise independence | $p(\mathbf n \mid \mathbf s) = \prod_i p(n_i \mid s_i)$ — 잡음이 신호 조건부로 픽셀 간 독립. / Noise is pixel-wise independent given signal. |
| Signal spatial dependence | $p(s_i \mid s_j) \ne p(s_i)$ for nearby $j$ — 이웃 픽셀이 신호 정보를 가짐. / Neighbouring signal pixels carry information about each other. |
| Identity collapse | 입력=타겟이면 신경망이 $f(x)=x$로 수렴 — N2V가 막아야 할 trivial 해. / Trivial $f(x)=x$ solution that arises when input equals target. |
| CSBDeep | Weigert+의 오픈소스 deep microscopy 프레임워크; N2V도 그 위에 구현. / Weigert+'s open-source deep-microscopy framework; N2V built on top. |
| Internal statistics | NL-means/BM3D 같은 단일 영상 통계 활용 방법. / Single-image-statistics methods like NL-means and BM3D. |
| Probabilistic N2V | 후속 논문 (Krull 2020) — noise model 학습으로 information loss 보완. / Follow-up that learns an explicit noise model to recover information lost by the blind spot. |
| StructN2V | Broaddus+ 2020, 상관 잡음 부분 처리. / Variant handling spatially correlated noise. |
| CTC | Cell Tracking Challenge — N2V 의 *unique* 적용 도메인. / Cell Tracking Challenge dataset; one of N2V's exclusive application domains. |

---

## 5. 수식 미리보기 / Equations Preview

### 5.1 Image formation / 영상 모델
$$
\mathbf x = \mathbf s + \mathbf n, \quad p(s_i \mid s_j) \ne p(s_i)\ \text{for nearby}\ j
$$
$$
p(\mathbf n \mid \mathbf s) = \prod_i p(n_i \mid s_i), \quad \mathbb E[n_i \mid s_i] = 0
$$
신호의 spatial dependence + 잡음의 conditional pixel-wise independence + zero-mean. / Spatial signal dependence + conditional pixel-wise noise independence + zero mean.

### 5.2 N2V training objective (Eq. 10) / N2V 학습 목표
$$
\arg\min_\theta \sum_{j,i} \bigl(f(\tilde{\mathbf x}^j_{\mathrm{RF}(i)}; \theta) - x_i^j\bigr)^2
$$
$\tilde{\mathbf x}_{\mathrm{RF}(i)}$는 blind-spot input (중심 픽셀 보이지 않음); 손실은 마스크 위치에서만. / $\tilde{\mathbf x}$ excludes $x_i$; loss is computed only at masked positions.

### 5.3 Optimum derivation / 최적해 유도
$$
\mathbb E[x_i \mid \mathbf x_{\setminus i}] = \mathbb E[s_i \mid \mathbf x_{\setminus i}] + \mathbb E[n_i \mid \mathbf x_{\setminus i}]
$$
잡음 조건부 독립으로 $\mathbb E[n_i \mid \mathbf x_{\setminus i}] = \mathbb E_{s_i\mid \mathbf x_{\setminus i}}[\mathbb E[n_i \mid s_i]] = 0$. 따라서:
$$
\boxed{\;f^*(\tilde{\mathbf x}_{\mathrm{RF}(i)}) = \mathbb E[s_i \mid \mathbf x_{\setminus i}]\;}
$$
*신호의* conditional expectation으로 수렴 — 항등 함수가 아니라 신호 회복. / Optimum is the conditional expectation of the *signal* given neighbours — not the identity.

### 5.4 Masking scheme / 마스킹 전략
For each $64\times64$ patch $P$:
- Stratified sampling으로 $N=64$ position $\{i_k\}$ 선택.
- 각 $i_k$에 대해 random nearby $x_{j_k}$ 값을 복사: $x_{i_k} \leftarrow x_{j_k}$.
- 손실 = $\frac{1}{N}\sum_{k=1}^N (f(P_{\rm masked})_{i_k} - x_{i_k}^{\rm orig})^2$.
표준 U-Net 그대로 사용; aritecture 수정 불필요. / Stratified position sampling, neighbour-value replacement, masked-only loss. Standard U-Net works as-is.

### 5.5 Comparison of training objectives / 세 학습 목표 비교
$$
\text{Supervised:}\quad \arg\min_\theta \sum (f(\mathbf x^j_{\rm RF(i)}) - s_i^j)^2
$$
$$
\text{N2N:}\quad \arg\min_\theta \sum (f(\mathbf x^j_{\rm RF(i)}) - x_i'^j)^2 \quad \text{with}\ \mathbf x' = \mathbf s + \mathbf n'
$$
$$
\text{N2V:}\quad \arg\min_\theta \sum (f(\tilde{\mathbf x}^j_{\rm RF(i)}) - x_i^j)^2 \quad \text{with blind spot}
$$
*데이터 요구 단조 감소*: clean pair → noisy pair → single noisy image. / Monotonically decreasing data requirements: clean pair → noisy pair → single noisy image.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
**우선 읽을 부분 / Focus first**:
1. **§3.1–3.4 영상 모델 + N2V 정의** — 두 핵심 가정 (signal dependence, noise conditional pixel-wise independence + zero-mean) 의 의미. *왜* 이 두 가정 하에 blind-spot 학습이 신호 회복으로 수렴하는지.
2. **§3.5 마스킹 트릭** — 별도 blind-spot 아키텍처 설계가 *왜 어렵고* 마스킹이 *왜 등가*인지. Neighbour-value replacement 가 *왜 zero replacement보다 좋은지* (out-of-distribution 회피).
3. **Fig. 4** — BSD68/microscopy/cryo-TEM/CTC-MSC/CTC-N2DH 의 5행 비교. *N2V만 적용 가능*한 마지막 두 행이 본 논문의 *실용적 핵심*.
4. **Notes §4.7 derivation** — 최적해가 $\mathbb E[s_i \mid \mathbf x_{\setminus i}]$ 임을 보이는 정식 유도. cross term이 zero-mean conditional independence 로 사라지는 단계 추적.

**자주 헷갈리는 지점 / Common stumbling blocks**:
- N2V는 *항등 함수를 막는다* — 정상 CNN이 input=target에서 학습하면 trivial $f(x)=x$로 수렴. Blind spot이 그 trivial 해를 *물리적으로 불가능*하게 만듬.
- "Blind-spot CNN을 진짜로 설계"와 "마스킹 트릭"의 관계: 둘 다 같은 효과를 *근사적으로* 달성. 마스킹은 표준 U-Net을 쓸 수 있다는 *실용적 장점*.
- 가정 (ii) — *conditional* pixel-wise independence — 픽셀 간 잡음이 *완전 독립*이 아니라 *signal-conditional* 독립. 자연 영상에서 픽셀 간 신호 상관과 잡음 독립이 *구별 가능*해야 N2V 동작.
- N2V 성능이 BM3D 보다 낮음을 *실패* 라 해석하면 안 됨. *적용 가능성*이 본 논문의 핵심 가치 — supervised/N2N 적용 불가 영역에서 *유일* 한 deep 방법.

### English
**Focus first**:
1. **§3.1–3.4 image model + N2V definition** — The two core assumptions (signal spatial dependence, noise conditional pixel-wise independence + zero-mean) and why blind-spot training converges to signal recovery under them.
2. **§3.5 masking trick** — Why a structurally blind-spot CNN is *hard to design* and why the masking trick is *equivalent*. Why neighbour-value replacement beats zero-replacement (avoids out-of-distribution input).
3. **Fig. 4** — Five-row comparison (BSD68 / simulated microscopy / cryo-TEM / CTC-MSC / CTC-N2DH). The last two rows — *only N2V applies* — are the practical heart of the paper.
4. **Notes §4.7 derivation** — Formal derivation that the optimum is $\mathbb E[s_i\mid\mathbf x_{\setminus i}]$. Track the step where the cross term vanishes via zero-mean conditional independence.

**Common stumbling blocks**:
- N2V *prevents identity learning*. A naive CNN with input = target collapses to $f(x)=x$. The blind spot makes that trivial solution *structurally impossible*.
- "True blind-spot CNN" vs "masking trick": both achieve the same effect *approximately*; masking has the practical advantage of using standard U-Nets.
- Assumption (ii) is *conditional* pixel-wise independence — given signal, not absolute pixel-wise independence. Real images must have signal correlation that is *distinguishable* from noise independence for N2V to work.
- Don't read N2V's lower PSNR than BM3D as failure. *Applicability* is the paper's core value — N2V is the *only* deep method in domains where supervised/N2N cannot apply.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
Noise2Void 는 *single-image self-supervised denoising 시대의 시작*이며, *데이터 요구가 가장 적은* deep denoiser 의 표준이다. 의료영상, 라이브 이미징, fluorescence microscopy, cryo-EM, 천문학 등 *깨끗한 GT가 없고 페어조차 만들 수 없는* 모든 분야에서 *기본 도구*로 채택. 후속 발전: Probabilistic N2V (2020, noise model 학습) 가 PSNR 격차를 거의 closer; StructN2V (2020, 더 큰 blind-spot region) 가 spatially correlated noise 처리; Self2Self (2020, Bernoulli dropout) 가 ensembling으로 분산 감소; Neighbor2Neighbor (2021, sub-sample pairs) 가 N2N과 N2V 의 중간을 메움. 더 일반적으로는 Noise2Self(paper #18) 의 J-invariance 이론이 N2V 를 *함수 클래스 추상화*로 일반화 — 모든 single-image self-supervised 방법의 통합 framework. PixelCNN/PixelRNN 의 masked convolution 과 유사한 *구조적 자기 정보 차단* 패턴은 후일 image inpainting, masked autoencoder (MAE 2022), video prediction 의 핵심 기법으로 발전.

### English
Noise2Void marks the **start of the single-image self-supervised denoising era** and stands as the *minimum-data-requirement* deep denoiser. It is the default tool in medical imaging, live imaging, fluorescence microscopy, cryo-EM, and astronomy — anywhere clean GT and noisy pairs are both unavailable. Follow-ups: Probabilistic N2V (2020, learns noise model) closes most of the PSNR gap; StructN2V (2020, larger blind-spot regions) handles spatially correlated noise; Self2Self (2020, Bernoulli dropout) reduces variance via ensembling; Neighbor2Neighbor (2021, sub-sampled pairs) bridges N2N and N2V. More broadly, Noise2Self's J-invariance theory (paper #18) generalises N2V into a *function-class abstraction*. The pattern of *structurally blocking self-information*, related to PixelCNN/PixelRNN masked convolutions, later became the central technique of image inpainting, masked autoencoders (MAE 2022), and self-supervised video prediction.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
