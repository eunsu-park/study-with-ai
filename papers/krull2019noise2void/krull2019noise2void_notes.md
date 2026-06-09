---
title: "Noise2Void — Learning Denoising from Single Noisy Images"
authors: Alexander Krull, Tim-Oliver Buchholz, Florian Jug
year: 2019
journal: "Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"
doi: "10.1109/CVPR.2019.00223"
topic: Low-SNR Imaging / Self-Supervised Deep Denoising
tags: [noise2void, n2v, self-supervised, single-image, blind-spot-network, masked-loss, conditional-independence, biomedical-imaging, krull-buchholz-jug, u-net]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 17. Noise2Void: Learning Denoising from Single Noisy Images / 단일 노이즈 영상으로부터의 디노이징 학습

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 **단 한 장의 노이즈 영상만으로** denoising 신경망을 학습할 수 있음을 보였다. **Noise2Noise(N2N, paper #16)** 가 *깨끗한 타겟*의 필요성을 제거했지만 여전히 *두 독립 노이즈 측정 페어*를 요구했다. **Noise2Void(N2V)** 는 그것마저 제거한다.

핵심 도구는 **blind-spot network**이다. 일반 fully-convolutional network는 픽셀 $\hat s_i$ 의 예측이 입력 $\mathbf x$ 의 *receptive field* 전체에 의존한다 — 그 안에 *픽셀 자체* $x_i$ 도 포함되므로, noisy 입력을 noisy 타겟과 *동일 영상*에서 학습하면 신경망은 단순히 *항등 함수*를 학습하고 만다. N2V는 receptive field에서 *중심 픽셀* $x_i$ 만 인위적으로 *공백(blind spot)* 으로 만들어, 신경망이 $x_i$을 *주변 픽셀로부터* 예측하도록 강제한다. 이것이 가능한 이유는 두 가정 때문이다:

1. **신호의 공간적 의존성**: 자연 영상에서 픽셀 $s_i$는 이웃 $s_{j\ne i}$와 통계적으로 의존 ($p(s_i\mid s_j) \ne p(s_i)$).
2. **노이즈의 조건부 독립성**: 잡음은 신호 조건부로 픽셀 간 독립 ($p(\mathbf n\mid \mathbf s) = \prod_i p(n_i\mid s_i)$) 이고 zero-mean ($\mathbb E[n_i]=0$).

이 두 조건이 만족되면 blind-spot network는 *항등 함수를 학습할 수 없고* (잡음은 이웃에서 추정 불가), 동시에 *신호는 이웃에서 추정 가능* — 결과적으로 학습이 신호 회복으로 수렴한다.

저자들은 이를 효율적으로 구현하기 위해 *masking scheme*를 제안한다: 각 학습 패치($64\times64$)에서 무작위 $N=64$ 픽셀의 값을 *주변에서 샘플링한 값*으로 대체해 *blind spot*을 만들고, 손실은 *그 마스크 픽셀 위치*에서만 계산한다. 이 방식은 표준 U-Net을 그대로 사용 가능하게 하며, blind-spot network를 별도 설계하지 않는다.

결과: BSD68 (가우시안 $\sigma=25$)에서 N2V 27.71 dB vs N2N 28.86 dB, BM3D 28.59 dB. **N2V는 BM3D보다 약간 낮지만** depth-2 baseline 28.36 dB보다 위. 핵심 가치는 PSNR이 아니라 *cryo-TEM, CTC-MSC, CTC-N2DH* 등 *깨끗한 페어조차 얻을 수 없는* 의료·생물학 데이터에 적용 가능하다는 점이다 (Fig. 4). 이 논문은 이후 모든 *single-image* self-supervised denoiser (Noise2Self, Self2Self, Neighbor2Neighbor, Probabilistic N2V, DivNoising)의 출발점이며, 실용적으로 가장 폭넓게 채택된 deep denoiser 중 하나가 되었다.

### English
The paper trains a denoising network using **a single noisy image** — no clean targets, no paired noisy acquisitions. **Noise2Noise** removed the clean-target requirement; **Noise2Void** removes the paired-image requirement.

The mechanism is the **blind-spot network**: a CNN whose receptive field at output pixel $\hat s_i$ excludes the input value $x_i$ at that very location. Trained with input = target = same noisy image, an ordinary CNN would learn the identity. The blind-spot architecture *cannot* learn the identity because $x_i$ is unavailable. It is forced to predict $x_i$ from its surroundings. Two assumptions make this work:

1. **Signal pixels are spatially dependent** — $p(s_i\mid s_j)\ne p(s_i)$ for nearby $j$.
2. **Noise is pixel-wise independent given signal**, zero-mean — $p(\mathbf n\mid\mathbf s) = \prod_i p(n_i\mid s_i)$, $\mathbb E[n_i\mid s_i]=0$.

Under these, the blind-spot network's prediction at pixel $i$ cannot include the noise of $x_i$ (independent of all other pixels), so the optimum is $\mathbb E[s_i\mid \mathbf x_{\setminus i}]$ — the signal estimate from the neighbourhood.

Implementation uses an efficient **masking scheme**: in every training patch (64×64), $N=64$ random pixels are replaced with values drawn from their surroundings, creating "voids"; the loss is computed only at those masked positions. A standard U-Net is used (with batch normalisation), without architectural modification.

Empirically (Table/Fig. 4), on BSD68 with $\sigma=25$ Gaussian noise, N2V scores 27.71 dB vs N2N 28.86 dB and BM3D 28.59 dB — slightly below BM3D but applicable where BM3D is not. The decisive contribution is qualitative: N2V is applied to cryo-TEM, the Cell Tracking Challenge MSC and N2DH datasets where neither clean targets nor noisy pairs exist (Fig. 4 rows 3–5), opening deep denoising to practically any imaging modality. N2V seeds an entire family of follow-ups (Noise2Self, Self2Self, Neighbor2Neighbor, Probabilistic N2V, DivNoising).

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction and §2 Related Work / 서론과 관련 연구

#### 한국어
- 영상 $\mathbf x = \mathbf s + \mathbf n$. 신호 $\mathbf s$와 잡음 $\mathbf n$을 분리.
- Denoising은 *이웃 픽셀로부터 한 픽셀의 값을 추정*할 수 있다는 가정에 기반.
- 기존 supervised learning은 $(\mathbf x^j, \mathbf s^j)$ 페어를 요구. ground-truth가 없으면 적용 불가.
- N2N은 $(\mathbf x^j, \mathbf x'^j)$ (독립 잡음, 같은 신호) 페어로 학습. 그러나 *quasi-static* 영상에 한해 — 살아있는 시료, 단일 노출 X-ray, *EM 단일-particle* 등에선 어려움.
- N2V는 단일 영상 만으로 학습. 가정: (i) signal pixels not statistically independent, (ii) noise pixel-wise independent given signal, zero-mean.

#### 한국어 — Related work landmarks
- **Discriminative deep**: Jain (2009 CNN denoiser), DnCNN (Zhang 2017), Mao (2016 RED30) — 모두 supervised.
- **Internal statistics**: Buades NL-means (paper #4), Dabov BM3D (paper #7) — 학습 없이 단일 영상에서 통계 추출. N2V는 이 family의 deep 버전.
- **Generative**: Chen GAN-based; van den Oord PixelCNN — 픽셀의 *분포*를 예측. PixelCNN은 *asymmetric receptive field*; N2V는 *symmetric blind-spot*.

#### English
N2V sits at the intersection of (a) **discriminative deep denoising** (DnCNN, RED30) which needs clean targets, (b) **internal-statistics methods** (NL-means, BM3D) which exploit a single noisy image but use hand-crafted statistics, and (c) **generative models** (PixelCNN's masked receptive field is a related architectural trick). N2V is a discriminative *deep* network that uses the same single-image principle as internal-statistics methods.

---

### Part II: §3 Methods / 방법

#### 한국어 — §3.1 Image Formation
- $\mathbf x = \mathbf s + \mathbf n$, joint $p(\mathbf s, \mathbf n) = p(\mathbf s) p(\mathbf n\mid \mathbf s)$.
- Signal: $p(s_i\mid s_j) \ne p(s_i)$ for $j$ within some radius (Eq. 2). 자연 영상의 spatial structure.
- Noise: pixel-wise conditional independence
$$
p(\mathbf n \mid \mathbf s) = \prod_i p(n_i \mid s_i) \tag{Eq.\,3}
$$
zero-mean: $\mathbb E[n_i]=0$ (Eq. 4) → $\mathbb E[x_i \mid s_i] = s_i$ (Eq. 5).

#### 한국어 — §3.2 Traditional Supervised Training
- FCN $f(\mathbf x; \theta)$를 patch-기반 시각으로 봄: 픽셀 예측 $\hat s_i = f(\mathbf x_{\mathrm{RF}(i)}; \theta)$.
- Receptive field $\mathbf x_{\mathrm{RF}(i)}$는 $i$ 주변의 정사각형 patch.
- 학습: $\arg\min_\theta \sum_{j,i} L\bigl(f(\mathbf x_{\mathrm{RF}(i)}^j; \theta), s_i^j\bigr)$ (Eq. 7), $L$은 MSE (Eq. 8).

#### 한국어 — §3.3 Noise2Noise Training (recap)
- Noisy 입력 $\mathbf x = \mathbf s + \mathbf n$, noisy 타겟 $\mathbf x' = \mathbf s + \mathbf n'$, $n \perp n'$.
- 학습 데이터 = patch-target 페어 $(\mathbf x_{\mathrm{RF}(i)}^j, x_i'^j)$.
- 깨끗 ground truth 없이도 같은 최적해 — paper #16의 핵심.

#### 한국어 — §3.4 Noise2Void Training (이 논문의 핵심)
- 단 하나의 noisy 학습 영상 $\mathbf x^j$가 있을 때 *naive* 방법은 $(\mathbf x_{\mathrm{RF}(i)}^j, x_i^j)$ — 입력 patch의 *중심* 픽셀을 *직접* 타겟으로 사용. 결과: 신경망이 *항등*을 배움.
- 해결책: **blind-spot network**. Receptive field에서 *중심 픽셀만 제외*. 그러면 신경망은 다음 위험 최소화로 학습:
$$
\arg\min_\theta \sum_{j,i} L\bigl(f(\tilde{\mathbf x}_{\mathrm{RF}(i)}^j; \theta), x_i^j\bigr) \tag{Eq.\,10}
$$
$\tilde{\mathbf x}$는 blind-spot input (중심 픽셀이 보이지 않음).

#### 한국어 — Why blind-spot 네트워크가 동작하는가
- Eq. 3에서 $n_i$는 *이웃 픽셀*과 *조건부 독립* (signal-conditional). 이웃은 $n_i$에 *zero* 정보 → 이웃에서 $n_i$ 추정 불가능.
- Eq. 4 (($\mathbb E[n_i\mid s_i]=0$)) → 신경망의 *최적* 출력은 $\mathbb E[s_i \mid \mathbf x_{\setminus i}]$.
- 자연 영상에선 $s_i$와 이웃 $s_j$ 의 dependency 가 강 → 신경망은 신호 회복.

#### 한국어 — §3.5 Implementation: Masking Scheme
- Naive blind-spot: 매 출력 픽셀당 patch 하나씩 forward. 매우 비효율 (전체 영상 한 번에 못 처리).
- 효율적 trick: $64\times64$ random patch를 뽑아 그 안의 *N개 픽셀*을 stratified sampling으로 선택. 그 픽셀의 *값을 주변에서 random하게 sampling한 값*으로 *덮어쓰기* — 이게 blind spot.
- 손실은 *그 N 위치*에서만 계산 (CSBDeep의 specialised loss).
- 한 patch에서 동시에 여러 blind-spot pixel의 gradient를 평균 → $64\times$ 가속 ($N=64$ per 64×64 patch).
- Architecture: U-Net (Ronneberger 2015) + batch norm 추가. Depth 2, kernel 3, 96 feature maps initial level (BSD68 실험).

#### English
N2V's contribution is twofold:
1. **Conceptual**: train on a single noisy image by masking the centre pixel from the receptive field. The blind-spot network cannot learn the identity, so it must learn to predict $s_i$ from $\mathbf x_{\setminus i}$ — under independent-noise + dependent-signal assumptions, this converges to $\mathbb E[s_i\mid \mathbf x_{\setminus i}]$.
2. **Practical**: rather than designing a special blind-spot architecture, replace centre-pixel values in random training patches with neighbour-sampled values (creating artificial voids), and compute the loss only at those positions. Standard U-Nets work as-is.

---

### Part III: §4 Experiments / 실험

#### 한국어 — §4.1 BSD68
- 학습: 400 grayscale 영상 ($180\times 180$), $\sigma=25$ 가우시안 잡음.
- Augmentation: 90° 회전 ×3 + mirroring (8 orbits).
- Patch $64\times 64$, 마스킹 $N=64$ per patch.
- U-Net depth 2, kernel 3, BN, 96 feature maps initial. lr 0.0004, CSBDeep schedule (plateau 시 halving).
- 결과 (Table/Fig. 4 row 1): N2V 27.71 dB, N2N 28.86 dB, traditional supervised 29.06 dB, BM3D 28.59 dB. 
- **N2V drops moderately** below BM3D, *expected* (less information available).

#### 한국어 — §4.2 Simulated microscopy data
- Membrane-labelled epithelia 시뮬레이션 영상 → 깨끗 ground truth 정확히 알려져 있음.
- 결과 (Fig. 4 row 2): Traditional 32.56 dB, N2N 32.43 dB, N2V 32.28 dB, BM3D 29.96 dB. **N2V가 BM3D에 ~2.3 dB 우월**, supervised에 ~0.3 dB 차이.

#### 한국어 — §4.3 Real microscopy data (no GT)
- Cryo-TEM (Buchholz+ 2019, paper #15와 같은 데이터): traditional/N2N은 적용 가능 (페어 존재), N2V도 적용 가능. 시각 비교 (Fig. 4 row 3): N2V는 traditional/N2N에 가깝고, BM3D보다 시각적으로 깨끗.
- CTC-MSC, CTC-N2DH (Cell Tracking Challenge): 페어 없음, ground truth 없음. *오직 N2V만 적용 가능*. 시각 결과 (Fig. 4 rows 4–5): 의미 있는 디노이징 (BM3D보다 sharp).
- Runtime: BM3D 33.2 s/4.6 s/5.2 s vs N2V 1.3 s/0.1 s/0.1 s — **inference에서 ~30–50× 빠름**.

#### English
On BSD68 N2V's PSNR is 1–2 dB below traditional/N2N (expected information loss) and below BM3D by 0.9 dB. On simulated microscopy N2V is ~2.3 dB above BM3D. On real cryo-TEM and CTC-MSC/N2DH N2V is the *only* applicable deep method. Inference is 30–50× faster than BM3D.

---

### Part IV: §5 Discussion / 논의

#### 한국어
- N2V는 정보 손실 (이웃만 사용) 때문에 supervised/N2N보다 PSNR이 낮을 수밖에 없음 — 이론적 한계.
- 그러나 **적용 가능 범위**가 폭발적으로 넓음: ground truth가 없거나 페어를 만들 수 없는 모든 데이터 (의료, 라이브 이미징, 단일-particle EM 등).
- 가정의 타당성: zero-mean signal-conditional-independent noise는 가산 가우시안·푸아송 photon 노이즈 (per pixel)에 잘 맞는다. 픽셀 간 상관 노이즈 (e.g. read-out 패턴, structured noise)에선 성능 저하.
- 후속 연구가 가정 완화 (Probabilistic N2V는 noise model을 학습), 아키텍처 개선, 성능 격차 축소를 탐구.

#### English
N2V trades a few dB of PSNR for the ability to operate on *any* single noisy image. Its assumptions (signal dependence, conditional pixel-wise independent zero-mean noise) hold well for additive Gaussian and Poisson photon shot noise, but degrade for spatially correlated read-out noise. Many follow-ups (Probabilistic N2V, Self2Self, Neighbor2Neighbor, DivNoising) close the gap.

---

### Part V: §3.5 Implementation details / 구현 세부사항

#### 한국어
- **Why not just use a true blind-spot CNN?** 이론적으로 receptive field에서 중심 픽셀을 *구조적으로* 제거하는 CNN을 설계할 수 있다. 그러나 (i) 표준 conv layer 위에 구현하기 까다롭고, (ii) feature map별 receptive field 크기 추적이 복잡. 마스킹은 *구현 단순*하면서 동일한 효과.
- **Why stratified sampling?** 패치 안에 $N=64$ 마스크 픽셀을 균등 분포로 흩뿌리되 *공간적 클러스터 회피*. 클러스터되면 마스크 위치들끼리 receptive field overlap이 커져 학습 시그널 다양성이 감소.
- **Why neighbour-value replacement (not zeros)?** 0으로 대체하면 *분포 외* (out-of-distribution) 입력 → 신경망이 "0 = 마스크" 패턴을 학습하고 부정확한 예측. 이웃에서 sampling 한 값은 분포 내 → blind-spot이 *데이터 분포에 자연스러움*.
- **Augmentation**: 90° 회전 ×3 + mirroring → 8× 데이터. 단, 마스킹 자체가 noise-injection augmentation 역할 → over-fit 거의 없음.
- **Learning rate / batch**: lr 0.0004 + plateau halving (CSBDeep schedule). batch 128 (BSD68), 16 (cryo-TEM에서 더 큰 patch 사용 시).

#### English
The masking trick replaces a structurally blind-spot CNN with a much simpler one. $N=64$ stratified positions per 64×64 patch maximise gradient signal without clustering. Replacing centre pixels with *neighbour-sampled values* (rather than zeros) keeps the input on the data distribution, preventing the network from latching onto "0 = mask" cues.

---

### Part VI: Failure modes / 실패 양상

#### 한국어
N2V는 두 가정에 의존하므로 가정 위반 시 명확한 실패 패턴을 보인다:

1. **픽셀 간 상관 노이즈 (Eq. 3 위반)**: CMOS read-out pattern, line noise (CCD striping), shot noise after pixel binning. → 신경망이 노이즈 *패턴* 자체를 신호로 학습 → 디노이징 X.
2. **신호의 spatial independence (Eq. 2 위반)**: 합성 white-noise 같은 영상 (자연 영상엔 거의 없음). → 신경망이 신호도 추정 못함 → mean-blur 결과.
3. **Non-zero-mean noise (Eq. 4 위반)**: bias-shifted detector. → 디노이저가 bias 만큼 shifted output → 추가 calibration 필요.
4. **매우 강한 노이즈 (signal SNR ≪ 1)**: blind-spot의 정보가 신호 회복에 부족 → 결과는 매우 smoothed.

후속 논문들이 이 실패 양상 각각을 공략:
- Probabilistic N2V: noise model을 *학습* → bias 자동 보정.
- StructN2V (Broaddus 2020): 더 큰 blind-spot region → spatially correlated noise 부분 처리.
- Noise2Self (paper #18): $\mathcal J$-invariance로 일반화.

#### English
Failure modes correspond exactly to assumption violations: (1) pixel-correlated noise → network treats pattern as signal; (2) spatially independent signal → trivial mean-blur; (3) non-zero-mean noise → systematic bias in output; (4) extreme SNR → over-smoothed results. Each is addressed by a successor (Probabilistic N2V, StructN2V, Noise2Self).

---

## 3. Key Takeaways / 핵심 시사점

1. **Blind spots prevent identity learning / 블라인드 스팟이 항등 학습을 막는다** — naive single-image training은 신경망이 입력을 그대로 출력하는 항등 함수를 배운다. 중심 픽셀을 receptive field에서 제거하면 그 *trivial 해*가 사라지고, 신경망은 *이웃에서* 신호를 추정해야 한다.

2. **Two assumptions sufficient / 두 가정이면 충분** — (i) 신호 픽셀의 spatial dependence, (ii) 잡음의 conditional pixel-wise independence + zero-mean. 이 두 조건만 만족되면 blind-spot 학습은 $\mathbb E[s_i\mid \mathbf x_{\setminus i}]$에 수렴.

3. **Masking trick avoids architecture surgery / 마스킹 트릭으로 구조 수정 불필요** — 별도 blind-spot 네트워크를 설계하는 대신 입력 패치에서 random N개 픽셀의 값을 주변에서 샘플링한 값으로 *덮어쓰기*하고, 손실을 그 위치에서만 계산. 표준 U-Net 그대로 사용.

4. **Few-dB PSNR loss for huge gain in applicability / 적용 범위를 위해 PSNR 몇 dB 양보** — BSD68에서 N2V는 supervised보다 1–2 dB 낮지만 *cryo-TEM, CTC, 라이브 이미징*등 supervised가 *불가능한* 영역에 적용된다. 실용적 가치는 PSNR 절대값보다 *적용 범위*에 있다.

5. **Eliminates paired-image requirement of N2N / N2N의 페어 영상 요구사항을 제거** — N2N은 두 독립 noisy 측정 페어가 필요한데 이는 *움직이는 시료*나 *단일 노출 측정*에선 불가능. N2V는 그것마저 제거 → 가장 *minimum data assumption* 디노이저.

6. **Faster inference than BM3D / BM3D보다 빠른 추론** — Cryo-TEM 33.2s vs 1.3s, CTC 4.6/5.2s vs 0.1s — 30–50× 가속. 학습은 한 번이지만 *반복적인 inference*가 필요한 임상·실험 환경에서 결정적.

7. **Failure modes signal the assumptions / 실패 양상이 가정을 드러낸다** — 픽셀 간 상관 잡음 (read-out 패턴)이나 strong signal-pixel-independence (white-noise-like 신호)에선 N2V가 무너진다. 가정의 *명시성*이 후속 연구가 한계를 정확히 공략하게 함 (Probabilistic N2V, Self2Self 등).

8. **Foundation of the single-image self-supervised line / 단일 영상 self-supervised 계보의 출발** — Noise2Self (paper #18, J-invariance), Self2Self (paper #19, Bernoulli dropout), Neighbor2Neighbor (paper #20, 부분 다운샘플 페어), Probabilistic N2V — 모두 N2V의 가정 또는 마스킹 전략을 변주한 직접적 후예.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Image formation / 영상 모델
$$
\mathbf x = \mathbf s + \mathbf n, \quad p(\mathbf s, \mathbf n) = p(\mathbf s)\,p(\mathbf n\mid \mathbf s)
$$
$$
p(s_i\mid s_j) \ne p(s_i)\quad\text{for } j\text{ near }i \tag{Eq.\,2; signal dependence}
$$
$$
p(\mathbf n\mid \mathbf s) = \prod_i p(n_i\mid s_i) \tag{Eq.\,3; conditional pixel-wise independence}
$$
$$
\mathbb E[n_i\mid s_i] = 0,\quad \mathbb E[x_i\mid s_i] = s_i \tag{Eqs.\,4–5; zero-mean}
$$
### 4.2 CNN as patch function / 패치 함수로서의 CNN
$$
\hat s_i = f(\mathbf x_{\mathrm{RF}(i)}; \boldsymbol\theta) \tag{Eq.\,6}
$$
where $\mathbf x_{\mathrm{RF}(i)}$ is the receptive field around pixel $i$ (square patch).

### 4.3 Three training objectives / 세 가지 학습 목표
**Supervised (clean targets):**
$$
\arg\min_\theta \sum_{j,i} \bigl(f(\mathbf x^j_{\mathrm{RF}(i)}; \theta) - s_i^j\bigr)^2 \tag{Eq.\,7}
$$
**Noise2Noise (paired noisy):**
$$
\arg\min_\theta \sum_{j,i} \bigl(f(\mathbf x^j_{\mathrm{RF}(i)}; \theta) - x_i'^j\bigr)^2 \quad\text{with } \mathbf x' = \mathbf s + \mathbf n'
$$
**Noise2Void (single noisy + blind spot):**
$$
\boxed{\arg\min_\theta \sum_{j,i} \bigl(f(\tilde{\mathbf x}^j_{\mathrm{RF}(i)}; \theta) - x_i^j\bigr)^2} \tag{Eq.\,10}
$$
where $\tilde{\mathbf x}_{\mathrm{RF}(i)}$ excludes $x_i$ (blind spot).

### 4.4 Why the blind spot forces signal estimation / 왜 신호 추정이 강제되는가
With the noise's conditional independence (Eq. 3), the neighbours $\mathbf x_{\setminus i}$ carry zero information about $n_i$. So:
$$
\hat s_i^* = \mathbb E[s_i\mid \mathbf x_{\setminus i}]
$$
is the optimum of the blind-spot regression — i.e. the best signal estimate from the surroundings.

### 4.5 Masking scheme / 마스킹 방식
For each training patch $P$ of size $64\times 64$:
1. Choose $N=64$ pixel positions $\{i_k\}$ by stratified sampling (avoids clustering).
2. For each $i_k$, copy the value of a *random nearby* pixel $x_{j_k}$ into position $i_k$: $x_{i_k}\leftarrow x_{j_k}$ — creates artificial blind spot.
3. Original (un-modified) values $x_{i_k}^{\mathrm{orig}}$ are kept as targets.
4. Loss: $\frac{1}{N}\sum_{k=1}^N (f(P_{\mathrm{masked}}; \theta)_{i_k} - x_{i_k}^{\mathrm{orig}})^2$.

### 4.6 Worked example: 1-D illustration / 작동 예시: 1차원 그림
Signal: $s = [10, 10, 10, 50, 50, 50]$ (step function). Noise: $n_i \sim \mathcal N(0,4)$, $\sigma=2$. Observed:
$$
\mathbf x \approx [9.8, 10.3, 10.1, 49.7, 50.4, 49.9]
$$
- *Naive identity training*: net learns $f(x_i)=x_i$ → output equals input.
- *Blind-spot training* on position 2 (value 10.1): network sees neighbours $[9.8, 10.3, ?, 49.7, 50.4, 49.9]$ and tries to predict $x_2=10.1$. It cannot copy $x_2$; it must predict from the LEFT neighbours that the signal there is ≈10. After training over many patches, $\hat s_2 \to 10$ — the *signal*, not the noise.
- For position 3 (the edge): blind-spot must guess from $[9.8,10.3,10.1,?, 50.4, 49.9]$ — ambiguous; net learns to weight RIGHT-side dependence more strongly when context allows.

### 4.7 Derivation: optimum of blind-spot regression / 유도: 블라인드 스팟 회귀의 최적해
Risk:
$$
R(\theta) = \mathbb E_{\mathbf x}\bigl[(f(\tilde{\mathbf x}_{\mathrm{RF}(i)}; \theta) - x_i)^2\bigr]
$$
$x_i = s_i + n_i$로 분해하고 *조건부 기댓값* over $\mathbf x_{\setminus i}$:
$$
R(\theta) = \mathbb E_{\mathbf x_{\setminus i}}\bigl[\mathbb E_{x_i\mid \mathbf x_{\setminus i}}[(f(\mathbf x_{\setminus i}) - x_i)^2]\bigr]
$$
내부의 최적 $f$는 $f^* = \mathbb E[x_i\mid \mathbf x_{\setminus i}]$. 신호-잡음 분해:
$$
\mathbb E[x_i\mid \mathbf x_{\setminus i}] = \mathbb E[s_i\mid \mathbf x_{\setminus i}] + \mathbb E[n_i\mid \mathbf x_{\setminus i}]
$$
- 둘째 항: $n_i$는 $\mathbf x_{\setminus i}$와 *조건부 독립 (Eq. 3)*. 하지만 $\mathbf x_{\setminus i}$는 $s_{\setminus i}$의 정보를 포함하고 $s$는 $s_i$와 dependent — *그러나* $n_i$와 $s$는 zero-mean이므로 $\mathbb E[n_i\mid \mathbf x_{\setminus i}] = \mathbb E_{s_i\mid\mathbf x_{\setminus i}}[\mathbb E[n_i\mid s_i]] = \mathbb E_{s_i\mid\mathbf x_{\setminus i}}[0] = 0$ (Eq. 4 적용).
- 첫째 항: $\mathbb E[s_i\mid \mathbf x_{\setminus i}]$은 *이웃에서 신호 추정*.

따라서:
$$
\boxed{\;f^*(\tilde{\mathbf x}_{\mathrm{RF}(i)}) = \mathbb E[s_i\mid \mathbf x_{\setminus i}]\;}
$$
이것이 N2V가 신호를 회복하는 *수학적* 이유.

### 4.8 Quantitative results table / 정량 결과 표
| Dataset | Input PSNR | BM3D | Traditional | N2N | N2V |
|---------|-----------|------|-------------|-----|-----|
| BSD68 ($\sigma=25$) | (Fig 4) | 28.59 | 29.06 | 28.86 | 27.71 |
| Simulated microscopy | — | 29.96 | 32.56 | 32.43 | 32.28 |
| Cryo-TEM | — | n/a | n/a | applicable | applicable |
| CTC-MSC, CTC-N2DH | — | n/a (slow) | n/a (no GT) | n/a (no pair) | applicable |
| Inference time CTC | — | 4.6/5.2 s | n/a | n/a | 0.1 s |

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1985-1998   Yaroslavsky / NL-means precursors — single-image neighbourhood
2005        Buades-Coll-Morel — NL-means (paper #4) — single-image, hand-crafted
2007        Dabov+ — BM3D (paper #7) — single-image, hand-crafted
2016        van den Oord+ — PixelCNN — masked-receptive-field generative model
2017        Zhang+ — DnCNN — supervised deep denoiser
2017        Weigert+ — CARE — supervised microscopy denoiser
2018        Lehtinen+ — NOISE2NOISE (paper #16)
                            ↳ removes need for clean targets
2018        Ulyanov+ — Deep Image Prior — single-image, training-free
2019        Buchholz+ — Cryo-CARE (paper #15) — N2N for cryo-TEM
2019 ★★    KRULL+ — NOISE2VOID (THIS PAPER)
                            ↳ removes need for paired noisy images
                            ↳ single noisy image suffices via blind-spot
2019        Batson-Royer — NOISE2SELF (paper #18)
                            ↳ generalises N2V to J-invariant function class
2020        Quan+ — SELF2SELF (paper #19) — Bernoulli-dropout self-train
2020        Krull+ — Probabilistic N2V — learns explicit noise model
2021        Huang+ — NEIGHBOR2NEIGHBOR (paper #20) — sub-sampled pairs
2021+       Many follow-ups in microscopy, astronomy, MRI, fluorescence
```

이 논문은 **single-image self-supervised denoising 시대의 시작**. N2N이 깨끗 타겟 요구를 제거했다면 N2V는 *페어 요구*까지 제거 → 가장 *minimum data assumption* 디노이저.

This paper marks the **start of the single-image self-supervised denoising era**. After N2N removed the clean-target requirement, N2V removed the paired-image requirement, leaving the minimum possible data assumption.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Lehtinen+ (2018)** Noise2Noise (paper #16) | Direct predecessor | N2V removes one more assumption from N2N (paired noisy → single noisy) using blind-spot networks. The two together form the theoretical core of self-supervised denoising. |
| **Buchholz+ (2019)** Cryo-CARE (paper #15) | Concurrent / shared authors | Buchholz/Jug appear on both. Cryo-CARE applies N2N to cryo-TEM with paired data; N2V is then applied to the same domain when pairs aren't available. Companion papers. |
| **Buades+ (2005)** NL-means (paper #4) | Single-image precursor | NL-means uses single-image *self-similarity*; N2V uses single-image *self-supervised learning*. Both rely on neighbourhood predicting centre, but N2V is *learned*. |
| **Dabov+ (2007)** BM3D (paper #7) | Internal-statistics baseline | BM3D is the strongest non-deep baseline; N2V matches it on simulated microscopy and beats it in inference speed. |
| **van den Oord+ (2016)** PixelCNN | Architectural inspiration | PixelCNN's masked convolution (causal receptive field) is the same architectural family as N2V's blind-spot. PixelCNN is asymmetric (causal); N2V is symmetric. |
| **Ulyanov+ (2018)** Deep Image Prior | Training-free single-image alternative | DIP also denoises a single image without ground truth via early stopping; N2V offers a more principled alternative with explicit assumptions. |
| **Batson-Royer (2019)** Noise2Self (paper #18) | Theoretical generalisation | N2S formalises N2V as an instance of the broader $\mathcal J$-invariance principle: any function class invariant to a partition of pixels admits self-supervised training. |
| **Quan+ (2020)** Self2Self (paper #19) | Bernoulli-mask extension | Combines N2V's masking idea with Bernoulli dropout at training and prediction, reducing variance via ensembling. |
| **Huang+ (2021)** Neighbor2Neighbor (paper #20) | Sub-sampled pair construction | Constructs the two N2N-required noisy images by sub-sampling neighbours from a single image — bridges N2N and N2V conceptually. |
| **Krull+ (2020)** Probabilistic Noise2Void | Direct successor | PN2V learns an explicit noise model on top of N2V to recover information lost by the blind-spot — closes most of the PSNR gap with supervised. |

---

## 7. References / 참고문헌

- Krull, A., Buchholz, T.-O., & Jug, F. "Noise2Void — Learning Denoising from Single Noisy Images", *Proc. IEEE/CVF CVPR*, 2129–2137 (2019). [DOI: 10.1109/CVPR.2019.00223]
- Lehtinen, J., Munkberg, J., Hasselgren, J., et al. "Noise2Noise: Learning Image Restoration without Clean Data", *Proc. ICML*, 2018. [arXiv:1803.04189]
- Buchholz, T.-O., Jordan, M., Pigino, G., & Jug, F. "Cryo-CARE: Content-Aware Image Restoration for Cryo-Transmission Electron Microscopy Data", *Proc. IEEE ISBI*, 2019.
- Weigert, M., Schmidt, U., Boothe, T., et al. "Content-Aware Image Restoration: Pushing the Limits of Fluorescence Microscopy" (CARE), *bioRxiv* (2017).
- Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising" (DnCNN), *IEEE TIP*, 26(7), 3142–3155 (2017).
- Mao, X., Shen, C., & Yang, Y.-B. "Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections" (RED30), *Proc. NIPS*, 2016.
- Ronneberger, O., Fischer, P., & Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation", *Proc. MICCAI*, 2015.
- van den Oord, A., Kalchbrenner, N., & Kavukcuoglu, K. "Pixel Recurrent Neural Networks", *Proc. ICML*, 2016.
- Buades, A., Coll, B., & Morel, J.-M. "A non-local algorithm for image denoising", *Proc. CVPR*, 2005.
- Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering" (BM3D), *IEEE TIP*, 16(8), 2080–2095 (2007).
- Ulyanov, D., Vedaldi, A., & Lempitsky, V. "Deep Image Prior", *Proc. CVPR*, 2018.
- Martin, D., Fowlkes, C., Tal, D., & Malik, J. "A Database of Human Segmented Natural Images" (BSD), *Proc. ICCV*, 2001.
- Batson, J., & Royer, L. "Noise2Self: Blind Denoising by Self-Supervision", *Proc. ICML*, 2019.
