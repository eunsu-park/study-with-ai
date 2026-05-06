---
title: "Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots"
authors: Zejin Wang, Jiazheng Liu, Guoqing Li, Hua Han
year: 2022
journal: "CVPR 2022, pp. 2027–2036"
doi: "10.1109/CVPR52688.2022.00207"
topic: Low-SNR Imaging / Self-Supervised Denoising
tags: [self-supervised, denoising, blind-spot, noise2void, re-visible-loss, global-mask-mapper, cvpr2022]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 22. Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots / Blind2Unblind: 보이는 맹점을 가진 자기지도 영상 잡음 제거

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 **Noise2Void류 맹점(blind-spot) 디노이저의 정보 손실 문제**를 정면으로 해결한다. N2V는 중심 픽셀을 가린 패치 $\Omega_y$로 학습하므로, blind-spot 픽셀의 정보가 영원히 사용되지 못해 성능 상한이 깨끗-목표 학습에 비해 낮았다. Blind2Unblind(B2U)는 **re-visible loss**를 도입한다:
$$
\mathcal L_{\rm rev} = \big\|\,h\!\big(f_\theta(\Omega_y)\big) + \lambda \hat f_\theta(y) - (\lambda+1)\,y\,\big\|_2^2,
$$
여기서 $h(\cdot)$는 blind-spot 위치만 sampling해서 한 평면에 투영하는 **global-aware mask mapper**이고, $\hat f_\theta(y)$는 그래디언트가 흐르지 않는 *non-blind* 분기. 이 손실의 stationary point는 단일 noisy 영상으로부터 시작해 깨끗한 추정을 회복하면서도 N2V의 blind-spot 학습 안정성을 유지한다. 추가로 정규화 항 $\eta\|h(f_\theta(\Omega_y))-y\|^2$로 학습 안정. 합성 가우시안/푸아송, 실세계 SIDD raw-RGB, 형광현미경 FMD 모두에서 R2R(paper #21), NBR2NBR, Self2Self를 능가하며 supervised N2C와 0.1~0.3 dB 격차로 좁힌다.

### English
The paper directly attacks the **information-loss bottleneck of N2V-style blind-spot denoisers**. By design N2V hides the centre pixel of every input patch ($\Omega_y$), so blind-spot information is never used during training, capping performance below clean-target supervision. Blind2Unblind (B2U) introduces a **re-visible loss**
$$
\mathcal L_{\rm rev} = \big\|\,h\!\big(f_\theta(\Omega_y)\big) + \lambda \hat f_\theta(y) - (\lambda+1)\,y\,\big\|_2^2,
$$
where $h(\cdot)$ is a **global-aware mask mapper** that gathers the denoised values only at blind-spot positions onto a single plane, and $\hat f_\theta(y)$ is a stop-gradient *non-blind* branch on the unmasked image. The stationary point of this loss recovers a clean estimate from a single noisy observation while keeping N2V's stable blind-spot training. A regulariser $\eta\|h(f_\theta(\Omega_y)) - y\|^2$ stabilises training. On synthetic Gaussian/Poisson, real raw-RGB SIDD, and fluorescence-microscopy FMD benchmarks, B2U beats R2R (paper #21), NBR2NBR, and Self2Self, narrowing the gap to fully-supervised N2C to 0.1–0.3 dB.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1–§2 Motivation and related work / §1–§2 동기와 선행 연구

#### 한국어
- 영상 잡음 제거의 두 흐름: (a) 비학습(BM3D, NLM, WNNM): 영상별 반복 최적화. (b) 학습(DnCNN, U-Net, FFDNet): noisy/clean pair 필요.
- 자기지도 학습 패러다임: (i) Noise2Noise — 두 noisy 캡처. (ii) Noise2Void / Noise2Self — 단일 영상 + blind-spot. (iii) R2R / Noisier2Noise — 단일 영상 + 알려진 잡음.
- N2V류의 한계: blind-spot이 중심 픽셀의 *정보를 제거*하고 *수용장의 일부 면적을 잃어* 평균화 효과 → 성능 상한 ↓.
- 본 논문의 질문: "Blind-spot 학습의 *수렴 안정성*은 살리고, blind-spot의 *정보 손실*만 회복할 수 있는가?"

#### English
- Two streams: traditional (BM3D, NLM, WNNM) need per-image optimisation; learning-based (DnCNN, U-Net, FFDNet) need clean targets.
- Self-supervised paradigms: N2N (two noisy captures), N2V/N2S (single image + blind spot), R2R/Noisier2Noise (single image + known noise).
- N2V family limitation: the blind spot *both* hides centre-pixel information *and* shrinks the effective receptive field, capping PSNR.
- Paper's question: "Can we keep blind-spot training's *convergence stability* while *recovering* the information lost by the blind spot?"

### Part II: §3 Theoretical framework / §3 이론적 틀

#### 한국어 — Eq. (1): N2V baseline
N2V는
$$
\arg\min_\theta \mathbb E_y \|f_\theta(y_{\mathrm{RF}(i)}) - y_i\|^2
$$
즉, 픽셀 $i$의 receptive field 패치에서 *중심 $i$를 가린 컨텍스트*만으로 $y_i$를 예측. 신호가 주변에 의존하고 잡음이 독립이라는 가정 하에 동작.

#### English — Eq. (1) N2V baseline
N2V minimises $\mathbb E_y \|f_\theta(y_{\mathrm{RF}(i)}) - y_i\|^2$, predicting the masked centre $y_i$ from its surrounding patch. Works because clean signal correlates spatially while noise does not.

#### 한국어 — Eq. (2): Multi-task form
B2U는 N2V를 multi-task로 재구성:
$$
\arg\min_\theta \mathbb E_y \|h(f_\theta(\Omega_y)) - y\|_2^2 + \lambda \|\hat f_\theta(y) - y\|_2^2,
$$
첫째 항은 *blind* 손실 (N2V 일반화), 둘째 항은 *non-blind* 손실. $\hat f_\theta(y)$는 *그래디언트 미흐름* 분기 → 이 항이 직접 학습을 일으키진 않으나 정규화 역할. 그러나 단순한 합은 identity mapping을 학습하는 위험.

#### English — Eq. (2) Multi-task form
B2U recasts N2V as a multi-task loss: blind term + non-blind term. The non-blind branch $\hat f_\theta(y)$ has stop-gradient; it acts as a regulariser, not a direct training signal. A naive sum risks identity mapping.

#### 한국어 — Eqs. (3)–(8): Re-visible loss derivation
1-norm으로 다시 쓴 뒤(Eq. 3) 곱셈으로 결합 → cond ≥ 0 / cond < 0 분기 분석을 통해 두 가지 경우를 모두 포괄하는 **단일 합산 형태**를 도출:
$$
\arg\min_\theta \mathbb E_y \big\|h(f_\theta(\Omega_y)) + \lambda \hat f_\theta(y) - (\lambda+1)y\big\|_2^2 \quad (8).
$$
이를 Eq. 8 형태로 정리하면: 입력은 두 분기 (masked / unmasked), 타겟은 $(\lambda+1)y$, gradient는 *masked* 분기에만. 이 손실의 minimiser는
$$
\tilde x = \frac{h(f_\theta^*(\Omega_y)) + \lambda \hat f_\theta^*(y)}{\lambda+1}.
$$
즉 두 분기의 가중 평균 = 깨끗한 영상 추정.

#### English — Eqs. (3)–(8) Re-visible loss derivation
After rewriting in 1-norm and analysing two sign cases for $\mathrm{cond} = (h(f_\theta(\Omega_y)) - y) \odot (\hat f_\theta(y) - y)$, both cases lead to the same unified objective Eq. (8). Its minimiser is the convex combination $\tilde x = (h(f_\theta^*(\Omega_y)) + \lambda \hat f_\theta^*(y))/(\lambda+1)$, i.e., the optimal denoised image is a weighted average of the masked and unmasked branches.

#### 한국어 — §3.4 Stabilisation
순수 re-visible 손실은 blind 부분의 누적 오차가 non-blind 부분에 영향을 주어 학습이 불안정. 정규화 항 추가:
$$
\mathcal L = \mathcal L_{\rm rev} + \eta\,\mathcal L_{\rm reg}, \quad \mathcal L_{\rm reg} = \|h(f_\theta(\Omega_y)) - y\|_2^2 \quad (12).
$$
$\eta$는 blind 항의 초기 기여를 통제. 추가로 $\lambda$를 학습 중 $\lambda_s = 2 \to \lambda_f = 20$으로 점진적으로 증가 — 초기엔 N2V-like 학습, 후반엔 visible 항의 비중 증가.

#### English — §3.4 Stabilisation
Pure re-visible loss is unstable because errors in the blind term propagate to the non-blind term. A regulariser $\eta \|h(f_\theta(\Omega_y))-y\|^2$ tames the initial contribution of the blind term, and $\lambda$ is annealed from $\lambda_s=2$ to $\lambda_f=20$ during training so the visible term dominates only after the network is reasonably trained.

### Part III: §4 Method – global-aware mask mapper / §4 방법 – global mask mapper

#### 한국어
- 입력 $y \in \mathbb R^{W \times H}$를 $s\times s = 2\times 2$ 셀로 분할. 셀당 한 픽셀을 *blind-spot*으로 가리는 4가지 마스크 $\Omega^{ij}_y$ ($i,j\in\{0,1\}$). 4개 마스킹된 영상을 batch dim에 stack → **masked volume $\Omega_y$** ($4 \times W \times H$).
- 디노이저 $f_\theta(\cdot)$가 이 4채널 볼륨을 한꺼번에 처리 → $f_\theta(\Omega_y)$ ($4 \times W \times H$).
- **Mask mapper $h(\cdot)$**: 각 셀의 4가지 마스크 위치에서 *해당 위치의* denoised 값을 모아 한 평면으로 mapping. 결과 $h(f_\theta(\Omega_y)) \in \mathbb R^{W \times H}$가 *글로벌하게 denoise된* 영상.
- 학습 비용: 한 forward로 4-channel 볼륨 처리 = 4× 픽셀 처리 ≈ N2V의 4× 데이터 효율 (한 영상 = 한 epoch).

#### English
- Input $y\in\mathbb R^{W\times H}$ partitioned into $s\times s = 2\times 2$ cells. Four masks $\Omega_y^{ij}$ ($i,j\in\{0,1\}$) hide one pixel per cell. Stack the four masked images along channel/batch dimension → **masked volume $\Omega_y$** of size $4\times W\times H$.
- Denoiser processes the volume in one forward pass → $f_\theta(\Omega_y)$ of size $4\times W\times H$.
- **Mask mapper $h$** gathers the denoised value at each blind-spot position from the appropriate channel of the volume, projecting all four onto a single global denoised plane $h(f_\theta(\Omega_y))\in\mathbb R^{W\times H}$.
- Compute cost: one forward over a 4× volume = effectively 4× data per image relative to standard N2V masking.

### Part IV: §5 Experiments / §5 실험

#### 한국어
- **합성 sRGB Gaussian** ($\sigma=25$, $\sigma\in[5,50]$, RGB): Kodak/BSD300/Set14에서 N2C 대비 +0.13~0.17 dB *우월* — 즉 supervised보다 좋은 결과! (N2C는 단일 영상-supervised의 노이즈 효과 때문). NBR2NBR, R2R, Self2Self, N2V를 모두 능가.
- **합성 sRGB Poisson** ($\lambda=30$, $\lambda\in[5,50]$): R2R보다 약간 우위.
- **실세계 raw-RGB (SIDD)**: SIDD Benchmark에서 NBR2NBR 대비 +0.32 dB, supervised N2C +0.19 dB *우월*. 핵심 메시지: blind→visible 전환이 sawtooth/over-smoothing 부작용을 막는다.
- **형광현미경 (FMD)**: Confocal Mice / Two-Photon Mice에서 supervised N2C도 능가.
- **Ablation (Tables 5-8)**: 
  - Loss: $\mathcal L_A$ (Eq. 6) vs $\mathcal L_B$ (Eq. 7) — A형이 일관되게 우수.
  - Mask mapper: Random Mask vs Global Mask vs +V (visible 추가) — GM+V 조합만 수렴, 32.27 dB; RM+V는 수렴 실패.
  - Visible weight $\lambda_f$: $\lambda_f=20$이 sweet spot. $\lambda_f=2$는 약함, $\lambda_f=100$은 변화 없음.
  - Regulariser $\eta$: $\eta=1$이 best.
- **시각적 비교 (Fig. 3-5)**: Kodak/BSD300/SIDD에서 NBR2NBR이 *sawtooth* artefact 보임 (sub-sampling 부작용); B2U는 더 부드럽고 자연스러운 텍스처 회복.

### Part V: §6 Discussion / §6 논의

#### 한국어
- **이론적 의의**: 자기지도 디노이저의 *수렴 안정성*과 *정보 사용량*은 별개의 차원임을 보임. blind-spot은 안정성만 담당, 정보 손실은 별도 메커니즘 (re-visible loss)으로 회복.
- **계산 비용**: 학습 시 4채널 볼륨 처리로 forward pass cost ≈ 4× 표준 단일채널. 추론은 단일 forward — R2R의 K=50 평균보다 현저히 빠름.
- **잡음 모델 가정**: 가우시안/푸아송 모두 직접 처리. 신호 의존 잡음에서도 동작 — *잡음 분포 사전지식 불필요*가 핵심 차별점.
- **기여의 일반화**: re-visible loss + global mapper 패턴은 다른 자기지도 시각 작업(인페인팅, 초해상도)으로 확장 가능.

#### English
- **Theoretical significance**: shows that *training stability* and *information utilisation* are separable axes for self-supervised denoisers. Blind-spot handles stability; information loss is recovered by a separate mechanism (re-visible loss).
- **Compute cost**: training uses 4× forward-pass cost (4-channel volume); inference is a single forward, far cheaper than R2R's $K=50$ MC average.
- **Noise-model agnostic**: handles Gaussian and Poisson directly with no prior — the key differentiator versus R2R.
- **Generalisation**: the re-visible-loss + global-mapper pattern extends to other self-supervised vision tasks like inpainting and super-resolution.

#### English
- **Synthetic sRGB Gaussian** ($\sigma=25$, $\sigma\in[5,50]$, RGB): on Kodak/BSD300/Set14, B2U beats *supervised* N2C by 0.13–0.17 dB and beats every other self-supervised method (NBR2NBR, R2R, Self2Self, N2V).
- **Synthetic sRGB Poisson** ($\lambda=30$, $\lambda\in[5,50]$): edges past R2R.
- **Real raw-RGB SIDD**: SIDD Benchmark — +0.32 dB over NBR2NBR, +0.19 dB over supervised N2C. The key message is that the blind→visible transition prevents the over-smoothing and sawtooth artefacts seen in NBR2NBR.
- **Fluorescence Microscopy (FMD)**: best on Confocal Mice and Two-Photon Mice, even versus supervised N2C.
- **Ablations**: $\mathcal L_A$ > $\mathcal L_B$; Random-Mask + visible fails to converge while Global-Mask + visible reaches 32.27 dB; $\lambda_f = 20$ is the sweet spot for the annealed visible weight; $\eta = 1$ is the best regulariser strength.

---

## 3. Key Takeaways / 핵심 시사점

1. **Blind spot의 의미는 보존하면서 정보 손실은 회복 / Keep the blind spot's role, recover its information** — re-visible loss는 blind-spot이 *학습 안정화 도구*이지 *정보 가림막*이 아니어야 함을 보인다. 두 분기(blind / non-blind)의 가중 평균이 최적해.
   The re-visible loss reframes the blind spot as a *training stabiliser* rather than an *information mask*; the optimum is a weighted convex combination of the two branches.

2. **Stop-gradient 분기가 정규화기 / The stop-gradient branch acts as a regulariser** — $\hat f_\theta(y)$는 그래디언트가 흐르지 않으므로 학습 신호 자체는 blind 분기에서만 옴. 그러나 stationary point의 형태에 직접 등장 → 학습 후 출력이 두 분기의 평균으로 수렴.
   Although $\hat f_\theta(y)$ carries no gradient, it appears in the loss expression so the stationary point is the average of the two branches, giving the regularising effect.

3. **2×2 셀 4-mask가 충분 / 2×2 cells with 4 masks suffice** — $s=2$로 충분. 더 큰 $s$는 receptive field 손실 키움. 4채널 볼륨은 네트워크가 mask 위치 패턴을 통합 학습하도록 함.
   $s=2$ (4 masks per cell) hits the sweet spot — larger $s$ loses too much receptive field, smaller $s$ loses the global-mask benefits.

4. **$\lambda$ annealing이 핵심 / Annealing $\lambda$ is critical** — $\lambda_s = 2 \to \lambda_f = 20$ 점진 증가. 초기에는 blind 항이 학습을 끌고, 학습이 안정된 후에는 visible 항이 PSNR을 끌어올림.
   The annealing schedule $\lambda_s=2 \to \lambda_f=20$ matters: blind term dominates early to drive learning, visible term takes over later to push PSNR.

5. **Random vs Global Mask의 차이 / Random vs Global Mask** — Random Mask + visible은 수렴 실패. Global Mask는 모든 blind-spot이 하나의 평면으로 집계되어 global gradient가 일관됨. 이것이 Mask Mapper의 필수성.
   Random masks + visible-loss diverge; the Global-Mask Mapper aggregates all blind-spot pixels into one consistent plane so gradients stay coherent. The mapper is non-optional.

6. **수렴 보장의 이론적 분석 / Theoretical convergence analysis** — 논문은 re-visible loss의 upper/lower bound를 증명: $h(f_\theta^*(\Omega_y)) \le \tilde x \le \hat f_\theta^*(y)$. 즉 최종 추정은 N2V의 blind 추정과 N2V-가능한 비-blind 추정 사이.
   The paper proves $h(f_\theta^*(\Omega_y)) \le \tilde x \le \hat f_\theta^*(y)$: the optimal denoised image is sandwiched between the (lower-quality) blind branch and the (potentially overfit) non-blind branch.

7. **Supervised보다 좋다는 의미 / "Better than supervised" interpretation** — 단일 영상 supervised N2C는 noisy *target*에 대한 MSE를 최소화 → noise leakage. B2U는 noise leakage를 피하면서 모든 픽셀 정보 사용 → 결국 N2C를 능가.
   B2U beats single-image-supervised N2C because the latter overfits to noisy targets while the former stays statistically grounded yet still uses every pixel's information.

8. **R2R(paper #21)의 보완 / Complement to R2R** — R2R: 단일 영상 + 알려진 잡음 → N2N pair 합성. B2U: 알려진 잡음 *불필요*, 대신 receptive field 정보 활용. 두 방법은 가정과 강점이 상보적.
   R2R needs known noise statistics but no spatial assumption; B2U needs spatial smoothness but no noise model. The two are statistically complementary attacks on the same problem.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Setting / 설정
$$
y = x + n, \quad x \in \mathbb R^{W\times H}, \quad n \text{ pixel-wise independent zero-mean.}
$$
### 4.2 Masked volume / 마스크된 볼륨
For $s=2$ and $i,j\in\{0,1\}$:
$$
\Omega_y^{ij}(p,q) = \begin{cases} 0 & (p \bmod 2, q \bmod 2) = (i,j) \\ y(p,q) & \text{otherwise} \end{cases}
$$
$$
\Omega_y = \mathrm{stack}(\{\Omega_y^{ij}\}_{i,j=0}^1) \in \mathbb R^{4\times W\times H}.
$$
### 4.3 Mask mapper / 마스크 매퍼
$$
h\!\big(f_\theta(\Omega_y)\big)(p,q) = f_\theta(\Omega_y^{i^*,j^*})(p,q), \quad (i^*,j^*) = (p \bmod 2, q \bmod 2).
$$
즉 각 픽셀 $(p,q)$의 출력은 그 위치를 blind-spot으로 가린 채널에서 추출.

### 4.4 N2V baseline (Eq. 1)
$$
\mathcal L_{\rm N2V} = \mathbb E_y \|f_\theta(y_{\mathrm{RF}(i)}) - y_i\|_2^2.
$$
### 4.5 Re-visible loss (Eq. 8)
$$
\mathcal L_{\rm rev} = \mathbb E_y\|h(f_\theta(\Omega_y)) + \lambda \hat f_\theta(y) - (\lambda+1) y\|_2^2.
$$
$\hat f_\theta$ has stop-gradient: $\nabla_\theta \hat f_\theta = 0$.

### 4.6 Regularised loss (Eq. 11–12)
$$
\mathcal L = \mathcal L_{\rm rev} + \eta\,\mathcal L_{\rm reg}, \quad \mathcal L_{\rm reg} = \|h(f_\theta(\Omega_y)) - y\|_2^2.
$$
### 4.7 Stationary-point characterisation (Eq. 9)
$$
\tilde x = \frac{h(f_\theta^*(\Omega_y)) + \lambda \hat f_\theta^*(y)}{\lambda+1}.
$$
### 4.8 Convergence bounds
The paper establishes (under the "$\varepsilon_1>\varepsilon_2$" assumption: blind branch is noisier than non-blind):
$$
h(f_\theta^*(\Omega_y)) \le \tilde x \le \hat f_\theta^*(y),
$$
giving an explicit envelope around the true clean image.

### 4.9 Inference / 추론
Single forward pass through the **non-blind** branch:
$$
\hat x = f_\theta(y).
$$
No mask volume, no MC averaging — much cheaper than R2R's $K\approx 50$ samples.

### 4.10 Worked numerical example / 수치 예시
Consider a $4\times 4$ patch with $\sigma = 25/255 \approx 0.098$:
- Build 4 masks ($i,j\in\{0,1\}$). Each mask zeros out 4 pixels; stack into a $4\times 4\times 4$ volume.
- After one forward of a small U-Net, $f_\theta(\Omega_y)$ is $4\times 4\times 4$. The mapper $h$ assembles a $4\times 4$ plane by reading channel $(i^*,j^*) = (p\bmod 2, q\bmod 2)$ at each pixel.
- For $\lambda = 20$, the loss target is $(\lambda+1)y = 21 y$. The blind branch contributes $h(f_\theta(\Omega_y)) \approx \tilde x$ with $\mathrm{std}\approx \sigma/\sqrt{N_{\rm RF}}$; the visible branch $\hat f_\theta(y)$ contributes its denoised estimate but with a noise leakage of $\mathcal O(\sigma)$ if untrained.
- Initially, the regulariser $\eta\|h(f_\theta(\Omega_y))-y\|^2 \approx \eta\sigma^2 \approx 0.0096$ dominates; once trained, $\mathcal L_{\rm rev}\to 0$ pixel-wise.

### 4.11 Algorithm pseudo-code / 알고리즘 의사코드
```
Input: noisy dataset {y_b}, mask cell s=2, lambda schedule, eta, T iters
Initialize denoiser f_theta (modified U-Net)
For t = 1, ..., T:
  For batch {y_b}:
    Build masked volume Omega_y (shape [4, H, W] per image)
    forward: f_theta(Omega_y) -> [4, H, W]
    apply mask mapper h(.) -> blind plane h(f_theta(Omega_y))
    compute non-blind branch: f_theta(y) with stop-gradient
    lambda_t = anneal(lambda_s, lambda_f, t/T)
    L_rev = || h(f_theta(Omega_y)) + lambda_t * stop_grad(f_theta(y)) - (lambda_t+1) y ||^2
    L_reg = || h(f_theta(Omega_y)) - y ||^2
    Loss  = L_rev + eta * L_reg
    Update theta via Adam
Inference: hat_x = f_theta(y)   (single forward, non-blind)
```

### 4.12 Hyper-parameter table / 하이퍼파라미터
| Symbol | Default | Role |
|---|---|---|
| $s$ | 2 | Cell size (4 masks total). |
| $\lambda_s$ | 2 | Initial visible weight. |
| $\lambda_f$ | 20 | Final visible weight (Table 7). |
| $\eta$ | 1 | Regulariser strength (Table 8). |
| LR | $3\cdot 10^{-4}$ (sRGB), $10^{-4}$ (real) | Adam, weight decay $10^{-8}$. |
| Patch | $128\times 128$ | Random crop. |
| Batch | 4 | V100. |
| Epochs | 100, halved every 20 | Simple step schedule. |

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
2018 ─── Lehtinen+ — Noise2Noise (two noisy captures)
2018 ─── Krull+ — Noise2Void: blind-spot single image
2019 ─── Batson-Royer — Noise2Self (J-invariant)
2019 ─── Laine+ — Masked-conv blind-spot network
2020 ─── Quan+ — Self2Self with dropout (single image)
2020 ─── Moran+ — Noisier2Noise
2021 ─── Pang+ — R2R (paper #21 in this study)
2021 ─── Wang+ — Neighbor2Neighbor (NBR2NBR, sub-sample pair)
2022 ★★ WANG-LIU-LI-HAN: Blind2Unblind (this paper)
                  ↳ re-visible loss + global-mask mapper, 
                    blind spots become visible without losing them.
2023+ ── AP-BSN, LG-BPN, PUCA, LAN ... (asymmetric pixel-shuffle BSN family)
2024+ ── Diffusion-based self-supervised denoisers
```

이 논문은 **"blind-spot 디노이저는 끝났다"라는 통념을 반박**하고, 적절한 손실로 정보를 회복시키면 supervised까지 능가할 수 있음을 보였다.

The paper rebuts the prevailing view that "blind-spot denoisers had hit a wall", showing that with the right loss the lost information can be recovered to surpass even supervised baselines.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Krull et al. (2018)** *Noise2Void* | Direct predecessor B2U is rebuilding. | B2U keeps N2V's blind-spot structure but adds a re-visible loss to make it useful. |
| **Pang et al. (2021)** *R2R* (paper #21) | Concurrent self-supervised approach with different assumptions. | R2R needs known noise statistics, B2U needs spatial smoothness. Compared head-to-head as baselines. |
| **Wang et al. (2021)** *NBR2NBR* | Sub-sampling-pair self-supervised. | B2U beats NBR2NBR by 0.32 dB on SIDD by avoiding sub-sampling's structural-continuity loss. |
| **Laine et al. (2019)** *High-Quality Self-Supervised Denoising* | Masked-conv blind-spot network. | B2U's global-mask mapper is a re-visible variant of Laine's masked conv. |
| **Quan et al. (2020)** *Self2Self with dropout* | Single-image dropout-averaging. | Common single-image starting point; B2U's MC inference is *not* needed (single forward) unlike Self2Self. |
| **Lehtinen et al. (2018)** *Noise2Noise* | Statistical foundation of self-supervised denoising. | B2U inherits the "noisy target as proxy for clean target" idea; the re-visible loss is the key extension. |
| **Ronneberger et al. (2015)** *U-Net* | Backbone architecture. | B2U uses the same modified U-Net as Laine et al. — backbone is identical, only the loss is novel. |

---

### 6.1 Detailed numeric results from the paper / 논문 수치 결과 상세

#### 한국어
- **sRGB Gaussian** $\sigma=25$ on Kodak: BM3D 32.05 dB; N2V 30.32; R2R 32.25; **B2U 32.27**; supervised N2C 32.43.
- **sRGB Gaussian** $\sigma\in[5,50]$ on Kodak: B2U 32.34, NBR2NBR 32.10, supervised 32.51.
- **SIDD raw-RGB Benchmark**: B2U 50.79 dB / 0.991 SSIM, supervised 50.60/0.991, NBR2NBR 50.47/0.990. B2U *exceeds* supervised by 0.19 dB.
- **FMD Confocal Mice**: B2U 38.44/0.964, supervised 38.40/0.966 (essentially tied), NBR2NBR 37.07/0.960.

이 수치들은 단일-이미지 자기지도 디노이저가 *공정한 비교*에서 supervised baseline을 초월할 수 있음을 보임.

#### English
- **sRGB Gaussian** $\sigma=25$ on Kodak: BM3D 32.05; N2V 30.32; R2R 32.25; **B2U 32.27**; supervised N2C 32.43.
- **sRGB Gaussian** $\sigma\in[5,50]$ on Kodak: B2U 32.34, NBR2NBR 32.10, supervised 32.51.
- **SIDD raw-RGB Benchmark**: B2U 50.79/0.991, supervised 50.60/0.991, NBR2NBR 50.47/0.990. B2U *exceeds* supervised by 0.19 dB.
- **FMD Confocal Mice**: B2U 38.44/0.964 vs supervised 38.40/0.966 (tied), NBR2NBR 37.07/0.960.

These numbers establish that single-image self-supervised denoisers can exceed supervised baselines under fair comparison.

### 6.2 Reproducibility checklist / 재현성 체크리스트

#### 한국어
- 백본: modified U-Net (Laine 2019 동일 구조).
- 손실: re-visible loss + regulariser, $\eta=1$, $\lambda_s=2 \to \lambda_f=20$ annealing.
- 학습: Adam lr=3e-4 (sRGB) / 1e-4 (real), batch 4, 100 epochs.
- 추론: 단일 forward $f_\theta(y)$ — MC averaging 불필요.
- 코드 공개: https://github.com/demonsjin/Blind2Unblind.

#### English
- Backbone: modified U-Net (same as Laine 2019).
- Loss: re-visible + regulariser, $\eta=1$, $\lambda_s=2\to\lambda_f=20$ annealed.
- Training: Adam lr=3e-4 (sRGB) / 1e-4 (real), batch 4, 100 epochs.
- Inference: single forward $f_\theta(y)$ — no MC averaging needed.
- Code released: https://github.com/demonsjin/Blind2Unblind.

---

## 7. References / 참고문헌

- Wang, Z., Liu, J., Li, G., & Han, H., "Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots", *CVPR 2022*, pp. 2027–2036. [DOI: 10.1109/CVPR52688.2022.00207]
- Krull, A., Buchholz, T.-O., & Jug, F., "Noise2Void — Learning Denoising from Single Noisy Images", *CVPR 2019*.
- Lehtinen, J., et al., "Noise2Noise: Learning Image Restoration without Clean Data", *ICML 2018*.
- Batson, J., & Royer, L., "Noise2Self: Blind Denoising by Self-Supervision", *ICML 2019*.
- Laine, S., Karras, T., Lehtinen, J., & Aila, T., "High-Quality Self-Supervised Deep Image Denoising", *NeurIPS 2019*.
- Quan, Y., Chen, M., Pang, T., & Ji, H., "Self2Self with Dropout", *CVPR 2020*.
- Pang, T., Zheng, H., Quan, Y., & Ji, H., "Recorrupted-to-Recorrupted (R2R)", *CVPR 2021*.
- Huang, T., Li, S., Jia, X., Lu, H., & Liu, J., "Neighbor2Neighbor", *CVPR 2021*.
- Moran, N., Schmidt, D., Zhong, Y., & Coady, P., "Noisier2Noise", *CVPR 2020*.
- Ronneberger, O., Fischer, P., & Brox, T., "U-Net: Convolutional Networks for Biomedical Image Segmentation", *MICCAI 2015*.
- Abdelhamed, A., Lin, S., & Brown, M. S., "A High-Quality Denoising Dataset for Smartphone Cameras (SIDD)", *CVPR 2018*.
- Code: https://github.com/demonsjin/Blind2Unblind
