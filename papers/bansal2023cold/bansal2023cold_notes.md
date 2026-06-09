---
title: "Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise"
authors: Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie S. Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, Tom Goldstein
year: 2023
journal: "Advances in Neural Information Processing Systems (NeurIPS) 36"
doi: "arxiv:2208.09392"
topic: Low-SNR Imaging / Generalised Diffusion, Deterministic Image Restoration
tags: [cold-diffusion, generalised-diffusion, deterministic-degradation, blur, masking, downsampling, snowification, animorphosis, iterative-restoration, noise-free-diffusion, bansal-goldstein]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 27. Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise / 콜드 디퓨전 — 노이즈 없이 임의 영상 변환 역전

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 확산 모델 (diffusion model)의 정설을 정면으로 흔든다. **표준 diffusion model**은 가우시안 노이즈 추가→제거의 alternation을 통해 동작하며, 이론적 정당화 (score matching, Langevin dynamics, ELBO)는 모두 *Gaussian noise의 무작위성*에 의존한다. 그러나 저자들은 다음을 실증한다:

> *Diffusion-style 생성·복원은 노이즈 없이도, 어떤 결정적 (deterministic) 영상 변환으로도 동작한다.*

구체적으로 forward process를 $x_t = D(x_0, t)$로 일반화 (단, $D$는 *임의의* degradation operator: blur, masking, downsampling, "snowification", "animorphosis" 등 — 모두 *결정적*). 그리고 restoration network $R_\theta(x_t, t) \approx x_0$를 simple $\ell_1$ loss로 학습. Test 시 standard diffusion sampling (Algorithm 1)은 결정적 degradation에서는 작동하지 않지만, 저자들은 **개선된 sampler (Algorithm 2)**:
$$
x_{s-1} = x_s - D(\hat x_0, s) + D(\hat x_0, s-1),\quad \hat x_0 = R(x_s, s)
$$
가 **$R$이 완벽한 inverse가 아니어도 결정적 degradation을 정확히 invert함**을 first-order Taylor 분석으로 증명 (§3.3). 핵심은 Algorithm 2가 *consistent* — 즉 모든 step에서 $x_s = D(x_0, s)$를 유지 (완벽한 $R$ 가정 시), 그리고 *imperfect $R$에서도 안정* (오차가 누적되지 않음).

저자들은 (i) blur (deblurring), (ii) inpainting (masking), (iii) super-resolution (downsampling), (iv) "snowification" (ImageNet-C operator), (v) "animorphosis" (animal-image overlay) 등 **5개 매우 다른 degradation에서 generative behaviour 입증**. CIFAR-10/CelebA에서 unconditional 생성 FID도 측정하여 표준 (hot) diffusion과 *경쟁할 수준*임을 보임. 결론: **Gaussian noise는 diffusion 모델의 *필수 요소가 아니다***. 이론적 토대 (Langevin, score matching, ELBO)가 unique한 가능성이 아니라 *한 가지 경로*임을 인정해야 한다.

### English
This paper challenges a foundational assumption of diffusion models: **noise is not necessary for diffusion**. Standard diffusion uses Gaussian noise as the forward corruption — and the entire theoretical scaffolding (score matching, Langevin dynamics, variational ELBO) rests on Gaussian-noise properties. The authors demonstrate empirically that:

> *Generative diffusion behaviour persists when forward noise is replaced by any **deterministic** image transform.*

They define generalised diffusion via $x_t = D(x_0, t)$ for *any* degradation $D$ (blur, masking, downsampling, "snowification", "animorphosis"), train a restoration network $R_\theta(x_t, t) \approx x_0$ with simple $\ell_1$ loss, and propose an **improved sampler (Algorithm 2)**:
$$
x_{s-1} = x_s - D(\hat x_0, s) + D(\hat x_0, s-1),\quad \hat x_0 = R(x_s, s),
$$
which they prove (§3.3, Taylor argument) **inverts smooth $D$ exactly even when $R$ is imperfect**. They demonstrate generative behaviour on five wildly different degradations and report FID scores competitive with standard noise-based diffusion on CIFAR-10/CelebA. The take-away: the Gaussian-noise framework underlying current theoretical understanding is *one option* among many — a generalised theory of diffusion is needed.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction & §2 Background / 서론과 배경

#### 한국어
- 표준 diffusion: Gaussian noise 추가 후 noise를 점진적으로 제거 → photo-realistic 영상.
- 이론적 해석:
  1. **Langevin dynamics** (Sohl-Dickstein 2015, Song-Ermon 2019): score $\nabla \log p(x)$를 따라 random walk. *Gaussian noise 필수.*
  2. **Variational inference** (Ho-Jain-Abbeel 2020): forward chain의 ELBO 최대화. *Gaussian conditional posterior 가정.*
  3. **Score matching** (Vincent 2011): denoiser ↔ score 동치. *Gaussian noise corruption 가정.*
- *모든 이론적 정당화*가 Gaussian noise의 statistical 특성에 의존 — random walk의 reversal 가능성, low-density region에서의 score signal 보강 등.
- **본 논문의 도발적 질문**: noise 없이도 diffusion 작동하는가?
- 가능한 답: 우연일 수 없음 — 5가지 다른 deterministic degradation (blur, mask, down-sampling, snowification, animorphosis)에서 모두 작동. 일반화된 framework가 더 적절.

#### English
- Standard diffusion theoretical foundations (Langevin, ELBO, score matching) all rely on Gaussian-noise properties. This paper asks: **is noise necessary?**
- Answer (empirical): no — replacing Gaussian noise with any deterministic image transform preserves diffusion-style generative and restoration behaviour. This calls for a generalised theoretical framework.

---

### Part II: §3 Generalized Diffusion / 일반화된 확산

#### 한국어
- **Forward (degradation) operator**: $D(x_0, t)$, $t \in [0, T]$, 연속적 변화 + boundary $D(x_0, 0) = x_0$.
- **Restoration operator**: $R(x_t, t) \approx x_0$, neural network.
- 학습 손실 (Eq. 1):
$$
\min_\theta \mathbb E_{x \sim \mathcal X}\,\bigl\|R_\theta(D(x, t), t) - x\bigr\|_1.
$$
$\ell_1$ 사용 (저자 선택; $\ell_2$도 가능).
- **Naive sampling (Algorithm 1)** — 표준 DDPM 형식:
```
x_{s-1} = D(R(x_s, s), s-1)   for s = T, T-1, ..., 1.
```
- *완벽한* $R$이면: $R(D(x_0, s), s) = x_0$ → $x_{s-1} = D(x_0, s-1)$. 정상 동작.
- *Imperfect* $R$이면: 오차가 누적될 수 있음. 특히 $D$가 smooth/differentiable인 cold diffusion 경우 노이즈 보정이 없어 오차가 빠르게 폭발 (Fig. 2 top: deblurring에서 compounding artifacts).
- **Improved sampling (Algorithm 2, key contribution)**:
$$
\boxed{\;x_{s-1} = x_s - D(\hat x_0, s) + D(\hat x_0, s-1),\;\hat x_0 = R(x_s, s).\;}
$$
- 직관: $x_s$의 *현재 추정값* $D(\hat x_0, s)$ (이는 $R$ 오차로 $x_s$와 다름)을 *제거*하고, 한 step 더 깨끗한 $D(\hat x_0, s-1)$을 *추가*. 결과: $R$의 오차가 두 항 모두에 동일하게 나타나 *상쇄*.

- **§3.3 Theorem (Taylor 안정성)**: $D(x, s) = x + s\cdot e + \mathcal O(s^2)$ for some vector $e$ (smooth degradation의 first-order 전개). Algorithm 2:
$$
x_{s-1} = D(x_0, s) - D(R(x_s, s), s) + D(R(x_s, s), s-1)
= x_0 + s\cdot e - sR\cdot \tilde e + (s-1)R\cdot \tilde e
$$
계산 결과 $x_{s-1} = x_0 + (s-1)\cdot e = D(x_0, s-1)$ — *$R$의 정확도와 무관*하게 정확한 forward chain을 따름. by induction → exact reconstruction.
- Algorithm 1은 이 안정성 결여 — $x_0 \ne D(R(x_0, 0), 0) = R(x_0, 0)$일 때 fixed point가 깨짐.

#### English
- **Two algorithms**: Naive (Algorithm 1) — works for noisy diffusion (Gaussian noise during training corrects errors at test time), but fails for cold (smooth-degradation) diffusion. Improved (Algorithm 2) — uses error cancellation:
$$
x_{s-1} = x_s - D(\hat x_0, s) + D(\hat x_0, s-1).
$$
- **Theorem (§3.3)**: For smooth $D(x, s) = x + s\cdot e + \mathcal O(s^2)$, Algorithm 2 produces *exactly* $x_s = D(x_0, s)$ for all $s < t$ — independent of $R$ accuracy. This is the central technical insight.
- Algorithm 1 lacks this property because $x_0$ is not even a fixed point of its update rule when $R$ is imperfect ($x_0 \ne D(R(x_0, 0), 0) = R(x_0, 0)$).

---

### Part III: §4 Generalized Diffusions with Various Transformations / 다양한 변환에서의 일반화 확산

이 절에서는 5개의 서로 다른 degradation으로 cold diffusion을 학습/sampling하여 generative & restoration behaviour를 입증. 모든 실험에서 FID는 reconstructed test image의 품질 측정.

#### 4.1 Deblurring (Gaussian blur) / 디블러링

##### 한국어
- Forward: $x_t = G_t * x_{t-1} = G_t * \cdots * G_1 * x_0 = \bar G_t * x_0$, $G_s$는 Gaussian blur kernel sequence.
- 학습: $\ell_1$ loss로 deblurring CNN 학습 (MNIST, CIFAR-10, CelebA).
- 결과 (Fig. 3, Table 1):
  - "Direct reconstruction" $R(D(x, T), T)$: blurred 영상에서 single shot 복원 → RMSE/PSNR 좋지만 perceptual 측면에서 흐릿.
  - "Algorithm 2 sampling": iterative sampling → RMSE/PSNR 약간 *나빠지지만* FID 향상 (perceptual quality 향상). 이는 sampling이 manifold로 더 잘 끌어당겨 결과가 sharp.
- **해석**: Blur sampling routine은 $D(\hat x_0, s) - D(\hat x_0, s-1) = \bar G_s * x_0 - \bar G_{s-1} * x_0$이 *band-pass filter* — 매 step에서 degradation 시 제거된 frequency를 다시 추가. Blurring의 frequency-removal sequence를 *역순*으로 frequency-add 하는 셈.

##### English
- Forward chain is iterated Gaussian blur. Training is $\ell_1$ deblurring on MNIST/CIFAR/CelebA.
- Algorithm 2 is interpretable as a band-pass-filter cascade: each step adds back the frequencies removed at one degradation step. This makes the iterative sampler conceptually like inverse-filtering with a learned prior.
- Quantitatively, RMSE/PSNR are slightly worse with Algorithm 2 vs direct reconstruction, but FID is much better — sampler stays on the natural-image manifold.

#### 4.2 Inpainting (masking) / 인페인팅

##### 한국어
- $D(x, t)$: 영상의 일부 픽셀을 zero (또는 mean) 으로 점진적으로 mask. severity $t$에 따라 mask 영역 확대.
- 학습 후 Algorithm 2 sampling: hole 영역이 자연스럽게 채워짐. 표준 diffusion-기반 inpainting (RePaint, DDRM)과 *다른* approach — 노이즈 없음.

##### English
- Mask grows with $t$. Algorithm 2 fills the masked region in a coarse-to-fine manner. Demonstrates that inpainting via diffusion does *not* require Gaussian noise.

#### 4.3 Super-resolution (downsampling) / 초해상화

##### 한국어
- $D(x, t)$: 점진적 downsampling. 결과는 standard diffusion-based SR (DDRM, Cascaded DM)과 비교.

##### English
- Downsample-upsample cascade. Generates high-resolution images from low-resolution inputs without any Gaussian noise.

#### 4.4 Snowification (ImageNet-C operator) / 눈 효과

##### 한국어
- ImageNet-C dataset의 *snow* corruption operator를 forward로 사용. 영상에 점진적으로 눈송이 패턴 overlay.
- Sampling: 눈 패턴 제거 + 자연스러운 영상 복원. unique — random noise가 아닌 *struct​ured* corruption.

##### English
- Uses the ImageNet-C "snow" operator. Demonstrates that *structured* (non-random) corruptions are also invertible via diffusion-style sampling.

#### 4.5 Animorphosis (animal-image overlay) / 애니모르포시스

##### 한국어
- 가장 도발적 실험: forward $D(x, t) = (1-\alpha_t) x + \alpha_t z$, $z$는 *AFHQ dataset에서 random sampled animal 영상*. $\alpha_t$ schedule에 따라 영상이 점진적으로 동물 영상으로 morphed.
- Sampling 결과: 동물 영상에서 시작해 *human face*로 reverse — 학습 데이터 (CelebA)의 manifold로 끌어당김.
- 의미: forward chain의 "noise"가 데이터 분포 자체와 무관해도 ($z$는 동물; $x$는 사람) reverse sampling이 동작 — generative behaviour는 *noise statistics*가 아닌 *coverage*에 의존.

##### English
- Most provocative: forward replaces the image with a random animal (from AFHQ). Reverse sampling produces faces (CelebA training distribution). Generative behaviour persists even when the "noise" is structurally far from random.
- This is the strongest empirical case that diffusion is *not* about noise at all — it's about coarse-to-fine restoration toward a learned manifold.

---

### Part IV: §5 Generation from Cold Diffusions / 콜드 디퓨전으로부터의 영상 생성

#### 한국어
- Unconditional generation: 학습 후 *random initial* $x_T \sim \text{some distribution}$에서 sampling.
- Cold diffusion에서 $x_T$가 deterministic이면 어떤 random init? 저자들은 random animal image (animorphosis), random Gaussian-init (deblur 시작점), random masked patch (inpainting) 등 task-specific noise 사용.
- **CIFAR-10 unconditional FID** (논문 Tables, Section 5):
  - Standard noise-based DDPM: ~3.2 FID.
  - Cold diffusion w/ blur: ~80 FID (worse but recognizable as digits/CIFAR objects).
  - Cold diffusion w/ animorphosis: ~30+ FID (better — random animal init provides variability).
- **CelebA**: cold diffusion w/ blur achieves human-like faces from coarse blur init.
- 결론: Cold diffusion은 noise-based보다 FID가 *조금 나쁘지만* generative behaviour는 분명히 존재.

#### English
- Unconditional generation works in cold diffusion but FID is somewhat worse than noise-based (e.g., CIFAR-10 DDPM ~3.2 vs cold-blur ~80). Animorphosis (random animal init) helps because it provides genuine input variability.
- The paper's claim is qualitative — generative behaviour exists — not that cold diffusion supplants noise-based.

---

### Part V: §6 Discussion / 논의

#### 한국어
- Cold diffusion이 정말로 작동한다는 사실은 현 diffusion 이론의 *허점*을 드러낸다:
  - Score matching은 Gaussian noise를 가정 — 다른 corruption에선 score 정의가 모호.
  - Langevin dynamics는 random walk를 가정 — deterministic step에선 reversibility 보장 불명확.
  - ELBO는 Gaussian conditional posterior를 가정 — 다른 forward에서는 *변분 lower bound*가 자명하지 않음.
- 더 일반적인 이론이 필요. 가능한 방향: (i) operator-theoretic — degradation operator의 spectral 분석, (ii) flow-based — $D$를 ODE의 시간 진화로 해석, (iii) constrained optimization — sampling을 manifold projection로 해석.
- Open question: 어떤 degradation이 더 좋은 generative model을 만드는가? Snow와 animorphosis는 왜 blur보다 잘 작동? Diversity vs invertibility trade-off?

#### English
- The success of cold diffusion exposes gaps in current theory: score matching, Langevin, and ELBO all assume Gaussian noise. A more general theory should be operator-theoretic, flow-based, or projection-based.
- Open: which degradations produce the best generative models? The empirical ranking (animorphosis > snow > blur ≈ inpainting > down-sampling) suggests *coverage* and *invertibility* both matter.

---

## 3. Key Takeaways / 핵심 시사점

1. **Diffusion does not require noise — 확산에 노이즈는 필수가 아니다.**
   - **English**: The central empirical claim. Five different deterministic degradations (blur, masking, downsampling, snow, animorphosis) all support diffusion-style generation and restoration.
   - **한국어**: 5가지 매우 다른 결정적 degradation이 모두 diffusion-style 생성·복원을 지원 — Gaussian noise는 *한 가지 선택지*에 불과.

2. **Algorithm 2 is the key technical contribution — 알고리즘 2가 핵심 기술 기여.**
   - **English**: $x_{s-1} = x_s - D(\hat x_0, s) + D(\hat x_0, s-1)$ uses *error cancellation* to be robust to imperfect $R$. Standard DDPM-style sampler (Algorithm 1) fails for smooth $D$.
   - **한국어**: 알고리즘 2는 $R$의 오차가 두 항에 동일하게 나타나 *상쇄*되는 구조 — Algorithm 1은 smooth $D$에서 실패.

3. **First-order Taylor proof of exact inversion — first-order 테일러로 정확 역변환 증명.**
   - **English**: For $D(x, s) = x + s\cdot e + \mathcal O(s^2)$, Algorithm 2 produces $x_{s-1} = D(x_0, s-1)$ regardless of $R$'s accuracy. Mathematical guarantee underlying empirical robustness.
   - **한국어**: smooth $D$에 대해 Algorithm 2가 $R$ 정확도와 무관하게 정확한 chain을 따른다는 first-order 증명.

4. **Restoration network can be any architecture with simple loss — 복원 네트워크는 단순 손실로 OK.**
   - **English**: $\ell_1$ loss + standard U-Net suffices. No complex score-matching loss, no variational objective. The architecture is the same as a supervised regressor.
   - **한국어**: $\ell_1$ 손실 + 표준 U-Net으로 충분. 복잡한 score matching loss나 변분 목적함수 불필요 — supervised regressor와 동일.

5. **Theory of diffusion models is incomplete — 확산 모델 이론은 미완성이다.**
   - **English**: All current theoretical frameworks (Langevin, ELBO, score matching) require Gaussian noise. Cold diffusion's success means diffusion is more general than these theories explain.
   - **한국어**: 현재 모든 이론 framework가 Gaussian noise에 의존. Cold diffusion의 성공은 diffusion이 이들 이론보다 더 일반적임을 시사.

6. **Animorphosis: noise distribution can be far from data — 애니모르포시스: 노이즈 분포가 데이터에서 멀어도 OK.**
   - **English**: Forward replaces images with random animals (AFHQ), reverse generates faces (CelebA). Generative behaviour persists even when "noise" is structurally distant from the data.
   - **한국어**: forward는 random animal로 변환, reverse는 face 생성 — "noise"가 데이터와 구조적으로 멀어도 동작.

7. **Generation FID slightly worse than noise-based — 생성 FID는 노이즈 기반보다 약간 나쁘다.**
   - **English**: CIFAR-10 unconditional: ~80 FID for cold-blur vs ~3.2 for DDPM. Cold diffusion is not yet a *replacement* — it's an *alternative paradigm* with different strengths/weaknesses.
   - **한국어**: CIFAR-10 unconditional: cold-blur ~80 FID vs DDPM ~3.2. Cold diffusion은 *대체*가 아닌 *대안*.

8. **Practical inverse problems benefit — 실용적 역문제에 도움.**
   - **English**: For deblurring, inpainting, SR, the cold-diffusion sampler often gives sharper, more natural results than direct $R$ application. Useful when noise model is poorly characterised (industrial, microscopy).
   - **한국어**: deblurring, inpainting, SR 등 실용적 역문제에서 cold diffusion sampler가 direct $R$보다 더 sharp하고 자연스러운 결과 — 노이즈 모델이 불명확한 산업·현미경 응용에 유리.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Generalised diffusion / 일반화 확산
**Forward (degradation)**:
$$
x_t = D(x_0, t),\qquad D(x_0, 0) = x_0,\qquad t \in [0, T].
$$
$D$는 임의의 (보통 $t$에 따라 *연속적이고 differentiable한*) 영상 변환 — Gaussian noise (standard diffusion), iterative blur, masking, downsampling, snow, animorphosis 등.

**Restoration**:
$$
R_\theta(x_t, t) \approx x_0.
$$
**Training**:
$$
\min_\theta \mathbb E_{x\sim\mathcal X, t \sim \text{Uniform}[0,T]}\,\bigl\|R_\theta(D(x, t), t) - x\bigr\|_1. \tag{Eq. 1}
$$

### 4.2 Naive sampler (Algorithm 1)
```
Input: degraded sample x_t
For s = t, t-1, ..., 1:
    x_hat_0 = R(x_s, s)
    x_{s-1} = D(x_hat_0, s-1)
Return x_0
```
완벽한 $R$이면 정확하지만 imperfect $R$에 대해서는 cold diffusion에서 fail.

### 4.3 Improved sampler (Algorithm 2, key contribution)
```
Input: degraded sample x_t
For s = t, t-1, ..., 1:
    x_hat_0 = R(x_s, s)
    x_{s-1} = x_s - D(x_hat_0, s) + D(x_hat_0, s-1)
Return x_0
```

**핵심 식**:
$$
\boxed{\;x_{s-1} = x_s - D(R(x_s, s), s) + D(R(x_s, s), s-1).\;}
$$

### 4.4 Stability theorem / 안정성 정리
**Setup**: smooth degradation의 first-order 전개:
$$
D(x, s) = x + s\cdot e + \mathcal O(s^2),\qquad D(x, 0) = x.
$$
$e$ = 1-step "degradation direction" vector. (예: blur의 경우 $e \propto \nabla^2 x$ — Laplacian.)

**Claim**: Algorithm 2의 update를 위 expansion에 대입:
$$
\begin{aligned}
x_{s-1} &= x_s - D(R(x_s, s), s) + D(R(x_s, s), s-1)\\
&= D(x_0, s) - D(R(x_s, s), s) + D(R(x_s, s), s-1)\\
&= [x_0 + s e + \mathcal O(s^2)] - [R(x_s, s) + s e + \mathcal O(s^2)] + [R(x_s, s) + (s-1) e + \mathcal O((s-1)^2)]\\
&= x_0 + s e - s e + (s-1) e + \mathcal O(s^2) - \mathcal O(s^2) + \mathcal O((s-1)^2)\\
&= x_0 + (s-1) e + \mathcal O(s^2)\\
&= D(x_0, s-1) + \mathcal O(s^2).
\end{aligned}
$$
**결론**: $x_{s-1} \approx D(x_0, s-1)$, **$R$의 정확도와 무관**. By induction, Algorithm 2는 어떤 $R$에 대해서도 $x_s = D(x_0, s)$를 따른다.

### 4.5 Comparison to Algorithm 1
Algorithm 1: $x_{s-1} = D(R(x_s, s), s-1)$.
- 전개: $x_{s-1} = R(x_s, s) + (s-1) e + \mathcal O(s^2)$.
- $R(x_s, s) \ne x_0$ (imperfect) → $x_{s-1}$이 $D(x_0, s-1) = x_0 + (s-1) e$가 아니다 → 오차 누적 → Fig. 2 top의 compounding artifacts.

### 4.6 Worked numerical example: 1-D blur diffusion / 수치 예시
**Setup**: 1-D signal $x_0 = [1, 0, 0, 0]$. Degradation $D(x, t)$ = $t$-step 3-tap averaging (blurring).
- $D(x_0, 1) = [0.5, 0.5, 0, 0]$ (1-step blur).
- $D(x_0, 2) = [0.25, 0.5, 0.25, 0]$ (2-step blur).
- $D(x_0, 3) = [0.125, 0.375, 0.375, 0.125]$ (3-step blur).

**Imperfect restoration**: assume $R(D(x_0, t), t) = x_0 + 0.1\cdot \mathbf 1_{\text{noise}}$ for some bias.

**Algorithm 1** at step $s = 3$:
- $\hat x_0 = R(x_3, 3) = x_0 + 0.1 \mathbf 1$.
- $x_2 = D(\hat x_0, 2) = D(x_0, 2) + 0.1 D(\mathbf 1, 2) = [0.25, 0.5, 0.25, 0] + 0.1\cdot[0.75, 1, 0.75, 0.5]$.
- 오차 누적: 다음 step에서 더 많은 bias, … → final $x_0$ has accumulated error.

**Algorithm 2** at step $s = 3$:
- $\hat x_0 = x_0 + 0.1 \mathbf 1$.
- $x_2 = x_3 - D(\hat x_0, 3) + D(\hat x_0, 2)$.
- Expand: $x_2 = D(x_0, 3) - [D(x_0, 3) + 0.1 D(\mathbf 1, 3)] + [D(x_0, 2) + 0.1 D(\mathbf 1, 2)]$.
- $= D(x_0, 2) + 0.1\cdot[D(\mathbf 1, 2) - D(\mathbf 1, 3)]$.
- $D(\mathbf 1, 2) - D(\mathbf 1, 3)$는 *$D$가 선형이고 $\mathbf 1$가 invariant* (constant function이 averaging의 fixed point) → 값이 매우 작음 (0에 근접).
- 따라서 $x_2 \approx D(x_0, 2)$ — *bias가 거의 사라짐*.

이 toy 예시는 정리의 실제 작동을 보여준다.

### 4.7 Loss interpretation as band-pass / 손실의 band-pass 해석
For blur diffusion, $D(\hat x_0, s) - D(\hat x_0, s-1) = (\bar G_s - \bar G_{s-1}) * \hat x_0$. Difference of Gaussians = band-pass filter. So Algorithm 2 sampling progressively re-introduces frequencies in band-pass order, from coarse (high $s$) to fine (low $s$). This is the *frequency-domain interpretation* of cold-blur diffusion.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
2015      Sohl-Dickstein+ — non-equilibrium thermodynamics generative model (proto-diffusion)
2019      Song-Ermon — NCSN (Gaussian-noise score matching, annealed Langevin)
2020      Ho-Jain-Abbeel — DDPM (Gaussian-noise variational inference)
2020      Song+ — DDIM (deterministic accelerated sampling, still Gaussian-noise forward)
2021      Song+ — Score-based SDEs (continuous-time Gaussian-noise diffusion)
2021      Kadkhodaie-Simoncelli (paper #25) — denoiser-as-prior, Gaussian-noise
2021      Dhariwal-Nichol — Diffusion beats GANs (Gaussian-noise)
2022      Karras+ — EDM (elucidating diffusion model design space, Gaussian-noise)
2022      Kawar+ — DDRM (paper #26, Gaussian-noise diffusion for inverse problems)
2022      Rissanen+ — Generative Modelling with Inverse Heat Dissipation (concurrent — blur diffusion!)
2022      Hoogeboom-Salimans — Blurring Diffusion (concurrent)
2023 ★★   BANSAL+ — COLD DIFFUSION (THIS PAPER)
                    ↳ Five degradations: blur, mask, downsample, snow, animorphosis
                    ↳ Algorithm 2 with first-order Taylor stability proof
                    ↳ Generative behaviour without Gaussian noise
2023      Daras+ — Soft Diffusion (interpolation between cold and hot)
2023      Blattmann+ — Latent diffusion-based video (Gaussian noise still standard)
2024+     Operator-theoretic generalisations (open)
```

이 논문은 *diffusion model 이해의 패러다임 분기*. 이전엔 "Gaussian noise = diffusion의 정의"였다면, 본 논문 후엔 "Gaussian noise = diffusion의 *한 인스턴스*"로 바뀜. 동시기 작업 (Rissanen+ 2022, Hoogeboom-Salimans 2022)이 blur-only diffusion을 제안했지만 본 논문이 가장 일반적·실험적으로 광범위.

This paper marks a *paradigm-divergence* in diffusion-model understanding. Pre-2022, Gaussian noise was synonymous with diffusion; post-2022, Gaussian noise is one instance among many. Concurrent works (Inverse Heat Dissipation, Blurring Diffusion) propose blur-only variants; this paper is the most general and empirically broad statement.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Sohl-Dickstein+ (2015)** Non-equilibrium thermodynamics | Foundational | First diffusion model with Gaussian-noise forward chain. Cold diffusion challenges the assumption of *Gaussian* noise being necessary. |
| **Song-Ermon (2019)** NCSN | Theoretical predecessor | Gaussian-noise score matching + annealed Langevin. Cold diffusion's success exposes that Langevin dynamics is *one* way, not the only way. |
| **Ho+ (2020)** DDPM | Direct conceptual reference | DDPM's Gaussian-noise variational ELBO is the standard cold diffusion compares to. Algorithm 1 mirrors DDPM's reverse step structure. |
| **Song+ (2020)** DDIM | Sibling deterministic sampler | DDIM is deterministic *sampling* with Gaussian-noise *forward*. Cold diffusion goes further: deterministic *forward* too. |
| **Kadkhodaie-Simoncelli (2021)** (paper #25) | Cited explicitly | Cold diffusion's introduction cites this paper for the role of noise in diffusion. Both papers strip the inverse-problem solver to its essentials, reaching different conclusions about noise. |
| **Kawar+ (2022)** DDRM (paper #26) | Inverse-problem context | DDRM uses pre-trained DDPM for linear inverse problems with Gaussian noise. Cold diffusion shows that the same problems (blur, mask, SR) can be solved without any noise. |
| **Rissanen+ (2022)** Inverse Heat Dissipation | Concurrent independent work | Proposes blur-only diffusion (a special case of cold diffusion). Different theoretical justification (heat equation reversal). |
| **Hoogeboom-Salimans (2022)** Blurring Diffusion | Concurrent independent work | Another blur-based diffusion variant. Cold diffusion is more general and includes 4 other degradations. |
| **Daras+ (2023)** Soft Diffusion | Direct successor | Bridges cold and hot diffusion by interpolating between Gaussian noise and deterministic degradation. Builds on cold diffusion's framework. |
| **Karras+ (2022)** EDM | Theoretical foil | EDM elucidates the design space of *Gaussian-noise* diffusion models. Cold diffusion shows that a much larger design space exists outside EDM's framework. |
| **Lehtinen+ (2018)** Noise2Noise (paper #16) | Indirect | The restoration network $R$ in cold diffusion is simply a supervised CNN regressor with $\ell_1$ loss — analogous to a Noise2Noise-style trained denoiser, but with deterministic targets. |
| **Buades+ (2005)** NL-means | Conceptual ancestor | NL-means/BM3D-style iterative restoration without any generative prior — same flavour as Algorithm 2 but without the diffusion-chain framework. |
| **Romano-Elad-Milanfar (2017)** RED | Iterative-restoration framework | RED uses denoiser as regularizer in iterative scheme. Cold diffusion's Algorithm 2 is in similar spirit but with explicit forward/reverse chain. |

---

## 7. Failure Modes and Limits / 실패 양상과 한계

### 한국어
1. **FID는 hot diffusion보다 나쁨**: CIFAR-10 unconditional FID는 cold-blur ~80 vs DDPM ~3.2. Cold diffusion은 *대체*가 아닌 *대안*.
2. **Random initialization 의존**: 결정적 forward에서 unconditional generation을 위해 random init이 필요 (animorphosis는 random animal, blur는 random Gaussian etc.). Forward chain이 진정한 mode-collapsing operator가 아니므로 input variability를 따로 주입.
3. **이론적 보장 1차원**: Algorithm 2의 안정성 증명은 first-order Taylor — high-order (large step) 에서는 saturate. 실험은 작은 step size 사용.
4. **Smooth degradation에 한정**: 증명은 $D$가 smooth/differentiable임을 가정. 비연속 (e.g., binary masking)에서는 다른 분석 필요. 실험은 작동하지만 이론적 보장은 없음.
5. **Animorphosis는 extreme**: 학습 데이터 (CelebA)와 forward noise (animal) 사이의 거리가 너무 크면 학습이 불안정해질 수 있음. 저자도 수렴이 standard보다 어려움 인정.
6. **Compute 비용**: $\ell_1$ regression은 cheap이지만 sampling은 여전히 N forward passes. Real-time보다는 offline restoration용.

### English
1. **FID gap**: cold diffusion's unconditional FID is significantly worse than DDPM (e.g., 80 vs 3.2 on CIFAR-10). Not a replacement; an alternative paradigm.
2. **Random-init dependence**: deterministic forward means input variability must be added externally (random animal for animorphosis, Gaussian for blur init).
3. **First-order proof**: Theorem 3.3 is first-order Taylor; high-order errors are not bounded but empirically small.
4. **Smoothness assumption**: theory assumes differentiable $D$. Discrete masking is empirical.
5. **Animorphosis training instability**: large distance between training distribution and forward "noise" makes training harder; authors acknowledge tougher convergence than DDPM.
6. **N forward passes for sampling**: not real-time.

---

## 8. References / 참고문헌

- Bansal, A., Borgnia, E., Chu, H.-M., Li, J. S., Kazemi, H., Huang, F., Goldblum, M., Geiping, J., & Goldstein, T. "Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise", *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2023. [arXiv:2208.09392]
- Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., & Ganguli, S. "Deep unsupervised learning using nonequilibrium thermodynamics", *Proc. ICML*, 2015.
- Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models" (DDPM), *Proc. NeurIPS*, 2020.
- Song, J., Meng, C., & Ermon, S. "Denoising Diffusion Implicit Models" (DDIM), *Proc. ICLR*, 2021.
- Song, Y., & Ermon, S. "Generative modeling by estimating gradients of the data distribution" (NCSN), *Proc. NeurIPS*, 2019.
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. "Score-based generative modeling through stochastic differential equations", *Proc. ICLR*, 2021.
- Vincent, P. "A connection between score matching and denoising autoencoders", *Neural Computation*, 23(7), 1661–1674 (2011).
- Kadkhodaie, Z., & Simoncelli, E. P. "Stochastic Solutions for Linear Inverse Problems Using the Prior Implicit in a Denoiser", *Proc. NeurIPS*, 2021. [arXiv:2007.13640]
- Kawar, B., Elad, M., Ermon, S., & Song, J. "Denoising Diffusion Restoration Models" (DDRM), *Proc. NeurIPS*, 2022. [arXiv:2201.11793]
- Rissanen, S., Heinonen, M., & Solin, A. "Generative Modelling With Inverse Heat Dissipation", *Proc. ICLR*, 2023. [arXiv:2206.13397]
- Hoogeboom, E., & Salimans, T. "Blurring Diffusion Models", *Proc. ICLR*, 2023. [arXiv:2209.05557]
- Daras, G., Delbracio, M., Talebi, H., Dimakis, A. G., & Milanfar, P. "Soft Diffusion: Score Matching for General Corruptions", *Proc. ICML*, 2023.
- Karras, T., Aittala, M., Aila, T., & Laine, S. "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM), *Proc. NeurIPS*, 2022.
- Dhariwal, P., & Nichol, A. "Diffusion models beat GANs on image synthesis", *Proc. NeurIPS*, 2021.
- Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. "GANs trained by a two time-scale update rule converge to a local Nash equilibrium" (FID), *Proc. NeurIPS*, 2017.
