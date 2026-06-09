---
title: "Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser"
authors: Zahra Kadkhodaie, Eero P. Simoncelli
year: 2021
journal: "Advances in Neural Information Processing Systems (NeurIPS) 34"
doi: "arxiv:2007.13640"
topic: Low-SNR Imaging / Denoiser-as-Prior, Empirical Bayes, Stochastic Sampling
tags: [denoiser-prior, empirical-bayes, miyasawa, tweedie, langevin, stochastic-sampling, score-matching, inpainting, super-resolution, deblurring, compressive-sensing, kadkhodaie-simoncelli, bf-cnn]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 25. Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser / 디노이저에 내재된 사전분포로 선형 역문제 풀기

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 한 가지 통계학의 오래된 결과 — **Miyasawa (1961) / Tweedie (Robbins 1956) 정리** — 가 현대 딥러닝 디노이저와 만나면서 어떻게 *일반적인 영상 사전분포*를 무료로 제공하는지를 보여준다. 핵심 한 줄은 다음과 같다. 가우시안 노이즈 $z\sim\mathcal N(0,\sigma^2 I)$로 오염된 관측 $y=x+z$의 MMSE 디노이저 $\hat x(y) = \mathbb E[x\mid y]$는 *관측 밀도 $p_\sigma(y)$의 로그 그래디언트*와 직접적으로 연결된다:
$$
\hat x(y) = y + \sigma^2 \nabla_y \log p_\sigma(y).
$$
즉 디노이저의 *잔차* $f(y) := \hat x(y) - y$는 $\sigma^2 \nabla_y \log p_\sigma(y)$의 추정치다. 따라서 *별도 학습 없이* — 단지 $L_2$ 손실로 학습된 bias-free CNN 디노이저(BF-CNN)만으로도 — 영상 사전의 score를 얻는다.

저자들은 이 결과를 **(i) 사전에서의 high-probability 샘플링 (image synthesis)** 과 **(ii) 임의의 선형 역문제 풀이 (inpainting, super-resolution, deblurring, compressive sensing)** 의 두 응용으로 발전시킨다. 핵심 알고리즘 (Algorithm 1, 2)은 noise level $\sigma$를 거대한 값에서 작은 값으로 점진적으로 줄이는 *coarse-to-fine stochastic gradient ascent*다. 매 step에서 디노이저 잔차로 update direction을 구하고 (deterministic), 적당량의 white noise를 다시 주입하여 (stochastic) local minima를 회피한다. 제약 sampling (constrained sampling, Algorithm 2)에서는 측정 부분공간 $M^Tx = x^c$에 직교한 성분에는 prior gradient를 적용하고, 측정 부분공간에 평행한 성분은 측정값을 향해 끌어당기는 *projected gradient*를 사용한다. DIP·DeepRED 대비 PSNR이 비슷하거나 더 높으면서 **2 orders of magnitude 빠르고**, super-resolution 4×의 경우 평균 결과가 27.14 dB → 31.20 dB (Set5)로 우월하다. 이 논문은 *score-based diffusion sampling이 인기를 얻기 직전 시점에* 같은 핵심 아이디어를 디노이저 잔차의 직접적 해석으로 단순화하여 제시한 결정적 다리 역할을 한다.

### English
The paper revives a 60-year-old statistical identity — **Miyasawa's lemma / Tweedie's formula** — and shows that any modern $L_2$-trained Gaussian denoiser implicitly provides the *score* of the noisy-image density. Concretely, for $y = x + z$, $z\sim\mathcal N(0,\sigma^2 I)$, the MMSE estimator satisfies $\hat x(y) = y + \sigma^2 \nabla_y \log p_\sigma(y)$, so the denoiser residual $f(y) = \hat x(y) - y$ is exactly $\sigma^2 \nabla_y \log p_\sigma(y)$. With a *bias-free* CNN denoiser trained over a wide range of noise levels (BF-CNN, Mohan et al. 2020), this gradient is automatically adapted to the local noise scale. The authors build on this in two ways: (1) a coarse-to-fine **stochastic gradient ascent algorithm** that draws high-probability samples from the implicit prior, and (2) a **constrained variant** that solves arbitrary linear inverse problems $x^c = M^T x$ (inpainting, super-resolution, deblurring, compressive sensing) by projecting the prior gradient onto the orthogonal complement of the measurement subspace and adding a measurement-consistency pull. Quantitatively, the method matches or beats DIP/DeepRED on Set5/Set14 super-resolution (e.g., 31.20 dB vs 30.22 dB at 4×) while being **two orders of magnitude faster** (Table 3: 9 s vs 1,584 s). Predating DDPM-style guidance methods, this paper is the cleanest derivation of "denoiser = score" for solving linear inverse problems.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction & §1.1 Image priors and manifolds / 서론과 영상 사전·매니폴드

#### 한국어
- 영상 사전분포 모델링은 고차원 자연 영상에선 분석적으로 어려움. 전통: 단순 parametric form (Gaussian, GMM) on transform domain (Fourier, wavelet).
- 최근: deep CNN denoiser의 성능은 거대한 implicit prior가 학습됨을 시사 — 그러나 prior가 *architecture + data + loss + optimizer* 의 조합으로 *얽혀 있어* 명시적으로 추출하기 어려움.
- "Plug and Play" (Venkatakrishnan+ 2013) 계열은 ADMM의 proximal operator 자리에 디노이저를 끼워 MAP 풀이 → 그러나 implicit prior에 대한 정직한 해석 부족.
- 또 다른 흐름: score matching ↔ denoising autoencoder 동치 (Vincent 2011). $\nabla_y \log p_\sigma(y) \approx (\hat x(y) - y)/\sigma^2$는 score matching에서도 도출.
- **본 논문의 새로움**: Miyasawa (1961)의 더 직접적인 결과를 사용. score matching처럼 학습 손실의 *최소화 조건*을 통한 간접 도출이 아니라, *최적 MMSE 추정값* 자체가 score를 정확히 표현. 이는 더 명확한 해석과 더 단순한 알고리즘으로 이어짐.
- 자연 영상은 $\mathbb R^N$ 안에서 *low-dimensional manifold* 근처에 집중 — 이미지를 살짝 변형(translation, dilation, intensity shift)하면 manifold 위 곡선 trajectory를 얻고 reasonable 영상이 유지됨. 사전 $p(x)$는 manifold 위에서 거의 일정·천천히 변하고 manifold 밖에서 빠르게 0으로.
- 노이즈 관측 $y = x + z$의 prior predictive density:
$$
p(y) = \int p(y\mid x) p(x) dx = \int g(y - x) p(x) dx,\qquad g(z) = \frac{1}{(2\pi\sigma^2)^{N/2}} e^{-\|z\|^2/2\sigma^2}.
$$
- $p_\sigma(y)$는 $p(x)$의 *Gaussian-blurred version* — 다양한 $\sigma$에 대해 $p_\sigma(y)$의 family는 *Gaussian scale-space* (heat-equation evolution)을 형성.

#### English
- Direct prior modelling for high-dimensional natural images is intractable. Modern CNN denoisers achieve great performance, suggesting they embed a sophisticated prior — but it is entangled with architecture, data, loss, optimizer.
- Two prior strands of work: Plug-and-Play uses denoisers as proximal operators (heuristic); score-matching shows denoising-autoencoder loss approximates score-of-Parzen-density (asymptotic).
- Authors invoke a more direct, classical result (Miyasawa 1961) saying the MMSE denoiser is *exactly* $y + \sigma^2 \nabla_y \log p_\sigma(y)$. No asymptotic appeals, no proximal-operator hand-waving — just an identity.
- Natural images concentrate on a low-dimensional manifold; $p_\sigma(y)$ is the manifold's *Gaussian scale-space* (heat-equation evolution of the prior with time = $\sigma^2$). The family of $p_\sigma$'s plays the role of a temporal scale-space, which the algorithm traverses coarse-to-fine.

---

### Part II: §1.2–1.3 Least-squares denoising and the Empirical Bayes identity / 최소제곱 디노이징과 경험적 베이즈 항등식

#### 한국어
- MMSE 디노이저의 정의:
$$
\hat x(y) = \mathbb E[x\mid y] = \int x\,p(x\mid y)\,dx = \int x \frac{p(y\mid x) p(x)}{p(y)}\,dx. \tag{Eq. 2}
$$
- **Bias-free CNN (BF-CNN; Mohan et al. 2020)**: 모든 additive bias 항을 제거한 DnCNN 변형. 두 가지 큰 장점:
  1. 모든 noise level에 자동 일반화 — 작은 σ에서 학습한 네트워크가 큰 σ에서도 동작.
  2. 국소적으로 *adaptive linear filter*로 분석 가능 — 출력 = 입력 노이즈의 어떤 subspace에 대한 projection.
- **Miyasawa의 결과 (Eq. 3)**:
$$
\boxed{\;\hat x(y) = y + \sigma^2 \nabla_y \log p_\sigma(y)\;}
$$
- **Proof sketch (Eq. 4)**:
$$
\nabla_y p_\sigma(y) = \nabla_y \int g(y-x) p(x) dx = \int (-\frac{y-x}{\sigma^2}) g(y-x) p(x) dx
= \frac{1}{\sigma^2}\int (x - y)\,p(y, x)\,dx.
$$
양변에 $\sigma^2/p_\sigma(y)$ 곱:
$$
\sigma^2 \frac{\nabla_y p_\sigma(y)}{p_\sigma(y)} = \int x\,p(x\mid y)\,dx - y\int p(x\mid y)\,dx = \hat x(y) - y.
$$
chain rule로 $\nabla_y \log p_\sigma(y) = \nabla_y p_\sigma(y) / p_\sigma(y)$ 을 적용 → Eq. 3 도출.
- **세 가지 직관적 관찰** (논문 §1.3 끝):
  1. 관련 density는 *prior $p(x)$가 아닌 noisy observation density $p_\sigma(y)$* 이다.
  2. 그래디언트는 *log* density (energy function)의 그래디언트.
  3. 한 step에 즉시 optimal solution — *iterative descent가 아닌 closed-form*. 어떤 $\sigma$에 대해서도 성립.

#### English
- The MMSE denoiser is the conditional mean (Eq. 2). Trained CNN denoisers approximate it with no analytical prior.
- **Miyasawa identity (Eq. 3)**: $\hat x(y) = y + \sigma^2 \nabla_y \log p_\sigma(y)$. Proof in two lines (Eq. 4): take $\nabla_y$ of $p_\sigma(y) = \int g(y-x) p(x) dx$, factor out $-(y-x)/\sigma^2$ from the derivative of the Gaussian kernel, multiply both sides by $\sigma^2/p_\sigma(y)$, and the right side splits into $\hat x(y) - y$.
- The relevant density is the **noisy observation density** $p_\sigma(y)$, not $p(x)$. Crucially, the identity holds for any $\sigma$: in the limit $\sigma \to 0$ the noisy density approaches the prior, and the gradient field becomes singular near the manifold; for large $\sigma$ the gradient is smooth and globally informative — this is what justifies coarse-to-fine sampling.
- BF-CNN (Mohan+ 2020) gives *automatic noise-level adaptation*: the same network can be queried at any noise level, and the residual $f(y) = \hat x(y) - y$ scales correctly. This is essential for §2's algorithm because $\sigma$ changes each iteration.

---

### Part III: §2 Drawing high-probability samples from the implicit prior / 내재 사전분포로부터 high-probability 샘플 추출

#### 한국어
- 목표: 사전 $p(x)$ — 자연 영상 manifold — 위에 있는 high-probability 영상을 random init $y_0$에서부터 *iterative gradient ascent*로 찾기.
- 매 iteration의 deterministic step: 디노이저 잔차 $f(y) = \hat x(y) - y$ 방향으로 이동 → manifold orthogonal subspace의 노이즈를 줄이고, manifold-parallel subspace의 영상 구조를 보존/생성.
- $p_\sigma(y)$의 Gaussian scale-space 해석: 큰 $\sigma$에선 $p_\sigma$가 부드럽고 manifold가 두꺼워 보이며 (낮은 차원 외부도 nontrivial probability), $\sigma$를 줄이면 manifold가 *얇아지면서 차원이 커지고* (관측 가능 detail이 늘어남), 그래디언트가 점점 sharp해진다 → 자연스러운 coarse-to-fine optimization.
- **Iteration update (Eq. 5)**:
$$
y_t = y_{t-1} + h_t f(y_{t-1}) + \gamma_t z_t,\quad z_t \sim \mathcal N(0, I).
$$
- $h_t \in [0,1]$: 그래디언트 step의 fraction. $\gamma_t$: injected white-noise amplitude.
- **Effective noise variance (Eq. 6)**:
$$
\sigma_t^2 = (1 - h_t)^2 \sigma_{t-1}^2 + \gamma_t^2.
$$
첫 항: 디노이저 보정 후 *남은* 노이즈 분산. 둘째 항: 새로 주입된 노이즈.
- **수렴 보장**: $\sigma_t^2 = (1 - \beta h_t)^2 \sigma_{t-1}^2$, $\beta \in [0,1]$로 두면 (Eq. 7) effective noise를 매 step 일정 비율로 줄일 수 있음. 이를 Eq. 6과 결합 (Eq. 8):
$$
\gamma_t^2 = [(1-\beta h_t)^2 - (1-h_t)^2] \sigma_{t-1}^2 = [(1-\beta h_t)^2 - (1-h_t)^2]\,\frac{\|f(y_{t-1})\|^2}{N}.
$$
*핵심*: 디노이저 잔차의 norm이 effective noise std의 추정량 → step size를 *adaptive*하게 자동 조정.
- **Step size schedule**: 고정 $h_t = h_0$는 Zeno's paradox식 exponential decay → 너무 느림. 저자들은 $h_t = h_0 t / (1 + h_0(t-1))$로 점진적으로 1에 접근시킴 (intuition: 거리가 줄어들수록 더 큰 fraction step OK).
- **Algorithm 1**: 위 식들을 묶음. parameters $\sigma_0$ (initial noise std), $\sigma_L$ (terminal threshold), $h_0$ (initial fractional step), $\beta$ (injected-noise factor; $\beta=1$이면 noise injection 없음).
- $\beta=1$일 때: deterministic ascent — same init → same output. $\beta < 1$이면 stochastic — 같은 init에서도 다른 결과 생성 (Fig. 2).
- **Image synthesis 결과 (Fig. 1, 2)**: BF-CNN을 BSD grayscale/color/MNIST에 학습한 후 random Gaussian $y_0$에서 sampling. 40 iterations 이내에 자연스러운 영상 (sharp contours, junctions, texture)이 *hallucinated*. $\beta=0.5$에선 적당한 다양성, $\beta=0.1$에선 더 부드럽지만 더 다양.

#### English
- **Iteration (Eq. 5)**: $y_t = y_{t-1} + h_t f(y_{t-1}) + \gamma_t z_t$.
- **Effective noise (Eq. 6)**: $\sigma_t^2 = (1-h_t)^2 \sigma_{t-1}^2 + \gamma_t^2$.
- **Convergence schedule (Eqs. 7–8)**: enforce $\sigma_t^2 = (1-\beta h_t)^2 \sigma_{t-1}^2$ ⇒ $\gamma_t^2 = [(1-\beta h_t)^2 - (1-h_t)^2]\|f(y_{t-1})\|^2/N$. Step size adapts via the *denoiser's own residual norm*.
- Algorithm 1 packages these into a coarse-to-fine ascent: start with $\sigma_0 = 1$ (broad search), terminate at $\sigma_L = 0.01$ (manifold convergence), $h_0 = 0.01$, $\beta \in \{0.1, 0.5, 1\}$.
- Convergence visible in Fig. 11 (appendix) typically within 40 iterations.

---

### Part IV: §3 Solving Linear Inverse Problems / 선형 역문제 풀이

#### 한국어
- 일반 형태: $x^c = M^T x$, $M \in \mathbb R^{N\times m}$ — 측정 행렬 (orthonormal columns, WLOG via SVD reparametrization).
- $MM^T \in \mathbb R^{N\times N}$: 측정 부분공간으로의 projection. $I - MM^T$: orthogonal complement.
- **§3.1 Constrained sampling (Bayes 분해)**:
$$
p(y\mid x^c) = p(y^c\mid x^c)\,p(y^u\mid y^c, x^c) = p(y^u\mid x^c)\,p(y^c\mid x^c)
$$
where $y^c = M^T y$ (projection onto measurement subspace), $y^u = \tilde M^T y$ (projection onto orthogonal complement, $\tilde M$ basis of $\mathrm{ker}(M^T)$).
- $\sigma^2 \nabla_y \log p(y\mid x^c) = \sigma^2 \nabla_y \log p(y^u\mid x^c) + \sigma^2 \nabla_y \log p(y^c\mid x^c)$.
- 첫 항: orthogonal complement에 정의된 함수의 그래디언트 — full gradient에서 measurement subspace를 *projected out* → $(I - MM^T) f(y)$.
- 둘째 항: 측정 분포가 Gaussian (variance $\sigma^2$)인 경우 → $M(y^c - x^c)$ — *measurement subspace로 끌어당김*.
- **결합 (Eq. 9)**:
$$
\boxed{\;\sigma^2 \nabla_y \log p(y\mid x^c) = (I - MM^T) f(y) + M(x^c - M^T y)\;}
$$
- 첫 항: prior gradient (manifold towards), measurement에 직교한 성분만 사용.
- 둘째 항: data-fidelity gradient, measurement subspace로 projection.
- **Algorithm 2**: Algorithm 1과 동일 구조이되 $f(y_{t-1})$를 $(I - MM^T) f(y_{t-1}) + M(x^c - M^T y_{t-1})$로 대체. Init: $y_0 \sim \mathcal N(0.5(I - MM^T)e + M x^c, \sigma_0^2 I)$ — 측정 영역은 측정값으로, 나머지는 mid-gray.

#### English
- Constraint $x^c = M^T x$ partitions the gradient (Eq. 9) into two orthogonal pieces: prior pull on the orthogonal complement of $M$, measurement pull on the measurement subspace. The *same* denoiser supplies the prior pull. Only the measurement matrix $M$ and observed values $x^c$ change between problems.
- This is the closest pre-2022 paper to modern conditional diffusion sampling — same algorithmic skeleton (coarse-to-fine, Langevin step, projection-based conditioning) without the DDPM machinery.

---

### Part V: §3.2 Linear inverse examples (results) / 선형 역문제 응용 결과

#### 한국어
- **Inpainting** (Fig. 3, 4): 30×30 missing block (BF-CNN의 receptive field 40×40 미만). 다양한 init → 다양한 plausible 복원 (manifold ∩ measurement subspace의 다른 점). MNIST 숫자도 자연스럽게 채워짐 — 다른 init이 7 vs 9 등 *different digits*을 생성할 수 있음을 시사.
- **Random missing pixels** (Fig. 5): 90% pixel 누락에서도 양호한 복원 (10% retained).
- **Spatial super-resolution** (Fig. 6, Tables 1–2): 4×4 block-averaging downsampling에서 4×, 8× 복원.
  - 4×, Set5: $MM^Tx$ baseline 26.35 dB / DIP 30.04 dB / DeepRED 30.22 dB / **Ours 29.47 dB** (single sample) / **Ours-avg 31.20 dB** (10 sample 평균).
  - 8×, Set5: baseline 23.02 dB / DIP 24.98 dB / DeepRED 24.95 dB / Ours 25.07 dB / Ours-avg 25.64 dB.
  - "Ours-avg"이 더 흐릿하지만 PSNR 더 높음 — 여러 hallucination을 평균하면 manifold *밖*으로 나가지만 (convex combination of manifold points), L2 측정에서는 더 우월. 저자들 직접 언급: 측정 행렬 rank가 낮을 때 PSNR/SSIM은 의미 제한적.
- **Run time (Table 3)**: 4× SR 평균 — DIP 1190 s, DeepRED 1584 s, **Ours 9 s** → ~150× 빠름. 단 한 forward pass per iteration vs DIP/DeepRED의 full backprop.
- **Spectral super-resolution (deblurring)**: $M$ = preserved low-frequency Fourier columns. $MM^Tx$는 sinc 커널 blurred. nontrivial — Gaussian deblurring과 달리 closed-form 없음. 결과는 Fig. 7.
- **Compressive sensing** (Fig. 8): $M$ = random Gaussian. Sparse-prior 베이스라인 (e.g., L1 minimization) 보다 훨씬 나은 결과 — 자연 영상 사전이 임의 sparse-union prior보다 강력함을 직접 증명.

#### English
- Same algorithm + same denoiser, only $M$ changes. Demonstrated on inpainting (30×30 block, 90 % random missing), 4×/8× super-resolution (Set5, Set14), spectral deblurring, compressive sensing.
- Quantitative wins (Tables 1–2): "Ours-avg" 31.20 dB vs DeepRED 30.22 dB on 4× Set5; 25.64 dB vs 24.95 dB on 8×. Run-time 9 s vs DIP's 1190 s — two orders of magnitude faster.
- Compressive-sensing experiments outperform sparse-union-of-subspace priors typical in compressive-sensing literature, demonstrating that an implicitly learned natural-image prior is genuinely richer than hand-designed sparse priors.

---

## 3. Key Takeaways / 핵심 시사점

1. **Miyasawa-Tweedie identity is the bridge — Miyasawa-Tweedie 항등식이 다리이다.**
   - **English**: A denoiser trained with $L_2$ loss provides $\nabla_y \log p_\sigma(y)$ directly via $f(y) = (\hat x(y) - y)/\sigma^2 \cdot \sigma^2$. No score-matching gymnastics needed. This single identity is the conceptual heart of all subsequent diffusion-style restoration.
   - **한국어**: $L_2$로 학습된 디노이저는 자동으로 score를 제공한다. score matching처럼 손실 형태를 분석할 필요가 없다 — 디노이저 잔차 자체가 score의 정의다.

2. **Bias-free architecture enables noise-level transfer — Bias-free 구조가 노이즈 레벨 전이를 가능하게 한다.**
   - **English**: BF-CNN (Mohan et al. 2020) removes additive bias terms, making the denoiser *homogeneous* in input scale. A single network trained over a noise range works at any $\sigma$, which is essential for the coarse-to-fine schedule.
   - **한국어**: 모든 bias 항을 제거하면 디노이저가 입력 스케일에 *homogeneous*하게 되어 한 네트워크가 모든 $\sigma$에서 동작. coarse-to-fine schedule의 결정적 enabler.

3. **Coarse-to-fine via Gaussian scale-space — 가우시안 스케일-공간을 통한 coarse-to-fine.**
   - **English**: Different $\sigma$'s correspond to different temperatures in the prior's Gaussian scale-space. Decreasing $\sigma$ over iterations is analogous to annealing — large $\sigma$ provides global, smooth gradient; small $\sigma$ adds high-frequency detail.
   - **한국어**: 서로 다른 $\sigma$는 사전의 가우시안 스케일-공간의 다른 온도. $\sigma$를 줄이는 것은 simulated annealing과 유사 — 큰 $\sigma$는 global·smooth, 작은 $\sigma$는 detail 추가.

4. **Step sizes adapt via residual norm — Step size가 잔차 norm으로 자동 조정된다.**
   - **English**: $\gamma_t^2 \propto \|f(y_{t-1})\|^2$ (Eq. 8). The denoiser tells you both the direction *and the magnitude* of the move — no manual schedule tuning. This is a free benefit of using $L_2$-optimal denoisers.
   - **한국어**: $\gamma_t$가 $\|f\|^2$에 비례 — 디노이저가 방향과 *크기*를 모두 알려준다. step schedule을 손으로 튜닝할 필요 없음.

5. **Constrained sampling = projected gradient — 제약 샘플링은 projected gradient로 풀린다.**
   - **English**: Eq. 9 splits the conditional gradient into orthogonal-complement prior pull plus measurement-subspace data pull. The same denoiser handles every linear inverse problem; only $M$ changes. This is the cleanest "denoiser-as-universal-prior" framework prior to DDRM.
   - **한국어**: 조건부 그래디언트가 직교 보충공간의 prior pull과 측정 부분공간의 data pull로 분해된다. 어떤 선형 역문제든 동일 디노이저로 해결.

6. **Two orders of magnitude faster than DIP/DeepRED — DIP·DeepRED 대비 100배 이상 빠름.**
   - **English**: 9 s vs 1,190 s for 4× super-resolution (Table 3). DIP/DeepRED retrain per-image; this method does single forward passes through a pretrained network.
   - **한국어**: 매 영상마다 재학습하는 DIP/DeepRED와 달리 사전학습 네트워크를 forward만 → ~150× 가속.

7. **Multi-sample averaging trades sharpness for PSNR — 다중 샘플 평균은 sharpness를 PSNR과 교환한다.**
   - **English**: Stochastic sampling produces multiple plausible reconstructions. Averaging them is a convex combination — leaves the manifold (so blurrier) but improves L2 PSNR. Discussed honestly by the authors as a *limit* of PSNR/SSIM at low rank.
   - **한국어**: 확률적 샘플링은 여러 plausible 복원을 만듦. 평균은 convex combination → manifold *밖*으로 나가지만 L2 PSNR은 향상. PSNR/SSIM의 측정값이 measurement rank가 낮을 때 의미 제한적임을 정직하게 인정.

8. **Bridge to diffusion-based restoration — 확산 기반 복원으로의 다리.**
   - **English**: Predates DDRM (paper #26) and DPS by ~1 year. The same algorithm skeleton — Langevin steps with projected gradient — is the backbone of all later diffusion inverse-problem solvers. The crucial simplification here is that *one denoiser does all the work*.
   - **한국어**: DDRM (논문 #26)과 DPS보다 1년 앞섬. Langevin step + projected gradient의 동일 골격이 이후 모든 diffusion-기반 역문제 풀이의 기반. 본 논문의 결정적 단순화: *하나의 디노이저로 모두 해결*.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Miyasawa-Tweedie identity / Miyasawa-Tweedie 항등식
$$
\hat x(y) = \mathbb E[x\mid y] = y + \sigma^2 \nabla_y \log p_\sigma(y). \tag{Eq. 3}
$$
Equivalently in terms of the denoiser residual $f(y) = \hat x(y) - y$:
$$
f(y) = \sigma^2 \nabla_y \log p_\sigma(y).
$$
- $\hat x(y)$: posterior mean (MMSE).
- $p_\sigma(y) = \int g(y - x) p(x) dx$: noisy observation density (heat-equation evolution of $p$).
- $\nabla_y \log p_\sigma$: score field.

### 4.2 Proof of Miyasawa (Eq. 4)
$$
\nabla_y p_\sigma(y) = \int \nabla_y g(y - x) p(x) dx = \int \frac{x - y}{\sigma^2} g(y-x) p(x) dx = \frac{1}{\sigma^2}\int (x - y) p(y, x) dx.
$$
Multiply by $\sigma^2 / p_\sigma(y)$:
$$
\sigma^2 \frac{\nabla_y p_\sigma(y)}{p_\sigma(y)} = \int x\,p(x\mid y) dx - y \int p(x\mid y) dx = \hat x(y) - y. \quad\square
$$

### 4.3 Stochastic ascent on prior (Eq. 5)
$$
y_t = y_{t-1} + h_t f(y_{t-1}) + \gamma_t z_t,\qquad z_t \sim \mathcal N(0, I).
$$
- $h_t \in [0, 1]$: deterministic step fraction (1 = full denoiser correction).
- $\gamma_t \ge 0$: injected-noise std.
- $f(y_{t-1})$: denoiser residual = score scaled by $\sigma^2$.

### 4.4 Effective noise variance (Eq. 6)
$$
\sigma_t^2 = (1 - h_t)^2 \sigma_{t-1}^2 + \gamma_t^2.
$$
First term: residual noise after the deterministic correction. Second term: freshly injected noise.

### 4.5 Convergence-enforcing schedule (Eqs. 7, 8)
Enforce $\sigma_t^2 = (1 - \beta h_t)^2 \sigma_{t-1}^2$ for $\beta \in [0,1]$:
$$
\gamma_t^2 = [(1 - \beta h_t)^2 - (1 - h_t)^2]\,\sigma_{t-1}^2 = [(1 - \beta h_t)^2 - (1 - h_t)^2]\,\frac{\|f(y_{t-1})\|^2}{N}.
$$
The right form uses the empirical estimate $\sigma_{t-1}^2 \approx \|f(y_{t-1})\|^2/N$ — the residual norm self-estimates the noise std.

### 4.6 Constrained-sampling gradient (Eq. 9)
$$
\sigma^2 \nabla_y \log p(y\mid x^c) = (I - MM^T) f(y) + M(x^c - M^T y).
$$
- $(I - MM^T) f(y)$: prior gradient projected to the orthogonal complement of measurement subspace.
- $M(x^c - M^T y)$: measurement-consistency pull, brings $M^T y$ towards $x^c$.

### 4.7 Step-size schedule
$$
h_t = \frac{h_0\,t}{1 + h_0(t - 1)}.
$$
Starts at $h_0$, asymptotes to 1 as $t \to \infty$. Avoids exponential-decay (Zeno's-paradox) slowdown.

### 4.8 Worked numerical example / 수치 예시
**1-D toy**: $p(x) = 0.5\,\mathcal N(-2, 0.1) + 0.5\,\mathcal N(+2, 0.1)$ (bimodal "manifold"). Noise $\sigma = 1$.
- Compute $p_\sigma(y) = 0.5\,\mathcal N(-2, 1.1) + 0.5\,\mathcal N(+2, 1.1)$.
- For $y = 1$:
  - $p_\sigma(1) \approx 0.5\,\phi((1+2)/\sqrt{1.1}) + 0.5\,\phi((1-2)/\sqrt{1.1})$ where $\phi$ is std normal pdf.
  - Score $\nabla_y \log p_\sigma(1) \approx +0.91$ (pulls toward right mode).
  - Optimal denoiser $\hat x(1) = 1 + 1^2 \cdot 0.91 = 1.91$ — close to right-mode mean $+2$.
- Initialise stochastic ascent at $y_0 \sim \mathcal N(0, 1)$. After ~15 iterations the trajectory snaps to one of the two modes (which one depends on injected noise).
- The noise injection $\gamma_t z_t$ with $\beta = 0.5$ allows mode hopping early; with $\beta = 1$ the trajectory is deterministic and converges to the closer mode.

### 4.9 Algorithm 1 (synthesis) and Algorithm 2 (inverse problems) pseudocode
**Algorithm 1 (synthesis)** — initialise $y_0 \sim \mathcal N(0.5, \sigma_0^2 I)$; for $t = 1, 2, \ldots$ until $\sigma_{t-1} \le \sigma_L$:
1. $h_t \leftarrow h_0 t / (1 + h_0(t-1))$.
2. $d_t \leftarrow f(y_{t-1})$.
3. $\sigma_t^2 \leftarrow \|d_t\|^2 / N$.
4. $\gamma_t^2 \leftarrow [(1 - \beta h_t)^2 - (1 - h_t)^2]\,\sigma_t^2$.
5. Draw $z_t \sim \mathcal N(0, I)$; $y_t \leftarrow y_{t-1} + h_t d_t + \gamma_t z_t$.

**Algorithm 2 (inverse problems)** — same loop but $d_t = (I - MM^T) f(y_{t-1}) + M(x^c - M^T y_{t-1})$; init $y_0 \sim \mathcal N(0.5(I - MM^T)e + M x^c, \sigma_0^2 I)$.

### 4.10 Run-time complexity / 실행시간 복잡도
- Per-iteration cost: 1 forward pass through BF-CNN + element-wise ops on $y$. Total ~40 iterations → ~9 s on DGX GPU for 256×256.
- DIP/DeepRED: per-image network optimization — 1190–1584 s.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1956      Robbins — Empirical Bayes (general framework)
1961      Miyasawa — denoiser-score identity for Gaussian noise
2009      Vincent — denoising autoencoder ↔ score matching equivalence
2011      Vincent (formal) — DAE as score estimator
2013      Venkatakrishnan+ — Plug-and-Play Priors (denoiser as proximal op)
2015      Sohl-Dickstein+ — non-equilibrium thermodynamics generative model (proto-diffusion)
2017      Romano+Elad+Milanfar — Regularization by Denoising (RED)
2019      Song+Ermon — Generative modeling by score matching (NCSN, annealed Langevin)
2020      Mohan+ — Bias-free CNN denoisers (BF-CNN)
2020      Ho+ — DDPM (denoising diffusion probabilistic models)
2021 ★★   KADKHODAIE-SIMONCELLI — THIS PAPER
                    ↳ direct Miyasawa-Tweedie based stochastic sampler
                    ↳ unifies image synthesis & linear inverse problems
                    ↳ adaptive step-size schedule via residual norm
2022      Kawar+ — DDRM (paper #26) — SVD-decomposed DDPM-based linear-inverse solver
2022      Chung+ — DPS (Diffusion Posterior Sampling) — generalized version with conditional gradients
2022      Kawar+ — DDPM-IR — supervised diffusion for restoration
2022      Bansal+ — Cold Diffusion (paper #27) — replaces noise with arbitrary corruption
2023      Saharia+ — Imagen (text-to-image diffusion at scale)
```

이 논문은 *score-based diffusion 시대로의 결정적 다리*. NCSN/DDPM의 복잡한 학습 손실 (denoising score matching, ELBO) 없이도 *그냥 $L_2$ 디노이저로 score를 얻을 수 있음*을 명시적으로 보였다 — 그 결과 후속 inverse-problem 논문들이 사용하는 conditional sampling의 가장 단순한 청사진을 제공.

This paper is the **decisive bridge to score-based diffusion**. Without invoking the elaborate training objectives of NCSN/DDPM (denoising score matching, ELBO), it explicitly shows that *a vanilla $L_2$-trained denoiser already provides the score*. The paper provides the simplest blueprint for conditional sampling that all later inverse-problem works (DDRM, DPS, ΠGDM, RED-Diff) refine.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Miyasawa (1961)** | Source identity | The whole paper hinges on this 60-year-old result — denoiser = $y + \sigma^2 \nabla \log p_\sigma$. |
| **Robbins (1956)** Empirical Bayes | Conceptual ancestor | General framework for using observed samples to estimate priors implicitly; Miyasawa's Gaussian case is one instance. |
| **Vincent (2011)** Score matching ↔ DAE | Alternative derivation | Derives the same residual-as-score relation from the *minimization condition* of denoising loss; this paper uses the more direct Miyasawa identity. |
| **Venkatakrishnan+ (2013)** Plug-and-Play | Prior algorithmic family | PnP heuristically inserts a denoiser as ADMM proximal operator; this paper provides the principled justification. |
| **Romano+Elad+Milanfar (2017)** RED | Algorithmic competitor | RED constructs a regularizer via denoiser (compared in Tables 1–2); this paper outperforms while being faster. |
| **Song+Ermon (2019)** NCSN | Closest predecessor | Annealed Langevin sampling with score networks — same algorithmic skeleton, different score derivation (denoising score matching loss). |
| **Mohan+ (2020)** Bias-free CNN | Architectural enabler | BF-CNN allows the same network to work at all noise levels; without this, the coarse-to-fine schedule would require multiple networks. |
| **Ho+ (2020)** DDPM | Concurrent independent work | DDPM uses the same Tweedie connection in its parameterization $\epsilon_\theta$; this paper makes the connection explicit and uses it for inverse problems. |
| **Ulyanov+ (2018)** Deep Image Prior | Compared baseline | DIP per-image optimization is 2 orders of magnitude slower; this paper uses pre-trained denoiser instead. |
| **Mataev+ (2019)** DeepRED | Compared baseline | Combines RED + DIP; outperformed by this paper in run-time and slightly in PSNR. |
| **Kawar+ (2022)** DDRM (paper #26) | Direct successor | DDRM extends this idea to leverage pre-trained DDPM models with SVD-based subspace decomposition; same conditional-sampling spirit. |
| **Chung+ (2022)** DPS | Sibling extension | DPS handles non-linear inverse problems via approximate posterior sampling on a pretrained diffusion; generalizes Algorithm 2 to non-linear $H$. |
| **Bansal+ (2023)** Cold Diffusion (paper #27) | Counterexample | Demonstrates that even *deterministic* (non-Gaussian) corruption can drive diffusion-style restoration — questioning whether the Miyasawa-noise framework is the only path. |
| **Lehtinen+ (2018)** Noise2Noise (paper #16) | Training methodology | Provides a way to train the underlying denoiser without clean data; combined with this paper, gives a fully self-supervised inverse-problem solver. |

---

## 7. Failure Modes and Limits / 실패 양상과 한계

### 한국어
1. **Bias from non-Gaussian noise**: Miyasawa 항등식은 *Gaussian* additive noise에만 정확. 다른 노이즈 (Poisson, impulse)에서는 $f(y)$가 score를 정확히 복원하지 않음 → 일반화는 후속 연구의 주제.
2. **PSNR이 낮은 rank measurement에서 misleading**: 측정 행렬 rank가 매우 낮을 때 (e.g., 90% missing) 여러 plausible 복원이 존재하므로 ground truth와의 PSNR이 perceptual quality와 잘 맞지 않음. 저자도 "Ours-avg가 더 흐릿하지만 PSNR 더 높음" 직접 언급.
3. **Receptive-field 제한**: BF-CNN의 RF (40×40)보다 큰 missing block은 정직한 통계로 채울 수 없음. 30×30 block은 OK, 50×50는 hallucination이 부자연스러워질 수 있음.
4. **Coarse-to-fine schedule tuning**: $\sigma_0$, $\sigma_L$, $h_0$, $\beta$의 4가지 hyperparameter가 sample 다양성·수렴속도·품질을 trade-off — 각 문제별로 약간의 튜닝 필요.
5. **Linear inverse problems에 한정**: 비선형 measurement (e.g., phase retrieval)에는 직접 적용 불가 — Eq. 9의 직교분해가 깨짐. DPS (Chung+ 2022)이 이 한계를 보완.

### English
1. **Gaussian-noise specificity**: the Miyasawa identity is exact only for additive Gaussian noise; Poisson/impulse cases require generalisations (later works extend to broader noise families).
2. **PSNR-perception mismatch at low measurement rank**: with 90 % missing pixels, many plausible reconstructions exist, so PSNR is a poor surrogate for perceptual quality. The authors honestly acknowledge that averaging samples improves PSNR but blurs the result.
3. **Receptive-field limit**: the BF-CNN's RF (~40×40) caps the missing-block size that can be filled with statistically meaningful texture; bigger holes would need a wider-context denoiser or hierarchical model.
4. **Hyper-parameter tuning**: $\sigma_0, \sigma_L, h_0, \beta$ trade off diversity, convergence speed, and quality — small per-problem tuning required.
5. **Linear-only**: the orthogonal decomposition of Eq. 9 fails for non-linear measurement operators; DPS and similar methods generalise this to non-linear forward models.

---

## 8. References / 참고문헌

- Kadkhodaie, Z., & Simoncelli, E. P. "Stochastic Solutions for Linear Inverse Problems Using the Prior Implicit in a Denoiser", *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2021. [arXiv:2007.13640]
- Miyasawa, K. "An empirical Bayes estimator of the mean of a normal population", *Bulletin of the International Statistical Institute*, 38 (1961).
- Robbins, H. "An empirical Bayes approach to statistics", *Proc. 3rd Berkeley Symposium*, 1956.
- Vincent, P. "A connection between score matching and denoising autoencoders", *Neural Computation*, 23(7), 1661–1674 (2011).
- Mohan, S., Kadkhodaie, Z., Simoncelli, E. P., & Fernandez-Granda, C. "Robust and interpretable blind image denoising via bias-free convolutional neural networks" (BF-CNN), *Proc. ICLR*, 2020.
- Venkatakrishnan, S. V., Bouman, C. A., & Wohlberg, B. "Plug-and-play priors for model based reconstruction", *Proc. GlobalSIP*, 2013.
- Romano, Y., Elad, M., & Milanfar, P. "The little engine that could: Regularization by denoising (RED)", *SIAM J. Imaging Sciences*, 10(4), 1804–1844 (2017).
- Song, Y., & Ermon, S. "Generative modeling by estimating gradients of the data distribution" (NCSN), *Proc. NeurIPS*, 2019.
- Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models" (DDPM), *Proc. NeurIPS*, 2020.
- Ulyanov, D., Vedaldi, A., & Lempitsky, V. "Deep Image Prior" (DIP), *Proc. CVPR*, 2018.
- Mataev, G., Milanfar, P., & Elad, M. "DeepRED: Deep image prior powered by RED", *Proc. ICCV Workshops*, 2019.
- Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. "Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising" (DnCNN), *IEEE TIP*, 26(7), 3142–3155 (2017).
- Kawar, B., Elad, M., Ermon, S., & Song, J. "Denoising Diffusion Restoration Models" (DDRM), *Proc. NeurIPS*, 2022.
- Chung, H., Kim, J., Mccann, M. T., Klasky, M. L., & Ye, J. C. "Diffusion Posterior Sampling for general noisy inverse problems" (DPS), *Proc. ICLR*, 2023.
