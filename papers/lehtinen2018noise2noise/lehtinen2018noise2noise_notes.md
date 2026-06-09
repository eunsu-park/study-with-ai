---
title: "Noise2Noise: Learning Image Restoration without Clean Data"
authors: Jaakko Lehtinen, Jacob Munkberg, Jon Hasselgren, Samuli Laine, Tero Karras, Miika Aittala, Timo Aila
year: 2018
journal: "Proc. 35th International Conference on Machine Learning (ICML), PMLR 80"
doi: "arxiv:1803.04189"
topic: Low-SNR Imaging / Self-Supervised Deep Denoising
tags: [noise2noise, n2n, self-supervised, deep-learning, regression, mse-loss, conditional-expectation, gaussian-noise, poisson-noise, bernoulli-noise, monte-carlo-rendering, mri-reconstruction, lehtinen-aila]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 16. Noise2Noise: Learning Image Restoration without Clean Data / 깨끗한 데이터 없이 학습하는 영상 복원

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 영상 복원(denoising, MRI 재구성, Monte-Carlo 렌더링 디노이징 등)을 위한 회귀(regression) 신경망을 학습할 때 **깨끗한 ground-truth 영상이 전혀 필요 없다**는 놀라운 사실을 *기초 통계학*만으로 보였다.

핵심 통찰은 단 한 줄이다. 평균 제곱 오차(MSE) 손실로 학습되는 회귀기 $f_\theta$의 최적해는 입력에 *조건부* 출력 분포의 *기댓값*을 출력한다. 즉
$$
\theta^* = \arg\min_\theta \mathbb E_{(x,y)}\bigl[(f_\theta(x) - y)^2\bigr]
$$
의 해는 $f_{\theta^*}(x) = \mathbb E[y \mid x]$다. 따라서 깨끗한 타겟 $y$을 *어떤 zero-mean noise* 로 오염시킨 새로운 타겟 $\hat y$으로 바꿔도 $\mathbb E[\hat y \mid x] = \mathbb E[y \mid x]$이므로 *최적 파라미터가 변하지 않는다*. 유한 데이터에서는 추가 분산이 생기지만 학습 데이터가 충분하면 이 노이즈는 평균되어 사라진다.

저자들은 이 단순한 통찰로 (i) 가우시안, (ii) 푸아송 (촬영기 photon shot noise), (iii) 베르누이 (랜덤 마스킹·텍스트 오버레이), (iv) 임펄스 ($L_0$ 손실로 mode-seeking), (v) Monte-Carlo path-traced 렌더링 (해석 불가능한 비정상 분포), (vi) under-sampled MRI 재구성을 *깨끗한 영상 한 장도 보지 않고* 학습하는 데 성공했다. 결과 PSNR은 깨끗한 타겟으로 학습한 *오라클* 수준과 0.1 dB 이내에서 일치한다 (Table 1: BSD300 가우시안 $\sigma=25$에서 깨끗한 타겟 31.07 dB vs noisy 타겟 31.06 dB). 이 논문은 이후 모든 *self-supervised denoising*(Noise2Void, Noise2Self, Cryo-CARE, Self2Self 등) 의 출발점이 된다.

### English
The paper establishes — using only basic statistics — that a regression network for image restoration (denoising, MRI reconstruction, Monte-Carlo rendering, etc.) can be trained **without any clean reference images**. A network trained with MSE loss converges to the conditional expectation of its target. Replacing clean targets with *zero-mean-corrupted* targets does not change $\mathbb E[\hat y \mid x] = \mathbb E[y \mid x]$, so the optimum is unchanged. With sufficient data the extra variance averages out, and training on noisy/noisy pairs reaches *clean-target performance* — verified across Gaussian, Poisson, Bernoulli, salt-and-pepper / impulse noise, Monte-Carlo path-tracing artefacts, and undersampled MRI. Quantitatively, on BSD300 with $\sigma=25$ Gaussian noise the clean-target baseline reaches 31.07 dB and the noisy-target N2N reaches 31.06 dB — within statistical noise. The result lifts the long-standing requirement of clean ground truth for supervised image restoration and seeds an entire family of self-supervised denoisers (Noise2Void, Noise2Self, Cryo-CARE, Self2Self, Neighbor2Neighbor).

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction & §2 Theoretical Background / 서론과 이론적 배경

#### 한국어
- 표준 supervised denoising은 노이즈-정답 쌍 $(\hat x_i, y_i)$을 사용해 $\arg\min_\theta \sum_i L(f_\theta(\hat x_i), y_i)$ 을 푼다 (Eq. 1).
- 그러나 깨끗한 $y$ 확보가 비싸거나 불가능한 경우가 많다: 사진의 long-exposure, MRI의 full k-space sampling, MC 렌더링의 수만 spp.
- 저자들은 통계학의 *point estimation* 시각으로 출발한다: 측정 $\{y_1,y_2,\dots\}$에서 단일 추정값 $z$을 구하려면 $\arg\min_z \mathbb E_y[L(z,y)]$ (Eq. 2).
  - $L_2(z,y) = (z-y)^2 \Rightarrow z = \mathbb E_y[y]$ (산술 평균).
  - $L_1(z,y) = |z-y| \Rightarrow z = \mathrm{median}(y)$.
  - $L_0$ 비슷한 mode-seeking → $z = \mathrm{mode}(y)$.
- 신경망 회귀기 $f_\theta$도 동일한 구조의 일반화: $\arg\min_\theta \mathbb E_{(x,y)}[L(f_\theta(x), y)]$ (Eq. 4) → 조건부 분해
$$
\arg\min_\theta \mathbb E_x\!\left\{\mathbb E_{y\mid x}[L(f_\theta(x), y)]\right\} \tag{Eq.\,5}
$$
- **결정적 관찰** (이 논문의 모든 내용): 위 분해에서 *내부 기댓값*만 입력 $x$에 의존한다. 그러므로 타겟의 분포 $p(y\mid x)$을 *조건부 기댓값이 같은 다른 분포*로 바꾸어도 최적 $\theta$이 바뀌지 않는다. $L_2$ 손실의 경우 $\mathbb E[\hat y \mid x] = \mathbb E[y \mid x]$이면 충분.
- 따라서 학습은 다음으로 대체 가능:
$$
\arg\min_\theta \sum_i L\bigl(f_\theta(\hat x_i),\, \hat y_i\bigr) \tag{Eq.\,6}
$$
where both $\hat x_i$ *and* $\hat y_i$ are independent corrupted measurements of the same latent.

#### English
The classical training objective (Eq. 1) requires clean targets $y_i$. Statistically, training a regressor with MSE means converging to the conditional mean (Eqs. 2–5). The key insight is that **only the conditional expectation matters**: replacing clean targets with any zero-mean-corrupted version preserves $\mathbb E[\hat y\mid x] = y$, so the optimum is unchanged (Eq. 6). The variance contributed by noisy targets is finite-sample only — equal to (per appendix) "average target-noise variance / number of training samples."

---

### Part II: §3.1 Additive Gaussian Noise / 가산 가우시안 노이즈

#### 한국어
- 노이즈 $n \sim \mathcal N(0,\sigma^2)$, zero-mean → $L_2$ 적합. 
- 베이스라인: RED30 (Mao et al. 2016), 30층 hierarchical residual net, 128 feature maps. 학습은 ImageNet 50k에서 $256\times 256$ crop.
- $\sigma$를 학습 시 $[0,50]$ 범위에서 *랜덤*으로 골라 *blind denoising* 가능 (네트워크가 노이즈 강도까지 추정).
- 결과 (Table 1, BSD300/$\sigma=25$): 깨끗한 타겟 31.07 dB, noisy 타겟 31.06 dB, BM3D 30.34 dB. **Noisy 타겟으로 학습해도 깨끗한 타겟과 차이 ~0.01 dB**.
- 수렴 속도도 동일 (Fig. 1a). 더 빠른 U-Net으로도 결과 동일 ($-0.2$ dB).
- **왜 학습이 빠른가?** 활성 그래디언트는 noisy해도 가중치 그래디언트는 i.i.d. 픽셀 잡음의 평균 → $2^{16}$ 픽셀 위에서 평균되어 거의 동일. 직관적으로 "한 노이즈를 다른 노이즈로 매핑하는 것은 불가능"이지만 통계적 평균은 같음.
- Brown noise (Gaussian blur로 spatial correlation 도입, Fig. 1b): correlation이 커질수록 *수렴 속도*는 느려지지만 *최종 PSNR*은 유사 ($-0.1$ dB 이내). 그래디언트 평균화가 어려워지기 때문.
- **Capture-budget 분석** (Fig. 1c): 고정 측정 budget $N \cdot M = 2000$. 깨끗한 타겟 (1 noisy + 1 clean = average of 19 noisy)은 $N=100, M=20$ → 100 학습 페어. N2N은 같은 데이터로 $19\cdot 20 = 380$ 페어 (Case 2) → $0.1$dB 향상. $N=1000, M=2$면 (Case 3) 더 향상. **N2N은 데이터를 더 효율적으로 사용**.

#### English
On Gaussian noise, RED30 trained with noisy targets matches the clean-target baseline within 0.01 dB on BSD300 (Table 1) and within 0.1 dB on Kodak/Set14. Convergence speed is identical because per-pixel i.i.d. noise averages over the $2^{16}$ pixels of each crop. Spatial correlation (brown noise) slows convergence but not eventual PSNR. Capture-budget analysis (Fig. 1c) shows N2N is *more* sample-efficient than the traditional clean-target setup because it can form $M(M-1)$ noisy-noisy pairs per latent.

---

### Part III: §3.2 Other Synthetic Noises / 다른 합성 노이즈

#### 한국어
- **Poisson** ($\lambda\in[0,50]$): zero-mean이지만 signal-dependent. $L_2$ 손실 OK. 깨끗한 타겟 30.59 dB, noisy 타겟 30.57 dB. Anscombe + BM3D 베이스라인 28.36 dB → **N2N이 2 dB 우위**.
- **Bernoulli** (multiplicative mask $m$, $p\in[0.0,0.95]$): 손실은 mask된 픽셀만 기여:
$$
\arg\min_\theta \sum_i \bigl(m \odot (f_\theta(\hat x_i) - \hat y_i)\bigr)^2 \tag{Eq.\,7}
$$
zero-mean 만들기 위해 binomial 변환 적용. $p=0.5$에서 깨끗한 타겟 31.85 dB, noisy 타겟 32.02 dB → **N2N이 약간 우월** (dropout 효과로 추정). DIP 30.14 dB.
- **Random-valued impulse** (각 픽셀 $p$ 확률로 $[0,1]^3$ uniform 색으로 대체): mode-seeking 필요. 
  - $L_2$ → 평균 → 평균 텍스트 색으로 bias.
  - $L_1$ → median → 50% 미만에선 OK (텍스트 색이 random하면 원본이 majority).
  - $L_0$ annealed ($(|f-\hat y|+\epsilon)^\gamma$, $\gamma: 2 \to 0$) → mode → 90% 손상에서도 견딤 (Fig. 5).
- **Text-overlay** (Fig. 3): $L_2$는 평균 26.89 dB (gray bias), $L_1$은 35.75 dB (median, 깨끗 타겟의 35.82 dB와 동등).

#### English
The same N2N principle works across noise types when paired with the matching M-estimator loss: $L_2$ for zero-mean (Gaussian, Poisson), $L_1$ for median-seeking (text overlay, salt-and-pepper), annealed $L_0$ for mode-seeking (random-valued impulse). The chosen loss must match the noise's relevant summary statistic (mean / median / mode). Quantitative results (Table 1, Figs. 3–5) consistently match clean-target baselines.

---

### Part IV: §3.3 Monte-Carlo Rendering / 몬테 카를로 렌더링

#### 한국어
- MC path-tracing은 픽셀당 $N$ light-path를 샘플링; 렌더 평균이 진짜 픽셀 luminance. 잡음은 *zero-mean by construction* (importance sampling 빼고는).
- 64 spp 학습 페어 (서로 다른 random seed, noisy 입력 + noisy 타겟), 131k spp reference. 860개 architectural 영상.
- HDR 처리: tone mapping $T(v) = (v/(1+v))^{1/2.2}$ (Reinhard), 그러나 $\mathbb E[T(\hat y)] \ne T(\mathbb E[\hat y])$ → tone mapping은 *입력에만* 적용, 네트워크 출력은 linear-scale luminance, 손실은 relative-MSE
$$
L_{HDR} = (f_\theta(\hat x) - \hat y)^2 / (f_\theta(\hat x) + 0.01)^2 \tag{denominator gradient zeroed}
$$
- 64 spp validation PSNR 22.31 dB → 깨끗 타겟 학습 후 31.83 dB, noisy 타겟 31.83 dB (Tesla P100 12h 학습).
- 4000 epochs에서도 noisy 타겟이 $0.5$ dB 정도만 느림. **$8\times$P100 GPU + Xeon 40-core로 131k spp 한 장 렌더 40분 → noisy 타겟이 *훨씬* 경제적**.

#### English
For Monte-Carlo path tracing, the per-pixel noise distribution is heavy-tailed and analytically intractable, but it is zero-mean. With a relative-MSE HDR loss (Eq. given) and tone-mapped inputs, N2N matches the clean-target baseline of 31.83 dB and saves an order of magnitude of rendering time per latent (40 min per 131k-spp reference vs ~4 s per 64-spp pair).

---

### Part V: §3.4 MRI Reconstruction / MRI 재구성

#### 한국어
- Under-sampled k-space (Bernoulli mask $M_b$): 입력은 sub-Nyquist 샘플의 IFFT (artefacts). 두 *독립* 마스크로 sampling → noisy 입력/노이즈 타겟 페어.
- 손실은 일반화된 spectral loss (k-space에서 비교). 깨끗 ground-truth 절대 사용 안 함.
- 결과: 부분 측정만으로도 N2N이 supervised baseline과 비교 가능한 PSNR 달성. 임상에서 *짝 안 맞은 빠른 스캔만으로 학습 가능* → 임상 응용 가능.

#### English
For undersampled MRI, two independent k-space masks supply the noisy input and noisy target. Loss is computed in the k-space domain. The method matches supervised reconstruction, with the practical advantage that training requires only paired *fast* (undersampled) acquisitions — never a fully sampled reference, which is sometimes physically impossible.

---

### Part VI: Engineering details / 엔지니어링 세부사항

#### 한국어
- **Architecture**: §3.1은 RED30로 시작하지만 §3 이후의 모든 실험은 *얕은 U-Net* (Ronneberger 2015)으로 전환. RED30 대비 $\sim 10\times$ 빠르고 가우시안 노이즈에서 $-0.2$ dB만 떨어짐. 같은 결론이 모든 noise type에 적용.
- **Blind denoising**: $\sigma$ 또는 $\lambda$를 학습 시 *랜덤 범위*에서 뽑음 ($[0,50]$ for Gaussian, $[0,50]$ for Poisson). 추론 시 노이즈 강도 추정 불필요.
- **Crop size**: 256×256 ImageNet crops, batch size unspecified (논문) but 4–32 typical. ImageNet 50k validation images 사용.
- **Training time**: Tesla P100 GPU 1대로 RED30 training $\sim 14$h, U-Net $\sim 1.5$h, MC denoiser 12h.
- **Confidence intervals**: 5 random initialisations로 $\pm 0.02$ dB 수준 confidence interval — 즉 깨끗 타겟 31.07 dB vs noisy 타겟 31.06 dB는 *통계적으로 동등*.

#### English
- All §3 experiments after the Gaussian baseline switch from RED30 to a shallow U-Net (~10× faster, ~0.2 dB worse on Gaussian, conclusion preserved).
- Blind denoising is achieved by sampling $\sigma$ or $\lambda$ per crop from a wide range during training.
- Confidence intervals from 5 random seeds are ~±0.02 dB, so clean-target 31.07 dB vs N2N 31.06 dB are statistically indistinguishable.

---

### Part VII: Appendix-level intuition (variance contribution) / 부록 수준 직관 (분산 기여)

#### 한국어
유한 데이터에서 N2N의 추가 variance는 부록에서 다음과 같이 정리:
$$
\mathrm{Var}_{\theta^*}^{\mathrm{N2N}} - \mathrm{Var}_{\theta^*}^{\mathrm{clean}} \approx \frac{\sigma_n^2}{N_{\mathrm{train}}}
$$
$\sigma_n^2$은 타겟 잡음 분산, $N_{\mathrm{train}}$은 학습 영상 수. $N_{\mathrm{train}} \to \infty$ 일 때 0. 이는 *그래디언트 평균화*에 의한 수렴 보장의 정확한 형태.

직관적 검증: $\sigma_n=25$, $N_{\mathrm{train}} = 50000$ (ImageNet 일부) → 추가 variance $\sim 25^2/50000 \approx 0.0125$ — PSNR로 $\le 0.005$ dB 손실. 관측된 0.01 dB와 *정량적으로* 일치.

#### English
The appendix bounds N2N's excess variance by $\sigma_n^2 / N_{\mathrm{train}}$. For $\sigma_n=25$ and $N_{\mathrm{train}}=50\,000$ the predicted PSNR loss is $\le 0.005$ dB, matching the observed $\sim 0.01$ dB.

---

## 3. Key Takeaways / 핵심 시사점

1. **Conditional-expectation invariance is the entire trick / 조건부 기댓값 불변이 비결의 전부** — Loss = MSE → optimum = $\mathbb E[y\mid x]$. Replacing $y$ with zero-mean-corrupted $\hat y$ preserves the conditional mean, hence the optimum. 이론은 단 두 줄이고 모든 응용을 통합한다.

2. **Match the loss to the noise's statistic / 손실은 노이즈의 통계량에 맞춰야** — Zero-mean → $L_2$. Median-preserving (salt-pepper, text overlay) → $L_1$. Mode-preserving (random-valued impulse) → annealed $L_0$. 잘못된 손실은 mean-shifted bias (Fig. 3 $L_2$ 결과) 등의 가시적 오류를 일으킨다.

3. **Finite-sample variance scales as 1/N / 유한 샘플 분산은 1/N** — 깨끗한 타겟 대신 noisy 타겟을 쓰면 *최적의 위치*는 같지만 *학습 그래디언트*가 더 noisy. Appendix는 이 추가 분산이 (target-noise variance / training-set size)임을 증명. 따라서 충분히 큰 데이터셋에서는 사실상 *공짜*.

4. **Spatial correlation slows convergence, not the limit / 공간 상관은 수렴을 늦출 뿐 한계는 아니다** — Brown Gaussian noise에서 inter-pixel correlation이 wide할수록 그래디언트 평균화가 비효율적이지만 (Fig. 1b) 충분한 epoch 후 PSNR은 $<0.1$ dB 이내에서 일치.

5. **N2N is more sample-efficient than clean-target / N2N은 깨끗 타겟보다 표본 효율적** — 같은 capture budget에서 $M$ noisy realisations from $N$ latents → $M(M-1)$ noisy pairs (vs 1 clean per latent). 결과 PSNR이 더 높을 수 있음 (Fig. 1c, Cases 2–3).

6. **Monte Carlo & MRI artefacts: not Gaussian, but still zero-mean / MC·MRI 잡음은 가우시안이 아니지만 여전히 zero-mean** — N2N이 *해석적으로 모델링 불가능한* 잡음에도 적용된다는 결정적 시연. 이는 NLM/BM3D/Anscombe-BM3D 등 explicit-prior 방법으로는 불가능.

7. **Bernoulli targets give dropout-like benefit / Bernoulli 타겟은 dropout 같은 부수 효과** — $p=0.5$에서 noisy 타겟 PSNR이 *깨끗 타겟보다 0.17 dB 높음*. Target-side stochasticity가 implicit regularisation을 제공.

8. **Lifts a hard practical constraint / 실질적 제약을 제거** — 이 결과는 *생물학적 라이브 이미징, 저선량 cryo-EM, 임상 MRI* 등 깨끗한 ground truth 측정이 *물리적으로 불가능한* 분야에 deep denoising을 직접 가능하게 했다. 이후의 모든 self-supervised denoising은 N2N의 직접 자손이다.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Point estimation as the ancestor / 점 추정의 조상
$$
z^* = \arg\min_z \mathbb E_y[L(z,y)] \tag{Eq.\,2}
$$
$L=L_2 \Rightarrow z^* = \mathbb E[y]$; $L=L_1 \Rightarrow z^* = \mathrm{median}(y)$; $L = L_0 \Rightarrow z^* = \mathrm{mode}(y)$.

### 4.2 Standard supervised regression / 표준 지도학습 회귀
$$
\theta^* = \arg\min_\theta \mathbb E_{(x,y)}\bigl[L(f_\theta(x), y)\bigr] \tag{Eq.\,4}
$$
조건부 분해:
$$
= \arg\min_\theta \mathbb E_x\!\left\{\mathbb E_{y\mid x}[L(f_\theta(x),y)]\right\} \tag{Eq.\,5}
$$
$L_2$에서:
$$
f_{\theta^*}(x) = \mathbb E[y \mid x]
$$
### 4.3 The Noise2Noise principle / 노이즈투노이즈 원리
입력 $\hat x = s + n$, 타겟 $\hat y = s + n'$, $n \perp n'$, $\mathbb E[n] = \mathbb E[n'] = 0$:
$$
\boxed{\;\mathbb E[\hat y \mid \hat x] = \mathbb E[s \mid \hat x] + \mathbb E[n' \mid \hat x] = \mathbb E[s \mid \hat x]\;}
$$
즉 *깨끗한 타겟 $s$을 noisy 타겟 $\hat y$으로 대체해도 conditional mean이 바뀌지 않는다*. 결과:
$$
\theta^* = \arg\min_\theta \sum_i L\bigl(f_\theta(\hat x_i), \hat y_i\bigr) \tag{Eq.\,6}
$$
은 깨끗한 타겟으로 학습한 것과 동일한 최적해.

### 4.4 Bernoulli (masked) loss / 베르누이 (마스크) 손실
$$
\arg\min_\theta \sum_i \bigl(m \odot (f_\theta(\hat x_i) - \hat y_i)\bigr)^2 \tag{Eq.\,7}
$$
masked 픽셀만 기여, mask 자체도 *binomial*해 zero-mean 보장.

### 4.5 HDR Monte-Carlo loss / HDR 몬테 카를로 손실
$$
L_{HDR} = \frac{(f_\theta(\hat x) - \hat y)^2}{(f_\theta(\hat x) + 0.01)^2}
$$
분모의 그래디언트는 0으로 처리; tone mapping $T(v) = (v/(1+v))^{1/2.2}$는 *입력에만* 적용.

### 4.6 Worked numerical example: Monte-Carlo verification / 수치 예시: 몬테 카를로 검증
Latent 픽셀 $s = 100$. 노이즈 $n,n' \sim \mathcal N(0,\sigma^2)$, $\sigma=25$. $N = 10^4$ 샘플.

* Sample mean of clean-target loss $(f - s)^2$ for $f=100$: 0.
* Sample mean of N2N loss $(f - \hat y)^2$ for $f=100$: $\approx \sigma^2 = 625$.
* Sample mean for $f=110$: clean = $100$, N2N = $100 + \sigma^2 = 725$.
* **Difference of losses ($f=110$ vs $f=100$): clean = 100, N2N = 100** — 두 손실의 *gradient*가 같음을 확인.

$N$ 샘플 평균에서 추가 분산은 $\mathrm{Var}/N \to 0$ → 충분한 데이터에서는 동등한 학습.

### 4.7 Algebraic derivation of the N2N invariance / N2N 불변성의 대수적 유도
조건부 분해 (Eq. 5)에서 $L = L_2$:
$$
\mathbb E_{y\mid x}[(f_\theta(x) - y)^2]
= (f_\theta(x))^2 - 2 f_\theta(x)\,\mathbb E[y\mid x] + \mathbb E[y^2\mid x]
$$
$\theta$에 대한 미분:
$$
\frac{\partial}{\partial \theta} \mathbb E_{y\mid x}[(f_\theta(x) - y)^2]
= 2\bigl(f_\theta(x) - \mathbb E[y\mid x]\bigr)\frac{\partial f_\theta(x)}{\partial \theta}
$$
즉 그래디언트는 *오직 $\mathbb E[y\mid x]$에만 의존*. 따라서 $y$을 $\hat y$로 대체하더라도 $\mathbb E[\hat y\mid x] = \mathbb E[y\mid x]$이면 그래디언트가 동일 → 학습 동역학 동일.

$L_1$의 경우 절댓값 미분이 sign 함수 → median-preserving 분포 변경에 불변. $L_0$의 경우 mode-preserving에 불변.

### 4.8 Capture-budget table / 수집 예산 표
$N$ latents × $M$ noisy/latent = $NM = 2000$:

| Case | $N$ | $M$ | Setup | Pairs / Latent | PSNR (Fig. 1c) |
|------|-------|-------|-------|----------------|----------------|
| 1 (traditional) | 100 | 20 | 1 noisy + 1 clean (avg of 19) | 1 | baseline |
| 2 (N2N, same data) | 100 | 20 | all noisy/noisy combos | $M(M-1) = 380$ | +0.1–0.3 dB |
| 3 (N2N, more latents) | 1000 | 2 | 1 noisy/noisy pair | 1 | even higher |

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1950s-90s  Classical M-estimators (Huber 1964) — mean/median/mode
1994-95   Donoho-Johnstone — wavelet thresholding (paper #1, #2)
2005      Buades-Coll-Morel — NL-means (paper #4) — patch self-similarity
2007      Dabov+ — BM3D (paper #7) — block-matching + collaborative filtering
2009      Burger+ — first "MLP for denoising" (NIPS)
2017      Zhang+ — DnCNN — supervised deep CNN denoising (clean targets needed)
2017      Mao+ — RED30 — deep encoder-decoder for restoration
2017      Weigert+ — CARE for fluorescence microscopy (clean targets needed)
2018 ★★   LEHTINEN+ — NOISE2NOISE (THIS PAPER)
                            ↳ removes the clean-target requirement entirely
                            ↳ unifies Gaussian/Poisson/Bernoulli/MC/MRI denoising
2019      Buchholz+ — Cryo-CARE (paper #15) — N2N applied to cryo-EM tomography
2019      Krull+ — NOISE2VOID (paper #17) — single-image self-supervised
2019      Batson-Royer — NOISE2SELF (paper #18) — J-invariance framework
2020      Quan+ — SELF2SELF (paper #19) — Bernoulli dropout self-training
2021      Huang+ — NEIGHBOR2NEIGHBOR (paper #20) — random sub-sampling N2N
```

이 논문은 *self-supervised denoising 시대의 시작점*. 모든 이후 연구는 N2N의 가정 (independent noise pair)을 *어떻게 완화하느냐*의 변주이다.

This paper is the **foundation of the self-supervised denoising era**. Every subsequent work (N2V, N2S, S2S, Neighbor2Neighbor, Cryo-CARE) relaxes one of N2N's assumptions — the requirement of *paired* independent noisy realisations — to operate from progressively less data.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Huber (1964)** "Robust estimation" | M-estimators framework | The L1/L2/L0 loss-statistic mapping in §2 is classical Huber theory; N2N just lifts it from point estimation to neural regressors. |
| **Mao+ (2016)** RED30 | Network architecture used | RED30 is the supervised baseline; N2N replaces its clean targets with noisy ones at zero PSNR loss. |
| **Weigert+ (2017)** CARE | Supervised microscopy denoising | CARE requires clean low-dose pairs; N2N (and its successor Cryo-CARE, paper #15) frees CARE from that requirement. |
| **Zhang+ (2017)** DnCNN | Residual CNN denoiser | The "discriminative deep denoising" paradigm N2N replaces; both use $L_2$ regression but DnCNN needs clean targets. |
| **Ulyanov+ (2018)** Deep Image Prior (DIP) | Training-free baseline | DIP also avoids clean targets but exploits CNN structure as prior; N2N is 2 dB better on Bernoulli noise (Table 1). |
| **Mäkitalo-Foi (2013)** (paper #14) | Anscombe + BM3D | Explicit prior for Poisson; N2N beats it by 2 dB on $\lambda=30$ without analytical noise model. |
| **Buchholz+ (2019)** Cryo-CARE (paper #15) | Direct application | Applies N2N to cryo-EM tomography by splitting even/odd projections — no algorithmic change. |
| **Krull+ (2019)** Noise2Void (paper #17) | Removes paired requirement | N2V drops the need for *paired* noisy images — a single noisy image suffices via blind-spot networks. |
| **Batson-Royer (2019)** Noise2Self (paper #18) | Theoretical generalisation | Generalises "paired noise" to "$\mathcal J$-invariant function class"; N2N is one instance. |
| **Quan+ (2020)** Self2Self (paper #19) | Bernoulli-dropout extension | Combines N2N's principle with Bernoulli sub-sampling for single-image training. |
| **Huang+ (2021)** Neighbor2Neighbor (paper #20) | Sub-sampled pair construction | Constructs the two N2N-required noisy versions from spatial neighbours of a single image. |

---

### Failure modes and limits / 실패 양상과 한계

#### 한국어
N2N이 *항상* 동작하는 것은 아니다:
1. **Saturation / gamut clipping**: ADC clip 또는 detector saturation은 noise를 *truncated*하게 만들어 zero-mean을 위배 → 결과 bias. 논문도 Poisson 실험에서 saturation 지적.
2. **Strongly correlated noise within target**: target내 픽셀 간 잡음 상관이 매우 강하면 (e.g. 같은 detector row의 read-out noise) gradient averaging이 효과적이지 않아 수렴 매우 느림.
3. **Heavy-tailed losses with wrong matching**: Cauchy-distributed noise를 $L_2$로 학습하면 outlier가 mean을 끌어당김 → $L_1$ 또는 robust loss 필요.
4. **Small datasets**: 추가 분산 $\sigma_n^2/N_{\mathrm{train}}$이 무시할 수 없을 때 (예: $N=10$) 깨끗 타겟 학습이 더 우월. 본 논문 capture-budget 분석은 이 trade-off의 정량.
5. **Distribution mismatch input vs target**: $\hat x$와 $\hat y$이 *다른 분포*에서 추출되면 $\mathbb E[\hat y\mid \hat x] \ne \mathbb E[s\mid \hat x]$ → 일반적으로 적용 불가 (단, conditional mean이 같으면 OK).

#### English
N2N can fail when (i) saturation breaks zero-mean, (ii) target-side noise is strongly within-image correlated, (iii) loss does not match the noise's relevant statistic, (iv) the dataset is too small to average target variance, or (v) input and target are drawn from different distributions whose conditional means differ. The paper's capture-budget study addresses (iv) directly.

### Conceptual progression / 개념적 진화
N2N 이전과 이후 영상 디노이저들의 *정보 가정* 진행:

| Paradigm | Required data | Year | Key paper |
|---------|--------------|------|-----------|
| Hand-crafted prior | None (single image) | 1990s–2000s | NL-means, BM3D, TV |
| Supervised CNN | Many $(x, y_{\mathrm{clean}})$ pairs | 2012– | DnCNN, RED30 |
| Noise2Noise | Many $(x_{\mathrm{noisy}}, y_{\mathrm{noisy}})$ pairs | 2018 | **THIS PAPER** |
| Cryo-CARE | N2N applied to cryo-EM (paired by even/odd splits) | 2019 | Buchholz+ |
| Noise2Void | Single noisy image | 2019 | Krull+ (paper #17) |
| Noise2Self | Single noisy image ($\mathcal J$-invariance) | 2019 | Batson+Royer |
| Self2Self | Single noisy image (Bernoulli dropout) | 2020 | Quan+ |
| Neighbor2Neighbor | Single noisy image (sub-sampled pairs) | 2021 | Huang+ |

이 표는 *정보 요구의 단조 감소*를 보임 — N2N이 그 감소의 결정적 분기점.

The progression shows monotonically decreasing data requirements, with N2N marking the decisive break from clean-pair supervised methods.

---

## 7. References / 참고문헌

- Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., & Aila, T. "Noise2Noise: Learning Image Restoration without Clean Data", *Proc. ICML*, PMLR 80, 2018. [arXiv:1803.04189]
- Huber, P. J. "Robust estimation of a location parameter", *Annals of Math. Stat.*, 35(1), 73–101 (1964).
- Mao, X., Shen, C., & Yang, Y.-B. "Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections" (RED30), *Proc. NIPS*, 2016.
- Weigert, M., Schmidt, U., Boothe, T., et al. "Content-aware image restoration: pushing the limits of fluorescence microscopy" (CARE), *bioRxiv*, 2017.
- Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising" (DnCNN), *IEEE TIP*, 26(7), 3142–3155 (2017).
- Ulyanov, D., Vedaldi, A., & Lempitsky, V. "Deep Image Prior", *Proc. CVPR*, 2018.
- Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering" (BM3D), *IEEE TIP*, 16(8), 2080–2095 (2007).
- Mäkitalo, M., & Foi, A. "Optimal Inversion of the Generalized Anscombe Transformation for Poisson-Gaussian Noise", *IEEE TIP*, 22(1), 91–103 (2013).
- Ronneberger, O., Fischer, P., & Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation", *Proc. MICCAI*, 2015.
- Reinhard, E., Stark, M., Shirley, P., & Ferwerda, J. "Photographic tone reproduction for digital images", *ACM TOG*, 21(3), 267–276 (2002).
