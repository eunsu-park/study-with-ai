---
title: "Noise2Self: Blind Denoising by Self-Supervision"
authors: Joshua Batson, Loïc Royer
year: 2019
journal: "Proc. 36th ICML, PMLR 97, pp. 524–533"
doi: "arxiv:1901.11365"
topic: Low-SNR Imaging / Self-Supervised Deep Denoising
tags: [noise2self, j-invariance, self-supervision, blind-denoising, donut-median, single-image, conditional-independence, batson-royer]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 18. Noise2Self: Blind Denoising by Self-Supervision / 자기지도 블라인드 잡음 제거

---

## 1. Core Contribution / 핵심 기여

### 한국어
Noise2Self (N2S)는 **잡음 모델·신호 사전·클린 GT 없이** 단일 잡음 측정으로부터 denoiser를 학습할 수 있는 일반 프레임워크다. 핵심 가정은 단 하나 — 측정 차원 $\{1,\dots,m\}$을 분할 $\mathcal J = \{J_1, J_2, \dots\}$로 쪼갰을 때 **잡음이 $J$ 사이에서 조건부 독립**이라는 것. 본 논문의 결정적 기여는 네 가지다:

(i) **J-invariance 정의**: 함수 $f: \mathbb R^m \to \mathbb R^m$이 임의 $J \in \mathcal J$에 대해 출력 $f(x)_J$가 입력 $x_J$에 의존하지 않으면 J-invariant라 한다 — 즉 $f$는 자기 자신을 보지 않고 자기 자신을 예측한다.

(ii) **자기지도 손실 정리 (Proposition 1)**: $x$가 unbiased 측정($\mathbb E[x|y] = y$)이고 $J$와 $J^c$ 위 잡음이 $y$에 조건부 독립이라면, J-invariant $f$에 대해
$$
\mathbb E\|f(x) - x\|^2 = \mathbb E\|f(x) - y\|^2 + \mathbb E\|x - y\|^2
$$
즉 self-supervised loss(왼쪽) = supervised loss(첫째 항) + 잡음 분산(둘째 항). 잡음 분산은 $f$와 무관한 상수 → **self-supervised loss 최소화 = true MSE 최소화**.

(iii) **고전 denoiser의 J-invariant 변형으로 calibration**: 모든 파라미터 매개 denoiser $g_\theta$(median, wavelet thresholding, NLM 등)에 대해 *donut-style J-invariant 변형* $f_\theta$를 정의 (Eq. 3) — 마스크 위치는 이웃 평균으로 채운 후 $g_\theta$를 적용. 그러면 $\theta$의 self-supervised loss 곡선의 최소점이 ground-truth loss 최소점과 일치 (Fig. 2).

(iv) **딥러닝 적용**: DnCNN/UNet에 4×4 그리드 마스킹을 적용해 J-invariant version을 만들고 self-supervised loss로 학습 → Hànzì, ImageNet, CellNet (sCMOS noise) 세 데이터셋에서 NLM·BM3D를 능가하고 Noise2Noise/Noise2Truth에 근접 (Table 2).

### English
Noise2Self (N2S) provides a **general framework for blind denoising** that requires no noise model, no signal prior, and no clean targets. Its single assumption is that the measurement dimensions $\{1,\dots,m\}$ admit a partition $\mathcal J = \{J_1, J_2, \dots\}$ such that **noise is conditionally independent across $J$ given the signal**. Four contributions:

(i) **J-invariant function class**: $f$ is J-invariant if $f(x)_J$ does not depend on $x_J$ for any $J \in \mathcal J$ — the function predicts each block from its complement.

(ii) **Self-supervised theorem (Proposition 1)**: For unbiased $x$ and J-invariant $f$, $\mathbb E\|f(x)-x\|^2 = \mathbb E\|f(x)-y\|^2 + \mathbb E\|x-y\|^2$. The second term is constant in $f$, so minimising self-supervised loss is equivalent to minimising the true (unobservable) MSE.

(iii) **Calibration of classical denoisers**: Any parametric denoiser $g_\theta$ can be J-invariantised by replacing pixels in $J$ with the average of their neighbours before applying $g_\theta$ (Eq. 3, "donut" filters). The optimum of the self-supervised loss aligns with the optimum of ground-truth loss — calibration without ground truth (Fig. 2).

(iv) **Deep learning instantiation**: DnCNN / UNet trained with 4×4 grid masking achieve PSNR competitive with Noise2Noise and Noise2Truth on Hànzì, ImageNet, and CellNet (Table 2), and beat classical NLM / BM3D on all three.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction & Motivation / 도입과 동기

#### 한국어
- **Goal** — 고차원 측정 $x \in \mathbb R^m$ (latent signal $y$, $x = y + n$)에서 $y$ 추정. 클린 GT, 잡음 모델, 신호 prior 어느 것도 없는 가장 가혹한 setting.
- **기존 가정 분류**:
  - **Smoothness** (Buades 2005): NLM·Gaussian filter — 부드러움 정도가 hyperparameter (필터 폭).
  - **Self-similarity** (BM3D): 비국소 패치 평균.
  - **Sparsity** (BayesShrink): 변환 영역 임계.
  - **Compressibility**: lossy compression as denoiser.
  - **Generative prior** (DIP).
  - **Gaussianity** (SURE).
- **Noise2Noise (Lehtinen 2018)**: 두 독립 잡음 측정 쌍 $(x_1, x_2)$로 학습 — clean GT 불필요. **N2S의 직접적 모티브**.
- **N2V (Krull 2018, concurrent)**: blind-spot CNN — 입력 픽셀이 자신을 보지 못하게 함. N2S와 *철학적으로 동일* 하나 N2S는 더 일반적인 partition 구조와 이론.

#### English
- Estimate $y$ from $x = y + n$ without clean targets, noise model, or signal prior.
- Prior approaches assume smoothness, self-similarity, sparsity, compressibility, generative prior, or Gaussianity.
- Noise2Noise relaxes the clean-target requirement to noisy-noisy pairs; N2S goes further to single-image blind denoising. Concurrent N2V uses a blind-spot CNN — N2S generalises the principle via J-invariance.

---

### Part II: §2 Related Work / 관련 연구

#### 한국어
- **Smoothness**: Gaussian/median filters — width as hyperparameter, often picked by visual inspection.
- **Self-similarity**: NLM (Buades 2005), BM3D (Dabov 2007). 큰 영향, 그러나 dataset/noise별 hyperparameter 의존.
- **Sparsity**: JPEG, wavelet shrinkage (Chang 2000), dictionary learning (Elad-Aharon 2006).
- **Compressibility**: lossy → 역압축이 implicit denoiser.
- **Autoencoders / UNet**: bottleneck width 선택의 자유. UNet은 skip connection 때문에 noisy data로 직접 학습하면 identity로 수렴.
- **Generative**: GAN-projection denoising (Tripathi 2018).
- **Gaussianity**: SURE (Stein) 기반 Stein-trained CNN (Metzler 2018, Zhussip 2018) — Gaussian iid 잡음 가정 필요.
- **Noise2Noise**: 잡음 쌍 만으로 학습 — $\mathbb E\|f(x_1) - x_2\|^2 = \mathbb E\|f(x_1) - y\|^2 + \mathrm{const}$. N2S는 *J-invariant 재구성*으로 single-image regime으로 확장.
- **Noise2Void (Krull 2018, concurrent)**: blind-spot CNN. **차이점**: N2V는 픽셀 위치를 *random pixel value*로 교체 → strict J-invariance 미달성. N2S는 partition 위에서 *이웃 평균 또는 마스킹*으로 strict J-invariant.

#### English
Surveys smoothness / self-similarity / sparsity / compressibility / autoencoder / generative / Gaussianity priors. Lehtinen et al.'s Noise2Noise reformulated in terms of J-invariance (composite signal $(y,y) \in \mathbb R^{2m}$, partition $\{1,\dots,m\}, \{m+1,\dots,2m\}$). Krull et al.'s Noise2Void replaces masked pixels with random nearby values — *not strictly J-invariant* (a given pixel may be replaced by itself). Vincent et al.'s "fully emphasised denoising autoencoders" used masked-input MSE loss but for representation learning, not denoising.

---

### Part III: §3 Calibrating Traditional Models / 전통 모델 calibration

#### 한국어
- **Setup**: parametric denoiser $g_\theta$ (예: median radius $\theta$, wavelet threshold $\theta$, NLM bandwidth $\theta$).
- **Optimal supervised** $\theta^* = \arg\min_\theta \|g_\theta(x) - y\|^2$ — GT 필요.
- **N2S calibration**: $g_\theta$를 J-invariant $f_\theta$로 변환 (Eq. 3):
$$
f_\theta(x)_J := g_\theta(\mathbf 1_J \cdot s(x) + \mathbf 1_{J^c} \cdot x)_J
$$
where $s(x)$ replaces each pixel by its neighbour average. 그 후
$$
\hat\theta = \arg\min_\theta \|f_\theta(x) - x\|^2
$$
를 GT 없이 계산 가능.
- **Donut median filter** ($\mathcal J = \{\{1\},\dots,\{m\}\}$): 각 픽셀을 *자기를 제외한* 반경 $r$ 디스크의 median으로 교체.
- **Fig. 2 핵심 그림**: 'cameraman' 이미지에 i.i.d. Gaussian noise → median filter (orange) vs donut median (blue), self-supervised loss vs ground-truth loss를 $r$에 대해 plot. **Donut의 self-sup loss와 GT loss는 곡선 모양·최적값 일치($r=3$)**. Classic median은 GT 곡선과 무관.
- **Vertical displacement = noise variance** (Eq. 2의 두 번째 항).

#### English
- For any classical denoiser $g_\theta$, define J-invariant variant $f_\theta$ by replacing pixels in $J$ with neighbour average before applying $g_\theta$. For partition into singletons, this is the "donut" trick.
- The self-supervised loss curve of the donut version traces the ground-truth loss curve up to a vertical offset equal to noise variance — optimal radius can be selected without GT.
- Demonstrated for donut median, donut-wavelet, donut-NLM (Supp. Fig. 1). Table 1 (cameraman, i.i.d. Gaussian): donut-NLM PSNR 30.4 dB (default 28.9), donut-wavelet 26.0 (default 24.6), donut-median 27.5 (default 27.1). Adding back noisy input (J-INVT+) recovers 0.4-0.9 dB additional improvement.

#### 한국어 — §3.1 Single-Cell
single-cell RNA-seq: $\sim 20\,000$개 유전자 mRNA count 측정. 매우 sparse, 잡음(under-sampling) 큼. 분자를 두 그룹 $J_1, J_2$로 random partition → $x_{J_1}$과 $x_{J_2}$는 **mRNA 분자 분포의 conditional independence**로 jointly Poisson sub-sample. **Principal Component Regression** (rank $k$)을 self-sup loss로 calibrate → 최적 $k=17$ (Fig. 3) — under-correction(stem cell marker invisible)과 over-correction(평균화) 사이 sweet spot.

#### English — §3.1 Single-Cell
Bone-marrow single-cell RNA-seq (Paul 2015, 2730 cells). Random molecule-level partition gives conditionally independent expression vectors. Self-supervised loss for principal-component regression rank-$k$ selects $k=17$; too few components misses stem-cell marker Ifitm1, too many over-smooths populations. Demonstrates the framework beyond image data.

#### 한국어 — §3.2 PCA cross-validation
Owen & Perry 2009 bi-cross-validation은 *feature 분할 + sample 분할* 양쪽으로 cross-validate. N2S는 이를 일반화한 special case.

#### English — §3.2 PCA
Bi-cross-validation (Owen-Perry) is a special case of N2S calibration where partition is over both features and samples.

---

### Part IV: §4 Theory / 이론

#### 한국어 — Proposition 1 (자기지도 손실 분해)
가정: $x_J$와 $x_{J^c}$가 $y$에 *조건부 독립* + $\mathbb E[x|y] = y$. $f$가 J-invariant. 그러면
$$
\mathbb E\|f(x) - x\|^2 = \mathbb E\|f(x) - y\|^2 + \mathbb E\|x - y\|^2 \quad (\text{Eq. 2})
$$
**증명 스케치**:
$$
\mathbb E\|f(x) - x\|^2 = \mathbb E\|f(x) - y\|^2 + \mathbb E\|y - x\|^2 - 2 \mathbb E\langle f(x) - y, x - y\rangle
$$
세 번째 항 $= -2\sum_J \mathbb E_y(\mathbb E_{x|y}[f(x)_J - y_J])(\mathbb E_{x|y}[x_J - y_J])$. $\mathbb E_{x|y}[x_J - y_J] = 0$ (unbiased) → 0. $f$의 J-invariance로 $f(x)_J$는 $x_{J^c}$만 의존하므로 $x_J$와 conditional independence가 cross-product 분해를 합법화.

#### English — Proposition 1 proof
Expand $\|f(x)-x\|^2 = \|f(x) - y\|^2 + \|y-x\|^2 - 2\langle f(x)-y, x-y\rangle$. Decompose the cross term over $J$; independence + unbiasedness kills it.

#### 한국어 — Proposition 2 (최적 J-invariant 예측기)
Self-supervised loss를 J-invariant 함수 클래스 위에서 최소화한 $f^*_{\mathcal J}$는
$$
f^*_{\mathcal J}(x)_J = \mathbb E[y_J | x_{J^c}]
$$
을 만족. **Bayesian conditional mean**.

#### English — Proposition 2
The unique minimiser is $f^*_{\mathcal J}(x)_J = \mathbb E[y_J | x_{J^c}]$ — the conditional expectation of the signal block given the *complementary* observation block.

#### 한국어 — §4.1 How good is the optimum?
$x_J$를 버리면 정보 손실. 측정 간 상관(correlation between features)이 클수록 $\mathbb E[y|x_{J^c}]$가 진짜 conditional mean $\mathbb E[y|x]$에 근접. **Gaussian Process** (Fig. 4)에서 length-scale $\ell$이 커질수록 J-invariant 최적 예측기가 진짜 최적과 가까워짐.

**Proposition 3**: 동일 covariance를 갖는 모든 신호 분포에 대해 Gaussian Process가 *worst case* — 즉 실제 자연 신호는 GP보다 항상 더 잘 복원됨 (Fig. 5: alphabet vs same-covariance GP).

#### English — §4.1
The "cost of giving up $x_J$" decreases with feature correlation. For Gaussian Processes, the J-invariant optimum gap closes as length scale grows. Proposition 3: GP is the worst case among signal distributions with given covariance — real-world structured data (e.g., MNIST alphabet) is denoised much better than its same-covariance GP (Fig. 5).

#### 한국어 — §4.2 Doing better
$f$가 J-invariant라 $f(x)_J$는 $x_J$의 정보를 *완전히 무시*. 최적 affine combination
$$
\lambda f(x)_J + (1-\lambda) x_J
$$
의 $\lambda$ = (noise variance)/(self-supervised loss). $f$가 PSNR을 10 dB 개선했다면 mix-back으로 +0.4 dB 추가 (Table 1 J-INVT+).

#### English — §4.2
Optimally mix denoiser output and noisy input by $\lambda = \text{noise var} / \text{self-sup loss}$. Yields ~0.4 dB extra at 10 dB SNR.

---

### Part V: §5 Deep Learning Denoisers / 딥러닝

#### 한국어
- **Architecture**: UNet, DnCNN (Zhang 2017, 560k parameters).
- **J-invariantisation**: 4×4 grid ($|\mathcal J| = 25$) — 매 minibatch마다 하나의 $J$에 대해 입력에서 마스크된 좌표를 *local average*로 채워넣고 ($s(x)_j$ for $j \in J$), 손실은 마스크된 좌표에서만 계산:
$$
\mathcal L = \mathbb E \sum_{j \in J} (g_\theta(\tilde x)_j - x_j)^2, \quad \tilde x = \mathbf 1_J \cdot s(x) + \mathbf 1_{J^c} \cdot x
$$
- **Datasets**:
  - **Hànzì**: 한자 문자, mixture of Poisson + Gaussian + Bernoulli noise.
  - **ImageNet** patches.
  - **CellNet**: fluorescence microscopy + simulated sCMOS camera noise (heteroscedastic).
- **Single-image extension**: DnCNN (560k params) trained on the **single 260k-pixel cameraman image** with self-sup loss → PSNR 31.2 dB. **Deep CNN denoising one image with no GT, no other images.**

#### English
- UNet & DnCNN trained with 4×4 grid J-invariantisation. Loss computed only on masked positions.
- Hànzì / ImageNet / CellNet (heteroscedastic sCMOS noise simulated).
- Single-image experiment on cameraman: DnCNN with 560k parameters reaches 31.2 dB on a 260k-pixel image, demonstrating per-image self-supervised training on a deep CNN.

---

### Part VI: §6 Discussion & Results Summary / 결과 정리

#### 한국어 — Table 2 (PSNR on held-out test data)

| Method     | Hànzì         | ImageNet | CellNet      |
|------------|---------------|----------|--------------|
| Raw        | 6.5           | 9.4      | 15.1         |
| NLM        | 8.4           | 15.7     | 29.0         |
| BM3D       | 11.8          | 17.8     | 31.4         |
| UNet (N2S) | 13.8 ± 0.3    | 18.6     | 32.8 ± 0.2   |
| DnCNN(N2S) | 13.4 ± 0.3    | 18.7     | 33.7 ± 0.2   |
| UNet (N2N) | 13.3 ± 0.5    | 17.8     | 34.4 ± 0.1   |
| DnCNN(N2N) | 13.6 ± 0.2    | 18.8     | 34.4 ± 0.1   |
| UNet (N2T) | 13.1 ± 0.7    | 21.1     | 34.5 ± 0.4   |
| DnCNN(N2T) | 13.9 ± 0.6    | 22.0     | 34.4 ± 0.4   |

핵심: N2S ≈ N2N ≈ N2T on Hànzì (구조적 신호, 강한 noise), CellNet에서 N2S가 N2N/N2T 대비 ~1.5 dB 부족하나 NLM/BM3D 대비 +1.4–4.7 dB 우수. ImageNet에서 N2T가 N2S보다 +2-3 dB — 자연 영상의 복잡성이 J-invariant 손실과 GT 손실의 gap을 키움.

#### English — Discussion
- N2S ≈ N2N on highly-structured signals (Hànzì) and microscopy with heteroscedastic noise (CellNet).
- ImageNet shows the largest N2T - N2S gap, reflecting that natural-image priors are far from J-invariant — increasing block correlation reduces this gap (§4.1).
- Open questions: optimal partition design, bias-variance tradeoff in $|J|/|J^c|$, and extension beyond i.i.d. noise.

---

## 3. Key Takeaways / 핵심 시사점

1. **J-invariance generalises blind-spot networks** — N2V의 blind-spot CNN을 partition-based 함수 클래스로 추상화 → 어떤 알고리즘(median, wavelet, NLM, CNN)에도 일관된 self-supervised calibration 제공. / J-invariance abstracts the blind-spot CNN into a partition-defined function class — providing self-supervised calibration uniformly across median, wavelet, NLM, and CNN denoisers.

2. **자기지도 손실 = MSE + 잡음 분산** — 단 두 가지 가정(unbiasedness + conditional independence) 아래 N2S 손실 최소화는 진짜 MSE 최소화와 동치. **잡음 모델 가정 불필요** — 가우시안, Poisson, Bernoulli 모두 작동. / Under unbiasedness and conditional independence, self-supervised loss differs from true MSE only by a constant noise variance term — minimisers coincide. **No noise-model assumption required**.

3. **Donut filter = parametric denoiser hyperparameter calibration** — median radius, wavelet threshold, NLM bandwidth 같은 단일-파라미터 알고리즘을 GT 없이 calibrate 가능. Fig. 2의 빨간 화살표(self-sup loss 최소)가 dashed line(GT loss 최소)과 정확히 같은 $r=3$ 위치. / The donut trick converts any parametric classical denoiser into a J-invariant variant whose self-supervised loss curve has the same minimiser as the ground-truth loss curve — calibration without ground truth.

4. **Optimum is conditional expectation $\mathbb E[y_J | x_{J^c}]$** — 진짜 Bayes posterior의 *complement-only* 변형. Block correlation이 클수록 진짜 $\mathbb E[y|x]$에 근접 (Prop. 3: GP는 worst case). / The optimal J-invariant predictor is $\mathbb E[y_J|x_{J^c}]$ — the Bayes conditional mean restricted to the complement. Approaches the unrestricted Bayes posterior as feature correlation grows.

5. **Single-image deep learning denoising** — 560k 파라미터 DnCNN을 *단일 cameraman 이미지* (260k 픽셀)에서 self-sup만으로 학습해 PSNR 31.2 dB 달성. DIP, Self2Self의 직접적 선행 사례. / Trains a 560k-parameter DnCNN on a single 260k-pixel image with only self-supervision — direct precursor to DIP-style and Self2Self single-image methods.

6. **Single-cell genomics 일반화** — partition을 픽셀이 아닌 *RNA molecule* 단위로 설정하면 PCA rank 선택에도 동일 정리 적용. Image 영역 밖에서도 작동하는 modality-agnostic 프레임워크. / By partitioning RNA molecules instead of pixels, the same theorem calibrates principal-component regression rank in single-cell RNA-seq — modality-agnostic.

7. **Noise2Noise를 J-invariance로 재해석** — composite signal $(y,y) \in \mathbb R^{2m}$에 partition $\{\{1,\dots,m\}, \{m+1,\dots,2m\}\}$ 적용 → N2N이 N2S의 special case가 된다. 통일적 시각 제공. / Recasts Noise2Noise as a J-invariance with the composite signal $(y,y)$ and the natural two-block partition — N2N becomes a special case.

8. **Mix-back regularisation ($\lambda x + (1-\lambda)f(x)$)** — J-invariant $f$는 $x_J$의 정보를 무시하므로 진짜 정보 손실이 있음. 최적 affine mixture $\lambda^* = \mathrm{Var}(n)/\mathcal L_{\text{self-sup}}$가 PSNR을 추가로 끌어올림 (Table 1: J-INVT+). / J-invariance discards $x_J$'s information; an optimal affine mixture with raw $x$ ($\lambda^* = \text{noise variance}/\text{self-sup loss}$) recovers part of that loss — adds 0.4 dB even when the denoiser already improves PSNR by 10 dB.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Definitions / 정의

**J-invariance**: $\mathcal J$ is a partition of $\{1,\dots,m\}$. Function $f: \mathbb R^m \to \mathbb R^m$ is *J-invariant* for $J \in \mathcal J$ if
$$
f(x)_J \text{ does not depend on } x_J
$$
and *$\mathcal J$-invariant* if J-invariant for every $J \in \mathcal J$.

**Self-supervised loss**:
$$
\mathcal L(f) = \mathbb E \|f(x) - x\|^2 \tag{1}
$$
### 4.2 Main theorem / 핵심 정리 (Proposition 1)

Assume $\mathbb E[x|y] = y$ and $x_J \perp x_{J^c} \mid y$ for each $J \in \mathcal J$. For J-invariant $f$:
$$
\boxed{\mathbb E\|f(x) - x\|^2 = \mathbb E\|f(x) - y\|^2 + \mathbb E\|x - y\|^2} \tag{2}
$$
**Term-by-term**:
- LHS: observable self-supervised loss.
- 1st term RHS: unobservable supervised loss (true MSE).
- 2nd term RHS: noise variance — constant in $f$.
→ **$\arg\min_f \mathcal L_{\text{self-sup}}(f) = \arg\min_f \mathcal L_{\text{sup}}(f)$**.

### 4.3 Optimal J-invariant predictor / 최적 J-invariant 예측기 (Proposition 2)
$$
f^*_{\mathcal J}(x)_J = \mathbb E[y_J | x_{J^c}]
$$
### 4.4 Donut J-invariantisation / 도넛 J-invariant 변형 (Eq. 3)

For classical denoiser $g_\theta$ and a "neighbour-average" function $s(x)$:
$$
f_\theta(x)_J := g_\theta\bigl(\mathbf 1_J \cdot s(x) + \mathbf 1_{J^c} \cdot x\bigr)_J \tag{3}
$$
### 4.5 Optimal mix-back / 최적 mix-back

$$
\hat y = \lambda f(x) + (1 - \lambda) x, \quad \lambda^* = \frac{\mathrm{Var}(n)}{\mathcal L_{\text{self-sup}}(f)}
$$
### 4.6 Worked numerical example / 수치 예시

i.i.d. Gaussian noise on cameraman image ($\sigma = 0.1$, so $\mathrm{Var}(n) = 0.01$). Donut median filter at radius $r=3$ yields self-supervised loss $\mathcal L_{\text{self-sup}} = 0.0107$ (Table 1).

By Eq. (2):
$$
\mathcal L_{\text{sup}} = \mathcal L_{\text{self-sup}} - \mathrm{Var}(n) = 0.0107 - 0.01 = 0.0007
$$
PSNR (peak = 1) $= 10\log_{10}(1/0.0007) \approx 31.5$ dB — close to reported 27.5 (J-INVT) plus the +0.7 dB mix-back to 28.2 (J-INVT+). Mix-back ratio:
$$
\lambda^* = 0.01/0.0107 \approx 0.935
$$
i.e., 93.5% denoiser output, 6.5% raw noisy image — still mostly the donut median, but a small amount of $x_J$ information is recovered.

---

## 5. Paper in the Arc of History / 역사 속의 논문

```
1948  Anscombe transform (#11) — Poisson VST
                  ↓
1994  Donoho-Johnstone wavelet shrinkage (#1) — universal threshold
                  ↓
2005  Buades NLM (#4) — patch self-similarity
                  ↓
2007  Dabov BM3D (#7) — collaborative filtering
                  ↓
2017  Ulyanov DIP — single-image deep prior, no GT
                  ↓
2018  Lehtinen Noise2Noise (#16) — clean GT unnecessary, noisy pairs suffice
                  ↓
2018-19 Krull Noise2Void (#17) — single-image, blind-spot CNN
                  ↓
2019  Batson Noise2Self (★) — J-invariance theory, calibrates ANY denoiser
                  ↓
2020  Quan Self2Self (#19) — Bernoulli sampling + dropout ensemble
                  ↓
2021  Huang Neighbor2Neighbor (#20) — sub-sample pairs, N2N loss + reg
                  ↓
2022  Wang Blind2Unblind (#22) — visible blind spots, current SOTA
```

**위치 / Position**: N2S sits at the *theoretical apex* of the 2018-19 self-supervised denoising explosion. While Noise2Noise needed two noisy realisations and Noise2Void needed a custom blind-spot architecture, N2S unifies the principle into the J-invariance framework, applicable to *any* denoiser (median, wavelet, NLM, DnCNN, UNet) and *any* modality (images, RNA-seq) where a conditional-independence partition exists.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문                     | Connection / 연결                                                                                        |
|----------------------------------|--------------------------------------------------------------------------------------------------------|
| #16 Lehtinen 2018 (Noise2Noise)  | N2S §2 reformulates N2N in terms of J-invariance with composite signal $(y,y)$ — N2N is special case. |
| #17 Krull 2019 (Noise2Void)      | Concurrent work; N2V's blind-spot CNN is *approximately* J-invariant. N2S provides theory N2V lacks.    |
| #19 Quan 2020 (Self2Self)        | Builds on N2S/N2V single-image idea; adds Bernoulli sampling + dropout ensembling for variance.        |
| #20 Huang 2021 (Neighbor2Neighbor) | Replaces masking by neighbour sub-sampling — N2N loss on sub-image pair + reg. Inherits J-invariance idea via spatial decorrelation. |
| #4  Buades 2005 (NLM)            | Calibrated by N2S framework via "donut NLM" — bandwidth chosen by self-sup loss (Supp. Fig. 1).        |
| #7  Dabov 2007 (BM3D)            | N2S Table 2 baseline; N2S deep CNN beats BM3D on Hànzì/CellNet, ties on ImageNet.                       |
| #1  Donoho 1994 (VisuShrink)     | Wavelet thresholding's threshold can be selected by donut-wavelet self-sup loss — Bayesian hyperparameter selection.|
| #25 Kadkhodaie 2021              | N2S optimum $\mathbb E[y_J|x_{J^c}]$ is a partial score — connects to score-based prior learning.    |

---

## 7. References / 참고문헌

- **Primary**: Batson, J. & Royer, L. "Noise2Self: Blind Denoising by Self-Supervision". Proc. 36th ICML, PMLR 97, 524–533, 2019. arXiv:1901.11365.
- **Code**: https://github.com/czbiohub/noise2self
- **Cited related**:
  - Lehtinen et al., "Noise2Noise: Learning image restoration without clean data", ICML 2018.
  - Krull, Buchholz, Jug, "Noise2Void", arXiv:1811.10980, CVPR 2019.
  - Ulyanov, Vedaldi, Lempitsky, "Deep image prior", arXiv:1711.10925, 2018.
  - Buades, Coll, Morel, "A non-local algorithm for image denoising", CVPR 2005.
  - Dabov, Foi, Katkovnik, Egiazarian, "Image denoising by sparse 3-D transform-domain collaborative filtering" (BM3D), IEEE TIP 16(8), 2007.
  - Zhang et al., "Beyond a Gaussian denoiser: Residual learning of deep CNN" (DnCNN), IEEE TIP 26(7), 2017.
  - Owen & Perry, "Bi-cross-validation of the SVD and the nonnegative matrix factorization", Annals of Applied Statistics 3(2), 2009.
  - Paul et al., "Transcriptional heterogeneity and lineage commitment in myeloid progenitors", Cell 163(7), 2015.
