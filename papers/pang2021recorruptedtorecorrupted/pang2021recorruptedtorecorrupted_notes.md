---
title: "Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising"
authors: Tongyao Pang, Huan Zheng, Yuhui Quan, Hui Ji
year: 2021
journal: "CVPR 2021, pp. 2043–2052"
doi: "10.1109/CVPR46437.2021.00208"
topic: Low-SNR Imaging / Self-Supervised Denoising
tags: [self-supervised, denoising, noise2noise, recorruption, single-image, gaussian-noise, poisson-noise, cvpr2021]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 21. Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising / 재손상-대-재손상: 영상 잡음 제거를 위한 비지도 심층학습

---

## 1. Core Contribution / 핵심 기여

### 한국어
이 논문은 **단 하나의 잡음 영상**만으로 Noise2Noise(N2N)와 동등한 학습을 가능하게 하는 **재손상(recorruption) 변환**을 제안한다. 핵심 아이디어는 noisy observation $y = x + n$($n$은 알려진 분포의 잡음)에 두 개의 결정적·가역 가능한 변환을 적용하여 두 개의 합성 잡음 영상 쌍을 만드는 것이다:
$$
y_1 = y + \alpha D^{T} z, \qquad y_2 = y - \alpha^{-1} D z, \qquad z \sim \mathcal N(0, I)
$$
여기서 $D$는 임의의 가역 행렬, $\alpha > 0$는 스칼라 하이퍼파라미터. 이 한 쌍은 (i) **$y_1$의 잡음과 $y_2$의 잡음이 무상관**(uncorrelated)이고, (ii) **$E[y_1 \mid x] = x$**(즉 평균이 깨끗한 신호와 같음)을 만족한다. 따라서 N2N 손실 $\|f_\theta(y_1) - y_2\|^2$를 최소화하는 것은 (보정 항을 더해주면) 깨끗한 영상 $x$를 회귀하는 것과 통계적으로 동등하다. R2R은 N2N의 *대규모 잡음/잡음 쌍* 수집 부담과 Noise2Void(N2V)의 *맹점(blind-spot) 제약*을 동시에 해소한다. CVPR 2021 시점에서 합성·실세계 raw-RGB 데이터 모두에서 자기지도 SOTA를 달성했다.

### English
The paper proposes a **recorruption transform** that lets a network be trained à la Noise2Noise but starting from a *single* noisy observation $y = x + n$. Two deterministic–stochastic transformations of $y$ produce a synthetic noisy/noisy training pair:
$$
y_1 = y + \alpha D^{T} z, \qquad y_2 = y - \alpha^{-1} D z, \qquad z \sim \mathcal N(0, I),
$$
where $D$ is any invertible matrix and $\alpha>0$ a scalar hyper-parameter. The pair is engineered so that (i) the noises in $y_1$ and $y_2$ are *uncorrelated* and (ii) $\mathbb E[y_1 \mid x] = x$. Consequently, minimising the N2N loss $\|f_\theta(y_1)-y_2\|^2$ becomes statistically equivalent to regressing onto the clean image $x$ (up to an additive correction). R2R thereby removes the two main practical drawbacks of earlier methods — N2N's need for paired noisy captures of the same scene and N2V's blind-spot constraint — and achieved self-supervised SOTA on both synthetic Gaussian/Poisson noise and real raw-RGB datasets at the time of CVPR 2021.

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Motivation / §1 동기

#### 한국어
- 깊은 디노이저(DnCNN, FFDNet, RIDNet 등)는 **noisy/clean 쌍**으로 학습하지만 실세계에서는 깨끗한 ground truth를 얻기가 매우 비싸거나 불가능 (의료영상, 천체영상, 형광현미경 등 저-SNR 도메인).
- 우회 경로 (a) **Noise2Noise (Lehtinen et al. 2018)**: 같은 장면의 두 독립 noisy 캡처가 있으면 clean 없이 학습 가능. 그러나 정적 장면 두 번 촬영이 필요.
- 우회 경로 (b) **Noise2Void / Noise2Self / Self2Self**: 단일 영상 학습 가능하지만 **blind-spot** 제약 때문에 중심 픽셀 정보 손실, 성능 한계.
- R2R의 목표: **단일 noisy 영상 + 잡음 분포 정보** → blind-spot 없이 N2N과 동등한 학습 가능 여부.

#### English
- Supervised denoisers (DnCNN, FFDNet, RIDNet, …) train on noisy/clean pairs, but in real low-SNR domains (medical, astronomical, fluorescence microscopy) clean ground truth is impossible or impractically expensive.
- Workaround (a) — **Noise2Noise**: two independent noisy captures of the same scene suffice. Requires static scene & two acquisitions.
- Workaround (b) — **Noise2Void / Noise2Self / Self2Self**: single-image training but the *blind-spot* constraint hides the centre pixel, capping performance.
- R2R's goal: starting from a *single* noisy image with known noise statistics, can we get N2N-equivalent training without any blind-spot mechanism?

### Part II: §3 Method – the recorruption transform / §3 방법 – 재손상 변환

#### 한국어 — 가우시안 잡음 케이스
잡음이 $n \sim \mathcal N(0, \Sigma)$이고 covariance $\Sigma$가 알려져 있다고 하자. 보조 가우시안 $z \sim \mathcal N(0, I)$를 sampling해서 다음 두 합성 영상을 만든다:
$$
y_1 = y + \alpha D^T z, \quad y_2 = y - \alpha^{-1} D z,
$$
$D$는 $D D^T = \Sigma$를 만족하면 자유 (예: $\Sigma = \sigma^2 I$이면 $D = \sigma I$).

**잡음 분해**:
- $y_1 = x + n + \alpha D^T z = x + n_1$ where $n_1 = n + \alpha D^T z$
- $y_2 = x + n - \alpha^{-1} D z = x + n_2$ where $n_2 = n - \alpha^{-1} D z$

**핵심 두 보조정리** (논문 Proposition 1):
1. $\mathbb E[n_1] = 0$, $\mathbb E[n_2] = 0$ — 두 합성 잡음 모두 평균 0.
2. $\mathbb E[n_1 n_2^T] = \mathbb E[n n^T] - \alpha^{-1} \mathbb E[n z^T] D^T + \alpha \mathbb E[D^T z n^T] - D^T \mathbb E[z z^T] D = \Sigma - 0 + 0 - D^T D$.
   $D D^T = \Sigma$이고 trace 차원에서 둘이 일치하면 $D^T D = \Sigma$이므로 $\mathbb E[n_1 n_2^T] = 0$ — 두 잡음은 *상관없음*.

따라서 $(y_1, y_2)$는 N2N의 두 독립 noisy 캡처와 통계적으로 *호환*되며 N2N 정리를 그대로 사용할 수 있다.

#### English — Gaussian case
Assume $n \sim \mathcal N(0,\Sigma)$ with known $\Sigma$. Draw an auxiliary $z \sim \mathcal N(0,I)$ and form
$$
y_1 = y + \alpha D^T z, \quad y_2 = y - \alpha^{-1} D z,
$$
with $D$ any matrix satisfying $D D^T = \Sigma$ (e.g. $D = \sigma I$ when $\Sigma = \sigma^2 I$).

**Two key properties** (Proposition 1):
1. $\mathbb E[n_1] = \mathbb E[n_2] = 0$.
2. $\mathbb E[n_1 n_2^T] = \Sigma - D^T D$, which vanishes when $D D^T = D^T D = \Sigma$ (so e.g. any symmetric square root of $\Sigma$ works).

The pair $(y_1,y_2)$ is therefore statistically interchangeable with the two independent noisy observations that N2N requires — and N2N's theorem (E-loss equivalent to oracle MSE) carries over verbatim.

#### 한국어 — Noise2Noise 정리 재진술
N2N이 보장하는 핵심 사실: 두 noisy 영상 $\hat y_1, \hat y_2$가 $\mathbb E[\hat y_i | x] = x$ 이고 $\mathbb E[(\hat y_1 - x)(\hat y_2-x)^T] = 0$을 만족하면,
$$
\arg\min_\theta \mathbb E\|f_\theta(\hat y_1) - \hat y_2\|^2 = \arg\min_\theta \mathbb E\|f_\theta(\hat y_1) - x\|^2.
$$
즉 *clean target 손실*과 *noisy target 손실*의 최소화가 같은 $\theta^*$로 수렴.

#### English — Noise2Noise restated
N2N's underpinning: if $(\hat y_1,\hat y_2)$ are two noisy versions of $x$ with **zero-mean residuals** and **uncorrelated residuals**, then minimising $\mathbb E\|f_\theta(\hat y_1) - \hat y_2\|^2$ has the same global minimiser as minimising the supervised MSE $\mathbb E\|f_\theta(\hat y_1) - x\|^2$. R2R inherits this directly from its Proposition 1.

### Part III: §3.3 General signal-dependent noise / §3.3 신호-의존 잡음

#### 한국어
가우시안만이 아니라 신호 의존 잡음(예: Poisson, heteroscedastic Gaussian, Poisson-Gaussian read-noise+shot-noise)도 다룬다. 신호 의존 분포의 경우 covariance가 신호의 함수: $\Sigma(x)$. 논문은 noisy 영상 $y$에서 noise covariance 추정치 $\hat\Sigma$를 얻은 뒤 동일한 변환을 적용. Poisson 잡음의 경우 Anscombe 변환을 사전 단계로 사용해 가우시안 근사로 환원.

알고리즘 1 (R2R training):
1. Noisy mini-batch $\{y_b\}$ sampling.
2. 각 $y_b$에 대해 auxiliary $z_b \sim \mathcal N(0,I)$ 샘플링.
3. $y_1^b = y_b + \alpha D^T z_b$, $y_2^b = y_b - \alpha^{-1} D z_b$.
4. Loss: $\sum_b \|f_\theta(y_1^b) - y_2^b\|^2$, gradient step.

#### English
Beyond i.i.d. Gaussian, R2R covers signal-dependent noise (Poisson, heteroscedastic Gaussian, Poisson–Gaussian read-noise + shot-noise) by estimating $\hat\Sigma$ from $y$ first, then applying the same transform. For Poisson, Anscombe variance-stabilising transform reduces to Gaussian. Algorithm 1 sketches the training loop: per mini-batch, sample $z$, build $(y_1, y_2)$, take a gradient step on $\|f_\theta(y_1) - y_2\|^2$.

### Part IV: §3.4 Inference / §3.4 추론

#### 한국어
추론 시에는 단일 noisy $y$를 그대로 네트워크에 넣지 않는다. 대신 R2R sampling을 $K$번 반복(논문 권장 $K \approx 50$)하여 평균:
$$
\hat x = \frac{1}{K} \sum_{k=1}^K f_\theta\!\left(y + \alpha D^T z^{(k)}\right).
$$
이 Monte-Carlo 평균은 (a) 네트워크가 학습한 입력 분포($y_1$ 형태)와 정확히 일치하는 입력을 사용하고, (b) 보조 잡음 분산을 평균을 통해 줄여준다. 추론 비용은 $K$배지만 V100에서도 합리적.

#### English
At inference time the network is *not* applied to bare $y$. Instead R2R draws $K$ auxiliary noise vectors and averages
$$
\hat x = \tfrac{1}{K}\sum_{k=1}^K f_\theta\!\big(y + \alpha D^T z^{(k)}\big), \quad K \approx 50.
$$
This (a) feeds the network inputs that match its training distribution and (b) Monte-Carlo-averages out the auxiliary corruption. The price is a $K\times$ inference cost, deemed acceptable on a V100.

### Part V: §4 Experiments / §4 실험

#### 한국어
- **합성 가우시안** ($\sigma=25,50$, 컬러/그레이): BSD68, Set12, Urban100. R2R은 BM3D, N2V, Self2Self, Laine19, Noisier2Noise, NBR2NBR을 능가하거나 동등 — 특히 N2C(supervised baseline) 대비 0.1~0.3 dB 차이.
- **합성 Poisson** ($\lambda=30$ 및 $\lambda \in [5,50]$): Anscombe + R2R로 강력. SOTA 자기지도 베이스라인 NBR2NBR과 경쟁.
- **실세계 raw-RGB**: SIDD benchmark/validation. R2R은 N2C 대비 약 -0.1 dB로 거의 동일, NBR2NBR 등 자기지도 방법보다 우수.
- **하이퍼파라미터 영향**: $\alpha \in [0.5, 2]$ 영역에서 안정. 너무 작으면 $y_1$이 거의 $y$와 같아 과적합, 너무 크면 보조 잡음이 진잡음을 압도.
- **네트워크 백본**: DnCNN-S(17 layer)와 U-Net 모두 평가. R2R은 백본 비의존적이며 두 경우 모두 일관된 향상.
- **추론 횟수 $K$**: 실험에서 $K=50$이 sweet spot. $K=10$에서도 유의미한 PSNR 향상. $K=100$ 이상은 marginal한 추가 이득.
- **학습 안정성**: R2R 학습은 N2N과 동일한 수렴 속도, 약 100k iteration이면 충분.

#### English
- **Synthetic Gaussian** ($\sigma=25, 50$, RGB & gray): on BSD68, Set12, Urban100, R2R matches or beats BM3D, N2V, Self2Self, Laine19, Noisier2Noise, NBR2NBR — and lands within 0.1–0.3 dB of the supervised N2C baseline.
- **Synthetic Poisson** ($\lambda=30$, $\lambda \in [5,50]$): Anscombe + R2R is competitive with the strongest self-supervised baselines.
- **Real raw-RGB SIDD**: roughly $-0.1$ dB versus the supervised N2C baseline, ahead of every other self-supervised method tested.
- **Hyper-parameter sensitivity**: $\alpha\in[0.5,2]$ is robust. Too small ⇒ $y_1\approx y$ and the network overfits; too large ⇒ auxiliary noise dominates and the network learns a different problem.
- **Backbone**: both DnCNN-S (17 layers) and U-Net evaluated. R2R is backbone-agnostic and both gain consistently.
- **MC sample count $K$**: $K=50$ is the sweet spot. $K=10$ already gives meaningful PSNR improvement; gains plateau beyond $K=100$.
- **Training stability**: R2R training converges at the same rate as N2N, ~100k iterations are sufficient.

### Part VI: §5 Discussion and Limitations / §5 논의와 한계

#### 한국어
- **장점**: (i) 단일 영상 학습. (ii) blind-spot 없음 → 정보 보존. (iii) 백본 자유. (iv) Anscombe로 Poisson 대응.
- **한계**: (i) 잡음 covariance $\Sigma$ 사전 지식 필요. (ii) 추론 비용이 $K$배. (iii) 비-가우시안 잡음 분포에서는 추가 검증 필요.
- **확장 가능성**: 비독립 잡음(공간적 상관 잡음)에 대해서는 $D D^T = \Sigma$만 만족하면 됨. correlated noise도 처리 가능 (CMOS sensor의 fixed-pattern noise 등).

#### English
- **Strengths**: (i) trains from a single image, (ii) no blind-spot constraint, (iii) backbone-agnostic, (iv) handles Poisson via Anscombe.
- **Limitations**: (i) requires known noise covariance $\Sigma$, (ii) inference is $K$× more expensive, (iii) non-Gaussian non-Poisson noise distributions need extra verification.
- **Extensibility**: spatially correlated noise (e.g. fixed-pattern noise of CMOS sensors) is supported as long as $D D^T = \Sigma$ — the recorruption identity is *not* limited to i.i.d. cases.

---

## 3. Key Takeaways / 핵심 시사점

1. **단일 noisy 영상으로 N2N 등가 학습 가능 / Single-noisy-image training is N2N-equivalent** — Proposition 1이 핵심: $D D^T = \Sigma$, $D^T D = \Sigma$만 만족하면 합성된 두 영상의 잡음이 무상관·평균 0. 따라서 N2N 정리가 그대로 적용된다.
   The single-image transform inherits N2N's full theoretical guarantee as long as the auxiliary corruption matches the noise covariance.

2. **Blind-spot이 없다 / No blind spot** — N2V/N2S와 달리 입력의 어떤 픽셀도 가리지 않는다. 따라서 receptive field 전체 정보를 사용 가능 → 더 높은 PSNR 상한.
   Unlike N2V/N2S/Self2Self, no input pixel is masked out, so the entire receptive field carries information; this raises the achievable PSNR ceiling.

3. **잡음 모델이 알려져 있어야 한다 / Requires a known noise model** — 이것이 N2V 대비 약점. 그러나 천체·현미경처럼 read-noise+shot-noise 모델이 캘리브레이션으로 알려진 도메인에서는 거의 무료.
   The pricier assumption is that $\Sigma$ is known. In astronomy, microscopy, and other calibrated detectors this assumption is essentially free.

4. **추론은 Monte-Carlo 평균 / Inference is a Monte-Carlo average** — $K=50$번 평균. 이는 학습 입력과 추론 입력의 분포 mismatch를 없애는 본질적 단계. 단일 forward는 noticeable PSNR drop 발생.
   Inference uses MC averaging because feeding bare $y$ violates the input distribution the network has learned. Skipping it leads to a measurable PSNR drop.

5. **$\alpha$는 SNR 트레이드오프 슬라이더 / $\alpha$ is an SNR trade-off slider** — $y_1$의 잡음 분산은 $(1+\alpha^2)\sigma^2$, $y_2$는 $(1+\alpha^{-2})\sigma^2$. 두 영상 SNR을 어떻게 분배하느냐를 통제. 비대칭 ($\alpha\ne 1$) 가능하므로 N2N의 *symmetric* 가정보다 일반적.
   $\alpha$ trades off SNR between the two synthetic images: $\mathrm{Var}(n_1)=(1+\alpha^2)\sigma^2$, $\mathrm{Var}(n_2)=(1+\alpha^{-2})\sigma^2$. The pair can be asymmetric, generalising the symmetric N2N pair.

6. **자기지도 + 학습 가능 prior / Self-supervised yet retains a learnable prior** — BM3D 같은 비학습 방법과 달리 어떤 backbone (DnCNN, U-Net, NAFNet …)이든 사용 가능. 따라서 학습 가능 prior의 강력함과 ground truth-free의 실용성을 동시에 가짐.
   Backbone-agnostic — it can wrap DnCNN, U-Net, or NAFNet — so it inherits the expressive power of learnable priors while staying ground-truth-free.

7. **Poisson은 Anscombe로 환원 / Poisson handled by Anscombe** — 신호 의존 잡음을 분산-안정화 변환을 통해 가우시안 근사로 만들면 $\Sigma$가 다시 신호 독립이 되어 R2R이 그대로 적용. Anscombe의 inverse는 추론 후 적용 (Makitalo & Foi 2010).
   For Poisson statistics, the Anscombe variance-stabilising transform reduces to a (near-)Gaussian regime so the same $D$ machinery applies; the inverse Anscombe is applied at inference (Makitalo & Foi 2010).

8. **R2R은 후속 자기지도 SOTA들의 베이스라인이 됨 / R2R becomes a self-supervised SOTA baseline** — Blind2Unblind (CVPR 2022, paper #22), NBR2NBR, AP-BSN 등 후속 논문이 R2R을 표준 비교 대상으로 삼는다. 실제 SIDD 벤치 차트에서 R2R은 거의 항상 상위 3위 안에 위치.
   R2R has since become a default baseline in subsequent self-supervised denoising papers (Blind2Unblind, NBR2NBR, AP-BSN), almost always landing in the top-3 of SIDD benchmark tables.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Setting / 설정
$$
y = x + n, \quad n \sim \mathcal N(0, \Sigma), \quad \Sigma \text{ known}.
$$
### 4.2 Recorruption transform / 재손상 변환
$$
y_1 = y + \alpha D^T z, \qquad y_2 = y - \alpha^{-1} D z, \qquad z \sim \mathcal N(0, I), \qquad D D^T = D^T D = \Sigma.
$$
Decompose $y_1 = x + n_1$, $y_2 = x + n_2$ with
$$
n_1 = n + \alpha D^T z, \quad n_2 = n - \alpha^{-1} D z.
$$
### 4.3 Proposition 1 (zero-mean and uncorrelated) / 명제 1
$$
\mathbb E[n_1 \mid x] = 0, \qquad \mathbb E[n_2 \mid x] = 0,
$$
$$
\mathbb E[n_1 n_2^T \mid x] = \Sigma - D^T D = 0.
$$
### 4.4 Variance bookkeeping / 분산 회계
$$
\mathrm{Cov}(n_1) = \Sigma + \alpha^2 D^T D = (1+\alpha^2)\Sigma,
$$
$$
\mathrm{Cov}(n_2) = \Sigma + \alpha^{-2} D D^T = (1+\alpha^{-2})\Sigma.
$$
설정 $\alpha=1$, $\Sigma=\sigma^2 I$이면 $y_1, y_2$ 둘 다 $\mathcal N(x, 2\sigma^2 I)$ — 분산 두 배.

### 4.5 N2N theorem inherited / 상속된 N2N 정리
$$
\arg\min_\theta \mathbb E\,\|f_\theta(y_1) - y_2\|^2 = \arg\min_\theta \mathbb E\,\|f_\theta(y_1) - x\|^2.
$$
### 4.6 Inference / 추론
$$
\hat x(y) = \frac{1}{K} \sum_{k=1}^K f_\theta\!\big(y + \alpha D^T z^{(k)}\big), \quad z^{(k)} \overset{iid}{\sim} \mathcal N(0, I).
$$
### 4.7 Worked numerical example / 수치 예시

$\sigma = 25/255 \approx 0.098$, $\alpha = 1$, single pixel $x = 0.6$.
- Draw $n \sim \mathcal N(0, \sigma^2)$, e.g. $n = 0.04$. $y = 0.64$.
- Draw $z \sim \mathcal N(0,1)$, e.g. $z = -1.2$. Then $\alpha D^T z = 1 \cdot 0.098 \cdot (-1.2) = -0.118$.
- $y_1 = 0.64 - 0.118 = 0.522$, $n_1 = -0.078$.
- $\alpha^{-1} D z = 1 \cdot 0.098 \cdot (-1.2) = -0.118$ → $y_2 = 0.64 - (-0.118) = 0.758$, $n_2 = 0.158$.
- Sanity check: $n_1 \cdot n_2 = -0.0123$ — single-sample product is *not* zero, but the population expectation is. Monte Carlo with $10^6$ samples gives $\frac{1}{N}\sum n_1 n_2 \to 0$ as required.
- SNR check: $\mathrm{Var}(n_1) = 2\sigma^2 = 0.0192$ → $\mathrm{std}(n_1) \approx 0.139$, matching $|n_1| \sim O(0.1)$ scale.

### 4.8 Practical $\alpha$ choices / 실용적 $\alpha$ 선택
| $\alpha$ | $\mathrm{Var}(n_1)/\sigma^2$ | $\mathrm{Var}(n_2)/\sigma^2$ | Comment |
|---|---|---|---|
| 0.5 | 1.25 | 5.00 | $y_1$ near $y$, $y_2$ very noisy → asymmetric, target is loud. |
| 1.0 | 2.00 | 2.00 | Symmetric, equal-SNR case (closest to N2N). |
| 1.5 | 3.25 | 1.44 | $y_1$ noisier, $y_2$ cleaner → input loud, target clean. |
| 2.0 | 5.00 | 1.25 | Aggressive auxiliary noise. |
Paper recommends symmetric $\alpha \approx 1$; ablations show $\pm 50\%$ tolerance.

### 4.9 Inference variance reduction / 추론 분산 감소
Monte-Carlo estimator with $K$ draws has variance scaled by $1/K$:
$$
\mathrm{Var}[\hat x(y)] \approx \frac{1}{K}\,\mathrm{Var}_z[f_\theta(y + \alpha D^T z)].
$$
$K=50$ gives $\sqrt{50}\approx 7\times$ noise reduction relative to a single pass.

### 4.10 Algorithm 1 (training loop) / 알고리즘 1 학습 루프
```
Input: noisy dataset {y_b}, known Sigma, alpha, total iterations T
Initialize network f_theta
For t = 1, ..., T:
  Sample mini-batch {y_b}
  For each y_b:
    Sample z_b ~ N(0, I)
    y_1 = y_b + alpha D^T z_b
    y_2 = y_b - alpha^{-1} D z_b
  Compute loss = sum_b || f_theta(y_1) - y_2 ||^2
  Update theta via Adam
Return f_theta
```
Single epoch ≈ standard N2N epoch. Mini-batch size 4–32 in paper; lr = 1e-4, Adam.

### 4.11 Inference Monte-Carlo / 추론 몬테카를로
```
Input: trained f_theta, noisy y, K=50
For k = 1, ..., K:
  Sample z^{(k)} ~ N(0, I)
  hat_x^{(k)} = f_theta(y + alpha D^T z^{(k)})
Return mean_k hat_x^{(k)}
```

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
2007 ─── Dabov+ — BM3D (transform-domain self-similarity, no learning)
2017 ─── Zhang+ — DnCNN (residual CNN, supervised denoising baseline)
2018 ★ Lehtinen+ — Noise2Noise: noisy/noisy training without clean targets
2018 ─── Krull+ — Noise2Void: single-image self-supervised, blind-spot
2019 ─── Batson-Royer — Noise2Self (J-invariant generalisation of N2V)
2019 ─── Laine+ — High-quality self-supervised via masked convolution
2020 ─── Quan+ — Self2Self with dropout (single-image)
2020 ─── Moran+ — Noisier2Noise (pre-corrupt input by extra noise)
2021 ★★ PANG-ZHENG-QUAN-JI: Recorrupted-to-Recorrupted (this paper)
                ↳ unifies single-image + N2N via covariance-matched recorruption
2021 ─── Wang+ — Neighbor2Neighbor (NBR2NBR, sub-sampling pair)
2022 ─── Wang+ — Blind2Unblind (re-visible loss; paper #22 in this study)
2023+ ── AP-BSN, LG-BPN, Steerable BSN ... (asymmetric pixel-shuffle BSN family)
```

이 논문은 **"단일 영상 + 알려진 잡음 모델 → N2N과 등가"** 라는 주장이 받아들여진 분기점이다. 이후 거의 모든 자기지도 디노이저는 R2R을 표준 비교 베이스라인으로 사용한다.

This paper marks the moment **"single noisy image + known noise statistics ⇒ Noise2Noise-equivalent training"** entered the field's vocabulary. From CVPR 2021 onward almost every self-supervised denoiser cites R2R as a baseline.

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Lehtinen et al. (2018)** *Noise2Noise* | Direct statistical foundation. R2R replaces "two independent noisy captures" with one captured + one synthesised. | Without N2N's theorem, R2R has no proof of equivalence to supervised training. |
| **Krull et al. (2018)** *Noise2Void* (paper #20) | Alternative single-image self-supervised paradigm with blind-spot. | R2R explicitly drops the blind-spot constraint and exceeds N2V on most benchmarks. |
| **Moran et al. (2020)** *Noisier2Noise* | Closest precursor: pre-corrupt $y$ by extra noise to form $(y+n', y)$ pair. | R2R is mathematically the same idea taken to its full *symmetric* form with covariance-matching, removing biased remainders. |
| **Wang et al. (2022)** *Blind2Unblind* (paper #22) | Subsequent self-supervised denoiser; uses R2R as a direct baseline in their tables. | B2U revives blind-spot but re-visits hidden pixels with a re-visible loss; the two papers give complementary attacks on the same problem. |
| **Wang et al. (2021)** *Neighbor2Neighbor* | Concurrent CVPR 2021 self-supervised work using sub-sampling to form noisy pairs. | NBR2NBR sub-samples spatial neighbours; R2R synthesises pairs in the noise-channel — different ways to manufacture independent noisy targets from one image. |
| **Anscombe (1948) / Makitalo-Foi (2010)** | Variance-stabilising transform for Poisson noise. | R2R uses Anscombe in the Poisson regime to reduce signal-dependent noise to (approximately) homoscedastic Gaussian where the recorruption identity applies. |
| **Zhang et al. (2017)** *DnCNN* | Standard supervised backbone. | R2R is backbone-agnostic; experiments use both DnCNN and U-Net. |
| **Lehtinen-style real raw-RGB** *SIDD* | Real-world benchmark | Demonstrates R2R works on real heteroscedastic Poisson–Gaussian noise found in mobile-phone CMOS sensors. |

---

### 6.1 Detailed comparison with key precursors / 주요 선행 연구와의 상세 비교

#### 한국어
- **vs Noisier2Noise (Moran et al. 2020)**: Noisier2Noise는 $y$에 추가 잡음 $n'$을 더해 $(y+n', y)$ 쌍 학습. 그러나 이 쌍은 *비대칭* — target $y$는 원래 noisy 그대로. R2R은 *양방향 비대칭화*로 (Eq. 명시적 수식) 분산을 정확히 일치시킴 → 더 깨끗한 통계적 보장.
- **vs Self2Self (Quan et al. 2020)**: Self2Self는 dropout 평균으로 단일 영상 학습. 추론 시 dropout 평균이 필수. R2R은 dropout 없이 결정적 네트워크 + 입력 측 stochasticity로 같은 효과.
- **vs N2V (Krull et al. 2018)**: N2V의 blind-spot은 정보 손실. R2R은 입력 전체 사용. 단점은 R2R이 잡음 모델 알아야 함.

#### English
- **vs Noisier2Noise (Moran 2020)**: Noisier2Noise pre-corrupts $y$ asymmetrically, training on $(y+n', y)$. The target is still the original noisy $y$ — biased remainder. R2R's *symmetric two-sided* corruption matches variances exactly → cleaner statistical guarantee.
- **vs Self2Self (Quan 2020)**: Self2Self averages dropout passes on a single image. R2R is dropout-free and uses input-side stochasticity for the same averaging effect.
- **vs N2V (Krull 2018)**: N2V's blind spot loses information. R2R uses the full input but requires a known noise model.

### 6.2 Reproducibility checklist / 재현성 체크리스트

#### 한국어
- 잡음 표준편차 $\sigma$ 추정: MAD/0.6745 또는 calibration data.
- $\alpha = 1$로 시작 → ablation에서 0.5/2.0 시도.
- 학습 iter: $10^5$ steps, Adam lr=1e-4, batch 4-32.
- 추론 K=50 MC averaging.
- 코드는 GitHub 공개되지는 않았으나 알고리즘은 단순해 직접 구현 가능 (본 study 노트북 참조).

#### English
- Estimate noise std $\sigma$ via MAD/0.6745 or instrument calibration data.
- Start with $\alpha=1$, then ablate at 0.5 and 2.0.
- Training: $10^5$ steps, Adam lr=1e-4, batch size 4–32.
- Inference: $K=50$ MC averaging.
- The paper code is not publicly released but the algorithm is simple to reproduce (see this study's accompanying notebook).

---

## 7. References / 참고문헌

- Pang, T., Zheng, H., Quan, Y., & Ji, H., "Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising", *CVPR 2021*, pp. 2043–2052. [DOI: 10.1109/CVPR46437.2021.00208]
- Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., & Aila, T., "Noise2Noise: Learning Image Restoration without Clean Data", *ICML 2018*.
- Krull, A., Buchholz, T.-O., & Jug, F., "Noise2Void — Learning Denoising from Single Noisy Images", *CVPR 2019*.
- Batson, J., & Royer, L., "Noise2Self: Blind Denoising by Self-Supervision", *ICML 2019*.
- Laine, S., Karras, T., Lehtinen, J., & Aila, T., "High-Quality Self-Supervised Deep Image Denoising", *NeurIPS 2019*.
- Quan, Y., Chen, M., Pang, T., & Ji, H., "Self2Self with Dropout: Learning Self-Supervised Denoising from Single Image", *CVPR 2020*.
- Moran, N., Schmidt, D., Zhong, Y., & Coady, P., "Noisier2Noise: Learning to Denoise from Unpaired Noisy Data", *CVPR 2020*.
- Huang, T., Li, S., Jia, X., Lu, H., & Liu, J., "Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images", *CVPR 2021*.
- Wang, Z., Liu, J., Li, G., & Han, H., "Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots", *CVPR 2022*.
- Mäkitalo, M., & Foi, A., "Optimal Inversion of the Anscombe Transformation in Low-Count Poisson Image Denoising", *IEEE TIP*, 20(1), 99–109 (2010).
- Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising (DnCNN)", *IEEE TIP*, 26(7), 3142–3155 (2017).
- Abdelhamed, A., Lin, S., & Brown, M. S., "A High-Quality Denoising Dataset for Smartphone Cameras (SIDD)", *CVPR 2018*.
