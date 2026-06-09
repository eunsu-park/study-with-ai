---
title: "Poisson NL Means: Unsupervised Non-Local Means for Poisson Noise"
authors: Charles-Alban Deledalle, Florence Tupin, Loïc Denis
year: 2010
journal: "IEEE ICIP 2010, pp. 801-804"
doi: "10.1109/ICIP.2010.5653394"
topic: Low-SNR Imaging / Patch-Based Denoising
tags: [non-local-means, poisson-noise, likelihood, kullback-leibler, pure, newton-method, patch-similarity, photon-limited-imaging]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 12. Poisson NL Means: Unsupervised Non-Local Means for Poisson Noise / 포아송 잡음용 비지역 평균

---

## 1. Core Contribution / 핵심 기여

### 한국어
Deledalle, Tupin, Denis(2010)는 NL Means(Buades et al. 2005; paper #4)를 포아송 잡음 모델에 적합하게 재설계하고 — 단지 패치 거리만 가우시안 가정에서 포아송 가정으로 바꾸는 것이 아니라, 잡음 영상과 사전 추정 영상을 **같이** 활용하는 refined NL Means 형태 ($\alpha$와 $\beta$ 두 파라미터) — PURE(Poisson Unbiased Risk Estimate)를 통해 두 파라미터를 *비지도적으로* 결정한다. 핵심 기여 세 가지:
(i) **포아송 패치 유사도**: 잡음 패치 비교에는 generalised likelihood ratio test (GLRT, Eq. 5):
$f_L(k_1, k_2) = k_1 \log k_1 + k_2 \log k_2 - (k_1 + k_2)\log\bigl(\tfrac{k_1+k_2}{2}\bigr)$
사전 추정 패치 비교에는 symmetric Kullback-Leibler divergence (Eq. 6).
(ii) **PURE for NL Means**: Chen(1975)의 포아송 식별 ($E[\lambda h(k)] = E[k\,h(\bar k)]$; Eq. 8)을 활용해 NL Means의 MSE를 잡음 영상만으로 비편향 추정. 결과: Eq. (9) $R(\hat\lambda) = \tfrac{1}{N}\sum_s (\lambda^2_s + \hat\lambda^2_s - 2 k_s \bar\lambda_s)$.
(iii) **Newton 최적화**: 두 필터링 파라미터 ($\alpha, \beta$)를 Newton's method로 5–14 반복 안에 PURE 최소점으로 수렴 (Eq. 11). 1차·2차 도함수 closed form 제공.

수치 결과 (Table 1, Peppers/Cameraman 256×256): Poisson NL Means가 PURE-LET (paper #13)와 classical NL Means보다 일관되게 0.1–1.5 dB 우수. 특히 매우 낮은 PSNR (3.14 dB → 19.90 dB) 영역에서 가장 큰 개선.

### English
Deledalle, Tupin, Denis (2010) adapt NL Means (Buades et al. 2005; paper #4) to the Poisson noise model. Crucially, the method goes beyond simply replacing the patch distance: it adopts a **refined NL Means** scheme that uses both the noisy image and a pre-filtered estimate (with two filter parameters $\alpha, \beta$) and tunes both parameters *unsupervisedly* via PURE (Poisson Unbiased Risk Estimate). Three contributions:
(i) **Poisson patch-similarity**: For comparing noisy patches, use the generalised likelihood-ratio test (GLRT, Eq. 5)
$f_L(k_1, k_2) = k_1 \log k_1 + k_2 \log k_2 - (k_1 + k_2)\log\bigl(\tfrac{k_1+k_2}{2}\bigr)$.
For comparing pre-estimated patches, use the symmetric Kullback-Leibler divergence (Eq. 6).
(ii) **PURE for NL Means**: Using Chen's (1975) Poisson identity ($E[\lambda h(k)] = E[k\,h(\bar k)]$; Eq. 8), the MSE of NL Means is estimated unbiasedly from the noisy image alone (Eq. 9).
(iii) **Newton optimisation**: The pair $(\alpha, \beta)$ is optimised in 5–14 iterations of Newton's method on PURE (Eq. 11), with closed-form first and second derivatives.

Numerical results (Table 1, Peppers/Cameraman 256×256): Poisson NL Means consistently beats PURE-LET (paper #13) and classical NL Means by 0.1–1.5 dB, with the biggest gain at very low PSNR (3.14 dB input → 19.90 dB output).

---

## 2. Reading Notes / 읽기 노트

### Part I: §1 Introduction & NL Means Setup / 도입과 NL Means 설정

#### 한국어
- **포아송 잡음의 어려움**: 잡음 분산이 신호에 의존 ($\mathrm{var}(k) = \lambda$)하므로 가우시안 가정 위에 설계된 도구를 직접 쓸 수 없다. 광학현미경, 천문 영상이 대표적.
- **고전 NL Means** (Buades+ 2005, Eq. 1): $\hat\lambda_s = \tfrac{\sum_t w_{s,t} k_t}{\sum_t w_{s,t}}$, 가중치 $w_{s,t} = \exp(-\sum_b (k_{s+b} - k_{t+b})^2 / \alpha)$ (Eq. 2). 가우시안 잡음에서 잘 작동하지만 포아송에서는 부적절.
- **Refined NL Means** (Eq. 3): 사전 추정 영상 $\hat\theta$를 활용:
  $w_{s,t} = \exp\bigl(-F_{s,t}/\alpha - G_{s,t}/\beta\bigr)$, 여기서 $F_{s,t}$는 잡음 영상의 패치 거리, $G_{s,t}$는 사전 추정 영상의 패치 거리.
- 두 파라미터 $\alpha, \beta$의 동시 최적화 문제는 미해결이었음 — 본 논문이 해결.

#### English
- Poisson noise is signal-dependent ($\mathrm{var}(k) = \lambda$), so Gaussian-tailored algorithms must be adapted.
- Classical NL Means (Eq. 1) uses Euclidean patch distance and one filtering parameter $\alpha$ (Eq. 2).
- Refined NL Means (Eq. 3) uses a *pre-filter* $\hat\theta$ (e.g., from a moving-average filter) and two filter parameters $\alpha, \beta$. Joint optimisation of $(\alpha, \beta)$ was an open question.

---

### Part II: §2 Patch-Similarities Under Poisson Noise / 포아송 잡음에서의 패치 유사도

#### 한국어
- **잡음 패치 거리 $f_L$**: Generalized likelihood ratio test from the hypothesis "$k_1, k_2$ share the same $\lambda$" vs "they have independent $\lambda_1, \lambda_2$":
  $$
  f_L(k_1, k_2) = -\log \frac{\max_\lambda p(k_1|\lambda) p(k_2|\lambda)}{\max_{\lambda_1} p(k_1|\lambda_1)\max_{\lambda_2} p(k_2|\lambda_2)}
  $$
  포아송 PMF $p(k|\lambda) = \lambda^k e^{-\lambda}/k!$을 대입하고 정리:
  $$
  f_L(k_1, k_2) = k_1 \log k_1 + k_2 \log k_2 - (k_1 + k_2)\log\bigl(\tfrac{k_1 + k_2}{2}\bigr)
  $$
  - 작은 $k$ 영역에서 잘 작동 (Euclidean보다 잘 함; 큰 광자수에서는 두 거리가 점근적으로 동등).
  - 패치 전체에 대해서는 $F_{s,t} = \sum_b f_L(k_{s+b}, k_{t+b})$.
- **사전 추정 패치 거리 $g_{KL}$**: 사전 추정값 $\hat\theta$는 (대략) 결정론적이라 가정 → 두 추정 강도 $\hat\theta_1, \hat\theta_2$의 동질성 검정에 symmetric KL divergence 사용:
  $$
  g_{KL}(\hat\theta_1, \hat\theta_2) = D_{KL}(\hat\theta_1 \| \hat\theta_2) = (\hat\theta_1 - \hat\theta_2)\log(\hat\theta_1/\hat\theta_2)
  $$
- 가우시안에서 잡음 분산 또는 quantile 기반으로 $\alpha$를 잡는 기존 방식은 포아송에 적용 불가 (분산이 $\lambda$에 의존).

#### English
- **Noisy-patch distance $f_L$** (Eq. 5): GLRT for the hypothesis that $k_1, k_2$ have a common Poisson rate vs independent rates. Closed form is the expression above.
- **Pre-estimate patch distance $g_{KL}$** (Eq. 6): symmetric Kullback-Leibler divergence between two intensity estimates. KL is the natural distance under the Poisson likelihood.
- These distances replace the squared Euclidean distance of classical NL Means.

---

### Part III: §3 Automatic Setting of Parameters via PURE / PURE에 의한 자동 파라미터 설정

#### 한국어
- **MSE risk** (Eq. 7): $E[\tfrac{1}{N}\|\lambda - \hat\lambda\|^2] = \tfrac{1}{N}\sum_s (\lambda^2_s + E[\hat\lambda^2_s] - 2 E[\lambda_s \hat\lambda_s])$. 첫 항 $\lambda^2$는 $\hat\lambda$ 무관하므로 최적화에서 제외.
- **Chen's identity** (Eq. 8): For Poisson $k$,
  $$
  E[\lambda_s h(k)_s] = E[k_s h(\bar k)_s]
  $$
  여기서 $\bar k_t = k_t$ ($t \ne s$인 경우) 또는 $\bar k_t = k_t - 1$ ($t = s$인 경우). 이는 SURE의 Stein lemma의 포아송 대응물.
- **PURE for NL Means** (Eq. 9): Chen identity를 (7)에 대입:
  $$
  R(\hat\lambda) = \tfrac{1}{N}\sum_s \bigl(\lambda^2_s + \hat\lambda^2_s - 2 k_s \bar\lambda_s\bigr)
  $$
  여기서 $\bar\lambda_s$는 $k$ 대신 $\bar k$에 NL Means를 적용한 결과 (Eq. 10).
- **계산 효율**: $\bar F_{s,t}$는 $F_{s,t}$에서 patch 안 한두 개 픽셀 차이만 고려하면 되므로 추가 비용 적음 (paper Eq. between 10 and 11).
- **Newton optimisation** (Eq. 11): $(\alpha, \beta)^{(n+1)} = (\alpha, \beta)^{(n)} - H^{-1}\nabla$. 1차·2차 미분 closed form 제공:
  - $\partial R/\partial x = \tfrac{2}{N}\sum_s \hat\lambda_s \tfrac{\partial \hat\lambda_s}{\partial x} - \tfrac{2}{N}\sum_s k_s \tfrac{\partial \bar\lambda_s}{\partial x}$
  - $\tfrac{\partial \hat\lambda_s}{\partial x} = \tfrac{\sum X_{s,t} w_{s,t}(k_t - \hat\lambda_s)}{x^2 \sum w_{s,t}}$ where $X = F$ when $x = \alpha$, $X = G$ when $x = \beta$.

#### English
- The MSE (Eq. 7) is decomposed into three terms; the first is independent of $\hat\lambda$.
- Chen's identity (Eq. 8) is the Poisson analogue of Stein's lemma, providing an unbiased estimator of $E[\lambda h(k)]$ without knowing $\lambda$.
- Substituting yields PURE (Eq. 9), which is *unbiased* and *closed-form* for NL Means.
- Two-parameter optimisation by Newton's method converges in 6–14 iterations with closed-form gradients.

---

### Part IV: §4 Experiments and Results / 실험과 결과

#### 한국어
- **Setup**: 21×21 search window, 7×7 patches. Pre-estimate $\hat\theta$ is a moving average with 13×13 disk kernel. Newton iterates until $\Delta$PURE $\le \epsilon$.
- **Visual diagnostic** (Fig. 1): MSE와 PURE 곡선 (그리고 1차·2차 도함수)이 $\alpha$와 $\beta$ 평면에서 거의 일치 — PURE가 MSE의 충실한 대리.
- **Table 1 결과**: 두 영상(Peppers, Cameraman) × 4 잡음 레벨 (peak 1, peak 5, peak 10, peak 30 정도에 해당하는 입력 PSNR 3-24 dB). 비교 알고리즘:
  - **MA filter** (단순 baseline): 약 19-21 dB.
  - **PURE-LET** (Luisier+ 2010 — paper #13의 ICIP 2010 conference 버전): 19.3-30.8 dB.
  - **NL means** (classical, Euclidean): 18.1-30.6 dB.
  - **Refined NL means** ($(x-y)^2$ for both): 19.8-30.9 dB.
  - **Poisson NL means** (proposed): **19.9-31.1 dB** — 모든 시나리오에서 1위.
- **개선 폭**: Peppers, peak 1 (input 3.14 dB):
  - MA: 19.20, NL: 18.12, refined: 19.84, **Poisson NL: 19.90 dB**.
  - PURE-LET: 19.33 (Poisson NL이 0.57 dB 우위).
- **$\alpha_{\rm opt}$, $\beta_{\rm opt}$의 행동**:
  - 잡음이 클수록 $\alpha$ 작아짐 (잡음 영상의 신뢰도 낮음 → $F$를 신중히 평가).
  - 잡음이 작을수록 $\beta$ 커짐 (사전 추정의 신뢰도 높음 → $G$에 의존).
- **시간**: 21×21 window + 7×7 patch, 256×256 영상 → $\sim 10$초/iteration (C 구현). 6-14 iteration → $\sim 1$분.

#### English
- Setup: 21×21 search window, 7×7 patches; pre-estimate from 13×13 moving average.
- Fig. 1 verifies that PURE accurately tracks the true MSE as a function of $(\alpha, \beta)$, validating the unbiased estimate.
- Table 1 shows Poisson NL Means consistently outperforms classical NL Means, refined NL Means, and PURE-LET on standard test images at all four noise levels (input PSNR 3-24 dB).
- The optimal $\alpha$ decreases with noise level (low-quality noisy data forces caution), and $\beta$ increases when the noise is small (high-quality pre-estimate is trusted more).
- Real fluorescence-microscopy results (Fig. 2 mitochondrion image) show fewer processing artefacts than PURE-LET.

---

### Part V: §5 Conclusion / 결론

#### 한국어
- 본 논문은 (i) 포아송 잡음에 적합한 likelihood 기반 패치 거리, (ii) Chen identity로 유도한 PURE for NL Means, (iii) 두 파라미터 동시 최적화의 Newton 방식 — 세 요소를 결합해 비지도적 포아송 NL Means 알고리즘을 완성.
- 한계: 사전 추정 $\hat\theta$의 잡음 분산이 충분히 작아야 (10) 식의 가정이 성립; 첫 통과에 MA filter, 두 번째 통과에 본 알고리즘으로 iterative refinement 가능.

#### English
- The paper combines three ingredients: Poisson-likelihood patch distance, PURE for NL Means via Chen's identity, and joint Newton optimisation of $(\alpha, \beta)$.
- The pre-estimate $\hat\theta$ must be sufficiently smooth so its variance is negligible (Eq. 10's independence assumption); the recipe is a coarse MA filter as starting point.

---

## 3. Key Takeaways / 핵심 시사점

1. **포아송에서는 Euclidean 패치 거리 대신 likelihood-ratio 거리를 써야 한다 / Poisson noise calls for likelihood-based patch distance** — Eq. (5)의 $f_L$은 두 패치가 같은 $\lambda$에서 왔다는 가설의 GLRT. 포아송 PMF의 비대칭성을 직접 반영하기 때문에 매우 낮은 광자 수 영역에서도 강건하다. Euclidean 거리는 $k = 0$ 같은 경계에서 정보가 손실됨.
   The $f_L$ distance directly reflects the asymmetry of the Poisson PMF and remains informative even at very low photon counts where Euclidean distance becomes unreliable (e.g., when many patch pixels are zero).

2. **Refined NL Means는 sigle-parameter NL의 진정한 후속 / Refined NL Means with two parameters is the proper successor to classical NL Means** — 잡음 영상과 사전 추정 영상을 *동시에* 가중치에 반영하는 것이 진짜 본질. 이 paper의 핵심 기여는 두 파라미터를 동시에 자동 결정하는 방법을 제공한 것.
   Refined NL Means uses *both* the noisy image (via $F_{s,t}/\alpha$) and the pre-estimate (via $G_{s,t}/\beta$) in the weight formula. The paper's central novelty is the joint, automatic tuning of both parameters.

3. **PURE는 Poisson에서 SURE의 정확한 대응물 / PURE is the exact Poisson analogue of SURE** — Chen identity $E[\lambda h(k)] = E[k h(\bar k)]$는 Stein lemma의 Poisson 버전 (paper #13에서도 사용됨). 이 identity 하나로 $\lambda$를 모르고도 MSE를 비편향 추정. Wavelet shrinkage(Luisier+ 2010), NLM, 그 외 어떤 추정량에도 적용 가능한 보편적 도구.
   Chen's identity is the Poisson Stein's lemma — it allows unbiased MSE estimation without knowing $\lambda$. PURE generalises across estimators (wavelet thresholding, NLM, neural networks).

4. **$\bar k$는 $s$번째 픽셀에서만 1을 빼는 단순 변형 / The $\bar k$ trick is just "subtract 1 at the test pixel"** — Chen identity에서 $\bar k_t = k_t - 1$ ($t = s$) 또는 $k_t$ (otherwise). 따라서 PURE 계산은 NL Means를 한 번 더 돌리는 비용 없이, $F_{s,t}$를 점진적으로 갱신 (paper의 $\bar F$ 식)함으로써 가능.
   The $\bar k$ trick — subtracting 1 only at the test pixel — has minimal computational overhead because the patch distances $\bar F_{s,t}$ differ from $F_{s,t}$ by only one or two terms.

5. **Newton method는 두 변수에서 6-14회 안에 수렴 / Newton converges in 6-14 iterations on the 2D $(\alpha, \beta)$ parameter space** — Closed-form 1차/2차 미분이 있어 quasi-Newton보다 효율적. PURE가 $(\alpha, \beta)$의 매끄러운 함수이므로 Newton 안정.
   The closed-form Hessian and gradient make Newton's method efficient on the 2D parameter space; convergence in fewer than 15 iterations is typical.

6. **$\alpha$와 $\beta$는 잡음 영상과 사전 추정의 “신뢰도”를 자동 균형 / $\alpha$ and $\beta$ automatically balance the trust in the noisy image vs the pre-estimate** — 직관적 해석: 사전 추정이 좋으면 (낮은 잡음) $\alpha$가 커지고 (잡음 영상 거리는 무시), 사전 추정이 나쁘면 $\beta$가 커지고 (사전 추정 거리는 무시). 이는 paper의 Table 1 $(\alpha_{\rm opt}, \beta_{\rm opt})$값에서 직접 관찰됨.
   The two parameters automatically encode the relative quality of the noisy and pre-estimated images: high $\alpha$ means "trust pre-estimate", high $\beta$ means "trust noisy image". This is empirically visible in Table 1.

7. **저광자 영역에서 PURE-LET을 1.0+ dB 능가 / Outperforms PURE-LET (paper #13) by 1+ dB at very low photon counts** — Peppers peak 1: PURE-LET 19.33 dB vs Poisson NL 19.90 dB. 패치 기반 비지역 평균이 wavelet shrinkage보다 매우 낮은 SNR에서 더 강건. 이는 NLM 계열의 일반적 강점.
   At very low photon counts (input PSNR 3 dB), Poisson NL Means achieves 19.90 dB versus PURE-LET's 19.33 dB on Peppers — a 0.57 dB gap reflecting the patch-based method's robustness at low SNR.

8. **Iterative refinement이 가능한 알고리즘 / The algorithm naturally lends itself to iterative refinement** — 사전 추정 $\hat\theta$를 반복마다 업데이트하면 ($\hat\theta^{(0)} = $ MA filter, $\hat\theta^{(n+1)} = $ Poisson NL Means output)) 추가 개선 가능. 본 paper는 단일 반복만 보여주지만 후속 작업에서 표준화.
   Setting $\hat\theta$ to the previous Poisson-NLM output and iterating yields further improvements — a recipe inherited from PPB (Probabilistic Patch-Based) denoising of Deledalle+ 2009.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Setting / 설정
$$
k_s \sim \mathrm{Poisson}(\lambda_s), \quad p(k|\lambda) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad s = 1, ..., N
$$
### 4.2 Refined NL Means estimator (Eq. 3) / 개선된 NL Means
$$
\hat\lambda_s = \frac{\sum_t w_{s,t} k_t}{\sum_t w_{s,t}}, \quad w_{s,t} = \exp\!\Bigl(-\frac{F_{s,t}}{\alpha} - \frac{G_{s,t}}{\beta}\Bigr)
$$
$$
F_{s,t} = \sum_b f(k_{s+b}, k_{t+b}), \qquad G_{s,t} = \sum_b g(\hat\theta_{s+b}, \hat\theta_{t+b})
$$
### 4.3 Poisson similarity (Eq. 5) / 포아송 유사도
$$
\boxed{\; f_L(k_1, k_2) = k_1 \log k_1 + k_2 \log k_2 - (k_1 + k_2) \log\!\Bigl(\frac{k_1 + k_2}{2}\Bigr) \;}
$$
- $k_1 = k_2$이면 $f_L = 0$ (완벽 일치).
- $k = 0$ 항은 convention $0 \log 0 = 0$로 처리.
- 패치 전체: $F_{s,t} = \sum_b f_L(k_{s+b}, k_{t+b})$.

### 4.4 KL divergence for pre-estimates (Eq. 6) / 사전 추정 KL
$$
g_{KL}(\hat\theta_1, \hat\theta_2) = (\hat\theta_1 - \hat\theta_2) \log\!\frac{\hat\theta_1}{\hat\theta_2}
$$
### 4.5 Chen identity (Eq. 8) / Chen 항등식
$$
E[\lambda_s\,h(k)_s] = E[k_s\,h(\bar k)_s], \quad \bar k_t = \begin{cases} k_t & t \ne s \\ k_t - 1 & t = s \end{cases}
$$
### 4.6 PURE (Eq. 9) / Poisson Unbiased Risk Estimate
$$
\boxed{\; R(\hat\lambda) = \frac{1}{N}\sum_s \bigl(\lambda^2_s + \hat\lambda^2_s - 2 k_s \bar\lambda_s\bigr) \;}
$$
where $\bar\lambda_s$ is NL Means applied to $\bar k$ instead of $k$. The first term $\lambda^2_s$ is constant in $(\alpha, \beta)$ and dropped during optimisation.

### 4.7 Newton update (Eq. 11) / Newton 갱신
$$
\binom{\alpha^{(n+1)}}{\beta^{(n+1)}} = \binom{\alpha^{(n)}}{\beta^{(n)}} - H^{-1} \nabla R
$$
$$
\nabla R = \binom{\partial R / \partial \alpha}{\partial R / \partial \beta}, \quad
H = \begin{pmatrix} \partial^2 R/\partial \alpha^2 & \partial^2 R/\partial \alpha \partial \beta \\ \partial^2 R/\partial \beta\partial \alpha & \partial^2 R/\partial \beta^2 \end{pmatrix}
$$
Closed-form derivatives:
$$
\frac{\partial \hat\lambda_s}{\partial x} = \frac{\sum_t X_{s,t} w_{s,t} (k_t - \hat\lambda_s)}{x^2 \sum_t w_{s,t}}
$$
where $X = F$ ($x = \alpha$) or $X = G$ ($x = \beta$).

### 4.8 Worked numerical example / 수치 예시
For a 64×64 synthetic Poisson image with peak 10:
- input PSNR $\approx 13$ dB.
- Run Poisson NL Means with 21×21 search, 7×7 patches, MA pre-estimate (13×13 disk).
- Newton starts at $(\alpha_0, \beta_0) = (10, 1)$, converges to $(\alpha^*, \beta^*) \approx (10.05, 2.76)$ (close to paper's Peppers peak-10 entry, Table 1 $(10.05, 2.76)$) in $\approx 7$ iterations.
- PSNR rises from 13 dB → $\approx 28$ dB. Comparable to paper Table 1's Peppers peak-10 result of 28.07 dB.

### 4.9 Derivation of the Poisson GLRT distance / 포아송 GLRT 거리 유도

Poisson PMF: $p(k|\lambda) = \lambda^k e^{-\lambda}/k!$.

Single-rate hypothesis ($H_0$): $\lambda_1 = \lambda_2 = \lambda$. MLE for $\lambda$ given $k_1, k_2$:
$$
\hat\lambda_{H_0} = \frac{k_1 + k_2}{2}
$$
Independent-rates hypothesis ($H_1$): $\lambda_1 \ne \lambda_2$. MLE: $\hat\lambda_{1,H_1} = k_1$, $\hat\lambda_{2, H_1} = k_2$.

Likelihood ratio:
$$
\Lambda = \frac{\max p(k_1|\lambda) p(k_2|\lambda)}{\max p(k_1|\lambda_1)\max p(k_2|\lambda_2)} = \frac{(\hat\lambda_{H_0})^{k_1+k_2} e^{-2\hat\lambda_{H_0}}}{k_1^{k_1} k_2^{k_2} e^{-(k_1+k_2)}}
$$
$$
-\log \Lambda = k_1 \log k_1 + k_2 \log k_2 - (k_1 + k_2)\log\!\Bigl(\frac{k_1+k_2}{2}\Bigr)
$$
이는 정확히 (5)의 $f_L$. 따라서 $f_L$은 “두 패치가 같은 rate에서 왔다”는 가설에 대한 -log likelihood ratio.

Derivation: substitute the MLE under each hypothesis into the likelihood ratio and take the negative log; the result is exactly Eq. (5)'s $f_L$. It is therefore the *negative log-likelihood ratio* test statistic for "do these two pixels share a Poisson rate?".

### 4.10 Computational complexity / 계산 복잡도

For an $N \times N$ image with patch size $(2p+1)^2$ and search window $(2s+1)^2$:
- Patch distance computation: $O(N^2 \cdot (2s+1)^2 \cdot (2p+1)^2)$ for full evaluation, or $O(N^2 \cdot (2s+1)^2)$ using integral images.
- Newton iteration: each step requires one full NL Means evaluation $\approx O(N^2 \cdot (2s+1)^2)$.
- Total: 6-14 Newton steps × 1 NL Means each $\sim 10$ NL Means evaluations.
- Paper reports $\sim 10$ s/iteration on 256×256 images (C implementation, Intel Core 2 Duo 3 GHz) → $\sim 100$ s total per image. Modern GPU implementations can be $>100\times$ faster.

The PURE evaluation has minimal overhead because $\bar k$ differs from $k$ only at one or two pixels per patch.

---

### 4.11 Behaviour of $(\alpha, \beta)$ at extreme noise levels / 양극단 잡음 영역에서의 행동

From Table 1 of the paper:

| Image | Peak | Input PSNR | $\alpha_{\rm opt}$ | $\beta_{\rm opt}$ |
|---|---|---|---|---|
| Peppers | very low (peak 1) | 3.14 dB | 209 | 0.72 |
| Peppers | low | 13.14 dB | 13.6 | 1.31 |
| Peppers | medium | 17.91 dB | 10.05 | 2.76 |
| Peppers | high | 23.92 dB | 9.21 | 7.64 |

**Observation**:
- 잡음이 매우 높을 때 (peak 1): $\alpha$ 매우 큼 (잡음 영상 거리 무시), $\beta$ 작음 (사전 추정에 의존).
- 잡음이 낮을 때 (peak 30): $\alpha$ 보통, $\beta$ 큼 (사전 추정 거리도 무시; 잡음 영상이 신뢰할 만함).

이는 paper의 직관적 해석을 정확히 반영: PURE는 두 정보원의 *상대적 신뢰도*에 따라 자동으로 가중치를 조정.

The behaviour exactly matches the paper's intuition: PURE automatically balances the trust in noisy and pre-estimated images according to their relative quality.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1948 ─── Anscombe — VST for Poisson data (paper #11)
1981 ─── Stein — SURE for Gaussian noise estimation
1975 ─── Chen — Poisson approximation lemma (Eq. 8 origin)
1995 ─── Donoho-Johnstone — SureShrink for Gaussian wavelets
2005 ─── Buades-Coll-Morel — Non-Local Means (paper #4)
2006 ─── Kervrann-Boulanger — Optimal spatial adaptation NLM
2007 ─── Brox-Kleinschmidt-Cremers — efficient nonlocal means
2009 ─── Deledalle-Denis-Tupin — PPB (Probabilistic Patch-Based)
                                ↳ direct precursor to this paper
2009 ─── Van De Ville-Kocher — SURE-based NL Means (Gaussian)
2010 ★★ DELEDALLE-TUPIN-DENIS: Poisson NL Means (this paper)
                                ↳ likelihood patch distance + PURE
2010 ─── Luisier-Vonesch-Blu-Unser — fast PURE-LET (precursor to #13)
2011 ─── Luisier-Blu-Unser — PURE-LET for Poisson-Gaussian (paper #13)
2013 ─── Mäkitalo-Foi — exact unbiased inverse of GAT (paper #14)
2017+ ── Deep-learning denoisers (Noise2Noise, Noise2Self)
                                ↳ many descend from PURE/SURE ideas.
```

이 논문은 **NL Means가 가우시안 잡음 외의 잡음 모델로 확장되는 결정적 분기점**이자, **PURE를 NLM에 적용한 첫 작업**이다. 후일 self-supervised denoising (Noise2Noise) 등에 PURE-style risk estimator가 표준 도구로 자리잡는다.

This paper is **the decisive extension of NL Means beyond Gaussian noise**, and **the first application of PURE to NLM**. PURE-style unbiased risk estimators became a standard tool for self-supervised denoising (Noise2Noise et al.).

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Buades-Coll-Morel (2005)** *CVPR* (#4 in topic) | NL Means original | This paper's direct ancestor. |
| **Anscombe (1948)** *Biometrika* (#11) | VST | Alternative route to Poisson denoising; this paper bypasses VST. |
| **Luisier-Blu-Unser (2011)** *IEEE TIP* (#13) | PURE-LET (Poisson-Gaussian wavelet thresholding) | Direct comparison in Table 1; uses same PURE concept. |
| **Mäkitalo-Foi (2013)** *IEEE TIP* (#14) | Exact unbiased inverse of GAT | Different approach to same problem (Poisson-Gaussian denoising). |
| **Stein (1981)** *Annals of Statistics* | SURE | Gaussian Stein lemma; Chen identity is Poisson analogue. |
| **Chen (1975)** *Annals of Probability* | Poisson approximation identity (Eq. 8) | Mathematical foundation of PURE. |
| **Van De Ville-Kocher (2009)** *IEEE SPL* | SURE-based NL Means for Gaussian | Direct precursor; this paper extends to Poisson. |
| **Deledalle-Denis-Tupin (2009)** *IEEE TIP* | PPB (Probabilistic Patch-Based) | Same authors' earlier work on probabilistic patch weights. |
| **Dabov-Foi-Katkovnik-Egiazarian (2007)** *IEEE TIP* (#7) | BM3D | Alternative patch-based denoiser; later extended to Poisson by Mäkitalo-Foi 2013 (paper #14). |
| **Donoho-Johnstone (1995)** *JASA* (#2) | SureShrink (Gaussian) | Original SURE-based denoising; PURE is its Poisson generalisation. |

---

## 7. References / 참고문헌

- Deledalle, C.-A., Tupin, F., & Denis, L., "Poisson NL means: Unsupervised non local means for Poisson noise", *Proc. IEEE ICIP*, pp. 801–804 (2010). [DOI: 10.1109/ICIP.2010.5653394]
- Buades, A., Coll, B., & Morel, J.-M., "A non-local algorithm for image denoising", *Proc. IEEE CVPR*, pp. 60–65 (2005).
- Chen, L. H. Y., "Poisson approximation for dependent trials", *Annals of Probability*, 3(3), 534–545 (1975).
- Deledalle, C.-A., Denis, L., & Tupin, F., "Iterative weighted maximum likelihood denoising with probabilistic patch-based weights", *IEEE Trans. Image Process.*, 18(12), 2661–2672 (2009).
- Kervrann, C., & Boulanger, J., "Optimal spatial adaptation for patch-based image denoising", *IEEE Trans. Image Process.*, 15(10), 2866–2878 (2006).
- Luisier, F., Vonesch, C., Blu, T., & Unser, M., "Fast interscale wavelet denoising of Poisson-corrupted images", *Signal Processing*, 90(2), 415–427 (2010).
- Stein, C. M., "Estimation of the mean of a multivariate normal distribution", *Annals of Statistics*, 9(6), 1135–1151 (1981).
- Van De Ville, D., & Kocher, M., "SURE-based non-local means", *IEEE Signal Process. Letters*, 16(11), 973–976 (2009).
- Donoho, D. L., & Johnstone, I. M., "Adapting to unknown smoothness via wavelet shrinkage", *J. American Statistical Association*, 90(432), 1200–1224 (1995).
