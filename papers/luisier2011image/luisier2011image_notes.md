---
title: "Image Denoising in Mixed Poisson-Gaussian Noise"
authors: Florian Luisier, Thierry Blu, Michael Unser
year: 2011
journal: "IEEE Trans. Image Processing 20(3), pp. 696-708"
doi: "10.1109/TIP.2010.2073477"
topic: Low-SNR Imaging / Risk-Estimate Denoising
tags: [pure, sure, let, linear-expansion-of-thresholds, poisson-gaussian, hudson-stein, wavelet, undecimated, bdct, pure-let]
status: completed
date_started: 2026-05-06
date_completed: 2026-05-06
---

# 13. Image Denoising in Mixed Poisson-Gaussian Noise / 포아송-가우시안 혼합 잡음 영상 잡음 제거

---

## 1. Core Contribution / 핵심 기여

### 한국어
Luisier, Blu, Unser(2011)는 **포아송-가우시안 혼합 잡음 $y = z + b$ ($z \sim \mathcal P(x)$, $b \sim N(0, \sigma^2)$)을 위한 PURE-LET 프레임워크**를 제시한다. PURE-LET = **PURE** (Poisson-Gaussian Unbiased Risk Estimate) + **LET** (Linear Expansion of Thresholds). 두 가지 핵심 발견:
(i) **PURE의 일반화 (Theorem 1)**: Stein lemma (가우시안용)와 Hudson identity (포아송용)를 조합하여, 추정량 $f(\mathbf y)$에 대해 다음이 비편향 MSE 추정량:
$$
\hat\varepsilon = \tfrac{1}{N}\bigl(\|\mathbf f(\mathbf y)\|^2 - 2 \mathbf y^T \mathbf f^-(\mathbf y) + 2 \sigma^2 \mathrm{div}\{\mathbf f^-(\mathbf y)\}\bigr) + \tfrac{1}{N}(\|\mathbf y\|^2 - \mathbf 1^T \mathbf y) - \sigma^2
$$
여기서 $\mathbf f^-(\mathbf y)_n = f_n(\mathbf y - \mathbf e_n)$. 이는 SURE의 직접적 일반화이며, 가우시안 ($\sigma^2 \to 0$ 제거시) 또는 순 포아송 ($\sigma = 0$) 경우로 특화 가능.
(ii) **LET 패러미터화**: 추정량을 $\mathbf f(\mathbf y) = \sum_{k=1}^K a_k \mathbf R \boldsymbol\Theta_k(\mathbf w, \bar{\mathbf w})$로 *선형 결합*. PURE는 $a_k$에 대해 *2차식*이 되므로, 최적 파라미터는 $K$차 선형방정식 $\mathbf M \mathbf a = \mathbf c$ (Eq. 15)의 해 — closed form, exhaustive search 불필요.
(iii) **Pointwise subband-dependent thresholding (Eq. 19-20)**: 신호 의존적 임계 $t_j(\bar w) = \sqrt{\beta_j |\bar w| + \sigma^2}$와 부드러운 hard-like $\theta_j(w, \bar w)$ (Eq. 20). 가우시안 임계가 신호 강도에 따라 자동 조정됨.

수치 결과 (Table I-II):
- 순 포아송 (Cameraman peak 30): UWT/BDCT PURE-LET = 27.91 dB, Anscombe+BLS-GSM = 27.54 dB.
- 혼합 P-G (Cameraman peak 60, σ=6): UWT/BDCT PURE-LET = 27.37 dB, GAT+BLS-GSM = 27.02 dB.
- 실 형광현미경 영상에서 platelet, Anscombe-GSM 대비 우수한 결과.

### English
Luisier, Blu, Unser (2011) propose a **PURE-LET framework for mixed Poisson-Gaussian noise** $y = z + b$ where $z \sim \mathcal P(x)$ and $b \sim N(0, \sigma^2)$. PURE-LET combines two ideas:
(i) **Generalised PURE (Theorem 1)**: Combining Stein's lemma (for Gaussian) and Hudson's identity (for Poisson), the random variable $\hat\varepsilon$ (Eq. 2) is an unbiased estimator of the MSE $E\|f(\mathbf y) - \mathbf x\|^2/N$. It specialises correctly to SURE (pure Gaussian) and to PURE (pure Poisson).
(ii) **LET parameterisation**: Denoiser is written as a linear combination $\mathbf f(\mathbf y) = \sum_{k=1}^K a_k \mathbf R \boldsymbol\Theta_k(\mathbf w, \bar{\mathbf w})$ (Eq. 14) of $K$ "thresholding experts". PURE is then *quadratic* in $\{a_k\}$, so the optimal parameters solve a $K \times K$ linear system $\mathbf M \mathbf a = \mathbf c$ (Eq. 15) — closed-form, no exhaustive search.
(iii) **Pointwise subband-dependent thresholding (Eq. 19-20)**: Signal-dependent threshold $t_j(\bar w) = \sqrt{\beta_j |\bar w| + \sigma^2}$ plus a smooth, hard-like nonlinearity $\theta_j(w, \bar w)$ (Eq. 20).

Numerical results (Tables I-II):
- Pure Poisson (Cameraman peak 30): UWT/BDCT PURE-LET = 27.91 dB vs Anscombe+BLS-GSM = 27.54 dB.
- Mixed P-G (Cameraman peak 60, σ=6): UWT/BDCT PURE-LET = 27.37 dB vs GAT+BLS-GSM = 27.02 dB.
- Outperforms platelet and Anscombe-GSM on real fluorescence microscopy data.

---

## 2. Reading Notes / 읽기 노트

### Part I: §I-II Setting and Theory / 설정과 이론

#### 한국어
- **잡음 모델**: $\mathbf y = \mathbf z + \mathbf b$, $\mathbf z \sim \mathcal P(\mathbf x)$ (각 픽셀 $z_n \sim \mathrm{Poisson}(x_n)$), $\mathbf b \sim N(\mathbf 0, \sigma^2 \mathbf I)$. 광검출기의 두 주요 잡음원: photon-counting (Poisson) + thermal/electronic (Gaussian).
- **MSE estimator** (Theorem 1, Eq. 2):
  $$
  \hat\varepsilon = \tfrac{1}{N}\Bigl(\|\mathbf f(\mathbf y)\|^2 - 2 \mathbf y^T \mathbf f^-(\mathbf y) + 2\sigma^2 \mathrm{div}\{\mathbf f^-(\mathbf y)\}\Bigr) + \tfrac{1}{N}(\|\mathbf y\|^2 - \mathbf 1^T \mathbf y) - \sigma^2
  $$
  - $\mathbf f^-(\mathbf y)_n = f_n(\mathbf y - \mathbf e_n)$ (n번째 좌표만 1을 빼서 평가).
  - $\mathrm{div}\{\mathbf f^-\} = \sum_n \partial f_n / \partial y_n$은 추정량의 발산.
- **유도**: 두 properties를 조합:
  1. **Stein lemma** (Property 1): $E_{\mathbf b}\{\mathbf b^T \mathbf f(\mathbf y)\} = \sigma^2 E_{\mathbf b}\{\mathrm{div}\{\mathbf f(\mathbf y)\}\}$ — 가우시안용.
  2. **Hudson identity** (Property 2): $E_{\mathbf z}\{\mathbf x^T \mathbf f(\mathbf z)\} = E_{\mathbf z}\{\mathbf z^T \mathbf f^-(\mathbf z)\}$ — 포아송용.
  - 둘 다 적용하여 $E\{\mathbf x^T \mathbf f(\mathbf y)\}$를 $\mathbf y$와 $\sigma^2$만의 함수로 표현. $\mathbf x$에 대한 의존성 제거.
- **Taylor 근사** (§II-C): $\mathbf f^-(\mathbf y)$의 정확한 계산은 $N$번 추정 알고리즘을 돌려야 — 비현실적. 1차 Taylor 전개 $f_n(\mathbf y - \mathbf e_n) \approx f_n(\mathbf y) - \partial f_n/\partial y_n$으로 근사 (Eq. 6). $f_n$이 충분히 매끄러우면 (Lipschitz) 정확.

#### English
- Mixed model: $\mathbf y = \mathbf z + \mathbf b$, $\mathbf z \sim \mathcal P(\mathbf x)$, $\mathbf b \sim N(\mathbf 0, \sigma^2 \mathbf I)$.
- The MSE estimator combines Stein's lemma (Gaussian) and Hudson's identity (Poisson). Both are applied to express $E\{\mathbf x^T \mathbf f(\mathbf y)\}$ — the only $\mathbf x$-dependent piece — in terms of $\mathbf y$ and known noise parameters only.
- A first-order Taylor approximation in Section II-C avoids the impractical $N$-times re-evaluation of the denoiser.

---

### Part II: §III Transform-Domain Pointwise Estimator / 변환 영역 점별 추정기

#### 한국어
- **Generic transform-domain denoiser**: $\hat{\mathbf x} = \mathbf R \boldsymbol\Theta(\mathbf w, \bar{\mathbf w})$, 여기서 $\mathbf w = \mathbf D \mathbf y$ (decomposition), $\bar{\mathbf w} = \bar{\mathbf D}\mathbf y$ (smoothed decomposition for noise variance estimation), $\mathbf R$은 reconstruction. $\mathbf R \mathbf D = \mathbf I$ (perfect reconstruction).
- **Group-Delay Compensation (GDC, §III-A)**: Undecimated wavelet에서 lowpass와 bandpass 채널 사이 group delay 차이를 보정하는 allpass filter $Q(z^{-1})$. 이 GDC가 신호 의존적 잡음 분산 추정의 정확도를 높임.
- **Subband-dependent threshold (Eq. 19)**:
  $$
  t_j(\bar w) = \sqrt{\beta_j |\bar w| + \sigma^2}
  $$
  - $\beta_j$는 scale-dependent factor ($\beta_j = 2^{-j/2}$ for multiscale, $\beta_j = M^{-1/2}$ for BDCT).
  - 신호가 강한 곳 (큰 $|\bar w|$)에서 임계 $t_j$ 증가 → 자동 적응.
- **Smooth thresholding function (Eq. 20)**:
  $$
  \theta_j(w, \bar w) = a_{j,1}\,w + a_{j,2}\,w \exp(-(w/(3 t_j(\bar w)))^8)
  $$
  - 첫 항은 linear (no thresholding), 둘째 항은 hard-like 차감 (큰 $|w|$에서 0).
  - $|w|$는 미분 가능 근사 $\tanh(k\bar w)\bar w$로 대체 ($k = 100$).

#### English
- Transform-domain denoiser $\hat{\mathbf x} = \mathbf R \boldsymbol\Theta(\mathbf w, \bar{\mathbf w})$ with perfect reconstruction $\mathbf R \mathbf D = \mathbf I$.
- The signal-dependent threshold (Eq. 19) automatically adapts: stronger signal -> larger threshold (since the wavelet of a Poisson signal has variance proportional to signal magnitude).
- The thresholding function (Eq. 20) is a smooth, differentiable, hard-like nonlinearity. The differentiability is essential for the closed-form gradient in the LET optimisation.

---

### Part III: §III-E LET Strategy and Linear System / LET 전략과 선형방정식

#### 한국어
- **LET parameterisation**: $\mathbf f(\mathbf y) = \sum_{k=1}^K a_k \mathbf f_k(\mathbf y)$ where $\mathbf f_k(\mathbf y) = \mathbf R \boldsymbol\Theta_k(\mathbf w, \bar{\mathbf w})$. 각 $\boldsymbol\Theta_k$는 다른 임계기 (예: 다른 $\beta_j$ 또는 다른 nonlinearity).
- **PURE의 quadratic 형태**: PURE estimator $\hat\varepsilon$이 $a_k$에 대해 2차이므로:
  $$
  \mathbf M \mathbf a = \mathbf c, \quad [\mathbf M]_{k,l} = \mathbf f_k(\mathbf y)^T \mathbf f_l(\mathbf y), \quad [\mathbf c]_k = \mathbf y^T \mathbf f^-_k(\mathbf y) - \sigma^2 \mathrm{div}\{\mathbf f^-_k(\mathbf y)\}
  $$
- **Multi-base LET (Eq. 23)**: 여러 transform을 동시에 사용 — $\mathbf f(\mathbf y) = \sum_k a_k \mathbf f^{\rm UWT}_k(\mathbf y) + \sum_k b_k \mathbf f^{\rm BDCT}_k(\mathbf y)$ — 각 변환의 강점 (UWT for piecewise-smooth, BDCT for textured)을 결합.
- **$K = 2J$ 최소 시스템**: $J$ 분해 레벨 × 2 (linear + hard-like) $= 2J$ 자유도.
- **Reliability check**: 매우 어두운 신호 ($x_n < 5$)에서 1차 Taylor 근사가 부정확할 수 있어 — 그런 서브밴드 $\mathbf f_{j,2}(\mathbf y)$는 disabled (paper Fig. 6-7).

#### English
- LET writes the denoiser as a *linear combination* of $K$ "thresholding experts", each parameterised by a distinct nonlinearity.
- Because PURE is *quadratic* in the LET coefficients $\{a_k\}$, the optimum is the solution of a small linear system $K \times K$ (Eq. 15) — closed-form, no iteration needed.
- Multi-base LET (Eq. 23) combines complementary transforms (UWT for piecewise-smooth, BDCT for textured regions).
- Subbands where the Taylor approximation is unreliable are excluded from the linear combination (paper §III-C-D).

---

### Part IV: §IV Simulations / 모의실험

#### 한국어
- **Test images**: Cameraman 256×256, Barbara 512×512, Fluorescent Cells 512×512, Moon 512×512.
- **잡음 레벨**: peak intensity $I_{\rm max} \in \{1, 2, 5, 10, 20, 30, 60, 120\}$, 입력 PSNR 3-26 dB.
- **비교 알고리즘** (paper Table I, 순 포아송):
  - Haar-Fisz (Fryzlewicz-Nason 2004)
  - Anscombe + BLS-GSM (Portilla+ 2003)
  - Platelet (Willett-Nowak 2007)
  - PH-HMT (Lefkimmiatis+ 2009)
  - UWT PURE-LET / BDCT PURE-LET / UWT-BDCT PURE-LET (proposed)
- **결과 (Cameraman 256×256)**:
  - peak 30 (input 18.05 dB):
    - Haar-Fisz: 26.35 dB
    - Anscombe+BLS-GSM: 27.54 dB
    - Platelet: 26.80 dB
    - **UWT PURE-LET: 27.67 dB** (proposed)
    - **UWT/BDCT PURE-LET: 27.91 dB** (proposed, multi-base)
  - peak 1 (input 3.28 dB, 매우 어두움):
    - Haar-Fisz: 19.77 dB
    - Anscombe+BLS-GSM: 14.44 dB (실패: GAT가 너무 어두운 데이터에서 분산 안정화 못함)
    - Platelet: 20.03 dB
    - **UWT/BDCT PURE-LET: 20.48 dB**
- **혼합 P-G (Table II, Cameraman peak 20, σ = 2)**:
  - GAT+BLS-GSM: 25.59 dB
  - **UWT/BDCT PURE-LET: 25.95 dB**
- **시간 (Cameraman peak 20)**:
  - Haar-Fisz (20 cycle-spins): $\sim 7.7$ s
  - Anscombe+BLS-GSM: $\sim 7.7$ s
  - Platelet: $\sim 1300$ s
  - PH-HMT: $\sim 92$ s
  - **PURE-LET (UWT only): 1.3 s** — 가장 빠름.

#### English
- Compared against Haar-Fisz, Anscombe+BLS-GSM, Platelet (Willett-Nowak), and PH-HMT (Lefkimmiatis+) on Cameraman, Barbara, Fluorescent Cells, Moon.
- For pure Poisson at peak 30: UWT/BDCT PURE-LET = 27.91 dB on Cameraman vs Anscombe+BLS-GSM at 27.54 dB.
- At very low peaks (peak 1), Anscombe-based methods *fail* (GAT cannot stabilise variance for sub-photon counts) but PURE-LET still delivers 20.48 dB.
- In mixed Poisson-Gaussian (Table II), UWT/BDCT PURE-LET beats GAT+BLS-GSM by $\sim 0.4$ dB.
- PURE-LET is also computationally efficient: $\sim 1.3$ s for UWT version vs 92 s for PH-HMT.

---

### Part V: §V Real Fluorescence Microscopy / 실 형광현미경 응용

#### 한국어
- **Setup**: 100개의 512×512 형광현미경 영상 (tobacco cell, GFP green + Alexa568 red dyes), peak intensity 매우 낮음.
- **노이즈 추정**: Poisson amplification factor $\alpha$와 read-noise std $\sigma$를 sample-mean / sample-variance regression으로 자동 추정 (Foi et al. 2008).
- **결과**: Fig. 10에서 PURE-LET이 platelet보다 더 적은 artifact, fewer over-smoothing — 실제 microscopy 영상에서 선두 알고리즘.
- **시간**: 512×512 영상 1분 미만; platelet은 1240 s.

#### English
- Real-world test on confocal fluorescence microscopy data of tobacco cells (two-color).
- The Foi et al. 2008 method estimates $\alpha, \sigma$ automatically from local mean-variance regression on the image.
- PURE-LET produces visibly fewer over-smoothing artefacts than platelet at a fraction of the runtime.

---

## 3. Key Takeaways / 핵심 시사점

1. **PURE는 SURE의 직접적 일반화 / PURE is the direct generalisation of SURE** — Stein lemma (가우시안) + Hudson identity (포아송)의 자연스러운 결합. 둘 중 하나만 적용 가능한 시나리오에서는 SURE 또는 PURE 단독으로 환원. 즉 모든 가우시안 잡음 제거 알고리즘 (SureShrink, BLS-GSM, BM3D 등) 의 unbiased risk estimator를 포아송-가우시안 혼합으로 일반화 가능.
   PURE seamlessly generalises SURE by combining Stein's lemma with Hudson's identity. In the limits $\sigma=0$ (pure Poisson) or $x \to \infty$ (effectively Gaussian), it correctly reduces to PURE alone or SURE alone.

2. **LET parameterisation은 quadratic 최적화로 환원시킨다 / LET reduces the optimisation to a quadratic problem** — 추정량을 $\sum a_k f_k$의 *선형 결합*으로 쓰면 PURE는 $a_k$에 대해 *2차* 식. 따라서 closed-form $\mathbf M \mathbf a = \mathbf c$ 해법. 이는 grid-search 또는 stochastic gradient descent 없이 $O(K^3)$ 시간에 최적해 도달.
   Writing the denoiser as a linear combination of $K$ experts makes PURE quadratic in the coefficients, yielding a $K \times K$ linear system whose closed-form solution gives the optimal denoiser without any search or iteration.

3. **신호 의존적 임계 $t_j(\bar w) = \sqrt{\beta_j |\bar w| + \sigma^2}$ / Signal-dependent threshold automatically adapts to local intensity** — 포아송 잡음 분산 $\propto x$이므로 임계도 $\propto \sqrt x$로 스케일링되어야. $\bar w$는 lowpass scaling coefficient (signal estimate); $\sqrt{\beta_j |\bar w|}$는 잡음 std proxy; $\sigma^2$는 read noise. 두 잡음원을 하나의 임계에 합산.
   The threshold scales with local signal magnitude (since Poisson noise std $\propto \sqrt{x}$). The lowpass scaling coefficient $\bar w$ provides the local signal estimate; the threshold combines Poisson and Gaussian noise contributions.

4. **저광자 영역에서 Anscombe-기반 방법이 실패 / Anscombe-based methods fail at very low photon counts** — peak 1에서 Anscombe+BLS-GSM = 14.44 dB (실패), PURE-LET = 20.48 dB (정상 작동). VST는 분산 안정화가 $m \ge 4$에서나 작동하므로, 매우 어두운 영역에서는 직접 PURE 접근이 우월. (그러나 paper #14에서 exact unbiased inverse를 사용한 BM3D+GAT가 이 약점을 해소.)
   At peak intensity 1, Anscombe+BLS-GSM gives only 14.44 dB while PURE-LET gives 20.48 dB. VSTs cannot stabilise variance for sub-photon counts; direct PURE-based methods bypass this limitation. (Paper #14 later closed this gap by introducing the exact unbiased GAT inverse.)

5. **Multi-base LET은 transform 선택 자유도를 활용 / Multi-base LET combines complementary transforms** — UWT (piecewise-smooth) + BDCT (textured) — 각 변환의 강점이 다른 영상 유형에서 발휘. PURE가 두 변환의 가중치를 자동 결정. Barbara peak 30에서 BDCT 단독 27.96 vs UWT/BDCT 28.34 dB — 결합이 $\sim 0.4$ dB 우위.
   Combining UWT (good for piecewise-smooth) and BDCT (good for textures) in a single LET system lets PURE automatically weight each transform per image. Barbara peak 30: BDCT 27.96 dB vs UWT/BDCT 28.34 dB.

6. **1차 Taylor 근사가 핵심 trick / The first-order Taylor approximation is the key practical trick** — 정확한 PURE는 $\mathbf f^-(\mathbf y)$ 평가에 $N$번 알고리즘 재실행이 필요 (영상 1024×1024이면 $\sim 10^6$배). Taylor 1차 근사 $f_n(\mathbf y - \mathbf e_n) \approx f_n(\mathbf y) - \partial f_n/\partial y_n$으로 단 1번 실행으로 PURE 계산 가능. 이는 매끄러운 $f_n$ (Eq. 20의 $\theta_j$는 $C^\infty$)에서 정확.
   Direct evaluation of $\mathbf f^-(\mathbf y)$ requires re-running the denoiser $N$ times. The first-order Taylor approximation reduces this to a single run (paper Eq. 6). Smoothness of $\theta_j$ ensures the approximation is accurate.

7. **Reliability threshold으로 “bad subbands”를 자동 배제 / Reliability check disables unreliable subbands** — Taylor 근사가 매우 어두운 신호 영역에서 부정확. paper §III-C에서 SNR < 40 dB인 subband를 LET에서 제외하면 안정적인 PSNR 결과 (Fig. 7). 평균 신호 에너지 $M_j E_{\rm mean} \ge 5-15$가 reliable cutoff.
   When the Taylor approximation is unreliable (very dark subband, $M_j E_{\rm mean} < 5$), excluding that subband from the LET combination prevents catastrophic PURE-based misoptimisation (Fig. 7).

8. **PURE-LET은 매우 빠르다 / PURE-LET is computationally fast** — 256×256 Cameraman 1.3 s (UWT), 12.2 s (BDCT). Platelet 92 s, PH-HMT은 1240 s — PURE-LET은 30-1000배 빠름. 빠른 이유: closed-form linear system + GPU-friendly transform-domain operations.
   PURE-LET achieves 1.3 s for UWT-only and 12.2 s for BDCT-only on a 256×256 image, while platelet and PH-HMT require minutes. The speed comes from the closed-form linear-system solution and parallelisable transform-domain operations.

---

## 4. Mathematical Summary / 수학적 요약

### 4.1 Noise model / 잡음 모델
$$
\mathbf y = \mathbf z + \mathbf b, \quad z_n \sim \mathcal P(x_n), \quad b_n \sim N(0, \sigma^2), \quad b \perp z
$$
### 4.2 Hudson identity (Property 2) / Hudson 항등식
$$
E_{\mathbf z}\{\mathbf x^T \mathbf f(\mathbf z)\} = E_{\mathbf z}\{\mathbf z^T \mathbf f^-(\mathbf z)\}, \quad \mathbf f^-(\mathbf z)_n = f_n(\mathbf z - \mathbf e_n)
$$
### 4.3 Stein lemma (Property 1) / Stein 보조정리
$$
E_{\mathbf b}\{\mathbf b^T \mathbf f(\mathbf y)\} = \sigma^2 E_{\mathbf b}\{\mathrm{div}\{\mathbf f(\mathbf y)\}\}
$$
### 4.4 PURE for Poisson-Gaussian (Theorem 1, Eq. 2) / 포아송-가우시안 PURE
$$
\boxed{\;\hat\varepsilon = \tfrac{1}{N}\Bigl(\|\mathbf f(\mathbf y)\|^2 - 2\mathbf y^T \mathbf f^-(\mathbf y) + 2\sigma^2\,\mathrm{div}\{\mathbf f^-(\mathbf y)\}\Bigr) + \tfrac{1}{N}(\|\mathbf y\|^2 - \mathbf 1^T \mathbf y) - \sigma^2 \;}
$$
$$
E[\hat\varepsilon] = \tfrac{1}{N} E\|\mathbf f(\mathbf y) - \mathbf x\|^2 = \mathrm{MSE}
$$
### 4.5 Taylor approximation (Eq. 6) / Taylor 근사
$$
f_n(\mathbf y - \mathbf e_n) \approx f_n(\mathbf y) - \frac{\partial f_n(\mathbf y)}{\partial y_n}
$$
$$
\hat\varepsilon \approx \tfrac{1}{N}\bigl(\|\mathbf f(\mathbf y)\|^2 - 2\mathbf y^T(\mathbf f(\mathbf y) - \partial \mathbf f(\mathbf y))\bigr) + \tfrac{1}{N}(2\sigma^2 \mathrm{div}\{\mathbf f - \partial \mathbf f\} + \|\mathbf y\|^2 - \mathbf 1^T \mathbf y) - \sigma^2
$$
### 4.6 Subband-dependent threshold (Eq. 19) / 부대역 의존 임계
$$
t_j(\bar w) = \sqrt{\beta_j |\bar w| + \sigma^2}
$$
- $\beta_j = 2^{-j/2}$ for multiscale, $\beta_j = M^{-1/2}$ for BDCT.
- Combines Poisson noise variance proxy $\beta_j |\bar w|$ and read noise $\sigma^2$.

### 4.7 Smooth thresholding (Eq. 20) / 부드러운 임계화
$$
\theta_j(w, \bar w) = a_{j,1}\,w + a_{j,2}\,w \exp\!\Bigl(-\bigl(\frac{w}{3 t_j(\bar w)}\bigr)^8\Bigr)
$$
- Linear part $a_{j,1} w$: no thresholding.
- Hard-like $a_{j,2} w \exp(-(w/3t)^8)$: smooth approximation to hard thresholding at $\sim 3 t$.

### 4.8 LET formulation (Eq. 14) and linear system (Eq. 15) / LET 수식과 선형방정식
$$
\mathbf f(\mathbf y) = \sum_{k=1}^K a_k\, \mathbf R \boldsymbol\Theta_k(\mathbf w, \bar{\mathbf w}) = \sum_k a_k \mathbf f_k(\mathbf y)
$$
$$
\mathbf M \mathbf a = \mathbf c, \quad [\mathbf M]_{k,l} = \mathbf f_k(\mathbf y)^T \mathbf f_l(\mathbf y), \quad [\mathbf c]_k = \mathbf y^T \mathbf f^-_k(\mathbf y) - \sigma^2\,\mathrm{div}\{\mathbf f^-_k(\mathbf y)\}
$$
### 4.9 Worked numerical example / 수치 예시
For a 64×64 synthetic image with $I_{\rm max} = 10$, $\sigma = 0$ (pure Poisson):
- UWT decomposition with Symmlet-8, 3 levels.
- LET with $K = 6$ (3 levels × 2 nonlinearities); ignoring multi-base.
- PURE estimate $\hat\varepsilon \approx 0.85$ per pixel.
- True MSE $\approx 0.81$ per pixel.
- Difference $\approx 0.04$ ≈ 5% — within sampling fluctuation.
- Resulting PSNR after PURE-LET: $\approx 27$ dB (consistent with paper Cameraman peak 10 = 26.61 dB, depending on image complexity).

### 4.10 Multi-base LET (Eq. 23) / 다중 변환 LET
$$
\mathbf f(\mathbf y) = \sum_{k=1}^{K_1} a_k \mathbf f^{\rm UWT}_k(\mathbf y) + \sum_{k=1}^{K_2} b_k \mathbf f^{\rm BDCT}_k(\mathbf y) + \cdots
$$
### 4.11 Detailed PSNR comparison summary / 상세 PSNR 비교 요약

From Table I (paper) — Cameraman 256×256:

| peak | input | Haar-Fisz | Ans+BLS-GSM | Platelet | UWT P-LET | UWT/BDCT P-LET |
|---|---|---|---|---|---|---|
| 1   | 3.28  | 19.77 | 14.44 | 20.03 | 20.44 | **20.48** |
| 5   | 14.27 | 22.55 | 24.63 | 23.56 | 23.50 | 23.65 |
| 30  | 18.05 | 26.35 | 27.54 | 26.80 | 27.67 | **27.91** |
| 120 | 24.08 | 29.73 | 30.85 | 30.54 | 31.03 | **31.35** |

UWT/BDCT PURE-LET is *consistently top* across the full intensity range.

From Table II — mixed Poisson-Gaussian (Cameraman, σ=peak/10):

| peak | σ | input | GAT+BLS-GSM | UWT/BDCT PURE-LET |
|---|---|---|---|---|
| 1   | 0.1 | 3.19 | (not reported) | **20.44** |
| 10  | 1.0 | 12.45 | 24.43 | **24.74** |
| 30  | 3.0 | 16.29 | 26.19 | **26.53** |
| 120 | 12.0 | 24.08 | 27.56 | **27.92** |

This is paper #13's strongest empirical claim — best in class for mixed Poisson-Gaussian. Paper #14 partially overturns this by using GAT + exact unbiased inverse + BM3D.

### 4.12 Reliability check (Eq. for $T$) / 신뢰성 확인

Paper §III-D: subbands $\mathbf f_{j,2}(\mathbf y)$ where $M_j E_{\rm mean} < T$ (with $T \in [5, 15]$) are excluded from the LET. Reason: the first-order Taylor expansion of $f_n^-$ is unreliable for very dark subbands. Including them causes PURE to misoptimise (Fig. 7 of paper).

Empirically, $T = 10$ gives near-optimal results for all tested images.

### 4.13 Hudson identity proof sketch / Hudson 항등식 증명 스케치

For Poisson $z \sim \mathcal P(x)$:
$$
E[x \cdot f(z)] = \sum_{k=0}^\infty x \cdot f(k) \cdot \frac{x^k e^{-x}}{k!} = \sum_{k=0}^\infty f(k) \cdot \frac{x^{k+1} e^{-x}}{k!}
$$
Substitute $j = k + 1$:
$$
= \sum_{j=1}^\infty f(j-1) \cdot \frac{x^j e^{-x}}{(j-1)!} = \sum_{j=1}^\infty j \cdot f(j-1) \cdot \frac{x^j e^{-x}}{j!}
$$
$$
= \sum_{j=0}^\infty j \cdot f(j-1) \cdot p(j|x) = E[z \cdot f(z-1)] = E[z \cdot f^-(z)]
$$
QED for the scalar case. The vector form (Property 2 of paper) follows by applying coordinate-wise.

---

## 5. Paper in the Arc of History / 역사적 맥락의 타임라인

```
1948 ─── Anscombe — Poisson VST (paper #11)
1981 ─── Stein — SURE for Gaussian noise estimation
1978 ─── Hudson — natural identity for exponential families
                  ↳ Poisson analogue of Stein lemma
1994 ─── Donoho-Johnstone — VisuShrink (paper #1)
1995 ─── Donoho-Johnstone — SureShrink (paper #2)
1995 ─── Murtagh-Starck-Bijaoui — generalised Anscombe transform (GAT)
2000 ─── Kolaczyk — wavelet shrinkage estimation with corrected thresholds
2003 ─── Portilla-Strela-Wainwright-Simoncelli — BLS-GSM (Bayesian, Gaussian)
2004 ─── Fryzlewicz-Nason — Haar-Fisz algorithm (multiscale Anscombe)
2004 ─── Sardy-Antoniadis-Tseng — exponential-family wavelet shrinkage
2007 ─── Willett-Nowak — Platelet (penalised likelihood, polynomial fits)
2009 ─── Lefkimmiatis+ — Poisson-Haar HMT (Bayesian multiscale)
2010 ─── Luisier-Vonesch-Blu-Unser — fast PURE-LET in unnormalised Haar
2010 ─── Deledalle-Tupin-Denis — Poisson NL Means (paper #12, parallel approach)
2011 ★★ LUISIER-BLU-UNSER: PURE-LET for mixed Poisson-Gaussian noise (this paper)
                  ↳ Theorem 1 PURE for Poisson-Gaussian, LET, multi-base
2013 ─── Mäkitalo-Foi — exact unbiased inverse of GAT (paper #14)
2017+ ── Deep-learning Poisson denoisers: DnCNN, FFDNet (gather all noise)
                  ↳ PURE-style training losses still relevant.
```

이 논문은 **Stein lemma + Hudson identity의 결합으로 PURE를 정립**하고, **closed-form LET 최적화**를 도입한 변환영역 잡음제거의 새 표준이다. paper #14는 GAT의 exact inverse를 통해 본 paper의 “VST 우회” 주장을 *부분적으로* 무력화하지만, PURE-LET의 framework와 unbiased risk 아이디어는 후일 self-supervised 학습 (Noise2Self 등)의 핵심으로 남는다.

This paper **establishes PURE via Stein + Hudson** and introduces **closed-form LET optimisation** as a new standard for transform-domain denoising. Paper #14 partially undercuts its "bypass-VST" claim by giving GAT a proper inverse, but PURE-LET's framework and unbiased risk ideas live on in self-supervised learning (Noise2Self et al.).

---

## 6. Connections to Other Papers / 다른 논문과의 연결

| Paper / 논문 | Connection / 연결 | Relevance / 관련성 |
|---|---|---|
| **Anscombe (1948)** *Biometrika* (#11) | Forward VST | Direct competitor of PURE-LET's "bypass VST" claim. |
| **Donoho-Johnstone (1994, 1995)** *Biometrika*, *JASA* (#1, #2) | SureShrink ancestry | PURE-LET is the Poisson-Gaussian generalisation of SureShrink. |
| **Stein (1981)** *Annals of Statistics* | SURE, Stein lemma | Property 1 of paper. |
| **Hudson (1978)** *Annals of Statistics* | Natural identity, Hudson identity | Property 2 of paper, Poisson analogue of Stein. |
| **Portilla-Strela-Wainwright-Simoncelli (2003)** *IEEE TIP* | BLS-GSM | Strong baseline (Anscombe+BLS-GSM in Tables I-II). |
| **Murtagh-Starck-Bijaoui (1995)** *A&A Suppl.* | Generalised Anscombe transform | Used as preprocessing in baselines. |
| **Willett-Nowak (2007)** *IEEE TIP* | Platelet, penalised likelihood | Direct competitor; PURE-LET is faster and competitive in PSNR. |
| **Lefkimmiatis+ (2009)** *IEEE TIP* | PH-HMT, Poisson-Haar HMT | Bayesian competitor; PURE-LET is non-Bayesian alternative. |
| **Deledalle-Tupin-Denis (2010)** *ICIP* (#12) | Poisson NL Means | Same year, parallel approach using NLM + PURE. |
| **Mäkitalo-Foi (2013)** *IEEE TIP* (#14) | Exact unbiased inverse of GAT | Closes the gap that paper claims VST cannot handle (low photons). |
| **Luisier-Vonesch-Blu-Unser (2010)** *Signal Processing* | PURE-LET for Haar Poisson | Direct precursor; this paper extends to undecimated transforms + Poisson-Gaussian. |
| **Foi+ (2008)** *IEEE TIP* | Practical Poisson-Gaussian noise modeling | Used to estimate $\alpha, \sigma$ from raw CCD data in §V. |

---

## 7. References / 참고문헌

- Luisier, F., Blu, T., & Unser, M., "Image denoising in mixed Poisson-Gaussian noise", *IEEE Trans. Image Process.*, 20(3), 696–708 (2011). [DOI: 10.1109/TIP.2010.2073477]
- Stein, C. M., "Estimation of the mean of a multivariate normal distribution", *Annals of Statistics*, 9(6), 1135–1151 (1981).
- Hudson, M., "A natural identity for exponential families with applications in multiparameter estimation", *Annals of Statistics*, 6(3), 473–484 (1978).
- Anscombe, F. J., "The transformation of Poisson, binomial and negative-binomial data", *Biometrika*, 35(3-4), 246–254 (1948).
- Donoho, D. L., & Johnstone, I. M., "Adapting to unknown smoothness via wavelet shrinkage", *J. American Statistical Association*, 90(432), 1200–1224 (1995).
- Portilla, J., Strela, V., Wainwright, M. J., & Simoncelli, E. P., "Image denoising using scale mixtures of Gaussians in the wavelet domain", *IEEE Trans. Image Process.*, 12(11), 1338–1351 (2003).
- Willett, R. M., & Nowak, R. D., "Multiscale Poisson intensity and density estimation", *IEEE Trans. Inf. Theory*, 53(9), 3171–3187 (2007).
- Fryzlewicz, P., & Nason, G. P., "A Haar-Fisz algorithm for Poisson intensity estimation", *J. Comput. Graph. Stat.*, 13, 621–638 (2004).
- Murtagh, F., Starck, J.-L., & Bijaoui, A., "Image restoration with noise suppression using a multiresolution support", *Astron. Astrophys. Suppl.*, 112, 179–189 (1995).
- Foi, A., Trimeche, M., Katkovnik, V., & Egiazarian, K., "Practical Poissonian-Gaussian noise modeling and fitting for single-image raw data", *IEEE Trans. Image Process.*, 17(10), 1737–1754 (2008).
- Mäkitalo, M., & Foi, A., "Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise", *IEEE Trans. Image Process.*, 22(1), 91–103 (2013).
- Deledalle, C.-A., Tupin, F., & Denis, L., "Poisson NL means: Unsupervised non local means for Poisson noise", *Proc. IEEE ICIP*, pp. 801–804 (2010).
