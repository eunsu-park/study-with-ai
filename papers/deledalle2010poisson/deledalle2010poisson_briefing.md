---
title: "Pre-Reading Briefing: Poisson NL Means — Unsupervised Non-Local Means for Poisson Noise"
paper_id: "12_deledalle_2010"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Poisson NL Means (Deledalle+ 2010): Pre-Reading Briefing / 사전 읽기 브리핑

**Paper**: Deledalle, C.-A., Tupin, F., & Denis, L., "Poisson NL means: unsupervised non local means for Poisson noise", *Proc. IEEE ICIP* 2010, 801–804. [DOI: 10.1109/ICIP.2010.5653394]
**Author(s)**: Charles-Alban Deledalle, Florence Tupin, Loïc Denis
**Year**: 2010

---

## 1. 핵심 기여 / Core Contribution

### 한국어
이 논문은 NL Means(Buades+ 2005, 논문 #4)를 Poisson 잡음에 맞춰 재설계하고 — 단순히 패치 거리만 Poisson용으로 바꾸는 게 아니라 **잡음 영상과 사전 추정 영상을 동시에 사용하는 refined NL Means** ($\alpha, \beta$ 두 파라미터) 구조를 도입 — *비지도적*으로 두 파라미터를 결정한다. 세 핵심 요소: (i) **Poisson 패치 유사도**: 잡음 패치에는 *generalised likelihood-ratio test* (GLRT) $f_L(k_1, k_2) = k_1\log k_1 + k_2\log k_2 - (k_1+k_2)\log\!\bigl(\tfrac{k_1+k_2}{2}\bigr)$, 사전 추정 패치에는 *symmetric KL divergence*. (ii) **PURE for NL Means**: Chen(1975)의 Poisson identity $E[\lambda h(k)] = E[k\,h(\bar k)]$ (Stein lemma의 Poisson 대응)로 NL Means의 MSE를 $\lambda$를 모른 채 비편향 추정. (iii) **Newton 최적화**: 두 파라미터 $(\alpha, \beta)$를 closed-form 1차/2차 도함수로 6–14 iteration 안에 PURE 최소점 수렴. 결과: Peppers/Cameraman peak 1–30에서 PURE-LET, classical NLM, refined NLM 모두를 0.1–1.5 dB 능가, 특히 매우 낮은 SNR(input PSNR 3 dB)에서 가장 큰 개선.

### English
This paper adapts NL Means (Buades+ 2005, paper #4) to Poisson noise — but goes beyond replacing the patch distance: it adopts a **refined NL Means** scheme that uses *both* the noisy image and a pre-filter estimate (with two parameters $\alpha, \beta$) and tunes them *unsupervisedly* via PURE. Three ingredients: (i) **Poisson patch-similarity**: GLRT $f_L(k_1, k_2)$ on noisy patches, symmetric KL divergence on pre-estimates. (ii) **PURE for NL Means**: Chen's (1975) Poisson identity $E[\lambda h(k)] = E[k\,h(\bar k)]$ — the Poisson analogue of Stein's lemma — yields an unbiased MSE estimator from $k$ alone. (iii) **Newton optimisation**: closed-form Hessian and gradient drive $(\alpha, \beta)$ to PURE's minimum in 6–14 iterations. Results on Peppers/Cameraman across peak 1–30 beat PURE-LET, classical NLM, and refined NLM by 0.1–1.5 dB, with the biggest gains at very low input SNR (≈ 3 dB).

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
2010년 시점, Poisson 잡음 영상 처리에는 두 갈래가 경쟁하고 있었다. 한쪽은 *VST 경로* (논문 #11 Anscombe + Gaussian denoiser + inverse VST) — 단순하지만 매우 낮은 광자수 영역에서 부정확. 다른 쪽은 *직접 likelihood 경로* — Poisson PMF를 직접 활용하는 알고리즘. 두 번째 진영의 핵심 도구가 SURE의 Poisson 대응 PURE이었고, Luisier-Vonesch-Blu-Unser(2010)의 fast PURE-LET (논문 #13의 conference version)가 *wavelet shrinkage*에 PURE를 적용한 첫 결정판이었다. 본 논문은 같은 PURE 원리를 *NL Means*에 처음 적용 — patch-based denoising의 직접적 Poisson 확장. 또한 같은 저자들의 PPB(Probabilistic Patch-Based, 2009) 작업의 자연스러운 후속이다.

#### English
By 2010 Poisson-noise image processing had two competing routes. The *VST route* (Anscombe paper #11 + Gaussian denoiser + inverse VST) was simple but inaccurate at very low photon counts. The *direct-likelihood route* used Poisson PMF natively. The key tool for the latter was PURE — the Poisson analogue of SURE — and Luisier-Vonesch-Blu-Unser's (2010) fast PURE-LET (conference version of paper #13) was its definitive realisation for *wavelet shrinkage*. This paper applies the same PURE principle to *NL Means* for the first time — the direct Poisson extension of patch-based denoising. It is also a natural follow-up to the same authors' PPB (Probabilistic Patch-Based, 2009) work.

### 타임라인 / Timeline

```
1948 ─── Anscombe — VST for Poisson (paper #11)
1975 ─── Chen — Poisson approximation identity (Eq. 8 origin)
1981 ─── Stein — SURE for Gaussian
1995 ─── Donoho-Johnstone — SureShrink (Gaussian wavelet)
2005 ─── Buades-Coll-Morel — NLM (paper #4)
2009 ─── Deledalle-Denis-Tupin — PPB (precursor to this paper)
2009 ─── Van De Ville-Kocher — SURE-NLM (Gaussian)
2010 ★★ DELEDALLE-TUPIN-DENIS — Poisson NL Means (THIS PAPER)
2010 ─── Luisier-Vonesch-Blu-Unser — fast PURE-LET (precursor to #13)
2011 ─── Mäkitalo-Foi — exact unbiased inverse of Anscombe
2013 ─── Mäkitalo-Foi — exact inverse of GAT (paper #14)
```

---

## 3. 필요한 배경 지식 / Prerequisites

### 한국어
- **Non-Local Means** (논문 #4): Eq. 1–2의 weighted patch averaging, Gaussian patch distance
- **Poisson PMF**: $p(k|\lambda) = \lambda^k e^{-\lambda}/k!$, $E[k] = \mathrm{var}(k) = \lambda$
- **Likelihood-ratio test**: 두 가설 $H_0$ vs $H_1$의 max-likelihood 비교
- **Maximum likelihood estimation**: Poisson rate $\hat\lambda_{\rm MLE}$
- **Kullback-Leibler divergence**: $D_{KL}(P\|Q)$, symmetric form
- **Stein's lemma & SURE**: Gaussian unbiased risk estimation의 원리
- **Newton's method**: $x^{(n+1)} = x^{(n)} - H^{-1}\nabla f$, 1차/2차 미분
- **Anscombe transform** (논문 #11): VST 우회 ([대안] 경로)

### English
- **Non-Local Means** (paper #4) — Eq. 1–2 weighted patch averaging
- **Poisson PMF**: $p(k|\lambda) = \lambda^k e^{-\lambda}/k!$
- **Likelihood-ratio test**: comparing $H_0$ vs $H_1$ via max-likelihood
- **Maximum-likelihood estimation** for Poisson rate
- **Kullback-Leibler divergence** (symmetric form)
- **Stein's lemma & SURE** — Gaussian unbiased risk estimation
- **Newton's method** with closed-form gradient and Hessian
- **Anscombe transform** (paper #11) — alternative VST route bypassed here

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| Refined NL Means / 개선된 NLM | 잡음 영상의 거리 $F$와 사전 추정 영상의 거리 $G$를 *동시에* 가중치에 사용 ($w \propto \exp(-F/\alpha - G/\beta)$). / Uses both noisy-image and pre-estimate distances in the weights. |
| GLRT distance $f_L$ / GLRT 거리 | Poisson 두 패치가 같은 $\lambda$에서 왔다는 가설의 -log likelihood ratio. / Negative-log likelihood ratio for "same Poisson rate" hypothesis. |
| Symmetric KL / 대칭 KL | 사전 추정 패치 비교용: $(\hat\theta_1 - \hat\theta_2)\log(\hat\theta_1/\hat\theta_2)$. / Symmetric KL divergence between two intensity estimates. |
| Pre-estimate $\hat\theta$ / 사전 추정 | 거친 사전 평활(예: 13×13 disk 평균)로 얻은 첫 번째 영상 추정. / Coarse pre-smoothed image (e.g., 13×13 disk average). |
| Chen's identity / Chen 항등식 | $E[\lambda h(k)] = E[k\,h(\bar k)]$, $\bar k_t = k_t - \mathbf 1\{t = s\}$. / Poisson Stein's lemma analogue. |
| PURE / 포아송 비편향 위험 추정 | Poisson Unbiased Risk Estimate — $\lambda$를 모르고 MSE 비편향 추정. / Unbiased MSE estimate without knowing $\lambda$. |
| $\bar k$ trick | "test 픽셀에서만 $-1$"하는 단순 변형 — 비용 거의 없음. / Subtract 1 only at the test pixel — negligible overhead. |
| Filtering parameters $(\alpha, \beta)$ / 필터링 파라미터 | $\alpha$: 잡음 영상 신뢰도, $\beta$: 사전 추정 신뢰도. / $\alpha$: trust in noisy image; $\beta$: trust in pre-estimate. |
| Newton optimisation / Newton 최적화 | $(\alpha, \beta)^{(n+1)} = (\alpha,\beta)^{(n)} - H^{-1}\nabla R$, 6–14 iter. / 2D Newton update on PURE. |
| PURE-LET (논문 #13) | Poisson wavelet shrinkage via PURE — 본 논문의 직접 비교 대상. / Direct competitor: Poisson wavelet shrinkage via PURE. |
| Search window / 검색 창 | 21×21 영역에서 유사 패치 찾음 (보통). / Region (≈21×21) for similar-patch search. |
| Patch size / 패치 크기 | 7×7 (보통). / Typically 7×7. |

---

## 5. 수식 미리보기 / Equations Preview

**관측 모델 / Observation model**:
$$
k_s \sim \mathrm{Poisson}(\lambda_s), \quad p(k|\lambda) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

**Refined NL Means (Eq. 3)**:
$$
\hat\lambda_s = \frac{\sum_t w_{s,t}\,k_t}{\sum_t w_{s,t}}, \quad w_{s,t} = \exp\!\Bigl(-\frac{F_{s,t}}{\alpha} - \frac{G_{s,t}}{\beta}\Bigr)
$$

**Poisson patch similarity (Eq. 5)**:
$$
f_L(k_1, k_2) = k_1 \log k_1 + k_2 \log k_2 - (k_1 + k_2)\,\log\!\Bigl(\frac{k_1 + k_2}{2}\Bigr)
$$

**Chen's identity (Eq. 8)**:
$$
E[\lambda_s\,h(k)_s] = E[k_s\,h(\bar k)_s], \quad \bar k_t = \begin{cases} k_t & t \ne s \\ k_t - 1 & t = s \end{cases}
$$

**PURE for NL Means (Eq. 9)**:
$$
R(\hat\lambda) = \frac{1}{N}\sum_s \bigl(\lambda^2_s + \hat\lambda^2_s - 2\,k_s\,\bar\lambda_s\bigr)
$$
where $\bar\lambda_s$ applies NL Means to $\bar k$ in place of $k$.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **§1 (도입)**: classical NLM (Eq. 1–2)와 refined NLM (Eq. 3)의 차이를 명확히. 두 파라미터 $(\alpha, \beta)$가 본 논문이 풀고자 하는 핵심 미해결 문제임을 인지.
- **§2 (패치 유사도)**: Eq. 5의 $f_L$ 유도를 따라가 보기 — Poisson PMF 두 hypothesis의 likelihood ratio. *왜 $k_1 = k_2 = 0$일 때 0인가*를 확인 (convention $0\log 0 = 0$). KL divergence가 *왜 symmetric form*인지도 답해보기 (단방향 KL은 비대칭이라 distance 역할 못함).
- **§3 (PURE)**: *논문의 수학적 핵심*. Eq. 7의 MSE 분해 → Eq. 8의 Chen identity → Eq. 9의 PURE. Chen identity가 *왜* Stein lemma의 Poisson 대응인지 직관 — Stein은 Gaussian의 IBP, Chen은 Poisson factorial moment identity.
- **§3 ($\bar k$ trick)**: $\bar k_t = k_t - 1$ ($t = s$)이 추가 비용이 거의 없는 이유 — 패치 거리 $F_{s,t}$의 차이가 한두 항뿐.
- **§3 (Newton)**: closed-form 1차/2차 미분 (paper Eq. 10 직후). 코드 작성 시 *모든 미분이 NL Means 한 번 도는 비용 안에 계산*된다는 점 인지.
- **§4 (실험)**: Fig. 1의 PURE vs MSE 곡선 일치를 확인 (PURE의 *비편향성 검증*). Table 1의 PSNR을 PURE-LET과 비교.
- **흔한 오해**: PURE는 *MSE를 알기 위한 도구*가 아니라 *MSE를 최적화하는 surrogate*. PURE 값 자체가 작을 필요 없음 — 그 *최소점*이 의미.
- **놓치기 쉬운 점**: $\hat\theta$가 *너무 noisy하면* PURE 가정이 깨짐 (Eq. 10의 독립성 가정). 첫 통과에 *충분히 smooth*한 MA filter 사용 권장.

### English
- **§1 (introduction)**: clarify the distinction between classical NLM (Eq. 1–2) and refined NLM (Eq. 3). Recognise that the two parameters $(\alpha, \beta)$ are the open problem this paper solves.
- **§2 (patch similarity)**: walk through the derivation of $f_L$ (Eq. 5) — the negative log-likelihood ratio between two Poisson hypotheses. Verify *why $k_1 = k_2 = 0$ gives $f_L = 0$* (convention $0\log 0 = 0$). Ask *why the symmetric form* of KL — one-sided KL is not a metric.
- **§3 (PURE)**: *the mathematical heart*. Eq. 7 MSE decomposition → Eq. 8 Chen identity → Eq. 9 PURE. The intuition: Stein is Gaussian integration-by-parts; Chen is the Poisson factorial-moment identity.
- **§3 ($\bar k$ trick)**: $\bar k_t = k_t - 1$ at $t = s$ has negligible cost — patch distances $F_{s,t}$ change in only one or two terms.
- **§3 (Newton)**: closed-form gradient and Hessian (paper just after Eq. 10). Note that every derivative is computable within one NL Means pass.
- **§4 (experiments)**: confirm Fig. 1's PURE vs MSE agreement (validation of unbiasedness). Compare Table 1 PSNR to PURE-LET.
- **Pitfall**: PURE is *not* a tool to know MSE — it is a *surrogate to optimise* it. PURE's value need not be small; its *minimiser* is what matters.
- **Easy to miss**: if $\hat\theta$ is *too noisy*, the PURE derivation breaks (Eq. 10's independence assumption). Use a *sufficiently smooth* MA filter for the initial pre-estimate.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
이 논문은 (i) **NLM의 비-Gaussian 잡음 모델로의 결정적 확장** — 이후 SAR, X-ray, 형광현미경, photon-counting 영상에서 표준 도구; (ii) **PURE가 wavelet shrinkage(PURE-LET, 논문 #13) 외의 추정자에도 적용 가능함을 입증** — 이후 PURE/SURE 스타일 risk estimator가 self-supervised denoising(Noise2Noise, Noise2Self, Stein-PnP)의 표준 도구로 자리잡음; (iii) **VST 경로의 대안적 정당화** — 매우 낮은 광자수 영역(input PSNR 3 dB)에서 likelihood-based 접근이 VST + Gaussian denoiser를 능가함을 명확히 보임. 후속 작업: Deledalle+의 PPB 진화, Lebrun+ NL-Bayes, Manjón+ PRI-NLM3D(MR), 그리고 최근 SAR despeckling/X-ray denoising/저조도 photography의 deep methods 다수가 PURE 또는 likelihood-based patch distance를 hybrid 형태로 사용. 또한 fluorescence microscopy의 *count-domain* 영상 처리 라이브러리(scikit-image의 Poisson denoising 모듈, PySAP-MRI 등)에서 직접 구현되어 있다.

### English
This paper is (i) **the decisive extension of NLM beyond Gaussian noise** — a standard tool ever since in SAR, X-ray, fluorescence microscopy, and photon-counting imaging; (ii) **a demonstration that PURE applies beyond wavelet shrinkage** — establishing PURE/SURE-style risk estimators as a standard tool in self-supervised denoising (Noise2Noise, Noise2Self, Stein-PnP); (iii) **an alternative justification of the likelihood-based route** — clearly showing that at very low photon counts (input PSNR ≈ 3 dB) likelihood-based approaches surpass VST + Gaussian denoising. Direct descendants: Deledalle+'s evolved PPB framework, Lebrun+ NL-Bayes, Manjón+ PRI-NLM3D (MR), and many modern SAR-despeckling / X-ray-denoising / low-light-photography deep methods that hybridise PURE or likelihood-based patch distances. Implementations appear in count-domain imaging libraries (scikit-image's Poisson denoising module, PySAP-MRI, etc.).

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
