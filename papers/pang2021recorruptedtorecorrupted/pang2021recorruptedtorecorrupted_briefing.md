---
title: "Pre-Reading Briefing: Recorrupted-to-Recorrupted (R2R)"
paper_id: "21_pang_2021"
topic: Low_SNR_Imaging
date: 2026-05-06
type: briefing
---

# Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising / 사전 읽기 브리핑

**Paper**: Pang, T., Zheng, H., Quan, Y., Ji, H. "Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising". *IEEE/CVF CVPR 2021*, pp. 2043–2052. DOI: 10.1109/CVPR46437.2021.00208.
**Authors**: Tongyao Pang, Huan Zheng, Yuhui Quan, Hui Ji
**Year**: 2021

---

## 1. 핵심 기여 / Core Contribution

### 한국어
R2R은 **단 하나의 잡음 영상**에서 Noise2Noise(N2N)와 등가인 학습을 가능하게 하는 **재손상(recorruption) 변환**을 제안한다. 잡음 분포의 covariance $\Sigma$를 알고 있을 때 보조 잡음 $z\sim\mathcal N(0,I)$를 샘플링해 두 개의 합성 잡음 영상을 만든다: $y_1 = y + \alpha D^T z$, $y_2 = y - \alpha^{-1} D z$ ($D D^T = D^T D = \Sigma$). Proposition 1로 (i) 두 합성 잡음의 *평균이 0*이고 (ii) *상관도 0*임을 보여, $(y_1, y_2)$가 N2N의 두 독립 noisy 캡처와 통계적으로 동등함을 증명한다. 따라서 N2V의 blind-spot 정보 손실 없이 N2N 정리를 그대로 사용할 수 있다. 합성 가우시안/푸아송, 실세계 SIDD raw-RGB 모두에서 자기지도 SOTA를 달성하며 supervised N2C와 0.1~0.3 dB 격차로 좁혔다.

### English
R2R introduces a **recorruption transform** that lets a network be trained à la Noise2Noise from a *single* noisy observation $y = x + n$. Given known noise covariance $\Sigma$, draw $z\sim\mathcal N(0,I)$ and form the synthetic pair $y_1 = y + \alpha D^T z$, $y_2 = y - \alpha^{-1} D z$ where $D D^T = D^T D = \Sigma$. Proposition 1 proves the engineered pair satisfies (i) zero conditional mean of both residuals and (ii) zero conditional cross-correlation — making it statistically interchangeable with the two independent noisy captures N2N requires. Crucially, no blind spot is needed: every input pixel carries information. R2R achieves self-supervised SOTA across synthetic Gaussian/Poisson noise and real raw-RGB SIDD data, sitting within 0.1–0.3 dB of fully-supervised N2C.

---

## 2. 역사적 맥락 / Historical Context

### 시대 배경 / The Setting

#### 한국어
2020년까지 단일 영상 self-supervised denoising은 두 갈래로 나뉘어 있었다: (a) **blind-spot 계열** (N2V/N2S/Self2Self) — 단일 영상 가능하지만 정보 손실, (b) **noise-injection 계열** (Noisier2Noise) — 비대칭 보정으로 인한 잔여 bias. R2R은 이 두 한계를 동시에 깨고 *covariance-matched 양방향 corruption*으로 N2N의 통계적 보장을 직접 단일 영상에 옮긴다. Anscombe 변환을 통해 Poisson도 처리한다.

#### English
By 2020, single-image self-supervised denoising had split into two camps: (a) the **blind-spot family** (N2V, N2S, Self2Self) — single image but information loss, (b) the **noise-injection family** (Noisier2Noise) — asymmetric corruption with residual bias. R2R breaks both ceilings simultaneously by introducing a *covariance-matched two-sided corruption* that transfers N2N's full statistical guarantee to a single image. The Anscombe transform extends the framework to Poisson noise.

### 타임라인 / Timeline

```
2007  Dabov BM3D — non-learning baseline
2017  Zhang DnCNN — supervised denoising baseline
2018  Lehtinen N2N (#16) — noisy/noisy training
2019  Krull N2V (#17) / Batson N2S (#18) — blind-spot single-image
2020  Quan Self2Self (#19) — dropout ensemble, beats BM3D
2020  Moran Noisier2Noise — pre-corrupt single image (asymmetric, biased)
2021 ★ Pang R2R — covariance-matched recorruption, N2N-equivalent from one image
2021  Huang Neighbor2Neighbor (#20) — concurrent: sub-sampling pair (no noise model)
2022  Wang Blind2Unblind (#22) — visible blind spots
```

---

## 3. 필요한 배경 지식 / Prerequisites

#### 한국어
- **Noise2Noise 정리** (Lehtinen 2018, 논문 #16) — R2R이 직접 상속하는 statistical foundation
- **다변량 가우시안**, covariance matrix $\Sigma$, square-root 분해 ($D D^T=\Sigma$)
- **Conditional independence**, conditional mean/variance
- **Anscombe 변환** (논문 #11) — Poisson → 근사 Gaussian (signal-dependent → signal-independent)
- **Monte-Carlo averaging** 추론 — $K=50$ 평균
- **MAD/0.6745**으로 noise std 추정 (선택적)
- **DnCNN, U-Net** 백본 아키텍처
- **단일 영상 self-sup 맥락**: N2V(#17), N2S(#18), S2S(#19), Noisier2Noise

#### English
- The Noise2Noise theorem (Lehtinen 2018, #16) — R2R's direct statistical foundation.
- Multivariate Gaussian, covariance $\Sigma$, square-root factorisations ($D D^T=\Sigma$).
- Conditional independence, conditional mean and variance.
- Anscombe variance-stabilising transform (#11) for Poisson statistics.
- Monte-Carlo averaging for inference ($K=50$).
- Noise-std estimation via MAD/0.6745 (optional).
- DnCNN and U-Net backbones.
- Prior single-image self-sup methods: N2V (#17), N2S (#18), Self2Self (#19), Noisier2Noise.

---

## 4. 핵심 용어 / Key Vocabulary

| 용어 / Term | 설명 / Explanation |
|---|---|
| **Recorruption transform** | $y_1=y+\alpha D^T z$, $y_2=y-\alpha^{-1} D z$ 변환 / Two-sided injection of auxiliary noise. |
| **Auxiliary noise $z$** | $\mathcal N(0,I)$에서 샘플 / Drawn from standard Gaussian, independent of $n$. |
| **Square-root factor $D$** | $D D^T = \Sigma$를 만족하는 행렬 / Any matrix with $D D^T = \Sigma$. |
| **$\alpha$ trade-off slider** | $y_1, y_2$의 SNR 분배 통제 (default 1) / Controls SNR partition between $y_1$ and $y_2$. |
| **Proposition 1** | $\mathbb E[n_1]=\mathbb E[n_2]=0$, $\mathbb E[n_1 n_2^T]=0$ / Zero-mean and uncorrelated property of the synthesised pair. |
| **Anscombe transform** | Poisson → 근사 Gaussian variance stabilisation (논문 #11) / Variance-stabilising transform for Poisson. |
| **Noise covariance $\Sigma$** | 사전에 알려진 잡음 통계 / Pre-known noise covariance — the only assumption R2R needs. |
| **$K$ Monte-Carlo samples** | 추론 시 평균하는 forward 횟수 (default 50) / Number of inference forward passes averaged. |
| **N2N equivalence** | 단일 영상에서 N2N 학습 정리 그대로 사용 가능 / R2R inherits N2N's exact loss-equivalence theorem. |
| **Blind-spot avoidance** | 입력 픽셀 가리지 않음 — 정보 손실 없음 / No pixel masking — full input information preserved. |
| **Backbone-agnostic** | DnCNN, U-Net, NAFNet 등 자유 / Any denoising backbone slots in. |
| **SIDD raw-RGB** | 실세계 smartphone CMOS 잡음 벤치마크 / Real-world smartphone-CMOS noise benchmark. |

---

## 5. 수식 미리보기 / Equations Preview

### 한국어
**재손상 변환 (Eqs. 1-2)** — 두 합성 잡음 영상 생성:

$$
y_1 = y + \alpha D^T z, \qquad y_2 = y - \alpha^{-1} D z, \qquad z\sim\mathcal N(0, I)
$$

**Proposition 1** — 평균 0 + 무상관:

$$
\mathbb E[n_1\mid x] = \mathbb E[n_2\mid x] = 0, \qquad \mathbb E[n_1 n_2^T \mid x] = \Sigma - D^T D = 0
$$

**상속된 N2N 정리** — 합성 쌍이 N2N과 등가:

$$
\arg\min_\theta \mathbb E\,\|f_\theta(y_1) - y_2\|^2 = \arg\min_\theta \mathbb E\,\|f_\theta(y_1) - x\|^2
$$

**추론 Monte-Carlo 평균** — $K=50$ 평균:

$$
\hat x(y) = \frac{1}{K}\sum_{k=1}^K f_\theta\!\big(y + \alpha D^T z^{(k)}\big), \quad z^{(k)}\overset{iid}{\sim}\mathcal N(0,I)
$$

### English
The recorruption transform manufactures a synthetic noisy/noisy pair from one observation. Proposition 1 verifies the two key statistical properties — zero conditional mean and zero cross-correlation — letting N2N's theorem apply verbatim to single-image data. The minimiser of the synthetic loss equals the minimiser of the supervised MSE. At inference, $K=50$ Monte-Carlo draws of $z^{(k)}$ feed the network with inputs that match the *training* distribution (otherwise feeding bare $y$ would mismatch), and the average reduces variance ~$1/K$.

---

## 6. 읽기 가이드 / Reading Guide

### 한국어
- **꼭 정독할 부분**: §3 (재손상 변환의 정의와 Proposition 1 증명), §3.3 (Poisson/Anscombe 확장), §3.4 (추론 시 MC 평균이 *왜 필요한가*), §4 (실험 결과, 특히 SIDD).
- **빠르게 훑을 부분**: §2 related work, §5 implementation 세부.
- **흔한 걸림돌 / Common stumbling blocks**:
  - "왜 추론 시 bare $y$를 그대로 넣지 못하는가?" — 학습 분포가 $y_1=y+\alpha D^T z$ 형태라 mismatch 발생.
  - "$\alpha$의 의미": $\text{Var}(n_1)=(1+\alpha^2)\sigma^2$, $\text{Var}(n_2)=(1+\alpha^{-2})\sigma^2$ — 두 영상의 SNR 분배.
  - "$D D^T = \Sigma$만 충분한가, $D^T D = \Sigma$가 추가로 필요한가": 두 조건 모두 필요 (proposition 1의 cross-correlation 항이 사라지려면).
  - "Anscombe + R2R": Poisson 잡음을 근사 Gaussian으로 환원한 후 R2R을 적용, 추론 후 inverse Anscombe 적용.
- 동반 자료: Lehtinen N2N 원논문, Anscombe(#11) / Mäkitalo-Foi(#14) 변환 논문.

### English
- **Read carefully**: §3 (recorruption transform and Proposition 1 proof), §3.3 (Poisson/Anscombe extension), §3.4 (*why* inference must use MC averaging), §4 (results, especially SIDD).
- **Skim**: §2 related work, §5 implementation details.
- **Common stumbling blocks**:
  - Why bare $y$ at inference is wrong — distribution mismatch with the trained input form $y_1 = y + \alpha D^T z$.
  - The role of $\alpha$ — controls $\text{Var}(n_1)=(1+\alpha^2)\sigma^2$ and $\text{Var}(n_2)=(1+\alpha^{-2})\sigma^2$, the SNR split.
  - Why both $D D^T = \Sigma$ AND $D^T D = \Sigma$ are needed for Proposition 1's cross-correlation term to vanish.
  - The Anscombe + R2R chain: stabilise variance, recorrupt, train, MC-average, then invert Anscombe.
- Companion reading: original Lehtinen N2N paper, Anscombe (#11) / Mäkitalo-Foi (#14) transform papers.

---

## 7. 현대적 의의 / Modern Significance

### 한국어
R2R은 **"단일 영상 + 알려진 잡음 모델 ⇒ N2N 등가 학습"**이라는 명제를 학계에 정착시킨 분기점이다. 이후 거의 모든 self-supervised denoising 논문 — Blind2Unblind(#22), NBR2NBR, AP-BSN — 이 R2R을 표준 비교 baseline으로 사용한다. 특히 *잡음 모델이 calibration으로 알려진 도메인* — 천체 (read-noise + shot-noise 모델), 형광현미경 (Poisson-Gaussian), 의료 영상 — 에서는 R2R의 noise-model 요구사항이 사실상 무료다. Self2Self(#19)와 비교하면 dropout ensemble을 *입력 측 stochasticity*로 대체해 *결정적 네트워크* 사용이 가능하며, Neighbor2Neighbor(#20)와 비교하면 *noise model 정확성* 대 *spatial smoothness 가정*의 trade-off를 보여준다. 두 논문은 보완적이며, B2U(#22)는 두 가정을 모두 회피한다.

### English
R2R cemented the proposition **"single noisy image + known noise statistics ⇒ Noise2Noise-equivalent training"** in the field. From CVPR 2021 onward almost every self-supervised denoiser — Blind2Unblind (#22), NBR2NBR, AP-BSN — cites R2R as a default baseline. R2R's known-noise assumption is essentially free in *calibration-rich domains*: astronomy (read-noise + shot-noise models), fluorescence microscopy (Poisson-Gaussian), medical imaging. Versus Self2Self (#19) it replaces dropout ensembles with input-side stochasticity, allowing deterministic networks; versus Neighbor2Neighbor (#20) it trades a noise-model assumption for not requiring spatial smoothness — the two papers are complementary, while Blind2Unblind (#22) sidesteps both assumptions.

---

## Q&A

(Populated during reading session / 읽기 세션 중 추가됨)
